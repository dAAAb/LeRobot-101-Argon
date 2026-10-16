"""Continuous wrist-video feature identities for settled-keyframe geometry.

Public API:
    matches, (width, height), metadata = track_wrist_video(observations, history)

Each observation supplies source='wrist', sequence:int, path:str, joints:dict.
History supplies every sequence in the observation interval as
{sequence:int, jpeg:bytes, capture_monotonic:float}. Timestamps, when present
anywhere, are mandatory throughout, strictly increasing, and <=0.4s apart.
Set require_timestamps=True for production; untimed history is for synthetic
tests only. Split recording chunks around pauses/restarts; never join identities.
 If a history entry supplies source, it must be wrist.
An absent endpoint/intermediate frame, duplicate sequence, changed image size,
or invalid image fails closed; no pair bridges unknown temporal motion.

Matches are (observation_index_i, observation_index_j, p1, p2), where p1/p2
are float64 N x 2 ORIGINAL IMAGE PIXELS. Do not normalize these coordinates a
second time before a downstream camera-matrix/undistortPoints pipeline. No pose
is interpolated: pairs refer only to the caller's measured settled keyframes.

Only continuous pyramidal LK establishes identity. There is no descriptor
matching between separate views. New Shi-Tomasi corners may be seeded at a
keyframe, but get new identities and need at least three keyframe sightings.

Dependencies: numpy, opencv-python. No camera, robot, network, or model calls.
Synthetic regressions are in tests/test_video_tracks.py.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


def pyramid_support_radius(window_size=21, max_level=2):
    """Conservative original-pixel radius of a final accepted LK patch.

    At pyramid level L, include the patch half-width, one pixel for bilinear
    interpolation, one for Scharr derivatives, and all 5x5 pyrDown convolutions:
      (window//2 + 1 + 1) * 2**L + 2 * (2**L - 1).
    For a 21px window: L=0/1/2/3 gives 12/26/54/110 original pixels.
    Source: OpenCV 4.12 modules/video/src/lkpyramid.cpp (pyrDown, Scharr,
    interpolated image/derivative patches), and the documented 5x5 pyrDown kernel.
    https://raw.githubusercontent.com/opencv/opencv/4.12.0/modules/video/src/lkpyramid.cpp
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
    """
    if (not isinstance(window_size, int) or window_size < 3 or window_size % 2 != 1
            or not isinstance(max_level, int) or not 0 <= max_level <= 3):
        raise ValueError('Expected an odd window and a pyramid level from zero to three.')
    scale = 1 << max_level
    return (window_size // 2 + 2) * scale + 2 * (scale - 1)


def track_wrist_video(
    observations: list[dict[str, Any]],
    history: list[dict[str, Any]],
    cancel=None,
    *,
    max_features: int = 1200,
    min_keyframes: int = 3,
    min_pair_tracks: int = 12,
    fb_max_px: float = 0.8,
    max_step_px: float = 24.0,
    max_step_reference_width_px: int | None = None,
    max_lk_error: float = 25.0,
    gripper_y: float = 0.595,
    min_distance_px: float = 7.0,
    max_seconds: float = 20.0,
    reseed_keyframes: bool = True,
    pyramid_max_level: int = 2,
    require_timestamps: bool = False,
    max_frame_gap_seconds: float = 0.4,
):
    """Return identity-preserving keyframe correspondences, size, and diagnostics.

    `status == 'ok'` establishes tracking continuity, not metric reconstruction,
    object association, reachability, hand-eye calibration, or grasp readiness.
    Cancellation/deadline/gap failures always return an empty matches list.

    max_step_px is an explicit original-image pixel limit by default. To keep
    its fractional-image meaning across resolutions, opt in with a reference
    width: max_step_reference_width_px=960 makes 24 become 48 at width 1920.
    Only this step cap scales; FB, patch support, and all other gates do not.
    """
    started = time.monotonic()
    size = (0, 0)
    support_radius = 0
    effective_max_step_px = max_step_px if max_step_reference_width_px is None else None
    tracking_boundary = 0
    codec_guard_px = 16
    meta = {
        'method': 'continuous-wrist-LK-forward-backward',
        'source': 'wrist-only',
        'status': 'initializing',
        'reason': '',
        'cancelled': False,
        'coordinate_system': 'original-keyframe-pixels',
        'image_normalization': 'uint8 grayscale; gripper plus 16px codec band replaced by a fixed constant BEFORE Gaussian pyramids; no resizing or coordinate normalization',
        'poses': 'only caller-supplied measured settled keyframe poses; never interpolated',
        'grasp_ready': False,
        'visibility': [],
        'pairs': [],
        'tracks': {'seeded': 0, 'eligible': 0, 'alive_at_end': 0, 'dropped': {}},
        'thresholds': {'fb_max_px': fb_max_px, 'max_step_px': max_step_px,
                       'effective_max_step_px': effective_max_step_px,
                       'max_step_reference_width_px': max_step_reference_width_px,
                       'max_lk_error': max_lk_error, 'gripper_y': gripper_y,
                       'min_keyframes': min_keyframes, 'min_pair_tracks': min_pair_tracks,
                       'pyramid_max_level': pyramid_max_level, 'max_frame_gap_seconds': max_frame_gap_seconds},
    }

    def finish(status, reason, matches=None):
        meta.update(status=status, reason=reason,
                    elapsed_seconds=round(time.monotonic() - started, 4))
        return matches or [], size, meta

    def interrupted():
        if cancel is not None and cancel.is_set():
            meta['cancelled'] = True
            return 'cancelled', 'Tracking cancelled; no correspondences published.'
        if time.monotonic() - started > max_seconds:
            return 'time_budget', 'Tracking exceeded its bounded time budget.'
        return None

    def check_interruption():
        state = interrupted()
        if state:
            raise _Stopped(*state)

    def integer_sequence(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 0

    def in_bounds(points):
        # Source and accepted destination centers both obey full original-image
        # pyramid support, not merely the finest-level half-window.
        margin = float(support_radius)
        return (np.isfinite(points).all(axis=1) &
                (points[:, 0] >= margin) & (points[:, 0] < size[0] - margin) &
                (points[:, 1] >= margin) &
                (points[:, 1] < min(size[1] - margin, tracking_boundary - margin)))

    try:
        check_interruption()
        if not (3 <= min_keyframes <= 32 and min_pair_tracks >= 12 and
                1 <= max_features <= 3000 and 0 < fb_max_px <= .8 and
                0 < gripper_y <= .595 and max_seconds > 0 and
                isinstance(pyramid_max_level, int) and 0 <= pyramid_max_level <= 3 and
                0 < max_frame_gap_seconds <= .4):
            return finish('invalid_input', 'Tracking thresholds violate required conservative limits.')
        if (isinstance(max_step_px, (bool, np.bool_))
                or not isinstance(max_step_px, (int, float, np.integer, np.floating))):
            return finish('invalid_input', 'Expected a finite positive maximum tracking step.')
        try:
            numeric_max_step_px = float(max_step_px)
        except (OverflowError, ValueError):
            return finish('invalid_input', 'Maximum tracking step cannot be represented as a finite float.')
        if not math.isfinite(numeric_max_step_px) or numeric_max_step_px <= 0:
            return finish('invalid_input', 'Expected a finite positive maximum tracking step.')
        if max_step_reference_width_px is not None and (
                isinstance(max_step_reference_width_px, (bool, np.bool_))
                or not isinstance(max_step_reference_width_px, (int, np.integer))
                or max_step_reference_width_px <= 0):
            return finish('invalid_input', 'Expected a positive integer step reference width, or None.')
        support_radius = pyramid_support_radius(21, pyramid_max_level)
        meta['pyramid_support_radius_px'] = support_radius
        if not observations or len(observations) < min_keyframes:
            return finish('insufficient_keyframes', 'At least three settled keyframe observations are required.')
        if len(observations) > 32:
            return finish('input_limit', 'At most 32 settled keyframes are accepted per call.')
        if not history:
            return finish('no_history', 'Continuous intermediate wrist video is required.')
        if len(history) > 500:
            return finish('input_limit', 'At most 500 wrist-history frames are accepted per call.')

        keyframes = {}
        for index, obs in enumerate(observations):
            if obs.get('source') != 'wrist' or not integer_sequence(obs.get('sequence')):
                return finish('invalid_input', 'Only explicitly identified wrist keyframes with valid sequence ids are accepted.')
            if not obs.get('path') or not isinstance(obs.get('joints'), dict) or not obs['joints']:
                return finish('invalid_input', 'Each keyframe needs an image path and measured joints.')
            if obs.get('settled', True) is not True:
                return finish('invalid_input', 'A moving keyframe cannot provide a settled geometry pose.')
            if any(isinstance(v, bool) or not isinstance(v, (int, float, np.number)) or
                   not math.isfinite(float(v)) for v in obs['joints'].values()):
                return finish('invalid_input', 'Keyframe joint measurements must be finite numbers.')
            seq = int(obs['sequence'])
            if seq in keyframes:
                return finish('invalid_input', 'Keyframes must have distinct sequence ids.')
            keyframes[seq] = index
        ordered_keyframes = sorted(keyframes)
        start_seq, end_seq = ordered_keyframes[0], ordered_keyframes[-1]
        if end_seq - start_seq + 1 > 500:
            return finish('history_gap', 'Keyframe interval exceeds the complete bounded video history.')

        video = {}
        capture_times = {}
        byte_count = 0
        for entry in history:
            if entry.get('source', 'wrist') != 'wrist' or not integer_sequence(entry.get('sequence')):
                return finish('invalid_input', 'Observer or invalid history data is excluded.')
            seq = int(entry['sequence'])
            if seq < start_seq or seq > end_seq:
                continue
            if seq in video:
                return finish('invalid_input', 'Duplicate history sequence ids are ambiguous.')
            jpeg = entry.get('jpeg')
            if not isinstance(jpeg, (bytes, bytearray, memoryview)) or not jpeg:
                return finish('invalid_image', 'Every intermediate sequence requires JPEG bytes.')
            byte_count += len(jpeg)
            if byte_count > 80_000_000:
                return finish('input_limit', 'History exceeds the 80 MB encoded-data budget.')
            video[seq] = jpeg
            if 'capture_monotonic' in entry:
                capture_times[seq] = entry['capture_monotonic']
        missing = [seq for seq in range(start_seq, end_seq + 1) if seq not in video]
        if missing:
            meta['missing_sequences'] = missing[:20]
            return finish('history_gap', 'Missing intermediate/end-point video; identities must not bridge a gap.')
        if capture_times or require_timestamps:
            if len(capture_times) != len(video):
                return finish('history_gap', 'Every selected history frame requires capture_monotonic when timestamps are present/required.')
            ordered_times = [capture_times[seq] for seq in range(start_seq, end_seq + 1)]
            if any(isinstance(t, bool) or not isinstance(t, (int, float, np.number)) or
                   not math.isfinite(float(t)) for t in ordered_times):
                return finish('history_gap', 'Capture timestamps must be finite monotonic seconds.')
            gaps = np.diff(np.asarray(ordered_times, dtype=np.float64))
            meta['largest_capture_gap_seconds'] = float(np.max(gaps)) if len(gaps) else 0.0
            if np.any(gaps <= 0) or np.any(gaps > max_frame_gap_seconds + 1e-9):
                return finish('history_gap', 'Non-increasing or stalled/restarted capture timestamps break temporal identity.')
            meta['temporal_integrity'] = 'sequence-and-capture-monotonic'
        else:
            meta['temporal_integrity'] = 'sequence-only; synthetic/legacy input, not proof of hardware continuity'
        meta['history_frames'] = len(video)
        meta['sequence_span'] = [start_seq, end_seq]

        tracks = {}
        active_ids = []
        active_points = np.empty((0, 2), dtype=np.float32)
        dropped = Counter()
        next_id = 0
        previous = None
        previous_seq = None
        lk = dict(winSize=(21, 21), maxLevel=pyramid_max_level,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
                  minEigThreshold=1e-4)

        def seed(image, keyframe_index):
            nonlocal active_ids, active_points, next_id
            capacity = max_features - len(active_ids)
            if capacity <= 0:
                return 0
            mask = np.zeros_like(image)
            bottom = int(tracking_boundary - support_radius)
            mask[support_radius:bottom, support_radius:size[0]-support_radius] = 255
            for p in active_points:
                cv2.circle(mask, tuple(np.rint(p).astype(int)), int(math.ceil(min_distance_px)), 0, -1)
            corners = cv2.goodFeaturesToTrack(image, maxCorners=capacity, qualityLevel=.01,
                                             minDistance=min_distance_px, mask=mask,
                                             blockSize=5, useHarrisDetector=False)
            if corners is None:
                return 0
            corners = cv2.cornerSubPix(image, corners, (3, 3), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .02)).reshape(-1, 2)
            corners = corners[in_bounds(corners)]
            # Subpixel refinement can move detections together. Enforce spatial
            # uniqueness again after refinement so pair counts cannot be inflated
            # by duplicate corners/identities.
            accepted = []
            occupied = active_points.copy()
            for point in corners:
                if len(occupied) and np.any(np.linalg.norm(occupied-point, axis=1) < min_distance_px):
                    continue
                accepted.append(point)
                occupied = np.vstack((occupied, point))
            corners = np.asarray(accepted, dtype=np.float32).reshape(-1, 2)
            for point in corners:
                tracks[next_id] = {'keyframes': {keyframe_index: point.copy()}, 'max_fb_px': 0.0}
                active_ids.append(next_id)
                next_id += 1
            if len(corners):
                active_points = np.vstack((active_points, corners)).astype(np.float32)
            return len(corners)

        for seq in range(start_seq, end_seq + 1):
            check_interruption()
            gray = cv2.imdecode(np.frombuffer(video[seq], np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.ndim != 2:
                return finish('invalid_image', f'Cannot decode wrist sequence {seq}.')
            frame_size = (gray.shape[1], gray.shape[0])
            if previous is None:
                size = frame_size
                effective_max_step_px = numeric_max_step_px
                if max_step_reference_width_px is not None:
                    effective_max_step_px *= size[0] / max_step_reference_width_px
                if not math.isfinite(effective_max_step_px) or effective_max_step_px <= 0:
                    return finish('invalid_input', 'Scaled maximum tracking step must be finite and positive.')
                meta['thresholds']['effective_max_step_px'] = effective_max_step_px
                tracking_boundary = max(0, int(math.floor(gripper_y * size[1])) - codec_guard_px)
                meta['tracking_boundary_px'] = tracking_boundary
                meta['codec_guard_px'] = codec_guard_px
                if min(size) < 64 or max(size) > 1920:
                    return finish('input_limit', 'Expected unresized wrist frames between 64 and 1920 pixels per side.')
            elif frame_size != size:
                return finish('image_geometry_changed', 'Image dimensions changed within a continuous track.')
            kf_index = keyframes.get(seq)
            if kf_index is not None:
                key_image = cv2.imread(str(observations[kf_index]['path']), cv2.IMREAD_GRAYSCALE)
                if key_image is None or key_image.shape != gray.shape:
                    return finish('invalid_keyframe', 'Stored settled keyframe is unavailable or has different image geometry.')
                # Different JPEG quality is fine; an unrelated image/sequence is not.
                difference = float(np.mean(cv2.absdiff(key_image, gray)))
                if difference > 12:
                    meta['keyframe_difference_mean_px'] = difference
                    return finish('keyframe_mismatch', 'Keyframe image does not agree with its video-history sequence.')

            if tracking_boundary <= 2 * support_radius or size[0] <= 2 * support_radius:
                return finish('insufficient_world_region', 'Image has no safe world region after full pyramid-support exclusion.')
            # Mask BEFORE LK builds any pyramid or derivatives. This also prevents
            # a transient iterative estimate outside the accepted guard from
            # sampling real attached-gripper pixels. The source is constant, not
            # blurred/copied gripper texture. The extra codec band excludes JPEG
            # block/chroma boundary leakage. Seed detection/refinement uses this
            # SAME masked image and the SAME full-support center guard.
            gray = gray.copy()
            gray[tracking_boundary:] = 127
            before = len(active_ids)
            frame_drops = Counter()
            if previous is not None and active_ids:
                p0 = np.ascontiguousarray(active_points.reshape(-1, 1, 2))
                p1, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(previous, gray, p0, None, **lk)
                check_interruption()
                if p1 is None or forward_status is None or forward_error is None:
                    valid = np.zeros(before, dtype=bool)
                    fb = np.full(before, np.inf)
                    proposed = active_points.copy()
                    frame_drops['lk_status'] = before
                else:
                    proposed = p1.reshape(-1, 2)
                    forward_good = forward_status.reshape(-1).astype(bool) & np.isfinite(proposed).all(axis=1)
                    # Never feed NaN coordinates back into OpenCV.
                    safe_p1 = p1.copy()
                    safe_p1[~forward_good] = p0[~forward_good]
                    back, backward_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, safe_p1, None, **lk)
                    if back is None or backward_status is None:
                        fb = np.full(before, np.inf)
                        status_good = np.zeros(before, dtype=bool)
                    else:
                        fb = np.linalg.norm(back.reshape(-1, 2) - active_points, axis=1)
                        status_good = forward_good & backward_status.reshape(-1).astype(bool) & np.isfinite(fb)
                    tests = [
                        ('lk_status', status_good),
                        ('forward_backward', fb < fb_max_px),
                        ('image_error', np.isfinite(forward_error.reshape(-1)) & (forward_error.reshape(-1) <= max_lk_error)),
                        ('step_limit', np.linalg.norm(proposed - active_points, axis=1) <= effective_max_step_px),
                        ('bounds_or_gripper', in_bounds(proposed)),
                    ]
                    valid = np.ones(before, dtype=bool)
                    for label, passed in tests:
                        frame_drops[label] += int(np.count_nonzero(valid & ~passed))
                        valid &= passed
                active_ids = [identity for identity, good in zip(active_ids, valid) if good]
                active_points = proposed[valid].astype(np.float32)
                for identity, error in zip(active_ids, fb[valid]):
                    tracks[identity]['max_fb_px'] = max(tracks[identity]['max_fb_px'], float(error))
                dropped.update(frame_drops)

            if kf_index is not None:
                for identity, point in zip(active_ids, active_points):
                    tracks[identity]['keyframes'][kf_index] = point.copy()
            added = seed(gray, kf_index) if kf_index is not None and (previous is None or reseed_keyframes) else 0
            meta['visibility'].append({
                'sequence': seq, 'keyframe_index': kf_index,
                'active_before': before, 'survived': len(active_ids) - added,
                'seeded': added, 'active_after': len(active_ids),
                'dropped': dict(frame_drops),
            })
            # Preserve meaningful counts even when the next frame is cancelled,
            # corrupt, or exceeds the deadline.
            meta['tracks'].update(seeded=next_id, alive_at_end=len(active_ids), dropped=dict(dropped))
            if kf_index is not None:
                meta['tracks']['eligible'] = sum(len(t['keyframes']) >= min_keyframes for t in tracks.values())
            previous, previous_seq = gray, seq

        check_interruption()
        eligible = {identity: track for identity, track in tracks.items()
                    if len(track['keyframes']) >= min_keyframes}
        meta['tracks'] = {'seeded': next_id, 'eligible': len(eligible),
                          'alive_at_end': len(active_ids), 'dropped': dict(dropped)}
        pair_tracks = {}
        for identity, track in eligible.items():
            for i, j in combinations(sorted(track['keyframes']), 2):
                pair_tracks.setdefault((i, j), []).append(identity)
        matches = []
        for (i, j), identities in sorted(pair_tracks.items()):
            check_interruption()
            if len(identities) < min_pair_tracks:
                continue
            # Identity appears once in a pair. Re-seeded corners can never inherit
            # a disappeared track's earlier keyframe positions.
            p1 = np.array([eligible[t]['keyframes'][i] for t in identities], dtype=np.float64)
            p2 = np.array([eligible[t]['keyframes'][j] for t in identities], dtype=np.float64)
            matches.append((i, j, p1, p2))
            meta['pairs'].append({'i': i, 'j': j, 'unique_tracks': len(identities),
                                  'median_displacement_px': float(np.median(np.linalg.norm(p2-p1, axis=1))),
                                  'worst_track_fb_px': max(eligible[t]['max_fb_px'] for t in identities)})
        if not matches:
            return finish('insufficient_tracks', 'Fewer than 12 continuous identities span any pair while each appearing in at least three keyframes.')
        return finish('ok', 'Keyframe correspondences preserve continuous video identities.', matches)
    except _Stopped as exc:
        return finish(exc.status, exc.reason)
    except (cv2.error, OSError, ValueError, TypeError, KeyError) as exc:
        return finish('invalid_input', str(exc))


class _Stopped(Exception):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason
        super().__init__(reason)

