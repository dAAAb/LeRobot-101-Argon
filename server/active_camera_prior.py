"""Explicit, bounded refinement of an uncertain camera prior from wrist motion.

This is NOT a persisted or validated calibration. Strict mode excludes every
target feature from fitting. Coarse mode partitions target identities before
fitting and requires two whole-pose holdouts with disjoint validation IDs.
Both modes verify independent 3-view geometry and compare split sensitivity.
No D435, known object dimensions, previous fit, hardware or network is used.
"""
from __future__ import annotations

from itertools import combinations
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .selfcal import fit_focal_and_handeye, inv, sampson_errors


class _Deadline:
    def __init__(self, cancel, until):
        self.cancel, self.until = cancel, until

    def is_set(self):
        return (self.cancel is not None and self.cancel.is_set()) or time.monotonic() >= self.until


def _camera_uncertainty(poses, first, second, target_disagreement, residual_angle_deg=0.):
    """Separate relative-motion uncertainty from absolute mount-origin error.

Triangulation uses camera displacement, so a common mount translation cancels
under equal orientation. It still matters for positions expressed in the base
frame and is therefore reported separately, never discarded.
    """
    cameras_a = [pose @ first for pose in poses]
    cameras_b = [pose @ second for pose in poses]
    translations, rotations = [], []
    for i, j in combinations(range(len(poses)), 2):
        a = inv(cameras_a[j]) @ cameras_a[i]
        b = inv(cameras_b[j]) @ cameras_b[i]
        translations.append(float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
        rotations.append(float(np.rad2deg(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())))
    absolute_translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    absolute_rotation = float(np.rad2deg(Rotation.from_matrix(first[:3, :3] @ second[:3, :3].T).magnitude()))
    return {
        'relative_translation_uncertainty_m': max(.002, max(translations, default=0.)),
        'relative_rotation_uncertainty_deg': max(.2, residual_angle_deg, max(rotations, default=0.)),
        'handeye_translation_disagreement_m': absolute_translation,
        'handeye_rotation_disagreement_deg': absolute_rotation,
        'target_split_p95_disagreement_m': float(target_disagreement),
        'base_position_uncertainty_m': max(.002, absolute_translation, float(target_disagreement)),
    }


def refine_active_camera_prior(frames, model, *, roi, roi_frame_id=None, cancel=None, max_seconds=12., mode='strict',
                               _prepared_observations=None):
    """Return ``status='refined'`` plus an unvalidated ``calibration`` prior.

    Four images permit held-out image pairs, not independent unseen poses.
    Five or more permit complete pose holdout. Neither proves physical accuracy.
    Strict mode reserves every target identity for validation. Coarse mode
    requires five pose families and partitions target identities before fitting.
    """
    if mode == 'coarse':
        return _refine_coarse_prior(frames, model, roi=roi, roi_frame_id=roi_frame_id,
                                    cancel=cancel, max_seconds=max_seconds,
                                    prepared=_prepared_observations)
    if mode != 'strict':
        raise ValueError('Unknown camera-prior refinement mode')
    from .active_depth import _roi, _read_frames, _image_tracks, _solve_track
    from .geometry import observation_families
    result = {'status': 'insufficient', 'metric_calibrated': False, 'source': 'wrist-only',
              'calibration': None, 'grasp_ready': False}
    started = time.monotonic()
    deadline = None
    try:
        if type(max_seconds) not in (int, float) or not 0 < max_seconds <= 15:
            raise ValueError('相機參數更新最多使用 15 秒')
        deadline = _Deadline(cancel, started + max_seconds)
        if len(frames) < 4:
            raise ValueError('更新相機參數需要至少四個不同的腕部視角，並保留未參與擬合的影像配對')
        box = _roi(roi)
        if box is None:
            raise ValueError('需要標出物件範圍，才能保留物件特徵作為獨立核對')
        images, transforms = _read_frames(frames, model, deadline)
        families = observation_families(frames)
        if len(families) < 4:
            raise ValueError('需要四組不同的實測姿勢；回到同一姿勢不算新的核對視角')
        reference = len(frames) - 1
        if roi_frame_id is not None:
            indexes = [i for i, f in enumerate(frames) if f.get('id') == roi_frame_id]
            if len(indexes) != 1:
                raise ValueError('物件範圍的原始影像不在這批視角中')
            reference = indexes[0]
        stable_jaw = max(f['joints']['gripper'] for f in frames) - min(f['joints']['gripper'] for f in frames) <= 6
        tracks, metadata = _image_tracks(images, box, deadline, roi_index=reference,
            allow_roi_below_gripper=stable_jaw, restrict_to_roi=False)
        width, height = images[0].shape[::-1]
        x, y, w, h = box
        background, target = [], []
        fixed_background = 0
        for track in tracks:
            if reference not in track:
                # Without that sighting the feature cannot be ruled out as target.
                continue
            u, v = track[reference]
            inside = x * width <= u <= (x + w) * width and y * height <= v <= (y + h) * height
            if not inside and max(np.linalg.norm(a - b) for a, b in combinations(track.values(), 2)) <= 3:
                fixed_background += 1
                continue
            (target if inside else background).append(track)
        target = [t for t in target if len(t) >= 3]
        result['tracking'] = {**metadata, 'training_background_identities': len(background),
                              'heldout_target_identities': len(target), 'target_used_for_fit': False,
                              'camera_fixed_background_identities_rejected': fixed_background}
        if len(target) < 6:
            raise ValueError('同一物件還沒有六個可跨三張影像核對的特徵')
        matches, pair_support = [], []
        for i, j in combinations(range(len(frames)), 2):
            shared = [t for t in background if i in t and j in t]
            pair_support.append({'views': [i, j], 'background_identities': len(shared)})
            if len(shared) >= 12:
                matches.append((i, j, np.array([t[i] for t in shared]), np.array([t[j] for t in shared])))
        result['tracking']['background_pair_support'] = pair_support
        if len(matches) < 5:
            raise ValueError('物件以外的背景特徵配對不足，不能同時擬合與獨立核對相機參數')
        g = [t['gripper_link'] for t in transforms]
        x0 = inv(g[0]) @ transforms[0]['wrist_camera_optical']
        # Baseline ranking only picks validation splits; no CAD points are used
        # as visual ground truth or as object dimensions.
        order = sorted(range(len(matches)), key=lambda n: np.linalg.norm(
            transforms[matches[n][0]]['wrist_camera_optical'][:3, 3] -
            transforms[matches[n][1]]['wrist_camera_optical'][:3, 3]), reverse=True)
        splits = []
        if len(families) >= 5:
            for withheld in (families[-1], families[0]):
                train = [m for m in matches if not set(withheld).intersection(m[:2])]
                test = [m for m in matches if set(withheld).intersection(m[:2])]
                if len(train) >= 4 and test:
                    splits.append(('whole_pose', withheld, train, test))
        if len(splits) < 2:
            splits = [('whole_pair', matches[n][:2], [m for k, m in enumerate(matches) if k != n], [matches[n]])
                      for n in order[:2]]
        fits = []
        for split, withheld, train, testing in splits:
            if deadline.is_set():
                raise RuntimeError('相機參數更新已停止或超過本次計算時間')
            xfit, kfit, fit = fit_focal_and_handeye(g, train, (width, height), x0,
                width * .75, max_nfev=90, cancel=deadline)
            error = np.abs(sampson_errors(g, xfit, kfit, testing))
            p95 = float(np.percentile(error, 95))
            if p95 > 3:
                raise ValueError('未參與擬合的影像仍無法重現相機移動，先不採用新參數')
            if np.linalg.norm(xfit[:3, 3] - x0[:3, 3]) > .10:
                raise ValueError('估計的鏡頭位置偏離支架先驗超過十公分，先不採用')
            cameras = [pose @ xfit for pose in g]
            target_points = {}
            for identity, track in enumerate(target):
                if deadline.is_set():
                    raise RuntimeError('相機參數更新已停止或超過本次計算時間')
                point, rejected = _solve_track(track, cameras, kfit, np.zeros(5), (width, height))
                if rejected is None:
                    target_points[identity] = point['point']
            if len(target_points) < 6:
                raise ValueError('背景可重現，但未參與擬合的物件特徵還沒有一致的三維位置')
            fits.append({'x': xfit, 'k': kfit, 'target_points': target_points,
                         'metrics': {'holdout_kind': split, 'withheld': withheld,
                                     'holdout_p95_sampson_px': p95,
                                     'target_three_view_points': len(target_points),
                                     'fit': fit}})
        shared = set(fits[0]['target_points']) & set(fits[1]['target_points'])
        if len(shared) < 6:
            raise ValueError('不同擬合結果沒有足夠共同物件特徵可核對')
        disagreements = [np.linalg.norm(fits[0]['target_points'][i] - fits[1]['target_points'][i]) for i in shared]
        p95_difference = float(np.percentile(disagreements, 95))
        if p95_difference > .04:
            raise ValueError('不同影像配對推算的物件位置相差超過四公分，還不能採用這組相機參數')
        a, b = fits
        chosen = min(fits, key=lambda f: f['metrics']['holdout_p95_sampson_px'])
        # Split disagreement is one sensitivity component; retain nonzero floors
        # for unmodelled optics, encoder/FK errors and static-scene assumptions.
        focal_error = max(.05, abs(a['k'][0, 0] - b['k'][0, 0]) / min(a['k'][0, 0], b['k'][0, 0]))
        residual_angle = max(float(np.rad2deg(np.arctan(max(f['metrics']['holdout_p95_sampson_px'],
            f['metrics']['fit']['p95_sampson_px']) / f['k'][0, 0]))) for f in fits)
        uncertainty = _camera_uncertainty(g, a['x'], b['x'], p95_difference, residual_angle)
        calibration = {'camera_matrix': chosen['k'].tolist(), 'handeye': chosen['x'].tolist(),
            'image_size': [width, height], 'camera_generation': frames[-1]['camera_generation'],
            'metric_calibrated': False, 'intrinsics_validated': False, 'handeye_validated': False,
            'robot_geometry_validated': False, 'needs_more_views': True,
            'artifact': None, 'prior_refinement': 'same-attempt-background-fit-target-heldout',
            'focal_relative_uncertainty': float(focal_error), **uncertainty}
        if deadline.is_set():
            raise RuntimeError('相機參數更新已停止或超過本次計算時間')
        result.update(status='refined', calibration=calibration,
            validation={'splits': [f['metrics'] for f in fits],
                        'target_used_for_fit': False, 'shared_target_identities': len(shared),
                        'target_split_p95_disagreement_m': p95_difference,
                        'unseen_pose_validation': all(f['metrics']['holdout_kind'] == 'whole_pose' for f in fits)},
            reason='已用背景移動更新相機先驗，並用未參與擬合的物件特徵核對；仍不是實測精度保證')
    except (ValueError, RuntimeError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        result.update(reason=str(exc), calibration=None)
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
    if deadline is not None and deadline.is_set():
        result.update(status='insufficient', calibration=None,
                      reason='相機參數更新已停止或超過本次計算時間')
    return result


def _refine_coarse_prior(frames, model, *, roi, roi_frame_id, cancel, max_seconds, prepared=None):
    """Same angular acceptance across resolutions; never a validated calibration.

The target can supply training texture, but its fixed, disjoint held-out
identities never train the image correspondence models or the camera model.
Two whole pose families remain absent from their respective camera fits.
"""
    from .active_depth import _roi, _read_frames, _solve_track
    from .active_prior_tracks import multi_anchor_tracks
    from .geometry import observation_families
    result = {'status': 'insufficient', 'metric_calibrated': False, 'source': 'wrist-only',
              'calibration': None, 'grasp_ready': False, 'prior_quality': 'coarse'}
    started = time.monotonic()
    deadline = None
    try:
        if type(max_seconds) not in (int, float) or not 0 < max_seconds <= 15:
            raise ValueError('相機參數更新最多使用 15 秒')
        deadline = _Deadline(cancel, started + max_seconds)
        box = _roi(roi)
        if box is None:
            raise ValueError('需要標出物件範圍，才能保留獨立特徵核對')
        if len(frames) < 5:
            raise ValueError('粗估相機位置需要至少五組不同的腕部視角')
        features = None
        if prepared is not None:
            if (not isinstance(prepared, tuple) or len(prepared) != 5 or prepared[0] is not frames
                    or prepared[1] is not model or any(not isinstance(values, (list, tuple))
                        or len(values) != len(frames) for values in prepared[2:])):
                raise ValueError('預備影像特徵不屬於這一批腕部觀察')
            if deadline.is_set():
                raise RuntimeError('相機參數更新已停止或超過本次計算時間')
            images, transforms, features = prepared[2:]
        else:
            images, transforms = _read_frames(frames, model, deadline)
        families = observation_families(frames)
        if len(families) < 5:
            raise ValueError('需要五組不同實測姿勢；回到同一姿勢不算新視角')
        reference = len(frames) - 1
        if roi_frame_id is not None:
            indexes = [i for i, frame in enumerate(frames) if frame.get('id') == roi_frame_id]
            if len(indexes) != 1:
                raise ValueError('物件範圍的原始影像不在這批視角中')
            reference = indexes[0]
        stable_jaw = max(f['joints']['gripper'] for f in frames) - min(f['joints']['gripper'] for f in frames) <= 6
        training, validation, metadata = multi_anchor_tracks(images, box, reference,
            stable_jaw=stable_jaw, cancel=deadline, max_feature_width=960, features=features)
        result['tracking'] = metadata
        if len(validation) < 6:
            raise ValueError('還沒有六個未參與擬合、且跨三張影像的物件特徵可核對')
        width, height = images[0].shape[::-1]
        scale = width / 960.
        pixel_limit = 3. * scale
        result['residual_policy'] = {'reference_width': 960, 'reference_limit_px': 3.,
                                     'image_width': width, 'limit_px': pixel_limit}
        g = [t['gripper_link'] for t in transforms]
        x0 = inv(g[0]) @ transforms[0]['wrist_camera_optical']
        fits = []
        for withheld in (families[-1], families[0]):
            if deadline.is_set():
                raise RuntimeError('相機參數更新已停止或超過本次計算時間')
            train_pairs, test_pairs = [], []
            for i, j in combinations(range(len(frames)), 2):
                if set(withheld).intersection((i, j)):
                    shared = [t for t in validation.values() if i in t and j in t]
                    if len(shared) >= 6:
                        test_pairs.append((i, j, np.array([t[i] for t in shared]), np.array([t[j] for t in shared])))
                else:
                    shared = [t for t in training if i in t and j in t]
                    if len(shared) >= 12:
                        train_pairs.append((i, j, np.array([t[i] for t in shared]), np.array([t[j] for t in shared])))
            if len(train_pairs) < 4 or not test_pairs:
                raise ValueError('保留完整視角與獨立特徵後，剩下的配對還不足以估算相機位置')
            xfit, kfit, fit = fit_focal_and_handeye(g, train_pairs, (width, height), x0,
                width * .75, max_nfev=140, cancel=deadline, robust_loss='cauchy',
                robust_scale_px=.75 * scale, max_sampson_px=pixel_limit)
            errors = np.abs(sampson_errors(g, xfit, kfit, test_pairs))
            p95 = float(np.percentile(errors, 95))
            result.setdefault('candidate_fits', []).append({
                'withheld': withheld, 'camera_matrix': kfit.tolist(), 'handeye': xfit.tolist(),
                'fit': fit, 'holdout_p95_sampson_px': p95, 'accepted_calibration': False})
            if p95 > pixel_limit:
                raise ValueError('未參與擬合的視角仍對不上；先保留目前姿勢，再補一個不同方向的畫面')
            if np.linalg.norm(xfit[:3, 3] - x0[:3, 3]) > .10:
                raise ValueError('估計鏡頭位置偏離支架先驗超過十公分，先不採用')
            points = {}
            for identity, track in validation.items():
                if deadline.is_set():
                    raise RuntimeError('相機參數更新已停止或超過本次計算時間')
                point, reason = _solve_track(track, [pose @ xfit for pose in g], kfit, np.zeros(5),
                                            (width, height), max_reprojection_px=pixel_limit)
                if reason is None:
                    points[identity] = point['point']
            if len(points) < 6:
                raise ValueError('獨立核對的物件特徵還沒有一致的三維位置')
            fits.append({'x': xfit, 'k': kfit, 'points': points,
                'metrics': {'holdout_kind': 'whole_pose', 'withheld': withheld,
                            'holdout_p95_sampson_px': p95, 'residual_limit_px': pixel_limit,
                            'target_three_view_points': len(points), 'fit': fit}})
        shared = set(fits[0]['points']) & set(fits[1]['points'])
        if len(shared) < 6:
            raise ValueError('不同擬合結果沒有足夠共同物件特徵可核對')
        disagreement = float(np.percentile([np.linalg.norm(fits[0]['points'][i] - fits[1]['points'][i]) for i in shared], 95))
        result['validation'] = {'splits': [f['metrics'] for f in fits], 'target_used_for_fit': True,
            'heldout_feature_identities_used_for_fit': False, 'shared_target_identities': len(shared),
            'target_split_p95_disagreement_m': disagreement, 'unseen_pose_validation': True,
            'heldout_pose_used_for_image_correspondence_filter_only': True}
        if disagreement > .04:
            raise ValueError('兩組獨立估算相差超過四公分，還不能採用')
        first, second = fits
        # Choose the last-pose holdout deterministically, not by test-set score.
        chosen = first
        residual_angle = max(float(np.arctan(max(f['metrics']['holdout_p95_sampson_px'],
            f['metrics']['fit']['p95_sampson_px']) / f['k'][0, 0])) for f in fits)
        focal_error = max(.05, abs(first['k'][0, 0] - second['k'][0, 0]) / min(first['k'][0, 0], second['k'][0, 0]))
        uncertainty = _camera_uncertainty(g, first['x'], second['x'], disagreement, float(np.rad2deg(residual_angle)))
        calibration = {'camera_matrix': chosen['k'].tolist(), 'handeye': chosen['x'].tolist(),
            'image_size': [width, height], 'camera_generation': frames[-1]['camera_generation'],
            'metric_calibrated': False, 'intrinsics_validated': False, 'handeye_validated': False,
            'robot_geometry_validated': False, 'needs_more_views': True, 'artifact': None,
            'prior_refinement': 'same-attempt-disjoint-feature-fit-target-heldout',
            'prior_quality': 'coarse', 'residual_reference_width': 960,
            'residual_limit_px': pixel_limit, 'target_used_for_fit': True,
            'observation_frame_ids': [frame['id'] for frame in frames],
            'roi_frame_id': frames[reference]['id'],
            'focal_relative_uncertainty': float(focal_error), **uncertainty,
            'heldout_angular_residual_deg': float(np.rad2deg(residual_angle))}
        if deadline.is_set():
            raise RuntimeError('相機參數更新已停止或超過本次計算時間')
        result.update(status='refined', calibration=calibration,
            validation={'splits': [f['metrics'] for f in fits], 'target_used_for_fit': True,
                'heldout_feature_identities_used_for_fit': False,
                'heldout_pose_used_for_image_correspondence_filter_only': True,
                'shared_target_identities': len(shared), 'target_split_p95_disagreement_m': disagreement,
                'unseen_pose_validation': True},
            reason='已用移動前後的影像粗估鏡頭與物件位置，並用不同特徵、不同姿勢核對；仍保留誤差範圍')
    except (ValueError, RuntimeError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        result.update(reason=str(exc), calibration=None)
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
    if deadline is not None and deadline.is_set():
        result.update(status='insufficient', calibration=None, reason='相機參數更新已停止或超過本次計算時間')
    return result
