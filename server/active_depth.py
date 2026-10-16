"""Bounded wrist-only sparse depth from measured camera motion.

``estimate_active_depth(frames, model, roi=..., calibration=...)`` consumes
settled image records, never hardware. ``roi`` is an optional normalized bbox
in the LAST image. Distances describe tracked visible surfaces, not an object
center, jaw gap, contact, collision clearance, or successful grasp.

Default camera poses/focal length are explicitly uncertain CAD priors. A prior
can still give useful coarse depth; it is never promoted to calibrated metric
evidence by a small reprojection residual. Two-view rigid object motion is
unobservable in general; even three-view consistency is not proof of stasis.
"""
from __future__ import annotations

from itertools import combinations
from copy import deepcopy
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .local_cartesian import NAMES
from .selfcal import intrinsics, inv

GRIPPER_IMAGE_BOUNDARY = .595
MIN_BASELINE_M = .001
MIN_PARALLAX_DEG = .15
MAX_REPROJECTION_PX = 3.
COARSE_RESIDUAL_REFERENCE_WIDTH = 960.
MIN_POINTS = 6


def load_camera_calibration(path):
    """Read an existing local artifact without assuming that a fitted model is valid.

    Legacy boardless fits have no acquisition-resolution declaration. Inferring
    it from their fixed principal point is allowed only as an explicit prior.
    Nothing in this loader writes or automatically selects calibration files.
    """
    data = json.loads(Path(path).read_text())
    k = np.asarray(data['camera_matrix'], float)
    size = data.get('image_size')
    inferred = size is None
    if inferred:
        size = [int(round(2 * k[0, 2] + 1)), int(round(2 * k[1, 2] + 1))]
    return {**data, 'image_size': size, 'artifact': str(Path(path)),
            'resolution_inferred_from_principal_point': inferred,
            'metric_calibrated': False}


def _cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise RuntimeError('距離估算已停止')


def _rigid(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all() or
            not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7) or
            not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or
            abs(np.linalg.det(t[:3, :3]) - 1) > 1e-5):
        raise ValueError('相機姿態需要有效的三維旋轉與平移')
    return t


def _roi(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'x', 'y', 'w', 'h'}:
        raise ValueError('物件範圍需要 x、y、w、h')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value.values()):
        raise ValueError('物件範圍需要有限數值')
    x, y, w, h = [float(value[k]) for k in ('x', 'y', 'w', 'h')]
    if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x + w <= 1 and y + h <= 1):
        raise ValueError('物件範圍必須落在目前影像內')
    return x, y, w, h


def _read_frames(frames, model, cancel):
    if not 2 <= len(frames) <= 8:
        raise ValueError('距離估算需要 2–8 張不同時刻的腕部影像')
    images, transforms = [], []
    generation = None
    previous = None
    for frame in frames:
        _cancel(cancel)
        if not isinstance(frame, dict) or frame.get('source') != 'wrist':
            raise ValueError('距離估算只使用腕部相機，不使用 D435')
        if not frame.get('camera_generation') or (generation and frame['camera_generation'] != generation):
            raise ValueError('不能混合不同相機連線的影像')
        generation = frame['camera_generation']
        stamp = frame.get('capture_monotonic')
        sequence = frame.get('sequence')
        if (type(sequence) is not int or sequence <= 0 or type(stamp) not in (int, float) or
                not math.isfinite(stamp) or (previous and (sequence <= previous[0] or stamp <= previous[1]))):
            raise ValueError('腕部影像必須有依序增加的序號與取像時間')
        previous = sequence, stamp
        raw = frame.get('joints')
        if (not isinstance(raw, dict) or set(raw) != set((*NAMES, 'gripper')) or
                any(type(v) is not int or not 0 <= v <= 4095 for v in raw.values())):
            raise ValueError('距離估算需要每張照片實際量到的六軸位置')
        if not isinstance(frame.get('path'), str) or not frame['path'].strip():
            raise ValueError('腕部影像檔案不存在')
        image = cv2.imread(frame['path'], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('腕部影像檔案不存在')
        if images and images[0].shape != image.shape:
            raise ValueError('取像尺寸已改變，請重新觀察')
        images.append(image)
        transforms.append(model.fk.transforms(raw))
    return images, transforms


def _camera_model(size, transforms, generation, calibration):
    c = calibration or {}
    if c and tuple(c.get('image_size', ())) != tuple(size):
        raise ValueError('相機參數的影像尺寸不符，不能把裁切或縮放當成物件移動')
    k = np.asarray(c.get('camera_matrix', intrinsics(size[0] * .75, size)), float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or
            not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-6):
        raise ValueError('相機內參無效')
    distortion = np.asarray(c.get('distortion', np.zeros(5)), float).reshape(-1)
    if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError('相機畸變參數無效')
    x = _rigid(c['handeye']) if c.get('handeye') is not None else None
    cameras = [_rigid(t['gripper_link'] @ x if x is not None else t['wrist_camera_optical']) for t in transforms]
    validated = (all(c.get(n) is True for n in ('intrinsics_validated', 'handeye_validated', 'robot_geometry_validated'))
                 and isinstance(c.get('validation_id'), str) and bool(c['validation_id'])
                 and c.get('camera_generation') == generation
                 and c.get('needs_more_views') is not True
                 and not c.get('resolution_inferred_from_principal_point', False))
    # These are declared engineering bounds, not fitted confidence intervals.
    uncertainty = {
        'focal_relative': float(c.get('focal_relative_uncertainty', .01 if validated else .20)),
        'relative_translation_m': float(c.get('relative_translation_uncertainty_m', .001 if validated else .005)),
        'relative_rotation_deg': float(c.get('relative_rotation_uncertainty_deg', .1 if validated else 1.)),
    }
    if any(not math.isfinite(v) or v < 0 for v in uncertainty.values()):
        raise ValueError('相機誤差範圍無效')
    refinement = c.get('prior_refinement')
    refined = refinement in ('same-attempt-background-fit-target-heldout',
                             'same-attempt-disjoint-feature-fit-target-heldout')
    coarse = c.get('prior_quality') == 'coarse'
    if coarse and (not refined or c.get('camera_generation') != generation or validated):
        raise ValueError('粗略相機先驗只能用於同一次腕部觀察，不能標成已校準')
    if not validated:
        floors = (.05, .002, .2) if refined else (.15, .003, .5)
        uncertainty = {n: max(value, floor) for (n, value), floor in zip(uncertainty.items(), floors)}
    # Relative motion determines triangulation, while absolute mount error
    # still matters when expressing a point in the robot/gripper frame.
    for name in ('handeye_translation_disagreement_m', 'handeye_rotation_disagreement_deg',
                 'base_position_uncertainty_m', 'target_split_p95_disagreement_m'):
        value = float(c.get(name, 0.))
        if not math.isfinite(value) or value < 0:
            raise ValueError('相機安裝位置誤差無效')
        uncertainty[name] = value
    provenance = {
        'metric_calibrated': bool(validated),
        'camera_model': 'validated-calibration' if validated else 'same-attempt-refined-prior' if refined
                        else 'unvalidated-handeye-prior' if x is not None else 'cad-camera-prior',
        'prior_quality': 'coarse' if coarse else 'strict' if refined else None,
        'reprojection_limit_px': (MAX_REPROJECTION_PX * size[0] / COARSE_RESIDUAL_REFERENCE_WIDTH
                                  if coarse else MAX_REPROJECTION_PX),
        'reprojection_reference_width_px': COARSE_RESIDUAL_REFERENCE_WIDTH if coarse else None,
        'reprojection_model': ('calibrated-relative-pose-angular-aperture-v1' if validated
                               else 'fixed-pixel-residual'),
        'intrinsics': ('same-attempt disjoint feature fit' if coarse else 'same-attempt background feature fit') if refined else 'artifact' if c
                      else 'focal=0.75*image_width; centered principal point; square pixels',
        'distortion': 'artifact' if 'distortion' in c else 'zero-distortion assumption',
        'calibration_artifact': c.get('artifact'), 'validation_id': c.get('validation_id') if validated else None,
        'calibration_revision': c.get('calibration_revision'),
        'calibration_fingerprint': c.get('calibration_fingerprint'),
        'runtime_calibration_scopes': c.get('runtime_calibration_scopes', []),
        'geometry_revision': c.get('geometry_revision'),
        'artifact_needs_more_views': c.get('needs_more_views'),
        'artifact_stability': c.get('stability'),
        'camera_generation': generation, 'image_size': list(size),
        'source': 'wrist-only', 'object_dimensions_used': False,
        'gripper_reference': 'CAD fixed-pad frame +Z; not the jaw-gap center',
    }
    return cameras, k, distortion, uncertainty, provenance


def _mutual_matches(a, b):
    if a is None or b is None or min(len(a), len(b)) < 2:
        return {}
    matcher = cv2.BFMatcher()
    def ratio(x, y):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(x, y, k=2)
                if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance}
    forward, reverse = ratio(a, b), ratio(b, a)
    return {i: j for i, j in forward.items() if reverse.get(j) == i}


def _image_tracks(images, roi, cancel, *, roi_index=None, allow_roi_below_gripper=False,
                  restrict_to_roi=True, _feature_cache=None):
    """Bounded detection; original-pixel identities, support checks and cycles."""
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=.012)
    features = []; detection_sizes = []
    expanded = roi is not None and allow_roi_below_gripper and roi[1] + roi[3] > GRIPPER_IMAGE_BOUNDARY
    for image in images:
        _cancel(cancel)
        height, width = image.shape
        boundary = int(height * GRIPPER_IMAGE_BOUNDARY) - 16
        # Remove gripper pixels before building SIFT pyramids, not just centers.
        masked = image.copy()
        mask = np.full(image.shape, 255, np.uint8)
        if not expanded:
            masked[boundary:] = 127
            mask[boundary:] = 0
        # SIFT pyramids need not process a full 1080p image. Mask BEFORE area
        # resampling, then express every retained keypoint/support radius back
        # in original pixels so ROI, intrinsics, and reprojection units stay
        # unchanged. Rounded resize dimensions can make sx differ from sy.
        detection_width = min(width, 960)
        detection_height = max(1, round(height * detection_width / width))
        sx, sy = detection_width / width, detection_height / height
        if (detection_width, detection_height) != (width, height):
            masked = cv2.resize(masked, (detection_width, detection_height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (detection_width, detection_height), interpolation=cv2.INTER_NEAREST)
        detection_sizes.append([detection_width, detection_height])
        keys, desc = sift.detectAndCompute(masked, mask)
        if sx != 1 or sy != 1:
            keys = [cv2.KeyPoint((key.pt[0] + .5) / sx - .5,
                                (key.pt[1] + .5) / sy - .5,
                                key.size / min(sx, sy), key.angle, key.response,
                                key.octave, key.class_id) for key in keys]
        keep, seen = [], set()
        for i in sorted(range(len(keys)), key=lambda index: keys[index].response, reverse=True):
            key = keys[i]
            pixel = tuple(round(v) for v in key.pt)
            if (pixel not in seen and 6 * key.size <= key.pt[0] < width - 6 * key.size
                    and 6 * key.size <= key.pt[1] < (height if expanded else boundary) - 6 * key.size):
                keep.append(i); seen.add(pixel)
        features.append(([keys[i] for i in keep], desc[keep] if desc is not None else None))
    if _feature_cache is not None:
        _feature_cache['features'] = features
    maps = {}
    for i, j in combinations(range(len(images)), 2):
        _cancel(cancel)
        maps[i, j] = _mutual_matches(features[i][1], features[j][1])
        maps[j, i] = {b: a for a, b in maps[i, j].items()}
    last = len(images) - 1
    roi_index = last if roi_index is None else roi_index
    tracks = []; fixed_rejected = 0
    for anchor, key in enumerate(features[last][0]):
        ids = {i: maps[last, i][anchor] for i in range(last) if anchor in maps.get((last, i), {})}
        ids[last] = anchor
        if len(ids) < 2:
            continue
        inside = False
        if roi is not None and roi_index in ids:
            pixel = features[roi_index][0][ids[roi_index]].pt
            x, y, w, h = roi; width, height = images[roi_index].shape[::-1]
            inside = x * width <= pixel[0] <= (x + w) * width and y * height <= pixel[1] <= (y + h) * height
        if roi is not None and restrict_to_roi and not inside:
            continue
        if not (expanded and inside):
            # Outside the target identity set, all sightings still need the
            # full upper-image descriptor support, even when detection expanded.
            if any(features[i][0][identity].pt[1] + 6 * features[i][0][identity].size >=
                   int(images[i].shape[0] * GRIPPER_IMAGE_BOUNDARY) - 16 for i, identity in ids.items()):
                continue
        if any(maps.get((i, j), {}).get(ids[i]) != ids[j] for i, j in combinations(ids, 2)):
            continue
        track = {i: np.array(features[i][0][identity].pt) for i, identity in ids.items()}
        if expanded and inside and max(np.linalg.norm(a - b) for a, b in combinations(track.values(), 2)) <= 3:
            fixed_rejected += 1
            continue
        tracks.append(track)
    return tracks, {'method': 'mutual-ratio-SIFT-with-cross-view-identity-cycles',
                    'feature_detection_width': max(size[0] for size in detection_sizes),
                    'feature_detection_sizes': detection_sizes,
                    'feature_coordinates': 'original_image_pixels_with_pixel_center_resize_mapping',
                    'features_per_view': [len(f[0]) for f in features], 'candidate_tracks': len(tracks),
                    'roi_view_index': roi_index if roi is not None else None,
                    'lower_target_roi_enabled': expanded,
                    'camera_fixed_target_identities_rejected': fixed_rejected,
                    'self_gripper_exclusion': (f'full descriptor support above {GRIPPER_IMAGE_BOUNDARY} image height minus codec band'
                        + ('; anchored target exception with unchanged jaw and >3px image motion' if expanded else ''))}


def _project(point, camera, k, distortion):
    extrinsic = inv(camera)
    return cv2.projectPoints(np.asarray(point, float).reshape(1, 3),
        cv2.Rodrigues(extrinsic[:3, :3])[0], extrinsic[:3, 3], k, distortion)[0].reshape(2)


def _angle(a, b):
    return float(math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def _calibrated_track_sensitivity(point, track, cameras, k, distortion, *,
                                  pixel_noise, pose_uncertainty, pair, parallax, baseline):
    """Project declared relative pose disagreement at this point's geometry.

    A 3px feature residual and camera/FK disagreement are different quantities.
    Work in undistorted ray angles so distortion/off-axis pixels do not silently
    change that distinction. These remain engineering sensitivities, not a
    probabilistic confidence interval or proof that the object stayed still.
    """
    translation = pose_uncertainty['relative_translation_m']
    rotation = math.radians(pose_uncertainty['relative_rotation_deg'])
    focal = pose_uncertainty['focal_relative']
    if (not all(math.isfinite(v) and v >= 0 for v in (translation, rotation, focal))
            or focal >= 1 or baseline <= translation):
        return None, 'pose_uncertainty_exceeds_parallax'
    noise_angles, observed_rays, allowances, projected_limits = {}, {}, [], []
    residual_angles = []
    for i, pixel in track.items():
        local = (inv(cameras[i]) @ np.r_[point, 1.])[:3]
        distance = float(np.linalg.norm(local))
        if distance <= translation:
            return None, 'pose_uncertainty_exceeds_parallax'
        predicted_ray = local / distance
        samples = np.array([pixel, *[np.asarray(pixel) + pixel_noise * np.array([math.cos(a), math.sin(a)])
                                    for a in np.linspace(0., 2 * math.pi, 16, endpoint=False)]])
        rectified = cv2.undistortPoints(samples.reshape(-1, 1, 2), k, distortion).reshape(-1, 2)
        observed = np.c_[rectified, np.ones(len(rectified))]
        observed /= np.linalg.norm(observed, axis=1)[:, None]
        noise = max(_angle(observed[0], ray) for ray in observed[1:])
        aperture = rotation + math.asin(translation / distance) + noise
        # Do not manufacture a finite image error if the uncertainty cone can
        # cross the focal plane. The full aperture also limits depth below.
        if not math.isfinite(aperture) or math.acos(np.clip(predicted_ray[2], -1, 1)) + aperture >= math.pi / 2:
            return None, 'unbounded_pose_projection'
        residual_angles.append(_angle(predicted_ray, observed[0]))
        allowances.append(aperture)
        noise_angles[i] = noise
        observed_rays[i] = cameras[i][:3, :3] @ observed[0]
        # Pixel-space diagnostic only; acceptance is the angle above. Sampling
        # this circle is not advertised as a mathematically certified bound.
        tangent = np.cross(predicted_ray, [0., 1., 0.])
        tangent /= np.linalg.norm(tangent)
        other = np.cross(predicted_ray, tangent)
        cone = np.array([math.cos(aperture) * predicted_ray + math.sin(aperture) *
                         (math.cos(a) * tangent + math.sin(a) * other)
                         for a in np.linspace(0., 2 * math.pi, 16, endpoint=False)])
        pixels = cv2.projectPoints(cone, np.zeros(3), np.zeros(3), k, distortion)[0].reshape(-1, 2)
        center = cv2.projectPoints(local.reshape(1, 3), np.zeros(3), np.zeros(3), k, distortion)[0].reshape(2)
        if not np.isfinite(pixels).all():
            return None, 'unbounded_pose_projection'
        projected_limits.append(float(np.max(np.linalg.norm(pixels - center, axis=1))))
    if any(error > allowed for error, allowed in zip(residual_angles, allowances)):
        return None, 'nonrigid_or_pose_inconsistent'
    first, second = pair
    # Profile bounds describe RELATIVE pose error, so count that budget once
    # for the pair; feature localization noise belongs to both images.
    angular_error = rotation + noise_angles[first] + noise_angles[second]
    lower, upper = parallax - angular_error, parallax + angular_error
    if lower <= 0 or upper >= math.pi / 2:
        return None, 'pose_uncertainty_exceeds_parallax'
    # For two rays, range = |baseline cross other_ray| / sin(parallax).
    # Translation changes that numerator, NOT ray parallax. Counting t/r
    # again in the denominator would charge the same translation twice.
    # Use the smaller endpoint lever arm to cover both endpoint ranges.
    displacement = cameras[second][:3, 3] - cameras[first][:3, 3]
    lever = min(float(np.linalg.norm(np.cross(displacement, observed_rays[i])))
                for i in (first, second))
    lever_error = translation + baseline * 2 * math.sin((rotation + max(noise_angles[first], noise_angles[second])) / 2)
    if lever <= lever_error:
        return None, 'pose_uncertainty_exceeds_parallax'
    high_factor = ((1 + focal) * (lever + lever_error) / lever
                   * math.sin(parallax) / math.sin(lower))
    low_factor = ((1 - focal) * (lever - lever_error) / lever
                  * math.sin(parallax) / math.sin(upper))
    sensitivity = max(high_factor - 1, 1 - low_factor)
    if not math.isfinite(sensitivity):
        return None, 'unbounded_pose_projection'
    return {'relative_depth_error': sensitivity,
            'minimum_disparity_deg': math.degrees(lower),
            'relative_pose_and_tracking_angle_deg': math.degrees(angular_error),
            'triangulation_lever_arm_m': lever,
            'triangulation_lever_error_m': lever_error,
            'tracking_noise_angle_deg': math.degrees(max(noise_angles.values())),
            'max_reprojection_aperture_deg': math.degrees(max(allowances)),
            'max_projected_pose_and_tracking_px': max(projected_limits)}, None


def _solve_track(track, cameras, k, distortion, size, *, max_reprojection_px=MAX_REPROJECTION_PX,
                 pose_uncertainty=None):
    if (type(max_reprojection_px) not in (int, float) or not math.isfinite(max_reprojection_px)
            or not 0 < max_reprojection_px <= 12):
        raise ValueError('重投影誤差範圍無效')
    indexes = list(track)
    normalized = {i: cv2.undistortPoints(np.asarray(track[i], float).reshape(1, 1, 2), k, distortion).reshape(2) for i in indexes}
    rays = {i: cameras[i][:3, :3] @ np.r_[normalized[i], 1.] for i in indexes}
    rays = {i: value / np.linalg.norm(value) for i, value in rays.items()}
    pairs = []
    for i, j in combinations(indexes, 2):
        baseline = float(np.linalg.norm(cameras[i][:3, 3] - cameras[j][:3, 3]))
        parallax = float(np.rad2deg(np.arccos(np.clip(rays[i] @ rays[j], -1, 1))))
        pairs.append((parallax, baseline, i, j))
    angle, baseline, first, second = max(pairs)
    if baseline < MIN_BASELINE_M or angle < MIN_PARALLAX_DEG:
        return None, 'insufficient_translation_parallax'
    rows = []
    for i in indexes:
        p = inv(cameras[i])[:3]
        u, v = normalized[i]
        rows.extend((u * p[2] - p[0], v * p[2] - p[1]))
    _, _, vh = np.linalg.svd(rows)
    if abs(vh[-1, 3]) < 1e-10:
        return None, 'degenerate_depth'
    point = vh[-1, :3] / vh[-1, 3]
    depths = [(inv(cameras[i]) @ np.r_[point, 1])[2] for i in indexes]
    if not np.isfinite(point).all() or min(depths) < .02 or max(depths) > 2.:
        return None, 'invalid_depth'
    errors = [float(np.linalg.norm(_project(point, cameras[i], k, distortion) - track[i])) for i in indexes]
    sensitivity = None
    if pose_uncertainty is not None:
        sensitivity, reason = _calibrated_track_sensitivity(point, track, cameras, k, distortion,
            pixel_noise=max_reprojection_px, pose_uncertainty=pose_uncertainty, pair=(first, second),
            parallax=math.radians(angle), baseline=baseline)
        if reason is not None:
            return {'max_reprojection_px': max(errors)}, reason
    elif max(errors) > max_reprojection_px:
        return {'max_reprojection_px': max(errors)}, 'nonrigid_or_pose_inconsistent'
    # Rotation-compensated displacement: rotate the earlier ray into the later
    # camera before comparing pixels. Raw image motion alone is not parallax.
    ray = cameras[second][:3, :3].T @ rays[first]
    predicted = k @ ray
    current = k @ np.r_[normalized[second], 1.]
    displacement = float(np.linalg.norm(predicted[:2] / predicted[2] - current[:2] / current[2]))
    return {'point': point, 'track': track, 'views': len(indexes), 'baseline_m': baseline,
            'parallax_deg': angle, 'rotation_compensated_disparity_px': displacement,
            'max_reprojection_px': max(errors), 'calibrated_sensitivity': sensitivity}, None


def _scale_check(points, cameras, k, distortion):
    """Compare feature-cloud growth only when an axial translation supports it.

    The same tracked features are rectified into the FIRST camera orientation.
    This is a consistency diagnostic, not independent ground-truth validation.
    It never estimates depth from bbox growth or fills in a missing baseline.
    """
    if not points:
        return {'status': 'unavailable', 'reason': 'no triangulated features'}
    last = len(cameras) - 1
    eligible = [p for p in points if 0 in p['track'] and last in p['track']]
    translation = cameras[0][:3, :3].T @ (cameras[last][:3, 3] - cameras[0][:3, 3])
    if len(eligible) < MIN_POINTS or abs(translation[2]) < .002 or np.linalg.norm(translation[:2]) > abs(translation[2]) * .35:
        return {'status': 'not_applicable', 'reason': 'needs primarily axial translation and common visible features'}
    a, b, expected = [], [], []
    for point in eligible:
        for index, destination in ((0, a), (last, b)):
            pixel = point['track'][index]
            ray = cv2.undistortPoints(np.asarray(pixel, float).reshape(1, 1, 2), k, distortion).reshape(2)
            ray = cameras[0][:3, :3].T @ cameras[index][:3, :3] @ np.r_[ray, 1.]
            destination.append(ray[:2] / ray[2])
        z0 = (cameras[0][:3, :3].T @ (point['point'] - cameras[0][:3, 3]))[2]
        z1 = (cameras[0][:3, :3].T @ (point['point'] - cameras[last][:3, 3]))[2]
        expected.append(z0 / z1)
    a, b = np.asarray(a), np.asarray(b)
    a -= np.median(a, axis=0); b -= np.median(b, axis=0)
    ra, rb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    good = ra > .01
    if good.sum() < MIN_POINTS:
        return {'status': 'not_applicable', 'reason': 'tracked feature cloud too small for stable scale comparison'}
    observed = float(np.median(rb[good] / ra[good])); predicted = float(np.median(expected))
    residual = abs(observed / predicted - 1)
    return {'status': 'consistent' if residual <= .15 else 'inconsistent',
            'rotation_compensated_scale_ratio': observed, 'predicted_scale_ratio': predicted,
            'relative_disagreement': residual, 'independent_validation': False,
            'axial_translation_m': float(translation[2])}


def _tracked_roi_current(tracks, roi, reference, frames, size, stable_jaw):
    """Propagate a box as a 2D framing hint, independently of metric geometry.

    A robust similarity transform has fewer unsupported degrees of freedom than
    a homography extrapolated from a narrow label. It can fail on perspective
    deformation; low residuals still do not verify the object's hidden outline.
    """
    last = len(frames) - 1
    result = {'status': 'unavailable', 'framing_hint_only': True,
              'not_object_detection': True, 'not_clearance_evidence': True,
              'not_depth_evidence': True, 'not_physical_scale': True, 'source': 'wrist-only',
              'reference_frame_id': frames[reference].get('id'),
              'current_frame_id': frames[last].get('id')}
    if roi is None or reference == last:
        return {**result, 'reason': '需要先前影像中的物件範圍，才能追蹤到目前畫面'}
    if not stable_jaw:
        return {**result, 'reason': '夾爪開合已改變，不能把畫面中的固定夾爪當成物件追蹤'}
    width, height = size
    x, y, w, h = roi
    matched = [track for track in tracks if reference in track and last in track and
               x * width <= track[reference][0] <= (x + w) * width and
               y * height <= track[reference][1] <= (y + h) * height and
               np.linalg.norm(track[last] - track[reference]) > 3]
    if len(matched) < 8:
        return {**result, 'reason': '目前可辨識的移動特徵不足，先不外推物件範圍'}
    before = np.asarray([track[reference] for track in matched], float)
    after = np.asarray([track[last] for track in matched], float)
    affine, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC,
        ransacReprojThreshold=2.5, maxIters=2000, confidence=.995, refineIters=10)
    if affine is None or inliers is None or not np.isfinite(affine).all():
        return {**result, 'reason': '這批影像特徵無法形成一致的物件範圍移動'}
    keep = inliers.ravel().astype(bool)
    if keep.sum() < 8 or keep.mean() < .6:
        return {**result, 'reason': '物件特徵對應不一致或已被遮擋，先不更新範圍'}
    a, b = before[keep], after[keep]
    predicted = np.einsum('ij,kj->ki', affine, np.c_[a, np.ones(len(a))])
    error = np.linalg.norm(predicted - b, axis=1)
    p95 = float(np.percentile(error, 95))
    hull = float(cv2.contourArea(cv2.convexHull(a.astype(np.float32))))
    coverage = hull / (w * width * h * height)
    span = np.ptp(a, axis=0) / [w * width, h * height]
    eig = np.linalg.eigvalsh(np.cov(a.T))
    scale = float(np.sqrt(np.linalg.det(affine[:, :2])))
    if (p95 > 2.5 or coverage < .015 or min(span) < .1 or
            eig[0] < 4 or eig[0] / max(eig[1], 1e-12) < .01 or not .5 <= scale <= 2.):
        return {**result, 'reason': '追蹤點太集中或形變不一致，無法可靠推估完整物件範圍'}
    corners = np.array([[x * width, y * height], [(x + w) * width, y * height],
                        [(x + w) * width, (y + h) * height], [x * width, (y + h) * height]])
    projected = np.einsum('ij,kj->ki', affine, np.c_[corners, np.ones(4)]) / [width, height]
    low, high = projected.min(axis=0), projected.max(axis=0)
    clipped_low, clipped_high = np.maximum(low, 0), np.minimum(high, 1)
    if np.any(clipped_high <= clipped_low):
        return {**result, 'reason': '追蹤到的範圍已離開目前畫面'}
    visible = float(np.prod(clipped_high - clipped_low) / np.prod(high - low))
    return {**result, 'status': 'tracked',
            'bbox': {'x': float(clipped_low[0]), 'y': float(clipped_low[1]),
                     'w': float(clipped_high[0] - clipped_low[0]), 'h': float(clipped_high[1] - clipped_low[1])},
            'projected_bbox': {'x': float(low[0]), 'y': float(low[1]),
                              'w': float(high[0] - low[0]), 'h': float(high[1] - low[1])},
            'projected_corners_normalized': projected.tolist(),
            'visible_fraction_of_projected_bbox': visible,
            'affine_reference_to_current_px': affine.tolist(),
            'support': {'matched_identities': len(matched), 'inlier_identities': int(keep.sum()),
                        'inlier_fraction': float(keep.mean()), 'p95_reprojection_px': p95,
                        'reference_feature_hull_fraction': coverage,
                        'reference_span_fraction': span.tolist()},
            'reason': '依同一批影像特徵推估目前取景範圍；仍需重新辨識物件與可行動作'}


def _visible_extent(box, reference, frames, points, cameras, k, distortion, size, relative_error):
    """Visible bbox projected at its own view's surface depth, never full size.

    Keep the original ROI coordinate system; applying an earlier bbox at the
    current camera's depth would manufacture size changes as the camera moves.
    Distortion is removed before measuring normalized image extent.
    """
    if box is None:
        return None
    x, y, w, h = box
    # Distortion can put the rectified extrema midway along an image edge.
    # Sample all four edges rather than shrinking the span to four corners.
    pixels = np.array([[u * size[0], v * size[1]]
                       for t in np.linspace(0., 1., 65)
                       for u, v in ((x + t * w, y), (x + t * w, y + h),
                                    (x, y + t * h), (x + w, y + t * h))], float)
    rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, distortion).reshape(-1, 2)
    local = np.einsum('ij,kj->ki', inv(cameras[reference]), np.c_[points, np.ones(len(points))])
    z = float(np.median(local[:, 2]))
    extent = np.ptp(rays, axis=0) * z
    if z <= 0 or not np.isfinite(extent).all() or np.any(extent <= 0):
        return None
    return {'source': 'wrist-only', 'reference_frame_id': frames[reference].get('id'),
            'scope': 'visible_bbox_projection_at_tracked_surface_depth',
            'width_m': float(extent[0]), 'height_m': float(extent[1]),
            'reference_camera_depth_m': z,
            'width_interval_m': [float(extent[0] * max(0., 1 - relative_error)), float(extent[0] * (1 + relative_error))],
            'height_interval_m': [float(extent[1] * max(0., 1 - relative_error)), float(extent[1] * (1 + relative_error))],
            'uncertainty_kind': 'same heuristic camera sensitivity as depth; not a statistical interval',
            'touches_image_edges': [name for name, touches in
                (('left', x <= .005), ('top', y <= .005), ('right', x + w >= .995), ('bottom', y + h >= .995)) if touches],
            'full_object_dimensions_verified': False, 'object_dimensions_used': False,
            'assumption': 'visible bounds projected onto one camera-parallel plane at tracked surface depth',
            'limitations': ['not cylinder diameter or complete height', 'occlusion and unseen surfaces remain unknown']}


def estimate_active_depth(frames, model, *, roi=None, roi_frame_id=None, calibration=None, cancel=None,
                          refine_camera_prior=False, workspace_floor_z_m=None, _observation_cache=None):
    """Return JSON-safe sparse geometry and uncertainty; performs no motor I/O.

    ROI is in ``roi_frame_id`` coordinates (last frame when omitted), and must
    track to the current last view. An absent reference is rejected, never
    silently reinterpreted in a newer image's coordinates.
    Without ROI the output is scene geometry and must not be assigned to an
    object. Historical photos may be old; the caller checks the live last image.
    ``refine_camera_prior=True`` explicitly enables a <=12s same-attempt prior
    fit after CAD failure. Its target-heldout checks never certify calibration.
    """
    result = {'status': 'insufficient', 'source': 'wrist-only', 'metric_calibrated': False,
              'grasp_ready': False, 'scope': 'object_roi_surface' if roi is not None else 'unassigned_scene_surface',
              'points': [], 'distance': None, 'visible_extent': None, 'roi': None, 'view_count': len(frames),
              'assumptions': ['scene stationary during observation', 'measured CAD FK approximates camera movement'],
              'limitations': ['Consistent object motion can mimic depth; image geometry cannot prove stasis.',
                             'Visible feature surfaces are not hidden object geometry or the jaw-gap center.']}
    prepared = None
    try:
        if workspace_floor_z_m is not None and (type(workspace_floor_z_m) not in (int, float)
                or not math.isfinite(workspace_floor_z_m)):
            raise ValueError('工作平面需要有限的基座座標')
        box = _roi(roi)
        if box is not None:
            result['roi'] = dict(zip(('x', 'y', 'w', 'h'), box))
        if _observation_cache is None:
            images, transforms = _read_frames(frames, model, cancel)
        else:
            # Reuse this call's immutable images/tracks after fitting a new
            # camera model; do not decode and run SIFT twice on the same views.
            cached_frames, images, transforms, _, _, cached_roi, _ = _observation_cache
            if cached_frames is not frames or cached_roi != (box, roi_frame_id):
                raise ValueError('影像快取不屬於本次觀察')
        roi_index = len(frames) - 1
        if box is not None and roi_frame_id is not None:
            indexes = [i for i, frame in enumerate(frames) if frame.get('id') == roi_frame_id]
            if len(indexes) != 1:
                raise ValueError('物件範圍對應的腕部影像不在這批觀察中，不能套用到新畫面')
            roi_index = indexes[0]
        result['roi_frame_id'] = frames[roi_index].get('id') if box is not None else None
        result['current_frame_id'] = frames[-1].get('id')
        result['current_measured_joints'] = dict(frames[-1]['joints'])
        if calibration and calibration.get('prior_quality') == 'coarse':
            if (calibration.get('observation_frame_ids') != [f.get('id') for f in frames]
                    or calibration.get('roi_frame_id') != result['roi_frame_id']):
                raise ValueError('粗略相機先驗不屬於這一批照片與物件範圍')
        size = images[0].shape[::-1]
        cameras, k, distortion, uncertainty, provenance = _camera_model(size, transforms, frames[-1]['camera_generation'], calibration)
        result.update(provenance=provenance, metric_calibrated=provenance['metric_calibrated'],
                      frame_ids=[f.get('id', str(f['sequence'])) for f in frames],
                      camera_transform_base_m=cameras[-1].tolist())
        baselines = [np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in combinations(cameras, 2)]
        result['maximum_camera_translation_mm'] = float(max(baselines) * 1000)
        stable_jaw = max(f['joints']['gripper'] for f in frames) - min(f['joints']['gripper'] for f in frames) <= 6
        feature_cache = {}
        if _observation_cache is None:
            tracks, diagnostics = _image_tracks(images, box, cancel, roi_index=roi_index,
                                                allow_roi_below_gripper=stable_jaw, _feature_cache=feature_cache)
        else:
            tracks, diagnostics = _observation_cache[3], deepcopy(_observation_cache[4])
            feature_cache = _observation_cache[6]
        prepared = (frames, images, transforms, tracks, deepcopy(diagnostics), (box, roi_frame_id), feature_cache)
        diagnostics['jaw_unchanged_for_roi_expansion'] = stable_jaw
        result['tracking'] = diagnostics
        result['tracked_roi_current'] = _tracked_roi_current(tracks, box, roi_index, frames, size, stable_jaw)
        _cancel(cancel)
        if max(baselines) < MIN_BASELINE_M:
            raise ValueError('這些畫面只有轉動、張爪或太小的平移；還沒有能估距的相機位移')
        solved, rejected, rejected_residuals = [], {}, []
        for track in tracks:
            _cancel(cancel)
            point, reason = _solve_track(track, cameras, k, distortion, size,
                max_reprojection_px=provenance['reprojection_limit_px'],
                **({'pose_uncertainty': uncertainty} if provenance['metric_calibrated'] else {}))
            if reason is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
                if point is not None:
                    rejected_residuals.append(point['max_reprojection_px'])
            else:
                solved.append(point)
        diagnostics.update(retained=len(solved), rejected=rejected)
        if rejected_residuals:
            diagnostics['rejected_median_reprojection_px'] = float(np.median(rejected_residuals))
        if len(solved) < MIN_POINTS:
            raise ValueError('相機位移已量到，但畫面與相機模型還對不上；目前不能給出可靠距離')
        three_view = [point for point in solved if point['views'] >= 3]
        available_three_view = len(three_view)
        available_two_view = len(solved) - available_three_view
        excluded_two_view = 0
        if provenance['prior_quality'] == 'coarse' and available_three_view < MIN_POINTS:
            raise ValueError('粗略相機先驗還需要六個至少跨三張影像一致的物件特徵')
        if provenance['camera_model'] == 'same-attempt-refined-prior' and available_three_view >= MIN_POINTS:
            # A refined prior was qualified on independent multi-view target
            # identities. Do not let many weaker two-view tracks dominate the
            # published distance after that qualification.
            excluded_two_view = available_two_view
            solved = three_view
        _cancel(cancel)
        scale = _scale_check(solved, cameras, k, distortion)
        result['scale_crosscheck'] = scale
        if scale['status'] == 'inconsistent':
            raise ValueError('轉動補償後的影像縮放與移動距離不一致，暫不採用距離')
        points = np.array([p['point'] for p in solved])
        if workspace_floor_z_m is not None:
            median_z = float(np.median(points[:, 2]))
            result['workspace_consistency'] = {
                'source': 'caller_declared_CAD_workspace_plane', 'floor_z_m': workspace_floor_z_m,
                'median_surface_z_m': median_z, 'allowed_plane_disagreement_m': .05,
                'passed': median_z >= workspace_floor_z_m - .05,
                'not_independent_calibration': True}
            if not result['workspace_consistency']['passed']:
                raise ValueError('推算位置落到工作桌面下方，與目前工作空間矛盾；需要修正相機模型，不能拿這個距離規劃抓取')
        camera = inv(cameras[-1]); gripper = inv(_rigid(transforms[-1]['gripper_frame_link']))
        homogeneous = np.c_[points, np.ones(len(points))]
        cam = np.einsum('ij,kj->ki', camera, homogeneous)[:, :3]
        tip = np.einsum('ij,kj->ki', gripper, homogeneous)[:, :3]
        if not np.isfinite(cam).all() or not np.isfinite(tip).all():
            raise ValueError('距離轉換沒有得到有限數值')
        median_depth = float(np.median(cam[:, 2]))
        angle = float(np.median([p['parallax_deg'] for p in solved]))
        baseline = float(np.median([p['baseline_m'] for p in solved]))
        residual = float(np.percentile([p['max_reprojection_px'] for p in solved], 95))
        pixel_angle = math.degrees(math.atan2(max(.5, residual), min(k[0, 0], k[1, 1])))
        relative_error = uncertainty['focal_relative'] + uncertainty['relative_translation_m'] / baseline
        relative_error += (uncertainty['relative_rotation_deg'] + 2 * pixel_angle) / max(angle, MIN_PARALLAX_DEG)
        if provenance['metric_calibrated']:
            # Use the SAME finite pose/feature aperture that admitted the
            # points. Residuals explained by pose are not added again as if
            # they were independent feature-localization noise.
            relative_error = max(p['calibrated_sensitivity']['relative_depth_error'] for p in solved)
            diagnostics['calibrated_reprojection'] = {
                'model': provenance['reprojection_model'],
                'tracking_noise_px': provenance['reprojection_limit_px'],
                'minimum_disparity_deg': min(p['calibrated_sensitivity']['minimum_disparity_deg'] for p in solved),
                'maximum_projected_pose_and_tracking_px': max(p['calibrated_sensitivity']['max_projected_pose_and_tracking_px'] for p in solved),
                'maximum_relative_depth_sensitivity': relative_error,
                'observed_reprojection_residual_is_not_tracking_noise': True}
        depth_error = median_depth * relative_error
        reference_translation_error = max(uncertainty['relative_translation_m'],
                                          uncertainty['handeye_translation_disagreement_m'],
                                          uncertainty['base_position_uncertainty_m'])
        reference_rotation_error = float(np.median(np.linalg.norm(cam, axis=1))) * math.sin(
            math.radians(min(90., uncertainty['handeye_rotation_disagreement_deg'])))
        absolute_error = reference_translation_error + reference_rotation_error
        multiview_count = sum(p['views'] >= 3 for p in solved)
        _cancel(cancel)
        result.update(status='estimated',
            visible_extent=_visible_extent(box, roi_index, frames, points, cameras, k, distortion, size, relative_error),
            static_consistency=('three_or_more_views' if multiview_count == len(solved)
                                else 'mixed_two_and_three_view_support' if multiview_count
                                else 'two_view_only_unverified_motion'),
            geometry={'retained_points': len(points), 'three_view_points': multiview_count,
                      'available_three_view_points': available_three_view,
                      'available_two_view_points': available_two_view,
                      'two_view_points_excluded_from_distance': excluded_two_view,
                      'median_parallax_deg': angle, 'p95_reprojection_px': residual,
                      'median_rotation_compensated_disparity_px': float(np.median([p['rotation_compensated_disparity_px'] for p in solved]))},
            distance={'camera_optical_axis_m': median_depth,
                      'camera_range_m': float(np.median(np.linalg.norm(cam, axis=1))),
                      'gripper_forward_axis_m': float(np.median(tip[:, 2])),
                      'gripper_lateral_m': float(np.median(np.linalg.norm(tip[:, :2], axis=1))),
                      'position_base_m': np.median(points, axis=0).tolist(),
                      'reference': 'median tracked visible surface in last view'},
            uncertainty={'depth_error_bound_m': depth_error,
                         'camera_depth_interval_m': [max(0., median_depth - depth_error), median_depth + depth_error],
                         'gripper_axis_error_bound_m': depth_error + absolute_error,
                         'base_position_error_bound_m': depth_error + absolute_error,
                         'position_error_components': {
                             'model': 'relative-triangulation-plus-reference-placement-v1',
                             'relative_surface_m': depth_error,
                             'reference_translation_m': reference_translation_error,
                             'reference_rotation_m': reference_rotation_error,
                             'interpretation': 'Same total point-position sensitivity. Relative triangulation error remains shape error; a common reference translation changes center but not radius. Reference rotation is retained for a cylinder fitted against a fixed upright axis.'},
                         'assumed_model_bounds': uncertainty,
                         'bound_kind': 'heuristic sensitivity to assumed model error, not a guaranteed physical bound',
                         'statistical_confidence_interval': False,
                         'quality': 'coarse' if relative_error > .3 else 'moderate'},
            reason=('有粗略距離估計，但目前相機模型誤差較大，還不能當成精準抓取位置'
                    if relative_error > .3 else '已由實際相機位移與影像視差估出可見表面的距離；仍須核對接近與接觸位置'))
        # Bounded samples for planner/simulator consumers, always with provenance.
        for i in np.linspace(0, len(points) - 1, min(64, len(points)), dtype=int):
            _cancel(cancel)
            result['points'].append({'position_base_m': points[i].tolist(),
                'current_pixel': solved[i]['track'][len(frames) - 1].tolist(),
                'camera_depth_m': float(cam[i, 2]), 'gripper_forward_m': float(tip[i, 2]),
                'views': solved[i]['views']})
    except (ValueError, RuntimeError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        result.update(status='insufficient', reason=str(exc), distance=None, visible_extent=None, points=[])
    if (refine_camera_prior and calibration is None and result['status'] == 'insufficient'
            and len(frames) >= 3 and not (cancel is not None and cancel.is_set())):
        from .active_camera_prior import refine_active_camera_prior
        refinement_args = {'mode': 'coarse' if len(frames) >= 5 else 'strict'}
        if len(frames) >= 5 and prepared is not None and prepared[6].get('features') is not None:
            refinement_args['_prepared_observations'] = (frames, model, prepared[1], prepared[2], prepared[6]['features'])
        refinement = refine_active_camera_prior(frames, model, roi=roi, roi_frame_id=roi_frame_id, cancel=cancel,
            **refinement_args)
        if refinement['status'] == 'refined':
            refined_result = estimate_active_depth(frames, model, roi=roi, roi_frame_id=roi_frame_id,
                calibration=refinement['calibration'], cancel=cancel, workspace_floor_z_m=workspace_floor_z_m,
                _observation_cache=prepared)
            refined_result['initial_camera_model_failure'] = {key: result.get(key) for key in ('reason', 'tracking')}
            result = refined_result
        result['camera_prior_refinement'] = {key: value for key, value in refinement.items() if key != 'calibration'}
    if cancel is not None and cancel.is_set():
        result.update(status='insufficient', reason='距離估算已停止', distance=None, visible_extent=None, points=[])
    return result
