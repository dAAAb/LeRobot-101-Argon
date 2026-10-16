"""Turn wrist triangulation into a frame-bound object measurement.

This is an evidence adapter, not a motion permission. In particular a surface
point is not a grasp center, and a bounding box of textured points is not the
size of the whole object. No persistent scene state, device access or model
inference lives here. Callers must still check image age before an action.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np


def _number(value, *, minimum=0.):
    return (type(value) in (int, float) and math.isfinite(value)
            and value >= minimum)


def _vector(value):
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and all(_number(v, minimum=-math.inf) for v in value))


def _matching_frame(depth, frame):
    if ((depth.get('provenance') or {}).get('camera_generation') != frame.get('camera_generation')
            or not frame.get('camera_generation') or not frame.get('id')):
        return False
    if depth.get('current_frame_id') == frame['id']:
        return True
    # The runtime already compares pixels AND encoder poses while the depth
    # calculation is running. Only that explicitly recorded refresh can rebind
    # the estimate; a nearby timestamp or equal joint goals cannot do so.
    refresh = depth.get('stationary_refresh') or {}
    return (refresh.get('current_frame_id') == frame['id']
            and bool(refresh.get('image_comparison'))
            and _number(refresh.get('max_joint_drift_ticks'))
            and refresh['max_joint_drift_ticks'] <= 3)


def _calibration_matches(depth, observation, frame):
    if not isinstance(observation, dict):
        return False
    provenance = depth.get('provenance') or {}
    revision = observation.get('calibration_revision')
    if not (observation.get('source') == 'wrist'
            and observation.get('frame_id') == frame.get('id')
            and observation.get('camera_generation') == frame.get('camera_generation')
            and observation.get('metric_calibrated') is True
            and depth.get('metric_calibrated') is True
            and provenance.get('metric_calibrated') is True
            and provenance.get('camera_model') == 'validated-calibration'
            and isinstance(revision, str) and bool(revision)
            and provenance.get('calibration_revision') == revision
            and bool(observation.get('calibration_fingerprint'))
            and provenance.get('calibration_fingerprint') == observation['calibration_fingerprint']
            and provenance.get('geometry_revision') == observation.get('geometry_revision')
            and bool(observation.get('geometry_revision'))):
        return False
    return True


def _position_error_components(uncertainty, total):
    """Use only a complete additive decomposition; old records stay conservative.

    This partitions the existing heuristic, not a newly measured covariance or
    independent bound. Missing, malformed or inconsistent metadata never lowers
    shape uncertainty. In particular relative per-view errors do NOT cancel.
    """
    value = uncertainty.get('position_error_components')
    fields = ('relative_surface_m', 'reference_translation_m', 'reference_rotation_m')
    if (not isinstance(value, dict)
            or value.get('model') != 'relative-triangulation-plus-reference-placement-v1'
            or any(not _number(value.get(key)) for key in fields)
            or not math.isclose(sum(value[key] for key in fields), total, rel_tol=1e-9, abs_tol=1e-12)):
        return None
    return {key: value[key] for key in ('model', *fields)}


def _surface(depth):
    distance = depth.get('distance') or {}
    uncertainty = depth.get('uncertainty') or {}
    center = distance.get('position_base_m')
    error = uncertainty.get('base_position_error_bound_m')
    optical_z = distance.get('camera_optical_axis_m')
    if not (_vector(center) and _number(error) and _number(optical_z, minimum=.001)):
        return None
    points = []
    for item in depth.get('points', []):
        if (isinstance(item, dict) and _vector(item.get('position_base_m'))
                and type(item.get('views')) is int and item['views'] >= 2):
            points.append(deepcopy(item))
    if len(points) < 6:
        return None
    xyz = np.asarray([p['position_base_m'] for p in points], float)
    # Report an actual sampled point near the published feature median. This
    # point can be used for camera aiming; it is never an inferred back face.
    representative = xyz[np.argmin(np.linalg.norm(xyz - np.asarray(center), axis=1))]
    lower, upper = np.quantile(xyz, [.05, .95], axis=0)
    result = {
        'position_base_m': list(center),
        'representative_measured_point_base_m': representative.tolist(),
        'reference': 'median tracked visible surface, not object or jaw center',
        'camera_optical_axis_m': optical_z,
        'camera_depth_interval_m': deepcopy(uncertainty.get('camera_depth_interval_m')),
        'position_error_bound_m': error,
        'uncertainty_kind': uncertainty.get('bound_kind', 'unspecified engineering estimate'),
        'statistical_confidence_interval': False,
        'observed_sample_bounds_base_m': {'lower_m': lower.tolist(), 'upper_m': upper.tolist(),
            'scope': 'central 90 percent of tracked surface samples, not full object bounds'},
        'sample_count': len(points), 'points': points,
    }
    components = _position_error_components(uncertainty, error)
    if components is not None:
        result['position_error_components'] = components
    return result


def _historical_measurement(previous_estimates, generation):
    """Diagnostic history never supplies the current target or current frame."""
    for depth in reversed(tuple(previous_estimates)):
        if not isinstance(depth, dict) or depth.get('source') != 'wrist-only':
            continue
        if (depth.get('provenance') or {}).get('camera_generation') != generation:
            continue
        if depth.get('status') != 'estimated' or depth.get('scope') != 'object_roi_surface':
            continue
        surface = _surface(depth)
        if surface is not None:
            return {'frame_id': depth.get('current_frame_id'),
                    'camera_optical_axis_m': surface['camera_optical_axis_m'],
                    'camera_depth_interval_m': surface['camera_depth_interval_m'],
                    'usable_as_current_target': False,
                    'reason': 'A previous view had a surface estimate; new movement invalidated its live binding.'}
    return None


def _decision(result, action, fields, message, *, further_views=False):
    result.update(recommended_action=action, missing_fields=fields, message=message,
                  further_target_views_useful=further_views)
    return result


def build_object_geometry(depth, *, current_frame, track_id, observation=None,
                          previous_estimates=()):
    """Describe what is measured and which operation can reduce uncertainty.

    ``observation`` is the spatial-contract packet for ``current_frame``. Both
    depth and packet must name the SAME validated calibration/geometry revision
    before metric status is granted. Profile selection alone cannot upgrade
    coordinates previously computed using an unvalidated camera prior.

    ``plan_approach`` means the local planner may generate a proposal towards a
    measured surface, with its own uncertainty margin. It never authorizes
    execution, closure, or treats the surface as a complete object center.
    """
    if not isinstance(current_frame, dict) or current_frame.get('source') != 'wrist':
        raise ValueError('物件幾何只接受腕部相機資料')
    if not isinstance(track_id, str) or not track_id.strip():
        raise ValueError('物件幾何需要穩定的物件 ID')
    if depth is not None and (not isinstance(depth, dict) or depth.get('source') != 'wrist-only'):
        raise ValueError('物件幾何不能使用其他相機的深度')
    if observation is not None and (not isinstance(observation, dict)
            or observation.get('source') != 'wrist'
            or observation.get('frame_id') != current_frame.get('id')
            or observation.get('camera_generation') != current_frame.get('camera_generation')):
        raise ValueError('座標封包與目前腕部影像不符')
    depth = depth or {}
    historical = _historical_measurement(previous_estimates, current_frame.get('camera_generation'))
    result = {
        'schema_version': 1, 'source': 'wrist-only', 'track_id': track_id,
        'frame_id': current_frame.get('id'), 'camera_generation': current_frame.get('camera_generation'),
        'capture_monotonic': current_frame.get('capture_monotonic'),
        'coordinate_frame': 'base', 'units': 'meters',
        'geometry_revision': (observation or {}).get('geometry_revision'),
        'calibration_revision': (observation or {}).get('calibration_revision'),
        'calibration_fingerprint': (observation or {}).get('calibration_fingerprint'),
        'status': 'unmeasured', 'metric_calibrated': False,
        'observed_surface': None, 'object_center_base_m': None,
        'full_dimensions_m': None, 'grasp_center_base_m': None,
        'grasp_ready': False, 'execution_authorized': False,
        'last_successful_measurement': historical,
        'visible_extent': None,
        'diagnostics': {'depth_status': depth.get('status'), 'depth_reason': depth.get('reason'),
            'view_count': depth.get('view_count', 0),
            'maximum_camera_translation_mm': depth.get('maximum_camera_translation_mm'),
            'camera_model': (depth.get('provenance') or {}).get('camera_model'),
            'refinement_reason': (depth.get('camera_prior_refinement') or {}).get('reason')},
    }
    if not depth:
        return _decision(result, 'measure_target', ['target_surface'],
                         '先定位物件，再用腕部影像與實際位移估距。', further_views=True)
    if depth.get('scope') != 'object_roi_surface':
        return _decision(result, 'measure_target', ['target_roi'],
                         '目前量到的是背景；需要先找出你指定的物件。', further_views=True)
    if not _matching_frame(depth, current_frame):
        result['status'] = 'stale'
        return _decision(result, 'measure_target', ['current_target_measurement'],
                         '手臂或畫面已更新，需要重新量測目前的物件位置。', further_views=True)
    if depth.get('status') == 'estimated':
        surface = _surface(depth)
        if surface is not None:
            result.update(status='observed', observed_surface=surface,
                          visible_extent=deepcopy(depth.get('visible_extent')),
                          metric_calibrated=_calibration_matches(depth, observation, current_frame))
            if not result['metric_calibrated']:
                cm = surface['camera_optical_axis_m'] * 100
                return _decision(result, 'calibrate_camera', ['validated_camera_and_arm_geometry'],
                    f'已估出物件表面離鏡頭約 {cm:.1f} 公分；現在缺的是相機與手臂座標校準，繼續重複換角度不能取代校準。')
            jaw = (observation or {}).get('jaw_geometry') or {}
            if jaw.get('planning_eligible', jaw.get('metric_calibrated')) is not True:
                return _decision(result, 'calibrate_jaws', ['validated_jaw_center_and_aperture'],
                    '物件表面位置已量到；接著需要對準兩側夾墊中心，才能計算夾爪要伸到哪裡。')
            return _decision(result, 'plan_approach', ['grasp_center_and_contact_geometry'],
                '已取得物件表面位置，可以先由本機規劃接近路徑；合爪前仍需確認物件位於兩側夾墊之間。')
    baseline = depth.get('maximum_camera_translation_mm')
    tracking = depth.get('tracking') or {}
    rejected = tracking.get('rejected') or {}
    candidates = tracking.get('candidate_tracks', 0)
    model_rejected = sum(v for k, v in rejected.items()
                         if k in ('nonrigid_or_pose_inconsistent', 'invalid_depth') and type(v) is int)
    # Camera/model inconsistency and lost target features are different from
    # too little parallax. Repeating the same target-only observation cannot
    # independently certify intrinsics, hand-eye or the CAD joint zero points.
    prior = (depth.get('provenance') or {}).get('metric_calibrated') is not True
    if prior and (historical is not None or (_number(baseline, minimum=1.)
            and type(candidates) is int and candidates >= 6 and model_rejected >= candidates / 2)):
        return _decision(result, 'calibrate_camera', ['validated_camera_and_arm_geometry'],
            '已移動到不同位置，但影像與相機／手臂座標對不上；需要先校準座標，增加同樣的觀察次數無法解決。')
    if _number(baseline) and baseline < 1.:
        return _decision(result, 'change_baseline', ['translational_parallax'],
            '目前主要是轉動鏡頭；需要換到有實際側移的位置，才能估出距離。', further_views=True)
    if rejected.get('insufficient_translation_parallax', 0):
        return _decision(result, 'change_baseline', ['translational_parallax'],
            '物件在畫面中的視差仍太小；下一個視角應增加側向位移。', further_views=True)
    return _decision(result, 'measure_target', ['trackable_target_surface'],
        '目前物件特徵被遮住或對應不足；需要換到能看清同一物件表面的角度。', further_views=True)


def surface_planning_evidence(geometry):
    """Export local-planner inputs without silently granting grasp clearance."""
    if not isinstance(geometry, dict) or geometry.get('source') != 'wrist-only':
        raise ValueError('規劃需要腕部物件幾何')
    if geometry.get('status') != 'observed' or geometry.get('observed_surface') is None:
        raise ValueError('目前沒有可供規劃的物件表面量測')
    surface = geometry['observed_surface']
    return {'surface_base_m': deepcopy(surface['position_base_m']),
            'target_uncertainty_m': surface['position_error_bound_m'],
            'evidence': {'source': 'wrist', 'frame_id': geometry['frame_id'],
                'camera_generation': geometry['camera_generation'],
                'geometry_revision': geometry['geometry_revision'],
                'calibration_revision': geometry['calibration_revision'],
                'geometry_verified': geometry['metric_calibrated'],
                'target_verified': geometry['metric_calibrated'],
                'scope': 'observed_surface_not_grasp_center',
                'track_id': geometry['track_id']}}


def fit_upright_cylinder(geometry, *, shape_intent, axis_base):
    """Fit a circular cross-section only for an explicitly identified cylinder.

    ``axis_base`` is the caller's declared upright direction; it is not guessed
    from a narrow texture strip. Radius is estimated from measured XYZ points,
    never from an object name, a known product size, image box or model default.
    The axis point lies at the median *visible* height, not the object's middle.
    Split-height fits reject cylinders invented from a single planar/noisy strip.
    """
    from scipy.optimize import least_squares

    if shape_intent != 'upright_cylinder':
        raise ValueError('圓柱擬合需要明確的直立圓柱形狀假設')
    if not _vector(axis_base) or abs(np.linalg.norm(axis_base) - 1) > 1e-5:
        raise ValueError('圓柱軸向需要基座座標中的單位向量')
    if not isinstance(geometry, dict) or geometry.get('source') != 'wrist-only':
        raise ValueError('圓柱擬合只接受腕部量測')
    output = {'status': 'insufficient', 'source': 'wrist-only',
        'frame_id': geometry.get('frame_id'), 'track_id': geometry.get('track_id'),
        'camera_generation': geometry.get('camera_generation'),
        'geometry_revision': geometry.get('geometry_revision'),
        'calibration_revision': geometry.get('calibration_revision'),
        'shape_intent': shape_intent, 'axis_base': list(axis_base),
        'full_height_m': None, 'grasp_ready': False, 'execution_authorized': False,
        'metric_calibrated': geometry.get('metric_calibrated') is True,
        'diagnostics': {},
        'limitations': ['upright circular-cylinder hypothesis, not arbitrary object shape',
                        'axis point is at visible median height, not complete object center',
                        'occluded surfaces and both-pad contact remain unverified']}

    def reject(reason):
        return {**output, 'reason': reason}

    surface = geometry.get('observed_surface') or {}
    records = surface.get('points', [])
    if geometry.get('status') != 'observed' or len(records) < 24:
        return reject('需要更多分布在圓筒不同側面與高度的表面點')
    if any(not isinstance(p, dict) or not _vector(p.get('position_base_m')) for p in records):
        return reject('表面點座標無效')
    xyz = np.asarray([p['position_base_m'] for p in records])
    axis = np.asarray(axis_base, float)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    basis = np.stack((u, v), axis=1)
    xy = xyz @ basis
    z = xyz @ axis

    def circle(data):
        origin = np.mean(data, axis=0)
        centered = data - origin
        design = np.c_[2 * centered, np.ones(len(data))]
        if np.linalg.matrix_rank(design) < 3:
            raise ValueError('flat samples')
        beta = np.linalg.lstsq(design, np.sum(centered ** 2, axis=1), rcond=None)[0]
        r2 = beta[2] + beta[:2] @ beta[:2]
        if r2 <= 0:
            raise ValueError('invalid circle')
        initial = np.r_[origin + beta[:2], math.sqrt(r2)]
        fitted = least_squares(lambda p: np.linalg.norm(data - p[:2], axis=1) - p[2],
                               initial, loss='soft_l1', f_scale=.001,
                               bounds=([-np.inf, -np.inf, .0001], [np.inf, np.inf, np.inf]),
                               max_nfev=150)
        if not fitted.success:
            raise ValueError('circle fit did not converge')
        c, radius = fitted.x[:2], fitted.x[2]
        residuals = np.abs(np.linalg.norm(data - c, axis=1) - radius)
        return c, radius, residuals

    try:
        center, radius, residuals = circle(xy)
        residual_limit = min(.004, max(.0015, 3 * 1.4826 * float(np.median(residuals))))
        inliers = residuals <= residual_limit
        if inliers.sum() < 24 or inliers.mean() < .8:
            return reject('表面點無法由同一個圓柱截面解釋')
        center, radius, residuals = circle(xy[inliers])
        in_xy, in_z = xy[inliers], z[inliers]
        angles = np.sort(np.arctan2(in_xy[:, 1] - center[1], in_xy[:, 0] - center[0]))
        coverage = float(360 - np.degrees(np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))))
        centered = in_xy - center
        jacobian = np.c_[centered / np.linalg.norm(centered, axis=1)[:, None], np.ones(len(centered))]
        condition = float(np.linalg.cond(jacobian))
        radial_rms = float(np.sqrt(np.mean(residuals ** 2)))
        line_rms = float(np.linalg.svd(in_xy - np.mean(in_xy, axis=0), compute_uv=False)[-1] / np.sqrt(len(in_xy)))
        output['diagnostics'].update(sample_count=len(xyz), inlier_count=int(inliers.sum()),
            arc_coverage_deg=coverage, circle_condition=condition,
            radial_rms_m=radial_rms, best_line_rms_m=line_rms)
        if coverage < 110 or condition > 25:
            return reject('只看到太窄的一側弧面，還不能確定圓筒半徑與中心')
        if radial_rms > min(.003, radius * .12) or radial_rms > line_rms * .65:
            return reject('這批點較像平面或雜訊，還不能證實圓柱曲率')
        height = float(np.ptp(in_z))
        if height < radius:
            return reject('不同高度的點太少，還不能核對圓柱截面是否一致')
        # Low and high slices observe disjoint surface points. This checks
        # cylindrical consistency, NOT independent camera calibration.
        split = float(np.median(in_z))
        sections = [in_xy[in_z < split], in_xy[in_z >= split]]
        if min(map(len, sections)) < 8:
            return reject('上下兩段的獨立表面點不足')
        fits = [circle(section) for section in sections]
        disagreement = max(float(np.linalg.norm(fits[0][0] - fits[1][0])), abs(fits[0][1] - fits[1][1]))
        heldout = max(float(np.quantile(np.abs(np.linalg.norm(sections[1 - i] - fits[i][0], axis=1) - fits[i][1]), .95))
                      for i in range(2))
        output['diagnostics'].update(split_height_disagreement_m=disagreement,
                                     heldout_radial_p95_m=heldout)
        if disagreement > max(.003, radius * .15) or heldout > min(.004, radius * .2):
            return reject('上下兩段估出的半徑或中心不同，不能把它當成一致圓筒')
        axial_range = np.quantile(in_z, [.05, .95])
        centerline = basis @ center + axis * split
        shared_error = surface.get('position_error_bound_m')
        if not _number(shared_error):
            return reject('缺少相機與手臂共用的量測誤差範圍')
        shape_error = max(heldout, disagreement, radial_rms * condition)
        components = _position_error_components(surface, shared_error)
        # The same rigid translation of all points cannot alter a radius. It
        # still moves the grasp center. Relative triangulation distortion and
        # rotation against the fixed upright assumption remain in BOTH budgets.
        radius_measurement_error = (shared_error if components is None else
            components['relative_surface_m'] + components['reference_rotation_m'])
        output.update(status='hypothesis', centerline_point_base_m=centerline.tolist(),
            radius_m=float(radius), diameter_m=float(2 * radius),
            observed_axial_interval_m=axial_range.tolist(),
            observed_axial_span_m=float(axial_range[1] - axial_range[0]),
            center_position_error_m=float(shared_error + shape_error),
            radius_sensitivity_m=float(radius_measurement_error + shape_error),
            uncertainty_components={'center_measurement_m': float(shared_error),
                'radius_measurement_m': float(radius_measurement_error),
                'fitted_shape_sensitivity_m': float(shape_error),
                'common_translation_excluded_from_radius_m': (0. if components is None else
                    components['reference_translation_m']),
                'partition': 'legacy-unsplit' if components is None else components['model']},
            uncertainty_kind='heuristic camera error plus split-surface shape sensitivity; not a statistical interval',
            reason='已從表面曲率估出圓柱截面；仍須核對夾墊接觸與看不見的部分')
        return output
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return reject('表面點的曲率不足或擬合退化，還不能估出圓柱中心')
