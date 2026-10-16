"""Deterministic, no-hardware SO101 task-space proposals.

The five body joints constrain a point and one tool axis (3 + 2 DOF), never a
general six-DOF pose. The same FK, table screen and joint interpolation used by
``motion_program`` produce the preview and executor contracts. No raw targets
come from a language model here, and no function dispatches a motor command.

A caller must still authorize the complete external-object path against fresh
wrist evidence. A verified flag is reported input provenance, not proof or a
substitute for that authorization. Missing calibration leaves useful proposals
and precise blockers, rather than fabricating a grasp center from a surface.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np
from scipy.optimize import least_squares

from .local_cartesian import ALL, NAMES, RAD_PER_TICK, inward_departure_bounds
from .motion_program import MOVING_LIMIT_MARGIN, _validate_segment_limits, prepare_joint_segment


class TaskSpaceError(ValueError):
    """A failed numerical/geometry proposal; no partial motion was executed."""
    def __init__(self, code, message, **diagnostics):
        super().__init__(message)
        self.code, self.diagnostics = code, diagnostics


def _vector(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be a finite three-vector') from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{label} must be a finite three-vector')
    return result


def _axis(value, label):
    result = _vector(value, label)
    length = np.linalg.norm(result)
    if length < 1e-9:
        raise ValueError(f'{label} cannot be zero')
    return result / length


def _number(value, label, lo, hi, *, open_lower=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value > hi
            or (value <= lo if open_lower else value < lo)):
        raise ValueError(f'{label} outside supported range')
    return float(value)


def _pose(value):
    if (not isinstance(value, Mapping) or set(value) != set(ALL)
            or any(type(v) is not int or not 0 <= v <= 4095 for v in value.values())):
        raise ValueError('Complete measured six-joint integer pose required')
    return {n: value[n] for n in ALL}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def _transform(value):
    result = np.asarray(value, dtype=float)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(result[:3, :3]), 1., atol=1e-7)):
        raise ValueError('TCP transform must be a finite rigid transform')
    return result


def _tcp(tcp, raw):
    """Consume spatial_contract.JawGeometry without a calibration fallback.

    Its +Z is approach; +X is closing. A missing TCP explicitly falls back to
    the old fixed-pad reference (+Y), only for prior-based approach proposals.
    """
    if tcp is None:
        return {'frame': 'gripper_frame_link', 'transform': None,
                'axis_local': np.array([0., 1., 0.]), 'metric_calibrated': False,
                'planning_eligible': False,
                'position_uncertainty_m': None, 'calibration_revision': None,
                'geometry_revision': None,
                'fingerprint': None, 'jaw_raw': raw['gripper'],
                'geometry_scope': 'fixed-pad prior; not the two-jaw grasp center'}
    if (tcp.T_wrist_tcp.parent, tcp.T_wrist_tcp.child) != ('gripper_link', 'grasp_tcp'):
        raise ValueError('TCP must map grasp_tcp into gripper_link')
    transform = _transform(tcp.T_wrist_tcp.matrix)
    if type(tcp.metric_calibrated) is not bool:
        raise ValueError('TCP calibration status must be explicit')
    eligible = getattr(tcp, 'planning_eligible', tcp.metric_calibrated)
    if type(eligible) is not bool:
        raise ValueError('TCP planning eligibility must be explicit')
    jaw_raw = getattr(tcp, 'jaw_raw', getattr(tcp, 'raw', None))
    if jaw_raw != raw['gripper']:
        raise ValueError('TCP calibration must be evaluated at the measured jaw opening')
    uncertainty = (None if tcp.position_uncertainty_m is None else
                   _number(tcp.position_uncertainty_m, 'TCP uncertainty', 0., .1))
    if eligible and uncertainty is None:
        raise ValueError('Planning TCP needs an explicit uncertainty')
    revision, fingerprint = tcp.calibration_revision, tcp.fingerprint
    if not isinstance(revision, str) or not revision or not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError('TCP revision and fingerprint required')
    result = {'frame': 'tcp', 'transform': transform,
            'axis_local': np.array([0., 0., 1.]),
            'metric_calibrated': tcp.metric_calibrated,
            'planning_eligible': eligible,
            'position_uncertainty_m': uncertainty, 'calibration_revision': revision,
            'geometry_revision': getattr(tcp, 'geometry_revision', None),
            'fingerprint': fingerprint, 'jaw_raw': jaw_raw,
            'geometry_scope': 'measured two-pad center at the current opening'}
    if getattr(tcp, 'model_assisted_json', None) is not None:
        public = tcp.to_dict()
        for name in ('accuracy_basis', 'geometry_method', 'external_metric_measurement',
                     'uncertainty_frame', 'aperture_uncertainty_m', 'camera_calibration_fingerprint',
                     'assumptions', 'contact_section', 'validation_kind'):
            result[name] = public[name]
        result['geometry_scope'] = public['scope']
    return result


def _endpoint(model, raw, context):
    if context['transform'] is None:
        return model.tip(raw)
    return model.fk.transforms(raw)['gripper_link'] @ context['transform']


def _axis_error(actual, desired):
    return float(np.degrees(np.arccos(np.clip(actual @ desired, -1., 1.))))


def _model_hash(model):
    # Bind actual FK conventions and onboard ranges; a caller-supplied revision
    # alone cannot notice an altered URDF/model or homing/range configuration.
    joints = [[n, kind, parent, child, origin.tolist(), axis.tolist()]
              for n, kind, parent, child, origin, axis in model.fk.joints]
    return _hash({'joints': joints, 'calibration': model.calibration,
                  'encoder_geometry_convention': '(raw-2047)*2*pi/4096; motor homing already applied',
                  'joint_limits': model.joint_limits})


def _evidence(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError('Evidence must be a mapping')
    result = deepcopy(dict(value))
    # External cameras have no route through this planner. Absence is retained
    # as a blocker for a numerical replay, never silently promoted to wrist.
    if result.get('source') not in (None, 'wrist', 'wrist-only'):
        raise ValueError('Task-space inputs must come from the wrist camera')
    for key in ('geometry_verified', 'target_verified'):
        if key in result and type(result[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    _hash(result)
    return result


def _solve_attempt(model, seed, target, axis, context, origin, position_tolerance_m,
                   axis_tolerance_deg, *, held_goals=None, allow_inward=False):
    held = dict(seed) if held_goals is None else _pose(held_goals)
    # A valid stationary hold near a stop need not invalidate the other axes.
    # Check nominal/held hard limits first, then optimize only joints whose
    # START and END can retain the executor's original moving reserve plus its
    # tracking/command envelope. The first attempt keeps boundary joints fixed.
    # Its explicit fallback can move inward into the original reserve; it may
    # not resend an unchanged hold or propose outward boundary motion.
    try:
        _validate_segment_limits(model, seed, held, seed, held, [], {n: 0 for n in ALL})
        if any(abs(seed[n] - origin[n]) > 360 for n in NAMES):
            raise ValueError('Task pose exceeds fixed program origin')
    except ValueError as exc:
        raise TaskSpaceError('joint_limits', '目前實測位置或保持目標超出關節範圍。') from exc
    free, locked, lo, hi, inward = [], [], [], [], []
    for n in NAMES:
        envelope = max(12, abs(held[n] - seed[n]) + 6)
        lower = math.ceil(max(model.calibration[n]['range_min'] + MOVING_LIMIT_MARGIN + envelope,
                              2047 + model.joint_limits[n][0] / RAD_PER_TICK + envelope,
                              origin[n] - 360, seed[n] - 180))
        upper = math.floor(min(model.calibration[n]['range_max'] - MOVING_LIMIT_MARGIN - envelope,
                               2047 + model.joint_limits[n][1] / RAD_PER_TICK - envelope,
                               origin[n] + 360, seed[n] + 180))
        if envelope <= 60 and lower <= seed[n] <= upper and lower < upper:
            free.append(n); lo.append(lower); hi.append(upper)
        elif allow_inward and envelope <= 60 and lower < upper:
            # This branch is considered only after the unchanged-axis solution
            # failed. Endpoints must enter the FULL old reserve in one bounded
            # segment, while the start remains within measured hard limits.
            candidate = {**seed, n: min(upper, max(lower, seed[n]))}
            try:
                inward_departure_bounds(model, seed, candidate,
                    {j: envelope if j == n else 0 for j in ALL}, [n])
            except ValueError:
                locked.append(n)
            else:
                free.append(n); lo.append(lower); hi.append(upper); inward.append(n)
        else:
            locked.append(n)
    q0 = np.array([seed[n] for n in free], dtype=float)

    def residual(q):
        T = _endpoint(model, {**seed, **dict(zip(free, map(float, q)))}, context)
        actual_axis = T[:3, :3] @ context['axis_local']
        # Unit-vector difference has two local DOF. Unlike cross-product alone,
        # it does not falsely accept an axis pointing 180 degrees backwards.
        return np.r_[T[:3, 3] - target, .08 * (actual_axis - axis), 1e-7 * (q - q0)]

    fit = (least_squares(residual, np.clip(q0, lo, hi), bounds=(lo, hi), max_nfev=180,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10) if free else None)
    q = np.rint(fit.x).astype(int).tolist() if fit is not None else []
    result = {**seed, **dict(zip(free, q))}
    active = [n for n in NAMES if result[n] != seed[n]]
    envelope = {n: max(12 if n in active else 4,
                       abs(held[n] - seed[n]) + (6 if n in active else 0)) for n in NAMES}
    physical = {n: held[n] + result[n] - seed[n] for n in ALL}
    _validate_segment_limits(model, seed, held, result, physical, active, {**envelope, 'gripper': 0},
                             inward_limit_joints=inward)
    T = _endpoint(model, result, context)
    error = float(np.linalg.norm(T[:3, 3] - target))
    angle = _axis_error(T[:3, :3] @ context['axis_local'], axis)
    if (fit is not None and not fit.success) or error > position_tolerance_m or angle > axis_tolerance_deg:
        raise TaskSpaceError('unreachable_task_pose',
                             '這個接近位置或方向超出目前姿勢可達的範圍。',
                             position_error_m=error, axis_error_deg=angle,
                             target_base_m=target.tolist(), held_limit_joints=locked)
    return result, {'position_error_m': error, 'axis_error_deg': angle,
                    'iterations': int(fit.nfev) if fit is not None else 0,
                    'held_limit_joints': locked, 'optimized_joints': free,
                    **({'inward_limit_joints': inward} if inward else {})}


def _solve(model, seed, target, axis, context, origin, position_tolerance_m,
           axis_tolerance_deg, *, held_goals=None):
    """Keep an existing fixed-axis solution; otherwise try a one-way departure."""
    try:
        return _solve_attempt(model, seed, target, axis, context, origin,
            position_tolerance_m, axis_tolerance_deg, held_goals=held_goals)
    except TaskSpaceError as original:
        if original.code != 'unreachable_task_pose' or not original.diagnostics.get('held_limit_joints'):
            raise
        try:
            return _solve_attempt(model, seed, target, axis, context, origin,
                position_tolerance_m, axis_tolerance_deg, held_goals=held_goals, allow_inward=True)
        except (TaskSpaceError, ValueError) as fallback:
            original.diagnostics['inward_fallback'] = {
                'code': getattr(fallback, 'code', 'joint_limits'), 'reason': str(fallback),
                'diagnostics': getattr(fallback, 'diagnostics', {})}
            raise original


def solve_task_pose(model, measured_start, *, target_base_m, approach_axis_base,
                    tcp=None, position_tolerance_m=.002, axis_tolerance_deg=3.):
    """Solve one bounded 3-position + 2-axis proposal, with free axial rotation.

    This API intentionally has no roll/quaternion/orientation-matrix target.
    It does not perform collision screening; use the path planners below for
    the exact executor-compatible segments and path diagnostics.
    """
    raw = _pose(measured_start)
    target = _vector(target_base_m, 'Target')
    axis = _axis(approach_axis_base, 'Approach axis')
    tolerance = _number(position_tolerance_m, 'Position tolerance', 0., .01, open_lower=True)
    angle = _number(axis_tolerance_deg, 'Axis tolerance', 0., 15., open_lower=True)
    context = _tcp(tcp, raw)
    goal, diagnostics = _solve(model, raw, target, axis, context, raw, tolerance, angle)
    return {'proposal_only': True, 'joint_targets': goal,
            'body_target': {n: goal[n] for n in NAMES},
            'constraint': 'position_3d_plus_approach_axis_2d_free_axial_rotation',
            'endpoint_frame': context['frame'], 'diagnostics': diagnostics,
            'collision_screened': False}


def _interpolate_axis(a, b, fraction):
    dot = np.clip(a @ b, -1., 1.)
    if dot > .999999:
        return _axis((1 - fraction) * a + fraction * b, 'Interpolated axis')
    if dot < -.999999:
        basis = np.eye(3)[np.argmin(np.abs(a))]
        tangent = _axis(basis - (basis @ a) * a, 'Axis tangent')
    else:
        tangent = _axis(b - dot * a, 'Axis tangent')
    theta = math.acos(dot) * fraction
    return a * math.cos(theta) + tangent * math.sin(theta)


def _path(model, raw, held, target, axis, context, origin, *, tolerance,
          axis_tolerance, max_path_deviation_m, phase):
    start_T = _endpoint(model, raw, context)
    start_p = start_T[:3, 3]
    start_axis = start_T[:3, :3] @ context['axis_local']
    distance = np.linalg.norm(target - start_p)
    # These are offline numerical waypoints, not cloud queries or physical
    # observation steps. Each checked joint segment is locally continuous.
    count = max(1, math.ceil(distance / .008),
                math.ceil(_axis_error(start_axis, axis) / 8.))
    if count > 64:
        raise TaskSpaceError('path_too_long', '這次規劃的移動距離太長，請先換到較近的姿勢。')
    numerical = [(dict(raw), {})]
    previous = dict(raw)
    held_bias = {n: held[n] - raw[n] for n in ALL}
    for step in range(1, count + 1):
        fraction = step / count
        point = start_p + fraction * (target - start_p)
        direction = _interpolate_axis(start_axis, axis, fraction)
        goal, fit = _solve(model, previous, point, direction, context, origin,
                           tolerance, axis_tolerance,
                           held_goals={n: previous[n] + held_bias[n] for n in ALL})
        numerical.append((goal, fit))
        previous = goal
    # Coalesce the numerical waypoints into the longest valid joint segments.
    # This avoids turning an offline 8 mm solver subdivision into a physical
    # stop/capture/restart every 8 mm. A merge is accepted only after checking
    # the actual joint interpolation (not the desired Cartesian line).
    segments, physical, index, max_deviation = [], dict(held), 0, 0.
    delta = target - start_p
    length_sq = float(delta @ delta)
    while index < count:
        previous = numerical[index][0]
        failure = None
        selected = None
        for end in range(count, index, -1):
            goal, fit = numerical[end]
            if goal == previous:
                if end == count:
                    selected = (end, None, [], 0., fit)
                    break
                continue
            try:
                inward = []
                for n in NAMES:
                    if goal[n] == previous[n]:
                        continue
                    envelope = max(12, abs(physical[n] - previous[n]) + 6)
                    try:
                        inward_departure_bounds(model, previous, goal,
                            {j: envelope if j == n else 0 for j in ALL}, [n])
                    except ValueError:
                        continue
                    inward.append(n)
                segment = prepare_joint_segment(model, previous, physical, goal,
                                                fixed_origin=origin, inward_limit_joints=inward)
                sample_path, segment_deviation = [], 0.
                for sample in segment['sampled_path']:
                    t = sample['fraction']
                    pose = {n: previous[n] + t * (goal[n] - previous[n]) for n in ALL}
                    T = _endpoint(model, pose, context)
                    point = T[:3, 3]
                    along = (np.clip(float((point - start_p) @ delta / length_sq), 0., 1.)
                             if length_sq else 0.)
                    deviation = float(np.linalg.norm(point - (start_p + along * delta)))
                    segment_deviation = max(segment_deviation, deviation)
                    if deviation > max_path_deviation_m:
                        raise TaskSpaceError('interpolated_path_deviation',
                            '關節連續移動時會偏離預計的接近路徑。', deviation_m=deviation)
                    path_fraction = (index + t * (end - index)) / count
                    desired_axis = _interpolate_axis(start_axis, axis, path_fraction)
                    actual_axis = T[:3, :3] @ context['axis_local']
                    if _axis_error(actual_axis, desired_axis) > axis_tolerance:
                        raise TaskSpaceError('interpolated_axis_deviation',
                            '關節連續移動時會偏離預計的夾爪方向。')
                    sample_path.append({'fraction': t, 'position_base_m': point.tolist(),
                                        'approach_axis_base': actual_axis.tolist()})
                selected = (end, segment, sample_path, segment_deviation, fit)
                break
            except ValueError as exc:
                failure = exc
        if selected is None:
            raise failure or TaskSpaceError('no_progress', '目前姿勢無法完成這段移動。')
        end, segment, sample_path, deviation, fit = selected
        if segment is not None:
            segments.append({'kind': phase, 'body_target': {n: numerical[end][0][n] for n in NAMES},
                             'contract': segment, 'endpoint_path': sample_path,
                             'ik_diagnostics': fit,
                             'offline_waypoints_merged': end - index})
            physical = dict(segment['physical_goal_targets'])
            max_deviation = max(max_deviation, deviation)
        index = end
    return segments, numerical[-1][0], physical, max_deviation


def _result(model, raw, held, context, evidence, target, segments, *, blockers,
            diagnostics, input_kind):
    contact_blockers = []
    for flag, code in (('geometry_verified', 'geometry_calibration_unverified'),
                       ('target_verified', 'target_metric_position_unverified')):
        if evidence.get(flag) is not True:
            blockers.append(code)
    for field in ('source', 'frame_id', 'camera_generation', 'geometry_revision',
                  'calibration_revision'):
        if not evidence.get(field):
            blockers.append('missing_' + field)
    if not context['planning_eligible']:
        contact_blockers.append('two_pad_tcp_unverified')
    if context['position_uncertainty_m'] is None:
        contact_blockers.append('tcp_uncertainty_unknown')
    if context['transform'] is not None:
        if context['geometry_revision'] != evidence.get('geometry_revision'):
            blockers.append('tcp_geometry_revision_mismatch')
    # A validated body model knows its fixed-pad reference even before the
    # moving pad/opening is calibrated. It can support an EMPTY-HAND standoff
    # assessment; it must not be promoted to a two-pad contact/grasp center.
    if input_kind != 'observed_surface_standoff' or context['transform'] is not None:
        blockers.extend(contact_blockers)
    tcp_info = {k: v for k, v in context.items() if k not in ('transform', 'axis_local')}
    tcp_info['T_gripper_link_tcp'] = (None if context['transform'] is None
                                     else context['transform'].tolist())
    binding = {'measured_origin': raw, 'held_goals': held, 'target': target,
               'tcp': tcp_info, 'evidence': evidence, 'model_sha256': _model_hash(model)}
    result = {
        'version': 1, 'planner': 'so101_task_space', 'proposal_only': True,
        'input_kind': input_kind, 'constraint': 'position_3d_plus_approach_axis_2d_free_axial_rotation',
        'metric_inputs_verified': not blockers,
        'input_accuracy_basis': context.get('accuracy_basis', 'independently-validated-or-prior-see-binding'),
        'conditional_geometry_estimate': context.get('accuracy_basis') == 'conditional-cad-estimate',
        'ready_for_external_assessment': not blockers and bool(segments),
        'execution_blockers': list(dict.fromkeys(blockers)),
        'contact_blockers': list(dict.fromkeys(contact_blockers)),
        'requires_runtime_authorization': True,
        'not_environment_collision_certificate': True,
        'collision_scope': 'existing robot-mesh/table screen only; external objects, self and held-object checks remain caller responsibilities',
        'tracking_uncertainty_scope': 'per-joint bounds are included in segment contracts; target/TCP position budget does not replace whole-path tracking clearance assessment',
        'jaw_command_included': False, 'cloud_calls_during_planning': 0,
        'segments': segments, 'binding': binding,
        'input_sha256': _hash(binding), 'diagnostics': diagnostics,
    }
    result['plan_sha256'] = _hash(result)
    return result


def plan_surface_approach(model, measured_start, held_goals, *, surface_base_m,
                          target_uncertainty_m, standoff_m=.025,
                          approach_axis_base=None, tcp=None, evidence=None,
                          position_tolerance_m=.002, axis_tolerance_deg=3.,
                          max_path_deviation_m=.003):
    """Plan an empty-hand approach ending BEFORE an observed surface.

    The measured surface is never treated as an object center or grasp target.
    Standoff includes the target/TCP uncertainty and numerical path allowance.
    A TCP/geometry prior still returns a inspectable path and specific blockers.
    No jaw opening, closure, contact, or automatic lift is part of this plan.
    """
    raw, held = _pose(measured_start), _pose(held_goals)
    context, provenance = _tcp(tcp, raw), _evidence(evidence)
    surface = _vector(surface_base_m, 'Observed surface')
    uncertainty = _number(target_uncertainty_m, 'Target uncertainty', 0., .2)
    standoff = _number(standoff_m, 'Standoff', 0., .2, open_lower=True)
    tolerance = _number(position_tolerance_m, 'Position tolerance', 0., .01, open_lower=True)
    axis_tolerance = _number(axis_tolerance_deg, 'Axis tolerance', 0., 15., open_lower=True)
    deviation = _number(max_path_deviation_m, 'Path deviation', tolerance, .02)
    start_T = _endpoint(model, raw, context)
    ray = _axis(surface - start_T[:3, 3], 'Direction to visible surface')
    tool_axis = (_axis(approach_axis_base, 'Approach axis') if approach_axis_base is not None
                 else start_T[:3, :3] @ context['axis_local'])
    # Translation follows the line of sight from the endpoint, independent of
    # tool orientation. The grasp axis is not necessarily the camera ray.
    reserve = standoff + uncertainty + (context['position_uncertainty_m'] or 0.) + deviation
    travel = float(np.linalg.norm(surface - start_T[:3, 3]) - reserve)
    if travel <= tolerance:
        raise TaskSpaceError('already_at_standoff',
                             '已經接近物件表面；下一步需要確認夾爪與接觸位置，不必再向前。',
                             endpoint_to_surface_m=float(np.linalg.norm(surface - start_T[:3, 3])),
                             reserved_standoff_m=reserve)
    target = start_T[:3, 3] + ray * travel
    segments, _, _, actual_deviation = _path(
        model, raw, held, target, tool_axis, context, raw, tolerance=tolerance,
        axis_tolerance=axis_tolerance, max_path_deviation_m=deviation, phase='approach')
    return _result(model, raw, held, context, provenance,
        {'observed_surface_base_m': surface.tolist(), 'endpoint_target_base_m': target.tolist(),
         'target_uncertainty_m': uncertainty, 'standoff_m': standoff,
         'reserved_standoff_m': reserve, 'approach_axis_base': tool_axis.tolist(),
         'surface_is_grasp_center': False}, segments, blockers=[],
        diagnostics={'maximum_path_deviation_m': actual_deviation,
                     'planned_translation_m': travel,
                     'next_checkpoint': 'observe_inner_pads_and_target_before_contact'},
        input_kind='observed_surface_standoff')


def plan_grasp_lift(model, measured_start, held_goals, *, target_base_m,
                     approach_axis_base, target_uncertainty_m, available_clearance_m,
                     tcp=None, evidence=None, pregrasp_distance_m=.025,
                     lift_distance_m=.015, position_tolerance_m=.002,
                     axis_tolerance_deg=3., max_path_deviation_m=.003):
    """Plan pregrasp/contact plus a conditional lift, from a real grasp center.

    ``available_clearance_m`` is the caller's measured one-sided free clearance,
    not an invented object width. The lift is a preview until the existing
    contact checkpoint confirms a grasp and recomputes TCP at measured closure.
    """
    raw, held = _pose(measured_start), _pose(held_goals)
    context, provenance = _tcp(tcp, raw), _evidence(evidence)
    target, axis = _vector(target_base_m, 'Grasp center'), _axis(approach_axis_base, 'Approach axis')
    uncertainty = _number(target_uncertainty_m, 'Target uncertainty', 0., .1)
    clearance = _number(available_clearance_m, 'Available clearance', 0., .2, open_lower=True)
    pregrasp = _number(pregrasp_distance_m, 'Pregrasp distance', 0., .1, open_lower=True)
    lift = _number(lift_distance_m, 'Lift distance', 0., .08, open_lower=True)
    tolerance = _number(position_tolerance_m, 'Position tolerance', 0., .01, open_lower=True)
    axis_tolerance = _number(axis_tolerance_deg, 'Axis tolerance', 0., 15., open_lower=True)
    deviation = _number(max_path_deviation_m, 'Path deviation', tolerance, .02)
    budget = uncertainty + (context['position_uncertainty_m'] or 0.) + deviation
    blockers = ['grasp_clearance_smaller_than_position_budget'] if budget >= clearance else []
    segments, pose, physical, maximum = [], raw, held, 0.
    for phase, point in [('approach', target - axis * pregrasp),
                         ('approach', target),
                         ('verify_lift', target + np.array([0., 0., lift]))]:
        part, pose, physical, error = _path(model, pose, physical, point, axis, context, raw,
            tolerance=tolerance, axis_tolerance=axis_tolerance,
            max_path_deviation_m=deviation, phase=phase)
        for segment in part:
            if phase == 'verify_lift':
                segment['requires_checkpoint'] = 'confirmed_contact_and_updated_tcp_at_measured_jaw'
        if phase == 'verify_lift':
            segments.append({'kind': 'close', 'body_target': None,
                'requires_checkpoint': 'target_between_inner_pads_and_contact_transaction',
                'jaw_command_included': False})
        segments.extend(part)
        maximum = max(maximum, error)
    return _result(model, raw, held, context, provenance,
        {'grasp_center_base_m': target.tolist(), 'target_uncertainty_m': uncertainty,
         'approach_axis_base': axis.tolist(), 'available_clearance_m': clearance,
         'position_budget_m': budget, 'pregrasp_distance_m': pregrasp, 'lift_distance_m': lift},
        segments, blockers=blockers,
        diagnostics={'maximum_path_deviation_m': maximum,
            'lift_is_conditional': True,
            'post_contact_replan_required': True,
            'next_checkpoint': 'inner_pad_contact_then_measured_jaw_tcp_and_held_object_clearance'},
        input_kind='grasp_center_with_clearance')


def plan_lift_from_contact(model, measured_start, held_goals, *, tcp, evidence,
                            contact_context, lift_distance_m=.010,
                            position_tolerance_m=.002, axis_tolerance_deg=3.,
                            max_path_deviation_m=.003):
    """Replan only the verification lift, using the actual post-close TCP.

    ``contact_context`` is a snapshot of the existing visual + low-power
    contact transaction. It must identify the current wrist frame, motor-owner
    state and jaw readback. Obstruction is a contact candidate, never a claim
    that grasp succeeded. The runtime must assess this NEW whole lift path
    using a fresh wrist view, execute it, then verify that the object follows.

    No approach or close is repeated. The known fixed-pad reference is checked
    against the legacy verification limits in addition to the calibrated TCP.
    """
    raw, held = _pose(measured_start), _pose(held_goals)
    provenance = _evidence(evidence)
    if tcp is None:
        raise ValueError('Post-contact lift requires TCP evaluated at measured jaw opening')
    context = _tcp(tcp, raw)
    if not isinstance(contact_context, Mapping):
        raise TaskSpaceError('contact_checkpoint_required', '需要先完成輕合夾爪與接觸確認，才能規劃抬起。')
    contact = deepcopy(dict(contact_context))
    if (contact.get('state') not in ('contact_candidate', 'holding_unverified', 'holding')
            or any(contact.get(key) is not True for key in (
                'obstruction_candidate', 'target_between_inner_pads', 'pwm_owner_active'))
            or type(contact.get('measured_jaw_raw')) is not int
            or contact['measured_jaw_raw'] != raw['gripper']
            or not provenance.get('frame_id') or not provenance.get('camera_generation')
            or contact.get('frame_id') != provenance['frame_id']
            or contact.get('camera_generation') != provenance['camera_generation']):
        raise TaskSpaceError('contact_checkpoint_required',
                             '合爪接觸、目前夾爪開度與腕部影像需要一致，才能規劃抬起。')
    _hash(contact)
    lift = _number(lift_distance_m, 'Verification lift', .002, .015)
    tolerance = _number(position_tolerance_m, 'Position tolerance', 0., .01, open_lower=True)
    axis_tolerance = _number(axis_tolerance_deg, 'Axis tolerance', 0., 15., open_lower=True)
    deviation = _number(max_path_deviation_m, 'Path deviation', tolerance, .02)
    initial = _endpoint(model, raw, context)
    target = initial[:3, 3] + np.array([0., 0., lift])
    direction = initial[:3, :3] @ context['axis_local']
    segments, final, _, maximum = _path(model, raw, held, target, direction, context, raw,
        tolerance=tolerance, axis_tolerance=axis_tolerance,
        max_path_deviation_m=deviation, phase='verify_lift')
    pad_delta = (model.tip(final)[:3, 3] - model.tip(raw)[:3, 3]) * 1000
    # Preserve the existing ObjectProgram's independent fixed-pad check. TCP
    # calibration must not expand the established proof-lift envelope.
    if not (2 <= pad_delta[2] <= 15 and np.linalg.norm(pad_delta[:2]) <= 8):
        raise TaskSpaceError('verification_lift_limits',
                             '這次抬起的路徑超出原有的小幅驗證範圍。',
                             fixed_pad_delta_mm=pad_delta.tolist())
    for step in segments:
        step['requires_checkpoint'] = 'fresh_post_contact_whole_lift_clearance'
    return _result(model, raw, held, context, provenance,
        {'endpoint_origin_base_m': initial[:3, 3].tolist(),
         'endpoint_target_base_m': target.tolist(), 'approach_axis_base': direction.tolist(),
         'lift_distance_m': lift, 'contact_context': contact},
        segments, blockers=[], diagnostics={
            'maximum_path_deviation_m': maximum,
            'fixed_pad_delta_mm': pad_delta.tolist(), 'post_contact_replan_required': False,
            'contact_is_grasp_proof': False, 'requires_new_external_assessment': True,
            'next_checkpoint': 'object_follows_gripper_and_is_clear_of_table'},
        input_kind='post_contact_verification_lift')
