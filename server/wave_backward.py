"""Pure, separately assessed Cartesian retreat behind the open gripper.

The camera direction uses the mounting prior, not calibrated obstacle depth.
Every asynchronous three-axis corner is screened against the modeled table;
external clearance still requires a fresh wrist observation before each move.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .local_cartesian import ALL, NAMES


COORDINATED_JOINTS = ('shoulder_lift', 'elbow_flex', 'wrist_flex')
FIXED_JOINTS = ('shoulder_pan', 'wrist_roll', 'gripper')
LOCAL_CORRIDOR = {'minimum_back': -1., 'maximum_back': 12.,
                  'maximum_transverse': 12., 'maximum_rotation_degrees': 3.}
ATTEMPT_LIMITS = {'minimum_back_mm': -1., 'maximum_back_mm': 40.,
                  'maximum_transverse_mm': 12., 'maximum_rotation_degrees': 3.,
                  'maximum_fixed_joint_drift_ticks': 4}


def _pose(value):
    if not isinstance(value, Mapping) or set(value) != set(ALL):
        raise ValueError('Backward retreat requires all six measured joints')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or int(v) != v or not 0 <= v <= 4095 for v in value.values()):
        raise ValueError('Backward retreat requires finite integer encoder positions')
    return {n: int(value[n]) for n in ALL}


def _angle(a, b):
    return float(np.degrees(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()))


def _direction(model, raw):
    direction = np.asarray(model.fk.transforms(raw)['gripper_link'][:3, 2], dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1., atol=1e-9):
        raise ValueError('Backward retreat requires a finite unit tool direction')
    return direction


def _movement(model, origin, actual):
    origin = _pose(origin); actual = _pose(actual)
    if any(abs(actual[n] - origin[n]) > 4 for n in FIXED_JOINTS):
        raise RuntimeError('後退時底座方向、手腕旋轉或夾爪開口改變')
    start = model.tip(origin); end = model.tip(actual)
    delta = (end[:3, 3] - start[:3, 3]) * 1000
    direction = _direction(model, origin)
    along = float(delta @ direction)
    transverse = float(np.linalg.norm(delta - along * direction))
    angle = _angle(end, start)
    if not np.isfinite(delta).all() or not all(math.isfinite(v) for v in (along, transverse, angle)):
        raise ValueError('Backward retreat model returned nonfinite geometry')
    return delta, along, transverse, angle


def check_backward_attempt(model, origin, actual):
    """Guard one distinct backward phase from its unchanged measured origin."""
    _, along, transverse, angle = _movement(model, origin, actual)
    if not (-1. <= along <= 40. and transverse <= 12. and angle <= 3.):
        raise RuntimeError('後退已到這次可確認的範圍，先停下來')
    return max(0., along)


def check_backward_corridor(model, origin, actual):
    """Guard the entire local excursion disclosed to the wrist assessor."""
    _, along, transverse, angle = _movement(model, origin, actual)
    if not (-1. <= along <= 12. and transverse <= 12. and angle <= 3.):
        raise RuntimeError('後退離開已確認的局部範圍，先停下來')


def backward_endpoint(model, origin, actual, proposal, *, attempt_origin=None):
    """Require the original Cartesian endpoint accuracy, never arc tolerance."""
    origin = _pose(origin); actual = _pose(actual)
    if not isinstance(proposal, Mapping) or proposal.get('motion_kind') != 'cartesian_backward':
        raise ValueError('Expected a Cartesian-backward proposal')
    target = _pose(proposal.get('joint_targets'))
    if any(target[n] != origin[n] for n in FIXED_JOINTS) or any(abs(target[n] - origin[n]) > 30 for n in COORDINATED_JOINTS):
        raise ValueError('Backward retreat must preserve pan, roll and jaw within the IK trust region')
    request = np.asarray(proposal.get('requested_delta_m'), dtype=float)
    direction = _direction(model, origin)
    distance = float(np.linalg.norm(request))
    if (request.shape != (3,) or not np.isfinite(request).all() or not .0005 <= distance <= .0030000001
            or not np.allclose(request, direction * distance, atol=1e-9, rtol=0)):
        raise ValueError('Backward retreat request must follow the tool backward ray')
    try:
        check_backward_corridor(model, origin, actual)
        if attempt_origin is not None:
            check_backward_attempt(model, attempt_origin, actual)
    except RuntimeError:
        return False
    start = model.tip(origin); endpoint = model.tip(actual); goal = model.tip(target)
    progress = float((endpoint[:3, 3] - start[:3, 3]) @ direction)
    reference_ok = attempt_origin is None or _angle(endpoint, model.tip(_pose(attempt_origin))) <= 1.
    return bool(np.linalg.norm(endpoint[:3, 3] - goal[:3, 3]) <= .001
                and _angle(endpoint, goal) <= 1. and reference_ok
                and progress >= max(.0005, distance * .5)
                and all(min(origin[n], target[n]) - 12 <= actual[n] <= max(origin[n], target[n]) + 12
                        for n in COORDINATED_JOINTS))


def propose_backward(model, raw, held_goals, origin):
    """Propose <=3 mm behind the fingers, with all asynchronous corners screened."""
    raw = _pose(raw); held_goals = _pose(held_goals); origin = _pose(origin)
    check_backward_attempt(model, origin, raw)
    envelope = {n: max(30, abs(held_goals[n] - raw[n]) + 6) for n in NAMES}
    if any(v > 60 for v in envelope.values()):
        raise ValueError('Backward retreat hold compensation exceeds the screened envelope')
    model.validate_raw(raw, uncertainty_ticks=envelope)
    model.validate_raw(held_goals)
    direction = _direction(model, raw)
    proposal = model.propose_adaptive(raw, direction * .003, held_goals=held_goals,
                                      uncertainty_ticks=envelope, orientation_origin=origin)
    target = _pose(proposal['joint_targets'])
    if any(target[n] != raw[n] for n in FIXED_JOINTS):
        raise ValueError('Backward IK would move a fixed pan, roll or jaw axis')
    if (proposal['predicted_position_error_mm'] > 1. or proposal['full_orientation_error_deg'] > 2.
            or _angle(model.tip(target), model.tip(origin)) > 1.):
        raise ValueError('Backward IK does not preserve the original Cartesian accuracy')
    delta_ticks = {n: target[n] - raw[n] for n in NAMES}
    if any(abs(v) > 30 for v in delta_ticks.values()):
        raise ValueError('Backward IK exceeds the original joint trust region')
    start = model.tip(raw)
    predicted = model.tip(target)[:3, 3] - start[:3, 3]
    camera_delta = model.fk.camera(raw)[:3, :3].T @ predicted
    if not np.isfinite(camera_delta).all() or camera_delta[2] >= 0:
        raise ValueError('Backward IK does not move behind the wrist camera in the mounting model')
    offsets = list(itertools.product(*[(0, delta_ticks[n]) for n in COORDINATED_JOINTS]))
    deltas = []; angles = []; checks = []
    for offset in offsets:
        corner = {**raw, **{n: raw[n] + v for n, v in zip(COORDINATED_JOINTS, offset)}}
        model.validate_raw(corner, uncertainty_ticks=envelope)
        check_backward_corridor(model, raw, corner)
        check_backward_attempt(model, origin, corner)
        end = model.tip(corner)
        deltas.append((end[:3, 3] - start[:3, 3]).tolist())
        angles.append(0. if corner == raw else _angle(end, start))
        for a, b in ((raw, corner), (corner, target)):
            check = model.screen_segment(a, b, uncertainty_ticks=envelope)
            if not check['model_table_pass']:
                raise ValueError('Asynchronous backward path fails the modeled table check')
            checks.append(check)
    proposal.update(
        motion_kind='cartesian_backward', delta_ticks=delta_ticks,
        active_joint_targets={n: target[n] for n in COORDINATED_JOINTS if delta_ticks[n]},
        camera_relative_delta_m=camera_delta.tolist(), image_side='backward',
        camera_direction_from_mounting_prior=True,
        asynchronous_clearance=checks,
        backward_path={
            'backward_direction_base': direction.tolist(),
            'coordinated_joints': list(COORDINATED_JOINTS),
            'sampled_corner_offsets_ticks': [list(v) for v in offsets],
            'sampled_corner_delta_m': deltas, 'sampled_corner_orientation_deg': angles,
            'actual_tip_corridor_mm': dict(LOCAL_CORRIDOR),
            'attempt_limits': dict(ATTEMPT_LIMITS),
        },
        not_environment_collision_certificate=True,
        requires_new_observation_after_step=True,
    )
    return proposal, envelope
