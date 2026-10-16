"""Pure single-pan retreat candidates for a separately assessed waving phase.

Every direction is tied to the wrist-camera mounting model. This is not a
calibrated obstacle-distance estimate and this module performs no motor I/O.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .local_cartesian import ALL, NAMES, RAD_PER_TICK


ACTIVE = 'shoulder_pan'
LOCAL_CORRIDOR = {'minimum_up': -3., 'maximum_up': 3., 'maximum_lateral': 12.,
                  'maximum_rotation_degrees': 3.}
ATTEMPT_LIMITS = {'minimum_up_mm': -5., 'maximum_up_mm': 5., 'maximum_lateral_mm': 40.,
                  'maximum_rotation_degrees': 12., 'maximum_other_joint_drift_ticks': 4}


def _pose(value):
    if not isinstance(value, Mapping) or set(value) != set(ALL):
        raise ValueError('Pan retreat requires all six measured joints')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or int(v) != v or not 0 <= v <= 4095 for v in value.values()):
        raise ValueError('Pan retreat requires finite integer encoder positions')
    return {n: int(value[n]) for n in ALL}


def _movement(model, origin, actual):
    origin = _pose(origin); actual = _pose(actual)
    if any(abs(actual[n] - origin[n]) > 4 for n in ALL if n != ACTIVE):
        raise RuntimeError('側移時其他關節或夾爪開口改變')
    start = model.tip(origin); end = model.tip(actual)
    delta_mm = (end[:3, 3] - start[:3, 3]) * 1000
    angle = float(np.degrees(Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).magnitude()))
    if not np.isfinite(delta_mm).all() or not math.isfinite(angle):
        raise ValueError('Pan retreat model returned nonfinite geometry')
    return delta_mm, angle


def check_retreat_attempt(model, origin, actual):
    """Guard a distinct retreat phase; never reset an earlier raise budget."""
    delta, angle = _movement(model, origin, actual)
    distance = float(np.linalg.norm(delta[:2]))
    if not (-5. <= delta[2] <= 5. and distance <= 40. and angle <= 12.):
        raise RuntimeError('側移已到這次可確認的範圍，先停下來')
    return distance


def check_retreat_corridor(model, origin, actual):
    """Guard the actual local excursion disclosed to the visual assessor."""
    delta, angle = _movement(model, origin, actual)
    if not (-3. <= delta[2] <= 3. and np.linalg.norm(delta[:2]) <= 12. and angle <= 3.):
        raise RuntimeError('側移離開已確認的局部範圍，先停下來')


def retreat_endpoint(model, origin, actual, proposal):
    """Accept stable, correctly directed progress only inside the local arc."""
    origin = _pose(origin); actual = _pose(actual)
    if not isinstance(proposal, Mapping) or proposal.get('motion_kind') != 'pan_retreat':
        raise ValueError('Expected an explicit pan-retreat proposal')
    target = _pose(proposal.get('joint_targets'))
    nominal_ticks = target[ACTIVE] - origin[ACTIVE]
    if not 1 <= abs(nominal_ticks) <= 10 or any(target[n] != origin[n] for n in ALL if n != ACTIVE):
        raise ValueError('Pan retreat must keep every other joint unchanged')
    nominal, angle = _movement(model, origin, target)
    distance = float(np.linalg.norm(nominal[:2]))
    if not (.5 < distance and np.linalg.norm(nominal) <= 5.0000001
            and abs(nominal[2]) <= 1.0000001 and angle <= 1.):
        raise ValueError('Pan-retreat endpoint exceeds nominal arc bounds')
    try:
        check_retreat_corridor(model, origin, actual)
    except RuntimeError:
        return False
    delta, _ = _movement(model, origin, actual)
    progress = float(delta[:2] @ nominal[:2] / distance)
    return bool((actual[ACTIVE] - origin[ACTIVE]) * nominal_ticks > 0
                and progress >= max(.5, distance * .5)
                and min(origin[ACTIVE], target[ACTIVE]) - 12 <= actual[ACTIVE]
                <= max(origin[ACTIVE], target[ACTIVE]) + 12)


def propose_retreat(model, raw, held_goals, origin, sign):
    """Return one <=10-tick pan arc and matching error envelope, or reject."""
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError('Pan-retreat sign must be integer -1 or +1')
    raw = _pose(raw); held_goals = _pose(held_goals); origin = _pose(origin)
    check_retreat_attempt(model, origin, raw)
    envelope = {n: max(30 if n == ACTIVE else 12, abs(held_goals[n] - raw[n]) + 6) for n in NAMES}
    if any(v > 60 for v in envelope.values()):
        raise ValueError('Pan retreat hold compensation exceeds the screened envelope')
    model.validate_raw(raw, uncertainty_ticks=envelope)
    model.validate_raw(held_goals)
    current_check = model.screen_segment(raw, raw, uncertainty_ticks=envelope)
    if not current_check['model_table_pass']:
        raise ValueError('Current pose fails modeled table check for pan retreat')
    start = model.tip(raw); camera_rotation = model.fk.camera(raw)[:3, :3]
    for count in range(10, 0, -1):
        ticks = sign * count
        target = {**raw, ACTIVE: raw[ACTIVE] + ticks}
        try:
            model.validate_raw(target, uncertainty_ticks=envelope)
            angle_rad = (target[ACTIVE] - 2047) * RAD_PER_TICK
            lo, hi = model.joint_limits[ACTIVE]
            if not lo <= angle_rad <= hi:
                continue
            offsets = list(range(0, ticks + sign, sign))
            deltas = []; angles = []
            for offset in offsets:
                pose = {**raw, ACTIVE: raw[ACTIVE] + offset}
                check_retreat_attempt(model, origin, pose)
                end = model.tip(pose)
                delta = end[:3, 3] - start[:3, 3]
                angle = float(np.degrees(Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).magnitude()))
                if np.linalg.norm(delta) > .0050000001 or abs(delta[2]) > .0010000001 or angle > 1.:
                    raise ValueError('Nominal pan arc leaves its bounded lateral region')
                deltas.append([0., 0., 0.] if offset == 0 else delta.tolist())
                angles.append(0. if offset == 0 else angle)
            if np.linalg.norm(deltas[-1][:2]) <= .0005:
                continue
            camera_delta = camera_rotation.T @ np.asarray(deltas[-1])
            if not np.isfinite(camera_delta).all() or abs(camera_delta[0]) <= .0005:
                continue
            clearance = model.screen_segment(raw, target, uncertainty_ticks=envelope)
            if not clearance['model_table_pass']:
                continue
            physical = {n: held_goals[n] + (ticks if n == ACTIVE else 0) for n in NAMES}
            model.validate_raw({**raw, **physical})
        except (ValueError, RuntimeError):
            continue
        delta_ticks = {n: ticks if n == ACTIVE else 0 for n in NAMES}
        proposal = {
            'proposal_only': True, 'motion_kind': 'pan_retreat',
            'endpoint_frame': 'gripper_frame_link (fixed pad, not jaw center)',
            'measured_origin': dict(raw), 'joint_targets': target,
            'body_joint_targets': {n: target[n] for n in NAMES},
            'active_joint_targets': {ACTIVE: target[ACTIVE]},
            'physical_goal_targets': physical, 'delta_ticks': delta_ticks,
            'requested_delta_m': list(deltas[-1]), 'predicted_delta_m': list(deltas[-1]),
            'camera_relative_delta_m': camera_delta.tolist(),
            'image_side': 'left' if camera_delta[0] < 0 else 'right',
            'full_orientation_error_deg': angles[-1], 'predicted_position_error_mm': 0.,
            'clearance': clearance,
            'retreat_path': {
                'sampled_pan_offsets_ticks': offsets, 'sampled_delta_m': deltas,
                'sampled_orientation_deg': angles,
                'actual_tip_corridor_mm': dict(LOCAL_CORRIDOR),
                'attempt_limits': dict(ATTEMPT_LIMITS),
            },
            'requires_new_observation_after_step': True,
            'not_environment_collision_certificate': True,
            'camera_direction_from_mounting_prior': True,
        }
        return proposal, envelope
    raise RuntimeError('這個方向尚未找到可確認的側移弧線')
