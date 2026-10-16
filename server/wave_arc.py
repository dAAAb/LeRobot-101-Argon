"""Pure, explicit elbow-arc proposals for an empty-hand greeting raise.

This is a planned change of orientation, not Cartesian drift compensation.
External-object clearance and all motor I/O belong to the caller.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .local_cartesian import ALL, NAMES, RAD_PER_TICK


ACTIVE = 'elbow_flex'
NOMINAL_LIMITS = {'maximum_distance_mm': 8., 'maximum_lateral_mm': 2.,
                  'maximum_rotation_degrees': 2., 'maximum_ticks': 20}
LOCAL_CORRIDOR = {'minimum_up': -1., 'maximum_up': 12., 'maximum_lateral': 3.,
                  'maximum_rotation_degrees': 3.}
MACRO_NOMINAL_LIMITS = {'maximum_distance_mm': 20., 'maximum_lateral_mm': 6.,
                        'maximum_rotation_degrees': 5., 'maximum_ticks': 50}
MACRO_LOCAL_CORRIDOR = {'minimum_up': -1., 'maximum_up': 26., 'maximum_lateral': 8.,
                        'maximum_rotation_degrees': 6.}
ATTEMPT_LIMITS = {'minimum_up_mm': -1., 'maximum_up_mm': 60., 'maximum_lateral_mm': 15.,
                  'maximum_rotation_degrees': 15., 'maximum_other_joint_drift_ticks': 4}


def arc_profile(motion_kind='elbow_raise'):
    """Fixed controller contracts; a model cannot provide its own limits."""
    if motion_kind == 'elbow_raise':
        return dict(NOMINAL_LIMITS), dict(LOCAL_CORRIDOR)
    if motion_kind == 'elbow_raise_macro':
        return dict(MACRO_NOMINAL_LIMITS), dict(MACRO_LOCAL_CORRIDOR)
    raise ValueError('Unknown elbow-raise motion contract')


def _pose(value):
    if not isinstance(value, Mapping) or set(value) != set(ALL):
        raise ValueError('Elbow raise requires all six measured joints')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or int(v) != v or not 0 <= v <= 4095 for v in value.values()):
        raise ValueError('Elbow raise requires finite integer encoder positions')
    return {n: int(value[n]) for n in ALL}


def _movement(model, origin, actual):
    origin = _pose(origin); actual = _pose(actual)
    if any(abs(actual[n] - origin[n]) > 4 for n in ALL if n != ACTIVE):
        raise RuntimeError('抬手時其他關節或夾爪開口改變')
    start = model.tip(origin); end = model.tip(actual)
    delta_mm = (end[:3, 3] - start[:3, 3]) * 1000
    angle = float(np.degrees(Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).magnitude()))
    if not np.isfinite(delta_mm).all() or not math.isfinite(angle):
        raise ValueError('Elbow raise model returned nonfinite geometry')
    return delta_mm, angle


def check_arc_attempt(model, origin, actual):
    """Enforce the separate planned-arc attempt; return model-derived rise mm."""
    delta, angle = _movement(model, origin, actual)
    if not (ATTEMPT_LIMITS['minimum_up_mm'] <= delta[2] <= ATTEMPT_LIMITS['maximum_up_mm']
            and np.linalg.norm(delta[:2]) <= ATTEMPT_LIMITS['maximum_lateral_mm']
            and angle <= ATTEMPT_LIMITS['maximum_rotation_degrees']):
        raise RuntimeError('抬手弧線已到這次可確認的範圍，先停下來')
    return max(0., float(delta[2]))


def check_arc_corridor(model, origin, actual, *, motion_kind='elbow_raise'):
    """Measured local excursion of the specific visually assessed arc."""
    _, corridor = arc_profile(motion_kind)
    delta, angle = _movement(model, origin, actual)
    if not (corridor['minimum_up'] <= delta[2] <= corridor['maximum_up']
            and np.linalg.norm(delta[:2]) <= corridor['maximum_lateral']
            and angle <= corridor['maximum_rotation_degrees']):
        raise RuntimeError('抬手弧線離開已確認的局部範圍，先停下來')


def arc_endpoint(model, origin, actual, proposal):
    """Useful measured progress inside the assessed arc; no grasp assertion."""
    origin = _pose(origin); actual = _pose(actual)
    if not isinstance(proposal, Mapping):
        raise ValueError('Expected an explicit elbow-raise proposal')
    kind = proposal.get('motion_kind')
    nominal_limits, _ = arc_profile(kind)
    target = _pose(proposal.get('joint_targets'))
    nominal_ticks = target[ACTIVE] - origin[ACTIVE]
    if not 1 <= abs(nominal_ticks) <= nominal_limits['maximum_ticks'] or any(target[n] != origin[n] for n in ALL if n != ACTIVE):
        raise ValueError('Elbow-raise endpoint must keep every other joint unchanged')
    nominal, nominal_angle = _movement(model, origin, target)
    if not (.5 < nominal[2] and np.linalg.norm(nominal) <= nominal_limits['maximum_distance_mm'] + 1e-7
            and np.linalg.norm(nominal[:2]) <= nominal_limits['maximum_lateral_mm'] + 1e-7
            and nominal_angle <= nominal_limits['maximum_rotation_degrees']):
        raise ValueError('Elbow-raise endpoint exceeds nominal arc bounds')
    try:
        check_arc_corridor(model, origin, actual, motion_kind=kind)
    except RuntimeError:
        return False
    delta, _ = _movement(model, origin, actual)
    return bool((actual[ACTIVE] - origin[ACTIVE]) * nominal_ticks > 0
                and delta[2] >= max(.5, nominal[2] * .5)
                and min(origin[ACTIVE], target[ACTIVE]) - 12 <= actual[ACTIVE]
                <= max(origin[ACTIVE], target[ACTIVE]) + 12)


def propose_raise(model, raw, held_goals, attempt_origin, *, max_distance_mm=NOMINAL_LIMITS['maximum_distance_mm'],
                  motion_kind='elbow_raise'):
    """Choose one bounded upward elbow arc; no resends or motor writes.

    Nominal sample geometry is disclosed to the visual assessor. The controller
    must enforce both the returned command box and tighter live pose guards.
    """
    nominal_limits, corridor = arc_profile(motion_kind)
    if (isinstance(max_distance_mm, bool) or not isinstance(max_distance_mm, (int, float))
            or not math.isfinite(max_distance_mm) or not 0 < max_distance_mm <= nominal_limits['maximum_distance_mm']):
        raise ValueError(f"Elbow raise distance cap must be positive and at most {nominal_limits['maximum_distance_mm']:g} mm")
    raw = _pose(raw); held_goals = _pose(held_goals); attempt_origin = _pose(attempt_origin)
    check_arc_attempt(model, attempt_origin, raw)
    attempt_delta, _ = _movement(model, attempt_origin, raw)
    # Reserve the entire visually assessed local corridor, not just a smaller
    # nominal endpoint. Stiction release may advance beyond that endpoint.
    if (np.linalg.norm(attempt_delta[:2]) + corridor['maximum_lateral']
            > ATTEMPT_LIMITS['maximum_lateral_mm']):
        raise RuntimeError('這次上抬的側向範圍不足以保留完整局部走廊，先評估退讓')
    if motion_kind == 'elbow_raise_macro' and (
            max(0., attempt_delta[2]) + corridor['maximum_up'] > ATTEMPT_LIMITS['maximum_up_mm']
            or _movement(model, attempt_origin, raw)[1] + corridor['maximum_rotation_degrees']
            > ATTEMPT_LIMITS['maximum_rotation_degrees']):
        raise RuntimeError('這次抬手剩餘範圍不足以容納整段弧線')
    envelope = {n: max(30 if n == ACTIVE else 12, abs(held_goals[n] - raw[n]) + 6) for n in NAMES}
    if any(v > 60 for v in envelope.values()):
        raise ValueError('Elbow raise hold compensation exceeds the screened envelope')
    model.validate_raw(raw, uncertainty_ticks=envelope)
    model.validate_raw(held_goals)
    current_check = model.screen_segment(raw, raw, uncertainty_ticks=envelope)
    if not current_check['model_table_pass']:
        raise ValueError('Current pose fails modeled table check for elbow raise')
    start = model.tip(raw)
    for count in range(nominal_limits['maximum_ticks'], 0, -1):
        for sign in (-1, 1):
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
                    check_arc_attempt(model, attempt_origin, pose)
                    end = model.tip(pose)
                    delta = end[:3, 3] - start[:3, 3]
                    angle = float(np.degrees(Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).magnitude()))
                    if (delta[2] < -1e-12 or np.linalg.norm(delta) * 1000 > max_distance_mm + 1e-7
                            or np.linalg.norm(delta[:2]) * 1000 > nominal_limits['maximum_lateral_mm'] + 1e-7
                            or angle > nominal_limits['maximum_rotation_degrees']):
                        raise ValueError('Nominal elbow arc leaves its bounded upward region')
                    # The zero-offset pose is exactly the origin; avoid tiny
                    # floating-point rotation residue in the visual contract.
                    deltas.append([0., 0., 0.] if offset == 0 else delta.tolist())
                    angles.append(0. if offset == 0 else angle)
                if deltas[-1][2] <= .0005:
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
                'proposal_only': True, 'motion_kind': motion_kind,
                'endpoint_frame': 'gripper_frame_link (fixed pad, not jaw center)',
                'measured_origin': dict(raw), 'joint_targets': target,
                'body_joint_targets': {n: target[n] for n in NAMES},
                'active_joint_targets': {ACTIVE: target[ACTIVE]},
                'physical_goal_targets': physical, 'delta_ticks': delta_ticks,
                'requested_delta_m': list(deltas[-1]), 'predicted_delta_m': list(deltas[-1]),
                'full_orientation_error_deg': angles[-1], 'predicted_position_error_mm': 0.,
                'clearance': clearance,
                'arc_path': {
                    'sampled_elbow_offsets_ticks': offsets, 'sampled_delta_m': deltas,
                    'sampled_orientation_deg': angles,
                    'actual_tip_corridor_mm': dict(corridor),
                    'attempt_limits': dict(ATTEMPT_LIMITS),
                },
                'requires_new_observation_after_step': True,
                'not_environment_collision_certificate': True,
            }
            return proposal, envelope
    raise RuntimeError('這個姿勢尚未找到可確認的上抬弧線')
