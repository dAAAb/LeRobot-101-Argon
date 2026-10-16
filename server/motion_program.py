"""Screened joint segments for locally executed, finite robot programs.

No model inference or external camera data enters this module. The caller must
assess external-object clearance for the complete program before dispatch.
Camera freshness below is a transport watchdog, not visual collision detection.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import time

import numpy as np

from .gripper_power import SessionCancelled
from .local_cartesian import ALL, NAMES, RAD_PER_TICK, tracking_envelope, inward_departure_bounds
from .local_feedback import adopt_goal_anchor, check_bounds

MAX_SEGMENT_TICKS = 180
MAX_SEGMENT_TIP_MM = 80.
MAX_PROGRAM_SECONDS = 30.
FIXED_AXIS_TOLERANCE = 4
MAX_STATIONARY_TOLERANCE = 12
JAW_TOLERANCE = 4
INITIAL_TOLERANCE = 3
MOVING_LIMIT_MARGIN = 12
PHASE_STALL_SECONDS = 2.
# Explicit engineering-only workspace survey. This does not change the normal
# 360-tick program limit. The survey keeps one original measured origin for its
# entire lifetime; callers cannot provide arbitrary larger bounds.
WORKSPACE_ORIGIN_PROFILE = 'engineering_workspace_calibration_v1'
WORKSPACE_ORIGIN_DELTA_BOUNDS = {
    'shoulder_pan': (-300, 300), 'shoulder_lift': (-100, 1250),
    'elbow_flex': (-520, 24), 'wrist_flex': (-1050, 240),
    'wrist_roll': (-300, 300),
}
JOINT_LABELS = {'shoulder_pan': '肩部轉向', 'shoulder_lift': '肩部抬升',
                'elbow_flex': '手肘', 'wrist_flex': '手腕俯仰', 'wrist_roll': '手腕旋轉'}


def _pose(value):
    if not isinstance(value, Mapping) or set(value) != set(ALL):
        raise ValueError('動作段需要完整六軸位置')
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or int(v) != v or not 0 <= v <= 4095
           for v in value.values()):
        raise ValueError('動作位置必須是有限整數刻度')
    return {n: int(value[n]) for n in ALL}


def _fingerprint(segment):
    value = {k: v for k, v in segment.items() if k != 'contract_sha256'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


class MotionPathViolation(RuntimeError):
    """The exact rejected pose, before a stop/hold changes motor readings."""
    def __init__(self, message, actual, origin, target, envelope, *, source, reason):
        super().__init__(message)
        intervals = {}
        lower, upper = 0., 1.
        for n in NAMES:
            delta = target[n] - origin[n]
            row = {'delta_ticks': delta, 'offset_ticks': actual[n] - origin[n],
                   'envelope_ticks': envelope[n]}
            if delta:
                a, b = sorted(((actual[n] - origin[n] - envelope[n]) / delta,
                               (actual[n] - origin[n] + envelope[n]) / delta))
                row['fraction_interval'] = [a, b]
                lower, upper = max(lower, a), min(upper, b)
            else:
                row['stationary_drift_ticks'] = abs(actual[n] - origin[n])
            intervals[n] = row
        self.path_diagnostic = {'reason': reason, 'check_source': source,
            'rejected_pose': dict(actual), 'measured_origin': dict(origin),
            'joint_targets': dict(target), 'joint_intervals': intervals,
            'common_fraction_interval': [lower, upper],
            'fraction_gap': max(0., lower - upper)}


def _tube(actual, origin, target, envelope, *, source='unspecified'):
    """Every axis must be near the SAME fraction of the screened line.

    A per-axis min/max box alone would admit unscreened asynchronous paths.
    Intersect the fraction intervals instead, including stationary axes.
    """
    lower, upper = 0., 1.
    for n in NAMES:
        delta = target[n] - origin[n]
        if not delta:
            drift = abs(actual[n] - origin[n])
            if drift > envelope[n]:
                raise MotionPathViolation(
                    f'{JOINT_LABELS[n]}偏離保持位置 {drift * 360 / 4096:.2f}°（{drift:g} 刻度），'
                    f'超過這段動作容許的 {envelope[n] * 360 / 4096:.2f}°（{envelope[n]:g} 刻度）。',
                    actual, origin, target, envelope, source=source, reason='stationary_drift')
            continue
        a = (actual[n] - origin[n] - envelope[n]) / delta
        b = (actual[n] - origin[n] + envelope[n]) / delta
        lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
        if lower > upper + 1e-12:
            raise MotionPathViolation('關節不同步，已離開整段確認的路徑',
                actual, origin, target, envelope, source=source, reason='no_common_path_fraction')
    return lower, upper


def _partial_observation_progress(actual, origin, targets):
    """Useful measured survey motion, not endpoint or depth accuracy.

    The existing path/limit guards still apply separately. Stationary axes
    have no progress requirement; no commanded axis may end behind its start.
    """
    progress = [(actual[n] - origin[n]) * (1 if value > origin[n] else -1)
                for n, value in targets.items() if value != origin[n]]
    return bool(progress) and min(progress) >= 0 and max(progress) >= 8


class SegmentPhaseGovernor:
    """Limit command progress to the already screened tube, never widen it.

    The driver writes axes sequentially. Each prospective single-axis goal
    must therefore coexist with the other axes' CURRENT measured positions.
    Elapsed wall time is only the upper bound, not permission to jump ahead.
    """
    def __init__(self, segment):
        self.segment = segment
        self.origin = segment['measured_origin']
        self.target = segment['joint_targets']
        self.held = segment['held_goals']
        self.delta = segment['delta_ticks']
        self.envelope = segment['uncertainty_ticks']
        self.tick_fraction = 1. / max(abs(v) for v in self.delta.values())
        self.last_progress = None
        self.progress_mark = 0.

    def __call__(self, actual, requested, previous):
        _tube(actual, self.origin, self.target, self.segment['actual_tracking_envelope_ticks'],
              source='phase_governor_measured')
        intervals = {}
        for n in NAMES:
            d = self.delta[n]
            if d:
                intervals[n] = sorted(((actual[n] - self.origin[n] - self.envelope[n]) / d,
                                       (actual[n] - self.origin[n] + self.envelope[n]) / d))
        lower, upper = 0., 1.
        for n, d in self.delta.items():
            if not d:
                continue
            lo = max([0., *[bounds[0] for j, bounds in intervals.items() if j != n]])
            hi = min([1., *[bounds[1] for j, bounds in intervals.items() if j != n]])
            # Reserve half a tick so integer rounding cannot cross a face of
            # the ORIGINAL envelope. Runtime screens the exact rounded pose.
            reserve = self.envelope[n] - .500001
            low_goal = min(self.origin[n] + d * lo, self.origin[n] + d * hi) - reserve
            high_goal = max(self.origin[n] + d * lo, self.origin[n] + d * hi) + reserve
            a, b = sorted(((low_goal - self.held[n]) / d, (high_goal - self.held[n]) / d))
            lower, upper = max(lower, a), min(upper, b)
        phase = min(requested, upper)
        allowed = phase >= max(lower, previous) - 1e-12
        now = time.monotonic()
        actual_progress = sum((actual[n] - self.origin[n]) * self.delta[n] for n in NAMES) / sum(
            d * d for d in self.delta.values())
        # Only progress already observed or actually commanded resets the
        # wait budget, not a proposed phase that a later prewrite check rejects.
        progress = max(previous, actual_progress)
        if self.last_progress is None or progress >= self.progress_mark + self.tick_fraction:
            self.last_progress, self.progress_mark = now, progress
        elif now - self.last_progress >= PHASE_STALL_SECONDS:
            raise RuntimeError('馬達沒有繼續跟上這段路徑，已停止保持姿勢。')
        return max(previous, phase) if allowed else None


def _fixed_origin(pose, origin, profile=None):
    if profile is not None:
        if profile != WORKSPACE_ORIGIN_PROFILE or origin is None:
            raise ValueError('未知或缺少固定起點的工程工作範圍')
        if any(not low <= pose[n] - origin[n] <= high
               for n, (low, high) in WORKSPACE_ORIGIN_DELTA_BOUNDS.items()):
            raise ValueError('動作已超出這份工程協定原始起點的關節範圍')
        return
    if origin is not None and any(abs(pose[n] - origin[n]) > 360 for n in NAMES):
        raise ValueError('動作已超出這次計畫的關節範圍')


def _validate_segment_limits(model, raw, held, target, physical, active, envelope, *, inward_limit_joints=()):
    """Keep the moving-axis reserve without treating fixed holds as commands.

    Stationary axes are never rewritten by the executor. Their actual measured
    interval is clipped to onboard/URDF limits; nominal and held positions must
    fit those limits. Geometry still uses the full symmetric box. Moving axes retain
    the additional 12-tick reserve around BOTH the declared box and the actual
    physical start/end goals. An explicit inward departure exempts only its
    hard-valid start; its endpoint restores that same reserve. Jaw movement is
    owned by a separate transaction.
    """
    inward = inward_departure_bounds(model, raw, target, envelope, inward_limit_joints)
    if not set(inward) <= set(active):
        raise ValueError('向內退開的關節必須有位移')
    limit_envelope = {n: envelope[n] if n in active else 0 for n in ALL}
    for pose in (raw, target):
        checked_envelope = {n: 0 if pose is raw and n in inward else limit_envelope[n] for n in ALL}
        model.validate_raw(pose, margin_ticks=0, uncertainty_ticks=checked_envelope)
        for n in active:
            if pose is raw and n in inward:
                continue
            c = model.calibration[n]
            if not (c['range_min'] + MOVING_LIMIT_MARGIN <= pose[n] - envelope[n]
                    <= pose[n] + envelope[n] <= c['range_max'] - MOVING_LIMIT_MARGIN):
                raise ValueError('移動關節未保留原有行程餘量：' + n)
        if hasattr(model, 'joint_limits'):
            for n in NAMES:
                lo, hi = model.joint_limits[n]
                if not lo <= (pose[n] - checked_envelope[n] - 2047) * RAD_PER_TICK <= (
                        pose[n] + checked_envelope[n] - 2047) * RAD_PER_TICK <= hi:
                    raise ValueError('動作路徑超出模型關節範圍：' + n)
    for pose in (held, physical):
        model.validate_raw(pose, margin_ticks=0)
        for n in active:
            if pose is held and n in inward:
                continue
            c = model.calibration[n]
            if not c['range_min'] + MOVING_LIMIT_MARGIN <= pose[n] <= c['range_max'] - MOVING_LIMIT_MARGIN:
                raise ValueError('移動關節保持目標未保留原有行程餘量：' + n)
        if hasattr(model, 'joint_limits'):
            for n in NAMES:
                lo, hi = model.joint_limits[n]
                if not lo <= (pose[n] - 2047) * RAD_PER_TICK <= hi:
                    raise ValueError('保持目標超出模型關節範圍：' + n)
    for n, departure in inward.items():
        if departure['direction'] * (physical[n] - held[n]) <= 0:
            raise ValueError('向內退開不可把保持目標往外送：' + n)
    return inward


def _segment_bounds(model, raw, target, active, envelope, actual_envelope, *, inward_limit_joints=()):
    """Clip held/inward runtime bounds to hard limits; never shrink mesh uncertainty."""
    raw_bounds = {n: [min(raw[n], target[n]) - envelope[n],
                      max(raw[n], target[n]) + envelope[n]] for n in NAMES}
    measured_bounds = {n: [min(raw[n], target[n]) - actual_envelope[n],
                           max(raw[n], target[n]) + actual_envelope[n]] for n in NAMES}
    stationary_limits = {}
    for n in NAMES:
        if n in active and n not in inward_limit_joints:
            continue
        c = model.calibration[n]
        lo, hi = c['range_min'], c['range_max']
        if hasattr(model, 'joint_limits'):
            lower, upper = model.joint_limits[n]
            lo = max(lo, math.ceil(2047 + lower / RAD_PER_TICK - 1e-9))
            hi = min(hi, math.floor(2047 + upper / RAD_PER_TICK + 1e-9))
        if n not in active:
            stationary_limits[n] = [lo, hi]
        for bounds in (raw_bounds, measured_bounds):
            bounds[n] = [max(bounds[n][0], lo), min(bounds[n][1], hi)]
            if not bounds[n][0] <= raw[n] <= bounds[n][1]:
                raise ValueError('固定關節位置超出目前行程範圍：' + n)
    return raw_bounds, measured_bounds, stationary_limits


def prepare_joint_segment(model, raw, held, target, *, endpoint_tolerance_ticks=10,
                          stationary_tolerance_ticks=FIXED_AXIS_TOLERANCE,
                          approved_envelope=None, fixed_origin=None, inward_limit_joints=(),
                          origin_limit_profile=None):
    """Return the exact whole-segment geometry to show the visual assessor.

    ``raw`` may be a future nominal program boundary. ``held`` then contains
    the predicted physical goals at that boundary, with the ORIGINAL measured
    load compensation retained. The executor never moves the intended target
    to a partially reached measured endpoint.
    """
    raw, held, target = _pose(raw), _pose(held), _pose(target)
    fixed_origin = None if fixed_origin is None else _pose(fixed_origin)
    if type(endpoint_tolerance_ticks) is not int or not 1 <= endpoint_tolerance_ticks <= 20:
        raise ValueError('動作段終點容許誤差必須是 1 至 20 刻度')
    if type(stationary_tolerance_ticks) is not int or not 1 <= stationary_tolerance_ticks <= MAX_STATIONARY_TOLERANCE:
        raise ValueError('固定關節容許偏移必須是 1 至 12 刻度')
    if target['gripper'] != raw['gripper']:
        raise ValueError('身體動作段不可改變夾爪，請使用接觸或放置流程')
    delta = {n: target[n] - raw[n] for n in NAMES}
    if not any(delta.values()) or max(map(abs, delta.values())) > MAX_SEGMENT_TICKS:
        raise ValueError('每段需有位移，且單軸不得超過 180 刻度')
    for pose in (raw, target):
        _fixed_origin(pose, fixed_origin, origin_limit_profile)
    actual_envelope = {n: max(12, endpoint_tolerance_ticks) if delta[n]
                       else stationary_tolerance_ticks for n in NAMES}
    # Moving goals are interpolated and retain the existing six-tick command
    # allowance. A fixed goal is never resent: its exact existing bias and the
    # declared measured box suffice, with no hypothetical extra goal change.
    required = {n: max(actual_envelope[n], abs(held[n] - raw[n]) + (6 if delta[n] else 0))
                for n in NAMES}
    envelope = tracking_envelope(required if approved_envelope is None else approved_envelope)
    if any(envelope[n] < required[n] for n in NAMES):
        raise ValueError('保持補償或實測誤差超出已確認的路徑範圍')
    physical = {n: held[n] + delta[n] for n in NAMES}
    physical['gripper'] = held['gripper']
    active = [n for n in NAMES if delta[n]]
    stationary = [n for n in NAMES if not delta[n]]
    inward = _validate_segment_limits(model, raw, held, target, physical, active, envelope,
                                     inward_limit_joints=inward_limit_joints)
    raw_bounds, measured_bounds, stationary_limits = _segment_bounds(
        model, raw, target, active, envelope, actual_envelope, inward_limit_joints=inward)
    check_bounds(held, raw_bounds)
    check_bounds(physical, raw_bounds)
    _tube(held, raw, target, envelope)
    _tube(physical, raw, target, envelope)
    clearance = model.screen_segment(raw, target, uncertainty_ticks=envelope, margin_ticks=0,
                                     stationary_limit_joints=stationary,
                                     **({'inward_limit_joints': list(inward)} if inward else {}))
    if not clearance['model_table_pass']:
        raise ValueError('這段動作會太靠近桌面，尚未執行')
    start_tip = model.tip(raw)[:3, 3]
    samples = []
    count = max(abs(v) for v in delta.values())
    last = start_tip
    length = 0.
    for i in range(count + 1):
        fraction = i / count
        pose = {n: raw[n] + fraction * (target[n] - raw[n]) for n in ALL}
        tip = model.tip(pose)[:3, 3]
        offset = (tip - start_tip) * 1000
        if not np.isfinite(offset).all():
            raise ValueError('動作模型輸出了無效座標')
        length += float(np.linalg.norm(tip - last) * 1000)
        last = tip
        if np.linalg.norm(offset) > MAX_SEGMENT_TIP_MM + 1e-7 or length > MAX_SEGMENT_TIP_MM + 1e-7:
            raise ValueError('單段手端移動超過 80 mm')
        samples.append({'fraction': fraction, 'tip_delta_mm': offset.tolist()})
    result = {
        'version': 1, 'motion_kind': 'local_joint_segment',
        'measured_origin': raw, 'held_goals': held, 'joint_targets': target,
        'physical_goal_targets': physical, 'delta_ticks': delta,
        'active_joints': active,
        **({'inward_limit_joints': list(inward), 'inward_limit_departures': inward,
            'inward_limit_rule': 'hard-valid start; commands only inward; endpoint restores full moving reserve; geometry box remains symmetric'} if inward else {}),
        'onboard_margin_ticks': {n: MOVING_LIMIT_MARGIN if n in active else 0 for n in ALL},
        'stationary_goal_resends': False,
        'stationary_limit_joints': stationary,
        'stationary_joint_limits': stationary_limits,
        'limit_validation_uncertainty_ticks': {n: envelope[n] if n in active else 0 for n in ALL},
        'stationary_bounds_rule': 'runtime intervals intersect unchanged hold box with onboard/URDF limits; full symmetric geometry envelope retained',
        'uncertainty_rule': 'moving: max(actual envelope, abs(held bias)+6); stationary: max(stationary_tolerance_ticks, abs(held bias)); jaw geometry: full360',
        'uncertainty_ticks': envelope, 'actual_tracking_envelope_ticks': actual_envelope,
        'endpoint_tolerance_ticks': endpoint_tolerance_ticks,
        'stationary_tolerance_ticks': stationary_tolerance_ticks,
        'fixed_origin': fixed_origin, 'jaw_tolerance_ticks': JAW_TOLERANCE,
        **({'origin_limit_profile': origin_limit_profile} if origin_limit_profile is not None else {}),
        'raw_bounds': raw_bounds,
        'measured_bounds': measured_bounds,
        'geometry_bounds': {n: [min(raw[n], target[n]) - envelope[n],
                                max(raw[n], target[n]) + envelope[n]] for n in NAMES},
        'predicted_tip_delta_mm': samples[-1]['tip_delta_mm'],
        'nominal_tip_path_mm': length, 'sampled_path': samples,
        'clearance': clearance, 'gripper_command_included': False,
        'camera_monitor_scope': 'fresh wrist frames and generation only; no visual obstacle detection',
        'not_environment_collision_certificate': True,
        'endpoint_frame': 'gripper_frame_link (fixed pad, not jaw center)',
    }
    result['contract_sha256'] = _fingerprint(result)
    return result


def rebind_segment_jaw(segment, measured_gripper, physical_gripper_goal):
    """Locally bind a post-contact boundary; body geometry cannot change.

    The CAD screen already encloses the full 360-degree moving jaw. This only
    transfers the measured jaw and existing physical goal into the contract;
    it is never a command, contact assertion, or clearance for a held object.
    """
    _validate_contract(segment)
    result = deepcopy(segment)
    measured = _pose({**segment['measured_origin'], 'gripper': measured_gripper})['gripper']
    held = _pose({**segment['held_goals'], 'gripper': physical_gripper_goal})['gripper']
    result['measured_origin']['gripper'] = measured
    result['joint_targets']['gripper'] = measured
    result['held_goals']['gripper'] = held
    result['physical_goal_targets']['gripper'] = held
    if result['fixed_origin'] is not None:
        result['fixed_origin']['gripper'] = measured
    result['contract_sha256'] = _fingerprint(result)
    return result


def _validate_contract(segment):
    if (not isinstance(segment, Mapping) or segment.get('version') != 1
            or segment.get('motion_kind') != 'local_joint_segment'
            or segment.get('contract_sha256') != _fingerprint(segment)):
        raise ValueError('動作段與已確認的計畫不同')


def execute_prepared_segment(grasp, segment, reference, cancel, report, *, speed=60,
                             deadline=None, continuation=False, measured_guard=None,
                             engineering_workspace=False):
    """Execute one approved segment locally, without consulting Astra.

    Continuation allows the previous segment's measured endpoint tolerance at
    the nominal next boundary. It never widens the ORIGINAL screened tube or
    rebases the target. First dispatch accepts at most three ticks of settling.
    """
    from .grasp import unchanged_wrist

    _validate_contract(segment)
    origin_profile = segment.get('origin_limit_profile')
    if type(engineering_workspace) is not bool or (origin_profile is not None and (
            not engineering_workspace or origin_profile != WORKSPACE_ORIGIN_PROFILE)):
        raise ValueError('較大的工作範圍只能由明確的工程校準協定執行')
    if engineering_workspace and origin_profile != WORKSPACE_ORIGIN_PROFILE:
        raise ValueError('工程校準動作缺少固定原點的工作範圍契約')
    _fixed_origin(segment['measured_origin'], segment['fixed_origin'], origin_profile)
    _fixed_origin(segment['joint_targets'], segment['fixed_origin'], origin_profile)
    if measured_guard is not None and not callable(measured_guard):
        raise ValueError('Measured guard must be callable')
    extra_measured_guard = measured_guard
    if type(continuation) is not bool:
        raise ValueError('Invalid continuation flag')
    r, model = grasp.robot, grasp.model
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not 0 < speed <= 60:
        raise ValueError('計畫執行速度需介於 0 與 60 刻度／秒')
    started = time.monotonic()
    if deadline is None:
        deadline = started + MAX_PROGRAM_SECONDS
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError('Invalid program deadline')
    deadline = min(deadline, started + MAX_PROGRAM_SECONDS)
    generation = reference.get('camera_generation')
    if reference.get('source') != 'wrist' or not generation:
        raise ValueError('動作計畫只接受腕部相機')
    origin = segment['measured_origin']
    target = segment['joint_targets']
    active = segment['active_joints']
    inward = segment.get('inward_limit_joints', [])

    def live():
        if cancel is not None and cancel.is_set():
            raise SessionCancelled('動作已停止')
        if r.stop.is_set():
            raise SessionCancelled('動作已停止')
        if time.monotonic() >= deadline:
            raise RuntimeError('這次連續動作已到執行時間上限，先保持姿勢')
        if not r.camera.fresh():
            raise RuntimeError('腕部畫面中斷，已停止移動')
        if r.camera.generation != generation:
            raise RuntimeError('腕部相機重新連接，請重新觀察')

    def guard(actual):
        live()
        _pose(actual)
        _tube(actual, origin, target, segment['uncertainty_ticks'],
              source='geometry_guard_actual_or_prospective_goal')
        check_bounds(actual, segment['raw_bounds'])
        if abs(actual['gripper'] - origin['gripper']) > JAW_TOLERANCE:
            raise RuntimeError('移動時夾爪開口改變，先保持姿勢')
        floor = model.floor_minimum(actual)
        if any(z < .005 for n, z in floor.items() if n != 'base_link'):
            raise RuntimeError('動作太靠近桌面，先保持姿勢')

    def measured_guard(actual):
        live()
        _tube(actual, origin, target, segment['actual_tracking_envelope_ticks'],
              source='measured_encoder_guard')
        _fixed_origin(actual, segment['fixed_origin'], origin_profile)
        check_bounds(actual, segment['measured_bounds'])
        if extra_measured_guard is not None:
            extra_measured_guard(actual)

    live()
    r.begin(allow_rearm=False)
    before = grasp.capture()
    unchanged_wrist(reference, before)
    allowance = segment['endpoint_tolerance_ticks'] if continuation else INITIAL_TOLERANCE
    if any(abs(before['joints'][n] - origin[n]) > (allowance if n in active else INITIAL_TOLERANCE) for n in ALL):
        raise RuntimeError('手臂位置已離開計畫的起點，請重新觀察')
    if dict(r.goals) != segment['held_goals']:
        raise RuntimeError('保持目標已改變，不能接續先前計畫')
    guard(before['joints'])
    measured_guard(before['joints'])
    # Screen again locally to catch changed calibration/model/config, without
    # a new inference, new target, or a larger uncertainty envelope.
    checked_inward = _validate_segment_limits(model, origin, segment['held_goals'], target,
                             segment['physical_goal_targets'], active, segment['uncertainty_ticks'],
                             inward_limit_joints=inward)
    if checked_inward != segment.get('inward_limit_departures', {}):
        raise ValueError('向內退開的行程範圍已改變，請重新規劃')
    raw_bounds, measured_bounds, _ = _segment_bounds(model, origin, target, active,
        segment['uncertainty_ticks'], segment['actual_tracking_envelope_ticks'], inward_limit_joints=inward)
    if raw_bounds != segment['raw_bounds'] or measured_bounds != segment['measured_bounds']:
        raise ValueError('固定關節行程範圍已改變，請重新規劃')
    stationary = [n for n in NAMES if n not in active]
    if stationary != segment['stationary_limit_joints']:
        raise ValueError('固定關節與已確認的計畫不同')
    checked = model.screen_segment(origin, target, uncertainty_ticks=segment['uncertainty_ticks'], margin_ticks=0,
                                   stationary_limit_joints=stationary,
                                   **({'inward_limit_joints': inward} if inward else {}))
    if not checked['model_table_pass']:
        raise ValueError('這段動作未通過目前的桌面檢查')
    with r.io:
        live()
        if dict(r.goals) != segment['held_goals']:
            raise RuntimeError('保持目標已改變，不能接續先前計畫')
        if continuation or r.pwm_owner is not None:
            anchor = adopt_goal_anchor(before['joints'], r.goals,
                                       previous_goals=r.goals, previous_bias=r.bias)
        else:
            # Bind compensation to the assessed origin, never to a new partial
            # endpoint. Up to three ticks of settling are already in its tube.
            anchor = adopt_goal_anchor(origin, r.goals)
        expected_bias = {n: segment['held_goals'][n] - origin[n] for n in NAMES}
        if any(anchor['bias'][n] != expected_bias[n] for n in NAMES):
            raise RuntimeError('保持補償已改變，不能接續先前計畫')
        r.bias = anchor['bias']
    logical = {n: target[n] for n in active}
    report('依照整段計畫連續移動，正在核對馬達位置')
    grasp.record({'type': 'motion_program_segment_start', 'contract_sha256': segment['contract_sha256'],
                  'before': before, 'targets': logical, 'continuation': continuation,
                  'command_progress': 'measured_path_fraction_governor',
                  'motion_contract': {key: deepcopy(segment[key]) for key in (
                      'measured_origin', 'held_goals', 'actual_tracking_envelope_ticks',
                      'uncertainty_ticks', 'stationary_tolerance_ticks')},
                  'camera_monitor_scope': segment['camera_monitor_scope']})
    governor_stats = {'kind': 'measured_path_fraction', 'updates': 0, 'limited_updates': 0,
                      'waiting_updates': 0, 'maximum_nominal_lead_fraction': 0.,
                      'stall_timeout_seconds': PHASE_STALL_SECONDS}
    def observe_motion(event):
        if event.get('kind') == 'endpoint_settling':
            if engineering_workspace:
                grasp.record({'type': 'motion_program_endpoint_settling',
                    'contract_sha256': segment['contract_sha256'], **event})
            return
        live()
        if 'nominal_fraction' in event:
            governor_stats['updates'] += 1
            lead = max(0., event['nominal_fraction'] - event['fraction'])
            governor_stats['limited_updates'] += int(lead > 1e-9)
            governor_stats['waiting_updates'] += int(event['waiting_for_path'])
            governor_stats['maximum_nominal_lead_fraction'] = max(
                governor_stats['maximum_nominal_lead_fraction'], lead)
    dispatched = False
    try:
        live()
        dispatched = True
        outcome = r.glide(logical, speed=speed, precision=segment['endpoint_tolerance_ticks'],
                feedback=False, guard=guard, measured_guard=measured_guard,
                raw_bounds=segment['raw_bounds'], measured_bounds=segment['measured_bounds'],
                observe=observe_motion, allow_rearm=False,
                phase_governor=SegmentPhaseGovernor(segment),
                **({'settle_timeout_s': 2.5, 'settle_telemetry': True,
                    'allow_stable_partial_observation': True} if engineering_workspace else {}),
                **({'inward_limit_contract': segment} if inward else {}))
        live()
        after = grasp.capture()
        if after.get('source') != 'wrist' or after.get('camera_generation') != generation:
            raise RuntimeError('動作後腕部影像來源改變')
        if after['capture_monotonic'] <= before['capture_monotonic']:
            raise RuntimeError('動作後缺少更新的腕部畫面')
        guard(after['joints'])
        measured_guard(after['joints'])
        partial = (engineering_workspace and isinstance(outcome, dict)
                   and outcome.get('completion') == 'stable_partial_observation')
        reached = all(abs(after['joints'][n] - target[n]) <= segment['endpoint_tolerance_ticks'] for n in active)
        partial = partial and not reached
        if partial and not _partial_observation_progress(after['joints'], origin, logical):
            raise RuntimeError('停穩後沒有保留有效的取景位移，已保持姿勢')
        if not reached and not partial:
            raise RuntimeError('動作後未達到計畫終點，已保持姿勢')
        if not any((after['joints'][n] - before['joints'][n]) * segment['delta_ticks'][n] > 0 for n in active):
            raise RuntimeError('馬達未產生計畫方向的位移，已保持姿勢')
        displacement = (model.tip(after['joints'])[:3, 3] - model.tip(before['joints'])[:3, 3]) * 1000
        receipt = {'before': before, 'after': after, 'elapsed': time.monotonic() - started,
                   'actual_delta_model_mm': displacement.tolist(), 'joint_targets': deepcopy(target),
                   'endpoint_error_ticks': {n: after['joints'][n] - target[n] for n in NAMES},
                   'completion': 'stable_partial_observation' if partial else 'within_declared_endpoint_tolerance',
                   'model_displacement_not_external_metrology': True,
                   'camera_monitor_scope': segment['camera_monitor_scope'],
                   'phase_governor': dict(governor_stats),
                   'contract_sha256': segment['contract_sha256'], 'cloud_calls_during_segment': 0}
        if engineering_workspace:
            receipt.update(reached_nominal_endpoint=reached,
                           partial_observation=partial,
                           observation_accuracy_verified=False)
            if partial:
                receipt['endpoint_settling'] = deepcopy(outcome)
        grasp.record({'type': 'motion_program_segment_result', **receipt})
        return receipt
    except Exception as exc:
        if dispatched:
            r.error = str(exc)
            r.stop.set()
            r.hold()
        diagnostic = {key: deepcopy(getattr(exc, key)) for key in ('path_diagnostic', 'guard_context')
                      if hasattr(exc, key)}
        grasp.record({'type': 'motion_program_segment_failed', 'error': str(exc),
                      'contract_sha256': segment['contract_sha256'], 'dispatched': dispatched,
                      'phase_governor': dict(governor_stats),
                      **diagnostic})
        raise


def execute_joint_segment(grasp, target, reference, cancel, report, *, fixed_origin=None,
                          endpoint_tolerance_ticks=10, speed=60, deadline=None,
                          approved_envelope=None, measured_guard=None):
    """Convenience for a caller that has already approved this exact target.

    Prefer prepare + assess full program + execute_prepared_segment when a
    model chooses among candidate trajectories.
    """
    segment = prepare_joint_segment(grasp.model, reference['joints'], grasp.robot.goals, target,
                                    fixed_origin=fixed_origin, endpoint_tolerance_ticks=endpoint_tolerance_ticks,
                                    approved_envelope=approved_envelope)
    return execute_prepared_segment(grasp, segment, reference, cancel, report,
                                    speed=speed, deadline=deadline, measured_guard=measured_guard)
