"""Engineering-only empty-jaw wrist images, never a metric calibration profile.

Only the gripper receives trajectory goals. Existing RAM-PWM ownership handles
neutralization/restoration; faults retain its recovery barrier and protection.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .calibration_store import _validate_frame
from .gripper_context import BODY, require_matched_body
from .gripper_power import SessionCancelled

TARGETS = (1550, 1625, 1700, 1775, 1850)
PROTOCOLS = {'standard': TARGETS, 'wide': (1550, 1675, 1800, 1925, 2050)}
SPEED = 50


def validate_empty_evidence(frame, generation, *, empty_verified):
    if empty_verified is not True:
        raise ValueError('請明確確認剛檢查的腕部影像中夾爪空置。')
    if (not isinstance(frame, dict) or frame.get('source') != 'wrist'
            or frame.get('camera_generation') != generation or not isinstance(generation, str) or not generation
            or not isinstance(frame.get('id'), str) or not frame['id']
            or type(frame.get('sequence')) is not int or frame['sequence'] <= 0
            or type(frame.get('captured_at')) not in (int, float)
            or not math.isfinite(frame['captured_at']) or not 0 <= time.time() - frame['captured_at'] < 30
            or type(frame.get('capture_monotonic')) not in (int, float)
            or not math.isfinite(frame['capture_monotonic'])
            or not 0 <= time.monotonic() - frame['capture_monotonic'] < 30):
        raise ValueError('請先檢查 30 秒內、同一腕部相機的新影像，確認夾爪空置。')


def _save_image(directory, frame, name):
    data = Path(frame['path']).read_bytes()
    if not 0 < len(data) <= 16 * 1024 * 1024:
        raise ValueError('腕部影像大小無效。')
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or list(image.shape[1::-1]) != frame.get('image_size'):
        raise ValueError('腕部影像與原生解析度紀錄不一致。')
    destination = directory / name
    with destination.open('xb') as output:
        output.write(data)
    return {**deepcopy(frame), 'path': str(destination),
            'image_sha256': hashlib.sha256(data).hexdigest()}


def collect_jaw_images(robot, snap, report, cancel, empty_frame, *, empty_verified,
                       camera_id, mount_revision, geometry_revision, camera_profile=None,
                       protocol='standard'):
    """Capture five fixed opening labels with actual settled encoder positions.

    Targets are protocol constants, not arbitrary API inputs. The source image
    and every captured image are copied once into a new session directory.
    A failed session retains partial evidence and never returns to its origin.
    """
    if protocol not in PROTOCOLS:
        raise ValueError('未知的空夾爪採樣範圍')
    targets = PROTOCOLS[protocol]
    generation = robot.camera.generation
    validate_empty_evidence(empty_frame, generation, empty_verified=empty_verified)
    if (not robot.connected or robot.moving or robot.pwm_owner is not None
            or robot.stop.is_set() or cancel.is_set()):
        raise RuntimeError('需要已連接、空閒且未停止的手臂；不能接管持物或其他動作。')
    power = robot.gripper_power
    power.assert_motion_allowed()
    for target in targets:
        robot.validate({'gripper': target})
    # Read-only adoption of an existing stable hold is allowed; torque enable
    # and reseeding body goals are not part of this acquisition.
    robot.begin(allow_rearm=False)
    origin = robot.read()
    reference = {n: origin[n] for n in BODY}
    # Settled inspection frames carry measured joints; ordinary inspected
    # snapshots do not. Never invent a pose binding for the latter.
    if 'joints' in empty_frame:
        require_matched_body({n: empty_frame['joints'][n] for n in BODY}, reference)
        if abs(empty_frame['joints']['gripper'] - origin['gripper']) > 4:
            raise ValueError('已檢查影像的夾爪開口與目前實測位置不同。')
    body_goals = {n: robot.goals[n] for n in BODY}
    initial = robot.gripper_telemetry()
    if initial['pwm_limit'] != 500 or initial['temperature_c'] > 60 or initial['status'] != 0:
        raise RuntimeError('夾爪原始 PWM 或保護狀態不符合工程採樣條件。')
    require_matched_body(initial['body_position'], reference)
    identity = uuid.uuid4().hex
    directory = Path(robot.runtime) / 'jaw-calibration-captures' / identity
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'frames').mkdir()
    evidence = _save_image(directory, empty_frame, 'empty-inspection.jpg')
    manifest = dict(schema_version=1, session_id=identity, source='wrist-only',
        camera_id=camera_id, mount_revision=mount_revision, geometry_revision=geometry_revision,
        camera_generation=generation, image_size=empty_frame['image_size'],
        empty_verified=True, empty_evidence=evidence, targets_raw=list(targets), protocol=protocol,
        opening_protocol=dict(name=protocol, targets_raw=list(targets),
            training_targets_raw=list(targets[::2]), heldout_targets_raw=list(targets[1::2]),
            declared_before_capture=True),
        speed_ticks_per_second=SPEED, pwm_limit=100, original_pwm_limit=500,
        body_reference=reference, body_hold_goals=body_goals, original_joints=origin,
        frames=[], metric_calibrated=False, profile_produced=False,
        interpretation='Empty-jaw image evidence only. Opening labels select views; geometry must use measured joints, never the commanded raw label.')
    if camera_profile is not None:
        profile_json = json.dumps(camera_profile, sort_keys=True, separators=(',', ':'), allow_nan=False)
        manifest.update(camera_profile=deepcopy(camera_profile),
                        camera_profile_sha256=hashlib.sha256(profile_json.encode()).hexdigest())
    completed, receipt, failure = False, None, None
    entered = False
    with (directory / 'events.jsonl').open('x') as events:
        def record(row):
            events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); events.flush()

        def check_cancel():
            if cancel.is_set() or robot.stop.is_set():
                raise SessionCancelled('夾爪影像採樣已停止。')
            if robot.camera.generation != generation or not robot.camera.fresh():
                raise RuntimeError('腕部相機已中斷或更換工作階段。')

        def measured_guard(actual):
            check_cancel()
            require_matched_body({n: actual[n] for n in BODY}, reference)
            if any(robot.goals[n] != body_goals[n] for n in BODY):
                raise RuntimeError('採樣期間身體關節的保持目標已改變。')

        def telemetry():
            check_cancel()
            sample = robot.gripper_telemetry()
            if sample['pwm_limit'] != 100 or sample['temperature_c'] > 60 or sample['status'] != 0:
                raise RuntimeError('夾爪 PWM、溫度或保護狀態改變。')
            require_matched_body(sample['body_position'], reference)
            return sample

        def move(target, *, returning=False):
            window = deque()
            start = robot.read()['gripper']
            sign = 1 if target >= start else -1
            def observe(progress):
                sample = telemetry()
                record(dict(type='telemetry', target_raw=target, returning=returning, **sample))
                window.append(sample)
                while len(window) > 1 and sample['capture_monotonic'] - window[1]['capture_monotonic'] >= .4:
                    window.popleft()
                first = window[0]
                if (progress['elapsed'] >= .8 and sample['capture_monotonic'] - first['capture_monotonic'] >= .4
                        and sign * (sample['goal'] - first['goal']) >= 15
                        and sign * (sample['position'] - first['position']) < 3):
                    raise SessionCancelled('低輸出下夾爪未持續前進；停止並清除待執行目標。')
            robot.glide({'gripper': target}, speed=SPEED, precision=16, return_origin=returning,
                        allow_rearm=False, measured_guard=measured_guard, observe=observe)

        try:
            check_cancel()
            validate_empty_evidence(empty_frame, generation, empty_verified=empty_verified)
            measured_guard(robot.read())
            robot.pwm_owner = power.enter(); entered = True
            for index, target in enumerate(targets):
                check_cancel()
                report(f'拍攝空夾爪開口 · {index + 1} / {len(targets)} · 目標 {target}')
                move(target)
                telemetry()
                try:
                    frame = robot.settled_snapshot(snap, guard=measured_guard)
                except RuntimeError as exc:
                    # settled_snapshot predates typed cancellation. Translate
                    # ONLY its exact stop exception, never an I/O/servo failure.
                    if str(exc) == '環顧已停止' and (cancel.is_set() or robot.stop.is_set()):
                        raise SessionCancelled(str(exc)) from exc
                    raise
                check_cancel()
                frame = {**frame, 'camera_id': camera_id, 'mount_revision': mount_revision,
                         'geometry_revision': geometry_revision}
                _validate_frame(frame)
                measured_guard(frame['joints'])
                if (frame['camera_generation'] != generation or frame['image_size'] != manifest['image_size']
                        or abs(frame['joints']['gripper'] - target) > 16
                        or frame['sequence'] <= (manifest['frames'][-1]['sequence'] if manifest['frames'] else empty_frame['sequence'])):
                    raise RuntimeError('採樣影像與新開口實測位置不一致。')
                after = telemetry()
                if abs(after['position'] - frame['joints']['gripper']) > 2:
                    raise RuntimeError('拍照後夾爪仍在移動。')
                saved = _save_image(directory / 'frames', frame, f'{index + 1:02d}-{target}.jpg')
                saved.update(target_raw=target, actual_gripper_raw=frame['joints']['gripper'],
                             telemetry_after_capture=after)
                manifest['frames'].append(saved)
                record(dict(type='capture', frame=saved))
            # No return is attempted after cancellation, drift, or hardware fault.
            check_cancel()
            move(origin['gripper'], returning=True)
            measured_guard(robot.read())
            telemetry()
            completed = True
        except BaseException as exc:
            failure = exc
            if entered and not isinstance(exc, (SessionCancelled, KeyboardInterrupt)):
                power.leave_pending()
        finally:
            try:
                if power.state == 'active' and entered:
                    receipt = power.finish(cancelled=failure is not None or cancel.is_set())
                    if receipt and 'neutral_goal' in receipt:
                        robot.goals['gripper'] = receipt['neutral_goal']; robot.bias['gripper'] = 0
            except BaseException as exc:
                failure = exc; completed = False
            finally:
                robot.pwm_owner = None
            if failure:
                robot.error = str(failure); robot.stop.set(); robot.hold()
            completed = bool(completed and failure is None and receipt and receipt.get('completed')
                             and not cancel.is_set() and not robot.stop.is_set())
            manifest.update(completed=completed, return_to_original_completed=completed,
                cleanup_receipt=receipt, error=str(failure) if failure else None,
                protected_state=power.state, completed_at=time.time())
            record(dict(type='end', completed=completed, cleanup_receipt=receipt, error=manifest['error']))
            with (directory / 'session.json').open('x') as output:
                json.dump(manifest, output, ensure_ascii=False, indent=2, allow_nan=False)
    if failure:
        raise failure
    if not completed:
        raise SessionCancelled('採樣未完整完成；已保存目前證據。')
    report('五個空夾爪開口影像已保存，已回到原開口；尚未形成幾何校準。')
    return dict(session_id=identity, manifest_path=str(directory / 'session.json'),
                completed=True, frame_count=len(manifest['frames']), metric_calibrated=False,
                actual_gripper_raw=[f['actual_gripper_raw'] for f in manifest['frames']],
                target_raw=list(targets), restore_verified=bool(receipt.get('completed')))
