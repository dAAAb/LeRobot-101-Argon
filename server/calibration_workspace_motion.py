"""Finite, staged shoulder/elbow survey; no automatic multi-station execution.

One immutable origin and declared total bounds cover the whole protocol. Each
station is freshly prepared and separately reviewed using the wrist camera.
CAD/table screening and predicted chessboard pixels do NOT verify obstacles.
Nothing here connects, rearms, captures a board sample, or changes calibration.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import time

import cv2
import numpy as np

from . import active_observation as observation
from .board_calibration import BoardSpec, _detection
from .calibration_extension import _camera_board
from .local_cartesian import ALL, NAMES, inward_departure_bounds
from .motion_program import (prepare_joint_segment, execute_prepared_segment,
    WORKSPACE_ORIGIN_PROFILE, WORKSPACE_ORIGIN_DELTA_BOUNDS, _fixed_origin)
from .spatial_runtime import geometry_revision

VERSION = 1
MAX_ORIGIN_TIP_MM = 180.
ENDPOINT_TOLERANCE = 16
STATIONARY_TOLERANCE = 12
MAX_NOMINAL_SEGMENT_TICKS = 160  # Reserve room for the last actual endpoint.
MAX_TTL_SECONDS = 120.
REFERENCE_ORIGIN = dict(zip(NAMES, (2181, 1032, 3042, 2174, 884)))
# Six pairs span proximal joints, two wrist axes and the prospective side-grasp
# neighborhood. These are a finite survey template, not a manipulation plan.
# Poses are relative to the original measured origin, NEVER the preceding one.
CAPTURE_OFFSETS = (
    (160, 90, -42, -65, 0), (160, 90, -42, -100, 60),
    (184, 358, -95, -227, 36), (184, 358, -95, -262, 96),
    (184, 536, -142, -348, 54), (184, 536, -142, -383, 114),
    (184, 715, -189, -478, 73), (184, 715, -189, -513, 133),
    (186, 894, -237, -617, 91), (186, 894, -237, -652, 151),
    (187, 1073, -284, -873, 109), (187, 1155, -428, -774, 169),
)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _seal(value):
    value['contract_sha256'] = _hash({k: v for k, v in value.items() if k != 'contract_sha256'})
    return value


def _validate(value, kind):
    if (not isinstance(value, Mapping) or value.get('version') != VERSION
            or value.get('kind') != kind or value.get('contract_sha256') !=
            _hash({k: v for k, v in value.items() if k != 'contract_sha256'})):
        raise ValueError('工程工作範圍內容已改變，請重新建立並核對。')


def _profile(profile, model, camera_id, mount_revision):
    if (not isinstance(profile, dict) or profile.get('source') != 'wrist-only'
            or profile.get('metric_calibrated') is not True
            or profile.get('camera_id') != camera_id
            or profile.get('mount_revision') != mount_revision
            or profile.get('geometry_revision') != geometry_revision(model)):
        raise ValueError('工程取景需要目前已驗證的腕部相機版本。')
    for key, shape in (('camera_matrix', (3, 3)), ('handeye', (4, 4))):
        value = np.asarray(profile[key], float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('相機參數不完整。')
    if profile.get('native_image_size') != [1920, 1080]:
        raise ValueError('這份工程取景協定使用原生 1920 × 1080 腕部影像。')
    return _hash(profile)


def _origin_guard(model, origin, pose):
    _fixed_origin(pose, origin, WORKSPACE_ORIGIN_PROFILE)
    distance = float(np.linalg.norm(model.tip(pose)[:3, 3] - model.tip(origin)[:3, 3]) * 1000)
    if not math.isfinite(distance) or distance > MAX_ORIGIN_TIP_MM:
        raise ValueError('超出這份工程協定原始起點的 180 mm 手端範圍。')


def _segment(model, raw, held, target, origin):
    # Only an actual hard-valid near-end elbow can opt into inward departure.
    inward = ()
    if target['elbow_flex'] < raw['elbow_flex']:
        envelope = {n: 0 for n in ALL}
        envelope['elbow_flex'] = max(ENDPOINT_TOLERANCE,
                                    abs(held['elbow_flex'] - raw['elbow_flex']) + 6)
        try:
            inward_departure_bounds(model, raw, target, envelope, ['elbow_flex'])
            inward = ('elbow_flex',)
        except ValueError:
            pass  # An interior joint uses all the normal moving reserves.
    result = prepare_joint_segment(model, raw, held, target, fixed_origin=origin,
        endpoint_tolerance_ticks=ENDPOINT_TOLERANCE,
        stationary_tolerance_ticks=STATIONARY_TOLERANCE,
        inward_limit_joints=inward, origin_limit_profile=WORKSPACE_ORIGIN_PROFILE)
    reserve = observation._tracking_reserve(model, result['uncertainty_ticks'])
    initial_tip = model.tip(origin)[:3, 3]
    for sample in result['sampled_path']:
        fraction = sample['fraction']
        pose = {n: raw[n] + fraction * (target[n] - raw[n]) for n in ALL}
        envelope = result['uncertainty_ticks']
        for n, (lo, hi) in WORKSPACE_ORIGIN_DELTA_BOUNDS.items():
            if not (lo <= pose[n] - origin[n] - envelope[n]
                    and pose[n] - origin[n] + envelope[n] <= hi):
                raise ValueError('工程路徑沒有保留原始總範圍內的追蹤誤差。')
        distance = float(np.linalg.norm(model.tip(pose)[:3, 3] - initial_tip) * 1000)
        if distance + reserve > MAX_ORIGIN_TIP_MM:
            raise ValueError('工程路徑沒有保留 180 mm 總範圍內的位置誤差。')
    return result


def _projection(model, pose, profile, board_base):
    board = np.asarray(board_base, float)
    if board.shape != (4, 4) or not np.isfinite(board).all():
        raise ValueError('棋盤取景模型無效。')
    camera = model.fk.transforms(pose)['gripper_link'] @ np.asarray(profile['handeye'])
    transform = np.linalg.inv(camera) @ board
    points = BoardSpec().object_points() @ transform[:3, :3].T + transform[:3, 3]
    pixels = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
        np.asarray(profile['camera_matrix']), np.asarray(profile['distortion']))[0].reshape(-1, 2)
    if not np.isfinite(pixels).all():
        raise ValueError('棋盤投影不是有限值。')
    visible = ((points[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < 1920)
               & (pixels[:, 1] >= 0) & (pixels[:, 1] < 1080))
    return {'visible_corner_count': int(sum(visible)), 'total_corner_count': len(points),
        'bbox_px': [pixels.min(0).tolist(), pixels.max(0).tolist()],
        'pixels_px': pixels.tolist(), 'occlusion_checked': False,
        'not_external_clearance_evidence': True}


def board_reference_from_capture(grasp, camera_profile, frame, *, camera_id, mount_revision):
    """Re-estimate framing from one recorded wrist board view; no fit/activation.

    This PnP estimate is only for keeping the printed board in frame. It does
    not validate the current FK or permit object approach.
    """
    profile_hash = _profile(camera_profile, grasp.model, camera_id, mount_revision)
    if frame.get('source') != 'wrist' or frame.get('image_size') != [1920, 1080]:
        raise ValueError('棋盤重投影需要原生腕部影像。')
    detection = _detection(frame, BoardSpec())
    if not detection.get('found'):
        raise ValueError('這張影像沒有完整棋盤，先重新取景。')
    camera_board = _camera_board(BoardSpec().object_points(),
        np.asarray(detection['corners_px']), np.asarray(camera_profile['camera_matrix']),
        np.asarray(camera_profile['distortion']))
    transform = grasp.model.fk.transforms(frame['joints'])['gripper_link'] @ np.asarray(camera_profile['handeye']) @ camera_board
    return _seal({'version': VERSION, 'kind': 'engineering_workspace_board_reference',
        'frame_id': frame['id'], 'camera_generation': frame['camera_generation'],
        'camera_profile_sha256': profile_hash, 'geometry_revision': geometry_revision(grasp.model),
        'board_transform_base': transform.tolist(), 'source': 'wrist-only',
        'use': 'nominal_board_framing_only_not_calibration_or_clearance'})


def build_workspace_protocol(grasp, frame, *, session_id, camera_profile, camera_id, mount_revision):
    """Seal a twelve-capture survey and any required intermediate transit stops.

    ``origin`` is this frame's actual pose, stored once by the calling API.
    All stations are nominally screened now; each is screened AGAIN using its
    actual start/held goals immediately before a separate reviewed execution.
    """
    reference = observation._frame(frame)
    if not isinstance(session_id, str) or not session_id or len(session_id) > 160:
        raise ValueError('需要固定棋盤工作階段識別。')
    if grasp.state not in ('idle', 'approaching') or grasp.robot.pwm_owner is not None:
        raise ValueError('先完成夾爪目前的工作，再建立工程取景。')
    origin = observation._pose(reference['joints'])
    if any(abs(origin[n] - REFERENCE_ORIGIN[n]) > 24 for n in NAMES):
        raise ValueError('目前姿勢不在這份有限肩肘取景協定的起始區域，請重新設計完整協定。')
    profile_hash = _profile(camera_profile, grasp.model, camera_id, mount_revision)
    board = camera_profile['validation_scope']['board_plane_base']
    held = observation._pose(grasp.robot.goals)
    nominal, predicted_held, stations = dict(origin), dict(held), []
    # Leave the elbow's near-limit starting pose by itself. A simultaneous
    # shoulder/pan/wrist start can stall the shared path fraction while the
    # loaded elbow has not yet overcome its initial deadband. This is a fully
    # screened, separately reviewed transit, not a tolerance relaxation.
    target = {**origin, 'elbow_flex': origin['elbow_flex'] - 42}
    segment = _segment(grasp.model, nominal, predicted_held, target, origin)
    stations.append({'station_id': 'station-01', 'station_index': 0,
        'target_joints': target, 'capture_id': None, 'split': None,
        'capture_required': False, 'nominal_segment': segment,
        'purpose': 'single_elbow_inward_departure',
        'nominal_board_projection': _projection(grasp.model, target, camera_profile, board),
        'nominal_projection_source': 'parent_profile_board_plane_prior',
        'external_clearance_verified': False})
    nominal, predicted_held = target, segment['physical_goal_targets']
    for index, offsets in enumerate(CAPTURE_OFFSETS):
        goal = {**origin, **{n: origin[n] + d for n, d in zip(NAMES, offsets)}}
        # Nominal steps reserve 20 ticks so actual endpoint errors cannot turn
        # a planned <=180 segment into an unexecutable 196-tick segment.
        parts = math.ceil(max(abs(goal[n] - nominal[n]) for n in NAMES) / MAX_NOMINAL_SEGMENT_TICKS)
        start = dict(nominal)
        for part in range(1, parts + 1):
            target = {n: round(start[n] + part / parts * (goal[n] - start[n])) for n in ALL}
            segment = _segment(grasp.model, nominal, predicted_held, target, origin)
            capture = part == parts
            stations.append({'station_id': f'station-{len(stations) + 1:02d}',
                'station_index': len(stations), 'target_joints': target,
                'capture_id': f'capture-{index + 1:02d}' if capture else None,
                'split': ('training' if index % 2 == 0 else 'heldout') if capture else None,
                'capture_required': capture, 'nominal_segment': segment,
                'nominal_board_projection': _projection(grasp.model, target, camera_profile, board),
                'nominal_projection_source': 'parent_profile_board_plane_prior',
                'external_clearance_verified': False})
            nominal, predicted_held = target, segment['physical_goal_targets']
    if dict(grasp.robot.goals) != held:
        raise ValueError('建立工程協定期間保持目標改變。')
    return _seal({'version': VERSION, 'kind': 'engineering_workspace_calibration',
        'session_id': session_id, 'source': 'wrist-only', 'origin': origin,
        'reference': deepcopy(reference), 'initial_held_goals': held,
        'camera_id': camera_id, 'mount_revision': mount_revision,
        'camera_profile': deepcopy(camera_profile), 'camera_profile_sha256': profile_hash,
        'geometry_revision': geometry_revision(grasp.model),
        'origin_limit_profile': WORKSPACE_ORIGIN_PROFILE,
        'joint_delta_bounds_from_original_origin': {n: list(b) for n, b in WORKSPACE_ORIGIN_DELTA_BOUNDS.items()},
        'maximum_tip_distance_from_original_origin_mm': MAX_ORIGIN_TIP_MM,
        'maximum_segment_ticks': 180, 'maximum_segment_tip_mm': 80.,
        'maximum_segment_seconds': 30., 'stations': stations,
        'split_fixed_before_capture': True, 'capture_count': 12,
        'automatic_station_chaining': False, 'motors_commanded': False,
        'requires_individual_fresh_path_review': True})


def _next(protocol, station_index, completed_station_ids):
    _validate(protocol, 'engineering_workspace_calibration')
    if (type(station_index) is not int or not 0 <= station_index < len(protocol['stations'])
            or completed_station_ids != [s['station_id'] for s in protocol['stations'][:station_index]]):
        raise ValueError('工程取景必須接續已記錄的上一站，不能跳站或重設原點。')
    return protocol['stations'][station_index]


def _camera_generation(protocol, camera_rebind):
    """Accept only an explicit offline rebind of this unchanged camera setup."""
    if camera_rebind is None:
        return protocol['reference']['camera_generation']
    _validate(camera_rebind, 'engineering_workspace_camera_rebind')
    expected = dict(protocol_sha256=protocol['contract_sha256'], camera_id=protocol['camera_id'],
        mount_revision=protocol['mount_revision'], camera_profile_sha256=protocol['camera_profile_sha256'],
        from_generation=protocol['reference']['camera_generation'],
        original_reference_sha256=_hash(protocol['reference']))
    if (any(camera_rebind.get(k) != value for k, value in expected.items())
            or not isinstance(camera_rebind.get('to_generation'), str) or not camera_rebind['to_generation']):
        raise ValueError('相機重新連線紀錄與原始工程協定不符。')
    return camera_rebind['to_generation']


def _station_start_and_target(protocol, station_index, raw, previous_receipt, generation):
    """Retain measured progress; a nominally unchanged axis stays held."""
    station = protocol['stations'][station_index]
    nominal = station['target_joints']
    if station_index == 0:
        if previous_receipt is not None:
            raise ValueError('第一站不能帶入另一站的移動紀錄。')
        previous = protocol.get('replan', {}).get('checkpoint_start_joints',
            protocol.get('recovery', {}).get('recovery_start', protocol['origin']))
        target = dict(nominal)
        receipt_hash = None
        tolerance = 3
    else:
        prior_station = protocol['stations'][station_index - 1]
        if (not isinstance(previous_receipt, Mapping) or previous_receipt.get('completed') is not True
                or previous_receipt.get('protocol_sha256') != protocol['contract_sha256']
                or previous_receipt.get('station_id') != prior_station['station_id']):
            raise ValueError('下一站需要上一站保存的實測移動紀錄，不能假設已到名目位置。')
        after = previous_receipt.get('after')
        if (not isinstance(after, Mapping) or after.get('source') != 'wrist'
                or not after.get('id') or after.get('camera_generation') != generation):
            raise ValueError('上一站實測取像與目前相機工作階段不符。')
        previous = observation._pose(after.get('joints'))
        target = {n: nominal[n] if nominal[n] != prior_station['target_joints'][n] else raw[n]
                  for n in ALL}
        receipt_hash = _hash(previous_receipt)
        tolerance = STATIONARY_TOLERANCE
    if 'planned_active_joints' in station:
        active = station['planned_active_joints']
        if (not isinstance(active, list) or len(set(active)) != len(active)
                or any(n not in NAMES for n in active)):
            raise ValueError('衍生路徑的活動關節清單無效。')
        target = {n: nominal[n] if n in active else raw[n] for n in ALL}
    if any(abs(raw[n] - previous[n]) > (4 if n == 'gripper' else tolerance) for n in ALL):
        raise ValueError('目前姿勢已離開工程協定的上一站，先保持姿勢並重新檢查。')
    return target, receipt_hash


def prepare_workspace_station(grasp, protocol, frame, *, station_index,
                              completed_station_ids, ttl_seconds=90., board_reference=None,
                              previous_receipt=None, camera_rebind=None):
    """Build ONE exact executable station from a new measured wrist snapshot."""
    station = _next(protocol, station_index, completed_station_ids)
    reference = observation._frame(frame)
    generation = _camera_generation(protocol, camera_rebind)
    if (reference['camera_generation'] != generation
            or geometry_revision(grasp.model) != protocol['geometry_revision']):
        raise ValueError('相機或幾何版本已改變，這份工程協定不能繼續。')
    if grasp.state not in ('idle', 'approaching') or grasp.robot.pwm_owner is not None:
        raise ValueError('先完成夾爪目前的工作，再移動工程視角。')
    if type(ttl_seconds) not in (int, float) or not math.isfinite(ttl_seconds) or not 0 < ttl_seconds <= MAX_TTL_SECONDS:
        raise ValueError('工程候選有效時間必須在 0–120 秒。')
    target, previous_hash = _station_start_and_target(protocol, station_index,
        reference['joints'], previous_receipt, generation)
    _origin_guard(grasp.model, protocol['origin'], reference['joints'])
    held = observation._pose(grasp.robot.goals)
    segment = _segment(grasp.model, reference['joints'], held,
                       target, protocol['origin'])
    board = protocol['camera_profile']['validation_scope']['board_plane_base']
    board_source = 'parent_profile_board_plane_prior'
    if board_reference is not None:
        _validate(board_reference, 'engineering_workspace_board_reference')
        if (board_reference['camera_generation'] != reference['camera_generation']
                or board_reference['camera_profile_sha256'] != protocol['camera_profile_sha256']
                or board_reference['geometry_revision'] != protocol['geometry_revision']):
            raise ValueError('棋盤重投影資料與工程協定不一致。')
        board = board_reference['board_transform_base']
        board_source = 'wrist_board_frame:' + board_reference['frame_id']
    projection = _projection(grasp.model, target, protocol['camera_profile'], board)
    issued = time.monotonic()
    return _seal({'version': VERSION, 'kind': 'engineering_workspace_station',
        'protocol_sha256': protocol['contract_sha256'], 'session_id': protocol['session_id'],
        'station_id': station['station_id'], 'station_index': station_index,
        'capture_id': station['capture_id'], 'capture_required': station['capture_required'],
        'split': station['split'], 'origin': deepcopy(protocol['origin']),
        'original_nominal_target': deepcopy(station['target_joints']),
        'resolved_target_joints': target, 'previous_receipt_sha256': previous_hash,
        'camera_rebind_sha256': None if camera_rebind is None else _hash(camera_rebind),
        'reference': reference, 'held_goals': held, 'segment': segment,
        'issued_monotonic': issued, 'expires_monotonic': issued + ttl_seconds,
        'board_projection': projection, 'board_projection_source': board_source,
        'framing_is_not_obstacle_clearance': True, 'requires_explicit_path_review': True,
        'camera_change_model': observation._camera_change(grasp.model, reference['joints'], target)})


def describe_workspace_station(candidate):
    _validate(candidate, 'engineering_workspace_station')
    return deepcopy(candidate)


def validate_workspace_station(grasp, protocol, candidate, reference, *,
                               session_id, completed_station_ids, approval,
                               previous_receipt=None, camera_rebind=None):
    """Read-only preflight before the caller journals an in-flight attempt.

    No motor, capture, goal, state or journal mutation is performed. Execution
    repeats this validation to catch changes between preflight and dispatch.
    """
    _validate(candidate, 'engineering_workspace_station')
    station = _next(protocol, candidate['station_index'], completed_station_ids)
    observation._frame(reference)
    generation = _camera_generation(protocol, camera_rebind)
    target, previous_hash = _station_start_and_target(protocol, candidate['station_index'],
        candidate['reference']['joints'], previous_receipt, generation)
    if (candidate['protocol_sha256'] != protocol['contract_sha256']
            or candidate['station_id'] != station['station_id']
            or candidate['session_id'] != session_id or protocol['session_id'] != session_id
            or candidate.get('original_nominal_target') != station['target_joints']
            or candidate.get('resolved_target_joints') != target
            or candidate['segment']['joint_targets'] != target
            or candidate.get('previous_receipt_sha256') != previous_hash
            or candidate.get('camera_rebind_sha256') != (None if camera_rebind is None else _hash(camera_rebind))
            or candidate['origin'] != protocol['origin']
            or candidate['segment']['fixed_origin'] != protocol['origin']
            or protocol['origin_limit_profile'] != WORKSPACE_ORIGIN_PROFILE
            or candidate['segment'].get('origin_limit_profile') != WORKSPACE_ORIGIN_PROFILE):
        raise ValueError('本次工程動作不屬於已核對的完整協定。')
    if (geometry_revision(grasp.model) != protocol['geometry_revision']
            or reference['camera_generation'] != generation
            or dict(grasp.robot.goals) != candidate['held_goals']):
        raise ValueError('模型、相機或保持目標已改變，請重新核對下一站。')
    if not candidate['issued_monotonic'] <= time.monotonic() <= candidate['expires_monotonic']:
        raise ValueError('工程候選已過期，請重新取得畫面。')
    if (not isinstance(approval, Mapping) or any(approval.get(k) is not True for k in
            ('approved', 'gripper_empty', 'human_clear', 'clear_path'))
            or approval.get('protocol_sha256') != protocol['contract_sha256']
            or approval.get('candidate_sha256') != candidate['contract_sha256']
            or approval.get('frame_id') != candidate['reference']['id']
            or not isinstance(approval.get('reason'), str) or not 1 <= len(approval['reason'].strip()) <= 240):
        raise ValueError('需要核對這張腕部畫面、空夾爪與這一站的完整路徑。')
    if grasp.state not in ('idle', 'approaching') or grasp.robot.pwm_owner is not None:
        raise ValueError('夾爪還有未完成的工作，不能移動工程視角。')
    from .grasp import unchanged_wrist
    unchanged_wrist(candidate['reference'], reference)
    return station


def execute_workspace_station(grasp, protocol, candidate, reference, cancel, report, *,
                              session_id, completed_station_ids, approval,
                              previous_receipt=None, camera_rebind=None):
    """Run exactly one reviewed segment. The caller journals it and captures."""
    observation._cancel(cancel)
    station = validate_workspace_station(grasp, protocol, candidate, reference,
        session_id=session_id, completed_station_ids=completed_station_ids, approval=approval,
        previous_receipt=previous_receipt, camera_rebind=camera_rebind)
    grasp.record({'type': 'engineering_workspace_station_approved',
        'protocol_sha256': protocol['contract_sha256'], 'station_id': station['station_id'],
        'origin': protocol['origin'], 'approval': dict(approval)})
    result = execute_prepared_segment(grasp, candidate['segment'], candidate['reference'], cancel,
        report, speed=60, engineering_workspace=True,
        measured_guard=lambda actual: _origin_guard(grasp.model, protocol['origin'], actual))
    actual = result['after']['joints']
    reached_original = all(abs(actual[n] - station['target_joints'][n]) <=
        (4 if n == 'gripper' else ENDPOINT_TOLERANCE) for n in ALL)
    reached_resolved = all(abs(actual[n] - candidate['resolved_target_joints'][n]) <=
        (4 if n == 'gripper' else ENDPOINT_TOLERANCE) for n in ALL)
    return {**result, 'completed': True, 'station_id': station['station_id'],
        'original_nominal_target': deepcopy(station['target_joints']),
        'resolved_target_joints': deepcopy(candidate['resolved_target_joints']),
        'previous_receipt_sha256': candidate['previous_receipt_sha256'],
        'reached_nominal_endpoint': reached_original, 'reached_resolved_endpoint': reached_resolved,
        'protocol_sha256': protocol['contract_sha256'], 'capture_id': station['capture_id'],
        'capture_required': station['capture_required'], 'split': station['split'],
        'calibration_sample_recorded': False, 'next_station_executed': False}
