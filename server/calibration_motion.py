"""Finite engineering-only viewpoints for wrist/board calibration.

Building or describing a batch does not capture images or command motors.
Execution reuses the sealed observation executor and requires an explicit
review of one exact candidate. Nothing here arms the robot, opens its jaw, or
uses a D435 observation. The calibration session's initial pose stays fixed.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import time

import numpy as np

from . import active_observation as observation
from .gripper_power import SessionCancelled
from .local_cartesian import ALL, RAD_PER_TICK, inward_departure_bounds
from .motion_program import prepare_joint_segment
from .spatial_runtime import geometry_revision

VERSION = 1
# Calibration uses the settled, measured pose rather than the intended pose.
# A roughly 1.4-degree endpoint miss need not discard a useful board view. This
# value also expands the screened motion tube; it is not an after-the-fact
# exception to an already reviewed ten-tick path.
ENDPOINT_TOLERANCE = 16
STATIONARY_TOLERANCE = 12
MAX_TTL_SECONDS = 120.
EXCURSION_PROFILE = observation.CALIBRATION_EXCURSION_PROFILE


def _fingerprint(value):
    return hashlib.sha256(json.dumps(
        {k: v for k, v in value.items() if k != 'contract_sha256'},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()).hexdigest()


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError(f'{label} needs a nonempty identifier')
    return value


def _candidate(grasp, frame, origin, target, candidate_id, label, purpose, *, inward_limit_joints=()):
    # _body_candidate currently hardcodes its object-observation endpoint
    # tolerance. Compose the same common primitives explicitly, without
    # monkeypatching process-wide tolerances during a calibration request.
    segment = prepare_joint_segment(grasp.model, frame['joints'], dict(grasp.robot.goals),
        target, fixed_origin=origin, endpoint_tolerance_ticks=ENDPOINT_TOLERANCE,
        stationary_tolerance_ticks=STATIONARY_TOLERANCE, inward_limit_joints=inward_limit_joints)
    descriptor = {**observation._compact_segment(grasp.model, segment, origin,
                    excursion_profile=EXCURSION_PROFILE),
        'candidate_id': candidate_id, 'kind': 'body', 'label': label,
        'expected_view_change': purpose,
        'purpose': EXCURSION_PROFILE,
        'external_clearance_verified': False,
        'calibration_quality_verified': False,
        'capture_board_separately_after_motion': True}
    before = grasp.model.fk.transforms(frame['joints'])['wrist_camera_optical']
    projections = []
    for fraction in (.25, .5, .75, 1.):
        pose = {n: frame['joints'][n] + fraction * (target[n] - frame['joints'][n]) for n in ALL}
        after = grasp.model.fk.transforms(pose)['wrist_camera_optical']
        projections.append({'fraction': fraction,
            'reference_to_candidate_camera_transform': (np.linalg.inv(after) @ before).tolist()})
    return observation._seal({'id': candidate_id, 'kind': 'body', 'label': label,
        'purpose': EXCURSION_PROFILE, 'excursion_profile': EXCURSION_PROFILE,
        'expected_view_change': purpose, 'origin': dict(origin), 'reference': deepcopy(frame),
        'segment': segment, 'descriptor': descriptor, 'framing_projection_path': projections})


def build_calibration_candidates(grasp, frame, *, origin, session_id,
                                  cancel=None, ttl_seconds=90., rotations_only=False):
    """Return up to 15 current-pose candidates, including BOTH rotation axes.

    ``origin`` must be the session's original measured six-joint pose, never the
    latest pose. Translations use the existing 20–40 mm camera-baseline solver.
    Rotation candidates are available even when translations are feasible;
    multiple nonparallel views, rather than extra same-axis images, identify
    hand-eye parameters. The engineering profile retains a fixed 240-tick /
    120-mm excursion from the original pose, including the tracking reserve,
    while each segment keeps its existing 180-tick / 80-mm and table limits.
    ``rotations_only`` skips translation IK and boundary-departure candidates
    once the board is already framed. Explicit elbow departures diversify the
    actual proximal joint only when its current pose lacks moving reserve.
    """
    observation._cancel(cancel)
    reference = observation._frame(frame)
    _text(reference.get('id'), 'Frame')
    _text(session_id, 'Calibration session')
    initial = observation._pose(origin)
    if type(rotations_only) is not bool:
        raise ValueError('rotations_only must be a boolean')
    if (type(ttl_seconds) not in (int, float) or not math.isfinite(ttl_seconds)
            or not 0 < ttl_seconds <= MAX_TTL_SECONDS):
        raise ValueError('Calibration candidate lifetime must be 0–120 seconds')
    if grasp.state not in ('idle', 'approaching') or grasp.robot.pwm_owner is not None:
        raise ValueError('先完成夾爪目前的工作，再移動校準視角。')
    observation._outer_guard(grasp.model, initial, reference['joints'],
                             excursion_profile=EXCURSION_PROFILE)
    held = observation._pose(grasp.robot.goals)
    candidates, rejected = [], []

    def rejection(candidate_id, exc, **details):
        if isinstance(exc, SessionCancelled):
            raise exc
        rejected.append({'candidate_id': candidate_id, 'reason': str(exc), **details})

    translations = () if rotations_only else (
        ('backward', '後退換距離'), ('left', '向左移動視角'),
        ('right', '向右移動視角'), ('up', '抬高視角'))
    for direction, label in translations:
        candidate_id = 'calibration_move_' + direction
        for distance in (40., 30., 20.):
            observation._cancel(cancel)
            try:
                target = observation._translation_target(grasp, reference['joints'], direction, distance, cancel)
                candidate = _candidate(grasp, reference, initial, target, candidate_id,
                    label, '以不同距離或位置觀察固定棋盤，完成後另行記錄棋盤影像。')
                # The baseline solver checks its endpoint. Preserve the
                # original observer's whole-path no-forward-advance check too.
                before = grasp.model.fk.transforms(reference['joints'])['wrist_camera_optical']
                advance = 0.
                for sample in candidate['segment']['sampled_path']:
                    t = sample['fraction']
                    pose = {n: reference['joints'][n] + t * (target[n] - reference['joints'][n]) for n in ALL}
                    camera = grasp.model.fk.transforms(pose)['wrist_camera_optical']
                    advance = max(advance, float((camera[:3, 3] - before[:3, 3]) @ before[:3, 2]) * 1000)
                if advance > 2:
                    raise ValueError('換位路徑會先朝未知物件方向前進。')
                candidate['descriptor'].update(requested_translation_mm=distance,
                    maximum_nominal_optical_forward_advance_mm=advance)
                observation._seal(candidate)
                candidates.append(candidate)
                break
            except (ValueError, RuntimeError) as exc:
                rejection(candidate_id, exc, requested_translation_mm=distance)

    for joint, word, angles in (('wrist_roll', '旋轉手腕', (-14., -8., 8., 14.)),
                                ('wrist_flex', '俯仰手腕', (-12., -6., 6., 12.))):
        for degrees in angles:
            observation._cancel(cancel)
            ticks = int(round(math.radians(degrees) / RAD_PER_TICK))
            sign = 'positive' if degrees > 0 else 'negative'
            candidate_id = f'calibration_{joint}_{sign}_{abs(int(degrees))}'
            try:
                target = {**reference['joints'], joint: reference['joints'][joint] + ticks}
                candidate = _candidate(grasp, reference, initial, target, candidate_id,
                    f'{word} {degrees:+g}°',
                    '提供另一個相機轉軸；棋盤需保持不動，完成後另行記錄影像。')
                candidate['descriptor'].update(rotation_joint=joint,
                    requested_rotation_degrees=degrees,
                    nominal_rotation_degrees=ticks * 360 / 4096,
                    information_goal='nonparallel rotation axes for hand-eye calibration')
                observation._seal(candidate)
                candidates.append(candidate)
            except (ValueError, RuntimeError) as exc:
                rejection(candidate_id, exc)
    if not rotations_only:
        # These are three alternatives from ONE measured image/pose, never a
        # cumulative command sequence. A regular interior joint gets none.
        joint = 'elbow_flex'
        envelope = max(ENDPOINT_TOLERANCE, abs(held[joint] - reference['joints'][joint]) + 6)
        direction = None
        for sign in (-1, 1):
            try:
                probe = {**reference['joints'], joint: reference['joints'][joint] + sign * 126}
                inward_departure_bounds(grasp.model, reference['joints'], probe,
                    {n: envelope if n == joint else 0 for n in ALL}, [joint])
            except ValueError:
                continue
            direction = sign
            break
        if direction is not None:
            for distance in (42, 84, 126):
                observation._cancel(cancel)
                candidate_id = f'calibration_elbow_inward_{distance}'
                try:
                    target = {**reference['joints'], joint: reference['joints'][joint] + direction * distance}
                    candidate = _candidate(grasp, reference, initial, target, candidate_id,
                        f'手肘向內退開 {distance * 360 / 4096:.1f}°',
                        '手肘離開行程端點，整個前臂與腕部相機會一起轉動；用新姿勢看固定棋盤，完成後另行記錄實際角度與影像。',
                        inward_limit_joints=[joint])
                    candidate['descriptor'].update(
                        inward_limit_joints=[joint],
                        inward_limit_departures=deepcopy(candidate['segment']['inward_limit_departures']),
                        inward_limit_rule=candidate['segment']['inward_limit_rule'],
                        requested_inward_ticks=distance, active_joints=[joint],
                        viewpoint_scope='proximal elbow rotates the entire downstream forearm, wrist and camera; not an optical-only rotation',
                        information_goal='extend independently observed elbow poses without editing calibration bounds')
                    observation._seal(candidate)
                    candidates.append(candidate)
                except (ValueError, RuntimeError) as exc:
                    rejection(candidate_id, exc, requested_inward_ticks=distance)
    if dict(grasp.robot.goals) != held:
        raise ValueError('產生校準候選期間保持目標改變，請重新取得候選。')
    # Timestamp is issued AFTER local screening; TTL does not extend image
    # freshness. Dispatch always requires a fresh image + unchanged-scene test.
    issued = time.monotonic()
    batch = {'version': VERSION, 'kind': 'engineering_calibration_motion',
        'session_id': session_id, 'origin': initial, 'rotations_only': rotations_only,
        'reference': reference, 'held_goals': held,
        'geometry_revision': geometry_revision(grasp.model),
        'issued_monotonic': issued, 'expires_monotonic': issued + ttl_seconds,
        'candidates': candidates, 'rejections': rejected,
        'calibration_samples_captured': False,
        'requires_explicit_path_review': True}
    batch['contract_sha256'] = _fingerprint(batch)
    return batch


def describe_calibration_candidates(batch):
    """JSON for the engineering page; no preview can authorize execution."""
    _validate_batch(batch)
    return {'version': VERSION, 'session_id': batch['session_id'],
        'rotations_only': batch.get('rotations_only', False),
        'batch_sha256': batch['contract_sha256'], 'frame_id': batch['reference']['id'],
        'camera_generation': batch['reference']['camera_generation'],
        'expires_monotonic': batch['expires_monotonic'],
        'origin': deepcopy(batch['origin']), 'held_goals': deepcopy(batch['held_goals']),
        'geometry_revision': batch['geometry_revision'],
        'requires_explicit_path_review': True,
        'candidates': [{**deepcopy(c['descriptor']), 'candidate_sha256': c['contract_sha256']}
                       for c in batch['candidates']], 'rejections': deepcopy(batch['rejections'])}


def _validate_batch(batch):
    if (not isinstance(batch, Mapping) or batch.get('version') != VERSION
            or batch.get('kind') != 'engineering_calibration_motion'
            or batch.get('contract_sha256') != _fingerprint(batch)):
        raise ValueError('校準候選內容已改變，請重新產生。')


def execute_calibration_candidate(grasp, batch, candidate_id, reference, cancel,
                                   report, *, session_id, approval, deadline=None):
    """Execute only an exact, explicitly reviewed engineering candidate.

    approval fields: approved, batch_sha256, candidate_sha256, frame_id,
    gripper_empty, human_clear, clear_path, reason. All four booleans must be
    true; hashes/frame identify the reviewed path and image, not merely a label.
    A fresh ``reference`` is still compared against that reviewed image by the
    existing executor. The caller owns the live calibration-session identity.
    """
    observation._cancel(cancel)
    _validate_batch(batch)
    if batch['session_id'] != session_id:
        raise ValueError('這份候選不屬於目前校準工作。')
    now = time.monotonic()
    if not batch['issued_monotonic'] <= now <= batch['expires_monotonic']:
        raise ValueError('校準候選已過期，請重新取得目前視角。')
    if geometry_revision(grasp.model) != batch['geometry_revision']:
        raise ValueError('機器幾何或馬達校準已改變，請重新取得候選。')
    if dict(grasp.robot.goals) != batch['held_goals']:
        raise ValueError('保持目標已改變，請重新取得候選。')
    selected = next((c for c in batch['candidates'] if c['id'] == candidate_id), None)
    if selected is None:
        raise ValueError('只能執行這份候選中已檢查的動作。')
    if not isinstance(approval, Mapping) or any(approval.get(k) is not True
            for k in ('approved', 'gripper_empty', 'human_clear', 'clear_path')):
        raise ValueError('先檢查夾爪為空、沒有人在路徑上，並確認這條完整校準路徑。')
    if (approval.get('batch_sha256') != batch['contract_sha256']
            or approval.get('candidate_sha256') != selected['contract_sha256']
            or approval.get('frame_id') != batch['reference']['id']):
        raise ValueError('確認內容與候選路徑或腕部畫面不一致。')
    reason = approval.get('reason')
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 240:
        raise ValueError('需要這條工程路徑的簡短檢查記錄。')
    # This is an explicit engineering/operator attestation, not a fabricated
    # VLM confidence score. ObservationVerdict uses 1.0 for affirmative review.
    verdict = {'candidate_id': selected['id'], 'gripper_empty': True,
        'human_clear': True, 'clear_path': True, 'confidence': 1., 'reason': reason}
    grasp.record({'type': 'engineering_calibration_motion_approved',
                  'session_id': session_id, 'candidate_id': candidate_id,
                  'approval_source': 'explicit_engineering_path_review',
                  'batch_sha256': batch['contract_sha256'], 'approval': dict(approval)})
    result = observation.execute_observation(grasp, selected, reference, cancel, report,
                                             verdict=verdict, deadline=deadline,
                                             engineering_calibration=True)
    return {**result, 'session_id': session_id,
            'batch_sha256': batch['contract_sha256'],
            'calibration_sample_recorded': False,
            'next_step': 'capture_board_from_settled_measured_pose'}
