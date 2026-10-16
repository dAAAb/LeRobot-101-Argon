"""Resolve a recorded calibration interruption as an observation, never a retry.

Pure evidence validation. The caller supplies saved telemetry and fresh settled
wrist measurements; this module neither reads devices nor dispatches motion.
The original protocol, nominal targets, origin and camera reference stay intact.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
import time

from . import calibration_workspace_motion as workspace
from .local_cartesian import ALL
from .motion_program import _tube
from .spatial_runtime import geometry_revision


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def build_workspace_interrupted_observation(grasp, parent_record, failed_candidate,
                                            failure_evidence, telemetry, fresh_frame, *,
                                            camera_profile, camera_id, mount_revision,
                                            settled_samples, rebind_camera_generation=False):
    """Record the stopped pose after the second prelude missed its endpoint.

    `completed` in the receipt means that this interrupted observation has
    been accounted for; `motion_completed` and `endpoint_reached` are false.
    No third command to the prelude target is generated. A later, different
    station needs its own measured-pose path calculation and fresh review.
    """
    if not isinstance(parent_record, Mapping):
        raise ValueError('需要完整的未完成工程紀錄。')
    protocol = parent_record.get('protocol')
    workspace._validate(protocol, 'engineering_workspace_calibration')
    recovery = protocol.get('recovery', {})
    if (recovery.get('kind') != 'first_station_settling_retry' or recovery.get('attempt') != 2
            or parent_record.get('session_id') != protocol['session_id']
            or parent_record.get('completed_station_ids') != []
            or parent_record.get('receipts') != [] or parent_record.get('captures') != []
            or parent_record.get('recovered_from') != recovery.get('parent_protocol_sha256')):
        raise ValueError('這次核對只適用尚未完成第一站的最後一次單肘驗證。')
    if not isinstance(failed_candidate, Mapping):
        raise ValueError('需要對應的未完成候選。')
    candidate = {k:deepcopy(v) for k,v in failed_candidate.items() if k not in ('candidate_id','frame')}
    workspace._validate(candidate, 'engineering_workspace_station')
    stations = protocol.get('stations')
    inflight = parent_record.get('inflight')
    segment = candidate['segment']
    if (not isinstance(stations, list) or len(stations) != 18
            or stations[0].get('station_id') != 'recovery-elbow-01'
            or not isinstance(inflight, Mapping) or inflight.get('station_id') != 'recovery-elbow-01'
            or not isinstance(inflight.get('candidate_id'), str) or not inflight['candidate_id']
            or failed_candidate.get('candidate_id') != inflight['candidate_id']
            or candidate.get('station_id') != inflight['station_id'] or candidate.get('station_index') != 0
            or candidate.get('session_id') != protocol['session_id']
            or candidate.get('protocol_sha256') != protocol['contract_sha256']
            or candidate.get('origin') != protocol['origin']
            or segment.get('fixed_origin') != protocol['origin']
            or segment.get('joint_targets') != stations[0]['target_joints']
            or segment.get('active_joints') != ['elbow_flex']):
        raise ValueError('未完成紀錄必須對應這次單肘候選及原始協定。')
    profile_hash = workspace._profile(camera_profile, grasp.model, camera_id, mount_revision)
    if (profile_hash != protocol['camera_profile_sha256']
            or geometry_revision(grasp.model) != protocol['geometry_revision']
            or workspace._hash(grasp.model.calibration) != recovery.get('onboard_calibration_sha256')):
        raise ValueError('事故核對不能更換相機校準或馬達內的校準。')
    if (not isinstance(failure_evidence, Mapping)
            or not isinstance(failure_evidence.get('error'), str)
            or not failure_evidence['error'].startswith('未到達目標位置')
            or 'elbow_flex' not in failure_evidence['error']
            or not _finite(failure_evidence.get('time'))):
        raise ValueError('只可核對已送出目標但單肘停在途中之紀錄。')
    failed = workspace.observation._pose(failure_evidence.get('actual'))
    goals = workspace.observation._pose(failure_evidence.get('goals'))
    target = segment['joint_targets']
    start = segment['measured_origin']
    if (goals != segment['physical_goal_targets']
            or start['elbow_flex'] - failed['elbow_flex'] <= 0
            or failed['elbow_flex'] - target['elbow_flex'] <= workspace.ENDPOINT_TOLERANCE):
        raise ValueError('需要實際前進、但尚未到達單肘名目目標的證據。')

    def in_original_path(pose):
        _tube(pose, start, target, segment['actual_tracking_envelope_ticks'],
              source='workspace_interrupted_observation')
        workspace._origin_guard(grasp.model, protocol['origin'], pose)

    in_original_path(failed)
    if not isinstance(telemetry, list) or len(telemetry) < 3:
        raise ValueError('需要這次移動連續的到位等待紀錄。')
    previous_elapsed, previous_time, previous_index = -1., -1., 0
    poses = []
    for row in telemetry:
        if (not isinstance(row, Mapping) or row.get('type') != 'motion_program_endpoint_settling'
                or row.get('kind') != 'endpoint_settling'
                or row.get('contract_sha256') != segment['contract_sha256']
                or not _finite(row.get('elapsed_s')) or row['elapsed_s'] <= previous_elapsed
                or not _finite(row.get('time')) or not previous_time < row['time'] <= failure_evidence['time']
                or type(row.get('sample_index')) is not int or row['sample_index'] != previous_index + 1
                or row.get('logical_targets') != {'elbow_flex':target['elbow_flex']}
                or row.get('goals') != goals):
            raise ValueError('到位等待紀錄不連續或不屬於這次移動。')
        pose = workspace.observation._pose(row.get('actual'))
        if row.get('endpoint_error_ticks') != {'elbow_flex':pose['elbow_flex']-target['elbow_flex']}:
            raise ValueError('到位等待紀錄的實測誤差不符。')
        in_original_path(pose)
        poses.append(pose)
        previous_elapsed, previous_time, previous_index = row['elapsed_s'], row['time'], row['sample_index']
    if poses[-1] != failed or failure_evidence['time'] - telemetry[-1]['time'] > 1.:
        raise ValueError('失敗紀錄與最後一筆等待量測不一致。')
    # Use the last continuous half-second, not one coincidentally repeated
    # encoder value, to distinguish a plateau from an arm still approaching.
    tail_start = len(telemetry)-1
    while tail_start > 0 and telemetry[-1]['elapsed_s']-telemetry[tail_start]['elapsed_s'] < .5:
        tail_start -= 1
    tail = poses[tail_start:]
    if (len(tail) < 3 or telemetry[-1]['elapsed_s']-telemetry[tail_start]['elapsed_s'] < .5
            or any(max(p[n] for p in tail)-min(p[n] for p in tail) > 2 for n in ALL)):
        raise ValueError('等待紀錄尚未顯示手臂停穩，不能記為途中觀察。')
    reference = workspace.observation._frame(fresh_frame)
    if not reference.get('id') or fresh_frame.get('image_size') != [1920,1080]:
        raise ValueError('需要新的原生腕部影像。')
    actual = reference['joints']
    if (grasp.state not in ('idle','approaching') or grasp.robot.pwm_owner is not None
            or getattr(grasp.robot,'moving',False)):
        raise ValueError('先保持手臂及空夾爪，再核對停穩位置。')
    if any(abs(actual[n]-failed[n]) > (4 if n=='gripper' else workspace.STATIONARY_TOLERANCE) for n in ALL):
        raise ValueError('新姿勢已離開失敗後保持範圍，不能沿用這次觀察。')
    in_original_path(actual)
    if not isinstance(settled_samples, list) or len(settled_samples) < 2:
        raise ValueError('需要目前姿勢的連續停穩量測。')
    latest, settled = -1., []
    for sample in settled_samples:
        if (not isinstance(sample, Mapping) or not _finite(sample.get('monotonic'))
                or sample['monotonic'] <= latest):
            raise ValueError('目前停穩量測的時間順序不符。')
        latest = sample['monotonic']
        pose = workspace.observation._pose(sample.get('joints'))
        in_original_path(pose)
        settled.append(pose)
    if (settled_samples[-1]['monotonic']-settled_samples[0]['monotonic'] < .3
            or not 0 <= time.monotonic()-latest <= 5
            or abs(reference['capture_monotonic']-latest) > 5
            or any(max(p[n] for p in settled)-min(p[n] for p in settled) > 2 for n in ALL)
            or any(abs(actual[n]-settled[-1][n]) > 2 for n in ALL)):
        raise ValueError('目前姿勢尚未停穩或量測已過期。')
    held = workspace.observation._pose(grasp.robot.goals)
    rebind = None
    if reference['camera_generation'] != protocol['reference']['camera_generation']:
        if rebind_camera_generation is not True:
            raise ValueError('腕部相機重啟後需要明確綁定同一相機及校準版本。')
        rebind = workspace._seal(dict(version=workspace.VERSION,
            kind='engineering_workspace_camera_rebind', protocol_sha256=protocol['contract_sha256'],
            camera_id=camera_id, mount_revision=mount_revision, camera_profile_sha256=profile_hash,
            from_generation=protocol['reference']['camera_generation'],
            to_generation=reference['camera_generation'],
            original_reference_sha256=workspace._hash(protocol['reference'])))
    receipt = dict(completed=True, completion='stable_partial_observation',
        interrupted_observation_resolved=True, motion_completed=False, endpoint_reached=False,
        nominal_unreached=True, motors_commanded=False, calibration_sample_recorded=False,
        next_station_executed=False, protocol_sha256=protocol['contract_sha256'],
        station_id=candidate['station_id'], candidate_id=inflight['candidate_id'],
        before=deepcopy(candidate['reference']), after=reference,
        actual=dict(actual), held_goals=held, nominal_target=deepcopy(target),
        capture_id=None, capture_required=False, split=None)
    return workspace._seal(dict(version=workspace.VERSION,
        kind='engineering_workspace_interrupted_observation',
        session_id=protocol['session_id'], protocol_sha256=protocol['contract_sha256'],
        parent_record_sha256=workspace._hash(parent_record),
        station_id=candidate['station_id'], candidate_id=inflight['candidate_id'],
        origin=deepcopy(protocol['origin']), original_reference=deepcopy(protocol['reference']),
        camera_profile_sha256=profile_hash, geometry_revision=protocol['geometry_revision'],
        onboard_calibration_sha256=recovery['onboard_calibration_sha256'],
        failed_candidate=deepcopy(failed_candidate), failure_evidence=deepcopy(failure_evidence),
        telemetry=deepcopy(telemetry), settled_samples=deepcopy(settled_samples),
        receipt=receipt, camera_rebind=rebind, motors_commanded=False,
        original_protocol_changed=False, original_origin_reset=False,
        old_path_approval_reused=False, next_station_requires_fresh_review=True))
