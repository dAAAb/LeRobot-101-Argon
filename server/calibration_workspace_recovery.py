"""Rebuild one interrupted first survey station without resetting its origin.

Pure preparation: no device, file, journal, camera capture, or motor I/O.
The caller preserves the parent journal and reviews each new station against
a fresh wrist image. No old path approval transfers to the recovery protocol.
"""
from collections.abc import Mapping
from copy import deepcopy
import math

from . import calibration_workspace_motion as workspace
from .local_cartesian import ALL, NAMES
from .motion_program import _tube, WORKSPACE_ORIGIN_PROFILE, WORKSPACE_ORIGIN_DELTA_BOUNDS
from .spatial_runtime import geometry_revision


def build_workspace_recovery_protocol(grasp, parent_record, failed_candidate,
                                     failure_evidence, fresh_frame, *, session_id,
                                     camera_profile, camera_id, mount_revision):
    """Add an elbow-only prelude, then retain all seventeen frozen targets.

    This narrowly repairs the recorded first-station interruption. Later
    failures, already recovered protocols, missing in-flight evidence, and
    changed camera/arm calibration require a separately reviewed protocol.
    A restarted camera generation is allowed only in the NEW reference; the
    parent reference, failed candidate, and failure are retained by hash and
    complete copies for the caller's immutable journal archive.
    """
    if not isinstance(parent_record, Mapping):
        raise ValueError('需要原始工程協定及未完成移動紀錄。')
    parent = parent_record.get('protocol')
    workspace._validate(parent, 'engineering_workspace_calibration')
    if (parent.get('recovery') is not None or parent_record.get('completed_station_ids') != []
            or parent_record.get('receipts') != [] or parent_record.get('captures') != []
            or parent_record.get('session_id') != parent.get('session_id')):
        raise ValueError('此復原只適用尚未完成第一站的原始協定，不能再次重設或跳站。')
    if not isinstance(session_id, str) or not session_id or len(session_id) > 160:
        raise ValueError('需要有效的復原工作階段識別。')
    stations = parent.get('stations')
    if not isinstance(stations, list) or len(stations) != 17 or stations[0].get('station_id') != 'station-01':
        raise ValueError('此復原需要原先固定的十七站校準協定。')
    if not isinstance(failed_candidate, Mapping):
        raise ValueError('需要原先送出但未完成的候選紀錄。')
    # The API response appends these two display/lookup fields after sealing.
    candidate = {k: deepcopy(v) for k,v in failed_candidate.items() if k not in ('candidate_id','frame')}
    workspace._validate(candidate, 'engineering_workspace_station')
    inflight = parent_record.get('inflight')
    if (not isinstance(inflight, Mapping) or inflight.get('station_id') != 'station-01'
            or not isinstance(inflight.get('candidate_id'),str) or not inflight['candidate_id']
            or failed_candidate.get('candidate_id') != inflight['candidate_id']
            or candidate.get('station_id') != 'station-01' or candidate.get('station_index') != 0
            or candidate.get('session_id') != parent['session_id']
            or candidate.get('protocol_sha256') != parent['contract_sha256']
            or candidate.get('origin') != parent['origin']
            or candidate['segment'].get('fixed_origin') != parent['origin']
            or candidate['segment'].get('joint_targets') != stations[0]['target_joints']):
        raise ValueError('失敗候選與原始第一站及未完成紀錄不一致。')
    profile_hash = workspace._profile(camera_profile,grasp.model,camera_id,mount_revision)
    if (profile_hash != parent['camera_profile_sha256']
            or geometry_revision(grasp.model) != parent['geometry_revision']
            or parent['origin_limit_profile'] != WORKSPACE_ORIGIN_PROFILE
            or parent['joint_delta_bounds_from_original_origin'] !=
                {n:list(b) for n,b in WORKSPACE_ORIGIN_DELTA_BOUNDS.items()}
            or parent['maximum_tip_distance_from_original_origin_mm'] != workspace.MAX_ORIGIN_TIP_MM):
        raise ValueError('復原不能更換相機校準、馬達幾何或放寬原始總範圍。')
    if (not isinstance(failure_evidence, Mapping) or not isinstance(failure_evidence.get('error'),str)
            or not failure_evidence['error'].strip() or type(failure_evidence.get('time')) not in (int,float)
            or not math.isfinite(failure_evidence['time']) or failure_evidence['time'] <= 0):
        raise ValueError('需要原先控制器停止時的失敗證據。')
    failed_actual = workspace.observation._pose(failure_evidence.get('actual'))
    failed_goals = workspace.observation._pose(failure_evidence.get('goals'))
    old_segment = candidate['segment']
    _tube(failed_actual,old_segment['measured_origin'],old_segment['joint_targets'],
          old_segment['actual_tracking_envelope_ticks'],source='workspace_recovery_failure')
    for n in NAMES:
        lo,hi = old_segment['raw_bounds'][n]
        if not lo <= failed_goals[n] <= hi:
            raise ValueError('失敗時的保持指令不屬於原先候選。')
    reference = workspace.observation._frame(fresh_frame)
    if not reference.get('id') or fresh_frame.get('image_size') != [1920,1080]:
        raise ValueError('復原需要新的原生腕部影像識別。')
    raw = workspace.observation._pose(reference['joints'])
    if (grasp.state not in ('idle','approaching') or grasp.robot.pwm_owner is not None
            or getattr(grasp.robot,'moving',False)):
        raise ValueError('先保持手臂及空夾爪，才能建立復原路徑。')
    for n in ALL:
        tolerance = 4 if n=='gripper' else workspace.STATIONARY_TOLERANCE
        if abs(raw[n]-failed_actual[n]) > tolerance:
            raise ValueError('目前姿勢已離開失敗後的保持範圍，不能沿用這次復原。')
    if abs(failed_actual['gripper']-parent['origin']['gripper']) > 4 or abs(failed_goals['gripper']-parent['origin']['gripper']) > 4:
        raise ValueError('失敗紀錄的夾爪開口已改變。')
    workspace._origin_guard(grasp.model,parent['origin'],raw)
    held = workspace.observation._pose(grasp.robot.goals)
    prelude_target = {**raw,'elbow_flex':stations[0]['target_joints']['elbow_flex']}
    if raw['elbow_flex']-prelude_target['elbow_flex'] <= workspace.ENDPOINT_TOLERANCE:
        raise ValueError('這次姿勢不需要原先設計的單肘退開，請重新核對原因。')
    templates = [dict(station_id='recovery-elbow-01',station_index=0,
        target_joints=prelude_target,capture_id=None,split=None,capture_required=False)] + deepcopy(stations)
    nominal,predicted_held,new_stations = dict(raw),dict(held),[]
    for index,template in enumerate(templates):
        target = template['target_joints']
        segment = workspace._segment(grasp.model,nominal,predicted_held,target,parent['origin'])
        new_stations.append({**template,'station_index':index,'nominal_segment':segment,
            'nominal_board_projection':workspace._projection(grasp.model,target,camera_profile,
                camera_profile['validation_scope']['board_plane_base']),
            'nominal_projection_source':'parent_profile_board_plane_prior',
            'external_clearance_verified':False})
        nominal,predicted_held = target,segment['physical_goal_targets']
    if dict(grasp.robot.goals) != held:
        raise ValueError('建立復原路徑期间保持目標改變。')
    result = deepcopy(parent)
    result.update(session_id=session_id,reference=reference,initial_held_goals=held,stations=new_stations,
        recovery={'kind':'first_station_single_elbow_prelude','parent_session_id':parent['session_id'],
            'parent_protocol_sha256':parent['contract_sha256'],
            'parent_record_sha256':workspace._hash(parent_record),
            'parent_completed_station_ids':[], 'original_failed_station_id':'station-01',
            'failed_station_id':'station-01','failed_candidate_id':inflight['candidate_id'],
            'original_failed_candidate_id':inflight['candidate_id'],
            'original_failed_candidate_sha256':candidate['contract_sha256'],
            'original_failed_candidate':deepcopy(candidate),
            'failure_evidence_sha256':workspace._hash(failure_evidence),
            'failure_evidence':deepcopy(failure_evidence),
            'recovery_start':dict(raw),'recovery_held_goals':dict(held),
            'original_camera_generation':parent['reference']['camera_generation'],
            'recovery_camera_generation':reference['camera_generation'],
            'frozen_camera_profile_sha256':profile_hash,
            'frozen_onboard_geometry_revision':geometry_revision(grasp.model),
            'onboard_calibration_sha256':workspace._hash(grasp.model.calibration),
            'old_path_approval_reused':False,'original_origin_reset':False,
            'parent_journal_must_be_preserved':True})
    return workspace._seal(result)


def build_workspace_settling_recovery_protocol(grasp, parent_record, failed_candidate,
                                               failure_evidence, fresh_frame, *, session_id,
                                               camera_profile, camera_id, mount_revision):
    """Prepare one final retry of the same elbow target after a settling fix.

    Only the first elbow prelude of the first recovery may be retried. This
    does not execute it or inherit an image/path approval. The original
    survey origin, calibration, bounds and seventeen capture/transfer targets
    remain frozen; all segments are screened again from the measured stop.
    """
    if not isinstance(parent_record, Mapping):
        raise ValueError('需要第一次復原的完整工程紀錄。')
    parent = parent_record.get('protocol')
    workspace._validate(parent, 'engineering_workspace_calibration')
    prior = parent.get('recovery')
    if (not isinstance(prior, Mapping)
            or prior.get('kind') != 'first_station_single_elbow_prelude'
            or prior.get('attempt', 1) != 1
            or parent_record.get('completed_station_ids') != []
            or parent_record.get('receipts') != [] or parent_record.get('captures') != []
            or parent_record.get('session_id') != parent.get('session_id')
            or parent_record.get('recovered_from') != prior.get('parent_protocol_sha256')
            or session_id != parent.get('session_id')):
        raise ValueError('只允許第一次復原的第一站再驗證一次，不能再次重試或跳站。')
    stations = parent.get('stations')
    if (not isinstance(stations, list) or len(stations) != 18
            or stations[0].get('station_id') != 'recovery-elbow-01'
            or stations[0].get('target_joints', {}).get('elbow_flex') != 3000
            or stations[0].get('nominal_segment', {}).get('active_joints') != ['elbow_flex']):
        raise ValueError('需要第一次復原的固定單肘目標及後續十七站。')
    if not isinstance(failed_candidate, Mapping):
        raise ValueError('需要未完成的單肘候選紀錄。')
    candidate = {k: deepcopy(v) for k,v in failed_candidate.items() if k not in ('candidate_id','frame')}
    workspace._validate(candidate, 'engineering_workspace_station')
    inflight = parent_record.get('inflight')
    old_segment = candidate['segment']
    if (not isinstance(inflight, Mapping) or inflight.get('station_id') != 'recovery-elbow-01'
            or not isinstance(inflight.get('candidate_id'), str) or not inflight['candidate_id']
            or failed_candidate.get('candidate_id') != inflight['candidate_id']
            or candidate.get('station_id') != 'recovery-elbow-01' or candidate.get('station_index') != 0
            or candidate.get('session_id') != parent['session_id']
            or candidate.get('protocol_sha256') != parent['contract_sha256']
            or candidate.get('origin') != parent['origin']
            or old_segment.get('fixed_origin') != parent['origin']
            or old_segment.get('joint_targets') != stations[0]['target_joints']
            or old_segment.get('active_joints') != ['elbow_flex']):
        raise ValueError('失敗候選與第一次復原的單肘站不一致。')
    profile_hash = workspace._profile(camera_profile, grasp.model, camera_id, mount_revision)
    if (profile_hash != parent['camera_profile_sha256']
            or geometry_revision(grasp.model) != parent['geometry_revision']
            or workspace._hash(grasp.model.calibration) != prior.get('onboard_calibration_sha256')
            or parent['origin_limit_profile'] != WORKSPACE_ORIGIN_PROFILE
            or parent['joint_delta_bounds_from_original_origin'] !=
                {n:list(b) for n,b in WORKSPACE_ORIGIN_DELTA_BOUNDS.items()}
            or parent['maximum_tip_distance_from_original_origin_mm'] != workspace.MAX_ORIGIN_TIP_MM):
        raise ValueError('復原不能更換相機校準、馬達幾何或放寬原始總範圍。')
    if (not isinstance(failure_evidence, Mapping) or not isinstance(failure_evidence.get('error'), str)
            or not failure_evidence['error'].startswith('未到達目標位置')
            or 'elbow_flex' not in failure_evidence['error']
            or type(failure_evidence.get('time')) not in (int, float)
            or not math.isfinite(failure_evidence['time']) or failure_evidence['time'] <= 0):
        raise ValueError('這次再驗證只適用單肘等待後未到位的失敗。')
    failed_actual = workspace.observation._pose(failure_evidence.get('actual'))
    failed_goals = workspace.observation._pose(failure_evidence.get('goals'))
    _tube(failed_actual, old_segment['measured_origin'], old_segment['joint_targets'],
          old_segment['actual_tracking_envelope_ticks'], source='workspace_settling_failure')
    if (failed_goals != old_segment['physical_goal_targets']
            or failed_actual['elbow_flex'] - stations[0]['target_joints']['elbow_flex']
                <= workspace.ENDPOINT_TOLERANCE):
        raise ValueError('需要已送到原目標、但單肘仍未到位的證據。')
    reference = workspace.observation._frame(fresh_frame)
    if not reference.get('id') or fresh_frame.get('image_size') != [1920,1080]:
        raise ValueError('復原需要新的原生腕部影像識別。')
    raw = workspace.observation._pose(reference['joints'])
    if (grasp.state not in ('idle','approaching') or grasp.robot.pwm_owner is not None
            or getattr(grasp.robot, 'moving', False)):
        raise ValueError('先保持手臂及空夾爪，才能建立復原路徑。')
    for n in ALL:
        tolerance = 4 if n == 'gripper' else workspace.STATIONARY_TOLERANCE
        if abs(raw[n] - failed_actual[n]) > tolerance:
            raise ValueError('目前姿勢已離開失敗後的保持範圍，不能沿用這次復原。')
    if (abs(failed_actual['gripper'] - parent['origin']['gripper']) > 4
            or abs(failed_goals['gripper'] - parent['origin']['gripper']) > 4):
        raise ValueError('失敗紀錄的夾爪開口已改變。')
    workspace._origin_guard(grasp.model, parent['origin'], raw)
    held = workspace.observation._pose(grasp.robot.goals)
    if raw['elbow_flex'] - stations[0]['target_joints']['elbow_flex'] <= workspace.ENDPOINT_TOLERANCE:
        raise ValueError('目前单肘已在目標容許範圍，無需再次移動。')
    templates = deepcopy(stations)
    # Holding joints can settle by a tick without becoming commanded axes.
    # The elbow target and all following targets remain exactly the same.
    templates[0]['target_joints'] = {**raw, 'elbow_flex': stations[0]['target_joints']['elbow_flex']}
    nominal, predicted_held, new_stations = dict(raw), dict(held), []
    for template in templates:
        target = template['target_joints']
        segment = workspace._segment(grasp.model, nominal, predicted_held, target, parent['origin'])
        new_stations.append({**template, 'nominal_segment': segment,
            'nominal_board_projection': workspace._projection(grasp.model, target, camera_profile,
                camera_profile['validation_scope']['board_plane_base']),
            'nominal_projection_source': 'parent_profile_board_plane_prior',
            'external_clearance_verified': False})
        nominal, predicted_held = target, segment['physical_goal_targets']
    if dict(grasp.robot.goals) != held:
        raise ValueError('建立復原路徑期間保持目標改變。')
    result = deepcopy(parent)
    recovery = deepcopy(prior)
    recovery.update(kind='first_station_settling_retry', attempt=2,
        parent_session_id=parent['session_id'], parent_protocol_sha256=parent['contract_sha256'],
        parent_record_sha256=workspace._hash(parent_record),
        root_original_protocol_sha256=prior['parent_protocol_sha256'],
        failed_station_id='recovery-elbow-01', failed_candidate_id=inflight['candidate_id'],
        failed_candidate_sha256=candidate['contract_sha256'], failed_candidate=deepcopy(candidate),
        failure_evidence_sha256=workspace._hash(failure_evidence), failure_evidence=deepcopy(failure_evidence),
        recovery_start=dict(raw), recovery_held_goals=dict(held),
        recovery_camera_generation=reference['camera_generation'],
        old_path_approval_reused=False, original_origin_reset=False, parent_journal_must_be_preserved=True)
    result.update(reference=reference, initial_held_goals=held, stations=new_stations, recovery=recovery)
    return workspace._seal(result)
