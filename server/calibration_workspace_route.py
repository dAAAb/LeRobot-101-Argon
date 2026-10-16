"""Offline elbow-first remaining survey route; no hardware or journal mutation.

The parent remains intact. A child camera session can start from a fresh
measured checkpoint while retaining the ORIGINAL robot excursion origin.
Each resulting station still requires its own fresh wrist/path review.
"""
from copy import deepcopy
from collections.abc import Mapping

from . import calibration_workspace_motion as workspace
from .local_cartesian import ALL, NAMES
from .spatial_runtime import geometry_revision


def decouple_workspace_stations(model, protocol, stations, *, start, held_goals,
                                source_index=0):
    """Screen a finite route with elbow movement separated from other axes.

    Each source target/split is retained. Explicit active axes preserve the
    meaning of a stationary joint even when the measured checkpoint differs
    from its old nominal pose. No endpoint is assumed to be reached in use.
    """
    raw, held = dict(start), dict(held_goals)
    output, mapping = [], []
    board = protocol['camera_profile']['validation_scope']['board_plane_base']
    for index, source in enumerate(stations, source_index):
        original = source['target_joints']
        original_active = source.get('planned_active_joints',
            source['nominal_segment']['active_joints'])
        if (not isinstance(original_active, list) or len(set(original_active)) != len(original_active)
                or any(n not in NAMES for n in original_active)):
            raise ValueError('工程取景的動作軸必須來自原始肩肘腕計畫。')
        elbow_mixed = 'elbow_flex' in original_active and len(original_active) > 1
        parts = [('elbow_prelude', ['elbow_flex']),
                 ('source_target', [n for n in original_active if n != 'elbow_flex'])] if elbow_mixed else [
                     ('source_target', list(original_active))]
        for phase, active in parts:
            prelude = phase == 'elbow_prelude'
            target = {n: original[n] if n in active else raw[n] for n in ALL}
            segment = workspace._segment(model, raw, held, target, protocol['origin'])
            identity = source['station_id'] + '-elbow' if prelude else source['station_id']
            # The original source target remains visible in the manifest;
            # planned_active_joints specifies the exact subset for this stage.
            station = {**deepcopy(source), 'station_id': identity, 'station_index': len(output),
                'target_joints': deepcopy(original), 'planned_active_joints': active,
                'capture_id': None if prelude else source['capture_id'],
                'split': None if prelude else source['split'],
                'capture_required': False if prelude else source['capture_required'],
                'purpose': 'single_elbow_before_body' if prelude else 'remaining_source_view',
                'nominal_segment': segment,
                'nominal_board_projection': workspace._projection(model, target,
                    protocol['camera_profile'], board),
                'nominal_projection_source': 'parent_profile_board_plane_prior',
                'external_clearance_verified': False}
            output.append(station)
            mapping.append(dict(station_id=identity, source_station_id=source['station_id'],
                source_station_index=index, phase=phase,
                original_target_joints=deepcopy(original), planned_active_joints=list(active)))
            raw, held = target, segment['physical_goal_targets']
    if len({s['station_id'] for s in output}) != len(output):
        raise ValueError('剩餘工程路徑的姿勢識別重複。')
    return output, mapping


def build_decoupled_workspace_protocol(grasp, parent_record, frame, *, session_id,
                                      camera_id, mount_revision, reason):
    """Derive a child survey from an interrupted parent and actual checkpoint."""
    if not isinstance(parent_record, Mapping):
        raise ValueError('需要完整原始工程紀錄。')
    parent = parent_record.get('protocol')
    workspace._validate(parent, 'engineering_workspace_calibration')
    if (parent_record.get('session_id') != parent['session_id']
            or not isinstance(session_id, str) or not session_id or len(session_id) > 160
            or session_id == parent['session_id']):
        raise ValueError('剩餘取像必須使用明確的另一個棋盤工作階段。')
    completed = parent_record.get('completed_station_ids')
    if (not isinstance(completed, list) or len(completed) >= len(parent['stations'])
            or completed != [s['station_id'] for s in parent['stations'][:len(completed)]]):
        raise ValueError('原工程紀錄的已完成姿勢不是完整前綴。')
    inflight = parent_record.get('inflight')
    if (not isinstance(inflight, Mapping)
            or inflight.get('station_id') != parent['stations'][len(completed)]['station_id']):
        raise ValueError('檢查點必須接續原工程紀錄尚未完成的姿勢。')
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 500:
        raise ValueError('需要記錄這次明確重新規劃的原因。')
    if grasp.state not in ('idle', 'approaching') or grasp.robot.pwm_owner is not None:
        raise ValueError('先完成夾爪目前的工作，再規劃剩餘工程視角。')
    reference = workspace.observation._frame(frame)
    if (camera_id != parent['camera_id'] or mount_revision != parent['mount_revision']
            or frame.get('camera_id') != camera_id
            or geometry_revision(grasp.model) != parent['geometry_revision']):
        raise ValueError('新的相機工作階段必須維持原相機、安裝與機內校準幾何。')
    profile_hash = workspace._profile(parent['camera_profile'], grasp.model, camera_id, mount_revision)
    if profile_hash != parent['camera_profile_sha256']:
        raise ValueError('不能在接續取景時更換腕部相機校準。')
    raw = workspace.observation._pose(reference['joints'])
    held = workspace.observation._pose(grasp.robot.goals)
    workspace._origin_guard(grasp.model, parent['origin'], raw)
    stations, mapping = decouple_workspace_stations(grasp.model, parent,
        parent['stations'][len(completed):], start=raw, held_goals=held, source_index=len(completed))
    if dict(grasp.robot.goals) != held:
        raise ValueError('重新規劃期間保持目標改變。')
    result = deepcopy(parent)
    result.pop('contract_sha256', None)
    result.pop('recovery', None)
    result.update(session_id=session_id, reference=deepcopy(reference), initial_held_goals=held,
        stations=stations, capture_count=sum(s['capture_required'] for s in stations),
        motors_commanded=False, automatic_station_chaining=False,
        replan=dict(kind='elbow_first_remaining_workspace',
            parent_record_sha256=workspace._hash(parent_record),
            parent_protocol_sha256=parent['contract_sha256'], parent_session_id=parent['session_id'],
            root_original_origin=deepcopy(parent['origin']), completed_station_ids=list(completed),
            checkpoint_frame=deepcopy(reference), checkpoint_start_joints=raw,
            checkpoint_held_goals=held, interrupted_station_id=inflight['station_id'],
            interrupted_candidate_id=inflight.get('candidate_id'), reason=reason.strip(),
            source_station_mapping=mapping, source_camera_generation=workspace._camera_generation(
                parent, parent_record.get('camera_rebind')), checkpoint_camera_generation=reference['camera_generation'],
            camera_id=camera_id, mount_revision=mount_revision, camera_profile_sha256=profile_hash,
            original_captures_retained_in_parent=True, original_path_approval_reused=False))
    return workspace._seal(result)
