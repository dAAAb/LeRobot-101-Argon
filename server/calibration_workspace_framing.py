"""Four finite wrist-only board views after a completed workspace survey.

Offline planning only. The latest validated board plane predicts framing;
it does not replace the active camera calibration or prove obstacle clearance.
"""
from copy import deepcopy
from collections.abc import Mapping

from . import calibration_workspace_motion as workspace
from .spatial_runtime import geometry_revision


WRIST_FRAMING_OFFSETS = ((150, -50), (160, 50), (150, -10), (150, 100))
BOARD_BOTTOM_LIMIT_PX = 650.


def build_wrist_framing_addendum(grasp, parent_record, frame, *, extension_report,
                                camera_id, mount_revision):
    """Screen four reviewed-once-at-a-time captures using the original origin."""
    if not isinstance(parent_record, Mapping):
        raise ValueError('需要完成的工程取景紀錄。')
    parent = parent_record.get('protocol')
    workspace._validate(parent, 'engineering_workspace_calibration')
    if (parent_record.get('session_id') != parent['session_id']
            or parent_record.get('inflight') is not None
            or parent_record.get('completed_station_ids') != [s['station_id'] for s in parent['stations']]
            or 'addendum' in parent or parent_record.get('previous_complete_record')):
        raise ValueError('這次補拍只接續已完成、尚未補拍的工程取景。')
    reference = workspace.observation._frame(frame)
    if (reference['camera_generation'] != workspace._camera_generation(parent, parent_record.get('camera_rebind'))
            or camera_id != parent['camera_id'] or frame.get('camera_id') != camera_id
            or mount_revision != parent['mount_revision']
            or geometry_revision(grasp.model) != parent['geometry_revision']):
        raise ValueError('補拍必須維持同一腕部相機工作階段、安裝與機內幾何。')
    profile_hash = workspace._profile(parent['camera_profile'], grasp.model, camera_id, mount_revision)
    if profile_hash != parent['camera_profile_sha256']:
        raise ValueError('補拍期間不能更換相機校準。')
    if grasp.state not in ('idle', 'approaching') or grasp.robot.pwm_owner is not None:
        raise ValueError('先完成夾爪目前的工作，再補拍棋盤。')
    report = extension_report
    if (not isinstance(report, Mapping) or report.get('status') != 'validated'
            or report.get('source') != 'wrist-only'
            or report.get('method') != 'frozen-camera-workspace-extension'
            or report.get('board_session_id') != parent['session_id']):
        raise ValueError('取景預測需要這批棋盤剛驗證的固定相機報告。')
    candidate_profile = report.get('calibration_profile', {})
    for key in ('camera_id', 'mount_revision', 'geometry_revision', 'native_image_size',
                'camera_matrix', 'distortion', 'handeye'):
        if candidate_profile.get(key) != parent['camera_profile'].get(key):
            raise ValueError('取景報告更動了原相機或幾何參數：' + key)
    board = report['validation']['board_plane_base']
    raw = workspace.observation._pose(reference['joints'])
    held = workspace.observation._pose(grasp.robot.goals)
    initial_held = dict(held)
    checkpoint = dict(raw)
    workspace._origin_guard(grasp.model, parent['origin'], raw)
    stations = []
    for index, (flex, roll) in enumerate(WRIST_FRAMING_OFFSETS):
        target = {**checkpoint, 'wrist_flex': checkpoint['wrist_flex'] + flex,
                  'wrist_roll': checkpoint['wrist_roll'] + roll}
        active = [n for n in ('wrist_flex', 'wrist_roll') if target[n] != raw[n]]
        segment = workspace._segment(grasp.model, raw, held, target, parent['origin'])
        projection = workspace._projection(grasp.model, target, parent['camera_profile'], board)
        if projection['visible_corner_count'] != 54 or projection['bbox_px'][1][1] > BOARD_BOTTOM_LIMIT_PX:
            raise ValueError('這組腕部補拍尚未把完整棋盤移到指尖上方，需重新取景規劃。')
        stations.append(dict(station_id=f'framing-{index + 1:02d}', station_index=index,
            target_joints=target, planned_active_joints=active,
            capture_id=f'framing-capture-{index + 1:02d}', capture_required=True,
            split='training' if index % 2 == 0 else 'heldout',
            purpose='move_board_above_fingertip_image_region', nominal_segment=segment,
            nominal_board_projection=projection,
            nominal_projection_source='latest_validated_session_board_plane_framing_only',
            external_clearance_verified=False))
        raw, held = target, segment['physical_goal_targets']
    if dict(grasp.robot.goals) != initial_held:
        raise ValueError('補拍規劃期間保持目標改變。')
    result = deepcopy(parent)
    result.pop('contract_sha256', None)
    result.pop('recovery', None)
    result.update(reference=deepcopy(reference), initial_held_goals=initial_held,
        stations=stations, capture_count=4, automatic_station_chaining=False, motors_commanded=False,
        replan=dict(checkpoint_start_joints=checkpoint, checkpoint_frame=deepcopy(reference),
                    checkpoint_held_goals=initial_held),
        addendum=dict(kind='completed_survey_wrist_addendum',
            parent_record_sha256=workspace._hash(parent_record), parent_protocol_sha256=parent['contract_sha256'],
            parent_session_id=parent['session_id'], framing_report_sha256=workspace._hash(report),
            framing_board_plane_base=deepcopy(board), framing_only=True,
            original_calibration_profile_preserved=True, offsets_from_checkpoint=[list(p) for p in WRIST_FRAMING_OFFSETS],
            board_bottom_limit_px=BOARD_BOTTOM_LIMIT_PX,
            predicted_visibility_not_actual_detection=True))
    return workspace._seal(result)
