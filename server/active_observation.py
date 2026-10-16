"""Finite, wrist-only observation motions; geometry is not visual clearance.

Candidate creation performs no capture, model inference or motor I/O. The
caller asks Astra to select and assess one complete candidate, executes that
same contract, then uses the new view to continue the original object task.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .gripper_power import SessionCancelled
from .local_cartesian import ALL, NAMES, RAD_PER_TICK
from .motion_policy import LOCAL_MOVE_MIN_CONFIDENCE, EMPTY_JAW_OPEN_MIN_CONFIDENCE
from .motion_program import prepare_joint_segment, execute_prepared_segment
from .jaw_opening import empty_opening_contract


MAX_ORIGIN_TICKS = 240
MAX_ORIGIN_TIP_MM = 80.
CALIBRATION_EXCURSION_PROFILE = 'engineering_board_calibration'
CALIBRATION_MAX_ORIGIN_TIP_MM = 120.
MAX_EXECUTION_SECONDS = 30.
JAW_OPEN_TARGET = 1860  # Same exercised raw target as WristGrasp.open_empty.
JAW_TOLERANCE = 6
BODY_DRIFT = 4
OBSERVATION_ENDPOINT_TOLERANCE = 20
OBSERVATION_STATIONARY_TOLERANCE = 12
MINIMUM_OBSERVATION_PROGRESS = .4
METRIC_VIEW_IDS = ('move_backward', 'move_left', 'move_right', 'move_up')
MIN_CAMERA_BASELINE_MM = 15.
FRAMING_DEPTH_SAMPLES_M = (.12, .25, .6, 6.)
FRAMING_FOCAL_WIDTH_SAMPLES = (.6375, .75, .8625)
MIN_PREDICTED_ROI_VISIBILITY = .85
DUPLICATE_VIEW_TRANSLATION_MM = 5.
DUPLICATE_VIEW_ROTATION_DEGREES = 2.
MAX_DIRECTION_VISITS = 3
MIN_RECENTER_IMPROVEMENT = .03
TRANSLATION_RECOVERY_ID = 'prepare_translation_pitch'
TRANSLATION_RECOVERY_CONTINUE_ID = TRANSLATION_RECOVERY_ID + '_continue_2'


class ObservationVerdict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    candidate_id: str = Field(min_length=1, max_length=80)
    gripper_empty: bool
    human_clear: bool
    clear_path: bool
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=240)


def _pose(value):
    if (not isinstance(value, Mapping) or set(value) != set(ALL)
            or any(type(v) is not int or not 0 <= v <= 4095 for v in value.values())):
        raise ValueError('補看需要完整的六軸實測位置')
    return {n: value[n] for n in ALL}


def _frame(frame):
    if not isinstance(frame, Mapping) or frame.get('source') != 'wrist':
        raise ValueError('補看只使用腕部相機')
    if (not isinstance(frame.get('camera_generation'), str) or not frame['camera_generation']
            or not isinstance(frame.get('path'), str) or not frame['path']):
        raise ValueError('補看需要目前腕部影像與相機身分')
    stamp = frame.get('capture_monotonic')
    if (type(stamp) not in (int, float) or not math.isfinite(stamp)
            or not 0 <= time.monotonic() - stamp <= 5):
        raise ValueError('腕部照片已過期，請重新取像')
    result = {'source': 'wrist', 'path': frame['path'],
              'camera_generation': frame['camera_generation'],
              'capture_monotonic': stamp, 'joints': _pose(frame.get('joints'))}
    if isinstance(frame.get('id'), str) and frame['id']:
        result['id'] = frame['id']
    return result


def _cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise SessionCancelled('補看已停止')


def _fingerprint(candidate):
    data = {k: v for k, v in candidate.items() if k != 'contract_sha256'}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _seal(candidate):
    candidate['contract_sha256'] = _fingerprint(candidate)
    return candidate


def _origin_tip_limit(excursion_profile=None):
    if excursion_profile is None:
        return MAX_ORIGIN_TIP_MM
    if excursion_profile == CALIBRATION_EXCURSION_PROFILE:
        return CALIBRATION_MAX_ORIGIN_TIP_MM
    raise ValueError('未知的取景工作範圍')


def _outer_guard(model, origin, pose, *, excursion_profile=None):
    if any(abs(pose[n] - origin[n]) > MAX_ORIGIN_TICKS for n in NAMES):
        raise RuntimeError('補看已到這次開始姿勢的關節範圍')
    distance = float(np.linalg.norm(model.tip(pose)[:3, 3] - model.tip(origin)[:3, 3]) * 1000)
    if not math.isfinite(distance) or distance > _origin_tip_limit(excursion_profile):
        raise RuntimeError('補看已到這次開始姿勢的移動範圍')


def _tracking_reserve(model, envelope):
    radii = model.radius_bounds('gripper_frame_link', np.zeros((1, 3)))
    return sum(2 * radius * math.sin((envelope[n] + .5) * RAD_PER_TICK / 2)
               for n, radius in radii.items()) * 1000


def _camera_change(model, before, after):
    """CAD camera motion, distinct from measured object depth or information."""
    a = model.fk.transforms(before)['wrist_camera_optical']
    b = model.fk.transforms(after)['wrist_camera_optical']
    delta = (b[:3, 3] - a[:3, 3]) * 1000
    optical = a[:3, :3].T @ delta
    return {
        'position_delta_base_mm': np.round(delta, 3).tolist(),
        'baseline_mm': round(float(np.linalg.norm(delta)), 3),
        'translation_in_initial_camera_axes_mm': np.round(optical, 3).tolist(),
        'optical_forward_translation_mm': round(float(optical[2]), 3),
        'transverse_baseline_mm': round(float(np.linalg.norm(optical[:2])), 3),
        'rotation_degrees': round(float(np.degrees(Rotation.from_matrix(
            b[:3, :3] @ a[:3, :3].T).magnitude())), 3),
        'before_camera_transform_base_m': a.tolist(),
        'after_camera_transform_base_m': b.tolist(),
        'before_optical_axis_base': a[:3, 2].tolist(),
        'after_optical_axis_base': b[:3, 2].tolist(),
        'source': 'CAD_prior_not_calibrated_object_depth',
    }


def _framing_context(target_bbox, image_size, frame):
    """Optional current-image localization; never infer a metric object size."""
    if target_bbox is None:
        return None
    if (not isinstance(target_bbox, Mapping) or set(target_bbox) != {'x', 'y', 'w', 'h'}
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   for v in target_bbox.values())):
        raise ValueError('換位構圖需要目前影像內有效的物件框')
    x, y, w, h = (target_bbox[n] for n in ('x', 'y', 'w', 'h'))
    if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0
            and x + w <= 1 and y + h <= 1):
        raise ValueError('換位構圖的物件框必須位於目前影像內')
    size = image_size if image_size is not None else frame.get('image_size')
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)):
        raise ValueError('換位構圖需要目前影像的寬與高')
    return {'target_bbox': dict(target_bbox), 'image_size': list(size),
            'reference_frame_id': frame.get('id'), 'reference_path': frame['path'],
            'reference_joints': dict(frame['joints']),
            'reference_capture_monotonic': frame['capture_monotonic'],
            'reference_camera_generation': frame['camera_generation']}


def _selected_framing_camera(grasp, frame):
    """Snapshot the validated camera once; a selected broken model must fail.

    This reads the local selected artifact, not an image or a sensor. Legacy
    operation retains its explicitly hypothetical pinhole framing model.
    """
    config = getattr(grasp.robot, 'config', {})
    if config.get('spatial_backend', 'legacy') != 'v2' or not config.get('spatial_calibration_profile'):
        return None
    from .spatial_runtime import depth_calibration
    calibration = depth_calibration(grasp.model, [frame], config)
    if calibration is None or calibration.get('metric_calibrated') is not True:
        raise ValueError('目前腕部影像不符合已選相機校準，請重新取像或核對校準')
    focal_error = calibration.get('focal_relative_uncertainty')
    if type(focal_error) not in (int, float) or not math.isfinite(focal_error) or not 0 <= focal_error < 1:
        raise ValueError('已選相機校準缺少有效的焦距誤差')
    transforms = grasp.model.fk.transforms(frame['joints'])
    calibrated = transforms['gripper_link'] @ np.asarray(calibration['handeye'])
    correction = np.linalg.inv(transforms['wrist_camera_optical']) @ calibrated
    return {'source': 'selected_validated_wrist_camera', 'reference': _frame(frame),
            'profile_path': config['spatial_calibration_profile'],
            'calibration_fingerprint': calibration['calibration_fingerprint'],
            'calibration_revision': calibration['calibration_revision'],
            'validation_id': calibration['validation_id'],
            'geometry_revision': calibration['geometry_revision'],
            'camera_id': calibration['camera_id'], 'mount_revision': calibration['mount_revision'],
            'camera_generation': frame['camera_generation'],
            'image_size': calibration['image_size'],
            'camera_matrix': calibration['camera_matrix'], 'distortion': calibration['distortion'],
            'handeye': calibration['handeye'], 'focal_relative_uncertainty': focal_error,
            'cad_camera_to_calibrated_camera': correction.tolist(),
            'runtime_calibration_scopes': calibration['runtime_calibration_scopes'],
            'model_role': 'framing hypotheses only; not object depth or collision clearance'}


def _attach_framing_camera(candidate, camera_model):
    if camera_model is not None:
        if camera_model['reference'] != candidate['reference']:
            raise ValueError('取景相機模型必須綁定這次候選的腕部影像')
        candidate['framing_camera_model'] = deepcopy(camera_model)
        candidate['contract_sha256'] = _fingerprint(candidate)


def _check_framing_camera_selection(grasp, candidate, frame):
    selected = candidate.get('framing_camera_model')
    if selected is None:
        return
    current = _selected_framing_camera(grasp, frame)
    keys = ('profile_path', 'calibration_fingerprint', 'geometry_revision',
            'camera_id', 'mount_revision', 'camera_generation', 'image_size')
    if current is None or any(current[key] != selected[key] for key in keys):
        raise ValueError('取景使用的相機校準已改變，請重新規劃')


def _framing_prediction(candidate, context, cancel):
    """Reject obviously cropped views using explicitly hypothetical pinholes.

    The visible ROI corners are projected on several depth planes and focal
    priors, including intermediate nominal poses. These hypotheses rank useful
    views; they are neither depth estimates nor external-clearance evidence.
    Servo tracking error, distortion and object shape remain unmodeled here.
    """
    width, height = context['image_size']
    box = context['target_bbox']; x, y, w, h = (box[n] for n in ('x', 'y', 'w', 'h'))
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    center_y = y + h / 2
    minimum_visible = 1.; maximum_shift = 0.; endpoint_bounds = []
    minimum_centering_improvement = float('inf')
    samples = candidate['framing_projection_path']
    calibrated = candidate.get('framing_camera_model')
    camera_hypotheses = []
    if calibrated is not None:
        if (calibrated['reference'] != candidate.get('reference')
                or calibrated['image_size'] != context['image_size']
                or calibrated['camera_generation'] != context['reference_camera_generation']):
            raise ValueError('取景相機模型與腕部影像、尺寸或相機身分不符')
        correction = np.asarray(calibrated['cad_camera_to_calibrated_camera'])
        inverse_correction = np.linalg.inv(correction)
        distortion = np.asarray(calibrated['distortion'])
        focal_error = calibrated['focal_relative_uncertainty']
        for factor in (1 - focal_error, 1., 1 + focal_error):
            matrix = np.array(calibrated['camera_matrix'], dtype=float)
            matrix[0, 0] *= factor; matrix[1, 1] *= factor
            normalized = cv2.undistortPoints((corners * [width, height]).reshape(-1, 1, 2),
                                            matrix, distortion).reshape(-1, 2)
            camera_hypotheses.append((np.c_[normalized, np.ones(4)], matrix))
    else:
        for focal in FRAMING_FOCAL_WIDTH_SAMPLES:
            scale = np.array([focal, focal * width / height])
            camera_hypotheses.append((np.c_[(corners - .5) / scale, np.ones(4)], scale))
    for sample in samples:
        _cancel(cancel)
        fraction = sample['fraction']
        transform = np.asarray(sample['reference_to_candidate_camera_transform'])
        if calibrated is not None:
            # All existing paths use the fixed CAD camera mount. Conjugation
            # expresses the same robot motion in the selected optical frame,
            # including composed lookahead paths, without inventing a capture.
            transform = inverse_correction @ transform @ correction
        for rays, intrinsics in camera_hypotheses:
            for depth in FRAMING_DEPTH_SAMPLES_M:
                projected = (transform[:3, :3] @ (rays * depth).T).T + transform[:3, 3]
                if np.any(projected[:, 2] <= 0):
                    minimum_visible = 0.
                    if fraction == 1:
                        minimum_centering_improvement = -1.
                    continue
                if calibrated is not None:
                    pixels = cv2.projectPoints(projected, np.zeros(3), np.zeros(3),
                        intrinsics, distortion)[0].reshape(-1, 2) / [width, height]
                else:
                    pixels = projected[:, :2] / projected[:, 2, None] * intrinsics + .5
                lower, upper = pixels.min(axis=0), pixels.max(axis=0)
                area = float(np.prod(upper - lower))
                inside = np.maximum(0., np.minimum(upper, 1.) - np.maximum(lower, 0.))
                visible = float(np.prod(inside) / area) if area > 0 else 0.
                minimum_visible = min(minimum_visible, visible)
                maximum_shift = max(maximum_shift, abs(float((lower[1] + upper[1]) / 2) - center_y))
                if fraction == 1:
                    endpoint_bounds.append([*lower.tolist(), *upper.tolist()])
                    improvement = abs(center_y - .5) - abs(float((lower[1] + upper[1]) / 2) - .5)
                    minimum_centering_improvement = min(minimum_centering_improvement, improvement)
    provenance = ({'source': 'nominal_FK_projection_with_selected_validated_wrist_camera',
                  'calibration_fingerprint': calibrated['calibration_fingerprint'],
                  'calibration_revision': calibrated['calibration_revision'],
                  'geometry_revision': calibrated['geometry_revision'],
                  'camera_id': calibrated['camera_id'], 'mount_revision': calibrated['mount_revision'],
                  'camera_matrix': calibrated['camera_matrix'], 'distortion': calibrated['distortion'],
                  'focal_relative_uncertainty': calibrated['focal_relative_uncertainty'],
                  'focal_scale_hypotheses': [1 - focal_error, 1., 1 + focal_error],
                  'principal_point_normalized': [calibrated['camera_matrix'][0][2] / width,
                                                 calibrated['camera_matrix'][1][2] / height]}
                 if calibrated is not None else {
                  'source': 'nominal_CAD_projection_with_uncalibrated_pinhole_priors',
                  'hypothetical_focal_lengths_per_image_width': list(FRAMING_FOCAL_WIDTH_SAMPLES),
                  'principal_point_normalized': [.5, .5]})
    return {**context, **provenance,
            'hypothetical_depth_planes_m': list(FRAMING_DEPTH_SAMPLES_M),
            'sampled_path_fractions': [sample['fraction'] for sample in samples],
            'minimum_predicted_visible_fraction': round(minimum_visible, 4),
            'maximum_predicted_vertical_center_shift': round(maximum_shift, 4),
            'minimum_endpoint_vertical_centering_improvement': round(minimum_centering_improvement, 4),
            'endpoint_bbox_extent_normalized': (np.round(
                [np.min(np.array(endpoint_bounds)[:, 0]), np.min(np.array(endpoint_bounds)[:, 1]),
                 np.max(np.array(endpoint_bounds)[:, 2]), np.max(np.array(endpoint_bounds)[:, 3])], 4).tolist()
                if endpoint_bounds else None),
            'minimum_required_visible_fraction': MIN_PREDICTED_ROI_VISIBILITY,
            'target_depth_estimated': False, 'collision_clearance_verified': False,
            'includes_tracking_uncertainty': False,
            'scope': 'visible ROI only; finite priors, nominal path; verify actual new image'}


def _apply_framing(candidate, context, cancel):
    if context is None:
        return
    prediction = _framing_prediction(candidate, context, cancel)
    if prediction['minimum_predicted_visible_fraction'] < MIN_PREDICTED_ROI_VISIBILITY:
        raise ValueError('這條換位路徑可能讓物件移出畫面，改試較小或側向換位')
    lookahead = candidate['descriptor'].get('translation_lookahead', {})
    if lookahead.get('framing_projection_path'):
        future_prediction = _framing_prediction({
            'reference': candidate['reference'],
            'framing_projection_path': lookahead['framing_projection_path'],
            'framing_camera_model': candidate.get('framing_camera_model')}, context, cancel)
        if future_prediction['minimum_predicted_visible_fraction'] < MIN_PREDICTED_ROI_VISIBILITY:
            raise ValueError('手腕調整後的全臂換位可能讓物件移出畫面，改試能保留物件的路徑')
        lookahead['target_framing_model'] = future_prediction
        lookahead['framing_reference'] = 'actual_current_image_before_preparation'
    candidate['descriptor']['target_framing_model'] = prediction
    candidate['descriptor']['information_gain_model']['framing'] = (
        'current visible ROI projected over explicit focal/depth hypotheses; actual framing unverified; '
        + prediction['source'])


def _rank_framed(candidates):
    body = [c for c in candidates if c['kind'] == 'body']
    body.sort(key=lambda c: (
        c['descriptor'].get('purpose') not in ('recenter_for_translation', 'prepare_translation_clearance'),
        -c['descriptor']['target_framing_model']['minimum_predicted_visible_fraction'],
        c['descriptor']['target_framing_model']['maximum_predicted_vertical_center_shift'],
        -c['descriptor']['camera_change_model']['baseline_mm']))
    for rank, candidate in enumerate(body, 1):
        candidate['descriptor']['framing_preference_rank'] = rank
        candidate['contract_sha256'] = _fingerprint(candidate)
    return body + [c for c in candidates if c['kind'] != 'body']


def filter_observation_framing(candidates, target_bbox, image_size, *, cancel=None,
                               reference_frame=None):
    """Filter already-screened paths with the box from their planning image.

    No camera, motor, inference or path changes occur. The caller binds the
    returned box to the candidates' unchanged image/pose before using this.
    reference_frame identifies the actual planning image after a stationary
    refresh. Metadata permits <=3 ticks of drift; the caller must also verify
    unchanged image content. Motor paths remain anchored to their original pose.
    Existing external-path approval applies only to an unchanged surviving
    segment; choosing a different candidate still needs its own assessment.
    """
    _cancel(cancel)
    if not isinstance(candidates, (list, tuple)):
        raise ValueError('構圖檢查需要完整補看候選')
    if not candidates:
        return []
    reference = None
    for candidate in candidates:
        if (not isinstance(candidate, Mapping)
                or candidate.get('contract_sha256') != _fingerprint(candidate)
                or candidate.get('kind') not in ('body', 'jaw_open')):
            raise ValueError('補看候選已改變，請重新規劃')
        if reference is None:
            reference = candidate['reference']
        elif candidate['reference'] != reference:
            raise ValueError('構圖檢查的候選必須來自同一張腕部影像與姿勢')
    image_reference = reference
    if reference_frame is not None:
        if (not isinstance(reference_frame, Mapping) or reference_frame.get('source') != 'wrist'
                or reference_frame.get('camera_generation') != reference['camera_generation']
                or not isinstance(reference_frame.get('path'), str) or not reference_frame['path']):
            raise ValueError('構圖物件框必須來自同一台腕部相機')
        stamp = reference_frame.get('capture_monotonic')
        same_id = (isinstance(reference.get('id'), str) and bool(reference['id'])
                   and reference_frame.get('id') == reference['id'])
        identical_capture = (reference_frame['path'] == reference['path']
                             and stamp == reference['capture_monotonic'])
        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                or not (stamp > reference['capture_monotonic'] or same_id or identical_capture)):
            raise ValueError('構圖物件框的影像早於這次候選，請重新取像')
        pose = _pose(reference_frame.get('joints'))
        if any(abs(pose[n] - reference['joints'][n]) > 3 for n in ALL):
            raise ValueError('構圖影像的姿勢已改變，請重新產生候選')
        image_reference = reference_frame
    context = _framing_context(target_bbox, image_size, image_reference)
    if context is None:
        return deepcopy(list(candidates))
    context.update(candidate_reference_frame_id=reference.get('id'),
                   candidate_reference_path=reference['path'],
                   candidate_reference_capture_monotonic=reference['capture_monotonic'],
                   candidate_reference_joints=dict(reference['joints']),
                   stationary_refresh_drift_ticks={n: image_reference['joints'][n] - reference['joints'][n]
                                                   for n in ALL},
                   unchanged_scene_check='caller must verify image content; helper checks metadata only')
    output = []
    for original in candidates:
        _cancel(cancel)
        candidate = deepcopy(original)
        if candidate['kind'] == 'body':
            try:
                _apply_framing(candidate, context, cancel)
            except ValueError:
                continue
        output.append(candidate)
    return _rank_framed(output)


def filter_observation_novelty(candidates, history, *, model=None, cancel=None):
    """Avoid nominal camera-pose revisits when a novel screened path exists.

    The caller supplies only this attempt's history. Foreign-camera, future or
    malformed history records contribute no evidence. This compares camera FK
    at actual previous encoder poses to the proposed endpoint, not image
    content, recovered object scale, or guaranteed information gain. Paths and
    their geometry approvals are never changed.
    """
    _cancel(cancel)
    if not isinstance(candidates, (list, tuple)) or not isinstance(history, (list, tuple)):
        raise ValueError('視角比較需要完整候選與這次腕部觀察紀錄')
    if not candidates:
        return []
    reference = candidates[0].get('reference') if isinstance(candidates[0], Mapping) else None
    for candidate in candidates:
        if (not isinstance(candidate, Mapping)
                or candidate.get('contract_sha256') != _fingerprint(candidate)
                or candidate.get('kind') not in ('body', 'jaw_open')
                or candidate.get('reference') != reference):
            raise ValueError('視角比較的候選已改變或來自不同影像')
    if model is None:
        from .local_cartesian import default_model
        model = default_model()
    priors = []; excluded = 0
    for frame in history:
        _cancel(cancel)
        stamp = frame.get('capture_monotonic') if isinstance(frame, Mapping) else None
        if (not isinstance(frame, Mapping) or frame.get('source') != 'wrist'
                or frame.get('camera_generation') != reference['camera_generation']
                or not isinstance(frame.get('path'), str) or not frame['path']
                or type(stamp) not in (int, float) or not math.isfinite(stamp)
                or stamp > reference['capture_monotonic']):
            excluded += 1
            continue
        try:
            pose = _pose(frame.get('joints'))
        except ValueError:
            excluded += 1
            continue
        priors.append((frame, model.fk.transforms(pose)['wrist_camera_optical']))
    output = []
    for original in candidates:
        _cancel(cancel)
        candidate = deepcopy(original)
        if candidate['kind'] != 'body':
            output.append(candidate)
            continue
        endpoint = np.asarray(candidate['descriptor']['camera_change_model']['after_camera_transform_base_m'])
        comparisons = []
        for frame, view in priors:
            translation = float(np.linalg.norm(endpoint[:3, 3] - view[:3, 3]) * 1000)
            rotation = float(np.degrees(Rotation.from_matrix(endpoint[:3, :3] @ view[:3, :3].T).magnitude()))
            comparisons.append({'frame_id': frame.get('id'), 'path': frame['path'],
                'capture_monotonic': frame['capture_monotonic'],
                'translation_mm': translation, 'rotation_degrees': rotation})
        nearest = min(comparisons, key=lambda p: max(p['translation_mm'] / DUPLICATE_VIEW_TRANSLATION_MM,
                       p['rotation_degrees'] / DUPLICATE_VIEW_ROTATION_DEGREES), default=None)
        duplicate = bool(nearest is not None
                         and nearest['translation_mm'] <= DUPLICATE_VIEW_TRANSLATION_MM
                         and nearest['rotation_degrees'] <= DUPLICATE_VIEW_ROTATION_DEGREES)
        if nearest is not None:
            nearest = {**nearest, 'translation_mm': round(nearest['translation_mm'], 3),
                       'rotation_degrees': round(nearest['rotation_degrees'], 3)}
        candidate['descriptor']['view_novelty_model'] = {
            'source': 'wrist_encoder_history_and_CAD_camera_prior',
            'history_scope': 'caller-supplied same-attempt history',
            'camera_generation': reference['camera_generation'],
            'eligible_prior_views': len(priors), 'excluded_history_records': excluded,
            'nearest_prior': nearest,
            'nearest_metric': 'minimum max(translation/5mm, rotation/2degrees)',
            'predicted_revisit': duplicate,
            'duplicate_thresholds': {'translation_mm': DUPLICATE_VIEW_TRANSLATION_MM,
                                     'rotation_degrees': DUPLICATE_VIEW_ROTATION_DEGREES},
            'actual_information_gain_verified': False, 'object_scale_inferred': False,
            'scope': 'nominal endpoint vs prior encoder camera poses; image changes and tracking error unmodeled'}
        output.append(candidate)
    novel_exists = any(c['kind'] == 'body' and not c['descriptor']['view_novelty_model']['predicted_revisit']
                       for c in output)
    result = []
    for candidate in output:
        if candidate['kind'] == 'body':
            novelty = candidate['descriptor']['view_novelty_model']
            preparation = candidate['descriptor'].get('preparation_progress_model', {})
            # Finishing a proven, partly achieved preparation is deliberately
            # smaller than a useful depth baseline. Keep that distinction
            # explicit instead of relabeling a near-identical image as novel.
            progressing_preparation = (
                candidate['id'] == TRANSLATION_RECOVERY_CONTINUE_ID
                and preparation.get('same_direction_actual_progress_verified') is True
                and preparation.get('endpoint_unvisited_in_preparation_direction') is True)
            if novel_exists and novelty['predicted_revisit'] and not progressing_preparation:
                continue
            novelty['retained_without_novel_alternative'] = novelty['predicted_revisit'] and not novel_exists
            novelty['retained_for_translation_preparation'] = (
                novelty['predicted_revisit'] and progressing_preparation)
            candidate['contract_sha256'] = _fingerprint(candidate)
        result.append(candidate)
    return result


def _compact_segment(model, segment, origin, *, excursion_profile=None):
    tip_limit = _origin_tip_limit(excursion_profile)
    envelope = segment['actual_tracking_envelope_ticks']
    reserve = _tracking_reserve(model, envelope)
    raw, target = segment['measured_origin'], segment['joint_targets']
    initial_tip = model.tip(origin)[:3, 3]
    rows = []
    samples = segment['sampled_path']
    for sample in samples:
        t = sample['fraction']
        q = {n: raw[n] + t * (target[n] - raw[n]) for n in ALL}
        if any(abs(q[n] - origin[n]) + envelope[n] > MAX_ORIGIN_TICKS for n in NAMES):
            raise ValueError('補看路徑未保留本次關節誤差範圍')
        distance = float(np.linalg.norm(model.tip(q)[:3, 3] - initial_tip) * 1000)
        if distance + reserve > tip_limit:
            raise ValueError('補看路徑未保留本次位置誤差範圍')
    # The complete joint line and tube are exact; samples summarize that line.
    for index in sorted({0, len(samples) - 1, *range(0, len(samples), max(1, len(samples) // 12))}):
        sample = samples[index]
        rows.append([round(sample['fraction'], 5), *[round(v, 3) for v in sample['tip_delta_mm']]])
    camera_change = None
    transforms = getattr(model, 'fk', None)
    if transforms is not None:
        a = transforms.transforms(raw).get('wrist_camera_optical')
        b = transforms.transforms(target).get('wrist_camera_optical')
        if a is not None and b is not None:
            camera_change = _camera_change(model, raw, target)
    return {
        'origin_joints': raw, 'target_joints': target,
        'joint_interpolation': 'one continuous straight line in joint coordinates',
        'sample_columns': ['fraction', 'tip_dx_mm', 'tip_dy_mm', 'tip_dz_mm'],
        'sampled_path': rows,
        'actual_tracking_envelope_ticks': envelope,
        'command_tracking_envelope_ticks': segment['uncertainty_ticks'],
        'measured_bounds': segment['measured_bounds'],
        'physical_goal_bounds': segment['raw_bounds'],
        'geometry_bounds': segment.get('geometry_bounds', segment['raw_bounds']),
        'stationary_limit_joints': segment.get('stationary_limit_joints', []),
        'stationary_bounds_rule': segment.get('stationary_bounds_rule', 'full stationary box inside joint limits'),
        'endpoint_tolerance_ticks': segment['endpoint_tolerance_ticks'],
        'stationary_tolerance_ticks': segment['stationary_tolerance_ticks'],
        'onboard_margin_ticks': segment['onboard_margin_ticks'],
        'stationary_goal_resends': segment['stationary_goal_resends'],
        'nominal_tip_path_mm': round(segment['nominal_tip_path_mm'], 3),
        'camera_change_model': camera_change,
        'table_guard': 'full known robot mesh and full360 jaw vs modeled table; excludes external objects',
        'fixed_attempt_origin': origin,
        'attempt_limits': {'maximum_body_ticks_from_origin': MAX_ORIGIN_TICKS,
                           'maximum_tip_distance_from_origin_mm': tip_limit},
        'jaw_changes': False,
    }


def _translation_target(grasp, raw, direction, distance, cancel, *, calibrated_joint_bounds=None):
    """Solve a camera baseline jointly, freezing axes lacking moving reserve.

    This is one offline IK solve, never a motor-step loop. A near-limit elbow
    can stay fixed while pan/lift/wrist cooperate. The normal segment preflight
    still validates every moving reserve, fixed hold, URDF and full mesh path.
    Orientation is softly preserved; no target depth or toe-in is invented.
    """
    model = grasp.model
    camera = model.fk.transforms(raw)['wrist_camera_optical']
    rays = {'left': -camera[:3, 0], 'right': camera[:3, 0],
            'backward': -camera[:3, 2], 'up': np.array([0., 0., 1.])}
    if direction not in rays:
        raise ValueError('只提供側移、後退或抬高的觀察方向')
    ray = rays[direction]
    free, lower, upper = [], [], []
    for n in NAMES:
        envelope = max(OBSERVATION_ENDPOINT_TOLERANCE, abs(grasp.robot.goals[n] - raw[n]) + 6)
        limits = model.calibration[n]
        lo = math.ceil(max(raw[n] - 180, limits['range_min'] + 12 + envelope,
                           2047 + model.joint_limits[n][0] / RAD_PER_TICK + envelope))
        hi = math.floor(min(raw[n] + 180, limits['range_max'] - 12 - envelope,
                            2047 + model.joint_limits[n][1] / RAD_PER_TICK - envelope))
        if calibrated_joint_bounds is not None:
            calibrated_lo, calibrated_hi = calibrated_joint_bounds[n]
            if not calibrated_lo <= raw[n] <= calibrated_hi:
                raise ValueError('目前換位起點超出已校準區域')
            lo = max(lo, math.ceil(calibrated_lo))
            hi = min(hi, math.floor(calibrated_hi))
        if envelope <= 60 and lo <= raw[n] <= hi and lo < hi:
            free.append(n); lower.append(lo); upper.append(hi)
    if not free:
        raise ValueError('目前沒有可形成相機基線的關節空間')
    initial = np.array([raw[n] for n in free], dtype=float)
    desired = camera[:3, 3] + ray * distance / 1000
    def residual(values):
        _cancel(cancel)
        pose = {**raw, **dict(zip(free, values))}
        view = model.fk.transforms(pose)['wrist_camera_optical']
        turn = Rotation.from_matrix(view[:3, :3] @ camera[:3, :3].T).as_rotvec()
        return np.r_[view[:3, 3] - desired, .025 * turn, 1e-6 * (values - initial)]
    solution = least_squares(residual, initial, bounds=(lower, upper), max_nfev=100,
                             xtol=1e-9, ftol=1e-9, gtol=1e-9)
    _cancel(cancel)
    if not solution.success:
        raise ValueError('相機換位規劃尚未收斂')
    rounded = np.rint(solution.x).astype(int)
    # Tiny changes only create extra actuated axes and do not improve baseline.
    target = {**raw, **{n: raw[n] if abs(int(v) - raw[n]) < 3 else int(v)
                       for n, v in zip(free, rounded)}}
    metrics = _camera_change(model, raw, target)
    delta = np.array(metrics['position_delta_base_mm'])
    along = float(delta @ ray)
    if (metrics['baseline_mm'] < MIN_CAMERA_BASELINE_MM or metrics['rotation_degrees'] > 18
            or metrics['optical_forward_translation_mm'] > 2
            or along < max(12., distance * .5)):
        raise ValueError('這條路徑沒有形成足夠的後退、側移或抬高視角')
    if direction in ('left', 'right') and np.linalg.norm(delta - along * ray) > distance * .4:
        raise ValueError('側移路徑偏離相機基線方向')
    return target


def _maximum_optical_advance(model, start, target, segment):
    camera = model.fk.transforms(start)['wrist_camera_optical']
    maximum = 0.
    for sample in segment['sampled_path']:
        fraction = sample['fraction']
        raw = {n: start[n] + fraction * (target[n] - start[n]) for n in ALL}
        point = model.fk.transforms(raw)['wrist_camera_optical'][:3, 3]
        maximum = max(maximum, float((point - camera[:3, 3]) @ camera[:3, 2]) * 1000)
    return maximum


def _body_candidate(grasp, frame, origin, target, candidate_id, label, expected):
    segment = prepare_joint_segment(grasp.model, frame['joints'], dict(grasp.robot.goals),
                                    target, fixed_origin=origin,
                                    endpoint_tolerance_ticks=OBSERVATION_ENDPOINT_TOLERANCE,
                                    stationary_tolerance_ticks=OBSERVATION_STATIONARY_TOLERANCE)
    descriptor = {**_compact_segment(grasp.model, segment, origin),
                  'candidate_id': candidate_id, 'kind': 'body', 'label': label,
                  'expected_view_change': expected}
    raw = segment['measured_origin']
    before = grasp.model.fk.transforms(raw)['wrist_camera_optical']
    projection_path = []
    for fraction in (.25, .5, .75, 1.):
        pose = {n: raw[n] + fraction * (target[n] - raw[n]) for n in ALL}
        after = grasp.model.fk.transforms(pose)['wrist_camera_optical']
        projection_path.append({'fraction': fraction,
            'reference_to_candidate_camera_transform': (np.linalg.inv(after) @ before).tolist()})
    return _seal({'id': candidate_id, 'kind': 'body', 'label': label,
                  'expected_view_change': expected, 'origin': origin,
                  'reference': frame, 'segment': segment, 'descriptor': descriptor,
                  'framing_projection_path': projection_path})


def _translation_recovery_progress(frame, history, used, held):
    """Bind a second preparation to the immediately preceding actual move.

    Object-program appends the selected ID and its before-image together.
    Unordered/misaligned history cannot supply that evidence. No old planned
    endpoint is substituted for the current encoder pose.
    """
    if (not isinstance(used, (list, tuple)) or not used
            or len(history) != len(used) or used[-1] != TRANSLATION_RECOVERY_ID
            or used.count(TRANSLATION_RECOVERY_ID) != 1
            or TRANSLATION_RECOVERY_CONTINUE_ID in used):
        return None
    prior = history[-1]
    try:
        if (not isinstance(prior, Mapping) or prior.get('source') != 'wrist'
                or prior.get('camera_generation') != frame['camera_generation']
                or not isinstance(prior.get('path'), str) or not prior['path']
                or type(prior.get('capture_monotonic')) not in (int, float)
                or not math.isfinite(prior['capture_monotonic'])
                or prior['capture_monotonic'] >= frame['capture_monotonic']):
            return None
        before = _pose(prior.get('joints')); raw = frame['joints']; goals = _pose(held)
        progress = raw['wrist_flex'] - before['wrist_flex']
        commanded = goals['wrist_flex'] - before['wrist_flex']
        if (abs(progress) < 12 or progress * commanded <= 0
                or abs(raw['wrist_flex'] - goals['wrist_flex']) > OBSERVATION_ENDPOINT_TOLERANCE
                or any(abs(raw[n] - before[n]) > (JAW_TOLERANCE if n == 'gripper'
                    else OBSERVATION_STATIONARY_TOLERANCE) for n in ALL if n != 'wrist_flex')):
            return None
        return {'source': 'same_attempt_before_and_after_wrist_encoders',
                'before_frame_id': prior.get('id'), 'current_frame_id': frame.get('id'),
                'before_capture_monotonic': prior['capture_monotonic'],
                'current_capture_monotonic': frame['capture_monotonic'],
                'before_joints': before, 'actual_joints': dict(raw),
                'held_goal_joints': goals, 'actual_pitch_progress_ticks': progress,
                'direction_sign': 1 if progress > 0 else -1,
                'same_direction_actual_progress_verified': True}
    except (ValueError, TypeError, KeyError):
        return None


def _translation_recovery(grasp, frame, origin, cancel, *, directions,
                          calibrated_joint_bounds=None, framing=None,
                          preparation_progress=None, history=(), camera_model=None):
    """Offer one short pitch only when it unlocks a screened body baseline.

    The future segment is an offline feasibility witness, not another motor
    command or a synthetic observation. After executing the prefix, the normal
    next iteration captures actual joints and solves all candidates afresh.
    Both segments retain the original whole-mesh guard and attempt origin.
    """
    raw = frame['joints']
    if calibrated_joint_bounds is not None and any(
            not lo <= raw[n] <= hi for n, (lo, hi) in calibrated_joint_bounds.items()):
        return None
    continuation = preparation_progress is not None
    candidate_id = TRANSLATION_RECOVERY_CONTINUE_ID if continuation else TRANSLATION_RECOVERY_ID
    deltas = (preparation_progress['direction_sign'] * 30,) if continuation else (-30, 30)
    for delta in deltas:
        _cancel(cancel)
        value = raw['wrist_flex'] + delta
        if calibrated_joint_bounds is not None:
            lo, hi = calibrated_joint_bounds['wrist_flex']
            value = int(min(max(value, math.ceil(lo)), math.floor(hi)))
        if abs(value - raw['wrist_flex']) < 12:
            continue
        if continuation:
            sign = preparation_progress['direction_sign']
            # A continuation advances the same preparation beyond every prior
            # actual pitch, rather than revisiting an old preparation endpoint.
            if any(sign * (value - previous['joints']['wrist_flex']) < 12
                   for previous in history if isinstance(previous, Mapping)
                   and previous.get('source') == 'wrist'
                   and previous.get('camera_generation') == frame['camera_generation']
                   and isinstance(previous.get('joints'), Mapping)
                   and type(previous['joints'].get('wrist_flex')) in (int, float)):
                continue
        target = {**raw, 'wrist_flex': value}
        try:
            prefix = _body_candidate(grasp, frame, origin, target, candidate_id,
                '先調整手腕，讓手臂能換個位置看',
                '這一步只準備後續全臂換位；尚未取得物件深度，下一張影像會用實測姿勢重新規劃。')
            _attach_framing_camera(prefix, camera_model)
            maximum = _maximum_optical_advance(grasp.model, raw, target, prefix['segment'])
            if maximum > 2:
                continue
            hypothetical = SimpleNamespace(model=grasp.model,
                robot=SimpleNamespace(goals=prefix['segment']['physical_goal_targets']))
            for direction in ('up', 'backward', 'left', 'right'):
                if direction not in directions:
                    continue
                for distance in (40., 30., 20.):
                    _cancel(cancel)
                    try:
                        future = _translation_target(hypothetical, target, direction, distance, cancel,
                            **({'calibrated_joint_bounds': calibrated_joint_bounds}
                               if calibrated_joint_bounds is not None else {}))
                        segment = prepare_joint_segment(grasp.model, target,
                            hypothetical.robot.goals, future, fixed_origin=origin,
                            endpoint_tolerance_ticks=OBSERVATION_ENDPOINT_TOLERANCE,
                            stationary_tolerance_ticks=OBSERVATION_STATIONARY_TOLERANCE)
                        # Enforce the SAME total excursion and optical approach
                        # restrictions as a normal body observation candidate.
                        _compact_segment(grasp.model, segment, origin)
                        if _maximum_optical_advance(grasp.model, target, future, segment) > 2:
                            continue
                        # The ROI belongs to the ACTUAL current photo, not an
                        # invented image at the preparation endpoint. Retain
                        # this composed path also when the bbox arrives later.
                        reference_camera = grasp.model.fk.transforms(raw)['wrist_camera_optical']
                        projection_path = []
                        for fraction in (0., .25, .5, .75, 1.):
                            pose = {n: target[n] + fraction * (future[n] - target[n]) for n in ALL}
                            camera = grasp.model.fk.transforms(pose)['wrist_camera_optical']
                            projection_path.append(dict(fraction=fraction,
                                reference_to_candidate_camera_transform=(
                                    np.linalg.inv(camera) @ reference_camera).tolist()))
                        future_framing = None
                        if framing is not None:
                            future_framing = _framing_prediction(
                                {'framing_projection_path': projection_path,
                                 'framing_camera_model': prefix.get('framing_camera_model'),
                                 'reference': prefix['reference']}, framing, cancel)
                            if future_framing['minimum_predicted_visible_fraction'] < MIN_PREDICTED_ROI_VISIBILITY:
                                raise ValueError('準備後的換位仍可能把物件移出畫面，改試其他準備方向')
                        change = _camera_change(grasp.model, target, future)
                        prefix['descriptor'].update(purpose='prepare_translation_clearance',
                            translation_recovery_visit=2 if continuation else 1,
                            maximum_translation_recovery_visits=2,
                            maximum_nominal_optical_forward_advance_mm=round(maximum, 3),
                            information_gain_model={
                                'intent': 'unlock_screened_full_arm_translation',
                                'metric_depth_verified': False, 'object_scale_inferred': False,
                                'subsequent_translation_required_for_metric_depth': True,
                                'framing': 'prefix is not a new depth measurement'},
                            translation_lookahead={
                                'basis': 'nominal_offline_feasibility_only',
                                'direction': direction, 'requested_translation_mm': distance,
                                'nominal_target_joints': future,
                                'nominal_camera_change_model': change,
                                'minimum_model_table_clearance_mm': min(v for n, v in
                                    segment['clearance']['continuous_model_lower_bound_mm'].items()
                                    if n != 'base_link'),
                                'fixed_attempt_origin': origin,
                                'calibrated_bounds_checked': calibrated_joint_bounds is not None,
                                'framing_projection_path': projection_path,
                                'target_framing_model': future_framing,
                                'framing_reference': 'actual_current_image_before_preparation' if framing is not None else None,
                                'execution_authorized': False, 'automatic_chaining': False,
                                'requires_actual_capture_and_replanning': True})
                        if continuation:
                            prefix['descriptor']['preparation_progress_model'] = {
                                **deepcopy(preparation_progress),
                                'endpoint_unvisited_in_preparation_direction': True,
                                'additional_nominal_pitch_ticks': value - raw['wrist_flex'],
                                'current_metric_translation_available': False,
                                'metric_depth_verified': False}
                        _apply_framing(prefix, framing, cancel)
                        prefix['contract_sha256'] = _fingerprint(prefix)
                        return prefix
                    except (ValueError, RuntimeError) as exc:
                        if isinstance(exc, SessionCancelled):
                            raise
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, SessionCancelled):
                raise
    return None


def _jaw_table_screen(model, raw, held, target_raw=JAW_OPEN_TARGET):
    """A fixed-body jaw sweep needs its actual 4-tick box, not motion margin.

    No body goal changes in this action. The generic 12-tick body-motion margin
    is not applicable: validate the full stationary +/-4 measured box against
    onboard limits, and separately validate unchanged physical hold goals.
    Robot meshes, camera and the entire rotating jaw remain in the table check.
    """
    actual = {**{n: BODY_DRIFT for n in NAMES}, 'gripper': 0}
    model.validate_raw(raw, margin_ticks=0, uncertainty_ticks=actual)
    model.validate_raw(held, margin_ticks=0)
    target = {**raw, 'gripper': target_raw}
    model.validate_raw(target, margin_ticks=0)
    jaw_range = model.calibration['gripper']
    if not jaw_range['range_min'] + 12 <= target_raw <= jaw_range['range_max'] - 12:
        raise ValueError('張開目標超出夾爪保留行程')
    for n in NAMES:
        lo, hi = model.joint_limits[n]
        if not lo <= (raw[n] - BODY_DRIFT - 2047) * RAD_PER_TICK <= (
                raw[n] + BODY_DRIFT - 2047) * RAD_PER_TICK <= hi:
            raise ValueError('固定手臂的實測範圍超出模型關節界線')
    floor = model.floor_minimum(raw)
    if model.jaw_label not in floor:
        raise ValueError('缺少完整夾爪旋轉範圍的桌面模型')
    residual = {link: sum(2 * radius * math.sin(BODY_DRIFT * RAD_PER_TICK / 2)
                         for radius in radii.values())
                for link, radii in model.radii.items()}
    lower = {link: z - residual[link] for link, z in floor.items()}
    passed = all(z >= .005 for link, z in lower.items() if link != 'base_link')
    return {'model_table_pass': passed,
            'method': 'stationary full-mesh/full360-jaw plus analytic +/-4-body-tick radius bound',
            'actual_tracking_envelope_ticks': actual,
            'continuous_model_lower_bound_mm': {n: v * 1000 for n, v in lower.items()},
            'body_motion_margin_not_applied': True,
            'not_environment_collision_certificate': True}


def _jaw_candidate(grasp, frame, origin):
    raw = frame['joints']; held = _pose(grasp.robot.goals)
    opening = empty_opening_contract(grasp.model, getattr(grasp.robot, 'config', {}))
    target = opening['target_raw']; low, high = opening['measured_raw_range']
    if grasp.robot.pwm_owner is not None or not low <= raw['gripper'] < min(high + 1, target - JAW_TOLERANCE):
        return None
    clearance = _jaw_table_screen(grasp.model, raw, held, target)
    if not clearance['model_table_pass']:
        return None
    descriptor = {
        'candidate_id': 'jaw_open', 'kind': 'jaw_open', 'label': '張開空夾爪再看',
        'expected_view_change': '可能減少夾爪遮擋；張開本身不證明已看清或抓住物件。',
        'jaw_target_raw': target, 'jaw_start_raw': raw['gripper'],
        'jaw_raw_sweep': [max(low, raw['gripper'] - JAW_TOLERANCE), target + JAW_TOLERANCE],
        'endpoint_tolerance_ticks': JAW_TOLERANCE,
        'opening_contract': deepcopy(opening),
        'aperture_mm': (None if opening['target_geometry'] is None else opening['target_geometry']['aperture_m'] * 1000),
        'geometry_note': ('raw opening target only; jaw zero/aperture are not calibrated'
            if opening['mode'] == 'legacy' else 'Selected jaw profile at the disclosed target; retain its uncertainty/accuracy basis. This is opening preparation, not object contact or metric depth.'),
        'body_joints_held': {n: raw[n] for n in NAMES},
        'actual_body_drift_ticks': BODY_DRIFT,
        'table_guard': 'full360 moving-jaw envelope plus all known robot meshes vs modeled table',
        'external_opening_path_requires_visual_clearance': True,
        'empty_jaw_required': True, 'opens_only': True,
        'minimum_visual_confidence': EMPTY_JAW_OPEN_MIN_CONFIDENCE,
        'fixed_attempt_origin': origin,
        'attempt_limits': {'maximum_body_ticks_from_origin': MAX_ORIGIN_TICKS,
                           'maximum_tip_distance_from_origin_mm': MAX_ORIGIN_TIP_MM},
    }
    return _seal({'id': 'jaw_open', 'kind': 'jaw_open', 'label': '張開空夾爪再看',
                  'expected_view_change': '可能減少夾爪遮擋；張開本身不證明已看清或抓住物件。',
                  'origin': origin, 'reference': frame, 'held_goals': held,
                  'opening_contract': opening,
                  'segment': None, 'descriptor': descriptor,
                  'clearance': clearance})


def build_observation_candidates(grasp, frame, *, origin=None, used=(), cancel=None,
                                 target_bbox=None, image_size=None, history=(), diagnostics=None,
                                 calibrated_joint_bounds=None):
    """Screen finite current-pose candidates; no model/camera/motor operation.

    target_bbox must refer to this exact frame, not a prior view after movement.
    When supplied, image_size is [width, height] (or comes from the frame).
    """
    _cancel(cancel)
    source_frame = frame
    frame = _frame(frame); raw = frame['joints']
    framing = _framing_context(target_bbox, image_size, source_frame)
    origin = dict(raw) if origin is None else _pose(origin)
    # Try a shorter useful translation before accepting a view that the next
    # spatial-v2 step would reject. Joint-space interpolation is convex, so
    # endpoint bounds cover the nominal path without expanding calibration.
    if calibrated_joint_bounds is not None:
        if (not isinstance(calibrated_joint_bounds, Mapping)
                or set(calibrated_joint_bounds) != set(NAMES)
                or any(not isinstance(v, (list, tuple)) or len(v) != 2
                       or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
                       or not 0 <= v[0] <= v[1] <= 4095
                       for v in calibrated_joint_bounds.values())):
            raise ValueError('換位需要完整、有效的相機校準關節範圍')
    def check_calibrated_pose(target):
        if calibrated_joint_bounds is not None and any(
                not lo <= pose[n] <= hi
                for n, (lo, hi) in calibrated_joint_bounds.items() for pose in (raw, target)):
            raise ValueError('這個換位超出已校準區域，改試區域內的路徑')
    if not isinstance(used, (tuple, list, set, frozenset)) or any(not isinstance(i, str) for i in used):
        raise ValueError('補看紀錄必須是已使用的候選 ID')
    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            raise ValueError('候選診斷需要字典')
        diagnostics.clear()
        diagnostics.update(frame_id=source_frame.get('id'), framing_applied=framing is not None,
                           used_candidate_ids=list(used), rejections=[], rejection_counts={})
    def rejected(candidate_id, exc, **details):
        if diagnostics is not None:
            reason = str(exc)
            diagnostics['rejections'].append({'candidate_id': candidate_id, 'reason': reason, **details})
            counts = diagnostics['rejection_counts']
            counts[reason] = counts.get(reason, 0) + 1
    if grasp.state not in ('idle', 'approaching') or grasp.robot.pwm_owner is not None:
        if diagnostics is not None:
            diagnostics.update(candidate_ids=[], unavailable_reason='gripper transaction owns the robot')
        return []
    camera_model = _selected_framing_camera(grasp, source_frame)
    _outer_guard(grasp.model, origin, raw)
    if not isinstance(history, (list, tuple)):
        raise ValueError('補看歷史必須是這次的腕部觀察紀錄')
    candidates = []
    attempted_directions = []
    for direction, label in [('backward', '退後換個距離看看'), ('left', '移到左側比較前後距離'),
                             ('right', '移到右側比較前後距離'), ('up', '從較高的位置看看')]:
        family = 'move_' + direction
        visit = 1
        if family in used:
            if not history:
                continue
            visit = max([1, *[i for i in range(2, MAX_DIRECTION_VISITS + 1)
                               if f'{family}_continue_{i}' in used]]) + 1
            if visit > MAX_DIRECTION_VISITS:
                continue
        candidate_id = family if visit == 1 else f'{family}_continue_{visit}'
        attempted_directions.append(direction)
        for distance in (40., 30., 20.):
            _cancel(cancel)
            try:
                target = _translation_target(grasp, raw, direction, distance, cancel,
                    **({'calibrated_joint_bounds': calibrated_joint_bounds}
                       if calibrated_joint_bounds is not None else {}))
                check_calibrated_pose(target)
                candidate = _body_candidate(grasp, frame, origin, target, candidate_id, label,
                    '用相機實際換位比較視差或大小變化；新影像仍須確認是否提供了物件距離。')
                _attach_framing_camera(candidate, camera_model)
                view = candidate['descriptor']['camera_change_model']
                maximum_advance = _maximum_optical_advance(grasp.model, raw, target, candidate['segment'])
                if maximum_advance > 2:
                    raise ValueError('這條觀察路徑會先朝未知物件方向前進')
                candidate['descriptor'].update(
                    direction_family=family, direction_visit=visit,
                    maximum_direction_visits=MAX_DIRECTION_VISITS,
                    requested_translation_mm=distance,
                    translation_reference='camera optical center, not gripper fixed pad',
                    maximum_nominal_optical_forward_advance_mm=round(maximum_advance, 3),
                    information_gain_model={
                        'intent': {'backward': 'scale_change_and_parallax', 'up': 'higher_view_and_parallax'}.get(direction, 'lateral_parallax'),
                        'nominal_baseline_mm': view['baseline_mm'],
                        'nominal_transverse_baseline_mm': view['transverse_baseline_mm'],
                        'nominal_optical_forward_translation_mm': view['optical_forward_translation_mm'],
                        'framing': 'softly preserve current camera orientation; object centering is unverified',
                        'target_depth_assumed': False, 'object_scale_inferred': False,
                        'requires_after_images_and_actual_encoder_poses': True,
                        'tracking_tube_can_include_forward_drift': True,
                    })
                _apply_framing(candidate, framing, cancel)
                candidate['contract_sha256'] = _fingerprint(candidate)
                candidates.append(candidate)
                break
            except (ValueError, RuntimeError) as exc:
                if isinstance(exc, SessionCancelled):
                    raise
                rejected(candidate_id, exc, requested_translation_mm=distance)
    # An edge-of-image target may need aiming before another translation can
    # retain it. Offer that preparation alongside metric views only when an
    # anchored ROI predicts improved centering; ordinary rolls remain fallback.
    metric_available = bool(candidates)
    preparation_progress = (_translation_recovery_progress(frame, history, used, grasp.robot.goals)
                            if TRANSLATION_RECOVERY_ID in used else None)
    if (not metric_available and calibrated_joint_bounds is not None and attempted_directions
            and TRANSLATION_RECOVERY_CONTINUE_ID not in used
            and (TRANSLATION_RECOVERY_ID not in used or preparation_progress is not None)):
        recovery = _translation_recovery(grasp, frame, origin, cancel,
            directions=attempted_directions, calibrated_joint_bounds=calibrated_joint_bounds,
            framing=framing, preparation_progress=preparation_progress, history=history,
            camera_model=camera_model)
        if recovery is not None:
            candidates.append(recovery)
        else:
            rejected(TRANSLATION_RECOVERY_ID, ValueError('有限的手腕準備姿勢仍無法接上已檢查的全臂換位'))
    if not metric_available or framing is not None:
        rotations = [('roll_positive', 'wrist_roll', 113, '換個轉腕角度看看'),
                     ('roll_negative', 'wrist_roll', -113, '從另一側轉腕看看'),
                     ('pitch_positive', 'wrist_flex', 68, '換個俯仰角度看看'),
                     ('pitch_negative', 'wrist_flex', -68, '從另一個俯仰角度看看')]
        for candidate_id, joint, delta, label in rotations:
            _cancel(cancel)
            if candidate_id in used or (metric_available and joint != 'wrist_flex'):
                continue
            try:
                target = {**raw, joint: raw[joint] + delta}
                check_calibrated_pose(target)
                candidate = _body_candidate(grasp, frame, origin, target,
                    candidate_id, label, '目前沒有較大的相機換位路徑，先嘗試解除遮擋；這不代表已知物件距離。')
                _attach_framing_camera(candidate, camera_model)
                candidate['descriptor']['information_gain_model'] = {
                    'intent': 'framing_or_deocclusion', 'metric_depth_verified': False,
                    'reason': 'no unused screened larger-baseline candidate is feasible'}
                _apply_framing(candidate, framing, cancel)
                prediction = candidate['descriptor'].get('target_framing_model')
                improves_centering = (joint == 'wrist_flex' and prediction is not None
                    and prediction['minimum_endpoint_vertical_centering_improvement'] >= MIN_RECENTER_IMPROVEMENT)
                if metric_available and not improves_centering:
                    continue
                if improves_centering:
                    candidate['descriptor']['purpose'] = 'recenter_for_translation'
                    candidate['descriptor']['information_gain_model'].update(
                        intent='recenter_for_translation',
                        reason='improve current target framing before a subsequent metric translation',
                        minimum_predicted_vertical_centering_improvement=prediction[
                            'minimum_endpoint_vertical_centering_improvement'],
                        subsequent_translation_required_for_metric_depth=True,
                        object_scale_inferred=False)
                    candidate['label'] = candidate['descriptor']['label'] = '先把物件移回畫面中央'
                    candidate['expected_view_change'] = candidate['descriptor']['expected_view_change'] = (
                        '先調整鏡頭朝向，讓物件留在畫面裡，再換位比較距離；這一步本身不會確認深度。')
                candidate['contract_sha256'] = _fingerprint(candidate)
                candidates.append(candidate)
            except (ValueError, RuntimeError) as exc:
                if isinstance(exc, SessionCancelled):
                    raise
                rejected(candidate_id, exc)
    if framing is not None:
        # First preserve the visible object, then favor low vertical drift and
        # a useful baseline. Astra still assesses the complete external path.
        candidates = _rank_framed(candidates)
    if 'jaw_open' not in used:
        try:
            candidate = _jaw_candidate(grasp, frame, origin)
            if candidate is not None:
                candidates.append(candidate)
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, SessionCancelled):
                raise
            rejected('jaw_open', exc)
    if history:
        candidates = filter_observation_novelty(candidates, [*history, frame], model=grasp.model, cancel=cancel)
        # Continuations require actual eligible prior wrist evidence, and must
        # be novel even when every available path would revisit an old view.
        candidates = [c for c in candidates if c['descriptor'].get('direction_visit', 1) == 1
                      or (c['descriptor']['view_novelty_model']['eligible_prior_views'] > 1
                          and not c['descriptor']['view_novelty_model']['predicted_revisit'])]
    if diagnostics is not None:
        diagnostics['candidate_ids'] = [c['id'] for c in candidates]
    return candidates


def _live(grasp, candidate, cancel, deadline):
    _cancel(cancel)
    if grasp.robot.stop.is_set():
        raise SessionCancelled('補看已停止')
    if time.monotonic() >= deadline:
        raise RuntimeError('這次補看的動作時間已到')
    if (not grasp.robot.camera.fresh()
            or grasp.robot.camera.generation != candidate['reference']['camera_generation']):
        raise RuntimeError('腕部畫面中斷或相機已更換，先停止補看')


def _execution_excursion_profile(candidate, engineering_calibration):
    """Only the explicit engineering entry point can use its sealed bounds."""
    if type(engineering_calibration) is not bool:
        raise ValueError('取景工作類型無效')
    profile = candidate.get('excursion_profile')
    if not engineering_calibration:
        if profile is not None:
            raise ValueError('工程校準候選只能由工程校準流程執行')
        return None
    descriptor = candidate.get('descriptor', {})
    if (profile != CALIBRATION_EXCURSION_PROFILE or candidate.get('kind') != 'body'
            or candidate.get('purpose') != CALIBRATION_EXCURSION_PROFILE
            or descriptor.get('purpose') != CALIBRATION_EXCURSION_PROFILE
            or descriptor.get('attempt_limits') != {
                'maximum_body_ticks_from_origin': MAX_ORIGIN_TICKS,
                'maximum_tip_distance_from_origin_mm': CALIBRATION_MAX_ORIGIN_TIP_MM}):
        raise ValueError('校準候選缺少一致的工作範圍')
    return profile


def execute_observation(grasp, candidate, reference, cancel, report, *, verdict, deadline=None,
                        engineering_calibration=False):
    """Execute exactly one explicitly assessed candidate, then return new evidence."""
    from .grasp import unchanged_wrist
    if (not isinstance(candidate, Mapping) or candidate.get('contract_sha256') != _fingerprint(candidate)
            or candidate.get('kind') not in ('body', 'jaw_open')):
        raise ValueError('補看候選已改變，請重新規劃')
    excursion_profile = _execution_excursion_profile(candidate, engineering_calibration)
    verdict = ObservationVerdict.model_validate(verdict)
    minimum = EMPTY_JAW_OPEN_MIN_CONFIDENCE if candidate['kind'] == 'jaw_open' else LOCAL_MOVE_MIN_CONFIDENCE
    if verdict.candidate_id != candidate['id']:
        raise ValueError('這次判斷不屬於目前動作，請重新規劃')
    if not (verdict.gripper_empty and verdict.human_clear and verdict.clear_path):
        raise ValueError('這個補看動作尚未確認空夾爪與完整路徑')
    if verdict.confidence < minimum:
        raise ValueError(f'對這次補看動作的把握還不夠（{verdict.confidence:.0%}，至少需要 {minimum:.0%}）')
    if grasp.state not in ('idle', 'approaching') or grasp.robot.pwm_owner is not None:
        raise RuntimeError('先處理目前的夾爪工作，再換視角')
    started = time.monotonic()
    if deadline is None:
        deadline = started + MAX_EXECUTION_SECONDS
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('補看需要有效的動作期限')
    deadline = min(deadline, started + MAX_EXECUTION_SECONDS)
    _live(grasp, candidate, cancel, deadline)
    _frame(reference)
    unchanged_wrist(candidate['reference'], reference)
    _check_framing_camera_selection(grasp, candidate, reference)
    _outer_guard(grasp.model, candidate['origin'], reference['joints'], excursion_profile=excursion_profile)
    if candidate['kind'] == 'body':
        # Keep the reviewed image as the final pre-motion comparison anchor.
        # A->B and B->C may each pass while A->C exceeds the allowed change.
        # B remains a fresh dispatch check above, never a new scene approval.
        moved = execute_prepared_segment(grasp, candidate['segment'], candidate['reference'], cancel, report,
            deadline=deadline, speed=60,
            measured_guard=lambda actual: _outer_guard(grasp.model, candidate['origin'], actual,
                                                      excursion_profile=excursion_profile))
        planned = np.asarray([candidate['segment']['delta_ticks'][n] for n in NAMES], float)
        actual = np.asarray([moved['after']['joints'][n] - moved['before']['joints'][n]
                             for n in NAMES], float)
        progress = float(actual @ planned / (planned @ planned))
        # Observation tolerates cheap-servo endpoint error, but still needs a
        # meaningful change of view in the intended direction. The executor
        # already enforces the original whole-path tube and endpoint bound.
        if not math.isfinite(progress) or progress < MINIMUM_OBSERVATION_PROGRESS:
            reason = '這次補看的實際移動不足，先保持目前姿勢'
            grasp.robot.error = reason
            grasp.robot.stop.set(); grasp.robot.hold()
            grasp.record({'type': 'active_observation_failed', 'candidate_id': candidate['id'],
                          'reason': reason, 'movement': moved,
                          'directional_progress_fraction': progress})
            raise RuntimeError(reason)
        result = {'candidate_id': candidate['id'], 'kind': 'body',
                  'before': moved['before'], 'after': moved['after'], 'completed': True,
                  'movement': moved, 'physical_seconds_charged': time.monotonic() - started,
                  'information_gain_verified': False, 'task_completed': False,
                  'measured_camera_change_model': _camera_change(grasp.model, moved['before']['joints'], moved['after']['joints']),
                  'directional_progress_fraction': progress,
                  'cloud_calls_during_motion': 0}
        grasp.record({'type': 'active_observation_result', **result})
        return result

    # A jaw opening has no body trajectory. The existing empty-jaw routine owns
    # low PWM, with an additional measured-only guard. A fault must not
    # automatically restore PWM or close a pending contact transaction.
    before = grasp.capture(); unchanged_wrist(candidate['reference'], before)
    if dict(grasp.robot.goals) != candidate['held_goals']:
        raise RuntimeError('保持目標已改變，先重新規劃補看')
    if deadline - time.monotonic() < 10:
        raise RuntimeError('剩餘時間不足以完成張開與空夾爪確認')
    _outer_guard(grasp.model, candidate['origin'], before['joints'])
    current_opening = _jaw_candidate(grasp, _frame(before), candidate['origin'])
    if current_opening is None:
        raise RuntimeError('目前無法執行這個張開補看動作')
    if current_opening['opening_contract'] != candidate['opening_contract']:
        raise RuntimeError('張開目標或夾爪模型已改變，先重新審核這個開口')
    opening = candidate['opening_contract']; target = opening['target_raw']
    def measured_guard(actual):
        _live(grasp, candidate, cancel, deadline)
        actual = _pose(actual)
        if any(abs(actual[n] - candidate['reference']['joints'][n]) > BODY_DRIFT for n in NAMES):
            raise RuntimeError('張開補看時手臂偏離固定姿勢')
        if (opening['mode'] == 'profile' and not
                candidate['descriptor']['jaw_raw_sweep'][0] <= actual['gripper'] <= candidate['descriptor']['jaw_raw_sweep'][1]):
            raise RuntimeError('張開補看偏離已審核的夾爪掃掠範圍')
        grasp.model.validate_raw(actual, margin_ticks=0)
        _outer_guard(grasp.model, candidate['origin'], actual)
        if any(z < .005 for n, z in grasp.model.floor_minimum(actual).items() if n != 'base_link'):
            raise RuntimeError('張開補看時太靠近模型桌面')
    attempted = False
    try:
        _live(grasp, candidate, cancel, deadline)
        attempted = True
        options = {'opening_contract': opening} if opening['mode'] == 'profile' else {}
        opened = grasp.open_empty(before, report, cancel=cancel, guard=measured_guard, **options)
        _live(grasp, candidate, cancel, deadline)
        measured_guard(opened['joints'])
        if (opened.get('source') != 'wrist'
                or opened.get('camera_generation') != before['camera_generation']
                or opened.get('capture_monotonic', 0) <= before['capture_monotonic']
                or abs(opened['joints']['gripper'] - target) > JAW_TOLERANCE
                or any(abs(opened['joints'][n] - before['joints'][n]) > BODY_DRIFT for n in NAMES)):
            raise RuntimeError('張開後的實測位置或腕部影像不符合這次補看')
        power_record = grasp.robot.gripper_power._record() or {}
        if power_record.get('contact_possible'):
            raise RuntimeError('出現可能接觸物件的紀錄，不能當成空夾爪清理')
        # open_empty already returned the most recent camera frame. release's
        # recovery snapshot may reuse that same frame, so first obtain a newer
        # settled view. Keep the OLDER opened frame as the release reference:
        # its raw snapshot can now be equal to renewed and still be newer than
        # the reference. This wait retains cancellation and all live guards.
        _live(grasp, candidate, cancel, deadline)
        renewed = grasp.capture()
        _live(grasp, candidate, cancel, deadline)
        _frame(renewed)
        unchanged_wrist(opened, renewed)
        measured_guard(renewed['joints'])
        # Prior explicit empty-jaw + opening-clearance verdict applies to this
        # opening only; it never authorizes a close or release of a held object.
        power = grasp.robot.gripper_power
        old_read, old_write = power.read, power.write
        def power_check():
            measured_guard(grasp.robot.read())
        def checked_read(*args):
            power_check(); value = old_read(*args); power_check(); return value
        def checked_write(*args):
            power_check(); value = old_write(*args); power_check(); return value
        try:
            with grasp.robot.io:
                power_check(); power.read = checked_read; power.write = checked_write
            released = grasp.release(opened, report, cancel, empty_verified=True)
        finally:
            with grasp.robot.io:
                power.read, power.write = old_read, old_write
        _live(grasp, candidate, cancel, deadline)
        after = grasp.capture()
        if (after.get('source') != 'wrist' or after.get('camera_generation') != before['camera_generation']
                or after.get('capture_monotonic', 0) <= opened['capture_monotonic']
                or grasp.robot.pwm_owner is not None):
            raise RuntimeError('張開補看尚未取得新的腕部影像或完成空夾爪工作')
        _outer_guard(grasp.model, candidate['origin'], after['joints'])
        measured_guard(after['joints'])
        if abs(after['joints']['gripper'] - target) > JAW_TOLERANCE:
            raise RuntimeError('空爪工作結束後開口偏離已審核的目標')
        result = {'candidate_id': candidate['id'], 'kind': 'jaw_open', 'completed': True,
                  'opening_contract': opening,
                  'before': before, 'after': after, 'release': released,
                  'physical_seconds_charged': time.monotonic() - started,
                  'information_gain_verified': False, 'task_completed': False,
                  'cloud_calls_during_motion': 0}
        grasp.record({'type': 'active_observation_result', **result})
        return result
    except Exception:
        if attempted:
            record = grasp.robot.gripper_power._record() or {}
            if record and record.get('phase') != 'completed':
                grasp.robot.gripper_power.leave_pending()
            grasp.robot.stop.set()
            grasp.robot.hold()
        raise
