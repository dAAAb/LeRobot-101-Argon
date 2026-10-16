"""Offline shared optics/hand-eye fit with one fixed board pose per session.

Explicit, pre-existing splits are mandatory. This produces an inactive candidate,
never moves hardware, changes FK, or selects a production configuration.
"""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math

import cv2
import numpy as np
from scipy.optimize import least_squares

from .board_calibration import (BoardSpec, ValidationLimits, _detection, _fit_intrinsics,
    _hash, _inverse, _mean_pose, _observability, _pack, _project, _transform)
from .calibration_extension import _bounds, _camera_board, _rotation_error
from .calibration_store import _validate_frame
from .geometry import observation_families
from .spatial_contract import (JOINTS, UNCERTAINTY_FIELDS, RigidTransform, WristCalibration,
    build_wrist_observation, require_compatible)
from .spatial_runtime import geometry_revision


def fit_multi_session_calibration(sessions, model, parent_profile, *, splits,
                                  board_session_id, parent_artifact=None, split_sources=None):
    """Refit K/D/X on training views only; evaluate every held-out view unchanged.

All sessions must have >=3 independent training and >=3 held-out poses. Across
sessions, return visits cannot cross folds or inflate the independent view count.
No caller-supplied accuracy thresholds are accepted.
"""
    limits, board = ValidationLimits(), BoardSpec()
    report = dict(status='rejected', source='wrist-only', calibration_profile=None,
        method='multi-session-camera-fk', validation_policy_version=2,
        board_session_id=board_session_id, board_spec=board.to_dict(),
        metric_accuracy='approximate', grasp_ready=False, activation_performed=False,
        validation={}, blockers=[])
    try:
        sessions, parent, splits = deepcopy(sessions), deepcopy(parent_profile), deepcopy(splits)
        json.dumps([sessions, parent, splits, parent_artifact, split_sources], allow_nan=False)
        if not isinstance(sessions, list) or not 2 <= len(sessions) <= 8:
            raise ValueError('需要 2–8 個分別固定棋盤的工作階段。')
        ids = [s['session_id'] for s in sessions]
        if len(set(ids)) != len(ids) or board_session_id not in ids or set(splits) != set(ids):
            raise ValueError('工作階段與預先指定的 training/heldout 分組不一致。')
        if parent.get('board_session_id') not in ids:
            raise ValueError('必須重新驗證父版本原本的工作階段。')
        revision = geometry_revision(model)
        camera = WristCalibration.from_mapping(parent, native_image_size=sessions[0]['image_size'],
            camera_id=sessions[0]['camera_id'], mount_revision=sessions[0]['mount_revision'],
            geometry_revision=revision, cad_handeye=np.eye(4))
        if (not camera.metric_calibrated or parent.get('source') != 'wrist-only'
                or parent.get('metric_calibrated') is not True or parent.get('needs_more_views') is not False):
            raise ValueError('需要已驗證的腕部相機父版本，以保留回復來源。')
        if any(type(parent.get(n)) not in (int, float) or not math.isfinite(parent[n])
               or parent[n] < 0 for n in UNCERTAINTY_FIELDS):
            raise ValueError('父版本缺少有限且完整的 uncertainty。')
        views, manifest, seen = [], [], set()
        for si, session in enumerate(sessions):
            sid, frames = session['session_id'], session['frames']
            if (session.get('board_static_declared') is not True
                    or session.get('status') not in ('collecting', 'solved')
                    or session.get('board_spec') != board.to_dict()
                    or session.get('board_fingerprint') != board.fingerprint
                    or not 6 <= len(frames) <= 48):
                raise ValueError('每個工作階段需要固定的相同規格棋盤與至少六個已保存視角。')
            for key in ('camera_id', 'mount_revision', 'geometry_revision'):
                if session[key] != parent[key]:
                    raise ValueError(f'工作階段與父版本的 {key} 不符。')
            if session['image_size'] != list(camera.native_image_size):
                raise ValueError('工作階段相機解析度不一致。')
            split = splits[sid]
            train_ids, held_ids = split['training_frame_ids'], split['heldout_frame_ids']
            if (not isinstance(train_ids, list) or not isinstance(held_ids, list)
                    or len(set(train_ids)) != len(train_ids) or len(set(held_ids)) != len(held_ids)
                    or set(train_ids) & set(held_ids)
                    or set(train_ids) | set(held_ids) != {f['id'] for f in frames}
                    or min(len(train_ids), len(held_ids)) < 3):
                raise ValueError('每個工作階段需要明確、不重疊且完整的至少 3 train / 3 heldout 分組。')
            observations = []
            for frame in frames:
                _validate_frame(frame)
                if frame['id'] in seen:
                    raise ValueError('同一影像不能重複充當校準證據。')
                seen.add(frame['id'])
                for key, expected in (('board_session_id', sid), ('camera_id', session['camera_id']),
                        ('camera_generation', session['camera_generation']), ('image_size', session['image_size'])):
                    if frame.get(key) != expected:
                        raise ValueError(f'影像與工作階段的 {key} 不一致。')
                image_path = Path(frame['path'])
                if not 0 < image_path.stat().st_size <= 16 * 1024 * 1024:
                    raise ValueError('校準影像檔案大小無效。')
                data = image_path.read_bytes()
                if hashlib.sha256(data).hexdigest() != frame.get('image_sha256'):
                    raise ValueError('校準影像 hash 已改變。')
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None or list(image.shape[1::-1]) != session['image_size']:
                    raise ValueError('校準影像尺寸與原紀錄不同。')
                transforms = model.fk.transforms(frame['joints'])
                observation = build_wrist_observation(frame, transforms, geometry_revision=revision,
                    calibration=camera, camera_id=camera.camera_id, mount_revision=camera.mount_revision)
                if not observation.timing.pose_bound:
                    raise ValueError('校準影像缺少停穩後實測關節配對。')
                observations.append(observation)
                detection = _detection(frame, board)
                pixels = np.asarray(detection['corners_px'], float)
                if (not detection['found'] or pixels.shape != (board.rows * board.columns, 2)
                        or not np.isfinite(pixels).all() or np.any(pixels < 0)
                        or np.any(pixels >= np.asarray(session['image_size']))):
                    raise ValueError('每張影像必須有完整且綁定原影像的有限角點。')
                views.append(dict(session_index=si, session_id=sid, frame=frame,
                    pose=np.asarray(transforms['gripper_link']), pixels=pixels,
                    split='training' if frame['id'] in train_ids else 'heldout'))
            require_compatible(observations)  # Camera restarts may occur BETWEEN sessions only.
            observable = _observability([v['pose'] for v in views if v['session_id'] == sid
                                        and v['split'] == 'training'])
            if (observable['rotation_span_deg'] < limits.minimum_rotation_span_deg
                    or observable['second_axis_ratio'] < limits.minimum_second_axis_ratio):
                raise ValueError('各工作階段 training 視角需要足夠旋轉跨度與兩個不同轉軸。')
            manifest.append(dict(session_id=sid, session_sha256=_hash(session),
                training_frame_ids=train_ids, heldout_frame_ids=held_ids, observability=observable,
                frames=deepcopy(frames)))
        families = observation_families([v['frame'] for v in views])
        if any(len({views[i]['split'] for i in family}) != 1 for family in families):
            raise ValueError('heldout pose 洩漏：跨工作階段的相同或近似姿勢必須留在同一分組。')
        if len(families) != len(views):
            raise ValueError('請每個近似姿勢只保留一張；重拍不能增加獨立姿勢或擬合權重。')
        train = [v for v in views if v['split'] == 'training']
        held = [v for v in views if v['split'] == 'heldout']
        if len(views) < limits.minimum_views or len(train) < limits.minimum_training_views:
            raise ValueError('共享相機擬合需要至少 12 個不同姿勢與 8 個 training 視角。')
        points, size = board.object_points(), camera.native_image_size
        validation = report['validation']
        validation.update(training_frame_ids=[v['frame']['id'] for v in train],
            heldout_frame_ids=[v['frame']['id'] for v in held], independent_pose_groups=len(families),
            heldout_data_used_for_fitting=False, heldout_used_for_shared_parameter_fitting=False,
            per_view_pnp_pose_fitted_for_evaluation=True,
            acceptance_limits=dict(per_view_pnp_reprojection_p95_px=6.,
                board_translation_m=.015, board_rotation_deg=3.), chain_pixels_are_diagnostic=True)
        optics, selection = _fit_intrinsics(points, [v['pixels'] for v in train], size)
        validation['intrinsics_model_selection'] = selection
        if optics is None:
            raise ValueError('保守鏡頭模型沒有有限、合理且全影像可逆的解。')
        rms, k, distortion, _, _, intrinsic_std, _, _ = optics
        distortion = distortion.ravel()
        if not math.isfinite(rms) or rms > limits.training_rms_px:
            raise ValueError('training 相機內參投影誤差未通過。')
        for view in views:
            view['pnp'] = _camera_board(points, view['pixels'], k, distortion)
        parent_x = np.asarray(parent['handeye'])
        initial_ys = [_mean_pose([v['pose'] @ parent_x @ v['pnp'] for v in train
                                 if v['session_index'] == si]) for si in range(len(sessions))]

        def unpack(parameters):
            return [_transform(parameters[i:i + 3], parameters[i + 3:i + 6])
                    for i in range(0, len(parameters), 6)]

        def residual(parameters):
            x, *ys = unpack(parameters)
            result = []
            for v in train:
                try:
                    projection = _project(points, _inverse(v['pose'] @ x) @ ys[v['session_index']], k, distortion)
                    result.extend((projection - v['pixels']).ravel())
                except ValueError:
                    result.extend(np.full(len(points) * 2, 1e5))
            return np.asarray(result)

        fit = least_squares(residual, np.r_[_pack(parent_x), *[_pack(y) for y in initial_ys]],
                            loss='soft_l1', f_scale=1., max_nfev=240)
        if not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun).all():
            raise ValueError('多工作階段手眼擬合沒有有限且收斂的解。')
        x, *ys = unpack(fit.x)
        x = RigidTransform('gripper_link', 'wrist_camera_optical', x, 'candidate').as_array()
        for y in ys:
            RigidTransform('base', 'board', y, 'candidate')
        if np.linalg.norm(x[:3, 3]) > .30:
            raise ValueError('相機安裝位置不合理，請檢查棋盤尺寸與固定狀態。')
        rows = []
        for v in views:
            y, c = ys[v['session_index']], v['pose'] @ x
            observed_board, observed_camera = c @ v['pnp'], y @ _inverse(v['pnp'])
            chain = np.linalg.norm(_project(points, _inverse(c) @ y, k, distortion) - v['pixels'], axis=1)
            optics_error = np.linalg.norm(_project(points, v['pnp'], k, distortion) - v['pixels'], axis=1)
            corner_delta = points @ y[:3, :3].T + y[:3, 3] - (
                points @ observed_board[:3, :3].T + observed_board[:3, 3])
            rows.append(dict(frame_id=v['frame']['id'], session_id=v['session_id'], split=v['split'],
                pnp_reprojection_p95_px=float(np.percentile(optics_error, 95)),
                chain_reprojection_p95_px=float(np.percentile(chain, 95)),
                board_translation_error_m=float(np.linalg.norm(y[:3, 3] - observed_board[:3, 3])),
                board_rotation_error_deg=_rotation_error(y, observed_board),
                board_corner_max_displacement_m=float(np.max(np.linalg.norm(corner_delta, axis=1))),
                camera_center_displacement_m=float(np.linalg.norm(c[:3, 3] - observed_camera[:3, 3]))))
        pairs = []
        for a, b in combinations(views, 2):
            if a['session_id'] != b['session_id'] or a['split'] == b['split'] == 'training':
                continue  # PnP relative pose is comparable only for the SAME stationary board.
            predicted, observed = _inverse(a['pose'] @ x) @ (b['pose'] @ x), a['pnp'] @ _inverse(b['pnp'])
            pairs.append(dict(session_id=a['session_id'], frame_ids=[a['frame']['id'], b['frame']['id']],
                translation_error_m=float(np.linalg.norm(predicted[:3, 3] - observed[:3, 3])),
                rotation_error_deg=_rotation_error(predicted, observed)))
        empirical = dict(base_position_uncertainty_m=max(r[n] for r in rows for n in
            ('board_translation_error_m', 'board_corner_max_displacement_m', 'camera_center_displacement_m')),
            relative_translation_uncertainty_m=max(p['translation_error_m'] for p in pairs),
            relative_rotation_uncertainty_deg=max(p['rotation_error_deg'] for p in pairs),
            focal_relative_uncertainty=float(max(intrinsic_std.ravel()[0] / k[0, 0],
                                                intrinsic_std.ravel()[1] / k[1, 1])))
        held_rows = [r for r in rows if r['split'] == 'heldout']
        validation.update(training_intrinsics_rms_px=float(rms), fit_success=True,
            evaluations=rows, pairwise_camera_errors=pairs, empirical_uncertainty=empirical,
            heldout_max_translation_m=max(r['board_translation_error_m'] for r in held_rows),
            heldout_max_rotation_deg=max(r['board_rotation_error_deg'] for r in held_rows),
            # Keep the established report key's camera/FK chain meaning. This
            # pixel summary is diagnostic; the separate PnP summary gates optics.
            heldout_max_reprojection_p95_px=max(r['chain_reprojection_p95_px'] for r in held_rows),
            heldout_max_pnp_reprojection_p95_px=max(r['pnp_reprojection_p95_px'] for r in held_rows),
            max_pnp_reprojection_p95_px=max(r['pnp_reprojection_p95_px'] for r in rows),
            reprojection_summary_interpretation='heldout_max_reprojection_p95_px is camera/FK chain projection, diagnostic only. heldout_max_pnp_reprojection_p95_px and max_pnp_reprojection_p95_px evaluate fixed optics with per-view nuisance board pose; the latter includes training and heldout views and is gated at 6 px.',
            uncertainty_rule='max(parent, current empirical residual); no sum because both describe the same combined camera/FK error. Focal term is training linearized precision, not a physical bound.',
            board_base_transforms={sid: y.tolist() for sid, y in zip(ids, ys)})
        json.dumps(validation, allow_nan=False)
        if any(v < 0 or not math.isfinite(v) for v in empirical.values()):
            raise ValueError('uncertainty 估計必須有限且非負。')
        if any(r['pnp_reprojection_p95_px'] > limits.heldout_reprojection_px for r in rows):
            raise ValueError('固定鏡頭模型的逐張 PnP 投影品質未通過。')
        if any(r['board_translation_error_m'] > limits.heldout_translation_m
               or r['board_rotation_error_deg'] > limits.heldout_rotation_deg for r in rows):
            raise ValueError('相機/FK 度量誤差未通過 15 mm / 3°。')
        scope = dict(kind='combined-camera-fk-sampled-workspace',
            joint_raw_bounds=_bounds([v['frame'] for v in views]),
            sampled_joint_poses=[{n: v['frame']['joints'][n] for n in JOINTS[:-1]} for v in views],
            extrapolation_allowed=False, individually_identified_joint_zeros=False,
            jaw_contact_geometry_calibrated=False,
            board_plane_base=ys[ids.index(board_session_id)].tolist(),
            board_planes_by_session=validation['board_base_transforms'],
            image_points_bbox_px=[np.min(np.concatenate([v['pixels'] for v in views]), axis=0).tolist(),
                                  np.max(np.concatenate([v['pixels'] for v in views]), axis=0).tolist()],
            image_domain_validity=selection['selected_image_domain'],
            interpretation='Only recorded poses and pixels are empirically checked; joint bounds do not prove unsampled combinations. Board planes are session-specific. No individual joint-zero or jaw measurement.',
            image_domain_interpretation='Mathematical lens invertibility is not empirical accuracy outside sampled pixels.')
        provenance = dict(parent_profile=parent, parent_profile_sha256=_hash(parent),
            parent_artifact=deepcopy(parent_artifact), sessions=manifest,
            split_sources=deepcopy(split_sources),
            rollback=dict(calibration_revision=parent['calibration_revision'],
                          validation_id=parent['validation_id'], parent_artifact=deepcopy(parent_artifact)))
        identity = _hash(dict(provenance=provenance, validation=validation, K=k.tolist(),
                              distortion=distortion.tolist(), handeye=x.tolist()))
        profile = dict(schema_version=1, source='wrist-only', calibration_revision=f'multi-{identity[:20]}',
            camera_id=camera.camera_id, mount_revision=camera.mount_revision, geometry_revision=revision,
            native_image_size=list(size), image_size=list(size), camera_matrix=k.tolist(),
            distortion=distortion.tolist(), distortion_model='opencv-brown', handeye=x.tolist(),
            validated_components=['intrinsics', 'handeye', 'robot_geometry'],
            validation_id=f'multi-holdout-{identity}', validation_kind='independent-heldout',
            validation_scope=scope, board_session_id=board_session_id, board_fingerprint=board.fingerprint,
            multi_session_provenance=provenance, camera_generation=None,
            metric_calibrated=True, needs_more_views=False, metric_accuracy='approximate', grasp_ready=False,
            uncertainty_interpretation=validation['uncertainty_rule'] + ' Approximate sampled-workspace error, not guaranteed bounds or grasp clearance.',
            **{n: max(parent[n], empirical[n]) for n in UNCERTAINTY_FIELDS})
        WristCalibration.from_mapping(profile, native_image_size=size, camera_id=camera.camera_id,
            mount_revision=camera.mount_revision, geometry_revision=revision, cad_handeye=np.eye(4))
        report.update(status='validated', calibration_profile=profile)
    except (ValueError, KeyError, TypeError, OSError, cv2.error, np.linalg.LinAlgError) as exc:
        report['blockers'] = [str(exc)]
    return report
