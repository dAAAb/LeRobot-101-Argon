"""Offline held-out workspace validation with an already validated wrist camera.

K, distortion and T_gripper_link_camera stay byte-for-byte unchanged. Only one
stationary T_base_board is fitted, using training poses. New profiles are
candidates: no hardware, configuration writes, or activation occur here.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .board_calibration import (BoardSpec, ValidationLimits, _detection, _hash,
                                _inverse, _observability, _pack, _project, _transform)
from .geometry import observation_families
from .spatial_contract import (JOINTS, UNCERTAINTY_FIELDS, WristCalibration,
                               build_wrist_observation, require_compatible)


EXTENSION_LIMITS = ValidationLimits(minimum_views=6, minimum_training_views=3,
                                     minimum_heldout_views=3)
UNCERTAINTY_AGGREGATION = 'max-parent-and-current-total-empirical-residual-v1'


def _bounds(frames):
    return {name: [min(f['joints'][name] for f in frames),
                   max(f['joints'][name] for f in frames)] for name in JOINTS[:-1]}


def _camera_board(points, pixels, k, distortion):
    ok, r, t = cv2.solvePnP(points, pixels, k, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('棋盤姿態無法解算，請檢查完整角點與影像品質。')
    result = _transform(r, t)
    _project(points, result, k, distortion)  # Reject behind-camera PnP branches.
    return result


def _rotation_error(a, b):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(a[:3, :3].T @ b[:3, :3])[0])))


def _fit_board(poses, pixels, points, k, distortion, handeye):
    # PnP initialization and least-squares both consume training views only.
    boards = [pose @ handeye @ _camera_board(points, image, k, distortion)
              for pose, image in zip(poses, pixels)]
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_matrix(np.array([b[:3, :3] for b in boards])).mean().as_matrix()
    initial[:3, 3] = np.mean([b[:3, 3] for b in boards], axis=0)

    def residual(parameters):
        board = _transform(parameters[:3], parameters[3:])
        return np.concatenate([(_project(points, _inverse(pose @ handeye) @ board,
                                          k, distortion) - image).ravel()
                               for pose, image in zip(poses, pixels)])

    fit = least_squares(residual, _pack(initial), loss='soft_l1', f_scale=1., max_nfev=240)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('固定相機參數下的棋盤位置尚未收斂。')
    return _transform(fit.x[:3], fit.x[3:])


def validate_workspace_extension(frames, model, profile, *, camera_id, mount_revision,
                                 geometry_revision, board_session_id, board=BoardSpec(),
                                 holdout_indices=None):
    """Validate new measured poses; only a passing report contains a new profile.

    Splits use whole near-duplicate pose families, as in board_calibration.
    The first member represents each family, so repeats neither inflate sample
    counts nor weight fitting. Automatic holdout uses alternate pose families,
    chosen from encoder positions before any corner residual is computed.
    """
    limits = EXTENSION_LIMITS
    report = {'status': 'rejected', 'source': 'wrist-only', 'calibration_profile': None,
              'method': 'frozen-camera-workspace-extension', 'validation_policy_version': 3,
              'metric_accuracy': 'approximate', 'grasp_ready': False, 'activation_performed': False,
              'board_session_id': board_session_id, 'board_spec': board.to_dict(),
              'validation': {}, 'blockers': []}
    try:
        if not isinstance(board_session_id, str) or not board_session_id.strip():
            raise ValueError('需要獨立固定棋盤的工作階段識別。')
        if not frames:
            raise ValueError('需要至少 6 個不同且停穩的棋盤姿勢。')
        if not isinstance(profile, dict):
            raise ValueError('需要完整的原校準 profile。')
        original = deepcopy(profile)
        parent_hash = _hash(original)
        if (original.get('source') != 'wrist-only' or original.get('metric_calibrated') is not True
                or original.get('needs_more_views') is not False):
            raise ValueError('只能延伸已通過獨立驗證的腕部相機 profile。')
        ancestor = original
        while isinstance(ancestor, dict):
            if ancestor.get('board_session_id') == board_session_id:
                raise ValueError('需要新的固定棋盤工作階段，不能重用原 profile 的校準資料。')
            provenance = ancestor.get('extension_provenance', {})
            if not isinstance(provenance, dict):
                raise ValueError('原 profile 的延伸驗證來源無效。')
            if provenance.get('board_session_id') == board_session_id:
                raise ValueError('這個工作階段已用於延伸驗證，不能再次計為新證據。')
            ancestor = provenance.get('parent_profile')
        camera = WristCalibration.from_mapping(original, native_image_size=frames[0]['image_size'],
            camera_id=camera_id, mount_revision=mount_revision, geometry_revision=geometry_revision,
            cad_handeye=np.eye(4))
        if not camera.metric_calibrated or camera.validation_scope_json is None:
            raise ValueError('原 profile 缺少獨立驗證或實測工作範圍。')
        for name in UNCERTAINTY_FIELDS:
            value = original.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('原 profile 必須保留完整且有限的 uncertainty。')
        scope = original['validation_scope']
        old_poses = scope.get('sampled_joint_poses')
        if (not isinstance(old_poses, list) or not old_poses
                or any(not isinstance(p, dict) or set(p) != set(JOINTS[:-1])
                       or any(type(p[n]) is not int or not 0 <= p[n] <= 4095
                              for n in JOINTS[:-1]) for p in old_poses)):
            raise ValueError('原 profile 缺少實測姿勢，不能以範圍邊界代替舊驗證。')
        observations, poses, pixels = [], [], []
        frame_ids = set()
        for frame in frames:
            if frame.get('board_session_id') != board_session_id:
                raise ValueError('不能混用不同固定棋盤工作階段。')
            if frame.get('camera_id') != camera_id:
                raise ValueError('每張影像都必須明確識別同一腕部相機。')
            if frame['id'] in frame_ids:
                raise ValueError('同一張影像不能重複充當獨立驗證。')
            frame_ids.add(frame['id'])
            transforms = model.fk.transforms(frame['joints'])
            observation = build_wrist_observation(frame, transforms, geometry_revision=geometry_revision,
                calibration=camera, camera_id=camera_id, mount_revision=mount_revision)
            if not observation.timing.pose_bound:
                raise ValueError('棋盤影像未與停穩後實測關節位置配對。')
            detection = _detection(frame, board)
            corners = np.asarray(detection['corners_px'], dtype=float)
            w, h = frame['image_size']
            if (not detection['found'] or corners.shape != (board.rows * board.columns, 2)
                    or not np.isfinite(corners).all() or np.any(corners < 0)
                    or np.any(corners[:, 0] >= w) or np.any(corners[:, 1] >= h)):
                raise ValueError('每個驗證姿勢都需要完整、有效且綁定原影像的棋盤角點。')
            observations.append(observation); poses.append(np.asarray(transforms['gripper_link']))
            pixels.append(corners)
        require_compatible(observations)
        groups = observation_families(frames)
        validation = report['validation']
        validation.update(captured_views=len(frames), independent_pose_groups=len(groups),
            pose_groups=[[frames[i]['id'] for i in group] for group in groups],
            fitted_parameters=['T_base_board'], frozen_parameters=['camera_matrix', 'distortion', 'handeye'],
            heldout_data_used_for_fitting=False,
            heldout_used_for_shared_parameter_fitting=False,
            per_view_pnp_pose_fitted_for_evaluation=True,
            acceptance_limits={name: getattr(limits, name) for name in ('minimum_views',
                'minimum_training_views', 'minimum_heldout_views',
                'heldout_translation_m', 'heldout_rotation_deg')},
            parent_profile_sha256=parent_hash, parent_validation_id=original['validation_id'])
        validation['acceptance_limits'].update(per_view_pnp_reprojection_p95_px=limits.heldout_reprojection_px,
            training_translation_m=limits.heldout_translation_m, training_rotation_deg=limits.heldout_rotation_deg)
        if len(groups) < limits.minimum_views:
            report['status'] = 'insufficient'
            raise ValueError('需要至少 6 個不同姿勢；原地重拍與返回同一姿勢不算新驗證。')
        if holdout_indices is None:
            held_groups = groups[1::2]
        else:
            if (not isinstance(holdout_indices, (tuple, list)) or not holdout_indices
                    or len(set(holdout_indices)) != len(holdout_indices)
                    or any(type(i) is not int or not 0 <= i < len(frames) for i in holdout_indices)):
                raise ValueError('heldout indices 必須是本次影像的不重複有效索引。')
            held_groups = [g for g in groups if set(g).intersection(holdout_indices)]
        held_members = {i for group in held_groups for i in group}
        train_groups = [g for g in groups if not set(g).intersection(held_members)]
        train, held = [g[0] for g in train_groups], [g[0] for g in held_groups]
        validation.update(training_frame_ids=[frames[i]['id'] for i in train],
                          heldout_frame_ids=[frames[i]['id'] for i in held])
        if len(train) < limits.minimum_training_views or len(held) < limits.minimum_heldout_views:
            report['status'] = 'insufficient'
            raise ValueError('需要至少 3 個擬合姿勢及 3 個獨立驗證姿勢；重複姿勢整組留在同一側。')
        k, distortion, handeye = (np.asarray(original[n], dtype=float)
                                  for n in ('camera_matrix', 'distortion', 'handeye'))
        points = board.object_points()
        base_board = _fit_board([poses[i] for i in train], [pixels[i] for i in train],
                                points, k, distortion, handeye)
        # Per-view PnP is a nuisance-pose fit for optical residuals and metric
        # evaluation only. It never changes K/X/FK or the training-only Y.
        pnp, optics = [], []
        for index, image in enumerate(pixels):
            camera_board = _camera_board(points, image, k, distortion)
            pnp.append(camera_board)
            error = np.linalg.norm(_project(points, camera_board, k, distortion) - image, axis=1)
            optics.append({'frame_id': frames[index]['id'], 'rms_px': float(np.sqrt(np.mean(error ** 2))),
                           'reprojection_p95_px': float(np.percentile(error, 95))})
        validation.update(per_view_optics=optics,
            max_pnp_reprojection_p95_px=max(row['reprojection_p95_px'] for row in optics),
            optics_interpretation='Fixed K/distortion with one fitted board pose per image: checks local optical consistency, not a new intrinsic calibration.',
            chain_reprojection_is_diagnostic=True)
        rows, training_rows, train_errors = [], [], []
        predicted_cameras = [pose @ handeye for pose in poses]
        reference_corners = points @ base_board[:3, :3].T + base_board[:3, 3]
        for index in train + held:
            camera = predicted_cameras[index]
            predicted = _inverse(camera) @ base_board
            errors = np.linalg.norm(_project(points, predicted, k, distortion) - pixels[index], axis=1)
            measured = camera @ pnp[index]
            inferred_camera = base_board @ _inverse(pnp[index])
            measured_corners = points @ measured[:3, :3].T + measured[:3, 3]
            row = {'frame_id': frames[index]['id'],
                'reprojection_p95_px': float(np.percentile(errors, 95)),
                'board_translation_error_m': float(np.linalg.norm(measured[:3, 3] - base_board[:3, 3])),
                'board_rotation_error_deg': _rotation_error(base_board, measured),
                'board_corner_max_displacement_m': float(np.max(np.linalg.norm(measured_corners - reference_corners, axis=1))),
                'camera_center_displacement_m': float(np.linalg.norm(inferred_camera[:3, 3] - camera[:3, 3]))}
            if index in train:
                train_errors.extend(errors)
                training_rows.append(row)
                continue
            rows.append(row)
        pairwise = []
        for a, b in combinations(sorted(train + held), 2):
            if a not in held and b not in held:
                continue
            predicted_relative = _inverse(predicted_cameras[a]) @ predicted_cameras[b]
            measured_relative = pnp[a] @ _inverse(pnp[b])
            pairwise.append({'frame_ids': [frames[a]['id'], frames[b]['id']],
                'translation_error_m': float(np.linalg.norm(predicted_relative[:3, 3] - measured_relative[:3, 3])),
                'rotation_error_deg': _rotation_error(predicted_relative, measured_relative)})
        translation = max(row['board_translation_error_m'] for row in rows)
        rotation = max(row['board_rotation_error_deg'] for row in rows)
        pixel = max(row['reprojection_p95_px'] for row in rows)
        relative_translation = max(row['translation_error_m'] for row in pairwise)
        relative_rotation = max(row['rotation_error_deg'] for row in pairwise)
        base_displacement = max(row[key] for row in rows + training_rows for key in
            ('board_translation_error_m', 'board_corner_max_displacement_m', 'camera_center_displacement_m'))
        rms = float(np.sqrt(np.mean(np.square(train_errors))))
        p95 = float(np.percentile(train_errors, 95))
        train_bounds, held_bounds = _bounds([frames[i] for i in train]), _bounds([frames[i] for i in held])
        measured_frames = [frames[i] for i in train + held]
        bounds = _bounds(measured_frames)
        coverage = {n: {'previous_raw_bounds': scope['joint_raw_bounds'][n],
            'training_raw_bounds': train_bounds[n], 'heldout_raw_bounds': held_bounds[n],
            'training_span_deg': (train_bounds[n][1] - train_bounds[n][0]) * 360 / 4096,
            'heldout_span_deg': (held_bounds[n][1] - held_bounds[n][0]) * 360 / 4096,
            'extends_previous_bounds': bounds[n][0] < scope['joint_raw_bounds'][n][0]
                                       or bounds[n][1] > scope['joint_raw_bounds'][n][1],
            'heldout_outside_previous_count': sum(not scope['joint_raw_bounds'][n][0]
                <= frames[i]['joints'][n] <= scope['joint_raw_bounds'][n][1] for i in held)} for n in JOINTS[:-1]}
        validation.update(training_chain_rms_px=rms, training_chain_p95_px=p95,
            training_views=training_rows, heldout_views=rows,
            heldout_max_translation_m=translation, heldout_max_rotation_deg=rotation,
            heldout_max_reprojection_p95_px=pixel, board_plane_base=base_board.tolist(),
            heldout_max_board_corner_displacement_m=max(row['board_corner_max_displacement_m'] for row in rows),
            heldout_max_camera_center_displacement_m=max(row['camera_center_displacement_m'] for row in rows),
            sampled_base_displacement_bound_m=base_displacement,
            pairwise_camera_errors=pairwise, pairwise_includes_heldout=True,
            max_relative_translation_error_m=relative_translation,
            max_relative_rotation_error_deg=relative_rotation,
            measured_axis_coverage=coverage, observability=_observability([poses[i] for i in train + held]),
            empirical_bounds_only=True,
            interpretation='Only shared T_base_board is fitted from training. Per-view PnP poses only evaluate optical/metric residuals. Accuracy is approximate; residuals combine camera, FK, measurement and stationary-board assumptions, not identified servo error or grasp clearance.')
        # Include the aggregation policy in the hashed validation provenance,
        # so this candidate cannot share the older additive solve identity.
        validation.update(validation_policy_version=3, uncertainty_aggregation=UNCERTAINTY_AGGREGATION)
        if validation['max_pnp_reprojection_p95_px'] > limits.heldout_reprojection_px:
            raise ValueError('固定鏡頭模型的逐張棋盤投影品質未通過；請檢查角點與鏡頭模型。')
        if any(row['board_translation_error_m'] > limits.heldout_translation_m
               or row['board_rotation_error_deg'] > limits.heldout_rotation_deg for row in rows + training_rows):
            raise ValueError('固定相機參數未通過新工作範圍的獨立棋盤驗證；請查看各姿勢殘差。')
        extended = deepcopy(original)
        extended_scope = extended['validation_scope']
        extended_scope['joint_raw_bounds'] = {n: [min(scope['joint_raw_bounds'][n][0], bounds[n][0]),
            max(scope['joint_raw_bounds'][n][1], bounds[n][1])] for n in JOINTS[:-1]}
        extended_scope['sampled_joint_poses'] += [{n: f['joints'][n] for n in JOINTS[:-1]}
                                                 for f in measured_frames]
        pixel_bounds = np.array([np.min(np.concatenate(pixels), axis=0),
                                 np.max(np.concatenate(pixels), axis=0)])
        if scope.get('image_points_bbox_px') is not None:
            old_pixel_bounds = np.asarray(scope['image_points_bbox_px'], float)
            if old_pixel_bounds.shape != (2, 2) or not np.isfinite(old_pixel_bounds).all():
                raise ValueError('原 profile 的影像取樣範圍無效。')
            pixel_bounds = np.array([np.minimum(pixel_bounds[0], old_pixel_bounds[0]),
                                     np.maximum(pixel_bounds[1], old_pixel_bounds[1])])
        extended_scope.update(extrapolation_allowed=False, individually_identified_joint_zeros=False,
            jaw_contact_geometry_calibrated=False, image_points_bbox_px=pixel_bounds.tolist(),
            board_plane_base=base_board.tolist(),
            interpretation='Union of recorded camera/FK workspace samples from independent board sessions; joint bounds alone do not prove all combinations. This board plane belongs only to the latest session; parent planes remain in provenance.')
        extended['extension_provenance'] = {'parent_profile': original, 'parent_profile_sha256': parent_hash,
            'board_session_id': board_session_id, 'board_spec': board.to_dict(),
            'validation': deepcopy(validation),
            'frames': [{key: deepcopy(f[key]) for key in ('id', 'camera_generation', 'sequence',
                       'capture_monotonic', 'joints', 'image_sha256') if key in f} for f in frames]}
        identity = _hash(extended['extension_provenance'])
        extended.update(calibration_revision=f'workspace-{identity[:20]}',
            validation_id=f'workspace-holdout-{identity}', needs_more_views=False, metric_calibrated=True,
            board_session_id=board_session_id, board_fingerprint=board.fingerprint,
            metric_accuracy='approximate', grasp_ready=False)
        # Both sets evaluate the TOTAL error of the same frozen K/D/X/FK
        # estimator. The new measurements are not an additional error source.
        # Preserve the parent floor and the worst observed sample, rather than
        # making repeated workspace validations accumulate the same error.
        empirical = {'relative_translation_uncertainty_m': relative_translation,
                     'base_position_uncertainty_m': base_displacement,
                     'relative_rotation_uncertainty_deg': relative_rotation}
        for name, observed in empirical.items():
            extended[name] = max(original[name], observed)
        extended['uncertainty_interpretation'] = ('Maximum of parent and new empirical residual for the same frozen camera/FK estimator; '
            'the parent floor is retained and the two total-error estimates are not added. '
            'Relative-camera errors use same-board pairs containing heldout poses; unmeasured cross-session pairs are not validated. '
            'Focal uncertainty unchanged. Approximate sampled-workspace accuracy, not guaranteed error bounds or grasp-ready clearance.')
        report.update(status='validated', calibration_profile=extended)
    except (ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
        report['blockers'] = [str(exc)]
    return report
