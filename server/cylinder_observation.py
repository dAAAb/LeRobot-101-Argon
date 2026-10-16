"""Bind semantic image contours to measured wrist packets, without motor I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .cylinder_silhouette import fit_cylinder_silhouettes
from .spatial_contract import require_compatible
from .spatial_runtime import _profile_snapshot, _packet, packet_uncertainty

CONTOUR_PIXEL_SENSITIVITY = 6.  # Explicit local sensitivity assumption, not measured VLM accuracy.
# Same nominal revisit definition as active_observation; these are NOT an
# object-depth accuracy threshold or proof of useful triangulation.
DUPLICATE_VIEW_TRANSLATION_MM = 5.
DUPLICATE_VIEW_ROTATION_DEGREES = 2.


def distinct_camera_views(records):
    """Select latest representatives without counting jaw opening as a view.

    Inputs have already been bound to calibrated packets, in capture order.
    Compare calibrated camera SE(3), not jaw aperture or frame IDs. Greedy
    latest-first selection preserves the current photo and avoids chaining a
    sequence of small moves into one arbitrarily large pose family. A changed
    angle can remain useful, but the fitter must still establish translation
    baseline, rank and its unchanged uncertainty bounds.
    """
    from .active_depth import _rigid
    transforms = [_rigid(row['camera_transform_base_m']) for row in records]
    retained, groups, comparisons = [], {}, []
    for i in reversed(range(len(records))):
        current = transforms[i]
        distances = []
        for j in retained:
            previous = transforms[j]
            mm = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]) * 1000.)
            cosine = np.clip((np.trace(current[:3, :3] @ previous[:3, :3].T) - 1.) / 2., -1., 1.)
            degrees = float(np.degrees(np.arccos(cosine)))
            distances.append((max(mm / DUPLICATE_VIEW_TRANSLATION_MM,
                                  degrees / DUPLICATE_VIEW_ROTATION_DEGREES), j, mm, degrees))
        nearest = min(distances, default=None)
        if (nearest is not None and nearest[2] <= DUPLICATE_VIEW_TRANSLATION_MM
                and nearest[3] <= DUPLICATE_VIEW_ROTATION_DEGREES):
            _, representative, mm, degrees = nearest
            groups[representative].append(i)
            comparisons.append(dict(frame_id=records[i]['frame_id'],
                representative_frame_id=records[representative]['frame_id'],
                translation_mm=mm, rotation_degrees=degrees))
        else:
            retained.append(i)
            groups[i] = [i]
    retained.sort()
    baseline = max((float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000.)
                    for a in transforms for b in transforms), default=0.)
    return retained, dict(policy='calibrated-camera-latest-pose-representatives-v1',
        input_frame_count=len(records), distinct_pose_count=len(retained), required_pose_count=3,
        retained_frame_ids=[records[i]['frame_id'] for i in retained],
        families=[dict(representative_frame_id=records[i]['frame_id'],
                       frame_ids=[records[j]['frame_id'] for j in sorted(groups[i])]) for i in retained],
        duplicate_comparisons=comparisons, maximum_camera_translation_mm=baseline,
        duplicate_thresholds=dict(translation_mm=DUPLICATE_VIEW_TRANSLATION_MM,
                                  rotation_degrees=DUPLICATE_VIEW_ROTATION_DEGREES),
        actual_information_gain_verified=False,
        interpretation='Distinct nominal camera poses only; jaw opening may reveal pixels but does not add camera baseline. The existing solver still checks scale observability and all uncertainty bounds.')


def hypothesis_from_wrist_contours(model, frames, contours, config, *, current_frame_id,
                                  track_id, support_surface):
    """One inference's contours may refer ONLY to its actual supplied images.

    Fewer views is an ordinary observation need, never missing robot setup.
    The calibrated paper plane applies only when the semantic observation says
    the object directly rests on that same main table; another object/support
    or an unknown support never inherits that plane.
    """
    if not contours or len(contours) < 3:
        return {'status': 'insufficient', 'reason': '還需要至少三張不同位置的可見筒身輪廓。'}
    if support_surface not in ('calibrated_main_table', 'other_or_unknown'):
        raise ValueError('物件支撐面需要明確標示')
    by_id = {}
    for frame in frames:
        if frame.get('id') in by_id and by_id[frame['id']] != frame:
            raise ValueError('同一影像 ID 對應到不同取像資料')
        by_id[frame.get('id')] = frame
    ids = [row['frame_id'] for row in contours]
    if len(set(ids)) != len(ids) or current_frame_id not in ids or any(fid not in by_id for fid in ids):
        raise ValueError('輪廓必須對應本次提供的影像，並包含目前畫面')
    selected = sorted([by_id[fid] for fid in ids], key=lambda f: f['capture_monotonic'])
    if any(frame.get('source') != 'wrist' for frame in selected):
        raise ValueError('圓筒輪廓只使用腕部相機影像')
    if selected[-1]['id'] != current_frame_id:
        raise ValueError('輪廓的最後一張必須是目前腕部影像')
    snapshot = _profile_snapshot(model, config, include_jaws=False)
    packets = [_packet(model, frame, snapshot) for frame in selected]
    require_compatible(packets)
    if not all(packet.metric_calibrated for packet in packets):
        raise ValueError('輪廓影像必須在同一相機校準範圍內，並與停穩後的實測關節位置對齊')
    if any(a.sequence >= b.sequence or a.capture_monotonic >= b.capture_monotonic
           for a, b in zip(packets, packets[1:])):
        raise ValueError('輪廓影像需有依序增加的實際取像時間')
    profile = snapshot[1]
    annotations = {row['frame_id']: row for row in contours}
    records = []
    for frame, packet in zip(selected, packets):
        image = cv2.imread(frame['path'], cv2.IMREAD_GRAYSCALE)
        size = np.asarray(packet.processing.output_size)
        if image is None or tuple(image.shape[::-1]) != tuple(size):
            raise ValueError('輪廓的像素尺寸與實際腕部照片不一致')
        row = annotations[frame['id']]
        sides = np.asarray(row['body_sides_normalized'], float)
        foot = np.asarray(row.get('support_boundary_normalized', []), float)
        if (sides.ndim != 3 or sides.shape[0] != 2 or not 2 <= sides.shape[1] <= 6 or sides.shape[2] != 2
                or not np.isfinite(sides).all() or np.any(sides < 0) or np.any(sides >= 1)
                or foot.size and (foot.ndim != 2 or foot.shape[1] != 2 or len(foot) > 6
                    or not np.isfinite(foot).all() or np.any(foot < 0) or np.any(foot >= 1))):
            raise ValueError('輪廓座標需要明確的 0–1 畫面比例，不能混用像素或物件尺寸')
        records.append(dict(source='wrist', frame_id=frame['id'], camera_generation=packet.camera_generation,
            calibration_fingerprint=packet.calibration.fingerprint, image_size=size.tolist(),
            camera_transform_base_m=packet.T_base_camera.as_array().tolist(),
            camera_matrix=[list(r) for r in packet.camera_matrix], distortion=list(packet.distortion),
            body_sides_px=(sides * size).tolist(), support_boundary_px=(foot * size).tolist() if foot.size else [],
            measured_joints=dict(packet.measured_joints), capture_monotonic=packet.capture_monotonic,
            image_sha256=hashlib.sha256(Path(frame['path']).read_bytes()).hexdigest()))
    retained, diversity = distinct_camera_views(records)
    records = [records[i] for i in retained]
    current = packets[-1]
    if len(retained) < 3:
        return dict(status='insufficient', reason='角度變化還不夠，需要移動肩肘換個位置。',
            source='wrist-only', frame_id=current_frame_id, track_id=track_id,
            camera_generation=current.camera_generation,
            calibration_fingerprint=current.calibration.fingerprint,
            calibration_revision=current.calibration.calibration_revision,
            geometry_revision=current.calibration.geometry_revision,
            coordinate_calibrated=True, metric_calibrated=False,
            planning_eligible=False, execution_authorized=False,
            silhouette_evidence=records, view_diversity=diversity)
    scope = profile.get('validation_scope') or {}
    plane_value = (scope.get('board_planes_by_session') or {}).get(profile.get('board_session_id'))
    if plane_value is None:
        plane_value = scope.get('board_plane_base')
    support = None; axis = [0., 0., 1.]
    if (support_surface == 'calibrated_main_table' and plane_value is not None
            and sum(bool(row['support_boundary_px']) for row in records) >= 2):
        from .active_depth import _rigid
        plane = _rigid(plane_value); normal = plane[:3, 2]
        if normal[2] < 0:
            normal = -normal
        axis = normal.tolist()
        support = dict(point_base_m=plane[:3, 3].tolist(), normal_base=axis,
            assumption='The identified cylinder directly rests on the same main table plane as the calibrated paper. Camera/FK residual is relative to this plane gauge; this is not independent physical support verification.')
    uncertainty = packet_uncertainty(packets)
    bounds = dict(pixel_edge_mislocalization_px=CONTOUR_PIXEL_SENSITIVITY,
        relative_translation_m=uncertainty['relative_translation_uncertainty_m'],
        relative_rotation_deg=uncertainty['relative_rotation_uncertainty_deg'],
        focal_relative=uncertainty['focal_relative_uncertainty'],
        common_translation_m=uncertainty['base_position_uncertainty_m'], common_rotation_deg=0.,
        # The selected calibration declares one combined camera/FK residual.
        # Do not charge that SAME residual again as independent plane noise.
        support_plane_translation_m=0., support_plane_rotation_deg=0.)
    result = fit_cylinder_silhouettes(records, shape_intent='upright_cylinder',
        axis_base=axis, uncertainty=bounds, support_plane=support)
    result.update(frame_id=current_frame_id, track_id=track_id,
        calibration_revision=current.calibration.calibration_revision,
        geometry_revision=current.calibration.geometry_revision,
        coordinate_calibrated=True, metric_calibrated=False,
        planning_eligible=True,  # Conditional input; never motor permission.
        silhouette_evidence=records, support_surface=support_surface,
        view_diversity=diversity,
        runtime_calibration_scopes=[json.loads(packet.scope_reserve_json) if packet.scope_reserve_json else None
                                    for packet in packets],
        support_plane_used=support is not None,
        support_plane_provenance='same camera calibration fingerprint / validation_scope' if support else None,
        observed_axial_span_m=float(np.ptp(result['observed_axial_interval_m'])))
    return result
