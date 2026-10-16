"""Conditional upright-cylinder geometry from actual wrist silhouettes.

No device access, dimensions from labels, object-size defaults, or motion
authorization. Each observed side is a visible BODY boundary, not a cap,
finger, bounding-box edge, or correspondence between material surface points.
"""
from __future__ import annotations

from copy import deepcopy
import math

import cv2
import numpy as np

from .active_depth import _rigid


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def _basis(axis):
    axis = np.asarray(axis, float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis) - 1) > 1e-6:
        raise ValueError('An explicit unit upright axis is required')
    u = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))]); u /= np.linalg.norm(u)
    return axis, np.stack([u, np.cross(axis, u)], axis=1)


def _views(records):
    if not isinstance(records, list) or not 3 <= len(records) <= 8:
        raise ValueError('Use three to eight actual wrist views')
    views, ids, generations, calibrations = [], set(), set(), set()
    for row in records:
        if row.get('source') != 'wrist' or not isinstance(row.get('frame_id'), str) or not row['frame_id']:
            raise ValueError('Every contour must identify its actual wrist frame')
        if row['frame_id'] in ids:
            raise ValueError('Repeated frames do not create independent observations')
        ids.add(row['frame_id'])
        for name, destination in [('camera_generation', generations), ('calibration_fingerprint', calibrations)]:
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError('Bind camera generation and calibration fingerprint')
            destination.add(row[name])
        t = _rigid(row['camera_transform_base_m'])
        k = np.asarray(row['camera_matrix'], float)
        d = np.asarray(row['distortion'], float)
        size = np.asarray(row['image_size'])
        if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1]) or d.ndim != 1 or len(d) not in (4, 5, 8, 12, 14)
                or not np.isfinite(d).all() or size.shape != (2,) or np.any(size <= 0)):
            raise ValueError('Invalid native image camera parameters')
        sides = np.asarray(row['body_sides_px'], float)
        if (sides.ndim != 3 or sides.shape[0] != 2 or not 2 <= sides.shape[1] <= 32 or sides.shape[2] != 2
                or not np.isfinite(sides).all() or np.any(sides < 0) or np.any(sides >= size)
                or min(np.linalg.norm(side[-1] - side[0]) for side in sides) < 20):
            raise ValueError('Need two distinct visible body-edge polylines in native image pixels')
        foot = np.asarray(row.get('support_boundary_px', []), float)
        if foot.size and (foot.ndim != 2 or foot.shape[1] != 2 or len(foot) > 32
                or not np.isfinite(foot).all() or np.any(foot < 0) or np.any(foot >= size)):
            raise ValueError('Support contour must contain actual visible bottom-boundary pixels')
        views.append(dict(transform=t.copy(), k=k.copy(), distortion=d.copy(), sides=sides.copy(), foot=foot.copy()))
    if len(generations) != 1 or len(calibrations) != 1:
        raise ValueError('Do not mix camera sessions or calibration profiles')
    baseline = max(np.linalg.norm(a['transform'][:3, 3] - b['transform'][:3, 3]) for a in views for b in views)
    if baseline < .001:
        raise ValueError('Camera rotation alone does not establish metric object scale')
    return views


def _plane(value):
    if value is None:
        return None
    point, normal = np.asarray(value['point_base_m'], float), np.asarray(value['normal_base'], float)
    if (point.shape != (3,) or normal.shape != (3,) or not np.isfinite([point, normal]).all()
            or abs(np.linalg.norm(normal) - 1) > 1e-6
            or not isinstance(value.get('assumption'), str) or len(value['assumption']) < 20):
        raise ValueError('Declare the calibrated support plane and its tabletop assumption')
    return dict(point_base_m=point.copy(), normal_base=normal.copy(), assumption=value['assumption'])


def _solve(views, axis, support_plane=None):
    """Signed tangent planes give a joint linear center/radius fit."""
    axis, basis = _basis(axis)
    rows, rhs, planes, alignment = [], [], [], []
    for view in views:
        rays = []
        for side in view['sides']:
            xy = cv2.undistortPoints(side.reshape(-1, 1, 2), view['k'], view['distortion']).reshape(-1, 2)
            ray = np.c_[xy, np.ones(len(xy))] @ view['transform'][:3, :3].T
            ray /= np.linalg.norm(ray, axis=1)[:, None]
            rays.append(ray)
        interior = np.mean(rays, axis=(0, 1))
        for ray in rays:
            # Fit the closest plane containing the declared cylinder axis.
            # Its disagreement with the observed free plane is NOT discarded.
            _, _, vt = np.linalg.svd(ray @ basis, full_matrices=False)
            normal = basis @ vt[-1]
            if normal @ interior < 0:
                normal = -normal
            if normal @ interior <= 1e-8:
                raise ValueError('The two body boundaries do not enclose a visible interior')
            observed = np.cross(ray[0], ray[-1]); observed /= np.linalg.norm(observed)
            alignment.append(float(math.degrees(math.asin(np.clip(abs(observed @ axis), 0, 1)))))
            rows.append([*(normal @ basis), -1.])
            origin = view['transform'][:3, 3]
            rhs.append(normal @ origin)
            planes.append((normal, origin, ray))
    a, b = np.asarray(rows), np.asarray(rhs)
    if np.linalg.matrix_rank(a) != 3:
        raise ValueError('Silhouette baselines do not resolve center and radius')
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    feet, foot_cameras = [], []
    if support_plane is not None:
        plane_point, plane_normal = support_plane['point_base_m'], support_plane['normal_base']
        if abs(float(plane_normal @ axis)) < .999:
            raise ValueError('A standing cylindrical body must be normal to its assumed support plane')
        for view in views:
            if not len(view['foot']):
                continue
            xy = cv2.undistortPoints(view['foot'].reshape(-1, 1, 2), view['k'], view['distortion']).reshape(-1, 2)
            rays = np.c_[xy, np.ones(len(xy))] @ view['transform'][:3, :3].T
            origin = view['transform'][:3, 3]
            denom = rays @ plane_normal
            if np.any(abs(denom) < 1e-6):
                raise ValueError('A bottom-boundary ray is parallel to its support plane')
            distance = ((plane_point - origin) @ plane_normal) / denom
            if np.any(distance <= 0) or not np.isfinite(distance).all():
                raise ValueError('A support-plane intersection is behind the camera')
            feet.extend(origin + distance[:, None] * rays)
            foot_cameras.extend([origin] * len(rays))
        if not feet:
            raise ValueError('The support plane needs a visible actual bottom boundary')
        from scipy.optimize import minimize
        footprint, camera_xy = np.asarray(feet) @ basis, np.asarray(foot_cameras) @ basis
        def front_distance(p):
            toward = camera_xy - p[:2]
            toward /= np.maximum(np.linalg.norm(toward, axis=1), 1e-12)[:, None]
            return np.einsum('ij,ij->i', footprint - p[:2], toward)
        def front_jacobian(p):
            toward = camera_xy - p[:2]
            length = np.maximum(np.linalg.norm(toward, axis=1), 1e-12)
            direction = toward / length[:, None]
            offset = footprint - p[:2]
            distance = np.einsum('ij,ij->i', offset, direction)
            derivative = -direction - (offset - distance[:, None] * direction) / length[:, None]
            return np.c_[derivative, np.zeros(len(footprint))]
        def residual(p):
            return np.r_[a @ p - b, np.linalg.norm(footprint - p[:2], axis=1) - p[2]]
        def objective(p):
            error = residual(p)
            return float(error @ error) * 1e6
        def objective_jacobian(p):
            offset = p[:2] - footprint
            length = np.maximum(np.linalg.norm(offset, axis=1), 1e-12)
            jacobian = np.vstack([a, np.c_[offset / length[:, None], -np.ones(len(footprint))]])
            return 2e6 * (residual(p) @ jacobian)
        # Circle/ray intersections have a front and a hidden-back branch.
        # A visible lower outline cannot authorize the hidden branch merely
        # because that local optimizer minimum has a smaller plane residual.
        away = np.mean(footprint - camera_xy, axis=0); away /= np.linalg.norm(away)
        r0 = max(.0001, solution[2])
        seeds = [np.r_[solution[:2], r0], np.r_[np.mean(footprint, axis=0) + r0 * away, r0]]
        # Visibility is a constraint, not a soft residual followed by a
        # rejection. A noisy near-tangent pixel can put the penalized optimum
        # just behind the center even when a valid boundary optimum exists.
        # Millimetre-scaled errors and analytic derivatives keep that boundary
        # stable during the small perturbations used for sensitivity below.
        # Optimize local millimetres as well as millimetre residuals. Using
        # metre variables with a large residual scale makes SLSQP's line
        # search unreliable at an active visibility constraint.
        offset = np.r_[np.mean(footprint, axis=0), 0.]
        def metres(q):
            return offset + .001 * q
        fits = []
        for seed in seeds:
            fit = minimize(lambda q: objective(metres(q)), (seed - offset) * 1000,
                jac=lambda q: .001 * objective_jacobian(metres(q)), method='SLSQP',
                bounds=[(None, None), (None, None), (.01, None)],
                constraints=[dict(type='ineq', fun=lambda q: 1000 * front_distance(metres(q)),
                                  jac=lambda q: front_jacobian(metres(q)))],
                options=dict(maxiter=200, ftol=1e-10))
            fit.x = metres(fit.x)
            fits.append(fit)
        fits = [fit for fit in fits if fit.success and np.min(front_distance(fit.x)) >= -1e-7]
        if not fits:
            raise ValueError('Silhouettes and support boundary do not yield a converged cylinder')
        fit = min(fits, key=lambda fit: float(fit.fun))
        solution = fit.x
    center, radius = basis @ solution[:2], float(solution[2])
    if not np.isfinite(solution).all() or radius <= 0:
        raise ValueError('No positive finite cylinder explains these silhouettes')
    heights, pixel_residuals = [], []
    ray_residuals_by_view = [[] for _ in views]
    for plane_index, (normal, origin, rays) in enumerate(planes):
        if np.linalg.norm((center - origin) @ basis) <= radius:
            raise ValueError('A camera cannot lie inside the hypothesized cylinder')
        tangent = center - radius * normal
        for ray in rays:
            coefficients = np.linalg.lstsq(np.stack([ray, -axis], axis=1), tangent - origin, rcond=None)[0]
            if coefficients[0] <= 0:
                raise ValueError('Silhouette body is behind its camera')
            heights.append(float(coefficients[1]))
        # Actual rays, rather than a projected bounding-box width, constrain
        # the tangent plane. Angular residual remains explicit below.
        ray_errors = np.arcsin(np.clip(np.abs(rays @ normal), 0, 1)).tolist()
        pixel_residuals.extend(ray_errors)
        ray_residuals_by_view[plane_index // 2].extend(ray_errors)
    coordinate = float(np.median(heights))
    output = np.r_[center + axis * coordinate, radius]
    return output, dict(tangent_plane_rms_m=float(np.sqrt(np.mean((a @ solution - b) ** 2))),
        tangent_plane_residuals_m=(a @ solution - b).tolist(),
        condition_number=float(np.linalg.cond(a)),
        upright_plane_disagreement_deg=alignment,
        side_ray_residual_p95_deg=float(np.degrees(np.quantile(pixel_residuals, .95))),
        side_ray_residual_max_deg_by_view=[float(np.degrees(max(errors))) for errors in ray_residuals_by_view],
        observed_axial_interval_m=np.quantile(heights, [.05, .95]).tolist(),
        support_boundary_points_base_m=np.asarray(feet).tolist(),
        support_boundary_front_distance_m=([] if not feet else front_distance(solution).tolist()),
        support_boundary_radial_residual_m=([] if not feet else
            (np.linalg.norm(np.asarray(feet) @ basis - solution[:2], axis=1) - radius).tolist()))


def fit_cylinder_silhouettes(records, *, shape_intent, axis_base, uncertainty, support_plane=None):
    """Return a conditional shape prior with correlated center/radius sensitivity.

    Uncertainty inputs are explicit engineering scales, not claimed Gaussian
    measurements. Independent relative-view and coherent camera-local mount
    perturbations are alternative modes, not two independent error sources.
    Their largest position/radius sensitivities are used. Shared focal/rigid
    placement perturbations remain shared in both modes.
    Coherent edge shift/tilt avoids treating many adjacent contour pixels as
    independent measurements that spuriously improve accuracy.
    """
    if shape_intent != 'upright_cylinder':
        raise ValueError('This fit requires an explicit upright-cylinder shape prior')
    axis, _ = _basis(axis_base)
    views = _views(records)
    plane = _plane(support_plane)
    fields = ('pixel_edge_mislocalization_px', 'relative_translation_m', 'relative_rotation_deg',
              'focal_relative', 'common_translation_m', 'common_rotation_deg')
    scales = {n: _number(uncertainty[n], n) for n in fields}
    if scales['focal_relative'] >= 1:
        raise ValueError('Focal sensitivity must preserve positive focal length')
    if plane is not None:
        for n in ('support_plane_translation_m', 'support_plane_rotation_deg'):
            scales[n] = _number(uncertainty[n], n)
    nominal, diagnostics = _solve(views, axis, plane)
    # Joint center/radius agreement cannot establish that each observed edge
    # really is upright. Fitting the nearest axis-containing tangent plane
    # must not silently discard a tilted or curved contour. Check every ray,
    # not just endpoints or a percentile that could hide one bad annotation.
    # Pure camera translation and a common rigid base relabeling do not
    # change the compatibility between an edge direction and this axis.
    ray_allowances = []
    for view in views:
        pixel_angle = math.atan2(scales['pixel_edge_mislocalization_px'], min(view['k'][0, 0], view['k'][1, 1]))
        # Conservative angular bound for the declared shared focal stretch.
        focal_angle = math.asin(min(1., scales['focal_relative'] / (1. - scales['focal_relative'])))
        ray_allowances.append(scales['relative_rotation_deg'] + math.degrees(pixel_angle + focal_angle))
    if any(error > allowance + 1e-7 for error, allowance in
           zip(diagnostics['side_ray_residual_max_deg_by_view'], ray_allowances)):
        raise ValueError('Body-edge rays disagree with the upright axis beyond the declared pixel/rotation error')
    diagnostics['side_ray_allowance_deg_by_view'] = ray_allowances
    # A fitted positive radius is not enough: hold the actual plane residuals
    # against the same explicit pose/pixel scales used below, without fitting
    # those scales to the requested object or its familiar size.
    tolerances = []
    for view in views:
        distance = float(np.linalg.norm(nominal[:3] - view['transform'][:3, 3]))
        angular = math.radians(scales['relative_rotation_deg'])
        angular += math.atan2(scales['pixel_edge_mislocalization_px'], min(view['k'][0, 0], view['k'][1, 1]))
        tolerances.extend([scales['relative_translation_m'] + distance * math.sin(min(math.pi / 2, angular))] * 2)
    if any(abs(error) > allowance + 1e-7 for error, allowance in
           zip(diagnostics['tangent_plane_residuals_m'], tolerances)):
        raise ValueError('Views disagree beyond the declared contour/camera error')
    diagnostics['tangent_plane_allowance_m'] = tolerances
    covariance = np.zeros((4, 4))
    contributions = {}

    def evaluate(change, step, *, change_plane=None):
        values = []
        for sign in (-1., 1.):
            modified = deepcopy(views)
            modified_plane = deepcopy(plane)
            changed_axis = change(modified, axis.copy(), sign * step)
            if change_plane is not None:
                changed_axis = change_plane(modified_plane, changed_axis, sign * step)
            values.append(_solve(modified, changed_axis, modified_plane)[0])
        return (values[1] - values[0]) / (2 * step)

    def accumulate(name, columns, scale):
        nonlocal covariance
        j = np.column_stack(columns)
        c = j @ j.T * scale ** 2
        covariance += c
        contributions[name] = c.tolist()

    for i, view in enumerate(views):
        for kind, step, scale in [('translation', 1e-5, scales['relative_translation_m']),
                                  ('rotation', 1e-5, math.radians(scales['relative_rotation_deg']))]:
            columns = []
            for j in range(3):
                def change(v, ax, amount, i=i, j=j, kind=kind):
                    if kind == 'translation':
                        v[i]['transform'][j, 3] += amount
                    else:
                        rotation = cv2.Rodrigues(np.eye(3)[j] * amount)[0]
                        v[i]['transform'][:3, :3] = rotation @ v[i]['transform'][:3, :3]
                    return ax
                columns.append(evaluate(change, step))
            accumulate(f'view_{i}_{kind}', columns, scale)
        for side_index, side in enumerate(view['sides']):
            direction = side[-1] - side[0]; direction /= np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            for mode in ('shift', 'tilt'):
                weights = np.ones(len(side)) if mode == 'shift' else np.linspace(-1., 1., len(side))
                def change(v, ax, amount, i=i, side_index=side_index, normal=normal, weights=weights):
                    v[i]['sides'][side_index] += amount * weights[:, None] * normal
                    return ax
                accumulate(f'view_{i}_side_{side_index}_{mode}', [evaluate(change, .01)],
                           scales['pixel_edge_mislocalization_px'])
        if plane is not None and len(view['foot']):
            columns = []
            for j in range(2):
                def change(v, ax, amount, i=i, j=j):
                    v[i]['foot'][:, j] += amount
                    return ax
                columns.append(evaluate(change, .01))
            accumulate(f'view_{i}_bottom_contour', columns, scales['pixel_edge_mislocalization_px'])
    def focal(v, ax, amount):
        for view in v:
            view['k'][0, 0] *= 1 + amount; view['k'][1, 1] *= 1 + amount
        return ax
    accumulate('shared_focal', [evaluate(focal, 1e-5)], scales['focal_relative'])
    # A common SE3 displacement affects center but CANNOT change a radius.
    shared_translation = np.vstack([np.eye(3), np.zeros(3)])
    accumulate('common_translation', list(shared_translation.T), scales['common_translation_m'])
    columns = []
    for j in range(3):
        def rotate(v, ax, amount, j=j):
            r = cv2.Rodrigues(np.eye(3)[j] * amount)[0]
            for view in v:
                view['transform'][:3] = r @ view['transform'][:3]
            return r @ ax
        def rotate_plane(p, ax, amount, j=j):
            if p is not None:
                r = cv2.Rodrigues(np.eye(3)[j] * amount)[0]
                p['point_base_m'] = r @ p['point_base_m']; p['normal_base'] = r @ p['normal_base']
            return ax
        columns.append(evaluate(rotate, 1e-5, change_plane=rotate_plane))
    accumulate('common_rotation_with_axis', columns, math.radians(scales['common_rotation_deg']))
    if plane is not None:
        # Relative plane uncertainty stays separate from a common coordinate
        # change. These explicit scales must not be silently charged twice
        # when a caller's camera bound already contains the same plane error.
        for kind, scale in [('translation', scales['support_plane_translation_m']),
                            ('rotation', math.radians(scales['support_plane_rotation_deg']))]:
            columns = []
            for j in range(3):
                def change(p, ax, amount, j=j, kind=kind):
                    if kind == 'translation':
                        p['point_base_m'][j] += amount
                        return ax
                    r = cv2.Rodrigues(np.eye(3)[j] * amount)[0]
                    p['normal_base'] = r @ p['normal_base']
                    return r @ ax
                columns.append(evaluate(lambda v, ax, amount: ax, 1e-5, change_plane=change))
            accumulate('relative_support_plane_' + kind, columns, scale)
    # A common mount error is NOT a common change of base coordinates: the
    # same camera-local displacement follows each view's different rotation.
    # Replace, rather than add to, the independent relative-pose covariance.
    independent_covariance = covariance.copy()
    independent_contributions = deepcopy(contributions)
    relative_names = [name for name in contributions
                      if name.startswith('view_') and name.rsplit('_', 1)[-1] in ('translation', 'rotation')]
    shared_contributions = {name: value for name, value in contributions.items() if name not in relative_names}
    covariance = sum((np.asarray(value) for value in shared_contributions.values()), np.zeros((4, 4)))
    contributions = shared_contributions
    for kind, scale in [('translation', scales['relative_translation_m']),
                        ('rotation', math.radians(scales['relative_rotation_deg']))]:
        columns = []
        for j in range(3):
            def change(v, ax, amount, j=j, kind=kind):
                delta = np.eye(3)[j] * amount
                for view in v:
                    rotation = view['transform'][:3, :3]
                    if kind == 'translation':
                        view['transform'][:3, 3] += rotation @ delta
                    else:
                        view['transform'][:3, :3] = rotation @ cv2.Rodrigues(delta)[0]
                return ax
            columns.append(evaluate(change, 1e-5))
        accumulate('shared_camera_local_' + kind, columns, scale)
    mode_covariances = {'independent_views': independent_covariance,
                        'shared_camera_local_mount': covariance.copy()}
    mode_contributions = {'independent_views': independent_contributions,
                          'shared_camera_local_mount': deepcopy(contributions)}
    if not all(np.isfinite(value).all() for value in mode_covariances.values()):
        raise ValueError('Unbounded conditional cylinder sensitivity')
    modes = {name: dict(conditional_center_radius_covariance=value.tolist(),
        center_position_sensitivity_m=float(np.sqrt(max(0., np.linalg.eigvalsh(value[:3, :3])[-1]))),
        radius_sensitivity_m=float(np.sqrt(max(0., value[3, 3]))),
        covariance_contributions=mode_contributions[name]) for name, value in mode_covariances.items()}
    center_mode = max(modes, key=lambda name: modes[name]['center_position_sensitivity_m'])
    radius_mode = max(modes, key=lambda name: modes[name]['radius_sensitivity_m'])
    # Retain a named real covariance, not a fabricated covariance made by
    # adding mutually exclusive modes. Scalar planning budgets below take
    # the larger sensitivity independently for center and radius.
    covariance = mode_covariances[center_mode]
    contributions = mode_contributions[center_mode]
    # Hold out entire views; adjacent edge samples are not held-out geometry.
    alternatives = [_solve(views[:i] + views[i + 1:], axis, plane)[0] for i in range(len(views))]
    spread = np.asarray(alternatives) - nominal
    center_spread = float(np.max(np.linalg.norm(spread[:, :3], axis=1)))
    radius_spread = float(np.max(abs(spread[:, 3])))
    center_error = max(modes[center_mode]['center_position_sensitivity_m'], center_spread)
    radius_error = max(modes[radius_mode]['radius_sensitivity_m'], radius_spread)
    budget_sources = dict(center=(center_mode if modes[center_mode]['center_position_sensitivity_m'] >= center_spread
                                  else 'leave_one_view_out'),
                          radius=(radius_mode if modes[radius_mode]['radius_sensitivity_m'] >= radius_spread
                                  else 'leave_one_view_out'))
    return dict(status='hypothesis', source='wrist-only', shape_intent=shape_intent,
        geometry_method='multiview-body-silhouette-tangent-planes' + ('-and-support-plane' if plane is not None else ''),
        accuracy_basis='conditional-upright-cylinder-and-camera-model',
        frame_ids=[r['frame_id'] for r in records], camera_generation=records[0]['camera_generation'],
        calibration_fingerprint=records[0]['calibration_fingerprint'],
        centerline_point_base_m=nominal[:3].tolist(), radius_m=float(nominal[3]), diameter_m=float(2 * nominal[3]),
        axis_base=axis.tolist(), full_height_m=None,
        observed_axial_interval_m=diagnostics['observed_axial_interval_m'],
        center_position_error_m=center_error, radius_sensitivity_m=radius_error,
        conditional_center_radius_covariance=covariance.tolist(), covariance_contributions=contributions,
        conditional_covariance_mode=center_mode, uncertainty_modes=modes,
        uncertainty_budget_sources=budget_sources,
        uncertainty_mode_combination='maximum of alternative modes and whole-view refit spread; modes are not added',
        leave_one_view_out_solutions=[v.tolist() for v in alternatives], diagnostics=diagnostics,
        assumptions={**scales, 'support_plane': None if plane is None else plane['assumption'],
                     'relative_views': 'maximum of independent-view and shared camera-local mount sensitivity modes; same scales, not measured covariance',
                     'edges': 'coherent side shift and tilt; adjacent pixels are not independent samples',
                     'shape': 'circular upright body; caps, taper, hidden surfaces and complete height are not modeled'},
        metric_accuracy_independently_validated=False, planning_eligible=False,
        grasp_ready=False, execution_authorized=False, object_dimensions_used=False,
        uncertainty_kind='conditional linearized sensitivity and whole-view refit spread, not a physical guarantee',
        reason='依多視角筒身兩側輪廓估計圓柱截面；形狀、相機誤差與遮擋仍保留在結果中。')
