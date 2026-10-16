"""Pure target-metric sensitivity check; no capture, model, or robot calls.

Usage after retaining X/K from each successful leave-one-pose-out fit:
    result = target_fold_stability(
        Gs, X, K, target_matches,
        [{'pose': pose, 'passed': error <= 3, 'X': xf, 'K': kf}, ...],
        reconstruct_pair=reconstruct_pair,
    )

`target_matches` contains (i,j,p1,p2), selected by the target's observed pixel
region BEFORE any 3D validity filtering. Coordinates are original Nx2 pixels;
select identical row indices in p1 and p2. No dimensions, shape priors,
silhouettes, visual-width assumptions, or external-camera inputs are accepted.

The same pair/row is reconstructed with the full (X,K) and each fold's complete
(xf,kf). Only their shared reconstruct_pair validity mask contributes. The
existing function supplies positive-depth, <2px reprojection and >2deg parallax
checks; this helper neither replaces nor relaxes them. Quantiles include every
shared valid row, with no metric trimming. Repeated observations of a feature in
different pairs remain different pair reconstructions, reported transparently.

Three DISTINCT passed folds, >=5 shared target rows per fold, and worst fold
p95 <=8mm are required. Five rows matches the current object's per-pair support
minimum; callers may choose a higher evidence minimum. A failed/invalid fold or
cancellation always leaves the result uncertain. Focal/rotation spreads are
reported, but the gate uses their resulting target-position sensitivity.
"""
from __future__ import annotations
import math
import numpy as np


def _matrix(value, shape, label):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{label} must be a finite {shape} matrix')
    if shape == (4, 4):
        if (not np.allclose(a[3], [0, 0, 0, 1], atol=1e-7) or
                not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(a[:3, :3]), 1, atol=1e-5)):
            raise ValueError(f'{label} must be a rigid transform')
    elif a[0, 0] <= 0 or a[1, 1] <= 0 or not np.allclose(a[2], [0, 0, 1], atol=1e-7):
        raise ValueError(f'{label} must be a positive-focal camera matrix')
    return a


def _spread(values):
    if not values:
        return {'count': 0, 'min': None, 'median': None, 'max': None, 'range': None}
    v = np.asarray(values, dtype=float)
    return {'count': len(v), 'min': float(v.min()), 'median': float(np.median(v)),
            'max': float(v.max()), 'range': float(np.ptp(v))}


def target_fold_stability(Gs, X, K, target_matches, folds, *,
                          reconstruct_pair, cancel=None,
                          min_validated_folds=3, min_shared_points=5,
                          max_p95_difference_m=.008):
    """Return JSON-safe diagnostics; this never grants grasp readiness.

    `passed` must explicitly be True: the caller's independent fit, observability,
    and held-out epipolar gates must already have succeeded. Each fold needs a
    unique integer `pose` identifier plus its actual `X` and `K` matrices.
    """
    result = {'status': 'uncertain', 'needs_more_views': True, 'grasp_ready': False,
              'method': 'same-target-pixel-pairs-under-full-fold-X-and-K',
              'coordinate_system': 'robot-base metres', 'folds': [],
              'validated_folds': 0, 'worst_fold_p95_difference_m': None,
              'thresholds': {'min_validated_folds': min_validated_folds,
                             'min_shared_points': min_shared_points,
                             'max_p95_difference_m': max_p95_difference_m}}
    def stopped():
        return cancel is not None and cancel.is_set()
    try:
        if (isinstance(min_validated_folds, bool) or min_validated_folds < 3 or
                isinstance(min_shared_points, bool) or min_shared_points < 5 or
                not math.isfinite(max_p95_difference_m) or
                not 0 < max_p95_difference_m <= .008):
            raise ValueError('Stability thresholds may not relax required evidence limits')
        X = _matrix(X, (4, 4), 'X'); K = _matrix(K, (3, 3), 'K')
        if not target_matches:
            raise ValueError('No target pixel correspondences')
        reference = []
        seen_pairs = set()
        for i, j, p1, p2 in target_matches:
            if stopped():
                raise InterruptedError('Cancelled before fold stability completed')
            if (isinstance(i, bool) or isinstance(j, bool) or
                    not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer)) or
                    not 0 <= i < len(Gs) or not 0 <= j < len(Gs) or i == j):
                raise ValueError('Target pairs require two distinct measured pose indices')
            if tuple(sorted((i, j))) in seen_pairs:
                raise ValueError('Duplicate target pair would double-count metric evidence')
            seen_pairs.add(tuple(sorted((i, j))))
            p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
            if (p1.ndim != 2 or p1.shape[1:] != (2,) or p1.shape != p2.shape or
                    len(p1) == 0 or not np.isfinite(p1).all() or not np.isfinite(p2).all()):
                raise ValueError('Target pixel rows must be finite matching nonempty Nx2 arrays')
            with np.errstate(all='ignore'):
                points, valid, _ = reconstruct_pair(Gs, X, K, i, j, p1, p2)
            points = np.asarray(points, dtype=float); valid = np.asarray(valid, dtype=bool)
            if points.shape != (len(p1), 3) or valid.shape != (len(p1),):
                raise ValueError('Unexpected reconstruction shape')
            valid = valid & np.isfinite(points).all(axis=1)
            reference.append((int(i), int(j), p1, p2, points, valid))
        result['target_pairs'] = len(reference)
        result['reference_valid_rows'] = sum(int(r[5].sum()) for r in reference)
        result['reference_focal_px'] = [float(K[0, 0]), float(K[1, 1])]
        focal_x, focal_y, rotations, translations, p95s, seen_poses = [], [], [], [], [], set()
        failed_fold = False
        for fold in folds:
            if stopped():
                raise InterruptedError('Cancelled before fold stability completed')
            pose = fold.get('pose')
            row = {'pose': int(pose) if isinstance(pose, (int, np.integer)) else None,
                   'validated': False, 'shared_valid_rows': 0, 'pairs': []}
            result['folds'].append(row)
            if (isinstance(pose, bool) or not isinstance(pose, (int, np.integer)) or
                    not 0 <= pose < len(Gs) or pose in seen_poses):
                row['reason'] = 'Missing, duplicate, or invalid held-out pose'; failed_fold = True; continue
            seen_poses.add(pose)
            if fold.get('passed') is not True:
                row['reason'] = 'Fold did not pass the caller’s fit and held-out gates'; failed_fold = True; continue
            try:
                xf = _matrix(fold['X'], (4, 4), 'fold X'); kf = _matrix(fold['K'], (3, 3), 'fold K')
            except (KeyError, ValueError, TypeError) as exc:
                row['reason'] = str(exc); failed_fold = True; continue
            angle = float(np.rad2deg(np.arccos(np.clip((np.trace(xf[:3, :3] @ X[:3, :3].T)-1)/2, -1, 1))))
            shift = float(np.linalg.norm(xf[:3, 3] - X[:3, 3])*1000)
            focal_x.append(float(kf[0, 0])); focal_y.append(float(kf[1, 1]))
            rotations.append(angle); translations.append(shift)
            row.update(focal_px=[float(kf[0, 0]), float(kf[1, 1])],
                       focal_change_percent=[float((kf[d, d]/K[d, d]-1)*100) for d in (0, 1)],
                       rotation_change_deg=angle, translation_change_mm=shift)
            distances = []
            for i, j, p1, p2, ref_points, ref_valid in reference:
                if stopped():
                    raise InterruptedError('Cancelled before fold stability completed')
                with np.errstate(all='ignore'):
                    points, valid, _ = reconstruct_pair(Gs, xf, kf, i, j, p1, p2)
                points = np.asarray(points, dtype=float); valid = np.asarray(valid, dtype=bool)
                if points.shape != ref_points.shape or valid.shape != ref_valid.shape:
                    raise ValueError('Fold reconstruction shape does not match reference')
                shared = ref_valid & valid & np.isfinite(points).all(axis=1)
                indices = np.flatnonzero(shared)
                delta = np.linalg.norm(points[indices] - ref_points[indices], axis=1)
                distances.extend(delta.tolist())
                row['pairs'].append({'i': i, 'j': j, 'input_rows': len(p1),
                                     'reference_valid_rows': int(ref_valid.sum()),
                                     'fold_valid_rows': int(valid.sum()),
                                     'shared_valid_rows': len(indices),
                                     'shared_indices': indices.tolist(),
                                     'p95_difference_m': float(np.percentile(delta, 95)) if len(delta) else None})
            row['shared_valid_rows'] = len(distances)
            row['shared_reference_fraction'] = len(distances)/max(result['reference_valid_rows'], 1)
            if len(distances) < min_shared_points:
                row['reason'] = 'Insufficient shared valid target rows'; failed_fold = True; continue
            p95 = float(np.percentile(distances, 95))
            row.update(validated=True, p95_difference_m=p95,
                       median_difference_m=float(np.median(distances)),
                       max_difference_m=float(np.max(distances)),
                       metric_stable=p95 <= max_p95_difference_m)
            p95s.append(p95)
        result['validated_folds'] = len(p95s)
        result['focal_x_px'] = _spread(focal_x); result['focal_y_px'] = _spread(focal_y)
        result['rotation_change_deg'] = _spread(rotations)
        result['translation_change_mm'] = _spread(translations)
        result['worst_fold_p95_difference_m'] = max(p95s) if p95s else None
        stable = (not failed_fold and len(p95s) >= min_validated_folds and
                  max(p95s) <= max_p95_difference_m)
        result.update(status='repeatable' if stable else 'uncertain', needs_more_views=not stable,
                      reason='Target reconstructions agree across held-out fits' if stable else
                      'Target metric stability lacks enough valid folds or exceeds tolerance')
    except InterruptedError as exc:
        result.update(cancelled=True, reason=str(exc), status='uncertain', needs_more_views=True)
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        result.update(reason=str(exc), status='uncertain', needs_more_views=True)
    return result
