"""Numerical motion observability diagnostics, with no cameras or motor I/O.

For base_T_gripper poses G, the hand-eye equation is A X = X B, where
A = inv(G_j) G_i. Its translation equation contains (R_A - I) t_X.
Stacking those blocks exposes translation directions that motion cannot
constrain. Full column rank is necessary, never sufficient, for calibration.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.transform import Rotation


def _rank_and_svd(matrix):
    _, values, vt = np.linalg.svd(matrix, full_matrices=False)
    # The unit lower scale handles round-off in R.T @ R for zero rotation.
    # This is floating-point numerical rank, not a physical angle threshold.
    tolerance = max(matrix.shape) * np.finfo(float).eps * max(1., float(values[0]))
    return int(np.count_nonzero(values > tolerance)), values, vt, tolerance


def _canonical_direction(value):
    direction = np.asarray(value, dtype=float)
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    return direction.tolist()


def analyze_motion_observability(gripper_poses):
    """Diagnose hand-eye translation nullspace from rigid base_T_gripper poses.

    Reported directions use the gripper coordinate convention of X, not the
    robot-base frame. Translations/baseline and feature quality cannot remove
    the nullspace of this rotational block. Intrinsics, image correspondence,
    distortion, FK accuracy, conditioning and withheld-view validations remain
    independent requirements even when the block has full numerical rank.
    """
    poses = np.asarray(gripper_poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError('At least two finite 4x4 gripper transforms are required')
    for pose in poses:
        R = pose[:3, :3]
        if (not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-10, rtol=0)
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-10, rtol=0)
                or not np.isclose(np.linalg.det(R), 1., atol=1e-10, rtol=0)):
            raise ValueError('Expected rigid base_T_gripper transforms')
    relative_rotations = [poses[j, :3, :3].T @ poses[i, :3, :3]
                          for i, j in itertools.combinations(range(len(poses)), 2)]
    blocks = np.concatenate([R - np.eye(3) for R in relative_rotations], axis=0)
    vectors = np.array([Rotation.from_matrix(R).as_rotvec() for R in relative_rotations])
    rank, singular, vt, tolerance = _rank_and_svd(blocks)
    rotation_rank, rotation_singular, _, rotation_tolerance = _rank_and_svd(vectors)
    nullspace = [_canonical_direction(axis) for axis in vt[rank:]]
    return {
        'method': 'stacked relative-rotation hand-eye translation constraints',
        'pose_count': len(poses), 'relative_pair_count': len(relative_rotations),
        'translation_constraint_rank': rank,
        'translation_constraint_singular_values': singular.tolist(),
        'translation_rank_tolerance': tolerance,
        'unobservable_translation_dimension': 3 - rank,
        'unobservable_translation_directions_gripper': nullspace,
        'translation_identifiable_from_rotations': rank == 3,
        'rotation_vector_rank': rotation_rank,
        'rotation_vector_singular_values_rad': rotation_singular.tolist(),
        'rotation_rank_tolerance': rotation_tolerance,
        'single_rotation_axis': rotation_rank == 1,
        'status': 'necessary_rotation_rank_satisfied' if rank == 3 else 'unobservable_translation',
        'calibration_validated': False,
        'note': ('Full numerical rank is necessary, not sufficient. Image tracks, intrinsics, '
                 'distortion, robot geometry, conditioning and independent withheld views '
                 'still require validation.'),
    }
