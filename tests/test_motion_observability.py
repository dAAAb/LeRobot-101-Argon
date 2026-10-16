"""Rank diagnostics only; full rank must never imply calibrated motion."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from server.motion_observability import analyze_motion_observability


def pose(rotvec=(0., 0., 0.), translation=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    result[:3, 3] = translation
    return result


class MotionObservabilityTests(unittest.TestCase):
    def test_single_axis_leaves_exact_translation_null_direction(self):
        axis = np.array([1., 2., -3.]); axis /= np.linalg.norm(axis)
        poses = [pose(axis * angle, [angle, 2 * angle, -.2 * angle])
                 for angle in np.linspace(0, .2, 12)]
        result = analyze_motion_observability(poses)
        self.assertEqual(result['translation_constraint_rank'], 2)
        self.assertEqual(result['rotation_vector_rank'], 1)
        self.assertTrue(result['single_rotation_axis'])
        self.assertFalse(result['translation_identifiable_from_rotations'])
        self.assertEqual(result['unobservable_translation_dimension'], 1)
        null = result['unobservable_translation_directions_gripper'][0]
        self.assertAlmostEqual(abs(np.dot(null, axis)), 1.)
        self.assertFalse(result['calibration_validated'])

    def test_nonparallel_rotations_supply_only_necessary_full_rank(self):
        result = analyze_motion_observability([pose(), pose([.1, 0, 0]), pose([0, .1, 0])])
        self.assertEqual(result['translation_constraint_rank'], 3)
        self.assertEqual(result['unobservable_translation_directions_gripper'], [])
        self.assertTrue(result['translation_identifiable_from_rotations'])
        self.assertFalse(result['single_rotation_axis'])
        self.assertFalse(result['calibration_validated'])
        self.assertEqual(result['status'], 'necessary_rotation_rank_satisfied')

    def test_pure_translation_does_not_constrain_handeye_translation(self):
        poses = [pose([.4, -.2, .7], [x, x * x, -.3 * x]) for x in np.linspace(0, .4, 6)]
        result = analyze_motion_observability(poses)
        self.assertEqual(result['translation_constraint_rank'], 0)
        self.assertEqual(result['rotation_vector_rank'], 0)
        self.assertEqual(result['unobservable_translation_dimension'], 3)
        np.testing.assert_allclose(np.array(result['unobservable_translation_directions_gripper']) @
                                   np.array(result['unobservable_translation_directions_gripper']).T,
                                   np.eye(3), atol=1e-12)

    def test_repeated_poses_and_large_baseline_cannot_fix_single_axis(self):
        poses = [pose([0, 0, .1]), pose([0, 0, .2], [10, -20, 5])]
        first = analyze_motion_observability(poses)
        repeated = analyze_motion_observability(poses * 5)
        self.assertEqual(first['translation_constraint_rank'], 2)
        self.assertEqual(repeated['translation_constraint_rank'], 2)
        self.assertEqual(repeated['relative_pair_count'], 45)

    def test_small_nonparallel_rotations_have_no_arbitrary_angle_gate(self):
        result = analyze_motion_observability([pose(), pose([1e-4, 0, 0]), pose([0, 1e-4, 0])])
        self.assertEqual(result['translation_constraint_rank'], 3)
        self.assertFalse(result['calibration_validated'])

    def test_directions_are_in_gripper_coordinates_under_world_change(self):
        local = [pose([0, 0, a], [a, 0, 0]) for a in (0., .1, .2)]
        world = pose([.2, -.3, .6], [1, 2, 3])
        result = analyze_motion_observability([world @ p for p in local])
        np.testing.assert_allclose(result['unobservable_translation_directions_gripper'][0], [0, 0, 1], atol=1e-12)

    def test_invalid_or_nonrigid_inputs_are_rejected(self):
        scaled = np.eye(4); scaled[0, 0] = 2
        mirror = np.eye(4); mirror[0, 0] = -1
        wrong_bottom = np.eye(4); wrong_bottom[3, 0] = .1
        for value in ([], [np.eye(4)], np.zeros((2, 3, 3)),
                      [np.eye(4), np.full((4, 4), np.nan)],
                      [np.eye(4), scaled], [np.eye(4), mirror], [np.eye(4), wrong_bottom]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                analyze_motion_observability(value)


if __name__ == '__main__':
    unittest.main()
