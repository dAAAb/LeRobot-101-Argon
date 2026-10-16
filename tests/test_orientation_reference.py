"""Pure IK tests: attempt orientation must not re-anchor accumulated drift."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from server.local_cartesian import ALL, NAMES, default_model


class OrientationReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()

    def setUp(self):
        self.origin = dict(zip(ALL, [2162, 1139, 2902, 2407, 984, 1857]))
        self.raw = {**self.origin, 'elbow_flex': 2894}
        self.envelope = {**dict.fromkeys(NAMES, 30), 'shoulder_pan': 12, 'wrist_roll': 12}

    def test_default_solver_is_unchanged_by_same_pose_reference(self):
        legacy = self.model.local_ik(self.raw, [0., 0., .003])
        explicit = self.model.local_ik(self.raw, [0., 0., .003], orientation_origin=self.raw)
        self.assertEqual(legacy['joint_targets'], explicit['joint_targets'])
        self.assertEqual(legacy['predicted_delta_m'], explicit['predicted_delta_m'])
        self.assertEqual(legacy['full_orientation_error_deg'], explicit['full_orientation_error_deg'])
        self.assertNotIn('orientation_reference_origin', legacy)

    def test_attempt_axis_reduces_drift_without_reanchoring_position(self):
        legacy = self.model.local_ik(self.raw, [0., 0., .003])
        anchored = self.model.local_ik(self.raw, [0., 0., .003], orientation_origin=self.origin)
        desired = self.model.tip(self.origin)[:3, 1]
        def axis_error(proposal):
            axis = self.model.tip(proposal['joint_targets'])[:3, 1]
            return np.degrees(np.arccos(np.clip(axis @ desired, -1., 1.)))
        self.assertLess(axis_error(anchored), axis_error(legacy))
        np.testing.assert_allclose(anchored['held_tool_axis_error_deg'], axis_error(anchored), atol=1e-9)
        before = self.model.tip(self.raw)
        after = self.model.tip(anchored['joint_targets'])
        np.testing.assert_allclose(anchored['predicted_delta_m'], after[:3, 3] - before[:3, 3])
        self.assertLess(anchored['predicted_position_error_mm'], 1.)
        self.assertEqual(anchored['measured_origin'], self.raw)
        self.assertEqual(anchored['orientation_reference_origin'], self.origin)

    def test_local_orientation_error_still_uses_current_measured_pose(self):
        result = self.model.local_ik(self.raw, [0., 0., .003], orientation_origin=self.origin)
        end = self.model.tip(result['joint_targets'])[:3, :3]
        current = self.model.tip(self.raw)[:3, :3]
        origin = self.model.tip(self.origin)[:3, :3]
        self.assertAlmostEqual(result['full_orientation_error_deg'],
                               np.degrees(Rotation.from_matrix(end @ current.T).magnitude()))
        self.assertAlmostEqual(result['orientation_reference_error_deg'],
                               np.degrees(Rotation.from_matrix(end @ origin.T).magnitude()))
        self.assertNotAlmostEqual(result['full_orientation_error_deg'], result['orientation_reference_error_deg'])

    def test_adaptive_proposal_preserves_reference_table_checks_and_held_bias(self):
        held = {**self.raw, 'elbow_flex': self.raw['elbow_flex'] - 8}
        result = self.model.propose_adaptive(self.raw, [0., 0., .005], orientation_origin=self.origin,
                                           held_goals=held, uncertainty_ticks=self.envelope)
        self.assertTrue(result['clearance']['model_table_pass'])
        self.assertEqual(result['orientation_reference_origin'], self.origin)
        self.assertLessEqual(result['full_orientation_error_deg'], 2.)
        self.assertLessEqual(result['predicted_position_error_mm'], 1.)
        self.assertLessEqual(np.linalg.norm(result['predicted_delta_m']), .0050000001)
        for name in NAMES:
            self.assertLessEqual(abs(result['delta_ticks'][name]), 30)
            self.assertEqual(result['physical_goal_targets'][name], held[name] + result['delta_ticks'][name])

    def test_invalid_reference_is_rejected_and_never_becomes_fallback(self):
        bad = [dict(self.origin, wrist_flex=float('nan')), dict(self.origin, wrist_roll=True),
               dict(self.origin, shoulder_pan=-1), {'wrist_flex': self.origin['wrist_flex']}]
        for reference in bad:
            with self.subTest(reference=reference):
                with self.assertRaises(ValueError):
                    self.model.propose_adaptive(self.raw, [0., 0., .003],
                                                orientation_origin=reference, uncertainty_ticks=self.envelope)

    def test_reference_does_not_expand_input_or_joint_limits(self):
        with self.assertRaises(ValueError):
            self.model.local_ik(self.raw, [0., 0., .006], orientation_origin=self.origin)
        with self.assertRaises(ValueError):
            self.model.local_ik(self.raw, [0., 0., .003], orientation_origin=self.origin,
                                max_joint_step_ticks=31)


if __name__ == '__main__':
    unittest.main()
