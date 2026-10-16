"""Progress acceptance is useful step completion, never grasp/task success."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from server.local_cartesian import ALL
from server.motion_policy import (SettledProgressWindow, classify_translation_progress,
                                  minimum_visual_confidence)


def transform(x=0., y=0., z=0., angle=0.):
    T = np.eye(4); T[:3, 3] = np.array([x, y, z]) / 1000
    T[:3, :3] = Rotation.from_euler('y', angle, degrees=True).as_matrix()
    return T


class MotionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.origin = dict.fromkeys(ALL, 2047)
        self.actual = {**self.origin, 'shoulder_lift': 2057}

    def classify(self, x=1.5, y=0., z=0., angle=0., *, empty=True, corridor=True,
                 joints=None, requested=3.):
        return classify_translation_progress(
            transform(), transform(x, y, z, angle), [requested / 1000, 0., 0.],
            target_tip=transform(requested), origin_joints=self.origin,
            actual_joints=self.actual if joints is None else joints,
            empty_hand=empty, corridor_pass=corridor)

    def test_half_of_three_mm_is_useful_partial_not_exact_or_grasp_success(self):
        result = self.classify()
        self.assertTrue(result['accepted'])
        self.assertEqual(result['completion'], 'partial_progress')
        self.assertAlmostEqual(result['progress_fraction'], .5)
        self.assertFalse(result['exact_target_reached'])
        self.assertTrue(result['requires_reobservation'])
        self.assertFalse(result['task_success'])
        self.assertFalse(result['grasp_verified'])

    def test_exact_target_remains_a_separate_classification(self):
        result = self.classify(3.)
        self.assertEqual(result['completion'], 'exact_target')
        self.assertTrue(result['exact_target_reached'])
        self.assertFalse(result['grasp_verified'])

    def test_zero_backwards_tiny_or_less_than_half_progress_rejected(self):
        for progress in (0., -.5, -3., .2, 1.499):
            with self.subTest(progress=progress):
                self.assertFalse(self.classify(progress)['accepted'])

    def test_encoder_noise_and_jaw_only_motion_cannot_count_as_progress(self):
        for joints in (self.origin, {**self.origin, 'shoulder_lift': 2049},
                       {**self.origin, 'gripper': 2080}):
            with self.subTest(joints=joints):
                self.assertFalse(self.classify(joints=joints)['accepted'])
        self.assertTrue(self.classify(joints={**self.origin, 'shoulder_lift': 2050})['accepted'])

    def test_held_or_unknown_hand_never_uses_partial_acceptance(self):
        self.assertFalse(self.classify(empty=False)['accepted'])
        # Exact translational completion alone still says nothing about a grasp.
        result = self.classify(3., empty=False)
        self.assertTrue(result['accepted'])
        self.assertFalse(result['grasp_verified'])

    def test_screened_corridor_failure_cannot_be_overridden(self):
        for progress in (1.5, 3.):
            result = self.classify(progress, corridor=False)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['reason'], 'outside_screened_corridor')

    def test_cross_track_and_rotation_tolerance_is_bounded(self):
        self.assertTrue(self.classify(1.5, y=1.5, angle=2.)['accepted'])
        self.assertFalse(self.classify(1.5, y=1.501)['accepted'])
        self.assertFalse(self.classify(1.5, angle=2.001)['accepted'])
        self.assertEqual(self.classify(1., requested=1.)['cross_track_limit_mm'], .5)

    def test_partial_overshoot_must_be_small_and_reduce_target_error(self):
        self.assertTrue(self.classify(4.5)['accepted'])
        self.assertFalse(self.classify(4.501)['accepted'])
        self.assertFalse(self.classify(8.)['accepted'])
        # Within a broad screened corridor is not enough for useful progress.
        self.assertFalse(self.classify(1.5, y=5.)['accepted'])

    def test_three_spaced_settled_samples_are_needed(self):
        result = self.classify(); window = SettledProgressWindow()
        self.assertFalse(window.observe(result, self.actual, 0.))
        self.assertFalse(window.observe(result, self.actual, .01))
        self.assertFalse(window.observe(result, self.actual, .12))
        self.assertTrue(window.observe(result, self.actual, .24))

    def test_failed_sample_even_when_rapid_breaks_streak(self):
        accepted = self.classify(); rejected = self.classify(0.)
        window = SettledProgressWindow()
        window.observe(accepted, self.actual, 0.)
        window.observe(accepted, self.actual, .12)
        self.assertFalse(window.observe(rejected, self.actual, .13))
        self.assertFalse(window.observe(accepted, self.actual, .24))
        self.assertFalse(window.observe(accepted, self.actual, .36))
        self.assertTrue(window.observe(accepted, self.actual, .48))

    def test_more_than_two_tick_drift_restarts_settling(self):
        result = self.classify(); window = SettledProgressWindow()
        window.observe(result, self.actual, 0.)
        window.observe(result, self.actual, .12)
        shifted = {**self.actual, 'shoulder_lift': self.actual['shoulder_lift'] + 3}
        self.assertFalse(window.observe(result, shifted, .24))
        self.assertFalse(window.observe(result, shifted, .36))
        self.assertTrue(window.observe(result, shifted, .48))
        self.assertFalse(window.observe(result, {**shifted, 'gripper': 2050}, .6))

    def test_visual_thresholds_are_action_specific(self):
        self.assertGreaterEqual(.78, minimum_visual_confidence('local_move'))
        self.assertLess(.78, minimum_visual_confidence('wave'))
        self.assertEqual(minimum_visual_confidence('wave'), .8)
        self.assertEqual(minimum_visual_confidence('empty_jaw_open'), .7)
        self.assertEqual(minimum_visual_confidence('close'), .9)
        self.assertEqual(minimum_visual_confidence('grasp_verify'), .9)
        with self.assertRaises(ValueError): minimum_visual_confidence('unknown')

    def test_invalid_inputs_do_not_silently_enable_partial_acceptance(self):
        for empty, corridor in ((None, True), (True, 1), ('yes', True)):
            with self.subTest(empty=empty, corridor=corridor), self.assertRaises(ValueError):
                self.classify(empty=empty, corridor=corridor)
        for requested in (0., .1, 5.1, float('nan')):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                self.classify(requested=requested)
        with self.assertRaises(ValueError): self.classify(joints={'shoulder_lift': 2050})
        window = SettledProgressWindow(); window.observe(self.classify(), self.actual, 1.)
        with self.assertRaises(ValueError): window.observe(self.classify(), self.actual, .9)


if __name__ == '__main__':
    unittest.main()
