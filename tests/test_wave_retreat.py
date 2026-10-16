"""Offline contracts for a distinct, visually assessed single-pan retreat."""
import unittest
from unittest.mock import Mock

import numpy as np

from server.local_cartesian import ALL, NAMES, default_model
from server.wave_retreat import (ACTIVE, ATTEMPT_LIMITS, LOCAL_CORRIDOR,
                                 check_retreat_attempt, check_retreat_corridor,
                                 propose_retreat, retreat_endpoint)


class WaveRetreatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()

    def setUp(self):
        self.raw = dict(zip(ALL, [2162, 1141, 2727, 2429, 984, 1857]))
        self.held = {**self.raw, 'elbow_flex': self.raw['elbow_flex'] - 32}

    def propose(self, sign=1):
        return propose_retreat(self.model, self.raw, self.held, self.raw, sign)

    def test_both_signs_move_only_pan_and_preserve_every_hold_offset(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                p, envelope = self.propose(sign)
                self.assertEqual(p['motion_kind'], 'pan_retreat')
                self.assertEqual(set(p['active_joint_targets']), {ACTIVE})
                self.assertEqual(p['delta_ticks'][ACTIVE], sign * 10)
                self.assertEqual(envelope[ACTIVE], 30)
                self.assertEqual(envelope['elbow_flex'], 38)
                for name in NAMES:
                    self.assertEqual(p['physical_goal_targets'][name], self.held[name] + p['delta_ticks'][name])
                    if name != ACTIVE:
                        self.assertEqual(p['delta_ticks'][name], 0)
                        self.assertEqual(p['joint_targets'][name], self.raw[name])
                self.assertEqual(p['joint_targets']['gripper'], self.raw['gripper'])
                self.assertTrue(p['clearance']['model_table_pass'])

    def test_camera_side_comes_from_actual_model_transform_not_motor_sign_guess(self):
        sides = set()
        for sign in (-1, 1):
            p, _ = self.propose(sign)
            expected = self.model.fk.camera(self.raw)[:3, :3].T @ np.asarray(p['predicted_delta_m'])
            np.testing.assert_allclose(p['camera_relative_delta_m'], expected)
            self.assertAlmostEqual(np.linalg.norm(expected), np.linalg.norm(p['predicted_delta_m']))
            self.assertGreater(abs(expected[0]), .0005)
            self.assertEqual(p['image_side'], 'left' if expected[0] < 0 else 'right')
            self.assertTrue(p['camera_direction_from_mounting_prior'])
            sides.add(p['image_side'])
        self.assertEqual(sides, {'left', 'right'})

    def test_every_nominal_sample_is_disclosed_and_bounded(self):
        p, _ = self.propose(-1); path = p['retreat_path']
        self.assertEqual(path['sampled_pan_offsets_ticks'], list(range(0, -11, -1)))
        self.assertEqual(path['sampled_delta_m'][0], [0., 0., 0.])
        self.assertEqual(path['sampled_orientation_deg'][0], 0.)
        self.assertEqual(path['sampled_delta_m'][-1], p['predicted_delta_m'])
        self.assertEqual(p['requested_delta_m'], p['predicted_delta_m'])
        self.assertEqual(path['actual_tip_corridor_mm'], LOCAL_CORRIDOR)
        self.assertEqual(path['attempt_limits'], ATTEMPT_LIMITS)
        for offset, delta, angle in zip(path['sampled_pan_offsets_ticks'], path['sampled_delta_m'], path['sampled_orientation_deg']):
            expected = self.model.tip({**self.raw, ACTIVE: self.raw[ACTIVE] + offset})[:3, 3] - self.model.tip(self.raw)[:3, 3]
            np.testing.assert_allclose(delta, expected, atol=1e-15)
            self.assertLessEqual(np.linalg.norm(delta), .0050000001)
            self.assertLessEqual(abs(delta[2]), .0010000001)
            self.assertLessEqual(angle, 1)
        self.assertGreater(np.linalg.norm(p['predicted_delta_m'][:2]), .0005)

    def test_endpoint_requires_measured_progress_in_selected_direction(self):
        for sign in (-1, 1):
            p, _ = self.propose(sign)
            self.assertTrue(retreat_endpoint(self.model, self.raw, p['joint_targets'], p))
            self.assertFalse(retreat_endpoint(self.model, self.raw, self.raw, p))
            self.assertFalse(retreat_endpoint(self.model, self.raw, {**self.raw, ACTIVE: self.raw[ACTIVE] - sign * 10}, p))
            self.assertFalse(retreat_endpoint(self.model, self.raw, {**self.raw, ACTIVE: self.raw[ACTIVE] + sign}, p))
            self.assertFalse(retreat_endpoint(self.model, self.raw, {**self.raw, ACTIVE: self.raw[ACTIVE] + sign * 23}, p))

    def test_other_axes_and_jaw_must_stay_within_four_ticks(self):
        p, _ = self.propose()
        for name in ALL:
            if name == ACTIVE: continue
            with self.subTest(name=name):
                actual = {**p['joint_targets'], name: self.raw[name] + 5}
                with self.assertRaises(RuntimeError): check_retreat_attempt(self.model, self.raw, actual)
                with self.assertRaises(RuntimeError): check_retreat_corridor(self.model, self.raw, actual)
                self.assertFalse(retreat_endpoint(self.model, self.raw, actual, p))

    def test_separate_whole_phase_is_bounded_without_reusing_raise_orientation(self):
        actual = {**self.raw, ACTIVE: self.raw[ACTIVE] + 90}
        distance = check_retreat_attempt(self.model, self.raw, actual)
        self.assertGreater(distance, 12)
        self.assertLess(distance, 40)
        with self.assertRaises(RuntimeError): check_retreat_corridor(self.model, self.raw, actual)
        with self.assertRaises(RuntimeError):
            check_retreat_attempt(self.model, self.raw, {**self.raw, ACTIVE: self.raw[ACTIVE] + 150})

    def test_sign_pose_and_hold_budget_are_strict(self):
        for sign in (0, 2, True, -1., 'left'):
            with self.subTest(sign=sign), self.assertRaises(ValueError): self.propose(sign)
        for pose in ({ACTIVE: 2162}, {**self.raw, ACTIVE: True}, {**self.raw, ACTIVE: float('nan')}):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                propose_retreat(self.model, pose, self.held, self.raw, 1)
        with self.assertRaises(ValueError):
            propose_retreat(self.model, self.raw, {**self.held, ACTIVE: self.raw[ACTIVE] - 55}, self.raw, 1)

    def test_failed_current_table_guard_does_not_try_a_candidate(self):
        model = Mock(wraps=self.model)
        model.screen_segment.return_value = {'model_table_pass': False}
        with self.assertRaisesRegex(ValueError, 'Current pose'):
            propose_retreat(model, self.raw, self.held, self.raw, 1)
        model.screen_segment.assert_called_once()

    def test_failed_path_table_guard_cannot_yield_a_proposal(self):
        model = Mock(wraps=self.model); model.joint_limits = self.model.joint_limits
        model.screen_segment.side_effect = lambda a, b, **kw: {'model_table_pass': a == b}
        with self.assertRaises(RuntimeError): propose_retreat(model, self.raw, self.held, self.raw, 1)

    def test_new_origin_cannot_hide_other_joint_drift(self):
        actual = {**self.raw, 'elbow_flex': self.raw['elbow_flex'] + 5}
        with self.assertRaises(RuntimeError): propose_retreat(self.model, actual, actual, self.raw, 1)

    def test_endpoint_rejects_a_foreign_or_multi_joint_proposal(self):
        p, _ = self.propose()
        for bad in ({**p, 'motion_kind': 'elbow_raise'},
                    {**p, 'joint_targets': {**p['joint_targets'], 'elbow_flex': self.raw['elbow_flex'] + 1}},
                    {**p, 'joint_targets': {**p['joint_targets'], ACTIVE: self.raw[ACTIVE] + 11}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                retreat_endpoint(self.model, self.raw, self.raw, bad)


if __name__ == '__main__':
    unittest.main()
