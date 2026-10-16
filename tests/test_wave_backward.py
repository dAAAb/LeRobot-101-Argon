"""Offline Cartesian backward contracts; these tests never access hardware."""
import copy
import itertools
import unittest
from unittest.mock import Mock, patch

import numpy as np

from server.local_cartesian import ALL, NAMES, default_model
from server.wave_backward import (ATTEMPT_LIMITS, COORDINATED_JOINTS, FIXED_JOINTS,
                                  LOCAL_CORRIDOR, backward_endpoint,
                                  check_backward_attempt, check_backward_corridor,
                                  propose_backward)


class WaveBackwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()
        cls.raw = dict(zip(ALL, [2162, 1141, 2694, 2429, 984, 1857]))
        cls.held = {n: v + 1 for n, v in cls.raw.items()}
        cls.held['gripper'] = cls.raw['gripper']
        cls.held['elbow_flex'] = cls.raw['elbow_flex'] - 41
        cls.proposal, cls.envelope = propose_backward(cls.model, cls.raw, cls.held, cls.raw)

    def test_current_pose_has_exact_backward_ik_and_preserves_hold_offsets(self):
        p = self.proposal
        self.assertEqual(p['motion_kind'], 'cartesian_backward')
        self.assertEqual(p['delta_ticks'], dict(zip(NAMES, [0, -18, -1, 19, 0])))
        self.assertEqual(set(p['active_joint_targets']), set(COORDINATED_JOINTS))
        for name in NAMES:
            self.assertEqual(p['physical_goal_targets'][name], self.held[name] + p['delta_ticks'][name])
            self.assertGreaterEqual(self.envelope[name], 30)
        self.assertEqual(self.envelope['elbow_flex'], 47)
        for name in FIXED_JOINTS:
            self.assertEqual(p['joint_targets'][name], self.raw[name])
        self.assertLessEqual(p['predicted_position_error_mm'], 1)
        self.assertLessEqual(p['full_orientation_error_deg'], 2)
        self.assertLessEqual(p['orientation_reference_error_deg'], 1)
        self.assertEqual(p['orientation_reference_origin'], self.raw)

    def test_camera_retreat_is_model_derived_without_claiming_calibration(self):
        p = self.proposal
        direction = self.model.fk.transforms(self.raw)['gripper_link'][:3, 2]
        np.testing.assert_allclose(p['requested_delta_m'], direction * .003)
        camera = self.model.fk.camera(self.raw)[:3, :3].T @ np.asarray(p['predicted_delta_m'])
        np.testing.assert_allclose(p['camera_relative_delta_m'], camera)
        self.assertLess(camera[2], 0)
        self.assertEqual(p['image_side'], 'backward')
        self.assertTrue(p['camera_direction_from_mounting_prior'])
        self.assertTrue(p['not_environment_collision_certificate'])

    def test_every_asynchronous_corner_is_disclosed_with_sixteen_table_checks(self):
        p = self.proposal; path = p['backward_path']
        expected = [list(v) for v in itertools.product(*[(0, p['delta_ticks'][n]) for n in COORDINATED_JOINTS])]
        self.assertEqual(path['coordinated_joints'], list(COORDINATED_JOINTS))
        self.assertEqual(path['sampled_corner_offsets_ticks'], expected)
        self.assertEqual(path['sampled_corner_delta_m'][0], [0., 0., 0.])
        self.assertEqual(path['sampled_corner_orientation_deg'][0], 0.)
        self.assertEqual(path['sampled_corner_delta_m'][-1], p['predicted_delta_m'])
        self.assertEqual(path['actual_tip_corridor_mm'], LOCAL_CORRIDOR)
        self.assertEqual(path['attempt_limits'], ATTEMPT_LIMITS)
        self.assertEqual(len(p['asynchronous_clearance']), 16)
        self.assertTrue(all(c['model_table_pass'] for c in p['asynchronous_clearance']))
        for offset, delta, angle in zip(expected, path['sampled_corner_delta_m'], path['sampled_corner_orientation_deg']):
            corner = {**self.raw, **{n: self.raw[n] + v for n, v in zip(COORDINATED_JOINTS, offset)}}
            np.testing.assert_allclose(delta, self.model.tip(corner)[:3, 3] - self.model.tip(self.raw)[:3, 3])
            check_backward_corridor(self.model, self.raw, corner)
            self.assertLessEqual(angle, 3)
        # A wrist-first move dips over 4 mm despite a slightly rising final target.
        # The planner sees this curved/asynchronous corridor explicitly.
        self.assertLess(path['sampled_corner_delta_m'][1][2], -.004)

    def test_endpoint_keeps_one_mm_accuracy_not_twelve_mm_arc_tolerance(self):
        p = self.proposal; goal = p['joint_targets']
        self.assertTrue(backward_endpoint(self.model, self.raw, goal, p, attempt_origin=self.raw))
        self.assertFalse(backward_endpoint(self.model, self.raw, self.raw, p))
        off_goal = {**goal, 'wrist_flex': goal['wrist_flex'] + 8}
        check_backward_corridor(self.model, self.raw, off_goal)
        self.assertGreater(np.linalg.norm(self.model.tip(off_goal)[:3, 3] - self.model.tip(goal)[:3, 3]), .001)
        self.assertFalse(backward_endpoint(self.model, self.raw, off_goal, p))
        reverse = {**self.raw, **{n: self.raw[n] - p['delta_ticks'][n] for n in NAMES}}
        self.assertFalse(backward_endpoint(self.model, self.raw, reverse, p))

    def test_fixed_axes_and_jaw_drift_is_rejected_in_both_guards(self):
        for name in FIXED_JOINTS:
            actual = {**self.proposal['joint_targets'], name: self.raw[name] + 5}
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError): check_backward_attempt(self.model, self.raw, actual)
                with self.assertRaises(RuntimeError): check_backward_corridor(self.model, self.raw, actual)
                self.assertFalse(backward_endpoint(self.model, self.raw, actual, self.proposal))

    def test_phase_guard_does_not_rebase_or_expand_local_corridor(self):
        p = self.proposal
        farther = {**self.raw, **{n: self.raw[n] + 6 * p['delta_ticks'][n] for n in NAMES}}
        progress = check_backward_attempt(self.model, self.raw, farther)
        self.assertGreater(progress, 12)
        self.assertLess(progress, 40)
        with self.assertRaises(RuntimeError): check_backward_corridor(self.model, self.raw, farther)
        excessive = {**self.raw, **{n: self.raw[n] + 15 * p['delta_ticks'][n] for n in NAMES}}
        with self.assertRaises(RuntimeError): check_backward_attempt(self.model, self.raw, excessive)
        turned = {**self.raw, 'wrist_flex': self.raw['wrist_flex'] + 35}
        with self.assertRaises(RuntimeError): check_backward_attempt(self.model, self.raw, turned)

    def test_pose_and_hold_inputs_are_strict(self):
        for raw in ({'elbow_flex': 2694}, {**self.raw, 'elbow_flex': True},
                    {**self.raw, 'elbow_flex': float('nan')}, {**self.raw, 'elbow_flex': 2694.5}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                propose_backward(self.model, raw, self.held, self.raw)
        with self.assertRaises(ValueError):
            propose_backward(self.model, self.raw, {**self.held, 'elbow_flex': self.raw['elbow_flex'] - 55}, self.raw)

    def test_failed_corner_table_check_cannot_reach_visual_assessment(self):
        model = Mock(wraps=self.model)
        # The adaptive result was screened already; now fail an asynchronous leg.
        model.propose_adaptive.return_value = copy.deepcopy(self.proposal)
        model.screen_segment.return_value = {'model_table_pass': False}
        with self.assertRaisesRegex(ValueError, 'Asynchronous'):
            propose_backward(model, self.raw, self.held, self.raw)
        model.screen_segment.assert_called_once()

    def test_original_invalid_table_pose_is_not_repaired_by_shortening(self):
        with patch.object(self.model, 'screen_segment', return_value={'model_table_pass': False}) as screen:
            with self.assertRaisesRegex(ValueError, 'Current pose'):
                propose_backward(self.model, self.raw, self.held, self.raw)
            screen.assert_called_once()

    def test_fixed_axis_ik_or_attempt_orientation_error_cannot_yield_proposal(self):
        for change in ('pan', 'orientation'):
            p = copy.deepcopy(self.proposal)
            if change == 'pan': p['joint_targets']['shoulder_pan'] += 1
            else: p['joint_targets']['wrist_flex'] += 15
            model = Mock(wraps=self.model); model.propose_adaptive.return_value = p
            with self.subTest(change=change), self.assertRaises(ValueError):
                propose_backward(model, self.raw, self.held, self.raw)

    def test_endpoint_rejects_foreign_or_forward_proposals(self):
        p = self.proposal
        for bad in ({**p, 'motion_kind': 'pan_retreat'},
                    {**p, 'requested_delta_m': (-np.asarray(p['requested_delta_m'])).tolist()},
                    {**p, 'joint_targets': {**p['joint_targets'], 'gripper': self.raw['gripper'] + 1}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                backward_endpoint(self.model, self.raw, self.raw, bad)


if __name__ == '__main__':
    unittest.main()
