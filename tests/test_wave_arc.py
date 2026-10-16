"""Offline geometry/contracts for the independently assessed elbow arc."""
from copy import deepcopy
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from server.local_cartesian import ALL, NAMES, default_model
from server.wave_arc import (ACTIVE, ATTEMPT_LIMITS, LOCAL_CORRIDOR, NOMINAL_LIMITS, arc_endpoint,
                             check_arc_attempt, check_arc_corridor, propose_raise)


class WaveArcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()

    def setUp(self):
        self.raw = dict(zip(ALL, [2162, 1141, 2839, 2429, 984, 1857]))
        self.held = {**self.raw, ACTIVE: self.raw[ACTIVE] - 32}

    def propose(self, **kwargs):
        return propose_raise(self.model, self.raw, self.held, self.raw, **kwargs)

    def test_real_pose_uses_only_elbow_and_retains_existing_hold_compensation(self):
        proposal, envelope = self.propose()
        self.assertEqual(proposal['motion_kind'], 'elbow_raise')
        self.assertEqual(set(proposal['active_joint_targets']), {ACTIVE})
        ticks = proposal['delta_ticks'][ACTIVE]
        self.assertTrue(1 <= abs(ticks) <= 20)
        self.assertEqual(envelope[ACTIVE], 38)
        for name in NAMES:
            self.assertEqual(proposal['physical_goal_targets'][name], self.held[name] + proposal['delta_ticks'][name])
            if name != ACTIVE:
                self.assertEqual(envelope[name], 12)
                self.assertEqual(proposal['delta_ticks'][name], 0)
                self.assertEqual(proposal['joint_targets'][name], self.raw[name])
        self.assertEqual(proposal['joint_targets']['gripper'], self.raw['gripper'])
        self.assertTrue(proposal['clearance']['model_table_pass'])

    def test_nominal_arc_samples_disclose_actual_curve_not_a_pure_vertical_target(self):
        p, _ = self.propose()
        path = p['arc_path']; ticks = p['delta_ticks'][ACTIVE]
        sign = 1 if ticks > 0 else -1
        self.assertEqual(path['sampled_elbow_offsets_ticks'], list(range(0, ticks + sign, sign)))
        self.assertEqual(len(path['sampled_delta_m']), abs(ticks) + 1)
        self.assertEqual(path['sampled_delta_m'][0], [0., 0., 0.])
        self.assertEqual(path['sampled_orientation_deg'][0], 0.)
        self.assertEqual(path['sampled_delta_m'][-1], p['predicted_delta_m'])
        self.assertEqual(p['requested_delta_m'], p['predicted_delta_m'])
        self.assertGreater(np.linalg.norm(p['predicted_delta_m'][:2]), 0)
        self.assertEqual(path['actual_tip_corridor_mm'], LOCAL_CORRIDOR)
        self.assertEqual(path['attempt_limits'], ATTEMPT_LIMITS)
        for offset, delta, angle in zip(path['sampled_elbow_offsets_ticks'], path['sampled_delta_m'], path['sampled_orientation_deg']):
            expected = self.model.tip({**self.raw, ACTIVE: self.raw[ACTIVE] + offset})[:3, 3] - self.model.tip(self.raw)[:3, 3]
            np.testing.assert_allclose(delta, expected, atol=1e-15)
            self.assertLessEqual(np.linalg.norm(delta), .0080000001)
            self.assertLessEqual(np.linalg.norm(delta[:2]), .0020000001)
            self.assertGreaterEqual(delta[2], 0)
            self.assertLessEqual(angle, 2)

    def test_larger_nominal_arc_at_recorded_pose_keeps_actual_and_command_bounds(self):
        # Offline recorded-pose fixture: a larger desired arc is a controlled
        # comparison, not evidence that the real motor will overcome stiction.
        raw = dict(zip(ALL, [2161, 1138, 2654, 2332, 984, 1858]))
        held = {**raw, ACTIVE: 2647}
        p, envelope = propose_raise(self.model, raw, held, raw)
        self.assertEqual(NOMINAL_LIMITS, {'maximum_distance_mm': 8., 'maximum_lateral_mm': 2.,
                                        'maximum_rotation_degrees': 2., 'maximum_ticks': 20})
        self.assertEqual(p['delta_ticks'][ACTIVE], -17)
        self.assertEqual(p['joint_targets'][ACTIVE], 2637)
        self.assertEqual(p['physical_goal_targets'][ACTIVE], 2630)
        self.assertAlmostEqual(np.linalg.norm(p['predicted_delta_m']) * 1000, 7.461643, places=5)
        self.assertAlmostEqual(np.linalg.norm(p['predicted_delta_m'][:2]) * 1000, 1.912429, places=5)
        self.assertGreater(p['full_orientation_error_deg'], 1.)
        self.assertEqual(envelope, {n: 30 if n == ACTIVE else 12 for n in NAMES})
        self.assertTrue(p['clearance']['model_table_pass'])
        self.assertEqual(p['arc_path']['actual_tip_corridor_mm'], LOCAL_CORRIDOR)
        self.assertEqual(p['arc_path']['attempt_limits'], ATTEMPT_LIMITS)
        check_arc_corridor(self.model, raw, p['joint_targets'])
        self.assertTrue(arc_endpoint(self.model, raw, p['joint_targets'], p))
        for n in ALL:
            if n != ACTIVE:
                self.assertEqual(p['joint_targets'][n], raw[n])

    def test_requested_distance_cap_is_respected(self):
        p, _ = self.propose(max_distance_mm=3.)
        self.assertLessEqual(np.linalg.norm(p['predicted_delta_m']) * 1000, 3.0000001)
        self.assertGreater(p['predicted_delta_m'][2] * 1000, .5)
        for bad in (0, -1, 8.1, True, float('nan')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.propose(max_distance_mm=bad)

    def test_endpoint_requires_actual_correct_direction_and_sufficient_progress(self):
        p, _ = self.propose()
        self.assertTrue(arc_endpoint(self.model, self.raw, p['joint_targets'], p))
        self.assertFalse(arc_endpoint(self.model, self.raw, self.raw, p))
        ticks = p['delta_ticks'][ACTIVE]
        wrong = {**self.raw, ACTIVE: self.raw[ACTIVE] - ticks}
        tiny = {**self.raw, ACTIVE: self.raw[ACTIVE] + (1 if ticks > 0 else -1)}
        self.assertFalse(arc_endpoint(self.model, self.raw, wrong, p))
        self.assertFalse(arc_endpoint(self.model, self.raw, tiny, p))

    def test_nonmoving_body_and_jaw_drift_is_not_accepted(self):
        p, _ = self.propose()
        for name in ALL:
            if name == ACTIVE: continue
            with self.subTest(name=name):
                actual = {**p['joint_targets'], name: self.raw[name] + 5}
                with self.assertRaises(RuntimeError): check_arc_corridor(self.model, self.raw, actual)
                with self.assertRaises(RuntimeError): check_arc_attempt(self.model, self.raw, actual)
                self.assertFalse(arc_endpoint(self.model, self.raw, actual, p))

    def test_local_and_whole_attempt_limits_are_distinct(self):
        # A planned 30-tick elbow rotation is over this step's nominal command
        # cap, but can be part of a separately reassessed cumulative arc.
        actual = {**self.raw, ACTIVE: self.raw[ACTIVE] - 30}
        self.assertGreater(check_arc_attempt(self.model, self.raw, actual), 0)
        with self.assertRaises(RuntimeError): check_arc_corridor(self.model, self.raw, actual)
        with self.assertRaises(RuntimeError):
            check_arc_attempt(self.model, self.raw, {**self.raw, ACTIVE: self.raw[ACTIVE] - 200})

    def test_recorded_failure_pose_fits_new_whole_arc_without_larger_local_corridor(self):
        origin = dict(zip(ALL, [2161, 1140, 2688, 2332, 984, 1858]))
        before = dict(zip(ALL, [2161, 1137, 2645, 2332, 984, 1858]))
        actual = dict(zip(ALL, [2161, 1137, 2641, 2332, 984, 1858]))
        delta = (self.model.tip(actual)[:3, 3] - self.model.tip(origin)[:3, 3]) * 1000
        self.assertAlmostEqual(np.linalg.norm(delta[:2]), 5.204893, places=5)
        self.assertAlmostEqual(check_arc_attempt(self.model, origin, actual), 20.879829, places=5)
        check_arc_corridor(self.model, before, actual)
        self.assertEqual(ATTEMPT_LIMITS['maximum_lateral_mm'], 15.)
        self.assertEqual(LOCAL_CORRIDOR['maximum_lateral'], 3.)

    def test_real_arc_over_fifteen_mm_lateral_is_rejected_below_other_limits(self):
        origin = dict(zip(ALL, [2161, 1140, 2688, 2332, 984, 1858]))
        actual = {**origin, ACTIVE: 2560}
        a = self.model.tip(origin); b = self.model.tip(actual)
        delta = (b[:3, 3] - a[:3, 3]) * 1000
        self.assertGreater(np.linalg.norm(delta[:2]), 15.)
        self.assertTrue(0 < delta[2] < 60.)
        self.assertLess(np.degrees(Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).magnitude()), 15.)
        with self.assertRaises(RuntimeError): check_arc_attempt(self.model, origin, actual)

    def test_cannot_shrink_a_nominal_step_to_evade_full_lateral_reserve(self):
        origin = dict(zip(ALL, [2161, 1140, 2688, 2332, 984, 1858]))
        raw = {**origin, ACTIVE: 2580}
        held = {**raw, ACTIVE: raw[ACTIVE] - 32}
        delta = (self.model.tip(raw)[:3, 3] - self.model.tip(origin)[:3, 3]) * 1000
        self.assertTrue(12. < np.linalg.norm(delta[:2]) < 15.)
        # A two-tick endpoint fits the whole limit, but the declared actual
        # local corridor would not; rejecting precedes every candidate screen.
        check_arc_attempt(self.model, origin, {**raw, ACTIVE: raw[ACTIVE] - 2})
        model = Mock(wraps=self.model)
        for cap in (8., 5., 3., 1.):
            with self.subTest(cap=cap), self.assertRaisesRegex(RuntimeError, '完整局部走廊'):
                propose_raise(model, raw, held, origin, max_distance_mm=cap)
        model.screen_segment.assert_not_called()

    def test_real_pose_inside_reserved_lateral_budget_still_has_candidate(self):
        origin = dict(zip(ALL, [2161, 1140, 2688, 2332, 984, 1858]))
        raw = {**origin, ACTIVE: 2590}; held = {**raw, ACTIVE: raw[ACTIVE] - 32}
        p, _ = propose_raise(self.model, raw, held, origin)
        self.assertTrue(p['clearance']['model_table_pass'])
        self.assertEqual(p['arc_path']['attempt_limits']['maximum_lateral_mm'], 15.)
        self.assertLessEqual(np.linalg.norm(p['requested_delta_m'][:2]), .002)

    def test_other_global_and_local_geometric_bounds_are_unchanged(self):
        # Isolate each guard from the others with a deterministic pose model;
        # these are contract boundaries, not simulated motor/collision evidence.
        actual = {**self.raw, ACTIVE: self.raw[ACTIVE] - 1}
        cases = [(check_arc_attempt, [0., 0., 60.01], 0.),
                 (check_arc_attempt, [0., 0., 1.], 15.01),
                 (check_arc_corridor, [3.01, 0., 1.], 0.),
                 (check_arc_corridor, [0., 0., 12.01], 0.),
                 (check_arc_corridor, [0., 0., 1.], 3.01)]
        for guard, delta_mm, angle in cases:
            with self.subTest(guard=guard.__name__, delta=delta_mm, angle=angle):
                target = np.eye(4); target[:3, 3] = np.asarray(delta_mm) / 1000
                target[:3, :3] = Rotation.from_euler('y', angle, degrees=True).as_matrix()
                model = Mock(); model.tip.side_effect = [np.eye(4), target]
                with self.assertRaises(RuntimeError): guard(model, self.raw, actual)

    def test_invalid_pose_and_hold_budget_rejected(self):
        for bad in ({'elbow_flex': 2839}, {**self.raw, ACTIVE: float('nan')},
                    {**self.raw, ACTIVE: True}, {**self.raw, ACTIVE: 2839.5}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                propose_raise(self.model, bad, self.held, self.raw)
        with self.assertRaises(ValueError):
            propose_raise(self.model, self.raw, {**self.held, ACTIVE: self.raw[ACTIVE] - 55}, self.raw)

    def test_failed_current_table_guard_does_not_try_shorter_candidates(self):
        model = Mock(wraps=self.model)
        model.screen_segment.return_value = {'model_table_pass': False}
        with self.assertRaisesRegex(ValueError, 'Current pose'):
            propose_raise(model, self.raw, self.held, self.raw)
        model.screen_segment.assert_called_once()

    def test_failed_candidate_table_guard_never_yields_a_proposal(self):
        model = Mock(wraps=self.model); model.joint_limits = self.model.joint_limits
        model.screen_segment.side_effect = lambda a, b, **kw: {'model_table_pass': a == b}
        with self.assertRaises(RuntimeError): propose_raise(model, self.raw, self.held, self.raw)
        self.assertGreater(model.screen_segment.call_count, 1)

    def test_attempt_reference_cannot_discard_body_drift(self):
        changed = {**self.raw, 'wrist_flex': self.raw['wrist_flex'] + 5}
        with self.assertRaises(RuntimeError): propose_raise(self.model, changed, changed, self.raw)

    def test_endpoint_rejects_a_different_motion_contract(self):
        p, _ = self.propose()
        for bad in ({**p, 'motion_kind': 'cartesian'},
                    {**p, 'joint_targets': {**p['joint_targets'], 'wrist_flex': self.raw['wrist_flex'] + 1}},
                    {**p, 'joint_targets': {**p['joint_targets'], ACTIVE: self.raw[ACTIVE] - 21}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                arc_endpoint(self.model, self.raw, self.raw, bad)


if __name__ == '__main__':
    unittest.main()
