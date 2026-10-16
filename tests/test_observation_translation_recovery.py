"""Real-CAD replay of the September 30 blocked view; no devices or cloud."""
from copy import deepcopy
import time
from types import SimpleNamespace
import unittest

from server.active_observation import (build_observation_candidates, _translation_target,
    _translation_recovery, filter_observation_framing, filter_observation_novelty,
    TRANSLATION_RECOVERY_ID, TRANSLATION_RECOVERY_CONTINUE_ID)
from server.local_cartesian import ALL, NAMES, default_model


class ObservationTranslationRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()

    def setUp(self):
        self.raw = dict(zip(ALL, (2367, 2199, 2630, 1547, 1025, 2034)))
        self.held = {**self.raw, 'shoulder_lift': 2195, 'elbow_flex': 2627, 'wrist_roll': 1022}
        self.origin = dict(self.raw)
        self.bounds = dict(shoulder_pan=[2172, 2464], shoulder_lift=[929, 2199],
            elbow_flex=[2613, 3050], wrist_flex=[1514, 2381], wrist_roll=[561, 1143])
        self.grasp = SimpleNamespace(model=self.model, state='idle',
            robot=SimpleNamespace(goals=dict(self.held), pwm_owner=None, config={}))

    def frame(self, raw=None):
        return dict(source='wrist', id='offline-actual-frame', path='/unused/wrist.jpg',
            camera_generation='offline', capture_monotonic=time.monotonic(),
            image_size=[1920, 1080], joints=dict(self.raw if raw is None else raw))

    def candidates(self, raw=None, used=('jaw_open',), bounds=None, history=()):
        return build_observation_candidates(self.grasp, self.frame(raw), origin=self.origin,
            used=used, history=history,
            calibrated_joint_bounds=self.bounds if bounds is None else bounds)

    def test_real_pose_gets_one_guarded_prefix_and_full_arm_feasibility_witness(self):
        before = deepcopy((self.raw, self.held, self.bounds, self.grasp.robot.goals))
        candidates = self.candidates()
        self.assertFalse(any(c['id'].startswith('move_') for c in candidates))
        prefix = next(c for c in candidates if c['id'] == TRANSLATION_RECOVERY_ID)
        self.assertEqual(prefix['segment']['active_joints'], ['wrist_flex'])
        self.assertEqual(prefix['segment']['joint_targets'], {**self.raw, 'wrist_flex': 1517})
        self.assertTrue(prefix['segment']['clearance']['model_table_pass'])
        self.assertGreaterEqual(min(v for k, v in
            prefix['segment']['clearance']['continuous_model_lower_bound_mm'].items() if k != 'base_link'), 5.)
        self.assertEqual(prefix['segment']['endpoint_tolerance_ticks'], 20)
        self.assertEqual(prefix['segment']['stationary_tolerance_ticks'], 12)
        witness = prefix['descriptor']['translation_lookahead']
        self.assertEqual(witness['direction'], 'up')
        self.assertGreater(witness['nominal_camera_change_model']['baseline_mm'], 25.)
        self.assertGreaterEqual(witness['minimum_model_table_clearance_mm'], 5.)
        self.assertGreater(abs(witness['nominal_target_joints']['shoulder_lift'] - self.raw['shoulder_lift']), 100)
        self.assertTrue(all(lo <= witness['nominal_target_joints'][n] <= hi for n, (lo, hi) in self.bounds.items()))
        self.assertFalse(witness['execution_authorized'])
        self.assertFalse(witness['automatic_chaining'])
        self.assertTrue(witness['requires_actual_capture_and_replanning'])
        self.assertFalse(prefix['descriptor']['information_gain_model']['metric_depth_verified'])
        self.assertEqual(before, (self.raw, self.held, self.bounds, self.grasp.robot.goals))
        # The old uncalibrated camera model crops the FUTURE baseline for this
        # edge ROI, even though the prefix itself fits. It cannot be retained
        # until the selected real camera model is attached (separate replay).
        framed = filter_observation_framing(candidates,
            dict(x=0., y=0., w=.227, h=.871), [1920, 1080])
        self.assertNotIn(TRANSLATION_RECOVERY_ID, [c['id'] for c in framed])

    def test_calibrated_limits_constrain_ik_instead_of_discarding_unconstrained_answer(self):
        unconstrained = _translation_target(self.grasp, self.raw, 'right', 20., None)
        self.assertGreater(unconstrained['shoulder_lift'], self.bounds['shoulder_lift'][1])
        constrained = _translation_target(self.grasp, self.raw, 'right', 20., None,
            calibrated_joint_bounds=self.bounds)
        self.assertTrue(all(lo <= constrained[n] <= hi for n, (lo, hi) in self.bounds.items()))
        self.assertEqual(constrained['gripper'], 2034)

    def test_next_iteration_uses_actual_endpoint_and_original_origin_without_chaining(self):
        prefix = next(c for c in self.candidates() if c['id'] == TRANSLATION_RECOVERY_ID)
        # A cheap servo stops one tick short: this is the next actual image,
        # not the prefix's hypothetical endpoint used by its offline witness.
        actual = {**self.raw, 'wrist_flex': 1518}
        self.grasp.robot.goals = dict(prefix['segment']['physical_goal_targets'])
        candidates = self.candidates(actual, used=('jaw_open', TRANSLATION_RECOVERY_ID))
        self.assertTrue(any(c['id'].startswith('move_') for c in candidates))
        self.assertNotIn(TRANSLATION_RECOVERY_ID, [c['id'] for c in candidates])
        for candidate in candidates:
            self.assertEqual(candidate['reference']['joints'], actual)
            self.assertEqual(candidate['segment']['measured_origin'], actual)
            self.assertEqual(candidate['origin'], self.origin)
            self.assertEqual(candidate['segment']['held_goals'], self.grasp.robot.goals)
            self.assertEqual(candidate['segment']['joint_targets']['gripper'], 2034)

    def test_no_prefix_when_remaining_calibrated_domain_cannot_fit_any_body_baseline(self):
        confined = {n: [self.raw[n], self.raw[n]] for n in NAMES}
        confined['wrist_flex'] = [1514, 1547]
        candidate = _translation_recovery(self.grasp, self.frame(), self.origin, None,
            directions=['up', 'backward', 'left', 'right'], calibrated_joint_bounds=confined)
        self.assertIsNone(candidate)

    def test_used_prefix_cannot_repeat_or_rebase_the_attempt(self):
        candidates = self.candidates(used=('jaw_open', TRANSLATION_RECOVERY_ID))
        self.assertNotIn(TRANSLATION_RECOVERY_ID, [c['id'] for c in candidates])
        self.assertNotIn(TRANSLATION_RECOVERY_CONTINUE_ID, [c['id'] for c in candidates])
        self.assertTrue(all(c['origin'] == self.origin for c in candidates))

    def test_partial_actual_progress_can_finish_preparation_once_without_claiming_view_novelty(self):
        prior = {**self.frame(), 'id': 'before-first-prefix',
                 'capture_monotonic': time.monotonic() - 10}
        actual = {**self.raw, 'wrist_flex': 1527}
        self.grasp.robot.goals['wrist_flex'] = 1517
        candidates = self.candidates(actual, used=(TRANSLATION_RECOVERY_ID,), history=[prior])
        continuation = next(c for c in candidates if c['id'] == TRANSLATION_RECOVERY_CONTINUE_ID)
        self.assertEqual(continuation['origin'], self.origin)
        self.assertEqual(continuation['reference']['joints'], actual)
        segment = continuation['segment']
        self.assertEqual(segment['active_joints'], ['wrist_flex'])
        self.assertEqual(segment['joint_targets']['wrist_flex'], 1514)
        self.assertEqual(segment['physical_goal_targets']['wrist_flex'], 1504)
        self.assertTrue(segment['clearance']['model_table_pass'])
        self.assertEqual(segment['endpoint_tolerance_ticks'], 20)
        descriptor = continuation['descriptor']
        self.assertEqual(descriptor['preparation_progress_model']['actual_pitch_progress_ticks'], -20)
        self.assertEqual(descriptor['translation_recovery_visit'], 2)
        self.assertEqual(descriptor['maximum_translation_recovery_visits'], 2)
        self.assertTrue(descriptor['view_novelty_model']['predicted_revisit'])
        self.assertTrue(descriptor['view_novelty_model']['retained_for_translation_preparation'])
        self.assertFalse(descriptor['information_gain_model']['metric_depth_verified'])
        self.assertGreater(descriptor['translation_lookahead']['minimum_model_table_clearance_mm'], 7)
        self.assertFalse(descriptor['translation_lookahead']['execution_authorized'])
        # The object-program filters again after receiving the current bbox.
        repeated = filter_observation_novelty(candidates, [prior, continuation['reference']], model=self.model)
        self.assertIn(TRANSLATION_RECOVERY_CONTINUE_ID, [c['id'] for c in repeated])

    def test_second_preparation_requires_fresh_same_attempt_progress_and_cannot_reverse(self):
        prior = {**self.frame(), 'capture_monotonic': time.monotonic() - 10}
        self.grasp.robot.goals['wrist_flex'] = 1517
        for name, actual, history, used in [
            ('no_progress', dict(self.raw), [prior], (TRANSLATION_RECOVERY_ID,)),
            ('reverse', {**self.raw, 'wrist_flex': 1560}, [prior], (TRANSLATION_RECOVERY_ID,)),
            ('other_camera', {**self.raw, 'wrist_flex': 1527},
                [{**prior, 'camera_generation': 'old'}], (TRANSLATION_RECOVERY_ID,)),
            ('future', {**self.raw, 'wrist_flex': 1527},
                [{**prior, 'capture_monotonic': time.monotonic() + 100}], (TRANSLATION_RECOVERY_ID,)),
            ('unaligned_history', {**self.raw, 'wrist_flex': 1527}, [], (TRANSLATION_RECOVERY_ID,)),
            ('intervening_move', {**self.raw, 'wrist_flex': 1527}, [prior, prior],
                (TRANSLATION_RECOVERY_ID, 'roll_positive')),
            ('third_attempt', {**self.raw, 'wrist_flex': 1527}, [prior, prior],
                (TRANSLATION_RECOVERY_ID, TRANSLATION_RECOVERY_CONTINUE_ID)),
        ]:
            with self.subTest(name=name):
                candidates = self.candidates(actual, used=used, history=history)
                self.assertNotIn(TRANSLATION_RECOVERY_CONTINUE_ID, [c['id'] for c in candidates])
                self.assertNotIn(TRANSLATION_RECOVERY_ID, [c['id'] for c in candidates])


if __name__ == '__main__':
    unittest.main()
