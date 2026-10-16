"""Jev may rank cleared options; it never supplies a robot path or clearance.

Uses the real WristGrasp.move and CAD with synthetic images and the existing
in-memory plant. No credential access, network, physical motors or cameras.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_wave as helpers
import test_wave_retreat_move as retreat_helpers
from server.gripper_power import SessionCancelled


def choice(selected='backward', *, status='selected', reason='selected'):
    return {'provider': 'jev', 'status': status, 'selected': selected,
            'confidence': .9, 'reason': reason}


class JevWaveIntegrationTests(unittest.TestCase):
    model = helpers.default_model()
    capture = helpers.WaveTests.capture
    options = retreat_helpers.RetreatMoveTests.options
    prepare = retreat_helpers.RetreatMoveTests.prepare

    def setUp(self):
        helpers.WaveTests.setUp(self)
        self.reset_plant()
        self.tactician = SimpleNamespace(choose=Mock(return_value=choice()))
        self.grasp.tactician = self.tactician

    def reset_plant(self):
        self.robot.actual.update(dict(shoulder_pan=2162, shoulder_lift=1141, elbow_flex=2694,
                                      wrist_flex=2429, wrist_roll=984, gripper=1857))
        self.robot.goals = {**self.robot.actual, 'elbow_flex': 2653,
                            'shoulder_lift': 1142, 'wrist_flex': 2430}
        self.robot.bias = {n: self.robot.goals[n] - self.robot.actual[n] for n in self.robot.actual}
        self.robot.calls.clear(); self.robot.writes.clear()

    def available(self):
        result = self.options(left=True, backward=True)
        result['left']['confidence'] = .91
        result['backward']['confidence'] = .78
        return result

    def run_one_retreat(self, first=None):
        with patch('server.wave_planner.assess_retreat_options', side_effect=[
                first or self.available(), self.options(wave=True)]) as assess:
            result = self.prepare()
        return result, assess

    def test_jev_can_choose_lower_scored_eligible_backward_path(self):
        result, assess = self.run_one_retreat()
        self.tactician.choose.assert_called_once()
        state, candidates = self.tactician.choose.call_args.args
        self.assertEqual(set(candidates), {'left', 'backward'})
        self.assertEqual(set(state['evidence']), {'left', 'backward'})
        self.assertEqual(set(state['candidate_geometry']), {'left', 'backward'})
        for candidate in state['candidate_geometry'].values():
            self.assertEqual(len(candidate['delta_base_model_m']),3)
            self.assertEqual(len(candidate['delta_camera_model_m']),3)
            self.assertEqual(candidate['geometry_accuracy'],'cad_prior_not_measured_obstacle_distance')
        from server.jev import _semantic_json
        _semantic_json(state)
        self.assertNotIn('right', state['evidence'])
        self.assertIs(self.tactician.choose.call_args.kwargs['cancel'], self.cancel)
        for evidence in state['evidence'].values():
            self.assertTrue(all(evidence[k] for k in ('gripper_empty', 'human_clear', 'clear_path')))
        self.assertEqual(len(result['retreat_steps']), 1)
        receipt = result['retreat_steps'][0]
        self.assertEqual(receipt['direction'], 'backward')
        self.assertEqual(receipt['motion_kind'], 'cartesian_backward')
        self.assertNotIn('wave_completed', result)
        self.assertEqual(set(n for n, _ in self.robot.writes), {'shoulder_lift', 'elbow_flex', 'wrist_flex'})
        self.assertIsNone(assess.call_args.kwargs['retreat_proposals']['left'])
        self.assertIsNone(assess.call_args.kwargs['retreat_proposals']['right'])
        self.assertGreater(receipt['before']['capture_monotonic'], 3., 'real move captures after Jev selection')

    def test_remote_unknown_or_blocked_choice_cannot_add_a_candidate(self):
        for selected in ('right', 'unlisted_skill'):
            with self.subTest(selected=selected):
                self.reset_plant(); self.tactician.choose.reset_mock()
                self.tactician.choose.return_value = choice(selected)
                result, _ = self.run_one_retreat()
                self.assertEqual(result['retreat_steps'][0]['direction'], 'left')
                self.assertEqual(set(n for n, _ in self.robot.writes), {'shoulder_pan'})
                self.assertEqual(set(self.tactician.choose.call_args.args[1]), {'left', 'backward'})

    def test_timeout_or_abstention_keeps_deterministic_cleared_fallback(self):
        for reason in ('timeout', 'model_abstained', 'invalid_response', 'missing_key'):
            with self.subTest(reason=reason):
                self.reset_plant()
                self.tactician.choose.return_value = choice('right', status='abstain', reason=reason)
                result, _ = self.run_one_retreat()
                self.assertEqual(result['retreat_steps'][0]['direction'], 'left')
                self.assertEqual(set(n for n, _ in self.robot.writes), {'shoulder_pan'})

    def test_every_local_clearance_flag_excludes_candidate_before_jev(self):
        for field in ('gripper_empty', 'human_clear', 'clear_path', 'confidence'):
            with self.subTest(field=field):
                self.reset_plant(); self.tactician.choose.reset_mock()
                options = self.available()
                options['right'] = {**options['left'], field: .699 if field == 'confidence' else False}
                self.tactician.choose.return_value = choice('right')
                result, _ = self.run_one_retreat(options)
                self.assertNotIn('right', self.tactician.choose.call_args.args[1])
                self.assertNotIn('right', self.tactician.choose.call_args.args[0]['evidence'])
                self.assertEqual(result['retreat_steps'][0]['direction'], 'left')

    def test_no_eligible_candidate_means_no_jev_request_or_motor_write(self):
        with patch('server.wave_planner.assess_retreat_options', return_value=self.options()):
            with self.assertRaises(RuntimeError): self.prepare()
        self.tactician.choose.assert_not_called(); self.assertEqual(self.robot.writes, [])

    def test_single_eligible_candidate_does_not_need_jev(self):
        result, _ = self.run_one_retreat(self.options(backward=True))
        self.tactician.choose.assert_not_called()
        self.assertEqual(result['retreat_steps'][0]['direction'], 'backward')

    def test_cancellation_during_jev_cannot_fall_back_to_any_motion(self):
        def cancelled(*args, **kwargs):
            self.cancel.set()
            return choice(None, status='abstain', reason='cancelled')
        self.tactician.choose.side_effect = cancelled
        with patch('server.wave_planner.assess_retreat_options', return_value=self.available()):
            with self.assertRaises(SessionCancelled): self.prepare()
        self.assertEqual(self.robot.writes, [])
        self.assertEqual(self.robot.calls, [])

    def test_late_changed_image_is_rejected_by_real_move_after_jev(self):
        def choose_then_change(*args, **kwargs):
            self.capture_hook = lambda frame, index: frame.update(path=str(self.changed))
            return choice()
        self.tactician.choose.side_effect = choose_then_change
        with patch('server.wave_planner.assess_retreat_options', return_value=self.available()):
            with self.assertRaisesRegex(ValueError, '場景已改變'): self.prepare()
        self.tactician.choose.assert_called_once()
        self.assertGreaterEqual(self.frames, 4)
        self.assertEqual(self.robot.writes, []); self.assertEqual(self.robot.calls, [])

    def test_late_generation_change_is_rejected_by_real_move_after_jev(self):
        def choose_then_reconnect(*args, **kwargs):
            self.robot.camera.generation = 'reconnected-after-jev'
            return choice()
        self.tactician.choose.side_effect = choose_then_reconnect
        with patch('server.wave_planner.assess_retreat_options', return_value=self.available()):
            with self.assertRaisesRegex(ValueError, '相機已重新連接'): self.prepare()
        self.assertEqual(self.robot.writes, []); self.assertEqual(self.robot.calls, [])

    def test_late_hold_goal_change_rejects_the_exact_assessed_path(self):
        def choose_then_goal_change(*args, **kwargs):
            self.robot.goals['elbow_flex'] += 1
            return choice()
        self.tactician.choose.side_effect = choose_then_goal_change
        with patch('server.wave_planner.assess_retreat_options', return_value=self.available()):
            with self.assertRaisesRegex(RuntimeError, '位置有變'): self.prepare()
        self.assertEqual(self.robot.writes, []); self.assertEqual(self.robot.calls, [])


if __name__ == '__main__':
    unittest.main()
