"""One view-recovery prefix with real local driver and a fake motor bus."""
import json
from copy import deepcopy
import threading
import unittest
from unittest.mock import Mock, patch

import test_wave_stationary_limits as helpers
from server.gripper_power import SessionCancelled
from server.grasp import PLANNED_WAVE_EXECUTION_SECONDS


class WaveViewRecoveryTests(unittest.TestCase):
    model = helpers.WaveStationaryLimitsTests.model
    capture = helpers.WaveStationaryLimitsTests.capture

    @classmethod
    def setUpClass(cls):
        from server.wave_observation import build_wave_observation_prefixes
        # One real CAD/IK fixture, then independent executions and fault cases.
        # Recomputing the same planner in every runtime case adds minutes while
        # providing no extra controller coverage. The planner has its own suite.
        cls.prefix_fixture = build_wave_observation_prefixes(cls.model, helpers.RAW, helpers.HELD)

    def setUp(self):
        helpers.WaveStationaryLimitsTests.setUp(self)
        self.final_clear = True
        self.prefix_clear = True
        self.prefix_confidence = .78
        self.prefix_kind = 'wrist_pitch_raise'
        self.prefix_direction = None
        self.cloud_wait = 0
        self.after_final = None
        self.inferences = []
        self.brain._infer = Mock(side_effect=self.infer)
        def prefixes(model, raw, held, **kwargs):
            self.assertIs(model, self.model)
            self.assertEqual(raw, helpers.RAW)
            self.assertEqual(held, helpers.HELD)
            return deepcopy(self.prefix_fixture)
        builder = patch('server.wave_observation.build_wave_observation_prefixes', side_effect=prefixes)
        builder.start(); self.addCleanup(builder.stop)

    def infer(self, prompt, frames, schema, **kwargs):
        data = json.loads(prompt.split('\nINPUT_DATA_JSON\n')[1])
        self.inferences.append((data, len(self.bus.events)))
        step = len(self.inferences)
        self.assertFalse(self.robot.moving)
        if step == 1:
            self.assertIn('programs', data)
            self.assertEqual(self.bus.events, [])
            return schema.model_validate({'program_id': None, 'gripper_empty': True,
                'human_clear': True, 'clear_path': False, 'confidence': .9,
                'reason': '目前夾爪擋住視線，尚不能確認整段揮手。'})
        if step == 2:
            self.assertEqual(self.bus.events, [])
            self.assertIn('observation_prefixes', data)
            self.assertTrue(any(p['kind'] == 'pan_retreat' for p in data['observation_prefixes']))
            self.assertTrue(any(p['kind'] == 'wrist_pitch_raise' for p in data['observation_prefixes']))
            selected = next(p for p in data['observation_prefixes'] if p['kind'] == self.prefix_kind
                            and (self.prefix_direction is None or p['spatial_direction'] == self.prefix_direction))
            self.selected_prefix = selected
            return schema.model_validate({'program_id': selected['program_id'],
                'gripper_empty': True, 'human_clear': True,
                'clear_prefix': self.prefix_clear, 'confidence': self.prefix_confidence,
                'reason': '這一段換角度的路徑有空間。'})
        self.assertEqual(step, 3)
        self.assertEqual([p['program_id'] for p in data['programs']], ['in_place'])
        self.assertEqual(len(frames), 2)
        executed_before = next(event['before'] for event in self.events
                               if event['type'] == 'motion_program_segment_start')
        self.assertEqual(frames[0], executed_before)
        self.assertEqual(frames[-1]['joints'], self.robot.read())
        self.assertGreater(frames[-1]['capture_monotonic'], frames[0]['capture_monotonic'])
        self.assertEqual(frames[0]['camera_generation'], frames[-1]['camera_generation'])
        self.assertEqual(data['current_frame_index'], 1)
        self.assertEqual(data['current_measured_joints'], frames[-1]['joints'])
        self.assertIn('observation_motion', data)
        self.assertIsNotNone(data['observation_motion'])
        self.assertTrue(self.bus.events)
        active = self.selected_prefix.get('active_joints', [self.selected_prefix.get('active_joint')])
        self.assertEqual({servo for servo, _, _ in self.bus.events},
                         {self.robot.calibration[n]['id'] for n in active})
        self.clock.now += self.cloud_wait
        if self.after_final:self.after_final()
        return schema.model_validate({'program_id': 'in_place' if self.final_clear else None,
            'gripper_empty': True, 'human_clear': True, 'clear_path': self.final_clear,
            'confidence': .86, 'reason': '現在可確認揮手空間。' if self.final_clear else '圓筒仍靠近夾爪。'})

    def run_wave(self, **kwargs):
        return self.grasp.planned_wave(self.reports.append, self.cancel, **kwargs)

    def assert_no_wave_result(self):
        self.assertFalse(any(event['type'] == 'wave_result' for event in self.events))

    def test_point78_prefix_then_reobserve_and_wave_uses_three_calls_and_one_raise(self):
        # Stationary cloud latency must not consume the remaining motion budget.
        self.cloud_wait = 120
        # Reproduce roughly four seconds per measured endpoint, including the
        # real driver's execution and the camera/settling latency seen live.
        # The total physical sequence now exceeds the old 30-second cutoff.
        real_sweep = self.grasp.roll_sweep
        sweep_starts = []
        def settled_endpoint(*args, **kwargs):
            if sweep_starts:
                self.clock.now = max(self.clock.now, sweep_starts[0] + 4.2 * len(sweep_starts))
            sweep_starts.append(self.clock.now)
            return real_sweep(*args, **kwargs)
        with patch.object(self.grasp, 'roll_sweep', side_effect=settled_endpoint):
            result = self.run_wave()
        self.assertTrue(result['wave_completed'])
        self.assertTrue(result['view_recovery'])
        self.assertEqual(result['planning_calls'], 3)
        self.assertEqual(result['program_id'], 'in_place')
        self.assertEqual(result['view_recovery_clearance']['confidence'], .78)
        self.assertEqual(len(result['raise_steps']), 1)
        self.assertEqual(len(result['frames']), 8)
        self.assertGreater(result['view_recovery_execution_seconds'], 0)
        self.assertLess(result['view_recovery_execution_seconds'], PLANNED_WAVE_EXECUTION_SECONDS)
        physical_elapsed = result['view_recovery_execution_seconds'] + self.clock.now - sweep_starts[0]
        self.assertGreater(physical_elapsed, 30)
        self.assertLess(physical_elapsed, 45)
        self.assertEqual(set(self.calls[0]['targets']), {'wrist_flex'})
        self.assertTrue(all(set(c['targets']) == {'wrist_roll'} for c in self.calls[1:]))

    def test_final_rejection_holds_after_one_prefix_without_any_roll_or_retry(self):
        self.final_clear = False
        with self.assertRaisesRegex(RuntimeError, '已換角度.*仍不能揮手'):
            self.run_wave()
        self.assertEqual(self.brain._infer.call_count, 3)
        self.assertEqual({servo for servo, _, _ in self.bus.events}, {4})
        self.assertEqual(len(self.calls), 1)
        self.assert_no_wave_result()

    def test_point69_prefix_is_rejected_without_motion(self):
        self.prefix_confidence = .69
        with self.assertRaisesRegex(RuntimeError, '還找不到可換角度'):
            self.run_wave()
        self.assertEqual(self.brain._infer.call_count, 2)
        self.assertEqual(self.bus.events, [])

    def test_initial_occupied_hand_or_human_veto_is_never_reasked(self):
        for field in ('gripper_empty', 'human_clear'):
            with self.subTest(field=field):
                self.brain._infer = Mock(return_value={'program_id': None,
                    'gripper_empty': True, 'human_clear': True, 'clear_path': False,
                    'confidence': .9, 'reason': '手邊有人或正拿著物件。', field: False})
                with self.assertRaisesRegex(RuntimeError, '目前還不能完成揮手'):
                    self.run_wave()
                self.brain._infer.assert_called_once()
                self.assertEqual(self.bus.events, [])

    def test_changed_scene_after_final_assessment_does_not_start_roll(self):
        def change():
            self.capture_hook = lambda frame, _: frame.update(path=str(self.changed))
        self.after_final = change
        with self.assertRaises(ValueError):
            self.run_wave()
        self.assertEqual({servo for servo, _, _ in self.bus.events}, {4})
        self.assert_no_wave_result()

    def test_camera_reconnect_after_prefix_cannot_authorize_wave(self):
        self.after_final = lambda: setattr(self.robot.camera, 'generation', 'reconnected')
        with self.assertRaises(ValueError):
            self.run_wave()
        self.assertEqual({servo for servo, _, _ in self.bus.events}, {4})
        self.assert_no_wave_result()

    def test_cancel_during_final_assessment_does_not_start_roll(self):
        self.after_final = self.cancel.set
        with self.assertRaises(SessionCancelled):
            self.run_wave()
        self.assertEqual({servo for servo, _, _ in self.bus.events}, {4})
        self.assert_no_wave_result()

    def test_provider_failure_never_becomes_recovery_permission(self):
        self.brain._infer = Mock(side_effect=RuntimeError('provider unavailable'))
        with self.assertRaisesRegex(RuntimeError, 'provider unavailable'):
            self.run_wave()
        self.brain._infer.assert_called_once()
        self.assertEqual(self.bus.events, [])

    def test_manual_override_after_prefix_keeps_remaining_physical_budget(self):
        from server.motion_program import execute_prepared_segment
        override = threading.Event()
        self.after_final = override.set
        prefix_end = []
        def almost_expired(*args, **kwargs):
            started = self.clock.now
            result = execute_prepared_segment(*args, **kwargs)
            self.clock.now = started + 44.95
            prefix_end.append(self.clock.now)
            return result
        with patch('server.motion_program.execute_prepared_segment', side_effect=almost_expired):
            with self.assertRaisesRegex(RuntimeError, '已到時間'):
                self.run_wave(manual_override=override)
        self.assertLess(self.clock.now - prefix_end[0], 1.)
        self.assertTrue(self.robot.stop.is_set())
        self.assert_no_wave_result()

    def test_single_sideways_prefix_then_wave_only_commands_pan_then_roll(self):
        self.prefix_kind = 'pan_retreat'
        result = self.run_wave()
        self.assertTrue(result['wave_completed'])
        self.assertEqual(result['planning_calls'], 3)
        self.assertEqual(result['view_recovery_kind'], 'pan_retreat')
        self.assertEqual(result['raise_steps'], [])
        self.assertEqual(len(result['retreat_steps']), 1)
        self.assertGreater(result['retreated_model_mm'], 15)
        self.assertLessEqual(result['retreated_model_mm'], 40)
        self.assertEqual(set(self.calls[0]['targets']), {'shoulder_pan'})
        self.assertTrue(all(set(c['targets']) == {'wrist_roll'} for c in self.calls[1:]))
        writes = [(servo, address) for servo, address, _ in self.bus.events]
        self.assertEqual(set(writes), {(1, 42), (5, 42)})
        first_roll = writes.index((5, 42))
        self.assertTrue(all(write == (1, 42) for write in writes[:first_roll]))
        self.assertTrue(all(write == (5, 42) for write in writes[first_roll:]))
        for n in ('shoulder_lift', 'elbow_flex', 'wrist_flex', 'gripper'):
            self.assertEqual(self.robot.read()[n], helpers.RAW[n])
            self.assertEqual(self.robot.goals[n], helpers.HELD[n])

    def test_sideways_prefix_final_refusal_cannot_start_roll_or_try_another_prefix(self):
        self.prefix_kind = 'pan_retreat'; self.final_clear = False
        with self.assertRaisesRegex(RuntimeError, '已換角度.*仍不能揮手'):
            self.run_wave()
        self.assertEqual(self.brain._infer.call_count, 3)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual({servo for servo, _, _ in self.bus.events}, {1})
        self.assert_no_wave_result()

    def test_whole_arm_up_prefix_coordinates_multiple_joints_then_waves(self):
        self.prefix_kind = 'body_translation'; self.prefix_direction = 'up'
        result = self.run_wave()
        self.assertTrue(result['wave_completed'])
        self.assertEqual(result['planning_calls'], 3)
        self.assertEqual(result['view_recovery_kind'], 'body_translation')
        self.assertEqual(result['view_recovery_direction'], 'up')
        self.assertEqual(len(result['raise_steps']), 1)
        self.assertEqual(result['retreat_steps'], [])
        self.assertGreater(result['lifted_model_mm'], 20)
        active = set(self.calls[0]['targets'])
        self.assertGreaterEqual(len(active), 3)
        self.assertNotIn('elbow_flex', active)
        self.assertTrue(all(set(c['targets']) == {'wrist_roll'} for c in self.calls[1:]))
        self.assertEqual(self.robot.read()['elbow_flex'], helpers.RAW['elbow_flex'])
        self.assertEqual(self.robot.goals['elbow_flex'], helpers.HELD['elbow_flex'])
        self.assertNotIn(3, {servo for servo, _, _ in self.bus.events})
        self.assertNotIn(6, {servo for servo, _, _ in self.bus.events})

    def test_whole_arm_backward_prefix_final_refusal_keeps_one_move_only(self):
        self.prefix_kind = 'body_translation'; self.prefix_direction = 'backward'; self.final_clear = False
        with self.assertRaisesRegex(RuntimeError, '已換角度.*仍不能揮手'):
            self.run_wave()
        self.assertEqual(self.brain._infer.call_count, 3)
        self.assertEqual(len(self.calls), 1)
        self.assertGreaterEqual(len(self.calls[0]['targets']), 3)
        self.assertNotIn(3, {servo for servo, _, _ in self.bus.events})
        self.assertNotIn(6, {servo for servo, _, _ in self.bus.events})
        self.assert_no_wave_result()


if __name__ == '__main__':
    unittest.main()
