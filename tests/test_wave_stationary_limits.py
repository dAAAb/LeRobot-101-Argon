"""Recorded near-limit pose through CAD and the real driver with a fake bus.

No hardware, HTTP or camera access. The plant keeps held body joints still and
simulates five ticks of wrist load error; only wrist-roll commands are emitted.
"""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

import test_wave_tolerance as tolerance_helpers
from server.local_cartesian import ALL, NAMES, default_model


RAW = dict(zip(ALL, [2314, 1016, 3042, 2140, 765, 1859]))
HELD = dict(zip(ALL, [2313, 1017, 3041, 2137, 765, 1859]))


class RotationStationaryLimitsTests(unittest.TestCase):
    model = default_model()

    def proposal(self, raw=None, held=None, **kwargs):
        return self.model.propose_rotation(raw or RAW, 'roll', 30 * 360 / 4096,
            held_goals=held or HELD, uncertainty_ticks=12, **kwargs)

    def test_fixed_elbow_near_limit_has_clipped_runtime_and_full_geometry_box(self):
        p = self.proposal()
        self.assertEqual(p['active_joint_targets'], {'wrist_roll': 795})
        self.assertEqual(p['physical_goal_targets'], {'wrist_roll': 795})
        self.assertFalse(p['stationary_goal_resends'])
        self.assertEqual(p['preserved_physical_hold_goals']['elbow_flex'], 3041)
        self.assertEqual(p['measured_bounds']['elbow_flex'], [3030, 3045])
        self.assertEqual(p['physical_goal_bounds']['elbow_flex'], [3030, 3045])
        self.assertEqual(p['clearance']['actual_tracking_envelope_ticks']['elbow_flex'], 12)
        self.assertEqual(p['clearance']['limit_validation_uncertainty_ticks']['elbow_flex'], 0)
        self.assertTrue(p['clearance']['model_table_pass'])
        # Geometry deliberately includes hypothetical positions past the hard
        # boundary; runtime clipping must not reduce the table error bound.
        self.assertGreater(p['clearance']['continuous_residual_bound_mm']['gripper_link'], 0)
        self.assertEqual(p['clearance']['actual_tracking_envelope_ticks']['wrist_roll'], 12)

    def test_fixed_measured_or_held_position_outside_hard_limit_still_rejects(self):
        for raw, held in [({**RAW, 'elbow_flex': 3046}, HELD),
                          (RAW, {**HELD, 'elbow_flex': 3046})]:
            with self.subTest(raw=raw, held=held), self.assertRaises(ValueError):
                self.proposal(raw, held)

    def test_moving_axis_still_needs_twelve_ticks_beyond_entire_error_box(self):
        model = deepcopy(self.model)
        # Target 795 + envelope 12 = 807, but the moving reserve needs 819.
        model.calibration['wrist_roll']['range_max'] = 818
        with self.assertRaisesRegex(ValueError, '行程餘量'):
            model.propose_rotation(RAW, 'roll', 30 * 360 / 4096,
                                   held_goals=HELD, uncertainty_ticks=12)

    def test_table_failure_remains_a_veto_for_fixed_near_limit_pose(self):
        with self.assertRaisesRegex(ValueError, 'table guard'):
            self.proposal(min_clearance_m=1.)


class WaveStationaryLimitsTests(unittest.TestCase):
    model = default_model()
    capture = tolerance_helpers.WaveEndpointToleranceTests.capture

    def setUp(self):
        tolerance_helpers.WaveEndpointToleranceTests.setUp(self)
        self.robot.calibration = deepcopy(self.model.calibration)
        for name, entry in self.robot.calibration.items():
            self.bus.reg[entry['id']].update({56: RAW[name], 42: HELD[name]})
        self.robot.goals = dict(HELD)
        self.robot.bias = {n: HELD[n] - RAW[n] for n in ALL}
        self.initial = dict(RAW); self.initial_holds = dict(HELD)
        self.selected_raise_joint = None

        def infer(prompt, frames, schema, **kwargs):
            data = json.loads(prompt.split('\nINPUT_DATA_JSON\n')[1])
            self.assertIn('in_place', [p['program_id'] for p in data['programs']])
            program = next(p for p in data['programs']
                           if p['raise_joint'] == self.selected_raise_joint)
            return schema.model_validate({'program_id': program['program_id'],
                'gripper_empty': True, 'human_clear': True, 'clear_path': True,
                'confidence': .86, 'reason': 'Synthetic whole-path clearance.'})
        self.brain._infer = Mock(side_effect=infer)

    def test_planned_wave_completes_at_recorded_pose_and_writes_only_roll(self):
        result = self.grasp.planned_wave(self.reports.append, self.cancel)
        self.assertTrue(result['wave_completed'])
        self.assertEqual(result['cycles'], 2)
        self.assertEqual(len(result['frames']), 8)
        self.brain._infer.assert_called_once()
        self.assertEqual({(servo, address) for servo, address, _ in self.bus.events}, {(5, 42)})
        for call in self.calls:
            self.assertEqual(set(call['targets']), {'wrist_roll'})
            self.assertEqual(call['kwargs']['measured_bounds']['elbow_flex'], [3038, 3045])
        for n in NAMES:
            if n != 'wrist_roll':
                self.assertEqual(self.robot.read()[n], RAW[n])
                self.assertEqual(self.robot.goals[n], HELD[n])

    def test_program_guard_rejects_four_tick_drift_that_crosses_hard_limit(self):
        real_sweep = self.grasp.roll_sweep
        def crossing(goal, reference, cancel, **kwargs):
            # 3046 is only four ticks away, but exceeds the actual 3045 limit.
            kwargs['program_guard']({**RAW, 'elbow_flex': 3046})
            return real_sweep(goal, reference, cancel, **kwargs)
        with patch.object(self.grasp, 'roll_sweep', side_effect=crossing):
            with self.assertRaisesRegex(ValueError, 'screened raw interval.*elbow_flex'):
                self.grasp.planned_wave(self.reports.append, self.cancel)
        self.assertEqual(self.bus.events, [])

    def test_planned_wave_45_second_budget_is_not_reset_at_each_leg(self):
        real_sweep = self.grasp.roll_sweep
        sweeps = []
        def late_leg(*args, **kwargs):
            sweeps.append(self.clock.now)
            if len(sweeps) == 3:
                self.clock.now = sweeps[0] + 45
                self.writes_at_expiry = len(self.bus.events)
            return real_sweep(*args, **kwargs)
        with patch.object(self.grasp, 'roll_sweep', side_effect=late_leg):
            with self.assertRaisesRegex(RuntimeError, '已到時間'):
                self.grasp.planned_wave(self.reports.append, self.cancel)
        self.assertEqual(len(sweeps), 3)
        self.assertGreater(self.writes_at_expiry, 0)
        self.assertEqual(len(self.bus.events), self.writes_at_expiry)
        self.assertFalse(any(event['type'] == 'wave_result' for event in self.events))

    def test_wrist_raise_then_wave_uses_one_plan_and_never_commands_near_limit_elbow(self):
        # Subsequent real snapshot differed by one wrist-flex encoder tick.
        self.bus.reg[self.robot.calibration['wrist_flex']['id']][56] = 2141
        self.selected_raise_joint = 'wrist_flex'
        result = self.grasp.planned_wave(self.reports.append, self.cancel)
        self.assertTrue(result['wave_completed'])
        self.assertEqual(result['cycles'], 2)
        self.assertEqual(result['planning_calls'], 1)
        self.assertGreater(result['lifted_model_mm'], 0)
        self.assertEqual(len(result['raise_steps']), 1)
        self.assertEqual(len(result['frames']), 8)
        self.brain._infer.assert_called_once()
        self.assertEqual(set(self.calls[0]['targets']), {'wrist_flex'})
        self.assertTrue(all(set(c['targets']) == {'wrist_roll'} for c in self.calls[1:]))
        writes = [(servo, address) for servo, address, _ in self.bus.events]
        self.assertEqual(set(writes), {(4, 42), (5, 42)})
        first_roll = writes.index((5, 42))
        self.assertTrue(all(write == (4, 42) for write in writes[:first_roll]))
        self.assertTrue(all(write == (5, 42) for write in writes[first_roll:]))
        for n in ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'gripper'):
            self.assertEqual(self.robot.read()[n], RAW[n])
            self.assertEqual(self.robot.goals[n], HELD[n])

    def test_runtime_drift_inside_hard_limit_still_cannot_exceed_four_ticks(self):
        def drifting(goal, reference, cancel, **kwargs):
            kwargs['program_guard']({**RAW, 'elbow_flex': 3037})
        with patch.object(self.grasp, 'roll_sweep', side_effect=drifting):
            with self.assertRaisesRegex(ValueError, 'screened raw interval.*elbow_flex'):
                self.grasp.planned_wave(self.reports.append, self.cancel)
        self.assertEqual(self.bus.events, [])


if __name__ == '__main__':
    unittest.main()
