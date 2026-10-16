"""Actual SO101 CAD numerical regressions; no motors, camera, or model calls.

Synthetic jaw landmarks below exercise the calibration contract. They are not
measurements of the user's robot and are never saved as a real jaw profile.
"""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from server.local_cartesian import ALL, NAMES, default_model
from server.motion_program import _validate_contract
from server.spatial_contract import JawCalibration, JawSample
from server.task_space_plan import (TaskSpaceError, plan_grasp_lift,
                                    plan_lift_from_contact, plan_surface_approach, solve_task_pose,
                                    _path, _tcp)


class TaskSpacePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()
        cls.raw = dict(zip(ALL, [2155, 1182, 2946, 2268, 977, 1857]))
        transforms = cls.model.fk.transforms(cls.raw)
        pad = np.linalg.inv(transforms['gripper_link']) @ transforms['gripper_frame_link']
        fixed, closing = pad[:3, 3], pad[:3, 0]
        cls.profile = JawCalibration('synthetic-jaw', 'test-geometry', (
            JawSample(1757, fixed, fixed + closing * .020),
            JawSample(1957, fixed, fixed + closing * .040)), pad[:3, 1],
            .001, 'validated', 'synthetic-heldout-only', 'independent-heldout')
        cls.tcp = cls.profile.at(cls.raw['gripper'])
        cls.evidence = {'source': 'wrist', 'frame_id': 'fixture-1',
                        'camera_generation': 'fixture-camera',
                        'geometry_revision': 'test-geometry',
                        'calibration_revision': 'test-camera',
                        'geometry_verified': True, 'target_verified': True}

    def tip(self, raw=None, tcp=None):
        return self.model.fk.transforms(raw or self.raw)['gripper_link'] @ np.asarray(
            (tcp or self.tcp).T_wrist_tcp.matrix)

    def surface(self, *, tcp=None, evidence=None, **kwargs):
        T = self.tip(tcp=tcp) if tcp else self.model.tip(self.raw)
        return plan_surface_approach(self.model, self.raw, self.raw,
            surface_base_m=T[:3, 3] + [0, 0, .045], target_uncertainty_m=.002,
            tcp=tcp, evidence=evidence, **kwargs)

    def test_reaches_known_fk_target_with_full_calibrated_tcp_and_axis(self):
        target_raw = {**self.raw, 'shoulder_pan': 2161, 'shoulder_lift': 1200,
                      'elbow_flex': 2910, 'wrist_flex': 2287, 'wrist_roll': 985}
        target = self.tip(target_raw)
        result = solve_task_pose(self.model, self.raw, target_base_m=target[:3, 3],
                                approach_axis_base=target[:3, 2], tcp=self.tcp)
        reached = self.tip(result['joint_targets'])
        self.assertLess(np.linalg.norm(reached[:3, 3] - target[:3, 3]), .001)
        self.assertGreater(reached[:3, 2] @ target[:3, 2], .999)
        self.assertEqual(result['joint_targets']['gripper'], self.raw['gripper'])
        self.assertEqual(set(result['body_target']), set(NAMES))

    def test_five_dof_contract_accepts_axis_but_no_six_dof_orientation_request(self):
        T = self.tip()
        result = solve_task_pose(self.model, self.raw, target_base_m=T[:3, 3],
                                approach_axis_base=T[:3, 2], tcp=self.tcp)
        self.assertIn('free_axial_rotation', result['constraint'])
        self.assertEqual(result['joint_targets'], self.raw)
        with self.assertRaises(TypeError):
            solve_task_pose(self.model, self.raw, target_base_m=T[:3, 3],
                approach_axis_base=T[:3, 2], orientation_quaternion=[0, 0, 0, 1])

    def test_opposite_axis_is_not_mistaken_for_zero_error(self):
        T = self.tip()
        with self.assertRaises(TaskSpaceError) as failure:
            solve_task_pose(self.model, self.raw, target_base_m=T[:3, 3],
                            approach_axis_base=-T[:3, 2], tcp=self.tcp)
        self.assertEqual(failure.exception.code, 'unreachable_task_pose')
        self.assertGreater(failure.exception.diagnostics['axis_error_deg'], 30)

    def test_unreachable_target_has_structured_error_not_current_pose_success(self):
        T = self.tip()
        with self.assertRaises(TaskSpaceError) as failure:
            solve_task_pose(self.model, self.raw, target_base_m=[2., 2., 2.],
                            approach_axis_base=T[:3, 2], tcp=self.tcp)
        self.assertEqual(failure.exception.code, 'unreachable_task_pose')
        self.assertGreater(failure.exception.diagnostics['position_error_m'], 1.)

    def test_surface_standoff_retains_surface_semantics_and_unknown_calibration(self):
        plan = self.surface()
        self.assertTrue(plan['segments'])
        self.assertTrue(plan['proposal_only'])
        self.assertFalse(plan['metric_inputs_verified'])
        self.assertIn('two_pad_tcp_unverified', plan['contact_blockers'])
        self.assertIn('tcp_uncertainty_unknown', plan['contact_blockers'])
        target = plan['binding']['target']
        self.assertFalse(target['surface_is_grasp_center'])
        self.assertAlmostEqual(np.linalg.norm(np.asarray(target['observed_surface_base_m'])
            - target['endpoint_target_base_m']), .030)
        self.assertTrue(all(s['kind'] == 'approach' for s in plan['segments']))

    def test_empty_hand_fixed_pad_approach_does_not_require_grasp_center(self):
        plan = self.surface(evidence=self.evidence)
        self.assertTrue(plan['ready_for_external_assessment'])
        self.assertTrue(plan['metric_inputs_verified'])
        self.assertIn('two_pad_tcp_unverified', plan['contact_blockers'])
        self.assertEqual(plan['binding']['tcp']['frame'], 'gripper_frame_link')

    def test_offline_ik_steps_are_merged_into_continuous_physical_segment(self):
        plan = self.surface()
        self.assertEqual(len(plan['segments']), 1)
        self.assertGreater(plan['segments'][0]['offline_waypoints_merged'], 1)

    def test_exact_executor_contracts_and_fk_preview_match_all_samples(self):
        plan = self.surface(tcp=self.tcp, evidence=self.evidence)
        self.assertTrue(plan['metric_inputs_verified'])
        self.assertTrue(plan['ready_for_external_assessment'])
        self.assertTrue(plan['requires_runtime_authorization'])
        self.assertTrue(plan['not_environment_collision_certificate'])
        for step in plan['segments']:
            contract = step['contract']
            _validate_contract(contract)
            self.assertTrue(contract['clearance']['model_table_pass'])
            self.assertEqual(step['body_target'], {n: contract['joint_targets'][n] for n in NAMES})
            for sample in step['endpoint_path']:
                f = sample['fraction']
                pose = {n: contract['measured_origin'][n] + f * (
                    contract['joint_targets'][n] - contract['measured_origin'][n]) for n in ALL}
                np.testing.assert_allclose(sample['position_base_m'], self.tip(pose)[:3, 3])
        json.dumps(plan, allow_nan=False)

    def test_held_goal_compensation_and_no_jaw_commands_preserved(self):
        held = {**self.raw, 'wrist_flex': self.raw['wrist_flex'] - 6}
        T = self.tip()
        plan = plan_surface_approach(self.model, self.raw, held,
            surface_base_m=T[:3, 3] + [0, 0, .045], target_uncertainty_m=.002,
            tcp=self.tcp, evidence=self.evidence)
        for step in plan['segments']:
            c = step['contract']
            self.assertEqual(c['physical_goal_targets']['wrist_flex'], c['joint_targets']['wrist_flex'] - 6)
            self.assertEqual(c['joint_targets']['gripper'], self.raw['gripper'])
            self.assertFalse(c['gripper_command_included'])

    def test_table_violation_cannot_be_hidden_by_more_ik_waypoints(self):
        with patch.object(self.model, 'screen_segment', return_value={'model_table_pass': False}):
            with self.assertRaisesRegex(ValueError, '桌面'):
                self.surface()

    def test_post_close_lift_is_conditional_and_requires_updated_jaw_tcp(self):
        T = self.tip()
        plan = plan_grasp_lift(self.model, self.raw, self.raw,
            target_base_m=T[:3, 3] + T[:3, 2] * .006, approach_axis_base=T[:3, 2],
            target_uncertainty_m=.001, available_clearance_m=.02,
            pregrasp_distance_m=.004, lift_distance_m=.01,
            tcp=self.tcp, evidence=self.evidence)
        kinds = [s['kind'] for s in plan['segments']]
        self.assertEqual(kinds.count('close'), 1)
        self.assertEqual(kinds[kinds.index('close') + 1], 'verify_lift')
        self.assertTrue(plan['diagnostics']['post_contact_replan_required'])
        for step in plan['segments'][kinds.index('close') + 1:]:
            self.assertIn('updated_tcp', step['requires_checkpoint'])
        self.assertFalse(plan['jaw_command_included'])

    def test_uncertain_grasp_budget_is_not_accepted_by_a_verified_label(self):
        T = self.tip()
        plan = plan_grasp_lift(self.model, self.raw, self.raw,
            target_base_m=T[:3, 3] + T[:3, 2] * .006, approach_axis_base=T[:3, 2],
            target_uncertainty_m=.012, available_clearance_m=.01,
            pregrasp_distance_m=.004, lift_distance_m=.01,
            tcp=self.tcp, evidence=self.evidence)
        self.assertIn('grasp_clearance_smaller_than_position_budget', plan['execution_blockers'])
        self.assertFalse(plan['ready_for_external_assessment'])

    def test_target_tcp_and_calibration_evidence_are_fingerprinted(self):
        first = self.surface(tcp=self.tcp, evidence=self.evidence)
        second = self.surface(tcp=self.tcp,
            evidence={**self.evidence, 'calibration_revision': 'changed-camera'})
        self.assertNotEqual(first['input_sha256'], second['input_sha256'])
        self.assertNotEqual(first['plan_sha256'], second['plan_sha256'])
        mismatch = self.surface(tcp=self.tcp,
            evidence={**self.evidence, 'geometry_revision': 'different-model'})
        self.assertIn('tcp_geometry_revision_mismatch', mismatch['execution_blockers'])

    def test_wrong_jaw_opening_is_rejected_before_numeric_planning(self):
        with self.assertRaisesRegex(ValueError, 'measured jaw opening'):
            self.surface(tcp=self.profile.at(1858))

    def test_prior_jaw_uncertainty_stays_unknown(self):
        profile = replace(self.profile, status='prior', position_uncertainty_m=None)
        plan = self.surface(tcp=profile.at(1857), evidence=self.evidence)
        self.assertIsNone(plan['binding']['tcp']['position_uncertainty_m'])
        self.assertIn('tcp_uncertainty_unknown', plan['execution_blockers'])

    def test_d435_input_and_invalid_numbers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'wrist'):
            self.surface(evidence={**self.evidence, 'source': 'd435'})
        for bad in ([float('nan'), 0, 0], [1, 2], [1, 2, float('inf')]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                solve_task_pose(self.model, self.raw, target_base_m=bad,
                                approach_axis_base=[0, 0, 1])
        with self.assertRaisesRegex(ValueError, 'cannot be zero'):
            solve_task_pose(self.model, self.raw, target_base_m=[0, 0, .1],
                            approach_axis_base=[0, 0, 0])
        with self.assertRaises(ValueError):
            self.surface(standoff_m=-.1)

    def test_already_close_surface_reports_next_contact_step_without_moving(self):
        T = self.model.tip(self.raw)
        with self.assertRaises(TaskSpaceError) as failure:
            plan_surface_approach(self.model, self.raw, self.raw,
                surface_base_m=T[:3, 3] + [0, 0, .02], target_uncertainty_m=.002)
        self.assertEqual(failure.exception.code, 'already_at_standoff')

    def lift_context(self, raw):
        return {'state': 'contact_candidate', 'obstruction_candidate': True,
                'target_between_inner_pads': True, 'pwm_owner_active': True,
                'measured_jaw_raw': raw['gripper'], 'frame_id': self.evidence['frame_id'],
                'camera_generation': self.evidence['camera_generation']}

    def test_lift_replans_from_measured_closed_jaw_without_repeating_approach(self):
        raw = {**self.raw, 'gripper': 1790}
        tcp = self.profile.at(raw['gripper'])
        held = {**raw, 'gripper': 1778, 'wrist_flex': raw['wrist_flex'] - 4}
        plan = plan_lift_from_contact(self.model, raw, held, tcp=tcp,
            evidence=self.evidence, contact_context=self.lift_context(raw))
        self.assertTrue(plan['ready_for_external_assessment'])
        self.assertEqual(len(plan['segments']), 1)
        self.assertEqual(plan['segments'][0]['kind'], 'verify_lift')
        self.assertEqual(plan['binding']['tcp']['jaw_raw'], raw['gripper'])
        expected = self.tip(raw, tcp)[:3, 3]
        np.testing.assert_allclose(plan['binding']['target']['endpoint_origin_base_m'], expected)
        np.testing.assert_allclose(plan['binding']['target']['endpoint_target_base_m'], expected + [0, 0, .01])
        self.assertEqual(plan['segments'][0]['contract']['held_goals']['gripper'], 1778)
        self.assertEqual(plan['segments'][0]['contract']['joint_targets']['gripper'], 1790)
        self.assertFalse(plan['diagnostics']['post_contact_replan_required'])
        self.assertFalse(plan['diagnostics']['contact_is_grasp_proof'])
        self.assertTrue(plan['diagnostics']['requires_new_external_assessment'])
        delta = plan['diagnostics']['fixed_pad_delta_mm']
        self.assertTrue(2 <= delta[2] <= 15)
        self.assertLessEqual(np.linalg.norm(delta[:2]), 8)

    def test_lift_does_not_reuse_preclose_tcp(self):
        closed = {**self.raw, 'gripper': 1790}
        with self.assertRaisesRegex(ValueError, 'measured jaw opening'):
            plan_lift_from_contact(self.model, closed, closed, tcp=self.tcp,
                evidence=self.evidence, contact_context=self.lift_context(closed))

    def test_lift_requires_visual_contact_and_low_power_owner_at_current_frame(self):
        bad_cases = [None, {}, {**self.lift_context(self.raw), 'state': 'idle'},
            {**self.lift_context(self.raw), 'obstruction_candidate': False},
            {**self.lift_context(self.raw), 'target_between_inner_pads': False},
            {**self.lift_context(self.raw), 'pwm_owner_active': False},
            {**self.lift_context(self.raw), 'frame_id': 'old-frame'},
            {**self.lift_context(self.raw), 'camera_generation': 'other-camera'},
            {**self.lift_context(self.raw), 'measured_jaw_raw': 1858}]
        for contact in bad_cases:
            with self.subTest(contact=contact), self.assertRaises(TaskSpaceError) as failure:
                plan_lift_from_contact(self.model, self.raw, self.raw, tcp=self.tcp,
                    evidence=self.evidence, contact_context=contact)
            self.assertEqual(failure.exception.code, 'contact_checkpoint_required')

    def test_lift_cannot_expand_legacy_verification_height(self):
        for height in (.001, .02):
            with self.subTest(height=height), self.assertRaises(ValueError):
                plan_lift_from_contact(self.model, self.raw, self.raw, tcp=self.tcp,
                    evidence=self.evidence, contact_context=self.lift_context(self.raw),
                    lift_distance_m=height)


class TaskSpaceHeldLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()
        cls.raw = dict(zip(ALL, [2314, 1016, 3042, 2141, 765, 1859]))
        cls.held = {**cls.raw, 'shoulder_pan': 2313, 'shoulder_lift': 1017,
                    'elbow_flex': 3041, 'wrist_flex': 2137}
        cls.target = {**cls.raw, 'shoulder_pan': 2334, 'shoulder_lift': 1006,
                      'wrist_flex': 2121, 'wrist_roll': 773}

    def path(self, *, held=None):
        T = self.model.tip(self.target)
        return _path(self.model, self.raw, held or self.held, T[:3, 3], T[:3, 1],
            _tcp(None, self.raw), self.raw, tolerance=.002, axis_tolerance=3.,
            max_path_deviation_m=.003, phase='approach')

    def test_four_axis_solution_holds_actual_elbow_near_onboard_limit(self):
        T = self.model.tip(self.target)
        result = solve_task_pose(self.model, self.raw, target_base_m=T[:3, 3],
                                 approach_axis_base=T[:3, 1])
        self.assertEqual(result['joint_targets'], self.target)
        self.assertEqual(result['diagnostics']['held_limit_joints'], ['elbow_flex'])
        self.assertNotIn('elbow_flex', result['diagnostics']['optimized_joints'])
        self.assertFalse(result['collision_screened'])

    def test_whole_path_preserves_fixed_hold_and_full_mesh_error_box(self):
        with patch.object(self.model, 'screen_segment', wraps=self.model.screen_segment) as screen:
            segments, goal, physical, _ = self.path()
        self.assertEqual(len(segments), 1)
        self.assertEqual(goal, self.target)
        self.assertEqual(physical['elbow_flex'], self.held['elbow_flex'])
        contract = segments[0]['contract']
        _validate_contract(contract)
        self.assertNotIn('elbow_flex', contract['active_joints'])
        self.assertEqual(contract['measured_bounds']['elbow_flex'], [3038, 3045])
        self.assertEqual(contract['geometry_bounds']['elbow_flex'], [3038, 3046])
        self.assertFalse(contract['stationary_goal_resends'])
        self.assertTrue(contract['clearance']['model_table_pass'])
        self.assertEqual(screen.call_args.kwargs['uncertainty_ticks']['elbow_flex'], 4)
        self.assertIn('elbow_flex', screen.call_args.kwargs['stationary_limit_joints'])
        for n in contract['active_joints']:
            limits = self.model.calibration[n]
            error = contract['uncertainty_ticks'][n]
            for pose in (self.raw, goal):
                self.assertGreaterEqual(pose[n] - error, limits['range_min'] + 12)
                self.assertLessEqual(pose[n] + error, limits['range_max'] - 12)
            self.assertEqual(physical[n] - goal[n], self.held[n] - self.raw[n])

    def test_infeasible_fixed_elbow_goal_is_not_a_current_pose_success(self):
        T = self.model.tip(self.raw)
        with self.assertRaises(TaskSpaceError) as failure:
            solve_task_pose(self.model, self.raw, target_base_m=[2., 2., 2.],
                             approach_axis_base=T[:3, 1])
        self.assertEqual(failure.exception.code, 'unreachable_task_pose')
        self.assertEqual(failure.exception.diagnostics['held_limit_joints'], ['elbow_flex'])
        self.assertGreater(failure.exception.diagnostics['position_error_m'], 1.)

    def test_hard_limit_violation_is_not_laundered_as_a_stationary_joint(self):
        outside = self.model.calibration['elbow_flex']['range_max'] + 1
        T = self.model.tip(self.raw)
        with self.assertRaises(TaskSpaceError) as failure:
            solve_task_pose(self.model, {**self.raw, 'elbow_flex': outside},
                             target_base_m=T[:3, 3], approach_axis_base=T[:3, 1])
        self.assertEqual(failure.exception.code, 'joint_limits')
        with self.assertRaises(TaskSpaceError) as failure:
            self.path(held={**self.held, 'elbow_flex': outside})
        self.assertEqual(failure.exception.code, 'joint_limits')

    def test_table_collision_still_rejects_the_complete_fixed_elbow_path(self):
        with patch.object(self.model, 'screen_segment', return_value={'model_table_pass': False}):
            with self.assertRaisesRegex(ValueError, '桌面'):
                self.path()


if __name__ == '__main__':
    unittest.main()
