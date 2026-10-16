"""Exercise real move orchestration with synthetic vertical geometry and motors.

This is controller contract evidence only, never physical clearance or metrology.
The shared fixture has no serial, network, app startup, or real camera access.
"""
import copy
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

import test_measured_step_anchor as helpers
from server import grasp as gm
from server.gripper_power import SessionCancelled


class VerticalRobot(helpers.FakeRobot):
    def __init__(self):
        super().__init__()
        self.pwm_owner = None
        self.frame_hook = None
        self.glide_options = []

    def frame(self):
        frame = super().frame()
        if self.frame_hook:
            self.frame_hook(self, frame)
        frame['joints'] = dict(self.actual)
        return frame

    def glide(self, logical, **kwargs):
        self.glide_options.append(kwargs)
        return super().glide(logical, **kwargs)


class VerticalModel(helpers.FakeModel):
    def __init__(self, robot):
        super().__init__(robot)
        self.roll_target_ticks = 0

    def tip(self, raw):
        pose = np.eye(4)
        pose[2, 3] = (raw['shoulder_pan'] - helpers.ORIGIN['shoulder_pan']) / 3000
        pose[:3, :3] = Rotation.from_euler(
            'z', (raw['wrist_roll'] - helpers.ORIGIN['wrist_roll']) * .1,
            degrees=True,
        ).as_matrix()
        return pose

    def propose_adaptive(self, raw, delta, *, held_goals, uncertainty_ticks):
        # A 3 mm request rounds to a 3.333 mm synthetic IK target. This exercises
        # a useful 2 mm partial move whose old target error is still >1 mm.
        ticks = {name: 0 for name in gm.NAMES}
        ticks['shoulder_pan'] = round(delta[2] * 3000) + 1
        ticks['wrist_roll'] = self.roll_target_ticks
        target = {**raw, **{name: raw[name] + ticks[name] for name in gm.NAMES}}
        proposal = {
            'requested_delta_m': delta.tolist(),
            'joint_targets': target,
            'physical_goal_targets': {name: held_goals[name] + ticks[name] for name in gm.NAMES},
        }
        self.proposals.append(copy.deepcopy(proposal))
        return proposal


class WaveLiftMoveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.robot = VerticalRobot()
        self.model = VerticalModel(self.robot)
        self.grasp = gm.WristGrasp(self.robot, object(), self.robot.frame, self.tmp.name)
        self.grasp._model = self.model
        # This fixture tests motor/capture orchestration. Real CAD coordination
        # path screening is covered by the wave planner/raise model tests.
        self.grasp._wave_lift_proposal = lambda distance,raw,origin=None:self.grasp._translation_proposal('up',distance,raw)
        self.grasp.vector = lambda direction, distance, raw: np.array([0., 0., distance / 1000])
        self.events = []
        self.grasp.record = lambda event: self.events.append(copy.deepcopy(event))
        images = patch.object(gm.cv2, 'imread', return_value=np.zeros((24, 24), np.uint8))
        images.start()
        self.addCleanup(images.stop)

    def move(self, **kwargs):
        return self.grasp.move(
            kwargs.pop('direction', 'up'), 3, self.robot.frame(), lambda message: None,
            free_space_lift=kwargs.pop('free_space_lift', True), **kwargs,
        )

    def assert_no_result(self):
        self.assertFalse(any(event['type'] == 'movement_result' for event in self.events))

    def test_opt_in_rejects_nonidle_nonup_and_owned_jaw_before_begin(self):
        for state, direction, owner in [
            ('approaching', 'up', None), ('holding', 'up', None),
            ('holding_unverified', 'up', None), ('idle', 'down', None),
            ('idle', 'left', None), ('idle', 'up', 'power-owner'),
        ]:
            with self.subTest(state=state, direction=direction, owner=owner):
                self.grasp.state = state
                self.robot.pwm_owner = owner
                with patch.object(self.robot, 'begin') as begin:
                    with self.assertRaisesRegex(RuntimeError, '空手待機'):
                        self.move(direction=direction)
                    begin.assert_not_called()
                self.assertEqual(self.robot.glide_calls, [])
                self.assert_no_result()

    def test_partial_lift_reports_measured_progress_and_requires_stable_endpoint(self):
        self.robot.accepted_residuals = [{'shoulder_pan': -4}]
        result = self.move(lift_origin=dict(self.robot.actual))
        self.assertAlmostEqual(result['actual_delta_model_mm'][2], 2.)
        self.assertEqual(result['requested_mm'], 3)
        self.assertEqual(result['completion'], 'partial_progress')
        self.assertFalse(result['exact_target_reached'])
        self.assertEqual(result['purpose'], 'wave_raise')
        self.assertEqual(self.robot.glide_options[0]['endpoint_stability_ticks'], 2)
        self.assertFalse(self.robot.glide_options[0]['stall_response_wait'])
        self.assertEqual(self.robot.hold_calls, 0)
        self.assertNotIn('grasp_confirmed', result)
        self.assertNotIn('wave_completed', result)

    def test_assessed_lift_corridor_accepts_settled_eight_mm_without_claiming_exact_or_wave(self):
        # Represent a separately assessed 12 mm corridor. This fixture exercises
        # move's endpoint/receipt contract, not the visual clearance decision.
        original = self.grasp._wave_lift_proposal

        def assessed_proposal(distance, raw, origin=None):
            proposal, envelope = original(distance, raw, origin)
            proposal['coordination'] = {
                'intermediate_delta_m': [0., 0., .010],
                'intermediate_orientation_deg': 0.,
                'model_table_pass': True,
                'actual_tip_corridor_mm': {
                    'minimum_up': -1., 'maximum_up': 12.,
                    'maximum_lateral': 3., 'maximum_rotation_degrees': 3.,
                },
            }
            return proposal, envelope

        self.grasp._wave_lift_proposal = assessed_proposal
        self.robot.accepted_residuals = [{'shoulder_pan': 8}]
        origin = dict(self.robot.actual)
        result = self.grasp.move(
            'up', 5, self.robot.frame(), lambda message: None,
            free_space_lift=True, lift_origin=origin,
        )
        # Nominal target is 16 ticks; 8 residual ticks give 24/3 = 8 mm,
        # inside both raw and measured bounds checked by the fake driver.
        self.assertEqual(self.robot.glide_calls[0]['logical']['shoulder_pan'] - origin['shoulder_pan'], 16)
        self.assertAlmostEqual(result['actual_delta_model_mm'][2], 8.)
        self.assertEqual(result['requested_mm'], 5)
        self.assertEqual(result['accepted_mm'], 5)
        self.assertEqual(result['completion'], 'partial_progress')
        self.assertFalse(result['exact_target_reached'])
        self.assertEqual(result['purpose'], 'wave_raise')
        self.assertEqual(self.robot.glide_options[0]['endpoint_stability_ticks'], 2)
        self.assertEqual(self.robot.hold_calls, 0)
        self.assertFalse(self.robot.stop.is_set())
        self.assertNotIn('grasp_confirmed', result)
        self.assertNotIn('wave_completed', result)
        self.assertEqual(len([e for e in self.events if e['type'] == 'movement_result']), 1)

    def test_exact_flag_still_requires_original_target_orientation(self):
        self.model.roll_target_ticks = 15  # Proposed target is rotated 1.5 degrees.
        self.robot.accepted_residuals = [{'wrist_roll': -15}]
        result = self.move()
        self.assertAlmostEqual(result['actual_delta_model_mm'][2], 10 / 3)
        self.assertFalse(result['exact_target_reached'])
        self.assertEqual(result['completion'], 'partial_progress')

    def test_exact_assessed_proposal_executes_without_running_another_solver(self):
        origin = dict(self.robot.actual)
        proposal, envelope = self.grasp._translation_proposal('up', 3, origin)
        assessed = {
            'origin': origin, 'hold_goals': dict(self.robot.goals),
            'proposal': copy.deepcopy(proposal), 'envelope': dict(envelope),
        }
        with patch.object(self.grasp, '_wave_lift_proposal', side_effect=AssertionError('no new wave solver')):
            with patch.object(self.grasp, '_translation_proposal', side_effect=AssertionError('no new translation solver')):
                result = self.move(assessed_lift=assessed, lift_origin=origin)
        self.assertEqual(len(self.robot.glide_calls), 1)
        self.assertEqual(self.robot.glide_calls[0]['logical'], {
            name: proposal['joint_targets'][name] for name in gm.NAMES
        })
        self.assertEqual(self.robot.glide_calls[0]['physical'], proposal['physical_goal_targets'])
        recorded = [e for e in self.events if e['type'] == 'proposal']
        self.assertEqual(recorded[0]['proposal'], assessed['proposal'])
        self.assertEqual(result['before']['joints'], assessed['origin'])
        self.assertEqual(self.robot.hold_calls, 0)
        self.assertNotIn('wave_completed', result)

    def test_stale_assessed_origin_or_hold_goal_rejects_before_rebase_or_motion(self):
        origin = dict(self.robot.actual)
        held = dict(self.robot.goals)
        bias = dict(self.robot.bias)
        proposal, envelope = self.grasp._translation_proposal('up', 3, origin)
        assessed = {
            'origin': origin, 'hold_goals': held,
            'proposal': proposal, 'envelope': envelope,
        }
        for changed in ('origin', 'hold_goal'):
            with self.subTest(changed=changed):
                self.robot.actual = dict(origin)
                self.robot.goals = dict(held)
                if changed == 'origin':
                    self.robot.actual['shoulder_pan'] += 1
                else:
                    self.robot.goals['wrist_flex'] += 1
                before_goals = dict(self.robot.goals)
                with patch.object(self.grasp, '_wave_lift_proposal', side_effect=AssertionError('must not replan stale approval')):
                    with self.assertRaisesRegex(RuntimeError, '判斷後手臂位置有變'):
                        self.move(assessed_lift=assessed, lift_origin=origin)
                self.assertEqual(self.robot.glide_calls, [])
                self.assertEqual(self.robot.goals, before_goals)
                self.assertEqual(self.robot.bias, bias)
                self.assertEqual(self.robot.hold_calls, 0)
                self.assert_no_result()

    def test_whole_attempt_lift_limit_is_checked_before_a_new_goal(self):
        attempt_origin = dict(self.robot.actual)
        # The current pose is already 59 mm above the attempt's original pose.
        # The locally valid next step must not reset that accumulated origin.
        attempt_origin['shoulder_pan'] -= 177
        with self.assertRaisesRegex(RuntimeError, '這次可確認的範圍'):
            self.move(lift_origin=attempt_origin)
        self.assertEqual(self.robot.glide_calls, [])
        self.assert_no_result()

    def test_original_grasp_verification_does_not_accept_the_partial_endpoint(self):
        self.grasp.state = 'verifying'
        candidate = {**self.robot.actual, 'shoulder_pan': helpers.ORIGIN['shoulder_pan'] + 6}

        def inspect_original_predicate(logical, **kwargs):
            self.assertIsNone(kwargs['endpoint_stability_ticks'])
            self.assertFalse(kwargs['endpoint_accept'](candidate))
            raise RuntimeError('synthetic strict endpoint rejected')

        with patch.object(self.robot, 'glide', side_effect=inspect_original_predicate):
            with self.assertRaisesRegex(RuntimeError, 'strict endpoint rejected'):
                self.move(free_space_lift=False)
        self.assert_no_result()

    def test_postcapture_drift_outside_endpoint_or_joint_corridor_stops_and_holds(self):
        for joint, drift, expected in [
            ('shoulder_pan', 10, '未穩定'),
            ('elbow_flex', 13, 'screened raw interval'),
            ('gripper', 5, '夾爪開口改變'),
        ]:
            with self.subTest(joint=joint):
                # Each case needs a fresh independent plant and origin.
                self.robot = VerticalRobot()
                self.grasp.robot = self.robot
                self.grasp.snapshot = self.robot.frame
                self.model.robot = self.robot
                self.robot.accepted_residuals = [{'shoulder_pan': -4}]

                def drift_after_acknowledgement(robot, frame):
                    if robot.sequence == 4:
                        robot.actual[joint] += drift

                self.robot.frame_hook = drift_after_acknowledgement
                with self.assertRaisesRegex((ValueError, RuntimeError), expected):
                    self.move()
                self.assertTrue(self.robot.stop.is_set())
                self.assertEqual(self.robot.hold_calls, 1)
                self.assert_no_result()

    def test_final_capture_failure_holds_instead_of_returning_prior_acknowledgement(self):
        self.robot.accepted_residuals = [{'shoulder_pan': -4}]

        def camera_failed(robot, frame):
            if robot.sequence == 4:
                raise RuntimeError('synthetic camera disconnected')

        self.robot.frame_hook = camera_failed
        with self.assertRaisesRegex(RuntimeError, 'camera disconnected'):
            self.move()
        self.assertTrue(self.robot.stop.is_set())
        self.assertEqual(self.robot.hold_calls, 1)
        self.assert_no_result()

    def test_cancellation_during_fresh_capture_prevents_any_glide_goal(self):
        cancel = threading.Event()
        self.robot.frame_hook = lambda robot, frame: cancel.set() if robot.sequence == 3 else None
        with self.assertRaises(SessionCancelled):
            self.move(cancel=cancel)
        self.assertEqual(self.robot.glide_calls, [])
        self.assert_no_result()


if __name__ == '__main__':
    unittest.main(verbosity=2)
