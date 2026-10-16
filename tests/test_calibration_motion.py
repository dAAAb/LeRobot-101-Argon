"""Actual-CAD engineering viewpoint tests; dispatch is mocked, no hardware."""
from copy import deepcopy
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from server import calibration_motion as cm
from server.gripper_power import SessionCancelled
from server.local_cartesian import ALL, NAMES, default_model
from server.motion_program import _validate_contract, execute_prepared_segment


class CalibrationMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()
        cls.raw = dict(zip(ALL, [2294, 938, 3038, 2192, 747, 1859]))
        grasp = SimpleNamespace(model=cls.model, state='idle',
            robot=SimpleNamespace(goals=dict(cls.raw), pwm_owner=None))
        frame = {'source': 'wrist', 'id': 'calibration-frame', 'path': '/tmp/offline-wrist.jpg',
            'camera_generation': 'camera-one', 'capture_monotonic': time.monotonic(), 'joints': cls.raw}
        cls.original = cm.build_calibration_candidates(grasp, frame, origin=cls.raw,
                                                       session_id='test-session')

    def setUp(self):
        self.batch = deepcopy(self.original)
        self.events = []
        self.grasp = SimpleNamespace(model=self.model, state='idle', record=self.events.append,
            robot=SimpleNamespace(goals=dict(self.raw), pwm_owner=None))
        self.cancel = threading.Event()
        self.reference = {**self.batch['reference'], 'id': 'fresh-dispatch',
                          'capture_monotonic': time.monotonic()}
        self.candidate = self.batch['candidates'][0]
        self.approval = {'approved': True, 'batch_sha256': self.batch['contract_sha256'],
            'candidate_sha256': self.candidate['contract_sha256'],
            'frame_id': self.batch['reference']['id'], 'gripper_empty': True,
            'human_clear': True, 'clear_path': True, 'reason': '已檢查整條工程路徑。'}

    def execute(self, **kwargs):
        return cm.execute_calibration_candidate(self.grasp, self.batch, self.candidate['id'],
            self.reference, self.cancel, lambda _: None,
            session_id=kwargs.pop('session_id', 'test-session'),
            approval=kwargs.pop('approval', self.approval), **kwargs)

    def test_current_near_limit_pose_has_fifteen_screened_choices(self):
        self.assertEqual(len(self.batch['candidates']), 15)
        # A wider measured tube can reject the 40 mm proposal while allowing
        # the shorter screened fallback. Never keep the old narrower sweep.
        for rejected in self.batch['rejections']:
            self.assertEqual(rejected['requested_translation_mm'], 40.)
            self.assertIn('位置誤差範圍', rejected['reason'])
        self.assertFalse(self.batch['calibration_samples_captured'])
        for candidate in self.batch['candidates']:
            c = candidate['segment']
            _validate_contract(c)
            self.assertTrue(c['clearance']['model_table_pass'])
            self.assertEqual(c['endpoint_tolerance_ticks'], 16)
            self.assertEqual(c['stationary_tolerance_ticks'], 12)
            self.assertEqual(c['joint_targets']['gripper'], self.raw['gripper'])
            self.assertEqual(c['fixed_origin'], self.raw)
            self.assertEqual(candidate['origin'], self.raw)
            self.assertFalse(c['gripper_command_included'])
            for n in c['active_joints']:
                self.assertEqual(c['actual_tracking_envelope_ticks'][n], 16)
                self.assertGreaterEqual(c['uncertainty_ticks'][n], 16)
                self.assertLessEqual(c['geometry_bounds'][n][0],
                    min(c['measured_origin'][n], c['joint_targets'][n]) - 16)
                self.assertGreaterEqual(c['geometry_bounds'][n][1],
                    max(c['measured_origin'][n], c['joint_targets'][n]) + 16)

    def _execute_with_measured_error(self, endpoint_error):
        """Real CAD and executor; only the motor and image source are fakes."""
        segment = self.candidate['segment']
        r = self.grasp.robot
        r.position = dict(segment['measured_origin'])
        r.bias = {n: r.goals[n] - r.position[n] for n in NAMES}
        r.camera = SimpleNamespace(generation='camera-one', fresh=lambda: True)
        r.io, r.stop = threading.RLock(), threading.Event()
        r.begin = lambda **kwargs: None
        r.hold = lambda: None
        def capture():
            return {'source': 'wrist', 'camera_generation': 'camera-one',
                    'joints': dict(r.position), 'capture_monotonic': time.monotonic()}
        self.grasp.capture = capture
        def glide(targets, **kwargs):
            self.assertEqual(kwargs['precision'], 16)
            kwargs['guard'](r.position)
            r.goals.update({n: v + r.bias[n] for n, v in targets.items()})
            r.position.update({n: v + endpoint_error.get(n, 0) for n, v in targets.items()})
            kwargs['measured_guard'](r.position)
            kwargs['guard'](r.position)
        r.glide = glide
        with patch('server.grasp.unchanged_wrist', return_value={'unchanged': True}):
            return execute_prepared_segment(self.grasp, segment, capture(), self.cancel, lambda _: None)

    def test_settled_thirteen_tick_miss_keeps_actual_pose_for_calibration(self):
        result = self._execute_with_measured_error({'wrist_flex': -13})
        self.assertEqual(result['completion'], 'within_declared_endpoint_tolerance')
        self.assertEqual(result['after']['joints']['wrist_flex'],
                         self.candidate['segment']['joint_targets']['wrist_flex'] - 13)
        self.assertEqual(result['endpoint_error_ticks']['wrist_flex'], -13)
        self.assertFalse(self.grasp.robot.stop.is_set())

    def test_outside_declared_calibration_tube_still_stops(self):
        with self.assertRaises(RuntimeError):
            self._execute_with_measured_error({'wrist_flex': -17})
        self.assertTrue(self.grasp.robot.stop.is_set())

    def test_two_rotation_axes_are_offered_even_with_metric_translation(self):
        rotations = [c for c in self.batch['candidates'] if 'rotation_joint' in c['descriptor']]
        self.assertEqual({c['descriptor']['rotation_joint'] for c in rotations}, {'wrist_roll', 'wrist_flex'})
        for axis in ('wrist_roll', 'wrist_flex'):
            positive = [c for c in rotations if c['descriptor']['rotation_joint'] == axis
                        and c['descriptor']['nominal_rotation_degrees'] >= 12]
            negative = [c for c in rotations if c['descriptor']['rotation_joint'] == axis
                        and c['descriptor']['nominal_rotation_degrees'] <= -12]
            self.assertTrue(positive and negative)

    def test_rotations_only_skips_translation_solver_before_planning(self):
        with patch.object(cm.observation, '_translation_target') as translation:
            batch = cm.build_calibration_candidates(self.grasp, self.reference,
                origin=self.raw, session_id='test-session', rotations_only=True)
            translation.assert_not_called()
        self.assertEqual(len(batch['candidates']), 8)
        self.assertEqual({c['descriptor']['rotation_joint'] for c in batch['candidates']},
                         {'wrist_roll', 'wrist_flex'})
        self.assertTrue(cm.describe_calibration_candidates(batch)['rotations_only'])
        # Selection changes only the candidates requested, not their geometry.
        originals = {c['id']: c for c in self.batch['candidates']}
        for candidate in batch['candidates']:
            self.assertEqual(candidate['segment'], originals[candidate['id']]['segment'])
            self.assertEqual(candidate['descriptor'], originals[candidate['id']]['descriptor'])

    def test_rotations_only_requires_boolean(self):
        for value in (1, 'true', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                cm.build_calibration_candidates(self.grasp, self.reference,
                    origin=self.raw, session_id='test-session', rotations_only=value)

    def test_inward_elbow_alternatives_share_one_origin_and_full_tracking_tube(self):
        choices=[c for c in self.batch['candidates'] if c['id'].startswith('calibration_elbow_inward_')]
        self.assertEqual(len(choices),3)
        for ticks,candidate in zip((42,84,126),choices):
            segment=candidate['segment'];descriptor=candidate['descriptor']
            self.assertEqual(segment['measured_origin'],self.raw)
            self.assertEqual(candidate['origin'],self.raw)
            self.assertEqual(segment['joint_targets']['elbow_flex'],self.raw['elbow_flex']-ticks)
            self.assertEqual(segment['active_joints'],['elbow_flex'])
            self.assertEqual(segment['inward_limit_joints'],['elbow_flex'])
            self.assertEqual(segment['actual_tracking_envelope_ticks']['elbow_flex'],16)
            self.assertEqual(segment['geometry_bounds']['elbow_flex'][1],self.raw['elbow_flex']+16)
            self.assertEqual(segment['measured_bounds']['elbow_flex'][1],3045)
            self.assertTrue(segment['clearance']['model_table_pass'])
            self.assertLessEqual(segment['nominal_tip_path_mm'],80.)
            self.assertEqual(descriptor['attempt_limits'],{
                'maximum_body_ticks_from_origin':240,'maximum_tip_distance_from_origin_mm':120.})
            self.assertGreater(descriptor['camera_change_model']['baseline_mm'],10.)
            self.assertGreater(descriptor['camera_change_model']['rotation_degrees'],3.)
            self.assertEqual(descriptor['requested_inward_ticks'],ticks)
            self.assertIn('forearm',descriptor['viewpoint_scope'])

    def test_interior_elbow_does_not_offer_a_false_limit_departure(self):
        current={**self.raw,'elbow_flex':3000}
        self.grasp.robot.goals=dict(current)
        with patch.object(cm.observation,'_translation_target',side_effect=ValueError('skip unrelated IK')):
            batch=cm.build_calibration_candidates(self.grasp,{**self.reference,'joints':current},
                origin=self.raw,session_id='test-session')
        self.assertFalse(any(c['id'].startswith('calibration_elbow_inward_') for c in batch['candidates']))
        self.assertFalse(any('requested_inward_ticks' in r for r in batch['rejections']))

    def test_inward_candidate_uses_same_exact_operator_approval_contract(self):
        self.candidate=next(c for c in self.batch['candidates'] if c['id']=='calibration_elbow_inward_42')
        self.approval['candidate_sha256']=self.candidate['contract_sha256']
        with patch.object(cm.observation,'execute_observation',return_value={'completed':True}) as execute:
            self.execute()
        forwarded=execute.call_args.args[1]
        self.assertIs(forwarded,self.candidate)
        self.assertEqual(forwarded['segment']['inward_limit_joints'],['elbow_flex'])

    def test_translation_holds_near_limit_elbow_and_does_not_advance_optically(self):
        translations = self.batch['candidates'][:4]
        for c in translations:
            self.assertEqual(c['segment']['joint_targets']['elbow_flex'], self.raw['elbow_flex'])
            self.assertIn('elbow_flex', c['segment']['stationary_limit_joints'])
            self.assertLessEqual(c['descriptor']['maximum_nominal_optical_forward_advance_mm'], 2.)
            self.assertGreaterEqual(c['descriptor']['camera_change_model']['baseline_mm'], 15.)

    def test_descriptor_is_a_copy_with_binding_hashes(self):
        public = cm.describe_calibration_candidates(self.batch)
        self.assertEqual(public['batch_sha256'], self.batch['contract_sha256'])
        self.assertEqual(public['frame_id'], 'calibration-frame')
        public['candidates'][0]['target_joints']['wrist_roll'] = 0
        self.assertNotEqual(self.batch['candidates'][0]['segment']['joint_targets']['wrist_roll'], 0)

    def test_explicit_review_delegates_exact_candidate_to_existing_executor(self):
        with patch.object(cm.observation, 'execute_observation',
                          return_value={'completed': True, 'after': self.reference}) as executor:
            result = self.execute()
        self.assertIs(executor.call_args.args[1], self.candidate)
        self.assertEqual(executor.call_args.args[2], self.reference)
        self.assertEqual(executor.call_args.kwargs['verdict']['candidate_id'], self.candidate['id'])
        self.assertIs(executor.call_args.kwargs['engineering_calibration'], True)
        self.assertTrue(result['completed'])
        self.assertFalse(result['calibration_sample_recorded'])
        self.assertEqual(self.grasp.robot.goals, self.raw)
        self.assertIsNone(self.grasp.robot.pwm_owner)

    def test_missing_clearance_or_wrong_frame_cannot_dispatch(self):
        changes = ({'approved': False}, {'clear_path': False}, {'gripper_empty': False},
                   {'human_clear': False}, {'frame_id': 'different'},
                   {'candidate_sha256': 'other'}, {'batch_sha256': 'other'}, {'reason': ''})
        with patch.object(cm.observation, 'execute_observation') as executor:
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.execute(approval={**self.approval, **change})
            executor.assert_not_called()

    def test_changed_segment_or_session_or_goals_rejects(self):
        with self.assertRaisesRegex(ValueError, '校準工作'):
            self.execute(session_id='different-session')
        self.grasp.robot.goals['wrist_roll'] += 1
        with self.assertRaisesRegex(ValueError, '保持目標'):
            self.execute()
        self.grasp.robot.goals['wrist_roll'] -= 1
        self.candidate['segment']['joint_targets']['wrist_roll'] += 1
        with self.assertRaisesRegex(ValueError, '候選內容'):
            self.execute()

    def test_expired_candidate_and_model_revision_cannot_dispatch(self):
        with patch.object(cm.time, 'monotonic', return_value=self.batch['expires_monotonic'] + 1):
            with self.assertRaisesRegex(ValueError, '過期'):
                self.execute()
        with patch.object(cm, 'geometry_revision', return_value='changed'):
            with self.assertRaisesRegex(ValueError, '幾何或馬達校準'):
                self.execute()

    def test_original_session_origin_is_not_reset_to_latest_pose(self):
        far = {**self.raw, 'wrist_roll': self.raw['wrist_roll'] + 250}
        with self.assertRaisesRegex(RuntimeError, '開始姿勢'):
            cm.build_calibration_candidates(self.grasp, {**self.reference, 'joints': far},
                                             origin=self.raw, session_id='test-session')

    def test_post_retreat_pose_can_reframe_with_same_fixed_origin(self):
        # The real second measured pose is 53.17 mm from the initial pose.
        # Its full tracking reserve exhausts the object-observation 80 mm
        # budget before moving. A bounded calibration workspace is distinct.
        current = dict(zip(ALL, [2435, 979, 3040, 2031, 658, 1859]))
        self.grasp.robot.goals = dict(current)
        batch = cm.build_calibration_candidates(self.grasp,
            {**self.reference, 'joints': current, 'capture_monotonic': time.monotonic()},
            origin=self.raw, session_id='test-session')
        candidate = next(c for c in batch['candidates']
                         if c['id'] == 'calibration_wrist_flex_positive_12')
        self.assertEqual(candidate['origin'], self.raw)
        self.assertEqual(candidate['segment']['fixed_origin'], self.raw)
        self.assertEqual(candidate['descriptor']['attempt_limits'], {
            'maximum_body_ticks_from_origin': 240,
            'maximum_tip_distance_from_origin_mm': 120.})
        with self.assertRaisesRegex(ValueError, '位置誤差範圍'):
            cm.observation._compact_segment(self.model, candidate['segment'], self.raw)
        with self.assertRaisesRegex(ValueError, '工程校準'):
            cm.observation._execution_excursion_profile(candidate, False)
        self.assertEqual(cm.observation._execution_excursion_profile(candidate, True),
                         cm.EXCURSION_PROFILE)
        altered = deepcopy(candidate)
        altered['descriptor']['attempt_limits']['maximum_tip_distance_from_origin_mm'] = 121.
        with self.assertRaisesRegex(ValueError, '工作範圍'):
            cm.observation._execution_excursion_profile(altered, True)

    def test_runtime_origin_bounds_remain_scoped_and_bounded(self):
        # A simple metric model exercises the runtime origin guard exactly;
        # table/segment validation is independently covered with the real CAD.
        def tip(pose):
            transform = np.eye(4)
            transform[0, 3] = (pose['wrist_flex'] - self.raw['wrist_flex']) * .001
            return transform
        model = SimpleNamespace(tip=tip)
        inside = {**self.raw, 'wrist_flex': self.raw['wrist_flex'] + 100}
        with self.assertRaisesRegex(RuntimeError, '移動範圍'):
            cm.observation._outer_guard(model, self.raw, inside)
        cm.observation._outer_guard(model, self.raw, inside, excursion_profile=cm.EXCURSION_PROFILE)
        outside = {**inside, 'wrist_flex': self.raw['wrist_flex'] + 121}
        with self.assertRaisesRegex(RuntimeError, '移動範圍'):
            cm.observation._outer_guard(model, self.raw, outside, excursion_profile=cm.EXCURSION_PROFILE)
        outside = {**inside, 'wrist_roll': self.raw['wrist_roll'] + 241}
        with self.assertRaisesRegex(RuntimeError, '關節範圍'):
            cm.observation._outer_guard(model, self.raw, outside, excursion_profile=cm.EXCURSION_PROFILE)

    def test_no_camera_or_owned_jaw_fallback(self):
        with self.assertRaisesRegex(ValueError, '腕部相機'):
            cm.build_calibration_candidates(self.grasp, {**self.reference, 'source': 'd435'},
                                             origin=self.raw, session_id='test-session')
        self.grasp.robot.pwm_owner = 'holding'
        with self.assertRaisesRegex(ValueError, '夾爪目前的工作'):
            cm.build_calibration_candidates(self.grasp, self.reference, origin=self.raw,
                                             session_id='test-session')

    def test_cancel_does_not_dispatch(self):
        self.cancel.set()
        with patch.object(cm.observation, 'execute_observation') as executor:
            with self.assertRaises(SessionCancelled):
                self.execute()
            executor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
