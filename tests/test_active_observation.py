"""Finite observation selection/execution using CAD and fake motors only."""
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from pydantic import ValidationError

from server.active_observation import (ObservationVerdict, build_observation_candidates,
                                       execute_observation, filter_observation_framing,
                                       filter_observation_novelty,
                                       METRIC_VIEW_IDS)
from server.grasp import WristGrasp, unchanged_wrist
from server.gripper_power import SessionCancelled
from server.local_cartesian import ALL, NAMES, default_model
from test_motion_program import Robot


class Power:
    def __init__(self):
        self.phase = 'completed'
        self.contact_possible = False
        self.leave_pending = Mock(side_effect=self.pending)
        self.read = Mock(return_value=1860)
        self.write = Mock()
    def _record(self):
        return {'phase': self.phase, 'contact_possible': self.contact_possible}
    def pending(self):
        self.phase = 'blocked'


class ActiveObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'wrist.png'
        cv2.imwrite(str(self.path), np.full((270, 480, 3), 128, dtype=np.uint8))
        self.raw = dict(zip(ALL, [2161, 1135, 2490, 2328, 1100, 1531]))
        self.held = {**self.raw, 'elbow_flex': 2482, 'shoulder_lift': 1128}
        self.robot = Robot(self.raw, self.held)
        self.robot.gripper_power = Power()
        self.robot.read = lambda: dict(self.robot.position)
        self.brain = SimpleNamespace(_infer=Mock(side_effect=AssertionError('no cloud in helper')))
        self.grasp = WristGrasp(self.robot, self.brain, self.capture, self.tmp.name)
        self.grasp._model = self.model; self.grasp.capture = self.capture
        self.grasp.record = Mock()
        self.cancel = threading.Event(); self.report = Mock()
        self.open_hook = None; self.release_hook = None
        self.grasp.open_empty = Mock(side_effect=self.open_empty)
        self.grasp.release = Mock(side_effect=self.release)

    def capture(self):
        return {'source': 'wrist', 'path': str(self.path),
                'camera_generation': self.robot.camera.generation,
                'joints': dict(self.robot.position), 'capture_monotonic': time.monotonic()}

    def candidates(self, **kwargs):
        return build_observation_candidates(self.grasp, self.capture(), cancel=self.cancel, **kwargs)

    def candidate(self, name):
        used = METRIC_VIEW_IDS if name.startswith(('roll_', 'pitch_')) else ()
        return next(c for c in self.candidates(used=used) if c['id'] == name)

    def verdict(self, candidate, **changes):
        return {'candidate_id': candidate['id'], 'gripper_empty': True, 'human_clear': True,
                'clear_path': True, 'confidence': .95, 'reason': '這個角度可以換個視角。', **changes}

    def execute(self, candidate, **kwargs):
        return execute_observation(self.grasp, candidate, self.capture(), self.cancel, self.report,
            verdict=kwargs.pop('verdict', self.verdict(candidate)), **kwargs)

    def open_empty(self, frame, report, *, cancel, guard, opening_contract=None):
        guard(self.robot.position)
        self.robot.pwm_owner = 'empty-session'
        self.robot.gripper_power.phase = 'active'
        if self.open_hook:
            self.open_hook(guard)
        if cancel.is_set():
            raise SessionCancelled('stopped')
        target = 1860 if opening_contract is None else opening_contract['target_raw']
        self.robot.position['gripper'] = self.robot.goals['gripper'] = target
        guard(self.robot.position)
        return self.capture()

    def release(self, frame, report, cancel, *, empty_verified=False, support_verified=False):
        self.assertTrue(empty_verified)
        self.assertFalse(support_verified)
        self.assertFalse(self.robot.gripper_power.contact_possible)
        if self.release_hook:
            self.release_hook()
        self.robot.gripper_power.read(6, 56, 2)
        self.robot.gripper_power.write(6, 48, 500, 2)
        self.robot.pwm_owner = None
        self.robot.gripper_power.phase = 'completed'
        self.grasp.state = 'idle'
        return {'completed': True, 'neutral_goal': self.robot.position['gripper']}

    def test_build_is_pure_and_provides_finite_complete_candidates(self):
        candidates = self.candidates()
        ids = [c['id'] for c in candidates]
        self.assertEqual(set(ids), {'move_up', 'move_backward', 'move_left', 'move_right', 'jaw_open'})
        self.assertEqual(len(ids), len(set(ids)))
        for c in candidates:
            self.assertEqual(c['descriptor']['candidate_id'], c['id'])
            self.assertEqual(c['descriptor']['label'], c['label'])
            self.assertEqual(c['reference']['source'], 'wrist')
            self.assertEqual(c['origin'], self.raw)
            if c['kind'] == 'body':
                s = c['segment']
                self.assertEqual(s['held_goals'], self.held)
                self.assertEqual(s['joint_targets']['gripper'], self.raw['gripper'])
                self.assertTrue(s['clearance']['model_table_pass'])
                self.assertIn('actual_tracking_envelope_ticks', c['descriptor'])
                self.assertEqual(c['descriptor']['endpoint_tolerance_ticks'], 20)
                self.assertEqual(c['descriptor']['stationary_tolerance_ticks'], 12)
                self.assertEqual(s['stationary_tolerance_ticks'], 12)
                for active in s['active_joints']:
                    self.assertEqual(c['descriptor']['actual_tracking_envelope_ticks'][active], 20)
                    self.assertGreaterEqual(c['descriptor']['command_tracking_envelope_ticks'][active], 20)
                for stationary in s['stationary_limit_joints']:
                    self.assertEqual(c['descriptor']['actual_tracking_envelope_ticks'][stationary], 12)
                self.assertLess(len(c['descriptor']['sampled_path']), 17)
            else:
                self.assertEqual(c['descriptor']['actual_body_drift_ticks'], 4)
        self.assertEqual(self.robot.calls, [])
        self.grasp.open_empty.assert_not_called(); self.grasp.release.assert_not_called()
        self.brain._infer.assert_not_called()

    def test_translations_are_meaningful_and_ik_steps_are_not_dispatches(self):
        translations = [c for c in self.candidates() if c['id'].startswith('move_')]
        self.assertEqual(len(translations), 4)
        for c in translations:
            descriptor = c['descriptor']
            self.assertIn(descriptor['requested_translation_mm'], (20., 30., 40.))
            view = descriptor['camera_change_model']
            self.assertGreaterEqual(view['baseline_mm'], 15.)
            self.assertLessEqual(view['optical_forward_translation_mm'], 2.)
            self.assertLessEqual(descriptor['maximum_nominal_optical_forward_advance_mm'], 2.)
            self.assertIn('camera optical center', descriptor['translation_reference'])
            self.assertFalse(descriptor['information_gain_model']['target_depth_assumed'])
            self.assertFalse(descriptor['information_gain_model']['object_scale_inferred'])
            a = np.asarray(view['before_camera_transform_base_m'])
            b = np.asarray(view['after_camera_transform_base_m'])
            delta = (b[:3, 3] - a[:3, 3]) * 1000
            self.assertAlmostEqual(view['baseline_mm'], np.linalg.norm(delta), places=3)
            self.assertAlmostEqual(view['optical_forward_translation_mm'], delta @ a[:3, 2], places=3)
            np.testing.assert_allclose(view['translation_in_initial_camera_axes_mm'], a[:3, :3].T @ delta, atol=.00051)
        self.assertEqual(self.robot.calls, [])

    def test_orientation_only_views_are_fallbacks_after_metric_options_are_used(self):
        preferred = self.candidates()
        self.assertFalse(any(c['id'].startswith(('roll_', 'pitch_')) for c in preferred))
        fallback = self.candidates(used=METRIC_VIEW_IDS)
        self.assertEqual({c['id'] for c in fallback}, {
            'roll_positive', 'roll_negative', 'pitch_positive', 'pitch_negative', 'jaw_open'})
        for c in fallback:
            if c['kind'] == 'body':
                self.assertEqual(c['descriptor']['information_gain_model']['intent'], 'framing_or_deocclusion')
                self.assertFalse(c['descriptor']['information_gain_model']['metric_depth_verified'])

    def test_used_choices_cannot_repeat_and_total_origin_stays_fixed(self):
        candidates = self.candidates(used=('roll_positive', 'move_up', 'jaw_open'))
        self.assertTrue(all(c['id'] not in ('roll_positive', 'move_up', 'jaw_open') for c in candidates))
        origin = {**self.raw, 'wrist_roll': self.raw['wrist_roll'] - 240}
        candidates = self.candidates(origin=origin)
        self.assertTrue(all(c['origin'] == origin for c in candidates))
        self.assertNotIn('roll_positive', [c['id'] for c in candidates])
        with self.assertRaises(RuntimeError):
            self.candidates(origin={**origin, 'wrist_roll': origin['wrist_roll'] - 1})

    def test_stale_observer_or_changed_encoder_frame_never_creates_candidates(self):
        frame = self.capture()
        for changes in ({'source': 'd435'}, {'source': 'engineering_observer'},
                        {'capture_monotonic': time.monotonic() - 10},
                        {'joints': {**self.raw, 'elbow_flex': True}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_observation_candidates(self.grasp, {**frame, **changes})
        self.cancel.set()
        with self.assertRaises(SessionCancelled):
            self.candidates()
        self.brain._infer.assert_not_called(); self.assertFalse(self.robot.calls)

    def test_owned_or_already_open_jaw_not_offered(self):
        self.robot.position['gripper'] = self.robot.goals['gripper'] = 1855
        self.assertNotIn('jaw_open', [c['id'] for c in self.candidates()])
        self.robot.pwm_owner = 'someone'
        self.assertEqual(self.candidates(), [])
        self.robot.pwm_owner = None; self.grasp.state = 'holding_unverified'
        self.assertEqual(self.candidates(), [])

    def test_fixed_elbow_boundary_allows_wrist_observation_and_jaw_but_no_elbow_move(self):
        boundary = dict(zip(ALL, [2152, 931, 3040, 2334, 1037, 1531]))
        self.robot.position = dict(boundary); self.robot.goals = dict(boundary)
        with self.assertRaises(ValueError):
            self.model.validate_raw(boundary)  # Existing 12-tick motion margin.
        candidates = self.candidates()
        self.assertEqual({c['id'] for c in candidates}, {
            'move_up', 'move_backward', 'move_left', 'move_right', 'jaw_open'})
        for c in candidates:
            if c['kind'] != 'body':
                continue
            segment = c['segment']
            self.assertNotIn('elbow_flex', segment['active_joints'])
            self.assertEqual(segment['joint_targets']['elbow_flex'], boundary['elbow_flex'])
            self.assertEqual(segment['measured_bounds']['elbow_flex'], [3028, 3045])
            self.assertEqual(segment['geometry_bounds']['elbow_flex'], [3028, 3052])
            self.assertEqual(segment['onboard_margin_ticks']['elbow_flex'], 0)
            self.assertFalse(c['descriptor']['stationary_goal_resends'])
            for active in segment['active_joints']:
                self.assertEqual(c['descriptor']['onboard_margin_ticks'][active], 12)
        jaw = next(c for c in candidates if c['kind'] == 'jaw_open')
        self.assertTrue(jaw['clearance']['model_table_pass'])
        self.assertTrue(jaw['clearance']['body_motion_margin_not_applied'])
        lower = jaw['clearance']['continuous_model_lower_bound_mm']
        self.assertGreater(lower[self.model.jaw_label], 100)
        self.assertGreater(min(v for n, v in lower.items() if n != 'base_link'), 5)
        self.assertIsNone(jaw['descriptor']['aperture_mm'])
        self.assertEqual(jaw['descriptor']['jaw_target_raw'], 1860)
        self.robot.position['elbow_flex'] = self.robot.goals['elbow_flex'] = 3043
        near_limit = self.candidates()
        self.assertTrue(near_limit)
        for candidate in near_limit:
            self.assertEqual(candidate['kind'], 'body')
            segment = candidate['segment']
            self.assertNotIn('elbow_flex', segment['active_joints'])
            self.assertEqual(segment['measured_bounds']['elbow_flex'], [3031, 3045])
            self.assertEqual(segment['geometry_bounds']['elbow_flex'], [3031, 3055])
            self.assertEqual(candidate['descriptor']['measured_bounds']['elbow_flex'], [3031, 3045])
        self.robot.position['elbow_flex'] = self.robot.goals['elbow_flex'] = 3046
        self.assertEqual(self.candidates(), [])  # An actual out-of-range joint still blocks motion.

    def test_wrist_only_dispatch_near_fixed_elbow_boundary_never_writes_elbow(self):
        boundary = dict(zip(ALL, [2152, 931, 3040, 2334, 1037, 1857]))
        self.robot.position = dict(boundary); self.robot.goals = dict(boundary)
        candidate = self.candidate('pitch_negative')
        receipt = self.execute(candidate)
        self.assertTrue(receipt['completed'])
        self.assertEqual(len(self.robot.calls), 1)
        self.assertEqual(set(self.robot.calls[0][0]), {'wrist_flex'})
        self.assertEqual(self.robot.position['elbow_flex'], 3040)
        self.assertEqual(self.robot.goals['elbow_flex'], 3040)
        self.grasp.open_empty.assert_not_called()

    def test_current_pose_produces_real_camera_baselines_without_elbow_recovery(self):
        current = dict(zip(ALL, [2156, 931, 3040, 2251, 820, 1857]))
        self.robot.position = dict(current); self.robot.goals = dict(current)
        with patch.object(self.model, 'local_ik', side_effect=AssertionError('old local IK cannot start here')):
            candidates = self.candidates()
        self.assertEqual({c['id'] for c in candidates}, set(METRIC_VIEW_IDS))
        for c in candidates:
            segment = c['segment']; descriptor = c['descriptor']
            self.assertNotIn('elbow_flex', segment['active_joints'])
            self.assertEqual(segment['joint_targets']['elbow_flex'], 3040)
            self.assertGreaterEqual(len(segment['active_joints']), 2)
            self.assertGreaterEqual(descriptor['camera_change_model']['baseline_mm'], 20)
            self.assertTrue(segment['clearance']['model_table_pass'])
            self.assertLessEqual(segment['nominal_tip_path_mm'], 80)
        left = next(c for c in candidates if c['id'] == 'move_left')
        self.assertGreater(left['descriptor']['camera_change_model']['transverse_baseline_mm'], 25)
        back = next(c for c in candidates if c['id'] == 'move_backward')
        self.assertLess(back['descriptor']['camera_change_model']['optical_forward_translation_mm'], -15)
        result = self.execute(left)
        self.assertEqual(len(self.robot.calls), 1)
        self.assertNotIn('elbow_flex', self.robot.calls[0][0])
        self.assertNotIn('gripper', self.robot.calls[0][0])
        self.assertEqual(self.robot.goals['elbow_flex'], 3040)
        self.assertEqual(self.robot.position['elbow_flex'], 3040)
        self.assertGreater(result['measured_camera_change_model']['transverse_baseline_mm'], 25)
        self.assertFalse(result['information_gain_verified'])
        self.assertFalse(result['task_completed'])

    def test_current_roi_filters_retreat_that_cropped_cylinder_and_preserves_lateral_paths(self):
        # Recorded pre-retreat pose/ROI: the real backward motion later moved
        # the cylinder roughly 300 pixels down. Predict that framing loss
        # before selecting a motor path, without claiming a known depth.
        current = dict(zip(ALL, [2156, 931, 3040, 2251, 820, 1857]))
        self.robot.position = dict(current)
        self.robot.goals = {**current, 'shoulder_pan': 2158, 'wrist_flex': 2238, 'wrist_roll': 814}
        bbox = dict(x=.449, y=.257, w=.139, h=.526)
        original = self.candidates(); unchanged = deepcopy(original)
        framed = filter_observation_framing(original, bbox, [1920, 1080])
        self.assertEqual([c['id'] for c in framed], ['move_right', 'move_left'])
        self.assertEqual(original, unchanged)  # No rewrite of an assessed plan.
        for rank, candidate in enumerate(framed, 1):
            old = next(c for c in original if c['id'] == candidate['id'])
            self.assertEqual(candidate['segment'], old['segment'])
            self.assertEqual(candidate['reference'], old['reference'])
            prediction = candidate['descriptor']['target_framing_model']
            self.assertEqual(prediction['target_bbox'], bbox)
            self.assertEqual(prediction['reference_joints'], current)
            self.assertEqual(prediction['hypothetical_depth_planes_m'], [.12, .25, .6, 6.])
            self.assertEqual(prediction['minimum_predicted_visible_fraction'], 1.)
            self.assertFalse(prediction['target_depth_estimated'])
            self.assertFalse(prediction['collision_clearance_verified'])
            self.assertFalse(prediction['includes_tracking_uncertainty'])
            self.assertEqual(candidate['descriptor']['framing_preference_rank'], rank)
            # The sealed projection comes from the exact same model and path.
            before = self.model.fk.transforms(current)['wrist_camera_optical']
            for sample in candidate['framing_projection_path']:
                q = {n: current[n] + sample['fraction'] * (candidate['segment']['joint_targets'][n] - current[n])
                     for n in ALL}
                after = self.model.fk.transforms(q)['wrist_camera_optical']
                np.testing.assert_allclose(sample['reference_to_candidate_camera_transform'],
                                           np.linalg.inv(after) @ before)
        result = self.execute(framed[0])  # Resealed descriptor remains dispatchable.
        self.assertTrue(result['completed']); self.assertEqual(len(self.robot.calls), 1)

    def test_builder_uses_smaller_lateral_views_for_target_near_image_bottom(self):
        current = dict(zip(ALL, [2267, 945, 3039, 2145, 782, 1858]))
        self.robot.position = dict(current); self.robot.goals = dict(current)
        bbox = dict(x=.361, y=.47, w=.168, h=.509)
        frame = {**self.capture(), 'id': 'current-box-image', 'image_size': [1920, 1080]}
        framed = build_observation_candidates(self.grasp, frame, target_bbox=bbox)
        self.assertEqual([c['id'] for c in framed], ['pitch_positive', 'move_right', 'move_left'])
        for candidate in framed:
            descriptor = candidate['descriptor']
            if candidate['id'].startswith('move_'):
                self.assertEqual(descriptor['requested_translation_mm'], 20.)
            self.assertEqual(descriptor['target_framing_model']['reference_frame_id'], 'current-box-image')
            self.assertGreaterEqual(descriptor['target_framing_model']['minimum_predicted_visible_fraction'], .85)
            self.assertNotIn('elbow_flex', candidate['segment']['active_joints'])
            self.assertTrue(candidate['segment']['clearance']['model_table_pass'])
        self.assertFalse(self.robot.calls)

    def test_recenter_pitch_is_offered_with_metric_view_but_claims_no_depth(self):
        current = dict(zip(ALL, [2266, 999, 3040, 2079, 783, 1858]))
        self.robot.position = dict(current); self.robot.goals = dict(current)
        bbox = dict(x=.3625, y=.4815, w=.1682, h=.5167)
        candidates = self.candidates(target_bbox=bbox, image_size=[1920, 1080])
        self.assertEqual(candidates[0]['id'], 'pitch_positive')
        self.assertTrue(any(c['id'].startswith('move_') for c in candidates))
        self.assertNotIn('pitch_negative', [c['id'] for c in candidates])
        recovery = candidates[0]; descriptor = recovery['descriptor']
        self.assertEqual(descriptor['purpose'], 'recenter_for_translation')
        self.assertEqual(descriptor['target_framing_model']['minimum_predicted_visible_fraction'], 1.)
        self.assertGreater(descriptor['target_framing_model']['minimum_endpoint_vertical_centering_improvement'], .1)
        self.assertTrue(descriptor['information_gain_model']['subsequent_translation_required_for_metric_depth'])
        self.assertFalse(descriptor['information_gain_model']['metric_depth_verified'])
        self.assertFalse(descriptor['information_gain_model']['object_scale_inferred'])
        self.assertEqual(recovery['segment']['active_joints'], ['wrist_flex'])
        self.assertEqual(recovery['segment']['delta_ticks']['wrist_flex'], 68)
        self.assertTrue(recovery['segment']['clearance']['model_table_pass'])
        # An already-centered ROI does not justify spending a move on aiming.
        centered = self.candidates(target_bbox=dict(x=.45, y=.3, w=.1, h=.4), image_size=[1920, 1080])
        self.assertTrue(any(c['id'].startswith('move_') for c in centered))
        self.assertFalse(any(c['descriptor'].get('purpose') == 'recenter_for_translation' for c in centered))
        receipt = self.execute(recovery)
        self.assertTrue(receipt['completed']); self.assertFalse(receipt['information_gain_verified'])

    def test_continuing_direction_requires_history_novelty_and_has_finite_visit_limit(self):
        prior = {**self.capture(), 'id': 'initial-view', 'capture_monotonic': time.monotonic() - 10}
        self.assertFalse(any(c['id'].startswith('move_left') for c in self.candidates(used=('move_left',))))
        second = self.candidates(used=('move_left',), history=[prior])
        continuation = next(c for c in second if c['id'] == 'move_left_continue_2')
        descriptor = continuation['descriptor']
        self.assertEqual(descriptor['direction_family'], 'move_left')
        self.assertEqual(descriptor['direction_visit'], 2)
        self.assertEqual(descriptor['maximum_direction_visits'], 3)
        self.assertFalse(descriptor['view_novelty_model']['predicted_revisit'])
        self.assertEqual(continuation['origin'], self.raw)
        third = self.candidates(used=('move_left', 'move_left_continue_2'), history=[prior])
        self.assertIn('move_left_continue_3', [c['id'] for c in third])
        exhausted = self.candidates(used=('move_left', 'move_left_continue_2', 'move_left_continue_3'), history=[prior])
        self.assertFalse(any(c['id'].startswith('move_left') for c in exhausted))
        observer = self.candidates(used=('move_left',), history=[{**prior, 'source': 'd435'}])
        self.assertFalse(any(c['id'].startswith('move_left') for c in observer))
        # A syntactically new continuation ID cannot authorize an old viewpoint.
        old_destination = {**prior, 'id': 'already-observed-left-view',
                           'joints': dict(continuation['segment']['joint_targets'])}
        duplicate = self.candidates(used=('move_left',), history=[old_destination])
        self.assertNotIn('move_left_continue_2', [c['id'] for c in duplicate])
        self.assertFalse(self.robot.calls)

    def test_framing_requires_valid_current_image_box_and_size(self):
        valid = dict(x=.4, y=.3, w=.15, h=.5)
        for box, size in [(dict(valid, w=.8), [1920, 1080]),
                          (dict(valid, x=True), [1920, 1080]),
                          (dict(valid, h=float('nan')), [1920, 1080]),
                          (valid, None), (valid, [1920, 0]), (valid, [True, 1080])]:
            with self.subTest(box=box, size=size), self.assertRaises(ValueError):
                self.candidates(target_bbox=box, image_size=size)
        self.assertFalse(self.robot.calls)

    def test_framing_filter_rejects_changed_contract_or_mixed_reference_images(self):
        from server.active_observation import _seal
        candidates = self.candidates()
        bbox = dict(x=.4, y=.3, w=.15, h=.5)
        changed = deepcopy(candidates)
        changed[0]['segment']['joint_targets']['wrist_flex'] += 1
        with self.assertRaisesRegex(ValueError, '候選已改變'):
            filter_observation_framing(changed, bbox, [1920, 1080])
        mixed = deepcopy(candidates)
        mixed[0]['reference'] = {**mixed[0]['reference'], 'path': 'another-image.png'}
        _seal(mixed[0])
        with self.assertRaisesRegex(ValueError, '同一張'):
            filter_observation_framing(mixed, bbox, [1920, 1080])
        self.cancel.set()
        with self.assertRaises(SessionCancelled):
            filter_observation_framing(candidates, bbox, [1920, 1080], cancel=self.cancel)
        self.assertFalse(self.robot.calls)

    def test_framing_refresh_binds_actual_planning_image_without_rewriting_motor_path(self):
        original = {**self.capture(), 'id': 'candidate-image'}
        candidates = build_observation_candidates(self.grasp, original)
        bbox = dict(x=.45, y=.4, w=.1, h=.2)
        refreshed = {**original, 'id': 'planning-image', 'path': 'new-planning-image.png',
                     'capture_monotonic': original['capture_monotonic'] + .1,
                     'joints': {**original['joints'], 'wrist_roll': original['joints']['wrist_roll'] + 3}}
        framed = filter_observation_framing(candidates, bbox, [1920, 1080], reference_frame=refreshed)
        self.assertTrue(any(c['kind'] == 'body' for c in framed))
        for candidate in framed:
            old = next(c for c in candidates if c['id'] == candidate['id'])
            self.assertEqual(candidate['segment'], old['segment'])
            self.assertEqual(candidate['reference'], old['reference'])
            if candidate['kind'] != 'body':
                continue
            prediction = candidate['descriptor']['target_framing_model']
            self.assertEqual(prediction['reference_frame_id'], 'planning-image')
            self.assertEqual(prediction['reference_path'], 'new-planning-image.png')
            self.assertEqual(prediction['reference_joints'], refreshed['joints'])
            self.assertEqual(prediction['candidate_reference_frame_id'], 'candidate-image')
            self.assertEqual(prediction['candidate_reference_path'], original['path'])
            self.assertEqual(prediction['candidate_reference_joints'], original['joints'])
            self.assertEqual(prediction['stationary_refresh_drift_ticks']['wrist_roll'], 3)
        for invalid in ({**refreshed, 'source': 'd435'},
                        {**refreshed, 'camera_generation': 'different-camera'},
                        {**refreshed, 'capture_monotonic': original['capture_monotonic'] - .1},
                        {**refreshed, 'joints': {**refreshed['joints'],
                                               'wrist_roll': original['joints']['wrist_roll'] + 4}}):
            with self.subTest(frame=invalid), self.assertRaises(ValueError):
                filter_observation_framing(candidates, bbox, [1920, 1080], reference_frame=invalid)
        self.assertTrue(filter_observation_framing(candidates, bbox, [1920, 1080], reference_frame=original))
        self.assertFalse(self.robot.calls)

    def test_novelty_removes_return_to_prior_pose_only_with_another_novel_body_path(self):
        candidates = self.candidates(); unchanged = deepcopy(candidates)
        left = next(c for c in candidates if c['id'] == 'move_left')
        prior = {**left['reference'], 'id': 'earlier-left-view', 'path': 'prior-left.png',
                 'capture_monotonic': left['reference']['capture_monotonic'] - 10,
                 'joints': dict(left['segment']['joint_targets'])}
        novel = filter_observation_novelty(candidates, [prior], model=self.model)
        self.assertNotIn('move_left', [c['id'] for c in novel])
        self.assertIn('move_right', [c['id'] for c in novel])
        self.assertIn('jaw_open', [c['id'] for c in novel])
        self.assertEqual(candidates, unchanged)
        for candidate in novel:
            original = next(c for c in candidates if c['id'] == candidate['id'])
            self.assertEqual(candidate['segment'], original['segment'])
            self.assertEqual(candidate['reference'], original['reference'])
        # Repeating a view can still be useful if no novel screened move fits.
        only_revisit = filter_observation_novelty([left], [prior], model=self.model)
        self.assertEqual(len(only_revisit), 1)
        novelty = only_revisit[0]['descriptor']['view_novelty_model']
        self.assertTrue(novelty['predicted_revisit'])
        self.assertTrue(novelty['retained_without_novel_alternative'])
        self.assertEqual(novelty['nearest_prior']['frame_id'], 'earlier-left-view')
        self.assertEqual(novelty['nearest_prior']['translation_mm'], 0.)
        self.assertEqual(novelty['nearest_prior']['rotation_degrees'], 0.)
        self.assertFalse(novelty['actual_information_gain_verified'])
        self.assertFalse(novelty['object_scale_inferred'])
        result = self.execute(next(c for c in novel if c['id'] == 'move_right'))
        self.assertTrue(result['completed'])  # Descriptor changes are resealed.

    def test_novelty_uses_both_rotation_and_translation_and_ignores_foreign_history(self):
        candidates = self.candidates()
        left = next(c for c in candidates if c['id'] == 'move_left')
        target = left['segment']['joint_targets']
        prior = {**left['reference'], 'id': 'nearby-turned-view', 'path': 'prior-turned.png',
                 'capture_monotonic': left['reference']['capture_monotonic'] - 10,
                 'joints': {**target, 'wrist_roll': target['wrist_roll'] + 30}}
        exact = {**prior, 'joints': dict(target)}
        foreign = [{**exact, 'id': 'observer-view', 'source': 'd435'},
                   {**exact, 'id': 'replaced-camera', 'camera_generation': 'other-generation'},
                   {**exact, 'id': 'future-view',
                    'capture_monotonic': left['reference']['capture_monotonic'] + 10}]
        filtered = filter_observation_novelty(candidates, [prior, *foreign], model=self.model)
        retained = next(c for c in filtered if c['id'] == 'move_left')
        novelty = retained['descriptor']['view_novelty_model']
        self.assertEqual(novelty['eligible_prior_views'], 1)
        self.assertEqual(novelty['excluded_history_records'], 3)
        self.assertEqual(novelty['nearest_prior']['frame_id'], 'nearby-turned-view')
        self.assertLess(novelty['nearest_prior']['translation_mm'], 5.)
        self.assertGreater(novelty['nearest_prior']['rotation_degrees'], 2.)
        self.assertFalse(novelty['predicted_revisit'])
        self.cancel.set()
        with self.assertRaises(SessionCancelled):
            filter_observation_novelty(candidates, [prior], model=self.model, cancel=self.cancel)
        self.assertFalse(self.robot.calls)

    def test_table_failure_blocks_jaw_and_body_candidates(self):
        with patch.object(self.model, 'floor_minimum', return_value={
                'base_link': -.002, self.model.jaw_label: .004}):
            self.assertEqual(self.candidates(), [])

    def test_body_executes_one_continuous_segment_without_inference_or_jaw(self):
        candidate = self.candidate('move_up')
        receipt = self.execute(candidate)
        self.assertTrue(receipt['completed'])
        self.assertFalse(receipt['task_completed'])
        self.assertFalse(receipt['information_gain_verified'])
        self.assertEqual(receipt['cloud_calls_during_motion'], 0)
        self.assertGreaterEqual(receipt['physical_seconds_charged'], 0)
        self.assertGreater(receipt['after']['capture_monotonic'], receipt['before']['capture_monotonic'])
        self.assertEqual(len(self.robot.calls), 1)
        self.assertNotIn('gripper', self.robot.calls[0][0])
        self.grasp.open_empty.assert_not_called(); self.brain._infer.assert_not_called()

    def test_final_capture_compares_original_review_not_pairwise_scene_drift(self):
        from server.active_observation import _body_candidate, _jaw_candidate
        from server.calibration_motion import _candidate as calibration_candidate
        from server.grasp import WristSceneChangedError
        images = []
        for index in range(3):
            image = np.full((270, 480), 128, dtype=np.uint8)
            for patch_index in range(index):
                image[10:40, 10 + patch_index * 55:48 + patch_index * 55] = 198
            path = Path(self.tmp.name) / f'scene-{index}.png'
            cv2.imwrite(str(path), image)
            images.append(str(path))
        for kind in ('object_body', 'calibration_body', 'jaw_open'):
            with self.subTest(kind=kind):
                reviewed = {**self.capture(), 'id': 'reviewed-A', 'path': images[0],
                            'capture_monotonic': time.monotonic() - 2}
                target = {**self.raw, 'wrist_roll': self.raw['wrist_roll'] + 91}
                if kind == 'jaw_open':
                    candidate = _jaw_candidate(self.grasp, reviewed, self.raw)
                elif kind == 'calibration_body':
                    candidate = calibration_candidate(self.grasp, reviewed, self.raw, target,
                                                       'calibration-turn', '轉腕', '校準')
                else:
                    candidate = _body_candidate(self.grasp, reviewed, self.raw, target,
                                                'object-turn', '轉腕', '觀察')
                refreshed = {**reviewed, 'id': 'dispatch-B', 'path': images[1],
                             'capture_monotonic': time.monotonic() - 1}
                latest = {**reviewed, 'id': 'start-C', 'path': images[2],
                          'capture_monotonic': time.monotonic()}
                # Each adjacent 0.88% patch passes; the original-to-final
                # 1.76% difference must stop BEFORE any motion is dispatched.
                unchanged_wrist(reviewed, refreshed)
                unchanged_wrist(refreshed, latest)
                self.grasp.capture = Mock(return_value=latest)
                with self.assertRaises(WristSceneChangedError):
                    execute_observation(self.grasp, candidate, refreshed, self.cancel, self.report,
                        verdict=self.verdict(candidate),
                        engineering_calibration=kind == 'calibration_body')
                self.assertEqual(self.robot.calls, [])
                self.grasp.open_empty.assert_not_called()
                self.brain._infer.assert_not_called()

    def test_body_observation_accepts_elbow_settling_inside_declared_hold_box(self):
        candidate = self.candidate('roll_positive')
        self.robot.callback = lambda r, _: r.position.update(elbow_flex=self.raw['elbow_flex'] - 5)
        receipt = self.execute(candidate)
        self.assertTrue(receipt['completed'])
        self.assertEqual(receipt['after']['joints']['elbow_flex'], self.raw['elbow_flex'] - 5)
        self.assertEqual(self.robot.goals['elbow_flex'], self.held['elbow_flex'])
        self.assertEqual(self.robot.calls[0][0], {'wrist_roll': candidate['segment']['joint_targets']['wrist_roll']})
        self.assertEqual(candidate['segment']['stationary_tolerance_ticks'], 12)
        self.assertEqual(candidate['descriptor']['stationary_tolerance_ticks'], 12)
        self.assertEqual(candidate['descriptor']['measured_bounds']['elbow_flex'],
                         [self.raw['elbow_flex'] - 12, self.raw['elbow_flex'] + 12])
        self.assertFalse(self.robot.stop.is_set())

    def test_body_observation_exceeding_stationary_box_stops(self):
        candidate = self.candidate('roll_positive')
        self.robot.callback = lambda r, _: r.position.update(elbow_flex=self.raw['elbow_flex'] - 13)
        with self.assertRaisesRegex(RuntimeError, '手肘.*13 刻度.*12 刻度'):
            self.execute(candidate)
        self.assertTrue(self.robot.stop.is_set())
        self.assertFalse(any(call.args[0].get('type') == 'active_observation_result'
                             for call in self.grasp.record.call_args_list))

    def test_reported_observation_path_keeps_full_hold_reserve_inside_attempt_budget(self):
        from server.active_observation import _body_candidate
        # Incident encoder start/target, with its known unchanged elbow goal.
        # Other original held goals were not logged, so neutral biases below
        # are an explicit offline fixture rather than a full hardware replay.
        raw = dict(zip(ALL, [2113, 1159, 3042, 1891, 830, 1859]))
        target = dict(zip(ALL, [2124, 979, 3042, 2053, 814, 1859]))
        self.robot.position = dict(raw)
        self.robot.goals = {**raw, 'elbow_flex': 3037}
        candidate = _body_candidate(self.grasp, self.capture(), raw, target,
            'incident_view', '重現觀察路徑', '離線幾何驗證')
        descriptor = candidate['descriptor']
        self.assertEqual(descriptor['stationary_tolerance_ticks'], 12)
        self.assertEqual(descriptor['measured_bounds']['elbow_flex'], [3030, 3045])
        self.assertEqual(descriptor['geometry_bounds']['elbow_flex'], [3030, 3054])
        self.assertTrue(candidate['segment']['clearance']['model_table_pass'])
        self.assertEqual(descriptor['attempt_limits']['maximum_tip_distance_from_origin_mm'], 80.)
        self.assertFalse(self.robot.calls)

    def test_observation_declared_twenty_tick_geometry_accepts_eighteen_tick_shortfall(self):
        candidate = self.candidate('pitch_positive')
        self.robot.endpoint_error = {'wrist_flex': -18}
        receipt = self.execute(candidate)
        self.assertTrue(receipt['completed'])
        self.assertEqual(candidate['segment']['endpoint_tolerance_ticks'], 20)
        self.assertEqual(candidate['descriptor']['actual_tracking_envelope_ticks']['wrist_flex'], 20)
        self.assertEqual(self.robot.calls[0][1]['precision'], 20)
        self.assertEqual(receipt['movement']['endpoint_error_ticks']['wrist_flex'], -18)
        self.assertAlmostEqual(receipt['directional_progress_fraction'], 50 / 68)
        self.assertEqual(receipt['after']['joints']['wrist_flex'], self.raw['wrist_flex'] + 50)
        self.assertGreater(receipt['after']['capture_monotonic'], receipt['before']['capture_monotonic'])
        self.assertFalse(self.robot.stop.is_set())

    def test_observation_more_than_twenty_ticks_short_stops_without_completion(self):
        candidate = self.candidate('pitch_positive')
        self.robot.endpoint_error = {'wrist_flex': -21}
        with self.assertRaisesRegex(RuntimeError, '終點'):
            self.execute(candidate)
        self.assertTrue(self.robot.stop.is_set())
        self.assertEqual(self.robot.hold_calls, 1)
        self.assertFalse(any(call.args[0].get('type') == 'active_observation_result'
                             for call in self.grasp.record.call_args_list))

    def test_endpoint_tolerance_does_not_count_less_than_forty_percent_progress(self):
        # A short candidate can be within 20 ticks of its endpoint yet make
        # little useful progress. Execute a real CAD-screened short segment
        # through the fake plant to exercise this separate completion check.
        from server.active_observation import _body_candidate
        target = {**self.raw, 'wrist_flex': self.raw['wrist_flex'] + 20}
        candidate = _body_candidate(self.grasp, self.capture(), self.raw, target,
            'synthetic_short_view', '測試短視角', '離線幾何測試')
        self.robot.endpoint_error = {'wrist_flex': -15}
        with self.assertRaisesRegex(RuntimeError, '實際移動不足'):
            self.execute(candidate)
        self.assertEqual(self.robot.position['wrist_flex'], self.raw['wrist_flex'] + 5)
        self.assertTrue(self.robot.stop.is_set())
        self.assertEqual(self.robot.hold_calls, 1)
        self.assertFalse(any(call.args[0].get('type') == 'active_observation_result'
                             for call in self.grasp.record.call_args_list))

    def test_denied_or_wrong_or_mutated_verdict_never_dispatches(self):
        candidate = self.candidate('roll_positive')
        for change in ({'candidate_id': 'jaw_open'}, {'gripper_empty': False}, {'human_clear': False},
                       {'clear_path': False}, {'confidence': .69}, {'gripper_empty': 'true'}):
            with self.subTest(change=change), self.assertRaises((ValueError, ValidationError)):
                self.execute(candidate, verdict=self.verdict(candidate, **change))
        mutated = deepcopy(candidate); mutated['segment']['joint_targets']['wrist_roll'] += 1
        with self.assertRaises(ValueError):
            self.execute(mutated)
        self.assertFalse(self.robot.calls)
        self.grasp.open_empty.assert_not_called()

    def test_jaw_opens_then_finishes_only_explicit_empty_transaction(self):
        candidate = self.candidate('jaw_open')
        self.assertEqual(candidate['descriptor']['minimum_visual_confidence'], .7)
        result = self.execute(candidate, verdict=self.verdict(candidate, confidence=.8))
        self.assertTrue(result['completed'])
        self.assertFalse(result['task_completed'])
        self.grasp.open_empty.assert_called_once()
        self.grasp.release.assert_called_once()
        self.assertTrue(self.grasp.release.call_args.kwargs['empty_verified'])
        self.assertNotIn('support_verified', self.grasp.release.call_args.kwargs)
        self.assertIsNone(self.robot.pwm_owner)
        self.assertEqual(self.robot.gripper_power.phase, 'completed')
        self.brain._infer.assert_not_called()

    def test_selected_profile_opening_is_disclosed_assessed_executed_and_retained(self):
        from test_jaw_opening import selection
        self.robot.position['gripper']=self.robot.goals['gripper']=1856
        self.robot.config={'spatial_backend':'v2','jaw_calibration_profile':'test-only.json'}
        with patch('server.jaw_opening._profile_snapshot',return_value=selection()):
            candidate=self.candidate('jaw_open')
            self.assertEqual(candidate['descriptor']['jaw_target_raw'],2038)
            self.assertEqual(candidate['descriptor']['jaw_raw_sweep'],[1850,2044])
            self.assertGreater(candidate['descriptor']['aperture_mm'],64.)
            result=self.execute(candidate)
        self.assertEqual(self.robot.position['gripper'],2038)
        self.assertEqual(result['after']['joints']['gripper'],2038)
        self.assertEqual(self.grasp.open_empty.call_args.kwargs['opening_contract'],candidate['opening_contract'])
        self.assertFalse(result['task_completed']);self.assertIsNone(self.robot.pwm_owner)

    def test_profile_replacement_after_visual_assessment_never_dispatches_new_opening(self):
        from test_jaw_opening import selection,profile
        self.robot.position['gripper']=self.robot.goals['gripper']=1856
        self.robot.config={'spatial_backend':'v2','jaw_calibration_profile':'test-only.json'}
        with patch('server.jaw_opening._profile_snapshot',return_value=selection()):
            candidate=self.candidate('jaw_open')
        with patch('server.jaw_opening._profile_snapshot',return_value=selection(profile('changed'))):
            with self.assertRaisesRegex(RuntimeError,'重新審核'):
                self.execute(candidate)
        self.grasp.open_empty.assert_not_called();self.assertIsNone(self.robot.pwm_owner)

    def test_wide_opening_must_remain_inside_the_original_disclosed_raw_sweep(self):
        from test_jaw_opening import selection
        self.robot.position['gripper']=self.robot.goals['gripper']=1856
        self.robot.config={'spatial_backend':'v2','jaw_calibration_profile':'test-only.json'}
        def outside(guard):
            self.robot.position['gripper']=1849
            guard(self.robot.position)
        self.open_hook=outside
        with patch('server.jaw_opening._profile_snapshot',return_value=selection()):
            candidate=self.candidate('jaw_open')
            self.assertEqual(candidate['descriptor']['jaw_raw_sweep'],[1850,2044])
            with self.assertRaisesRegex(RuntimeError,'掃掠範圍'):
                self.execute(candidate)
        self.grasp.release.assert_not_called()
        self.assertEqual(self.robot.gripper_power.phase,'blocked')

    def test_jaw_low_confidence_or_pending_work_has_no_opening(self):
        candidate = self.candidate('jaw_open')
        with self.assertRaisesRegex(ValueError, '把握還不夠'):
            self.execute(candidate, verdict=self.verdict(candidate, confidence=.69))
        self.robot.pwm_owner = 'existing'
        with self.assertRaises(RuntimeError):
            self.execute(candidate, verdict=self.verdict(candidate, confidence=.8))
        self.grasp.open_empty.assert_not_called(); self.grasp.release.assert_not_called()

    def test_jaw_point_eight_still_requires_all_flags_and_no_held_object(self):
        candidate = self.candidate('jaw_open')
        for flag in ('gripper_empty', 'human_clear', 'clear_path'):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, '空夾爪與完整路徑'):
                self.execute(candidate, verdict=self.verdict(candidate, confidence=.8, **{flag: False}))
        for state in ('contact_candidate', 'holding_unverified', 'holding'):
            self.grasp.state = state
            with self.subTest(state=state), self.assertRaisesRegex(RuntimeError, '目前的夾爪工作'):
                self.execute(candidate, verdict=self.verdict(candidate, confidence=.8))
        self.grasp.open_empty.assert_not_called(); self.grasp.release.assert_not_called()

    def test_jaw_failure_preserves_pending_low_power_and_never_releases(self):
        candidate = self.candidate('jaw_open')
        def drift(guard):
            self.robot.position['elbow_flex'] += 5
            guard(self.robot.position)
        self.open_hook = drift
        with self.assertRaises(RuntimeError):
            self.execute(candidate)
        self.grasp.release.assert_not_called()
        self.robot.gripper_power.leave_pending.assert_called_once()
        self.assertTrue(self.robot.stop.is_set())
        self.assertEqual(self.robot.hold_calls, 1)

    def test_jaw_contact_record_cannot_be_finished_as_empty(self):
        candidate = self.candidate('jaw_open')
        self.open_hook = lambda guard: setattr(self.robot.gripper_power, 'contact_possible', True)
        with self.assertRaisesRegex(RuntimeError, '接觸'):
            self.execute(candidate)
        self.grasp.release.assert_not_called()
        self.robot.gripper_power.leave_pending.assert_called_once()

    def test_jaw_deadline_and_cancel_never_restore_power(self):
        candidate = self.candidate('jaw_open')
        with self.assertRaisesRegex(RuntimeError, '時間不足'):
            self.execute(candidate, deadline=time.monotonic() + 5)
        self.grasp.open_empty.assert_not_called()
        self.open_hook = lambda guard: self.cancel.set()
        with self.assertRaises(SessionCancelled):
            self.execute(candidate)
        self.grasp.release.assert_not_called()
        self.robot.gripper_power.leave_pending.assert_called_once()

    def test_changed_camera_or_goals_cannot_use_previous_jaw_approval(self):
        candidate = self.candidate('jaw_open')
        self.robot.camera.generation = 'new'
        with self.assertRaises(RuntimeError): self.execute(candidate)
        self.robot.camera.generation = 'one'
        self.robot.goals['elbow_flex'] += 1
        with self.assertRaisesRegex(RuntimeError, '保持目標'):
            self.execute(candidate)
        self.grasp.open_empty.assert_not_called()

    def test_cancel_during_empty_cleanup_prevents_cap_restore_and_restores_callbacks(self):
        candidate = self.candidate('jaw_open')
        power = self.robot.gripper_power
        read, write = power.read, power.write
        power.read.side_effect = lambda *args: self.cancel.set() or 1860
        with self.assertRaises(SessionCancelled):
            self.execute(candidate)
        write.assert_not_called()
        self.assertIs(power.read, read); self.assertIs(power.write, write)
        power.leave_pending.assert_called_once()
        self.assertEqual(power.phase, 'blocked')
        self.assertTrue(self.robot.stop.is_set())

    def test_body_drift_during_empty_cleanup_prevents_cap_restore(self):
        candidate = self.candidate('jaw_open')
        power = self.robot.gripper_power
        write = power.write
        self.release_hook = lambda: self.robot.position.update(elbow_flex=self.raw['elbow_flex'] + 5)
        with self.assertRaisesRegex(RuntimeError, '固定姿勢'):
            self.execute(candidate)
        write.assert_not_called()
        power.leave_pending.assert_called_once()

    def test_release_raw_snapshot_may_repeat_latest_camera_frame(self):
        # A camera snapshot returns the latest exposure, not a new timestamp
        # for each read. Only settled capture waits for a newer exposure.
        candidate = self.candidate('jaw_open')
        latest = {}
        capture = self.capture
        def settled_capture():
            frame = capture()
            latest['frame'] = frame
            return frame
        self.capture = settled_capture
        self.grasp.capture = settled_capture
        release = self.release
        cleanup_references = []
        def repeated_raw_snapshot(reference, report, cancel, **kwargs):
            raw_snapshot = latest['frame']
            unchanged_wrist(reference, raw_snapshot)
            cleanup_references.append((reference, raw_snapshot))
            return release(reference, report, cancel, **kwargs)
        self.grasp.release.side_effect = repeated_raw_snapshot
        result = self.execute(candidate)
        self.assertTrue(result['completed'])
        self.assertEqual(len(cleanup_references), 1)
        opened, latest_before_cleanup = cleanup_references[0]
        self.assertGreater(latest_before_cleanup['capture_monotonic'], opened['capture_monotonic'])
        self.robot.gripper_power.leave_pending.assert_not_called()
        self.assertEqual(self.robot.gripper_power.phase, 'completed')


if __name__ == '__main__':
    unittest.main()
