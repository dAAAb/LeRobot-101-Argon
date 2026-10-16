"""No-device integration of measured geometry, local plans, and VLM gates."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from server.gripper_power import SessionCancelled
from server.local_cartesian import ALL, default_model
from server.object_program import ObjectProgramBlocked
from server.spatial_contract import JawCalibration, JawSample
from server.spatial_object_program import (SpatialObjectIntent, _plan_cylinder,
    _side_approaches, plan_spatial_object_program, prepare_contact_lift, refresh_depth_binding,
    required_grasp_setups)


class Brain:
    def __init__(self):
        self.calls = []
        self.callback = None
        self.reject = False

    def _infer(self, prompt, frames, schema, cancel=None):
        self.calls.append((prompt, frames, schema))
        if self.callback:
            self.callback()
        keys = set(schema.model_fields) - {'confidence', 'reason'}
        result = {k: True for k in keys}
        if self.reject:
            result['external_path_clear'] = False
        result.update(confidence=.85, reason='這條路徑旁邊有障礙。' if self.reject else '這條路徑可通過。')
        return result


class SpatialObjectProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()
        cls.raw = dict(zip(ALL, [2155, 1182, 2946, 2268, 977, 1857]))
        wrist = cls.model.fk.transforms(cls.raw)['gripper_link']
        pad = cls.model.tip(cls.raw)
        center = (np.linalg.inv(wrist) @ np.r_[pad[:3, 3], 1.])[:3]
        # Synthetic landmarks deliberately give TCP +Z=base +X and closing
        # +X=base +Y. They are test geometry, never the user's calibration.
        closing = wrist[:3, :3].T @ [0., 1., 0.]
        approach = wrist[:3, :3].T @ [1., 0., 0.]
        cls.profile = JawCalibration('test-jaw', 'test-geometry', (
            JawSample(1700, center - .020 * closing, center + .020 * closing),
            JawSample(2000, center - .035 * closing, center + .035 * closing)),
            approach, .0005, 'validated', 'synthetic-only', 'independent-heldout')

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = str(Path(self.directory.name) / 'wrist.png')
        cv2.imwrite(self.image, np.full((60, 80, 3), 128, np.uint8))
        self.cancel = threading.Event()
        self.base_time = time.monotonic() - .2
        self.brain = Brain()
        camera = SimpleNamespace(generation='camera-1', fresh=lambda: True)
        robot = SimpleNamespace(goals=dict(self.raw), pwm_owner='owner-1', config={},
                                camera=camera, stop=threading.Event())
        self.events = []
        self.grasp = SimpleNamespace(model=self.model, brain=self.brain, robot=robot,
            state='contact_candidate', spatial_attempt={'track_id': 'object-1'}, record=self.events.append)
        self.intent = SpatialObjectIntent(source_object='圓筒', target_visible=True,
            target_bbox={'x': .3, 'y': .2, 'w': .2, 'h': .6}, shape='upright_cylinder',
            initial_jaw_empty=True, human_clear=True, observation=None, reason='可見直立圓筒。',
            cylinder_contours=[], support_surface='other_or_unknown')

    def frame(self, sequence=1, *, raw=None, generation='camera-1'):
        return {'id': f'frame-{sequence}', 'source': 'wrist', 'sequence': sequence,
                'camera_generation': generation, 'capture_monotonic': self.base_time + sequence * .001,
                'joints': dict(raw or self.raw), 'path': self.image, 'image_size': [80, 60]}

    def packet(self, frame):
        calibration = SimpleNamespace(geometry_revision='test-geometry',
            calibration_revision='test-camera', fingerprint='test-fingerprint', metric_calibrated=True,
            contains_pose=lambda raw: True)
        return SimpleNamespace(calibration=calibration, metric_calibrated=True,
            jaw_geometry=self.profile.at(frame['joints']['gripper']),
            T_base_camera=SimpleNamespace(as_array=lambda: np.diag([1., -1., -1., 1.])))

    def geometry(self):
        target = self.model.tip(self.raw)[:3, 3] + [.006, 0., 0.]
        points = []
        for height in np.linspace(-.03, .03, 5):
            for angle in np.linspace(np.pi / 2, 3 * np.pi / 2, 18):
                points.append({'position_base_m': (target + [.012 * np.cos(angle),
                    .012 * np.sin(angle), height]).tolist(), 'views': 3})
        return {'source': 'wrist-only', 'status': 'observed', 'frame_id': 'frame-1',
                'camera_generation': 'camera-1', 'track_id': 'object-1',
                'geometry_revision': 'test-geometry', 'calibration_revision': 'test-camera',
                'metric_calibrated': True, 'observed_surface': {
                    'points': points, 'position_error_bound_m': .0005}}

    def plan(self, geometry=None):
        frame = self.frame()
        return _plan_cylinder(self.grasp, 'grasp', self.intent, geometry or self.geometry(),
                              self.packet(frame), frame, self.cancel, 30., ())

    def test_cylinder_fitter_to_exact_local_program_uses_real_api(self):
        result = self.plan()
        self.assertEqual(result.status, 'ready', result.unsupported_reason)
        self.assertEqual(self.grasp.spatial_attempt['contact_geometry']['status'], 'hypothesis')
        kinds = [step.kind for step in result.steps]
        self.assertEqual(kinds.count('close'), 1)
        self.assertEqual(kinds.count('verify_lift'), 1)
        self.assertEqual(kinds[-2:], ['close', 'verify_lift'])
        self.assertLessEqual(self.grasp.spatial_attempt['contact_alignment']['closing_vertical_component'], .25)
        proposal = self.grasp.spatial_attempt['proposal']
        for step, original in zip(result.steps, proposal['segments']):
            self.assertEqual(step.body_target.model_dump() if step.body_target else None, original['body_target'])
        self.assertEqual(len(self.brain.calls), 1)

    def test_model_assisted_eligibility_keeps_aperture_error_in_available_clearance(self):
        packet=self.packet(self.frame());jaw=packet.jaw_geometry
        estimate=SimpleNamespace(**vars(jaw))
        estimate.metric_calibrated=False;estimate.planning_eligible=True
        estimate.to_dict=lambda: {'accuracy_basis':'conditional-cad-estimate',
            'aperture_uncertainty_m':jaw.aperture_m}
        packet.jaw_geometry=estimate
        self.assertEqual(required_grasp_setups(packet),[])
        result=_plan_cylinder(self.grasp,'grasp',self.intent,self.geometry(),packet,
            self.frame(),self.cancel,30.,())
        self.assertEqual(result.status,'unsupported')
        self.assertIn('還不能確認',result.unsupported_reason)
        self.assertEqual(self.brain.calls,[])
        estimate.planning_eligible=False
        self.assertEqual(required_grasp_setups(packet),['jaw_calibration'])

    def test_insufficient_gap_reuses_only_the_already_assessed_jaw_opening(self):
        packet=self.packet(self.frame());jaw=packet.jaw_geometry
        packet.jaw_geometry=SimpleNamespace(**vars(jaw))
        packet.jaw_geometry.to_dict=lambda: {'aperture_uncertainty_m':jaw.aperture_m}
        verdict=dict(candidate_id='jaw_open',gripper_empty=True,human_clear=True,
            clear_path=True,confidence=.95,reason='先張到已揭露的較大開口，再重新觀察筒身。')
        intent=SpatialObjectIntent.model_validate({**self.intent.model_dump(),'observation':verdict})
        candidate={'id':'jaw_open','kind':'jaw_open'}
        result=_plan_cylinder(self.grasp,'grasp',intent,self.geometry(),packet,
            self.frame(),self.cancel,30.,[candidate])
        self.assertEqual(result.status,'observe')
        self.assertEqual(result.observation.model_dump(),verdict)
        self.assertEqual(self.brain.calls,[])
        for choice,candidates in ((self.intent,[candidate]),(intent,[])):
            blocked=_plan_cylinder(self.grasp,'grasp',choice,self.geometry(),packet,
                self.frame(),self.cancel,30.,candidates)
            self.assertEqual(blocked.status,'unsupported')

    def test_gap_uncertainty_can_reobserve_with_exact_assessed_body_candidate(self):
        # A wide but uncertain measured surface must not force another opening
        # when Astra has assessed a new shoulder/elbow viewpoint instead.
        packet = self.packet(self.frame()); jaw = packet.jaw_geometry
        packet.jaw_geometry = SimpleNamespace(**vars(jaw))
        packet.jaw_geometry.to_dict = lambda: {'aperture_uncertainty_m': jaw.aperture_m}
        verdict = dict(candidate_id='move_up', gripper_empty=True, human_clear=True,
            clear_path=True, confidence=.8, reason='用肩肘抬高相機，看清罐身兩側。')
        intent = SpatialObjectIntent.model_validate({**self.intent.model_dump(), 'observation': verdict})
        candidate = {'id': 'move_up', 'kind': 'body'}
        result = _plan_cylinder(self.grasp, 'grasp', intent, self.geometry(), packet,
            self.frame(), self.cancel, 30., [candidate])
        self.assertEqual(result.status, 'observe')
        self.assertEqual(result.observation.model_dump(), verdict)
        self.assertEqual(result.steps, [])
        self.assertEqual(self.brain.calls, [])
        self.assertLessEqual(self.grasp.spatial_attempt['grasp_budget']['one_side_clearance_m'], 0)
        for choice, candidates in ((self.intent, [candidate]),
                (intent, [{'id': 'other', 'kind': 'body'}]), (intent, [])):
            blocked = _plan_cylinder(self.grasp, 'grasp', choice, self.geometry(), packet,
                self.frame(), self.cancel, 30., candidates)
            self.assertEqual(blocked.status, 'unsupported')
            self.assertEqual(blocked.steps, [])

    def test_side_approach_uses_tcp_and_target_not_camera_optical_z(self):
        packet = self.packet(self.frame())
        center = self.model.tip(self.raw)[:3, 3] + [.02, 0., 0.]
        candidates = _side_approaches(self.model, self.raw, packet.jaw_geometry, center, np.array([0., 0., 1.]))
        self.assertTrue(candidates)
        np.testing.assert_allclose(candidates[0], [1., 0., 0.], atol=1e-8)
        self.assertTrue(all(abs(axis[2]) <= .25 for axis in candidates))

    def test_bad_cylinder_curvature_is_not_renamed_a_grasp_center(self):
        geometry = self.geometry()
        geometry['observed_surface']['points'] = geometry['observed_surface']['points'][:8]
        result = self.plan(geometry)
        self.assertEqual(result.status, 'unsupported')
        self.assertIn('表面點', result.unsupported_reason)
        self.assertEqual(self.brain.calls, [])

    def test_vertical_closing_axis_rejects_before_visual_clearance(self):
        original = self.model.tip(self.raw)
        bad_contact = np.eye(4)
        bad_contact[:3, 0] = [0., 0., 1.]
        bad_contact[:3, 3] = original[:3, 3]
        with patch('server.spatial_object_program._contact_pose', return_value=bad_contact):
            result = self.plan()
        self.assertEqual(result.status, 'unsupported')
        self.assertIn('夾墊方向', result.unsupported_reason)
        self.assertEqual(self.brain.calls, [])

    def test_external_clearance_rejection_preserves_numerical_proposal(self):
        self.brain.reject = True
        result = self.plan()
        self.assertEqual(result.status, 'unsupported')
        self.assertIn('障礙', result.unsupported_reason)
        self.assertIn('proposal', self.grasp.spatial_attempt)

    def contact(self, frame):
        return {'state': 'contact_candidate', 'obstruction_candidate': True,
                'target_between_inner_pads': True, 'pwm_owner_active': True,
                'frame_id': frame['id'], 'camera_generation': frame['camera_generation'],
                'measured_jaw_raw': frame['joints']['gripper']}

    def lift(self, *, callback=None, contact=None, frames=None):
        raw = {**self.raw, 'gripper': 1780}
        self.grasp.robot.goals = {**raw, 'gripper': 1768}
        initial = self.frame(raw=raw)
        remaining = iter(frames or [self.frame(i, raw=raw) for i in (2, 3, 4)])
        if callback:
            self.brain.callback = callback
        with patch('server.spatial_object_program.observation_packet',
                   side_effect=lambda model, frame, config: self.packet(frame)):
            return prepare_contact_lift(self.grasp, '圓筒', initial,
                contact or self.contact(initial), self.cancel, capture_fresh=lambda: next(remaining))

    def test_post_close_lift_checks_held_path_without_empty_jaw_requirement(self):
        result = self.lift()
        self.assertEqual(result['segment']['measured_origin']['gripper'], 1780)
        self.assertEqual(result['segment']['held_goals']['gripper'], 1768)
        self.assertEqual(result['frame']['sequence'], 4)
        self.assertEqual(result['expected_pwm_owner'], 'owner-1')
        self.assertNotIn('gripper_empty', self.brain.calls[0][2].model_fields)
        self.assertIn('held_object_path_clear', result['assessment'])
        self.assertTrue(result['proposal']['diagnostics']['requires_new_external_assessment'])
        self.assertEqual(self.grasp.state, 'contact_candidate')
        self.assertEqual(self.grasp.robot.goals['gripper'], 1768)

    def test_lift_refuses_owner_change_during_assessment(self):
        with self.assertRaisesRegex(ObjectProgramBlocked, '持物或保持目標'):
            self.lift(callback=lambda: setattr(self.grasp.robot, 'pwm_owner', None))

    def test_lift_refuses_goal_change_during_assessment(self):
        with self.assertRaisesRegex(ObjectProgramBlocked, '持物或保持目標'):
            self.lift(callback=lambda: self.grasp.robot.goals.update(wrist_roll=1000))

    def test_lift_rejects_mismatched_checkpoint_before_model_call(self):
        initial = self.frame(raw={**self.raw, 'gripper': 1780})
        with self.assertRaisesRegex(ObjectProgramBlocked, '接觸記錄'):
            self.lift(contact={**self.contact(initial), 'frame_id': 'old-frame'})
        self.assertEqual(self.brain.calls, [])

    def test_lift_propagates_cancel_after_model_reply(self):
        with self.assertRaises(SessionCancelled):
            self.lift(callback=self.cancel.set)

    def test_lift_drift_is_compared_to_original_anchor(self):
        raw = {**self.raw, 'gripper': 1780}
        frames = [self.frame(2, raw=raw),
                  self.frame(3, raw={**raw, 'elbow_flex': raw['elbow_flex'] + 2}),
                  self.frame(4, raw={**raw, 'elbow_flex': raw['elbow_flex'] + 4})]
        with self.assertRaisesRegex(ValueError, '姿勢已改變'):
            self.lift(frames=frames)

    def test_lift_rejects_obstacle_in_new_held_object_path(self):
        self.brain.reject = True
        with self.assertRaisesRegex(ObjectProgramBlocked, '障礙'):
            self.lift()

    def test_refresh_reuses_original_encoder_anchor_and_never_accumulates_drift(self):
        before = self.frame(1)
        depth = {'source': 'wrist-only', 'current_frame_id': before['id'],
                 'current_measured_joints': dict(self.raw),
                 'provenance': {'camera_generation': 'camera-1'}}
        comparison = {'changed_fraction': 0., 'p99_difference': 0.}
        second = self.frame(2, raw={**self.raw, 'elbow_flex': self.raw['elbow_flex'] + 2})
        refreshed = refresh_depth_binding(depth, before, second, comparison)
        third = self.frame(3, raw={**self.raw, 'elbow_flex': self.raw['elbow_flex'] + 4})
        rejected = refresh_depth_binding(refreshed, second, third, comparison)
        self.assertEqual(rejected['stationary_refresh']['current_frame_id'], second['id'])
        self.assertEqual(rejected['stationary_refresh']['max_joint_drift_ticks'], 2)

    def test_legacy_refresh_without_initial_readback_does_not_reset_each_anchor(self):
        before = self.frame(1)
        depth = {'source': 'wrist-only', 'current_frame_id': before['id'],
                 'provenance': {'camera_generation': 'camera-1'}}
        comparison = {'changed_fraction': 0., 'p99_difference': 0.}
        second = self.frame(2, raw={**self.raw, 'elbow_flex': self.raw['elbow_flex'] + 2})
        refreshed = refresh_depth_binding(depth, before, second, comparison)
        third = self.frame(3, raw={**self.raw, 'elbow_flex': self.raw['elbow_flex'] + 4})
        self.assertEqual(refresh_depth_binding(refreshed, second, third, comparison), refreshed)

    def test_refresh_rejects_other_camera_and_unverified_comparison(self):
        depth = {'source': 'wrist-only', 'current_frame_id': 'frame-1',
                 'current_measured_joints': dict(self.raw),
                 'provenance': {'camera_generation': 'camera-1'}}
        with self.assertRaises(ValueError):
            refresh_depth_binding(depth, self.frame(1), self.frame(2, generation='other'),
                                  {'changed_fraction': 0., 'p99_difference': 0.})
        with self.assertRaises(ValueError):
            refresh_depth_binding(depth, self.frame(1), self.frame(2), {'okay': True})

    def test_current_packet_scope_overrides_valid_calibration_profile(self):
        packet = self.packet(self.frame())
        packet.metric_calibrated = False
        packet.to_dict = lambda: {'validation_scope_reason': '目前姿勢超出棋盤校準範圍。'}
        geometry = {'recommended_action': 'measure_target', 'message': '需要量測。'}
        with patch('server.spatial_object_program.observation_packet', return_value=packet), \
                patch('server.spatial_object_program.build_object_geometry', return_value=geometry):
            result = plan_spatial_object_program(self.grasp, '抓圓筒', self.frame(), self.cancel)
        self.assertEqual(result.status, 'unsupported')
        self.assertIn('抓取設定尚未完成', result.unsupported_reason)
        self.assertIn('目前手臂姿勢超出已校準範圍', result.unsupported_reason)
        self.assertEqual(self.grasp.spatial_attempt['required_setup'], 'camera_calibration')
        self.assertEqual(self.grasp.spatial_attempt['required_setups'], ['camera_calibration'])
        self.assertEqual(self.brain.calls, [])

    def test_camera_scope_and_missing_jaw_are_reported_together_without_inference(self):
        packet = self.packet(self.frame())
        packet.metric_calibrated = False; packet.jaw_geometry = None
        packet.to_dict = lambda: {'validation_scope_reason': 'outside measured workspace'}
        with patch('server.spatial_object_program.observation_packet', return_value=packet), \
                patch('server.spatial_object_program.build_object_geometry', return_value={'status': 'unmeasured'}):
            result = plan_spatial_object_program(self.grasp, '抓圓筒', self.frame(), self.cancel)
        self.assertEqual(result.status, 'unsupported')
        self.assertEqual(self.grasp.spatial_attempt['required_setups'], ['camera_calibration', 'jaw_calibration'])
        self.assertEqual(self.grasp.spatial_attempt['required_setup'], 'camera_calibration')
        self.assertIn('目前手臂姿勢超出已校準範圍', result.unsupported_reason)
        self.assertIn('兩側夾墊的位置與開口', result.unsupported_reason)
        self.assertEqual(self.brain.calls, [])
        self.assertEqual(self.events[-1]['required_setups'], ['camera_calibration', 'jaw_calibration'])

    def test_missing_jaw_is_reported_before_wasting_unmeasured_target_views(self):
        packet = self.packet(self.frame()); packet.jaw_geometry = None
        packet.to_dict = lambda: {'validation_scope_reason': ''}
        with patch('server.spatial_object_program.observation_packet', return_value=packet), \
                patch('server.spatial_object_program.build_object_geometry', return_value={'status': 'unmeasured'}):
            result = plan_spatial_object_program(self.grasp, '抓圓筒', self.frame(), self.cancel,
                observation_candidates=[{'id': 'move_left', 'descriptor': {}}])
        self.assertEqual(result.status, 'unsupported')
        self.assertEqual(self.grasp.spatial_attempt['required_setups'], ['jaw_calibration'])
        self.assertEqual(self.grasp.spatial_attempt['required_setup'], 'jaw_calibration')
        self.assertIn('抓取設定尚未完成', result.unsupported_reason)
        self.assertNotIn('已量到物件', result.unsupported_reason)
        self.assertEqual(self.brain.calls, [])

    def test_ready_setup_keeps_target_observation_and_clears_stale_missing_fields(self):
        packet = self.packet(self.frame()); packet.to_dict = lambda: {'validation_scope_reason': ''}
        observation = {'candidate_id': 'move_left', 'gripper_empty': True, 'human_clear': True,
                       'clear_path': True, 'confidence': .8, 'reason': '側移取得物件視差。'}
        self.grasp.spatial_attempt.update(required_setup='jaw_calibration',
            required_setups=['jaw_calibration'], calibration_page='/engineering/calibration')
        self.brain._infer = Mock(return_value={**self.intent.model_dump(), 'observation': observation})
        with patch('server.spatial_object_program.observation_packet', return_value=packet), \
                patch('server.spatial_object_program.build_object_geometry', return_value={'status': 'unmeasured'}):
            result = plan_spatial_object_program(self.grasp, '抓圓筒', self.frame(), self.cancel,
                observation_candidates=[{'id': 'move_left', 'descriptor': {}}])
        self.assertEqual(result.status, 'observe')
        self.assertEqual(result.observation.candidate_id, 'move_left')
        self.assertEqual(self.grasp.spatial_attempt['required_setups'], [])
        self.assertNotIn('required_setup', self.grasp.spatial_attempt)
        self.assertNotIn('calibration_page', self.grasp.spatial_attempt)
        self.brain._infer.assert_called_once()

    def test_unbound_timing_requests_settled_capture_not_new_calibration(self):
        packet = self.packet(self.frame())
        packet.metric_calibrated = False
        packet.to_dict = lambda: {'validation_scope_reason': '', 'timing': {'pose_bound': False}}
        geometry = {'recommended_action': 'measure_target', 'message': '需要量測。'}
        with patch('server.spatial_object_program.observation_packet', return_value=packet), \
                patch('server.spatial_object_program.build_object_geometry', return_value=geometry):
            result = plan_spatial_object_program(self.grasp, '抓圓筒', self.frame(), self.cancel)
        self.assertEqual(result.status, 'unsupported')
        self.assertIn('馬達讀值', result.unsupported_reason)
        self.assertEqual(self.grasp.spatial_attempt['required_setup'], 'settled_wrist_capture')
        self.assertEqual(self.grasp.spatial_attempt['required_setups'], ['settled_wrist_capture'])
        self.assertNotIn('calibration_page', self.grasp.spatial_attempt)

    def test_planned_endpoint_outside_calibration_domain_is_not_dispatched(self):
        packet = self.packet(self.frame())
        packet.calibration.contains_pose = lambda raw: raw == self.raw
        result = _plan_cylinder(self.grasp, 'grasp', self.intent, self.geometry(),
            packet, self.frame(), self.cancel, 30., ())
        self.assertEqual(result.status, 'unsupported')
        self.assertIn('超出棋盤校準', result.unsupported_reason)
        self.assertEqual(self.brain.calls, [])

    def test_lift_profile_change_during_inference_invalidates_plan(self):
        raw = {**self.raw, 'gripper': 1780}
        self.grasp.robot.goals = {**raw, 'gripper': 1768}
        initial = self.frame(raw=raw)
        frames = iter([self.frame(i, raw=raw) for i in (2, 3, 4)])
        first, changed = self.packet(self.frame(2, raw=raw)), self.packet(self.frame(4, raw=raw))
        changed.calibration.fingerprint = 'new-profile'
        with patch('server.spatial_object_program.observation_packet', side_effect=[first, changed]):
            with self.assertRaisesRegex(ObjectProgramBlocked, '校準已改變'):
                prepare_contact_lift(self.grasp, '圓筒', initial, self.contact(initial), self.cancel,
                    capture_fresh=lambda: next(frames))


if __name__ == '__main__':
    unittest.main()
