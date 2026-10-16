"""Fixed small domain slack keeps measured bounds and charges FK sensitivity."""
from types import SimpleNamespace
import unittest

import numpy as np

import test_spatial_runtime as fixtures
from server import spatial_runtime as runtime
from server.spatial_contract import build_wrist_observation
from server.spatial_object_program import _check_calibrated_path
from server.task_space_plan import _hash


class CalibrationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SpatialRuntimeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model, self.config = self.fixture.model, self.fixture.config

    def test_four_tick_boundary_uses_actual_fk_and_extra_uncertainty_without_rebinding(self):
        before = self.fixture.profile_path.read_bytes()
        strict = runtime.observation_packet(self.model, self.fixture.frame(), self.config)
        frame = self.fixture.frame(2, 2014)
        packet = runtime.observation_packet(self.model, frame, self.config)
        self.assertTrue(packet.metric_calibrated)
        public = packet.to_dict()
        self.assertFalse(public['inside_validation_joint_bounds'])
        self.assertEqual(public['validation_scope_status'], 'near-boundary-extrapolation')
        self.assertAlmostEqual(packet.T_base_camera.as_array()[0, 3], .014)
        self.assertAlmostEqual(public['calibration_boundary_reserve']['translation_m'], .004)
        self.assertAlmostEqual(public['calibration_uncertainty']['relative_translation_uncertainty_m'], .012)
        self.assertAlmostEqual(public['calibration_uncertainty']['base_position_uncertainty_m'], .004)
        self.assertEqual(strict.calibration.fingerprint, packet.calibration.fingerprint)
        self.assertEqual(before, self.fixture.profile_path.read_bytes())
        adapter = runtime.depth_calibration(self.model, [self.fixture.frame(), frame], self.config)
        self.assertTrue(adapter['metric_calibrated'])
        self.assertAlmostEqual(adapter['relative_translation_uncertainty_m'], .012)
        self.assertEqual(adapter['validation_scope'], self.fixture.profile['validation_scope'])
        bounds = runtime.observation_pose_bounds(self.model, frame,
            {**self.config, 'motion_backend': 'task_space_v2'})
        self.assertEqual(bounds['shoulder_pan'], [1982, 2018])
        # Repeated observations never use their own accepted pose as a new origin.
        self.assertFalse(runtime.observation_packet(self.model, self.fixture.frame(3, 2019), self.config).metric_calibrated)
        self.assertEqual(packet.calibration.accepted_pose_bounds()['shoulder_pan'], [1982, 2018])

    def test_direct_packet_without_model_reserve_does_not_promote_boundary_pose(self):
        frame = self.fixture.frame(2, 2014)
        packet = build_wrist_observation(frame, self.model.fk.transforms(frame['joints']),
            geometry_revision=self.fixture.revision, calibration=self.fixture.profile,
            camera_id='wrist-id', mount_revision='mount-v1')
        self.assertTrue(packet.calibration.contains_pose(frame['joints']))
        self.assertFalse(packet.metric_calibrated)
        self.assertIn('尚未計算', packet.to_dict()['validation_scope_reason'])

    def test_future_tcp_path_reserve_changes_budget_and_sealed_proposal(self):
        # The optical centre does not move under pure rotation, but the TCP does.
        def poses(raw):
            angle = (raw['elbow_flex'] - 2000) * .001
            c, s = np.cos(angle), np.sin(angle)
            t = np.array([[c, -s, 0., 0.], [s, c, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
            return {'gripper_link': t, 'wrist_camera_optical': t}
        self.model.fk.transforms = poses
        packet = runtime.observation_packet(self.model, self.fixture.frame(), self.config)
        tcp = np.eye(4); tcp[0, 3] = .1
        wrapped = SimpleNamespace(calibration=packet.calibration,
            jaw_geometry=SimpleNamespace(T_wrist_tcp=SimpleNamespace(as_array=lambda: tcp)))
        raw = self.fixture.frame()['joints']; target = {**raw, 'elbow_flex': 2014}
        proposal = dict(segments=[dict(contract=dict(measured_origin=raw, joint_targets=target,
            sampled_path=[{'fraction': 0.}, {'fraction': .75}, {'fraction': 1.}]))],
            binding=dict(target=dict(position_budget_m=.001, available_clearance_m=.0012), evidence={}),
            execution_blockers=[], metric_inputs_verified=True, ready_for_external_assessment=True)
        _check_calibrated_path(wrapped, proposal, model=self.model)
        value = proposal['binding']['target']['calibration_boundary_position_reserve_m']
        self.assertAlmostEqual(value, 2 * .1 * np.sin(.004 / 2))
        self.assertGreater(proposal['binding']['evidence']['calibration_scope_path']['maximum_rotation_reserve_deg'], 0.)
        self.assertAlmostEqual(proposal['binding']['target']['position_budget_m'], .001 + value)
        self.assertFalse(proposal['ready_for_external_assessment'])
        self.assertEqual(proposal['input_sha256'], _hash(proposal['binding']))
        self.assertEqual(proposal['plan_sha256'], _hash({k: v for k, v in proposal.items() if k != 'plan_sha256'}))
        # The same checker remains idempotent, not accumulating a new allowance.
        _check_calibrated_path(wrapped, proposal, model=self.model)
        self.assertAlmostEqual(proposal['binding']['target']['position_budget_m'], .001 + value)
        proposal['segments'][0]['contract']['joint_targets']['elbow_flex'] = 2019
        with self.assertRaisesRegex(ValueError, '超出棋盤校準'):
            _check_calibrated_path(wrapped, proposal, model=self.model)


if __name__ == '__main__':
    unittest.main()
