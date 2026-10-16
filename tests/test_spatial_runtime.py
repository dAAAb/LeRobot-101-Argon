"""Actual runtime profile selection and depth adapter, no devices/services."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from server import spatial_runtime as runtime
from server.active_depth import _camera_model
from server.object_geometry import build_object_geometry
from server.spatial_contract import JOINTS


class SpatialRuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        urdf = self.root / 'test.urdf'; urdf.write_text('<robot name="synthetic" />')
        def poses(raw):
            t = np.eye(4); t[0, 3] = (raw['shoulder_pan'] - 2000) * .001
            return {'gripper_link': t, 'wrist_camera_optical': t}
        self.model = SimpleNamespace(urdf_path=urdf, calibration={n: {'range_min': 0, 'range_max': 4095} for n in JOINTS},
                                     fk=SimpleNamespace(transforms=poses))
        self.revision = runtime.geometry_revision(self.model)
        self.profile_path = self.root / 'camera.json'
        self.profile = {'schema_version': 1, 'source': 'wrist-only', 'calibration_revision': 'camera-v1',
            'geometry_revision': self.revision, 'camera_id': 'wrist-id', 'mount_revision': 'mount-v1',
            'native_image_size': [640, 480], 'camera_matrix': [[480., 0, 319.5], [0, 480., 239.5], [0, 0, 1]],
            'distortion': [0.] * 5, 'handeye': np.eye(4).tolist(),
            'validated_components': ['intrinsics', 'handeye', 'robot_geometry'],
            'validation_id': 'synthetic-heldout', 'validation_kind': 'independent-heldout',
            'validation_scope': {'joint_raw_bounds': {n: [1990, 2010] for n in JOINTS[:-1]}},
            'relative_translation_uncertainty_m': .004, 'relative_rotation_uncertainty_deg': .2,
            'focal_relative_uncertainty': .01, 'metric_calibrated': True}
        self.save_profile(self.profile)
        self.config = {'spatial_backend': 'v2', 'spatial_calibration_profile': str(self.profile_path),
                       'wrist_camera_id': 'wrist-id', 'camera_mount_revision': 'mount-v1'}

    def save_profile(self, profile):
        self.profile_path.write_text(json.dumps(profile))

    def frame(self, sequence=1, shoulder=2000, **changes):
        joints = {n: 2000 for n in JOINTS}; joints['shoulder_pan'] = shoulder
        stamp = sequence + 1.
        value = {'source': 'wrist', 'id': f'frame-{sequence}', 'image_size': [640, 480],
                 'camera_generation': 'session-A', 'sequence': sequence, 'capture_monotonic': stamp, 'joints': joints,
                 'joint_observation_bracket_monotonic': [stamp - .05, stamp + .05],
                 'joint_capture_span_ticks': 0, 'image_wait_joint_span_ticks': 0,
                 'settled_joint_window': [
                     {'joints': dict(joints), 'read_start_monotonic': stamp - .5, 'read_end_monotonic': stamp - .4},
                     {'joints': dict(joints), 'read_start_monotonic': stamp - .3, 'read_end_monotonic': stamp - .2}]}
        value.update(changes)
        return value

    def test_no_profile_legacy_and_shadow_keep_existing_depth_path(self):
        with patch.object(runtime, '_read_profile', side_effect=AssertionError('must not read selected v2 profile')):
            self.assertIsNone(runtime.depth_calibration(self.model, [self.frame()], {}))
            self.assertIsNone(runtime.depth_calibration(self.model, [self.frame()], {'spatial_backend': 'v2'}))
            for mode in ('legacy', 'shadow'):
                self.assertIsNone(runtime.depth_calibration(self.model, [self.frame()], {**self.config, 'spatial_backend': mode}))

    def test_incompatible_motion_and_spatial_modes_fail_configuration(self):
        for mode in ('legacy', 'shadow'):
            with self.assertRaisesRegex(ValueError, 'requires spatial_backend=v2'):
                runtime.depth_calibration(self.model, [self.frame()],
                    {**self.config, 'spatial_backend': mode, 'motion_backend': 'task_space_v2'})

    def test_adapter_preserves_measured_uncertainty_and_exact_packet_fingerprint(self):
        frames = [self.frame(), self.frame(2, 2005)]
        adapter = runtime.depth_calibration(self.model, frames, self.config)
        packet = runtime.observation_packet(self.model, frames[-1], self.config)
        transforms = [self.model.fk.transforms(f['joints']) for f in frames]
        _, k, d, uncertainty, provenance = _camera_model((640, 480), transforms, 'session-A', adapter)
        self.assertTrue(provenance['metric_calibrated'])
        self.assertEqual(provenance['calibration_fingerprint'], packet.calibration.fingerprint)
        self.assertEqual(provenance['geometry_revision'], self.revision)
        self.assertEqual(uncertainty['relative_translation_m'], .004)
        np.testing.assert_array_equal(k, self.profile['camera_matrix'])
        self.assertEqual(adapter['camera_generation'], 'session-A')

    def test_scope_failure_overwrites_stale_legacy_true_flags_and_metric_field(self):
        profile = {**self.profile, 'intrinsics_validated': True, 'handeye_validated': True,
                   'robot_geometry_validated': True, 'metric_calibrated': True}
        self.save_profile(profile)
        frames = [self.frame(1, 2020), self.frame(2, 2000)]
        # Even when current is in scope, every triangulation contributor matters.
        adapter = runtime.depth_calibration(self.model, frames, self.config)
        self.assertFalse(adapter['metric_calibrated'])
        self.assertTrue(all(adapter[k] is False for k in
                           ('intrinsics_validated', 'handeye_validated', 'robot_geometry_validated')))
        *_, provenance = _camera_model((640, 480), [self.model.fk.transforms(f['joints']) for f in frames],
                                      'session-A', adapter)
        self.assertFalse(provenance['metric_calibrated'])

    def test_source_geometry_device_mount_and_sessions_do_not_mix(self):
        with self.assertRaisesRegex(ValueError, 'wrist camera'):
            runtime.observation_packet(self.model, self.frame(source='d435'), self.config)
        with self.assertRaisesRegex(ValueError, 'captured wrist device'):
            runtime.observation_packet(self.model, self.frame(camera_id='observer'), self.config)
        for key, value in (('source', 'd435'), ('geometry_revision', 'another-model'),
                           ('camera_id', 'another-device'), ('mount_revision', 'remounted')):
            self.save_profile({**self.profile, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.depth_calibration(self.model, [self.frame()], self.config)
        self.save_profile(self.profile)
        with self.assertRaisesRegex(ValueError, 'sessions'):
            runtime.depth_calibration(self.model, [self.frame(), self.frame(2, camera_generation='session-B')], self.config)

    def test_camera_artifact_is_read_once_even_if_replaced_during_packet_build(self):
        frames = [self.frame(), self.frame(2, 2005)]
        original_packet = runtime.observation_packet(self.model, frames[-1], self.config)
        changed = deepcopy(self.profile)
        changed['camera_matrix'][0][0] += 40
        changed['relative_translation_uncertainty_m'] = .009
        real_packet = runtime._packet
        def replace_file_after_first_packet(model, frame, snapshot):
            value = real_packet(model, frame, snapshot)
            self.save_profile(changed)
            return value
        with patch.object(runtime, '_packet', side_effect=replace_file_after_first_packet), \
                patch.object(runtime, '_read_profile', wraps=runtime._read_profile) as read:
            adapter = runtime.depth_calibration(self.model, frames, self.config)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(adapter['camera_matrix'], self.profile['camera_matrix'])
        self.assertEqual(adapter['relative_translation_uncertainty_m'], .004)
        self.assertEqual(adapter['calibration_fingerprint'], original_packet.calibration.fingerprint)
        self.assertNotEqual(runtime.observation_packet(self.model, frames[-1], self.config).calibration.fingerprint,
                            adapter['calibration_fingerprint'])

    def test_jaw_artifact_is_not_a_dependency_of_camera_triangulation(self):
        config = {**self.config, 'jaw_calibration_profile': str(self.root / 'not-calibrated-yet.json')}
        adapter = runtime.depth_calibration(self.model, [self.frame(), self.frame(2)], config)
        self.assertTrue(adapter['metric_calibrated'])

    def test_new_profile_selection_cannot_upgrade_depth_with_old_fingerprint(self):
        frame = self.frame()
        old = runtime.observation_packet(self.model, frame, self.config).to_dict()
        profile = deepcopy(self.profile); profile['relative_translation_uncertainty_m'] = .009
        self.save_profile(profile)
        current = runtime.observation_packet(self.model, frame, self.config).to_dict()
        depth = {'source': 'wrist-only', 'status': 'estimated', 'scope': 'object_roi_surface',
            'current_frame_id': frame['id'], 'metric_calibrated': True,
            'provenance': {'camera_generation': frame['camera_generation'], 'metric_calibrated': True,
                'camera_model': 'validated-calibration', 'calibration_revision': old['calibration_revision'],
                'calibration_fingerprint': old['calibration_fingerprint'], 'geometry_revision': old['geometry_revision']},
            'distance': {'position_base_m': [0., 0., .3], 'camera_optical_axis_m': .3},
            'uncertainty': {'base_position_error_bound_m': .01},
            'points': [{'position_base_m': [i * .001, 0., .3], 'views': 3} for i in range(6)]}
        result = build_object_geometry(depth, current_frame=frame, track_id='target-1', observation=current)
        self.assertEqual(result['status'], 'observed')
        self.assertFalse(result['metric_calibrated'])
        self.assertEqual(result['recommended_action'], 'calibrate_camera')


if __name__ == '__main__':
    unittest.main()
