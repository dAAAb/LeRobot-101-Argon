"""Config/activation ownership boundaries; temp files, no lifespan or hardware."""
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from server.gripper_power import RecoveryRequired
from server.spatial_runtime import depth_calibration


class SpatialModeApiTests(unittest.TestCase):
    def setUp(self):
        self.app = importlib.import_module('server.app')
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / 'config').mkdir()
        self.config_path = self.root / 'config/robot.json'
        self.config = {'wrist_camera_id': 'wrist-A', 'camera_mount_revision': 'mount-A',
                       'wrist_capture_width': 640, 'spatial_backend': 'legacy',
                       'motion_backend': 'legacy', 'object_model_provider': 'geometry',
                       'custom_setting': 'preserve-me'}
        self.config_path.write_text(json.dumps(self.config, indent=2) + '\n')
        self.original_bytes = self.config_path.read_bytes()
        stop = threading.Event(); stop.set()
        self.hardware = Mock()
        self.power = SimpleNamespace(assert_motion_allowed=Mock())
        self.robot = SimpleNamespace(stop=stop, pwm_owner=None, moving=False, revision=8,
            gripper_power=self.power, config=self.config, connect=self.hardware.connect,
            hold=self.hardware.hold, write=self.hardware.write, resume=self.hardware.resume,
            arm=self.hardware.arm, torque=self.hardware.torque)
        self.grasp = SimpleNamespace(state='idle', model=object(), latest={'old': 'coordinates'},
                                     state_memory=object())
        self.camera = SimpleNamespace(lock=threading.RLock(), frame=np.zeros((480, 640, 3), dtype=np.uint8))
        self.session_id = 'a' * 32; self.solve_id = 'b' * 32
        self.store = Mock(); self.store.root = self.root / '.runtime/calibration-sessions'
        self.session = {'session_id': self.session_id, 'camera_id': 'wrist-A', 'mount_revision': 'mount-A',
            'geometry_revision': 'geometry-A', 'status': 'solved', 'image_size': [640, 480],
            'last_solve': {'solve_id': self.solve_id, 'status': 'validated'}}
        self.store.status.return_value = self.session
        self.profile = {'schema_version': 1, 'source': 'wrist-only', 'camera_id': 'wrist-A',
            'mount_revision': 'mount-A', 'geometry_revision': 'geometry-A', 'calibration_revision': 'camera-v1',
            'native_image_size': [640, 480], 'camera_matrix': [[480., 0, 319.5], [0, 480., 239.5], [0, 0, 1]],
            'distortion': [0.] * 5, 'handeye': np.eye(4).tolist(),
            'validated_components': ['intrinsics', 'handeye', 'robot_geometry'],
            'validation_kind': 'independent-heldout', 'validation_id': 'synthetic-heldout',
            'board_session_id': self.session_id,
            'validation_scope': {'joint_raw_bounds': {n: [1980, 2020] for n in
                ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')}}}
        self.solve_dir = self.store.root / self.session_id / 'solves' / self.solve_id
        self.solve_dir.mkdir(parents=True)
        self.write_candidate()
        memory_factory = Mock(return_value=object())
        for name, value in {'ROOT': self.root, 'robot': self.robot, 'grasp': self.grasp,
                'camera': self.camera, 'CONFIG': self.config, 'calibration_sessions': self.store,
                'active': None, 'jobs': {}, 'scenes': {'old': {'geometry': 'stale'}},
                'world_memory': object(), 'WorldStateMemory': memory_factory}.items():
            guard = patch.object(self.app, name, value); guard.start(); self.addCleanup(guard.stop)
        guard = patch('server.spatial_runtime.geometry_revision', return_value='geometry-A')
        guard.start(); self.addCleanup(guard.stop)
        self.client = TestClient(self.app.app)
        self.addCleanup(self.client.close)
        self.headers = {'x-lerobot-token': self.app.TOKEN}
        self.base_url = '/api/engineering/calibration'

    def write_candidate(self, profile=None, report=None):
        profile = self.profile if profile is None else profile
        (self.solve_dir / 'candidate-profile.json').write_text(json.dumps(profile))
        if report is None:
            report = {'status': 'validated', 'source': 'wrist-only', 'board_session_id': self.session_id,
                      'calibration_profile': profile}
        (self.solve_dir / 'report.json').write_text(json.dumps(report))

    def mode(self, mode, headers=None):
        return self.client.post(self.base_url + '/mode', json={'mode': mode},
                                headers=self.headers if headers is None else headers)

    def activate(self, solve_id=None):
        return self.client.post(self.base_url + f'/sessions/{self.session_id}/activate',
            json={'solve_id': solve_id or self.solve_id}, headers=self.headers)

    def assert_no_hardware(self):
        self.assertEqual(self.hardware.mock_calls, [])
        self.assertTrue(self.robot.stop.is_set())

    def test_mode_change_has_previous_snapshot_and_invalidates_cached_coordinates(self):
        response = self.mode('v2')
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result['motors_commanded'])
        saved = self.root / '.runtime/spatial-config-history' / (result['previous_config_id'] + '.json')
        self.assertEqual(saved.read_bytes(), self.original_bytes)
        self.assertEqual(json.loads(self.config_path.read_text()), self.config)
        self.assertEqual(self.config['motion_backend'], 'task_space_v2')
        self.assertEqual(self.config['custom_setting'], 'preserve-me')
        self.assertEqual(self.robot.revision, 9)
        self.assertIsNone(self.grasp.latest)
        self.assertEqual(self.app.scenes, {})
        self.assertIs(self.grasp.state_memory, self.app.world_memory)
        self.assertFalse(list((self.root / 'config').glob('*.tmp')))
        self.assert_no_hardware()

    def test_legacy_and_shadow_modes_really_bypass_selected_depth_profile(self):
        self.assertEqual(self.activate().status_code, 200)
        self.assertEqual(self.mode('v2').status_code, 200)
        for mode in ('legacy', 'shadow'):
            response = self.mode(mode)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(self.config['spatial_backend'], mode)
            self.assertEqual(self.config['motion_backend'], 'legacy')
            # No model/device/disk profile inspection required by rollback.
            self.assertIsNone(depth_calibration(object(), [], self.config))
        self.assert_no_hardware()

    def test_active_motion_stop_latch_and_contact_owner_block_both_changes(self):
        for state in ('running-job', 'moving', 'not-stopped', 'pwm-owner'):
            self.app.active = None; self.robot.moving = False
            self.robot.stop.set(); self.robot.pwm_owner = None
            if state == 'running-job':
                self.app.active = 'job'; self.app.jobs['job'] = {'status': 'running'}
            elif state == 'moving': self.robot.moving = True
            elif state == 'not-stopped': self.robot.stop.clear()
            else: self.robot.pwm_owner = object()
            with self.subTest(state=state):
                self.assertEqual(self.mode('v2').status_code, 409)
                self.assertEqual(self.activate().status_code, 409)
                self.assertEqual(self.config_path.read_bytes(), self.original_bytes)
        self.assertEqual(self.hardware.mock_calls, [])

    def test_contact_state_and_durable_recovery_barrier_block_ownerless_switch(self):
        for state in ('holding', 'holding_unverified', 'contact_candidate', 'releasing'):
            self.grasp.state = state
            self.assertEqual(self.mode('legacy').status_code, 409)
            self.assertEqual(self.activate().status_code, 409)
        self.grasp.state = 'idle'
        self.power.assert_motion_allowed.side_effect = RecoveryRequired('contact record remains')
        self.assertEqual(self.mode('v2').status_code, 409)
        self.assertEqual(self.activate().status_code, 409)
        self.assert_no_hardware()

    def test_disk_config_change_before_switch_is_not_overwritten(self):
        externally_updated = {**self.config, 'custom_setting': 'someone-else-edited'}
        self.config_path.write_text(json.dumps(externally_updated))
        response = self.mode('v2')
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(json.loads(self.config_path.read_text()), externally_updated)
        self.assertEqual(self.config['spatial_backend'], 'legacy')
        self.assertEqual(self.robot.revision, 8)
        self.assert_no_hardware()

    def test_disk_config_race_during_staging_preserves_other_writer(self):
        external = {**self.config, 'custom_setting': 'changed-during-staging'}
        real_write_text = Path.write_text
        def race(path, data, *args, **kwargs):
            value = real_write_text(path, data, *args, **kwargs)
            if path.parent == self.config_path.parent and path.suffix == '.tmp':
                real_write_text(self.config_path, json.dumps(external))
            return value
        with patch.object(Path, 'write_text', race):
            response = self.mode('v2')
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(json.loads(self.config_path.read_text()), external)
        self.assertEqual(self.config['spatial_backend'], 'legacy')
        self.assertEqual(self.robot.revision, 8)
        self.assertFalse(list((self.root / 'config').glob('*.tmp')))
        self.assert_no_hardware()

    def test_only_latest_validated_session_solve_may_activate(self):
        self.assertEqual(self.activate('c' * 32).status_code, 409)
        for changed in ({'status': 'collecting'}, {'last_solve': None},
                        {'last_solve': {'solve_id': self.solve_id, 'status': 'rejected'}}):
            self.store.status.return_value = {**self.session, **changed}
            self.assertEqual(self.activate().status_code, 409)
        self.assertEqual(self.config_path.read_bytes(), self.original_bytes)
        self.assert_no_hardware()

    def test_activation_accepts_matching_profile_but_does_not_move_or_change_mode(self):
        response = self.activate()
        self.assertEqual(response.status_code, 200, response.text)
        selected = self.root / self.config['spatial_calibration_profile']
        self.assertEqual(selected.parent, self.root / 'config/calibration')
        self.assertEqual(json.loads(selected.read_text()), self.profile)
        self.assertEqual(selected.read_bytes(), (self.solve_dir / 'candidate-profile.json').read_bytes())
        report = selected.with_name(selected.name.replace('-profile.json', '-report.json'))
        self.assertEqual(report.read_bytes(), (self.solve_dir / 'report.json').read_bytes())
        self.assertEqual(self.config['spatial_backend'], 'legacy')
        self.assertEqual(self.config['motion_backend'], 'legacy')
        self.assert_no_hardware()

    def test_selected_profile_loads_from_checkout_without_runtime_session(self):
        self.assertEqual(self.activate().status_code, 200)
        shutil.rmtree(self.store.root)
        from server.spatial_runtime import _read_profile
        with patch('server.spatial_runtime.ROOT', self.root):
            self.assertEqual(_read_profile(self.config, 'spatial_calibration_profile'), self.profile)
        self.assertEqual(len(list((self.root / 'config/calibration').iterdir())), 2)
        self.assert_no_hardware()

    def test_reactivation_reuses_identical_published_files_without_rewriting_them(self):
        self.assertEqual(self.activate().status_code, 200)
        files = list((self.root / 'config/calibration').iterdir())
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
        selected = self.config['spatial_calibration_profile']
        self.assertEqual(self.activate().status_code, 200)
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in files})
        self.assertEqual(self.config['spatial_calibration_profile'], selected)
        self.assert_no_hardware()

    def test_same_version_changed_profile_or_report_cannot_replace_published_artifacts(self):
        self.assertEqual(self.activate().status_code, 200)
        original_config = self.config_path.read_bytes()
        files = list((self.root / 'config/calibration').iterdir())
        before = {p.name: p.read_bytes() for p in files}
        for change in ('profile', 'report'):
            if change == 'profile':
                profile = deepcopy(self.profile)
                profile['camera_matrix'][0][0] += 1
                self.write_candidate(profile)
            else:
                self.write_candidate(report={'status': 'validated', 'source': 'wrist-only',
                    'board_session_id': self.session_id, 'calibration_profile': self.profile,
                    'validation': {'changed_after_publication': True}})
            with self.subTest(change=change):
                response = self.activate()
                self.assertEqual(response.status_code, 409, response.text)
                self.assertEqual(self.config_path.read_bytes(), original_config)
                self.assertEqual(before, {p.name: p.read_bytes() for p in files})
        self.assert_no_hardware()

    def test_report_publication_failure_keeps_active_config_and_can_retry_same_pair(self):
        real_link = os.link
        def fail_report(source, destination, *args, **kwargs):
            if str(destination).endswith('-report.json'):
                raise OSError('simulated report write failure')
            return real_link(source, destination, *args, **kwargs)
        with patch('server.app.os.link', side_effect=fail_report):
            response = self.activate()
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.config_path.read_bytes(), self.original_bytes)
        self.assertNotIn('spatial_calibration_profile', self.config)
        self.assertEqual(self.robot.revision, 8)
        self.assertFalse(list((self.root / 'config/calibration').glob('*.tmp')))
        self.assertEqual(self.activate().status_code, 200)
        self.assertEqual(len(list((self.root / 'config/calibration').iterdir())), 2)
        self.assert_no_hardware()

    def test_existing_conflicting_report_is_rejected_before_publishing_profile(self):
        published = self.root / 'config/calibration'; published.mkdir()
        conflict = published / f'{self.session_id}-{self.solve_id}-report.json'
        conflict.write_text('{"different":"report"}')
        response = self.activate()
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(list(published.iterdir()), [conflict])
        self.assertEqual(self.config_path.read_bytes(), self.original_bytes)
        self.assert_no_hardware()

    def test_profile_must_equal_its_validated_report_and_be_wrist_only(self):
        report = {'status': 'validated', 'calibration_profile': {**self.profile, 'handeye': (np.eye(4) * 2).tolist()}}
        self.write_candidate(report=report)
        self.assertEqual(self.activate().status_code, 409)
        self.write_candidate({**self.profile, 'source': 'd435'})
        self.assertEqual(self.activate().status_code, 409)
        self.write_candidate({**self.profile, 'validated_components': ['intrinsics', 'handeye']})
        self.assertEqual(self.activate().status_code, 409)
        self.assert_no_hardware()

    def test_session_and_profile_must_match_current_camera_mount_model_and_resolution(self):
        for key in ('camera_id', 'mount_revision', 'geometry_revision'):
            self.store.status.return_value = {**self.session, key: 'another'}
            self.assertEqual(self.activate().status_code, 409)
            self.store.status.return_value = self.session
            self.write_candidate({**self.profile, key: 'another'})
            self.assertEqual(self.activate().status_code, 409)
            self.write_candidate()
        self.store.status.return_value = {**self.session, 'image_size': [640, 360]}
        self.assertEqual(self.activate().status_code, 409)
        self.store.status.return_value = self.session
        self.camera.frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.assertEqual(self.activate().status_code, 409)
        self.assert_no_hardware()

    def test_token_validation_and_invalid_mode_do_not_change_state(self):
        self.assertEqual(self.mode('v2', headers={}).status_code, 403)
        self.assertEqual(self.mode('unknown').status_code, 422)
        self.assertEqual(self.activate('../bad').status_code, 422)
        self.assertEqual(self.config_path.read_bytes(), self.original_bytes)
        self.assert_no_hardware()


if __name__ == '__main__':
    unittest.main()
