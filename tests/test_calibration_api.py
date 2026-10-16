"""Calibration ownership boundaries; never start ASGI lifespan or hardware."""
import importlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class CalibrationApiTests(unittest.TestCase):
    def setUp(self):
        self.app = importlib.import_module('server.app')
        self.queue = []
        self.robot = SimpleNamespace(stop=threading.Event(), pwm_owner=None, connect=Mock())
        self.frame = {'id': 'measured-frame', 'source': 'wrist', 'joints': {'gripper': 1800}}
        self.grasp = SimpleNamespace(model=object(), capture=Mock(return_value=self.frame))
        self.summary = {'session_id': 'a' * 32, 'geometry_revision': 'model-rev',
                        'message': '已保存', 'collected_view_count': 1}
        self.store = Mock()
        self.store.status.return_value = self.summary
        self.store.start.return_value = self.summary
        self.store.capture.return_value = self.summary
        self.store.solve.return_value = {**self.summary, 'active_profile_changed': False}
        self.store.list_sessions.return_value = [self.summary]
        for name, value in {'robot': self.robot, 'grasp': self.grasp,
                'calibration_sessions': self.store, 'active': None, 'jobs': {},
                'executor': SimpleNamespace(submit=self.queue.append),
                'snapshot_owner': threading.local(),
                'CONFIG': {'wrist_camera_id': 'wrist-usb'}}.items():
            guard = patch.object(self.app, name, value); guard.start(); self.addCleanup(guard.stop)
        guard = patch('server.spatial_runtime.geometry_revision', return_value='model-rev')
        guard.start(); self.addCleanup(guard.stop)
        self.client = TestClient(self.app.app)
        self.addCleanup(self.client.close)
        self.headers = {'x-lerobot-token': self.app.TOKEN}
        self.url = '/api/engineering/calibration/sessions/' + 'a' * 32

    def post(self, suffix, payload=None):
        return self.client.post(self.url + suffix, json=payload, headers=self.headers)

    def finish(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        self.queue.pop(0)()
        return self.client.get('/api/jobs/' + response.json()['id']).json()

    def test_read_and_start_never_capture_or_connect(self):
        result = self.client.get('/api/engineering/calibration').json()
        self.assertFalse(result['spatial']['observer_in_production'])
        response = self.client.post('/api/engineering/calibration/sessions',
            json={'board_static': True, 'print_scale_checked': True}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.robot.connect.assert_not_called(); self.grasp.capture.assert_not_called()
        self.assertEqual(self.store.start.call_args.kwargs['camera_id'], 'wrist-usb')

    def test_capture_is_single_owner_and_binds_configured_camera(self):
        accepted = self.post('/capture', {'board_static': True})
        self.assertEqual(self.post('/capture', {'board_static': True}).status_code, 409)
        result = self.finish(accepted)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['calibration_result']['collected_view_count'], 1)
        frame = self.store.capture.call_args.args[1]
        self.assertEqual(frame['camera_id'], 'wrist-usb')
        self.assertNotIn('camera_id', self.frame)  # Does not mutate shared camera cache.

    def test_stop_and_contact_owner_block_capture_without_rearming(self):
        self.robot.stop.set()
        self.assertEqual(self.post('/capture', {'board_static': True}).status_code, 409)
        self.robot.stop.clear(); self.robot.pwm_owner = object()
        self.assertEqual(self.post('/capture', {'board_static': True}).status_code, 409)
        self.grasp.capture.assert_not_called(); self.robot.connect.assert_not_called()

    def test_cancelled_capture_never_reads_device_or_saves_image(self):
        response = self.post('/capture', {'board_static': True})
        self.app.jobs[response.json()['id']]['cancel'].set()
        result = self.finish(response)
        self.assertEqual(result['status'], 'cancelled')
        self.robot.connect.assert_not_called(); self.store.capture.assert_not_called()

    def test_solve_while_stopped_has_no_device_access_or_activation(self):
        self.robot.stop.set()
        result = self.finish(self.post('/solve'))
        self.assertEqual(result['status'], 'completed')
        self.assertFalse(result['calibration_result']['active_profile_changed'])
        self.robot.connect.assert_not_called(); self.grasp.capture.assert_not_called()

    def test_wrong_geometry_is_rejected_before_queue(self):
        self.store.status.return_value = {**self.summary, 'geometry_revision': 'other'}
        self.assertEqual(self.post('/capture', {'board_static': True}).status_code, 409)
        self.assertEqual(self.post('/solve').status_code, 409)
        self.assertFalse(self.queue)

    def test_workspace_extension_is_offline_and_does_not_activate(self):
        self.robot.stop.set()
        identity={'camera_id':'wrist-usb','mount_revision':'existing-mount-unverified','geometry_revision':'model-rev'}
        self.store.status.return_value={**self.summary,**identity}
        self.app.CONFIG['spatial_calibration_profile']='selected.json'
        with patch('server.spatial_runtime._read_profile',return_value=identity):
            result=self.finish(self.post('/extend'))
        self.assertEqual(result['status'],'completed')
        self.assertEqual(self.store.solve.call_args.kwargs['frozen_profile'],identity)
        self.robot.connect.assert_not_called();self.grasp.capture.assert_not_called()
        self.assertEqual(self.app.CONFIG['spatial_calibration_profile'],'selected.json')

    def test_workspace_extension_rejects_missing_or_mismatched_parent(self):
        self.assertEqual(self.post('/extend').status_code,409)
        self.app.CONFIG['spatial_calibration_profile']='selected.json'
        self.store.status.return_value={**self.summary,'camera_id':'wrist-usb','mount_revision':'existing-mount-unverified'}
        with patch('server.spatial_runtime._read_profile',return_value={'camera_id':'another-camera'}):
            self.assertEqual(self.post('/extend').status_code,409)
        self.assertFalse(self.queue)

    def test_explicit_print_and_fixed_board_declaration_and_token_required(self):
        for payload in ({}, {'board_static': True}, {'board_static': False, 'print_scale_checked': True}):
            response = self.client.post('/api/engineering/calibration/sessions', json=payload, headers=self.headers)
            self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.post(self.url + '/capture', json={'board_static': True}).status_code, 403)


if __name__ == '__main__':
    unittest.main()
