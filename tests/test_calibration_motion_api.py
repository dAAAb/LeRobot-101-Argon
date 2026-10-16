"""Engineering motion HTTP ownership, queue and one-shot batch boundaries.

No ASGI lifespan or devices: numerical candidate generation and the final
observation executor are mocked, while the real batch/approval guard executes.
"""
from copy import deepcopy
import importlib
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from server import calibration_motion as motion


class CalibrationMotionApiTests(unittest.TestCase):
    def setUp(self):
        self.app = importlib.import_module('server.app')
        self.session_id = 'a' * 32
        self.queue = []
        self.raw = dict(shoulder_pan=2100, shoulder_lift=1400, elbow_flex=3000,
                        wrist_flex=2000, wrist_roll=1700, gripper=1850)
        self.frame = dict(id='frame-1', source='wrist', camera_generation='generation-1',
                          path='/unused/wrist.jpg', sequence=1, capture_monotonic=time.monotonic(),
                          joints=dict(self.raw), image_size=[1920, 1080])
        self.robot = SimpleNamespace(stop=threading.Event(), pwm_owner=None,
            moving=False, connect=Mock(), begin=Mock(), goals=dict(self.raw),
            gripper_power=SimpleNamespace(assert_motion_allowed=Mock()))
        self.grasp = SimpleNamespace(model=object(), state='idle', robot=self.robot,
                                     capture=Mock(side_effect=lambda: deepcopy(self.frame)), record=Mock())
        self.session = dict(session_id=self.session_id, status='collecting',
            camera_id='wrist-usb', mount_revision='mount-1', geometry_revision='model-1',
            board_static_declared=True, camera_generation=None, collected_view_count=0)
        self.store = Mock()
        self.store.status.side_effect = lambda _: deepcopy(self.session)
        self.saved_origin = None
        def motion_origin(session_id, frame):
            if self.saved_origin is None:
                self.saved_origin = dict(frame['joints'])
            return {'generation': frame['camera_generation'], 'joints': dict(self.saved_origin)}
        self.store.motion_origin.side_effect = motion_origin
        self.store.capture.return_value = {**self.session, 'accepted': True,
            'collected_view_count': 1, 'message': '已保存校準照片'}
        self.build = Mock(side_effect=self.make_batch)
        self.execute = Mock(side_effect=self.completed_motion)
        for name, value in {'robot': self.robot, 'grasp': self.grasp,
                'calibration_sessions': self.store, 'active': None, 'jobs': {},
                'executor': SimpleNamespace(submit=self.queue.append),
                'snapshot_owner': threading.local(), 'calibration_view_batches': {},
                'calibration_view_origins': {},
                'CONFIG': {'wrist_camera_id': 'wrist-usb', 'camera_mount_revision': 'mount-1'}}.items():
            guard = patch.object(self.app, name, value); guard.start(); self.addCleanup(guard.stop)
        for target, value in [('server.spatial_runtime.geometry_revision', lambda _: 'model-1'),
                ('server.calibration_motion.geometry_revision', lambda _: 'model-1'),
                ('server.calibration_motion.build_calibration_candidates', self.build),
                ('server.active_observation.execute_observation', self.execute)]:
            guard = patch(target, value); guard.start(); self.addCleanup(guard.stop)
        self.client = TestClient(self.app.app)
        self.addCleanup(self.client.close)
        self.headers = {'x-lerobot-token': self.app.TOKEN}
        self.base = '/api/engineering/calibration/views'

    def make_batch(self, grasp, frame, *, origin, session_id, cancel, rotations_only=False):
        now = time.monotonic()
        candidate = {'id': 'turn-wrist', 'contract_sha256': 'candidate-hash',
                     'descriptor': {'candidate_id': 'turn-wrist', 'label': '旋轉手腕 +8°'}}
        batch = dict(version=1, kind='engineering_calibration_motion', session_id=session_id,
            origin=deepcopy(origin), held_goals=dict(self.robot.goals), reference=deepcopy(frame),
            geometry_revision='model-1', issued_monotonic=now, expires_monotonic=now + 90,
            candidates=[candidate], rejections=[])
        batch['contract_sha256'] = motion._fingerprint(batch)
        return batch

    def completed_motion(self, grasp, candidate, reference, cancel, report, **kwargs):
        after = {**deepcopy(reference), 'id': 'after-motion', 'sequence': reference['sequence'] + 1}
        return dict(completed=True, before=reference, after=after, physical_seconds_charged=1.2)

    def plan(self):
        return self.client.post(self.base + '/plan', json={'session_id': self.session_id}, headers=self.headers)

    def finish(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        self.queue.pop(0)()
        return self.client.get('/api/jobs/' + response.json()['id']).json()

    def planned(self):
        job = self.finish(self.plan())
        self.assertEqual(job['status'], 'completed', job)
        return job['calibration_motion']

    def dispatch(self, presented, *, approval=None, candidate_id='turn-wrist'):
        approval = approval if approval is not None else dict(approved=True,
            batch_sha256=presented['batch_sha256'],
            candidate_sha256=presented['candidates'][0]['candidate_sha256'],
            frame_id=presented['frame_id'], gripper_empty=True, human_clear=True, clear_path=True,
            reason='已核對這一張腕部畫面及完整候選路徑')
        return self.client.post(self.base + '/execute', headers=self.headers,
            json={'batch_id': presented['batch_id'], 'candidate_id': candidate_id, 'approval': approval})

    def test_plan_is_queued_and_publishes_wrist_only_review_without_motion(self):
        response = self.plan()
        self.assertEqual(response.json()['status'], 'running')
        self.build.assert_not_called(); self.grasp.capture.assert_not_called()
        job = self.finish(response)
        public = job['calibration_motion']
        self.assertEqual(public['source'], 'wrist-only')
        self.assertTrue(public['requires_explicit_path_review'])
        self.assertNotIn('path', public['frame'])
        self.robot.begin.assert_called_once_with(allow_rearm=False)
        self.execute.assert_not_called(); self.store.capture.assert_not_called()

    def test_plan_can_request_rotation_views_without_translation_ik(self):
        response = self.client.post(self.base + '/plan',
            json={'session_id': self.session_id, 'rotations_only': True}, headers=self.headers)
        self.assertEqual(self.finish(response)['status'], 'completed')
        self.assertIs(self.build.call_args.kwargs['rotations_only'], True)
        self.execute.assert_not_called()

    def test_success_consumes_batch_and_captures_measured_post_motion_frame(self):
        presented = self.planned()
        response = self.dispatch(presented)
        self.assertNotIn(presented['batch_id'], self.app.calibration_view_batches)
        self.assertIn(self.dispatch(presented).status_code, (404, 409))
        job = self.finish(response)
        self.assertEqual(job['status'], 'completed', job)
        self.assertTrue(job['calibration_motion_result']['completed'])
        self.assertEqual(job['calibration_result']['collected_view_count'], 1)
        self.execute.assert_called_once()
        saved = self.store.capture.call_args.args[1]
        self.assertEqual(saved['id'], 'after-motion')
        self.assertEqual(saved['camera_id'], 'wrist-usb')
        self.assertEqual(self.store.capture.call_args.kwargs, {'board_static': True})

    def test_successful_motion_still_published_when_board_is_not_visible(self):
        self.store.capture.return_value = {**self.session, 'accepted': False,
            'collected_view_count': 0, 'message': '棋盤沒有完整入鏡'}
        job = self.finish(self.dispatch(self.planned()))
        self.assertEqual(job['status'], 'completed')
        self.assertTrue(job['calibration_motion_result']['completed'])
        self.assertFalse(job['calibration_result']['accepted'])

    def test_board_storage_error_preserves_physical_motion_receipt(self):
        self.store.capture.side_effect = ValueError('棋盤工作階段不符')
        job = self.finish(self.dispatch(self.planned()))
        self.assertEqual(job['status'], 'failed')
        self.assertTrue(job['calibration_motion_result']['completed'])
        self.assertNotIn('calibration_result', job)

    def test_stop_contact_and_other_job_block_dispatch_without_consuming_batch(self):
        presented = self.planned()
        self.robot.stop.set()
        self.assertEqual(self.dispatch(presented).status_code, 409)
        self.robot.stop.clear(); self.robot.pwm_owner = object()
        self.assertEqual(self.dispatch(presented).status_code, 409)
        self.robot.pwm_owner = None
        pending = self.plan()
        self.assertEqual(pending.status_code, 200)
        self.assertEqual(self.dispatch(presented).status_code, 409)
        self.assertIn(presented['batch_id'], self.app.calibration_view_batches)
        self.execute.assert_not_called()

    def test_moving_robot_cannot_accept_calibration_plan(self):
        self.robot.moving = True
        self.assertEqual(self.plan().status_code, 409)
        self.assertFalse(self.queue)

    def test_moving_robot_cannot_consume_reviewed_batch(self):
        presented = self.planned()
        self.robot.moving = True
        self.assertEqual(self.dispatch(presented).status_code, 409)
        self.assertIn(presented['batch_id'], self.app.calibration_view_batches)
        self.assertFalse(self.queue)
        self.execute.assert_not_called()

    def test_plan_checks_session_and_camera_configuration_before_queue(self):
        for field, value in [('status', 'invalidated'), ('board_static_declared', False),
                             ('camera_id', 'external'), ('mount_revision', 'other'), ('geometry_revision', 'other')]:
            old = self.session[field]; self.session[field] = value
            self.assertEqual(self.plan().status_code, 409)
            self.session[field] = old
        self.build.assert_not_called()

    def test_queued_session_change_cannot_execute_motion(self):
        response = self.dispatch(self.planned())
        self.session['status'] = 'invalidated'
        job = self.finish(response)
        self.assertEqual(job['status'], 'failed')
        self.execute.assert_not_called(); self.store.capture.assert_not_called()

    def test_cancelled_queue_does_not_capture_or_move(self):
        presented = self.planned(); self.grasp.capture.reset_mock()
        response = self.dispatch(presented)
        self.app.jobs[response.json()['id']]['cancel'].set()
        job = self.finish(response)
        self.assertEqual(job['status'], 'cancelled')
        self.grasp.capture.assert_not_called(); self.execute.assert_not_called()

    def test_cancel_after_movement_preserves_receipt_without_starting_board_capture(self):
        presented = self.planned()
        response = self.dispatch(presented)
        cancel = self.app.jobs[response.json()['id']]['cancel']
        def moved_then_cancel(*args, **kwargs):
            result = self.completed_motion(*args, **kwargs)
            cancel.set()
            return result
        self.execute.side_effect = moved_then_cancel
        job = self.finish(response)
        self.assertEqual(job['status'], 'cancelled', job)
        self.assertTrue(job['calibration_motion_result']['completed'])
        self.store.capture.assert_not_called()

    def test_changed_hold_or_geometry_cannot_dispatch_old_candidate(self):
        presented = self.planned()
        self.robot.goals['elbow_flex'] += 1
        job = self.finish(self.dispatch(presented))
        self.assertEqual(job['status'], 'failed', job)
        self.execute.assert_not_called()
        presented = self.planned()
        with patch('server.calibration_motion.geometry_revision', return_value='new-model'):
            job = self.finish(self.dispatch(presented))
        self.assertEqual(job['status'], 'failed', job)
        self.execute.assert_not_called()

    def test_unknown_candidate_fails_without_motion(self):
        job = self.finish(self.dispatch(self.planned(), candidate_id='unplanned-motion'))
        self.assertEqual(job['status'], 'failed', job)
        self.execute.assert_not_called(); self.store.capture.assert_not_called()

    def test_expired_batch_fails_before_observation_executor(self):
        presented = self.planned()
        batch = self.app.calibration_view_batches[presented['batch_id']]
        batch.update(issued_monotonic=time.monotonic() - 150, expires_monotonic=time.monotonic() - 60)
        batch['contract_sha256'] = motion._fingerprint(batch)
        presented['batch_sha256'] = batch['contract_sha256']
        response = self.dispatch(presented)
        if response.status_code == 200:
            job = self.finish(response)
            self.assertEqual(job['status'], 'failed', job)
        else:
            self.assertEqual(response.status_code, 409)
        self.execute.assert_not_called(); self.store.capture.assert_not_called()

    def test_exact_path_review_is_required_not_just_button_or_candidate_name(self):
        for approval in ({}, {'approved': True}, {'approved': True, 'gripper_empty': True,
                'human_clear': True, 'clear_path': True, 'batch_sha256': 'wrong',
                'candidate_sha256': 'candidate-hash', 'frame_id': self.frame['id'], 'reason': 'reviewed'}):
            job = self.finish(self.dispatch(self.planned(), approval=approval))
            self.assertEqual(job['status'], 'failed', job)
        self.execute.assert_not_called()

    def test_second_plan_keeps_original_pose_and_empty_session_survives_reconnect(self):
        self.planned()
        self.frame['joints']['wrist_roll'] += 50; self.frame['id'] = 'new-pose'
        self.planned()
        self.assertEqual(self.build.call_args.kwargs['origin'], self.raw)
        self.frame['camera_generation'] = 'generation-2'
        job = self.finish(self.plan())
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(self.build.call_count, 3)
        self.assertEqual(self.build.call_args.kwargs['origin'], self.raw)

    def test_session_with_collected_frames_rejects_camera_reconnect(self):
        self.planned()
        self.session['camera_generation'] = 'generation-1'
        self.session['collected_view_count'] = 1
        self.frame['camera_generation'] = 'generation-2'
        job = self.finish(self.plan())
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(self.build.call_count, 1)

    def test_batch_cache_is_bounded(self):
        first = self.planned()['batch_id']
        for _ in range(13): self.planned()
        self.assertEqual(len(self.app.calibration_view_batches), 12)
        self.assertNotIn(first, self.app.calibration_view_batches)

    def test_token_and_valid_session_identifiers_required(self):
        response = self.client.post(self.base + '/plan', json={'session_id': self.session_id})
        self.assertEqual(response.status_code, 403)
        response = self.client.post(self.base + '/plan', json={'session_id': '../other'}, headers=self.headers)
        self.assertEqual(response.status_code, 422)
        self.assertFalse(self.queue)


if __name__ == '__main__':
    unittest.main()
