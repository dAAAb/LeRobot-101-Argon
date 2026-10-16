"""Workspace HTTP queue/ownership and durable one-station progress, no devices.

The numerical planner/executor are mocked; the real HTTP routes, job owner and
temporary on-disk journal run. No ASGI lifespan or motor driver is started.
"""
from copy import deepcopy
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from server.calibration_workspace_journal import WorkspaceJournal


class CalibrationWorkspaceApiTests(unittest.TestCase):
    def setUp(self):
        self.app = importlib.import_module('server.app')
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.journal = WorkspaceJournal(Path(self.temporary.name))
        self.session_id = 'a' * 32
        self.queue = []
        self.raw = dict(shoulder_pan=2181, shoulder_lift=1032, elbow_flex=3042,
                        wrist_flex=2174, wrist_roll=884, gripper=1861)
        self.frame = dict(id='frame-1', source='wrist', camera_generation='generation-1',
            path='/unused/wrist.jpg', sequence=1, capture_monotonic=time.monotonic(),
            joints=dict(self.raw), image_size=[1920, 1080])
        self.robot = SimpleNamespace(stop=threading.Event(), pwm_owner=None,
            moving=False, connect=Mock(), begin=Mock(), goals=dict(self.raw))
        self.grasp = SimpleNamespace(model=object(), state='idle', robot=self.robot,
            capture=Mock(side_effect=self.capture), record=Mock())
        self.session = dict(session_id=self.session_id, status='collecting',
            camera_id='wrist-usb', mount_revision='mount-1', geometry_revision='model-1',
            board_static_declared=True, camera_generation=None, collected_view_count=0)
        self.store = Mock()
        self.store.status.side_effect = lambda _: deepcopy(self.session)
        self.saved_origin = None
        self.store.motion_origin.side_effect = self.origin
        self.store.capture.return_value = {**self.session, 'accepted': True,
            'collected_view_count': 1, 'message': '已保存校準照片'}
        self.profile = {'selected_hash': 'profile-a'}
        self.build = Mock(side_effect=self.build_protocol)
        self.prepare = Mock(side_effect=self.prepare_station)
        self.validate = Mock(side_effect=self.validate_station)
        self.execute = Mock(side_effect=self.execute_station)
        values = dict(robot=self.robot, grasp=self.grasp, calibration_sessions=self.store,
            calibration_workspace_journal=self.journal, calibration_workspace_candidates={},
            active=None, jobs={}, executor=SimpleNamespace(submit=self.queue.append),
            snapshot_owner=threading.local(),
            CONFIG={'wrist_camera_id': 'wrist-usb', 'camera_mount_revision': 'mount-1'})
        for name, value in values.items():
            guard = patch.object(self.app, name, value)
            guard.start(); self.addCleanup(guard.stop)
        patches = {
            'server.spatial_runtime.geometry_revision': lambda _: 'model-1',
            'server.spatial_runtime._read_profile': lambda *_: deepcopy(self.profile),
            'server.calibration_workspace_motion._profile': lambda p, *_: p['selected_hash'],
            'server.calibration_workspace_motion.build_workspace_protocol': self.build,
            'server.calibration_workspace_motion.prepare_workspace_station': self.prepare,
            'server.calibration_workspace_motion.validate_workspace_station': self.validate,
            'server.calibration_workspace_motion.execute_workspace_station': self.execute,
            'server.calibration_workspace_motion.board_reference_from_capture': Mock(return_value=None),
            'server.calibration_workspace_motion.describe_workspace_station': deepcopy,
        }
        for target, value in patches.items():
            guard = patch(target, value)
            guard.start(); self.addCleanup(guard.stop)
        self.client = TestClient(self.app.app)
        self.addCleanup(self.client.close)
        self.headers = {'x-lerobot-token': self.app.TOKEN}
        self.base = '/api/engineering/calibration/workspace'

    def assert_actor_bound(self):
        owner = getattr(self.app.snapshot_owner, 'job', None)
        self.assertIsNotNone(owner)
        self.assertIs(owner, self.app.jobs[self.app.active])

    def capture(self):
        self.assert_actor_bound()
        return deepcopy(self.frame)

    def origin(self, session_id, frame):
        if self.saved_origin is None:
            self.saved_origin = dict(frame['joints'])
        return {'generation': frame['camera_generation'], 'joints': dict(self.saved_origin)}

    def build_protocol(self, grasp, frame, *, session_id, camera_profile, **kwargs):
        self.assert_actor_bound()
        return dict(session_id=session_id, origin=deepcopy(frame['joints']),
            camera_profile_sha256=camera_profile['selected_hash'],
            camera_profile=deepcopy(camera_profile), reference=deepcopy(frame),
            contract_sha256='protocol-hash', geometry_revision='model-1')

    def prepare_station(self, grasp, protocol, frame, *, station_index, completed_station_ids, **kwargs):
        self.assert_actor_bound()
        self.assertEqual(station_index, len(completed_station_ids))
        return dict(session_id=protocol['session_id'], station_id=f'station-{station_index}',
            station_index=station_index, reference=deepcopy(frame), origin=deepcopy(protocol['origin']),
            capture_required=True, contract_sha256='candidate-hash',
            protocol_sha256=protocol['contract_sha256'])

    def execute_station(self, grasp, protocol, candidate, reference, cancel, report, **kwargs):
        self.assert_actor_bound()
        pending = self.journal.load(self.session_id)['inflight']
        self.assertEqual(pending['station_id'], candidate['station_id'])
        self.assertEqual(kwargs['session_id'], self.session_id)
        after = {**deepcopy(reference), 'id': 'after-motion', 'sequence': reference['sequence'] + 1}
        return dict(completed=True, before=reference, after=after,
                    station_id=candidate['station_id'], next_station_executed=False)

    def validate_station(self, grasp, protocol, candidate, reference, **kwargs):
        self.assert_actor_bound()
        # Preflight must happen before the durable dispatch marker. A bad
        # review or expired candidate cannot poison this calibration session.
        self.assertIsNone(self.journal.load(self.session_id)['inflight'])
        self.assertEqual(kwargs['session_id'], self.session_id)
        self.assertEqual(kwargs['completed_station_ids'],
                         self.journal.load(self.session_id)['completed_station_ids'])
        self.assertEqual(kwargs['approval']['candidate_sha256'], candidate['contract_sha256'])
        return {'station_id': candidate['station_id']}

    def post(self, operation, data=None, **kwargs):
        return self.client.post(self.base + '/' + operation,
            json=data if data is not None else {'session_id': self.session_id},
            headers=kwargs.get('headers', self.headers))

    def finish(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        self.queue.pop(0)()
        return self.client.get('/api/jobs/' + response.json()['id']).json()

    def planned(self):
        result = self.finish(self.post('plan'))
        self.assertEqual(result['status'], 'completed', result)
        return result['workspace_protocol']

    def prepared(self):
        if self.journal.load(self.session_id) is None:
            self.planned()
        result = self.finish(self.post('prepare', {'session_id': self.session_id, 'station_index': 0}))
        self.assertEqual(result['status'], 'completed', result)
        return result['workspace_candidate']

    def dispatch(self, candidate):
        approval = dict(approved=True, gripper_empty=True, human_clear=True, clear_path=True,
            protocol_sha256=candidate['protocol_sha256'], candidate_sha256=candidate['contract_sha256'],
            frame_id=candidate['reference']['id'], reason='已核對目前腕部影像與這一站完整路徑')
        return self.post('execute', {'candidate_id': candidate['candidate_id'], 'approval': approval})

    def test_plan_uses_job_actor_and_repeated_plan_never_resets_origin(self):
        response = self.post('plan')
        self.grasp.capture.assert_not_called(); self.build.assert_not_called()
        first = self.finish(response)
        self.assertEqual(first['status'], 'completed')
        self.frame['joints']['elbow_flex'] -= 100
        self.assertEqual(self.planned()['origin'], self.raw)
        self.assertEqual(self.journal.load(self.session_id)['protocol']['origin'], self.raw)
        self.build.assert_called_once(); self.store.motion_origin.assert_called_once()
        self.execute.assert_not_called()
        self.assertFalse(hasattr(self.app.snapshot_owner, 'job'))

    def test_new_protocol_cannot_reset_an_existing_calibration_origin(self):
        self.saved_origin = dict(self.raw)
        self.frame['joints']['elbow_flex'] -= 100
        result = self.finish(self.post('plan'))
        self.assertEqual(result['status'], 'failed')
        self.build.assert_not_called()
        self.assertIsNone(self.journal.load(self.session_id))

    def test_prepare_replaces_review_and_execute_consumes_exactly_one_candidate(self):
        old = self.prepared(); current = self.prepared()
        self.assertNotEqual(old['candidate_id'], current['candidate_id'])
        self.assertEqual(self.dispatch(old).status_code, 409)
        response = self.dispatch(current)
        self.assertNotIn(current['candidate_id'], self.app.calibration_workspace_candidates)
        self.assertEqual(self.dispatch(current).status_code, 409)
        result = self.finish(response)
        self.assertEqual(result['status'], 'completed', result)
        self.execute.assert_called_once()
        self.assertFalse(result['calibration_motion_result']['next_station_executed'])
        stored = self.journal.load(self.session_id)
        self.assertEqual(stored['completed_station_ids'], ['station-0'])
        self.assertIsNone(stored['inflight'])
        self.assertEqual(len(stored['receipts']), 1)
        self.assertEqual(self.store.capture.call_args.args[1]['id'], 'after-motion')

    def test_next_station_forwards_saved_actual_receipt_to_prepare_and_both_checks(self):
        first = self.prepared()
        self.assertEqual(self.finish(self.dispatch(first))['status'], 'completed')
        saved = self.journal.load(self.session_id)['receipts'][-1]
        prepared = self.finish(self.post('prepare', {'session_id': self.session_id, 'station_index': 1}))
        self.assertEqual(prepared['status'], 'completed', prepared)
        self.assertEqual(self.prepare.call_args.kwargs['previous_receipt'], saved)
        second = self.finish(self.dispatch(prepared['workspace_candidate']))
        self.assertEqual(second['status'], 'completed', second)
        self.assertEqual(self.validate.call_args.kwargs['previous_receipt'], saved)
        self.assertEqual(self.execute.call_args.kwargs['previous_receipt'], saved)
        self.assertEqual(self.journal.load(self.session_id)['protocol']['origin'], self.raw)

    def test_stop_pwm_owner_and_active_job_preserve_unconsumed_candidate(self):
        candidate = self.prepared()
        self.robot.stop.set()
        self.assertEqual(self.dispatch(candidate).status_code, 409)
        self.robot.stop.clear(); self.robot.pwm_owner = object()
        self.assertEqual(self.dispatch(candidate).status_code, 409)
        self.robot.pwm_owner = None
        self.assertEqual(self.post('plan').status_code, 200)
        self.assertEqual(self.dispatch(candidate).status_code, 409)
        self.assertIn(candidate['candidate_id'], self.app.calibration_workspace_candidates)
        self.execute.assert_not_called()

    def test_moving_robot_cannot_queue_plan_prepare_or_consume_candidate(self):
        candidate = self.prepared()
        self.robot.moving = True
        for operation, payload in (
                ('plan', {'session_id': self.session_id}),
                ('prepare', {'session_id': self.session_id, 'station_index': 0})):
            with self.subTest(operation=operation):
                self.assertEqual(self.post(operation, payload).status_code, 409)
        self.assertEqual(self.dispatch(candidate).status_code, 409)
        self.assertIn(candidate['candidate_id'], self.app.calibration_workspace_candidates)
        self.assertFalse(self.queue)

    def test_cancelled_plan_and_prepare_never_connect_or_capture(self):
        response = self.post('plan')
        self.app.jobs[response.json()['id']]['cancel'].set()
        self.assertEqual(self.finish(response)['status'], 'cancelled')
        self.robot.connect.assert_not_called(); self.robot.begin.assert_not_called()
        self.grasp.capture.assert_not_called(); self.build.assert_not_called()
        self.assertIsNone(self.journal.load(self.session_id))
        self.planned()
        for operation in ('plan', 'prepare'):
            self.robot.connect.reset_mock(); self.robot.begin.reset_mock(); self.grasp.capture.reset_mock()
            payload = {'session_id': self.session_id}
            if operation == 'prepare':
                payload['station_index'] = 0
            response = self.post(operation, payload)
            self.app.jobs[response.json()['id']]['cancel'].set()
            with self.subTest(operation=operation):
                self.assertEqual(self.finish(response)['status'], 'cancelled')
                self.robot.connect.assert_not_called(); self.robot.begin.assert_not_called()
                self.grasp.capture.assert_not_called()
        self.execute.assert_not_called()

    def test_cancelled_execute_never_captures_or_creates_inflight_record(self):
        candidate = self.prepared(); self.grasp.capture.reset_mock()
        response = self.dispatch(candidate)
        self.app.jobs[response.json()['id']]['cancel'].set()
        self.assertEqual(self.finish(response)['status'], 'cancelled')
        self.execute.assert_not_called(); self.grasp.capture.assert_not_called()
        self.assertIsNone(self.journal.load(self.session_id)['inflight'])
        # Cancellation arriving after physical completion must retain that
        # receipt, but must not initiate the subsequent chessboard capture.
        candidate = self.prepared()
        response = self.dispatch(candidate)
        cancel = self.app.jobs[response.json()['id']]['cancel']
        def completed_then_cancel(*args, **kwargs):
            receipt = self.execute_station(*args, **kwargs)
            cancel.set()
            return receipt
        self.execute.side_effect = completed_then_cancel
        result = self.finish(response)
        self.assertEqual(result['status'], 'cancelled')
        self.assertTrue(result['calibration_motion_result']['completed'])
        self.assertEqual(self.journal.load(self.session_id)['completed_station_ids'], ['station-0'])
        self.store.capture.assert_not_called()

    def test_profile_change_while_queued_cannot_execute_or_advance_journal(self):
        candidate = self.prepared(); response = self.dispatch(candidate)
        self.profile['selected_hash'] = 'profile-b'
        result = self.finish(response)
        self.assertEqual(result['status'], 'failed', result)
        self.execute.assert_not_called(); self.store.capture.assert_not_called()
        self.assertIsNone(self.journal.load(self.session_id)['inflight'])
        self.assertEqual(self.journal.load(self.session_id)['completed_station_ids'], [])

    def test_board_capture_failure_preserves_physical_receipt_and_fixed_origin(self):
        self.store.capture.side_effect = ValueError('棋盤照片儲存失敗')
        result = self.finish(self.dispatch(self.prepared()))
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['calibration_motion_result']['completed'])
        record = self.journal.load(self.session_id)
        self.assertEqual(record['completed_station_ids'], ['station-0'])
        self.assertEqual(len(record['receipts']), 1)
        self.assertEqual(record['captures'], [])
        self.assertEqual(record['protocol']['origin'], self.raw)
        self.assertIsNone(record['inflight'])

    def test_invisible_board_reports_motion_without_claiming_a_calibration_sample(self):
        self.store.capture.return_value = {'accepted': False, 'message': '棋盤不完整'}
        result = self.finish(self.dispatch(self.prepared()))
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['calibration_motion_result']['completed'])
        self.assertFalse(result['calibration_result']['accepted'])
        self.assertFalse(self.journal.load(self.session_id)['captures'][0]['result']['accepted'])

    def test_preflight_rejection_is_retryable_but_unknown_execution_stays_unresolved(self):
        # The numerical validator owns TTL/hash checks. At the HTTP boundary,
        # either rejection must leave the durable state available for a fresh
        # candidate, while an executor failure remains ambiguous and blocked.
        for rejection in ('工程候選已過期', 'approval candidate hash mismatch'):
            candidate = self.prepared()
            self.validate.side_effect = ValueError(rejection)
            result = self.finish(self.dispatch(candidate))
            with self.subTest(rejection=rejection):
                self.assertEqual(result['status'], 'failed')
                self.assertIsNone(self.journal.load(self.session_id)['inflight'])
                self.assertEqual(self.journal.load(self.session_id)['completed_station_ids'], [])
                self.assertNotIn(candidate['candidate_id'], self.app.calibration_workspace_candidates)
                self.execute.assert_not_called(); self.store.capture.assert_not_called()
        self.validate.side_effect = self.validate_station
        self.execute.side_effect = RuntimeError('移動中斷，完成狀態未知')
        candidate = self.prepared()
        result = self.finish(self.dispatch(candidate))
        self.assertEqual(result['status'], 'failed')
        record = WorkspaceJournal(Path(self.temporary.name)).load(self.session_id)
        self.assertEqual(record['inflight']['candidate_id'], candidate['candidate_id'])
        self.assertEqual(record['completed_station_ids'], [])
        for operation, payload in (
                ('plan', {'session_id': self.session_id}),
                ('prepare', {'session_id': self.session_id, 'station_index': 0})):
            with self.subTest(operation=operation):
                self.assertEqual(self.finish(self.post(operation, payload))['status'], 'failed')
        self.assertEqual(self.execute.call_count, 1)

    def test_local_session_origin_and_valid_identifier_required_before_queue(self):
        for headers in ({}, {**self.headers, 'origin': 'https://example.invalid'}):
            with self.subTest(headers=list(headers)):
                self.assertEqual(self.post('plan', headers=headers).status_code, 403)
        self.assertEqual(self.post('plan', {'session_id': '../other'}).status_code, 422)
        self.assertEqual(self.post('prepare', {'session_id': self.session_id, 'station_index': True}).status_code, 422)
        self.assertFalse(self.queue)
        self.execute.assert_not_called(); self.grasp.capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
