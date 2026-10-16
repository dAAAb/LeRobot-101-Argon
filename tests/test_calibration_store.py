"""Storage/session boundaries; detector and fitter have independent optic tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from server.calibration_store import CalibrationSessionStore


class _Board:
    fingerprint = 'board-9x6-20mm'
    def to_dict(self):
        return {'columns': 9, 'rows': 6, 'square_size_m': .02}


class CalibrationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / 'wrist.jpg'
        cv2.imwrite(str(self.original), np.full((480, 640, 3), 127, np.uint8))
        self.store = CalibrationSessionStore(self.root)
        self.board_module = ModuleType('server.board_calibration')
        self.board_module.BoardSpec = _Board
        self.board_module.detect_board = Mock(side_effect=self.detection)
        self.board_module.fit_board_calibration = Mock(return_value={
            'status': 'validated', 'calibration_profile': {'source': 'wrist-only', 'calibration_revision': 'profile-1'},
            'validation': {'heldout_views': 3}, 'blockers': [], 'detections': []})
        self.modules = patch.dict('sys.modules', {'server.board_calibration': self.board_module})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.isolated = patch('server.calibration_detection_worker.detect_board_isolated',
            side_effect=lambda jpeg, **kwargs: self.board_module.detect_board(
                cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR), **kwargs))
        self.isolated.start()
        self.addCleanup(self.isolated.stop)
        self.session = self.store.start(camera_id='icspring-wrist', mount_revision='mount-a',
                                       geometry_revision='geometry-a', board_static=True)['session_id']

    @staticmethod
    def detection(image, *, frame, board):
        return {'found': True, 'corners_px': [[100. + c * 25, 120. + r * 25] for r in range(6) for c in range(9)],
                'image_size': [640, 480], 'board_spec': board.to_dict(), 'board_fingerprint': board.fingerprint,
                'coverage_fraction': .25, 'reason': '', 'frame_id': frame['id'],
                'camera_generation': frame['camera_generation'], 'sequence': frame['sequence'],
                'capture_monotonic': frame['capture_monotonic']}

    def frame(self, number=1):
        joints = dict(shoulder_pan=2000 + number * 20, shoulder_lift=1500,
                      elbow_flex=3000, wrist_flex=2000, wrist_roll=1700, gripper=1900)
        stamp = number * 3.
        return {'id': 'wrist-' + str(number), 'source': 'wrist', 'camera_id': 'icspring-wrist',
            'camera_generation': 'generation-a', 'path': str(self.original), 'sequence': number,
            'capture_monotonic': stamp, 'image_size': [640, 480], 'joints': joints,
            'timestamp_basis': 'jpeg-arrival after settled joint window',
            'joint_capture_span_ticks': 0, 'image_wait_joint_span_ticks': 0,
            'joint_observation_bracket_monotonic': [stamp - .02, stamp + .02],
            'settled_joint_window': [{'read_start_monotonic': stamp - .5,
                'read_end_monotonic': stamp - .49, 'joints': dict(joints)},
                {'read_start_monotonic': stamp - .1, 'read_end_monotonic': stamp - .09, 'joints': dict(joints)}]}

    def capture(self, frame=None, **kwargs):
        return self.store.capture(self.session, frame or self.frame(), board_static=kwargs.get('board_static', True))

    def manifest(self):
        return json.loads((self.store.root / self.session / 'session.json').read_text())

    def test_copy_is_immutable_and_preserves_detection_and_measurement_binding(self):
        frame = self.frame()
        result = self.capture(frame)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['collected_view_count'], 1)
        self.assertEqual(result['distinct_pose_count'], 1)
        saved = self.manifest()['frames'][0]
        self.assertNotEqual(saved['path'], frame['path'])
        self.assertEqual(Path(saved['path']).read_bytes(), self.original.read_bytes())
        self.assertEqual(saved['image_sha256'], hashlib.sha256(self.original.read_bytes()).hexdigest())
        self.assertEqual(saved['board_session_id'], self.session)
        self.assertEqual(saved['board_detection']['frame_id'], frame['id'])
        self.original.unlink()
        self.assertEqual(self.store.solve(self.session, object())['last_solve']['status'], 'validated')

    def test_rejected_detector_image_does_not_count_as_collected_view(self):
        self.board_module.detect_board.side_effect = None
        self.board_module.detect_board.return_value = {'found': False, 'reason': '板子未完整入鏡'}
        result = self.capture()
        self.assertFalse(result['accepted'])
        self.assertEqual(result['collected_view_count'], 0)
        self.assertEqual(result['message'], '板子未完整入鏡')
        self.assertEqual(list((self.store.root / self.session / 'frames').iterdir()), [])

    def test_native_detector_failure_leaves_session_and_images_unchanged(self):
        manifest = self.store.root / self.session / 'session.json'
        before = manifest.read_bytes()
        with patch('server.calibration_detection_worker.detect_board_isolated',
                   side_effect=ValueError('獨立棋盤辨識中斷')):
            with self.assertRaisesRegex(ValueError, '辨識中斷'):
                self.capture()
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual(list((self.store.root / self.session / 'frames').iterdir()), [])

    def test_distinct_count_and_board_coverage_do_not_count_same_pose_twice(self):
        self.capture()
        frame = self.frame(2); frame['joints']['shoulder_pan'] = 2021
        for sample in frame['settled_joint_window']: sample['joints'] = dict(frame['joints'])
        result = self.capture(frame)
        self.assertEqual(result['collected_view_count'], 2)
        self.assertEqual(result['distinct_pose_count'], 1)
        self.assertEqual(result['coverage']['minimum_board_fraction'], .25)
        result = self.capture(self.frame(3))
        self.assertEqual(result['distinct_pose_count'], 2)

    def test_wrong_sources_and_camera_ids_cannot_enter_session(self):
        for field, value in [('source', 'D435'), ('camera_id', 'external-iphone'),
                             ('board_session_id', 'other-session')]:
            frame = self.frame(); frame[field] = value
            with self.assertRaises(ValueError): self.capture(frame)
        self.assertEqual(self.store.status(self.session)['collected_view_count'], 0)

    def test_camera_restart_and_resolution_change_require_new_session(self):
        self.capture()
        frame = self.frame(2); frame['camera_generation'] = 'new-generation'
        with self.assertRaises(ValueError): self.capture(frame)
        cv2.imwrite(str(self.original), np.full((720, 1280, 3), 127, np.uint8))
        frame = self.frame(2); frame['image_size'] = [1280, 720]
        with self.assertRaises(ValueError): self.capture(frame)
        self.assertEqual(self.store.status(self.session)['collected_view_count'], 1)

    def test_repeated_frames_or_old_timestamps_are_rejected(self):
        self.capture()
        with self.assertRaises(ValueError): self.capture()
        frame = self.frame(2); frame['capture_monotonic'] = 1.
        with self.assertRaises(ValueError): self.capture(frame)

    def test_moving_board_invalidates_session_and_new_start_does_not_reuse_frames(self):
        self.capture()
        with self.assertRaises(ValueError): self.capture(self.frame(2), board_static=False)
        self.assertEqual(self.store.status(self.session)['status'], 'invalidated')
        with self.assertRaises(ValueError): self.capture(self.frame(3))
        with self.assertRaises(ValueError): self.store.solve(self.session, object())
        new = self.store.start(camera_id='icspring-wrist', mount_revision='mount-a',
                               geometry_revision='geometry-a', board_static=True)
        self.assertNotEqual(new['session_id'], self.session)
        self.assertEqual(new['collected_view_count'], 0)
        self.assertEqual(len(self.store.list_sessions()), 2)

    def test_board_static_declaration_is_required(self):
        with self.assertRaises(ValueError):
            self.store.start(camera_id='icspring-wrist', mount_revision='mount-a',
                             geometry_revision='geometry-a', board_static=False)

    def test_unsettled_or_unbound_motor_samples_are_rejected(self):
        for update in ({'joint_capture_span_ticks': 6}, {'settled_joint_window': []},
                       {'joint_observation_bracket_monotonic': [0., 1.]},
                       {'image_wait_joint_span_ticks': True}):
            frame = self.frame(); frame.update(update)
            with self.assertRaises(ValueError): self.capture(frame)
        frame = self.frame(); frame['settled_joint_window'][1]['joints']['elbow_flex'] += 8
        with self.assertRaises(ValueError): self.capture(frame)

    def test_solve_persists_new_report_and_profile_without_replacing_active_config(self):
        config = self.root / 'config.json'; config.write_text('{"active": "legacy"}')
        self.capture()
        result = self.store.solve(self.session, object())
        self.assertEqual(result['status'], 'solved')
        report = Path(result['last_solve']['report_path'])
        candidate = Path(result['last_solve']['candidate_profile_path'])
        self.assertTrue(report.is_file()); self.assertTrue(candidate.is_file())
        self.assertFalse(result['active_profile_changed'])
        self.assertEqual(config.read_text(), '{"active": "legacy"}')
        original = candidate.read_bytes()
        next_result = self.store.solve(self.session, object())
        self.assertNotEqual(next_result['last_solve']['candidate_profile_path'], str(candidate))
        self.assertEqual(candidate.read_bytes(), original)
        _, args = self.board_module.fit_board_calibration.call_args
        self.assertEqual(args['board_session_id'], self.session)
        self.assertEqual(args['camera_id'], 'icspring-wrist')

    def test_extension_uses_same_immutable_journal_and_never_full_refits(self):
        self.capture()
        extension=ModuleType('server.calibration_extension')
        extension.validate_workspace_extension=Mock(return_value={
            'status':'validated','calibration_profile':{'source':'wrist-only','calibration_revision':'extension-1'},
            'validation':{'heldout_views':3},'blockers':[]})
        parent={'calibration_revision':'parent','handeye':[[1]]}
        with patch.dict('sys.modules',{'server.calibration_extension':extension}):
            result=self.store.solve(self.session,object(),frozen_profile=parent)
        self.board_module.fit_board_calibration.assert_not_called()
        call=extension.validate_workspace_extension.call_args
        self.assertEqual(call.args[2],parent)
        self.assertIsNot(call.args[2],parent)
        self.assertEqual(call.kwargs['board_session_id'],self.session)
        self.assertEqual(json.loads(Path(result['last_solve']['candidate_profile_path']).read_text())['calibration_revision'],'extension-1')
        self.assertFalse(result['active_profile_changed'])

    def test_insufficient_solve_keeps_session_open_for_more_views(self):
        self.capture()
        self.board_module.fit_board_calibration.return_value = {
            'status': 'insufficient', 'calibration_profile': None,
            'validation': {}, 'blockers': ['need more views'], 'detections': []}
        result = self.store.solve(self.session, object())
        self.assertEqual(result['status'], 'collecting')
        self.assertIsNone(result['last_solve']['candidate_profile_path'])
        self.assertEqual(self.capture(self.frame(2))['collected_view_count'], 2)

    def test_modified_saved_image_is_rejected_before_fit(self):
        self.capture()
        Path(self.manifest()['frames'][0]['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError): self.store.solve(self.session, object())
        self.board_module.fit_board_calibration.assert_not_called()

    def test_cancellation_and_fit_exception_leave_captures_intact(self):
        self.capture()
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(RuntimeError): self.store.solve(self.session, object(), cancel=cancel)
        self.board_module.fit_board_calibration.side_effect = RuntimeError('fit failed')
        with self.assertRaises(RuntimeError): self.store.solve(self.session, object())
        result = self.store.status(self.session)
        self.assertEqual(result['status'], 'collecting')
        self.assertEqual(result['collected_view_count'], 1)
        self.assertIsNone(result['last_solve'])

    def test_session_path_traversal_is_rejected(self):
        for value in ('../other', '/tmp/elsewhere', 'a' * 31, self.session + '/..'):
            with self.assertRaises(ValueError): self.store.status(value)


if __name__ == '__main__':
    unittest.main()
