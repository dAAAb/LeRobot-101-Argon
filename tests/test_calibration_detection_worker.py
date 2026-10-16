"""A native detector failure cannot terminate the caller or replay motion."""
import base64
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server.board_calibration import BoardSpec
from server.calibration_detection_worker import detect_board_isolated


class CalibrationDetectionWorkerTests(unittest.TestCase):
    def setUp(self):
        self.board = BoardSpec()
        self.jpeg = b'exact-original-jpeg-bytes'
        self.frame = {'id': 'wrist-frame', 'source': 'wrist', 'camera_generation': 'same-camera',
                      'sequence': 7, 'capture_monotonic': 12.5, 'image_size': [1920, 1080]}
        self.result = {'found': True, 'frame_id': self.frame['id'],
                      'camera_generation': self.frame['camera_generation'],
                      'sequence': self.frame['sequence'], 'capture_monotonic': self.frame['capture_monotonic'],
                      'image_size': self.frame['image_size'], 'board_fingerprint': self.board.fingerprint,
                      'corners_px': [], 'coverage_fraction': .2}

    def call(self):
        return detect_board_isolated(self.jpeg, frame=self.frame, board=self.board)

    def test_exact_bytes_and_frame_go_to_cpu_only_child_with_bounded_lifetime(self):
        process = SimpleNamespace(returncode=0, stdout=json.dumps(self.result).encode(), stderr=b'')
        with patch('server.calibration_detection_worker.subprocess.run', return_value=process) as run:
            self.assertEqual(self.call(), self.result)
        args, kwargs = run.call_args
        self.assertEqual(args[0][-2:], ['-m', 'server.calibration_detection_worker'])
        self.assertEqual(kwargs['env']['OPENCV_OPENCL_RUNTIME'], 'disabled')
        self.assertEqual(kwargs['timeout'], 30.)
        payload = json.loads(kwargs['input'])
        self.assertEqual(base64.b64decode(payload['jpeg']), self.jpeg)
        self.assertEqual(payload['frame'], self.frame)

    def test_native_crash_is_a_regular_failure_and_never_retried(self):
        with patch('server.calibration_detection_worker.subprocess.run',
                   return_value=SimpleNamespace(returncode=-11, stdout=b'', stderr=b'native crash')) as run:
            with self.assertRaisesRegex(ValueError, '辨識中斷'):
                self.call()
        run.assert_called_once()

    def test_timeout_is_a_regular_failure_and_never_retried(self):
        with patch('server.calibration_detection_worker.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('worker', 30)) as run:
            with self.assertRaisesRegex(ValueError, '辨識逾時'):
                self.call()
        run.assert_called_once()

    def test_corrupt_or_wrong_frame_response_does_not_become_a_detection(self):
        responses = [b'not-json', b'[]', json.dumps({**self.result, 'frame_id': 'other-frame'}).encode(),
                     json.dumps({**self.result, 'board_fingerprint': 'another-board'}).encode()]
        for response in responses:
            with self.subTest(response=response):
                with patch('server.calibration_detection_worker.subprocess.run',
                           return_value=SimpleNamespace(returncode=0, stdout=response, stderr=b'')):
                    with self.assertRaisesRegex(ValueError, '回傳資料不完整'):
                        self.call()

    def test_invalid_image_or_timeout_never_starts_child(self):
        with patch('server.calibration_detection_worker.subprocess.run') as run:
            for kwargs in ({'jpeg': b''}, {'jpeg': self.jpeg, 'timeout': 31},
                           {'jpeg': self.jpeg, 'timeout': float('nan')}):
                with self.assertRaises(ValueError):
                    detect_board_isolated(**kwargs, frame=self.frame, board=self.board)
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
