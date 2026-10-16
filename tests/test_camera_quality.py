"""Near-black camera arrivals must never authorize an old usable wrist image.

Synthetic images, temporary files and a fake helper process only. No camera,
serial connection, application restart, or real capture process is used.
"""
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from server.camera import WristCamera, frame_quality


def visible_scene():
    # Most of this scene is a dark object; 10% is brighter visible context.
    image = np.full((160, 240, 3), 8, dtype=np.uint8)
    image[:16] = 120
    return image


def dark_noise():
    random = np.random.default_rng(17)
    image = random.integers(8, 18, (160, 240, 3), dtype=np.uint8)
    # Hot pixels/sporadic bright noise must not turn an unusable image valid.
    pixels = image.reshape(-1, 3)
    pixels[random.choice(len(pixels), len(pixels)//50, replace=False)] = 255
    return image


class CameraQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.camera = WristCamera(
            {'wrist_camera_name': 'synthetic', 'wrist_camera_id': 'never-open'},
            Path(self.temp.name)/'frames')
        self.clock = 100.
        clock = patch('server.camera.time.monotonic', side_effect=lambda: self.clock)
        clock.start()
        self.addCleanup(clock.stop)

    def test_bright_noise_does_not_authorize_near_black_frame(self):
        image = dark_noise()
        self.assertEqual(int(image.max()), 255)
        quality = frame_quality(image)
        self.assertFalse(quality['usable'])
        self.assertEqual(quality['status'], 'too_dark')
        self.assertLess(quality['mean_luma'], 24)
        self.assertLess(quality['p95_luma'], 32)

    def test_dark_object_with_visible_context_remains_usable(self):
        quality = frame_quality(visible_scene())
        self.assertLess(quality['mean_luma'], 24)
        self.assertGreater(quality['p95_luma'], 32)
        self.assertTrue(quality['usable'])
        self.assertEqual(quality['reason'], '')

    def test_black_and_undecodable_frames_are_unusable(self):
        for frame in (np.zeros((64, 64, 3), np.uint8), None,
                      np.empty((0, 0, 3), np.uint8), np.ones((64, 64, 3))):
            with self.subTest(shape=getattr(frame, 'shape', None)):
                self.assertFalse(frame_quality(frame)['usable'])

    def test_bad_arrival_immediately_revokes_old_frame_and_snapshot_authority(self):
        camera = self.camera
        self.assertTrue(camera._accept_frame(visible_scene(), 'same-helper'))
        self.assertTrue(camera.fresh())
        self.clock += .1
        self.assertFalse(camera._accept_frame(dark_noise(), 'same-helper'))
        self.assertIsNone(camera.frame)
        self.assertEqual(camera.timestamp, 0.)
        self.assertFalse(camera.fresh())
        self.assertIn('太暗', camera.error)
        status = camera.status()
        self.assertFalse(status['connected'])
        self.assertIsNone(status['age_ms'])
        self.assertEqual(status['image_quality']['status'], 'too_dark')
        for read in (camera.jpeg, camera.snapshot, camera.latest_stamp):
            with self.assertRaises(RuntimeError): read()
        self.assertEqual(list(camera.directory.iterdir()), [])

    def test_only_new_valid_arrival_clears_error_without_restarting_generation(self):
        camera = self.camera
        camera._accept_frame(visible_scene(), 'same-helper')
        self.clock += .1
        camera._accept_frame(dark_noise(), 'same-helper')
        self.clock += .1
        camera.status()
        self.assertFalse(camera.fresh())
        self.assertIn('太暗', camera.error)
        self.clock += .1
        self.assertTrue(camera._accept_frame(visible_scene(), 'same-helper'))
        self.assertTrue(camera.fresh())
        self.assertEqual(camera.generation, 'same-helper')
        self.assertEqual(camera.timestamp, self.clock)
        self.assertEqual(camera.error, '')
        self.assertTrue(camera.status()['image_quality']['usable'])
        self.assertEqual([entry['sequence'] for entry in camera.tracking_history(1, 3)], [1, 3],
                         'Unusable arrivals must leave a detectable history gap')
        frame = camera.snapshot()
        self.assertEqual(frame['source'], 'wrist')
        self.assertEqual(frame['sequence'], 3)
        self.assertEqual(frame['camera_generation'], 'same-helper')
        self.assertTrue(Path(frame['path']).exists())

    def test_decode_and_capture_errors_cannot_retain_previous_usable_timestamp(self):
        camera = self.camera
        camera._accept_frame(visible_scene(), 'same-helper')
        camera._accept_frame(None, 'same-helper')
        self.assertFalse(camera.fresh())
        self.assertEqual(camera.timestamp, 0.)
        self.assertEqual(camera.status()['image_quality']['status'], 'decode_failed')
        camera._accept_frame(visible_scene(), 'same-helper')
        with camera.lock:
            camera._invalidate('synthetic capture stopped', 'capture_error')
        self.assertFalse(camera.fresh())
        self.assertIsNone(camera.frame)
        self.assertEqual(camera.timestamp, 0.)

    def test_usable_image_still_expires_and_status_quality_is_not_mutable(self):
        camera = self.camera
        camera._accept_frame(visible_scene(), 'same-helper')
        status = camera.status()
        status['image_quality']['usable'] = False
        self.assertTrue(camera.fresh())
        self.clock += 1.1
        self.assertFalse(camera.fresh())
        with self.assertRaisesRegex(RuntimeError, '過期'): camera.jpeg()

    def test_failed_jpeg_encoding_cannot_leave_a_previous_frame_usable(self):
        for failure in (False, cv2.error('synthetic encoding failure')):
            self.camera._accept_frame(visible_scene(), 'same-helper')
            with patch('server.camera.cv2.imencode') as encode:
                if isinstance(failure, Exception): encode.side_effect = failure
                else: encode.return_value = (False, None)
                self.assertFalse(self.camera._accept_frame(visible_scene(), 'same-helper'))
            self.assertFalse(self.camera.fresh())
            self.assertEqual(self.camera.timestamp, 0.)
            self.assertEqual(self.camera.image_quality['status'], 'encode_failed')

    def test_capture_loop_keeps_same_process_through_darkness_and_recovers(self):
        camera = self.camera
        latest = Path(self.temp.name)/'wrist-live.jpg'
        samples = [visible_scene(), dark_noise(), visible_scene()]
        states = []
        fake_quit = types.SimpleNamespace(done=False, index=0)
        fake_quit.is_set = lambda: fake_quit.done
        def wait(timeout):
            if timeout != .06:
                return fake_quit.done
            if fake_quit.index:
                states.append((camera.status(), camera.generation, camera.frame is None))
            if fake_quit.index == len(samples):
                fake_quit.done = True
                return True
            fake_quit.index += 1
            latest.write_bytes(b'synthetic JPEG arrival')
            os.utime(latest, ns=(fake_quit.index, fake_quit.index))
            self.clock += .125
            return False
        fake_quit.wait = wait
        camera.quit = fake_quit
        helper = Mock()
        helper.poll.return_value = None
        with (patch('subprocess.Popen', return_value=helper) as launch,
              patch('server.camera.cv2.imread', side_effect=samples)):
            camera._capture()
        launch.assert_called_once()
        helper.terminate.assert_called_once()  # Only explicit fake shutdown.
        self.assertEqual([state[0]['connected'] for state in states], [True, False, True])
        self.assertEqual(len({state[1] for state in states}), 1)
        self.assertTrue(states[1][2])
        self.assertEqual(states[1][0]['image_quality']['status'], 'too_dark')


if __name__ == '__main__':
    unittest.main(verbosity=2)
