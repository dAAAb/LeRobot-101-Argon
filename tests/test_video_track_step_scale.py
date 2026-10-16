"""Step-limit units at native resolution; never camera/robot access."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from server.geometry import SpatialEstimator
from server.video_tracks import track_wrist_video


class VideoStepScaleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='wrist-step-units-')
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(84)
        # Multi-scale unique texture avoids repeated-label/aperture ambiguity.
        small = rng.integers(30, 220, (18, 32), dtype=np.uint8)
        self.texture = cv2.resize(small, (960, 540), interpolation=cv2.INTER_CUBIC)
        self.texture = cv2.GaussianBlur(self.texture, (5, 5), 1.)

    def tearDown(self):
        self.temp.cleanup()

    def inputs(self, width, step):
        height = width * 9 // 16
        base = cv2.resize(self.texture, (width, height), interpolation=cv2.INTER_LINEAR)
        observations, history = [], []
        for seq in range(3):
            image = cv2.warpAffine(base, np.float32([[1, 0, seq*step], [0, 1, 0]]),
                                   (width, height), borderValue=127)
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            self.assertTrue(ok)
            path = self.root / f'{width}-{step}-{seq}.jpg'
            path.write_bytes(encoded.tobytes())
            observations.append({'id': str(seq), 'source': 'wrist', 'sequence': seq,
                                 'path': str(path), 'joints': {'shoulder_pan': 2100 + seq}})
            history.append({'source': 'wrist', 'sequence': seq, 'jpeg': encoded.tobytes(),
                            'capture_monotonic': 1000. + seq/30})
        return observations, history

    def assert_motion(self, matches, step):
        pair = next((m for m in matches if m[:2] == (0, 2)), None)
        self.assertIsNotNone(pair)
        self.assertGreaterEqual(len(pair[2]), 12)
        errors = np.linalg.norm(pair[3] - pair[2] - [2*step, 0], axis=1)
        self.assertLess(float(np.percentile(errors, 95)), .7)

    def test_opt_in_preserves_fractional_step_and_explicit_default_is_unchanged(self):
        for width, step, effective in [(960, 18, 24), (1920, 36, 48)]:
            with self.subTest(width=width):
                observations, history = self.inputs(width, step)
                default, _, default_meta = track_wrist_video(
                    observations, history, max_features=180, reseed_keyframes=False)
                self.assertEqual(default_meta['thresholds']['effective_max_step_px'], 24)
                self.assertIsNone(default_meta['thresholds']['max_step_reference_width_px'])
                if width == 960:
                    self.assert_motion(default, step)
                else:
                    self.assertEqual(default, [])
                    self.assertGreaterEqual(default_meta['tracks']['dropped'].get('step_limit', 0), 12)
                scaled, _, meta = track_wrist_video(
                    observations, history, max_features=180, reseed_keyframes=False,
                    max_step_reference_width_px=960)
                self.assertEqual(meta['status'], 'ok', meta)
                self.assert_motion(scaled, step)
                self.assertEqual(meta['thresholds']['max_step_px'], 24)
                self.assertEqual(meta['thresholds']['max_step_reference_width_px'], 960)
                self.assertEqual(meta['thresholds']['effective_max_step_px'], effective)
                self.assertEqual(meta['thresholds']['fb_max_px'], .8)
                self.assertEqual(meta['pyramid_support_radius_px'], 54)
                self.assertFalse(meta['grasp_ready'])

    def test_explicit_native_pixel_override_stays_literal(self):
        observations, history = self.inputs(1920, 36)
        matches, _, meta = track_wrist_video(
            observations, history, max_features=180, reseed_keyframes=False, max_step_px=40)
        self.assert_motion(matches, 36)
        self.assertEqual(meta['thresholds']['effective_max_step_px'], 40)
        self.assertIsNone(meta['thresholds']['max_step_reference_width_px'])

    def test_opt_in_does_not_scale_forward_backward_acceptance(self):
        observations, history = self.inputs(1920, 36)
        real_lk = cv2.calcOpticalFlowPyrLK
        calls = 0

        def add_backward_error(*args, **kwargs):
            nonlocal calls
            points, status, error = real_lk(*args, **kwargs)
            calls += 1
            if calls % 2 == 0 and points is not None:
                points = points.copy()
                points[:, 0, 1] += 1.0
            return points, status, error

        with patch.object(cv2, 'calcOpticalFlowPyrLK', add_backward_error):
            matches, _, meta = track_wrist_video(
                observations, history, max_features=180, reseed_keyframes=False,
                max_step_reference_width_px=960)
        self.assertEqual(matches, [])
        self.assertGreaterEqual(meta['tracks']['dropped'].get('forward_backward', 0), 12)
        self.assertEqual(meta['thresholds']['fb_max_px'], .8)
        self.assertEqual(meta['thresholds']['effective_max_step_px'], 48)

    def test_invalid_units_fail_closed_and_unknown_size_is_not_invented(self):
        observations, history = self.inputs(960, 18)
        for reference in [True, 0, -960, 960.0, '960', float('nan')]:
            with self.subTest(reference=reference):
                matches, _, meta = track_wrist_video(
                    observations, history, max_step_reference_width_px=reference)
                self.assertEqual(matches, [])
                self.assertEqual(meta['status'], 'invalid_input')
        for step in [True, 0, -1, float('nan'), float('inf'), 10**400, complex(1, 2)]:
            with self.subTest(step=step):
                matches, _, meta = track_wrist_video(observations, history, max_step_px=step)
                self.assertEqual(matches, [])
                self.assertEqual(meta['status'], 'invalid_input')
        _, _, meta = track_wrist_video(observations, [], max_step_reference_width_px=960)
        self.assertEqual(meta['status'], 'no_history')
        self.assertIsNone(meta['thresholds']['effective_max_step_px'])
        self.assertEqual(meta['thresholds']['max_step_reference_width_px'], 960)

    def test_opt_in_cannot_change_units_mid_track(self):
        small_observations, small_history = self.inputs(960, 18)
        large_observations, large_history = self.inputs(1920, 36)
        observations = small_observations[:1] + large_observations[1:]
        history = small_history[:1] + large_history[1:]
        matches, _, meta = track_wrist_video(
            observations, history, max_step_reference_width_px=960)
        self.assertEqual(matches, [])
        self.assertEqual(meta['status'], 'image_geometry_changed')
        self.assertEqual(meta['thresholds']['effective_max_step_px'], 24)

    def test_spatial_estimator_opts_in_with_real_native_motion(self):
        observations, history = self.inputs(1920, 36)
        estimator = SpatialEstimator(Path(__file__).resolve().parents[1])
        matches, size, diagnostics = estimator.correspondences(
            observations, histories=[{'keyframes': [o['id'] for o in observations],
                                      'frames': history}])
        self.assertEqual(size, (1920, 1080))
        self.assert_motion(matches, 36)
        self.assertEqual(diagnostics[0]['thresholds']['max_step_reference_width_px'], 960)
        self.assertEqual(diagnostics[0]['thresholds']['effective_max_step_px'], 48)
        self.assertEqual(diagnostics[0]['thresholds']['fb_max_px'], .8)
        self.assertFalse(diagnostics[0]['grasp_ready'])
