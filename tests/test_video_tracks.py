import tempfile, threading, unittest
from pathlib import Path
from unittest.mock import patch
import cv2, numpy as np
from server.video_tracks import track_wrist_video, pyramid_support_radius

class VideoTrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='wrist-video-synthetic-')
        self.root = Path(self.temp.name)
        # Repeated labels deliberately create descriptor ambiguity across views.
        base = np.full((480, 640), 185, np.uint8)
        for y in range(40, 270, 42):
            for x in range(16, 590, 74):
                cv2.putText(base, 'ABAB', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, 25, 1, cv2.LINE_AA)
        self.images = [cv2.warpAffine(base, np.float32([[1, 0, 3*i], [0, 1, 0]]),
                                     (640, 480), borderValue=185) for i in range(11)]

    def tearDown(self):
        self.temp.cleanup()

    def inputs(self, images=None, keys=(0, 5, 10)):
        images = self.images if images is None else images
        history, observations = [], []
        for seq, image in enumerate(images):
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.assertTrue(ok)
            history.append({'sequence': seq, 'jpeg': encoded.tobytes()})
            if seq in keys:
                path = self.root / f'{seq}.jpg';path.write_bytes(encoded.tobytes())
                observations.append({'sequence': seq, 'source': 'wrist', 'path': str(path),
                                     'joints': {'shoulder_pan': 2000 + seq, 'elbow_flex': 1900}})
        return observations, history

    def test_repeated_text_tracks_known_30px_identity(self):
        observations, history = self.inputs()
        matches, size, meta = track_wrist_video(observations, history, reseed_keyframes=False)
        self.assertEqual(meta['status'], 'ok', meta)
        self.assertEqual(size, (640, 480))
        pair = next((m for m in matches if m[0:2] == (0, 2)), None)
        self.assertIsNotNone(pair)
        _, _, p0, p10 = pair
        self.assertGreaterEqual(len(p0), 12)
        error = np.linalg.norm((p10-p0) - np.array([30., 0.]), axis=1)
        self.assertLess(float(np.percentile(error, 95)), .5)
        self.assertLess(float(np.max(error)), 1.0)
        self.assertTrue(np.all(p0[:, 1] < 480 * .595))
        self.assertTrue(np.all(p10[:, 1] < 480 * .595))
        self.assertEqual(meta['tracks']['eligible'], len(p0))
        print('synthetic identity:', len(p0), 'tracks; final p95 error', round(float(np.percentile(error,95)),4), 'px')

    def test_full_support_radius_includes_gaussian_and_derivatives(self):
        self.assertEqual([pyramid_support_radius(21, level) for level in range(4)], [12, 26, 54, 110])

    def test_moving_gripper_cannot_change_world_tracks_or_reseeds(self):
        rng = np.random.default_rng(94)
        row = int(.595 * self.images[0].shape[0])
        corrupted = [image.copy() for image in self.images]
        for seq, image in enumerate(corrupted):
            # Independently moving/high contrast attached-gripper pattern.
            band = rng.integers(0, 256, image[row:].shape, dtype=np.uint8)
            image[row:] = np.roll(band, -9*seq, axis=1)
        for level in (2, 3):
            with self.subTest(level=level):
                # Four keyframes exercise reseeding as well as initial seeds.
                observations, history = self.inputs(keys=(0, 3, 6, 10))
                baseline, _, meta = track_wrist_video(observations, history, pyramid_max_level=level)
                self.assertEqual(meta['status'], 'ok', meta)
                obs2, history2 = self.inputs(corrupted, keys=(0, 3, 6, 10))
                altered, _, altered_meta = track_wrist_video(obs2, history2, pyramid_max_level=level)
                self.assertEqual(altered_meta['status'], 'ok', altered_meta)
                self.assertEqual(len(baseline), len(altered))
                self.assertTrue(any(v['seeded'] > 0 and v['keyframe_index'] not in (None, 0)
                                    for v in meta['visibility']), 'Test must exercise reseeding')
                guard = pyramid_support_radius(21, level)
                for (i, j, p1, p2), (ii, jj, q1, q2) in zip(baseline, altered):
                    self.assertEqual((i, j), (ii, jj))
                    np.testing.assert_array_equal(p1, q1)
                    np.testing.assert_array_equal(p2, q2)
                    for points in (p1, p2):
                        self.assertTrue(np.all(points[:, 0] >= guard))
                        self.assertTrue(np.all(points[:, 1] >= guard))
                        self.assertTrue(np.all(points[:, 1] + guard < meta['tracking_boundary_px']))
                print('masked gripper:', level, 'guard', guard, 'px;', len(baseline), 'pairs bit-identical')

    def test_timestamp_gap_restart_and_missing_value_fail_closed(self):
        observations, history = self.inputs()
        for entry in history:
            entry['capture_monotonic'] = 1000.0 + .125 * entry['sequence']
        matches, _, meta = track_wrist_video(observations, history, require_timestamps=True)
        self.assertTrue(matches)
        self.assertEqual(meta['temporal_integrity'], 'sequence-and-capture-monotonic')
        cases = []
        gap = [dict(entry) for entry in history]
        for entry in gap[6:]:
            entry['capture_monotonic'] += 20
        cases.append(gap)
        restart = [dict(entry) for entry in history]
        for entry in restart[6:]:
            entry['capture_monotonic'] -= 100
        cases.append(restart)
        missing = [dict(entry) for entry in history]
        missing[5].pop('capture_monotonic')
        cases.append(missing)
        for broken in cases:
            matches, _, meta = track_wrist_video(observations, broken)
            self.assertEqual(matches, [])
            self.assertEqual(meta['status'], 'history_gap')
        observations, untimed = self.inputs()
        matches, _, meta = track_wrist_video(observations, untimed, require_timestamps=True)
        self.assertEqual(matches, [])
        self.assertEqual(meta['status'], 'history_gap')

    def test_abrupt_occlusion_drops_old_identities(self):
        images = [im.copy() for im in self.images]
        images[6] = np.zeros_like(images[6])
        observations, history = self.inputs(images)
        matches, _, meta = track_wrist_video(observations, history)
        self.assertEqual(matches, [])
        self.assertEqual(meta['status'], 'insufficient_tracks')
        lost = next(v for v in meta['visibility'] if v['sequence'] == 6)
        self.assertEqual(lost['survived'], 0)
        self.assertGreater(sum(lost['dropped'].values()), 12)

    def test_lk_status_lost_is_not_recovered_as_same_identity(self):
        observations, history = self.inputs()
        real_lk = cv2.calcOpticalFlowPyrLK
        calls = [0]
        def lose_status(*args, **kwargs):
            points, status, error = real_lk(*args, **kwargs)
            calls[0] += 1
            if calls[0] == 1 and status is not None:
                status[:] = 0
            return points, status, error
        with patch.object(cv2, 'calcOpticalFlowPyrLK', lose_status):
            matches, _, meta = track_wrist_video(observations, history, reseed_keyframes=False)
        self.assertEqual(matches, [])
        self.assertGreater(meta['tracks']['dropped'].get('lk_status', 0), 12)

    def test_gap_or_absent_history_fails_closed(self):
        observations, history = self.inputs()
        for data, status in [(history[:4] + history[5:], 'history_gap'), ([], 'no_history')]:
            with self.subTest(status=status):
                matches, _, meta = track_wrist_video(observations, data)
                self.assertEqual(matches, [])
                self.assertEqual(meta['status'], status)

    def test_cancel_returns_no_matches(self):
        observations, history = self.inputs()
        cancel = threading.Event();cancel.set()
        matches, _, meta = track_wrist_video(observations, history, cancel)
        self.assertEqual(matches, [])
        self.assertTrue(meta['cancelled'])
        self.assertEqual(meta['status'], 'cancelled')

    def test_observer_cannot_enter_tracker(self):
        observations, history = self.inputs()
        observations[1]['source'] = 'observer'
        matches, _, meta = track_wrist_video(observations, history)
        self.assertEqual(matches, [])
        self.assertEqual(meta['status'], 'invalid_input')

