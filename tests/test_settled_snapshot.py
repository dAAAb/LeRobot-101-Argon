"""Fresh exposure/encoder pairing under delayed camera delivery; no hardware."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from server.robot import JOINTS, Robot, SETTLED_IMAGE_WAIT_SECONDS


class Clock:
    def __init__(self):
        self.now = 10.
    def advance(self, duration):
        self.now += duration


class Stop:
    def __init__(self, clock):
        self.clock = clock
        self.stopped = False
        self.cancel_at = None
    def wait(self, duration):
        self.clock.advance(duration)
        if self.cancel_at is not None and self.clock.now >= self.cancel_at:
            self.stopped = True
        return self.stopped
    def set(self):
        self.stopped = True


class Camera:
    generation = 'same-wrist'
    def __init__(self, clock):
        self.clock = clock
        # Six120-ms settle intervals + seven2-ms register reads end at10.734.
        # The old exposure is fresh but precedes that required pose bracket.
        self.old_exposure = 10.73
        self.exposure = 11.63
        self.available_at = 11.65
        self.unavailable_until = 0.
        self.reconnect_at = None
        self.metadata_override = None
        self.polls = 0
    def latest_stamp(self):
        self.polls += 1
        if self.reconnect_at is not None and self.clock.now >= self.reconnect_at:
            self.generation = 'replacement-wrist'
        if self.clock.now < self.unavailable_until:
            raise RuntimeError('腕部相機畫面已過期')
        captured = self.exposure if self.clock.now >= self.available_at else self.old_exposure
        if self.clock.now - captured >= 1:
            raise RuntimeError('腕部相機畫面已過期')
        return {'frame_sequence': 2 if self.clock.now >= self.available_at else 1,
                'frame_capture_monotonic': captured, 'camera_source': 'wrist',
                **(self.metadata_override or {})}


class SettledSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.camera = Camera(self.clock)
        self.stop = Stop(self.clock)
        self.raw = {n: 2000 for n in JOINTS}
        self.read_samples = []
        self.pose_at = lambda now: dict(self.raw)
        self.snapshot_delay = .04
        self.snapshot_override = {}
        self.g = SimpleNamespace(stop=self.stop, camera=self.camera, read=self.read,
                                 hold=Mock(), error='')
        self.snap = Mock(side_effect=self.serialize)
        self.guard = Mock()
        timing = patch('server.robot.time.monotonic', side_effect=lambda: self.clock.now)
        timing.start(); self.addCleanup(timing.stop)

    def read(self):
        self.clock.advance(.002)
        value = self.pose_at(self.clock.now)
        self.read_samples.append((self.clock.now, value))
        return dict(value)

    def serialize(self):
        self.assertGreaterEqual(self.clock.now, self.camera.available_at,
                                'Do not encode/write the old exposure during metadata polls')
        self.clock.advance(self.snapshot_delay)
        return {'id': 'new-image', 'sequence': 2, 'source': 'wrist',
                'camera_generation': self.camera.generation,
                'capture_monotonic': self.camera.exposure, **self.snapshot_override}

    def capture(self):
        return Robot.settled_snapshot(self.g, self.snap, guard=self.guard)

    def test_delayed_fresh_exposure_serializes_once_and_keeps_tight_stable_joint_bracket(self):
        frame = self.capture()
        self.snap.assert_called_once()
        self.assertGreater(frame['image_wait_seconds'], .7)
        self.assertLess(frame['image_wait_seconds'], SETTLED_IMAGE_WAIT_SECONDS)
        self.assertGreater(frame['image_wait_polls'], 10)
        self.assertEqual(frame['image_wait_joint_span_ticks'], 0)
        self.assertEqual(frame['joints'], self.raw)
        self.assertEqual(self.guard.call_count, len(self.read_samples))
        self.assertLessEqual(frame['joint_observation_bracket_monotonic'][0], frame['capture_monotonic'])
        self.assertGreaterEqual(frame['joint_observation_bracket_monotonic'][1], frame['capture_monotonic'])
        self.assertLess(frame['image_joint_skew_s'], .15)
        self.assertFalse(self.stop.stopped); self.g.hold.assert_not_called()

    def test_unavailable_old_frame_can_recover_but_never_authorizes_a_snapshot(self):
        self.camera.unavailable_until = self.camera.available_at
        frame = self.capture()
        self.assertEqual(frame['capture_monotonic'], self.camera.exposure)
        self.snap.assert_called_once()
        self.assertGreater(frame['image_wait_polls'], 10)

    def test_missing_new_exposure_times_out_without_serializing_old_image(self):
        self.camera.available_at = 100.
        with self.assertRaisesRegex(RuntimeError, '同步的新影像'):
            self.capture()
        self.snap.assert_not_called()
        self.assertLess(self.clock.now, 10.734 + SETTLED_IMAGE_WAIT_SECONDS + .07)
        self.assertTrue(self.stop.stopped); self.g.hold.assert_called_once()

    def test_transient_joint_drift_while_waiting_rejects_before_it_returns_to_origin(self):
        self.pose_at = lambda now: {**self.raw, 'elbow_flex': 2003 if 11.1 <= now < 11.3 else 2000}
        with self.assertRaisesRegex(RuntimeError, '仍在移動'):
            self.capture()
        self.assertLess(self.clock.now, 11.3)
        self.snap.assert_not_called(); self.g.hold.assert_called_once()

    def test_stop_during_metadata_wait_never_serializes_or_continues(self):
        self.stop.cancel_at = 11.1
        with self.assertRaisesRegex(RuntimeError, '已停止'):
            self.capture()
        self.snap.assert_not_called(); self.g.hold.assert_called_once()

    def test_stop_during_serialization_prevents_returning_a_successful_observation(self):
        def cancelled_snapshot():
            frame = self.serialize()
            self.stop.set()
            return frame
        self.snap.side_effect = cancelled_snapshot
        with self.assertRaisesRegex(RuntimeError, '已停止'):
            self.capture()
        self.snap.assert_called_once(); self.g.hold.assert_called_once()

    def test_reconnected_camera_never_pairs_a_new_generation_with_old_joint_window(self):
        self.camera.reconnect_at = 11.1
        with self.assertRaisesRegex(RuntimeError, '重新連接'):
            self.capture()
        self.snap.assert_not_called(); self.g.hold.assert_called_once()

    def test_invalid_timestamp_or_wrong_camera_metadata_cannot_trigger_serialization(self):
        for fields in ({'frame_capture_monotonic': float('nan')},
                       {'frame_capture_monotonic': True},
                       {'frame_capture_monotonic': 100.}, {'camera_source': 'd435'}):
            with self.subTest(fields=fields):
                self.clock.now = 10.; self.stop.stopped = False
                self.camera.metadata_override = fields
                with self.assertRaisesRegex(RuntimeError, '時刻無效'):
                    self.capture()
        self.snap.assert_not_called()

    def test_snapshot_must_match_new_exposure_and_remain_fresh_after_serialization(self):
        for fields, delay, reason in [({'capture_monotonic': 10.73}, .04, '同步的新影像'),
                                      ({'capture_monotonic': 100.}, .04, '不在關節觀察期間'),
                                      ({'camera_generation': 'another-wrist'}, .04, '重新連接'),
                                      ({}, 1.1, '已過期')]:
            with self.subTest(fields=fields, delay=delay):
                self.clock.now = 10.; self.stop.stopped = False
                self.snapshot_override = fields; self.snapshot_delay = delay
                with self.assertRaisesRegex(RuntimeError, reason):
                    self.capture()
                self.assertTrue(self.stop.stopped)


if __name__ == '__main__':
    unittest.main()
