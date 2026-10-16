"""Real move, driver, feedback and capture over an in-memory servo plant.

Synthetic FK is configurable controller-contract evidence, not robot accuracy
or obstacle clearance evidence. There is no serial, network or real camera I/O.
"""
import copy
import math
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scservo_sdk import COMM_SUCCESS

from server import grasp as gm
from server.gripper_power import SessionCancelled
from server.robot import JOINTS, Robot


class VirtualStop(threading.Event):
    def __init__(self, clock):
        super().__init__(); self.clock = clock

    def wait(self, timeout=None):
        self.clock.now += timeout or 0
        return self.is_set()


class VirtualWristCamera:
    """A 25-Hz sensor clock, independent of metadata reads or serialization."""
    def __init__(self, clock):
        self.clock = clock
        self.origin = clock.now
        self.generation = 'synthetic-wrist'

    def _exposure(self):
        index = math.floor((self.clock.now - self.origin) / .04)
        return index + 1, self.origin + index * .04

    def fresh(self):
        return 0 <= self.clock.now - self._exposure()[1] < 1.

    def latest_stamp(self):
        if not self.fresh():
            raise RuntimeError('腕部相機畫面已過期')
        sequence, captured = self._exposure()
        return {'frame_sequence': sequence, 'frame_capture_monotonic': captured,
                'camera_source': 'wrist'}


class HalfStepPlant:
    """Pan follows commands until six ticks, then settles rather than the target."""
    def __init__(self, calibration):
        self.reg = {c['id']: {40: 1, 42: 2000, 56: 2000, 60: 0, 63: 30, 65: 0, 66: 0, 69: 0}
                    for c in calibration.values()}
        self.events = []
        self.pan_limit = 2006
        self.after_positive_pan = None

    def read(self, port, motor, address):
        return self.reg[motor][address], COMM_SUCCESS, 0

    read2ByteTxRx = read
    read1ByteTxRx = read

    def write(self, port, motor, address, value):
        self.events.append((motor, address, value))
        self.reg[motor][address] = value
        if address == 42:
            self.reg[motor][56] = min(value, self.pan_limit) if motor == 1 else value
            if motor == 1 and value > 2000 and self.after_positive_pan:
                hook = self.after_positive_pan
                self.after_positive_pan = None
                hook()
        return COMM_SUCCESS, 0

    write2ByteTxRx = write
    write1ByteTxRx = write


class SyntheticTranslationModel:
    def tip(self, raw):
        pose = np.eye(4)
        pose[0, 3] = (raw['shoulder_pan'] - 2000) / 4000
        pose[1, 3] = (raw['shoulder_lift'] - 2000) / 4000
        return pose

    def floor_minimum(self, raw):
        return {'base_link': 0, 'tool': .1}

    def propose_adaptive(self, raw, delta, *, held_goals, uncertainty_ticks):
        ticks = round(delta[0] * 4000)
        target = {**raw, 'shoulder_pan': raw['shoulder_pan'] + ticks}
        return {'requested_delta_m': delta.tolist(), 'joint_targets': target,
                'physical_goal_targets': {name: held_goals[name] + target[name] - raw[name] for name in gm.NAMES}}


class ProgressIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        calibration = {name: {'id': index + 1, 'range_min': 0, 'range_max': 4095}
                       for index, name in enumerate(JOINTS)}
        self.clock = SimpleNamespace(now=100.)
        timer = patch('server.robot.time.monotonic', side_effect=lambda: self.clock.now)
        timer.start(); self.addCleanup(timer.stop)
        self.camera = VirtualWristCamera(self.clock)
        self.robot = Robot({'follower_port': 'fake://no-hardware', 'max_ticks_per_second': 90,
                            'max_tracking_error_ticks': 90}, calibration, self.tmp.name, self.camera)
        self.bus = HalfStepPlant(calibration)
        self.robot.ph = self.bus; self.robot.port = SimpleNamespace(is_using=False); self.robot.connected = True
        self.robot.armed = True; self.robot.goals = dict.fromkeys(JOINTS, 2000)
        self.robot.stop = VirtualStop(self.clock)
        self.robot.gripper_power.stop_latch = self.robot.stop.set
        self.sequence = 0; self.frame_hook = None
        self.grasp = gm.WristGrasp(self.robot, object(), self.snapshot, self.tmp.name)
        self.grasp._model = SyntheticTranslationModel()
        self.grasp.vector = lambda direction, distance, raw: np.array([distance / 1000, 0., 0.])
        self.events = []
        self.grasp.record = lambda event: self.events.append(copy.deepcopy(event))
        images = patch.object(gm.cv2, 'imread', return_value=np.zeros((24, 24), np.uint8))
        images.start(); self.addCleanup(images.stop)

    def snapshot(self):
        stamp = self.camera.latest_stamp()
        # Hooks count serialized snapshots, not sensor exposures. A metadata
        # poll cannot fire the final-capture drift hook or manufacture a frame.
        self.sequence += 1
        if self.frame_hook: self.frame_hook(self.sequence)
        return {'source': 'wrist', 'camera_generation': self.camera.generation,
                'capture_monotonic': stamp['frame_capture_monotonic'],
                'sequence': stamp['frame_sequence'],
                'path': 'synthetic-memory-only'}

    def move(self, *, partial=True, cancel=None):
        return self.grasp.move('right', 3., self.grasp.capture(), lambda message: None,
                               empty_hand_progress=partial, cancel=cancel)

    def assert_failed_without_receipt(self):
        self.assertTrue(self.robot.stop.is_set())
        self.assertIsNone(self.grasp.latest)
        self.assertFalse(any(event['type'] == 'movement_result' for event in self.events))

    def test_three_mm_request_accepts_measured_one_point_five_only_after_three_stable_samples(self):
        result = self.move()
        self.assertEqual(result['requested_mm'], 3.)
        self.assertEqual(result['accepted_mm'], 3.)
        self.assertEqual(result['actual_delta_model_mm'], [1.5, 0., 0.])
        self.assertEqual(result['completion'], 'partial_progress')
        self.assertFalse(result['exact_target_reached'])
        self.assertTrue(result['requires_reobservation'])
        self.assertFalse(result['progress']['task_success'])
        self.assertFalse(result['progress']['grasp_verified'])
        self.assertEqual(result['progress']['progress_fraction'], .5)
        samples = [event['decision'] for event in self.events if event['type'] == 'motor_feedback'
                   and event['decision'].get('phase') == 'cartesian_endpoint']
        self.assertEqual([event['accepted_samples'] for event in samples], [1, 2, 3])
        self.assertGreaterEqual(samples[-1]['read_start_monotonic'] - samples[0]['read_start_monotonic'], .239)
        self.assertFalse(self.robot.stop.is_set())
        self.assertFalse(any(motor == 6 for motor, address, value in self.bus.events))

    def test_same_half_step_remains_rejected_in_exact_grasp_verification_mode(self):
        self.grasp.state = 'verifying'
        with self.assertRaisesRegex(RuntimeError, '位置修正中止|未達到接受條件'):
            self.move(partial=False)
        self.assert_failed_without_receipt()
        self.assertFalse(any(event.get('decision', {}).get('status') == 'accepted' for event in self.events))

    def test_partial_mode_is_not_available_to_held_or_verifying_states(self):
        for state in ('holding', 'holding_unverified', 'verifying'):
            self.grasp.state = state
            with self.subTest(state=state), self.assertRaisesRegex(RuntimeError, '持有或接觸'):
                self.move()
        self.assertEqual(self.bus.events, [])

    def test_motor_fault_still_blocks_goal_and_latches_stop(self):
        self.bus.reg[1][65] = 8
        with self.assertRaisesRegex(RuntimeError, '馬達保護狀態'):
            self.move()
        self.assert_failed_without_receipt()
        self.assertFalse(any(motor == 1 for motor, address, value in self.bus.events))

    def test_overtemperature_still_stops_before_partial_acceptance(self):
        self.bus.reg[1][63] = 61
        with self.assertRaisesRegex(RuntimeError, '溫度過高'):
            self.move()
        self.assert_failed_without_receipt()

    def test_interrupted_camera_still_stops_in_trajectory(self):
        self.bus.after_positive_pan = lambda: setattr(self.camera, 'fresh', lambda: False)
        with self.assertRaisesRegex(RuntimeError, '腕部畫面中斷'):
            self.move()
        self.assert_failed_without_receipt()

    def test_changed_camera_generation_still_stops_in_trajectory(self):
        self.bus.after_positive_pan = lambda: setattr(self.camera, 'generation', 'reconnected-camera')
        with self.assertRaisesRegex(RuntimeError, '相機重新連接'):
            self.move()
        self.assert_failed_without_receipt()

    def test_jaw_drift_still_stops_before_partial_classification(self):
        self.bus.after_positive_pan = lambda: self.bus.reg[6].update({56: 2005})
        with self.assertRaisesRegex(RuntimeError, '夾爪開口改變'):
            self.move()
        self.assert_failed_without_receipt()

    def test_cancellation_still_stops_before_partial_acceptance(self):
        cancel = threading.Event()
        self.bus.after_positive_pan = cancel.set
        with self.assertRaises(SessionCancelled):
            self.move(cancel=cancel)
        self.assert_failed_without_receipt()

    def test_final_capture_drift_revokes_earlier_partial_acknowledgement(self):
        def drift(sequence):
            if sequence == 4:
                self.bus.reg[6][56] += 5
        self.frame_hook = drift
        with self.assertRaisesRegex(RuntimeError, '取像期間手臂仍在移動'):
            self.move()
        self.assert_failed_without_receipt()
        self.assertTrue(any(event.get('decision', {}).get('status') == 'accepted' for event in self.events))


if __name__ == '__main__':
    unittest.main()
