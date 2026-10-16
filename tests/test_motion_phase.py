"""Full serial driver with a delayed fake plant; no device or camera access."""
import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scservo_sdk import COMM_SUCCESS
from server.local_cartesian import ALL, NAMES
from server.motion_program import MotionPathViolation, SegmentPhaseGovernor, prepare_joint_segment, _tube
from server.robot import Robot
from server.gripper_power import SessionCancelled
from test_local_feedback import Clock, FakeStop, FakeCamera
from test_motion_program import Model


class LagPlant:
    def __init__(self, clock, calibration, initial):
        self.clock = clock
        self.reg = {c['id']: {40: 1, 42: initial[n], 56: initial[n], 63: 30, 65: 0, 66: 1}
                    for n, c in calibration.items()}
        self.position = {i: float(v[56]) for i, v in self.reg.items()}
        self.offset = {i: 0 for i in self.reg}
        self.last = clock.now
        self.writes = []
        self.jump = True
        self.stuck = False
        self.inject_deviation = False

    def read(self, port, servo, address):
        self.clock.now += .004  # Sequential status reads consume real control time.
        if self.jump and len(self.writes) >= 4:
            self.clock.now += .7  # A CPU/USB pause must not cause command catch-up.
            self.jump = False
        dt = self.clock.now - self.last
        self.last = self.clock.now
        if not self.stuck:
            for i in self.position:
                tau = .18 if i == 4 else .09
                self.position[i] += (self.reg[i][42] + self.offset[i] - self.position[i]) * (1 - math.exp(-dt / tau))
                self.reg[i][56] = round(self.position[i])
        if self.inject_deviation and len(self.writes) >= 4:
            self.reg[1][56] = 2400
            self.reg[4][56] = 2373
        return self.reg[servo][address], COMM_SUCCESS, 0

    read1ByteTxRx = read
    read2ByteTxRx = read

    def write(self, port, servo, address, value):
        self.reg[servo][address] = value
        self.writes.append((self.clock.now, servo, address, value))
        return COMM_SUCCESS, 0

    write1ByteTxRx = write
    write2ByteTxRx = write


class MotionPhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.addCleanup(patch.stopall)
        patch('server.robot.time.monotonic', self.clock.time).start()
        self.raw = dict(zip(ALL, [2456, 937, 3039, 2373, 878, 1859]))
        self.target = dict(zip(ALL, [2307, 1037, 3039, 2200, 885, 1859]))
        cal = {n: {'id': i + 1, 'range_min': 0, 'range_max': 4095} for i, n in enumerate(ALL)}
        self.robot = Robot({'follower_port': 'fake://no-hardware',
                           'max_ticks_per_second': 90, 'max_tracking_error_ticks': 90},
                           cal, self.tmp.name, FakeCamera())
        self.bus = LagPlant(self.clock, cal, self.raw)
        r = self.robot
        r.ph = self.bus; r.port = SimpleNamespace(is_using=False); r.connected = r.armed = True
        r.goals = dict(self.raw); r.stop = FakeStop(self.clock)
        r.gripper_power.stop_latch = r.stop.set
        self.segment = prepare_joint_segment(Model(), self.raw, self.raw, self.target,
                                             endpoint_tolerance_ticks=16, stationary_tolerance_ticks=12)
        self.observations = []

    def run_path(self, *, governed=True, observe=None):
        s = self.segment
        def measured(pose):
            if self.clock.now > 30:
                raise RuntimeError('deadline')
            _tube(pose, self.raw, self.target, s['actual_tracking_envelope_ticks'], source='measured')
        def report(event):
            self.observations.append(dict(event))
            if observe: observe(event)
        self.robot.glide({n: self.target[n] for n in s['active_joints']}, speed=60,
            precision=16, feedback=False, allow_rearm=False,
            raw_bounds=s['raw_bounds'], measured_bounds=s['measured_bounds'],
            guard=lambda pose: _tube(pose, self.raw, self.target, s['uncertainty_ticks']),
            measured_guard=measured, observe=report,
            phase_governor=SegmentPhaseGovernor(s) if governed else None)

    def test_slow_motor_and_io_jump_complete_inside_original_tube(self):
        self.run_path()
        self.assertFalse(self.robot.stop.is_set())
        self.assertLess(self.clock.now, 30)
        self.assertFalse(self.bus.jump)
        for n in self.segment['active_joints']:
            self.assertLessEqual(abs(self.robot.position[n] - self.target[n]), 16)
        self.assertGreater(len(self.observations), 5)
        self.assertTrue(any(e['fraction'] < e['nominal_fraction'] - .01 for e in self.observations))
        self.assertTrue(all(e['fraction'] <= e['speed_limited_fraction'] + 1e-12 for e in self.observations))
        previous = {}
        for stamp, servo, address, value in self.bus.writes:
            if address != 42: continue
            if servo in previous:
                before_time, before_goal = previous[servo]
                # One tick permits integer goal quantization, not a bank of
                # unused waiting time or a jump to elapsed nominal progress.
                self.assertLessEqual(abs(value - before_goal), 60 * (stamp - before_time) + 1 + 1e-8)
            previous[servo] = (stamp, value)
        self.assertEqual([e for e in self.bus.writes if e[2] == 40], [])

    def test_same_io_jump_without_governor_reproduces_prospective_stop(self):
        with self.assertRaises(MotionPathViolation) as caught:
            self.run_path(governed=False)
        self.assertEqual(caught.exception.guard_context['checked_pose_kind'], 'prospective_goal')
        self.assertTrue(self.robot.stop.is_set())

    def test_real_hold_bias_is_preserved_through_governed_motion(self):
        held = dict(zip(ALL, [2459, 930, 3040, 2376, 874, 1859]))
        self.robot.goals = dict(held)
        self.robot.bias = {n: held[n] - self.raw[n] for n in ALL}
        for servo, n in enumerate(ALL, 1):
            self.bus.reg[servo][42] = held[n]
            self.bus.offset[servo] = self.raw[n] - held[n]
        self.segment = prepare_joint_segment(Model(), self.raw, held, self.target,
                                             endpoint_tolerance_ticks=16, stationary_tolerance_ticks=12)
        self.run_path()
        self.assertFalse(self.robot.stop.is_set())
        for n in NAMES:
            self.assertEqual(self.robot.goals[n], self.target[n] + held[n] - self.raw[n])
            self.assertEqual(self.robot.bias[n], held[n] - self.raw[n])
            self.assertLessEqual(abs(self.robot.position[n] - self.target[n]), 16)

    def test_stuck_motors_have_finite_wait_and_stop(self):
        self.bus.stuck = True
        with self.assertRaisesRegex(RuntimeError, '沒有繼續跟上'):
            self.run_path()
        self.assertTrue(self.robot.stop.is_set())
        self.assertLess(self.clock.now, 6)

    def test_real_measured_departure_is_never_reclassified_as_wait(self):
        self.bus.inject_deviation = True
        with self.assertRaises(MotionPathViolation) as caught:
            self.run_path()
        self.assertEqual(caught.exception.path_diagnostic['check_source'], 'measured')
        self.assertTrue(self.robot.stop.is_set())

    def test_stop_during_governed_path_does_not_wait_for_stall_timeout(self):
        def stop(event):
            if event['commanded_fraction'] > .03: self.robot.stop.set()
        with self.assertRaises(SessionCancelled):
            self.run_path(observe=stop)
        self.assertTrue(self.robot.stop.is_set())
        self.assertLess(self.clock.now, 2)


if __name__ == '__main__':
    unittest.main()
