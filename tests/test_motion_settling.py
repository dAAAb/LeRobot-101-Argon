"""Bounded read-only endpoint waiting with a delayed fake servo, no devices."""
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from server.gripper_power import SessionCancelled
from server.local_cartesian import ALL
from server.motion_program import (
    SegmentPhaseGovernor, WORKSPACE_ORIGIN_PROFILE, _tube, prepare_joint_segment,
    execute_prepared_segment,
)
from server.robot import Robot
from test_local_feedback import Clock, FakeStop
import test_motion_phase as phase
import test_motion_program as program


class DelayedEndpointPlant(phase.LagPlant):
    def __init__(self, clock, calibration, initial, target, *, delay=2.):
        super().__init__(clock, calibration, initial)
        self.elbow = calibration['elbow_flex']['id']
        self.target = target
        self.delay = delay
        self.final_goal_time = None
        self.stuck = True
        self.jump = False
        self.on_settling_read = None

    def write(self, port, servo, address, value):
        result = super().write(port, servo, address, value)
        if servo == self.elbow and address == 42 and value == self.target and self.final_goal_time is None:
            self.final_goal_time = self.clock.now
        return result

    write1ByteTxRx = write
    write2ByteTxRx = write

    def read(self, port, servo, address):
        result = super().read(port, servo, address)
        if self.final_goal_time is not None:
            elapsed = self.clock.now - self.final_goal_time
            if self.delay is not None and elapsed >= self.delay:
                self.reg[self.elbow][56] = self.target
            if self.on_settling_read is not None:
                self.on_settling_read(elapsed)
        return result

    read1ByteTxRx = read
    read2ByteTxRx = read


class MotionSettlingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        timer = patch('server.robot.time.monotonic', self.clock.time)
        timer.start(); self.addCleanup(timer.stop)
        self.raw = dict(zip(ALL, [2254, 1074, 3042, 2157, 884, 1861]))
        self.target = {**self.raw, 'elbow_flex': 3000}
        calibration = {n: {'id': i + 1, 'range_min': 0, 'range_max': 4095} for i, n in enumerate(ALL)}
        self.camera_ok = True
        self.robot = Robot({'follower_port': 'fake://no-hardware', 'max_ticks_per_second': 90,
                            'max_tracking_error_ticks': 90}, calibration, self.tmp.name,
                           SimpleNamespace(fresh=lambda: self.camera_ok))
        self.bus = DelayedEndpointPlant(self.clock, calibration, self.raw, 3000)
        self.robot.ph = self.bus; self.robot.port = SimpleNamespace(is_using=False)
        self.robot.connected = self.robot.armed = True
        self.robot.goals = dict(self.raw); self.robot.stop = FakeStop(self.clock)
        self.robot.gripper_power.stop_latch = self.robot.stop.set
        self.segment = prepare_joint_segment(program.Model(), self.raw, self.raw, self.target,
            endpoint_tolerance_ticks=16, stationary_tolerance_ticks=12)
        self.extra_guard = lambda actual: None

    def run_path(self, **kwargs):
        segment = self.segment
        def measured(actual):
            _tube(actual, self.raw, self.target, segment['actual_tracking_envelope_ticks'])
            self.extra_guard(actual)
        return self.robot.glide({'elbow_flex': 3000}, speed=60, precision=16, feedback=False,
            allow_rearm=False, raw_bounds=segment['raw_bounds'], measured_bounds=segment['measured_bounds'],
            guard=lambda actual: _tube(actual, self.raw, self.target, segment['uncertainty_ticks']),
            measured_guard=measured, phase_governor=SegmentPhaseGovernor(segment), **kwargs)

    def test_default_point_seven_second_wait_still_rejects_two_second_response(self):
        with self.assertRaisesRegex(RuntimeError, '未到達目標位置'):
            self.run_path()
        self.assertTrue(self.robot.stop.is_set())
        self.assertLess(self.clock.now - self.bus.final_goal_time, 1.2)

    def test_two_second_response_completes_with_no_extra_goal_or_force_write(self):
        self.run_path(settle_timeout_s=2.5)
        self.assertFalse(self.robot.stop.is_set())
        self.assertEqual(self.robot.position['elbow_flex'], 3000)
        self.assertGreaterEqual(self.clock.now - self.bus.final_goal_time, 2.)
        self.assertLess(self.clock.now - self.bus.final_goal_time, 2.5)
        self.assertFalse([event for event in self.bus.writes if event[0] > self.bus.final_goal_time])
        self.assertTrue(all(address == 42 for _, _, address, _ in self.bus.writes))
        self.assertEqual(self.robot.bias, {n: 0 for n in ALL})

    def test_no_response_still_stops_at_bounded_three_second_limit(self):
        self.bus.delay = None
        with self.assertRaisesRegex(RuntimeError, '未到達目標位置'):
            self.run_path(settle_timeout_s=3.)
        self.assertTrue(self.robot.stop.is_set())
        elapsed = self.clock.now - self.bus.final_goal_time
        self.assertGreaterEqual(elapsed, 3.)
        self.assertLess(elapsed, 3.4)

    def test_settling_telemetry_reports_measured_sequence_without_extra_io(self):
        events = []
        self.run_path(settle_timeout_s=2.5, settle_telemetry=True, observe=events.append)
        samples = [e for e in events if e.get('kind') == 'endpoint_settling']
        self.assertGreater(len(samples), 2)
        self.assertLessEqual(len(samples), 32)
        self.assertEqual(samples[0]['actual']['elbow_flex'], self.raw['elbow_flex'])
        self.assertEqual(samples[-1]['actual']['elbow_flex'], 3000)
        self.assertEqual(samples[0]['endpoint_error_ticks']['elbow_flex'], 42)
        self.assertEqual(samples[-1]['endpoint_error_ticks']['elbow_flex'], 0)
        self.assertTrue(all(e['goals']['elbow_flex'] == 3000 for e in samples))
        self.assertFalse([event for event in self.bus.writes if event[0] > self.bus.final_goal_time])
        telemetry_writes = list(self.bus.writes); telemetry_time = self.clock.now
        # Identical fake-plant timing and register writes without telemetry:
        # every serial read advances this clock, so extra reads would differ.
        self.setUp()
        self.run_path(settle_timeout_s=2.5)
        self.assertEqual(self.bus.writes, telemetry_writes)
        self.assertEqual(self.clock.now, telemetry_time)

    def test_invalid_wait_is_rejected_before_motor_access(self):
        for timeout in (True, False, 0, -1, 3.01, float('nan'), float('inf'), None, '2.5'):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.run_path(settle_timeout_s=timeout)
        self.assertEqual(self.bus.writes, [])
        self.assertFalse(self.robot.stop.is_set())

    def test_stop_remains_immediate_during_extended_settling(self):
        self.bus.on_settling_read = lambda elapsed: self.robot.stop.set() if elapsed > .9 else None
        with self.assertRaises(SessionCancelled):
            self.run_path(settle_timeout_s=2.5)
        self.assertLess(self.clock.now - self.bus.final_goal_time, 1.3)

    def test_camera_loss_still_aborts_during_extended_settling(self):
        def dark(elapsed):
            if elapsed > .9:
                self.camera_ok = False
        self.bus.on_settling_read = dark
        with self.assertRaisesRegex(RuntimeError, '腕部畫面中斷'):
            self.run_path(settle_timeout_s=2.5)
        self.assertLess(self.clock.now - self.bus.final_goal_time, 1.3)

    def test_temperature_limit_is_checked_without_waiting_for_endpoint_timeout(self):
        def hot(elapsed):
            if elapsed > .9:
                self.bus.reg[self.bus.elbow][63] = 61
        self.bus.on_settling_read = hot
        with self.assertRaisesRegex(RuntimeError, '溫度過高'):
            self.run_path(settle_timeout_s=2.5)
        self.assertLess(self.clock.now - self.bus.final_goal_time, 1.3)

    def test_existing_measured_guard_can_enforce_shorter_program_deadline(self):
        def deadline(actual):
            if self.bus.final_goal_time is not None and self.clock.now - self.bus.final_goal_time > .9:
                raise RuntimeError('existing program deadline reached')
        self.extra_guard = deadline
        with self.assertRaisesRegex(RuntimeError, 'existing program deadline'):
            self.run_path(settle_timeout_s=2.5)
        self.assertLess(self.clock.now - self.bus.final_goal_time, 1.3)

    def partial_survey(self, plant, **kwargs):
        self.segment = prepare_joint_segment(program.Model(), self.raw, self.raw, self.target,
            endpoint_tolerance_ticks=16, stationary_tolerance_ticks=12,
            fixed_origin=self.raw, origin_limit_profile=WORKSPACE_ORIGIN_PROFILE)
        self.bus.delay = None
        self.bus.on_settling_read = plant
        return self.run_path(settle_timeout_s=2.5, allow_stable_partial_observation=True, **kwargs)

    def test_static_31_tick_residual_is_a_partial_view_without_goal_rewrite(self):
        outcome = self.partial_survey(lambda elapsed: self.bus.reg[3].update({56: 3031}))
        self.assertEqual(outcome['completion'], 'stable_partial_observation')
        self.assertFalse(outcome['reached_nominal_endpoint'])
        self.assertEqual(outcome['actual']['elbow_flex'], 3031)
        self.assertEqual(outcome['goals']['elbow_flex'], 3000)
        self.assertEqual(outcome['endpoint_error_ticks']['elbow_flex'], 31)
        self.assertGreaterEqual(outcome['stable_duration_s'], .4)
        self.assertTrue(all(v <= 2 for v in outcome['stable_joint_span_ticks'].values()))
        self.assertFalse(self.robot.stop.is_set())
        self.assertFalse([event for event in self.bus.writes if event[0] > self.bus.final_goal_time])

    def test_stationary_oscillating_and_reverse_partial_poses_do_not_complete(self):
        plants = {'stationary': lambda elapsed: None,
            'seven_ticks_only': lambda elapsed: self.bus.reg[3].update({56: self.raw['elbow_flex'] - 7}),
            'oscillating': lambda elapsed: self.bus.reg[3].update({56: 3031 + (int(elapsed / .1) % 2) * 6}),
            'reverse': lambda elapsed: self.bus.reg[3].update({56: self.raw['elbow_flex'] + 1}),
            'fixed_axis_oscillating': lambda elapsed: (
                self.bus.reg[3].update({56: 3031}), self.bus.reg[1].update({56: 2254 + (int(elapsed / .1) % 2) * 6}))}
        for name, plant in plants.items():
            with self.subTest(name=name):
                self.setUp()
                with self.assertRaisesRegex(RuntimeError, '未到達目標位置'):
                    self.partial_survey(plant)
                self.assertTrue(self.robot.stop.is_set())

    def test_partial_survey_still_rejects_tube_departure_and_stop(self):
        with self.assertRaises((RuntimeError, ValueError)):
            self.partial_survey(lambda elapsed: self.bus.reg[3].update({56: 2980}))
        self.assertTrue(self.robot.stop.is_set())
        self.setUp()
        def stopped(elapsed):
            self.bus.reg[3][56] = 3031
            if elapsed > 1.2:
                self.robot.stop.set()
        with self.assertRaises(SessionCancelled):
            self.partial_survey(stopped)
        self.assertTrue(self.robot.stop.is_set())

    def test_partial_acceptance_requires_engineering_contract(self):
        with self.assertRaisesRegex(ValueError, 'engineering workspace'):
            self.run_path(allow_stable_partial_observation=True)
        self.assertEqual(self.bus.writes, [])

    def test_production_static_residual_still_requires_its_declared_endpoint(self):
        self.bus.delay = None
        self.bus.on_settling_read = lambda elapsed: self.bus.reg[3].update({56: 3031})
        with self.assertRaisesRegex(RuntimeError, '未到達目標位置'):
            self.run_path(settle_timeout_s=2.5)
        self.assertTrue(self.robot.stop.is_set())

    def test_real_driver_receipt_rechecks_actual_after_partial_before_returning_view(self):
        self.segment = prepare_joint_segment(program.Model(), self.raw, self.raw, self.target,
            endpoint_tolerance_ticks=16, stationary_tolerance_ticks=12,
            fixed_origin=self.raw, origin_limit_profile=WORKSPACE_ORIGIN_PROFILE)
        self.bus.delay = None
        self.bus.on_settling_read = lambda elapsed: self.bus.reg[3].update({56: 3031})
        self.robot.camera.generation = 'settling-fixture'
        frames = []
        def capture():
            frame = {'source': 'wrist', 'camera_generation': 'settling-fixture',
                     'capture_monotonic': self.clock.now, 'joints': self.robot.read()}
            frames.append(frame)
            return frame
        records = []
        grasp = SimpleNamespace(robot=self.robot, model=program.Model(), capture=capture, record=records.append)
        reference = capture()
        with patch('server.grasp.unchanged_wrist'):
            result = execute_prepared_segment(grasp, self.segment, reference,
                threading.Event(), lambda message: None, engineering_workspace=True)
        self.assertIs(result['after'], frames[-1])
        self.assertEqual(result['after']['joints']['elbow_flex'], 3031)
        self.assertEqual(result['completion'], 'stable_partial_observation')
        self.assertFalse(result['reached_nominal_endpoint'])
        self.assertFalse(result['observation_accuracy_verified'])
        self.assertEqual(result['joint_targets']['elbow_flex'], 3000)
        self.assertFalse(self.robot.stop.is_set())
        self.assertTrue(any(e['type'] == 'motion_program_endpoint_settling' for e in records))


class WorkspaceSettlingWiringTests(unittest.TestCase):
    def test_only_explicit_engineering_workspace_uses_longer_settling(self):
        fixture = program.MotionProgramTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.execute()
        self.assertNotIn('settle_timeout_s', fixture.grasp.robot.calls[-1][1])
        # A fresh fixture binds the workspace contract to the original pose.
        fixture = program.MotionProgramTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.target = {**fixture.raw, 'elbow_flex': 1900}
        segment = fixture.prepare(fixed_origin=fixture.raw, origin_limit_profile=WORKSPACE_ORIGIN_PROFILE)
        fixture.execute(segment, engineering_workspace=True)
        self.assertEqual(fixture.grasp.robot.calls[-1][1]['settle_timeout_s'], 2.5)
        self.assertTrue(fixture.grasp.robot.calls[-1][1]['settle_telemetry'])
        self.assertEqual(fixture.grasp.robot.calls[-1][1]['precision'], segment['endpoint_tolerance_ticks'])
        self.assertFalse(fixture.grasp.robot.calls[-1][1]['feedback'])


if __name__ == '__main__':
    unittest.main()
