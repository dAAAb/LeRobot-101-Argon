"""A finite fake breakaway threshold exposes a common-phase start deadlock.

This is a model of one possible servo response, not a measured diagnosis of the
physical elbow. It changes no force, motor setting, tolerance, or real device.
"""
import json
from pathlib import Path
import unittest

from server.motion_program import prepare_joint_segment, _tube
import test_motion_phase as phase
from test_motion_program import Model


class BreakawayPlant(phase.LagPlant):
    def __init__(self, clock, calibration, initial, *, elbow_threshold_ticks=25):
        super().__init__(clock, calibration, initial)
        self.elbow = calibration['elbow_flex']['id']
        self.initial_elbow = initial['elbow_flex']
        self.elbow_threshold_ticks = elbow_threshold_ticks
        self.elbow_started = False
        self.jump = False

    def read(self, port, servo, address):
        goal = self.reg[self.elbow][42]
        if abs(goal - self.initial_elbow) >= self.elbow_threshold_ticks:
            self.elbow_started = True
        if not self.elbow_started:
            # Preserve the commanded register while suppressing physical
            # response until this fake motor's initial breakaway threshold.
            self.reg[self.elbow][42] = self.initial_elbow
        try:
            return super().read(port, servo, address)
        finally:
            self.reg[self.elbow][42] = goal

    read1ByteTxRx = read
    read2ByteTxRx = read


class MotionBreakawayTests(unittest.TestCase):
    def fixture(self, target):
        fixture = phase.MotionPhaseTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.raw = dict(shoulder_pan=2181, shoulder_lift=1032, elbow_flex=3042,
                           wrist_flex=2174, wrist_roll=884, gripper=1861)
        fixture.target = {**fixture.raw, **target}
        fixture.robot.goals = dict(fixture.raw)
        fixture.robot.bias = {n: 0 for n in fixture.raw}
        fixture.bus = BreakawayPlant(fixture.clock, fixture.robot.calibration, fixture.raw)
        fixture.robot.ph = fixture.bus
        fixture.segment = prepare_joint_segment(Model(), fixture.raw, fixture.raw, fixture.target,
            endpoint_tolerance_ticks=16, stationary_tolerance_ticks=12)
        return fixture

    def test_common_phase_can_stall_before_a_lagging_motor_gets_its_start_threshold(self):
        fixture = self.fixture(dict(shoulder_pan=2341, shoulder_lift=1122,
                                    elbow_flex=3000, wrist_flex=2109))
        with self.assertRaisesRegex(RuntimeError, '沒有繼續跟上'):
            fixture.run_path()
        failure = json.loads((Path(fixture.tmp.name) / 'last-motion-failure.json').read_text())
        self.assertEqual(failure['actual']['elbow_flex'], 3042)
        self.assertFalse(fixture.bus.elbow_started)
        self.assertGreater(failure['goals']['elbow_flex'], 3042 - 25)
        self.assertGreater(failure['actual']['shoulder_pan'], 2181 + 40)
        # The failure occurs while still inside the original path tube;
        # it is not an asynchronous path departure or a measured collision.
        lower, upper = _tube(failure['actual'], fixture.raw, fixture.target,
                            fixture.segment['actual_tracking_envelope_ticks'])
        self.assertLessEqual(lower, upper)
        self.assertLess(fixture.clock.now, 8.)
        self.assertFalse(any(address == 40 for _, _, address, _ in fixture.bus.writes))

    def test_separately_screened_elbow_station_starts_without_changing_force_or_tolerance(self):
        fixture = self.fixture({'elbow_flex': 3000})
        fixture.run_path()
        self.assertTrue(fixture.bus.elbow_started)
        self.assertFalse(fixture.robot.stop.is_set())
        self.assertLessEqual(abs(fixture.robot.position['elbow_flex'] - 3000), 16)
        self.assertLess(fixture.clock.now, 8.)
        for _, servo, address, _ in fixture.bus.writes:
            if address == 42:
                self.assertEqual(servo, fixture.bus.elbow)
            self.assertNotEqual(address, 40)


if __name__ == '__main__':
    unittest.main()
