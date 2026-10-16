"""Replay synchronization checks without a serial port or motor writes."""
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from server.local_cartesian import ALL, NAMES
from server.motion_program import MotionPathViolation, _tube, prepare_joint_segment, execute_prepared_segment
from server.robot import Robot
from test_motion_program import Grasp


class MotionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.origin = dict(zip(ALL, [2456, 937, 3039, 2373, 878, 1859]))
        self.target = dict(zip(ALL, [2307, 1037, 3039, 2200, 885, 1859]))
        self.actual = dict(zip(ALL, [2437, 945, 3039, 2361, 878, 1859]))
        self.envelope = {n: 12 if n == 'elbow_flex' else 16 for n in NAMES}
        self.goals = dict(zip(ALL, [2433, 947, 3040, 2346, 875, 1859]))
        self.robot = SimpleNamespace(stop=threading.Event(),
            camera=SimpleNamespace(fresh=lambda: True), goals=dict(self.goals),
            read=lambda: dict(self.actual), validate=lambda *a, **kw: None)

    def guard(self, pose):
        return _tube(pose, self.origin, self.target, self.envelope, source='replay')

    def test_recorded_encoders_pass_but_leading_single_goal_can_fail(self):
        lower, upper = self.guard(self.actual)
        self.assertAlmostEqual(lower, 3 / 149)
        self.assertAlmostEqual(upper, 28 / 173)
        # This is a possible next command, NOT a claim that it was the missing
        # command in the real log. The read-only replay demonstrates the gap.
        with self.assertRaises(MotionPathViolation) as caught:
            Robot._screen_local_goal(self.robot, {'shoulder_pan': 2414}, None, self.guard, 12)
        exc = caught.exception
        self.assertEqual(exc.guard_context['checked_pose_kind'], 'prospective_goal')
        self.assertEqual(exc.guard_context['actual'], self.actual)
        self.assertEqual(exc.guard_context['physical_goals'], self.goals)
        self.assertEqual(exc.path_diagnostic['rejected_pose']['shoulder_pan'], 2414)
        self.assertEqual(exc.path_diagnostic['joint_intervals']['wrist_roll']['fraction_interval'],
                         [-16 / 7, 16 / 7])
        self.assertGreater(exc.path_diagnostic['fraction_gap'], 0)
        self.assertLessEqual(exc.guard_context['read_start_monotonic'],
                             exc.guard_context['read_end_monotonic'])
        self.assertFalse(self.robot.stop.is_set())

    def test_true_measured_deviation_is_identified_before_prospective_check(self):
        self.actual['shoulder_pan'] = 2414
        with self.assertRaises(MotionPathViolation) as caught:
            Robot._screen_local_goal(self.robot, {'shoulder_pan': 2413}, None, self.guard, 12)
        self.assertEqual(caught.exception.guard_context['checked_pose_kind'], 'measured')
        self.assertIsNone(caught.exception.guard_context['prospective_pose'])
        self.assertEqual(caught.exception.path_diagnostic['rejected_pose'], self.actual)

    def test_segment_failure_record_keeps_pre_hold_violation_evidence(self):
        raw = {n: 2000 for n in ALL}
        target = {**raw, 'shoulder_pan': 2100, 'wrist_flex': 2100}
        g = Grasp(raw, raw)
        segment = prepare_joint_segment(g.model, raw, raw, target)
        def desynchronize(robot, kwargs):
            robot.position['shoulder_pan'] = 2050
            kwargs['measured_guard'](robot.position)
        g.robot.callback = desynchronize
        with patch('server.grasp.unchanged_wrist', return_value={}):
            with self.assertRaises(MotionPathViolation):
                execute_prepared_segment(g, segment, g.capture(), threading.Event(), lambda _: None)
        failure = next(e for e in g.events if e['type'] == 'motion_program_segment_failed')
        self.assertEqual(failure['path_diagnostic']['check_source'], 'measured_encoder_guard')
        self.assertEqual(failure['path_diagnostic']['rejected_pose']['shoulder_pan'], 2050)
        self.assertTrue(g.robot.stop.is_set())


if __name__ == '__main__':
    unittest.main()
