"""Whole translation, real CAD with a synthetic plant; no hardware/cloud I/O."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from server.grasp import WristGrasp
from server.gripper_power import SessionCancelled
from server.local_cartesian import ALL, default_model
from server.skill_graph import checkpoint
from server.skill_runtime import SkillRunner
from server.translation_program import run_translation
from test_motion_program import Robot


def step(distance=10., direction='up'):
    return {'skill': 'move', 'direction': direction, 'distance_mm': distance,
            'target': '', 'label': '移動'}


class TranslationProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'wrist.png'
        cv2.imwrite(str(self.path), np.full((270, 480, 3), 128, dtype=np.uint8))
        self.raw = dict(zip(ALL, [2161, 1135, 2490, 2328, 1100, 1858]))
        self.robot = Robot(self.raw, {**self.raw, 'elbow_flex': 2482, 'shoulder_lift': 1128})
        self.brain = SimpleNamespace(_infer=Mock(return_value={
            'confidence': .78, 'gripper_empty': True, 'human_clear': True,
            'clear_path': True, 'reason': '整段移動有空間'}))
        self.grasp = WristGrasp(self.robot, self.brain, self.capture, self.tmp.name)
        self.grasp._model = self.model; self.grasp.capture = self.capture
        self.grasp.record = Mock()
        self.cancel = threading.Event(); self.report = Mock()
        self.frame_change = {}

    def capture(self):
        return {'source': 'wrist', 'path': str(self.path),
                'camera_generation': self.robot.camera.generation,
                'joints': dict(self.robot.position),
                'capture_monotonic': time.monotonic(), **self.frame_change}

    def run_move(self, value=None):
        return run_translation(self.grasp, self.brain, value or step(), self.report, self.cancel)

    def test_ten_and_twenty_mm_each_use_one_inference_and_one_continuous_dispatch(self):
        for distance in (10., 20.):
            with self.subTest(distance=distance):
                self.robot.position = dict(self.raw)
                self.robot.goals = {**self.raw, 'elbow_flex': 2482, 'shoulder_lift': 1128}
                self.robot.calls.clear(); self.brain._infer.reset_mock()
                receipt = self.run_move(step(distance))
                self.assertTrue(receipt['completed'])
                self.assertEqual(receipt['planning_calls'], 1)
                self.assertEqual(receipt['step_count'], 1)
                self.assertEqual(len(self.robot.calls), 1)
                self.brain._infer.assert_called_once()
                self.assertTrue(checkpoint(step(distance), receipt)['passed'])
                data = json.loads(self.brain._infer.call_args.args[0].split('INPUT_DATA_JSON\n')[1])
                self.assertEqual(data['task']['distance_mm'], distance)
                self.assertIn('actual_tracking_envelope_ticks', data['controller_contract'])
                self.assertEqual(len(data['sampled_tip_path_mm']), 17)

    def test_default_skill_runner_dispatches_new_whole_program(self):
        runner = SkillRunner(self.grasp, self.brain)
        runner._move_incremental = Mock(side_effect=AssertionError('legacy movement'))
        result = runner._move(step(), self.report, self.cancel)
        self.assertEqual(result['execution_mode'], 'planned_segment')
        runner._move_incremental.assert_not_called()
        self.assertEqual(len(self.robot.calls), 1)

    def test_blocked_whole_path_does_not_execute_or_retry_short_steps(self):
        self.brain._infer.return_value['clear_path'] = False
        with self.assertRaisesRegex(RuntimeError, '先不走'):
            self.run_move()
        self.brain._infer.assert_called_once()
        self.assertFalse(self.robot.calls)

    def test_cancel_during_inference_has_no_dispatch(self):
        verdict = dict(self.brain._infer.return_value)
        def cancel(*args, **kwargs):
            self.cancel.set(); return verdict
        self.brain._infer.side_effect = cancel
        with self.assertRaises(SessionCancelled): self.run_move()
        self.assertFalse(self.robot.calls)

    def test_scene_change_after_approval_cannot_execute(self):
        verdict = dict(self.brain._infer.return_value)
        def change(*args, **kwargs):
            self.robot.position['elbow_flex'] += 4
            return verdict
        self.brain._infer.side_effect = change
        with self.assertRaisesRegex(ValueError, '姿勢已改變'): self.run_move()
        self.assertFalse(self.robot.calls)

    def test_stale_or_observer_image_never_enters_brain(self):
        for changes in ({'source': 'd435'}, {'capture_monotonic': time.monotonic() - 30}):
            with self.subTest(changes=changes):
                self.frame_change = changes
                with self.assertRaises(ValueError): self.run_move()
        self.brain._infer.assert_not_called(); self.assertFalse(self.robot.calls)

    def test_no_geometry_solution_never_calls_brain_or_robot(self):
        with patch.object(self.model, 'local_ik', side_effect=ValueError('unreachable')):
            with self.assertRaisesRegex(ValueError, 'unreachable'): self.run_move()
        self.brain._infer.assert_not_called(); self.assertFalse(self.robot.calls)

    def test_one_mm_request_does_not_claim_zero_or_reverse_progress_as_completion(self):
        for progress in (0., -.5):
            with self.subTest(progress=progress):
                tip = self.model.tip(self.raw).copy(); tip[2, 3] += progress / 1000
                actual_tip = self.model.tip
                marker = {**self.raw, 'gripper': 1859}
                with patch('server.motion_program.execute_prepared_segment',
                           return_value={'after': {'joints': marker}}), \
                     patch.object(self.model, 'tip', side_effect=lambda q: tip if q == marker else actual_tip(q)):
                    with self.assertRaisesRegex(RuntimeError, '尚未完成'): self.run_move(step(1.))
                receipt = {'skill': 'move', 'completed': True, 'requested_mm': 1.,
                           'measured_model_progress_mm': progress, 'step_count': 1,
                           'execution_mode': 'planned_segment'}
                self.assertFalse(checkpoint(step(1.), receipt)['passed'])

    def test_owned_jaw_stopped_and_invalid_distance_never_start_planning(self):
        self.robot.pwm_owner = object()
        with self.assertRaises(RuntimeError): self.run_move()
        self.robot.pwm_owner = None; self.robot.stop.set()
        with self.assertRaises(SessionCancelled): self.run_move()
        self.robot.stop.clear()
        for value in (0, 31, True, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError): self.run_move(step(value))
        self.brain._infer.assert_not_called(); self.assertFalse(self.robot.calls)


if __name__ == '__main__': unittest.main()
