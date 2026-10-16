"""Real-model replay of the calibration-edge observation dead end; no devices."""
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from copy import deepcopy

from server.active_observation import build_observation_candidates
from server.local_cartesian import default_model, NAMES
from test_motion_program import Robot
import test_spatial_runtime as runtime_tests
from server import spatial_runtime


class ObservationScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()

    def test_recorded_edge_pose_chooses_in_scope_shorter_translations(self):
        raw = dict(zip((*NAMES, 'gripper'), [2180, 1032, 3042, 2174, 884, 1856]))
        bounds = dict(zip(NAMES, ([2180, 2456], [937, 1068], [3039, 3042], [2118, 2373], [569, 884])))
        robot = Robot(raw, raw)
        grasp = SimpleNamespace(model=self.model, robot=robot, state='idle')
        frame = {'source': 'wrist', 'path': 'recorded-pose-no-pixels-needed',
                 'camera_generation': 'offline', 'joints': raw, 'capture_monotonic': time.monotonic()}
        diag = {}
        candidates = build_observation_candidates(grasp, frame, calibrated_joint_bounds=bounds, diagnostics=diag)
        body = [c for c in candidates if c['kind'] == 'body']
        self.assertTrue(body, diag)
        for candidate in body:
            target = candidate['segment']['joint_targets']
            self.assertTrue(all(lo <= target[n] <= hi for n, (lo, hi) in bounds.items()), target)
        self.assertTrue(any(c['descriptor'].get('requested_translation_mm', 40) < 40 for c in body))
        self.assertTrue(any('校準' in r['reason'] for r in diag['rejections']))
        self.assertEqual(robot.calls, [])

    def test_bad_bounds_are_not_silently_ignored(self):
        raw = dict(zip((*NAMES, 'gripper'), [2180, 1032, 3042, 2174, 884, 1856]))
        grasp = SimpleNamespace(model=self.model, robot=Robot(raw, raw), state='idle')
        frame = {'source': 'wrist', 'path': 'unused', 'camera_generation': 'offline',
                 'joints': raw, 'capture_monotonic': time.monotonic()}
        for bad in ({'elbow_flex': [3039, 3042]}, {n: [0, float('nan')] for n in NAMES}):
            with self.assertRaises(ValueError):
                build_observation_candidates(grasp, frame, calibrated_joint_bounds=bad)


class RuntimeScopeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runtime_tests.SpatialRuntimeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_scope_is_camera_only_and_v2_motion_only(self):
        f = self.fixture
        config = {**f.config, 'motion_backend': 'task_space_v2', 'jaw_calibration_profile': '/missing-jaw-profile'}
        self.assertEqual(spatial_runtime.observation_pose_bounds(f.model, f.frame(), config),
                         f.profile['validation_scope']['joint_raw_bounds'])
        with patch.object(spatial_runtime, '_profile_snapshot', side_effect=AssertionError('legacy unchanged')):
            self.assertIsNone(spatial_runtime.observation_pose_bounds(f.model, f.frame(), {}))
            self.assertIsNone(spatial_runtime.observation_pose_bounds(f.model, f.frame(), f.config))


if __name__ == '__main__':
    unittest.main()
