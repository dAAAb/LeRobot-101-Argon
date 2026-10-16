"""A new jaw opening/photo ID must not manufacture independent camera poses."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import test_cylinder_observation as observation
from server.cylinder_observation import distinct_camera_views


def record(name, x_mm=0., rotation_deg=0.):
    transform = np.eye(4)
    transform[0, 3] = x_mm / 1000.
    transform[:3, :3] = cv2.Rodrigues(np.array([0., 0., np.deg2rad(rotation_deg)]))[0]
    return dict(frame_id=name, camera_transform_base_m=transform.tolist())


class CameraPoseDiversityTests(unittest.TestCase):
    def test_latest_near_repeat_replaces_older_even_with_different_jaw_and_frame_id(self):
        rows = [record('first'), record('before-open', 8., 9.5), record('current-open', 8.07, 9.59)]
        rows[1]['measured_joints'] = {'gripper': 1860}
        rows[2]['measured_joints'] = {'gripper': 2034}
        kept, diagnostics = distinct_camera_views(rows)
        self.assertEqual(kept, [0, 2])
        self.assertEqual(diagnostics['distinct_pose_count'], 2)
        self.assertEqual(diagnostics['families'][-1]['frame_ids'], ['before-open', 'current-open'])
        self.assertFalse(diagnostics['actual_information_gain_verified'])

    def test_translation_or_rotation_can_distinguish_a_pose_both_are_needed_for_repeat(self):
        for rows in ([record('a'), record('b', 6.), record('c', 12.)],
                     [record('a'), record('b', rotation_deg=3.), record('c', rotation_deg=6.)]):
            kept, diagnostics = distinct_camera_views(rows)
            self.assertEqual(kept, [0, 1, 2])
            self.assertFalse(diagnostics['actual_information_gain_verified'])

    def test_small_step_chain_does_not_collapse_distinct_endpoints(self):
        kept, diagnostics = distinct_camera_views([record('a'), record('b', 4.), record('c', 8.)])
        self.assertEqual(kept, [0, 2])
        self.assertEqual(diagnostics['maximum_camera_translation_mm'], 8.)

    def test_thresholds_match_existing_observation_novelty(self):
        from server import active_observation, cylinder_observation
        self.assertEqual(active_observation.DUPLICATE_VIEW_TRANSLATION_MM,
                         cylinder_observation.DUPLICATE_VIEW_TRANSLATION_MM)
        self.assertEqual(active_observation.DUPLICATE_VIEW_ROTATION_DEGREES,
                         cylinder_observation.DUPLICATE_VIEW_ROTATION_DEGREES)


class ContourDiversityBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = observation.CylinderObservationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def make_last_a_jaw_only_repeat(self):
        fixture = self.fixture
        previous, current = fixture.frames[-2:]
        transform = fixture.packets[previous['id']].T_base_camera.as_array().copy()
        fixture.packets[current['id']].T_base_camera.as_array = lambda: transform.copy()
        fixture.packets[previous['id']].measured_joints = (('gripper', 1860),)
        fixture.packets[current['id']].measured_joints = (('gripper', 2034),)
        fixture.contours[-1] = {**deepcopy(fixture.contours[-2]), 'frame_id': current['id']}

    def test_three_photos_two_camera_poses_return_observation_need_without_solver(self):
        fixture = self.fixture
        fixture.frames = fixture.frames[:3]
        fixture.contours = fixture.contours[:3]
        self.make_last_a_jaw_only_repeat()
        with patch('server.cylinder_observation.fit_cylinder_silhouettes') as solve:
            result = fixture.fit()
        solve.assert_not_called()
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['reason'], '角度變化還不夠，需要移動肩肘換個位置。')
        self.assertEqual(result['frame_id'], fixture.frames[-1]['id'])
        self.assertEqual(result['camera_generation'], 'one-session')
        self.assertEqual(result['calibration_fingerprint'], 'synthetic-camera')
        self.assertFalse(result['planning_eligible'])
        self.assertFalse(result['execution_authorized'])
        self.assertNotIn('radius_m', result)
        self.assertNotIn('required_setup', result)
        current = result['silhouette_evidence'][-1]
        self.assertEqual(current['measured_joints']['gripper'], 2034)
        np.testing.assert_allclose(current['support_boundary_px'],
            np.asarray(fixture.contours[-1]['support_boundary_normalized']) * [1920, 1080])

    def test_three_real_poses_still_fit_once_preserving_current_pixels_and_binding(self):
        fixture = self.fixture
        self.make_last_a_jaw_only_repeat()
        from server.cylinder_observation import fit_cylinder_silhouettes
        with patch('server.cylinder_observation.fit_cylinder_silhouettes', wraps=fit_cylinder_silhouettes) as solve:
            result = fixture.fit()
        self.assertEqual(result['status'], 'hypothesis')
        self.assertAlmostEqual(result['radius_m'], fixture.fixture.radius, places=8)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual([row['frame_id'] for row in solve.call_args.args[0]],
                         [fixture.frames[i]['id'] for i in (0, 1, 3)])
        self.assertEqual(result['view_diversity']['input_frame_count'], 4)
        self.assertEqual(result['view_diversity']['distinct_pose_count'], 3)
        self.assertEqual(result['frame_id'], fixture.frames[-1]['id'])
        self.assertEqual(result['silhouette_evidence'][-1]['measured_joints']['gripper'], 2034)
        np.testing.assert_allclose(result['silhouette_evidence'][-1]['support_boundary_px'],
            np.asarray(fixture.contours[-1]['support_boundary_normalized']) * [1920, 1080])

    def test_distinct_rotations_do_not_bypass_existing_metric_baseline_check(self):
        fixture = self.fixture
        for i, frame in enumerate(fixture.frames):
            transform = np.asarray(record(str(i), rotation_deg=i * 3.)['camera_transform_base_m'])
            fixture.packets[frame['id']].T_base_camera.as_array = lambda t=transform: t.copy()
        with self.assertRaisesRegex(ValueError, 'rotation alone'):
            fixture.fit()

    def test_insufficient_diversity_keeps_existing_selected_observation_route(self):
        fixture = self.fixture
        fixture.frames = fixture.frames[:3]
        fixture.contours = fixture.contours[:3]
        self.make_last_a_jaw_only_repeat()
        result = fixture.fit()
        routing = observation.CylinderIntentRoutingTests()
        routing.setUpClass()
        routing.setUp()
        self.addCleanup(routing.doCleanups)
        with patch('server.cylinder_observation.hypothesis_from_wrist_contours', return_value=result):
            program = routing.run_plan()
        self.assertEqual(program.status, 'observe')
        self.assertEqual(program.observation.candidate_id, 'move_left')
        self.assertEqual(routing.fixture.grasp.spatial_attempt['required_setups'], [])


if __name__ == '__main__':
    unittest.main()
