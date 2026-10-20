import json
import math
import unittest
from pathlib import Path

import numpy as np

from server.geometry import Kinematics
from server.kinematic_offsets import OffsetKinematics
from server.kinematic_grasp import (
    Geometry,
    cylinder_from_feet,
    extract_grasp_query_info,
    foveate_poses,
    is_kinematic_grasp_command,
    look_plan,
    plan_grasp,
    BODY,
    HOME,
)
from server.local_cartesian import default_model
from server.spatial_runtime import _profile_snapshot, geometry_revision

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'assets/so101_follower_cam.urdf'


class OffsetKinematicsTests(unittest.TestCase):
    def test_offsets_shift_joint_angles_only(self):
        q = {'shoulder_pan': 2100, 'shoulder_lift': 1500, 'elbow_flex': 2600, 'wrist_flex': 2200, 'wrist_roll': 900}
        plain, shifted = Kinematics(URDF), OffsetKinematics(URDF, {'elbow_flex': 31.5})
        moved = dict(q, elbow_flex=q['elbow_flex'] + 31.5)
        np.testing.assert_allclose(shifted.transforms(q)['gripper_link'], plain.transforms(moved)['gripper_link'], atol=1e-12)
        self.assertEqual(OffsetKinematics(URDF).offsets, {})

    def test_offsets_reject_gripper_and_nonfinite(self):
        with self.assertRaises(ValueError):
            OffsetKinematics(URDF, {'gripper': 3})
        with self.assertRaises(ValueError):
            OffsetKinematics(URDF, {'elbow_flex': math.nan})

    def test_revision_binds_offsets(self):
        plain = default_model(ROOT, offsets={})
        a = default_model(ROOT, offsets={'wrist_flex': -15.0})
        b = default_model(ROOT, offsets={'wrist_flex': -16.0})
        self.assertEqual(len({geometry_revision(plain), geometry_revision(a), geometry_revision(b)}), 3)

    def test_default_config_keeps_astra_revision_offset_free(self):
        self.assertEqual(default_model(ROOT).fk.offsets if hasattr(default_model(ROOT).fk, 'offsets') else {}, {})


class KinematicGraspGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((ROOT / 'config/robot.json').read_text())
        if not config.get('kinematic_model'):
            raise unittest.SkipTest('no kinematic model configured')
        jaw = _profile_snapshot(default_model(ROOT), config, include_jaws=True)[2]
        cls.geo = Geometry(ROOT, config, jaw)

    def synthetic_feet(self, q, centre, r, height):
        geo = self.geo
        C = geo.camera(q)[:3, 3]
        F = C - geo.n * (geo.n @ C - geo.d)
        u = centre - F
        u -= geo.n * (geo.n @ u)
        d = np.linalg.norm(u)
        u /= d
        perp = np.cross(geo.n, u)
        a = math.acos(r / d)
        t1 = centre - u * r * math.cos(a) + perp * r * math.sin(a)
        t2 = centre - u * r * math.cos(a) - perp * r * math.sin(a)
        uv, _ = geo.project(q, np.array([t1, t2, centre + geo.n * height]))
        return uv

    def test_cylinder_recovered_from_exact_feet(self):
        geo = self.geo
        q = look_plan(geo)['views'][0]['pose']
        centre = np.array([0.33, 0.02, 0.0])
        centre -= geo.n * (geo.n @ centre - geo.d)
        uv = self.synthetic_feet(q, centre, 0.0175, 0.145)
        est = cylinder_from_feet(geo, q, uv[0], uv[1], uv[2])
        self.assertLess(np.linalg.norm(np.array(est['centre']) - centre), 5e-4)
        self.assertAlmostEqual(est['radius'], 0.0175, delta=3e-4)
        self.assertAlmostEqual(est['height'], 0.145, delta=2e-3)

    def test_look_views_only_turn_the_shoulder_and_stay_high(self):
        plan = look_plan(self.geo)
        for v in plan['views']:
            self.assertEqual({n: v['pose'][n] for n in BODY if n != 'shoulder_pan'},
                             {n: plan['home'][n] for n in BODY if n != 'shoulder_pan'})
            self.assertGreater(v['transit_m'], 0.12)

    def test_directional_look_plan_turns_left_first(self):
        left_plan = look_plan(self.geo, direction_hint='left')
        self.assertLess(left_plan['views'][0]['pan_offset'], 0)
        right_plan = look_plan(self.geo, direction_hint='right')
        self.assertGreater(right_plan['views'][0]['pan_offset'], 0)

    def test_extract_directional_grasp_instruction(self):
        q = '往左找到牙籤罐，把它拿起來再放回去'
        self.assertTrue(is_kinematic_grasp_command(q))
        info = extract_grasp_query_info(q)
        self.assertEqual(info['direction_hint'], 'left')
        self.assertEqual(info['target_query'], '牙籤罐')
        self.assertTrue(info['return_object'])

    def test_foveate_poses_centers_bottom_edge_toothpick_box(self):
        view_frame = {
            'id': 'test-toothpick',
            'joints': dict(HOME, shoulder_pan=1684, gripper=2036),
        }
        answer = {
            'bottom_left': {'x': 0.519, 'y': 0.903},
            'bottom_right': {'x': 0.642, 'y': 0.939},
            'top_center': {'x': 0.606, 'y': 0.651},
            'bottom_fully_visible': False,
            'top_visible': True,
        }
        poses = foveate_poses(self.geo, view_frame, answer)
        self.assertTrue(len(poses) >= 1)
        # Object at bottom-right of image (x > 0.5, y > 0.9) requires turning right (pan > 1684) and tilting down (wrist_flex > 2318)
        self.assertGreater(poses[0]['shoulder_pan'], 1684)
        self.assertGreater(poses[0]['wrist_flex'], HOME['wrist_flex'])

    def test_plan_clears_table_and_object(self):
        geo = self.geo
        centre = np.array([0.30, 0.03, 0.0])
        centre -= geo.n * (geo.n @ centre - geo.d)
        obj = {'centre': centre.tolist(), 'radius': 0.0175, 'height': 0.145}
        start = look_plan(geo)['home']
        plan = plan_grasp(geo, obj, 2033, start)
        self.assertEqual(plan['problems'], [])
        labels = [w['label'] for w in plan['waypoints']]
        self.assertEqual(labels[0], 'above_pregrasp')
        self.assertEqual(labels[-2:], ['grasp', 'lift'])
        for w in plan['waypoints']:
            self.assertGreaterEqual(w['table_mm'], 10)
            if w['object_mm'] is not None:
                self.assertGreaterEqual(w['object_mm'], 4)

    def test_short_left_toothpick_container_plan_succeeds(self):
        geo = self.geo
        centre = np.array([0.197, 0.088, 0.0])
        centre -= geo.n * (geo.n @ centre - geo.d)
        obj = {'centre': centre.tolist(), 'radius': 0.025, 'height': 0.068}
        start = dict(HOME, shoulder_pan=1684)
        plan = plan_grasp(geo, obj, 2033, start)
        self.assertEqual(plan['problems'], [])

    def test_foveated_toothpick_box_plans_quickly_without_problems(self):
        import time
        geo = self.geo
        centre = np.array([0.22866, 0.08846, 0.0])
        centre -= geo.n * (geo.n @ centre - geo.d)
        obj = {'centre': centre.tolist(), 'radius': 0.02067, 'height': 0.0903, 'foveated': True}
        start = {'shoulder_pan': 1756, 'shoulder_lift': 1105, 'elbow_flex': 2917, 'wrist_flex': 2639, 'wrist_roll': 998}
        t0 = time.monotonic()
        plan = plan_grasp(geo, obj, 2031, start)
        elapsed = time.monotonic() - t0
        self.assertEqual(plan['problems'], [])
        self.assertLess(elapsed, 12.0)

    def test_too_wide_object_is_refused(self):
        geo = self.geo
        centre = np.array([0.30, 0.0, 0.0])
        centre -= geo.n * (geo.n @ centre - geo.d)
        plan = plan_grasp(geo, {'centre': centre.tolist(), 'radius': 0.030, 'height': 0.145}, 2033, look_plan(geo)['home'])
        self.assertTrue(plan['problems'])

    def test_lying_flat_tipped_toothpick_box_lowers_grasp_height_and_shifts_centre(self):
        geo = self.geo
        q = {'shoulder_pan': 1811, 'shoulder_lift': 1106, 'elbow_flex': 2916, 'wrist_flex': 2578, 'wrist_roll': 998}
        left_uv = (938.288, 667.44)
        right_uv = (1205.76, 610.2)
        top_uv = (1036.8, 272.16)
        upright = cylinder_from_feet(geo, q, left_uv, right_uv, top_uv, lying_flat=False)
        flat = cylinder_from_feet(geo, q, left_uv, right_uv, top_uv, lying_flat=True)
        # Upright misinterpretation gave > 94 mm height; lying_flat gives ~46 mm cross-section height
        self.assertGreater(upright['height'], 0.094)
        self.assertLessEqual(flat['height'], 0.052)
        self.assertTrue(flat['lying_flat'])
        # Centre is shifted deeper along the horizontal axis into the middle of the lying-down box
        self.assertGreater(np.linalg.norm(np.asarray(flat['centre'])[:2]), np.linalg.norm(np.asarray(upright['centre'])[:2]) + 0.015)
        plan = plan_grasp(geo, flat, 2033, q)
        self.assertEqual(plan['problems'], [])
        self.assertLessEqual(plan['grasp']['height_m'], 0.035)

    def test_loop_engineer_diagnoses_missed_empty_close_and_updates_tuning(self):
        import tempfile
        from server.kinematic_grasp import KinematicGraspProgram
        from server.loop_engineer import LoopEngineer
        from server.robot import RETRYABLE_COMM_RESULTS
        self.assertEqual(len(RETRYABLE_COMM_RESULTS), 2)
        with tempfile.TemporaryDirectory() as tmp:
            prog = KinematicGraspProgram(None, None, None, tmp, ROOT)
            le = LoopEngineer(ROOT, Path(tmp), None, None, prog)
            diag = le.diagnose_attempt(
                query='toothpick container',
                plan_summary={'plan_id': 'test-plan', 'object': {'centre': [0.228, 0.088, 0.038], 'radius': 0.025, 'height': 0.108}, 'grasp': {'height_m': 0.065}},
                exec_result={'plan_id': 'test-plan', 'grasp_confirmed': False, 'missed_empty_close': True},
                error_text='夾爪已接近空夾位置（抓空）',
            )
            self.assertEqual(diag['updated_tuning'].get('force_posture'), 'lying_flat')
            self.assertLess(diag['updated_tuning'].get('height_offset_mm', 0.0), 0.0)
            self.assertGreater(diag['updated_tuning'].get('depth_offset_mm', 0.0), 0.0)


if __name__ == '__main__':
    unittest.main()

