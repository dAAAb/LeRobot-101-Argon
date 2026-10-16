import json
import math
import unittest
from pathlib import Path

import numpy as np

from server.geometry import Kinematics
from server.kinematic_offsets import OffsetKinematics
from server.kinematic_grasp import Geometry, cylinder_from_feet, look_plan, plan_grasp, BODY
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

    def test_too_wide_object_is_refused(self):
        geo = self.geo
        centre = np.array([0.30, 0.0, 0.0])
        centre -= geo.n * (geo.n @ centre - geo.d)
        plan = plan_grasp(geo, {'centre': centre.tolist(), 'radius': 0.030, 'height': 0.145}, 2033, look_plan(geo)['home'])
        self.assertTrue(plan['problems'])


if __name__ == '__main__':
    unittest.main()
