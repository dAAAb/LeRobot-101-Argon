from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.kinematic_grasp import HOME, IMAGE, OPEN_RAW
from server.mira_real_to_sim import (
    ROOT,
    WorkspaceSceneMap,
    bev_norm_to_metric_3d,
    export_real_to_sim_twin,
    load_calibrated_geometry,
    metric_3d_to_bev_norm,
    plan_mira_grasp_or_place,
    solve_oriented_object,
    synthesize_orthographic_bev,
)


class MiraRealToSimTest(unittest.TestCase):
    def setUp(self):
        self.geo = load_calibrated_geometry(ROOT)

    def test_bev_metric_roundtrip(self):
        p = bev_norm_to_metric_3d(420, 580, self.geo, height_m=0.0)
        yn, xn = metric_3d_to_bev_norm(p, self.geo)
        self.assertAlmostEqual(yn, 420, delta=2)
        self.assertAlmostEqual(xn, 580, delta=2)

    def test_orthographic_bev_and_scene_twin_and_pick_place_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create a synthetic wrist image
            img = np.full((IMAGE[1], IMAGE[0], 3), (210, 215, 220), dtype=np.uint8)
            cv2.circle(img, (960, 540), 90, (50, 180, 70), -1)
            frame_file = tmp_path / 'wrist_test.jpg'
            cv2.imwrite(str(frame_file), img)

            frame = {
                'id': 'test_frame_1',
                'path': str(frame_file),
                'joints': {**HOME, 'gripper': OPEN_RAW},
                'look_pan_offset': 0,
            }
            bev = synthesize_orthographic_bev(self.geo, [frame], tmp_path)
            self.assertEqual(bev['width_px'], 720)
            self.assertEqual(bev['height_px'], 720)
            self.assertTrue(Path(bev['bev_path']).is_file())
            self.assertTrue(Path(bev['bev_grid_path']).is_file())
            self.assertGreater(bev['coverage_ratio'], 0.15)

            scene_map = WorkspaceSceneMap(tmp_path)
            er2_item_1 = {
                'label': '綠色品客罐',
                'point': [500, 500],
                'box_2d': [380, 440, 620, 560],
                'bev_center_point': [520, 500],
                'yaw_deg': 15.0,
                'length_mm': 62.0,
                'width_mm': 56.0,
                'height_mm': 110.0,
                'shape_primitive': 'cylinder',
                'confidence': 0.95,
                'physics': {
                    'estimated_mass_g': 135,
                    'material': 'cardboard_metal',
                    'friction_coeff': 0.58,
                    'rigidity': 'rigid',
                },
            }
            er2_item_2 = {
                'label': '藍色方塊',
                'point': [520, 680],
                'box_2d': [450, 630, 580, 730],
                'bev_center_point': [540, 680],
                'yaw_deg': -20.0,
                'length_mm': 48.0,
                'width_mm': 44.0,
                'height_mm': 65.0,
                'shape_primitive': 'box',
                'confidence': 0.92,
                'physics': {
                    'estimated_mass_g': 85,
                    'material': 'plastic',
                    'friction_coeff': 0.52,
                    'rigidity': 'rigid',
                },
            }
            obj1 = solve_oriented_object(self.geo, frame, er2_item_1, crop_dir=scene_map.crops_dir)
            obj2 = solve_oriented_object(self.geo, frame, er2_item_2, crop_dir=scene_map.crops_dir)
            self.assertIsNotNone(obj1)
            self.assertIsNotNone(obj2)

            rec1 = scene_map.upsert_object(obj1)
            rec2 = scene_map.upsert_object(obj2)
            self.assertEqual(len(scene_map.objects), 2)

            rel = scene_map.compute_spatial_relations(self.geo)
            self.assertGreaterEqual(len(rel['relations']), 2)
            self.assertGreaterEqual(len(rel['free_space_slots']), 2)

            # Plan a Pick-and-Place from rec1 to the first free-space slot
            slot = rel['free_space_slots'][0]
            plan = plan_mira_grasp_or_place(
                self.geo,
                scene_map,
                rec1,
                HOME,
                OPEN_RAW,
                place_target_m=slot['position_m'],
                place_label_zh=slot['slot_label'],
            )
            self.assertEqual(plan['mode'], 'pick_and_place')
            self.assertGreater(len(plan['pose_stream']), 10)
            self.assertIn('physics_verification', plan['simulation'])

            # Export 3D Twin GLB
            twin = export_real_to_sim_twin(
                self.geo,
                scene_map,
                {**HOME, 'gripper': OPEN_RAW},
                tmp_path / 'current',
                active_plan=plan,
            )
            self.assertTrue((tmp_path / 'current' / 'scene.glb').is_file())
            self.assertEqual(twin['scene_map']['object_count'], 2)


if __name__ == '__main__':
    unittest.main()
