"""Offline scene conversion checks; never instantiate a hardware controller."""
import importlib.util
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from server.local_cartesian import ALL, NAMES, RAD_PER_TICK, default_model, xyz_transform
from server.sim_scene import build_scene, export_scene, verify_mujoco
from scripts.export_sim_scene import export_bundle, main as export_main, validated_depth


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = all(importlib.util.find_spec(name) for name in ('mujoco', 'trimesh'))


def frame():
    return {'id': 'offline_test', 'source': 'wrist', 'camera_generation': 'test-camera',
            'capture_monotonic': 100., 'captured_at': 1000., 'path': '/test/archived.jpg',
            'joints': dict(zip(ALL, [2156, 931, 3040, 2251, 820, 1857]))}


class SceneStateTests(unittest.TestCase):
    def test_unknown_depth_stays_absent_and_raw_jaw_is_not_a_cad_angle(self):
        a = build_scene(frame())
        newer_jaw = frame(); newer_jaw['joints']['gripper'] = 1600
        b = build_scene(newer_jaw)
        self.assertEqual(a['robot']['links'], b['robot']['links'])
        self.assertEqual(a['targets'], [])
        self.assertEqual(a['robot']['jaw']['rendering'], 'mesh_with_visual_only_travel_prior')
        self.assertFalse(a['robot']['jaw']['angle_calibrated'])
        self.assertFalse(a['robot']['jaw']['is_actual_jaw_pose'])
        self.assertNotEqual(a['robot']['jaw']['display_pose'], b['robot']['jaw']['display_pose'])
        self.assertEqual(a['robot']['raw_to_angle_scope'], 'five_body_joints_only')
        self.assertTrue(a['snapshot_only'])
        self.assertFalse(a['production_eligible'])
        self.assertFalse(a['camera']['intrinsics_calibrated'])
        self.assertFalse(a['table']['extent_is_measured'])
        self.assertFalse(a['robot']['metric_accuracy_verified'])

    def test_jaw_display_maps_travel_endpoints_without_claiming_mechanical_calibration(self):
        model = default_model(ROOT)
        jaw_limit = model.tree.find("joint[@name='gripper']/limit")
        for range_key, angle_key in [('range_min', 'lower'), ('range_max', 'upper')]:
            observation = frame()
            observation['joints']['gripper'] = model.calibration['gripper'][range_key]
            jaw = build_scene(observation)['robot']['jaw']
            display = jaw['display_pose']
            self.assertAlmostEqual(display['angle_rad'], float(jaw_limit.get(angle_key)))
            self.assertFalse(display['mechanical_zero_calibrated'])
            self.assertFalse(display['aperture_calibrated'])
            self.assertFalse(display['homing_offset_applied_again'])
            self.assertTrue(display['visual_only'])
            self.assertEqual(jaw['raw'], observation['joints']['gripper'])
            # Displaying a jaw cannot add it to the trusted FK or remove the
            # controller's original angle-independent full-sweep screening.
            self.assertNotIn('moving_jaw_so101_v1_link', model.fk.transforms(observation['joints']))
            self.assertIn(model.jaw_label, model.floor_minimum(observation['joints']))
            self.assertEqual(jaw['collision_guard'], 'moving_jaw_full360_envelope_unchanged')

    def test_incomplete_or_observer_robot_pose_cannot_become_measured_robot_state(self):
        for mutation in ({'source': 'engineering_observer'}, {'capture_monotonic': float('nan')},
                         {'camera_generation': ''}, {'joints': {}},
                         {'joints': {**frame()['joints'], 'wrist_roll': True}}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_scene({**frame(), **mutation})

    def test_target_requires_registered_coordinates_and_uncertainty(self):
        target = {'id': 'surface', 'label': 'white cylinder visible surface',
                  'source': 'wrist_multiview_estimate', 'evidence_id': 'pair-test',
                  'coordinate_frame': 'robot_base_model', 'reference': 'visible_surface',
                  'position_m': [.3, .0, .12], 'uncertainty_radius_m': .02}
        state = build_scene(frame(), targets=[target])
        saved = state['targets'][0]
        self.assertEqual(saved['position_m'], target['position_m'])
        self.assertEqual(saved['uncertainty_radius_m'], .02)
        self.assertFalse(saved['collision_volume_known'])
        for mutation in ({'uncertainty_radius_m': None}, {'uncertainty_radius_m': -1},
                         {'coordinate_frame': 'camera_optical'}, {'source': 'assumed'},
                         {'evidence_id': ''}, {'position_m': [1, float('inf'), 0]}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_scene(frame(), targets=[{**target, **mutation}])

    def test_observer_target_is_engineering_only_and_requires_registration_evidence(self):
        target = {'id': 'observer-point', 'label': 'engineering depth',
                  'source': 'engineering_observer', 'evidence_id': 'depth-frame-1',
                  'position_m': [.3, .1, .1], 'coordinate_frame': 'robot_base_model',
                  'uncertainty_radius_m': .025}
        with self.assertRaises(ValueError):
            build_scene(frame(), targets=[target])
        target['registration_evidence_id'] = 'extrinsics-offline-test'
        result = build_scene(frame(), targets=[target])['targets'][0]
        self.assertFalse(result['production_eligible'])
        self.assertEqual(result['registration_evidence_id'], 'extrinsics-offline-test')

    def test_unlocated_target_is_text_only_and_proposed_camera_pose_is_not_execution(self):
        target = {'id': 'unknown', 'label': 'Object with unknown distance',
                  'source': 'wrist_multiview_estimate', 'evidence_id': 'image-1'}
        candidate = {'id': 'pitch-change', 'segment': {'joint_targets': {
            **frame()['joints'], 'wrist_flex': frame()['joints']['wrist_flex'] - 68}}}
        state = build_scene(frame(), targets=[target], candidates=[candidate])
        self.assertIsNone(state['targets'][0]['position_m'])
        self.assertIsNone(state['targets'][0]['uncertainty_radius_m'])
        self.assertFalse(state['candidates'][0]['executed'])
        self.assertFalse(state['candidates'][0]['motion_authorized'])
        self.assertNotEqual(state['camera']['optical_pose'], state['candidates'][0]['camera_pose'])


@unittest.skipUnless(OPTIONAL, 'Optional engineering MuJoCo/trimesh packages are not installed')
class SceneExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.output = Path(cls.folder.name) / 'export'
        cls.state = export_scene(cls.output, frame())

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_mujoco_matches_all_existing_fk_links_at_several_encoder_poses(self):
        poses = [frame()['joints'], dict(zip(ALL, [2161, 1135, 2490, 2328, 1100, 1531])),
                 dict(zip(ALL, [1800, 1800, 2200, 1800, 2047, 2000]))]
        for raw in poses:
            with self.subTest(raw=raw):
                result = verify_mujoco(self.output / 'scene.xml', raw)
                self.assertTrue(result['passed'])
                self.assertLess(result['maximum_position_error_m'], 1e-9)
                self.assertLess(result['camera_forward_vector_error'], 1e-9)
                self.assertEqual(result['actuator_count'], 0)
                self.assertFalse(result['real_world_accuracy_verified'])

    def test_keyframe_loads_recorded_pose_and_jaw_has_no_invented_joint(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(self.output / 'scene.xml'))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        for name in NAMES:
            self.assertAlmostEqual(data.qpos[model.joint(name).qposadr[0]],
                                   (frame()['joints'][name] - 2047) * RAD_PER_TICK)
        self.assertEqual(model.nq, 5)
        tree = ET.parse(self.output / 'scene.xml')
        self.assertIsNone(tree.find('.//joint[@name="gripper"]'))
        envelope = tree.find('.//geom[@name="uncalibrated_jaw_sweep_envelope"]')
        self.assertEqual(envelope.get('contype'), '0')
        # An illustrative fixed snapshot is visible, but has no joint or
        # collision geom that could masquerade as calibrated jaw mechanics.
        jaw = tree.find('.//body[@name="moving_jaw_so101_v1_link"]')
        self.assertIsNotNone(jaw)
        self.assertIsNone(jaw.find('joint'))
        self.assertTrue(all(g.get('contype') == '0' for g in jaw.findall('geom')))
        self.assertEqual(len(jaw.findall('geom')), 1)
        mujoco.mj_forward(model, data)
        displayed = self.state['robot']['jaw']['display_pose']['link_pose']
        np.testing.assert_allclose(data.body('moving_jaw_so101_v1_link').xpos,
                                   displayed['position_m'], atol=1e-10)

    def test_portable_glb_contains_robot_table_and_unknown_jaw_envelope_without_objects(self):
        import trimesh
        scene = trimesh.load_scene(self.output / 'scene.glb')
        self.assertIn('jaw_unknown_full_sweep', scene.graph.nodes)
        self.assertIn('table_model_prior', scene.graph.nodes)
        self.assertTrue(any(name.startswith('wrist_camera_link') for name in scene.graph.nodes))
        self.assertFalse(any(name.startswith('target_') for name in scene.graph.nodes))
        tree = ET.parse(self.output / 'scene.xml')
        for mesh in tree.findall('./asset/mesh'):
            self.assertFalse(Path(mesh.get('file')).is_absolute())
            self.assertTrue((self.output / mesh.get('file')).is_file())
        self.assertTrue((self.output / 'LICENSE.SO-ARM100').is_file())
        saved = json.loads((self.output / 'scene.json').read_text())
        self.assertEqual(saved['frame']['joints'], frame()['joints'])
        self.assertEqual(saved['up_axis'], 'Z')

    def test_both_real_jaw_meshes_are_solid_white_and_motor_remains_dark(self):
        import trimesh
        scene = trimesh.load_scene(self.output / 'scene.glb')
        model = default_model(ROOT)
        for node, asset in [('gripper_link_1', 'wrist_roll_follower_so101_v1.stl'),
                            ('moving_jaw_so101_v1_link_0', 'moving_jaw_so101_v1.stl')]:
            with self.subTest(node=node):
                self.assertIn(node, scene.graph.nodes)
                _, geometry = scene.graph[node]
                mesh = scene.geometry[geometry]
                source = trimesh.load_mesh(model.mesh_root / 'assets' / asset, process=False)
                self.assertEqual(len(mesh.faces), len(source.faces))
                color = mesh.visual.material.baseColorFactor
                self.assertTrue(np.all(color[:3] >= 230))
                self.assertEqual(int(max(color[:3])) - int(min(color[:3])), 0)
                self.assertEqual(color[3], 255)
                self.assertEqual(mesh.visual.material.alphaMode, 'OPAQUE')
        _, motor = scene.graph['gripper_link_0']
        self.assertTrue(np.all(scene.geometry[motor].visual.material.baseColorFactor[:3] < 30))
        _, sweep = scene.graph['jaw_unknown_full_sweep']
        self.assertLess(scene.geometry[sweep].visual.material.baseColorFactor[3], 50)
        self.assertEqual(scene.geometry[sweep].metadata['engineering_role'], 'jaw_uncertainty_envelope')

    def test_visual_jaw_follows_recorded_travel_while_fixed_finger_and_fk_do_not(self):
        import trimesh
        from scipy.spatial.transform import Rotation
        closed = frame(); closed['joints']['gripper'] = 1531
        output = Path(self.folder.name) / 'other-jaw-pose'
        state = export_scene(output, closed)
        original = trimesh.load_scene(self.output / 'scene.glb')
        other = trimesh.load_scene(output / 'scene.glb')
        model = default_model(ROOT)
        name = 'moving_jaw_so101_v1_link_0'
        original_T = original.graph[name][0]
        actual_T = other.graph[name][0]
        self.assertGreater(np.linalg.norm(original_T - actual_T), .1)
        # Independent composition checks the branch origin and mesh origin,
        # which previously made the entire movable half disappear.
        angle = state['robot']['jaw']['display_pose']['angle_rad']
        rotate = np.eye(4); rotate[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        visual = model.tree.find("link[@name='moving_jaw_so101_v1_link']/visual/origin")
        expected = (model.fk.transforms(closed['joints'])['gripper_link'] @ model.jaw_origin
                    @ rotate @ xyz_transform(visual))
        np.testing.assert_allclose(actual_T, expected, atol=1e-9)
        np.testing.assert_allclose(original.graph['gripper_link_1'][0],
                                   other.graph['gripper_link_1'][0], atol=1e-12)
        self.assertEqual(state['robot']['links'], self.state['robot']['links'])
        self.assertEqual(model.floor_minimum(frame()['joints']),
                         model.floor_minimum(closed['joints']))

    def test_export_never_overwrites_existing_scene(self):
        previous = (self.output / 'scene.json').read_bytes()
        with self.assertRaises(FileExistsError):
            export_scene(self.output, frame())
        self.assertEqual(previous, (self.output / 'scene.json').read_bytes())


class SceneCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.observation = frame()
        self.image = self.folder / 'saved-wrist.jpg'
        ok, jpeg = cv2.imencode('.jpg', np.full((24, 32, 3), 127, np.uint8))
        self.assertTrue(ok)
        self.image.write_bytes(jpeg.tobytes())
        self.observation.update(path=str(self.image), image_size=[32, 24])
        self.frame_path = self.folder / 'frame.json'
        self.frame_path.write_text(json.dumps(self.observation))
        self.depth_path = self.folder / 'depth.json'
        self.output = self.folder / 'result'
        self.depth = {'source': 'wrist-only', 'status': 'estimated', 'metric_calibrated': False,
                      'scope': 'object_roi_surface', 'current_frame_id': self.observation['id'],
                      'provenance': {'source': 'wrist-only', 'camera_generation': 'test-camera'},
                      'distance': {'camera_optical_axis_m': .3, 'gripper_forward_axis_m': .2},
                      'uncertainty': {'camera_depth_interval_m': [.24, .36],
                                      'depth_error_bound_m': .06,
                                      'bound_kind': 'heuristic axial model sensitivity'},
                      'points': [{'position_base_m': [.3, .01, .15]}]}

    def invoke(self, depth):
        self.depth_path.write_text(json.dumps(depth))
        with redirect_stdout(io.StringIO()):
            export_main(['--frame', str(self.frame_path), '--depth', str(self.depth_path),
                         '--output', str(self.output)])

    def test_cli_rejects_wrong_exposure_camera_or_observer_before_any_export(self):
        examples = [{**self.depth, 'current_frame_id': 'older-exposure'},
                    {**self.depth, 'source': 'engineering_observer'},
                    {**self.depth, 'provenance': {'camera_generation': 'another-camera'}},
                    {**self.depth, 'provenance': {}},
                    {**self.depth, 'provenance': {'source': 'engineering_observer',
                                                'camera_generation': 'test-camera'}}]
        for data in examples:
            with self.subTest(data=data), patch('scripts.export_sim_scene.export_scene') as export:
                with self.assertRaises(ValueError):
                    self.invoke(data)
                export.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_depth_retains_axial_uncertainty_without_creating_spherical_target(self):
        accepted = validated_depth(self.depth, self.observation)
        self.assertEqual(accepted['uncertainty'], self.depth['uncertainty'])
        self.assertEqual(accepted['points'], self.depth['points'])
        self.assertFalse(accepted['production_eligible'])
        self.assertFalse(accepted['motion_authorized'])
        self.assertNotIn('uncertainty_radius_m', accepted)
        self.assertNotIn('targets', accepted)
        self.assertNotIn('engineering_only', self.depth, 'input must remain unchanged')

    def test_depth_rejects_nonfinite_points_and_invalid_or_missing_uncertainty(self):
        examples = []
        for value in (float('nan'), float('inf'), True, 4):
            bad = deepcopy(self.depth); bad['points'][0]['position_base_m'][0] = value
            examples.append(bad)
        for uncertainty in (None, {'camera_depth_interval_m': [.35, .36], 'depth_error_bound_m': .06},
                            {'camera_depth_interval_m': [.24, .36], 'depth_error_bound_m': -1}):
            examples.append({**self.depth, 'uncertainty': uncertainty})
        examples.append({**self.depth, 'status': 'insufficient'})
        for data in examples:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validated_depth(data, self.observation)

    def test_missing_corrupt_or_dimension_mismatched_wrist_image_blocks_bundle(self):
        examples = [{**self.observation, 'path': str(self.folder / 'missing.jpg')},
                    {**self.observation, 'image_size': [640, 480]}]
        invalid = self.folder / 'invalid.jpg'; invalid.write_bytes(b'\xff\xd8\xffnot-a-real-image')
        examples.append({**self.observation, 'path': str(invalid)})
        for observation in examples:
            with self.subTest(observation=observation), patch('scripts.export_sim_scene.export_scene') as export:
                with self.assertRaises(ValueError):
                    export_bundle(self.output, observation, depth=self.depth)
                export.assert_not_called()
                self.assertFalse(self.output.exists())

    @unittest.skipUnless(OPTIONAL, 'Optional engineering MuJoCo/trimesh packages are not installed')
    def test_cli_exports_matching_jpeg_and_depth_as_complete_bundle(self):
        self.invoke(self.depth)
        accepted = json.loads((self.output / 'depth.json').read_text())
        scene = json.loads((self.output / 'scene.json').read_text())
        self.assertEqual((self.output / 'wrist.jpg').read_bytes(), self.image.read_bytes())
        self.assertEqual(accepted['current_frame_id'], scene['frame']['id'])
        self.assertEqual(accepted['provenance']['camera_generation'], scene['frame']['camera_generation'])
        self.assertEqual(accepted['uncertainty'], self.depth['uncertainty'])
        self.assertEqual(scene['targets'], [], 'axial bounds must not become invented solid objects')
        self.assertEqual(scene['artifacts']['depth'], 'depth.json')
        self.assertEqual(scene['artifacts']['wrist_image'], 'wrist.jpg')
        self.assertFalse(any(path.name.startswith('.bundle-') for path in self.folder.iterdir()))


if __name__ == '__main__':
    unittest.main()
