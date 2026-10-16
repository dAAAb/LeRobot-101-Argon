"""Contour/frame binding and one-inference runtime routing, without devices."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
import test_cylinder_silhouette as optics
import test_spatial_object_program as programs
from server.cylinder_observation import hypothesis_from_wrist_contours
from server.spatial_object_program import plan_spatial_object_program
from server.spatial_object_program import _plan_cylinder
from server.object_geometry import fit_upright_cylinder


class CylinderObservationTests(unittest.TestCase):
    def setUp(self):
        fixture = optics.CylinderSilhouetteTests(); fixture.setUp(); self.fixture = fixture
        self.directory = TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.frames, self.packets, self.contours = [], {}, []
        cal = SimpleNamespace(fingerprint='synthetic-camera', calibration_revision='camera-v1', geometry_revision='arm-v1')
        for i, record in enumerate(fixture.records):
            path = Path(self.directory.name)/f'{i}.png'; cv2.imwrite(str(path), np.zeros((1080,1920),np.uint8))
            f = dict(id=record['frame_id'],source='wrist',path=str(path),sequence=i+1,capture_monotonic=float(i+1),camera_generation='one-session')
            self.frames.append(f)
            self.packets[f['id']] = SimpleNamespace(metric_calibrated=True, calibration=cal,
                frame_id=f['id'], camera_generation='one-session', sequence=i+1,capture_monotonic=float(i+1),
                compatibility_key=('one-session','synthetic-camera'), processing=SimpleNamespace(output_size=(1920,1080)),
                T_base_camera=SimpleNamespace(as_array=lambda r=record:np.array(r['camera_transform_base_m'])),
                camera_matrix=record['camera_matrix'],distortion=record['distortion'],measured_joints=(('synthetic',i),))
            self.contours.append(dict(frame_id=f['id'],body_sides_normalized=(np.asarray(record['body_sides_px'])/[1920,1080]).tolist(),
                support_boundary_normalized=(np.asarray(record['support_boundary_px'])/[1920,1080]).tolist()))
        self.profile = dict(validation_scope={'board_plane_base':np.eye(4).tolist()},
            relative_translation_uncertainty_m=.001, relative_rotation_uncertainty_deg=.1,
            focal_relative_uncertainty=.001,base_position_uncertainty_m=.001)
        cal.uncertainty_json = json.dumps({k: v for k, v in self.profile.items() if k != 'validation_scope'})
        for packet in self.packets.values():
            packet.scope_reserve_json = None

    def fit(self, contours=None, support='calibrated_main_table'):
        with patch('server.cylinder_observation._profile_snapshot',return_value=('arm-v1',self.profile,None,{})), \
             patch('server.cylinder_observation._packet',side_effect=lambda m,f,s:self.packets[f['id']]):
            return hypothesis_from_wrist_contours(None,self.frames,self.contours if contours is None else contours,{},
                current_frame_id=self.frames[-1]['id'],track_id='target-1',support_surface=support)

    def test_normalized_real_pixels_are_bound_to_current_frame_and_camera(self):
        result=self.fit()
        self.assertAlmostEqual(result['radius_m'],self.fixture.radius,places=8)
        self.assertTrue(result['support_plane_used']);self.assertTrue(result['coordinate_calibrated'])
        self.assertFalse(result['metric_calibrated']);self.assertFalse(result['metric_accuracy_independently_validated'])
        self.assertTrue(result['planning_eligible']);self.assertFalse(result['execution_authorized'])
        self.assertEqual(result['frame_id'],self.frames[-1]['id'])
        self.assertEqual(len(result['silhouette_evidence'][0]['image_sha256']),64)
        np.testing.assert_allclose(result['silhouette_evidence'][0]['body_sides_px'],self.fixture.records[0]['body_sides_px'])

    def test_unknown_or_stacked_support_does_not_use_board_plane(self):
        result=self.fit(support='other_or_unknown')
        self.assertFalse(result['support_plane_used']);self.assertIsNone(result['support_plane_provenance'])
        self.assertEqual(result['diagnostics']['support_boundary_points_base_m'],[])

    def test_boundary_reserve_reaches_silhouette_uncertainty_without_mutating_profile(self):
        from server.cylinder_observation import fit_cylinder_silhouettes
        before = deepcopy(self.profile)
        self.packets[self.frames[-1]['id']].scope_reserve_json = json.dumps(
            dict(translation_m=.002, rotation_deg=.15, status='near-boundary-extrapolation'))
        with patch('server.cylinder_observation.fit_cylinder_silhouettes', wraps=fit_cylinder_silhouettes) as fit:
            result = self.fit()
        bounds = fit.call_args.kwargs['uncertainty']
        self.assertAlmostEqual(bounds['relative_translation_m'], .005)
        self.assertAlmostEqual(bounds['common_translation_m'], .003)
        self.assertAlmostEqual(bounds['relative_rotation_deg'], .4)
        self.assertEqual(result['runtime_calibration_scopes'][-1]['status'], 'near-boundary-extrapolation')
        self.assertEqual(before, self.profile)

    def test_insufficient_history_is_observation_need_not_setup_failure(self):
        for n in range(3):
            result=self.fit(self.contours[:n])
            self.assertEqual(result['status'],'insufficient');self.assertNotIn('required_setup',result)

    def test_unknown_frame_missing_current_scope_and_mixed_calibration_are_rejected(self):
        with self.assertRaises(ValueError):self.fit(self.contours[:-1])
        contours=deepcopy(self.contours);contours[0]['frame_id']='not-provided'
        with self.assertRaises(ValueError):self.fit(contours)
        self.packets[self.frames[0]['id']].metric_calibrated=False
        with self.assertRaises(ValueError):self.fit()
        self.packets[self.frames[0]['id']].metric_calibrated=True
        self.packets[self.frames[0]['id']].compatibility_key=('different','profile')
        with self.assertRaises(ValueError):self.fit()

    def test_pixel_units_and_misdeclared_native_image_size_are_rejected(self):
        contours=deepcopy(self.contours);contours[0]['body_sides_normalized']=self.fixture.records[0]['body_sides_px']
        with self.assertRaises(ValueError):self.fit(contours)
        self.packets[self.frames[0]['id']].processing.output_size=(960,540)
        with self.assertRaises(ValueError):self.fit()


class CylinderIntentRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): programs.SpatialObjectProgramTests.setUpClass()

    def setUp(self):
        self.fixture=programs.SpatialObjectProgramTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;self.frames=[f.frame(i) for i in (1,2,3)]
        self.packet=f.packet(self.frames[-1]);self.packet.to_dict=lambda:{'validation_scope_reason':''}
        self.intent=f.intent.model_dump()
        self.intent['observation']=dict(candidate_id='move_left',gripper_empty=True,human_clear=True,clear_path=True,confidence=.8,reason='側移觀察')
        self.candidates=[dict(id='move_left',descriptor={})]
        self.intent['cylinder_contours']=[dict(frame_id=frame['id'],body_sides_normalized=[[[.2,.2],[.2,.6]],[[.4,.2],[.4,.6]]],support_boundary_normalized=[]) for frame in self.frames]

    def run_plan(self):
        f=self.fixture;f.brain._infer=Mock(return_value=self.intent)
        with patch('server.spatial_object_program.observation_packet',return_value=self.packet), \
             patch('server.spatial_object_program.build_object_geometry',return_value={'status':'unmeasured','track_id':'object-1','metric_calibrated':False}):
            return plan_spatial_object_program(f.grasp,'抓圓筒',self.frames[-1],f.cancel,history=self.frames[:-1],observation_candidates=self.candidates)

    def test_unmeasured_sparse_points_do_not_block_valid_silhouette_route_or_add_inference(self):
        hypothesis={'status':'hypothesis','conditional':True}
        sentinel=object()
        with patch('server.cylinder_observation.hypothesis_from_wrist_contours',return_value=hypothesis) as fit, \
             patch('server.spatial_object_program._plan_cylinder',return_value=sentinel) as planner:
            self.assertIs(self.run_plan(),sentinel)
        self.assertEqual(self.fixture.brain._infer.call_count,1)
        self.assertIs(planner.call_args.kwargs['fitted'],hypothesis)
        self.assertEqual(fit.call_args.kwargs['current_frame_id'],self.frames[-1]['id'])
        prompt,images,_=self.fixture.brain._infer.call_args.args
        self.assertEqual(images,self.frames);self.assertIn('support_surface',prompt)

    def test_zero_one_two_contours_keep_valid_observation_path(self):
        all_contours=deepcopy(self.intent['cylinder_contours'])
        for n in range(3):
            self.intent['cylinder_contours']=all_contours[:n]
            with patch('server.cylinder_observation.hypothesis_from_wrist_contours') as fit:
                result=self.run_plan()
            self.assertEqual(result.status,'observe');fit.assert_not_called()
            self.assertEqual(self.fixture.grasp.spatial_attempt['required_setups'],[])

    def test_bad_contour_binding_falls_back_to_observation_not_a_grasp(self):
        with patch('server.cylinder_observation.hypothesis_from_wrist_contours',side_effect=ValueError('wrong frame')):
            result=self.run_plan()
        self.assertEqual(result.status,'observe')
        self.assertEqual(self.fixture.grasp.spatial_attempt['silhouette_geometry']['status'],'insufficient')

    def test_conditional_hypothesis_reaches_real_local_planner_and_retains_provenance(self):
        f=self.fixture;frame=f.frame();packet=f.packet(frame)
        fitted=fit_upright_cylinder(f.geometry(),shape_intent='upright_cylinder',axis_base=[0.,0.,1.])
        fitted.update(metric_calibrated=False,coordinate_calibrated=True,planning_eligible=True,
            calibration_fingerprint=packet.calibration.fingerprint,
            accuracy_basis='conditional-upright-cylinder-and-camera-model',
            geometry_method='synthetic-silhouette-contract',frame_ids=['older-a','older-b',frame['id']],
            support_plane_used=False,assumptions={'shape':'synthetic cylinder for integration'},
            conditional_center_radius_covariance=np.zeros((4,4)).tolist())
        geometry={'status':'unmeasured','metric_calibrated':False,'track_id':'object-1'}
        result=_plan_cylinder(f.grasp,'grasp',f.intent,geometry,packet,frame,f.cancel,30.,(),fitted=fitted)
        self.assertEqual(result.status,'ready',result.unsupported_reason)
        evidence=f.grasp.spatial_attempt['proposal']['binding']['evidence']
        self.assertEqual(evidence['target_accuracy_basis'],fitted['accuracy_basis'])
        self.assertFalse(evidence['target_metric_accuracy_independently_validated'])
        self.assertFalse(f.grasp.spatial_attempt['contact_geometry']['metric_calibrated'])
        self.assertEqual(len(f.brain.calls),1)  # Existing whole-path assessment only.
        fitted['calibration_fingerprint']='stale'
        with self.assertRaisesRegex(ValueError,'校準'):
            _plan_cylinder(f.grasp,'grasp',f.intent,geometry,packet,frame,f.cancel,30.,(),fitted=fitted)


if __name__=='__main__':unittest.main()
