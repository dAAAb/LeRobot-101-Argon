"""Model-assisted jaw provenance and correlated, camera-relative uncertainty."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from server.jaw_acquisition import collect_jaw_images
from server.jaw_model_assisted import (ArticulatedJawModel, propagate_camera_relative_uncertainty,
    MatchedSectionJawModel, build_model_assisted_profile, residual_covariance, validate_silhouette_evidence)
from server.spatial_contract import JawCalibration, WristCalibration, build_wrist_observation
from server.task_space_plan import _tcp
import test_spatial_contract as spatial_fixtures
import test_jaw_acquisition as capture_fixtures


class RelativeUncertaintyTests(unittest.TestCase):
    def setUp(self):
        camera=np.eye(4);camera[2,3]=.3
        self.model=ArticulatedJawModel(camera,np.eye(4),[0,0,1],[-.03,0,0],[.04,0,0],1)
        self.parameters=np.array([0.,0.,0.,0.,0.,0.,30.])
        self.assumptions=dict(sigma_multiplier=1.,cad_pad_tolerance_m=0.,encoder_tolerance_ticks=0.,
            assumption_description='Synthetic exact CAD; one-sigma local sensitivity only, not measured hardware accuracy.')
    def test_common_camera_translation_changes_tcp_but_cancels_gap_error(self):
        covariance=np.zeros((7,7));covariance[3,3]=100.  # 10 mm shared translation SD.
        result=propagate_camera_relative_uncertainty(self.model,self.parameters,covariance,1700,**self.assumptions)
        error=result['conditional_error_m']
        self.assertAlmostEqual(error['fixed_pad'],.010,places=8)
        self.assertAlmostEqual(error['moving_pad'],.010,places=8)
        self.assertAlmostEqual(error['tcp'],.010,places=8)
        self.assertLess(error['separation'],1e-9);self.assertLess(error['aperture'],1e-9)
        self.assertEqual(result['uncertainty_frame'],'wrist_camera_optical')
        self.assertIsNone(result['calibration_profile']);self.assertFalse(result['external_metric_measurement'])
    def test_shared_rotation_changes_points_not_aperture(self):
        covariance=np.zeros((7,7));covariance[2,2]=1.
        result=propagate_camera_relative_uncertainty(self.model,self.parameters,covariance,1700,**self.assumptions)
        self.assertGreater(result['conditional_error_m']['moving_pad'],.0005)
        self.assertLess(result['conditional_error_m']['aperture'],1e-9)
    def test_encoder_uncertainty_and_declared_cad_assumptions_propagate_without_relabelling_raw(self):
        assumptions={**self.assumptions,'encoder_tolerance_ticks':2.,'cad_pad_tolerance_m':.0005,
            'assumption_description':'Engineering assumption for a printed CAD part; tolerance is not a mechanical survey.'}
        result=propagate_camera_relative_uncertainty(self.model,self.parameters,np.zeros((7,7)),1848,**assumptions)
        self.assertGreater(result['encoder_error_m']['moving_pad'],0.)
        self.assertEqual(result['encoder_error_m']['fixed_pad'],0.)
        self.assertAlmostEqual(result['conditional_error_m']['fixed_pad'],.0005)
        self.assertGreaterEqual(result['conditional_error_m']['aperture'],.001)
        expected=self.model.points(self.parameters,1848)
        np.testing.assert_allclose(result['nominal']['moving_pad_m'],expected[1])
    def test_alternative_fit_spread_is_a_floor_not_independent_noise_added_twice(self):
        alternative=self.parameters.copy();alternative[3]+=8.
        covariance=np.zeros((7,7));covariance[3,3]=4.
        result=propagate_camera_relative_uncertainty(self.model,self.parameters,covariance,1700,
            alternative_parameters=[alternative],**self.assumptions)
        self.assertAlmostEqual(result['conditional_error_m']['tcp'],.008)
        self.assertLess(result['conditional_error_m']['aperture'],1e-9)
    def test_independent_optics_error_adds_to_same_image_floor_without_adding_base_fk_error(self):
        edge=self.parameters.copy();edge[3]+=8.
        optics=self.parameters.copy();optics[3]+=3.
        covariance=np.zeros((7,7));covariance[3,3]=4.
        result=propagate_camera_relative_uncertainty(self.model,self.parameters,covariance,1700,
            alternative_parameters=[edge],independent_parameter_families={'focal_scale':[optics]},**self.assumptions)
        self.assertAlmostEqual(result['conditional_error_m']['tcp'],.011)
        self.assertAlmostEqual(result['independent_parameter_error_m']['focal_scale']['tcp'],.003)
        self.assertLess(result['conditional_error_m']['aperture'],1e-9)
    def test_rank_deficiency_never_supplies_covariance(self):
        j=np.ones((15,7));r=np.ones(15)
        result=residual_covariance(j,r)
        self.assertFalse(result['identifiable']);self.assertIsNone(result['covariance'])
    def test_finite_full_rank_covariance_matches_known_linear_system(self):
        j=np.vstack([np.eye(7),np.eye(7),np.eye(7)])
        result=residual_covariance(j,np.ones(21))
        self.assertTrue(result['identifiable'])
        np.testing.assert_allclose(result['covariance'],np.eye(7)*.5,atol=1e-12)
        with self.assertRaises(ValueError):residual_covariance(j,np.full(21,np.nan))
        with self.assertRaises(ValueError):
            propagate_camera_relative_uncertainty(self.model,self.parameters,-np.eye(7),1700,**self.assumptions)


class ModelEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.robot=capture_fixtures.AcquisitionRobot(self.root)
        snap=self.robot.snapshot
        def distinct():
            frame=snap();cv2.imwrite(str(self.robot.image),np.full((48,64),self.robot.sequence+100,np.uint8))
            return frame
        self.profile=dict(schema_version=1,source='wrist-only',calibration_revision='camera',
            camera_id='wrist-test',mount_revision='mount',geometry_revision='geometry',native_image_size=[64,48],
            camera_matrix=[[100.,0,31.5],[0,100.,23.5],[0,0,1]],distortion=[0.,0.,0.,0.,0.],
            handeye=np.eye(4).tolist(),validated_components=['intrinsics','handeye','robot_geometry'],
            validation_id='camera-heldout',validation_kind='independent-heldout')
        result=collect_jaw_images(self.robot,distinct,lambda _:None,self.robot.cancel,distinct(),
            empty_verified=True,camera_id='wrist-test',mount_revision='mount',geometry_revision='geometry',
            camera_profile=self.profile)
        self.path=Path(result['manifest_path']);self.manifest=json.loads(self.path.read_text())
        self.annotations={'frames':{f['id']:dict(target_raw=f['target_raw'],image_sha256=f['image_sha256'],
            groups={part:dict(part=part,polylines=[[[10.,10.],[20.,10.]]]) for part in ('fixed','moving')})
            for f in self.manifest['frames']}}
        self.annotation_path=self.root/'annotations.json';self.annotation_path.write_text(json.dumps(self.annotations))
        self.cad=self.root/'nominal.urdf';self.cad.write_text('<nominal/>')
    def validate(self):
        return validate_silhouette_evidence(self.path,self.annotation_path,self.profile,
            expected_geometry_revision='geometry',cad_files=[self.cad])
    def test_real_acquisition_binding_uses_actual_raw_and_fixed_target_split(self):
        result=self.validate()
        self.assertEqual(len(result['training_frame_ids']),3)
        self.assertEqual(len(result['heldout_frame_ids']),2)
        self.assertEqual(result['measured_raw_range'],[1549,1849])
        self.assertFalse(result['external_metric_measurement'])
        self.assertEqual(result['geometry_method'],'articulated-cad-wrist-silhouette')
    def test_tampered_image_or_annotation_camera_mismatch_rejected(self):
        self.annotations['frames'][self.manifest['frames'][0]['id']]['image_sha256']='0'*64
        self.annotation_path.write_text(json.dumps(self.annotations))
        with self.assertRaisesRegex(ValueError,'Annotation'):self.validate()
        self.annotation_path.write_text(json.dumps({'frames':{f['id']:dict(target_raw=f['target_raw'],
            image_sha256=f['image_sha256'],groups={part:dict(part=part,polylines=[[[10.,10.],[20.,10.]]])
            for part in ('fixed','moving')}) for f in self.manifest['frames']}}))
        Path(self.manifest['frames'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash'):self.validate()
    def test_commanded_label_cannot_replace_measured_encoder(self):
        self.manifest['frames'][0]['actual_gripper_raw']=1550
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'actual encoder'):self.validate()
    def test_new_preregistered_opening_protocol_is_bound_without_hardcoding_labels(self):
        targets=[1550,1675,1800,1925,2050]
        self.manifest['opening_protocol']=dict(name='wider-five-v2',targets_raw=targets,
            training_targets_raw=targets[::2],heldout_targets_raw=targets[1::2])
        for frame,target in zip(self.manifest['frames'],targets):
            frame['target_raw']=target;frame['actual_gripper_raw']=target-1;frame['joints']['gripper']=target-1
            for sample in frame['settled_joint_window']:sample['joints']['gripper']=target-1
            self.annotations['frames'][frame['id']]['target_raw']=target
        self.path.write_text(json.dumps(self.manifest));self.annotation_path.write_text(json.dumps(self.annotations))
        result=self.validate()
        self.assertEqual(result['measured_raw_range'],[1549,2049])
        self.assertEqual(result['opening_protocol'],self.manifest['opening_protocol'])
        self.manifest['opening_protocol']['heldout_targets_raw']=[1550,1925]
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'disjoint'):self.validate()
    def test_observer_source_and_repeated_holdout_image_rejected(self):
        original=deepcopy(self.manifest)
        self.manifest['frames'][1]['source']='D435';self.path.write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):self.validate()
        self.manifest=original
        self.manifest['frames'][1]['image_sha256']=self.manifest['frames'][0]['image_sha256']
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'distinct'):self.validate()


class ConditionalProfileTests(unittest.TestCase):
    """Synthetic contracts only; never persisted as robot calibration."""
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.camera=spatial_fixtures.camera_profile()
        self.camera['source']='wrist-only'
        calibrated=WristCalibration.from_mapping(self.camera,native_image_size=[640,480],camera_id='wrist',
            mount_revision='mount-v1',geometry_revision='geometry-v1',cad_handeye=np.eye(4))
        self.provenance=dict(geometry_method='articulated-cad-wrist-silhouette',
            validation_kind='independent-silhouette-heldout',source='wrist-only',
            camera_calibration_fingerprint=calibrated.fingerprint,
            external_metric_measurement=False,training_frame_ids=['train0','train1','train2'],
            heldout_frame_ids=['held0','held1'],measured_raw_range=[1700,1704])
        self.files=[]
        for name in ('manifest','annotations','fit','sensitivity'):
            path=self.root/name;path.write_text('synthetic-contract-test-only');self.files.append(path)
        artifacts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in self.files]
        self.provenance.update(manifest=artifacts[0],annotations=artifacts[1],cad_files=[artifacts[2]])
        face=np.array([[0,1,2],[0,2,3]])
        def plane(x):return np.array([[x,-.1,-.1],[x,.1,-.1],[x,.1,.1],[x,-.1,.1]])
        camera=np.eye(4);camera[2,3]=.3
        self.model=MatchedSectionJawModel(camera,np.eye(4),[0,0,1],plane(-.02),face,
            plane(.03),face,[0.,0.],1)
        self.parameters=np.array([0.,0.,0.,0.,0.,0.,30.])
        self.assumptions=dict(sigma_multiplier=2.,cad_pad_tolerance_m=.002,encoder_tolerance_ticks=8.,
            description='Synthetic declared CAD/hinge allowance and encoder sensitivity; not external metric measurements.')
        self.covariance=residual_covariance(np.vstack([np.eye(7)]*3),np.ones(21)*.001)
        self.sections=[]
        for raw in range(1700,1705):
            assumptions={k:v for k,v in self.assumptions.items() if k!='description'}
            estimate=propagate_camera_relative_uncertainty(self.model,self.parameters,self.covariance['covariance'],raw,
                assumption_description=self.assumptions['description'],**assumptions)
            self.sections.append(dict(raw=raw,estimate=estimate,approach_camera=self.model.approach_camera(self.parameters)))
        self.kwargs=dict(provenance=self.provenance,camera_profile=self.camera,calibration_revision='synthetic-estimate',
            validation_id='synthetic-heldout',geometry_revision='geometry-v1',sections=self.sections,
            contact_section=dict(cad_yz_m=[0.,0.],description='Synthetic matched plane; local cross-section only, no full corridor claim.'),
            assumptions=self.assumptions,validation=dict(parameter_rank=7,accepted=True,
                heldout_silhouette_p95_limit_px=6.,acceptance_basis='Synthetic pixel criterion to exercise validation; not a production policy.',
                heldout=[dict(frame_id=x,silhouette_p95_px=2.) for x in ['held0','held1']]),
            covariance_evidence=self.covariance,evidence_files=self.files)
    def profile(self):return build_model_assisted_profile(**self.kwargs)
    def test_mesh_section_is_recomputed_not_material_point_distance(self):
        points=self.model.points(self.parameters,1700)
        self.assertAlmostEqual(points[1,0],.03/np.cos(np.radians(30.)))
        self.assertAlmostEqual(points[1,1],0.)
        self.assertAlmostEqual(points[1,2],.3)
        with self.assertRaisesRegex(ValueError,'intersect'):
            MatchedSectionJawModel(self.model.camera_from_cad_prior,np.eye(4),[0,0,1],
                self.model.fixed_vertices,self.model.fixed_faces,self.model.moving_vertices,
                self.model.moving_faces,[0.,.5],1).points(self.parameters,1700)
    def test_estimate_is_planning_eligible_but_not_metric_certified_and_keeps_assumptions(self):
        profile=self.profile();jaw=JawCalibration.from_mapping(profile,geometry_revision='geometry-v1')
        geometry=jaw.at(1702);public=geometry.to_dict()
        self.assertFalse(geometry.metric_calibrated);self.assertTrue(geometry.planning_eligible)
        self.assertFalse(public['external_metric_measurement'])
        self.assertEqual(public['accuracy_basis'],'conditional-cad-estimate')
        self.assertGreater(public['aperture_uncertainty_m'],.004)
        self.assertEqual(public['assumptions'],self.assumptions)
        context=_tcp(geometry,{'gripper':1702})
        self.assertTrue(context['planning_eligible']);self.assertFalse(context['metric_calibrated'])
        self.assertEqual(context['assumptions'],self.assumptions)
        self.assertNotIn('measured two-pad',context['geometry_scope'])
        with self.assertRaises(ValueError):jaw.at(1705)
    def test_reject_unobservable_failed_heldout_reused_group_and_sparse_interpolation(self):
        original=deepcopy(self.kwargs)
        for mutate in (lambda k:k['covariance_evidence'].update(identifiable=False),
                       lambda k:k['validation']['heldout'][0].update(silhouette_p95_px=7.),
                       lambda k:k['provenance'].update(heldout_frame_ids=['train0','held1']),
                       lambda k:k.update(sections=k['sections'][::2])):
            self.kwargs=deepcopy(original);mutate(self.kwargs)
            with self.assertRaises(ValueError):self.profile()
    def test_camera_mismatch_rejected_and_metadata_changes_identity(self):
        profile=self.profile();jaw=JawCalibration.from_mapping(profile,geometry_revision='geometry-v1')
        changed=deepcopy(profile);changed['model_assisted']['assumptions']['description']+=' Changed assumption.'
        self.assertNotEqual(jaw.fingerprint,JawCalibration.from_mapping(changed,geometry_revision='geometry-v1').fingerprint)
        record=spatial_fixtures.frame();record['joints']['gripper']=1702
        transforms={'gripper_link':np.eye(4),'wrist_camera_optical':np.array(self.camera['handeye'])}
        packet=build_wrist_observation(record,transforms,geometry_revision='geometry-v1',
            calibration=self.camera,mount_revision='mount-v1',jaw_calibration=jaw)
        self.assertTrue(packet.jaw_geometry.planning_eligible)
        camera=deepcopy(self.camera);camera['camera_matrix'][0][0]+=1.
        with self.assertRaisesRegex(ValueError,'different wrist camera'):
            build_wrist_observation(record,transforms,geometry_revision='geometry-v1',
                calibration=camera,mount_revision='mount-v1',jaw_calibration=jaw)
        changed=deepcopy(profile);changed['status']='validated';changed['validation_kind']='independent-heldout'
        with self.assertRaisesRegex(ValueError,'relabeled'):JawCalibration.from_mapping(changed,geometry_revision='geometry-v1')
    def test_evidence_mutation_between_validation_and_profile_is_rejected(self):
        self.files[0].write_text('changed acquisition')
        with self.assertRaisesRegex(ValueError,'Evidence changed'):self.profile()


if __name__=='__main__':unittest.main()
