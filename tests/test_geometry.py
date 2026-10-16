import unittest
from pathlib import Path
from server.geometry import Kinematics, SpatialEstimator
from server.selfcal import synthetic_demo, synthetic_focal_demo

ROOT = Path(__file__).resolve().parents[1]

class GeometryTests(unittest.TestCase):
    def test_metric_scale_from_known_robot_motion(self):
        result = synthetic_demo()
        self.assertLess(result['handeye_translation_error_mm'], .001)
        self.assertLess(result['baseline_max_error_mm'], .001)

    def test_single_axis_motion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'degeneracy'):
            synthetic_demo(single_axis=True)

    def test_unknown_focal_and_metric_reconstruction(self):
        result = synthetic_focal_demo()
        self.assertLess(result['focal_error_pct'], 1)
        self.assertLess(result['holdout_p95_sampson_px'], 1)
        self.assertLess(result['heldout_metric_reconstruction']['median_metric_error_mm'], 2)

    def test_partial_joint_data_cannot_invent_a_pose(self):
        fk = Kinematics(ROOT / 'assets/so101_follower_cam.urdf')
        with self.assertRaises(ValueError):
            fk.camera({'wrist_roll': 2047})

    def test_unmeasured_jaw_zero_does_not_invent_finger_geometry(self):
        import numpy as np
        fk = Kinematics(ROOT / 'assets/so101_follower_cam.urdf')
        raw = dict.fromkeys(('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'),2047)
        opened = fk.transforms(dict(raw,gripper=1860))
        closed = fk.transforms(dict(raw,gripper=1520))
        self.assertNotIn('moving_jaw_so101_v1_link',opened)
        self.assertIn('gripper_frame_link',opened)  # fixed finger, not jaw center
        np.testing.assert_array_equal(opened['wrist_camera_optical'],closed['wrist_camera_optical'])

    def test_observer_never_enters_reconstruction(self):
        estimator = SpatialEstimator(ROOT)
        with self.assertRaisesRegex(ValueError, 'wrist'):
            estimator.correspondences([{'source': 'observer', 'joints': {'x': 1}, 'path': 'not-read'}])

    def test_single_elbow_arc_is_rejected_before_feature_fitting(self):
        from unittest.mock import patch
        estimator = SpatialEstimator(ROOT)
        # Synthetic increments from the measured v8 starting pose reproduce
        # its single-elbow family; there are deliberately no readable images.
        raw=dict(shoulder_pan=2162,shoulder_lift=1141,elbow_flex=2839,
                 wrist_flex=2429,wrist_roll=984,gripper=1857)
        observations=[{'source':'wrist','joints':{**raw,'elbow_flex':2839-i*10},
                       'path':'must-not-be-read'} for i in range(12)]
        with patch.object(estimator,'correspondences') as tracks:
            result=estimator.estimate(observations,[],[])
        tracks.assert_not_called()
        self.assertEqual(result['status'],'insufficient')
        self.assertFalse(result['grasp_ready'])
        self.assertTrue(result['needs_more_views'])
        self.assertEqual(result['next_observation_requirement'],'nonparallel_rotation_axes')
        self.assertEqual(result['motion_observability']['translation_constraint_rank'],2)
        self.assertNotIn('camera_matrix',result)

    def test_multiaxis_motion_still_requires_visual_evidence(self):
        from unittest.mock import patch
        estimator = SpatialEstimator(ROOT)
        raw=dict(shoulder_pan=2162,shoulder_lift=1141,elbow_flex=2839,
                 wrist_flex=2429,wrist_roll=984,gripper=1857)
        poses=[raw,{**raw,'shoulder_pan':2182},{**raw,'wrist_flex':2449},
               {**raw,'shoulder_pan':2142},{**raw,'wrist_roll':1004}]
        observations=[{'source':'wrist','joints':p,'path':'not-read'} for p in poses]
        with patch.object(estimator,'correspondences',return_value=([],(1920,1080))) as tracks:
            result=estimator.estimate(observations,[],[])
        tracks.assert_called_once()
        self.assertTrue(result['motion_observability']['translation_identifiable_from_rotations'])
        self.assertFalse(result['motion_observability']['calibration_validated'])
        self.assertEqual(result['status'],'insufficient')
        self.assertFalse(result['grasp_ready'])

    def test_observer_cannot_enter_motion_observability_precheck(self):
        from unittest.mock import patch
        estimator = SpatialEstimator(ROOT)
        with patch.object(estimator.fk,'transforms') as transforms:
            result=estimator.estimate([{'source':'observer','joints':{}}]*5,[],[])
        transforms.assert_not_called()
        self.assertIn('wrist',result['diagnostic_reason'])

    def test_missing_history_endpoint_does_not_remove_expected_view(self):
        estimator = SpatialEstimator(ROOT)
        observations = [{'id':str(i),'source':'wrist','sequence':i,'joints':{'shoulder_pan':2047},'path':'not-read'} for i in range(3)]
        history = [{'source':'wrist','sequence':i,'jpeg':b'not decoded','capture_monotonic':i*.125} for i in range(2)]
        matches, _, diagnostics = estimator.correspondences(observations, histories=[{'keyframes':['0','1','2'],'frames':history}])
        self.assertEqual(matches, [])
        self.assertEqual(diagnostics[0]['status'], 'history_gap')

    def test_observer_cannot_be_hidden_in_unused_video_segment(self):
        estimator = SpatialEstimator(ROOT)
        with self.assertRaisesRegex(ValueError, 'wrist'):
            estimator.correspondences([{'id':'other','source':'observer'}], histories=[])

    def test_failed_observation_archive_keeps_self_contained_wrist_keyframes(self):
        import tempfile,zipfile,json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'.runtime').mkdir();picture=root/'source.jpg';picture.write_bytes(b'original wrist bytes')
            estimator=SpatialEstimator(ROOT);estimator.root=root
            observation={'id':'a','source':'wrist','path':str(picture)}
            estimator.record_inputs([observation],[],[observation],[],archive_name='wrist-local-gaze-input.zip')
            with zipfile.ZipFile(root/'.runtime/wrist-local-gaze-input.zip') as archive:
                manifest=json.loads(archive.read('manifest.json'))
                self.assertEqual(archive.read(manifest['keyframe_images']['a']),picture.read_bytes())
                self.assertEqual(manifest['histories'],[])
            with self.assertRaisesRegex(ValueError,'wrist'):
                estimator.record_inputs([dict(observation,source='observer',path='must-not-read')],[],[],[])

if __name__ == '__main__':
    unittest.main()
