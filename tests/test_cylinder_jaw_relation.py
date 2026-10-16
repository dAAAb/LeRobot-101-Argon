"""Pure relative geometry and error propagation; never motor permission."""
from copy import deepcopy
import unittest

import cv2
import numpy as np

import test_cylinder_silhouette as cylinder_fixture
from server.cylinder_jaw_relation import analyze_cylinder_jaw_relation


class CylinderJawRelationTests(unittest.TestCase):
    def setUp(self):
        fixture = cylinder_fixture.CylinderSilhouetteTests()
        fixture.setUp()
        self.records = fixture.records
        for row in self.records:
            row['measured_joints'] = {'gripper': 1859}
        self.uncertainty = fixture.uncertainty
        self.plane = fixture.plane
        self.axis = [0., 0., 1.]
        last = np.asarray(self.records[-1]['camera_transform_base_m'])
        # Synthetic observed pads straddle the cylinder, independently of
        # the solver. These dimensions are a fixture, never real-object input.
        pads = [last[:3,:3].T @ (fixture.center+[0., sign*.035, .065]-last[:3,3])
                for sign in (-1, 1)]
        self.jaw = dict(source='wrist-only', camera_calibration_fingerprint='synthetic-camera',
            reference_frame_id=self.records[-1]['frame_id'], camera_generation='one-session', jaw_raw=1859,
            jaw_calibration_fingerprint='synthetic-jaw', fixed_pad_camera_m=pads[0].tolist(),
            moving_pad_camera_m=pads[1].tolist(), tcp_position_uncertainty_m=.001,
            aperture_uncertainty_m=.002)

    def analyze(self, **kw):
        return analyze_cylinder_jaw_relation(kw.pop('records', self.records),
            reference_frame_id=kw.pop('reference_frame_id', self.records[-1]['frame_id']),
            jaw_camera=kw.pop('jaw_camera', self.jaw),
            axis_base=kw.pop('axis_base', self.axis),
            uncertainty=kw.pop('uncertainty', self.uncertainty),
            support_plane=kw.pop('support_plane', self.plane), **kw)

    def test_nominal_signed_gaps_and_original_jaw_error_are_retained(self):
        result = self.analyze()
        self.assertAlmostEqual(result['left_gap_m'], .017, places=8)
        self.assertAlmostEqual(result['right_gap_m'], .017, places=8)
        self.assertAlmostEqual(result['retained_jaw_allowance_per_side_m'], .002)
        self.assertAlmostEqual(result['left_conditional_gap_lower_m'], .015, places=8)
        for key in ('planning_eligible', 'grasp_ready', 'execution_authorized'):
            self.assertIs(result[key], False)
        self.assertTrue(result['absolute_path_budget_unchanged'])
        changed = self.analyze(jaw_camera={**self.jaw,
            'tcp_position_uncertainty_m': .004, 'aperture_uncertainty_m': .006})
        self.assertAlmostEqual(changed['left_conditional_gap_lower_m'], .01, places=8)
        self.assertEqual(result['absolute_estimate'], changed['absolute_estimate'])

    def test_rigid_base_relabeling_preserves_relation_and_full_covariance(self):
        scales = {**self.uncertainty, 'relative_translation_m': .002,
                  'relative_rotation_deg': .2, 'pixel_edge_mislocalization_px': 2.}
        original = self.analyze(uncertainty=scales)
        rotation = cv2.Rodrigues(np.array([.2, -.3, .1]))[0]
        translation = np.array([.12, -.08, .05])
        records = deepcopy(self.records)
        for row in records:
            transform = np.asarray(row['camera_transform_base_m'])
            transform[:3,:3] = rotation @ transform[:3,:3]
            transform[:3,3] = rotation @ transform[:3,3]+translation
            row['camera_transform_base_m'] = transform.tolist()
        plane = {**self.plane, 'point_base_m': translation.tolist(),
                 'normal_base': (rotation @ self.axis).tolist()}
        changed = self.analyze(records=records, axis_base=plane['normal_base'],
                               support_plane=plane, uncertainty=scales)
        for key in ('left_gap_m', 'right_gap_m', 'left_object_gap_sensitivity_m',
                    'right_object_gap_sensitivity_m'):
            self.assertAlmostEqual(original[key], changed[key], places=6)
        np.testing.assert_allclose(original['relative_center_nearest_jaw_camera_m'],
            changed['relative_center_nearest_jaw_camera_m'], atol=1e-8)
        for mode in original['uncertainty_modes']:
            np.testing.assert_allclose(original['uncertainty_modes'][mode]['covariance'],
                changed['uncertainty_modes'][mode]['covariance'], rtol=2e-4, atol=1e-10)

    def test_shared_coordinate_error_cancels_only_in_current_relative_output(self):
        original = self.analyze()
        changed = self.analyze(uncertainty={**self.uncertainty,
            'common_translation_m': .04, 'common_rotation_deg': 3.})
        self.assertGreaterEqual(changed['absolute_estimate']['center_position_error_m'], .04)
        self.assertGreater(changed['absolute_estimate']['center_position_error_m'],
                           original['absolute_estimate']['center_position_error_m'])
        for key in ('left_object_gap_sensitivity_m', 'right_object_gap_sensitivity_m',
                    'left_conditional_gap_lower_m', 'right_conditional_gap_lower_m'):
            self.assertEqual(original[key], changed[key])
        self.assertIn('not evidence', changed['cancellation_scope'])

    def test_independent_pose_error_and_local_mount_error_do_not_cancel(self):
        original = self.analyze()
        changed = self.analyze(uncertainty={**self.uncertainty,
            'relative_translation_m': .003, 'relative_rotation_deg': .4})
        self.assertGreater(changed['left_object_gap_sensitivity_m'],
                           original['left_object_gap_sensitivity_m']+.0001)
        for mode in changed['uncertainty_modes'].values():
            self.assertGreater(mode['left_gap_sensitivity_m'], .0001)
            np.testing.assert_allclose(mode['covariance'],
                sum((np.asarray(c) for c in mode['contributions'].values()), np.zeros((6,6))))
        mount = changed['uncertainty_modes']['shared_camera_local_mount']
        self.assertIn('shared_camera_local_translation', mount['contributions'])
        self.assertNotIn('view_0_translation', mount['contributions'])
        self.assertAlmostEqual(changed['left_object_gap_sensitivity_m'],
            max(changed['leave_one_view_out_relation_spread'][4],
                *(m['left_gap_sensitivity_m'] for m in changed['uncertainty_modes'].values())))

    def test_independent_plane_and_pixel_errors_still_increase_relative_budget(self):
        original = self.analyze()
        changed = self.analyze(uncertainty={**self.uncertainty,
            'pixel_edge_mislocalization_px': 2., 'support_plane_translation_m': .002,
            'support_plane_rotation_deg': .2})
        self.assertGreater(changed['left_object_gap_sensitivity_m'],
                           original['left_object_gap_sensitivity_m']+.0001)
        for mode in changed['uncertainty_modes'].values():
            self.assertIn('relative_support_plane_translation', mode['contributions'])
            self.assertIn('relative_support_plane_rotation', mode['contributions'])
            self.assertIn('view_0_side_0_shift', mode['contributions'])

    def test_gap_covariance_contains_joint_center_radius_response(self):
        result = self.analyze(uncertainty={**self.uncertainty,
            'relative_translation_m': .002, 'pixel_edge_mislocalization_px': 2.})
        # For this fixed transverse axis, the exact derivative of left gap
        # is u.dcenter - dradius. The cross term is essential: marginal
        # center norm + radius sensitivity is a different, looser quantity.
        direction = np.asarray(self.jaw['moving_pad_camera_m'])-self.jaw['fixed_pad_camera_m']
        direction /= np.linalg.norm(direction)
        for mode in result['uncertainty_modes'].values():
            covariance = np.asarray(mode['covariance'])
            self.assertGreater(np.linalg.norm(covariance[:3,3]), 1e-10)
            derivative = np.r_[direction, -1.]
            self.assertAlmostEqual(covariance[4,4], derivative @ covariance[:4,:4] @ derivative,
                                   places=10)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance)[0], -1e-12)

    def test_reference_and_calibration_cannot_be_reused_for_another_pose(self):
        for args in ({'reference_frame_id': self.records[0]['frame_id']},
                     {'jaw_camera': {**self.jaw, 'camera_calibration_fingerprint': 'different'}},
                     {'jaw_camera': {**self.jaw, 'reference_frame_id': self.records[0]['frame_id']}},
                     {'jaw_camera': {**self.jaw, 'camera_generation': 'another'}},
                     {'jaw_camera': {**self.jaw, 'jaw_raw': 1900}},
                     {'jaw_camera': {**self.jaw, 'jaw_raw': True}},
                     {'jaw_camera': {**self.jaw, 'source': 'D435'}}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.analyze(**args)

    def test_original_contour_inconsistency_gate_is_not_bypassed(self):
        records = deepcopy(self.records)
        records[1]['body_sides_px'] = (np.asarray(records[1]['body_sides_px'])+[80., 0.]).tolist()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.analyze(records=records, uncertainty={**self.uncertainty,
                'pixel_edge_mislocalization_px': 2.})


if __name__ == '__main__':
    unittest.main()
