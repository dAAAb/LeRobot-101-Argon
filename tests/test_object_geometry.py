"""Object geometry decisions retain units, revisions and evidence scope."""
from copy import deepcopy
import json
import unittest
import numpy as np

from server.object_geometry import build_object_geometry, surface_planning_evidence, fit_upright_cylinder


class ObjectGeometryTests(unittest.TestCase):
    def setUp(self):
        self.frame = {'id': 'frame-3', 'source': 'wrist', 'camera_generation': 'session-a',
                      'capture_monotonic': 3.}
        self.depth = {
            'source': 'wrist-only', 'scope': 'object_roi_surface', 'status': 'estimated',
            'current_frame_id': 'frame-3', 'frame_ids': ['frame-1', 'frame-2', 'frame-3'],
            'metric_calibrated': False, 'view_count': 3,
            'provenance': {'camera_generation': 'session-a', 'metric_calibrated': False,
                           'camera_model': 'cad-camera-prior'},
            'distance': {'position_base_m': [.2, -.03, .08], 'camera_optical_axis_m': .16},
            'uncertainty': {'base_position_error_bound_m': .035,
                            'camera_depth_interval_m': [.13, .19],
                            'bound_kind': 'heuristic sensitivity, not physical guarantee'},
            'points': [{'position_base_m': [.2 + i * .001, -.03, .08 + (i % 2) * .01],
                        'views': 3, 'current_pixel': [200 + i, 100]} for i in range(8)],
        }
        self.observation = {'source': 'wrist', 'frame_id': 'frame-3',
            'camera_generation': 'session-a', 'metric_calibrated': True,
            'geometry_revision': 'geometry-1', 'calibration_revision': 'camera-1',
            'calibration_fingerprint': 'same-profile-content-hash',
            'jaw_geometry': {'metric_calibrated': True}}

    def build(self, depth=None, **kwargs):
        return build_object_geometry(self.depth if depth is None else depth,
                                     current_frame=self.frame, track_id='target-1', **kwargs)

    def validate_depth(self):
        self.depth['metric_calibrated'] = True
        self.depth['provenance'].update(metric_calibrated=True,
            camera_model='validated-calibration', geometry_revision='geometry-1',
            calibration_revision='camera-1', calibration_fingerprint='same-profile-content-hash')

    def test_coarse_measurement_is_known_distance_not_unknown_or_grasp_center(self):
        result = self.build()
        self.assertEqual(result['status'], 'observed')
        self.assertEqual(result['recommended_action'], 'calibrate_camera')
        self.assertIn('16.0', result['message'])
        self.assertFalse(result['further_target_views_useful'])
        self.assertEqual(result['observed_surface']['position_base_m'], [.2, -.03, .08])
        self.assertIsNone(result['object_center_base_m'])
        self.assertIsNone(result['full_dimensions_m'])
        self.assertIsNone(result['grasp_center_base_m'])
        self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['grasp_ready'])
        json.dumps(result, allow_nan=False)

    def test_point_cloud_has_sample_extent_not_full_object_size_or_reduced_uncertainty(self):
        result = self.build()['observed_surface']
        self.assertEqual(result['position_error_bound_m'], .035)
        self.assertIn('not full object', result['observed_sample_bounds_base_m']['scope'])
        self.assertIn(result['representative_measured_point_base_m'],
                      [p['position_base_m'] for p in self.depth['points']])

    def test_current_frame_binding_rejects_motion_or_camera_restart(self):
        for changed in ({'id': 'new-frame'}, {'camera_generation': 'session-b'}):
            self.frame.update(changed)
            result = self.build()
            self.assertEqual(result['status'], 'stale')
            self.assertIsNone(result['observed_surface'])

    def test_explicit_stationary_refresh_rebinds_only_with_both_comparisons(self):
        self.frame['id'] = 'refreshed'
        self.depth['stationary_refresh'] = {'current_frame_id': 'refreshed',
            'image_comparison': {'median_luma_delta': 0.}, 'max_joint_drift_ticks': 1}
        self.assertEqual(self.build()['status'], 'observed')
        self.depth['stationary_refresh']['image_comparison'] = None
        self.assertEqual(self.build()['status'], 'stale')

    def test_old_success_is_diagnostic_not_a_live_target_after_lost_tracking(self):
        previous = deepcopy(self.depth)
        self.depth.update(status='insufficient', points=[], distance=None)
        result = self.build(previous_estimates=[previous])
        self.assertEqual(result['recommended_action'], 'calibrate_camera')
        self.assertIsNone(result['observed_surface'])
        self.assertFalse(result['last_successful_measurement']['usable_as_current_target'])
        self.assertEqual(result['last_successful_measurement']['camera_optical_axis_m'], .16)

    def test_history_from_another_camera_session_is_not_a_target_measurement(self):
        previous = deepcopy(self.depth)
        previous['provenance']['camera_generation'] = 'other-session'
        self.depth.update(status='insufficient', points=[], distance=None)
        result = self.build(previous_estimates=[previous])
        self.assertIsNone(result['last_successful_measurement'])
        self.assertEqual(result['recommended_action'], 'measure_target')

    def test_model_rejection_after_real_translation_asks_for_calibration(self):
        self.depth.update(status='insufficient', points=[], distance=None,
            maximum_camera_translation_mm=35., tracking={'candidate_tracks': 15,
                'rejected': {'invalid_depth': 9, 'nonrigid_or_pose_inconsistent': 6}})
        self.assertEqual(self.build()['recommended_action'], 'calibrate_camera')

    def test_rotation_without_translation_requests_useful_baseline(self):
        self.depth.update(status='insufficient', points=[], distance=None,
                          maximum_camera_translation_mm=.02)
        self.assertEqual(self.build()['recommended_action'], 'change_baseline')

    def test_rotation_compensated_low_parallax_requests_baseline(self):
        self.depth.update(status='insufficient', points=[], distance=None,
            maximum_camera_translation_mm=3., tracking={'candidate_tracks': 12,
                'rejected': {'insufficient_translation_parallax': 12}})
        self.assertEqual(self.build()['recommended_action'], 'change_baseline')

    def test_calibrated_packet_cannot_upgrade_depth_computed_using_old_prior(self):
        result = self.build(observation=self.observation)
        self.assertFalse(result['metric_calibrated'])
        self.assertEqual(result['recommended_action'], 'calibrate_camera')

    def test_calibrated_depth_requires_matching_profile_revision(self):
        self.validate_depth()
        self.observation['calibration_revision'] = 'new-camera-2'
        result = self.build(observation=self.observation)
        self.assertFalse(result['metric_calibrated'])
        self.assertEqual(result['recommended_action'], 'calibrate_camera')

    def test_same_revision_label_with_different_profile_content_is_not_compatible(self):
        self.validate_depth()
        self.observation['calibration_fingerprint'] = 'different-profile-content'
        self.assertFalse(self.build(observation=self.observation)['metric_calibrated'])

    def test_calibrated_depth_without_metric_jaw_center_requests_jaw_calibration(self):
        self.validate_depth()
        self.observation['jaw_geometry'] = {'metric_calibrated': False}
        self.assertEqual(self.build(observation=self.observation)['recommended_action'], 'calibrate_jaws')

    def test_model_assisted_jaw_eligibility_does_not_require_measured_metric_claim(self):
        self.validate_depth()
        self.observation['jaw_geometry'] = {'metric_calibrated': False, 'planning_eligible': True,
                                           'accuracy_basis': 'model-assisted'}
        result = self.build(observation=self.observation)
        self.assertEqual(result['recommended_action'], 'plan_approach')
        self.assertFalse(result['grasp_ready'])
        self.observation['jaw_geometry']['planning_eligible'] = False
        self.assertEqual(self.build(observation=self.observation)['recommended_action'], 'calibrate_jaws')

    def test_matching_validated_geometry_proposes_surface_approach_not_grasp(self):
        self.validate_depth()
        result = self.build(observation=self.observation)
        self.assertEqual(result['recommended_action'], 'plan_approach')
        self.assertTrue(result['metric_calibrated'])
        self.assertFalse(result['grasp_ready'])
        proposal = surface_planning_evidence(result)
        self.assertEqual(proposal['surface_base_m'], [.2, -.03, .08])
        self.assertEqual(proposal['target_uncertainty_m'], .035)
        self.assertEqual(proposal['evidence']['scope'], 'observed_surface_not_grasp_center')

    def test_rejects_engineering_observer_inputs(self):
        self.depth['source'] = 'D435'
        with self.assertRaises(ValueError): self.build()
        self.depth['source'] = 'wrist-only'
        self.frame['source'] = 'D435'
        with self.assertRaises(ValueError): self.build()

    def test_unassigned_scene_depth_is_not_an_object(self):
        self.depth['scope'] = 'unassigned_scene_surface'
        result = self.build()
        self.assertIsNone(result['observed_surface'])
        self.assertEqual(result['recommended_action'], 'measure_target')
        self.assertIn('target_roi', result['missing_fields'])

    def test_invalid_geometry_never_escapes_as_nan_or_planning_target(self):
        self.depth['distance']['position_base_m'][0] = float('nan')
        result = self.build()
        self.assertIsNone(result['observed_surface'])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError): surface_planning_evidence(result)

    def test_result_owns_its_samples(self):
        result = self.build()
        result['observed_surface']['points'][0]['position_base_m'][0] = 42
        self.assertEqual(self.depth['points'][0]['position_base_m'][0], .2)

    def cylinder(self, *, arc=170, noise=.0002, radius=.021):
        rng = np.random.default_rng(31)
        xyz = np.array([[.23 + radius * np.cos(a), -.04 + radius * np.sin(a), h]
                        for h in np.linspace(.04, .12, 5)
                        for a in np.linspace(-np.radians(arc / 2), np.radians(arc / 2), 12)])
        xyz += rng.normal(0, noise, xyz.shape)
        result = self.build()
        result['observed_surface']['points'] = [{'position_base_m': p.tolist(), 'views': 3} for p in xyz]
        return result

    def fit(self, geometry):
        return fit_upright_cylinder(geometry, shape_intent='upright_cylinder', axis_base=[0., 0., 1.])

    def test_cylinder_estimates_unknown_radius_from_visible_curvature(self):
        result = self.fit(self.cylinder())
        self.assertEqual(result['status'], 'hypothesis', result)
        self.assertAlmostEqual(result['radius_m'], .021, delta=.001)
        np.testing.assert_allclose(result['centerline_point_base_m'][:2], [.23, -.04], atol=.001)
        self.assertGreater(result['diagnostics']['arc_coverage_deg'], 150)
        self.assertIsNone(result['full_height_m'])
        self.assertFalse(result['grasp_ready'])
        self.assertGreaterEqual(result['center_position_error_m'], .035)

    def test_cylinder_requires_explicit_shape_intent_and_axis(self):
        for intent, axis in [('unknown', [0., 0., 1.]), ('upright_cylinder', [0., 0., 2.])]:
            with self.assertRaises(ValueError):
                fit_upright_cylinder(self.cylinder(), shape_intent=intent, axis_base=axis)

    def test_narrow_arc_does_not_invent_unseen_cylinder_center(self):
        result = self.fit(self.cylinder(arc=35, noise=.0001))
        self.assertEqual(result['status'], 'insufficient', result)
        self.assertNotIn('centerline_point_base_m', result)

    def test_flat_surface_is_not_a_cylinder(self):
        geometry = self.cylinder()
        for i, p in enumerate(geometry['observed_surface']['points']):
            p['position_base_m'] = [.2, -.025 + (i % 12) * .004, .04 + (i // 12) * .02]
        self.assertEqual(self.fit(geometry)['status'], 'insufficient')

    def test_noisy_surface_and_taper_are_not_accepted_as_constant_cylinder(self):
        self.assertEqual(self.fit(self.cylinder(noise=.005))['status'], 'insufficient')
        geometry = self.cylinder()
        for p in geometry['observed_surface']['points']:
            xyz = p['position_base_m']
            if xyz[2] > .085:
                xyz[0] = .23 + (xyz[0] - .23) * 1.65
                xyz[1] = -.04 + (xyz[1] + .04) * 1.65
        self.assertEqual(self.fit(geometry)['status'], 'insufficient')

    def test_prior_camera_remains_prior_after_a_good_cylinder_fit(self):
        result = self.fit(self.cylinder())
        self.assertEqual(result['status'], 'hypothesis')
        self.assertFalse(result['metric_calibrated'])
        self.assertFalse(result['execution_authorized'])

    def test_axis_projection_works_for_non_default_base_orientation(self):
        geometry = self.cylinder()
        for p in geometry['observed_surface']['points']:
            x, y, z = p['position_base_m']
            p['position_base_m'] = [z, y, -x]
        result = fit_upright_cylinder(geometry, shape_intent='upright_cylinder', axis_base=[1., 0., 0.])
        self.assertEqual(result['status'], 'hypothesis', result)
        self.assertAlmostEqual(result['radius_m'], .021, delta=.001)
        np.testing.assert_allclose(result['centerline_point_base_m'][1:], [-.04, -.23], atol=.001)

    def partition(self, geometry, *, relative=.012, translation=.020, rotation=.003):
        surface = geometry['observed_surface']
        surface['position_error_bound_m'] = relative + translation + rotation
        surface['position_error_components'] = {
            'model': 'relative-triangulation-plus-reference-placement-v1',
            'relative_surface_m': relative, 'reference_translation_m': translation,
            'reference_rotation_m': rotation}
        return geometry

    def test_common_translation_moves_center_not_radius_or_shape_budget(self):
        geometry = self.partition(self.cylinder(noise=0))
        before = self.fit(geometry)
        translated = deepcopy(geometry)
        delta = np.array([.04, -.023, .019])
        for point in translated['observed_surface']['points']:
            point['position_base_m'] = (np.array(point['position_base_m']) + delta).tolist()
        after = self.fit(translated)
        self.assertEqual(after['status'], 'hypothesis')
        np.testing.assert_allclose(after['centerline_point_base_m'],
            np.array(before['centerline_point_base_m']) + delta, atol=1e-9)
        self.assertAlmostEqual(after['radius_m'], before['radius_m'], places=9)
        self.assertAlmostEqual(after['radius_sensitivity_m'], before['radius_sensitivity_m'], places=9)
        self.assertAlmostEqual(after['center_position_error_m'] - after['radius_sensitivity_m'], .020)
        # Raising only reference-placement uncertainty preserves the full center
        # error while leaving shape error unchanged.
        wider = self.fit(self.partition(translated, translation=.040))
        self.assertAlmostEqual(wider['radius_sensitivity_m'], after['radius_sensitivity_m'])
        self.assertAlmostEqual(wider['center_position_error_m'] - after['center_position_error_m'], .020)

    def test_common_rotation_of_points_and_axis_preserves_radius(self):
        import cv2
        geometry = self.partition(self.cylinder(noise=0))
        before = self.fit(geometry)
        rotation = cv2.Rodrigues(np.array([.12, -.25, .31]))[0]
        for point in geometry['observed_surface']['points']:
            point['position_base_m'] = (rotation @ point['position_base_m']).tolist()
        after = fit_upright_cylinder(geometry, shape_intent='upright_cylinder',
                                    axis_base=(rotation @ [0., 0., 1.]).tolist())
        self.assertEqual(after['status'], 'hypothesis', after)
        self.assertAlmostEqual(after['radius_m'], before['radius_m'], places=9)
        self.assertAlmostEqual(after['radius_sensitivity_m'], before['radius_sensitivity_m'], places=9)
        # Unknown reference rotation relative to an unchanged upright axis is
        # deliberately retained, not cancelled as though the axis were known.
        self.assertGreaterEqual(after['radius_sensitivity_m'], .012 + .003)

    def test_relative_surface_distortion_still_changes_radius_and_its_budget(self):
        geometry = self.partition(self.cylinder(noise=0))
        before = self.fit(geometry)
        for point in geometry['observed_surface']['points']:
            p = point['position_base_m']
            p[0] = .23 + 1.1 * (p[0] - .23)
            p[1] = -.04 + 1.1 * (p[1] + .04)
        after = self.fit(self.partition(geometry, relative=.016))
        self.assertEqual(after['status'], 'hypothesis')
        self.assertAlmostEqual(after['radius_m'], before['radius_m'] * 1.1, places=9)
        self.assertGreaterEqual(after['radius_sensitivity_m'] - before['radius_sensitivity_m'], .004 - 1e-9)
        self.assertGreaterEqual(after['center_position_error_m'] - before['center_position_error_m'], .004 - 1e-9)

    def test_legacy_or_malformed_partition_never_reduces_radius_uncertainty(self):
        geometry = self.cylinder()
        legacy = self.fit(geometry)
        for partition in ({}, {'model': 'untrusted', 'relative_surface_m': 0},
                          {'model': 'relative-triangulation-plus-reference-placement-v1',
                           'relative_surface_m': .001, 'reference_translation_m': .001,
                           'reference_rotation_m': .001}):
            geometry['observed_surface']['position_error_components'] = partition
            result = self.fit(geometry)
            self.assertEqual(result['radius_sensitivity_m'], legacy['radius_sensitivity_m'])
            self.assertEqual(result['center_position_error_m'], legacy['center_position_error_m'])
            self.assertEqual(result['uncertainty_components']['partition'], 'legacy-unsplit')

    def test_surface_adapter_preserves_only_complete_additive_partition(self):
        parts = {'model': 'relative-triangulation-plus-reference-placement-v1',
                 'relative_surface_m': .012, 'reference_translation_m': .020,
                 'reference_rotation_m': .003}
        self.depth['uncertainty']['position_error_components'] = parts
        self.assertEqual(self.build()['observed_surface']['position_error_components'], parts)
        self.depth['uncertainty']['position_error_components']['relative_surface_m'] = float('nan')
        self.assertNotIn('position_error_components', self.build()['observed_surface'])


if __name__ == '__main__':
    unittest.main()
