"""Wrist-motion depth: synthetic optics and real feature extraction, no devices."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from server import active_depth as depth
from server.local_cartesian import NAMES
from server.selfcal import intrinsics, inv


def transform(x=0., y=0., z=0., rotation=(0., 0., 0.)):
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(np.asarray(rotation, float))[0]
    t[:3, 3] = [x, y, z]
    return t


class ActiveDepthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.size = (640, 480)
        self.k = intrinsics(480, self.size)
        self.cameras = [transform(), transform(.02, 0, .005, (0., .025, 0.)),
                        transform(-.015, -.003, .01, (0., -.02, .025))]
        self.calibration = {'camera_matrix': self.k.tolist(), 'image_size': list(self.size),
                            'intrinsics_validated': True, 'handeye_validated': True,
                            'robot_geometry_validated': True, 'validation_id': 'synthetic-optics',
                            'camera_generation': 'synthetic-wrist', 'distortion': [0.] * 5}

    def fixture(self, cameras=None, *, bottom_only=False):
        cameras = cameras or self.cameras
        rng = np.random.default_rng(22)
        texture = rng.integers(0, 256, self.size[::-1], dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), .5)
        if bottom_only:
            texture[:int(self.size[1] * .65)] = 127
        frames = []
        for i, camera in enumerate(cameras):
            extrinsic = inv(camera)
            # A textured z=.4m plane, projected with the exact rotating camera.
            h = self.k @ (extrinsic[:3, :3] + np.outer(extrinsic[:3, 3], [0., 0., 1.]) / .4) @ np.linalg.inv(self.k)
            image = cv2.warpPerspective(texture, h, self.size)
            path = Path(self.directory.name) / f'view-{i}.png'
            cv2.imwrite(str(path), image)
            frames.append({'id': f'view-{i}', 'path': str(path), 'source': 'wrist',
                'sequence': i + 1, 'capture_monotonic': float(i + 1), 'camera_generation': 'synthetic-wrist',
                'joints': {**{n: 2000 for n in NAMES}, 'shoulder_pan': 2000 + i, 'gripper': 1860}})
        def poses(raw):
            camera = cameras[raw['shoulder_pan'] - 2000]
            return {'wrist_camera_optical': camera, 'gripper_link': camera,
                    'gripper_frame_link': camera @ transform(z=.05)}
        return frames, SimpleNamespace(fk=SimpleNamespace(transforms=poses))

    def exact_tracks(self, cameras=None, *, offsets=None):
        cameras = cameras or self.cameras
        points = np.array([[x, y, .4] for x in np.linspace(-.07, .07, 6)
                           for y in np.linspace(-.11, -.03, 5)])
        tracks = []
        for point in points:
            track = {}
            for i, camera in enumerate(cameras):
                pixel = depth._project(point, camera, self.k, np.zeros(5))
                track[i] = pixel + (np.asarray(offsets[i]) if offsets else 0)
            tracks.append(track)
        return tracks

    def estimate_exact(self, cameras=None, *, offsets=None, calibration=True):
        frames, model = self.fixture(cameras)
        tracks = self.exact_tracks(cameras, offsets=offsets)
        with patch.object(depth, '_image_tracks', return_value=(tracks, {'candidate_tracks': len(tracks)})):
            return depth.estimate_active_depth(frames, model,
                calibration=self.calibration if calibration else None)

    def test_real_image_features_recover_plane_depth_with_camera_rotation(self):
        frames, model = self.fixture()
        result = depth.estimate_active_depth(frames, model, roi={'x': .2, 'y': .1, 'w': .6, 'h': .4},
                                              calibration=self.calibration)
        self.assertEqual(result['status'], 'estimated', result)
        self.assertTrue(result['metric_calibrated'])
        self.assertGreaterEqual(result['geometry']['three_view_points'], 6)
        self.assertAlmostEqual(result['distance']['position_base_m'][2], .4, delta=.004)
        self.assertAlmostEqual(result['distance']['camera_optical_axis_m'] - result['distance']['gripper_forward_axis_m'], .05, delta=1e-6)
        self.assertEqual(result['scope'], 'object_roi_surface')
        for point in result['points']:
            self.assertGreaterEqual(point['current_pixel'][0], .2 * self.size[0])
            self.assertLessEqual(point['current_pixel'][0], .8 * self.size[0])
            self.assertLess(point['current_pixel'][1], depth.GRIPPER_IMAGE_BOUNDARY * self.size[1])
        self.assertFalse(result['grasp_ready'])
        json.dumps(result, allow_nan=False)

    def test_known_translation_and_rotation_are_compensated_not_counted_as_depth(self):
        result = self.estimate_exact()
        self.assertEqual(result['status'], 'estimated', result)
        self.assertAlmostEqual(result['distance']['position_base_m'][2], .4, delta=1e-8)
        self.assertGreater(result['geometry']['median_parallax_deg'], 2)
        self.assertLess(result['geometry']['p95_reprojection_px'], 1e-6)
        self.assertGreater(result['maximum_camera_translation_mm'], 30)
        self.assertEqual(result['static_consistency'], 'three_or_more_views')

    def test_reference_placement_partition_preserves_total_and_relative_error(self):
        original = self.estimate_exact()['uncertainty']
        self.calibration.update(base_position_uncertainty_m=.020,
                                handeye_rotation_disagreement_deg=.4)
        changed = self.estimate_exact()['uncertainty']
        parts = changed['position_error_components']
        self.assertAlmostEqual(changed['depth_error_bound_m'], original['depth_error_bound_m'])
        self.assertAlmostEqual(parts['relative_surface_m'], changed['depth_error_bound_m'])
        self.assertEqual(parts['reference_translation_m'], .020)
        self.assertGreater(parts['reference_rotation_m'], 0)
        self.assertAlmostEqual(changed['base_position_error_bound_m'],
            parts['relative_surface_m'] + parts['reference_translation_m'] + parts['reference_rotation_m'])
        self.assertEqual(changed['gripper_axis_error_bound_m'], changed['base_position_error_bound_m'])

    def test_relative_view_uncertainty_remains_in_shape_component(self):
        before = self.estimate_exact()['uncertainty']['position_error_components']
        self.calibration.update(relative_translation_uncertainty_m=.003,
                                relative_rotation_uncertainty_deg=.3)
        after = self.estimate_exact()['uncertainty']['position_error_components']
        self.assertGreater(after['relative_surface_m'], before['relative_surface_m'])

    def test_measured_pose_disagreement_is_not_mislabeled_as_bad_pixel_matching(self):
        actual = [camera.copy() for camera in self.cameras]
        actual[1] = actual[1] @ transform(rotation=(.018, 0., 0.))
        actual[2] = actual[2] @ transform(y=.002, rotation=(-.014, 0., 0.))
        frames, model = self.fixture()  # Nominal FK, independently perturbed actual optics.
        tracks = self.exact_tracks(actual)
        self.calibration.update(relative_translation_uncertainty_m=.003,
                                relative_rotation_uncertainty_deg=2.)
        with patch.object(depth, '_image_tracks', return_value=(tracks, {})):
            result = depth.estimate_active_depth(frames, model, calibration=self.calibration)
        self.assertEqual(result['status'], 'estimated', result)
        self.assertGreater(result['geometry']['p95_reprojection_px'], 3.)
        allowance = result['tracking']['calibrated_reprojection']
        self.assertEqual(allowance['tracking_noise_px'], 3.)
        self.assertGreater(allowance['minimum_disparity_deg'], 0.)
        self.assertGreater(allowance['maximum_relative_depth_sensitivity'], .3)
        low, high = result['uncertainty']['camera_depth_interval_m']
        # Model disagreement is accepted as a COARSE estimate, not erased.
        true_depth = np.median([(inv(actual[-1]) @ np.r_[p, 1.])[2]
            for p in [[x, y, .4] for x in np.linspace(-.07, .07, 6)
                      for y in np.linspace(-.11, -.03, 5)]])
        self.assertLessEqual(low, true_depth)
        self.assertGreaterEqual(high, true_depth)
        self.assertEqual(result['uncertainty']['quality'], 'coarse')
        self.assertFalse(result['grasp_ready'])
        json.dumps(result, allow_nan=False)

    def test_pose_aperture_still_rejects_bad_matches_and_unresolved_disparity(self):
        actual = [camera.copy() for camera in self.cameras]
        actual[1] = actual[1] @ transform(rotation=(.018, 0., 0.))
        actual[2] = actual[2] @ transform(y=.002, rotation=(-.014, 0., 0.))
        track = self.exact_tracks(actual)[10]
        uncertainty = {'relative_translation_m': .003, 'relative_rotation_deg': 2.,
                       'focal_relative': .01}
        good, reason = depth._solve_track(track, self.cameras, self.k, np.zeros(5),
            self.size, pose_uncertainty=uncertainty)
        self.assertIsNone(reason)
        self.assertGreater(good['max_reprojection_px'], 3.)
        _, reason = depth._solve_track({**track, 2: track[2] + [0., 100.]},
            self.cameras, self.k, np.zeros(5), self.size, pose_uncertainty=uncertainty)
        self.assertEqual(reason, 'nonrigid_or_pose_inconsistent')
        for changes in ({'relative_translation_m': .05}, {'relative_rotation_deg': 8.}):
            _, reason = depth._solve_track(track, self.cameras, self.k, np.zeros(5),
                self.size, pose_uncertainty={**uncertainty, **changes})
            self.assertEqual(reason, 'pose_uncertainty_exceeds_parallax')
        _, reason = depth._solve_track(track, self.cameras, self.k, np.zeros(5), self.size)
        self.assertEqual(reason, 'nonrigid_or_pose_inconsistent')

    def test_unvalidated_model_cannot_opt_into_calibrated_pose_aperture(self):
        frames, model = self.fixture()
        tracks = self.exact_tracks(offsets=[[0., 0.], [0., 8.], [0., -8.]])
        calibration = {**self.calibration, 'handeye_validated': False,
                       'relative_translation_uncertainty_m': .013,
                       'relative_rotation_uncertainty_deg': 2.}
        with patch.object(depth, '_image_tracks', return_value=(tracks, {})):
            result = depth.estimate_active_depth(frames, model, calibration=calibration)
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['provenance']['reprojection_model'], 'fixed-pixel-residual')
        self.assertEqual(result['provenance']['reprojection_limit_px'], 3.)

    def test_translation_aperture_projects_larger_at_nearer_surface(self):
        uncertainty = {'relative_translation_m': .001, 'relative_rotation_deg': .1,
                       'focal_relative': .01}
        diagnostics = []
        for z in (.25, .6):
            point = np.array([0., -.05, z])
            track = {i: depth._project(point, camera, self.k, np.zeros(5))
                     for i, camera in enumerate(self.cameras)}
            solved, reason = depth._solve_track(track, self.cameras, self.k, np.zeros(5),
                self.size, pose_uncertainty=uncertainty)
            self.assertIsNone(reason)
            diagnostics.append(solved['calibrated_sensitivity'])
        self.assertGreater(diagnostics[0]['max_projected_pose_and_tracking_px'],
                           diagnostics[1]['max_projected_pose_and_tracking_px'])
        self.assertGreater(diagnostics[1]['relative_depth_error'], diagnostics[0]['relative_depth_error'])

    def test_relative_translation_is_not_counted_twice_as_ray_angle_noise(self):
        point = np.array([0., -.05, .4])
        track = {i: depth._project(point, camera, self.k, np.zeros(5))
                 for i, camera in enumerate(self.cameras)}
        results = []
        for translation in (.001, .003):
            solved, reason = depth._solve_track(track, self.cameras, self.k, np.zeros(5), self.size,
                pose_uncertainty={'relative_translation_m': translation,
                                  'relative_rotation_deg': .1, 'focal_relative': .01})
            self.assertIsNone(reason)
            results.append(solved['calibrated_sensitivity'])
        self.assertAlmostEqual(results[0]['minimum_disparity_deg'], results[1]['minimum_disparity_deg'])
        self.assertGreater(results[1]['triangulation_lever_error_m'], results[0]['triangulation_lever_error_m'])
        self.assertGreater(results[1]['relative_depth_error'], results[0]['relative_depth_error'])
        self.assertGreater(results[1]['max_projected_pose_and_tracking_px'], results[0]['max_projected_pose_and_tracking_px'])

    def test_visible_extent_uses_original_bbox_view_depth_and_marks_cropped_top(self):
        frames, model = self.fixture()
        tracks = self.exact_tracks()
        with patch.object(depth, '_image_tracks', return_value=(tracks, {})):
            result = depth.estimate_active_depth(frames, model,
                roi={'x': .3, 'y': 0., 'w': .25, 'h': .4}, roi_frame_id='view-0',
                calibration=self.calibration)
        self.assertEqual(result['status'], 'estimated', result)
        extent = result['visible_extent']
        self.assertEqual(extent['reference_frame_id'], 'view-0')
        self.assertAlmostEqual(extent['reference_camera_depth_m'], .4, delta=1e-8)
        self.assertAlmostEqual(extent['width_m'], .25 * 640 * .4 / 480, delta=1e-8)
        self.assertAlmostEqual(extent['height_m'], .4 * 480 * .4 / 480, delta=1e-8)
        self.assertEqual(extent['touches_image_edges'], ['top'])
        self.assertFalse(extent['full_object_dimensions_verified'])
        self.assertFalse(extent['object_dimensions_used'])
        self.assertLess(extent['width_interval_m'][0], extent['width_m'])
        self.assertGreater(extent['width_interval_m'][1], extent['width_m'])

    def test_scene_without_target_box_cannot_claim_object_dimensions(self):
        result = self.estimate_exact()
        self.assertEqual(result['status'], 'estimated')
        self.assertIsNone(result['visible_extent'])

    def test_coarse_reprojection_acceptance_and_geometry_are_resize_invariant(self):
        for offset, expected in ((2.8, None), (8., 'nonrigid_or_pose_inconsistent')):
            track = self.exact_tracks()[0]
            track[2] = track[2] + [0., offset]
            points = []
            for scale in (1, 2, 3):
                k = self.k.copy(); k[:2] *= scale
                result, reason = depth._solve_track({i: pixel * scale for i, pixel in track.items()},
                    self.cameras, k, np.zeros(5), (640 * scale, 480 * scale),
                    max_reprojection_px=depth.MAX_REPROJECTION_PX * 640 * scale / depth.COARSE_RESIDUAL_REFERENCE_WIDTH)
                self.assertEqual(reason, expected)
                if reason is None:
                    points.append(result['point'])
            if points:
                np.testing.assert_allclose(points, np.repeat([points[0]], 3, axis=0), atol=1e-9)
                # The strict default is preserved; coarse is an explicit opt-in.
                _, strict_reason = depth._solve_track({i: pixel * 3 for i, pixel in track.items()},
                    self.cameras, k, np.zeros(5), (1920, 1440))
                self.assertEqual(strict_reason, 'nonrigid_or_pose_inconsistent')

    def test_coarse_prior_cannot_cross_camera_sessions_or_claim_validated(self):
        frames, model = self.fixture()
        transforms = [model.fk.transforms(frame['joints']) for frame in frames]
        calibration = {**self.calibration, 'intrinsics_validated': False,
            'prior_quality': 'coarse', 'camera_generation': 'synthetic-wrist',
            'prior_refinement': 'same-attempt-disjoint-feature-fit-target-heldout'}
        *_, provenance = depth._camera_model(self.size, transforms, 'synthetic-wrist', calibration)
        self.assertEqual(provenance['prior_quality'], 'coarse')
        self.assertEqual(provenance['reprojection_limit_px'], 2.)
        self.assertFalse(provenance['metric_calibrated'])
        with self.assertRaisesRegex(ValueError, '同一次'):
            depth._camera_model(self.size, transforms, 'another-session', calibration)
        with self.assertRaisesRegex(ValueError, '已校準'):
            depth._camera_model(self.size, transforms, 'synthetic-wrist',
                {**calibration, 'intrinsics_validated': True})

    def test_coarse_prior_is_bound_to_its_contributing_photos_and_roi_frame(self):
        frames, model = self.fixture()
        calibration = {**self.calibration, 'intrinsics_validated': False,
            'prior_quality': 'coarse', 'camera_generation': 'synthetic-wrist',
            'prior_refinement': 'same-attempt-disjoint-feature-fit-target-heldout',
            'observation_frame_ids': [frame['id'] for frame in frames], 'roi_frame_id': 'view-0'}
        box = {'x': .2, 'y': .1, 'w': .6, 'h': .4}
        with patch.object(depth, '_image_tracks', return_value=(self.exact_tracks(), {})):
            valid = depth.estimate_active_depth(frames, model, roi=box, roi_frame_id='view-0', calibration=calibration)
        self.assertEqual(valid['status'], 'estimated', valid)
        for altered in ({**calibration, 'observation_frame_ids': ['earlier-observation']},
                        {**calibration, 'roi_frame_id': 'view-2'}):
            with patch.object(depth, '_image_tracks') as tracker:
                result = depth.estimate_active_depth(frames, model, roi=box, roi_frame_id='view-0', calibration=altered)
            self.assertEqual(result['status'], 'insufficient', result)
            self.assertIn('不屬於這一批照片', result['reason'])
            tracker.assert_not_called()

    def test_refitting_camera_does_not_decode_or_extract_same_images_twice(self):
        frames, model = self.fixture()
        tracks = self.exact_tracks()
        original = depth._solve_track
        count = [0]
        def initially_inconsistent(*args, **kwargs):
            count[0] += 1
            if count[0] <= len(tracks):
                return None, 'nonrigid_or_pose_inconsistent'
            return original(*args, **kwargs)
        with patch.object(depth, '_read_frames', wraps=depth._read_frames) as reader, \
             patch.object(depth, '_image_tracks', return_value=(tracks, {})) as tracker, \
             patch.object(depth, '_solve_track', side_effect=initially_inconsistent), \
             patch('server.active_camera_prior.refine_active_camera_prior', return_value={
                 'status': 'refined', 'calibration': self.calibration}):
            result = depth.estimate_active_depth(frames, model, refine_camera_prior=True)
        self.assertEqual(result['status'], 'estimated', result)
        reader.assert_called_once()
        tracker.assert_called_once()
        self.assertAlmostEqual(result['distance']['position_base_m'][2], .4, delta=1e-8)

    def test_visible_extent_includes_rectified_edge_extrema_with_distortion(self):
        # With radial distortion, the widest rectified span is not at the corners.
        extent = depth._visible_extent((.1, .1, .8, .8), 0, [{'id': 'reference'}],
            np.array([[0., 0., .4]]), [np.eye(4)], self.k,
            np.array([.5, 0., 0., 0., 0.]), self.size, .1)
        mid_edges = np.array([[64., 240.], [576., 240.], [320., 48.], [320., 432.]])
        normalized = cv2.undistortPoints(mid_edges.reshape(-1, 1, 2), self.k,
            np.array([.5, 0., 0., 0., 0.])).reshape(-1, 2)
        self.assertAlmostEqual(extent['width_m'], (normalized[1, 0] - normalized[0, 0]) * .4, delta=1e-5)
        self.assertAlmostEqual(extent['height_m'], (normalized[3, 1] - normalized[2, 1]) * .4, delta=1e-5)

    def test_below_declared_workspace_plane_rejects_even_consistent_reprojection(self):
        frames, model = self.fixture()
        with patch.object(depth, '_image_tracks', return_value=(self.exact_tracks(), {})):
            result = depth.estimate_active_depth(frames, model,
                roi={'x': .2, 'y': .1, 'w': .6, 'h': .4}, calibration=self.calibration,
                workspace_floor_z_m=1.)
        self.assertEqual(result['status'], 'insufficient', result)
        self.assertFalse(result['workspace_consistency']['passed'])
        self.assertAlmostEqual(result['workspace_consistency']['median_surface_z_m'], .4, delta=1e-8)
        self.assertIsNone(result['distance']); self.assertIsNone(result['visible_extent'])
        self.assertEqual(result['points'], [])
        self.assertIn('工作桌面下方', result['reason'])

    def test_workspace_check_is_explicit_and_does_not_fit_geometry_to_the_plane(self):
        frames, model = self.fixture()
        with patch.object(depth, '_image_tracks', return_value=(self.exact_tracks(), {})):
            result = depth.estimate_active_depth(frames, model, calibration=self.calibration,
                workspace_floor_z_m=0.)
        self.assertEqual(result['status'], 'estimated', result)
        self.assertTrue(result['workspace_consistency']['passed'])
        self.assertAlmostEqual(result['distance']['position_base_m'][2], .4, delta=1e-8)

    def test_camera_reference_is_the_transform_actually_used_for_depth(self):
        frames, model = self.fixture()
        handeye = transform(.01, -.005, .015, (.025, -.04, .01))
        cameras = [camera @ handeye for camera in self.cameras]
        tracks = self.exact_tracks(cameras)
        calibration = {**self.calibration, 'handeye': handeye.tolist(),
                       'intrinsics_validated': False,
                       'prior_refinement': 'same-attempt-background-fit-target-heldout'}
        with patch.object(depth, '_image_tracks', return_value=(tracks, {})):
            result = depth.estimate_active_depth(frames, model, calibration=calibration)
        self.assertEqual(result['status'], 'estimated', result)
        used_camera = np.asarray(result['camera_transform_base_m'])
        np.testing.assert_allclose(used_camera, cameras[-1])
        self.assertFalse(np.allclose(used_camera, self.cameras[-1]))
        np.testing.assert_allclose(used_camera[3], [0., 0., 0., 1.])
        for point in result['points']:
            local = inv(used_camera) @ np.r_[point['position_base_m'], 1.]
            self.assertAlmostEqual(float(local[2]), point['camera_depth_m'], delta=1e-10)
        json.dumps(result['camera_transform_base_m'], allow_nan=False)

    def test_rotation_without_camera_center_translation_cannot_create_depth(self):
        cameras = [transform(), transform(rotation=(0., .15, 0.)), transform(rotation=(.1, 0., .1))]
        frames, model = self.fixture(cameras)
        result = depth.estimate_active_depth(frames, model, calibration=self.calibration)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['distance'])
        self.assertEqual(result['maximum_camera_translation_mm'], 0)

    def test_zoom_or_bbox_growth_alone_cannot_create_metric_depth(self):
        frames, model = self.fixture([transform(), transform()])
        image = cv2.imread(frames[1]['path'])
        zoom = cv2.getRotationMatrix2D((320, 240), 0, 1.2)
        cv2.imwrite(frames[1]['path'], cv2.warpAffine(image, zoom, self.size))
        result = depth.estimate_active_depth(frames, model, roi={'x': .1, 'y': .1, 'w': .8, 'h': .8})
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['distance'])

    def test_real_tracking_rejects_self_gripper_only_texture(self):
        frames, model = self.fixture(bottom_only=True)
        result = depth.estimate_active_depth(frames, model, calibration=self.calibration)
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result.get('tracking', {}).get('retained'), 0)

    def test_lower_target_roi_survives_while_camera_fixed_jaw_features_are_removed(self):
        frames, model = self.fixture()
        jaw = np.random.default_rng(14).integers(0, 256, (120, 135), dtype=np.uint8)
        jaw = cv2.GaussianBlur(jaw, (3, 3), .5)
        for frame in frames:
            image = cv2.imread(frame['path'], cv2.IMREAD_GRAYSCALE)
            image[340:460, 435:570] = jaw
            cv2.imwrite(frame['path'], image)
        box = {'x': .1, 'y': .64, 'w': .8, 'h': .32}
        result = depth.estimate_active_depth(frames, model, roi=box, roi_frame_id='view-0',
                                              calibration=self.calibration)
        self.assertEqual(result['status'], 'estimated', result)
        self.assertTrue(result['tracking']['lower_target_roi_enabled'])
        self.assertGreater(result['tracking']['camera_fixed_target_identities_rejected'], 0)
        self.assertGreaterEqual(result['tracking']['retained'], 6)
        self.assertAlmostEqual(result['distance']['position_base_m'][2], .4, delta=.005)
        self.assertTrue(any(point['current_pixel'][1] > .595 * 480 for point in result['points']))

    def test_changed_jaw_disables_lower_image_exception(self):
        frames, model = self.fixture()
        frames[-1]['joints']['gripper'] += 100
        result = depth.estimate_active_depth(frames, model,
            roi={'x': .1, 'y': .7, 'w': .8, 'h': .25}, roi_frame_id='view-0',
            calibration=self.calibration)
        self.assertEqual(result['status'], 'insufficient')
        self.assertFalse(result['tracking']['lower_target_roi_enabled'])
        self.assertFalse(result['tracking']['jaw_unchanged_for_roi_expansion'])

    def test_inconsistent_object_motion_rejected_across_three_views(self):
        result = self.estimate_exact(offsets=[(0, 0), (0, 25), (0, -30)])
        self.assertEqual(result['status'], 'insufficient')
        self.assertGreater(result['tracking']['rejected'].get('nonrigid_or_pose_inconsistent', 0), 0)
        self.assertIsNone(result['distance'])

    def test_bad_tracks_are_removed_without_poisoning_consistent_static_features(self):
        frames, model = self.fixture()
        good = self.exact_tracks()
        bad = self.exact_tracks(offsets=[(0, 0), (30, 40), (-20, -30)])
        with patch.object(depth, '_image_tracks', return_value=(good + bad, {})):
            result = depth.estimate_active_depth(frames, model, calibration=self.calibration)
        self.assertEqual(result['status'], 'estimated', result)
        self.assertEqual(result['tracking']['retained'], len(good))
        self.assertAlmostEqual(result['distance']['position_base_m'][2], .4, delta=1e-8)

    def test_occluded_target_features_cannot_be_replaced_with_background_distance(self):
        frames, model = self.fixture()
        cv2.imwrite(frames[-1]['path'], np.full(self.size[::-1], 127, np.uint8))
        result = depth.estimate_active_depth(frames, model, roi={'x': .3, 'y': .1, 'w': .3, 'h': .35},
                                              calibration=self.calibration)
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['points'], [])

    def test_previous_roi_tracks_forward_instead_of_being_reused_as_current_pixels(self):
        frames, model = self.fixture()
        box = {'x': .2, 'y': .15, 'w': .08, 'h': .25}
        images = [cv2.imread(f['path'], cv2.IMREAD_GRAYSCALE) for f in frames]
        tracks, _ = depth._image_tracks(images, depth._roi(box), None, roi_index=0)
        self.assertGreaterEqual(len(tracks), 6)
        for track in tracks:
            self.assertTrue(.2 * 640 <= track[0][0] <= .28 * 640)
            self.assertTrue(.15 * 480 <= track[0][1] <= .4 * 480)
        self.assertTrue(any(track[2][0] > .28 * 640 for track in tracks))
        result = depth.estimate_active_depth(frames, model, roi=box,
            roi_frame_id=frames[0]['id'], calibration=self.calibration)
        self.assertEqual(result['status'], 'estimated', result)
        self.assertEqual(result['roi_frame_id'], frames[0]['id'])
        self.assertEqual(result['current_frame_id'], frames[-1]['id'])
        self.assertAlmostEqual(result['distance']['position_base_m'][2], .4, delta=.004)
        self.assertEqual(result['tracked_roi_current']['status'], 'tracked')
        self.assertTrue(result['tracked_roi_current']['framing_hint_only'])

    def test_framing_hint_survives_unusable_cad_camera_baseline_without_claiming_depth(self):
        frames, model = self.fixture()
        # Real translated images remain trackable when the camera model is
        # wrong: a framing hint must not depend on a successful metric solve.
        stationary = model.fk.transforms(frames[0]['joints'])
        model.fk.transforms = lambda raw: stationary
        result = depth.estimate_active_depth(frames, model,
            roi={'x': .2, 'y': .15, 'w': .08, 'h': .25}, roi_frame_id=frames[0]['id'])
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['distance'])
        self.assertEqual(result['tracked_roi_current']['status'], 'tracked')
        self.assertTrue(result['tracked_roi_current']['not_depth_evidence'])
        self.assertEqual(result['tracked_roi_current']['current_frame_id'], frames[-1]['id'])

    def test_missing_roi_reference_is_not_silently_applied_to_latest_image(self):
        frames, model = self.fixture()
        with patch.object(depth, '_image_tracks') as tracker:
            result = depth.estimate_active_depth(frames, model,
                roi={'x': .2, 'y': .1, 'w': .4, 'h': .4}, roi_frame_id='a-prior-job')
        self.assertEqual(result['status'], 'insufficient')
        self.assertIn('不在這批觀察', result['reason'])
        tracker.assert_not_called()

    def test_default_cad_scale_remains_an_explicit_coarse_prior(self):
        result = self.estimate_exact(calibration=False)
        self.assertEqual(result['status'], 'estimated', result)
        self.assertFalse(result['metric_calibrated'])
        self.assertEqual(result['provenance']['camera_model'], 'cad-camera-prior')
        self.assertFalse(result['provenance']['object_dimensions_used'])
        self.assertGreater(result['uncertainty']['depth_error_bound_m'], .1)
        self.assertFalse(result['uncertainty']['statistical_confidence_interval'])
        self.assertEqual(result['scope'], 'unassigned_scene_surface')

    def test_two_view_depth_explicitly_does_not_verify_static_object_motion(self):
        result = self.estimate_exact(self.cameras[:2])
        self.assertEqual(result['status'], 'estimated', result)
        self.assertEqual(result['static_consistency'], 'two_view_only_unverified_motion')
        self.assertFalse(result['grasp_ready'])
        self.assertIn('mimic depth', result['limitations'][0])

    def test_refined_prior_distance_uses_multi_view_subset_instead_of_two_view_majority(self):
        for three_count in (6, 3):
            with self.subTest(three_view_points=three_count):
                frames, model = self.fixture()
                tracks = self.exact_tracks()[:three_count]
                for x in np.linspace(-.1, .1, 24):
                    point = np.array([x, -.12, .8])
                    tracks.append({i: depth._project(point, self.cameras[i], self.k, np.zeros(5)) for i in (0, 2)})
                calibration = {**self.calibration, 'intrinsics_validated': False,
                               'prior_refinement': 'same-attempt-background-fit-target-heldout'}
                with patch.object(depth, '_image_tracks', return_value=(tracks, {})):
                    result = depth.estimate_active_depth(frames, model, calibration=calibration)
                self.assertEqual(result['status'], 'estimated', result)
                self.assertEqual(result['geometry']['available_two_view_points'], 24)
                if three_count >= 6:
                    self.assertEqual(result['static_consistency'], 'three_or_more_views')
                    self.assertEqual(result['geometry']['retained_points'], three_count)
                    self.assertEqual(result['geometry']['two_view_points_excluded_from_distance'], 24)
                    self.assertAlmostEqual(result['distance']['position_base_m'][2], .4, delta=1e-8)
                    self.assertTrue(all(point['views'] >= 3 for point in result['points']))
                else:
                    self.assertEqual(result['static_consistency'], 'mixed_two_and_three_view_support')
                    self.assertEqual(result['geometry']['retained_points'], 27)
                    self.assertAlmostEqual(result['distance']['position_base_m'][2], .8, delta=1e-8)

    def test_axial_translation_scale_crosscheck_removes_camera_rotation(self):
        cameras = [transform(), transform(z=.02, rotation=(0., .02, 0.)),
                   transform(z=.04, rotation=(.01, -.03, .035))]
        result = self.estimate_exact(cameras)
        self.assertEqual(result['status'], 'estimated', result)
        check = result['scale_crosscheck']
        self.assertEqual(check['status'], 'consistent')
        self.assertAlmostEqual(check['rotation_compensated_scale_ratio'], .4 / .36, delta=1e-8)
        self.assertAlmostEqual(check['predicted_scale_ratio'], .4 / .36, delta=1e-8)
        self.assertFalse(check['independent_validation'])

    def test_stale_boardless_artifact_is_never_silently_metric_calibration(self):
        artifact = {**self.calibration, 'needs_more_views': True,
                    'camera_generation': 'old-generation', 'handeye': np.eye(4).tolist(),
                    'status': 'estimated', 'metric_calibrated': True}
        path = Path(self.directory.name) / 'spatial.json'
        path.write_text(json.dumps(artifact))
        loaded = depth.load_camera_calibration(path)
        frames, model = self.fixture()
        with patch.object(depth, '_image_tracks', return_value=(self.exact_tracks(), {})):
            result = depth.estimate_active_depth(frames, model, calibration=loaded)
        self.assertFalse(result['metric_calibrated'])
        self.assertEqual(result['provenance']['camera_model'], 'unvalidated-handeye-prior')

    def test_d435_camera_change_bad_pose_and_bad_roi_never_reach_tracking(self):
        for change in ({'source': 'd435'}, {'camera_generation': 'other'}, {'sequence': 1},
                       {'joints': {'shoulder_pan': 2000}}):
            with self.subTest(change=change):
                frames, model = self.fixture()
                frames[-1].update(change)
                with patch.object(depth, '_image_tracks') as tracker:
                    result = depth.estimate_active_depth(frames, model)
                    self.assertEqual(result['status'], 'insufficient')
                    tracker.assert_not_called()
        frames, model = self.fixture()
        for roi in ({'x': .5, 'y': 0, 'w': .8, 'h': .4}, {'x': 0}, {'x': True, 'y': 0, 'w': .2, 'h': .2}):
            with patch.object(depth, '_image_tracks') as tracker:
                self.assertEqual(depth.estimate_active_depth(frames, model, roi=roi)['status'], 'insufficient')
                tracker.assert_not_called()

    def test_image_geometry_change_and_cancellation_fail_without_distance(self):
        frames, model = self.fixture()
        c = {**self.calibration, 'image_size': [320, 240]}
        result = depth.estimate_active_depth(frames, model, calibration=c)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIn('尺寸不符', result['reason'])
        cancel = threading.Event(); cancel.set()
        result = depth.estimate_active_depth(frames, model, cancel=cancel)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIn('已停止', result['reason'])

    def test_cancel_during_final_sample_serialization_does_not_publish_distance(self):
        frames, model = self.fixture()
        tracks = self.exact_tracks()
        cancel = threading.Event()
        original = depth.np.linspace
        def indexes(*args, **kwargs):
            yield from original(*args, **kwargs)
            cancel.set()  # Just after the final sample, before function return.
        with patch.object(depth, '_image_tracks', return_value=(tracks, {})), \
             patch.object(depth.np, 'linspace', side_effect=indexes):
            result = depth.estimate_active_depth(frames, model,
                calibration=self.calibration, cancel=cancel)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['distance'])
        self.assertEqual(result['points'], [])
        self.assertIn('已停止', result['reason'])


class TrackedROIFramingTests(unittest.TestCase):
    def setUp(self):
        self.frames = [{'id': 'old-exact-view'}, {'id': 'current-exact-view'}]
        self.roi = (.3, .4, .3, .4)
        self.size = (640, 480)
        self.points = np.array([[x, y] for x in np.linspace(200, 370, 6)
                                for y in np.linspace(205, 370, 5)])

    def tracks(self, affine):
        after = np.einsum('ij,kj->ki', affine, np.c_[self.points, np.ones(len(self.points))])
        return [{0: a, 1: b} for a, b in zip(self.points, after)]

    def hint(self, tracks, stable=True):
        return depth._tracked_roi_current(tracks, self.roi, 0, self.frames, self.size, stable)

    def test_known_affine_tracks_box_corners_without_claiming_detection_or_depth(self):
        affine = cv2.getRotationMatrix2D((290, 280), 9, 1.05)
        affine[:, 2] += [25, -12]
        result = self.hint(self.tracks(affine))
        self.assertEqual(result['status'], 'tracked', result)
        self.assertEqual(result['reference_frame_id'], 'old-exact-view')
        self.assertEqual(result['current_frame_id'], 'current-exact-view')
        np.testing.assert_allclose(result['affine_reference_to_current_px'], affine, atol=1e-4)
        self.assertTrue(result['framing_hint_only'])
        self.assertTrue(result['not_object_detection'])
        self.assertTrue(result['not_clearance_evidence'])
        self.assertTrue(result['not_depth_evidence'])
        self.assertTrue(result['not_physical_scale'])
        self.assertEqual(result['support']['inlier_identities'], len(self.points))
        self.assertGreater(result['support']['reference_feature_hull_fraction'], .5)
        json.dumps(result, allow_nan=False)

    def test_offscreen_extent_is_preserved_and_clipping_is_explicit(self):
        affine = np.array([[1., 0., 320.], [0., 1., 0.]])
        tracks = [t for t in self.tracks(affine) if t[1][0] < 640]
        result = self.hint(tracks)
        self.assertEqual(result['status'], 'tracked', result)
        self.assertAlmostEqual(result['projected_bbox']['x'], .8, delta=1e-6)
        self.assertAlmostEqual(result['projected_bbox']['w'], .3, delta=1e-6)
        self.assertAlmostEqual(result['bbox']['x'] + result['bbox']['w'], 1., delta=1e-6)
        self.assertAlmostEqual(result['visible_fraction_of_projected_bbox'], 2 / 3, delta=1e-6)

    def test_stationary_jaw_identities_are_excluded_from_framing_fit(self):
        moving = self.tracks(np.array([[1., 0., 15.], [0., 1., 7.]]))
        fixed = [{0: p, 1: p + [1., 0.]} for p in self.points]
        self.assertEqual(self.hint(fixed)['status'], 'unavailable')
        result = self.hint(moving + fixed)
        self.assertEqual(result['status'], 'tracked')
        self.assertEqual(result['support']['matched_identities'], len(moving))
        self.assertEqual(self.hint(moving, stable=False)['status'], 'unavailable')

    def test_occlusion_concentrated_features_and_inconsistent_matches_are_unavailable(self):
        moving = self.tracks(np.array([[1., 0., 15.], [0., 1., 7.]]))
        self.assertEqual(self.hint(moving[:7])['status'], 'unavailable')
        line = [{0: np.array([210., y]), 1: np.array([225., y + 7])} for y in np.linspace(210, 365, 12)]
        self.assertEqual(self.hint(line)['status'], 'unavailable')
        rng = np.random.default_rng(12)
        random = [{0: p, 1: rng.uniform([50, 50], [590, 430])} for p in self.points]
        self.assertEqual(self.hint(random)['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
