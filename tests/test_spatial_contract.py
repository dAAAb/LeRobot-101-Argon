"""Synthetic measured-geometry contracts; no devices or inferred calibration."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import unittest

import cv2
import numpy as np

from server.spatial_contract import (
    JOINTS, RigidTransform, ImageProcessing, WristCalibration, JawCalibration,
    build_wrist_observation, require_compatible, rectify_image,
)


def pose(translation=(0, 0, 0), rotation=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(np.array(rotation, dtype=float))[0]
    t[:3, 3] = translation
    return t


def frame(sequence=1, **changes):
    joints = {n: 2000 for n in JOINTS}
    stamp = float(sequence + 1)
    value = {'id': f'wrist-{sequence}', 'source': 'wrist', 'camera_generation': 'session-A',
             'sequence': sequence, 'capture_monotonic': stamp, 'image_size': [640, 480], 'joints': joints,
             'timestamp_basis': 'jpeg-arrival after settled joint window',
             'joint_observation_bracket_monotonic': [stamp - .1, stamp + .1],
             'joint_capture_span_ticks': 1, 'image_wait_joint_span_ticks': 2,
             'settled_joint_window': [
                 {'read_start_monotonic': stamp - .5, 'read_end_monotonic': stamp - .4, 'joints': dict(joints)},
                 {'read_start_monotonic': stamp - .3, 'read_end_monotonic': stamp - .2, 'joints': dict(joints)}]}
    value.update(changes)
    return value


def camera_profile():
    return {'schema_version': 1, 'calibration_revision': 'camera-v1', 'geometry_revision': 'geometry-v1',
            'camera_id': 'wrist', 'mount_revision': 'mount-v1', 'native_image_size': [640, 480],
            'camera_matrix': [[500., 0, 319.5], [0, 490., 239.5], [0, 0, 1]],
            'distortion': [.12, -.035, .002, -.001, 0.],
            'handeye': pose((.02, -.03, .04), (.1, -.15, .2)).tolist(),
            'validated_components': ['intrinsics', 'handeye', 'robot_geometry'],
            'validation_kind': 'independent-heldout', 'validation_id': 'synthetic-independent-optics'}


def jaw_profile():
    return {'schema_version': 1, 'calibration_revision': 'jaws-v1', 'geometry_revision': 'geometry-v1',
            'status': 'validated', 'validation_id': 'synthetic-two-pad', 'validation_kind': 'independent-heldout',
            'position_uncertainty_m': .002, 'approach_axis': [0., 0., 1.],
            'samples': [{'raw': 1500, 'fixed_pad_m': [-.01, 0., .1], 'moving_pad_m': [.01, 0., .1]},
                        {'raw': 2500, 'fixed_pad_m': [-.01, 0., .1], 'moving_pad_m': [.05, 0., .1]}]}


class SpatialContractTests(unittest.TestCase):
    def setUp(self):
        self.wrist = pose((.1, .2, .3), (.2, -.3, .4))
        self.mount = pose((.02, -.03, .04), (.1, -.15, .2))
        self.transforms = {'gripper_link': self.wrist, 'wrist_camera_optical': self.wrist @ self.mount}

    def observation(self, record=None, calibration=None, **kwargs):
        return build_wrist_observation(record or frame(), self.transforms, geometry_revision='geometry-v1',
            mount_revision='mount-v1', calibration=calibration, **kwargs)

    def test_transform_direction_composition_inverse_and_immutable_storage(self):
        source = pose((.2, -.4, .1), (.3, .2, -.1))
        a = RigidTransform('base', 'wrist', source, 'g1')
        b = RigidTransform('wrist', 'camera', self.mount, 'c1')
        point = np.array([.07, -.1, .4])
        expected = (source @ self.mount @ np.r_[point, 1])[:3]
        np.testing.assert_allclose(a.compose(b).apply(point), expected)
        np.testing.assert_allclose(a.compose(b).inverse().apply(expected), point)
        source[:] = 0
        self.assertGreater(np.linalg.det(a.as_array()[:3, :3]), .999)
        a.as_array()[:] = 0
        self.assertEqual(a.matrix[3][3], 1)
        with self.assertRaises(FrozenInstanceError):
            a.revision = 'changed'
        with self.assertRaises(ValueError):
            b.compose(a)

    def test_reject_reflection_scale_and_nonfinite_rigid_transforms(self):
        for diagonal in ((-1, 1, 1), (1.01, 1, 1), (1, 1, float('nan'))):
            t = np.eye(4); t[:3, :3] = np.diag(diagonal)
            with self.assertRaises(ValueError):
                RigidTransform('a', 'b', t, 'g1')

    def test_pixel_centre_resize_and_crop_preserve_projection(self):
        process = ImageProcessing((640, 480), (301, 177), (23, 17, 511, 339))
        k = np.array(camera_profile()['camera_matrix'])
        p = np.array([.12, -.04, .8])
        native = (k @ p)[:2] / p[2]
        expected = ((native - [23, 17] + .5) * np.array([301 / 511, 177 / 339]) - .5)
        actual = (process.camera_matrix(k) @ p)[:2] / p[2]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        with self.assertRaises(ValueError):
            ImageProcessing((640, 480), (300, 200), (0, 0, 641, 480))

    def test_native_resolution_changes_are_not_silent_intrinsics_resizing(self):
        with self.assertRaisesRegex(ValueError, 'resolution'):
            self.observation(frame(image_size=[1280, 720]), camera_profile())
        with self.assertRaisesRegex(ValueError, 'different acquired image'):
            self.observation(processing=ImageProcessing((320, 240), (320, 240)))

    def test_prior_reconstructs_measured_fk_without_claiming_calibration(self):
        observation = self.observation()
        np.testing.assert_allclose(observation.T_base_camera.as_array(), self.wrist @ self.mount, atol=1e-11)
        self.assertFalse(observation.metric_calibrated)
        self.assertEqual(observation.to_dict()['camera_model_status'], 'prior')
        self.assertTrue(observation.timing.pose_bound)
        self.assertFalse(observation.timing.exposure_time_verified)
        self.assertIsNone(observation.jaw_geometry)

    def test_actual_robot_fk_fixed_mount_has_one_identity_across_wrist_poses(self):
        from pathlib import Path
        from server.geometry import Kinematics
        fk = Kinematics(Path(__file__).resolve().parents[1] / 'assets/so101_follower_cam.urdf')
        initial = {'shoulder_pan': 2435, 'shoulder_lift': 979, 'elbow_flex': 3040,
                   'wrist_flex': 2168, 'wrist_roll': 658, 'gripper': 1859}
        observations = []
        for sequence, (flex, roll) in enumerate(((0, 0), (60, 60), (-60, -60),
                                                (20, -60), (-120, 120), (120, -120)), 1):
            joints = {**initial, 'wrist_flex': initial['wrist_flex'] + flex,
                      'wrist_roll': initial['wrist_roll'] + roll}
            record = frame(sequence, joints=joints, image_size=[1920, 1080])
            for sample in record['settled_joint_window']:
                sample['joints'] = dict(joints)
            transforms = fk.transforms(joints)
            packet = build_wrist_observation(record, transforms, geometry_revision='actual-urdf')
            np.testing.assert_allclose(packet.T_base_camera.as_array(),
                                       transforms['wrist_camera_optical'], atol=1e-11)
            self.assertTrue(packet.timing.pose_bound)
            self.assertFalse(packet.metric_calibrated)
            observations.append(packet)
        require_compatible(observations)
        self.assertEqual(len({p.calibration.calibration_revision for p in observations}), 1)
        self.assertEqual(len({p.calibration.fingerprint for p in observations}), 1)
        self.assertGreater(np.linalg.norm(observations[0].T_base_camera.as_array()
                                           - observations[-1].T_base_camera.as_array()), .1)
        # A real camera-mount translation must still invalidate fusion.
        moved_record = frame(7, joints=initial, image_size=[1920, 1080])
        moved_transforms = fk.transforms(initial)
        moved_transforms['wrist_camera_optical'][:3, 3] += [.001, 0., 0.]
        changed = build_wrist_observation(moved_record, moved_transforms,
                                          geometry_revision='actual-urdf')
        with self.assertRaisesRegex(ValueError, 'revisions'):
            require_compatible([observations[0], changed])

    def test_legacy_validation_booleans_do_not_promote_unversioned_profile(self):
        legacy = camera_profile()
        legacy.pop('schema_version')
        legacy.update(metric_calibrated=True, intrinsics_validated=True, handeye_validated=True,
                      robot_geometry_validated=True)
        observation = self.observation(calibration=legacy)
        self.assertFalse(observation.metric_calibrated)
        self.assertEqual(observation.calibration.validation_kind, 'prior')

    def test_profile_validation_requires_explicit_parameters_and_bindings(self):
        for missing in ('camera_matrix', 'distortion', 'handeye', 'geometry_revision', 'camera_id',
                        'mount_revision', 'native_image_size'):
            profile = camera_profile(); profile.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.observation(calibration=profile)
        for field in ('needs_more_views', 'resolution_inferred_from_principal_point'):
            profile = camera_profile(); profile[field] = True
            self.assertFalse(self.observation(calibration=profile).metric_calibrated)

    def test_versioned_profile_is_durable_across_sessions_but_observations_not_fused(self):
        first = self.observation(calibration=camera_profile())
        second = self.observation(frame(2, camera_generation='session-B'), camera_profile())
        self.assertTrue(first.metric_calibrated)
        self.assertTrue(second.metric_calibrated)
        self.assertEqual(first.calibration.fingerprint, second.calibration.fingerprint)
        with self.assertRaisesRegex(ValueError, 'sessions'):
            require_compatible([first, second])
        tied = camera_profile(); tied['camera_generation'] = 'session-A'
        with self.assertRaisesRegex(ValueError, 'session'):
            self.observation(frame(camera_generation='session-B'), tied)

    def test_schema_mount_camera_geometry_and_source_must_match(self):
        for field, value in (('schema_version', 2), ('mount_revision', 'reinstalled'),
                             ('camera_id', 'another-camera'), ('geometry_revision', 'new-geometry'),
                             ('source', 'd435'), ('distortion_model', 'kb4')):
            profile = camera_profile(); profile[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.observation(calibration=profile)
        for change in ({'source': 'd435'}, {'geometry_revision': 'new'}, {'sequence': 0},
                       {'capture_monotonic': float('nan')}, {'image_size': None}, {'id': ''}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.observation(frame(**change))

    def test_profile_content_fingerprint_catches_same_revision_different_intrinsics(self):
        profile = camera_profile()
        first = self.observation(calibration=profile)
        profile['camera_matrix'][0][0] += 2
        second = self.observation(frame(2), profile)
        self.assertEqual(first.calibration.calibration_revision, second.calibration.calibration_revision)
        self.assertNotEqual(first.calibration.fingerprint, second.calibration.fingerprint)
        with self.assertRaises(ValueError):
            require_compatible([first, second])

    def test_calibration_scope_and_empirical_errors_are_fingerprinted_and_enforced(self):
        profile = camera_profile()
        profile['validation_scope'] = {'joint_raw_bounds': {n: [1980, 2020] for n in JOINTS[:-1]},
                                       'individually_identified_joint_zeros': False}
        profile['relative_translation_uncertainty_m'] = .006
        inside = self.observation(calibration=profile)
        self.assertTrue(inside.metric_calibrated)
        self.assertEqual(inside.to_dict()['calibration_uncertainty']['relative_translation_uncertainty_m'], .006)
        moved = frame()
        moved['joints']['elbow_flex'] = 2030
        for sample in moved['settled_joint_window']:
            sample['joints']['elbow_flex'] = 2030
        outside = self.observation(moved, profile)
        self.assertTrue(outside.timing.pose_bound)
        self.assertFalse(outside.metric_calibrated)
        self.assertEqual(outside.to_dict()['validation_scope_status'], 'outside-measured-joint-bounds')
        self.assertIn('擴充校準', outside.to_dict()['validation_scope_reason'])
        changed = deepcopy(profile); changed['validation_scope']['joint_raw_bounds']['elbow_flex'][1] = 2050
        self.assertNotEqual(self.observation(calibration=changed).calibration.fingerprint, inside.calibration.fingerprint)
        changed = deepcopy(profile); changed['relative_translation_uncertainty_m'] = .007
        self.assertNotEqual(self.observation(calibration=changed).calibration.fingerprint, inside.calibration.fingerprint)
        bad = deepcopy(profile); bad['validation_scope']['joint_raw_bounds'].pop('wrist_roll')
        with self.assertRaises(ValueError):
            self.observation(calibration=bad)

    def test_compatibility_permits_distinct_processed_resolutions_with_explicit_K(self):
        first = self.observation(calibration=camera_profile())
        second = self.observation(frame(2), camera_profile(), processing=ImageProcessing((640, 480), (320, 240)))
        require_compatible([first, second])
        with self.assertRaisesRegex(ValueError, 'increasing'):
            require_compatible([second, first])

    def test_brown_projection_and_unprojection_roundtrip_in_robot_base_after_crop(self):
        observation = self.observation(calibration=camera_profile(),
            processing=ImageProcessing((640, 480), (321, 257), (20, 10, 600, 450)))
        camera_points = np.array([[-.11, -.07, .6], [.08, .09, .4], [.01, -.08, .3]])
        base_points = observation.T_base_camera.apply(camera_points)
        pixels = observation.project(base_points)
        reconstructed = observation.unproject(pixels, camera_points[:, 2])
        np.testing.assert_allclose(reconstructed, base_points, atol=2e-8)
        # Radial distance would produce the wrong answer at off-axis pixels.
        wrong = observation.unproject(pixels, np.linalg.norm(camera_points, axis=1))
        self.assertGreater(np.linalg.norm(wrong - base_points), .01)
        for bad_depth in (0, -.1, float('nan')):
            with self.assertRaises(ValueError):
                observation.unproject(pixels, bad_depth)

    def test_rectification_moves_distorted_marker_to_ideal_pinhole_pixel(self):
        profile = camera_profile()
        observation = self.observation(calibration=profile)
        point = np.array([[.14, -.09, .4]])
        distorted_pixel = observation.project(observation.T_base_camera.apply(point))[0]
        image = np.zeros((480, 640), np.uint8)
        cv2.circle(image, tuple(np.round(distorted_pixel).astype(int)), 6, 255, -1)
        rectified, rectified_observation = rectify_image(image, observation)
        moments = cv2.moments(rectified)
        actual = np.array([moments['m10'], moments['m01']]) / moments['m00']
        k = np.array(profile['camera_matrix']); projected = k @ point[0]
        np.testing.assert_allclose(actual, projected[:2] / projected[2], atol=.7)
        self.assertEqual(rectified_observation.to_dict()['distortion_model'], 'pinhole')
        self.assertEqual(rectified_observation.calibration.fingerprint, observation.calibration.fingerprint)
        self.assertEqual(observation.distortion, tuple(profile['distortion']))
        with self.assertRaises(ValueError):
            rectify_image(rectified, rectified_observation)

    def test_missing_or_unstable_timing_never_becomes_verified_pose(self):
        record = frame(); record.pop('joint_observation_bracket_monotonic')
        result = self.observation(record, camera_profile())
        self.assertTrue(result.calibration.metric_calibrated)
        self.assertFalse(result.metric_calibrated)
        self.assertEqual(result.timing.mode, 'unbound')
        record = frame(image_wait_joint_span_ticks=6)
        self.assertFalse(self.observation(record, camera_profile()).metric_calibrated)
        record = frame()
        record['settled_joint_window'][0]['joints']['elbow_flex'] += 20
        self.assertFalse(self.observation(record, camera_profile()).metric_calibrated)
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.observation(frame(joint_observation_bracket_monotonic=[3., 4.]), camera_profile())

    def test_actual_six_joint_readings_not_goal_targets_or_incomplete_pose(self):
        for joints in ({n: 2000 for n in JOINTS[:-1]}, {n: True for n in JOINTS},
                       {n: 5000 for n in JOINTS}, {n: 2000. for n in JOINTS}):
            with self.assertRaises(ValueError):
                self.observation(frame(joints=joints))

    def test_jaw_tcp_tracks_moving_center_and_measured_opening(self):
        calibration = JawCalibration.from_mapping(jaw_profile(), geometry_revision='geometry-v1')
        geometry = calibration.at(2000)
        self.assertEqual(geometry.jaw_raw, 2000)
        self.assertEqual(geometry.geometry_revision, 'geometry-v1')
        self.assertTrue(geometry.metric_calibrated)
        self.assertAlmostEqual(geometry.aperture_m, .04)
        np.testing.assert_allclose(geometry.T_wrist_tcp.as_array()[:3, 3], [.01, 0, .1])
        np.testing.assert_allclose(calibration.at(1500).T_wrist_tcp.as_array()[:3, 3], [0, 0, .1])
        np.testing.assert_allclose(calibration.at(2500).T_wrist_tcp.as_array()[:3, 3], [.02, 0, .1])
        self.assertNotEqual(geometry.T_wrist_tcp.child, 'gripper_frame_link')
        for raw in (1499, 2501, 2000., True):
            with self.assertRaises(ValueError):
                calibration.at(raw)

    def test_jaw_orientation_is_full_transform_and_bound_to_geometry(self):
        profile = jaw_profile(); profile['approach_axis'] = [0, 1, 0]
        calibration = JawCalibration.from_mapping(profile, geometry_revision='geometry-v1')
        geometry = calibration.at(2000)
        np.testing.assert_allclose(geometry.T_wrist_tcp.as_array()[:3, 2], [0, 1, 0])
        self.assertAlmostEqual(np.linalg.det(geometry.T_wrist_tcp.as_array()[:3, :3]), 1.)
        observation = self.observation(calibration=camera_profile(), jaw_calibration=calibration)
        self.assertEqual(observation.to_dict()['jaw_geometry']['jaw_raw'], frame()['joints']['gripper'])
        with self.assertRaises(ValueError):
            self.observation(jaw_calibration=replace(calibration, geometry_revision='different'))

    def test_jaw_prior_is_distinct_and_validation_is_not_default_zero_error(self):
        prior = jaw_profile(); prior.update(status='prior', validation_kind='prior')
        prior.pop('position_uncertainty_m')
        geometry = JawCalibration.from_mapping(prior, geometry_revision='geometry-v1').at(2000)
        self.assertFalse(geometry.metric_calibrated)
        self.assertIsNone(geometry.position_uncertainty_m)
        for missing in ('position_uncertainty_m', 'validation_id', 'validation_kind'):
            bad = jaw_profile(); bad.pop(missing)
            with self.assertRaises(ValueError):
                JawCalibration.from_mapping(bad, geometry_revision='geometry-v1')

    def test_invalid_jaw_sample_axis_or_source_does_not_invent_tcp(self):
        for change in ({'source': 'd435'}, {'approach_axis': [0, 0, 2]}, {'schema_version': 2},
                       {'position_uncertainty_m': -1}):
            profile = {**jaw_profile(), **change}
            with self.assertRaises(ValueError):
                JawCalibration.from_mapping(profile, geometry_revision='geometry-v1')
        profile = jaw_profile(); profile['approach_axis'] = [1, 0, 0]
        with self.assertRaises(ValueError):
            JawCalibration.from_mapping(profile, geometry_revision='geometry-v1').at(2000)

    def test_per_opening_contact_section_axes_preserve_actual_interpolated_gap(self):
        profile = jaw_profile()
        profile['landmark_definition'] = 'matched_contact_cross_section'
        for sample, angle in zip(profile['samples'], (0., np.deg2rad(20.))):
            rotation = cv2.Rodrigues(np.array([0., angle, 0.]))[0]
            fixed = np.array([0., 0., .1])
            sample.update(fixed_pad_m=fixed.tolist(),
                moving_pad_m=(fixed + .04 * rotation[:, 0]).tolist(),
                approach_axis=rotation[:, 2].tolist())
        calibration = JawCalibration.from_mapping(profile, geometry_revision='geometry-v1')
        start, end = profile['samples']
        moving = (np.array(start['moving_pad_m']) + end['moving_pad_m']) / 2
        fixed = np.array(start['fixed_pad_m'])
        geometry = calibration.at(2000)
        np.testing.assert_allclose(geometry.T_wrist_tcp.as_array()[:3, 3], (fixed + moving) / 2)
        self.assertAlmostEqual(geometry.aperture_m, np.linalg.norm(moving - fixed))
        expected_axis = np.array(start['approach_axis']) + end['approach_axis']
        expected_axis /= np.linalg.norm(expected_axis)
        np.testing.assert_allclose(geometry.T_wrist_tcp.as_array()[:3, 2], expected_axis)
        self.assertEqual(geometry.landmark_definition, 'matched_contact_cross_section')
        self.assertIn('not the entire insertion corridor', geometry.to_dict()['aperture_scope'])
        self.assertFalse(np.allclose(expected_axis, profile['approach_axis']))

    def test_per_opening_axes_cannot_hide_unmatched_fore_aft_pad_points(self):
        profile = jaw_profile()
        profile['landmark_definition'] = 'matched_contact_cross_section'
        for sample in profile['samples']:
            sample['approach_axis'] = [0., 0., 1.]
            sample['moving_pad_m'][2] += .02
        calibration = JawCalibration.from_mapping(profile, geometry_revision='geometry-v1')
        with self.assertRaisesRegex(ValueError, 'perpendicular'):
            calibration.at(2000)

    def test_section_axes_are_complete_unit_and_consistently_oriented(self):
        for axes in ((None, None), ([0., 0., 1.], None),
                     ([0., 0., 2.], [0., 0., 1.]), ([0., 0., 1.], [0., 0., -1.])):
            profile = jaw_profile(); profile['landmark_definition'] = 'matched_contact_cross_section'
            for sample, axis in zip(profile['samples'], axes):
                sample['approach_axis'] = axis
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                JawCalibration.from_mapping(profile, geometry_revision='geometry-v1')

    def test_legacy_jaw_identity_is_stable_and_new_axes_are_fingerprinted(self):
        legacy = JawCalibration.from_mapping(jaw_profile(), geometry_revision='geometry-v1')
        old_payload = {'revision': legacy.calibration_revision, 'geometry': legacy.geometry_revision,
            'samples': [(s.raw, s.fixed_pad_m, s.moving_pad_m) for s in legacy.samples],
            'approach_axis': legacy.approach_axis, 'position_uncertainty_m': legacy.position_uncertainty_m,
            'status': legacy.status, 'validation_id': legacy.validation_id,
            'validation_kind': legacy.validation_kind}
        expected = hashlib.sha256(json.dumps(old_payload, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.assertEqual(legacy.fingerprint, expected)
        profile = jaw_profile(); profile['landmark_definition'] = 'matched_contact_cross_section'
        for sample in profile['samples']:
            sample['approach_axis'] = [0., 0., 1.]
        section = JawCalibration.from_mapping(profile, geometry_revision='geometry-v1')
        self.assertNotEqual(section.fingerprint, legacy.fingerprint)
        profile['samples'][-1]['approach_axis'] = [0., np.sin(.01), np.cos(.01)]
        changed = JawCalibration.from_mapping(profile, geometry_revision='geometry-v1')
        self.assertNotEqual(changed.fingerprint, section.fingerprint)

    def test_serialized_data_is_detached_finite_and_retains_provenance(self):
        observation = self.observation(calibration=camera_profile())
        value = observation.to_dict()
        json.dumps(value, allow_nan=False)
        self.assertEqual(value['calibration_validation_id'], 'synthetic-independent-optics')
        self.assertEqual(value['mount_revision'], 'mount-v1')
        value['T_base_camera']['matrix'][0][0] = 900
        value['camera_matrix'][0][0] = 0
        self.assertNotEqual(observation.T_base_camera.matrix[0][0], 900)
        self.assertEqual(observation.camera_matrix[0][0], 500.)


if __name__ == '__main__':
    unittest.main()
