"""Synthetic optics with known hand-eye and fixed metric board; no hardware."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from server.board_calibration import BoardSpec, detect_board, fit_board_calibration, _fit_intrinsics, _brown_image_domain
from server.spatial_contract import JOINTS, build_wrist_observation


def transform(translation=(0., 0., 0.), rotation=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.array(rotation, dtype=float))[0]
    result[:3, 3] = translation
    return result


def inverse(t):
    result = np.eye(4)
    result[:3, :3] = t[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ t[:3, 3]
    return result


class BoardCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.board = BoardSpec()
        self.size = [1280, 960]
        self.k = np.array([[930., 0, 639.5], [0, 915., 479.5], [0, 0, 1.]])
        self.distortion = np.array([.055, -.012, .0007, -.0005, 0.])
        self.handeye = transform((.025, -.018, .042), (.15, -.1, .08))
        self.base_board = transform((0., 0., .55), (.03, -.02, .04))

    def dataset(self, count=16, single_axis=False, noise=.025):
        rng = np.random.default_rng(732)
        frames, poses = [], {}
        for i in range(count):
            angle = i * 2.399
            rotation = ((0., 0., .35 * np.sin(angle)) if single_axis else
                        (.30 * np.sin(angle), .28 * np.cos(angle * 1.27), .23 * np.sin(angle * .72)))
            camera = transform((.08 + .025 * np.sin(angle * .5),
                                .05 + .025 * np.cos(angle * .8), .04 * np.sin(angle)), rotation)
            gripper = camera @ inverse(self.handeye)
            joints = {n: 2000 + int(80 * np.sin(angle + j)) for j, n in enumerate(JOINTS)}
            joints['shoulder_pan'] = 1500 + 13 * i
            poses[joints['shoulder_pan']] = gripper
            camera_board = inverse(camera) @ self.base_board
            points = cv2.projectPoints(self.board.object_points(), cv2.Rodrigues(camera_board[:3, :3])[0],
                camera_board[:3, 3], self.k, self.distortion)[0].reshape(-1, 2)
            points += rng.normal(0., noise, points.shape)
            self.assertTrue(np.all(points >= 0))
            self.assertTrue(np.all(points < self.size))
            stamp = float(i + 2)
            frame = {'id': f'board-frame-{i}', 'source': 'wrist', 'sequence': i + 1,
                'camera_generation': 'camera-A', 'capture_monotonic': stamp, 'image_size': list(self.size),
                'joints': joints, 'board_session_id': 'fixed-board-A',
                'timestamp_basis': 'jpeg-arrival after settled joint window',
                'joint_observation_bracket_monotonic': [stamp - .05, stamp + .05],
                'joint_capture_span_ticks': 0, 'image_wait_joint_span_ticks': 0,
                'settled_joint_window': [
                    {'read_start_monotonic': stamp - .7, 'read_end_monotonic': stamp - .6, 'joints': dict(joints)},
                    {'read_start_monotonic': stamp - .2, 'read_end_monotonic': stamp - .1, 'joints': dict(joints)}]}
            frame['board_detection'] = {'found': True, 'corners_px': points.tolist(), 'image_size': list(self.size),
                'board_spec': self.board.to_dict(), 'board_fingerprint': self.board.fingerprint,
                'frame_id': frame['id'], 'camera_generation': frame['camera_generation'],
                'sequence': frame['sequence'], 'capture_monotonic': stamp, 'coverage_fraction': .1}
            frames.append(frame)
        def fk(joints):
            wrist = poses[joints['shoulder_pan']]
            return {'gripper_link': wrist, 'wrist_camera_optical': wrist @ self.handeye}
        return frames, SimpleNamespace(fk=SimpleNamespace(transforms=fk))

    def fit(self, frames, model, **kwargs):
        return fit_board_calibration(frames, model, camera_id='wrist-A', mount_revision='mount-A',
            geometry_revision='geometry-A', board_session_id='fixed-board-A', **kwargs)

    def test_recovers_metric_intrinsics_and_handeye_from_measured_multi_pose_dataset(self):
        frames, model = self.dataset()
        result = self.fit(frames, model)
        self.assertEqual(result['status'], 'validated', result)
        profile = result['calibration_profile']
        np.testing.assert_allclose(np.diag(profile['camera_matrix'])[:2], np.diag(self.k)[:2], rtol=.006)
        fitted = np.array(profile['handeye'])
        self.assertLess(np.linalg.norm(fitted[:3, 3] - self.handeye[:3, 3]), .003)
        self.assertLess(np.linalg.norm(cv2.Rodrigues(fitted[:3, :3].T @ self.handeye[:3, :3])[0]), .008)
        validation = result['validation']
        self.assertFalse(validation['heldout_data_used_for_fitting'])
        self.assertLess(validation['heldout_max_translation_m'], .004)
        self.assertLess(validation['heldout_max_reprojection_p95_px'], 1.)
        scope = profile['validation_scope']
        self.assertFalse(scope['individually_identified_joint_zeros'])
        self.assertFalse(scope['jaw_contact_geometry_calibrated'])
        self.assertFalse(scope['extrapolation_allowed'])
        self.assertIsNone(profile['camera_generation'])
        self.assertEqual(len(validation['heldout_frame_ids']), 4)
        self.assertEqual(set(profile['validated_components']), {'intrinsics', 'handeye', 'robot_geometry'})
        packet = build_wrist_observation(frames[0], model.fk.transforms(frames[0]['joints']),
            geometry_revision='geometry-A', mount_revision='mount-A', camera_id='wrist-A', calibration=profile)
        self.assertTrue(packet.metric_calibrated)

    def test_heldout_pixels_never_change_fitted_intrinsics_or_handeye(self):
        frames, model = self.dataset()
        reference = self.fit(frames, model)
        changed = deepcopy(frames)
        for i in (3, 7, 11, 15):
            points = np.array(changed[i]['board_detection']['corners_px']) + [.8, -.6]
            changed[i]['board_detection']['corners_px'] = points.tolist()
        second = self.fit(changed, model)
        np.testing.assert_array_equal(reference['calibration_profile']['camera_matrix'], second['calibration_profile']['camera_matrix'])
        np.testing.assert_array_equal(reference['calibration_profile']['handeye'], second['calibration_profile']['handeye'])
        self.assertEqual(reference['validation']['intrinsics_model_selection'],
                         second['validation']['intrinsics_model_selection'])
        self.assertGreater(second['validation']['heldout_max_reprojection_p95_px'],
                           reference['validation']['heldout_max_reprojection_p95_px'])

    def test_training_only_selection_rejects_low_residual_but_folded_lens_extrapolation(self):
        # Nine tilted views of a board near the image centre do not strongly
        # constrain high-order edge distortion. Fit actual synthetic optics,
        # not mocked calibrateCamera results, to reproduce that ambiguity.
        points = self.board.object_points()
        k = np.array([[965., 0., 880.], [0., 970., 515.], [0., 0., 1.]])
        distortion = np.array([.095, 0., 0., 0., 0.])
        rng = np.random.default_rng(0)
        pixels = []
        for index in range(9):
            angle = index * 2.399
            rvec = np.array([.65 + .12 * np.sin(angle), .16 * np.cos(angle * 1.2),
                             .18 * np.sin(angle * .7)])
            tvec = np.array([-.08 + .035 * np.sin(angle), -.04 + .035 * np.cos(angle),
                             .4 + .015 * np.sin(angle * .5)])
            projected = cv2.projectPoints(points, rvec, tvec, k, distortion)[0].reshape(-1, 2)
            pixels.append(projected + rng.normal(0., .22, projected.shape))
        fit, selection = _fit_intrinsics(points, pixels, (1920, 1080))
        full = selection['candidates'][0]
        self.assertLess(full['training_rms_px'], .31)
        self.assertFalse(full['admissible'])
        self.assertIn('folds', full['image_domain']['reason'])
        self.assertEqual(selection['selected_model'], 'brown-k1')
        self.assertFalse(selection['heldout_used_for_selection'])
        self.assertEqual(selection['training_view_count'], 9)
        self.assertLess(float(fit[0]), full['training_rms_px'] + .002)
        domain = _brown_image_domain(fit[1], fit[2], (1920, 1080))
        self.assertTrue(domain['valid'])
        self.assertGreater(domain['jacobian_eigenvalue_lower_bound'], 0.)
        self.assertGreaterEqual(domain['boundary_radius_lower_bound'], domain['image_corner_radius'])
        self.assertLess(domain['inverse_roundtrip_max_px'], .25)

    def test_image_domain_rejects_tangential_folding_even_with_monotone_radial_term(self):
        k = np.array([[960., 0., 959.5], [0., 960., 539.5], [0., 0., 1.]])
        self.assertTrue(_brown_image_domain(k, np.zeros(5), (1920, 1080))['valid'])
        tangential_fold = _brown_image_domain(k, np.array([0., 0., .25, .25, 0.]), (1920, 1080))
        self.assertFalse(tangential_fold['valid'])
        self.assertLess(tangential_fold['jacobian_eigenvalue_lower_bound'], 0.)

    def test_moved_board_in_heldout_view_fails_with_observed_metric_error(self):
        frames, model = self.dataset(noise=0.)
        moved = deepcopy(frames)
        index = 7
        wrist = model.fk.transforms(frames[index]['joints'])['gripper_link']
        shifted_board = self.base_board.copy(); shifted_board[0, 3] += .05
        camera_board = inverse(wrist @ self.handeye) @ shifted_board
        moved[index]['board_detection']['corners_px'] = cv2.projectPoints(self.board.object_points(),
            cv2.Rodrigues(camera_board[:3, :3])[0], camera_board[:3, 3], self.k, self.distortion)[0].reshape(-1, 2).tolist()
        result = self.fit(moved, model)
        self.assertEqual(result['status'], 'rejected', result)
        self.assertGreater(result['validation']['heldout_max_translation_m'], .045)
        self.assertFalse(result['calibration_profile']['metric_calibrated'])
        self.assertEqual(result['calibration_profile']['validated_components'], [])

    def test_single_rotation_axis_is_not_observable_handeye(self):
        frames, model = self.dataset(single_axis=True)
        result = self.fit(frames, model)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIn('不同轉向', result['blockers'][0])
        self.assertIsNone(result['calibration_profile'])
        self.assertLess(result['validation']['observability']['second_axis_ratio'], 1e-6)

    def test_no_near_duplicate_pose_leaks_from_holdout_into_training(self):
        frames, model = self.dataset()
        duplicate = deepcopy(frames[3])
        duplicate.update(id='return-to-heldout', sequence=17, capture_monotonic=18.)
        duplicate['joint_observation_bracket_monotonic'] = [17.95, 18.05]
        duplicate['board_detection'].update(frame_id=duplicate['id'], sequence=17, capture_monotonic=18.)
        frames.append(duplicate)
        result = self.fit(frames, model, holdout_indices=[3, 7, 11, 15])
        self.assertEqual(result['status'], 'validated', result)
        self.assertIn('return-to-heldout', result['validation']['heldout_frame_ids'])
        self.assertNotIn('return-to-heldout', result['validation']['training_frame_ids'])

    def test_session_frame_source_resolution_and_corner_cache_are_bound(self):
        original, model = self.dataset()
        for key, value in (('source', 'd435'), ('board_session_id', 'moved-board'),
                           ('camera_generation', 'another-session'), ('image_size', [640, 480])):
            frames = deepcopy(original); frames[5][key] = value
            result = self.fit(frames, model)
            self.assertEqual(result['status'], 'rejected', (key, result))
            self.assertIsNone(result['calibration_profile'])
        frames = deepcopy(original); frames[5]['board_detection']['sequence'] = 1234
        self.assertIn('stale', self.fit(frames, model)['blockers'][0])
        frames = deepcopy(original); frames[5]['board_detection']['board_fingerprint'] = 'wrong-print-size'
        self.assertIn('mismatched', self.fit(frames, model)['blockers'][0])

    def test_unbound_pose_or_insufficient_views_yields_specific_next_action(self):
        frames, model = self.dataset()
        result = self.fit(frames[:4], model)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIn('8 個', result['blockers'][0])
        frames[3].pop('joint_observation_bracket_monotonic')
        self.assertIn('實測關節配對', self.fit(frames, model)['blockers'][0])
        self.assertEqual(self.fit([], model)['status'], 'insufficient')

    def test_actual_chessboard_detector_fixes_corner_order_through_180_degree_roll(self):
        square = 40
        image = np.full((480, 640), 255, np.uint8)
        origin = np.array([120, 80])
        for y in range(self.board.rows + 1):
            for x in range(self.board.columns + 1):
                if (x + y) % 2 == 0:
                    image[origin[1]+y*square:origin[1]+(y+1)*square,
                          origin[0]+x*square:origin[0]+(x+1)*square] = 0
        first = detect_board(image)
        self.assertTrue(first['found'], first)
        expected = origin + square - .5
        np.testing.assert_allclose(first['corners_px'][0], expected, atol=.6)
        rotated = detect_board(cv2.rotate(image, cv2.ROTATE_180))
        self.assertTrue(rotated['found'], rotated)
        np.testing.assert_allclose(rotated['corners_px'][0], [639, 479] - expected, atol=.6)
        np.testing.assert_allclose(np.array(rotated['corners_px']) + np.array(first['corners_px']),
                                   np.tile([639, 479], (54, 1)), atol=.7)

    def test_detector_missing_pattern_reports_found_false_and_source_is_wrist_only(self):
        blank = np.full((480, 640), 180, np.uint8)
        result = detect_board(blank)
        self.assertFalse(result['found'])
        self.assertEqual(result['corners_px'], [])
        with self.assertRaises(ValueError):
            detect_board(blank, frame={'source': 'd435', 'image_size': [640, 480]})
        with self.assertRaises(ValueError):
            BoardSpec(columns=8, rows=6)

    def test_bounded_search_refines_native_perspective_pixels_and_rejects_partial_board(self):
        from unittest.mock import patch
        square = 72
        source = np.full((800, 1000), 255, np.uint8)
        for y in range(self.board.rows + 1):
            for x in range(self.board.columns + 1):
                if (x + y) % 2 == 0:
                    source[100+y*square:100+(y+1)*square,
                           100+x*square:100+(x+1)*square] = 0
        # Nonintegral image reduction and a tilted board exercise both pixel
        # axes. Ground truth comes from the renderer, not the detector output.
        homography = cv2.getPerspectiveTransform(
            np.array([[0, 0], [999, 0], [999, 799], [0, 799]], np.float32),
            np.array([[360, 160], [1600, 260], [1420, 1120], [270, 960]], np.float32))
        image = cv2.warpPerspective(source, homography, (2003, 1199), borderValue=255)
        corners = np.mgrid[1:10, 1:7].T.reshape(-1, 2) * square + [99.5, 99.5]
        expected = cv2.perspectiveTransform(corners.astype(np.float32)[None], homography)[0]
        record = {'id': 'full-native', 'source': 'wrist', 'image_size': [2003, 1199],
                  'camera_generation': 'same-session', 'sequence': 1, 'capture_monotonic': 2.}
        native_detector = cv2.findChessboardCornersSB
        with patch('server.board_calibration.cv2.findChessboardCornersSB', wraps=native_detector) as search:
            result = detect_board(image, frame=record)
            self.assertTrue(result['found'], result)
            self.assertEqual(result['image_size'], record['image_size'])
            self.assertEqual(result['frame_id'], record['id'])
            errors = np.linalg.norm(np.array(result['corners_px']) - expected, axis=1)
            self.assertLess(float(np.percentile(errors, 95)), .2)
            self.assertGreater(np.max(result['corners_px']), 960.)
            self.assertLessEqual(max(search.call_args.args[0].shape), 960)
            self.assertFalse(search.call_args.kwargs['flags'] & cv2.CALIB_CB_EXHAUSTIVE)
            self.assertFalse(search.call_args.kwargs['flags'] & cv2.CALIB_CB_ACCURACY)
        # An incomplete pattern must fail promptly at the bounded search size;
        # it must not invent occluded corners or fall back to exhaustive 2 MP.
        source[:, 100:100+3*square] = 255
        occluded = cv2.warpPerspective(source, homography, (2003, 1199), borderValue=255)
        with patch('server.board_calibration.cv2.findChessboardCornersSB', wraps=native_detector) as search:
            missing = detect_board(occluded)
            self.assertFalse(missing['found'])
            self.assertEqual(missing['corners_px'], [])
            self.assertEqual(search.call_count, 1)
            self.assertLessEqual(max(search.call_args.args[0].shape), 960)


if __name__ == '__main__':
    unittest.main()
