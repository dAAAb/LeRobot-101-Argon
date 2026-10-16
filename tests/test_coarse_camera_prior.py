"""Coarse wrist-only priors keep disjoint evidence and resolution-scaled errors."""
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from server import active_camera_prior as prior
from server import active_prior_tracks as tracking
import test_active_camera_prior as fixture
from test_active_depth import transform


class CoarseCameraPriorTests(unittest.TestCase):
    setUp = fixture.ActiveCameraPriorTests.setUp
    tracks = fixture.ActiveCameraPriorTests.tracks

    def split(self):
        tracks = self.tracks(5)
        train = tracks[:100] + tracks[100::2]
        test = {i: t for i, t in enumerate(tracks[101::2])}
        return train, test

    def run_prior(self, train, validation):
        # Numeric fit tests should not depend on unrelated CPU load. Expiry and
        # cancellation have separate tests using an explicitly advancing clock.
        with patch.object(tracking, 'multi_anchor_tracks', return_value=(train, validation, {'target_used_for_fit': True})), \
             patch.object(prior, 'time', SimpleNamespace(monotonic=lambda: 0.)):
            return prior.refine_active_camera_prior(self.frames, self.model,
                roi=self.roi, roi_frame_id='0', mode='coarse')

    def test_real_fit_holds_out_target_identities_and_entire_poses(self):
        train, validation = self.split()
        with patch.object(prior, 'fit_focal_and_handeye', wraps=prior.fit_focal_and_handeye) as fits:
            result = self.run_prior(train, validation)
        self.assertEqual(result['status'], 'refined', result)
        self.assertEqual(fits.call_count, 2)
        self.assertTrue(result['validation']['target_used_for_fit'])
        self.assertFalse(result['validation']['heldout_feature_identities_used_for_fit'])
        self.assertEqual(result['calibration']['prior_quality'], 'coarse')
        self.assertEqual(result['calibration']['residual_limit_px'], 2.)
        self.assertEqual(result['calibration']['observation_frame_ids'], [f['id'] for f in self.frames])
        self.assertEqual(result['calibration']['roi_frame_id'], '0')
        self.assertFalse(result['metric_calibrated'])
        self.assertFalse(result['grasp_ready'])
        self.assertFalse(result['calibration']['intrinsics_validated'])
        self.assertAlmostEqual(result['calibration']['camera_matrix'][0][0], 520, delta=8)
        self.assertLess(result['validation']['target_split_p95_disagreement_m'], .003)
        for call, heldout in zip(fits.call_args_list, (4, 0)):
            self.assertEqual(call.kwargs['robust_loss'], 'cauchy')
            self.assertEqual(call.kwargs['max_sampson_px'], 2.)
            for i, j, a, b in call.args[1]:
                self.assertNotIn(heldout, (i, j))
                for view, points in ((i, a), (j, b)):
                    test_pixels = np.array([t[view] for t in validation.values()])
                    self.assertTrue(np.all(np.linalg.norm(points[:, None] - test_pixels[None, :], axis=2) > 1e-6))

    def test_moving_heldout_target_cannot_be_fit_away(self):
        train, validation = self.split()
        for track in validation.values():
            track[4] = track[4] + [0, 20]
        result = self.run_prior(train, validation)
        self.assertEqual(result['status'], 'insufficient', result)
        self.assertIsNone(result['calibration'])

    def test_requires_five_distinct_pose_families(self):
        for frames in (self.frames[:4], self.frames):
            frames[-1]['joints'] = dict(frames[0]['joints'])
            with patch.object(tracking, 'multi_anchor_tracks') as tracks:
                result = prior.refine_active_camera_prior(frames, self.model,
                    roi=self.roi, roi_frame_id='0', mode='coarse')
            self.assertEqual(result['status'], 'insufficient')
            tracks.assert_not_called()

    def test_cancel_after_last_validation_does_not_publish(self):
        from server import active_depth
        train, validation = self.split()
        event = threading.Event()
        original = active_depth._solve_track
        count = [0]
        def reconstruct(*args, **kwargs):
            point = original(*args, **kwargs)
            count[0] += 1
            if count[0] == len(validation) * 2:
                event.set()
            return point
        with patch.object(tracking, 'multi_anchor_tracks', return_value=(train, validation, {})), \
             patch.object(active_depth, '_solve_track', side_effect=reconstruct):
            result = prior.refine_active_camera_prior(self.frames, self.model,
                roi=self.roi, roi_frame_id='0', mode='coarse', cancel=event)
        self.assertEqual(count[0], len(validation) * 2)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['calibration'])

    def test_image_consensus_rejects_wrong_matches_without_target_leakage(self):
        train, validation = self.split()
        # Gross descriptor mistakes are independent of a camera fit.
        wrong_training = {i: p.copy() for i, p in train[0].items()}
        wrong_training[2] += [100, 70]
        wrong_test = {i: p.copy() for i, p in validation[0].items()}
        wrong_test[3] += [80, -90]
        validation[999] = wrong_test
        with patch.object(cv2, 'findFundamentalMat', wraps=cv2.findFundamentalMat) as fundamentals:
            clean_train, clean_test, metadata = tracking.image_consensus(train + [wrong_training], validation, 5, 640)
        self.assertFalse(any(t is wrong_training for t in clean_train))
        self.assertNotIn(999, clean_test)
        self.assertGreaterEqual(len(clean_test), 12)
        self.assertFalse(metadata['image_geometry_trained_on_heldout_identities'])
        for call, (i, j) in zip(fundamentals.call_args_list, __import__('itertools').combinations(range(5), 2)):
            for view, points in ((i, call.args[0]), (j, call.args[1])):
                test_pixels = np.array([t[view] for t in validation.values()])
                self.assertTrue(np.all(np.linalg.norm(points[:, None] - test_pixels[None, :], axis=2) > 1e-6))

    def test_image_consensus_is_resize_invariant(self):
        train, validation = self.split()
        validation[999] = {i: p + ([60, -40] if i == 2 else [0, 0]) for i, p in validation[0].items()}
        cv2.setRNGSeed(21)
        small_train, small_test, small_metadata = tracking.image_consensus(train, validation, 5, 640)
        cv2.setRNGSeed(21)
        large_train, large_test, large_metadata = tracking.image_consensus(
            [{i: p * 3 for i, p in t.items()} for t in train],
            {key: {i: p * 3 for i, p in t.items()} for key, t in validation.items()}, 5, 1920)
        self.assertEqual(set(small_test), set(large_test))
        self.assertEqual(len(small_train), len(large_train))
        self.assertAlmostEqual(large_metadata['image_geometry_threshold_px'], small_metadata['image_geometry_threshold_px'] * 3)

    def test_coarse_fit_resize_invariance_and_residual_uncertainty(self):
        train, validation = self.split()
        small = self.run_prior(train, validation)
        self.assertEqual(small['status'], 'refined', small)
        for frame in self.frames:
            image = cv2.imread(frame['path'], cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(frame['path'], cv2.resize(image, (1920, 1440)))
        large = self.run_prior([{i: (p + .5) * 3 - .5 for i, p in t.items()} for t in train],
            {key: {i: (p + .5) * 3 - .5 for i, p in t.items()} for key, t in validation.items()})
        self.assertEqual(large['status'], 'refined', large)
        self.assertAlmostEqual(large['calibration']['camera_matrix'][0][0] / 3,
                               small['calibration']['camera_matrix'][0][0], delta=.01)
        self.assertLess(np.linalg.norm(np.array(large['calibration']['handeye'])[:3, 3] -
                                      np.array(small['calibration']['handeye'])[:3, 3]), .002)
        self.assertEqual(large['calibration']['residual_limit_px'], 6.)
        for result in (small, large):
            self.assertGreaterEqual(result['calibration']['relative_rotation_uncertainty_deg'],
                                    result['calibration']['heldout_angular_residual_deg'])

    def test_multi_anchor_retains_partial_views_and_rejects_camera_fixed_jaw(self):
        rng = np.random.default_rng(48)
        descriptors = rng.normal(size=(33, 128)).astype(np.float32)
        features = []
        for view in range(5):
            keys, desc = [], []
            for identity in range(33):
                if identity < 6 and view >= 3:
                    continue  # No latest-frame anchor for these valid identities.
                x = 360. + (identity % 8) * 35 + view * 9
                y = 140. + (identity // 8) * 50 + view * 5
                if identity == 31:
                    x, y = 600. + view * 9, 410. + view * 5
                if identity == 32:
                    x, y = 450., 430.  # Finger fixed to the camera.
                keys.append(cv2.KeyPoint(x, y, 2))
                desc.append(descriptors[identity])
            features.append((keys, np.array(desc)))
        images = [np.zeros((540, 960), np.uint8) for _ in range(5)]
        box = (.3, .1, .5, .85)
        detector = type('Detector', (), {})()
        detector.detectAndCompute = unittest.mock.Mock(side_effect=features)
        def unfiltered(train, validation, *_args):
            return train, validation, {}
        with patch.object(cv2, 'SIFT_create', return_value=detector), \
             patch.object(tracking, 'image_consensus', side_effect=unfiltered):
            training, validation, metadata = tracking.multi_anchor_tracks(images, box, 0, stable_jaw=True)
        all_tracks = training + list(validation.values())
        self.assertEqual(metadata['camera_fixed_identities_rejected'], 1)
        self.assertEqual(len(all_tracks), 32)
        self.assertTrue(any(4 not in t and len(t) == 3 for t in all_tracks))
        self.assertTrue(any(t[0][1] == 410 for t in all_tracks))
        self.assertFalse(any(np.array_equal(t[0], [450., 430.]) for t in all_tracks))

    def test_detector_uses_pixel_center_mapping_and_deadline(self):
        event = threading.Event()
        event.set()
        images = [np.zeros((1080, 1920), np.uint8)] * 5
        with patch.object(cv2, 'SIFT_create') as detector:
            with self.assertRaises(RuntimeError):
                tracking.multi_anchor_tracks(images, (.2, .1, .6, .8), 0, stable_jaw=True, cancel=event)
            detector.return_value.detectAndCompute.assert_not_called()
        # Reuse one moving identity; inspect its coordinates before consensus.
        detector = type('Detector', (), {})()
        detector.detectAndCompute = unittest.mock.Mock(side_effect=[
            ([cv2.KeyPoint(300. + i * 5, 200. + i * 3, 2), cv2.KeyPoint(500., 300., 2)],
             np.array([[1., 0.], [0., 1.]], np.float32)) for i in range(5)])
        with patch.object(cv2, 'SIFT_create', return_value=detector), \
             patch.object(tracking, 'image_consensus', side_effect=lambda a, b, *_: (a, b, {})):
            train, validation, metadata = tracking.multi_anchor_tracks(images, (.2, .1, .6, .8), 0, stable_jaw=True)
        self.assertEqual(metadata['feature_detection_width'], 960)
        self.assertEqual(len(train), 1)
        np.testing.assert_allclose(train[0][0], [600.5, 400.5])
        self.assertEqual(len(validation), 0)

    def test_common_mount_shift_is_not_false_baseline_error(self):
        poses = [transform(), transform(.03, 0, 0), transform(0, .02, 0)]
        first, second = transform(z=.03), transform(.02, -.01, .03)
        result = prior._camera_uncertainty(poses, first, second, .025, .35)
        self.assertEqual(result['relative_translation_uncertainty_m'], .002)
        self.assertAlmostEqual(result['handeye_translation_disagreement_m'], np.sqrt(.02 ** 2 + .01 ** 2))
        self.assertEqual(result['base_position_uncertainty_m'], .025)
        self.assertEqual(result['relative_rotation_uncertainty_deg'], .35)
        # Rotation makes that same uncertain mounting lever arm affect baseline.
        poses.append(transform(.01, .01, 0, (0, .4, .2)))
        rotating = prior._camera_uncertainty(poses, first, second, .025)
        self.assertGreater(rotating['relative_translation_uncertainty_m'], .007)

    def test_expired_coarse_computation_never_starts_fit(self):
        train, validation = self.split()
        clock = [0.]
        def extraction(*_args, **_kwargs):
            clock[0] = 13.
            return train, validation, {}
        with patch.object(tracking, 'multi_anchor_tracks', side_effect=extraction), \
             patch.object(prior, 'time', SimpleNamespace(monotonic=lambda: clock[0])), \
             patch.object(prior, 'fit_focal_and_handeye') as fit:
            result = prior.refine_active_camera_prior(self.frames, self.model,
                roi=self.roi, roi_frame_id='0', mode='coarse')
        fit.assert_not_called()
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['calibration'])

    def test_prepared_observations_are_bound_to_exact_frames_and_model(self):
        images = [np.zeros((480, 640), np.uint8)] * 5
        transforms = [self.model.fk.transforms(frame['joints']) for frame in self.frames]
        features = [([], None)] * 5
        for frames, model in ((list(self.frames), self.model), (self.frames, object())):
            with patch.object(tracking, 'multi_anchor_tracks') as tracks:
                result = prior.refine_active_camera_prior(self.frames, self.model,
                    roi=self.roi, roi_frame_id='0', mode='coarse',
                    _prepared_observations=(frames, model, images, transforms, features))
            self.assertEqual(result['status'], 'insufficient')
            self.assertIn('這一批腕部觀察', result['reason'])
            tracks.assert_not_called()
        from server import active_depth
        train, validation = self.split()
        with patch.object(active_depth, '_read_frames', side_effect=AssertionError('duplicate read')), \
             patch.object(tracking, 'multi_anchor_tracks', return_value=(train, validation, {})) as tracks, \
             patch.object(prior, 'fit_focal_and_handeye', side_effect=ValueError('fit reached')):
            result = prior.refine_active_camera_prior(self.frames, self.model,
                roi=self.roi, roi_frame_id='0', mode='coarse',
                _prepared_observations=(self.frames, self.model, images, transforms, features))
        self.assertIn('fit reached', result['reason'])
        self.assertIs(tracks.call_args.kwargs['features'], features)

    def test_prepared_features_skip_duplicate_detector(self):
        images = [np.zeros((480, 640), np.uint8)] * 5
        with patch.object(cv2, 'SIFT_create') as detector:
            train, validation, metadata = tracking.multi_anchor_tracks(images,
                (.2, .1, .6, .8), 0, stable_jaw=True, features=[([], None)] * 5)
        detector.assert_not_called()
        self.assertEqual(train, [])
        self.assertEqual(validation, {})
        self.assertTrue(metadata['reused_same_call_features'])

    def test_identity_partition_is_invariant_to_pixel_origin_translation(self):
        target = [{j: np.array([39.8 + (i % 3) * .3, 19.8 + i * .12]) for j in range(3)} for i in range(12)]
        original_training, original_test = tracking._partition_target_tracks(target, 0)
        shifted = [{j: (p + .5) * 2 - .5 for j, p in t.items()} for t in target]
        training, testing = tracking._partition_target_tracks(shifted, 0)
        original_train_ids = [next(i for i, t in enumerate(target) if t is item) for item in original_training]
        changed_train_ids = [next(i for i, t in enumerate(shifted) if t is item) for item in training]
        self.assertEqual(original_train_ids, changed_train_ids)
        self.assertEqual(list(original_test), list(testing))


if __name__ == '__main__':
    unittest.main()
