"""Bounded SIFT detection keeps original-pixel geometry and exclusion support."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from server import active_depth as depth


class DepthDetectionResolutionTests(unittest.TestCase):
    def test_native_matches_recover_original_pixel_motion_and_metric_depth(self):
        # A textured, fronto-parallel plane with known image motion. Native
        # SIFT + bidirectional matching + three-view cycles remain exercised.
        rng = np.random.default_rng(714)
        texture = rng.integers(0, 256, (540, 960), dtype=np.uint8)
        image = cv2.resize(texture, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        shifts = (0., 24., 48.)
        images = [cv2.warpAffine(image, np.array([[1., 0, dx], [0, 1., 0]]),
                                (1920, 1080), borderValue=127) for dx in shifts]
        tracks, metadata = depth._image_tracks(images, (.15, .08, .55, .35), None, roi_index=0)
        triples = [track for track in tracks if len(track) == 3]
        self.assertGreater(len(triples), 20)
        motion = np.median([track[2] - track[0] for track in triples], axis=0)
        np.testing.assert_allclose(motion, [48., 0.], atol=.15)
        self.assertEqual(metadata['feature_detection_sizes'], [[960, 540]] * 3)
        self.assertEqual(metadata['feature_detection_width'], 960)
        self.assertTrue(all(.15 * 1920 <= t[0][0] <= .7 * 1920 for t in triples))
        k = np.array([[1440., 0, 959.5], [0, 1440., 539.5], [0, 0, 1.]])
        cameras = []
        for dx in shifts:
            camera = np.eye(4)
            camera[0, 3] = -dx * .4 / 1440.
            cameras.append(camera)
        z = []
        for track in triples[:40]:
            solved, reason = depth._solve_track(track, cameras, k, np.zeros(5), (1920, 1080))
            self.assertIsNone(reason)
            z.append(solved['point'][2])
        self.assertAlmostEqual(float(np.median(z)), .4, delta=.002)

    def test_rounded_resize_uses_pixel_centers_and_original_support_mask(self):
        width, height = 1931, 1083
        small_width, small_height = 960, round(height * 960 / width)
        sx, sy = small_width / width, small_height / height
        detector_calls = []
        class Detector:
            def detectAndCompute(self, image, mask):
                detector_calls.append((image.copy(), mask.copy()))
                # Third center is above the mask, but its descriptor reaches
                # the excluded gripper band after scaling back to raw pixels.
                keys = [cv2.KeyPoint(100.25, 100.75, 2., response=3.),
                        cv2.KeyPoint(200.25, 200.75, 2., response=2.),
                        cv2.KeyPoint(300.25, 300.75, 8., response=1.)]
                return keys, np.eye(3, 128, dtype=np.float32)
        with patch.object(depth.cv2, 'SIFT_create', return_value=Detector()):
            tracks, metadata = depth._image_tracks(
                [np.zeros((height, width), np.uint8)] * 2, None, None)
        self.assertEqual(metadata['features_per_view'], [2, 2])
        self.assertEqual(len(tracks), 2)
        expected = [(100.25 + .5) / sx - .5, (100.75 + .5) / sy - .5]
        np.testing.assert_allclose(tracks[0][0], expected, atol=2e-5)
        for image, mask in detector_calls:
            self.assertEqual(image.shape, (small_height, small_width))
            self.assertTrue(np.all(image[-1] == 127))
            self.assertTrue(np.all(mask[-1] == 0))
        self.assertEqual(metadata['feature_detection_sizes'], [[small_width, small_height]] * 2)

    def test_smaller_images_keep_native_pixel_coordinates(self):
        class Detector:
            def detectAndCompute(self, image, mask):
                return [cv2.KeyPoint(100.25, 60.75, 2.), cv2.KeyPoint(200.25, 100.75, 2.)], \
                    np.eye(2, 128, dtype=np.float32)
        with patch.object(depth.cv2, 'SIFT_create', return_value=Detector()):
            tracks, metadata = depth._image_tracks([np.zeros((360, 640), np.uint8)] * 2, None, None)
        self.assertEqual(metadata['feature_detection_sizes'], [[640, 360]] * 2)
        np.testing.assert_array_equal(tracks[0][0], [100.25, 60.75])

    def test_native_feature_cache_preserves_descriptors_coordinates_and_readonly_sources(self):
        image = np.full((720, 1280), 127, np.uint8)
        for row in range(4):
            for column in range(5):
                cv2.putText(image, f'{row}{column}', (110 + column * 190, 100 + row * 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1., 235, 2, cv2.LINE_AA)
        shifted = cv2.warpAffine(image, np.array([[1., 0, 12.], [0, 1., 0]]),
                                 (1280, 720), borderValue=127)
        images = [image, shifted]
        originals = [source.copy() for source in images]
        for source in images:
            source.setflags(write=False)
        native = cv2.SIFT_create(nfeatures=2000, contrastThreshold=.012)
        detections = []
        class Detector:
            def detectAndCompute(self, detection_image, mask):
                keys, descriptors = native.detectAndCompute(detection_image, mask)
                detections.append((tuple((key.pt, key.size, key.angle) for key in keys),
                                   descriptors.copy(), detection_image.shape))
                return keys, descriptors
        cache = {}
        with patch.object(depth.cv2, 'SIFT_create', return_value=Detector()):
            tracks, metadata = depth._image_tracks(images, None, None, _feature_cache=cache)
        self.assertGreater(len(tracks), 6)
        self.assertEqual(len(cache['features']), 2)
        for index, (keys, descriptors) in enumerate(cache['features']):
            original_keys, original_descriptors, detection_shape = detections[index]
            sx, sy = detection_shape[1] / 1280, detection_shape[0] / 720
            self.assertGreater(len(keys), 6)
            self.assertEqual(len(keys), len(descriptors))
            self.assertEqual(metadata['features_per_view'][index], len(keys))
            for key, descriptor in zip(keys, descriptors):
                # Reverse the pixel-center mapping and locate the actual
                # native keypoint, then compare its descriptor row exactly.
                detected_pixel = (np.array(key.pt) + .5) * [sx, sy] - .5
                matches = [i for i, (pixel, size, angle) in enumerate(original_keys)
                           if np.linalg.norm(detected_pixel - pixel) < .0001
                           and abs(key.size * min(sx, sy) - size) < .0001
                           and abs(key.angle - angle) < .0001]
                self.assertEqual(len(matches), 1)
                np.testing.assert_array_equal(descriptor, original_descriptors[matches[0]])
            self.assertFalse(images[index].flags.writeable)
            np.testing.assert_array_equal(images[index], originals[index])

    def test_coarse_consumer_reuses_cache_without_changing_images_keys_or_descriptors(self):
        from server import active_prior_tracks as prior
        images = [np.zeros((480, 640), np.uint8) for _ in range(3)]
        features = []
        for view in range(3):
            keys = [cv2.KeyPoint(150. + identity * 30 + view * 8, 80. + identity * 20, 2.)
                    for identity in range(4)]
            descriptors = np.eye(4, 128, dtype=np.float32)
            descriptors.setflags(write=False)
            images[view].setflags(write=False)
            features.append((keys, descriptors))
        saved = [(tuple((key.pt, key.size, key.angle) for key in keys), descriptors.copy())
                 for keys, descriptors in features]
        def identity_matches(first, second):
            return {i: i for i in range(len(first))}
        def pass_image_consensus(training, validation, *args, **kwargs):
            return training, validation, {}
        with patch.object(depth, '_mutual_matches', side_effect=identity_matches), \
             patch.object(prior, 'image_consensus', side_effect=pass_image_consensus), \
             patch.object(prior.cv2, 'SIFT_create', side_effect=AssertionError('Cached features must not redetect')):
            training, validation, metadata = prior.multi_anchor_tracks(
                images, (.1, .1, .6, .4), 0, stable_jaw=True, features=features)
        self.assertTrue(metadata['reused_same_call_features'])
        self.assertEqual(len(training), 2)
        self.assertEqual(len(validation), 2)
        for index, (keys, descriptors) in enumerate(features):
            self.assertEqual(tuple((key.pt, key.size, key.angle) for key in keys), saved[index][0])
            np.testing.assert_array_equal(descriptors, saved[index][1])
            self.assertFalse(descriptors.flags.writeable)
            self.assertFalse(images[index].flags.writeable)
            self.assertFalse(images[index].any())


if __name__ == '__main__':
    unittest.main()
