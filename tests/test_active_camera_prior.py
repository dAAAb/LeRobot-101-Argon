"""Same-attempt camera refinement is opt-in and independently checked."""
import json
import tempfile
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from server import active_camera_prior as prior
from server import active_depth as depth
from server.local_cartesian import NAMES
from server.selfcal import intrinsics
from test_active_depth import transform


class ActiveCameraPriorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.k = intrinsics(520, (640, 480))
        self.true_x = transform(.006, .014, .025, (.08, -.02, .04))
        self.cad_x = transform(.006, .020, .013, (.10, .025, .04))
        self.g = [transform(), transform(.008, 0, .005, (.10, 0, 0)),
                  transform(.012, -.004, .006, (0, 0, .12)),
                  transform(-.025, .012, .006, (0, .09, 0)),
                  transform(.025, -.014, .018, (-.04, -.07, .03))]
        self.frames = []
        for i in range(5):
            path = Path(self.directory.name) / f'{i}.png'
            cv2.imwrite(str(path), np.full((480, 640), 127, np.uint8))
            self.frames.append({'id': str(i), 'path': str(path), 'source': 'wrist',
                'camera_generation': 'same-attempt', 'sequence': i + 1, 'capture_monotonic': float(i),
                'joints': {**{n: 2000 for n in NAMES}, 'shoulder_pan': 2000 + i * 100, 'gripper': 1800}})
        def poses(raw):
            g = self.g[(raw['shoulder_pan'] - 2000) // 100]
            return {'gripper_link': g, 'wrist_camera_optical': g @ self.cad_x,
                    'gripper_frame_link': g @ transform(z=.05)}
        self.model = SimpleNamespace(fk=SimpleNamespace(transforms=poses))
        self.roi = {'x': .4, 'y': .18, 'w': .2, 'h': .22}
        self.rng = np.random.default_rng(127)

    def tracks(self, count=4, *, moving_target=False):
        cameras = [g @ self.true_x for g in self.g[:count]]
        tracks = []
        for is_target, amount in ((False, 100), (True, 30)):
            for n in range(amount):
                u = self.rng.uniform(270, 365) if is_target else self.rng.uniform(80, 210) if n % 2 else self.rng.uniform(430, 550)
                v = self.rng.uniform(95, 180)
                z = self.rng.uniform(.35, .65)
                camera_point = np.linalg.inv(self.k) @ [u, v, 1.] * z
                point = (cameras[0] @ np.r_[camera_point, 1.])[:3]
                track = {}
                for i, camera in enumerate(cameras):
                    pixel = depth._project(point, camera, self.k, np.zeros(5))
                    track[i] = pixel + self.rng.normal(0, .015, 2)
                    if moving_target and is_target and i == count - 1:
                        track[i] += [5, 25]
                tracks.append(track)
        return tracks

    def refine(self, count=4, *, moving_target=False):
        tracks = self.tracks(count, moving_target=moving_target)
        with patch.object(depth, '_image_tracks', return_value=(tracks, {})):
            return prior.refine_active_camera_prior(self.frames[:count], self.model,
                roi=self.roi, roi_frame_id='0')

    def test_real_nonlinear_fit_recovers_prior_using_heldout_pair_and_target(self):
        result = self.refine()
        self.assertEqual(result['status'], 'refined', result)
        self.assertFalse(result['metric_calibrated'])
        self.assertFalse(result['validation']['target_used_for_fit'])
        self.assertFalse(result['validation']['unseen_pose_validation'])
        self.assertGreaterEqual(result['validation']['shared_target_identities'], 6)
        self.assertLess(result['validation']['target_split_p95_disagreement_m'], .003)
        self.assertAlmostEqual(result['calibration']['camera_matrix'][0][0], 520, delta=8)
        self.assertLess(np.linalg.norm(np.array(result['calibration']['handeye'])[:3, 3] - self.true_x[:3, 3]), .004)
        for split in result['validation']['splits']:
            self.assertEqual(split['holdout_kind'], 'whole_pair')
            self.assertLess(split['holdout_p95_sampson_px'], .5)
            self.assertEqual(split['fit']['scaled_jacobian_rank'], 7)
        json.dumps(result, allow_nan=False)

    def test_target_feature_identities_are_absent_from_every_training_pair(self):
        tracks = self.tracks()
        with patch.object(depth, '_image_tracks', return_value=(tracks, {})), \
             patch.object(prior, 'fit_focal_and_handeye', wraps=prior.fit_focal_and_handeye) as fit:
            result = prior.refine_active_camera_prior(self.frames[:4], self.model,
                roi=self.roi, roi_frame_id='0')
        self.assertEqual(result['status'], 'refined', result)
        self.assertEqual(fit.call_count, 2)
        for call in fit.call_args_list:
            for i, j, points_i, points_j in call.args[1]:
                for view, points in ((i, points_i), (j, points_j)):
                    target_pixels = np.array([track[view] for track in tracks[100:]])
                    self.assertTrue(np.all(np.linalg.norm(points[:, None, :] - target_pixels[None, :, :], axis=2) > 1e-6))

    def test_camera_fixed_background_identities_are_excluded_from_calibration_fit(self):
        tracks = self.tracks()
        stationary = [{i: np.array([100. + n * 2, 70. + i]) for i in range(4)} for n in range(12)]
        with patch.object(depth, '_image_tracks', return_value=(tracks + stationary, {})), \
             patch.object(prior, 'fit_focal_and_handeye', wraps=prior.fit_focal_and_handeye) as fit:
            result = prior.refine_active_camera_prior(self.frames[:4], self.model,
                roi=self.roi, roi_frame_id='0')
        self.assertEqual(result['status'], 'refined', result)
        self.assertEqual(result['tracking']['camera_fixed_background_identities_rejected'], len(stationary))
        for call in fit.call_args_list:
            for i, j, points_i, points_j in call.args[1]:
                for view, points in ((i, points_i), (j, points_j)):
                    fixed_pixels = np.array([track[view] for track in stationary])
                    self.assertTrue(np.all(np.linalg.norm(points[:, None, :] - fixed_pixels[None, :, :], axis=2) > 1e-6))

    def test_return_visits_are_not_independent_pose_families(self):
        self.frames[3]['joints'] = dict(self.frames[0]['joints'])
        with patch.object(prior, 'fit_focal_and_handeye') as fit:
            result = prior.refine_active_camera_prior(self.frames[:4], self.model,
                roi=self.roi, roi_frame_id='0')
        self.assertEqual(result['status'], 'insufficient')
        self.assertIn('同一姿勢', result['reason'])
        fit.assert_not_called()

    def test_five_views_withhold_entire_pose_families(self):
        result = self.refine(count=5)
        self.assertEqual(result['status'], 'refined', result)
        self.assertTrue(result['validation']['unseen_pose_validation'])
        self.assertTrue(all(s['holdout_kind'] == 'whole_pose' for s in result['validation']['splits']))

    def test_moving_target_cannot_train_away_its_own_inconsistency(self):
        result = self.refine(moving_target=True)
        self.assertEqual(result['status'], 'insufficient', result)
        self.assertIn('物件特徵', result['reason'])
        self.assertIsNone(result['calibration'])
        self.assertFalse(result['tracking']['target_used_for_fit'])

    def test_too_few_views_and_missing_roi_never_start_fit(self):
        for frames, roi in ((self.frames[:3], self.roi), (self.frames[:4], None)):
            with patch.object(prior, 'fit_focal_and_handeye') as fit:
                result = prior.refine_active_camera_prior(frames, self.model, roi=roi)
                self.assertEqual(result['status'], 'insufficient')
                fit.assert_not_called()

    def test_cancellation_and_deadline_do_not_return_a_calibration(self):
        event = threading.Event(); event.set()
        result = prior.refine_active_camera_prior(self.frames[:4], self.model,
            roi=self.roi, cancel=event)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['calibration'])

    def test_cancel_during_target_reconstruction_stops_before_next_identity(self):
        event = threading.Event()
        original = depth._solve_track
        def solve(*args, **kwargs):
            result = original(*args, **kwargs)
            event.set()
            return result
        with patch.object(depth, '_image_tracks', return_value=(self.tracks(), {})), \
             patch.object(depth, '_solve_track', side_effect=solve) as reconstruction:
            result = prior.refine_active_camera_prior(self.frames[:4], self.model,
                roi=self.roi, roi_frame_id='0', cancel=event)
        self.assertEqual(reconstruction.call_count, 1)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['calibration'])

    def test_cancellation_or_expiry_after_last_validation_cannot_publish_refined(self):
        for expired in (False, True):
            with self.subTest(expired=expired):
                event = threading.Event()
                clock = [0.]
                original = depth._solve_track
                calls = [0]
                def solve(*args, **kwargs):
                    result = original(*args, **kwargs)
                    calls[0] += 1
                    if calls[0] == 60:  # Final target identity of the second holdout.
                        if expired:
                            clock[0] = 13.
                        else:
                            event.set()
                    return result
                with patch.object(depth, '_image_tracks', return_value=(self.tracks(), {})), \
                     patch.object(depth, '_solve_track', side_effect=solve), \
                     patch.object(prior, 'time', SimpleNamespace(monotonic=lambda: clock[0])):
                    result = prior.refine_active_camera_prior(self.frames[:4], self.model,
                        roi=self.roi, roi_frame_id='0', cancel=event)
                self.assertEqual(calls[0], 60)
                self.assertEqual(result['status'], 'insufficient')
                self.assertIsNone(result['calibration'])
                self.assertIn('已停止或超過', result['reason'])
        with patch.object(depth, '_image_tracks', return_value=(self.tracks(), {})):
            result = prior.refine_active_camera_prior(self.frames[:4], self.model,
                roi=self.roi, roi_frame_id='0', max_seconds=.00001)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['calibration'])

    def test_unobservable_fit_never_promotes_a_camera_prior(self):
        with patch.object(depth, '_image_tracks', return_value=(self.tracks(), {})), \
             patch.object(prior, 'fit_focal_and_handeye', side_effect=ValueError('rank 6/7')):
            result = prior.refine_active_camera_prior(self.frames[:4], self.model,
                roi=self.roi, roi_frame_id='0')
        self.assertEqual(result['status'], 'insufficient')
        self.assertIn('rank 6/7', result['reason'])

    def test_estimator_only_refines_when_explicitly_enabled(self):
        tracks = self.tracks()
        with patch.object(depth, '_image_tracks', return_value=(tracks, {})), \
             patch.object(prior, 'refine_active_camera_prior', wraps=prior.refine_active_camera_prior) as helper:
            default = depth.estimate_active_depth(self.frames[:4], self.model,
                roi=self.roi, roi_frame_id='0')
            self.assertEqual(default['status'], 'insufficient')
            helper.assert_not_called()
            refined = depth.estimate_active_depth(self.frames[:4], self.model,
                roi=self.roi, roi_frame_id='0', refine_camera_prior=True)
        self.assertEqual(refined['status'], 'estimated', refined)
        self.assertEqual(helper.call_count, 1)
        self.assertEqual(refined['camera_prior_refinement']['status'], 'refined')
        self.assertEqual(refined['provenance']['camera_model'], 'same-attempt-refined-prior')
        self.assertFalse(refined['metric_calibrated'])
        self.assertFalse(refined['grasp_ready'])
        self.assertAlmostEqual(refined['distance']['camera_optical_axis_m'], .5, delta=.08)


if __name__ == '__main__':
    unittest.main()
