"""Preparation feasibility uses one actual image for both nominal path legs."""
import unittest
from unittest.mock import patch

import numpy as np

import test_observation_translation_recovery as recovery
from server.active_observation import (_framing_context, _framing_prediction,
                                      _translation_recovery, filter_observation_framing)
from server.local_cartesian import ALL


class TranslationLookaheadFramingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recovery.ObservationTranslationRecoveryTests.setUpClass()

    def setUp(self):
        self.fixture = recovery.ObservationTranslationRecoveryTests()
        self.fixture.setUp()
        self.frame = self.fixture.frame()

    def prepare(self, bbox=None):
        context = _framing_context(bbox, [1920, 1080], self.frame)
        return _translation_recovery(self.fixture.grasp, self.frame, self.fixture.origin, None,
            directions=['up', 'backward', 'left', 'right'],
            calibrated_joint_bounds=self.fixture.bounds, framing=context)

    def test_visible_prefix_cannot_claim_a_future_translation_that_crops_target(self):
        # This exercises the explicitly uncalibrated fallback. Calibrated
        # callers may correctly make a different visibility prediction.
        prefix = self.prepare()
        self.assertIsNotNone(prefix)
        context = _framing_context(dict(x=0., y=0., w=.227, h=.871), [1920, 1080], self.frame)
        prediction = _framing_prediction(prefix, context, None)
        self.assertGreaterEqual(prediction['minimum_predicted_visible_fraction'], .95)
        self.assertIsNone(self.prepare(context['target_bbox']))

    def test_future_projection_uses_actual_before_preparation_camera_not_synthetic_image(self):
        with patch('server.active_observation._framing_prediction', wraps=_framing_prediction) as predict:
            prefix = self.prepare(dict(x=.4, y=.35, w=.1, h=.3))
        self.assertIsNotNone(prefix)
        witness = prefix['descriptor']['translation_lookahead']
        prediction = witness['target_framing_model']
        self.assertEqual(prediction['reference_frame_id'], self.frame['id'])
        self.assertEqual(prediction['reference_path'], self.frame['path'])
        self.assertEqual(prediction['reference_joints'], self.frame['joints'])
        self.assertEqual(witness['framing_reference'], 'actual_current_image_before_preparation')
        self.assertFalse(witness['execution_authorized'])
        self.assertFalse(witness['automatic_chaining'])
        self.assertTrue(witness['requires_actual_capture_and_replanning'])
        future_calls = [call for call in predict.call_args_list
                        if call.args[0]['framing_projection_path'][0]['fraction'] == 0.]
        self.assertTrue(future_calls)
        path = future_calls[-1].args[0]['framing_projection_path']
        self.assertEqual([s['fraction'] for s in path], [0., .25, .5, .75, 1.])
        before = self.fixture.model.fk.transforms(self.frame['joints'])['wrist_camera_optical']
        prepared = prefix['segment']['joint_targets']
        future = witness['nominal_target_joints']
        for sample in path:
            fraction = sample['fraction']
            pose = {n: prepared[n] + fraction * (future[n] - prepared[n]) for n in ALL}
            nominal = self.fixture.model.fk.transforms(pose)['wrist_camera_optical']
            np.testing.assert_allclose(sample['reference_to_candidate_camera_transform'],
                                       np.linalg.inv(nominal) @ before)
        # At the start of the SECOND leg, the camera already differs from the
        # actual image. Reusing its ROI as if newly photographed would give I.
        self.assertFalse(np.allclose(path[0]['reference_to_candidate_camera_transform'], np.eye(4)))
        self.assertFalse(prediction['target_depth_estimated'])
        self.assertFalse(prediction['collision_clearance_verified'])

    def test_unlocalized_witness_does_not_claim_target_framing_was_checked(self):
        prefix = self.prepare()
        witness = prefix['descriptor']['translation_lookahead']
        self.assertIsNone(witness['target_framing_model'])
        self.assertIsNone(witness['framing_reference'])

    def test_bbox_returned_after_planning_also_checks_future_witness(self):
        prefix = self.prepare()
        witness = prefix['descriptor']['translation_lookahead']
        self.assertTrue(witness['framing_projection_path'])
        self.assertIsNone(witness['target_framing_model'])
        # Ordinary object-program starts with no bbox, then applies the one
        # returned by Astra to its already screened candidates. This must not
        # check only the preparation and silently omit the future leg.
        filtered = filter_observation_framing([prefix],
            dict(x=0., y=0., w=.227, h=.871), [1920, 1080], reference_frame=self.frame)
        self.assertEqual(filtered, [])
        self.assertIsNone(witness['target_framing_model'])  # Input remains immutable.

    def test_later_centered_bbox_preserves_path_and_binds_future_prediction(self):
        prefix = self.prepare()
        filtered = filter_observation_framing([prefix],
            dict(x=.4, y=.35, w=.1, h=.3), [1920, 1080], reference_frame=self.frame)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['segment'], prefix['segment'])
        witness = filtered[0]['descriptor']['translation_lookahead']
        self.assertEqual(witness['target_framing_model']['reference_frame_id'], self.frame['id'])
        self.assertFalse(witness['execution_authorized'])
        self.assertEqual(witness['framing_projection_path'],
                         prefix['descriptor']['translation_lookahead']['framing_projection_path'])


if __name__ == '__main__':
    unittest.main()
