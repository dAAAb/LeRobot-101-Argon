"""Frozen-camera extension against synthetic board pixels and real local FK.

All images, profiles and session journals are temporary files. No app import,
controller, camera, cloud, or motor access is used, including the CLI test.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from server.board_calibration import BoardSpec, _inverse, _project, _transform
from server.calibration_extension import validate_workspace_extension
from server.geometry import Kinematics
from server.spatial_contract import JOINTS, WristCalibration
from server.spatial_runtime import geometry_revision
from scripts.extend_workspace_calibration import extend_saved_session, read_session

ROOT = Path(__file__).resolve().parents[1]


class CalibrationExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.urdf = ROOT / 'assets/so101_follower_cam.urdf'
        self.motor = {n: {'id': i + 1, 'range_min': 0, 'range_max': 4095,
                         'homing_offset': 0, 'drive_mode': 0} for i, n in enumerate(JOINTS)}
        self.model = SimpleNamespace(urdf_path=self.urdf, calibration=self.motor, fk=Kinematics(self.urdf))
        self.revision = geometry_revision(self.model)
        self.board = BoardSpec(); self.size = [1280, 960]
        self.k = np.array([[900., 0., 639.5], [0., 890., 479.5], [0., 0., 1.]])
        self.distortion = np.array([.02, -.005, .0005, -.0003, 0.])
        self.origin = {n: 2047 for n in JOINTS}
        transforms = self.model.fk.transforms(self.origin)
        self.handeye = _inverse(transforms['gripper_link']) @ transforms['wrist_camera_optical']
        self.base_board = transforms['wrist_camera_optical'] @ _transform(np.zeros(3), [-.08, -.05, .55])
        self.profile = {'schema_version': 1, 'source': 'wrist-only', 'calibration_revision': 'parent-validated',
            'camera_id': 'wrist-A', 'mount_revision': 'mount-A', 'geometry_revision': self.revision,
            'native_image_size': self.size, 'image_size': self.size, 'camera_matrix': self.k.tolist(),
            'distortion': self.distortion.tolist(), 'distortion_model': 'opencv-brown', 'handeye': self.handeye.tolist(),
            'validated_components': ['intrinsics', 'handeye', 'robot_geometry'], 'validation_id': 'old-holdout',
            'validation_kind': 'independent-heldout', 'metric_calibrated': True, 'needs_more_views': False,
            'camera_generation': None, 'board_session_id': 'old-fixed-board',
            'validation_scope': {'joint_raw_bounds': {n: [2040, 2054] for n in JOINTS[:-1]},
                'sampled_joint_poses': [{n: self.origin[n] for n in JOINTS[:-1]}],
                'board_plane_base': np.eye(4).tolist(), 'extrapolation_allowed': False},
            'relative_translation_uncertainty_m': .002, 'base_position_uncertainty_m': .003,
            'relative_rotation_uncertainty_deg': .2, 'focal_relative_uncertainty': .004}
        offsets = [(-35, -20, 15, 25, -10), (40, 10, -20, 0, 20), (-15, 35, 25, -30, 30),
                   (25, -30, -15, 40, -25), (-30, 15, -35, -20, -35), (15, 25, 40, 10, 15)]
        self.frames = []
        rng = np.random.default_rng(951)
        for index, offset in enumerate(offsets):
            joints = {**self.origin, **{n: self.origin[n] + v for n, v in zip(JOINTS[:-1], offset)}}
            camera = self.model.fk.transforms(joints)['gripper_link'] @ self.handeye
            pixels = _project(self.board.object_points(), _inverse(camera) @ self.base_board, self.k, self.distortion)
            pixels += rng.normal(0., .02, pixels.shape)
            self.assertTrue(np.all(pixels >= 0) and np.all(pixels < self.size))
            stamp = index + 2.
            frame = {'id': f'new-frame-{index}', 'source': 'wrist', 'camera_id': 'wrist-A',
                'camera_generation': 'new-generation', 'sequence': index + 1, 'capture_monotonic': stamp,
                'captured_at': 1700000000. + index, 'image_size': self.size, 'joints': joints,
                'board_session_id': 'new-fixed-board', 'joint_capture_span_ticks': 0,
                'image_wait_joint_span_ticks': 0, 'joint_observation_bracket_monotonic': [stamp - .05, stamp + .05],
                'timestamp_basis': 'jpeg-arrival after settled joint window',
                'settled_joint_window': [
                    {'read_start_monotonic': stamp - .7, 'read_end_monotonic': stamp - .6, 'joints': deepcopy(joints)},
                    {'read_start_monotonic': stamp - .2, 'read_end_monotonic': stamp - .1, 'joints': deepcopy(joints)}]}
            frame['board_detection'] = {'found': True, 'corners_px': pixels.tolist(), 'image_size': self.size,
                'frame_id': frame['id'], 'camera_generation': frame['camera_generation'], 'sequence': frame['sequence'],
                'capture_monotonic': stamp, 'board_fingerprint': self.board.fingerprint,
                'board_spec': self.board.to_dict(), 'coverage_fraction': .1}
            self.frames.append(frame)

    def validate(self, frames=None, profile=None, **kwargs):
        return validate_workspace_extension(self.frames if frames is None else frames, self.model,
            self.profile if profile is None else profile, camera_id='wrist-A', mount_revision='mount-A',
            geometry_revision=self.revision, board_session_id='new-fixed-board', **kwargs)

    def test_pass_freezes_camera_and_preserves_parent_scope_provenance_and_uncertainty(self):
        before = deepcopy(self.profile)
        result = self.validate()
        self.assertEqual(result['status'], 'validated', result['blockers'])
        profile = result['calibration_profile']; evidence = result['validation']
        for name in ('camera_matrix', 'distortion', 'handeye', 'focal_relative_uncertainty'):
            self.assertEqual(profile[name], before[name])
        self.assertEqual(self.profile, before)
        self.assertEqual(profile['extension_provenance']['parent_profile'], before)
        self.assertNotEqual(profile['validation_id'], before['validation_id'])
        self.assertEqual(result['validation_policy_version'], 3)
        self.assertEqual(evidence['validation_policy_version'], 3)
        self.assertEqual(evidence['uncertainty_aggregation'],
                         'max-parent-and-current-total-empirical-residual-v1')
        self.assertEqual(profile['extension_provenance']['validation'], evidence)
        self.assertFalse(result['activation_performed'])
        self.assertFalse(evidence['heldout_data_used_for_fitting'])
        self.assertEqual(evidence['fitted_parameters'], ['T_base_board'])
        self.assertEqual(len(evidence['training_frame_ids']), 3)
        self.assertEqual(len(evidence['heldout_frame_ids']), 3)
        for name in ('relative_translation_uncertainty_m', 'base_position_uncertainty_m', 'relative_rotation_uncertainty_deg'):
            self.assertEqual(profile[name], before[name])
        for name, coverage in evidence['measured_axis_coverage'].items():
            self.assertTrue(coverage['extends_previous_bounds'])
            self.assertGreater(coverage['heldout_span_deg'], 1)
            self.assertGreater(coverage['heldout_outside_previous_count'], 0)
            lo, hi = profile['validation_scope']['joint_raw_bounds'][name]
            self.assertLessEqual(lo, min(f['joints'][name] for f in self.frames))
            self.assertGreaterEqual(hi, max(f['joints'][name] for f in self.frames))
        camera = WristCalibration.from_mapping(profile, native_image_size=self.size, camera_id='wrist-A',
            mount_revision='mount-A', geometry_revision=self.revision, cad_handeye=np.eye(4))
        self.assertTrue(camera.metric_calibrated)
        self.assertTrue(all(camera.contains_pose(f['joints']) for f in self.frames))

    def test_heldout_pixels_cannot_change_board_fit_or_camera_parameters(self):
        reference = self.validate()
        altered = deepcopy(self.frames)
        altered[1]['board_detection']['corners_px'] = (np.array(altered[1]['board_detection']['corners_px']) + [2., 0]).tolist()
        result = self.validate(altered)
        self.assertEqual(result['status'], 'validated', result['blockers'])
        self.assertEqual(result['validation']['board_plane_base'], reference['validation']['board_plane_base'])
        for name in ('camera_matrix', 'distortion', 'handeye'):
            self.assertEqual(result['calibration_profile'][name], reference['calibration_profile'][name])
        self.assertGreater(result['validation']['heldout_max_translation_m'], reference['validation']['heldout_max_translation_m'])

    def test_board_motion_or_inconsistent_training_fails_without_a_candidate(self):
        for index in (1, 2):
            with self.subTest(index=index):
                frames = deepcopy(self.frames)
                frames[index]['board_detection']['corners_px'] = (np.array(frames[index]['board_detection']['corners_px']) + [70., 0.]).tolist()
                result = self.validate(frames)
                self.assertEqual(result['status'], 'rejected')
                self.assertIsNone(result['calibration_profile'])

    def test_optical_quality_and_metric_consistency_have_separate_limits(self):
        frames = deepcopy(self.frames)
        for index, dx in zip((0, 2, 4), (-2.5, 0., 2.5)):
            frames[index]['board_detection']['corners_px'] = (np.array(frames[index]['board_detection']['corners_px']) + [dx, 0.]).tolist()
        result = self.validate(frames)
        self.assertEqual(result['status'], 'validated', result['blockers'])
        validation = result['validation']
        self.assertGreater(validation['training_chain_rms_px'], 1.5)
        self.assertLess(validation['training_chain_p95_px'], 6.)
        self.assertTrue(validation['chain_reprojection_is_diagnostic'])
        self.assertEqual(validation['acceptance_limits']['per_view_pnp_reprojection_p95_px'], 6.)
        self.assertEqual(validation['acceptance_limits']['heldout_translation_m'], .015)
        self.assertEqual(validation['acceptance_limits']['heldout_rotation_deg'], 3.)
        self.assertNotIn('training_rms_px', validation['acceptance_limits'])
        self.assertNotIn('training_chain_p95_px', validation['acceptance_limits'])

    def shifted_camera_frames(self, translation=.008, rotation_about_board=0.):
        frames = deepcopy(self.frames)
        camera = self.model.fk.transforms(frames[1]['joints'])['gripper_link'] @ self.handeye
        changed = (camera @ _transform(np.zeros(3), [translation, 0., 0.])
                   if not rotation_about_board else self.base_board @
                   _transform([0., rotation_about_board, 0.], np.zeros(3)) @ _inverse(self.base_board) @ camera)
        frames[1]['board_detection']['corners_px'] = _project(self.board.object_points(),
            _inverse(changed) @ self.base_board, self.k, self.distortion).tolist()
        return frames

    def test_millimetre_pose_error_passes_above_six_chain_pixels_and_increases_uncertainty(self):
        result = self.validate(self.shifted_camera_frames())
        self.assertEqual(result['status'], 'validated', result['blockers'])
        v, profile = result['validation'], result['calibration_profile']
        self.assertGreater(v['heldout_max_reprojection_p95_px'], 6.)
        self.assertLess(v['max_pnp_reprojection_p95_px'], .1)
        self.assertAlmostEqual(v['heldout_max_translation_m'], .008, delta=.0002)
        self.assertGreater(v['heldout_max_camera_center_displacement_m'], .007)
        self.assertGreater(v['max_relative_translation_error_m'], .007)
        self.assertAlmostEqual(profile['base_position_uncertainty_m'], v['sampled_base_displacement_bound_m'])
        self.assertAlmostEqual(profile['relative_translation_uncertainty_m'], v['max_relative_translation_error_m'])
        self.assertEqual(profile['metric_accuracy'], 'approximate')
        self.assertFalse(profile['grasp_ready'])
        self.assertFalse(v['heldout_used_for_shared_parameter_fitting'])
        self.assertTrue(v['per_view_pnp_pose_fitted_for_evaluation'])
        rejected = self.validate(self.shifted_camera_frames(translation=.025))
        self.assertEqual(rejected['status'], 'rejected')
        self.assertGreater(rejected['validation']['heldout_max_translation_m'], .015)
        self.assertIsNone(rejected['calibration_profile'])

    def test_camera_center_lever_arm_and_board_corners_contribute_to_base_uncertainty(self):
        result = self.validate(self.shifted_camera_frames(rotation_about_board=.02))
        self.assertEqual(result['status'], 'validated', result['blockers'])
        v = result['validation']
        self.assertLess(v['heldout_max_translation_m'], .0002)
        self.assertGreater(v['heldout_max_board_corner_displacement_m'], .002)
        self.assertGreater(v['heldout_max_camera_center_displacement_m'], .008)
        self.assertGreaterEqual(v['sampled_base_displacement_bound_m'], v['heldout_max_camera_center_displacement_m'])
        self.assertGreaterEqual(v['sampled_base_displacement_bound_m'], v['heldout_max_board_corner_displacement_m'])
        self.assertAlmostEqual(result['calibration_profile']['base_position_uncertainty_m'],
                               v['sampled_base_displacement_bound_m'])
        rejected = self.validate(self.shifted_camera_frames(rotation_about_board=.08))
        self.assertEqual(rejected['status'], 'rejected')
        self.assertGreater(rejected['validation']['heldout_max_rotation_deg'], 3.)

    def test_total_error_union_takes_larger_observation_without_adding_parent_twice(self):
        result = self.validate(self.shifted_camera_frames(rotation_about_board=.02))
        self.assertEqual(result['status'], 'validated', result['blockers'])
        profile, evidence = result['calibration_profile'], result['validation']
        empirical = {'relative_translation_uncertainty_m': evidence['max_relative_translation_error_m'],
                     'base_position_uncertainty_m': evidence['sampled_base_displacement_bound_m'],
                     'relative_rotation_uncertainty_deg': evidence['max_relative_rotation_error_deg']}
        for name, observed in empirical.items():
            self.assertGreater(observed, self.profile[name])
            self.assertEqual(profile[name], observed)
            self.assertLess(profile[name], self.profile[name]+observed)
        for name in ('camera_matrix', 'distortion', 'handeye', 'focal_relative_uncertainty'):
            self.assertEqual(profile[name], self.profile[name])
        self.assertIn('cross-session pairs are not validated', profile['uncertainty_interpretation'])

    def test_repeated_equivalent_validation_does_not_accumulate_same_error(self):
        first = self.validate(self.shifted_camera_frames())
        frames = deepcopy(self.shifted_camera_frames())
        for frame in frames:
            frame['board_session_id'] = 'another-fixed-board'
            frame['id'] = 'repeat-' + frame['id']
            frame['board_detection']['frame_id'] = frame['id']
        second = validate_workspace_extension(frames, self.model, first['calibration_profile'],
            camera_id='wrist-A', mount_revision='mount-A', geometry_revision=self.revision,
            board_session_id='another-fixed-board')
        self.assertEqual(second['status'], 'validated', second['blockers'])
        for name in ('relative_translation_uncertainty_m', 'base_position_uncertainty_m',
                     'relative_rotation_uncertainty_deg', 'focal_relative_uncertainty'):
            self.assertEqual(second['calibration_profile'][name], first['calibration_profile'][name])

    def test_poor_fixed_lens_projection_still_rejects(self):
        frames = deepcopy(self.frames)
        corners = np.array(frames[1]['board_detection']['corners_px'])
        corners[:, 0] += np.where(np.arange(len(corners)) % 2, -12., 12.)
        frames[1]['board_detection']['corners_px'] = corners.tolist()
        result = self.validate(frames)
        self.assertEqual(result['status'], 'rejected')
        self.assertGreater(result['validation']['max_pnp_reprojection_p95_px'], 6.)
        self.assertIn('逐張棋盤投影品質', result['blockers'][0])
        self.assertIsNone(result['calibration_profile'])

    def test_measured_extension_uncertainty_flows_into_actual_depth_budget(self):
        from server import active_depth
        result = self.validate(self.shifted_camera_frames())
        extended = result['calibration_profile']
        self.assertIsNotNone(extended)
        _profile, session_path, _motor = self.save_inputs()
        frames = json.loads(session_path.read_text())['frames']
        # Exact same optical tracks and geometry for both estimates: only the
        # measured candidate uncertainty changes. No network/features are used.
        tracks = [{i: np.asarray(self.frames[i]['board_detection']['corners_px'][point])
                   for i in range(len(frames))} for point in range(self.board.rows * self.board.columns)]
        estimates = []
        for profile in (self.profile, extended):
            calibration = {**profile, 'intrinsics_validated': True, 'handeye_validated': True,
                           'robot_geometry_validated': True, 'camera_generation': 'new-generation'}
            with patch.object(active_depth, '_image_tracks', return_value=(tracks, {})):
                estimates.append(active_depth.estimate_active_depth(frames, self.model, calibration=calibration))
        for estimate in estimates:
            self.assertEqual(estimate['status'], 'estimated', estimate.get('reason'))
            self.assertFalse(estimate['grasp_ready'])
        before, after = [e['uncertainty'] for e in estimates]
        self.assertEqual(after['assumed_model_bounds']['relative_translation_m'], extended['relative_translation_uncertainty_m'])
        self.assertEqual(after['assumed_model_bounds']['base_position_uncertainty_m'], extended['base_position_uncertainty_m'])
        self.assertGreater(after['depth_error_bound_m'], before['depth_error_bound_m'])
        self.assertGreater(after['base_position_error_bound_m'], before['base_position_error_bound_m'])

    def test_repeated_pose_is_one_family_and_never_leaks_across_the_split(self):
        frames = deepcopy(self.frames)
        duplicate = deepcopy(frames[1]); duplicate.update(id='return-pose', sequence=7, capture_monotonic=8.)
        duplicate['board_detection'].update(frame_id=duplicate['id'], sequence=7, capture_monotonic=8.)
        duplicate['joint_observation_bracket_monotonic'] = [7.95, 8.05]
        for sample in duplicate['settled_joint_window']:
            sample['read_start_monotonic'] += 5.; sample['read_end_monotonic'] += 5.
        frames.append(duplicate)
        result = self.validate(frames, holdout_indices=[3, 5, 6])
        self.assertEqual(result['status'], 'validated', result['blockers'])
        self.assertEqual(result['validation']['independent_pose_groups'], 6)
        self.assertIn(['new-frame-1', 'return-pose'], result['validation']['pose_groups'])
        self.assertNotIn('new-frame-1', result['validation']['training_frame_ids'])
        self.assertEqual(len(result['validation']['heldout_frame_ids']), 3)
        result = self.validate(frames, holdout_indices=[1, 3, 6])
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(result['calibration_profile'])
        self.assertEqual(self.validate(self.frames[:5])['status'], 'insufficient')

    def test_minimum_three_independent_poses_per_split(self):
        for indices in ([1, 3], [0, 1, 2, 3], [1, 1, 3], [True, 3, 5], [1, 3, 99]):
            with self.subTest(indices=indices):
                self.assertIsNone(self.validate(holdout_indices=indices)['calibration_profile'])

    def test_mismatched_camera_mount_geometry_source_timing_or_cache_never_fits(self):
        for field, value in [('source', 'observer'), ('camera_id', 'D435'), ('camera_generation', 'other'),
                             ('mount_revision', 'new-mount'), ('geometry_revision', 'other-geometry'),
                             ('board_session_id', 'moved-board'), ('joint_capture_span_ticks', 12)]:
            with self.subTest(field=field):
                frames = deepcopy(self.frames); frames[2][field] = value
                with patch('server.calibration_extension._fit_board') as fit:
                    result = self.validate(frames)
                self.assertEqual(result['status'], 'rejected')
                self.assertIsNone(result['calibration_profile']); fit.assert_not_called()
        frames = deepcopy(self.frames); frames[1]['board_detection']['sequence'] = 90
        self.assertIsNone(self.validate(frames)['calibration_profile'])

    def test_unvalidated_or_incompatible_parent_cannot_be_upgraded(self):
        for changes in ({'source': 'observer'}, {'metric_calibrated': False}, {'needs_more_views': True},
                        {'validation_kind': 'prior'}, {'validated_components': ['intrinsics', 'handeye']},
                        {'camera_id': 'wrong-camera'}, {'mount_revision': 'wrong-mount'},
                        {'geometry_revision': 'wrong-geometry'}, {'native_image_size': [640, 480]},
                        {'camera_generation': 'old-generation'}, {'base_position_uncertainty_m': None},
                        {'validation_scope': None}):
            with self.subTest(changes=changes):
                result = self.validate(profile={**self.profile, **changes})
                self.assertEqual(result['status'], 'rejected'); self.assertIsNone(result['calibration_profile'])

    def test_original_or_previously_extended_session_cannot_be_reused_as_new_evidence(self):
        for profile in ({**self.profile, 'board_session_id': 'new-fixed-board'},
                        self.validate()['calibration_profile']):
            result = self.validate(profile=profile)
            self.assertEqual(result['status'], 'rejected')
            self.assertIsNone(result['calibration_profile'])

    def save_inputs(self):
        session_dir = self.directory / 'session'; (session_dir / 'frames').mkdir(parents=True)
        frames = deepcopy(self.frames)
        for index, frame in enumerate(frames):
            image = np.full((self.size[1], self.size[0]), 127, np.uint8)
            encoded = cv2.imencode('.jpg', image)[1].tobytes()
            path = session_dir / 'frames' / f'{index}.jpg'; path.write_bytes(encoded)
            frame.update(path=str(path), image_sha256=hashlib.sha256(encoded).hexdigest())
        session = {'session_id': 'new-fixed-board', 'status': 'collecting', 'board_static_declared': True,
            'camera_id': 'wrist-A', 'camera_generation': 'new-generation', 'mount_revision': 'mount-A',
            'geometry_revision': self.revision, 'image_size': self.size, 'frames': frames,
            'board_spec': self.board.to_dict(), 'board_fingerprint': self.board.fingerprint}
        paths = [self.directory / 'profile.json', session_dir / 'session.json', self.directory / 'motors.json']
        for path, value in zip(paths, (self.profile, session, self.motor)):
            path.write_text(json.dumps(value))
        return paths

    def test_real_cli_creates_separate_candidate_and_never_changes_inputs_or_overwrites_outputs(self):
        profile, session, motor = self.save_inputs()
        before = {p: p.read_bytes() for p in (profile, session, motor)}
        output = self.directory / 'extension'
        command = [sys.executable, str(ROOT / 'scripts/extend_workspace_calibration.py'), '--profile', str(profile),
            '--session', str(session), '--output', str(output), '--motor-calibration', str(motor)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
        self.assertTrue((output / 'candidate-profile.json').is_file())
        self.assertFalse(json.loads(run.stdout)['activation_performed'])
        candidate_before = (output / 'candidate-profile.json').read_bytes()
        repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(repeated.returncode, 2)
        self.assertEqual((output / 'candidate-profile.json').read_bytes(), candidate_before)
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_cli_rejection_retains_report_without_publishing_a_profile(self):
        profile, session, motor = self.save_inputs()
        value = json.loads(session.read_text())
        value['frames'][1]['board_detection']['corners_px'] = (np.array(value['frames'][1]['board_detection']['corners_px']) + [70., 0.]).tolist()
        session.write_text(json.dumps(value)); output = self.directory / 'rejected'
        result = extend_saved_session(profile, session, output, urdf_path=self.urdf, motor_calibration_path=motor)
        self.assertEqual(result['status'], 'rejected')
        self.assertTrue((output / 'report.json').is_file())
        self.assertFalse((output / 'candidate-profile.json').exists())

    def test_cli_rejects_mutated_images_invalid_session_and_different_model(self):
        profile, session, motor = self.save_inputs()
        value = json.loads(session.read_text())
        path = Path(value['frames'][0]['path']); saved = path.read_bytes(); path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'content has changed'):
            read_session(session)
        path.write_bytes(saved)
        value['board_static_declared'] = False; session.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'fixed board'):
            read_session(session)
        value['board_static_declared'] = True; session.write_text(json.dumps(value))
        changed = deepcopy(self.motor); changed['wrist_roll']['range_min'] = 12; motor.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'geometry revision'):
            extend_saved_session(profile, session, self.directory / 'bad-model', urdf_path=self.urdf,
                                 motor_calibration_path=motor)
        self.assertFalse((self.directory / 'bad-model').exists())


if __name__ == '__main__':
    unittest.main()
