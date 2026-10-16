"""Train-only multi-board optics/handeye, using saved synthetic wrist evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import uuid

import cv2
import numpy as np

from server.board_calibration import _inverse, _project
from server.calibration_multi_session import fit_multi_session_calibration
from server.calibration_store import CalibrationSessionStore
from server.spatial_contract import JOINTS, UNCERTAINTY_FIELDS, WristCalibration
from server.spatial_runtime import geometry_revision
import test_board_calibration as fixtures

ROOT = Path(__file__).resolve().parents[1]


class MultiSessionCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture = fixtures.BoardCalibrationTests(); fixture.setUp()
        frames, self.model = fixture.dataset(count=22)
        self.model.urdf_path = ROOT / 'assets/so101_follower_cam.urdf'
        self.model.calibration = {n: {'range_min': 0, 'range_max': 4095} for n in JOINTS}
        revision = geometry_revision(self.model)
        self.store = CalibrationSessionStore(self.root)
        self.sessions, self.splits, self.split_source_solve_ids = [], {}, {}
        for si, subset in enumerate([frames[:16], frames[16:]]):
            sid = self.store.start(camera_id='wrist-A', mount_revision='mount-A',
                geometry_revision=revision, board_static=True)['session_id']
            session = self.store._load(sid)
            for i, frame in enumerate(subset):
                frame.update(camera_id='wrist-A', camera_generation=f'camera-{si}', board_session_id=sid,
                             mount_revision='mount-A', geometry_revision=revision)
                frame['board_detection']['camera_generation'] = frame['camera_generation']
                if si:
                    # The second session deliberately has a different, fixed board location.
                    second_board = fixture.base_board.copy(); second_board[0, 3] += .025
                    wrist = self.model.fk.transforms(frame['joints'])['gripper_link']
                    frame['board_detection']['corners_px'] = _project(fixture.board.object_points(),
                        _inverse(wrist @ fixture.handeye) @ second_board, fixture.k, fixture.distortion).tolist()
                path = self.store._directory(sid) / 'frames' / (frame['id'] + '.jpg')
                cv2.imwrite(str(path), np.full((960, 1280), 100 + si * 20 + i, np.uint8))
                frame.update(path=str(path), image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            session.update(frames=subset, status='collecting', camera_generation=f'camera-{si}', image_size=fixture.size)
            self.store._save(session)
            self.sessions.append(session)
            held = [f['id'] for i, f in enumerate(subset) if (i % 4 == 3 if si == 0 else i % 2 == 1)]
            self.splits[sid] = dict(training_frame_ids=[f['id'] for f in subset if f['id'] not in held],
                                   heldout_frame_ids=held)
            solve_id = uuid.uuid4().hex
            old_solve = self.store._directory(sid) / 'solves' / solve_id
            old_solve.mkdir()
            (old_solve / 'report.json').write_text(json.dumps(dict(board_session_id=sid,
                source='wrist-only', validation=self.splits[sid], status='rejected')))
            session['solves'].append(dict(solve_id=solve_id, status='rejected'))
            self.store._save(session)
            self.split_source_solve_ids[sid] = solve_id
        self.target = self.sessions[-1]['session_id']
        self.parent = dict(schema_version=1, source='wrist-only', calibration_revision='synthetic-parent',
            camera_id='wrist-A', mount_revision='mount-A', geometry_revision=revision,
            native_image_size=fixture.size, image_size=fixture.size, camera_matrix=fixture.k.tolist(),
            distortion=fixture.distortion.tolist(), distortion_model='opencv-brown', handeye=fixture.handeye.tolist(),
            validated_components=['intrinsics', 'handeye', 'robot_geometry'], validation_id='parent-heldout',
            validation_kind='independent-heldout', metric_calibrated=True, needs_more_views=False,
            camera_generation=None, board_session_id=self.sessions[0]['session_id'],
            validation_scope=dict(joint_raw_bounds={n: [min(f['joints'][n] for f in frames[:16]),
                max(f['joints'][n] for f in frames[:16])] for n in JOINTS[:-1]}),
            base_position_uncertainty_m=.003, relative_translation_uncertainty_m=.002,
            relative_rotation_uncertainty_deg=.2, focal_relative_uncertainty=.004)
        self.parent_path = self.root / 'parent-profile.json'
        self.parent_path.write_text(json.dumps(self.parent))

    def fit(self, sessions=None, **kwargs):
        return fit_multi_session_calibration(self.sessions if sessions is None else sessions,
            self.model, self.parent, splits=self.splits, board_session_id=self.target, **kwargs)

    def test_fit_preserves_split_separate_boards_parent_floor_and_real_sample_scope(self):
        result = self.fit()
        self.assertEqual(result['status'], 'validated', result['blockers'])
        p, v = result['calibration_profile'], result['validation']
        self.assertEqual(len(v['training_frame_ids']), 15)
        self.assertEqual(len(v['heldout_frame_ids']), 7)
        self.assertFalse(v['heldout_data_used_for_fitting'])
        held_rows = [r for r in v['evaluations'] if r['split'] == 'heldout']
        for summary, field in [('heldout_max_translation_m', 'board_translation_error_m'),
                ('heldout_max_rotation_deg', 'board_rotation_error_deg'),
                ('heldout_max_reprojection_p95_px', 'chain_reprojection_p95_px'),
                ('heldout_max_pnp_reprojection_p95_px', 'pnp_reprojection_p95_px')]:
            self.assertEqual(v[summary], max(r[field] for r in held_rows))
        self.assertEqual(v['max_pnp_reprojection_p95_px'],
                         max(r['pnp_reprojection_p95_px'] for r in v['evaluations']))
        self.assertIn('diagnostic only', v['reprojection_summary_interpretation'])
        self.assertEqual(len(p['validation_scope']['sampled_joint_poses']), 22)
        self.assertFalse(p['validation_scope']['extrapolation_allowed'])
        self.assertEqual(p['multi_session_provenance']['parent_profile'], self.parent)
        self.assertEqual(p['board_session_id'], self.target)
        ys = list(v['board_base_transforms'].values())
        self.assertGreater(np.linalg.norm(np.asarray(ys[0])[:3, 3] - np.asarray(ys[1])[:3, 3]), .02)
        for name in UNCERTAINTY_FIELDS:
            self.assertEqual(p[name], max(self.parent[name], v['empirical_uncertainty'][name]))
        self.assertFalse(p['grasp_ready'])
        self.assertTrue(WristCalibration.from_mapping(p, native_image_size=p['image_size'],
            camera_id=p['camera_id'], mount_revision=p['mount_revision'],
            geometry_revision=p['geometry_revision'], cad_handeye=np.eye(4)).metric_calibrated)

    def test_heldout_pixels_never_change_shared_fit(self):
        first = self.fit()
        changed = deepcopy(self.sessions)
        for s in changed:
            for f in s['frames']:
                if f['id'] in self.splits[s['session_id']]['heldout_frame_ids']:
                    f['board_detection']['corners_px'] = (np.asarray(f['board_detection']['corners_px']) + [.7, -.4]).tolist()
        second = self.fit(changed)
        self.assertEqual(first['status'], 'validated', first['blockers'])
        self.assertEqual(second['status'], 'validated', second['blockers'])
        for name in ('camera_matrix', 'distortion', 'handeye'):
            np.testing.assert_array_equal(first['calibration_profile'][name], second['calibration_profile'][name])

    def test_metric_mm_error_can_pass_chain_over_six_and_increase_error_budget(self):
        original = self.model.fk.transforms
        target = self.sessions[1]['frames'][1]['joints']['shoulder_pan']
        def perturbed(joints):
            transforms = {n: t.copy() for n, t in original(joints).items()}
            if joints['shoulder_pan'] == target:
                for t in transforms.values(): t[0, 3] += .008
            return transforms
        self.model.fk.transforms = perturbed
        result = self.fit()
        self.assertEqual(result['status'], 'validated', result['blockers'])
        rows = result['validation']['evaluations']
        self.assertGreater(max(r['chain_reprojection_p95_px'] for r in rows), 6)
        self.assertGreater(result['validation']['heldout_max_reprojection_p95_px'], 6)
        self.assertLess(result['validation']['heldout_max_pnp_reprojection_p95_px'], .1)
        self.assertGreater(result['calibration_profile']['base_position_uncertainty_m'], .007)
        self.assertGreater(result['calibration_profile']['relative_translation_uncertainty_m'], .007)

    def test_cross_session_holdout_pose_leak_is_rejected_without_resplitting(self):
        sessions = deepcopy(self.sessions)
        f = sessions[1]['frames'][1]
        f['joints'] = deepcopy(sessions[0]['frames'][0]['joints'])
        for sample in f['settled_joint_window']: sample['joints'] = deepcopy(f['joints'])
        result = self.fit(sessions)
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('heldout pose', result['blockers'][0])
        self.assertIsNone(result['calibration_profile'])

    def test_image_tamper_stale_corners_timing_and_camera_mismatch_reject(self):
        for field, value in [('camera_id', 'D435'), ('joint_capture_span_ticks', 3)]:
            sessions = deepcopy(self.sessions); sessions[0]['frames'][0][field] = value
            self.assertEqual(self.fit(sessions)['status'], 'rejected')
        sessions = deepcopy(self.sessions)
        sessions[0]['frames'][0]['board_detection']['frame_id'] = 'another-frame'
        self.assertIn('mismatched', self.fit(sessions)['blockers'][0])
        Path(self.sessions[0]['frames'][0]['path']).write_bytes(b'changed')
        self.assertIn('hash', self.fit()['blockers'][0])

    def test_under_observed_training_rejects_without_new_thresholds(self):
        original = self.model.fk.transforms
        constant = original(self.sessions[0]['frames'][0]['joints'])
        self.model.fk.transforms = lambda joints: constant
        result = self.fit()
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('轉軸', result['blockers'][0])

    def test_bad_heldout_optics_does_not_produce_profile(self):
        sessions = deepcopy(self.sessions)
        pixels = np.asarray(sessions[1]['frames'][1]['board_detection']['corners_px'])
        pixels += np.random.default_rng(853).normal(0, 10, pixels.shape)
        sessions[1]['frames'][1]['board_detection']['corners_px'] = pixels.tolist()
        result = self.fit(sessions)
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('PnP', result['blockers'][0])
        self.assertIsNone(result['calibration_profile'])

    def test_store_registers_real_solver_pair_leaves_config_and_parent_journal_unchanged(self):
        selected = self.root / 'config.json'; selected.write_text('{"selected":"parent"}')
        old_journal = self.store._directory(self.sessions[0]['session_id']) / 'session.json'
        before = old_journal.read_bytes()
        summary = self.store.solve_multi_session(self.target, self.model, splits=self.splits,
            split_source_solve_ids=self.split_source_solve_ids, parent_profile_path=self.parent_path)
        self.assertEqual(summary['status'], 'solved', summary['last_solve']['blockers'])
        solve = summary['last_solve']; self.assertFalse(solve['active_profile_changed'])
        self.assertEqual(solve['view_count'], 22)
        self.assertEqual(solve['target_view_count'], summary['collected_view_count'])
        self.assertEqual(solve['target_view_count'], 6)
        self.assertEqual(solve['source_session_view_counts'],
                         {s['session_id']: len(s['frames']) for s in self.sessions})
        self.assertEqual(solve['source_session_ids'], list(self.splits))
        report = json.loads(Path(solve['report_path']).read_text())
        profile = json.loads(Path(solve['candidate_profile_path']).read_text())
        self.assertEqual(profile, report['calibration_profile'])
        self.assertEqual(profile['board_session_id'], self.target)
        self.assertEqual(report['board_session_id'], self.target)
        self.assertEqual(old_journal.read_bytes(), before)
        self.assertEqual(selected.read_text(), '{"selected":"parent"}')
        self.assertEqual(profile['multi_session_provenance']['parent_artifact']['path'], str(self.parent_path.resolve()))

    def test_store_rejects_changed_original_holdout_split(self):
        changed = deepcopy(self.splits)
        split = changed[self.target]
        split['training_frame_ids'][0], split['heldout_frame_ids'][0] = (
            split['heldout_frame_ids'][0], split['training_frame_ids'][0])
        with self.assertRaisesRegex(ValueError, '不能改動'):
            self.store.solve_multi_session(self.target, self.model, splits=changed,
                split_source_solve_ids=self.split_source_solve_ids, parent_profile_path=self.parent_path)


if __name__ == '__main__':
    unittest.main()
