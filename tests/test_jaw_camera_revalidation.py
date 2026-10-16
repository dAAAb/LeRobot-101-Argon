"""Offline frozen-camera equivalence; synthetic artifacts, no device access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from server.jaw_camera_revalidation import (revalidate_saved_jaw_camera,
    validate_frozen_camera_extension, _canonical_hash)
from server.jaw_model_assisted import validate_silhouette_evidence, build_model_assisted_profile
from server.spatial_contract import JOINTS, WristCalibration, JawCalibration


class JawCameraRevalidationTests(unittest.TestCase):
    def setUp(self):
        # The acquisition fixture creates five temporary JPEGs and a complete
        # encoder-bound manifest on an in-memory bus. These are NOT robot data.
        from test_jaw_model_assisted import ModelEvidenceTests
        ModelEvidenceTests.setUp(self)
        self.old = self.profile
        body = self.manifest['body_reference']
        self.old.update(board_session_id='old-board', needs_more_views=False,
            metric_calibrated=True, relative_translation_uncertainty_m=.002,
            base_position_uncertainty_m=.003, relative_rotation_uncertainty_deg=.2,
            focal_relative_uncertainty=.004,
            validation_scope={'joint_raw_bounds': {n: [body[n], body[n]] for n in JOINTS[:-1]},
                'sampled_joint_poses': [dict(body)], 'board_plane_base': np.eye(4).tolist(),
                'extrapolation_allowed': False, 'individually_identified_joint_zeros': False,
                'jaw_contact_geometry_calibrated': False})
        # Complete the synthetic original before it is frozen/bound by hashes.
        self.manifest['camera_profile'] = deepcopy(self.old)
        self.manifest['camera_profile_sha256'] = _canonical_hash(self.old, ensure_ascii=True)
        self.save(self.path, self.manifest)
        self.old_camera_path = self.root / 'old-camera.json'; self.save(self.old_camera_path, self.old)
        self.new = deepcopy(self.old)
        frames = [{'id': 'board-' + str(i), 'joints': {n: body[n] + (i + 1) * 10 for n in JOINTS[:-1]}}
                  for i in range(6)]
        training, held = ['board-' + str(i) for i in (0, 2, 4)], ['board-' + str(i) for i in (1, 3, 5)]
        validation = dict(parent_profile_sha256=_canonical_hash(self.old),
            frozen_parameters=['camera_matrix', 'distortion', 'handeye'], fitted_parameters=['T_base_board'],
            heldout_data_used_for_fitting=False, heldout_used_for_shared_parameter_fitting=False,
            training_frame_ids=training, heldout_frame_ids=held, board_plane_base=np.eye(4).tolist())
        self.new.update(calibration_revision='frozen-new', validation_id='frozen-new-holdout',
            board_session_id='new-board', relative_translation_uncertainty_m=.004,
            base_position_uncertainty_m=.005, relative_rotation_uncertainty_deg=.4)
        self.new['validation_scope']['sampled_joint_poses'] += [frames[i]['joints'] for i in (0, 2, 4, 1, 3, 5)]
        self.new['validation_scope']['joint_raw_bounds'] = {n: [body[n], body[n] + 60] for n in JOINTS[:-1]}
        self.new['extension_provenance'] = dict(parent_profile=deepcopy(self.old),
            parent_profile_sha256=_canonical_hash(self.old), board_session_id='new-board',
            validation=deepcopy(validation), frames=frames)
        self.extension = dict(status='validated', method='frozen-camera-workspace-extension',
            source='wrist-only', board_session_id='new-board', validation=validation,
            calibration_profile=deepcopy(self.new))
        self.new_camera_path = self.root / 'new-camera.json'; self.save(self.new_camera_path, self.new)
        self.extension_path = self.root / 'extension.json'; self.save(self.extension_path, self.extension)
        self.provenance = validate_silhouette_evidence(self.path, self.annotation_path, self.old,
            expected_geometry_revision='geometry', cad_files=[self.cad])
        assumptions = dict(sigma_multiplier=2., cad_pad_tolerance_m=.002, encoder_tolerance_ticks=8.,
            description='Synthetic conditional section records for exercising immutable artifact rederivation only.')
        covariance = dict(identifiable=True, parameter_rank=7, covariance=(np.eye(7) * .001).tolist())
        lo, hi = self.provenance['measured_raw_range']
        sections = []
        for raw in range(lo, hi + 1):
            gap = .02 + (raw - lo) * .0001
            sections.append({'raw': raw, 'approach_camera': [0., 0., 1.], 'estimate': {
                'geometry_method': 'articulated-cad-wrist-silhouette', 'uncertainty_frame': 'wrist_camera_optical',
                'assumptions': assumptions, 'conditional_error_m': dict(tcp=.003, fixed_pad=.004,
                    moving_pad=.004, aperture=.005),
                'nominal': dict(fixed_pad_m=[-.01, 0., .3], moving_pad_m=[gap - .01, 0., .3])}})
        heldout = [dict(frame_id=i, silhouette_p95_px=2.) for i in self.provenance['heldout_frame_ids']]
        self.old_report = dict(provenance=self.provenance, assumptions=assumptions,
            conditional_covariance=covariance, heldout=heldout, sections=sections)
        self.old_report_path = self.root / 'old-jaw-report.json'; self.save(self.old_report_path, self.old_report)
        self.jaw = build_model_assisted_profile(provenance=self.provenance, camera_profile=self.old,
            calibration_revision='synthetic-jaw', validation_id='synthetic-heldout', geometry_revision='geometry',
            sections=sections, assumptions=assumptions, covariance_evidence=covariance,
            contact_section=dict(cad_yz_m=[.003, -.08], description='Synthetic local section; never real production calibration.'),
            validation=dict(parameter_rank=7, accepted=True, heldout_silhouette_p95_limit_px=6.,
                acceptance_basis='Synthetic retained criterion, not evidence of real robot accuracy.', heldout=heldout),
            evidence_files=[self.path, self.annotation_path, self.cad, self.old_report_path])
        self.jaw_path = self.root / 'jaw-profile.json'; self.save(self.jaw_path, self.jaw)

    @staticmethod
    def save(path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def run_revalidation(self):
        return revalidate_saved_jaw_camera(jaw_profile_path=self.jaw_path, jaw_report_path=self.old_report_path,
            old_camera_path=self.old_camera_path, new_camera_path=self.new_camera_path,
            camera_extension_report_path=self.extension_path, manifest_path=self.path,
            output=self.root / 'new-result')

    def test_new_artifact_rederives_all_sections_and_preserves_original_snapshot(self):
        original = {p: p.read_bytes() for p in (self.path, self.old_report_path, self.jaw_path, self.old_camera_path)}
        result = self.run_revalidation()
        self.assertFalse(result['activation_performed'])
        profile = json.loads(Path(result['profile']).read_text())
        report = json.loads(Path(result['report']).read_text())
        self.assertEqual(profile['samples'], self.jaw['samples'])
        self.assertEqual(profile['model_assisted']['uncertainty'], self.jaw['model_assisted']['uncertainty'])
        self.assertEqual(profile['model_assisted']['assumptions'], self.jaw['model_assisted']['assumptions'])
        self.assertEqual(profile['model_assisted']['validation'], self.jaw['model_assisted']['validation'])
        self.assertNotEqual(profile['model_assisted']['camera_calibration_fingerprint'],
                            self.jaw['model_assisted']['camera_calibration_fingerprint'])
        chain = profile['model_assisted']['camera_revalidation']
        self.assertEqual(chain['acquisition_camera_fingerprint'], self.jaw['model_assisted']['camera_calibration_fingerprint'])
        self.assertFalse(chain['new_hardware_evidence'])
        self.assertEqual(chain['original_acquisition_provenance'], self.provenance)
        self.assertEqual(report['saved_sections_rederived'], len(self.old_report['sections']))
        self.assertFalse(report['motors_commanded'])
        parsed = JawCalibration.from_mapping(profile, geometry_revision='geometry')
        self.assertTrue(parsed.at(parsed.samples[0].raw).planning_eligible)
        self.assertFalse(parsed.at(parsed.samples[0].raw).metric_calibrated)
        for path, data in original.items(): self.assertEqual(path.read_bytes(), data)
        with self.assertRaises(FileExistsError): self.run_revalidation()

    def test_optics_handeye_focal_hardware_and_source_changes_rejected(self):
        changes = [('camera_matrix', lambda x: x[0].__setitem__(0, x[0][0] + 1)),
                   ('distortion', lambda x: x.__setitem__(0, x[0] + .001)),
                   ('handeye', lambda x: x[0].__setitem__(3, x[0][3] + .001))]
        for key, mutate in changes:
            value = deepcopy(self.new); mutate(value[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_frozen_camera_extension(self.old, value, self.extension)
        for key, value in (('focal_relative_uncertainty', .005), ('camera_id', 'changed'),
                           ('mount_revision', 'changed'), ('geometry_revision', 'changed'), ('source', 'D435')):
            new = {**self.new, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_frozen_camera_extension(self.old, new, self.extension)

    def test_uncertainty_cannot_be_lowered_and_parent_cannot_be_replaced(self):
        new = {**self.new, 'base_position_uncertainty_m': .001}
        with self.assertRaisesRegex(ValueError, 'retain or increase'):
            validate_frozen_camera_extension(self.old, new, self.extension)
        report = deepcopy(self.extension); report['status'] = 'rejected'
        with self.assertRaisesRegex(ValueError, 'lineage'):
            validate_frozen_camera_extension(self.old, self.new, report)
        new = deepcopy(self.new); new['extension_provenance']['parent_profile_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'lineage'):
            validate_frozen_camera_extension(self.old, new, {**self.extension, 'calibration_profile': new})

    def test_equal_parent_pose_error_floor_accepts_frozen_camera_lineage(self):
        new = deepcopy(self.new)
        for key in ('base_position_uncertainty_m', 'relative_translation_uncertainty_m',
                    'relative_rotation_uncertainty_deg'):
            new[key] = self.old[key]
        report = {**self.extension, 'calibration_profile': new}
        result = validate_frozen_camera_extension(self.old, new, report)
        self.assertTrue(result['numeric_projection_equivalent'])
        self.assertEqual(result['new_pose_uncertainty'], result['old_pose_uncertainty'])

    def test_chinese_parent_profile_uses_the_extension_producers_hash_format(self):
        from server.board_calibration import _hash as producer_hash
        old, new, report = deepcopy(self.old), deepcopy(self.new), deepcopy(self.extension)
        # Real camera profiles retain these detections from earlier sessions.
        provenance = {'sessions':[{'frames':[{'board_detection':{'reason':'已看見完整棋盤。'}}]}]}
        old['multi_session_provenance'] = deepcopy(provenance)
        new['multi_session_provenance'] = deepcopy(provenance)
        expected = producer_hash(old)
        self.assertNotEqual(expected, _canonical_hash(old))
        lineage = new['extension_provenance']
        lineage['parent_profile'] = old
        lineage['parent_profile_sha256'] = expected
        lineage['validation']['parent_profile_sha256'] = expected
        report['validation'] = deepcopy(lineage['validation'])
        report['calibration_profile'] = deepcopy(new)
        result = validate_frozen_camera_extension(old, new, report)
        self.assertTrue(result['numeric_projection_equivalent'])
        # Both lineage sites still need the exact producer identity; this is
        # not permission to accept either of two conflicting hash formats.
        for site in ('lineage', 'validation'):
            changed, changed_report = deepcopy(new), deepcopy(report)
            if site == 'lineage':
                changed['extension_provenance']['parent_profile_sha256'] = _canonical_hash(old)
            else:
                changed['extension_provenance']['validation']['parent_profile_sha256'] = _canonical_hash(old)
                changed_report['validation'] = deepcopy(changed['extension_provenance']['validation'])
            changed_report['calibration_profile'] = changed
            with self.subTest(site=site), self.assertRaisesRegex(ValueError, 'lineage'):
                validate_frozen_camera_extension(old, changed, changed_report)

    def test_forged_workspace_bounds_and_reused_holdout_rejected(self):
        new = deepcopy(self.new); new['validation_scope']['joint_raw_bounds']['elbow_flex'][1] += 100
        with self.assertRaisesRegex(ValueError, 'pose union'):
            validate_frozen_camera_extension(self.old, new, {**self.extension, 'calibration_profile': new})
        report = deepcopy(self.extension); new = deepcopy(self.new)
        report['validation']['heldout_frame_ids'] = report['validation']['training_frame_ids']
        new['extension_provenance']['validation'] = deepcopy(report['validation']); report['calibration_profile'] = new
        with self.assertRaisesRegex(ValueError, 'separate'):
            validate_frozen_camera_extension(self.old, new, report)

    def test_report_or_image_tampering_rejected_before_writing_candidate(self):
        self.old_report['sections'][0]['estimate']['nominal']['fixed_pad_m'][0] += .01
        self.save(self.old_report_path, self.old_report)
        with self.assertRaisesRegex(ValueError, 'Original jaw evidence changed'):
            self.run_revalidation()
        self.assertFalse((self.root / 'new-result').exists())

    def test_old_strict_camera_snapshot_validator_is_not_relaxed(self):
        with self.assertRaisesRegex(ValueError, 'camera snapshot'):
            validate_silhouette_evidence(self.path, self.annotation_path, self.new,
                expected_geometry_revision='geometry', cad_files=[self.cad])


if __name__ == '__main__':
    unittest.main()
