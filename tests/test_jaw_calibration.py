"""Synthetic offline metrology fixtures, never a profile of the user's robot."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from server.jaw_calibration import fit_jaw_calibration
from server.spatial_contract import JawCalibration


GEOMETRY = 'a' * 64


class JawCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        def evidence(name):
            path = self.root / (name + '.txt')
            path.write_text('SYNTHETIC TEST EVIDENCE ONLY: ' + name)
            return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.data = {'schema_version': 1, 'source': 'wrist-only',
            'measurement_origin': 'physical-measurement', 'measurement_method': 'mechanical-survey',
            'coordinate_frame': 'gripper_link', 'units': 'm', 'geometry_revision': GEOMETRY,
            'landmark_definition': 'matched_contact_cross_section',
            'jaw_mount_revision': 'synthetic-mount-v1',
            'registration_description': 'Synthetic test coordinates; not measurements of a physical robot.',
            'registration_evidence': [evidence('registration')], 'measurements': []}
        for index, raw in enumerate(range(1500, 1901, 50)):
            self.data['measurements'].append({'measurement_id': f'm-{index}', 'acquisition_id': f'a-{index}',
                'split': 'training' if raw % 100 == 0 else 'heldout', 'raw': raw,
                'fixed_pad_m': [-.020, .010, .050],
                'moving_pad_m': [.010 + (raw - 1500) * .000075, .010, .050],
                'approach_axis': [0., 0., 1.], 'position_uncertainty_m': .0005,
                'axis_uncertainty_deg': .1, 'evidence': [evidence(f'sample-{index}')]})

    def run_fit(self, data=None, **kwargs):
        return fit_jaw_calibration(data or self.data, evidence_root=self.root,
                                   expected_geometry_revision=kwargs.get('geometry', GEOMETRY),
                                   expected_wrist_calibration_fingerprint=kwargs.get('wrist_fingerprint'))

    def test_linear_measured_landmarks_validate_and_preserve_nonzero_uncertainty(self):
        report = self.run_fit()
        self.assertTrue(report['passed'])
        profile = report['calibration_profile']
        calibration = JawCalibration.from_mapping(profile, geometry_revision=GEOMETRY)
        self.assertEqual(report['profile_fingerprint'], calibration.fingerprint)
        self.assertEqual(len(profile['samples']), 5)
        self.assertEqual(len(report['validation']['heldout']), 4)
        self.assertAlmostEqual(profile['position_uncertainty_m'], .001)
        self.assertTrue(calibration.at(1650).metric_calibrated)
        np.testing.assert_allclose(calibration.at(1650).T_wrist_tcp.as_array()[:3, 3], [.000625, .01, .05])
        self.assertAlmostEqual(calibration.at(1650).aperture_m, .04125)
        self.assertFalse(report['automatically_activated'])
        for raw in (1499, 1901):
            with self.assertRaisesRegex(ValueError, 'outside measured'):
                calibration.at(raw)

    def test_heldout_error_does_not_refit_training_model(self):
        original = self.run_fit()['calibration_profile']
        for row in self.data['measurements']:
            if row['split'] == 'heldout':
                row['fixed_pad_m'][1] += .0002
                row['moving_pad_m'][1] += .0002
        report = self.run_fit()
        self.assertTrue(report['passed'])
        self.assertEqual(report['calibration_profile']['samples'], original['samples'])
        self.assertAlmostEqual(report['validation']['maxima']['tcp_residual_m'], .0002)
        self.assertAlmostEqual(report['calibration_profile']['position_uncertainty_m'], .0012)
        self.assertNotEqual(report['input_sha256'], original['measurement_provenance']['input_sha256'])

    def test_three_knots_and_two_independent_intermediate_checks_can_validate(self):
        self.data['measurements'] = [self.data['measurements'][i] for i in (0, 1, 4, 5, 8)]
        report = self.run_fit()
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['calibration_profile']['samples']), 3)
        self.assertEqual(report['heldout_intervals'], [0, 1])

    def test_revolving_opening_preserves_measured_axis_at_each_knot(self):
        for sample in self.data['measurements']:
            theta = np.radians((sample['raw'] - 1700) * .04)
            width = .030 + (sample['raw'] - 1500) * .000075
            sample['moving_pad_m'] = (np.asarray(sample['fixed_pad_m']) +
                                     width * np.array([np.cos(theta), 0., np.sin(theta)])).tolist()
            sample['approach_axis'] = [-float(np.sin(theta)), 0., float(np.cos(theta))]
        report = self.run_fit()
        self.assertTrue(report['passed'], report['blockers'])
        profile = report['calibration_profile']
        self.assertNotEqual(profile['samples'][0]['approach_axis'], profile['samples'][-1]['approach_axis'])
        loaded = JawCalibration.from_mapping(profile, geometry_revision=GEOMETRY)
        np.testing.assert_allclose(loaded.at(1900).T_wrist_tcp.as_array()[:3, 2],
                                   self.data['measurements'][-1]['approach_axis'])

    def test_bad_holdout_only_produces_failed_report(self):
        self.data['measurements'][1]['moving_pad_m'][0] += .009
        report = self.run_fit()
        self.assertFalse(report['passed'])
        self.assertIsNone(report['calibration_profile'])
        self.assertIn('aperture_error_envelope_m exceeds validation limit', report['blockers'])

    def test_declared_measurement_noise_is_not_erased_by_zero_fit_residual(self):
        for sample in self.data['measurements']:
            sample['position_uncertainty_m'] = .003
        report = self.run_fit()
        self.assertFalse(report['passed'])
        self.assertGreater(report['validation']['maxima']['tcp_error_envelope_m'], .005)

    def test_geometry_evidence_and_dataset_fingerprints_are_bound(self):
        with self.assertRaisesRegex(ValueError, 'geometry fingerprint'):
            self.run_fit(geometry='b' * 64)
        (self.root / 'sample-1.txt').write_text('changed capture')
        with self.assertRaisesRegex(ValueError, 'Evidence fingerprint mismatch'):
            self.run_fit()

    def test_cad_observer_pixels_and_unknown_methods_are_rejected(self):
        for field, value in [('measurement_origin', 'cad'), ('source', 'd435'),
                             ('measurement_method', 'pixel-landmarks'), ('coordinate_frame', 'camera'),
                             ('units', 'mm')]:
            with self.subTest(field=field):
                data = deepcopy(self.data); data[field] = value
                with self.assertRaises(ValueError):self.run_fit(data)
        data = deepcopy(self.data); data['measurements'][0]['fixed_pad_m'] = [10., 20.]
        with self.assertRaises(ValueError):self.run_fit(data)

    def test_wrist_3d_requires_calibration_fingerprint(self):
        self.data['measurement_method'] = 'calibrated-wrist-3d'
        with self.assertRaisesRegex(ValueError, 'calibration fingerprint'):
            self.run_fit()
        self.data['wrist_calibration_fingerprint'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'matching validated calibration fingerprint'):
            self.run_fit(wrist_fingerprint='d' * 64)
        self.assertTrue(self.run_fit(wrist_fingerprint='c' * 64)['passed'])

    def test_copied_or_insufficient_holdouts_are_rejected(self):
        for change in ('acquisition', 'evidence', 'heldout_copy', 'raw', 'count'):
            with self.subTest(change=change):
                data = deepcopy(self.data); a, b = data['measurements'][:2]
                if change == 'acquisition':b['acquisition_id'] = a['acquisition_id']
                if change == 'evidence':b['evidence'] = a['evidence']
                if change == 'heldout_copy':data['measurements'][3]['evidence'] = b['evidence']
                if change == 'raw':b['raw'] = a['raw']
                if change == 'count':data['measurements'] = data['measurements'][:4]
                with self.assertRaises(ValueError):self.run_fit(data)

    def test_heldouts_must_span_multiple_intervals_without_extrapolation(self):
        data = deepcopy(self.data)
        held = [s for s in data['measurements'] if s['split'] == 'heldout']
        for i, sample in enumerate(held):sample['raw'] = 1501 + i
        with self.assertRaisesRegex(ValueError, 'two interpolation intervals'):
            self.run_fit(data)
        self.data['measurements'][1]['raw'] = 1950
        with self.assertRaisesRegex(ValueError, 'inside the measured training range'):
            self.run_fit()

    def test_axes_and_units_must_define_a_physical_tcp(self):
        for change in ('zero', 'parallel', 'nan', 'meters', 'coincident'):
            with self.subTest(change=change):
                data = deepcopy(self.data); sample = data['measurements'][0]
                if change == 'zero':sample['approach_axis'] = [0., 0., 0.]
                if change == 'parallel':sample['approach_axis'] = [1., 0., 0.]
                if change == 'nan':sample['moving_pad_m'][0] = float('nan')
                if change == 'meters':sample['fixed_pad_m'][1] = 50.
                if change == 'coincident':sample['moving_pad_m'] = sample['fixed_pad_m']
                with self.assertRaises(ValueError):self.run_fit(data)

    def test_changed_mount_changes_profile_binding(self):
        first = self.run_fit()
        self.data['jaw_mount_revision'] = 'synthetic-mount-v2'
        second = self.run_fit()
        self.assertNotEqual(first['profile_fingerprint'], second['profile_fingerprint'])
        self.assertNotEqual(first['calibration_profile']['validation_id'], second['calibration_profile']['validation_id'])

    def test_evidence_cannot_escape_dataset_directory(self):
        self.data['measurements'][0]['evidence'][0]['path'] = '../outside.txt'
        with self.assertRaisesRegex(ValueError, 'inside the measurement directory'):
            self.run_fit()

    def test_cli_writes_new_artifacts_without_activating_and_rejects_overwrite(self):
        from scripts.calibrate_jaws import main
        source = self.root / 'measurements.json'
        source.write_text(json.dumps(self.data))
        destination = self.root / 'result'
        with patch('scripts.calibrate_jaws.default_model'), patch('scripts.calibrate_jaws.geometry_revision', return_value=GEOMETRY):
            self.assertEqual(main([str(source), '--output', str(destination)]), 0)
            with self.assertRaises(SystemExit):main([str(source), '--output', str(destination)])
        self.assertTrue((destination / 'profile.json').exists())
        self.assertTrue((destination / 'report.json').exists())
        self.assertFalse((self.root / 'robot.json').exists())

    def test_cli_failure_preserves_report_without_profile(self):
        from scripts.calibrate_jaws import main
        source = self.root / 'measurements.json'
        self.data['measurements'][1]['moving_pad_m'][0] += .009
        source.write_text(json.dumps(self.data))
        destination = self.root / 'failed'
        with patch('scripts.calibrate_jaws.default_model'), patch('scripts.calibrate_jaws.geometry_revision', return_value=GEOMETRY):
            self.assertEqual(main([str(source), '--output', str(destination)]), 2)
        self.assertFalse((destination / 'profile.json').exists())
        self.assertFalse(json.loads((destination / 'report.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
