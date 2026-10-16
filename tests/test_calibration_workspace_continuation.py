"""Measured workspace continuation and offline reconciliation, no devices."""
from copy import deepcopy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import test_calibration_workspace_motion as fixtures
from server import calibration_workspace_motion as workspace
from server.calibration_workspace_journal import WorkspaceJournal


class MeasuredContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.WorkspaceProtocolTests.setUpClass()
        cls.model = fixtures.WorkspaceProtocolTests.model
        cls.protocol = fixtures.WorkspaceProtocolTests.protocol
        cls.frame = staticmethod(fixtures.WorkspaceProtocolTests.frame)

    def setUp(self):
        self.raw = {**self.protocol['origin'], 'shoulder_pan': 2255, 'shoulder_lift': 1074,
                    'elbow_flex': 3038, 'wrist_flex': 2157}
        self.held = {**self.raw, 'elbow_flex': 3031}
        self.grasp = SimpleNamespace(model=self.model, state='idle', record=lambda _: None,
            robot=SimpleNamespace(goals=self.held, pwm_owner=None))
        self.receipt = dict(completed=True, completion='stable_partial_observation',
            endpoint_reached=False, protocol_sha256=self.protocol['contract_sha256'],
            station_id=self.protocol['stations'][0]['station_id'], after=self.frame(self.raw))

    def prepare(self, **overrides):
        frame = overrides.pop('frame', self.frame(self.raw))
        kwargs = dict(station_index=1, completed_station_ids=[self.receipt['station_id']],
                      previous_receipt=self.receipt)
        kwargs.update(overrides)
        return workspace.prepare_workspace_station(self.grasp, self.protocol, frame, **kwargs)

    def validate(self, candidate, **overrides):
        kwargs = dict(session_id=self.protocol['session_id'],
            completed_station_ids=[self.receipt['station_id']], previous_receipt=self.receipt,
            approval=dict(approved=True, gripper_empty=True, human_clear=True, clear_path=True,
                protocol_sha256=self.protocol['contract_sha256'], candidate_sha256=candidate['contract_sha256'],
                frame_id=candidate['reference']['id'], reason='Offline complete path review.'))
        kwargs.update(overrides)
        with patch('server.grasp.unchanged_wrist'):
            return workspace.validate_workspace_station(self.grasp, self.protocol,
                candidate, candidate['reference'], **kwargs)

    def test_partial_actual_is_start_and_unchanged_elbow_retains_held_goal(self):
        candidate = self.prepare()
        self.assertEqual(candidate['origin'], self.protocol['origin'])
        self.assertEqual(candidate['segment']['measured_origin'], self.raw)
        self.assertEqual(candidate['original_nominal_target']['elbow_flex'], 3000)
        self.assertEqual(candidate['resolved_target_joints']['elbow_flex'], 3038)
        self.assertEqual(candidate['segment']['physical_goal_targets']['elbow_flex'], 3031)
        self.assertNotIn('elbow_flex', candidate['segment']['active_joints'])
        self.assertFalse(candidate['segment']['stationary_goal_resends'])
        self.assertEqual(candidate['previous_receipt_sha256'], workspace._hash(self.receipt))
        for name in ('shoulder_pan', 'shoulder_lift', 'wrist_flex'):
            self.assertEqual(candidate['resolved_target_joints'][name],
                             self.protocol['stations'][1]['target_joints'][name])
        self.validate(candidate)

    def test_child_first_station_uses_checkpoint_and_only_declared_elbow_axis(self):
        protocol = deepcopy(self.protocol)
        protocol['replan'] = {'checkpoint_start_joints':dict(self.raw)}
        protocol['stations'][0]['planned_active_joints'] = ['elbow_flex']
        workspace._seal(protocol)
        candidate = workspace.prepare_workspace_station(self.grasp, protocol, self.frame(self.raw),
            station_index=0, completed_station_ids=[])
        self.assertEqual(candidate['segment']['measured_origin'],self.raw)
        self.assertEqual(candidate['segment']['fixed_origin'],self.protocol['origin'])
        self.assertEqual(candidate['segment']['active_joints'],['elbow_flex'])
        self.assertEqual(candidate['original_nominal_target']['shoulder_pan'],2181)
        self.assertEqual(candidate['resolved_target_joints']['shoulder_pan'],2255)
        self.assertEqual(candidate['segment']['physical_goal_targets']['shoulder_pan'],2255)

    def test_missing_wrong_or_stale_actual_receipt_cannot_supply_a_nominal_start(self):
        for receipt in (None, {**self.receipt, 'station_id': 'elsewhere'},
                        {**self.receipt, 'protocol_sha256': 'different'},
                        {**self.receipt, 'completed': False}):
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                self.prepare(previous_receipt=receipt)
        self.raw['shoulder_pan'] += 13
        with self.assertRaisesRegex(ValueError, '上一站'):
            self.prepare()

    def test_candidate_binds_actual_receipt_and_nominal_and_resolved_targets(self):
        candidate = self.prepare()
        variants = []
        for field in ('original_nominal_target', 'resolved_target_joints'):
            changed = deepcopy(candidate); changed[field]['elbow_flex'] += 1
            variants.append(changed)
        changed = deepcopy(candidate); changed['previous_receipt_sha256'] = 'other'
        variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                self.validate(workspace._seal(changed))
        altered_receipt = {**self.receipt, 'note': 'new receipt after review'}
        with self.assertRaises(ValueError):
            self.validate(candidate, previous_receipt=altered_receipt)

    def test_resolved_endpoint_does_not_claim_original_nominal_endpoint_was_reached(self):
        candidate = self.prepare()
        actual = self.frame(candidate['resolved_target_joints'])
        with patch.object(workspace, 'validate_workspace_station',
                return_value=self.protocol['stations'][1]), patch.object(workspace,
                'execute_prepared_segment', return_value={'after': actual, 'reached_nominal_endpoint': True}):
            result = workspace.execute_workspace_station(self.grasp, self.protocol, candidate,
                candidate['reference'], threading.Event(), lambda _: None,
                session_id=self.protocol['session_id'], completed_station_ids=[self.receipt['station_id']],
                previous_receipt=self.receipt, approval={})
        self.assertTrue(result['reached_resolved_endpoint'])
        self.assertFalse(result['reached_nominal_endpoint'])
        self.assertEqual(result['after']['joints']['elbow_flex'], 3038)
        self.assertEqual(result['original_nominal_target']['elbow_flex'], 3000)

    def test_camera_rebind_is_explicit_and_leaves_original_reference_untouched(self):
        original = deepcopy(self.protocol)
        rebind = workspace._seal(dict(version=1, kind='engineering_workspace_camera_rebind',
            protocol_sha256=self.protocol['contract_sha256'], camera_id=self.protocol['camera_id'],
            mount_revision=self.protocol['mount_revision'],
            camera_profile_sha256=self.protocol['camera_profile_sha256'],
            from_generation=self.protocol['reference']['camera_generation'], to_generation='restarted',
            original_reference_sha256=workspace._hash(self.protocol['reference'])))
        frame = {**self.frame(self.raw), 'camera_generation': 'restarted'}
        self.receipt['after'] = frame
        with self.assertRaises(ValueError):
            workspace.prepare_workspace_station(self.grasp, self.protocol, frame, station_index=1,
                completed_station_ids=[self.receipt['station_id']], previous_receipt=self.receipt)
        candidate = workspace.prepare_workspace_station(self.grasp, self.protocol, frame, station_index=1,
            completed_station_ids=[self.receipt['station_id']], previous_receipt=self.receipt, camera_rebind=rebind)
        self.validate(candidate, camera_rebind=rebind)
        self.assertEqual(self.protocol, original)
        bad = workspace._seal({**rebind, 'camera_profile_sha256': 'different'})
        with self.assertRaises(ValueError):
            self.validate(candidate, camera_rebind=bad)


class InterruptedObservationJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.journal = WorkspaceJournal(self.temp.name)
        self.session = 'a' * 32
        self.protocol = workspace._seal(dict(version=1, kind='engineering_workspace_calibration',
            session_id=self.session, origin={'elbow_flex': 3042},
            reference={'camera_generation': 'old'}, camera_id='wrist', mount_revision='mount',
            camera_profile_sha256='profile', stations=[{'station_id': 'elbow'}]))
        self.journal.create(self.session, self.protocol)
        self.journal.begin(self.session, 'elbow', 'candidate')
        self.old = self.journal.load(self.session)
        self.receipt = dict(completed=True, completion='stable_partial_observation',
            endpoint_reached=False, motors_commanded=False, station_id='elbow',
            protocol_sha256=self.protocol['contract_sha256'],
            after={'id': 'new', 'source': 'wrist', 'camera_generation': 'old', 'joints': {'elbow_flex': 3038}})
        self.result = workspace._seal(dict(version=1, kind='engineering_workspace_interrupted_observation',
            parent_record_sha256=workspace._hash(self.old), protocol_sha256=self.protocol['contract_sha256'],
            station_id='elbow', candidate_id='candidate', receipt=self.receipt,
            failure_evidence={'error': 'endpoint'}, telemetry=[{'actual': 3031}],
            failed_candidate={'id': 'candidate'}, camera_rebind=None))

    def test_reconcile_archives_journal_and_evidence_then_advances_only_observation(self):
        record = self.journal.resolve_interrupted_observation(self.session, self.result)
        self.assertEqual(record['protocol'], self.protocol)
        self.assertEqual(record['completed_station_ids'], ['elbow'])
        self.assertIsNone(record['inflight'])
        self.assertFalse(record['receipts'][0]['endpoint_reached'])
        self.assertEqual(record['receipts'][0]['after']['joints']['elbow_flex'], 3038)
        for value in (self.old, self.result):
            archive = self.journal.root / 'revisions' / (workspace._hash(value) + '.json')
            self.assertTrue(archive.exists()); self.assertEqual(archive.stat().st_mode & 0o222, 0)
        with self.assertRaises(ValueError):
            self.journal.resolve_interrupted_observation(self.session, self.result)

    def test_changed_parent_target_claim_or_missing_evidence_never_resolves(self):
        variants = []
        for field, value in (('parent_record_sha256', 'stale'), ('candidate_id', 'other'),
                             ('telemetry', []), ('failure_evidence', {})):
            variants.append({**deepcopy(self.result), field: value})
        changed = deepcopy(self.result); changed['receipt']['endpoint_reached'] = True
        variants.append(changed)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.journal.resolve_interrupted_observation(self.session, workspace._seal(value))
        self.assertEqual(self.journal.load(self.session), self.old)

    def test_changed_generation_requires_same_setup_explicit_rebind(self):
        changed = deepcopy(self.result); changed['receipt']['after']['camera_generation'] = 'new-generation'
        with self.assertRaises(ValueError):
            self.journal.resolve_interrupted_observation(self.session, workspace._seal(changed))
        changed['camera_rebind'] = workspace._seal(dict(version=1, kind='engineering_workspace_camera_rebind',
            protocol_sha256=self.protocol['contract_sha256'], camera_id='wrist', mount_revision='mount',
            camera_profile_sha256='profile', from_generation='old', to_generation='new-generation',
            original_reference_sha256=workspace._hash(self.protocol['reference'])))
        record = self.journal.resolve_interrupted_observation(self.session, workspace._seal(changed))
        self.assertEqual(record['camera_rebind']['to_generation'], 'new-generation')
        self.assertEqual(record['protocol']['reference'], {'camera_generation': 'old'})
