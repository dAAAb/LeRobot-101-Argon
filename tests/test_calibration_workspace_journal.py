from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.calibration_workspace_journal import WorkspaceJournal


class WorkspaceJournalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.journal = WorkspaceJournal(self.directory.name)
        self.session_id = 'a' * 32
        self.protocol = {'origin': {'elbow_flex': 3042}, 'stations': ['one', 'two']}

    def test_origin_cannot_be_replaced_after_reopening_store(self):
        self.journal.create(self.session_id, self.protocol)
        other = WorkspaceJournal(self.directory.name)
        with self.assertRaises(ValueError):
            other.create(self.session_id, {'origin': {'elbow_flex': 2600}})
        self.assertEqual(other.load(self.session_id)['protocol'], self.protocol)

    def test_interrupted_station_stays_unresolved_after_restart(self):
        self.journal.create(self.session_id, self.protocol)
        self.journal.begin(self.session_id, 'one', 'candidate-1')
        other = WorkspaceJournal(self.directory.name)
        for station in ('one', 'two'):
            with self.assertRaises(ValueError):
                other.begin(self.session_id, station, 'candidate-2')
        with self.assertRaises(ValueError):
            other.moved(self.session_id, 'one', {'completed': False})
        self.assertEqual(other.load(self.session_id)['completed_station_ids'], [])

    def test_movement_and_board_visibility_are_separate_receipts(self):
        self.journal.create(self.session_id, self.protocol)
        self.journal.begin(self.session_id, 'one', 'candidate-1')
        self.journal.moved(self.session_id, 'one', {'completed': True, 'after': {'joints': {}}})
        self.journal.captured(self.session_id, 'one', {'accepted': False})
        record = self.journal.load(self.session_id)
        self.assertEqual(record['completed_station_ids'], ['one'])
        self.assertFalse(record['captures'][0]['result']['accepted'])
        self.assertIsNone(record['inflight'])
        with self.assertRaises(ValueError):
            self.journal.begin(self.session_id, 'one', 'old-candidate')
        self.journal.begin(self.session_id, 'two', 'candidate-2')

    def test_path_and_receipt_identity_are_checked(self):
        for session_id in ('../bad', 'bad', 'f' * 33):
            with self.assertRaises(ValueError):
                self.journal.load(session_id)
        self.journal.create(self.session_id, self.protocol)
        self.journal.begin(self.session_id, 'one', 'candidate-1')
        with self.assertRaises(ValueError):
            self.journal.moved(self.session_id, 'two', {'completed': True})
        with self.assertRaises(ValueError):
            self.journal.captured(self.session_id, 'two', {'accepted': True})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(protocol):
    result = deepcopy(protocol)
    result['contract_sha256'] = digest({k: v for k, v in result.items() if k != 'contract_sha256'})
    return result


class WorkspaceJournalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.journal = WorkspaceJournal(self.directory.name)
        self.session_id = 'a' * 32
        self.pose = dict(shoulder_pan=2254, shoulder_lift=1074, elbow_flex=3042,
                         wrist_flex=2154, wrist_roll=884, gripper=1861)
        self.protocol = seal(dict(version=1, kind='engineering_workspace_calibration',
            session_id=self.session_id, origin=self.pose,
            stations=[dict(station_id=f'station-0{i}', station_index=i-1,
                target_joints={**self.pose, 'elbow_flex': 3000-i*10}, capture_id=f'capture-{i}',
                split='training' if i == 1 else 'heldout', capture_required=True) for i in (1, 2)],
            camera_profile_sha256='camera-v1', geometry_revision='model-v1',
            origin_limit_profile='engineering_workspace_calibration_v1',
            joint_delta_bounds_from_original_origin={'elbow_flex': [-520, 24]},
            maximum_tip_distance_from_original_origin_mm=180., maximum_segment_ticks=180,
            maximum_segment_tip_mm=80., maximum_segment_seconds=30.))
        self.journal.create(self.session_id, self.protocol)
        self.journal.begin(self.session_id, 'station-01', 'failed-candidate')
        self.old = self.journal.load(self.session_id)
        self.recovery = self.replacement(self.old)
        self.archive = self.journal.root / 'revisions' / (self.protocol['contract_sha256'] + '.json')

    def replacement(self, old):
        stations = [dict(station_id='recovery-elbow-01', station_index=0,
            target_joints={**self.pose, 'elbow_flex': 3000}, capture_id=None, split=None, capture_required=False),
            *deepcopy(old['protocol']['stations'])]
        for index, station in enumerate(stations):
            station['station_index'] = index
        return seal({**old['protocol'], 'stations': stations, 'recovery': {
                'kind': 'first_station_single_elbow_prelude',
                'parent_protocol_sha256': old['protocol']['contract_sha256'],
                'parent_record_sha256': digest(old), 'failed_station_id': old['inflight']['station_id']}})

    def settling_replacement(self, old):
        start = {**old['protocol']['stations'][0]['target_joints'], 'elbow_flex': 3038}
        return seal({**old['protocol'], 'recovery': {
            'kind': 'first_station_settling_retry', 'attempt': 2,
            'parent_protocol_sha256': old['protocol']['contract_sha256'],
            'parent_record_sha256': digest(old), 'failed_station_id': old['inflight']['station_id'],
            'root_original_protocol_sha256': old['recovered_from'],
            'recovery_start': start, 'failure_evidence': {'actual': {**start, 'elbow_flex': 3031}}}})

    def test_recovery_archives_complete_old_state_and_keeps_original_origin(self):
        unrelated = Path(self.directory.name) / '.runtime' / 'motion-origin.json'
        unrelated.write_text('original independent session origin')
        result = self.journal.recover(self.session_id, self.recovery)
        self.assertEqual(json.loads(self.archive.read_text()), self.old)
        self.assertEqual(self.archive.stat().st_mode & 0o222, 0)
        self.assertEqual(result['protocol'], self.recovery)
        self.assertEqual(result['protocol']['origin'], self.old['protocol']['origin'])
        self.assertEqual(result['recovered_from'], self.protocol['contract_sha256'])
        self.assertIsNone(result['inflight'])
        for key in ('completed_station_ids', 'receipts', 'captures'):
            self.assertEqual(result[key], [])
        self.assertEqual(unrelated.read_text(), 'original independent session origin')
        with self.assertRaises(ValueError):
            self.journal.create(self.session_id, self.recovery)
        self.assertEqual(WorkspaceJournal(self.directory.name).load(self.session_id), result)

    def test_parent_session_origin_failed_station_and_seal_are_bound_before_writes(self):
        variants = []
        for key, value in (('parent_protocol_sha256', 'b' * 64),
                           ('parent_record_sha256', 'c' * 64),
                           ('failed_station_id', 'station-02')):
            changed = deepcopy(self.recovery); changed['recovery'][key] = value
            variants.append(seal(changed))
        for key, value in (('session_id', 'b' * 32), ('origin', {'elbow_flex': 3000})):
            variants.append(seal({**self.recovery, key: value}))
        variants.append({**self.recovery, 'contract_sha256': 'd' * 64})
        for changed in variants:
            with self.subTest(changed=changed['contract_sha256']), self.assertRaises(ValueError):
                self.journal.recover(self.session_id, changed)
            self.assertEqual(self.journal.load(self.session_id), self.old)
            self.assertFalse(self.archive.exists())

    def test_profile_and_original_scope_cannot_be_changed_by_a_recovery(self):
        replacements = dict(camera_profile_sha256='camera-v2', geometry_revision='model-v2',
            origin_limit_profile='unbounded', joint_delta_bounds_from_original_origin={'elbow_flex': [-900, 24]},
            maximum_tip_distance_from_original_origin_mm=250., maximum_segment_ticks=240,
            maximum_segment_tip_mm=100., maximum_segment_seconds=60.)
        for key, value in replacements.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.journal.recover(self.session_id, seal({**self.recovery, key: value}))
        self.assertEqual(self.journal.load(self.session_id), self.old)
        self.assertFalse(self.archive.exists())

    def test_only_an_unfinished_first_station_can_be_recovered(self):
        for change in ({'inflight': None}, {'completed_station_ids': ['station-01']},
                       {'inflight': {'station_id': 'station-02', 'candidate_id': 'later'}}):
            old = {**deepcopy(self.old), **change}
            self.journal._save(old)
            new = self.replacement(old) if old['inflight'] else self.recovery
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.journal.recover(self.session_id, new)
            self.assertEqual(self.journal.load(self.session_id), old)
        self.assertFalse(self.archive.exists())

    def test_different_or_symlinked_revision_is_never_overwritten(self):
        self.archive.parent.mkdir(parents=True)
        self.archive.write_text(json.dumps({'different': True}))
        with self.assertRaisesRegex(ValueError, '不可覆寫'):
            self.journal.recover(self.session_id, self.recovery)
        self.assertEqual(json.loads(self.archive.read_text()), {'different': True})
        self.archive.unlink()
        target = Path(self.directory.name) / 'keep.json'; target.write_text(json.dumps(self.old))
        self.archive.symlink_to(target)
        with self.assertRaisesRegex(ValueError, '符號連結'):
            self.journal.recover(self.session_id, self.recovery)
        self.assertEqual(self.journal.load(self.session_id), self.old)
        self.assertEqual(json.loads(target.read_text()), self.old)

    def test_archive_precedes_current_save_and_identical_archive_allows_retry(self):
        with patch.object(self.journal, '_save', side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError):
                self.journal.recover(self.session_id, self.recovery)
        self.assertEqual(self.journal.load(self.session_id), self.old)
        self.assertEqual(json.loads(self.archive.read_text()), self.old)
        identity = self.archive.stat().st_ino
        self.journal.recover(self.session_id, self.recovery)
        self.assertEqual(self.archive.stat().st_ino, identity)
        self.assertEqual(json.loads(self.archive.read_text()), self.old)

    def test_arbitrary_second_recovery_is_forbidden_even_if_first_station_fails(self):
        self.journal.recover(self.session_id, self.recovery)
        self.journal.begin(self.session_id, 'recovery-elbow-01', 'second-failed-candidate')
        current = self.journal.load(self.session_id)
        with self.assertRaisesRegex(ValueError, '無限重試鏈'):
            self.journal.recover(self.session_id, self.replacement(current))
        self.assertEqual(self.journal.load(self.session_id), current)
        self.assertEqual(json.loads(self.archive.read_text()), self.old)

    def test_one_final_settling_retry_keeps_targets_and_archives_both_parents(self):
        self.journal.recover(self.session_id, self.recovery)
        self.journal.begin(self.session_id, 'recovery-elbow-01', 'second-failed-candidate')
        first = self.journal.load(self.session_id)
        second_protocol = self.settling_replacement(first)
        second = self.journal.recover(self.session_id, second_protocol)
        self.assertEqual(second['protocol']['origin'], self.protocol['origin'])
        self.assertEqual(second['protocol']['stations'], first['protocol']['stations'])
        first_hash = first['protocol']['contract_sha256']
        self.assertEqual(second['recovered_from'], first_hash)
        self.assertEqual(second['recovered_from_chain'], [self.protocol['contract_sha256'], first_hash])
        self.assertEqual(second['recovery_attempt'], 2)
        self.assertIsNone(second['inflight'])
        second_archive = self.archive.parent / (first_hash + '.json')
        self.assertEqual(json.loads(second_archive.read_text()), first)
        self.assertEqual(json.loads(self.archive.read_text()), self.old)
        self.journal.begin(self.session_id, 'recovery-elbow-01', 'third-failed-candidate')
        current = self.journal.load(self.session_id)
        for kind in ('first_station_single_elbow_prelude', 'first_station_settling_retry'):
            third = self.settling_replacement(current)
            third['recovery'].update(kind=kind, attempt=3)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, '第三次'):
                self.journal.recover(self.session_id, seal(third))
        self.assertEqual(self.journal.load(self.session_id), current)

    def test_final_settling_retry_cannot_change_any_station_or_reset_attempt(self):
        self.journal.recover(self.session_id, self.recovery)
        self.journal.begin(self.session_id, 'recovery-elbow-01', 'second-failed-candidate')
        current = self.journal.load(self.session_id)
        nominal = self.settling_replacement(current)
        variants = []
        for index in (0, 1, 2):
            changed = deepcopy(nominal)
            changed['stations'][index]['target_joints']['elbow_flex'] -= 20
            variants.append(changed)
        changed = deepcopy(nominal); changed['stations'].append(deepcopy(changed['stations'][-1]))
        variants.append(changed)
        changed = deepcopy(nominal); changed['stations'][1]['split'] = 'heldout'
        variants.append(changed)
        for attempt in (1, 3, True):
            changed = deepcopy(nominal); changed['recovery']['attempt'] = attempt
            variants.append(changed)
        changed = deepcopy(nominal); changed['recovery']['root_original_protocol_sha256'] = 'f' * 64
        variants.append(changed)
        for index, changed in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(ValueError):
                self.journal.recover(self.session_id, seal(changed))
        self.assertEqual(self.journal.load(self.session_id), current)

    def test_final_retry_uses_new_measured_hold_axes_but_same_elbow_endpoint(self):
        self.journal.recover(self.session_id, self.recovery)
        self.journal.begin(self.session_id, 'recovery-elbow-01', 'second-failed-candidate')
        current = self.journal.load(self.session_id)
        nominal = self.settling_replacement(current)
        nominal['recovery']['recovery_start']['shoulder_pan'] += 1
        nominal['stations'][0]['target_joints']['shoulder_pan'] += 1
        result = self.journal.recover(self.session_id, seal(nominal))
        self.assertEqual(result['protocol']['stations'][0]['target_joints']['shoulder_pan'], 2255)
        self.assertEqual(result['protocol']['stations'][0]['target_joints']['elbow_flex'], 3000)
        self.assertEqual(result['protocol']['stations'][1:], current['protocol']['stations'][1:])

    def test_final_retry_rejects_unbound_or_excessively_drifted_hold_targets(self):
        self.journal.recover(self.session_id, self.recovery)
        self.journal.begin(self.session_id, 'recovery-elbow-01', 'second-failed-candidate')
        current = self.journal.load(self.session_id)
        nominal = self.settling_replacement(current)
        variants = []
        # A changed hold endpoint must be its measured start, not an extra move.
        changed = deepcopy(nominal); changed['stations'][0]['target_joints']['shoulder_pan'] += 1
        variants.append(changed)
        changed = deepcopy(nominal); changed['recovery']['recovery_start']['shoulder_pan'] += 1
        variants.append(changed)
        for name, delta in (('shoulder_pan', 13), ('gripper', 5)):
            changed = deepcopy(nominal)
            changed['recovery']['recovery_start'][name] += delta
            changed['stations'][0]['target_joints'][name] += delta
            variants.append(changed)
        changed = deepcopy(nominal); del changed['recovery']['failure_evidence']
        variants.append(changed)
        changed = deepcopy(nominal); changed['recovery']['recovery_start']['shoulder_pan'] = True
        variants.append(changed)
        for index, changed in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(ValueError):
                self.journal.recover(self.session_id, seal(changed))
        self.assertEqual(self.journal.load(self.session_id), current)

    def test_legacy_first_recovery_without_attempt_or_chain_is_still_bounded(self):
        first = self.journal.recover(self.session_id, self.recovery)
        first.pop('recovery_attempt'); first.pop('recovered_from_chain')
        self.journal._save(first)
        self.journal.begin(self.session_id, 'recovery-elbow-01', 'second-failed-candidate')
        current = self.journal.load(self.session_id)
        self.assertNotIn('attempt', current['protocol']['recovery'])
        result = self.journal.recover(self.session_id, self.settling_replacement(current))
        self.assertEqual(result['recovery_attempt'], 2)
        self.assertEqual(len(result['recovered_from_chain']), 2)
