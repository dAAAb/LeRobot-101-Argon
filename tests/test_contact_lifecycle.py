"""Pure fake-IO safety regression tests. Never import Robot or camera drivers."""
import importlib.util
import json
import pathlib
import tempfile
import threading
import unittest

import server.gripper_power as m


class FakeServo:
    def __init__(self, journal):
        self.reg = {3:777, 33:0, 65:0, 48:500, 42:300, 56:300, 66:0}
        self.journal = journal; self.events = []; self.now = 20.; self.stop = False
        self.ack_loss = None; self.read_failure = None; self.settle_reads = 0
        self.before_write = None

    def read(self, servo, address, width):
        self.events.append(('read', servo, address, self.reg[address]))
        if self.read_failure and self.read_failure(address): raise IOError('lost telemetry')
        return self.reg[address]

    def write(self, servo, address, value, width):
        record = self.journal.get()
        self.events.append(('write', servo, address, value, record))
        assert servo == 6 and address in (42, 48) and width == 2
        assert record and record['phase'] != 'completed', 'write without pending journal'
        if self.before_write: self.before_write(address, value, record)
        if address == 42:
            assert self.reg[48] == 100 and self.reg[65] == 0, 'goal with unsafe cap/fault'
        if address == 48 and value > self.reg[48]:
            # Independent physical + durable evidence invariants.
            assert not record.get('contact_possible'), 'power increase while contact is possible'
            if 'contact_id' in record:
                assert record.get('release_receipt', {}).get('empty_verified') is True
                assert record['release_receipt']['source'] == 'wrist'
            assert self.reg[42] == self.reg[56] and self.reg[66] == 0 and self.reg[65] == 0
        outcome = self.ack_loss(address, value) if self.ack_loss else None
        if outcome == 'before': raise IOError('write not applied')
        self.reg[address] = value
        if outcome == 'after': raise IOError('write applied; ACK lost')

    def latch(self): self.stop = True
    def sleep(self, duration): self.now += duration
    def writes(self): return [(e[1], e[2], e[3]) for e in self.events if e[0] == 'write']


class ContactLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.journal = m.AtomicJournal(pathlib.Path(self.directory.name)/'power.json')
        self.bus = FakeServo(self.journal); self.lock = threading.RLock()
        self.session = self.new_session(); self.token = None

    def new_session(self, **kwargs):
        settings = dict(read=self.bus.read, write=self.bus.write, journal=self.journal,
                        lock=self.lock, stop_latch=self.bus.latch, device_key='test-device:6',
                        position_bounds=(100, 1100), verified_open_position=900, opening_direction=1,
                        clock=lambda: self.bus.now, sleep=self.bus.sleep)
        settings.update(kwargs)
        return m.GripperPWMSession(**settings)

    def contact(self):
        self.token = self.session.enter()
        return self.session.mark_contact_possible(self.token)

    def start_release(self):
        return self.session.begin_release(self.token, camera_generation='wrist-boot-A', frame_sequence=100)

    def open_and_observe(self):
        self.start_release()
        self.session.release_goal(self.token, 900)
        self.bus.reg[56] = 900  # Independent simulated physical opening.
        challenge = self.session.verify_release_open(self.token)
        self.bus.now += .2
        return dict(source='wrist', empty_verified=True, session_id=self.token,
                    release_id=challenge['release_id'], camera_generation='wrist-boot-A', sequence=104,
                    capture_monotonic=self.bus.now-.02, frame_sha256='f'*64)

    def assert_low_pending(self):
        self.assertEqual(self.bus.reg[48], 100)
        self.assertIs(self.journal.get()['contact_possible'], True)
        self.assertFalse(any(a == 48 and v == 500 for _,a,v in self.bus.writes()))

    def test_contact_mark_persists_before_caller_can_close_and_emits_no_goal(self):
        self.token = self.session.enter(); before = self.bus.writes()
        result = self.session.mark_contact_possible(self.token)
        self.assertTrue(result['contact_possible']); self.assertEqual(self.bus.writes(), before)
        self.assertEqual(self.journal.get()['contact_id'], result['contact_id'])
        self.session.assert_motion_allowed(self.token)

    def test_contact_mark_failed_journal_never_authorizes_closing(self):
        self.token = self.session.enter(); put = self.journal.put
        def failure(record):
            if record.get('contact_possible'): raise OSError('disk full')
            put(record)
        self.journal.put = failure
        with self.assertRaises(m.SessionFailure): self.session.mark_contact_possible(self.token)
        self.assertEqual(self.bus.writes(), [(6,48,100)])
        with self.assertRaises(m.RecoveryRequired): self.session.assert_motion_allowed(self.token)

    def test_contact_journal_error_after_atomic_replace_keeps_restoration_barrier(self):
        self.token = self.session.enter(); put = self.journal.put
        def failure(record):
            put(record)
            if record.get('contact_possible'): raise OSError('directory fsync error after replace')
        self.journal.put = failure
        with self.assertRaises(m.SessionFailure): self.session.mark_contact_possible(self.token)
        self.assert_low_pending(); before = list(self.bus.events)
        with self.assertRaises(m.RecoveryRequired): self.new_session().recover()
        self.assertEqual(self.bus.events, before)

    def test_finish_cancel_and_recover_never_touch_flagged_gripper(self):
        for operation in ('finish', 'cancel', 'recover'):
            with self.subTest(operation=operation):
                self.setUp(); self.contact(); before = list(self.bus.events)
                with self.assertRaises(m.RecoveryRequired):
                    self.session.finish(cancelled=True) if operation == 'cancel' else getattr(self.session, operation)()
                self.assertEqual(self.bus.events, before); self.assert_low_pending()
                self.assertTrue(self.bus.stop)

    def test_deferred_context_keeps_live_owner_and_never_claims_grasp(self):
        with self.session:
            self.token = self.session.token; self.session.mark_contact_possible(self.token)
            result = self.session.defer_holding(self.token)
        self.assertFalse(result['completed']); self.assertFalse(result['grasp_confirmed'])
        self.assertEqual(self.session.state, 'holding'); self.session.assert_motion_allowed(self.token)
        self.assert_low_pending(); self.assertEqual(self.bus.writes(), [(6,48,100)])

    def test_measured_stationarity_accepts_moving_flag_with_constant_contact_goal(self):
        self.contact(); self.bus.reg.update({42:288, 56:300, 66:1})
        result = self.session.defer_holding(self.token)
        self.assertFalse(result['grasp_confirmed']); self.assert_low_pending()
        proof = self.journal.get()['holding_stationary_samples']
        self.assertGreaterEqual(len(proof), 3)
        self.assertGreaterEqual(proof[-1]['monotonic']-proof[0]['monotonic'], .35)
        self.assertTrue(all(s['moving'] == 1 and s['goal'] == 288 and s['position'] == 300 for s in proof))
        self.assertEqual(self.bus.writes(), [(6,48,100)])

    def test_holding_rejects_encoder_drift_even_if_moving_flag_is_zero(self):
        self.contact(); sleep = self.session.sleep
        def drift(seconds):
            sleep(seconds); self.bus.reg[56] += 2
        self.session.sleep = drift
        with self.assertRaises(m.SessionFailure): self.session.defer_holding(self.token)
        self.assert_low_pending()

    def test_holding_rejects_goal_change_and_nonfresh_clock(self):
        for failure in ('goal', 'clock'):
            with self.subTest(failure=failure):
                self.setUp(); self.contact(); sleep = self.session.sleep
                def change(seconds):
                    if failure == 'goal': sleep(seconds); self.bus.reg[42] += 1
                self.session.sleep = change
                with self.assertRaises(m.SessionFailure): self.session.defer_holding(self.token)
                self.assert_low_pending()

    def test_restart_blocks_old_token_and_generic_recovery_with_no_servo_io(self):
        self.contact(); self.session.defer_holding(self.token); fresh = self.new_session()
        before = list(self.bus.events)
        for operation in (lambda: fresh.assert_motion_allowed(self.token), fresh.enter, fresh.recover):
            with self.assertRaises(m.RecoveryRequired): operation()
        self.assertEqual(self.bus.events, before); self.assert_low_pending()

    def test_stop_removes_outstanding_closing_demand_without_increasing_cap(self):
        self.contact(); self.bus.reg.update({42:180, 56:320})
        result = self.session.stop_contact(self.token)
        self.assertEqual(self.bus.reg[42], 320); self.assertEqual(result['stop_goal'], 320)
        self.assertTrue(result['closing_demand_removed']); self.assertFalse(result['stationary_verified'])
        self.assert_low_pending(); before = list(self.bus.events)
        self.assertTrue(self.session.stop_contact(self.token)['already_stopped'])
        self.assertEqual(self.bus.events, before)
        with self.assertRaises(m.RecoveryRequired): self.session.assert_motion_allowed(self.token)

    def test_stop_never_changes_an_opening_goal_toward_closing(self):
        self.contact(); self.bus.reg.update({42:550, 56:320})
        self.session.stop_contact(self.token)
        self.assertEqual(self.bus.writes(), [(6,48,100)]); self.assert_low_pending()

    def test_stop_handles_opposite_encoder_opening_direction(self):
        self.session = self.new_session(verified_open_position=150, opening_direction=-1)
        self.contact(); self.bus.reg.update({42:800, 56:500})
        self.session.stop_contact(self.token)
        self.assertEqual(self.bus.reg[42], 500); self.assert_low_pending()

    def test_stop_fault_or_read_failure_emits_no_goal(self):
        for failure in ('fault', 'read'):
            with self.subTest(failure=failure):
                self.setUp(); self.contact(); self.bus.reg.update({42:180, 56:320})
                if failure == 'fault': self.bus.reg[65] = 1
                else: self.bus.read_failure = lambda a: a == 56
                with self.assertRaises(m.SessionFailure): self.session.stop_contact(self.token)
                self.assertEqual(self.bus.writes(), [(6,48,100)]); self.assert_low_pending()

    def test_stop_ack_loss_is_not_misreported_success_or_retried(self):
        for mode in ('before', 'after'):
            with self.subTest(mode=mode):
                self.setUp(); self.contact(); self.bus.reg.update({42:180, 56:320})
                self.bus.ack_loss = lambda a,v: mode if a == 42 else None
                with self.assertRaises(m.SessionFailure): self.session.stop_contact(self.token)
                before = list(self.bus.events)
                with self.assertRaises(m.RecoveryRequired): self.session.stop_contact(self.token)
                self.assertEqual(self.bus.events, before); self.assert_low_pending()

    def test_contact_context_cancel_preserves_original_exception_and_low_power(self):
        with self.assertRaises(m.SessionCancelled):
            with self.session:
                self.token = self.session.token; self.session.mark_contact_possible(self.token)
                self.bus.reg.update({42:180, 56:320})
                raise m.SessionCancelled('cancelled by owner')
        self.assertEqual(self.bus.reg[42], 320); self.assert_low_pending()

    def test_generic_context_fault_performs_no_cleanup_io(self):
        with self.assertRaises(IOError):
            with self.session:
                self.token = self.session.token; self.session.mark_contact_possible(self.token)
                before = list(self.bus.events)
                raise IOError('telemetry lost')
        self.assertEqual(self.bus.events, before); self.assert_low_pending()

    def test_release_requires_open_direction_bound_and_active_owner(self):
        for goal in (299, 901, True):
            with self.subTest(goal=goal):
                self.setUp(); self.contact(); self.start_release()
                before = self.bus.writes()
                with self.assertRaises(m.RecoveryRequired): self.session.release_goal(self.token, goal)
                self.assertEqual(self.bus.writes(), before); self.assert_low_pending()

    def test_release_never_rewinds_an_already_more_open_goal(self):
        self.contact(); self.bus.reg[42] = 500; self.start_release()
        with self.assertRaises(m.RecoveryRequired): self.session.release_goal(self.token, 400)
        self.assertEqual(self.bus.writes(), [(6,48,100)]); self.assert_low_pending()

    def test_release_ack_loss_preserves_contact_and_blocks_generic_recovery(self):
        self.contact(); self.start_release()
        self.bus.ack_loss = lambda a,v: 'after' if a == 42 else None
        with self.assertRaises(m.SessionFailure): self.session.release_goal(self.token, 500)
        self.assertEqual(self.bus.reg[42], 500); self.assert_low_pending()
        before = list(self.bus.events)
        with self.assertRaises(m.RecoveryRequired): self.session.recover()
        self.assertEqual(self.bus.events, before)

    def test_open_measurement_does_not_accept_position_without_matching_goal_or_moving_clear(self):
        for mismatch in ('goal', 'moving'):
            with self.subTest(mismatch=mismatch):
                self.setUp(); self.contact(); self.start_release()
                self.bus.reg.update({42:900, 56:900, 66:0})
                self.bus.reg[42 if mismatch == 'goal' else 66] = 800 if mismatch == 'goal' else 1
                with self.assertRaises(m.SessionFailure): self.session.verify_release_open(self.token)
                self.assert_low_pending()

    def test_release_evidence_rejects_each_wrong_source_binding_and_stale_time(self):
        changes = [dict(source='d435'), dict(empty_verified=1), dict(session_id='b'*32),
                   dict(release_id='c'*32), dict(camera_generation='wrist-restarted'),
                   dict(sequence=100), dict(sequence=True), dict(capture_monotonic=19.),
                   dict(capture_monotonic=1e6), dict(capture_monotonic=float('nan')),
                   dict(frame_sha256='g'*64)]
        for change in changes:
            with self.subTest(change=change):
                self.setUp(); self.contact(); evidence = self.open_and_observe(); evidence.update(change)
                before = self.bus.writes()
                with self.assertRaises(m.RecoveryRequired): self.session.confirm_released(self.token, evidence)
                self.assertEqual(self.bus.writes(), before); self.assert_low_pending()

    def test_good_image_cannot_override_jaw_that_moved_after_image(self):
        self.contact(); evidence = self.open_and_observe(); self.bus.reg[56] = 890
        with self.assertRaises(m.SessionFailure): self.session.confirm_released(self.token, evidence)
        self.assert_low_pending()

    def test_old_inference_result_cannot_clear_contact(self):
        self.contact(); evidence = self.open_and_observe(); self.bus.now += 2.01
        with self.assertRaises(m.RecoveryRequired): self.session.confirm_released(self.token, evidence)
        self.assert_low_pending()

    def test_release_clear_journal_failure_never_restores(self):
        self.contact(); evidence = self.open_and_observe(); put = self.journal.put
        def failure(record):
            if not record['contact_possible']: raise OSError('cannot persist release')
            put(record)
        self.journal.put = failure
        with self.assertRaises(m.SessionFailure): self.session.confirm_released(self.token, evidence)
        self.assert_low_pending()

    def test_verified_release_is_durable_before_neutralize_and_restore(self):
        self.contact(); evidence = self.open_and_observe()
        result = self.session.confirm_released(self.token, evidence)
        self.assertTrue(result['released']); self.assertFalse(result['contact_possible'])
        self.assertEqual(self.bus.writes(), [(6,48,100), (6,42,900), (6,42,900), (6,48,500)])
        receipt = self.journal.get()['release_receipt']
        self.assertEqual(receipt['frame_sha256'], evidence['frame_sha256'])
        self.assertGreater(receipt['capture_monotonic'], receipt['stationary_monotonic'])
        self.session.assert_motion_allowed(); before = list(self.bus.events)
        self.assertTrue(self.session.finish()['already_completed']); self.assertEqual(self.bus.events, before)

    def test_restore_ack_loss_after_real_release_still_needs_explicit_recovery(self):
        self.contact(); evidence = self.open_and_observe()
        self.bus.ack_loss = lambda a,v: 'after' if a == 48 and v == 500 else None
        with self.assertRaises(m.SessionFailure): self.session.confirm_released(self.token, evidence)
        self.assertEqual(self.bus.reg[48], 500)
        self.assertFalse(self.journal.get()['contact_possible'])
        self.assertIn('release_receipt', self.journal.get())
        before = list(self.bus.events)
        with self.assertRaises(m.RecoveryRequired): self.session.finish()
        self.assertEqual(self.bus.events, before)
        self.bus.ack_loss = None; fresh = self.new_session(); fresh.recover()
        self.assertEqual(self.bus.writes()[-3:], [(6,48,100), (6,42,900), (6,48,500)])

    def test_explicit_recovery_only_authorizes_opening_and_requires_new_release_evidence(self):
        self.contact(); old_evidence = self.open_and_observe(); fresh = self.new_session()
        challenge = fresh.recover_for_release(camera_generation='wrist-boot-A', frame_sequence=110)
        self.assertNotEqual(challenge['release_id'], old_evidence['release_id'])
        self.assertTrue(self.bus.stop)
        before = list(self.bus.events)
        with self.assertRaises(m.RecoveryRequired): fresh.assert_motion_allowed(challenge['owner_token'])
        self.assertEqual(self.bus.events, before); self.assert_low_pending()

    def test_explicit_release_recovery_has_no_cap_write_or_goal_and_refuses_fault_or_changed_cap(self):
        for address,value in ((65,1),(48,500)):
            with self.subTest(address=address):
                self.setUp(); self.contact(); self.bus.reg[address] = value
                fresh = self.new_session(); before = self.bus.writes()
                with self.assertRaises(m.SessionFailure):
                    fresh.recover_for_release(camera_generation='wrist-boot-A', frame_sequence=100)
                self.assertEqual(self.bus.writes(), before)
                self.assertTrue(self.journal.get()['contact_possible'])

    def test_missing_flag_or_forged_release_cannot_turn_pending_contact_into_empty(self):
        for mutation in ('remove', 'false', 'legacy'):
            with self.subTest(mutation=mutation):
                self.setUp(); self.contact(); record = self.journal.get()
                if mutation == 'remove': del record['contact_possible']
                elif mutation == 'false': record.update(contact_possible=False, phase='lowered')
                else: record['version'] = 1
                self.journal.put(record); before = list(self.bus.events)
                with self.assertRaises(m.RecoveryRequired): self.new_session().recover()
                self.assertEqual(self.bus.events, before)


if __name__ == '__main__': unittest.main(verbosity=2)
