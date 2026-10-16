"""Stateful fake-servo tests; all journal persistence uses a temporary directory."""
import json,pathlib,tempfile,threading,unittest
from server import gripper_power as m

class FakeBus:
    def __init__(self,journal):
        self.reg={3:777,33:0,65:0,48:500,56:300,42:900,66:0}
        self.journal=journal;self.events=[];self.stop=False;self.now=0.
        self.write_failure=None;self.read_failure=None;self.fault_when_position_read=False
    def read(self,servo,address,width):
        self.events.append(('read',servo,address,self.reg[address]))
        if self.read_failure and self.read_failure(self,address):raise IOError('injected read failure')
        value=self.reg[address]
        if address==56 and self.fault_when_position_read:self.reg[65]=1
        return value
    def write(self,servo,address,value,width):
        self.events.append(('write',servo,address,value))
        # Independent physical invariants: all modifications are gripper RAM;
        # a goal must never be issued under original power or protected status.
        if servo!=6 or address not in (42,48):raise AssertionError('wrong servo/register')
        record=self.journal.get()
        if not record or record['phase']=='completed':raise AssertionError('write without durable pending journal')
        if address==42 and (self.reg[48]!=100 or self.reg[65]):raise AssertionError('unsafe goal')
        if address==48 and value>self.reg[48]:
            if self.reg[42]!=self.reg[56] or self.reg[66] or self.reg[65]:
                raise AssertionError('power increased with outstanding/moving/faulted goal')
        if self.write_failure:
            outcome=self.write_failure(self,address,value)
            if outcome=='before':raise IOError('injected write failed before application')
            if outcome=='after':self.reg[address]=value;raise IOError('injected acknowledgement loss')
        self.reg[address]=value
    def latch(self):self.stop=True
    def sleep(self,seconds):self.now+=seconds
    def writes(self):return [e[1:] for e in self.events if e[0]=='write']

class PWMSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=pathlib.Path(self.tmp.name)/'pending.json';self.journal=m.AtomicJournal(self.path)
        self.bus=FakeBus(self.journal);self.lock=threading.RLock();self.session=self.new_session()
    def new_session(self):
        return m.GripperPWMSession(read=self.bus.read,write=self.bus.write,lock=self.lock,
            stop_latch=self.bus.latch,journal=self.journal,device_key='verified-controller-A',
            position_bounds=(100,1100),clock=lambda:self.bus.now,sleep=self.bus.sleep)
    def assert_pending(self):
        self.assertNotEqual(self.journal.get()['phase'],'completed');self.assertTrue(self.bus.stop)
    def test_normal_cleanup_neutralizes_before_restoring_and_only_touches_gripper_ram(self):
        token=self.session.enter();self.session.assert_motion_allowed(token)
        self.assertEqual(self.bus.reg[48],100)
        # Outstanding goal deliberately differs from actual position, as after a
        # stopped empty-cycle command. Restoration must first replace it.
        self.bus.reg[42]=950
        result=self.session.finish()
        self.assertEqual(self.bus.writes(),[(6,48,100),(6,42,300),(6,48,500)])
        self.assertEqual(self.bus.reg[42],self.bus.reg[56]);self.assertEqual(result['restored_cap'],500)
        self.assertEqual(self.journal.get()['phase'],'completed');self.session.assert_motion_allowed()
    def test_persistence_failure_prevents_first_motor_write(self):
        self.journal.put=lambda record:(_ for _ in ()).throw(OSError('disk unavailable'))
        with self.assertRaises(m.SessionFailure):self.session.enter()
        self.assertEqual(self.bus.writes(),[]);self.assertTrue(self.bus.stop)
    def test_limit_write_failure_before_or_after_application_keeps_pending(self):
        for mode in ('before','after'):
            with self.subTest(mode=mode):
                self.setUp();self.bus.write_failure=lambda b,a,v:mode if a==48 and v==100 else None
                with self.assertRaises(m.SessionFailure):self.session.enter()
                self.assert_pending();self.assertFalse(any(a==42 for _,a,_ in self.bus.writes()))
                self.assertFalse(any(a==48 and v==500 for _,a,v in self.bus.writes()))
    def test_low_limit_read_failure_cannot_restore(self):
        self.bus.read_failure=lambda b,a:a==48 and b.reg[48]==100
        with self.assertRaises(m.SessionFailure):self.session.enter()
        self.assert_pending();self.assertEqual(self.bus.writes(),[(6,48,100)])
    def test_fault_during_cleanup_prohibits_any_goal_or_restoration(self):
        self.session.enter();self.bus.reg[65]=1
        with self.assertRaises(m.SessionFailure):self.session.finish()
        self.assert_pending();self.assertEqual(self.bus.writes(),[(6,48,100)])
    def test_fault_appearing_after_position_read_is_checked_before_goal(self):
        self.session.enter();self.bus.fault_when_position_read=True
        with self.assertRaises(m.SessionFailure):self.session.finish()
        self.assert_pending();self.assertEqual(self.bus.writes(),[(6,48,100)])
    def test_neutral_goal_write_failure_does_not_raise_cap(self):
        self.session.enter();self.bus.write_failure=lambda b,a,v:'after' if a==42 else None
        with self.assertRaises(m.SessionFailure):self.session.finish()
        self.assert_pending();self.assertEqual(self.bus.reg[48],100)
        self.assertEqual(self.bus.writes(),[(6,48,100),(6,42,300)])
    def test_unsettled_position_prevents_restore(self):
        self.session.enter();self.bus.reg[66]=1
        with self.assertRaises(m.SessionFailure):self.session.finish()
        self.assert_pending();self.assertEqual(self.bus.reg[48],100)
        self.assertFalse(any(a==48 and v==500 for _,a,v in self.bus.writes()))
    def test_neutral_goal_read_failure_prevents_restore(self):
        self.session.enter();self.bus.read_failure=lambda b,a:a==42
        with self.assertRaises(m.SessionFailure):self.session.finish()
        self.assert_pending();self.assertEqual(self.bus.reg[48],100)
    def test_restoration_ack_loss_needs_explicit_recovery_and_relowering(self):
        self.session.enter();self.bus.write_failure=lambda b,a,v:'after' if a==48 and v==500 else None
        with self.assertRaises(m.SessionFailure):self.session.finish()
        self.assert_pending();self.assertEqual(self.bus.reg[48],500)
        before=list(self.bus.writes())
        with self.assertRaises(m.RecoveryRequired):self.session.finish()
        self.assertEqual(self.bus.writes(),before)
        self.bus.write_failure=None
        self.session.recover()
        self.assertEqual(self.bus.writes()[len(before):],[(6,48,100),(6,42,300),(6,48,500)])
        self.assertTrue(self.bus.stop)
    def test_restoration_read_failure_leaves_pending(self):
        self.session.enter()
        self.bus.read_failure=lambda b,a:a==48 and b.reg[48]==500
        with self.assertRaises(m.SessionFailure):self.session.finish()
        self.assert_pending();self.assertEqual(self.journal.get()['phase'],'restoring')
    def test_completion_journal_failure_keeps_barrier_after_safe_restore(self):
        self.session.enter();original_put=self.journal.put
        def failing_put(record):
            if record['phase']=='completed':raise OSError('journal unavailable')
            original_put(record)
        self.journal.put=failing_put
        with self.assertRaises(m.SessionFailure):self.session.finish()
        self.assert_pending();self.assertEqual(self.bus.reg[48],500)
        self.assertEqual(self.bus.reg[42],self.bus.reg[56])
        with self.assertRaises(m.RecoveryRequired):self.new_session().assert_motion_allowed()

    def test_incomplete_completed_receipt_cannot_bypass_startup_barrier(self):
        self.session.enter();record=self.journal.get();record['phase']='completed';self.journal.put(record)
        before=list(self.bus.events)
        with self.assertRaises(m.RecoveryRequired):self.new_session().assert_motion_allowed()
        self.assertEqual(self.bus.events,before)

    def test_repeated_successful_cleanup_has_no_hardware_side_effects(self):
        self.session.enter();self.session.finish();before=list(self.bus.events)
        self.assertTrue(self.session.finish()['already_completed'])
        self.assertEqual(self.bus.events,before)
    def test_context_cancellation_still_neutralizes_before_restoration(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.session:raise KeyboardInterrupt('user stopped')
        self.assertEqual(self.bus.writes(),[(6,48,100),(6,42,300),(6,48,500)])
        self.assertTrue(self.bus.stop);self.assertEqual(self.journal.get()['phase'],'completed')
    def test_body_communication_error_is_not_treated_as_cancel_cleanup(self):
        with self.assertRaises(IOError):
            with self.session:raise IOError('telemetry transport failed')
        self.assert_pending();self.assertEqual(self.bus.writes(),[(6,48,100)])
        before=list(self.bus.events)
        with self.assertRaises(m.RecoveryRequired):self.session.finish(cancelled=True)
        self.assertEqual(self.bus.events,before)

    def test_explicit_stop_exception_uses_safe_cleanup(self):
        with self.assertRaises(m.SessionCancelled):
            with self.session:raise m.SessionCancelled('user stop')
        self.assertEqual(self.bus.writes(),[(6,48,100),(6,42,300),(6,48,500)])
        self.assertTrue(self.bus.stop)

    def test_crash_journal_blocks_new_begin_resume_without_hardware_reads(self):
        self.session.enter();replacement=self.new_session();before=list(self.bus.events)
        with self.assertRaises(m.RecoveryRequired):replacement.assert_motion_allowed()
        with self.assertRaises(m.RecoveryRequired):replacement.finish()
        with self.assertRaises(m.RecoveryRequired):replacement.enter()
        self.assertEqual(self.bus.events,before)
        replacement.recover()
        self.assertEqual(self.bus.writes()[-2:],[(6,42,300),(6,48,500)])
    def test_corrupt_or_wrong_identity_journal_blocks_without_writes(self):
        self.path.write_text('{broken')
        with self.assertRaises(m.RecoveryRequired):self.session.assert_motion_allowed()
        self.assertEqual(self.bus.events,[])
        self.path.unlink();self.session.enter();record=self.journal.get();record['device_key']='another-controller'
        self.journal.put(record);before=list(self.bus.events)
        with self.assertRaises(m.RecoveryRequired):self.new_session().recover()
        self.assertEqual(self.bus.events,before)
    def test_unexpected_limit_or_model_change_is_not_implicitly_recovered(self):
        self.session.enter();self.bus.reg[48]=200;before=list(self.bus.writes())
        with self.assertRaises(m.SessionFailure):self.session.recover()
        self.assert_pending();self.assertEqual(self.bus.writes(),before)
        self.bus.reg[48]=100;self.bus.reg[3]=776
        with self.assertRaises(m.SessionFailure):self.session.recover()
        self.assertEqual(self.bus.writes(),before)

if __name__=='__main__':unittest.main(verbosity=2)
