"""Real empty-jaw opening and low-PWM session on a synthetic register bus."""
import unittest
from unittest.mock import patch

from server.gripper_power import SessionCancelled, SessionFailure
from server.local_cartesian import NAMES, default_model
import test_wave as wave_helpers
import test_wave_tolerance as controller_helpers


class ActiveOpenTests(unittest.TestCase):
    model=default_model()
    capture=wave_helpers.WaveTests.capture

    def setUp(self):
        controller_helpers.WaveEndpointToleranceTests.setUp(self)
        self.bus.reg[6].update({3:777,33:0,48:500,42:1531,56:1531})
        self.robot.goals['gripper']=1531
        self.robot.bias['gripper']=0
        self.initial['gripper']=1531
        self.initial_holds['gripper']=1531
        for name,entry in self.robot.calibration.items():
            self.bus.offset[entry['id']]=self.initial[name]-self.initial_holds[name]
        self.original_read=self.robot.gripper_power.read
        self.original_write=self.robot.gripper_power.write

    def open(self,**kwargs):
        return self.grasp.open_empty(self.capture(),self.reports.append,**kwargs)

    def assert_restored_callbacks(self):
        self.assertIs(self.robot.gripper_power.read,self.original_read)
        self.assertIs(self.robot.gripper_power.write,self.original_write)

    def assert_no_power_restore(self):
        self.assertNotIn((6,48,500),self.bus.events)
        self.assert_restored_callbacks()

    def test_existing_call_shape_streams_open_and_preserves_body_and_low_session(self):
        frame=self.open()
        self.assertEqual(frame['joints']['gripper'],1860)
        self.assertEqual(self.bus.reg[6][48],100)
        self.assertIsNotNone(self.robot.pwm_owner)
        self.assertEqual(self.robot.gripper_power.state,'active')
        self.assertFalse(self.robot.gripper_power._record()['contact_possible'])
        self.assertGreater(len([e for e in self.bus.events if e[:2]==(6,42)]),100)
        self.assertEqual({e[:2] for e in self.bus.events},{(6,48),(6,42)})
        for name in NAMES:
            self.assertEqual(self.robot.read()[name],self.initial[name])
            self.assertEqual(self.robot.goals[name],self.initial_holds[name])
        self.assert_no_power_restore()

    def test_external_guard_only_sees_measured_pose_throughout_open(self):
        seen=[]
        def guard(actual):
            self.assertEqual(actual,self.robot.read())
            seen.append(dict(actual))
        frame=self.open(cancel=self.cancel,guard=guard)
        self.assertGreater(len(seen),100)
        self.assertEqual(seen[-1],frame['joints'])
        self.assertEqual(seen[0]['gripper'],1531)
        self.assertEqual(seen[-1]['gripper'],1860)
        self.assert_no_power_restore()

    def test_cancel_before_entry_performs_no_power_or_goal_writes(self):
        self.cancel.set()
        with self.assertRaises(SessionCancelled):self.open(cancel=self.cancel)
        self.assertEqual(self.bus.events,[])
        self.assertIsNone(self.robot.gripper_power._record())
        self.assert_restored_callbacks()

    def test_cancel_during_journal_publication_prevents_the_pending_cap_write(self):
        power=self.robot.gripper_power;original=power.journal.put
        def journal(record):
            original(record)
            if record['phase']=='prepared':self.cancel.set()
        with patch.object(power.journal,'put',side_effect=journal):
            with self.assertRaises(SessionFailure):self.open(cancel=self.cancel)
        self.assertNotIn((6,48,100),self.bus.events)
        self.assertFalse(any(e[:2]==(6,42) for e in self.bus.events))
        self.assertEqual(power._record()['phase'],'prepared')
        self.assertTrue(self.robot.stop.is_set())
        self.assert_no_power_restore()

    def test_cancel_after_cap_write_retains_low_cap_and_never_starts_opening(self):
        power=self.robot.gripper_power;original=power.write
        def write(servo,address,value,width):
            result=original(servo,address,value,width)
            if address==48 and value==100:self.cancel.set()
            return result
        power.write=write;self.original_write=write
        with self.assertRaises(SessionFailure):self.open(cancel=self.cancel)
        self.assertEqual(self.bus.reg[6][48],100)
        self.assertFalse(any(e[:2]==(6,42) for e in self.bus.events))
        self.assertTrue(self.robot.stop.is_set())
        self.assert_no_power_restore()

    def test_table_guard_during_power_entry_prevents_cap_write(self):
        power=self.robot.gripper_power;original=power.journal.put;changed=False
        def journal(record):
            nonlocal changed
            original(record);changed=True
        def guard(actual):
            if changed:raise RuntimeError('measured table corridor changed')
        with patch.object(power.journal,'put',side_effect=journal):
            with self.assertRaisesRegex(SessionFailure,'table corridor'):
                self.open(cancel=self.cancel,guard=guard)
        self.assertNotIn((6,48,100),self.bus.events)
        self.assertFalse(any(e[:2]==(6,42) for e in self.bus.events))
        self.assert_no_power_restore()

    def test_camera_reconnect_during_motor_interpolation_stops_at_low_cap(self):
        original=self.robot._write;opening=[]
        def write(name,address,value,*args):
            result=original(name,address,value,*args)
            if name=='gripper' and address==42:
                opening.append(value)
                if len(opening)==5:self.robot.camera.generation='reconnected'
            return result
        self.robot._write=write
        with self.assertRaisesRegex(RuntimeError,'重新連接'):self.open(cancel=self.cancel)
        self.assertEqual(len(opening),5)
        self.assertEqual(self.bus.reg[6][48],100)
        self.assertTrue(self.robot.stop.is_set())
        self.assert_no_power_restore()

    def test_job_cancel_during_opening_does_not_wait_for_target(self):
        original=self.robot._write;opening=[]
        def write(name,address,value,*args):
            result=original(name,address,value,*args)
            if name=='gripper' and address==42:
                opening.append(value)
                if len(opening)==5:self.cancel.set()
            return result
        self.robot._write=write
        with self.assertRaises(SessionCancelled):self.open(cancel=self.cancel)
        self.assertEqual(len(opening),5)
        self.assertLess(self.bus.reg[6][56],1860)
        self.assertEqual(self.bus.reg[6][48],100)
        self.assert_no_power_restore()

    def test_after_capture_cancel_does_not_return_success_or_finish_session(self):
        def capture_hook(frame,index):
            if index==3:self.cancel.set()
        self.capture_hook=capture_hook
        with self.assertRaises(SessionCancelled):self.open(cancel=self.cancel)
        self.assertEqual(self.bus.reg[6][56],1860)
        self.assertFalse(any(event.get('type')=='opened' for event in self.events))
        self.assert_no_power_restore()

    def test_after_capture_measured_guard_failure_is_not_an_opened_receipt(self):
        final=False
        def capture_hook(frame,index):
            nonlocal final
            if index==3:final=True
        self.capture_hook=capture_hook
        def guard(actual):
            if final:raise RuntimeError('final measured corridor rejected')
        with self.assertRaisesRegex(RuntimeError,'final measured'):
            self.open(cancel=self.cancel,guard=guard)
        self.assertFalse(any(event.get('type')=='opened' for event in self.events))
        self.assert_no_power_restore()


if __name__=='__main__':unittest.main()
