"""Real Robot retry/power state over a fake SDK; no ports, camera or network."""
from collections import deque
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scservo_sdk import COMM_SUCCESS, COMM_RX_TIMEOUT, COMM_RX_CORRUPT
from server.robot import JOINTS, Robot, MotorReadError, READ_RETRY_DELAY_S


class Protocol:
    def __init__(self,robot):
        self.robot=robot;self.events=[];self.failures={};self.on_read=None
        self.registers={i:{3:777,33:0,40:1,42:1800,48:500,56:1800,
                           60:10,62:120,63:30,65:0,66:0,69:20} for i in range(1,7)}
        self.write_result=(COMM_SUCCESS,0)
    def read(self,port,servo,address):
        assert self.robot.io._is_owned(), 'Read retry escaped serial ownership'
        self.events.append(('read',servo,address))
        if self.on_read:self.on_read(servo,address)
        pending=self.failures.get((servo,address))
        if pending:return pending.popleft()
        return self.registers[servo][address],COMM_SUCCESS,0
    read1ByteTxRx=read
    read2ByteTxRx=read
    def write(self,port,servo,address,value):
        self.events.append(('write',servo,address,value))
        return self.write_result
    write1ByteTxRx=write
    write2ByteTxRx=write


class RobotTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        cal={name:{'id':i+1,'range_min':0,'range_max':4095} for i,name in enumerate(JOINTS)}
        self.r=Robot({'follower_port':'fake://no-hardware'},cal,self.tmp.name,Mock())
        self.r.connected=True;self.r.models=dict.fromkeys(JOINTS,777)
        self.reset_input=Mock(side_effect=lambda:self.assertTrue(self.r.io._is_owned()))
        self.r.port=SimpleNamespace(ser=SimpleNamespace(reset_input_buffer=self.reset_input))
        self.bus=Protocol(self.r);self.r.ph=self.bus
        self.sleep=Mock(side_effect=lambda delay:self.assertTrue(self.r.io._is_owned()))
        guard=patch('server.robot.time.sleep',self.sleep);guard.start();self.addCleanup(guard.stop)

    def fail(self,servo,address,*outcomes):
        self.bus.failures[servo,address]=deque(outcomes)

    def count(self,servo,address):
        return self.bus.events.count(('read',servo,address))

    def active_hold(self):
        # The real persisted transaction stays pending if communication fails.
        self.r.gripper_power.state='holding';self.r.pwm_owner='transport-test-owner'
        self.r.gripper_power.journal.put({'phase':'holding','session_id':'transport-test-owner',
            'device_key':'fake://no-hardware:6','contact_possible':True})

    def assert_pending(self):
        self.assertEqual(self.r.gripper_power.state,'blocked')
        self.assertTrue(self.r.stop.is_set())
        self.assertEqual(self.r.gripper_power.journal.get()['phase'],'holding')
        self.assertFalse(any(e[0]=='write' for e in self.bus.events))

    def test_missing_then_corrupt_reply_recovers_with_fresh_signed_position(self):
        self.fail(4,56,(999,COMM_RX_TIMEOUT,0),(777,COMM_RX_CORRUPT,0))
        self.bus.registers[4][56]=0x8005
        self.assertEqual(self.r._read('wrist_flex',56),-5)
        self.assertEqual(self.count(4,56),3)
        self.assertEqual([c.args for c in self.sleep.call_args_list],[(READ_RETRY_DELAY_S,)]*2)
        self.assertEqual(self.reset_input.call_count,2)
        self.assertFalse(self.r.stop.is_set())
        self.assertFalse(any(e[0]=='write' for e in self.bus.events))

    def test_exhausted_read_has_three_attempts_and_preserves_pending_hold(self):
        self.active_hold();self.fail(4,56,*[(0,COMM_RX_TIMEOUT,0)]*4)
        with self.assertRaisesRegex(MotorReadError,r'wrist_flex.*-6/0'):
            self.r._read('wrist_flex',56)
        self.assertEqual(self.count(4,56),3);self.assertEqual(self.sleep.call_count,2)
        self.assertEqual(len(self.bus.failures[4,56]),1)
        self.assert_pending()

    def test_servo_errors_and_other_transport_errors_never_retry(self):
        for result,error in ((COMM_SUCCESS,32),(COMM_RX_TIMEOUT,8),(COMM_RX_CORRUPT,1),(-1,0),(-2,0),(-5,0)):
            with self.subTest(result=result,error=error):
                self.bus.events.clear();self.sleep.reset_mock()
                self.fail(1,65,(0,result,error))
                with self.assertRaises(MotorReadError):self.r._read('shoulder_pan',65,1)
                self.assertEqual(self.count(1,65),1);self.sleep.assert_not_called()

    def test_write_ack_loss_is_never_retried_even_for_retryable_read_codes(self):
        for result in (COMM_RX_TIMEOUT,COMM_RX_CORRUPT):
            with self.subTest(result=result):
                self.bus.events.clear();self.sleep.reset_mock();self.bus.write_result=(result,0)
                with self.assertRaisesRegex(RuntimeError,'寫入失敗'):
                    self.r._write('gripper',48,100)
                self.assertEqual(self.bus.events,[('write',6,48,100)])
                self.sleep.assert_not_called()

    def test_fresh_fault_after_retry_still_blocks_goal_write(self):
        self.fail(6,65,(0,COMM_RX_TIMEOUT,0));self.bus.registers[6][65]=32
        with self.assertRaisesRegex(RuntimeError,'保護狀態'):
            self.r._write('gripper',42,1850)
        self.assertEqual(self.count(6,65),2)
        self.assertFalse(any(e[0]=='write' for e in self.bus.events))

    def test_buffer_reset_happens_before_each_fresh_read_under_same_lock(self):
        self.fail(4,56,(1234,COMM_RX_TIMEOUT,0))
        order=[]
        def read_event(servo,address):
            order.append('read')
            if self.count(servo,address)==2:self.assertEqual(order,['read','wait','reset','read'])
        self.bus.on_read=read_event
        self.sleep.side_effect=lambda delay:order.append('wait')
        def clear():
            self.assertTrue(self.r.io._is_owned());order.append('reset')
            self.bus.registers[4][56]=2100
        self.reset_input.side_effect=clear
        self.assertEqual(self.r._read('wrist_flex',56),2100)
        self.assertEqual(order,['read','wait','reset','read'])

    def test_failed_buffer_reset_does_not_retry_and_preserves_pending(self):
        self.active_hold();self.fail(4,56,(0,COMM_RX_TIMEOUT,0))
        self.reset_input.side_effect=OSError('fake disconnected port')
        with self.assertRaisesRegex(OSError,'disconnected'):self.r._read('wrist_flex',56)
        self.assertEqual(self.count(4,56),1);self.assert_pending()

    def test_failed_joint_batch_does_not_publish_partial_or_cached_positions(self):
        self.r.position=dict.fromkeys(JOINTS,1111)
        self.fail(4,56,*[(0,COMM_RX_CORRUPT,0)]*3)
        with self.assertRaises(MotorReadError):self.r.read()
        self.assertEqual(self.r.position,dict.fromkeys(JOINTS,1111))
        for registers in self.bus.registers.values():registers[56]=2222
        self.assertEqual(self.r.read(),dict.fromkeys(JOINTS,2222))

    def test_telemetry_discards_prefailure_registers_and_next_call_reads_everything_fresh(self):
        self.fail(3,56,(0,COMM_RX_TIMEOUT,0))
        def update_after_fault(servo,address):
            if (servo,address)==(3,56) and self.count(3,56)==1:
                for registers in self.bus.registers.values():
                    registers.update({56:1900,60:70,69:80})
        self.bus.on_read=update_after_fault
        with self.assertRaises(MotorReadError):self.r.gripper_telemetry()
        self.assertEqual(self.count(3,56),1);self.sleep.assert_not_called()
        sample=self.r.gripper_telemetry()
        self.assertEqual(sample['position'],1900)
        self.assertEqual(sample['load_raw'],70);self.assertEqual(sample['current_raw'],80)
        self.assertEqual(set(sample['body_position'].values()),{1900})
        self.assertEqual(self.count(6,60),2);self.assertEqual(self.count(3,56),2)
        # The failed call returns no sample; a new call checks status at both ends.
        self.assertEqual(self.count(6,65),3);self.sleep.assert_not_called()
        self.assertLessEqual(sample['read_start_monotonic'],sample['capture_monotonic'])
        self.assertLessEqual(sample['capture_monotonic'],sample['read_end_monotonic'])

    def test_contact_telemetry_is_fail_fast_without_nested_retry_budget(self):
        self.active_hold();self.fail(6,69,*[(0,COMM_RX_TIMEOUT,0)]*9)
        with self.assertRaises(MotorReadError):self.r.gripper_telemetry()
        self.assertEqual(self.count(6,69),1);self.assertEqual(self.count(6,60),1)
        self.assertEqual(self.count(6,65),1);self.sleep.assert_not_called()
        self.assert_pending()

    def test_telemetry_servo_error_fails_immediately_and_keeps_pending(self):
        self.active_hold();self.fail(6,69,(0,COMM_SUCCESS,32))
        with self.assertRaises(MotorReadError):self.r.gripper_telemetry()
        self.assertEqual(self.count(6,69),1);self.assertEqual(self.count(6,65),1)
        self.sleep.assert_not_called();self.assert_pending()

    def test_fault_appearing_during_telemetry_cannot_be_returned_as_healthy(self):
        self.active_hold()
        def trip_protection(servo,address):
            if (servo,address)==(5,56):self.bus.registers[6][65]=32
        self.bus.on_read=trip_protection
        with self.assertRaisesRegex(RuntimeError,'保護狀態'):self.r.gripper_telemetry()
        self.assertEqual(self.count(6,65),2);self.sleep.assert_not_called();self.assert_pending()


if __name__=='__main__':unittest.main()
