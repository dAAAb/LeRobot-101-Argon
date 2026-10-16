"""Installed SDK against a byte-level fake port; no robot/camera/serial access."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scservo_sdk import COMM_PORT_BUSY, COMM_SUCCESS, PacketHandler
from server.robot import JOINTS, MotorReadError, Robot


class FaultPort:
    """Exercise the actual SDK's busy flag, packet parser and exception paths."""
    def __init__(self,robot=None):
        self.robot=robot;self.is_using=False;self.rx=deque();self.packets=[]
        self.fail_stage=None;self.failed=0;self.status=0;self.read_hook=None
        self.ser=SimpleNamespace(reset_input_buffer=Mock())
    def stage(self,name):
        if self.robot is not None:
            assert self.robot.io._is_owned(), 'SDK operation escaped serial lock'
        if self.fail_stage==name:
            self.fail_stage=None;self.failed+=1
            raise OSError('injected serial '+name)
    def clearPort(self):self.stage('flush')
    def writePort(self,packet):
        self.stage('write');self.packets.append(list(packet));servo=packet[2]
        if packet[4]==2:
            address,width=packet[5:7];value=self.status if address==65 else 1800
            data=[(value>>(8*i))&255 for i in range(width)]
        else:data=[]
        reply=[255,255,servo,len(data)+2,0]+data
        reply.append((~sum(reply[2:]))&255);self.rx=deque(reply)
        return len(packet)
    def setPacketTimeout(self,length):self.stage('timeout_setup')
    def isPacketTimeout(self):return not self.rx
    def readPort(self,length):
        self.stage('read')
        if self.read_hook is not None:self.read_hook()
        return [self.rx.popleft() for _ in range(min(length,len(self.rx)))]


class SDKOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        cal={n:{'id':i+1,'range_min':0,'range_max':4095} for i,n in enumerate(JOINTS)}
        self.robot=Robot({'follower_port':'fake://no-hardware'},cal,self.tmp.name,Mock())
        self.robot.connected=True;self.robot.models=dict.fromkeys(JOINTS,777)
        self.robot.ph=PacketHandler(0);self.port=FaultPort(self.robot);self.robot.port=self.port

    def test_installed_sdk_without_wrapper_latches_busy_after_serial_exception(self):
        for stage in ('flush','write','timeout_setup','read'):
            with self.subTest(stage=stage):
                port=FaultPort();port.fail_stage=stage;sdk=PacketHandler(0)
                with self.assertRaisesRegex(OSError,stage):sdk.read2ByteTxRx(port,1,56)
                self.assertTrue(port.is_using)
                self.assertEqual(sdk.read2ByteTxRx(port,1,56),(0,COMM_PORT_BUSY,0))
                self.assertTrue(port.is_using)

    def test_read_exception_releases_owned_flag_without_retry_or_motor_write(self):
        self.robot.stop.set();self.robot.error='previous safety failure'
        for stage in ('flush','write','timeout_setup','read'):
            with self.subTest(stage=stage):
                self.port.fail_stage=stage;self.port.packets.clear()
                with self.assertRaisesRegex(OSError,stage):self.robot._read('shoulder_pan',56)
                self.assertFalse(self.port.is_using)
                self.assertIsNone(self.robot._sdk_transaction_owner)
                self.assertLessEqual(len(self.port.packets),1)
                self.assertEqual(self.robot._read_once('shoulder_pan',56),1800)
                self.assertTrue(all(p[4]==2 for p in self.port.packets))
                self.port.ser.reset_input_buffer.assert_not_called()
                self.assertTrue(self.robot.stop.is_set())
                self.assertEqual(self.robot.error,'previous safety failure')

    def test_lost_write_ack_is_not_reissued_after_finally_cleanup(self):
        self.port.fail_stage='read'
        with self.assertRaisesRegex(OSError,'read'):self.robot._write('wrist_roll',48,123)
        self.assertFalse(self.port.is_using)
        self.assertEqual(len(self.port.packets),1)
        self.assertEqual(self.port.packets[0][4:7],[3,48,123])
        self.assertEqual(self.robot._read('wrist_roll',56),1800)
        self.assertEqual(sum(p[4]==3 for p in self.port.packets),1)
        self.port.ser.reset_input_buffer.assert_not_called()

    def test_preexisting_busy_flag_is_not_cleared_or_retried(self):
        self.port.is_using=True
        with self.assertRaises(MotorReadError) as caught:self.robot._read('elbow_flex',56)
        self.assertFalse(caught.exception.retryable)
        with self.assertRaisesRegex(RuntimeError,'寫入失敗'):self.robot._write('wrist_roll',48,123)
        self.assertTrue(self.port.is_using);self.assertEqual(self.port.packets,[])
        self.assertIsNone(self.robot._sdk_transaction_owner)
        self.port.ser.reset_input_buffer.assert_not_called()

    def test_reentrant_sdk_call_cannot_steal_outer_transaction(self):
        observed=[]
        def nested():
            self.port.read_hook=None
            owner=self.robot._sdk_transaction_owner
            with self.assertRaises(MotorReadError):self.robot._read_once('elbow_flex',56)
            self.assertIs(self.robot._sdk_transaction_owner,owner)
            observed.append(self.port.is_using)
        self.port.read_hook=nested
        self.assertEqual(self.robot._read_once('shoulder_pan',56),1800)
        self.assertEqual(observed,[True]);self.assertEqual(len(self.port.packets),1)
        self.assertFalse(self.port.is_using)

    def test_threaded_write_waits_for_read_and_never_clears_its_flag(self):
        entered=threading.Event();release=threading.Event();write_started=threading.Event()
        def blocked_read():
            self.port.read_hook=None;entered.set()
            if not release.wait(2):raise RuntimeError('test read not released')
        def write():
            write_started.set();return self.robot._write('wrist_roll',48,123)
        self.port.read_hook=blocked_read
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(self.robot._read_once,'shoulder_pan',56)
            self.assertTrue(entered.wait(1));second=pool.submit(write)
            try:
                self.assertTrue(write_started.wait(1))
                self.assertFalse(second.done());self.assertTrue(self.port.is_using)
                self.assertEqual(len(self.port.packets),1)
            finally:release.set()
            self.assertEqual(first.result(timeout=2),1800);second.result(timeout=2)
        self.assertEqual([p[4] for p in self.port.packets],[2,3])
        self.assertFalse(self.port.is_using)

    def test_cleanup_does_not_clear_servo_fault_or_allow_goal_write(self):
        self.port.status=32;self.port.fail_stage='flush'
        with self.assertRaises(OSError):self.robot._read('wrist_roll',56)
        with self.assertRaisesRegex(RuntimeError,'保護狀態'):self.robot._write('wrist_roll',42,1800)
        self.assertEqual(self.port.status,32)
        self.assertTrue(all(p[4]==2 for p in self.port.packets))
        self.assertEqual(self.robot.goals,{})

    def test_exception_leaves_contact_journal_pending_and_stops(self):
        self.robot.gripper_power.state='holding'
        record={'phase':'holding','session_id':'fake-hold','device_key':'fake://no-hardware:6',
                'contact_possible':True}
        self.robot.gripper_power.journal.put(record)
        self.port.fail_stage='read'
        with self.assertRaises(OSError):self.robot.gripper_telemetry()
        self.assertEqual(self.robot.gripper_power.state,'blocked')
        self.assertEqual(self.robot.gripper_power.journal.get(),record)
        self.assertTrue(self.robot.stop.is_set());self.assertFalse(self.port.is_using)
        self.assertTrue(all(p[4]==2 for p in self.port.packets))


if __name__=='__main__':unittest.main()
