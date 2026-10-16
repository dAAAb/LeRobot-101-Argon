import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from server.robot import Robot,JOINTS
ROOT=Path(__file__).resolve().parents[1]
class ServoFaultTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.r=Robot(json.loads((ROOT/'config/robot.json').read_text()),json.loads((ROOT/'config/follower-calibration.json').read_text()),self.temp.name,Mock())
  self.r.port=SimpleNamespace(is_using=False);self.r.ph=Mock();self.r.ph.write2ByteTxRx.return_value=(0,0);self.r.ph.write1ByteTxRx.return_value=(0,0)
  self.r.ph.read1ByteTxRx.return_value=(0,0,0);self.r.ph.read2ByteTxRx.return_value=(2000,0,0)
 def tearDown(self):self.temp.cleanup()
 def test_every_fault_bit_blocks_goal_and_torque_enable(self):
  for fault in [1,2,4,8,16,32,64,128]:
   self.r.ph.read1ByteTxRx.return_value=(fault,0,0)
   for address in [42,40]:
    with self.assertRaises(RuntimeError):self.r._write('gripper',address,1,1 if address==40 else 2)
  self.r.ph.write2ByteTxRx.assert_not_called();self.r.ph.write1ByteTxRx.assert_not_called()
 def test_hold_preserves_faulted_servo_protection_and_holds_others(self):
  self.r.ph.read1ByteTxRx.side_effect=lambda port,motor,address:(32 if motor==6 else 0,0,0)
  result=self.r.hold();self.assertFalse(result['held']);self.assertIn('gripper',result['failures'])
  self.assertEqual([call.args[1] for call in self.r.ph.write2ByteTxRx.call_args_list],[1,2,3,4,5])
 def test_communication_error_cannot_become_healthy_status(self):
  self.r.ph.read1ByteTxRx.return_value=(0,-1,0)
  with self.assertRaises(RuntimeError):self.r._write('gripper',42,1600)
  self.r.ph.write2ByteTxRx.assert_not_called()
 def test_moving_register_does_not_become_fault_flag(self):
  self.r.ph.read1ByteTxRx.side_effect=lambda port,motor,address:(1 if address==66 else 0,0,0)
  self.r._write('gripper',42,1600)
  self.r.ph.write2ByteTxRx.assert_called_once()
 def test_resume_cannot_clear_a_servo_fault(self):
  self.r.connected=True;self.r.stop.set();self.r.ph.read1ByteTxRx.return_value=(8,0,0)
  with self.assertRaises(RuntimeError):self.r.resume()
  self.assertTrue(self.r.stop.is_set());self.r.ph.write2ByteTxRx.assert_not_called()
