import json, tempfile, threading, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from server.robot import Robot, JOINTS
from server.brain import Brain, Scene
ROOT=Path(__file__).resolve().parents[1]
class SafetyTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.camera=Mock();self.camera.fresh.return_value=True
  self.r=Robot(json.loads((ROOT/'config/robot.json').read_text()),json.loads((ROOT/'config/follower-calibration.json').read_text()),self.temp.name,self.camera)
 def tearDown(self):self.temp.cleanup()
 def test_limits_nan_unknown(self):
  for target in [{'shoulder_pan':float('nan')},{'elbow_flex':10000},{'wrong':4},{'gripper':True}]:
   with self.assertRaises(ValueError):self.r.validate(target)
 def test_speed_must_respect_configured_motion_limit(self):
  self.r.begin=Mock()
  for speed in [0,-1,True,float('nan'),float('inf'),self.r.config['max_ticks_per_second']+1]:
   with self.assertRaises(ValueError):self.r.glide({'gripper':1600},speed=speed)
  self.r.begin.assert_not_called()
 def test_local_guard_rejects_drift_before_next_goal_write(self):
  self.r.begin=Mock();self.r.hold=Mock();self.r._write=Mock();self.r.goals={n:2000 for n in JOINTS}
  self.r.read=Mock(side_effect=[{n:2000 for n in JOINTS},{n:2005 for n in JOINTS}])
  def guard(pose):
   if pose['elbow_flex']!=2000:raise RuntimeError('unscreened drift')
  with self.assertRaisesRegex(RuntimeError,'unscreened drift'):
   self.r.glide({'wrist_flex':2020},guard=guard,allow_rearm=False)
  self.r.begin.assert_called_once_with(allow_rearm=False);self.r._write.assert_not_called()
 def test_init_failure_latches_stop_and_holds(self):
  self.r.begin=Mock(side_effect=RuntimeError('servo missing'));self.r.hold=Mock()
  with self.assertRaises(RuntimeError):self.r.glide({'shoulder_pan':2100})
  self.assertTrue(self.r.stop.is_set());self.assertFalse(self.r.moving);self.r.hold.assert_called_once()
 def test_stop_prevents_enable(self):
  self.r.connect=Mock();self.r.stop.set();self.r._write=Mock()
  with self.assertRaises(RuntimeError):self.r.begin()
  self.r._write.assert_not_called()
 def test_partial_hold_still_holds_healthy_joints(self):
  self.r.port=SimpleNamespace(is_using=False);self.r._read=Mock(side_effect=lambda n,*_: (_ for _ in ()).throw(RuntimeError('lost')) if n=='gripper' else 2000);self.r._write=Mock()
  result=self.r.hold();self.assertEqual(self.r._write.call_count,5);self.assertFalse(result['held']);self.assertIn('gripper',result['failures'])
 def test_stale_camera_blocks_motion(self):
  self.r.connect=Mock();self.camera.fresh.return_value=False;self.r._write=Mock()
  with self.assertRaises(RuntimeError):self.r.begin()
  self.r._write.assert_not_called()
 def test_held_goals_survive_partial_segments(self):
  actual={n:2000 for n in JOINTS};physical=dict(actual)
  def read_register(n,address,*_):
   if address==40:return 1
   if address in (65,66):return 0
   if address==42:return physical[n]
   raise AssertionError(f'Unexpected register {address}')
  self.r.connect=Mock();self.r.read=Mock(return_value=actual);self.r._read=Mock(side_effect=read_register);self.r._write=Mock()
  self.r.begin()
  self.r._write.assert_not_called()
  self.assertEqual(self.r.goals,physical)
  self.r.goals['elbow_flex']=physical['elbow_flex']=1900
  self.r._write.reset_mock()
  self.r.read.return_value={n:1937 if n=='elbow_flex' else 2000 for n in JOINTS}
  self.r.begin()
  self.r._write.assert_not_called()
  self.assertEqual(self.r.goals['elbow_flex'],1900)
 def test_scan_snapshot_failure_holds_and_latches(self):
  self.r._scan=Mock(side_effect=RuntimeError('capture failed'));self.r.hold=Mock()
  with self.assertRaises(RuntimeError):self.r.scan(Mock(),Mock())
  self.assertTrue(self.r.stop.is_set());self.r.hold.assert_called_once()
 def test_observer_rejected_before_subprocess(self):
  b=Brain(json.loads((ROOT/'config/robot.json').read_text()),self.temp.name)
  with patch('server.brain.subprocess.Popen') as popen:
   with self.assertRaises(ValueError):b.perceive('object',[{'source':'observer','path':'any'}])
   popen.assert_not_called()
 def test_scene_schema_invalid_confidence(self):
  from pydantic import ValidationError
  with self.assertRaises(ValidationError):Scene.model_validate({'query':'','summary':'','objects':[],'empty':True,'selected_index':-1})
if __name__=='__main__':unittest.main()

class StopRaceTests(unittest.TestCase):
 def test_hold_reasserts_stop_after_waiting_for_io(self):
  with tempfile.TemporaryDirectory() as directory:
   r=Robot(json.loads((ROOT/'config/robot.json').read_text()),json.loads((ROOT/'config/follower-calibration.json').read_text()),directory,Mock())
   with r.io:
    worker=threading.Thread(target=r.hold);worker.start()
    self.assertTrue(r.stop.wait(.5))
    r.resume()
    self.assertFalse(r.stop.is_set())
   worker.join(timeout=1)
   self.assertFalse(worker.is_alive());self.assertTrue(r.stop.is_set())
 def test_corrupt_power_journal_blocks_motion_before_connect(self):
  with tempfile.TemporaryDirectory() as directory:
   Path(directory,'gripper-pwm-session.json').write_text('{')
   r=Robot(json.loads((ROOT/'config/robot.json').read_text()),json.loads((ROOT/'config/follower-calibration.json').read_text()),directory,Mock())
   r.connect=Mock()
   self.assertTrue(r.stop.is_set())
   with self.assertRaises(RuntimeError):r.begin()
   with self.assertRaises(RuntimeError):r.resume()
   r.connect.assert_not_called()

 def test_schema_corruption_latches_stop_without_hardware_io(self):
  for change in [{'phase':[]},{'phase':'completed','neutral_samples':'3'}]:
   with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
    config=json.loads((ROOT/'config/robot.json').read_text())
    record={'version':1,'session_id':'a'*32,'phase':'lowered','servo_id':6,'model':777,
            'device_key':config['follower_port']+':6','position_bounds':[1456,2983],
            'original_cap':500,'low_cap':100,'restored_cap':500,'neutral_goal':1520,
            'neutral_position':1520,'neutral_samples':3,**change}
    Path(directory,'gripper-pwm-session.json').write_text(json.dumps(record))
    r=Robot(config,json.loads((ROOT/'config/follower-calibration.json').read_text()),directory,Mock())
    r.connect=Mock();r._write=Mock()
    self.assertTrue(r.stop.is_set())
    with self.assertRaises(RuntimeError):r.begin()
    with self.assertRaises(RuntimeError):r.resume()
    r.connect.assert_not_called();r._write.assert_not_called()
