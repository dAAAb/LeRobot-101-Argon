"""Fake SDK + real Robot/active_touch regression; never opens hardware."""
import json,tempfile,threading,types,unittest
from pathlib import Path
from server.robot import Robot,JOINTS
from server.active_touch import collect_empty_jaw
from server.gripper_power import SessionFailure,RecoveryRequired
from scservo_sdk import COMM_SUCCESS

class InstantStop(threading.Event):
 def wait(self,timeout=None):return self.is_set()
class FakeCamera:
 def fresh(self):return True
class FakeProtocol:
 def __init__(self,cal):
  self.reg={};self.events=[];self.fail_restore_ack=True;self.now=0
  for name,c in cal.items():
   pos=1520 if name=='gripper' else 2000
   self.reg[c['id']]={3:777,33:0,40:1,42:pos,48:500,56:pos,60:0,62:120,63:30,65:0,66:0,69:0}
 def read(self,port,servo,address):
  self.events.append(('read',servo,address,self.reg[servo][address]));return self.reg[servo][address],COMM_SUCCESS,0
 read2ByteTxRx=read;read1ByteTxRx=read
 def write(self,port,servo,address,value):
  self.events.append(('write',servo,address,value));self.reg[servo][address]=value
  if servo==6 and address==48 and value==500 and self.fail_restore_ack:
   self.fail_restore_ack=False;self.events.append(('restore_ack_lost',servo,address,value));return -3001,0
  return COMM_SUCCESS,0
 write2ByteTxRx=write;write1ByteTxRx=write
 def writes(self,start=0,servo=6):return [(e[2],e[3]) for e in self.events[start:] if e[0]=='write' and e[1]==servo]

class Integration(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='lerobot-pwm-integration-');self.addCleanup(self.tmp.cleanup)
  cal={name:{'id':i+1,'range_min':1456 if name=='gripper' else 0,'range_max':2983 if name=='gripper' else 4095} for i,name in enumerate(JOINTS)}
  self.robot=Robot({'follower_port':'fake://no-hardware','max_ticks_per_second':90,'max_tracking_error_ticks':90},cal,self.tmp.name,FakeCamera())
  self.bus=FakeProtocol(cal);r=self.robot;r.connected=True;r.port=types.SimpleNamespace(is_using=False);r.ph=self.bus;r.models={n:777 for n in JOINTS};r.armed=True;r.position={n:self.bus.reg[c['id']][56] for n,c in cal.items()};r.goals=dict(r.position);r.stop=InstantStop();r.gripper_power.stop_latch=r.stop.set;r.gripper_power.clock=lambda:self.bus.now;r.gripper_power.sleep=lambda dt:setattr(self.bus,'now',self.bus.now+dt)
 def lose_restore_ack(self):
  r=self.robot;r.pwm_owner=r.gripper_power.enter()
  with self.assertRaises(SessionFailure):r.gripper_power.finish()
  self.assertEqual(self.bus.reg[6][48],500);self.assertEqual(r.gripper_power.state,'blocked')
  return len(self.bus.events)
 def test_adopt_existing_stationary_hold_never_writes_or_reseeds(self):
  from unittest.mock import patch
  self.robot.armed=False
  def wait(timeout=None):
   self.bus.now+=timeout or 0;return self.robot.stop.is_set()
  # This fixture must model elapsed waits now that adoption requires a real
  # six-sample >=0.5s window, rather than an InstantStop that elides all time.
  with patch.object(self.robot.stop,'wait',side_effect=wait),patch('server.robot.time.monotonic',lambda:self.bus.now):
   anchor=self.robot.adopt_stationary_hold()
  self.assertEqual(len(anchor['stationary_samples']),6)
  self.assertGreaterEqual(anchor['stationary_duration_s'],.5)
  self.assertTrue(self.robot.armed)
  self.assertFalse(any(e[0]=='write' for e in self.bus.events))
  self.robot.begin(allow_rearm=False)
  self.assertFalse(any(e[0]=='write' for e in self.bus.events))
 def test_local_begin_cannot_reenable_a_disabled_motor(self):
  self.bus.reg[2][40]=0
  with self.assertRaisesRegex(RuntimeError,'重新啟用'):
   self.robot.begin(allow_rearm=False)
  self.assertFalse(any(e[0]=='write' for e in self.bus.events))
 def check_recovery(self,start):
  r=self.robot
  self.assertEqual(self.bus.writes(start),[], 'Pending/blocked session emitted a new gripper write during ordinary cleanup')
  before=len(self.bus.events);receipt=r.recover_gripper_power()
  self.assertTrue(receipt['completed']);self.assertEqual(self.bus.writes(before),[(48,100),(42,1520),(48,500)])
  self.assertTrue(r.stop.is_set(),'Recovery must not automatically resume')
 def test_restore_ack_loss_then_real_glide_cleanup_and_repeated_stop(self):
  start=self.lose_restore_ack()
  with self.assertRaises(RecoveryRequired):self.robot.glide({'gripper':1600},speed=50)
  a=self.robot.hold();b=self.robot.hold()
  self.assertIn('gripper',a['failures']);self.assertIn('gripper',b['failures'])
  self.check_recovery(start)
 def test_active_touch_cleanup_after_restore_ack_loss_does_not_write_goal(self):
  r=self.robot
  # Isolate acquisition duration while preserving the real power session,
  # telemetry, active_touch finally/exception handling, and Robot.hold.
  def quick_glide(this,targets,**kw):
   for n,p in targets.items():
    this.goals[n]=p;self.bus.reg[this.calibration[n]['id']][42]=p;self.bus.reg[this.calibration[n]['id']][56]=p
  r.glide=types.MethodType(quick_glide,r)
  with self.assertRaises(SessionFailure):collect_empty_jaw(r,lambda message:None,threading.Event(),{'source':'wrist','method':'fake-test'})
  loss=next(i for i,e in enumerate(self.bus.events) if e[0]=='restore_ack_lost')+1
  self.assertEqual(r.gripper_power.state,'blocked');self.assertIsNone(r.pwm_owner)
  r.hold();self.check_recovery(loss)
  end=json.loads((Path(self.tmp.name)/'gripper-empty-latest.jsonl').read_text().splitlines()[-1]);self.assertFalse(end['complete']);self.assertFalse(end['restore_verified'])
 def test_restart_pending_then_stop_skips_gripper_without_recovery(self):
  r=self.robot;r.pwm_owner=r.gripper_power.enter()
  # Simulate a fresh runtime object sharing only the durable journal and device.
  replacement=Robot(r.config,r.calibration,self.tmp.name,FakeCamera());replacement.connected=True;replacement.port=types.SimpleNamespace(is_using=False);replacement.ph=self.bus
  before=len(self.bus.events);out=replacement.hold();self.assertIn('gripper',out['failures']);self.assertEqual(self.bus.writes(before),[])

if __name__=='__main__':unittest.main(verbosity=2)
