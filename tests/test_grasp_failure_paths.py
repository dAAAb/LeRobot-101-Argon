"""Real WristGrasp + real power session; fake registers, camera, model and clock.

Never imports Robot/app or opens hardware/network. These tests exercise advice
dispatch and closing/cancellation/failure lifecycle, not image truth or collision planning.
"""
from pathlib import Path
import sys, tempfile, threading, types, unittest
from unittest.mock import Mock, patch

from server import grasp as gm
from server.gripper_power import GripperPWMSession, AtomicJournal, SessionCancelled, RecoveryRequired


class Clock:
    def __init__(self): self.now = 10.
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class Event:
    def __init__(self, clock, stop_after_waits=None):
        self.clock=clock; self.flag=False; self.waits=0; self.stop_after_waits=stop_after_waits
    def set(self): self.flag=True
    def is_set(self): return self.flag
    def wait(self, seconds):
        self.waits += 1; self.clock.sleep(seconds)
        if self.stop_after_waits is not None and self.waits >= self.stop_after_waits: self.flag=True
        return self.flag


class FakeRobot:
    def __init__(self, root, clock):
        self.clock=clock; self.io=threading.RLock(); self.stop=Event(clock)
        self.camera=types.SimpleNamespace(fresh=lambda: True, generation='fake-wrist-generation')
        self.registers={3:777, 33:0, 65:0, 48:500, 42:1800, 56:1800, 66:1}
        self.events=[]; self.telemetry_count=0; self.fail_telemetry_at=None; self.fail_closing_ack=False
        self.pwm_owner=None; self.moving=False; self.goals={'gripper':1800}; self.holds=0
        self.journal=AtomicJournal(Path(root)/'fake-journal.json')
        self.gripper_power=GripperPWMSession(read=self.read_register, write=self.write_register,
            lock=self.io, stop_latch=self.stop.set, journal=self.journal, device_key='fake-gripper',
            position_bounds=(1456,2983), verified_open_position=1860, opening_direction=1,
            position_tolerance=2, clock=clock.monotonic, sleep=clock.sleep)

    def read_register(self, servo, address, width):
        assert servo == 6; self.events.append(('read',address,self.registers[address]))
        return self.registers[address]

    def write_register(self, servo, address, value, width):
        assert servo == 6 and address in (42,48)
        record=self.journal.get(); self.events.append(('write',address,value,record))
        assert record and record['phase'] != 'completed'
        if address == 42:
            assert self.registers[48] == 100 and self.registers[65] == 0
            if value < self.registers[56]:
                assert record['contact_possible'] is True, 'closing before durable contact barrier'
                assert self.registers[56]-value <= 12, 'excessive outstanding closing lead'
        if address == 48 and value == 500:
            assert not record.get('contact_possible'), 'restored cap with possible object'
        self.registers[address]=value
        if address == 42 and value < self.registers[56] and self.fail_closing_ack:
            raise IOError('closing goal applied but ACK lost')

    def _read(self, name, address, width=2):
        assert name == 'gripper'; return self.read_register(6,address,width)
    def _write(self, name, address, value, width=2):
        assert name == 'gripper'; return self.write_register(6,address,value,width)
    def begin(self, **kwargs):
        self.gripper_power.assert_motion_allowed(self.pwm_owner)
        if self.stop.is_set(): raise SessionCancelled('fake stop')
    def hold(self, **kwargs):
        # Fake body hold has no gripper writes, matching the required fault path.
        self.stop.set(); self.holds += 1
        return {'held':False,'failures':{'gripper':'pending contact'}}
    def gripper_telemetry(self):
        self.telemetry_count += 1
        if self.telemetry_count == self.fail_telemetry_at: raise IOError('lost closing telemetry')
        if self.registers[65]: raise RuntimeError('servo fault')
        self.clock.sleep(.01)
        return {'capture_monotonic':self.clock.monotonic(), 'position':self.registers[56],
                'goal':self.registers[42], 'pwm_limit':self.registers[48], 'temperature_c':25,
                'status':0, 'load_signed_pwm':25, 'body_position':{n:2000 for n in gm.NAMES}}
    def writes(self): return [(e[1],e[2]) for e in self.events if e[0]=='write']


class GraspFailurePaths(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.clock=Clock(); self.robot=FakeRobot(self.tmp.name,self.clock)
        self.cancel=Event(self.clock)
        self.brain=types.SimpleNamespace(verify_grasp=lambda *a,**kw: self.verification())
        self.grasp=gm.WristGrasp(self.robot,self.brain,self.frame,self.tmp.name)
        self.grasp.capture=self.frame; self.grasp.record=lambda record: None
        self.patch_time=patch.object(gm,'time',self.clock); self.patch_time.start(); self.addCleanup(self.patch_time.stop)
        self.patch_vision=patch.object(gm,'unchanged_wrist',return_value={})
        self.patch_vision.start(); self.addCleanup(self.patch_vision.stop)

    def frame(self):
        self.clock.sleep(.01)
        return {'source':'wrist','camera_generation':'fake-wrist-generation',
                'capture_monotonic':self.clock.monotonic(), 'path':'unused-mocked-image',
                'joints':{**{n:2000 for n in gm.NAMES},'gripper':self.robot.registers[56]}}

    def verification(self):
        return {'confidence':.4,'same_object':False,'between_fingers':True,
                'clear_of_table':False,'follows_gripper':False,'gripper_empty':False}

    def run_setup(self):
        self.grasp.advice=lambda *a,**kw: ({'step':'close','target_visible':True,'clear_path':True,
            'confidence':.99,'target_between_fingers':True}, self.frame())
        self.grasp.move=lambda direction,distance,before,report: {
            'before':before,'after':self.frame(),'actual_delta_model_mm':[0,0,3]}

    def assert_low_contact(self):
        self.assertEqual(self.robot.registers[48],100)
        self.assertIs(self.robot.journal.get()['contact_possible'],True)
        self.assertNotIn((48,500),self.robot.writes())

    def advice_dispatch_setup(self, *, step='move', confidence=.84, **changes):
        advice={'step':step,'direction':'left','distance_mm':3,
                'target_visible':True,'clear_path':True,'confidence':confidence,
                'target_between_fingers':True,'gripper_empty':True,
                'warning':'test advice rejected','reason':'synthetic advice',**changes}
        frame=self.frame()
        self.grasp.advice=Mock(return_value=(advice,frame))
        self.grasp.move=Mock(return_value={'direction':'left','actual_delta_model_mm':[0.,3.,0.]})
        self.grasp.open_empty=Mock()
        self.grasp.close_candidate=Mock()
        return frame

    def test_clear_path_point84_move_dispatches_one_bounded_step(self):
        frame=self.advice_dispatch_setup()
        report=lambda message:None
        # The explicit one-step budget ends the run after the permitted move.
        with self.assertRaisesRegex(RuntimeError,'小步觀察上限'):
            self.grasp.run('synthetic object',report,self.cancel,max_steps=1)
        self.grasp.move.assert_called_once_with('left',3,frame,report,cancel=self.cancel,empty_hand_progress=True)
        self.grasp.close_candidate.assert_not_called()
        self.grasp.open_empty.assert_not_called()
        self.assertEqual(self.robot.writes(),[])

    def test_new_attempt_does_not_reuse_previous_attempt_movement_evidence(self):
        def capture():
            frame=self.frame()
            frame['id']=f"capture_{frame['capture_monotonic']}"
            return frame
        self.grasp.capture=capture
        self.brain.grasp_advice=Mock(side_effect=[
            {'step':'move','direction':'left','distance_mm':3,'target_visible':True,
             'clear_path':True,'confidence':.9,'gripper_empty':True,'warning':'','reason':'synthetic move'},
            *[{'step':'abort','target_visible':True,'clear_path':True,'confidence':.99,
               'warning':'synthetic stop','reason':'synthetic stop'} for _ in range(2)]])

        def move(direction,distance,before,report,**kwargs):
            receipt={'direction':direction,'before':before,'after':capture(),
                     'actual_delta_model_mm':[0.,3.,0.]}
            self.grasp.latest=receipt
            return receipt
        self.grasp.move=Mock(side_effect=move)
        for query in ('first object','second object'):
            with self.assertRaisesRegex(RuntimeError,'synthetic stop'):
                self.grasp.run(query,lambda message:None,self.cancel,max_steps=2)

        calls=self.brain.grasp_advice.call_args_list
        self.assertEqual(len(calls),3)
        # History is useful within its own attempt, including a measured move.
        first_continuation=calls[1].args[2]
        self.assertIn('latest_before_move',first_continuation['frame_roles'])
        self.assertEqual(len(first_continuation['completed_steps']),1)
        self.assertIsNotNone(first_continuation['last_measured_step'])
        # The next attempt uses the same camera generation, but no old evidence.
        new_views,new_context=calls[2].args[1:3]
        self.assertEqual(len(new_views),1)
        self.assertEqual(new_context['frame_roles'],['current'])
        self.assertEqual(new_context['completed_steps'],[])
        self.assertIsNone(new_context['last_measured_step'])
        self.grasp.move.assert_called_once()
        self.assertEqual(self.robot.writes(),[])

    def test_same_point84_close_does_not_dispatch_even_when_between_fingers(self):
        self.advice_dispatch_setup(step='close')
        with self.assertRaisesRegex(RuntimeError,'test advice rejected'):
            self.grasp.run('synthetic object',lambda message:None,self.cancel,max_steps=1)
        self.grasp.move.assert_not_called()
        self.grasp.close_candidate.assert_not_called()
        self.grasp.open_empty.assert_not_called()
        self.assertEqual(self.robot.writes(),[])

    def test_no_visible_target_or_no_clear_path_never_dispatches_move(self):
        for missing in ('target_visible','clear_path'):
            with self.subTest(missing=missing):
                self.advice_dispatch_setup(confidence=.99,**{missing:False})
                with self.assertRaisesRegex(RuntimeError,'test advice rejected'):
                    self.grasp.run('synthetic object',lambda message:None,self.cancel,max_steps=1)
                self.grasp.move.assert_not_called()
                self.grasp.close_candidate.assert_not_called()
                self.grasp.open_empty.assert_not_called()
                self.assertEqual(self.robot.writes(),[])

    def test_exploratory_move_point70_is_allowed_but_below_it_is_not(self):
        for confidence,allowed in ((.70,True),(.699,False)):
            with self.subTest(confidence=confidence):
                self.advice_dispatch_setup(confidence=confidence)
                message='小步觀察上限' if allowed else 'test advice rejected'
                with self.assertRaisesRegex(RuntimeError,message):
                    self.grasp.run('synthetic object',lambda message:None,self.cancel,max_steps=1)
                self.assertEqual(self.grasp.move.call_count,int(allowed))
                self.grasp.close_candidate.assert_not_called()

    def test_close_point90_is_required_before_contact_controller_dispatch(self):
        class CloseDispatched(Exception):pass
        for confidence,allowed in ((.90,True),(.899,False)):
            with self.subTest(confidence=confidence):
                self.advice_dispatch_setup(step='close',confidence=confidence)
                self.grasp.close_candidate.side_effect=CloseDispatched('contact controller reached')
                with self.assertRaises(CloseDispatched if allowed else RuntimeError):
                    self.grasp.run('synthetic object',lambda message:None,self.cancel,max_steps=1)
                self.assertEqual(self.grasp.close_candidate.call_count,int(allowed))
                self.grasp.move.assert_not_called()
                self.assertEqual(self.robot.writes(),[])

    def test_other_action_keeps_point85_threshold(self):
        for confidence,allowed in ((.85,True),(.84,False)):
            with self.subTest(confidence=confidence):
                self.advice_dispatch_setup(step='open',confidence=confidence)
                message='小步觀察上限' if allowed else 'test advice rejected'
                with self.assertRaisesRegex(RuntimeError,message):
                    self.grasp.run('synthetic object',lambda message:None,self.cancel,max_steps=1)
                self.assertEqual(self.grasp.open_empty.call_count,int(allowed))
                self.grasp.move.assert_not_called()
                self.grasp.close_candidate.assert_not_called()

    def test_normal_obstruction_keeps_short_goal_and_never_asserts_true_grasp(self):
        result=self.grasp.close_candidate(self.frame(),lambda message:None,self.cancel)
        self.assertTrue(result['obstruction_candidate']); self.assertTrue(result['not_force_measurement'])
        self.assertNotIn('grasp_confirmed',result); self.assert_low_contact()
        closing=[v for a,v in self.robot.writes() if a==42]
        self.assertEqual(closing,[1795,1790,1788]); self.assertEqual(self.robot.registers[56],1800)
        self.assertFalse(self.robot.moving)

    def test_cancel_after_first_close_removes_closing_demand_once_under_100(self):
        self.cancel.stop_after_waits=1
        with self.assertRaises(SessionCancelled):
            self.grasp.close_candidate(self.frame(),lambda message:None,self.cancel)
        self.assert_low_contact(); self.assertEqual(self.robot.writes(),[(48,100),(42,1795),(42,1800)])
        self.assertTrue(self.robot.stop.is_set()); self.assertEqual(self.robot.gripper_power.state,'blocked')

    def test_lost_closing_ack_never_attempts_cleanup_goal_or_restore(self):
        self.robot.fail_closing_ack=True
        with self.assertRaises(IOError):
            self.grasp.close_candidate(self.frame(),lambda message:None,self.cancel)
        self.assert_low_contact(); self.assertEqual(self.robot.writes(),[(48,100),(42,1795)])
        self.assertEqual(self.robot.gripper_power.state,'blocked'); self.assertTrue(self.robot.stop.is_set())

    def test_telemetry_failure_preserves_contact_and_does_not_retry_jaw(self):
        self.robot.fail_telemetry_at=2
        with self.assertRaises(IOError):
            self.grasp.close_candidate(self.frame(),lambda message:None,self.cancel)
        self.assert_low_contact(); self.assertEqual(self.robot.writes(),[(48,100),(42,1795)])
        before=list(self.robot.events)
        with self.assertRaises(RecoveryRequired): self.robot.gripper_power.recover()
        self.assertEqual(self.robot.events,before)

    def test_post_contact_brain_failure_blocks_without_extra_jaw_write(self):
        self.run_setup()
        def failure(*a,**kw): raise IOError('verification unavailable')
        self.brain.verify_grasp=failure
        with self.assertRaises(IOError): self.grasp.run('object',lambda message:None,self.cancel)
        self.assert_low_contact(); self.assertEqual(self.robot.registers[42],1788)
        self.assertEqual(self.robot.gripper_power.state,'blocked'); self.assertTrue(self.robot.stop.is_set())
        self.assertNotIn(self.grasp.state,('idle','approaching','holding'))

    def test_post_contact_explicit_cancel_removes_closing_demand_at_low_cap(self):
        self.run_setup()
        def cancelled(*a,**kw): raise SessionCancelled('verification cancelled')
        self.brain.verify_grasp=cancelled
        with self.assertRaises(SessionCancelled): self.grasp.run('object',lambda message:None,self.cancel)
        self.assert_low_contact(); self.assertEqual(self.robot.registers[42],1800)
        self.assertEqual(self.robot.gripper_power.state,'blocked'); self.assertTrue(self.robot.stop.is_set())

    def test_negative_visual_verification_keeps_unverified_hold_at_100(self):
        self.run_setup(); result=self.grasp.run('object',lambda message:None,self.cancel)
        self.assertFalse(result['grasp_confirmed']); self.assertEqual(result['state'],'holding_unverified')
        self.assert_low_contact(); self.assertEqual(self.robot.gripper_power.state,'holding')
        self.assertIsNotNone(self.robot.pwm_owner)
        self.assertEqual(self.robot.registers[42],1788)


if __name__=='__main__': unittest.main(verbosity=2)
