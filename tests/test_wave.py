"""Real wrist wave dispatch/CAD with an in-memory plant; no motor or camera I/O.

Synthetic PNGs exercise the real unchanged-image gate. This verifies software
bounds and receipts, not the truth of visual clearance or physical collision.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile, threading, unittest
from unittest.mock import Mock

import cv2
import numpy as np

from server.grasp import WristGrasp
from server.gripper_power import SessionCancelled
from server.local_cartesian import ALL, NAMES, default_model
from server.local_feedback import check_bounds


class Cancel:
    def __init__(self):
        self.flag=False;self.waits=0;self.cancel_after_wait=None
    def set(self):self.flag=True
    def is_set(self):return self.flag
    def wait(self,seconds):
        self.waits+=1
        if self.waits==self.cancel_after_wait:self.set()
        return self.flag


class FakeRobot:
    def __init__(self):
        self.actual=dict(zip(ALL,[2216,1163,2952,2300,987,1857]))
        self.goals={**self.actual,'shoulder_pan':2221,'shoulder_lift':1156,
                    'elbow_flex':2946,'wrist_flex':2302,'wrist_roll':994}
        self.bias={n:self.goals[n]-self.actual[n] for n in ALL}
        self.camera=SimpleNamespace(generation='synthetic-wrist',fresh=lambda:True)
        self.pwm_owner=None;self.stop=Cancel();self.io=threading.RLock()
        self.calls=[];self.writes=[];self.before_glide=None;self.begin_calls=[]
    def begin(self,**kwargs):
        self.begin_calls.append(kwargs)
        if self.stop.is_set():raise SessionCancelled('fake stop')
    def read(self):return dict(self.actual)
    def glide(self,targets,**kwargs):
        # This plant preserves physical-minus-measured offsets. It also checks
        # both boxes and the same before-write guard expected of Robot.glide.
        before=self.read();goals_before=dict(self.goals)
        if self.before_glide:self.before_glide()
        if self.stop.is_set():raise SessionCancelled('fake stop')
        physical={n:v+self.bias[n] for n,v in targets.items()}
        check_bounds(before,kwargs['raw_bounds'])
        check_bounds(before,kwargs['measured_bounds'])
        check_bounds(goals_before,kwargs['raw_bounds'],targets)
        check_bounds({**before,**physical},kwargs['raw_bounds'])
        kwargs['guard'](before);kwargs['guard']({**before,**physical})
        self.calls.append({'before':before,'goals_before':goals_before,
                           'targets':dict(targets),'physical':physical,'kwargs':kwargs})
        self.writes.extend((n,v) for n,v in physical.items())
        self.goals.update(physical);self.actual.update(targets)
        check_bounds(self.actual,kwargs['measured_bounds']);kwargs['guard'](self.read())


class WaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=default_model()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        base=np.full((540,960,3),80,np.uint8)
        self.base=root/'stationary.png';self.changed=root/'changed.png'
        self.assertTrue(cv2.imwrite(str(self.base),base))
        base[100:300,200:600]=240
        self.assertTrue(cv2.imwrite(str(self.changed),base))
        self.robot=FakeRobot();self.cancel=Cancel();self.frames=0;self.capture_hook=None
        self.clearance={'confidence':.95,'gripper_empty':True,'clear_sweep':True,
                        'human_clear':True,'reason':'synthetic clear sweep'}
        self.brain=SimpleNamespace(wave_clearance=Mock(side_effect=lambda *a,**kw:self.clearance))
        self.grasp=WristGrasp(self.robot,self.brain,self.capture,self.tmp.name)
        self.grasp._model=self.model;self.grasp.capture=self.capture
        def options(frames,**kwargs):
            return {'wave':self.brain.wave_clearance(frames,kwargs.get('cancel'),controller_geometry=kwargs['roll_geometry']),
                    'lift':{'gripper_empty':True,'human_clear':True,'clear_path':False,'confidence':.8,'reason':'synthetic blocked lift'}}
        self.grasp._wave_options=Mock(side_effect=options)
        self.events=[];self.grasp.record=self.events.append;self.reports=[]
    def capture(self):
        self.frames+=1
        frame={'source':'wrist','path':str(self.base),'camera_generation':self.robot.camera.generation,
               'capture_monotonic':float(self.frames),'joints':self.robot.read()}
        if self.capture_hook:self.capture_hook(frame,self.frames)
        return frame
    def wave(self):return self.grasp.wave(self.reports.append,self.cancel)
    def assert_no_motion(self):
        self.assertEqual(self.robot.writes,[])
        self.assertFalse(any(e.get('type')=='wave_result' for e in self.events))

    def test_holding_or_owned_jaw_rejects_before_capture_or_motion(self):
        for state,owner in [('holding',None),('holding_unverified',None),('idle','owner')]:
            with self.subTest(state=state,owner=owner):
                self.grasp.state=state;self.robot.pwm_owner=owner
                with self.assertRaises(RuntimeError):self.wave()
                self.assertEqual(self.frames,0);self.brain.wave_clearance.assert_not_called()
                self.assert_no_motion()

    def test_all_three_clearance_fields_and_confidence_are_required(self):
        valid=dict(self.clearance)
        for field in ('gripper_empty','clear_sweep','human_clear','confidence'):
            with self.subTest(field=field):
                self.clearance={**valid,field:.799 if field=='confidence' else False}
                with self.assertRaises(RuntimeError):self.wave()
                self.assert_no_motion()

    def test_inference_changed_image_rejects_before_any_roll(self):
        def changed(frame,index):
            if index>=2:frame['path']=str(self.changed)
        self.capture_hook=changed
        self.brain.wave_clearance.side_effect=[self.clearance,{**self.clearance,'clear_sweep':False}]
        with self.assertRaisesRegex(RuntimeError,'空間尚未確認'):self.wave()
        self.assertEqual(self.brain.wave_clearance.call_count,2)
        self.assertEqual(self.brain.wave_clearance.call_args.args[0][-1]['path'],str(self.changed))
        self.assert_no_motion()

    def test_inference_changed_generation_or_body_rejects_before_motion(self):
        for kind in ('generation','body'):
            with self.subTest(kind=kind):
                self.frames=0
                def changed(frame,index):
                    if index==2:
                        if kind=='generation':frame['camera_generation']='reconnected'
                        else:frame['joints']['elbow_flex']+=4
                self.capture_hook=changed
                with self.assertRaises(ValueError):self.wave()
                self.assert_no_motion()

    def test_cancellation_after_inference_prevents_every_motion(self):
        def infer(*args,**kwargs):self.cancel.set();return self.clearance
        self.brain.wave_clearance.side_effect=infer
        with self.assertRaises(SessionCancelled):self.wave()
        self.assert_no_motion()

    def test_cancel_immediately_before_segment_write_stops_without_return(self):
        self.robot.before_glide=self.cancel.set
        with self.assertRaises(SessionCancelled):self.wave()
        self.assert_no_motion()

    def test_camera_reconnect_before_segment_write_stops(self):
        def reconnect():self.robot.camera.generation='reconnected'
        self.robot.before_glide=reconnect
        with self.assertRaisesRegex(RuntimeError,'相機重新連接'):self.wave()
        self.assert_no_motion()

    def test_cancel_between_legs_does_not_issue_a_return_trajectory(self):
        self.cancel.cancel_after_wait=1
        with self.assertRaises(SessionCancelled):self.wave()
        self.assertEqual(len(self.robot.calls),4)
        measured_angle=(self.robot.actual['wrist_roll']-987)*360/4096
        self.assertGreaterEqual(measured_angle,9.6);self.assertLessEqual(measured_angle,10.)
        self.assertFalse(any(e.get('type')=='wave_result' for e in self.events))

    def test_two_measured_cycles_preserve_holds_and_never_command_jaw(self):
        initial=self.robot.read();holds=dict(self.robot.goals)
        result=self.wave()
        self.assertTrue(result['wave_completed']);self.assertEqual(result['cycles'],2)
        self.assertLessEqual(result['measured_roll_range_deg'][0],-9.6)
        self.assertGreaterEqual(result['measured_roll_range_deg'][1],9.6)
        self.assertGreaterEqual(result['measured_roll_range_deg'][0],-10.)
        self.assertLessEqual(result['measured_roll_range_deg'][1],10.)
        self.assertLessEqual(abs(result['return_error_deg']),3*360/4096)
        self.assertTrue(result['not_external_metrology'])
        self.assertEqual(self.robot.actual,initial);self.assertEqual(self.robot.goals,holds)
        self.assertEqual(len(self.robot.calls),32)
        self.assertEqual({n for n,v in self.robot.writes},{'wrist_roll'})
        for call in self.robot.calls:
            self.assertEqual(set(call['targets']),{'wrist_roll'})
            self.assertLessEqual(abs(call['targets']['wrist_roll']-call['before']['wrist_roll']),30)
            self.assertLessEqual(abs(call['physical']['wrist_roll']-call['goals_before']['wrist_roll']),30)
            self.assertEqual(call['physical']['wrist_roll']-call['targets']['wrist_roll'],7)
            self.assertFalse(call['kwargs']['allow_rearm'])
            for n in NAMES:
                if n!='wrist_roll':
                    self.assertEqual(call['kwargs']['measured_bounds'][n],[initial[n]-4,initial[n]+4])
        self.assertTrue(all(c['allow_rearm'] is False for c in self.robot.begin_calls))
        self.assertEqual(self.events[-1]['type'],'wave_result')
        self.assertEqual(result['requested_roll_range_deg'],[-113*360/4096,113*360/4096])
        self.brain.wave_clearance.assert_called_once()
        geometry=self.brain.wave_clearance.call_args.kwargs['controller_geometry']
        self.assertEqual(geometry['max_roll_degrees'],113*360/4096)
        self.assertEqual(geometry['tracking_margin_degrees'],12*360/4096)

    def test_receipt_rejects_insufficient_measured_extremes(self):
        def incomplete(goal,reference,cancel,**kwargs):
            self.robot.actual['wrist_roll']=987+max(-100,min(100,goal-987))
            return self.capture()
        self.grasp.roll_sweep=Mock(side_effect=incomplete)
        with self.assertRaisesRegex(RuntimeError,'幅度.*不足'):self.wave()
        self.assertFalse(any(e.get('type')=='wave_result' for e in self.events))

    def test_endpoint_camera_reconnect_cannot_become_next_legs_reference(self):
        def reconnect(frame,index):
            if index==3:
                self.robot.camera.generation='reconnected';frame['camera_generation']='reconnected'
        self.capture_hook=reconnect
        with self.assertRaisesRegex((ValueError,RuntimeError),'相機|generation'):self.wave()
        self.assertEqual(len(self.robot.calls),4)
        self.assertFalse(any(e.get('type')=='wave_result' for e in self.events))

    def test_jaw_opening_drift_before_first_roll_write_stops_without_motion(self):
        for drift in (-5,5):
            with self.subTest(drift=drift):
                self.frames=0;self.robot=FakeRobot();self.grasp.robot=self.robot
                def drift_after_start_frame(frame,index):
                    # The image/pose pair was valid when captured. The next
                    # fresh controller read must catch a later jaw change.
                    if index==2:self.robot.actual['gripper']+=drift
                self.capture_hook=drift_after_start_frame
                with self.assertRaisesRegex(RuntimeError,'夾爪開口改變'):self.wave()
                self.assert_no_motion()
                self.assertEqual(self.robot.goals['gripper'],1857)

    def test_jaw_opening_drift_at_leg_endpoint_prevents_next_leg_and_receipt(self):
        writes_at_endpoint=[]
        def drift_at_endpoint(frame,index):
            if index==3:
                writes_at_endpoint.extend(self.robot.writes)
                self.robot.actual['gripper']+=5
                frame['joints']['gripper']=self.robot.actual['gripper']
        self.capture_hook=drift_at_endpoint
        with self.assertRaisesRegex(RuntimeError,'夾爪開口改變'):self.wave()
        self.assertEqual(len(writes_at_endpoint),4)
        self.assertEqual(self.robot.writes,writes_at_endpoint)
        self.assertEqual({n for n,v in self.robot.writes},{'wrist_roll'})
        self.assertEqual(self.robot.actual['wrist_roll'],987+113)
        self.assertEqual(self.robot.goals['gripper'],1857)
        self.assertFalse(any(e.get('type')=='wave_result' for e in self.events))

    def test_each_cycle_needs_measured_positive_and_negative_extremes(self):
        legs=0
        def second_cycle_missing(goal,reference,cancel,**kwargs):
            nonlocal legs
            legs+=1
            self.robot.actual['wrist_roll']=goal if legs<=4 else 987
            return self.capture()
        self.grasp.roll_sweep=Mock(side_effect=second_cycle_missing)
        with self.assertRaisesRegex(RuntimeError,'幅度.*不足'):self.wave()
        self.assertFalse(any(e.get('type')=='wave_result' for e in self.events))

    def test_receipt_requires_measured_final_return(self):
        legs=0
        def failed_final_return(goal,reference,cancel,**kwargs):
            nonlocal legs
            legs+=1
            self.robot.actual['wrist_roll']=goal+12 if legs==8 else goal
            return self.capture()
        self.grasp.roll_sweep=Mock(side_effect=failed_final_return)
        with self.assertRaises(RuntimeError):self.wave()
        self.assertFalse(any(e.get('type')=='wave_result' for e in self.events))

    def test_body_corridor_does_not_reanchor_at_each_leg(self):
        original=self.robot.read()
        self.robot.actual['shoulder_pan']+=5
        # The latest reference by itself looks stationary. It must not replace
        # the body pose against which the whole wave was visually cleared.
        reference=self.capture()
        with self.assertRaises((ValueError,RuntimeError)):
            self.grasp.roll_sweep(original['wrist_roll']+30,reference,self.cancel,
                                  body_origin=original,camera_generation='synthetic-wrist')
        self.assert_no_motion()

    def test_model_table_veto_propagates_without_goal_write(self):
        self.grasp._model=SimpleNamespace(tip=self.model.tip,propose_rotation=Mock(side_effect=ValueError('table model veto')))
        with self.assertRaisesRegex(ValueError,'table model veto'):self.wave()
        self.assert_no_motion()

    def test_late_arc_model_veto_happens_before_inference_and_first_motion(self):
        calls=0
        def propose(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==12:raise ValueError('negative arc preflight veto')
            return self.model.propose_rotation(*args,**kwargs)
        self.grasp._model=SimpleNamespace(tip=self.model.tip,propose_rotation=propose)
        with self.assertRaisesRegex(ValueError,'negative arc preflight veto'):self.wave()
        self.assertEqual(calls,12);self.brain.wave_clearance.assert_not_called()
        self.assert_no_motion()


if __name__=='__main__':unittest.main(verbosity=2)
