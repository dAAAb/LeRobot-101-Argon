"""Gesture endpoint tolerances through real glide/CAD, with no hardware I/O.

The simulated wrist develops a few encoder ticks of steady load error.
This tests measured receipts and the actual endpoint wait, rather than making
the motion stub succeed whenever its caller asks for a looser tolerance.
"""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import test_local_feedback as controller_helpers
import test_wave as wave_helpers
from server.robot import JOINTS, Robot
from server.grasp import WAVE_ENDPOINT_TOLERANCE_TICKS as WAVE_TOLERANCE


class DeadbandProtocol(controller_helpers.FakeProtocol):
    def __init__(self, calibration, residual=5, mode='forward'):
        super().__init__(calibration)
        self.residual=residual;self.mode=mode;self.segment=None
        self.loaded_offset=None;self.base_offset=None

    def write(self, port, servo, address, value):
        previous=self.reg[servo][56]
        result=super().write(port,servo,address,value)
        if servo==5 and address==42 and self.segment:
            actual_start,physical_start=self.segment
            delta=value-physical_start
            direction=(delta>0)-(delta<0)
            movement=direction*max(0,abs(delta)-self.residual)
            if self.mode=='frozen':self.reg[servo][56]=actual_start
            elif self.mode=='reverse':self.reg[servo][56]=actual_start-movement
            elif self.mode=='accumulating':self.reg[servo][56]=actual_start+movement
            elif self.mode=='backlash':
                if self.base_offset is None:self.base_offset=physical_start-actual_start
                if delta:self.loaded_offset=self.base_offset+direction*self.residual
                self.reg[servo][56]=previous if self.loaded_offset is None else value-self.loaded_offset
            else:
                if self.loaded_offset is None and delta:
                    self.loaded_offset=physical_start-actual_start+direction*self.residual
                self.reg[servo][56]=previous if self.loaded_offset is None else value-self.loaded_offset
        return result

    write2ByteTxRx=write
    write1ByteTxRx=write


class WaveEndpointToleranceTests(unittest.TestCase):
    model=wave_helpers.default_model()
    capture=wave_helpers.WaveTests.capture

    def setUp(self):
        wave_helpers.WaveTests.setUp(self)
        initial=self.robot.read();holds=dict(self.robot.goals)
        camera=self.robot.camera
        calibration={name:{'id':index+1,'range_min':0,'range_max':4095}
                     for index,name in enumerate(JOINTS)}
        self.robot=Robot({'follower_port':'fake://no-hardware',
                          'max_ticks_per_second':90,'max_tracking_error_ticks':90},
                         calibration,self.tmp.name,camera)
        self.bus=DeadbandProtocol(calibration)
        for name,entry in calibration.items():
            self.bus.reg[entry['id']].update({56:initial[name],42:holds[name]})
        self.robot.ph=self.bus;self.robot.port=SimpleNamespace(is_using=False);self.robot.connected=True
        self.robot.armed=True;self.robot.goals=holds
        self.robot.bias={name:holds[name]-initial[name] for name in JOINTS}
        self.clock=controller_helpers.Clock()
        self.robot.stop=controller_helpers.FakeStop(self.clock)
        self.robot.gripper_power.stop_latch=self.robot.stop.set
        timer=patch('server.robot.time.monotonic',self.clock.time)
        timer.start();self.addCleanup(timer.stop)
        self.grasp.robot=self.robot
        self.calls=[]
        real_glide=self.robot.glide

        def glide(targets,**kwargs):
            before=self.robot.read()
            self.bus.segment=(before['wrist_roll'],self.robot.goals['wrist_roll'])
            call={'before':before,'targets':dict(targets),'kwargs':kwargs}
            self.calls.append(call)
            try:return real_glide(targets,**kwargs)
            finally:call['after']=self.robot.read()

        self.robot.glide=glide
        self.initial=initial;self.initial_holds=holds

    def wave(self,manual=False):
        override=threading.Event()
        if manual:override.set()
        return self.grasp.wave(self.reports.append,self.cancel,manual_override=override)

    def assert_measured_cycles(self,manual):
        origin=self.robot.read()['wrist_roll']
        result=self.wave(manual)
        self.assertTrue(result['wave_completed'])
        self.assertEqual(result['cycles'],2)
        self.assertEqual(result.get('manual_override',False),manual)
        offsets=(113,0,-113,0)*2
        self.assertEqual(len(result['frames']),8)
        errors=[frame['joints']['wrist_roll']-(origin+offset)
                for frame,offset in zip(result['frames'],offsets)]
        self.assertTrue(all(abs(error)<=WAVE_TOLERANCE for error in errors),errors)
        self.assertLessEqual(abs(result['return_error_deg']),WAVE_TOLERANCE*360/4096)
        self.assertTrue(self.calls)
        # This is the reported failure: glide must accept the first measured
        # endpoint five ticks short. Subsequent holds retain that real offset.
        self.assertEqual(self.calls[0]['after']['wrist_roll']-
                         self.calls[0]['targets']['wrist_roll'],-5)
        for call in self.calls:
            self.assertEqual(call['kwargs']['precision'],WAVE_TOLERANCE)
            self.assertEqual(set(call['targets']),{'wrist_roll'})
            self.assertFalse(call['kwargs']['allow_rearm'])
            self.assertLessEqual(abs(call['targets']['wrist_roll']-
                                     call['before']['wrist_roll']),30)
            for bounds in ('raw_bounds','measured_bounds'):
                low,high=call['kwargs'][bounds]['wrist_roll']
                self.assertGreaterEqual(low,origin-113-12)
                self.assertLessEqual(high,origin+113+12)
        for servo,address,value in self.bus.events:
            self.assertEqual((servo,address),(5,42))
            self.assertLessEqual(abs(value-origin),113+12)
        current=self.robot.read()
        for name in JOINTS:
            if name!='wrist_roll':
                self.assertEqual(current[name],self.initial[name])
                self.assertEqual(self.robot.goals[name],self.initial_holds[name])
        self.assertFalse(self.robot.stop.is_set())
        self.assertEqual(self.events[-1]['type'],'wave_result')

    def test_five_tick_load_error_completes_both_measured_cycles(self):
        self.assert_measured_cycles(False)

    def test_manual_wave_accepts_five_tick_load_error_with_same_measured_bounds(self):
        self.assert_measured_cycles(True)

    def test_direction_dependent_friction_completes_both_cycles_after_reversal(self):
        self.bus.mode='backlash'
        self.assert_measured_cycles(True)
        # The retained +5-tick loaded offset and the new -5-tick offset combine
        # at direction reversal; the real controller must accept measured +10.
        reversal_errors=[call['after']['wrist_roll']-call['targets']['wrist_roll']
                         for call in self.calls
                         if call['targets']['wrist_roll']<call['before']['wrist_roll']]
        self.assertIn(10,reversal_errors)

    def test_accumulating_segment_error_cannot_expand_original_sweep(self):
        self.bus.mode='accumulating'
        with self.assertRaisesRegex(ValueError,'screened raw interval'):
            self.wave(True)
        self.assertEqual(len(self.calls),4)
        self.assertTrue(self.robot.stop.is_set())
        self.assertFalse(any(event['type']=='wave_result' for event in self.events))

    def test_default_roll_sweep_keeps_three_tick_precision(self):
        frame=self.capture()
        with self.assertRaisesRegex(RuntimeError,'未到達目標位置'):
            self.grasp.roll_sweep(frame['joints']['wrist_roll']+30,frame,self.cancel)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0]['kwargs']['precision'],3)
        self.assertTrue(self.robot.stop.is_set())
        self.assertFalse(any(event['type']=='observation_views' for event in self.events))

    def test_seven_views_accept_load_error_and_report_actual_camera_baseline(self):
        self.bus.mode='backlash'
        origin=self.robot.read()['wrist_roll']
        result=self.grasp.observe_views(self.reports.append,self.cancel)
        frames=result['frames'];offsets=(0,113,227,0,-113,-227,0)
        actual_offsets=[frame['joints']['wrist_roll']-origin for frame in frames]
        self.assertEqual(len(frames),7)
        self.assertTrue(all(abs(actual-expected)<=WAVE_TOLERANCE
                            for actual,expected in zip(actual_offsets,offsets)))
        self.assertEqual(self.calls[0]['after']['wrist_roll']-
                         self.calls[0]['targets']['wrist_roll'],-5)
        self.assertEqual(result['endpoint_tolerance_deg'],WAVE_TOLERANCE*360/4096)
        self.assertEqual(result['measured_roll_range_deg'],
                         [min(actual_offsets)*360/4096,max(actual_offsets)*360/4096])
        self.assertEqual(result['return_error_deg'],actual_offsets[-1]*360/4096)
        camera_positions=[self.model.fk.transforms(frame['joints'])['wrist_camera_optical'][:3,3]
                          for frame in frames]
        baseline=max(float(np.linalg.norm(a-b))*1000
                     for a in camera_positions for b in camera_positions)
        self.assertAlmostEqual(result['camera_baseline_model_mm'],baseline,places=9)
        self.assertTrue(result['not_external_metrology'])
        self.assertNotIn('wave_completed',result)
        self.assertFalse(self.robot.stop.is_set())
        for call in self.calls:
            self.assertEqual(call['kwargs']['precision'],WAVE_TOLERANCE)
            for key in ('raw_bounds','measured_bounds'):
                low,high=call['kwargs'][key]['wrist_roll']
                self.assertGreaterEqual(low,origin-227-12)
                self.assertLessEqual(high,origin+227+12)
            for name in JOINTS:
                if name not in ('wrist_roll','gripper'):
                    self.assertEqual(call['kwargs']['measured_bounds'][name],
                                     [self.initial[name]-4,self.initial[name]+4])
        for servo,address,value in self.bus.events:
            self.assertEqual((servo,address),(5,42))
            self.assertLessEqual(abs(value-origin),227+12)
        current=self.robot.read()
        self.assertTrue(all(current[name]==self.initial[name]
                            for name in JOINTS if name!='wrist_roll'))

    def test_more_than_ten_ticks_still_latches_stop_without_success_receipt(self):
        self.bus.residual=WAVE_TOLERANCE+1
        with self.assertRaisesRegex(RuntimeError,'未到達目標位置'):
            self.wave(True)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0]['kwargs']['precision'],WAVE_TOLERANCE)
        self.assertTrue(self.robot.stop.is_set())
        self.assertFalse(any(event['type']=='wave_result' for event in self.events))

    def test_frozen_wrist_cannot_use_gesture_tolerance_as_progress(self):
        self.bus.mode='frozen'
        with self.assertRaisesRegex(RuntimeError,'未到達目標位置'):
            self.wave(True)
        self.assertTrue(self.robot.stop.is_set())
        self.assertFalse(any(event['type']=='wave_result' for event in self.events))

    def test_reverse_wrist_is_rejected_by_unchanged_measured_bounds(self):
        self.bus.mode='reverse'
        with self.assertRaisesRegex(ValueError,'screened raw interval'):
            self.wave(True)
        self.assertTrue(self.robot.stop.is_set())
        self.assertFalse(any(event['type']=='wave_result' for event in self.events))

    def test_settled_endpoint_outside_ten_ticks_is_not_accepted(self):
        def unstable_capture(frame,index):
            if index==3:frame['joints']['wrist_roll']=self.initial['wrist_roll']+113-(WAVE_TOLERANCE+1)
        self.capture_hook=unstable_capture
        with self.assertRaisesRegex(RuntimeError,'終點未穩定'):
            self.wave(True)
        self.assertFalse(any(event['type']=='wave_result' for event in self.events))

    def test_cycle_endpoints_cannot_reanchor_after_each_leg(self):
        accumulated=0;legs=0
        def drifting_receipt(goal,reference,cancel,**kwargs):
            nonlocal accumulated,legs
            legs+=1;accumulated+=5
            # Every leg could look locally close after adopting its own new
            # center. The wave receipt must use its one original center.
            self.bus.reg[5][56]=goal+accumulated
            return self.capture()
        self.grasp.roll_sweep=Mock(side_effect=drifting_receipt)
        with self.assertRaisesRegex(RuntimeError,'幅度.*不足'):
            self.wave(True)
        self.assertEqual(legs,4)
        self.assertFalse(any(event['type']=='wave_result' for event in self.events))

    def test_final_return_eleven_ticks_away_rejects_complete_receipt(self):
        legs=0
        def wrong_final_return(goal,reference,cancel,**kwargs):
            nonlocal legs
            legs+=1;self.bus.reg[5][56]=goal+(WAVE_TOLERANCE+1 if legs==8 else 0)
            return self.capture()
        self.grasp.roll_sweep=Mock(side_effect=wrong_final_return)
        with self.assertRaisesRegex(RuntimeError,'幅度.*不足'):
            self.wave(True)
        self.assertEqual(legs,8)
        self.assertFalse(any(event['type']=='wave_result' for event in self.events))

    def test_tolerance_is_not_an_arbitrary_precision_escape_hatch(self):
        frame=self.capture();goal=frame['joints']['wrist_roll']+30
        for invalid in (None,True,3.,float(WAVE_TOLERANCE),0,4,9,11,30,'10'):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    self.grasp.roll_sweep(goal,frame,self.cancel,
                                          endpoint_tolerance_ticks=invalid)
        self.assertEqual(self.bus.events,[])


if __name__=='__main__':unittest.main()
