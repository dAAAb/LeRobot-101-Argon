"""Seven-view controller tests: synthetic images, fake plant, real local CAD."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import test_wave as helpers
from server.gripper_power import SessionCancelled
from server.local_cartesian import NAMES, default_model
from server.grasp import WAVE_ENDPOINT_TOLERANCE_TICKS as ROLL_TOLERANCE


class ObservationViewsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=default_model()
    setUp=helpers.WaveTests.setUp
    capture=helpers.WaveTests.capture

    def observe(self):return self.grasp.observe_views(self.reports.append,self.cancel)
    def assert_no_receipt(self):
        self.assertFalse(any(e.get('type') in ('observation_views','wave_result') for e in self.events))
    def assert_no_motion(self):
        self.assertEqual(self.robot.writes,[]);self.assert_no_receipt()

    def test_seven_views_return_to_center_preserve_holds_and_use_measured_fk(self):
        initial=self.robot.read();holds=dict(self.robot.goals)
        result=self.observe();frames=result['frames']
        self.assertEqual(len(frames),7)
        self.assertEqual([f['joints']['wrist_roll']-initial['wrist_roll'] for f in frames],
                         [0,113,227,0,-113,-227,0])
        self.assertEqual(self.robot.actual,initial);self.assertEqual(self.robot.goals,holds)
        self.assertEqual(result['body_origin'],initial)
        self.assertEqual(result['initial_hold_goals'],holds)
        self.assertEqual(result['camera_generation'],'synthetic-wrist')
        self.assertEqual(result['initial_frame']['capture_monotonic'],1.)
        self.assertEqual(frames[0]['capture_monotonic'],2.)
        self.assertTrue(all(b['capture_monotonic']>a['capture_monotonic'] for a,b in zip(frames,frames[1:])))
        self.assertEqual(result['return_error_deg'],0.)
        self.assertEqual(result['endpoint_tolerance_deg'],ROLL_TOLERANCE*360/4096)
        expected=[-227*360/4096,227*360/4096]
        self.assertEqual(result['measured_roll_range_deg'],expected)
        positions=[self.model.fk.transforms(f['joints'])['wrist_camera_optical'][:3,3] for f in frames]
        baseline=max(np.linalg.norm(a-b)*1000 for a in positions for b in positions)
        self.assertAlmostEqual(result['camera_baseline_model_mm'],baseline,places=9)
        self.assertGreater(baseline,20.);self.assertLess(baseline,40.)
        self.assertTrue(result['not_external_metrology'])
        for claim in ('wave_completed','cycles','grasp_confirmed','metric_depth','success'):
            self.assertNotIn(claim,result)
        self.assertEqual(self.events[-1]['type'],'observation_views')
        self.assertEqual(len(self.robot.calls),32)
        self.assertEqual({n for n,v in self.robot.writes},{'wrist_roll'})
        for call in self.robot.calls:
            self.assertEqual(set(call['targets']),{'wrist_roll'})
            self.assertLessEqual(abs(call['targets']['wrist_roll']-call['before']['wrist_roll']),30)
            self.assertEqual(call['physical']['wrist_roll']-call['targets']['wrist_roll'],7)
            self.assertFalse(call['kwargs']['allow_rearm'])
            self.assertEqual(call['kwargs']['precision'],ROLL_TOLERANCE)
            for n in NAMES:
                if n!='wrist_roll':self.assertEqual(call['kwargs']['measured_bounds'][n],[initial[n]-4,initial[n]+4])

    def test_model_baseline_uses_returned_actual_poses_not_requested_extremes(self):
        origin=self.robot.actual['wrist_roll'];calls=[]
        def measured_shortfall(goal,reference,cancel,**kwargs):
            calls.append(kwargs)
            self.robot.actual['wrist_roll']=goal-(5 if goal>origin else -5 if goal<origin else 0)
            return self.capture()
        self.grasp.roll_sweep=Mock(side_effect=measured_shortfall)
        result=self.observe()
        self.assertEqual(result['measured_roll_range_deg'],[-222*360/4096,222*360/4096])
        frames=result['frames'];poses=[self.model.fk.transforms(f['joints'])['wrist_camera_optical'][:3,3] for f in frames]
        actual=max(np.linalg.norm(a-b)*1000 for a in poses for b in poses)
        nominal=[]
        for offset in (227,-227):
            raw={**frames[0]['joints'],'wrist_roll':origin+offset}
            nominal.append(self.model.fk.transforms(raw)['wrist_camera_optical'][:3,3])
        nominal_baseline=np.linalg.norm(nominal[0]-nominal[1])*1000
        self.assertAlmostEqual(result['camera_baseline_model_mm'],actual,places=9)
        self.assertGreater(abs(actual-nominal_baseline),.1)
        self.assertEqual(len(calls),6)
        self.assertTrue(all(c['body_origin']==frames[0]['joints'] and
                            c['camera_generation']=='synthetic-wrist' for c in calls))

    def test_holding_and_owned_jaw_reject_before_capture(self):
        for state,owner in [('holding',None),('holding_unverified',None),('idle','owner')]:
            with self.subTest(state=state,owner=owner):
                self.grasp.state=state;self.robot.pwm_owner=owner
                with self.assertRaises(RuntimeError):self.observe()
                self.assertEqual(self.frames,0);self.brain.wave_clearance.assert_not_called()
                self.assert_no_motion()

    def test_empty_jaw_and_external_clearance_are_required(self):
        valid=dict(self.clearance)
        for field in ('gripper_empty','clear_sweep','human_clear','confidence'):
            with self.subTest(field=field):
                self.clearance={**valid,field:.899 if field=='confidence' else False}
                with self.assertRaises(RuntimeError):self.observe()
                self.assert_no_motion()

    def test_inference_changed_scene_never_moves(self):
        def changed(frame,index):
            if index==2:frame['path']=str(self.changed)
        self.capture_hook=changed
        with self.assertRaisesRegex(ValueError,'場景已改變'):self.observe()
        self.assert_no_motion()

    def test_inference_body_or_generation_change_never_moves(self):
        for kind in ('body','generation'):
            with self.subTest(kind=kind):
                self.frames=0
                def changed(frame,index):
                    if index==2:
                        if kind=='body':frame['joints']['elbow_flex']+=4
                        else:frame['camera_generation']='reconnected'
                self.capture_hook=changed
                with self.assertRaises(ValueError):self.observe()
                self.assert_no_motion()

    def test_preexisting_cancel_prevents_capture_and_preflight(self):
        self.cancel.set()
        with self.assertRaises(SessionCancelled):self.observe()
        self.assertEqual(self.frames,0);self.brain.wave_clearance.assert_not_called()
        self.assert_no_motion()

    def test_cancel_after_inference_prevents_first_motion(self):
        def infer(*args,**kwargs):self.cancel.set();return self.clearance
        self.brain.wave_clearance.side_effect=infer
        with self.assertRaises(SessionCancelled):self.observe()
        self.assert_no_motion()

    def test_cancel_between_views_does_not_issue_return_trajectory(self):
        self.cancel.cancel_after_wait=1
        with self.assertRaises(SessionCancelled):self.observe()
        self.assertEqual(len(self.robot.calls),4)
        self.assertEqual(self.robot.actual['wrist_roll'],987+113)
        self.assert_no_receipt()

    def test_prewrite_cancel_or_reconnect_prevents_first_goal(self):
        def cancelled():self.cancel.set()
        self.robot.before_glide=cancelled
        with self.assertRaises(SessionCancelled):self.observe()
        self.assert_no_motion()
        self.cancel.flag=False
        def reconnect():self.robot.camera.generation='reconnected'
        self.robot.before_glide=reconnect
        with self.assertRaisesRegex(RuntimeError,'相機重新連接'):self.observe()
        self.assert_no_motion()

    def test_endpoint_generation_change_stops_before_next_view(self):
        def changed(frame,index):
            if index==3:
                self.robot.camera.generation='reconnected';frame['camera_generation']='reconnected'
        self.capture_hook=changed
        with self.assertRaisesRegex(RuntimeError,'相機重新連接'):self.observe()
        self.assertEqual(len(self.robot.calls),4);self.assert_no_receipt()

    def test_endpoint_body_drift_cannot_reset_shared_corridor(self):
        def changed(frame,index):
            if index==3:frame['joints']['shoulder_pan']+=5
        self.capture_hook=changed
        with self.assertRaisesRegex(RuntimeError,'其他關節偏移'):self.observe()
        self.assertEqual(len(self.robot.calls),4);self.assert_no_receipt()

    def test_endpoint_jaw_drift_does_not_use_roll_tolerance(self):
        def changed(frame,index):
            if index==3:frame['joints']['gripper']+=5
        self.capture_hook=changed
        with self.assertRaisesRegex(RuntimeError,'夾爪開口改變'):self.observe()
        self.assertEqual(len(self.robot.calls),4);self.assert_no_receipt()

    def test_endpoint_observer_or_stale_frame_is_not_evidence(self):
        for kind in ('observer','stale'):
            with self.subTest(kind=kind):
                self.frames=0;self.robot=helpers.FakeRobot();self.grasp.robot=self.robot
                def changed(frame,index):
                    if index==3:
                        if kind=='observer':frame['source']='observer'
                        else:frame['capture_monotonic']=2.
                self.capture_hook=changed
                with self.assertRaises((ValueError,RuntimeError)):self.observe()
                self.assertEqual(len(self.robot.calls),4);self.assert_no_receipt()

    def test_missing_final_return_cannot_produce_observation_receipt(self):
        legs=0
        def missing_return(goal,reference,cancel,**kwargs):
            nonlocal legs
            legs+=1;self.robot.actual['wrist_roll']=goal+(ROLL_TOLERANCE+1 if legs==6 else 0)
            return self.capture()
        self.grasp.roll_sweep=Mock(side_effect=missing_return)
        with self.assertRaisesRegex(RuntimeError,'實測角度或回程不足'):self.observe()
        self.assert_no_receipt()

    def test_far_arc_preflight_veto_precedes_brain_and_all_motion(self):
        calls=0
        def propose(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==24:raise ValueError('far arc table veto')
            return self.model.propose_rotation(*args,**kwargs)
        self.grasp._model=SimpleNamespace(propose_rotation=propose)
        with self.assertRaisesRegex(ValueError,'far arc table veto'):self.observe()
        self.assertEqual(calls,24);self.brain.wave_clearance.assert_not_called()
        self.assert_no_motion()


if __name__=='__main__':unittest.main(verbosity=2)
