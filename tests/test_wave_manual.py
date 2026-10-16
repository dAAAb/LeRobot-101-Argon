"""Per-job human wave requests using synthetic images and an in-memory plant."""
from types import SimpleNamespace
import threading,unittest
from unittest.mock import Mock,patch

import test_wave as helpers
import test_wave_raise as raises
import test_wave_retreat_move as retreats
from server.grasp import PreMotionSceneChangedError,WristSceneChangedError,_WaveDecisionCancel
from server.gripper_power import SessionCancelled


class ManualWaveTests(unittest.TestCase):
    model=helpers.default_model()
    capture=helpers.WaveTests.capture
    assert_no_motion=helpers.WaveTests.assert_no_motion
    options=raises.WaveRaiseTests.options
    fake_lift=raises.WaveRaiseTests.fake_lift
    def setUp(self):
        helpers.WaveTests.setUp(self)
        self.manual=threading.Event()
    def wave(self,report=None):
        return self.grasp.wave(report or self.reports.append,self.cancel,manual_override=self.manual)
    def test_requested_wave_skips_inference_and_pixels_but_retains_measured_receipt(self):
        self.manual.set()
        self.capture_hook=lambda frame,index:frame.update(path=str(self.changed)) if index>1 else None
        with patch('server.grasp.unchanged_wrist',side_effect=AssertionError('visual gate used')):
            result=self.wave()
        self.grasp._wave_options.assert_not_called();self.brain.wave_clearance.assert_not_called()
        self.assertTrue(result['manual_override']);self.assertTrue(result['wave_completed'])
        self.assertEqual(result['clearance'],{'source':'human_request','manual_override':True,
                         'visual_clearance_checked':False})
        self.assertEqual(result['cycles'],2)
        self.assertLess(result['measured_roll_range_deg'][0],-9.6)
        self.assertGreater(result['measured_roll_range_deg'][1],9.6)
        self.assertEqual({joint for joint,_ in self.robot.writes},{'wrist_roll'})
        self.assertTrue(all(call['allow_rearm'] is False for call in self.robot.begin_calls))
        self.assertEqual(len([event for event in self.events if event['type']=='manual_wave_override']),1)
        self.assertFalse(self.cancel.is_set());self.assertFalse(self.robot.stop.is_set())
    def test_inference_wait_is_interrupted_without_setting_job_cancel(self):
        def infer(frames,**kwargs):
            signal=kwargs['cancel']
            self.assertIsNot(signal,self.cancel);self.assertFalse(signal.is_set())
            self.assertFalse(hasattr(signal,'set'))
            self.manual.set();self.assertTrue(signal.is_set())
            raise RuntimeError('inference stopped')
        self.grasp._wave_options.side_effect=infer
        result=self.wave()
        self.assertTrue(result['manual_override']);self.assertFalse(self.cancel.is_set())
        self.grasp._wave_options.assert_called_once()
    def test_late_ai_answer_is_discarded_without_dispatching_its_lift(self):
        def infer(*args,**kwargs):
            self.manual.set();return self.options(lift=True)
        self.grasp._wave_options.side_effect=infer;self.grasp.move=Mock()
        result=self.wave()
        self.grasp.move.assert_not_called()
        self.assertEqual(result['raise_steps'],[])
        self.assertEqual(result['clearance']['source'],'human_request')
        self.assertFalse(any(event['type']=='wave_options' for event in self.events))
    def test_original_cancel_or_stop_takes_priority_over_override(self):
        for signal in (self.cancel,self.robot.stop):
            with self.subTest(signal=type(signal).__name__):
                self.cancel.flag=False;self.robot.stop.flag=False;self.manual.clear()
                def infer(*args,**kwargs):
                    self.manual.set();signal.set();raise RuntimeError('decision cancelled')
                self.grasp._wave_options.side_effect=infer
                with self.assertRaises(SessionCancelled):self.wave()
                self.assertTrue(signal.is_set());self.assert_no_motion()
    def test_finished_dispatched_lift_is_kept_then_only_wrist_moves(self):
        self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=self.options(lift=True)
        def move(*args,**kwargs):
            self.manual.set()
            self.assertIs(kwargs['cancel'],self.cancel)
            self.assertFalse(kwargs['cancel'].is_set())
            return self.fake_lift(*args,**kwargs)
        self.grasp.move=Mock(side_effect=move)
        result=self.wave()
        self.grasp.move.assert_called_once();self.grasp._wave_options.assert_called_once()
        self.assertEqual(len(result['raise_steps']),1);self.assertGreater(result['lifted_model_mm'],2.)
        self.assertTrue(result['manual_override']);self.assertTrue(result['wave_completed'])
    def test_pre_dispatch_image_change_can_switch_but_post_motion_error_cannot(self):
        for before_motion in (True,False):
            with self.subTest(before_motion=before_motion):
                self.manual.clear();self.robot.writes.clear();self.events.clear()
                self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=self.options(lift=True)
                def move(*args,**kwargs):
                    self.manual.set()
                    if before_motion:raise PreMotionSceneChangedError('changed before dispatch',self.capture())
                    raise WristSceneChangedError('failed after dispatched move')
                self.grasp.move=Mock(side_effect=move)
                if before_motion:self.assertTrue(self.wave()['manual_override'])
                else:
                    with self.assertRaisesRegex(WristSceneChangedError,'after dispatched'):self.wave()
                    self.assert_no_motion()
    def test_request_at_execution_handoff_is_honored(self):
        def report(message):
            self.reports.append(message)
            if message.startswith('向你揮揮手'):self.manual.set()
        result=self.wave(report)
        self.assertTrue(result['manual_override'])
        self.assertEqual(result['clearance']['source'],'human_request')
    def test_override_during_final_lift_checks_discards_unstarted_motor_step(self):
        self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=self.options(lift=True)
        def report(message):
            self.reports.append(message)
            if message.startswith(('小步移動','抬高約')):self.manual.set()
        result=self.wave(report)
        self.assertTrue(result['manual_override']);self.assertEqual(result['raise_steps'],[])
        self.assertEqual({joint for joint,_ in self.robot.writes},{'wrist_roll'})
    def test_next_wave_and_observation_do_not_inherit_override(self):
        self.manual.set();self.assertTrue(self.wave()['manual_override'])
        self.robot.writes.clear();self.events.clear()
        self.clearance={**self.clearance,'clear_sweep':False}
        with self.assertRaisesRegex(RuntimeError,'空間尚未確認'):
            self.grasp.wave(self.reports.append,self.cancel)
        self.assert_no_motion()
        with self.assertRaisesRegex(RuntimeError,'空間尚未確認'):
            self.grasp.observe_views(self.reports.append,self.cancel)
        self.assert_no_motion()
    def test_state_and_pwm_ownership_are_never_overridden(self):
        self.manual.set()
        for state,owner in [('holding',None),('holding_unverified',None),('idle','gripper-session')]:
            with self.subTest(state=state,owner=owner):
                self.grasp.state=state;self.robot.pwm_owner=owner
                with self.assertRaises(RuntimeError):self.wave()
                self.assert_no_motion();self.assertEqual(self.frames,0)
    def test_manual_wave_requires_current_wrist_generation_and_timestamp(self):
        self.manual.set()
        for change in (lambda f:f.update(source='observer'),
                       lambda f:f.update(camera_generation='reconnected'),
                       lambda f:f.update(capture_monotonic=1.)):
            self.frames=0
            self.capture_hook=lambda frame,index:change(frame) if index==2 else None
            with self.assertRaises((ValueError,RuntimeError)):self.wave()
            self.assert_no_motion()
    def test_expired_camera_is_not_overridden(self):
        self.manual.set();self.robot.camera.fresh=lambda:False
        with self.assertRaisesRegex(RuntimeError,'過期'):self.wave()
        self.assert_no_motion()
    def test_full_negative_arc_screen_failure_prevents_every_write(self):
        self.manual.set();calls=0
        def propose(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==12:raise ValueError('negative arc table veto')
            return self.model.propose_rotation(*args,**kwargs)
        self.grasp._model=SimpleNamespace(tip=self.model.tip,propose_rotation=propose)
        with self.assertRaisesRegex(ValueError,'table veto'):self.wave()
        self.assertEqual(calls,12);self.assert_no_motion()
    def test_manual_cancel_during_motor_write_remains_effective(self):
        self.manual.set();self.robot.before_glide=self.cancel.set
        with self.assertRaises(SessionCancelled):self.wave()
        self.assert_no_motion()
    def test_manual_drift_and_motor_faults_remain_effective(self):
        self.manual.set()
        def drift_after_start_frame(frame,index):
            if index==2:self.robot.actual['gripper']+=5
        self.capture_hook=drift_after_start_frame
        with self.assertRaisesRegex(RuntimeError,'夾爪開口改變'):self.wave()
        self.assert_no_motion()
        self.robot.actual['gripper']-=5;self.capture_hook=None
        self.robot.before_glide=lambda:(_ for _ in ()).throw(RuntimeError('motor temperature fault'))
        with self.assertRaisesRegex(RuntimeError,'temperature'):self.wave()
        self.assert_no_motion()
    def test_manual_requires_actual_completed_extremes(self):
        self.manual.set()
        self.grasp.roll_sweep=Mock(side_effect=lambda *args,**kwargs:self.capture())
        with self.assertRaisesRegex(RuntimeError,'幅度.*不足'):self.wave()
        self.assertFalse(any(event['type']=='wave_result' for event in self.events))
    def test_no_override_keeps_original_receipt_and_visual_requirements(self):
        result=self.wave()
        self.assertNotIn('manual_override',result)
        self.assertEqual(result['clearance'],self.clearance)
        self.grasp._wave_options.assert_called_once()

    def high_pose(self):
        self.robot.actual.update(dict(shoulder_pan=2162,shoulder_lift=1141,elbow_flex=2694,
                                     wrist_flex=2429,wrist_roll=984,gripper=1857))
        self.robot.goals=dict(self.robot.actual);self.robot.bias=dict.fromkeys(self.robot.actual,0)
        self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=self.options()
    def test_override_interrupts_retreat_ai_assessment(self):
        self.high_pose()
        def decide(*args,**kwargs):
            self.manual.set();self.assertTrue(kwargs['cancel'].is_set())
            raise SessionCancelled('retreat inference stopped')
        with patch('server.wave_planner.assess_retreat_options',side_effect=decide) as assess:
            result=self.wave()
        assess.assert_called_once();self.assertTrue(result['manual_override'])
        self.assertEqual(result['retreat_steps'],[])
        self.assertEqual({joint for joint,_ in self.robot.writes},{'wrist_roll'})
    def test_override_interrupts_jev_ranking_and_discards_late_choice(self):
        self.high_pose()
        def choose(*args,**kwargs):
            self.manual.set();self.assertTrue(kwargs['cancel'].is_set())
            return {'status':'selected','selected':'left'}
        self.grasp.tactician=SimpleNamespace(choose=Mock(side_effect=choose));self.grasp.move=Mock()
        options=retreats.RetreatMoveTests.options(self,left=True,right=True,backward=True)
        with patch('server.wave_planner.assess_retreat_options',return_value=options):
            result=self.wave()
        self.grasp.tactician.choose.assert_called_once();self.grasp.move.assert_not_called()
        self.assertTrue(result['manual_override']);self.assertEqual(result['retreat_steps'],[])
    def test_completed_retreat_is_preserved_before_waving(self):
        self.high_pose()
        options=retreats.RetreatMoveTests.options(self,left=True)
        def move(direction,distance,reference,report,**kwargs):
            self.assertIs(kwargs['cancel'],self.cancel)
            before=self.capture();proposal=kwargs['assessed_lift']['proposal']
            self.manual.set();self.assertFalse(kwargs['cancel'].is_set())
            self.robot.actual.update(proposal['joint_targets']);self.robot.goals.update(proposal['physical_goal_targets'])
            return {'before':before,'after':self.capture(),'direction':direction}
        self.grasp.move=Mock(side_effect=move)
        with patch('server.wave_planner.assess_retreat_options',return_value=options) as assess:
            result=self.wave()
        assess.assert_called_once();self.grasp.move.assert_called_once()
        self.assertEqual(len(result['retreat_steps']),1);self.assertGreater(result['retreated_model_mm'],.5)
        self.assertTrue(result['manual_override'])


class DecisionCancelTests(unittest.TestCase):
    def test_combined_wait_wakes_on_manual_without_mutating_cancellation(self):
        cancel=threading.Event();manual=threading.Event();signal=_WaveDecisionCancel(cancel,manual)
        timer=threading.Timer(.03,manual.set);timer.start()
        try:self.assertTrue(signal.wait(.5))
        finally:timer.join()
        self.assertFalse(cancel.is_set());self.assertFalse(hasattr(signal,'set'))
    def test_combined_wait_obeys_timeout_and_job_cancel(self):
        cancel=threading.Event();manual=threading.Event();signal=_WaveDecisionCancel(cancel,manual)
        self.assertFalse(signal.wait(.001));cancel.set();self.assertTrue(signal.wait(0))


if __name__=='__main__':unittest.main()
