"""Wave fallback sequencing with synthetic views and a fake lift, no hardware."""
import unittest
from unittest.mock import Mock,patch
import numpy as np
import test_wave as helpers
from server.gripper_power import SessionCancelled
from server.grasp import WristSceneChangedError,PreMotionSceneChangedError


class WaveRaiseTests(unittest.TestCase):
    model=helpers.default_model()
    setUp=helpers.WaveTests.setUp
    capture=helpers.WaveTests.capture

    def options(self,*,wave=False,lift=False):
        return {'wave':{**self.clearance,'clear_sweep':wave,'reason':'圓筒可能擋住揮手'},
                'lift':{'gripper_empty':True,'human_clear':True,'clear_path':lift,
                        'confidence':.95,'reason':'上方可見空間足夠' if lift else '看不清楚上方'}}
    def fake_lift(self,direction,distance,reference,report,**kwargs):
        self.assertEqual(direction,'up');self.assertTrue(kwargs['free_space_lift'])
        before=self.capture()
        proposal=kwargs['assessed_lift']['proposal']
        initial=self.model.tip(self.robot.actual)
        self.robot.actual.update(proposal['joint_targets'])
        self.robot.goals.update(proposal['physical_goal_targets'])
        delta=(self.model.tip(self.robot.actual)[:3,3]-initial[:3,3])*1000
        return {'before':before,'after':self.capture(),'actual_delta_model_mm':delta.tolist()}
    def run_wave(self):return self.grasp.wave(self.reports.append,self.cancel)

    def test_blocked_roll_selects_independently_clear_lift_then_reassesses(self):
        self.grasp._wave_options.side_effect=[self.options(lift=True),self.options(wave=True)]
        self.grasp.move=Mock(side_effect=self.fake_lift)
        result=self.run_wave()
        self.assertTrue(result['wave_completed']);self.grasp.move.assert_called_once()
        self.assertEqual(len(result['raise_steps']),1)
        self.assertGreater(result['lifted_model_mm'],2)
        calls=self.grasp._wave_options.call_args_list
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0].kwargs['step_count'],0);self.assertEqual(calls[1].kwargs['step_count'],1)
        self.assertAlmostEqual(calls[1].kwargs['lifted_mm'],result['lifted_model_mm'])
        self.assertGreater(calls[1].args[0][-1]['capture_monotonic'],calls[0].args[0][-1]['capture_monotonic'])
        candidate=calls[0].kwargs['lift_proposal']
        self.assertEqual(candidate['delta_ticks']['shoulder_pan'],0)
        self.assertEqual(candidate['delta_ticks']['wrist_roll'],0)
        self.assertEqual(candidate['clearance']['actual_tracking_envelope_ticks']['shoulder_pan'],12)
        self.assertEqual(candidate['clearance']['actual_tracking_envelope_ticks']['wrist_roll'],13)
        self.assertEqual(candidate['motion_kind'],'elbow_raise_macro')
        self.assertGreater(np.linalg.norm(candidate['requested_delta_m'])*1000,8.)
        self.assertIs(self.grasp.move.call_args.kwargs['assessed_lift']['proposal'],candidate)
        self.assertTrue(candidate['clearance']['model_table_pass'])
    def test_blocked_lift_never_moves_or_reasks(self):
        for field in ('gripper_empty','human_clear','clear_path','confidence'):
            options=self.options(lift=True);options['lift'][field]=.699 if field=='confidence' else False
            self.grasp._wave_options.reset_mock();self.grasp._wave_options.side_effect=None
            self.grasp._wave_options.return_value=options;self.grasp.move=Mock()
            with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'空間尚未確認'):self.run_wave()
            self.grasp.move.assert_not_called();self.grasp._wave_options.assert_called_once()
            self.assertEqual(self.robot.writes,[])
    def test_local_lift_and_wave_have_distinct_action_thresholds(self):
        options=self.options(lift=True);options['lift']['confidence']=.7
        options['wave']['confidence']=.799;options['wave']['clear_sweep']=True
        self.grasp._wave_options.side_effect=[options,self.options(wave=True)]
        self.grasp.move=Mock(side_effect=self.fake_lift)
        result=self.run_wave()
        self.grasp.move.assert_called_once()
        self.assertTrue(result['wave_completed'])
    def test_scene_change_after_lift_verdict_prevents_move(self):
        self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=self.options(lift=True)
        self.grasp.move=Mock()
        # Unknown/ordinary validation errors must not be treated as the
        # specifically classified image-difference refresh opportunity.
        with patch('server.grasp.unchanged_wrist',side_effect=ValueError('場景已改變')):
            with self.assertRaisesRegex(ValueError,'場景已改變'):self.run_wave()
        self.grasp.move.assert_not_called();self.assertEqual(self.robot.writes,[])
        self.grasp._wave_options.assert_called_once()
    def test_scene_refresh_uses_new_view_and_rejected_verdict_never_moves(self):
        self.grasp._wave_options.side_effect=[self.options(lift=True),self.options()]
        self.capture_hook=lambda f,i:f.update(path=str(self.changed)) if i>=2 else None
        self.grasp.move=Mock()
        with self.assertRaisesRegex(RuntimeError,'空間尚未確認'):self.run_wave()
        self.grasp.move.assert_not_called();self.assertEqual(self.robot.writes,[])
        calls=self.grasp._wave_options.call_args_list
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0].args[0][-1]['path'],str(self.base))
        self.assertEqual(calls[1].args[0][-1]['path'],str(self.changed))
        self.assertGreater(calls[1].args[0][-1]['capture_monotonic'],calls[0].args[0][-1]['capture_monotonic'])
        events=[e for e in self.events if e['type']=='wave_scene_refresh']
        self.assertEqual([e['refresh_count'] for e in events],[1])
        self.assertEqual(events[0]['step_count'],0)
    def test_scene_refresh_stops_at_third_change_without_motion(self):
        self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=self.options(lift=True)
        self.capture_hook=lambda f,i:f.update(path=str(self.changed)) if i%2==0 else None
        self.grasp.move=Mock()
        with self.assertRaises(WristSceneChangedError):self.run_wave()
        self.grasp.move.assert_not_called();self.assertEqual(self.robot.writes,[])
        self.assertEqual(self.grasp._wave_options.call_count,3)
        self.assertEqual([c.args[0][-1]['capture_monotonic'] for c in self.grasp._wave_options.call_args_list],[1.,2.,3.])
        self.assertEqual([e['refresh_count'] for e in self.events if e['type']=='wave_scene_refresh'],[1,2])
    def test_real_move_labels_only_its_two_final_image_comparisons_before_writes(self):
        for changed_capture in (2,3):
            with self.subTest(changed_capture=changed_capture):
                self.frames=0;self.events.clear();self.capture_hook=None
                reference=self.capture()
                proposal,envelope=self.grasp._wave_lift_proposal(8,reference['joints'],reference['joints'])
                self.capture_hook=lambda f,i:f.update(path=str(self.changed)) if i==changed_capture else None
                with self.assertRaises(PreMotionSceneChangedError) as caught:
                    self.grasp.move('up',float(np.linalg.norm(proposal['requested_delta_m'])*1000),
                        reference,self.reports.append,free_space_lift=True,cancel=self.cancel,
                        assessed_lift={'proposal':proposal,'envelope':envelope,
                                       'origin':reference['joints'],'hold_goals':dict(self.robot.goals)})
                self.assertEqual(caught.exception.frame['capture_monotonic'],float(changed_capture))
                self.assertEqual(caught.exception.frame['path'],str(self.changed))
                self.assertIsInstance(caught.exception.__cause__,WristSceneChangedError)
                self.assertEqual(self.robot.writes,[]);self.assertEqual(self.robot.calls,[])
                self.assertFalse(any(e['type']=='proposal' for e in self.events))

    def test_late_move_image_change_requires_new_proposal_and_rejected_verdict_never_moves(self):
        for changed_capture in (3,4):
            with self.subTest(changed_capture=changed_capture):
                self.frames=0;self.events.clear()
                self.grasp._wave_options.reset_mock()
                self.grasp._wave_options.side_effect=[self.options(lift=True),self.options()]
                self.capture_hook=lambda f,i:f.update(path=str(self.changed)) if i>=changed_capture else None
                with patch.object(self.grasp,'_wave_raise_proposal',wraps=self.grasp._wave_raise_proposal) as proposals:
                    with self.assertRaisesRegex(RuntimeError,'空間尚未確認'):self.run_wave()
                    self.assertEqual(proposals.call_count,2)
                self.assertEqual(self.robot.writes,[]);self.assertEqual(self.robot.calls,[])
                calls=self.grasp._wave_options.call_args_list
                self.assertEqual(len(calls),2)
                self.assertEqual(calls[1].args[0][-1]['path'],str(self.changed))
                self.assertEqual(calls[1].args[0][-1]['capture_monotonic'],float(changed_capture))
                self.assertIsNot(calls[0].kwargs['lift_proposal'],calls[1].kwargs['lift_proposal'])
                self.assertEqual([e['refresh_count'] for e in self.events if e['type']=='wave_scene_refresh'],[1])

    def test_outer_and_late_move_changes_share_two_refresh_budget(self):
        self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=self.options(lift=True)
        self.capture_hook=lambda f,i:f.update(path=str(self.changed)) if i in (2,3,6) else None
        with self.assertRaises(PreMotionSceneChangedError) as caught:self.run_wave()
        self.assertEqual(caught.exception.frame['capture_monotonic'],6.)
        self.assertEqual(self.grasp._wave_options.call_count,3)
        self.assertEqual([e['refresh_count'] for e in self.events if e['type']=='wave_scene_refresh'],[1,2])
        self.assertEqual(self.robot.writes,[]);self.assertEqual(self.robot.calls,[])

    def test_late_refresh_after_completed_lift_keeps_origin_steps_and_progress(self):
        initial=self.robot.read();attempts=0
        def delayed_change(*args,**kwargs):
            nonlocal attempts
            attempts+=1
            if attempts==2:
                self.capture_hook=lambda f,i:f.update(path=str(self.changed))
                frame=self.capture()
                raise PreMotionSceneChangedError('late synthetic scene change',frame)
            return self.fake_lift(*args,**kwargs)
        self.grasp.move=Mock(side_effect=delayed_change)
        self.grasp._wave_options.side_effect=[self.options(lift=True),self.options(lift=True),
                                             self.options(lift=True),self.options(wave=True)]
        result=self.run_wave()
        self.assertTrue(result['wave_completed']);self.assertEqual(len(result['raise_steps']),2)
        calls=self.grasp._wave_options.call_args_list
        self.assertEqual([c.kwargs['step_count'] for c in calls],[0,1,1,2])
        self.assertGreater(calls[1].kwargs['lifted_mm'],2)
        self.assertEqual(calls[1].kwargs['lifted_mm'],calls[2].kwargs['lifted_mm'])
        self.assertEqual(calls[2].args[0][0]['joints'],initial)
        for call in self.grasp.move.call_args_list:self.assertEqual(call.kwargs['lift_origin'],initial)
        refresh=[e for e in self.events if e['type']=='wave_scene_refresh']
        self.assertEqual(len(refresh),1);self.assertEqual(refresh[0]['step_count'],1)

    def test_real_postmotion_scene_error_is_not_reclassified_or_replayed(self):
        self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=self.options(lift=True)
        def postmotion_change(frame,index):
            if self.robot.writes:raise WristSceneChangedError('post-motion scene failure')
        self.capture_hook=postmotion_change
        with patch.object(self.robot,'hold',Mock(),create=True) as hold:
            with self.assertRaisesRegex(WristSceneChangedError,'post-motion') as caught:self.run_wave()
            self.assertNotIsInstance(caught.exception,PreMotionSceneChangedError)
            hold.assert_called_once()
        self.assertEqual(len(self.robot.calls),1)
        self.grasp._wave_options.assert_called_once()
        self.assertFalse(any(e['type']=='wave_scene_refresh' for e in self.events))

    def test_late_refresh_never_bypasses_cancel_stop_or_camera_failure(self):
        for condition in ('cancel','stop','camera'):
            with self.subTest(condition=condition):
                self.frames=0;self.events.clear();self.cancel.flag=False;self.robot.stop.flag=False
                self.robot.camera.fresh=lambda:condition!='camera' or self.frames<3
                self.grasp._wave_options.reset_mock();self.grasp._wave_options.side_effect=None
                self.grasp._wave_options.return_value=self.options(lift=True)
                def change(frame,index):
                    if index==3:
                        frame['path']=str(self.changed)
                        if condition=='cancel':self.cancel.set()
                        if condition=='stop':self.robot.stop.set()
                self.capture_hook=change
                with self.assertRaises((SessionCancelled,RuntimeError)):self.run_wave()
                self.grasp._wave_options.assert_called_once()
                self.assertEqual(self.robot.writes,[])
                self.assertFalse(any(e['type']=='wave_scene_refresh' for e in self.events))

    def test_real_move_non_image_precheck_failure_does_not_get_refresh_type(self):
        changes={'generation':lambda f:f.update(camera_generation='different'),
                 'joint_drift':lambda f:f['joints'].update(elbow_flex=f['joints']['elbow_flex']+4),
                 'timestamp':lambda f:f.update(capture_monotonic=1.),
                 'source':lambda f:f.update(source='observer')}
        for kind,change in changes.items():
            for changed_capture in (2,3):
                with self.subTest(kind=kind,changed_capture=changed_capture):
                    self.frames=0;self.capture_hook=None;reference=self.capture()
                    self.capture_hook=lambda f,i:change(f) if i==changed_capture else None
                    with self.assertRaises(ValueError) as caught:
                        self.grasp.move('up',3,reference,self.reports.append,free_space_lift=True)
                    self.assertNotIsInstance(caught.exception,PreMotionSceneChangedError)
                    self.assertEqual(self.robot.writes,[]);self.assertEqual(self.robot.calls,[])
    def test_scene_refresh_after_raises_preserves_original_origin_and_height(self):
        initial=self.robot.read()
        self.grasp._wave_options.side_effect=[self.options(lift=True),self.options(lift=True),
                                             self.options(lift=True),self.options(wave=True),self.options(wave=True)]
        self.capture_hook=lambda f,i:f.update(path=str(self.changed)) if 5<=i<=8 else None
        self.grasp.move=Mock(side_effect=self.fake_lift)
        result=self.run_wave()
        self.assertTrue(result['wave_completed'])
        self.assertEqual(len(result['raise_steps']),2)
        for call in self.grasp.move.call_args_list:self.assertEqual(call.kwargs['lift_origin'],initial)
        calls=self.grasp._wave_options.call_args_list
        self.assertEqual([c.kwargs['step_count'] for c in calls],[0,1,1,2,2])
        self.assertGreater(calls[1].kwargs['lifted_mm'],2)
        self.assertEqual(calls[1].kwargs['lifted_mm'],calls[2].kwargs['lifted_mm'])
        self.assertGreater(calls[3].kwargs['lifted_mm'],calls[2].kwargs['lifted_mm'])
        self.assertEqual(calls[3].kwargs['lifted_mm'],calls[4].kwargs['lifted_mm'])
        self.assertAlmostEqual(calls[4].kwargs['lifted_mm'],result['lifted_model_mm'])
        self.assertEqual([e['step_count'] for e in self.events if e['type']=='wave_scene_refresh'],[1,2])
    def test_non_image_validation_errors_never_refresh(self):
        changes={'generation':lambda f:f.update(camera_generation='different'),
                 'joint_drift':lambda f:f['joints'].update(elbow_flex=f['joints']['elbow_flex']+4),
                 'timestamp':lambda f:f.update(capture_monotonic=1.),
                 'source':lambda f:f.update(source='observer')}
        for kind,change in changes.items():
            with self.subTest(kind=kind):
                self.frames=0;self.events.clear();self.grasp._wave_options.reset_mock()
                self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=self.options(lift=True)
                self.capture_hook=lambda f,i:change(f) if i==2 else None
                self.grasp.move=Mock()
                with self.assertRaises(ValueError) as caught:self.run_wave()
                self.assertNotIsInstance(caught.exception,WristSceneChangedError)
                self.grasp._wave_options.assert_called_once();self.grasp.move.assert_not_called()
                self.assertEqual(self.robot.writes,[])
                self.assertFalse(any(e['type']=='wave_scene_refresh' for e in self.events))
    def test_cancel_after_lift_does_not_roll_or_plan_again(self):
        self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=self.options(lift=True)
        def lift(*args,**kwargs):
            result=self.fake_lift(*args,**kwargs);self.cancel.set();return result
        self.grasp.move=Mock(side_effect=lift)
        with self.assertRaises(SessionCancelled):self.run_wave()
        self.grasp._wave_options.assert_called_once();self.assertEqual(self.robot.writes,[])
    def test_high_pose_keeps_independently_approved_lifts_after_three_steps(self):
        self.robot.actual.update(dict(shoulder_pan=2162,shoulder_lift=1141,elbow_flex=2694,
                                     wrist_flex=2429,wrist_roll=984,gripper=1857))
        self.robot.goals={**self.robot.actual,'elbow_flex':2653}
        self.robot.bias={n:self.robot.goals[n]-self.robot.actual[n] for n in self.robot.actual}
        initial=self.robot.read()
        self.assertGreater(self.model.tip(initial)[2,3],.18)
        self.grasp._wave_options.side_effect=[*[self.options(lift=True) for _ in range(4)],
                                             self.options(wave=True)]
        self.grasp.move=Mock(wraps=self.grasp.move)
        self.grasp._prepare_retreat=Mock(side_effect=AssertionError('approved lift discarded'))
        result=self.run_wave()
        self.assertTrue(result['wave_completed'])
        self.assertEqual(self.grasp.move.call_count,4)
        self.assertEqual(len(result['raise_steps']),4)
        self.grasp._prepare_retreat.assert_not_called()
        calls=self.grasp._wave_options.call_args_list
        for index,call in enumerate(calls):
            self.assertEqual(call.kwargs['step_count'],index)
            # Four completed raises still need separate approved proposals;
            # the final wave decision may have exhausted the next-lift budget.
            if index<4:
                self.assertIsNotNone(call.kwargs['lift_proposal'])
                self.assertIn(call.kwargs['lift_proposal']['motion_kind'],('elbow_raise','elbow_raise_macro'))
                self.assertTrue(call.kwargs['lift_proposal']['clearance']['model_table_pass'])
            if index:
                self.assertGreater(call.args[0][-1]['capture_monotonic'],
                                   calls[index-1].args[0][-1]['capture_monotonic'])
        self.assertGreater(result['lifted_model_mm'],10)
        self.assertLess(result['lifted_model_mm'],60)
        self.assertEqual({n for n,_ in self.robot.writes},{'elbow_flex','wrist_roll'})
        self.assertEqual(self.robot.actual['gripper'],initial['gripper'])

    def test_twenty_step_budget_still_falls_back_to_an_independently_assessed_retreat(self):
        scene={'changed':False};triggered=set()
        def options(frames,**kwargs):
            step=kwargs['step_count']
            if step in (1,19) and step not in triggered:
                triggered.add(step);scene['changed']=not scene['changed']
            return self.options(lift=True)
        self.grasp._wave_options.side_effect=options
        self.capture_hook=lambda f,i:f.update(path=str(self.changed)) if scene['changed'] else None
        # Keep the real CAD origin fixed in this sequencing fixture; only the
        # attempt budget is exercised. Real move validates measured progress.
        self.grasp.move=Mock(side_effect=lambda *a,**kw:{'before':self.capture(),'after':self.capture(),
                            'actual_delta_model_mm':[0,0,1.]})
        self.grasp._prepare_retreat=Mock(side_effect=RuntimeError('new side phase'))
        with self.assertRaisesRegex(RuntimeError,'new side phase'):self.run_wave()
        self.assertEqual(self.grasp.move.call_count,20)
        self.assertEqual(self.grasp._wave_options.call_count,23)
        self.assertEqual([e['step_count'] for e in self.events if e['type']=='wave_scene_refresh'],[1,19])
        self.assertIsNone(self.grasp._wave_options.call_args.kwargs['lift_proposal'])
        self.grasp._prepare_retreat.assert_called_once()
        self.assertEqual(self.robot.writes,[])


if __name__=='__main__':unittest.main()
