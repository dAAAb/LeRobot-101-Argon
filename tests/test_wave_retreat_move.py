"""Real side-phase dispatch against synthetic images and an in-memory plant."""
import unittest
from unittest.mock import Mock,patch

import test_wave as helpers
from server.gripper_power import SessionCancelled


class RetreatMoveTests(unittest.TestCase):
    model=helpers.default_model()
    setUp=helpers.WaveTests.setUp
    capture=helpers.WaveTests.capture

    def options(self,*,wave=False,left=False,right=False,backward=False):
        def side(clear):return {'gripper_empty':True,'human_clear':True,'clear_path':clear,
                               'confidence':.95,'reason':'synthetic side path'}
        return {'wave':{**self.clearance,'clear_sweep':wave},'left':side(left),'right':side(right),'backward':side(backward)}

    def prepare(self):
        start=self.capture()
        return self.grasp._prepare_retreat(self.reports.append,self.cancel,(113,0,-113,0),start,
                                          [{'before':start,'after':start}],10.)

    def test_side_phase_uses_pan_only_and_requires_new_pose_before_reconsidering_wave(self):
        original=self.robot.read();goals=dict(self.robot.goals)
        with patch('server.wave_planner.assess_retreat_options',side_effect=[
                self.options(wave=True,left=True),self.options(wave=True)]) as assess:
            result=self.prepare()
        self.assertEqual(assess.call_count,2)
        self.assertEqual(set(n for n,_ in self.robot.writes),{'shoulder_pan'})
        self.assertEqual(len(result['retreat_steps']),1)
        self.assertGreater(result['retreated_model_mm'],.5)
        self.assertEqual(result['lifted_model_mm'],0.,'replace the stale caller height with the checked transition pose')
        self.assertNotIn('wave_completed',result)
        for n in original:
            if n!='shoulder_pan':
                self.assertEqual(self.robot.actual[n],original[n])
                self.assertEqual(self.robot.goals[n],goals[n])
        calls=assess.call_args_list
        self.assertIsNone(calls[1].kwargs['retreat_proposals']['right'])
        self.assertGreater(calls[1].args[1][-1]['capture_monotonic'],calls[0].args[1][-1]['capture_monotonic'])
        options=self.robot.calls[0]['kwargs']
        self.assertEqual(options['deferred_joints'],())
        drift={**self.robot.read(),'elbow_flex':original['elbow_flex']+5}
        with self.assertRaises(RuntimeError):options['measured_guard'](drift)

    def test_neither_clear_side_means_no_motion_and_no_implicit_retry(self):
        with patch('server.wave_planner.assess_retreat_options',return_value=self.options()) as assess:
            with self.assertRaisesRegex(RuntimeError,'退讓路線'):self.prepare()
        assess.assert_called_once();self.assertEqual(self.robot.writes,[])

    def test_cancel_during_side_assessment_cannot_send_a_goal(self):
        def assessment(*args,**kwargs):
            self.cancel.set();return self.options(left=True)
        with patch('server.wave_planner.assess_retreat_options',side_effect=assessment):
            with self.assertRaises(SessionCancelled):self.prepare()
        self.assertEqual(self.robot.writes,[])

    def test_backward_path_uses_assessed_ik_and_locks_out_sideways_alternatives(self):
        self.robot.actual.update(dict(shoulder_pan=2162,shoulder_lift=1141,elbow_flex=2694,
                                     wrist_flex=2429,wrist_roll=984,gripper=1857))
        self.robot.goals={**self.robot.actual,'elbow_flex':2653,'shoulder_lift':1142,'wrist_flex':2430}
        self.robot.bias={n:self.robot.goals[n]-self.robot.actual[n] for n in self.robot.actual}
        original=self.robot.read()
        with patch('server.wave_planner.assess_retreat_options',side_effect=[
                self.options(backward=True),self.options(wave=True)]) as assess:
            result=self.prepare()
        self.assertEqual(len(result['retreat_steps']),1)
        self.assertEqual(result['retreat_steps'][0]['motion_kind'],'cartesian_backward')
        self.assertTrue(result['retreat_steps'][0]['exact_target_reached'])
        self.assertGreater(result['retreated_model_mm'],1.)
        self.assertIsNone(assess.call_args.kwargs['retreat_proposals']['left'])
        self.assertIsNone(assess.call_args.kwargs['retreat_proposals']['right'])
        for n in ('shoulder_pan','wrist_roll','gripper'):
            self.assertEqual(self.robot.actual[n],original[n])
            self.assertNotIn(n,{joint for joint,_ in self.robot.writes})

    def test_preferred_height_assesses_retreat_independently_of_blocked_wave_flags(self):
        from server.grasp import WAVE_PREFERRED_HEIGHT_M
        self.robot.actual.update(dict(shoulder_pan=2162,shoulder_lift=1141,elbow_flex=2694,
                                     wrist_flex=2429,wrist_roll=984,gripper=1857))
        self.robot.goals=dict(self.robot.actual);self.robot.bias=dict.fromkeys(self.robot.actual,0)
        self.assertGreater(self.model.tip(self.robot.actual)[2,3],WAVE_PREFERRED_HEIGHT_M)
        for wave_change in ({'human_clear':False},{'gripper_empty':False},
                            {'confidence':.79,'clear_sweep':True}):
            with self.subTest(wave_change=wave_change):
                wave={**self.clearance,'clear_sweep':False,**wave_change}
                self.grasp._wave_options=Mock(return_value={'wave':wave,'lift':{
                    'gripper_empty':False,'human_clear':False,'clear_path':False,
                    'confidence':0.,'reason':'no upward candidate'}})
                with patch('server.wave_planner.assess_retreat_options',return_value=self.options()) as assess:
                    with self.assertRaisesRegex(RuntimeError,'退讓路線'):
                        self.grasp._prepare_roll_sweep(self.reports.append,self.cancel,
                            (113,0,-113,0),allow_raise=True)
                    assess.assert_called_once()
                    self.assertIsNotNone(assess.call_args.kwargs['retreat_proposals']['backward'])
                self.grasp._wave_options.assert_called_once()
                self.assertEqual(self.robot.writes,[])

    def test_independent_retreat_assessment_cannot_bypass_candidate_empty_or_human_flags(self):
        for field in ('gripper_empty','human_clear','clear_path','confidence'):
            with self.subTest(field=field):
                response=self.options(left=True)
                response['left'][field]=.699 if field=='confidence' else False
                with patch('server.wave_planner.assess_retreat_options',return_value=response) as assess:
                    with self.assertRaisesRegex(RuntimeError,'退讓路線'):self.prepare()
                    assess.assert_called_once()
                self.assertEqual(self.robot.writes,[])

    def test_retreat_transition_cannot_reset_accumulated_raise_joint_drift(self):
        attempt_start=self.capture()
        # Three ticks were present at the last raise image; another two ticks
        # pass the local unchanged-image gate but breach the original <=4 bound.
        self.robot.actual['shoulder_pan']+=3
        raised=self.capture()
        self.robot.actual['shoulder_pan']+=2
        with patch.object(self.robot,'hold',create=True) as hold:
            with patch('server.wave_planner.assess_retreat_options') as assess:
                with self.assertRaisesRegex(RuntimeError,'其他關節'):
                    self.grasp._prepare_retreat(self.reports.append,self.cancel,(113,0,-113,0),
                        attempt_start,[{'before':attempt_start,'after':raised}],10.)
                assess.assert_not_called()
            hold.assert_called_once_with()
        self.assertTrue(self.robot.stop.is_set())
        self.assertEqual(self.robot.writes,[])
        self.assertEqual(self.robot.calls,[])
        self.assertEqual(self.events[-1]['type'],'retreat_transition_failed')

    def test_retreat_transition_rejects_changed_scene_before_new_origin_or_assessment(self):
        start=self.capture()
        self.capture_hook=lambda frame,index:frame.update(path=str(self.changed))
        with patch.object(self.robot,'hold',create=True) as hold:
            with patch('server.wave_planner.assess_retreat_options') as assess:
                with self.assertRaisesRegex(ValueError,'場景已改變'):
                    self.grasp._prepare_retreat(self.reports.append,self.cancel,(113,0,-113,0),
                        start,[{'before':start,'after':start}],10.)
                assess.assert_not_called()
            hold.assert_called_once_with()
        self.assertTrue(self.robot.stop.is_set())
        self.assertEqual(self.robot.writes,[])
        self.assertEqual(self.events[-1]['type'],'retreat_transition_failed')


if __name__=='__main__':unittest.main()
