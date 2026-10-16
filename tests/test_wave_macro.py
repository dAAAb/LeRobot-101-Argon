"""A larger, separately assessed elbow arc; no serial/camera/network I/O."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

import test_wave as helpers
import test_wave_tolerance as controller_helpers
from server.gripper_power import SessionCancelled
from server.local_cartesian import ALL, default_model
from server.wave_arc import (ACTIVE, MACRO_LOCAL_CORRIDOR,
    LOCAL_CORRIDOR, arc_endpoint, check_arc_corridor, propose_raise)
from server.wave_planner import WaveOptions, _lift_context, assess_wave_options


class MacroGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model=default_model()

    def setUp(self):
        self.raw=dict(zip(ALL,[2161,1138,2654,2332,984,1858]))
        self.held={**self.raw,ACTIVE:2647}

    def propose(self,**kwargs):
        return propose_raise(self.model,self.raw,self.held,self.raw,
                             max_distance_mm=20.,motion_kind='elbow_raise_macro',**kwargs)

    def test_macro_is_one_screened_sampled_arc_larger_than_small_profile(self):
        p,envelope=self.propose()
        self.assertEqual(p['motion_kind'],'elbow_raise_macro')
        self.assertEqual(p['delta_ticks'][ACTIVE],-45)
        self.assertGreater(p['predicted_delta_m'][2]*1000,18.)
        self.assertLessEqual(np.linalg.norm(p['predicted_delta_m'])*1000,20.)
        self.assertEqual(set(p['active_joint_targets']),{ACTIVE})
        self.assertEqual(envelope[ACTIVE],30)
        path=p['arc_path']
        self.assertEqual(path['actual_tip_corridor_mm'],MACRO_LOCAL_CORRIDOR)
        self.assertEqual(path['sampled_elbow_offsets_ticks'],list(range(0,-46,-1)))
        for offset,delta in zip(path['sampled_elbow_offsets_ticks'],path['sampled_delta_m']):
            expected=self.model.tip({**self.raw,ACTIVE:self.raw[ACTIVE]+offset})[:3,3]-self.model.tip(self.raw)[:3,3]
            np.testing.assert_allclose(delta,expected,atol=1e-15)
        self.assertTrue(p['clearance']['model_table_pass'])
        check_arc_corridor(self.model,self.raw,p['joint_targets'],motion_kind='elbow_raise_macro')
        self.assertTrue(arc_endpoint(self.model,self.raw,p['joint_targets'],p))
        with self.assertRaises(RuntimeError):
            check_arc_corridor(self.model,self.raw,p['joint_targets'])

    def test_macro_reserves_whole_lateral_and_up_corridor_even_for_small_request(self):
        # Synthetic pose isolates budget reservation before table screening.
        for delta,angle in (([7.01,0,0],0),([0,0,34.01],0),([0,0,0],9.01)):
            origin=dict(self.raw);raw={**origin,ACTIVE:origin[ACTIVE]-20}
            shifted=np.eye(4);shifted[:3,3]=np.array(delta)/1000
            shifted[:3,:3]=Rotation.from_euler('y',angle,degrees=True).as_matrix()
            model=Mock();model.tip.side_effect=lambda pose:np.eye(4) if pose==origin else shifted
            for cap in (20.,3.):
                with self.subTest(delta=delta,angle=angle,cap=cap),self.assertRaises(RuntimeError):
                    propose_raise(model,raw,raw,origin,max_distance_mm=cap,motion_kind='elbow_raise_macro')
                model.screen_segment.assert_not_called()

    def test_unknown_profile_and_incorrect_sampled_contract_cannot_expand_bounds(self):
        p,_=self.propose()
        for change in ('small_kind','small_corridor','missing_sample','oversized_corridor','wrong_endpoint'):
            bad=deepcopy(p)
            if change=='small_kind':bad['motion_kind']='elbow_raise'
            if change=='small_corridor':bad['arc_path']['actual_tip_corridor_mm']=dict(LOCAL_CORRIDOR)
            if change=='missing_sample':bad['arc_path']['sampled_elbow_offsets_ticks'].pop(1)
            if change=='oversized_corridor':bad['arc_path']['actual_tip_corridor_mm']['maximum_up']=27.
            if change=='wrong_endpoint':bad['arc_path']['sampled_delta_m'][-1][2]+=.001
            with self.subTest(change=change),self.assertRaises(ValueError):_lift_context(bad)
        with self.assertRaises(ValueError):
            propose_raise(self.model,self.raw,self.held,self.raw,motion_kind='anything')

    def test_one_inference_contains_entire_macro_and_exact_corridor(self):
        p,_=self.propose()
        response={'wave':{'gripper_empty':True,'human_clear':True,'clear_sweep':False,'confidence':.8,'reason':'前方有物件'},
                  'lift':{'gripper_empty':True,'human_clear':True,'clear_path':True,'confidence':.8,'reason':'整段抬手路徑可見'}}
        brain=SimpleNamespace(_infer=Mock(return_value=WaveOptions.model_validate(response)))
        frames=[{'source':'wrist','camera_generation':'same','path':'synthetic.jpg'}]
        geometry={'full_mesh_table_guard':True,'max_roll_degrees':10.,'tracking_margin_degrees':1.05}
        result=assess_wave_options(brain,frames,roll_geometry=geometry,lift_proposal=p,
                                  lifted_mm=0.,step_count=0)
        self.assertEqual(result,response);brain._infer.assert_called_once()
        prompt=brain._infer.call_args.args[0]
        data=json.loads(prompt.split('\nINPUT_DATA_JSON\n')[1])
        self.assertEqual(data['lift_proposal']['arc_path'],p['arc_path'])
        self.assertEqual(data['lift_proposal']['motion_kind'],'elbow_raise_macro')
        for text in ('at-most-50 ticks','at most 26 mm upward','8 mm lateral displacement','6 degrees actual local rotation','entirety'):
            self.assertIn(text,prompt)
        brain._infer.reset_mock()
        with self.assertRaises(ValueError):
            assess_wave_options(brain,frames,roll_geometry=geometry,lift_proposal=p,
                                lifted_mm=34.01,step_count=0)
        brain._infer.assert_not_called()


class MacroMoveTests(unittest.TestCase):
    model=helpers.default_model()
    setUp=helpers.WaveTests.setUp
    capture=helpers.WaveTests.capture

    def approved(self):
        frame=self.capture()
        proposal,envelope=self.grasp._wave_raise_proposal(frame['joints'],frame['joints'])
        self.assertEqual(proposal['motion_kind'],'elbow_raise_macro')
        return frame,{'proposal':proposal,'envelope':envelope,'origin':dict(frame['joints']),
                      'hold_goals':dict(self.robot.goals)}

    def execute(self,frame,approved,**kwargs):
        return self.grasp.move('up',float(np.linalg.norm(approved['proposal']['requested_delta_m'])*1000),
            frame,self.reports.append,free_space_lift=True,lift_origin=frame['joints'],
            cancel=self.cancel,assessed_lift=approved,**kwargs)

    def test_same_assessed_proposal_executes_once_with_live_guards_and_fresh_endpoint(self):
        frame,approved=self.approved();p=approved['proposal']
        with patch.object(self.grasp,'_wave_raise_proposal',side_effect=AssertionError('must not replan approved path')):
            result=self.execute(frame,approved)
        self.assertEqual(len(self.robot.calls),1)
        call=self.robot.calls[0];opts=call['kwargs']
        self.assertEqual(call['targets'],p['active_joint_targets'])
        self.assertEqual(call['physical'],{ACTIVE:p['physical_goal_targets'][ACTIVE]})
        self.assertEqual({n for n,_ in self.robot.writes},{ACTIVE})
        self.assertEqual(opts['speed'],40);self.assertTrue(opts['feedback'])
        self.assertFalse(opts['allow_rearm']);self.assertEqual(opts['endpoint_stability_ticks'],2)
        self.assertEqual(result['motion_kind'],'elbow_raise_macro')
        self.assertGreater(result['actual_delta_model_mm'][2],8.)
        self.assertGreater(result['after']['capture_monotonic'],frame['capture_monotonic'])
        opts['measured_guard'](self.robot.read())
        self.assertTrue(opts['endpoint_accept'](self.robot.read()))
        with self.assertRaises(RuntimeError):
            opts['measured_guard']({**self.robot.read(),'gripper':frame['joints']['gripper']+5})
        with self.assertRaises(RuntimeError):
            opts['measured_guard']({**self.robot.read(),ACTIVE:frame['joints'][ACTIVE]-100})
        with patch.object(self.model,'floor_minimum',return_value={'base_link':0.,'gripper':.004}):
            with self.assertRaises(RuntimeError):opts['guard'](self.robot.read())
        self.cancel.set()
        with self.assertRaises(SessionCancelled):opts['guard'](self.robot.read())
        self.assertNotIn('wave_completed',result);self.assertNotIn('grasp_confirmed',result)

    def test_small_approval_cannot_be_used_to_execute_macro_or_larger_distance(self):
        frame=self.capture();p,e=self.grasp._wave_lift_proposal(8.,frame['joints'],frame['joints'])
        approved={'proposal':p,'envelope':e,'origin':frame['joints'],'hold_goals':dict(self.robot.goals)}
        with self.assertRaises(ValueError):
            self.grasp.move('up',18.,frame,self.reports.append,free_space_lift=True,
                assessed_lift=approved,lift_origin=frame['joints'])
        p['motion_kind']='elbow_raise_macro'
        with self.assertRaises(ValueError):self.execute(frame,approved)
        self.assertEqual(self.robot.calls,[]);self.assertEqual(self.robot.writes,[])

    def test_geometry_unavailable_uses_old_profile_before_visual_decision(self):
        from server.wave_arc import propose_raise as real_propose
        kinds=[]
        def propose(*args,**kwargs):
            kind=kwargs.get('motion_kind','elbow_raise');kinds.append(kind)
            if kind=='elbow_raise_macro':raise RuntimeError('macro does not fit remaining corridor')
            return real_propose(*args,**kwargs)
        with patch('server.wave_arc.propose_raise',side_effect=propose):
            p,_=self.grasp._wave_raise_proposal(self.robot.read(),self.robot.read())
        self.assertEqual(kinds,['elbow_raise_macro','elbow_raise'])
        self.assertEqual(p['motion_kind'],'elbow_raise')
        self.assertLessEqual(np.linalg.norm(p['requested_delta_m'])*1000,8.)
        self.grasp._wave_options.assert_not_called();self.assertEqual(self.robot.writes,[])

    def test_blocked_macro_is_not_retried_as_smaller_ai_approval(self):
        blocked={'wave':{**self.clearance,'clear_sweep':False},
                 'lift':{'gripper_empty':True,'human_clear':True,'clear_path':False,'confidence':.9,'reason':'上方有障礙'}}
        self.grasp._wave_options.side_effect=None;self.grasp._wave_options.return_value=blocked
        with patch.object(self.grasp,'_wave_lift_proposal',side_effect=AssertionError('cannot fall back after refusal')):
            with self.assertRaises(RuntimeError):self.grasp.wave(self.reports.append,self.cancel)
        self.grasp._wave_options.assert_called_once();self.assertEqual(self.robot.writes,[])


class MacroControllerTests(unittest.TestCase):
    """Real 25 Hz Robot.glide execution, synthetic register bus and clock."""
    model=helpers.default_model()
    capture=helpers.WaveTests.capture

    def setUp(self):
        controller_helpers.WaveEndpointToleranceTests.setUp(self)
        for name,entry in self.robot.calibration.items():
            servo=entry['id']
            self.bus.offset[servo]=self.initial[name]-self.initial_holds[name]
            self.bus.reg[servo].update({60:0,69:0})

    def execute(self):
        frame=self.capture();p,e=self.grasp._wave_raise_proposal(frame['joints'],frame['joints'])
        self.assertEqual(p['motion_kind'],'elbow_raise_macro')
        return self.grasp.move('up',float(np.linalg.norm(p['requested_delta_m'])*1000),
            frame,self.reports.append,free_space_lift=True,lift_origin=frame['joints'],cancel=self.cancel,
            assessed_lift={'proposal':p,'envelope':e,'origin':frame['joints'],'hold_goals':dict(self.robot.goals)})

    def test_one_macro_interpolates_many_elbow_writes_without_inference(self):
        result=self.execute()
        self.assertEqual(len(self.calls),1)
        self.assertGreater(len(self.bus.events),20)
        self.assertEqual({(servo,address) for servo,address,_ in self.bus.events},{(3,42)})
        self.assertGreater(result['actual_delta_model_mm'][2],18.)
        self.assertEqual(result['completion'],'exact_target')
        self.grasp._wave_options.assert_not_called()
        self.assertFalse(self.robot.stop.is_set())

    def test_camera_loss_during_macro_stops_remaining_interpolation(self):
        write=self.robot._write;writes=[]
        def disconnect(name,address,value):
            result=write(name,address,value);writes.append((name,value,self.robot.stop.is_set()))
            if len(writes)==4:self.robot.camera.fresh=lambda:False
            return result
        self.robot._write=disconnect
        with self.assertRaisesRegex(RuntimeError,'畫面中斷'):self.execute()
        self.assertEqual(sum(not stopped for _,_,stopped in writes),4)
        self.assertEqual({name for name,_,stopped in writes if stopped},set(ALL))
        self.assertTrue(self.robot.stop.is_set())
        self.assertFalse(any(e['type']=='movement_result' for e in self.events))

    def test_cancel_during_macro_stops_remaining_interpolation(self):
        write=self.robot._write;writes=[]
        def cancel(name,address,value):
            result=write(name,address,value);writes.append((name,value,self.robot.stop.is_set()))
            if len(writes)==4:self.cancel.set()
            return result
        self.robot._write=cancel
        with self.assertRaises(SessionCancelled):self.execute()
        self.assertEqual(sum(not stopped for _,_,stopped in writes),4)
        self.assertEqual({name for name,_,stopped in writes if stopped},set(ALL))
        self.assertTrue(self.robot.stop.is_set())
        self.assertFalse(any(e['type']=='movement_result' for e in self.events))


if __name__=='__main__':unittest.main()
