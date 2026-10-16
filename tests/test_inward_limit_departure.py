"""Directional near-stop departure through CAD and real driver/fake registers."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from server.local_cartesian import ALL, NAMES, default_model
from server.motion_program import prepare_joint_segment, execute_prepared_segment, _validate_contract, rebind_segment_jaw
from server.task_space_plan import solve_task_pose, _path, _tcp, TaskSpaceError


class InwardContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=default_model()
        cls.raw=dict(zip(ALL,[2180,1032,3042,2174,884,1861]))
        cls.held={**cls.raw,'elbow_flex':3037}
        cls.target={**cls.raw,'elbow_flex':3000}

    def prepare(self, **kwargs):
        return prepare_joint_segment(self.model,self.raw,self.held,self.target,
            inward_limit_joints=['elbow_flex'],**kwargs)

    def test_explicit_contract_retains_full_geometry_and_compensation(self):
        with self.assertRaises(ValueError):
            prepare_joint_segment(self.model,self.raw,self.held,self.target)
        segment=self.prepare();_validate_contract(segment)
        self.assertEqual(segment['geometry_bounds']['elbow_flex'],[2988,3054])
        self.assertEqual(segment['raw_bounds']['elbow_flex'],[2988,3045])
        self.assertEqual(segment['measured_bounds']['elbow_flex'],[2988,3045])
        self.assertEqual(segment['physical_goal_targets']['elbow_flex'],2995)
        self.assertEqual(segment['inward_limit_departures']['elbow_flex']['direction'],-1)
        self.assertEqual(segment['clearance']['actual_tracking_envelope_ticks']['elbow_flex'],12)
        self.assertTrue(segment['clearance']['model_table_pass'])
        rebound=rebind_segment_jaw(segment,1860,1860)
        self.assertEqual(rebound['inward_limit_departures'],segment['inward_limit_departures'])

    def test_outward_noop_short_departure_and_hard_invalid_starts_rejected(self):
        for elbow in (3043,3042,3035,3022):
            with self.subTest(elbow=elbow),self.assertRaises(ValueError):
                prepare_joint_segment(self.model,self.raw,self.held,{**self.target,'elbow_flex':elbow},
                                      inward_limit_joints=['elbow_flex'])
        for raw,held in (({**self.raw,'elbow_flex':3046},self.held),
                         (self.raw,{**self.held,'elbow_flex':3046}),
                         ({**self.raw,'elbow_flex':2980},{**self.held,'elbow_flex':2975})):
            with self.subTest(raw=raw,held=held),self.assertRaises(ValueError):
                prepare_joint_segment(self.model,raw,held,self.target,inward_limit_joints=['elbow_flex'])

    def test_full_tube_table_failure_still_rejects(self):
        original=self.model.screen_segment
        seen=[]
        def reject(*args,**kwargs):
            value=original(*args,**kwargs);seen.append(value)
            return {**value,'model_table_pass':False}
        with patch.object(self.model,'screen_segment',side_effect=reject),self.assertRaisesRegex(ValueError,'桌面'):
            self.prepare()
        self.assertEqual(seen[0]['actual_tracking_envelope_ticks']['elbow_flex'],12)
        self.assertEqual(seen[0]['inward_limit_departures']['elbow_flex']['direction'],-1)

    def test_lower_boundary_uses_opposite_direction_and_keeps_original_reserve(self):
        from test_motion_program import Model
        raw={n:2000 for n in ALL};raw['elbow_flex']=22
        held={**raw,'elbow_flex':25};target={**raw,'elbow_flex':44}
        segment=prepare_joint_segment(Model(),raw,held,target,inward_limit_joints=['elbow_flex'])
        self.assertEqual(segment['inward_limit_departures']['elbow_flex']['direction'],1)
        self.assertEqual(segment['physical_goal_targets']['elbow_flex'],47)
        self.assertEqual(segment['geometry_bounds']['elbow_flex'],[10,56])
        self.assertEqual(segment['raw_bounds']['elbow_flex'],[20,56])
        for elbow in (21,22,43):
            with self.subTest(elbow=elbow),self.assertRaises(ValueError):
                prepare_joint_segment(Model(),raw,held,{**target,'elbow_flex':elbow},
                                      inward_limit_joints=['elbow_flex'])

    def test_urdf_hard_limits_are_also_clipped_and_direction_is_required(self):
        changed=dict(self.model.joint_limits)
        from server.local_cartesian import RAD_PER_TICK
        changed['elbow_flex']=(changed['elbow_flex'][0],(3043-2047)*RAD_PER_TICK)
        with patch.object(self.model,'joint_limits',changed):
            segment=self.prepare()
            self.assertEqual(segment['raw_bounds']['elbow_flex'][1],3043)
            self.assertEqual(segment['geometry_bounds']['elbow_flex'][1],3054)
        with self.assertRaises(ValueError):
            self.model.screen_segment(self.raw,self.target,inward_limit_joints=['wrist_roll'])

    def test_ik_tries_inward_only_after_fixed_solution_fails(self):
        target={**self.raw,'elbow_flex':2960};T=self.model.tip(target)
        result=solve_task_pose(self.model,self.raw,target_base_m=T[:3,3],approach_axis_base=T[:3,1])
        self.assertEqual(result['joint_targets'],target)
        self.assertEqual(result['diagnostics']['inward_limit_joints'],['elbow_flex'])
        self.assertFalse(result['collision_screened'])
        unchanged=solve_task_pose(self.model,self.raw,target_base_m=self.model.tip(self.raw)[:3,3],
                                 approach_axis_base=self.model.tip(self.raw)[:3,1])
        self.assertEqual(unchanged['joint_targets'],self.raw)
        self.assertNotIn('inward_limit_joints',unchanged['diagnostics'])

    def test_numerical_substeps_coalesce_into_one_sealed_inward_segment(self):
        target={**self.raw,'elbow_flex':2960};T=self.model.tip(target)
        segments,goal,held,_=_path(self.model,self.raw,self.held,T[:3,3],T[:3,1],
            _tcp(None,self.raw),self.raw,tolerance=.002,axis_tolerance=3.,
            max_path_deviation_m=.003,phase='approach')
        self.assertEqual(goal,target);self.assertEqual(len(segments),1)
        self.assertEqual(held['elbow_flex'],2955)
        contract=segments[0]['contract'];_validate_contract(contract)
        self.assertEqual(contract['inward_limit_joints'],['elbow_flex'])
        self.assertTrue(contract['clearance']['model_table_pass'])

    def test_unreachable_and_program_range_are_not_relaxed(self):
        T=self.model.tip(self.raw)
        with self.assertRaises(TaskSpaceError) as failed:
            solve_task_pose(self.model,self.raw,target_base_m=[2.,2.,2.],approach_axis_base=T[:3,1])
        self.assertEqual(failed.exception.code,'unreachable_task_pose')
        self.assertIn('inward_fallback',failed.exception.diagnostics)
        with self.assertRaisesRegex(ValueError,'180'):
            prepare_joint_segment(self.model,self.raw,self.held,{**self.target,'elbow_flex':2800},
                                  inward_limit_joints=['elbow_flex'])


class InwardDriverTests(unittest.TestCase):
    # Reuse the actual Robot.glide fixture, not its test cases or motor mocks.
    from test_motion_program import ActualGlideProgramTests as Fixtures
    model=default_model()
    capture=Fixtures.capture

    def setUp(self):
        self.Fixtures.setUp(self)
        self.raw=dict(zip(ALL,[2180,1032,3042,2174,884,1861]))
        self.held={**self.raw,'elbow_flex':3037}
        for n,c in self.robot.calibration.items():
            servo=c['id'];c.update(self.model.calibration[n])
            self.bus.reg[servo].update({56:self.raw[n],42:self.held[n],60:0,69:0})
            self.bus.offset[servo]=self.raw[n]-self.held[n]
        self.robot.goals=dict(self.held)
        self.robot.bias={n:self.held[n]-self.raw[n] for n in ALL}
        self.segment=prepare_joint_segment(self.model,self.raw,self.held,{**self.raw,'elbow_flex':3000},
                                           inward_limit_joints=['elbow_flex'])

    def execute(self):
        return execute_prepared_segment(self.grasp,self.segment,self.capture(),self.cancel,self.reports.append)

    def test_real_driver_sends_only_monotonic_inward_goals_from_physical_hold(self):
        receipt=self.execute()
        goals=[value for servo,address,value in self.bus.events if address==42]
        self.assertGreater(len(goals),5)
        self.assertTrue(all(servo==3 and address==42 for servo,address,_ in self.bus.events))
        self.assertTrue(all(a>=b for a,b in zip([3037]+goals,goals)))
        self.assertGreater(goals[0],3033)  # Exercises the start outside 12-tick reserve.
        self.assertEqual(goals[-1],2995)
        self.assertEqual(receipt['after']['joints']['elbow_flex'],3000)
        self.assertFalse(self.robot.stop.is_set())
        self.assertFalse(self.calls[0]['kwargs'].get('return_origin',False))

    def test_changed_model_or_tampered_contract_rejected_before_goal(self):
        self.segment['inward_limit_departures']['elbow_flex']['direction']=1
        with self.assertRaisesRegex(ValueError,'計畫不同'):self.execute()
        self.assertEqual(self.bus.events,[])

    def test_outward_measured_drift_hits_hard_limit_and_stops(self):
        original=self.robot._write;attempt=[]
        def write(n,address,value,*args):
            result=original(n,address,value,*args)
            if address==42 and not self.robot.stop.is_set():
                attempt.append(value)
                self.bus.reg[3][56]=3046
            return result
        self.robot._write=write
        with self.assertRaisesRegex(ValueError,'raw interval'):self.execute()
        self.assertTrue(self.robot.stop.is_set())
        self.assertEqual(len(attempt),1)


if __name__=='__main__':unittest.main()
