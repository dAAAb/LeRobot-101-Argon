"""Wrist rotation geometry tests; no motor, camera or object-size inputs."""
import unittest
import numpy as np
from server.local_cartesian import ALL,NAMES,default_model

class WristRotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=default_model();cls.raw=dict(zip(ALL,[2216,1163,2952,2300,987,1857]))
        cls.held={**cls.raw,'shoulder_pan':2221,'shoulder_lift':1156,'elbow_flex':2946,'wrist_flex':2302,'wrist_roll':994}
        cls.envelope={n:max(12,abs(cls.held[n]-cls.raw[n])+6) for n in NAMES}
    def propose(self,axis='roll',angle=3,**kwargs):
        return self.model.propose_rotation(self.raw,axis,angle,held_goals=self.held,uncertainty_ticks=self.envelope,**kwargs)
    def test_roll_changes_one_joint_and_never_emits_gripper_command(self):
        p=self.propose()
        self.assertEqual(set(p['physical_goal_targets']),{'wrist_roll'})
        self.assertEqual(p['joint_targets']['gripper'],self.raw['gripper'])
        self.assertEqual(p['physical_goal_targets']['wrist_roll'],self.held['wrist_roll']+30)
        for n in NAMES:
            if n!='wrist_roll':self.assertEqual(p['joint_targets'][n],self.raw[n])
        self.assertFalse(p['gripper_command_included'])
    def test_three_degrees_keeps_30tick_trust_cap_and_reports_true_angle(self):
        p=self.propose();self.assertTrue(p['limited_by_30tick_trust_region'])
        self.assertAlmostEqual(p['accepted_angle_deg'],30*360/4096)
        self.assertLessEqual(p['full_orientation_change_deg'],3)
        self.assertLess(p['predicted_fixed_pad_travel_mm'],.5)
        self.assertGreater(p['predicted_camera_baseline_mm'],1.)
    def test_pitch_reports_endpoint_sweep_not_a_cartesian_5mm_step(self):
        p=self.propose('pitch',-3)
        self.assertEqual(set(p['physical_goal_targets']),{'wrist_flex'})
        self.assertGreater(p['predicted_fixed_pad_travel_mm'],5)
        self.assertGreater(p['predicted_fixed_pad_delta_mm'][2],0)
    def test_read_only_full_plus_minus_20degree_wave_model_screen(self):
        # Absolute waypoints avoid accumulating rounding or creating diagonal
        # return paths. Each segment is only an offline geometry proposal.
        raw=dict(self.raw);hold=dict(self.held);origin=self.raw['wrist_roll'];minimum=1e9
        for endpoint in (origin+227,origin,origin-227,origin):
            remaining=abs(endpoint-raw['wrist_roll'])
            steps=max(1,int(np.ceil(remaining/30)))
            positions=np.rint(np.linspace(raw['wrist_roll'],endpoint,steps+1)).astype(int)[1:]
            for value in positions:
                p=self.model.propose_rotation(raw,'roll',(int(value)-raw['wrist_roll'])*360/4096,
                    held_goals=hold,uncertainty_ticks=self.envelope)
                self.assertEqual(p['joint_targets']['wrist_roll'],value)
                self.assertLessEqual(abs(p['delta_ticks']['wrist_roll']),30)
                self.assertTrue(p['clearance']['model_table_pass'])
                self.assertTrue(p['requires_new_observation_after_step'])
                minimum=min(minimum,p['clearance']['continuous_model_lower_bound_mm'][self.model.jaw_label])
                raw=p['joint_targets'];hold.update(p['physical_goal_targets'])
        self.assertEqual(raw,self.raw);self.assertEqual(hold,self.held)
        self.assertGreater(minimum,40)
    def test_measured_tube_is_contained_by_screened_bounds(self):
        p=self.propose()
        for n in NAMES:
            self.assertGreaterEqual(p['measured_bounds'][n][0],p['physical_goal_bounds'][n][0])
            self.assertLessEqual(p['measured_bounds'][n][1],p['physical_goal_bounds'][n][1])
        with self.assertRaisesRegex(ValueError,'Actual motion tube'):
            self.model.propose_rotation(self.raw,'roll',2,uncertainty_ticks=3)
    def test_table_or_model_joint_violation_rejects(self):
        with self.assertRaisesRegex(ValueError,'table guard'):self.propose(min_clearance_m=.03)
        bad={**self.raw,'wrist_roll':270}
        with self.assertRaisesRegex(ValueError,'URDF|模型關節範圍'):
            self.model.propose_rotation(bad,'roll',-3,uncertainty_ticks=12)
    def test_hold_offset_outside_screened_box_rejects(self):
        bad={**self.held,'wrist_roll':self.raw['wrist_roll']+20}
        with self.assertRaisesRegex(ValueError,'hold offset'):
            self.model.propose_rotation(self.raw,'roll',2,held_goals=bad,uncertainty_ticks=self.envelope)
    def test_invalid_axis_angle_or_zero_tick_request_rejects(self):
        for axis,angle in [('yaw',2),('roll',0),('roll',3.1),('pitch',float('nan')),('roll',True),('roll',.01)]:
            with self.subTest(axis=axis,angle=angle),self.assertRaises(ValueError):self.propose(axis,angle)

if __name__=='__main__':unittest.main()
