"""Actual-mesh pure IK regressions; no motors, cameras or object dimensions."""
import unittest
from unittest.mock import patch
import numpy as np
from server.local_cartesian import ALL,NAMES,default_model

class AdaptiveProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=default_model()
        cls.raw=dict(zip(ALL,[2155,1182,2946,2268,977,1857]))
        cls.forward=-cls.model.fk.transforms(cls.raw)['gripper_link'][:3,2]
        cls.held={**cls.raw,'shoulder_pan':2151,'shoulder_lift':1183,'elbow_flex':2944,'wrist_flex':2259,'wrist_roll':973}
        cls.envelope={n:max(12,abs(cls.held[n]-cls.raw[n])+6) for n in NAMES}
    def propose(self,mm,**kwargs):
        return self.model.propose_adaptive(self.raw,self.forward*mm/1000,
            held_goals=self.held,uncertainty_ticks=self.envelope,**kwargs)
    def test_forward_5mm_saturates_but_shorter_proposal_is_feasible(self):
        full=self.model.local_ik(self.raw,self.forward*.005)
        self.assertGreater(full['predicted_position_error_mm'],1)
        p=self.propose(5)
        self.assertTrue(p['adaptive_shortened']);self.assertLess(p['accepted_requested_distance_mm'],5)
        self.assertLessEqual(p['predicted_position_error_mm'],1)
        self.assertLessEqual(max(abs(v) for v in p['delta_ticks'].values()),30)
        self.assertTrue(p['clearance']['model_table_pass'])
        self.assertTrue(p['requires_new_observation_after_step'])
        self.assertTrue(np.allclose(np.cross(p['requested_delta_m'],self.forward),0,atol=1e-12))
    def test_feasible_original_2mm_is_not_shortened(self):
        p=self.propose(2)
        self.assertFalse(p['adaptive_shortened']);self.assertEqual(len(p['proposal_attempts']),1)
        self.assertAlmostEqual(p['accepted_requested_distance_mm'],2)
    def test_physical_hold_compensation_and_gripper_are_preserved(self):
        p=self.propose(5)
        for n in NAMES:self.assertEqual(p['physical_goal_targets'][n],self.held[n]+p['delta_ticks'][n])
        self.assertEqual(p['joint_targets']['gripper'],self.raw['gripper'])
        self.assertNotIn('gripper',p['physical_goal_targets'])
    def test_invalid_origin_is_not_retried_with_smaller_goal(self):
        bad={**self.raw,'elbow_flex':3044}
        with patch.object(self.model,'propose') as propose:
            with self.assertRaisesRegex(ValueError,'Onboard'):
                self.model.propose_adaptive(bad,self.forward*.005)
            propose.assert_not_called()
    def test_hold_outside_screened_axis_is_not_hidden_by_shortening(self):
        with patch.object(self.model,'propose') as propose:
            with self.assertRaisesRegex(ValueError,'hold offset'):
                self.model.propose_adaptive(self.raw,self.forward*.005,
                    held_goals={**self.held,'wrist_roll':self.raw['wrist_roll']+20},uncertainty_ticks=self.envelope)
            propose.assert_not_called()
    def test_existing_table_violation_cannot_be_repaired_by_shortening(self):
        with patch.object(self.model,'propose') as propose:
            with self.assertRaisesRegex(ValueError,'Current pose'):
                self.propose(3,min_clearance_m=.03)
            propose.assert_not_called()
    def test_unachievable_precision_fails_with_attempt_evidence(self):
        with self.assertRaisesRegex(ValueError,'No feasible same-direction'):
            self.propose(5,max_position_error_mm=1e-8)
    def test_sub_mm_request_is_never_enlarged(self):
        p=self.propose(.84)
        self.assertLessEqual(p['accepted_requested_distance_mm'],.84+1e-10)
        self.assertAlmostEqual(p['original_requested_distance_mm'],.84)
    def test_invalid_inputs_do_not_relax_limits(self):
        for kwargs in ({'max_position_error_mm':1.1},{'max_orientation_error_deg':3},
                       {'shrink_factor':1.},{'min_step_mm':0},{'min_step_mm':1e-300},{'shrink_factor':float('nan')}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.propose(5,**kwargs)
        for delta in ([0,0,0],[0,0,.006],[float('nan'),0,0]):
            with self.subTest(delta=delta),self.assertRaises(ValueError):self.model.propose_adaptive(self.raw,delta)

if __name__=='__main__':unittest.main()
