"""Bounded reposition acceptance, independent of hardware and object size."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from server.wave_lift import lift_endpoint,lift_corridor,lift_attempt


class LiftEndpointTests(unittest.TestCase):
    def check(self,xyz,angle=0,requested=3):
        pose=np.eye(4);pose[:3,3]=np.array(xyz)/1000
        pose[:3,:3]=Rotation.from_euler('x',angle,degrees=True).as_matrix()
        return lift_endpoint(np.eye(4),pose,np.array([0,0,requested/1000]))
    def test_prior_partial_lift_is_progress_not_exact_target_success(self):
        result=self.check([-.222638,.039329,1.843778],.3)
        self.assertTrue(result['accepted']);self.assertAlmostEqual(result['progress_mm'],1.843778)
    def test_zero_backward_insufficient_lateral_overshoot_and_rotation_reject(self):
        for xyz,angle in [([0,0,0],0),([0,0,-1],0),([0,0,1.49],0),
                          ([1.01,0,2],0),([0,0,4.01],0),([0,0,2],1.01)]:
            with self.subTest(xyz=xyz,angle=angle):self.assertFalse(self.check(xyz,angle)['accepted'])
    def test_only_local_upward_requests_allowed(self):
        for delta in ([0,0,0],[0,0,-.001],[.001,0,.003],[0,0,.006],[0,0,float('nan')]):
            with self.subTest(delta=delta),self.assertRaises(ValueError):
                lift_endpoint(np.eye(4),np.eye(4),delta)
    def test_one_tick_shoulder_elbow_residual_keeps_existing_mm_tolerance(self):
        result=self.check([.1151,-.0203,4.3643],.1758,requested=3.75)
        self.assertTrue(result['accepted'])
    def test_verified_reposition_corridor_accepts_settled_extra_upward_progress(self):
        pose=np.eye(4);pose[:3,3]=[.00065,0,.0058865]
        pose[:3,:3]=Rotation.from_euler('x',.8789,degrees=True).as_matrix()
        self.assertFalse(lift_endpoint(np.eye(4),pose,[0,0,.0028125])['accepted'])
        self.assertTrue(lift_endpoint(np.eye(4),pose,[0,0,.0028125],verified_max_up_mm=12)['accepted'])
        pose[2,3]=.01201
        self.assertFalse(lift_endpoint(np.eye(4),pose,[0,0,.0028125],verified_max_up_mm=12)['accepted'])
        for limit in (True,13,float('nan'),1):
            with self.assertRaises(ValueError):lift_endpoint(np.eye(4),pose,[0,0,.003],verified_max_up_mm=limit)
    def test_stepwise_small_rotations_cannot_accumulate_past_attempt_orientation(self):
        attempt=np.eye(4)
        before=np.eye(4);before[2,3]=.004
        before[:3,:3]=Rotation.from_euler('x',.8,degrees=True).as_matrix()
        after=np.eye(4);after[2,3]=.008
        after[:3,:3]=Rotation.from_euler('x',1.6,degrees=True).as_matrix()
        # Each separate 0.8-degree change used to pass by resetting its origin.
        self.assertTrue(lift_endpoint(attempt,before,[0,0,.004])['accepted'])
        self.assertTrue(lift_endpoint(before,after,[0,0,.004])['accepted'])
        result=lift_endpoint(before,after,[0,0,.004],orientation_origin=attempt)
        self.assertFalse(result['accepted'])
        self.assertAlmostEqual(result['orientation_error_deg'],.8)
        self.assertAlmostEqual(result['reference_orientation_error_deg'],1.6)
    def test_correcting_toward_attempt_orientation_passes_without_relaxing_local_change(self):
        attempt=np.eye(4)
        before=np.eye(4);before[:3,:3]=Rotation.from_euler('x',.8,degrees=True).as_matrix()
        after=np.eye(4);after[2,3]=.004
        after[:3,:3]=Rotation.from_euler('x',.2,degrees=True).as_matrix()
        result=lift_endpoint(before,after,[0,0,.004],orientation_origin=attempt)
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['orientation_error_deg'],.6)
        self.assertAlmostEqual(result['reference_orientation_error_deg'],.2)
        before[:3,:3]=Rotation.from_euler('x',1.6,degrees=True).as_matrix()
        self.assertFalse(lift_endpoint(before,after,[0,0,.004],orientation_origin=attempt)['accepted'])
    def test_intermediate_corridor_does_not_replace_stricter_endpoint(self):
        pose=np.eye(4);pose[2,3]=.010
        lift_corridor(np.eye(4),pose)
        self.assertFalse(lift_endpoint(np.eye(4),pose,[0,0,.005])['accepted'])
        for xyz in ([0,0,.0121],[.0031,0,.005],[0,0,-.0011]):
            pose[:3,3]=xyz
            with self.assertRaises(RuntimeError):lift_corridor(np.eye(4),pose)


if __name__=='__main__':unittest.main()
