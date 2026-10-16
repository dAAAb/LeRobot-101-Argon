import math
import unittest

import torch

from engine import inverse_rotate, recover_joints


class RecoveryTests(unittest.TestCase):
    def test_identity_rotation_and_root_translation_match_humanml_convention(self):
        features = torch.zeros(1, 4, 263)
        features[..., 3] = 1.0
        features[..., 1] = 0.1
        features[..., 2] = 0.2
        features[..., 4] = 0.05  # Left hip has positive X.
        features[..., 7] = -0.05  # Right hip has negative X.
        joints = recover_joints(features)
        self.assertEqual(tuple(joints.shape), (1, 4, 22, 3))
        self.assertTrue(torch.allclose(joints[0, :, 0, 0], torch.tensor([0, .1, .2, .3])))
        self.assertTrue(torch.allclose(joints[0, :, 0, 2], torch.tensor([0, .2, .4, .6])))
        self.assertTrue(torch.allclose(joints[0, :, 0, 1], torch.ones(4)))
        self.assertGreater(joints[0, 0, 1, 0], 0)
        self.assertLess(joints[0, 0, 2, 0], 0)

    def test_inverse_root_yaw_rotates_rightward_velocity_toward_positive_z(self):
        q = torch.tensor([[math.sqrt(.5), 0, math.sqrt(.5), 0]])
        v = torch.tensor([[1., 0, 0]])
        self.assertTrue(torch.allclose(inverse_rotate(q, v), torch.tensor([[0., 0, 1.]]), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
