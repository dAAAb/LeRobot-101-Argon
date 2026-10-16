"""Small deterministic geometry/bucketing tests; real-model parity is separate."""
import math
import unittest

try:
    import mlx.core as mx
    from mlx_model import inverse_rotate, recover_joints, relative_buckets
except ImportError:
    mx = None


@unittest.skipIf(mx is None, 'Optional MLX runtime is not installed')
class MlxGeometryTests(unittest.TestCase):
    def test_signed_relative_buckets_follow_t5(self):
        values = mx.array([-256, -8, -1, 0, 1, 8, 256])
        self.assertEqual(relative_buckets(values, True).tolist(), [15, 8, 1, 0, 17, 24, 31])
        self.assertEqual(relative_buckets(values, False).tolist(), [31, 8, 1, 0, 0, 0, 0])

    def test_inverse_quaternion_has_correct_forward_direction(self):
        q = mx.array([[math.sqrt(.5), 0, math.sqrt(.5), 0]])
        result = inverse_rotate(q, mx.array([[1., 0, 0]]))
        self.assertLess(mx.max(mx.abs(result - mx.array([[0., 0, 1.]]))).item(), 1e-6)

    def test_root_integration_and_left_right_preserved(self):
        features = mx.zeros((1, 4, 263))
        features[..., 3] = 1
        features[..., 1] = .1
        features[..., 2] = .2
        features[..., 4] = .05
        features[..., 7] = -.05
        joints = recover_joints(features)
        self.assertEqual(joints.shape, (1, 4, 22, 3))
        self.assertLess(mx.max(mx.abs(joints[0, :, 0, 0] - mx.array([0, .1, .2, .3]))).item(), 1e-6)
        self.assertLess(mx.max(mx.abs(joints[0, :, 0, 2] - mx.array([0, .2, .4, .6]))).item(), 1e-6)
        self.assertGreater(joints[0, 0, 1, 0].item(), 0)
        self.assertLess(joints[0, 0, 2, 0].item(), 0)


if __name__ == '__main__':
    unittest.main()
