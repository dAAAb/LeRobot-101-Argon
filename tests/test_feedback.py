import unittest
from server.feedback import EndpointFeedback

class EndpointFeedbackTests(unittest.TestCase):
    def make(self, **kw):
        args = dict(joint="elbow_flex", logical_target=2000, lower=850, upper=3033)
        args.update(kw)
        return EndpointFeedback(**args)

    def test_constant_37_tick_load_converges(self):
        controller = self.make()
        goal = controller.motor_goal
        writes = []
        for sample in range(26):
            now = sample * .12
            decision = controller.observe(goal + 37, now)
            if decision.status == "propose":
                self.assertLessEqual(abs(decision.goal - goal), 6)
                self.assertLessEqual(abs(decision.bias), 60)
                writes.append((now, decision.goal))
                goal = decision.goal
                controller.commit(decision, now)
            elif decision.status in ("done", "abort"):
                break
        self.assertEqual(decision.status, "done", decision.reason)
        self.assertLessEqual(abs(goal + 37 - 2000), 2)
        self.assertLess(now, 3)
        self.assertTrue(all(b[0] - a[0] >= .12 - 1e-9 for a, b in zip(writes, writes[1:])))
        self.assertEqual(controller.motor_goal, goal)  # successful bias is retained

    def test_frozen_joint_aborts_after_four_writes(self):
        controller = self.make()
        for sample in range(8):
            decision = controller.observe(2037, sample * .12)
            if decision.status == "propose":
                controller.commit(decision, sample * .12)
            elif decision.status == "abort":
                break
        self.assertEqual(decision.status, "abort")
        self.assertIn("no progress", decision.reason)
        self.assertEqual(controller.corrections, 4)
        self.assertEqual(controller.bias, -24)

    def test_large_overshoot_reversal_aborts_without_unwind_write(self):
        controller = self.make()
        first = controller.observe(2037, 0)
        controller.commit(first, 0)
        decision = controller.observe(1991, .12)
        self.assertEqual(decision.status, "abort")
        self.assertIn("reversed", decision.reason)
        self.assertIsNone(decision.goal)
        self.assertEqual(controller.bias, -6)

    def test_boundary_guard_does_not_clip_or_wind_up(self):
        controller = self.make(logical_target=3020, upper=3024)
        decision = controller.observe(2983, 0)
        self.assertEqual(decision.status, "abort")
        self.assertIn("joint limit", decision.reason)
        self.assertEqual(controller.bias, 0)
        self.assertEqual(controller.corrections, 0)
        self.assertIsNone(decision.goal)

    def test_second_large_reversal_also_aborts(self):
        controller = self.make()
        decision = controller.observe(2037, 0)
        controller.commit(decision, 0)
        decision = controller.observe(1995, .12)  # small reversal may unwind
        self.assertEqual(decision.status, "propose")
        controller.commit(decision, .12)
        decision = controller.observe(2009, .24)
        self.assertEqual(decision.status, "abort")
        self.assertIn("reversed", decision.reason)

    def test_bias_budget_guard(self):
        controller = self.make(initial_bias=-58)
        decision = controller.observe(2037, 0)
        self.assertEqual(decision.status, "abort")
        self.assertIn("budget", decision.reason)
        self.assertEqual(controller.bias, -58)

    def test_sustained_samples_cannot_be_fast_polled(self):
        controller = self.make()
        self.assertEqual(controller.observe(2001, 0).status, "settling")
        self.assertEqual(controller.observe(2001, .01).status, "wait")
        self.assertEqual(controller.observe(2001, .12).status, "settling")
        self.assertEqual(controller.observe(2001, .24).status, "done")
        self.assertEqual(controller.corrections, 0)

    def test_out_of_tolerance_fast_sample_breaks_stability(self):
        controller = self.make()
        controller.observe(2001, 0)
        controller.observe(2001, .12)
        self.assertEqual(controller.observe(2005, .13).status, "wait")
        self.assertEqual(controller.observe(2001, .24).status, "settling")
        self.assertEqual(controller.observe(2001, .36).status, "settling")
        self.assertEqual(controller.observe(2001, .48).status, "done")

    def test_three_second_deadline(self):
        controller = self.make()
        controller.observe(2000, 0)
        decision = controller.observe(2000, 3)
        self.assertEqual(decision.status, "abort")
        self.assertIn("3 seconds", decision.reason)

    def test_gripper_is_excluded(self):
        with self.assertRaises(ValueError):
            self.make(joint="gripper")

    def test_proposal_does_not_change_committed_bias(self):
        controller = self.make(initial_bias=-10)
        decision = controller.observe(2027, 0)
        self.assertEqual(controller.bias, -10)
        self.assertEqual(decision.bias, -16)
        controller.commit(decision, 0)
        self.assertEqual(controller.bias, -16)
        with self.assertRaises(ValueError):
            controller.commit(decision, 0)

    def test_uncommitted_write_cannot_be_counted_as_progress(self):
        controller = self.make()
        controller.observe(2037, 0)
        decision = controller.observe(2037, .12)
        self.assertEqual(decision.status, "abort")
        self.assertEqual(controller.corrections, 0)
        self.assertEqual(controller.bias, 0)

    def test_preserved_bias_on_next_endpoint(self):
        controller = self.make(logical_target=2010, initial_bias=-35)
        self.assertEqual(controller.motor_goal, 1975)
        for t in [0, .12, .24]:
            decision = controller.observe(controller.motor_goal + 37, t)
        self.assertEqual(decision.status, "done")
        self.assertEqual(controller.bias, -35)
        self.assertEqual(controller.corrections, 0)

    def test_rate_limit_starts_after_completed_write(self):
        controller = self.make()
        decision = controller.observe(2037, 0)
        controller.commit(decision, .10)  # simulated slow successful write
        self.assertEqual(controller.observe(2031, .12).status, "wait")
        self.assertEqual(controller.observe(2031, .20).status, "wait")
        self.assertEqual(controller.observe(2031, .24).status, "propose")


if __name__ == "__main__":
    unittest.main(verbosity=2)
