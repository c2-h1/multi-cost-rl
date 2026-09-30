"""Physics integration checks: PYTHONPATH=external/humanoid-bench python -m unittest test_humanoid_costs."""
import unittest

import gymnasium as gym
import humanoid_bench  # noqa: F401: registers the benchmark
import mujoco
import numpy as np

from humanoid_costs import HumanoidCostConfig, HumanoidCostWrapper


class HumanoidCostIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.env = HumanoidCostWrapper(gym.make("h1-walk-v0", render_mode=None))
        self.env.reset(seed=41)

    def tearDown(self):
        self.env.close()

    def test_reward_dynamics_and_termination_are_preserved(self):
        reference = gym.make("h1-walk-v0", render_mode=None)
        try:
            reference.reset(seed=41)
            rng = np.random.default_rng(42)
            for _ in range(25):
                action = rng.uniform(-0.2, 0.2, size=self.env.action_space.shape)
                actual = self.env.step(action)
                expected = reference.step(action)
                np.testing.assert_array_equal(actual[0], expected[0])
                self.assertEqual(actual[1:4], expected[1:4])
                self.assertEqual(actual[4]["costs"].shape, (3,))
                self.assertTrue(np.all(np.isfinite(actual[4]["costs"])))
                self.assertTrue(np.all(actual[4]["costs"] >= 0))
                if actual[2] or actual[3]:
                    break
        finally:
            reference.close()

    def test_low_head_and_physical_failure_activate_fall_channel(self):
        self.assertEqual(self.env.safety_signals()[0][0], 0)
        base = self.env.unwrapped
        base.data.qpos[2] -= 0.6
        mujoco.mj_forward(base.model, base.data)
        self.assertEqual(self.env.safety_signals()[0][0], 1)
        self.env.reset(seed=41)
        self.assertEqual(self.env.safety_signals(terminated=True)[0][0], 1)

    def test_near_joint_stop_is_visible_and_free_root_is_excluded(self):
        base = self.env.unwrapped
        baseline = self.env.safety_signals()[0][1]
        self.assertNotIn(0, self.env.qpos_ids.tolist())
        joint_qpos = self.env.qpos_ids[0]
        lower, upper = self.env.joint_bounds[0]
        base.data.qpos[joint_qpos] = lower + 0.01 * (upper - lower)
        mujoco.mj_forward(base.model, base.data)
        costs, metrics = self.env.safety_signals()
        self.assertGreater(costs[1], baseline)
        self.assertLess(metrics["safety/min_joint_margin_fraction"], 0.02)

    def test_force_load_is_simulated_force_not_normalized_action(self):
        _, _, _, _, info = self.env.step(np.zeros(self.env.action_space.shape))
        # Position servos exert forces even though all normalized commands are 0.
        self.assertGreater(info["cost/actuator_load"], 0.0)
        self.assertGreater(info["safety/absolute_mechanical_power_w"], 0.0)
        self.assertLessEqual(info["safety/max_actuator_load_fraction"], 1.0 + 1e-6)

    def test_invalid_margin_fails(self):
        with self.assertRaises(ValueError):
            HumanoidCostWrapper(self.env, HumanoidCostConfig(joint_margin_fraction=0.6))


if __name__ == "__main__":
    unittest.main()
