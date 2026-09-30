"""Numerical checks for constraint semantics, GAE and absorbing transitions."""
import unittest
import numpy as np
import torch

from toy_env import TwoCostNavigation
from baseline_ppo import rollout_geometry
from train import ActorCritic, collect, generalized_advantage, update_multipliers, combine_advantages


class CoreTests(unittest.TestCase):
    def test_finite_horizon_returns_without_bootstrap(self):
        signal = torch.tensor([[[1., 2.]], [[3., 4.]], [[5., 6.]]])
        values = torch.ones_like(signal) * 7
        advantage, returns = generalized_advantage(signal, values, gamma=1, gae_lambda=1)
        torch.testing.assert_close(returns, torch.tensor([[[9., 12.]], [[8., 10.]], [[5., 6.]]]))
        torch.testing.assert_close(advantage, returns - values)

    def test_multiplier_signs_and_projection(self):
        result = update_multipliers(torch.tensor([0.1, 0.1]), torch.tensor([2., 0.]), "multi", .5)
        torch.testing.assert_close(result, torch.tensor([.6, 0.]))
        # A loose aggregate can pass while an individual cost fails.
        costs = torch.tensor([1.5, .3])
        result = update_multipliers(torch.tensor([0.1]), costs, "aggregate_loose", .5)
        self.assertLess(float(result[0]), 1e-6)
        conservative = update_multipliers(torch.tensor([0.1]), costs, "aggregate_conservative", .5)
        self.assertGreater(float(conservative[0]), .1)

    def test_pilot_multiplier_cases(self):
        # (1.5, 0.5): only the hazard multiplier rises; aggregate sum sits exactly on target 2.
        multi = update_multipliers(torch.tensor([0.1, 0.1]), torch.tensor([1.5, .5]), "multi", .02)
        self.assertGreater(float(multi[0]), .1)
        self.assertLess(float(multi[1]), .1)
        loose = update_multipliers(torch.tensor([0.1]), torch.tensor([1.5, .5]), "aggregate_loose", .02)
        torch.testing.assert_close(loose, torch.tensor([.1]))
        # (0.5, 0.5): every positive multiplier falls.
        self.assertTrue(bool((update_multipliers(torch.tensor([0.1, 0.1]), torch.tensor([.5, .5]), "multi", .02) < .1).all()))
        self.assertLess(float(update_multipliers(torch.tensor([0.1]), torch.tensor([.5, .5]), "aggregate_loose", .02)[0]), .1)

    def test_advantage_normalization_preserves_cost_ratios(self):
        a = torch.tensor([[1., 0., 2.], [2., 1., 0.], [0., 2., 1.]])
        multipliers = torch.tensor([2., 3.])
        raw = a[:, 0] - 2*a[:, 1] - 3*a[:, 2]
        expected = (raw - raw.mean()) / raw.std(unbiased=False)
        torch.testing.assert_close(combine_advantages(a, multipliers, "multi"), expected)

    def test_absorbing_goal_and_budget_units(self):
        env = TwoCostNavigation(2, 64, seed=0)
        env.reset()
        env.pos[:] = env.goal
        _, reward, _ = env.step(np.zeros((2, 2)))
        self.assertTrue(env.success.all())
        _, reward, cost = env.step(np.ones((2, 2)))
        np.testing.assert_equal(reward, 0)
        np.testing.assert_equal(cost, 0)
        np.testing.assert_allclose(env.budgets, [1.92, 7.68])

    def test_reproducible_environment(self):
        a, b = TwoCostNavigation(3, seed=42), TwoCostNavigation(3, seed=42)
        np.testing.assert_equal(a.reset(), b.reset())
        for x, y in zip(a.step(np.ones((3, 2))), b.step(np.ones((3, 2)))):
            np.testing.assert_equal(x, y)

    def test_squash_is_bounded_and_initial_ratio_is_one(self):
        policy = ActorCritic(7, 2, 2)
        obs = torch.zeros((16, 7))
        action, latent, logp, _ = policy.act(obs)
        self.assertTrue(bool((action.abs() <= 1).all()))
        torch.testing.assert_close((policy.distribution(obs).log_prob(latent).sum(-1) - logp).exp(), torch.ones(16))

    def test_baseline_reuses_small_environment_pool(self):
        steps, episodes = rollout_geometry(30_000, 1_000, 2)
        self.assertEqual((steps, episodes), (15_000, 30))
        with self.assertRaises(ValueError):
            rollout_geometry(30_000, 1_000, 4)

    def test_evaluation_does_not_retain_training_tensors(self):
        env = TwoCostNavigation(2, horizon=8, seed=3)
        model = ActorCritic(env.obs_dim, env.action_dim, len(env.cost_names))
        batch = collect(model, env, torch.device("cpu"), retain_rollout=False)
        self.assertEqual(set(batch), {"returns", "costs", "diagnostics"})
        self.assertEqual(batch["returns"].shape, (2,))
        self.assertEqual(batch["costs"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
