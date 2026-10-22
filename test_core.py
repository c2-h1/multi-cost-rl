"""Numerical checks for constraint semantics, GAE and absorbing transitions."""
import unittest
import numpy as np
import torch

from toy_env import TwoCostNavigation
from baseline_ppo import rollout_geometry
from train import ActorCritic, collect, generalized_advantage, update_multipliers, combine_advantages, parse_args

import importlib.util
import json
import tempfile
from pathlib import Path

_spec = importlib.util.spec_from_file_location("summarize_runs", Path(__file__).parent / "scripts" / "summarize_runs.py")
summarize_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize_runs)


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

    def test_budget_weights_reproduce_old_multiplier_update(self):
        # Previous update_multipliers, verbatim: costs already divided by budgets, threshold one.
        def old(lambdas, normalized, method, lr, target=1.0):
            violation = normalized - target if method == "multi" else (
                normalized.sum().reshape(1) - target * (len(normalized) if method == "aggregate_loose" else 1))
            return (lambdas + lr * violation).clamp(min=0)
        rng = np.random.default_rng(0)
        budgets = np.array([31.9, 73.9], dtype=np.float32)
        weighted_budgets = torch.as_tensor(budgets / budgets)  # w_i * d_i with w_i = 1/d_i, as train.py builds it
        for _ in range(20):
            normalized = torch.as_tensor((rng.uniform(0, 150, (4, 2)).astype(np.float32) / budgets).mean(0))
            for method, size in (("multi", 2), ("aggregate_loose", 1), ("aggregate_conservative", 1)):
                lambdas = torch.as_tensor(rng.uniform(0, 1, size).astype(np.float32))
                for target in (1.0, 0.8):
                    expected = old(lambdas, normalized, method, .02, target)
                    self.assertTrue(torch.equal(update_multipliers(lambdas, normalized, method, .02, target, weighted_budgets), expected))
                    self.assertTrue(torch.equal(update_multipliers(lambdas, normalized, method, .02, target), expected))

    def test_weighted_violations(self):
        costs, budgets = np.array([40., 3.], dtype=np.float32), np.array([31.9, 73.9], dtype=np.float32)
        scale = np.maximum(np.array([25., .2], dtype=np.float32), 1.0)  # floor 1 on C^0
        w = 1 / scale
        weighted_costs, weighted_budgets = torch.as_tensor(costs / scale), torch.as_tensor(budgets / scale)
        start = torch.tensor([100.])  # far from the projection at zero
        aggregate = update_multipliers(start, weighted_costs, "aggregate_loose", 1., 1., weighted_budgets) - start
        torch.testing.assert_close(aggregate, torch.tensor([float((w * (costs - budgets)).sum())]))
        multi = update_multipliers(start.repeat(2), weighted_costs, "multi", 1., 1., weighted_budgets) - 100
        torch.testing.assert_close(multi, torch.as_tensor(w * (costs - budgets)))
        conservative = update_multipliers(start, weighted_costs, "aggregate_conservative", 1., 1., weighted_budgets) - start
        torch.testing.assert_close(conservative, torch.tensor([float((w * costs).sum() - (w * budgets).min())]))

    def test_collect_signals_use_cost_scale(self):
        env = TwoCostNavigation(4, horizon=16, seed=5)
        model = ActorCritic(env.obs_dim, env.action_dim, len(env.cost_names))

        def rollout(**kwargs):  # fresh env and action RNG: identical episodes
            torch.manual_seed(0)
            return collect(model, TwoCostNavigation(4, horizon=16, seed=5), torch.device("cpu"), **kwargs)
        default, budget = rollout(), rollout(cost_scale=env.budgets)
        self.assertTrue(torch.equal(default["signals"], budget["signals"]))
        scale = np.array([0.5, 4.0], dtype=np.float32)
        weighted = rollout(cost_scale=scale)
        np.testing.assert_equal(weighted["costs"], default["costs"])
        np.testing.assert_allclose(weighted["signals"][..., 1:].sum(0).numpy() * scale, weighted["costs"], rtol=1e-5)

    def test_native_feasibility_differs_from_per_cost_budgets(self):
        budgets, weights = [31.9, 73.9], [1 / 31.9, 1 / 73.9]
        costs = [1.5 * 31.9, 0.4 * 73.9]  # x = (1.5, 0.4): sum 1.9
        self.assertTrue(summarize_runs.native_feasible("aggregate_loose", costs, budgets, weights))
        self.assertFalse(summarize_runs.native_feasible("aggregate_conservative", costs, budgets, weights))
        self.assertFalse(summarize_runs.native_feasible("multi", costs, budgets, weights))
        self.assertIsNone(summarize_runs.native_feasible("unconstrained", costs, budgets, weights))
        self.assertTrue(summarize_runs.native_feasible("multi", [0.9 * 31.9, 73.9], budgets, weights))

    def test_config_defaults_extends_and_cli_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "base.json").write_text(json.dumps({"train_args": {"dual_lr": 0.02, "num_envs": 4}}))
            (Path(tmp) / "exp.json").write_text(json.dumps({"id": "EX", "extends": "base.json",
                                                            "train_args": {"num_envs": 8, "updates": 7}}))
            args = parse_args(["--config", f"{tmp}/exp.json", "--updates", "9", "--output", "o"])
            self.assertEqual((args.dual_lr, args.num_envs, args.updates, args.experiment_id), (0.02, 8, 9, "EX"))
            (Path(tmp) / "old.json").write_text(json.dumps({"train_args": {"eval_chunk": 10}}))
            with self.assertRaises(ValueError):
                parse_args(["--config", f"{tmp}/old.json", "--output", "o"])
        self.assertEqual(parse_args(["--output", "o"]).dual_lr, 0.15)  # parser default unchanged

    def test_repository_configs_load(self):
        for path in Path(__file__).parent.joinpath("configs").rglob("*.json"):
            config = json.loads(path.read_text())
            if config.get("code", {}).get("branch") == "branch1":
                continue  # branch1 arguments (eval_chunk) are refused by this train.py on purpose
            parse_args(["--config", str(path), "--output", "o"])

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
