"""Numerical checks for constraint semantics, GAE and absorbing transitions."""
import unittest
import numpy as np
import torch

from toy_env import TwoCostNavigation
from baseline_ppo import rollout_geometry
from train import ActorCritic, collect, generalized_advantage, update_multipliers, pi_update, combine_advantages, parse_args

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

    def test_pi_with_zero_kp_matches_integral_update(self):
        # Random sequence with positive and negative errors and hits on the zero boundary.
        g = torch.Generator().manual_seed(0)
        for method, size in (("multi", 2), ("aggregate_loose", 1), ("aggregate_conservative", 1)):
            lam = torch.full((size,), 0.1)
            integral = lam.clone()
            for _ in range(500):
                x = torch.rand(2, generator=g) * 3
                lam = update_multipliers(lam, x, method, .04)
                integral, pi_lam, _, p_term = pi_update(integral, x, method, .04, 0.)
                torch.testing.assert_close(pi_lam, lam, rtol=0, atol=0)
                torch.testing.assert_close(integral, lam, rtol=0, atol=0)
                self.assertTrue(bool((p_term == 0).all()))

    def test_pi_signs_projection_and_separate_integral_state(self):
        zero = torch.zeros(2)
        # Positive error (1.5 - 1) on hazard, negative (0.5 - 1) on vase, from I = lambda = 0.
        integral, lam, error, p_term = pi_update(zero, torch.tensor([1.5, .5]), "multi", .02, 4.)
        torch.testing.assert_close(error, torch.tensor([.5, -.5]))
        torch.testing.assert_close(integral, torch.tensor([.01, 0.]))      # vase integral clipped at 0
        torch.testing.assert_close(p_term, torch.tensor([2., -2.]))
        torch.testing.assert_close(lam, torch.tensor([2.01, 0.]))          # lambda clipped at 0
        # Next step starts from I (0.01), not from lambda (2.01).
        integral2, lam2, _, _ = pi_update(integral, torch.tensor([.9, 1.]), "multi", .02, 4.)
        torch.testing.assert_close(integral2, torch.tensor([.008, 0.]))
        torch.testing.assert_close(lam2, torch.tensor([0., 0.]))           # 0.008 - 0.4 < 0 -> 0, I kept
        # A negative proportional term can zero lambda while I stays positive and recovers it later.
        integral3, lam3, _, _ = pi_update(torch.tensor([1., 1.]), torch.tensor([.5, 1.]), "multi", .02, 4.)
        torch.testing.assert_close(integral3, torch.tensor([.99, 1.]))
        torch.testing.assert_close(lam3, torch.tensor([0., 1.]))
        # Exactly on target: no change.
        integral4, lam4, _, _ = pi_update(torch.tensor([.3, .3]), torch.tensor([1., 1.]), "multi", .02, 4.)
        torch.testing.assert_close(lam4, torch.tensor([.3, .3]))

    def test_pi_flags_require_pi_controller(self):
        args = parse_args(["--output", "x", "--dual-controller", "pi", "--dual-ki", ".02", "--dual-kp", "4",
                           "--checkpoint-steps", "3000000", "4250000"])
        self.assertEqual((args.dual_controller, args.dual_ki, args.dual_kp), ("pi", .02, 4.))
        self.assertEqual(args.checkpoint_steps, [3000000, 4250000])
        default = parse_args(["--output", "x"])
        self.assertEqual((default.dual_controller, default.dual_ki, default.dual_kp), ("integral", None, 0.))

    def test_native_feasibility_differs_from_per_cost_budgets(self):
        budgets = [31.9, 73.9]
        costs = [1.5 * 31.9, 0.4 * 73.9]  # x = (1.5, 0.4): sum 1.9
        self.assertTrue(summarize_runs.native_feasible("aggregate_loose", costs, budgets))
        self.assertFalse(summarize_runs.native_feasible("aggregate_conservative", costs, budgets))
        self.assertFalse(summarize_runs.native_feasible("multi", costs, budgets))
        self.assertIsNone(summarize_runs.native_feasible("unconstrained", costs, budgets))
        self.assertTrue(summarize_runs.native_feasible("multi", [0.9 * 31.9, 73.9], budgets))

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

    def test_render_layout_matches_chunked_evaluation(self):
        from render_policy import layout_seed
        for seed, chunk in [(900000, 10), (10001, 2)]:
            expected = []
            for start in range(0, 2 * chunk, chunk):  # what SafetyGoalBatch.reset draws per chunk
                expected += np.random.default_rng(seed + start).integers(0, 2**31 - 1, chunk).tolist()
            self.assertEqual([layout_seed(seed, chunk, k) for k in range(2 * chunk)], expected)


if __name__ == "__main__":
    unittest.main()
