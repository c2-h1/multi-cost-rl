"""Finite-horizon PPO with scalar or vector Lagrangian constraints.

All methods share the actor and a reward + m cost value network. Keeping cost
heads even for scalar/unconstrained controls matches auxiliary supervision.
Cost critics predict weighted episode costs w_i * C_i, where w_i = 1/d_i
(--cost-weights budget, thresholds are then one) or 1/max(C_i^0, floor) from the
untrained policy pi_0 (--cost-weights pi0, see calibrate_pi0.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from toy_env import TwoCostNavigation


METHODS = ("unconstrained", "aggregate_loose", "aggregate_conservative", "multi")


def initialize(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, num_costs):
        super().__init__()
        self.actor = nn.Sequential(initialize(nn.Linear(obs_dim, 64)), nn.Tanh(),
                                   initialize(nn.Linear(64, 64)), nn.Tanh(),
                                   initialize(nn.Linear(64, action_dim), 0.01))
        self.critic = nn.Sequential(initialize(nn.Linear(obs_dim, 64)), nn.Tanh(),
                                    initialize(nn.Linear(64, 64)), nn.Tanh(),
                                    initialize(nn.Linear(64, 1 + num_costs), 1.0))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-3, 1).exp())

    def act(self, obs, deterministic=False):
        dist = self.distribution(obs)
        latent = dist.mean if deterministic else dist.sample()
        action = torch.tanh(latent)
        # Ratios may use latent densities: the fixed tanh Jacobian cancels.
        return action, latent, dist.log_prob(latent).sum(-1), self.critic(obs)


def generalized_advantage(signals, values, gamma=1.0, gae_lambda=0.95):
    """Complete intrinsic finite-horizon episodes, final V=0 (no truncation)."""
    advantages = torch.zeros_like(signals)
    last = torch.zeros_like(signals[0])
    next_value = torch.zeros_like(values[0])
    for t in reversed(range(len(signals))):
        delta = signals[t] + gamma * next_value - values[t]
        last = delta + gamma * gae_lambda * last
        advantages[t] = last
        next_value = values[t]
    return advantages, advantages + values


def update_multipliers(lambdas, normalized_costs, method, dual_lr, target=1.0, weighted_budgets=None):
    """normalized_costs are w_i*C_i; weighted_budgets are w_i*d_i (all ones when w_i = 1/d_i).

    multi: w_i C_i <= w_i d_i.  aggregate_loose: sum w_i C_i <= sum w_i d_i.
    aggregate_conservative: sum w_i C_i <= min_i w_i d_i, the largest shared
    bound that still implies every per-cost constraint (1 when w_i = 1/d_i).
    """
    if method == "unconstrained":
        return torch.zeros_like(lambdas)
    if weighted_budgets is None:
        weighted_budgets = torch.ones_like(normalized_costs)
    limits = target * weighted_budgets
    violation = normalized_costs - limits if method == "multi" else (
        normalized_costs.sum().reshape(1) - (limits.sum() if method == "aggregate_loose" else limits.min()))
    return (lambdas + dual_lr * violation).clamp(min=0)


def cost_penalty(advantages, lambdas, method):
    if method == "unconstrained":
        return torch.zeros_like(advantages[:, 0])
    if method == "multi":
        return (advantages[:, 1:] * lambdas).sum(-1)
    return lambdas[0] * advantages[:, 1:].sum(-1)


def combine_advantages(advantages, lambdas, method):
    mixed = advantages[:, 0] - cost_penalty(advantages, lambdas, method)
    # Normalize once, after mixing, preserving relative cost units.
    return (mixed - mixed.mean()) / (mixed.std(unbiased=False) + 1e-8)


def make_env(args, seed, num_envs=None):
    n = num_envs or args.num_envs
    if args.env == "toy":
        return TwoCostNavigation(n, args.horizon, seed)
    if args.env == "safety":
        from safety_adapter import SafetyGoalBatch
        return SafetyGoalBatch(n, seed, args.safety_id, (args.budget_hazard, args.budget_vase))
    from humanoid_adapter import HumanoidBatch
    return HumanoidBatch(n, args.horizon, seed, args.humanoid_id)


def load_pi0_costs(path, cost_names, env_id):
    """Per-cost mean episode costs C_i^0 of pi_0 from a calibrate_pi0.py record."""
    record = json.loads(Path(path).read_text())
    if list(record["cost_names"]) != list(cost_names) or record["env_id"] != env_id:
        raise ValueError(f"{path} is for {record['env_id']} {record['cost_names']}, not {env_id} {list(cost_names)}")
    return np.asarray([record["pi0_mean_costs"][n] for n in cost_names], dtype=np.float32), record


def collect(model, env, device, deterministic=False, retain_rollout=True, cost_scale=None):
    """Collect complete episodes.

    Evaluation only needs episode summaries.  Keeping every observation,
    action, value, and signal on the accelerator during evaluation caused
    avoidable memory spikes when several jobs evaluated together.
    Training signals use cost / cost_scale (default: the budgets).
    """
    cost_scale = env.budgets if cost_scale is None else cost_scale
    obs = env.reset()
    observations, latents, log_probs, values, signals = [], [], [], [], []
    rewards, raw_costs = [], []
    for _ in range(env.horizon):
        obs_t = torch.as_tensor(obs, device=device)
        with torch.no_grad():
            action, latent, logp, value = model.act(obs_t, deterministic)
        next_obs, reward, cost = env.step(action.cpu().numpy())
        if retain_rollout:
            observations.append(obs_t)
            latents.append(latent)
            log_probs.append(logp)
            values.append(value)
            signals.append(torch.as_tensor(
                np.concatenate([reward[:, None], cost / cost_scale], -1), device=device
            ))
        rewards.append(reward)
        raw_costs.append(cost)
        obs = next_obs
    result = {"returns": np.stack(rewards).sum(0),
              "costs": np.stack(raw_costs).sum(0), "diagnostics": env.diagnostics()}
    if retain_rollout:
        result.update({"obs": torch.stack(observations), "latent": torch.stack(latents),
                       "logp": torch.stack(log_probs), "values": torch.stack(values),
                       "signals": torch.stack(signals)})
    return result


def collect_chunked(model, args, seed, episodes, device, deterministic=False, chunk=None):
    """Evaluate `episodes` complete episodes, at most `chunk` simulators alive at once.

    Each Safety-Gymnasium env holds ~200 MB, so one 50-env batch per process
    would exhaust memory when several runs finish together.
    """
    chunk = chunk or args.eval_num_envs
    if chunk <= 0:
        raise ValueError("eval-num-envs must be positive")
    parts, sizes = [], []
    for start in range(0, episodes, chunk):
        size = min(chunk, episodes - start)
        env = make_env(args, seed + start, size)
        try:
            batch = collect(model, env, device, deterministic, retain_rollout=False)
            batch["layout_seeds"] = np.asarray(getattr(env, "layout_seeds", []))
        finally:
            env.close()
        parts.append(batch)
        sizes.append(size)
    weights = np.asarray(sizes) / episodes
    return {"returns": np.concatenate([b["returns"] for b in parts]),
            "costs": np.concatenate([b["costs"] for b in parts]),
            "layout_seeds": np.concatenate([b["layout_seeds"] for b in parts]),
            "diagnostics": {k: float(sum(w * b["diagnostics"][k] for w, b in zip(weights, parts)))
                            for k in parts[0]["diagnostics"]}}


def summarize_rollout(batch, env, prefix):
    costs, returns = batch["costs"], batch["returns"]
    normalized = costs / env.budgets
    result = {f"{prefix}/return_mean": float(returns.mean()),
              f"{prefix}/return_std": float(returns.std()),
              f"{prefix}/joint_episode_budget_pass_rate": float((normalized <= 1).all(1).mean()),
              f"{prefix}/max_mean_normalized_cost": float(normalized.mean(0).max()),
              f"{prefix}/all_mean_costs_feasible": bool((normalized.mean(0) <= 1).all())}
    for i, name in enumerate(env.cost_names):
        result[f"{prefix}/cost_{name}"] = float(costs[:, i].mean())
        result[f"{prefix}/normalized_{name}"] = float(normalized[:, i].mean())
        result[f"{prefix}/episode_violation_{name}"] = float((normalized[:, i] > 1).mean())
    result.update({f"{prefix}/{k}": v for k, v in batch["diagnostics"].items()})
    return result


def git_state():
    """Commit and whether tracked files differ from it (untracked files ignored)."""
    try:
        run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True,
                                        cwd=Path(__file__).parent).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "branch": "unknown", "dirty": None}


def load_experiment_config(path):
    """configs/*.json: train.py arguments live under "train_args" (argparse dest names).

    "extends": "<relative path>" merges a base config's train_args underneath.
    """
    path = Path(path)
    config = json.loads(path.read_text())
    if "extends" in config:
        base = load_experiment_config(path.parent / config["extends"])
        config["train_args"] = {**base.get("train_args", {}), **config.get("train_args", {})}
    return config


def start_wandb(args, output, config):
    if args.wandb_mode == "disabled":
        return None
    import wandb
    for key, suffix in [("WANDB_CACHE_DIR", "cache"), ("WANDB_CONFIG_DIR", "config"),
                        ("WANDB_DATA_DIR", "data")]:
        os.environ.setdefault(key, str(Path(".cache/wandb", suffix).resolve()))
    os.environ.setdefault("WANDB_SILENT", "true")
    try:
        return wandb.init(project=args.project, entity=args.entity, name=output.name,
                          group=args.group, config=config, dir=str(output),
                          mode=args.wandb_mode, settings=wandb.Settings(init_timeout=40))
    except Exception as error:
        print(f"W&B {args.wandb_mode} initialization failed ({type(error).__name__}); retaining local logs and using offline mode", flush=True)
        return wandb.init(project=args.project, name=output.name, group=args.group,
                          config=config, dir=str(output), mode="offline")


def train(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = make_env(args, args.seed)
    # Intrinsic horizon comes from the environment when it defines one.
    args.horizon = env.horizon
    model = ActorCritic(env.obs_dim, env.action_dim, len(env.cost_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    lambdas = torch.full((len(env.cost_names) if args.method == "multi" else 1,), args.lambda_init, device=device)
    if args.method == "unconstrained":
        lambdas.zero_()
    # Cost weights w_i = 1/cost_scale_i. Budget mode keeps the old division by d_i exactly.
    env_id = args.safety_id if args.env == "safety" else args.env
    pi0_costs = pi0_record = None
    if args.pi0_calibration:
        pi0_costs, pi0_record = load_pi0_costs(args.pi0_calibration, env.cost_names, env_id)
        if pi0_record["seed"] != args.seed:
            print(f"Note: pi_0 calibration seed {pi0_record['seed']} != training seed {args.seed}", flush=True)
    if args.cost_weights == "pi0":
        cost_scale = np.maximum(pi0_costs, args.weight_floor).astype(env.budgets.dtype)
    else:
        cost_scale = env.budgets
    weighted_budgets = torch.as_tensor(env.budgets / cost_scale, device=device)
    config = vars(args).copy()
    named = lambda values: None if values is None else dict(zip(env.cost_names, np.asarray(values, dtype=float).tolist()))
    config.update({"cost_weight_values": named(1 / cost_scale), "pi0_costs": named(pi0_costs),
                   "budgets_by_cost": named(env.budgets), "weighted_budgets": named(env.budgets / cost_scale)})
    config.update({"cost_names": list(env.cost_names), "budgets": env.budgets.tolist(),
                   "python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "platform": platform.platform(),
                   "start_utc": datetime.now(timezone.utc).isoformat(),
                   "actor_parameters": sum(p.numel() for p in model.actor.parameters()) + env.action_dim,
                   "total_parameters": sum(p.numel() for p in model.parameters()),
                   "gamma": 1.0, "episode_semantics": "intrinsic_fixed_horizon_with_absorption",
                   "environment_metadata": getattr(env, "metadata", {})})
    source_dir = output / "source"
    source_dir.mkdir()
    hashes = {}
    for name in ["train.py", "toy_env.py", "safety_adapter.py", "humanoid_adapter.py", "humanoid_costs.py"]:
        path = Path(name)
        if path.exists():
            shutil.copy2(path, source_dir / name)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    config["source_sha256"] = hashes
    config["git"] = git_state()
    config["experiment_config"] = load_experiment_config(args.config) if args.config else None
    config["evaluation_protocol"] = {
        "checkpoint_eval": {"environment_seed": 10000 + args.seed, "episodes": args.eval_episodes,
                            "chunk": args.eval_num_envs, "policy": "stochastic"},
        "final_eval": {"environment_seed": 200000 + args.seed, "action_seed": 300000 + args.seed,
                       "episodes": args.final_eval_episodes, "chunk": args.eval_num_envs,
                       "policy": "stochastic and deterministic"}}
    if args.pi0_calibration:
        shutil.copy2(args.pi0_calibration, output / "pi0_calibration.json")
    (output / "config.json").write_text(json.dumps(config, indent=2))
    wb = start_wandb(args, output, config)
    wandb_info = {"mode": wb.settings.mode, "id": wb.id, "url": wb.url} if wb else {"mode": "disabled"}
    (output / "wandb-info.json").write_text(json.dumps(wandb_info, indent=2))
    log = (output / "metrics.jsonl").open("w", buffering=1)
    start = time.monotonic()
    cumulative_cost = np.zeros(len(env.cost_names))
    cumulative_excess = np.zeros(len(env.cost_names))

    def emit(metrics):
        metrics["elapsed_seconds"] = time.monotonic() - start
        log.write(json.dumps(metrics) + "\n")
        if wb:
            wb.log(metrics, step=int(metrics["env_steps"]))
        print(json.dumps(metrics), flush=True)

    with torch.no_grad():
        initial = collect_chunked(model, args, 10000 + args.seed, args.eval_episodes, device)
    initial_metrics = summarize_rollout(initial, env, "eval")
    emit({"update": 0, "env_steps": 0, **initial_metrics})
    for update in range(1, args.updates + 1):
        batch = collect(model, env, device, cost_scale=cost_scale)
        advantage, returns = generalized_advantage(batch["signals"], batch["values"], gae_lambda=args.gae_lambda)
        flat = {name: batch[name].flatten(0, 1) for name in ("obs", "latent", "logp")}
        # The rollout uses the old multipliers; update dual variables after PPO.
        flat_advantage = advantage.flatten(0, 1)
        mixed = combine_advantages(flat_advantage, lambdas, args.method)
        # Scale check: a growing multiplier with negligible cost term suggests a units problem.
        advantage_scale = {"train/adv_reward_abs": float(flat_advantage[:, 0].abs().mean()),
                           "train/adv_weighted_cost_abs": float(cost_penalty(flat_advantage, lambdas, args.method).abs().mean())}
        returns = returns.flatten(0, 1)
        count = len(mixed)
        kls, policy_losses, value_losses = [], [], []
        stop = False
        for _ in range(args.epochs):
            indices = torch.randperm(count, device=device)
            for idx in indices.split(args.minibatch_size):
                dist = model.distribution(flat["obs"][idx])
                new_logp = dist.log_prob(flat["latent"][idx]).sum(-1)
                log_ratio = new_logp - flat["logp"][idx]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                if kl > args.target_kl:
                    stop = True
                    break
                policy_loss = -torch.minimum(ratio * mixed[idx], ratio.clamp(1-args.clip, 1+args.clip) * mixed[idx]).mean()
                value_loss = 0.5 * (model.critic(flat["obs"][idx]) - returns[idx]).square().mean()
                # Latent Gaussian entropy is only an exploration heuristic;
                # default coefficient=0, avoiding a squashed-entropy claim.
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * dist.entropy().sum(-1).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                kls.append(float(kl))
                policy_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
            if stop:
                break
        normalized = torch.as_tensor((batch["costs"] / cost_scale).mean(0), device=device)
        lambdas = update_multipliers(lambdas, normalized, args.method, args.dual_lr, args.constraint_target,
                                     weighted_budgets)
        cumulative_cost += batch["costs"].sum(0)
        cumulative_excess += np.maximum(batch["costs"] - env.budgets, 0).sum(0)
        metrics = {"update": update, "env_steps": update * count,
                   "train/approx_kl": float(np.mean(kls)) if kls else 0.0,
                   "train/policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
                   "train/value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
                   "train/kl_early_stop": stop, **advantage_scale,
                   **summarize_rollout(batch, env, "train")}
        for i, value in enumerate(lambdas):
            metrics[f"dual/lambda_{i}"] = float(value)
        for i, name in enumerate(env.cost_names):
            metrics[f"train/cumulative_cost_{name}"] = float(cumulative_cost[i])
            metrics[f"train/cumulative_episode_excess_{name}"] = float(cumulative_excess[i])
        if update % args.eval_every == 0 or update == args.updates:
            # Evaluation does not consume training environment or torch RNG state.
            rng_state = torch.random.get_rng_state()
            if device.type == "cuda":
                accelerator_state = torch.cuda.get_rng_state(device)
            elif device.type == "mps":
                accelerator_state = torch.mps.get_rng_state()
            # Same evaluation layouts at every checkpoint; environments are
            # created in a small pool and closed immediately after use.
            with torch.no_grad():
                evaluated = collect_chunked(model, args, 10000 + args.seed, args.eval_episodes, device)
            torch.random.set_rng_state(rng_state)
            if device.type == "cuda":
                torch.cuda.set_rng_state(accelerator_state, device)
            elif device.type == "mps":
                torch.mps.set_rng_state(accelerator_state)
            metrics.update(summarize_rollout(evaluated, env, "eval"))
        if update % args.log_every == 0 or update % args.eval_every == 0 or update == args.updates:
            emit(metrics)
        if update % args.checkpoint_every == 0 or update == args.updates:
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "multipliers": lambdas, "config": config, "update": update, "env_steps": update * count}
            # Step-specific copies keep a common earlier checkpoint for matched-budget comparison.
            torch.save(state, output / f"checkpoint-{update * count:09d}.pt")
            torch.save(state, output / "checkpoint.pt")
    # Fresh final evaluation seeds, never used for policy or checkpoint selection.
    torch.manual_seed(300000 + args.seed)
    with torch.no_grad():
        final_stochastic = collect_chunked(model, args, 200000 + args.seed, args.final_eval_episodes, device)
        final_deterministic = collect_chunked(model, args, 200000 + args.seed, args.final_eval_episodes,
                                              device, deterministic=True)
    np.savez_compressed(output / "evaluation.npz", returns=final_stochastic["returns"],
                        costs=final_stochastic["costs"], deterministic_returns=final_deterministic["returns"],
                        deterministic_costs=final_deterministic["costs"], budgets=env.budgets)
    summary = {"method": args.method, "env": args.env, "seed": args.seed,
               "env_steps": args.updates * args.horizon * args.num_envs,
               "elapsed_seconds": time.monotonic() - start,
               "initial": initial_metrics,
               **summarize_rollout(final_stochastic, env, "final"),
               **summarize_rollout(final_deterministic, env, "deterministic"),
               "multipliers": lambdas.cpu().tolist(), "wandb": wandb_info}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if wb:
        wb.summary.update(summary)
        wb.finish()
    log.close()
    env.close()
    print("COMPLETE " + json.dumps(summary), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=("toy", "safety", "humanoid"), default="toy")
    p.add_argument("--safety-id", default="SafetyPointGoal2-v0")
    p.add_argument("--budget-hazard", type=float, default=None, help="Safety-Gymnasium hazard budget, affected steps/episode")
    p.add_argument("--budget-vase", type=float, default=None, help="Safety-Gymnasium vase budget, affected steps/episode")
    p.add_argument("--cost-weights", choices=("budget", "pi0"), default="budget",
                   help="budget: w_i = 1/d_i (old behavior); pi0: w_i = 1/max(C_i^0, floor) from --pi0-calibration")
    p.add_argument("--pi0-calibration", default=None, help="calibrate_pi0.py JSON with C_i^0")
    p.add_argument("--weight-floor", type=float, default=1.0, help="floor on C_i^0 for pi0 weights")
    p.add_argument("--budget-from-pi0", type=float, default=None, metavar="ALPHA",
                   help="Safety only: set every budget d_i = ALPHA * C_i^0 instead of --budget-hazard/--budget-vase")
    p.add_argument("--humanoid-id", default="h1-walk-v0")
    p.add_argument("--method", choices=METHODS, default="multi")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dual-lr", type=float, default=0.15)
    p.add_argument("--constraint-target", type=float, default=1.0,
                   help="Training threshold as fraction of reported budgets; <1 is an explicit safety margin")
    p.add_argument("--lambda-init", type=float, default=0.1)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--value-coef", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--final-eval-episodes", type=int, default=512)
    p.add_argument("--eval-num-envs", type=int, default=2,
                   help="maximum evaluation simulators alive at once")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    p.add_argument("--project", default="multi-cost-rl")
    p.add_argument("--entity", default=None)
    p.add_argument("--group", default="toy-pilot")
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=None, help="configs/*.json; its train_args become defaults, CLI flags override")
    p.add_argument("--experiment-id", default=None, help="registry ID (docs/experiments/INDEX.md); default from --config")
    return p


def parse_args(argv=None):
    """CLI flags override --config train_args, which override parser defaults."""
    p = parser()
    known, _ = p.parse_known_args(argv)
    if known.config:
        experiment = load_experiment_config(known.config)
        train_args = experiment.get("train_args", {})
        unknown = sorted(set(train_args) - {a.dest for a in p._actions})
        if unknown:
            raise ValueError(f"{known.config} has train_args unknown to this train.py: {unknown} "
                             f"(it may describe another commit: {experiment.get('code')})")
        p.set_defaults(**train_args, experiment_id=experiment.get("id"))
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if not 0 < args.constraint_target <= 1:
        raise ValueError("constraint-target must be in (0, 1]")
    if (args.cost_weights == "pi0" or args.budget_from_pi0 is not None) and not args.pi0_calibration:
        raise ValueError("--cost-weights pi0 and --budget-from-pi0 need --pi0-calibration")
    if not args.weight_floor > 0:
        raise ValueError("weight-floor must be positive")
    if args.budget_from_pi0 is not None:
        if args.env != "safety" or not args.budget_from_pi0 > 0:
            raise ValueError("--budget-from-pi0 needs --env safety and ALPHA > 0")
        if args.budget_hazard is not None or args.budget_vase is not None:
            raise ValueError("--budget-from-pi0 replaces --budget-hazard/--budget-vase; pass only one")
        c0, _ = load_pi0_costs(args.pi0_calibration, ("hazard", "vase"), args.safety_id)
        args.budget_hazard, args.budget_vase = (float(args.budget_from_pi0 * c) for c in c0)
    if args.env == "safety" and (args.budget_hazard is None or args.budget_vase is None):
        raise ValueError("--env safety requires explicit --budget-hazard and --budget-vase")
    if args.env == "humanoid" and args.device == "mps":
        raise ValueError("Use CPU for the small local humanoid pilot; CUDA is supported on a cluster")
    train(args)
