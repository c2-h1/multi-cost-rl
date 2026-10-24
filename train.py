"""Finite-horizon PPO with scalar or vector Lagrangian constraints.

All methods share the actor and a reward + m cost value network. Keeping cost
heads even for scalar/unconstrained controls matches auxiliary supervision.
Cost critics predict normalized episode costs; thresholds are therefore one.
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


def constraint_error(normalized_costs, method, target=1.0):
    """Signed error per multiplier: x_i - target (multi); sum_i x_i - m * target (aggregate_loose); sum_i x_i - target (aggregate_conservative)."""
    if method == "multi":
        return normalized_costs - target
    return normalized_costs.sum().reshape(1) - target * (len(normalized_costs) if method == "aggregate_loose" else 1)


def update_multipliers(lambdas, normalized_costs, method, dual_lr, target=1.0):
    if method == "unconstrained":
        return torch.zeros_like(lambdas)
    violation = normalized_costs - target if method == "multi" else (
        normalized_costs.sum().reshape(1) - target * (len(normalized_costs) if method == "aggregate_loose" else 1))
    return (lambdas + dual_lr * violation).clamp(min=0)


def pi_update(integral, normalized_costs, method, ki, kp, target=1.0):
    """Projected PI variant with a signed proportional term (not a reproduction of a paper's code).

        e[k]   = x[k] - target
        I[k+1] = max(0, I[k] + ki * e[k])
        lam[k+1] = max(0, I[k+1] + kp * e[k])

    I and lambda are separate states: the next integral step starts from I, never from lambda.
    With kp = 0 this equals update_multipliers with dual_lr = ki when I starts at lambda.
    Returns (integral, lambdas, error, proportional term)."""
    if method == "unconstrained":
        zero = torch.zeros_like(integral)
        return zero, zero.clone(), zero.clone(), zero.clone()
    error = constraint_error(normalized_costs, method, target)
    integral = (integral + ki * error).clamp(min=0)
    proportional = kp * error
    return integral, (integral + proportional).clamp(min=0), error, proportional


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


def collect(model, env, device, deterministic=False, retain_rollout=True):
    """Collect complete episodes.

    Evaluation only needs episode summaries.  Keeping every observation,
    action, value, and signal on the accelerator during evaluation caused
    avoidable memory spikes when several jobs evaluated together.
    """
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
                np.concatenate([reward[:, None], cost / env.budgets], -1), device=device
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
    # PI controller: I and lambda both start at lambda_init (use --lambda-init 0 for I = lambda = 0).
    integral = lambdas.clone()
    dual_ki = args.dual_lr if args.dual_ki is None else args.dual_ki
    checkpoint_steps = sorted(set(args.checkpoint_steps or []))
    config = vars(args).copy()
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
        batch = collect(model, env, device)
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
        normalized = torch.as_tensor((batch["costs"] / env.budgets).mean(0), device=device)
        if args.dual_controller == "pi":
            integral, lambdas, error, proportional = pi_update(integral, normalized, args.method, dual_ki,
                                                               args.dual_kp, args.constraint_target)
        else:
            lambdas = update_multipliers(lambdas, normalized, args.method, args.dual_lr, args.constraint_target)
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
        if args.dual_controller == "pi":
            metrics.update({"dual/ki": dual_ki, "dual/kp": args.dual_kp})
            for i in range(len(lambdas)):
                metrics[f"dual/error_{i}"] = float(error[i])
                metrics[f"dual/integral_{i}"] = float(integral[i])
                metrics[f"dual/proportional_{i}"] = float(proportional[i])
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
        # --checkpoint-steps: the first update whose env_steps reaches each listed value.
        at_listed_step = any((update - 1) * count < s <= update * count for s in checkpoint_steps)
        if update % args.checkpoint_every == 0 or update == args.updates or at_listed_step:
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "multipliers": lambdas, "config": config, "update": update, "env_steps": update * count}
            if args.dual_controller == "pi":
                state["dual_controller"] = {"type": "pi", "ki": dual_ki, "kp": args.dual_kp, "kd": 0.0,
                                            "integral": integral.clone(), "lambdas": lambdas.clone()}
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
    p.add_argument("--dual-controller", choices=("integral", "pi"), default="integral",
                   help="integral: lambda += dual_lr * error (default, unchanged). pi: see pi_update")
    p.add_argument("--dual-ki", type=float, default=None, help="PI integral gain (default: --dual-lr)")
    p.add_argument("--dual-kp", type=float, default=0.0, help="PI proportional gain on the signed error")
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
    p.add_argument("--checkpoint-steps", type=int, nargs="*", default=None,
                   help="also save at the first update reaching each of these env steps")
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
    if args.dual_controller == "integral" and (args.dual_ki is not None or args.dual_kp != 0):
        raise ValueError("--dual-ki/--dual-kp need --dual-controller pi")
    if not 0 < args.constraint_target <= 1:
        raise ValueError("constraint-target must be in (0, 1]")
    if args.env == "safety" and (args.budget_hazard is None or args.budget_vase is None):
        raise ValueError("--env safety requires explicit --budget-hazard and --budget-vase")
    if args.env == "humanoid" and args.device == "mps":
        raise ValueError("Use CPU for the small local humanoid pilot; CUDA is supported on a cluster")
    train(args)
