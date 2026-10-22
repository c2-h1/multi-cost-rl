"""Measure per-cost episode costs C_i^0 of the untrained policy pi_0.

pi_0 is built exactly as train.py builds its policy for the same --seed:
same global seeding, training env constructed first, then ActorCritic. It is
rolled out stochastically on the fixed calibration layouts used for the
previous budgets (evaluate.py --episodes 30 --environment-seed 500000 with
eval-num-envs 2; layouts depend on that chunk size, so keep it at 2).

By default action sampling continues from the torch RNG state right after
model construction, as train.py's update-0 evaluation does. With
--environment-seed 10000+seed and --episodes = train.py --eval-episodes, the
result therefore matches summary.json["initial"] of that training run.

    python calibrate_pi0.py --seed 0
    -> runs/calibration/pi0_SafetyPointGoal2-v0_0.json
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from train import ActorCritic, collect_chunked, make_env, parser as train_parser


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=("toy", "safety", "humanoid"), default="safety")
    p.add_argument("--safety-id", default="SafetyPointGoal2-v0")
    p.add_argument("--seed", type=int, default=0, help="train.py --seed whose pi_0 is measured")
    p.add_argument("--num-envs", type=int, default=4, help="train.py --num-envs (training env built before the actor)")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--environment-seed", type=int, default=500000)
    p.add_argument("--eval-num-envs", type=int, default=2)
    p.add_argument("--action-seed", type=int, default=None,
                   help="reseed torch before the rollout; default continues the RNG as train.py does")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    # train.py's own defaults; budgets do not affect pi_0 or its rollout.
    targs = train_parser().parse_args([
        "--env", args.env, "--safety-id", args.safety_id, "--seed", str(args.seed),
        "--num-envs", str(args.num_envs), "--eval-num-envs", str(args.eval_num_envs),
        "--budget-hazard", "1", "--budget-vase", "1", "--threads", str(args.threads), "--output", "unused"])
    device = torch.device("cpu")

    # Same order as train.train(): seeds, training env, then the model.
    torch.set_num_threads(targs.threads)
    random.seed(targs.seed)
    np.random.seed(targs.seed)
    torch.manual_seed(targs.seed)
    env = make_env(targs, targs.seed)
    targs.horizon = env.horizon
    model = ActorCritic(env.obs_dim, env.action_dim, len(env.cost_names)).to(device)
    cost_names, env_id = list(env.cost_names), getattr(env, "metadata", {}).get("env_id", args.env)
    env.close()
    actor_hash = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

    if args.action_seed is not None:
        torch.manual_seed(args.action_seed)
    with torch.no_grad():
        batch = collect_chunked(model, targs, args.environment_seed, args.episodes, device)
    costs = batch["costs"]
    record = {
        "env": args.env, "env_id": env_id, "seed": args.seed, "policy": "pi_0 (untrained, stochastic)",
        "episodes": int(len(costs)), "environment_seed": args.environment_seed,
        "eval_num_envs": args.eval_num_envs, "action_seed": args.action_seed,
        "horizon": int(targs.horizon), "cost_names": cost_names,
        "pi0_mean_costs": {n: float(costs[:, i].mean()) for i, n in enumerate(cost_names)},
        "pi0_std_costs": {n: float(costs[:, i].std()) for i, n in enumerate(cost_names)},
        "pi0_zero_cost_fraction": {n: float((costs[:, i] == 0).mean()) for i, n in enumerate(cost_names)},
        "return_mean": float(batch["returns"].mean()),
        "diagnostics": batch["diagnostics"],
        "layout_seeds": batch["layout_seeds"].tolist(),
        "episode_costs": costs.tolist(),
        "model_sha256": actor_hash,
    }
    name = args.safety_id if args.env == "safety" else args.env
    output = Path(args.output or f"runs/calibration/pi0_{name}_{args.seed}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("env_id", "seed", "episodes", "pi0_mean_costs",
                                              "pi0_std_costs", "pi0_zero_cost_fraction")}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
