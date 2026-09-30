"""Evaluate a trusted local checkpoint using fresh complete episodes."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train import ActorCritic, collect, make_env, summarize_rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--environment-seed", type=int, default=None)
    p.add_argument("--action-seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--verify-saved", action="store_true", help="Check exact stored stochastic evaluation with original seeds/count")
    args = p.parse_args()
    torch.set_num_threads(2)
    # Only load your own trusted checkpoints: torch serialization can execute code.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**checkpoint["config"])
    if args.verify_saved:
        args.episodes = config.final_eval_episodes
    environment_seed = args.environment_seed if args.environment_seed is not None else 200000 + config.seed
    action_seed = args.action_seed if args.action_seed is not None else 300000 + config.seed
    env = make_env(config, environment_seed, args.episodes)
    model = ActorCritic(env.obs_dim, env.action_dim, len(env.cost_names))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.manual_seed(action_seed)
    with torch.no_grad():
        batch = collect(model, env, torch.device("cpu"), args.deterministic)
    metrics = summarize_rollout(batch, env, "replay")
    if args.verify_saved:
        if args.deterministic:
            raise ValueError("Saved verification checks the primary stochastic evaluation")
        saved = np.load(Path(args.checkpoint).parent / "evaluation.npz")
        np.testing.assert_allclose(batch["returns"], saved["returns"], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(batch["costs"], saved["costs"], atol=1e-5, rtol=1e-5)
        metrics["saved_evaluation_verified"] = True
    print(json.dumps(metrics, indent=2))
    env.close()


if __name__ == "__main__":
    main()
