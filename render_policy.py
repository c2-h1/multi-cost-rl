"""Render a trained SafetyPointGoal2 policy to mp4, one full episode per layout.

    python render_policy.py runs/sg-pilot-v2-loose/aggregate_loose-s0     # run dir -> checkpoint.pt
    python render_policy.py runs/sg-pilot-v2-loose/*/ --layouts 0 1        # same layouts for every run
    python render_policy.py runs/X/checkpoint-002000000.pt --camera track

Layouts follow evaluate.py: layout k is the k-th episode that
`evaluate.py --environment-seed S --eval-num-envs C` would play. The default
(900000, chunk 10) is the E4/E10 held-out set, so a video shows an episode that
the held-out numbers include (same layout; actions differ unless --action-seed
matches the replay). The episode runs through the training adapter, so the
observation, cost and absorption semantics are exactly the trainer's.

Videos go to <run dir>/videos/ unless --output-dir is given, with a JSON line
of the episode's return and costs next to each video. Reading a checkpoint
does not touch a running job, but rendering uses a CPU core: use `nice`.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from train import ActorCritic


class _FixedLayout:
    """Stands in for the adapter's RNG so reset() uses one chosen layout seed."""

    def __init__(self, seed):
        self.seed = seed

    def integers(self, low, high, size):
        return np.full(size, self.seed, dtype=np.int64)


def layout_seed(environment_seed, chunk, index):
    """Layout seed of episode `index` in collect_chunked(seed=environment_seed, chunk=chunk).

    Each chunk builds SafetyGoalBatch(size, environment_seed + start), whose reset()
    draws rng.integers(0, 2**31 - 1, size); numpy draws these sequentially, so the
    i-th value does not depend on the chunk's size.
    """
    start = index - index % chunk
    rng = np.random.default_rng(environment_seed + start)
    return int(rng.integers(0, 2**31 - 1, index - start + 1)[-1])


def load(path):
    path = Path(path)
    checkpoint_path = path / "checkpoint.pt" if path.is_dir() else path
    # Only load your own trusted checkpoints: torch serialization can execute code.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**checkpoint["config"])
    if config.env != "safety":
        raise ValueError(f"{checkpoint_path}: render_policy.py supports --env safety runs, not {config.env}")
    return checkpoint_path, checkpoint, config


def make_render_env(config, seed, args):
    import safety_gymnasium
    from safety_adapter import SafetyGoalBatch

    batch = SafetyGoalBatch(1, 0, config.safety_id, (config.budget_hazard, config.budget_vase))
    batch.envs[0].close()
    batch.envs = [safety_gymnasium.make(config.safety_id, render_mode="rgb_array", width=args.size,
                                        height=args.size, camera_name=args.camera)]
    batch.rng = _FixedLayout(seed)
    return batch


def overlay(frame, lines, flash):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    height = max(12, image.width // 32)
    try:
        font = ImageFont.load_default(size=height)
    except TypeError:  # Pillow < 10.1 has one fixed-size default font
        font, height = ImageFont.load_default(), 11
    step = height + 4
    draw.rectangle([0, 0, image.width, step * len(lines) + 6], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((6, 3 + step * i), line, fill=(255, 255, 255), font=font)
    if flash:  # red border on steps that count toward a cost
        draw.rectangle([0, 0, image.width - 1, image.height - 1], outline=(230, 30, 30), width=6)
    return np.asarray(image)


def render_episode(model, config, label, seed, args):
    env = make_render_env(config, seed, args)
    obs = env.reset()
    assert env.layout_seeds[0] == seed
    frames, total_return, costs = [], 0.0, np.zeros(2)
    try:
        for t in range(env.horizon):
            with torch.no_grad():
                action, *_ = model.act(torch.as_tensor(obs), args.deterministic)
            obs, reward, cost = env.step(action.numpy())
            total_return += float(reward[0])
            costs += cost[0]
            if env.absorbed[0]:
                break
            if t % args.frame_skip == 0:
                x = costs / env.budgets
                lines = [f"{label}  step {t + 1}/{env.horizon}",
                         f"return {total_return:6.2f}   goals {int(env.goals[0])}",
                         f"hazard {costs[0]:4.0f}/{env.budgets[0]:.1f} (x={x[0]:.2f})   "
                         f"vase {costs[1]:4.0f}/{env.budgets[1]:.1f} (x={x[1]:.2f})"]
                frames.append(overlay(env.envs[0].render(), lines, cost[0].any()))
        goals = int(env.goals[0])
    finally:
        env.close()
    return frames, {"return": total_return, "goals": goals, "cost_hazard": float(costs[0]),
                    "cost_vase": float(costs[1]), "normalized_hazard": float(costs[0] / env.budgets[0]),
                    "normalized_vase": float(costs[1] / env.budgets[1])}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="run directories (uses checkpoint.pt) or checkpoint files")
    p.add_argument("--layouts", type=int, nargs="+", default=[0], help="layout indices, see above")
    p.add_argument("--environment-seed", type=int, default=900000, help="900000 = held-out layouts")
    p.add_argument("--eval-num-envs", type=int, default=10, help="chunk size that defines the layouts (held-out: 10)")
    p.add_argument("--action-seed", type=int, default=900001)
    p.add_argument("--deterministic", action="store_true", help="mean action instead of sampling (label the video)")
    p.add_argument("--camera", default="fixedfar", choices=("fixedfar", "fixednear", "track", "vision"))
    p.add_argument("--size", type=int, default=512, help="frame width and height in pixels")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-skip", type=int, default=1, help="keep every n-th frame (2 halves the video length)")
    p.add_argument("--output-dir", default=None, help="default: <run dir>/videos")
    args = p.parse_args()
    torch.set_num_threads(1)
    import imageio

    for path in args.paths:
        checkpoint_path, checkpoint, config = load(path)
        out = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "videos"
        out.mkdir(parents=True, exist_ok=True)
        probe = make_render_env(config, 0, args)
        model = ActorCritic(probe.obs_dim, probe.action_dim, len(probe.cost_names))
        probe.close()
        model.load_state_dict(checkpoint["model"])
        model.eval()
        steps = int(checkpoint.get("env_steps", 0))
        policy = "det" if args.deterministic else "stoch"
        for k in args.layouts:
            seed = layout_seed(args.environment_seed, args.eval_num_envs, k)
            torch.manual_seed(args.action_seed + k)
            label = f"{config.method} s{config.seed} @ {steps / 1e6:.2f}M ({policy})"
            frames, stats = render_episode(model, config, label, seed, args)
            name = (f"{checkpoint_path.parent.name}-{steps:09d}-env{args.environment_seed}"
                    f"-c{args.eval_num_envs}-layout{k}-{policy}-{args.camera}")
            imageio.mimsave(out / f"{name}.mp4", frames, fps=args.fps, macro_block_size=1)
            record = {"video": str(out / f"{name}.mp4"), "checkpoint": str(checkpoint_path), "env_steps": steps,
                      "method": config.method, "seed": config.seed, "layout_index": k, "layout_seed": seed,
                      "environment_seed": args.environment_seed, "eval_num_envs": args.eval_num_envs,
                      "action_seed": args.action_seed + k, "deterministic": args.deterministic,
                      "camera": args.camera, **stats}
            (out / f"{name}.json").write_text(json.dumps(record, indent=2))
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
