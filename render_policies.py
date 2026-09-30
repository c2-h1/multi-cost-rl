"""Side-by-side video of trained policies on the same SafetyPointGoal2 layout.

Each panel is one checkpoint (e.g. the three methods for one seed), all on the
same layout, labeled with the method and its running hazard/vase counts vs
budgets. Deterministic (mean) actions by default: videos are for illustration;
reported numbers come from stochastic evaluation (evaluate.py).

    python render_policies.py runs/sg-pilot-v1/{unconstrained,aggregate_loose,multi}-s0 \\
        --output runs/sg-pilot-v1/analysis/methods-s0.mp4
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless; use MUJOCO_GL=osmesa without a GPU
import numpy as np
import torch

from analyze_pilot import LABEL
from safety_adapter import SafetyGoalBatch
from train import ActorCritic


def rollout(run_dir, layout_seed, deterministic, size, camera):
    checkpoint = torch.load(Path(run_dir) / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = SimpleNamespace(**checkpoint["config"])
    env = SafetyGoalBatch(1, layout_seed, config.safety_id, (config.budget_hazard, config.budget_vase),
                          render_mode="rgb_array", width=size, height=size, camera_name=camera)
    model = ActorCritic(env.obs_dim, env.action_dim, len(env.cost_names))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.manual_seed(layout_seed)
    obs = env.reset()
    frames, totals, info = [], np.zeros(2), []
    ret = 0.0
    with torch.no_grad():
        for _ in range(env.horizon):
            action, *_ = model.act(torch.as_tensor(obs), deterministic)
            obs, reward, cost = env.step(action.numpy())
            totals += cost[0]
            ret += float(reward[0])
            frames.append(env.envs[0].render())
            info.append((totals.copy(), ret))
    result = {"run": str(run_dir), "method": config.method, "seed": config.seed,
              "layout_seed": int(env.layout_seeds[0]), "return": ret, "costs": totals.tolist(),
              "budgets": env.budgets.tolist(), "goals": int(env.goals[0])}
    env.close()
    return frames, info, result


def label(frame, lines):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(frame)
    bar = Image.new("RGB", (image.width, 40), (255, 255, 255))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for i, (text, color) in enumerate(lines):
        draw.text((6, 3 + 18 * i), text, fill=color, font=font)
    canvas = Image.new("RGB", (image.width, image.height + 40))
    canvas.paste(bar, (0, 0))
    canvas.paste(image, (0, 40))
    return np.asarray(canvas)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="Run directories containing checkpoint.pt")
    p.add_argument("--output", required=True, help=".mp4 path")
    p.add_argument("--layout-seed", type=int, default=700000)
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of the mean")
    p.add_argument("--size", type=int, default=360)
    p.add_argument("--camera", default="fixedfar", help="fixedfar (whole arena), fixednear, track, vision")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    panels, results = [], []
    for run in args.runs:
        frames, info, result = rollout(run, args.layout_seed, not args.stochastic, args.size, args.camera)
        budgets = result["budgets"]
        name = f"{LABEL.get(result['method'], result['method'])}, seed {result['seed']}"
        panels.append([label(f, [(name, (0, 0, 0)),
                                 (f"return {r:5.1f} | hazard {c[0]:.0f}/{budgets[0]:g} | vase {c[1]:.0f}/{budgets[1]:g}",
                                  (200, 0, 0) if (c > np.asarray(budgets)).any() else (0, 110, 0))])
                       for f, (c, r) in zip(frames, info)])
        results.append(result)
        print(json.dumps(result), flush=True)
    video = [np.concatenate(step, axis=1) for step in zip(*panels)]
    import imageio
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, video, fps=args.fps, macro_block_size=8)
    out.with_suffix(".json").write_text(json.dumps({"deterministic": not args.stochastic, "panels": results}, indent=2))
    print(f"Wrote {out} ({len(video)} frames, {len(video) / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
