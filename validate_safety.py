"""Hours 0-6 validation for the Safety-Gymnasium adapter.

Records versions, runs a random full episode through the adapter, and uses a
scripted steering controller to drive into a hazard and then a vase so both
cost indicators are seen positive and zero (random rollouts rarely touch them).
Writes a JSON record (and optionally an mp4) under --output.
"""
import argparse
import json
import os
import platform
from pathlib import Path

import sys
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # macOS has no EGL; MuJoCo's default (glfw) works there
import numpy as np

from safety_adapter import REQUIRED_FIELDS, SafetyGoalBatch, cost_indicators


def steer(task, target):
    """Point-robot action [forward, turn] that heads toward `target` (xy)."""
    ego = (np.r_[target[:2], 0] - task.agent.pos) @ task.agent.mat
    angle = np.arctan2(ego[1], ego[0])
    return np.array([1.0 if abs(angle) < 0.5 else 0.1, np.clip(3 * angle, -1, 1)])


def drive(env, target, steps, frames=None):
    rows = []
    for _ in range(steps):
        _, _, _, terminated, truncated, info = env.step(steer(env.unwrapped.task, target))
        rows.append({k: float(info[k]) for k in REQUIRED_FIELDS} | {"indicators": cost_indicators(info)})
        if frames is not None:
            frames.append(env.render())
        if terminated or truncated:
            break
    return rows


def versions():
    import gymnasium, mujoco, safety_gymnasium, torch
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpus": torch.cuda.device_count(), "mujoco": mujoco.__version__, "gymnasium": gymnasium.__version__,
            "safety_gymnasium": safety_gymnasium.__version__, "platform": platform.platform()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="SafetyPointGoal2-v0")
    p.add_argument("--output", default="runs/validation")
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    record = {"versions": versions()}

    # One full random episode through the adapter.
    batch = SafetyGoalBatch(2, seed=100, env_id=args.env_id, budgets=(1, 1))
    obs = batch.reset()
    rng = np.random.default_rng(0)
    costs = np.zeros((2, 2))
    for _ in range(batch.horizon):
        obs, reward, cost = batch.step(rng.uniform(-1, 1, (2, batch.action_dim)))
        costs += cost
    record["adapter_random_episode"] = {"horizon": batch.horizon, "obs_dim": batch.obs_dim,
                                        "final_elapsed_fraction": float(obs[0, -2]),
                                        "episode_costs": costs.tolist(), **batch.diagnostics()}
    batch.close()

    # Scripted contact with a hazard, then a vase.
    import safety_gymnasium
    env = safety_gymnasium.make(args.env_id, render_mode="rgb_array" if args.video else None,
                                width=256, height=256)
    env.reset(seed=100)
    task = env.unwrapped.task
    frames = [] if args.video else None
    hazard = min(task.hazards.pos, key=lambda x: np.linalg.norm(x[:2] - task.agent.pos[:2]))
    hazard_rows = drive(env, hazard, 250, frames)
    vase = min(task.vases.pos, key=lambda x: np.linalg.norm(x[:2] - task.agent.pos[:2]))
    vase_rows = drive(env, vase, 400, frames)
    rows = hazard_rows + vase_rows
    ind = np.array([r["indicators"] for r in rows])
    both_vase = [r for r in rows if r["cost_vases_contact"] > 0 and r["cost_vases_velocity"] > 0]
    checks = {"hazard_positive": bool(ind[:, 0].max() == 1), "hazard_zero": bool(ind[:, 0].min() == 0),
              "vase_positive": bool(ind[:, 1].max() == 1), "vase_zero": bool(ind[:, 1].min() == 0),
              "contact_and_velocity_steps": len(both_vase),
              "contact_and_velocity_gives_one": all(r["indicators"][1] == 1.0 for r in both_vase),
              "indicators_binary": bool(np.isin(ind, [0, 1]).all())}
    try:
        cost_indicators({"cost_hazards": 0.0})
        checks["missing_field_raises"] = False
    except KeyError:
        checks["missing_field_raises"] = True
    record["scripted"] = {"steps": len(rows), "hazard_steps": int(ind[:, 0].sum()), "vase_steps": int(ind[:, 1].sum()),
                          "sample_positive_rows": [r for r in rows if max(r["indicators"]) > 0][:5], "checks": checks}
    env.close()
    if frames:
        import imageio
        imageio.mimsave(out / "scripted-contact.mp4", frames, fps=30)
        record["video"] = str(out / "scripted-contact.mp4")
    (out / "validation.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    failed = [k for k, v in checks.items() if v is False]
    if failed or not checks["contact_and_velocity_steps"]:
        raise SystemExit(f"Validation checks failed: {failed or ['no contact+velocity step observed']}")
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
