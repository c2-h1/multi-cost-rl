"""Export static figures illustrating the feasible sets and learned behaviors."""
import os
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
import torch
from train import ActorCritic
from toy_env import TwoCostNavigation


def main():
    out = Path("results")
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    ax.add_patch(Polygon([[0, 0], [2, 0], [0, 2]], color="#ee9037", alpha=.18, label="Loose sum: x + y ≤ 2"))
    ax.add_patch(Rectangle((0, 0), 1, 1, color="#178d88", alpha=.35, label="Separate: x ≤ 1 and y ≤ 1"))
    ax.add_patch(Polygon([[0, 0], [1, 0], [0, 1]], color="#935fb0", alpha=.42, label="Conservative sum: x + y ≤ 1"))
    ax.plot([0, 2], [2, 0], color="#ee9037")
    ax.plot([1, 1, 0], [0, 1, 1], color="#178d88")
    ax.plot([0, 1], [1, 0], color="#935fb0")
    ax.scatter([1.6], [.2], color="#b32b3c", zorder=5)
    ax.annotate("Aggregate passes,\ncost 1 violates", (1.6, .2), xytext=(1.04, .58), arrowprops={"arrowstyle": "->"}, fontsize=10)
    ax.set(xlim=(0, 2.1), ylim=(0, 2.1), aspect="equal", xlabel="Expected cost 1 / budget 1", ylabel="Expected cost 2 / budget 2",
           title="Aggregation changes which policies are feasible")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(out / "feasible-sets.png", dpi=180)
    fig.savefig(out / "feasible-sets.pdf")
    torch.set_num_threads(2)
    methods = ["unconstrained", "aggregate_loose", "aggregate_conservative", "multi"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7))
    for ax, method in zip(axes, methods):
        path = Path("runs") / f"toy-comparison-v1-{method}-s0/checkpoint.pt"
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ActorCritic(7, 2, 2)
        model.load_state_dict(checkpoint["model"])
        env = TwoCostNavigation(12, seed=74000)
        obs = env.reset()
        trace = [env.pos.copy()]
        for _ in range(env.horizon):
            with torch.no_grad():
                action = model.act(torch.as_tensor(obs), deterministic=True)[0].numpy()
            obs, _, _ = env.step(action)
            trace.append(env.pos.copy())
        trace = np.stack(trace)
        ax.add_patch(Circle((0, 0), .34, color="#b32b3c", alpha=.2))
        for i in range(env.num_envs):
            ax.plot(trace[:, i, 0], trace[:, i, 1], alpha=.6, linewidth=1)
        ax.scatter(trace[0, :, 0], trace[0, :, 1], c="black", s=12, label="Start")
        ax.scatter(env.goal[:, 0], env.goal[:, 1], c="#178d88", marker="*", s=30, label="Goal")
        ax.set(title=method, xlim=(-1.3, 1.3), ylim=(-.9, .9), aspect="equal")
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle("Seed 0, deterministic policy trajectories: qualitative illustration, not safety evaluation")
    fig.tight_layout()
    fig.savefig(out / "toy-trajectories.png", dpi=180)
    fig.savefig(out / "toy-trajectories.pdf")


if __name__ == "__main__":
    main()
