"""Training curves and held-out scatter for Safety Gym runs. Never trains.

    python scripts/plot_runs.py runs/spg-v2-aggregate-loose runs/spg-v2-multi --output reports/spg-v2

For each group: <method>-curves.png (rows: normalized cost x_i, multipliers, return;
one column per seed). For all runs with heldout.npz: heldout.png (x_hazard vs x_vase
with standard errors, the multi square and the aggregate_loose triangle drawn in).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COST = {"hazard": ("#2a78d6", "-"), "vase": ("#eb6834", "--")}  # identity = color + line style
METHOD = {"aggregate_loose": ("#2a78d6", "o"), "multi": ("#eb6834", "s")}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e4e3df"


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def smooth(values, window):
    values = np.asarray(values, float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid"), window // 2


def load(run):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return [r for r in rows if "train/return_mean" in r], json.loads((run / "config.json").read_text())


def curves(group, output, window):
    runs = sorted(p for p in group.iterdir() if (p / "metrics.jsonl").exists())
    fig, axes = plt.subplots(3, len(runs), figsize=(3.6 * len(runs), 7.2), sharex=True, sharey="row", squeeze=False)
    for col, run in enumerate(runs):
        rows, config = load(run)
        steps = np.array([r["env_steps"] for r in rows]) / 1e6
        ax = axes[0, col]
        for name, (color, ls) in COST.items():
            y, off = smooth([r[f"train/normalized_{name}"] for r in rows], window)
            ax.plot(steps[off:off + len(y)], y, color=color, ls=ls, lw=2, label=f"x_{name}")
        ax.axhline(1, color=INK, lw=1, ls=":")
        ax.set_title(f"{config['method']} seed {config['seed']}", fontsize=10, color=INK)
        ax = axes[1, col]
        keys = sorted(k for k in rows[0] if k.startswith("dual/lambda_"))
        names = list(COST) if len(keys) == 2 else ["shared"]
        for key, name in zip(keys, names):
            color, ls = COST.get(name, ("#52514e", "-"))
            ax.plot(steps, [r[key] for r in rows], color=color, ls=ls, lw=2, label=f"λ_{name}")
        ax = axes[2, col]
        y, off = smooth([r["train/return_mean"] for r in rows], window)
        ax.plot(steps[off:off + len(y)], y, color="#52514e", lw=2)
        ax.set_xlabel("env steps (M)", fontsize=9, color=INK)
        for ax in axes[:, col]:
            style(ax)
    axes[0, 0].set_ylabel(f"x_i = C_i / d_i\n(training, {window}-update mean)", fontsize=9)
    axes[1, 0].set_ylabel("multiplier λ", fontsize=9)
    axes[2, 0].set_ylabel(f"return\n(training, {window}-update mean)", fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[1, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = output / f"{load(runs[0])[1]['method']}-curves.png"
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def heldout(groups, output, budgets):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.fill_between([0, 1], 0, 1, color="#eb6834", alpha=0.08, lw=0)
    ax.plot([0, 1, 1], [1, 1, 0], color="#eb6834", lw=1, ls="--")
    ax.plot([0, 2], [2, 0], color="#2a78d6", lw=1, ls="--")
    ax.text(0.04, 0.06, "multi feasible\n(both ≤ 1)", fontsize=8, color="#52514e")
    ax.text(1.25, 0.82, "aggregate_loose\nboundary: sum = 2", fontsize=8, color="#52514e", rotation=-40)
    seen = set()
    for group in groups:
        for run in sorted(p for p in group.iterdir() if (p / "heldout.npz").exists()):
            config = json.loads((run / "config.json").read_text())
            x = np.load(run / "heldout.npz")["costs"] / np.asarray(budgets)
            mean, se = x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x))
            color, marker = METHOD[config["method"]]
            label = config["method"] if config["method"] not in seen else None
            seen.add(config["method"])
            ax.errorbar(*mean, xerr=se[0], yerr=se[1], fmt=marker, color=color, ms=8, lw=1.2, capsize=3,
                        mec="#fcfcfb", mew=1.5, label=label)
            ax.annotate(f"s{config['seed']}", mean, xytext=(6, 5), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x_hazard (held-out, 50 episodes)", fontsize=9)
    ax.set_ylabel("x_vase (held-out, 50 episodes)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    path = output / "heldout.png"
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("groups", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=int, default=50, help="moving-average window in updates")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    budgets = None
    for group in args.groups:
        print(curves(group, args.output, args.window))
        if budgets is None:
            config = load(next(p for p in sorted(group.iterdir()) if (p / "metrics.jsonl").exists()))[1]
            budgets = [config["budget_hazard"], config["budget_vase"]]
    print(heldout(args.groups, args.output, budgets))


if __name__ == "__main__":
    main()
