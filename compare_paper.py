"""Compare paper-reproduction runs (baseline_ppo.py) with Safety Gym Fig. 7, PointGoal2.

Paper values are read off the end of the Fig. 7 PointGoal2 curves (3 seeds,
1e7 steps; docs/reference/safetygym-fig7-pointgoal2.png), so they are
approximate. Works on partially finished runs: metrics use whatever epochs exist,
and the report states the step count.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = {"ppo": {"EpRet": 22.5, "EpCost": 200.0, "CostRate": 0.20},
         "ppo_lagrangian": {"EpRet": 1.0, "EpCost": 30.0, "CostRate": 0.037}}
LABEL = {"ppo": "PPO", "ppo_lagrangian": "PPO-Lagrangian"}
COLOR = {"ppo": "#2a78d6", "ppo_lagrangian": "#eb6834"}
METRICS = [("EpRet", "Average episode return"), ("EpCost", "Average episode cost"),
           ("CostRate", "Cost rate (cumulative cost / steps)")]


def load(root):
    runs = {}
    for progress in sorted(Path(root).glob("*/progress.jsonl")):
        rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
        if rows:
            algo, seed = progress.parent.name.rsplit("-s", 1)
            runs.setdefault(algo, {})[int(seed)] = rows
    return runs


def characteristic(rows):
    # Paper footnote 3: return/cost averaged over the last five epochs; cost rate from the final epoch.
    last = rows[-5:]
    return {"EpRet": np.mean([r["EpRet"] for r in last]), "EpCost": np.mean([r["EpCost"] for r in last]),
            "CostRate": rows[-1]["CostRate"], "steps": rows[-1]["TotalEnvInteracts"],
            "Penalty": rows[-1]["Penalty"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="runs/paper-repro")
    args = p.parse_args()
    root = Path(args.root)
    runs = load(root)
    if not runs:
        raise SystemExit(f"No progress.jsonl under {root}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title) in zip(axes, METRICS):
        for algo in ("ppo", "ppo_lagrangian"):
            if algo not in runs:
                continue
            curves = runs[algo].values()
            for rows in curves:
                ax.plot([r["TotalEnvInteracts"] / 1e6 for r in rows], [r[key] for r in rows],
                        color=COLOR[algo], linewidth=1, alpha=0.35)
            n = min(len(rows) for rows in curves)
            mean = np.mean([[r[key] for r in rows[:n]] for rows in curves], axis=0)
            steps = [r["TotalEnvInteracts"] / 1e6 for r in next(iter(curves))[:n]]
            ax.plot(steps, mean, color=COLOR[algo], linewidth=2, label=f"{LABEL[algo]} (ours, {len(curves)} seeds)")
            ax.axhline(PAPER[algo][key], color=COLOR[algo], linestyle="--", linewidth=1.2,
                       label=f"{LABEL[algo]} paper, end of training")
        if key == "EpCost":
            ax.axhline(25, color="#555555", linestyle=":", linewidth=1.2, label="cost limit 25")
        if key == "CostRate":
            ax.axhline(0.025, color="#555555", linestyle=":", linewidth=1.2, label="limit rate 0.025")
        ax.set(title=title, xlabel="Environment steps (millions)", xlim=(0, 10))
        ax.grid(alpha=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("SafetyPointGoal2-v0: our PPO / PPO-Lagrangian vs. Safety Gym paper (Fig. 7)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(root / "paper-comparison.png", dpi=150)

    lines = ["# Paper reproduction: SafetyPointGoal2-v0", "",
             "Paper values are read off the end of Safety Gym Fig. 7 (approximate). Ours: return and cost "
             "averaged over the last 5 epochs, cost rate at the final epoch (paper footnote 3). "
             "Training-rollout (stochastic policy) metrics, as in the paper.", "",
             "| Algorithm | Seed | Steps | EpRet | EpCost | CostRate | Penalty |",
             "|---|---|---:|---:|---:|---:|---:|"]
    summary = {}
    for algo in ("ppo", "ppo_lagrangian"):
        if algo not in runs:
            continue
        stats = {seed: characteristic(rows) for seed, rows in sorted(runs[algo].items())}
        for seed, s in stats.items():
            lines.append(f"| {LABEL[algo]} | {seed} | {s['steps']:,} | {s['EpRet']:.2f} | {s['EpCost']:.1f} | "
                         f"{s['CostRate']:.3f} | {s['Penalty']:.2f} |")
        mean = {k: float(np.mean([s[k] for s in stats.values()])) for k in ("EpRet", "EpCost", "CostRate")}
        lines.append(f"| **{LABEL[algo]} mean** | | | **{mean['EpRet']:.2f}** | **{mean['EpCost']:.1f}** | "
                     f"**{mean['CostRate']:.3f}** | |")
        paper = PAPER[algo]
        lines.append(f"| {LABEL[algo]} paper | | 10,000,000 | ≈{paper['EpRet']} | ≈{paper['EpCost']:.0f} | "
                     f"≈{paper['CostRate']} | |")
        summary[algo] = {"seeds": {str(k): v for k, v in stats.items()}, "mean": mean, "paper": paper}
    lines += ["", "![Comparison](paper-comparison.png)", ""]
    (root / "paper-comparison.md").write_text("\n".join(lines))
    (root / "paper-comparison.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
