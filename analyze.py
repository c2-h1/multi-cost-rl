"""Produce results table and learning curves from completed runs, no selection."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


def mean_ci(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(student_t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="toy-comparison-v1")
    args = p.parse_args()
    files = sorted(Path("runs").glob(f"{args.group}-*/summary.json"))
    if not files:
        raise ValueError("No completed runs")
    groups = {}
    for file in files:
        summary = json.loads(file.read_text())
        groups.setdefault(summary["method"], []).append((file.parent, summary))
    output = Path("results")
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["eval/return_mean", "eval/normalized_hazard", "eval/normalized_effort"]
    titles = ["Task return (higher is better)", "Hazard / budget", "Effort / budget"]
    colors = {"unconstrained": "#777777", "aggregate_loose": "#ee9037", "aggregate_conservative": "#935fb0", "multi": "#178d88"}
    rows = []
    aggregate = {}
    for method, runs in groups.items():
        vals = {k: mean_ci([s[k] for _, s in runs]) for k in ["final/return_mean", "final/cost_hazard", "final/cost_effort", "final/success_rate", "final/joint_episode_budget_pass_rate"]}
        feasible = sum(s["final/all_mean_costs_feasible"] for _, s in runs)
        fmt = lambda key: f"{vals[key][0]:.3f} ± {vals[key][1]:.3f}"
        rows.append(f"| {method} | {len(runs)} | {fmt('final/return_mean')} | {fmt('final/cost_hazard')} | {fmt('final/cost_effort')} | {100*vals['final/success_rate'][0]:.1f}% | {feasible}/{len(runs)} |")
        aggregate[method] = {"num_seeds": len(runs), "metrics_mean_ci95": vals, "mean_feasible_seeds": feasible}
        histories = [[json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()] for directory, _ in runs]
        for ax, metric, title in zip(axes, metrics, titles):
            curves = [{row["env_steps"]: row[metric] for row in hist if metric in row} for hist in histories]
            steps = sorted(set.intersection(*(set(c) for c in curves)))
            data = np.array([[c[x] for x in steps] for c in curves])
            ax.plot(steps, data.mean(0), label=method, color=colors[method])
            ax.fill_between(steps, data.min(0), data.max(0), alpha=.12, color=colors[method])
            ax.set(title=title, xlabel="Training transitions")
            ax.grid(alpha=.2)
    axes[1].axhline(1, linestyle="--", color="black", linewidth=1)
    axes[2].axhline(1, linestyle="--", color="black", linewidth=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Toy two-cost navigation: seed mean, shading = min–max across seeds")
    fig.tight_layout()
    fig.savefig(output / f"{args.group}-learning-curves.png", dpi=170)
    fig.savefig(output / f"{args.group}-learning-curves.pdf")
    report = ["# Completed toy training results", "", f"Group: `{args.group}`. This is a custom navigation CMDP, **not HumanoidBench**.", "",
              "All policies use the same actor, three critic heads and training budget. Columns show final stochastic-policy held-out evaluation, with mean ± Student-t 95% confidence interval **across training seeds**, not a claim of statistical significance. Three seeds give imprecise intervals. Episode samples are not independent training replicates.", "",
              "Raw expected-episode budgets: hazard ≤ 1.92 occupied steps, effort ≤ 7.68 (sum of mean squared action). 512 held-out episodes per trained policy. Feasible seeds means both point estimates satisfy their budgets; it is not a high-confidence safety certificate.", "",
              "| Method | Seeds | Return | Hazard | Effort | Goal success | Mean-feasible seeds |",
              "|---|---:|---:|---:|---:|---:|---:|", *rows, "",
              "The scalar controls impose different feasible sets: aggregate_loose uses hazard/1.92 + effort/7.68 ≤ 2; aggregate_conservative uses the same sum ≤ 1. Multi uses both individual ratios ≤ 1. Return differences alone cannot identify an optimization improvement at matched safety.", "",
              f"![Learning curves]({args.group}-learning-curves.png)", "", "## Run records", ""]
    for method, runs in groups.items():
        for directory, summary in runs:
            url = summary["wandb"].get("url")
            report.append(f"- `{directory}`: {summary['env_steps']:,} training transitions, {summary['elapsed_seconds']:.1f} training/evaluation seconds. " + (f"[W&B]({url})" if url else "W&B offline/disabled."))
    report += ["", "Each directory contains configuration and source hashes, exact source snapshot, JSONL learning curves, final checkpoint, held-out episode returns/costs, and summary. Checkpoints are final-budget policies, never selected on test return."]
    (output / f"{args.group}.md").write_text("\n".join(report) + "\n")
    (output / f"{args.group}.json").write_text(json.dumps(aggregate, indent=2))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
