"""Stage 3 analysis of a run_pilot.py group: figures and per-seed tables.

Reads runs/<group>/<method>-s<seed>/{config.json, metrics.jsonl, summary.json}
and, when present, the held-out evaluation heldout.{npz,json} written by
evaluate.py (falls back to the end-of-training evaluation.npz otherwise).

Writes runs/<group>/analysis/:
    learning_curves.png  return, goals, normalized costs (budget = 1), sum (aggregate limit = 2)
    multipliers.png      Lagrange multiplier trajectories
    cost_space.png       held-out (C_h/d_h, C_v/d_v): "both <= 1" square vs "sum <= 2" line
    results.md/.csv      per-seed table and per-method mean +- std
    results.json         the same numbers, machine-readable
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["unconstrained", "aggregate_loose", "aggregate_conservative", "multi"]
LABEL = {"unconstrained": "Unconstrained", "aggregate_loose": "Aggregate (sum ≤ 2)",
         "aggregate_conservative": "Aggregate (sum ≤ 1)", "multi": "Separate (each ≤ 1)"}
COLOR = {"unconstrained": "#7f7f7f", "aggregate_loose": "#eb6834",
         "aggregate_conservative": "#b8860b", "multi": "#2a78d6"}


def smooth(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    head = np.cumsum(values[:window - 1]) / np.arange(1, window)  # no edge dip at the start
    return np.concatenate([head, np.convolve(values, kernel, mode="valid")])


def load_runs(root):
    runs = []
    for config_path in sorted(root.glob("*/config.json")):
        run = config_path.parent
        config = json.loads(config_path.read_text())
        rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
        summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
        heldout_json = None
        if (run / "heldout.json").exists():
            try:
                heldout_json = json.loads((run / "heldout.json").read_text())
            except json.JSONDecodeError:
                pass
        if (run / "heldout.npz").exists():
            evaluation, source = np.load(run / "heldout.npz"), "held-out"
        elif (run / "evaluation.npz").exists():
            evaluation, source = np.load(run / "evaluation.npz"), "final (training seeds)"
        else:
            evaluation, source = None, "none (run incomplete)"
        runs.append({"dir": run, "method": config["method"], "seed": config["seed"],
                     "budgets": np.asarray(config["budgets"], dtype=float), "cost_names": config["cost_names"],
                     "rows": rows, "summary": summary, "heldout_json": heldout_json,
                     "evaluation": evaluation, "eval_source": source})
    return sorted(runs, key=lambda r: (ORDER.index(r["method"]) if r["method"] in ORDER else 99, r["seed"]))


def series(run, key):
    rows = [r for r in run["rows"] if key in r]
    return np.array([r["env_steps"] for r in rows], dtype=float), np.array([r[key] for r in rows], dtype=float)


def plot_learning_curves(runs, out, window):
    names = runs[0]["cost_names"]
    panels = [("train/return_mean", "Episode return"), ("train/goals_per_episode", "Goals per episode")]
    panels += [(f"train/normalized_{n}", f"C_{n} / d_{n}  (budget = 1)") for n in names]
    panels += [("sum", "Sum of normalized costs  (aggregate limit = 2)"),
               ("train/joint_episode_budget_pass_rate", "Episodes within both budgets")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, panels):
        for method in dict.fromkeys(r["method"] for r in runs):
            curves = []
            for run in [r for r in runs if r["method"] == method]:
                if key == "sum":
                    x, a = series(run, f"train/normalized_{names[0]}")
                    _, b = series(run, f"train/normalized_{names[1]}")
                    y = a + b
                else:
                    x, y = series(run, key)
                if not len(x):
                    continue
                y = smooth(y, window)
                ax.plot(x / 1e6, y, color=COLOR.get(method), alpha=0.3, lw=0.8)
                curves.append((x, y))
            if curves:
                n = min(len(c[0]) for c in curves)
                mean = np.mean([c[1][:n] for c in curves], axis=0)
                ax.plot(curves[0][0][:n] / 1e6, mean, color=COLOR.get(method), lw=2,
                        label=f"{LABEL.get(method, method)} (n={len(curves)})")
        if key.startswith("train/normalized_"):
            ax.axhline(1, color="k", ls="--", lw=1)
        if key == "sum":
            ax.axhline(2, color="k", ls=":", lw=1, label="aggregate limit")
            ax.axhline(1, color="k", ls="--", lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Environment steps (M)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Training curves (thin: seeds, thick: mean; moving average over {window} updates)")
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_multipliers(runs, out):
    names = runs[0]["cost_names"]
    constrained = [m for m in dict.fromkeys(r["method"] for r in runs) if m != "unconstrained"]
    if not constrained:
        return
    fig, axes = plt.subplots(1, len(constrained), figsize=(6 * len(constrained), 4), squeeze=False,
                             constrained_layout=True)
    for ax, method in zip(axes[0], constrained):
        for run in [r for r in runs if r["method"] == method]:
            for i, style in enumerate(["-", "--"]):
                x, y = series(run, f"dual/lambda_{i}")
                if not len(x):
                    continue
                label = (names[i] if method == "multi" else "shared") + f", seed {run['seed']}"
                ax.plot(x / 1e6, y, ls=style, lw=1.2, label=label)
        ax.set_title(f"{LABEL.get(method, method)}: multipliers")
        ax.set_xlabel("Environment steps (M)")
        ax.set_ylabel("λ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.savefig(out / "multipliers.png", dpi=150)
    plt.close(fig)


def plot_cost_space(runs, out):
    names = runs[0]["cost_names"]
    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)
    top = 2.5
    for run in runs:
        if run["evaluation"] is None:
            continue
        x = run["evaluation"]["costs"] / run["budgets"]
        top = max(top, float(np.percentile(x, 95)) * 1.1, float(x.mean(0).max()) * 1.2)
    ax.fill_between([0, 1], 0, 1, color="#2a78d6", alpha=0.08, label="both ≤ 1 (separate budgets)")
    ax.plot([0, 2], [2, 0], "k:", lw=1.2, label="sum = 2 (aggregate budget)")
    ax.plot([0, 1, 1], [1, 1, 0], color="#2a78d6", lw=1)
    seen = set()
    for run in runs:
        if run["evaluation"] is None:
            continue
        x = run["evaluation"]["costs"] / run["budgets"]
        color = COLOR.get(run["method"])
        ax.scatter(x[:, 0], x[:, 1], s=6, alpha=0.15, color=color, lw=0)
        label = None if run["method"] in seen else LABEL.get(run["method"], run["method"])
        seen.add(run["method"])
        ax.scatter(*x.mean(0), s=90, color=color, edgecolor="k", zorder=5, label=label)
        ax.annotate(f"s{run['seed']}", x.mean(0), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel(f"C_{names[0]} / d_{names[0]}")
    ax.set_ylabel(f"C_{names[1]} / d_{names[1]}")
    sources = sorted({r["eval_source"] for r in runs})
    ax.set_title(f"Per-episode normalized costs (dots) and run means (circles)\nevaluation: {', '.join(sources)}",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(out / "cost_space.png", dpi=150)
    plt.close(fig)


def run_row(run):
    names = run["cost_names"]
    last = run["rows"][-1] if run["rows"] else {}
    row = {"method": run["method"], "seed": run["seed"], "steps": last.get("env_steps", 0),
           "evaluation": run["eval_source"]}
    if run["evaluation"] is not None:
        costs, returns = run["evaluation"]["costs"], run["evaluation"]["returns"]
        x = costs / run["budgets"]
        row.update({"episodes": len(returns), "return": float(returns.mean())})
        for i, n in enumerate(names):
            row[f"C_{n}/d_{n}"] = float(x[:, i].mean())
        means = x.mean(0)
        row["both_within"] = bool((means <= 1).all())
        row["sum_within_2"] = bool(means.sum() <= 2)
        # The motivating failure case: passes the aggregate budget while breaking one budget.
        row["hidden_violation"] = bool(means.sum() <= 2 and (means > 1).any())
        row["joint_episode_pass_rate"] = float((x <= 1).all(1).mean())
        for i, n in enumerate(names):
            row[f"exceed_rate_{n}"] = float((x[:, i] > 1).mean())
    goals = None
    if run["heldout_json"]:
        goals = run["heldout_json"].get("replay/goals_per_episode")
    elif run["summary"]:
        goals = run["summary"].get("final/goals_per_episode")
    row["goals_per_episode"] = goals
    for n in names:
        row[f"cumulative_train_cost_{n}"] = last.get(f"train/cumulative_cost_{n}")
    for i in range(len(names)):
        if f"dual/lambda_{i}" in last:
            row[f"final_lambda_{i}"] = last[f"dual/lambda_{i}"]
    return row


def fmt(value):
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3g}" if abs(value) < 1000 else f"{value:,.0f}"
    return str(value)


def write_tables(runs, out):
    rows = [run_row(r) for r in runs]
    names = runs[0]["cost_names"]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    shown = ["method", "seed", "steps", "return", "goals_per_episode"] + [f"C_{n}/d_{n}" for n in names] + \
            ["both_within", "hidden_violation", "joint_episode_pass_rate"] + [f"exceed_rate_{n}" for n in names] + \
            [f"cumulative_train_cost_{n}" for n in names]
    lines = [f"# Results: {out.parent.name}", "",
             f"Evaluation: {', '.join(sorted({r['evaluation'] for r in rows}))}. "
             "Normalized cost 1.0 = exactly on budget. `hidden_violation` = mean costs pass the "
             "aggregate budget (sum ≤ 2) but break a separate budget.", "",
             "## Per seed", "", "| " + " | ".join(shown) + " |", "|" + "---|" * len(shown)]
    lines += ["| " + " | ".join(fmt(r.get(k)) for k in shown) + " |" for r in rows]
    numeric = ["return", "goals_per_episode"] + [f"C_{n}/d_{n}" for n in names] + ["joint_episode_pass_rate"]
    lines += ["", "## Per method (mean ± std over seeds)", "",
              "| method | seeds | " + " | ".join(numeric) + " | seeds with both ≤ 1 |", "|" + "---|" * (len(numeric) + 3)]
    per_method = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        group = [r for r in rows if r["method"] == method and "return" in r]
        if not group:
            continue
        stats = {}
        cells = []
        for k in numeric:
            vals = [r[k] for r in group if r.get(k) is not None]
            stats[k] = (float(np.mean(vals)), float(np.std(vals))) if vals else None
            cells.append(f"{stats[k][0]:.3g} ± {stats[k][1]:.2g}" if vals else "–")
        within = sum(r["both_within"] for r in group)
        per_method[method] = {"seeds": len(group), "both_within_seeds": within, **stats}
        lines.append(f"| {LABEL.get(method, method)} | {len(group)} | " + " | ".join(cells) + f" | {within}/{len(group)} |")
    lines += ["", "With 3 seeds, report per-seed results; no significance claims."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps({"per_seed": rows, "per_method": per_method}, indent=2, default=float))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", help="Group directory, e.g. runs/sg-pilot-v1; analysis/ is written here")
    p.add_argument("extra", nargs="*", help="More group directories with the same budgets, analyzed together")
    p.add_argument("--smooth", type=int, default=20, help="Moving-average window in updates")
    args = p.parse_args()
    root = Path(args.root)
    runs = sorted(sum((load_runs(Path(r)) for r in [root, *args.extra]), []),
                  key=lambda r: (ORDER.index(r["method"]) if r["method"] in ORDER else 99, r["seed"]))
    if not runs:
        raise SystemExit(f"No runs with config.json under {root}")
    budgets = {tuple(r["budgets"]) for r in runs}
    if len(budgets) > 1:
        raise SystemExit(f"Runs use different budgets {sorted(budgets)}; analyze them separately")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    plot_learning_curves(runs, out, args.smooth)
    plot_multipliers(runs, out)
    plot_cost_space(runs, out)
    print(write_tables(runs, out))
    print(f"\nWrote {out}/learning_curves.png, multipliers.png, cost_space.png, results.md/.csv/.json")


if __name__ == "__main__":
    main()
