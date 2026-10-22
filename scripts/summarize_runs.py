"""Rebuild experiment tables from saved configs and run outputs. Never trains.

    python scripts/summarize_runs.py                     # experiments + every train.py run under runs/
    python scripts/summarize_runs.py --root runs/sg-pilot-v1 --output reports/sg-pilot-v1/runs.md

A run directory is any folder with train.py's config.json. Status: `completed`
when summary.json exists, otherwise `incomplete` (still running, crashed or
stopped; check its log). Two safety readings are reported side by side:

- native: the constraint the method was trained on (multi: every w_i C_i <= w_i d_i;
  aggregate_loose: sum w_i C_i <= sum w_i d_i; aggregate_conservative:
  sum w_i C_i <= min_i w_i d_i; unconstrained: none), times constraint_target.
- both<=1: every mean cost within its own budget d_i, the common per-cost metric.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def native_feasible(method, costs, budgets, weights, target=1.0):
    """Mean costs C_i against the method's own trained constraint; None if unconstrained."""
    wc = [w * c for w, c in zip(weights, costs)]
    wd = [target * w * d for w, d in zip(weights, budgets)]
    if method == "multi":
        return all(a <= b for a, b in zip(wc, wd))
    if method == "aggregate_loose":
        return sum(wc) <= sum(wd)
    if method == "aggregate_conservative":
        return sum(wc) <= min(wd)
    return None


def read_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a run killed mid-write
                pass
    return rows


def summarize_run(run):
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
    rows = read_jsonl(run / "metrics.jsonl")
    names, budgets = config["cost_names"], config["budgets"]
    weights = [config["cost_weight_values"][n] for n in names] if config.get("cost_weight_values") else [1 / d for d in budgets]
    git = config.get("git") or {}
    record = {"run": str(run.relative_to(ROOT) if run.is_relative_to(ROOT) else run),
              "experiment": config.get("experiment_id") or config.get("group"),
              "method": config["method"], "seed": config["seed"],
              "status": "completed" if summary else "incomplete",
              "env_steps": rows[-1]["env_steps"] if rows else 0,
              "updates": f"{rows[-1]['update'] if rows else 0}/{config['updates']}",
              "dual_lr": config.get("dual_lr"), "cost_weights": config.get("cost_weights", "budget"),
              "commit": (git.get("commit") or "unknown")[:7] + ("+dirty" if git.get("dirty") else ""),
              "lambda_final": [round(v, 3) for k, v in sorted(rows[-1].items()) if k.startswith("dual/")] if rows else []}
    lambdas = {k: [r[k] for r in rows if k in r] for k in (rows[-1] if rows else {}) if k.startswith("dual/")}
    if lambdas and min(map(len, lambdas.values())) >= 2:
        # change over the last quarter of logged updates: large => multipliers not settled
        record["lambda_change_last_quarter"] = [round(v[-1] - v[-max(2, len(v) // 4)], 3) for v in lambdas.values()]
    if summary:
        costs = [summary[f"final/cost_{n}"] for n in names]
        record.update({"return": round(summary["final/return_mean"], 3),
                       **{f"x_{n}": round(summary[f"final/normalized_{n}"], 3) for n in names},
                       "native_feasible": native_feasible(config["method"], costs, budgets, weights,
                                                          config.get("constraint_target", 1.0)),
                       "both<=1": summary["final/all_mean_costs_feasible"],
                       "episode_pass_rate": round(summary["final/joint_episode_budget_pass_rate"], 3),
                       "hours": round(summary["elapsed_seconds"] / 3600, 2)})
    heldout = run / "heldout.json"
    if heldout.exists():
        h = json.loads(heldout.read_text())
        record.update({"heldout_return": round(h["replay/return_mean"], 3),
                       **{f"heldout_x_{n}": round(h[f"replay/normalized_{n}"], 3) for n in names},
                       "heldout_both<=1": h["replay/all_mean_costs_feasible"]})
    return record


def experiments_table(config_dir):
    lines = ["| ID | name | status | code | file |", "|---|---|---|---|---|"]
    configs = [(json.loads(path.read_text()), path) for path in config_dir.rglob("*.json")]
    for c, path in sorted([(c, p) for c, p in configs if "id" in c], key=lambda x: (len(x[0]["id"]), x[0]["id"])):
        code = c.get("code", {})
        lines.append(f"| {c['id']} | {c['name']} | {c.get('status', '?')} | "
                     f"{code.get('branch', '')} {code.get('commit', '')} | {path.relative_to(ROOT)} |")
    return "\n".join(lines)


def markdown(records):
    if not records:
        return "_no train.py runs found_"
    keys = list(dict.fromkeys(k for r in records for k in r))
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in records:
        lines.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT / "runs"))
    p.add_argument("--configs", default=str(ROOT / "configs"))
    p.add_argument("--output", default=None, help="write markdown here instead of stdout")
    p.add_argument("--json", action="store_true", help="print run records as JSON")
    args = p.parse_args()
    runs = sorted(c.parent for c in Path(args.root).rglob("config.json") if (c.parent / "source").is_dir())
    records = [summarize_run(r.resolve()) for r in runs]
    if args.json:
        print(json.dumps(records, indent=2))
        return
    text = f"# Experiments (configs/)\n\n{experiments_table(Path(args.configs))}\n\n# Runs under {args.root}\n\n{markdown(records)}\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
