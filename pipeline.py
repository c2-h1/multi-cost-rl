"""Unattended Stage 1 -> 3 pipeline (PLAN.md section 6), resumable.

    nohup python pipeline.py > runs/pipeline.out 2>&1 < /dev/null &

1. Unconstrained development run (seed 100, 500k steps) with the FROZEN settings.
2. Budget calibration: d_i = max(1, 0.5 x mean C_i) over 30 held-out layouts.
3. Short constrained pilots (aggregate_loose, multi; seed 100) as a sanity check.
   The one manual adjustment PLAN.md allows is NOT made automatically; the
   pilot numbers are recorded in runs/pipeline/status.json for review.
4. Waits for the paper-reproduction runs (baseline_ppo.py) to finish, so jobs
   do not compete for cores, then writes the paper comparison.
5. Stage 2: run_pilot.py, 3 methods x 3 seeds.
6. Stage 3: held-out evaluation, analyze_pilot.py, and a video (optional).
7. Packs everything into results.tgz.

Every step is skipped if its output exists, so rerunning resumes. Progress
is appended to runs/pipeline/pipeline.log and runs/pipeline/status.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from run_pilot import FROZEN, HORIZON, train_command

STATE = Path("runs/pipeline")
PY = sys.executable


def log(message):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with open(STATE / "pipeline.log", "a") as f:
        f.write(line + "\n")


def status(**updates):
    path = STATE / "status.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(updates)
    path.write_text(json.dumps(data, indent=2, default=float))
    return data


def run(command, log_path=None, check=True, json_output=False):
    """Run to completion, output to `log_path`. With json_output, only stdout goes
    there (the file must stay valid JSON) and stderr goes to <log_path>.stderr."""
    log("$ " + " ".join(map(str, command)))
    with open(log_path, "w") if log_path else open(os.devnull, "w") as out, \
            open(f"{log_path}.stderr", "w") if json_output else open(os.devnull, "w") as err:
        code = subprocess.run(list(map(str, command)), stdout=out,
                              stderr=err if json_output else subprocess.STDOUT if log_path else None).returncode
    if check and code:
        log(f"FAILED (exit {code}); see {log_path}")
        raise SystemExit(code)
    return code


def processes(pattern):
    found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [int(pid) for pid in found if int(pid) != os.getpid()]


def train_run(method, seed, transitions, eval_transitions, budgets, output, group, args):
    """Train one run unless complete; a stale or wrongly configured run is set aside."""
    output = Path(output)
    if (output / "summary.json").exists():
        log(f"complete: {output}")
        return
    if output.exists():
        config = json.loads((output / "config.json").read_text()) if (output / "config.json").exists() else {}
        for pid in processes(f"train.py.*--output {output}( |$)"):
            log(f"stopping stale process {pid} for {output}")
            subprocess.run(["kill", str(pid)])
        time.sleep(5)
        aside = output.with_name(f"{output.name}.aborted-{datetime.now():%m%d-%H%M%S}")
        log(f"moving incomplete run aside (num_envs={config.get('num_envs')}): {output} -> {aside}")
        output.rename(aside)
    per_update = FROZEN["num_envs"] * HORIZON
    updates = -(-transitions // per_update)
    eval_every = max(1, eval_transitions // per_update)
    command = train_command(method, seed, updates, eval_every, budgets, output, group,
                            "cpu", args.threads, args.wandb_mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    run(command, output.parent / f"{output.name}.log")


def calibrate(checkpoint, episodes):
    out = checkpoint.parent
    try:
        json.loads((out / "calibration.json").read_text())
        done = (out / "calibration.npz").exists()
    except (OSError, json.JSONDecodeError):
        done = False
    if not done:
        run([PY, "evaluate.py", checkpoint, "--episodes", episodes, "--environment-seed", 500000,
             "--output", out / "calibration.npz"], out / "calibration.json", json_output=True)
    metrics = json.loads((out / "calibration.json").read_text())
    costs = np.load(out / "calibration.npz")["costs"]
    means = costs.mean(0)
    budgets = [max(1.0, round(0.5 * float(m), 1)) for m in means]
    warnings = []
    goals = metrics.get("replay/goals_per_episode", 0.0)
    if goals < 1:
        warnings.append(f"unconstrained policy reaches only {goals:.2f} goals/episode: task may not be learned")
    for name, m, zero in zip(("hazard", "vase"), means, (costs == 0).mean(0)):
        if m < 2 or zero > 0.8:
            warnings.append(f"{name} cost is rarely incurred (mean {m:.2f}, zero in {zero:.0%} of episodes): "
                            "constraint may be inactive")
    return {"episodes": int(len(costs)), "mean_costs": means.tolist(),
            "p50_costs": np.percentile(costs, 50, axis=0).tolist(), "goals_per_episode": goals,
            "return_mean": metrics.get("replay/return_mean"), "budgets": budgets, "warnings": warnings}


def summarize_pilot(output):
    summary = json.loads((output / "summary.json").read_text())
    rows = [json.loads(l) for l in (output / "metrics.jsonl").read_text().splitlines() if l.strip()]
    lambdas = {k: [r[k] for r in rows if k in r] for k in rows[-1] if k.startswith("dual/")}
    return {"final_normalized": [summary.get("final/normalized_hazard"), summary.get("final/normalized_vase")],
            "final_return": summary.get("final/return_mean"), "multipliers": summary.get("multipliers"),
            "lambda_max": {k: max(v) for k, v in lambdas.items()},
            "lambda_last5": {k: v[-5:] for k, v in lambdas.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", default="sg-pilot-v1")
    p.add_argument("--dev-transitions", type=int, default=500_000)
    p.add_argument("--pilot-transitions", type=int, default=200_000)
    p.add_argument("--transitions", type=int, default=2_000_000)
    p.add_argument("--eval-every-transitions", type=int, default=100_000)
    p.add_argument("--calibration-episodes", type=int, default=30)
    p.add_argument("--heldout-episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-jobs", type=int, default=9)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    p.add_argument("--no-wait-for-repro", action="store_true")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    log(f"pipeline start: {vars(args)}")
    status(started=datetime.now().isoformat(), args=vars(args), stage="1: unconstrained development run")

    # Stage 1: unconstrained development run and budget calibration.
    dev = Path("runs/dev/unconstrained-s100")
    train_run("unconstrained", 100, args.dev_transitions, args.eval_every_transitions, (1, 1), dev, "dev", args)
    status(stage="1: budget calibration")
    calibration = calibrate(dev / "checkpoint.pt", args.calibration_episodes)
    budgets = calibration["budgets"]
    log(f"calibration: mean costs {np.round(calibration['mean_costs'], 2).tolist()}, "
        f"goals/episode {calibration['goals_per_episode']:.2f} -> budgets hazard={budgets[0]} vase={budgets[1]}")
    for warning in calibration["warnings"]:
        log(f"WARNING: {warning}")
    status(calibration=calibration, stage="1: constrained pilots")

    # Short constrained pilots (seed 100) in parallel: sanity check only.
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda m: train_run(m, 100, args.pilot_transitions, args.pilot_transitions // 2, budgets,
                                          Path(f"runs/dev/{m}-s100-pilot"), "dev", args),
                      ["aggregate_loose", "multi"]))
    pilots = {m: summarize_pilot(Path(f"runs/dev/{m}-s100-pilot")) for m in ["aggregate_loose", "multi"]}
    for m, s in pilots.items():
        log(f"pilot {m}: final normalized costs {s['final_normalized']}, multipliers {s['multipliers']}")
    status(pilots=pilots, budgets_frozen=budgets,
           note="Budgets frozen automatically from calibration; the one manual adjustment in PLAN.md was not made.")

    # Wait for the paper reproduction so the 9 main runs get the cores.
    if not args.no_wait_for_repro:
        status(stage="waiting for paper reproduction to finish")
        if processes("baseline_ppo.py"):
            log("waiting for baseline_ppo.py (paper reproduction) to finish")
        while processes("baseline_ppo.py"):
            time.sleep(60)
        if Path("runs/paper-repro").exists():
            run([PY, "compare_paper.py"], STATE / "compare_paper.log", check=False)

    # Stage 2: main runs.
    status(stage="2: main runs (3 methods x 3 seeds)")
    root = Path("runs") / args.group
    # A restart mid-stage leaves incomplete runs that run_pilot.py refuses to touch.
    for d in sorted(root.glob("*-s*")):
        if d.is_dir() and not (d / "summary.json").exists() and not processes(f"--output {d}( |$)"):
            aside = d.with_name(f"{d.name}.aborted-{datetime.now():%m%d-%H%M%S}")
            log(f"moving incomplete run aside: {d} -> {aside}")
            d.rename(aside)
    run([PY, "run_pilot.py", "--group", args.group, "--budget-hazard", budgets[0], "--budget-vase", budgets[1],
         "--transitions", args.transitions, "--eval-every-transitions", args.eval_every_transitions,
         "--seeds", *args.seeds, "--device", "cpu", "--max-jobs", args.max_jobs, "--threads", args.threads,
         "--wandb-mode", args.wandb_mode], STATE / "run_pilot.log")

    # Stage 3: held-out evaluation on the same 50 fresh layouts for every run.
    status(stage="3: held-out evaluation")
    pending = [d for d in sorted(root.glob("*-s[0-9]*")) if (d / "summary.json").exists() and not (d / "heldout.npz").exists()]

    def heldout(d):
        return run([PY, "evaluate.py", d / "checkpoint.pt", "--episodes", args.heldout_episodes,
                    "--environment-seed", 900000, "--action-seed", 900001, "--output", d / "heldout.npz"],
                   d / "heldout.json", check=False, json_output=True)

    with ThreadPoolExecutor(args.max_jobs) as pool:
        codes = list(pool.map(heldout, pending))
    if any(codes):
        log(f"WARNING: {sum(1 for c in codes if c)} held-out evaluations failed; analysis falls back to final evaluation")

    status(stage="3: analysis")
    run([PY, "analyze_pilot.py", root], STATE / "analysis.log")
    if not args.no_video:
        methods = [m for m in ("unconstrained", "aggregate_loose", "multi") if (root / f"{m}-s{args.seeds[0]}").exists()]
        video = root / "analysis" / f"methods-s{args.seeds[0]}.mp4"
        command = [PY, "render_policies.py", *[root / f"{m}-s{args.seeds[0]}" for m in methods], "--output", video]
        if run(command, STATE / "video.log", check=False):
            log("EGL rendering failed; retrying with MUJOCO_GL=osmesa")
            os.environ["MUJOCO_GL"] = "osmesa"
            if run(command, STATE / "video.log", check=False):
                log("WARNING: video rendering failed (see video.log); everything else is done")

    archive = Path("results.tgz")
    folders = [str(d) for d in (Path("runs/paper-repro"), Path("runs/dev"), root, STATE) if d.exists()]
    run(["tar", "czf", archive, "--exclude=checkpoint-*.pt", *folders], check=False)
    status(stage="DONE", finished=datetime.now().isoformat(), archive=str(archive.resolve()))
    log(f"DONE. Results: {root}/analysis/results.md; archive: {archive.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if not isinstance(error, SystemExit):
            import traceback
            log("CRASHED:\n" + traceback.format_exc())
        STATE.mkdir(parents=True, exist_ok=True)
        status(stage=f"FAILED ({type(error).__name__}); see runs/pipeline/pipeline.log")
        raise
