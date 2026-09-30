"""Launch the Safety-Gymnasium method x seed matrix in parallel across GPUs.

One job per GPU slot; a freed slot takes the next queued (seed, method) job,
ordered seed-major so complete three-method seed sets finish first. Frozen
settings live in FROZEN below; budgets are required so they are chosen
explicitly after the unconstrained development pilot.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import memguard
from train import METHODS

# Plan section 2 "Initial settings"; change only via a new --group.
FROZEN = {"env": "safety", "safety_id": "SafetyPointGoal2-v0", "num_envs": 4,
          "epochs": 6, "minibatch_size": 512, "lr": 3e-4, "clip": 0.2,
          "gae_lambda": 0.95, "target_kl": 0.03, "value_coef": 1.0,
          "entropy_coef": 0.0, "lambda_init": 0.1, "dual_lr": 0.02,
          "constraint_target": 1.0, "eval_episodes": 10, "final_eval_episodes": 50}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", required=True)
    p.add_argument("--budget-hazard", type=float, required=True)
    p.add_argument("--budget-vase", type=float, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--methods", choices=METHODS, nargs="+",
                   default=["unconstrained", "aggregate_loose", "multi"])
    p.add_argument("--transitions", type=int, default=500_000)
    p.add_argument("--eval-every-transitions", type=int, default=100_000)
    p.add_argument("--gpus", type=int, nargs="+", default=None, help="Default: all visible GPUs (none -> CPU)")
    p.add_argument("--jobs-per-gpu", type=int, default=3, help="Simulation is CPU-bound; 9 concurrent jobs measured at ~1450 transitions/s total")
    p.add_argument("--max-jobs", type=int, default=None,
                   help="Concurrent job cap. Default: what fits in available memory and CPU cores")
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--threads", type=int, default=2, help="Torch/OMP threads per job")
    p.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    horizon = 1000  # verified native SafetyPointGoal2-v0 limit; train.py re-checks
    per_update = FROZEN["num_envs"] * horizon
    updates = -(-args.transitions // per_update)
    eval_every = max(1, args.eval_every_transitions // per_update)
    root = Path("runs") / args.group
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in args.seeds:
        for method in args.methods:
            output = root / f"{method}-s{seed}"
            if (output / "summary.json").exists():
                print(f"Already complete: {output}", flush=True)
                continue
            if output.exists():
                raise RuntimeError(f"Incomplete run exists: {output}; inspect it or use a new --group")
            command = [sys.executable, "train.py", "--method", method, "--seed", str(seed),
                       "--updates", str(updates), "--eval-every", str(eval_every),
                       "--checkpoint-every", str(eval_every), "--log-every", "1",
                       "--budget-hazard", str(args.budget_hazard), "--budget-vase", str(args.budget_vase),
                       "--device", args.device, "--threads", str(args.threads),
                       "--wandb-mode", args.wandb_mode, "--group", args.group, "--output", str(output)]
            for key, value in FROZEN.items():
                command += ["--" + key.replace("_", "-"), str(value)]
            jobs.append((output, command))
    manifest = {"group": args.group, "frozen": FROZEN, "updates": updates,
                "transitions_per_run": updates * per_update,
                "budgets": {"hazard": args.budget_hazard, "vase": args.budget_vase},
                "jobs": [str(o) for o, _ in jobs]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if args.gpus is None:
        import torch
        args.gpus = list(range(torch.cuda.device_count())) if args.device == "cuda" else []
    if args.device == "cuda" and not args.gpus:
        raise SystemExit("No GPU visible; pass --device cpu")
    # Per job: train envs + one eval chunk (10) of ~0.2 GB each, plus torch/CUDA.
    per_job = memguard.estimate_gib(FROZEN["num_envs"] + 10, 1, args.device == "cuda")
    available = memguard.available_bytes()
    fit_memory = int((available / memguard.GIB - 4) // per_job) if available else len(jobs)
    fit_cpu = max(1, len(os.sched_getaffinity(0)) // args.threads)
    max_jobs = args.max_jobs or min(fit_memory, fit_cpu)
    if max_jobs < 1:
        raise SystemExit(f"Not enough memory for one job (~{per_job:.1f} GiB + 4 GiB reserve)")
    slots = [gpu for _ in range(args.jobs_per_gpu) for gpu in args.gpus] if args.gpus else [None] * max_jobs
    slots = slots[:max_jobs]
    print(f"{len(slots)} concurrent jobs (~{per_job:.1f} GiB each; memory fits {fit_memory}, cores fit {fit_cpu})",
          flush=True)
    if args.dry_run:
        print(" ".join(jobs[0][1]) if jobs else "No jobs")
        return

    running, failed = {}, []
    while jobs or running:
        for slot in [s for s in range(len(slots)) if s not in running]:
            if not jobs:
                break
            available = memguard.available_bytes()
            if available is not None and available / memguard.GIB < per_job + 4:
                break  # wait for memory to free up (other users, finishing evaluations)
            output, command = jobs.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if slots[slot] is None else str(slots[slot]),
                       OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                       MUJOCO_GL="egl")
            log = open(root / f"{output.name}.log", "w")
            print(f"[gpu {slots[slot]}] start {output}", flush=True)
            running[slot] = (output, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log)
        time.sleep(10)
        for slot, (output, proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[slot]
            status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            print(f"[gpu {slots[slot]}] {status} {output}", flush=True)
            if proc.returncode:
                failed.append(str(output))
    if failed:
        raise SystemExit(f"Failed runs (see logs): {failed}")


if __name__ == "__main__":
    main()
