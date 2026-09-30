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

import torch

from train import METHODS

# Plan section 2 "Initial settings"; change only via a new --group.
FROZEN = {"env": "safety", "safety_id": "SafetyPointGoal2-v0", "num_envs": 4,
          "epochs": 6, "minibatch_size": 512, "lr": 3e-4, "clip": 0.2,
          "gae_lambda": 0.95, "target_kl": 0.03, "value_coef": 1.0,
          "entropy_coef": 0.0, "lambda_init": 0.1, "dual_lr": 0.02,
          "constraint_target": 1.0, "eval_episodes": 10, "final_eval_episodes": 50,
          "eval_num_envs": 2}


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
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="visible GPU indices; default: detect all visible GPUs")
    p.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="keep at 1 on 12 GB Titan XP cards")
    p.add_argument("--max-parallel", type=int, default=None,
                   help="CPU mode only; default 1")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--threads", type=int, default=2, help="Torch/OMP threads per job")
    p.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        visible = torch.cuda.device_count()
        gpus = args.gpus if args.gpus is not None else list(range(visible))
        if not gpus or any(g < 0 or g >= visible for g in gpus):
            raise ValueError(f"GPU indices {gpus} invalid for {visible} visible GPU(s)")
        if args.jobs_per_gpu < 1:
            raise ValueError("jobs-per-gpu must be positive")
        slots = [gpu for gpu in gpus for _ in range(args.jobs_per_gpu)]
    else:
        parallel = args.max_parallel or 1
        if parallel < 1:
            raise ValueError("max-parallel must be positive")
        slots = [None] * parallel
    print(f"resource plan: device={device}, concurrent_jobs={len(slots)}, slots={slots}", flush=True)

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
                       "--device", device, "--threads", str(args.threads),
                       "--wandb-mode", args.wandb_mode, "--group", args.group, "--output", str(output)]
            for key, value in FROZEN.items():
                command += ["--" + key.replace("_", "-"), str(value)]
            jobs.append((output, command))
    manifest = {"group": args.group, "frozen": FROZEN, "updates": updates,
                "transitions_per_run": updates * per_update,
                "resources": {"device": device, "slots": slots, "threads_per_job": args.threads},
                "budgets": {"hazard": args.budget_hazard, "vase": args.budget_vase},
                "jobs": [str(o) for o, _ in jobs]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        print(" ".join(jobs[0][1]) if jobs else "No jobs")
        return

    running, failed = {}, []
    while jobs or running:
        for slot in [s for s in range(len(slots)) if s not in running]:
            if not jobs:
                break
            output, command = jobs.pop(0)
            env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
                       MKL_NUM_THREADS=str(args.threads), MUJOCO_GL="egl")
            if device == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = str(slots[slot])
            log = open(root / f"{output.name}.log", "w")
            label = f"gpu {slots[slot]}" if device == "cuda" else "cpu"
            print(f"[{label}] start {output}", flush=True)
            running[slot] = (output, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log)
        time.sleep(10)
        for slot, (output, proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[slot]
            status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            label = f"gpu {slots[slot]}" if device == "cuda" else "cpu"
            print(f"[{label}] {status} {output}", flush=True)
            if proc.returncode:
                failed.append(str(output))
    if failed:
        raise SystemExit(f"Failed runs (see logs): {failed}")


if __name__ == "__main__":
    main()
