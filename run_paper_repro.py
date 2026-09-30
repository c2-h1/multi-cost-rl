"""Resource-bounded launcher for the six Safety Gym paper reproductions.

The launcher keeps one training process per GPU by default (or one process in
CPU mode) and queues the remaining runs.  Each trainer reuses a two-environment
pool, so concurrency is explicit instead of allocating 180 MuJoCo environments
at once.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def resource_slots(device, requested_gpus, jobs_per_gpu, max_parallel):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        visible = torch.cuda.device_count()
        gpus = requested_gpus if requested_gpus is not None else list(range(visible))
        if not gpus or any(g < 0 or g >= visible for g in gpus):
            raise ValueError(f"GPU indices {gpus} invalid for {visible} visible GPU(s)")
        if jobs_per_gpu < 1:
            raise ValueError("jobs-per-gpu must be positive")
        return device, [gpu for gpu in gpus for _ in range(jobs_per_gpu)]
    parallel = max_parallel or 1
    if parallel < 1:
        raise ValueError("max-parallel must be positive")
    return device, [None] * parallel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", default="paper-repro-safe")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--algos", nargs="+", choices=("ppo", "ppo_lagrangian"),
                   default=["ppo", "ppo_lagrangian"])
    p.add_argument("--total-steps", type=int, default=10_000_000)
    p.add_argument("--steps-per-epoch", type=int, default=30_000)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="visible GPU indices; default: all visible GPUs")
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--max-parallel", type=int, default=None,
                   help="CPU mode only; default 1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not 1 <= args.workers <= args.num_envs:
        raise ValueError("workers must be between 1 and num-envs")
    device, slots = resource_slots(args.device, args.gpus, args.jobs_per_gpu, args.max_parallel)
    root = Path("runs") / args.group
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in args.seeds:
        for algo in args.algos:
            output = root / f"{algo}-s{seed}"
            if (output / "summary.json").exists():
                print(f"Already complete: {output}", flush=True)
                continue
            if output.exists():
                raise RuntimeError(f"Incomplete run exists: {output}; inspect it or choose a new --group")
            command = [sys.executable, "baseline_ppo.py", "--algo", algo, "--seed", str(seed),
                       "--total-steps", str(args.total_steps), "--steps-per-epoch", str(args.steps_per_epoch),
                       "--num-envs", str(args.num_envs), "--workers", str(args.workers),
                       "--threads", str(args.threads), "--device", device, "--output", str(output)]
            jobs.append((output, command))

    manifest = {"group": args.group, "device": device, "slots": slots,
                "jobs_per_gpu": args.jobs_per_gpu, "num_envs_per_job": args.num_envs,
                "workers_per_job": args.workers, "threads_per_job": args.threads,
                "total_steps_per_run": args.total_steps, "steps_per_epoch": args.steps_per_epoch,
                "jobs": [str(output) for output, _ in jobs]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        for _, command in jobs:
            print(" ".join(command))
        return

    pending = list(jobs)
    running = {}
    failed = []
    try:
        while pending or running:
            for slot_index in [i for i in range(len(slots)) if i not in running]:
                if not pending:
                    break
                output, command = pending.pop(0)
                env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
                           MKL_NUM_THREADS=str(args.threads), MUJOCO_GL="egl")
                if device == "cuda":
                    env["CUDA_VISIBLE_DEVICES"] = str(slots[slot_index])
                log = (root / f"{output.name}.log").open("w")
                label = f"gpu {slots[slot_index]}" if device == "cuda" else "cpu"
                print(f"[{label}] start {output}", flush=True)
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                running[slot_index] = (output, proc, log, label)
            time.sleep(2)
            for slot_index, (output, proc, log, label) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[slot_index]
                status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                print(f"[{label}] {status} {output}", flush=True)
                if proc.returncode:
                    failed.append(str(output))
    except BaseException:
        for _, proc, log, _ in running.values():
            proc.terminate()
            log.close()
        raise
    if failed:
        raise SystemExit(f"Failed runs (see logs): {failed}")


if __name__ == "__main__":
    main()
