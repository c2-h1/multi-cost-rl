# Multi-cost safe RL

PPO-Lagrangian with separate cost critics and scalar or vector multipliers. The current work is a Safety-Gymnasium pilot comparing one aggregate budget with two separate budgets (hazard vs. vase). **See [PLAN.md](PLAN.md)** for the spec, measured throughput, and commands.

## Setup

```bash
./setup_env.sh                 # uv + Python 3.10 .venv, Safety-Gymnasium 1.0.0, CUDA torch; writes requirements-lock.txt
source .venv/bin/activate
export MUJOCO_GL=egl OMP_NUM_THREADS=2
python -m unittest test_core   # core math tests
python validate_safety.py --video   # cost-signal validation -> runs/validation/
```

### macOS

`./setup_env.sh` works on Apple Silicon and Intel Macs (CPU torch from PyPI). Skip the
`MUJOCO_GL=egl` export; MuJoCo's default works. Use `--device cpu` (the sim is
CPU-bound anyway). One paper-repro run needs ~10 GiB, so a 16 GB Mac runs one at a time.

### Memory

Each Safety-Gymnasium env holds ~0.2 GB and each process ~0.7 GB (torch), so one
`baseline_ppo.py` run (30 envs) needs ~9-11 GiB and a `train.py` job ~3.5-5 GiB.
Six paper-repro runs launched at once OOM-killed a 62 GB host. Now:

- every entry point refuses to start if its estimate does not fit in available memory
  (cgroup-aware, so it respects SLURM `--mem`) and aborts itself if free memory drops
  below 1.5 GiB (`memguard.py`);
- `run_paper_repro.sh` and `run_pilot.py` queue jobs and run only as many as fit in
  memory, CPU cores and GPUs (override with `MAX_JOBS=` / `--max-jobs`);
- `train.py` evaluates in chunks of `--eval-chunk` envs instead of keeping
  `--eval-episodes` simulators alive for the whole run.

On the `titanxp` SLURM partition (2 GPUs, 6 cores per node), run the paper baseline as
an array job with per-task memory limits:

```bash
mkdir -p runs/paper-repro && sbatch slurm/paper_repro.sbatch
```

## Files

| File | Purpose |
|---|---|
| `train.py` | Trainer (`--env toy\|safety\|humanoid`, `--method unconstrained\|aggregate_loose\|aggregate_conservative\|multi`) |
| `safety_adapter.py` | Batched `SafetyPointGoal2-v0` with hazard/vase indicator costs |
| `run_pilot.py` | Parallel launcher for the method × seed matrix, with frozen settings |
| `evaluate.py` | Checkpoint evaluation on fixed layout seeds; per-episode `.npz`, cost quantiles |
| `pipeline.py` | Unattended Stages 1–3: calibration, pilots, main runs, held-out eval, analysis, video |
| `analyze_pilot.py` | Stage 3 figures (curves, multipliers, cost space) and per-seed tables |
| `render_policies.py` | Side-by-side video of trained policies on one layout |
| `validate_safety.py` | Installation, cost-signal, and video checks |
| `baseline_ppo.py` | Safety Gym paper baseline (PPO / PPO-Lagrangian, scalar cost, reference settings) |
| `run_paper_repro.sh`, `slurm/paper_repro.sbatch`, `compare_paper.py` | Launch the 6 reproduction runs (local queue or SLURM array); compare to paper Fig. 7 |
| `memguard.py` | Memory preflight check and watchdog used by all launchers |
| `toy_env.py` | Custom 2-cost toy (fallback only, not Safety-Gymnasium) |
| `humanoid_*.py`, `test_humanoid_costs.py` | HumanoidBench H1 costs, for the semester extension (needs `external/humanoid-bench`) |
| `docs/` | Math background (Korean), literature review |

Run outputs go to `runs/`, which is git-ignored.
