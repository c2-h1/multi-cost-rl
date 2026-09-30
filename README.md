# Multi-cost safe RL

PPO-Lagrangian with separate cost critics and scalar or vector multipliers. The current work is a Safety-Gymnasium pilot comparing one aggregate budget with two separate budgets (hazard vs. vase). **See [PLAN.md](PLAN.md)** for the spec, measured throughput, and commands.

## Setup

```bash
./setup_env.sh                 # portable CPU install (recommended; simulator-bound)
# Titan XP CUDA option: TORCH_BACKEND=cu118 ./setup_env.sh
# Existing wrong-Python .venv: MCRL_RESET_VENV=1 ./setup_env.sh
source .venv/bin/activate
export MUJOCO_GL=egl OMP_NUM_THREADS=2
python -m unittest test_core   # core math tests
python validate_safety.py --video   # cost-signal validation -> runs/validation/
```

## Resource-safe runs

The paper baseline now reuses two MuJoCo environments per run and queues one
run per detected GPU. The old launcher created 30 environments per run and
started all six runs at once (180 environments plus 24 worker processes), which
could exhaust host RAM.

```bash
./run_paper_repro.sh --dry-run
./run_paper_repro.sh                         # detected GPUs, one job/GPU
./run_paper_repro.sh --device cpu            # safe local fallback, one job
sbatch slurm_titanxp.sh                       # titanxp partition, 6-job array
```

Titan XP is compute capability 6.1. Startup checks that the installed CUDA
wheel contains `sm_61` and executes a CUDA operation before any simulator is
allocated; use the CPU install/run if that check fails.

## Files

| File | Purpose |
|---|---|
| `train.py` | Trainer (`--env toy\|safety\|humanoid`, `--method unconstrained\|aggregate_loose\|aggregate_conservative\|multi`) |
| `safety_adapter.py` | Batched `SafetyPointGoal2-v0` with hazard/vase indicator costs |
| `run_pilot.py` | Parallel launcher for the method × seed matrix, with frozen settings |
| `evaluate.py` | Checkpoint evaluation on fixed layout seeds; per-episode `.npz`, cost quantiles |
| `validate_safety.py` | Installation, cost-signal, and video checks |
| `baseline_ppo.py` | Safety Gym paper baseline (PPO / PPO-Lagrangian, scalar cost, reference settings) |
| `run_paper_repro.py`, `run_paper_repro.sh` | Queue the 6 reproduction runs within detected resources |
| `slurm_titanxp.sh` | One-run-per-GPU Slurm array for the `titanxp` partition |
| `toy_env.py` | Custom 2-cost toy (fallback only, not Safety-Gymnasium) |
| `humanoid_*.py`, `test_humanoid_costs.py` | HumanoidBench H1 costs, for the semester extension (needs `external/humanoid-bench`) |
| `docs/` | Math background (Korean), literature review |

Run outputs go to `runs/`, which is git-ignored.
