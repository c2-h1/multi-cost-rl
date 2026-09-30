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

## Files

| File | Purpose |
|---|---|
| `train.py` | Trainer (`--env toy\|safety\|humanoid`, `--method unconstrained\|aggregate_loose\|aggregate_conservative\|multi`) |
| `safety_adapter.py` | Batched `SafetyPointGoal2-v0` with hazard/vase indicator costs |
| `run_pilot.py` | Parallel launcher for the method × seed matrix, with frozen settings |
| `evaluate.py` | Checkpoint evaluation on fixed layout seeds; per-episode `.npz`, cost quantiles |
| `validate_safety.py` | Installation, cost-signal, and video checks |
| `baseline_ppo.py` | Safety Gym paper baseline (PPO / PPO-Lagrangian, scalar cost, reference settings) |
| `run_paper_repro.sh`, `compare_paper.py` | Launch the 6 reproduction runs; compare to paper Fig. 7 |
| `toy_env.py` | Custom 2-cost toy (fallback only, not Safety-Gymnasium) |
| `humanoid_*.py`, `test_humanoid_costs.py` | HumanoidBench H1 costs, for the semester extension (needs `external/humanoid-bench`) |
| `docs/` | Math background (Korean), literature review |

Run outputs go to `runs/`, which is git-ignored.
