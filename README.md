# Multi-cost safe RL

**Question.** A robot often has several distinct ways to be unsafe. Standard safe RL
(e.g. Safety Gym) merges them into one scalar cost with one Lagrange multiplier. What
changes if each cost gets its **own budget and multiplier**? With one shared budget the
agent can trade one violation against another (all hazards, no vases); separate budgets
forbid that, but are a stricter specification and may be harder to optimize.

**Implemented.** PPO-Lagrangian with a shared actor, a critic with one reward head plus one
head per cost, and four constraint modes (`train.py --method`):

| method | constraint (x_i = C_i / d_i) | multipliers |
|---|---|---|
| `unconstrained` | none | 0 |
| `aggregate_loose` | x_1 + x_2 ≤ 2 (sum of budgets) | 1 |
| `aggregate_conservative` | x_1 + x_2 ≤ 1 (implies every x_i ≤ 1) | 1 |
| `multi` | every x_i ≤ 1 | one per cost |

`--cost-weights pi0` weights costs by the untrained policy's costs instead of 1/d_i
(experiment E8). Environments: a 2-cost toy (`toy_env.py`), Safety-Gymnasium
`SafetyPointGoal2-v0` with hazard and vase costs (`safety_adapter.py`), HumanoidBench H1
with 3 costs (`humanoid_*.py`). The Safety Gym paper baseline (`baseline_ppo.py`) is a
separate, deliberately faithful PPO / PPO-Lag implementation.

## How the experiments relate

```text
E1 toy comparison ──> E2 reproduce Safety Gym paper (1 cost) ──> E3 calibrate 2 budgets
   ──> E4 aggregate vs separate on SafetyPointGoal2 ──> E5 follow-ups (faster duals,
   strict aggregate) / E6 hparam search ──> E8 pi0 weights ──> E9 hierarchical constraints
E7 humanoid pilot (side branch, failed: robot never learned to walk)
```

Status, findings and evidence for each: **[docs/experiments/INDEX.md](docs/experiments/INDEX.md)**.
Research plan and history: [PLAN.md](PLAN.md). Agent conventions: [CLAUDE.md](CLAUDE.md).

## Layout

| path | purpose |
|---|---|
| `train.py` | multi-cost trainer (toy / safety / humanoid); writes `config.json` (resolved args, git commit, eval protocol), `metrics.jsonl`, checkpoints, `evaluation.npz`, `summary.json` |
| `evaluate.py` | re-evaluate a checkpoint on fixed layouts (held-out: `--environment-seed 900000 --action-seed 900001`) |
| `calibrate_pi0.py` | per-cost episode costs of the untrained policy (E8) |
| `run_pilot.py` | queue a method × seed matrix with frozen Safety Gym settings; resumable |
| `baseline_ppo.py`, `run_paper_repro.{py,sh}`, `compare_paper.py` | E2 paper reproduction and comparison plots |
| `slurm_*.sh` | titanxp cluster jobs (paper repro, calibration, main experiment) |
| `safety_adapter.py`, `toy_env.py`, `humanoid_adapter.py`, `humanoid_costs.py` | environments and cost definitions |
| `validate_safety.py`, `setup_env.sh`, `wandb_tail.py` | install checks, environment setup, upload JSONL logs to W&B |
| `configs/` | one JSON per experiment ([configs/README.md](configs/README.md)) |
| `scripts/summarize_runs.py` | rebuild run tables from saved outputs (no training) |
| `reports/` | small curated results and figures (tracked) |
| `docs/` | experiment index, math background (Korean), literature review |
| `runs/` | raw outputs (git-ignored) |
| `test_core.py`, `test_humanoid_costs.py` | tests |

## Setup

```bash
./setup_env.sh                 # Python 3.10 .venv via uv, CPU torch (simulation is CPU-bound)
# Titan XP: TORCH_BACKEND=cu118 ./setup_env.sh
source .venv/bin/activate
export MUJOCO_GL=egl OMP_NUM_THREADS=2
python -m pytest test_core.py
python validate_safety.py --video      # cost-signal check -> runs/validation/
```

## Reproduce an existing experiment

A run is defined by **commit + config + seed + evaluation protocol**. Check out the commit
in the config's `code` field first; historical configs may use arguments that only exist
there (`train.py` refuses unknown ones).

```bash
# E1 toy, one run on the current code. Exact reproduction: check out 4ac4491
# (no --config there) and pass the config's train_args as flags.
python train.py --config configs/toy/toy-comparison-v1.json --method multi --seed 0 \
  --output runs/toy-comparison-v1/multi-s0
# E4-style Safety Gym matrix on this branch (main settings: eval chunk 2)
python run_pilot.py --group sg-pilot-v1-rerun --budget-hazard 31.9 --budget-vase 73.9 \
  --transitions 2000000 --device cpu --dry-run          # drop --dry-run to start
# E2 paper reproduction
./run_paper_repro.sh --dry-run
# Held-out evaluation of a finished run
python evaluate.py runs/<group>/<method>-s<seed>/checkpoint.pt --episodes 50 \
  --environment-seed 900000 --action-seed 900001 --output runs/<group>/<method>-s<seed>/heldout.npz
```

## Start a new experiment

1. Add a row to `docs/experiments/INDEX.md` (next free ID, question, status `planned`).
2. Add `configs/<area>/<name>.json` with `"extends": "base.json"` and only what differs.
3. Launch through the existing tools, e.g. E8:
   ```bash
   for s in 0 1 2; do python calibrate_pi0.py --seed $s; done
   python run_pilot.py --group sg-pi0-v1 --methods multi aggregate_conservative aggregate_loose \
     --budget-hazard 31.9 --budget-vase 73.9 --transitions 2000000 --device cpu \
     --extra "--config configs/safety/pi0-weights.json --eval-num-envs 10 --pi0-calibration runs/calibration/pi0_SafetyPointGoal2-v0_{seed}.json"
   ```
4. When done, write a short `reports/<name>/README.md` and update the index.

## Results

- Curated: `reports/` (toy comparison, humanoid pilot, paper repro, sg-pilot-v1).
- Regenerate tables from raw runs: `python scripts/summarize_runs.py [--root runs/<group>] [--output reports/<group>/runs.md]`.
  It reports status, steps, return, x_i, native feasibility vs. both ≤ 1, final λ, and λ drift.
- Paper comparison: `python compare_paper.py --root runs/paper-repro-safe`.

## Monitor and resume

```bash
tail -n1 runs/<group>/*.log | cut -c1-200          # latest metrics line per run
python scripts/summarize_runs.py --root runs/<group>  # completed vs incomplete
```
`run_pilot.py` and `run_paper_repro.py` skip runs that already have `summary.json`. They
refuse to touch an incomplete run directory: inspect it, move it aside, and rerun the same
command. Memory-guarded launching (`memguard.py`, `pipeline.py`) and the analysis/video
scripts (`analyze_pilot.py`, `render_policies.py`) exist only on branch `branch1`.
