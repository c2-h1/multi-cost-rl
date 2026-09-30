#!/usr/bin/env bash
# Calibrate the two SafetyPointGoal2 budgets from one unconstrained 500k-step run.
# Submit after the paper baseline array succeeds.
#SBATCH --job-name=sg-calibrate
#SBATCH --partition=titanxp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=runs/sg-calibrate-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

group="${CALIBRATION_GROUP:-sg-calibration-titanxp}"
output="runs/$group/unconstrained-s100"
mkdir -p "runs/$group"

.venv/bin/python train.py \
    --env safety --method unconstrained --seed 100 \
    --budget-hazard 1 --budget-vase 1 \
    --updates 125 --num-envs 4 --device cuda --threads 1 \
    --dual-lr 0.02 --eval-every 25 --checkpoint-every 25 \
    --eval-episodes 10 --final-eval-episodes 50 --eval-num-envs 2 \
    --wandb-mode disabled --group "$group" --output "$output"

.venv/bin/python evaluate.py "$output/checkpoint.pt" \
    --episodes 30 --environment-seed 500000 \
    --output "$output/calibration.npz" > "$output/calibration.json"

.venv/bin/python - "$output" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

output = Path(sys.argv[1])
summary = json.loads((output / "summary.json").read_text())
initial_goals = float(summary["initial"]["eval/goals_per_episode"])
final_goals = float(summary["final/goals_per_episode"])
if final_goals <= initial_goals + 0.1:
    raise SystemExit(
        f"Calibration policy did not learn enough: goals/episode {initial_goals:.3f} -> {final_goals:.3f}"
    )

costs = np.load(output / "calibration.npz")["costs"]
means = costs.mean(axis=0)
active_rates = (costs > 0).mean(axis=0)
if (active_rates < 0.1).any():
    raise SystemExit(
        f"Inactive cost in calibration: means={means.tolist()}, positive_episode_rates={active_rates.tolist()}"
    )
budgets = np.maximum(1.0, 0.5 * means)
record = {
    "source": str(output / "calibration.npz"),
    "episodes": int(len(costs)),
    "unconstrained_mean_costs": {"hazard": float(means[0]), "vase": float(means[1])},
    "positive_episode_rates": {"hazard": float(active_rates[0]), "vase": float(active_rates[1])},
    "budgets": {"hazard": float(budgets[0]), "vase": float(budgets[1])},
    "goals_per_episode": {"initial": initial_goals, "final": final_goals},
}
(output.parent / "budgets.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

