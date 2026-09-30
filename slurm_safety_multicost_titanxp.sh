#!/usr/bin/env bash
# Main two-cost SafetyPointGoal2 experiment: 3 methods x 3 seeds, 2M steps each.
# Submit as an array after slurm_safety_calibrate_titanxp.sh succeeds.
#SBATCH --job-name=sg-multicost
#SBATCH --partition=titanxp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-8
#SBATCH --output=runs/sg-multicost-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

calibration_group="${CALIBRATION_GROUP:-sg-calibration-titanxp}"
group="${RUN_GROUP:-sg-pilot-titanxp-v1}"
budget_file="runs/$calibration_group/budgets.json"
test -s "$budget_file"

read -r budget_hazard budget_vase < <(.venv/bin/python - "$budget_file" <<'PY'
import json
import sys

budgets = json.load(open(sys.argv[1]))["budgets"]
print(budgets["hazard"], budgets["vase"])
PY
)

methods=(unconstrained aggregate_loose multi)
method="${methods[$((SLURM_ARRAY_TASK_ID % 3))]}"
seed="$((SLURM_ARRAY_TASK_ID / 3))"
output="runs/$group/$method-s$seed"
mkdir -p "runs/$group"

.venv/bin/python train.py \
    --env safety --method "$method" --seed "$seed" \
    --budget-hazard "$budget_hazard" --budget-vase "$budget_vase" \
    --updates 500 --num-envs 4 --device cuda --threads 1 \
    --epochs 6 --minibatch-size 512 --lr 3e-4 --dual-lr 0.02 \
    --gae-lambda 0.95 --clip 0.2 --target-kl 0.03 \
    --value-coef 1 --entropy-coef 0 --lambda-init 0.1 \
    --eval-every 25 --checkpoint-every 25 --log-every 1 \
    --eval-episodes 10 --final-eval-episodes 50 --eval-num-envs 2 \
    --wandb-mode disabled --group "$group" --output "$output"

.venv/bin/python evaluate.py "$output/checkpoint.pt" \
    --episodes 50 --environment-seed 900000 --action-seed 900001 \
    --output "$output/heldout.npz" > "$output/heldout.json"

