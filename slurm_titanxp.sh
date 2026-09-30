#!/usr/bin/env bash
# Submit with: sbatch slurm_titanxp.sh
# Six array tasks are independent; Slurm can place at most two on each 2-GPU,
# 6-core node because every task requests one GPU and three CPU cores.
#SBATCH --job-name=mc-rl-paper
#SBATCH --partition=titanxp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5
#SBATCH --output=runs/slurm-%A_%a.out

set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

algos=(ppo ppo_lagrangian)
seeds=(0 0 1 1 2 2)
algo="${algos[$((SLURM_ARRAY_TASK_ID % 2))]}"
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
task_group="${RUN_GROUP:-paper-repro-titanxp}"
task_output="runs/$task_group/$algo-s$seed"
mkdir -p "runs/$task_group"

.venv/bin/python baseline_ppo.py \
    --algo "$algo" --seed "$seed" --device auto \
    --num-envs 2 --workers 2 --threads 1 \
    --total-steps "${TOTAL_STEPS:-10000000}" \
    --output "$task_output"
