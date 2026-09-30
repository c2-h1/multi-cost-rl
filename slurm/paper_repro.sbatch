#!/usr/bin/env bash
# Stage R on the titanxp partition: one array task per (algo, seed), 6 tasks.
# Each task gets 1 Titan XP, 3 of the node's 6 cores and a 14 GB cgroup, so two
# tasks share a node and a runaway job is killed by SLURM, not the node.
#   mkdir -p runs/paper-repro && sbatch slurm/paper_repro.sbatch   (from the repo root)
#SBATCH --job-name=paper-repro
#SBATCH --partition=titanxp
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=14G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/paper-repro/slurm-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export MUJOCO_GL=egl OMP_NUM_THREADS=1
algos=(ppo ppo_lagrangian)
seed=$(( SLURM_ARRAY_TASK_ID / 2 ))
algo=${algos[$(( SLURM_ARRAY_TASK_ID % 2 ))]}
run=runs/paper-repro/$algo-s$seed
if [ -e $run/summary.json ]; then echo "already complete: $run"; exit 0; fi
if [ -e $run ]; then echo "incomplete run exists, move it away first: $run" >&2; exit 1; fi
# 2 sim workers + the learner = 3 cores. DEVICE=cpu if the GPU build lacks sm_61.
.venv/bin/python baseline_ppo.py --algo $algo --seed $seed --workers 2 --threads 1 \
    --device "${DEVICE:-cuda}" --output $run
