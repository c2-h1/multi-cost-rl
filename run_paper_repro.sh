#!/usr/bin/env bash
# Stage R: Safety Gym paper baseline on SafetyPointGoal2-v0, 2 algos x 3 seeds, 10M steps each.
set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl OMP_NUM_THREADS=2
out=runs/paper-repro; mkdir -p $out
i=0
for seed in 0 1 2; do for algo in ppo ppo_lagrangian; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) nohup .venv/bin/python baseline_ppo.py --algo $algo --seed $seed \
    --workers 4 --output $out/$algo-s$seed > $out/$algo-s$seed.log 2>&1 &
  i=$((i + 1))
done; done
echo "launched $i runs; monitor with: tail -n1 $out/*.log | cut -c1-200"
