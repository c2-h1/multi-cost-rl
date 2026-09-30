#!/usr/bin/env bash
# Stage R: Safety Gym paper baseline on SafetyPointGoal2-v0, 2 algos x 3 seeds, 10M steps each.
#
# One run (30 envs, WORKERS sim processes) needs ~9-11 GiB of RAM. Launching all
# six at once OOM-killed a 62 GB host, so runs are queued: at most MAX_JOBS run
# together (default: what fits in available memory, CPU cores and GPUs).
# On the SLURM cluster use slurm/paper_repro.sbatch instead.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
# Headless GPU rendering on Linux; macOS has no EGL and needs no setting.
[ "$(uname)" = Linux ] && export MUJOCO_GL=${MUJOCO_GL:-egl}
out=runs/paper-repro; mkdir -p $out
PY=${PY:-.venv/bin/python}
WORKERS=${WORKERS:-4}
PER_RUN_GIB=${PER_RUN_GIB:-11}
GPUS=${GPUS:-$( (nvidia-smi -L 2>/dev/null || true) | wc -l)}
ncpu=$(getconf _NPROCESSORS_ONLN)  # nproc does not exist on macOS
avail_gib=$($PY -c "import memguard; print(int(memguard.available_bytes() / memguard.GIB))")
fit_mem=$(( (avail_gib - 4) / PER_RUN_GIB ))
fit_cpu=$(( ncpu / (WORKERS + 1) )); (( fit_cpu >= 1 )) || fit_cpu=1
MAX_JOBS=${MAX_JOBS:-$(( fit_mem < fit_cpu ? fit_mem : fit_cpu ))}
(( MAX_JOBS >= 1 )) || { echo "Not enough memory: ${avail_gib} GiB available, need ~$((PER_RUN_GIB + 4))" >&2; exit 1; }
echo "running at most $MAX_JOBS at once (${avail_gib} GiB available, $GPUS GPUs, $ncpu cores)"

for run in $out/{ppo,ppo_lagrangian}-s{0,1,2}; do
  if [ -e $run ] && [ ! -e $run/summary.json ]; then echo "incomplete run exists, move it away first: $run" >&2; exit 1; fi
done
i=0
for seed in 0 1 2; do for algo in ppo ppo_lagrangian; do
  run=$out/$algo-s$seed
  if [ -e $run/summary.json ]; then echo "done: $run"; continue; fi
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 10; done  # no `wait -n` in macOS bash 3.2
  if (( GPUS > 0 )); then dev=(--device cuda); export CUDA_VISIBLE_DEVICES=$((i % GPUS)); else dev=(--device cpu); fi
  echo "start $run ${dev[*]} gpu=${CUDA_VISIBLE_DEVICES:-none}"
  $PY baseline_ppo.py --algo $algo --seed $seed --workers $WORKERS "${dev[@]}" --output $run > $run.log 2>&1 &
  i=$((i + 1))
done; done
wait
echo "finished; compare with: $PY compare_paper.py"
