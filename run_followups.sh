#!/usr/bin/env bash
# Follow-ups to sg-pilot-v1, with the same calibrated budgets:
#   sg-pilot-v1-duallr004     multi, dual_lr 0.04 (matches the shared multiplier's growth rate)
#   sg-pilot-v1-conservative  aggregate_conservative (hazard + vase <= 1), the strict shared control
# Then held-out evaluation and analysis for both groups. Resumable.
#   nohup bash run_followups.sh > runs/followups.out 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
[ "$(uname)" = Linux ] && export MUJOCO_GL=${MUJOCO_GL:-egl}
common=(--budget-hazard "${BUDGET_HAZARD:-31.9}" --budget-vase "${BUDGET_VASE:-73.9}"
        --transitions "${TRANSITIONS:-2000000}" --seeds ${SEEDS:-0 1 2}
        --device cpu --max-jobs 3 --wandb-mode disabled)
mkdir -p runs
# Shared-server safety: every run aborts itself if free memory falls below this,
# and all jobs run at low CPU priority.
export MEMGUARD_MIN_FREE_GIB=${MEMGUARD_MIN_FREE_GIB:-6}
PY="nice -n 10 $PY"
# One job (4 train + 10 eval envs) peaks at ~3.7 GiB; budget 4 GiB each plus 8 GiB headroom.
free_gib=$($PY -c "import memguard; print(int(memguard.available_bytes() / memguard.GIB))")
echo "$(date '+%F %T') start: ${common[*]} | ${free_gib} GiB available"
MIN_START_GIB=${MIN_START_GIB:-12}; PARALLEL_GIB=${PARALLEL_GIB:-32}
if (( free_gib < MIN_START_GIB )); then
  echo "Not enough free memory (${free_gib} GiB) for even one job plus headroom; not starting."; exit 1
fi

group_a() { $PY run_pilot.py --group sg-pilot-v1-duallr004 --methods multi --set dual_lr=0.04 "${common[@]}" > runs/duallr004.out 2>&1 \
            || { echo "duallr004 group failed, see runs/duallr004.out"; return 1; }; }
group_b() { $PY run_pilot.py --group sg-pilot-v1-conservative --methods aggregate_conservative "${common[@]}" > runs/conservative.out 2>&1 \
            || { echo "conservative group failed, see runs/conservative.out"; return 1; }; }
if (( free_gib >= PARALLEL_GIB )); then
  echo "running both groups side by side (up to 6 jobs)"
  group_a & a=$!; group_b & b=$!
  wait $a; wait $b
else
  echo "running the groups one after the other to stay within memory"
  group_a; group_b
fi
echo "$(date '+%F %T') training done; held-out evaluation"

for group in sg-pilot-v1-duallr004 sg-pilot-v1-conservative; do
  for d in runs/$group/*-s[0-9]; do
    [ -s "$d/heldout.npz" ] && continue
    # One evaluation at a time (~2.5 GiB each) to stay gentle on a shared server.
    $PY evaluate.py "$d/checkpoint.pt" --episodes 50 --environment-seed 900000 --action-seed 900001 \
      --output "$d/heldout.npz" > "$d/heldout.json" 2> "$d/heldout.stderr" \
      || { echo "held-out evaluation failed for $d, see $d/heldout.stderr"; exit 1; }
  done
done
for group in sg-pilot-v1-duallr004 sg-pilot-v1-conservative; do
  $PY analyze_pilot.py runs/$group > runs/$group/analysis.log 2>&1
  echo; sed -n '/Per method/,$p' runs/$group/analysis/results.md
done
tar czf followups.tgz --exclude='checkpoint-*.pt' runs/sg-pilot-v1-duallr004 runs/sg-pilot-v1-conservative
echo "$(date '+%F %T') DONE: followups.tgz"
