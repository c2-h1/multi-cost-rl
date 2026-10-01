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
echo "$(date '+%F %T') start: ${common[*]}"

$PY run_pilot.py --group sg-pilot-v1-duallr004 --methods multi --set dual_lr=0.04 "${common[@]}" > runs/duallr004.out 2>&1 &
a=$!
$PY run_pilot.py --group sg-pilot-v1-conservative --methods aggregate_conservative "${common[@]}" > runs/conservative.out 2>&1 &
b=$!
wait $a || { echo "duallr004 group failed, see runs/duallr004.out"; exit 1; }
wait $b || { echo "conservative group failed, see runs/conservative.out"; exit 1; }
echo "$(date '+%F %T') training done; held-out evaluation"

for group in sg-pilot-v1-duallr004 sg-pilot-v1-conservative; do
  for d in runs/$group/*-s[0-9]; do
    [ -s "$d/heldout.npz" ] && continue
    $PY evaluate.py "$d/checkpoint.pt" --episodes 50 --environment-seed 900000 --action-seed 900001 \
      --output "$d/heldout.npz" > "$d/heldout.json" 2> "$d/heldout.stderr" &
  done
done
wait
for group in sg-pilot-v1-duallr004 sg-pilot-v1-conservative; do
  $PY analyze_pilot.py runs/$group > runs/$group/analysis.log 2>&1
  echo; sed -n '/Per method/,$p' runs/$group/analysis/results.md
done
tar czf followups.tgz --exclude='checkpoint-*.pt' runs/sg-pilot-v1-duallr004 runs/sg-pilot-v1-conservative
echo "$(date '+%F %T') DONE: followups.tgz"
