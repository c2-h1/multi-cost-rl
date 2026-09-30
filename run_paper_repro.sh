#!/usr/bin/env bash
# Stage R: resource-bounded Safety Gym paper reproduction.
set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Extra arguments are forwarded. Examples:
#   ./run_paper_repro.sh --dry-run
#   ./run_paper_repro.sh --device cpu --max-parallel 1
#   nohup ./run_paper_repro.sh > runs/paper-repro-safe.launch.log 2>&1 &
exec .venv/bin/python run_paper_repro.py "$@"
