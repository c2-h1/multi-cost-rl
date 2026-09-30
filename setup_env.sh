#!/usr/bin/env bash
# Creates .venv (Python 3.10 via uv) with Safety-Gymnasium and CUDA PyTorch.
set -euo pipefail
cd "$(dirname "$0")"
UV="${UV:-$HOME/.local/bin/uv}"
[ -x "$UV" ] || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
"$UV" venv --python 3.10 .venv
# cu126 wheels still include sm_75 (RTX 2080 Ti).
"$UV" pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match -r requirements.txt
"$UV" pip freeze --python .venv/bin/python > requirements-lock.txt
.venv/bin/python - <<'PY'
import platform, torch, mujoco, gymnasium, safety_gymnasium
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpus", torch.cuda.device_count(), "| mujoco", mujoco.__version__,
      "| gymnasium", gymnasium.__version__, "| safety_gymnasium", safety_gymnasium.__version__)
PY
