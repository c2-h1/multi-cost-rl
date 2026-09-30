#!/usr/bin/env bash
# Creates .venv (Python 3.10 via uv) with Safety-Gymnasium and PyTorch.
# Linux defaults to the CPU wheel because simulation is CPU-bound and this is
# the most portable option.  For Titan XP, request the CUDA 11.8 wheel with:
#   TORCH_BACKEND=cu118 ./setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"
UV="${UV:-$HOME/.local/bin/uv}"
[ -x "$UV" ] || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
if [ -x .venv/bin/python ]; then
    task_python_version="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$task_python_version" = 3.10 ]; then
        "$UV" venv --python 3.10 --allow-existing .venv
    elif [ "${MCRL_RESET_VENV:-0}" = 1 ]; then
        "$UV" venv --python 3.10 --clear .venv
    else
        echo ".venv uses Python $task_python_version; rerun with MCRL_RESET_VENV=1 to replace this project venv" >&2
        exit 2
    fi
else
    "$UV" venv --python 3.10 .venv
fi
if [ "$(uname -s)" = Darwin ]; then
    "$UV" pip install --python .venv/bin/python -r requirements.txt
else
    task_torch_backend="${TORCH_BACKEND:-cpu}"
    case "$task_torch_backend" in
        cpu|cu118|cu126) ;;
        *) echo "TORCH_BACKEND must be cpu, cu118, or cu126" >&2; exit 2 ;;
    esac
    "$UV" pip install --python .venv/bin/python \
        --index-url "https://download.pytorch.org/whl/$task_torch_backend" \
        --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
        -r requirements.txt
fi
"$UV" pip freeze --python .venv/bin/python > requirements-lock.txt
.venv/bin/python - <<'PY'
import platform, torch, mujoco, gymnasium, safety_gymnasium
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpus", torch.cuda.device_count(), "| mujoco", mujoco.__version__,
      "| gymnasium", gymnasium.__version__, "| safety_gymnasium", safety_gymnasium.__version__)
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    required = f"sm_{capability[0]}{capability[1]}"
    arches = torch.cuda.get_arch_list()
    print("gpu", torch.cuda.get_device_name(), "| capability", capability, "| wheel arches", arches)
    if arches and required not in arches:
        raise SystemExit(f"Installed PyTorch wheel does not contain {required}")
    (torch.ones(1, device="cuda") + 1).cpu()
    torch.cuda.synchronize()
PY
