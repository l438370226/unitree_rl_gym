#!/usr/bin/env bash
# Quick start script to run policy playback and export
# Usage: ./scripts/start_play.sh [--task TASK] [--headless] [--rl_device DEVICE] [additional args...]

set -euo pipefail

# Activate conda environment (default: unitree). Can override by setting CONDA_ENV_NAME
CONDA_ENV_NAME="${CONDA_ENV_NAME:-unitree}"
if command -v conda >/dev/null 2>&1; then
  if ! conda activate "${CONDA_ENV_NAME}" 2>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null || true)
    if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME}" || true
    fi
  fi
fi

TASK="g1"
RL_DEVICE="cuda:0"
HEADLESS=false
EXPORT=true

while [[ "$#" -gt 0 ]]; do
  key="$1"
  case $key in
    --task) TASK="$2"; shift; shift;;
    --rl_device) RL_DEVICE="$2"; shift; shift;;
    --headless) HEADLESS=true; shift;;
    --no-export|--no-export_policy) EXPORT=false; shift;;
    --export_policy=false)
      EXPORT=false; shift;;
    --export_policy)
      # check next token: if it's 'false' treat as disabling export
      if [[ "${2:-}" == "false" || "${2:-}" == "False" ]]; then
        EXPORT=false; shift; shift;
      else
        # keep as explicit enable - pass through later
        EXTRA_ARGS+="--export_policy "; shift;
      fi;;
    *) EXTRA_ARGS+="$1 "; shift;;
  esac
done

PYTHON=python

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="${SCRIPT_DIR}/.."
CMD=("$PYTHON" "$ROOT_DIR/legged_gym/scripts/play.py" --task "$TASK" --rl_device "$RL_DEVICE")
if [[ "$EXPORT" == true ]]; then
  CMD+=(--export_policy)
fi
if [[ "$HEADLESS" == true ]]; then
  CMD+=(--headless)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  CMD+=($EXTRA_ARGS)
fi

echo "Running: ${CMD[@]}"
exec "${CMD[@]}"
