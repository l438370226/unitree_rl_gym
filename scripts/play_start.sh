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

while [[ "$#" -gt 0 ]]; do
  key="$1"
  case $key in
    --task) TASK="$2"; shift; shift;;
    --rl_device) RL_DEVICE="$2"; shift; shift;;
    --headless) HEADLESS=true; shift;;
    *) EXTRA_ARGS+="$1 "; shift;;
  esac
done

PYTHON=python

CMD=("$PYTHON" "legged_gym/scripts/play.py" --task "$TASK" --rl_device "$RL_DEVICE")
if [[ "$HEADLESS" == true ]]; then
  CMD+=(--headless)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  CMD+=($EXTRA_ARGS)
fi

echo "Running: ${CMD[@]}"
exec "${CMD[@]}"
