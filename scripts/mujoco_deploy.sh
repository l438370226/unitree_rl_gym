#!/usr/bin/env bash
# Quick start script to run MuJoCo deployment viewer using a config in deploy/deploy_mujoco/configs
# Usage: ./scripts/start_mujoco.sh <robot_name> [--python PYTHON]

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

# If no argument provided, print usage
if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <robot_name> [--python PYTHON]"
  echo "Example: $0 g1"
  exit 1
fi

# Accept robot name (no .yaml). Default robot is g1
ROBOT_NAME="$1"
shift || true
CONFIG_FILE="${ROBOT_NAME}.yaml"

PYTHON=python
while [[ "$#" -gt 0 ]]; do
  key="$1"
  case $key in
    --python) PYTHON="$2"; shift; shift;;
    *) EXTRA_ARGS+="$1 "; shift;;
  esac
done

DEPLOY_SCRIPT="deploy/deploy_mujoco/deploy_mujoco.py"

CMD=("$PYTHON" "$DEPLOY_SCRIPT" "$CONFIG_FILE")
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  CMD+=($EXTRA_ARGS)
fi

echo "Running: ${CMD[@]}"
exec "${CMD[@]}"
