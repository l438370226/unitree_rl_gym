#!/usr/bin/env bash
# Quick start script to launch training for unitree_rl_gym
# Usage: ./scripts/start_train.sh [--task TASK] [--rl_device DEVICE] [--num_envs N] [--experiment_name NAME] [--run_name NAME] [--max_iterations N]

set -euo pipefail

# Activate conda environment (default: unitree). Can override by setting CONDA_ENV_NAME
CONDA_ENV_NAME="${CONDA_ENV_NAME:-unitree}"
if command -v conda >/dev/null 2>&1; then
  # Try standard conda activate
  if ! conda activate "${CONDA_ENV_NAME}" 2>/dev/null; then
    # Fallback: source conda.sh then activate
    CONDA_BASE=$(conda info --base 2>/dev/null || true)
    if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME}" || true
    fi
  fi
fi

# Defaults (can be overridden by passing arguments)
TASK="g1"
RL_DEVICE="cuda:0"
NUM_ENVS="-1"
EXPERIMENT_NAME=""
RUN_NAME=""
# default to 10000 as requested
MAX_ITERATIONS="10000"
# default training is headless
HEADLESS=true

# Forward all args to python script while allowing simple env var overrides
while [[ "$#" -gt 0 ]]; do
  key="$1"
  case $key in
    --task) TASK="$2"; shift; shift;;
    --rl_device) RL_DEVICE="$2"; shift; shift;;
    --num_envs) NUM_ENVS="$2"; shift; shift;;
    --experiment_name) EXPERIMENT_NAME="$2"; shift; shift;;
    --run_name) RUN_NAME="$2"; shift; shift;;
    --max_iterations) MAX_ITERATIONS="$2"; shift; shift;;
    --headless) HEADLESS=true; shift;;
    --no-headless) HEADLESS=false; shift;;
    *)
      # unknown option passed through
      EXTRA_ARGS+="$1 "; shift;;
  esac
done

PYTHON=python

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="${SCRIPT_DIR}/.."
CMD=("$PYTHON" "${ROOT_DIR}/legged_gym/scripts/train.py" --task "$TASK" --rl_device "$RL_DEVICE")
if [[ "$HEADLESS" == true ]]; then
  CMD+=(--headless)
fi
if [[ "$NUM_ENVS" != "-1" && -n "$NUM_ENVS" ]]; then
  CMD+=(--num_envs "$NUM_ENVS")
fi
if [[ -n "$EXPERIMENT_NAME" ]]; then
  CMD+=(--experiment_name "$EXPERIMENT_NAME")
fi
if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run_name "$RUN_NAME")
fi
if [[ -n "$MAX_ITERATIONS" ]]; then
  CMD+=(--max_iterations "$MAX_ITERATIONS")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  CMD+=($EXTRA_ARGS)
fi

echo "Running: ${CMD[@]}"
exec "${CMD[@]}"
