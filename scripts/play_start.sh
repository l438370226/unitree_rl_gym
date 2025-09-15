#!/usr/bin/env bash
# Quick start script to run policy playback and export
# Usage: ./scripts/start_play.sh [--task TASK] [--headless] [--rl_device DEVICE] [additional args...]

set -euo pipefail

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
