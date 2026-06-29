#!/usr/bin/env bash
set -e

PROJECT_ROOT="$HOME/cv_final"
VENV_PATH="$PROJECT_ROOT/.venv"

: "${CHECKPOINT:?Set CHECKPOINT to an explicit checkpoint path, e.g. data/scene/gs_custom_ba/runs/<run>/checkpoints/iter_10000.pt}"
: "${RECONSTRUCTION:?Set RECONSTRUCTION to an explicit reconstruction.npz path}"

DEVICE="${DEVICE:-cuda}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
VIEW_INDEX="${VIEW_INDEX:-0}"
WIDTH="${WIDTH:-768}"
HEIGHT="${HEIGHT:-432}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

python -m src.gaussian.viewer \
    --checkpoint "$CHECKPOINT" \
    --reconstruction "$RECONSTRUCTION" \
    --view_index "$VIEW_INDEX" \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT" \
    --width "$WIDTH" \
    --height "$HEIGHT"
