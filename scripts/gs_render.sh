#!/usr/bin/env bash
set -e

PROJECT_ROOT="$HOME/cv_final"
VENV_PATH="$PROJECT_ROOT/.venv"
CHECKPOINT="/home/dhr/cv_final/data/scene/gs_custom_ba/checkpoints/latest.pt"
RECONSTRUCTION="/home/dhr/cv_final/data/scene/ba_custom/runs/20260627_075839_ba_custom/reconstruction.npz"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

python -m src.gaussian.viewer \
    --checkpoint "$CHECKPOINT" \
    --reconstruction "$RECONSTRUCTION" \
    --view_index 0 \
    --device cuda \
    --host 127.0.0.1 \
    --port 8080 \
    --width 768 \
    --height 432
