#!/usr/bin/env bash
set -e

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
ASSET_DIR="${ASSET_DIR:-report/assets}"
METRICS_DIR="${METRICS_DIR:-report/metrics}"
VIEW_COUNT="${VIEW_COUNT:-4}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

python -m src.tools.report_assets \
    --project_root "$PROJECT_ROOT" \
    --asset_dir "$ASSET_DIR" \
    --metrics_dir "$METRICS_DIR" \
    --view_count "$VIEW_COUNT"
