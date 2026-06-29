#!/usr/bin/env bash
set -e

# Custom Bundle Adjustment runner.
# Defaults target the small 16-frame scene used to validate the pipeline.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/scene}"

INPUT_RECON="${INPUT_RECON:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCENE_DIR/ba_custom}"
OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-}"

HUBER_DELTA="${HUBER_DELTA:-1.0}"
MAX_NFEV="${MAX_NFEV:-50}"
OUTLIER_THRESHOLD="${OUTLIER_THRESHOLD:-5.0}"
N_FIXED_CAMERAS="${N_FIXED_CAMERAS:-2}"
CAMERA_TYPE="${CAMERA_TYPE:-PINHOLE}"
VERBOSE="${VERBOSE:-2}"

# Set to 1 to disable round-1 outlier filtering before round 2.
NO_OUTLIER_REMOVAL="${NO_OUTLIER_REMOVAL:-0}"

# Set to 1 to skip COLMAP sparse export.
NO_COLMAP_EXPORT="${NO_COLMAP_EXPORT:-0}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

if [[ -z "$INPUT_RECON" ]]; then
    echo "ERROR: set INPUT_RECON to an explicit reconstruction.npz path." >&2
    exit 2
fi

echo "============================================"
echo "  Custom BA"
echo "============================================"
echo "  Input:             $INPUT_RECON"
echo "  Output root:       $OUTPUT_ROOT"
echo "  Huber delta:       $HUBER_DELTA"
echo "  Max nfev:          $MAX_NFEV"
echo "  Outlier threshold: $OUTLIER_THRESHOLD"
echo "  Fixed cameras:     $N_FIXED_CAMERAS"
echo "  Camera type:       $CAMERA_TYPE"
echo "  Verbose:           $VERBOSE"
echo "  No outlier rm:     $NO_OUTLIER_REMOVAL"
echo "  No COLMAP export:  $NO_COLMAP_EXPORT"
echo "============================================"

CMD=(python -m src.ba.run
    --input "$INPUT_RECON"
    --output "$OUTPUT_ROOT"
    --huber_delta "$HUBER_DELTA"
    --max_nfev "$MAX_NFEV"
    --outlier_threshold "$OUTLIER_THRESHOLD"
    --n_fixed_cameras "$N_FIXED_CAMERAS"
    --camera_type "$CAMERA_TYPE"
    --verbose "$VERBOSE")

if [[ -n "$OUTPUT_RUN_DIR" ]]; then
    CMD+=(--output_run_dir "$OUTPUT_RUN_DIR")
fi

if [[ "$NO_OUTLIER_REMOVAL" == "1" || "$NO_OUTLIER_REMOVAL" == "true" || "$NO_OUTLIER_REMOVAL" == "TRUE" ]]; then
    CMD+=(--no-outlier-removal)
fi

if [[ "$NO_COLMAP_EXPORT" == "1" || "$NO_COLMAP_EXPORT" == "true" || "$NO_COLMAP_EXPORT" == "TRUE" ]]; then
    CMD+=(--no-colmap-export)
fi

"${CMD[@]}"

echo ""
echo "Done! Output root: $OUTPUT_ROOT"
echo "  - reconstruction.npz"
echo "  - ba_stats.json"
if [[ "$NO_COLMAP_EXPORT" != "1" && "$NO_COLMAP_EXPORT" != "true" && "$NO_COLMAP_EXPORT" != "TRUE" ]]; then
    echo "  - sparse/"
fi
echo "Run directory is printed above and recorded in run_config.json. latest symlinks are not updated."
