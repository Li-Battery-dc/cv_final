#!/usr/bin/env bash
set -e

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/scene}"

INPUT_RECON="${INPUT_RECON:-/home/dhr/cv_final/data/scene/vggt_raw/runs/20260627_074913_vggt_export/reconstruction.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCENE_DIR/ba_pycolmap}"
OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-}"

CAMERA_TYPE="${CAMERA_TYPE:-PINHOLE}"
LOSS_SCALE="${LOSS_SCALE:-1.0}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-100}"
OUTLIER_THRESHOLD="${OUTLIER_THRESHOLD:-5.0}"
N_FIXED_CAMERAS="${N_FIXED_CAMERAS:-2}"
REFINE_FOCAL_LENGTH="${REFINE_FOCAL_LENGTH:-0}"
REFINE_PRINCIPAL_POINT="${REFINE_PRINCIPAL_POINT:-0}"
NO_OUTLIER_REMOVAL="${NO_OUTLIER_REMOVAL:-0}"
NO_COLMAP_EXPORT="${NO_COLMAP_EXPORT:-0}"
PRINT_SOLVER_SUMMARY="${PRINT_SOLVER_SUMMARY:-0}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

echo "============================================"
echo "  PyCOLMAP BA"
echo "============================================"
echo "  Input:             $INPUT_RECON"
echo "  Output root:       $OUTPUT_ROOT"
echo "  Camera type:       $CAMERA_TYPE"
echo "  Loss scale:        $LOSS_SCALE"
echo "  Max iterations:    $MAX_NUM_ITERATIONS"
echo "  Outlier threshold: $OUTLIER_THRESHOLD"
echo "  Fixed cameras:     $N_FIXED_CAMERAS"
echo "============================================"

CMD=(python -m src.ba.pycolmap_run
    --input "$INPUT_RECON"
    --output "$OUTPUT_ROOT"
    --camera_type "$CAMERA_TYPE"
    --loss_scale "$LOSS_SCALE"
    --max_num_iterations "$MAX_NUM_ITERATIONS"
    --outlier_threshold "$OUTLIER_THRESHOLD"
    --n_fixed_cameras "$N_FIXED_CAMERAS")

if [[ -n "$OUTPUT_RUN_DIR" ]]; then
    CMD+=(--output_run_dir "$OUTPUT_RUN_DIR")
fi

if [[ "$REFINE_FOCAL_LENGTH" == "1" || "$REFINE_FOCAL_LENGTH" == "true" || "$REFINE_FOCAL_LENGTH" == "TRUE" ]]; then
    CMD+=(--refine_focal_length)
fi
if [[ "$REFINE_PRINCIPAL_POINT" == "1" || "$REFINE_PRINCIPAL_POINT" == "true" || "$REFINE_PRINCIPAL_POINT" == "TRUE" ]]; then
    CMD+=(--refine_principal_point)
fi
if [[ "$NO_OUTLIER_REMOVAL" == "1" || "$NO_OUTLIER_REMOVAL" == "true" || "$NO_OUTLIER_REMOVAL" == "TRUE" ]]; then
    CMD+=(--no_outlier_removal)
fi
if [[ "$NO_COLMAP_EXPORT" == "1" || "$NO_COLMAP_EXPORT" == "true" || "$NO_COLMAP_EXPORT" == "TRUE" ]]; then
    CMD+=(--no_colmap_export)
fi
if [[ "$PRINT_SOLVER_SUMMARY" == "1" || "$PRINT_SOLVER_SUMMARY" == "true" || "$PRINT_SOLVER_SUMMARY" == "TRUE" ]]; then
    CMD+=(--print_solver_summary)
fi

"${CMD[@]}"

echo ""
echo "Done! Output root: $OUTPUT_ROOT"
echo "  - reconstruction.npz"
echo "  - ba_stats.json"
echo "  - summary.json"
if [[ "$NO_COLMAP_EXPORT" != "1" && "$NO_COLMAP_EXPORT" != "true" && "$NO_COLMAP_EXPORT" != "TRUE" ]]; then
    echo "  - sparse/"
fi
echo "Latest run: $OUTPUT_ROOT/latest"
