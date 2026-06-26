#!/usr/bin/env bash
set -e

# Headless 3DGS renderer for server environments.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/test_small}"
GS_DIR="${GS_DIR:-$SCENE_DIR/gs_custom_ba}"

CHECKPOINT="${CHECKPOINT:-$GS_DIR/checkpoints/latest.pt}"
RECONSTRUCTION="${RECONSTRUCTION:-$SCENE_DIR/ba_custom/reconstruction.npz}"
IMAGE_DIR="${IMAGE_DIR:-$SCENE_DIR/images}"
OUTPUT_DIR="${OUTPUT_DIR:-$GS_DIR/headless_renders}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"
RESOLUTION_W="${RESOLUTION_W:-320}"
RESOLUTION_H="${RESOLUTION_H:-180}"
VIEWS="${VIEWS:-val}"
VAL_EVERY="${VAL_EVERY:-8}"
SIDE_BY_SIDE="${SIDE_BY_SIDE:-1}"
BG_R="${BG_R:-0.0}"
BG_G="${BG_G:-0.0}"
BG_B="${BG_B:-0.0}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES

echo "============================================"
echo "  Headless 3DGS Render"
echo "============================================"
echo "  Checkpoint:     $CHECKPOINT"
echo "  Reconstruction: $RECONSTRUCTION"
echo "  Image dir:      $IMAGE_DIR"
echo "  Output dir:     $OUTPUT_DIR"
echo "  CUDA devices:   $CUDA_VISIBLE_DEVICES"
echo "  Device:         $DEVICE"
echo "  Resolution:     ${RESOLUTION_W}x${RESOLUTION_H}"
echo "  Views:          $VIEWS"
echo "  Side-by-side:   $SIDE_BY_SIDE"
echo "============================================"

CMD=(python -m src.gaussian.render
    --checkpoint "$CHECKPOINT"
    --reconstruction "$RECONSTRUCTION"
    --image_dir "$IMAGE_DIR"
    --output "$OUTPUT_DIR"
    --device "$DEVICE"
    --resolution "$RESOLUTION_W" "$RESOLUTION_H"
    --views "$VIEWS"
    --val_every "$VAL_EVERY"
    --bg "$BG_R" "$BG_G" "$BG_B")

if [[ "$SIDE_BY_SIDE" == "1" || "$SIDE_BY_SIDE" == "true" || "$SIDE_BY_SIDE" == "TRUE" ]]; then
    CMD+=(--side_by_side)
fi

"${CMD[@]}"

echo ""
echo "Done! Renders saved to: $OUTPUT_DIR"
