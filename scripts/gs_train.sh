#!/usr/bin/env bash
set -e

# Custom 3D Gaussian Splatting training runner.
# Defaults target the small BA result so the full pipeline can be validated quickly.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/test_small}"

RECONSTRUCTION="${RECONSTRUCTION:-$SCENE_DIR/ba_custom/reconstruction.npz}"
IMAGE_DIR="${IMAGE_DIR:-$SCENE_DIR/images}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCENE_DIR/gs_custom_ba}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

N_ITERATIONS="${N_ITERATIONS:-10000}"
RESOLUTION_W="${RESOLUTION_W:-768}"
RESOLUTION_H="${RESOLUTION_H:-432}"
MAX_INIT_GAUSSIANS="${MAX_INIT_GAUSSIANS:-100000}"
MAX_N_GAUSSIANS="${MAX_N_GAUSSIANS:-300000}"
SCALE_FACTOR="${SCALE_FACTOR:-1.0}"

L1_WEIGHT="${L1_WEIGHT:-0.8}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0.2}"

DENSIFY_FROM="${DENSIFY_FROM:-500}"
DENSIFY_UNTIL="${DENSIFY_UNTIL:-6000}"
DENSIFY_INTERVAL="${DENSIFY_INTERVAL:-200}"

SH_DEGREE="${SH_DEGREE:-2}"
SH_DEGREE_INTERVAL="${SH_DEGREE_INTERVAL:-1000}"

LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
VAL_EVERY="${VAL_EVERY:-8}"

LR_XYZ="${LR_XYZ:-1.6e-4}"
LR_SCALES="${LR_SCALES:-5.0e-3}"
LR_QUAT="${LR_QUAT:-1.0e-3}"
LR_OPACITY="${LR_OPACITY:-5.0e-2}"
LR_SH="${LR_SH:-2.5e-3}"

RESUME="${RESUME:-}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES

echo "============================================"
echo "  Custom 3D Gaussian Splatting"
echo "============================================"
echo "  Reconstruction:     $RECONSTRUCTION"
echo "  Image dir:          $IMAGE_DIR"
echo "  Output dir:         $OUTPUT_DIR"
echo "  CUDA devices:       $CUDA_VISIBLE_DEVICES"
echo "  Device:             $DEVICE"
echo "  Iterations:         $N_ITERATIONS"
echo "  Resolution:         ${RESOLUTION_W}x${RESOLUTION_H}"
echo "  Max init gaussians: $MAX_INIT_GAUSSIANS"
echo "  Max gaussians:      $MAX_N_GAUSSIANS"
echo "  Densify:            $DENSIFY_FROM-$DENSIFY_UNTIL every $DENSIFY_INTERVAL"
echo "  SH degree:          $SH_DEGREE"
echo "  Val every:          $VAL_EVERY"
echo "============================================"

CMD=(python -m src.gaussian.train
    --reconstruction "$RECONSTRUCTION"
    --image_dir "$IMAGE_DIR"
    --output "$OUTPUT_DIR"
    --device "$DEVICE"
    --seed "$SEED"
    --n_iterations "$N_ITERATIONS"
    --resolution "$RESOLUTION_W" "$RESOLUTION_H"
    --max_init_gaussians "$MAX_INIT_GAUSSIANS"
    --max_n_gaussians "$MAX_N_GAUSSIANS"
    --scale_factor "$SCALE_FACTOR"
    --l1_weight "$L1_WEIGHT"
    --ssim_weight "$SSIM_WEIGHT"
    --densify_from "$DENSIFY_FROM"
    --densify_until "$DENSIFY_UNTIL"
    --densify_interval "$DENSIFY_INTERVAL"
    --sh_degree "$SH_DEGREE"
    --sh_degree_interval "$SH_DEGREE_INTERVAL"
    --log_interval "$LOG_INTERVAL"
    --val_interval "$VAL_INTERVAL"
    --checkpoint_interval "$CHECKPOINT_INTERVAL"
    --val_every "$VAL_EVERY"
    --lr_xyz "$LR_XYZ"
    --lr_scales "$LR_SCALES"
    --lr_quat "$LR_QUAT"
    --lr_opacity "$LR_OPACITY"
    --lr_sh "$LR_SH")

if [[ -n "$RESUME" ]]; then
    CMD+=(--resume "$RESUME")
fi

"${CMD[@]}"

echo ""
echo "Done! Output saved to: $OUTPUT_DIR"
echo "  - checkpoints/latest.pt"
echo "  - validation/"
echo "  - final.ply"
echo "  - metrics.json"
