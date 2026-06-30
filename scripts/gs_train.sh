#!/usr/bin/env bash
set -e

# Custom 3D Gaussian Splatting training runner.
# Defaults target the small BA result so the full pipeline can be validated quickly.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/scene}"

RECONSTRUCTION="${RECONSTRUCTION:-}"
IMAGE_DIR="${IMAGE_DIR:-$SCENE_DIR/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCENE_DIR/gs_custom_ba}"
OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

N_ITERATIONS="${N_ITERATIONS:-10000}"
RESOLUTION_W="${RESOLUTION_W:-768}"
RESOLUTION_H="${RESOLUTION_H:-432}"
INIT_MODE="${INIT_MODE:-reconstruction}"
MAX_INIT_GAUSSIANS="${MAX_INIT_GAUSSIANS:-15000}"
RANDOM_INIT_GAUSSIANS="${RANDOM_INIT_GAUSSIANS:-15000}"
MAX_N_GAUSSIANS="${MAX_N_GAUSSIANS:-300000}"
SCALE_FACTOR="${SCALE_FACTOR:-1.0}"

L1_WEIGHT="${L1_WEIGHT:-0.8}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0.2}"

DENSIFY_FROM="${DENSIFY_FROM:-500}"
DENSIFY_UNTIL="${DENSIFY_UNTIL:-6000}"
DENSIFY_INTERVAL="${DENSIFY_INTERVAL:-200}"
DENSIFY_CLONE_SCALE_FACTOR="${DENSIFY_CLONE_SCALE_FACTOR:-1.5}"
DENSIFY_SPLIT_SCALE_FACTOR="${DENSIFY_SPLIT_SCALE_FACTOR:-2.5}"
DENSIFY_PRUNE_SCALE_FACTOR="${DENSIFY_PRUNE_SCALE_FACTOR:-8.0}"

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

if [[ -z "$RECONSTRUCTION" ]]; then
    echo "ERROR: set RECONSTRUCTION to an explicit reconstruction.npz path. latest symlinks are no longer used." >&2
    exit 2
fi

echo "============================================"
echo "  Custom 3D Gaussian Splatting"
echo "============================================"
echo "  Reconstruction:     $RECONSTRUCTION"
echo "  Image dir:          $IMAGE_DIR"
echo "  Output root:        $OUTPUT_ROOT"
echo "  CUDA devices:       $CUDA_VISIBLE_DEVICES"
echo "  Device:             $DEVICE"
echo "  Iterations:         $N_ITERATIONS"
echo "  Resolution:         ${RESOLUTION_W}x${RESOLUTION_H}"
echo "  Init mode:          $INIT_MODE"
echo "  Max init gaussians: $MAX_INIT_GAUSSIANS"
echo "  Max gaussians:      $MAX_N_GAUSSIANS"
echo "  Densify:            $DENSIFY_FROM-$DENSIFY_UNTIL every $DENSIFY_INTERVAL"
echo "  SH degree:          $SH_DEGREE"
echo "  Val every:          $VAL_EVERY"
echo "============================================"

CMD=(python -m src.gaussian.train
    --reconstruction "$RECONSTRUCTION"
    --image_dir "$IMAGE_DIR"
    --output "$OUTPUT_ROOT"
    --device "$DEVICE"
    --seed "$SEED"
    --n_iterations "$N_ITERATIONS"
    --resolution "$RESOLUTION_W" "$RESOLUTION_H"
    --init_mode "$INIT_MODE"
    --max_init_gaussians "$MAX_INIT_GAUSSIANS"
    --random_init_gaussians "$RANDOM_INIT_GAUSSIANS"
    --max_n_gaussians "$MAX_N_GAUSSIANS"
    --scale_factor "$SCALE_FACTOR"
    --l1_weight "$L1_WEIGHT"
    --ssim_weight "$SSIM_WEIGHT"
    --densify_from "$DENSIFY_FROM"
    --densify_until "$DENSIFY_UNTIL"
    --densify_interval "$DENSIFY_INTERVAL"
    --densify_clone_scale_factor "$DENSIFY_CLONE_SCALE_FACTOR"
    --densify_split_scale_factor "$DENSIFY_SPLIT_SCALE_FACTOR"
    --densify_prune_scale_factor "$DENSIFY_PRUNE_SCALE_FACTOR"
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

if [[ -n "$OUTPUT_RUN_DIR" ]]; then
    CMD+=(--output_run_dir "$OUTPUT_RUN_DIR")
fi

if [[ -n "$RESUME" ]]; then
    CMD+=(--resume "$RESUME")
fi

"${CMD[@]}"

echo ""
echo "Done! Output root: $OUTPUT_ROOT"
echo "  - checkpoints/latest.pt"
echo "  - validation/"
echo "  - final.ply"
echo "  - metrics.json"
echo "Run directory is printed above and recorded in run_config.json. latest symlinks are not updated."
