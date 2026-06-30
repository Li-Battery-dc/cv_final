#!/usr/bin/env bash
set -e

# VGGT Export Shell Script
# Runs VGGT inference + track prediction + COLMAP export on the office scene.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/scene}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCENE_DIR/vggt_raw}"
OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-}"

# 分阶段跑省显存
STAGE="${STAGE:-all}" # tracks, dense, vggt, or all
VGGT_CACHE="${VGGT_CACHE:-}"

MAX_QUERY_PTS="${MAX_QUERY_PTS:-512}" # 非常影响显存，64帧基本上512满，32帧768满
QUERY_FRAME_NUM="${QUERY_FRAME_NUM:-12}"
VIS_THRESH="${VIS_THRESH:-0.1}"
MAX_REPROJ_ERROR="${MAX_REPROJ_ERROR:-0.0}"
MIN_VISIBLE_FRAMES="${MIN_VISIBLE_FRAMES:-2}"
CAMERA_TYPE="${CAMERA_TYPE:-PINHOLE}"
FINE_TRACKING="${FINE_TRACKING:-1}"

IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-448}"
CONF_THRES_VALUE="${CONF_THRES_VALUE:-2.0}"
MAX_DENSE_POINTS="${MAX_DENSE_POINTS:-100000}"
MASK_DIR="${MASK_DIR:-}"
MASK_FOREGROUND_THRESHOLD="${MASK_FOREGROUND_THRESHOLD:-0.5}"
MASK_MIN_OBSERVATIONS="${MASK_MIN_OBSERVATIONS:-2}"
MASK_MIN_RATIO="${MASK_MIN_RATIO:-0.5}"
INIT_POINTS_SOURCE="${INIT_POINTS_SOURCE:-depth}" # depth or point_head
ENABLE_POINT_HEAD="${ENABLE_POINT_HEAD:-0}"
SAVE_DENSE_FILTERED_RECONSTRUCTION="${SAVE_DENSE_FILTERED_RECONSTRUCTION:-0}"
DENSE_RECONSTRUCTION_VARIANTS="${DENSE_RECONSTRUCTION_VARIANTS:-}"
DENSE_FILTER_DISAGREEMENT_PERCENTILE="${DENSE_FILTER_DISAGREEMENT_PERCENTILE:-70}"
DENSE_FILTER_REPROJ_PERCENTILE="${DENSE_FILTER_REPROJ_PERCENTILE:-70}"
DENSE_FILTER_MIN_VOTES="${DENSE_FILTER_MIN_VOTES:-1}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

if [[ ("$STAGE" == "tracks" || "$STAGE" == "dense") && -z "$VGGT_CACHE" ]]; then
    echo "ERROR: STAGE=$STAGE requires an explicit VGGT_CACHE path. latest symlinks are no longer used." >&2
    exit 2
fi

echo "============================================"
echo "  VGGT Export Pipeline"
echo "============================================"
echo "  Scene dir:      $SCENE_DIR"
echo "  Output root:    $OUTPUT_ROOT"
echo "  Stage:          $STAGE"
echo "  VGGT cache:     $VGGT_CACHE"
echo "  Image res:      $IMAGE_RESOLUTION"
echo "  Query frames:   $QUERY_FRAME_NUM"
echo "  Max query pts:  $MAX_QUERY_PTS"
echo "  Fine tracking:  $FINE_TRACKING"
echo "  Vis thresh:     $VIS_THRESH"
echo "  Reproj error:   $MAX_REPROJ_ERROR"
echo "  Min vis frames: $MIN_VISIBLE_FRAMES"
echo "  Mask dir:       $MASK_DIR"
echo "  Init points:    $INIT_POINTS_SOURCE"
echo "  Point head:     $ENABLE_POINT_HEAD"
echo "  Dense filtered: $SAVE_DENSE_FILTERED_RECONSTRUCTION"
echo "  Dense variants: $DENSE_RECONSTRUCTION_VARIANTS"
echo "  Max dense pts:  $MAX_DENSE_POINTS"
echo "============================================"

CMD=(python -m src.vggt_export
    --scene_dir "$SCENE_DIR"
    --output_dir "$OUTPUT_ROOT"
    --stage "$STAGE"
    --img_load_resolution "$IMAGE_RESOLUTION"
    --max_query_pts "$MAX_QUERY_PTS"
    --query_frame_num "$QUERY_FRAME_NUM"
    --vis_thresh "$VIS_THRESH"
    --max_reproj_error "$MAX_REPROJ_ERROR"
    --min_visible_frames "$MIN_VISIBLE_FRAMES"
    --init_points_source "$INIT_POINTS_SOURCE"
    --dense_reconstruction_variants "$DENSE_RECONSTRUCTION_VARIANTS"
    --conf_thres_value "$CONF_THRES_VALUE"
    --max_dense_points "$MAX_DENSE_POINTS"
    --dense_filter_disagreement_percentile "$DENSE_FILTER_DISAGREEMENT_PERCENTILE"
    --dense_filter_reproj_percentile "$DENSE_FILTER_REPROJ_PERCENTILE"
    --dense_filter_min_votes "$DENSE_FILTER_MIN_VOTES"
    --camera_type "$CAMERA_TYPE")

if [[ -n "$OUTPUT_RUN_DIR" ]]; then
    CMD+=(--output_run_dir "$OUTPUT_RUN_DIR")
fi
if [[ -n "$VGGT_CACHE" ]]; then
    CMD+=(--vggt_cache "$VGGT_CACHE")
fi
if [[ -n "$MASK_DIR" ]]; then
    CMD+=(--mask_dir "$MASK_DIR"
        --mask_foreground_threshold "$MASK_FOREGROUND_THRESHOLD"
        --mask_min_observations "$MASK_MIN_OBSERVATIONS"
        --mask_min_ratio "$MASK_MIN_RATIO")
fi
if [[ "$FINE_TRACKING" = "1" ]]; then
    CMD+=(--fine_tracking)
else
    CMD+=(--no-fine_tracking)
fi
if [[ "$ENABLE_POINT_HEAD" = "1" ]]; then
    CMD+=(--enable_point_head)
fi
if [[ "$SAVE_DENSE_FILTERED_RECONSTRUCTION" = "1" ]]; then
    CMD+=(--save_dense_filtered_reconstruction)
fi

"${CMD[@]}"

echo ""
echo "Done! Output root: $OUTPUT_ROOT"
echo "Run directory is printed above and recorded in run_config.json. latest symlinks are not updated."
