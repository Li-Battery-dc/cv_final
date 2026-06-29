#!/usr/bin/env bash
set -e

# Video frame selection wrapper for training-free VGGT improvement.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
VIDEO_PATH="${VIDEO_PATH:-$PROJECT_ROOT/data/raw/3_scene.mp4}"
OUTPUT_SCENE_DIR="${OUTPUT_SCENE_DIR:-$PROJECT_ROOT/data/scene_selected}"

CANDIDATE_COUNT="${CANDIDATE_COUNT:-192}"
FINAL_COUNT="${FINAL_COUNT:-64}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-768}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-432}"
SEED="${SEED:-42}"
IMG_LOAD_RESOLUTION="${IMG_LOAD_RESOLUTION:-448}"
VGGT_RESOLUTION="${VGGT_RESOLUTION:-518}"
CENTRALITY_TOPK="${CENTRALITY_TOPK:-8}"
GAP_TOP_PERCENT="${GAP_TOP_PERCENT:-10}"
LAMBDA_DIVERSITY="${LAMBDA_DIVERSITY:-0.35}"
OVERWRITE="${OVERWRITE:-0}"

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

echo "============================================"
echo "  Video Frame Selection"
echo "============================================"
echo "  Video:          $VIDEO_PATH"
echo "  Output dir:     $OUTPUT_SCENE_DIR"
echo "  CUDA devices:   $CUDA_VISIBLE_DEVICES"
echo "  Candidates:     $CANDIDATE_COUNT"
echo "  Final frames:   $FINAL_COUNT"
echo "  Image size:     ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
echo "  VGGT res:       $VGGT_RESOLUTION"
echo "  Img load res:   $IMG_LOAD_RESOLUTION"
echo "  Centrality topk:$CENTRALITY_TOPK"
echo "  Gap top pct:    $GAP_TOP_PERCENT"
echo "  Lambda div:     $LAMBDA_DIVERSITY"
echo "============================================"

CMD=(python -m src.improvement.video_select
    --video "$VIDEO_PATH"
    --output_scene_dir "$OUTPUT_SCENE_DIR"
    --candidate_count "$CANDIDATE_COUNT"
    --final_count "$FINAL_COUNT"
    --output_width "$OUTPUT_WIDTH"
    --output_height "$OUTPUT_HEIGHT"
    --seed "$SEED"
    --img_load_resolution "$IMG_LOAD_RESOLUTION"
    --vggt_resolution "$VGGT_RESOLUTION"
    --centrality_topk "$CENTRALITY_TOPK"
    --gap_top_percent "$GAP_TOP_PERCENT"
    --lambda_diversity "$LAMBDA_DIVERSITY")

if [[ "$OVERWRITE" = "1" ]]; then
    CMD+=(--overwrite)
fi

"${CMD[@]}"

echo ""
echo "Done! Output scene dir: $OUTPUT_SCENE_DIR"
