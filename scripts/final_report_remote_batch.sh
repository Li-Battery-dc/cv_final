#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/cv_final}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/final_report/$BATCH_ID}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/remote_runs/final_report_batch_$BATCH_ID}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ITERATIONS="${ITERATIONS:-10000}"
RESOLUTION="${RESOLUTION:-768}"
SH_DEGREE="${SH_DEGREE:-2}"
TEST_EVERY="${TEST_EVERY:-8}"
MAX_NFEV="${MAX_NFEV:-100}"

mkdir -p "$OUT_ROOT" "$RUN_ROOT"

STATUS_TSV="$RUN_ROOT/status.tsv"
MANIFEST_TSV="$OUT_ROOT/manifest.tsv"
SUMMARY_JSON="$OUT_ROOT/aggregate_summary.json"

printf "batch_id\t%s\nproject_root\t%s\nout_root\t%s\nrun_root\t%s\n" \
  "$BATCH_ID" "$PROJECT_ROOT" "$OUT_ROOT" "$RUN_ROOT" > "$OUT_ROOT/batch_info.tsv"
printf "step_id\tname\tstatus\tstart_utc\tend_utc\texit_code\tlog\n" > "$STATUS_TSV"
printf "key\tpath\tnote\n" > "$MANIFEST_TSV"

log_manifest() {
  local key="$1"
  local path="$2"
  local note="${3:-}"
  printf "%s\t%s\t%s\n" "$key" "$path" "$note" >> "$MANIFEST_TSV"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing required file: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "missing required directory: $path"
}

disk_check() {
  df -h "$PROJECT_ROOT" | tee -a "$RUN_ROOT/disk.log"
}

run_cmd() {
  local step_id="$1"
  local name="$2"
  local cmd="$3"
  local log="$RUN_ROOT/${step_id}_${name}.log"
  local start end rc

  if [[ -f "$RUN_ROOT/${step_id}_${name}.success" ]]; then
    echo "SKIP $step_id $name"
    return 0
  fi

  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "%s\t%s\tRUNNING\t%s\t\t\t%s\n" "$step_id" "$name" "$start" "$log" >> "$STATUS_TSV"
  echo "START $step_id $name $start"
  echo "CMD: $cmd" > "$log"
  disk_check >> "$log" 2>&1

  set +e
  bash -lc "$cmd" >> "$log" 2>&1
  rc=$?
  set -u

  end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$rc" -eq 0 ]]; then
    touch "$RUN_ROOT/${step_id}_${name}.success"
    printf "%s\t%s\tSUCCESS\t%s\t%s\t%s\t%s\n" "$step_id" "$name" "$start" "$end" "$rc" "$log" >> "$STATUS_TSV"
    echo "DONE $step_id $name $end"
  else
    touch "$RUN_ROOT/${step_id}_${name}.failed"
    printf "%s\t%s\tFAILED\t%s\t%s\t%s\t%s\n" "$step_id" "$name" "$start" "$end" "$rc" "$log" >> "$STATUS_TSV"
    echo "FAILED $step_id $name rc=$rc log=$log" >&2
    tail -120 "$log" >&2 || true
    exit "$rc"
  fi
}

vggt_cmd() {
  local scene_dir="$1"
  local out_dir="$2"
  local stage="$3"
  local extra_env="$4"
  printf "cd %q && source .venv/bin/activate && export PYTHONPATH=src:vggt && export CUDA_VISIBLE_DEVICES=%q && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && SCENE_DIR=%q OUTPUT_ROOT=%q OUTPUT_RUN_DIR=%q STAGE=%q MAX_REPROJ_ERROR=0.0 VIS_THRESH=0.1 MIN_VISIBLE_FRAMES=2 IMAGE_RESOLUTION=448 %s bash scripts/vggt_export.sh" \
    "$PROJECT_ROOT" "$CUDA_VISIBLE_DEVICES" "$scene_dir" "$out_dir" "$out_dir" "$stage" "$extra_env"
}

ba_cmd() {
  local input_recon="$1"
  local out_dir="$2"
  printf "cd %q && source .venv/bin/activate && export PYTHONPATH=src:vggt && INPUT_RECON=%q OUTPUT_ROOT=%q OUTPUT_RUN_DIR=%q MAX_NFEV=%q HUBER_DELTA=1.0 OUTLIER_THRESHOLD=5.0 bash scripts/ba_run.sh" \
    "$PROJECT_ROOT" "$input_recon" "$out_dir" "$out_dir" "$MAX_NFEV"
}

official_cmd() {
  local recon="$1"
  local image_dir="$2"
  local out_dir="$3"
  local init_mode="${4:-reconstruction}"
  local extra_env="${5:-}"
  printf "cd %q && source .venv/bin/activate && export PYTHONPATH=src:vggt && export CUDA_VISIBLE_DEVICES=%q && RECONSTRUCTION=%q IMAGE_DIR=%q OUTPUT_ROOT=%q OUTPUT_RUN_DIR=%q INIT_MODE=%q ITERATIONS=%q RESOLUTION=%q SH_DEGREE=%q TEST_EVERY=%q %s bash scripts/gs_train_official.sh" \
    "$PROJECT_ROOT" "$CUDA_VISIBLE_DEVICES" "$recon" "$image_dir" "$out_dir" "$out_dir" "$init_mode" "$ITERATIONS" "$RESOLUTION" "$SH_DEGREE" "$TEST_EVERY" "$extra_env"
}

custom_gs_cmd() {
  local recon="$1"
  local image_dir="$2"
  local out_dir="$3"
  printf "cd %q && source .venv/bin/activate && export PYTHONPATH=src:vggt && export CUDA_VISIBLE_DEVICES=%q && RECONSTRUCTION=%q IMAGE_DIR=%q OUTPUT_ROOT=%q OUTPUT_RUN_DIR=%q N_ITERATIONS=%q RESOLUTION_W=768 RESOLUTION_H=432 SH_DEGREE=%q bash scripts/gs_train.sh" \
    "$PROJECT_ROOT" "$CUDA_VISIBLE_DEVICES" "$recon" "$image_dir" "$out_dir" "$out_dir" "$ITERATIONS" "$SH_DEGREE"
}

video_select_cmd() {
  local out_scene="$1"
  printf "cd %q && source .venv/bin/activate && export PYTHONPATH=src:vggt && export CUDA_VISIBLE_DEVICES=%q && OUTPUT_SCENE_DIR=%q OVERWRITE=1 CANDIDATE_COUNT=192 FINAL_COUNT=64 bash scripts/video_select.sh" \
    "$PROJECT_ROOT" "$CUDA_VISIBLE_DEVICES" "$out_scene"
}

write_pointer_dir() {
  local out_dir="$1"
  shift
  mkdir -p "$out_dir"
  printf "%s\n" "$@" > "$out_dir/pointers.txt"
}

collect_summary() {
  "$PROJECT_ROOT/.venv/bin/python" - "$OUT_ROOT" "$SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
summary = {
    "out_root": str(root),
    "official_results": [],
    "ba_stats": [],
    "vggt_summaries": [],
    "custom_gs_metrics": [],
}
for path in sorted(root.rglob("results.json")):
    try:
        summary["official_results"].append({"path": str(path), "data": json.loads(path.read_text())})
    except Exception as exc:
        summary["official_results"].append({"path": str(path), "error": repr(exc)})
for path in sorted(root.rglob("ba_stats.json")):
    try:
        summary["ba_stats"].append({"path": str(path), "data": json.loads(path.read_text())})
    except Exception as exc:
        summary["ba_stats"].append({"path": str(path), "error": repr(exc)})
for path in sorted(root.rglob("summary.json")):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        data = {"error": repr(exc)}
    if data.get("stage") == "vggt_export":
        summary["vggt_summaries"].append({"path": str(path), "data": data})
for path in sorted(root.rglob("metrics.json")):
    try:
        summary["custom_gs_metrics"].append({"path": str(path), "data": json.loads(path.read_text())})
    except Exception as exc:
        summary["custom_gs_metrics"].append({"path": str(path), "error": repr(exc)})
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(out)
PY
}

cd "$PROJECT_ROOT" || exit 1
require_dir "$PROJECT_ROOT/data/scene/images"
require_dir "$PROJECT_ROOT/data/scene_32/images"
require_dir "$PROJECT_ROOT/data/1-human/images"
require_dir "$PROJECT_ROOT/data/1-human/masks"
require_dir "$PROJECT_ROOT/data/2-human/images"
require_dir "$PROJECT_ROOT/data/2-human/masks"
require_file "$PROJECT_ROOT/data/raw/3_scene.mp4"
require_file "$PROJECT_ROOT/.venv/bin/python"
require_file "$PROJECT_ROOT/.venv_3dgs/bin/python"

SCENE_DIR="$PROJECT_ROOT/data/scene"
SCENE_IMG="$SCENE_DIR/images"
SCENE32_DIR="$PROJECT_ROOT/data/scene_32"
SCENE32_IMG="$SCENE32_DIR/images"
H1_DIR="$PROJECT_ROOT/data/1-human"
H2_DIR="$PROJECT_ROOT/data/2-human"
SEL_SCENE="$OUT_ROOT/17_i1_selected_final/scene_selected"

SCENE_VGGT="$OUT_ROOT/01_scene_vggt_raw_no_filter"
SCENE_BA="$OUT_ROOT/02_scene_custom_ba"
SCENE_GS_RAW="$OUT_ROOT/03_scene_official_raw_sparse"
SCENE_GS_BA="$OUT_ROOT/04_scene_official_ba_sparse"
SCENE_GS_RAND_RAW="$OUT_ROOT/05_scene_random_init_raw_camera"
SCENE_GS_RAND_BA="$OUT_ROOT/06_scene_random_init_ba_camera"
SCENE_CUSTOM_GS="$OUT_ROOT/13_scene_custom_3dgs"

run_cmd "01" "scene_vggt_raw_no_filter" "$(vggt_cmd "$SCENE_DIR" "$SCENE_VGGT" "all" "MAX_QUERY_PTS=512 QUERY_FRAME_NUM=12 FINE_TRACKING=1")"
require_file "$SCENE_VGGT/reconstruction.npz"
log_manifest "01_scene_vggt_raw_reconstruction" "$SCENE_VGGT/reconstruction.npz" "64-frame VGGT raw, max_reproj_error=0.0"

run_cmd "02" "scene_custom_ba" "$(ba_cmd "$SCENE_VGGT/reconstruction.npz" "$SCENE_BA")"
require_file "$SCENE_BA/reconstruction.npz"
log_manifest "02_scene_ba_reconstruction" "$SCENE_BA/reconstruction.npz" "custom BA from scene raw"

run_cmd "03" "scene_official_raw_sparse" "$(official_cmd "$SCENE_VGGT/reconstruction.npz" "$SCENE_IMG" "$SCENE_GS_RAW" "reconstruction")"
log_manifest "03_scene_official_raw" "$SCENE_GS_RAW" "official 3DGS, VGGT raw sparse init"

run_cmd "04" "scene_official_ba_sparse" "$(official_cmd "$SCENE_BA/reconstruction.npz" "$SCENE_IMG" "$SCENE_GS_BA" "reconstruction")"
log_manifest "04_scene_official_ba" "$SCENE_GS_BA" "official 3DGS, custom BA sparse init"

run_cmd "05" "scene_random_raw_camera" "$(official_cmd "$SCENE_VGGT/reconstruction.npz" "$SCENE_IMG" "$SCENE_GS_RAND_RAW" "random")"
log_manifest "05_scene_random_raw" "$SCENE_GS_RAND_RAW" "official 3DGS random init, raw camera"

run_cmd "06" "scene_random_ba_camera" "$(official_cmd "$SCENE_BA/reconstruction.npz" "$SCENE_IMG" "$SCENE_GS_RAND_BA" "random")"
log_manifest "06_scene_random_ba" "$SCENE_GS_RAND_BA" "official 3DGS random init, BA camera"

run_cmd "13" "scene_custom_3dgs" "$(custom_gs_cmd "$SCENE_BA/reconstruction.npz" "$SCENE_IMG" "$SCENE_CUSTOM_GS")"
log_manifest "13_scene_custom_3dgs" "$SCENE_CUSTOM_GS" "custom 3DGS from custom BA reconstruction"

run_human_pipeline() {
  local exp_id="$1"
  local scene_name="$2"
  local scene_dir="$3"
  local qpts="$4"
  local qframes="$5"
  local label="$6"
  local base="$OUT_ROOT/${exp_id}_${scene_name}_${label}"
  local vggt="$base/vggt_raw"
  local ba="$base/ba_custom"
  local gs="$base/gs_official_ba_mask_white"

  run_cmd "$exp_id-a" "${scene_name}_${label}_vggt" "$(vggt_cmd "$scene_dir" "$vggt" "all" "MAX_QUERY_PTS=$qpts QUERY_FRAME_NUM=$qframes FINE_TRACKING=1")"
  require_file "$vggt/reconstruction.npz"
  run_cmd "$exp_id-b" "${scene_name}_${label}_ba" "$(ba_cmd "$vggt/reconstruction.npz" "$ba")"
  require_file "$ba/reconstruction.npz"
  run_cmd "$exp_id-c" "${scene_name}_${label}_official_ba_mask_white" "$(official_cmd "$ba/reconstruction.npz" "$scene_dir/images" "$gs" "reconstruction" "MASK_DIR=$scene_dir/masks MASK_BACKGROUND=white WHITE_BACKGROUND=1")"
  log_manifest "${exp_id}_${scene_name}_${label}_vggt" "$vggt/reconstruction.npz" "human VGGT $label"
  log_manifest "${exp_id}_${scene_name}_${label}_ba" "$ba/reconstruction.npz" "human BA $label"
  log_manifest "${exp_id}_${scene_name}_${label}_gs" "$gs" "human official 3DGS mask-white $label"
}

run_human_pipeline "07" "1-human" "$H1_DIR" 1024 16 "main_hq"
run_human_pipeline "08" "2-human" "$H2_DIR" 1024 16 "main_hq"
run_human_pipeline "09" "1-human" "$H1_DIR" 512 8 "low_tracks"
run_human_pipeline "10" "1-human" "$H1_DIR" 1024 16 "high_tracks"
run_human_pipeline "11" "2-human" "$H2_DIR" 512 8 "low_tracks"
run_human_pipeline "12" "2-human" "$H2_DIR" 1024 16 "high_tracks"

SCENE32_VGGT="$OUT_ROOT/14_scene32_raw_hq/vggt_raw"
SCENE32_GS_RAW="$OUT_ROOT/14_scene32_raw_hq/gs_official_raw"
SCENE32_BA="$OUT_ROOT/15_scene32_ba_hq/ba_custom"
SCENE32_GS_BA="$OUT_ROOT/15_scene32_ba_hq/gs_official_ba"
run_cmd "14-a" "scene32_vggt_raw_hq" "$(vggt_cmd "$SCENE32_DIR" "$SCENE32_VGGT" "all" "MAX_QUERY_PTS=768 QUERY_FRAME_NUM=16 FINE_TRACKING=1")"
require_file "$SCENE32_VGGT/reconstruction.npz"
run_cmd "14-b" "scene32_official_raw_hq" "$(official_cmd "$SCENE32_VGGT/reconstruction.npz" "$SCENE32_IMG" "$SCENE32_GS_RAW" "reconstruction")"
run_cmd "15-a" "scene32_ba_hq" "$(ba_cmd "$SCENE32_VGGT/reconstruction.npz" "$SCENE32_BA")"
require_file "$SCENE32_BA/reconstruction.npz"
run_cmd "15-b" "scene32_official_ba_hq" "$(official_cmd "$SCENE32_BA/reconstruction.npz" "$SCENE32_IMG" "$SCENE32_GS_BA" "reconstruction")"
log_manifest "14_scene32_raw_hq" "$SCENE32_GS_RAW" "32-frame raw HQ official 3DGS"
log_manifest "15_scene32_ba_hq" "$SCENE32_GS_BA" "32-frame BA HQ official 3DGS"

UNIFORM_DIR="$OUT_ROOT/16_i1_uniform_baseline"
write_pointer_dir "$UNIFORM_DIR" \
  "scene_vggt=$SCENE_VGGT" \
  "scene_ba=$SCENE_BA" \
  "scene_gs_raw=$SCENE_GS_RAW" \
  "scene_gs_ba=$SCENE_GS_BA"
log_manifest "16_i1_uniform_baseline" "$UNIFORM_DIR" "pointers to rerun uniform scene baseline"

run_cmd "17-a" "i1_video_select" "$(video_select_cmd "$SEL_SCENE")"
require_dir "$SEL_SCENE/images"
SEL_VGGT="$OUT_ROOT/17_i1_selected_final/vggt_raw_dense"
SEL_BA="$OUT_ROOT/17_i1_selected_final/ba_custom"
run_cmd "17-b" "i1_selected_vggt_ba_dense_export" "$(vggt_cmd "$SEL_SCENE" "$SEL_VGGT" "all" "ENABLE_POINT_HEAD=1 INIT_POINTS_SOURCE=depth MAX_QUERY_PTS=512 QUERY_FRAME_NUM=12 FINE_TRACKING=1 MAX_DENSE_POINTS=200000 DENSE_RECONSTRUCTION_VARIANTS=depth_only,pointmap_only,disagreement_only,reprojection_only,filtered_full")"
require_file "$SEL_VGGT/reconstruction.npz"
run_cmd "17-c" "i1_selected_ba" "$(ba_cmd "$SEL_VGGT/reconstruction.npz" "$SEL_BA")"
require_file "$SEL_BA/reconstruction.npz"
log_manifest "17_i1_selected_scene" "$SEL_SCENE" "selected 64-frame scene"
log_manifest "17_i1_selected_vggt" "$SEL_VGGT/reconstruction.npz" "selected VGGT sparse"
log_manifest "17_i1_selected_ba" "$SEL_BA/reconstruction.npz" "selected custom BA"

SEL_GS_RAW="$OUT_ROOT/18_i2_sparse_track_baseline_gs"
SEL_GS_BA="$OUT_ROOT/19_i2_sparse_ba_baseline_gs"
run_cmd "18" "i2_sparse_track_baseline" "$(official_cmd "$SEL_VGGT/reconstruction.npz" "$SEL_SCENE/images" "$SEL_GS_RAW" "reconstruction")"
run_cmd "19" "i2_sparse_ba_baseline" "$(official_cmd "$SEL_BA/reconstruction.npz" "$SEL_SCENE/images" "$SEL_GS_BA" "reconstruction")"
log_manifest "18_i2_sparse_track_baseline" "$SEL_GS_RAW" "selected sparse track official 3DGS"
log_manifest "19_i2_sparse_ba_baseline" "$SEL_GS_BA" "selected sparse BA official 3DGS"

for item in \
  "20 depth_only" \
  "21 pointmap_only" \
  "22 disagreement_only" \
  "23 reprojection_only" \
  "24 filtered_full"
do
  set -- $item
  exp_id="$1"
  variant="$2"
  recon="$SEL_VGGT/reconstruction_dense_${variant}.npz"
  out="$OUT_ROOT/${exp_id}_i2_${variant}_gs"
  require_file "$recon"
  run_cmd "$exp_id" "i2_dense_${variant}" "$(official_cmd "$recon" "$SEL_SCENE/images" "$out" "reconstruction")"
  log_manifest "${exp_id}_i2_${variant}" "$out" "selected dense ablation $variant"
done

FULL_DIR="$OUT_ROOT/25_full_method_comparison"
write_pointer_dir "$FULL_DIR" \
  "uniform_raw_sparse=$SCENE_GS_RAW" \
  "uniform_ba_sparse=$SCENE_GS_BA" \
  "selected_raw_sparse=$SEL_GS_RAW" \
  "selected_ba_sparse=$SEL_GS_BA" \
  "selected_filtered_dense=$OUT_ROOT/24_i2_filtered_full_gs"
log_manifest "25_full_method_comparison" "$FULL_DIR" "pointers to five full-method comparison runs"

run_cmd "99" "collect_summary" "cd $(printf %q "$PROJECT_ROOT") && source .venv/bin/activate && $(declare -f collect_summary); OUT_ROOT=$(printf %q "$OUT_ROOT") SUMMARY_JSON=$(printf %q "$SUMMARY_JSON") PROJECT_ROOT=$(printf %q "$PROJECT_ROOT") collect_summary"

echo "BATCH_COMPLETE $BATCH_ID"
echo "OUT_ROOT=$OUT_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
