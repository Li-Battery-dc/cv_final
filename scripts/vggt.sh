#!/usr/bin/env bash
set -e
# Project root, usually the parent directory of vggt/
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Lesson_Project/cv_final}"

# VGGT repo path
VGGT_DIR="${VGGT_DIR:-$PROJECT_ROOT/vggt}"

# Python virtual environment path
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

# Scene directory.
# It should contain:
#   SCENE_DIR/images/*.jpg
SCENE_DIR="${SCENE_DIR:-$PROJECT_ROOT/data/scene}"

# Whether to run viser visualization first: 1 or 0
RUN_VISER="${RUN_VISER:-0}"

# Whether to export COLMAP sparse files: 1 or 0
RUN_COLMAP="${RUN_COLMAP:-1}"

# Whether to use VGGT built-in BA in demo_colmap.py: 1 or 0
# For your homework, keep this 0 initially. Later you can set it to 1 as a reference baseline.
USE_BUILTIN_BA="${USE_BUILTIN_BA:-0}"

# Lightweight BA parameters for VGGT built-in BA.
# Only used when USE_BUILTIN_BA=1.
MAX_QUERY_PTS="${MAX_QUERY_PTS:-2048}"
QUERY_FRAME_NUM="${QUERY_FRAME_NUM:-5}"

# Output backup directory name after COLMAP export
BACKUP_NAME="${BACKUP_NAME:-sparse_vggt_raw}"

source "$VENV_PATH/bin/activate"
cd "$VGGT_DIR"

# viser visualization
if [ "$RUN_VISER" = "1" ]; then
    echo "[INFO] Running VGGT viser demo..."
    python demo_viser.py --image_folder "$SCENE_DIR/images"
else
    echo "[INFO] Skip viser demo."
fi

# colmap export
if [ "$RUN_COLMAP" = "1" ]; then
    echo "[INFO] Running VGGT COLMAP export..."

    if [ "$USE_BUILTIN_BA" = "1" ]; then
        echo "[INFO] Using VGGT built-in BA with max_query_pts=$MAX_QUERY_PTS, query_frame_num=$QUERY_FRAME_NUM"
        python demo_colmap.py \
            --scene_dir="$SCENE_DIR" \
            --use_ba \
            --max_query_pts="$MAX_QUERY_PTS" \
            --query_frame_num="$QUERY_FRAME_NUM"
    else
        echo "[INFO] Exporting feed-forward VGGT reconstruction without built-in BA."
        python demo_colmap.py \
            --scene_dir="$SCENE_DIR"
    fi

    echo "[INFO] Checking exported sparse directory..."
    if [ -d "$SCENE_DIR/sparse" ]; then
        ls -lh "$SCENE_DIR/sparse"

        echo "[INFO] Backing up sparse to $SCENE_DIR/$BACKUP_NAME"
        rm -rf "$SCENE_DIR/$BACKUP_NAME"
        cp -r "$SCENE_DIR/sparse" "$SCENE_DIR/$BACKUP_NAME"

        echo "[INFO] Export completed."
        echo "[INFO] COLMAP files:"
        echo "  $SCENE_DIR/$BACKUP_NAME/cameras.bin"
        echo "  $SCENE_DIR/$BACKUP_NAME/images.bin"
        echo "  $SCENE_DIR/$BACKUP_NAME/points3D.bin"
    else
        echo "[ERROR] COLMAP export failed: $SCENE_DIR/sparse not found."
        exit 1
    fi
else
    echo "[INFO] Skip COLMAP export."
fi