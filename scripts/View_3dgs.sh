#!/usr/bin/env bash


# on local wsl. 
MODEL="data/gs_official_dense_filtered_full/runs/20260629_152158_gaussian_official"
VIEWER="gaussian-splatting/SIBR_viewers/install/bin/SIBR_gaussianViewer_app"

MESA_GL_VERSION_OVERRIDE=4.5 \
MESA_GLSL_VERSION_OVERRIDE=450 \
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
GALLIUM_DRIVER=d3d12 \
MESA_LOADER_DRIVER_OVERRIDE=d3d12 \
"$VIEWER" -m "$MODEL"