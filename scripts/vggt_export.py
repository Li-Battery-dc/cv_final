#!/usr/bin/env python3
"""VGGT initialization pipeline: export unified reconstruction + COLMAP format.

Runs VGGT inference on scene images, predicts tracks via VGGSfM tracker,
filters observations, and saves both .npz reconstruction and COLMAP sparse model.

Usage:
    python scripts/vggt_export.py \
        --scene_dir data/scene \
        --output_dir outputs/vggt_raw

    # With custom thresholds:
    python scripts/vggt_export.py \
        --scene_dir data/scene \
        --output_dir outputs/vggt_raw \
        --max_query_pts 2048 \
        --query_frame_num 5 \
        --vis_thresh 0.2 \
        --max_reproj_error 8.0 \
        --min_visible_frames 3
"""

import os
import sys
import random
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

# Add project root and vggt to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_VGGT_PATH = os.path.join(_PROJECT_ROOT, 'vggt')
if _VGGT_PATH not in sys.path:
    sys.path.insert(0, _VGGT_PATH)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.projection import project_3D_points_np

from src.data.reconstruction import Reconstruction
from src.data.colmap_io import reconstruction_to_colmap_sparse


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Export Pipeline")
    parser.add_argument("--scene_dir", type=str, default="data/scene",
                        help="Directory containing images/ subfolder")
    parser.add_argument("--output_dir", type=str, default="outputs/vggt_raw",
                        help="Output directory for reconstruction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_load_resolution", type=int, default=1024,
                        help="Resolution for loading images (square)")
    parser.add_argument("--vggt_resolution", type=int, default=518,
                        help="Resolution for VGGT inference")
    parser.add_argument("--max_query_pts", type=int, default=2048,
                        help="Max keypoints per query frame")
    parser.add_argument("--query_frame_num", type=int, default=5,
                        help="Number of query frames")
    parser.add_argument("--vis_thresh", type=float, default=0.2,
                        help="Visibility threshold for track filtering")
    parser.add_argument("--max_reproj_error", type=float, default=8.0,
                        help="Max reprojection error for BA inlier filtering")
    parser.add_argument("--min_visible_frames", type=int, default=3,
                        help="Min frames a point must be visible in")
    parser.add_argument("--fine_tracking", action="store_true", default=True,
                        help="Use fine (slower) tracking")
    parser.add_argument("--camera_type", type=str, default="PINHOLE",
                        help="COLMAP camera type: PINHOLE or SIMPLE_PINHOLE")
    parser.add_argument("--conf_thres_value", type=float, default=5.0,
                        help="Depth confidence threshold (used for dense point cloud)")
    parser.add_argument("--max_dense_points", type=int, default=100000,
                        help="Max dense points for .ply export")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_vggt_model(device, dtype):
    """Load VGGT model from HuggingFace."""
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    return model


def run_vggt_inference(model, images, dtype, device, vggt_resolution=518):
    """Run VGGT: camera pose + depth estimation.

    Args:
        model: VGGT model.
        images: (S, 3, H, W) tensor at img_load_resolution.
        dtype: torch dtype for autocast.
        device: torch device.
        vggt_resolution: inference resolution (default 518).

    Returns:
        extrinsic: (S, 3, 4) numpy camera-from-world.
        intrinsic: (S, 3, 3) numpy intrinsics (at vggt_resolution).
        depth_map: (S, 1, H_vggt, W_vggt) numpy.
        depth_conf: (S, 1, H_vggt, W_vggt) numpy.
    """
    images_resized = F.interpolate(images, size=(vggt_resolution, vggt_resolution),
                                   mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = images_resized[None]  # (1, S, 3, H, W)
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])

        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()

    return extrinsic, intrinsic, depth_map, depth_conf


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Dtype: {dtype}")

    scene_dir = os.path.abspath(args.scene_dir)
    image_dir = os.path.join(scene_dir, "images")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    import glob as _glob
    image_paths = sorted(_glob.glob(os.path.join(image_dir, "*")))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    image_names = [os.path.basename(p) for p in image_paths]
    print(f"Found {len(image_paths)} images in {image_dir}")

    # Load and preprocess (square padding + resize)
    images, original_coords = load_and_preprocess_images_square(image_paths, args.img_load_resolution)
    images = images.to(device)
    original_coords_np = original_coords.cpu().numpy()
    S = images.shape[0]
    print(f"Loaded images: shape={images.shape}")

    # ---- Step 1: VGGT Inference ----
    print("Running VGGT inference...")
    t0 = time.time()
    extrinsic, intrinsic, depth_map, depth_conf = run_vggt_inference(
        model=load_vggt_model(device, dtype),
        images=images, dtype=dtype, device=device,
        vggt_resolution=args.vggt_resolution,
    )
    print(f"VGGT inference done in {time.time() - t0:.1f}s")

    # Unproject depth to 3D points (per-pixel dense)
    points_3d_dense = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    # points_3d_dense: (S, H_vggt, W_vggt, 3)

    # ---- Step 2: Track Prediction ----
    print("Running track prediction...")
    t0 = time.time()
    with torch.cuda.amp.autocast(dtype=dtype):
        pred_tracks, pred_vis, pred_confs, points_3d_track, points_rgb_track = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d_dense,
            masks=None,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            fine_tracking=args.fine_tracking,
        )
        torch.cuda.empty_cache()
    print(f"Track prediction done in {time.time() - t0:.1f}s")
    print(f"Tracks: {pred_tracks.shape}, points: {points_3d_track.shape}")

    # Scale intrinsics from VGGT resolution to image load resolution
    scale = args.img_load_resolution / args.vggt_resolution
    intrinsic_scaled = intrinsic.copy()
    intrinsic_scaled[:, :2, :] *= scale

    # Compute original image sizes
    image_size_hw = np.zeros((S, 2), dtype=np.int32)
    for s in range(S):
        # original_coords_np[s] = [x1, y1, x2, y2, width, height]
        image_size_hw[s] = original_coords_np[s, -2:][::-1]  # (h, w) order

    # ---- Step 3: Build Reconstruction ----
    print("Building reconstruction with tracks...")
    recon = Reconstruction.from_tracks(
        points3d=points_3d_track,
        points_rgb=points_rgb_track,
        pred_tracks=pred_tracks,
        pred_vis=pred_vis,
        pred_confs=pred_confs,
        extrinsics=extrinsic,
        intrinsics=intrinsic_scaled,
        image_names=image_names,
        image_size_hw=image_size_hw,
        vis_thresh=args.vis_thresh,
        max_reproj_error=args.max_reproj_error,
        min_visible_frames=args.min_visible_frames,
    )
    print(f"Reconstruction: {recon}")

    recon.metadata['vis_thresh'] = args.vis_thresh
    recon.metadata['max_reproj_error'] = args.max_reproj_error
    recon.metadata['min_visible_frames'] = args.min_visible_frames
    recon.metadata['vggt_resolution'] = args.vggt_resolution
    recon.metadata['img_load_resolution'] = args.img_load_resolution
    recon.metadata['original_coords'] = original_coords_np

    # ---- Step 4: Save .npz ----
    npz_path = os.path.join(output_dir, "reconstruction.npz")
    recon.to_npz(npz_path)
    print(f"Saved reconstruction to {npz_path}")

    # ---- Step 5: Export COLMAP sparse model ----
    # Rescale 2D observations from square coordinates to original image coords
    recon_colmap = recon.copy()

    # Transform observation coordinates from square coords to original coords
    # Square coords: padded square at img_load_resolution
    # Original coords: original image in [x1, y1, x2, y2] region
    for s in range(S):
        x1, y1, x2, y2, orig_w, orig_h = original_coords_np[s]
        resize_ratio = max(orig_w, orig_h) / args.img_load_resolution

        # Update camera intrinsics
        K = recon_colmap.intrinsics[s].copy()
        K[0, 0] *= resize_ratio
        K[1, 1] *= resize_ratio
        # Principal point at original image center
        K[0, 2] = orig_w / 2.0
        K[1, 2] = orig_h / 2.0
        recon_colmap.intrinsics[s] = K

        # Shift observations from square space to original space
        cam_mask = recon_colmap.obs_camera_id == s
        if np.any(cam_mask):
            # Observations are in 1024x1024 square space
            # Need to map: square_coord -> original image coord
            # original = (square - top_left) * resize_ratio
            top_left = np.array([x1, y1])
            recon_colmap.obs_xy[cam_mask] = (recon_colmap.obs_xy[cam_mask] - top_left) * resize_ratio

    # Update image sizes
    recon_colmap.image_size_hw = image_size_hw

    # Save COLMAP
    sparse_dir = os.path.join(output_dir, "sparse")
    reconstruction_to_colmap_sparse(recon_colmap, sparse_dir, args.camera_type)
    print(f"Saved COLMAP model to {sparse_dir}")

    # ---- Step 6: Save dense point cloud (.ply) for visualization ----
    import trimesh
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues

    # Build dense point cloud from depth (conf-filtered)
    conf_mask = depth_conf >= args.conf_thres_value
    conf_mask = randomly_limit_trues(conf_mask, args.max_dense_points)

    pts_dense = points_3d_dense[conf_mask]
    # Get colors from resized images
    images_518 = F.interpolate(images, size=(args.vggt_resolution, args.vggt_resolution),
                               mode="bilinear", align_corners=False)
    images_np = (images_518.cpu().numpy() * 255).astype(np.uint8)
    images_np = images_np.transpose(0, 2, 3, 1)  # (S, H, W, 3)
    pts_rgb_dense = images_np[conf_mask]

    ply_path = os.path.join(output_dir, "points3d_dense.ply")
    trimesh.PointCloud(pts_dense, colors=pts_rgb_dense).export(ply_path)
    print(f"Saved dense point cloud ({len(pts_dense)} pts) to {ply_path}")

    # ---- Step 7: Summary ----
    print("\n" + "=" * 60)
    print("VGGT Export Summary")
    print("=" * 60)
    print(f"  Images:           {S}")
    print(f"  Track points:     {recon.num_points}")
    print(f"  Observations:     {recon.num_observations}")
    print(f"  Dense points:     {len(pts_dense)}")
    print(f"  Output .npz:      {npz_path}")
    print(f"  Output COLMAP:    {sparse_dir}")
    print(f"  Output PLY:       {ply_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
