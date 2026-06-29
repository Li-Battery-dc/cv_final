#!/usr/bin/env python3
"""VGGT initialization pipeline: export unified reconstruction + COLMAP format.

Runs VGGT inference on scene images, predicts tracks via VGGSfM tracker,
filters observations, and saves both .npz reconstruction and COLMAP sparse model.

Usage:
    python -m src.vggt_export \
        --scene_dir data/scene \
        --output_dir outputs/vggt_raw

    # With custom thresholds:
    python -m src.vggt_export \
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
import gc

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
from src.data.mask_filter import filter_reconstruction_by_masks
from src.data.colmap_io import reconstruction_to_colmap_space, reconstruction_to_colmap_sparse
from src.improvement.geometry_filter import (
    build_filtered_dense_reconstruction,
    save_dense_ply,
)
from src.utils.experiment import (
    prepare_output_dir,
    save_json,
    save_run_metadata,
    utc_timestamp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Export Pipeline")
    parser.add_argument("--scene_dir", type=str, default="data/scene",
                        help="Directory containing images/ subfolder")
    parser.add_argument("--output_dir", type=str, default="outputs/vggt_raw",
                        help="Output root directory for reconstruction")
    parser.add_argument("--output_run_dir", type=str, default=None,
                        help="Optional explicit output run directory")
    parser.add_argument("--use_timestamp", action=argparse.BooleanOptionalAction, default=True,
                        help="If true, save to output_dir/runs/<timestamp>_vggt_export")
    parser.add_argument("--stage", type=str, default="all", choices=("all", "vggt", "tracks"),
                        help="Run full pipeline, VGGT-only cache export, or tracks/reconstruction from cache.")
    parser.add_argument("--vggt_cache", type=str, default=None,
                        help="Path to VGGT cache .npz. Defaults to output_dir/vggt_predictions.npz")
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
    parser.add_argument("--fine_tracking", action=argparse.BooleanOptionalAction, default=True,
                        help="Use fine (slower) tracking")
    parser.add_argument("--camera_type", type=str, default="PINHOLE",
                        help="COLMAP camera type: PINHOLE or SIMPLE_PINHOLE")
    parser.add_argument("--conf_thres_value", type=float, default=5.0,
                        help="Depth confidence threshold (used for dense point cloud)")
    parser.add_argument("--max_dense_points", type=int, default=100000,
                        help="Max dense points for .ply export")
    parser.add_argument("--enable_point_head", action="store_true", default=False,
                        help="Also run VGGT direct point-map head for geometry consistency checks")
    parser.add_argument("--save_dense_filtered_reconstruction", action="store_true", default=False,
                        help="Save a depth-camera dense Reconstruction filtered by point-map and reprojection consistency")
    parser.add_argument("--dense_filter_disagreement_percentile", type=float, default=70.0,
                        help="Keep depth/point-map disagreement below this scene percentile")
    parser.add_argument("--dense_filter_reproj_percentile", type=float, default=70.0,
                        help="Keep neighbor reprojection depth error below this scene percentile")
    parser.add_argument("--dense_filter_min_votes", type=int, default=1,
                        help="Minimum passing neighbor views for dense point retention")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Optional foreground masks aligned with scene images")
    parser.add_argument("--mask_foreground_threshold", type=float, default=0.5,
                        help="Mask value threshold for foreground observations")
    parser.add_argument("--mask_min_observations", type=int, default=2,
                        help="Minimum foreground observations to keep a 3D point")
    parser.add_argument("--mask_min_ratio", type=float, default=0.5,
                        help="Minimum foreground observation ratio to keep a 3D point")
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


def run_vggt_inference(model, images, dtype, device, vggt_resolution=518,
                       return_frame_features=False, return_point_map=False):
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
        depth_map: (S, H_vggt, W_vggt, 1) numpy.
        depth_conf: (S, H_vggt, W_vggt) numpy.
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

        frame_features = None
        if return_frame_features:
            last_tokens = aggregated_tokens_list[-1]
            patch_tokens = last_tokens[:, :, ps_idx:]
            frame_features = patch_tokens.mean(dim=2)
            frame_features = F.normalize(frame_features.float(), dim=-1)

        point_map = None
        point_conf = None
        if return_point_map:
            if model.point_head is None:
                raise ValueError("VGGT model was created without point_head")
            point_map, point_conf = model.point_head(aggregated_tokens_list, images_batch, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    if frame_features is not None:
        frame_features = frame_features.squeeze(0).cpu().numpy()
    if point_map is not None:
        point_map = point_map.squeeze(0).cpu().numpy()
    if point_conf is not None:
        point_conf = point_conf.squeeze(0).cpu().numpy()

    return extrinsic, intrinsic, depth_map, depth_conf, frame_features, point_map, point_conf


def compute_image_size_hw(original_coords_np: np.ndarray) -> np.ndarray:
    """Convert VGGT original_coords to image sizes in [height, width] order."""
    image_size_hw = np.zeros((len(original_coords_np), 2), dtype=np.int32)
    for s in range(len(original_coords_np)):
        # original_coords_np[s] = [x1, y1, x2, y2, width, height]
        image_size_hw[s] = original_coords_np[s, -2:][::-1]
    return image_size_hw


def save_dense_point_cloud(images, points_3d_dense, depth_conf, args, output_dir):
    """Save a confidence-filtered dense VGGT point cloud for quick visualization."""
    import trimesh
    from vggt.utils.helper import randomly_limit_trues

    conf_mask = depth_conf >= args.conf_thres_value
    conf_mask = randomly_limit_trues(conf_mask, args.max_dense_points)

    pts_dense = points_3d_dense[conf_mask]
    images_vggt = F.interpolate(images, size=(args.vggt_resolution, args.vggt_resolution),
                                mode="bilinear", align_corners=False)
    images_np = (images_vggt.cpu().numpy() * 255).astype(np.uint8)
    images_np = images_np.transpose(0, 2, 3, 1)
    pts_rgb_dense = images_np[conf_mask]

    ply_path = os.path.join(output_dir, "points3d_dense.ply")
    trimesh.PointCloud(pts_dense, colors=pts_rgb_dense).export(ply_path)
    print(f"Saved dense point cloud ({len(pts_dense)} pts) to {ply_path}")
    return ply_path, len(pts_dense)


def save_vggt_cache(path, image_names, image_size_hw, original_coords_np,
                    extrinsic, intrinsic, depth_map, depth_conf, points_3d_dense,
                    args, frame_features=None, point_map=None, point_conf=None):
    """Save VGGT-only outputs so VGGSfM tracking can be run later."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(
        image_names=np.array(image_names, dtype=str),
        image_size_hw=image_size_hw.astype(np.int32),
        original_coords=original_coords_np,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth_map=depth_map,
        depth_conf=depth_conf,
        points_3d_dense=points_3d_dense,
        img_load_resolution=np.array(args.img_load_resolution, dtype=np.int32),
        vggt_resolution=np.array(args.vggt_resolution, dtype=np.int32),
    )
    if frame_features is not None:
        payload["frame_features"] = frame_features
    if point_map is not None:
        payload["point_map"] = point_map
    if point_conf is not None:
        payload["point_conf"] = point_conf
    np.savez_compressed(path, **payload)
    print(f"Saved VGGT cache to {path}")


def load_vggt_cache(path):
    """Load VGGT-only outputs saved by --stage vggt."""
    data = np.load(path, allow_pickle=True)
    return {
        "image_names": data["image_names"].astype(str).tolist(),
        "image_size_hw": data["image_size_hw"],
        "original_coords": data["original_coords"],
        "extrinsic": data["extrinsic"],
        "intrinsic": data["intrinsic"],
        "depth_map": data["depth_map"],
        "depth_conf": data["depth_conf"],
        "points_3d_dense": data["points_3d_dense"],
        "frame_features": data["frame_features"] if "frame_features" in data else None,
        "point_map": data["point_map"] if "point_map" in data else None,
        "point_conf": data["point_conf"] if "point_conf" in data else None,
        "img_load_resolution": int(data["img_load_resolution"]),
        "vggt_resolution": int(data["vggt_resolution"]),
    }


def export_reconstruction_outputs(recon, original_coords_np, img_load_resolution,
                                  output_dir, camera_type):
    """Save Reconstruction .npz and COLMAP sparse model."""
    npz_path = os.path.join(output_dir, "reconstruction.npz")
    recon.to_npz(npz_path)
    print(f"Saved reconstruction to {npz_path}")

    recon_colmap = reconstruction_to_colmap_space(recon)
    sparse_dir = os.path.join(output_dir, "sparse")
    reconstruction_to_colmap_sparse(recon_colmap, sparse_dir, camera_type)
    print(f"Saved COLMAP model to {sparse_dir}")
    return npz_path, sparse_dir


def main():
    args = parse_args()
    set_seed(args.seed)
    stage_times = {}
    dense_count = None
    dense_filtered_count = None
    dense_filtered_npz_path = None
    dense_filtered_ply_path = None
    dense_filter_stats = None
    ply_path = None
    npz_path = None
    sparse_dir = None

    # Setup
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Dtype: {dtype}")

    scene_dir = os.path.abspath(args.scene_dir)
    image_dir = os.path.join(scene_dir, "images")
    output_dir = prepare_output_dir(
        output_root=args.output_dir,
        stage_name="vggt_export",
        explicit_output_dir=args.output_run_dir,
        use_timestamp=args.use_timestamp,
    )
    vggt_cache = os.path.abspath(args.vggt_cache or os.path.join(output_dir, "vggt_predictions.npz"))

    # Load images
    import glob as _glob
    image_paths = sorted(_glob.glob(os.path.join(image_dir, "*")))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    image_names = [os.path.basename(p) for p in image_paths]
    print(f"Found {len(image_paths)} images in {image_dir}")
    config_path = save_run_metadata(
        output_dir,
        stage="vggt_export",
        params=vars(args),
        inputs={
            "scene_dir": scene_dir,
            "image_dir": image_dir,
            "image_count": len(image_paths),
            "mask_dir": os.path.abspath(args.mask_dir) if args.mask_dir else None,
        },
        outputs={"output_dir": output_dir, "vggt_cache": vggt_cache},
    )
    print(f"Saved run config: {config_path}")

    # Load and preprocess (square padding + resize)
    images, original_coords = load_and_preprocess_images_square(image_paths, args.img_load_resolution)
    images = images.to(device)
    original_coords_np = original_coords.cpu().numpy()
    S = images.shape[0]
    print(f"Loaded images: shape={images.shape}")

    if args.stage in ("all", "vggt"):
        # ---- Step 1: VGGT Inference ----
        print("Running VGGT inference...")
        t0 = time.time()
        model = load_vggt_model(device, dtype)
        extrinsic, intrinsic, depth_map, depth_conf, frame_features, point_map, point_conf = run_vggt_inference(
            model=model,
            images=images, dtype=dtype, device=device,
            vggt_resolution=args.vggt_resolution,
            return_frame_features=True,
            return_point_map=args.enable_point_head or args.save_dense_filtered_reconstruction,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stage_times['vggt_inference_seconds'] = time.time() - t0
        print(f"VGGT inference done in {stage_times['vggt_inference_seconds']:.1f}s")

        points_3d_dense = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        image_size_hw = compute_image_size_hw(original_coords_np)
        save_vggt_cache(
            vggt_cache, image_names, image_size_hw, original_coords_np,
            extrinsic, intrinsic, depth_map, depth_conf, points_3d_dense, args,
            frame_features=frame_features, point_map=point_map, point_conf=point_conf,
        )
        ply_path, dense_count = save_dense_point_cloud(
            images, points_3d_dense, depth_conf, args, output_dir,
        )

        if args.stage == "vggt":
            print("\n" + "=" * 60)
            print("VGGT Stage Summary")
            print("=" * 60)
            print(f"  Images:           {S}")
            print(f"  Dense points:     {dense_count}")
            print(f"  VGGT cache:       {vggt_cache}")
            print(f"  Output PLY:       {ply_path}")
            print("=" * 60)
            save_json(os.path.join(output_dir, "summary.json"), {
                "stage": args.stage,
                "timestamp_utc": utc_timestamp(),
                "images": S,
                "dense_points": dense_count,
                "vggt_cache": vggt_cache,
                "output_ply": ply_path,
                **stage_times,
            })
            return 0
    else:
        # ---- Load cached VGGT outputs ----
        print(f"Loading VGGT cache from {vggt_cache}")
        cache = load_vggt_cache(vggt_cache)
        cached_names = cache["image_names"]
        if cached_names != image_names:
            raise ValueError("Image list differs from VGGT cache; use the same scene/images ordering.")
        if cache["img_load_resolution"] != args.img_load_resolution:
            raise ValueError(
                f"VGGT cache was made with img_load_resolution={cache['img_load_resolution']}, "
                f"but current args use {args.img_load_resolution}. Re-run with matching resolution."
            )
        if cache["vggt_resolution"] != args.vggt_resolution:
            raise ValueError(
                f"VGGT cache was made with vggt_resolution={cache['vggt_resolution']}, "
                f"but current args use {args.vggt_resolution}. Re-run with matching resolution."
            )
        image_size_hw = cache["image_size_hw"]
        original_coords_np = cache["original_coords"]
        extrinsic = cache["extrinsic"]
        intrinsic = cache["intrinsic"]
        depth_conf = cache["depth_conf"]
        depth_map = cache["depth_map"]
        points_3d_dense = cache["points_3d_dense"]
        frame_features = cache.get("frame_features")
        point_map = cache.get("point_map")
        point_conf = cache.get("point_conf")
        if args.save_dense_filtered_reconstruction and point_map is None:
            raise ValueError(
                "VGGT cache has no point_map. Re-run --stage vggt/all with --enable_point_head "
                "before requesting --save_dense_filtered_reconstruction."
            )

    # ---- Step 2: Track Prediction ----
    print("Running track prediction...")
    t0 = time.time()
    with torch.no_grad():
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
    stage_times['track_prediction_seconds'] = time.time() - t0
    print(f"Track prediction done in {stage_times['track_prediction_seconds']:.1f}s")
    print(f"Tracks: {pred_tracks.shape}, points: {points_3d_track.shape}")

    # Scale intrinsics from VGGT resolution to image load resolution
    scale = args.img_load_resolution / args.vggt_resolution
    intrinsic_scaled = intrinsic.copy()
    intrinsic_scaled[:, :2, :] *= scale

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

    if args.save_dense_filtered_reconstruction:
        if point_map is None:
            raise ValueError("--save_dense_filtered_reconstruction requires --enable_point_head")
        print("Building filtered dense depth-camera reconstruction...")
        t0 = time.time()
        images_vggt = F.interpolate(images, size=(args.vggt_resolution, args.vggt_resolution),
                                    mode="bilinear", align_corners=False)
        images_np = (images_vggt.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        images_np = images_np.transpose(0, 2, 3, 1)
        dense_result = build_filtered_dense_reconstruction(
            image_names=image_names,
            image_size_hw=image_size_hw,
            intrinsics=intrinsic,
            extrinsics=extrinsic,
            depth_map=depth_map,
            depth_conf=depth_conf,
            points_depth=points_3d_dense,
            points_point=point_map,
            images_np=images_np,
            disagreement_percentile=args.dense_filter_disagreement_percentile,
            reproj_percentile=args.dense_filter_reproj_percentile,
            min_reproj_votes=args.dense_filter_min_votes,
            max_points=args.max_dense_points,
            rng=np.random.default_rng(args.seed),
            metadata={
                "original_coords": original_coords_np,
                "img_load_resolution": args.img_load_resolution,
                "vggt_resolution": args.vggt_resolution,
                "vggt_cache": vggt_cache,
                "run_stage": "vggt_export_dense_filtered",
                "run_timestamp_utc": utc_timestamp(),
            },
        )
        dense_result.reconstruction.intrinsics = intrinsic_scaled.copy()
        dense_result.reconstruction.metadata["filter_intrinsics_space"] = "vggt_resolution"
        dense_result.reconstruction.metadata["output_intrinsics_space"] = "img_load_resolution"
        dense_filtered_npz_path = os.path.join(output_dir, "reconstruction_dense_filtered.npz")
        dense_result.reconstruction.to_npz(dense_filtered_npz_path)
        dense_filtered_ply_path = os.path.join(output_dir, "points3d_dense_filtered.ply")
        save_dense_ply(
            dense_filtered_ply_path,
            dense_result.reconstruction.points3d,
            dense_result.reconstruction.points_rgb,
        )
        dense_filter_stats = dense_result.stats
        dense_filtered_count = dense_result.reconstruction.num_points
        save_json(os.path.join(output_dir, "geometry_filter_stats.json"), dense_filter_stats)
        stage_times['dense_geometry_filter_seconds'] = time.time() - t0
        print(
            f"Filtered dense reconstruction: {dense_filtered_count} pts -> "
            f"{dense_filtered_npz_path}"
        )

    mask_filter_stats = None
    if args.mask_dir:
        print(f"Filtering reconstruction with masks from {args.mask_dir}")
        recon, mask_filter_stats = filter_reconstruction_by_masks(
            recon,
            mask_dir=args.mask_dir,
            threshold=args.mask_foreground_threshold,
            min_observations=args.mask_min_observations,
            min_ratio=args.mask_min_ratio,
        )
        print(f"Mask-filtered reconstruction: {recon}")

    recon.metadata['vis_thresh'] = args.vis_thresh
    recon.metadata['max_reproj_error'] = args.max_reproj_error
    recon.metadata['min_visible_frames'] = args.min_visible_frames
    recon.metadata['vggt_resolution'] = args.vggt_resolution
    recon.metadata['img_load_resolution'] = args.img_load_resolution
    recon.metadata['original_coords'] = original_coords_np
    recon.metadata['vggt_cache'] = vggt_cache
    if frame_features is not None:
        recon.metadata['frame_features_in_cache'] = True
    if point_map is not None:
        recon.metadata['point_map_in_cache'] = True
    if dense_filtered_npz_path:
        recon.metadata['dense_filtered_reconstruction'] = dense_filtered_npz_path
    recon.metadata['run_stage'] = 'vggt_export'
    recon.metadata['run_timestamp_utc'] = utc_timestamp()
    recon.metadata['run_config_path'] = config_path

    npz_path, sparse_dir = export_reconstruction_outputs(
        recon, original_coords_np, args.img_load_resolution, output_dir, args.camera_type,
    )

    # ---- Step 4: Summary ----
    print("\n" + "=" * 60)
    print("VGGT Export Summary")
    print("=" * 60)
    print(f"  Images:           {S}")
    print(f"  Track points:     {recon.num_points}")
    print(f"  Observations:     {recon.num_observations}")
    print(f"  VGGT cache:       {vggt_cache}")
    if dense_filtered_npz_path:
        print(f"  Dense filtered:   {dense_filtered_count} pts")
    print(f"  Output .npz:      {npz_path}")
    print(f"  Output COLMAP:    {sparse_dir}")
    print("=" * 60)
    save_json(os.path.join(output_dir, "summary.json"), {
        "stage": args.stage,
        "timestamp_utc": utc_timestamp(),
        "images": S,
        "track_points": recon.num_points,
        "observations": recon.num_observations,
        "vggt_cache": vggt_cache,
        "output_npz": npz_path,
        "output_colmap": sparse_dir,
        "dense_filtered_npz": dense_filtered_npz_path,
        "dense_filtered_ply": dense_filtered_ply_path,
        "dense_filtered_points": dense_filtered_count,
        "dense_filter": dense_filter_stats,
        "mask_filter": mask_filter_stats,
        **stage_times,
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
