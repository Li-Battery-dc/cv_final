#!/usr/bin/env python3
"""PyCOLMAP / Ceres bundle adjustment baseline."""

import argparse
import json
import os
import sys
import time

import numpy as np
import pyceres
import pycolmap

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ba.problem import BAProblem
from src.ba.utils import compute_reprojection_errors_from_data
from src.data.colmap_io import reconstruction_to_colmap_sparse, reconstruction_to_pycolmap
from src.data.reconstruction import Reconstruction
from src.utils.experiment import (
    prepare_output_dir,
    save_json,
    save_run_metadata,
    utc_timestamp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PyCOLMAP bundle adjustment baseline")
    parser.add_argument("--input", type=str, required=True, help="Path to input reconstruction.npz")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--output_run_dir", type=str, default=None,
                        help="Optional explicit output run directory")
    parser.add_argument("--use_timestamp", action=argparse.BooleanOptionalAction, default=True,
                        help="If true, save to output/runs/<timestamp>_ba_pycolmap")
    parser.add_argument("--camera_type", type=str, default="PINHOLE",
                        help="COLMAP camera model to use for export / conversion")
    parser.add_argument("--loss_scale", type=float, default=1.0,
                        help="Robust Cauchy loss scale")
    parser.add_argument("--max_num_iterations", type=int, default=100,
                        help="Maximum Ceres iterations per BA round")
    parser.add_argument("--outlier_threshold", type=float, default=5.0,
                        help="Outlier threshold for optional second round")
    parser.add_argument("--n_fixed_cameras", type=int, default=2,
                        help="Number of initial cameras to keep fixed")
    parser.add_argument("--refine_focal_length", action="store_true", default=False,
                        help="Allow COLMAP to refine focal length")
    parser.add_argument("--refine_principal_point", action="store_true", default=False,
                        help="Allow COLMAP to refine principal point")
    parser.add_argument("--no_outlier_removal", action="store_true", default=False,
                        help="Disable outlier filtering and the second BA round")
    parser.add_argument("--no_colmap_export", action="store_true", default=False,
                        help="Skip COLMAP sparse export")
    parser.add_argument("--print_solver_summary", action="store_true", default=False,
                        help="Let PyCOLMAP print the Ceres summary")
    return parser.parse_args()


def _build_config(reconstruction: pycolmap.Reconstruction,
                  n_fixed_cameras: int,
                  refine_focal_length: bool,
                  refine_principal_point: bool) -> pycolmap.BundleAdjustmentConfig:
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images.keys()):
        config.add_image(image_id)
    for image_id in sorted(reconstruction.images.keys())[:n_fixed_cameras]:
        config.set_constant_cam_pose(image_id)
    if not refine_focal_length and not refine_principal_point:
        for camera_id in reconstruction.cameras.keys():
            config.set_constant_cam_intrinsics(camera_id)
    return config


def _build_options(args) -> pycolmap.BundleAdjustmentOptions:
    options = pycolmap.BundleAdjustmentOptions()
    options.print_summary = args.print_solver_summary
    options.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    options.loss_function_scale = args.loss_scale
    options.refine_extrinsics = True
    options.refine_focal_length = args.refine_focal_length
    options.refine_principal_point = args.refine_principal_point
    options.refine_extra_params = False
    solver_options = options.solver_options
    solver_options.max_num_iterations = args.max_num_iterations
    solver_options.num_threads = max(1, min(os.cpu_count() or 1, 16))
    return options


def _run_pycolmap_round(recon: Reconstruction, args) -> tuple[Reconstruction, dict]:
    recon_colmap = reconstruction_to_pycolmap(recon, camera_type=args.camera_type)
    config = _build_config(
        recon_colmap,
        n_fixed_cameras=args.n_fixed_cameras,
        refine_focal_length=args.refine_focal_length,
        refine_principal_point=args.refine_principal_point,
    )
    options = _build_options(args)
    bundle_adjuster = pycolmap.BundleAdjuster(options, config)
    ok = bundle_adjuster.solve(recon_colmap)
    recon_opt = Reconstruction.from_pycolmap(
        recon_colmap,
        image_names=recon.image_names.tolist(),
        image_size_hw=recon.image_size_hw,
    )
    summary = bundle_adjuster.summary
    stats = {
        "solver_success": bool(ok),
        "ceres_brief_report": summary.BriefReport(),
        "ceres_initial_cost": float(summary.initial_cost),
        "ceres_final_cost": float(summary.final_cost),
        "ceres_num_residuals": int(summary.num_residuals),
        "ceres_num_residual_evaluations": int(summary.num_residual_evaluations),
        "ceres_num_jacobian_evaluations": int(summary.num_jacobian_evaluations),
        "ceres_num_successful_steps": int(summary.num_successful_steps),
        "ceres_num_unsuccessful_steps": int(summary.num_unsuccessful_steps),
        "ceres_termination_type": str(summary.termination_type),
    }
    return recon_opt, stats


def main():
    args = parse_args()
    output_dir = prepare_output_dir(
        output_root=args.output,
        stage_name="ba_pycolmap",
        explicit_output_dir=args.output_run_dir,
        use_timestamp=args.use_timestamp,
    )

    print(f"Loading reconstruction from: {args.input}")
    recon = Reconstruction.from_npz(args.input)
    print(f"Input: {recon}")

    config_path = save_run_metadata(
        output_dir,
        stage="ba_pycolmap",
        params=vars(args),
        inputs={"input_reconstruction": os.path.abspath(args.input)},
        outputs={"output_dir": output_dir},
    )
    print(f"Saved run config: {config_path}")

    t_start = time.time()
    stats_before = compute_reprojection_errors_from_data(
        recon.points3d, recon.extrinsics, recon.intrinsics,
        recon.obs_camera_id, recon.obs_point_id, recon.obs_xy,
    )
    print(f"Before: RMSE={stats_before['rmse']:.3f}, median={stats_before['median']:.3f}, "
          f"P90={stats_before['p90']:.3f}")

    recon_round1, solver_round1 = _run_pycolmap_round(recon, args)
    stats_round1 = compute_reprojection_errors_from_data(
        recon_round1.points3d, recon_round1.extrinsics, recon_round1.intrinsics,
        recon_round1.obs_camera_id, recon_round1.obs_point_id, recon_round1.obs_xy,
    )
    print(f"Round 1: RMSE={stats_round1['rmse']:.3f}, median={stats_round1['median']:.3f}, "
          f"P90={stats_round1['p90']:.3f}")

    if args.no_outlier_removal:
        recon_opt = recon_round1
        n_outliers_removed = 0
        solver_round2 = {}
    else:
        filtered_recon = recon_round1.filter_outlier_observations(args.outlier_threshold)
        n_outliers_removed = int(recon_round1.num_observations - filtered_recon.num_observations)
        print(f"Outliers removed for round 2: {n_outliers_removed}")
        recon_opt, solver_round2 = _run_pycolmap_round(filtered_recon, args)

    stats_after = compute_reprojection_errors_from_data(
        recon_opt.points3d, recon_opt.extrinsics, recon_opt.intrinsics,
        recon_opt.obs_camera_id, recon_opt.obs_point_id, recon_opt.obs_xy,
    )

    problem = BAProblem(recon, n_fixed_cameras=args.n_fixed_cameras, huber_delta=args.loss_scale)
    rot_deltas, trans_deltas = problem.compute_camera_deltas(recon_opt.extrinsics)

    elapsed = time.time() - t_start
    stats = {
        "rmse_before": stats_before["rmse"],
        "rmse_round1": stats_round1["rmse"],
        "rmse_after": stats_after["rmse"],
        "median_before": stats_before["median"],
        "median_after": stats_after["median"],
        "p90_before": stats_before["p90"],
        "p90_after": stats_after["p90"],
        "n_points_before": recon.num_points,
        "n_points_after": recon_opt.num_points,
        "n_obs_before": recon.num_observations,
        "n_obs_after": recon_opt.num_observations,
        "n_outliers_removed": n_outliers_removed,
        "camera_rotation_delta_mean": float(np.mean(rot_deltas)) if len(rot_deltas) > 0 else 0.0,
        "camera_rotation_delta_max": float(np.max(rot_deltas)) if len(rot_deltas) > 0 else 0.0,
        "camera_translation_delta_mean": float(np.mean(trans_deltas)) if len(trans_deltas) > 0 else 0.0,
        "camera_translation_delta_max": float(np.max(trans_deltas)) if len(trans_deltas) > 0 else 0.0,
        "time_seconds": elapsed,
        "convergence_success": bool(solver_round2.get("solver_success", solver_round1.get("solver_success", False))),
        "solver_round1": solver_round1,
        "solver_round2": solver_round2,
    }

    recon_opt.metadata["run_stage"] = "ba_pycolmap"
    recon_opt.metadata["run_timestamp_utc"] = utc_timestamp()
    recon_opt.metadata["run_config_path"] = config_path
    recon_opt.metadata["ba_pycolmap_stats"] = dict(stats)

    npz_path = os.path.join(output_dir, "reconstruction.npz")
    recon_opt.to_npz(npz_path)
    stats_path = os.path.join(output_dir, "ba_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=float)

    if not args.no_colmap_export:
        sparse_dir = os.path.join(output_dir, "sparse")
        reconstruction_to_colmap_sparse(recon_opt, sparse_dir, args.camera_type)
        print(f"Saved COLMAP model: {sparse_dir}")

    save_json(os.path.join(output_dir, "summary.json"), {
        "stage": "ba_pycolmap",
        "timestamp_utc": utc_timestamp(),
        "input_reconstruction": os.path.abspath(args.input),
        "output_reconstruction": npz_path,
        "stats": stats,
        "output_dir": output_dir,
    })

    print("\n" + "=" * 60)
    print("PyCOLMAP BA Results Summary")
    print("=" * 60)
    print(f"  RMSE:    {stats['rmse_before']:.4f} -> {stats['rmse_after']:.4f} px")
    print(f"  Median:  {stats['median_before']:.4f} -> {stats['median_after']:.4f} px")
    print(f"  P90:     {stats['p90_before']:.4f} -> {stats['p90_after']:.4f} px")
    print(f"  Points:  {stats['n_points_before']} -> {stats['n_points_after']}")
    print(f"  Obs:     {stats['n_obs_before']} -> {stats['n_obs_after']}")
    print(f"  Outliers removed: {stats['n_outliers_removed']}")
    print(f"  Cam rot delta (mean): {stats['camera_rotation_delta_mean']:.3f} deg")
    print(f"  Cam rot delta (max):  {stats['camera_rotation_delta_max']:.3f} deg")
    print(f"  Cam trans delta (mean): {stats['camera_translation_delta_mean']:.4f}")
    print(f"  Time:  {stats['time_seconds']:.1f}s")
    print(f"  Success: {stats['convergence_success']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
