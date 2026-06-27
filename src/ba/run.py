#!/usr/bin/env python3
"""Bundle Adjustment CLI entry point.

Usage:
    python -m src.ba.run \\
        --input outputs/vggt_raw/reconstruction.npz \\
        --output outputs/ba_custom

    python -m src.ba.run \\
        --input outputs/vggt_raw/reconstruction.npz \\
        --output outputs/ba_custom \\
        --huber_delta 1.0 \\
        --no-outlier-removal
"""

import os
import sys
import json
import argparse
import time

import numpy as np

# Ensure project root in path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.reconstruction import Reconstruction
from src.data.colmap_io import reconstruction_to_colmap_sparse
from src.ba.optimize import run_ba
from src.ba.problem import BAProblem
from src.utils.experiment import (
    prepare_output_dir,
    save_json,
    save_run_metadata,
    update_latest_symlink,
    utc_timestamp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Custom Bundle Adjustment")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input reconstruction.npz")
    parser.add_argument("--output", type=str, required=True,
                        help="Output root directory for BA results")
    parser.add_argument("--output_run_dir", type=str, default=None,
                        help="Optional explicit output run directory")
    parser.add_argument("--use_timestamp", action=argparse.BooleanOptionalAction, default=True,
                        help="If true, save to output/runs/<timestamp>_ba_custom")
    parser.add_argument("--huber_delta", type=float, default=1.0,
                        help="Huber loss delta (pixels)")
    parser.add_argument("--max_nfev", type=int, default=100,
                        help="Max function evaluations")
    parser.add_argument("--outlier_threshold", type=float, default=5.0,
                        help="Outlier removal threshold (pixels)")
    parser.add_argument("--n_fixed_cameras", type=int, default=2,
                        help="Number of fixed cameras (gauge anchor)")
    parser.add_argument("--no-outlier-removal", action="store_true", default=False,
                        help="Disable two-round outlier removal")
    parser.add_argument("--camera_type", type=str, default="PINHOLE",
                        help="COLMAP camera type for export")
    parser.add_argument("--no-colmap-export", action="store_true", default=False,
                        help="Skip COLMAP export")
    parser.add_argument("--verbose", type=int, default=2,
                        help="SciPy verbose level (0, 1, 2)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = prepare_output_dir(
        output_root=args.output,
        stage_name="ba_custom",
        explicit_output_dir=args.output_run_dir,
        use_timestamp=args.use_timestamp,
    )

    # Load input
    print(f"Loading reconstruction from: {args.input}")
    recon = Reconstruction.from_npz(args.input)
    print(f"Input: {recon}")
    config_path = save_run_metadata(
        output_dir,
        stage="ba_custom",
        params={
            "huber_delta": args.huber_delta,
            "max_nfev": args.max_nfev,
            "outlier_threshold": args.outlier_threshold,
            "n_fixed_cameras": args.n_fixed_cameras,
            "camera_type": args.camera_type,
            "no_outlier_removal": args.no_outlier_removal,
            "no_colmap_export": args.no_colmap_export,
            "verbose": args.verbose,
        },
        inputs={"input_reconstruction": os.path.abspath(args.input)},
    )
    print(f"Saved run config: {config_path}")

    # Check minimum requirements
    if recon.num_images < 3:
        print("Error: need at least 3 cameras for BA")
        return 1
    if recon.num_observations < 10:
        print("Error: need at least 10 observations for BA")
        return 1

    # Run BA
    t_start = time.time()
    recon_opt, stats = run_ba(
        reconstruction=recon,
        huber_delta=args.huber_delta,
        max_nfev=args.max_nfev,
        remove_outliers_after_round1=not args.no_outlier_removal,
        outlier_threshold_pixels=args.outlier_threshold,
        n_fixed_cameras=args.n_fixed_cameras,
        verbose=args.verbose,
    )

    if stats.get('failed', False):
        print(f"BA failed: {stats.get('error', 'Unknown error')}")
        # Still save the input as output
        recon_opt = recon
        stats['time_seconds'] = time.time() - t_start

    recon_opt.metadata['run_stage'] = 'ba_custom'
    recon_opt.metadata['run_timestamp_utc'] = utc_timestamp()
    recon_opt.metadata['run_config_path'] = config_path
    recon_opt.metadata['ba_stats'] = dict(stats)

    # Save optimized reconstruction
    npz_path = os.path.join(output_dir, "reconstruction.npz")
    recon_opt.to_npz(npz_path)
    print(f"Saved optimized reconstruction: {npz_path}")

    # Save statistics
    stats_path = os.path.join(output_dir, "ba_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"Saved BA statistics: {stats_path}")

    # Export COLMAP
    if not args.no_colmap_export:
        sparse_dir = os.path.join(output_dir, "sparse")
        reconstruction_to_colmap_sparse(recon_opt, sparse_dir, args.camera_type)
        print(f"Saved COLMAP model: {sparse_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("BA Results Summary")
    print("=" * 60)
    print(f"  RMSE:    {stats.get('rmse_before', 0):.4f} -> {stats.get('rmse_after', 0):.4f} px")
    print(f"  Median:  {stats.get('median_before', 0):.4f} -> {stats.get('median_after', 0):.4f} px")
    print(f"  P90:     {stats.get('p90_before', 0):.4f} -> {stats.get('p90_after', 0):.4f} px")
    print(f"  Points:  {stats.get('n_points_before', 0)} -> {stats.get('n_points_after', 0)}")
    print(f"  Obs:     {stats.get('n_obs_before', 0)} -> {stats.get('n_obs_after', 0)}")
    print(f"  Outliers removed: {stats.get('n_outliers_removed', 0)}")
    print(f"  Cam rot delta (mean): {stats.get('camera_rotation_delta_mean', 0):.3f} deg")
    print(f"  Cam rot delta (max):  {stats.get('camera_rotation_delta_max', 0):.3f} deg")
    print(f"  Cam trans delta (mean): {stats.get('camera_translation_delta_mean', 0):.4f}")
    print(f"  Time:  {stats.get('time_seconds', 0):.1f}s")
    print(f"  Success: {stats.get('convergence_success', False)}")
    print("=" * 60)
    save_json(os.path.join(output_dir, "summary.json"), {
        "stage": "ba_custom",
        "timestamp_utc": utc_timestamp(),
        "input_reconstruction": os.path.abspath(args.input),
        "output_reconstruction": npz_path,
        "camera_type": args.camera_type,
        "stats": stats,
        "output_dir": output_dir,
    })
    update_latest_symlink(args.output, output_dir)

    return 0 if not stats.get('failed', False) else 1


if __name__ == "__main__":
    sys.exit(main())
