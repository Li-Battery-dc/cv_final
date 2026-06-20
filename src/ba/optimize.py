"""Two-round Bundle Adjustment solver using SciPy least_squares."""

import time
import numpy as np
import scipy.optimize

from src.data.reconstruction import Reconstruction
from src.ba.problem import BAProblem


def run_ba(
    reconstruction: Reconstruction,
    huber_delta: float = 1.0,
    max_nfev: int = 100,
    ftol: float = 1e-8,
    gtol: float = 1e-12,
    remove_outliers_after_round1: bool = True,
    outlier_threshold_pixels: float = 5.0,
    n_fixed_cameras: int = 2,
    verbose: int = 2,
) -> tuple:
    """Two-round bundle adjustment.

    Round 1: Solve with Huber loss on all observations.
    Round 2: Remove observations with |residual| > outlier_threshold_pixels,
             re-solve with Huber loss.

    Args:
        reconstruction: Initial Reconstruction with observations.
        huber_delta: Huber loss threshold (pixels).
        max_nfev: Max function evaluations per solve.
        ftol: Function tolerance.
        gtol: Gradient tolerance.
        remove_outliers_after_round1: Enable two-round outlier removal.
        outlier_threshold_pixels: Max residual for round 2 inlier.
        n_fixed_cameras: Number of initial cameras to fix as gauge anchor.
        verbose: SciPy verbose level (0, 1, 2).

    Returns:
        (optimized_reconstruction, stats_dict).
        On failure: (reconstruction, {'failed': True, 'error': str}).
    """
    t_start = time.time()
    stats = {}

    # Validate input
    if reconstruction.num_observations == 0:
        return reconstruction, {'failed': True, 'error': 'No observations'}

    if reconstruction.num_images < n_fixed_cameras + 1:
        return reconstruction, {'failed': True,
                                'error': f'Need at least {n_fixed_cameras + 1} cameras'}

    print(f"BA: {reconstruction.num_images} cameras, {reconstruction.num_points} points, "
          f"{reconstruction.num_observations} obs, {n_fixed_cameras} fixed cameras")

    # ---- Round 1 ----
    print("\n--- BA Round 1 ---")
    problem = BAProblem(reconstruction, n_fixed_cameras, huber_delta)

    stats_before = problem.compute_stats()
    print(f"Before: RMSE={stats_before['rmse']:.3f}, median={stats_before['median']:.3f}, "
          f"P90={stats_before['p90']:.3f}")

    x0 = problem.pack_params()
    J_sparsity = problem.compute_sparsity()

    print(f"Variables: {problem.n_vars}, Observations: {2 * problem.N}")
    print(f"Sparsity: {J_sparsity.nnz} nonzeros (density: {J_sparsity.nnz / (2 * problem.N * problem.n_vars):.6f})")

    result1 = scipy.optimize.least_squares(
        problem.compute_residuals,
        x0,
        jac_sparsity=J_sparsity,
        loss='huber',
        f_scale=huber_delta,
        method='trf',
        max_nfev=max_nfev,
        ftol=ftol,
        gtol=gtol,
        verbose=verbose,
        x_scale='jac',
    )

    nfev_r1 = result1.nfev
    print(f"Round 1: success={result1.success}, nfev={result1.nfev}, "
          f"cost={result1.cost:.6f}, optimality={result1.optimality:.6f}")

    extrinsics_r1, points3d_r1 = problem.unpack_params(result1.x)
    stats_r1 = problem.compute_stats(result1.x)
    print(f"Round 1: RMSE={stats_r1['rmse']:.3f}, median={stats_r1['median']:.3f}, "
          f"P90={stats_r1['p90']:.3f}")

    stats['rmse_before'] = stats_before['rmse']
    stats['rmse_round1'] = stats_r1['rmse']
    stats['median_before'] = stats_before['median']
    stats['p90_before'] = stats_before['p90']
    stats['nfev_round1'] = nfev_r1
    stats['success_round1'] = bool(result1.success)

    # ---- Outlier Removal ----
    if remove_outliers_after_round1:
        print(f"\n--- Outlier Removal (threshold={outlier_threshold_pixels}px) ---")
        errors_r1 = stats_r1.get('errors_2d', None)
        if errors_r1 is None:
            extrinsics_r1, points3d_r1 = problem.unpack_params(result1.x)
            errors_r1 = problem.compute_residuals(result1.x)
            # Reshape from (2*N,) to per-obs L2 error
            errors_r1 = np.sqrt(
                errors_r1[0::2] ** 2 + errors_r1[1::2] ** 2
            )

        outlier_mask = np.isfinite(errors_r1) & (errors_r1 > outlier_threshold_pixels)
        n_outliers = int(np.sum(outlier_mask))
        stats['n_outliers_removed'] = n_outliers
        print(f"Outliers removed: {n_outliers} / {len(errors_r1)} "
              f"({100 * n_outliers / len(errors_r1):.1f}%)")

        # Build filtered reconstruction
        keep = np.isfinite(errors_r1) & (errors_r1 <= outlier_threshold_pixels)
        filtered_recon = _build_filtered_reconstruction(reconstruction, extrinsics_r1, points3d_r1, keep)
        stats['n_obs_before_round2'] = filtered_recon.num_observations

        if filtered_recon.num_observations < 2 * filtered_recon.num_points:
            print("Warning: Very few observations after outlier removal. Results may be poor.")

    else:
        filtered_recon = _build_reconstruction_from_result(reconstruction, extrinsics_r1, points3d_r1)
        stats['n_outliers_removed'] = 0

    # ---- Round 2 ----
    print(f"\n--- BA Round 2 ---")
    print(f"Filtered: {filtered_recon.num_images} cameras, {filtered_recon.num_points} points, "
          f"{filtered_recon.num_observations} obs")

    if filtered_recon.num_observations < 10:
        print("Too few observations for round 2, using round 1 result")
        optimized_recon = _build_reconstruction_from_result(reconstruction, extrinsics_r1, points3d_r1)
        stats['rmse_after'] = stats_r1['rmse']
        stats['median_after'] = stats_r1['median']
        stats['p90_after'] = stats_r1['p90']
    else:
        problem2 = BAProblem(filtered_recon, n_fixed_cameras, huber_delta)
        x0_2 = problem2.pack_params()
        J_sparsity_2 = problem2.compute_sparsity()

        result2 = scipy.optimize.least_squares(
            problem2.compute_residuals,
            x0_2,
            jac_sparsity=J_sparsity_2,
            loss='huber',
            f_scale=huber_delta,
            method='trf',
            max_nfev=max_nfev,
            ftol=ftol,
            gtol=gtol,
            verbose=verbose,
            x_scale='jac',
        )

        stats['nfev_round2'] = result2.nfev
        stats['success_round2'] = bool(result2.success)
        print(f"Round 2: success={result2.success}, nfev={result2.nfev}, cost={result2.cost:.6f}")

        extrinsics_opt, points3d_opt = problem2.unpack_params(result2.x)
        optimized_recon = _build_reconstruction_from_result(filtered_recon, extrinsics_opt, points3d_opt)
        stats_after = problem2.compute_stats(result2.x)
        stats['rmse_after'] = stats_after['rmse']
        stats['median_after'] = stats_after['median']
        stats['p90_after'] = stats_after['p90']

    # ---- Final Statistics ----
    t_elapsed = time.time() - t_start
    stats['time_seconds'] = t_elapsed
    stats['n_points_before'] = reconstruction.num_points
    stats['n_points_after'] = optimized_recon.num_points
    stats['n_obs_before'] = reconstruction.num_observations
    stats['n_obs_after'] = optimized_recon.num_observations
    stats['failed'] = False

    # Camera deltas
    rot_deltas, trans_deltas = problem.compute_camera_deltas(optimized_recon.extrinsics)
    stats['camera_rotation_delta_mean'] = float(np.mean(rot_deltas)) if len(rot_deltas) > 0 else 0.0
    stats['camera_rotation_delta_max'] = float(np.max(rot_deltas)) if len(rot_deltas) > 0 else 0.0
    stats['camera_translation_delta_mean'] = float(np.mean(trans_deltas)) if len(trans_deltas) > 0 else 0.0
    stats['camera_translation_delta_max'] = float(np.max(trans_deltas)) if len(trans_deltas) > 0 else 0.0

    # Convergence curve (approximate from solution)
    stats['convergence_nfev'] = stats.get('nfev_round2', stats.get('nfev_round1', 0))
    stats['convergence_success'] = stats.get('success_round2', stats.get('success_round1', False))

    print(f"\n--- BA Complete ({t_elapsed:.1f}s) ---")
    print(f"RMSE: {stats.get('rmse_before', 0):.3f} -> {stats.get('rmse_after', 0):.3f} px")
    print(f"Points: {stats.get('n_points_before', 0)} -> {stats.get('n_points_after', 0)}")
    print(f"Obs: {stats.get('n_obs_before', 0)} -> {stats.get('n_obs_after', 0)}")
    print(f"Mean camera rotation delta: {stats.get('camera_rotation_delta_mean', 0):.3f} deg")
    print(f"Mean camera translation delta: {stats.get('camera_translation_delta_mean', 0):.4f}")

    return optimized_recon, stats


def _build_filtered_reconstruction(recon: Reconstruction,
                                   extrinsics: np.ndarray,
                                   points3d: np.ndarray,
                                   keep_mask: np.ndarray) -> Reconstruction:
    """Build a new Reconstruction keeping only selected observations."""
    new_recon = Reconstruction(
        image_names=recon.image_names.copy(),
        image_size_hw=recon.image_size_hw.copy(),
        intrinsics=recon.intrinsics.copy(),
        extrinsics=extrinsics,
        points3d=points3d.copy(),
        points_rgb=recon.points_rgb.copy(),
        points_conf=recon.points_conf.copy(),
        obs_camera_id=recon.obs_camera_id[keep_mask],
        obs_point_id=recon.obs_point_id[keep_mask],
        obs_xy=recon.obs_xy[keep_mask],
        obs_conf=recon.obs_conf[keep_mask] if len(recon.obs_conf) > 0 else recon.obs_conf,
        metadata=dict(recon.metadata),
    )

    # Remove points that now have fewer than 2 observations
    if new_recon.num_observations > 0:
        counts = np.bincount(new_recon.obs_point_id, minlength=new_recon.num_points)
        valid_pts = counts >= 2
        old_to_new = np.full(new_recon.num_points, -1, dtype=np.int32)
        old_to_new[valid_pts] = np.arange(np.sum(valid_pts))

        new_recon.points3d = new_recon.points3d[valid_pts]
        new_recon.points_rgb = new_recon.points_rgb[valid_pts]
        new_recon.points_conf = new_recon.points_conf[valid_pts]

        obs_valid = old_to_new[new_recon.obs_point_id] >= 0
        new_recon.obs_camera_id = new_recon.obs_camera_id[obs_valid]
        new_recon.obs_point_id = old_to_new[new_recon.obs_point_id[obs_valid]]
        new_recon.obs_xy = new_recon.obs_xy[obs_valid]
        if len(new_recon.obs_conf) > 0:
            new_recon.obs_conf = new_recon.obs_conf[obs_valid]

    return new_recon


def _build_reconstruction_from_result(recon: Reconstruction,
                                      extrinsics: np.ndarray,
                                      points3d: np.ndarray) -> Reconstruction:
    """Build Reconstruction with updated parameters (same observations)."""
    return Reconstruction(
        image_names=recon.image_names.copy(),
        image_size_hw=recon.image_size_hw.copy(),
        intrinsics=recon.intrinsics.copy(),
        extrinsics=extrinsics,
        points3d=points3d,
        points_rgb=recon.points_rgb,
        points_conf=recon.points_conf,
        obs_camera_id=recon.obs_camera_id,
        obs_point_id=recon.obs_point_id,
        obs_xy=recon.obs_xy,
        obs_conf=recon.obs_conf,
        metadata=dict(recon.metadata),
    )
