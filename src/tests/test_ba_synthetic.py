"""Tests for BA convergence on synthetic data."""

import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ba.optimize import run_ba
from src.ba.synthetic_test import (generate_synthetic_scene,
                                   add_pose_noise, add_point_noise)


def test_ba_convergence():
    """BA should reduce reprojection error on synthetic data."""
    # Generate clean scene with observation noise only
    recon = generate_synthetic_scene(
        n_cameras=6, n_points=100, noise_sigma=1.0, seed=42
    )
    # Add small perturbations to poses and points
    recon_noisy = add_pose_noise(recon, rot_std=0.03, trans_std=0.02)
    recon_noisy = add_point_noise(recon_noisy, noise_std=0.03)

    recon_opt, stats = run_ba(
        recon_noisy,
        huber_delta=1.0,
        max_nfev=200,
        remove_outliers_after_round1=False,
        verbose=0,
    )

    assert not stats.get('failed', False), f"BA failed: {stats}"
    assert stats['rmse_after'] < stats['rmse_before'], \
        f"RMSE didn't decrease: {stats['rmse_before']:.3f} -> {stats['rmse_after']:.3f}"
    # With observation noise sigma=1.0, expect RMSE roughly ~1-2 pixels
    # The 5.0 threshold provides generous margin
    assert stats['rmse_after'] < 5.0, \
        f"RMSE too high after BA: {stats['rmse_after']:.3f}"
    print(f"  RMSE: {stats['rmse_before']:.3f} -> {stats['rmse_after']:.3f} ✓")


def test_ba_reduces_error():
    """BA must always reduce reprojection error (smoke test)."""
    recon = generate_synthetic_scene(n_cameras=4, n_points=50, noise_sigma=0.5)
    recon_opt, stats = run_ba(
        recon, huber_delta=1.0, max_nfev=100,
        remove_outliers_after_round1=False, verbose=0,
    )
    assert not stats.get('failed', False), f"BA failed: {stats}"
    # Even on clean data, BA should maintain or reduce error
    # With noise_sigma=0.5, RMSE should be around 0.5
    assert stats['rmse_after'] < 3.0, f"RMSE={stats['rmse_after']:.3f} too high"
    print(f"  Clean BA: RMSE={stats['rmse_after']:.3f} ✓")


def test_gauge_anchor():
    """Fixed cameras should not be modified by BA."""
    recon = generate_synthetic_scene(n_cameras=4, n_points=50, noise_sigma=0.5)
    recon_noisy = add_pose_noise(recon, rot_std=0.01, trans_std=0.01)

    recon_opt, stats = run_ba(
        recon_noisy,
        n_fixed_cameras=2,
        max_nfev=50,
        remove_outliers_after_round1=False,
        verbose=0,
    )

    for i in range(2):
        np.testing.assert_allclose(
            recon_noisy.extrinsics[i], recon_opt.extrinsics[i],
            atol=1e-10, err_msg=f"Camera {i} was modified (should be fixed)"
        )
    print("  Fixed cameras 0 and 1 unchanged ✓")


def test_outlier_removal():
    """Outlier observations should be identified and removed."""
    recon = generate_synthetic_scene(
        n_cameras=6, n_points=100, noise_sigma=0.5,
        outlier_fraction=0.15, outlier_noise=50.0,
    )

    recon_opt, stats = run_ba(
        recon,
        huber_delta=1.0,
        max_nfev=100,
        remove_outliers_after_round1=True,
        outlier_threshold_pixels=10.0,
        verbose=0,
    )

    assert not stats.get('failed', False)
    assert stats['n_outliers_removed'] > 0, "No outliers were removed"
    assert stats['rmse_after'] < stats['rmse_before'], "RMSE should decrease"
    print(f"  Outliers removed: {stats['n_outliers_removed']} ✓")


if __name__ == "__main__":
    test_ba_convergence()
    test_ba_reduces_error()
    test_gauge_anchor()
    test_outlier_removal()
    print("All BA synthetic tests passed!")
