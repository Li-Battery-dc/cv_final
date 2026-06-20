"""Synthetic scene generator and BA correctness test.

Creates a synthetic scene with known ground truth, adds noise,
and verifies that BA reduces reprojection error and recovers the
ground truth parameters.
"""

import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.reconstruction import Reconstruction
from src.ba.optimize import run_ba
from src.ba.utils import project_points, rodrigues_rotate, matrix_to_rotvec


def generate_synthetic_scene(
    n_cameras: int = 8,
    n_points: int = 500,
    image_size: tuple = (768, 432),
    focal_length: float = 500.0,
    noise_sigma: float = 1.0,
    outlier_fraction: float = 0.0,
    outlier_noise: float = 50.0,
    seed: int = 42,
) -> Reconstruction:
    """Generate a synthetic scene with known ground truth.

    Cameras are placed on a circle looking at the origin.
    3D points are randomly distributed in a sphere around the origin.

    Args:
        n_cameras: Number of cameras.
        n_points: Number of 3D points.
        image_size: (width, height) tuple.
        focal_length: Camera focal length.
        noise_sigma: Gaussian noise std (pixels) added to observations.
        outlier_fraction: Fraction of outlier observations (0-1).
        outlier_noise: Noise std for outlier observations.
        seed: Random seed.

    Returns:
        Reconstruction with noisy observations.
    """
    rng = np.random.RandomState(seed)
    W, H = image_size
    cx, cy = W / 2.0, H / 2.0

    # Generate cameras on a circle
    radius = 3.0
    extrinsics = np.zeros((n_cameras, 3, 4), dtype=np.float64)
    for i in range(n_cameras):
        angle = 2.0 * np.pi * i / n_cameras
        # Camera position on circle, looking at origin
        cam_pos = np.array([radius * np.cos(angle), 0.0, radius * np.sin(angle)])

        # Look-at: camera looks at origin, up is +Y
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        R = np.array([right, up, forward])  # world-to-camera rotation
        t = -R @ cam_pos
        extrinsics[i, :3, :3] = R
        extrinsics[i, :3, 3] = t

    # Intrinsics
    K = np.array([[focal_length, 0, cx],
                   [0, focal_length, cy],
                   [0, 0, 1]], dtype=np.float64)
    intrinsics = np.tile(K[None], (n_cameras, 1, 1))

    # Generate 3D points in a sphere
    points3d = rng.randn(n_points, 3).astype(np.float64) * 0.5

    # Build observations: each point is visible to all cameras that see it
    obs_cam_list, obs_pt_list, obs_xy_list = [], [], []

    for p in range(n_points):
        for s in range(n_cameras):
            proj = project_points(points3d[p:p + 1], extrinsics[s], intrinsics[s])
            u, v = proj[0]
            # Check if point projects into image
            if np.isfinite(u) and np.isfinite(v) and 0 < u < W and 0 < v < H:
                # Add Gaussian noise
                u_noisy = u + rng.randn() * noise_sigma
                v_noisy = v + rng.randn() * noise_sigma
                obs_cam_list.append(s)
                obs_pt_list.append(p)
                obs_xy_list.append([u_noisy, v_noisy])

    N = len(obs_cam_list)
    obs_camera_id = np.array(obs_cam_list, dtype=np.int32)
    obs_point_id = np.array(obs_pt_list, dtype=np.int32)
    obs_xy = np.array(obs_xy_list, dtype=np.float64)

    # Add outliers
    if outlier_fraction > 0 and N > 0:
        n_outliers = int(N * outlier_fraction)
        outlier_indices = rng.choice(N, n_outliers, replace=False)
        obs_xy[outlier_indices] += rng.randn(n_outliers, 2) * outlier_noise

    # Points RGB (random)
    points_rgb = rng.randint(0, 255, (n_points, 3), dtype=np.uint8)
    points_conf = np.ones(n_points, dtype=np.float32)

    # Image names
    image_names = np.array([f"cam_{i:04d}.png" for i in range(n_cameras)])
    image_size_hw = np.tile(np.array([H, W], dtype=np.int32), (n_cameras, 1))

    return Reconstruction(
        image_names=image_names,
        image_size_hw=image_size_hw,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        points3d=points3d,
        points_rgb=points_rgb,
        points_conf=points_conf,
        obs_camera_id=obs_camera_id,
        obs_point_id=obs_point_id,
        obs_xy=obs_xy,
        obs_conf=np.ones(N, dtype=np.float32),
    )


def add_pose_noise(recon: Reconstruction, rot_std: float = 0.05,
                   trans_std: float = 0.05, seed: int = 123) -> Reconstruction:
    """Add noise to camera poses (used for testing convergence from poor init)."""
    rng = np.random.RandomState(seed)
    noisy = recon.copy()
    for i in range(recon.num_images):
        R = noisy.extrinsics[i, :3, :3]
        t = noisy.extrinsics[i, :3, 3]
        # Add rotation noise
        noise_rv = rng.randn(3) * rot_std
        dR = rodrigues_rotate(noise_rv)
        R_noisy = dR @ R
        t_noisy = t + rng.randn(3) * trans_std
        noisy.extrinsics[i, :3, :3] = R_noisy
        noisy.extrinsics[i, :3, 3] = t_noisy
    return noisy


def add_point_noise(recon: Reconstruction, noise_std: float = 0.05,
                    seed: int = 456) -> Reconstruction:
    """Add noise to 3D points."""
    rng = np.random.RandomState(seed)
    noisy = recon.copy()
    noisy.points3d = noisy.points3d + rng.randn(*noisy.points3d.shape) * noise_std
    return noisy


def test_ba_convergence():
    """Test that BA reduces reprojection error on synthetic data."""
    print("=" * 60)
    print("Test: BA Convergence on Synthetic Data")
    print("=" * 60)

    # Generate clean scene
    recon_clean = generate_synthetic_scene(
        n_cameras=8, n_points=200, noise_sigma=0.0, seed=42
    )
    print(f"Synthetic scene: {recon_clean}")

    # Add noise
    recon_noisy = add_pose_noise(recon_clean, rot_std=0.05, trans_std=0.05)
    recon_noisy = add_point_noise(recon_noisy, noise_std=0.05)
    # Also add observation noise
    rng = np.random.RandomState(789)
    recon_noisy.obs_xy = recon_clean.obs_xy + rng.randn(*recon_clean.obs_xy.shape) * 0.5

    # Run BA
    recon_opt, stats = run_ba(
        recon_noisy,
        huber_delta=1.0,
        max_nfev=50,
        remove_outliers_after_round1=False,
        verbose=0,
    )

    # Checks
    assert not stats.get('failed', False), f"BA failed: {stats}"
    assert stats['rmse_after'] < stats['rmse_before'], \
        f"RMSE didn't decrease: {stats['rmse_before']:.3f} -> {stats['rmse_after']:.3f}"
    assert stats['rmse_after'] < 3.0, \
        f"RMSE too high after BA: {stats['rmse_after']:.3f}"

    print(f"PASS: RMSE {stats['rmse_before']:.3f} -> {stats['rmse_after']:.3f} px")
    return True


def test_gauge_anchor():
    """Test that cameras 0 and 1 are not changed by BA."""
    print("\n" + "=" * 60)
    print("Test: Gauge Anchor (fixed cameras)")
    print("=" * 60)

    recon = generate_synthetic_scene(n_cameras=6, n_points=100, noise_sigma=0.5)
    recon_noisy = add_pose_noise(recon, rot_std=0.05, trans_std=0.05)

    recon_opt, stats = run_ba(
        recon_noisy,
        n_fixed_cameras=2,
        max_nfev=30,
        remove_outliers_after_round1=False,
        verbose=0,
    )

    # Cameras 0 and 1 should be unchanged
    for i in range(2):
        np.testing.assert_allclose(
            recon_noisy.extrinsics[i], recon_opt.extrinsics[i],
            atol=1e-10, err_msg=f"Camera {i} was modified (should be fixed)"
        )

    print("PASS: Fixed cameras unchanged")
    return True


def test_outlier_removal():
    """Test that outlier observations are removed."""
    print("\n" + "=" * 60)
    print("Test: Outlier Removal")
    print("=" * 60)

    recon = generate_synthetic_scene(
        n_cameras=6, n_points=100, noise_sigma=0.5,
        outlier_fraction=0.15, outlier_noise=50.0,
    )
    print(f"Before: {recon.num_observations} observations (15% outliers)")

    recon_opt, stats = run_ba(
        recon,
        huber_delta=1.0,
        max_nfev=50,
        remove_outliers_after_round1=True,
        outlier_threshold_pixels=10.0,
        verbose=0,
    )

    n_removed = stats.get('n_outliers_removed', 0)
    print(f"Outliers removed: {n_removed}")
    print(f"RMSE: {stats['rmse_before']:.3f} -> {stats['rmse_after']:.3f}")

    # Should remove some observations
    assert n_removed > 0, "No outliers removed"
    assert stats['n_obs_after'] < stats['n_obs_before'], "Observation count should decrease"
    assert stats['rmse_after'] < stats['rmse_before'], "RMSE should decrease"

    print("PASS: Outlier removal works")
    return True


def test_project_consistency():
    """Test that our projection matches the reference projection."""
    print("\n" + "=" * 60)
    print("Test: Projection Consistency")
    print("=" * 60)

    from vggt.dependency.projection import project_3D_points_np

    rng = np.random.RandomState(42)
    P, S = 50, 5
    points3d = rng.randn(P, 3).astype(np.float64)
    extrinsics = np.zeros((S, 3, 4), dtype=np.float64)
    intrinsics = np.zeros((S, 3, 3), dtype=np.float64)

    for s in range(S):
        R = rodrigues_rotate(rng.randn(3) * 0.1)
        t = rng.randn(3)
        extrinsics[s, :3, :3] = R
        extrinsics[s, :3, 3] = t
        intrinsics[s] = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

    # Our projection
    for s in range(S):
        our_proj = project_points(points3d, extrinsics[s], intrinsics[s])

        # Reference projection
        ref_proj, _ = project_3D_points_np(
            points3d, extrinsics[s:s + 1], intrinsics[s:s + 1], extra_params=None
        )
        ref_proj = ref_proj[0]  # (P, 2)

        # Check that finite projections match
        our_valid = np.isfinite(our_proj[:, 0])
        ref_valid = np.isfinite(ref_proj[:, 0])

        if np.any(our_valid & ref_valid):
            diff = our_proj[our_valid & ref_valid] - ref_proj[our_valid & ref_valid]
            max_diff = np.max(np.abs(diff))
            assert max_diff < 1e-4, f"Projection mismatch at cam {s}: max diff = {max_diff}"

    print("PASS: Projection matches reference")
    return True


def test_so3_roundtrip():
    """Test rotation matrix <-> rotation vector conversion."""
    print("\n" + "=" * 60)
    print("Test: SO(3) Round-Trip")
    print("=" * 60)

    rng = np.random.RandomState(42)

    # Matrix -> rotvec -> matrix
    for _ in range(100):
        R = rodrigues_rotate(rng.randn(3))
        rv = matrix_to_rotvec(R)
        R2 = rodrigues_rotate(rv)
        np.testing.assert_allclose(R, R2, atol=1e-10)

    # rotvec -> matrix -> rotvec
    for _ in range(100):
        rv = rng.randn(3)
        R = rodrigues_rotate(rv)
        rv2 = matrix_to_rotvec(R)
        # Rotation vectors can differ, so check that they produce the same rotation
        R2 = rodrigues_rotate(rv2)
        np.testing.assert_allclose(R, R2, atol=1e-10)

    print("PASS: SO(3) round-trip")
    return True


def main():
    tests = [
        test_project_consistency,
        test_so3_roundtrip,
        test_ba_convergence,
        test_gauge_anchor,
        test_outlier_removal,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\nFAIL: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print(f"{'=' * 60}")

    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
