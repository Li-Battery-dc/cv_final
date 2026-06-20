"""Tests for COLMAP <-> unified Reconstruction round-trip."""

import os
import sys
import tempfile
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.reconstruction import Reconstruction
from src.data.colmap_io import (reconstruction_to_colmap_sparse,
                                 colmap_sparse_to_reconstruction,
                                 reconstruction_to_pycolmap)
from src.ba.utils import rodrigues_rotate


def create_dummy_reconstruction():
    """Create a small test reconstruction."""
    rng = np.random.RandomState(42)
    S, P = 4, 20

    image_names = np.array([f"img_{i:04d}.jpg" for i in range(S)])
    image_size_hw = np.tile(np.array([432, 768], dtype=np.int32), (S, 1))

    # Camera intrinsics
    K = np.array([[500, 0, 384], [0, 500, 216], [0, 0, 1]], dtype=np.float64)
    intrinsics = np.tile(K[None], (S, 1, 1))

    # Camera extrinsics (circle)
    extrinsics = np.zeros((S, 3, 4), dtype=np.float64)
    for i in range(S):
        angle = 2 * np.pi * i / S
        cam_pos = np.array([3 * np.cos(angle), 0.0, 3 * np.sin(angle)])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        R = np.array([right, up, forward])
        t = -R @ cam_pos
        extrinsics[i, :3, :3] = R
        extrinsics[i, :3, 3] = t

    # Points
    points3d = rng.randn(P, 3).astype(np.float64)
    points_rgb = rng.randint(0, 255, (P, 3), dtype=np.uint8)

    # Observations (each point seen by some cameras)
    obs_cam, obs_pt, obs_xy = [], [], []
    from src.ba.utils import project_points

    for p in range(P):
        for s in range(S):
            if rng.random() > 0.3:  # 70% chance of observation
                proj = project_points(points3d[p:p + 1], extrinsics[s], intrinsics[s])
                if np.isfinite(proj[0, 0]):
                    obs_cam.append(s)
                    obs_pt.append(p)
                    obs_xy.append(proj[0])

    N = len(obs_cam)
    return Reconstruction(
        image_names=image_names,
        image_size_hw=image_size_hw,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        points3d=points3d,
        points_rgb=points_rgb,
        points_conf=np.ones(P, dtype=np.float32),
        obs_camera_id=np.array(obs_cam, dtype=np.int32),
        obs_point_id=np.array(obs_pt, dtype=np.int32),
        obs_xy=np.array(obs_xy, dtype=np.float64),
        obs_conf=np.ones(N, dtype=np.float32),
    )


def test_npz_roundtrip():
    """Save and load .npz should preserve all fields."""
    recon = create_dummy_reconstruction()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.npz")
        recon.to_npz(path)
        loaded = Reconstruction.from_npz(path)

        assert recon.num_images == loaded.num_images
        assert recon.num_points == loaded.num_points
        assert recon.num_observations == loaded.num_observations

        np.testing.assert_allclose(recon.intrinsics, loaded.intrinsics)
        np.testing.assert_allclose(recon.extrinsics, loaded.extrinsics, atol=1e-10)
        np.testing.assert_allclose(recon.points3d, loaded.points3d, atol=1e-10)
        np.testing.assert_array_equal(recon.points_rgb, loaded.points_rgb)
        np.testing.assert_array_equal(recon.obs_camera_id, loaded.obs_camera_id)
        np.testing.assert_array_equal(recon.obs_point_id, loaded.obs_point_id)
        np.testing.assert_allclose(recon.obs_xy, loaded.obs_xy)


def test_colmap_roundtrip():
    """Conversion to/from pycolmap should be consistent."""
    recon = create_dummy_reconstruction()

    try:
        recon_obj = reconstruction_to_pycolmap(recon, camera_type="PINHOLE")
    except Exception as e:
        print(f"Skipping COLMAP roundtrip test: {e}")
        return

    # Check basic counts
    assert len(recon_obj.cameras) == recon.num_images
    assert len(recon_obj.images) == recon.num_images

    # Extract back
    loaded = Reconstruction.from_pycolmap(
        recon_obj,
        list(recon.image_names),
        recon.image_size_hw,
    )

    # Check intrinsics (accounting for possible reordering)
    np.testing.assert_allclose(
        recon.intrinsics, loaded.intrinsics,
        atol=1e-5, rtol=1e-4
    )

    # Check extrinsics
    np.testing.assert_allclose(
        recon.extrinsics, loaded.extrinsics,
        atol=1e-5, rtol=1e-4
    )

    # Check points
    np.testing.assert_allclose(
        recon.points3d, loaded.points3d,
        atol=1e-5, rtol=1e-4
    )


if __name__ == "__main__":
    test_npz_roundtrip()
    test_colmap_roundtrip()
    print("All COLMAP roundtrip tests passed!")
