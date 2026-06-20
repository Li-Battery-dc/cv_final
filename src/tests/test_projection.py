"""Tests for projection consistency against VGGT reference."""

import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_VGGT_PATH = os.path.join(_PROJECT_ROOT, 'vggt')
if _VGGT_PATH not in sys.path:
    sys.path.insert(0, _VGGT_PATH)

from src.ba.utils import project_points, rodrigues_rotate


def test_projection_vs_vggt_reference():
    """Our projection should match VGGT's project_3D_points_np."""
    from vggt.dependency.projection import project_3D_points_np

    rng = np.random.RandomState(42)
    P, S = 50, 5
    points3d = rng.randn(P, 3).astype(np.float64)
    extrinsics = np.zeros((S, 3, 4), dtype=np.float64)
    intrinsics = np.zeros((S, 3, 3), dtype=np.float64)

    for s in range(S):
        R = rodrigues_rotate(rng.randn(3) * 0.5)
        t = rng.randn(3)
        extrinsics[s, :3, :3] = R
        extrinsics[s, :3, 3] = t
        intrinsics[s] = np.array([[500, 0, 320],
                                   [0, 500, 240],
                                   [0, 0, 1]], dtype=np.float64)

    for s in range(S):
        our = project_points(points3d, extrinsics[s], intrinsics[s])
        ref, _ = project_3D_points_np(
            points3d, extrinsics[s:s + 1], intrinsics[s:s + 1], extra_params=None
        )
        ref = ref[0]  # (P, 2)

        # Compare finite projections
        our_valid = np.isfinite(our[:, 0])
        ref_valid = np.isfinite(ref[:, 0])
        common = our_valid & ref_valid

        if np.any(common):
            diff = np.abs(our[common] - ref[common])
            max_diff = diff.max()
            assert max_diff < 1e-4, f"Max projection diff: {max_diff} at camera {s}"


def test_behind_camera_points():
    """Points behind the camera should get inf projection."""
    # Camera at origin looking +z (OpenCV convention)
    R = np.eye(3)
    t = np.zeros(3)
    extrinsic = np.column_stack([R, t])
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

    # Points behind camera (negative z in camera space)
    pts_behind = np.array([[0, 0, -1.0], [1, 1, -0.5], [-1, -1, -2.0]], dtype=np.float64)
    proj_behind = project_points(pts_behind, extrinsic, K)
    assert np.all(~np.isfinite(proj_behind[:, 0])), "Behind-camera points should get inf"

    # Points in front
    pts_front = np.array([[0, 0, 1.0], [0.1, 0.1, 2.0]], dtype=np.float64)
    proj_front = project_points(pts_front, extrinsic, K)
    assert np.all(np.isfinite(proj_front[:, 0])), "Front points should get finite projections"


def test_zero_depth_points():
    """Points at depth=0 should be handled gracefully."""
    R = np.eye(3)
    t = np.zeros(3)
    extrinsic = np.column_stack([R, t])
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

    pts_zero = np.array([[0, 0, 0.0]], dtype=np.float64)
    proj = project_points(pts_zero, extrinsic, K)
    assert not np.isfinite(proj[0, 0]), "Zero-depth point should be non-finite"


if __name__ == "__main__":
    test_projection_vs_vggt_reference()
    test_behind_camera_points()
    test_zero_depth_points()
    print("All projection tests passed!")
