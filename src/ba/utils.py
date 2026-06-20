"""Projection, rotation, and reprojection error utilities for Bundle Adjustment."""

import numpy as np
from scipy.spatial.transform import Rotation


def rodrigues_rotate(rv: np.ndarray) -> np.ndarray:
    """Convert SO(3) rotation vector (3,) to rotation matrix (3, 3) via Rodrigues formula.

    Args:
        rv: (3,) rotation vector (angle * axis).

    Returns:
        R: (3, 3) rotation matrix.
    """
    return Rotation.from_rotvec(rv).as_matrix()


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to SO(3) rotation vector.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        rv: (3,) rotation vector.
    """
    return Rotation.from_matrix(R).as_rotvec()


def project_points(points3d: np.ndarray, extrinsic: np.ndarray,
                   intrinsic: np.ndarray) -> np.ndarray:
    """Project (P, 3) world points → (P, 2) pixel coordinates for one camera.

    Uses standard pinhole projection: K @ [R|t] @ [X; 1]

    Args:
        points3d: (P, 3) world coordinates.
        extrinsic: (3, 4) camera-from-world [R|t] (OpenCV convention).
        intrinsic: (3, 3) calibration matrix.

    Returns:
        points2d: (P, 2) pixel coordinates. Points behind camera get inf.
    """
    P = points3d.shape[0]
    # Homogeneous: (P, 4)
    pts_h = np.column_stack([points3d, np.ones(P, dtype=points3d.dtype)])
    # Camera coordinates: (P, 3) = (P, 4) @ (3, 4)^T
    cam_pts = pts_h @ extrinsic.T  # (P, 3)
    z = cam_pts[:, 2]
    # Only project points in front of camera (z > 0, OpenCV convention)
    valid = z > 1e-12
    uv = np.full((P, 2), np.inf, dtype=points3d.dtype)
    if np.any(valid):
        u = cam_pts[valid, 0] / cam_pts[valid, 2]
        v = cam_pts[valid, 1] / cam_pts[valid, 2]
        # K @ [u; v; 1]
        pts_img_h = np.column_stack([u, v, np.ones_like(u)])  # (V, 3)
        pts_img = pts_img_h @ intrinsic.T  # (V, 3)
        uv[valid, 0] = pts_img[:, 0]
        uv[valid, 1] = pts_img[:, 1]
    return uv


def project_points_batch(points3d: np.ndarray, extrinsics: np.ndarray,
                         intrinsics: np.ndarray) -> np.ndarray:
    """Project (P, 3) world points into all S cameras.

    Args:
        points3d: (P, 3) world coordinates.
        extrinsics: (S, 3, 4) camera-from-world matrices.
        intrinsics: (S, 3, 3) calibration matrices.

    Returns:
        points2d: (S, P, 2) pixel coordinates. Points behind camera get inf.
    """
    S, P = extrinsics.shape[0], points3d.shape[0]
    pts_h = np.column_stack([points3d, np.ones(P, dtype=points3d.dtype)])  # (P, 4)
    pts2d = np.full((S, P, 2), np.inf, dtype=points3d.dtype)

    for i in range(S):
        pts2d[i] = project_points(points3d, extrinsics[i], intrinsics[i])

    return pts2d


def compute_reprojection_errors_from_data(points3d: np.ndarray, extrinsics: np.ndarray,
                                          intrinsics: np.ndarray, obs_camera_id: np.ndarray,
                                          obs_point_id: np.ndarray,
                                          obs_xy: np.ndarray) -> dict:
    """Compute reprojection error statistics.

    Args:
        points3d: (P, 3) world points.
        extrinsics: (S, 3, 4) camera extrinsics.
        intrinsics: (S, 3, 3) camera intrinsics.
        obs_camera_id: (M,) camera index for each observation.
        obs_point_id: (M,) point index for each observation.
        obs_xy: (M, 2) observed pixel coordinates.

    Returns:
        dict with 'rmse', 'median', 'p90', 'errors_2d' (per-obs L2 error).
    """
    errors_2d = np.zeros(len(obs_camera_id), dtype=np.float64)
    S = extrinsics.shape[0]

    for cam_idx in range(S):
        mask = obs_camera_id == cam_idx
        if not np.any(mask):
            continue
        pid = obs_point_id[mask]
        pts3d = points3d[pid]
        proj = project_points(pts3d, extrinsics[cam_idx], intrinsics[cam_idx])
        diff = proj - obs_xy[mask]
        errors_2d[mask] = np.sqrt(np.sum(diff ** 2, axis=1))

    # Filter invalid (inf/nan)
    valid = np.isfinite(errors_2d)
    if np.any(valid):
        valid_errors = errors_2d[valid]
    else:
        valid_errors = errors_2d

    return {
        'rmse': float(np.sqrt(np.mean(valid_errors ** 2))),
        'median': float(np.median(valid_errors)),
        'p90': float(np.percentile(valid_errors, 90)),
        'errors_2d': errors_2d,
        'n_valid': int(np.sum(valid)),
        'n_total': len(errors_2d),
    }


def compute_camera_delta(R1: np.ndarray, t1: np.ndarray,
                         R2: np.ndarray, t2: np.ndarray) -> tuple:
    """Compute rotation angle (degrees) and translation distance between two cameras.

    Args:
        R1, R2: (3, 3) rotation matrices.
        t1, t2: (3,) translation vectors.

    Returns:
        (rot_deg, trans_dist): rotation angle in degrees and translation L2 distance.
    """
    # Relative rotation: R_rel = R2 @ R1^T
    R_rel = R2 @ R1.T
    # Rotation angle via trace: cos(theta) = (trace(R) - 1) / 2
    trace = np.trace(R_rel)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_deg = np.arccos(cos_theta) * 180.0 / np.pi
    trans_dist = float(np.linalg.norm(t2 - t1))
    return float(rot_deg), trans_dist
