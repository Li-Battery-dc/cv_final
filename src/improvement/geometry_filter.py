"""Depth/camera point construction and geometry consistency filtering."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from src.ba.utils import project_points
from src.data.reconstruction import Reconstruction


@dataclass
class GeometryFilterResult:
    reconstruction: Reconstruction
    stats: dict
    keep_mask: np.ndarray


def _as_depth_hw(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape (S,H,W) or (S,H,W,1), got {depth_map.shape}")
    return depth.astype(np.float64, copy=False)


def _as_conf_hw(conf: np.ndarray | None, shape: tuple[int, int, int]) -> np.ndarray:
    if conf is None:
        return np.ones(shape, dtype=np.float32)
    arr = np.asarray(conf)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.shape != shape:
        raise ValueError(f"Expected confidence shape {shape}, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _normalize_world_points(points: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(points)
    if arr.shape[:3] != shape or arr.shape[-1] != 3:
        raise ValueError(f"Expected point map shape {shape + (3,)}, got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def _camera_depth(points3d: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    pts_h = np.column_stack([points3d, np.ones(len(points3d), dtype=points3d.dtype)])
    return (pts_h @ extrinsic.T)[:, 2]


def _bilinear_sample_depth(depth: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    x = xy[:, 0]
    y = xy[:, 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)

    out = np.full(len(x), np.nan, dtype=np.float64)
    if not np.any(valid):
        return out, valid

    xv = x[valid]
    yv = y[valid]
    x0 = np.floor(xv).astype(np.int64)
    y0 = np.floor(yv).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wa = (x1 - xv) * (y1 - yv)
    wb = (x1 - xv) * (yv - y0)
    wc = (xv - x0) * (y1 - yv)
    wd = (xv - x0) * (yv - y0)

    same_x = x0 == x1
    same_y = y0 == y1
    if np.any(same_x):
        wa[same_x] = (y1[same_x] - yv[same_x])
        wb[same_x] = (yv[same_x] - y0[same_x])
        wc[same_x] = 0.0
        wd[same_x] = 0.0
    if np.any(same_y):
        wa[same_y] = (x1[same_y] - xv[same_y])
        wc[same_y] = (xv[same_y] - x0[same_y])
        wb[same_y] = 0.0
        wd[same_y] = 0.0
    if np.any(same_x & same_y):
        wa[same_x & same_y] = 1.0

    sampled = (
        wa * depth[y0, x0]
        + wb * depth[y1, x0]
        + wc * depth[y0, x1]
        + wd * depth[y1, x1]
    )
    out[valid] = sampled
    return out, valid


def _sample_rgb(images_np: np.ndarray, height: int, width: int) -> np.ndarray:
    if images_np.ndim != 4 or images_np.shape[-1] != 3:
        raise ValueError(f"Expected images shape (S,H,W,3), got {images_np.shape}")
    if images_np.shape[1:3] == (height, width):
        rgb = images_np
    else:
        from PIL import Image

        resized = []
        for image in images_np:
            pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
            pil = pil.resize((width, height), Image.BILINEAR)
            resized.append(np.asarray(pil, dtype=np.uint8))
        rgb = np.stack(resized, axis=0)
    return rgb.reshape(-1, 3).astype(np.uint8)


def compute_reprojection_votes(
    points_depth: np.ndarray,
    source_frame_ids: np.ndarray,
    depth_map: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    neighbor_offsets: tuple[int, ...] = (-2, -1, 1, 2),
    reproj_percentile: float = 70.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Count neighboring depth-consistency votes for flattened dense points."""
    n = len(points_depth)
    vote_counts = np.zeros(n, dtype=np.int16)
    valid_counts = np.zeros(n, dtype=np.int16)
    best_errors = np.full(n, np.inf, dtype=np.float64)
    all_errors = []
    s_count = depth_map.shape[0]

    for src in range(s_count):
        src_mask = source_frame_ids == src
        if not np.any(src_mask):
            continue
        src_indices = np.flatnonzero(src_mask)
        src_points = points_depth[src_indices]

        for offset in neighbor_offsets:
            dst = src + offset
            if dst < 0 or dst >= s_count:
                continue
            xy = project_points(src_points, extrinsics[dst], intrinsics[dst])
            z_proj = _camera_depth(src_points, extrinsics[dst])
            sampled, inside = _bilinear_sample_depth(depth_map[dst], xy)
            valid = inside & np.isfinite(sampled) & (sampled > eps) & np.isfinite(z_proj) & (z_proj > eps)
            if not np.any(valid):
                continue
            rel_err = np.full(len(src_points), np.inf, dtype=np.float64)
            rel_err[valid] = np.abs(z_proj[valid] - sampled[valid]) / (np.abs(sampled[valid]) + eps)
            all_errors.append(rel_err[valid])
            target_indices = src_indices[valid]
            valid_counts[target_indices] += 1
            best_errors[target_indices] = np.minimum(best_errors[target_indices], rel_err[valid])

    if all_errors:
        threshold = float(np.percentile(np.concatenate(all_errors), reproj_percentile))
        vote_counts = ((best_errors <= threshold) & np.isfinite(best_errors)).astype(np.int16)

        # Count all passing neighbor checks, not only whether each point passed once.
        vote_counts[:] = 0
        for src in range(s_count):
            src_mask = source_frame_ids == src
            if not np.any(src_mask):
                continue
            src_indices = np.flatnonzero(src_mask)
            src_points = points_depth[src_indices]
            for offset in neighbor_offsets:
                dst = src + offset
                if dst < 0 or dst >= s_count:
                    continue
                xy = project_points(src_points, extrinsics[dst], intrinsics[dst])
                z_proj = _camera_depth(src_points, extrinsics[dst])
                sampled, inside = _bilinear_sample_depth(depth_map[dst], xy)
                valid = inside & np.isfinite(sampled) & (sampled > eps) & np.isfinite(z_proj) & (z_proj > eps)
                if not np.any(valid):
                    continue
                rel_err = np.abs(z_proj[valid] - sampled[valid]) / (np.abs(sampled[valid]) + eps)
                passed = rel_err <= threshold
                if np.any(passed):
                    vote_counts[src_indices[valid][passed]] += 1
    else:
        threshold = float("inf")

    return vote_counts, valid_counts, threshold


def build_filtered_dense_reconstruction(
    *,
    image_names: list[str] | np.ndarray,
    image_size_hw: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_map: np.ndarray,
    depth_conf: np.ndarray | None,
    points_depth: np.ndarray,
    points_point: np.ndarray,
    images_np: np.ndarray,
    disagreement_percentile: float = 70.0,
    reproj_percentile: float = 70.0,
    min_reproj_votes: int = 1,
    max_points: int = 200_000,
    rng: np.random.Generator | None = None,
    metadata: dict | None = None,
) -> GeometryFilterResult:
    """Build a dense Reconstruction from depth points after full adaptive filtering."""
    return build_dense_reconstruction_variant(
        image_names=image_names,
        image_size_hw=image_size_hw,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth_map=depth_map,
        depth_conf=depth_conf,
        points_depth=points_depth,
        points_point=points_point,
        images_np=images_np,
        variant="filtered_full",
        disagreement_percentile=disagreement_percentile,
        reproj_percentile=reproj_percentile,
        min_reproj_votes=min_reproj_votes,
        max_points=max_points,
        rng=rng,
        metadata=metadata,
    )


def build_dense_reconstruction_variant(
    *,
    image_names: list[str] | np.ndarray,
    image_size_hw: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_map: np.ndarray,
    depth_conf: np.ndarray | None,
    points_depth: np.ndarray,
    points_point: np.ndarray | None,
    images_np: np.ndarray,
    variant: str,
    disagreement_percentile: float = 70.0,
    reproj_percentile: float = 70.0,
    min_reproj_votes: int = 1,
    max_points: int = 200_000,
    rng: np.random.Generator | None = None,
    metadata: dict | None = None,
) -> GeometryFilterResult:
    """Build one report ablation dense Reconstruction.

    Variants:
        depth_only: depth/camera unprojection, no consistency filtering.
        pointmap_only: VGGT direct point map, no consistency filtering.
        disagreement_only: depth/camera points filtered by point-map disagreement.
        reprojection_only: depth/camera points filtered by neighboring depth reprojection.
        filtered_full: depth/camera points filtered by both checks.
    """
    valid_variants = {
        "depth_only",
        "pointmap_only",
        "disagreement_only",
        "reprojection_only",
        "filtered_full",
    }
    if variant not in valid_variants:
        raise ValueError(f"Unknown dense reconstruction variant {variant!r}; expected {sorted(valid_variants)}")

    depth = _as_depth_hw(depth_map)
    conf = _as_conf_hw(depth_conf, depth.shape)
    p_depth = _normalize_world_points(points_depth, depth.shape)
    p_point = None if points_point is None else _normalize_world_points(points_point, depth.shape)
    if variant in {"pointmap_only", "disagreement_only", "filtered_full"} and p_point is None:
        raise ValueError(f"{variant} requires points_point from the VGGT point head")
    s_count, height, width = depth.shape

    flat_depth = p_depth.reshape(-1, 3)
    flat_point = p_point.reshape(-1, 3) if p_point is not None else None
    flat_conf = conf.reshape(-1)
    source_frame_ids = np.repeat(np.arange(s_count, dtype=np.int32), height * width)
    depth_flat = depth.reshape(-1)

    source_points = flat_point if variant == "pointmap_only" else flat_depth
    finite = np.all(np.isfinite(source_points), axis=1) & np.isfinite(flat_conf) & (depth_flat > 1e-8)
    if flat_point is not None:
        finite &= np.all(np.isfinite(flat_point), axis=1)
        disagreement = np.linalg.norm(flat_depth - flat_point, axis=1) / (
            np.linalg.norm(flat_depth, axis=1) + 1e-8
        )
        finite &= np.isfinite(disagreement)
    else:
        disagreement = np.full(len(flat_depth), np.nan, dtype=np.float64)

    if flat_point is not None and np.any(finite):
        disagreement_threshold = float(np.percentile(disagreement[finite], disagreement_percentile))
    else:
        disagreement_threshold = float("inf")

    keep = finite.copy()
    if variant in {"disagreement_only", "filtered_full"}:
        keep &= disagreement <= disagreement_threshold

    needs_reprojection = variant in {"reprojection_only", "filtered_full"}
    if needs_reprojection:
        votes, valid_neighbor_counts, reproj_threshold = compute_reprojection_votes(
            flat_depth,
            source_frame_ids,
            depth,
            extrinsics,
            intrinsics,
            reproj_percentile=reproj_percentile,
        )
        keep &= votes >= int(min_reproj_votes)
    else:
        votes = np.zeros(len(flat_depth), dtype=np.int16)
        valid_neighbor_counts = np.zeros(len(flat_depth), dtype=np.int16)
        reproj_threshold = float("nan")

    kept_indices = np.flatnonzero(keep)
    rng = rng or np.random.default_rng(42)
    if max_points and len(kept_indices) > max_points:
        weights = np.maximum(flat_conf[kept_indices], 1e-6).astype(np.float64)
        weights = weights / weights.sum()
        kept_indices = rng.choice(kept_indices, size=max_points, replace=False, p=weights)
        kept_indices = np.sort(kept_indices)

    points3d = source_points[kept_indices].astype(np.float64)
    points_rgb = _sample_rgb(images_np, height, width)[kept_indices]
    points_conf = flat_conf[kept_indices].astype(np.float32)

    recon = Reconstruction(
        image_names=np.asarray(image_names, dtype=str),
        image_size_hw=np.asarray(image_size_hw, dtype=np.int32),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        extrinsics=np.asarray(extrinsics, dtype=np.float64),
        points3d=points3d,
        points_rgb=points_rgb,
        points_conf=points_conf,
        obs_camera_id=np.zeros(0, dtype=np.int32),
        obs_point_id=np.zeros(0, dtype=np.int32),
        obs_xy=np.zeros((0, 2), dtype=np.float64),
        obs_conf=np.zeros(0, dtype=np.float32),
        metadata=dict(metadata or {}),
    )
    point_source = {
        "depth_only": "depth_camera_unprojection",
        "pointmap_only": "direct_vggt_point_map",
        "disagreement_only": "depth_camera_unprojection_disagreement_filtered",
        "reprojection_only": "depth_camera_unprojection_reprojection_filtered",
        "filtered_full": "depth_camera_unprojection_filtered",
    }[variant]
    if flat_point is None:
        point_map_role = "none"
    elif variant == "pointmap_only":
        point_map_role = "main_output"
    elif variant in {"disagreement_only", "filtered_full"}:
        point_map_role = "consistency_check_only"
    else:
        point_map_role = "unused"

    recon.metadata.update({
        "point_source": point_source,
        "dense_variant": variant,
        "point_map_role": point_map_role,
        "dense_filter_disagreement_percentile": float(disagreement_percentile),
        "dense_filter_disagreement_threshold": disagreement_threshold,
        "dense_filter_reproj_percentile": float(reproj_percentile),
        "dense_filter_reproj_threshold": reproj_threshold,
        "dense_filter_min_votes": int(min_reproj_votes),
        "dense_filter_max_points": int(max_points),
    })

    stats = {
        "variant": variant,
        "total_pixels": int(flat_depth.shape[0]),
        "finite_points": int(np.sum(finite)),
        "after_disagreement": int(np.sum(finite & (disagreement <= disagreement_threshold))),
        "after_reprojection": int(np.sum(keep)),
        "output_points": int(len(points3d)),
        "disagreement_percentile": float(disagreement_percentile),
        "disagreement_threshold": disagreement_threshold,
        "reproj_percentile": float(reproj_percentile),
        "reproj_threshold": reproj_threshold,
        "min_reproj_votes": int(min_reproj_votes),
        "points_with_valid_neighbors": int(np.sum(valid_neighbor_counts > 0)),
        "mean_reproj_votes_kept": float(np.mean(votes[kept_indices])) if len(kept_indices) else 0.0,
    }
    if variant in {"depth_only", "pointmap_only", "reprojection_only"}:
        stats["after_disagreement"] = int(np.sum(finite))
        stats["after_reprojection"] = int(np.sum(keep))
    elif variant == "disagreement_only":
        stats["after_reprojection"] = int(np.sum(keep))
    return GeometryFilterResult(reconstruction=recon, stats=stats, keep_mask=keep)


def save_dense_ply(path: str, points3d: np.ndarray, points_rgb: np.ndarray) -> str:
    import trimesh

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    trimesh.PointCloud(points3d, colors=points_rgb).export(path)
    return path
