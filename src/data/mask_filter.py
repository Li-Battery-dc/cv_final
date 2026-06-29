"""Mask-based filtering for Reconstruction observations and points."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.data.reconstruction import Reconstruction


@dataclass
class MaskFilterStats:
    mask_dir: str
    threshold: float
    min_observations: int
    min_ratio: float
    points_before: int
    points_after: int
    observations_before: int
    observations_after: int
    foreground_observations: int


def _mask_path(mask_dir: str, image_name: str) -> str:
    stem, _ = os.path.splitext(os.path.basename(image_name))
    candidates = [
        os.path.join(mask_dir, image_name),
        os.path.join(mask_dir, f"{stem}.png"),
        os.path.join(mask_dir, f"{stem}.jpg"),
        os.path.join(mask_dir, f"{stem}.jpeg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No mask found for image {image_name!r} in {mask_dir}")


def _load_masks(mask_dir: str, image_names: np.ndarray) -> list[np.ndarray]:
    masks = []
    for name in image_names:
        path = _mask_path(mask_dir, str(name))
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        masks.append(mask)
    return masks


def _observation_xy_in_mask_space(recon: Reconstruction) -> np.ndarray:
    """Return observation xy coordinates in original image pixel space."""
    if recon.metadata.get("colmap_space", False):
        return recon.obs_xy.copy()

    original_coords = recon.metadata.get("original_coords")
    img_load_resolution = recon.metadata.get("img_load_resolution")
    if original_coords is None or img_load_resolution is None:
        return recon.obs_xy.copy()

    original_coords = np.asarray(original_coords, dtype=np.float64)
    xy = recon.obs_xy.copy()
    for cam_id in range(recon.num_images):
        cam_mask = recon.obs_camera_id == cam_id
        if not np.any(cam_mask):
            continue
        x1, y1, _x2, _y2, orig_w, orig_h = original_coords[cam_id]
        resize_ratio = max(orig_w, orig_h) / float(img_load_resolution)
        xy[cam_mask] = (xy[cam_mask] - np.array([x1, y1], dtype=np.float64)) * resize_ratio
    return xy


def foreground_observation_mask(
    recon: Reconstruction,
    mask_dir: str,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute a boolean foreground flag for every observation."""
    masks = _load_masks(mask_dir, recon.image_names)
    xy = _observation_xy_in_mask_space(recon)
    fg = np.zeros(recon.num_observations, dtype=bool)

    for cam_id, mask_img in enumerate(masks):
        obs_mask = recon.obs_camera_id == cam_id
        if not np.any(obs_mask):
            continue
        coords = np.round(xy[obs_mask]).astype(np.int64)
        h, w = mask_img.shape
        inside = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < w)
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < h)
        )
        local = np.zeros(coords.shape[0], dtype=bool)
        if np.any(inside):
            vals = mask_img[coords[inside, 1], coords[inside, 0]]
            local[inside] = vals >= threshold
        fg[obs_mask] = local

    return fg


def filter_reconstruction_by_masks(
    recon: Reconstruction,
    mask_dir: str,
    threshold: float = 0.5,
    min_observations: int = 2,
    min_ratio: float = 0.5,
) -> tuple[Reconstruction, dict]:
    """Keep points and observations supported by foreground mask pixels.

    The input reconstruction coordinate space is preserved. If VGGT square
    padding metadata is present, observations are mapped back to original image
    coordinates only for mask lookup.
    """
    fg_obs = foreground_observation_mask(recon, mask_dir, threshold)
    point_total = np.bincount(recon.obs_point_id, minlength=recon.num_points)
    point_fg = np.bincount(recon.obs_point_id[fg_obs], minlength=recon.num_points)
    ratio = np.divide(
        point_fg,
        np.maximum(point_total, 1),
        out=np.zeros_like(point_fg, dtype=np.float64),
        where=point_total > 0,
    )
    keep_points = (point_fg >= min_observations) & (ratio >= min_ratio)

    old_to_new = np.full(recon.num_points, -1, dtype=np.int32)
    old_to_new[keep_points] = np.arange(int(np.sum(keep_points)), dtype=np.int32)
    keep_obs = fg_obs & (old_to_new[recon.obs_point_id] >= 0)

    filtered = recon.copy()
    filtered.points3d = filtered.points3d[keep_points]
    filtered.points_rgb = filtered.points_rgb[keep_points]
    filtered.points_conf = filtered.points_conf[keep_points]
    filtered.obs_camera_id = filtered.obs_camera_id[keep_obs]
    filtered.obs_point_id = old_to_new[filtered.obs_point_id[keep_obs]]
    filtered.obs_xy = filtered.obs_xy[keep_obs]
    filtered.obs_conf = filtered.obs_conf[keep_obs]

    stats = MaskFilterStats(
        mask_dir=os.path.abspath(mask_dir),
        threshold=float(threshold),
        min_observations=int(min_observations),
        min_ratio=float(min_ratio),
        points_before=int(recon.num_points),
        points_after=int(filtered.num_points),
        observations_before=int(recon.num_observations),
        observations_after=int(filtered.num_observations),
        foreground_observations=int(np.sum(fg_obs)),
    ).__dict__
    filtered.metadata["mask_filter"] = stats
    return filtered, stats
