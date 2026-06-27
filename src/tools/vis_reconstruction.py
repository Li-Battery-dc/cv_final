#!/usr/bin/env python3
"""Visualize a Reconstruction with viser.

Usage:
    python -m src.tools.vis_reconstruction \
        --reconstruction data/scene/vggt_raw/reconstruction.npz \
        --image_dir data/scene/images \
        --port 8080
"""

import os
import sys
import math
import time
import argparse

import numpy as np
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.reconstruction import Reconstruction


def parse_args():
    parser = argparse.ArgumentParser(description="Viser viewer for reconstruction.npz")
    parser.add_argument("--reconstruction", type=str, required=True,
                        help="Path to reconstruction.npz")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Optional image directory for camera frustum thumbnails")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the viser server")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to bind the viser server")
    parser.add_argument("--max_points", type=int, default=50000,
                        help="Maximum number of sparse points to visualize")
    parser.add_argument("--camera_scale", type=float, default=0.06,
                        help="Scale used for camera axes and frustums")
    parser.add_argument("--point_size", type=float, default=0.008,
                        help="Rendered point size")
    parser.add_argument("--max_image_dim", type=int, default=160,
                        help="Max thumbnail dimension for frustum images")
    return parser.parse_args()


def _camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    centers = []
    for ext in extrinsics:
        R = ext[:, :3]
        t = ext[:, 3]
        centers.append(-R.T @ t)
    return np.asarray(centers, dtype=np.float64)


def _cam_to_world_matrix(ext: np.ndarray) -> np.ndarray:
    R = ext[:, :3]
    t = ext[:, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def _sample_indices(points_conf: np.ndarray, max_points: int) -> np.ndarray:
    n_points = len(points_conf)
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int32)

    order = np.argsort(points_conf)
    keep_tail = order[-max_points // 2:]
    keep_rand = np.random.choice(n_points, size=max_points - len(keep_tail), replace=False)
    sampled = np.unique(np.concatenate([keep_tail, keep_rand]))
    if len(sampled) > max_points:
        sampled = sampled[:max_points]
    return sampled.astype(np.int32)


def _load_thumbnail(image_path: str | None, max_image_dim: int) -> np.ndarray | None:
    if image_path is None or not os.path.exists(image_path):
        return None

    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    scale = min(1.0, float(max_image_dim) / max(width, height))
    if scale < 1.0:
        img = img.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.BILINEAR)
    return np.asarray(img)


def main():
    args = parse_args()

    import viser
    import viser.transforms as viser_tf

    recon = Reconstruction.from_npz(args.reconstruction)
    if recon.num_points == 0:
        raise ValueError("Reconstruction has no sparse points.")

    np.random.seed(42)

    point_ids = _sample_indices(recon.points_conf, args.max_points)
    points = recon.points3d[point_ids]
    colors = recon.points_rgb[point_ids]
    conf = recon.points_conf[point_ids]
    cam_centers = _camera_centers_from_extrinsics(recon.extrinsics)

    scene_center = np.mean(
        np.concatenate([points, cam_centers], axis=0),
        axis=0,
    )
    point_extent = np.max(np.linalg.norm(points - scene_center[None, :], axis=1))
    cam_extent = np.max(np.linalg.norm(cam_centers - scene_center[None, :], axis=1))
    scene_radius = max(float(point_extent), float(cam_extent), 1e-3)

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Viewer running at http://{args.host}:{args.port}")
    print(f"Reconstruction: {args.reconstruction}")
    print(f"Images: {recon.num_images}, sparse points: {recon.num_points}, shown points: {len(point_ids)}")

    gui_show_cameras = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_show_images = server.gui.add_checkbox(
        "Show Frustum Images",
        initial_value=args.image_dir is not None,
    )
    gui_conf_percent = server.gui.add_slider(
        "Confidence Percentile",
        min=0.0,
        max=100.0,
        step=0.5,
        initial_value=10.0,
    )
    gui_point_size = server.gui.add_slider(
        "Point Size",
        min=0.001,
        max=0.03,
        step=0.001,
        initial_value=args.point_size,
    )

    threshold = np.percentile(conf, gui_conf_percent.value)
    mask = conf >= threshold
    point_cloud = server.scene.add_point_cloud(
        "reconstruction/points",
        points=points[mask],
        colors=colors[mask],
        point_size=args.point_size,
        point_shape="circle",
    )

    camera_frames = []
    camera_frustums = []

    for idx in range(recon.num_images):
        T_world_camera = viser_tf.SE3.from_matrix(_cam_to_world_matrix(recon.extrinsics[idx])[:3, :])
        frame = server.scene.add_frame(
            f"reconstruction/cameras/{idx}",
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=args.camera_scale,
            axes_radius=args.camera_scale * 0.04,
            origin_radius=args.camera_scale * 0.04,
        )
        camera_frames.append(frame)

        image = None
        if args.image_dir is not None:
            image_path = os.path.join(args.image_dir, str(recon.image_names[idx]))
            image = _load_thumbnail(image_path, args.max_image_dim)

        height = float(recon.image_size_hw[idx, 0])
        width = float(recon.image_size_hw[idx, 1])
        fy = float(recon.intrinsics[idx, 1, 1])
        fov = 2.0 * math.atan2(height / 2.0, max(fy, 1e-6))
        frustum = server.scene.add_camera_frustum(
            f"reconstruction/cameras/{idx}/frustum",
            fov=fov,
            aspect=max(width / max(height, 1.0), 1e-6),
            scale=args.camera_scale,
            image=image,
            line_width=1.0,
        )
        frustum.visible = bool(gui_show_images.value and gui_show_cameras.value)
        camera_frustums.append(frustum)

    @gui_conf_percent.on_update
    def _(_) -> None:
        threshold = np.percentile(conf, gui_conf_percent.value)
        mask = conf >= threshold
        point_cloud.points = points[mask]
        point_cloud.colors = colors[mask]

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = float(gui_point_size.value)

    @gui_show_cameras.on_update
    def _(_) -> None:
        visible = bool(gui_show_cameras.value)
        for frame in camera_frames:
            frame.visible = visible
        for frustum in camera_frustums:
            frustum.visible = visible and bool(gui_show_images.value)

    @gui_show_images.on_update
    def _(_) -> None:
        visible = bool(gui_show_cameras.value and gui_show_images.value)
        for frustum in camera_frustums:
            frustum.visible = visible

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.look_at = scene_center
        client.camera.position = scene_center + np.array([0.0, -0.6 * scene_radius, 0.25 * scene_radius])

    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


if __name__ == "__main__":
    main()
