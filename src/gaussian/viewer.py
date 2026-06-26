#!/usr/bin/env python3
"""Viser-based interactive viewer for trained 3DGS models.

Usage:
    python -m src.gaussian.viewer --checkpoint <path.pt> [--port 8080]
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.gaussian.model import GaussianModel
from src.gaussian.renderer import render_view


class GaussianViewer:
    """Interactive viewer for Gaussian Splatting models using Viser."""

    def __init__(self, model: GaussianModel, width: int = 768, height: int = 432,
                 port: int = 8080, bg_color=(0.0, 0.0, 0.0)):
        """
        Args:
            model: Trained GaussianModel.
            width, height: Render resolution.
            port: Viser server port.
            bg_color: Background color (R, G, B) in [0, 1].
        """
        self.model = model
        self.width = width
        self.height = height
        self.port = port
        self._device = model._xyz.device
        self.bg_color = torch.tensor(bg_color, device=self._device)
        self._fps_buffer = []

    def run(self):
        """Start Viser server and register render callbacks."""
        import viser

        server = viser.ViserServer(port=self.port)
        device = self._device

        print(f"Viewer running at http://localhost:{self.port}")
        print(f"Gaussians: {self.model.num_gaussians}")

        # Add GUI controls
        with server.gui_folder("Render Settings"):
            resolution_slider = server.gui_slider("Resolution Scale", 0.25, 1.0, 0.5, step=0.25)
            bg_color_r = server.gui_slider("BG Red", 0.0, 1.0, 0.0, step=0.05)
            bg_color_g = server.gui_slider("BG Green", 0.0, 1.0, 0.0, step=0.05)
            bg_color_b = server.gui_slider("BG Blue", 0.0, 1.0, 0.0, step=0.05)

        with server.gui_folder("Info"):
            fps_text = server.gui_text("FPS", "0.0")
            gaussian_count = server.gui_text("Gaussians", str(self.model.num_gaussians))

        # FPS tracking
        last_time = [time.time()]
        frame_count = [0]

        @server.on_client_connect
        def _(client: viser.ClientHandle):
            print(f"Client connected: {client.client_id}")

        def render_for_camera(camera: viser.CameraHandle):
            """Render the Gaussian model from the current Viser camera."""
            nonlocal last_time, frame_count

            # Get camera pose from Viser
            # Viser camera: position and look_at
            cam_pos = camera.position  # (3,) world position
            look_at = camera.look_at   # (3,) target point
            up = camera.up_direction   # (3,)

            cam_pos_t = torch.tensor(cam_pos, dtype=torch.float32, device=device)
            look_at_t = torch.tensor(look_at, dtype=torch.float32, device=device)
            up_t = torch.tensor(up, dtype=torch.float32, device=device)

            # Build world-to-camera matrix (OpenCV convention).
            forward = cam_pos_t - look_at_t
            forward = forward / torch.norm(forward)

            right = torch.cross(up_t, forward, dim=0)
            right = right / torch.norm(right)

            new_up = torch.cross(forward, right, dim=0)
            new_up = new_up / torch.norm(new_up)

            R = torch.stack([right, new_up, forward], dim=0)  # (3, 3)
            t = -R @ cam_pos_t

            w2c = torch.eye(4, dtype=torch.float32, device=device)
            w2c[:3, :3] = R
            w2c[:3, 3] = t

            # Render
            resolution = max(0.1, float(resolution_slider.value))
            w, h = int(self.width * resolution), int(self.height * resolution)

            # Build intrinsics (using FOV from Viser camera)
            fov = camera.fov  # vertical FOV in radians
            fy = (h / 2.0) / torch.tan(torch.tensor(fov / 2.0, device=device))
            fx = fy
            cx = w / 2.0
            cy = h / 2.0

            K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=torch.float32, device=device)

            bg = torch.tensor([
                float(bg_color_r.value),
                float(bg_color_g.value),
                float(bg_color_b.value),
            ], device=device)

            result = render_view(
                model=self.model,
                world_view_transform=w2c,
                full_proj_transform=w2c,
                camera_center=cam_pos_t,
                K=K,
                width=w,
                height=h,
                bg_color=bg,
            )

            rendered = result['render']

            # FPS tracking
            frame_count[0] += 1
            now = time.time()
            if now - last_time[0] >= 1.0:
                fps = frame_count[0] / (now - last_time[0])
                fps_text.value = f"{fps:.1f}"
                frame_count[0] = 0
                last_time[0] = now

            # Convert to numpy image for Viser
            img_np = (rendered.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            img_np = img_np.transpose(1, 2, 0)

            return img_np

        # Register render callback
        @server.on_client_camera_update
        def _(camera: viser.CameraHandle):
            try:
                img = render_for_camera(camera)
                server.scene.add_image(
                    img,
                    render_width=self.width,
                    render_height=self.height,
                )
            except Exception as e:
                print(f"Render error: {e}")

        print("Viewer running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down viewer...")


def make_viewer_from_checkpoint(checkpoint_path: str, width: int = 768,
                                height: int = 432, port: int = 8080,
                                device: str = "cuda") -> GaussianViewer:
    """Create a viewer from a checkpoint file."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    sh_degree = state.get('active_sh_degree', 0)
    max_sh = state.get('_max_sh_degree', 2)
    model = GaussianModel(sh_degree=sh_degree, max_sh_degree=max_sh)
    model._xyz = torch.nn.Parameter(state['_xyz'].to(device))
    model._log_scales = torch.nn.Parameter(state['_log_scales'].to(device))
    model._quaternions = torch.nn.Parameter(state['_quaternions'].to(device))
    model._log_opacities = torch.nn.Parameter(state['_log_opacities'].to(device))
    model._features_dc = torch.nn.Parameter(state['_features_dc'].to(device))
    model._features_rest = torch.nn.Parameter(state['_features_rest'].to(device))
    model.active_sh_degree = sh_degree

    return GaussianViewer(model, width=width, height=height, port=port)


def parse_args():
    parser = argparse.ArgumentParser(description="3DGS Viser Viewer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--width", type=int, default=768, help="Render width")
    parser.add_argument("--height", type=int, default=432, help="Render height")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    viewer = make_viewer_from_checkpoint(
        args.checkpoint,
        width=args.width,
        height=args.height,
        port=args.port,
        device=args.device,
    )
    viewer.run()
