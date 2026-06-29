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
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.reconstruction import Reconstruction
from src.gaussian.model import GaussianModel
from src.gaussian.renderer import render_view


class GaussianViewer:
    """Interactive viewer for Gaussian Splatting models using Viser."""

    def __init__(self, model: GaussianModel, width: int = 768, height: int = 432,
                 port: int = 8080, host: str = "127.0.0.1",
                 bg_color=(0.0, 0.0, 0.0), fixed_view: dict | None = None,
                 output_image: str | None = None,
                 recon_views: list[dict] | None = None,
                 initial_view_index: int = 0):
        """
        Args:
            model: Trained GaussianModel.
            width, height: Render resolution.
            port: Viser server port.
            host: Viser server host.
            bg_color: Background color (R, G, B) in [0, 1].
            fixed_view: Optional fixed reconstruction camera view for diagnostics.
            output_image: Optional PNG/JPEG path to save the rendered view.
            recon_views: Optional list of reconstruction views for interactive browsing.
            initial_view_index: Initial reconstruction view index in browse mode.
        """
        self.model = model
        self.width = width
        self.height = height
        self.port = port
        self.host = host
        self.fixed_view = fixed_view
        self.output_image = output_image
        self.recon_views = recon_views
        self.current_view_index = int(initial_view_index)
        self._device = model._xyz.device
        self.bg_color = torch.tensor(bg_color, device=self._device)
        self._fps_buffer = []
        xyz = model.get_xyz().detach().cpu().numpy()
        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)
        self.scene_center = ((bbox_min + bbox_max) * 0.5).astype(np.float64)
        self.scene_extent = float(np.linalg.norm(bbox_max - bbox_min))
        if self.scene_extent < 1e-3:
            self.scene_extent = 1.0

    def save_fixed_view_image(self, path: str, resolution_scale: float = 1.0,
                              bg_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        if self.fixed_view is None:
            raise ValueError("save_fixed_view_image requires a fixed reconstruction view.")

        device = self._device
        resolution = max(0.1, float(resolution_scale))
        w = int(self.fixed_view["width"] * resolution)
        h = int(self.fixed_view["height"] * resolution)
        K = self.fixed_view["K_base"].clone()
        K[0, 0] *= resolution
        K[0, 2] *= resolution
        K[1, 1] *= resolution
        K[1, 2] *= resolution
        bg = torch.tensor(list(bg_rgb), dtype=torch.float32, device=device)

        with torch.no_grad():
            result = render_view(
                model=self.model,
                world_view_transform=self.fixed_view["w2c"],
                full_proj_transform=self.fixed_view["w2c"],
                camera_center=self.fixed_view["camera_center"],
                K=K,
                width=w,
                height=h,
                bg_color=bg,
            )
            rendered = result["render"]

        img_np = (rendered.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        img_np = img_np.transpose(1, 2, 0)
        Image.fromarray(img_np).save(path)
        print(f"Saved render to {path}")

    def run(self):
        """Start Viser server and register render callbacks."""
        import viser
        import viser.transforms as vtf

        server = viser.ViserServer(host=self.host, port=self.port)
        device = self._device

        print(f"Viewer running at http://{self.host}:{self.port}")
        print(f"Gaussians: {self.model.num_gaussians}")

        # Add GUI controls
        with server.gui.add_folder("Render Settings"):
            resolution_slider = server.gui.add_slider(
                "Resolution Scale", min=0.25, max=1.0, step=0.25,
                initial_value=1.0 if (self.fixed_view is not None or self.recon_views is not None) else 0.5
            )
            bg_color_r = server.gui.add_slider(
                "BG Red", min=0.0, max=1.0, step=0.05, initial_value=0.0
            )
            bg_color_g = server.gui.add_slider(
                "BG Green", min=0.0, max=1.0, step=0.05, initial_value=0.0
            )
            bg_color_b = server.gui.add_slider(
                "BG Blue", min=0.0, max=1.0, step=0.05, initial_value=0.0
            )

        with server.gui.add_folder("Info"):
            fps_text = server.gui.add_text("FPS", initial_value="0.0")
            gaussian_count = server.gui.add_text(
                "Gaussians", initial_value=str(self.model.num_gaussians)
            )
            if self.recon_views is not None:
                view_name_text = server.gui.add_text(
                    "Image", initial_value=self.recon_views[self.current_view_index]["image_name"]
                )
        if self.recon_views is not None:
            with server.gui.add_folder("Dataset View"):
                view_slider = server.gui.add_slider(
                    "View Index",
                    min=0,
                    max=len(self.recon_views) - 1,
                    step=1,
                    initial_value=self.current_view_index,
                )

        # FPS tracking
        last_time = [time.time()]
        frame_count = [0]
        render_busy = {}

        def render_dataset_view(view: dict):
            resolution = max(0.1, float(resolution_slider.value))
            w = int(view["width"] * resolution)
            h = int(view["height"] * resolution)
            K = view["K_base"].clone()
            K[0, 0] *= resolution
            K[0, 2] *= resolution
            K[1, 1] *= resolution
            K[1, 2] *= resolution

            bg = torch.tensor([
                float(bg_color_r.value),
                float(bg_color_g.value),
                float(bg_color_b.value),
            ], device=device)

            with torch.no_grad():
                result = render_view(
                    model=self.model,
                    world_view_transform=view["w2c"],
                    full_proj_transform=view["w2c"],
                    camera_center=view["camera_center"],
                    K=K,
                    width=w,
                    height=h,
                    bg_color=bg,
                )
                rendered = result["render"]

            img_np = (rendered.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            return img_np.transpose(1, 2, 0)

        def update_all_clients_background():
            if self.recon_views is None:
                return
            view = self.recon_views[self.current_view_index]
            img = render_dataset_view(view)
            if self.output_image is not None:
                Image.fromarray(img).save(self.output_image)
                print(f"Saved render to {self.output_image}")
                self.output_image = None
            for client in server.get_clients().values():
                client.scene.set_background_image(img, format="jpeg")
            view_name_text.value = view["image_name"]

        @server.on_client_connect
        def _(client: viser.ClientHandle):
            print(f"Client connected: {client.client_id}")
            render_busy[client.client_id] = False

            def update_background(camera: viser.CameraHandle):
                if self.recon_views is not None:
                    view = self.recon_views[self.current_view_index]
                    img = render_dataset_view(view)
                    client.scene.set_background_image(img, format="jpeg")
                    return
                if render_busy[client.client_id]:
                    return
                render_busy[client.client_id] = True
                try:
                    img = render_for_fixed_view() if self.fixed_view is not None else render_for_camera(camera)
                    client.scene.set_background_image(img, format="jpeg")
                except Exception as e:
                    print(f"Render error: {e}")
                finally:
                    render_busy[client.client_id] = False

            @client.camera.on_update
            def _(camera: viser.CameraHandle):
                if self.fixed_view is None and self.recon_views is None:
                    update_background(camera)

            if self.fixed_view is None and self.recon_views is None:
                client.camera.look_at = self.scene_center
                client.camera.position = self.scene_center + np.array(
                    [0.0, 0.0, -2.0 * self.scene_extent], dtype=np.float64
                )
                client.camera.up_direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            update_background(client.camera)

        if self.recon_views is not None:
            @view_slider.on_update
            def _(_handle):
                self.current_view_index = int(view_slider.value)
                update_all_clients_background()

            @resolution_slider.on_update
            def _(_handle):
                update_all_clients_background()

            @bg_color_r.on_update
            def _(_handle):
                update_all_clients_background()

            @bg_color_g.on_update
            def _(_handle):
                update_all_clients_background()

            @bg_color_b.on_update
            def _(_handle):
                update_all_clients_background()

        def render_for_camera(camera: viser.CameraHandle):
            """Render the Gaussian model from the current Viser camera."""
            nonlocal last_time, frame_count

            cam_pos = np.asarray(camera.position, dtype=np.float32)
            R_world_camera = vtf.SO3(np.asarray(camera.wxyz)).as_matrix().astype(np.float32)
            R = torch.tensor(R_world_camera.T, dtype=torch.float32, device=device)
            cam_pos_t = torch.tensor(cam_pos, dtype=torch.float32, device=device)
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

            with torch.no_grad():
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
            img_np = (rendered.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            img_np = img_np.transpose(1, 2, 0)

            return img_np

        def render_for_fixed_view():
            resolution = max(0.1, float(resolution_slider.value))
            w = int(self.fixed_view["width"] * resolution)
            h = int(self.fixed_view["height"] * resolution)
            K = self.fixed_view["K_base"].clone()
            K[0, 0] *= resolution
            K[0, 2] *= resolution
            K[1, 1] *= resolution
            K[1, 2] *= resolution

            bg = torch.tensor([
                float(bg_color_r.value),
                float(bg_color_g.value),
                float(bg_color_b.value),
            ], device=device)

            with torch.no_grad():
                result = render_view(
                    model=self.model,
                    world_view_transform=self.fixed_view["w2c"],
                    full_proj_transform=self.fixed_view["w2c"],
                    camera_center=self.fixed_view["camera_center"],
                    K=K,
                    width=w,
                    height=h,
                    bg_color=bg,
                )
                rendered = result["render"]

            img_np = (rendered.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            return img_np.transpose(1, 2, 0)

        print("Viewer running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down viewer...")


def make_viewer_from_checkpoint(checkpoint_path: str, width: int = 768,
                                height: int = 432, port: int = 8080,
                                host: str = "127.0.0.1",
                                device: str = "cuda",
                                fixed_view: dict | None = None,
                                output_image: str | None = None,
                                recon_views: list[dict] | None = None,
                                initial_view_index: int = 0) -> GaussianViewer:
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

    return GaussianViewer(
        model,
        width=width,
        height=height,
        port=port,
        host=host,
        fixed_view=fixed_view,
        output_image=output_image,
        recon_views=recon_views,
        initial_view_index=initial_view_index,
    )


def build_fixed_view(reconstruction_path: str, view_index: int,
                     width: int, height: int, device: str) -> dict:
    recon = Reconstruction.from_npz(reconstruction_path)
    if view_index < 0 or view_index >= recon.num_images:
        raise ValueError(f"view_index {view_index} out of range [0, {recon.num_images - 1}]")

    T = recon.extrinsics[view_index]
    R = torch.tensor(T[:3, :3], dtype=torch.float32, device=device)
    t = torch.tensor(T[:3, 3], dtype=torch.float32, device=device)

    w2c = torch.eye(4, dtype=torch.float32, device=device)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    camera_center = -R.T @ t

    K = recon.intrinsics[view_index].copy()
    scale_w = width / float(recon.image_size_hw[view_index, 1])
    scale_h = height / float(recon.image_size_hw[view_index, 0])
    K[0, 0] *= scale_w
    K[0, 2] *= scale_w
    K[1, 1] *= scale_h
    K[1, 2] *= scale_h

    return {
        "w2c": w2c,
        "camera_center": camera_center,
        "K_base": torch.tensor(K, dtype=torch.float32, device=device),
        "width": width,
        "height": height,
        "image_name": str(recon.image_names[view_index]),
    }


def build_reconstruction_views(reconstruction_path: str, width: int,
                               height: int, device: str) -> list[dict]:
    recon = Reconstruction.from_npz(reconstruction_path)
    views = []
    for view_index in range(recon.num_images):
        T = recon.extrinsics[view_index]
        R = torch.tensor(T[:3, :3], dtype=torch.float32, device=device)
        t = torch.tensor(T[:3, 3], dtype=torch.float32, device=device)

        w2c = torch.eye(4, dtype=torch.float32, device=device)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        camera_center = -R.T @ t

        K = recon.intrinsics[view_index].copy()
        scale_w = width / float(recon.image_size_hw[view_index, 1])
        scale_h = height / float(recon.image_size_hw[view_index, 0])
        K[0, 0] *= scale_w
        K[0, 2] *= scale_w
        K[1, 1] *= scale_h
        K[1, 2] *= scale_h

        views.append({
            "w2c": w2c,
            "camera_center": camera_center,
            "K_base": torch.tensor(K, dtype=torch.float32, device=device),
            "width": width,
            "height": height,
            "image_name": str(recon.image_names[view_index]),
        })
    return views


def parse_args():
    parser = argparse.ArgumentParser(description="3DGS Viser Viewer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--reconstruction", type=str, default=None,
                        help="Optional reconstruction.npz for fixed-camera diagnostics")
    parser.add_argument("--view_index", type=int, default=0,
                        help="Camera index used with --reconstruction")
    parser.add_argument("--output_image", type=str, default=None,
                        help="Optional output image path for saving the rendered view")
    parser.add_argument("--keep_viewer", action="store_true", default=False,
                        help="If set together with --output_image, save the image and keep the viewer running")
    parser.add_argument("--width", type=int, default=768, help="Render width")
    parser.add_argument("--height", type=int, default=432, help="Render height")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Viser server host")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fixed_view = None
    recon_views = None
    if args.reconstruction is not None:
        recon_views = build_reconstruction_views(
            reconstruction_path=args.reconstruction,
            width=args.width,
            height=args.height,
            device=args.device,
        )
        fixed_view = recon_views[args.view_index]

    print(f"Loading checkpoint: {args.checkpoint}")
    viewer = make_viewer_from_checkpoint(
        args.checkpoint,
        width=args.width,
        height=args.height,
        port=args.port,
        host=args.host,
        device=args.device,
        fixed_view=fixed_view,
        output_image=args.output_image,
        recon_views=recon_views if args.output_image is None else None,
        initial_view_index=args.view_index,
    )
    if args.output_image is not None and fixed_view is not None:
        output_dir = os.path.dirname(os.path.abspath(args.output_image))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        viewer.save_fixed_view_image(args.output_image)
        if not args.keep_viewer:
            raise SystemExit(0)
    viewer.run()
