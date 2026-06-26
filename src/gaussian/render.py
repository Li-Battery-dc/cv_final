#!/usr/bin/env python3
"""Headless renderer for trained 3D Gaussian Splatting checkpoints.

Renders checkpoint views using cameras from a Reconstruction file. This is useful
on servers without a graphical desktop.
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.reconstruction import Reconstruction
from src.gaussian.model import GaussianModel
from src.gaussian.renderer import render_view


def parse_args():
    parser = argparse.ArgumentParser(description="Headless 3DGS renderer")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--reconstruction", required=True, help="Path to reconstruction.npz")
    parser.add_argument("--image_dir", default=None, help="Optional original images for GT side-by-side")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--resolution", type=int, nargs=2, default=[320, 180], metavar=("W", "H"))
    parser.add_argument("--views", type=str, default="val",
                        help="'all', 'val', or comma-separated view indices, e.g. 0,4,8")
    parser.add_argument("--val_every", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--side_by_side", action="store_true",
                        help="Save render next to GT image when --image_dir is provided")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> GaussianModel:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GaussianModel(
        sh_degree=state.get("active_sh_degree", 0),
        max_sh_degree=state.get("_max_sh_degree", 2),
    )
    model._xyz = torch.nn.Parameter(state["_xyz"].to(device))
    model._log_scales = torch.nn.Parameter(state["_log_scales"].to(device))
    model._quaternions = torch.nn.Parameter(state["_quaternions"].to(device))
    model._log_opacities = torch.nn.Parameter(state["_log_opacities"].to(device))
    model._features_dc = torch.nn.Parameter(state["_features_dc"].to(device))
    model._features_rest = torch.nn.Parameter(state["_features_rest"].to(device))
    model.active_sh_degree = state.get("active_sh_degree", 0)
    model.eval()
    return model


def parse_view_ids(spec: str, n_views: int, val_every: int):
    if spec == "all":
        return list(range(n_views))
    if spec == "val":
        return [i for i in range(n_views) if i % val_every == 0]
    ids = [int(x) for x in spec.split(",") if x.strip()]
    return [i for i in ids if 0 <= i < n_views]


def build_view(recon: Reconstruction, idx: int, width: int, height: int, device: str):
    T = recon.extrinsics[idx]
    R = torch.tensor(T[:3, :3], dtype=torch.float32, device=device)
    t = torch.tensor(T[:3, 3], dtype=torch.float32, device=device)

    w2c = torch.eye(4, device=device)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    cam_center = -R.T @ t

    K = recon.intrinsics[idx].copy()
    scale_w = width / float(recon.image_size_hw[idx, 1])
    scale_h = height / float(recon.image_size_hw[idx, 0])
    K[0, 0] *= scale_w
    K[0, 2] *= scale_w
    K[1, 1] *= scale_h
    K[1, 2] *= scale_h
    K = torch.tensor(K, dtype=torch.float32, device=device)
    return w2c, cam_center, K


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    array = array.transpose(1, 2, 0)
    return Image.fromarray(array)


def load_gt(image_dir: str, image_name: str, width: int, height: int) -> Image.Image:
    img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
    return img.resize((width, height), Image.BILINEAR)


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    width, height = args.resolution
    os.makedirs(args.output, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print(f"Gaussians: {model.num_gaussians}, SH degree: {model.active_sh_degree}")

    print(f"Loading reconstruction: {args.reconstruction}")
    recon = Reconstruction.from_npz(args.reconstruction)
    view_ids = parse_view_ids(args.views, recon.num_images, args.val_every)
    print(f"Rendering {len(view_ids)} views at {width}x{height}: {view_ids}")

    bg = torch.tensor(args.bg, dtype=torch.float32, device=device)

    with torch.no_grad():
        for idx in view_ids:
            w2c, cam_center, K = build_view(recon, idx, width, height, device)
            result = render_view(
                model=model,
                world_view_transform=w2c,
                full_proj_transform=w2c,
                camera_center=cam_center,
                K=K,
                width=width,
                height=height,
                bg_color=bg,
            )
            render_img = tensor_to_image(result["render"])

            stem = os.path.splitext(str(recon.image_names[idx]))[0]
            out_path = os.path.join(args.output, f"{idx:04d}_{stem}_render.png")
            if args.side_by_side and args.image_dir:
                gt_img = load_gt(args.image_dir, str(recon.image_names[idx]), width, height)
                canvas = Image.new("RGB", (width * 2, height))
                canvas.paste(gt_img, (0, 0))
                canvas.paste(render_img, (width, 0))
                out_path = os.path.join(args.output, f"{idx:04d}_{stem}_compare.png")
                canvas.save(out_path)
            else:
                render_img.save(out_path)
            print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
