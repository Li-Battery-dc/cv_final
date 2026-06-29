#!/usr/bin/env python3
"""Wrapper around the official graphdeco-inria/gaussian-splatting training code."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.colmap_io import reconstruction_to_colmap_space, reconstruction_to_colmap_sparse
from src.data.mask_filter import filter_reconstruction_by_masks
from src.data.reconstruction import Reconstruction
from src.utils.experiment import (
    prepare_output_dir,
    save_json,
    save_run_metadata,
    utc_timestamp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Official Gaussian Splatting baseline runner")
    parser.add_argument("--repo", type=str, default="gaussian-splatting",
                        help="Path to the official gaussian-splatting repository")
    parser.add_argument("--official_python", type=str, default=sys.executable,
                        help="Python executable from the environment that has official 3DGS dependencies")
    parser.add_argument("--reconstruction", type=str, default=None,
                        help="Optional reconstruction.npz used to generate COLMAP sparse input")
    parser.add_argument("--sparse_dir", type=str, default=None,
                        help="Optional existing COLMAP sparse directory")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Image directory for the scene")
    parser.add_argument("--output", type=str, required=True,
                        help="Output root directory for official training")
    parser.add_argument("--output_run_dir", type=str, default=None,
                        help="Optional explicit output run directory")
    parser.add_argument("--use_timestamp", action=argparse.BooleanOptionalAction, default=True,
                        help="If true, save to output/runs/<timestamp>_gaussian_official")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE",
                        help="COLMAP camera model when reconstructing sparse input from npz")
    parser.add_argument("--init_mode", type=str, default="reconstruction",
                        choices=("reconstruction", "random"),
                        help="Official 3DGS point-cloud initialization source")
    parser.add_argument("--random_init_points", type=int, default=15000,
                        help="Number of random initial points when --init_mode=random")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for official random point initialization")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--resolution", type=int, default=768,
                        help="Official code uses width or scale factor here")
    parser.add_argument("--sh_degree", type=int, default=2)
    parser.add_argument("--checkpoint_iterations", nargs="*", type=int, default=None)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=None)
    parser.add_argument("--skip_render_metrics", action="store_true", default=False,
                        help="Skip official render.py / metrics.py after training")
    parser.add_argument("--white_background", action="store_true", default=False)
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Optional foreground masks aligned with image names")
    parser.add_argument("--mask_background", type=str, default="none",
                        choices=("none", "white", "black"),
                        help="Composite masked images onto a constant background")
    parser.add_argument("--mask_points", action=argparse.BooleanOptionalAction, default=False,
                        help="Filter reconstruction points/observations by foreground masks before COLMAP export")
    parser.add_argument("--mask_foreground_threshold", type=float, default=0.5)
    parser.add_argument("--mask_min_observations", type=int, default=2)
    parser.add_argument("--mask_min_ratio", type=float, default=0.5)
    parser.add_argument("--test_every", type=int, default=None,
                        help="Write sparse/0/test.txt using every Nth sorted image as test")
    parser.add_argument("--test_names", nargs="*", default=None,
                        help="Explicit image names to write to sparse/0/test.txt")
    parser.add_argument("--test_list", type=str, default=None,
                        help="Text file containing one test image name per line")
    return parser.parse_args()


def _replace_path(target: str):
    if os.path.islink(target) or os.path.isfile(target):
        os.unlink(target)
    elif os.path.isdir(target):
        shutil.rmtree(target)


def _link_or_copy(src: str, dst: str):
    src = os.path.abspath(src)
    if os.path.lexists(dst):
        _replace_path(dst)
    try:
        os.symlink(src, dst, target_is_directory=os.path.isdir(src))
    except OSError:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _find_mask(mask_dir: str, image_name: str) -> str:
    stem, _ = os.path.splitext(os.path.basename(image_name))
    for candidate in (
        os.path.join(mask_dir, image_name),
        os.path.join(mask_dir, f"{stem}.png"),
        os.path.join(mask_dir, f"{stem}.jpg"),
        os.path.join(mask_dir, f"{stem}.jpeg"),
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No mask found for {image_name!r} in {mask_dir}")


def _prepare_images(image_dir: str, output_image_dir: str, args) -> dict:
    if args.mask_background == "none":
        _link_or_copy(image_dir, output_image_dir)
        return {"mode": "linked", "image_dir": os.path.abspath(image_dir)}

    if not args.mask_dir:
        raise ValueError("--mask_background requires --mask_dir")

    os.makedirs(output_image_dir, exist_ok=True)
    bg_value = 255 if args.mask_background == "white" else 0
    image_names = sorted(
        name for name in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, name))
    )
    for name in image_names:
        image = Image.open(os.path.join(image_dir, name)).convert("RGB")
        mask = Image.open(_find_mask(args.mask_dir, name)).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        mask_np = (np.asarray(mask, dtype=np.float32) / 255.0) >= args.mask_foreground_threshold
        image_np = np.asarray(image, dtype=np.uint8)
        bg = np.full_like(image_np, bg_value, dtype=np.uint8)
        out = np.where(mask_np[..., None], image_np, bg)
        Image.fromarray(out).save(os.path.join(output_image_dir, name))

    return {
        "mode": args.mask_background,
        "mask_dir": os.path.abspath(args.mask_dir),
        "image_count": len(image_names),
    }


def _test_names_from_args(args, image_names: list[str]) -> list[str]:
    names = []
    if args.test_list:
        with open(args.test_list, "r", encoding="utf-8") as f:
            names.extend(line.strip() for line in f if line.strip())
    if args.test_names:
        names.extend(args.test_names)
    if args.test_every:
        if args.test_every <= 0:
            raise ValueError("--test_every must be positive")
        names.extend(name for idx, name in enumerate(sorted(image_names)) if idx % args.test_every == 0)
    return sorted(dict.fromkeys(names))


def _write_test_split(sparse_target: str, image_names: list[str], args) -> dict | None:
    test_names = _test_names_from_args(args, image_names)
    if not test_names:
        return None
    valid = set(image_names)
    missing = [name for name in test_names if name not in valid]
    if missing:
        raise ValueError(f"Test split names not found in image_dir: {missing}")
    path = os.path.join(sparse_target, "test.txt")
    with open(path, "w", encoding="utf-8") as f:
        for name in test_names:
            f.write(f"{name}\n")
    return {
        "test_txt": os.path.abspath(path),
        "test_count": len(test_names),
        "test_names": test_names,
    }


def _camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    centers = []
    for T in extrinsics:
        R = T[:3, :3]
        t = T[:3, 3]
        centers.append(-R.T @ t)
    return np.asarray(centers, dtype=np.float64)


def _write_random_points_ply(recon: Reconstruction, output_path: str,
                             n_points: int, seed: int) -> dict:
    from plyfile import PlyData, PlyElement

    rng = np.random.default_rng(seed)
    cam_centers = _camera_centers(recon.extrinsics)
    if recon.num_points > 0:
        support = np.concatenate([recon.points3d, cam_centers], axis=0)
    else:
        support = cam_centers

    bbox_min = support.min(axis=0)
    bbox_max = support.max(axis=0)
    xyz = rng.uniform(bbox_min, bbox_max, size=(n_points, 3)).astype(np.float32)

    if recon.num_points > 0 and len(recon.points_rgb) > 0:
        sample_ids = rng.integers(0, len(recon.points_rgb), size=n_points)
        rgb = recon.points_rgb[sample_ids].astype(np.uint8)
    else:
        rgb = np.full((n_points, 3), 127, dtype=np.uint8)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    vertices = np.empty(n_points, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["nx"], vertices["ny"], vertices["nz"] = 0.0, 0.0, 0.0
    vertices["red"], vertices["green"], vertices["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    PlyData([PlyElement.describe(vertices, "vertex")]).write(output_path)
    return {
        "random_points_ply": os.path.abspath(output_path),
        "random_init_points": int(n_points),
        "random_seed": int(seed),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
    }


def _prepare_scene(args, output: str) -> str:
    prepared_scene = os.path.join(output, "prepared_scene")
    sparse_target = os.path.join(prepared_scene, "sparse", "0")
    os.makedirs(os.path.join(prepared_scene, "sparse"), exist_ok=True)
    image_prep_info = _prepare_images(args.image_dir, os.path.join(prepared_scene, "images"), args)
    image_names = sorted(
        name for name in os.listdir(args.image_dir)
        if os.path.isfile(os.path.join(args.image_dir, name))
    )

    random_init_info = None
    mask_filter_info = None
    if args.sparse_dir:
        if args.init_mode == "random":
            raise ValueError("--init_mode=random requires --reconstruction so the random bbox can be defined")
        if os.path.lexists(sparse_target):
            _replace_path(sparse_target)
        shutil.copytree(args.sparse_dir, sparse_target)
    else:
        recon = Reconstruction.from_npz(args.reconstruction)
        if args.mask_points:
            if not args.mask_dir:
                raise ValueError("--mask_points requires --mask_dir")
            recon = reconstruction_to_colmap_space(recon)
            recon, mask_filter_info = filter_reconstruction_by_masks(
                recon,
                mask_dir=args.mask_dir,
                threshold=args.mask_foreground_threshold,
                min_observations=args.mask_min_observations,
                min_ratio=args.mask_min_ratio,
            )
        os.makedirs(sparse_target, exist_ok=True)
        reconstruction_to_colmap_sparse(recon, sparse_target, args.camera_type)
        if args.init_mode == "random":
            random_init_info = _write_random_points_ply(
                recon,
                os.path.join(sparse_target, "points3D.ply"),
                args.random_init_points,
                args.random_seed,
            )

    test_split_info = _write_test_split(sparse_target, image_names, args)
    return prepared_scene, {
        "images": image_prep_info,
        "random_init": random_init_info,
        "mask_filter": mask_filter_info,
        "test_split": test_split_info,
    }


def _run(cmd: list[str], cwd: str, env: dict[str, str] | None = None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _official_env(repo: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    paths = [repo]
    if pythonpath:
        paths.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def main():
    args = parse_args()
    if not args.sparse_dir and not args.reconstruction:
        raise ValueError("Provide either --sparse_dir or --reconstruction")

    repo = os.path.abspath(args.repo)
    official_python = os.path.abspath(args.official_python)
    if not os.path.exists(os.path.join(repo, "train.py")):
        raise FileNotFoundError(f"Official gaussian-splatting train.py not found in {repo}")
    if not os.path.exists(official_python):
        raise FileNotFoundError(f"Official Python executable not found: {official_python}")

    output = prepare_output_dir(
        output_root=args.output,
        stage_name="gaussian_official",
        explicit_output_dir=args.output_run_dir,
        use_timestamp=args.use_timestamp,
    )

    config_path = save_run_metadata(
        output,
        stage="gaussian_official",
        params={**vars(args), "official_python": official_python},
        inputs={
            "repo": repo,
            "reconstruction": os.path.abspath(args.reconstruction) if args.reconstruction else None,
            "sparse_dir": os.path.abspath(args.sparse_dir) if args.sparse_dir else None,
            "image_dir": os.path.abspath(args.image_dir),
            "mask_dir": os.path.abspath(args.mask_dir) if args.mask_dir else None,
        },
        outputs={"output_dir": output},
    )
    print(f"Saved run config: {config_path}")

    prepared_scene, prepare_info = _prepare_scene(args, output)
    train_py = os.path.join(repo, "train.py")
    render_py = os.path.join(repo, "render.py")
    metrics_py = os.path.join(repo, "metrics.py")

    train_cmd = [
        official_python,
        train_py,
        "-s", prepared_scene,
        "-m", output,
        "--disable_viewer",
        "--eval",
        "--iterations", str(args.iterations),
        "--sh_degree", str(args.sh_degree),
        "-r", str(args.resolution),
        "--save_iterations", str(args.iterations),
        "--quiet",
    ]
    if args.white_background:
        train_cmd.append("-w")
    if args.checkpoint_iterations:
        train_cmd += ["--checkpoint_iterations", *[str(v) for v in args.checkpoint_iterations]]
    if args.test_iterations:
        train_cmd += ["--test_iterations", *[str(v) for v in args.test_iterations]]

    official_env = _official_env(repo)
    _run(train_cmd, cwd=repo, env=official_env)

    commands = {"train": train_cmd}
    if not args.skip_render_metrics:
        render_cmd = [
            official_python,
            render_py,
            "-m", output,
            "--skip_train",
            "--quiet",
        ]
        metrics_cmd = [
            official_python,
            metrics_py,
            "-m", output,
        ]
        _run(render_cmd, cwd=repo, env=official_env)
        _run(metrics_cmd, cwd=repo, env=official_env)
        commands["render"] = render_cmd
        commands["metrics"] = metrics_cmd

    save_json(os.path.join(output, "summary.json"), {
        "stage": "gaussian_official",
        "timestamp_utc": utc_timestamp(),
        "prepared_scene": prepared_scene,
        "commands": commands,
        "results_json": os.path.join(output, "results.json") if os.path.exists(os.path.join(output, "results.json")) else None,
        "per_view_json": os.path.join(output, "per_view.json") if os.path.exists(os.path.join(output, "per_view.json")) else None,
        "output_dir": output,
        **prepare_info,
    })
    print(f"Official 3DGS output: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
