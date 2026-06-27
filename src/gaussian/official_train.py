#!/usr/bin/env python3
"""Wrapper around the official graphdeco-inria/gaussian-splatting training code."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.colmap_io import reconstruction_to_colmap_sparse
from src.data.reconstruction import Reconstruction
from src.utils.experiment import (
    prepare_output_dir,
    save_json,
    save_run_metadata,
    update_latest_symlink,
    utc_timestamp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Official Gaussian Splatting baseline runner")
    parser.add_argument("--repo", type=str, default="packages/gaussian-splatting",
                        help="Path to the official gaussian-splatting repository")
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
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--resolution", type=int, default=768,
                        help="Official code uses width or scale factor here")
    parser.add_argument("--sh_degree", type=int, default=2)
    parser.add_argument("--checkpoint_iterations", nargs="*", type=int, default=None)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=None)
    parser.add_argument("--skip_render_metrics", action="store_true", default=False,
                        help="Skip official render.py / metrics.py after training")
    parser.add_argument("--white_background", action="store_true", default=False)
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


def _prepare_scene(args) -> str:
    prepared_scene = os.path.join(args.output, "prepared_scene")
    sparse_target = os.path.join(prepared_scene, "sparse", "0")
    os.makedirs(os.path.join(prepared_scene, "sparse"), exist_ok=True)
    _link_or_copy(args.image_dir, os.path.join(prepared_scene, "images"))

    if args.sparse_dir:
        _link_or_copy(args.sparse_dir, sparse_target)
    else:
        recon = Reconstruction.from_npz(args.reconstruction)
        os.makedirs(sparse_target, exist_ok=True)
        reconstruction_to_colmap_sparse(recon, sparse_target, args.camera_type)

    return prepared_scene


def _run(cmd: list[str], cwd: str):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    args = parse_args()
    if not args.sparse_dir and not args.reconstruction:
        raise ValueError("Provide either --sparse_dir or --reconstruction")

    repo = os.path.abspath(args.repo)
    output = prepare_output_dir(
        output_root=args.output,
        stage_name="gaussian_official",
        explicit_output_dir=args.output_run_dir,
        use_timestamp=args.use_timestamp,
    )

    config_path = save_run_metadata(
        output,
        stage="gaussian_official",
        params={
            "camera_type": args.camera_type,
            "iterations": args.iterations,
            "resolution": args.resolution,
            "sh_degree": args.sh_degree,
            "skip_render_metrics": args.skip_render_metrics,
            "white_background": args.white_background,
        },
        inputs={
            "repo": repo,
            "reconstruction": os.path.abspath(args.reconstruction) if args.reconstruction else None,
            "sparse_dir": os.path.abspath(args.sparse_dir) if args.sparse_dir else None,
            "image_dir": os.path.abspath(args.image_dir),
            "output_dir": output,
        },
    )
    print(f"Saved run config: {config_path}")

    prepared_scene = _prepare_scene(args)
    train_py = os.path.join(repo, "train.py")
    render_py = os.path.join(repo, "render.py")
    metrics_py = os.path.join(repo, "metrics.py")

    train_cmd = [
        sys.executable,
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

    _run(train_cmd, cwd=repo)

    commands = {"train": train_cmd}
    if not args.skip_render_metrics:
        render_cmd = [
            sys.executable,
            render_py,
            "-m", output,
            "--skip_train",
            "--quiet",
        ]
        metrics_cmd = [
            sys.executable,
            metrics_py,
            "-m", output,
        ]
        _run(render_cmd, cwd=repo)
        _run(metrics_cmd, cwd=repo)
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
    })
    update_latest_symlink(args.output, output)
    print(f"Official 3DGS output: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
