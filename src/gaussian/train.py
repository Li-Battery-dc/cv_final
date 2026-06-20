#!/usr/bin/env python3
"""3D Gaussian Splatting training CLI entry point.

Usage:
    python -m src.gaussian.train \\
        --reconstruction outputs/ba_custom/reconstruction.npz \\
        --image_dir data/scene/images \\
        --output outputs/gs_custom_ba

    # With custom settings:
    python -m src.gaussian.train \\
        --reconstruction outputs/vggt_raw/reconstruction.npz \\
        --image_dir data/scene/images \\
        --output outputs/gs_raw \\
        --n_iterations 5000 \\
        --resolution 384 216
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.reconstruction import Reconstruction
from src.gaussian.model import GaussianModel
from src.gaussian.trainer import GaussianTrainer, CameraDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting")

    # Input
    parser.add_argument("--reconstruction", type=str, required=True,
                        help="Path to reconstruction.npz")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to original images")
    parser.add_argument("--output", type=str, default="outputs/gs_custom",
                        help="Output directory")

    # Training
    parser.add_argument("--n_iterations", type=int, default=10000,
                        help="Number of training iterations")
    parser.add_argument("--l1_weight", type=float, default=0.8,
                        help="Weight for L1 loss")
    parser.add_argument("--ssim_weight", type=float, default=0.2,
                        help="Weight for SSIM loss")
    parser.add_argument("--resolution", type=int, nargs=2, default=[768, 432],
                        metavar=('W', 'H'), help="Training resolution (width height)")

    # Model initialization
    parser.add_argument("--max_init_gaussians", type=int, default=100000,
                        help="Max Gaussians at initialization")
    parser.add_argument("--scale_factor", type=float, default=1.0,
                        help="Scale multiplier for initial covariance")

    # Densification
    parser.add_argument("--densify_from", type=int, default=500,
                        help="Start densification at iteration")
    parser.add_argument("--densify_until", type=int, default=6000,
                        help="Stop densification at iteration")
    parser.add_argument("--densify_interval", type=int, default=200,
                        help="Densification interval")
    parser.add_argument("--max_n_gaussians", type=int, default=300000,
                        help="Maximum total Gaussians")

    # SH scheduling
    parser.add_argument("--sh_degree", type=int, default=2,
                        help="Final SH degree")
    parser.add_argument("--sh_degree_interval", type=int, default=1000,
                        help="Interval for upgrading SH degree")

    # Logging
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Logging interval")
    parser.add_argument("--val_interval", type=int, default=500,
                        help="Validation interval")
    parser.add_argument("--checkpoint_interval", type=int, default=2000,
                        help="Checkpoint saving interval")
    parser.add_argument("--val_every", type=int, default=8,
                        help="Use every Nth image for validation")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device")

    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # LR
    parser.add_argument("--lr_xyz", type=float, default=1.6e-4)
    parser.add_argument("--lr_scales", type=float, default=5.0e-3)
    parser.add_argument("--lr_quat", type=float, default=1.0e-3)
    parser.add_argument("--lr_opacity", type=float, default=5.0e-2)
    parser.add_argument("--lr_sh", type=float, default=2.5e-3)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load reconstruction
    print(f"Loading reconstruction from: {args.reconstruction}")
    recon = Reconstruction.from_npz(args.reconstruction)
    print(f"Reconstruction: {recon}")

    if recon.num_points == 0:
        print("Error: No 3D points in reconstruction")
        return 1

    # Initialize Gaussian model
    print(f"Initializing Gaussian model (max {args.max_init_gaussians} Gaussians)...")
    model = GaussianModel.initialize_from_reconstruction(
        reconstruction=recon,
        max_n_gaussians=args.max_init_gaussians,
        scale_factor_multiplier=args.scale_factor,
        device=device,
    )
    print(f"Initialized {model.num_gaussians} Gaussians")

    # Set SH degree
    model.set_sh_degree(0)  # Start with degree 0

    # Build camera dataset
    print(f"Building camera dataset ({args.resolution[0]}x{args.resolution[1]})...")
    dataset = CameraDataset(
        reconstruction=recon,
        image_dir=args.image_dir,
        resolution=tuple(args.resolution),
        device=device,
    )
    print(f"Dataset: {len(dataset)} views")

    # Resume from checkpoint
    start_iteration = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        optimizer = model.create_optimizer()
        start_iteration = model.load_checkpoint(args.resume, optimizer)
        print(f"Resumed at iteration {start_iteration}")

    # Create trainer
    trainer = GaussianTrainer(
        model=model,
        dataset=dataset,
        output_dir=args.output,
        n_iterations=args.n_iterations,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        lr_dict={
            'xyz': args.lr_xyz,
            'log_scales': args.lr_scales,
            'quaternions': args.lr_quat,
            'log_opacities': args.lr_opacity,
            'features_dc': args.lr_sh,
            'features_rest': args.lr_sh / 20.0,
        },
        densify_from_iter=args.densify_from,
        densify_until_iter=args.densify_until,
        densify_interval=args.densify_interval,
        max_n_gaussians=args.max_n_gaussians,
        sh_degree_start=0,
        sh_degree_end=args.sh_degree,
        sh_degree_interval=args.sh_degree_interval,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        checkpoint_interval=args.checkpoint_interval,
        val_every=args.val_every,
        device=device,
    )

    # Update optimizer if resuming
    if args.resume and 'optimizer' in dir():
        trainer.optimizer = optimizer

    # Train
    try:
        metrics = trainer.train()
        print("\nTraining complete!")
        print(f"  Final PSNR: {metrics.get('psnr', 0):.2f} dB")
        print(f"  Final SSIM: {metrics.get('ssim', 0):.4f}")
        print(f"  Final Gaussians: {metrics.get('final_n_gaussians', 0)}")
        print(f"  Training time: {metrics.get('training_time_seconds', 0):.1f}s")
        print(f"  Output: {args.output}")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        model.save_checkpoint(
            os.path.join(args.output, "checkpoints", "interrupted.pt"),
            trainer.optimizer,
            0,
        )
        # Also export what we have
        model.export_ply(os.path.join(args.output, "interrupted.ply"))
        print(f"Saved interrupted checkpoint to {args.output}/checkpoints/interrupted.pt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
