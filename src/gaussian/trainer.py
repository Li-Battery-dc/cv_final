"""Training loop for 3D Gaussian Splatting.

Implements: L1+SSIM loss, Adam optimizer, densification/pruning schedule,
periodic validation and checkpointing.
"""

import os
import json
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.reconstruction import Reconstruction
from src.gaussian.model import GaussianModel
from src.gaussian.renderer import render_view


class CameraDataset(torch.utils.data.Dataset):
    """Dataset of camera views from a Reconstruction."""

    def __init__(self, reconstruction: Reconstruction, image_dir: str,
                 resolution: tuple = (768, 432), device: str = "cuda"):
        """
        Args:
            reconstruction: Reconstruction with camera data.
            image_dir: Path to original images.
            resolution: (width, height) for rendering.
            device: Target device.
        """
        self.recon = reconstruction
        self.image_dir = image_dir
        self.width, self.height = resolution
        self.device = device
        self.views = []

        for s in range(reconstruction.num_images):
            self.views.append(self._build_view(s))

    def _build_view(self, idx: int) -> dict:
        """Build a camera view from reconstruction data."""
        recon = self.recon

        # Load image
        img_name = str(recon.image_names[idx])
        img_path = os.path.join(self.image_dir, img_name)

        try:
            pil_img = Image.open(img_path).convert('RGB')
            # Resize to target resolution
            pil_img = pil_img.resize((self.width, self.height), Image.BILINEAR)
            img_tensor = torch.tensor(np.array(pil_img), dtype=torch.float32, device=self.device) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
        except (IOError, OSError) as e:
            print(f"Warning: Could not load {img_path}: {e}")
            img_tensor = torch.zeros(3, self.height, self.width, device=self.device)

        # Build world-to-camera matrix
        T = recon.extrinsics[idx]  # (3, 4) camera-from-world
        R = torch.tensor(T[:3, :3], dtype=torch.float32, device=self.device)
        t = torch.tensor(T[:3, 3], dtype=torch.float32, device=self.device)

        w2c = torch.eye(4, device=self.device)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        # Camera center in world: -R^T @ t
        cam_center = -R.T @ t

        # Build intrinsics for target resolution
        K_orig = recon.intrinsics[idx].copy()
        # Scale K to target resolution
        scale_w = self.width / float(recon.image_size_hw[idx, 1])
        scale_h = self.height / float(recon.image_size_hw[idx, 0])
        K_orig[0, 0] *= scale_w
        K_orig[0, 2] *= scale_w
        K_orig[1, 1] *= scale_h
        K_orig[1, 2] *= scale_h

        K = torch.tensor(K_orig, dtype=torch.float32, device=self.device)

        return {
            'image': img_tensor,
            'world_view_transform': w2c,
            'camera_center': cam_center,
            'K': K,
            'width': self.width,
            'height': self.height,
            'image_name': img_name,
        }

    def __len__(self):
        return len(self.views)

    def __getitem__(self, idx):
        return self.views[idx]

    def get_train_val_split(self, val_every: int = 8):
        """Split into train/val (every Nth image = validation)."""
        train_ids = [i for i in range(len(self)) if i % val_every != 0]
        val_ids = [i for i in range(len(self)) if i % val_every == 0]
        return train_ids, val_ids


def compute_psnr(img: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute PSNR between two (3, H, W) tensors in [0, 1]."""
    mse = F.mse_loss(img, gt)
    if mse < 1e-10:
        return 100.0
    return float(20.0 * math.log10(1.0) - 10.0 * math.log10(float(mse)))


def compute_ssim(img: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute SSIM between two (3, H, W) tensors."""
    try:
        from torchmetrics import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(img.device)
        return ssim(img.unsqueeze(0), gt.unsqueeze(0))
    except ImportError:
        # Simple fallback using torch
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        # Convert to grayscale-like single channel
        img_gray = img.mean(dim=0, keepdim=True)
        gt_gray = gt.mean(dim=0, keepdim=True)
        mu1 = F.avg_pool2d(img_gray.unsqueeze(0), 11, 1, 5)
        mu2 = F.avg_pool2d(gt_gray.unsqueeze(0), 11, 1, 5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(img_gray.unsqueeze(0) ** 2, 11, 1, 5) - mu1_sq
        sigma2_sq = F.avg_pool2d(gt_gray.unsqueeze(0) ** 2, 11, 1, 5) - mu2_sq
        sigma12 = F.avg_pool2d(img_gray.unsqueeze(0) * gt_gray.unsqueeze(0), 11, 1, 5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class GaussianTrainer:
    """Training orchestrator for 3D Gaussian Splatting."""

    def __init__(
        self,
        model: GaussianModel,
        dataset: CameraDataset,
        output_dir: str = "outputs/gs_custom",
        # Training
        n_iterations: int = 10000,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        lr_dict: dict = None,
        # Densification
        densify_from_iter: int = 500,
        densify_until_iter: int = 6000,
        densify_interval: int = 200,
        densify_grad_threshold: float = 0.0002,
        max_n_gaussians: int = 300_000,
        # SH scheduling
        sh_degree_start: int = 0,
        sh_degree_end: int = 2,
        sh_degree_interval: int = 1000,
        # Logging
        log_interval: int = 100,
        val_interval: int = 500,
        checkpoint_interval: int = 2000,
        # Validation
        val_every: int = 8,
        device: str = "cuda",
    ):
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        self.n_iterations = n_iterations
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densify_interval = densify_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.max_n_gaussians = max_n_gaussians
        self.sh_degree_start = sh_degree_start
        self.sh_degree_end = sh_degree_end
        self.sh_degree_interval = sh_degree_interval
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.checkpoint_interval = checkpoint_interval
        self.val_every = val_every
        self.device = device

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)

        # Train/val split
        self.train_ids, self.val_ids = dataset.get_train_val_split(val_every)

        # Optimizer
        self.optimizer = model.create_optimizer(lr_dict)

        # Metrics history
        self.history = {
            'train_loss': [], 'train_psnr': [],
            'val_psnr': [], 'val_ssim': [],
            'n_gaussians': [], 'iteration': [],
        }

    def train(self) -> dict:
        """Main training loop."""
        model = self.model
        dataset = self.dataset
        device = self.device

        bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)

        print(f"Starting training: {self.n_iterations} iterations")
        print(f"  Training views: {len(self.train_ids)}")
        print(f"  Validation views: {len(self.val_ids)}")
        print(f"  Initial Gaussians: {model.num_gaussians}")

        t_start = time.time()
        for iteration in range(1, self.n_iterations + 1):
            # ---- Sample training view ----
            cam_idx = self.train_ids[np.random.randint(0, len(self.train_ids))]
            view = dataset[cam_idx]

            # ---- Render ----
            render_result = render_view(
                model=model,
                world_view_transform=view['world_view_transform'],
                full_proj_transform=view['world_view_transform'],
                camera_center=view['camera_center'],
                K=view['K'],
                width=view['width'],
                height=view['height'],
                bg_color=bg_color,
            )

            rendered = render_result['render']  # (3, H, W)
            gt = view['image']                    # (3, H, W)

            # ---- Loss ----
            l1_loss = F.l1_loss(rendered, gt)
            ssim_loss = 1.0 - compute_ssim(rendered, gt)
            total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss

            # ---- Backward ----
            total_loss.backward()

            # Accumulate gradients for densification
            if self.densify_from_iter <= iteration <= self.densify_until_iter:
                model.accumulate_gradients()

            # ---- Optimizer step ----
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # ---- Densification ----
            if (self.densify_from_iter <= iteration <= self.densify_until_iter
                    and iteration % self.densify_interval == 0):
                model.densify_and_prune(
                    grad_threshold=self.densify_grad_threshold,
                    max_n_gaussians=self.max_n_gaussians,
                )
                # Recreate optimizer for new parameter sizes
                self.optimizer = model.create_optimizer()

            # ---- SH degree scheduling ----
            if iteration % self.sh_degree_interval == 0 and model.active_sh_degree < self.sh_degree_end:
                new_degree = model.active_sh_degree + 1
                model.set_sh_degree(new_degree)
                print(f"Iter {iteration}: SH degree upgraded to {new_degree}")

            # ---- Logging ----
            if iteration % self.log_interval == 0:
                psnr = compute_psnr(rendered, gt)
                self.history['train_loss'].append(float(total_loss))
                self.history['train_psnr'].append(psnr)
                self.history['n_gaussians'].append(model.num_gaussians)
                self.history['iteration'].append(iteration)

                print(f"Iter {iteration:5d}: loss={total_loss:.4f} "
                      f"(L1={l1_loss:.4f}, SSIM={ssim_loss:.4f}) "
                      f"PSNR={psnr:.2f} dB, N={model.num_gaussians}")

            # ---- Validation ----
            if iteration % self.val_interval == 0:
                val_metrics = self._validate(bg_color)
                self.history['val_psnr'].append(val_metrics['psnr'])
                self.history['val_ssim'].append(val_metrics['ssim'])
                print(f"  VAL: PSNR={val_metrics['psnr']:.2f} dB, "
                      f"SSIM={val_metrics['ssim']:.4f}, N={model.num_gaussians}")

            # ---- Checkpoint ----
            if iteration % self.checkpoint_interval == 0 or iteration == self.n_iterations:
                ckpt_path = os.path.join(self.output_dir, "checkpoints",
                                         f"iter_{iteration:05d}.pt")
                model.save_checkpoint(ckpt_path, self.optimizer, iteration)
                # Save latest
                model.save_checkpoint(
                    os.path.join(self.output_dir, "checkpoints", "latest.pt"),
                    self.optimizer, iteration,
                )

        t_total = time.time() - t_start
        print(f"Training complete in {t_total:.1f}s ({t_total / self.n_iterations:.3f}s/iter)")

        # Final validation
        final_metrics = self._validate(bg_color)
        final_metrics['training_time_seconds'] = t_total
        final_metrics['final_n_gaussians'] = model.num_gaussians

        # Save final model and metrics
        model.export_ply(os.path.join(self.output_dir, "final.ply"))
        with open(os.path.join(self.output_dir, "metrics.json"), 'w') as f:
            json.dump({**self.history, 'final': final_metrics}, f, indent=2, default=float)

        # Save validation renders
        self._save_validation_renders(bg_color)

        return final_metrics

    def _validate(self, bg_color: torch.Tensor) -> dict:
        """Compute metrics on validation views."""
        model = self.model
        model.eval()
        psnrs = []
        ssims = []

        with torch.no_grad():
            for val_idx in self.val_ids[:4]:  # Limit to 4 for speed
                view = self.dataset[val_idx]
                render_result = render_view(
                    model=model,
                    world_view_transform=view['world_view_transform'],
                    full_proj_transform=view['world_view_transform'],
                    camera_center=view['camera_center'],
                    K=view['K'],
                    width=view['width'],
                    height=view['height'],
                    bg_color=bg_color,
                )
                rendered = render_result['render']
                gt = view['image']
                psnrs.append(compute_psnr(rendered, gt))
                ssims.append(float(compute_ssim(rendered, gt)))

        model.train()
        return {
            'psnr': float(np.mean(psnrs)) if psnrs else 0.0,
            'ssim': float(np.mean(ssims)) if ssims else 0.0,
        }

    def _save_validation_renders(self, bg_color: torch.Tensor):
        """Save validation view renders as images."""
        model = self.model
        val_dir = os.path.join(self.output_dir, "validation")
        model.eval()

        with torch.no_grad():
            for val_idx in self.val_ids:
                view = self.dataset[val_idx]
                render_result = render_view(
                    model=model,
                    world_view_transform=view['world_view_transform'],
                    full_proj_transform=view['world_view_transform'],
                    camera_center=view['camera_center'],
                    K=view['K'],
                    width=view['width'],
                    height=view['height'],
                    bg_color=bg_color,
                )
                rendered = render_result['render']

                # Save rendered image
                rendered_np = (rendered.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                rendered_np = rendered_np.transpose(1, 2, 0)
                img_name = view['image_name'].replace('.jpg', '_render.png').replace('.jpeg', '_render.png')
                Image.fromarray(rendered_np).save(os.path.join(val_dir, img_name))

                # Save ground truth
                gt = view['image']
                gt_np = (gt.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                gt_np = gt_np.transpose(1, 2, 0)
                gt_name = view['image_name'].replace('.jpg', '_gt.png').replace('.jpeg', '_gt.png')
                Image.fromarray(gt_np).save(os.path.join(val_dir, gt_name))

        model.train()
