"""Trainable 3D Gaussian Splatting model.

Parameterizes Gaussians with: xyz, log-scale, quaternion, opacity logit, SH colors.
Uses gsplat for CUDA rasterization.
"""

import numpy as np
import torch
import torch.nn as nn


def _quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (ijkr / scalar-last) to rotation matrices.

    Matches the VGGT convention: quat = [i, j, k, r].

    Args:
        quaternions: (N, 4) quaternions in i,j,k,r order.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j ** 2 + k ** 2),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i ** 2 + k ** 2),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i ** 2 + j ** 2),
        ),
        -1,
    )
    return o.reshape(-1, 3, 3)


def _quaternion_ijkr_to_wxyz(q_ijkr: torch.Tensor) -> torch.Tensor:
    """Convert from VGGT convention (ijkr, scalar last) to wxyz (scalar first)."""
    return torch.stack([
        q_ijkr[..., 3],  # r -> w
        q_ijkr[..., 0],  # i -> x
        q_ijkr[..., 1],  # j -> y
        q_ijkr[..., 2],  # k -> z
    ], dim=-1)


class GaussianModel(nn.Module):
    """Trainable 3D Gaussian model.

    Each Gaussian has:
      xyz:           (N, 3) center position.
      log_scales:    (N, 3) log of per-axis standard deviation.
      quaternions:   (N, 4) unit quaternion in ijkr format (scalar last).
      log_opacities: (N,)  logit of opacity.
      sh_coeffs:     (N, 3, (D+1)^2) spherical harmonics coefficients.

    Activations (applied during rasterization):
      scales = exp(log_scales)
      quat = normalize(quaternions)
      opacity = sigmoid(log_opacities)
      colors = SH evaluate(sh_coeffs, view_dirs)
    """

    def __init__(self, sh_degree: int = 0, max_sh_degree: int = 2):
        super().__init__()
        self._sh_degree = sh_degree
        self._max_sh_degree = max_sh_degree
        self.active_sh_degree = sh_degree

        # Parameters
        self._xyz = nn.Parameter(torch.empty(0, 3))
        self._log_scales = nn.Parameter(torch.empty(0, 3))
        self._quaternions = nn.Parameter(torch.empty(0, 4))
        self._log_opacities = nn.Parameter(torch.empty(0,))
        self._features_dc = nn.Parameter(torch.empty(0, 3, 1))   # degree 0 (RGB base)
        self._features_rest = nn.Parameter(torch.empty(0, 3, 0))  # degree 1+ coefficients

        # Non-trainable state
        self.register_buffer('_max_sh_degree_tensor', torch.tensor(max_sh_degree))
        self._grad_accum = None  # For densification
        self._denom_accum = 0

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    def get_scales(self) -> torch.Tensor:
        return torch.exp(self._log_scales)

    def get_quaternions(self, normalize: bool = True) -> torch.Tensor:
        if normalize:
            return nn.functional.normalize(self._quaternions, dim=-1)
        return self._quaternions

    def get_quaternions_wxyz(self) -> torch.Tensor:
        """Get quaternions in wxyz format for gsplat."""
        q = self.get_quaternions(normalize=True)
        return _quaternion_ijkr_to_wxyz(q)

    def get_opacities(self) -> torch.Tensor:
        return torch.sigmoid(self._log_opacities)

    def get_sh_coeffs(self) -> torch.Tensor:
        """Get all SH coefficients concatenated: (N, (D+1)^2, 3)."""
        if self._features_rest.numel() == 0:
            return self._features_dc.squeeze(-1).unsqueeze(-1)  # (N, 3, 1)
        # Concatenate dc and rest: (N, 3, 1) + (N, 3, R) -> (N, 3, D)
        return torch.cat([self._features_dc, self._features_rest], dim=-1)

    def get_colors_dc(self) -> torch.Tensor:
        """Get DC color component (RGB without view dependence)."""
        return self._features_dc.squeeze(-1)  # (N, 3)

    def set_sh_degree(self, degree: int):
        """Upgrade active SH degree (expands or truncates rest coefficients)."""
        self.active_sh_degree = min(degree, self._max_sh_degree)

        num_coeffs_needed = (self.active_sh_degree + 1) ** 2 - 1  # minus DC
        if num_coeffs_needed > self._features_rest.shape[-1]:
            # Need to expand
            pad_size = num_coeffs_needed - self._features_rest.shape[-1]
            padded = torch.nn.functional.pad(self._features_rest.data, (0, pad_size))
            self._features_rest = nn.Parameter(padded)
        elif num_coeffs_needed < self._features_rest.shape[-1]:
            # Need to truncate
            self._features_rest = nn.Parameter(self._features_rest.data[..., :num_coeffs_needed])

    def create_optimizer(self, lr_dict: dict = None) -> torch.optim.Adam:
        """Create Adam optimizer with per-parameter learning rates.

        Default learning rates from 3DGS paper:
          xyz: 1.6e-4, log_scales: 5e-3, quaternions: 1e-3,
          log_opacities: 5e-2, sh: 2.5e-3
        """
        if lr_dict is None:
            lr_dict = {
                'xyz': 1.6e-4,
                'log_scales': 5.0e-3,
                'quaternions': 1.0e-3,
                'log_opacities': 5.0e-2,
                'features_dc': 2.5e-3,
                'features_rest': 2.5e-3 / 20.0,
            }

        param_groups = [
            {'params': [self._xyz], 'lr': lr_dict['xyz'], 'name': 'xyz'},
            {'params': [self._log_scales], 'lr': lr_dict['log_scales'], 'name': 'log_scales'},
            {'params': [self._quaternions], 'lr': lr_dict['quaternions'], 'name': 'quaternions'},
            {'params': [self._log_opacities], 'lr': lr_dict['log_opacities'], 'name': 'log_opacities'},
            {'params': [self._features_dc], 'lr': lr_dict['features_dc'], 'name': 'features_dc'},
        ]
        if self._features_rest.numel() > 0:
            param_groups.append(
                {'params': [self._features_rest], 'lr': lr_dict['features_rest'], 'name': 'features_rest'}
            )

        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    # ---- Initialization ----

    @classmethod
    def initialize_from_reconstruction(cls, reconstruction,
                                        max_n_gaussians: int = 100_000,
                                        scale_factor_multiplier: float = 1.0,
                                        device: str = "cuda") -> "GaussianModel":
        """Initialize Gaussians from a Reconstruction point cloud.

        - xyz: from points3d.
        - scales: from k * mean_distance_to_nearest_neighbors.
        - quaternions: identity (0, 0, 0, 1) in ijkr.
        - opacity: initialized to 0.1.
        - SH dc: from RGB colors.

        Args:
            reconstruction: Reconstruction with points3d and points_rgb.
            max_n_gaussians: Max number of Gaussians to initialize.
            scale_factor_multiplier: Scale factor for initial covariance.
            device: Target device.

        Returns:
            GaussianModel instance.
        """
        points3d = reconstruction.points3d
        points_rgb = reconstruction.points_rgb

        # Subsample if needed
        P = len(points3d)
        if P > max_n_gaussians:
            idx = np.random.choice(P, max_n_gaussians, replace=False)
            points3d = points3d[idx]
            points_rgb = points_rgb[idx]
            P = max_n_gaussians

        # Compute initial scales from nearest neighbors
        scales = _compute_neighbor_scales(points3d, k=3) * scale_factor_multiplier

        model = cls(sh_degree=0, max_sh_degree=2)

        # Convert to torch tensors
        model._xyz = nn.Parameter(torch.tensor(points3d, dtype=torch.float32, device=device))
        model._log_scales = nn.Parameter(torch.log(torch.tensor(scales, dtype=torch.float32, device=device) + 1e-8))
        # Identity quaternion: (i, j, k, r) = (0, 0, 0, 1)
        quats = torch.zeros(P, 4, dtype=torch.float32, device=device)
        quats[:, 3] = 1.0
        model._quaternions = nn.Parameter(quats)
        # Opacity initialized to 0.1 -> logit
        init_opacity = 0.1
        logit_opacity = np.log(init_opacity / (1.0 - init_opacity))
        model._log_opacities = nn.Parameter(
            torch.full((P,), logit_opacity, dtype=torch.float32, device=device)
        )
        # SH DC: normalize RGB to [0, 1] and convert to SH coefficients
        rgb = torch.tensor(points_rgb, dtype=torch.float32, device=device) / 255.0
        # SH degree 0: constant * I. For RGB, the DC coefficient = color / sqrt(4*pi)
        # But gsplat expects raw RGB as the DC term
        model._features_dc = nn.Parameter(rgb.unsqueeze(-1))  # (P, 3, 1)

        # No rest features initially
        model._features_rest = nn.Parameter(
            torch.zeros(P, 3, 0, dtype=torch.float32, device=device)
        )

        return model

    # ---- Densification ----

    def densify_and_prune(self, grad_threshold: float = 0.0002,
                          clone_scale: float = 1.0, split_scale: float = 1.6,
                          max_n_gaussians: int = 300_000,
                          prune_opacity_threshold: float = 0.005,
                          prune_scale_threshold: float = 0.01,
                          max_screen_radius: float = 100.0,
                          min_opacity_reset: float = 0.01):
        """Perform densification (clone/split) and pruning.

        Args:
            grad_threshold: Position gradient threshold for densification.
            clone_scale: Max scale for cloning (small Gaussians).
            split_scale: Min scale for splitting (large Gaussians).
            max_n_gaussians: Maximum total Gaussians.
            prune_opacity_threshold: Remove Gaussians with opacity below this.
            prune_scale_threshold: Remove Gaussians with scales too large.
            max_screen_radius: Prune Gaussians too large on screen.
        """
        if self._grad_accum is None:
            return

        grad_norm = self._grad_accum / max(self._denom_accum, 1)
        scales = self.get_scales()
        opacities = self.get_opacities()

        # Prune low-opacity Gaussians
        prune_mask = opacities < prune_opacity_threshold

        # Prune extremely large scales
        max_scale = scales.max(dim=-1)[0]
        prune_mask |= max_scale > prune_scale_threshold * 50.0

        # Detect high-gradient Gaussians for densification
        high_grad = grad_norm > grad_threshold

        # Select for clone (small) or split (large)
        max_scale_all = scales.max(dim=-1)[0]
        clone_mask = high_grad & (max_scale_all <= clone_scale)
        split_mask = high_grad & (max_scale_all >= split_scale)

        # Apply pruning
        if prune_mask.any():
            self._prune_points(prune_mask)

        # Recompute masks after pruning
        grad_norm = grad_norm[~prune_mask] if prune_mask.sum() > 0 else grad_norm
        clone_mask = clone_mask[~prune_mask] if prune_mask.sum() > 0 else clone_mask
        split_mask = split_mask[~prune_mask] if prune_mask.sum() > 0 else split_mask

        # Clone
        if clone_mask.any() and self.num_gaussians < max_n_gaussians:
            self._clone_points(clone_mask)

        # Split
        if split_mask.any() and self.num_gaussians < max_n_gaussians:
            self._split_points(split_mask)

        # Reset opacity for near-transparent Gaussians (encourage them to adapt)
        reset_mask = opacities < min_opacity_reset
        if reset_mask.any():
            init_opacity = 0.01
            logit = np.log(init_opacity / (1.0 - init_opacity))
            with torch.no_grad():
                self._log_opacities[reset_mask] = logit

        # Reset accumulators
        self._grad_accum = None
        self._denom_accum = 0

    def accumulate_gradients(self):
        """Accumulate xyz gradients for densification decision."""
        if self._xyz.grad is not None:
            if self._grad_accum is None:
                self._grad_accum = torch.zeros_like(self._xyz)
            self._grad_accum += self._xyz.grad.detach().norm(dim=-1)
            self._denom_accum += 1

    def _prune_points(self, mask: torch.Tensor):
        """Remove Gaussians where mask is True."""
        valid = ~mask
        indices = valid.nonzero(as_tuple=True)[0]

        self._xyz = nn.Parameter(self._xyz.data[valid])
        self._log_scales = nn.Parameter(self._log_scales.data[valid])
        self._quaternions = nn.Parameter(self._quaternions.data[valid])
        self._log_opacities = nn.Parameter(self._log_opacities.data[valid])
        self._features_dc = nn.Parameter(self._features_dc.data[valid])
        if self._features_rest.numel() > 0:
            self._features_rest = nn.Parameter(self._features_rest.data[valid])

    def _clone_points(self, mask: torch.Tensor):
        """Duplicate Gaussians with small scale."""
        clone_indices = mask.nonzero(as_tuple=True)[0]

        new_xyz = self._xyz.data[clone_indices]
        new_log_scales = self._log_scales.data[clone_indices]
        new_quats = self._quaternions.data[clone_indices]
        new_opacities = self._log_opacities.data[clone_indices]
        new_dc = self._features_dc.data[clone_indices]

        self._xyz = nn.Parameter(torch.cat([self._xyz.data, new_xyz]))
        self._log_scales = nn.Parameter(torch.cat([self._log_scales.data, new_log_scales]))
        self._quaternions = nn.Parameter(torch.cat([self._quaternions.data, new_quats]))
        self._log_opacities = nn.Parameter(torch.cat([self._log_opacities.data, new_opacities]))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc.data, new_dc]))
        if self._features_rest.numel() > 0:
            new_rest = self._features_rest.data[clone_indices]
            self._features_rest = nn.Parameter(torch.cat([self._features_rest.data, new_rest]))

    def _split_points(self, mask: torch.Tensor):
        """Split large Gaussians into two smaller ones."""
        split_indices = mask.nonzero(as_tuple=True)[0]
        n_splits = len(split_indices)

        if n_splits == 0:
            return

        scales = self.get_scales()[split_indices]
        quats = self.get_quaternions(normalize=True)[split_indices]
        xyz = self._xyz.data[split_indices]

        # Sample new positions from the Gaussian distribution
        # Perturb along principal axes
        R = _quaternion_to_matrix(quats)  # (n_splits, 3, 3)
        samples = torch.randn(n_splits, 3, device=xyz.device) * scales * 0.8

        # Transform sample to world space: R @ (sample) + xyz
        samples_world = torch.bmm(R, samples.unsqueeze(-1)).squeeze(-1) + xyz

        # New parameters (half the scale)
        new_log_scales = self._log_scales.data[split_indices] - np.log(1.6)
        new_opacities = self._log_opacities.data[split_indices]
        new_quats = self._quaternions.data[split_indices]
        new_dc = self._features_dc.data[split_indices]

        # Concatenate original + new
        self._xyz = nn.Parameter(torch.cat([self._xyz.data, samples_world]))
        self._log_scales = nn.Parameter(torch.cat([self._log_scales.data, new_log_scales]))
        self._quaternions = nn.Parameter(torch.cat([self._quaternions.data, new_quats]))
        self._log_opacities = nn.Parameter(torch.cat([self._log_opacities.data, new_opacities]))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc.data, new_dc]))
        if self._features_rest.numel() > 0:
            new_rest = self._features_rest.data[split_indices]
            self._features_rest = nn.Parameter(torch.cat([self._features_rest.data, new_rest]))

    # ---- State management ----

    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None,
                       iteration: int = 0):
        """Save model checkpoint."""
        state = {
            'iteration': iteration,
            'active_sh_degree': self.active_sh_degree,
            '_xyz': self._xyz.data,
            '_log_scales': self._log_scales.data,
            '_quaternions': self._quaternions.data,
            '_log_opacities': self._log_opacities.data,
            '_features_dc': self._features_dc.data,
            '_features_rest': self._features_rest.data,
            '_max_sh_degree': self._max_sh_degree,
        }
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None) -> int:
        """Load model checkpoint. Returns iteration number."""
        state = torch.load(path, map_location=self._xyz.device, weights_only=False)
        self.active_sh_degree = state['active_sh_degree']
        self._xyz = nn.Parameter(state['_xyz'])
        self._log_scales = nn.Parameter(state['_log_scales'])
        self._quaternions = nn.Parameter(state['_quaternions'])
        self._log_opacities = nn.Parameter(state['_log_opacities'])
        self._features_dc = nn.Parameter(state['_features_dc'])
        self._features_rest = nn.Parameter(state['_features_rest'])
        self._max_sh_degree = state.get('_max_sh_degree', 2)
        if optimizer is not None and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        return state.get('iteration', 0)

    def export_ply(self, path: str):
        """Export Gaussians as a PLY file for visualization."""
        from plyfile import PlyData, PlyElement

        xyz = self._xyz.data.cpu().numpy()
        scales = self.get_scales().cpu().detach().numpy()
        quats = self.get_quaternions(normalize=True).cpu().detach().numpy()
        opacities = self.get_opacities().cpu().detach().numpy()
        colors = torch.clamp(self.get_colors_dc(), 0.0, 1.0).cpu().detach().numpy()

        n = self.num_gaussians
        vertices = np.zeros(n, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('opacity', 'f4'),
        ])

        vertices['x'] = xyz[:, 0]
        vertices['y'] = xyz[:, 1]
        vertices['z'] = xyz[:, 2]
        vertices['f_dc_0'] = colors[:, 0]
        vertices['f_dc_1'] = colors[:, 1]
        vertices['f_dc_2'] = colors[:, 2]
        vertices['scale_0'] = np.log(np.maximum(scales[:, 0], 1e-8))
        vertices['scale_1'] = np.log(np.maximum(scales[:, 1], 1e-8))
        vertices['scale_2'] = np.log(np.maximum(scales[:, 2], 1e-8))
        # Convert ijkr to wxyz for PLY
        vertices['rot_0'] = quats[:, 3]  # r -> w
        vertices['rot_1'] = quats[:, 0]  # i -> x
        vertices['rot_2'] = quats[:, 1]  # j -> y
        vertices['rot_3'] = quats[:, 2]  # k -> z
        vertices['opacity'] = opacities

        PlyData([PlyElement.describe(vertices, 'vertex')]).write(path)


def _compute_neighbor_scales(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Compute initial scales from mean distance to k nearest neighbors.

    Args:
        points: (N, 3) array of 3D points.
        k: Number of nearest neighbors.

    Returns:
        scales: (N, 3) array of per-axis scales.
    """
    from scipy.spatial import KDTree

    if len(points) < 2:
        return np.ones_like(points) * 0.1

    k = min(k, len(points) - 1)
    tree = KDTree(points)
    distances, _ = tree.query(points, k=k + 1)  # +1 because first is self

    # Average distance to k nearest neighbors (excluding self)
    if k == 1:
        avg_dist = distances[:, 1]
    else:
        avg_dist = distances[:, 1:].mean(axis=1)

    # Ensure minimum scale
    avg_dist = np.maximum(avg_dist, 1e-6)

    # Initialize as isotropic scale
    scales = np.stack([avg_dist, avg_dist, avg_dist], axis=1)
    return scales
