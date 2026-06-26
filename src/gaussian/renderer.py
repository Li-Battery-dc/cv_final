"""gsplat CUDA rasterizer wrapper for 3D Gaussian Splatting.

Provides thin wrappers around gsplat's rasterization functions,
adapting our GaussianModel parameterization to gsplat's expected formats.
"""

import torch
import torch.nn.functional as F

from src.gaussian.model import GaussianModel, _quaternion_ijkr_to_wxyz


def render_view(
    model: GaussianModel,
    world_view_transform: torch.Tensor,  # (4, 4) world-to-camera
    full_proj_transform: torch.Tensor,    # (4, 4) projection
    camera_center: torch.Tensor,          # (3,) camera position in world
    K: torch.Tensor,                      # (3, 3) intrinsics
    width: int,
    height: int,
    bg_color: torch.Tensor = None,        # (3,) or scalar
    sh_degree: int = None,
) -> dict:
    """Render a single view using gsplat rasterizer.

    Args:
        model: GaussianModel instance.
        world_view_transform: (4, 4) world-to-camera matrix.
        full_proj_transform: (4, 4) projection matrix.
        camera_center: (3,) camera center in world space.
        K: (3, 3) intrinsics matrix.
        width, height: Render resolution.
        bg_color: Background color (3,) or scalar.
        sh_degree: SH degree to use (defaults to model.active_sh_degree).

    Returns:
        dict with 'render' (3, H, W), 'alpha' (H, W), 'depth' (H, W) and 'info'.
    """
    try:
        from gsplat import rasterization
    except ImportError as e:
        raise RuntimeError("gsplat is required for Gaussian training/rendering") from e

    device = model._xyz.device

    if bg_color is None:
        bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)
    elif bg_color.ndim == 0 or bg_color.numel() == 1:
        bg_color = torch.full((3,), float(bg_color), device=device)

    # Prepare inputs for gsplat
    means = model.get_xyz()                       # (N, 3)
    quats = model.get_quaternions_wxyz()           # (N, 4) wxyz
    scales = model.get_scales()                    # (N, 3)
    opacities = model.get_opacities()              # (N,)
    colors = model.get_colors_dc()                  # (N, 3)

    if sh_degree is None:
        sh_degree = model.active_sh_degree

    # gsplat expects RGB colors as (N, 3) when sh_degree=None, or SH coeffs
    # as (N, K, 3) when sh_degree is active.
    if sh_degree > 0 and model._features_rest.numel() > 0:
        render_colors = model.get_sh_coeffs().permute(0, 2, 1).contiguous()  # (N, K, 3)
        render_sh_degree = sh_degree
    else:
        render_colors = colors  # (N, 3)
        render_sh_degree = None

    # Build viewmats and Ks
    viewmats = world_view_transform.unsqueeze(0)  # (1, 4, 4)
    Ks = K.unsqueeze(0)                            # (1, 3, 3)

    # Call gsplat rasterizer
    # gsplat API: rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, ...)
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=render_colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=render_sh_degree,
        backgrounds=bg_color.unsqueeze(0),  # (1, 3)
        render_mode="RGB",
        packed=False,
        absgrad=False,
    )

    render = render_colors.squeeze(0).permute(2, 0, 1)   # (3, H, W)
    alpha = render_alphas.squeeze(0).squeeze(-1)          # (H, W)

    # Depth approximation from median intersection
    depth = info.get('median_depth', torch.zeros(height, width, device=device))
    if depth.ndim == 3:
        depth = depth.squeeze(0).squeeze(-1)

    return {
        'render': render,
        'alpha': alpha,
        'depth': depth,
        'info': info,
    }


def render_view_simple(
    model: GaussianModel,
    R: torch.Tensor,           # (3, 3) rotation (world-to-camera)
    t: torch.Tensor,           # (3,) translation
    K: torch.Tensor,           # (3, 3) intrinsics
    width: int,
    height: int,
    bg_color: torch.Tensor = None,
) -> dict:
    """Simplified render: build transforms from R, t, K.

    This follows OpenCV convention:
      P_cam = R @ P_world + t  (camera-from-world)

    Args:
        model: GaussianModel.
        R: (3, 3) rotation matrix (world-to-camera).
        t: (3,) translation vector.
        K: (3, 3) intrinsics.
        width, height: Image size.
        bg_color: Background color.

    Returns:
        dict with 'render', 'alpha', etc.
    """
    device = R.device
    # Build world-to-camera 4x4 matrix
    T_w2c = torch.eye(4, device=device)
    T_w2c[:3, :3] = R
    T_w2c[:3, 3] = t

    # Camera center in world: -R^T @ t
    camera_center = -R.T @ t

    # Build full projection: [2*fx/W, 0, (cx - W/2)/(W/2), 0; ...] for OpenGL NDC
    # Simplified: we'll use K directly and let gsplat handle projection
    # gsplat's viewmats is world-to-camera, Ks is intrinsics
    return render_view(
        model=model,
        world_view_transform=T_w2c,
        full_proj_transform=T_w2c,  # placeholder
        camera_center=camera_center,
        K=K,
        width=width,
        height=height,
        bg_color=bg_color,
    )
