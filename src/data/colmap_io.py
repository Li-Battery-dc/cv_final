"""COLMAP <-> unified Reconstruction format round-trip conversion."""

import os
import numpy as np

from src.data.reconstruction import Reconstruction


def reconstruction_to_colmap_space(recon: Reconstruction) -> Reconstruction:
    """Convert a Reconstruction to original-image COLMAP coordinates if metadata is available."""
    if recon.metadata.get("colmap_space", False):
        return recon.copy()

    original_coords = recon.metadata.get("original_coords", None)
    img_load_resolution = recon.metadata.get("img_load_resolution", None)
    if original_coords is None or img_load_resolution is None:
        return recon.copy()

    original_coords = np.asarray(original_coords)
    recon_colmap = recon.copy()
    for s in range(recon_colmap.num_images):
        x1, y1, x2, y2, orig_w, orig_h = original_coords[s]
        resize_ratio = max(orig_w, orig_h) / float(img_load_resolution)

        K = recon_colmap.intrinsics[s].copy()
        K[0, 0] *= resize_ratio
        K[1, 1] *= resize_ratio
        K[0, 2] = orig_w / 2.0
        K[1, 2] = orig_h / 2.0
        recon_colmap.intrinsics[s] = K

        cam_mask = recon_colmap.obs_camera_id == s
        if np.any(cam_mask):
            top_left = np.array([x1, y1], dtype=np.float64)
            recon_colmap.obs_xy[cam_mask] = (recon_colmap.obs_xy[cam_mask] - top_left) * resize_ratio

    recon_colmap.metadata["colmap_space"] = True
    return recon_colmap


def reconstruction_to_colmap_sparse(recon: Reconstruction, output_dir: str,
                                    camera_type: str = "PINHOLE"):
    """Write COLMAP-format cameras.bin, images.bin, points3D.bin to output_dir.

    Args:
        recon: Reconstruction instance.
        output_dir: Directory to write sparse model to.
        camera_type: "PINHOLE" or "SIMPLE_PINHOLE".
    """
    import pycolmap

    os.makedirs(output_dir, exist_ok=True)
    _reconstruction_to_pycolmap(recon, output_dir, camera_type)


def reconstruction_to_pycolmap(recon: Reconstruction,
                               camera_type: str = "PINHOLE") -> "pycolmap.Reconstruction":
    """Convert Reconstruction to pycolmap.Reconstruction object.

    Args:
        recon: Reconstruction instance.
        camera_type: "PINHOLE" or "SIMPLE_PINHOLE".

    Returns:
        pycolmap.Reconstruction object.
    """
    import pycolmap

    recon = reconstruction_to_colmap_space(recon)
    reconstruction = pycolmap.Reconstruction()
    S, P = recon.num_images, recon.num_points

    # Build camera params per image
    camera_params_list = []
    for s in range(S):
        K = recon.intrinsics[s]
        if camera_type == "PINHOLE":
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            params = np.array([fx, fy, cx, cy], dtype=np.float64)
        elif camera_type == "SIMPLE_PINHOLE":
            f = (K[0, 0] + K[1, 1]) / 2.0
            cx, cy = K[0, 2], K[1, 2]
            params = np.array([f, cx, cy], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")
        camera_params_list.append(params)

    # Add cameras (one per image)
    for s in range(S):
        cam = pycolmap.Camera(
            model=camera_type,
            width=int(recon.image_size_hw[s, 1]),
            height=int(recon.image_size_hw[s, 0]),
            params=camera_params_list[s],
            camera_id=s + 1,
        )
        reconstruction.add_camera(cam)

    # Add 3D points using add_point3D(xyz, track, color) which returns point3D_id
    # Point3D IDs are 1-indexed in COLMAP, and add_point3D assigns them sequentially
    for p in range(P):
        rgb = np.array([
            int(recon.points_rgb[p, 0]),
            int(recon.points_rgb[p, 1]),
            int(recon.points_rgb[p, 2]),
        ], dtype=np.uint8)
        reconstruction.add_point3D(
            recon.points3d[p].astype(np.float64),
            pycolmap.Track(),
            rgb,
        )

    # Add images with camera poses
    for s in range(S):
        T = recon.extrinsics[s]  # (3, 4) camera-from-world
        R = T[:3, :3]
        t = T[:3, 3]
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(R),
            t.astype(np.float64),
        )

        img = pycolmap.Image(
            id=s + 1,
            name=str(recon.image_names[s]),
            camera_id=s + 1,
            cam_from_world=cam_from_world,
        )
        reconstruction.add_image(img)

    # Add observations (Point2D and track elements)
    for obs_idx in range(recon.num_observations):
        cam_id = int(recon.obs_camera_id[obs_idx])
        pt_id = int(recon.obs_point_id[obs_idx])
        xy = recon.obs_xy[obs_idx]

        if cam_id + 1 not in reconstruction.images:
            continue
        if pt_id + 1 not in reconstruction.points3D:
            continue

        img = reconstruction.images[cam_id + 1]

        # Check if point is within image bounds
        H, W = int(recon.image_size_hw[cam_id, 0]), int(recon.image_size_hw[cam_id, 1])
        if xy[0] < 0 or xy[0] >= W or xy[1] < 0 or xy[1] >= H:
            continue

        p2d_idx = len(img.points2D)
        point2D = pycolmap.Point2D(xy=xy.astype(np.float64), point3D_id=pt_id + 1)
        img.points2D.append(point2D)

        # Add track element
        track = reconstruction.points3D[pt_id + 1].track
        track.add_element(cam_id + 1, p2d_idx)

    return reconstruction


def _reconstruction_to_pycolmap(recon: Reconstruction, output_dir: str,
                                camera_type: str = "PINHOLE"):
    """Helper that writes to disk."""
    recon_obj = reconstruction_to_pycolmap(recon, camera_type)
    recon_obj.write(output_dir)


def colmap_sparse_to_reconstruction(sparse_dir: str, image_dir: str = None) -> Reconstruction:
    """Read COLMAP sparse model into Reconstruction.

    Args:
        sparse_dir: Directory containing cameras.bin, images.bin, points3D.bin.
        image_dir: Optional image directory for extracting image filenames.

    Returns:
        Reconstruction instance.
    """
    import pycolmap

    sparse_dir = str(sparse_dir)
    if not os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        raise FileNotFoundError(f"No COLMAP model found in {sparse_dir}")

    reconstruction = pycolmap.Reconstruction(sparse_dir)

    # Extract image names and sizes
    img_ids = sorted(reconstruction.images.keys())
    if image_dir is not None:
        image_names = []
        for iid in img_ids:
            img = reconstruction.images[iid]
            name = img.name if img.name else f"image_{iid:04d}.jpg"
            image_names.append(name)
    else:
        image_names = [reconstruction.images[iid].name or f"image_{iid:04d}.jpg"
                       for iid in img_ids]

    # Build image_size_hw
    S = len(img_ids)
    img_map = {iid: idx for idx, iid in enumerate(img_ids)}
    image_size_hw = np.zeros((S, 2), dtype=np.int32)
    for iid in img_ids:
        idx = img_map[iid]
        cam = reconstruction.cameras[reconstruction.images[iid].camera_id]
        image_size_hw[idx] = [cam.height, cam.width]

    return Reconstruction.from_pycolmap(reconstruction, image_names, image_size_hw)
