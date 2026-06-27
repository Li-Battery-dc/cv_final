"""Unified Reconstruction data class for VGGT + BA + 3DGS pipeline.

Stores cameras, 3D points, and observation graph in a flat .npz-serializable format.
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Reconstruction:
    """Unified reconstruction format for the VGGT → BA → 3DGS pipeline.

    Fields:
        image_names: (S,) str array of image filenames.
        image_size_hw: (S, 2) int32 [height, width] of original images.
        intrinsics: (S, 3, 3) float64 K matrices.
        extrinsics: (S, 3, 4) float64 [R|t] OpenCV camera-from-world.
        points3d: (P, 3) float64 world coordinates.
        points_rgb: (P, 3) uint8 RGB colors 0-255.
        points_conf: (P,) float32 per-point confidence.
        obs_camera_id: (N,) int32 camera index for each observation.
        obs_point_id: (N,) int32 point index for each observation.
        obs_xy: (N, 2) float64 observed pixel coordinates.
        obs_conf: (N,) float32 per-observation confidence.
        metadata: dict for extra info.
    """

    image_names: np.ndarray       # (S,) str
    image_size_hw: np.ndarray     # (S, 2) int32
    intrinsics: np.ndarray        # (S, 3, 3) float64
    extrinsics: np.ndarray        # (S, 3, 4) float64
    points3d: np.ndarray          # (P, 3) float64
    points_rgb: np.ndarray        # (P, 3) uint8
    points_conf: np.ndarray       # (P,) float32
    obs_camera_id: np.ndarray     # (N,) int32
    obs_point_id: np.ndarray      # (N,) int32
    obs_xy: np.ndarray            # (N, 2) float64
    obs_conf: np.ndarray           # (N,) float32
    metadata: dict = field(default_factory=dict)

    # ---- Properties ----

    @property
    def num_images(self) -> int:
        return len(self.image_names)

    @property
    def num_points(self) -> int:
        return len(self.points3d)

    @property
    def num_observations(self) -> int:
        return len(self.obs_camera_id)

    # ---- Factory constructors ----

    @staticmethod
    def from_npz(path: str) -> "Reconstruction":
        """Load reconstruction from .npz file."""
        data = np.load(path, allow_pickle=True)
        metadata = {}
        if 'metadata' in data:
            try:
                metadata_raw = data['metadata'].item()
                if isinstance(metadata_raw, (bytes, bytearray)):
                    metadata = pickle.loads(metadata_raw)
                elif isinstance(metadata_raw, dict):
                    metadata = metadata_raw
            except (ValueError, AttributeError):
                pass

        return Reconstruction(
            image_names=data['image_names'].astype(str),
            image_size_hw=data['image_size_hw'],
            intrinsics=data['intrinsics'],
            extrinsics=data['extrinsics'],
            points3d=data['points3d'],
            points_rgb=data['points_rgb'],
            points_conf=data['points_conf'],
            obs_camera_id=data['obs_camera_id'],
            obs_point_id=data['obs_point_id'],
            obs_xy=data['obs_xy'],
            obs_conf=data['obs_conf'],
            metadata=metadata,
        )

    @staticmethod
    def from_pycolmap(reconstruction, image_names: list, image_size_hw: np.ndarray) -> "Reconstruction":
        """Create Reconstruction from a pycolmap.Reconstruction object.

        Args:
            reconstruction: pycolmap.Reconstruction instance.
            image_names: list of image filenames.
            image_size_hw: (S, 2) original image size array.

        Returns:
            Reconstruction instance.
        """
        import pycolmap
        # Extract cameras
        cam_ids = sorted(reconstruction.cameras.keys())
        cam_map = {cid: idx for idx, cid in enumerate(cam_ids)}
        S = len(image_names)

        intrinsics = np.zeros((S, 3, 3), dtype=np.float64)
        for cid, cam in reconstruction.cameras.items():
            idx = cam_map[cid]
            model = str(cam.model)
            if "." in model:
                model = model.split(".")[-1]
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
                intrinsics[idx] = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            elif model == "PINHOLE":
                fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
                intrinsics[idx] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            else:
                # Unknown model: try to extract fx, fy, cx, cy
                fx = cam.params[0]
                fy = cam.params[1] if len(cam.params) > 1 else fx
                cx = cam.params[2] if len(cam.params) > 2 else 0
                cy = cam.params[3] if len(cam.params) > 3 else cx
                intrinsics[idx] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Extract images (poses)
        img_ids = sorted(reconstruction.images.keys())
        img_map = {iid: idx for idx, iid in enumerate(img_ids)}
        extrinsics = np.zeros((S, 3, 4), dtype=np.float64)

        for iid, img in reconstruction.images.items():
            idx = img_map[iid]
            T = img.cam_from_world.matrix()  # (3, 4)
            extrinsics[idx] = T
            # Assign camera to this image if shared
            if img.camera_id in cam_map:
                intrinsics[idx] = intrinsics[cam_map[img.camera_id]]

        # Extract 3D points and observations
        pt_ids = sorted(reconstruction.points3D.keys())
        pt_id_to_idx = {pid: idx for idx, pid in enumerate(pt_ids)}
        P = len(pt_ids)

        points3d = np.zeros((P, 3), dtype=np.float64)
        points_rgb = np.zeros((P, 3), dtype=np.uint8)
        points_conf = np.ones(P, dtype=np.float32)

        obs_list_cam = []
        obs_list_pt = []
        obs_list_xy = []

        for pid, pt in reconstruction.points3D.items():
            pidx = pt_id_to_idx[pid]
            points3d[pidx] = pt.xyz
            points_rgb[pidx] = np.clip(pt.color.astype(np.uint8), 0, 255)
            if hasattr(pt, 'error') and pt.error is not None:
                points_conf[pidx] = max(0.0, 1.0 - float(pt.error) / 10.0)

            for element in pt.track.elements:
                iid = element.image_id
                p2d_idx = element.point2D_idx
                if iid in img_map:
                    img = reconstruction.images[iid]
                    cam_idx = img_map[iid]
                    if p2d_idx < len(img.points2D):
                        p2d = img.points2D[p2d_idx]
                        obs_list_cam.append(cam_idx)
                        obs_list_pt.append(pidx)
                        obs_list_xy.append(p2d.xy)

        N = len(obs_list_cam)
        return Reconstruction(
            image_names=np.array(image_names, dtype=str),
            image_size_hw=image_size_hw.astype(np.int32),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            points3d=points3d,
            points_rgb=points_rgb,
            points_conf=points_conf,
            obs_camera_id=np.array(obs_list_cam, dtype=np.int32) if N > 0 else np.zeros(0, dtype=np.int32),
            obs_point_id=np.array(obs_list_pt, dtype=np.int32) if N > 0 else np.zeros(0, dtype=np.int32),
            obs_xy=np.array(obs_list_xy, dtype=np.float64) if N > 0 else np.zeros((0, 2), dtype=np.float64),
            obs_conf=np.ones(N, dtype=np.float32) if N > 0 else np.zeros(0, dtype=np.float32),
        )

    @staticmethod
    def from_tracks(points3d: np.ndarray, points_rgb: np.ndarray,
                    pred_tracks: np.ndarray, pred_vis: np.ndarray,
                    pred_confs: np.ndarray,
                    extrinsics: np.ndarray, intrinsics: np.ndarray,
                    image_names: list, image_size_hw: np.ndarray,
                    vis_thresh: float = 0.2,
                    max_reproj_error: float = 8.0,
                    min_visible_frames: int = 3) -> "Reconstruction":
        """Build Reconstruction from raw VGGT track output.

        Args:
            points3d: (P, 3) 3D world points from VGGT track prediction.
            points_rgb: (P, 3) uint8 RGB colors.
            pred_tracks: (S, P, 2) 2D track locations.
            pred_vis: (S, P) visibility scores.
            pred_confs: (P,) per-point confidence.
            extrinsics: (S, 3, 4) camera extrinsics.
            intrinsics: (S, 3, 3) camera intrinsics.
            image_names: list of image filenames.
            image_size_hw: (S, 2) original image sizes.
            vis_thresh: visibility threshold.
            max_reproj_error: max allowed reprojection error in pixels.
            min_visible_frames: minimum frames a point must be visible in.

        Returns:
            Reconstruction with filtered observations.
        """
        from src.ba.utils import project_points  # noqa: F811

        S, P = pred_tracks.shape[0], pred_tracks.shape[1]

        # Build observation arrays from tracks
        obs_cam_list, obs_pt_list, obs_xy_list, obs_conf_list = [], [], [], []

        for s in range(S):
            for p in range(P):
                if pred_vis[s, p] > vis_thresh:
                    obs_cam_list.append(s)
                    obs_pt_list.append(p)
                    obs_xy_list.append(pred_tracks[s, p])
                    obs_conf_list.append(pred_vis[s, p])

        if len(obs_cam_list) == 0:
            raise ValueError("No valid observations after visibility filtering")

        obs_camera_id = np.array(obs_cam_list, dtype=np.int32)
        obs_point_id = np.array(obs_pt_list, dtype=np.int32)
        obs_xy = np.array(obs_xy_list, dtype=np.float64)
        obs_conf = np.array(obs_conf_list, dtype=np.float32)

        # Reprojection error filter
        if max_reproj_error is not None and max_reproj_error > 0:
            reproj_mask = np.ones(len(obs_camera_id), dtype=bool)
            for s in range(S):
                s_mask = obs_camera_id == s
                if not np.any(s_mask):
                    continue
                pid = obs_point_id[s_mask]
                pts3d = points3d[pid]
                proj = project_points(pts3d, extrinsics[s], intrinsics[s])
                diff = proj - obs_xy[s_mask]
                errors = np.sqrt(np.sum(diff ** 2, axis=1))
                reproj_mask[s_mask] = np.isfinite(errors) & (errors < max_reproj_error)

            obs_camera_id = obs_camera_id[reproj_mask]
            obs_point_id = obs_point_id[reproj_mask]
            obs_xy = obs_xy[reproj_mask]
            obs_conf = obs_conf[reproj_mask]

        # Filter points visible in fewer than min_visible_frames
        if min_visible_frames > 0 and len(obs_point_id) > 0:
            frame_counts = np.bincount(obs_point_id, minlength=P)
            valid_point_mask = frame_counts >= min_visible_frames

            # Re-index: keep only valid points
            old_to_new = np.full(P, -1, dtype=np.int32)
            old_to_new[valid_point_mask] = np.arange(np.sum(valid_point_mask))

            points3d = points3d[valid_point_mask]
            points_rgb = points_rgb[valid_point_mask]
            if pred_confs is not None and len(pred_confs) == P:
                pred_confs = pred_confs[valid_point_mask]

            obs_valid = old_to_new[obs_point_id] >= 0
            obs_camera_id = obs_camera_id[obs_valid]
            obs_point_id = old_to_new[obs_point_id[obs_valid]]
            obs_xy = obs_xy[obs_valid]
            obs_conf = obs_conf[obs_valid]

        P_final = len(points3d)
        if pred_confs is not None and len(pred_confs) == P_final:
            points_conf = pred_confs
        else:
            points_conf = np.ones(P_final, dtype=np.float32)

        return Reconstruction(
            image_names=np.array(image_names, dtype=str),
            image_size_hw=image_size_hw.astype(np.int32),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            points3d=points3d,
            points_rgb=points_rgb,
            points_conf=points_conf,
            obs_camera_id=obs_camera_id,
            obs_point_id=obs_point_id,
            obs_xy=obs_xy,
            obs_conf=obs_conf,
        )

    # ---- Serialization ----

    def to_npz(self, path: str):
        """Save reconstruction to .npz file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez_compressed(
            path,
            image_names=self.image_names,
            image_size_hw=self.image_size_hw,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            points3d=self.points3d,
            points_rgb=self.points_rgb,
            points_conf=self.points_conf,
            obs_camera_id=self.obs_camera_id,
            obs_point_id=self.obs_point_id,
            obs_xy=self.obs_xy,
            obs_conf=self.obs_conf,
            metadata=np.array([pickle.dumps(self.metadata)], dtype=object),
        )

    def copy(self) -> "Reconstruction":
        """Deep copy of reconstruction."""
        return Reconstruction(
            image_names=self.image_names.copy(),
            image_size_hw=self.image_size_hw.copy(),
            intrinsics=self.intrinsics.copy(),
            extrinsics=self.extrinsics.copy(),
            points3d=self.points3d.copy(),
            points_rgb=self.points_rgb.copy(),
            points_conf=self.points_conf.copy(),
            obs_camera_id=self.obs_camera_id.copy(),
            obs_point_id=self.obs_point_id.copy(),
            obs_xy=self.obs_xy.copy(),
            obs_conf=self.obs_conf.copy(),
            metadata=dict(self.metadata),
        )

    # ---- Observation Accessors ----

    def get_point_observations(self, point_idx: int) -> tuple:
        """Get all observations of a single 3D point.

        Returns (camera_ids, xy, conf) or raises.
        """
        mask = self.obs_point_id == point_idx
        return self.obs_camera_id[mask], self.obs_xy[mask], self.obs_conf[mask]

    def get_camera_observations(self, cam_idx: int) -> tuple:
        """Get all observations seen by a single camera.

        Returns (point_ids, xy, conf) or raises.
        """
        mask = self.obs_camera_id == cam_idx
        return self.obs_point_id[mask], self.obs_xy[mask], self.obs_conf[mask]

    # ---- Filtering ----

    def filter_outlier_observations(self, max_error_pixels: float = 5.0) -> "Reconstruction":
        """Remove observations with reprojection error above threshold.

        Returns a new Reconstruction (does not modify in place).
        """
        from src.ba.utils import compute_reprojection_errors_from_data

        stats = compute_reprojection_errors_from_data(
            self.points3d, self.extrinsics, self.intrinsics,
            self.obs_camera_id, self.obs_point_id, self.obs_xy
        )
        errors = stats['errors_2d']
        keep = np.isfinite(errors) & (errors < max_error_pixels)

        # Also remove points that would have fewer than 2 observations after filtering
        new_obs = Reconstruction(
            image_names=self.image_names.copy(),
            image_size_hw=self.image_size_hw.copy(),
            intrinsics=self.intrinsics.copy(),
            extrinsics=self.extrinsics.copy(),
            points3d=self.points3d.copy(),
            points_rgb=self.points_rgb.copy(),
            points_conf=self.points_conf.copy(),
            obs_camera_id=self.obs_camera_id[keep],
            obs_point_id=self.obs_point_id[keep],
            obs_xy=self.obs_xy[keep],
            obs_conf=self.obs_conf[keep],
            metadata=dict(self.metadata),
        )

        # Remove points with fewer than 2 observations
        if new_obs.num_observations > 0:
            frame_counts = np.bincount(new_obs.obs_point_id, minlength=new_obs.num_points)
            valid_pts = frame_counts >= 2
            old_to_new = np.full(new_obs.num_points, -1, dtype=np.int32)
            old_to_new[valid_pts] = np.arange(np.sum(valid_pts))

            new_obs.points3d = new_obs.points3d[valid_pts]
            new_obs.points_rgb = new_obs.points_rgb[valid_pts]
            new_obs.points_conf = new_obs.points_conf[valid_pts]

            obs_valid = old_to_new[new_obs.obs_point_id] >= 0
            new_obs.obs_camera_id = new_obs.obs_camera_id[obs_valid]
            new_obs.obs_point_id = old_to_new[new_obs.obs_point_id[obs_valid]]
            new_obs.obs_xy = new_obs.obs_xy[obs_valid]
            new_obs.obs_conf = new_obs.obs_conf[obs_valid]

        return new_obs

    def __repr__(self) -> str:
        return (f"Reconstruction(S={self.num_images}, P={self.num_points}, "
                f"obs={self.num_observations})")
