"""Bundle Adjustment problem formulation.

Encapsulates: parameter packing, residual computation, sparse Jacobian structure.
"""

import numpy as np
from scipy.sparse import csc_matrix

from src.data.reconstruction import Reconstruction
from src.ba.utils import project_points, rodrigues_rotate, matrix_to_rotvec


class BAProblem:
    """Bundle Adjustment problem using SciPy least_squares.

    Variables (1D vector x):
      For each free camera i (i >= n_fixed_cameras):
        rv: SO(3) rotation vector (3), t: translation (3)  -> 6 * (S - n_fixed)
      For each 3D point j:
        X, Y, Z (3)                                      -> 3 * P
      Total: n_vars = 6 * (S - n_fixed) + 3 * P

    Fixed:
      - First n_fixed_cameras cameras (gauge anchor, remove 7-DOF ambiguity)
      - All intrinsic matrices K (unchanged)

    Residual: for each observation (cam i, point j):
      r_ij = pi(K_i, R(rv_i), t_i, X_j) - u_ij
      Stacked as [dx_1, dy_1, dx_2, dy_2, ...] -> 2 * N_obs elements.
    """

    def __init__(self, reconstruction: Reconstruction,
                 n_fixed_cameras: int = 2,
                 huber_delta: float = 1.0):
        self.recon = reconstruction
        self.S = reconstruction.num_images
        self.P = reconstruction.num_points
        self.N = reconstruction.num_observations
        self.n_fixed = n_fixed_cameras
        self.huber_delta = huber_delta

        if self.N == 0:
            raise ValueError("No observations in reconstruction")

        # Build variable index maps
        # Camera variable offset: 6 per free camera
        self._cam_var_start = np.full(self.S, -1, dtype=np.int32)
        for i in range(self.n_fixed, self.S):
            self._cam_var_start[i] = 6 * (i - self.n_fixed)

        # Point variable offset: after all camera variables
        self._point_var_start = np.zeros(self.P, dtype=np.int32)
        point_offset = 6 * (self.S - self.n_fixed)
        for j in range(self.P):
            self._point_var_start[j] = point_offset + 3 * j

        self.n_vars = 6 * (self.S - self.n_fixed) + 3 * self.P

        # Pre-compute observation grouping for efficiency
        self._build_obs_groups()

    def _build_obs_groups(self):
        """Group observations by camera for batch projection."""
        obs_by_cam = {}
        for s in range(self.S):
            mask = self.recon.obs_camera_id == s
            if np.any(mask):
                obs_by_cam[s] = {
                    'point_ids': self.recon.obs_point_id[mask],
                    'obs_xy': self.recon.obs_xy[mask],
                    'obs_indices': np.where(mask)[0],
                }
        self._obs_by_cam = obs_by_cam

    # ---- Parameter pack/unpack ----

    def pack_params(self) -> np.ndarray:
        """Pack current camera poses and points into 1D vector.

        Returns:
            x: (n_vars,) parameter vector.
        """
        x = np.zeros(self.n_vars, dtype=np.float64)

        # Camera parameters
        for i in range(self.n_fixed, self.S):
            start = self._cam_var_start[i]
            R = self.recon.extrinsics[i, :3, :3]
            t = self.recon.extrinsics[i, :3, 3]
            rv = matrix_to_rotvec(R)
            x[start:start + 3] = rv
            x[start + 3:start + 6] = t

        # Point parameters
        for j in range(self.P):
            start = self._point_var_start[j]
            x[start:start + 3] = self.recon.points3d[j]

        return x

    def unpack_params(self, x: np.ndarray) -> tuple:
        """Unpack parameter vector into extrinsics and points.

        Args:
            x: (n_vars,) parameter vector.

        Returns:
            extrinsics: (S, 3, 4) updated camera-from-world matrices.
            points3d: (P, 3) updated 3D points.
        """
        extrinsics = self.recon.extrinsics.copy()
        points3d = self.recon.points3d.copy()

        # Update cameras
        for i in range(self.n_fixed, self.S):
            start = self._cam_var_start[i]
            rv = x[start:start + 3]
            t = x[start + 3:start + 6]
            R = rodrigues_rotate(rv)
            extrinsics[i, :3, :3] = R
            extrinsics[i, :3, 3] = t

        # Update points
        for j in range(self.P):
            start = self._point_var_start[j]
            points3d[j] = x[start:start + 3]

        return extrinsics, points3d

    # ---- Residual computation ----

    def compute_residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute all reprojection residuals.

        Args:
            x: (n_vars,) current parameter estimate.

        Returns:
            residuals: (2 * N,) stacked [dx, dy, ...] for each observation.
        """
        extrinsics, points3d = self.unpack_params(x)
        residuals = np.zeros(2 * self.N, dtype=np.float64)

        for cam_idx, group in self._obs_by_cam.items():
            point_ids = group['point_ids']
            obs_xy = group['obs_xy']
            obs_indices = group['obs_indices']

            K = self.recon.intrinsics[cam_idx]
            T = extrinsics[cam_idx]
            pts3d = points3d[point_ids]

            # Project all points for this camera
            proj = project_points(pts3d, T, K)

            # Compute residuals
            diff = proj - obs_xy
            # Handle behind-camera points: assign large error
            behind = ~np.isfinite(diff[:, 0])
            if np.any(behind):
                diff[behind] = 1e6  # large penalty

            for k, obs_idx in enumerate(obs_indices):
                residuals[2 * obs_idx] = diff[k, 0]
                residuals[2 * obs_idx + 1] = diff[k, 1]

        return residuals

    # ---- Sparse Jacobian structure ----

    def compute_sparsity(self) -> csc_matrix:
        """Build boolean sparsity pattern of the Jacobian matrix.

        Each observation (cam i, point j) contributes:
          - Jacobian w.r.t camera i: 2x6 block (if cam i is free)
          - Jacobian w.r.t point j: 2x3 block

        Returns:
            J_sparsity: (2*N, n_vars) sparse matrix (pattern only).
        """
        M = 2 * self.N
        N_vars = self.n_vars

        row_indices = []
        col_indices = []

        for obs_idx in range(self.N):
            cam_idx = int(self.recon.obs_camera_id[obs_idx])
            pt_idx = int(self.recon.obs_point_id[obs_idx])

            row_base = 2 * obs_idx

            # Camera Jacobian block (cam_idx is free)
            if cam_idx >= self.n_fixed:
                cam_col = self._cam_var_start[cam_idx]
                for dr in range(2):
                    for dc in range(6):
                        row_indices.append(row_base + dr)
                        col_indices.append(cam_col + dc)

            # Point Jacobian block
            pt_col = self._point_var_start[pt_idx]
            for dr in range(2):
                for dc in range(3):
                    row_indices.append(row_base + dr)
                    col_indices.append(pt_col + dc)

        data = np.ones(len(row_indices), dtype=np.float64)
        J_sparsity = csc_matrix((data, (row_indices, col_indices)),
                                shape=(M, N_vars))
        return J_sparsity

    # ---- Statistics ----

    def compute_stats(self, x: np.ndarray = None) -> dict:
        """Compute reprojection error statistics at current state."""
        if x is not None:
            extrinsics, points3d = self.unpack_params(x)
        else:
            extrinsics = self.recon.extrinsics
            points3d = self.recon.points3d

        from src.ba.utils import compute_reprojection_errors_from_data
        return compute_reprojection_errors_from_data(
            points3d, extrinsics, self.recon.intrinsics,
            self.recon.obs_camera_id, self.recon.obs_point_id,
            self.recon.obs_xy,
        )

    def compute_camera_deltas(self, extrinsics_opt: np.ndarray) -> tuple:
        """Compute rotation/translation changes from initial to optimized cameras.

        Returns:
            rot_deltas: (S-n_fixed,) rotation angles in degrees.
            trans_deltas: (S-n_fixed,) translation distances.
        """
        from src.ba.utils import compute_camera_delta
        rot_deltas = []
        trans_deltas = []
        for i in range(self.n_fixed, self.S):
            R0 = self.recon.extrinsics[i, :3, :3]
            t0 = self.recon.extrinsics[i, :3, 3]
            R1 = extrinsics_opt[i, :3, :3]
            t1 = extrinsics_opt[i, :3, 3]
            rd, td = compute_camera_delta(R0, t0, R1, t1)
            rot_deltas.append(rd)
            trans_deltas.append(td)
        return np.array(rot_deltas), np.array(trans_deltas)
