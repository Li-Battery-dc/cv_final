import os
import tempfile

import cv2
import numpy as np

from src.improvement.geometry_filter import build_filtered_dense_reconstruction
from src.improvement.video_select import scan_video, select_candidates


def _simple_reconstruction_inputs():
    s, h, w = 2, 4, 4
    depth = np.ones((s, h, w, 1), dtype=np.float32) * 2.0
    extrinsics = np.zeros((s, 3, 4), dtype=np.float64)
    extrinsics[:, :3, :3] = np.eye(3)
    intrinsics = np.tile(
        np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float64),
        (s, 1, 1),
    )

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = (xx - intrinsics[0, 0, 2]) * depth[0, ..., 0] / intrinsics[0, 0, 0]
    y = (yy - intrinsics[0, 1, 2]) * depth[0, ..., 0] / intrinsics[0, 1, 1]
    z = depth[0, ..., 0]
    points_one = np.stack([x, y, z], axis=-1)
    points_depth = np.stack([points_one, points_one], axis=0).astype(np.float64)
    points_point = points_depth.copy()
    points_point[0, 0, 0] += np.array([100.0, 0.0, 0.0])

    images = np.zeros((s, h, w, 3), dtype=np.uint8)
    images[..., 0] = 128
    return depth, extrinsics, intrinsics, points_depth, points_point, images


def test_geometry_filter_builds_dense_reconstruction_without_observations():
    depth, extrinsics, intrinsics, points_depth, points_point, images = _simple_reconstruction_inputs()

    result = build_filtered_dense_reconstruction(
        image_names=["a.jpg", "b.jpg"],
        image_size_hw=np.array([[4, 4], [4, 4]], dtype=np.int32),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        depth_map=depth,
        depth_conf=np.ones((2, 4, 4), dtype=np.float32),
        points_depth=points_depth,
        points_point=points_point,
        images_np=images,
        disagreement_percentile=80,
        reproj_percentile=100,
        min_reproj_votes=1,
        max_points=100,
        rng=np.random.default_rng(0),
    )

    recon = result.reconstruction
    assert recon.num_points > 0
    assert recon.num_observations == 0
    assert result.stats["after_disagreement"] < result.stats["finite_points"]
    assert recon.metadata["point_source"] == "depth_camera_unprojection_filtered"


def test_video_scan_and_candidate_selection_without_ffmpeg():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "tiny.mp4")
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (64, 48),
        )
        assert writer.isOpened()
        for i in range(10):
            frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
            cv2.putText(frame, str(i), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            writer.write(frame)
        writer.release()

        frames, metrics, fps = scan_video(video_path)
        selected = select_candidates(metrics, 4)

    assert len(frames) == 10
    assert fps > 0
    assert len(selected) == 4
    assert selected == sorted(selected)

