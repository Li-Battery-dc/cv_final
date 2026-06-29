#!/usr/bin/env python3
"""Training-free video frame selection for VGGT reconstruction."""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from vggt.utils.load_fn import load_and_preprocess_images_square

from src.ba.utils import compute_camera_delta
from src.utils.experiment import save_json, save_run_metadata, utc_timestamp


@dataclass
class FrameMetric:
    frame_idx: int
    timestamp_sec: float
    blur: float
    exposure: float
    texture: float
    neighbor_similarity: float
    quality_score: float
    candidate_score: float


def parse_args():
    parser = argparse.ArgumentParser(description="Select VGGT keyframes from a video without ffmpeg")
    parser.add_argument("--video", type=str, default="data/raw/3_scene.mp4")
    parser.add_argument("--output_scene_dir", type=str, default="data/scene_selected")
    parser.add_argument("--candidate_count", type=int, default=192)
    parser.add_argument("--final_count", type=int, default=64)
    parser.add_argument("--output_width", type=int, default=768)
    parser.add_argument("--output_height", type=int, default=432)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_load_resolution", type=int, default=448)
    parser.add_argument("--vggt_resolution", type=int, default=518)
    parser.add_argument("--centrality_topk", type=int, default=8)
    parser.add_argument("--gap_top_percent", type=float, default=10.0)
    parser.add_argument("--lambda_diversity", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def _rank01(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if not np.any(finite):
        return out
    order = np.argsort(values[finite])
    ranks = np.empty(np.sum(finite), dtype=np.float64)
    if len(ranks) == 1:
        ranks[:] = 1.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, len(ranks))
    if not higher_is_better:
        ranks = 1.0 - ranks
    out[finite] = ranks
    return out


def _frame_quality(frame_bgr: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    texture = float(np.mean(np.sqrt(gx * gx + gy * gy)))

    dark = float(np.mean(gray < 12))
    bright = float(np.mean(gray > 243))
    exposure = 1.0 - min(1.0, dark + bright)

    vec = cv2.resize(frame_bgr, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    vec = vec - float(np.mean(vec))
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    return blur, exposure, texture, vec


def scan_video(video_path: str) -> tuple[list[np.ndarray], list[FrameMetric], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    blur_values, exposure_values, texture_values = [], [], []
    embeddings = []
    frame_indices = []
    timestamps = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        blur, exposure, texture, emb = _frame_quality(frame)
        frames.append(frame)
        frame_indices.append(idx)
        timestamps.append(idx / fps if fps > 0 else float(idx))
        blur_values.append(blur)
        exposure_values.append(exposure)
        texture_values.append(texture)
        embeddings.append(emb)
        idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")

    embeddings_np = np.stack(embeddings, axis=0)
    neighbor_similarity = np.zeros(len(frames), dtype=np.float64)
    if len(frames) > 1:
        sims = np.sum(embeddings_np[:-1] * embeddings_np[1:], axis=1)
        neighbor_similarity[:-1] = sims
        neighbor_similarity[1:] = np.maximum(neighbor_similarity[1:], sims)

    blur_rank = _rank01(np.asarray(blur_values), True)
    exposure_rank = _rank01(np.asarray(exposure_values), True)
    texture_rank = _rank01(np.asarray(texture_values), True)
    duplicate_rank = _rank01(neighbor_similarity, False)
    quality_score = 0.35 * blur_rank + 0.30 * exposure_rank + 0.25 * texture_rank + 0.10 * duplicate_rank
    candidate_score = 0.85 * quality_score + 0.15 * duplicate_rank

    metrics = []
    for i in range(len(frames)):
        metrics.append(FrameMetric(
            frame_idx=int(frame_indices[i]),
            timestamp_sec=float(timestamps[i]),
            blur=float(blur_values[i]),
            exposure=float(exposure_values[i]),
            texture=float(texture_values[i]),
            neighbor_similarity=float(neighbor_similarity[i]),
            quality_score=float(quality_score[i]),
            candidate_score=float(candidate_score[i]),
        ))
    return frames, metrics, fps


def select_candidates(metrics: list[FrameMetric], candidate_count: int) -> list[int]:
    n = len(metrics)
    if candidate_count >= n:
        return list(range(n))
    selected = []
    boundaries = np.linspace(0, n, candidate_count + 1, dtype=np.int64)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        local = max(range(start, end), key=lambda i: metrics[i].candidate_score)
        selected.append(local)
    if len(selected) < candidate_count:
        selected_set = set(selected)
        remaining = sorted(
            [i for i in range(n) if i not in selected_set],
            key=lambda i: metrics[i].candidate_score,
            reverse=True,
        )
        selected.extend(remaining[:candidate_count - len(selected)])
    return sorted(selected[:candidate_count])


def _write_frames(
    frames: list[np.ndarray],
    indices: list[int],
    output_dir: str,
    width: int,
    height: int,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for source_idx in indices:
        frame = cv2.resize(frames[source_idx], (width, height), interpolation=cv2.INTER_AREA)
        name = f"frame_{source_idx:06d}.jpg"
        path = os.path.join(output_dir, name)
        ok = cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise IOError(f"Failed to write frame: {path}")
        paths.append(path)
    return paths


def _run_reliability_vggt(
    image_paths: list[str],
    img_load_resolution: int,
    vggt_resolution: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from src.vggt_export import load_vggt_model, run_vggt_inference, set_seed

    set_seed(seed)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running scout VGGT on {len(image_paths)} candidate frames ({device}, {dtype})")

    images, _ = load_and_preprocess_images_square(image_paths, img_load_resolution)
    images = images.to(device)
    model = load_vggt_model(device, dtype)
    extrinsic, _, _, _, frame_features, _, _ = run_vggt_inference(
        model=model,
        images=images,
        dtype=dtype,
        device=device,
        vggt_resolution=vggt_resolution,
        return_frame_features=True,
        return_point_map=False,
    )
    del model, images
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return extrinsic, frame_features


def _feature_centrality(features: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sim = features @ features.T
    np.fill_diagonal(sim, -np.inf)
    k = min(max(1, int(topk)), max(1, len(features) - 1))
    top = np.partition(sim, -k, axis=1)[:, -k:]
    centrality = np.mean(top, axis=1)
    return centrality, sim


def _pose_jump_scores(extrinsics: np.ndarray) -> np.ndarray:
    n = len(extrinsics)
    jumps = np.zeros(n, dtype=np.float64)
    if n < 3:
        return jumps
    edge_scores = np.zeros(n - 1, dtype=np.float64)
    for i in range(n - 1):
        rot, trans = compute_camera_delta(
            extrinsics[i, :3, :3], extrinsics[i, :3, 3],
            extrinsics[i + 1, :3, :3], extrinsics[i + 1, :3, 3],
        )
        edge_scores[i] = float(rot) + float(trans)
    jumps[0] = edge_scores[0]
    jumps[-1] = edge_scores[-1]
    for i in range(1, n - 1):
        jumps[i] = max(edge_scores[i - 1], edge_scores[i])
    return jumps


def select_final_frames(
    candidate_indices: list[int],
    candidate_metrics: list[FrameMetric],
    extrinsics: np.ndarray,
    frame_features: np.ndarray,
    final_count: int,
    centrality_topk: int,
    lambda_diversity: float,
    gap_top_percent: float,
) -> tuple[list[int], dict]:
    n = len(candidate_indices)
    if final_count >= n:
        return list(candidate_indices), {"reason": "candidate_count_le_final_count"}

    centrality, sim = _feature_centrality(frame_features, centrality_topk)
    pose_jump = _pose_jump_scores(extrinsics)
    quality = np.asarray([m.quality_score for m in candidate_metrics], dtype=np.float64)

    centrality_rank = _rank01(centrality, True)
    pose_smooth_rank = _rank01(pose_jump, False)
    quality_rank = _rank01(quality, True)
    reliability = 0.55 * centrality_rank + 0.25 * pose_smooth_rank + 0.20 * quality_rank

    selected_local = [int(np.argmax(reliability))]
    target_before_gap = max(1, int(round(final_count * 0.85)))
    while len(selected_local) < target_before_gap:
        selected_set = set(selected_local)
        best_idx = None
        best_score = -np.inf
        for i in range(n):
            if i in selected_set:
                continue
            diversity_penalty = float(np.max(sim[i, selected_local])) if selected_local else 0.0
            temporal_bonus = min(abs(i - j) for j in selected_local) / max(1.0, n - 1)
            score = reliability[i] - lambda_diversity * diversity_penalty + 0.10 * temporal_bonus
            if score > best_score:
                best_score = score
                best_idx = i
        selected_local.append(int(best_idx))

    selected_set = set(selected_local)
    edge_scores = np.zeros(n - 1, dtype=np.float64)
    for i in range(n - 1):
        rot, trans = compute_camera_delta(
            extrinsics[i, :3, :3], extrinsics[i, :3, 3],
            extrinsics[i + 1, :3, :3], extrinsics[i + 1, :3, 3],
        )
        edge_scores[i] = float(rot) + float(trans)
    if len(edge_scores) > 0:
        gap_threshold = np.percentile(edge_scores, 100.0 - gap_top_percent)
        gap_edges = np.flatnonzero(edge_scores >= gap_threshold)
    else:
        gap_edges = np.zeros(0, dtype=np.int64)

    for edge in gap_edges:
        if len(selected_set) >= final_count:
            break
        left_frame = candidate_indices[edge]
        right_frame = candidate_indices[edge + 1]
        lo, hi = sorted((left_frame, right_frame))
        in_gap = [
            i for i, source_idx in enumerate(candidate_indices)
            if i not in selected_set and lo <= source_idx <= hi
        ]
        if not in_gap:
            continue
        midpoint = 0.5 * (left_frame + right_frame)
        best = max(
            in_gap,
            key=lambda i: reliability[i] - abs(candidate_indices[i] - midpoint) / max(1.0, hi - lo + 1),
        )
        selected_set.add(int(best))

    while len(selected_set) < final_count:
        remaining = [i for i in range(n) if i not in selected_set]
        best = max(remaining, key=lambda i: reliability[i])
        selected_set.add(int(best))

    final_local = sorted(selected_set)
    if len(final_local) > final_count:
        final_local = sorted(final_local, key=lambda i: reliability[i], reverse=True)[:final_count]
        final_local = sorted(final_local)

    stats = {
        "centrality_mean": float(np.mean(centrality)),
        "centrality_min": float(np.min(centrality)),
        "centrality_max": float(np.max(centrality)),
        "pose_jump_mean": float(np.mean(pose_jump)),
        "pose_jump_p90": float(np.percentile(pose_jump, 90)),
        "gap_edges_considered": int(len(gap_edges)),
        "selected_reliability_mean": float(np.mean(reliability[final_local])),
        "candidate_reliability_mean": float(np.mean(reliability)),
    }
    return [candidate_indices[i] for i in final_local], stats


def main():
    from src.vggt_export import set_seed

    args = parse_args()
    set_seed(args.seed)

    output_scene_dir = os.path.abspath(args.output_scene_dir)
    candidate_dir = os.path.join(output_scene_dir, "candidates")
    image_dir = os.path.join(output_scene_dir, "images")
    if os.path.exists(output_scene_dir) and args.overwrite:
        shutil.rmtree(output_scene_dir)
    os.makedirs(output_scene_dir, exist_ok=True)
    if os.path.exists(candidate_dir) or os.path.exists(image_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"{output_scene_dir} already contains candidate/images outputs; pass --overwrite to replace them."
            )
        shutil.rmtree(candidate_dir, ignore_errors=True)
        shutil.rmtree(image_dir, ignore_errors=True)

    config_path = save_run_metadata(
        output_scene_dir,
        stage="video_select",
        params=vars(args),
        inputs={"video": os.path.abspath(args.video)},
        outputs={"output_scene_dir": output_scene_dir},
    )
    print(f"Saved run config: {config_path}")

    frames, metrics, fps = scan_video(args.video)
    print(f"Decoded {len(frames)} frames from {args.video} at fps={fps:.3f}")

    candidate_source_indices = select_candidates(metrics, args.candidate_count)
    candidate_paths = _write_frames(
        frames, candidate_source_indices, candidate_dir, args.output_width, args.output_height,
    )
    candidate_metrics = [metrics[i] for i in candidate_source_indices]
    save_json(os.path.join(output_scene_dir, "frame_selection_candidates.json"), {
        "timestamp_utc": utc_timestamp(),
        "video": os.path.abspath(args.video),
        "fps": fps,
        "decoded_frames": len(frames),
        "candidate_count": len(candidate_source_indices),
        "candidates": [asdict(m) for m in candidate_metrics],
    })

    extrinsics, frame_features = _run_reliability_vggt(
        candidate_paths,
        img_load_resolution=args.img_load_resolution,
        vggt_resolution=args.vggt_resolution,
        seed=args.seed,
    )
    final_source_indices, reliability_stats = select_final_frames(
        candidate_source_indices,
        candidate_metrics,
        extrinsics,
        frame_features,
        final_count=args.final_count,
        centrality_topk=args.centrality_topk,
        lambda_diversity=args.lambda_diversity,
        gap_top_percent=args.gap_top_percent,
    )
    final_paths = _write_frames(frames, final_source_indices, image_dir, args.output_width, args.output_height)

    final_set = set(final_source_indices)
    save_json(os.path.join(output_scene_dir, "frame_selection_summary.json"), {
        "timestamp_utc": utc_timestamp(),
        "video": os.path.abspath(args.video),
        "fps": fps,
        "decoded_frames": len(frames),
        "candidate_count": len(candidate_source_indices),
        "final_count": len(final_source_indices),
        "output_image_dir": image_dir,
        "candidate_dir": candidate_dir,
        "reliability_stats": reliability_stats,
        "final_frames": [
            {
                **asdict(metrics[i]),
                "output_name": os.path.basename(path),
                "was_candidate": i in set(candidate_source_indices),
            }
            for i, path in zip(final_source_indices, final_paths)
        ],
        "candidate_frames": [
            {**asdict(metrics[i]), "selected_final": i in final_set}
            for i in candidate_source_indices
        ],
    })

    print("=" * 60)
    print("Video Selection Summary")
    print("=" * 60)
    print(f"  Decoded frames: {len(frames)}")
    print(f"  Candidates:     {len(candidate_source_indices)} -> {candidate_dir}")
    print(f"  Final frames:   {len(final_source_indices)} -> {image_dir}")
    print(f"  First/last:      {final_source_indices[0]} / {final_source_indices[-1]}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
