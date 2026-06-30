#!/usr/bin/env python3
"""Generate report-ready metrics tables and visual assets from experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    scene: str
    kind: str
    path: str
    note: str = ""


OFFICIAL_RUNS = [
    RunSpec("scene_raw", "scene", "official_3dgs", "data/scene/gs_official_raw/runs/20260627_135803_gaussian_official", "uniform VGGT raw"),
    RunSpec("scene_ba", "scene", "official_3dgs", "data/scene/gs_official_ba/runs/20260627_133737_gaussian_official", "uniform custom BA"),
    RunSpec("scene_random_raw", "scene", "official_3dgs", "data/scene/gs_official_random_raw/runs/20260627_141252_gaussian_official", "uniform raw cameras, random init"),
    RunSpec("scene_random_ba", "scene", "official_3dgs", "data/scene/gs_official_random_ba/runs/20260627_140637_gaussian_official", "uniform BA cameras, random init"),
    RunSpec("human1_ba_hq_masked", "1-human", "official_3dgs", "data/1-human/gs_official_ba_hq_masked/runs/20260629_084243_gaussian_official", "HQ tracks, mask-white"),
    RunSpec("human2_ba_hq_masked", "2-human", "official_3dgs", "data/2-human/gs_official_ba_hq_masked/runs/20260629_085004_gaussian_official", "HQ tracks, mask-white"),
    RunSpec("scene32_raw_hq", "scene_32", "official_3dgs", "data/scene_32/gs_official_raw_hq_768/runs/20260629_092440_gaussian_official", "32-frame raw negative result"),
    RunSpec("scene32_ba_hq", "scene_32", "official_3dgs", "data/scene_32/gs_official_ba_hq_768/runs/20260629_092117_gaussian_official", "32-frame BA negative result"),
]

SELECTED_OFFICIAL_NOTES = {
    "gs_official_selected_raw": "selected sparse raw",
    "gs_official_selected_ba": "selected sparse custom BA",
    "gs_official_dense_depth_only": "selected dense depth only",
    "gs_official_dense_pointmap_only": "selected dense point map only",
    "gs_official_dense_disagreement_only": "selected dense disagreement filter",
    "gs_official_dense_reprojection_only": "selected dense reprojection filter",
    "gs_official_dense_filtered_full": "selected dense full filter",
}

BA_RUNS = [
    RunSpec("scene_ba", "scene", "ba_custom", "data/scene/ba_custom/runs/20260627_075839_ba_custom"),
    RunSpec("human1_ba_hq", "1-human", "ba_custom", "data/1-human/ba_custom_hq/runs/20260629_083912_ba_custom"),
    RunSpec("human2_ba_hq", "2-human", "ba_custom", "data/2-human/ba_custom_hq/runs/20260629_084714_ba_custom"),
    RunSpec("scene32_ba_hq", "scene_32", "ba_custom", "data/scene_32/ba_custom_hq_768/runs/20260629_091651_ba_custom"),
]

VGGT_RUNS = [
    RunSpec("scene_vggt_raw", "scene", "vggt", "data/scene/vggt_raw/runs/20260627_074913_vggt_export"),
    RunSpec("human1_vggt_hq", "1-human", "vggt", "data/1-human/vggt_raw_hq_unmasked/runs/20260629_083702_vggt_export"),
    RunSpec("human2_vggt_hq", "2-human", "vggt", "data/2-human/vggt_raw_hq_unmasked/runs/20260629_084616_vggt_export"),
    RunSpec("scene32_vggt_hq", "scene_32", "vggt", "data/scene_32/vggt_raw_hq_768/runs/20260629_091545_vggt_export"),
]

CUSTOM_GS_RUNS = [
    RunSpec("scene_custom_gs_ba", "scene", "custom_3dgs", "data/scene/gs_custom_ba", "custom renderer baseline"),
]

EXPECTED_IMPROVEMENT_ROOTS = [
    "data/scene_selected/vggt_improved",
    "data/scene_selected/vggt_dense_ablation_200k",
    "data/scene_selected/ba_custom_depth",
    "data/scene_selected/gs_official_selected_raw",
    "data/scene_selected/gs_official_selected_ba",
    "data/scene_selected/gs_official_dense_depth_only",
    "data/scene_selected/gs_official_dense_pointmap_only",
    "data/scene_selected/gs_official_dense_disagreement_only",
    "data/scene_selected/gs_official_dense_reprojection_only",
    "data/scene_selected/gs_official_dense_filtered_full",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_results_run(root: Path) -> Path | None:
    runs = sorted(root.glob("runs/*/results.json"))
    if not runs:
        return None
    return runs[-1].parent


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def official_metrics_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    warnings = []
    specs = list(OFFICIAL_RUNS)
    for rel_root, note in SELECTED_OFFICIAL_NOTES.items():
        run_dir = latest_results_run(root / "data/scene_selected" / rel_root)
        if run_dir is None:
            continue
        specs.append(RunSpec(rel_root.replace("gs_official_", "selected_"), "scene_selected", "official_3dgs", str(run_dir.relative_to(root)), note))

    seen_dirs: set[str] = set()
    for spec in specs:
        run_dir = root / spec.path
        run_key = str(run_dir.resolve())
        if run_key in seen_dirs:
            continue
        seen_dirs.add(run_key)
        results = load_json(run_dir / "results.json")
        summary = load_json(run_dir / "summary.json")
        if not results:
            warnings.append(f"missing official metrics: {spec.run_id} -> {run_dir}")
            continue
        key = next(iter(results))
        metrics = results[key]
        split = (summary or {}).get("test_split") or {}
        rows.append({
            "run_id": spec.run_id,
            "scene": spec.scene,
            "method": spec.note,
            "iteration": key,
            "psnr": metrics.get("PSNR"),
            "ssim": metrics.get("SSIM"),
            "lpips": metrics.get("LPIPS"),
            "test_count": split.get("test_count"),
            "run_dir": str(run_dir),
        })
    return rows, warnings


def ba_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    warnings = []
    fields = [
        "rmse_before", "rmse_after", "median_before", "median_after",
        "p90_before", "p90_after", "n_outliers_removed",
        "n_points_before", "n_points_after", "n_obs_before", "n_obs_after",
        "time_seconds",
    ]
    for spec in BA_RUNS:
        run_dir = root / spec.path
        summary = load_json(run_dir / "summary.json")
        stats = (summary or {}).get("stats")
        if not stats:
            stats = load_json(run_dir / "ba_stats.json")
        if not stats:
            warnings.append(f"missing BA stats: {spec.run_id} -> {run_dir}")
            continue
        row = {"run_id": spec.run_id, "scene": spec.scene, "kind": spec.kind, "run_dir": str(run_dir)}
        row.update({field: stats.get(field) for field in fields})
        rows.append(row)
    return rows, warnings


def vggt_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    warnings = []
    for spec in VGGT_RUNS:
        run_dir = root / spec.path
        summary = load_json(run_dir / "summary.json")
        if not summary:
            warnings.append(f"missing VGGT summary: {spec.run_id} -> {run_dir}")
            continue
        rows.append({
            "run_id": spec.run_id,
            "scene": spec.scene,
            "images": summary.get("images"),
            "track_points": summary.get("track_points"),
            "observations": summary.get("observations"),
            "dense_points": summary.get("dense_points"),
            "vggt_inference_seconds": summary.get("vggt_inference_seconds"),
            "track_prediction_seconds": summary.get("track_prediction_seconds"),
            "run_dir": str(run_dir),
        })
    return rows, warnings


def custom_gs_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    warnings = []
    for spec in CUSTOM_GS_RUNS:
        run_dir = root / spec.path
        metrics = load_json(run_dir / "metrics.json")
        final = (metrics or {}).get("final") or {}
        if not final:
            warnings.append(f"missing custom 3DGS metrics: {spec.run_id} -> {run_dir}")
            continue
        rows.append({
            "run_id": spec.run_id,
            "scene": spec.scene,
            "method": spec.note,
            "psnr": final.get("psnr"),
            "ssim": final.get("ssim"),
            "lpips": final.get("lpips"),
            "final_n_gaussians": final.get("final_n_gaussians"),
            "training_time_seconds": final.get("training_time_seconds"),
            "run_dir": str(run_dir),
        })
    return rows, warnings


def geometry_filter_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    warnings = []
    run_roots = sorted((root / "data/scene_selected/vggt_dense_ablation_200k").glob("runs/*_vggt_export"))
    if not run_roots:
        warnings.append("missing 200k dense ablation export under: data/scene_selected/vggt_dense_ablation_200k")
        return rows, warnings
    run_dir = run_roots[-1]
    for stats_path in sorted(run_dir.glob("geometry_filter_*_stats.json")):
        if stats_path.name == "geometry_filter_stats.json":
            continue
        stats = load_json(stats_path)
        if not stats:
            continue
        rows.append({
            "run_id": run_dir.name,
            "variant": stats.get("variant", stats_path.stem.replace("geometry_filter_", "").replace("_stats", "")),
            "total_pixels": stats.get("total_pixels"),
            "finite_points": stats.get("finite_points"),
            "after_disagreement": stats.get("after_disagreement"),
            "after_reprojection": stats.get("after_reprojection"),
            "output_points": stats.get("output_points"),
            "disagreement_threshold": stats.get("disagreement_threshold"),
            "reproj_threshold": stats.get("reproj_threshold"),
            "points_with_valid_neighbors": stats.get("points_with_valid_neighbors"),
            "mean_reproj_votes_kept": stats.get("mean_reproj_votes_kept"),
            "run_dir": str(run_dir),
        })
    if not rows:
        warnings.append(f"no geometry filter stats found under: {run_dir}")
    return rows, warnings


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def make_contact_sheet(
    output_path: Path,
    rows: list[tuple[str, Path]],
    view_count: int = 4,
    thumb_width: int = 240,
) -> bool:
    existing_rows = [(label, path) for label, path in rows if path.exists()]
    if not existing_rows:
        return False

    images_by_row: list[tuple[str, list[Image.Image]]] = []
    for label, directory in existing_rows:
        files = sorted(directory.glob("*.png"))[:view_count]
        if not files:
            continue
        thumbs = []
        for file in files:
            img = Image.open(file).convert("RGB")
            scale = thumb_width / img.width
            thumb = img.resize((thumb_width, max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
            thumbs.append(thumb)
        if thumbs:
            images_by_row.append((label, thumbs))
    if not images_by_row:
        return False

    label_w = 180
    pad = 12
    row_h = max(img.height for _, imgs in images_by_row for img in imgs) + pad * 2
    col_count = max(len(imgs) for _, imgs in images_by_row)
    width = label_w + col_count * (thumb_width + pad) + pad
    height = len(images_by_row) * row_h + pad
    sheet = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)

    y = pad
    for label, imgs in images_by_row:
        draw_label(draw, (pad, y + 8), label)
        x = label_w
        for img in imgs:
            sheet.paste(img, (x, y))
            x += thumb_width + pad
        y += row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return True


def make_metric_chart(output_path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    scene_rows = [row for row in rows if row["scene"] in {"scene", "scene_selected"}]
    if not scene_rows:
        return False
    labels = [row["run_id"].replace("scene_", "").replace("selected_", "sel_") for row in scene_rows]
    psnr = [float(row["psnr"]) for row in scene_rows]
    ssim = [float(row["ssim"]) for row in scene_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].bar(labels, psnr, color="#386cb0")
    axes[0].set_title("Scene PSNR")
    axes[0].set_ylabel("dB")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, ssim, color="#7fc97f")
    axes[1].set_title("Scene SSIM")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def make_dense_metric_chart(output_path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    dense_rows = [row for row in rows if row["scene"] == "scene_selected" and "dense" in row["run_id"]]
    if not dense_rows:
        return False
    dense_rows = sorted(dense_rows, key=lambda row: row["run_id"])
    labels = [row["run_id"].replace("selected_dense_", "") for row in dense_rows]
    psnr = [float(row["psnr"]) for row in dense_rows]
    lpips = [float(row["lpips"]) for row in dense_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].bar(labels, psnr, color="#386cb0")
    axes[0].set_title("Selected Dense PSNR")
    axes[0].set_ylabel("dB")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, lpips, color="#f0027f")
    axes[1].set_title("Selected Dense LPIPS")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def audit_improvement_outputs(root: Path) -> list[str]:
    warnings = []
    for rel in EXPECTED_IMPROVEMENT_ROOTS:
        path = root / rel
        if not path.exists():
            warnings.append(f"pending improvement output: {rel}")
            continue
        if "gs_official" in rel and not list(path.glob("runs/*/results.json")):
            warnings.append(f"pending official 3DGS metrics under: {rel}")
        if rel.endswith("vggt_improved") and not list(path.glob("runs/*/reconstruction.npz")):
            warnings.append(f"incomplete selected VGGT export under: {rel}")
    return warnings


def write_audit(path: Path, warnings: list[str], generated: list[str]) -> None:
    lines = ["# Report Asset Audit", "", "## Generated", ""]
    lines.extend(f"- `{item}`" for item in generated)
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate report metrics and render comparison assets")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--asset_dir", type=str, default="report/assets")
    parser.add_argument("--metrics_dir", type=str, default="report/metrics")
    parser.add_argument("--view_count", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    asset_dir = root / args.asset_dir
    metrics_dir = root / args.metrics_dir
    generated: list[str] = []
    warnings: list[str] = []

    official_rows, official_warnings = official_metrics_rows(root)
    ba_metric_rows, ba_warnings = ba_rows(root)
    vggt_metric_rows, vggt_warnings = vggt_rows(root)
    custom_rows, custom_warnings = custom_gs_rows(root)
    geometry_rows, geometry_warnings = geometry_filter_rows(root)
    warnings.extend(official_warnings + ba_warnings + vggt_warnings + custom_warnings + geometry_warnings)
    warnings.extend(audit_improvement_outputs(root))

    write_csv(metrics_dir / "official_3dgs_metrics.csv", official_rows, [
        "run_id", "scene", "method", "iteration", "psnr", "ssim", "lpips", "test_count", "run_dir",
    ])
    generated.append(str(metrics_dir / "official_3dgs_metrics.csv"))
    write_csv(metrics_dir / "ba_metrics.csv", ba_metric_rows, [
        "run_id", "scene", "kind", "rmse_before", "rmse_after", "median_before", "median_after",
        "p90_before", "p90_after", "n_outliers_removed", "n_points_before", "n_points_after",
        "n_obs_before", "n_obs_after", "time_seconds", "run_dir",
    ])
    generated.append(str(metrics_dir / "ba_metrics.csv"))
    write_csv(metrics_dir / "vggt_metrics.csv", vggt_metric_rows, [
        "run_id", "scene", "images", "track_points", "observations", "dense_points",
        "vggt_inference_seconds", "track_prediction_seconds", "run_dir",
    ])
    generated.append(str(metrics_dir / "vggt_metrics.csv"))
    write_csv(metrics_dir / "custom_3dgs_metrics.csv", custom_rows, [
        "run_id", "scene", "method", "psnr", "ssim", "lpips",
        "final_n_gaussians", "training_time_seconds", "run_dir",
    ])
    generated.append(str(metrics_dir / "custom_3dgs_metrics.csv"))
    write_csv(metrics_dir / "geometry_filter_metrics.csv", geometry_rows, [
        "run_id", "variant", "total_pixels", "finite_points", "after_disagreement",
        "after_reprojection", "output_points", "disagreement_threshold", "reproj_threshold",
        "points_with_valid_neighbors", "mean_reproj_votes_kept", "run_dir",
    ])
    generated.append(str(metrics_dir / "geometry_filter_metrics.csv"))

    all_metrics = {
        "official_3dgs": official_rows,
        "ba": ba_metric_rows,
        "vggt": vggt_metric_rows,
        "custom_3dgs": custom_rows,
        "geometry_filter": geometry_rows,
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    generated.append(str(metrics_dir / "all_metrics.json"))

    chart_path = asset_dir / "scene_metric_bars.png"
    if make_metric_chart(chart_path, official_rows):
        generated.append(str(chart_path))
    else:
        warnings.append("matplotlib unavailable or no scene metrics; skipped scene_metric_bars.png")

    dense_chart_path = asset_dir / "selected_dense_metric_bars.png"
    if make_dense_metric_chart(dense_chart_path, official_rows):
        generated.append(str(dense_chart_path))

    scene_sheet = asset_dir / "scene_main_render_comparison.png"
    if make_contact_sheet(scene_sheet, [
        ("GT", root / "data/scene/gs_official_ba/runs/20260627_133737_gaussian_official/test/ours_10000/gt"),
        ("Raw", root / "data/scene/gs_official_raw/runs/20260627_135803_gaussian_official/test/ours_10000/renders"),
        ("BA", root / "data/scene/gs_official_ba/runs/20260627_133737_gaussian_official/test/ours_10000/renders"),
        ("Random Raw", root / "data/scene/gs_official_random_raw/runs/20260627_141252_gaussian_official/test/ours_10000/renders"),
        ("Random BA", root / "data/scene/gs_official_random_ba/runs/20260627_140637_gaussian_official/test/ours_10000/renders"),
    ], args.view_count):
        generated.append(str(scene_sheet))

    human_sheet = asset_dir / "human_render_comparison.png"
    if make_contact_sheet(human_sheet, [
        ("1H GT", root / "data/1-human/gs_official_ba_hq_masked/runs/20260629_084243_gaussian_official/test/ours_10000/gt"),
        ("1H Render", root / "data/1-human/gs_official_ba_hq_masked/runs/20260629_084243_gaussian_official/test/ours_10000/renders"),
        ("2H GT", root / "data/2-human/gs_official_ba_hq_masked/runs/20260629_085004_gaussian_official/test/ours_10000/gt"),
        ("2H Render", root / "data/2-human/gs_official_ba_hq_masked/runs/20260629_085004_gaussian_official/test/ours_10000/renders"),
    ], args.view_count):
        generated.append(str(human_sheet))

    dense_sheet = asset_dir / "selected_dense_render_comparison.png"
    dense_root = root / "data/scene_selected"
    if make_contact_sheet(dense_sheet, [
        ("GT", dense_root / "gs_official_dense_filtered_full/runs/20260629_152158_gaussian_official/test/ours_10000/gt"),
        ("Depth", dense_root / "gs_official_dense_depth_only/runs/20260629_152539_gaussian_official/test/ours_10000/renders"),
        ("Pointmap", dense_root / "gs_official_dense_pointmap_only/runs/20260629_152909_gaussian_official/test/ours_10000/renders"),
        ("Disagree", dense_root / "gs_official_dense_disagreement_only/runs/20260629_153239_gaussian_official/test/ours_10000/renders"),
        ("Reproj", dense_root / "gs_official_dense_reprojection_only/runs/20260629_153608_gaussian_official/test/ours_10000/renders"),
        ("Full", dense_root / "gs_official_dense_filtered_full/runs/20260629_152158_gaussian_official/test/ours_10000/renders"),
    ], args.view_count):
        generated.append(str(dense_sheet))

    write_audit(metrics_dir / "asset_audit.md", warnings, generated)
    print(f"Generated {len(generated)} report files. Audit: {metrics_dir / 'asset_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
