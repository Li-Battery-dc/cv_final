# Report Asset Gap Audit

This audit follows `docs/report_outline.md` and marks which report sections are
currently supported by assets and which sections need placeholders.

## 1. 摘要

- Existing data: main scene BA and 3DGS metrics, human metrics, selected dense
  support metrics.
- Missing: final original-scene dense ablation conclusion.
- Report action: use a placeholder sentence for final improvement conclusion.

## 2. 任务要求与系统设计

- Existing assets: no figure required; use a Mermaid pipeline diagram in the
  report.
- Missing: none.

## 3. 数据格式与坐标约定

- Existing assets: no figure required; use a field table for `Reconstruction`.
- Missing: none.

## 4. VGGT 初始重建

- Existing metrics: `report/metrics/vggt_metrics.csv`.
- Existing runs:
  - `data/scene/vggt_raw/runs/20260627_074913_vggt_export`
  - `data/1-human/vggt_raw_hq_unmasked/runs/20260629_083702_vggt_export`
  - `data/2-human/vggt_raw_hq_unmasked/runs/20260629_084616_vggt_export`
  - `data/scene_32/vggt_raw_hq_768/runs/20260629_091545_vggt_export`
- Missing: standalone point-cloud screenshots.
- Report action: use metric tables; optional screenshots can be added later.

## 5. 自实现 Bundle Adjustment

- Existing metrics: `report/metrics/ba_metrics.csv`.
- Existing asset: `report/assets/ba_reprojection_error_bars.png`.
- Missing: none for quantitative report.

## 6. 3DGS

### 自实现 3DGS 与官方 3DGS

- Existing metrics: `report/metrics/custom_3dgs_metrics.csv`.
- Existing comparison metrics: `report/assets/scene_core_3dgs_metrics.png`.
- Missing: a dedicated self-3DGS render sheet.
- Report action: describe self-3DGS as engineering baseline; use metrics and
  leave render screenshot optional.

### Human 场景

- Existing metrics: `report/metrics/official_3dgs_metrics.csv`.
- Existing assets:
  - `report/assets/human_metrics_bars.png`
  - `report/assets/human_render_comparison.png`
- Missing: none.

### Scene 主对比

- Existing assets:
  - `report/assets/scene_core_3dgs_metrics.png`
  - `report/assets/scene_main_render_comparison.png`
- Missing: none.

## 7. VGGT 改进

### I1 视频关键帧选择

- Existing data: selected frame images and partial selected VGGT cache exist.
- Missing:
  - final 512-query selected sparse reconstruction;
  - selected BA;
  - selected sparse official 3DGS metrics.
- Report action: include method and current hardware blocker; use placeholders
  for final metrics.

### I2 Depth-Camera 稠密点过滤

- Existing selected-scene support assets:
  - `report/assets/selected_dense_metric_bars.png`
  - `report/assets/selected_dense_render_comparison.png`
- Existing original-scene partial export:
  - `data/scene/vggt_dense_ablation_scene_200k/runs/20260630_025655_vggt_export`
  - Completed variants: `depth_only`, `reprojection_only`.
- Missing:
  - original-scene point-head VGGT cache;
  - original-scene `pointmap_only`, `disagreement_only`, `filtered_full`;
  - original-scene official 3DGS dense metrics and render sheet.
- Report action: use original-scene partial stats and placeholders for final I2
  table; mention selected-scene result only as implementation sanity check.

## 8. 实时交互渲染展示

- Existing output: official 3DGS run directories contain checkpoints and renders.
- Missing: viewer screenshot.
- Report action: leave a screenshot placeholder.

## 9. 分析和结论

- Existing evidence: BA metrics, 3DGS raw/BA/random-init metrics, human metrics,
  scene_32 negative result.
- Missing: completed original-scene dense ablation and selected sparse final.
- Report action: draw firm conclusions for BA and 3DGS; mark improvement
  conclusions as provisional.

## 10. 未来工作

- Existing content: can be written from known limitations.
- Missing: none.
