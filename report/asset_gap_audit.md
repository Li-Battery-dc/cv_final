# Report Asset Gap Audit

This audit follows `docs/report_outline.md` and marks which report sections are
currently supported by assets and which sections need placeholders.

## 1. 摘要

- Existing data: previous metrics exist on disk, but they are stale because they
  used VGGT track export with `max_reproj_error=8.0`.
- Missing: all final no-pre-filter metrics.
- Report action: keep placeholders until reruns with `max_reproj_error=0.0`
  complete.

## 2. 任务要求与系统设计

- Existing assets: no figure required; use a Mermaid pipeline diagram in the
  report.
- Missing: none.

## 3. 统一重建表示与评价指标

- Existing assets: no figure required; use a field table for `Reconstruction`.
- Report action: also define the reprojection-error metrics used throughout the
  report: RMSE, Median, and P90.
- Missing: none.

## 4. VGGT 初始重建

- Existing metrics: stale; do not use old `report/metrics/vggt_metrics.csv` for
  final claims until regenerated from no-filter runs.
- Missing: new VGGT raw metrics for `scene`, `1-human`, `2-human`, and optional
  `scene_32`.
- Report action: use placeholders and rerun with `max_reproj_error=0.0`.

## 5. 自实现 Bundle Adjustment

- Existing metrics/assets: stale; old BA metrics and charts used pre-filtered
  VGGT observations.
- Missing: regenerated BA metrics and chart from no-filter VGGT inputs.

## 6. 3DGS

### 自实现 3DGS 与官方 3DGS

- Existing metrics: stale; regenerate after no-filter BA.
- Existing comparison metrics: stale.
- Missing: a dedicated self-3DGS render sheet.
- Report action: describe self-3DGS as engineering baseline; use metrics and
  leave render screenshot optional.

### Human 场景

- Existing metrics/assets: stale.
- Missing: rerun human official 3DGS metrics and render sheets.

### Scene 主对比

- Existing assets: stale.
- Missing: rerun scene raw/BA/random-init metrics and render sheets.

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

- Existing selected-scene support assets: stale.
- Existing original-scene partial export: stale for final report organization;
  rerun or explicitly mark as support-only if reused.
- Missing:
  - original-scene point-head VGGT cache;
  - original-scene `pointmap_only`, `disagreement_only`, `filtered_full`;
  - original-scene official 3DGS dense metrics and render sheet.
- Report action: use original-scene partial stats and placeholders for final I2
  table; mention selected-scene result only as implementation sanity check.

## 8. 实时交互渲染展示

- Existing output: old official 3DGS run directories are stale for final claims.
- Missing: new viewer screenshot from no-filter final run.
- Report action: leave a screenshot placeholder.

## 9. 分析和结论

- Existing evidence: stale.
- Missing: all no-filter final metrics.
- Report action: keep conclusions as placeholders until reruns complete.

## 10. 未来工作

- Existing content: can be written from known limitations.
- Missing: none.
