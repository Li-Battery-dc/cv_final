# Training-Free VGGT Improvement Method

## Goal

The assignment requires an improvement over the original VGGT-based
reconstruction pipeline. Since VGGT is used as a frozen pretrained model, the
improvement is training-free and focuses on two controllable parts of the
pipeline:

1. selecting better video frames before final VGGT inference;
2. constructing a cleaner Gaussian initialization point cloud from VGGT depth,
   camera, and point-map predictions.

The final improved pipeline is:

```text
video
  -> quality-aware candidate frame extraction
  -> initial VGGT pass for feature and pose reliability
  -> diversity-aware keyframe selection and gap filling
  -> final VGGT rerun
  -> depth-camera point construction
  -> point-map disagreement filtering
  -> neighbor-view reprojection voting
  -> 3DGS initialization and evaluation
```

## Notation

Let the decoded video frames be:

```text
I_1, I_2, ..., I_N
```

The candidate frame set is:

```text
C = {I_c1, I_c2, ..., I_cM}, M = 192
```

The final selected keyframe set is:

```text
S = {I_s1, I_s2, ..., I_sK}, K = 64
```

For each selected frame `i`, final VGGT predicts:

```text
K_i      camera intrinsic matrix
T_i      camera-from-world extrinsic matrix [R_i | t_i]
D_i(u)   depth at pixel u = (x, y)
Q_i(u)   direct VGGT point-map world point at pixel u
c_i(u)   VGGT confidence
```

The camera center is:

```text
C_i = -R_i^T t_i
```

## Video Frame Selection

### Image Quality Score

For each decoded video frame, compute three lightweight quality signals:

```text
b_i = Var(Laplacian(gray(I_i)))        blur/sharpness score
e_i = 1 - ratio(clipped dark/bright)   exposure score
t_i = mean(||grad(gray(I_i))||)        texture score
```

The scores are converted to scene-relative ranks:

```text
rank(x_i) in [0, 1]
```

No fixed global threshold is used. The quality score is:

```text
q_i = 0.35 rank(b_i) + 0.30 rank(e_i) + 0.25 rank(t_i) + 0.10 rank(d_i)
```

where `d_i` is a duplicate-removal rank derived from neighboring low-resolution
RGB cosine similarity. Frames almost identical to adjacent frames receive lower
`d_i`.

Candidate frames are selected by dividing the video timeline into `M` temporal
bins and choosing the highest `q_i` frame in each bin. This preserves temporal
coverage while removing the worst blurred, over-exposed, under-exposed, or
textureless frames.

### VGGT Feature Centrality

Run a scout VGGT pass on the 192 candidate frames. From the final cached VGGT
aggregator layer, extract patch tokens and mean-pool them per frame:

```text
f_i = normalize(mean_pool(PatchTokens_i))
```

For each candidate frame, define VGGT feature centrality as the average top-k
cosine similarity to other frames:

```text
a_i = (1 / k) sum_{j in TopK(i)} cos(f_i, f_j)
```

where `TopK(i)` are the `k = 8` most similar frames to frame `i`.

Interpretation:

- high `a_i`: frame is consistent with many other views;
- low `a_i`: possible outlier, poor overlap, or unstable VGGT representation.

### Pose Smoothness Score

Because the input is a video, adjacent reliable frames should have a smooth
camera trajectory. For adjacent candidate frames:

```text
delta_t(i) = ||C_{i+1} - C_i||
delta_R(i) = angle(R_{i+1} R_i^T)
g_i = delta_t(i) + delta_R(i)
```

Each frame receives a local pose-jump penalty from neighboring edges:

```text
p_i = max(g_{i-1}, g_i)
```

The penalty is converted to a smoothness rank:

```text
s_i = 1 - rank(p_i)
```

This avoids fixed thresholds such as "remove if rotation > X degrees"; only
relative abnormal jumps inside the current scene are downweighted.

### Diversity-Aware Keyframe Selection

The per-frame reliability score is:

```text
r_i = 0.55 rank(a_i) + 0.25 s_i + 0.20 rank(q_i)
```

Keyframes are selected greedily. Given already selected frame set `A`, choose:

```text
i* = argmax_i [ r_i - lambda max_{j in A} cos(f_i, f_j) + beta temporal_gap(i, A) ]
```

Default:

```text
lambda = 0.35
beta = 0.10
```

The diversity penalty prevents selecting many high-confidence but nearly
identical views. The temporal bonus encourages coverage across the whole video.

### Camera Trajectory Gap Filling

For candidate-frame edges with unusually large pose jump, use:

```text
g_i >= percentile(g, 90)
```

Instead of simply deleting the unreliable interval, the algorithm searches the
candidate pool between the two endpoints and inserts a high-reliability
intermediate frame near the temporal midpoint:

```text
i_gap = argmax_i [ r_i - temporal_distance_to_midpoint(i) ]
```

This is useful for video reconstruction because a large pose jump may indicate
that the initial frame set skipped an important transition.

## Geometry Filtering

### Depth-Camera Unprojection

The main dense point source is not the direct point-map head. For each pixel
`u = (x, y)` in frame `i`, first unproject depth to camera coordinates:

```text
X_cam(u) =
[
  (x - cx_i) D_i(u) / fx_i,
  (y - cy_i) D_i(u) / fy_i,
  D_i(u)
]^T
```

Then transform it to world coordinates:

```text
P_depth_i(u) = R_i^T (X_cam(u) - t_i)
```

This uses VGGT camera and depth outputs as the primary representation.

### Point Map as Consistency Checker

VGGT also predicts a direct world point map:

```text
P_point_i(u) = Q_i(u)
```

The point map is not used as the main output. It is used to check whether the
depth-camera point is geometrically stable.

Define relative disagreement:

```text
E_dp(i, u) =
  ||P_depth_i(u) - P_point_i(u)||_2
  / (||P_depth_i(u)||_2 + epsilon)
```

Use an adaptive scene percentile threshold:

```text
tau_dp = percentile({E_dp(i, u)}, p_dp)
```

Default:

```text
p_dp = 70
```

Keep a point only if:

```text
E_dp(i, u) <= tau_dp
```

This avoids hard-coded VGGT confidence thresholds, which may not transfer across
different videos or lighting conditions.

### Multi-View Reprojection Consistency

For a depth-camera point from frame `i`, project it into neighboring frames
`j in {i - 2, i - 1, i + 1, i + 2}`:

```text
u_j = project(K_j, T_j, P_depth_i(u))
```

Let the projected camera-space depth be:

```text
z_proj_j = depth_of(T_j P_depth_i(u))
```

Sample the VGGT depth map in frame `j` at the projected pixel:

```text
D_j(u_j)
```

The relative reprojection depth error is:

```text
E_mv(i, u, j) =
  |z_proj_j - D_j(u_j)|
  / (|D_j(u_j)| + epsilon)
```

Compute an adaptive threshold:

```text
tau_mv = percentile({E_mv(i, u, j)}, p_mv)
```

Default:

```text
p_mv = 70
```

A point receives one vote from neighbor `j` if:

```text
E_mv(i, u, j) <= tau_mv
```

The final keep rule is:

```text
keep(i, u) =
  [E_dp(i, u) <= tau_dp]
  and
  [votes(i, u) >= m]
```

Default:

```text
m = 1
```

The final point cloud is sampled to at most 200,000 points, weighted by VGGT
confidence only as a relative sampling weight:

```text
Pr(select P_i) proportional to max(c_i, epsilon)
```

Confidence is not used as a fixed hard threshold.

## Compared Variants

### Frame Selection Variants

| Variant | Definition |
|---|---|
| Uniform | existing 64 uniformly sampled frames |
| Quality only | candidate selection from blur/exposure/texture/duplicate scores |
| VGGT centrality only | select frames by `a_i` |
| Reliability + diversity | select by `r_i` and feature diversity |
| Full frame selection | quality prefilter + centrality + pose smoothness + diversity + gap filling |

### Geometry Variants

| Variant | Point construction | Filtering |
|---|---|---|
| Sparse track | VGGT tracker points | visibility and reprojection filters |
| Depth only | `P_depth` | no point-map consistency |
| Point map only | `P_point` | no depth-camera check |
| Disagreement only | `P_depth` | `E_dp <= tau_dp` |
| Reprojection only | `P_depth` | `votes >= m` |
| Full geometry filter | `P_depth` | disagreement + reprojection voting |

In the current codebase, only the `Full geometry filter` path is exported as a
ready-to-train dense reconstruction. `Depth only` and `Point map only` remain
target ablations, but do not yet have standalone dense export commands.

### Final Method

The final method combines:

```text
selected frames + final VGGT rerun + filtered depth-camera point cloud
```

It should be compared against:

```text
uniform frames + raw VGGT sparse tracks
uniform frames + custom BA sparse tracks
selected frames + sparse tracks
selected frames + filtered dense depth points
```

## Report-Ready Claims to Verify

These statements should only be used in the final report after the corresponding
experiments are run:

1. Frame selection improves reconstruction if selected frames reduce VGGT/BA
   reprojection RMSE or improve 3DGS validation PSNR/SSIM over uniform frames.
2. Depth-camera initialization is better than point-map initialization if
   depth-only 3DGS has fewer floating artifacts or higher validation metrics
   than pointmap-only 3DGS.
3. Geometry filtering is effective if it removes a large fraction of dense
   inconsistent points while preserving or improving 3DGS metrics.
4. The full method is successful if selected frames plus filtered dense
   initialization outperform the uniform-frame raw/BA baselines under the same
   held-out split.

## Implementation Artifacts

Frame selection:

- `src/improvement/video_select.py`
- `data/scene_selected/candidates`
- `data/scene_selected/images`
- `data/scene_selected/frame_selection_candidates.json`
- `data/scene_selected/frame_selection_summary.json`

Geometry filtering:

- `src/improvement/geometry_filter.py`
- `reconstruction_dense_filtered.npz`
- `points3d_dense_filtered.ply`
- `geometry_filter_stats.json`
- standalone `pointmap-only` dense export: not implemented yet
- standalone `depth-only dense` export: not implemented yet

VGGT integration:

- `src/vggt_export.py --enable_point_head`
- `src/vggt_export.py --save_dense_filtered_reconstruction`
