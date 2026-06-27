# Experiment Plan

## Experiment Matrix

All experiments use the same 64 office images and the same held-out split (`val_every=8`).

| ID | Reconstruction | Renderer / Trainer | Purpose |
|---|---|---|---|
| A | VGGT raw | self 3DGS | no-BA baseline |
| B | custom BA | self 3DGS | current main baseline |
| C | PyCOLMAP BA | self 3DGS | stronger BA, same renderer |
| D | custom BA | official 3DGS | same geometry, stronger renderer |
| E | PyCOLMAP BA | official 3DGS | strongest practical baseline in current plan |
| F | random init | self 3DGS | no-geometry baseline |

## Metrics

- BA: RMSE, median, P90, runtime, removed outliers
- 3DGS: PSNR, SSIM, training time, final Gaussian count
- Visuals: held-out validation frames and viewer demo

## Priority

1. A vs B
2. B vs C
3. B vs F
4. D and E when the external baseline is runnable
