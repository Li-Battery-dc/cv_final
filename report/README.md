# Report Draft

This directory contains the preliminary LaTeX report for the final project.

Main file:

```bash
report/main.tex
```

Assets are generated from the current experiment outputs under `data/` and stored in:

```bash
report/assets/
```

Suggested build command, if a Chinese LaTeX environment is installed:

```bash
cd report
xelatex main.tex
xelatex main.tex
```

Current status:

- `scene` has complete VGGT raw, custom BA, official 3DGS raw/BA, metrics, and 8-view visual comparisons.
- `1-human` now has a high-density masked-white official 3DGS result with 8 held-out test views.
- `2-human` now has a high-density masked-white official 3DGS result with 8 held-out test views.
- Custom 3DGS is documented as a failed/weak baseline and is not used as the final renderer.
