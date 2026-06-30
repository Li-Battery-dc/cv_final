# Report Asset Audit

Previous generated metrics and figures are stale because the sparse VGGT export
used `max_reproj_error=8.0`.

Final report assets must be regenerated after rerunning the pipeline with:

```text
MAX_REPROJ_ERROR=0.0
```

Until then, all quantitative report entries should remain `[PLACEHOLDER]`.
