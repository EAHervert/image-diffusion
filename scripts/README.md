# scripts/

Diagnostic entry points. Library code lives in `src/image_diffusion/`.
Run from the repo root with the package installed: `pip install -e .`

| Script | Purpose |
| --- | --- |
| `download_data.py` | Fetch, checksum, and safely extract ImageNette-320. |
| `verify_data.py` | Read-only structure / count / size / decode checks. |
| `inspect_transforms.py` | Measure crop behavior; render `docs/assets/transform-check.png`. |
| `verify_splits.py` | Five-pass gate on the deterministic val/test carve. |
| `preview_data.py` | Render `docs/assets/dataset-overview.png`. |
| `train.py` | Training entry point. |

## AI assistance

The data pipeline here — acquisition, splits, augmentation, verification,
and tests — was built with substantial assistance from Anthropic's Claude
Opus 5. Individual files carry a header noting this.

**Mine.** The decisions, the evidence, and the verification.
`RandomResizedCrop` is rejected because `inspect_transforms.py` measured a
7.81% upsample rate at torchvision defaults, and a 60.5% deterministic
fallback rate when tightened to avoid it. Normalization maps to [-1, 1] to
match the N(0, I) endpoint the flow objective interpolates against. The
val/test carve is stratified and seed-fixed so splits reproduce from the
tarball alone.

`verify_data.py`, `verify_splits.py`, `inspect_transforms.py`, and
`tests/test_data.py` exist to be re-run, and exit nonzero on failure.

**Not mine.** Implementation surface: argparse plumbing, tar-extraction
hardening, percentile helpers, grid compositing. Reviewed line by line, not
written from scratch. Later steps narrow the method to API lookup only,
nothing pasted.
