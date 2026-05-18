# Backend Dimensional Dispatch Authority - 2026-05-18

## Full-Scan Evidence

The full scan reports closed-family numeric dispatch in active backend files:

- `openhcs/processing/backends/analysis/dxf_mask_pipeline.py`
- `openhcs/processing/backends/analysis/self_supervised_segmentation_3d.py`
- `openhcs/processing/backends/enhance/focus_torch.py`
- `openhcs/processing/backends/enhance/jax_nlm_processor.py`
- `openhcs/processing/backends/enhance/self_supervised_2d_deconvolution.py`
- `openhcs/processing/backends/enhance/self_supervised_3d_deconvolution.py`

The repeated axis is array/image dimensionality: `ndim`, original shape length,
and related 2D/3D/4D/5D cases.

## Current Problem

Dimensional behavior is encoded as local literal branches. That makes it hard to
know which dimensional cases are supported, which are errors, and which reshape
rules are shared across backends.

## Target Shape

Add reusable dimensional dispatch authorities:

- `ImageDimensionality`
- `DimensionalCase`
- `DimensionalDispatchTable`
- `ArrayShapeProjection`
- backend-specific `DimensionalOperation` records where behavior differs.

Use typed tables for metadata-only cases and nominal strategy classes only when
the case owns significant behavior.

## Phases

1. Characterize each target backend with shape-focused tests.
2. Introduce a small shared dimensionality module under processing/backend
   utilities.
3. Replace low-risk local ndim ladders with typed dispatch tables.
4. Extract reusable reshape/project/restore rules from self-supervised
   deconvolution and segmentation backends.
5. Keep numerical outputs unchanged; compare shapes, dtype, and representative
   values.

## Verification Gates

Focused tests should include existing tests plus new shape characterization:

```bash
.venv/bin/python -m pytest tests/unit/test_*deconvolution* tests/unit/test_*segmentation* tests/unit/test_cellprofiler_library_loading.py -q
timeout 120 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/processing/backends/analysis/dxf_mask_pipeline.py \
  openhcs/processing/backends/analysis/self_supervised_segmentation_3d.py \
  openhcs/processing/backends/enhance/focus_torch.py \
  openhcs/processing/backends/enhance/jax_nlm_processor.py \
  openhcs/processing/backends/enhance/self_supervised_2d_deconvolution.py \
  openhcs/processing/backends/enhance/self_supervised_3d_deconvolution.py
```

## Completion Criteria

- Supported dimensional cases are explicit data or strategy records.
- Local numeric dispatch ladders are removed from target backends.
- Shape behavior and numerical semantics are preserved.

## Progress

### Checkpoint 1 - DXF Mask Pipeline

Implemented:

- Added `DXFMaskStackProjection` as the typed 3D/4D stack projection authority.
- Replaced the local `image_stack.ndim` ladder with the projection.
- Fixed the pre-existing unreachable registration body: valid 3D/4D inputs no
  longer fall through after the dimension branch.
- Expanded compressed inline guards and normalization logic.

Verification:

```bash
.venv/bin/python -m py_compile openhcs/processing/backends/analysis/dxf_mask_pipeline.py
# clean

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/analysis/dxf_mask_pipeline.py
# No refactoring findings.
```

### Checkpoint 2 - Self-Supervised 3D Segmentation

Implemented:

- Added `SegmentationVolumeProjection` as the typed 3D/4D/5D input projection
  to `[B, C, D, H, W]`.
- Moved output mask shape restoration into the projection.
- Expanded compressed inline control flow in decoder initialization, affine
  augmentation, and k-means helper logic.

Verification:

```bash
.venv/bin/python -m py_compile openhcs/processing/backends/analysis/self_supervised_segmentation_3d.py
# clean

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/analysis/self_supervised_segmentation_3d.py
# No refactoring findings.
```

### Checkpoint 3 - Focus Torch

Implemented:

- Added `LaplacianImageProjection` for 2D/3D/4D conv2d layout projection and
  result restoration.
- Added `FocusStackProjection` for the `[Z, H, W]` CUDA stack contract.
- Added `FocusSharpnessMethod` plus `FOCUS_SHARPNESS_METHODS` to replace the
  `method` string ladder.

Verification:

```bash
.venv/bin/python -m py_compile openhcs/processing/backends/enhance/focus_torch.py
# clean

git diff --check
# clean

timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/enhance/focus_torch.py
# No refactoring findings.
```

### Checkpoint 4 - Deconvolution Input Projections

Implemented:

- Added `Deconvolution2DImageProjection` for 2D deconvolution input layout and
  output restoration.
- Added `Deconvolution3DVolumeProjection` for 3D deconvolution input layout and
  output restoration.
- Added `DeconvolutionBlurMode` enums and removed raw string checks from blur
  setup/apply sites.
- Removed dead `_blur_gaussian_conv_torch` from the 3D deconvolution backend.

Verification:

```bash
.venv/bin/python -m py_compile openhcs/processing/backends/enhance/self_supervised_2d_deconvolution.py openhcs/processing/backends/enhance/self_supervised_3d_deconvolution.py
# clean

git diff --check
# clean
```

Remaining:

- Advisor now asks for the blur-mode enum branches to become nominal strategy
  families in both 2D and 3D deconvolution.
