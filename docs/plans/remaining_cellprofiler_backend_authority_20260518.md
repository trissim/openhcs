# Remaining CellProfiler Backend Authority Refactor - 2026-05-18

## Evidence

Highest active CP files in the refreshed advisor scan:

- `cellprofiler/morphology.py`: 46 findings.
- `cellprofiler/thresholding.py`: 39 findings.
- `cellprofiler/grid.py`: 26 findings.
- `cellprofiler/zernike.py`: 24 findings.
- `cellprofiler/illumination.py`: 23 findings.
- `cellprofiler/watershed.py`: 23 findings.
- `cellprofiler/colocalization.py`: 22 findings.
- `cellprofiler/granularity.py`: 19 findings.
- `cellprofiler/secondary.py`, `shape.py`, `smoothing.py`, `intensity.py`,
  `texture.py`, `tracking.py`, `alignment.py`, `object_filtering.py`, `edge.py`,
  and `neighbors.py` remain active secondary targets.

Recent completed checkpoints:

- Threshold robust-center/profiler/export cleanup.
- Radial distribution constructors, request authority, profiler, and exports.
- Watershed profiler/request authority and exports.

## Problem

The CP backend layer still contains many local helper clusters, enum strategy
ladders, repeated field builders, and low-level algorithm copies. Some are real
architecture debt; some are compatibility wrappers or Numba-compatible kernel
boundaries that need careful classification.

## Target Shape

- Explicit nominal strategy families for CP behavior axes that currently use
  enum/string ladders.
- Request/context dataclasses for repeated semantic parameter bundles.
- Builder/classmethod authority for repeated output rows and measurement arrays.
- Low-level Numba kernels stay procedural, but their Python-side request
  preparation and result projection should be nominal.
- Public compatibility wrappers stay until a tested replacement surface exists.

## Priority Sequence

1. `thresholding.py`
   - Classify robust-background nominal subclass family as intended registry
     identity unless registry consumption changes.
   - Decide whether `cellprofiler_apply_threshold` is a public compatibility
     wrapper or should move callers to `ThresholdApplicationRequest.apply`.
   - Delete or explicitly wire `_unit_interval_quantized_codes`.
   - Extract histogram/bin-scan repeated algorithms only if a typed Numba-safe
     helper keeps performance and readability.
2. `watershed.py`
   - Extract distance-initial watershed phase profiling without inventing a fake
     loop over unrelated operations.
   - Address heap push/pop and legacy raveled helper context records if Numba
     signatures allow it cleanly.
   - Decide whether repeated `strategy_label` registry-key literals need an
     AutoRegister/advisor package fix instead of local constants.
3. `morphology.py`
   - Start with top advisor findings, likely helper nominalization and repeated
     parameter/context families.
   - Add focused characterization tests before touching behavior.
4. `grid.py`, `zernike.py`, `illumination.py`, `colocalization.py`,
   `granularity.py`
   - Work file by file, taking only high-confidence advisor findings with
     behavior-preserving tests.

## Verification Gates

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_library_loading.py \
  tests/unit/test_cellprofiler_module_execution.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/unit/test_runner_cellprofiler_compatibility.py -q

timeout 180 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/processing/backends/cellprofiler/thresholding.py \
  openhcs/processing/backends/cellprofiler/watershed.py \
  openhcs/processing/backends/cellprofiler/morphology.py
```

Run narrower CP module tests when the touched module has focused coverage.

## Deferral Rules

- Do not collapse explicit registered CP strategy subclasses into dynamic
  materialization.
- Do not force phase logging into declarative invocation tables when the calls
  are interleaved with distinct algorithmic work; extract a real phase object or
  leave it documented.
- Numba kernels may retain positional signatures if request objects would
  degrade compilation or performance; wrap at the Python boundary instead.
