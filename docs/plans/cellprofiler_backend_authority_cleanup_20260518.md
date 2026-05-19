# CellProfiler Backend Authority Cleanup - 2026-05-18

## Full-Scan Evidence

The full scan shows the densest active backend findings in CellProfiler modules:

- `thresholding.py`
- `morphology.py`
- `watershed.py`
- `intensity_distribution.py`
- `grid.py`
- `zernike.py`
- `illumination.py`
- `colocalization.py`
- `secondary.py`
- `granularity.py`
- `tracking.py`
- `alignment.py`

Finding families include:

- repeated field assignment builders;
- repeated threaded parameter families;
- anti-unified compound blocks;
- helper-backed wrapper classes;
- private helper clusters;
- package-authority helper duplication;
- semantic dict bags;
- structural type annotation repetition.

## Current Problem

Many CellProfiler compatibility backends encode module-specific behavior in
local helper clusters and repeated dict/parameter patterns. This makes parity
harder to audit because CP concepts are spread across backend-local helpers
instead of declared as typed semantic authorities.

## Target Shape

Add backend authorities where the domain repeats:

- `CellProfilerMeasurementFeatureSpec`
- `CellProfilerObjectSetRequest`
- `ThresholdOperationSpec`
- `MorphologyOperationSpec`
- `WatershedSegmentationRequest`
- `RadialDistributionRequest`
- `GridGeometryRequest`
- `BackendFunctionPatternSpec`

The goal is not generic abstraction for its own sake. Each authority must map to
a CellProfiler concept or a repeated OpenHCS compatibility concept.

## Phases

1. Run CP parity/compatibility tests as baseline.
2. Start with repeated request records in `watershed.py`,
   `intensity_distribution.py`, and `thresholding.py`.
3. Move morphology and threshold operation families into typed operation specs.
4. Extract feature-name/type aliases for `zernike.py`, `worms.py`, and runtime
   export-related CP measurement modules.
5. Collapse helper-backed wrapper classes only where wrappers do not preserve a
   public CP compatibility surface.
6. Rerun generated-pipeline and Official30 parity after behavior-adjacent
   changes.

## Verification Gates

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_library_loading.py \
  tests/unit/test_cellprofiler_module_execution.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/unit/test_runner_cellprofiler_compatibility.py -q
timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs/processing/backends/cellprofiler
```

## Checkpoints

Checkpoint 5:

- Routed Morph branchpoint/endpoint operations through the public
  `worm_geometry.branchpoints` and `worm_geometry.endpoints` authorities instead
  of maintaining local underscore-prefixed shadows.
- Fixed the CP compatibility gate failure in
  `MeasurementTableAxisQuery.slice` by restoring it as a classmethod constructor
  variant.
- Focused morphology advisor output is clean, and the full CP compatibility
  gate passes.

Checkpoint 6:

- Added `HelperBackedSmoothingPlaneStrategy` as the concrete template method
  for illumination smoothing modes implemented by one helper authority.
- Kept `FitPolynomialSmoothingPlaneStrategy`,
  `GaussianFilterSmoothingPlaneStrategy`, and
  `ConvexHullSmoothingPlaneStrategy` as explicit opaque strategy leaves; each
  leaf implements the abstract helper hook instead of being collapsed into
  sentinel dispatch.
- Fixed Nominal Refactor Advisor separately so abstract hook implementations
  are not misreported as removable trivial forwarding wrappers.
- Focused illumination tests pass, and the helper-backed wrapper advisor
  finding is cleared.

Checkpoint 7:

- Added `RankMedianProfilerPhase` as the single authority for rank-median
  profiler event construction in illumination smoothing.
- Removed repeated elapsed-time/log-field construction from the Numba rank
  median fast paths without changing event names or payload fields.
- Focused illumination median-smoothing tests pass, and advisor no longer
  reports the rank-median profiler log duplication.

Checkpoint 8:

- Moved CP masked grey erosion/dilation semantics into
  `CellProfilerMaskedGreyMorphology`, so exact convex-hull smoothing no longer
  shares hidden module-level helpers across native and Numba paths.
- Collapsed duplicated Fenwick selection control flow behind the shared
  `_fenwick_select_index` kernel while keeping dtype-specific Numba adapters.
- Focused illumination tests pass, and the illumination module is
  advisor-clean.

Checkpoint 9:

- Removed repeated inherited AutoRegister skip-policy boilerplate from
  morphology's `MorphOperationStrategy`, `RepeatModeStrategy`, and
  `OpenLineStructuringElement` roots.
- Verified the metaclass inheritance path with focused morphology registry
  tests; the morphology module is advisor-clean.

Checkpoint 10:

- Moved absorbed-function lookup/runtime metadata behavior behind
  `CellProfilerFunctionCatalog`.
- Preserved legacy module-level helper imports through an
  `AutoRegisterMeta`-owned compatibility export family instead of a string
  dispatch table or bare support-function surface.
- Focused CellProfiler processing/import tests pass, and
  `cellprofiler/__init__.py` is advisor-clean.

Optional after behavior-adjacent changes:

```bash
# Rerun the current official CP parity/Official30 command documented in the
# benchmark workflow before presentation/benchmark claims are refreshed.
```

## Completion Criteria

- CP backend repeated semantic records have typed owners.
- No fake generic bases for unrelated CP predicates.
- CP generated-pipeline and compatibility tests pass.
- Parity-sensitive changes have explicit rerun evidence.
