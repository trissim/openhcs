# Public API And Export Surface Authority - 2026-05-18

## Full-Scan Evidence

The full scan reports:

- 54 `Manual public API surfaces should derive from the module authority`
  findings.
- 5 `Manual export surfaces should derive from the authoritative type family`
  findings.
- repeated active findings in `__init__.py` files, backend modules,
  `unified_registry.py`, `func_registry.py`, and `callable_contract.py`.

## Current Problem

Public import/export surfaces are maintained manually across packages. This is
drift-prone because function registries, backend modules, and compatibility
exports can diverge from the actual authority that owns the objects.

## Target Shape

Introduce export authorities:

- `ModuleExportSpec`
- `BackendExportFamily`
- `CellProfilerBackendExportSpec`
- `RegistryExportProjection`
- generated or validated `__all__` where appropriate.

Avoid dynamic magic that makes imports opaque. The target is explicit export
schemas that can be tested.

## Phases

1. Inventory manual public surfaces from the full scan.
2. Pick one low-risk package family and add export characterization tests.
3. Introduce typed export spec records and derive `__all__` or validation checks.
4. Apply to backend CellProfiler export modules only after CP compatibility
   import tests are in place.
5. Add a test that import surfaces match their export specs.

## Verification Gates

```bash
.venv/bin/python -m pytest tests/unit/test_callable_contract.py tests/unit/test_cellprofiler_library_loading.py -q
.venv/bin/python - <<'PY'
import openhcs.processing.backends.cellprofiler
import openhcs.processing.backends.lib_registry.unified_registry
import openhcs.processing.func_registry
PY
timeout 120 .venv/bin/python -m nominal_refactor_advisor openhcs
```

## Checkpoints

Checkpoint 3:

- Replaced manual CP backend `__all__` lists in `projection.py`,
  `skeleton.py`, and `median_filter.py` with `public_names_from_objects(...)`
  while preserving the previous export names and order.
- Named the median-filter padding modes so the repeated `"constant"` and
  `"reflect"` literals are owned by one module authority.
- Focused advisor output for the three touched CP export modules is clean, and
  the CP compatibility gate passes.

Checkpoint 4:

- Replaced manual CP backend `__all__` lists in `classification.py`,
  `object_overlap.py`, and `primary_objects.py` with
  `public_names_from_objects(...)`.
- Added `ClassificationResult.empty(...)` as the empty-result builder authority
  for repeated ClassifyObjects no-data rows.
- Replaced ClassifyObjects threshold enum branching with explicit
  `ClassificationThresholdStrategy` subclasses keyed by
  `ClassificationThresholdMethod`.
- Focused advisor for these three modules is reduced to the deeper Numba
  anti-unified block in `classification.py`, and the CP compatibility gate
  passes.

## Completion Criteria

- Manual export surfaces for selected active packages derive from explicit
  authorities.
- Public imports remain backwards compatible.
- Import/export tests catch future drift.
