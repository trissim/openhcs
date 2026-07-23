# Remaining Public Registry And Export Authority Refactor - 2026-05-18

## Evidence

Refreshed advisor scan still reports:

- 45 manual public API surfaces.
- 36 findings in `openhcs/processing/backends/lib_registry/unified_registry.py`.
- 22 findings in `openhcs/processing/func_registry.py`.
- 24 findings in `openhcs/core/runtime_artifact_queries.py`.
- 11 findings in `openhcs/core/callable_contract.py`.
- Additional `__init__.py` and package export surfaces across `core`,
  `processing`, `materialization`, `interop`, and PyQt services.

Recent completed export checkpoints:

- `public_names_from_objects(...)` added for narrow compatibility export
  surfaces.
- CP `intensity_distribution.py`, `watershed.py`, and `thresholding.py`
  now derive selected `__all__` surfaces from object identity while preserving
  compatibility order/names.

## Problem

Public names, registry surfaces, callable contracts, and artifact query helpers
still have multiple authorities: explicit string lists, local forwarding
helpers, wrapper functions, and parallel registration surfaces.

## Target Shape

- Use `public_names_from_objects`, `declared_public_names`, or explicit
  `ExportSpec`-style records depending on whether the module exports a narrow
  compatibility surface or all local declarations.
- Registry modules should expose one typed projection from registered families
  to public lookup views.
- Runtime artifact query helpers should be grouped under nominal query/request
  objects rather than repeated public bare functions.
- Callable-contract public helpers should either be methods on contract/request
  authorities or explicit public facade functions with tests.

## Phases

1. Finish CP backend `__all__` conversions for modules with narrow
   compatibility surfaces.
2. Add characterization tests that compare selected module `__all__` tuples to
   expected compatibility names before converting each family.
3. Split `unified_registry.py` findings into:
   - public registry projection authority,
   - callable resolution/query helpers,
   - private helper nominalization,
   - real compatibility facades.
4. Refactor `func_registry.py` similarly, preserving public import paths.
5. Refactor `runtime_artifact_queries.py` into typed query/request authorities.
6. Re-run import-surface and registry tests after every checkpoint.

## Verification Gates

```bash
.venv/bin/python - <<'PY'
import openhcs
import openhcs.processing
import openhcs.processing.func_registry
import openhcs.processing.backends.lib_registry.unified_registry
import openhcs.core.runtime_artifact_queries
PY

.venv/bin/python -m pytest \
  tests/unit/test_callable_contract.py \
  tests/unit/test_cellprofiler_library_loading.py -q

timeout 180 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/processing/backends/lib_registry/unified_registry.py \
  openhcs/processing/func_registry.py \
  openhcs/core/runtime_artifact_queries.py \
  openhcs/core/callable_contract.py
```

## Risks

- Public import surfaces are compatibility-sensitive. Always preserve tuple
  names/order or explicitly document and test a migration.
- Some forwarding shells may be public API, not slop. Collapse only when the
  delegate authority is already public and equivalent.
