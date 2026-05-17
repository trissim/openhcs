# CellProfiler Runtime Compatibility Deletion and Parity Rerun Plan

## Goal

Remove legacy CellProfiler runtime compatibility paths only after proving they
are unused by current generated pipelines and preserving parity on the benchmark
corpus.

Generated pipeline boilerplate has been removed from the current target shape.
The remaining risk is old compatibility machinery retained for migration. That
machinery should be deleted deliberately, with generated-pipeline tests and
official parity gates after each deletion batch.

## Current Evidence

Current generated-source tests already assert:

- no `CELLPROFILER_MODULE_CONTRACTS` in generated source
- no generated `ModuleArtifactContract(...)` literals
- no generated `CellProfilerModuleExecutor`
- no generated direct `cellprofiler_module_callable(...)` calls
- no generated per-artifact decorator use

Remaining compatibility areas to inventory:

- `openhcs/interop/cellprofiler/runtime/generated_pipeline.py`
- `openhcs/interop/cellprofiler/runtime/module_execution.py`
- `openhcs/interop/cellprofiler/runtime/__init__.py`
- any benchmark/import code still accepting old generated modules
- contract sidecar codecs and registry hydration paths

## Target Shape

- Current generated modules remain declarative.
- Runtime wrappers, contract registry, adapter preparation, and materialization
  sidecars remain product-owned.
- Old generated-boilerplate import shims are deleted once tests prove no current
  path needs them.
- Sidecar schema/version checks stay fail-loud.

## Non-Goals

- Do not remove versioned sidecar support used by current generated modules.
- Do not remove public runtime exports until import tests prove they are unused
  or callers have migrated.
- Do not run expensive official30 after docs-only or pure GUI changes.
- Do not chase advisor cleanliness over compatibility semantics.

## Implementation Sequence

### Stage 1: Compatibility Inventory

1. Search for old generated symbols:
   - `CELLPROFILER_MODULE_CONTRACTS`
   - `attach_callable_contract_metadata`
   - `cellprofiler_module_callable`
   - `CellProfilerModuleExecutor`
   - artifact decorator emission
2. Classify each remaining hit as:
   - current product runtime authority
   - current generated source test
   - legacy compatibility shim
   - historical documentation
3. Write the inventory into this plan before deleting code.

### Stage 2: Delete One Compatibility Path

For each deletion:

1. Add or update a test proving current generated pipelines do not need the path.
2. Delete the path.
3. Run focused generated-pipeline tests.
4. Commit if clean.

### Stage 3: Generated Pipeline Parity Gate

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_symbol_table.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/integration/test_cellprofiler_generated_pipeline.py \
  -q --tb=short --disable-warnings
```

If integration tests are too slow, run the failing/affected subset first, then
the full integration gate before claiming completion.

### Stage 4: Official30 Cached Parity Rerun

Use the current benchmark manifest and cached CP references where available.
Prefer rerunning only affected/missing cases until all generated-pipeline tests
are green.

Required outputs:

- summary CSV
- parity status
- speedup status if runtime behavior changed
- list of skipped cases and skip reasons

### Stage 5: Final Full Unit Gate

```bash
.venv/bin/python -m pytest tests/unit -q --tb=short --disable-warnings
```

## Completion Criteria

- Every deleted compatibility path has a test proving current generated pipeline
  behavior.
- Generated-pipeline unit/integration tests pass.
- Official30 parity is rerun for affected runtime changes.
- Remaining compatibility exports are either current runtime authority or
  explicitly documented migration support.

