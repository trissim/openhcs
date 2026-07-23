# CellProfiler Runtime Compatibility Deletion And Parity Rerun Plan

## Goal

Delete remaining CellProfiler compatibility paths only when they are proven to be
legacy migration support rather than current product runtime authority, then
rerun generated-pipeline and official parity gates.

Generated `.cppipe` output no longer needs the old generated-runtime boilerplate
target. Current risk is accidental deletion of load-bearing runtime/import
semantics while chasing compatibility-looking names such as `legacy`, `sidecar`,
or `CellProfilerModuleExecutor`.

## Verified Current State

Generated boilerplate removal is complete for the main target:

- generated source does not emit `CELLPROFILER_MODULE_CONTRACTS`
- generated source does not emit `ModuleArtifactContract(...)` literals
- generated source does not emit per-module `CellProfilerModuleExecutor`
  globals
- generated source does not directly call `cellprofiler_module_callable(...)`
- generated source uses normal backend callables and `FunctionStep`
  declarations, with product runtime/import code applying wrappers and
  contract sidecars

Current compatibility grep shows several categories:

### Current Runtime Authorities, Not Deletion Targets Yet

- `CellProfilerModuleExecutor`
  - product runtime executor in `openhcs/interop/cellprofiler/runtime/module_execution.py`
  - directly tested by runtime adapter/module execution tests
- `cellprofiler_module_callable`
  - product-owned binding factory used by generated runtime step binding
- `attach_callable_contract_metadata`
  - product metadata attachment detail used by runtime callable binding
- `GeneratedPipelineContractSidecar`
  - versioned JSON sidecar for generated module contracts

### True Legacy/Compatibility Candidates

- broad package re-export surfaces in `openhcs/interop/cellprofiler/__init__.py`
  and `openhcs/interop/cellprofiler/runtime/__init__.py`
- `pipeline_generator.py` legacy registry loader path
- parser/source-schema legacy settings support:
  - legacy indented/unindented cppipe parser modes
  - legacy escaped match metadata
  - legacy `LoadImages` source type checks
  - legacy original-color aliases
- module-settings legacy upgrade paths:
  - legacy threshold log-transform defaults
  - legacy Align/Tile bindings
  - legacy MeasureColocalization accuracy choices
- benchmark compatibility cache keys and legacy source-tree digests

### Advisor Findings In CP Area

Fresh advisor scan over `openhcs/interop/cellprofiler` reports broad remaining
cleanup opportunities:

- metadata-only setting binding and ignore classes in
  `module_settings_binding.py`
- `ExpandShrinkOperationModeBinding` declaration family
- function-name/grid variant registry opportunity
- manual `__all__` surfaces
- private helper shadowing public authority:
  `_filter_objects_child_count_object_names` vs
  `filter_objects_child_count_object_names`
- threshold binding repeated keyword bundles
- typed aliases for worm/untangle kwargs
- display plot role-prefixed subrecords
- `product_record` usage in `module_roles.py`
- optional/effect carrier opportunities in measurement/worm parsing

These are not all compatibility deletion tasks; most are CP settings/runtime
architecture cleanup tasks. Do not conflate them with deleting migration
support.

## Target Architecture

- Current generated modules stay declarative.
- Product runtime owns execution wrappers, artifact contracts, adapter
  construction, and sidecar hydration.
- Versioned generated sidecars stay fail-loud and tested.
- Legacy import/parser/settings branches are only retained when tests prove they
  are needed for real historical `.cppipe` inputs.
- Public exports derive from module authority or are explicitly documented as
  compatibility surface.
- Official parity is rerun after behavior-affecting deletions.

## Non-Goals

- Do not delete `CellProfilerModuleExecutor` until there is a replacement
  product runtime executor.
- Do not delete `cellprofiler_module_callable` until runtime step binding no
  longer needs it.
- Do not delete generated contract sidecars; they are current architecture.
- Do not delete parser/source-schema legacy support unless replacement fixtures
  prove it is unused.
- Do not run official30 for docs-only or import-surface-only changes.

## Implementation Passes

### Pass 1: Compatibility Inventory Table

Create an inventory table in this plan or a follow-up doc with columns:

- symbol/path
- current callers
- category:
  - current runtime authority
  - public compatibility export
  - legacy parser/import support
  - benchmark-only compatibility
  - test-only historical fixture
- deletion precondition
- required test gate

Commands:

```bash
rg "CELLPROFILER_MODULE_CONTRACTS|ModuleArtifactContract\\(|CellProfilerModuleExecutor|cellprofiler_module_callable|attach_callable_contract_metadata|legacy|compat|sidecar" \
  openhcs/interop/cellprofiler tests docs/plans
```

### Pass 2: Public Export Surface Cleanup

1. Audit `__all__` findings.
2. Derive public names from module authority where safe.
3. Keep explicit exports only for stable public API or compatibility.
4. Add import tests before deleting exports.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_cellprofiler_interop_import_records.py -q
.venv/bin/python -m pytest tests/unit/test_cellprofiler_strategy_registries.py -q
```

### Pass 3: Settings Declaration Tables

1. Convert metadata-only setting ignore/binding leaves into typed declarations.
2. Prioritize:
   - `ModuleUnmappedSettingIgnore`
   - declarative module settings binding leaves
   - structuring-element module binding leaves
   - expand/shrink mode bindings
3. Preserve registry lookup semantics exactly.

Verification:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_settings_binder.py \
  tests/unit/test_cellprofiler_strategy_registries.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  -q
```

### Pass 4: Public Authority Deduplication

1. Replace private helper duplicates with public authorities, starting with
   `_filter_objects_child_count_object_names`.
2. Convert `product_record` runtime schemas to explicit dataclasses where
   advisor flags them.
3. Add semantic aliases/subrecords for worm and display plot record shapes.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_cellprofiler_module_execution.py -q
```

### Pass 5: Legacy Parser/Source-Schema Audit

1. List every legacy parser/source-schema branch.
2. Match each branch to a test fixture.
3. Delete only branches with no fixture and no real `.cppipe` corpus need.
4. Add fixtures before retaining any branch that currently lacks coverage.

Verification:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cppipe_parser.py \
  tests/unit/test_cellprofiler_source_schema.py \
  -q
```

### Pass 6: Generated Pipeline Gate

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_symbol_table.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/integration/test_cellprofiler_generated_pipeline.py \
  -q
```

If integration is slow or environment-sensitive, run focused unit gates first
but do not claim compatibility deletion complete until integration has passed or
been explicitly deferred with a reason.

### Pass 7: Official30 Parity Gate

Only after behavior-affecting CP runtime/planner/import changes:

1. Use cached CP references where available.
2. Rerun missing/affected cases first.
3. Then run all 30 if parity-risk code changed.
4. Persist summary CSV, parity status, skip reasons, and figure/table updates.

Required report fields:

- pipelines passed parity
- pipelines failed parity
- speedup status if runtime behavior changed
- skipped cases and why
- CP reference source/cache path

### Pass 8: Full Unit Gate

```bash
.venv/bin/python -m pytest tests/unit -q
```

## Completion Criteria

- Every deleted compatibility path has a test proving current behavior.
- Remaining compatibility paths are classified and justified.
- Generated-pipeline unit/integration gates pass.
- Official30 parity is rerun after behavior-changing CP runtime/planner changes.
- No compatibility shim is added without a deletion criterion.
