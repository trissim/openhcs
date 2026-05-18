# CellProfiler FunctionNameVariant Registry Plan

## Goal

Make `FunctionNameVariant` a proper metaclass-backed membership authority
without breaking the current grid module variant behavior.

This addresses the current advisor finding:

- "Semantic inheritance family should have a metaclass membership SSOT"
- evidence root: `openhcs/interop/cellprofiler/grid_settings.py`
- concrete leaves: `DefineGridVariant`, `IdentifyObjectsInGridVariant`

## Current Risk

A prior naive attempt to alter registry key declaration behavior caused
registry behavior to regress. This finding must not be treated as a string
cleanup. It is a registry-membership design change.

The existing registry key declaration refactor plan also warns that simple
constant or metaclass swaps can make advisor output and runtime behavior worse.

## Existing Seam

Relevant reusable infrastructure:

- `metaclass_registry.AutoRegisterMeta`
- `metaclass_registry.RegisteredEnumMeta`
- `RegistryFamily`, `RegistryKeyAttribute`, and generated leaf helpers in
  `openhcs/core/registry_strategies.py`
- existing `FunctionNameVariant` behavior in
  `openhcs/interop/cellprofiler/grid_settings.py`

## Target Shape

`FunctionNameVariant` should own:

- import-time membership,
- stable registry key extraction,
- lookup from CellProfiler module identity,
- the callable/function-name projection used by generated pipelines.

Leaves should own only:

- their module name or key,
- their function name behavior,
- any module-specific parsing hook.

## Refactor Sequence

### 1. Pin Current Behavior

Add or extend tests before changing the metaclass:

- `DefineGrid` resolves to the expected OpenHCS callable name.
- `IdentifyObjectsInGrid` resolves to the expected OpenHCS callable name.
- unsupported grid module names fail loudly.
- registry contains exactly the expected concrete variants.

Suggested focused test file:

- `tests/unit/test_cellprofiler_strategy_registries.py`

### 2. Inspect Current Class Shape

Read:

- `openhcs/interop/cellprofiler/grid_settings.py`
- all imports of `FunctionNameVariant`
- generated pipeline tests that exercise `DefineGrid` and
  `IdentifyObjectsInGrid`

Record whether membership is currently inferred from:

- enum members,
- class traversal,
- explicit tuple/list,
- downstream if/else selection.

### 3. Choose Registry Mechanism

Preferred options, in order:

1. If leaves are behavior-bearing classes, use `AutoRegisterMeta` with a
   canonical `registry_key`.
2. If leaves are metadata-only, collapse them into a typed declaration table and
   materialize classes only if class identity is required downstream.
3. If the current class is an enum-like closed vocabulary, use
   `RegisteredEnumMeta` or a generated enum helper instead of open plugin
   registration.

Do not add a second side registry.

### 4. Migrate One Slice

Change only `FunctionNameVariant` and its direct consumers first.

Expected implementation properties:

- no manual subclass roster remains,
- registry lookup is the single membership source,
- generated pipeline behavior is unchanged,
- advisor finding for `FunctionNameVariant` is removed.

### 5. Generalize Only After Success

If this produces a clean, stable pattern, extract the reusable part into
`openhcs/core/registry_strategies.py` or the upstream `metaclass-registry`
package. Do not generalize before one concrete family is proven.

## Verification

Focused:

```bash
.venv/bin/python -m nominal_refactor_advisor \
  openhcs/interop/cellprofiler/grid_settings.py
```

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_strategy_registries.py \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/unit/test_runner_cellprofiler_compatibility.py -q
```

Full:

```bash
.venv/bin/python -m pytest tests/unit -q
```

## Completion Criteria

- `FunctionNameVariant` advisor finding is gone.
- Registry behavior is explicitly tested.
- No new registry-key string or metadata-only class-family findings appear.
- No change to generated pipeline output behavior except intended simplification.

## Execution Note

Implemented by removing the artificial shared enum root and making
`FunctionNameVariantResolver` the registered module-name authority. The enum
classes now remain closed value vocabularies, while the AutoRegisterMeta family
owns module-name lookup and function-name projection. Focused generated-pipeline
and compatibility tests passed after the change.
