# CellProfiler Matcher Fanout Refactor Plan

## Goal

Resolve or justify the remaining CellProfiler advisor findings where matcher or
strategy infrastructure has only one external production consumer.

Current findings after checkpoint `d8ba6adf`:

- `GrayToColorInputNameResolver`
- `ModuleFunctionResolutionStrategy`
- `ModuleSettingsBindingStrategy`

The goal is not to delete abstractions mechanically. The goal is to ensure each
public matcher family either earns its surface area through fanout or is moved
inside the one orchestration boundary that owns it.

## Current Evidence

Advisor reports:

- `openhcs/interop/cellprofiler/gray_to_color_settings.py` exposes
  `GrayToColorInputNameResolver`; consumer:
  `GrayToColorContractBuilder.build`
- `openhcs/interop/cellprofiler/module_function_resolution.py` exposes
  `ModuleFunctionResolutionStrategy`; consumer:
  `PipelineGeneratorBuildStage.generate`
- `openhcs/interop/cellprofiler/module_settings_binding.py` exposes
  `ModuleSettingsBindingStrategy`; consumer:
  `PipelineGeneratorCodeEmitter.generate_steps_from_registry`

## Boundary Rule

For each family, choose exactly one of these outcomes:

- keep public and add real second production fanout,
- make it private to the owning orchestration stage,
- merge it into an already-amortized provider abstraction,
- document it as intentionally single-consumer because it is an extension seam
  with tests and expected future consumers.

Do not create wrapper methods solely to satisfy fanout. The advisor has already
shown that helper-only wrappers make the design worse.

## Refactor Sequence

### 1. Build a Fanout Map

Use `rg` and code inspection to classify imports and calls as:

- production runtime/generator consumer,
- test-only consumer,
- public export only,
- docs-only mention,
- generated-code consumer.

Commands:

```bash
rg "GrayToColorInputNameResolver|ModuleFunctionResolutionStrategy|ModuleSettingsBindingStrategy" \
  openhcs tests docs
```

### 2. GrayToColor Resolver Decision

Inspect:

- `openhcs/interop/cellprofiler/gray_to_color_settings.py`
- `openhcs/interop/cellprofiler/module_settings_binding.py`
- `GrayToColorContractBuilder.build`

Likely outcomes:

- If the resolver is only a contract-builder detail, move it inside the
  GrayToColor contract/binding boundary.
- If source bindings or generated code can use the same resolver, route that
  second production path through it and keep it public.

### 3. Module Function Resolution Decision

Inspect:

- `openhcs/interop/cellprofiler/module_function_resolution.py`
- `openhcs/interop/cellprofiler/pipeline_generator.py`
- registry loading and module partition stages

Likely outcomes:

- If resolution is part of generator build orchestration, make the strategy
  family a private collaborator of the build stage.
- If runtime debug, coverage reporting, or conversion diagnostics need the same
  resolution, route them through the same strategy and keep it public.

### 4. Module Settings Binding Decision

Inspect:

- `openhcs/interop/cellprofiler/module_settings_binding.py`
- `PipelineGeneratorCodeEmitter.generate_steps_from_registry`
- coverage summary/report generation
- generated pipeline integration tests

Likely outcomes:

- Keep public only if settings binding is used by both code emission and a
  separate validation/coverage path.
- Otherwise make the binding strategy family owned by the code emitter stage or
  a `CellProfilerModuleBindingProvider`.

### 5. Prefer Provider Consolidation

If two or more of these families are generator-only, consider one
`CellProfilerInvocationProjectionProvider` with subrecords for:

- function resolution,
- settings binding,
- artifact contract projection,
- source binding projection.

This should happen only if it reduces duplicated orchestration. Do not create a
monolithic new hub.

## Verification

For each migrated family:

```bash
.venv/bin/python -m nominal_refactor_advisor \
  openhcs/interop/cellprofiler/gray_to_color_settings.py \
  openhcs/interop/cellprofiler/module_function_resolution.py \
  openhcs/interop/cellprofiler/module_settings_binding.py \
  openhcs/interop/cellprofiler/pipeline_generator.py
```

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/unit/test_cellprofiler_strategy_registries.py \
  tests/unit/test_runner_cellprofiler_compatibility.py \
  tests/unit/test_cppipe_execution_validation.py -q
```

Full:

```bash
.venv/bin/python -m pytest tests/unit -q
```

## Completion Criteria

- Each single-consumer matcher finding is removed or explicitly justified in a
  short note with fanout evidence.
- No new "trivial forwarding wrapper" findings.
- Generated CellProfiler pipeline tests and compatibility tests pass.

## Execution Note

Implemented the fanout decision by narrowing generator/internal matcher
families instead of manufacturing fake second consumers:

- `GrayToColorInputNameResolver` became `_GrayToColorInputNameResolver`;
- `ModuleFunctionResolutionStrategy` became `_ModuleFunctionResolutionStrategy`;
- `ModuleSettingsBindingStrategy` became `_ModuleSettingsBindingStrategy`.

The classes remain nominal and tested, but they are no longer advertised as
public extension surfaces. CP-wide advisor no longer reports matcher-fanout
findings after this narrowing.
