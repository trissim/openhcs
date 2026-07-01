# Source Binding Activation Boundary Plan

## Purpose

Make source-binding activation follow the declared config hierarchy:

- `PipelineConfig.source_bindings_config` declares the plate/source universe and init-time source metadata. It is setup, not per-step runtime activation.
- `PipelineConfig.step_source_bindings_config` is an optional inherited default for step source-binding behavior.
- `FunctionStep.source_bindings` is the per-step `LazyStepSourceBindingsConfig`/`StepSourceBindingsConfig` that narrows or overrides bindings and decides whether that step uses source binding.

The current compiler violates this boundary by reviving disabled step source bindings when a module artifact contract mentions matching external aliases. That makes module contracts a second activation authority.

## Current Evidence

- `SourceBindingsConfig` is the pipeline/source declaration in `openhcs/core/source_bindings.py`.
- `StepSourceBindingsConfig` inherits `SourceBindingsConfig` and `Enableable` in `openhcs/core/source_bindings.py`.
- `SourceBindingsConfig` and `StepSourceBindingsConfig` are registered as global/pipeline config types in `openhcs/core/config.py`.
- `FunctionStep` receives `source_bindings: LazyStepSourceBindingsConfig = LazyStepSourceBindingsConfig()` through `AbstractStep`.
- `StepSnapshot.source_bindings` resolves `StepSourceBindingsConfig` from the ObjectState config universe.
- `PipelineGeneratorBuildStage._pipeline_config()` already sets `PipelineConfig.source_bindings_config` from the CellProfiler source schema.
- `CellProfilerSymbolTable._SymbolTableBuilder.source_bindings_for()` currently returns `StepSourceBindingsConfig(bindings=bindings)` for generated steps, so `enabled` is the concrete default `False`.
- `SourceBindingCompilationRequest.from_module_contracts()` currently inspects module contracts and builds `required_aliases`; `compile()` then emits a non-empty `CompiledSourceBindingPlan` even when `config.enabled` is false.
- After deleting `from_module_contracts()` and `required_aliases`, `SourceBindingCompilationRequest` has no remaining independent authority; delete it instead of preserving a one-field wrapper around `CompiledSourceBindingPlan.from_config()`.
- The only production call to `from_module_contracts()` is `PipelineCompiler._compile_source_binding_plan()`.

## Target Model

1. Pipeline init:
   - `PipelineConfig.source_bindings_config` is passed to the microscope handler through the effective config.
   - `SourceBindingsHandler` continues to require a non-empty `SourceBindingsConfig`.
   - This path does not imply that any `FunctionStep` uses source-bound runtime inputs.

2. Step inheritance:
   - A step with no local source bindings can inherit pipeline source metadata and bindings as a resolved `StepSourceBindingsConfig`, but `enabled=False` keeps it inert.
   - `PipelineConfig.step_source_bindings_config=LazyStepSourceBindingsConfig(enabled=True)` can intentionally bulk-enable inherited step source bindings.
   - A step-local `LazyStepSourceBindingsConfig(enabled=False)` must disable that step even if the pipeline default is enabled.

3. Generated CellProfiler steps:
   - The source schema still becomes `PipelineConfig.source_bindings_config`.
   - A generated step that consumes source-bound inputs gets `source_bindings=LazyStepSourceBindingsConfig(bindings=(...), enabled=True)`.
   - The step binding tuple narrows the inherited source universe to the aliases used by that step.
   - The step does not duplicate inherited metadata rules or match plans unless it actually overrides them.

4. Compiler:
   - The compiler freezes the resolved step config mechanically.
   - If `snapshot.source_bindings.enabled` is true, compile the resolved step config into `CompiledSourceBindingPlan`.
   - If `snapshot.source_bindings.enabled` is false, compile an empty `CompiledSourceBindingPlan`.
   - The compiler must not inspect `ModuleArtifactContract.external_input_names()` to decide whether disabled source bindings should run.

## Concrete Edits

### Removal inventory for the redundant activation path

Current `rg -n "SourceBindingCompilationRequest|from_module_contracts|required_aliases" openhcs tests -g '*.py'` output is limited to these live edit sites:

- `openhcs/core/source_bindings.py`
  - Delete the `ModuleArtifactContract` import near the top of the file.
  - Delete the full `SourceBindingCompilationRequest` class block, currently starting at the class definition and ending immediately before `CompiledSourceBindingPlan`.
  - Replace `CompiledSourceBindingPlan.from_config()` so it no longer calls `SourceBindingCompilationRequest(config=config).compile()`.
- `openhcs/core/pipeline/compiler.py`
  - Remove `SourceBindingCompilationRequest` from the source-bindings import list.
  - Replace `_compile_source_binding_plan()` with
    `return CompiledSourceBindingPlan.from_config(source_bindings)`.
  - Remove the `current_plan` parameter from `_compile_source_binding_plan()`.
  - Update the single caller in `_supplement_step_plans()` so it passes only
    `snapshot.source_bindings`.
- `tests/unit/test_cellprofiler_generated_pipeline_execution.py`
  - Remove `SourceBindingCompilationRequest` from imports.
  - Replace the helper call that builds `_CoreExecutionRequest.source_binding_plan` with `CompiledSourceBindingPlan.from_config(step.source_bindings)`.
  - Ensure any fixture expecting a non-empty source-binding plan sets `step.source_bindings.enabled=True`.

Required post-edit `rg` result:

```text
no matches for SourceBindingCompilationRequest
no matches for from_module_contracts
no matches for required_aliases
```

### 1. Delete the redundant compilation request abstraction

File: `openhcs/core/source_bindings.py`

- Delete the entire `SourceBindingCompilationRequest` dataclass.
- Delete `SourceBindingCompilationRequest.required_aliases`.
- Delete `SourceBindingCompilationRequest.from_module_contracts()`.
- Delete the `ModuleArtifactContract` import from this file.
- Move the `enabled is None` guard into `CompiledSourceBindingPlan.from_config()`.
- Change `CompiledSourceBindingPlan.from_config(config)` to:
  - validate `isinstance(config, StepSourceBindingsConfig)` and raise the existing type error if not;
  - raise the existing unresolved-lazy `ValueError` when `config.enabled is None`;
  - return `CompiledSourceBindingPlan.from_enabled_config(config)` when `config.enabled` is true;
  - return `CompiledSourceBindingPlan.empty()` otherwise.

Rationale: `StepSourceBindingsConfig.enabled` is the activation authority. Module artifact contracts declare artifact ABI, not source-binding enablement. Once contract-driven activation is gone, a request object with only `config` is not a real abstraction.

### 2. Make the compiler blind to module contracts for activation

File: `openhcs/core/pipeline/compiler.py`

- Replace `_compile_source_binding_plan()` with a single mechanical call:
  - `return CompiledSourceBindingPlan.from_config(source_bindings)`
- Remove the unused `current_plan` parameter from `_compile_source_binding_plan()`.
- Update the `_supplement_step_plans()` call site to pass only
  `snapshot.source_bindings`.
- Remove the branch that checks `current_plan.compiled_function_pattern`.
- Remove the tuple comprehension over `iter_invocations()`.
- Remove the `SourceBindingCompilationRequest` import.
- Add or keep the `CompiledSourceBindingPlan` import as needed.

Rationale: the compiler consumes the resolved step config. It does not infer enablement from CellProfiler or callable artifact contracts.

### 3. Generate lazy, explicitly enabled step source bindings

File: `openhcs/interop/cellprofiler/symbol_table.py`

- In `_SymbolTableBuilder.source_bindings_for()`, change the non-empty return to include `enabled=True`:
  - `return StepSourceBindingsConfig(bindings=bindings, enabled=True)`
- Keep returning `EMPTY_SOURCE_BINDINGS` for no external symbols.
- Do not copy pipeline metadata rules or match plans into this step config.

Rationale: this symbol-table method only returns a step config when the module has external source symbols. That is the declaration point where generated CellProfiler code knows the step needs source-bound runtime inputs.

### 4. Render generated FunctionStep source bindings as lazy config

Files:

- `openhcs/interop/cellprofiler/symbol_table.py`
- `openhcs/interop/cellprofiler/module_processing_components.py`
- `openhcs/interop/cellprofiler/pipeline_generator.py`

Edits:

- Replace the current generated step literal path:
  - from `source_bindings=StepSourceBindingsConfig(...)`
  - to `source_bindings=LazyStepSourceBindingsConfig(..., enabled=True)`
- Keep `source_bindings_config_literal()` rendering `SourceBindingsConfig(...)` for pipeline config.
- Split the renderer so the type-specific public functions choose constructors:
  - `source_bindings_config_literal(config: SourceBindingsConfig) -> str` renders `SourceBindingsConfig`.
  - `step_source_bindings_literal(config: StepSourceBindingsConfig) -> str` renders `LazyStepSourceBindingsConfig`.
- Have `step_source_bindings_literal()` append `enabled=True` when `config.enabled` is true.
- Do not add an `isinstance()` branch inside a shared renderer to decide whether enabled exists. The caller already knows whether it is rendering a pipeline config or a step config.
- Update `generated_function_step_semantic_argument_lines()` to call the step-specific renderer.
- Update the generated import block to import `LazyStepSourceBindingsConfig` from `openhcs.core.config`.
- After changing the generated step literal, run `rg -n "StepSourceBindingsConfig" openhcs/interop/cellprofiler/pipeline_generator.py`; remove `StepSourceBindingsConfig` from the generated import block when that import is only used for the old generated literal.

Rationale: generated step declarations participate in ObjectState lazy inheritance. Concrete `StepSourceBindingsConfig(bindings=...)` blocks inheritance of `enabled` and currently defaults to disabled.

### 5. Leave pipeline config generation in its current role

File: `openhcs/interop/cellprofiler/pipeline_generator.py`

- Keep `_pipeline_config()` setting `PipelineConfig(source_bindings_config=source_bindings_config)` whenever the source schema has runtime source-binding content.
- Keep `microscope=Microscope.SOURCE_BINDINGS` only when `PipelineImageSchemaSourceBindingsRepresentability` reports no unsupported fields.
- Do not set `PipelineConfig.step_source_bindings_config` just to make generated steps work.
- Do not treat `source_bindings_config` as a runtime enable switch.

Rationale: pipeline source bindings are the plate/source universe. Step runtime activation belongs on the step or inherited step default.

### 6. Update tests to enforce the boundary

Files:

- `tests/unit/test_source_bindings.py`
- `tests/unit/test_compilation_session.py`
- `tests/unit/test_cellprofiler_source_schema.py`
- `tests/unit/test_cellprofiler_symbol_table.py`
- `tests/unit/test_cellprofiler_generated_pipeline_execution.py`
- Runtime adapter tests that construct source-bound steps directly.

Concrete expectations:

- Keep the existing inheritance test where a step inherits pipeline bindings but compiles empty while disabled.
- Replace `test_compiler_freezes_contract_required_source_binding_subset` with two tests:
  - disabled `StepSourceBindingsConfig(bindings=(...))` plus a compiled module contract still compiles empty;
  - enabled `StepSourceBindingsConfig(bindings=(...))` compiles exactly the declared step binding set.
- Update generated-code assertions:
  - assert `"source_bindings=LazyStepSourceBindingsConfig(" in generated.code`;
  - assert `"enabled=True" in generated.code`;
  - assert `"source_bindings=StepSourceBindingsConfig(" not in generated.code`.
- Update direct test fixtures that expect non-empty source binding plans to pass `enabled=True`.
- For runtime adapter tests, only add `enabled=True` to fixtures that assert source-bound candidate resolution, source-bound input loading, source-bound object loading, or non-empty `source_binding_plan` behavior.
- Remove test usage of `SourceBindingCompilationRequest.from_module_contracts()`.

Rationale: tests must prove the compiler no longer has a hidden alias-required activation path.

## AST-Assisted Edit Checklist

Run these inventories before editing and after editing:

```bash
python - <<'PY'
import ast
from pathlib import Path

names = {
    "SourceBindingCompilationRequest",
    "from_module_contracts",
    "required_aliases",
    "source_bindings_literal",
    "StepSourceBindingsConfig",
    "LazyStepSourceBindingsConfig",
}

for root in (Path("openhcs"), Path("tests")):
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else None
                )
                if name in names:
                    print(f"{path}:{node.lineno}: call {name}")
            elif isinstance(node, ast.Attribute) and node.attr in names:
                print(f"{path}:{node.lineno}: attr {node.attr}")
            elif isinstance(node, ast.Name) and node.id in names:
                print(f"{path}:{node.lineno}: name {node.id}")
PY
```

Required post-edit AST result:

- No `from_module_contracts` in production code or tests.
- No `SourceBindingCompilationRequest` in production code or tests.
- No `required_aliases` in `openhcs/core/source_bindings.py`.
- No generated-code renderer path that emits `StepSourceBindingsConfig` for `FunctionStep.source_bindings`.
- `StepSourceBindingsConfig(...)` remains acceptable in core defaults, UI/view DTOs, runtime type declarations, and tests that intentionally exercise concrete resolved configs.
- `LazyStepSourceBindingsConfig(...)` appears in generated pipeline source for step declarations.

Use AST for batch edits only where the transformation is uniform:

- Rename imports and call sites from `source_bindings_literal(...)` to `step_source_bindings_literal(...)`.
- Replace calls to `SourceBindingCompilationRequest.from_module_contracts(...)` with `CompiledSourceBindingPlan.from_config(...)` only in tests that still need to compile an explicitly enabled resolved config.

Do not bulk-add `enabled=True` to every `StepSourceBindingsConfig(...)`. Classify each constructor:

- Add `enabled=True` only when the test or symbol-table fixture expects an active runtime source-binding plan.
- Leave default/resolved/UI/display constructors disabled unless the test is specifically about enabled source binding.

## Dry Runs

### Representable CellProfiler source schema

1. Generator builds `PipelineConfig(microscope=Microscope.SOURCE_BINDINGS, source_bindings_config=...)`.
2. Generated source-consuming steps render `LazyStepSourceBindingsConfig(bindings=(step aliases...), enabled=True)`.
3. ObjectState resolves the step:
   - bindings are the step-local subset;
   - metadata rules and match plan inherit from `PipelineConfig.source_bindings_config` when not overridden;
   - enabled is true.
4. Compiler sees enabled true and freezes the resolved step config into `CompiledSourceBindingPlan`.
5. Runtime consumes only the compiled plan.

### Pipeline source universe but no enabled step

1. Pipeline has `PipelineConfig(source_bindings_config=...)`.
2. A step has default `LazyStepSourceBindingsConfig()`.
3. ObjectState can resolve inherited source-binding payload, but enabled remains false.
4. Compiler returns `CompiledSourceBindingPlan.empty()`.
5. Runtime does not run source-binding resolution for that step.

### Pipeline default bulk-enable

1. Pipeline has `PipelineConfig(step_source_bindings_config=LazyStepSourceBindingsConfig(enabled=True))`.
2. A step with default source bindings inherits enabled true.
3. A step with `LazyStepSourceBindingsConfig(enabled=False)` disables itself.
4. Compiler follows the resolved step config only.

### Unsupported CellProfiler source schema

1. Generator still emits `PipelineConfig(source_bindings_config=...)` for runtime inheritance.
2. Generator does not set `microscope=Microscope.SOURCE_BINDINGS`.
3. Source-schema workspace materialization remains responsible for init-time projection.
4. Enabled source-consuming steps still compile source-binding runtime plans from their lazy step configs.

### Self-contained step source binding without source schema

1. Generator has no `PipelineImageSchema` content, so `_pipeline_config()` returns `None`.
2. A generated step with external source symbols still renders `LazyStepSourceBindingsConfig(bindings=(...), enabled=True)`.
3. ObjectState resolves no inherited metadata rules or match plan.
4. Compiler freezes the explicit step binding because `enabled=True`.
5. Runtime uses the explicit selectors in the step binding; no pipeline source-universe activation is inferred.

## Verification Commands

Use the repo venv:

```bash
source .venv/bin/activate
python -m pytest tests/unit/test_source_bindings.py tests/unit/test_compilation_session.py
python -m pytest tests/unit/test_cellprofiler_source_schema.py tests/unit/test_cellprofiler_symbol_table.py
python -m pytest tests/unit/test_cellprofiler_generated_pipeline_execution.py
python -m pytest tests/unit/test_cellprofiler_runtime_adapter.py
```

Add a quick generated-code probe after tests:

```bash
python - <<'PY'
from pathlib import Path
from openhcs.interop.cellprofiler.pipeline_generator import PipelineGenerator
from tests.unit.test_cellprofiler_symbol_table import _identify_primary, _identify_secondary

generated = PipelineGenerator().generate_from_registry(
    pipeline_name="source_binding_probe",
    source_cppipe=Path("source.cppipe"),
    modules=[_identify_primary(), _identify_secondary()],
)
assert "source_bindings=LazyStepSourceBindingsConfig(" in generated.code
assert "enabled=True" in generated.code
assert "source_bindings=StepSourceBindingsConfig(" not in generated.code
PY
```

## Non-Goals

- Do not change `SourceBindingsHandler` construction.
- Do not force `Microscope.SOURCE_BINDINGS` for unsupported source schemas.
- Do not add fallback behavior for disabled step bindings.
- Do not make module artifact contracts responsible for source-binding enablement.
- Do not add a new registry or mapping for source-binding activation.
