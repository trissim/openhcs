# CellProfiler Boundary Audit

Date: 2026-07-08

Status: boundary audit and migration record. The current branch implements the
core compiler/provider/generator changes described here and keeps the boundary
guarded by `tests/unit/test_cellprofiler_public_boundary_contracts.py`.

## Non-Negotiable Boundary

CellProfiler support must be expressible as normal OpenHCS public declarations:

```text
PipelineConfig + FunctionStep list
```

That is the complete public declaration surface. A converted `.cppipe` pipeline
and a hand-written CellProfiler/OpenHCS pipeline must compile from the same
kind of objects:

```python
PipelineConfig(...)

pipeline_steps = [
    FunctionStep(func=raw_public_cellprofiler_callable, ...),
    FunctionStep(func=(raw_public_cellprofiler_callable, public_kwargs), ...),
]
```

The following are not valid sources of runtime truth:

- selected `.cppipe` files during compilation,
- UI/import sidecars,
- generated hidden contract modules,
- pre-bound `FunctionStep.invocation_contracts`,
- `cellprofiler_module_callable(...)` in user/generated source,
- hidden compile-time kwargs in the callable kwargs dict,
- new CP-specific `PipelineConfig` groups for module artifact identity,
- per-module behavior tables that mirror `CellProfilerModule` declarations.

The only CellProfiler module authority is the nominal declaration registry:

```text
raw callable
  -> CellProfilerFunctionCatalog.runtime_metadata(callable)
  -> CellProfilerModule.for_module(metadata.module_name)
  -> CellProfilerModule subclass declaration methods
```

The compiler may build transient `ModuleBlock` values because that is the input
shape accepted by the existing CP parser/symbol compiler. Those transient values
are not an authority. The authority is still the module declaration plus the
typed symbol compiler. Do not add a second module-name-indexed behavior map.

The existing `openhcs/interop/cellprofiler/symbol_table.py` is acceptable and
load-bearing. Its module docstring says it is the one boundary where CP string
workspace names become typed OpenHCS artifact contracts
(`symbol_table.py:1-8`). Its contract compiler asks
`CellProfilerModule.for_module(module.name)` and then calls the declaration's
`artifact_contract(...)` (`symbol_table.py:508-525`). Preserve that direction.

## Existing Load-Bearing OpenHCS Patterns

The plan below is anchored to existing OpenHCS mechanisms.

- Generic invocation-contract extension already exists:
  `InvocationContractProviderFactory` is registered through `AutoRegisterMeta`,
  and `PipelineInvocationContractProviderAuthority` iterates the registry
  (`openhcs/core/invocation_artifacts.py:197-245`).

- The compiler already passes that provider into path and artifact planning:
  `compiler.py:523-532`, `path_planner.py:449-469`,
  `path_planner.py:782-818`, and `artifact_planning.py:220-243`.

- Callable runtime config already follows the dtype pattern:
  `CallableRuntimeConfig` is registry-owned (`openhcs/core/config.py:48-108`),
  `DtypeConfig` is one registered instance (`config.py:525-531`), and
  `StepConfigUniverse.runtime_parameter_bindings()` resolves inherited runtime
  parameter bindings (`step_config_universe.py:129-141`).

- Function-pattern compilation already strips config-owned runtime parameters
  before runtime invocation (`function_patterns.py:731-778`). CP compile-time
  artifact identity needs the same generic stripping shape, but as a compiler
  contract plan, not as a CellProfiler runtime patch.

- Source-binding inheritance is already generic:
  `resolve_effective_step_source_bindings(...)` composes pipeline source
  defaults, step-source defaults, and activation by `InputSource.PIPELINE_START`
  (`source_bindings.py:1025-1135`, `path_planner.py:1711-1723`).

- CP module declarations already carry setting bindings, ignored settings,
  processing defaults, group defaults, artifact hooks, and the
  `CellProfilerModule.__registry__` nominal root
  (`module_declarations.py:692-744`).

- CP declarations already have reconstruction and contract hooks:
  `compile_time_setting_records_from_kwargs(...)`,
  `compile_time_setting_records_for_invocation(...)`,
  `compile_time_source_binding_input_setting_records(...)`,
  `compile_time_public_setting_records_from_kwargs(...)`, and
  `artifact_contract(...)`
  (`module_declarations.py:1513-1684`, `module_declarations.py:1999-2114`).

- Generated pipelines already have sparse source-binding and dict-pattern
  ergonomics machinery:
  `GeneratedPipelineConfigDefaults` (`pipeline_generator.py:780-841`),
  grouped-function collapse (`pipeline_generator.py:873-888`), and sparse
  step source-binding emission (`pipeline_generator.py:1116-1143`).

These are the mechanisms to make load-bearing. The fix is not another table.

## Boundary Violations This Branch Guards

### 1. Compiler Reads `.cppipe` As Contract Authority

`cellprofiler_module_settings_invocation_contract_provider_for_session(...)`
previously called `_runtime_contracts_from_selected_cppipe(session)` before
trying public declarations. That path read prepared pipeline state or
regenerated runtime contracts from the selected `.cppipe`.

This violates the public boundary. A `.cppipe` file is an import source and
provenance object. It must not be a compile-time authority after conversion.

Required change:

- Remove `_runtime_contracts_from_selected_cppipe(...)` and
  `_runtime_contracts_from_cppipe_path(...)` from the normal provider flow.
- The provider must derive contracts from public `FunctionStep` snapshots and
  effective `PipelineConfig` only.
- A missing contract after public reconstruction is a module declaration bug and
  must fail at compile time with the module declaration named.

### 2. Compiler Accepts Hidden Step Contracts

`_module_items_from_session(...)` skips public items when
`snapshot.invocation_contracts.contract_for(item.key)` is present
(`compile_time_contracts.py:211-235`). A separate
`CellProfilerStepInvocationContractProvider` also reads those hidden contracts
(`compile_time_contracts.py:98-119`).

That makes two paths:

- clean public source path,
- hidden imported/rebound state path.

Required change:

- CP compilation must ignore `FunctionStep.invocation_contracts`.
- Compile every CP public invocation through the declaration registry.
- Quarantine `FunctionStep.invocation_contracts` as legacy/internal until it can
  be removed or restricted outside the CP path.

### 3. UI And Import Mutate Steps Before Compilation

UI/import code calls `CellProfilerPipelineRuntimeRebinder` before storing or
editing steps (`plate_manager.py:2553-2575`,
`plate_manager_workflows.py:411-460`,
`pipeline_editor_workflows.py:551-564`,
`cellprofiler_pipeline_rebinding.py:30-69`).

The rebinder calls `GeneratedPipelineRuntimeBindings.apply()`, which writes
`FunctionStep.invocation_contracts`
(`runtime/generated_pipeline.py:400-468`, `runtime/generated_pipeline.py:550-598`).

Required change:

- UI code mode and import must store public steps only.
- Import results may carry provenance and diagnostics, but not execution
  contracts that are required for compilation.
- Delete normal calls to `CellProfilerPipelineRuntimeRebinder.rebind(...)`.
- Delete generated import-module side effects that apply runtime bindings.

### 4. Benchmark Can Enter ZMQ With Rebound Steps

`CPPipePipelinePreparationRequest.prepare()` previously returned
`PreparedGeneratedPipeline` with rebound `runtime_pipeline_steps`. The benchmark
submitted those steps to ZMQ.

That explains benchmark/UI divergence: a benchmark can pass with hidden contracts
that a clean code-mode UI submission does not carry.

Required change:

- Official parity benchmarks must submit the same public source/config shape as
  UI and headless code mode.
- The benchmark path must compile before execute, receive a compile artifact,
  then execute that artifact, matching the UI workflow.
- Direct inline execution may remain as a low-level server capability, but it is
  not the official CP parity path.

### 5. Module Artifact Settings Are Too Broadly Exposed As Public Boilerplate

The base declaration previously projected all declared artifact input and output
settings as public compile-only kwargs. That is why generated code could grow boilerplate
such as `select_the_input_image`, `name_the_output_image`, indexed image names,
and tupled repeated CP setting values.

That direction is wrong for canonical OpenHCS flow. Source images come from
source bindings. Main image/object outputs come from OpenHCS artifact flow and
module declaration defaults. CP setting rows are a reconstruction detail unless
the user is intentionally naming a non-canonical artifact.

Required change:

- Keep artifact-affecting semantics on `CellProfilerModule` subclasses.
- Split declaration hooks by semantic role, as classmethods on the module
  declaration, not as external lists:
  - source/main-flow input settings that the compiler infers,
  - canonical output settings that the declaration defaults,
  - explicit non-canonical identity settings that remain public kwargs,
  - runtime algorithm settings that remain normal callable kwargs.
- Remove the base behavior that blindly exposes every artifact input/output
  setting as a public kwarg.
- Module subclasses that need special behavior override declaration hooks.
  Generic compiler code must not know concrete module names.

### 6. RuntimeInvocationOptions Is Not A Sidecar Lane

`RuntimeInvocationOptions` is valid only when the callable contract actually
uses it at runtime. `CalculateMathInvocationOptions` is a real runtime ABI:
`calculate_math(...)` accepts `runtime_invocation_options`, and the runtime uses
it to build measurement rows (`measurement_math.py:389-395`,
`measurement_math.py:860-948`).

Required change:

- Do not use tuple-level options to smuggle CP artifact contracts or source
  image names.
- Do not create CP options DTOs for data that module declarations can infer from
  source bindings, artifact lineage, or declaration defaults.
- For callables with a real runtime options parameter, keep the existing generic
  runtime invocation mechanism. For compile-time-only identity kwargs, use the
  compiler contract plan described below and strip those kwargs before runtime.

### 7. Generated Source Can Still Duplicate Inherited Source Bindings

OpenHCS already resolves source-binding inheritance generically
(`source_bindings.py:1025-1135`). The generator also has sparse emission helpers
(`pipeline_generator.py:1116-1143`). The generated source must rely on that.

Required change:

- Plate-wide image/source aliases go in `PipelineConfig.source_bindings_config`.
- Step `source_bindings` appears only for true overrides, subset selection, or a
  non-source artifact selection that generic lineage cannot express.
- `LazyStepSourceBindingsConfig(enabled=True)` without a real override must not
  be emitted just to repeat pipeline defaults.

### 8. Dict Patterns Must Mean Routing Differences

OpenHCS function-pattern semantics already say a plain callable applies to every
assembled array, while dict patterns route different keys to different callable
chains. The generator already has a collapse check
(`pipeline_generator.py:873-888`).

Required change:

- Use dict patterns only when callable chains, public kwargs, compile-time
  identity kwargs, or intentionally omitted group keys differ.
- When every group key uses the same callable and same kwargs/options, emit a
  plain callable or `(callable, kwargs)`.
- Modules that truly need grouped public emission opt in on their
  `CellProfilerModule` declaration through `force_grouped_public_function_spec`,
  not through a generic hardcoded module list.

## Target Compiler Shape

### Generic Provider Return

The generic provider needs to return the runtime contract and the public kwargs
that were consumed only for compile-time contract reconstruction.

Target API:

```python
@dataclass(frozen=True, slots=True)
class InvocationContractPlan:
    contract: CallableContract
    consumed_kwarg_names: tuple[str, ...] = ()
```

Provider type:

```python
InvocationContractProviderLike = Callable[
    [Any, ArtifactDeclarationStepContext],
    InvocationContractPlan | None,
]
```

Required compiler behavior:

- `artifact_planning.extract_artifact_declarations(...)` uses
  `plan.contract`.
- `function_patterns._compile_invocation(...)` replaces the callable contract
  with `plan.contract`.
- `_compile_invocation(...)` removes `plan.consumed_kwarg_names` from
  `CompiledFunctionInvocation.kwargs` before runtime parameter binding.
- No CP-specific kwarg filtering is added to `FunctionRuntime`.

This generic plan is not a new CP metadata lane. It is the compiler equivalent
of existing runtime-parameter stripping, scoped to compile-time declaration
data.

### CellProfiler Provider Flow

The CP provider must be one registered compiler provider. Its job is to ask the
module declaration registry, not to mirror it.

Required flow:

1. Iterate `CompilationSession.snapshots` in order.
2. For every function-step snapshot, normalize the public function pattern.
3. For every normalized item, resolve the raw callable.
4. Use `CellProfilerFunctionCatalog.runtime_metadata(raw_callable)`.
5. Use `CellProfilerModule.for_module(metadata.module_name)`.
6. Ask that module declaration to reconstruct CP setting rows from:
   - public callable kwargs,
   - effective source bindings,
   - group key,
   - prior typed artifact flow already compiled from preceding public modules,
   - declaration defaults.
7. Build the transient `ModuleBlock`.
8. Feed the ordered transient module sequence to the existing typed symbol
   compiler.
9. For each public invocation, return `InvocationContractPlan` with:
   - a runtime callable contract built from the declaration-derived
     `ModuleArtifactContract`,
   - consumed kwarg names reported by the module declaration.

The provider must not read `.cppipe`, import results, generated semantic
contract modules, or `FunctionStep.invocation_contracts`.

### Module Declaration Responsibilities

Each `CellProfilerModule` subclass owns the CP-specific rules for its module.
Generic provider code asks these classmethods and does not special-case module
names.

Required declaration methods or equivalent refactors:

```python
class CellProfilerModule(...):
    @classmethod
    def compile_time_setting_records_for_invocation(
        cls,
        request: CellProfilerCompileTimeSettingsRequest,
    ) -> tuple[ModuleSetting, ...]:
        ...

    @classmethod
    def compile_time_consumed_kwarg_names(cls) -> tuple[str, ...]:
        ...

    @classmethod
    def compile_time_required_source_input_settings(cls) -> tuple[...]:
        ...

    @classmethod
    def compile_time_canonical_output_setting_records(
        cls,
        request: CellProfilerCompileTimeSettingsRequest,
    ) -> tuple[ModuleSetting, ...]:
        ...

    @classmethod
    def compile_time_explicit_identity_settings(cls) -> tuple[...]:
        ...
```

These hooks are names for the responsibilities; the implementation can reuse and
rename the existing methods where that is cleaner. The important rule is that
the behavior is declared on the module subclass and inherited through the
existing `CellProfilerModule` nominal hierarchy.

Responsibilities:

- Algorithmic CP settings use `setting_bindings` and remain public runtime
  kwargs.
- Source-bound input image settings are inferred from effective source bindings.
- Main-flow input artifacts are inferred from the typed OpenHCS artifact flow
  compiled so far.
- Canonical output image/object settings are supplied by declaration defaults
  or output-preserving-source-stack rules.
- Non-canonical names, retained optional outputs, extra object topology, and
  measurement identities that cannot be inferred remain public kwargs and are
  consumed by the compiler when the backend callable does not accept them.
- Measurement and relationship naming rules live on the declaration or the
  existing nominal feature/artifact helpers, never in a generic module-name map.

### Source Bindings And OpenHCS Main Flow

Canonical source image identity is OpenHCS source-binding state:

```python
PipelineConfig(
    microscope=Microscope.SOURCE_BINDINGS,
    source_bindings_config=LazySourceBindingsConfig(
        bindings=(NamedSourceBinding(alias="OrigStain1", ...), ...)
    ),
)
```

A step that reads from `InputSource.PIPELINE_START` inherits that pipeline
source-binding config unless it declares a true override, subset selection, or
special non-source artifact input. It does not repeat the same bindings at step
scope.

Canonical outputs are OpenHCS artifact flow:

- image transforms use output specs and lineage relations,
- object producers use object-label artifacts,
- measurement producers use measurement artifacts,
- relationship producers use relationship artifacts.

CP names such as "Select the input image" and "Name the output image" are
CellProfiler workspace setting rows. For canonical OpenHCS flow they are
reconstructed by the declaration during compile. They are public kwargs only for
intentional non-canonical identity.

## Generated Source Rules

Generated CP source must look like normal OpenHCS code.

Allowed shape:

```python
FunctionStep(
    func=(identify_primary_objects, {
        "threshold_method": CellProfilerThresholdMethod.OTSU,
        "min_diameter": 3,
        "max_diameter": 15,
    }),
    name="IdentifyPrimaryObjects",
)
```

Allowed when a module truly needs non-canonical public identity:

```python
FunctionStep(
    func=(identify_primary_objects, {
        "name_the_primary_objects_to_be_identified": "Nuclei",
        "threshold_method": CellProfilerThresholdMethod.OTSU,
    }),
    name="IdentifyPrimaryObjects",
)
```

Not allowed:

```python
FunctionStep(func=(cellprofiler_module_callable(...), {...}))
FunctionStep(func=(callable, {"__openhcs_compile_time_kwargs__": ...}))
FunctionStep(func=callable, invocation_contracts=...)
```

Source-binding emission rule:

- pipeline default when the binding is plate-wide,
- omitted step config when inherited exactly,
- step config only for true override/subset/special selection.

Dict-pattern emission rule:

- plain callable when every key uses the same callable and kwargs,
- dict pattern when keys differ in callable, kwargs, explicit identity, or
  omitted groups,
- grouped public emission only through a declaration-owned opt-in.

All generated config objects must be lazy config objects, matching normal
OpenHCS code-mode expectations.

## Migration Phases

### Phase 1: Lock The Wrong Semantics Red

Keep or add tests that fail on the current behavior:

- compiler source does not contain
  `_runtime_contracts_from_selected_cppipe`,
  `_runtime_contracts_from_cppipe_path`, or `selected_pipeline_path` as a
  contract source;
- UI/import code does not call `CellProfilerPipelineRuntimeRebinder` or
  `GeneratedPipelineRuntimeBindings.apply()` for normal execution;
- generated modules do not contain `_openhcs_cp_contract_values`;
- ZMQ-transported source does not contain `FunctionStep.invocation_contracts`,
  `ModuleArtifactContract`, or `cellprofiler_module_callable(...)`;
- a clean public `PipelineConfig + FunctionStep` CP pipeline compiles without
  selected `.cppipe` context.

Current static guard file:

- `tests/unit/test_cellprofiler_public_boundary_contracts.py`

### Phase 2: Add The Generic Compiler Plan

Implement `InvocationContractPlan` in `openhcs/core/invocation_artifacts.py`.
Update:

- `openhcs/core/function_patterns.py`,
- `openhcs/core/pipeline/artifact_planning.py`,
- provider call sites that previously expected `CallableContract | None`.

The plan strips compile-time-only public kwargs before runtime. It does not add
CP-specific filtering.

### Phase 3: Make The CP Public Provider The Only Normal Provider

In `openhcs/interop/cellprofiler/compile_time_contracts.py`:

- remove selected `.cppipe` fallback from the provider;
- remove the hidden-step-contract provider from the CP normal path;
- compile all public CP invocation items;
- use effective source bindings from `resolve_effective_step_source_bindings`;
- return `InvocationContractPlan` values.

The provider builds transient `ModuleBlock` objects only by asking
`CellProfilerModule` declarations.

### Phase 4: Refactor Module Declaration Hooks

In `openhcs/interop/cellprofiler/module_declarations.py` and CP backend module
declaration subclasses:

- replace broad `compile_time_public_setting_names()` behavior with
  declaration-owned semantic categories;
- move module-specific canonical input/output inference into declaration hooks;
- keep algorithmic kwargs in `setting_bindings`;
- keep special measurement/relationship naming on declaration or existing
  nominal artifact/feature helpers;
- remove any need for generic code to test concrete module names.

Do this family by family with tests. Examples:

- illumination calculate/apply,
- alignment,
- identify primary objects,
- filter objects,
- measure image area occupied,
- calculate math,
- track objects,
- tile/overlay/display-like modules.

The family list is an implementation ordering aid, not a registry. The code must
discover behavior from the module declarations.

### Phase 5: Remove UI/Import Runtime Rebinding

In:

- `openhcs/interop/cellprofiler/runtime/generated_pipeline.py`,
- `openhcs/interop/cellprofiler/runtime_pipeline.py`,
- `openhcs/pyqt_gui/widgets/shared/services/cellprofiler_pipeline_rebinding.py`,
- UI workflows that call the rebinder,

remove normal-path mutation of steps. Keep validation/provenance objects only
where they are not execution inputs.

### Phase 6: Make Generated Source Sparse And Native

In `openhcs/interop/cellprofiler/pipeline_generator.py`:

- rely on `GeneratedPipelineConfigDefaults` for common processing/source-binding
  defaults;
- omit inherited step source bindings;
- collapse identical group specs to plain callable specs;
- emit explicit identity kwargs only when the module declaration reports that
  they are non-canonical and required.

### Phase 7: Align Benchmark With UI/Headless Compile Path

In the benchmark adapter:

- build public generated source/config,
- submit compile,
- wait for compile artifact,
- submit execution with that artifact,
- compare request signatures with the UI/headless code-mode path for the same
  pipeline.

Passing direct inline execution is not enough for official parity.

## Compile-Time Failure Rules

The compiler must fail before runtime when public declarations are incomplete.

Required failures:

- a module declaration cannot infer a required source image from source bindings
  or main-flow lineage;
- a module has multiple candidate artifact inputs and no public selector;
- a module needs a non-canonical output/measurement/relationship name and the
  public declaration omits it;
- a generated dict pattern represents no routing difference;
- step source bindings duplicate pipeline defaults instead of inheriting them;
- a CP provider tries to read selected `.cppipe` as contract authority.

Failures name the module declaration and the missing semantic role, not a late
runtime array-shape symptom.

## Completion Gates

The migration is complete only when all gates pass:

- clean generated CP source imports no runtime rebinder, module contract builder,
  or `cellprofiler_module_callable`;
- UI code mode, headless code mode, and official benchmark submit the same
  public pipeline/config shape before ZMQ;
- ExampleColocalization runs from clean public source with no hidden contracts;
- ExampleTrackObjects runs from clean public source with no hidden contracts;
- the official 30 `.cppipe` set passes parity through compile-before-execute;
- native OpenHCS examples still compile and run with unchanged public API;
- no new CP-specific module behavior table exists outside `CellProfilerModule`
  declarations and existing nominal artifact/feature strategy roots.

## Bottom Line

The architecture target is not "store more metadata somewhere else." The target
is:

```text
PipelineConfig + FunctionStep
  -> generic compiler provider registry
  -> CellProfilerModule declaration registry
  -> existing typed CP symbol compiler
  -> runtime callable contract
```

Every CP-specific semantic rule belongs on the authoritative module,
callable/artifact/measurement declaration, or existing nominal strategy root.
Generic compiler code must query those authorities. It must not grow a mirrored
module behavior table.
