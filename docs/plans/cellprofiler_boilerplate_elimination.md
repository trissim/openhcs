# CellProfiler Pipeline Boilerplate Elimination Plan

## Goal

Generated CellProfiler pipelines should be ordinary OpenHCS pipeline declarations. They should not contain generated runtime infrastructure.

The generated file should contain only:

- `FunctionStep` declarations.
- CellProfiler module settings as the step function kwargs.
- `source_bindings` when the module consumes images from the plate/file system.
- `processing_config` and any other normal OpenHCS step config dataclasses.
- Optional comments or provenance that help humans inspect the generated pipeline.

The generated file should not contain:

- `require_cellprofiler_function(...)` assignments for every module instance.
- `attach_callable_contract_metadata(...)` calls.
- `CELLPROFILER_MODULE_CONTRACTS`.
- `ModuleArtifactContract(...)` literals.
- `ArtifactSpec`, `ArtifactKind`, or materialization policy imports.
- Per-module `CellProfilerModuleExecutor(...)` globals.
- Per-module runtime wrapper functions.
- Per-module `@artifact_inputs`, `@artifact_outputs`, or `@runtime_adapter` glue.

The semantic rule is simple: a generated `.cppipe` pipeline declares the resolved OpenHCS callable and the module settings. OpenHCS derives the artifact contract, runtime wrapper, runtime adapter, prepare hook, and materialization behavior from that invocation.

## Target Generated Shape

The preferred generated form is step-only:

```python
from openhcs.core.steps import FunctionStep
from openhcs.processing.backends.cellprofiler import (
    correct_illumination_calculate,
    define_grid,
    identify_primary_objects,
)

pipeline_steps = [
    FunctionStep(
        func=(
            correct_illumination_calculate,
            {
                "intensity_choice": "regular",
                "block_size": 60,
                "rescale_option": "yes",
                "calculation_scope": "each",
                "smoothing_method": "fit_polynomial",
            },
        ),
        name="CorrectIlluminationCalculate",
        source_bindings=...,
        processing_config=...,
    ),
    FunctionStep(
        func=(
            identify_primary_objects,
            {
                "exclude_size": True,
                "unclump_method": "intensity",
                "threshold_method": "Otsu",
            },
        ),
        name="IdentifyPrimaryObjects",
        processing_config=...,
    ),
]
```

The third tuple slot should not be used for CellProfiler module identity. It is only for true runtime invocation controls that are not CellProfiler settings. For example, `DefineGrid` may need a runtime grid-cycle control:

```python
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerGridCycleScope,
    CellProfilerInvocationOptions,
)

FunctionStep(
    func=(
        define_grid,
        {"grid_name": "Grid"},
        CellProfilerInvocationOptions(
            grid_cycle_scope=CellProfilerGridCycleScope.ONCE,
        ),
    ),
    name="DefineGrid",
    processing_config=...,
)
```

Most modules should therefore be just `func=(callable, kwargs)`. Anything else in generated code is either product code, compiler output, or runtime state. In particular, there is no extra `cellprofiler_module(...)` factory in the target design.

## Current Problem

The current generator emits three independent layers that all repeat the same CellProfiler module identity:

- A raw function binding, such as `identify_primary_objects_10 = require_cellprofiler_function(...)`.
- A generated artifact contract, such as `CELLPROFILER_MODULE_CONTRACTS[10] = ModuleArtifactContract(...)`.
- A generated runtime wrapper, such as `identify_primary_objects_10_runtime(...)`.

That is underabstracted. The callable, settings, source bindings, outputs, runtime artifact inputs, runtime adapter, prepare hook, and execution behavior belong to one semantic object: a CellProfiler module invocation inside an OpenHCS function pattern.

Today the duplication exists because core OpenHCS artifact planning reads artifact metadata from `CallableContract.from_callable(func)`. That forces all artifact inputs and outputs to be attached to a callable before compilation. CellProfiler cannot express per-module-instance artifacts that way without generating a distinct wrapper function for every module instance.

The right fix is not to generate cleaner wrappers. The right fix is to let artifact declarations be derived from the normalized invocation, not only from static callable metadata.

## Architectural Target

Introduce a generic invocation-aware artifact declaration path in core OpenHCS.

The compiler already normalizes function pattern items into a structure that has:

- The callable contract.
- The invocation key.
- Frozen kwargs.
- Runtime invocation options.

That normalized invocation is the correct semantic boundary. Artifact declaration providers should receive the normalized invocation and the enclosing step/source-binding context, then return artifact declarations for that invocation.

Fact-check: the current compiler path is static-callable-first. `extract_artifact_declarations(...)` iterates `iter_enabled_function_invocations(pattern)` and reads `invocation.contract.artifact_inputs` / `invocation.contract.artifact_outputs`. `_compile_invocation(...)` then derives `artifact_input_keys` and `artifact_output_keys` by calling `item.contract.select_input_plan_keys(...)` and `item.contract.select_output_plan_keys(...)`. This is why generated CellProfiler wrappers currently have to attach per-module artifact metadata to callables.

The refactor therefore needs two linked changes, not just a provider:

- Artifact graph extraction must ask an invocation-aware declaration source for producer/consumer specs.
- Invocation compilation must select artifact keys from the same invocation-aware declarations or from graph ownership keyed by `FunctionInvocationKey`, not from static callable contract selectors alone.

Default behavior remains unchanged:

- Existing OpenHCS functions continue to declare artifacts through callable metadata.
- The default provider reads `contract.artifact_inputs` and `contract.artifact_outputs`.
- Existing generated pipelines and user pipelines keep working.

CellProfiler behavior becomes a provider:

- It recognizes CellProfiler backend callables through existing callable metadata/registry identity.
- It derives the module artifact contract from the resolved callable, kwargs, invocation key/order, optional runtime invocation options, and source bindings.
- It returns producers and consumers to the core artifact graph.
- It provides the same information to runtime compilation so each compiled invocation knows its artifact input/output plan keys.

This keeps CellProfiler-specific semantics inside the CellProfiler interop layer while keeping the extension point generic.

## Proposed Core API

Add a small generic provider abstraction. Naming can change during implementation, but the shape should be approximately:

```python
class InvocationArtifactDeclarationProvider(ABC):
    @abstractmethod
    def supports(
        self,
        *,
        invocation: NormalizedFunctionItem,
        step_context: ArtifactDeclarationStepContext,
    ) -> bool:
        ...

    @abstractmethod
    def declarations(
        self,
        *,
        invocation: NormalizedFunctionItem,
        step_context: ArtifactDeclarationStepContext,
    ) -> InvocationArtifactDeclarations:
        ...
```

`InvocationArtifactDeclarations` should be a typed value object, not a loose dict:

```python
@dataclass(frozen=True, slots=True)
class InvocationArtifactDeclarations:
    inputs: tuple[ArtifactSpec, ...] = ()
    outputs: tuple[ArtifactSpec, ...] = ()
    declared_outputs: tuple[ArtifactSpec, ...] = ()
```

`ArtifactDeclarationStepContext` should be a typed value object. It should include only compile-time semantic inputs:

- `step_name`
- `step_index`
- `source_bindings`
- `processing_config` if artifact semantics need grouping/input-source context
- optional source provenance from an imported format

It should not include runtime objects, filemanager instances, GUI state, or CellProfiler workspace state.

The core artifact extraction path should then become:

```python
for invocation in iter_enabled_function_invocations(pattern):
    declarations = artifact_declaration_registry.declarations_for(
        invocation=invocation,
        step_context=step_artifact_context,
    )
    add declarations to ArtifactGraph
```

This is the load-bearing abstraction. It removes the need to mutate callables or generate one-off wrappers just to feed the compiler.

## Proposed CellProfiler Invocation API

Use direct backend callable imports plus kwargs:

```python
func=(
    identify_primary_objects,
    {"threshold_method": "Otsu", ...},
)
```

The callable is the actual OpenHCS backend function. It should carry stable product-level metadata:

- Resolved OpenHCS backend function identity.
- Base processing contract if it is known without kwargs.
- Memory type metadata.
- Runtime adapter metadata.
- Prepare hook metadata.

The callable should not carry per-invocation artifact specs. Those depend on the module instance settings, invocation order, and source bindings. They should be supplied by the invocation-aware provider.

`CellProfilerInvocationOptions` should not carry module identity. It should remain limited to runtime-only controls that are not CellProfiler settings, such as `grid_cycle_scope`.

Module identity should be derived by product/compiler code:

- Module name comes from the resolved callable and `FunctionStep.name`.
- Module instance order comes from the normalized invocation/step order.
- If exact original `.cppipe` numbering must be preserved for disabled/skipped modules, add a small generic source-provenance field rather than putting CellProfiler identity into kwargs or runtime options.

This keeps a clean separation:

- `func[0]`: what OpenHCS callable runs.
- `func[1]`: CellProfiler module settings.
- `func[2]`: rare typed runtime controls, not module identity.

The CellProfiler provider should derive the current `ModuleArtifactContracts` value internally from:

- Resolved callable.
- Module name from callable/step metadata.
- Invocation order or source-provenance module number when exact original numbering is required.
- Function resolution strategy.
- Settings kwargs.
- Runtime invocation options.
- Source bindings for plate/file inputs.
- Module semantic descriptor metadata.

That derived contract should remain a runtime/compiler value. It should not be emitted into generated pipeline files.

Provider derivation cannot depend on reconstructing a full `.cppipe` module block at compile time unless the generated step carries explicit source provenance. If exact original module numbers/settings are needed, the generator should attach a small generic provenance object to the step or invocation. The provider should otherwise use the callable identity plus kwargs as the source of truth.

## Runtime Wrapper Elimination

The current generated wrappers do four things:

- Apply the CellProfiler runtime adapter.
- Instantiate `CellProfilerModuleExecutor`.
- Prepare the raw callable and executor.
- Forward execution to `CellProfilerModuleExecutor.run(...)`.

Those are invariant product behaviors. They should move behind CellProfiler backend callable metadata and runtime adapter/product code. The generated pipeline should import the callable and provide typed invocation options; it should not define a new function per module instance.

Implementation constraints:

- The runtime callable must still expose normal OpenHCS memory metadata.
- The runtime callable must still expose runtime adapter metadata.
- The runtime callable must still support `runtime_invocation_options`.
- The runtime callable must still handle disabled invocations consistently with current function-pattern behavior.
- The runtime callable must not own output artifact names as static metadata.

The compiled plan should carry artifact keys for the invocation from the artifact graph, not from `contract.select_input_plan_keys(...)` alone. That means `_compile_invocation(...)` should consume invocation-specific artifact declarations or graph ownership when selecting `artifact_input_keys` and `artifact_output_keys`.

The runtime adapter path should remain generic. CellProfiler-specific execution should be registered as callable/runtime metadata in product code, but the `runtime_adapter(...)` mechanism should still be the way artifact-managed inputs reach the executor.

## Source Bindings Boundary

Do not put output artifacts into `StepSourceBindingsConfig`.

`StepSourceBindingsConfig` is the correct home for named inputs from plate/file sources:

- `OrigStain1` selected from file name containing `N_R`.
- `OrigStain2` selected from file name containing `N_G`.

It is not the right home for outputs such as:

- `Stain1`.
- `Objects1`.
- `RelateObjects_12_measurements`.
- `Objects1_Objects2_relationships`.

Outputs belong to artifact declaration providers and the artifact graph.

## Migration Plan

### Progress: 2026-05-14

The first core seam is implemented:

- `InvocationArtifactDeclarations` is now a nominal value object for per-invocation artifact inputs and outputs.
- `ArtifactDeclarationStepContext` is now a nominal compile-time context for provider-visible step facts: step name, step index, source bindings, processing config, and optional source provenance.
- `extract_artifact_declarations(...)` accepts an invocation declaration provider and passes normalized invocation items plus `ArtifactDeclarationStepContext` to that provider.
- `compile_function_pattern(...)` accepts the same provider/context pair and selects `CompiledFunctionInvocation.artifact_input_keys` / `artifact_output_keys` from provider declarations instead of calling the static callable contract selectors directly.
- The default provider preserves current behavior by projecting from `CallableContract`, so existing OpenHCS and generated CellProfiler pipelines keep working.
- `PipelinePathPlanner` now carries the same declaration provider into future-input planning, per-step declaration extraction, and compiled function-pattern construction.
- `PipelinePathPlanner` builds provider contexts from `StepSnapshot`, so provider implementations do not need live step/config probing.
- `CallableContract` can now carry a typed `ModuleArtifactContract`, and the default declaration provider projects runtime artifact inputs and outputs from that contract when present.
- Generated CellProfiler runtime callables now attach their `ModuleArtifactContract` through that typed callable metadata path, and generated per-artifact decorators have been removed.
- The invariant CellProfiler executor/runtime-adapter glue now lives in product runtime code via `cellprofiler_module_callable(...)`; the importer binds that product callable after loading generated direct-function declarations.
- Absorbed CellProfiler function loading and callable metadata attachment now live in the backend/runtime product layer; generated code no longer imports `require_cellprofiler_function(...)` or calls `attach_callable_contract_metadata(...)` directly.
- Generated source no longer serializes `CELLPROFILER_MODULE_CONTRACTS` or `ModuleArtifactContract(...)` literals. The importer registers compiled runtime contracts in `CellProfilerModuleContractRegistry`, and product-owned runtime step bindings reference them through `CellProfilerModuleContractBinding`.
- Materialized generated import modules write/read a product-owned, versioned JSON contract sidecar so registry-backed generated modules remain importable without putting contract literals back into generated source.
- Generated source no longer calls `cellprofiler_module_callable(...)` directly, imports runtime binding classes, or emits product-owned binding expressions in `FunctionStep.func`.
- Generated source now declares normal backend callables in `FunctionStep` declarations. Product runtime code applies artifact-aware wrappers after import using `CellProfilerRuntimeStepBinding`, `CellProfilerModuleContractRegistry`, and the generated-module contract sidecar.
- Generated source no longer emits top-level per-module raw-function or runtime-callable assignments. Callable declarations are direct absorbed backend references, and runtime wrapping is importer-owned.
- `openhcs.processing.backends.cellprofiler` now owns CellProfiler callable runtime metadata lookup so generated pipeline import logic does not recover module identity through ad hoc attribute probing.
- Focused tests now prove that the same callable can declare different artifact outputs for different invocation kwargs, and that a typed module artifact contract can drive artifact planning without per-artifact decorators.

The remaining implementation work in this plan is parity hardening and removal of legacy compatibility paths after benchmark stability, not generated boilerplate removal. The generated output target is now represented by direct backend imports plus `FunctionStep` declarations; importer/runtime code owns contracts, wrappers, adapter preparation, registry registration, and materialization sidecar persistence.

1. Add the generic invocation-aware artifact declaration provider abstraction.

The first implementation should preserve old behavior exactly by defaulting to callable metadata. This should be covered by tests on ordinary OpenHCS functions before CellProfiler uses the hook.

2. Make artifact graph extraction use invocation-aware declarations.

Update `extract_artifact_declarations(...)` and the function-pattern compilation path so that invocation-specific artifact declarations become the source of truth for `ArtifactGraph`, `artifact_input_keys`, and `artifact_output_keys`.

Acceptance for this slice: two invocations of the same callable with different kwargs can produce different artifact outputs without defining two Python wrapper functions.

3. Keep `CellProfilerInvocationOptions` limited to runtime controls.

Do not use `CellProfilerInvocationOptions` for module name or module number. Most generated CellProfiler modules should not have a third tuple slot. Use it only for true runtime controls, and keep CellProfiler settings in the kwargs tuple slot.

4. Add the CellProfiler artifact declaration provider.

This provider should reuse the existing `CellProfilerSymbolTable`, module descriptors, function resolution strategies, and `ModuleArtifactContracts` derivation. The important change is ownership: the provider derives these contracts during compilation instead of the generator serializing them into Python source.

If `CellProfilerSymbolTable` remains pipeline-level, avoid rebuilding the entire symbol table independently for every invocation. The generator/importer should provide enough normalized pipeline-source context for the provider to derive per-invocation contracts deterministically, or the compiler should build a per-step/pipeline CellProfiler contract cache once and pass it through the provider context.

5. Move runtime wrapper behavior into product code.

Create one generic CellProfiler runtime callable path that wraps `CellProfilerModuleExecutor` and the runtime adapter. Generated files should not emit executor globals or wrapper functions.

This likely requires a product-level wrapper factory or decorator applied when backend functions are registered, because a plain backend callable currently does not by itself instantiate `CellProfilerModuleExecutor`. The generated file should not do it, but product code still must.

6. Update `pipeline_generator.py` to emit step-only pipelines.

Delete generation of:

- Raw function bindings.
- Contract literals.
- Runtime wrappers.
- Contract metadata attachment.

Keep generation of:

- Imports needed by step declarations.
- Direct imports of resolved CellProfiler backend callables.
- Step kwargs.
- Source bindings.
- Processing config.
- `CellProfilerInvocationOptions` only when a module needs runtime-only controls.

7. Keep legacy generated pipeline compatibility.

Do not break existing generated files immediately. The old runtime-wrapper path can remain loadable while the generator switches to the new format. Removal can happen after benchmark parity is stable.

## Test Plan

Add tests at three levels.

Compiler tests:

- Existing callable metadata artifact declarations still produce the same `ArtifactGraph`.
- Invocation-aware providers can emit different artifacts for the same callable with different kwargs.
- `CompiledFunctionInvocation.artifact_input_keys` and `artifact_output_keys` reflect invocation-derived declarations.
- Default provider behavior matches old callable-contract behavior exactly when no custom provider supports the invocation.
- Provider resolution is deterministic when multiple providers exist; ambiguous support should fail loudly.

CellProfiler unit tests:

- A generated step-only CellProfiler pipeline compiles to the same artifact graph as the old generated contract-heavy pipeline.
- `CorrectIlluminationCalculate`, `Align`, `IdentifyPrimaryObjects`, `RelateObjects`, `FilterObjects`, `MaskObjects`, `MeasureImageAreaOccupied`, and `CalculateMath` derive the same inputs/outputs as current `CELLPROFILER_MODULE_CONTRACTS`.
- Source-bound image inputs remain source bindings, not artifact consumers.
- Runtime artifact inputs are still passed through the runtime adapter for modules that consume prior artifacts.

Generator tests:

- Generated source does not contain `CELLPROFILER_MODULE_CONTRACTS`.
- Generated source does not contain `ModuleArtifactContract`.
- Generated source does not contain `CellProfilerModuleExecutor`.
- Generated source does not contain per-module runtime wrapper definitions.
- Generated source does contain direct backend callable imports.
- Generated source does not contain `CellProfilerInvocationOptions(module_name=..., module_num=...)`.
- Generated source uses `CellProfilerInvocationOptions` only for runtime controls when needed.
- Generated source still round-trips into a `Pipeline`.

Benchmark/parity tests:

- Run the existing 18 official pipelines with old-vs-new generated source artifact graph comparison before runtime parity.
- Then run runtime parity on the 18 official pipelines.
- Then run the expanded benchmark set.
- Performance should be measured after parity, because this refactor changes compiler/runtime shape and may expose warmup or caching opportunities.

## Acceptance Criteria

The plan is complete when a generated CellProfiler pipeline for the example in the prompt contains only step declarations and normal imports, while compiling to the same artifact graph and runtime behavior as the current generated file.

Specifically:

- No generated module-level contracts.
- No generated wrapper functions.
- No generated executor globals.
- No generated callable metadata mutation.
- Artifact declarations are derived automatically from the normalized invocation.
- Runtime execution still uses the CellProfiler runtime adapter and executor.
- Existing non-CellProfiler OpenHCS function artifact behavior is unchanged.
- Existing generated pipelines remain importable during migration.

## Why This Is The Correct Boundary

The module instance is the semantic object:

- `CorrectIlluminationCalculate` module 5 with `image_name=OrigStain1` and `illumination_function_name=IllumStain1`.
- `CorrectIlluminationCalculate` module 6 with `image_name=OrigStain2` and `illumination_function_name=IllumStain2`.

Those are not different functions in the OpenHCS product. They are different invocations of the same CellProfiler module with different settings. Generating different Python function definitions for them is therefore accidental complexity.

OpenHCS already has the right user-facing container: `FunctionStep`. The missing piece is that compiler artifact declaration needs to be invocation-aware. Once that exists, CellProfiler pipelines can be as small as they should have been: step declarations plus settings.
