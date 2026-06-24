# CellProfiler Generator Declaration Query Refactor

## Purpose

Refactor the CellProfiler pipeline generator so it emits OpenHCS pipeline code
by querying module declarations instead of owning module-specific semantics.

The generator should remain the code-emission and orchestration layer. It should
not be the place where OpenHCS learns that `TrackObjects` runs over timepoint,
`GrayToColor` preserves channel identity, `SaveImages` keeps otherwise-dead
image artifacts alive, or `CorrectIlluminationCalculate` changes grouping based
on one CellProfiler setting.

## Current Shape

The current architecture is already partly declaration-driven:

- `openhcs/interop/cellprofiler/module_semantics.py` declares manual module
  category, dimensionality, infrastructure status, and mask support.
- `openhcs/interop/cellprofiler/symbol_table.py` declares and infers
  `ModuleArtifactContracts` through `ModuleContractBuilder` and
  `InferredModuleContractPattern`.
- `openhcs/interop/cellprofiler/module_settings_binding.py` owns
  `_ModuleSettingsBindingStrategy` for translating `.cppipe` settings to
  backend kwargs and invocation options.
- `openhcs/interop/cellprofiler/module_function_resolution.py` owns
  `_ModuleFunctionResolutionStrategy` for selecting backend functions.
- `openhcs/interop/cellprofiler/module_runtime_semantics.py` owns revision-
  specific runtime kwargs such as Watershed runtime-family selection.
- `openhcs/core/pipeline_image_schema.py` owns source schema semantics lowered
  from CellProfiler setup modules.

But `openhcs/interop/cellprofiler/pipeline_generator.py` still owns semantic
exceptions that should be declarative queries:

- `TrackObjectsProcessingComponentStrategy` hardcodes timepoint execution.
- `StraightenWormsProcessingComponentStrategy` hardcodes channel source-identity
  preservation under multi-image bindings.
- `GrayToColorProcessingComponentStrategy` hardcodes channel source-identity
  stacking for composite color outputs.
- `MeasureImageAreaOccupiedProcessingComponentStrategy` changes pairwise
  object-domain scoping.
- `CorrectIlluminationCalculateProcessingComponentStrategy` inspects
  `calculation_scope` and changes variable/group-by axes.
- `_save_images_required_artifacts(...)` hardcodes infrastructure artifact
  retention for skipped `SaveImages` modules.
- `PipelineGeneratorCodeEmitter.generate_steps_from_registry(...)` directly
  coordinates settings binding, function resolution, artifact contracts,
  execution-scope strategy, invocation options, materialization pruning, and
  generated source emission.

This is better than a raw `if module.name == ...` script because most behavior
is behind registered strategy families, but the generator is still a semantic
hub. The target is a query architecture where those strategy families are owned
by module declarations and the generator asks for lowered facts.

## Design Goal

Create one module-declaration query facade that is the single CellProfiler
module semantic authority for generation.

Suggested owner:

```text
openhcs/interop/cellprofiler/module_declarations.py
```

The concrete file name can change if an existing module is a better fit, but
the architectural role should be explicit:

```python
@dataclass(frozen=True, slots=True)
class CellProfilerModuleDeclaration:
    module_name: str
    semantics: CellProfilerModuleSemantics | None
    contract_builder: ModuleContractBuilder
    settings_binding: _ModuleSettingsBindingStrategy
    function_resolution: _ModuleFunctionResolutionStrategy
    runtime_semantics: ModuleRuntimeSemanticsBinding | None
    lowering: CellProfilerModuleLoweringDeclaration
    infrastructure: CellProfilerInfrastructureDeclaration
```

The generator should call a query service, not registry classes directly:

```python
declaration = CellProfilerModuleDeclarations.for_module(module.name)
contract = declaration.artifact_contract(builder, module)
bound_settings = declaration.bind_settings(module, binder, ...)
function = declaration.resolve_function(module, default_function_name)
components = declaration.processing_components(request)
retention = declaration.infrastructure_artifact_requirements(module)
```

## Declaration Records

### Module Lowering Declaration

Execution-scope semantics should be declared as data plus a small number of
generic policy hooks:

```python
@dataclass(frozen=True, slots=True)
class CellProfilerModuleLoweringDeclaration:
    processing_scope_policy: ProcessingScopePolicyKey = ProcessingScopePolicyKey.AUTO
    forced_variable_components: tuple[AllComponents, ...] = ()
    forced_group_by_component: AllComponents | None = None
    preserved_source_identity_components: tuple[AllComponents, ...] = ()
    collapse_source_stack_when_required: bool = True
    pairwise_object_domain_scope: PairwiseObjectDomainPolicy = PairwiseObjectDomainPolicy.AUTO
    setting_derived_scope: ModuleSettingScopePolicy | None = None
```

The exact field names can change, but the important point is that module-specific
behavior is exposed as declarative facts or nominal policies. The generator
should not contain leaf classes named for individual CellProfiler modules.

### Infrastructure Declaration

Skipped infrastructure modules also need declarations:

```python
@dataclass(frozen=True, slots=True)
class CellProfilerInfrastructureDeclaration:
    runtime_step: bool = False
    requires_post_execution_export: bool = False
    retained_artifact_policy: RetainedArtifactPolicy | None = None
```

This replaces generator-local helpers like `_save_images_required_artifacts`.
For `SaveImages`, the retained artifact policy should parse the source-image
setting and return the required `ArtifactSpecKey`. For `ExportToDatabase`, the
declaration should state that a post-execution export is required, while the CPA
export implementation handles the project profile.

### Setting-Derived Scope

Some behavior is not static. `CorrectIlluminationCalculate` depends on
`calculation_scope`. That should be represented as a setting-derived policy:

```python
class ModuleSettingScopePolicy(ABC):
    def processing_components(
        self,
        request: ModuleProcessingComponentRequest,
        default_components: ModuleProcessingComponents,
    ) -> ModuleProcessingComponents: ...
```

The policy can stay module-specific, but it should live under the module
declaration layer, not in the generator.

## Adjacent High-Risk Follow-Up Files

A lightweight scan of `openhcs/interop/cellprofiler` and `openhcs/runtime`
found several files with the same risk profile as `pipeline_generator.py`: large
semantic hubs that mix dispatch, policy, external compatibility rules, and
runtime wiring. These should be considered after the generator boundary is in
place.

The ranking below is based on size, number of strategy/policy classes, direct
module/viewer-specific behavior, and proximity to bugs already seen in source
binding, streaming axes, and CellProfiler runtime output semantics.

### Priority 1: CellProfiler Runtime Module Execution

```text
openhcs/interop/cellprofiler/runtime/module_execution.py
```

Observed risk:

- approximately 4,300 lines;
- dozens of runtime policies in one file;
- owns runtime callable binding, image execution strategy selection, object
  input policies, special input policies, measurement materialization entry
  points, relationship handling, and dual-scope measurement policy;
- contains many module-named policy leaves, such as `MaskImage`,
  `RelateObjects`, `Watershed`, `StraightenWorms`, `ClassifyObjects`, and
  measurement-specific policies.

Why this is related:

The generator should query module declarations for compile-time lowering. The
runtime needs the same architecture for execution-time behavior. A module
declaration should be able to answer both:

- how this module lowers into an OpenHCS `FunctionStep`;
- how this module binds special/runtime inputs when the step executes.

Likely refactor:

- split runtime policies into declaration-query-owned families:
  - image execution declaration;
  - object input binding declaration;
  - special input binding declaration;
  - measurement-output declaration;
  - dual-scope measurement declaration;
- keep a small `CellProfilerRuntimeCallable` orchestration layer that asks those
  declarations instead of carrying module-specific leaves locally.

Acceptance:

- `module_execution.py` becomes an execution coordinator, not a policy catalog;
- module-named runtime behavior is discoverable through the same declaration
  query surface introduced for generation;
- generator and runtime cannot disagree about a module's special inputs,
  measurement scope, or object-domain semantics.

### Priority 2: CellProfiler Settings Binding

```text
openhcs/interop/cellprofiler/module_settings_binding.py
```

Observed risk:

- approximately 3,300 lines;
- more than 100 classes;
- owns generic setting binding plus many module-specific binding strategies;
- already imports `ModuleRuntimeSemanticsBinding`, which means runtime-family
  decisions can leak into settings translation.

Why this is related:

Settings binding is part of module declaration, not code generation. The
generator should ask for already-bound kwargs and invocation options. It should
not know which settings binder family a module uses.

Likely refactor:

- expose settings binding through `CellProfilerModuleDeclaration`;
- split large module-specific binders into focused setting-family modules where
  they are not already split;
- make setting coverage records an output of the module generation plan, not a
  generator-local side channel.

Acceptance:

- adding a new module-specific setting binding does not require touching
  `pipeline_generator.py`;
- declaration query can list modules with custom settings binding;
- setting-derived execution behavior is represented as typed policy, not
  incidental kwargs.

### Priority 3: Symbol Table And Artifact Contracts

```text
openhcs/interop/cellprofiler/symbol_table.py
```

Observed risk:

- approximately 2,800 lines;
- owns symbol identity, source bindings, artifact contracts, inferred contract
  patterns, literal code rendering, and many module-named contract builders.

Why this is related:

Artifact contracts are one of the core declaration products. The symbol table
should remain the workspace-name authority, but module-specific contract
builders should be reachable through the declaration query layer rather than
being a separate registry island.

Likely refactor:

- keep `CellProfilerSymbolTable` focused on symbol storage and lookup;
- move module contract builder lookup behind `CellProfilerModuleDeclaration`;
- split generated-code literal rendering out of the symbol table if it remains
  only for generated source emission;
- make inferred contract patterns a declaration feature with explicit
  confidence/diagnostic output.

Acceptance:

- symbol table can compile contracts by querying declarations;
- symbol storage, contract inference, and generated-code literal rendering are
  separate owners;
- contract builders remain nominal and fail loudly for unsupported modules.

### Priority 4: Source Candidate And Runtime Source Binding

```text
openhcs/interop/cellprofiler/runtime/source_candidates.py
openhcs/interop/cellprofiler/runtime/source_binding_runtime.py
openhcs/interop/cellprofiler/runtime/source_identity.py
```

Observed risk:

- `source_candidates.py` is approximately 1,900 lines;
- `source_binding_runtime.py` is approximately 1,360 lines;
- these files decide how source metadata, aliases, planes, path identity,
  current-step payloads, and pipeline-start payloads become runtime inputs.

Why this is related:

This is the runtime counterpart to source-schema lowering. The architecture only
works if source identity is centralized. Any local interpretation here can
recreate the same bugs that showed up in streaming axes and CPA source-row
projection.

Likely refactor:

- define a core-owned source candidate/query API that represents:
  - source image set identity;
  - alias-to-plane resolution;
  - structured source refs;
  - current-step versus pipeline-start source origin;
- make CellProfiler runtime source binding a profile over that core API;
- remove local fallback path identity logic where a typed source projection can
  answer the question.

Acceptance:

- source binding runtime does not infer channel/site/well identity from paths
  when structured metadata exists;
- source candidate matching is testable without executing a CellProfiler module;
- CPA source-row reconstruction and runtime source binding share the same core
  source projection authorities.

### Priority 5: Measurement Materialization And Object Row Policies

```text
openhcs/interop/cellprofiler/runtime/measurement_materialization.py
openhcs/interop/cellprofiler/runtime/object_measurement_row_policies.py
openhcs/interop/cellprofiler/runtime/measurement_rows.py
openhcs/interop/cellprofiler/runtime/relationship_measurement_rows.py
```

Observed risk:

- these files translate CellProfiler row shapes into core
  `MeasurementTable`, object row ownership, image numbers, and relationships;
- they sit directly on the boundary needed by CPA export and runtime parity.

Why this is related:

The generator declaration refactor makes compile-time module semantics
queryable. Measurement materialization needs the same treatment for row
semantics: scope, ownership, object id fields, source image name, relationship
endpoints, and image-number projection should be queryable declarations or core
row-projection policies.

Likely refactor:

- move row ownership and image-number behavior into core measurement row
  projection policies where possible;
- keep CellProfiler-specific dialect names at the interop edge;
- make relationship row construction produce typed relationship graphs rather
  than database/export-specific rows.

Acceptance:

- runtime measurement tables and CPA export tables consume the same row
  projection facts;
- object-label measurement, relationship, and per-image rows cannot diverge in
  their interpretation of image number or object identity;
- CellProfiler-specific feature-name parsing is isolated from core row
  projection.

### Priority 6: Napari/Fiji Viewer Servers

```text
openhcs/runtime/napari_viewer_server.py
openhcs/runtime/fiji_viewer_server.py
openhcs/runtime/napari_streaming_handlers.py
openhcs/runtime/viewer_component_system.py
openhcs/runtime/viewer_protocol.py
```

Observed risk:

- `fiji_viewer_server.py` is approximately 2,750 lines;
- `napari_viewer_server.py` is approximately 2,500 lines;
- both include transport/server code near viewer-specific payload handling;
- `napari_streaming_handlers.py` owns layer keys, batching, axis projection,
  label rasterization, and presentation behavior;
- `viewer_component_system.py` is the shared semantic owner but is also large
  enough that it can become a catch-all.

Why this is related:

The viewer bugs came from local interpretation of component axes and payload
identity. This is the same class of issue as generator-owned module semantics:
viewers should query a shared declaration/profile for component layout,
payload-type handling, and presentation behavior.

Likely refactor:

- introduce a viewer route/profile query layer:
  - shared route identity;
  - component axis projection;
  - data-type handler registry;
  - payload presentation profile;
  - transport-specific writer;
- keep Napari/Fiji differences as profile declarations, not duplicated control
  flow;
- split server lifecycle from payload rendering.

Acceptance:

- adding or changing a streaming payload type registers a handler/profile once;
- Napari and Fiji consume the same component-axis and route-identity decisions;
- viewer server files become transport/lifecycle coordinators.

### Priority 7: ZMQ Execution Server

```text
openhcs/runtime/zmq_execution_server.py
```

Observed risk:

- combines request payload resolution, config-code policy, orchestrator setup,
  compile-only behavior, execution behavior, progress publication, and result
  shaping;
- not as module-semantic-heavy as the CellProfiler files, but it is a runtime
  boundary hub.

Likely refactor:

- split transport request parsing from execution planning;
- make compile/execute modes explicit request strategies;
- keep ZMQ serialization separate from orchestrator execution policy.

Acceptance:

- direct execution, ZMQ execution, and future agent execution can share the same
  execution request model;
- ZMQ server owns transport and lifecycle, not execution semantics.

## Suggested Follow-Up Order

1. Finish the generator declaration query boundary.
2. Move runtime module execution policies behind the same declaration query
   layer.
3. Fold settings binding and symbol-table contract builders into that query
   layer.
4. Refactor source candidate/source binding runtime around core source
   projection authorities.
5. Refactor measurement/object/relationship row projection for CPA and runtime
   parity.
6. Apply the same query/profile pattern to Napari/Fiji viewer routes and payload
   handling.
7. Split ZMQ execution server after semantic boundaries are stable.

This order keeps the highest semantic-risk CellProfiler boundaries ahead of
transport cleanup. It also lets the CPA export plan reuse the source and
measurement projection work instead of building a parallel exporter-specific
model.

## Concrete Plans For Follow-Up Items 1 And 2

### Plan 1: Generator Declaration Query Boundary

Goal: `pipeline_generator.py` should orchestrate parsing, planning, pruning, and
source emission, but it should not be the owner of module-specific CellProfiler
facts. It should ask a declaration facade for those facts.

Target changes:

| File | Approx. lines | Change | Why |
| --- | ---: | --- | --- |
| `openhcs/interop/cellprofiler/module_declarations.py` | new file | Add `CellProfilerModuleDeclaration`, `CellProfilerModuleDeclarations`, `CellProfilerModuleLoweringDeclaration`, and `CellProfilerInfrastructureDeclaration`. The first commit should delegate to existing registries instead of moving behavior. | Creates one query boundary without creating a new behavior hub. |
| `openhcs/interop/cellprofiler/pipeline_generator.py` | 1-80 | Update the stale generator description/import surface so it describes declaration-query lowering, not LLM/category-owned lowering. Import the new declaration facade. | The file header currently reinforces the wrong authority model. |
| `openhcs/interop/cellprofiler/pipeline_generator.py` | 1053-1228 | Keep the generic `ModuleProcessingComponentStrategy`, `ModuleProcessingScopePolicy`, and default scope policy machinery initially, but stop calling `ModuleProcessingComponentStrategy.for_module(...)` directly from emitter/build code. Route through `CellProfilerModuleDeclaration.lowering.processing_components(...)`. | The generic scope algebra is useful infrastructure; the lookup authority belongs to declarations. |
| `openhcs/interop/cellprofiler/pipeline_generator.py` | 1272-1357 | Change `RuntimeArtifactSourceLineage` so pairwise-object-domain scope asks the declaration facade for the module lowering policy instead of calling `ModuleProcessingComponentStrategy.for_module(...)` directly. | This closes a hidden generator-owned semantic lookup that would otherwise remain after the emitter is fixed. |
| `openhcs/interop/cellprofiler/pipeline_generator.py` | 1359-1468 | Move the module-specific leaves `TrackObjectsProcessingComponentStrategy`, `StraightenWormsProcessingComponentStrategy`, `GrayToColorProcessingComponentStrategy`, `MeasureImageAreaOccupiedProcessingComponentStrategy`, and `CorrectIlluminationCalculateProcessingComponentStrategy` behind declarations. The first pass may leave class bodies in place if the declaration facade owns lookup; the second pass should move them to `module_declarations.py` or a sibling `module_lowering_declarations.py`. | These are exactly the generator-owned semantic exceptions the plan is trying to remove. |
| `openhcs/interop/cellprofiler/pipeline_generator.py` | 1491-1503 | Replace `_save_images_required_artifacts(...)` with `CellProfilerInfrastructureDeclaration.retained_artifacts(module)` or equivalent. `SaveImages` becomes one infrastructure declaration, not a generator helper. | Infrastructure retention must generalize to `ExportToDatabase`/CPA-style consumers without editing generator pruning logic. |
| `openhcs/interop/cellprofiler/pipeline_generator.py` | 2023-2154 | Split `PipelineGeneratorCodeEmitter.generate_steps_from_registry(...)` into two surfaces: a planner that builds `CellProfilerModuleGenerationPlan` objects, and an emitter that renders those plans. The planner queries declarations for settings binding, function resolution, processing components, invocation options, and artifact comments. | This method currently coordinates too many semantic registries and makes generated source the only inspectable plan artifact. |
| `openhcs/interop/cellprofiler/pipeline_generator.py` | 2252-2415 | Change `PipelineGeneratorBuildStage.generate(...)` to construct a declaration facade once, use it for infrastructure import notes, retained artifacts, function resolution, and module generation plans, then pass already-lowered plans to the emitter. | This is the top-level semantic junction where skipped infrastructure modules, contracts, pruning, and function names are currently stitched manually. |
| `openhcs/interop/cellprofiler/pipeline_generator.py` | 2476-2532 | Add an optional declaration facade dependency to `PipelineGenerator.__init__` with a default `CellProfilerModuleDeclarations.default()` instance. | Tests and future export code need to inject/probe declarations without monkeypatching global registries. |
| `openhcs/interop/cellprofiler/module_settings_binding.py` | 603-742, plus leaves below | Do not move settings binders in plan 1. Expose them through `CellProfilerModuleDeclaration.settings_binding` and keep `_ModuleSettingsBindingStrategy.for_module(...)` as the delegated implementation. | Keeps the first generator change behavior-preserving. |
| `openhcs/interop/cellprofiler/module_function_resolution.py` | 47-90 | Do not move function resolution in plan 1. Expose it through `CellProfilerModuleDeclaration.function_resolution`. | Same behavior, clearer authority. |
| `openhcs/interop/cellprofiler/symbol_table.py` | 1329-1574 | Do not move contract builders in plan 1. Expose `ModuleContractBuilder.for_module(...)` through `CellProfilerModuleDeclaration.contract_builder`. | Symbol contracts are shared by generator and runtime-facing export work; route through declarations before changing implementation. |
| `openhcs/interop/cellprofiler/module_semantics.py` | 86-160 | Expose existing `CellProfilerModuleSemantics` through `CellProfilerModuleDeclaration.semantics`. | Prevents a parallel semantic record from forming. |
| `openhcs/interop/cellprofiler/module_runtime_semantics.py` | 37-63 | Expose existing runtime-setting declarations through `CellProfilerModuleDeclaration.runtime_semantics`. | Keeps revision-specific runtime kwargs queryable from the same authority. |

Concrete implementation sequence:

1. Add `module_declarations.py` with a pure facade. It should contain no
   `if module_name == ...` switches. It should canonicalize once, then delegate
   to existing `for_module(...)` registries.
2. Introduce `CellProfilerModuleGenerationPlan` near the generator planner
   first, probably in `pipeline_generator.py` near `GeneratedStepSettings`
   around lines 652-710. Move it to `module_declarations.py` only if it becomes
   declaration-owned rather than generator-owned.
3. Redirect `PipelineGeneratorBuildStage.generate(...)` calls around lines
   2303-2365:
   - `_save_images_required_artifacts(skipped_modules)`
   - `_ModuleFunctionResolutionStrategy.for_module(module.name)`
   - infrastructure import notes if practical
4. Redirect `PipelineGeneratorCodeEmitter.generate_steps_from_registry(...)`
   around lines 2028-2154:
   - `_ModuleSettingsBindingStrategy.for_module(module.name)`
   - `ModuleProcessingComponentStrategy.for_module(module.name)`
   - generated comment text that says `LLM-inferred category`
5. Redirect `RuntimeArtifactSourceLineage._collect_pairwise_object_domain_scope`
   around lines 1323-1350 so lineage scope uses the declaration-owned lowering
   policy.
6. Move the module-specific processing strategy leaves only after the facade
   redirect is tested. The generic scope policies can stay in
   `pipeline_generator.py` until there is a clearer shared home.

Acceptance for plan 1:

- generated source for a representative `.cppipe` is behavior-identical, with
  only import/comment ordering differences allowed;
- `pipeline_generator.py` has no direct calls to `_ModuleSettingsBindingStrategy`
  or `_ModuleFunctionResolutionStrategy` outside the declaration facade path;
- generator-owned module-specific processing leaves are either moved or hidden
  behind declaration-owned lookup with a follow-up diff that physically moves
  them;
- `SaveImages` retained artifact behavior is declared as infrastructure policy;
- focused tests cover `TrackObjects`, `StraightenWorms`, `GrayToColor`,
  `CorrectIlluminationCalculate`, `MeasureImageAreaOccupiedBinary`, and
  `SaveImages`.

### Plan 2: Runtime Module Execution Declaration Query Boundary

Goal: `runtime/module_execution.py` should execute a prepared module runtime
plan, but module-specific runtime policy lookup should come from the same
declaration authority used by generation.

Target changes:

| File | Approx. lines | Change | Why |
| --- | ---: | --- | --- |
| `openhcs/interop/cellprofiler/module_declarations.py` | new file from plan 1 | Extend declarations with a `CellProfilerRuntimePolicyDeclaration` or equivalent fields for primary image input, object input, special input, invocation execution mode, main-flow replacement, object measurement rows, measurement record builder, and dual-scope measurement. Initial implementation delegates to existing runtime policy registries. | Runtime and generator must query the same module declaration surface instead of maintaining parallel semantic lookup paths. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py` | 662-779 | Keep `CellProfilerRuntimeCallable` focused on picklable callable wrapping. Do not add declaration logic here beyond passing or resolving the runtime declaration service for its executor. | The callable wrapper should not become the new semantic owner. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py` | 907-1070 | Change `CellProfilerModuleRuntimePlan.build(...)` so it receives a `CellProfilerModuleDeclaration` or `CellProfilerRuntimePolicyDeclaration` instead of independently calling `CellProfilerSpecialInputPolicy.for_module(...)`, `CellProfilerObjectInputPolicy.for_module(...)`, `CellProfilerDualScopeMeasurementPolicy.for_module(...)`, `CellProfilerInvocationExecutionModePolicy.for_module(...)`, `CellProfilerMainFlowReplacementPolicy.for_module(...)`, `CellProfilerObjectMeasurementRowPolicy.for_module(...)`, and `CellProfilerMeasurementRecordBuilder.for_module(...)`. | This is the runtime semantic convergence point; it currently assembles policies from many independent registries. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py` | 1079-1179 | Change `CellProfilerModuleExecutor.__post_init__` and `runtime_plan(...)` to resolve the declaration once by canonical module name and pass it into `CellProfilerModuleRuntimePlan.build(...)`. | The executor owns per-module runtime caching, so it is the right place to cache declaration-derived runtime policy. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py` | 2246-2334 | Keep the `CellProfilerInvocationExecutionModePolicy` class family, but make declaration runtime policy expose the selected instance. Later move leaves like `CorrectIlluminationCalculateExecutionModePolicy`, `ColorToGrayExecutionModePolicy`, and `DefineGridManualExecutionModePolicy` out of the monolith if the declaration module becomes the policy home. | These are module-specific runtime execution facts, not executor mechanics. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py` | 2649-2727 | Route `CellProfilerPrimaryImageInputPolicy.for_module(...)` through runtime declarations. `DefaultPrimaryImageInputPolicy` should ask the declaration's special-input policy rather than doing its own `CellProfilerSpecialInputPolicy.for_module(...)` lookup. | This removes cross-registry coupling inside default primary-image selection. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py` | 2749-3268 | Route object-label input policy lookup through runtime declarations. Keep generic bases such as `DeclaredSingleObjectLabelInputPolicy`, `ObjectRowsInputPolicy`, and `ObjectRowsWithMeasurementsInputPolicy`; make generated leaf specs declaration-visible. | Object-label binding is semantic module policy; the executor should consume the selected policy. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py` | 3427-4129 | Route special-input policy lookup through runtime declarations. Keep request helpers in runtime; move or expose module-specific leaves including `MaskImageSpecialInputPolicy`, `RelateObjectsSpecialInputPolicy`, `CropSpecialInputPolicy`, `ImageMathSpecialInputPolicy`, `WatershedSpecialInputPolicy`, `StraightenWormsSpecialInputPolicy`, `ConvertObjectsToImageSpecialInputPolicy`, `DisplayDataOnImageSpecialInputPolicy`, and `ClassifyObjectsMeasurementInputPolicy`. | Special inputs are the most common runtime source of source/label alignment bugs; they need one discoverable declaration surface. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py` | 4180-4265 | Route `CellProfilerPerImageMeasurementPolicy` and `CellProfilerDualScopeMeasurementPolicy` decisions through runtime declarations. Generated dual-scope leaves for `MeasureTexture` and `MeasureColocalization` should become declaration-visible module policies. | Measurement cardinality is module semantics and must align with generator contracts and future CPA export projections. |
| `openhcs/interop/cellprofiler/runtime/generated_pipeline.py` | around runtime binding helpers | Keep generated pipeline binding unchanged unless it needs to pass a declaration service into `cellprofiler_module_callable(...)`. Prefer resolving declarations inside `CellProfilerModuleExecutor` from the contract module name. | Avoid leaking declaration plumbing into generated Python unless there is a concrete need. |
| `openhcs/interop/cellprofiler/runtime/policy_registry.py` | whole file | Reuse the existing policy registry/metaclass helpers. Do not invent protocol-based registries. Add only the minimal helper needed for declaration-owned runtime policy lookup. | The current nominal registry pattern is the right mechanism; the problem is scattered lookup, not lack of registry infrastructure. |

Concrete implementation sequence:

1. Add runtime policy fields/methods to the declaration facade without moving
   policy class bodies. For example:
   `declaration.runtime_policy.object_input_policy`,
   `declaration.runtime_policy.special_input_policy`, and
   `declaration.runtime_policy.invocation_execution_mode_policy`.
2. Modify `CellProfilerModuleExecutor.__post_init__` around lines 1097-1108 to
   cache the declaration for `self._canonical_module_name`.
3. Modify `CellProfilerModuleRuntimePlan.build(...)` around lines 943-1067 to
   accept the declaration-derived runtime policy object and remove direct
   registry lookups from that method.
4. Fix `DefaultPrimaryImageInputPolicy.primary_image_inputs(...)` around lines
   2678-2704 so it does not perform a second special-input policy lookup. It
   should receive the selected special-input policy through the runtime plan or
   declaration runtime policy.
5. Move only the lookup authority first. After tests pass, physically move
   module-specific runtime leaves out of `runtime/module_execution.py` in
   coherent groups:
   - execution-mode leaves from lines 2246-2334;
   - primary-image leaves from lines 2649-2727;
   - object-input leaves from lines 2749-3268;
   - special-input leaves from lines 3427-4129;
   - dual-scope measurement leaves from lines 4180-4265.
6. Keep request/data classes that are pure runtime mechanics in
   `runtime/module_execution.py`: `SpecialInputBindingRequest`,
   `ObjectInputBindingRequest`, `CellProfilerModuleRuntimePlan`, and
   `CellProfilerModuleExecutor` should remain until a later split has a clear
   owner.

Acceptance for plan 2:

- `CellProfilerModuleRuntimePlan.build(...)` has one declaration-derived runtime
  policy input instead of many independent `*.for_module(...)` lookups;
- `DefaultPrimaryImageInputPolicy` no longer performs special-input lookup on
  its own;
- module-specific runtime policy leaves are declaration-visible and enumerable;
- no generated pipeline source changes unless required by a failing integration
  test;
- integration tests cover at least `MaskImage`, `RelateObjects`, `Crop`,
  `ImageMath`, `Watershed`, `StraightenWorms`, `ClassifyObjects`,
  `MeasureTexture`, and `MeasureColocalization`.

## Architectural Judgements To Encode Before Codemodding

These decisions are not codemod decisions. They are the semantic boundary the
codemod is allowed to enforce.

| Judgement | Decision | Why |
| --- | --- | --- |
| Declaration authority | `CellProfilerModuleDeclaration` is an authority facade over existing nominal policy registries. It delegates and aggregates; it must not grow `if module_name == ...` switches. | A facade is useful only if it collapses lookup traffic without becoming a new semantic hub. |
| Generator ownership | `pipeline_generator.py` owns parsing orchestration, module partitioning, dead-artifact pruning, generation-plan construction, and source emission. It does not own module-specific CellProfiler behavior. | The generator should be stable when a CellProfiler module gains new semantics; only declarations/policies should change. |
| Runtime ownership | `runtime/module_execution.py` owns runtime request construction, execution sequencing, adapter interaction, and output recording mechanics. It should consume declaration-selected policies rather than choosing module policies itself. | Runtime bugs come from compile/runtime semantic drift. A shared declaration query prevents separate policy selection paths from disagreeing. |
| Generic lowering machinery | Generic source-axis and runtime-artifact scope algebra can stay near the generator until it has a clearer shared owner. Module-named leaves must move behind declarations. | Moving generic algebra too early creates churn; leaving module leaves in the generator preserves the leak. |
| Runtime request records | Request/data classes such as `SpecialInputBindingRequest`, `ObjectInputBindingRequest`, `CellProfilerModuleRuntimePlan`, and `CellProfilerModuleExecutor` are runtime mechanics unless they contain module-specific selection. | These objects encode execution context, not CellProfiler module declarations. Moving them blindly would make declarations depend on runtime internals. |
| Static vs setting-derived semantics | Setting-derived behavior should be represented by typed dynamic policy objects, not flattened into static declaration constants. | Modules like `CorrectIlluminationCalculate` change behavior based on settings; a static record would hide real runtime/generator conditions. |
| Infrastructure modules | Skipped infrastructure modules such as `SaveImages` and later `ExportToDatabase` should declare artifact/export requirements. The generator should ask for requirements. | Pruning and export compatibility need to see infrastructure consumers without hardcoded generator helpers. |
| Generated pipeline boundary | Generated Python should not receive declaration plumbing unless integration tests prove it is required. Prefer resolving declarations from the runtime contract module name inside the executor. | Generated pipelines should stay small and stable; declaration lookup is product runtime infrastructure. |
| Advisor role | Advisor output is an acceptance gate on touched backend files, not the source of architectural truth. | The advisor can detect under-abstraction symptoms and local role-case logic; humans still choose the boundary and encode it as invariants. |

Codemod work is valid only after these judgements are encoded as tests or
explicit invariants. If a transform has to infer one of these judgements from
domain meaning, it should report an ambiguous candidate and stop.

## Codemod Plan

Add a focused LibCST codemod script:

```text
tools/codemods/cellprofiler_declaration_query.py
```

The script should support:

```bash
python tools/codemods/cellprofiler_declaration_query.py --dry-run
python tools/codemods/cellprofiler_declaration_query.py --apply
python tools/codemods/cellprofiler_declaration_query.py --check
```

Use LibCST rather than regex so imports, call expressions, keyword arguments,
and class definitions are parsed structurally. The codemod should default to
`--dry-run`; `--apply` should require a clean target set and should print every
changed file plus every skipped ambiguous match.

### Codemod Stage 0: Inventory And Guards

Purpose: identify exact mechanical candidates and enforce the architecture
before changing behavior.

Targets:

- `openhcs/interop/cellprofiler/pipeline_generator.py`
- `openhcs/interop/cellprofiler/runtime/module_execution.py`
- `openhcs/interop/cellprofiler/module_settings_binding.py`
- `openhcs/interop/cellprofiler/module_function_resolution.py`
- `openhcs/interop/cellprofiler/symbol_table.py`
- `openhcs/interop/cellprofiler/module_semantics.py`
- `openhcs/interop/cellprofiler/module_runtime_semantics.py`

Checks:

- list all direct `*.for_module(...)` call sites in generator/runtime files;
- list all module-named policy leaf classes that remain in
  `pipeline_generator.py` and `runtime/module_execution.py`;
- list all hardcoded infrastructure helpers such as
  `_save_images_required_artifacts(...)`;
- fail `--check` if new direct registry lookup appears in the generator or
  runtime plan after the facade migration.

Do not change files in stage 0.

### Codemod Stage 1: Generator Query Routing

Prerequisite: `module_declarations.py` exists and exposes a pure delegating
facade.

Safe transforms:

| Existing shape | New shape |
| --- | --- |
| `_ModuleSettingsBindingStrategy.for_module(module.name).bind(...)` | `declarations.for_module(module.name).bind_settings(...)` or `declaration.bind_settings(...)` |
| `_ModuleFunctionResolutionStrategy.for_module(module.name).resolve(...)` | `declarations.for_module(module.name).resolve_function(...)` or `declaration.resolve_function(...)` |
| `ModuleProcessingComponentStrategy.for_module(module.name).components(request)` | `declarations.for_module(module.name).processing_components(request)` |
| `ModuleProcessingComponentStrategy.for_module(contract.module_name).module_requires_pairwise_object_domain_scope(contract)` | `declarations.for_module(contract.module_name).module_requires_pairwise_object_domain_scope(contract)` |
| `_save_images_required_artifacts(skipped_modules)` | declaration-owned retained artifact collection over skipped modules |

Ambiguous cases:

- if the module name expression is not syntactically tied to the same `module`
  or `contract` used by the request, report and skip;
- if the lookup result is stored and used for more than the known method call,
  report and skip;
- if the target scope has no `declarations` object in scope, report the needed
  insertion rather than guessing where to create one.

Required manual setup before `--apply`:

- add a `declarations` dependency to `PipelineGenerator`;
- pass or construct declarations in `PipelineGeneratorBuildStage.generate(...)`;
- decide whether `CellProfilerModuleGenerationPlan` lives in
  `pipeline_generator.py` or `module_declarations.py`.

Post-check:

- no direct `_ModuleSettingsBindingStrategy.for_module(...)` or
  `_ModuleFunctionResolutionStrategy.for_module(...)` calls remain in
  `pipeline_generator.py`;
- no direct `ModuleProcessingComponentStrategy.for_module(...)` calls remain in
  generator orchestration or lineage code;
- generated pipeline parity tests pass.

### Codemod Stage 2: Runtime Policy Query Routing

Prerequisite: declaration facade exposes runtime policy selection while
delegating to existing runtime policy registries.

Safe transforms inside `CellProfilerModuleRuntimePlan.build(...)`:

| Existing shape | New shape |
| --- | --- |
| `CellProfilerSpecialInputPolicy.for_module(canonical_module_name)` | `runtime_policy.special_input_policy` |
| `CellProfilerObjectInputPolicy.for_module(canonical_module_name)` | `runtime_policy.object_input_policy` |
| `CellProfilerDualScopeMeasurementPolicy.for_module(canonical_module_name)` | `runtime_policy.dual_scope_measurement_policy` |
| `CellProfilerInvocationExecutionModePolicy.for_module(canonical_module_name)` | `runtime_policy.invocation_execution_mode_policy` |
| `CellProfilerMainFlowReplacementPolicy.for_module(canonical_module_name)` | `runtime_policy.main_flow_replacement_policy` |
| `CellProfilerObjectMeasurementRowPolicy.for_module(canonical_module_name)` | `runtime_policy.object_measurement_row_policy` |
| `CellProfilerMeasurementRecordBuilder.for_module(canonical_module_name)` | `runtime_policy.measurement_record_builder` |

Manual setup before `--apply`:

- add a declaration/runtime-policy parameter to
  `CellProfilerModuleRuntimePlan.build(...)`;
- cache the declaration or runtime policy in `CellProfilerModuleExecutor`;
- pass the cached runtime policy from `runtime_plan(...)`.

Special handling:

- `DefaultPrimaryImageInputPolicy.primary_image_inputs(...)` currently performs
  a nested `CellProfilerSpecialInputPolicy.for_module(...)` lookup. Do not let
  the codemod blindly rewrite this unless the selected special-input policy is
  already passed into the method or request object.

Post-check:

- `CellProfilerModuleRuntimePlan.build(...)` has no direct `*.for_module(...)`
  policy selection except declaration lookup itself;
- default primary-image selection does not perform a second special-input
  policy lookup;
- runtime module execution tests pass for object, image, special-input, and
  dual-scope measurement modules.

### Codemod Stage 3: Leaf Policy Moves

Prerequisite: stages 1 and 2 pass, and declaration-visible lookup owns all leaf
policy selection.

This stage can use LibCST to move class definitions, but only after dependency
closure is known. It should generate a report first:

- class name;
- base classes;
- class attributes;
- referenced imported names;
- referenced local helper functions/classes;
- destination module;
- imports required at destination;
- imports removable from source.

Move groups:

| Source | Approx. lines | Destination |
| --- | ---: | --- |
| generator processing leaves | `pipeline_generator.py` 1359-1468 | `module_declarations.py` or `module_lowering_declarations.py` |
| runtime execution-mode leaves | `runtime/module_execution.py` 2246-2334 | `module_runtime_declarations.py` or declaration-owned runtime policy module |
| runtime primary-image leaves | `runtime/module_execution.py` 2649-2727 | declaration-owned runtime policy module |
| runtime object-input leaves | `runtime/module_execution.py` 2749-3268 | declaration-owned runtime policy module, unless they depend too heavily on runtime request helpers |
| runtime special-input leaves | `runtime/module_execution.py` 3427-4129 | declaration-owned runtime policy module, keeping request helpers in runtime |
| runtime dual-scope measurement leaves | `runtime/module_execution.py` 4180-4265 | declaration-owned runtime policy module |

Skip a class move if:

- moving it creates an import cycle;
- more than two local runtime helper dependencies would need to move with it;
- it depends on mutable runtime execution state rather than request records;
- its base class or metaclass is not importable from the destination without
  pulling in the runtime executor.

Post-check:

- source modules still import and tests collect;
- declarations can enumerate every module with custom compile-time or runtime
  policy;
- source modules contain runtime mechanics and generic algebra, not module-named
  semantic leaves.

### Codemod Stage 4: Architecture Enforcement

After manual and codemod changes, `--check` should enforce these forbidden
patterns through the Nominal Refactor Advisor codemod guard API:

```python
from nominal_refactor_advisor import (
    ArchitectureGuardRule,
    evaluate_architecture_guards,
)
```

The OpenHCS codemod should build project-specific `ArchitectureGuardRule`
instances and pass its parsed `SourceIndex` plus changed source text to
`evaluate_architecture_guards(...)`. The advisor owns source-index addressing,
literal-dispatch recognition, and structured violation reporting; the OpenHCS
codemod owns the project-specific forbidden call names and dispatch subjects.

Rules should cover:

- no `_ModuleSettingsBindingStrategy.for_module(...)` in
  `pipeline_generator.py`;
- no `_ModuleFunctionResolutionStrategy.for_module(...)` in
  `pipeline_generator.py`;
- no `ModuleProcessingComponentStrategy.for_module(...)` in generator
  orchestration or lineage code;
- no direct runtime policy `*.for_module(canonical_module_name)` calls inside
  `CellProfilerModuleRuntimePlan.build(...)`;
- no new `if module.name == ...`, `if module_name == ...`, or dict dispatch over
  module names inside generator/runtime orchestration files;
- no declaration facade method with a long local role/case split over concrete
  module names.

If `--check` fails, the fix is architectural: move the behavior behind a nominal
policy/declaration rather than adding another exception to the codemod.

## Migration Plan

### Phase 1: Introduce Query Facade Without Behavior Change

Add a declaration query facade that delegates to existing registries:

- `cellprofiler_module_semantics(...)`
- `ModuleContractBuilder.for_module(...)`
- `_ModuleSettingsBindingStrategy.for_module(...)`
- `_ModuleFunctionResolutionStrategy.for_module(...)`
- `ModuleRuntimeSemanticsBinding.for_module(...)`
- `ModuleProcessingComponentStrategy.for_module(...)`

Do not move logic yet. Change generator call sites to ask the facade. This
creates the boundary first and keeps the diff reviewable.

Acceptance:

- generated pipelines are byte-for-byte identical except for comments/import
  ordering if unavoidable;
- existing generated pipeline tests pass;
- no module-specific behavior is added to the generator during this phase;
- codemod stage 0 inventory is committed or attached to the PR notes before
  stage 1 is applied.

### Phase 2: Move Processing Component Strategies Behind Declarations

Move these generator-owned leaf strategies into the declaration layer:

- `TrackObjectsProcessingComponentStrategy`
- `StraightenWormsProcessingComponentStrategy`
- `GrayToColorProcessingComponentStrategy`
- `MeasureImageAreaOccupiedProcessingComponentStrategy`
- `CorrectIlluminationCalculateProcessingComponentStrategy`

Keep the generic lowering machinery in or near the generator only if it is truly
code-generation infrastructure. The module-specific facts should be discoverable
through `CellProfilerModuleDeclaration.lowering`.

Acceptance:

- `pipeline_generator.py` has no module-name-specific processing strategy
  classes;
- processing component behavior is still tested for the modules above;
- the declaration layer can list all modules with custom execution-scope
  behavior;
- codemod stage 1 `--check` passes.

### Phase 3: Move Infrastructure Artifact Retention

Replace `_save_images_required_artifacts(...)` with an infrastructure declaration
query.

`PipelineGeneratorBuildStage` should ask skipped infrastructure module
declarations for retained artifacts:

```python
retained = tuple(
    declaration.infrastructure_artifact_requirements(module)
    for module in skipped_modules
)
```

This same surface should later support `ExportToDatabase` as a post-execution
export requirement rather than a generator special case.

Acceptance:

- `SaveImages` artifact retention behavior is unchanged;
- adding another infrastructure artifact consumer does not require editing
  `pipeline_generator.py`;
- skipped module handling uses declarations for both import notes and retained
  artifacts where practical;
- codemod stage 1 reports no remaining generator-owned infrastructure retention
  helper.

### Phase 4: Consolidate Per-Module Query Traffic

Refactor `PipelineGeneratorCodeEmitter.generate_steps_from_registry(...)` so it
does not separately know about:

- settings binding strategy registry;
- function resolution strategy registry;
- processing component strategy registry;
- artifact contract semantics.

Introduce a per-module generation context:

```python
@dataclass(frozen=True, slots=True)
class CellProfilerModuleGenerationPlan:
    module: ModuleBlock
    declaration: CellProfilerModuleDeclaration
    artifact_contract: ModuleArtifactContracts
    function_name: str
    bound_settings: BoundModuleSettings
    translated_kwargs: GeneratedStepSettings
    invocation_options: RuntimeInvocationOptions | None
    processing_components: ModuleProcessingComponents
```

Then the emitter receives already-lowered generation plans and only renders
source text.

Acceptance:

- source emission is separated from semantic lowering;
- tests can inspect `CellProfilerModuleGenerationPlan` without parsing generated
  Python source;
- generator comments that still mention "LLM-inferred category" are removed or
  replaced with declaration/query wording;
- codemod stage 1 and stage 2 `--check` pass for the migrated files.

### Phase 5: Advisor And Debt Checks

Run the advisor on touched backend files and make sure it does not report a new
semantic hub around the declaration facade.

Expected good shape:

- declaration facade delegates and aggregates;
- individual policies own actual module-specific behavior;
- generator owns sequencing and source rendering only;
- no replacement hub with many `if module_name` or string-switch behaviors.

## Test Plan

Focused tests:

- declaration query returns expected owners for representative modules:
  `GrayToColor`, `TrackObjects`, `CorrectIlluminationCalculate`, `SaveImages`,
  `ExportToDatabase`, and a default image operation.
- generation plans preserve function name, kwargs, invocation options,
  processing components, and artifact contracts for representative modules.
- `SaveImages` retained artifacts match current behavior.
- `CorrectIlluminationCalculate` grouping behavior matches current behavior for
  each supported calculation scope.
- source code generation is stable for at least one representative `.cppipe`
  fixture before and after the facade migration.

Integration checks:

- run existing CellProfiler generated pipeline execution tests;
- run source-schema ingestion/source-binding tests touched by generation scope;
- run CPA/export tests only if `ExportToDatabase` declaration behavior changes.

Advisor:

- run advisor on touched non-test backend files;
- do not require advisor cleanliness for test files unless tests are refactored
  to match backend architecture changes.

Codemod checks:

- run `python tools/codemods/cellprofiler_declaration_query.py --check`;
- run `python tools/codemods/cellprofiler_declaration_query.py --dry-run` and
  confirm it reports no pending safe rewrites before calling the refactor done;
- if the codemod reports ambiguous candidates, either resolve the architecture
  manually or add a focused invariant before applying a transform.

## Risks

- A declaration facade can become another god object if it owns behavior instead
  of routing to nominal policies.
- Moving strategies too early can make diffs noisy. First introduce the query
  facade, then migrate one behavior family at a time.
- Some module behavior is setting-derived, not static. Do not force it into
  static records; use typed setting-derived policy objects.
- Some semantics belong to backend callable metadata, but not all. Pipeline
  lowering behavior, source-axis behavior, and infrastructure export behavior
  should stay in CellProfiler interop declarations rather than being pushed into
  absorbed backend functions.
- Codemods can make an architectural leak look consistent. Every codemod stage
  must have an invariant and a test before applying broad rewrites.

## End State

`pipeline_generator.py` should read as:

1. load absorbed registry metadata;
2. partition executable and infrastructure modules;
3. compile source schema and symbol contracts through declarations;
4. build module generation plans by querying declarations;
5. prune dead artifacts using declaration-visible artifact requirements;
6. render source code from already-lowered plans.

The generator should not be the first place an engineer edits when a
CellProfiler module has new semantic behavior. The first place should be the
module declaration/policy layer.
