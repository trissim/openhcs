# Superseded: CellProfiler Runtime Execution, Import, And Dispatch Consolidation Plan

## Status And Relationship To The Runtime-Unification Plan

Date: 2026-07-13

Status: implementation plan produced from the current uncommitted working tree.
This document does not implement the redesign.

This is the required hard-cutover consolidation before the remaining
acceptance phase of the
[runtime-unification plan](cellprofiler_openhcs_runtime_unification_plan.md). The
earlier plan established the correct public boundary and nominal ownership
direction. Its Phases 0 through 2 are marked complete in the working tree, while
its official30 ZMQ, export, UI, and Napari Phase 3 remains pending. This plan
does not reopen that public-boundary decision. It replaces the implementation
order for the pending acceptance work: consolidation Phases 0 through 5 run
first, then this plan's Phase 6 runs the parity, export, UI, and Napari gates
once against the final architecture. It addresses the implementation depth
left behind by the cutover:

- .cppipe import and compile-time reconstruction advance the same artifact
  semantics through separate handwritten loops;
- source-axis facts are copied into an intermediate summary record and passed
  through module declarations;
- the compiled callable contract is wrapped by a runtime plan, an executor, and
  another runtime callable;
- image execution mode and ProcessingContract form two consecutive dispatch
  layers;
- runtime-plane values are projected by both a CellProfiler capability/wrapper
  path and the generic RuntimeSliceProjection path;
- errors expose the final backend assertion after a long call chain instead of
  the first violated projection boundary.

The audit verdict is direct: the redesign is substantially better at semantic
ownership and public transport than the pre-unification architecture, but the
runtime is still opaque. The opacity is no longer caused by hidden generated
state. It is caused by repeated reconstruction and stacked orchestration
layers around otherwise sound nominal authorities.

The required result is not a new consolidated runtime model. It is a shallower
connection among authorities that already exist.

### Implementation Order And Agent Handoff

An implementation agent working from the runtime-unification plan must stop
before its pending Phase 3 parity fixes and read this plan in full. The binding
order is:

1. preserve completed runtime-unification Phases 0 through 2 as the behavioral
   baseline;
2. implement this plan's Phases 0 through 5 as one hard architectural cutover;
3. implement no fix in a shell or projection layer that this plan deletes;
4. run this plan's Phase 6 once on the surviving architecture;
5. use that single Phase 6 result to discharge the earlier plan's pending
   Phase 3 gate.

The current ImagingFlow/FilterObjects failure is a characterization case, not
authorization to repair the soon-to-be-deleted dispatch chain. A fix lands on
the final CallableContract, ModuleArtifactContract, RuntimeSliceProjection,
CellProfilerModuleExecutor, or nominal module owner named here. There is no
first parity repair on the old path followed by a second repair after
consolidation.

### Hard-Cutover Interpretation

This plan is deliberately aggressive.

- KEEP means the named symbol is the active authoritative implementation with
  direct production consumers. It never means retained for compatibility,
  possible future use, rollback, or test convenience.
- COLLAPSE means every caller moves to the named owner and the old definition is
  deleted in that same phase.
- DELETE means definition, file, import, re-export, fixture, test construction,
  serialized spelling, documentation claim, environment switch, and dormant
  branch all disappear in that same phase.
- There is no deprecation window, dual-read period, dual-write period, feature
  flag, migration adapter, old constructor shape, alias, or rollback path.
- Generated source, compiled artifacts, caches, benchmark outputs, and fixtures
  are regenerated. Nothing reads an historical shape.
- A currently passing test has no preservation authority. If it asserts a
  superseded shell, the test is replaced with an assertion over the surviving
  owner.
- A local cache, wrapper, request, or strategy survives only when this plan
  names it as the final owner and identifies its direct production
  responsibility. There is no keep-in-case category.
- Adding a method to an existing owner is not the default. First reuse its
  existing query, registry, strategy, or graph operation. A method is added
  only for a responsibility that owner already possesses and no current method
  exposes; this plan names every such exception explicitly.

### Net-New Production Surface Budget

The only additive production API surface authorized by this plan is on the
already existing ArtifactDeclarationStepContext:

- group_by;
- input_source;
- available_artifacts;
- main_flow_artifacts;
- available_artifact_producers;
- with_source_declarations(source_specs), which composes the existing
  ArtifactSpecCollection and exact input_source semantics;
- advance_artifact_graph(graph, next_main_flow_artifacts), which composes the
  existing ArtifactGraph producers/outputs and installs the caller's exact
  next main-flow collection.

No other additive production class, dataclass, enum, registry, strategy, request,
context, plan, cache, debug field, executor entry point, or contract query is
added. Every other production change deletes a layer, narrows an existing
shape, replaces copied fields with an already compiled authority, moves callers
to an existing authority, or changes an existing method's signature.

## Audited Working-Tree Snapshot

The snapshot used for this plan had:

- 568 tracked files changed;
- 57,863 tracked insertions and 96,496 tracked deletions;
- 272 untracked files;
- 47 Python files and 20,693 lines under
  openhcs/interop/cellprofiler/runtime;
- 332 top-level class or function declarations in that runtime package;
- a 1,753-line module_execution.py;
- a 1,597-line adapter.py;
- an 823-line function_contract_execution.py;
- a 680-line pipeline_import.py;
- a 321-line compile_time_contracts.py;
- a 350-line module_processing_config.py;
- a 4,680-line module_declarations.py;
- an 18,500-line test_cellprofiler_module_execution.py.

These counts are evidence of review surface, not deletion quotas. A phase does
not pass because a file becomes shorter. It passes only when an authority and
its callers become singular and the behavioral gates pass.

The audit traced current code rather than the committed baseline. It included:

1. .cppipe parsing through import_cellprofiler_pipeline;
2. public FunctionStep construction and sparse kwargs reconstruction;
3. CellProfilerInvocationContractProviderFactory session precomputation;
4. generic ArtifactGraph and CompiledFunctionInvocation construction;
5. FunctionCoreExecutor adapter binding and callable resolution;
6. CellProfiler runtime callable, runtime plan, executor, image request, and
   invocation request construction;
7. image-execution-mode and ProcessingContract dispatch;
8. runtime-slice and source-binding projection;
9. adapter artifact reads, writes, and query caches;
10. output matching and recording;
11. generic debug events and the CP-only pure-2D trace hook;
12. generated-source, ZMQ, integration, and official30 acceptance tests.

Before accepting a replacement, the audit searched the required existing
ownership patterns:

- __registry__;
- AutoRegisterMeta;
- RegistryConfig;
- RegistryFamily;
- MostDerivedContextStrategyMixin;
- NominalTypeKeyedStrategyMixin;
- EnumKeyedStrategyMixin.

The audit found existing owners for every proposed responsibility. No new
public or private semantic carrier, plan record, strategy family, registry,
capability enum, compatibility shim, or dispatch table is authorized by this
plan.

## Fixed Outcome

The completed architecture must satisfy all of the following:

- The public import result remains exactly list[FunctionStep] plus
  PipelineConfig.
- A generated source file contains only reconstructable public declarations.
- CellProfilerModule.__registry__ remains the sole module declaration
  authority.
- CallableContract remains the sole compiled callable-semantic authority at
  runtime.
- ModuleArtifactContract and its partitions remain the sole module artifact
  authority.
- ArtifactDeclarationStepContext becomes the one forward declaration context
  used by import reconstruction and the compiler provider. No second cursor
  record is introduced.
- RuntimeAdapterRequest remains the one invocation-scoped bridge from generic
  FunctionStep execution to the CellProfiler adapter.
- RuntimePlaneProjection and RuntimePlaneAxisValueProjection remain the
  execution-axis coordinates.
- RuntimeSliceProjection and its nominal value strategies perform every actual
  value projection, including image and object-label values.
- CurrentSourcePayloadPlaneSelectionAuthority remains the owner of
  source-identity-to-plane selection. It selects an index; it does not project
  a payload itself.
- ProcessingContract remains the declared processing semantic. The interaction
  between it and ImagePayloadExecutionMode is visible at one executor method,
  validated before a raw callable is reached, and is not represented by a
  second registered strategy family.
- CellProfilerRuntimeAdapter remains an invocation-scoped adapter over
  RuntimeAdapterRequest. It does not choose execution mode, select a projection
  capability set, or retain copied contract state.
- Backend module declarations retain genuine module-specific behavior through
  inheritance and MRO. Generic core imports no concrete CellProfiler module.
- Output recording continues to use exact compiled output plans and nominal
  ArtifactType strategies. This consolidation does not replace the measurement
  or output artifact model.
- A projection or dispatch mismatch fails at the executor boundary with the
  step, invocation, module, artifact spec, value type, declared axis, selected
  plane, and cardinalities. The raw backend leaf is never the first code to
  discover an unprojected runtime value.

## Atomic Cutover Rule

Every phase below is atomic. In the same change it must:

1. add or update the behavioral and AST gates;
2. migrate every production caller;
3. migrate generated-source reconstruction and ZMQ reconstruction;
4. migrate focused tests and fixtures;
5. delete the replaced definitions, imports, exports, and files;
6. prove the deletion with AST and rg gates;
7. run the phase-specific behavior tests.

No phase may retain:

- a forwarding alias;
- an old and new field at the same time;
- a compatibility constructor;
- a fallback lookup;
- a default that silently preserves the old path;
- a parallel registry;
- a copied capability set;
- a string-based module dispatch;
- a concrete-backend import in generic core;
- an explanatory request or plan object that repeats state already held by
  CallableContract, ModuleArtifactContract, RuntimeAdapterRequest, or
  RuntimeFunctionInvocationRequest.

Tests that assert the shape of a deleted shell are deleted or rewritten to
assert the surviving authority. Keeping a shell only to satisfy its current AST
test is not an acceptable migration.

## Verified Current Paths

### .cppipe Import Path

The current import path is:

~~~text
import_cellprofiler_pipeline
  -> CPPipeParser.parse
  -> _public_pipeline
     -> CellProfilerModule.source_bindings_for_modules
     -> initialize available_artifacts
     -> initialize main_flow_artifacts
     -> initialize available_artifact_producers
     -> collect enabled CellProfilerModule declarations
     -> batch adjacent equal module types
     -> _lower_module_batch
        -> CellProfilerModule.artifact_contract
        -> CellProfilerModule.resolve_function with resolved source bindings
        -> CallableContract.from_callable
        -> SettingsBinder
        -> CellProfilerModule.processing_config with inherited ProcessingConfig
        -> combine same-module contracts
        -> _sparse_kwargs_for_contract
           -> repeatedly call invocation_artifact_contract
     -> ArtifactDeclarationStepContext.advance_artifact_graph
     -> derive common pipeline ProcessingConfig
     -> construct public FunctionSteps
~~~

The public boundary is correct. Import advances the same immutable artifact
context used by the compiler provider and does not construct an axis summary or
repeat processing-contract reconstruction.

### Compile-Time Provider Path

The current compiler-provider path is:

~~~text
PipelineInvocationContractProviderAuthority.provider_for_session
  -> CellProfilerInvocationContractProviderFactory.provider_for_session
     -> initialize available_artifacts
     -> initialize main_flow_artifacts
     -> initialize available_artifact_producers
     -> walk StepSnapshots in order
     -> compile source specs
     -> advance native callable outputs and main flow
     -> resolve CellProfilerModule by canonical callable
     -> derive source groups
     -> CellProfilerModule.invocation_artifact_contract
     -> enrich CallableContract
     -> create InvocationContractPlan
     -> advance available artifacts
     -> advance producers
     -> advance main flow
  -> CellProfilerInvocationContractProvider exact-key lookup
~~~

The immutable exact-key provider and generic unique-claim provider ABI are
correct. The provider's handwritten forward traversal duplicates the import
loop.

### Generic Runtime Path

The generic path is:

~~~text
FunctionRuntimeScope.execute_chain
  -> FunctionCoreExecutor
  -> RuntimeAdapterRequest.from_runtime_scope
  -> RuntimeAdapterSpec.factory
  -> FunctionInvocationCallableResolver
  -> RuntimeAdapterSpec.runtime_callable_factory
  -> resolved runtime callable
~~~

This is the correct generic boundary. It already supplies:

- the compiled CallableContract through CompiledFunctionInvocation;
- exact artifact input and output plans;
- source binding plan and source binding runtime context;
- group and execution-axis scope;
- RuntimePlaneProjection;
- variable components;
- source load plan;
- generic debug cursor and artifact projections.

The CellProfiler layer must consume those values directly rather than rebuild a
parallel plan.

### Current CellProfiler Runtime Path

The current standard-image path is:

~~~text
RuntimeAdapterSpec.runtime_callable_factory
  -> CellProfilerRuntimeCallable
  -> CellProfilerModuleRuntimePlan.build
  -> CellProfilerModuleExecutor.run
  -> active input and output selection
  -> CellProfilerImageRequest
  -> runtime input binding
  -> RuntimeFunctionInvocationRequest
  -> CellProfilerFunctionContractExecutor.execute
  -> ImagePayloadExecutionMode selection
  -> CellProfilerImageExecutionStrategy.for_mode
  -> Natural, FullStack, or AlignedMultiImage strategy
  -> ProcessingContract.execute, or a mode-specific bypass
  -> pure-2D, pure-3D, flexible, or volumetric executor
  -> RuntimeSliceProjection for internal slice batches
  -> raw backend callable
  -> output aggregation
  -> RuntimeReturnedOutputMatcher
  -> CellProfilerOutputRecorder
  -> main-flow selection
~~~

This path is semantically explicit in isolated files but difficult to follow as
one execution. In particular, the compiled contract is available before the
first CellProfiler call and is nevertheless re-read from the raw function at
least twelve times across the runtime package.

### Current Projection Path

Projection is currently split across three mechanisms:

1. RuntimePlaneProjection and RuntimePlaneAxisValueProjection represent the
   generic execution coordinates.
2. RuntimeSliceProjection and its nominal strategies project images,
   RuntimeSliceAlignedValueSet values, ObjectLabelValue values, measurement
   tables, relationships, grids, and sequences.
3. The CellProfiler-specific projection layer derives a
   RuntimePlaneProjectionCapability set, wraps it in
   CurrentRuntimePlaneKwargProjectionContract, projects selected kwargs through
   CurrentRuntimePlaneKwargProjection, and separately projects image and
   object-label payloads through classes in runtime/projection.py.

The CellProfiler kwarg projector reimplements the same cardinality and
ObjectLabelValue or RuntimeSliceAlignedValueSet decisions already owned by
RuntimeSliceProjectionStrategy leaves. The image and object projection classes
also perform value projection after a source-plane selection that can be
expressed as RuntimePlaneAxisValueProjection.

### Current Failure Evidence

The latest substantive ImagingFlow ZMQ failure captured during the audit is in
/tmp/openhcs-zmq-imagingflow-r23.log:

~~~text
FilterObjects requires one explicitly projected 2-D object-label plane.
~~~

The traceback reaches that assertion through:

~~~text
FunctionCoreExecutor.invoke
  -> CellProfilerRuntimeCallable.__call__
  -> CellProfilerModuleExecutor.run
  -> _run_standard_image
  -> CellProfilerFunctionContractExecutor.execute
  -> CellProfilerImageExecutionStrategy.execute
  -> ProcessingContract.execute
  -> FlexibleProcessingContract.execute
  -> ProcessingContract.PURE_2D.execute
  -> execute_pure_2d
  -> execute_pure_2d_slice_batch
  -> RuntimePure2DSliceBatchRequest
  -> execute_pure_2d_slice
  -> raw filter_objects
~~~

The failure is useful architectural evidence even if its immediate data bug is
fixed before this plan is implemented. A declared object-label axis survived
to the final backend leaf without the owning execution boundary proving that
one 2-D plane had been selected. That is the exact kind of late failure this
consolidation must prevent.

## Findings

### Finding 1: Import And Compiler Mirror One Forward Artifact State

pipeline_import.py and compile_time_contracts.py both own mutable local versions
of:

- available_artifacts;
- main_flow_artifacts;
- available_artifact_producers.

It does not own `ProcessingConfig`; processing lowering receives the inherited
concrete config separately from the resolved public step or import config
context.

Both append declared outputs, derive producer groups, construct
ArtifactProducer values, and replace or preserve main flow. The importer also
runs the same advancement inside a tentative batch and again after accepting
the batch.

This is a semantic mirror, not harmless local iteration. A change to producer
group ownership, unnamed native main flow, source reset behavior, or
participation in main flow can make import reconstruction disagree with
compilation.

Decision: extend the existing ArtifactDeclarationStepContext into the single
forward declaration context. Do not add ArtifactCursor,
CellProfilerImportState, ImportCompilationPlan, or another record.

The context must own:

- available ArtifactSpecCollection;
- current main-flow ArtifactSpecCollection;
- available ArtifactProducer tuple, which it already partially owns;
- the resolved source bindings and processing config it already carries;
- with_source_declarations and advance_artifact_graph, the two exact operations
  named in the net-new surface budget.

The operations accept existing ArtifactSpecCollection and ArtifactGraph values
plus exact next-main-flow specs. They do not accept a module name and do not
import CellProfiler. The nominal
CellProfilerModule still decides whether its image outputs replace main flow;
the generic context performs the one advancement.

The importer and provider perform distinct traversals because import discovers
public declarations before a CompilationSession exists. Both traversals call
the same ArtifactDeclarationStepContext advancement methods; neither contains
an independent advancement algorithm.

### Finding 2: Processing Axes Already Have Nominal Owners

The removed source-axis summary copied facts from resolved source bindings,
artifact contracts, and current main flow. It had no independent semantic
authority.

The surviving ownership is exact:

- `CellProfilerModule.resolve_function` receives the artifact contract and
  resolved `StepSourceBindingsConfig` needed for import-only variant selection.
- `ModuleArtifactContract` owns ordered artifact inputs and outputs only.
- `CallableContract` alone owns required variable components and allowed
  groupings.
- `CellProfilerModule.processing_config` receives the artifact contract,
  callable contract, and inherited concrete `ProcessingConfig` and returns the
  lowered concrete config.
- ObjectState-resolved public steps provide inherited compiler values directly;
  parsed import resolves the same inherited value in the pipeline config
  context.

No replacement axis plan, component request, semantic context, copied boolean
tuple, or helper file is authorized.

### Finding 3: Runtime Rewraps The Compiled Contract

CellProfilerModuleRuntimePlan stores raw_func, module_type, and
callable_contract. Its properties expose data already on CallableContract and
ModuleArtifactContract. CellProfilerRuntimeCallable stores only a
CellProfilerModuleExecutor. The executor stores only the plan.

This is a three-object chain around two existing inputs. Downstream request
types then accept the runtime plan, the raw function, and sometimes a copied
function name.

Decision:

- RuntimeAdapterSpec.runtime_callable_factory returns a callable
  CellProfilerModuleExecutor directly.
- CellProfilerModuleExecutor stores exactly raw_func and the compiled
  callable_contract.
- CellProfilerModuleExecutor.__post_init__ validates canonical callable,
  module registry ownership, module name, and source ProcessingContract once.
- module_type is a property resolved through CellProfilerModule's existing
  registry. It is not copied into another record.
- contract and processing_contract are direct CallableContract queries.
- active partition selection uses the existing
  ModuleArtifactContract.specs_for_partition, require_items_for_specs,
  has_ref_for_partition, declared_input_specs, declared_outputs, and outputs
  queries. No new partition selector is added.
- CellProfilerOutputRecordRequest replaces runtime_plan, func, and
  function_name with one compiled callable_contract.
- module declaration runtime hooks receive exact existing values rather than a
  CellProfilerModuleRuntimePlan parameter.
- CellProfilerMeasurementImageResolver loses its executor field. Its
  orchestration methods collapse into CellProfilerModuleExecutor; genuine
  measurement-image value projection stays on the existing
  CellProfilerMeasurementImage and measurement policy owners.

### Finding 4: Output Aggregation Copies Contract Facts

CellProfilerFunctionOutputAggregationContract stores:

- whether the main output replaces runtime flow;
- declared output specs.

Both are derived from the compiled ModuleArtifactContract and nominal module
declaration. The record is then cloned through with_output_aggregation_contract
and passed into the contract executor.

Decision: delete CellProfilerFunctionOutputAggregationContract. Do not replace
it with new ModuleArtifactContract methods. Reuse
ModuleArtifactContract.main_flow_return_specs for the canonical return,
declared_outputs and outputs for declared and recorded partitions, and
artifact_spec_participates_in_main_flow for the existing artifact-type policy.
The auxiliary outputs are the exact ordered remainder of declared_outputs
after main_flow_return_specs; that composition stays executor-local and is not
named as another semantic query. CellProfilerFunctionContractExecutor receives
the compiled CallableContract and derives output aggregation from its required
ModuleArtifactContract. Every CP runtime invocation, including tests, supplies
that compiled module contract. Delete raw-callable test branches that execute
without it.

### Finding 5: Image Mode And Processing Contract Form Two Dispatch Layers

CellProfilerFunctionContractExecutor first resolves
ImagePayloadExecutionMode, then dispatches through a registered
CellProfilerImageExecutionStrategy. The natural strategy reconstructs
CallableContract from the raw function and dispatches again through
ProcessingContract. Full-stack and aligned-stack strategies bypass that second
dispatch and call separate executor methods.

The current structure hides the actual decision matrix across three files and
two nominal families.

Decision:

- Keep ImagePayloadExecutionMode as the typed description of the resolved image
  payload.
- Keep ProcessingContract and its declaration classes as callable semantics.
- Delete CellProfilerImageExecutionStrategy and its three leaves.
- Delete requested_image_execution_mode and the force_full_stack boolean path.
  Callers provide a typed execution_mode only.
- Reuse the existing CellProfilerFunctionContractExecutor.execute method as the
  one visible decision point. Change its signature to accept the already
  compiled CallableContract, resolved image, kwargs, typed mode, exact plane
  projection, and exact output contract.
- The method validates the closed mode/contract matrix and then selects exactly
  one execution method. It does not call CallableContract.from_callable.

The required matrix is:

| ImagePayloadExecutionMode | ProcessingContract handling |
| --- | --- |
| NATURAL | Delegate once to the compiled ProcessingContract declaration. |
| FULL_STACK | Execute the raw callable once over the preserved payload after the compiled contract proves the mode is allowed. |
| ALIGNED_MULTI_IMAGE_STACK | Execute one aligned slice batch and aggregate once; reject PURE_3D before invocation. |

FLEXIBLE remains owned by its existing semantic-control parameter declaration.
The executor injects or consumes that control once. The matrix must not become
a dict keyed by strings, a tuple priority table, or a new strategy registry.

### Finding 6: Projection Is Both Capability Dispatch And Value Dispatch

The capability family decides whether to project artifact images and runtime
kwargs. The value strategy family decides how to project the actual values.
The capability leaves for natural and full-stack batch execution currently
return the same two capabilities. The contract wrapper re-queries the family
several times during one invocation.

Decision: projection timing is derived from existing runtime coordinates and
the resolved execution mode. Projection mechanics remain exclusively on
RuntimeSliceProjection.

The exact rules are:

1. If RuntimeAdapterRequest.plane_projection selects one RUNTIME_SLICE plane,
   adapter-resolved image and kwarg values with that declared axis are projected
   once before execution.
2. If the request preserves the RUNTIME_SLICE stack and NATURAL execution
   delegates to PURE_2D, values remain intact through adapter binding and are
   projected once inside the pure-2D slice loop.
3. If the request preserves the stack and FULL_STACK is selected, values remain
   intact for the one raw call.
4. ALIGNED_MULTI_IMAGE_STACK uses one projection per aligned slice for the
   composed image and every aligned kwarg.
5. SOURCE_BINDING projection occurs only after
   CurrentSourcePayloadPlaneSelectionAuthority selects an exact source plane.
   The selected index is converted to RuntimePlaneAxisValueProjection and the
   value is projected by RuntimeSliceProjection.
6. Values with no declared matching plane axis pass through. Array rank never
   invents an axis.
7. Any declared count conflict fails before the raw callable.

Delete:

- RuntimePlaneProjectionCapability;
- RuntimePlaneProjectionRequirementContext;
- RuntimePlaneProjectionRequirement and its leaves;
- CurrentRuntimePlaneKwargProjectionContract;
- CurrentRuntimePlaneKwargProjection;
- RuntimePlaneImagePayloadProjection;
- CurrentSourceImagePayloadProjection;
- CurrentSourceObjectLabelPayloadProjection;
- CurrentSourcePayloadPlaneSelector.
- CellProfilerImageRequest.source_axis_payload;
- CellProfilerImageRequest.composed_source_payload_for_name;
- CellProfilerImageRequest.source_payloads_for_names.

Keep:

- CurrentSourcePayloadPlaneSelection;
- CurrentSourcePayloadPlaneSelectionRequest;
- CurrentSourcePayloadPlaneSelectionAuthority;
- RuntimePlaneAxis;
- RuntimePlaneProjection;
- RuntimePlaneAxisValueProjection;
- RuntimeSliceProjectionStrategy and all genuine nominal value leaves;
- RuntimeSliceProjection.

CurrentImageObjectLabelPlaneAlignment remains only for the distinct operation
that aligns two multi-plane source-identity sequences. It calls the source
identity authority directly and delegates each actual label projection to
RuntimeSliceProjection.

All surviving CP runtime call sites in adapter.py, artifact_binding.py,
invocation.py, and module_execution.py must stop calling
RuntimePlaneAxisValueProjection.project or AlignedImageStack.project_plane_axis
directly. They construct or obtain the existing projection coordinate and pass
the value to RuntimeSliceProjection.value_for_slice. The generic strategy
registry remains the only value-type dispatch.

### Finding 7: CellProfilerImageRequest Stores Derived Policy Flags

CellProfilerImageRequest has genuine source identity state:

- payload;
- source_image_name;
- source_aliases;
- image_count;
- execution_mode;
- plane_projection.

It also stores projects_runtime_slice_kwargs and
publishes_side_effect_main_flow. Those booleans are derived from the execution
mode, plane projection, active output specs, and artifact kinds.

Decision: retain the genuine request fields and delete both booleans.

- Projection timing follows the rules above.
- Side-effect main-flow publication is decided at the existing main-flow call
  site from active output specs and input artifact kinds.
- CellProfilerSideEffectMainFlowPolicy accepts the existing
  ModuleArtifactContract and derives the exact publication decision from its
  input and output partitions. It does not receive a copied boolean.

### Finding 8: The Adapter Is A Data Bridge And A Policy Hub

CellProfilerRuntimeAdapter correctly owns no parallel image, object, or
measurement store. Its request points to the generic RuntimeValueStore and VFS
plans. It nevertheless also:

- accepts projection capability sets;
- constructs CP-specific projection wrappers;
- owns execution-adjacent source-plane projection decisions;
- carries eight local query caches;
- exposes cache invalidation methods called by a separate policy family;
- performs source-candidate caching in addition to process-level source
  authorities.

Decision: narrow the adapter and delete every adapter-local query cache. The
adapter is invocation-scoped; RuntimeValueStore and the existing registered
process caches already own reusable query lifetime.

Required changes:

- resolve_source_image and source object reads return typed values with declared
  axes; they do not accept projection capabilities;
- remove image_payload_for_current_runtime_plane once executor projection uses
  RuntimeSliceProjection directly;
- retain RuntimePlaneAxisProjector as the typed provider of execution and
  source-binding coordinates;
- retain exact artifact read and write operations backed by
  RuntimeAdapterRequest and RuntimeValueStore;
- delete _image_cache, _current_image_cache, _object_cache,
  _current_object_cache, _measurement_cache,
  _artifact_availability_cache, _source_candidate_cache, and
  _source_candidate_process_key_cache;
- route artifact record lookup through RuntimeValueStore's revision-keyed query
  cache;
- do not replace image/object composition caches. Recompose from the cached
  RuntimeValueStore records. This plan authorizes no replacement cache, and
  performance work must not resurrect adapter-local state;
- route measurement reuse through the existing store-bound object-measurement
  table caches and MeasurementLabelSliceFeatureBatchQueryCache;
- route source-candidate reuse through the existing
  SOURCE_CANDIDATE_PROCESS_CACHE, SOURCE_CANDIDATE_MATCH_PROCESS_CACHE,
  SOURCE_CANDIDATE_PATH_PROCESS_CACHE, and
  SOURCE_CANDIDATE_METADATA_PROCESS_CACHE owners in source_candidates.py;
- delete RuntimeArtifactCacheInvalidationPolicy and its leaves after the
  adapter-local caches disappear;
- do not add an AdapterContext, RuntimeWorkspace, RuntimeState, or cache facade;
- keep no invocation-local semantic cache on CellProfilerRuntimeAdapter.

The deletion is based on ownership, not reference count. Genuine registry
leaves elsewhere remain when registry construction and MRO selection are their
active production dispatch. The adapter cache invalidation family loses its
entire responsibility and is deleted.

### Finding 9: Diagnostics Observe The Outer Invocation, Not The Projection Boundary

FunctionCoreExecutor already owns DebugCursor, DebugArtifactRefProjection, and
DebugEvent emission. The CP pure-2D executor separately writes JSON lines under
OPENHCS_PURE2D_SLICE_TRACE_PATH using Pure2DTraceArrayStats.

Decision:

- delete the CP-only environment trace and Pure2DTraceArrayStats;
- keep generic RuntimeProfileSink and CellProfilerRuntimeProfileLogger for
  timing;
- move the generic BEFORE_INVOCATION observation to the point after runtime
  parameters and adapter inputs are bound, so the existing
  DebugInvocationParameter projection can summarize the actual call boundary;
- retain DebugArtifactRefProjection's existing artifact identity and populate
  its existing shape and dtype fields from bound artifact values;
- keep DebugInvocationParameter's existing name and value_repr fields. Replace
  unbounded repr construction with a bounded deterministic string assembled
  from existing value type, image/object metadata, and
  RuntimePlaneAxisValueProjection coordinates. It does not repeat artifact
  identity already carried by DebugArtifactRefProjection. Do not add summary
  fields or another debug value carrier;
- include the main image under an explicit parameter name in the existing
  invocation_parameters tuple;
- use the existing DebugEvent and DebugCursor. Do not add a CP debug event,
  projection trace record, execution explanation record, or sidecar file.

Every projection exception must be self-contained even when debugging is off.
The message must include:

- step index and name;
- FunctionInvocationKey;
- CellProfiler module and callable;
- artifact spec ref or kwarg name;
- nominal value type;
- declared RuntimePlaneAxis;
- selected plane index or preserved-stack status;
- declared value cardinality;
- execution-axis cardinality;
- effective ImagePayloadExecutionMode;
- ProcessingContract.

### Finding 10: Existing Tests Preserve Shell Shape

Current AST tests require:

- CellProfilerModuleRuntimePlan to have exactly three fields;
- CellProfilerModuleExecutor to have exactly one plan field;
- CellProfilerRuntimeCallable to retain exactly one executor;
- module policy hooks to accept CellProfilerModuleRuntimePlan.

Those tests correctly prevented prior state mirroring, but after this audit they
freeze the remaining wrapper chain.

Decision: replace them with gates over the surviving owners:

- executor fields are exactly raw_func and callable_contract;
- no runtime plan or runtime callable class exists;
- no CP runtime file calls CallableContract.from_callable;
- no runtime hook accepts a plan carrier;
- ModuleArtifactContract performs exact partition/ref selection;
- RuntimeSliceProjection is the only value projection call target;
- the mode/contract decision exists in one executor method;
- deleted projection and strategy files are absent.

## Authority Map

| Responsibility | Retained authority | Allowed change |
| --- | --- | --- |
| Public pipeline declaration | PipelineConfig and FunctionStep | None to shape |
| Parsed CP module ownership | CellProfilerModule.__registry__ | Narrow method signatures |
| Callable identity, required axes, and allowed grouping | CallableContract | Consume compiled instance directly |
| Module artifact semantics | ModuleArtifactContract | Artifact items, partitions, refs, declared outputs, and main-flow returns only |
| Resolved step configuration | ObjectState-resolved FunctionStep | Read through StepSnapshot.step directly |
| Invocation contract extension | InvocationContractPlan and provider factory ABI | Keep exact provider map |
| Forward declaration context | ArtifactDeclarationStepContext | Exact source bindings, group_by, input_source, available artifacts/producers, main flow, and generic advancement |
| Per-step artifact graph | ArtifactGraph | Reuse producers and outputs during advancement |
| Compiled invocation | CompiledFunctionInvocation | No CP sidecar |
| Runtime ABI injection | RuntimeCallableArgumentPlan and RuntimeAdapterSpec | Factory returns executor directly |
| Runtime coordinates | RuntimeAdapterRequest | No copied CP request |
| Runtime artifact storage | RuntimeValueStore and compiled plans | No adapter-local semantic store |
| Processing semantics | ProcessingContract | One compiled dispatch decision |
| Image payload mode | ImagePayloadExecutionMode | Keep typed closed enum |
| Plane coordinates | RuntimePlaneProjection and RuntimePlaneAxisValueProjection | Reuse |
| Value projection | RuntimeSliceProjectionStrategy registry and RuntimeSliceProjection | Sole mechanics |
| Source identity selection | CurrentSourcePayloadPlaneSelectionAuthority | Select index only |
| Output matching | RuntimeReturnedOutputMatcher | Reuse |
| Output writing | CellProfilerOutputRecorder nominal family | Reuse |
| Debug cursor and event | DebugCursor, DebugEvent, DebugInvocationParameter | Bound existing value_repr projection; no new fields or CP sidecar |
| Runtime profiling | RuntimeProfileSink and CellProfilerRuntimeProfileLogger | Timing only; no semantic trace state |

## Symbol Disposition

### Keep Without A Parallel Replacement

| Symbol | Reason |
| --- | --- |
| import_cellprofiler_pipeline | Sole public .cppipe import operation |
| CellProfilerModule and its registered subclasses | Nominal module semantics and MRO |
| SettingsBinder | Parsed setting-to-public-kwarg boundary |
| CallableContract | Compiled callable authority |
| ModuleArtifactContract and partition types | Artifact authority |
| InvocationContractPlan | Existing compiler replacement ABI |
| CellProfilerInvocationContractProvider | Exact immutable session lookup |
| CellProfilerInvocationContractProviderFactory | Registered compiler extension |
| PipelineInvocationContractProviderAuthority | Unique provider resolution |
| ArtifactDeclarationStepContext | Existing forward context, extended rather than mirrored |
| ArtifactGraph and ArtifactProducer | Generic producer/consumer authority |
| RuntimeAdapterSpec and RuntimeAdapterRequest | Generic runtime bridge |
| RuntimeFunctionInvocationRequest | Exact prepared invocation values |
| CellProfilerRuntimeAdapter | Narrow artifact/source adapter |
| ProcessingContract | Declared execution semantics |
| ImagePayloadExecutionMode | Typed payload interpretation |
| RuntimePlaneAxis | Typed plane domain |
| RuntimePlaneProjection | Execution-owned projection coordinates |
| RuntimePlaneAxisValueProjection | Exact value projection coordinates |
| RuntimeSliceProjectionStrategy | Nominal value projection family |
| RuntimeSliceProjection | Sole value projector |
| CurrentSourcePayloadPlaneSelectionAuthority | Exact source identity selection |
| RuntimeReturnedOutputMatcher | Returned value to declared spec matching |
| CellProfilerOutputRecorder | ArtifactType-polymorphic recording |
| DebugCursor, DebugEvent, DebugInvocationParameter | Generic debug substrate |

### Collapse Into Existing Owners

| Current symbol or behavior | Owning destination |
| --- | --- |
| _advance_artifact_cursors | ArtifactDeclarationStepContext advancement |
| _processing_group_keys | existing source_binding_group_keys_for_group_by plus exact normalization |
| Source declaration projections copied into processing summaries | Resolved StepSourceBindingsConfig at import-only callable selection |
| Artifact facts copied into processing summaries | Direct ModuleArtifactContract artifact queries |
| Default processing helper | CellProfilerModule.processing_config base behavior over contract, callable contract, and inherited config |
| CellProfilerModuleRuntimePlan active spec queries | ModuleArtifactContract exact partition/ref selection plus executor-local composition |
| CellProfilerRuntimeCallable.__call__ | CellProfilerModuleExecutor.__call__ |
| CellProfilerMeasurementImageResolver orchestration | CellProfilerModuleExecutor and existing measurement value/policy owners |
| CellProfilerFunctionOutputAggregationContract | Existing ModuleArtifactContract main_flow_return_specs/partitions plus executor-local ordered remainder |
| CurrentSourcePayloadPlaneSelector | RuntimeRecordSourceImageSetSelector and CurrentSourcePayloadPlaneSelectionAuthority |
| CP image/object projection mechanics | RuntimeSliceProjection |
| projects_runtime_slice_kwargs decision | resolved mode plus RuntimeAdapterRequest.plane_projection |
| publishes_side_effect_main_flow decision | active contract at main-flow call site |
| CP pure-2D trace | existing DebugEvent invocation parameter summaries |

### Delete

The following definitions must not survive their owning phase:

<!-- cellprofiler-runtime-consolidation-forbidden-symbols:start -->
~~~text
StepConfigUniverse
SourceProcessingAxisPlan
_advance_artifact_cursors
_processing_group_keys
CellProfilerRuntimeCallable
CellProfilerModuleRuntimePlan
LibraryWatershedInvocationRequest
CellProfiler4WatershedInvocationRequest
watershed_library_with_one_special_input
watershed_library_with_two_special_inputs
watershed_library_with_three_special_inputs
watershed_cellprofiler4_with_one_special_input
watershed_cellprofiler4_with_two_special_inputs
watershed_cellprofiler4_with_three_special_inputs
measure_image_area_occupied_binary
measure_image_area_occupied_objects
measure_image_intensity_masked
CropSpecialInputPolicy
CroppingObjectLabelInputPolicy
crop_with_mask
IdentifyObjectsInGridVariant
identify_objects_in_grid_with_guides
CellProfilerFunctionOutputAggregationContract
CellProfilerMeasurementImageResolver
CellProfilerImageExecutionStrategy
NaturalImageExecutionStrategy
FullStackImageExecutionStrategy
AlignedMultiImageStackExecutionStrategy
requested_image_execution_mode
force_full_stack
RuntimePlaneProjectionCapability
RuntimePlaneProjectionRequirementContext
RuntimePlaneProjectionRequirement
NoRuntimePlaneProjectionRequirement
NaturalRuntimePlaneProjectionRequirement
FullStackBatchRuntimePlaneProjectionRequirement
CurrentRuntimePlaneKwargProjectionContract
CurrentRuntimePlaneKwargProjection
CurrentSourcePayloadPlaneSelector
RuntimePlaneImagePayloadProjection
CurrentSourceImagePayloadProjection
CurrentSourceObjectLabelPayloadProjection
CellProfilerImageRequest.source_axis_payload
CellProfilerImageRequest.composed_source_payload_for_name
CellProfilerImageRequest.source_payloads_for_names
CellProfilerImageRequest.projects_runtime_slice_kwargs
CellProfilerImageRequest.publishes_side_effect_main_flow
CellProfilerRuntimeAdapter._image_cache
CellProfilerRuntimeAdapter._current_image_cache
CellProfilerRuntimeAdapter._object_cache
CellProfilerRuntimeAdapter._current_object_cache
CellProfilerRuntimeAdapter._measurement_cache
CellProfilerRuntimeAdapter._artifact_availability_cache
CellProfilerRuntimeAdapter._source_candidate_cache
CellProfilerRuntimeAdapter._source_candidate_process_key_cache
RuntimeArtifactCacheInvalidationPolicy
ImageRuntimeArtifactCacheInvalidationPolicy
ObjectLabelRuntimeArtifactCacheInvalidationPolicy
MeasurementRuntimeArtifactCacheInvalidationPolicy
RelationshipRuntimeArtifactCacheInvalidationPolicy
RuntimeArtifactInputProjection
RuntimeArtifactScopeTarget
ComponentGroupProjection
object_label_source_payloads
has_non_object_artifact_inputs
OPENHCS_PURE2D_SLICE_TRACE_PATH
_trace_pure_2d_slice
Pure2DTraceArrayStats
~~~
<!-- cellprofiler-runtime-consolidation-forbidden-symbols:end -->

The following files are deleted when their last listed authority is removed:

<!-- cellprofiler-runtime-consolidation-forbidden-files:start -->
~~~text
openhcs/core/pipeline/step_config_universe.py
openhcs/interop/cellprofiler/module_processing_config.py
openhcs/interop/cellprofiler/runtime/image_execution_strategies.py
openhcs/interop/cellprofiler/runtime/projection_requirements.py
openhcs/interop/cellprofiler/runtime/runtime_plane_kwargs.py
openhcs/interop/cellprofiler/runtime/projection.py
openhcs/interop/cellprofiler/runtime/runtime_artifact_cache_invalidation.py
tests/unit/test_runtime_plane_kwargs.py
tests/unit/test_cellprofiler_runtime_plan_shape.py
tests/unit/test_cellprofiler_runtime_plan_consumers.py
tests/unit/test_cellprofiler_module_runtime_policy_boundary.py
~~~
<!-- cellprofiler-runtime-consolidation-forbidden-files:end -->

Behavioral tests from deleted test files move to tests named after the retained
owner. They are not discarded merely because the production shell disappears.

### Explicitly Not Authorized

Do not create any of the following, regardless of name variation:

- CellProfilerExecutionPlan;
- CellProfilerImportPlan;
- CellProfilerProjectionPlan;
- CellProfilerDispatchPlan;
- CellProfilerRuntimeContext;
- CellProfilerInvocationContext;
- CellProfilerProjectionContext;
- ArtifactCompilationCursor as a new record;
- a module-name-to-policy dict;
- a ProcessingContract by ImagePayloadExecutionMode dispatch table;
- a projection capability enum or set;
- a compatibility adapter for a deleted plan;
- a copied callable metadata dataclass;
- a debug explanation or trace carrier.

## Target Import And Compiler Algorithm

### Existing Context Becomes Complete

ArtifactDeclarationStepContext must expose a complete immutable forward
declaration state. Its exact field ownership after the phase is:

- step_name;
- step_index;
- source_bindings;
- group_by;
- input_source;
- available_artifacts;
- main_flow_artifacts;
- available_artifact_producers.

Its two new generic methods are responsible for:

1. with_source_declarations(source_specs) adds source artifact declarations and
   resets main flow when the context's exact input_source requires it;
2. advance_artifact_graph(graph, next_main_flow_artifacts) merges the existing
   ArtifactGraph outputs into available artifacts, appends its ArtifactProducer
   values, and installs the exact next main-flow collection supplied by the
   caller.

The methods return a new context. They do not mutate a shared global compiler
state and do not inspect module names.

`CellProfilerModule.processing_config` receives exactly
`ModuleArtifactContract`, `CallableContract`, and inherited
`ProcessingConfig`. It sets input source to pipeline start when the artifact
contract has inputs and to previous step otherwise. Axis-scoped callables use
callable-required variable components when declared and inherited components
otherwise; module import `group_by` overrides the inherited grouping only for
the converted module. Plate-scoped callables always lower to empty variable
components and `GroupBy.NONE`. `FuncStepContractValidator` validates the final
axis/group relation. The artifact contract never stores those callable or
processing constraints.

### Import Flow

The target importer performs:

1. parse modules once;
2. fold setup modules through CellProfilerModule source declarations;
3. construct the initial ArtifactDeclarationStepContext from source specs;
4. select each enabled executable module through CellProfilerModule;
5. bind settings once;
6. derive the module contract through the nominal declaration;
7. select the canonical callable from the module contract and resolved source
   bindings;
8. derive its `CallableContract` from the raw callable;
9. lower `ProcessingConfig` from the module artifact contract, callable
   contract, and inherited concrete config;
10. reconstruct the inclusion-minimal public kwargs through the same
    invocation_artifact_contract declaration used by compilation;
11. build the public FunctionStep;
12. extract its ArtifactGraph using the same generic declaration provider;
13. advance ArtifactDeclarationStepContext through advance_artifact_graph;
14. continue in source order;
15. derive the common pipeline config and return the public values.

Tentative same-module batching uses a local ArtifactDeclarationStepContext
value. Accepting the batch assigns that value to the outer traversal. Rejecting
the batch discards it. It does not manually copy three cursor variables.

_sparse_kwargs_for_contract remains the sole private importer minimizer because
minimal public syntax is an import responsibility. Its reconstruction callback
accepts the complete ArtifactDeclarationStepContext and cannot receive separate
available/main-flow arguments.

### Compiler Provider Flow

The target provider performs:

1. create one ArtifactDeclarationStepContext for the session;
2. advance source declarations through with_source_declarations;
3. obtain native ArtifactGraph values through
   extract_artifact_declarations;
4. obtain CP ModuleArtifactContract values through the nominal module
   declaration;
5. enrich the existing CallableContract and construct InvocationContractPlan;
6. advance the context through the same advance_artifact_graph operation used
   by import;
7. store only the exact immutable plan map;
8. discard the forward context after provider construction.

CellProfilerInvocationContractProvider remains a lookup, not a compiler. It
must not reconstruct a contract in __call__.

### Import/Compiler Equivalence Gate

For every imported pipeline fixture, compile the returned public steps in a
fresh CompilationSession and assert:

- every public invocation key has exactly one InvocationContractPlan;
- reconstructed ModuleArtifactContract values equal the import target
  contracts used to minimize kwargs;
- available artifact refs after each step equal the import traversal;
- main-flow refs after each step equal the import traversal;
- ArtifactProducer specs, groups, and invocation keys equal the import
  traversal;
- compiling the pycodified source yields the same plans.

Tests exercise the pure ArtifactDeclarationStepContext advancement methods
directly and compare the final compiled plans. No test-only callback or
production observation hook is added. Production does not persist the context
on FunctionStep, PipelineConfig, CompiledStepPlan, or generated source.

## Target Runtime Algorithm

### Callable Construction

~~~text
CompiledFunctionInvocation.contract
  -> FunctionInvocationCallableResolver resolves canonical raw callable
  -> RuntimeAdapterSpec.runtime_callable_factory
  -> CellProfilerModuleExecutor(raw_func, compiled_callable_contract)
  -> process-local callable cache
~~~

CellProfilerModuleExecutor implements __call__ with the current adapter
parameter ABI. There is no separate CellProfilerRuntimeCallable.

### One Invocation

The executor performs one visible sequence:

1. validate and expose the compiled CallableContract;
2. resolve the CellProfilerModule nominal owner;
3. select active input and output specs from exact compiled plans;
4. resolve primary images and runtime artifact kwargs through the adapter;
5. compose one CellProfilerImageRequest containing only genuine image/source
   state;
6. resolve one effective ImagePayloadExecutionMode through module MRO;
7. project source-binding or already-selected runtime planes exactly once;
8. validate the mode, ProcessingContract, image axis, and kwarg axes together;
9. construct one RuntimeFunctionInvocationRequest;
10. call the existing CellProfilerFunctionContractExecutor.execute once with
    the compiled contract;
11. match and record declared outputs;
12. select main flow directly from the compiled output contract.

Every helper is either:

- a method on CellProfilerModuleExecutor for orchestration;
- a query on CallableContract or ModuleArtifactContract;
- a method on CellProfilerModule for module-specific behavior;
- a registered strategy selected by artifact type or nominal runtime value.

No helper returns a second request or plan merely to rename the same fields.

### Dispatch Preflight

Before execute reaches a raw callable, it validates:

- the resolved raw function is the callable owned by the compiled contract;
- the module registry owner agrees with ModuleArtifactContract.module_name;
- the source-declared and compiled ProcessingContract agree;
- the image execution mode is allowed for the ProcessingContract;
- PURE_3D receives no RUNTIME_SLICE-aligned kwarg;
- ALIGNED_MULTI_IMAGE_STACK receives an AlignedImageStack and no PURE_3D
  contract;
- a selected runtime plane has exact image and kwarg cardinality;
- a preserved stack does not accidentally arrive at a plane-only backend call;
- runtime-owned semantic controls are injected at most once.

Runtime validation is not a substitute for compile-time validation. Static
callable declarations and impossible mode/contract combinations fail in
FuncStepContractValidator. Runtime preflight validates payload-dependent
cardinality and module-MRO overrides.

## Target Projection Algorithm

### Projection Coordinates

RuntimeAdapterRequest supplies RuntimePlaneProjection. Convert it to
RuntimePlaneAxisValueProjection only when projecting an actual value.

For source identity:

1. CurrentSourcePayloadPlaneSelectionAuthority returns
   CurrentSourcePayloadPlaneSelection;
2. require an unambiguous exact selection when a plane-local value is needed;
3. construct RuntimePlaneAxisValueProjection for SOURCE_BINDING;
4. call RuntimeSliceProjection.value_for_slice.

For runtime slices:

1. use RuntimePlaneAxisValueProjection.from_projector with the adapter and
   RUNTIME_SLICE;
2. if the projection selects a plane, call
   RuntimeSliceProjection.value_for_slice;
3. if it preserves the stack, leave the value intact until the selected
   execution method owns slicing.

### Exactly-Once Projection

The production call graph must contain only these projection sites:

| Site | Axis | Purpose |
| --- | --- | --- |
| Adapter-bound outer invocation | selected RUNTIME_SLICE only | Project a value because generic execution already selected a plane |
| PURE_2D slice loop | RUNTIME_SLICE | Project every aligned kwarg to the current internal slice |
| Aligned multi-image loop | RUNTIME_SLICE | Project composed image and aligned kwargs together |
| Current-source selection | SOURCE_BINDING | Select a source-identity plane |

No value may pass through two sites for the same axis and plane. Tests must spy
on RuntimeSliceProjection.value_for_slice and assert one call per projecting
value per selected plane.

### Value Ownership

The existing RuntimeSliceProjectionStrategy leaves remain responsible for:

- ImageMetadataPayload and MaskedImagePayload;
- RuntimeSliceAlignedValueSet;
- ObjectLabelValue;
- MeasurementTable;
- RuntimeSliceProjectableValue;
- parent-child and object relationships;
- SparseIJVLabelRows;
- tuple and list containers;
- explicit pass-through scalar/config values.

CP code must not reproduce isinstance branches for these types. A genuinely new
projectable runtime type registers at the existing generic nominal root; this
plan introduces no new runtime value type.

## Migration Work Orders

### Phase 0: Freeze The Current Semantics And The Desired Shape

Add focused failing tests before production migration.

Required tests:

- import/compiler forward-state equivalence for a mixed native and CP pipeline;
- source reset at PIPELINE_START;
- main-flow preserve and replace cases;
- grouped ArtifactProducer ownership;
- same-module tentative batch rollback;
- processing config cases currently covered by
  test_cellprofiler_module_processing_config.py;
- callable selection for morphology 2-D/3-D and illumination all-image scope;
- dispatch matrix across all ImagePayloadExecutionMode and ProcessingContract
  combinations;
- exact once-only projection for images, labels, aligned values, grids,
  measurements, relationships, and sequences;
- FilterObjects receives one 2-D ObjectLabelValue for each selected plane;
- failures are raised before a raw callable spy is invoked;
- debug invocation parameters expose bounded axis and shape facts;
- AST gates for all planned deletions.

Add these to the existing files named after surviving authorities:
test_invocation_contract_provider.py,
test_module_artifact_contract_exactness.py,
test_runtime_slice_projection.py, test_cellprofiler_runtime_projection.py,
test_debug_runtime.py, and test_cellprofiler_generated_pipeline_execution.py.
Do not create a parallel test-helper family or append another several thousand
lines to test_cellprofiler_module_execution.py.

Exit gate: tests fail only because the old shells or duplicate paths still
exist, not because fixture setup is incomplete.

### Phase 1: Unify Import And Compiler Forward Semantics

Work:

1. extend ArtifactDeclarationStepContext with the two missing available and
   main-flow collections;
2. add only with_source_declarations and advance_artifact_graph, composing the
   existing ArtifactSpecCollection plus ArtifactGraph.outputs and
   ArtifactGraph.producers;
3. migrate CellProfilerInvocationContractProviderFactory;
4. migrate pipeline_import tentative and committed traversal;
5. make module artifact-contract methods consume the complete context;
6. remove separate available_artifacts and main_flow_artifacts parameters where
   the context owns them;
7. replace _processing_group_keys with the existing source-binding authority;
8. delete _advance_artifact_cursors;
9. run import/compiler equivalence and generated-source round trips.

Deletion gate:

~~~text
rg -n "_advance_artifact_cursors|def _processing_group_keys" openhcs tests
rg -n "available_artifacts=.*main_flow_artifacts=" openhcs/interop/cellprofiler
~~~

The second gate is reviewed for calls that merely happen to contain both
values; no module declaration may receive them separately once the complete
context is accepted.

### Phase 2: Bind Processing Lowering Directly To Existing Owners

Work:

1. make `CellProfilerModule.resolve_function` consume the artifact contract and
   resolved source bindings directly;
2. make `CellProfilerModule.processing_config` consume exactly the artifact
   contract, callable contract, and inherited concrete `ProcessingConfig`;
3. keep required variable components and allowed grouping solely on
   `CallableContract`;
4. keep `ModuleArtifactContract` artifact-only;
5. migrate all concrete module overrides to those exact nominal inputs;
6. preserve the existing processing-config behavior tests;
7. delete the source-axis summary and its helper file without a replacement.

Deletion gate:

~~~text
rg -n "SourceProcessingAxisPlan|axis_plan" openhcs tests
test ! -e openhcs/interop/cellprofiler/module_processing_config.py
~~~

Exit gate: every processing config and callable selection fixture produces the
same public FunctionStep config and canonical callable as before the phase.

### Phase 3: Collapse Runtime Callable And Plan Shells

Work:

1. make CellProfilerModuleExecutor directly callable;
2. store raw_func and compiled callable_contract only;
3. migrate module runtime hooks to exact existing arguments;
4. replace runtime-plan partition/ref queries with the existing
   ModuleArtifactContract partition and exact-ref methods;
5. replace CellProfilerOutputRecordRequest copied fields with
   callable_contract;
6. collapse CellProfilerMeasurementImageResolver orchestration;
7. collapse CellProfilerFunctionOutputAggregationContract;
8. update RuntimeAdapterSpec.runtime_callable_factory;
9. update process-local callable cache and introspection tests;
10. delete runtime callable and runtime plan shells.

Deletion gate:

~~~text
rg -n "CellProfilerRuntimeCallable|CellProfilerModuleRuntimePlan" openhcs tests
rg -n "CellProfilerFunctionOutputAggregationContract|CellProfilerMeasurementImageResolver" openhcs tests
rg -n "CallableContract.from_callable" openhcs/interop/cellprofiler/runtime
~~~

The final gate must return no production runtime calls.
CallableContract.from_callable is confined to declaration, import, and
compile-time code outside the runtime package.

### Phase 4: Collapse Dispatch And Projection

Work:

1. [x] change the existing execute signature to accept the compiled contract and
   own the visible mode/contract matrix;
2. [x] migrate natural, full-stack, and aligned-stack behavior;
3. [x] delete force_full_stack and requested_image_execution_mode; every caller
   supplies typed execution_mode;
4. [x] migrate adapter value projection to RuntimeSliceProjection;
5. [x] migrate executor value projection to RuntimeSliceProjection;
6. [x] migrate current-source adapter projection through the existing selection
   authority;
7. [x] delete CellProfilerImageRequest's source-axis projection methods;
8. [x] remove derived CellProfilerImageRequest flags;
9. [x] migrate main-flow policy to exact contract queries;
10. [x] migrate current-image object-label alignment;
11. [x] delete image execution strategy;
12. [x] delete CP projection files;
13. [x] run projection count, cardinality, output aggregation, and backend leaf
   tests.

Deletion gate:

~~~text
rg -n "CellProfilerImageExecutionStrategy|requested_image_execution_mode|force_full_stack" openhcs tests
rg -n "RuntimePlaneProjectionCapability|RuntimePlaneProjectionRequirement" openhcs tests
rg -n "CurrentRuntimePlaneKwargProjection|RuntimePlaneImagePayloadProjection" openhcs tests
rg -n "CurrentSourceImagePayloadProjection|CurrentSourceObjectLabelPayloadProjection" openhcs tests
AST gate: CellProfilerImageRequest defines none of source_axis_payload,
composed_source_payload_for_name, or source_payloads_for_names.
rg -n "projects_runtime_slice_kwargs|publishes_side_effect_main_flow" openhcs tests
test ! -e openhcs/interop/cellprofiler/runtime/image_execution_strategies.py
test ! -e openhcs/interop/cellprofiler/runtime/projection_requirements.py
test ! -e openhcs/interop/cellprofiler/runtime/runtime_plane_kwargs.py
test ! -e openhcs/interop/cellprofiler/runtime/projection.py
~~~

Exit gate: the FilterObjects regression fails before invocation under an
invalid projection and succeeds with exact per-plane labels under a valid
projection.

### Phase 5: Narrow Adapter State And Unify Diagnostics

Work:

1. [x] remove adapter projection capability parameters and projection wrappers;
2. [x] delete all eight adapter-local caches;
3. [x] route record lookups through RuntimeValueStore and measurement lookups
   through the existing store-bound query caches; add no image/object
   composition cache;
4. [x] route source candidate reuse through the existing named
   ProcessLocalBoundedCache instances in source_candidates.py;
5. [x] delete RuntimeArtifactCacheInvalidationPolicy and its file;
6. [x] move runtime-bound debug observation to the actual invocation boundary;
7. [x] bound DebugInvocationParameter.value_repr using existing metadata and
   projection coordinates without adding fields;
8. [x] delete the CP-only pure-2D JSON trace hook;
9. [x] verify exceptions contain the required context without debug mode.

Deletion gate:

~~~text
rg -n "OPENHCS_PURE2D_SLICE_TRACE_PATH|_trace_pure_2d_slice|Pure2DTraceArrayStats" openhcs tests
rg -n "projection_capabilities=" openhcs/interop/cellprofiler
rg -n "_image_cache|_current_image_cache|_object_cache|_current_object_cache|_measurement_cache|_artifact_availability_cache|_source_candidate_cache|_source_candidate_process_key_cache" openhcs/interop/cellprofiler/runtime/adapter.py
rg -n "RuntimeArtifactCacheInvalidationPolicy" openhcs tests
test ! -e openhcs/interop/cellprofiler/runtime/runtime_artifact_cache_invalidation.py
~~~

Exit gate: the generic debug event and ordinary exception each identify the
same invocation, axis, artifact, and cardinality mismatch.

### Phase 6: Fresh ZMQ And Parity Acceptance

Run acceptance in increasing cost order:

1. focused unit tests for the retained authorities;
2. all CellProfiler unit tests;
3. tests/integration/test_cellprofiler_generated_pipeline.py;
4. the ImagingFlow ZMQ case that currently reaches FilterObjects;
5. representative 2-D, multi-image, full-stack, object-measurement,
   relationship, grid, exporter, and plate-scoped cases;
6. generated-source compile in a fresh ZMQ server;
7. official30 strict native-reference parity, baseline;
8. official30 strict native-reference parity with inherited non-persistent
   Napari streaming.

The existing runtime-unification Phase 3 requirements remain binding:

- OpenHCS results are not reused from a prior runtime cache;
- native timing is measured;
- generated source is the submitted source;
- compile and execute are two ZMQ stages;
- runtime observations contain the exact declared artifact records;
- parity uses the existing strict tolerances and export checks.

No performance claim is made from unit microbenchmarks. After semantic and
parity gates pass, profile the consolidated runtime once and compare:

- adapter construction;
- image and artifact input resolution;
- projection;
- raw function execution;
- output recording;
- total invocation time.

Optimize only from that profile and preserve the ownership model.

## Static Architecture Gates

Extend tests/unit/test_cellprofiler_static_deletion_gates.py. Reuse its AST
walker, import/reference detection, and file inventory logic; do not create a
second architecture-gate implementation. Give this plan an exact separately
delimited forbidden-symbol and forbidden-file inventory and make the existing
test load both plans.

Required assertions:

1. forbidden definitions and files above are absent;
2. CellProfilerModuleExecutor annotated state is exactly raw_func and
   callable_contract;
3. the executor implements __call__;
4. RuntimeAdapterSpec.runtime_callable_factory returns the executor directly;
5. no CP runtime module invokes CallableContract.from_callable;
6. no CP module hook annotation names a runtime plan;
7. no generic core module imports
   openhcs.processing.backends.cellprofiler or
   openhcs.interop.cellprofiler;
8. ArtifactDeclarationStepContext is the only class with the combined fields
   available_artifacts, main_flow_artifacts, and
   available_artifact_producers;
9. ArtifactDeclarationStepContext also owns exact source_bindings, group_by,
   and input_source and owns no ProcessingConfig field;
10. ModuleArtifactContract contains artifact items only and has no required-axis
    or allowed-grouping field;
11. required variable components and allowed grouping are queried only from
    CallableContract metadata;
12. StepSnapshot stores the resolved step and compiler/path-planner consumers
    read configs through snapshot.step directly;
13. no class outside existing public config or source-binding owners mirrors
    source component tuple fields;
14. RuntimeSliceProjectionStrategy remains registry-selected by nominal value
    type;
15. CP runtime projection call sites call RuntimeSliceProjection and do not
    contain ObjectLabelValue or RuntimeSliceAlignedValueSet projection branches;
16. image execution mode dispatch occurs only in the existing execute method;
17. no module-name string comparison appears in generic runtime or compiler
    code;
18. deleted symbols are absent from imports, __all__, TYPE_CHECKING blocks,
    tests, fixtures, and docs that claim current architecture.
19. CellProfilerRuntimeAdapter annotated state is exactly request and backend;
    it owns no local cache field.
20. no CP runtime module calls RuntimePlaneAxisValueProjection.project or
    AlignedImageStack.project_plane_axis directly; all actual value projection
    enters through RuntimeSliceProjection.

Do not create a second AST parser or copy the first plan's inventory. Reuse the
existing gate mechanics while keeping each plan's deletion contract in its own
machine-delimited section.

## Focused Behavioral Gates

### Import And Compiler

- source-only input;
- prior-main-flow input;
- runtime artifact input;
- mixed source and runtime artifact input;
- one and multiple main-flow images;
- side-effect-only measurement output;
- main-flow replacement;
- default and grouped function patterns;
- repeated same-module batching;
- batch rollback;
- plate-scoped exporter;
- native callable between two CP modules;
- generated-source round trip;
- fresh-process canonical callable identity.

### Processing Config

- default 2-D;
- declared Z stack;
- ordered multi-image source stack;
- source-identity grouping;
- prior-main-flow consumption;
- repeated main-flow artifact kind;
- object-only runtime input;
- pure-2D measurement source grouping;
- TrackObjects timepoint semantics;
- CalculateMath source routing;
- illumination all-image override;
- morphology 2-D versus 3-D callable selection.

### Dispatch

- NATURAL with PURE_2D, PURE_3D, FLEXIBLE, and VOLUMETRIC_TO_SLICE;
- FULL_STACK for every declaration-owned callable that currently declares it;
- ALIGNED_MULTI_IMAGE_STACK for aligned multi-image callables;
- invalid PURE_3D aligned-stack rejection;
- dynamic module MRO overrides for grid and illumination;
- semantic-control injection exactly once;
- runtime batch executor selection from the compiled contract;
- no raw contract reconstruction.

### Projection

- bare ndarray does not invent a stack;
- declared image axis exact projection;
- mask and metadata preservation;
- ObjectLabelValue domain and identity preservation;
- RuntimeSliceAlignedValueSet exact projection;
- grids, measurements, relationships, and sequences;
- SOURCE_BINDING selection by exact identity;
- ambiguity and no-match behavior;
- selected plane count mismatch;
- preserved stack behavior;
- nested aligned values;
- projection exactly once;
- raw callable not invoked after failed preflight.

### Output And Main Flow

- one and multiple image outputs;
- object label output domain scope;
- recorded versus declared-only outputs;
- measurement-only pass-through;
- side-effect image source publication;
- relationship and grid output recording;
- pure-2D auxiliary aggregation;
- full-stack output preservation;
- source identity and plane-axis preservation.

### Debuggability

- generic DebugCursor identifies the exact invocation;
- final runtime argument summaries are bounded and deterministic;
- image and object label summaries include type, shape, axis, and cardinality;
- exception messages include all required projection coordinates;
- no CP-only trace file is written;
- fresh ZMQ exception transport preserves the diagnostic message.

## Review Checklist

- [x] Public import still returns only FunctionSteps and PipelineConfig.
- [x] Generated source contains no compiled or runtime carrier.
- [x] StepConfigUniverse and its file are deleted; resolved FunctionStep values
      are the compiler configuration authority.
- [x] CellProfilerModule remains the only module registry.
- [x] Import and compiler call one ArtifactDeclarationStepContext advancement.
- [x] No forward artifact cursor algorithm is duplicated.
- [x] SourceProcessingAxisPlan and its file are deleted.
- [x] Source component facts use existing source-binding properties and
      ComponentSet operations; no CP query is added to generic source bindings.
- [x] ModuleArtifactContract contains only artifact facts.
- [x] CallableContract is the sole owner of required variable components and
      allowed grouping.
- [x] ArtifactDeclarationStepContext owns exact source bindings, group_by,
      input_source, artifacts/producers, and main flow but no ProcessingConfig.
- [x] StepSnapshot stores the resolved FunctionStep; compiler and path planner
      consume its configs directly.
- [x] Pipeline ObjectState resolution runs once; per-axis planning requires
      resolved steps, ObjectStates, and snapshots and has no context-local
      resolver or resolution-mode switch.
- [x] Runtime receives the compiled CallableContract directly.
- [x] CellProfilerRuntimeCallable and CellProfilerModuleRuntimePlan are deleted.
- [x] Runtime output requests do not copy function, module, or contract identity.
- [x] Image mode and ProcessingContract interact at one visible decision point.
- [x] No image execution strategy registry remains.
- [x] No projection capability enum or wrapper remains.
- [x] RuntimeSliceProjection performs every actual value projection.
- [x] CP runtime contains no direct RuntimePlaneAxisValueProjection.project or
      AlignedImageStack.project_plane_axis call.
- [x] Current-source selection selects an index and does not project values.
- [x] CellProfilerImageRequest contains no derived policy booleans.
- [x] Adapter source reads do not accept projection capabilities.
- [x] Generic core imports no concrete CP module.
- [x] Module-specific semantics stay on nominal module declarations.
- [x] Registry-only leaves are not deleted based on text reference count.
- [x] Projection mismatches fail before raw backend callables.
- [x] Generic debug events replace the CP-only trace.
- [x] Static deletion gates cover definitions, imports, attributes, and files.
- [x] Focused tests are organized by retained authority.
- [ ] ImagingFlow passes through fresh ZMQ.
- [ ] Official30 strict parity passes in baseline and Napari configurations.
- [ ] Performance is profiled once after semantic verification.

## Resulting Data Flow

~~~text
.cppipe
  -> CPPipeParser
  -> CellProfilerModule nominal declarations
  -> public FunctionStep list + PipelineConfig
  -> generic source rendering and transport
  -> fresh ZMQ reconstruction
  -> ObjectState resolves FunctionStep config inheritance
  -> StepSnapshot.step
  -> CompilationSession
  -> PipelineInvocationContractProviderAuthority
  -> CellProfilerInvocationContractProviderFactory
  -> ArtifactDeclarationStepContext
     (source_bindings, group_by, input_source, artifacts/producers/main flow)
  -> CellProfilerModule.invocation_artifact_contract
  -> InvocationContractPlan
  -> CallableContract axes/grouping + artifact-only ModuleArtifactContract
  -> ArtifactGraph
  -> CompiledFunctionInvocation
  -> FunctionCoreExecutor
  -> RuntimeAdapterRequest
  -> CellProfilerRuntimeAdapter
  -> CellProfilerModuleExecutor
  -> RuntimeFunctionInvocationRequest
  -> CellProfilerFunctionContractExecutor.execute
  -> one mode/ProcessingContract decision inside that existing method
  -> RuntimeSliceProjection when an axis is actually selected
  -> raw declaration-owned callable
  -> RuntimeReturnedOutputMatcher
  -> CellProfilerOutputRecorder
  -> RuntimeValueStore and main flow
  -> generic DebugEvent / RuntimeProfileSink observations
~~~

There is no CellProfiler execution plan between the compiled contract and the
executor, no second artifact cursor beside ArtifactDeclarationStepContext, no
source-axis summary beside source bindings and callable metadata, no image
strategy registry beside ProcessingContract, and no CP value projector beside
RuntimeSliceProjection.
