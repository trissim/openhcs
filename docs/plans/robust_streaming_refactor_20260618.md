# Robust Streaming Refactor Plan - 2026-06-18

## Current Smoke Result

Dataset:

- `/tmp/openhcs_benchmark_dataset_cache_last8/CellProfiler_tutorials/data/BeginnerSegmentation`
- pipeline: `segmentation_final.cppipe`
- execution server: IPC ZMQ on port `7777`
- most recent direct validation viewer: Napari IPC ZMQ on port `5581`
- config: two wells, Napari streaming enabled

Observed:

- A direct ZMQ pycodified execution completed for wells `A14` and `B13`.
- Materialized image, measurement, ROI, and segmentation-summary outputs were
  produced under `/tmp/openhcs_zmq_projection_live/out/workspace_projection_live`.
- The first source-like streamed layer rendered with expected axes:
  - source layer shape: `(2, 5, 520, 696)`
  - layer-local axis labels: `site`, `channel`, `y`, `x`
- Later streamed layers now retain semantic axes instead of collapsing singleton
  route-local channel values into unlabeled planes:
  - derived routes can carry `well`, `site`, `channel`, `y`, `x`
  - singleton channel routes can be translated into the shared channel domain
- Outstanding failure: many Napari steps still appear completely black in the
  viewer. A single screenshot is not sufficient validation because a nonzero
  translated layer can look black when the current global channel/site/well plane
  is outside that layer's route-local coordinate domain.

Interpretation:

The immediate bug is no longer just "Napari axis labels are wrong." The viewer
projection layer has partial fixes, but the system still lacks a closed-loop
contract proving that each produced output carries stable source/component
semantics from runtime payload construction through materialization, PolyStore
transport, viewer route identity, current-plane projection, and screenshots.
Black layers must be debugged as a whole semantic chain: source provenance,
output identity, component metadata, stream DTO, Napari route projection, and
current displayed slice.

The previous "stuck execution" symptom is a separate status-observation issue:
runtime lifecycle can enter a terminal state while UI/MCP surfaces still look
busy. That remains in scope, but it should not distract from the streaming
semantic authority boundary.

## Advisor Audit - 2026-06-18

Commands run against production code only:

```bash
nominal-refactor-advisor --include-plans --include-execution-plan \
  --include-impact-ranking --impact-ranking-max 20 \
  openhcs/runtime \
  external/PolyStore/src/polystore/streaming \
  external/PolyStore/src/polystore/napari_stream.py \
  external/PolyStore/src/polystore/fiji_stream.py

nominal-refactor-advisor --include-plans --include-execution-plan \
  --include-impact-ranking --impact-ranking-max 20 \
  openhcs/core/runtime_values.py \
  openhcs/core/source_bindings.py \
  openhcs/core/source_matching.py \
  openhcs/core/source_binding_selection.py \
  openhcs/core/source_image_semantics.py \
  openhcs/core/steps/stream_component_semantics.py \
  openhcs/core/streaming_config_factory.py \
  openhcs/core/aligned_image_payload.py

nominal-refactor-advisor --include-plans --include-execution-plan \
  --include-impact-ranking --impact-ranking-max 20 \
  openhcs/interop/cellprofiler/runtime \
  openhcs/interop/cellprofiler/runtime_pipeline.py \
  openhcs/interop/cellprofiler/source_schema.py \
  openhcs/interop/cellprofiler/source_schema_ingestion.py \
  openhcs/interop/cellprofiler/pipeline_generator.py
```

Relevant findings:

- The runtime/PolyStore scan selected `distributed_boundary_fanout` as a top
  authority target and specifically flagged `StreamProducerIdentity` projection
  and payload serialization as a semantic-dict/projection boundary.
- The runtime-values/source scan selected `runtime_values.py` and
  `aligned_image_payload.py` as high-priority nominal-boundary failures around
  loose `value`, `data`, `payload`, and `return` families.
- The CellProfiler scan selected `CellProfilerRuntimeAdapter`,
  `CellProfilerModuleExecutor`, and `GeneratedPipelineMaterializationSidecarCodec`
  as distributed-fanout boundaries. That is the same class of bug as lost object
  artifact routing and source/component provenance drift.
- Raw targeted findings also flagged `StepOutputManifestStore` fallback-style
  mapping lookups and `ImagePayloadSourcePathResolver.resolve_backend_path`
  reflective fallback. These are smaller but related: the semantic absence cases
  are not classified at formal boundaries.

Strategic conclusion:

The load-bearing boundary should be a produced-output semantic record, not a
Napari-side patch. That record must be constructed from runtime payload/source
provenance and output identity, then used by materialization and streaming. The
viewer should project a declared record; it should not recover semantics from
filenames, dict bags, ad hoc kwargs, or current Napari UI state.

## Chosen Authority Boundary

Introduce a nominal produced-output semantic boundary that sits between
function execution/materialization and every downstream consumer.

Proposed authority:

```python
@dataclass(frozen=True, slots=True)
class ProducedOutputSemantics:
    producer_identity: StreamProducerIdentity
    output_identity: FunctionOutputIdentity
    source_provenance: SourceImageProvenance
    component_metadata: SourceComponentMetadata
    component_axis_semantics: ViewerComponentAxisSemantics
    output_path: str
```

The exact class names can change during implementation, but the responsibilities
must not:

1. Filename parsing is only an ingestion/source-binding authority.
2. Runtime payloads own source provenance and spatial/source-plane facts.
3. Function output identity owns filename/path projection.
4. The produced-output record owns the combined facts after execution.
5. PolyStore transports this record or its wire projection; it does not infer
   OpenHCS semantics.
6. Napari/Fiji project the declared component-axis semantics; they do not parse
   generated filenames or invent axis domains.
7. Viewer-state validation queries the same route records used to draw layers.

Any code path that cannot supply this record must fail loudly at the boundary.
Implicit filename fallback is allowed only for external source ingestion, never
for OpenHCS-generated outputs.

## Architectural Goals

1. A pycodified pipeline must carry CellProfiler artifact dependencies as typed
   runtime contracts, not comments or inferred side effects.
2. Streaming requests must be nominal records from OpenHCS to PolyStore and
   viewer servers, not open-ended kwargs with duplicated coercion logic.
3. Viewer component semantics must have one source of truth from compiler output
   identity through materialization, PolyStore transport, and Napari/Fiji layer
   projection.
4. Runtime status must expose one terminal lifecycle contract to the UI, MCP, and
   direct clients.
5. End-to-end streaming validation must run without human UI interpretation:
   compile, execute, stream, inspect viewer state, capture screenshot, assert
   axes/layers/status.

## Parallel Semantic Paths That Must Collapse

This plan is not a wrapper cleanup. The stop condition is collapsing duplicate
semantic authorities so one nominal path decides each fact and every downstream
system consumes that decision.

1. Runtime slice projection currently has competing authorities:
   `RuntimeSliceProjection` projects payload values, pure-2D backend execution
   restamps result identity, CellProfiler output recording slices native values,
   and materialization re-projects runtime slices before writing. These must
   collapse behind `openhcs.core.runtime_slice_projection`; backend-specific
   code can ask for a projected item, but it must not decide slice identity.
2. Stream item identity is split between `function_outputs.py`,
   `stream_component_semantics.py`, materialization input records, PolyStore
   batch metadata, and viewer backend kwargs. These must collapse into one
   stream-visible output item record that owns path, producer identity,
   component metadata, source-plane identity, and viewer source metadata.
3. Producer/output identity is built separately for main outputs and artifact
   outputs. `StepOutputStreamIdentityAuthority` and
   `ArtifactStreamIdentityAuthority` must become projections of one function
   step output identity authority, not parallel builders.
4. Component-axis semantics are declared in OpenHCS and recomputed in PolyStore.
   `ViewerComponentAxisSemantics` must be the source of truth crossing the wire;
   PolyStore may adapt a wire form, but it must not rebuild semantics from
   component-order tables and mode dictionaries.
5. Component value domains are sent explicitly by OpenHCS and also inferred from
   prepared batch metadata in PolyStore. Inference can only be a typed external
   compatibility adapter; normal OpenHCS streaming must carry a declared domain.
6. ROI/source naming is decided by source stems, component metadata stems,
   artifact filename descriptors, archive identity, and materialized path
   projection. ROI archive names must become projections of produced output
   identity plus declared source-plane identity.
7. Source metadata absence and merge rules are spread across artifact records,
   parser fallback, ROI identity validation, and viewer kwargs mutation. They
   must collapse into a typed source identity classification at the runtime
   payload/source-binding boundary; downstream code receives a declared source
   identity or a typed absence.
8. Viewer route/window projection interprets producer identity, component layout,
   slice/channel/window roles, and backend display extras in several receiver
   modules. Sender and receiver must share a viewer-neutral route projection
   contract derived from the declared component-axis semantics.

Each implementation pass must name which item above it collapses. Local
properties, adapters, or annotations only count when they remove one of these
parallel routes or make it impossible for a route to make an independent
semantic decision.

## Phase 1: Produced Output Semantic Record

Problem:

`FunctionOutputIdentity`, `SourceImageProvenance`, stream component metadata,
`StepOutputManifestStore`, and `StreamProducerIdentity` are all nearby semantic
owners, but no single record proves what a produced output is. That allows black
viewer layers and ROI metadata failures to appear late, after the original source
facts have already been split across payload metadata, filenames, manifests, and
stream kwargs.

Refactor:

- Add a produced-output semantic record in `openhcs.core.steps` that combines:
  - producer identity
  - function output identity
  - source provenance
  - source component metadata
  - output path
  - viewer component-axis semantics
- Have function output writing/materialization create this record at the same
  point it creates the output path.
- Replace `StepOutputManifestStore` path-only entries with semantic entries.
  Path queries can remain as projections, but the manifest's stored value must
  be semantic output records.
- Classify absence explicitly:
  - no producer for a pipeline-start step
  - no source component metadata on external source payload
  - no produced output for a step that emitted no files
  - stale path lookup
- Remove unclassified `mapping.get(..., ())` style absence in the manifest path.
  A consumer either receives a declared empty result or a typed no-match error.

Validation:

- Unit test that a produced image output has one semantic record containing
  source component metadata, output identity, producer identity, and path.
- Unit test that a previous-step consumer resolves its producer through semantic
  records, not filename matching.
- Unit test that stale producer path lookup raises the typed no-match error.
- Advisor scan on `function_output_identity.py`, `function_output_manifest.py`,
  and touched execution/materialization files.

## Phase 2: Restore CellProfiler Artifact Contracts

Problem:

The generated `FunctionStep` for `IdentifySecondaryObjects` comments that it
requires `object_labels:Nuclei`, but the step declaration only carries the image
source binding and function kwargs. Runtime therefore invokes:

```python
identify_secondary_objects(image, **kwargs)
```

instead of supplying the `primary_labels` artifact input.

Refactor:

- Make `CellProfilerSymbolTable` and `ModuleArtifactContract` the single source
  for generated runtime artifact inputs and outputs.
- Ensure `CellProfilerPipelineGenerator` emits a real third tuple member:

```python
func=(identify_secondary_objects, kwargs, CellProfilerInvocationOptions(...))
```

or an equivalent typed `RuntimeInvocationOptions` object that carries:

- `runtime_artifact_inputs`
- `artifact_outputs`
- grid-cycle scope
- module identity
- source-object aliases when needed for plane alignment

- Add a compile-time invariant: if a CellProfiler module contract contains
  runtime artifact inputs, the pycodified `FunctionStep` must carry invocation
  options that expose them.
- Remove comment-only artifact dependency emission once the invariant exists.

Validation:

- Generate `segmentation_final_openhcs.py` and assert step 1 has a typed
  `CellProfilerInvocationOptions` carrying `object_labels:Nuclei`.
- Execute two wells through at least `IdentifySecondaryObjects`.
- Add a focused regression for `primary_labels` binding from prior
  `IdentifyPrimaryObjects` output.

## Phase 3: Make Stream Request Typing Load Bearing

Problem:

OpenHCS, PolyStore, Napari, and Fiji still exchange viewer-stream data through
wide kwargs and repeated field/coercion logic. That allows semantic drift:
producer identity, component metadata, display config, and transport endpoint can
be present in one path but missing or re-parsed in another.

Refactor:

- Introduce one nominal stream request DTO owned by the shared transport layer,
  likely in `zmqruntime` or the narrowest shared package already used by
  PolyStore and OpenHCS.
- OpenHCS materialization/streaming code builds this DTO from
  `ProducedOutputSemantics`.
- PolyStore streaming backends accept that DTO, not untyped kwargs. Existing
  kwargs become a compatibility adapter at the public filemanager boundary only.
- Napari/Fiji server handlers consume the same DTO wire projection, not a
  parallel local schema.
- Use declaration-level field metadata or dataclass/attrs typing as the schema
  authority; do not hand-maintain per-field accessor methods.

Required DTO content:

- producer identity
- data type
- transport endpoint
- display config identity
- plate/workspace identity
- output identity
- component metadata for each payload
- component metadata by path when batch paths are materialized
- viewer display axis semantics
- optional message extras as an explicitly named extension map

PolyStore-specific follow-up from the advisor:

- Collapse `StreamProducerIdentity` constructor and payload projection into one
  authoritative schema. `pipeline_output`, `manual`, and `direct` can remain
  named constructors only if they derive from a declared variant catalog or one
  explicit constructor authority.
- Replace `StreamProducerIdentity.to_payload()` reflection with declared schema
  projection.
- Remove generated-output metadata parsing from
  `StreamingBackend._parse_component_metadata`; OpenHCS-generated outputs must
  pass explicit component metadata. Filename parsing remains available only for
  source files whose owning ingestion authority is a microscope parser.

Validation:

- Existing unit tests for Napari streaming, streaming service, and viewer
  protocol remain green.
- Add one integration test that serializes the DTO through OpenHCS -> PolyStore
  -> viewer server without losing component metadata.
- Advisor scan must not flag duplicated field accessor/coercion families in the
  stream request path.

## Phase 4: Centralize Component Axis Semantics

Problem:

Streaming is only robust if layer axes are derived from the same component
authority as filenames, materialized paths, source bindings, and output identity.
Local maps such as `channel -> AllComponents.CHANNEL` or duplicated component
orders are semantic drift risks.

Refactor:

- Treat `AllComponents` plus `ViewerComponentAxisSemantics` as the component
  authority for viewer projection.
- Make every streamed payload carry component metadata created by
  `ProducedOutputSemantics`.
- Ban viewer-side filename parsing when component metadata is available.
- Keep filename parsing only as an explicit source-ingestion authority for
  virtual files and source bindings, not as a streaming fallback.
- Ensure ROI/artifact streaming uses the same component metadata path as image
  streaming.
- Make Napari layer-state/current-slice inspection a control endpoint that
  reports route key, producer identity, axis labels, offsets, component values,
  full-data stats, and current-slice stats for each layer.
- Use that endpoint in tests and MCP rather than relying on one screenshot or
  human interpretation of the active slider plane.

Validation:

- Stream images and ROI artifacts with explicit component metadata.
- Assert no viewer stream path depends on parser-readable filenames for data
  generated inside OpenHCS.
- Napari layer axis labels for the beginner segmentation first step remain
  `site`, `channel`, `y`, `x`.

## Phase 5: Terminal Status Propagation For UI And MCP

Problem:

The server lifecycle is capable of reporting terminal failed status, but UI/MCP
testing still produced "seems stuck" symptoms. The architecture needs one
observable status model.

Refactor:

- Use `zmqruntime` execution lifecycle as the status source of truth.
- UI plate manager, pipeline editor, MCP tools, and direct clients should all
  read the same typed execution record/snapshot.
- Progress projection must merge progress events with terminal lifecycle state,
  with lifecycle terminal state taking precedence.
- On failure, UI and MCP surfaces must show:
  - terminal `failed`
  - execution id
  - failing step name/index when known
  - error message
  - traceback access through bounded detail
- Add "no active workers" to the observable state so clients can distinguish
  terminal failure from a real hang.

Validation:

- Submit a deliberately failing pipeline.
- Verify direct client, MCP, and UI all show failed status within one polling
  interval.
- Verify no plate remains visually stuck in compiling/running after server
  lifecycle enters failed/cancelled/complete.

## Phase 6: Headless Viewer Smoke Harness

Problem:

Streaming bugs currently require manual UI interpretation. The agent needs a
closed loop that can run a known pipeline, inspect viewer state, and capture
screenshots without guessing.

Refactor:

- Add an end-to-end smoke harness that:
  - starts or connects to the execution server
  - starts or connects to Napari viewer server
  - prepares a pycodified pipeline from a fixture dataset
  - runs two wells with streaming enabled
  - polls execution lifecycle
  - queries viewer state over ZMQ
  - captures a Napari screenshot
  - asserts layer count, layer names, shape, axis labels, route offsets, and
    nonzero full-data/current-slice stats for expected planes
- Keep the harness usable by MCP tools so agents and reviewers can run it.
- Make long waits bounded and explicit. A viewer/control request should fail
  loudly with its endpoint and operation name, not silently consume minutes.

Validation:

- `segmentation_final` passes when produced-output semantics and CP artifact
  contracts are both load bearing.
- A one-step smoke pipeline validates viewer routing while deeper CP artifact
  and object-measurement contracts are being repaired.
- Napari crash/disconnect produces terminal diagnostic output, not a stuck wait.

## Phase 7: Fiji/ROI Parity

Problem:

Earlier ROI streaming failed because component metadata was missing and the
streaming backend tried to parse an OpenHCS-generated ROI filename. That is the
same class of leak as image streaming metadata drift.

Refactor:

- Route ROI/artifact streaming through the same typed stream request DTO.
- Require `ProducedOutputSemantics` for OpenHCS-generated artifacts.
- Make Fiji and Napari artifact viewers consume common artifact identity and
  component metadata records.
- Remove implicit filename-parse fallback from generated-artifact streaming.

Validation:

- Stream ROI ZIPs to Fiji and Napari with explicit metadata.
- Confirm generated ROI filenames are display/storage names, not the semantic
  authority for axes.

## Priority Order

1. Add the produced-output semantic record and make the step output manifest
   store records instead of path-only tuples.
2. Build stream request DTOs from produced-output semantics; demote kwargs to a
   public compatibility adapter.
3. Collapse `StreamProducerIdentity` payload/constructor projection into one
   declared schema and remove reflection/fallback at that boundary.
4. Require explicit component metadata for every OpenHCS-generated image,
   object, and ROI/artifact stream.
5. Add Napari layer-state/current-slice inspection and assert nonzero data for
   expected route-local planes.
6. Repair any remaining CellProfiler artifact contract gaps exposed by the full
   smoke run.
7. Wire lifecycle terminal state into UI and MCP as a single typed status model.
8. Extend the smoke harness to full beginner segmentation, ROI/Fiji, screenshots,
   and MCP-driven execution.

## Advisor And Test Gates

Run after production code changes:

```bash
.venv/bin/python -m compileall -q openhcs
.venv/bin/python -m pytest \
  tests/unit/test_viewer_protocol.py \
  tests/unit/test_napari_streaming_handlers.py \
  tests/unit/test_streaming_service.py \
  tests/unit/test_runtime_semantics.py \
  tests/unit/test_runtime_values.py \
  tests/unit/test_function_runtime_source_projection.py \
  tests/unit/test_function_outputs.py \
  tests/unit/test_function_artifact_materialization.py \
  tests/unit/test_materialization_core.py \
  -q
```

Advisor scans:

- Production OpenHCS files touched in the current phase.
- Do not scan test files.
- For shared DTO work, scan the matching production files in `external/PolyStore`
  and `external/zmqruntime`.

End-to-end gates:

```bash
# compile-only
.venv/bin/python scripts/smoke_compile_cellprofiler_beginner_segmentation.py

# execute + Napari stream + viewer-state assertions
.venv/bin/python scripts/smoke_stream_cellprofiler_beginner_segmentation.py
```

The scripts above do not exist yet; they are part of Phase 5.
