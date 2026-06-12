# Clean Streaming Identity And Viewer Routing Plan - 2026-06-11

## Purpose

Replace the overloaded streaming `source` display component with explicit stream
identity.

The goal is not to hide `source_` with naming rules. The goal is to stop using
`source` as a viewer component at all. A stream item needs two separate ideas:

- producer identity: which step/output/manual action produced this item;
- source-plane metadata: real image coordinates such as well, site, channel,
  z_index, and timepoint.

Napari and Fiji should route by explicit producer/output identity plus real
display-mode components. Visible titles should be presentation only.

## Dry-Run Conclusion

The implementation is feasible, but a clean refactor is broader than a Napari
title fix.

Current behavior from code reading:

- a `FunctionStep` chain streams the final main output after the whole step
  completes;
- intermediate function invocations inside the same step do not stream as
  separate main-image layers today;
- therefore multiple functions inside one step do not normally overwrite each
  other as main image layers;
- step artifacts do stream independently through artifact materialization;
- duplicate `step_name` values and multiple artifacts from one step can still
  collide because route identity is currently derived from `source`, component
  metadata, and data type.

The clean fix crosses these boundaries:

- OpenHCS execution plans must provide producer/output identity;
- PolyStore streaming messages must carry that identity without converting it
  into component metadata;
- Napari must key internal state by route identity and create layers with a
  separate visible title;
- Fiji must build window groups from producer identity plus real window-mode
  components, not from `source`;
- tests must stop expecting `source` in streamed metadata.

## Non-Negotiables

- Remove `source` from streaming component modes.
- Remove `source` from streamed component metadata.
- Remove `source_mode` from Napari/Fiji streaming configs.
- Do not add compatibility aliases such as `source` or `unknown_source`.
- Do not make `source_mode=stack` as a workaround.
- Do not use visible layer names as authoritative route keys.
- Keep image, shapes/labels, and points routes distinct.
- Keep "source projection" terminology only where it means real OpenHCS data
  provenance, not viewer display grouping.

## Architecture Boundary

OpenHCS owns:

- pipeline step identity;
- artifact output identity;
- source projection and source-plane metadata;
- viewer defaults for OpenHCS workflows.

PolyStore owns:

- generic streaming backend payload assembly;
- generic receiver-side grouping primitives;
- generic route-key helpers that do not know OpenHCS pipeline semantics beyond
  the identity payload they receive.

Napari receiver owns:

- Napari layer objects;
- route-key to layer-object state;
- debounced/batched layer updates;
- manual deletion reconciliation;
- visible layer titles.

Fiji receiver owns:

- Fiji window and hyperstack grouping;
- ImageJ channel/slice/frame mapping;
- visible window titles.

zmqruntime remains transport only.

## Target Identity Model

Use explicit nominal records. Exact names can change, but these roles should not
collapse back into component metadata.

```python
@dataclass(frozen=True, slots=True)
class StreamProducerIdentity:
    origin: Literal["pipeline", "manual", "direct"]
    output_kind: Literal["main", "artifact", "invocation", "manual", "direct"]
    output_key: str
    step_name: str | None = None
    pipeline_position: int | None = None
    step_scope_id: str | None = None
    invocation_key: str | None = None
    artifact_kind: str | None = None


@dataclass(frozen=True, slots=True)
class StreamItemIdentity:
    producer: StreamProducerIdentity
    data_type: StreamingDataType
    component_metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ViewerRoute:
    key: str
    producer: StreamProducerIdentity
    data_type: StreamingDataType
    route_components: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ViewerPresentation:
    title: str
    route: ViewerRoute
```

Route identity should include:

- `origin`;
- `output_kind`;
- `output_key`;
- `pipeline_position` when present;
- `step_scope_id` when present;
- `invocation_key` only when invocation streaming exists;
- `artifact_kind` when present;
- `StreamingDataType`;
- only the real components whose configured mode creates a separate route or
  window.

Route identity should not include execution/run id by default. Re-running the
same pipeline step should replace/update the same logical viewer route unless a
future "preserve runs" mode is intentionally added.

## Message Schema Direction

Put identity on each streamed item, not only at the batch level. Current OpenHCS
paths usually send one producer per batch, but per-item identity is the cleaner
model and avoids another schema change if mixed batches appear later.

Conceptually:

```python
{
    "images": [
        {
            "path": "...",
            "metadata": {
                "component_metadata": {
                    "well": "A01",
                    "site": "1",
                    "channel": "DAPI",
                },
                "producer_identity": {
                    "origin": "pipeline",
                    "output_kind": "artifact",
                    "output_key": "nuclei",
                    "step_name": "OverlayObjects",
                    "pipeline_position": 3,
                    "step_scope_id": "...",
                    "artifact_kind": "rois",
                },
            },
        }
    ],
    "component_modes": {
        "well": "stack",
        "site": "stack",
        "channel": "stack",
        "z_index": "stack",
        "timepoint": "stack",
    },
}
```

No `source` field should be added to `component_metadata`.

## Display Modes

### Napari

Napari component modes should cover only real source-plane components.

| Component | Default mode |
| --- | --- |
| `well` | `stack` |
| `site` | `stack` |
| `channel` | `stack` |
| `z_index` | `stack` |
| `timepoint` | `stack` |

Napari routes are producer/output/data-type identity plus components configured
as `slice`.

### Fiji

Fiji component modes should also cover only real source-plane components.

| Mode | Meaning |
| --- | --- |
| `window` | Component value creates a separate Fiji window. |
| `channel` | Component maps to ImageJ channel dimension. |
| `slice` | Component maps to ImageJ Z/slice dimension. |
| `frame` | Component maps to ImageJ frame/time dimension. |

Fiji window identity is producer/output/data-type identity plus components
configured as `window`.

### Data Type

`StreamingDataType` is not a display component. It is always part of route
identity.

| Data type | Route impact |
| --- | --- |
| `image` | Separate image route. |
| `shapes` | Separate ROI/labels route. |
| `points` | Separate points route. |

## Sender Refactor

### Main Step Outputs

Files:

- `openhcs/core/steps/function_outputs.py`
- `external/PolyStore/src/polystore/streaming/_streaming_backend.py`

Work:

- remove `kwargs["source"] = plan.step_name`;
- construct a `StreamProducerIdentity` with `origin="pipeline"`,
  `output_kind="main"`, `output_key="main"`, `step_name`,
  `pipeline_position`, and `step_scope_id`;
- pass producer identity through backend kwargs;
- keep `component_metadata_by_path` for real source-plane metadata only.

### Artifact Outputs

Files:

- `openhcs/core/steps/function_artifact_materialization.py`
- `openhcs/processing/materialization/core.py`
- `external/PolyStore/src/polystore/streaming/_streaming_backend.py`

Work:

- remove artifact-path `kwargs["source"]`;
- construct producer identity inside the artifact loop so `output_key` and
  `artifact_kind` are per artifact output;
- replace `_backend_kwargs_with_component_metadata()` with a helper that attaches
  both producer identity and component metadata to the save call;
- ensure the single-item `filemanager.save()` path preserves identity into
  `StreamingBackend.save_batch()`.

### Manual And Direct Streaming

Manual/UI streaming should synthesize explicit producer identities:

- `origin="manual"`;
- `output_kind="manual"`;
- `output_key` derived from the selected path, artifact key, or caller-provided
  label;
- no `source` fallback.

Direct helper display should either require identity or synthesize:

- `origin="direct"`;
- `output_kind="direct"`;
- `output_key` from the helper label or path stem.

## PolyStore Refactor

Files:

- `external/PolyStore/src/polystore/streaming/_streaming_backend.py`
- `external/PolyStore/src/polystore/napari_stream.py`
- `external/PolyStore/src/polystore/streaming/receivers/napari/layer_key.py`
- `external/PolyStore/src/polystore/streaming/receivers/core/window_projection.py`

Work:

- remove `StreamingComponentMetadata.source`;
- remove `metadata["source"] = ...` from `to_payload()`;
- remove any default `source` or `unknown_source` behavior from stream kwargs;
- validate that every streamed item has producer identity and real component
  metadata;
- replace `build_layer_key()` with a route builder that accepts producer
  identity, data type, component metadata, and selected route/window components;
- remove source-specific ROI/window normalization from receiver core.

## Napari Receiver Refactor

Files:

- `openhcs/runtime/napari_viewer_server.py`
- `openhcs/runtime/napari_streaming_handlers.py`
- `external/PolyStore/src/polystore/streaming/receivers/napari/napari_batch_processor.py`

Work:

- parse producer identity from each stream item;
- build a hidden `route_key` from producer identity, data type, and real
  slice-mode components;
- build a visible title separately;
- key `component_groups`, debounced processors, state-store entries, and
  replacement logic by `route_key`;
- create/update Napari layers with visible title only;
- track route key to layer object so manual deletion reconciliation does not
  depend on `viewer.layers` names.

Recommended title policy:

- main pipeline output: `step_name`;
- artifact output: `step_name output_key` plus data-type suffix when useful;
- direct/manual output: caller label or path stem;
- real slice components: append compact `component_value` or
  `component_name_value` only when needed for readability;
- duplicate visible titles: append a compact disambiguator such as
  `[step 4]` or `[nuclei]`.

Examples:

| Scenario | Internal route includes | Visible title |
| --- | --- | --- |
| Step 3 main image | step 3 + main + image | `OverlayObjects` |
| Step 3 nuclei ROIs | step 3 + artifact `nuclei` + shapes | `OverlayObjects nuclei labels` |
| Step 3 spots points | step 3 + artifact `spots` + points | `OverlayObjects spots points` |
| Step 4 same name | step 4 + main + image | `OverlayObjects [step 4]` if title collides |
| Channel sliced image | producer + image + `channel=2` | `OverlayObjects channel 2` |

## Fiji Receiver Refactor

Files:

- `openhcs/runtime/fiji_viewer_server.py`
- `external/PolyStore/src/polystore/streaming/receivers/core/window_projection.py`

Work:

- build window keys from producer identity plus real `window` mode components;
- keep channel/slice/frame mapping for real components only;
- remove `source` special cases from ROI normalization;
- build visible window titles from producer/output identity and real window
  component values.

## Config Refactor

Files:

- `openhcs/core/config.py`
- `openhcs/core/streaming_config_factory.py`
- `openhcs/utils/display_config_factory.py`

Work:

- stop passing `VirtualComponents.SOURCE` into Napari/Fiji streaming display
  configs;
- make streaming `COMPONENT_ORDER` contain only real source-plane components;
- remove generated `source_mode` fields from Napari/Fiji streaming configs;
- fix `component_modes()` to resolve config defaults through normal config
  access, not `object.__getattribute__`;
- add direct-instance tests showing Napari/Fiji default component modes work and
  do not include `source`.

## Implementation Phases

### Phase 1: Config And Component Cleanup

- Remove virtual `source` from streaming display config generation.
- Fix `component_modes()` default resolution.
- Update tests that assert component mode maps.

### Phase 2: Identity Records And Payload Schema

- Add producer/item identity records.
- Update PolyStore message construction to serialize identity per item.
- Remove `source` from streamed metadata.
- Add validation failures for missing identity in normal stream paths.

### Phase 3: Sender Threading

- Thread main step identity from `FunctionStepExecutionPlan`.
- Thread artifact identity per `artifact output_key`.
- Thread manual/direct identity from caller/path labels.
- Verify `filemanager.save()` and `save_batch()` both preserve identity.

### Phase 4: Receiver Routing

- Replace source-based Napari route keys with identity-based route keys.
- Split route key from visible title.
- Update batch processor and state store to carry route/title objects.
- Fix manual deletion reconciliation by tracking layer objects.

### Phase 5: Fiji Window Routing

- Replace source-based window grouping with identity-based grouping.
- Remove source-specific ROI normalization.
- Update Fiji tests for real component modes only.

### Phase 6: Tests And Smoke Checks

Add or update tests for:

- `NapariStreamingConfig().component_modes()` without `source`;
- `FijiStreamingConfig().component_modes()` without `source`;
- streamed metadata does not contain `source`;
- same `step_name` but different `pipeline_position` routes separately;
- one step with multiple artifact `output_key` values routes separately;
- image/shapes/points route separation;
- visible Napari titles do not include `source_`;
- manual deletion works when visible title differs from route key;
- Fiji window grouping uses producer identity.

Focused command set:

```bash
PYTHONPATH=/home/ts/code/projects/openhcs-benchmark-platform:/home/ts/code/projects/nominal-refactor-advisor \
  uv run --no-sync pytest \
    tests/unit/test_function_outputs.py \
    tests/unit/test_streaming_service.py \
    tests/unit/test_napari_streaming_handlers.py \
    external/PolyStore/tests/test_streaming_metadata.py \
    external/PolyStore/tests/test_streaming_receiver_core.py \
    -q
```

## Expected Breakage

This is intentional breakage from the old model:

- saved configs or UI schemas that reference `source_mode` will fail until
  updated;
- tests expecting `source` in metadata must be rewritten;
- source-specific ROI/window normalization tests should be removed or replaced;
- callers that stream without explicit identity must be fixed to pass manual or
  direct identity;
- any code assuming Napari layer name equals route key must be updated.

## Completion Criteria

- No streaming component mode map contains `source`.
- No streamed item component metadata contains `source`.
- Internal viewer route identity is producer-qualified, output-qualified, and
  data-type-qualified.
- Same-named steps do not overwrite each other.
- Multiple artifacts from one step do not overwrite each other.
- Repeated streams from the same logical producer/output update the same route.
- Napari visible names are clean and do not contain `source_`.
- Fiji windows are grouped without source-specific hacks.
- Manual deletion reconciliation still works after route/title separation.
