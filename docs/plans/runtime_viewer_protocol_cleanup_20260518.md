# Runtime Viewer And Streaming Protocol Cleanup - 2026-05-18

## Full-Scan Evidence

The full scan shows high active finding density in:

- `openhcs/runtime/napari_stream_visualizer.py`
- `openhcs/runtime/napari_viewer_server.py`
- `openhcs/runtime/fiji_viewer_server.py`
- `openhcs/runtime/fiji_stream_visualizer.py`

Finding families include:

- closed-family string dispatch;
- manual class-marker membership;
- bidirectional registries;
- repeated threaded parameter families;
- oversized viewer/server orchestration methods;
- unreferenced private functions.

## Current State

The first-wave Napari campaign removed duplicated data-type handler tables.
Remaining debt is deeper: viewer/server code still recovers runtime roles from
strings, markers, and mirrored maps.

## Target Shape

Create typed viewer protocol records:

- `ViewerIdentity`
- `ViewerLifecycleState`
- `ViewerDimensionUpdate`
- `ViewerLayerUpdate`
- `ViewerConnectionRegistry`
- `ViewerEventRoute`
- `StreamingViewerBackend`

Napari/Fiji-specific code should implement these protocols rather than each
server maintaining its own string/marker/bidirectional logic.

## Phases

1. Characterize viewer import behavior with mocked optional dependencies.
2. Extract shared lifecycle/identity records without changing viewer startup.
3. Replace duplicated dimension/layer dispatch with route tables.
4. Replace manual bidirectional registries with one typed connection registry.
5. Split large listener/update methods into event-route stages.
6. Delete or make explicit unreferenced private viewer helpers.

## Verification Gates

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
.venv/bin/python - <<'PY'
import openhcs.runtime.napari_stream_visualizer
import openhcs.runtime.napari_viewer_server
import openhcs.runtime.fiji_stream_visualizer
import openhcs.runtime.fiji_viewer_server
PY
timeout 120 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/runtime/napari_stream_visualizer.py \
  openhcs/runtime/napari_viewer_server.py \
  openhcs/runtime/fiji_stream_visualizer.py \
  openhcs/runtime/fiji_viewer_server.py
```

## Completion Criteria

- Viewer lifecycle and connection identity are typed.
- Napari/Fiji route differences are explicit backend variants.
- Optional GUI/runtime imports remain lazy and smoke-testable.

## Execution Log

### Checkpoint 1

Implemented:

- `NapariLayerUpdateAuthority`
- `NapariLayerUpdateRequest`
- `NapariViewerServerRequest`
- shared use of `ViewerQtEnvironmentPolicy`

Changed behavior boundaries:

- Napari image/shapes/points layer replacement mechanics now have one shared
  owner in `napari_streaming_handlers.py`.
- `napari_stream_visualizer.py` and `napari_viewer_server.py` keep their public
  helper wrappers but no longer duplicate the layer replacement mechanics.
- Dead `NapariStreamVisualizer._prepare_data_for_display` was deleted after
  repository-wide call-site verification.
- Repeated Qt platform setup now uses `ViewerQtEnvironmentPolicy`.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 5 passed
```

Advisor status:

- Cleared duplicated Napari create/update layer helper findings.
- Cleared dead `_prepare_data_for_display` finding.
- Cleared Napari platform dispatch findings.
- Remaining: component/shape dispatch, Napari viewer-server role quotient,
  Fiji dimension-listener hub, Fiji dimension context record, and public
  Napari/Fiji process signature-family cleanup.

### Checkpoint 2

Implemented:

- Napari `StreamingDataType` route table for layer update execution.
- Deletion of dead `_parse_component_info_from_path` helpers from both Napari
  runtime modules.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 5 passed
```

Advisor status:

- Cleared Napari data-type strategy ladder.
- Cleared unreferenced private component parser findings.
- Remaining high-value work is now the larger decompositions: component/shape
  dispatch, Napari server role quotient, Fiji dimension-listener hub, Fiji
  dimension context record, and public process signature-family cleanup.

### Checkpoint 3

Implemented:

- `ComponentDimensionLabelPolicy` in `viewer_protocol.py`.
- Shared Napari dimension-label construction in both Napari runtime modules.
- Unit coverage for channel, well, generic metadata, and metadata-ignored
  fallback label behavior.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 6 passed
```

Advisor status:

- Cleared the Napari component-label string dispatch findings.
- Remaining viewer protocol findings are shape rasterization dispatch, stream
  visualizer lifecycle/membership boundaries, Napari/Fiji server role
  quotients, Fiji dimension context records, process signature records, and
  platform strategy ladders in the shared viewer protocol.

### Checkpoint 4

Implemented:

- `NapariShapeKind`, `NapariShapePaintContext`, and
  `NapariShapeLabelRasterizer` as the shared dense-label conversion authority.
- Direct rasterizer calls from both Napari runtime modules, deleting the
  forwarding `_shapes_to_labels` wrappers.
- `NapariLayerLogPolicy` for counted vs uncounted layer creation logging.
- Unit coverage for polygon/path raster projection and legacy point extent
  behavior.

Verification:

```bash
.venv/bin/python -m pytest tests/unit/test_napari_streaming_handlers.py -q
# 8 passed
```

Advisor status:

- Cleared Napari shape-kind string dispatch.
- Cleared Napari `_shapes_to_labels` identity-forwarding shells.
- Cleared inline layer-kind subset logging policy.
- Remaining viewer work is now lifecycle/membership boundaries,
  request-builder projection records, Napari/Fiji server role quotients, Fiji
  dimension context records, process signature records, and shared viewer
  platform strategy ladders.
