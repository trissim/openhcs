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

