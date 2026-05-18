# Napari Streaming Handler Axis Refactor - 2026-05-18

## Advisor Evidence

Full-repo scan flagged a parallel enum-keyed table:

- `openhcs/runtime/napari_stream_visualizer.py`
- `openhcs/runtime/napari_viewer_server.py`
- duplicated `_DATA_TYPE_HANDLERS`
- shared axis: `StreamingDataType.IMAGE`, `StreamingDataType.POINTS`,
  `StreamingDataType.SHAPES`

Both modules define the same data-type dispatch table with per-type
`build_nd_data` and `create_layer` handlers.

## Current Problem

The handler table is a real semantic axis:

- the key is `StreamingDataType`;
- values are paired data-build and layer-create operations;
- both runtime viewer modules need the same concept.

Duplicating the table makes it easy for one viewer path to drift from the other.

## Target Shape

Create a shared Napari streaming handler module, for example:

```python
# openhcs/runtime/napari_streaming_handlers.py

@dataclass(frozen=True, slots=True)
class NapariStreamingDataTypeHandler:
    data_type: StreamingDataType
    build_nd_data: Callable[..., object]
    create_layer: Callable[..., object]
```

Expose:

- `NAPARI_STREAMING_DATA_TYPE_HANDLERS`
- `handler_for_streaming_data_type(data_type)`
- optionally a small `NapariLayerUpdateRequest` if argument lists are repeated
  across modules.

## Phase 1: Extract Shared Record

Move only the table shape first. Keep module-specific helper functions where
they currently live if they depend on local viewer behavior.

If helper function bodies are identical, move them with the table. If they only
look identical but depend on different imports/runtime state, keep wrappers and
share the record type.

## Phase 2: Replace Duplicated Tables

Update both modules to import the shared handler table or to build their local
table from shared records.

The refactor should make the axis authority explicit without changing layer
creation behavior.

## Phase 3: Add Drift Tests

Add tests that assert:

- both runtime paths expose the same `StreamingDataType` keys;
- each key has both `build_nd_data` and `create_layer`;
- handler lookup fails loudly for unknown data types;
- image/shapes/points handlers still call the expected local functions.

## Phase 4: Consider Viewer Abstraction

Only after the table is shared, inspect whether
`napari_stream_visualizer.py` and `napari_viewer_server.py` duplicate broader
display lifecycle behavior. If yes, plan a separate viewer-service refactor.

Do not fold the two modules together in this campaign.

## Risks

- Napari imports may be optional or environment-sensitive. Keep imports lazy.
- Handler functions may close over module-specific state. Verify before moving
  implementations.
- Streaming behavior may be used interactively; preserve names and error text.

## Verification Gates

```bash
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python - <<'PY'
import openhcs.runtime.napari_stream_visualizer
import openhcs.runtime.napari_viewer_server
PY
python -m nominal_refactor_advisor \
  openhcs/runtime/napari_stream_visualizer.py \
  openhcs/runtime/napari_viewer_server.py
```

## Completion Criteria

- `_DATA_TYPE_HANDLERS` is not independently maintained in both runtime modules.
- `StreamingDataType` dispatch has one authoritative typed surface.
- The full-repo parallel enum-keyed table finding is removed.
