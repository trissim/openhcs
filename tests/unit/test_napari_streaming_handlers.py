from __future__ import annotations

import numpy as np
import pytest

from polystore.streaming_constants import StreamingDataType
from polystore.streaming.identity import StreamProducerIdentity

from openhcs.runtime.viewer_protocol import (
    ComponentDimensionLabelPolicy,
    ViewerComponentValueOrdering,
)
from openhcs.runtime.napari_streaming_handlers import (
    NapariBatchProcessorStore,
    NapariComponentMetadataNormalizer,
    NapariComponentValueTracker,
    NapariDisplayAxisDomain,
    NapariLayerUpdateAuthority,
    NapariLayerStateStore,
    NapariShapeLabelRasterizer,
    NapariStreamLayerItem,
    NapariStreamingDataTypeHandler,
    build_napari_streaming_data_type_handlers,
    napari_streaming_data_type_handler,
)


def _handler_marker(name: str):
    def marker(*args, **kwargs):
        return name, args, kwargs

    return marker


def _layer_item(
    components: dict,
    data=None,
    data_type: StreamingDataType = StreamingDataType.IMAGE,
) -> NapariStreamLayerItem:
    return NapariStreamLayerItem(
        data=data,
        components=components,
        path="test",
        data_type=data_type,
    )


def test_build_napari_streaming_data_type_handlers_declares_full_axis() -> None:
    build_image = _handler_marker("build_image")
    create_image = _handler_marker("create_image")
    build_shapes = _handler_marker("build_shapes")
    create_shapes = _handler_marker("create_shapes")
    build_points = _handler_marker("build_points")
    create_points = _handler_marker("create_points")

    handlers = build_napari_streaming_data_type_handlers(
        build_image_data=build_image,
        create_image_layer=create_image,
        build_shapes_data=build_shapes,
        create_shapes_layer=create_shapes,
        build_points_data=build_points,
        create_points_layer=create_points,
    )

    assert set(handlers) == {
        StreamingDataType.IMAGE,
        StreamingDataType.SHAPES,
        StreamingDataType.POINTS,
    }
    assert handlers[StreamingDataType.IMAGE] == NapariStreamingDataTypeHandler(
        data_type=StreamingDataType.IMAGE,
        build_nd_data=build_image,
        create_layer=create_image,
    )
    assert handlers[StreamingDataType.SHAPES].build_nd_data is build_shapes
    assert handlers[StreamingDataType.POINTS].create_layer is create_points


def test_napari_streaming_data_type_handler_accepts_string_data_type() -> None:
    handlers = build_napari_streaming_data_type_handlers(
        build_image_data=_handler_marker("build_image"),
        create_image_layer=_handler_marker("create_image"),
        build_shapes_data=_handler_marker("build_shapes"),
        create_shapes_layer=_handler_marker("create_shapes"),
        build_points_data=_handler_marker("build_points"),
        create_points_layer=_handler_marker("create_points"),
    )

    assert (
        napari_streaming_data_type_handler(handlers, "image").data_type
        is StreamingDataType.IMAGE
    )


def test_napari_streaming_data_type_handler_fails_loudly_for_missing_axis() -> None:
    with pytest.raises(ValueError, match="No Napari streaming handler registered"):
        napari_streaming_data_type_handler({}, StreamingDataType.IMAGE)


class _FakeLayerList(list):
    def remove(self, layer):
        super().remove(layer)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.dims = type("Dims", (), {"axis_labels": None})()
        self.calls = []

    def add_image(self, data, *, name, **kwargs):
        return self._add_layer("image", data, name, kwargs)

    def add_shapes(self, data, *, name, **kwargs):
        return self._add_layer("shapes", data, name, kwargs)

    def add_points(self, data, *, name, **kwargs):
        return self._add_layer("points", data, name, kwargs)

    def _add_layer(self, layer_type, data, name, kwargs):
        layer = type("Layer", (), {"name": name, "data": data, "kwargs": kwargs})()
        self.layers.append(layer)
        self.calls.append((layer_type, data, name, kwargs))
        return layer


class _FakeLayerState:
    def __init__(self):
        self.labels = {}

    def set_labels(self, layer_key, labels):
        self.labels[layer_key] = labels


class _FakeNapariServer:
    def __init__(self):
        self.layer_state = _FakeLayerState()
        self.component_metadata = {"channel": {"1": "DAPI"}}


def test_napari_layer_update_authority_replaces_existing_image_and_axis_labels():
    viewer = _FakeViewer()
    layers = {}
    authority = NapariLayerUpdateAuthority()

    first = authority.create_or_update_image(
        viewer,
        layers,
        "route-nuclei",
        "Nuclei",
        "image-1",
        colormap=None,
        axis_labels=("z", "y", "x"),
    )
    second = authority.create_or_update_image(
        viewer,
        layers,
        "route-nuclei",
        "Nuclei",
        "image-2",
        colormap="green",
        axis_labels=("c", "y", "x"),
    )

    assert first not in viewer.layers
    assert second in viewer.layers
    assert layers == {"route-nuclei": second}
    assert viewer.dims.axis_labels == ("c", "y", "x")
    assert viewer.calls[0] == ("image", "image-1", "Nuclei", {"colormap": "gray"})
    assert viewer.calls[1] == ("image", "image-2", "Nuclei", {"colormap": "green"})


def test_napari_display_pipeline_applies_dimension_labels_to_layer_state():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())

    axis_labels = pipeline.apply_dimension_labels(
        "route-labels",
        ("channel",),
        {"channel": [1, 2]},
    )

    assert axis_labels == ("channel", "y", "x")
    assert pipeline.server.layer_state.labels["route-labels"] == {
        "channel": ["Ch1: DAPI", "Ch 2"],
    }


def test_napari_axis_policy_uses_shared_display_domain_not_local_shape_axes():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    policy = napari_viewer_server.NapariLayerAxisPolicy()
    component_values = {
        "site": [1, 2],
        "timepoint": [1],
        "channel": [1, 2, 3, 4, 5],
        "z_index": [1],
        "well": ["A01"],
    }

    axes = policy.stack_components_for(
        stack_components=("site", "timepoint", "channel", "z_index", "well"),
        component_values=component_values,
    )

    assert axes == ("site", "channel")


def test_napari_layer_update_authority_declares_shapes_and_points_kwargs():
    viewer = _FakeViewer()
    layers = {}
    authority = NapariLayerUpdateAuthority()

    shapes = authority.create_or_update_shapes(
        viewer,
        layers,
        "route-rois",
        "ROIs",
        shapes_data=[[[0, 0], [1, 1]]],
        shape_types=["polygon"],
        properties={"id": [1]},
    )
    points = authority.create_or_update_points(
        viewer,
        layers,
        "route-spots",
        "Spots",
        points_data=[[0, 0]],
        properties={"id": [2]},
    )

    assert layers == {"route-rois": shapes, "route-spots": points}
    assert viewer.calls[0][3] == {
        "shape_type": ["polygon"],
        "properties": {"id": [1]},
        "edge_color": "red",
        "face_color": "transparent",
        "edge_width": 2,
    }
    assert viewer.calls[1][3] == {
        "properties": {"id": [2]},
        "face_color": "green",
        "size": 3,
    }


def test_napari_layer_title_authority_uses_stream_display_name_policy():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    producer = StreamProducerIdentity.pipeline_output(
        output_kind="artifact",
        output_key="NucleiObjects3D",
        step_name="ConvertObjectsToImage",
        pipeline_position=8,
        artifact_kind="object_labels",
    )
    component_layout = napari_viewer_server.ComponentLayout(
        component_modes={"well": "slice", "channel": "stack"},
        component_order=["well", "channel"],
    )

    title = napari_viewer_server.NapariLayerTitleAuthority.title(
        producer=producer,
        data_type=StreamingDataType.SHAPES,
        component_info={"well": "A01", "channel": 1},
        component_layout=component_layout,
    )

    assert title == "9. ConvertObjectsToImage NucleiObjects3D well A01 labels"


def test_napari_layer_title_disambiguation_uses_display_step_number():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    producer = StreamProducerIdentity.pipeline_output(
        output_kind="main",
        output_key="main",
        step_name="Measure",
        pipeline_position=3,
    )
    layer_state = NapariLayerStateStore.empty()
    layer_state.set_title("other-route", "4. Measure")

    assert (
        napari_viewer_server.NapariLayerTitleAuthority.disambiguate(
            title="4. Measure",
            producer=producer,
            route_key="current-route",
            layer_state=layer_state,
        )
        == "4. Measure [step 4]"
    )


class _FakeTimer:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_napari_layer_state_store_keeps_layer_labels_and_timers_together():
    store = NapariLayerStateStore.empty()
    timer = _FakeTimer()
    layer = object()

    store.set_layer("nuclei", layer)
    store.set_labels("nuclei", {"channel": ["Ch 1"]})
    store.set_pending_update("nuclei", timer)

    assert store.has_layer("nuclei")
    assert store.layer("nuclei") is layer
    assert store.labels_for("nuclei") == {"channel": ["Ch 1"]}
    assert store.cancel_pending_update("nuclei")
    assert timer.stopped
    assert store.pop_pending_update("nuclei") is timer
    assert store.labels_for("missing") == {}


def test_napari_batch_processor_store_creates_one_processor_per_layer(monkeypatch):
    import polystore.streaming.receivers.napari as napari_receivers

    created = []

    class FakeBatchProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(napari_receivers, "NapariBatchProcessor", FakeBatchProcessor)

    store = NapariBatchProcessorStore(debounce_delay_ms=123, max_debounce_wait_ms=456)
    server = object()

    first = store.get_or_create(layer_key="nuclei", napari_server=server, batch_size=7)
    second = store.get_or_create(layer_key="nuclei", napari_server=server, batch_size=9)

    assert first is second
    assert len(created) == 1
    assert first.kwargs == {
        "napari_server": server,
        "batch_size": 7,
        "debounce_delay_ms": 123,
        "max_debounce_wait_ms": 456,
    }


def test_napari_component_metadata_normalizer_coerces_indexed_strings():
    normalizer = NapariComponentMetadataNormalizer()

    normalized = normalizer.normalize(
        {
            "well": "A01",
            "site": "1",
            "channel": "2",
            "z_index": "0",
            "timepoint": "3",
            "source": "raw",
        }
    )

    assert normalized == {
        "well": "A01",
        "site": 1,
        "channel": 2,
        "z_index": 0,
        "timepoint": 3,
        "source": "raw",
    }


def test_napari_component_value_tracker_tracks_observed_axis_values_by_route():
    tracker = NapariComponentValueTracker()

    tracker.update(
        "main",
        ["channel", "well"],
        [
            _layer_item({"channel": 2, "well": "A01"}),
            _layer_item({"channel": 4, "well": "B02"}),
        ],
    )
    tracker.update(
        "artifact",
        ["channel", "well"],
        [_layer_item({"channel": 1, "well": "A01"})],
    )

    assert tracker.values_for("main", ["channel", "well"]) == {
        "channel": [2, 4],
        "well": ["A01", "B02"],
    }
    assert tracker.values_for("artifact", ["channel", "well"]) == {
        "channel": [1],
        "well": ["A01"],
    }
    assert tracker.values_for("main", ["site"]) == {"site": []}


def test_napari_display_axis_domain_tracks_shared_active_axis_values():
    domain = NapariDisplayAxisDomain()
    stack_components = ["site", "timepoint", "channel", "z_index", "well"]

    domain.update(
        stack_components,
        [
            _layer_item(
                {
                    "site": site,
                    "timepoint": 1,
                    "channel": channel,
                    "z_index": 1,
                    "well": "A01",
                }
            )
            for site in [1, 2]
            for channel in [1, 2, 3, 4, 5]
        ],
    )
    domain.update(
        stack_components,
        [
            _layer_item(
                {
                    "site": 1,
                    "timepoint": 1,
                    "channel": 1,
                    "z_index": 1,
                    "well": "A01",
                }
            )
        ],
    )

    assert domain.values_for(stack_components) == {
        "site": [1, 2],
        "timepoint": [1],
        "channel": [1, 2, 3, 4, 5],
        "z_index": [1],
        "well": ["A01"],
    }


def test_viewer_component_value_ordering_sorts_unpadded_indices_numerically():
    assert sorted(
        [1, 10, 100, 11, 2, 20, 3, 9],
        key=ViewerComponentValueOrdering.key,
    ) == [1, 2, 3, 9, 10, 11, 20, 100]
    assert sorted(
        ["1", "10", "100", "11", "2", "20", "3", "9"],
        key=ViewerComponentValueOrdering.key,
    ) == ["1", "2", "3", "9", "10", "11", "20", "100"]
    assert sorted(
        [("A01", "10"), ("A01", "2"), ("A01", "1")],
        key=ViewerComponentValueOrdering.tuple_key,
    ) == [("A01", "1"), ("A01", "2"), ("A01", "10")]


def test_napari_component_value_tracker_sorts_unpadded_indices_numerically():
    tracker = NapariComponentValueTracker()

    tracker.update(
        "main",
        ["z_index"],
        [
            _layer_item({"z_index": value})
            for value in [1, 10, 100, 11, 2, 20, 3, 9]
        ],
    )

    assert tracker.values_for("main", ["z_index"]) == {
        "z_index": [1, 2, 3, 9, 10, 11, 20, 100],
    }


def test_napari_component_value_tracker_skips_missing_axes_and_sorts_mixed_domains():
    tracker = NapariComponentValueTracker()

    tracker.update(
        "nuclei",
        ["well", "channel"],
        [
            _layer_item({"well": "A01", "channel": 1}),
            _layer_item({"channel": "D"}),
            _layer_item({}),
        ],
    )

    assert tracker.values_for("nuclei", ["well", "channel"]) == {
        "well": ["A01"],
        "channel": [1, "D"],
    }


def test_component_dimension_label_policy_owns_channel_well_and_generic_labels():
    policy = ComponentDimensionLabelPolicy()

    assert policy.labels_for(
        component="channel",
        values=[1, 2],
        metadata={"1": "DAPI", "2": "None"},
    ) == ["Ch1: DAPI", "Ch 2"]
    assert policy.labels_for(
        component="well",
        values=["A01"],
        metadata={"A01": "A01"},
    ) == ["A01"]
    assert policy.labels_for(
        component="site",
        values=[3],
        metadata={"3": "Field"},
    ) == ["Site 3: Field"]


def test_napari_shape_label_rasterizer_projects_polygon_and_path_by_component():
    rasterizer = NapariShapeLabelRasterizer()

    labels = rasterizer.rasterize(
        layer_items=[
            _layer_item(
                {"channel": 1},
                [
                    {
                        "type": "polygon",
                        "coordinates": [[0, 0], [0, 2], [2, 2], [2, 0]],
                    }
                ],
            ),
            _layer_item(
                {"channel": 2},
                [
                    {
                        "type": "path",
                        "coordinates": [[0, 1], [1, 1], [2, 1]],
                    }
                ],
            ),
        ],
        stack_components=["channel"],
        component_values={"channel": [1, 2]},
    )

    assert labels.shape == (2, 3, 3)
    assert np.count_nonzero(labels[0] == 1) > 0
    assert labels[1, 0, 1] == 2
    assert labels[1, 1, 1] == 2
    assert labels[1, 2, 1] == 2


def test_napari_shape_label_rasterizer_uses_source_canvas_shape_metadata():
    rasterizer = NapariShapeLabelRasterizer()

    labels = rasterizer.rasterize(
        layer_items=[
            _layer_item(
                {"channel": 1},
                [
                    {
                        "type": "polygon",
                        "coordinates": [[10, 20], [10, 22], [12, 22], [12, 20]],
                        "metadata": {"source_spatial_shape_yx": (100, 200)},
                    }
                ],
            )
        ],
        stack_components=["channel"],
        component_values={"channel": [1]},
    )

    assert labels.shape == (1, 100, 200)
    assert np.count_nonzero(labels[0] == 1) > 0


def test_napari_shape_label_rasterizer_keeps_singleton_site_axis():
    rasterizer = NapariShapeLabelRasterizer()

    labels = rasterizer.rasterize(
        layer_items=[
            _layer_item(
                {"site": 1, "channel": 1},
                [
                    {
                        "type": "polygon",
                        "coordinates": [[0, 0], [0, 2], [2, 2], [2, 0]],
                    }
                ],
            ),
            _layer_item(
                {"site": 1, "channel": 2},
                [
                    {
                        "type": "path",
                        "coordinates": [[0, 1], [1, 1], [2, 1]],
                    }
                ],
            ),
        ],
        stack_components=["site", "channel"],
        component_values={"site": [1, 2], "channel": [1, 2, 3, 4, 5]},
    )

    assert labels.shape == (2, 5, 3, 3)
    assert np.count_nonzero(labels[0, 0] == 1) > 0
    assert labels[0, 1, 0, 1] == 2
    assert np.count_nonzero(labels[1]) == 0
    assert np.count_nonzero(labels[:, 2:]) == 0


def test_napari_shape_label_rasterizer_tolerates_missing_stack_components():
    rasterizer = NapariShapeLabelRasterizer()

    labels = rasterizer.rasterize(
        layer_items=[
            _layer_item(
                {"source": "IdentifyPrimaryObjects"},
                [
                    {
                        "type": "polygon",
                        "coordinates": [[0, 0], [0, 2], [2, 2], [2, 0]],
                    }
                ],
            )
        ],
        stack_components=["well", "channel"],
        component_values={"well": ["A01"], "channel": [1, 2]},
    )

    assert labels.shape == (1, 2, 3, 3)
    assert np.count_nonzero(labels[0, 0] == 1) > 0


def test_napari_shape_label_rasterizer_keeps_points_as_extent_only():
    rasterizer = NapariShapeLabelRasterizer()

    labels = rasterizer.rasterize(
        layer_items=[
            _layer_item(
                {"channel": 1},
                [
                    {
                        "type": "points",
                        "coordinates": [[4, 5]],
                    }
                ],
            )
        ],
        stack_components=["channel"],
        component_values={"channel": [1]},
    )

    assert labels.shape == (1, 5, 6)
    assert np.count_nonzero(labels) == 0
