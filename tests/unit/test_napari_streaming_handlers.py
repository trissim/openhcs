from __future__ import annotations

import numpy as np
import pytest

from polystore.streaming_constants import StreamingDataType
from polystore.streaming.identity import StreamProducerIdentity

from openhcs.runtime.viewer_protocol import (
    ViewerComponentValueOrdering,
)
from openhcs.runtime.napari_streaming_handlers import (
    NapariBatchProcessorStore,
    NapariAxisPresentation,
    NapariComponentGroupStore,
    NapariDimensionLayerState,
    NapariLayerBatchDebouncePolicy,
    NapariLayerUpdateAuthority,
    NapariLayerRouteStateStore,
    NapariShapeLabelRasterizer,
    NapariStreamLayerAddress,
    NapariStreamLayerItem,
    NapariStreamingDataTypeHandler,
    build_napari_streaming_data_type_handlers,
    napari_streaming_data_type_handler,
)
from openhcs.runtime.viewer_component_system import (
    ViewerComponentAxisSemanticsAuthority,
    ViewerComponentLayout,
    ViewerComponentLabelAuthority,
    ViewerComponentMetadataNormalizer,
    ViewerComponentNameMetadata,
    ViewerComponentNameMetadataWirePayload,
    ViewerComponentValueDomainPayload,
    ViewerComponentCoordinateAuthority,
    ViewerDisplayAxisDomain,
    ViewerLayerAxisProjection,
    ViewerLayerAxisProjectionRequest,
    ViewerLayerAxisProjector,
    ViewerMappingDisplayConfigInput,
    ViewerRouteComponentValueTracker,
)


def _component_name_metadata(payload, context="test component metadata"):
    return ViewerComponentNameMetadata.from_wire_payload(
        ViewerComponentNameMetadataWirePayload.from_mapping(
            payload,
            context=context,
        ),
        context=context,
    )


def _component_value_domain(payload, context="test component value domain"):
    return ViewerComponentValueDomainPayload.from_wire_mapping(payload, context=context)


def _handler_marker(name: str):
    def marker(*args, **kwargs):
        return name, args, kwargs

    return marker


def _layer_item(
    components: dict,
    data=None,
    stream_layer_data_type: StreamingDataType = StreamingDataType.IMAGE,
) -> NapariStreamLayerItem:
    return NapariStreamLayerItem(
        data=data,
        address=NapariStreamLayerAddress(
            components=components,
            path="test",
            stream_layer_data_type=stream_layer_data_type,
        ),
    )


def _axis_presentation(
    *,
    layer_key: str,
    projected_axis_components: tuple[str, ...],
    component_values: dict | None = None,
    payload_axis_labels: tuple[str, ...] = (),
    axis_offsets: tuple[int, ...] | None = None,
) -> NapariAxisPresentation:
    if component_values is None:
        component_values = {component: [1] for component in projected_axis_components}
    if axis_offsets is None:
        axis_offsets = tuple(0 for _ in projected_axis_components)
    return NapariAxisPresentation(
        layer_key=layer_key,
        axis_projection=ViewerLayerAxisProjection(
            projected_axis_components=projected_axis_components,
            component_values=component_values,
            axis_offsets=axis_offsets,
        ),
        payload_axis_labels=payload_axis_labels,
    )


def _axis_projection(
    projected_axis_components,
    component_values,
) -> ViewerLayerAxisProjection:
    return ViewerLayerAxisProjection(
        projected_axis_components=tuple(projected_axis_components),
        component_values=component_values,
        axis_offsets=tuple(0 for _ in projected_axis_components),
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
        handled_stream_data_type=StreamingDataType.IMAGE,
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
        napari_streaming_data_type_handler(handlers, "image").handled_stream_data_type
        is StreamingDataType.IMAGE
    )


def test_napari_streaming_data_type_handler_fails_loudly_for_missing_axis() -> None:
    with pytest.raises(ValueError, match="No Napari streaming handler registered"):
        napari_streaming_data_type_handler({}, StreamingDataType.IMAGE)


class _FakeLayerList(list):
    def __init__(self):
        super().__init__()
        self.selection = type("Selection", (), {"active": None})()

    def remove(self, layer):
        super().remove(layer)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.dims = type("Dims", (), {"axis_labels": None, "ndim": 2})()
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
        self.dimension_states = {}

    def set_dimension_state(self, layer_key, state):
        self.dimension_states[layer_key] = state

    def dimension_state_for(self, layer_key):
        return self.dimension_states.get(layer_key, NapariDimensionLayerState.empty())


class _FakeNapariServer:
    def __init__(self):
        self.layer_route_state = _FakeLayerState()
        self.component_name_metadata = _component_name_metadata({"channel": {"1": "DAPI"}})
        self.component_values = ViewerRouteComponentValueTracker()
        self.display_axis_domain = ViewerDisplayAxisDomain()


def test_napari_layer_update_authority_replaces_existing_image_without_global_axis_labels():
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
    assert viewer.dims.axis_labels is None
    assert viewer.calls[0] == (
        "image",
        "image-1",
        "Nuclei",
        {"blending": "additive", "colormap": "gray", "axis_labels": ("z", "y", "x")},
    )
    assert viewer.calls[1] == (
        "image",
        "image-2",
        "Nuclei",
        {
            "blending": "additive",
            "colormap": "green",
            "axis_labels": ("c", "y", "x"),
        },
    )


def test_napari_layer_update_authority_preserves_user_selected_layer():
    viewer = _FakeViewer()
    selected_layer = type("Layer", (), {"name": "Selected"})()
    viewer.layers.append(selected_layer)
    viewer.layers.selection.active = selected_layer
    layers = {}
    authority = NapariLayerUpdateAuthority()

    new_layer = authority.create_or_update_image(
        viewer,
        layers,
        "route-overlay",
        "Overlay",
        "image",
        colormap=None,
    )

    assert new_layer in viewer.layers
    assert viewer.layers.selection.active is selected_layer


def test_napari_layer_update_authority_transfers_active_replaced_layer():
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
    )
    viewer.layers.selection.active = first
    second = authority.create_or_update_image(
        viewer,
        layers,
        "route-nuclei",
        "Nuclei",
        "image-2",
        colormap=None,
    )

    assert first not in viewer.layers
    assert viewer.layers.selection.active is second


def test_napari_layer_update_authority_marks_color_images_as_rgb():
    viewer = _FakeViewer()
    layers = {}
    authority = NapariLayerUpdateAuthority()

    layer = authority.create_or_update_image(
        viewer,
        layers,
        "route-overlay",
        "Overlay",
        np.zeros((2, 16, 16, 3), dtype=np.uint8),
        colormap="green",
    )

    assert layer in viewer.layers
    assert viewer.calls[0][3] == {"blending": "additive", "rgb": True}


def test_napari_display_pipeline_applies_dimension_labels_to_layer_route_state():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())

    axis_labels = pipeline.dimension_labels.store.apply(
        _axis_presentation(
            layer_key="route-labels",
            projected_axis_components=("channel",),
            component_values={"channel": [1, 2]},
        )
    )

    assert axis_labels == ("channel", "y", "x")
    state = pipeline.server.layer_route_state.dimension_state_for("route-labels")
    assert state.labels == {"channel": ["Ch1: DAPI", "Ch 2"]}
    assert state.stack_axes == ("channel",)
    assert state.axis_labels == ("channel", "y", "x")


def test_napari_display_pipeline_labels_payload_local_image_planes():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    policy = napari_viewer_server.NapariImagePayloadAxisLabelPolicy()
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())

    axis_labels = pipeline.dimension_labels.store.apply(
        _axis_presentation(
            layer_key="route-stack",
            projected_axis_components=("well",),
            component_values={"well": ["A14", "B13"]},
            payload_axis_labels=policy.axis_labels(np.zeros((8, 16, 16))),
        )
    )

    assert axis_labels == ("well", "plane", "y", "x")
    assert policy.axis_labels(np.zeros((4, 16, 16, 3))) == ("plane",)
    assert policy.axis_labels(np.zeros((16, 16, 3))) == ()


def test_napari_display_pipeline_rejects_nonsemantic_payload_axis_labels() -> None:
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())

    with pytest.raises(TypeError, match="semantic axis-name strings"):
        pipeline.dimension_labels.store.apply(
            _axis_presentation(
                layer_key="route-labels",
                projected_axis_components=("channel",),
                component_values={"channel": [1]},
                payload_axis_labels=(0,),  # type: ignore[arg-type]
            )
        )


def test_napari_display_pipeline_projects_route_axes_into_viewer_domain():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())
    component_value_domain = _component_value_domain(
        {"channel": [1, 2, 3, 4, 5], "site": [1, 2]}
    )
    component_axis_semantics = ViewerComponentAxisSemanticsAuthority.from_display_config(
        ViewerMappingDisplayConfigInput(
            {
                "component_modes": {"channel": "stack", "site": "stack"},
                "component_order": ["channel", "site"],
            }
        ),
        component_value_domain,
    )

    first = pipeline.display_axis_projection(
        "source",
        component_axis_semantics,
        [
            _layer_item({"channel": 1, "site": 1}),
            _layer_item({"channel": 1, "site": 2}),
            _layer_item({"channel": 2, "site": 1}),
            _layer_item({"channel": 2, "site": 2}),
            _layer_item({"channel": 3, "site": 1}),
            _layer_item({"channel": 3, "site": 2}),
            _layer_item({"channel": 4, "site": 1}),
            _layer_item({"channel": 4, "site": 2}),
            _layer_item({"channel": 5, "site": 1}),
            _layer_item({"channel": 5, "site": 2}),
        ],
    )
    second = pipeline.display_axis_projection(
        "derived",
        component_axis_semantics,
        [
            _layer_item({"channel": 4, "site": 1}),
            _layer_item({"channel": 4, "site": 2}),
        ],
    )

    assert first.projected_axis_components == ("channel", "site")
    assert first.component_values == {"channel": [1, 2, 3, 4, 5], "site": [1, 2]}
    assert first.axis_offsets == (0, 0)
    assert second.projected_axis_components == ("channel", "site")
    assert second.component_values == {"channel": [4], "site": [1, 2]}
    assert second.axis_offsets == (3, 0)
    assert second.translate() == (3.0, 0.0, 0.0, 0.0)


def test_napari_axis_projector_uses_declared_domain_and_drops_singletons():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    projection = ViewerLayerAxisProjector().project(
        ViewerLayerAxisProjectionRequest.from_component_values(
            projected_axis_components=("channel", "z_index"),
            route_component_values={"channel": [4], "z_index": [1]},
            viewer_component_values={"channel": [4], "z_index": [1]},
            declared_component_values={
                "channel": [1, 2, 3, 4, 5],
                "z_index": [1],
            },
        )
    )

    assert projection.projected_axis_components == ("channel",)
    assert projection.component_values == {"channel": [4]}
    assert projection.axis_offsets == (3,)


def test_napari_axis_projector_keeps_singleton_route_in_shared_viewer_domain():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")

    projection = ViewerLayerAxisProjector().project(
        ViewerLayerAxisProjectionRequest.from_component_values(
            projected_axis_components=("site", "channel"),
            route_component_values={"site": [1, 2], "channel": [4]},
            viewer_component_values={"site": [1, 2], "channel": [1, 2, 3, 4, 5]},
            declared_component_values={"site": [1, 2], "channel": [4]},
        )
    )

    assert projection.projected_axis_components == ("site", "channel")
    assert projection.component_values == {"site": [1, 2], "channel": [4]}
    assert projection.axis_offsets == (0, 3)


def test_napari_axis_projector_uses_declared_domain_for_noncontiguous_routes():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")

    projection = ViewerLayerAxisProjector().project(
        ViewerLayerAxisProjectionRequest.from_component_values(
            projected_axis_components=("channel",),
            route_component_values={"channel": [1, 3]},
            viewer_component_values={"channel": [1, 3]},
            declared_component_values={"channel": [1, 2, 3, 4, 5]},
        )
    )

    assert projection.projected_axis_components == ("channel",)
    assert projection.component_values == {"channel": [1, 2, 3, 4, 5]}
    assert projection.axis_offsets == (0,)


def test_napari_axis_projector_requires_declared_domain():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")

    with pytest.raises(ValueError, match="declared component domain missing 'channel'"):
        ViewerLayerAxisProjector().project(
            ViewerLayerAxisProjectionRequest.from_component_values(
                projected_axis_components=("channel",),
                route_component_values={"channel": [4]},
                viewer_component_values={"channel": [4]},
                declared_component_values={},
            )
        )


def test_napari_component_name_metadata_merges_without_erasing_values():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    metadata = _component_name_metadata(
        {
            "channel": {"1": "OrigDNA", "4": "OrigActin_Golgi_Membrane"},
            "well": {"A14": None},
        }
    )

    napari_viewer_server.NapariComponentNameMetadataMerge.merge_into(
        metadata,
        _component_name_metadata(
            {"channel": {"1": "OrigDNA"}, "well": {"B13": None}},
        ),
    )

    assert metadata.to_wire_mapping() == {
        "channel": {"1": "OrigDNA", "4": "OrigActin_Golgi_Membrane"},
        "well": {"A14": None, "B13": None},
    }


def test_napari_display_pipeline_applies_active_route_axis_labels_to_viewer():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    class FakeSelection:
        active = None

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = None
        current_step = (0,)
        ndim = 3

    class FakeTextOverlay:
        text = ""

    class FakeViewer:
        layers = FakeLayers()
        dims = FakeDims()
        text_overlay = FakeTextOverlay()

    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.viewer = FakeViewer()
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    server.layer_route_state.set_dimension_state(
        "nuclei",
        NapariDimensionLayerState(
            labels={"channel": ["Ch1: DAPI"]},
            presentation=_axis_presentation(
                layer_key="nuclei",
                projected_axis_components=("channel",),
                component_values={"channel": [1]},
            ),
        ),
    )
    server.layer_route_state.set_active_dimension_label_route("nuclei")

    pipeline.dimension_labels.setup_for_layer("nuclei")

    assert server.viewer.dims.axis_labels == ("channel", "y", "x")
    assert server.viewer.text_overlay.text == "Ch1: DAPI"


def test_napari_display_pipeline_offsets_viewer_steps_for_route_local_labels():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    class FakeSelection:
        active = None

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = None
        current_step = (3, 1, 0, 0)
        ndim = 4

    class FakeTextOverlay:
        text = ""

    class FakeViewer:
        layers = FakeLayers()
        dims = FakeDims()
        text_overlay = FakeTextOverlay()

    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.viewer = FakeViewer()
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    server.layer_route_state.set_dimension_state(
        "spots",
        NapariDimensionLayerState(
            labels={"channel": ["Ch4"], "site": ["Site 1", "Site 2"]},
            presentation=_axis_presentation(
                layer_key="spots",
                projected_axis_components=("channel", "site"),
                component_values={"channel": [4], "site": [1, 2]},
                axis_offsets=(3, 0),
            ),
        ),
    )
    server.layer_route_state.set_active_dimension_label_route("spots")

    pipeline.dimension_labels.setup_for_layer("spots")

    assert server.viewer.dims.axis_labels == ("channel", "site", "y", "x")
    assert server.viewer.text_overlay.text == "Ch4 | Site 2"


def test_napari_display_pipeline_rejects_axis_labels_for_wrong_viewer_ndim():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    class FakeSelection:
        active = None

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = ("well", "site", "channel", "y", "x")
        current_step = (0, 0, 0)
        ndim = 5

    class FakeTextOverlay:
        text = ""

    class FakeViewer:
        layers = FakeLayers()
        dims = FakeDims()
        text_overlay = FakeTextOverlay()

    layer = type("Layer", (), {"data": np.zeros((2, 2, 8, 16))})()
    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.viewer = FakeViewer()
    server.viewer.layers.selection.active = layer
    server.layer_route_state.set_layer("labels", layer)
    server.layer_route_state.set_dimension_state(
        "labels",
        NapariDimensionLayerState(
            labels={"well": ["A14", "B13"], "site": ["Site 1", "Site 2"]},
            presentation=_axis_presentation(
                layer_key="labels",
                projected_axis_components=("well", "site"),
                component_values={"well": ["A14", "B13"], "site": [1, 2]},
            ),
        ),
    )
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    pipeline.dimension_labels.setup_for_layer("labels")

    assert server.viewer.dims.axis_labels == (
        "well",
        "site",
        "channel",
        "y",
        "x",
    )


def test_napari_display_pipeline_registers_recreated_labels_before_selection_restore():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    route_key = "route-labels"
    old_layer = type("Layer", (), {"name": "old-labels"})()
    registration_observed = []

    class FakeEvent:
        def connect(self, _handler):
            pass

    class FakeSelection:
        def __init__(self, layer_route_state):
            self._active = None
            self.layer_route_state = layer_route_state
            self.events = type("SelectionEvents", (), {"active": FakeEvent()})()

        @property
        def active(self):
            return self._active

        @active.setter
        def active(self, layer):
            self._active = layer
            if layer is not old_layer and layer is not None:
                registration_observed.append(
                    self.layer_route_state.has_layer(route_key)
                    and self.layer_route_state.layer(route_key) is layer
                )

    class FakeLayers(list):
        def __init__(self, layer_route_state):
            super().__init__([old_layer])
            self.selection = FakeSelection(layer_route_state)

        def remove(self, layer):
            super().remove(layer)

    class FakeDims:
        axis_labels = None
        current_step = (0, 0)
        ndim = 2
        events = type("DimsEvents", (), {"current_step": FakeEvent()})()

    class FakeTextOverlay:
        text = ""

    class FakeViewer:
        def __init__(self, layer_route_state):
            self.layers = FakeLayers(layer_route_state)
            self.layers.selection.active = old_layer
            self.dims = FakeDims()
            self.text_overlay = FakeTextOverlay()

        def add_labels(self, data, *, name, **kwargs):
            layer = type("Layer", (), {"name": name, "data": data, "kwargs": kwargs})()
            self.layers.append(layer)
            return layer

    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.layer_route_state.set_title(route_key, "Labels")
    server.layer_route_state.set_layer(route_key, old_layer)
    server.viewer = FakeViewer(server.layer_route_state)
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    shapes = [
        {
            "type": "polygon",
            "coordinates": [[0, 0], [0, 1], [1, 0]],
            "metadata": {"source_spatial_shape_yx": (2, 2)},
        }
    ]

    pipeline.update_shapes_layer(
        napari_viewer_server.NapariLayerTypedUpdateRequest(
            layer_key=route_key,
            layer_items=[
                _layer_item({}, data=shapes, stream_layer_data_type=StreamingDataType.SHAPES)
            ],
            axis_projection=ViewerLayerAxisProjection(
                projected_axis_components=(),
                component_values={},
                axis_offsets=(),
            ),
        )
    )

    assert registration_observed == [True]


def test_napari_viewer_clear_state_resets_accumulated_axis_domains():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    server = napari_viewer_server.NapariViewerServer.__new__(
        napari_viewer_server.NapariViewerServer
    )
    server.component_groups = NapariComponentGroupStore()
    server.component_groups.items_for("old").append(_layer_item({"well": "A14"}))
    server.component_values = ViewerRouteComponentValueTracker()
    server.component_values.update("old", ["well"], [_layer_item({"well": "A14"})])
    server.display_axis_domain = ViewerDisplayAxisDomain()
    server.display_axis_domain.update(["well"], [_layer_item({"well": "A14"})])
    server.component_name_metadata = _component_name_metadata({"well": {"A14": "A14"}})
    server.layer_batch_processor_debounce_policy = NapariLayerBatchDebouncePolicy(
        delay_ms=123
    )
    server.batch_processors = NapariBatchProcessorStore(
        debounce_policy=NapariLayerBatchDebouncePolicy(delay_ms=1)
    )

    server.clear_accumulated_stream_state()

    assert len(server.component_groups) == 0
    assert server.component_values.domain.values_for(("old", ("well",)), ["well"]) == {
        "well": []
    }
    assert server.display_axis_domain.values_for(["well"]) == {"well": []}
    assert server.component_name_metadata.to_wire_mapping() == {}
    assert server.batch_processors.debounce_policy.delay_ms == 123


def test_napari_axis_projector_drops_only_globally_singleton_axes():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    component_values = {
        "site": [1, 2],
        "timepoint": [1],
        "channel": [1, 2, 3, 4, 5],
        "z_index": [1],
        "well": ["A01"],
    }

    projection = ViewerLayerAxisProjector().project(
        ViewerLayerAxisProjectionRequest.from_component_values(
            projected_axis_components=("site", "timepoint", "channel", "z_index", "well"),
            route_component_values=component_values,
            viewer_component_values=component_values,
            declared_component_values=component_values,
        )
    )

    assert projection.projected_axis_components == ("site", "channel")
    assert projection.component_values == {"site": [1, 2], "channel": [1, 2, 3, 4, 5]}
    assert projection.axis_offsets == (0, 0)


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
    component_layout = ViewerComponentLayout(
        component_modes={"well": "slice", "channel": "stack"},
        component_order=["well", "channel"],
    )

    title = napari_viewer_server.NapariLayerTitleAuthority.title(
        producer=producer,
        stream_layer_data_type=StreamingDataType.SHAPES,
        component_info={"well": "A01", "channel": 1},
        component_layout=component_layout,
    )

    assert title == "9. ConvertObjectsToImage NucleiObjects3D well A01 labels"


def test_napari_component_display_coordinator_splits_image_shape_roles():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    class _FakeDisplayPipeline:
        def __init__(self):
            self.scheduled = []

        def schedule_layer_update(
            self,
            layer_key,
            data_type,
            component_axis_semantics,
        ):
            self.scheduled.append(
                (layer_key, data_type, component_axis_semantics)
            )

    class _FakeServer:
        def __init__(self):
            self.viewer = _FakeViewer()
            self.layer_route_state = NapariLayerRouteStateStore.empty()
            self.display_pipeline = _FakeDisplayPipeline()
            self.component_groups = NapariComponentGroupStore()
            self.replace_layers = False

    server = _FakeServer()
    coordinator = napari_viewer_server.NapariComponentAwareDisplayCoordinator()
    producer = StreamProducerIdentity.pipeline_output(
        output_kind="main",
        output_key="main",
        step_name="OverlayOutlines",
        pipeline_position=4,
    )
    display_config = {
        "component_modes": {"well": "stack"},
        "component_order": ["well"],
    }

    for data in (np.zeros((16, 16)), np.zeros((4, 16, 16, 3))):
        coordinator.display(
            napari_viewer_server.NapariComponentAwareDisplayRequest.from_stream_layer_context(
                data=data,
                stream_layer_context=napari_viewer_server.NapariStreamLayerContext(
                    component_axis_semantics=ViewerComponentAxisSemanticsAuthority.from_display_config(
                        ViewerMappingDisplayConfigInput(display_config),
                        _component_value_domain({"well": ["A01"]}),
                    ),
                    producer=producer,
                    address=NapariStreamLayerAddress(
                        components={"well": "A01"},
                        path="/tmp/A01.tif",
                        stream_layer_data_type=StreamingDataType.IMAGE,
                    ),
                ),
                server=server,
            )
        )

    assert len(server.component_groups) == 2
    assert len({layer_key for layer_key, *_ in server.display_pipeline.scheduled}) == 2
    assert any("color_stack" in layer_key for layer_key in server.component_groups)
    assert sorted(server.layer_route_state.layer_titles.values()) == [
        "5. OverlayOutlines",
        "5. OverlayOutlines RGB stack",
    ]


def test_napari_layer_title_disambiguation_uses_display_step_number():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    producer = StreamProducerIdentity.pipeline_output(
        output_kind="main",
        output_key="main",
        step_name="Measure",
        pipeline_position=3,
    )
    layer_route_state = NapariLayerRouteStateStore.empty()
    layer_route_state.set_title("other-route", "4. Measure")

    assert (
        napari_viewer_server.NapariLayerTitleAuthority.disambiguate(
            title="4. Measure",
            producer=producer,
            route_key="current-route",
            layer_route_state=layer_route_state,
        )
        == "4. Measure [step 4]"
    )


class _FakeTimer:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_napari_layer_route_state_store_keeps_layer_labels_and_timers_together():
    store = NapariLayerRouteStateStore.empty()
    timer = _FakeTimer()
    layer = object()

    store.set_layer("nuclei", layer)
    store.set_dimension_state(
        "nuclei",
        NapariDimensionLayerState(
            labels={"channel": ["Ch 1"]},
            presentation=_axis_presentation(
                layer_key="nuclei",
                projected_axis_components=("channel",),
                component_values={"channel": [1]},
            ),
        ),
    )
    store.set_pending_update("nuclei", timer)

    assert store.has_layer("nuclei")
    assert store.layer("nuclei") is layer
    assert store.dimension_state_for("nuclei").labels == {"channel": ["Ch 1"]}
    assert store.cancel_pending_update("nuclei")
    assert timer.stopped
    assert store.pop_pending_update("nuclei") is timer
    assert store.dimension_state_for("missing").labels == {}


def test_napari_batch_processor_store_creates_one_processor_per_layer(monkeypatch):
    import polystore.streaming.receivers.napari as napari_receivers

    created = []

    class FakeBatchProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(napari_receivers, "NapariBatchProcessor", FakeBatchProcessor)

    store = NapariBatchProcessorStore(
        debounce_policy=NapariLayerBatchDebouncePolicy(
            delay_ms=123,
            max_wait_ms=456,
        )
    )
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
    normalizer = ViewerComponentMetadataNormalizer()

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


def test_viewer_component_coordinate_authority_rejects_missing_axis_component():
    with pytest.raises(ValueError, match="missing stack component 'channel'"):
        ViewerComponentCoordinateAuthority.index(
            components={"site": 1},
            component_values={"site": [1], "channel": [1]},
            component="channel",
            context="test item",
        )


def test_napari_shape_rasterizer_rejects_out_of_domain_stack_component():
    rasterizer = NapariShapeLabelRasterizer()
    shapes = [
        {
            "type": "polygon",
            "coordinates": [[0, 0], [0, 1], [1, 0]],
            "metadata": {"source_spatial_shape_yx": (2, 2)},
        }
    ]

    with pytest.raises(ValueError, match="outside axis domain"):
        rasterizer.rasterize(
            layer_items=[
                _layer_item(
                    {"site": 2},
                    data=shapes,
                    stream_layer_data_type=StreamingDataType.SHAPES,
                )
            ],
            axis_projection=_axis_projection(("site",), {"site": [1]}),
        )


def test_napari_component_value_tracker_tracks_observed_axis_values_by_route():
    tracker = ViewerRouteComponentValueTracker()

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

    assert tracker.domain.values_for(("main", ("channel", "well")), ["channel", "well"]) == {
        "channel": [2, 4],
        "well": ["A01", "B02"],
    }
    assert tracker.domain.values_for(
        ("artifact", ("channel", "well")),
        ["channel", "well"],
    ) == {
        "channel": [1],
        "well": ["A01"],
    }
    assert tracker.domain.values_for(("main", ("site",)), ["site"]) == {"site": []}


def test_napari_display_axis_domain_tracks_shared_active_axis_values():
    domain = ViewerDisplayAxisDomain()
    projected_axis_components = ["site", "timepoint", "channel", "z_index", "well"]

    domain.update(
        projected_axis_components,
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
        projected_axis_components,
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

    assert domain.values_for(projected_axis_components) == {
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
    tracker = ViewerRouteComponentValueTracker()

    tracker.update(
        "main",
        ["z_index"],
        [
            _layer_item({"z_index": value})
            for value in [1, 10, 100, 11, 2, 20, 3, 9]
        ],
    )

    assert tracker.domain.values_for(("main", ("z_index",)), ["z_index"]) == {
        "z_index": [1, 2, 3, 9, 10, 11, 20, 100],
    }


def test_napari_component_value_tracker_skips_missing_axes_and_sorts_mixed_domains():
    tracker = ViewerRouteComponentValueTracker()

    tracker.update(
        "nuclei",
        ["well", "channel"],
        [
            _layer_item({"well": "A01", "channel": 1}),
            _layer_item({"channel": "D"}),
            _layer_item({}),
        ],
    )

    assert tracker.domain.values_for(
        ("nuclei", ("well", "channel")),
        ["well", "channel"],
    ) == {
        "well": ["A01"],
        "channel": [1, "D"],
    }


def test_viewer_component_label_authority_owns_channel_well_and_generic_labels():
    authority = ViewerComponentLabelAuthority(
        _component_name_metadata(
            {
                "channel": {"1": "DAPI", "2": "None"},
                "well": {"A01": "A01"},
                "site": {"3": "Field"},
            },
            context="test",
        )
    )

    assert authority.axis_labels("channel", [1, 2]) == ["Ch1: DAPI", "Ch 2"]
    assert authority.axis_labels("well", ["A01"]) == ["A01"]
    assert authority.axis_labels("site", [3]) == ["Site 3: Field"]


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
        axis_projection=_axis_projection(["channel"], {"channel": [1, 2]}),
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
        axis_projection=_axis_projection(["channel"], {"channel": [1]}),
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
        axis_projection=_axis_projection(
            ["site", "channel"],
            {"site": [1, 2], "channel": [1, 2, 3, 4, 5]},
        ),
    )

    assert labels.shape == (2, 5, 3, 3)
    assert np.count_nonzero(labels[0, 0] == 1) > 0
    assert labels[0, 1, 0, 1] == 2
    assert np.count_nonzero(labels[1]) == 0
    assert np.count_nonzero(labels[:, 2:]) == 0


def test_napari_shape_label_rasterizer_rejects_missing_projected_axis_components():
    rasterizer = NapariShapeLabelRasterizer()

    with pytest.raises(ValueError, match="missing stack component 'well'"):
        rasterizer.rasterize(
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
            axis_projection=_axis_projection(
                ["well", "channel"],
                {"well": ["A01"], "channel": [1, 2]},
            ),
        )


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
        axis_projection=_axis_projection(["channel"], {"channel": [1]}),
    )

    assert labels.shape == (1, 5, 6)
    assert np.count_nonzero(labels) == 0
