from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polystore.streaming_constants import StreamingDataType
from polystore.streaming.identity import StreamProducerIdentity

from openhcs.core.artifacts import ObjectArtifactSubjectBinding
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.runtime.viewer_protocol import (
    NapariLayerKind,
    ViewerControlMessageType,
    ViewerControlResponseField,
    ViewerNavigationControlOptions,
    ViewerPayloadControlOptions,
    ViewerProtocolStatus,
    ViewerSettlePhase,
    ViewerSettleProgress,
    ViewerControlResponse,
    ViewerStateControlOptions,
    ViewerComponentValueOrdering,
)
from openhcs.runtime.napari_streaming_handlers import (
    NapariAggregateAxisBinding,
    NapariAggregateAxisBindingAuthority,
    NapariAggregateAxisBindingSet,
    NapariBatchProcessorStore,
    NapariAxisPresentation,
    NapariComponentGroupStore,
    NapariDimensionLayerState,
    NapariImageLayerPresentationPolicy,
    NapariLayerBatchDebouncePolicy,
    NapariPendingLayerUpdate,
    NapariLayerUpdateAuthority,
    NapariLayerRouteStateStore,
    NapariShapeLayerPayload,
    NapariStreamLayerAddress,
    NapariStreamLayerItem,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from openhcs.runtime.viewer_component_system import (
    ViewerComponentAxisSemanticsAuthority,
    ViewerComponentLayout,
    ViewerComponentMetadataNormalizer,
    ViewerComponentNameMetadata,
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
    return ViewerComponentNameMetadata.from_wire_mapping(
        payload,
        context=context,
    )


def _component_value_domain(payload, context="test component value domain"):
    return ViewerComponentValueDomainPayload.from_wire_mapping(payload, context=context)


def _layer_item(
    components: dict,
    data=None,
    producer: StreamProducerIdentity | None = None,
    stream_layer_data_type: StreamingDataType = StreamingDataType.IMAGE,
    image_metadata: ImagePayloadMetadata | None = None,
    plane_component_domain: ViewerComponentValueDomainPayload | None = None,
) -> NapariStreamLayerItem:
    return NapariStreamLayerItem(
        data=data,
        producer=producer
        or StreamProducerIdentity.pipeline_output(
            output_kind="main",
            output_key="main",
            projection_key="main",
            step_name="Test Step",
            pipeline_position=7,
            step_scope_id="test-step",
        ),
        address=NapariStreamLayerAddress(
            components=components,
            path="test",
            stream_layer_data_type=stream_layer_data_type,
        ),
        image_metadata=image_metadata or ImagePayloadMetadata(),
        plane_component_domain=(
            plane_component_domain or ViewerComponentValueDomainPayload(())
        ),
    )


def _payload_summary(item: NapariStreamLayerItem) -> dict:
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    return napari_viewer_server.NapariViewerStateProjection.payload_summary(
        item,
        item.address.components,
        item.data,
    )


def test_napari_stream_context_reconstructs_exact_source_spatial_domain():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    producer = StreamProducerIdentity.pipeline_output(
        output_kind="artifact",
        output_key="CropBlue",
        projection_key="CropBlue",
        step_name="Crop",
        pipeline_position=3,
        step_scope_id="crop-step",
        artifact_kind="image",
    )

    context = napari_viewer_server.NapariStreamLayerContext.from_payload_map(
        napari_viewer_server.PayloadMap(
            {
                "producer_identity": producer.to_payload(),
                "metadata": {"well": "A01", "channel": 1},
                "path": "CropBlue.tif",
                "data_type": "image",
                "spatial_origin_yx": (3, 5),
                "source_spatial_shape_yx": (20, 30),
            },
            "test image payload",
        ),
        ViewerComponentAxisSemanticsAuthority.empty(),
    )

    assert context.address.components == {"well": "A01", "channel": 1}
    assert context.image_metadata.source_spatial_domain == SourceSpatialDomain(
        origin_yx=(3, 5),
        source_shape_yx=(20, 30),
        value_name="Napari image payload",
    )


def _axis_presentation(
    *,
    layer_key: str,
    projected_axis_components: tuple[str, ...],
    component_values: dict | None = None,
    payload_axis_labels: tuple[str, ...] = (),
    axis_offsets: tuple[int, ...] | None = None,
    scalar_component_values: dict | None = None,
) -> NapariAxisPresentation:
    if component_values is None:
        component_values = {component: [1] for component in projected_axis_components}
    if axis_offsets is None:
        axis_offsets = tuple(0 for _ in projected_axis_components)
    if scalar_component_values is None:
        scalar_component_values = {}
    component_axis_semantics = ViewerComponentAxisSemanticsAuthority.empty()
    return NapariAxisPresentation(
        entries=component_axis_semantics.entries,
        layout=component_axis_semantics.layout,
        route_key=layer_key,
        projection=ViewerLayerAxisProjection(
            projected_axis_components=projected_axis_components,
            component_values=component_values,
            routed_component_values=component_values,
            axis_offsets=axis_offsets,
            scalar_component_values=scalar_component_values,
        ),
        payload_axis_labels=payload_axis_labels,
    )


def _axis_projection(
    projected_axis_components,
    component_values,
    scalar_component_values=None,
) -> ViewerLayerAxisProjection:
    if scalar_component_values is None:
        scalar_component_values = {}
    return ViewerLayerAxisProjection(
        projected_axis_components=tuple(projected_axis_components),
        component_values=component_values,
        routed_component_values=component_values,
        axis_offsets=tuple(0 for _ in projected_axis_components),
        scalar_component_values=scalar_component_values,
    )


def test_napari_viewer_state_projection_reports_shape_payload_occupancy():
    summary = _payload_summary(
        _layer_item(
            {"well": "A14", "site": 1, "channel": 1},
            data=[{"type": "polygon"}, {"type": "polygon"}],
            stream_layer_data_type=StreamingDataType.SHAPES,
        )
    )

    assert summary["payload_type"] == "list"
    assert summary["item_count"] == 2
    assert summary["nonzero_count"] == 2
    assert summary["shape_payload_count"] == 2
    assert summary["missing_source_spatial_shape_count"] == 2


def test_napari_viewer_state_projection_reports_empty_shape_payloads_as_zero():
    summary = _payload_summary(
        _layer_item(
            {"well": "A14", "site": 1, "channel": 1},
            data=[],
            stream_layer_data_type=StreamingDataType.SHAPES,
        )
    )

    assert summary["payload_type"] == "list"
    assert summary["item_count"] == 0
    assert summary["nonzero_count"] == 0


def test_napari_viewer_state_projection_reports_shape_spatial_evidence():
    summary = _payload_summary(
        _layer_item(
            {"well": "A14", "site": 1, "channel": 1},
            data=[
                {
                    "type": "polygon",
                    "coordinates": [[1, 2], [3, 4], [5, 6]],
                    "metadata": {"source_spatial_shape_yx": (16, 16)},
                },
            ],
            stream_layer_data_type=StreamingDataType.SHAPES,
            image_metadata=ImagePayloadMetadata(
                source_spatial_domain=SourceSpatialDomain(
                    origin_yx=(0, 0),
                    source_shape_yx=(16, 16),
                )
            ),
        )
    )

    assert summary["spatial_origin_yx"] == (0, 0)
    assert summary["source_spatial_shape_yx"] == (16, 16)
    assert summary["shape_coordinate_count"] == 3
    assert summary["shape_out_of_source_bounds_count"] == 0
    assert summary["shape_coordinate_bounds_yx"] == {
        "min_yx": (1.0, 2.0),
        "max_yx": (5.0, 6.0),
        "coordinate_count": 3,
    }


def test_napari_viewer_state_projection_treats_pixel_edge_shape_bounds_as_in_bounds():
    summary = _payload_summary(
        _layer_item(
            {"well": "A14", "site": 1, "channel": 1},
            data=[
                {
                    "type": "polygon",
                    "coordinates": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]],
                    "metadata": {"source_spatial_shape_yx": (16, 16)},
                },
                {
                    "type": "polygon",
                    "coordinates": [[15.5, 15.5], [16.0, 15.5], [15.5, 16.0]],
                    "metadata": {"source_spatial_shape_yx": (16, 16)},
                },
            ],
            stream_layer_data_type=StreamingDataType.SHAPES,
        )
    )

    assert summary["shape_out_of_source_bounds_count"] == 1
    assert summary["shape_coordinate_bounds_yx"] == {
        "min_yx": (-0.5, -0.5),
        "max_yx": (16.0, 16.0),
        "coordinate_count": 6,
    }


def test_component_value_domain_wire_mapping_normalizes_variable_axis_values():
    domain = ViewerComponentValueDomainPayload.from_wire_mapping(
        {"timepoint": ["1", 1], "channel": ["2", 2], "well": ["A01"]},
        context="mixed wire domain",
    )

    assert domain.to_wire_mapping() == {
        "timepoint": [1],
        "channel": [2],
        "well": ["A01"],
    }


def test_napari_viewer_payload_projection_reuses_state_route_key_contract():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    server = type(
        "Server",
        (),
        {
            "napari_window_title": "OpenHCS Napari Viewer",
            "layer_route_state": NapariLayerRouteStateStore.empty(),
            "component_groups": NapariComponentGroupStore(),
        },
    )()
    server.layer_route_state.set_title("mounted-route", "Mounted")
    server.component_groups.items_for("payload-route").append(
        _layer_item({"site": 1}, data=np.ones((2, 2), dtype=np.uint8))
    )

    projection = napari_viewer_server.NapariViewerPayloadProjection(
        server=server,
        viewer=_FakeViewer(),
        request=napari_viewer_server.ViewerPayloadControlOptions(),
    )

    assert projection.route_keys() == ("mounted-route", "payload-route")


def test_napari_viewer_payload_projection_reuses_state_producer_identities():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    server = type(
        "Server",
        (),
        {
            "napari_window_title": "OpenHCS Napari Viewer",
            "layer_route_state": NapariLayerRouteStateStore.empty(),
            "component_groups": NapariComponentGroupStore(),
        },
    )()
    route_key = "payload-route"
    producer = StreamProducerIdentity.pipeline_output(
        output_kind="artifact",
        output_key="CellCountingROI",
        projection_key="CellCountingROI",
        step_name="Export ROI",
        pipeline_position=11,
        step_scope_id="export-roi-step",
        artifact_kind="roi",
    )
    server.layer_route_state.set_title(route_key, "Cell counting ROI")
    server.component_groups.items_for(route_key).append(
        _layer_item(
            {"well": "A01", "site": 1},
            data=np.ones((2, 2), dtype=np.uint8),
            producer=producer,
        )
    )
    projection = napari_viewer_server.NapariViewerPayloadProjection(
        server=server,
        viewer=_FakeViewer(),
        request=ViewerPayloadControlOptions.from_overrides(route_key=route_key),
    )

    state_layer = projection.layer_state_for(route_key)
    payload_layer = projection.layer_payloads_for(route_key)

    assert payload_layer["producer_identities"] == state_layer["producer_identities"]
    assert payload_layer["producer_identities"] == (producer.to_payload(),)


def test_napari_viewer_payload_projection_filters_axis_and_samples_array_crop():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    server = type(
        "Server",
        (),
        {
            "napari_window_title": "OpenHCS Napari Viewer",
            "layer_route_state": NapariLayerRouteStateStore.empty(),
            "component_groups": NapariComponentGroupStore(),
        },
    )()
    route_key = "payload-route"
    server.layer_route_state.set_title(route_key, "Payload")
    server.layer_route_state.set_dimension_state(
        route_key,
        NapariDimensionLayerState(
            labels={},
            presentation=_axis_presentation(
                layer_key=route_key,
                projected_axis_components=("channel",),
                component_values={"channel": [1, 2]},
            ),
        ),
    )
    server.component_groups.items_for(route_key).extend(
        (
            _layer_item(
                {"channel": 1},
                data=np.arange(16, dtype=np.uint16).reshape(4, 4),
            ),
            _layer_item(
                {"channel": 2},
                data=np.arange(16, dtype=np.uint16).reshape(4, 4) + 100,
            ),
        )
    )

    projection = napari_viewer_server.NapariViewerPayloadProjection(
        server=server,
        viewer=_FakeViewer(),
        request=ViewerPayloadControlOptions.from_overrides(
            route_key=route_key,
            axis_indices=(1,),
            include_array_values=True,
            max_array_elements=4,
            array_slices=((1, 3), (2, 4)),
        ),
    )

    payload = projection.to_wire_mapping()

    assert payload["layer_count"] == 1
    layer = payload["layers"][0]
    assert layer["axis_labels"] == ("channel", "y", "x")
    assert layer["stack_axes"] == ("channel",)
    assert len(layer["payloads"]) == 1
    record = layer["payloads"][0]
    assert record["components"] == {"channel": 2}
    assert record["axis_indices"] == (1,)
    assert record["array_values"] == ((106, 107), (110, 111))
    assert record["array_value_summary"] == {
        "requested": True,
        "included": True,
        "slice_ranges": ((1, 3), (2, 4)),
        "requested_slice_ranges": ((1, 3), (2, 4)),
        "dtype": "uint16",
        "shape": (2, 2),
        "size": 4,
        "nonzero_count": 4,
        "min": 106,
        "max": 111,
    }


def test_napari_viewer_payload_projection_filters_semantic_axis_index():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    server = type(
        "Server",
        (),
        {
            "napari_window_title": "OpenHCS Napari Viewer",
            "layer_route_state": NapariLayerRouteStateStore.empty(),
            "component_groups": NapariComponentGroupStore(),
        },
    )()
    route_key = "payload-route"
    server.layer_route_state.set_title(route_key, "Payload")
    server.layer_route_state.set_dimension_state(
        route_key,
        NapariDimensionLayerState(
            labels={},
            presentation=_axis_presentation(
                layer_key=route_key,
                projected_axis_components=("channel",),
                component_values={"channel": [1, 2]},
            ),
        ),
    )
    server.component_groups.items_for(route_key).extend(
        (
            _layer_item({"channel": 1}, data=np.zeros((2, 2), dtype=np.uint16)),
            _layer_item({"channel": 2}, data=np.ones((2, 2), dtype=np.uint16)),
        )
    )

    projection = napari_viewer_server.NapariViewerPayloadProjection(
        server=server,
        viewer=_FakeViewer(),
        request=ViewerPayloadControlOptions.from_overrides(
            route_key=route_key,
            axis_indices={"channel": 1},
        ),
    )

    payload = projection.to_wire_mapping()

    layer = payload["layers"][0]
    assert len(layer["payloads"]) == 1
    record = layer["payloads"][0]
    assert record["components"] == {"channel": 2}
    assert record["axis_indices"] == (1,)


def test_napari_viewer_state_projection_filters_and_bounds_layer_details():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    server = type(
        "Server",
        (),
        {
            "napari_window_title": "OpenHCS Napari Viewer",
            "layer_route_state": NapariLayerRouteStateStore.empty(),
            "component_groups": NapariComponentGroupStore(),
        },
    )()
    server.layer_route_state.set_title("other-route", "Other")
    server.layer_route_state.set_title("payload-route", "Payload")
    first_producer = StreamProducerIdentity.pipeline_output(
        output_kind="main",
        output_key="Stain1",
        projection_key="main",
        step_name="Align",
        pipeline_position=7,
        step_scope_id="align-step",
    )
    second_producer = StreamProducerIdentity.pipeline_output(
        output_kind="main",
        output_key="Stain2",
        projection_key="main",
        step_name="Align",
        pipeline_position=7,
        step_scope_id="align-step",
    )
    server.component_groups.items_for("payload-route").extend(
        (
            _layer_item(
                {"site": 1},
                data=np.ones((2, 2), dtype=np.uint8),
                producer=first_producer,
            ),
            _layer_item(
                {"site": 2},
                data=np.ones((2, 2), dtype=np.uint8),
                producer=second_producer,
            ),
        )
    )

    viewer = _FakeViewer()
    viewer.dims.axis_labels = ("site", "y", "x")
    projection = napari_viewer_server.NapariViewerStateProjection(
        server=server,
        viewer=viewer,
        request=ViewerStateControlOptions.from_overrides(
            route_key="payload-route",
            include_component_values=True,
            max_component_values_per_layer=1,
            include_payload_summaries=False,
        ),
    )

    state = projection.to_wire_mapping()

    assert state["layer_count"] == 1
    layer = state["layers"][0]
    assert layer["route_key"] == "payload-route"
    assert tuple(
        producer["output_key"] for producer in layer["producer_identities"]
    ) == ("Stain1", "Stain2")
    assert layer["component_values"] == ({"site": 1},)
    assert layer["component_value_count"] == 2
    assert layer["component_values_truncated"] is True
    assert layer["payload_summaries"] == ()
    assert layer["payload_summary_count"] == 2
    assert layer["payload_summaries_truncated"] is True


def test_napari_state_control_message_honors_state_request_payload():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    viewer = _FakeViewer()
    viewer.dims.axis_labels = ("site", "y", "x")
    layer = type(
        "Layer",
        (),
        {
            "name": "Payload",
            "data": np.ones((2, 2), dtype=np.uint8),
            "translate": (0.0, 0.0),
            "visible": True,
        },
    )()
    viewer.layers.append(layer)

    server = type(
        "Server",
        (),
        {
            "viewer": viewer,
            "napari_window_title": "OpenHCS Napari Viewer",
            "layer_route_state": NapariLayerRouteStateStore.empty(),
            "component_groups": NapariComponentGroupStore(),
        },
    )()
    server.layer_route_state.set_title("payload-route", "Payload")
    server.layer_route_state.set_layer("payload-route", layer)
    server.component_groups.items_for("payload-route").append(
        _layer_item({"site": 1}, data=np.ones((2, 2), dtype=np.uint8))
    )

    response = napari_viewer_server.NapariStateControlMessageAction().handle(
        server,
        {
            "type": "state",
            ViewerControlResponseField.PAYLOAD.value: ViewerStateControlOptions(
                route_key="payload-route",
                include_component_values=False,
                include_payload_summaries=False,
            ),
        },
    )

    assert response["status"] == "success"
    assert response["layer_count"] == 1
    layer_response = response["layers"][0]
    assert layer_response["route_key"] == "payload-route"
    assert layer_response["component_values"] == ()
    assert layer_response["component_value_count"] == 1
    assert layer_response["component_values_truncated"] is True
    assert layer_response["payload_summaries"] == ()
    assert layer_response["payload_summary_count"] == 1
    assert layer_response["payload_summaries_truncated"] is True


class _FakeLayerList(list):
    def __init__(self):
        super().__init__()
        self.selection = type("Selection", (), {"active": None})()

    def remove(self, layer):
        super().remove(layer)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.dims = type(
            "Dims",
            (),
            {
                "axis_labels": None,
                "current_step": (3, 0, 0, 0),
                "ndim": 4,
            },
        )()
        self.text_overlay = type("TextOverlay", (), {"text": ""})()
        self.calls = []

    def add_image(self, data, *, name, **kwargs):
        return self._add_layer("image", data, name, kwargs)

    def add_shapes(self, data, *, name, **kwargs):
        return self._add_layer("shapes", data, name, kwargs)

    def add_points(self, data, *, name, **kwargs):
        return self._add_layer("points", data, name, kwargs)

    def add_labels(self, data, *, name, **kwargs):
        return self._add_layer("labels", data, name, kwargs)

    def _add_layer(self, layer_type, data, name, kwargs):
        layer_attributes = {"name": name, "data": data, "kwargs": kwargs}
        if layer_type == "shapes":
            layer_attributes.update(
                {
                    "edge_color_mode": "cycle",
                    "face_color_mode": "cycle",
                }
            )
        layer = type("Layer", (), layer_attributes)()
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
        self.component_name_metadata = _component_name_metadata(
            {"channel": {"1": "DAPI"}}
        )
        self.component_values = ViewerRouteComponentValueTracker()
        self.display_axis_domain = ViewerDisplayAxisDomain()
        self.transport_failure = None

    def bind_result_selection_layer(self, _layer):
        """Accept Shapes binding without modeling native Qt selection events."""


def test_napari_layer_update_authority_replaces_existing_image_without_global_axis_labels():
    viewer = _FakeViewer()
    layers = {}
    authority = NapariLayerUpdateAuthority()

    first = authority.create_or_update(
        layer_kind=NapariLayerKind.IMAGE,
        viewer=viewer,
        layers=layers,
        route_key="route-nuclei",
        layer_name="Nuclei",
        data="image-1",
        layer_kwargs={
            **NapariImageLayerPresentationPolicy.layer_kwargs(
                "image-1", ImagePayloadMetadata(), None
            ),
            "axis_labels": ("z", "y", "x"),
        },
    )
    second = authority.create_or_update(
        layer_kind=NapariLayerKind.IMAGE,
        viewer=viewer,
        layers=layers,
        route_key="route-nuclei",
        layer_name="Nuclei",
        data="image-2",
        layer_kwargs={
            **NapariImageLayerPresentationPolicy.layer_kwargs(
                "image-2",
                ImagePayloadMetadata(),
                "green",
            ),
            "axis_labels": ("c", "y", "x"),
        },
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

    new_layer = authority.create_or_update(
        layer_kind=NapariLayerKind.IMAGE,
        viewer=viewer,
        layers=layers,
        route_key="route-overlay",
        layer_name="Overlay",
        data="image",
        layer_kwargs=NapariImageLayerPresentationPolicy.layer_kwargs(
            "image",
            ImagePayloadMetadata(),
            None,
        ),
    )

    assert new_layer in viewer.layers
    assert viewer.layers.selection.active is selected_layer


def test_napari_layer_update_authority_transfers_active_replaced_layer():
    viewer = _FakeViewer()
    layers = {}
    authority = NapariLayerUpdateAuthority()

    first = authority.create_or_update(
        layer_kind=NapariLayerKind.IMAGE,
        viewer=viewer,
        layers=layers,
        route_key="route-nuclei",
        layer_name="Nuclei",
        data="image-1",
        layer_kwargs=NapariImageLayerPresentationPolicy.layer_kwargs(
            "image-1",
            ImagePayloadMetadata(),
            None,
        ),
    )
    viewer.layers.selection.active = first
    second = authority.create_or_update(
        layer_kind=NapariLayerKind.IMAGE,
        viewer=viewer,
        layers=layers,
        route_key="route-nuclei",
        layer_name="Nuclei",
        data="image-2",
        layer_kwargs=NapariImageLayerPresentationPolicy.layer_kwargs(
            "image-2",
            ImagePayloadMetadata(),
            None,
        ),
    )

    assert first not in viewer.layers
    assert viewer.layers.selection.active is second


def test_napari_layer_update_authority_marks_color_images_as_rgb():
    viewer = _FakeViewer()
    layers = {}
    authority = NapariLayerUpdateAuthority()

    image = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    layer = authority.create_or_update(
        layer_kind=NapariLayerKind.IMAGE,
        viewer=viewer,
        layers=layers,
        route_key="route-overlay",
        layer_name="Overlay",
        data=image,
        layer_kwargs=NapariImageLayerPresentationPolicy.layer_kwargs(
            image,
            ImagePayloadMetadata(source_channel_axis=-1),
            "green",
        ),
    )

    assert layer in viewer.layers
    assert viewer.calls[0][3] == {"blending": "additive", "rgb": True}


def test_napari_image_presentation_uses_payload_local_color_axis():
    image_payload = np.zeros((16, 16, 3), dtype=np.uint8)

    layer_kwargs = NapariImageLayerPresentationPolicy.layer_kwargs(
        image_payload,
        ImagePayloadMetadata(source_channel_axis=2),
        "green",
    )

    assert layer_kwargs == {"blending": "additive", "rgb": True}


def test_napari_image_display_stacks_sites_without_rebasing_payload_color_axis():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    route_key = "route-rgb-sites"
    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.layer_route_state.set_title(route_key, "RGB sites")
    server.viewer = _FakeViewer()
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    napari_viewer_server.NapariImageLayerDisplayHandler().handle(
        napari_viewer_server.NapariLayerDisplayRequest(
            pipeline=pipeline,
            presentation=_axis_presentation(
                layer_key=route_key,
                projected_axis_components=("site",),
                component_values={"site": [1, 2]},
            ),
            items=[
                _layer_item(
                    {"site": site},
                    data=np.full((16, 16, 3), site, dtype=np.uint8),
                    image_metadata=ImagePayloadMetadata(source_channel_axis=2),
                )
                for site in (1, 2)
            ],
        )
    )

    layer_type, data, name, layer_kwargs = server.viewer.calls[-1]
    assert layer_type == "image"
    assert name == "RGB sites"
    assert data.shape == (2, 16, 16, 3)
    assert layer_kwargs == {
        "axis_labels": ("site", "y", "x"),
        "blending": "additive",
        "rgb": True,
        "translate": (0.0, 0.0, 0.0),
    }


def test_napari_image_display_materializes_declared_source_spatial_domain():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    route_key = "route-cropped-image"
    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.layer_route_state.set_title(route_key, "Cropped image")
    server.viewer = _FakeViewer()
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    cropped = np.arange(6, dtype=np.uint8).reshape(2, 3)

    napari_viewer_server.NapariImageLayerDisplayHandler().handle(
        napari_viewer_server.NapariLayerDisplayRequest(
            pipeline=pipeline,
            presentation=_axis_presentation(
                layer_key=route_key,
                projected_axis_components=(),
            ),
            items=[
                _layer_item(
                    {},
                    data=cropped,
                    image_metadata=ImagePayloadMetadata(
                        source_spatial_domain=SourceSpatialDomain(
                            origin_yx=(2, 3),
                            source_shape_yx=(6, 8),
                        )
                    ),
                )
            ],
        )
    )

    layer_type, data, name, _layer_kwargs = server.viewer.calls[-1]
    assert layer_type == "image"
    assert name == "Cropped image"
    assert data.shape == (6, 8)
    np.testing.assert_array_equal(data[2:4, 3:6], cropped)
    assert np.count_nonzero(data[:2]) == 0


def test_napari_display_pipeline_applies_dimension_labels_to_layer_route_state():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())

    axis_labels = pipeline.dimension_label_store.apply(
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


def test_napari_display_pipeline_rejects_unbound_payload_local_image_axes():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    policy = napari_viewer_server.NapariImagePayloadAxisLabelPolicy()

    with pytest.raises(ValueError, match="component-axis binding"):
        policy.axis_labels(
            np.zeros((8, 16, 16)),
            ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
        )
    color_stack_metadata = ImagePayloadMetadata(
        source_channel_axis=-1,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    with pytest.raises(ValueError, match="component-axis binding"):
        policy.axis_labels(np.zeros((4, 16, 16, 3)), color_stack_metadata)
    assert (
        policy.axis_labels(np.zeros((4, 16, 16, 3)), color_stack_metadata, (0,)) == ()
    )
    assert (
        policy.axis_labels(
            np.zeros((16, 16, 3)), ImagePayloadMetadata(source_channel_axis=-1)
        )
        == ()
    )


def test_napari_display_pipeline_labels_collapsed_route_components():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())

    axis_labels = pipeline.dimension_label_store.apply(
        _axis_presentation(
            layer_key="route-stack",
            projected_axis_components=("site",),
            component_values={"site": [1, 2]},
            scalar_component_values={
                "channel": [1],
                "timepoint": [1],
                "z_index": [1],
            },
        )
    )

    state = pipeline.server.layer_route_state.dimension_state_for("route-stack")
    assert axis_labels == ("site", "y", "x")
    assert state.labels == {"site": ["Site 1", "Site 2"]}
    assert state.scalar_labels == ("Ch1: DAPI",)


def test_napari_display_pipeline_rejects_nonsemantic_payload_axis_labels() -> None:
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())

    with pytest.raises(TypeError, match="semantic axis-name strings"):
        pipeline.dimension_label_store.apply(
            _axis_presentation(
                layer_key="route-labels",
                projected_axis_components=("channel",),
                component_values={"channel": [1]},
                payload_axis_labels=(0,),  # type: ignore[arg-type]
            )
        )


def test_napari_display_pipeline_projects_route_axes_locally():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())
    component_value_domain = _component_value_domain(
        {"channel": [1, 2, 3, 4, 5], "site": [1, 2]}
    )
    component_axis_semantics = (
        ViewerComponentAxisSemanticsAuthority.from_display_config(
            ViewerMappingDisplayConfigInput(
                {
                    "component_modes": {"channel": "stack", "site": "stack"},
                    "component_order": ["channel", "site"],
                }
            ),
            component_value_domain,
        )
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
    assert second.scalar_component_values == {}
    assert second.translate() == (3.0, 0.0, 0.0, 0.0)


def test_napari_display_pipeline_aligns_derived_route_to_observed_source_domain():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())
    component_axis_semantics = (
        ViewerComponentAxisSemanticsAuthority.from_display_config(
            ViewerMappingDisplayConfigInput(
                {
                    "component_modes": {"channel": "stack"},
                    "component_order": ["channel"],
                }
            ),
            _component_value_domain({"channel": [1, 2, 4]}),
        )
    )

    source = pipeline.display_axis_projection(
        "source",
        component_axis_semantics,
        [
            _layer_item({"channel": 1}),
            _layer_item({"channel": 4}),
        ],
    )
    derived = pipeline.display_axis_projection(
        "derived",
        component_axis_semantics,
        [_layer_item({"channel": 4})],
    )

    assert source.component_values == {"channel": [1, 4]}
    assert source.axis_offsets == (0,)
    assert derived.component_values == {"channel": [4]}
    assert derived.axis_offsets == (1,)
    assert derived.translate() == (1.0, 0.0, 0.0)


def test_napari_display_pipeline_projects_aggregate_payload_axes_into_route_domain():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(_FakeNapariServer())
    component_value_domain = _component_value_domain(
        {"channel": [1], "z_index": [1, 2]}
    )
    component_axis_semantics = (
        ViewerComponentAxisSemanticsAuthority.from_display_config(
            ViewerMappingDisplayConfigInput(
                {
                    "component_modes": {"channel": "stack", "z_index": "stack"},
                    "component_order": ["z_index", "channel"],
                }
            ),
            component_value_domain,
        )
    )

    projection = pipeline.display_axis_projection(
        "z-stack",
        component_axis_semantics,
        [
            _layer_item({"channel": 1}, data=np.ones((2, 4, 4))),
        ],
        NapariAggregateAxisBindingSet(
            (NapariAggregateAxisBinding("z_index", 0, (1, 2)),)
        ),
    )

    assert projection.projected_axis_components == ("z_index",)
    assert projection.component_values == {"z_index": [1, 2]}
    assert projection.scalar_component_values == {"channel": [1]}
    assert projection.translate() == (0.0, 0.0, 0.0)


def test_napari_axis_projector_validates_declared_domain_and_drops_route_singletons():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")

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
    assert projection.routed_component_values == {"channel": [4]}
    assert projection.axis_offsets == (0,)
    assert projection.scalar_component_values == {"z_index": [1]}


def test_napari_axis_projector_rejects_missing_declared_singleton_component():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")

    with pytest.raises(
        ValueError, match="route component domain for 'timepoint' is empty"
    ):
        ViewerLayerAxisProjector().project(
            ViewerLayerAxisProjectionRequest.from_component_values(
                projected_axis_components=("timepoint", "well", "channel"),
                route_component_values={
                    "timepoint": [],
                    "well": ["A01"],
                    "channel": [1, 2],
                },
                viewer_component_values={
                    "timepoint": [],
                    "well": ["A01"],
                    "channel": [1, 2],
                },
                declared_component_values={
                    "timepoint": [1],
                    "well": ["A01"],
                    "channel": [1, 2],
                },
            )
        )


def test_napari_axis_projector_rejects_missing_empty_declared_component():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")

    with pytest.raises(
        ValueError, match="route component domain for 'timepoint' is empty"
    ):
        ViewerLayerAxisProjector().project(
            ViewerLayerAxisProjectionRequest.from_component_values(
                projected_axis_components=("timepoint", "well", "channel"),
                route_component_values={
                    "timepoint": [],
                    "well": ["A01"],
                    "channel": [1, 2],
                },
                viewer_component_values={
                    "timepoint": [],
                    "well": ["A01"],
                    "channel": [1, 2],
                },
                declared_component_values={
                    "timepoint": [],
                    "well": ["A01"],
                    "channel": [1, 2],
                },
            )
        )


def test_napari_axis_projector_rejects_missing_non_singleton_component():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")

    with pytest.raises(
        ValueError, match="route component domain for 'timepoint' is empty"
    ):
        ViewerLayerAxisProjector().project(
            ViewerLayerAxisProjectionRequest.from_component_values(
                projected_axis_components=("timepoint", "channel"),
                route_component_values={"timepoint": [], "channel": [1]},
                viewer_component_values={"timepoint": [], "channel": [1]},
                declared_component_values={"timepoint": [1, 2], "channel": [1]},
            )
        )


def test_napari_axis_projector_keeps_singleton_route_in_shared_viewer_domain():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")

    projection = ViewerLayerAxisProjector().project(
        ViewerLayerAxisProjectionRequest.from_component_values(
            projected_axis_components=("site", "channel"),
            route_component_values={"site": [1, 2], "channel": [4]},
            viewer_component_values={"site": [1, 2], "channel": [1, 2, 3, 4, 5]},
            declared_component_values={"site": [1, 2], "channel": [1, 2, 3, 4, 5]},
        )
    )

    assert projection.projected_axis_components == ("site", "channel")
    assert projection.component_values == {"site": [1, 2], "channel": [4]}
    assert projection.routed_component_values == {"site": [1, 2], "channel": [4]}
    assert projection.axis_offsets == (0, 3)
    assert projection.scalar_component_values == {}


def test_napari_axis_projector_preserves_route_offset_in_shared_viewer_domain():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")

    projection = ViewerLayerAxisProjector().project(
        ViewerLayerAxisProjectionRequest.from_component_values(
            projected_axis_components=("site", "channel"),
            route_component_values={"site": [1, 2], "channel": [4, 5]},
            viewer_component_values={"site": [1, 2], "channel": [1, 2, 3, 4, 5]},
            declared_component_values={"site": [1, 2], "channel": [1, 2, 3, 4, 5]},
        )
    )

    assert projection.projected_axis_components == ("site", "channel")
    assert projection.component_values == {"site": [1, 2], "channel": [4, 5]}
    assert projection.axis_offsets == (0, 3)
    assert projection.translate() == (0.0, 3.0, 0.0, 0.0)


def test_napari_axis_projector_uses_route_domain_for_noncontiguous_routes():
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
    assert projection.component_values == {"channel": [1, 3]}
    assert projection.routed_component_values == {"channel": [1, 3]}
    assert projection.axis_offsets == (0,)
    assert projection.scalar_component_values == {}


def test_napari_axis_projector_keeps_route_domain_for_noncontiguous_shared_axis():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")

    projection = ViewerLayerAxisProjector().project(
        ViewerLayerAxisProjectionRequest.from_component_values(
            projected_axis_components=("channel",),
            route_component_values={"channel": [1, 2, 3, 5]},
            viewer_component_values={"channel": [1, 2, 3, 4, 5]},
            declared_component_values={"channel": [1, 2, 3, 4, 5]},
        )
    )

    assert projection.projected_axis_components == ("channel",)
    assert projection.component_values == {"channel": [1, 2, 3, 4, 5]}
    assert projection.routed_component_values == {"channel": [1, 2, 3, 5]}
    assert projection.axis_offsets == (0,)
    assert projection.scalar_component_values == {}


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
    metadata = _component_name_metadata(
        {
            "channel": {"1": "OrigDNA", "4": "OrigActin_Golgi_Membrane"},
            "well": {"A14": None},
        }
    )

    metadata.merge(
        _component_name_metadata(
            {"channel": {"1": "OrigDNA"}, "well": {"B13": None}},
        )
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

    pipeline.dimension_label_overlay.setup_for_layer("nuclei")

    assert server.viewer.dims.axis_labels == ("channel", "y", "x")
    assert server.viewer.text_overlay.text == "Ch1: DAPI"


def test_napari_display_pipeline_applies_value_overlay_for_offset_route():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    class FakeSelection:
        active = None

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = None
        current_step = (0, 0, 0)
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
        "filtered-well",
        NapariDimensionLayerState(
            labels={"well": ["Well B03"]},
            presentation=_axis_presentation(
                layer_key="filtered-well",
                projected_axis_components=("well",),
                component_values={"well": ["B03"]},
                axis_offsets=(1,),
            ),
        ),
    )

    pipeline.dimension_label_overlay.setup_for_layer("filtered-well")

    assert server.viewer.dims.axis_labels == ("well", "y", "x")
    assert server.viewer.text_overlay.text == "Well B03"
    assert server.layer_route_state.active_dimension_label_route == "filtered-well"


def test_napari_display_pipeline_uses_route_local_steps_for_offset_labels():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    class FakeSelection:
        active = None

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = None
        current_step = (0, 1, 0, 0)
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

    pipeline.dimension_label_overlay.setup_for_layer("spots")

    assert server.viewer.dims.axis_labels == ("channel", "site", "y", "x")
    assert server.viewer.text_overlay.text == "Ch4 | Site 2"


def test_napari_navigation_control_selects_visible_layer_and_route_local_axes():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    viewer = _FakeViewer()
    layer = type(
        "Layer",
        (),
        {
            "name": "Objects",
            "data": np.zeros((1, 2, 20, 20), dtype=np.uint16),
            "translate": (3.0, 0.0, 0.0, 0.0),
            "visible": False,
        },
    )()
    viewer.layers.append(layer)

    server = type("Server", (), {})()
    server.viewer = viewer
    server.napari_window_title = "OpenHCS Napari Viewer"
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.layer_route_state.set_title("objects", "Objects")
    server.layer_route_state.set_layer("objects", layer)
    server.layer_route_state.set_dimension_state(
        "objects",
        NapariDimensionLayerState(
            labels={"channel": ["Ch4"], "site": ["Site 1", "Site 2"]},
            presentation=_axis_presentation(
                layer_key="objects",
                projected_axis_components=("channel", "site"),
                component_values={"channel": [4], "site": [1, 2]},
                axis_offsets=(3, 0),
            ),
        ),
    )
    server.component_groups = NapariComponentGroupStore()
    server.display_pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    response = napari_viewer_server.NapariNavigationControlMessageAction().handle(
        server,
        {
            "type": "navigate",
            ViewerControlResponseField.PAYLOAD.value: ViewerNavigationControlOptions(
                route_key="objects",
                axis_indices={"channel": 0, "site": 1},
                visible=True,
                selected=True,
            ),
        },
    )

    assert response["status"] == "success"
    assert viewer.dims.current_step == (0, 1, 0, 0)
    assert viewer.dims.axis_labels == ("channel", "site", "y", "x")
    assert viewer.layers.selection.active is layer
    assert layer.visible is True
    assert viewer.text_overlay.text == "Ch4 | Site 2"
    assert response["current_step"] == (0, 1, 0, 0)
    assert response["layers"][0]["selected"] is True
    assert response["layers"][0]["visible"] is True


def test_napari_display_pipeline_includes_collapsed_component_labels_in_overlay():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    class FakeSelection:
        active = None

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = None
        current_step = (1, 0, 0)
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
        "rna",
        NapariDimensionLayerState(
            labels={"site": ["Site 1", "Site 2"]},
            scalar_labels=("Ch4",),
            presentation=_axis_presentation(
                layer_key="rna",
                projected_axis_components=("site",),
                component_values={"site": [1, 2]},
                scalar_component_values={"channel": [4]},
            ),
        ),
    )
    server.layer_route_state.set_active_dimension_label_route("rna")

    pipeline.dimension_label_overlay.setup_for_layer("rna")

    assert server.viewer.dims.axis_labels == ("site", "y", "x")
    assert server.viewer.text_overlay.text == "Ch4 | Site 2"


def test_napari_display_pipeline_uses_updated_route_when_selected_route_ndim_mismatches():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    selected_layer = object()
    updated_layer = object()

    class FakeSelection:
        active = selected_layer

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = None
        current_step = (1, 4, 0, 0)
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
    server.layer_route_state.set_layer("identify-primary", selected_layer)
    server.layer_route_state.set_dimension_state(
        "identify-primary",
        NapariDimensionLayerState(
            labels={"site": ["Site 1", "Site 2"]},
            presentation=_axis_presentation(
                layer_key="identify-primary",
                projected_axis_components=("site",),
                component_values={"site": [1, 2]},
            ),
        ),
    )
    server.layer_route_state.set_layer("measure-colocalization", updated_layer)
    server.layer_route_state.set_dimension_state(
        "measure-colocalization",
        NapariDimensionLayerState(
            labels={
                "site": ["Site 1", "Site 2"],
                "channel": ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5"],
            },
            presentation=_axis_presentation(
                layer_key="measure-colocalization",
                projected_axis_components=("site", "channel"),
                component_values={
                    "site": [1, 2],
                    "channel": [1, 2, 3, 4, 5],
                },
            ),
        ),
    )
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    pipeline.dimension_label_overlay.setup_for_layer("measure-colocalization")

    assert server.layer_route_state.active_dimension_label_route == (
        "measure-colocalization"
    )
    assert server.viewer.dims.axis_labels == ("site", "channel", "y", "x")
    assert server.viewer.text_overlay.text == "Site 2 | Ch5"


def test_napari_display_pipeline_applies_updated_derived_route_overlay():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    selected_layer = object()
    derived_layer = object()

    class FakeSelection:
        active = selected_layer

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = None
        current_step = (0, 0, 0)
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
    server.layer_route_state.set_layer("source-map2", selected_layer)
    server.layer_route_state.set_dimension_state(
        "source-map2",
        NapariDimensionLayerState(
            labels={"channel": ["Ch2: MAP2"]},
            presentation=_axis_presentation(
                layer_key="source-map2",
                projected_axis_components=("channel",),
                component_values={"channel": [2]},
            ),
        ),
    )
    server.layer_route_state.set_layer("derived-neurite-skeleton", derived_layer)
    server.layer_route_state.set_dimension_state(
        "derived-neurite-skeleton",
        NapariDimensionLayerState(
            labels={"channel": ["Ch4: SMI312"]},
            presentation=_axis_presentation(
                layer_key="derived-neurite-skeleton",
                projected_axis_components=("channel",),
                component_values={"channel": [4]},
            ),
        ),
    )
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    pipeline.dimension_label_overlay.setup_for_layer("derived-neurite-skeleton")

    assert server.layer_route_state.active_dimension_label_route == (
        "derived-neurite-skeleton"
    )
    assert server.viewer.dims.axis_labels == ("channel", "y", "x")
    assert server.viewer.text_overlay.text == "Ch4: SMI312"
    assert server.viewer.layers.selection.active is selected_layer

    pipeline.dimension_label_overlay._update_overlay()

    assert server.layer_route_state.active_dimension_label_route == "source-map2"
    assert server.viewer.text_overlay.text == "Ch2: MAP2"


def test_napari_display_pipeline_preserves_active_non_openhcs_layer_during_update():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.viewer = _FakeViewer()
    server.viewer.dims.ndim = 3
    server.viewer.dims.current_step = (0, 0, 0)
    server.viewer.text_overlay.text = "stale OpenHCS overlay"
    server.viewer.layers.selection.active = object()
    derived_layer = object()
    server.layer_route_state.set_layer("derived", derived_layer)
    server.layer_route_state.set_dimension_state(
        "derived",
        NapariDimensionLayerState(
            labels={"channel": ["Ch4: SMI312"]},
            presentation=_axis_presentation(
                layer_key="derived",
                projected_axis_components=("channel",),
                component_values={"channel": [4]},
            ),
        ),
    )
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    pipeline.dimension_label_overlay.setup_for_layer("derived")

    assert server.layer_route_state.active_dimension_label_route is None
    assert server.viewer.dims.axis_labels is None
    assert server.viewer.text_overlay.text == ""


def test_napari_display_pipeline_falls_back_when_selected_route_lacks_current_step_labels():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    selected_layer = object()
    fallback_layer = object()

    class FakeSelection:
        active = selected_layer

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = None
        current_step = (0, 0, 1, 0, 0)
        ndim = 5

    class FakeTextOverlay:
        text = ""

    class FakeViewer:
        layers = FakeLayers()
        dims = FakeDims()
        text_overlay = FakeTextOverlay()

    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.viewer = FakeViewer()
    server.layer_route_state.set_layer("identify-primary", selected_layer)
    server.layer_route_state.set_dimension_state(
        "identify-primary",
        NapariDimensionLayerState(
            labels={
                "well": ["A14", "B13"],
                "site": ["Site 1", "Site 2"],
                "channel": ["Ch1: OrigDNA"],
            },
            presentation=_axis_presentation(
                layer_key="identify-primary",
                projected_axis_components=("well", "site", "channel"),
                component_values={
                    "well": ["A14", "B13"],
                    "site": [1, 2],
                    "channel": [1],
                },
            ),
        ),
    )
    server.layer_route_state.set_layer("measure-colocalization", fallback_layer)
    server.layer_route_state.set_dimension_state(
        "measure-colocalization",
        NapariDimensionLayerState(
            labels={
                "well": ["A14", "B13"],
                "site": ["Site 1", "Site 2"],
                "channel": [
                    "Ch1: OrigDNA",
                    "Ch2: OrigER",
                    "Ch3: OrigRNA",
                    "Ch4: OrigActin_Golgi_Membrane",
                    "Ch5: OrigMito",
                ],
            },
            presentation=_axis_presentation(
                layer_key="measure-colocalization",
                projected_axis_components=("well", "site", "channel"),
                component_values={
                    "well": ["A14", "B13"],
                    "site": [1, 2],
                    "channel": [1, 2, 3, 4, 5],
                },
            ),
        ),
    )
    server.layer_route_state.set_active_dimension_label_route("identify-primary")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    pipeline.dimension_label_overlay._update_overlay()

    assert server.viewer.dims.axis_labels == ("well", "site", "channel", "y", "x")
    assert server.viewer.text_overlay.text == "A14 | Site 1 | Ch2: OrigER"


def test_napari_display_pipeline_falls_back_from_selected_offset_channel():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    source_layer = object()
    selected_neurite_layer = object()

    class FakeSelection:
        active = selected_neurite_layer

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = None
        current_step = (0, 0, 0)
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
    server.layer_route_state.set_layer("source", source_layer)
    server.layer_route_state.set_dimension_state(
        "source",
        NapariDimensionLayerState(
            labels={"channel": ["Ch1: Hoechst", "Ch2: MAP2", "Ch4: SMI312"]},
            presentation=_axis_presentation(
                layer_key="source",
                projected_axis_components=("channel",),
                component_values={"channel": [1, 2, 4]},
            ),
        ),
    )
    server.layer_route_state.set_layer("neurites", selected_neurite_layer)
    server.layer_route_state.set_dimension_state(
        "neurites",
        NapariDimensionLayerState(
            labels={"channel": ["Ch4: SMI312"]},
            presentation=_axis_presentation(
                layer_key="neurites",
                projected_axis_components=("channel",),
                component_values={"channel": [4]},
                axis_offsets=(2,),
            ),
        ),
    )
    server.layer_route_state.set_active_dimension_label_route("neurites")
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    pipeline.dimension_label_overlay._update_overlay()

    assert server.layer_route_state.active_dimension_label_route == "source"
    assert server.viewer.text_overlay.text == "Ch1: Hoechst"


def test_napari_display_pipeline_labels_all_slices_from_nonzero_axis_origin():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    map2_layer = object()
    smi312_layer = object()

    class FakeSelection:
        active = map2_layer

    class FakeLayers:
        selection = FakeSelection()

    class FakeDims:
        axis_labels = None
        current_step = (0, 0, 0)
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
    server.layer_route_state.set_layer("map2", map2_layer)
    server.layer_route_state.set_dimension_state(
        "map2",
        NapariDimensionLayerState(
            labels={"channel": ["Ch2: MAP2"]},
            presentation=_axis_presentation(
                layer_key="map2",
                projected_axis_components=("channel",),
                component_values={"channel": [2]},
                axis_offsets=(1,),
            ),
        ),
    )
    server.layer_route_state.set_layer("smi312", smi312_layer)
    server.layer_route_state.set_dimension_state(
        "smi312",
        NapariDimensionLayerState(
            labels={"channel": ["Ch4: SMI312"]},
            presentation=_axis_presentation(
                layer_key="smi312",
                projected_axis_components=("channel",),
                component_values={"channel": [4]},
                axis_offsets=(2,),
            ),
        ),
    )
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    pipeline.dimension_label_overlay._update_overlay()

    assert server.layer_route_state.active_dimension_label_route == "map2"
    assert server.viewer.text_overlay.text == "Ch2: MAP2"

    server.viewer.dims.current_step = (1, 0, 0)
    pipeline.dimension_label_overlay._update_overlay()

    assert server.layer_route_state.active_dimension_label_route == "smi312"
    assert server.viewer.text_overlay.text == "Ch4: SMI312"


def test_napari_axis_origin_includes_implicit_zero_route_offsets():
    state_store = NapariLayerRouteStateStore.empty()
    state_store.set_dimension_state(
        "implicit-zero",
        NapariDimensionLayerState(
            labels={"channel": ["Ch1"]},
            presentation=_axis_presentation(
                layer_key="implicit-zero",
                projected_axis_components=("channel",),
                axis_offsets=(),
            ),
        ),
    )
    state_store.set_dimension_state(
        "offset",
        NapariDimensionLayerState(
            labels={"channel": ["Ch4"]},
            presentation=_axis_presentation(
                layer_key="offset",
                projected_axis_components=("channel",),
                axis_offsets=(2,),
            ),
        ),
    )

    assert state_store.axis_origins_for(("channel", "y", "x")) == (0, 0, 0)


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

    pipeline.dimension_label_overlay.setup_for_layer("labels")

    assert server.viewer.dims.axis_labels == (
        "well",
        "site",
        "channel",
        "y",
        "x",
    )


def test_napari_display_pipeline_registers_recreated_shapes_before_selection_restore():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    route_key = "route-shapes"
    old_layer = type("Layer", (), {"name": "old-shapes"})()
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

        def add_shapes(self, data, *, name, **kwargs):
            layer = type(
                "Layer",
                (),
                {
                    "name": name,
                    "data": data,
                    "kwargs": kwargs,
                    "edge_color_mode": "cycle",
                    "face_color_mode": "cycle",
                },
            )()
            self.layers.append(layer)
            return layer

    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.layer_route_state.set_title(route_key, "ROIs")
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

    work = pipeline.display_layer_batch(
        layer_key=route_key,
        items=[
            _layer_item(
                {}, data=shapes, stream_layer_data_type=StreamingDataType.SHAPES
            )
        ],
        display_payload=ViewerComponentAxisSemanticsAuthority.empty(),
        component_names_metadata=ViewerComponentNameMetadata.empty(),
    )
    assert work.advance()

    assert registration_observed == [True]


def test_napari_viewer_clear_state_resets_accumulated_axis_domains():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    class PendingTimer:
        stopped = False

        def stop(self):
            self.stopped = True

    server = napari_viewer_server.NapariViewerServer.__new__(
        napari_viewer_server.NapariViewerServer
    )
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    pending_timer = PendingTimer()
    server.layer_route_state.set_pending_update(
        "old",
        NapariPendingLayerUpdate.from_semantics(
            timer=pending_timer,
            data_type=StreamingDataType.IMAGE,
            semantics=ViewerComponentAxisSemanticsAuthority.empty(),
        ),
    )
    server.component_groups = NapariComponentGroupStore()
    server.component_groups.items_for("old").append(_layer_item({"well": "A14"}))
    server.component_values = ViewerRouteComponentValueTracker()
    server.component_values.update("old", ["well"], [_layer_item({"well": "A14"})])
    server.display_axis_domain = ViewerDisplayAxisDomain()
    server.display_axis_domain.record_display_axis_values(
        ["well"],
        [_layer_item({"well": "A14"})],
    )
    server.component_name_metadata = _component_name_metadata({"well": {"A14": "A14"}})
    server.layer_batch_processor_debounce_policy = NapariLayerBatchDebouncePolicy(
        delay_ms=123
    )
    server.display_pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    server.batch_processors = NapariBatchProcessorStore(
        debounce_policy=NapariLayerBatchDebouncePolicy(delay_ms=1)
    )

    server.clear_accumulated_stream_state()

    assert len(server.component_groups) == 0
    assert server.component_values.values_for(("old", ("well",)), ["well"]) == {
        "well": []
    }
    assert server.display_axis_domain.display_axis_values_for(["well"]) == {"well": []}
    assert server.component_name_metadata.to_wire_mapping() == {}
    assert server.batch_processors.debounce_policy.delay_ms == 123
    assert pending_timer.stopped is True
    assert server.layer_route_state.layer_pending_updates == {}


def _run_fake_napari_entrypoint(
    monkeypatch,
    *,
    start_error=None,
    ready=False,
    shutdown_during_service=False,
):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    from qtpy import QtWidgets

    events = []

    class FakeSignal:
        def __init__(self, event_name):
            self.event_name = event_name
            self.callback = None

        def connect(self, callback):
            self.callback = callback
            events.append(f"{self.event_name}_connect")

    class FakeRouteState:
        def set_layer(self, _name, _layer):
            events.append("existing_layer")

    class FakeDataSocket:
        def recv(self, _flags):
            events.append("data_receive")
            return b"payload"

    class FakeServer:
        def __init__(self, request):
            self.control_port = (
                request.port + OPENHCS_ZMQ_CONFIG.control_port_offset
            )
            self.layer_route_state = FakeRouteState()
            self.viewer = None
            self._ready = ready
            self._running = False
            self.data_socket = FakeDataSocket()
            self.result_selection_controller = object()
            events.append("server_construct")

        def start(self):
            events.append("server_start")
            if start_error is not None:
                raise start_error
            self._running = True

        def stop(self):
            self._running = False
            events.append("server_stop")

        def is_running(self):
            return self._running

        def process_messages(self):
            events.append("server_process")
            if shutdown_during_service:
                self._running = False

        def process_accepted_stream_messages(self):
            events.append("accepted_stream_process")
            return 0

    class FakeTimer:
        zero_shot_callback = None
        message_timer = None

        def __init__(self, parent=None):
            self.parent = parent
            self.timeout = FakeSignal("message_timer")
            type(self).message_timer = self
            events.append("message_timer_construct")

        @classmethod
        def singleShot(cls, interval_ms, callback):
            assert interval_ms == 0
            cls.zero_shot_callback = callback
            events.append("startup_callback_queued")

        def start(self, interval_ms):
            assert interval_ms == 50
            events.append("message_timer_start")

        def stop(self):
            events.append("message_timer_stop")

    class FakeApplication:
        current = None

        @classmethod
        def instance(cls):
            return cls.current

        def setQuitOnLastWindowClosed(self, enabled):
            assert enabled is True
            events.append("quit_on_close")

        def exec_(self):
            events.append("event_loop_enter")
            assert "server_start" not in events
            assert FakeTimer.zero_shot_callback is not None
            FakeTimer.zero_shot_callback()
            if start_error is None:
                assert FakeTimer.message_timer is not None
                assert FakeTimer.message_timer.timeout.callback is not None
                FakeTimer.message_timer.timeout.callback()
            events.append("event_loop_exit")

        def quit(self):
            events.append("application_quit")

    class FakeViewer:
        def __init__(self):
            self.layers = []
            self.text_overlay = type("TextOverlay", (), {})()

            class FakeGeometry:
                @staticmethod
                def width():
                    return 1920

                @staticmethod
                def height():
                    return 1080

            class FakeScreen:
                @staticmethod
                def availableGeometry():
                    return FakeGeometry()

            class FakeQtWindow:
                @staticmethod
                def screen():
                    return FakeScreen()

                @staticmethod
                def width():
                    return 1027

                @staticmethod
                def height():
                    return 596

                @staticmethod
                def setCorner(corner, area):
                    events.append(("dock_corner", corner, area))

                @staticmethod
                def addDockWidget(area, dock):
                    events.append(("dock_area", area, dock))

                @staticmethod
                def resizeDocks(docks, sizes, orientation):
                    events.append(("dock_resize", docks, sizes, orientation))

            qt_window = FakeQtWindow()

            class FakeDock:
                @staticmethod
                def window():
                    return qt_window

            feature_table_dock = FakeDock()

            class Window:
                qt_viewer = type(
                    "QtViewer",
                    (),
                    {
                        "destroyed": FakeSignal("destroyed"),
                        "window": staticmethod(lambda: qt_window),
                    },
                )()

                @staticmethod
                def add_plugin_dock_widget(plugin_name, widget_name):
                    assert plugin_name == "napari"
                    assert widget_name == "Features table widget"
                    events.append("features_table_open")
                    return feature_table_dock, object()

                @staticmethod
                def resize(width, height):
                    events.append(("window_resize", width, height))

            self.window = Window()

    application = FakeApplication()
    FakeApplication.current = application
    monkeypatch.setattr(QtWidgets, "QApplication", FakeApplication)
    monkeypatch.setattr(napari_viewer_server, "NapariViewerServer", FakeServer)
    monkeypatch.setattr(napari_viewer_server, "QTimer", FakeTimer)
    monkeypatch.setattr(
        napari_viewer_server,
        "_install_result_selection_toolbar",
        lambda _dock, _controller: (
            events.append("result_selection_toolbar_open") or object()
        ),
    )
    monkeypatch.setattr(
        napari_viewer_server.napari,
        "Viewer",
        lambda **_kwargs: FakeViewer(),
    )
    entrypoint_error = None
    try:
        napari_viewer_server.run_napari_viewer_process(
            5563,
            "test viewer",
        )
    except Exception as error:
        entrypoint_error = error
    return events, entrypoint_error


def test_napari_entrypoint_publishes_endpoints_from_live_qt_event_loop(monkeypatch):
    from qtpy.QtCore import Qt

    events, entrypoint_error = _run_fake_napari_entrypoint(monkeypatch)

    assert entrypoint_error is None
    assert events.index("startup_callback_queued") < events.index("event_loop_enter")
    assert events.index("features_table_open") < events.index("startup_callback_queued")
    assert events.index("features_table_open") < events.index(
        "result_selection_toolbar_open"
    )
    assert events.index("event_loop_enter") < events.index("message_timer_construct")
    assert events.index("message_timer_connect") < events.index("message_timer_start")
    assert events.index("message_timer_start") < events.index("server_start")
    assert events.index("server_start") < events.index("event_loop_exit")
    assert ("window_resize", 1728, 972) in events
    dock_area_event = next(
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "dock_area"
    )
    feature_table_dock = dock_area_event[2]
    assert dock_area_event == (
        "dock_area",
        Qt.DockWidgetArea.BottomDockWidgetArea,
        feature_table_dock,
    )
    assert (
        "dock_resize",
        [feature_table_dock],
        [311],
        Qt.Orientation.Vertical,
    ) in events
    assert {
        event[1]
        for event in events
        if isinstance(event, tuple) and event[0] == "dock_corner"
    } == {Qt.Corner.BottomLeftCorner, Qt.Corner.BottomRightCorner}
    assert "destroyed_connect" not in events
    assert events.count("server_stop") == 1
    assert events[-1] == "server_stop"


def test_napari_default_window_layout_places_features_table_below_canvas(qtbot):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    import napari
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    viewer = napari.Viewer(show=True)
    feature_table_dock, _feature_table = viewer.window.add_plugin_dock_widget(
        "napari",
        "Features table widget",
    )
    qt_window = feature_table_dock.window()
    qtbot.addWidget(qt_window)

    napari_viewer_server._apply_default_window_layout(viewer, feature_table_dock)
    QApplication.processEvents()

    available_geometry = qt_window.screen().availableGeometry()
    assert qt_window.width() >= round(
        available_geometry.width()
        * napari_viewer_server._DEFAULT_WINDOW_SCREEN_FRACTION
    )
    assert qt_window.height() >= round(
        available_geometry.height()
        * napari_viewer_server._DEFAULT_WINDOW_SCREEN_FRACTION
    )
    assert (
        qt_window.dockWidgetArea(feature_table_dock)
        is Qt.DockWidgetArea.BottomDockWidgetArea
    )
    assert (
        qt_window.corner(Qt.Corner.BottomLeftCorner)
        is Qt.DockWidgetArea.BottomDockWidgetArea
    )
    assert (
        qt_window.corner(Qt.Corner.BottomRightCorner)
        is Qt.DockWidgetArea.BottomDockWidgetArea
    )
    assert feature_table_dock.width() == qt_window.width()
    assert feature_table_dock.height() >= round(qt_window.height() * 0.25)


def test_napari_scope_accent_styles_the_existing_main_window(qtbot):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    import napari

    viewer = napari.Viewer(show=True)
    feature_table_dock, _feature_table = viewer.window.add_plugin_dock_widget(
        "napari",
        "Features table widget",
    )
    qt_window = feature_table_dock.window()
    qtbot.addWidget(qt_window)

    napari_viewer_server._apply_scope_accent_styling(
        feature_table_dock,
        "#1464c8",
    )

    assert qt_window.property("openhcs_scope_accent_color") == "#1464c8"
    assert "border: 6px solid #1464c8" in qt_window.styleSheet()


def test_napari_runtime_launch_carries_the_projected_scope_accent():
    from openhcs.core.config import NapariStreamingConfig
    from openhcs.runtime.napari_stream_visualizer import NapariStreamVisualizer

    config = NapariStreamingConfig(
        port=6200,
        scope_accent_color="#1464c8",
    )
    visualizer = NapariStreamVisualizer(
        filemanager=object(),
        runtime_config=config.viewer_runtime_config(),
    )

    arguments = visualizer.detached_server_arguments(
        log_file=Path("/tmp/napari-6200.log")
    )

    assert arguments.expressions[-1].source == "'#1464c8'"


def test_napari_features_table_selects_authoritative_shapes_members(qtbot):
    from napari.components import ViewerModel
    from napari_builtins._qt.features_table import FeaturesTable
    from qtpy.QtCore import QItemSelectionModel

    viewer = ViewerModel()
    layer = viewer.add_shapes(
        [
            np.array([[0, 0], [0, 2], [2, 2]], dtype=float),
            np.array([[4, 4], [4, 6], [6, 6]], dtype=float),
        ],
        shape_type=["polygon", "polygon"],
        features={"label": [11, 12], "area": [3.0, 4.0]},
        name="OpenHCS ROIs",
    )
    _native_roi_selection_server(
        pytest.importorskip("openhcs.runtime.napari_viewer_server"),
        viewer,
        (("rois", layer, 1),),
    )
    widget = FeaturesTable(viewer)
    qtbot.addWidget(widget)

    widget.table.selectionModel().select(
        widget.table.model().index(1, 0),
        QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QItemSelectionModel.SelectionFlag.Rows,
    )

    assert len(viewer.layers) == 1
    assert layer.selected_data == {1}
    assert all(isinstance(index, np.integer) for index in layer.selected_data)
    assert widget.table.model().rowCount() == len(layer.data)

    layer.selected_data = {0}
    layer.events.highlight()

    assert [index.row() for index in widget.table.selectionModel().selectedRows()] == [
        0
    ]


def _native_roi_selection_server(napari_viewer_server, viewer, layers):
    server = napari_viewer_server.NapariViewerServer.__new__(
        napari_viewer_server.NapariViewerServer
    )
    server.viewer = viewer
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    for route_key, layer, site_count in layers:
        has_stack_axis = layer.ndim > 2
        projected_axis_components = ("site",) if has_stack_axis else ()
        component_values = {"site": list(range(site_count))} if has_stack_axis else {}
        server.layer_route_state.set_title(route_key, layer.name)
        server.layer_route_state.set_layer(route_key, layer)
        server.layer_route_state.set_dimension_state(
            route_key,
            NapariDimensionLayerState(
                labels=(
                    {"site": [f"Site {index}" for index in range(site_count)]}
                    if has_stack_axis
                    else {}
                ),
                presentation=_axis_presentation(
                    layer_key=route_key,
                    projected_axis_components=projected_axis_components,
                    component_values=component_values,
                ),
            ),
        )
    server.result_selection_controller = (
        napari_viewer_server.NapariResultSelectionController(server)
    )
    for _route_key, layer, _site_count in layers:
        server.result_selection_controller.bind(layer)
    return server


def test_declared_object_subject_selects_all_neuron_paths_and_metrics_row(qtbot):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    from napari.components import ViewerModel

    subject_feature = (
        napari_viewer_server.ObjectArtifactSubjectBinding.SUBJECT_FEATURE
    )
    subject_id_feature = (
        napari_viewer_server.ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE
    )
    viewer = ViewerModel()
    graph_layer = viewer.add_shapes(
        [
            np.array([[0, 0], [0, 2]], dtype=float),
            np.array([[0, 2], [1, 4]], dtype=float),
            np.array([[5, 5], [6, 7]], dtype=float),
        ],
        shape_type=["path", "path", "path"],
        features={
            "edge_id": [11, 12, 21],
            "branch_distance_um": [2.0, 2.4, 3.1],
        },
        metadata={
            subject_feature: "neurons@step-4",
            subject_id_feature: (1, 1, 2),
        },
        edge_color=["magenta", "magenta", "lime"],
        name="Neuron branches",
    )
    metrics_layer = viewer.add_shapes(
        [
            np.array([[0, 0], [0, 1], [1, 1]], dtype=float),
            np.array([[5, 5], [5, 6], [6, 6]], dtype=float),
        ],
        shape_type=["polygon", "polygon"],
        features={
            "total_outgrowth_um": [41.2, 22.7],
            "branches": [2, 1],
        },
        metadata={
            subject_feature: "neurons@step-4",
            subject_id_feature: (1, 2),
        },
        edge_color=["magenta", "lime"],
        name="Per-neuron metrics",
    )
    server = _native_roi_selection_server(
        napari_viewer_server,
        viewer,
        (
            ("neuron-branches", graph_layer, 1),
            ("neuron-metrics", metrics_layer, 1),
        ),
    )
    original_order = tuple(viewer.layers)

    graph_layer.selected_data = {1}
    qtbot.waitUntil(
        lambda: graph_layer.selected_data == {0, 1}
        and metrics_layer.selected_data == {0},
        timeout=2_000,
    )
    server.result_selection_controller.set_result_group_color(
        graph_layer,
        (1.0, 0.5, 0.0, 1.0),
    )
    np.testing.assert_allclose(
        np.asarray(graph_layer.edge_color)[:2],
        np.tile(np.asarray((1.0, 0.5, 0.0, 1.0)), (2, 1)),
    )
    np.testing.assert_allclose(
        np.asarray(metrics_layer.edge_color)[0],
        np.asarray((1.0, 0.5, 0.0, 1.0)),
    )
    assert not np.allclose(
        np.asarray(graph_layer.edge_color)[2],
        np.asarray((1.0, 0.5, 0.0, 1.0)),
    )

    metrics_layer.selected_data = {1}
    qtbot.waitUntil(
        lambda: graph_layer.selected_data == {2}
        and metrics_layer.selected_data == {1},
        timeout=2_000,
    )
    assert metrics_layer.features.iloc[1]["total_outgrowth_um"] == 22.7
    assert tuple(viewer.layers) == original_order


def _select_feature_table_layer_row(widget, layer_name: str) -> None:
    from qtpy.QtCore import QItemSelectionModel

    proxy_model = widget.table.model()
    source_model = proxy_model.sourceModel()
    layer_column = source_model.df.columns.get_loc("Layer")
    source_row = next(
        row
        for row in range(source_model.rowCount())
        if source_model.df.iloc[row, layer_column] == layer_name
    )
    proxy_index = proxy_model.mapFromSource(source_model.index(source_row, 0))
    widget.table.selectionModel().select(
        proxy_index,
        QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QItemSelectionModel.SelectionFlag.Rows,
    )


def test_features_table_selection_reveals_3d_roi_on_its_exact_slice(qtbot, monkeypatch):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    from napari.components import ViewerModel
    from napari.settings import get_settings
    from napari_builtins._qt.features_table import FeaturesTable

    settings = get_settings()
    monkeypatch.setattr(settings.appearance.highlight, "highlight_thickness", 1)
    viewer = ViewerModel(ndisplay=2)
    viewer.add_image(
        np.zeros((6, 64, 64), dtype=np.uint8),
        name="Reference image",
    )
    selected_layer = viewer.add_shapes(
        [
            np.array(
                [[3, 10, 10], [3, 10, 20], [3, 20, 20]],
                dtype=float,
            )
        ],
        shape_type="polygon",
        features={"label": [11], "area": [50.0]},
        ndim=3,
        name="First ROIs",
    )
    other_layer = viewer.add_shapes(
        [
            np.array(
                [[5, 30, 30], [5, 30, 40], [5, 40, 40]],
                dtype=float,
            )
        ],
        shape_type="polygon",
        features={"label": [22], "area": [60.0]},
        ndim=3,
        name="Second ROIs",
    )
    _native_roi_selection_server(
        napari_viewer_server,
        viewer,
        (
            ("first-rois", selected_layer, 6),
            ("second-rois", other_layer, 6),
        ),
    )
    selected_layer.visible = False
    viewer.dims.set_current_step(0, 0)
    viewer.layers.selection.update({selected_layer, other_layer})
    widget = FeaturesTable(viewer)
    qtbot.addWidget(widget)
    original_layer_order = tuple(viewer.layers)

    _select_feature_table_layer_row(widget, selected_layer.name)

    qtbot.waitUntil(
        lambda: selected_layer.visible and viewer.dims.current_step[0] == 3,
        timeout=2_000,
    )
    assert set(viewer.layers.selection) == {selected_layer, other_layer}
    assert viewer.layers.selection.active is None
    assert selected_layer.selected_data == {0}
    assert other_layer.selected_data == set()
    assert tuple(viewer.layers) == original_layer_order
    assert settings.appearance.highlight.highlight_thickness >= 3

    # A later redraw of the already-selected ROI must not steal a deliberate
    # user layer selection or collapse the table back to its old owner.
    viewer.layers.selection.active = other_layer
    selected_layer.events.highlight()
    qtbot.wait(10)
    assert viewer.layers.selection.active is other_layer
    assert tuple(viewer.layers.selection) == (other_layer,)

    # Route ownership follows the native layer object, never its list position.
    viewer.layers.move(viewer.layers.index(selected_layer), 0)
    reordered_layers = tuple(viewer.layers)
    viewer.layers.selection.update({selected_layer, other_layer})
    viewer.dims.set_current_step(0, 0)
    selected_layer.selected_data = set()
    selected_layer.selected_data = {0}
    qtbot.waitUntil(lambda: viewer.dims.current_step[0] == 3, timeout=2_000)
    assert set(viewer.layers.selection) == {selected_layer, other_layer}
    assert selected_layer.selected_data == {0}
    assert tuple(viewer.layers) == reordered_layers


def test_empty_shapes_selection_does_not_change_viewer_context(qtbot):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    from napari.components import ViewerModel

    viewer = ViewerModel(ndisplay=2)
    viewer.add_image(
        np.zeros((5, 32, 32), dtype=np.uint8),
        name="Reference image",
    )
    layer = viewer.add_shapes(
        [
            np.array(
                [[4, 10, 10], [4, 10, 20], [4, 20, 20]],
                dtype=float,
            )
        ],
        shape_type="polygon",
        features={"label": [11]},
        ndim=3,
        name="ROIs",
    )
    _native_roi_selection_server(
        napari_viewer_server,
        viewer,
        (("rois", layer, 5),),
    )
    layer.visible = False
    viewer.layers.selection.active = None
    viewer.dims.set_current_step(0, 0)

    layer.selected_data = set()
    layer.events.highlight()
    qtbot.wait(10)

    assert layer.visible is False
    assert viewer.layers.selection.active is None
    assert viewer.dims.current_step[0] == 0


def test_roi_selection_preserves_user_adjusted_native_highlight_thickness(
    qtbot,
    monkeypatch,
):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    from napari.components import ViewerModel
    from napari.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings.appearance.highlight, "highlight_thickness", 7)
    monkeypatch.setattr(
        settings.appearance.highlight,
        "highlight_color",
        [0.3, 0.9, 0.2, 1.0],
    )
    viewer = ViewerModel()
    layer = viewer.add_shapes(
        [np.array([[10, 10], [10, 20], [20, 20]], dtype=float)],
        shape_type="polygon",
        features={"label": [11]},
        name="ROIs",
    )
    _native_roi_selection_server(
        napari_viewer_server,
        viewer,
        (("rois", layer, 1),),
    )

    layer.selected_data = {0}
    qtbot.waitUntil(lambda: viewer.layers.selection.active is layer, timeout=2_000)

    assert settings.appearance.highlight.highlight_thickness == 7
    assert settings.appearance.highlight.highlight_color == [0.3, 0.9, 0.2, 1.0]


def test_roi_selection_toolbar_adjusts_native_highlight_setting(qtbot, monkeypatch):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    from napari.settings import get_settings
    from qtpy.QtWidgets import QMainWindow, QPushButton, QSpinBox

    settings = get_settings()
    monkeypatch.setattr(settings.appearance.highlight, "highlight_thickness", 1)
    monkeypatch.setattr(
        settings.appearance.highlight,
        "highlight_color",
        [0.0, 0.6, 1.0, 1.0],
    )
    qt_window = QMainWindow()
    qtbot.addWidget(qt_window)
    feature_table_dock = type(
        "FeatureTableDock",
        (),
        {"window": lambda self: qt_window},
    )()
    server = type("ResultSelectionServer", (), {"viewer": None})()
    controller = napari_viewer_server.NapariResultSelectionController(server)

    toolbar = napari_viewer_server._install_result_selection_toolbar(
        feature_table_dock,
        controller,
    )
    thickness = toolbar.findChild(QSpinBox, "openhcs_roi_highlight_thickness")
    color = toolbar.findChild(QPushButton, "openhcs_roi_highlight_color")
    layer_color = toolbar.findChild(QPushButton, "openhcs_roi_layer_color")

    assert thickness is not None
    assert color is not None
    assert layer_color is not None
    assert layer_color.isEnabled() is False
    assert thickness.minimum() == 1
    assert thickness.maximum() == 10
    assert thickness.value() >= 3
    thickness.setValue(6)
    assert settings.appearance.highlight.highlight_thickness == 6

    settings.appearance.highlight.highlight_thickness = 8
    qtbot.waitUntil(lambda: thickness.value() == 8, timeout=2_000)
    assert settings.appearance.highlight.highlight_color == [1.0, 0.82, 0.0, 1.0]
    assert "#ffd100" in color.styleSheet()

    controller.set_highlight_color((1.0, 0.0, 0.0, 1.0))
    qtbot.waitUntil(lambda: "#ff0000" in color.styleSheet(), timeout=2_000)
    settings.appearance.highlight.highlight_color = [0.0, 1.0, 0.0, 1.0]
    qtbot.waitUntil(lambda: "#00ff00" in color.styleSheet(), timeout=2_000)


def test_roi_layer_color_button_recolors_every_shape(qtbot, monkeypatch):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    from napari.components import ViewerModel
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QColorDialog, QMainWindow, QPushButton

    viewer = ViewerModel()
    layer = viewer.add_shapes(
        [
            np.array([[0, 0], [0, 2], [2, 2]], dtype=float),
            np.array([[4, 4], [4, 6], [6, 6]], dtype=float),
        ],
        shape_type=["polygon", "polygon"],
        features={"label": [11, 12]},
        edge_color=["red", "blue"],
        name="Result ROIs",
    )
    server = _native_roi_selection_server(
        napari_viewer_server,
        viewer,
        (("result-rois", layer, 1),),
    )
    qt_window = QMainWindow()
    qtbot.addWidget(qt_window)
    feature_table_dock = type(
        "FeatureTableDock",
        (),
        {"window": lambda self: qt_window},
    )()
    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        lambda *_args, **_kwargs: QColor.fromRgbF(0.2, 0.8, 0.3, 1.0),
    )

    toolbar = napari_viewer_server._install_result_selection_toolbar(
        feature_table_dock,
        server.result_selection_controller,
    )
    layer_color = toolbar.findChild(QPushButton, "openhcs_roi_layer_color")
    group_color = toolbar.findChild(QPushButton, "openhcs_roi_group_color")

    assert layer_color is not None
    assert group_color is not None
    assert layer_color.isEnabled()
    assert group_color.isEnabled() is False
    layer.selected_data = {0}
    qtbot.waitUntil(group_color.isEnabled, timeout=2_000)
    qtbot.mouseClick(group_color, Qt.MouseButton.LeftButton)

    assert np.allclose(
        np.asarray(layer.edge_color)[0],
        np.array([0.2, 0.8, 0.3, 1.0]),
        atol=1e-4,
    )
    assert not np.allclose(
        np.asarray(layer.edge_color)[1],
        np.array([0.2, 0.8, 0.3, 1.0]),
        atol=1e-4,
    )
    assert "#33cc4d" in group_color.styleSheet()

    qtbot.mouseClick(layer_color, Qt.MouseButton.LeftButton)

    assert np.allclose(
        np.asarray(layer.edge_color),
        np.tile(np.array([0.2, 0.8, 0.3, 1.0]), (2, 1)),
        atol=1e-4,
    )
    assert "#33cc4d" in layer_color.styleSheet()


def test_napari_entrypoint_does_not_receive_data_after_shutdown_control(monkeypatch):
    events, entrypoint_error = _run_fake_napari_entrypoint(
        monkeypatch,
        ready=True,
        shutdown_during_service=True,
    )

    assert entrypoint_error is None
    assert "server_process" in events
    assert "data_receive" not in events
    assert "data_process" not in events


def test_napari_entrypoint_bind_failure_quits_event_loop_and_stops_server(monkeypatch):
    events, entrypoint_error = _run_fake_napari_entrypoint(
        monkeypatch,
        start_error=RuntimeError("test bind failure"),
    )

    assert isinstance(entrypoint_error, RuntimeError)
    assert str(entrypoint_error) == "Napari Qt message service failed during startup."
    assert isinstance(entrypoint_error.__cause__, RuntimeError)
    assert str(entrypoint_error.__cause__) == "test bind failure"
    assert events.index("message_timer_start") < events.index("server_start")
    assert events.index("server_start") < events.index("message_timer_stop")
    assert events.index("message_timer_stop") < events.index("application_quit")
    assert "destroyed_connect" not in events
    assert events.count("server_stop") == 1
    assert events[-1] == "server_stop"


def test_napari_unknown_control_message_returns_typed_protocol_error():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    response = napari_viewer_server.NapariUnknownControlMessageAction().handle(
        object(),
        {ViewerControlResponseField.TYPE.value: "unsupported"},
    )

    assert response[ViewerControlResponseField.STATUS.value] == (
        ViewerProtocolStatus.ERROR.value
    )
    assert response[ViewerControlResponseField.TYPE.value] == "error"
    assert "unsupported" in response[ViewerControlResponseField.MESSAGE.value]


@pytest.mark.parametrize(
    "action_name",
    (
        "NapariGracefulShutdownControlMessageAction",
        "NapariForceShutdownControlMessageAction",
    ),
)
def test_napari_shutdown_cancels_pending_updates_before_scheduling_viewer_close(
    monkeypatch,
    action_name,
):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    from qtpy import QtCore

    events = []
    pending_timers = []

    class PendingTimer:
        def __init__(self, route_key):
            self.route_key = route_key
            self.active = True
            pending_timers.append(self)

        def stop(self):
            self.active = False
            events.append(f"stop:{self.route_key}")

        def fire(self):
            if self.active:
                events.append(f"display:{self.route_key}")

    class FakeQTimer:
        @staticmethod
        def singleShot(interval_ms, callback):
            events.append(f"schedule_close:{interval_ms}")
            FakeQTimer.close_callback = callback

    class FakeViewer:
        def close(self):
            events.append("viewer_close")

    server = napari_viewer_server.NapariViewerServer.__new__(
        napari_viewer_server.NapariViewerServer
    )
    server._running = True
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.viewer = FakeViewer()
    server.display_pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    for route_key in ("step-0", "step-1"):
        server.layer_route_state.set_pending_update(
            route_key,
            NapariPendingLayerUpdate.from_semantics(
                timer=PendingTimer(route_key),
                data_type=StreamingDataType.IMAGE,
                semantics=ViewerComponentAxisSemanticsAuthority.empty(),
            ),
        )

    monkeypatch.setattr(QtCore, "QTimer", FakeQTimer)
    action = getattr(napari_viewer_server, action_name)()
    response = action.handle(server, {})

    assert server.is_running() is False
    assert server.layer_route_state.layer_pending_updates == {}
    assert events == ["stop:step-0", "stop:step-1", "schedule_close:100"]
    assert response[ViewerControlResponseField.STATUS.value] == (
        ViewerProtocolStatus.SUCCESS.value
    )
    assert response[ViewerControlResponseField.TYPE.value] == "shutdown_ack"

    for timer in pending_timers:
        timer.fire()
    FakeQTimer.close_callback()

    assert events[-1] == "viewer_close"
    assert not any(event.startswith("display:") for event in events)


def test_napari_control_dispatch_registry_is_module_local_and_eager():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    registry = napari_viewer_server.NapariControlMessageAction.__registry__

    assert type(registry) is dict
    assert registry[ViewerControlMessageType.CLEAR_STATE.value] is (
        napari_viewer_server.NapariClearStateControlMessageAction
    )


def test_napari_axis_projector_drops_only_globally_singleton_axes():
    pytest.importorskip("openhcs.runtime.napari_viewer_server")
    component_values = {
        "site": [1, 2],
        "timepoint": [1],
        "channel": [1, 2, 3, 4, 5],
        "z_index": [1],
        "well": ["A01"],
    }

    projection = ViewerLayerAxisProjector().project(
        ViewerLayerAxisProjectionRequest.from_component_values(
            projected_axis_components=(
                "site",
                "timepoint",
                "channel",
                "z_index",
                "well",
            ),
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

    shapes = authority.create_or_update(
        layer_kind=NapariLayerKind.SHAPES,
        viewer=viewer,
        layers=layers,
        route_key="route-rois",
        layer_name="ROIs",
        data=[[[0, 0], [1, 1]]],
        layer_kwargs={
            "shape_type": ["polygon"],
            "properties": {"id": [1]},
            "edge_color": "red",
            "face_color": "transparent",
            "edge_width": 2,
        },
    )
    points = authority.create_or_update(
        layer_kind=NapariLayerKind.POINTS,
        viewer=viewer,
        layers=layers,
        route_key="route-spots",
        layer_name="Spots",
        data=[[0, 0]],
        layer_kwargs={
            "properties": {"id": [2]},
            "face_color": "green",
            "size": 3,
        },
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
        projection_key="NucleiObjects3D",
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

    assert title == "9. ConvertObjectsToImage NucleiObjects3D well A01 ROIs"


def test_napari_component_display_coordinator_splits_declared_image_layouts():
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
            self.scheduled.append((layer_key, data_type, component_axis_semantics))

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
        projection_key="main",
        step_name="OverlayOutlines",
        pipeline_position=4,
    )
    display_config = {
        "component_modes": {"well": "stack"},
        "component_order": ["well"],
    }
    component_axis_semantics = (
        ViewerComponentAxisSemanticsAuthority.from_display_config(
            ViewerMappingDisplayConfigInput(display_config),
            _component_value_domain({"well": ["A01"]}),
        )
    )

    for data, image_metadata in (
        (np.zeros((16, 16)), ImagePayloadMetadata()),
        (
            np.zeros((4, 16, 16, 3)),
            ImagePayloadMetadata(
                source_channel_axis=-1,
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        ),
    ):
        coordinator.display(
            data=data,
            stream_layer_context=napari_viewer_server.NapariStreamLayerContext(
                entries=component_axis_semantics.entries,
                layout=component_axis_semantics.layout,
                producer=producer,
                address=NapariStreamLayerAddress(
                    components={"well": "A01"},
                    path="/tmp/A01.tif",
                    stream_layer_data_type=StreamingDataType.IMAGE,
                ),
                image_metadata=image_metadata,
                plane_component_domain=ViewerComponentValueDomainPayload(()),
            ),
            server=server,
        )

    assert len(server.component_groups) == 2
    assert len({layer_key for layer_key, *_ in server.display_pipeline.scheduled}) == 2
    assert any("color_stack" in layer_key for layer_key in server.component_groups)
    assert sorted(server.layer_route_state.layer_titles.values()) == [
        "5. OverlayOutlines",
        "5. OverlayOutlines RGB stack",
    ]


def test_napari_component_display_coordinator_preserves_declared_singleton_stack():
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
            self.scheduled.append((layer_key, data_type, component_axis_semantics))

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
        projection_key="main",
        step_name="OverlayOutlines",
        pipeline_position=7,
    )
    component_axis_semantics = (
        ViewerComponentAxisSemanticsAuthority.from_display_config(
            ViewerMappingDisplayConfigInput(
                {
                    "component_modes": {"well": "stack", "channel": "stack"},
                    "component_order": ["well", "channel"],
                }
            ),
            _component_value_domain({"well": ["A01"], "channel": [2]}),
        )
    )

    coordinator.display(
        data=np.zeros((1, 16, 16, 3)),
        stream_layer_context=napari_viewer_server.NapariStreamLayerContext(
            entries=component_axis_semantics.entries,
            layout=component_axis_semantics.layout,
            producer=producer,
            address=NapariStreamLayerAddress(
                components={"well": "A01", "channel": 2},
                path="/tmp/A01.tif",
                stream_layer_data_type=StreamingDataType.IMAGE,
            ),
            image_metadata=ImagePayloadMetadata(
                source_channel_axis=-1,
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
            plane_component_domain=ViewerComponentValueDomainPayload(()),
        ),
        server=server,
    )

    route_key = next(iter(server.component_groups))
    assert route_key.endswith("_color_stack")
    assert tuple(server.component_groups.items_for(route_key)[0].data.shape) == (
        1,
        16,
        16,
        3,
    )
    assert list(server.layer_route_state.layer_titles.values()) == [
        "8. OverlayOutlines RGB stack"
    ]


def test_napari_layer_title_disambiguation_uses_display_step_number():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    producer = StreamProducerIdentity.pipeline_output(
        output_kind="main",
        output_key="main",
        projection_key="main",
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
    pending_update = NapariPendingLayerUpdate.from_semantics(
        timer=timer,
        data_type=StreamingDataType.IMAGE,
        semantics=ViewerComponentAxisSemanticsAuthority.empty(),
    )
    store.set_pending_update("nuclei", pending_update)

    assert store.has_layer("nuclei")
    assert store.layer("nuclei") is layer
    assert store.dimension_state_for("nuclei").labels == {"channel": ["Ch 1"]}
    assert store.cancel_pending_update("nuclei")
    assert timer.stopped
    assert store.pop_pending_update("nuclei") is pending_update
    assert store.dimension_state_for("missing").labels == {}


def test_napari_settle_rejects_recorded_layer_update_failure():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    route_key = "failed-route"

    class FailingProcessor:
        def add_items(self, **_kwargs):
            raise ValueError("invalid payload axis")

    class SuccessfulProcessor:
        def add_items(self, **_kwargs):
            return napari_viewer_server.NapariImmediateLayerDisplayWork(lambda: None)

    class BatchProcessors:
        def __init__(self, processor):
            self.processor = processor

        def get_or_create(self, **_kwargs):
            return self.processor

    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.component_groups = NapariComponentGroupStore()
    server.component_groups.items_for(route_key).append(
        _layer_item({"site": 1}, data=np.ones((4, 4), dtype=np.uint8))
    )
    server.batch_processors = BatchProcessors(FailingProcessor())
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    with pytest.raises(ValueError, match="invalid payload axis"):
        pipeline.execute_layer_update(
            route_key,
            StreamingDataType.IMAGE,
            ViewerComponentAxisSemanticsAuthority.empty(),
        )
    failed_progress = pipeline.settlement_progress()
    assert failed_progress.phase is ViewerSettlePhase.FAILED
    assert (
        server.layer_route_state.update_failure_message()
        == "failed-route: invalid payload axis"
    )

    server.batch_processors = BatchProcessors(SuccessfulProcessor())
    pipeline.execute_layer_update(
        route_key,
        StreamingDataType.IMAGE,
        ViewerComponentAxisSemanticsAuthority.empty(),
    )

    server.layer_route_state.reset_settlement()
    assert pipeline.settlement_progress() == ViewerSettleProgress.complete()


def test_napari_settlement_reports_incremental_qt_progress(monkeypatch):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    callbacks = []

    class DeferredQtTimer:
        @staticmethod
        def singleShot(_delay_ms, callback):
            callbacks.append(callback)

    class SuccessfulProcessor:
        def add_items(self, **_kwargs):
            return napari_viewer_server.NapariImmediateLayerDisplayWork(lambda: None)

    class BatchProcessors:
        def get_or_create(self, **_kwargs):
            return SuccessfulProcessor()

    server = _FakeNapariServer()
    server.viewer = object()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.component_groups = NapariComponentGroupStore()
    server.batch_processors = BatchProcessors()
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    server.display_pipeline = pipeline
    for route_key in ("first-route", "second-route"):
        server.component_groups.items_for(route_key).append(
            _layer_item({"site": 1}, data=np.ones((4, 4), dtype=np.uint8))
        )
        server.layer_route_state.set_pending_update(
            route_key,
            NapariPendingLayerUpdate.from_semantics(
                timer=_FakeTimer(),
                data_type=StreamingDataType.IMAGE,
                semantics=ViewerComponentAxisSemanticsAuthority.empty(),
            ),
        )

    monkeypatch.setattr(napari_viewer_server, "QTimer", DeferredQtTimer)
    action = napari_viewer_server.NapariSettleControlMessageAction()

    first_response = ViewerControlResponse(action.handle(server, {}))
    first_progress = ViewerSettleProgress.from_response(first_response)
    assert first_progress.phase is ViewerSettlePhase.RUNNING
    assert first_progress.completed_update_count == 0
    assert first_progress.total_update_count == 2
    assert first_progress.active_route == "first-route"
    assert len(callbacks) == 1

    callbacks.pop(0)()
    middle_progress = ViewerSettleProgress.from_response(
        ViewerControlResponse(action.handle(server, {}))
    )
    assert middle_progress.phase is ViewerSettlePhase.RUNNING
    assert middle_progress.completed_update_count == 1
    assert middle_progress.active_route == "second-route"
    assert len(callbacks) == 1

    callbacks.pop(0)()
    final_response = ViewerControlResponse(action.handle(server, {}))
    final_progress = ViewerSettleProgress.from_response(final_response)
    assert final_response.succeeded()
    assert final_progress == ViewerSettleProgress.complete(2)
    assert callbacks == []


def test_napari_settlement_reports_progress_within_one_large_route(monkeypatch):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    callbacks = []

    class DeferredQtTimer:
        @staticmethod
        def singleShot(_delay_ms, callback):
            callbacks.append(callback)

    class ThreeUnitWork(napari_viewer_server.NapariLayerDisplayWork):
        def __init__(self):
            self.completed_units = 0

        def advance(self):
            self.completed_units += 1
            return self.completed_units == 3

    class Processor:
        def __init__(self):
            self.work = ThreeUnitWork()

        def add_items(self, **_kwargs):
            return self.work

    class BatchProcessors:
        def __init__(self):
            self.processor = Processor()

        def get_or_create(self, **_kwargs):
            return self.processor

    route_key = "large-shapes-route"
    server = _FakeNapariServer()
    server.viewer = object()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.component_groups = NapariComponentGroupStore()
    server.component_groups.items_for(route_key).append(
        _layer_item({"site": 1}, data=np.ones((4, 4), dtype=np.uint8))
    )
    server.batch_processors = BatchProcessors()
    update = NapariPendingLayerUpdate.from_semantics(
        timer=_FakeTimer(),
        data_type=StreamingDataType.IMAGE,
        semantics=ViewerComponentAxisSemanticsAuthority.empty(),
    )
    server.layer_route_state.set_pending_update(route_key, update)
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    server.display_pipeline = pipeline
    monkeypatch.setattr(napari_viewer_server, "QTimer", DeferredQtTimer)
    action = napari_viewer_server.NapariSettleControlMessageAction()

    initial = ViewerSettleProgress.from_response(
        ViewerControlResponse(action.handle(server, {}))
    )
    assert initial.active_route == route_key
    assert initial.active_route_work_unit_count == 0

    callbacks.pop(0)()
    after_first = ViewerSettleProgress.from_response(
        ViewerControlResponse(action.handle(server, {}))
    )
    assert after_first.completed_update_count == 0
    assert after_first.active_route_work_unit_count == 1

    callbacks.pop(0)()
    after_second = ViewerSettleProgress.from_response(
        ViewerControlResponse(action.handle(server, {}))
    )
    assert after_second.completed_update_count == 0
    assert after_second.active_route_work_unit_count == 2

    callbacks.pop(0)()
    final = ViewerSettleProgress.from_response(
        ViewerControlResponse(action.handle(server, {}))
    )
    assert final == ViewerSettleProgress.complete(1)
    assert callbacks == []


def test_napari_scheduled_update_retains_failure_without_escaping_qt_callback():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    route_key = "failed-scheduled-route"

    class BatchProcessors:
        def get_or_create(self, **_kwargs):
            raise ValueError("invalid scheduled processor route")

    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.component_groups = NapariComponentGroupStore()
    server.component_groups.items_for(route_key).append(
        _layer_item({"site": 1}, data=np.ones((4, 4), dtype=np.uint8))
    )
    server.batch_processors = BatchProcessors()
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)

    update = NapariPendingLayerUpdate.from_semantics(
        timer=_FakeTimer(),
        data_type=StreamingDataType.IMAGE,
        semantics=ViewerComponentAxisSemanticsAuthority.empty(),
    )
    server.layer_route_state.set_pending_update(route_key, update)
    pipeline.execute_scheduled_layer_update(route_key, update)

    failed_progress = pipeline.settlement_progress()
    assert failed_progress.phase is ViewerSettlePhase.FAILED
    assert (
        server.layer_route_state.update_failure_message()
        == "failed-scheduled-route: invalid scheduled processor route"
    )


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


def test_napari_batch_processor_returns_handler_owned_display_work():
    from polystore.streaming.receivers.napari import NapariBatchProcessor

    sentinel = object()

    class Server:
        def display_layer_batch(self, **_kwargs):
            return sentinel

    processor = NapariBatchProcessor(Server())

    assert (
        processor.add_items(
            layer_key="objects",
            items=(),
            display_payload=ViewerComponentAxisSemanticsAuthority.empty(),
            component_names_metadata=ViewerComponentNameMetadata.empty(),
        )
        is sentinel
    )


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


def test_napari_image_stack_builder_rejects_missing_projected_coordinates():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    with pytest.raises(ValueError, match="missing routed image"):
        napari_viewer_server._build_nd_image_array(
            [
                _layer_item({"site": 1}, data=np.ones((2, 2))),
            ],
            _axis_projection(["site"], {"site": [1, 2]}),
        )


def test_napari_image_stack_builder_rejects_wrong_collapsed_component_value():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    with pytest.raises(ValueError, match="collapsed component 'channel'"):
        napari_viewer_server._build_nd_image_array(
            [
                _layer_item({"site": 1, "channel": 3}, data=np.ones((2, 2))),
            ],
            _axis_projection(
                ["site"],
                {"site": [1]},
                scalar_component_values={"channel": [4]},
            ),
        )


def test_napari_image_stack_builder_preserves_route_local_collapsed_channels():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    image = napari_viewer_server._build_nd_image_array(
        [
            _layer_item({"site": 1, "channel": 4}, data=np.ones((2, 2))),
            _layer_item({"site": 2, "channel": 4}, data=np.full((2, 2), 2)),
        ],
        _axis_projection(
            ["site"],
            {"site": [1, 2]},
            scalar_component_values={"channel": [4]},
        ),
    )

    assert image.shape == (2, 2, 2)
    assert np.all(image[0] == 1)
    assert np.all(image[1] == 2)


def test_napari_image_stack_builder_consumes_aggregate_payload_axis_as_stack_component():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    image = napari_viewer_server._build_nd_image_array(
        [
            _layer_item(
                {"channel": 1},
                data=np.stack(
                    [
                        np.full((2, 2), 3, dtype=np.uint16),
                        np.full((2, 2), 7, dtype=np.uint16),
                    ]
                ),
            )
        ],
        _axis_projection(
            ["z_index"],
            {"z_index": [1, 2]},
            scalar_component_values={"channel": [1]},
        ),
        NapariAggregateAxisBindingSet(
            (NapariAggregateAxisBinding("z_index", 0, (1, 2)),)
        ),
    )

    assert image.shape == (2, 2, 2)
    assert np.all(image[0] == 3)
    assert np.all(image[1] == 7)


def test_napari_image_stack_builder_rejects_unrouted_axis_values():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")

    with pytest.raises(ValueError, match="missing routed image"):
        napari_viewer_server._build_nd_image_array(
            [
                _layer_item({"channel": 1}, data=np.ones((2, 2))),
                _layer_item({"channel": 3}, data=np.full((2, 2), 3)),
            ],
            _axis_projection(
                ["channel"],
                {"channel": [1, 2, 3]},
            ),
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

    assert tracker.values_for(("main", ("channel", "well")), ["channel", "well"]) == {
        "channel": [2, 4],
        "well": ["A01", "B02"],
    }
    assert tracker.values_for(
        ("artifact", ("channel", "well")),
        ["channel", "well"],
    ) == {
        "channel": [1],
        "well": ["A01"],
    }
    assert tracker.values_for(("main", ("site",)), ["site"]) == {"site": []}


def test_napari_display_axis_domain_tracks_shared_active_axis_values():
    domain = ViewerDisplayAxisDomain()
    projected_axis_components = ["site", "timepoint", "channel", "z_index", "well"]

    domain.record_display_axis_values(
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
    domain.record_display_axis_values(
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

    assert domain.display_axis_values_for(projected_axis_components) == {
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
        [_layer_item({"z_index": value}) for value in [1, 10, 100, 11, 2, 20, 3, 9]],
    )

    assert tracker.values_for(("main", ("z_index",)), ["z_index"]) == {
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

    assert tracker.values_for(
        ("nuclei", ("well", "channel")),
        ["well", "channel"],
    ) == {
        "well": ["A01"],
        "channel": [1, "D"],
    }


def test_viewer_component_name_metadata_owns_channel_well_and_generic_labels():
    metadata = _component_name_metadata(
        {
            "channel": {"1": "DAPI", "2": "None"},
            "well": {"A01": "A01"},
            "site": {"3": "Field"},
        },
        context="test",
    )

    assert metadata.axis_labels("channel", [1, 2]) == ["Ch1: DAPI", "Ch 2"]
    assert metadata.axis_labels("well", ["A01"]) == ["A01"]
    assert metadata.axis_labels("site", [3]) == ["Site 3: Field"]


def test_napari_shape_layer_payload_builds_native_nd_rois_from_plane_metadata():
    payload = NapariShapeLayerPayload.build(
        layer_items=[
            _layer_item(
                {"channel": 4},
                [
                    {
                        "type": "polygon",
                        "coordinates": [[0, 0], [0, 2], [2, 2], [2, 0]],
                        "metadata": {
                            "label": 7,
                            "plane_indices": (0,),
                            "plane_shape": (2,),
                        },
                    },
                    {
                        "type": "path",
                        "coordinates": [[0, 1], [1, 1], [2, 1]],
                        "metadata": {
                            "label": 7,
                            "plane_indices": (1,),
                            "plane_shape": (2,),
                        },
                    },
                ],
                stream_layer_data_type=StreamingDataType.SHAPES,
            )
        ],
        axis_projection=_axis_projection(
            ["z_index"],
            {"z_index": [1, 2]},
            scalar_component_values={"channel": [4]},
        ),
        aggregate_axis_bindings=NapariAggregateAxisBindingSet(
            (NapariAggregateAxisBinding("z_index", 0, (1, 2)),)
        ),
    )

    assert payload.ndim == 3
    assert payload.shape_types == ["polygon", "path"]
    assert np.all(payload.data[0][:, 0] == 0)
    assert np.all(payload.data[1][:, 0] == 1)
    assert np.array_equal(payload.data[1][:, 1:], [[0, 1], [1, 1], [2, 1]])
    assert payload.features["label"] == [7, 7]
    assert payload.features["path"] == ["test", "test"]


def test_napari_shape_payload_hides_framework_subject_columns_in_layer_metadata():
    payload = NapariShapeLayerPayload.build(
        layer_items=[
            _layer_item(
                {},
                [
                    {
                        "type": "path",
                        "coordinates": [[0, 0], [1, 1]],
                        "metadata": {
                            "label": 3,
                            "edge_id": 10,
                            ObjectArtifactSubjectBinding.SUBJECT_FEATURE: "neurons",
                            ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE: 3,
                        },
                    },
                    {
                        "type": "path",
                        "coordinates": [[1, 1], [2, 2]],
                        "metadata": {
                            "label": 3,
                            "edge_id": 11,
                            ObjectArtifactSubjectBinding.SUBJECT_FEATURE: "neurons",
                            ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE: 3,
                        },
                    },
                ],
                stream_layer_data_type=StreamingDataType.SHAPES,
            )
        ],
        axis_projection=_axis_projection([], {}),
    )

    assert ObjectArtifactSubjectBinding.SUBJECT_FEATURE not in payload.features
    assert ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE not in payload.features
    assert payload.features["edge_id"] == [10, 11]
    assert payload.result_metadata == {
        ObjectArtifactSubjectBinding.SUBJECT_FEATURE: "neurons",
        ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE: (3, 3),
    }


def test_napari_shape_layer_payload_accepts_registered_native_ellipse_kind():
    payload = NapariShapeLayerPayload.build(
        layer_items=[
            _layer_item(
                {},
                [
                    {
                        "type": "ellipse",
                        "coordinates": [
                            [7.0, 15.0],
                            [7.0, 25.0],
                            [13.0, 25.0],
                            [13.0, 15.0],
                        ],
                        "metadata": {"label": 7},
                    }
                ],
                stream_layer_data_type=StreamingDataType.SHAPES,
            )
        ],
        axis_projection=_axis_projection([], {}),
    )

    assert payload.shape_types == ["ellipse"]
    assert np.array_equal(
        payload.data[0],
        np.array(
            [
                [7.0, 15.0],
                [7.0, 25.0],
                [13.0, 25.0],
                [13.0, 15.0],
            ]
        ),
    )
    assert payload.features["label"] == [7]


def test_napari_shape_layer_payload_assigns_distinct_stable_label_colors():
    payload = NapariShapeLayerPayload(
        data=[],
        shape_types=[],
        features={"label": [1, 2, 1, 3]},
        ndim=2,
    )

    assert len(payload.label_color_cycle) == 3
    assert len(set(payload.label_color_cycle)) == 3


def test_napari_shape_layer_payload_chunks_by_member_and_vertex_limits():
    payload = NapariShapeLayerPayload(
        data=[
            np.zeros((3, 2)),
            np.zeros((4, 2)),
            np.zeros((8, 2)),
            np.zeros((2, 2)),
        ],
        shape_types=["polygon", "polygon", "path", "path"],
        features={"label": [1, 1, 2, 3], "path": ["a", "b", "c", "d"]},
        ndim=2,
    )

    chunks = payload.chunks(max_shape_count=3, max_vertex_count=10)

    assert [len(chunk.data) for chunk in chunks] == [2, 2]
    assert [chunk.shape_types for chunk in chunks] == [
        ["polygon", "polygon"],
        ["path", "path"],
    ]
    assert [chunk.features["path"] for chunk in chunks] == [
        ["a", "b"],
        ["c", "d"],
    ]
    assert payload.label_colors[0] == payload.label_colors[1]
    assert payload.label_colors[1] != payload.label_colors[2]
    assert (
        payload.colors_for_labels(chunks[1].features["label"])
        == (payload.label_colors[2:])
    )


def test_napari_aggregate_axis_binding_uses_declared_component_not_equal_extent():
    component_value_domain = _component_value_domain(
        {"site": [1, 2], "channel": [1, 2]}
    )
    component_axis_semantics = (
        ViewerComponentAxisSemanticsAuthority.from_display_config(
            ViewerMappingDisplayConfigInput(
                {
                    "component_modes": {"site": "stack", "channel": "stack"},
                    "component_order": ["site", "channel"],
                }
            ),
            component_value_domain,
        )
    )
    items = [
        _layer_item(
            {"site": 1, "channel": 1},
            np.zeros((2, 3, 3), dtype=np.float32),
            image_metadata=ImagePayloadMetadata(
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
            plane_component_domain=_component_value_domain({"channel": [1, 2]}),
        )
    ]

    bindings = NapariAggregateAxisBindingAuthority.bindings(
        items,
        component_axis_semantics,
    )

    assert bindings == NapariAggregateAxisBindingSet(
        (NapariAggregateAxisBinding("channel", 0, (1, 2)),)
    )


def test_napari_aggregate_axis_binding_preserves_singleton_scalar_identity():
    item = _layer_item(
        {"channel": 2},
        np.zeros((1, 3, 3), dtype=np.float32),
    )
    bindings = NapariAggregateAxisBindingSet(
        (
            NapariAggregateAxisBinding("site", 0, (1,)),
            NapariAggregateAxisBinding("z_index", 1, (1, 2)),
        )
    )

    assert bindings.item_scalar_components(item) == {
        "channel": 2,
        "site": 1,
    }


def test_napari_shape_plane_metadata_requires_declared_component_axis():
    component_value_domain = _component_value_domain(
        {"site": [1, 2], "channel": [1, 2]}
    )
    component_axis_semantics = (
        ViewerComponentAxisSemanticsAuthority.from_display_config(
            ViewerMappingDisplayConfigInput(
                {
                    "component_modes": {"site": "stack", "channel": "stack"},
                    "component_order": ["site", "channel"],
                }
            ),
            component_value_domain,
        )
    )

    with pytest.raises(ValueError, match="plane_component_values"):
        NapariAggregateAxisBindingAuthority.bindings(
            [
                _layer_item(
                    {"site": 1, "channel": 1},
                    [
                        {
                            "type": "polygon",
                            "coordinates": [[0, 0], [0, 2], [2, 2], [2, 0]],
                            "metadata": {
                                "source_spatial_shape_yx": (3, 3),
                                "plane_indices": (0,),
                                "plane_shape": (2,),
                            },
                        }
                    ],
                    stream_layer_data_type=StreamingDataType.SHAPES,
                )
            ],
            component_axis_semantics,
        )


def test_napari_shape_label_rasterizer_rejects_mixed_aggregate_plane_metadata():
    component_value_domain = _component_value_domain(
        {"channel": [1], "z_index": [1, 2]}
    )
    component_axis_semantics = (
        ViewerComponentAxisSemanticsAuthority.from_display_config(
            ViewerMappingDisplayConfigInput(
                {
                    "component_modes": {"channel": "stack", "z_index": "stack"},
                    "component_order": ["z_index", "channel"],
                }
            ),
            component_value_domain,
        )
    )

    with pytest.raises(ValueError, match="mixes plane-indexed and unindexed"):
        NapariAggregateAxisBindingAuthority.bindings(
            [
                _layer_item(
                    {"channel": 1},
                    [
                        {
                            "type": "polygon",
                            "coordinates": [[0, 0], [0, 2], [2, 2], [2, 0]],
                            "metadata": {
                                "source_spatial_shape_yx": (3, 3),
                                "plane_indices": (0,),
                                "plane_shape": (2,),
                            },
                        },
                        {
                            "type": "path",
                            "coordinates": [[0, 1], [1, 1], [2, 1]],
                            "metadata": {"source_spatial_shape_yx": (3, 3)},
                        },
                    ],
                    stream_layer_data_type=StreamingDataType.SHAPES,
                )
            ],
            component_axis_semantics,
        )


def test_napari_shapes_layer_display_applies_route_global_axis_translate():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.layer_route_state.set_title("objects", "Objects")
    server.viewer = _FakeViewer()
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    presentation = _axis_presentation(
        layer_key="objects",
        projected_axis_components=("channel", "site"),
        component_values={"channel": [4], "site": [1, 2]},
        axis_offsets=(3, 0),
    )

    napari_viewer_server.NapariShapesLayerDisplayHandler().handle(
        napari_viewer_server.NapariLayerDisplayRequest(
            pipeline=pipeline,
            presentation=presentation,
            items=[
                _layer_item(
                    {"channel": 4, "site": 1},
                    [
                        {
                            "type": "polygon",
                            "coordinates": [[0, 0], [0, 2], [2, 2], [2, 0]],
                            "metadata": {"source_spatial_shape_yx": (3, 3)},
                        }
                    ],
                    stream_layer_data_type=StreamingDataType.SHAPES,
                )
            ],
        )
    )

    layer_type, data, name, layer_kwargs = server.viewer.calls[-1]
    assert layer_type == "shapes"
    assert name == "Objects"
    assert len(data) == 1
    assert data[0].shape == (4, 4)
    assert tuple(data[0][0]) == (0, 0, 0, 0)
    assert tuple(data[0][-1]) == (0, 0, 2, 0)
    assert layer_kwargs["shape_type"] == ["polygon"]
    assert layer_kwargs["features"] == {
        "source_spatial_shape_yx": ["(3, 3)"],
        "label": [1],
        "path": ["test"],
    }
    assert layer_kwargs["ndim"] == 4
    assert layer_kwargs["edge_color"] == "label"
    assert len(layer_kwargs["edge_color_cycle"]) == 1
    assert layer_kwargs["face_color"] == "label"
    assert layer_kwargs["face_color_cycle"] == layer_kwargs["edge_color_cycle"]
    assert layer_kwargs["axis_labels"] == ("channel", "site", "y", "x")
    assert layer_kwargs["translate"] == (3.0, 0.0, 0.0, 0.0)


def test_napari_shapes_display_work_appends_bounded_chunks(monkeypatch):
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    events = []

    class FakeShapesLayer:
        def __init__(self, data, name, kwargs):
            self.name = name
            self.data = list(data)
            self._features = kwargs["features"]
            self._visible = kwargs["visible"]
            self.edge_color = kwargs["edge_color"]
            self.face_color = kwargs["face_color"]
            self.edge_color_mode = "cycle"
            self.face_color_mode = "cycle"
            self.edge_color_cycle = kwargs["edge_color_cycle"]
            self.face_color_cycle = kwargs["face_color_cycle"]
            self.add_calls = []
            events.append(("visible", self._visible))

        @property
        def features(self):
            return self._features

        @features.setter
        def features(self, features):
            self._features = features
            events.append(("features", features))

        @property
        def visible(self):
            return self._visible

        @visible.setter
        def visible(self, visible):
            self._visible = visible
            events.append(("visible", visible))

        def add(self, data, **kwargs):
            self.data.extend(data)
            self.add_calls.append((list(data), kwargs))

    class ChunkViewer(_FakeViewer):
        def add_shapes(self, data, *, name, **kwargs):
            layer = FakeShapesLayer(data, name, kwargs)
            self.layers.append(layer)
            self.calls.append(("shapes", data, name, kwargs))
            return layer

    monkeypatch.setattr(
        napari_viewer_server.NapariShapesLayerDisplayHandler,
        "MAX_SHAPES_PER_WORK_UNIT",
        2,
    )
    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.layer_route_state.set_title("objects", "Objects")
    server.viewer = ChunkViewer()
    server.bind_result_selection_layer = lambda layer: events.append(("bind", layer))
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    shapes = [
        {
            "type": "polygon",
            "coordinates": [[0, 0], [0, 1], [1, 0]],
            "metadata": {"label": index % 2 + 1},
        }
        for index in range(5)
    ]
    work = napari_viewer_server.NapariShapesLayerDisplayHandler().display_work(
        napari_viewer_server.NapariLayerDisplayRequest(
            pipeline=pipeline,
            presentation=_axis_presentation(
                layer_key="objects",
                projected_axis_components=(),
            ),
            items=[
                _layer_item(
                    {},
                    shapes,
                    stream_layer_data_type=StreamingDataType.SHAPES,
                )
            ],
        )
    )

    assert not work.advance()
    layer = server.viewer.layers[-1]
    assert len(layer.data) == 2
    assert layer.visible is False
    assert not work.advance()
    assert len(layer.data) == 4
    assert work.advance()

    assert len(server.viewer.calls) == 1
    assert [len(data) for data, _kwargs in layer.add_calls] == [2, 1]
    assert layer.add_calls[0][1]["edge_color"] == work.payload.label_colors[2:4]
    assert layer.add_calls[1][1]["edge_color"] == work.payload.label_colors[4:]
    assert len(layer.data) == 5
    assert layer.features["label"] == [1, 2, 1, 2, 1]
    assert layer.edge_color == "label"
    assert layer.face_color == "label"
    assert layer.visible is True
    final_features_event = next(
        index
        for index, event in enumerate(events)
        if event == ("features", work.payload.features)
    )
    bind_event = next(
        index for index, event in enumerate(events) if event[0] == "bind"
    )
    reveal_event = events.index(("visible", True))
    assert final_features_event < bind_event < reveal_event


def test_napari_shapes_display_work_batches_thousands_with_native_features():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    from napari.components import ViewerModel

    member_count = 4_097
    labels = [index % 17 + 1 for index in range(member_count)]
    shapes = [
        {
            "type": "polygon",
            "coordinates": [
                [index * 2, 0],
                [index * 2, 1],
                [index * 2 + 1, 0],
            ],
            "metadata": {"label": labels[index], "area": 0.5},
        }
        for index in range(member_count)
    ]
    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.layer_route_state.set_title("objects", "Objects")
    server.viewer = ViewerModel()
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    work = napari_viewer_server.NapariShapesLayerDisplayHandler().display_work(
        napari_viewer_server.NapariLayerDisplayRequest(
            pipeline=pipeline,
            presentation=_axis_presentation(
                layer_key="objects",
                projected_axis_components=(),
            ),
            items=[
                _layer_item(
                    {},
                    shapes,
                    stream_layer_data_type=StreamingDataType.SHAPES,
                )
            ],
        )
    )

    work_unit_count = 1
    while not work.advance():
        work_unit_count += 1

    layer = server.layer_route_state.layer("objects")
    assert work_unit_count == 3
    assert len(layer.data) == member_count
    assert layer.features["label"].tolist() == labels
    assert layer.features["area"].tolist() == [0.5] * member_count
    assert layer.edge_color_mode == "cycle"
    assert layer.face_color_mode == "cycle"
    label_colors = {
        label: tuple(layer.edge_color[index]) for index, label in enumerate(labels)
    }
    assert all(
        tuple(layer.edge_color[index]) == label_colors[label]
        for index, label in enumerate(labels)
    )
    layer.selected_data = {member_count - 1}
    assert layer.selected_data == {member_count - 1}


def test_napari_points_layer_display_applies_route_global_axis_translate():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    server = _FakeNapariServer()
    server.layer_route_state = NapariLayerRouteStateStore.empty()
    server.layer_route_state.set_title("spots", "Spots")
    server.viewer = _FakeViewer()
    pipeline = napari_viewer_server.NapariLayerDisplayPipeline(server)
    presentation = _axis_presentation(
        layer_key="spots",
        projected_axis_components=("channel",),
        component_values={"channel": [4]},
        axis_offsets=(3,),
    )

    napari_viewer_server.NapariPointsLayerDisplayHandler().handle(
        napari_viewer_server.NapariLayerDisplayRequest(
            pipeline=pipeline,
            presentation=presentation,
            items=[
                _layer_item(
                    {"channel": 4},
                    [
                        {
                            "type": "points",
                            "coordinates": [[1, 2]],
                            "metadata": {"label": 7, "component": 4},
                        }
                    ],
                    stream_layer_data_type=StreamingDataType.POINTS,
                )
            ],
        )
    )

    layer_type, data, name, layer_kwargs = server.viewer.calls[-1]
    assert layer_type == "points"
    assert name == "Spots"
    assert data.shape == (1, 3)
    assert tuple(data[0]) == (0, 1, 2)
    assert layer_kwargs["axis_labels"] == ("channel", "y", "x")
    assert layer_kwargs["translate"] == (3.0, 0.0, 0.0)
    assert layer_kwargs["properties"] == {"label": [7], "component": [4]}
