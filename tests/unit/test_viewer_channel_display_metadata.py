from types import SimpleNamespace

import pytest
from polystore.virtual_workspace import SourcePixelRef
from zmqruntime.viewer_protocol import ViewerComponentMetadataPayload

from openhcs.constants.constants import AllComponents
from openhcs.core.source_binding_workspace import PrimaryPlaneBindingProjection
from openhcs.core.source_bindings import ComponentSelector, NamedSourceBinding
from openhcs.core.source_projection import (
    OpenHCSPlaneAddress,
    SourceCandidate,
    SourceProjectionSet,
)
from openhcs.core.steps.stream_component_semantics import (
    StreamComponentMessageExtraPayload,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser
from openhcs.runtime.napari_streaming_handlers import (
    NapariAxisPresentation,
    NapariLayerRouteStateStore,
)
from openhcs.runtime.viewer_component_system import (
    ViewerComponentAxisSemanticsAuthority,
    ViewerComponentNameMetadata,
    ViewerLayerAxisProjection,
)


def _address(channel: str = "1") -> OpenHCSPlaneAddress:
    return OpenHCSPlaneAddress(
        well="A01",
        site="1",
        channel=channel,
        z_index="1",
        timepoint="1",
    )


def _component_name_metadata() -> ViewerComponentNameMetadata:
    wire_payload = StreamComponentMessageExtraPayload(
        component_names_metadata={"channel": {"1": "DNA"}},
        component_value_domain={"channel": [1]},
    ).to_wire_mapping()
    round_trip = ViewerComponentMetadataPayload.from_wire_mapping(wire_payload)
    return ViewerComponentNameMetadata.from_wire_mapping(
        round_trip.component_names_metadata,
        context="source-binding channel labels",
    )


def test_primary_plane_projection_preserves_exact_binding_alias_as_channel_label():
    address = _address()
    candidate = SourceCandidate(
        source_ref=SourcePixelRef("disk", "A01_dna.tif"),
        relative_path="A01_dna.tif",
        metadata={"channel": "1"},
        component_labels={"channel": "physical-channel-name"},
    )
    projection = PrimaryPlaneBindingProjection().projection(
        NamedSourceBinding(
            alias="DNA",
            component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
        ),
        candidate,
        address,
    )

    assert projection.address.channel == "1"
    assert dict(projection.component_labels) == {"channel": "DNA"}
    metadata = SourceProjectionSet((projection,)).metadata_dict(
        parser=SourceSchemaFilenameParser(),
        microscope_handler_name="source_bindings",
        source_filename_parser_name="SourceSchemaFilenameParser",
        grid_dimensions=[1, 1],
        pixel_size=1.0,
    )
    assert metadata["channels"] == {"1": "DNA"}

    projection_without_declared_channel = PrimaryPlaneBindingProjection().projection(
        NamedSourceBinding(alias="RGB"),
        candidate,
        address,
    )
    assert dict(projection_without_declared_channel.component_labels) == {
        "channel": "physical-channel-name"
    }


def test_channel_display_metadata_round_trips_without_replacing_numeric_coordinate():
    names = _component_name_metadata()

    assert names.to_wire_mapping() == {"channel": {"1": "DNA"}}
    assert names.display_name("channel", 1) == "DNA"
    assert names.axis_label("channel", 1) == "Ch1: DNA"


def test_napari_viewer_state_keeps_numeric_channel_and_declared_display_label():
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    semantics = ViewerComponentAxisSemanticsAuthority.empty()
    route_key = "source-dna"
    server = SimpleNamespace(
        component_name_metadata=_component_name_metadata(),
        layer_route_state=NapariLayerRouteStateStore.empty(),
    )
    presentation = NapariAxisPresentation(
        entries=semantics.entries,
        layout=semantics.layout,
        route_key=route_key,
        projection=ViewerLayerAxisProjection(
            projected_axis_components=("site",),
            component_values={"site": [1, 2]},
            routed_component_values={"site": [1, 2]},
            axis_offsets=(0,),
            scalar_component_values={"channel": [1]},
        ),
        payload_axis_labels=(),
    )

    napari_viewer_server.NapariDimensionLabelStore(server).apply(presentation)

    state = server.layer_route_state.dimension_state_for(route_key)
    assert state.presentation is not None
    assert state.presentation.projection.scalar_component_values["channel"] == [1]
    assert state.scalar_labels == ("Ch1: DNA",)
