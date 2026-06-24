from __future__ import annotations

import numpy as np

from openhcs.core.config import FijiDisplayConfig, FijiLUT
from openhcs.runtime.fiji_viewer_server import (
    FijiBatchWireParser,
    FijiControlMessageAuthority,
    FijiDimensionAxis,
    FijiDisplayConfigWireAdapter,
    FijiImagePayload,
    FijiImageIntensityRange,
    FijiPlaneGeometry,
    FijiStackSliceLabelBuilder,
    FijiHyperstackCoordinates,
    FijiWindowItemProjection,
    FijiWindowRegistry,
    FijiWireItem,
)
from openhcs.runtime.viewer_protocol import ViewerControlMessageType
from openhcs.runtime.viewer_component_system import (
    ViewerComponentAxisSemanticsAuthority,
    ViewerComponentValueDomainPayload,
    ViewerObjectDisplayConfigInput,
)


PRODUCER_IDENTITY = {
    "origin": "pipeline",
    "output_kind": "main",
    "output_key": "Nuclei",
    "step_name": "IdentifyPrimaryObjects",
    "pipeline_position": 0,
    "step_scope_id": None,
    "invocation_key": None,
    "artifact_kind": None,
}


def _component_value_domain(payload: dict) -> ViewerComponentValueDomainPayload:
    return ViewerComponentValueDomainPayload.from_wire_mapping(
        payload,
        context="test Fiji component value domain",
    )


class _FakeWindow:
    def __init__(self, image_plus: "_FakeImagePlus") -> None:
        self._image_plus = image_plus

    def isVisible(self) -> bool:
        return self._image_plus.visible


class _FakeImagePlus:
    def __init__(self) -> None:
        self.closed = False
        self.visible = True

    def getWindow(self) -> _FakeWindow | None:
        if self.closed:
            return None
        return _FakeWindow(self)

    def close(self) -> None:
        self.closed = True
        self.visible = False


def test_fiji_display_config_wire_adapter_rehydrates_existing_config_type() -> None:
    default_config = FijiDisplayConfig()
    config = FijiDisplayConfigWireAdapter.from_payload(
        {
            "component_modes": default_config.component_modes(),
            "component_order": list(default_config.COMPONENT_ORDER),
            "lut": "Fire",
            "auto_contrast": False,
        }
    ).to_config()

    assert isinstance(config, FijiDisplayConfig)
    assert config.lut is FijiLUT.FIRE
    assert config.auto_contrast is False
    assert config.component_modes() == default_config.component_modes()


def test_fiji_window_registry_owns_window_state_and_group_identity() -> None:
    registry = FijiWindowRegistry()
    image_plus = _FakeImagePlus()
    image = FijiImagePayload(
        data=np.zeros((2, 2), dtype=np.uint8),
        metadata={"well": "A14"},
        image_id="first",
    )

    registry.store_hyperstack("A14", image_plus, [image])
    registry.store_fixed_labels("A14", [("well", "A14")])

    assert registry.open_hyperstack("A14") is image_plus
    assert registry.images("A14") == [image]
    assert registry.fixed_labels("A14") == (("well", "A14"),)
    assert registry.group_id("A14") == registry.group_id("A14")

    image_plus.visible = False

    assert registry.open_hyperstack("A14") is None
    assert registry.fixed_labels("A14") == ()


def test_fiji_window_registry_closes_replaced_hyperstack() -> None:
    registry = FijiWindowRegistry()
    old_image_plus = _FakeImagePlus()
    new_image_plus = _FakeImagePlus()

    registry.store_hyperstack("A14", old_image_plus, [])
    registry.store_hyperstack("A14", new_image_plus, [])

    assert old_image_plus.closed is True
    assert registry.open_hyperstack("A14") is new_image_plus


def test_fiji_state_control_message_fails_loudly_until_projector_exists() -> None:
    response = FijiControlMessageAuthority(FijiWindowRegistry()).response_for(
        {"type": ViewerControlMessageType.STATE.value}
    )

    wire_response = response.to_wire_mapping()

    assert wire_response["status"] == "error"
    assert wire_response["type"] == "state_ack"
    assert (
        "Fiji live viewer state polling is not implemented"
        in wire_response["message"]
    )


def test_fiji_batch_message_normalizes_wire_items() -> None:
    default_config = FijiDisplayConfig()
    batch = FijiBatchWireParser(
        {
            "type": "batch",
            "images": [
                {
                    "data_type": "image",
                    "image_id": "img-1",
                    "metadata": {"channel": 1},
                    "data": np.zeros((2, 2), dtype=np.uint8),
                }
            ],
            "display_config": {
                "component_modes": default_config.component_modes(),
                "component_order": list(default_config.COMPONENT_ORDER),
                "lut": "Grays",
                "auto_contrast": True,
            },
            "images_dir": None,
            "component_names_metadata": {"channel": {"1": "DNA"}},
            "component_value_domain": {"channel": [1]},
        }
    ).batch_message()

    assert isinstance(batch.items[0], FijiWireItem)
    image_payload = batch.items[0].image_payload()
    assert image_payload is not None
    assert image_payload.data is batch.items[0].data
    assert image_payload.metadata == {"channel": 1}
    assert image_payload.image_id == "img-1"
    assert batch.store.display_name("channel", 1) == "DNA"
    assert batch.to_wire_mapping() == {"channel": [1]}


def test_fiji_window_item_projection_preserves_nominal_items() -> None:
    items = [
        FijiWireItem.from_payload(
                {
                    "data_type": "image",
                    "producer_identity": PRODUCER_IDENTITY,
                    "metadata": {
                        "well": "A14",
                        "site": 1,
                    "channel": 1,
                    "z_index": 0,
                    "timepoint": 0,
                },
            }
        ),
        FijiWireItem.from_payload(
                {
                    "data_type": "image",
                    "producer_identity": PRODUCER_IDENTITY,
                    "metadata": {
                        "well": "A14",
                        "site": 1,
                    "channel": 2,
                    "z_index": 0,
                    "timepoint": 0,
                },
            }
        ),
    ]

    projection = FijiWindowItemProjection.from_items(
        items,
        ViewerComponentAxisSemanticsAuthority.from_display_config(
            ViewerObjectDisplayConfigInput(FijiDisplayConfig()),
            _component_value_domain(
                {
                    "well": ["A14"],
                    "site": [1],
                    "channel": [1, 2],
                    "z_index": [0],
                    "timepoint": [0],
                }
            ),
        ),
    )
    window_items = next(iter(projection.windows.values()))

    assert window_items == items
    assert projection.coordinate_components.channel == ["channel"]
    assert projection.coordinate_components.z_axis_components == ["z_index"]
    assert projection.coordinate_components.frame == ["well", "site", "timepoint"]


def test_fiji_plane_geometry_owns_extraction_padding_and_shape() -> None:
    volume = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    plane = FijiPlaneGeometry.extract_2d_plane(volume)
    padded = FijiPlaneGeometry.pad_to_shape(plane, 4, 5)

    assert np.array_equal(plane, volume[1])
    assert padded.shape == (4, 5)
    assert np.array_equal(padded[:3, :3], plane)

    images = [
        FijiImagePayload(data=np.zeros((2, 3), dtype=np.uint8), metadata={}),
        FijiImagePayload(data=np.zeros((4, 1), dtype=np.uint8), metadata={}),
    ]

    assert FijiPlaneGeometry.target_spatial_shape(images) == (4, 3)


def test_fiji_image_intensity_range_uses_extracted_planes() -> None:
    images = [
        FijiImagePayload(
            data=np.array([[[1, 2]], [[9, 10]]], dtype=np.uint8),
            metadata={},
        ),
        FijiImagePayload(
            data=np.array([[3, 4]], dtype=np.uint8),
            metadata={},
        ),
    ]

    intensity_range = FijiImageIntensityRange.from_images(images)

    assert intensity_range.minimum == 3
    assert intensity_range.maximum == 10


def test_fiji_stack_slice_label_builder_uses_coordinate_axes() -> None:
    coordinates = FijiHyperstackCoordinates(
        channel=FijiDimensionAxis("channel", ["channel"], [(1,)]),
        z_axis_coordinates=FijiDimensionAxis("z_axis", ["z_index"], [(0,)]),
        frame=FijiDimensionAxis("frame", ["site", "well"], [(2, "A14")]),
    )

    assert (
        FijiStackSliceLabelBuilder(coordinates).label_for(((1,), (0,), (2, "A14")))
        == "C1_Z0_T2_A14"
    )
