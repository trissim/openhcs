from __future__ import annotations

import numpy as np
import tifffile

from openhcs.core.config import FijiDisplayConfig, FijiLUT, FijiStreamingConfig
from openhcs.runtime.fiji_viewer_server import (
    FijiBatchSettlementState,
    FijiBatchWireParser,
    FijiClearStateControlPlan,
    FijiControlMessageAuthority,
    FijiControlMessagePlan,
    FijiControlRequestContext,
    FijiDimensionAxis,
    FijiDisplayConfigWireAdapter,
    FijiHyperstackCoordinates,
    FijiImageIntensityRange,
    FijiImagePayload,
    FijiImagePlaneLookup,
    FijiImageStackBuilder,
    FijiPlaneGeometry,
    FijiPayloadHandlerRequest,
    FijiRoiPayloadHandler,
    FijiSettleControlPlan,
    FijiSharedMemoryItemCopier,
    FijiStackSliceLabelBuilder,
    FijiWindowItemProjection,
    FijiWindowRegistry,
    FijiWireItem,
)
from openhcs.runtime.fiji_macro_runtime import (
    FijiMacroExecutionRequest,
    FijiMacroExecutionResponse,
)
from openhcs.runtime.viewer_protocol import (
    ViewerControlMessageType,
    ViewerControlResponse,
    ViewerSettlePhase,
    ViewerSettleProgress,
)
from openhcs.runtime.viewer_component_system import (
    ViewerComponentAxisSemanticsAuthority,
    ViewerComponentNameMetadata,
    ViewerComponentValueDomainPayload,
    ViewerObjectDisplayConfigInput,
)


PRODUCER_IDENTITY = {
    "origin": "pipeline",
    "output_kind": "main",
    "output_key": "Nuclei",
    "projection_key": "main",
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


def test_fiji_runtime_mode_follows_inherited_streaming_enablement() -> None:
    assert FijiStreamingConfig(enabled=False).viewer_runtime_config().display_enabled is False
    assert FijiStreamingConfig(enabled=True).viewer_runtime_config().display_enabled is True


def test_fiji_control_dispatch_registry_is_module_local_and_eager() -> None:
    registry = FijiControlMessagePlan.__registry__

    assert type(registry) is dict
    assert registry[ViewerControlMessageType.CLEAR_STATE.value] is FijiClearStateControlPlan
    assert registry[ViewerControlMessageType.SETTLE.value] is FijiSettleControlPlan


def test_fiji_settlement_reports_typed_terminal_progress() -> None:
    response = FijiControlMessageAuthority(
        FijiControlRequestContext(
            FijiWindowRegistry(),
            object(),
            FijiBatchSettlementState(),
        )
    ).response_for({"type": ViewerControlMessageType.SETTLE.value})
    wire_response = ViewerControlResponse(response.to_wire_mapping())

    assert wire_response.succeeded()
    assert ViewerSettleProgress.from_response(wire_response) == (
        ViewerSettleProgress.complete()
    )


def test_fiji_settlement_tracks_queued_and_bounded_native_work() -> None:
    settlement = FijiBatchSettlementState()

    settlement.queue_items(2)
    settlement.queue_items(3)
    queued_response = FijiSettleControlPlan().response(
        FijiControlRequestContext(FijiWindowRegistry(), object(), settlement),
        None,
    )
    queued = ViewerSettleProgress.from_response(
        ViewerControlResponse(queued_response.to_wire_mapping())
    )

    assert queued.phase is ViewerSettlePhase.RUNNING
    assert queued.completed_update_count == 0
    assert queued.total_update_count == 5
    assert queued.active_route is None

    # A new enqueue can race after the debounce engine drained the first five
    # items but before its callback claims them. Exact item counts preserve the
    # second aggregate instead of folding it into the active one.
    settlement.queue_items(4)
    route = settlement.begin_pending_update(5)
    settlement.complete_active_work_unit(route)
    settlement.complete_active_work_unit(route)
    active = settlement.progress()

    assert active.phase is ViewerSettlePhase.RUNNING
    assert active.active_route == route
    assert active.active_route_work_unit_count == 2

    settlement.complete_active_update(route)
    between_batches = settlement.progress()

    assert between_batches.phase is ViewerSettlePhase.RUNNING
    assert between_batches.completed_update_count == 5
    assert between_batches.total_update_count == 9

    next_route = settlement.begin_pending_update(4)
    settlement.complete_active_work_unit(next_route)
    settlement.complete_active_update(next_route)

    assert settlement.progress() == ViewerSettleProgress.complete(
        total_update_count=9
    )


def test_fiji_clear_state_resets_terminal_settlement_for_next_run() -> None:
    settlement = FijiBatchSettlementState()
    settlement.queue_items(1)
    route = settlement.begin_pending_update(1)
    settlement.fail_active_update(route, ValueError("invalid Fiji ROI"))

    response = FijiClearStateControlPlan().response(
        FijiControlRequestContext(FijiWindowRegistry(), object(), settlement),
        None,
    )

    assert response.header.status.value == "success"
    assert settlement.progress() == ViewerSettleProgress.complete()
    assert settlement.failure_message() is None


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
    response = FijiControlMessageAuthority(
        FijiControlRequestContext(
            FijiWindowRegistry(),
            object(),
            FijiBatchSettlementState(),
        )
    ).response_for({"type": ViewerControlMessageType.STATE.value})

    wire_response = response.to_wire_mapping()

    assert wire_response["status"] == "error"
    assert wire_response["type"] == "state_ack"
    assert (
        "Fiji live viewer state polling is not implemented"
        in wire_response["message"]
    )


def test_fiji_macro_control_executes_inside_managed_imagej_runtime(tmp_path) -> None:
    macro_path = tmp_path / "copy.ijm"
    macro_path.write_text("// managed macro", encoding="utf-8")

    class FakePyImageJ:
        def run_script(self, extension, script, variables):
            assert extension == "ijm"
            assert script == "// managed macro"
            assert variables["Threshold"] == "0.25"
            tifffile.imwrite(
                f"{variables['Directory']}/output.tif",
                np.full((4, 5), 7, dtype=np.uint8),
            )

    class FakeImageJ:
        py = FakePyImageJ()

    request = FijiMacroExecutionRequest.from_arrays(
        macro_path=macro_path,
        input_filenames=("input.tif",),
        output_filenames=("output.tif",),
        directory_variable="Directory",
        macro_variables={"Threshold": "0.25"},
        input_images=(np.zeros((4, 5), dtype=np.uint8),),
    )
    response = FijiControlMessageAuthority(
        FijiControlRequestContext(
            FijiWindowRegistry(),
            FakeImageJ(),
            FijiBatchSettlementState(),
        )
    ).response_for(
        {
            "type": FijiMacroExecutionRequest.message_type,
            "payload": request,
        }
    )

    assert response.header.status.value == "success"
    assert isinstance(response.payload, FijiMacroExecutionResponse)
    assert len(response.payload.outputs) == 1


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


def test_fiji_shared_memory_copy_leaves_sender_allocation_owned(monkeypatch) -> None:
    from multiprocessing import resource_tracker, shared_memory

    source = np.arange(12, dtype=np.uint16).reshape(3, 4)
    shm = shared_memory.SharedMemory(create=True, size=source.nbytes)
    np.ndarray(source.shape, dtype=source.dtype, buffer=shm.buf)[:] = source
    item = FijiWireItem.from_payload(
        {
            "data_type": "image",
            "image_id": "fiji-copy",
            "shm_name": shm.name,
            "shape": source.shape,
            "dtype": str(source.dtype),
        }
    )
    errors = []

    try:
        # Production uses separate sender/receiver trackers; this unit probe
        # hosts both roles in one process.
        monkeypatch.setattr(resource_tracker, "unregister", lambda *_args: None)
        copied = FijiSharedMemoryItemCopier(
            lambda image_id, error: errors.append((image_id, error))
        ).copy([item])
        monkeypatch.undo()

        assert errors == []
        np.testing.assert_array_equal(copied[0].data, source)
        reopened = shared_memory.SharedMemory(name=shm.name)
        reopened.close()
    finally:
        monkeypatch.undo()
        shm.close()
        shm.unlink()


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
    assert projection.coordinate_components.frame == [
        component
        for component in FijiDisplayConfig.COMPONENT_ORDER
        if component in {"site", "well", "timepoint"}
    ]


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


def test_fiji_image_stack_builder_reports_bounded_native_work_units(
    monkeypatch,
) -> None:
    import jpype
    import scyjava

    built_stacks = []

    class FakeImageStack:
        def __init__(self, width: int, height: int) -> None:
            self.width = width
            self.height = height
            self.slices = []
            built_stacks.append(self)

        def addSlice(self, label, processor) -> None:
            self.slices.append((label, processor))

    class FakeByteProcessor:
        def __init__(self, width, height, pixels, color_model) -> None:
            self.width = width
            self.height = height
            self.pixels = pixels
            self.color_model = color_model

    imported_types = {
        "ij.ImageStack": FakeImageStack,
        "ij.process.ByteProcessor": FakeByteProcessor,
        "ij.process.ShortProcessor": object,
        "ij.process.FloatProcessor": object,
    }
    monkeypatch.setattr(scyjava, "jimport", imported_types.__getitem__)
    monkeypatch.setattr(jpype, "JByte", object())
    monkeypatch.setattr(jpype, "JArray", lambda _type: lambda values: values)

    channel_values = [(index,) for index in range(65)]
    coordinates = FijiHyperstackCoordinates(
        channel=FijiDimensionAxis("channel", ["channel"], channel_values),
        z_axis_coordinates=FijiDimensionAxis("z_axis", [], [()]),
        frame=FijiDimensionAxis("frame", [], [()]),
    )
    image_lookup = FijiImagePlaneLookup(
        {
            (channel_value, (), ()): np.full((2, 3), index, dtype=np.uint8)
            for index, channel_value in enumerate(channel_values)
        }
    )
    completed_units = []

    stack = FijiImageStackBuilder(
        image_lookup=image_lookup,
        coordinates=coordinates,
        width=3,
        height=2,
    ).build(
        work_unit_completed=lambda: completed_units.append(
            len(built_stacks[0].slices)
        )
    )

    assert len(stack.slices) == 65
    assert completed_units == [32, 64, 65]


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


def test_fiji_roi_handler_converts_and_adds_bounded_native_work_units(
    monkeypatch,
) -> None:
    from polystore.roi_converters import FijiROIConverter

    class FakeJavaRoi:
        def __init__(self) -> None:
            self.name = None
            self.position = None
            self.group = None

        def setName(self, name) -> None:
            self.name = name

        def setPosition(self, channel, z_slice, frame) -> None:
            self.position = (channel, z_slice, frame)

        def setGroup(self, group) -> None:
            self.group = group

    class FakeRoiManager:
        def __init__(self) -> None:
            self.rois = []
            self.visible = False

        def addRoi(self, roi) -> None:
            self.rois.append(roi)

        def isVisible(self) -> bool:
            return self.visible

        def setVisible(self, visible) -> None:
            self.visible = visible

    class FakeRoiManagerProvider:
        def __init__(self, manager) -> None:
            self._manager = manager

        def manager(self):
            return self._manager

        def add_rois(self, roi_manager, java_rois) -> None:
            for java_roi in java_rois:
                roi_manager.addRoi(java_roi)

        def show(self, roi_manager) -> None:
            roi_manager.setVisible(True)

    class FakeServer:
        def __init__(self) -> None:
            self.windows = FijiWindowRegistry()
            self.acknowledgments = []

        def _send_ack(self, image_id, status="success", error=None) -> None:
            self.acknowledgments.append((image_id, status, error))

    conversion_sizes = []

    def convert_work_unit(encoded_rois, _scyjava):
        conversion_sizes.append(len(encoded_rois))
        return [FakeJavaRoi() for _ in encoded_rois]

    monkeypatch.setattr(
        FijiROIConverter,
        "transmission_to_java_rois",
        convert_work_unit,
    )

    display_config = FijiDisplayConfig()
    semantics = ViewerComponentAxisSemanticsAuthority.from_display_config(
        ViewerObjectDisplayConfigInput(display_config),
        _component_value_domain(
            {
                "channel": [1],
                "z_index": [0],
                "timepoint": [0],
            }
        ),
    )
    component_names = ViewerComponentNameMetadata.empty()
    coordinates = FijiHyperstackCoordinates(
        channel=FijiDimensionAxis("channel", ["channel"], [(1,)]),
        z_axis_coordinates=FijiDimensionAxis("z_axis", ["z_index"], [(0,)]),
        frame=FijiDimensionAxis("frame", ["timepoint"], [(0,)]),
    )
    item = FijiWireItem.from_payload(
        {
            "data_type": "rois",
            "image_id": "roi-batch",
            "path": "/tmp/cells.tif",
            "metadata": {"channel": 1, "z_index": 0, "timepoint": 0},
            "rois": [{"index": index} for index in range(257)],
        }
    )
    manager = FakeRoiManager()
    server = FakeServer()
    completed_units = []
    request = FijiPayloadHandlerRequest(
        viewer_display_config=display_config,
        store=component_names.store,
        entries=semantics.entries,
        layout=semantics.layout,
        server=server,
        window_key="cells",
        items=[item],
        coordinates=coordinates,
        work_unit_completed=lambda: completed_units.append(len(manager.rois)),
    )

    FijiRoiPayloadHandler(
        roi_manager_provider=FakeRoiManagerProvider(manager)
    ).handle(request)

    assert conversion_sizes == [128, 128, 1]
    assert completed_units == [128, 256, 257]
    assert len(manager.rois) == 257
    assert manager.rois[0].name == "cells_0000"
    assert manager.rois[-1].name == "cells_0256"
    assert manager.rois[-1].position == (1, 1, 1)
    assert manager.rois[-1].group == server.windows.group_id("cells")
    assert manager.visible is True
    assert server.acknowledgments == [("roi-batch", "success", None)]
