import numpy as np

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    AlignedImageSliceContext,
    ImagePayloadExecutionMode,
    compose_aligned_image_payload,
    payload_slices_for_alignment,
    project_singleton_stack_image_domain,
)
from openhcs.core.image_shapes import (
    is_channel_first_volume_stack,
    is_channel_last_image_slice,
    is_color_image_slice,
    is_image_stack,
)
from openhcs.core.image_stack_layout import ImageStackLayout, SourceSliceUnstackRequest
from openhcs.core.memory import MEMORY_TYPE_NUMPY
from openhcs.core.pipeline_image_schema import ColorImageTypeSourceRole
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    MaskedImagePayload,
    RuntimeImagePayloadContext,
)
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.interop.cellprofiler.runtime.pure2d_output_aggregation import (
    CellProfilerPure2DImagePlaneSemantics,
)


def test_image_stack_layout_stacks_grayscale_volume_slices():
    first = np.zeros((3, 4, 5), dtype=np.uint16)
    second = np.ones((3, 4, 5), dtype=np.uint16)

    layout = ImageStackLayout.for_slices((first, second))
    stacked = layout.stack(
        slices=(first, second),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    )

    assert stacked.shape == (2, 3, 4, 5)
    assert is_image_stack(stacked)
    unstacked = ImageStackLayout.for_stack(stacked).unstack(
        array=stacked,
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    )
    np.testing.assert_array_equal(unstacked[0], first)
    np.testing.assert_array_equal(unstacked[1], second)


def test_image_stack_layout_stacks_color_volume_slices():
    first = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    second = np.ones((3, 4, 5, 3), dtype=np.uint8)

    layout = ImageStackLayout.for_slices((first, second))
    stacked = layout.stack(
        slices=(first, second),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    )

    assert stacked.shape == (2, 3, 4, 5, 3)
    assert is_image_stack(stacked)
    unstacked = ImageStackLayout.for_stack(stacked).unstack(
        array=stacked,
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    )
    np.testing.assert_array_equal(unstacked[0], first)
    np.testing.assert_array_equal(unstacked[1], second)


def test_image_stack_layout_stacks_channel_first_volume_slices():
    first = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    second = np.ones((2, 3, 4, 5), dtype=np.uint16)

    layout = ImageStackLayout.for_slices((first, second))
    stacked = layout.stack(
        slices=(first, second),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    )

    assert stacked.shape == (2, 2, 3, 4, 5)
    assert is_channel_first_volume_stack(stacked)
    assert is_image_stack(stacked)
    unstacked = ImageStackLayout.for_stack(stacked).unstack(
        array=stacked,
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    )
    np.testing.assert_array_equal(unstacked[0], first)
    np.testing.assert_array_equal(unstacked[1], second)


def test_channel_last_shape_predicate_accepts_non_rgb_planes():
    two_channel = np.zeros((4, 5, 2), dtype=np.float32)

    assert is_channel_last_image_slice(two_channel)
    assert not is_color_image_slice(two_channel)


def test_payload_alignment_unstacks_grayscale_volume_stacks():
    first = np.zeros((3, 4, 5), dtype=np.uint16)
    second = np.ones((3, 4, 5), dtype=np.uint16)
    stacked = np.stack((first, second))

    slices = payload_slices_for_alignment(stacked)

    assert len(slices) == 2
    np.testing.assert_array_equal(slices[0], first)
    np.testing.assert_array_equal(slices[1], second)


def test_payload_alignment_slices_plain_three_dimensional_stack():
    stack = np.zeros((3, 4, 5), dtype=np.uint16)
    payload = RuntimeImagePayloadContext(
        stack,
        metadata=ImagePayloadMetadata.for_array(stack, source_path="/tmp/source.tif"),
    mask = None).payload()

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 3
    for index, payload_slice in enumerate(slices):
        np.testing.assert_array_equal(payload_slice.data, stack[index])


def test_payload_alignment_preserves_declared_single_source_volume():
    volume = np.zeros((3, 4, 5), dtype=np.uint16)
    payload = RuntimeImagePayloadContext(
        volume,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/tmp/source.tif",) * 3,
                component_metadata=(
                    {"z_index": "1"},
                    {"z_index": "2"},
                    {"z_index": "3"},
                ),
            ),
        ),
    mask = None).payload()

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 1
    np.testing.assert_array_equal(slices[0].data, volume)


def test_payload_alignment_slices_multi_source_three_dimensional_stack():
    stack = np.zeros((3, 4, 5), dtype=np.uint16)
    payload = RuntimeImagePayloadContext(
        stack,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/tmp/source_s1.tif",
                    "/tmp/source_s2.tif",
                    "/tmp/source_s3.tif",
                ),
                component_metadata=(
                    {"site": "1"},
                    {"site": "2"},
                    {"site": "3"},
                ),
            ),
        ),
    mask = None).payload()

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 3
    for index, payload_slice in enumerate(slices):
        np.testing.assert_array_equal(payload_slice.data, stack[index])


def test_color_source_role_does_not_treat_grayscale_stack_as_channel_last_plane():
    role = ColorImageTypeSourceRole()

    assert role.is_channel_last_source_plane(np.zeros((4, 5, 2), dtype=np.uint8))
    assert role.is_channel_last_source_plane(np.zeros((4, 5, 3), dtype=np.uint8))
    assert not role.is_channel_last_source_plane(
        np.zeros((3, 4, 5), dtype=np.uint8)
    )
    assert role.is_channel_last_source_stack(
        np.zeros((3, 4, 5, 2), dtype=np.uint8)
    )
    assert role.is_channel_last_source_stack(
        np.zeros((3, 4, 5, 3), dtype=np.uint8)
    )
    assert not role.is_channel_last_source_stack(
        np.zeros((3, 4, 5), dtype=np.uint8)
    )


def test_cellprofiler_pure2d_semantics_keeps_hwc_color_slice_whole():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    payload = RuntimeImagePayloadContext(
        image,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/tmp/red.tif", "/tmp/green.tif", "/tmp/blue.tif"),
                component_metadata=(
                    {"channel": "1"},
                    {"channel": "2"},
                    {"channel": "3"},
                ),
            ),
        ),
    mask = None).payload()

    semantics = CellProfilerPure2DImagePlaneSemantics.from_image(payload)

    assert semantics.is_single_source_plane()
    assert semantics.slices(MEMORY_TYPE_NUMPY) == (payload,)


def test_masked_image_payload_accepts_two_channel_color_spatial_mask():
    plane = MaskedImagePayload(
        data=np.zeros((4, 5, 2), dtype=np.float32),
        mask=np.ones((4, 5), dtype=bool),
    )
    stack = MaskedImagePayload(
        data=np.zeros((3, 4, 5, 2), dtype=np.float32),
        mask=np.ones((3, 4, 5), dtype=bool),
    )

    assert plane.mask.shape == (4, 5)
    assert stack.mask.shape == (3, 4, 5)


def test_aligned_image_payload_composes_multiple_source_volumes_as_image_bundle():
    volumes = tuple(np.full((3, 4, 5), index, dtype=np.uint16) for index in range(3))
    payloads = tuple(
        RuntimeImagePayloadContext(
            volume,
            metadata=ImagePayloadMetadata(
                source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                    paths=(f"/tmp/source_{index}.tif",) * 3,
                    component_metadata=(
                        {"z_index": "1"},
                        {"z_index": "2"},
                        {"z_index": "3"},
                    ),
                ),
            ),
        mask = None).payload()
        for index, volume in enumerate(volumes)
    )

    composition = compose_aligned_image_payload("ImageMath", payloads)

    assert composition.execution_mode is ImagePayloadExecutionMode.FULL_STACK
    assert composition.payload.data.shape == (3, 3, 4, 5)
    for index, volume in enumerate(volumes):
        np.testing.assert_array_equal(composition.payload.data[index], volume)


def test_aligned_image_payload_preserves_declared_slice_contexts():
    payloads = tuple(np.full((4, 5), index, dtype=np.uint16) for index in range(2))
    contexts = (
        AlignedImageSliceContext.main_flow(
            output_key="CorrProtein",
            artifact_kind="image",
        ),
        AlignedImageSliceContext.main_flow(
            output_key="CorrDNA",
            artifact_kind="image",
        ),
    )

    composition = compose_aligned_image_payload(
        "CorrectIlluminationApply",
        payloads,
        slice_contexts=contexts,
    )

    assert composition.execution_mode is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    assert isinstance(composition.payload, AlignedImageStack)
    assert composition.payload.slices == payloads
    assert composition.payload.slice_contexts == contexts


def test_aligned_image_payload_cycles_factorized_runtime_axes():
    template_stack = np.stack(
        (
            np.full((4, 5), 1, dtype=np.float32),
            np.full((4, 5), 2, dtype=np.float32),
        )
    )
    first_stack = np.stack(
        tuple(np.full((4, 5), index, dtype=np.float32) for index in range(6))
    )
    second_stack = first_stack + 10

    composition = compose_aligned_image_payload(
        "Align",
        (template_stack, first_stack, second_stack),
    )

    assert composition.execution_mode is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    assert isinstance(composition.payload, AlignedImageStack)
    assert len(composition.payload.slices) == 6
    for slice_index, composed_slice in enumerate(composition.payload.slices):
        np.testing.assert_array_equal(
            composed_slice[0],
            template_stack[slice_index % 2],
        )
        np.testing.assert_array_equal(composed_slice[1], first_stack[slice_index])
        np.testing.assert_array_equal(composed_slice[2], second_stack[slice_index])


def test_image_stack_layout_preserves_unambiguous_single_volume_stack():
    stacked = np.zeros((2, 3, 4, 5), dtype=np.uint16)

    observed = ImageStackLayout.stack_slices_or_single_stack(
        (stacked,),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    )

    assert observed is stacked


def test_singleton_stack_image_domain_projects_volume_payload_context():
    volume_stack = np.ones((1, 3, 4, 5), dtype=np.float32)
    mask_stack = np.ones((1, 3, 4, 5), dtype=bool)
    payload = RuntimeImagePayloadContext(
        volume_stack,
        mask=mask_stack,
        metadata=ImagePayloadMetadata.for_array(
            volume_stack,
            source_path="/tmp/source.tif",
        ),
    ).payload()

    projected = project_singleton_stack_image_domain(payload)

    assert projected.data.shape == (3, 4, 5)
    assert projected.mask.shape == (3, 4, 5)


def test_image_stack_layout_stacks_single_ambiguous_volume_slice():
    volume = np.zeros((3, 4, 5), dtype=np.uint16)

    observed = ImageStackLayout.stack_slices_or_single_stack(
        (volume,),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    )

    assert observed.shape == (1, 3, 4, 5)


def test_image_stack_layout_unstacks_result_matching_source_volume_as_single_slice():
    volume = np.zeros((3, 4, 5), dtype=np.uint16)

    observed = SourceSliceUnstackRequest(
        array=volume,
        source_slice_shapes=(tuple(volume.shape),),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    ).slices()

    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], volume)


def test_image_stack_layout_unstacks_single_grayscale_result_from_color_source():
    image = np.zeros((4, 5), dtype=np.float32)

    observed = SourceSliceUnstackRequest(
        array=image,
        source_slice_shapes=((4, 5, 3),),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    ).slices()

    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], image)


def test_image_stack_layout_unstacks_singleton_stack_for_grayscale_source_slice():
    stack = np.zeros((1, 4, 5), dtype=np.float32)

    observed = SourceSliceUnstackRequest(
        array=stack,
        source_slice_shapes=((4, 5),),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    ).slices()

    assert len(observed) == 1
    assert observed[0].shape == (4, 5)
    np.testing.assert_array_equal(observed[0], stack[0])


def test_image_stack_layout_rejects_multiple_source_slices_without_stack_axis():
    image = np.zeros((4, 5), dtype=np.float32)

    try:
        SourceSliceUnstackRequest(
            array=image,
            source_slice_shapes=((4, 5, 3), (4, 5, 3)),
            memory_type=MEMORY_TYPE_NUMPY,
            gpu_id=0,
        ).slices()
    except ValueError as exc:
        assert "OpenHCS image stack must be shaped" in str(exc)
    else:
        raise AssertionError("Expected single-slice result to reject multi-source output")


def test_image_stack_layout_unstacks_masked_volume_stack_payload_with_slice_context():
    stack = np.zeros((1, 3, 4, 5), dtype=np.uint16)
    mask = np.ones((1, 3, 4, 5), dtype=bool)
    payload = MaskedImagePayload(data=stack, mask=mask)

    observed = SourceSliceUnstackRequest(
        array=payload,
        source_slice_shapes=((3, 4, 5),),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    ).slices()

    assert len(observed) == 1
    assert isinstance(observed[0], MaskedImagePayload)
    assert observed[0].data.shape == (3, 4, 5)
    assert observed[0].mask.shape == (3, 4, 5)


def test_image_stack_layout_stacks_masked_volume_slices_by_array_operand():
    first = MaskedImagePayload(
        data=np.zeros((3, 4, 5), dtype=np.uint16),
        mask=np.ones((3, 4, 5), dtype=bool),
    )
    second = MaskedImagePayload(
        data=np.ones((3, 4, 5), dtype=np.uint16),
        mask=np.ones((3, 4, 5), dtype=bool),
    )

    observed = ImageStackLayout.for_slices((first, second)).stack(
        slices=(first, second),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    )

    assert observed.shape == (2, 3, 4, 5)
