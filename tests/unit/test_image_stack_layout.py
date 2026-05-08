import numpy as np

from openhcs.core.aligned_image_payload import (
    ImagePayloadExecutionMode,
    compose_aligned_image_payload,
    payload_slices_for_alignment,
    project_singleton_stack_image_domain,
)
from openhcs.core.image_shapes import (
    is_channel_last_image_slice,
    is_color_image_slice,
    is_image_stack,
)
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import MEMORY_TYPE_NUMPY
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    image_payload_with_context,
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


def test_payload_alignment_preserves_single_source_volume():
    volume = np.zeros((3, 4, 5), dtype=np.uint16)
    payload = image_payload_with_context(
        volume,
        metadata=ImagePayloadMetadata.for_array(volume, source_path="/tmp/source.tif"),
    )

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 1
    np.testing.assert_array_equal(slices[0].data, volume)


def test_aligned_image_payload_composes_multiple_source_volumes_as_image_bundle():
    volumes = tuple(np.full((3, 4, 5), index, dtype=np.uint16) for index in range(3))
    payloads = tuple(
        image_payload_with_context(
            volume,
            metadata=ImagePayloadMetadata.for_array(
                volume,
                source_path=f"/tmp/source_{index}.tif",
            ),
        )
        for index, volume in enumerate(volumes)
    )

    composition = compose_aligned_image_payload("ImageMath", payloads)

    assert composition.execution_mode is ImagePayloadExecutionMode.FULL_STACK
    assert composition.payload.data.shape == (3, 3, 4, 5)
    for index, volume in enumerate(volumes):
        np.testing.assert_array_equal(composition.payload.data[index], volume)


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
    payload = image_payload_with_context(
        volume_stack,
        mask=mask_stack,
        metadata=ImagePayloadMetadata.for_array(
            volume_stack,
            source_path="/tmp/source.tif",
        ),
    )

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

    observed = ImageStackLayout.unstack_result_for_source_slices(
        volume,
        source_slice_shapes=(tuple(volume.shape),),
        memory_type=MEMORY_TYPE_NUMPY,
        gpu_id=0,
    )

    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], volume)
