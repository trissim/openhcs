import numpy as np
import pytest

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    AlignedImageSliceContext,
    ImageOutputBundle,
    ImagePayloadSliceProjector,
    ImagePayloadExecutionMode,
    compose_aligned_image_payload,
    pack_aligned_image_outputs,
    payload_slices_for_alignment,
    stack_image_payloads,
)
from openhcs.core.memory import (
    MEMORY_TYPE_NUMPY,
    stack_runtime_slices,
    unstack_runtime_slices,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
    RuntimePlaneAxisValueProjection,
    RuntimePlaneProjection,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    ImagePayloadMetadataCompositionMode,
    MaskedImagePayload,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)


@pytest.mark.parametrize(
    "slice_shape",
    ((4, 5), (3, 4, 5), (4, 5, 3), (2, 3, 4, 5)),
)
def test_explicit_runtime_slice_stack_round_trips_any_slice_shape(
    slice_shape: tuple[int, ...],
) -> None:
    slices = (
        np.zeros(slice_shape, dtype=np.uint16),
        np.ones(slice_shape, dtype=np.uint16),
    )

    stack = stack_runtime_slices(slices, MEMORY_TYPE_NUMPY, 0)
    unstacked = unstack_runtime_slices(
        stack,
        MEMORY_TYPE_NUMPY,
        0,
        expected_count=2,
    )

    assert stack.shape == (2, *slice_shape)
    np.testing.assert_array_equal(unstacked[0], slices[0])
    np.testing.assert_array_equal(unstacked[1], slices[1])


def test_explicit_runtime_slice_stack_rejects_mismatched_slice_shapes() -> None:
    with pytest.raises(ValueError, match="one exact shape"):
        stack_runtime_slices(
            (np.zeros((4, 5)), np.zeros((4, 6))),
            MEMORY_TYPE_NUMPY,
            0,
        )


def test_bare_ndarray_does_not_declare_runtime_alignment() -> None:
    array = np.zeros((3, 4, 5), dtype=np.uint16)

    assert payload_slices_for_alignment(array) == (array,)


def test_aligned_image_stack_nominally_declares_runtime_alignment() -> None:
    slices = (
        np.zeros((4, 5), dtype=np.uint16),
        np.ones((4, 5), dtype=np.uint16),
    )
    payload = AlignedImageStack(slices)

    assert payload_slices_for_alignment(payload) == slices


def test_runtime_plane_projection_requires_matching_payload_axis() -> None:
    data = np.stack(
        (
            np.zeros((4, 5), dtype=np.uint16),
            np.ones((4, 5), dtype=np.uint16),
        )
    )
    mask = np.ones_like(data, dtype=bool)
    payload = ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(data, mask)

    selected = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=1,
            axis_size=2,
        ),
    )

    np.testing.assert_array_equal(image_payload_data(selected), data[1])
    np.testing.assert_array_equal(image_payload_mask(selected), mask[1])
    assert image_payload_metadata(selected).plane_axis is None


def test_source_binding_slice_projection_preserves_shared_spatial_mask() -> None:
    data = np.zeros((2, 4, 5), dtype=np.float32)
    mask = np.ones((4, 5), dtype=bool)
    mask[1, 2] = False
    metadata = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(4, 5)),
    )

    projected = ImagePayloadSliceProjector(mask=mask, metadata=metadata).mask_for_slice(
        data[1],
        1,
    )

    np.testing.assert_array_equal(projected, mask)


def test_source_binding_runtime_projection_preserves_shared_spatial_mask() -> None:
    data = np.zeros((2, 4, 5), dtype=np.float32)
    mask = np.ones((4, 5), dtype=bool)
    mask[1, 2] = False
    payload = ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_spatial_domain=SourceSpatialDomain(source_shape_yx=(4, 5)),
        ).payload_with(data, mask)

    projected = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.SOURCE_BINDING,
            plane_index=1,
            axis_size=2,
        ),
    )

    np.testing.assert_array_equal(image_payload_data(projected), data[1])
    np.testing.assert_array_equal(image_payload_mask(projected), mask)


def test_runtime_plane_projection_does_not_infer_undeclared_ndarray_axis() -> None:
    value = np.zeros((2, 4, 5), dtype=np.uint16)

    projected = RuntimeSliceProjection.value_for_slice(
        value,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=0,
            axis_size=2,
        ),
    )

    assert projected is value


def test_channel_free_mask_requires_declared_channel_axis() -> None:
    payload = MaskedImagePayload(
        data=np.zeros((4, 5, 2), dtype=np.float32),
        mask=np.ones((4, 5), dtype=bool),
        metadata=ImagePayloadMetadata(source_channel_axis=-1),
    )

    assert payload.mask.shape == (4, 5)

    with pytest.raises(ValueError, match="mask shape"):
        MaskedImagePayload(
            data=np.zeros((4, 5, 2), dtype=np.float32),
            mask=np.ones((4, 5), dtype=bool),
        )


def test_image_bundle_composition_declares_source_binding_axis() -> None:
    first = np.zeros((4, 5), dtype=np.uint16)
    second = np.ones((4, 5), dtype=np.uint16)

    composition = compose_aligned_image_payload("ImageMath", (first, second))

    assert composition.execution_mode is ImagePayloadExecutionMode.FULL_STACK
    assert image_payload_metadata(composition.payload).plane_axis is (
        RuntimePlaneAxis.SOURCE_BINDING
    )
    assert image_payload_data(composition.payload).shape == (2, 4, 5)


def test_single_runtime_slice_payload_is_not_repacked_as_source_binding() -> None:
    payload = ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(np.zeros((2, 4, 5), dtype=np.uint16), None)

    composition = compose_aligned_image_payload("MeasureImage", (payload,))

    assert composition.execution_mode is ImagePayloadExecutionMode.NATURAL
    assert composition.payload is payload
    assert image_payload_metadata(composition.payload).plane_axis is (
        RuntimePlaneAxis.RUNTIME_SLICE
    )


def test_composed_runtime_slice_projection_ignores_outer_file_cardinality() -> None:
    payloads = tuple(
        ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ).payload_with(np.zeros((2, 4, 5), dtype=np.uint16), None)
        for _ in range(2)
    )

    composition = compose_aligned_image_payload("MeasureImage", payloads)
    projection = composition.preserved_plane_projection(
        RuntimePlaneProjection.stack(10),
        source_aliases=("DNA", "RNA"),
    )

    assert projection is not None
    assert projection.axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert projection.axis_size == 2
    assert projection.source_aliases == ()


class _ThreeBindingSourceProjector(RuntimePlaneAxisProjector):
    def runtime_slice_plane_index(self) -> int | None:
        return None

    def source_binding_axis_plane_index(
        self,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        del source_aliases
        return None

    def source_binding_axis_size(
        self,
        source_aliases: tuple[str, ...],
    ) -> int:
        del source_aliases
        return 3


def test_source_bundle_projection_uses_payload_cardinality_not_outer_bindings() -> None:
    composition = compose_aligned_image_payload(
        "MeasureColocalization",
        (
            np.zeros((4, 5), dtype=np.uint16),
            np.ones((4, 5), dtype=np.uint16),
        ),
    )

    projection = composition.preserved_plane_projection(
        _ThreeBindingSourceProjector(),
        source_aliases=("CropBlue", "CropGreen"),
    )

    assert projection is not None
    assert projection.axis is RuntimePlaneAxis.SOURCE_BINDING
    assert projection.axis_size == 2
    assert projection.source_aliases == ("CropBlue", "CropGreen")


def test_single_aligned_payload_is_preserved_without_rebundling() -> None:
    payload = AlignedImageStack(
        (
            np.zeros((4, 5), dtype=np.uint16),
            np.ones((4, 5), dtype=np.uint16),
        )
    )

    composition = compose_aligned_image_payload("MeasureImage", (payload,))

    assert composition.execution_mode is (
        ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    )
    assert composition.payload is payload


def test_declared_output_contexts_preserve_aligned_multi_image_payloads() -> None:
    payloads = (
        np.zeros((4, 5), dtype=np.uint16),
        np.ones((4, 5), dtype=np.uint16),
    )
    contexts = (
        AlignedImageSliceContext.main_flow("CorrectedDNA"),
        AlignedImageSliceContext.main_flow("CorrectedProtein"),
    )

    composition = compose_aligned_image_payload(
        "CorrectIlluminationApply",
        payloads,
        slice_contexts=contexts,
    )

    assert composition.execution_mode is (
        ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    )
    assert isinstance(composition.payload, ImageOutputBundle)
    assert composition.payload.slices == payloads
    assert composition.payload.slice_contexts == contexts


def test_named_main_flow_context_attaches_exact_derived_image_identity() -> None:
    payload = ImagePayloadMetadata(
        source_image_names=("OrigDNA",),
    ).payload_with(np.zeros((4, 5), dtype=np.uint16))

    contextualized = AlignedImageSliceContext.main_flow(
        "CorrectedDNA"
    ).contextualize_image_payload(payload)

    assert image_payload_metadata(
        contextualized
    ).source_provenance.represented_source_image_names == (
        "CorrectedDNA",
        "OrigDNA",
    )


def test_named_outputs_are_contextualized_when_packed_for_chained_execution() -> None:
    payloads = (
        ImagePayloadMetadata(source_image_names=("OrigDNA",)).payload_with(
            np.zeros((4, 5), dtype=np.uint16)
        ),
        ImagePayloadMetadata(source_image_names=("OrigProtein",)).payload_with(
            np.ones((4, 5), dtype=np.uint16)
        ),
    )
    contexts = (
        AlignedImageSliceContext.main_flow("AlignedDNA"),
        AlignedImageSliceContext.main_flow("AlignedProtein"),
    )

    packed = pack_aligned_image_outputs(payloads, slice_contexts=contexts)

    assert isinstance(packed, ImageOutputBundle)
    assert tuple(
        image_payload_metadata(payload).source_provenance.represented_source_image_names
        for payload in packed.slices
    ) == (
        ("AlignedDNA", "OrigDNA"),
        ("AlignedProtein", "OrigProtein"),
    )


def test_named_outputs_keep_active_identity_when_restacked_for_adjacent_step() -> None:
    payloads = (
        ImagePayloadMetadata(source_image_names=("OrigDNA",)).payload_with(
            np.zeros((4, 5), dtype=np.uint16)
        ),
        ImagePayloadMetadata(source_image_names=("OrigProtein",)).payload_with(
            np.ones((4, 5), dtype=np.uint16)
        ),
    )
    packed = pack_aligned_image_outputs(
        payloads,
        slice_contexts=(
            AlignedImageSliceContext.main_flow("AlignedDNA"),
            AlignedImageSliceContext.main_flow("AlignedProtein"),
        ),
    )

    restacked = stack_image_payloads(
        packed.slices,
        metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
    )
    provenance = image_payload_metadata(restacked).source_provenance

    assert provenance.source_image_names == ("AlignedDNA", "AlignedProtein")
    assert provenance.represented_source_image_names == (
        "AlignedDNA",
        "AlignedProtein",
        "OrigDNA",
        "OrigProtein",
    )


def test_single_named_output_is_contextualized_before_unwrapping() -> None:
    payload = ImagePayloadMetadata(
        source_image_names=("OrigDNA",),
    ).payload_with(np.zeros((4, 5), dtype=np.uint16))

    packed = pack_aligned_image_outputs(
        (payload,),
        slice_contexts=(AlignedImageSliceContext.main_flow("CorrectedDNA"),),
    )

    assert image_payload_metadata(
        packed
    ).source_provenance.represented_source_image_names == (
        "CorrectedDNA",
        "OrigDNA",
    )


def test_anonymous_main_flow_context_does_not_rename_image_payload() -> None:
    payload = ImagePayloadMetadata(
        source_image_names=("OrigDNA",),
    ).payload_with(np.zeros((4, 5), dtype=np.uint16))

    contextualized = (
        AlignedImageSliceContext.anonymous_main_flow().contextualize_image_payload(
            payload
        )
    )

    assert contextualized is payload


def test_aligned_inputs_require_exact_nominal_axis_cardinality() -> None:
    first = AlignedImageStack(
        tuple(np.zeros((4, 5), dtype=np.float32) for _ in range(2))
    )
    second = AlignedImageStack(
        tuple(np.zeros((4, 5), dtype=np.float32) for _ in range(3))
    )

    with pytest.raises(ValueError, match="cardinalities must match exactly"):
        compose_aligned_image_payload("Align", (first, second))
