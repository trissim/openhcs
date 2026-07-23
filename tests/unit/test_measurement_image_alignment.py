from dataclasses import dataclass, replace

import numpy as np
import pytest

from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
    ImageOutputBundle,
)
from openhcs.core.measurement_image_alignment import (
    MeasurementImageAlignmentSource,
    MeasurementImageLabelAlignmentStrategy,
    MeasurementLabelSourceAlignmentStrategy,
    PreparedMeasurementObjectLabels,
)
from openhcs.core.measurement_image_alignment import (
    MeasurementImageReferenceDomain,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ObjectLabelPlaneDomainStrategy,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelRepresentation,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelPayload,
)
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
from openhcs.core.source_spatial_domain import SourceSpatialDomain


@dataclass(frozen=True, slots=True)
class _MeasurementImageSource(MeasurementImageAlignmentSource):
    payload: object
    reference_domain: MeasurementImageReferenceDomain = (
        MeasurementImageReferenceDomain.SOURCE_IMAGE
    )
    source_aliases: tuple[str, ...] = ()

    @property
    def alignment_image(self) -> object:
        return self.payload

    @property
    def alignment_reference_domain(self) -> MeasurementImageReferenceDomain:
        return self.reference_domain

    @property
    def alignment_source_aliases(self) -> tuple[str, ...]:
        return self.source_aliases

    def with_alignment_image(self, image: object) -> "_MeasurementImageSource":
        return replace(self, payload=image)


@dataclass(frozen=True, slots=True)
class _PlaneProjector(RuntimePlaneAxisProjector):
    runtime_index: int | None = None
    runtime_count: int | None = None
    source_index: int | None = None
    source_count: int | None = None

    def runtime_slice_plane_index(self) -> int | None:
        return self.runtime_index

    def runtime_slice_axis_size(self) -> int | None:
        return self.runtime_count

    def source_binding_axis_plane_index(
        self,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        del source_aliases
        return self.source_index

    def source_binding_axis_size(
        self,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        del source_aliases
        return self.source_count


def _plane_domain(*ids: tuple[int, ...]) -> ObjectLabelDomain:
    return ObjectLabelDomain.declared(
        scope=ObjectLabelDomainScope.PLANE,
        declared_object_id_domains=ids,
    )


def test_source_reference_requires_exact_shape_after_domain_selection() -> None:
    source = _MeasurementImageSource(np.ones((4, 5), dtype=np.float32))

    with pytest.raises(ValueError, match="incompatible declared domains"):
        MeasurementImageLabelAlignmentStrategy.align(
            source.alignment_request(labels=np.ones((3, 5), dtype=np.int32))
        )


@pytest.mark.parametrize(
    "image",
    (
        np.ones((1, 4, 5), dtype=np.float32),
        np.ones((4, 5, 3), dtype=np.float32),
    ),
)
def test_source_reference_does_not_infer_singleton_or_color_projection(
    image: np.ndarray,
) -> None:
    source = _MeasurementImageSource(image)

    with pytest.raises(ValueError, match="incompatible declared domains"):
        MeasurementImageLabelAlignmentStrategy.align(
            source.alignment_request(labels=np.ones((4, 5), dtype=np.int32))
        )


def test_payload_scoped_labels_consume_declared_singleton_runtime_image_plane() -> None:
    image = ImageMetadataPayload(
        data=np.ones((1, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.ones((4, 5), dtype=np.int32),
        ),
    )
    source = _MeasurementImageSource(image)

    aligned = MeasurementImageLabelAlignmentStrategy.align(
        source.object_label_alignment_request(labels)
    )

    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == (4, 5)


def test_payload_scoped_labels_preserve_declared_runtime_volume() -> None:
    image = ImageMetadataPayload(
        data=np.ones((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.ones((2, 4, 5), dtype=np.int32),
        ),
    )
    source = _MeasurementImageSource(image)

    aligned = MeasurementImageLabelAlignmentStrategy.align(
        source.object_label_alignment_request(labels)
    )

    assert aligned is image


def test_payload_scoped_labels_reject_mismatched_declared_runtime_volume() -> None:
    image = ImageMetadataPayload(
        data=np.ones((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.ones((3, 4, 5), dtype=np.int32),
        ),
    )
    source = _MeasurementImageSource(image)

    with pytest.raises(ValueError, match="incompatible declared domains"):
        MeasurementImageLabelAlignmentStrategy.align(
            source.object_label_alignment_request(labels)
        )


def test_payload_scoped_labels_reject_non_runtime_image_plane_axis() -> None:
    image = ImageMetadataPayload(
        data=np.ones((1, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        ),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.ones((4, 5), dtype=np.int32),
        ),
    )
    source = _MeasurementImageSource(image)

    with pytest.raises(ValueError, match="incompatible declared domains"):
        MeasurementImageLabelAlignmentStrategy.align(
            source.object_label_alignment_request(labels)
        )


def test_object_reference_requires_nominal_label_payload() -> None:
    source = _MeasurementImageSource(
        np.ones((4, 5), dtype=np.float32),
        reference_domain=MeasurementImageReferenceDomain.OBJECT_LABELS,
    )

    with pytest.raises(ValueError, match="requires an ObjectLabelValue"):
        MeasurementImageLabelAlignmentStrategy.align(
            source.alignment_request(labels=np.ones((4, 5), dtype=np.int32))
        )


def test_object_reference_uses_declared_source_spatial_adapter() -> None:
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((2, 2), dtype=np.int32)),
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(1, 1),
            source_shape_yx=(5, 5),
        ),
    )
    source = _MeasurementImageSource(
        image,
        reference_domain=MeasurementImageReferenceDomain.OBJECT_LABELS,
    )

    aligned = MeasurementImageLabelAlignmentStrategy.align(
        source.object_label_alignment_request(labels)
    )

    np.testing.assert_array_equal(aligned, image[1:3, 1:3])


def test_object_reference_preserves_nominal_image_payload_context() -> None:
    image = np.stack(
        (
            np.full((4, 5), 10, dtype=np.float32),
            np.full((4, 5), 20, dtype=np.float32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.stack(
                (
                    np.full((4, 5), 1, dtype=np.int32),
                    np.full((4, 5), 2, dtype=np.int32),
                )
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=_plane_domain((1,), (2,)),
    )
    source = _MeasurementImageSource(
        ImageMetadataPayload(
            data=image,
            metadata=ImagePayloadMetadata(
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        ),
        reference_domain=MeasurementImageReferenceDomain.OBJECT_LABELS,
    )

    aligned = MeasurementImageLabelAlignmentStrategy.align(
        source.object_label_alignment_request(labels)
    )

    assert isinstance(aligned, ImageMetadataPayload)
    assert aligned.metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
    np.testing.assert_array_equal(aligned.data, image)


def test_explicit_projector_selects_declared_source_binding_planes() -> None:
    image = np.stack(
        (
            np.full((4, 5), 10, dtype=np.float32),
            np.full((4, 5), 20, dtype=np.float32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.stack(
                (
                    np.full((4, 5), 1, dtype=np.int32),
                    np.full((4, 5), 2, dtype=np.int32),
                )
            )
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=_plane_domain((1,), (2,)),
    )
    source = _MeasurementImageSource(
        ImageMetadataPayload(
            data=image,
            metadata=ImagePayloadMetadata(
                plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            ),
        ),
        source_aliases=("marker", "brightfield"),
    )

    prepared = PreparedMeasurementObjectLabels.from_source(
        source,
        labels,
        plane_projector=_PlaneProjector(source_index=1, source_count=2),
    )

    np.testing.assert_array_equal(prepared.aligned_image, image[1])
    np.testing.assert_array_equal(prepared.measurement_labels, labels.labels[1])


def test_object_reference_projects_declared_runtime_slice_only_with_context() -> None:
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.stack(
                (
                    np.full((4, 5), 1, dtype=np.int32),
                    np.full((4, 5), 2, dtype=np.int32),
                )
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=_plane_domain((1,), (2,)),
    )
    source = _MeasurementImageSource(
        np.ones((4, 5), dtype=np.float32),
        reference_domain=MeasurementImageReferenceDomain.OBJECT_LABELS,
    )

    with pytest.raises(ValueError, match="incompatible declared domains"):
        PreparedMeasurementObjectLabels.from_source(source, labels)

    prepared = PreparedMeasurementObjectLabels.from_source(
        source,
        labels,
        plane_projector=_PlaneProjector(runtime_index=1, runtime_count=2),
    )
    np.testing.assert_array_equal(prepared.measurement_labels, labels.labels[1])


def test_object_reference_projects_declared_single_runtime_slice() -> None:
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.full((1, 4, 5), 7, dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=_plane_domain((7,)),
    )
    source = _MeasurementImageSource(
        np.ones((4, 5), dtype=np.float32),
        reference_domain=MeasurementImageReferenceDomain.OBJECT_LABELS,
    )

    prepared = PreparedMeasurementObjectLabels.from_source(
        source,
        labels,
        plane_projector=_PlaneProjector(runtime_count=1),
    )

    assert (
        prepared.source_projected_payload.domain.scope is ObjectLabelDomainScope.PAYLOAD
    )
    np.testing.assert_array_equal(
        prepared.measurement_labels,
        np.full((4, 5), 7, dtype=np.int32),
    )


def test_object_reference_projects_image_with_declared_single_runtime_slice() -> None:
    image = np.full((1, 4, 5), 3, dtype=np.float32)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.full((1, 4, 5), 7, dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=_plane_domain((7,)),
    )
    source = _MeasurementImageSource(
        ImageMetadataPayload(
            data=image,
            metadata=ImagePayloadMetadata(
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        ),
        reference_domain=MeasurementImageReferenceDomain.OBJECT_LABELS,
    )

    prepared = PreparedMeasurementObjectLabels.from_source(
        source,
        labels,
        plane_projector=_PlaneProjector(runtime_count=1),
    )

    np.testing.assert_array_equal(prepared.aligned_image, image[0])
    np.testing.assert_array_equal(prepared.measurement_labels, labels.labels[0])


def test_object_reference_projects_local_singleton_in_wider_runtime_scope() -> None:
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.full((1, 4, 5), 7, dtype=np.int32)
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=_plane_domain((7,)),
    )
    source = _MeasurementImageSource(
        np.ones((4, 5), dtype=np.float32),
        reference_domain=MeasurementImageReferenceDomain.OBJECT_LABELS,
    )

    prepared = PreparedMeasurementObjectLabels.from_source(
        source,
        labels,
        plane_projector=_PlaneProjector(runtime_count=2),
    )

    assert (
        prepared.source_projected_payload.domain.scope is ObjectLabelDomainScope.PAYLOAD
    )
    np.testing.assert_array_equal(
        prepared.measurement_labels,
        np.full((4, 5), 7, dtype=np.int32),
    )


def test_projected_payload_domain_is_not_projected_twice() -> None:
    runtime_planes = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.stack((runtime_planes, runtime_planes))
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=_plane_domain((1, 2), (1, 2)),
    )
    source = _MeasurementImageSource(
        np.ones((4, 5), dtype=np.float32),
        reference_domain=MeasurementImageReferenceDomain.OBJECT_LABELS,
    )

    with pytest.raises(ValueError, match="incompatible declared domains"):
        PreparedMeasurementObjectLabels.from_source(
            source,
            labels,
            plane_projector=_PlaneProjector(runtime_index=1, runtime_count=2),
        )


def test_aligned_image_stack_requires_declared_runtime_slice_labels() -> None:
    image = AlignedImageStack(
        (
            np.ones((4, 5), dtype=np.float32),
            np.ones((4, 5), dtype=np.float32),
        )
    )
    source = _MeasurementImageSource(image)

    with pytest.raises(ValueError, match="declared plane-scoped runtime-slice axis"):
        MeasurementImageLabelAlignmentStrategy.align(
            source.alignment_request(labels=np.ones((4, 5), dtype=np.int32))
        )


def test_singleton_aligned_image_stack_accepts_payload_scoped_labels() -> None:
    image = AlignedImageStack((np.ones((4, 5), dtype=np.float32),))
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((4, 5), dtype=np.int32))
    )

    aligned = MeasurementLabelSourceAlignmentStrategy.align(
        image,
        labels,
        label_payload=labels,
    )

    assert aligned is labels


def test_aligned_image_stack_accepts_nominal_runtime_slice_payload() -> None:
    image = AlignedImageStack(
        (
            np.ones((4, 5), dtype=np.float32),
            np.ones((4, 5), dtype=np.float32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.stack(
                (
                    np.ones((4, 5), dtype=np.int32),
                    np.full((4, 5), 2, dtype=np.int32),
                )
            )
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=_plane_domain((1,), (2,)),
    )
    source = _MeasurementImageSource(image)

    aligned = MeasurementImageLabelAlignmentStrategy.align(
        source.object_label_alignment_request(labels)
    )

    assert aligned is image


def test_aligned_image_stack_preserves_sparse_runtime_slice_payload() -> None:
    image = AlignedImageStack(
        (
            np.ones((4, 5), dtype=np.float32),
            np.ones((4, 5), dtype=np.float32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=SparseIJVLabelRows.from_slices(
                (
                    SparseIJVLabelRows.from_dense_labels(
                        np.pad(
                            np.ones((2, 2), dtype=np.int32),
                            ((0, 2), (0, 3)),
                        )
                    ),
                    SparseIJVLabelRows.from_dense_labels(
                        np.pad(
                            np.full((2, 2), 2, dtype=np.int32),
                            ((2, 0), (3, 0)),
                        )
                    ),
                )
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=_plane_domain((1,), (2,)),
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(4, 5)),
    )
    source = _MeasurementImageSource(image)

    aligned = MeasurementImageLabelAlignmentStrategy.align(
        source.object_label_alignment_request(labels)
    )

    assert aligned is image


def test_plane_domain_measurement_projection_preserves_sparse_storage() -> None:
    sparse_labels = SparseIJVLabelRows.from_slices(
        (
            SparseIJVLabelRows(np.asarray(((0, 0, 1),), dtype=np.int32)),
            SparseIJVLabelRows(np.asarray(((1, 1, 2),), dtype=np.int32)),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=sparse_labels),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=_plane_domain((1,), (2,)),
    )
    strategy = ObjectLabelPlaneDomainStrategy.for_enum_member(
        ObjectLabelDomainScope.PLANE
    )
    projection = strategy.measurement_projection(
        labels,
        _PlaneProjector(runtime_count=2),
    )

    planes = strategy.measurement_planes(labels, projection)

    assert len(planes) == 2
    assert all(
        plane.representation is ObjectLabelRepresentation.SPARSE_IJV for plane in planes
    )
    assert all(isinstance(plane.labels, SparseIJVLabelRows) for plane in planes)


def test_named_image_output_bundle_broadcasts_one_label_domain() -> None:
    image = ImageOutputBundle(
        (
            np.ones((4, 5), dtype=np.float32),
            np.full((4, 5), 2, dtype=np.float32),
        ),
        (
            AlignedImageSliceContext.main_flow("CropBlue"),
            AlignedImageSliceContext.main_flow("CropGreen"),
        ),
    )
    labels = np.ones((4, 5), dtype=np.int32)
    source = _MeasurementImageSource(image)

    aligned = MeasurementImageLabelAlignmentStrategy.align(
        source.alignment_request(labels=labels)
    )

    assert aligned is image


def test_runtime_slice_aligned_values_require_matching_aligned_image() -> None:
    image = AlignedImageStack(
        (
            np.ones((4, 5), dtype=np.float32),
            np.ones((4, 5), dtype=np.float32),
        )
    )
    labels = RuntimeSliceAlignedValues(
        (
            np.ones((4, 5), dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )

    assert MeasurementLabelSourceAlignmentStrategy.align(image, labels) is labels

    mismatched = RuntimeSliceAlignedValues((labels.slices[0],))
    with pytest.raises(ValueError, match="must match the aligned image count"):
        MeasurementLabelSourceAlignmentStrategy.align(image, mismatched)


def test_object_reference_replaces_unrelated_aligned_image_stack() -> None:
    image = AlignedImageStack(
        (
            np.ones((4, 5), dtype=np.float32),
            np.full((4, 5), 2, dtype=np.float32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((4, 5), dtype=np.int32))
    )
    source = _MeasurementImageSource(
        image,
        reference_domain=MeasurementImageReferenceDomain.OBJECT_LABELS,
    )

    prepared = PreparedMeasurementObjectLabels.from_source(
        source,
        labels,
    )

    assert np.asarray(prepared.aligned_image).shape == (4, 5)
    np.testing.assert_array_equal(prepared.measurement_labels, labels.labels)


def test_equal_label_planes_are_not_collapsed() -> None:
    image = np.ones((2, 4, 5), dtype=np.float32)
    plane = np.arange(20, dtype=np.int32).reshape(4, 5)
    labels = np.stack((plane, plane))

    aligned = MeasurementLabelSourceAlignmentStrategy.align(image, labels)

    assert aligned is labels
