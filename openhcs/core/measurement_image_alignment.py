"""Nominal measurement-image alignment contracts."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

from openhcs.core.aligned_image_payload import AlignedImageStack, ImageOutputBundle
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelMeasurementSource,
    ObjectLabelValue,
    ObjectLabelVariantData,
    object_label_dense_array,
    object_label_project_plane,
)
from openhcs.core.runtime_profile import RuntimeProfileLogger, RuntimeProfileTimer
from openhcs.core.runtime_object_label_domains import ObjectLabelDomainScope
from openhcs.core.runtime_object_labels import ObjectLabelRepresentation
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.source_spatial_domain import SourceSpatialDomainAdapter

from enum import Enum

logger = logging.getLogger(__name__)


class MeasurementImageAlignmentSource(ABC):
    """Nominal source image contract consumed by measurement-image alignment."""

    @property
    @abstractmethod
    def alignment_image(self) -> RuntimeArrayData | AlignedImageStack:
        """Return the image-like payload used for measurement alignment."""

    @property
    def alignment_reference_domain(self) -> MeasurementImageReferenceDomain:
        """Return the domain that owns alignment shape decisions."""
        return MeasurementImageReferenceDomain.SOURCE_IMAGE

    @property
    def alignment_source_aliases(self) -> tuple[str, ...]:
        """Return ordered source aliases carried by a composed measurement image."""
        return ()

    @property
    def alignment_image_name(self) -> str:
        """Return a diagnostic image name for alignment errors."""
        return "measurement image"

    @abstractmethod
    def with_alignment_image(
        self,
        image: RuntimeArrayData | AlignedImageStack,
    ) -> "MeasurementImageAlignmentSource":
        """Return this source with projected image data and identical provenance."""

    def alignment_request(
        self,
        *,
        labels: ObjectLabelMeasurementSource,
        label_payload: ObjectLabelValue | None = None,
        plane_projector: RuntimePlaneAxisProjector | None = None,
        align_image_to_labels: bool = True,
    ) -> "MeasurementImageLabelAlignmentRequest":
        """Return a request carrying only label/projector-local alignment facts."""
        return MeasurementImageLabelAlignmentRequest(
            source=self,
            labels=labels,
            label_payload=label_payload,
            plane_projector=plane_projector,
            align_image_to_labels=align_image_to_labels,
        )

    def object_label_alignment_request(
        self,
        label_payload: ObjectLabelValue,
        *,
        plane_projector: RuntimePlaneAxisProjector | None = None,
        align_image_to_labels: bool = True,
    ) -> "MeasurementImageLabelAlignmentRequest":
        """Return a measurement-label alignment request from an object-label payload."""
        return self.alignment_request(
            labels=label_payload,
            label_payload=label_payload,
            plane_projector=plane_projector,
            align_image_to_labels=align_image_to_labels,
        )


@dataclass(slots=True)
class MeasurementImageLabelAlignmentRequest:
    """Typed facts required to align one measurement image to object labels."""

    source: MeasurementImageAlignmentSource
    labels: ObjectLabelMeasurementSource | RuntimeSliceAlignedValueSet
    label_payload: ObjectLabelValue | None = None
    plane_projector: RuntimePlaneAxisProjector | None = None
    align_image_to_labels: bool = True

    @property
    def image(self) -> RuntimeArrayData | AlignedImageStack:
        """Return image data from the owning measurement-image source."""
        return self.source.alignment_image

    @property
    def image_data(self) -> RuntimeArrayData | AlignedImageStack:
        """Return image data derived from the measurement image payload."""
        return image_payload_data(self.image)

    @property
    def reference_domain(self) -> MeasurementImageReferenceDomain:
        """Return the source-owned alignment reference domain."""
        return MeasurementImageReferenceDomain(
            self.source.alignment_reference_domain,
        )

    @property
    def source_aliases(self) -> tuple[str, ...]:
        """Return the source-owned aliases for source-binding projection."""
        return self.source.alignment_source_aliases

    @property
    def image_array(self) -> np.ndarray:
        """Return dense image data after strategy ownership has validated it."""
        if not isinstance(self.image_data, np.ndarray):
            raise TypeError(
                "Measurement image alignment requires ndarray image data, got "
                f"{type(self.image_data).__name__}."
            )
        return self.image_data

    def with_source_projected_image(self) -> "MeasurementImageLabelAlignmentRequest":
        """Return this request with only the measurement image source-projected."""
        projection = RuntimePlaneAxisValueProjection.from_projector(
            self.plane_projector,
            RuntimePlaneAxis.SOURCE_BINDING,
            self.source_aliases,
        )
        if projection is None or projection.plane_index is None:
            return self
        image = self.image
        if isinstance(image, AlignedImageStack):
            projected_image: RuntimeArrayData | AlignedImageStack = image.with_slices(
                tuple(
                    RuntimeSliceProjection.value_for_slice(
                        image_slice,
                        replace(
                            projection,
                            source_aliases=image_payload_metadata(
                                image_slice
                            ).source_image_names,
                        ),
                    )
                    for image_slice in image.slices
                )
            )
        else:
            projected_image = RuntimeSliceProjection.value_for_slice(
                image,
                replace(
                    projection,
                    source_aliases=image_payload_metadata(image).source_image_names,
                ),
            )
        return replace(
            self,
            source=self.source.with_alignment_image(projected_image),
        )

    def with_source_projected_labels(self) -> "MeasurementImageLabelAlignmentRequest":
        """Project labels only through their declared axis and runtime context."""
        if self.label_payload is None:
            return self
        if (
            self.label_payload.object_label_domain().scope
            is not ObjectLabelDomainScope.PLANE
        ):
            return self
        declared_projection = self.label_payload.declared_plane_projection()
        if declared_projection is None:
            raise ValueError(
                "Plane-scoped object labels require a declared local plane projection."
            )
        runtime_projection = RuntimePlaneAxisValueProjection.from_projector(
            self.plane_projector,
            self.label_payload.plane_axis,
            self.source_aliases,
        )
        if runtime_projection is None:
            return self
        if runtime_projection.plane_index is None:
            if declared_projection.axis_size != 1:
                return self
            projection = declared_projection.selected_plane(0)
        elif declared_projection.axis_size == 1:
            projection = declared_projection.selected_plane(0)
        elif runtime_projection.axis_size != declared_projection.axis_size:
            raise ValueError(
                "Object-label runtime plane-axis cardinality conflicts with its "
                "declared local payload: "
                f"{runtime_projection.axis_size} != {declared_projection.axis_size}."
            )
        else:
            projection = declared_projection.selected_plane(
                runtime_projection.require_plane_index()
            )
        source_payload = self.label_payload
        if not isinstance(self.labels, ObjectLabelValue):
            source_labels = object_label_dense_array(source_payload)
            if self.labels is not source_labels:
                source_payload = source_payload.with_measurement_labels(self.labels)
        payload_projection = RuntimeSliceProjection.value_for_slice(
            source_payload,
            replace(
                projection,
                source_aliases=source_payload.source_image_names,
            ),
        )
        if not isinstance(payload_projection, ObjectLabelValue):
            raise TypeError(
                "Object-label runtime projection must preserve ObjectLabelValue, got "
                f"{type(payload_projection).__name__}."
            )
        if payload_projection is source_payload:
            return self
        source = self.source
        image = self.image
        image_metadata = image_payload_metadata(image)
        if (
            not isinstance(image, AlignedImageStack)
            and image_metadata.plane_axis is projection.axis
        ):
            source = source.with_alignment_image(
                RuntimeSliceProjection.value_for_slice(
                    image,
                    replace(
                        projection,
                        source_aliases=image_metadata.source_image_names,
                    ),
                )
            )
        return replace(
            self,
            source=source,
            labels=(
                payload_projection
                if isinstance(self.labels, ObjectLabelValue)
                else object_label_dense_array(payload_projection)
            ),
            label_payload=payload_projection,
        )


@dataclass(frozen=True, slots=True)
class PreparedMeasurementObjectLabels:
    """Single-pass object-label preparation for measurement-image execution."""

    request: MeasurementImageLabelAlignmentRequest
    source_payload: ObjectLabelValue
    source_projected_payload: ObjectLabelValue
    source_projected_labels: ObjectLabelMeasurementSource
    aligned_source: MeasurementImageAlignmentSource
    measurement_labels: ObjectLabelMeasurementSource
    completion_payload: ObjectLabelValue

    @property
    def aligned_image(self) -> RuntimeArrayData | AlignedImageStack:
        """Return the image from the source that owns its projected semantics."""

        return self.aligned_source.alignment_image

    @classmethod
    def from_source(
        cls,
        source: MeasurementImageAlignmentSource,
        label_payload: ObjectLabelValue,
        *,
        plane_projector: RuntimePlaneAxisProjector | None = None,
        align_image_to_labels: bool = True,
    ) -> "PreparedMeasurementObjectLabels":
        """Prepare labels from one measurement-image source and payload."""
        return cls.from_request(
            request=source.object_label_alignment_request(
                label_payload,
                plane_projector=plane_projector,
                align_image_to_labels=align_image_to_labels,
            ),
        )

    @classmethod
    def from_request(
        cls,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> "PreparedMeasurementObjectLabels":
        """Prepare image, dense labels, and completion payload in one pass."""
        if request.label_payload is None:
            raise TypeError(
                "Measurement object-label preparation requires label_payload."
            )
        profile_enabled = RuntimeProfileLogger.enabled()
        source_payload = request.label_payload
        source_projection_timer = RuntimeProfileTimer.start()
        source_projected_request = request.with_source_projected_labels()
        source_projected_labels = source_projected_request.labels
        source_projection_payload = source_projected_request.label_payload
        if source_projection_payload is None:
            raise TypeError(
                "Measurement object-label source projection lost label_payload."
            )
        if profile_enabled:
            RuntimeProfileLogger.log(
                logger,
                "measurement_object_labels_source_projection",
                source_projection_timer.elapsed(),
                reference_domain=request.reference_domain.value,
                source_type=type(request.image).__name__,
                labels_type=type(request.labels).__name__,
                source_aliases=request.source_aliases,
            )
        source_context_timer = RuntimeProfileTimer.start()
        source_projected_payload = cls.source_context_payload(
            source_projected_request,
            source_projection_payload,
            source_projected_labels,
        )
        if profile_enabled:
            RuntimeProfileLogger.log(
                logger,
                "measurement_object_labels_source_context",
                source_context_timer.elapsed(),
                reference_domain=request.reference_domain.value,
                payload_reused=source_projected_payload is source_payload,
            )
        image_request = replace(
            source_projected_request,
            labels=source_projected_labels,
            label_payload=source_projected_payload,
        )
        image_align_timer = RuntimeProfileTimer.start()
        if request.align_image_to_labels:
            aligned_image = MeasurementImageLabelAlignmentStrategy.align(image_request)
        else:
            aligned_image = request.image
        aligned_source = image_request.source.with_alignment_image(aligned_image)
        if profile_enabled:
            RuntimeProfileLogger.log(
                logger,
                "measurement_object_labels_image_alignment",
                image_align_timer.elapsed(),
                reference_domain=request.reference_domain.value,
                align_image_to_labels=request.align_image_to_labels,
                aligned_type=type(aligned_image).__name__,
            )
        label_align_timer = RuntimeProfileTimer.start()
        aligned_measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(
            aligned_image,
            source_projected_labels,
            label_payload=source_projected_payload,
        )
        measurement_labels = (
            object_label_dense_array(aligned_measurement_labels)
            if isinstance(aligned_measurement_labels, ObjectLabelValue)
            else aligned_measurement_labels
        )
        if profile_enabled:
            RuntimeProfileLogger.log(
                logger,
                "measurement_object_labels_label_alignment",
                label_align_timer.elapsed(),
                reference_domain=request.reference_domain.value,
                labels_reused=aligned_measurement_labels is source_projected_labels,
            )
        completion_timer = RuntimeProfileTimer.start()
        completion_payload = source_projected_payload.with_measurement_labels(
            measurement_labels
        )
        if profile_enabled:
            RuntimeProfileLogger.log(
                logger,
                "measurement_object_labels_completion_payload",
                completion_timer.elapsed(),
                reference_domain=request.reference_domain.value,
                payload_reused=completion_payload is source_projected_payload,
            )
        return cls(
            request=request,
            source_payload=source_payload,
            source_projected_payload=source_projected_payload,
            source_projected_labels=source_projected_labels,
            aligned_source=aligned_source,
            measurement_labels=measurement_labels,
            completion_payload=completion_payload,
        )

    @staticmethod
    def source_context_payload(
        request: MeasurementImageLabelAlignmentRequest,
        source_payload: ObjectLabelValue,
        source_projected_labels: ObjectLabelMeasurementSource,
    ) -> ObjectLabelValue:
        """Attach measurement-image source context to projected object labels."""
        source_variants = source_payload.variant_data
        if isinstance(source_projected_labels, ObjectLabelValue):
            variants = source_projected_labels.variant_data
        elif source_projected_labels is request.labels:
            variants = source_variants
        else:
            variants = ObjectLabelVariantData.compatible_replacement(
                source_payload,
                source_projected_labels,
            )
        metadata = image_payload_metadata(request.image)
        payload_domain = source_payload.object_label_source_spatial_domain()
        source_spatial_domain = (
            metadata.object_label_source_spatial_domain()
            .with_missing_from(payload_domain)
            .with_fill_value(payload_domain.fill_value)
            .with_value_name(payload_domain.value_name)
        )
        if (
            variants.labels is source_variants.labels
            and variants.unedited_labels is source_variants.unedited_labels
            and variants.small_removed_labels is source_variants.small_removed_labels
            and source_spatial_domain == payload_domain
        ):
            return source_payload
        if (
            source_projected_labels is request.labels
            and source_spatial_domain == payload_domain
        ):
            return source_payload.with_variants(
                variants,
                source_spatial_domain=source_spatial_domain,
            )
        if source_spatial_domain != payload_domain:
            contextual_payload = source_payload.with_variants(
                variants,
                source_spatial_domain=source_spatial_domain,
            )
            image_source_domain = measurement_image_source_spatial_adapter(
                request.image
            )
            if image_source_domain is None:
                raise ValueError(
                    "Measurement source-spatial projection requires an image source domain."
                )
            contextual_source_domain = SourceSpatialDomainAdapter.for_value(
                contextual_payload
            )
            if contextual_source_domain is None:
                raise TypeError(
                    "Measurement source-spatial projection requires an object-label "
                    "source domain adapter."
                )
            variants = ObjectLabelVariantData.compatible_replacement(
                contextual_payload,
                image_source_domain.extract_source_array(
                    object_label_dense_array(contextual_payload),
                    spatial_axes_yx=contextual_source_domain.spatial_axes_yx,
                ),
            )
        return source_payload.with_variants(
            variants,
            representation=ObjectLabelRepresentation.DENSE_LABELS,
            source_spatial_domain=source_spatial_domain,
        )


def measurement_image_source_spatial_adapter(
    image: RuntimeArrayData | AlignedImageStack,
) -> SourceSpatialDomainAdapter | None:
    """Return the nominal source-domain adapter for a measurement image."""
    if isinstance(image, AlignedImageStack):
        return image.first_slice_source_spatial_adapter()
    return SourceSpatialDomainAdapter.for_value(image)


class MeasurementLabelSourceAlignmentStrategy:
    """Align labels through nominal image and source-spatial contracts."""

    @classmethod
    def align(
        cls,
        image: RuntimeArrayData | AlignedImageStack,
        labels: ObjectLabelMeasurementSource | RuntimeSliceAlignedValueSet,
        *,
        label_payload: ObjectLabelValue | None = None,
    ) -> ObjectLabelMeasurementSource | RuntimeSliceAlignedValueSet:
        if isinstance(labels, RuntimeSliceAlignedValueSet):
            if not isinstance(image, AlignedImageStack):
                raise ValueError(
                    "Runtime-slice-aligned labels require an AlignedImageStack "
                    "measurement image."
                )
            if labels.slice_count != len(image.slices):
                raise ValueError(
                    "Runtime-slice-aligned labels must match the aligned image "
                    f"count: {labels.slice_count} != {len(image.slices)}."
                )
            return labels
        if isinstance(image, AlignedImageStack):
            if isinstance(image, ImageOutputBundle):
                return labels
            if len(image.slices) == 1 and (
                label_payload is None or label_payload.declared_plane_count() is None
            ):
                return labels
            if (
                label_payload is None
                or label_payload.runtime_slice_plane_count() != len(image.slices)
            ):
                raise ValueError(
                    "AlignedImageStack measurement images require labels with a "
                    "declared plane-scoped runtime-slice axis."
                )
            return labels
        image_domain_adapter = measurement_image_source_spatial_adapter(image)
        if image_domain_adapter is None:
            return labels
        label_domain_adapter = SourceSpatialDomainAdapter.for_value(
            labels,
            source_shape_override_yx=image_domain_adapter.domain.source_shape_yx,
        )
        if label_domain_adapter is None:
            return labels
        return label_domain_adapter.value_in_payload_domain(image_domain_adapter)

    @classmethod
    def align_request_labels_to_image_source(
        cls,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> ObjectLabelMeasurementSource | RuntimeSliceAlignedValueSet:
        """Return request labels projected into the request image source domain."""
        request = request.with_source_projected_labels()
        return cls.align(
            request.image,
            request.labels,
            label_payload=request.label_payload,
        )


class MeasurementImageLabelAlignmentStrategy:
    """Select measurement-image alignment from existing nominal contracts."""

    @classmethod
    def align(
        cls,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> RuntimeArrayData | AlignedImageStack:
        request = request.with_source_projected_labels()
        if request.reference_domain is MeasurementImageReferenceDomain.SOURCE_IMAGE:
            request = request.with_source_projected_image()
            image = request.image
        elif request.reference_domain is MeasurementImageReferenceDomain.OBJECT_LABELS:
            image = cls.object_label_reference_image(request)
        else:
            raise ValueError(
                "Measurement image alignment has no declared reference-domain behavior "
                f"for {request.reference_domain!r}."
            )
        label_payload = request.label_payload
        image_runtime_projection = RuntimeSliceProjection.preserved_context_for_value(
            image
        )
        if (
            label_payload is not None
            and label_payload.object_label_domain().scope
            is ObjectLabelDomainScope.PAYLOAD
            and image_runtime_projection is not None
            and image_runtime_projection.axis_size == 1
        ):
            image = RuntimeSliceProjection.value_for_singleton_slice(
                image,
                source_description="Payload-scoped object-label measurement image",
            )
        aligned_labels = (
            MeasurementLabelSourceAlignmentStrategy.align(
                image,
                request.labels,
                label_payload=label_payload,
            )
            if request.reference_domain is MeasurementImageReferenceDomain.SOURCE_IMAGE
            else request.labels
        )
        cls.validate_alignment(
            image,
            aligned_labels,
            label_payload=label_payload,
        )
        return image

    @classmethod
    def object_label_reference_image(
        cls,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> RuntimeArrayData | AlignedImageStack:
        """Return an image selected by declared object-label-domain semantics."""
        label_payload = request.label_payload
        if label_payload is None:
            raise ValueError(
                "Object-label reference alignment requires an ObjectLabelValue "
                "declaring plane-axis and domain semantics."
            )
        if isinstance(request.image, AlignedImageStack):
            if label_payload.runtime_slice_plane_count() == len(request.image.slices):
                return request.image
            return label_payload.measurement_reference_image()
        domain = label_payload.object_label_domain()
        if domain.scope is not ObjectLabelDomainScope.PAYLOAD and not (
            domain.scope is ObjectLabelDomainScope.PLANE
            and label_payload.plane_axis
            in (RuntimePlaneAxis.RUNTIME_SLICE, RuntimePlaneAxis.SOURCE_BINDING)
        ):
            raise ValueError(
                "Object-label reference alignment has no behavior declared for "
                f"scope={domain.scope!r}, plane_axis={label_payload.plane_axis!r}."
            )
        label_domain_adapter = SourceSpatialDomainAdapter.for_value(label_payload)
        if label_domain_adapter is None:
            raise TypeError(
                "Object-label reference alignment requires a SourceSpatialDomainAdapter "
                f"for {type(label_payload).__name__}."
            )
        image_domain_adapter = measurement_image_source_spatial_adapter(request.image)
        if image_domain_adapter is None:
            raise TypeError(
                "Object-label reference alignment requires an image source domain "
                "adapter."
            )
        return with_image_payload_data(
            request.image,
            label_domain_adapter.extract_source_array(
                request.image_array,
                spatial_axes_yx=image_domain_adapter.spatial_axes_yx,
            ),
        )

    @classmethod
    def validate_alignment(
        cls,
        image: RuntimeArrayData | AlignedImageStack,
        labels: ObjectLabelMeasurementSource | RuntimeSliceAlignedValueSet,
        *,
        label_payload: ObjectLabelValue | None,
    ) -> None:
        """Validate shape only after nominal alignment behavior was selected."""
        if isinstance(image, AlignedImageStack):
            cls.validate_aligned_stack(image, labels, label_payload=label_payload)
            return
        if isinstance(labels, RuntimeSliceAlignedValueSet):
            raise ValueError(
                "Runtime-slice-aligned labels require an AlignedImageStack "
                "measurement image."
            )
        cls.validate_dense_pair(image, labels, label_payload=label_payload)

    @classmethod
    def validate_aligned_stack(
        cls,
        image: AlignedImageStack,
        labels: ObjectLabelMeasurementSource | RuntimeSliceAlignedValueSet,
        *,
        label_payload: ObjectLabelValue | None,
    ) -> None:
        """Validate explicitly aligned image and label slices pairwise."""
        slice_count = len(image.slices)
        if isinstance(image, ImageOutputBundle):
            label_slices = (labels,) * slice_count
        elif isinstance(labels, RuntimeSliceAlignedValueSet):
            if labels.slice_count != slice_count:
                raise ValueError(
                    "Runtime-slice-aligned labels must match the aligned image "
                    f"count: {labels.slice_count} != {slice_count}."
                )
            label_slices = tuple(
                labels.value_for_slice(index) for index in range(slice_count)
            )
        else:
            if (
                label_payload is None
                or label_payload.runtime_slice_plane_count() != slice_count
            ):
                raise ValueError(
                    "AlignedImageStack measurement images require labels with a "
                    "declared plane-scoped runtime-slice axis."
                )
            label_slices = tuple(
                label_payload.with_projected_plane(
                    object_label_project_plane(
                        label_payload.labels,
                        index,
                        plane_count=slice_count,
                    ),
                    index,
                )
                for index in range(slice_count)
            )
        for image_slice, label_slice in zip(
            image.slices,
            label_slices,
            strict=True,
        ):
            cls.validate_dense_pair(
                image_slice,
                label_slice,
                label_payload=label_payload,
            )

    @staticmethod
    def validate_dense_pair(
        image: object,
        labels: object,
        *,
        label_payload: ObjectLabelValue | None = None,
    ) -> None:
        """Require exact dense shapes after nominal source-domain projection."""
        image_data = image_payload_data(image)
        label_data = (
            object_label_dense_array(labels)
            if isinstance(labels, ObjectLabelValue)
            else labels
        )
        if not isinstance(image_data, np.ndarray) or not isinstance(
            label_data, np.ndarray
        ):
            raise TypeError(
                "Measurement image alignment requires dense image and label arrays "
                "after nominal projection; got "
                f"{type(image_data).__name__} and {type(label_data).__name__}."
            )
        if tuple(image_data.shape) != tuple(label_data.shape):
            label_domain = (
                None if label_payload is None else label_payload.object_label_domain()
            )
            raise ValueError(
                "Measurement image alignment produced incompatible declared domains: "
                f"image shape {image_data.shape!r}, label shape {label_data.shape!r}; "
                f"image plane axis={image_payload_metadata(image).plane_axis!r}; "
                f"label type={type(label_payload).__name__ if label_payload is not None else type(labels).__name__}; "
                f"label scope={None if label_domain is None else label_domain.scope!r}; "
                f"label plane axis={None if label_payload is None else label_payload.plane_axis!r}; "
                f"label source aliases={None if label_payload is None else label_payload.source_aliases!r}."
            )


class MeasurementImageReferenceDomain(str, Enum):
    """Semantic image domain used as the reference for object measurement."""

    SOURCE_IMAGE = "source_image"
    OBJECT_LABELS = "object_labels"
