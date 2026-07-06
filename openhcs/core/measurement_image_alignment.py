"""Nominal measurement-image alignment contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import logging
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.aligned_image_payload import AlignedImageStack
from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
    is_image_stack,
)
from openhcs.core.registry_strategies import (
    NominalTypeKeyedStrategyMixin,
    MostDerivedContextStrategyMixin,
)
from openhcs.core.runtime_semantics import (
    MeasurementImageReferenceDomain,
    ObjectLabelRepresentation,
    RegistryFamily,
    RegistryKeyAttribute,
    RuntimePlaneAxis,
    RuntimePlaneAxisProjectionRequest,
    RuntimePlaneAxisProjector,
    RuntimePlaneAxisValueProjection,
    coerce_enum,
)
from openhcs.core.runtime_profile import RuntimeProfileLogger, RuntimeProfileTimer
from openhcs.core.source_spatial_domain import SourceSpatialDomainAdapter
from openhcs.core.runtime_values import (
    ObjectLabelMeasurementPayloadStrategy,
    ObjectLabelMeasurementSource,
    ObjectLabelDenseDataStrategy,
    ObjectLabelReplacementRequest,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelSourcePlaneProjectionRequest,
    ObjectLabelSet,
    ObjectLabelValueConstructionContext,
    ObjectLabelValue,
    ObjectLabelVariantData,
    RuntimeArrayData,
    SingletonObjectLabelStackCollapseStrategy,
    image_payload_data,
    image_payload_metadata,
    object_label_dense_array,
)
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.core.source_plane_alignment import (
    SourcePayloadPlaneIdentity,
    SourcePayloadPlaneIdentitySequence,
    SourcePlaneIdentitySequenceAlignment,
)


logger = logging.getLogger(__name__)
MEASUREMENT_SOURCE_PLANE_IDENTITY_POLICY = SourceImageSetIdentityPolicy(
    plane_member_components=frozenset()
)


class MeasurementImageAlignmentContractNotDeclared(ValueError):
    """Raised when a specialized alignment contract does not own a request."""


class MeasurementImageMonochromeProjection(ABC):
    """Project supported multichannel measurement images into grayscale planes."""

    @abstractmethod
    def plane(self, payload: RuntimeArrayData, *, name: str) -> np.ndarray:
        """Return one grayscale measurement plane."""


@dataclass(frozen=True, slots=True)
class ReplicatedChannelMonochromeProjection(MeasurementImageMonochromeProjection):
    """Accept 2D planes and RGB/RGBA planes with identical visible channels."""

    def plane(self, payload: RuntimeArrayData, *, name: str) -> np.ndarray:
        array = self.collapse_singleton_plane_stack(np.asarray(payload))
        if array.ndim == 2:
            return array
        if is_color_image_slice(array) and array.shape[-1] >= 3:
            color = array[..., :3]
            if np.all(color == color[..., :1]):
                return array[..., 0]
        raise ValueError(
            f"Measurement image requires a 2-D grayscale {name} plane or replicated "
            f"RGB/RGBA grayscale plane, got shape {array.shape!r}."
        )

    @staticmethod
    def collapse_singleton_plane_stack(
        payload: RuntimeArrayData,
    ) -> RuntimeArrayData:
        """Collapse one-plane stacks before plane validation."""
        if isinstance(payload, np.ndarray) and payload.ndim == 3 and payload.shape[0] == 1:
            return payload[0]
        return payload


@dataclass(frozen=True, slots=True)
class ObjectLabelSourceSpatialCropAlignmentContract:
    """Declared source-domain projection from a full image to object-label labels."""

    image: np.ndarray
    labels: np.ndarray
    source_domain: SourceSpatialDomainAdapter

    @classmethod
    def from_request(
        cls,
        request: "MeasurementImageLabelAlignmentRequest",
    ) -> "ObjectLabelSourceSpatialCropAlignmentContract":
        """Build the crop contract or decline ownership at the semantic boundary."""
        if not request.has_array_pair:
            raise MeasurementImageAlignmentContractNotDeclared(
                "Object-label source-spatial crop alignment requires dense image and label arrays."
            )
        if request.reference_domain is not MeasurementImageReferenceDomain.OBJECT_LABELS:
            raise MeasurementImageAlignmentContractNotDeclared(
                "Object-label source-spatial crop alignment only owns object-label reference images."
            )

        image = request.image_array
        labels = request.label_array
        if tuple(image.shape) == tuple(labels.shape):
            raise MeasurementImageAlignmentContractNotDeclared(
                "Object-label source-spatial crop alignment only owns image/label shape mismatch."
            )

        label_payload = request.label_payload
        if label_payload is None:
            raise MeasurementImageAlignmentContractNotDeclared(
                "Object-label source-spatial crop alignment requires declared label payload placement."
            )

        source_domain = SourceSpatialDomainAdapter.for_value(
            label_payload,
            source_shape_override_yx=tuple(int(axis) for axis in image.shape[-2:]),
        )
        if source_domain is None:
            raise MeasurementImageAlignmentContractNotDeclared(
                "Object-label source-spatial crop alignment requires source-spatial label metadata."
            )
        return cls(
            image=image,
            labels=labels,
            source_domain=source_domain,
        )

    def project_image(self) -> np.ndarray:
        """Return image pixels projected into the object-label crop."""
        return np.asarray(self.source_domain.extract_source_array(self.image))


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
            labels=ObjectLabelDenseDataStrategy.for_payload(label_payload).data(
                label_payload
            ),
            label_payload=label_payload,
            plane_projector=plane_projector,
            align_image_to_labels=align_image_to_labels,
        )


@dataclass(slots=True)
class MeasurementImageLabelAlignmentRequest:
    """Typed facts required to align one measurement image to object labels."""

    source: MeasurementImageAlignmentSource
    labels: ObjectLabelMeasurementSource
    label_payload: ObjectLabelValue | None = None
    plane_projector: RuntimePlaneAxisProjector | None = None
    align_image_to_labels: bool = True
    monochrome_projection: MeasurementImageMonochromeProjection = field(
        default_factory=ReplicatedChannelMonochromeProjection
    )

    @property
    def has_array_pair(self) -> bool:
        """Return whether both image data and labels are dense NumPy arrays."""
        return isinstance(self.image_data, np.ndarray) and isinstance(self.labels, np.ndarray)

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
        return coerce_enum(
            MeasurementImageReferenceDomain,
            self.source.alignment_reference_domain,
            "MeasurementImageLabelAlignmentRequest.reference_domain",
        )

    @property
    def image_name(self) -> str:
        """Return the source-owned image name for diagnostics."""
        return self.source.alignment_image_name

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

    @property
    def label_array(self) -> np.ndarray:
        """Return dense labels after strategy ownership has validated them."""
        if not isinstance(self.labels, np.ndarray):
            raise TypeError(
                "Measurement image alignment requires ndarray labels, got "
                f"{type(self.labels).__name__}."
            )
        return self.labels

    def with_source_projected_image(self) -> "MeasurementImageLabelAlignmentRequest":
        """Return this request with only the measurement image source-projected."""
        projection = RuntimePlaneAxisValueProjection.from_projector(
            self.plane_projector,
            RuntimePlaneAxis.SOURCE_BINDING,
            self.source_aliases,
        )
        if projection is None:
            return self
        return replace(
            self,
            source=self.source.with_alignment_image(projection.project(self.image)),
        )

    def with_source_projected_labels(self) -> "MeasurementImageLabelAlignmentRequest":
        """Return this request with only object labels source-projected."""
        label_axis = (
            self.label_payload.plane_axis
            if self.label_payload is not None
            else RuntimePlaneAxis.SOURCE_BINDING
        )
        projection = RuntimePlaneAxisValueProjection.from_projector(
            self.plane_projector,
            label_axis,
            self.source_aliases,
        )
        if projection is None:
            return self.with_image_source_projected_labels()
        projected_labels = projection.project(self.labels)
        projected_payload = self.label_payload
        if self.label_payload is not None:
            payload_projection = projection.project(self.label_payload)
            if isinstance(payload_projection, ObjectLabelValue):
                projected_payload = payload_projection
                projected_labels = (
                    payload_projection
                    if isinstance(self.labels, ObjectLabelValue)
                    else object_label_dense_array(payload_projection)
                )
            elif projected_labels is not self.labels:
                projected_payload = (
                    ObjectLabelMeasurementPayloadStrategy.for_source(self.label_payload)
                    .materialize(
                        self.label_payload,
                        ObjectLabelReplacementRequest(projected_labels),
                    )
                )
        if projected_payload is not None:
            image_projected_payload = MeasurementImageSourcePlaneLabelProjection(
                image=self.image,
                label_payload=projected_payload,
            ).payload()
            if image_projected_payload is not projected_payload:
                projected_payload = image_projected_payload
                projected_labels = (
                    projected_payload
                    if isinstance(self.labels, ObjectLabelValue)
                    else object_label_dense_array(projected_payload)
                )
        return replace(
            self,
            labels=projected_labels,
            label_payload=projected_payload,
        )

    def with_image_source_projected_labels(self) -> "MeasurementImageLabelAlignmentRequest":
        """Return this request with labels projected to a planar image source plane."""
        if self.label_payload is None:
            return self
        projected_payload = MeasurementImageSourcePlaneLabelProjection(
            image=self.image,
            label_payload=self.label_payload,
        ).payload()
        if projected_payload is self.label_payload:
            return self
        return replace(
            self,
            labels=(
                projected_payload
                if isinstance(self.labels, ObjectLabelValue)
                else object_label_dense_array(projected_payload)
            ),
            label_payload=projected_payload,
        )


@dataclass(frozen=True, slots=True)
class MeasurementImageSourcePlaneLabelProjection:
    """Project label planes by matching their source identity to a planar image."""

    image: RuntimeArrayData | AlignedImageStack
    label_payload: ObjectLabelValue

    def payload(self) -> ObjectLabelValue:
        plane_index = self.label_plane_index()
        if plane_index is None:
            return self.label_payload
        labels = object_label_dense_array(self.label_payload)
        if not isinstance(labels, np.ndarray) or labels.ndim < 3:
            return self.label_payload
        return ObjectLabelMeasurementPayloadStrategy.for_source(
            self.label_payload
        ).materialize(
            self.label_payload,
            ObjectLabelSourcePlaneProjectionRequest(
                labels[plane_index],
                plane_index,
            ),
        )

    def label_plane_index(self) -> int | None:
        image_data = image_payload_data(self.image)
        if not isinstance(image_data, np.ndarray) or is_image_stack(image_data):
            return None
        image_identities = SourcePayloadPlaneIdentity.from_payload(
            self.image,
            MEASUREMENT_SOURCE_PLANE_IDENTITY_POLICY,
        ).identities()
        if not image_identities:
            return None
        label_identities = SourcePayloadPlaneIdentitySequence(
            self.label_payload,
            MEASUREMENT_SOURCE_PLANE_IDENTITY_POLICY,
        ).identities()
        if not label_identities:
            return None
        plane_indices = SourcePlaneIdentitySequenceAlignment(
            image_identities=(image_identities,),
            target_identities=label_identities,
        ).target_indexes_for_image_planes()
        if plane_indices is None or len(plane_indices) != 1:
            return None
        return plane_indices[0]


@dataclass(frozen=True, slots=True)
class PreparedMeasurementObjectLabels:
    """Single-pass object-label preparation for measurement-image execution."""

    request: MeasurementImageLabelAlignmentRequest
    source_payload: ObjectLabelValue
    source_projected_payload: ObjectLabelValue
    source_projected_labels: ObjectLabelMeasurementSource
    aligned_image: RuntimeArrayData | AlignedImageStack
    measurement_labels: ObjectLabelMeasurementSource
    completion_payload: ObjectLabelValue

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
            request,
            labels=source_projected_labels,
            label_payload=source_projected_payload,
        )
        image_align_timer = RuntimeProfileTimer.start()
        if request.align_image_to_labels:
            aligned_image = MeasurementImageLabelAlignmentStrategy.align(
                image_request
            )
        else:
            aligned_image = request.image
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
        measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(
            aligned_image,
            source_projected_labels,
            label_payload=source_projected_payload,
        )
        if profile_enabled:
            RuntimeProfileLogger.log(
                logger,
                "measurement_object_labels_label_alignment",
                label_align_timer.elapsed(),
                reference_domain=request.reference_domain.value,
                labels_reused=measurement_labels is source_projected_labels,
            )
        completion_timer = RuntimeProfileTimer.start()
        completion_payload = (
            ObjectLabelMeasurementPayloadStrategy.for_source(source_projected_payload)
            .materialize(
                source_projected_payload,
                ObjectLabelReplacementRequest(measurement_labels),
            )
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
            aligned_image=aligned_image,
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
        source_variants = ObjectLabelVariantData.from_value(source_payload)
        if source_projected_labels is request.labels:
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
        context = ObjectLabelValueConstructionContext.from_value(
            source_payload,
            source_provenance=source_payload.source_provenance,
            source_spatial_domain=source_spatial_domain,
        )
        if source_projected_labels is request.labels and source_spatial_domain == payload_domain:
            return source_payload.with_variants(context, variants)
        if source_spatial_domain != payload_domain:
            contextual_payload = source_payload.with_variants(context, variants)
            image_source_domain = measurement_image_source_spatial_adapter(request.image)
            if image_source_domain is None:
                raise ValueError(
                    "Measurement source-spatial projection requires an image source domain."
                )
            variants = ObjectLabelVariantData.compatible_replacement(
                contextual_payload,
                image_source_domain.extract_source_array(
                    object_label_dense_array(contextual_payload),
                ),
            )
        return context.value_from_variants(
            source_payload,
            variants,
            representation=ObjectLabelRepresentation.DENSE_LABELS,
        )


class MeasurementImageLabelAlignmentStrategy(
    MostDerivedContextStrategyMixin[MeasurementImageLabelAlignmentRequest],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered alignment semantics for measurement images and object labels."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True

    strategy_key: ClassVar[str | None] = None

    @classmethod
    def align(
        cls,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> RuntimeArrayData | AlignedImageStack:
        """Align the measurement image using the most-derived owning strategy."""
        strategy = cls.for_context(
            request,
            error_subject="Measurement image alignment",
        )
        if strategy is None:
            raise ValueError("Measurement image alignment requires a strategy.")
        return strategy.aligned_image(request)

    @abstractmethod
    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        """Return whether this strategy owns the alignment request."""

    @abstractmethod
    def aligned_image(
        self,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> RuntimeArrayData | AlignedImageStack:
        """Return the aligned measurement image."""


class MeasurementLabelSourceAlignmentStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Align measurement labels to the source domain of a measurement image."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def align(
        cls,
        image: RuntimeArrayData | AlignedImageStack,
        labels: ObjectLabelMeasurementSource,
        *,
        label_payload: ObjectLabelValue | None = None,
    ) -> ObjectLabelMeasurementSource:
        strategy = cls.for_nominal_value(image)
        if strategy is None:
            strategy = DefaultMeasurementLabelSourceAlignmentStrategy()
        return strategy.labels_for_image(
            image,
            labels,
            label_payload=label_payload,
        )

    @classmethod
    def align_request_labels_to_image_source(
        cls,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> ObjectLabelMeasurementSource:
        """Return request labels projected into the request image source domain."""
        request = request.with_source_projected_labels()
        return cls.align(
            request.image,
            request.labels,
            label_payload=request.label_payload,
        )

    @abstractmethod
    def labels_for_image(
        self,
        image: RuntimeArrayData | AlignedImageStack,
        labels: ObjectLabelMeasurementSource,
        *,
        label_payload: ObjectLabelValue | None = None,
    ) -> ObjectLabelMeasurementSource:
        """Return labels in the measurement image's execution domain."""


class DefaultMeasurementLabelSourceAlignmentStrategy(
    MeasurementLabelSourceAlignmentStrategy
):
    """Align labels directly to a non-aligned measurement image payload."""

    def labels_for_image(
        self,
        image: RuntimeArrayData | AlignedImageStack,
        labels: ObjectLabelMeasurementSource,
        *,
        label_payload: ObjectLabelValue | None = None,
    ) -> ObjectLabelMeasurementSource:
        if self.preserves_runtime_slice_label_stack(image, label_payload):
            return SingletonObjectLabelStackCollapseStrategy.for_labels(labels).collapse(
                labels
            )
        return self.source_aligned_labels(image, labels)

    @staticmethod
    def source_aligned_labels(
        image: RuntimeArrayData | AlignedImageStack,
        labels: ObjectLabelMeasurementSource,
    ) -> ObjectLabelMeasurementSource:
        labels = SingletonObjectLabelStackCollapseStrategy.for_labels(labels).collapse(
            labels
        )
        image_array = np.asarray(image)
        label_array = np.asarray(labels)
        if image_array.ndim == 0 or label_array.ndim == 0:
            return labels
        image_domain_adapter = measurement_image_source_spatial_adapter(image)
        if image_domain_adapter is not None:
            labels = image_domain_adapter.extract_source_array(labels)
        return collapse_repeated_label_stack_for_image(image, labels)

    @staticmethod
    def preserves_runtime_slice_label_stack(
        image: RuntimeArrayData | AlignedImageStack,
        label_payload: ObjectLabelValue | None,
    ) -> bool:
        """Return whether dense labels carry the image runtime-slice axis."""
        image_data = image_payload_data(image)
        if not isinstance(image_data, np.ndarray) or not is_image_stack(image_data):
            return False
        return ObjectLabelRuntimeSliceStackContract.preserves_runtime_slice_stack(
            label_payload,
            slice_count=int(image_data.shape[0]),
        )


class AlignedStackMeasurementLabelSourceAlignmentStrategy(
    DefaultMeasurementLabelSourceAlignmentStrategy
):
    """Preserve runtime-slice labels until aligned image slices are selected."""

    value_type = AlignedImageStack

    def labels_for_image(
        self,
        image: RuntimeArrayData | AlignedImageStack,
        labels: ObjectLabelMeasurementSource,
        *,
        label_payload: ObjectLabelValue | None = None,
    ) -> ObjectLabelMeasurementSource:
        if not isinstance(image, AlignedImageStack):
            raise TypeError(
                "AlignedStackMeasurementLabelSourceAlignmentStrategy requires "
                f"AlignedImageStack, got {type(image).__name__}."
            )
        if (
            label_payload is not None
            and ObjectLabelRuntimeSliceStackContract.preserves_runtime_slice_stack(
                label_payload,
                slice_count=len(image.slices),
            )
        ):
            return SingletonObjectLabelStackCollapseStrategy.for_labels(labels).collapse(
                labels
            )
        return self.source_aligned_labels(image, labels)


def measurement_image_source_spatial_adapter(
    image: RuntimeArrayData | AlignedImageStack,
) -> SourceSpatialDomainAdapter | None:
    """Return the nominal source-domain adapter for a measurement image."""
    if isinstance(image, AlignedImageStack):
        return image.first_slice_source_spatial_adapter()
    return SourceSpatialDomainAdapter.for_value(image)


def collapse_repeated_label_stack_for_image(
    image: RuntimeArrayData | AlignedImageStack,
    labels: ObjectLabelMeasurementSource,
) -> ObjectLabelMeasurementSource:
    """Collapse channel-broadcast object labels before measurement calls."""
    if not isinstance(image, np.ndarray) or not isinstance(labels, np.ndarray):
        return labels
    if not is_image_stack(image) or labels.ndim != image.ndim:
        return labels
    if tuple(labels.shape[1:]) != tuple(image.shape[1:]):
        return labels
    if labels.shape[0] == 0:
        return labels
    first_plane = labels[0]
    if all(
        np.array_equal(first_plane, labels[index])
        for index in range(1, labels.shape[0])
    ):
        return first_plane
    return labels


class PayloadMeasurementImageLabelAlignmentStrategy(MeasurementImageLabelAlignmentStrategy):
    """Fallback preserving non-array measurement payloads."""

    strategy_key = "payload"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return True

    def aligned_image(
        self,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> RuntimeArrayData | AlignedImageStack:
        return request.image


class ArrayMeasurementImageLabelAlignmentStrategy(
    PayloadMeasurementImageLabelAlignmentStrategy
):
    """Default dense-array alignment behavior."""

    strategy_key = "array"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return request.has_array_pair

    def aligned_image(
        self,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> np.ndarray:
        return self.object_label_shape_repaired(
            request,
            self.project_image(request),
        )

    def project_image(self, request: MeasurementImageLabelAlignmentRequest) -> np.ndarray:
        """Project image data into the label execution rank."""
        return request.image_array

    def object_label_shape_repaired(
        self,
        request: MeasurementImageLabelAlignmentRequest,
        image: np.ndarray,
    ) -> np.ndarray:
        """Return a label-domain reference image when object labels define the shape."""
        if request.reference_domain is not MeasurementImageReferenceDomain.OBJECT_LABELS:
            return image
        labels = request.label_array
        if tuple(image.shape) == tuple(labels.shape):
            return image
        if (
            labels.ndim > image.ndim
            and tuple(image.shape) == tuple(labels.shape[-image.ndim :])
        ):
            return image
        return np.zeros(tuple(labels.shape), dtype=image.dtype)


class MatchingRankMeasurementImageLabelAlignmentStrategy(
    ArrayMeasurementImageLabelAlignmentStrategy
):
    """Use dense image data directly when image and label ranks already agree."""

    strategy_key = "matching_rank"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return request.has_array_pair and request.image_array.ndim == request.label_array.ndim


class ObjectLabelLeadingPlaneMeasurementImageLabelAlignmentStrategy(
    ArrayMeasurementImageLabelAlignmentStrategy
):
    """Use the leading plane when object labels define a lower-rank image domain."""

    strategy_key = "object_label_leading_plane"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return (
            request.has_array_pair
            and request.reference_domain is MeasurementImageReferenceDomain.OBJECT_LABELS
            and request.image_array.ndim == request.label_array.ndim + 1
            and request.image_array.shape[0] >= 1
        )

    def project_image(self, request: MeasurementImageLabelAlignmentRequest) -> np.ndarray:
        return request.image_array[0]


class SourceBindingPlaneMeasurementImageLabelAlignmentStrategy(
    ArrayMeasurementImageLabelAlignmentStrategy
):
    """Select the source-binding plane that owns this measurement image alias."""

    strategy_key = "source_binding_plane"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return (
            request.has_array_pair
            and request.reference_domain is MeasurementImageReferenceDomain.SOURCE_IMAGE
            and request.plane_projector is not None
            and request.source_aliases
            and request.image_array.ndim == request.label_array.ndim + 1
            and self.plane_index(request) is not None
        )

    def aligned_image(
        self,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> RuntimeArrayData | AlignedImageStack:
        projected_request = request.with_source_projected_image()
        if projected_request is request:
            raise RuntimeError("Source-binding plane strategy matched without projection.")
        return projected_request.image

    @staticmethod
    def plane_index(request: MeasurementImageLabelAlignmentRequest) -> int | None:
        if request.plane_projector is None:
            return None
        return request.plane_projector.plane_index_for_axis(
            RuntimePlaneAxisProjectionRequest(
                axis=RuntimePlaneAxis.SOURCE_BINDING,
                source_aliases=request.source_aliases,
            )
        )


class ColorImageStackMeasurementImageLabelAlignmentStrategy(
    ObjectLabelLeadingPlaneMeasurementImageLabelAlignmentStrategy
):
    """Base for color image-stack alignment policies keyed by label rank."""

    label_rank: ClassVar[int | None] = None

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return (
            request.has_array_pair
            and is_color_image_stack(request.image_array)
            and request.label_array.ndim == self.label_rank
        )


class StackLabelColorImageStackMeasurementImageLabelAlignmentStrategy(
    ColorImageStackMeasurementImageLabelAlignmentStrategy
):
    """Project every color image stack plane when labels are stack-shaped."""

    strategy_key = "color_image_stack_labels"
    label_rank = 3

    def project_image(self, request: MeasurementImageLabelAlignmentRequest) -> np.ndarray:
        image = request.image_array
        return np.stack(
            tuple(
                request.monochrome_projection.plane(
                    image[index],
                    name=request.image_name,
                )
                for index in range(image.shape[0])
            )
        )


class PlaneLabelColorImageStackMeasurementImageLabelAlignmentStrategy(
    ColorImageStackMeasurementImageLabelAlignmentStrategy
):
    """Project the leading color image plane when labels are planar."""

    strategy_key = "color_image_stack_plane_labels"
    label_rank = 2

    def project_image(self, request: MeasurementImageLabelAlignmentRequest) -> np.ndarray:
        return request.monochrome_projection.plane(
            request.image_array[0],
            name=request.image_name,
        )


class ColorImageSliceMeasurementImageLabelAlignmentStrategy(
    MatchingRankMeasurementImageLabelAlignmentStrategy
):
    """Project color image slices to grayscale when labels are planar."""

    strategy_key = "color_image_slice"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return (
            request.has_array_pair
            and is_color_image_slice(request.image_array)
            and request.label_array.ndim == 2
        )

    def project_image(self, request: MeasurementImageLabelAlignmentRequest) -> np.ndarray:
        return request.monochrome_projection.plane(
            request.image_array,
            name=request.image_name,
        )


class ObjectLabelPlanarStackMeasurementImageLabelAlignmentStrategy(
    ObjectLabelLeadingPlaneMeasurementImageLabelAlignmentStrategy
):
    """Preserve or project image stacks when one planar object-label image is measured."""

    strategy_key = "object_label_planar_stack"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return (
            request.has_array_pair
            and request.reference_domain is MeasurementImageReferenceDomain.OBJECT_LABELS
            and is_image_stack(request.image_array)
            and request.label_array.ndim == 2
        )

    def aligned_image(
        self,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> np.ndarray:
        image = request.image_array
        labels = request.label_array
        if tuple(image.shape[1:]) == tuple(labels.shape):
            return image
        return self.object_label_shape_repaired(request, image[0])


class ObjectLabelSourceSpatialCropMeasurementImageLabelAlignmentStrategy(
    MatchingRankMeasurementImageLabelAlignmentStrategy,
    ObjectLabelPlanarStackMeasurementImageLabelAlignmentStrategy
):
    """Project a source-domain image into the declared object-label crop."""

    strategy_key = "object_label_source_spatial_crop"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        try:
            self.contract(request)
        except MeasurementImageAlignmentContractNotDeclared:
            return False
        return True

    def aligned_image(
        self,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> np.ndarray:
        image = self.contract(request).project_image()
        labels = request.label_array
        if (
            image.ndim == labels.ndim + 1
            and tuple(image.shape[1:]) == tuple(labels.shape)
        ):
            return image
        return self.object_label_shape_repaired(request, image)

    def contract(
        self,
        request: MeasurementImageLabelAlignmentRequest,
    ) -> ObjectLabelSourceSpatialCropAlignmentContract:
        """Return the declared source-spatial crop contract for this strategy."""
        return ObjectLabelSourceSpatialCropAlignmentContract.from_request(request)


class ColorObjectLabelPlanarStackMeasurementImageLabelAlignmentStrategy(
    ObjectLabelPlanarStackMeasurementImageLabelAlignmentStrategy,
    ColorImageStackMeasurementImageLabelAlignmentStrategy,
):
    """Use object-label planar-stack semantics for color stacks in object-label domain."""

    strategy_key = "color_object_label_planar_stack"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return (
            ObjectLabelPlanarStackMeasurementImageLabelAlignmentStrategy.matches(
                self,
                request,
            )
            and request.has_array_pair
            and is_color_image_stack(request.image_array)
        )


def prepare_measurement_image_alignment_strategies() -> None:
    """Warm registered measurement-image alignment strategy families."""

    MeasurementImageLabelAlignmentStrategy.registered_strategy_types()
    MeasurementLabelSourceAlignmentStrategy.registered_strategy_types()
