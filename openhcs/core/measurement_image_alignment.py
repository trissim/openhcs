"""Nominal measurement-image alignment contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

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
    RegistryFamily,
    RegistryKeyAttribute,
    RuntimePlaneAxis,
    RuntimePlaneAxisProjectionRequest,
    RuntimePlaneAxisProjector,
    SourceSpatialDomainAdapter,
    coerce_enum,
)
from openhcs.core.runtime_values import (
    ObjectLabelMeasurementPayloadStrategy,
    ObjectLabelPayload,
    ObjectLabelRuntimeSliceStackContract,
    ObjectLabelSet,
    ObjectLabelValue,
    SingletonObjectLabelStackCollapseStrategy,
    SourceImageContext,
    image_payload_data,
    object_label_dense_array,
)


class MeasurementImageAlignmentContractNotDeclared(ValueError):
    """Raised when a specialized alignment contract does not own a request."""


class MeasurementImageMonochromeProjection(ABC):
    """Project supported multichannel measurement images into grayscale planes."""

    @abstractmethod
    def plane(self, payload: Any, *, name: str) -> np.ndarray:
        """Return one grayscale measurement plane."""


@dataclass(frozen=True, slots=True)
class ReplicatedChannelMonochromeProjection(MeasurementImageMonochromeProjection):
    """Accept 2D planes and RGB/RGBA planes with identical visible channels."""

    def plane(self, payload: Any, *, name: str) -> np.ndarray:
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
    def collapse_singleton_plane_stack(payload: Any) -> Any:
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


@dataclass(frozen=True, slots=True)
class MeasurementImageLabelAlignmentRequest:
    """Typed facts required to align one measurement image to object labels."""

    image: Any
    image_data: Any
    labels: Any
    reference_domain: MeasurementImageReferenceDomain
    image_name: str = "measurement image"
    label_payload: Any | None = None
    plane_projector: RuntimePlaneAxisProjector | None = None
    source_image_names: tuple[str, ...] = ()
    monochrome_projection: MeasurementImageMonochromeProjection = field(
        default_factory=ReplicatedChannelMonochromeProjection
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_domain",
            coerce_enum(
                MeasurementImageReferenceDomain,
                self.reference_domain,
                "MeasurementImageLabelAlignmentRequest.reference_domain",
            ),
        )
        object.__setattr__(
            self,
            "source_image_names",
            tuple(str(name) for name in self.source_image_names),
        )

    @property
    def has_array_pair(self) -> bool:
        """Return whether both image data and labels are dense NumPy arrays."""
        return isinstance(self.image_data, np.ndarray) and isinstance(self.labels, np.ndarray)

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
    def align(cls, request: MeasurementImageLabelAlignmentRequest) -> Any:
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
    def aligned_image(self, request: MeasurementImageLabelAlignmentRequest) -> Any:
        """Return the aligned measurement image."""


@dataclass(frozen=True, slots=True)
class MeasurementDomainAlignmentRequest:
    """Two-stage alignment request for object-measurement image/label domains."""

    image: Any
    labels: Any
    label_payload: Any | None = None
    plane_projector: RuntimePlaneAxisProjector | None = None
    source_image_names: tuple[str, ...] = ()
    reference_domain: MeasurementImageReferenceDomain = (
        MeasurementImageReferenceDomain.SOURCE_IMAGE
    )

    def aligned_labels(self) -> Any:
        """Return labels projected into this measurement image source domain."""
        return MeasurementLabelSourceAlignmentStrategy.align(
            self.image,
            self.source_axis_projected_value(self.labels),
            label_payload=self.label_payload,
        )

    def aligned_image(self, labels: Any | None = None) -> Any:
        """Return the measurement image projected into the supplied label domain."""
        image = self.source_axis_projected_value(self.image)
        return MeasurementImageLabelAlignmentStrategy.align(
            MeasurementImageLabelAlignmentRequest(
                image=image,
                image_data=image_payload_data(image),
                labels=self.labels if labels is None else labels,
                label_payload=self.label_payload,
                plane_projector=self.plane_projector,
                source_image_names=self.source_image_names,
                reference_domain=self.reference_domain,
            )
        )

    def source_axis_projected_value(self, value: Any) -> Any:
        """Project dense values carrying the execution source-binding axis."""
        projection = self.source_axis_projection()
        if projection is None:
            return value
        return projection.project(value)

    def source_axis_projection(self) -> "SourceBindingAxisProjection | None":
        """Return the source-axis projection contract for this request."""
        if self.plane_projector is None or not self.source_image_names:
            return None
        if not hasattr(self.plane_projector, "plane_index_for_axis"):
            return None
        if not hasattr(self.plane_projector, "source_binding_axis_size"):
            return None
        plane_index = self.plane_projector.plane_index_for_axis(
            RuntimePlaneAxisProjectionRequest(
                axis=RuntimePlaneAxis.SOURCE_BINDING,
                source_aliases=self.source_image_names,
            )
        )
        axis_size = self.plane_projector.source_binding_axis_size(
            self.source_image_names,
        )
        if axis_size is None:
            return None
        return SourceBindingAxisProjection(
            source_aliases=self.source_image_names,
            plane_index=plane_index,
            axis_size=axis_size,
        )


@dataclass(frozen=True, slots=True)
class SourceBindingAxisProjection:
    """Projection of values that explicitly carry a source-binding leading axis."""

    source_aliases: tuple[str, ...]
    plane_index: int | None
    axis_size: int

    def project(self, value: Any) -> Any:
        """Return the selected source-binding plane when the value carries that axis."""
        plane_index = self.plane_index_for_value(value)
        if plane_index is None:
            return value
        if isinstance(value, (ObjectLabelPayload, ObjectLabelSet)):
            return self.project_object_labels(value, plane_index)
        return self.project_array_payload(value, plane_index)

    def plane_index_for_value(self, value: Any) -> int | None:
        """Return the selected source-binding plane for this value."""
        if (
            isinstance(value, SourceImageContext)
            and value.source_image_name in self.source_aliases
        ):
            return self.source_aliases.index(value.source_image_name)
        return self.plane_index

    def project_array_payload(self, value: Any, plane_index: int) -> Any:
        """Project image-like values carrying a source-binding leading axis."""
        data = image_payload_data(value)
        if not isinstance(data, np.ndarray):
            return value
        if not self.data_carries_axis(data):
            return value
        self.validate_plane_index(plane_index, data.shape)
        return data[plane_index]

    def project_object_labels(
        self,
        value: ObjectLabelValue,
        plane_index: int,
    ) -> ObjectLabelValue:
        """Project object labels while preserving their nominal metadata."""
        labels = object_label_dense_array(value)
        if not self.data_carries_axis(labels):
            return value
        self.validate_plane_index(plane_index, labels.shape)
        return ObjectLabelMeasurementPayloadStrategy.for_source(value).with_projected_plane(
            value,
            labels[plane_index],
            plane_index,
        )

    def data_carries_axis(self, data: np.ndarray) -> bool:
        """Return whether dense data explicitly carries this source-binding axis."""
        return data.ndim >= 3 and data.shape[0] == self.axis_size

    @staticmethod
    def validate_plane_index(plane_index: int, shape: tuple[int, ...]) -> None:
        """Validate a selected source-binding plane against dense data shape."""
        if plane_index < 0 or plane_index >= shape[0]:
            raise RuntimeError(
                "Source-binding axis projection produced an out-of-range plane "
                f"index {plane_index} for shape {shape!r}."
            )


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
        image: Any,
        labels: Any,
        *,
        label_payload: Any | None = None,
    ) -> Any:
        strategy = cls.for_nominal_value(image)
        if strategy is None:
            strategy = DefaultMeasurementLabelSourceAlignmentStrategy()
        return strategy.labels_for_image(
            image,
            labels,
            label_payload=label_payload,
        )

    @abstractmethod
    def labels_for_image(
        self,
        image: Any,
        labels: Any,
        *,
        label_payload: Any | None = None,
    ) -> Any:
        """Return labels in the measurement image's execution domain."""


class DefaultMeasurementLabelSourceAlignmentStrategy(
    MeasurementLabelSourceAlignmentStrategy
):
    """Align labels directly to a non-aligned measurement image payload."""

    def labels_for_image(
        self,
        image: Any,
        labels: Any,
        *,
        label_payload: Any | None = None,
    ) -> Any:
        if self.preserves_runtime_slice_label_stack(image, label_payload):
            return SingletonObjectLabelStackCollapseStrategy.for_labels(labels).collapse(
                labels
            )
        return self.source_aligned_labels(image, labels)

    @staticmethod
    def source_aligned_labels(image: Any, labels: Any) -> Any:
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
        image: Any,
        label_payload: Any | None,
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
        image: Any,
        labels: Any,
        *,
        label_payload: Any | None = None,
    ) -> Any:
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
    image: Any,
) -> SourceSpatialDomainAdapter | None:
    """Return the nominal source-domain adapter for a measurement image."""
    if isinstance(image, AlignedImageStack):
        return image.first_slice_source_spatial_adapter()
    return SourceSpatialDomainAdapter.for_value(image)


def collapse_repeated_label_stack_for_image(image: Any, labels: Any) -> Any:
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

    def aligned_image(self, request: MeasurementImageLabelAlignmentRequest) -> Any:
        return request.image


class ArrayMeasurementImageLabelAlignmentStrategy(
    PayloadMeasurementImageLabelAlignmentStrategy
):
    """Default dense-array alignment behavior."""

    strategy_key = "array"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return request.has_array_pair

    def aligned_image(self, request: MeasurementImageLabelAlignmentRequest) -> Any:
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
            and request.source_image_names
            and request.image_array.ndim == request.label_array.ndim + 1
            and self.plane_index(request) is not None
        )

    def project_image(self, request: MeasurementImageLabelAlignmentRequest) -> np.ndarray:
        plane_index = self.plane_index(request)
        if plane_index is None:
            raise RuntimeError("Source-binding plane strategy matched without an index.")
        if plane_index >= request.image_array.shape[0]:
            raise RuntimeError(
                "Source-binding measurement image projection produced an out-of-range "
                f"plane index {plane_index} for image shape {request.image_array.shape!r}."
            )
        return request.image_array[plane_index]

    @staticmethod
    def plane_index(request: MeasurementImageLabelAlignmentRequest) -> int | None:
        if request.plane_projector is None:
            return None
        return request.plane_projector.plane_index_for_axis(
            RuntimePlaneAxisProjectionRequest(
                axis=RuntimePlaneAxis.SOURCE_BINDING,
                source_aliases=request.source_image_names,
            )
        )


class ColorImageStackMeasurementImageLabelAlignmentStrategy(
    ObjectLabelLeadingPlaneMeasurementImageLabelAlignmentStrategy
):
    """Project color image stacks to grayscale measurement planes."""

    strategy_key = "color_image_stack"

    def matches(self, request: MeasurementImageLabelAlignmentRequest) -> bool:
        return request.has_array_pair and is_color_image_stack(request.image_array)

    def project_image(self, request: MeasurementImageLabelAlignmentRequest) -> np.ndarray:
        image = request.image_array
        labels = request.label_array
        if labels.ndim == 3:
            return np.stack(
                tuple(
                    request.monochrome_projection.plane(
                        image[index],
                        name=request.image_name,
                    )
                    for index in range(image.shape[0])
                )
            )
        if labels.ndim == 2:
            return request.monochrome_projection.plane(image[0], name=request.image_name)
        return image


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

    def aligned_image(self, request: MeasurementImageLabelAlignmentRequest) -> Any:
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

    def aligned_image(self, request: MeasurementImageLabelAlignmentRequest) -> Any:
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
            and ColorImageStackMeasurementImageLabelAlignmentStrategy.matches(
                self,
                request,
            )
        )
