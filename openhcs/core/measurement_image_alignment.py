"""Nominal measurement-image alignment contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
    is_image_stack,
)
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.core.runtime_semantics import (
    MeasurementImageReferenceDomain,
    SourceSpatialDomainAdapter,
    coerce_enum,
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

    def project_image(self, request: MeasurementImageLabelAlignmentRequest) -> np.ndarray:
        return self.contract(request).project_image()

    def aligned_image(self, request: MeasurementImageLabelAlignmentRequest) -> Any:
        image = self.project_image(request)
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
