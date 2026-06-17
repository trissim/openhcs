"""Typed runtime artifact values and validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field, replace
from enum import Enum
import logging
from operator import attrgetter
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Self, TypeVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
import numpy as np

try:
    from polystore.virtual_workspace import VirtualWorkspaceBackend
except ImportError:  # pragma: no cover - optional PolyStore backend
    VirtualWorkspaceBackend = None  # type: ignore[assignment]

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import (
    ArtifactKey,
    ArtifactKind,
    ArtifactOutputPlan,
    ArtifactPayloadShape,
)
from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
)
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementScope,
    MeasurementSubject,
    ObjectLabelDomain,
    ObjectLabelDomainDeclaration,
    ObjectLabelDomainMetadata,
    ObjectLabelDomainScope,
    ObjectLabelIdDomainStrategy,
    ObjectLabelPlaneDomainStrategy,
    ObjectLabelRepresentation,
    ObjectLabelVariant,
    PreserveSourceObjectLabelDomainDeclaration,
    RelationshipEndpoint,
    RelationshipSemantics,
    MeasurementObjectRowIdentity,
    RuntimePlaneAxis,
    SpatialGridOrigin,
    SpatialGridOrdering,
    DenseObjectLabelPairAligner,
    ExplicitObjectLabelDomainDeclaration,
    coerce_enum,
    dense_object_label_plane_id_domains,
    measurement_table_row_layout,
    measurement_table_row_layout_from_fields,
    normalize_measurement_table_rows,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    NominalTypeKeyedStrategyMixin,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet


PhysicalBorderEdgesYX = tuple[bool, bool, bool, bool] | None
SourceComponentMetadata = Mapping[str, Any]
ChannelSourceComponentMetadata = tuple[SourceComponentMetadata | None, ...]
MetadataValueT = TypeVar("MetadataValueT")
SliceAlignedValueT = TypeVar("SliceAlignedValueT")
SliceAlignedPayloadT = TypeVar("SliceAlignedPayloadT")
SourceProvenanceIdentity = tuple[str | None, tuple[tuple[str, str], ...] | None]
SourceProvenanceInitValues = tuple[
    str | None,
    SourceComponentMetadata | None,
    tuple[str | None, ...],
    ChannelSourceComponentMetadata,
    tuple[str, ...],
]

_TPayload = TypeVar("_TPayload", bound=type[Any])
_ARRAY_PAYLOAD_PREDICATES: list[Callable[[Any], bool]] = []
logger = logging.getLogger(__name__)


def _normalize_component_metadata(
    metadata: SourceComponentMetadata | None,
) -> SourceComponentMetadata | None:
    if metadata is None:
        return None
    return MappingProxyType(dict(metadata))


class RuntimeArrayPayload(ABC, metaclass=AutoRegisterMeta):
    """Nominal ABC for array payload types accepted by runtime artifacts."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True
    __array_priority__ = 1000

    @property
    @abstractmethod
    def shape(self) -> Any:
        ...

    @abstractmethod
    def array_payload_data(self) -> Any:
        ...

    @abstractmethod
    def with_data(self, data: Any) -> Self:
        ...

    def array_operand(self) -> Any:
        return self.array_payload_data()

    def array_ufunc_result(self, result: Any) -> Any:
        if isinstance(result, tuple):
            return tuple(self.array_ufunc_result(item) for item in result)
        if isinstance(result, np.ndarray) and np.issubdtype(result.dtype, np.bool_):
            return result
        if isinstance(result, np.ndarray):
            return self.with_data(result)
        return result

    def compare_array_payload(self, other: Any, ufunc: Any) -> Any:
        return ufunc(np.asarray(self), runtime_array_operand(other))

    def __lt__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.less)

    def __le__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.less_equal)

    def __gt__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.greater)

    def __ge__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.greater_equal)

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        converted_inputs = tuple(runtime_array_operand(value) for value in inputs)
        if "out" in kwargs:
            kwargs = {
                **kwargs,
                "out": tuple(runtime_array_operand(value) for value in kwargs["out"]),
            }
        result = getattr(ufunc, method)(*converted_inputs, **kwargs)
        return self.array_ufunc_result(result)


RuntimeArrayData = RuntimeArrayPayload | np.ndarray
ImagePayloadMetadataInput = RuntimeArrayData


class SourceImageProvenance:
    """Source-image identity carried by image and object-label payloads."""

    __slots__ = (
        "source_path",
        "source_component_metadata",
        "channel_source_paths",
        "channel_source_component_metadata",
        "source_image_names",
    )

    def __init__(
        self,
        source_path: str | None = None,
        source_component_metadata: SourceComponentMetadata | None = None,
        channel_source_paths: tuple[str | None, ...] = (),
        channel_source_component_metadata: ChannelSourceComponentMetadata = (),
        source_image_names: tuple[str, ...] = (),
    ) -> None:
        self.source_path = self.normalized_source_path(source_path)
        self.source_component_metadata = _normalize_component_metadata(
            source_component_metadata
        )
        self.channel_source_paths = tuple(
            self.normalized_source_path(path) for path in channel_source_paths
        )
        self.channel_source_component_metadata = tuple(
            _normalize_component_metadata(metadata)
            for metadata in channel_source_component_metadata
        )
        self.source_image_names = tuple(str(name) for name in source_image_names)

    @classmethod
    def from_init_values(
        cls,
        values: SourceProvenanceInitValues,
    ) -> "SourceImageProvenance":
        return cls(*values)

    @staticmethod
    def normalized_source_path(source_path: str | None) -> str | None:
        if source_path is None:
            return None
        return str(source_path)

    @property
    def has_values(self) -> bool:
        return any(
            (
                self.source_path is not None,
                self.source_component_metadata is not None,
                bool(self.channel_source_paths),
                bool(self.channel_source_component_metadata),
                bool(self.source_image_names),
            )
        )

    @property
    def plane_count_sources(self) -> tuple[int, ...]:
        return tuple(
            count
            for count in (
                len(self.channel_source_paths),
                len(self.channel_source_component_metadata),
            )
            if count > 0
        )

    @property
    def channel_plane_count(self) -> int:
        return max(self.plane_count_sources, default=0)

    @property
    def addressable(self) -> bool:
        return self.source_path is not None or self.source_component_metadata is not None

    def for_channel(self, channel_index: int) -> "SourceImageProvenance":
        source_path = _tuple_value(self.channel_source_paths, channel_index)
        source_component_metadata = _tuple_value(
            self.channel_source_component_metadata,
            channel_index,
        )
        return type(self)(
            source_path=source_path if source_path is not None else self.source_path,
            source_component_metadata=(
                source_component_metadata
                if source_component_metadata is not None
                else self.source_component_metadata
            ),
            source_image_names=self.source_image_names_for_channel(channel_index),
        )

    def source_image_names_for_channel(self, channel_index: int) -> tuple[str, ...]:
        """Return source-image aliases represented by a selected provenance plane."""
        if len(self.source_image_names) <= channel_index:
            return self.source_image_names
        return (self.source_image_names[channel_index],)

    def identity(self) -> SourceProvenanceIdentity:
        return (
            self.source_path,
            self.source_component_identity,
        )

    def with_missing_from(self, fallback: "SourceImageProvenance") -> Self:
        return type(self)(
            source_path=(
                self.source_path
                if self.source_path is not None
                else fallback.source_path
            ),
            source_component_metadata=(
                self.source_component_metadata
                if self.source_component_metadata is not None
                else fallback.source_component_metadata
            ),
            channel_source_paths=self.merged_channel_values(
                self.channel_source_paths,
                fallback.channel_source_paths,
            ),
            channel_source_component_metadata=self.merged_channel_values(
                self.channel_source_component_metadata,
                fallback.channel_source_component_metadata,
            ),
            source_image_names=(
                self.source_image_names
                if self.source_image_names
                else fallback.source_image_names
            ),
        )

    @staticmethod
    def merged_channel_values(
        values: tuple[MetadataValueT | None, ...],
        fallback: tuple[MetadataValueT | None, ...],
    ) -> tuple[MetadataValueT | None, ...]:
        """Fill missing per-channel identities without changing cardinality."""
        if not values:
            return fallback
        if not fallback or len(values) != len(fallback):
            return values
        return tuple(
            value if value is not None else fallback_value
            for value, fallback_value in zip(values, fallback)
        )

    @property
    def source_component_identity(self) -> tuple[tuple[str, str], ...] | None:
        if self.source_component_metadata is None:
            return None
        return tuple(
            sorted(
                (str(key), repr(value))
                for key, value in self.source_component_metadata.items()
            )
        )


@dataclass(frozen=True, slots=True)
class SourceImageProvenanceRequirement(ABC, metaclass=AutoRegisterMeta):
    """Nominal source-provenance invariant with policy-selected failure behavior."""

    __registry_key__ = "requirement_name"
    __skip_if_no_key__ = True
    requirement_name: ClassVar[str | None] = None

    provenance: SourceImageProvenance
    label_name: str

    @property
    @abstractmethod
    def satisfied(self) -> bool:
        """Return whether this source-provenance invariant is satisfied."""

    @abstractmethod
    def error(self) -> ValueError:
        """Return the fail-loud error for an unsatisfied invariant."""

    def validate(self) -> None:
        SourceImageProvenanceRequirementPolicy.for_requirement(self).validate(self)


class SourceImageProvenanceRequirementPolicy(ABC, metaclass=AutoRegisterMeta):
    """Registered satisfied/unsatisfied handling for provenance requirements."""

    __registry_key__ = "satisfied"
    __skip_if_no_key__ = True
    satisfied: ClassVar[bool | None] = None

    @classmethod
    def for_requirement(
        cls,
        requirement: SourceImageProvenanceRequirement,
    ) -> "SourceImageProvenanceRequirementPolicy":
        return cls.__registry__[requirement.satisfied]()

    def validate(self, requirement: SourceImageProvenanceRequirement) -> None:
        """Validate one source-provenance requirement."""
        return None


class SatisfiedSourceImageProvenanceRequirementPolicy(
    SourceImageProvenanceRequirementPolicy
):
    """Accept source-provenance requirements that are already satisfied."""

    satisfied = True


class UnsatisfiedSourceImageProvenanceRequirementPolicy(
    SourceImageProvenanceRequirementPolicy
):
    """Raise the requirement-specific error for failed provenance invariants."""

    satisfied = False

    def validate(self, requirement: SourceImageProvenanceRequirement) -> None:
        raise requirement.error()


@dataclass(frozen=True, slots=True)
class SourceImageProvenancePlaneCountRequirement(SourceImageProvenanceRequirement):
    """Require per-plane source identities to match source-aligned label planes."""

    requirement_name = "plane_count"
    expected_count: int

    @property
    def satisfied(self) -> bool:
        return self.provenance.channel_plane_count == self.expected_count

    def error(self) -> ValueError:
        return ValueError(
            f"Object-label artifact {self.label_name!r} has {self.expected_count} "
            "source-aligned label planes but "
            f"{self.provenance.channel_plane_count} per-plane source identities "
            "after source-image contextualization."
        )


@dataclass(frozen=True, slots=True)
class SourceImageProvenanceAddressRequirement(SourceImageProvenanceRequirement):
    """Require one plane source identity to be addressable."""

    requirement_name = "address"
    plane_index: int

    @property
    def satisfied(self) -> bool:
        return self.provenance.addressable

    def error(self) -> ValueError:
        return ValueError(
            f"Object-label artifact {self.label_name!r} plane {self.plane_index} "
            "has neither source_path nor source_component_metadata after "
            "source-image contextualization."
        )


@dataclass(kw_only=True)
class SourceImageProvenanceFields:
    """Source-image provenance fields shared by image and object-label carriers."""

    source_provenance: SourceImageProvenance = field(
        default_factory=SourceImageProvenance
    )
    source_path: InitVar[str | None] = None
    source_component_metadata: InitVar[SourceComponentMetadata | None] = None
    channel_source_paths: InitVar[tuple[str | None, ...]] = ()
    channel_source_component_metadata: InitVar[ChannelSourceComponentMetadata] = ()
    source_image_names: InitVar[tuple[str, ...]] = ()

    def absorb_explicit_source_provenance(
        self,
        explicit: SourceImageProvenance,
    ) -> None:
        self.source_provenance = explicit.with_missing_from(self.source_provenance)

    def normalize_source_provenance_fields(self) -> None:
        self.source_provenance = SourceImageProvenance(
            self.source_provenance.source_path,
            self.source_provenance.source_component_metadata,
            self.source_provenance.channel_source_paths,
            self.source_provenance.channel_source_component_metadata,
            self.source_provenance.source_image_names,
        )

def _source_path_getter(self: SourceImageProvenanceFields) -> str | None:
    return self.source_provenance.source_path


def _source_path_setter(
    self: SourceImageProvenanceFields,
    value: str | None,
) -> None:
    self.source_provenance = SourceImageProvenance(
        source_path=value,
        source_component_metadata=self.source_provenance.source_component_metadata,
        channel_source_paths=self.source_provenance.channel_source_paths,
        channel_source_component_metadata=(
            self.source_provenance.channel_source_component_metadata
        ),
        source_image_names=self.source_provenance.source_image_names,
    )


def _source_component_metadata_getter(
    self: SourceImageProvenanceFields,
) -> SourceComponentMetadata | None:
    return self.source_provenance.source_component_metadata


def _source_component_metadata_setter(
    self: SourceImageProvenanceFields,
    value: SourceComponentMetadata | None,
) -> None:
    self.source_provenance = SourceImageProvenance(
        source_path=self.source_provenance.source_path,
        source_component_metadata=value,
        channel_source_paths=self.source_provenance.channel_source_paths,
        channel_source_component_metadata=(
            self.source_provenance.channel_source_component_metadata
        ),
        source_image_names=self.source_provenance.source_image_names,
    )


def _channel_source_paths_getter(
    self: SourceImageProvenanceFields,
) -> tuple[str | None, ...]:
    return self.source_provenance.channel_source_paths


def _channel_source_paths_setter(
    self: SourceImageProvenanceFields,
    value: tuple[str | None, ...],
) -> None:
    self.source_provenance = SourceImageProvenance(
        source_path=self.source_provenance.source_path,
        source_component_metadata=self.source_provenance.source_component_metadata,
        channel_source_paths=value,
        channel_source_component_metadata=(
            self.source_provenance.channel_source_component_metadata
        ),
        source_image_names=self.source_provenance.source_image_names,
    )


def _channel_source_component_metadata_getter(
    self: SourceImageProvenanceFields,
) -> ChannelSourceComponentMetadata:
    return self.source_provenance.channel_source_component_metadata


def _channel_source_component_metadata_setter(
    self: SourceImageProvenanceFields,
    value: ChannelSourceComponentMetadata,
) -> None:
    self.source_provenance = SourceImageProvenance(
        source_path=self.source_provenance.source_path,
        source_component_metadata=self.source_provenance.source_component_metadata,
        channel_source_paths=self.source_provenance.channel_source_paths,
        channel_source_component_metadata=value,
        source_image_names=self.source_provenance.source_image_names,
    )


def _source_image_names_getter(self: SourceImageProvenanceFields) -> tuple[str, ...]:
    return self.source_provenance.source_image_names


def _source_image_names_setter(
    self: SourceImageProvenanceFields,
    value: tuple[str, ...],
) -> None:
    self.source_provenance = SourceImageProvenance(
        source_path=self.source_provenance.source_path,
        source_component_metadata=self.source_provenance.source_component_metadata,
        channel_source_paths=self.source_provenance.channel_source_paths,
        channel_source_component_metadata=(
            self.source_provenance.channel_source_component_metadata
        ),
        source_image_names=value,
    )


SourceImageProvenanceFields.source_path = property(
    _source_path_getter,
    _source_path_setter,
)
SourceImageProvenanceFields.source_component_metadata = property(
    _source_component_metadata_getter,
    _source_component_metadata_setter,
)
SourceImageProvenanceFields.channel_source_paths = property(
    _channel_source_paths_getter,
    _channel_source_paths_setter,
)
SourceImageProvenanceFields.channel_source_component_metadata = property(
    _channel_source_component_metadata_getter,
    _channel_source_component_metadata_setter,
)
SourceImageProvenanceFields.source_image_names = property(
    _source_image_names_getter,
    _source_image_names_setter,
)


@dataclass(slots=True)
class ImagePayloadMetadata(SourceImageProvenanceFields):
    """Generic source-image metadata that should travel with runtime pixels."""

    intensity_scale: float | None = None
    source_dtype: str | None = None
    unit_interval_intensity_scale: int | None = None
    channel_intensity_scales: tuple[float | None, ...] = ()
    channel_source_dtypes: tuple[str | None, ...] = ()
    channel_unit_interval_intensity_scales: tuple[int | None, ...] = ()
    spatial_origin_yx: tuple[int, int] | None = None
    source_spatial_shape_yx: tuple[int, int] | None = None
    physical_border_edges_yx: PhysicalBorderEdgesYX = None
    mask_defines_border: bool | None = None

    @classmethod
    def for_array(
        cls,
        array: Any,
        *,
        source_path: str | None = None,
    ) -> "ImagePayloadMetadata":
        """Build metadata from an image array's source dtype."""
        import numpy as np

        dtype = np.asarray(array).dtype
        return cls(
            intensity_scale=image_intensity_scale_for_dtype(dtype),
            source_dtype=str(dtype),
            source_path=source_path,
        )

    @classmethod
    def for_array_payload(
        cls,
        array: Any,
        *,
        source_path: str | None = None,
    ) -> "ImagePayloadMetadata":
        """Build source metadata from an arraybridge-detectable payload."""
        from openhcs.core.memory import MEMORY_TYPE_NUMPY, detect_memory_type

        memory_type = detect_memory_type(array)
        dtype = getattr(array, "dtype", None)
        if dtype is None:
            raise TypeError(
                "ImagePayloadMetadata.for_array_payload requires an array payload "
                f"with dtype; got {type(array).__name__}."
            )
        return cls(
            intensity_scale=(
                image_intensity_scale_for_dtype(dtype)
                if memory_type == MEMORY_TYPE_NUMPY
                else None
            ),
            source_dtype=str(dtype),
            source_path=source_path,
        )

    def __post_init__(self, *source_provenance_values: object) -> None:
        self.absorb_explicit_source_provenance(
            SourceImageProvenance.from_init_values(source_provenance_values)
        )
        self.normalize_source_provenance_fields()

    @property
    def has_values(self) -> bool:
        """Return whether this metadata carries any semantic image facts."""
        return any(
            (
                self.intensity_scale is not None,
                self.source_dtype is not None,
                self.source_path is not None,
                self.source_component_metadata is not None,
                self.unit_interval_intensity_scale is not None,
                bool(self.channel_intensity_scales),
                bool(self.channel_source_dtypes),
                bool(self.channel_source_paths),
                bool(self.channel_source_component_metadata),
                bool(self.channel_unit_interval_intensity_scales),
                self.spatial_origin_yx is not None,
                self.source_spatial_shape_yx is not None,
                self.physical_border_edges_yx is not None,
                self.mask_defines_border is not None,
                bool(self.source_image_names),
            )
        )

    def intensity_scale_for_channel(self, channel_index: int) -> float | None:
        """Return the best available intensity scale for one channel/plane."""
        if 0 <= channel_index < len(self.channel_intensity_scales):
            channel_scale = self.channel_intensity_scales[channel_index]
            if channel_scale is not None:
                return channel_scale
        return self.intensity_scale

    def unit_interval_intensity_scale_for_channel(
        self,
        channel_index: int,
    ) -> int | None:
        """Return the scale proving current pixels are exact integer/scale values."""
        if 0 <= channel_index < len(self.channel_unit_interval_intensity_scales):
            channel_scale = self.channel_unit_interval_intensity_scales[channel_index]
            if channel_scale is not None:
                return int(channel_scale)
        return self.unit_interval_intensity_scale

    def payload_with(self, data: Any, mask: Any | None = None) -> Any:
        """Return image payload data carrying this metadata."""
        return image_payload_with_context(data=data, mask=mask, metadata=self)

    def for_channel(self, channel_index: int) -> "ImagePayloadMetadata":
        """Return single-channel metadata sliced from a stacked payload."""
        source_provenance = self.source_provenance.for_channel(channel_index)
        return ImagePayloadMetadata(
            intensity_scale=self.intensity_scale_for_channel(channel_index),
            source_dtype=_tuple_value(self.channel_source_dtypes, channel_index)
            or self.source_dtype,
            source_path=source_provenance.source_path,
            source_component_metadata=source_provenance.source_component_metadata,
            unit_interval_intensity_scale=(
                self.unit_interval_intensity_scale_for_channel(channel_index)
            ),
            spatial_origin_yx=self.spatial_origin_yx,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
            physical_border_edges_yx=self.physical_border_edges_yx,
            mask_defines_border=self.mask_defines_border,
            source_image_names=source_provenance.source_image_names,
        )

    def with_unit_interval_intensity_scale(
        self,
        scale: int | None,
    ) -> "ImagePayloadMetadata":
        """Return metadata with the current unit-interval pixel proof updated."""
        return replace(self, unit_interval_intensity_scale=scale)

    def without_unit_interval_intensity_scale(self) -> "ImagePayloadMetadata":
        """Return metadata after an arithmetic transform changed pixel values."""
        return replace(
            self,
            unit_interval_intensity_scale=None,
            channel_unit_interval_intensity_scales=(),
        )

    def without_spatial_domain(self) -> "ImagePayloadMetadata":
        """Return metadata with invalidated source-spatial placement removed."""
        return replace(
            self,
            spatial_origin_yx=None,
            source_spatial_shape_yx=None,
            physical_border_edges_yx=None,
            mask_defines_border=None,
        )

    def with_source_context_from(
        self,
        source: "ImagePayloadMetadata",
    ) -> "ImagePayloadMetadata":
        """Fill missing source-image identity and spatial context from a source."""
        source_provenance = self.source_provenance.with_missing_from(
            source.source_provenance
        )
        return replace(
            self,
            source_provenance=source_provenance,
            spatial_origin_yx=(
                self.spatial_origin_yx
                if self.spatial_origin_yx is not None
                else source.spatial_origin_yx
            ),
            source_spatial_shape_yx=(
                self.source_spatial_shape_yx
                if self.source_spatial_shape_yx is not None
                else source.source_spatial_shape_yx
            ),
            physical_border_edges_yx=(
                self.physical_border_edges_yx
                if self.physical_border_edges_yx is not None
                else source.physical_border_edges_yx
            ),
            mask_defines_border=(
                self.mask_defines_border
                if self.mask_defines_border is not None
                else source.mask_defines_border
            ),
        )

    def physical_border_edges_for_shape(
        self,
        image_shape_yx: Sequence[int],
    ) -> tuple[bool, bool, bool, bool]:
        """Return which local image edges are true source-image edges.

        Edge tuple order is ``(top, bottom, left, right)``. Missing spatial
        metadata means the current image is treated as the full physical image,
        preserving historical behavior for plain arrays.
        """
        if self.physical_border_edges_yx is not None:
            return tuple(bool(edge) for edge in self.physical_border_edges_yx)
        if self.spatial_origin_yx is None or self.source_spatial_shape_yx is None:
            return True, True, True, True

        height, width = _spatial_shape_pair(image_shape_yx, "image_shape_yx")
        origin_y, origin_x = self.spatial_origin_yx
        source_height, source_width = self.source_spatial_shape_yx
        return (
            origin_y <= 0,
            origin_y + height >= source_height,
            origin_x <= 0,
            origin_x + width >= source_width,
        )

    def with_spatial_crop(
        self,
        *,
        input_shape_yx: Sequence[int],
        output_shape_yx: Sequence[int],
        offset_yx: tuple[int, int],
        physical_border_edges_yx: PhysicalBorderEdgesYX = None,
    ) -> "ImagePayloadMetadata":
        """Return metadata for a crop of this image payload."""
        input_shape = _spatial_shape_pair(input_shape_yx, "input_shape_yx")
        output_shape = _spatial_shape_pair(output_shape_yx, "output_shape_yx")
        parent_origin = self.spatial_origin_yx or (0, 0)
        source_shape = self.source_spatial_shape_yx or input_shape
        origin = (
            int(parent_origin[0]) + int(offset_yx[0]),
            int(parent_origin[1]) + int(offset_yx[1]),
        )
        if physical_border_edges_yx is None:
            physical_border_edges_yx = (
                origin[0] <= 0,
                origin[0] + output_shape[0] >= source_shape[0],
                origin[1] <= 0,
                origin[1] + output_shape[1] >= source_shape[1],
            )
        return replace(
            self,
            spatial_origin_yx=origin,
            source_spatial_shape_yx=source_shape,
            physical_border_edges_yx=tuple(
                bool(edge) for edge in physical_border_edges_yx
            ),
        )


@dataclass(frozen=True, slots=True)
class ImageMetadataPayload(RuntimeArrayPayload):
    """Image data plus metadata, without requiring a validity mask."""

    data: Any
    metadata: ImagePayloadMetadata

    def __post_init__(self) -> None:
        if np.ndim(self.data) == 0:
            raise TypeError(
                "ImageMetadataPayload.data requires array-like image data, "
                f"got {type(self.data).__name__}."
            )
        if not self.metadata.has_values:
            raise ValueError("ImageMetadataPayload.metadata cannot be empty.")

    @property
    def shape(self) -> Any:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def __array__(self, dtype: Any | None = None) -> Any:
        import numpy as np

        return np.asarray(self.data, dtype=dtype)

    def array_payload_data(self) -> Any:
        return self.data

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.data, name)

    def with_data(
        self,
        data: Any,
        *,
        metadata: ImagePayloadMetadata | None = None,
    ) -> "ImageMetadataPayload":
        """Return the same metadata attached to replacement data."""
        return type(self)(
            data=data,
            metadata=self.metadata if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class MaskedImagePayload(RuntimeArrayPayload):
    """Image data plus an authoritative per-pixel validity mask."""

    data: Any
    mask: Any
    metadata: ImagePayloadMetadata = field(default_factory=ImagePayloadMetadata)

    def __post_init__(self) -> None:
        if np.ndim(self.data) == 0:
            raise TypeError(
                "MaskedImagePayload.data requires array-like image data, "
                f"got {type(self.data).__name__}."
            )
        mask_shape = tuple(np.shape(self.mask))
        if not mask_shape:
            raise TypeError(
                "MaskedImagePayload.mask requires array-like mask data, "
                f"got {type(self.mask).__name__}."
            )
        data_shape = tuple(np.shape(self.data))
        if not ImageMaskDomain(data_shape).accepts(mask_shape):
            raise ValueError(
                "MaskedImagePayload.mask shape must match the image spatial "
                f"domain; got mask {mask_shape!r} for image {data_shape!r}."
            )

    @property
    def shape(self) -> Any:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def __array__(self, dtype: Any | None = None) -> Any:
        import numpy as np

        return np.asarray(self.data, dtype=dtype)

    def array_payload_data(self) -> Any:
        return self.data

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.data, name)

    def with_data(self, data: Any, mask: Any | None = None) -> "MaskedImagePayload":
        """Return the same semantic image mask attached to replacement data."""
        return type(self)(
            data=data,
            mask=self.mask if mask is None else mask,
            metadata=self.metadata,
        )


def runtime_array_operand(value: Any) -> Any:
    """Return the ndarray operand for nominal runtime array payloads."""
    if isinstance(value, RuntimeArrayPayload):
        return value.array_operand()
    return value


def image_payload_data(payload: Any) -> Any:
    """Return concrete image pixels from a runtime image payload."""
    if isinstance(payload, (MaskedImagePayload, ImageMetadataPayload)):
        return payload.data
    return payload


def image_payload_mask(payload: Any) -> Any | None:
    """Return a runtime image mask when present."""
    if isinstance(payload, MaskedImagePayload):
        return payload.mask
    return None


def image_payload_metadata(payload: Any) -> ImagePayloadMetadata:
    """Return runtime image metadata when present."""
    if isinstance(payload, (MaskedImagePayload, ImageMetadataPayload)):
        return payload.metadata
    if isinstance(payload, ObjectLabelPayload):
        return ImagePayloadMetadata(
            source_path=payload.source_path,
            source_component_metadata=payload.source_component_metadata,
            channel_source_paths=payload.channel_source_paths,
            channel_source_component_metadata=payload.channel_source_component_metadata,
            spatial_origin_yx=payload.spatial_origin_yx,
            source_spatial_shape_yx=payload.source_spatial_shape_yx,
            source_image_names=payload.source_image_names,
        )
    if isinstance(payload, ObjectLabelSet):
        return ImagePayloadMetadata(
            source_path=payload.source_path,
            source_component_metadata=payload.source_component_metadata,
            channel_source_paths=payload.channel_source_paths,
            channel_source_component_metadata=payload.channel_source_component_metadata,
            spatial_origin_yx=payload.spatial_origin_yx,
            source_spatial_shape_yx=payload.source_spatial_shape_yx,
            source_image_names=payload.source_image_names,
        )
    return ImagePayloadMetadata()


@dataclass(frozen=True, slots=True)
class RuntimeImagePayloadContext:
    """Nominal request for attaching optional mask and metadata to image pixels."""

    data: RuntimeArrayData
    mask: RuntimeArrayData | None
    metadata: ImagePayloadMetadata

    def payload(self) -> RuntimeArrayData:
        if self.mask is not None:
            return MaskedImagePayload(data=self.data, mask=self.mask, metadata=self.metadata)
        if self.metadata.has_values:
            return ImageMetadataPayload(data=self.data, metadata=self.metadata)
        return self.data


def image_payload_with_context(
    data: RuntimeArrayData,
    *,
    mask: RuntimeArrayData | None = None,
    metadata: ImagePayloadMetadata | None = None,
) -> RuntimeArrayData:
    """Attach generic image context to pixels only when context is present."""
    return RuntimeImagePayloadContext(
        data,
        mask,
        ImagePayloadMetadata() if metadata is None else metadata,
    ).payload()


def project_image_mask_to_data_domain(mask: Any, data: Any) -> Any | None:
    """Project a source mask into the concrete image data domain when possible."""
    if mask is None:
        return None
    mask_array = np.asarray(mask, dtype=bool)
    data_shape = tuple(np.asarray(data).shape)
    mask_shape = tuple(mask_array.shape)
    mask_domain = ImageMaskDomain(data_shape)
    if mask_domain.accepts(mask_shape):
        return mask_array
    if (
        mask_array.ndim >= 3
        and mask_domain.accepts(tuple(mask_array.shape[1:]))
    ):
        return np.all(mask_array, axis=0)
    if (
        mask_array.ndim >= 3
        and mask_domain.accepts(tuple(mask_array.shape[:-1]))
    ):
        return np.all(mask_array, axis=-1)
    if (
        len(mask_shape) == len(data_shape)
        and mask_shape[0] == 1
        and data_shape[0] != 1
        and mask_shape[1:] == data_shape[1:]
    ):
        return np.broadcast_to(mask_array, data_shape)
    if (
        len(mask_shape) > 0
        and len(mask_shape) < len(data_shape)
        and mask_shape == data_shape[-len(mask_shape):]
    ):
        return np.broadcast_to(mask_array, data_shape)
    return None


def image_mask_for_data_domain(
    *,
    source_payload: Any,
    data: Any,
    explicit_mask: Any | None = None,
) -> Any | None:
    """Return the effective image mask projected into a concrete data domain."""
    source_mask = (
        image_payload_mask(source_payload)
        if explicit_mask is None
        else image_payload_data(explicit_mask)
    )
    projected_mask = project_image_mask_to_data_domain(source_mask, data)
    if projected_mask is None:
        return None
    mask_array = np.asarray(projected_mask, dtype=bool)
    data_shape = tuple(np.asarray(data).shape)
    if (
        mask_array.ndim > 0
        and mask_array.shape != data_shape
        and mask_array.shape == data_shape[-mask_array.ndim:]
    ):
        return np.broadcast_to(mask_array, data_shape)
    return mask_array


@dataclass(frozen=True, slots=True)
class DerivedImagePayloadContext:
    """Project source image context onto a derived image payload."""

    source_payload: ImagePayloadMetadataInput | None
    data: RuntimeArrayData

    def payload(self) -> RuntimeArrayData:
        same_spatial_domain = self.same_spatial_domain()
        metadata = DerivedImagePayloadSourceMetadata(
            source_payload=self.source_payload,
            data=self.data,
        ).metadata()
        if not same_spatial_domain:
            metadata = metadata.without_spatial_domain()
        return RuntimeImagePayloadContext(
            image_payload_data(self.data),
            DerivedImagePayloadMaskContext(
                source_payload=self.source_payload,
                data=self.data,
                same_spatial_domain=same_spatial_domain,
            ).mask(),
            metadata,
        ).payload()

    def same_spatial_domain(self) -> bool:
        source_shape_yx = image_payload_spatial_shape_yx(self.source_payload)
        output_shape_yx = image_payload_spatial_shape_yx(self.data)
        return (
            source_shape_yx is not None
            and output_shape_yx is not None
            and source_shape_yx == output_shape_yx
        )


@dataclass(frozen=True, slots=True)
class DerivedImagePayloadSourceMetadata:
    """Resolve source provenance for derived image payloads."""

    source_payload: ImagePayloadMetadataInput | None
    data: RuntimeArrayData

    def metadata(self) -> ImagePayloadMetadata:
        output_metadata = image_payload_metadata(self.data)
        source_metadata = image_payload_metadata(self.source_payload)
        if not self.source_planes_should_replace_output(output_metadata, source_metadata):
            return output_metadata.with_source_context_from(source_metadata)
        return DerivedImagePayloadPlaneSourceReplacement(
            output_metadata=output_metadata,
            source_metadata=source_metadata,
        ).metadata()

    def source_planes_should_replace_output(
        self,
        output_metadata: ImagePayloadMetadata,
        source_metadata: ImagePayloadMetadata,
    ) -> bool:
        plane_count = self.output_plane_count()
        if plane_count is None:
            return False
        source_plane_count = source_metadata.source_provenance.channel_plane_count
        if source_plane_count != plane_count:
            return False
        output_plane_count = output_metadata.source_provenance.channel_plane_count
        return output_plane_count != plane_count

    def output_plane_count(self) -> int | None:
        data = np.asarray(image_payload_data(self.data))
        if data.ndim < 3 or is_color_image_slice(data):
            return None
        return int(data.shape[0])


@dataclass(frozen=True, slots=True)
class DerivedImagePayloadPlaneSourceReplacement:
    """Copy source plane provenance onto derived image payload metadata."""

    output_metadata: ImagePayloadMetadata
    source_metadata: ImagePayloadMetadata

    def metadata(self) -> ImagePayloadMetadata:
        metadata = replace(self.output_metadata)
        metadata.source_path = self.source_metadata.source_path
        metadata.source_component_metadata = self.source_metadata.source_component_metadata
        metadata.channel_source_paths = self.source_metadata.channel_source_paths
        metadata.channel_source_component_metadata = (
            self.source_metadata.channel_source_component_metadata
        )
        metadata.source_image_names = (
            self.source_metadata.source_image_names
            or self.output_metadata.source_image_names
        )
        metadata.spatial_origin_yx = (
            self.output_metadata.spatial_origin_yx
            if self.output_metadata.spatial_origin_yx is not None
            else self.source_metadata.spatial_origin_yx
        )
        metadata.source_spatial_shape_yx = (
            self.output_metadata.source_spatial_shape_yx
            if self.output_metadata.source_spatial_shape_yx is not None
            else self.source_metadata.source_spatial_shape_yx
        )
        metadata.physical_border_edges_yx = (
            self.output_metadata.physical_border_edges_yx
            if self.output_metadata.physical_border_edges_yx is not None
            else self.source_metadata.physical_border_edges_yx
        )
        metadata.mask_defines_border = (
            self.output_metadata.mask_defines_border
            if self.output_metadata.mask_defines_border is not None
            else self.source_metadata.mask_defines_border
        )
        return metadata


@dataclass(frozen=True, slots=True)
class DerivedImagePayloadMaskContext:
    """Resolve the mask for a derived image payload."""

    source_payload: ImagePayloadMetadataInput | None
    data: RuntimeArrayData
    same_spatial_domain: bool

    def mask(self) -> RuntimeArrayData | None:
        output_mask = image_payload_mask(self.data)
        if output_mask is not None:
            return output_mask
        if not self.same_spatial_domain:
            return None
        return project_image_mask_to_data_domain(
            image_payload_mask(self.source_payload),
            image_payload_data(self.data),
        )


def with_image_payload_data(
    payload: Any,
    data: Any,
    *,
    mask: Any | None = None,
    metadata: ImagePayloadMetadata | None = None,
) -> Any:
    """Preserve image-mask and metadata semantics while replacing pixels."""
    resolved_mask = project_image_mask_to_data_domain(
        image_payload_mask(payload) if mask is None else mask,
        data,
    )
    resolved_metadata = (
        image_payload_metadata(payload) if metadata is None else metadata
    )
    return image_payload_with_context(
        data,
        mask=resolved_mask,
        metadata=resolved_metadata,
    )


def image_payload_slice_context(
    payload: Any,
    data: Any,
    channel_index: int,
) -> Any:
    """Attach one channel/slice of a payload's image context to slice data."""
    mask = image_payload_mask(payload)
    return image_payload_with_context(
        data,
        mask=None if mask is None else image_payload_mask_slice(mask, channel_index),
        metadata=image_payload_metadata(payload).for_channel(channel_index),
    )


def image_payload_mask_slice(mask: Any, channel_index: int) -> Any:
    """Return the mask plane matching one image channel/slice."""
    mask_array = np.asarray(mask)
    if mask_array.ndim >= 3:
        return mask_array[channel_index]
    return mask


@dataclass(frozen=True, slots=True)
class ImagePayloadChannelProjection:
    """Project one channel while preserving payload metadata and mask semantics."""

    source_payload: Any
    source_data: Any
    channel_index: int
    channel_data: Any

    @classmethod
    def from_channel(
        cls,
        source_payload: Any,
        source_data: Any,
        channel_index: int,
    ) -> "ImagePayloadChannelProjection":
        return cls(
            source_payload=source_payload,
            source_data=source_data,
            channel_index=channel_index,
            channel_data=source_data[channel_index : channel_index + 1],
        )

    def payload(self) -> Any:
        return image_payload_with_context(
            self.channel_data,
            mask=self.projected_mask(),
            metadata=image_payload_metadata(self.source_payload).for_channel(
                self.channel_index,
            ),
        )

    def projected_mask(self) -> Any | None:
        mask = image_payload_mask(self.source_payload)
        if mask is None:
            return None
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape == np.asarray(self.source_data).shape:
            return mask_array[self.channel_index : self.channel_index + 1]
        return mask_array


def image_payload_spatial_shape_yx(payload: Any) -> tuple[int, int] | None:
    """Return the XY image shape for a nominal image payload."""
    import numpy as np

    array = np.asarray(image_payload_data(payload))
    if array.ndim < 2:
        return None
    if is_color_image_slice(array) or is_color_image_stack(array):
        return tuple(int(value) for value in array.shape[-3:-1])
    return tuple(int(value) for value in array.shape[-2:])


@dataclass(frozen=True, slots=True)
class ImagePayloadMetadataCompositionRequest:
    """Compose source-image metadata for a stack assembled from payload slices."""

    image_payloads: Sequence[ImagePayloadMetadataInput]

    @property
    def source_metadata(self) -> tuple[ImagePayloadMetadata, ...]:
        return tuple(image_payload_metadata(payload) for payload in self.image_payloads)

    @property
    def source_plane_metadata(self) -> tuple[ImagePayloadMetadata, ...]:
        return tuple(metadata.for_channel(0) for metadata in self.source_metadata)

    def metadata(self) -> ImagePayloadMetadata:
        metadata_by_payload = self.source_metadata
        if not any(metadata.has_values for metadata in metadata_by_payload):
            return ImagePayloadMetadata()
        source_metadata_by_payload = self.source_plane_metadata
        return ImagePayloadMetadata(
            channel_intensity_scales=tuple(
                metadata.intensity_scale
                for metadata in source_metadata_by_payload
            ),
            channel_source_dtypes=tuple(
                metadata.source_dtype
                for metadata in source_metadata_by_payload
            ),
            channel_source_paths=tuple(
                metadata.source_path
                for metadata in source_metadata_by_payload
            ),
            channel_source_component_metadata=tuple(
                metadata.source_component_metadata
                for metadata in source_metadata_by_payload
            ),
            channel_unit_interval_intensity_scales=tuple(
                metadata.unit_interval_intensity_scale
                for metadata in source_metadata_by_payload
            ),
            spatial_origin_yx=self.common_metadata_value(
                metadata.spatial_origin_yx for metadata in metadata_by_payload
            ),
            source_spatial_shape_yx=self.common_metadata_value(
                metadata.source_spatial_shape_yx for metadata in metadata_by_payload
            ),
            physical_border_edges_yx=self.common_metadata_value(
                metadata.physical_border_edges_yx for metadata in metadata_by_payload
            ),
            mask_defines_border=self.common_metadata_value(
                metadata.mask_defines_border for metadata in metadata_by_payload
            ),
            source_image_names=tuple(
                source_image_name
                for metadata in metadata_by_payload
                for source_image_name in metadata.source_image_names
            ),
        )

    @staticmethod
    def common_metadata_value(
        values: Iterable[MetadataValueT | None],
    ) -> MetadataValueT | None:
        values_tuple = tuple(values)
        present = tuple(value for value in values_tuple if value is not None)
        if not present:
            return None
        first = present[0]
        if all(value == first for value in present):
            return first
        return None


def image_intensity_scale_for_dtype(dtype: Any) -> float | None:
    """Return the conventional full-scale intensity for a pixel dtype."""
    import numpy as np

    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.bool_):
        return 1.0
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return None


def image_payload_intensity_scale(
    payload: Any,
    *,
    channel_index: int = 0,
) -> float | None:
    """Return the best semantic intensity scale for an image payload."""
    import numpy as np

    metadata_scale = image_payload_metadata(payload).intensity_scale_for_channel(
        channel_index
    )
    if metadata_scale is not None and metadata_scale > 0:
        return float(metadata_scale)
    return image_intensity_scale_for_dtype(np.asarray(image_payload_data(payload)).dtype)


def normalize_image_payload_intensity(
    payload: Any,
    *,
    dtype: Any = None,
    channel_index: int = 0,
) -> Any:
    """Normalize image pixels by payload metadata while preserving context."""
    import numpy as np

    array = np.asarray(image_payload_data(payload))
    target_dtype = np.float32 if dtype is None else np.dtype(dtype)
    if np.issubdtype(array.dtype, np.bool_):
        normalized = array.astype(target_dtype)
    elif np.issubdtype(array.dtype, np.integer):
        intensity_scale = image_payload_intensity_scale(
            payload,
            channel_index=channel_index,
        )
        if intensity_scale is None or intensity_scale <= 1:
            normalized = array.astype(target_dtype)
        else:
            normalized = array.astype(target_dtype) / float(intensity_scale)
            metadata = image_payload_metadata(payload).with_unit_interval_intensity_scale(
                int(intensity_scale)
            )
            return with_image_payload_data(payload, normalized, metadata=metadata)
    elif np.issubdtype(array.dtype, np.floating):
        normalized = array.astype(target_dtype, copy=False)
    else:
        return payload
    return with_image_payload_data(payload, normalized)


def image_payload_metadata_from_source(
    image: Any,
    source_path: str,
    source_component_metadata: SourceComponentMetadata | None = None,
    read_backend: str | None = None,
    filemanager: Any | None = None,
) -> ImagePayloadMetadata:
    """Return generic source metadata for an image loaded through OpenHCS I/O."""
    resolved_source_path = ImagePayloadSourcePathResolver(
        source_path=source_path,
        read_backend=read_backend,
        filemanager=filemanager,
    ).resolve()
    source_metadata = image_file_source_metadata(resolved_source_path)
    source_dtype = source_metadata.source_dtype
    resolved_path_text = (
        str(resolved_source_path) if resolved_source_path is not None else source_path
    )
    if source_dtype is None:
        array_metadata = ImagePayloadMetadata.for_array(
            image_payload_data(image),
            source_path=resolved_path_text,
        )
        return replace(
            array_metadata,
            source_component_metadata=source_component_metadata,
        )
    return ImagePayloadMetadata(
        intensity_scale=source_metadata.intensity_scale,
        source_dtype=str(source_dtype),
        source_path=resolved_path_text,
        source_component_metadata=source_component_metadata,
    )


@dataclass(frozen=True, slots=True)
class ImagePayloadSourcePathResolver:
    """Resolve physical image paths from optional backend-specific I/O context."""

    source_path: str
    read_backend: str | None = None
    filemanager: Any | None = None

    def resolve(self) -> Path | None:
        if self.read_backend is not None and self.filemanager is not None:
            resolved_backend_path = self.resolve_backend_path()
            if resolved_backend_path is not None:
                return resolved_backend_path
        path = Path(self.source_path)
        return path if path.exists() else None

    def resolve_backend_path(self) -> Path | None:
        try:
            registry = self.filemanager.registry
        except AttributeError:
            return None
        if not isinstance(registry, Mapping):
            return None
        backend = registry.get(self.read_backend)
        if VirtualWorkspaceBackend is not None and isinstance(
            backend,
            VirtualWorkspaceBackend,
        ):
            try:
                return Path(backend._resolve_path(self.source_path))
            except Exception:
                logger.debug(
                    "Could not resolve image source path %s via backend %s.",
                    self.source_path,
                    self.read_backend,
                    exc_info=True,
                )
        return None


def image_file_source_dtype(path: Path | None) -> Any | None:
    """Return an image file's stored dtype without loading pixel data."""
    return image_file_source_metadata(path).source_dtype


@dataclass(frozen=True, slots=True)
class ImageFileSourceMetadata:
    """Image-file metadata relevant to runtime image semantics."""

    source_dtype: Any | None = None
    intensity_scale: float | None = None


def image_file_source_metadata(path: Path | None) -> ImageFileSourceMetadata:
    """Return image file metadata without loading pixel data when possible."""
    if path is None or not path.exists():
        return ImageFileSourceMetadata()
    try:
        import imageio.v3 as iio

        dtype = iio.improps(path).dtype
    except Exception:
        logger.debug("Could not read image dtype metadata for %s.", path, exc_info=True)
        return ImageFileSourceMetadata()
    return ImageFileSourceMetadata(
        source_dtype=dtype,
        intensity_scale=(
            _image_file_declared_intensity_scale(path)
            or image_intensity_scale_for_dtype(dtype)
        ),
    )


def _image_file_declared_intensity_scale(path: Path) -> float | None:
    """Return declared source-file max intensity when the format exposes one."""
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tag = page.tags.get("SMaxSampleValue") or page.tags.get("MaxSampleValue")
        if tag is None:
            return None
        value = tag.value
    scale_value = value[0] if isinstance(value, (tuple, list)) else value
    if not isinstance(scale_value, (int, float, np.integer, np.floating)):
        return None
    scale = float(scale_value)
    return scale if scale > 0 else None


def _spatial_shape_pair(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) < 2:
        raise ValueError(f"{name} must have at least two spatial dimensions.")
    return int(value[0]), int(value[1])


@dataclass(frozen=True, slots=True)
class SpatialShapeYX:
    """Nominal two-dimensional spatial shape in row/column order."""

    height: int
    width: int

    @classmethod
    def from_sequence(
        cls,
        value: Sequence[int],
        *,
        field_name: str,
    ) -> "SpatialShapeYX":
        if len(value) < 2:
            raise ValueError(
                f"{field_name} must have at least two spatial dimensions."
            )
        return cls(height=int(value[0]), width=int(value[1]))

    @classmethod
    def optional_from_mapping(
        cls,
        data: Mapping[str, Any],
        field_name: str,
    ) -> "SpatialShapeYX | None":
        if field_name not in data or data[field_name] is None:
            return None
        return cls.from_sequence(data[field_name], field_name=field_name)

    def as_tuple(self) -> tuple[int, int]:
        return self.height, self.width


def _tuple_value(values: tuple[Any, ...], index: int) -> Any | None:
    if 0 <= index < len(values):
        return values[index]
    return None


@dataclass(frozen=True, slots=True)
class ImageMaskDomain:
    """Accepted mask shapes for a concrete grayscale/color image data domain."""

    data_shape: tuple[int, ...]

    def accepts(self, mask_shape: tuple[int, ...]) -> bool:
        return mask_shape in self.valid_shapes()

    def valid_shapes(self) -> frozenset[tuple[int, ...]]:
        valid: set[tuple[int, ...]] = {self.data_shape}
        if len(self.data_shape) == 2:
            valid.add(self.data_shape)
        if len(self.data_shape) == 3:
            if self.data_shape[-1] in (3, 4):
                valid.add(self.data_shape[:2])
            else:
                valid.add(self.data_shape[1:])
        if len(self.data_shape) == 4:
            if self.data_shape[-1] in (3, 4):
                valid.add(self.data_shape[:3])
                valid.add(self.data_shape[1:3])
            else:
                valid.add(self.data_shape[1:])
                valid.add(self.data_shape[-2:])
                valid.add((self.data_shape[0], *self.data_shape[-2:]))
        if len(self.data_shape) == 5:
            valid.add(self.data_shape[:4])
            valid.add(self.data_shape[1:4])
            valid.add((self.data_shape[0], *self.data_shape[-3:-1]))
            valid.add(self.data_shape[-3:-1])
        return frozenset(valid)


class ObjectLabelDomainMetadataFields(ObjectLabelDomainMetadata):
    """Object-label domain metadata carried as normalized runtime-value fields."""

    declared_object_count: int | None
    declared_object_ids: tuple[int, ...]
    declared_object_id_domains: tuple[tuple[int, ...], ...]
    domain_scope: ObjectLabelDomainScope

    def object_label_domain(self) -> ObjectLabelDomain:
        return ObjectLabelDomain.declared(
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            scope=self.domain_scope,
        )


class ObjectLabelValue(
    SourceImageProvenanceFields,
    ObjectLabelDomainMetadataFields,
    ABC,
):
    """Nominal object-label carrier with dense labels and domain metadata."""

    labels: ObjectLabelData
    unedited_labels: ObjectLabelData | None
    small_removed_labels: ObjectLabelData | None
    representation: ObjectLabelRepresentation
    plane_axis: RuntimePlaneAxis
    spatial_origin_yx: tuple[int, int] | None
    source_spatial_shape_yx: tuple[int, int] | None

    @property
    def source_image_name(self) -> str | None:
        """Return the semantic source image name when this carrier has one."""
        return None

    @abstractmethod
    def labels_for_variant(self, variant: ObjectLabelVariant | str) -> ObjectLabelData:
        """Return the dense labels for one semantic label variant."""

    @abstractmethod
    def with_labels(
        self,
        labels: ObjectLabelData,
        *,
        unedited_labels: ObjectLabelData | None = None,
        small_removed_labels: ObjectLabelData | None = None,
    ) -> "ObjectLabelValue":
        """Return this carrier's metadata with replacement labels."""

    @abstractmethod
    def with_projected_plane(
        self,
        labels: ObjectLabelData,
        plane_index: int,
        *,
        unedited_labels: ObjectLabelData | None = None,
        small_removed_labels: ObjectLabelData | None = None,
    ) -> "ObjectLabelValue":
        """Return one selected label plane with projected domain metadata."""

    @abstractmethod
    def with_source_image_context(self, image: RuntimeArrayData) -> "ObjectLabelValue":
        """Return this object-label value with missing provenance filled from image."""


@dataclass(frozen=True, slots=True)
class ObjectLabelSourceContextProvenanceMerge:
    """Merge image provenance into object labels without reviving stale stack axes."""

    label_provenance: SourceImageProvenance
    image_provenance: SourceImageProvenance

    @classmethod
    def for_label_and_image(
        cls,
        label: ObjectLabelValue,
        image: RuntimeArrayData,
    ) -> "ObjectLabelSourceContextProvenanceMerge":
        return cls(
            label_provenance=label.source_provenance,
            image_provenance=image_payload_metadata(image).source_provenance,
        )

    def provenance(self) -> SourceImageProvenance:
        merged = self.label_provenance.with_missing_from(self.image_provenance)
        if not self.label_has_scalar_source_identity:
            return merged
        return SourceImageProvenance(
            source_path=merged.source_path,
            source_component_metadata=merged.source_component_metadata,
            source_image_names=self.scalar_source_image_names(
                merged.source_image_names,
            ),
        )

    @property
    def label_has_scalar_source_identity(self) -> bool:
        return (
            self.label_provenance.addressable
            and self.label_provenance.channel_plane_count == 0
        )

    @staticmethod
    def scalar_source_image_names(names: tuple[str, ...]) -> tuple[str, ...]:
        if len(names) <= 1:
            return names
        unique_names = tuple(dict.fromkeys(names))
        if len(unique_names) == 1:
            return unique_names
        return ()


@dataclass(slots=True)
class ObjectLabelPayload(RuntimeArrayPayload, ObjectLabelValue):
    """Dense object labels plus optional semantic label variants."""

    labels: ObjectLabelData
    unedited_labels: ObjectLabelData | None = None
    small_removed_labels: ObjectLabelData | None = None
    declared_object_count: int | None = None
    declared_object_ids: tuple[int, ...] = ()
    declared_object_id_domains: tuple[tuple[int, ...], ...] = ()
    domain_scope: ObjectLabelDomainScope = ObjectLabelDomainScope.PAYLOAD
    plane_axis: RuntimePlaneAxis = RuntimePlaneAxis.RUNTIME_SLICE
    spatial_origin_yx: tuple[int, int] | None = None
    source_spatial_shape_yx: tuple[int, int] | None = None

    @classmethod
    def from_domain_metadata(
        cls,
        metadata: Any,
        *,
        labels: ObjectLabelData,
        unedited_labels: ObjectLabelData | None = None,
        small_removed_labels: ObjectLabelData | None = None,
        plane_axis: RuntimePlaneAxis | None = None,
    ) -> "ObjectLabelPayload":
        """Build a payload preserving object-label domain metadata from a carrier."""
        return cls(
            labels=labels,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
            declared_object_count=metadata.declared_object_count,
            declared_object_ids=metadata.declared_object_ids,
            declared_object_id_domains=metadata.declared_object_id_domains,
            domain_scope=metadata.domain_scope,
            plane_axis=metadata.plane_axis if plane_axis is None else plane_axis,
            spatial_origin_yx=metadata.spatial_origin_yx,
            source_spatial_shape_yx=metadata.source_spatial_shape_yx,
            source_path=metadata.source_path,
            source_component_metadata=metadata.source_component_metadata,
            channel_source_paths=metadata.channel_source_paths,
            channel_source_component_metadata=metadata.channel_source_component_metadata,
            source_image_names=metadata.source_image_names,
        )

    def __post_init__(self, *source_provenance_values: object) -> None:
        self.absorb_explicit_source_provenance(
            SourceImageProvenance.from_init_values(source_provenance_values)
        )
        if self.declared_object_count is not None:
            count = int(self.declared_object_count)
            if count < 0:
                raise ValueError("ObjectLabelPayload.declared_object_count cannot be negative.")
            self.declared_object_count = count
        ids = tuple(int(object_id) for object_id in self.declared_object_ids)
        if any(object_id <= 0 for object_id in ids):
            raise ValueError("ObjectLabelPayload.declared_object_ids must be positive.")
        self.declared_object_ids = tuple(sorted(dict.fromkeys(ids)))
        self.declared_object_id_domains = tuple(
                ObjectLabelDomain._normalize_ids(domain, "declared_object_id_domains")
                for domain in self.declared_object_id_domains
            )
        self.domain_scope = coerce_enum(
                ObjectLabelDomainScope,
                self.domain_scope,
                "ObjectLabelPayload.domain_scope",
            )
        self.plane_axis = coerce_enum(
                RuntimePlaneAxis,
                self.plane_axis,
                "ObjectLabelPayload.plane_axis",
            )
        if self.spatial_origin_yx is not None:
            self.spatial_origin_yx = _spatial_shape_pair(self.spatial_origin_yx, "spatial_origin_yx")
        if self.source_spatial_shape_yx is not None:
            self.source_spatial_shape_yx = _spatial_shape_pair(
                    self.source_spatial_shape_yx,
                    "source_spatial_shape_yx",
                )
        self.normalize_source_provenance_fields()

    @property
    def shape(self) -> Any:
        return self.labels.shape

    @property
    def ndim(self) -> int:
        return self.labels.ndim

    @property
    def dtype(self) -> Any:
        return self.labels.dtype

    def max(self) -> np.generic:
        labels = np.asarray(self.labels)
        if labels.size == 0:
            raise ValueError("ObjectLabelPayload.max requires non-empty labels.")
        return labels.max()

    def __array__(self, dtype: Any | None = None) -> Any:
        import numpy as np

        return np.asarray(self.labels, dtype=dtype)

    def array_payload_data(self) -> Any:
        return self.labels

    def __getitem__(self, key: Any) -> Any:
        return self.labels[key]

    def labels_for_variant(
        self,
        variant: ObjectLabelVariant | str,
    ) -> ObjectLabelData:
        normalized = coerce_enum(
            ObjectLabelVariant,
            variant,
            "ObjectLabelPayload.variant",
        )
        return ObjectLabelVariantDataStrategy.for_enum_member(normalized).labels(self)

    @property
    def variants(self) -> tuple[ObjectLabelVariant, ...]:
        variants = [ObjectLabelVariant.FINAL]
        if self.unedited_labels is not None:
            variants.append(ObjectLabelVariant.UNEDITED)
        if self.small_removed_labels is not None:
            variants.append(ObjectLabelVariant.SMALL_REMOVED)
        return tuple(variants)

    def with_labels(
        self,
        labels: ObjectLabelData,
        *,
        unedited_labels: ObjectLabelData | None = None,
        small_removed_labels: ObjectLabelData | None = None,
    ) -> "ObjectLabelPayload":
        """Return this payload's domain metadata with replacement labels."""
        return ObjectLabelPayload.from_domain_metadata(
            self,
            labels=labels,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
        )

    def with_data(self, data: Any) -> "ObjectLabelPayload":
        return self.with_labels(data)

    def with_projected_plane(
        self,
        labels: ObjectLabelData,
        plane_index: int,
        *,
        unedited_labels: ObjectLabelData | None = None,
        small_removed_labels: ObjectLabelData | None = None,
    ) -> "ObjectLabelPayload":
        """Return one selected payload plane with projected domain metadata."""
        return object_label_payload_with_projected_plane(
            self,
            labels,
            plane_index,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
        )

    def with_source_image_context(self, image: RuntimeArrayData) -> "ObjectLabelPayload":
        """Attach image provenance without changing this payload's label domain."""
        metadata = image_payload_metadata(image)
        source_provenance = ObjectLabelSourceContextProvenanceMerge.for_label_and_image(
            self,
            image,
        ).provenance()
        return ObjectLabelPayload(
            labels=self.labels,
            unedited_labels=self.unedited_labels,
            small_removed_labels=self.small_removed_labels,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
            plane_axis=self.plane_axis,
            spatial_origin_yx=(
                self.spatial_origin_yx
                if self.spatial_origin_yx is not None
                else metadata.spatial_origin_yx
            ),
            source_spatial_shape_yx=(
                self.source_spatial_shape_yx
                if self.source_spatial_shape_yx is not None
                else metadata.source_spatial_shape_yx
            ),
            source_path=source_provenance.source_path,
            source_component_metadata=source_provenance.source_component_metadata,
            channel_source_paths=source_provenance.channel_source_paths,
            channel_source_component_metadata=(
                source_provenance.channel_source_component_metadata
            ),
            source_image_names=source_provenance.source_image_names,
        )


class SharedStrategyLabelBase(ABC):
    """Shared registry-label declaration surface for strategy families."""

    __registry_key__ = RegistryKeyAttribute.STRATEGY_LABEL.value
    __skip_if_no_key__ = True

    strategy_label: ClassVar[str | None] = None


class ObjectLabelVariantDataStrategy(
    SharedStrategyLabelBase,
    EnumKeyedStrategyMixin[ObjectLabelVariant],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered semantics for object-label variant payload selection."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "variant"
    variant: ClassVar[ObjectLabelVariant]

    @abstractmethod
    def labels(self, payload: ObjectLabelPayload) -> ObjectLabelData:
        """Return labels for this variant from one payload."""

    @abstractmethod
    def present(self, payloads: Sequence[ObjectLabelPayload]) -> bool:
        """Return whether this variant has material data across payloads."""


class FinalObjectLabelVariantDataStrategy(ObjectLabelVariantDataStrategy):
    """Final labels are always present."""

    variant = ObjectLabelVariant.FINAL

    def labels(self, payload: ObjectLabelPayload) -> ObjectLabelData:
        return payload.labels

    def present(self, payloads: Sequence[ObjectLabelPayload]) -> bool:
        del payloads
        return True


class UneditedObjectLabelVariantDataStrategy(ObjectLabelVariantDataStrategy):
    """Unedited labels fall back to final labels when absent."""

    variant = ObjectLabelVariant.UNEDITED

    def labels(self, payload: ObjectLabelPayload) -> ObjectLabelData:
        return payload.unedited_labels if payload.unedited_labels is not None else payload.labels

    def present(self, payloads: Sequence[ObjectLabelPayload]) -> bool:
        return any(payload.unedited_labels is not None for payload in payloads)


class SmallRemovedObjectLabelVariantDataStrategy(ObjectLabelVariantDataStrategy):
    """Small-removed labels fall back to final labels when absent."""

    variant = ObjectLabelVariant.SMALL_REMOVED

    def labels(self, payload: ObjectLabelPayload) -> ObjectLabelData:
        return (
            payload.small_removed_labels
            if payload.small_removed_labels is not None
            else payload.labels
        )

    def present(self, payloads: Sequence[ObjectLabelPayload]) -> bool:
        return any(payload.small_removed_labels is not None for payload in payloads)


class ColumnarRows(ABC, metaclass=AutoRegisterMeta):
    """Nominal ABC for table payloads exposing named columns."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True

    @property
    @abstractmethod
    def columns(self) -> Any:
        ...

    def column_values(self, column: str) -> Sequence[object]:
        """Return one named column from this nominal columnar payload."""
        columns = self.columns
        if isinstance(columns, Mapping):
            return columns[column]
        return self[column]

    def row_count(self) -> int:
        """Return the number of rows represented by this columnar payload."""
        columns = self.columns
        if not columns:
            return 0
        if isinstance(columns, Mapping):
            return len(next(iter(columns.values())))
        first_column = next(iter(columns))
        return len(self.column_values(first_column))

    def row_mappings(self) -> tuple[Mapping[str, object], ...]:
        """Return row-wise mappings for this columnar payload."""
        columns = tuple(str(column) for column in self.columns)
        column_values = tuple(self.column_values(column) for column in columns)
        return tuple(
            dict(zip(columns, values, strict=True))
            for values in zip(*column_values, strict=True)
        )


ObjectLabelData = RuntimeArrayData | ColumnarRows
ObjectLabelValueBuildSource = ObjectLabelValue | ObjectLabelData
ObjectLabelPayloadBuildSource = ObjectLabelValue | ObjectLabelDomainMetadata | ObjectLabelData
ObjectLabelMeasurementSource = ObjectLabelValue | ObjectLabelData


@dataclass(frozen=True, slots=True)
class SparseIJVLabelRows(ColumnarRows):
    """Sparse object-label table with CellProfiler-compatible y/x/label columns."""

    data: Any

    def __post_init__(self) -> None:
        array = self.as_array()
        if array.ndim != 2 or array.shape[1] not in (3, 4):
            raise ValueError(
                "SparseIJVLabelRows.data must be an N x 3 y/x/label table "
                "or an N x 4 slice/y/x/label table."
            )

    @property
    def columns(self) -> Mapping[str, Any]:
        array = self.as_array()
        columns = {
            "y": array[:, self.y_column],
            "x": array[:, self.x_column],
            "label": array[:, self.label_column],
        }
        if self.has_slice_index:
            columns = {"slice_index": array[:, self.slice_column], **columns}
        return MappingProxyType(columns)

    @property
    def has_slice_index(self) -> bool:
        return int(self.as_array().shape[1]) == 4

    @property
    def slice_column(self) -> int:
        if not self.has_slice_index:
            raise ValueError("SparseIJVLabelRows has no slice_index column.")
        return 0

    @property
    def y_column(self) -> int:
        return 1 if self.has_slice_index else 0

    @property
    def x_column(self) -> int:
        return 2 if self.has_slice_index else 1

    @property
    def label_column(self) -> int:
        return 3 if self.has_slice_index else 2

    @classmethod
    def from_yx_label(cls, data: Any) -> "SparseIJVLabelRows":
        return cls(data)

    @classmethod
    def from_dense_labels(cls, labels: Any) -> "SparseIJVLabelRows":
        import numpy as _np

        label_array = _np.asarray(labels)
        if label_array.ndim != 2:
            raise ValueError(
                "SparseIJVLabelRows.from_dense_labels requires a 2-D label image."
            )
        rows, columns = _np.nonzero(label_array > 0)
        if rows.size == 0:
            return cls(_np.zeros((0, 3), dtype=_np.int32))
        return cls(
            _np.column_stack((rows, columns, label_array[rows, columns])).astype(
                _np.int32,
                copy=False,
            )
        )

    @classmethod
    def from_dense_stack(cls, labels: Any) -> "SparseIJVLabelRows":
        """Build sparse rows from a 2-D label image or runtime-slice stack."""
        import numpy as _np

        label_array = _np.asarray(labels)
        if label_array.ndim == 2:
            return cls.from_dense_labels(label_array)
        if label_array.ndim != 3:
            raise ValueError(
                "SparseIJVLabelRows.from_dense_stack requires a 2-D label image "
                "or a 3-D runtime-slice label stack."
            )
        slices = tuple(cls.from_dense_labels(slice_labels) for slice_labels in label_array)
        return cls.from_slices(slices)

    @classmethod
    def from_slices(cls, values: Sequence["SparseIJVLabelRows"]) -> "SparseIJVLabelRows":
        import numpy as _np

        arrays = []
        for slice_index, value in enumerate(values):
            array = value.as_yx_label_array()
            if not array.size:
                continue
            arrays.append(
                _np.column_stack(
                    (
                        _np.full(array.shape[0], slice_index, dtype=_np.int32),
                        array,
                    )
                )
            )
        return cls(
            _np.vstack(arrays).astype(_np.int32, copy=False)
            if arrays
            else _np.zeros((0, 4), dtype=_np.int32)
        )

    def as_yx_label_array(self) -> Any:
        array = self.as_array()
        if not self.has_slice_index:
            return array
        return array[:, (self.y_column, self.x_column, self.label_column)]

    def slice_indices(self) -> tuple[int, ...]:
        if not self.has_slice_index:
            return (0,)
        import numpy as _np

        return tuple(int(index) for index in _np.unique(self.as_array()[:, self.slice_column]))

    def slice(self, slice_index: int) -> "SparseIJVLabelRows":
        if not self.has_slice_index:
            if slice_index != 0:
                import numpy as _np

                return type(self)(_np.zeros((0, 3), dtype=_np.int32))
            return self
        array = self.as_array()
        rows = array[array[:, self.slice_column] == int(slice_index)]
        return type(self)(rows[:, (self.y_column, self.x_column, self.label_column)])

    def to_dense(
        self,
        *,
        source_spatial_shape_yx: tuple[int, int] | None = None,
        dtype: object | None = None,
    ) -> np.ndarray:
        """Materialize sparse IJV rows as dense 2-D or slice-stacked labels."""
        array = self.as_array()
        if dtype is None:
            dtype = array.dtype if array.size else np.int32
        height, width = self._dense_spatial_shape(source_spatial_shape_yx)
        if not self.has_slice_index:
            dense = np.zeros((height, width), dtype=dtype)
            if array.size:
                dense[
                    array[:, self.y_column].astype(np.intp, copy=False),
                    array[:, self.x_column].astype(np.intp, copy=False),
                ] = array[:, self.label_column].astype(dtype, copy=False)
            return dense
        slice_count = self.label_data_runtime_slice_count()
        dense = np.zeros((slice_count, height, width), dtype=dtype)
        if array.size:
            dense[
                array[:, self.slice_column].astype(np.intp, copy=False),
                array[:, self.y_column].astype(np.intp, copy=False),
                array[:, self.x_column].astype(np.intp, copy=False),
            ] = array[:, self.label_column].astype(dtype, copy=False)
        return dense

    def label_data_runtime_slice_count(self) -> int:
        """Return the encoded runtime-slice count, including empty stacks."""
        if not self.has_slice_index:
            return 1
        slice_indices = self.slice_indices()
        return max(slice_indices) + 1 if slice_indices else 0

    def _dense_spatial_shape(
        self,
        source_spatial_shape_yx: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if source_spatial_shape_yx is not None:
            return _spatial_shape_pair(source_spatial_shape_yx, "source_spatial_shape_yx")
        array = self.as_array()
        if not array.size:
            return (0, 0)
        return (
            int(np.max(array[:, self.y_column])) + 1,
            int(np.max(array[:, self.x_column])) + 1,
        )

    def as_array(self) -> Any:
        _ensure_runtime_payload_integrations_registered()
        try:
            import numpy as _np
        except Exception as exc:  # pragma: no cover - numpy is a core runtime dep.
            raise TypeError("SparseIJVLabelRows requires an array-like payload.") from exc
        return _np.asarray(self.data)


class SparseIJVLabelRowsIdDomainStrategy(ObjectLabelIdDomainStrategy):
    """Extract present object IDs from sparse IJV label rows without densifying."""

    value_type = SparseIJVLabelRows

    def present_ids(self, labels: Any) -> tuple[int, ...]:
        if not isinstance(labels, SparseIJVLabelRows):
            raise TypeError(
                "SparseIJVLabelRowsIdDomainStrategy requires SparseIJVLabelRows, "
                f"got {type(labels).__name__}."
            )
        array = labels.as_array()
        if array.size == 0:
            return ()
        label_column = array[:, labels.label_column]
        return self.positive_ids_from_array(label_column)


def register_array_payload_type(payload_type: _TPayload) -> _TPayload:
    """Declare an external type as a runtime array payload."""
    RuntimeArrayPayload.register(payload_type)
    return payload_type


def register_array_payload_predicate(
    predicate: Callable[[Any], bool],
) -> Callable[[Any], bool]:
    """Declare a semantic predicate for runtime array payload recognition."""
    if predicate not in _ARRAY_PAYLOAD_PREDICATES:
        _ARRAY_PAYLOAD_PREDICATES.append(predicate)
    return predicate


def is_array_payload(data: Any) -> bool:
    """Return whether a value satisfies the runtime array payload contract."""
    _ensure_runtime_payload_integrations_registered()
    return isinstance(data, RuntimeArrayPayload) or any(
        predicate(data)
        for predicate in _ARRAY_PAYLOAD_PREDICATES
    )


def register_columnar_rows_type(payload_type: _TPayload) -> _TPayload:
    """Declare an external type as a columnar rows payload."""
    ColumnarRows.register(payload_type)
    return payload_type


@dataclass(kw_only=True)
class SourceImageContext(metaclass=AutoRegisterMeta):
    """Shared source-image semantic context for values and schemas."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True

    dimensions: tuple[str, ...] = ()
    source_image_name: str | None = None

    def _validate_source_image_context(self, owner_name: str) -> None:
        if self.source_image_name == "":
            raise ValueError(f"{owner_name}.source_image_name cannot be empty.")


@dataclass(slots=True, kw_only=True)
class RuntimeValueSchema(SourceImageContext):
    """Semantic schema attached to a runtime artifact value."""

    kind: ArtifactKind
    slice_aligned: bool = False
    fields: tuple[FieldSpec, ...] = ()
    label_representation: ObjectLabelRepresentation | None = None
    measurement_subject: MeasurementSubject | None = None
    relationship: RelationshipSemantics | None = None
    object_name: str | None = None
    object_id_field: str | None = None
    measurement_schema_validated: bool = False
    measurement_schema_loss_reasons: frozenset[str] = frozenset()
    label_variants: tuple[ObjectLabelVariant, ...] = ()

    def __post_init__(self) -> None:
        self._validate_source_image_context("RuntimeValueSchema")
        self.kind = coerce_enum(ArtifactKind, self.kind, "RuntimeValueSchema.kind")
        if self.label_representation is not None:
            self.label_representation = coerce_enum(
                    ObjectLabelRepresentation,
                    self.label_representation,
                    "RuntimeValueSchema.label_representation",
                )
        if self.object_name == "":
            raise ValueError("RuntimeValueSchema.object_name cannot be empty.")
        if self.object_id_field == "":
            raise ValueError("RuntimeValueSchema.object_id_field cannot be empty.")
        self.measurement_schema_loss_reasons = frozenset(str(reason) for reason in self.measurement_schema_loss_reasons)
        if self.measurement_schema_validated and self.measurement_schema_loss_reasons:
            raise ValueError(
                "RuntimeValueSchema.measurement_schema_validated cannot be true "
                "when measurement_schema_loss_reasons is non-empty."
            )
        self.label_variants = tuple(
                coerce_enum(
                    ObjectLabelVariant,
                    variant,
                    "RuntimeValueSchema.label_variants",
                )
                for variant in self.label_variants
            )
        if (
            self.label_representation is not None
            and self.kind is not ArtifactKind.OBJECT_LABELS
        ):
            raise ValueError(
                "RuntimeValueSchema.label_representation requires "
                "OBJECT_LABELS kind."
            )
        if self.label_variants and self.kind is not ArtifactKind.OBJECT_LABELS:
            raise ValueError(
                "RuntimeValueSchema.label_variants requires OBJECT_LABELS kind."
            )
        if (
            self.measurement_subject is not None
            and self.kind is not ArtifactKind.MEASUREMENTS
        ):
            raise ValueError(
                "RuntimeValueSchema.measurement_subject requires "
                "MEASUREMENTS kind."
            )
        if (
            (
                self.measurement_schema_validated
                or self.measurement_schema_loss_reasons
            )
            and self.kind is not ArtifactKind.MEASUREMENTS
        ):
            raise ValueError(
                "RuntimeValueSchema measurement schema state requires "
                "MEASUREMENTS kind."
            )
        if (
            self.relationship is not None
            and self.kind is not ArtifactKind.RELATIONSHIPS
        ):
            raise ValueError(
                "RuntimeValueSchema.relationship requires RELATIONSHIPS kind."
            )


@dataclass(frozen=True, slots=True)
class RuntimeStoragePolicy:
    """Storage intent for a runtime value once stores/materializers consume it."""

    backend: str | None = None
    path: str | None = None
    materialize: bool = False

    @classmethod
    def from_output_plan(cls, output_plan: ArtifactOutputPlan) -> Self:
        return cls(
            backend=Backend.MEMORY.value,
            path=output_plan.path,
            materialize=output_plan.materialization is not None,
        )

    def __post_init__(self) -> None:
        if self.path and not self.backend:
            raise ValueError("RuntimeStoragePolicy.path requires a backend.")


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementSubjectIdentity:
    """Stable key component for measurement tables sharing one artifact name."""

    scope: MeasurementScope
    name: str | None
    id_field: str | None

    @classmethod
    def from_subject(
        cls,
        subject: MeasurementSubject,
    ) -> "RuntimeMeasurementSubjectIdentity":
        return cls(
            scope=subject.scope,
            name=subject.name,
            id_field=subject.id_field,
        )

    @property
    def token(self) -> str:
        return ":".join(
            (
                "measurement_subject",
                self.scope.value,
                self.name or "",
                self.id_field or "",
            )
        )


class RuntimeValueKeyDiscriminator(
    EnumKeyedStrategyMixin[ArtifactKind],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered semantic key extension for runtime values with subidentity."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True

    strategy_key: ClassVar[ArtifactKind | None] = None

    @classmethod
    def artifact_key(
        cls,
        output_plan: ArtifactOutputPlan,
        *,
        axis_id: str,
        schema: RuntimeValueSchema,
    ) -> ArtifactKey:
        base_key = output_plan.artifact_key(axis_id=axis_id)
        strategy_type = cls.__registry__.get(schema.kind.value)
        if strategy_type is None:
            return base_key
        semantic_id = strategy_type().semantic_id(schema)
        return replace(base_key, semantic_id=semantic_id)

    @abstractmethod
    def semantic_id(self, schema: RuntimeValueSchema) -> str | None:
        """Return the value's semantic subidentity within its artifact key."""


class MeasurementRuntimeValueKeyDiscriminator(RuntimeValueKeyDiscriminator):
    """Keep same-named measurement tables distinct by measurement subject."""

    strategy_key = ArtifactKind.MEASUREMENTS

    def semantic_id(self, schema: RuntimeValueSchema) -> str | None:
        subject = schema.measurement_subject
        if subject is None:
            return None
        return RuntimeMeasurementSubjectIdentity.from_subject(subject).token


@dataclass(frozen=True, slots=True)
class RuntimeValue:
    """Artifact payload validated against compiled runtime semantics."""

    key: ArtifactKey
    data: Any
    schema: RuntimeValueSchema
    storage: RuntimeStoragePolicy | None = None

    @classmethod
    def from_output_plan(
        cls,
        output_plan: ArtifactOutputPlan,
        data: Any,
        *,
        axis_id: str,
        schema: RuntimeValueSchema,
    ) -> Self:
        return cls(
            key=RuntimeValueKeyDiscriminator.artifact_key(
                output_plan,
                axis_id=axis_id,
                schema=schema,
            ),
            data=data,
            schema=schema,
            storage=RuntimeStoragePolicy.from_output_plan(output_plan),
        )

    def __post_init__(self) -> None:
        if self.key.kind is not self.schema.kind:
            raise ValueError(
                f"RuntimeValue key kind {self.key.kind.value} does not match "
                f"schema kind {self.schema.kind.value}."
            )

    @property
    def name(self) -> str:
        return self.key.name

    @property
    def kind(self) -> ArtifactKind:
        return self.key.kind


@dataclass(slots=True, kw_only=True)
class NativeRuntimeValue(ABC, metaclass=AutoRegisterMeta):
    """Native OpenHCS value that can become a validated RuntimeValue."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True

    name: str

    def __post_init__(self) -> None:
        _require_name(self.name, f"{type(self).__name__}.name")

    @abstractmethod
    def runtime_payload(self) -> Any:
        """Return the payload stored under the compiled artifact key."""

    @abstractmethod
    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        """Return the schema that validates the stored payload."""

    def to_runtime_value(
        self,
        output_plan: ArtifactOutputPlan,
        *,
        axis_id: str,
    ) -> RuntimeValue:
        payload = self.runtime_payload()
        return RuntimeValue.from_output_plan(
            output_plan,
            payload,
            axis_id=axis_id,
            schema=self.runtime_schema(payload),
        )


@dataclass(slots=True, kw_only=True)
class NamedImage(SourceImageContext, NativeRuntimeValue):
    """Native OpenHCS named image value."""

    data: Any

    def __post_init__(self) -> None:
        NativeRuntimeValue.__post_init__(self)
        self._validate_source_image_context(type(self).__name__)
        if not _is_array_like(self.data):
            raise TypeError(
                f"NamedImage '{self.name}' requires array-like data with "
                f"shape/ndim, got {type(self.data).__name__}."
            )

    def runtime_payload(self) -> Any:
        return self.data

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        return RuntimeValueSchema(
            kind=ArtifactKind.IMAGE,
            dimensions=self.dimensions,
            source_image_name=self.source_image_name,
        )


@dataclass(slots=True, kw_only=True)
class ObjectLabelSet(
    SourceImageContext,
    NativeRuntimeValue,
    ObjectLabelValue,
):
    """Native OpenHCS object-label value."""

    labels: ObjectLabelData
    unedited_labels: ObjectLabelData | None = None
    small_removed_labels: ObjectLabelData | None = None
    representation: ObjectLabelRepresentation = ObjectLabelRepresentation.DENSE_LABELS
    declared_object_count: int | None = None
    declared_object_ids: tuple[int, ...] = ()
    declared_object_id_domains: tuple[tuple[int, ...], ...] = ()
    domain_scope: ObjectLabelDomainScope = ObjectLabelDomainScope.PAYLOAD
    plane_axis: RuntimePlaneAxis = RuntimePlaneAxis.RUNTIME_SLICE
    spatial_origin_yx: tuple[int, int] | None = None
    source_spatial_shape_yx: tuple[int, int] | None = None

    @classmethod
    def from_runtime_value(cls, value: RuntimeValue) -> Self:
        """Reconstruct the native object-label view from a stored runtime value."""
        if value.kind is not ArtifactKind.OBJECT_LABELS:
            raise TypeError(
                "ObjectLabelSet.from_runtime_value requires an OBJECT_LABELS "
                f"runtime value, got {value.kind.value}."
            )
        payload = value.data
        schema = value.schema
        if isinstance(payload, ObjectLabelPayload):
            return cls(
                name=value.name,
                labels=payload.labels,
                unedited_labels=payload.unedited_labels,
                small_removed_labels=payload.small_removed_labels,
                declared_object_count=payload.declared_object_count,
                declared_object_ids=payload.declared_object_ids,
                declared_object_id_domains=payload.declared_object_id_domains,
                domain_scope=payload.domain_scope,
                plane_axis=payload.plane_axis,
                spatial_origin_yx=payload.spatial_origin_yx,
                source_spatial_shape_yx=payload.source_spatial_shape_yx,
                source_path=payload.source_path,
                source_component_metadata=payload.source_component_metadata,
                channel_source_paths=payload.channel_source_paths,
                channel_source_component_metadata=payload.channel_source_component_metadata,
                source_image_names=payload.source_image_names,
                dimensions=schema.dimensions,
                source_image_name=schema.source_image_name,
                representation=(
                    schema.label_representation
                    or ObjectLabelRepresentation.DENSE_LABELS
                ),
            )
        return cls(
            name=value.name,
            labels=payload,
            dimensions=schema.dimensions,
            source_image_name=schema.source_image_name,
            representation=(
                schema.label_representation
                or ObjectLabelRepresentation.DENSE_LABELS
            ),
        )

    def __post_init__(self, *source_provenance_values: object) -> None:
        self.absorb_explicit_source_provenance(
            SourceImageProvenance.from_init_values(source_provenance_values)
        )
        ObjectLabelSetInitialization(self).apply()

    def runtime_payload(self) -> Any:
        if (
            self.unedited_labels is not None
            or self.small_removed_labels is not None
            or self.declared_object_count is not None
            or self.declared_object_ids
            or self.declared_object_id_domains
            or self.domain_scope is not ObjectLabelDomainScope.PAYLOAD
            or self.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE
            or self.spatial_origin_yx is not None
            or self.source_spatial_shape_yx is not None
            or self.source_path is not None
            or self.source_component_metadata is not None
            or self.channel_source_paths
            or self.channel_source_component_metadata
            or self.source_image_names
        ):
            return ObjectLabelPayload.from_domain_metadata(
                self,
                labels=self.labels,
                unedited_labels=self.unedited_labels,
                small_removed_labels=self.small_removed_labels,
            )
        return self.labels

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        label_variants = (
            payload.variants
            if isinstance(payload, ObjectLabelPayload)
            else (ObjectLabelVariant.FINAL,)
        )
        return RuntimeValueSchema(
            kind=ArtifactKind.OBJECT_LABELS,
            dimensions=self.dimensions,
            label_representation=self.representation,
            label_variants=label_variants,
            object_name=self.name,
            source_image_name=self.source_image_name,
        )

    @property
    def shape(self) -> Any:
        return self.labels.shape

    @property
    def ndim(self) -> int:
        return self.labels.ndim

    @property
    def dtype(self) -> Any:
        return self.labels.dtype

    def __array__(self, dtype: Any | None = None) -> Any:
        import numpy as np

        return np.asarray(self.labels, dtype=dtype)

    def labels_for_variant(
        self,
        variant: ObjectLabelVariant | str,
    ) -> ObjectLabelData:
        payload = ObjectLabelPayload(
            labels=self.labels,
            unedited_labels=self.unedited_labels,
            small_removed_labels=self.small_removed_labels,
        )
        return payload.labels_for_variant(variant)

    def with_labels(
        self,
        labels: ObjectLabelData,
        *,
        unedited_labels: ObjectLabelData | None = None,
        small_removed_labels: ObjectLabelData | None = None,
    ) -> "ObjectLabelSet":
        """Return this runtime value's metadata with replacement labels."""
        return ObjectLabelSet(
            name=self.name,
            labels=ObjectLabelSetReplacementStrategy.for_source(self).replacement_labels(
                labels
            ),
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
            representation=self.representation,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
            plane_axis=self.plane_axis,
            spatial_origin_yx=self.spatial_origin_yx,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
            source_path=self.source_path,
            source_component_metadata=self.source_component_metadata,
            channel_source_paths=self.channel_source_paths,
            channel_source_component_metadata=self.channel_source_component_metadata,
            source_image_names=self.source_image_names,
            dimensions=self.dimensions,
            source_image_name=self.source_image_name,
        )

    def with_projected_plane(
        self,
        labels: ObjectLabelData,
        plane_index: int,
        *,
        unedited_labels: ObjectLabelData | None = None,
        small_removed_labels: ObjectLabelData | None = None,
    ) -> "ObjectLabelSet":
        """Return one selected label-set plane with projected domain metadata."""
        return object_label_set_with_projected_plane(
            self,
            labels,
            plane_index,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
        )

    def with_source_image_context(self, image: RuntimeArrayData) -> "ObjectLabelSet":
        """Attach image provenance without changing this label set's domain."""
        metadata = image_payload_metadata(image)
        source_provenance = ObjectLabelSourceContextProvenanceMerge.for_label_and_image(
            self,
            image,
        ).provenance()
        return ObjectLabelSet(
            name=self.name,
            labels=self.labels,
            unedited_labels=self.unedited_labels,
            small_removed_labels=self.small_removed_labels,
            representation=self.representation,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
            plane_axis=self.plane_axis,
            spatial_origin_yx=(
                self.spatial_origin_yx
                if self.spatial_origin_yx is not None
                else metadata.spatial_origin_yx
            ),
            source_spatial_shape_yx=(
                self.source_spatial_shape_yx
                if self.source_spatial_shape_yx is not None
                else metadata.source_spatial_shape_yx
            ),
            source_path=source_provenance.source_path,
            source_component_metadata=source_provenance.source_component_metadata,
            channel_source_paths=source_provenance.channel_source_paths,
            channel_source_component_metadata=(
                source_provenance.channel_source_component_metadata
            ),
            source_image_names=source_provenance.source_image_names,
            dimensions=self.dimensions,
            source_image_name=self.source_image_name,
        )

@dataclass(frozen=True, slots=True)
class ObjectLabelSetInitialization:
    """Initialization authority for native object-label values."""

    target: ObjectLabelSet

    def apply(self) -> None:
        NativeRuntimeValue.__post_init__(self.target)
        self.target._validate_source_image_context(type(self.target).__name__)
        representation = self.normalize_representation()
        self.absorb_payload_metadata()
        self.normalize_domain_metadata()
        self.normalize_provenance_metadata()
        self.validate_payload(representation)

    def normalize_representation(self) -> ObjectLabelRepresentation:
        representation = coerce_enum(
            ObjectLabelRepresentation,
            self.target.representation,
            "ObjectLabelSet.representation",
        )
        self.target.representation = representation
        return representation

    def absorb_payload_metadata(self) -> None:
        if not isinstance(self.target.labels, ObjectLabelPayload):
            return
        payload = self.target.labels
        self.target.labels = payload.labels
        self.target.unedited_labels = payload.unedited_labels
        self.target.small_removed_labels = payload.small_removed_labels
        ObjectLabelPayloadMetadataAbsorption(self.target, payload).apply()

    def normalize_domain_metadata(self) -> None:
        if self.target.declared_object_count is not None:
            count = int(self.target.declared_object_count)
            if count < 0:
                raise ValueError("ObjectLabelSet.declared_object_count cannot be negative.")
            self.target.declared_object_count = count

        ids = tuple(int(object_id) for object_id in self.target.declared_object_ids)
        if any(object_id <= 0 for object_id in ids):
            raise ValueError("ObjectLabelSet.declared_object_ids must be positive.")
        self.target.declared_object_ids = tuple(sorted(dict.fromkeys(ids)))
        self.target.declared_object_id_domains = tuple(
                ObjectLabelDomain._normalize_ids(domain, "declared_object_id_domains")
                for domain in self.target.declared_object_id_domains
            )
        self.target.domain_scope = coerce_enum(
                ObjectLabelDomainScope,
                self.target.domain_scope,
                "ObjectLabelSet.domain_scope",
            )
        self.target.plane_axis = coerce_enum(
                RuntimePlaneAxis,
                self.target.plane_axis,
                "ObjectLabelSet.plane_axis",
            )
        if self.target.spatial_origin_yx is not None:
            self.target.spatial_origin_yx = _spatial_shape_pair(self.target.spatial_origin_yx, "spatial_origin_yx")
        if self.target.source_spatial_shape_yx is not None:
            self.target.source_spatial_shape_yx = _spatial_shape_pair(
                    self.target.source_spatial_shape_yx,
                    "source_spatial_shape_yx",
                )

    def normalize_provenance_metadata(self) -> None:
        self.target.normalize_source_provenance_fields()

    def validate_payload(self, representation: ObjectLabelRepresentation) -> None:
        validator = _PAYLOAD_VALIDATORS[representation.payload_shape]
        if validator is not None and not validator(self.target.labels):
            raise TypeError(
                f"ObjectLabelSet '{self.target.name}' requires "
                f"{representation.value} payload, got "
                f"{type(self.target.labels).__name__}."
            )
        _validate_object_label_variant(
            self.target.name,
            "unedited_labels",
            self.target.labels,
            self.target.unedited_labels,
            validator,
        )
        _validate_object_label_variant(
            self.target.name,
            "small_removed_labels",
            self.target.labels,
            self.target.small_removed_labels,
            validator,
        )


@dataclass(frozen=True, slots=True)
class ObjectLabelPayloadMetadataAbsorption:
    """Copies payload-owned metadata into an ObjectLabelSet when not overridden."""

    target: ObjectLabelSet
    payload: ObjectLabelPayload

    def apply(self) -> None:
        self.absorb_declared_domains()
        self.absorb_label_domain_axes()
        self.absorb_spatial_domains()
        self.absorb_source_provenance()

    def absorb_declared_domains(self) -> None:
        if self.target.declared_object_count is None:
            self.target.declared_object_count = self.payload.declared_object_count
        if not self.target.declared_object_ids:
            self.target.declared_object_ids = self.payload.declared_object_ids
        if not self.target.declared_object_id_domains:
            self.target.declared_object_id_domains = self.payload.declared_object_id_domains

    def absorb_label_domain_axes(self) -> None:
        self.target.domain_scope = self.payload.domain_scope
        self.target.plane_axis = self.payload.plane_axis

    def absorb_spatial_domains(self) -> None:
        if self.target.spatial_origin_yx is None:
            self.target.spatial_origin_yx = self.payload.spatial_origin_yx
        if self.target.source_spatial_shape_yx is None:
            self.target.source_spatial_shape_yx = self.payload.source_spatial_shape_yx

    def absorb_source_provenance(self) -> None:
        provenance = self.target.source_provenance.with_missing_from(
            self.payload.source_provenance
        )
        self.target.source_path = provenance.source_path
        self.target.source_component_metadata = provenance.source_component_metadata
        self.target.channel_source_paths = provenance.channel_source_paths
        self.target.channel_source_component_metadata = provenance.channel_source_component_metadata
        self.target.source_image_names = provenance.source_image_names


class RuntimePayloadDataStrategy(ABC):
    """Base contract for nominal payload-to-data extraction strategies."""

    value_type: ClassVar[type[object] | None] = None

    @abstractmethod
    def data(self, payload: object) -> object:
        """Return the concrete data represented by payload."""


class ObjectLabelDenseDataStrategy(
    NominalTypeKeyedStrategyMixin,
    RuntimePayloadDataStrategy,
    metaclass=AutoRegisterMeta,
):
    """Registered dense-label extractor for one nominal object-label runtime type."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)
    value_type_label: ClassVar[str | None] = None

    @classmethod
    def for_payload(cls, payload: object) -> "ObjectLabelDenseDataStrategy":
        strategy = cls.for_nominal_value(payload)
        return strategy if strategy is not None else RawObjectLabelDenseDataStrategy()

    @classmethod
    def spatial_rank(cls, payload: object) -> int | None:
        """Return object-label dense spatial rank when the payload can materialize it."""
        dense_data = cls.for_payload(payload).data(payload)
        if isinstance(dense_data, np.ndarray):
            return int(dense_data.ndim)
        return int(np.asarray(dense_data).ndim)

    @abstractmethod
    def data(self, payload: object) -> object:
        """Return the dense label data represented by payload."""


class ObjectLabelContainerDenseDataStrategy(ObjectLabelDenseDataStrategy):
    """Dense-label extractor for payloads with label-container semantics."""

    value_type: ClassVar[type[object] | None] = None

    def data(self, payload: object) -> object:
        value_type = type(self).value_type
        if value_type is None:
            raise TypeError(
                f"{type(self).__name__} must declare a concrete value_type."
            )
        if not isinstance(payload, value_type):
            raise TypeError(
                f"{type(self).__name__} requires {value_type.__name__}, "
                f"got {type(payload).__name__}."
            )
        if isinstance(payload.labels, SparseIJVLabelRows):
            return payload.labels.to_dense(
                source_spatial_shape_yx=payload.source_spatial_shape_yx,
            )
        return payload.labels


class ObjectLabelPayloadDenseDataStrategy(ObjectLabelContainerDenseDataStrategy):
    """Extract dense labels from serialized object-label payloads."""

    value_type = ObjectLabelPayload


class ObjectLabelSetDenseDataStrategy(ObjectLabelContainerDenseDataStrategy):
    """Extract dense labels from native object-label runtime values."""

    value_type = ObjectLabelSet


class RawObjectLabelDenseDataStrategy(ObjectLabelDenseDataStrategy):
    """Pass through already-dense array payloads."""

    def data(self, payload: object) -> object:
        return payload


class ObjectLabelPayloadIdDomainStrategy(ObjectLabelIdDomainStrategy):
    """Extract present object IDs from serialized object-label payloads."""

    value_type = ObjectLabelPayload

    def present_ids(self, labels: Any) -> tuple[int, ...]:
        if not isinstance(labels, ObjectLabelPayload):
            raise TypeError(
                "ObjectLabelPayloadIdDomainStrategy requires ObjectLabelPayload, "
                f"got {type(labels).__name__}."
            )
        return ObjectLabelIdDomainStrategy.for_value(labels.labels).present_ids(labels.labels)


class ObjectLabelSetIdDomainStrategy(ObjectLabelIdDomainStrategy):
    """Extract present object IDs from native object-label runtime values."""

    value_type = ObjectLabelSet

    def present_ids(self, labels: Any) -> tuple[int, ...]:
        if not isinstance(labels, ObjectLabelSet):
            raise TypeError(
                "ObjectLabelSetIdDomainStrategy requires ObjectLabelSet, "
                f"got {type(labels).__name__}."
            )
        return ObjectLabelIdDomainStrategy.for_value(labels.labels).present_ids(labels.labels)


def object_label_dense_array(
    payload: object,
    *,
    dtype: object | None = None,
    copy: bool | None = None,
) -> np.ndarray:
    """Materialize object-label dense data as a NumPy array via nominal extraction."""
    dense_data = ObjectLabelDenseDataStrategy.for_payload(payload).data(payload)
    if copy is None:
        return np.asarray(dense_data, dtype=dtype)
    return np.array(dense_data, dtype=dtype, copy=copy)


@dataclass(frozen=True, slots=True)
class DenseObjectLabelAggregation:
    """Vectorized reductions over dense object-label IDs."""

    labels: np.ndarray
    object_count: int

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        dtype: object = np.int32,
    ) -> "DenseObjectLabelAggregation":
        """Build reductions from any nominal object-label payload."""
        labels = object_label_dense_array(payload, dtype=dtype)
        return cls(
            labels=labels,
            object_count=int(np.max(labels)) if labels.size else 0,
        )

    def counts(self) -> np.ndarray:
        """Return per-object pixel counts excluding the background label."""
        return np.bincount(
            self.labels,
            minlength=self.object_count + 1,
        )[1:].astype(float, copy=False)

    def sum(self, values: object) -> np.ndarray:
        """Return per-object sums for values aligned with ``labels``."""
        return np.bincount(
            self.labels,
            weights=np.asarray(values, dtype=float),
            minlength=self.object_count + 1,
        )[1:].astype(float, copy=False)

    def maximum(self, values: object) -> np.ndarray:
        """Return per-object maxima for values aligned with ``labels``."""
        maxima = np.zeros(self.object_count + 1, dtype=float)
        np.maximum.at(maxima, self.labels, np.asarray(values, dtype=float))
        return maxima[1:]

    def subset(self, mask: object) -> "DenseObjectLabelAggregation":
        """Return reductions over a masked subset of the same object ID domain."""
        return DenseObjectLabelAggregation(
            labels=self.labels[np.asarray(mask, dtype=bool)],
            object_count=self.object_count,
        )


class ObjectLabelDataRuntimeSliceStackContract(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declare runtime-slice preservation for a concrete label representation."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def preserves_runtime_slice_stack(
        cls,
        labels: ObjectLabelData,
        *,
        plane_axis: RuntimePlaneAxis,
        slice_count: int,
    ) -> bool:
        if plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return False
        strategy = cls.for_nominal_value(labels)
        if strategy is None:
            return False
        return strategy.label_data_preserves_runtime_slice_stack(
            labels,
            slice_count=slice_count,
        )

    @classmethod
    def runtime_slice_count(
        cls,
        labels: ObjectLabelData,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> int | None:
        """Return the runtime-slice count encoded by this label data."""
        if plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return None
        strategy = cls.for_nominal_value(labels)
        if strategy is None:
            return None
        return strategy.label_data_runtime_slice_count(labels)

    @abstractmethod
    def label_data_preserves_runtime_slice_stack(
        self,
        labels: ObjectLabelData,
        *,
        slice_count: int,
    ) -> bool:
        """Return whether this label representation carries one row per runtime slice."""

    @abstractmethod
    def label_data_runtime_slice_count(
        self,
        labels: ObjectLabelData,
    ) -> int | None:
        """Return the runtime-slice count carried by this label representation."""


class SparseIJVLabelRowsRuntimeSliceStackContract(
    ObjectLabelDataRuntimeSliceStackContract
):
    """Sparse IJV labels preserve runtime slicing when they declare slice indexes."""

    value_type = SparseIJVLabelRows

    def label_data_preserves_runtime_slice_stack(
        self,
        labels: ObjectLabelData,
        *,
        slice_count: int,
    ) -> bool:
        del slice_count
        if not isinstance(labels, SparseIJVLabelRows):
            raise TypeError(
                "SparseIJVLabelRowsRuntimeSliceStackContract requires "
                f"SparseIJVLabelRows, got {type(labels).__name__}."
            )
        return labels.has_slice_index

    def label_data_runtime_slice_count(
        self,
        labels: ObjectLabelData,
    ) -> int | None:
        if not isinstance(labels, SparseIJVLabelRows):
            raise TypeError(
                "SparseIJVLabelRowsRuntimeSliceStackContract requires "
                f"SparseIJVLabelRows, got {type(labels).__name__}."
            )
        if not labels.has_slice_index:
            return None
        return labels.label_data_runtime_slice_count()


class DenseArrayLabelRuntimeSliceStackContract(
    ObjectLabelDataRuntimeSliceStackContract
):
    """Dense array labels preserve runtime slicing when axis 0 is the slice axis."""

    value_type = np.ndarray

    def label_data_preserves_runtime_slice_stack(
        self,
        labels: ObjectLabelData,
        *,
        slice_count: int,
    ) -> bool:
        if not isinstance(labels, np.ndarray):
            raise TypeError(
                "DenseArrayLabelRuntimeSliceStackContract requires ndarray, "
                f"got {type(labels).__name__}."
            )
        return labels.ndim >= 3 and int(labels.shape[0]) == int(slice_count)

    def label_data_runtime_slice_count(
        self,
        labels: ObjectLabelData,
    ) -> int | None:
        if not isinstance(labels, np.ndarray):
            raise TypeError(
                "DenseArrayLabelRuntimeSliceStackContract requires ndarray, "
                f"got {type(labels).__name__}."
            )
        if labels.ndim < 3:
            return None
        return int(labels.shape[0])


class ObjectLabelDataPlaneStackContract(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declare the leading plane count carried by object-label data."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def plane_count(cls, labels: ObjectLabelData) -> int | None:
        """Return the leading plane count for stack-shaped label data."""
        strategy = cls.for_nominal_value(labels)
        if strategy is None:
            return None
        return strategy.label_data_plane_count(labels)

    @abstractmethod
    def label_data_plane_count(self, labels: ObjectLabelData) -> int | None:
        """Return the number of semantic planes carried by this label data."""


class SparseIJVLabelRowsPlaneStackContract(ObjectLabelDataPlaneStackContract):
    """Sparse IJV labels carry planes when they declare slice indexes."""

    value_type = SparseIJVLabelRows

    def label_data_plane_count(self, labels: ObjectLabelData) -> int | None:
        if not isinstance(labels, SparseIJVLabelRows):
            raise TypeError(
                "SparseIJVLabelRowsPlaneStackContract requires SparseIJVLabelRows, "
                f"got {type(labels).__name__}."
            )
        if not labels.has_slice_index:
            return None
        return labels.label_data_runtime_slice_count()


class DenseArrayLabelPlaneStackContract(ObjectLabelDataPlaneStackContract):
    """Dense labels carry a plane stack when they have a leading stack axis."""

    value_type = np.ndarray

    def label_data_plane_count(self, labels: ObjectLabelData) -> int | None:
        if not isinstance(labels, np.ndarray):
            raise TypeError(
                "DenseArrayLabelPlaneStackContract requires ndarray, "
                f"got {type(labels).__name__}."
            )
        if labels.ndim < 3:
            return None
        return int(labels.shape[0])


class ObjectLabelPlaneStackContract(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declare whether an object-label payload carries a semantic plane stack."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def plane_count(cls, value: object) -> int | None:
        """Return the number of semantic planes carried by one label value."""
        strategy = cls.for_nominal_value(value)
        if strategy is None:
            return None
        return strategy.value_plane_count(value)

    @abstractmethod
    def value_plane_count(self, value: object) -> int | None:
        """Return the semantic plane count carried by this object-label value."""


class ObjectLabelContainerContract(ABC):
    """Shared typed-value validation for object-label container contracts."""

    value_type: ClassVar[type[Any] | None] = None

    def typed_value(self, value: object) -> Any:
        """Return value after validating it belongs to this registered family."""
        value_type = type(self).value_type
        if value_type is None:
            raise TypeError(
                f"{type(self).__name__} must declare a concrete value_type."
            )
        if not isinstance(value, value_type):
            raise TypeError(
                f"{type(self).__name__} requires {value_type.__name__}, "
                f"got {type(value).__name__}."
            )
        return value


class ObjectLabelContainerPlaneStackContract(
    ObjectLabelContainerContract,
    ObjectLabelPlaneStackContract,
):
    """Plane-stack contract shared by nominal object-label containers."""

    def value_plane_count(self, value: object) -> int | None:
        payload = self.typed_value(value)
        data_count = ObjectLabelDataPlaneStackContract.plane_count(payload.labels)
        if payload.domain_scope is not ObjectLabelDomainScope.PLANE:
            return ObjectLabelSourceBackedPlaneCountResolution(
                payload,
                data_count,
            ).result.count
        declared_count = (
            len(payload.declared_object_id_domains)
            if payload.declared_object_id_domains
            else None
        )
        if declared_count is not None and data_count is not None:
            ObjectLabelPlaneDomainCardinalityResolution(
                payload_type=type(payload).__name__,
                declared_count=declared_count,
                data_count=data_count,
            ).validate()
            if int(declared_count) != int(data_count):
                return data_count
        return declared_count if declared_count is not None else data_count


class ObjectLabelSourceBackedPlaneCountAbsenceReason(str, Enum):
    """Reasons source provenance does not establish an object-label plane count."""

    NOT_PLANE_STACK = "not_plane_stack"
    NO_SOURCE_PROVENANCE = "no_source_provenance"


@dataclass(frozen=True, slots=True)
class ObjectLabelSourceBackedPlaneCountResult:
    """Typed result for source-provenanced object-label plane counts."""

    count: int | None
    absence_reason: ObjectLabelSourceBackedPlaneCountAbsenceReason | None = None


@dataclass(frozen=True, slots=True)
class ObjectLabelSourceBackedPlaneCountResolution:
    """Resolve source-provenanced stacks that lack explicit domain metadata."""

    payload: ObjectLabelValue
    data_count: int | None

    @property
    def result(self) -> ObjectLabelSourceBackedPlaneCountResult:
        if self.data_count is None or self.data_count <= 1:
            return ObjectLabelSourceBackedPlaneCountResult(
                None,
                ObjectLabelSourceBackedPlaneCountAbsenceReason.NOT_PLANE_STACK,
            )
        if not self.source_counts:
            return ObjectLabelSourceBackedPlaneCountResult(
                None,
                ObjectLabelSourceBackedPlaneCountAbsenceReason.NO_SOURCE_PROVENANCE,
            )
        self.validate_source_counts()
        return ObjectLabelSourceBackedPlaneCountResult(self.data_count)

    @property
    def source_counts(self) -> tuple[int, ...]:
        return self.payload.source_provenance.plane_count_sources

    def validate_source_counts(self) -> None:
        if any(count != self.data_count for count in self.source_counts):
            raise ValueError(
                f"{type(self.payload).__name__} source provenance declares plane counts "
                f"{self.source_counts!r}, but label data carries {self.data_count} planes."
            )


@dataclass(frozen=True, slots=True)
class ObjectLabelPlaneDomainCardinalityResolution:
    """Validate when label data may override stale declared plane domains."""

    payload_type: str
    declared_count: int
    data_count: int

    def validate(self) -> None:
        if self.declared_count == self.data_count:
            return
        if self.declared_count > 0 and self.data_count > 0:
            return
        raise ValueError(
            f"{self.payload_type} declares {self.declared_count} plane domains "
            f"but label data carries {self.data_count} planes."
        )


class ObjectLabelPayloadPlaneStackContract(ObjectLabelContainerPlaneStackContract):
    """Plane-stack contract for dense object-label payloads."""

    value_type = ObjectLabelPayload


class ObjectLabelSetPlaneStackContract(ObjectLabelContainerPlaneStackContract):
    """Plane-stack contract for native object-label sets."""

    value_type = ObjectLabelSet


class ObjectLabelRuntimeSliceStackContract(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declare whether an object-label payload carries the runtime slice axis."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def preserves_runtime_slice_stack(
        cls,
        value: object,
        *,
        slice_count: int,
    ) -> bool:
        strategy = cls.for_nominal_value(value)
        return (
            False
            if strategy is None
            else strategy.value_preserves_runtime_slice_stack(
                value,
                slice_count=slice_count,
            )
        )

    @classmethod
    def runtime_slice_count(cls, value: object) -> int | None:
        """Return the runtime-slice count encoded by one object-label value."""
        strategy = cls.for_nominal_value(value)
        if strategy is None:
            return None
        return strategy.value_runtime_slice_count(value)

    @abstractmethod
    def value_preserves_runtime_slice_stack(
        self,
        value: object,
        *,
        slice_count: int,
    ) -> bool:
        """Return whether the payload must remain stack-shaped until execution."""

    @abstractmethod
    def value_runtime_slice_count(self, value: object) -> int | None:
        """Return the runtime-slice count carried by this object-label value."""


class ObjectLabelContainerRuntimeSliceStackContract(
    ObjectLabelContainerContract,
    ObjectLabelRuntimeSliceStackContract,
):
    """Runtime-slice contract shared by nominal object-label containers."""

    def value_preserves_runtime_slice_stack(
        self,
        value: object,
        *,
        slice_count: int,
    ) -> bool:
        payload = self.typed_value(value)
        if payload.domain_scope is not ObjectLabelDomainScope.PLANE:
            return False
        return ObjectLabelDataRuntimeSliceStackContract.preserves_runtime_slice_stack(
            payload.labels,
            plane_axis=payload.plane_axis,
            slice_count=slice_count,
        )

    def value_runtime_slice_count(self, value: object) -> int | None:
        payload = self.typed_value(value)
        if payload.domain_scope is not ObjectLabelDomainScope.PLANE:
            return None
        return ObjectLabelDataRuntimeSliceStackContract.runtime_slice_count(
            payload.labels,
            plane_axis=payload.plane_axis,
        )


class ObjectLabelPayloadRuntimeSliceStackContract(
    ObjectLabelContainerRuntimeSliceStackContract
):
    """Runtime-slice contract for dense object-label payloads."""

    value_type = ObjectLabelPayload


class ObjectLabelSetRuntimeSliceStackContract(
    ObjectLabelContainerRuntimeSliceStackContract
):
    """Runtime-slice contract for native object-label sets."""

    value_type = ObjectLabelSet


class ObjectLabelPure2DSliceAggregator(ABC, metaclass=AutoRegisterMeta):
    """Aggregate object-label slices while preserving label-domain semantics."""

    __registry_key__ = "label_type"
    __registry__: ClassVar[
        dict[type[ObjectLabelValue], type["ObjectLabelPure2DSliceAggregator"]]
    ] = {}
    label_type: ClassVar[type[ObjectLabelValue] | None] = None

    def __init__(
        self,
        values: Sequence[ObjectLabelValue],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis = RuntimePlaneAxis.RUNTIME_SLICE,
        force_plane_axis: RuntimePlaneAxis | None = None,
    ) -> None:
        self.values: tuple[ObjectLabelValue, ...] = tuple(values)
        self.memory_type = memory_type
        self.requested_plane_axis = plane_axis
        self.force_plane_axis = force_plane_axis

    @classmethod
    def aggregate(
        cls,
        values: Sequence[ObjectLabelValue],
        memory_type: str,
        *,
        plane_axis: RuntimePlaneAxis = RuntimePlaneAxis.RUNTIME_SLICE,
        force_plane_axis: RuntimePlaneAxis | None = None,
    ) -> ObjectLabelValue:
        for aggregator_type in cls.__registry__.values():
            if aggregator_type.supports(values):
                return aggregator_type(
                    values,
                    memory_type,
                    plane_axis=plane_axis,
                    force_plane_axis=force_plane_axis,
                ).aggregate_values()
        raise TypeError("No object-label slice aggregator owns these values.")

    @classmethod
    def supports(cls, values: Sequence[ObjectLabelValue]) -> bool:
        return cls.label_type is not None and bool(values) and all(
            isinstance(value, cls.label_type) for value in values
        )

    @property
    def first(self) -> ObjectLabelValue:
        return self.values[0]

    @property
    def declared_object_count(self) -> int | None:
        return self.common_value(value.declared_object_count for value in self.values)

    @property
    def declared_object_ids(self) -> tuple[int, ...]:
        common_ids = self.common_value(
            value.declared_object_ids for value in self.values
        )
        if common_ids is None:
            return ()
        return common_ids

    @property
    def declared_object_id_domains(self) -> tuple[tuple[int, ...], ...]:
        if len(self.values) == 1:
            return ObjectLabelDomain.explicit_plane_id_domains(
                value.object_label_domain() for value in self.values
            )
        return tuple(
            plane_domain
            for value in self.values
            for plane_domain in self.value_plane_id_domains(value)
        )

    def value_plane_id_domains(
        self,
        value: ObjectLabelValue,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the object-id domain represented by one PURE_2D output slice."""
        return ObjectLabelPlaneDomainStrategy.for_enum_member(
            value.domain_scope
        ).identity_domains(
            value.labels_for_variant(ObjectLabelVariant.FINAL),
            declared_object_count=value.declared_object_count,
            declared_object_ids=value.declared_object_ids,
            declared_object_id_domains=value.declared_object_id_domains,
        )

    @property
    def domain_scope(self) -> ObjectLabelDomainScope:
        if len(self.values) > 1:
            return ObjectLabelDomainScope.PLANE
        return ObjectLabelDomainScope.common(value.domain_scope for value in self.values)

    @property
    def plane_axis(self) -> RuntimePlaneAxis:
        if self.force_plane_axis is not None:
            return self.force_plane_axis
        if len(self.values) == 1:
            return self.first.plane_axis
        source_image_names = self.source_image_names
        if len(source_image_names) > 1 and len(set(source_image_names)) > 1:
            return RuntimePlaneAxis.SOURCE_BINDING
        if (
            self.requested_plane_axis is RuntimePlaneAxis.SOURCE_BINDING
            and self.source_provenance_varies
        ):
            return RuntimePlaneAxis.SOURCE_BINDING
        axes = {value.plane_axis for value in self.values}
        if len(axes) == 1:
            return axes.pop()
        return RuntimePlaneAxis.RUNTIME_SLICE

    @property
    def source_provenance_varies(self) -> bool:
        identities = tuple(
            self.value_source_provenance(slice_index, value).identity()
            for slice_index, value in enumerate(self.values)
        )
        present = tuple(identity for identity in identities if identity != (None, None))
        if len(present) <= 1:
            return False
        first = present[0]
        return any(identity != first for identity in present[1:])

    @property
    def source_image_names(self) -> tuple[str, ...]:
        return tuple(
            source_image_name
            for value in self.values
            for source_image_name in value.source_image_names
        )

    @property
    def source_spatial_shape_yx(self) -> tuple[int, int] | None:
        return self.common_value(value.source_spatial_shape_yx for value in self.values)

    @property
    def source_path(self) -> str | None:
        return self.common_value(
            self.value_source_provenance(slice_index, value).source_path
            for slice_index, value in enumerate(self.values)
        )

    @property
    def source_component_metadata(self) -> SourceComponentMetadata | None:
        values = tuple(
            self.value_source_provenance(
                slice_index,
                value,
            ).source_component_metadata
            for slice_index, value in enumerate(self.values)
        )
        present = tuple(value for value in values if value is not None)
        if not present:
            return None
        first = dict(present[0])
        if all(dict(value) == first for value in present):
            return MappingProxyType(first)
        return None

    @property
    def channel_source_paths(self) -> tuple[str | None, ...]:
        paths = tuple(
            self.value_source_provenance(slice_index, value).source_path
            for slice_index, value in enumerate(self.values)
        )
        if any(path is not None for path in paths):
            return paths
        common_paths = self.common_value(
            value.channel_source_paths for value in self.values
        )
        if common_paths is None:
            return ()
        return common_paths

    @property
    def channel_source_component_metadata(self) -> ChannelSourceComponentMetadata:
        metadata = tuple(
            self.value_source_provenance(
                slice_index,
                value,
            ).source_component_metadata
            for slice_index, value in enumerate(self.values)
        )
        if any(item is not None for item in metadata):
            return metadata
        channel_metadata = tuple(
            value.channel_source_component_metadata for value in self.values
        )
        present = tuple(value for value in channel_metadata if value)
        if not present:
            return ()
        first = self.channel_source_component_identity(present[0])
        if all(
            self.channel_source_component_identity(value) == first
            for value in present
        ):
            return present[0]
        return ()

    @staticmethod
    def channel_source_component_identity(
        metadata: ChannelSourceComponentMetadata,
    ) -> tuple[SourceProvenanceIdentity, ...]:
        return tuple(
            SourceImageProvenance(
                source_component_metadata=component_metadata,
            ).identity()
            for component_metadata in metadata
        )

    @staticmethod
    def value_source_provenance(
        slice_index: int,
        value: ObjectLabelValue,
    ) -> SourceImageProvenance:
        """Return the source provenance represented by one aggregate slice."""
        return value.source_provenance.for_channel(slice_index)

    @property
    def spatial_origin_yx(self) -> tuple[int, int] | None:
        if self.expands_to_source_domain:
            return None
        return self.common_value(value.spatial_origin_yx for value in self.values)

    @property
    def expands_to_source_domain(self) -> bool:
        if len(self.values) <= 1:
            return False
        domains = tuple(
            (value.spatial_origin_yx, value.source_spatial_shape_yx)
            for value in self.values
        )
        if any(origin is None or source_shape is None for origin, source_shape in domains):
            return False
        return len(set(domains)) > 1

    @staticmethod
    def common_value(
        values: Iterable[MetadataValueT],
    ) -> MetadataValueT | None:
        values_tuple = tuple(values)
        if not values_tuple:
            return None
        first = values_tuple[0]
        if all(value == first for value in values_tuple):
            return first
        return None

    def aggregate_values(self) -> ObjectLabelValue:
        return self.output_value(
            labels=self.aggregate_variant(ObjectLabelVariant.FINAL),
            unedited_labels=self.aggregate_optional_variant(
                ObjectLabelVariant.UNEDITED
            ),
            small_removed_labels=self.aggregate_optional_variant(
                ObjectLabelVariant.SMALL_REMOVED
            ),
        )

    def aggregate_optional_variant(
        self,
        variant: ObjectLabelVariant,
    ) -> ObjectLabelData | None:
        if ObjectLabelVariantDataStrategy.for_enum_member(variant).present(self.values):
            return self.aggregate_variant(variant)
        return None

    def aggregate_variant(self, variant: ObjectLabelVariant) -> ObjectLabelData:
        return stack_runtime_object_label_slices(
            [self.slice_labels(value, variant) for value in self.values],
            self.memory_type,
        )

    def slice_labels(
        self,
        value: ObjectLabelValue,
        variant: ObjectLabelVariant,
    ) -> ObjectLabelData:
        if not self.expands_to_source_domain:
            return value.labels_for_variant(variant)
        domain_value = self.domain_value_for_variant(value, variant)
        aligned, _ = DenseObjectLabelPairAligner(domain_value, domain_value).aligned()
        return aligned

    @abstractmethod
    def domain_value_for_variant(
        self,
        value: ObjectLabelValue,
        variant: ObjectLabelVariant,
    ) -> ObjectLabelValue:
        """Return a typed label value carrying the selected variant and domain."""

    @abstractmethod
    def output_value(
        self,
        *,
        labels: ObjectLabelData,
        unedited_labels: ObjectLabelData | None,
        small_removed_labels: ObjectLabelData | None,
    ) -> ObjectLabelValue:
        """Build the aggregated object-label value."""


class ObjectLabelPayloadPure2DSliceAggregator(ObjectLabelPure2DSliceAggregator):
    """Aggregate dense object-label payload slices."""

    label_type = ObjectLabelPayload

    def domain_value_for_variant(
        self,
        value: ObjectLabelPayload,
        variant: ObjectLabelVariant,
    ) -> ObjectLabelPayload:
        return ObjectLabelPayload.from_domain_metadata(
            value,
            labels=value.labels_for_variant(variant),
        )

    def output_value(
        self,
        *,
        labels: ObjectLabelData,
        unedited_labels: ObjectLabelData | None,
        small_removed_labels: ObjectLabelData | None,
    ) -> ObjectLabelPayload:
        return ObjectLabelPayload.from_domain_metadata(
            self,
            labels=labels,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
            plane_axis=self.plane_axis,
        )


class ObjectLabelSetPure2DSliceAggregator(ObjectLabelPure2DSliceAggregator):
    """Aggregate native object-label set slices."""

    label_type = ObjectLabelSet

    @property
    def representation(self) -> ObjectLabelRepresentation:
        representations = {value.representation for value in self.values}
        if len(representations) != 1:
            raise ValueError(
                "Cannot aggregate mixed object-label representations across PURE_2D slices."
            )
        return representations.pop()

    @property
    def source_image_names(self) -> tuple[str, ...]:
        names = super().source_image_names
        if names:
            return names
        return tuple(
            value.source_image_name
            for value in self.values
            if value.source_image_name is not None
        )

    def aggregate_values(self) -> ObjectLabelSet:
        if self.representation is ObjectLabelRepresentation.SPARSE_IJV:
            return self.aggregate_sparse_ijv()
        return super().aggregate_values()

    def aggregate_sparse_ijv(self) -> ObjectLabelSet:
        return ObjectLabelSet(
            name=self.first.name,
            labels=SparseIJVLabelRows.from_slices(
                tuple(
                    value.labels
                    if isinstance(value.labels, SparseIJVLabelRows)
                    else SparseIJVLabelRows.from_yx_label(value.labels)
                    for value in self.values
                )
            ),
            representation=self.representation,
            dimensions=self.first.dimensions,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
            plane_axis=self.plane_axis,
            source_path=self.source_path,
            source_component_metadata=self.source_component_metadata,
            channel_source_paths=self.channel_source_paths,
            channel_source_component_metadata=self.channel_source_component_metadata,
            source_image_names=self.source_image_names,
            source_image_name=self.first.source_image_name,
        )

    def domain_value_for_variant(
        self,
        value: ObjectLabelSet,
        variant: ObjectLabelVariant,
    ) -> ObjectLabelSet:
        return ObjectLabelSet(
            name=value.name,
            labels=value.labels_for_variant(variant),
            representation=value.representation,
            declared_object_count=value.declared_object_count,
            declared_object_ids=value.declared_object_ids,
            declared_object_id_domains=value.declared_object_id_domains,
            domain_scope=value.domain_scope,
            plane_axis=value.plane_axis,
            spatial_origin_yx=value.spatial_origin_yx,
            source_spatial_shape_yx=value.source_spatial_shape_yx,
            source_path=value.source_path,
            source_component_metadata=value.source_component_metadata,
            channel_source_paths=value.channel_source_paths,
            channel_source_component_metadata=value.channel_source_component_metadata,
            source_image_names=value.source_image_names,
            dimensions=value.dimensions,
            source_image_name=value.source_image_name,
        )

    def output_value(
        self,
        *,
        labels: ObjectLabelData,
        unedited_labels: ObjectLabelData | None,
        small_removed_labels: ObjectLabelData | None,
    ) -> ObjectLabelSet:
        return ObjectLabelSet(
            name=self.first.name,
            labels=labels,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
            plane_axis=self.plane_axis,
            representation=self.representation,
            dimensions=self.first.dimensions,
            source_image_name=self.first.source_image_name,
            spatial_origin_yx=self.spatial_origin_yx,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
            source_path=self.source_path,
            source_component_metadata=self.source_component_metadata,
            channel_source_paths=self.channel_source_paths,
            channel_source_component_metadata=self.channel_source_component_metadata,
            source_image_names=self.source_image_names,
        )


class RuntimeObjectLabelSliceStackStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal stack strategy for dense runtime object-label slice values."""

    __registry_key__ = "strategy_name"
    strategy_name: ClassVar[str | None] = None

    @classmethod
    @abstractmethod
    def supports(cls, values: tuple[Any, ...]) -> bool:
        """Return whether this strategy owns the supplied slice values."""

    @abstractmethod
    def stack(self, values: tuple[Any, ...], memory_type: str) -> Any:
        """Stack the supplied slice values."""

    @classmethod
    def for_values(
        cls,
        values: tuple[Any, ...],
    ) -> "RuntimeObjectLabelSliceStackStrategy":
        for strategy_type in cls.__registry__.values():
            if strategy_type.supports(values):
                return strategy_type()
        raise ValueError(
            "No object-label runtime slice stack strategy accepts shapes "
            f"{[getattr(value, 'shape', None) for value in values]!r}."
        )


class ImageLayoutObjectLabelSliceStackStrategy(RuntimeObjectLabelSliceStackStrategy):
    """Stack object labels whose slices match a declared image-stack layout."""

    strategy_name = "image_layout"

    @classmethod
    def supports(cls, values: tuple[Any, ...]) -> bool:
        return any(
            all(layout_type.slice_predicate(value) for value in values)
            for layout_type in ImageStackLayout.__registry__.values()
        )

    def stack(self, values: tuple[Any, ...], memory_type: str) -> Any:
        return RuntimeSliceStackRequest(
            slices=values,
            memory_type=memory_type,
        ).stack()


class HomogeneousArrayObjectLabelSliceStackStrategy(
    RuntimeObjectLabelSliceStackStrategy
):
    """Stack same-shaped dense object-label arrays outside image-layout roles."""

    strategy_name = "homogeneous_array"

    @classmethod
    def supports(cls, values: tuple[Any, ...]) -> bool:
        if not values:
            return False
        arrays = tuple(np.asarray(value) for value in values)
        return len({tuple(array.shape) for array in arrays}) == 1

    def stack(self, values: tuple[Any, ...], memory_type: str) -> Any:
        del memory_type
        return np.stack(tuple(np.asarray(value) for value in values), axis=0)


def stack_runtime_object_label_slices(values: Sequence[Any], memory_type: str) -> Any:
    """Stack dense runtime object-label slices using a declared stack strategy."""
    slice_values = tuple(values)
    return RuntimeObjectLabelSliceStackStrategy.for_values(slice_values).stack(
        slice_values,
        memory_type,
    )


@dataclass(frozen=True, slots=True)
class RuntimeSliceStackRequest:
    """Nominal request for stacking runtime image/object-label slices."""

    slices: Sequence[Any]
    memory_type: str
    gpu_id: int = 0

    def stack(self) -> Any:
        slice_values = tuple(self.slices)
        return ImageStackLayout.for_slices(slice_values).stack(
            slices=slice_values,
            memory_type=self.memory_type,
            gpu_id=self.gpu_id,
        )


class NominalSourceStrategyLookup(NominalTypeKeyedStrategyMixin, ABC):
    """Shared lookup authority for strategies keyed by a source value."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)
    fallback_strategy_type: ClassVar[type["NominalSourceStrategyLookup"] | None] = None

    @classmethod
    def for_source(cls, source: object) -> Self:
        strategy = cls.for_nominal_value(source)
        if strategy is not None:
            return strategy
        fallback_type = cls.fallback_strategy_type
        if fallback_type is None:
            raise TypeError(f"{cls.__name__} must declare fallback_strategy_type.")
        return fallback_type()


class ObjectLabelPayloadBuilderStrategy(
    NominalSourceStrategyLookup,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered constructor for preserving object-label metadata across transforms."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @abstractmethod
    def build(
        self,
        source: object,
        labels: object,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelPayload:
        """Return transformed labels wrapped in the source object's semantic domain."""


class ObjectLabelPayloadBuilder(ObjectLabelPayloadBuilderStrategy):
    """Preserve metadata from serialized object-label payloads."""

    value_type = ObjectLabelPayload

    def build(
        self,
        source: object,
        labels: object,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelPayload:
        if not isinstance(source, ObjectLabelPayload):
            raise TypeError(
                "ObjectLabelPayloadBuilder requires ObjectLabelPayload, "
                f"got {type(source).__name__}."
            )
        return ObjectLabelPayload(
            labels=labels,
            declared_object_count=declared_domain.declared_object_count,
            declared_object_ids=declared_domain.declared_object_ids,
            declared_object_id_domains=declared_domain.declared_object_id_domains,
            domain_scope=declared_domain.scope,
            plane_axis=source.plane_axis,
            spatial_origin_yx=source.spatial_origin_yx,
            source_spatial_shape_yx=source.source_spatial_shape_yx,
            source_path=source.source_path,
            source_component_metadata=source.source_component_metadata,
            channel_source_paths=source.channel_source_paths,
            channel_source_component_metadata=source.channel_source_component_metadata,
            source_image_names=source.source_image_names,
        )


class ObjectLabelSetPayloadBuilder(ObjectLabelPayloadBuilderStrategy):
    """Preserve metadata from native object-label runtime values."""

    value_type = ObjectLabelSet

    def build(
        self,
        source: object,
        labels: object,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelPayload:
        if not isinstance(source, ObjectLabelSet):
            raise TypeError(
                "ObjectLabelSetPayloadBuilder requires ObjectLabelSet, "
                f"got {type(source).__name__}."
            )
        return ObjectLabelPayload(
            labels=labels,
            declared_object_count=declared_domain.declared_object_count,
            declared_object_ids=declared_domain.declared_object_ids,
            declared_object_id_domains=declared_domain.declared_object_id_domains,
            domain_scope=declared_domain.scope,
            plane_axis=source.plane_axis,
            spatial_origin_yx=source.spatial_origin_yx,
            source_spatial_shape_yx=source.source_spatial_shape_yx,
            source_path=source.source_path,
            source_component_metadata=source.source_component_metadata,
            channel_source_paths=source.channel_source_paths,
            channel_source_component_metadata=source.channel_source_component_metadata,
            source_image_names=source.source_image_names,
        )


class RawObjectLabelPayloadBuilderStrategy(ObjectLabelPayloadBuilderStrategy):
    """Build a semantic payload for already-dense object labels."""

    def build(
        self,
        source: object,
        labels: object,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelPayload:
        return ObjectLabelPayload(
            labels=labels,
            declared_object_count=declared_domain.declared_object_count,
            declared_object_ids=declared_domain.declared_object_ids,
            declared_object_id_domains=declared_domain.declared_object_id_domains,
            domain_scope=declared_domain.scope,
        )


ObjectLabelPayloadBuilderStrategy.fallback_strategy_type = (
    RawObjectLabelPayloadBuilderStrategy
)


def object_label_payload_with_dense_labels(
    source: object,
    labels: object,
    *,
    domain_declaration: ObjectLabelDomainDeclaration = (
        PreserveSourceObjectLabelDomainDeclaration()
    ),
) -> ObjectLabelPayload:
    """Build a dense-label payload while preserving nominal object-label metadata."""
    declared_domain = domain_declaration.declared_domain(source, labels)
    return ObjectLabelPayloadBuilderStrategy.for_source(source).build(
        source,
        labels,
        declared_domain=declared_domain,
    )


class ObjectLabelValueBuilderStrategy(
    NominalSourceStrategyLookup,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered constructor preserving the source object-label value form."""

    value_type: ClassVar[type[ObjectLabelValue] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @abstractmethod
    def build(
        self,
        source: ObjectLabelValueBuildSource,
        labels: ObjectLabelData,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelValue:
        """Return transformed labels preserving the source value category."""


class ObjectLabelPayloadValueBuilder(ObjectLabelValueBuilderStrategy):
    """Payload sources stay serialized object-label payloads."""

    value_type = ObjectLabelPayload

    def build(
        self,
        source: ObjectLabelValueBuildSource,
        labels: ObjectLabelData,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelPayload:
        return object_label_payload_with_dense_labels(
            source,
            labels,
            domain_declaration=ExplicitObjectLabelDomainDeclaration(declared_domain),
        )


class ObjectLabelSetValueBuilder(ObjectLabelValueBuilderStrategy):
    """Native object-label sets stay native so source context remains authoritative."""

    value_type = ObjectLabelSet

    def build(
        self,
        source: ObjectLabelValueBuildSource,
        labels: ObjectLabelData,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelSet:
        if not isinstance(source, ObjectLabelSet):
            raise TypeError(
                "ObjectLabelSetValueBuilder requires ObjectLabelSet, "
                f"got {type(source).__name__}."
            )
        return ObjectLabelSet(
            name=source.name,
            labels=ObjectLabelSetReplacementStrategy.for_source(
                source
            ).replacement_labels(labels),
            unedited_labels=object_label_variant_matching_labels(
                source.unedited_labels,
                labels,
            ),
            small_removed_labels=object_label_variant_matching_labels(
                source.small_removed_labels,
                labels,
            ),
            representation=source.representation,
            declared_object_count=declared_domain.declared_object_count,
            declared_object_ids=declared_domain.declared_object_ids,
            declared_object_id_domains=declared_domain.declared_object_id_domains,
            domain_scope=declared_domain.scope,
            plane_axis=source.plane_axis,
            spatial_origin_yx=source.spatial_origin_yx,
            source_spatial_shape_yx=source.source_spatial_shape_yx,
            source_path=source.source_path,
            source_component_metadata=source.source_component_metadata,
            channel_source_paths=source.channel_source_paths,
            channel_source_component_metadata=source.channel_source_component_metadata,
            source_image_names=source.source_image_names,
            dimensions=source.dimensions,
            source_image_name=source.source_image_name,
        )


class RawObjectLabelValueBuilderStrategy(ObjectLabelValueBuilderStrategy):
    """Unknown dense-label sources lower to serialized object-label payloads."""

    def build(
        self,
        source: ObjectLabelValueBuildSource,
        labels: ObjectLabelData,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelPayload:
        return object_label_payload_with_dense_labels(
            source,
            labels,
            domain_declaration=ExplicitObjectLabelDomainDeclaration(declared_domain),
        )


ObjectLabelValueBuilderStrategy.fallback_strategy_type = RawObjectLabelValueBuilderStrategy


def object_label_value_with_dense_labels(
    source: ObjectLabelValueBuildSource,
    labels: ObjectLabelData,
    *,
    domain_declaration: ObjectLabelDomainDeclaration = (
        PreserveSourceObjectLabelDomainDeclaration()
    ),
) -> ObjectLabelValue:
    """Build transformed object labels preserving the source value category."""
    declared_domain = domain_declaration.declared_domain(source, labels)
    return ObjectLabelValueBuilderStrategy.for_source(source).build(
        source,
        labels,
        declared_domain=declared_domain,
    )


@dataclass(frozen=True, slots=True)
class SourceImageObjectLabelBuildRequest:
    """Build object-label runtime values from one source-image context."""

    image: object
    labels: object
    declared_object_count: int | None = None
    declared_object_ids: tuple[int, ...] = ()
    unedited_labels: object | None = None
    small_removed_labels: object | None = None

    @property
    def metadata(self) -> ImagePayloadMetadata:
        return image_payload_metadata(self.image)

    @property
    def domain(self) -> ObjectLabelDomain:
        if (
            SourceImageObjectLabelDomainRequest(
                image=self.image,
                labels=self.labels,
            ).plane_semantics()
            is not None
        ):
            return ObjectLabelDomain(
                declared_object_id_domains=dense_object_label_plane_id_domains(
                    self.labels,
                    declared_object_count=self.declared_object_count,
                    declared_object_ids=self.declared_object_ids,
                    domain_scope=ObjectLabelDomainScope.PLANE,
                ),
                scope=ObjectLabelDomainScope.PLANE,
            )
        return ObjectLabelDomain(
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
        )

    @property
    def plane_axis(self) -> RuntimePlaneAxis:
        semantics = SourceImageObjectLabelDomainRequest(
            image=self.image,
            labels=self.labels,
        ).plane_semantics()
        return (
            RuntimePlaneAxis.RUNTIME_SLICE
            if semantics is None
            else semantics.plane_axis
        )

    def payload(self) -> ObjectLabelPayload:
        metadata = self.metadata
        label_domain = self.domain
        return ObjectLabelPayload(
            labels=self.labels,
            unedited_labels=self.unedited_labels,
            small_removed_labels=self.small_removed_labels,
            declared_object_count=label_domain.declared_object_count,
            declared_object_ids=label_domain.declared_object_ids,
            declared_object_id_domains=label_domain.declared_object_id_domains,
            domain_scope=label_domain.scope,
            plane_axis=self.plane_axis,
            spatial_origin_yx=metadata.spatial_origin_yx,
            source_spatial_shape_yx=metadata.source_spatial_shape_yx,
            source_path=metadata.source_path,
            source_component_metadata=metadata.source_component_metadata,
            channel_source_paths=metadata.channel_source_paths,
            channel_source_component_metadata=metadata.channel_source_component_metadata,
            source_image_names=metadata.source_image_names,
        )

    def label_set(
        self,
        *,
        name: str,
        source_image_name: str | None = None,
        representation: ObjectLabelRepresentation = ObjectLabelRepresentation.DENSE_LABELS,
    ) -> ObjectLabelSet:
        metadata = self.metadata
        label_domain = self.domain
        return ObjectLabelSet(
            name=name,
            labels=self.labels,
            unedited_labels=self.unedited_labels,
            small_removed_labels=self.small_removed_labels,
            representation=representation,
            declared_object_count=label_domain.declared_object_count,
            declared_object_ids=label_domain.declared_object_ids,
            declared_object_id_domains=label_domain.declared_object_id_domains,
            domain_scope=label_domain.scope,
            plane_axis=self.plane_axis,
            spatial_origin_yx=metadata.spatial_origin_yx,
            source_spatial_shape_yx=metadata.source_spatial_shape_yx,
            source_path=metadata.source_path,
            source_component_metadata=metadata.source_component_metadata,
            channel_source_paths=metadata.channel_source_paths,
            channel_source_component_metadata=metadata.channel_source_component_metadata,
            source_image_names=metadata.source_image_names,
            source_image_name=source_image_name,
        )


@dataclass(frozen=True, slots=True)
class SourceAlignedObjectLabelProvenanceRequest:
    """Validation request for source-image-aligned object-label runtime storage."""

    image: ImagePayloadMetadataInput
    labels: ObjectLabelValue
    label_name: str

    @property
    def alignment_state(self) -> str:
        return ("not_source_aligned", "source_aligned")[int(self.source_aligned)]

    @property
    def source_aligned(self) -> bool:
        return (
            SourceImageObjectLabelDomainRequest(
                image=self.image,
                labels=self.labels,
            ).plane_semantics()
            is not None
        )

    @property
    def plane_count(self) -> int:
        return int(np.asarray(self.labels.labels).shape[0])

    @property
    def plane_provenance(self) -> tuple[SourceImageProvenance, ...]:
        return tuple(
            self.labels.source_provenance.for_channel(plane_index)
            for plane_index in range(self.plane_count)
        )

    def validate(self) -> None:
        SourceAlignedObjectLabelProvenancePolicy.for_request(self).validate(self)


class SourceAlignedObjectLabelProvenancePolicy(ABC, metaclass=AutoRegisterMeta):
    """Registered policy for source-aligned object-label provenance validation."""

    __registry_key__ = "alignment_state"
    __skip_if_no_key__ = True
    alignment_state: ClassVar[str | None] = None

    @classmethod
    def for_request(
        cls,
        request: SourceAlignedObjectLabelProvenanceRequest,
    ) -> "SourceAlignedObjectLabelProvenancePolicy":
        return cls.__registry__[request.alignment_state]()

    def validate(self, request: SourceAlignedObjectLabelProvenanceRequest) -> None:
        """Validate the request under this policy."""
        del request


class UnalignedObjectLabelProvenancePolicy(SourceAlignedObjectLabelProvenancePolicy):
    """No per-plane provenance requirement exists without source-aligned planes."""

    alignment_state = "not_source_aligned"


class AlignedObjectLabelProvenancePolicy(SourceAlignedObjectLabelProvenancePolicy):
    """Require one addressable source identity per source-aligned label plane."""

    alignment_state = "source_aligned"

    def validate(self, request: SourceAlignedObjectLabelProvenanceRequest) -> None:
        SourceImageProvenancePlaneCountRequirement(
            provenance=request.labels.source_provenance,
            expected_count=request.plane_count,
            label_name=request.label_name,
        ).validate()
        for plane_index, provenance in enumerate(request.plane_provenance):
            SourceImageProvenanceAddressRequirement(
                provenance=provenance,
                label_name=request.label_name,
                plane_index=plane_index,
            ).validate()


@dataclass(frozen=True, slots=True)
class SourceImagePlaneAxisRequest:
    """Classify the runtime meaning of a source-image stack plane axis."""

    image: ImagePayloadMetadataInput

    @property
    def image_array(self) -> np.ndarray:
        return np.asarray(image_payload_data(self.image))

    @property
    def metadata(self) -> ImagePayloadMetadata:
        return image_payload_metadata(self.image)

    @property
    def source_provenance(self) -> SourceImageProvenance:
        """Return the source-image provenance carried by this payload."""
        return self.metadata.source_provenance

    @property
    def channel_plane_count(self) -> int:
        """Return the number of source-provenance planes declared by metadata."""
        return self.source_provenance.channel_plane_count

class SourceImagePlaneAxisPolicy(ABC, metaclass=AutoRegisterMeta):
    """Registered source-image plane-axis classifier."""

    __registry_key__ = "plane_axis_policy_name"
    __skip_if_no_key__ = True
    plane_axis_policy_name: ClassVar[str | None] = None
    priority: ClassVar[int]

    @classmethod
    def for_request(
        cls,
        request: SourceImagePlaneAxisRequest,
    ) -> "SourceImagePlaneAxisPolicy":
        for policy_type in sorted(cls.__registry__.values(), key=attrgetter("priority")):
            policy = policy_type()
            if policy.matches(request):
                return policy
        raise RuntimeError("No source-image plane-axis policy matched the request.")

    @abstractmethod
    def matches(self, request: SourceImagePlaneAxisRequest) -> bool:
        """Return whether this policy owns the request."""

    @abstractmethod
    def axis(self) -> RuntimePlaneAxis | None:
        """Return the OpenHCS runtime plane axis for this policy."""


class SourceBindingPlaneAxisPolicy(SourceImagePlaneAxisPolicy):
    """Policy base for source-image facts that define source-binding planes."""

    def axis(self) -> RuntimePlaneAxis | None:
        return RuntimePlaneAxis.SOURCE_BINDING


class RuntimeSliceSourceImagePlaneAxisPolicy(SourceImagePlaneAxisPolicy):
    """Policy base for source-image facts that select the runtime-slice axis."""

    def axis(self) -> RuntimePlaneAxis | None:
        return RuntimePlaneAxis.RUNTIME_SLICE


class RepeatedSourceNamePlaneAxisPolicy(RuntimeSliceSourceImagePlaneAxisPolicy):
    """Repeated semantic source names identify runtime-slice planes, not bindings."""

    plane_axis_policy_name = "repeated_source_image_name"
    priority = 0

    def matches(self, request: SourceImagePlaneAxisRequest) -> bool:
        names = request.source_provenance.source_image_names
        return (
            request.channel_plane_count > 1
            and len(names) == request.channel_plane_count
            and len(set(names)) == 1
        )


class ChannelSourcePathPlaneAxisPolicy(SourceBindingPlaneAxisPolicy):
    """Channel source paths declare a source-binding stack axis."""

    plane_axis_policy_name = "channel_source_paths"
    priority = 1

    def matches(self, request: SourceImagePlaneAxisRequest) -> bool:
        return len(request.metadata.channel_source_paths) > 1


class ChannelSourceComponentPlaneAxisPolicy(SourceBindingPlaneAxisPolicy):
    """Channel source component metadata declares a source-binding stack axis."""

    plane_axis_policy_name = "channel_source_component_metadata"
    priority = 2

    def matches(self, request: SourceImagePlaneAxisRequest) -> bool:
        return len(request.metadata.channel_source_component_metadata) > 1


class ColorStackPlaneAxisPolicy(SourceBindingPlaneAxisPolicy):
    """Color stacks use their leading axis as source-binding planes."""

    plane_axis_policy_name = "color_stack"
    priority = 3

    def matches(self, request: SourceImagePlaneAxisRequest) -> bool:
        return is_color_image_stack(request.image_array)


class VolumetricSourceImagePlaneAxisPolicy(SourceImagePlaneAxisPolicy):
    """Plain 3D grayscale image stacks do not imply source-bound label planes."""

    plane_axis_policy_name = "volumetric"
    priority = 4

    def matches(self, request: SourceImagePlaneAxisRequest) -> bool:
        return request.image_array.ndim == 3

    def axis(self) -> RuntimePlaneAxis | None:
        return None


class RuntimeSlicePlaneAxisPolicy(RuntimeSliceSourceImagePlaneAxisPolicy):
    """Non-stack source images align object labels to the runtime slice axis."""

    plane_axis_policy_name = "runtime_slice"
    priority = 5

    def matches(self, request: SourceImagePlaneAxisRequest) -> bool:
        del request
        return True


@dataclass(frozen=True, slots=True)
class SourceImageObjectLabelPlaneSemantics:
    """Plane-domain classification for object labels produced from an image."""

    plane_axis: RuntimePlaneAxis

    @property
    def domain_scope(self) -> ObjectLabelDomainScope:
        return ObjectLabelDomainScope.PLANE


@dataclass(frozen=True, slots=True)
class SourceImageObjectLabelDomainRequest:
    """Classify object-label domain scope from source-image plane semantics."""

    image: object
    labels: object

    def plane_semantics(self) -> SourceImageObjectLabelPlaneSemantics | None:
        """Return the semantic plane axis carried by source-aligned labels."""
        image_array = np.asarray(image_payload_data(self.image))
        label_array = np.asarray(self.labels)
        if not self.labels_share_source_image_planes(image_array, label_array):
            return None
        plane_axis = self.source_image_plane_axis(image_array)
        if plane_axis is None:
            return None
        return SourceImageObjectLabelPlaneSemantics(plane_axis)

    def runtime_slice_aligned(self) -> bool:
        """Return whether labels share a runtime-slice axis with the source image."""
        semantics = self.plane_semantics()
        return (
            semantics is not None
            and semantics.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
        )

    def labels_share_source_image_planes(
        self,
        image_array: np.ndarray,
        label_array: np.ndarray,
    ) -> bool:
        """Return whether labels carry one leading plane per source image plane."""
        if label_array.ndim < 3:
            return False
        if image_array.ndim < 3 or is_color_image_slice(image_array):
            return False
        if is_color_image_stack(image_array):
            return (
                int(label_array.shape[0]) == int(image_array.shape[0])
                and tuple(label_array.shape[-2:]) == tuple(image_array.shape[-3:-1])
            )
        return (
            int(label_array.shape[0]) == int(image_array.shape[0])
            and tuple(label_array.shape[-2:]) == tuple(image_array.shape[-2:])
        )

    def source_image_plane_axis(self, image_array: np.ndarray) -> RuntimePlaneAxis | None:
        """Return the OpenHCS runtime axis represented by source-image planes."""
        del image_array
        return SourceImagePlaneAxisPolicy.for_request(
            SourceImagePlaneAxisRequest(self.image)
        ).axis()


class ObjectLabelSetReplacementStrategy(
    EnumKeyedStrategyMixin[ObjectLabelRepresentation],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered replacement policy for ObjectLabelSet label representations."""

    representation: ClassVar[ObjectLabelRepresentation | None] = None
    representation_label: ClassVar[str | None] = None
    __registry_key__ = "representation_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "representation"
    __enum_label_attr__ = "representation_label"

    @classmethod
    def for_source(
        cls,
        source: "ObjectLabelSet",
    ) -> "ObjectLabelSetReplacementStrategy":
        return cls.for_enum_member(source.representation)

    @abstractmethod
    def replacement_labels(self, labels: object) -> object:
        """Return labels compatible with this representation."""


class DenseObjectLabelSetReplacementStrategy(ObjectLabelSetReplacementStrategy):
    """Dense labels are already the concrete replacement payload."""

    representation = ObjectLabelRepresentation.DENSE_LABELS

    def replacement_labels(self, labels: object) -> object:
        return labels


class SparseIJVObjectLabelSetReplacementStrategy(ObjectLabelSetReplacementStrategy):
    """Sparse-IJV replacements use the sparse rows carried by nominal label sets."""

    representation = ObjectLabelRepresentation.SPARSE_IJV

    def replacement_labels(self, labels: object) -> object:
        if isinstance(labels, SparseIJVLabelRows):
            return labels
        if isinstance(labels, ObjectLabelSet):
            if labels.representation is not ObjectLabelRepresentation.SPARSE_IJV:
                return SparseIJVLabelRows.from_dense_stack(
                    ObjectLabelDenseDataStrategy.for_payload(labels).data(labels)
                )
            return labels.labels
        if isinstance(labels, ObjectLabelPayload):
            if isinstance(labels.labels, SparseIJVLabelRows):
                return labels.labels
            return SparseIJVLabelRows.from_dense_stack(
                ObjectLabelDenseDataStrategy.for_payload(labels).data(labels)
            )
        return SparseIJVLabelRows.from_dense_stack(labels)


def object_label_set_with_replacement_labels(
    source: ObjectLabelSet,
    labels: object,
    *,
    unedited_labels: object | None = None,
    small_removed_labels: object | None = None,
) -> ObjectLabelSet:
    """Return an ObjectLabelSet with representation-compatible replacement labels."""
    projected_plane_index = object_label_projected_plane_index(source, labels)
    if projected_plane_index is not None:
        return object_label_set_with_projected_plane(
            source,
            labels,
            projected_plane_index,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
        )
    return source.with_labels(
        labels,
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
    )


def object_label_payload_with_replacement_labels(
    source: ObjectLabelPayload,
    labels: object,
    *,
    unedited_labels: object | None = None,
    small_removed_labels: object | None = None,
) -> ObjectLabelPayload:
    """Return an object-label payload with replacement labels and preserved domain."""
    projected_plane_index = object_label_projected_plane_index(source, labels)
    if projected_plane_index is not None:
        return object_label_payload_with_projected_plane(
            source,
            labels,
            projected_plane_index,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
        )
    return source.with_labels(
        labels,
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
    )


def object_label_payload_with_projected_plane(
    source: ObjectLabelPayload,
    labels: object,
    plane_index: int,
    *,
    unedited_labels: object | None = None,
    small_removed_labels: object | None = None,
) -> ObjectLabelPayload:
    """Return one selected object-label plane with matching domain metadata."""
    domain = object_label_domain_for_projected_label_plane(source, plane_index)
    source_provenance = source.source_provenance.for_channel(plane_index)
    return ObjectLabelPayload(
        labels=labels,
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
        declared_object_count=domain.declared_object_count,
        declared_object_ids=domain.declared_object_ids,
        declared_object_id_domains=domain.declared_object_id_domains,
        domain_scope=domain.scope,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        spatial_origin_yx=source.spatial_origin_yx,
        source_spatial_shape_yx=source.source_spatial_shape_yx,
        source_path=source_provenance.source_path,
        source_component_metadata=source_provenance.source_component_metadata,
        source_image_names=source_provenance.source_image_names,
    )


def object_label_set_with_projected_plane(
    source: ObjectLabelSet,
    labels: object,
    plane_index: int,
    *,
    unedited_labels: object | None = None,
    small_removed_labels: object | None = None,
) -> ObjectLabelSet:
    """Return one selected object-label-set plane with matching domain metadata."""
    domain = object_label_domain_for_projected_label_plane(source, plane_index)
    source_provenance = source.source_provenance.for_channel(plane_index)
    return ObjectLabelSet(
        name=source.name,
        labels=ObjectLabelSetReplacementStrategy.for_source(source).replacement_labels(
            labels
        ),
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
        representation=source.representation,
        declared_object_count=domain.declared_object_count,
        declared_object_ids=domain.declared_object_ids,
        declared_object_id_domains=domain.declared_object_id_domains,
        domain_scope=domain.scope,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        spatial_origin_yx=source.spatial_origin_yx,
        source_spatial_shape_yx=source.source_spatial_shape_yx,
        source_path=source_provenance.source_path,
        source_component_metadata=source_provenance.source_component_metadata,
        source_image_names=source_provenance.source_image_names,
        dimensions=source.dimensions,
        source_image_name=source.source_image_name,
    )


def source_image_names_for_projected_label_plane(
    source: ObjectLabelValue,
    plane_index: int,
) -> tuple[str, ...]:
    """Return provenance aliases carried by a selected object-label plane."""
    return source.source_provenance.source_image_names_for_channel(plane_index)


def object_label_projected_plane_index(
    source: ObjectLabelValue,
    labels: object,
) -> int | None:
    """Return the source plane represented by replacement labels, if unique."""
    source_stack = object_label_dense_array(source)
    replacement = object_label_dense_array(labels)
    if source_stack.ndim < 3 or replacement.ndim != source_stack.ndim - 1:
        return None
    if tuple(replacement.shape) != tuple(source_stack.shape[1:]):
        return None
    matches = tuple(
        plane_index
        for plane_index, plane in enumerate(source_stack)
        if np.array_equal(plane, replacement)
    )
    if len(matches) != 1:
        return None
    return matches[0]


def object_label_domain_for_projected_label_plane(
    source: ObjectLabelValue,
    plane_index: int,
) -> ObjectLabelDomain:
    """Return the payload-domain object IDs carried by one selected plane."""
    plane_domain = source.object_label_domain().project_planes((plane_index,))
    return ObjectLabelDomain.declared(
        scope=ObjectLabelDomainScope.PAYLOAD,
        declared_object_count=plane_domain.declared_object_count,
        declared_object_ids=plane_domain.declared_object_ids,
    )


def object_label_domain_for_derived_label_plane(labels: object) -> ObjectLabelDomain:
    """Return the object-id domain materially produced by one derived label plane."""
    identity_domains = dense_object_label_plane_id_domains(
        labels,
        domain_scope=ObjectLabelDomainScope.PAYLOAD,
    )
    return ObjectLabelDomain.declared(
        scope=ObjectLabelDomainScope.PAYLOAD,
        declared_object_ids=identity_domains[0] if identity_domains else (),
    )


def object_label_payload_with_derived_projected_plane(
    source: ObjectLabelPayload,
    labels: object,
    plane_index: int,
    *,
    unedited_labels: object | None = None,
    small_removed_labels: object | None = None,
) -> ObjectLabelPayload:
    """Return a derived payload plane with source-plane provenance."""
    domain = object_label_domain_for_derived_label_plane(labels)
    return ObjectLabelPayload(
        labels=labels,
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
        declared_object_count=domain.declared_object_count,
        declared_object_ids=domain.declared_object_ids,
        declared_object_id_domains=domain.declared_object_id_domains,
        domain_scope=domain.scope,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        spatial_origin_yx=source.spatial_origin_yx,
        source_spatial_shape_yx=source.source_spatial_shape_yx,
        source_path=source.source_path,
        source_component_metadata=source.source_component_metadata,
        channel_source_paths=source.channel_source_paths,
        channel_source_component_metadata=source.channel_source_component_metadata,
        source_image_names=source_image_names_for_projected_label_plane(
            source,
            plane_index,
        ),
    )


def object_label_value_with_execution_slice(
    source: ObjectLabelValue,
    labels: object,
    plane_index: int,
) -> ObjectLabelValue:
    """Project PURE_2D execution-slice identity without discarding explicit domains."""
    domain = source.object_label_domain()
    if domain.explicit_id_domain() is not None or domain.declared_object_id_domains:
        return ObjectLabelMeasurementPayloadStrategy.for_source(
            source
        ).with_projected_plane(
            source,
            labels,
            plane_index,
        )
    return ObjectLabelMeasurementPayloadStrategy.for_source(
        source
    ).with_derived_projected_plane(
        source,
        labels,
        plane_index,
    )


def object_label_set_with_derived_projected_plane(
    source: ObjectLabelSet,
    labels: object,
    plane_index: int,
    *,
    unedited_labels: object | None = None,
    small_removed_labels: object | None = None,
) -> ObjectLabelSet:
    """Return a derived label-set plane with source-plane provenance."""
    domain = object_label_domain_for_derived_label_plane(labels)
    return ObjectLabelSet(
        name=source.name,
        labels=ObjectLabelSetReplacementStrategy.for_source(source).replacement_labels(
            labels
        ),
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
        representation=source.representation,
        declared_object_count=domain.declared_object_count,
        declared_object_ids=domain.declared_object_ids,
        declared_object_id_domains=domain.declared_object_id_domains,
        domain_scope=domain.scope,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        spatial_origin_yx=source.spatial_origin_yx,
        source_spatial_shape_yx=source.source_spatial_shape_yx,
        source_path=source.source_path,
        source_component_metadata=source.source_component_metadata,
        channel_source_paths=source.channel_source_paths,
        channel_source_component_metadata=source.channel_source_component_metadata,
        source_image_names=source_image_names_for_projected_label_plane(
            source,
            plane_index,
        ),
        dimensions=source.dimensions,
        source_image_name=source.source_image_name,
    )


class ObjectLabelVariantCompatibilityStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered policy for retaining label variants after label replacement."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def for_variant(
        cls,
        variant: object,
    ) -> "ObjectLabelVariantCompatibilityStrategy":
        strategy = cls.for_nominal_value(variant)
        return strategy if strategy is not None else RawObjectLabelVariantCompatibilityStrategy()

    @abstractmethod
    def matching_labels(self, variant: object, labels: object) -> object | None:
        """Return variant when it is compatible with replacement labels."""


class RuntimeArrayObjectLabelVariantCompatibilityStrategy(
    ObjectLabelVariantCompatibilityStrategy
):
    """Runtime-array variants must inhabit the same dense label shape."""

    value_type = RuntimeArrayPayload

    def matching_labels(self, variant: object, labels: object) -> object | None:
        if not isinstance(variant, RuntimeArrayPayload):
            raise TypeError(
                "RuntimeArrayObjectLabelVariantCompatibilityStrategy requires "
                f"RuntimeArrayPayload, got {type(variant).__name__}."
            )
        if not isinstance(labels, RuntimeArrayPayload):
            return variant
        if tuple(variant.shape) == tuple(labels.shape):
            return variant
        return None


class NumpyObjectLabelVariantCompatibilityStrategy(
    ObjectLabelVariantCompatibilityStrategy
):
    """NumPy variants must inhabit the same dense label shape."""

    value_type = np.ndarray

    def matching_labels(self, variant: object, labels: object) -> object | None:
        if not isinstance(variant, np.ndarray):
            raise TypeError(
                "NumpyObjectLabelVariantCompatibilityStrategy requires ndarray, "
                f"got {type(variant).__name__}."
            )
        if not isinstance(labels, np.ndarray):
            return variant
        if tuple(variant.shape) == tuple(labels.shape):
            return variant
        return None


class RawObjectLabelVariantCompatibilityStrategy(
    ObjectLabelVariantCompatibilityStrategy
):
    """Unknown variants are metadata-only and remain attached."""

    def matching_labels(self, variant: object, labels: object) -> object | None:
        return variant


def object_label_variant_matching_labels(
    variant: object | None,
    labels: object,
) -> object | None:
    """Return a variant only when it is compatible with replacement labels."""
    if variant is None:
        return None
    return ObjectLabelVariantCompatibilityStrategy.for_variant(
        variant
    ).matching_labels(variant, labels)


class ObjectLabelMeasurementPayloadStrategy(
    NominalSourceStrategyLookup,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered policy for replacing labels used in measurement contexts."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @abstractmethod
    def with_labels(self, source: object, labels: object) -> object:
        """Return source metadata over labels selected for measurement."""

    @abstractmethod
    def with_projected_plane(
        self,
        source: object,
        labels: object,
        plane_index: int,
    ) -> object:
        """Return one selected label plane with matching domain metadata."""

    @abstractmethod
    def with_derived_projected_plane(
        self,
        source: object,
        labels: object,
        plane_index: int,
    ) -> object:
        """Return one derived label plane with projected provenance metadata."""


class ObjectLabelSetMeasurementPayloadStrategy(ObjectLabelMeasurementPayloadStrategy):
    """Replace labels for native object-label runtime values."""

    value_type = ObjectLabelSet

    def with_labels(self, source: object, labels: object) -> object:
        if not isinstance(source, ObjectLabelSet):
            raise TypeError(
                "ObjectLabelSetMeasurementPayloadStrategy requires ObjectLabelSet, "
                f"got {type(source).__name__}."
            )
        return object_label_set_with_replacement_labels(
            source,
            labels,
            unedited_labels=object_label_variant_matching_labels(
                source.unedited_labels,
                labels,
            ),
            small_removed_labels=object_label_variant_matching_labels(
                source.small_removed_labels,
                labels,
            ),
        )

    def with_projected_plane(
        self,
        source: object,
        labels: object,
        plane_index: int,
    ) -> object:
        if not isinstance(source, ObjectLabelSet):
            raise TypeError(
                "ObjectLabelSetMeasurementPayloadStrategy requires ObjectLabelSet, "
                f"got {type(source).__name__}."
            )
        return object_label_set_with_projected_plane(
            source,
            labels,
            plane_index,
            unedited_labels=object_label_variant_matching_labels(
                source.unedited_labels,
                labels,
            ),
            small_removed_labels=object_label_variant_matching_labels(
                source.small_removed_labels,
                labels,
            ),
        )

    def with_derived_projected_plane(
        self,
        source: object,
        labels: object,
        plane_index: int,
    ) -> object:
        if not isinstance(source, ObjectLabelSet):
            raise TypeError(
                "ObjectLabelSetMeasurementPayloadStrategy requires ObjectLabelSet, "
                f"got {type(source).__name__}."
            )
        return object_label_set_with_derived_projected_plane(
            source,
            labels,
            plane_index,
            unedited_labels=object_label_variant_matching_labels(
                source.unedited_labels,
                labels,
            ),
            small_removed_labels=object_label_variant_matching_labels(
                source.small_removed_labels,
                labels,
            ),
        )


class ObjectLabelPayloadMeasurementPayloadStrategy(
    ObjectLabelMeasurementPayloadStrategy
):
    """Replace labels for serialized object-label payloads."""

    value_type = ObjectLabelPayload

    def with_labels(self, source: object, labels: object) -> object:
        if not isinstance(source, ObjectLabelPayload):
            raise TypeError(
                "ObjectLabelPayloadMeasurementPayloadStrategy requires "
                f"ObjectLabelPayload, got {type(source).__name__}."
            )
        return object_label_payload_with_replacement_labels(
            source,
            labels,
            unedited_labels=object_label_variant_matching_labels(
                source.unedited_labels,
                labels,
            ),
            small_removed_labels=object_label_variant_matching_labels(
                source.small_removed_labels,
                labels,
            ),
        )

    def with_projected_plane(
        self,
        source: object,
        labels: object,
        plane_index: int,
    ) -> object:
        if not isinstance(source, ObjectLabelPayload):
            raise TypeError(
                "ObjectLabelPayloadMeasurementPayloadStrategy requires "
                f"ObjectLabelPayload, got {type(source).__name__}."
            )
        return object_label_payload_with_projected_plane(
            source,
            labels,
            plane_index,
            unedited_labels=object_label_variant_matching_labels(
                source.unedited_labels,
                labels,
            ),
            small_removed_labels=object_label_variant_matching_labels(
                source.small_removed_labels,
                labels,
            ),
        )

    def with_derived_projected_plane(
        self,
        source: object,
        labels: object,
        plane_index: int,
    ) -> object:
        if not isinstance(source, ObjectLabelPayload):
            raise TypeError(
                "ObjectLabelPayloadMeasurementPayloadStrategy requires "
                f"ObjectLabelPayload, got {type(source).__name__}."
            )
        return object_label_payload_with_derived_projected_plane(
            source,
            labels,
            plane_index,
            unedited_labels=object_label_variant_matching_labels(
                source.unedited_labels,
                labels,
            ),
            small_removed_labels=object_label_variant_matching_labels(
                source.small_removed_labels,
                labels,
            ),
        )


class RawObjectLabelMeasurementPayloadStrategy(ObjectLabelMeasurementPayloadStrategy):
    """Dense arrays have no nominal metadata to preserve."""

    def with_labels(self, source: object, labels: object) -> object:
        return labels

    def with_projected_plane(
        self,
        source: object,
        labels: object,
        plane_index: int,
    ) -> object:
        del source, plane_index
        return labels

    def with_derived_projected_plane(
        self,
        source: object,
        labels: object,
        plane_index: int,
    ) -> object:
        del source, plane_index
        return labels


ObjectLabelMeasurementPayloadStrategy.fallback_strategy_type = (
    RawObjectLabelMeasurementPayloadStrategy
)


class SingletonObjectLabelStackCollapseStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered collapse policy for singleton object-label stacks."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def for_labels(cls, labels: object) -> "SingletonObjectLabelStackCollapseStrategy":
        strategy = cls.for_nominal_value(labels)
        return strategy if strategy is not None else RawSingletonObjectLabelStackCollapseStrategy()

    @abstractmethod
    def collapse(self, labels: object) -> object:
        """Collapse singleton stacked labels when the strategy applies."""


class ObjectLabelPayloadStackCollapseStrategy(
    SingletonObjectLabelStackCollapseStrategy
):
    """Collapse singleton serialized object-label payload stacks."""

    value_type = ObjectLabelPayload

    def collapse(self, labels: object) -> object:
        if not isinstance(labels, ObjectLabelPayload):
            raise TypeError(
                "ObjectLabelPayloadStackCollapseStrategy requires ObjectLabelPayload, "
                f"got {type(labels).__name__}."
            )
        if labels.ndim != 3 or labels.shape[0] != 1:
            return labels
        return object_label_payload_with_replacement_labels(
            labels,
            labels.labels[0],
            unedited_labels=(
                None if labels.unedited_labels is None else labels.unedited_labels[0]
            ),
            small_removed_labels=(
                None
                if labels.small_removed_labels is None
                else labels.small_removed_labels[0]
            ),
        )


class RuntimeArrayStackCollapseStrategy(SingletonObjectLabelStackCollapseStrategy):
    """Collapse singleton nominal array payload stacks."""

    value_type = RuntimeArrayPayload

    def collapse(self, labels: object) -> object:
        if not isinstance(labels, RuntimeArrayPayload):
            raise TypeError(
                "RuntimeArrayStackCollapseStrategy requires RuntimeArrayPayload, "
                f"got {type(labels).__name__}."
            )
        if labels.ndim == 3 and labels.shape[0] == 1:
            return labels[0]
        return labels


class NumpyObjectLabelStackCollapseStrategy(SingletonObjectLabelStackCollapseStrategy):
    """Collapse singleton NumPy label stacks."""

    value_type = np.ndarray

    def collapse(self, labels: object) -> object:
        if not isinstance(labels, np.ndarray):
            raise TypeError(
                "NumpyObjectLabelStackCollapseStrategy requires ndarray, "
                f"got {type(labels).__name__}."
            )
        if labels.ndim == 3 and labels.shape[0] == 1:
            return labels[0]
        return labels


class RawSingletonObjectLabelStackCollapseStrategy(
    SingletonObjectLabelStackCollapseStrategy
):
    """Unknown payloads are not singleton object-label stacks."""

    def collapse(self, labels: object) -> object:
        return labels


@dataclass(frozen=True, slots=True)
class DenseObjectLabelSliceStack:
    """Dense object labels projected onto a fixed slice axis."""

    labels: np.ndarray
    payload: object | None = None
    preserves_payload_domain: bool = False

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        slice_count: int,
        dtype: object | None = None,
    ) -> "DenseObjectLabelSliceStack | None":
        return DenseObjectLabelSliceStackRequest(payload, slice_count, dtype).stack()

    def slice(self, slice_index: int) -> object:
        labels = self.labels[slice_index]
        if self.preserves_payload_domain and self.payload is not None:
            return ObjectLabelMeasurementPayloadStrategy.for_source(
                self.payload
            ).with_projected_plane(
                self.payload,
                labels,
                slice_index,
            )
        return labels


@dataclass(frozen=True, slots=True)
class DenseObjectLabelSliceStackRequest:
    """Typed request for projecting dense labels onto a fixed slice axis."""

    payload: object
    slice_count: int
    dtype: object | None = None

    def stack(self) -> DenseObjectLabelSliceStack | None:
        label_array = object_label_dense_array(self.payload, dtype=self.dtype)
        if label_array.ndim == 3 and label_array.shape[0] == self.slice_count:
            return DenseObjectLabelSliceStack(
                np.ascontiguousarray(label_array),
                payload=self.payload,
                preserves_payload_domain=isinstance(self.payload, ObjectLabelValue),
            )
        if label_array.ndim == 2:
            return DenseObjectLabelSliceStack(
                np.ascontiguousarray(
                    np.broadcast_to(
                        label_array,
                        (self.slice_count, *label_array.shape),
                    )
                )
            )
        return None


@dataclass(frozen=True, slots=True)
class DenseObjectLabelPlaneDomainStack:
    """Dense object-label planes that represent distinct object domains."""

    labels: np.ndarray
    payload: object
    object_id_domains: tuple[tuple[int, ...], ...]

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        dtype: object | None = None,
        allow_single_plane: bool = False,
        collapse_repeated: bool = False,
    ) -> "DenseObjectLabelPlaneDomainStack | None":
        return DenseObjectLabelPlaneDomainStackRequest(
            payload,
            dtype,
            allow_single_plane,
            collapse_repeated,
        ).stack()

    @staticmethod
    def _project_measurement_stack(
        payload: ObjectLabelValue,
        labels: np.ndarray,
        *,
        collapse_repeated: bool,
    ) -> tuple[np.ndarray | None, tuple[tuple[int, ...], ...]]:
        if labels.ndim == 3:
            domains = DenseObjectLabelPlaneDomainStack._domains_for_stack(
                payload,
                labels,
            )
            if (
                collapse_repeated
                and domains
                and DenseObjectLabelPlaneDomainStack._planes_are_repeated(
                    labels,
                    domains,
                )
            ):
                return labels[:1], (domains[0],)
            return labels, domains
        if labels.ndim != 4 or labels.shape[0] != labels.shape[1]:
            return None, ()
        diagonal = np.stack(
            tuple(labels[index, index] for index in range(labels.shape[0])),
            axis=0,
        )
        domains = DenseObjectLabelPlaneDomainStack._domains_for_square_diagonal(
            payload,
            labels.shape[0],
            diagonal,
        )
        if collapse_repeated and domains and DenseObjectLabelPlaneDomainStack._planes_are_repeated(
            diagonal,
            domains,
        ):
            return diagonal[:1], (domains[0],)
        return diagonal, domains

    @staticmethod
    def _planes_are_repeated(
        labels: np.ndarray,
        domains: tuple[tuple[int, ...], ...],
    ) -> bool:
        return labels.shape[0] > 1 and all(
            np.array_equal(labels[0], plane) for plane in labels[1:]
        ) and all(domain == domains[0] for domain in domains[1:])

    @staticmethod
    def _domains_for_stack(
        payload: ObjectLabelValue,
        labels: np.ndarray,
    ) -> tuple[tuple[int, ...], ...]:
        domains = payload.declared_object_id_domains
        if domains:
            if len(domains) != labels.shape[0]:
                raise ValueError(
                    f"{type(payload).__name__} declares {len(domains)} plane domains "
                    f"but measurement labels carry {labels.shape[0]} planes."
                )
            return domains
        return dense_object_label_plane_id_domains(
            labels,
            declared_object_count=payload.declared_object_count,
            declared_object_ids=payload.declared_object_ids,
            domain_scope=payload.domain_scope,
        )

    @staticmethod
    def _domains_for_square_diagonal(
        payload: ObjectLabelValue,
        diagonal_count: int,
        diagonal_labels: np.ndarray,
    ) -> tuple[tuple[int, ...], ...]:
        domains = payload.declared_object_id_domains
        if not domains:
            return DenseObjectLabelPlaneDomainStack._domains_for_stack(
                payload,
                diagonal_labels,
            )
        if len(domains) == diagonal_count:
            return domains
        square_domain_count = diagonal_count * diagonal_count
        if len(domains) != square_domain_count:
            raise ValueError(
                f"{type(payload).__name__} declares {len(domains)} square plane "
                f"domains but diagonal measurement labels require {square_domain_count}."
            )
        return tuple(
            domains[index * diagonal_count + index]
            for index in range(diagonal_count)
        )

    @property
    def plane_count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def measurement_row_identity(self) -> MeasurementObjectRowIdentity | None:
        if self.plane_count > 1 and len(self.object_id_domains) > 1:
            return MeasurementObjectRowIdentity.ROW_SEQUENCE
        return None

    def plane(self, plane_index: int) -> object:
        if plane_index < 0 or plane_index >= self.plane_count:
            raise IndexError(plane_index)
        projected = ObjectLabelMeasurementPayloadStrategy.for_source(
            self.payload
        ).with_projected_plane(
            self.payload,
            self.labels[plane_index],
            plane_index,
        )
        return ObjectLabelValueBuilderStrategy.for_source(projected).build(
            projected,
            self.labels[plane_index],
            declared_domain=ObjectLabelDomain.declared(
                scope=ObjectLabelDomainScope.PAYLOAD,
                declared_object_ids=self.object_id_domains[plane_index],
            ),
        )


@dataclass(frozen=True, slots=True)
class DenseObjectLabelPlaneDomainStackRequest:
    """Typed request for object-label planes with distinct measurement domains."""

    payload: object
    dtype: object | None = None
    allow_single_plane: bool = False
    collapse_repeated: bool = False

    def stack(self) -> DenseObjectLabelPlaneDomainStack | None:
        if not isinstance(self.payload, ObjectLabelValue):
            return None
        if self.payload.domain_scope is not ObjectLabelDomainScope.PLANE:
            return None
        labels = object_label_dense_array(self.payload, dtype=self.dtype)
        if not isinstance(labels, np.ndarray):
            return None
        stack_labels, object_id_domains = (
            DenseObjectLabelPlaneDomainStack._project_measurement_stack(
                self.payload,
                labels,
                collapse_repeated=self.collapse_repeated,
            )
        )
        if stack_labels is None or (
            len(object_id_domains) <= 1 and not self.allow_single_plane
        ):
            return None
        return DenseObjectLabelPlaneDomainStack(
            labels=np.ascontiguousarray(stack_labels),
            payload=self.payload,
            object_id_domains=object_id_domains,
        )


class RuntimeSliceAlignedPayloadNormalizationStrategy(
    SharedStrategyLabelBase,
    Generic[SliceAlignedValueT, SliceAlignedPayloadT],
    EnumKeyedStrategyMixin[ArtifactKind],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Normalize nominal slice-aligned payloads before runtime storage."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "kind"

    kind: ClassVar[ArtifactKind]

    @classmethod
    def for_output_plan(
        cls,
        output_plan: ArtifactOutputPlan,
    ) -> "RuntimeSliceAlignedPayloadNormalizationStrategy | None":
        strategy_type = cls.__registry__.get(output_plan.kind.value)
        if strategy_type is None:
            return None
        return strategy_type()

    @abstractmethod
    def normalize(
        self,
        value: RuntimeSliceAlignedValueSet[SliceAlignedValueT],
    ) -> SliceAlignedPayloadT | None:
        """Return an aggregate payload, or ``None`` when slices are not owned."""


class ObjectLabelSliceAlignedPayloadNormalizationStrategy(
    RuntimeSliceAlignedPayloadNormalizationStrategy[ObjectLabelValue, ObjectLabelValue]
):
    """Aggregate object-label slice payloads into one plane-scoped label domain."""

    kind = ArtifactKind.OBJECT_LABELS

    def normalize(
        self,
        value: RuntimeSliceAlignedValueSet[ObjectLabelValue],
    ) -> ObjectLabelValue | None:
        from openhcs.core.memory import detect_memory_type

        slices = tuple(value.value_for_slice(index) for index in range(value.slice_count))
        if not slices or not all(
            isinstance(item, (ObjectLabelPayload, ObjectLabelSet))
            for item in slices
        ):
            return None
        label_values = tuple(
            item
            for item in slices
            if isinstance(item, (ObjectLabelPayload, ObjectLabelSet))
        )
        if len(label_values) != len(slices):
            return None
        return ObjectLabelPure2DSliceAggregator.aggregate(
            label_values,
            detect_memory_type(label_values[0].labels),
            force_plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )


@dataclass(slots=True, kw_only=True)
class MeasurementTable(NativeRuntimeValue):
    """Native OpenHCS measurement table value."""

    rows: Any
    object_name: str | None = None
    fields: tuple[FieldSpec, ...] = ()
    object_id_field: str | None = None
    source_image_name: str | None = None
    subject: MeasurementSubject | None = None
    validated_runtime_schema: bool = False
    schema_loss_reasons: frozenset[str] = frozenset()

    @classmethod
    def from_runtime_value(cls, value: RuntimeValue) -> Self:
        """Reconstruct the native measurement view from a stored runtime value."""
        if value.kind is not ArtifactKind.MEASUREMENTS:
            raise TypeError(
                "MeasurementTable.from_runtime_value requires a MEASUREMENTS "
                f"runtime value, got {value.kind.value}."
            )
        return cls(
            name=value.name,
            rows=value.data,
            object_name=value.schema.object_name,
            fields=value.schema.fields,
            object_id_field=value.schema.object_id_field,
            source_image_name=value.schema.source_image_name,
            subject=value.schema.measurement_subject,
            validated_runtime_schema=value.schema.measurement_schema_validated,
            schema_loss_reasons=value.schema.measurement_schema_loss_reasons,
        )

    def __post_init__(self) -> None:
        NativeRuntimeValue.__post_init__(self)
        if self.object_name == "":
            raise ValueError("MeasurementTable.object_name cannot be empty.")
        if self.object_id_field == "":
            raise ValueError("MeasurementTable.object_id_field cannot be empty.")
        if self.source_image_name == "":
            raise ValueError("MeasurementTable.source_image_name cannot be empty.")
        subject = _resolve_measurement_subject(
            self.subject,
            artifact_name=self.name,
            object_name=self.object_name,
            object_id_field=self.object_id_field,
            source_image_name=self.source_image_name,
        )
        self.subject = subject
        self.schema_loss_reasons = frozenset(str(reason) for reason in self.schema_loss_reasons)
        if self.validated_runtime_schema and self.schema_loss_reasons:
            raise ValueError(
                "MeasurementTable.validated_runtime_schema cannot be true when "
                "schema_loss_reasons is non-empty."
            )
        if not _is_table_like(self.rows):
            raise TypeError(
                f"MeasurementTable '{self.name}' requires table-like rows, "
                f"got {type(self.rows).__name__}."
            )
        declared_layout = measurement_table_row_layout_from_fields(self.fields)
        normalized_rows = (
            self.rows
            if declared_layout is not None
            else normalize_measurement_table_rows(self.rows, fields=self.fields)
        )
        if normalized_rows is not self.rows:
            self.rows = normalized_rows
            self.fields = ()
            declared_layout = None
        if declared_layout is None:
            measurement_table_row_layout(self.rows)

    def runtime_payload(self) -> Any:
        return self.rows

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        subject_resolver = MeasurementTableSubjectResolver(self)
        return RuntimeValueSchema(
            kind=ArtifactKind.MEASUREMENTS,
            fields=self.fields or RuntimePayloadFieldInference(payload).fields(),
            measurement_subject=self.subject,
            object_name=subject_resolver.object_name,
            source_image_name=subject_resolver.source_image_name,
            object_id_field=subject_resolver.object_id_field,
            measurement_schema_validated=self.validated_runtime_schema,
            measurement_schema_loss_reasons=self.schema_loss_reasons,
        )


@dataclass(frozen=True, slots=True)
class SpatialGridAxis:
    """One physical axis of a rectangular spatial grid."""

    spacing: float
    origin: float
    locations: tuple[float, ...] | None = None

    def normalized(self, count: int, field_name: str) -> "SpatialGridAxis":
        """Return this axis with explicit center locations for every index."""
        spacing = float(self.spacing)
        origin = float(self.origin)
        if spacing <= 0:
            raise ValueError(f"{field_name}.spacing must be positive.")
        if self.locations is None:
            locations = tuple(origin + index * spacing for index in range(count))
        elif len(self.locations) != count:
            raise ValueError(f"{field_name}.locations must match axis length.")
        else:
            locations = tuple(float(value) for value in self.locations)
        return type(self)(spacing=spacing, origin=origin, locations=locations)


@dataclass(slots=True)
class SpatialGridDimensions:
    """Row and column cardinality for a rectangular spatial grid."""

    rows: int
    columns: int

    def __post_init__(self) -> None:
        self.rows = int(self.rows)
        self.columns = int(self.columns)
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("SpatialGrid dimensions must be positive.")


@dataclass(slots=True)
class SpatialGridCoordinateSystem:
    """Origin and object-number ordering for a spatial grid."""

    origin: SpatialGridOrigin
    ordering: SpatialGridOrdering

    def __post_init__(self) -> None:
        self.origin = coerce_enum(SpatialGridOrigin, self.origin, "SpatialGrid.origin")
        self.ordering = coerce_enum(SpatialGridOrdering, self.ordering, "SpatialGrid.ordering")


@dataclass(slots=True, init=False)
class SpatialGridTopology:
    """Nominal object-number topology and centers for a spatial grid."""

    dimensions: SpatialGridDimensions
    coordinate_system: SpatialGridCoordinateSystem
    column_axis: SpatialGridAxis
    row_axis: SpatialGridAxis
    spot_table: tuple[tuple[int, ...], ...] | None = None

    def __init__(
        self,
        *,
        dimensions: SpatialGridDimensions,
        coordinate_system: SpatialGridCoordinateSystem,
        column_axis: SpatialGridAxis,
        row_axis: SpatialGridAxis,
        spot_table: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.coordinate_system = coordinate_system
        self.column_axis = column_axis.normalized(
                count=self.dimensions.columns,
                field_name="SpatialGridTopology.column_axis",
            )
        self.row_axis = row_axis.normalized(
                count=self.dimensions.rows,
                field_name="SpatialGridTopology.row_axis",
            )
        self.spot_table = spot_table
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.spot_table is None:
            self.spot_table = self.derived_spot_table()
        elif len(self.spot_table) != self.rows or any(
            len(row) != self.columns for row in self.spot_table
        ):
            raise ValueError("SpatialGridTopology.spot_table must match rows x columns.")
        else:
            self.spot_table = tuple(tuple(int(value) for value in row) for row in self.spot_table)

    @property
    def x_spacing(self) -> float:
        return self.column_axis.spacing

    @property
    def y_spacing(self) -> float:
        return self.row_axis.spacing

    @property
    def x_origin(self) -> float:
        return self.column_axis.origin

    @property
    def y_origin(self) -> float:
        return self.row_axis.origin

    @property
    def x_locations(self) -> tuple[float, ...]:
        if self.column_axis.locations is None:
            raise ValueError("SpatialGridTopology.column_axis is not normalized.")
        return self.column_axis.locations

    @property
    def y_locations(self) -> tuple[float, ...]:
        if self.row_axis.locations is None:
            raise ValueError("SpatialGridTopology.row_axis is not normalized.")
        return self.row_axis.locations

    @property
    def rows(self) -> int:
        return self.dimensions.rows

    @property
    def columns(self) -> int:
        return self.dimensions.columns

    @property
    def origin(self) -> SpatialGridOrigin:
        return self.coordinate_system.origin

    @property
    def ordering(self) -> SpatialGridOrdering:
        return self.coordinate_system.ordering

    def derived_spot_table(self) -> tuple[tuple[int, ...], ...]:
        """Return the CellProfiler-compatible object-number topology."""
        object_ids = np.arange(1, self.rows * self.columns + 1, dtype=np.int32)
        if self.ordering is SpatialGridOrdering.BY_COLUMNS:
            table = object_ids.reshape(self.rows, self.columns)
        else:
            table = object_ids.reshape(self.columns, self.rows).T
        if self.origin.reverses_rows:
            table = table[::-1, :]
        if self.origin.reverses_columns:
            table = table[:, ::-1]
        return tuple(tuple(int(value) for value in row) for row in table)


@dataclass(slots=True, kw_only=True, init=False)
class SpatialGrid(NativeRuntimeValue):
    """Native OpenHCS rectangular spatial grid definition."""

    rows: int
    columns: int
    column_axis: SpatialGridAxis
    row_axis: SpatialGridAxis
    slice_index: int
    total_width: float | None
    total_height: float | None
    origin: SpatialGridOrigin
    ordering: SpatialGridOrdering
    spot_table: tuple[tuple[int, ...], ...] | None
    source_spatial_shape_yx: tuple[int, int] | None
    x_location_of_lowest_x_spot = AliasProperty[float]("x_origin")
    y_location_of_lowest_y_spot = AliasProperty[float]("y_origin")

    def __init__(
        self,
        *,
        name: str,
        rows: int,
        columns: int,
        x_spacing: float | None = None,
        y_spacing: float | None = None,
        x_origin: float | None = None,
        y_origin: float | None = None,
        slice_index: int = 0,
        total_width: float | None = None,
        total_height: float | None = None,
        origin: SpatialGridOrigin = SpatialGridOrigin.TOP_LEFT,
        ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS,
        x_locations: tuple[float, ...] | None = None,
        y_locations: tuple[float, ...] | None = None,
        column_axis: SpatialGridAxis | None = None,
        row_axis: SpatialGridAxis | None = None,
        spot_table: tuple[tuple[int, ...], ...] | None = None,
        source_spatial_shape_yx: tuple[int, int] | None = None,
    ) -> None:
        self.name = name
        self.rows = int(rows)
        self.columns = int(columns)
        self.slice_index = int(slice_index)
        self.total_width = total_width
        self.total_height = total_height
        self.origin = origin
        self.ordering = ordering
        topology = SpatialGridTopology(
            dimensions=SpatialGridDimensions(self.rows, self.columns),
            coordinate_system=SpatialGridCoordinateSystem(origin, ordering),
            column_axis=(
                column_axis
                if column_axis is not None
                else SpatialGridAxis(
                    spacing=_required_constructor_float(
                        x_spacing,
                        "SpatialGrid.x_spacing",
                    ),
                    origin=_required_constructor_float(
                        x_origin,
                        "SpatialGrid.x_origin",
                    ),
                    locations=x_locations,
                )
            ),
            row_axis=(
                row_axis
                if row_axis is not None
                else SpatialGridAxis(
                    spacing=_required_constructor_float(
                        y_spacing,
                        "SpatialGrid.y_spacing",
                    ),
                    origin=_required_constructor_float(
                        y_origin,
                        "SpatialGrid.y_origin",
                    ),
                    locations=y_locations,
                )
            ),
            spot_table=spot_table,
        )
        self.column_axis = topology.column_axis
        self.row_axis = topology.row_axis
        self.spot_table = topology.spot_table
        self.source_spatial_shape_yx = source_spatial_shape_yx
        self.__post_init__()

    @classmethod
    def from_runtime_value(cls, value: RuntimeValue) -> Self:
        """Reconstruct a spatial grid from a stored runtime value."""
        if value.kind is not ArtifactKind.SPATIAL_GRID:
            raise TypeError(
                "SpatialGrid.from_runtime_value requires a SPATIAL_GRID "
                f"runtime value, got {value.kind.value}."
            )
        if not isinstance(value.data, Mapping):
            raise TypeError(
                f"Spatial grid '{value.name}' payload must be mapping-backed, "
                f"got {type(value.data).__name__}."
            )
        return cls.from_mapping(value.name, value.data)

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> Self:
        """Build a spatial grid from canonical or legacy grid field names."""
        rows = _required_int(data, "rows")
        columns = _required_int(data, "columns")
        x_spacing = _required_float(data, "x_spacing")
        y_spacing = _required_float(data, "y_spacing")
        x_origin = _required_float(
            data,
            "x_origin",
            aliases=("x_location_of_lowest_x_spot",),
        )
        y_origin = _required_float(
            data,
            "y_origin",
            aliases=("y_location_of_lowest_y_spot",),
        )
        origin = (
            SpatialGridOrigin.TOP_LEFT
            if data.get("origin") is None
            else coerce_enum(SpatialGridOrigin, data["origin"], "SpatialGrid.origin")
        )
        ordering = OptionalMappingField(data, "ordering").grid_ordering()
        x_locations_value = data.get("x_locations")
        y_locations_value = data.get("y_locations")
        spot_table_value = data.get("spot_table")
        topology = SpatialGridTopology(
            dimensions=SpatialGridDimensions(rows, columns),
            coordinate_system=SpatialGridCoordinateSystem(origin, ordering),
            column_axis=SpatialGridAxis(
                spacing=x_spacing,
                origin=x_origin,
                locations=(
                    None
                    if x_locations_value is None
                    else tuple(float(item) for item in x_locations_value)
                ),
            ),
            row_axis=SpatialGridAxis(
                spacing=y_spacing,
                origin=y_origin,
                locations=(
                    None
                    if y_locations_value is None
                    else tuple(float(item) for item in y_locations_value)
                ),
            ),
            spot_table=(
                None
                if spot_table_value is None
                else tuple(
                    tuple(int(item) for item in row)
                    for row in spot_table_value
                )
            ),
        )
        return cls(
            name=name,
            rows=rows,
            columns=columns,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            x_origin=x_origin,
            y_origin=y_origin,
            slice_index=OptionalMappingField(data, "slice_index").int_or(default=0),
            total_width=OptionalMappingField(data, "total_width").nullable_float(),
            total_height=OptionalMappingField(data, "total_height").nullable_float(),
            origin=origin,
            ordering=ordering,
            x_locations=topology.x_locations,
            y_locations=topology.y_locations,
            spot_table=topology.spot_table,
            source_spatial_shape_yx=(
                None
                if (
                    shape := SpatialShapeYX.optional_from_mapping(
                        data,
                        "source_spatial_shape_yx",
                    )
                )
                is None
                else shape.as_tuple()
            ),
        )

    def __post_init__(self) -> None:
        NativeRuntimeValue.__post_init__(self)
        self.ordering = coerce_enum(SpatialGridOrdering, self.ordering, "SpatialGrid.ordering")
        self.origin = coerce_enum(SpatialGridOrigin, self.origin, "SpatialGrid.origin")
        if self.rows <= 0:
            raise ValueError("SpatialGrid.rows must be positive.")
        if self.columns <= 0:
            raise ValueError("SpatialGrid.columns must be positive.")
        if self.total_width is None:
            self.total_width = self.x_spacing * self.columns
        if self.total_height is None:
            self.total_height = self.y_spacing * self.rows
        if self.source_spatial_shape_yx is not None:
            self.source_spatial_shape_yx = SpatialShapeYX.from_sequence(
                    self.source_spatial_shape_yx,
                    field_name="SpatialGrid.source_spatial_shape_yx",
                ).as_tuple()

    @property
    def x_spacing(self) -> float:
        return self.column_axis.spacing

    @property
    def y_spacing(self) -> float:
        return self.row_axis.spacing

    @property
    def x_origin(self) -> float:
        return self.column_axis.origin

    @property
    def y_origin(self) -> float:
        return self.row_axis.origin

    @property
    def x_locations(self) -> tuple[float, ...]:
        if self.column_axis.locations is None:
            raise ValueError("SpatialGrid.column_axis is not normalized.")
        return self.column_axis.locations

    @property
    def y_locations(self) -> tuple[float, ...]:
        if self.row_axis.locations is None:
            raise ValueError("SpatialGrid.row_axis is not normalized.")
        return self.row_axis.locations

    def with_name(self, name: str) -> Self:
        """Return the same grid under a different artifact name."""
        return type(self)(
            name=name,
            rows=self.rows,
            columns=self.columns,
            x_spacing=self.x_spacing,
            y_spacing=self.y_spacing,
            x_origin=self.x_origin,
            y_origin=self.y_origin,
            slice_index=self.slice_index,
            total_width=self.total_width,
            total_height=self.total_height,
            origin=self.origin,
            ordering=self.ordering,
            x_locations=self.x_locations,
            y_locations=self.y_locations,
            spot_table=self.spot_table,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
        )

    def as_mapping(self) -> dict[str, Any]:
        """Return a JSON/metadata-compatible grid payload."""
        return {
            "slice_index": self.slice_index,
            "rows": self.rows,
            "columns": self.columns,
            "x_spacing": self.x_spacing,
            "y_spacing": self.y_spacing,
            "x_origin": self.x_origin,
            "y_origin": self.y_origin,
            "x_location_of_lowest_x_spot": self.x_origin,
            "y_location_of_lowest_y_spot": self.y_origin,
            "total_width": self.total_width,
            "total_height": self.total_height,
            "origin": self.origin.value,
            "ordering": self.ordering.value,
            "x_locations": self.x_locations,
            "y_locations": self.y_locations,
            "spot_table": self.spot_table,
            "source_spatial_shape_yx": self.source_spatial_shape_yx,
        }

    def runtime_payload(self) -> Any:
        return self.as_mapping()

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        return RuntimeValueSchema(kind=ArtifactKind.SPATIAL_GRID)

    def spot_table_array(self) -> np.ndarray:
        """Return the nominal grid object-number topology as a dense table."""
        return np.asarray(self.spot_table, dtype=np.int32)

    def x_locations_array(self) -> np.ndarray:
        """Return x center coordinates for grid columns."""
        return np.asarray(self.x_locations, dtype=np.float64)

    def y_locations_array(self) -> np.ndarray:
        """Return y center coordinates for grid rows."""
        return np.asarray(self.y_locations, dtype=np.float64)


@dataclass(slots=True, kw_only=True)
class ObjectRelationship(NativeRuntimeValue):
    """Native OpenHCS directed object relationship value."""

    source: RelationshipEndpoint
    target: RelationshipEndpoint
    source_ids: Any
    target_ids: Any
    relationship_type: str = "related"
    slice_indices: tuple[int, ...] = ()
    slice_count: int | None = None
    source_path: str | None = None
    source_component_metadata: SourceComponentMetadata | None = None
    channel_source_paths: tuple[str | None, ...] = ()
    channel_source_component_metadata: ChannelSourceComponentMetadata = ()

    @classmethod
    def from_runtime_value(cls, value: RuntimeValue) -> Self:
        """Reconstruct the native relationship view from a runtime value."""
        if value.kind is not ArtifactKind.RELATIONSHIPS:
            raise TypeError(
                "ObjectRelationship.from_runtime_value requires a RELATIONSHIPS "
                f"runtime value, got {value.kind.value}."
            )
        if not isinstance(value.data, Mapping):
            raise TypeError(
                f"Relationship '{value.name}' payload must be mapping-backed, "
                f"got {type(value.data).__name__}."
            )
        relationship = value.schema.relationship
        if relationship is None:
            raise TypeError(
                f"Relationship '{value.name}' is missing typed relationship "
                "schema."
            )
        return cls(
            name=value.name,
            source=relationship.source,
            target=relationship.target,
            source_ids=value.data[relationship.source.id_field],
            target_ids=value.data[relationship.target.id_field],
            relationship_type=relationship.relationship_type,
            slice_indices=OptionalMappingField(
                value.data,
                "slice_indices",
                aliases=("slice_index",),
            ).int_tuple(),
            slice_count=OptionalMappingField(value.data, "slice_count").nullable_int(),
            source_path=OptionalMappingField(value.data, "source_path").nullable_str(),
            source_component_metadata=OptionalMappingField(
                value.data,
                "source_component_metadata",
            ).nullable_mapping(),
            channel_source_paths=OptionalMappingField(
                value.data,
                "channel_source_paths",
            ).nullable_str_tuple(),
            channel_source_component_metadata=OptionalMappingField(
                value.data,
                "channel_source_component_metadata",
            ).nullable_mapping_tuple(),
        )

    def __post_init__(self) -> None:
        NativeRuntimeValue.__post_init__(self)
        if not isinstance(self.source, RelationshipEndpoint):
            raise TypeError(
                "ObjectRelationship.source must be RelationshipEndpoint, "
                f"got {type(self.source).__name__}."
            )
        if not isinstance(self.target, RelationshipEndpoint):
            raise TypeError(
                "ObjectRelationship.target must be RelationshipEndpoint, "
                f"got {type(self.target).__name__}."
            )
        _require_name(self.relationship_type, "ObjectRelationship.relationship_type")
        _validate_relationship_ids(self.source_ids, self.target_ids, self.name)
        slice_indices = tuple(int(slice_index) for slice_index in self.slice_indices)
        if slice_indices and len(slice_indices) != len(self.source_ids):
            raise ValueError(
                f"ObjectRelationship '{self.name}' slice_indices must be empty "
                "or match source_ids/target_ids length, got "
                f"{len(slice_indices)} for {len(self.source_ids)} relationships."
            )
        if any(slice_index < 0 for slice_index in slice_indices):
            raise ValueError(
                f"ObjectRelationship '{self.name}' slice_indices cannot be negative."
            )
        slice_count = None if self.slice_count is None else int(self.slice_count)
        if slice_count is not None and slice_count < 0:
            raise ValueError(
                f"ObjectRelationship '{self.name}' slice_count cannot be negative."
            )
        if (
            slice_count is not None
            and slice_indices
            and max(slice_indices) >= slice_count
        ):
            raise ValueError(
                f"ObjectRelationship '{self.name}' slice_indices must be smaller "
                f"than slice_count {slice_count}."
            )
        self.slice_indices = slice_indices
        self.slice_count = slice_count
        self.source_component_metadata = _normalize_component_metadata(self.source_component_metadata)
        self.channel_source_component_metadata = tuple(
                _normalize_component_metadata(metadata)
                for metadata in self.channel_source_component_metadata
            )

    @property
    def semantics(self) -> RelationshipSemantics:
        return RelationshipSemantics(
            source=self.source,
            target=self.target,
            relationship_type=self.relationship_type,
        )

    def as_table(self) -> dict[str, Any]:
        """Return table-like relationship columns for materialization."""
        table = {
            "relationship_type": self.relationship_type,
            "source_role": self.source.role,
            "target_role": self.target.role,
            "source_object": self.source.name,
            "target_object": self.target.name,
            self.source.id_field: self.source_ids,
            self.target.id_field: self.target_ids,
        }
        if self.slice_indices:
            table["slice_index"] = self.slice_indices
        if self.slice_count is not None:
            table["slice_count"] = self.slice_count
        if self.source_path is not None:
            table["source_path"] = self.source_path
        if self.source_component_metadata is not None:
            table["source_component_metadata"] = self.source_component_metadata
        if self.channel_source_paths:
            table["channel_source_paths"] = self.channel_source_paths
        if self.channel_source_component_metadata:
            table["channel_source_component_metadata"] = (
                self.channel_source_component_metadata
            )
        return table

    def runtime_payload(self) -> Any:
        return self.as_table()

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        return RuntimeValueSchema(
            kind=ArtifactKind.RELATIONSHIPS,
            fields=RuntimePayloadFieldInference(payload).fields(),
            relationship=self.semantics,
        )


def normalize_artifact_value(
    output_plan: ArtifactOutputPlan,
    value: Any,
    *,
    axis_id: str,
) -> RuntimeValue:
    """Normalize a raw function artifact return into a validated RuntimeValue."""
    if isinstance(value, RuntimeValue):
        return validate_runtime_value(value, output_plan, axis_id=axis_id)

    native_value = _normalize_native_value(output_plan, value, axis_id=axis_id)
    if native_value is not None:
        return validate_runtime_value(native_value, output_plan, axis_id=axis_id)

    runtime_value = RuntimeValue.from_output_plan(
        output_plan,
        value,
        axis_id=axis_id,
        schema=RuntimeValueSchema(kind=output_plan.kind),
    )
    return validate_runtime_value(runtime_value, output_plan, axis_id=axis_id)


def _normalize_native_value(
    output_plan: ArtifactOutputPlan,
    value: Any,
    *,
    axis_id: str,
) -> RuntimeValue | None:
    if _is_runtime_slice_aligned_values(value):
        return _normalize_slice_aligned_value(output_plan, value, axis_id=axis_id)
    if isinstance(value, NativeRuntimeValue):
        _validate_native_name(output_plan, value.name)
        return value.to_runtime_value(output_plan, axis_id=axis_id)
    return None


def _normalize_slice_aligned_value(
    output_plan: ArtifactOutputPlan,
    value: Any,
    *,
    axis_id: str,
) -> RuntimeValue:
    payload_strategy = RuntimeSliceAlignedPayloadNormalizationStrategy.for_output_plan(
        output_plan
    )
    if payload_strategy is not None:
        payload = payload_strategy.normalize(value)
        if payload is not None:
            return RuntimeValue.from_output_plan(
                output_plan,
                payload,
                axis_id=axis_id,
                schema=RuntimeValueSchema(kind=output_plan.kind),
            )

    slice_values: list[Any] = []
    slice_schemas: list[RuntimeValueSchema] = []
    for item in value.slices:
        if isinstance(item, NativeRuntimeValue):
            _validate_native_name(output_plan, item.name)
            runtime_item = item.to_runtime_value(output_plan, axis_id=axis_id)
            slice_values.append(runtime_item.data)
            slice_schemas.append(runtime_item.schema)
            continue
        if isinstance(item, RuntimeValue):
            slice_values.append(
                validate_runtime_value(item, output_plan, axis_id=axis_id).data
            )
            slice_schemas.append(item.schema)
            continue
        slice_values.append(item)

    schema = (
        _merge_slice_aligned_schemas(output_plan, slice_schemas)
        if slice_schemas
        else RuntimeValueSchema(kind=output_plan.kind, slice_aligned=True)
    )
    return RuntimeValue.from_output_plan(
        output_plan,
        tuple(slice_values),
        axis_id=axis_id,
        schema=schema,
    )


def _is_runtime_slice_aligned_values(value: Any) -> bool:
    return isinstance(value, RuntimeSliceAlignedValueSet)


def _merge_slice_aligned_schemas(
    output_plan: ArtifactOutputPlan,
    schemas: Sequence[RuntimeValueSchema],
) -> RuntimeValueSchema:
    first = schemas[0]
    for schema in schemas:
        if schema.kind is not output_plan.kind:
            raise ValueError(
                f"Slice-aligned artifact '{output_plan.name}' expected "
                f"{output_plan.kind.value}, got {schema.kind.value}."
            )
        if replace(schema, slice_aligned=False) != replace(first, slice_aligned=False):
            raise ValueError(
                f"Slice-aligned artifact '{output_plan.name}' has inconsistent "
                "per-slice runtime schemas."
            )
    return replace(first, slice_aligned=True)


def validate_runtime_value(
    value: RuntimeValue,
    output_plan: ArtifactOutputPlan,
    *,
    axis_id: str,
) -> RuntimeValue:
    """Validate a runtime value against the compiled output plan."""
    if value.key.name != output_plan.name:
        raise ValueError(
            f"RuntimeValue name '{value.key.name}' does not match planned "
            f"artifact '{output_plan.name}'."
        )
    if value.kind is not output_plan.kind:
        raise ValueError(
            f"Artifact '{output_plan.name}' expected {output_plan.kind.value}, "
            f"got {value.kind.value}."
        )
    if value.schema.kind is not output_plan.kind:
        raise ValueError(
            f"Artifact '{output_plan.name}' schema kind {value.schema.kind.value} "
            f"does not match planned kind {output_plan.kind.value}."
        )
    if value.key.scope.axis_id != axis_id:
        raise ValueError(
            f"Artifact '{output_plan.name}' belongs to axis "
            f"'{value.key.scope.axis_id}', not '{axis_id}'."
        )

    _validate_payload_kind(output_plan.name, value.kind, value.data, value.schema)
    return value


def _validate_payload_kind(
    name: str,
    kind: ArtifactKind,
    data: Any,
    schema: RuntimeValueSchema,
) -> None:
    ArtifactPayloadValidationStrategy.for_kind(kind).validate(name, data, schema)


class ArtifactPayloadValidationStrategy(
    EnumKeyedStrategyMixin[ArtifactKind],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered validation contract for artifact runtime payloads."""

    __registry_key__ = "kind_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "kind"
    __enum_label_attr__ = "kind_label"

    kind: ClassVar[ArtifactKind | None] = None
    kind_label: ClassVar[str | None] = None

    def __init__(self, kind: ArtifactKind | None = None) -> None:
        self._kind = kind

    @classmethod
    def for_kind(cls, kind: ArtifactKind) -> "ArtifactPayloadValidationStrategy":
        strategy_type = cls.__registry__.get(kind.value)
        strategy = strategy_type() if strategy_type is not None else None
        return (
            strategy
            if strategy is not None
            else GenericArtifactPayloadValidationStrategy(kind)
        )

    @property
    def artifact_kind(self) -> ArtifactKind:
        kind = self._kind or self.kind
        if kind is None:
            raise TypeError(
                f"{type(self).__name__} must declare an artifact kind or be "
                "constructed with one."
            )
        return kind

    def validate(
        self,
        name: str,
        data: Any,
        schema: RuntimeValueSchema,
    ) -> None:
        validator = self._validator(schema)
        if validator is None:
            return
        if schema.slice_aligned:
            if _is_slice_aligned_payload(data) and all(validator(item) for item in data):
                return
            raise TypeError(
                f"Artifact '{name}' expected slice-aligned "
                f"{self.artifact_kind.payload_description}, got {type(data).__name__}."
            )
        if validator(data):
            return
        raise TypeError(
            f"Artifact '{name}' expected {self.artifact_kind.payload_description}, "
            f"got {type(data).__name__}."
        )

    def _validator(
        self,
        schema: RuntimeValueSchema,
    ) -> Callable[[Any], bool] | None:
        return _PAYLOAD_VALIDATORS[_payload_shape_for(self.artifact_kind, schema)]


class GenericArtifactPayloadValidationStrategy(ArtifactPayloadValidationStrategy):
    """Validate artifacts by their declared payload shape."""

    kind = ArtifactKind.SPECIAL


class ObjectLabelsArtifactPayloadValidationStrategy(ArtifactPayloadValidationStrategy):
    """Validate object-label artifacts by nominal label payload semantics."""

    kind = ArtifactKind.OBJECT_LABELS

    def validate(
        self,
        name: str,
        data: Any,
        schema: RuntimeValueSchema,
    ) -> None:
        if isinstance(data, (ObjectLabelPayload, ObjectLabelSet)):
            return
        super().validate(name, data, schema)


def _payload_shape_for(
    kind: ArtifactKind,
    schema: RuntimeValueSchema,
) -> ArtifactPayloadShape:
    if kind.uses_label_representation_payload_shape:
        representation = (
            schema.label_representation or ObjectLabelRepresentation.DENSE_LABELS
        )
        return representation.payload_shape
    return kind.payload_shape


def _is_table_like(data: Any) -> bool:
    _ensure_runtime_payload_integrations_registered()
    return (
        isinstance(data, ColumnarRows)
        or isinstance(data, Mapping)
        or (
            isinstance(data, Sequence)
            and not isinstance(data, (str, bytes, bytearray))
        )
    )


def _is_array_like(data: Any) -> bool:
    return is_array_payload(data)


def _is_mapping_like(data: Any) -> bool:
    return isinstance(data, Mapping)


def _is_slice_aligned_payload(data: Any) -> bool:
    return isinstance(data, Sequence) and not isinstance(
        data,
        (str, bytes, bytearray),
    )


def _validate_object_label_variant(
    object_name: str,
    variant_name: str,
    labels: Any,
    variant_labels: Any | None,
    validator: Callable[[Any], bool] | None,
) -> None:
    if variant_labels is None:
        return
    if validator is not None and not validator(variant_labels):
        raise TypeError(
            f"ObjectLabelSet '{object_name}' {variant_name} requires an "
            f"array-compatible payload, got {type(variant_labels).__name__}."
        )
    labels_shape = tuple(np.shape(labels))
    variant_labels_shape = tuple(np.shape(variant_labels))
    if labels_shape and variant_labels_shape and labels_shape != variant_labels_shape:
        raise ValueError(
            f"ObjectLabelSet '{object_name}' {variant_name} shape "
            f"{variant_labels_shape!r} does not match final labels "
            f"shape {labels_shape!r}."
        )


@dataclass(frozen=True, slots=True)
class _PayloadValidator:
    shape: ArtifactPayloadShape
    predicate: Callable[[Any], bool] | None


def _payload_validators(
    rows: tuple[_PayloadValidator, ...],
) -> Mapping[ArtifactPayloadShape, Callable[[Any], bool] | None]:
    validators = {row.shape: row.predicate for row in rows}
    if set(validators) != set(ArtifactPayloadShape):
        raise TypeError("Incomplete runtime payload validator table.")
    return MappingProxyType(validators)


_PAYLOAD_VALIDATORS = _payload_validators(
    (
        _PayloadValidator(ArtifactPayloadShape.ANY, None),
        _PayloadValidator(ArtifactPayloadShape.ARRAY, _is_array_like),
        _PayloadValidator(ArtifactPayloadShape.TABLE, _is_table_like),
        _PayloadValidator(ArtifactPayloadShape.MAPPING, _is_mapping_like),
    )
)


@dataclass(frozen=True, slots=True)
class RequiredMappingField:
    """Resolve a required canonical mapping field with legacy aliases."""

    data: Mapping[str, Any]
    name: str
    aliases: tuple[str, ...] = ()

    def value(self) -> Any:
        for key in (self.name, *self.aliases):
            if key in self.data:
                return self.data[key]
        names = ", ".join(repr(key) for key in (self.name, *self.aliases))
        raise KeyError(f"Missing required mapping field {names}.")


@dataclass(frozen=True, slots=True)
class OptionalMappingField:
    """Optional value lookup and coercion over mapping-backed runtime values."""

    data: Mapping[str, Any]
    name: str
    aliases: tuple[str, ...] = ()

    def int_or(self, *, default: int) -> int:
        if self.name not in self.data:
            return default
        return int(self.data[self.name])

    def nullable_int(self) -> int | None:
        if self.name not in self.data or self.data[self.name] is None:
            return None
        return int(self.data[self.name])

    def int_tuple(self) -> tuple[int, ...]:
        for key in (self.name, *self.aliases):
            if key not in self.data or self.data[key] is None:
                continue
            value = self.data[key]
            if isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                return tuple(int(item) for item in value)
            raise TypeError(
                f"Optional integer tuple field '{key}' must be a sequence, "
                f"got {type(value).__name__}."
            )
        return ()

    def nullable_float(self) -> float | None:
        if self.name not in self.data or self.data[self.name] is None:
            return None
        return float(self.data[self.name])

    def nullable_str(self) -> str | None:
        if self.name not in self.data or self.data[self.name] is None:
            return None
        return str(self.data[self.name])

    def nullable_str_tuple(self) -> tuple[str | None, ...]:
        if self.name not in self.data or self.data[self.name] is None:
            return ()
        value = self.data[self.name]
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return tuple(None if item is None else str(item) for item in value)
        raise TypeError(
            f"Optional string tuple field '{self.name}' must be a sequence, "
            f"got {type(value).__name__}."
        )

    def nullable_mapping(self) -> SourceComponentMetadata | None:
        if self.name not in self.data or self.data[self.name] is None:
            return None
        value = self.data[self.name]
        if isinstance(value, Mapping):
            return MappingProxyType(dict(value))
        raise TypeError(
            f"Optional mapping field '{self.name}' must be a mapping, "
            f"got {type(value).__name__}."
        )

    def nullable_mapping_tuple(self) -> ChannelSourceComponentMetadata:
        if self.name not in self.data or self.data[self.name] is None:
            return ()
        value = self.data[self.name]
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return tuple(
                None if item is None else MappingProxyType(dict(item))
                for item in value
            )
        raise TypeError(
            f"Optional mapping tuple field '{self.name}' must be a sequence, "
            f"got {type(value).__name__}."
        )

    def grid_ordering(self) -> SpatialGridOrdering:
        if self.name not in self.data or self.data[self.name] is None:
            return SpatialGridOrdering.BY_ROWS
        return coerce_enum(
            SpatialGridOrdering,
            self.data[self.name],
            f"SpatialGrid.{self.name}",
        )


def _required_int(
    data: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> int:
    return int(RequiredMappingField(data, name, aliases).value())


def _required_float(
    data: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> float:
    return float(RequiredMappingField(data, name, aliases).value())


def _required_constructor_float(value: float | None, name: str) -> float:
    if value is None:
        raise TypeError(f"{name} is required.")
    return float(value)


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")


def _validate_native_name(output_plan: ArtifactOutputPlan, name: str) -> None:
    if name != output_plan.name:
        raise ValueError(
            f"Native runtime value '{name}' does not match planned artifact "
            f"'{output_plan.name}'."
        )


def _resolve_measurement_subject(
    subject: MeasurementSubject | None,
    *,
    artifact_name: str,
    object_name: str | None,
    object_id_field: str | None,
    source_image_name: str | None,
) -> MeasurementSubject:
    if subject is None:
        if object_name is not None:
            return MeasurementSubject(
                MeasurementScope.OBJECT,
                object_name,
                object_id_field,
            )
        if source_image_name is not None:
            return MeasurementSubject(MeasurementScope.IMAGE, source_image_name)
        return MeasurementSubject(MeasurementScope.ARTIFACT, artifact_name)

    if object_name is not None and (
        subject.scope is not MeasurementScope.OBJECT or subject.name != object_name
    ):
        raise ValueError(
            "MeasurementTable.object_name conflicts with "
            "MeasurementTable.subject."
        )
    if object_id_field is not None and subject.id_field != object_id_field:
        raise ValueError(
            "MeasurementTable.object_id_field conflicts with "
            "MeasurementTable.subject."
        )
    if (
        source_image_name is not None
        and subject.scope is MeasurementScope.IMAGE
        and subject.name != source_image_name
    ):
        raise ValueError(
            "MeasurementTable.source_image_name conflicts with "
            "MeasurementTable.subject."
        )
    return subject


@dataclass(frozen=True, slots=True)
class MeasurementTableSubjectResolver:
    """Resolve legacy measurement-table fields against the nominal subject."""

    table: MeasurementTable

    @property
    def object_name(self) -> str | None:
        if self.table.object_name is not None:
            return self.table.object_name
        if self.table.subject and self.table.subject.scope is MeasurementScope.OBJECT:
            return self.table.subject.name
        return None

    @property
    def object_id_field(self) -> str | None:
        if self.table.object_id_field is not None:
            return self.table.object_id_field
        if self.table.subject and self.table.subject.scope is MeasurementScope.OBJECT:
            return self.table.subject.id_field
        return None

    @property
    def source_image_name(self) -> str | None:
        if self.table.source_image_name is not None:
            return self.table.source_image_name
        if self.table.subject and self.table.subject.scope is MeasurementScope.IMAGE:
            return self.table.subject.source_image_name
        return None


@dataclass(frozen=True, slots=True)
class RuntimePayloadFieldInference:
    """Infer tabular field declarations from runtime payload row shapes."""

    rows: Any

    def fields(self) -> tuple[FieldSpec, ...]:
        _ensure_runtime_payload_integrations_registered()
        if isinstance(self.rows, ColumnarRows):
            return tuple(FieldSpec(str(column)) for column in self.rows.columns)
        if isinstance(self.rows, Mapping):
            return tuple(FieldSpec(str(column)) for column in self.rows)
        if (
            isinstance(self.rows, Sequence)
            and self.rows
            and isinstance(self.rows[0], Mapping)
        ):
            return tuple(FieldSpec(str(column)) for column in self.rows[0])
        return ()


def _ensure_runtime_payload_integrations_registered() -> None:
    """Load optional external payload capability registrations."""
    from openhcs.core.runtime_payload_integrations import (
        register_runtime_payload_integrations,
    )

    register_runtime_payload_integrations()


def _validate_relationship_ids(source_ids: Any, target_ids: Any, name: str) -> None:
    if isinstance(source_ids, Sequence) and isinstance(target_ids, Sequence):
        if (
            not isinstance(source_ids, (str, bytes, bytearray))
            and not isinstance(target_ids, (str, bytes, bytearray))
            and len(source_ids) != len(target_ids)
        ):
            raise ValueError(
                f"ObjectRelationship '{name}' source_ids and target_ids must "
                f"have equal length, got {len(source_ids)} and {len(target_ids)}."
            )
