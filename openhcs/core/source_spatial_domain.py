"""Source-image XY placement domain and dense materialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Self, TypeVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
import numpy as np

from zmqruntime.viewer_protocol import (
    ViewerSourceSpatialDomainPayload,
    ViewerWireMapping,
    ViewerWireValue,
)

from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin

SourceSpatialAliasValueT = TypeVar("SourceSpatialAliasValueT")


def _spatial_shape_pair(value: Sequence[int], field_name: str) -> tuple[int, int]:
    """Return the leading Y/X dimensions from a spatial shape value."""
    if len(value) < 2:
        raise ValueError(f"{field_name} must have at least two spatial dimensions.")
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
        height, width = _spatial_shape_pair(value, field_name)
        return cls(height=height, width=width)

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


@dataclass(frozen=True, slots=True)
class SourceSpatialDomain:
    """Dense XY placement contract for a source-image coordinate domain."""

    origin_yx: tuple[int, int] | None = None
    source_shape_yx: tuple[int, int] | None = None
    fill_value: Any = 0
    value_name: str = "Dense array"

    @property
    def has_values(self) -> bool:
        """Return whether this domain carries source-image placement metadata."""
        return self.origin_yx is not None or self.source_shape_yx is not None

    @classmethod
    def from_viewer_wire_mapping(
        cls,
        payload: ViewerWireMapping,
        *,
        source_label: str,
        fill_value: Any = 0,
        value_name: str = "Dense array",
    ) -> "SourceSpatialDomain":
        """Build a source-spatial domain from viewer-wire metadata."""
        source_payload = ViewerSourceSpatialDomainPayload.from_wire_mapping(
            payload,
            source_label=source_label,
        )
        return cls(
            origin_yx=source_payload.origin_yx,
            source_shape_yx=source_payload.source_shape_yx,
            fill_value=fill_value,
            value_name=value_name,
        )

    def to_viewer_wire_mapping(self) -> dict[str, ViewerWireValue]:
        """Return viewer-wire metadata for this source-spatial domain."""
        return ViewerSourceSpatialDomainPayload(
            origin_yx=self.origin_yx,
            source_shape_yx=self.source_shape_yx,
        ).to_wire_mapping()

    def with_missing_from(self, fallback: "SourceSpatialDomain") -> Self:
        """Fill missing spatial placement values from another source domain."""
        return type(self)(
            origin_yx=(
                self.origin_yx if self.origin_yx is not None else fallback.origin_yx
            ),
            source_shape_yx=(
                self.source_shape_yx
                if self.source_shape_yx is not None
                else fallback.source_shape_yx
            ),
            fill_value=self.fill_value,
            value_name=self.value_name,
        )

    @classmethod
    def common_from_domains(
        cls,
        domains: Iterable["SourceSpatialDomain"],
        *,
        expand_varying_domains: bool = False,
        fill_value: Any = 0,
        value_name: str = "Dense array",
    ) -> "SourceSpatialDomain":
        """Return the shared source-spatial domain represented by many values."""
        domains_tuple = tuple(domains)
        if not domains_tuple:
            return cls(fill_value=fill_value, value_name=value_name)

        source_shape = CommonRuntimeValue.from_values(
            domain.source_shape_yx for domain in domains_tuple
        ).single
        if expand_varying_domains and cls.domains_have_varying_complete_placement(
            domains_tuple
        ):
            return cls(
                origin_yx=None,
                source_shape_yx=source_shape,
                fill_value=fill_value,
                value_name=value_name,
            )

        return cls(
            origin_yx=CommonRuntimeValue.from_values(
                domain.origin_yx for domain in domains_tuple
            ).single,
            source_shape_yx=source_shape,
            fill_value=fill_value,
            value_name=value_name,
        )

    @staticmethod
    def domains_have_varying_complete_placement(
        domains: tuple["SourceSpatialDomain", ...],
    ) -> bool:
        if len(domains) <= 1:
            return False
        identities = tuple(
            (domain.origin_yx, domain.source_shape_yx) for domain in domains
        )
        if any(origin is None or shape is None for origin, shape in identities):
            return False
        return len(set(identities)) > 1

    def with_origin_yx(self, origin_yx: tuple[int, int] | None) -> Self:
        """Return this domain with a replacement source-image origin."""
        return type(self)(
            origin_yx=origin_yx,
            source_shape_yx=self.source_shape_yx,
            fill_value=self.fill_value,
            value_name=self.value_name,
        )

    def with_source_shape_yx(
        self,
        source_shape_yx: tuple[int, int] | None,
    ) -> Self:
        """Return this domain with a replacement full source-image shape."""
        return type(self)(
            origin_yx=self.origin_yx,
            source_shape_yx=source_shape_yx,
            fill_value=self.fill_value,
            value_name=self.value_name,
        )

    def with_value_name(self, value_name: str) -> Self:
        """Return this domain with a consumer-specific value label."""
        return type(self)(
            origin_yx=self.origin_yx,
            source_shape_yx=self.source_shape_yx,
            fill_value=self.fill_value,
            value_name=value_name,
        )

    def with_fill_value(self, fill_value: Any) -> Self:
        """Return this domain with a replacement dense materialization fill."""
        return type(self)(
            origin_yx=self.origin_yx,
            source_shape_yx=self.source_shape_yx,
            fill_value=fill_value,
            value_name=self.value_name,
        )

    def normalized(self) -> Self:
        """Return this domain with canonical tuple metadata values."""
        origin_yx = (
            None
            if self.origin_yx is None
            else self._shape_yx(self.origin_yx, "spatial_origin_yx")
        )
        source_shape_yx = (
            None
            if self.source_shape_yx is None
            else self._shape_yx(
                self.source_shape_yx,
                "source_spatial_shape_yx",
            )
        )
        return type(self)(
            origin_yx=origin_yx,
            source_shape_yx=source_shape_yx,
            fill_value=self.fill_value,
            value_name=self.value_name,
        )

    def physical_border_edges_for_shape(
        self,
        image_shape_yx: Sequence[int],
    ) -> tuple[bool, bool, bool, bool]:
        """Return which local image edges coincide with the full source image."""
        if self.origin_yx is None or self.source_shape_yx is None:
            return True, True, True, True

        height, width = self._shape_yx(image_shape_yx, "image_shape_yx")
        origin_y, origin_x = self.origin_yx
        source_height, source_width = self.source_shape_yx
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
    ) -> Self:
        """Return this source domain projected through one spatial crop."""
        input_shape = self._shape_yx(input_shape_yx, "input_shape_yx")
        self._shape_yx(output_shape_yx, "output_shape_yx")
        parent_origin = self.origin_yx if self.origin_yx is not None else (0, 0)
        source_shape = (
            self.source_shape_yx if self.source_shape_yx is not None else input_shape
        )
        return type(self)(
            origin_yx=(
                int(parent_origin[0]) + int(offset_yx[0]),
                int(parent_origin[1]) + int(offset_yx[1]),
            ),
            source_shape_yx=source_shape,
            fill_value=self.fill_value,
            value_name=self.value_name,
        )

    def with_spatial_resize(
        self,
        output_shape_yx: Sequence[int],
    ) -> Self:
        """Return the local coordinate domain established by a spatial resize."""

        output_shape = self._shape_yx(output_shape_yx, "output_shape_yx")
        return type(self)(
            origin_yx=(0, 0),
            source_shape_yx=output_shape,
            fill_value=self.fill_value,
            value_name=self.value_name,
        )

    def as_materialized_source_domain(
        self,
        target_domain: "SourceSpatialDomain",
    ) -> Self:
        """Return this domain after its value is expanded to source-image XY."""
        source_shape = self.source_shape_yx or target_domain.source_shape_yx
        if source_shape is None:
            raise ValueError(
                f"{self.value_name} source-domain materialization requires shape."
            )
        return type(self)(
            origin_yx=(0, 0),
            source_shape_yx=source_shape,
            fill_value=self.fill_value,
            value_name=self.value_name,
        )

    @staticmethod
    def _shape_yx(value: Sequence[int], field_name: str) -> tuple[int, int]:
        return _spatial_shape_pair(value, field_name)

    def materialize(
        self,
        value: Any,
        *,
        spatial_axes_yx: tuple[int, int],
    ) -> Any:
        """Place an array-like value into this source spatial domain."""
        return dense_array_in_source_spatial_domain(
            value,
            spatial_axes_yx=spatial_axes_yx,
            spatial_origin_yx=self.origin_yx,
            source_spatial_shape_yx=self.source_shape_yx,
            fill_value=self.fill_value,
            value_name=self.value_name,
        )


@dataclass(frozen=True, slots=True)
class SourceSpatialDomainAlias(Generic[SourceSpatialAliasValueT]):
    """Descriptor for scalar aliases backed by a source-spatial domain."""

    getter: Callable[[SourceSpatialDomain], SourceSpatialAliasValueT]
    setter: Callable[
        [SourceSpatialDomain, SourceSpatialAliasValueT],
        SourceSpatialDomain,
    ]

    def __get__(
        self,
        instance: "SourceSpatialDomainFields | None",
        _owner: type["SourceSpatialDomainFields"],
    ) -> SourceSpatialAliasValueT | Self:
        if instance is None:
            return self
        return self.getter(instance.source_spatial_domain)

    def __set__(
        self,
        instance: "SourceSpatialDomainFields",
        value: SourceSpatialAliasValueT,
    ) -> None:
        instance.source_spatial_domain = self.setter(
            instance.source_spatial_domain,
            value,
        )


@dataclass(kw_only=True)
class SourceSpatialDomainFields:
    """Source-image spatial domain carried by runtime payload metadata."""

    source_spatial_domain: SourceSpatialDomain = field(
        default_factory=SourceSpatialDomain
    )

    def normalize_source_spatial_domain_fields(self) -> None:
        self.source_spatial_domain = self.source_spatial_domain.normalized()


SourceSpatialDomainFields.spatial_origin_yx = SourceSpatialDomainAlias(
    lambda domain: domain.origin_yx,
    SourceSpatialDomain.with_origin_yx,
)
SourceSpatialDomainFields.source_spatial_shape_yx = SourceSpatialDomainAlias(
    lambda domain: domain.source_shape_yx,
    SourceSpatialDomain.with_source_shape_yx,
)


@dataclass(frozen=True, slots=True)
class SourceSpatialPayloadDomain:
    """Native payload placement identity inside an optional source XY domain."""

    origin_yx: tuple[int, int]
    spatial_shape_yx: tuple[int, int]
    source_shape_yx: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class CommonRuntimeValue:
    """Projection of a value family that is only valid when all values agree."""

    values: tuple[Any, ...]

    @classmethod
    def from_values(
        cls,
        values: Iterable[Any],
        *,
        ignore_none: bool = False,
    ) -> "CommonRuntimeValue":
        unique_values: list[Any] = []
        for value in values:
            if ignore_none and value is None:
                continue
            if not any(value == existing for existing in unique_values):
                unique_values.append(value)
        return cls(tuple(unique_values))

    @property
    def single(self) -> Any | None:
        """Return the shared value, or None when values disagree or are absent."""
        if len(self.values) == 1:
            return self.values[0]
        return None


class SourceSpatialDomainAdapter(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Adapter for dense XY payloads that carry source-domain coordinates."""

    value_type: ClassVar[type[object] | tuple[type[object], ...] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def for_value(
        cls,
        value: Any,
        *,
        source_shape_override_yx: tuple[int, int] | None = None,
    ) -> "SourceSpatialDomainAdapter | None":
        strategy_types = cls.strategy_types_for_nominal_value(value)
        if not strategy_types:
            return None
        return strategy_types[0].for_value(
            value,
            source_shape_override_yx=source_shape_override_yx,
        )

    @property
    @abstractmethod
    def array(self) -> Any: ...

    @property
    @abstractmethod
    def domain(self) -> SourceSpatialDomain: ...

    @property
    @abstractmethod
    def spatial_axes_yx(self) -> tuple[int, int]:
        """Return the payload axes declared to carry Y/X coordinates."""

    @abstractmethod
    def value_in_payload_domain(
        self,
        target: "SourceSpatialDomainAdapter",
    ) -> Any:
        """Project this value into ``target`` while preserving its carrier type."""

    @property
    def spatial_shape_yx(self) -> tuple[int, int]:
        """Return the payload's native XY shape before source-domain expansion."""
        array = np.asarray(self.array)
        y_axis, x_axis = self.spatial_axes_yx
        return int(array.shape[y_axis]), int(array.shape[x_axis])

    @property
    def payload_domain(self) -> SourceSpatialPayloadDomain:
        """Return this payload's native placement identity."""
        return SourceSpatialPayloadDomain(
            origin_yx=self.domain.origin_yx or (0, 0),
            spatial_shape_yx=self.spatial_shape_yx,
            source_shape_yx=self.domain.source_shape_yx,
        )

    @classmethod
    def common_payload_domain(
        cls,
        adapters: tuple["SourceSpatialDomainAdapter", ...],
    ) -> SourceSpatialPayloadDomain | None:
        """Return the shared native payload domain, if every adapter agrees."""
        return CommonRuntimeValue.from_values(
            adapter.payload_domain for adapter in adapters
        ).single

    @classmethod
    def common_source_shape_yx(
        cls,
        adapters: tuple["SourceSpatialDomainAdapter", ...],
    ) -> tuple[int, int] | None:
        """Return the shared source XY shape, if every declared source agrees."""
        return CommonRuntimeValue.from_values(
            (adapter.domain.source_shape_yx for adapter in adapters),
            ignore_none=True,
        ).single

    @classmethod
    def common_source_domain(
        cls,
        adapters: tuple["SourceSpatialDomainAdapter", ...],
        *,
        value_name: str,
        fill_value: Any = 0,
    ) -> SourceSpatialDomain | None:
        """Return the shared source-image domain declared by these adapters."""
        source_shape = cls.common_source_shape_yx(adapters)
        if source_shape is None:
            return None
        return SourceSpatialDomain(
            source_shape_yx=source_shape,
            fill_value=fill_value,
            value_name=value_name,
        )

    @classmethod
    def requires_source_domain_alignment(
        cls,
        adapters: tuple["SourceSpatialDomainAdapter", ...],
    ) -> bool:
        """Return whether payloads must be expanded before joint execution."""
        source_shape = cls.common_source_shape_yx(adapters)
        if source_shape is None:
            return False
        common_payload_domain = cls.common_payload_domain(adapters)
        if common_payload_domain is not None:
            return False
        return True

    def materialize(self) -> Any:
        """Return the payload array in source-image XY coordinates."""
        return self.domain.materialize(
            self.array,
            spatial_axes_yx=self.spatial_axes_yx,
        )

    @classmethod
    def aligned_values(
        cls,
        values: tuple[Any, ...],
    ) -> tuple[tuple[Any, ...], tuple["SourceSpatialDomainAdapter", ...]]:
        """Materialize nominal values into one exact source-spatial geometry."""

        adapters: list[SourceSpatialDomainAdapter] = []
        for value in values:
            adapter = cls.for_value(value)
            if adapter is None:
                raise TypeError(
                    "Source-spatial alignment requires a registered nominal adapter "
                    f"for {type(value).__name__}."
                )
            adapters.append(adapter)
        resolved_adapters = tuple(adapters)
        declared_source_domains = tuple(
            adapter.domain.source_shape_yx is not None for adapter in resolved_adapters
        )
        if any(declared_source_domains) and not all(declared_source_domains):
            raise ValueError(
                "Source-spatial alignment requires every value to declare a source "
                "domain or no value to declare one."
            )
        if declared_source_domains and declared_source_domains[0]:
            source_shape = cls.common_source_shape_yx(resolved_adapters)
            if source_shape is None:
                raise ValueError(
                    "Source-spatial alignment requires one compatible declared "
                    "source shape for every value."
                )
        arrays = tuple(
            np.asarray(adapter.materialize()) for adapter in resolved_adapters
        )
        shapes = tuple(array.shape for array in arrays)
        if shapes and any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError(
                "Source-spatial values must share a common geometry after alignment; "
                f"got {shapes!r}."
            )
        return arrays, resolved_adapters

    def extract_source_array(
        self,
        value: Any,
        *,
        spatial_axes_yx: tuple[int, int],
    ) -> Any:
        """Project a source-domain dense array into this payload's native domain."""
        import numpy as np

        array = np.asarray(value)
        payload_domain = self.payload_domain
        source_shape_yx = payload_domain.source_shape_yx
        if source_shape_yx is None:
            return value
        y_axis, x_axis = spatial_axes_yx
        spatial_shape_yx = int(array.shape[y_axis]), int(array.shape[x_axis])
        if spatial_shape_yx != tuple(source_shape_yx):
            return value
        if spatial_shape_yx == payload_domain.spatial_shape_yx:
            return value
        origin_y, origin_x = payload_domain.origin_yx
        height, width = payload_domain.spatial_shape_yx
        slices = [slice(None)] * array.ndim
        slices[y_axis] = slice(origin_y, origin_y + height)
        slices[x_axis] = slice(origin_x, origin_x + width)
        return array[tuple(slices)]


def dense_array_in_source_spatial_domain(
    value: Any,
    *,
    spatial_axes_yx: tuple[int, int],
    spatial_origin_yx: tuple[int, int] | None,
    source_spatial_shape_yx: tuple[int, int] | None,
    fill_value: Any = 0,
    value_name: str = "Dense array",
) -> Any:
    """Place a dense XY array payload into its declared source XY domain."""
    import numpy as np

    label_array = np.asarray(value)
    origin = spatial_origin_yx
    source_shape = source_spatial_shape_yx
    if origin is None or source_shape is None:
        return label_array

    source_y, source_x = (int(source_shape[0]), int(source_shape[1]))
    origin_y, origin_x = (int(origin[0]), int(origin[1]))
    if source_y < 0 or source_x < 0 or origin_y < 0 or origin_x < 0:
        raise ValueError(
            f"{value_name} spatial domains require non-negative source shape "
            f"and origin; got source={source_shape!r}, origin={origin!r}."
        )
    if label_array.ndim < 2:
        raise ValueError(
            f"{value_name} spatial domains require at least 2D arrays; got "
            f"shape {label_array.shape!r}."
        )
    y_axis, x_axis = (int(axis) for axis in spatial_axes_yx)
    if (
        y_axis == x_axis
        or y_axis < 0
        or x_axis < 0
        or y_axis >= label_array.ndim
        or x_axis >= label_array.ndim
    ):
        raise ValueError(
            f"{value_name} spatial axes {spatial_axes_yx!r} are invalid for "
            f"shape {label_array.shape!r}."
        )
    payload_y = int(label_array.shape[y_axis])
    payload_x = int(label_array.shape[x_axis])
    if (payload_y, payload_x) == (source_y, source_x) and origin == (0, 0):
        return label_array

    if origin_y + payload_y > source_y or origin_x + payload_x > source_x:
        raise ValueError(
            f"{value_name} crop exceeds its declared source domain; got array "
            f"{label_array.shape!r}, source={source_shape!r}, origin={origin!r}."
        )

    expanded_shape = list(label_array.shape)
    expanded_shape[y_axis] = source_y
    expanded_shape[x_axis] = source_x
    expanded = np.full(expanded_shape, fill_value, dtype=label_array.dtype)
    target_slices = [slice(None)] * label_array.ndim
    target_slices[y_axis] = slice(origin_y, origin_y + payload_y)
    target_slices[x_axis] = slice(origin_x, origin_x + payload_x)
    expanded[tuple(target_slices)] = label_array
    return expanded
