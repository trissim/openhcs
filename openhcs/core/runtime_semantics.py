"""Generic semantic contracts for typed runtime artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import Any, ClassVar, cast

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
import numpy as np

from openhcs.core.artifacts import ArtifactKind, ArtifactPayloadShape
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    GeneratedEnumClassSpec,
    GeneratedLeafClassSpec,
    MostDerivedContextStrategyMixin,
    NominalTypeKeyedStrategyMixin,
    str_enum_member_with_payload,
)
from openhcs.core.process_local_cache import ProcessLocalBoundedCache


DeclaredObjectIds = tuple[int, ...] | list[int] | None


@dataclass(frozen=True, slots=True)
class SourceSpatialDomain:
    """Dense XY placement contract for a source-image coordinate domain."""

    origin_yx: tuple[int, int] | None = None
    source_shape_yx: tuple[int, int] | None = None
    fill_value: Any = 0
    value_name: str = "Dense array"

    def materialize(self, value: Any) -> Any:
        """Place an array-like value into this source spatial domain."""
        return dense_array_in_source_spatial_domain(
            value,
            spatial_origin_yx=self.origin_yx,
            source_spatial_shape_yx=self.source_shape_yx,
            fill_value=self.fill_value,
            value_name=self.value_name,
        )

    def materialize_for_slice(
        self,
        value: Any,
        slice_index: int,
        slice_count: int,
    ) -> Any:
        """Place an array-like value and return one aligned execution slice."""
        import numpy as np

        materialized = self.materialize(value)
        array = np.asarray(materialized)
        if array.ndim >= 3 and array.shape[0] == slice_count:
            return array[slice_index]
        return materialized


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
        for adapter_type in cls.registered_strategy_types():
            adapter = adapter_type.for_value(
                value,
                source_shape_override_yx=source_shape_override_yx,
            )
            if adapter is not None:
                return adapter
        return None

    @property
    @abstractmethod
    def array(self) -> Any:
        ...

    @property
    @abstractmethod
    def domain(self) -> SourceSpatialDomain:
        ...

    @property
    def spatial_shape_yx(self) -> tuple[int, int]:
        """Return the payload's native XY shape before source-domain expansion."""
        import numpy as np

        array = np.asarray(self.array)
        if array.ndim < 2:
            raise ValueError(
                "Source-spatial payloads require at least two dimensions, "
                f"got {array.ndim}."
            )
        return tuple(int(axis) for axis in array.shape[-2:])

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
        return self.domain.materialize(self.array)

    def materialize_for_slice(self, slice_index: int, slice_count: int) -> Any:
        """Return one aligned execution slice from the materialized payload."""
        return self.domain.materialize_for_slice(
            self.array,
            slice_index,
            slice_count,
        )

    def extract_source_array(self, value: Any) -> Any:
        """Project a source-domain dense array into this payload's native domain."""
        import numpy as np

        array = np.asarray(value)
        payload_domain = self.payload_domain
        source_shape_yx = payload_domain.source_shape_yx
        if source_shape_yx is None:
            return value
        if array.ndim < 2 or tuple(array.shape[-2:]) != tuple(source_shape_yx):
            return value
        if tuple(array.shape[-2:]) == payload_domain.spatial_shape_yx:
            return value
        origin_y, origin_x = payload_domain.origin_yx
        height, width = payload_domain.spatial_shape_yx
        return array[..., origin_y : origin_y + height, origin_x : origin_x + width]


@dataclass(frozen=True, slots=True)
class DenseArraySourceSpatialDomainAdapter(SourceSpatialDomainAdapter):
    """Source-domain adapter for dense array-like payloads."""

    value: Any
    source_domain: SourceSpatialDomain = SourceSpatialDomain()
    array = AliasProperty[Any]("value")
    domain = AliasProperty[SourceSpatialDomain]("source_domain")

    @classmethod
    def for_value(
        cls,
        value: Any,
        *,
        source_shape_override_yx: tuple[int, int] | None = None,
    ) -> "DenseArraySourceSpatialDomainAdapter | None":
        return None

@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One named field expected in a tabular runtime value."""

    name: str
    dtype: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Runtime value field name cannot be empty.")


class ObjectLabelRepresentation(str, Enum):
    """Storage representation used by an object-label artifact payload."""

    def __new__(cls, value: str, payload_shape: ArtifactPayloadShape):
        return str_enum_member_with_payload(
            cls,
            value,
            payload_attribute="_payload_shape",
            payload=payload_shape,
        )

    DENSE_LABELS = ("dense_labels", ArtifactPayloadShape.ARRAY)
    SPARSE_IJV = ("sparse_ijv", ArtifactPayloadShape.TABLE)
    payload_shape = AliasProperty[ArtifactPayloadShape]("_payload_shape")


class ObjectLabelVariant(str, Enum):
    """Named semantic variants carried by an object-label artifact."""

    FINAL = "final"
    UNEDITED = "unedited"
    SMALL_REMOVED = "small_removed"


class ObjectLabelDomainScope(str, Enum):
    """How declared object-label IDs apply across dense label planes."""

    PAYLOAD = "payload"
    PLANE = "plane"

    @classmethod
    def common(cls, scopes: Any) -> "ObjectLabelDomainScope":
        """Return the common scope for merged labels, defaulting to payload scope."""
        unique_scopes = tuple(dict.fromkeys(coerce_enum(cls, scope, "ObjectLabelDomain.scope") for scope in scopes))
        if len(unique_scopes) == 1:
            return unique_scopes[0]
        return cls.PAYLOAD


class RuntimePlaneAxis(str, Enum):
    """Semantic meaning of the leading plane axis on runtime array stacks."""

    RUNTIME_SLICE = "runtime_slice"
    SOURCE_BINDING = "source_binding"

    @classmethod
    def common(cls, axes: Any) -> "RuntimePlaneAxis":
        """Return the common plane axis for merged labels."""
        unique_axes = tuple(
            dict.fromkeys(
                coerce_enum(cls, axis, "RuntimePlaneAxis") for axis in axes
            )
        )
        if len(unique_axes) != 1:
            raise ValueError(
                "Cannot merge object-label stacks with different plane-axis semantics: "
                f"{unique_axes!r}."
            )
        return unique_axes[0]


class MeasurementImageReferenceDomain(str, Enum):
    """Semantic image domain used as the reference for object measurement."""

    SOURCE_IMAGE = "source_image"
    OBJECT_LABELS = "object_labels"


class RuntimePlaneProjectionScope(str, Enum):
    """Execution scope that determines whether a runtime slice is selectable."""

    STACK = "stack"
    GROUP = "group"


@dataclass(frozen=True, slots=True)
class RuntimePlaneProjection:
    """Nominal runtime-plane selection for one callable invocation."""

    scope: RuntimePlaneProjectionScope
    plane_index: int | None = None

    def __post_init__(self) -> None:
        scope = coerce_enum(
            RuntimePlaneProjectionScope,
            self.scope,
            "RuntimePlaneProjection.scope",
        )
        object.__setattr__(self, "scope", scope)
        if scope is RuntimePlaneProjectionScope.STACK:
            if self.plane_index is not None:
                raise ValueError(
                    "Stack runtime-plane projection cannot carry a plane index."
                )
            return
        if self.plane_index is None:
            raise ValueError(
                "Grouped runtime-plane projection requires a plane index."
            )
        plane_index = int(self.plane_index)
        if plane_index < 0:
            raise ValueError(
                "Grouped runtime-plane projection plane_index cannot be negative."
            )
        object.__setattr__(self, "plane_index", plane_index)

    @classmethod
    def stack(cls) -> "RuntimePlaneProjection":
        """Preserve runtime-slice stacks for stack-scoped execution."""
        return cls(RuntimePlaneProjectionScope.STACK)

    @classmethod
    def group(cls, plane_index: int) -> "RuntimePlaneProjection":
        """Select one runtime-slice plane for grouped execution."""
        return cls(RuntimePlaneProjectionScope.GROUP, plane_index)

    @classmethod
    def for_group_key(
        cls,
        group_key: Any,
        *,
        plane_index: int | None,
    ) -> "RuntimePlaneProjection":
        """Derive validated projection semantics from compiled group identity."""
        if group_key is None:
            if plane_index is not None:
                raise ValueError(
                    "Ungrouped runtime execution cannot carry a plane index."
                )
            return cls.stack()
        if plane_index is None:
            raise ValueError(
                "Grouped runtime execution requires the OpenHCS component plane "
                "index."
            )
        return cls.group(plane_index)

    def runtime_slice_plane_index(self) -> int | None:
        """Return selected runtime-slice plane, or None when stacks are preserved."""
        return self.plane_index


StackRuntimePlaneProjection = RuntimePlaneProjection
GroupRuntimePlaneProjection = RuntimePlaneProjection


class RuntimePlaneAxisProjector(ABC):
    """Nominal provider for execution-local runtime plane selection."""

    @abstractmethod
    def runtime_slice_plane_index(self) -> int | None:
        """Return the execution-local runtime-slice plane index."""

    @abstractmethod
    def source_binding_axis_plane_index(
        self,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        """Return the execution-local source-binding plane index."""


class RuntimePlaneAxisProjectionStrategy(
    EnumKeyedStrategyMixin[RuntimePlaneAxis],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Polymorphic projection policy for runtime plane axes."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "axis"
    axis: ClassVar[RuntimePlaneAxis]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def plane_index(
        self,
        projector: RuntimePlaneAxisProjector,
        *,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        """Return the current execution plane for this axis."""


class RuntimeSlicePlaneAxisProjectionStrategy(RuntimePlaneAxisProjectionStrategy):
    """Runtime-slice planes are selected by the current execution axis."""

    axis = RuntimePlaneAxis.RUNTIME_SLICE

    def plane_index(
        self,
        projector: RuntimePlaneAxisProjector,
        *,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        return projector.runtime_slice_plane_index()


class SourceBindingPlaneAxisProjectionStrategy(RuntimePlaneAxisProjectionStrategy):
    """Source-binding planes are selected by source alias bindings."""

    axis = RuntimePlaneAxis.SOURCE_BINDING

    def plane_index(
        self,
        projector: RuntimePlaneAxisProjector,
        *,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        return projector.source_binding_axis_plane_index(source_aliases)


@dataclass(frozen=True, slots=True)
class ObjectLabelDomain:
    """Declared object-label identity domain metadata."""

    declared_object_count: int | None = None
    declared_object_ids: tuple[int, ...] = ()
    declared_object_id_domains: tuple[tuple[int, ...], ...] = ()
    scope: ObjectLabelDomainScope = ObjectLabelDomainScope.PAYLOAD

    def __post_init__(self) -> None:
        if self.declared_object_count is not None:
            count = int(self.declared_object_count)
            if count < 0:
                raise ValueError("ObjectLabelDomain.declared_object_count cannot be negative.")
            object.__setattr__(self, "declared_object_count", count)
        object.__setattr__(
            self,
            "declared_object_ids",
            self._normalize_ids(self.declared_object_ids, "declared_object_ids"),
        )
        object.__setattr__(
            self,
            "declared_object_id_domains",
            tuple(
                self._normalize_ids(domain, "declared_object_id_domains")
                for domain in self.declared_object_id_domains
            ),
        )
        object.__setattr__(
            self,
            "scope",
            coerce_enum(ObjectLabelDomainScope, self.scope, "ObjectLabelDomain.scope"),
        )

    @staticmethod
    def _normalize_ids(ids: tuple[int, ...] | list[int], field_name: str) -> tuple[int, ...]:
        normalized = tuple(int(object_id) for object_id in ids)
        if any(object_id <= 0 for object_id in normalized):
            raise ValueError(f"ObjectLabelDomain.{field_name} IDs must be positive.")
        return tuple(sorted(dict.fromkeys(normalized)))

    @classmethod
    def declared(
        cls,
        *,
        scope: ObjectLabelDomainScope = ObjectLabelDomainScope.PAYLOAD,
        declared_object_count: int | None = None,
        declared_object_ids: DeclaredObjectIds = (),
        declared_object_id_domains: tuple[tuple[int, ...], ...] = (),
    ) -> "ObjectLabelDomain":
        """Return a normalized object-label domain declaration."""
        return cls(
            declared_object_count=declared_object_count,
            declared_object_ids=(
                tuple(declared_object_ids)
                if declared_object_ids is not None
                else ()
            ),
            declared_object_id_domains=declared_object_id_domains,
            scope=scope,
        )

    def explicit_id_domain(self) -> tuple[int, ...] | None:
        """Return this declaration as IDs, or ``None`` if it is undeclared."""
        if self.declared_object_ids:
            return self.declared_object_ids
        if self.declared_object_count is not None:
            return tuple(range(1, self.declared_object_count + 1))
        return None

    def with_runtime_declaration_overrides(
        self,
        *,
        declared_object_count: int | None,
        declared_object_ids: DeclaredObjectIds,
        declared_object_id_domains: tuple[tuple[int, ...], ...],
    ) -> "ObjectLabelDomain":
        """Return this domain with explicit runtime declarations applied."""
        return ObjectLabelDomain(
            declared_object_count=(
                declared_object_count
                if declared_object_count is not None
                else self.declared_object_count
            ),
            declared_object_ids=(
                tuple(declared_object_ids)
                if declared_object_ids is not None
                else self.declared_object_ids
            ),
            declared_object_id_domains=(
                self.declared_object_id_domains or declared_object_id_domains
            ),
            scope=self.scope,
        )

    @classmethod
    def explicit_plane_id_domains(
        cls,
        domains: Iterable["ObjectLabelDomain"],
    ) -> tuple[tuple[int, ...], ...]:
        """Return per-plane declared IDs, preserving undeclared domains as absent."""
        plane_domains: list[tuple[int, ...]] = []
        saw_declared = False
        for domain in domains:
            if domain.declared_object_id_domains:
                saw_declared = True
                plane_domains.extend(domain.declared_object_id_domains)
                continue
            id_domain = domain.explicit_id_domain()
            if id_domain is None:
                if saw_declared:
                    raise ValueError(
                        "Cannot combine declared and undeclared object-label "
                        "plane domains."
                    )
                plane_domains.append(())
                continue
            saw_declared = True
            plane_domains.append(id_domain)
        if not saw_declared:
            return ()
        if any(not domain for domain in plane_domains):
            raise ValueError(
                "Cannot combine declared and undeclared object-label plane domains."
            )
        return tuple(plane_domains)

    def project_slice(self, slice_index: int, slice_count: int) -> "ObjectLabelDomain":
        """Return the object-label domain carried by one PURE_2D slice."""
        normalized_index = int(slice_index)
        normalized_count = int(slice_count)
        if normalized_count <= 0:
            raise ValueError("Object-label slice_count must be positive.")
        if normalized_index < 0 or normalized_index >= normalized_count:
            raise ValueError(
                f"Object-label slice_index {normalized_index} is outside "
                f"slice_count {normalized_count}."
            )
        if not self.declared_object_id_domains:
            return self
        if len(self.declared_object_id_domains) == 1:
            return ObjectLabelDomain.declared(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_ids=self.declared_object_id_domains[0],
            )
        if len(self.declared_object_id_domains) % normalized_count == 0:
            domains_per_slice = len(self.declared_object_id_domains) // normalized_count
            start = normalized_index * domains_per_slice
            return self.project_planes(range(start, start + domains_per_slice))
        if normalized_count % len(self.declared_object_id_domains) == 0:
            return self.project_planes(
                (normalized_index % len(self.declared_object_id_domains),)
            )
        if len(self.declared_object_id_domains) != normalized_count:
            raise ValueError(
                "Plane-scoped object-label domains must match PURE_2D slice "
                f"count: {len(self.declared_object_id_domains)} domains for "
                f"{normalized_count} slices."
            )
        return self.project_planes((normalized_index,))

    def project_planes(self, plane_indices: Iterable[int]) -> "ObjectLabelDomain":
        """Return the object-label domain carried by selected plane indexes."""
        normalized_indices = tuple(int(index) for index in plane_indices)
        if not self.declared_object_id_domains:
            return self
        if not normalized_indices:
            domains: tuple[tuple[int, ...], ...] = ()
        elif len(self.declared_object_id_domains) == 1:
            domains = (self.declared_object_id_domains[0],)
        elif any(
            index < 0 or index >= len(self.declared_object_id_domains)
            for index in normalized_indices
        ):
            raise ValueError(
                "Object-label plane projection index is outside declared "
                f"domain count {len(self.declared_object_id_domains)}: "
                f"{normalized_indices!r}."
            )
        else:
            domains = tuple(
                self.declared_object_id_domains[index] for index in normalized_indices
            )
        if len(domains) == 1:
            return ObjectLabelDomain.declared(
                scope=ObjectLabelDomainScope.PLANE,
                declared_object_ids=domains[0],
            )
        return ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=domains,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeObjectMeasurementQuery(ABC):
    """Store-stable identity for object measurement queries."""

    group_key: str | None
    object_name: str
    feature_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_name",
            self.required_name("object_name", self.object_name),
        )
        object.__setattr__(
            self,
            "feature_name",
            self.required_name("feature_name", self.feature_name),
        )
        if self.group_key is not None:
            object.__setattr__(self, "group_key", str(self.group_key))

    @staticmethod
    def required_name(field_name: str, value: str) -> str:
        """Normalize a required object-measurement query name."""
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"RuntimeObjectMeasurementQuery.{field_name} is required.")
        return normalized


@dataclass(frozen=True, slots=True)
class RuntimeObjectFeatureMeasurementQuery(RuntimeObjectMeasurementQuery):
    """Store-stable identity for object-domain feature vector queries."""

    object_domain: tuple[int, ...]

    def __post_init__(self) -> None:
        RuntimeObjectMeasurementQuery.__post_init__(self)
        object.__setattr__(
            self,
            "object_domain",
            ObjectLabelDomain._normalize_ids(tuple(self.object_domain), "object_domain"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeObjectLabelMeasurementQuery(RuntimeObjectMeasurementQuery):
    """Store-stable identity for label-aligned object measurement queries."""

    axis_id: str
    label_domain: tuple[int, ...]
    image_number: int | None = None

    def __post_init__(self) -> None:
        RuntimeObjectMeasurementQuery.__post_init__(self)
        object.__setattr__(self, "axis_id", str(self.axis_id))
        object.__setattr__(
            self,
            "label_domain",
            ObjectLabelDomain._normalize_ids(tuple(self.label_domain), "label_domain"),
        )
        if self.image_number is not None:
            object.__setattr__(self, "image_number", int(self.image_number))


class ObjectLabelDomainMetadata(ABC, metaclass=AutoRegisterMeta):
    """Nominal provider for object-label ID domain metadata."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True

    @abstractmethod
    def object_label_domain(self) -> ObjectLabelDomain:
        """Return the declared object-label identity domain."""


class ObjectLabelDomainMetadataStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered extractor for nominal object-label domain metadata."""

    value_type: ClassVar[type[object] | tuple[type[object], ...] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def for_value(cls, value: object) -> "ObjectLabelDomainMetadataStrategy":
        strategy = cls.for_nominal_value(value)
        return strategy if strategy is not None else RawObjectLabelDomainMetadataStrategy()

    @abstractmethod
    def object_label_domain(self, value: object) -> ObjectLabelDomain:
        """Return the declared object-label identity domain for ``value``."""


class NominalObjectLabelDomainMetadataStrategy(ObjectLabelDomainMetadataStrategy):
    """Use the domain declared by a nominal object-label domain provider."""

    value_type = ObjectLabelDomainMetadata

    def object_label_domain(self, value: object) -> ObjectLabelDomain:
        if not isinstance(value, ObjectLabelDomainMetadata):
            raise TypeError(
                "NominalObjectLabelDomainMetadataStrategy requires "
                f"ObjectLabelDomainMetadata, got {type(value).__name__}."
            )
        return value.object_label_domain()


class RawObjectLabelDomainMetadataStrategy(ObjectLabelDomainMetadataStrategy):
    """Default domain metadata for values that do not carry object-label identity."""

    def object_label_domain(self, value: object) -> ObjectLabelDomain:
        del value
        return ObjectLabelDomain()


@dataclass(frozen=True, slots=True)
class ObjectLabelMeasurementValues:
    """Numeric measurements bound to explicit object-label identities."""

    object_ids: tuple[int, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        object_ids = ObjectLabelDomain._normalize_ids(
            self.object_ids,
            "ObjectLabelMeasurementValues.object_ids",
        )
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if len(object_ids) != values.size:
            raise ValueError(
                "ObjectLabelMeasurementValues requires one value per object ID, "
                f"got {len(object_ids)} IDs and {values.size} values."
            )
        object.__setattr__(self, "object_ids", object_ids)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_label_indexed_values(
        cls,
        object_ids: Iterable[int],
        values: Any,
    ) -> "ObjectLabelMeasurementValues":
        """Bind dense label-indexed values where index ``label_id - 1``."""
        normalized_ids = ObjectLabelDomain._normalize_ids(
            tuple(object_ids),
            "ObjectLabelMeasurementValues.object_ids",
        )
        source_values = np.asarray(values, dtype=np.float64).reshape(-1)
        bound_values = np.array(
            [
                source_values[object_id - 1]
                if object_id - 1 < source_values.size
                else np.nan
                for object_id in normalized_ids
            ],
            dtype=np.float64,
        )
        return cls(normalized_ids, bound_values)

    @classmethod
    def from_positional_values(
        cls,
        object_ids: Iterable[int],
        values: Any,
    ) -> "ObjectLabelMeasurementValues":
        """Bind values that are already ordered like ``object_ids``."""
        normalized_ids = ObjectLabelDomain._normalize_ids(
            tuple(object_ids),
            "ObjectLabelMeasurementValues.object_ids",
        )
        source_values = np.asarray(values, dtype=np.float64).reshape(-1)
        bound_values = np.full(len(normalized_ids), np.nan, dtype=np.float64)
        copied = min(source_values.size, bound_values.size)
        if copied:
            bound_values[:copied] = source_values[:copied]
        return cls(normalized_ids, bound_values)

    @classmethod
    def from_value_mapping(
        cls,
        object_ids: Iterable[int],
        values_by_object_id: Mapping[int, float],
    ) -> "ObjectLabelMeasurementValues":
        """Bind sparse object-id keyed values to an explicit object domain."""
        normalized_ids = ObjectLabelDomain._normalize_ids(
            tuple(object_ids),
            "ObjectLabelMeasurementValues.object_ids",
        )
        return cls(
            normalized_ids,
            np.array(
                [
                    float(values_by_object_id.get(object_id, np.nan))
                    for object_id in normalized_ids
                ],
                dtype=np.float64,
            ),
        )

    def __len__(self) -> int:
        return len(self.object_ids)

    def ids_within_limits(
        self,
        *,
        min_value: float | None,
        max_value: float | None,
        use_minimum: bool,
        use_maximum: bool,
    ) -> tuple[int, ...]:
        """Return object IDs whose finite values satisfy configured bounds."""
        if not self.object_ids:
            return ()
        hits = np.isfinite(self.values)
        if use_minimum and min_value is not None:
            hits[self.values < min_value] = False
        if use_maximum and max_value is not None:
            hits[self.values > max_value] = False
        return tuple(
            object_id
            for object_id, hit in zip(self.object_ids, hits, strict=True)
            if bool(hit)
        )

    def extremum_id(self, *, keep_max: bool) -> int | None:
        """Return the object ID with the finite minimum or maximum value."""
        if not self.object_ids:
            return None
        finite_indexes = np.flatnonzero(np.isfinite(self.values))
        if finite_indexes.size == 0:
            return None
        finite_values = self.values[finite_indexes]
        selected_index = finite_indexes[
            int(np.argmax(finite_values) if keep_max else np.argmin(finite_values))
        ]
        return self.object_ids[int(selected_index)]

    def dense_label_indexed(
        self,
        *,
        max_label: int | None = None,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """Return values as a dense ``label_id - 1`` indexed vector."""
        largest_id = max(self.object_ids, default=0)
        output_size = max(largest_id, int(max_label or 0))
        output = np.full(output_size, fill_value, dtype=np.float64)
        for object_id, value in zip(self.object_ids, self.values, strict=True):
            output[object_id - 1] = value
        return output


class ObjectLabelIdDomainStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered extractor for materially present positive object-label IDs."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def for_value(cls, labels: Any) -> "ObjectLabelIdDomainStrategy":
        strategy = cls.for_nominal_value(labels)
        return strategy if strategy is not None else RawObjectLabelIdDomainStrategy()

    @abstractmethod
    def present_ids(self, labels: Any) -> tuple[int, ...]:
        """Return positive object IDs materially present in the label payload."""

    def max_present_id(self, labels: Any) -> int:
        """Return the largest materially present positive object ID."""
        ids = self.present_ids(labels)
        return max(ids) if ids else 0

    @staticmethod
    def positive_ids_from_array(labels: Any) -> tuple[int, ...]:
        """Return positive IDs from one numeric dense object-label array."""
        import numpy as np

        label_array = np.asarray(labels)
        if not label_array.size or not (
            np.issubdtype(label_array.dtype, np.number)
            or np.issubdtype(label_array.dtype, np.bool_)
        ):
            return ()
        return tuple(
            int(object_id)
            for object_id in np.unique(label_array)
            if object_id > 0
        )


class DenseArrayObjectLabelIdDomainStrategy(ObjectLabelIdDomainStrategy):
    """Extract present object IDs from dense NumPy label arrays."""

    value_type = np.ndarray

    def present_ids(self, labels: Any) -> tuple[int, ...]:
        return self.positive_ids_from_array(labels)


class RawObjectLabelIdDomainStrategy(ObjectLabelIdDomainStrategy):
    """Compatibility extractor for legacy array-like label payloads."""

    def present_ids(self, labels: Any) -> tuple[int, ...]:
        return self.positive_ids_from_array(labels)


class ObjectLabelPlaneDomainStrategy(
    EnumKeyedStrategyMixin[ObjectLabelDomainScope],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Projection strategy from object-label domain metadata to measurement planes."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "scope"
    scope: ClassVar[ObjectLabelDomainScope]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def plane_domains(
        self,
        labels: Any,
        *,
        declared_object_count: int | None,
        declared_object_ids: DeclaredObjectIds,
        declared_object_id_domains: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Return the object-id domain attached to each dense measurement plane."""

    def identity_domains(
        self,
        labels: Any,
        *,
        declared_object_count: int | None,
        declared_object_ids: DeclaredObjectIds,
        declared_object_id_domains: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Return object-id domains for identity rows represented by the payload."""
        return self.plane_domains(
            labels,
            declared_object_count=declared_object_count,
            declared_object_ids=declared_object_ids,
            declared_object_id_domains=declared_object_id_domains,
        )


class PayloadObjectLabelPlaneDomainStrategy(ObjectLabelPlaneDomainStrategy):
    """Payload-scope declarations apply equally to every dense label plane."""

    scope = ObjectLabelDomainScope.PAYLOAD

    def plane_domains(
        self,
        labels: Any,
        *,
        declared_object_count: int | None,
        declared_object_ids: DeclaredObjectIds,
        declared_object_id_domains: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        import numpy as np

        if declared_object_id_domains:
            return declared_object_id_domains
        label_array = np.asarray(labels)
        plane_count = 1 if label_array.ndim <= 2 else label_array.shape[0]
        domain = dense_object_label_id_domain(
            labels,
            declared_object_count=declared_object_count,
            declared_object_ids=declared_object_ids,
        )
        return (domain,) * plane_count

    def identity_domains(
        self,
        labels: Any,
        *,
        declared_object_count: int | None,
        declared_object_ids: DeclaredObjectIds,
        declared_object_id_domains: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        if declared_object_id_domains:
            return declared_object_id_domains
        return (
            dense_object_label_id_domain(
                labels,
                declared_object_count=declared_object_count,
                declared_object_ids=declared_object_ids,
            ),
        )


class PlaneObjectLabelPlaneDomainStrategy(ObjectLabelPlaneDomainStrategy):
    """Plane-scope declarations apply to one 2D plane or are re-derived per stack plane."""

    scope = ObjectLabelDomainScope.PLANE

    def plane_domains(
        self,
        labels: Any,
        *,
        declared_object_count: int | None,
        declared_object_ids: DeclaredObjectIds,
        declared_object_id_domains: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        import numpy as np

        label_array = np.asarray(labels)
        if declared_object_id_domains:
            plane_count = 1 if label_array.ndim <= 2 else label_array.shape[0]
            if plane_count == 1 and len(declared_object_id_domains) != 1:
                return (dense_object_label_id_domain(labels),)
            if len(declared_object_id_domains) != plane_count:
                raise ValueError(
                    "Plane-scoped object-label domains must match dense label "
                    f"plane count: {len(declared_object_id_domains)} domains for "
                    f"{plane_count} planes."
                )
            return declared_object_id_domains
        if declared_object_count is not None or declared_object_ids:
            plane_count = 1 if label_array.ndim <= 2 else label_array.shape[0]
            domain = dense_object_label_id_domain(
                labels,
                declared_object_count=declared_object_count,
                declared_object_ids=declared_object_ids,
            )
            return (domain,) * plane_count
        if label_array.ndim <= 2:
            return (dense_object_label_id_domain(labels),)
        return tuple(dense_object_label_id_domain(plane) for plane in label_array)


class SpatialGridOrdering(str, Enum):
    """Primary axis used when numbering positions in a spatial grid."""

    BY_ROWS = "rows"
    BY_COLUMNS = "columns"


class SpatialGridOrigin(str, Enum):
    """Corner used as the numbering origin for a spatial grid."""

    def __new__(cls, value: str, reverses_rows: bool, reverses_columns: bool):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._reverses_rows = reverses_rows
        obj._reverses_columns = reverses_columns
        return obj

    TOP_LEFT = ("top_left", False, False)
    BOTTOM_LEFT = ("bottom_left", True, False)
    TOP_RIGHT = ("top_right", False, True)
    BOTTOM_RIGHT = ("bottom_right", True, True)
    reverses_rows = AliasProperty[bool]("_reverses_rows")
    reverses_columns = AliasProperty[bool]("_reverses_columns")


class MeasurementScope(str, Enum):
    """Semantic entity scope for measurement rows."""

    def __new__(
        cls,
        value: str,
        requires_subject_name: bool = False,
        projects_runtime_slices: bool = False,
    ):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._requires_subject_name = requires_subject_name
        obj._projects_runtime_slices = projects_runtime_slices
        return obj

    ARTIFACT = ("artifact", False)
    IMAGE = ("image", True, True)
    OBJECT = ("object", True, True)
    RELATIONSHIP = ("relationship", True)
    EXPERIMENT = ("experiment", False)
    requires_subject_name = AliasProperty[bool]("_requires_subject_name")
    projects_runtime_slices = AliasProperty[bool]("_projects_runtime_slices")


class RuntimeMeasurementFeature(str, Enum):
    """Base for generated runtime measurement feature enums."""

    feature_name = AliasProperty[str]("value")


for _measurement_feature_spec in (
    GeneratedEnumClassSpec(
        class_name="PairMeasurementFeature",
        base_type=RuntimeMeasurementFeature,
        members={
            "CORRELATION": "correlation",
            "REGRESSION_SLOPE": "slope",
            "OVERLAP": "overlap",
            "COSTES_MANDERS": "costes",
            "MANDERS": "manders",
            "RANK_WEIGHTED_COLOCALIZATION": "rwc",
            "OVERLAP_K": "k",
        },
    ),
):
    _measurement_feature_spec.declare_in(globals())


class ObjectMeasurementFeatureRole(str, Enum):
    """Nominal semantic roles for generic object measurement features."""

    COUNT = "count"
    IDENTIFIER = "identifier"
    MEASURED_OBJECT_ANCHOR = "measured_object_anchor"
    LOCATION = "location"
    INTENSITY = "intensity"
    CALCULATED = "calculated"
    SHAPE_DESCRIPTOR = "shape_descriptor"
    ZERNIKE_DESCRIPTOR = "zernike_descriptor"


class MeasurementStatistic(str, Enum):
    """Canonical runtime measurement statistic labels."""

    VALUE = "value"
    COUNT = "count"
    MEAN = "mean"


for _measurement_feature_spec in (
    GeneratedEnumClassSpec(
        class_name="ObjectCoreMeasurementFeature",
        base_type=RuntimeMeasurementFeature,
        members={
            "OBJECT_COUNT": "object_count",
            "OBJECT_NUMBER": "object_number",
            "CENTER_X": "center_x",
            "CENTER_Y": "center_y",
            "CENTER_Z": "center_z",
        },
    ),
):
    _measurement_feature_spec.declare_in(globals())


@dataclass(frozen=True, slots=True)
class ObjectLocationCoordinateValues:
    """Dense label-indexed values and missing-row policy for one coordinate."""

    values: Any
    include_missing: bool


class ObjectLocationCoordinateProjectionStrategy(
    EnumKeyedStrategyMixin[ObjectCoreMeasurementFeature],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project dense-label coordinates for one nominal object-location feature."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "coordinate_feature"

    coordinate_feature: ClassVar[ObjectCoreMeasurementFeature]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def coordinate_values(
        self,
        axis_centers: Sequence[Any],
        counts: Any,
    ) -> ObjectLocationCoordinateValues:
        """Return dense label-indexed coordinate values for this feature."""

    @staticmethod
    def missing_for_absent_labels(values: Any, counts: Any) -> Any:
        import numpy as np

        result = np.asarray(values, dtype=float).copy()
        result[counts == 0] = np.nan
        return result


class AxisBackedObjectLocationCoordinateProjectionStrategy(
    ObjectLocationCoordinateProjectionStrategy
):
    """Base for coordinates backed by a concrete dense-array axis when present."""

    required_ndim: ClassVar[int]
    axis_offset: ClassVar[int]

    def coordinate_values(
        self,
        axis_centers: Sequence[Any],
        counts: Any,
    ) -> ObjectLocationCoordinateValues:
        import numpy as np

        if len(axis_centers) >= type(self).required_ndim:
            return ObjectLocationCoordinateValues(
                axis_centers[type(self).axis_offset],
                include_missing=False,
            )
        return ObjectLocationCoordinateValues(
            self.missing_for_absent_labels(np.zeros(len(counts)), counts),
            include_missing=False,
        )


for _coordinate_projection_spec in (
    GeneratedLeafClassSpec(
        class_name="CenterXObjectLocationCoordinateProjectionStrategy",
        base_type=AxisBackedObjectLocationCoordinateProjectionStrategy,
        attributes={
            "coordinate_feature": ObjectCoreMeasurementFeature.CENTER_X,
            "required_ndim": 1,
            "axis_offset": -1,
        },
    ),
    GeneratedLeafClassSpec(
        class_name="CenterYObjectLocationCoordinateProjectionStrategy",
        base_type=AxisBackedObjectLocationCoordinateProjectionStrategy,
        attributes={
            "coordinate_feature": ObjectCoreMeasurementFeature.CENTER_Y,
            "required_ndim": 2,
            "axis_offset": -2,
        },
    ),
    GeneratedLeafClassSpec(
        class_name="CenterZObjectLocationCoordinateProjectionStrategy",
        base_type=AxisBackedObjectLocationCoordinateProjectionStrategy,
        attributes={
            "coordinate_feature": ObjectCoreMeasurementFeature.CENTER_Z,
            "required_ndim": 3,
            "axis_offset": -3,
        },
    ),
):
    _coordinate_projection_spec.declare_in(globals())


def object_location_coordinate_arrays(
    axis_centers: Sequence[Any],
    counts: Any,
) -> tuple[tuple[str, ObjectLocationCoordinateValues], ...]:
    """Return nominal object-location coordinate arrays in core feature order."""
    return tuple(
        (
            feature.value,
            ObjectLocationCoordinateProjectionStrategy.for_enum_member(
                feature,
            ).coordinate_values(axis_centers, counts),
        )
        for feature in (
            ObjectCoreMeasurementFeature.CENTER_X,
            ObjectCoreMeasurementFeature.CENTER_Y,
            ObjectCoreMeasurementFeature.CENTER_Z,
        )
    )


class ObjectLocationMeasurementFeature(str, Enum):
    """Canonical CellProfiler-style object location feature names."""

    CENTER_X = "Location_Center_X"
    CENTER_Y = "Location_Center_Y"
    CENTER_Z = "Location_Center_Z"

    @property
    def core_feature(self) -> ObjectCoreMeasurementFeature:
        """Return the normalized core feature represented by this location feature."""
        return ObjectCoreMeasurementFeature[self.name]


for _measurement_feature_spec in (
    GeneratedEnumClassSpec(
        class_name="ObjectIntensityMeasurementFeature",
        base_type=RuntimeMeasurementFeature,
        members={
            "INTEGRATED_INTENSITY": "IntegratedIntensity",
            "MEAN_INTENSITY": "MeanIntensity",
            "STD_INTENSITY": "StdIntensity",
            "MIN_INTENSITY": "MinIntensity",
            "MAX_INTENSITY": "MaxIntensity",
            "INTEGRATED_INTENSITY_EDGE": "IntegratedIntensityEdge",
            "MEAN_INTENSITY_EDGE": "MeanIntensityEdge",
            "STD_INTENSITY_EDGE": "StdIntensityEdge",
            "MIN_INTENSITY_EDGE": "MinIntensityEdge",
            "MAX_INTENSITY_EDGE": "MaxIntensityEdge",
            "MASS_DISPLACEMENT": "MassDisplacement",
            "LOWER_QUARTILE_INTENSITY": "LowerQuartileIntensity",
            "MEDIAN_INTENSITY": "MedianIntensity",
            "MAD_INTENSITY": "MADIntensity",
            "UPPER_QUARTILE_INTENSITY": "UpperQuartileIntensity",
            "CENTER_MASS_INTENSITY_X": "CenterMassIntensity_X",
            "CENTER_MASS_INTENSITY_Y": "CenterMassIntensity_Y",
            "CENTER_MASS_INTENSITY_Z": "CenterMassIntensity_Z",
            "MAX_INTENSITY_X": "MaxIntensity_X",
            "MAX_INTENSITY_Y": "MaxIntensity_Y",
            "MAX_INTENSITY_Z": "MaxIntensity_Z",
        },
    ),
    GeneratedEnumClassSpec(
        class_name="ImageAreaOccupiedMeasurementFeature",
        base_type=RuntimeMeasurementFeature,
        members={
            "AREA_OCCUPIED": "AreaOccupied",
            "PERIMETER": "Perimeter",
            "TOTAL_AREA": "TotalArea",
        },
    ),
    GeneratedEnumClassSpec(
        class_name="ObjectShapeMeasurementFeature",
        base_type=RuntimeMeasurementFeature,
        members={
            "AREA": "Area",
            "PERIMETER": "Perimeter",
            "VOLUME": "Volume",
            "SURFACE_AREA": "SurfaceArea",
            "ECCENTRICITY": "Eccentricity",
            "SOLIDITY": "Solidity",
            "CONVEX_AREA": "ConvexArea",
            "EXTENT": "Extent",
            "CENTER_X": "Center_X",
            "CENTER_Y": "Center_Y",
            "CENTER_Z": "Center_Z",
            "BOUNDING_BOX_AREA": "BoundingBoxArea",
            "BOUNDING_BOX_VOLUME": "BoundingBoxVolume",
            "BOUNDING_BOX_MINIMUM_X": "BoundingBoxMinimum_X",
            "BOUNDING_BOX_MAXIMUM_X": "BoundingBoxMaximum_X",
            "BOUNDING_BOX_MINIMUM_Y": "BoundingBoxMinimum_Y",
            "BOUNDING_BOX_MAXIMUM_Y": "BoundingBoxMaximum_Y",
            "BOUNDING_BOX_MINIMUM_Z": "BoundingBoxMinimum_Z",
            "BOUNDING_BOX_MAXIMUM_Z": "BoundingBoxMaximum_Z",
            "EULER_NUMBER": "EulerNumber",
            "FORM_FACTOR": "FormFactor",
            "MAJOR_AXIS_LENGTH": "MajorAxisLength",
            "MINOR_AXIS_LENGTH": "MinorAxisLength",
            "ORIENTATION": "Orientation",
            "COMPACTNESS": "Compactness",
            "MAXIMUM_RADIUS": "MaximumRadius",
            "MEDIAN_RADIUS": "MedianRadius",
            "MEAN_RADIUS": "MeanRadius",
            "MIN_FERET_DIAMETER": "MinFeretDiameter",
            "MAX_FERET_DIAMETER": "MaxFeretDiameter",
            "EQUIVALENT_DIAMETER": "EquivalentDiameter",
            "SPATIAL_MOMENT": "SpatialMoment",
            "CENTRAL_MOMENT": "CentralMoment",
            "NORMALIZED_MOMENT": "NormalizedMoment",
            "HU_MOMENT": "HuMoment",
            "INERTIA_TENSOR": "InertiaTensor",
            "INERTIA_TENSOR_EIGENVALUES": "InertiaTensorEigenvalues",
            "ZERNIKE": "Zernike",
        },
    ),
):
    _measurement_feature_spec.declare_in(globals())


class ObjectZernikeDescriptorFeature(str, Enum):
    """Canonical object Zernike descriptor families."""

    SHAPE = "zernike"
    INTENSITY_MAGNITUDE = "zernike_magnitude"
    INTENSITY_PHASE = "zernike_phase"


for _measurement_feature_spec in (
    GeneratedEnumClassSpec(
        class_name="ObjectIntensityDistributionMeasurementFeature",
        base_type=RuntimeMeasurementFeature,
        members={
            "FRACTION_AT_DISTANCE": "FracAtD",
            "MEAN_FRACTION": "MeanFrac",
            "RADIAL_CV": "RadialCV",
        },
    ),
):
    _measurement_feature_spec.declare_in(globals())


@dataclass(frozen=True, slots=True)
class ObjectMeasurementValueRow:
    """Nominal long-form object measurement row."""

    object_label: int
    feature_name: str
    result_value: float


class ObjectIntensityZernikeFeatureNameStrategy(
    EnumKeyedStrategyMixin[ObjectZernikeDescriptorFeature],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Render intensity Zernike feature families with nominal dispatch."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "feature"

    feature: ClassVar[ObjectZernikeDescriptorFeature]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def family_name(self) -> str:
        """Return the external feature-family name for this descriptor."""

    def feature_name(self, *, degree: int, repetition: int) -> str:
        """Return CP-compatible long-form intensity Zernike feature identity."""
        return (
            f"IntensityDistribution_{self.family_name()}_"
            f"{int(degree)}_{int(repetition)}"
        )


class ObjectIntensityZernikeMagnitudeFeatureNameStrategy(
    ObjectIntensityZernikeFeatureNameStrategy
):
    """Render intensity Zernike magnitude rows."""

    feature = ObjectZernikeDescriptorFeature.INTENSITY_MAGNITUDE

    def family_name(self) -> str:
        return "ZernikeMagnitude"


class ObjectIntensityZernikePhaseFeatureNameStrategy(
    ObjectIntensityZernikeFeatureNameStrategy
):
    """Render intensity Zernike phase rows."""

    feature = ObjectZernikeDescriptorFeature.INTENSITY_PHASE

    def family_name(self) -> str:
        return "ZernikePhase"


@lru_cache(maxsize=None)
def indexed_object_intensity_distribution_feature_name(
    feature: ObjectIntensityDistributionMeasurementFeature | str,
    *,
    bin_index: int,
    bin_count: int,
) -> str:
    """Return CP-compatible long-form radial distribution feature identity."""
    feature = coerce_enum(
        ObjectIntensityDistributionMeasurementFeature,
        feature,
        "indexed_object_intensity_distribution_feature_name.feature",
    )
    return f"IntensityDistribution_{feature.value}_{int(bin_index)}of{int(bin_count)}"


@dataclass(frozen=True, slots=True)
class ObjectIntensityDistributionMeasurementRows:
    """Materialize long-form object radial intensity-distribution rows."""

    radial_arrays: Any
    object_ids: Sequence[int]
    bin_count: int

    def rows(self) -> list[ObjectMeasurementValueRow]:
        object_ids = tuple(int(object_id) for object_id in self.object_ids)
        rows = cast(
            list[ObjectMeasurementValueRow],
            [None] * (len(object_ids) * int(self.radial_arrays.n_bins) * 3),
        )
        row_index = 0
        row_type = ObjectMeasurementValueRow
        fraction_at_distance = self.radial_arrays.fraction_at_distance
        mean_pixel_fraction = self.radial_arrays.mean_pixel_fraction
        radial_cv_by_bin = self.radial_arrays.radial_cv_by_bin
        object_has_pixels_by_index = self.radial_arrays.object_has_pixels
        for bin_idx in range(self.radial_arrays.n_bins):
            bin_index = bin_idx + 1
            fraction_at_distance_feature = (
                indexed_object_intensity_distribution_feature_name(
                    ObjectIntensityDistributionMeasurementFeature.FRACTION_AT_DISTANCE,
                    bin_index=bin_index,
                    bin_count=self.bin_count,
                )
            )
            mean_fraction_feature = indexed_object_intensity_distribution_feature_name(
                ObjectIntensityDistributionMeasurementFeature.MEAN_FRACTION,
                bin_index=bin_index,
                bin_count=self.bin_count,
            )
            radial_cv_feature = indexed_object_intensity_distribution_feature_name(
                ObjectIntensityDistributionMeasurementFeature.RADIAL_CV,
                bin_index=bin_index,
                bin_count=self.bin_count,
            )
            radial_cv = radial_cv_by_bin[bin_idx]
            for object_label in object_ids:
                obj_idx = object_label - 1
                object_has_pixels = bool(object_has_pixels_by_index[obj_idx])
                frac_at_d = (
                    float(fraction_at_distance[obj_idx, bin_idx])
                    if object_has_pixels
                    else np.nan
                )
                mean_frac = (
                    float(mean_pixel_fraction[obj_idx, bin_idx])
                    if object_has_pixels
                    else np.nan
                )
                rows[row_index] = row_type(
                    object_label=object_label,
                    feature_name=fraction_at_distance_feature,
                    result_value=frac_at_d,
                )
                row_index += 1
                rows[row_index] = row_type(
                    object_label=object_label,
                    feature_name=mean_fraction_feature,
                    result_value=mean_frac,
                )
                row_index += 1
                rows[row_index] = row_type(
                    object_label=object_label,
                    feature_name=radial_cv_feature,
                    result_value=float(radial_cv[obj_idx]),
                )
                row_index += 1
        return rows

@lru_cache(maxsize=None)
def indexed_object_intensity_zernike_feature_name(
    feature: ObjectZernikeDescriptorFeature | str,
    *,
    degree: int,
    repetition: int,
) -> str:
    """Return CP-compatible long-form intensity Zernike feature identity."""
    feature = coerce_enum(
        ObjectZernikeDescriptorFeature,
        feature,
        "indexed_object_intensity_zernike_feature_name.feature",
    )
    return ObjectIntensityZernikeFeatureNameStrategy.for_enum_member(
        feature
    ).feature_name(degree=degree, repetition=repetition)


@dataclass(frozen=True, slots=True)
class ObjectIntensityZernikeMeasurementRows:
    """Materialize long-form object intensity-Zernike rows from backend arrays."""

    object_ids: Sequence[int]
    zernike_indexes: Sequence[tuple[int, int]]
    magnitudes: Any
    phases: Any
    include_phase: bool

    def rows(self) -> list[ObjectMeasurementValueRow]:
        """Return rows in canonical magnitude-then-phase feature order."""
        object_ids = np.asarray(self.object_ids, dtype=np.int32)
        zernike_indexes = tuple((int(n), int(m)) for n, m in self.zernike_indexes)
        if object_ids.size == 0 or len(zernike_indexes) == 0:
            return []

        magnitude_values = np.asarray(self.magnitudes, dtype=np.float64)
        phase_values = np.asarray(self.phases, dtype=np.float64)
        descriptor_count = 2 if self.include_phase else 1
        rows = cast(
            list[ObjectMeasurementValueRow],
            [None] * (object_ids.size * len(zernike_indexes) * descriptor_count),
        )
        row_index = 0
        row_type = ObjectMeasurementValueRow
        for index, (degree, repetition) in enumerate(zernike_indexes):
            magnitude_feature = indexed_object_intensity_zernike_feature_name(
                ObjectZernikeDescriptorFeature.INTENSITY_MAGNITUDE,
                degree=degree,
                repetition=repetition,
            )
            magnitude_column = magnitude_values[:, index]
            for object_label, value in zip(object_ids, magnitude_column, strict=True):
                rows[row_index] = row_type(
                    object_label=int(object_label),
                    feature_name=magnitude_feature,
                    result_value=float(value),
                )
                row_index += 1
            if self.include_phase:
                phase_feature = indexed_object_intensity_zernike_feature_name(
                    ObjectZernikeDescriptorFeature.INTENSITY_PHASE,
                    degree=degree,
                    repetition=repetition,
                )
                phase_column = phase_values[:, index]
                for object_label, value in zip(object_ids, phase_column, strict=True):
                    rows[row_index] = row_type(
                        object_label=int(object_label),
                        feature_name=phase_feature,
                        result_value=float(value),
                    )
                    row_index += 1
        return rows


class MeasurementRowAxisField(str, Enum):
    """Canonical row-axis fields for long/tall measurement tables."""

    IMAGE_NUMBER = "image_number"
    SLICE_INDEX = "slice_index"
    FEATURE_NAME = "feature_name"
    MEASUREMENT_NAME = "measurement_name"
    OUTPUT_NAME = "output_name"
    OBJECT_NAME = "object_name"
    OBJECT_LABEL = "object_label"
    OBJECT_NUMBER = "object_number"
    OBJECT_ID = "object_id"
    LABEL = "label"
    SOURCE_IMAGE_NAME = "source_image_name"
    BIN_INDEX = "bin_index"
    BIN_COUNT = "bin_count"
    SCALE = "scale"
    DIRECTION = "direction"
    GRAY_LEVELS = "gray_levels"
    ZERNIKE_N = "n"
    ZERNIKE_M = "m"


class MeasurementRowValueField(str, Enum):
    """Canonical scalar value fields for long/tall measurement rows."""

    RESULT_VALUE = "result_value"
    MEASUREMENT_VALUE = "measurement_value"
    VALUE = "value"
    MEAN_VALUE = "mean_value"


class ObjectFeatureArrayDomain(str, Enum):
    """How a feature array indexes values for an object-feature table."""

    MEASURED_OBJECT_ID = "measured_object_id"
    LABEL_ID = "label_id"
    ROW_ORDINAL = "row_ordinal"


class ObjectFeatureMissingValue(str, Enum):
    """How an object-feature table represents unmeasured feature values."""

    NAN = "nan"
    ZERO = "zero"


def zernike_shape_feature_names(*, max_order: int) -> tuple[str, ...]:
    """Return canonical shape-Zernike feature names for a maximum order."""
    names: list[str] = []
    for n in range(max_order + 1):
        for m in range(n % 2, n + 1, 2):
            names.append(f"{ObjectShapeMeasurementFeature.ZERNIKE.value}_{n}_{m}")
    return tuple(names)


@dataclass(frozen=True, slots=True)
class ObjectFeatureArrayDomainContext:
    """Feature-array indexing inputs for one object-feature table."""

    object_id: int
    values: np.ndarray
    measured_object_ids: tuple[int, ...]
    object_domain: tuple[int, ...]

    @property
    def value_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def measured_object_count(self) -> int:
        return len(self.measured_object_ids)

    @property
    def measured_object_max(self) -> int:
        return max(self.measured_object_ids, default=0)


class ObjectFeatureArrayDomainStrategy(
    EnumKeyedStrategyMixin[ObjectFeatureArrayDomain],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project feature arrays according to their declared object domain."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "domain"

    domain: ClassVar[ObjectFeatureArrayDomain]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def value_index(self, context: ObjectFeatureArrayDomainContext) -> int | None:
        """Return the value index for ``context.object_id``."""

    def value_indexes(
        self,
        context: ObjectFeatureArrayDomainContext,
    ) -> Mapping[int, int]:
        """Return object-id to value-index mappings for a feature array."""
        indexes: dict[int, int] = {}
        for object_id in context.object_domain:
            value_index = self.value_index(replace(context, object_id=object_id))
            if value_index is not None:
                indexes[object_id] = value_index
        return indexes

    @abstractmethod
    def accepts(self, context: ObjectFeatureArrayDomainContext) -> bool:
        """Return whether the feature array shape is valid for this domain."""


class MeasuredObjectFeatureArrayDomainStrategy(ObjectFeatureArrayDomainStrategy):
    """Feature arrays indexed by compact measured-object IDs."""

    domain = ObjectFeatureArrayDomain.MEASURED_OBJECT_ID

    def value_index(self, context: ObjectFeatureArrayDomainContext) -> int | None:
        try:
            value_index = context.measured_object_ids.index(context.object_id)
        except ValueError:
            return None
        return value_index if value_index < context.value_count else None

    def accepts(self, context: ObjectFeatureArrayDomainContext) -> bool:
        return context.value_count == context.measured_object_count

    def value_indexes(
        self,
        context: ObjectFeatureArrayDomainContext,
    ) -> Mapping[int, int]:
        return {
            object_id: index
            for index, object_id in enumerate(context.measured_object_ids)
            if index < context.value_count
        }


class LabelIdFeatureArrayDomainStrategy(ObjectFeatureArrayDomainStrategy):
    """Feature arrays indexed by dense label ID minus one."""

    domain = ObjectFeatureArrayDomain.LABEL_ID

    def value_index(self, context: ObjectFeatureArrayDomainContext) -> int | None:
        value_index = context.object_id - 1
        return value_index if 0 <= value_index < context.value_count else None

    def accepts(self, context: ObjectFeatureArrayDomainContext) -> bool:
        return context.value_count >= context.measured_object_max

    def value_indexes(
        self,
        context: ObjectFeatureArrayDomainContext,
    ) -> Mapping[int, int]:
        return {
            object_id: object_id - 1
            for object_id in context.object_domain
            if 0 <= object_id - 1 < context.value_count
        }


class RowOrdinalFeatureArrayDomainStrategy(ObjectFeatureArrayDomainStrategy):
    """Feature arrays indexed by the emitted row ordinal."""

    domain = ObjectFeatureArrayDomain.ROW_ORDINAL

    def value_index(self, context: ObjectFeatureArrayDomainContext) -> int | None:
        try:
            value_index = context.object_domain.index(context.object_id)
        except ValueError:
            return None
        return value_index if value_index < context.value_count else None

    def accepts(self, context: ObjectFeatureArrayDomainContext) -> bool:
        return context.value_count <= len(context.object_domain)

    def value_indexes(
        self,
        context: ObjectFeatureArrayDomainContext,
    ) -> Mapping[int, int]:
        return {
            object_id: index
            for index, object_id in enumerate(context.object_domain)
            if index < context.value_count
        }


class ObjectFeatureMissingValueStrategy(
    EnumKeyedStrategyMixin[ObjectFeatureMissingValue],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Emit declared missing values for unmeasured object-feature rows."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "missing_value"

    missing_value: ClassVar[ObjectFeatureMissingValue]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def value(self) -> float:
        """Return the scalar missing value represented by this policy."""


class NanObjectFeatureMissingValueStrategy(ObjectFeatureMissingValueStrategy):
    """Represent missing object features as NaN."""

    missing_value = ObjectFeatureMissingValue.NAN

    def value(self) -> float:
        return float(np.nan)


class ZeroObjectFeatureMissingValueStrategy(ObjectFeatureMissingValueStrategy):
    """Represent missing object features as numeric zero."""

    missing_value = ObjectFeatureMissingValue.ZERO

    def value(self) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class ObjectFeatureValueTable(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Wide object-feature values aligned onto a declared object-id domain."""

    __registry_key__ = "table_label"
    __skip_if_no_key__ = True

    value_type: ClassVar[type[object] | tuple[type[object], ...] | None] = None
    value_type_label: ClassVar[str | None] = None
    table_label: ClassVar[str | None] = None
    feature_values: Mapping[str, Any]
    measured_object_ids: tuple[int, ...]
    object_domain: tuple[int, ...]
    object_id_field: str = MeasurementRowAxisField.OBJECT_LABEL.value
    slice_index_field: str = MeasurementRowAxisField.SLICE_INDEX.value
    slice_index: int = 0
    feature_array_domains: ClassVar[Mapping[str, ObjectFeatureArrayDomain]] = {}
    feature_missing_values: ClassVar[Mapping[str, ObjectFeatureMissingValue]] = {}

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "measured_object_ids",
            ObjectLabelDomain._normalize_ids(
                self.measured_object_ids,
                "ObjectFeatureValueTable.measured_object_ids",
            ),
        )
        object.__setattr__(
            self,
            "object_domain",
            ObjectLabelDomain._normalize_ids(
                self.object_domain,
                "ObjectFeatureValueTable.object_domain",
            ),
        )
        object.__setattr__(self, "slice_index", int(self.slice_index))

    @classmethod
    def from_feature_arrays(
        cls,
        feature_values: Mapping[str, Any],
        measured_object_ids: Iterable[int],
        object_domain: Iterable[int],
        **kwargs: Any,
    ) -> "ObjectFeatureValueTable":
        """Build a declared-domain table from measured feature arrays."""
        return cls(
            feature_values=feature_values,
            measured_object_ids=tuple(int(object_id) for object_id in measured_object_ids),
            object_domain=tuple(int(object_id) for object_id in object_domain),
            **kwargs,
        )

    def rows(self) -> list[dict[str, float | int]]:
        """Return wide rows ordered by the declared object domain."""
        feature_items = tuple(
            (
                feature_name,
                np.asarray(values),
                self.python_feature_values(values),
                ObjectFeatureMissingValueStrategy.for_enum_member(
                    self.feature_missing_value(feature_name)
                ).value(),
                self.feature_value_indexes(feature_name, np.asarray(values)),
            )
            for feature_name, values in self.feature_values.items()
        )
        rows: list[dict[str, float | int]] = []
        for object_id in self.object_domain:
            row: dict[str, float | int] = {
                self.slice_index_field: self.slice_index,
                self.object_id_field: object_id,
            }
            for (
                feature_name,
                values,
                python_values,
                missing_value,
                value_indexes,
            ) in feature_items:
                if values.ndim == 0:
                    row[feature_name] = python_values
                    continue
                value_index = value_indexes.get(object_id)
                row[feature_name] = (
                    missing_value if value_index is None else python_values[value_index]
                )
            self.complete_row(row)
            rows.append(row)
        return rows

    def feature_value_indexes(
        self,
        feature_name: str,
        values: np.ndarray,
    ) -> Mapping[int, int]:
        """Return object-id to feature-value indexes for one feature array."""
        if values.ndim == 0:
            return {}
        self.validate_feature_value_domain(feature_name, values)
        return ObjectFeatureArrayDomainStrategy.for_enum_member(
            self.feature_array_domain(feature_name)
        ).value_indexes(
            ObjectFeatureArrayDomainContext(
                object_id=0,
                values=values,
                measured_object_ids=self.measured_object_ids,
                object_domain=self.object_domain,
            )
        )

    def feature_value_index(
        self,
        feature_name: str,
        object_id: int,
        *,
        values: np.ndarray,
    ) -> int | None:
        """Return the feature-array index for one declared object ID."""
        return ObjectFeatureArrayDomainStrategy.for_enum_member(
            self.feature_array_domain(feature_name)
        ).value_index(
            ObjectFeatureArrayDomainContext(
                object_id=object_id,
                values=values,
                measured_object_ids=self.measured_object_ids,
                object_domain=self.object_domain,
            )
        )

    def feature_array_domain(self, feature_name: str) -> ObjectFeatureArrayDomain:
        """Return the declared indexing domain for one feature array."""
        return self.feature_array_domains.get(
            feature_name,
            ObjectFeatureArrayDomain.MEASURED_OBJECT_ID,
        )

    def validate_feature_value_domain(
        self,
        feature_name: str,
        values: np.ndarray,
    ) -> None:
        """Fail when a feature array is not aligned to a declared object domain."""
        if values.ndim == 0:
            return
        context = ObjectFeatureArrayDomainContext(
            object_id=0,
            values=values,
            measured_object_ids=self.measured_object_ids,
            object_domain=self.object_domain,
        )
        if ObjectFeatureArrayDomainStrategy.for_enum_member(
            self.feature_array_domain(feature_name)
        ).accepts(context):
            return
        raise ValueError(
            f"{type(self).__name__} feature {feature_name!r} has {context.value_count} "
            f"values for {context.measured_object_count} measured objects. Feature arrays "
            "must align to measured_object_ids unless the table declares another "
            "feature-array domain."
        )

    def python_feature_values(self, values: Any) -> Any:
        """Return Python-native feature values for row serialization."""
        array = np.asarray(values)
        if array.ndim == 0:
            return array.item()
        return array.tolist()

    def complete_row(self, row: dict[str, float | int]) -> None:
        """Add table-specific axis/value fields after feature projection."""
        del row

    def feature_missing_value(self, feature_name: str) -> ObjectFeatureMissingValue:
        """Return the declared missing-value policy for one feature."""
        return self.feature_missing_values.get(
            feature_name,
            ObjectFeatureMissingValue.NAN,
        )


class GenericObjectFeatureValueTable(ObjectFeatureValueTable):
    """Generic object feature table with NaN missing values."""

    table_label = "generic"


class ShapeObjectFeatureValueTable(ObjectFeatureValueTable):
    """Object shape feature rows with CellProfiler-compatible missing values."""

    table_label = "shape"
    feature_array_domains: ClassVar[Mapping[str, ObjectFeatureArrayDomain]] = {
        **{
            feature.value: ObjectFeatureArrayDomain.ROW_ORDINAL
            for feature in (
                ObjectShapeMeasurementFeature.MIN_FERET_DIAMETER,
                ObjectShapeMeasurementFeature.MAX_FERET_DIAMETER,
            )
        },
        **{
            field_name: ObjectFeatureArrayDomain.ROW_ORDINAL
            for field_name in zernike_shape_feature_names(max_order=9)
        },
        **{
            feature.value: ObjectFeatureArrayDomain.LABEL_ID
            for feature in (
                ObjectShapeMeasurementFeature.CENTER_X,
                ObjectShapeMeasurementFeature.CENTER_Y,
            )
        },
    }
    feature_missing_values: ClassVar[Mapping[str, ObjectFeatureMissingValue]] = {
        feature.value: ObjectFeatureMissingValue.ZERO
        for feature in (
            ObjectShapeMeasurementFeature.MIN_FERET_DIAMETER,
            ObjectShapeMeasurementFeature.MAX_FERET_DIAMETER,
            ObjectShapeMeasurementFeature.MAXIMUM_RADIUS,
            ObjectShapeMeasurementFeature.MEAN_RADIUS,
            ObjectShapeMeasurementFeature.MEDIAN_RADIUS,
        )
    }

    def complete_row(self, row: dict[str, float | int]) -> None:
        row[ObjectShapeMeasurementFeature.CENTER_Z.value] = 0.0


class MeasurementTableRowLayout(str, Enum):
    """Nominal row layout for measurement tables."""

    EMPTY = "empty"
    LONG = "long"
    WIDE = "wide"


class MeasurementRowLayoutProjectionStrategy(
    EnumKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project one nominal measurement row layout into canonical long form."""

    __registry_key__ = "layout"
    __skip_if_no_key__ = True
    layout: ClassVar[MeasurementTableRowLayout | None] = None

    @abstractmethod
    def long_rows(self, row: object) -> tuple[Mapping[str, object], ...]:
        """Return canonical long-form rows for one source row."""


class LongMeasurementRowProjectionStrategy(MeasurementRowLayoutProjectionStrategy):
    """Preserve already-long rows."""

    layout = MeasurementTableRowLayout.LONG

    def long_rows(self, row: object) -> tuple[Mapping[str, object], ...]:
        return (measurement_row_mapping(row),)


class WideMeasurementRowProjectionStrategy(MeasurementRowLayoutProjectionStrategy):
    """Explode wide feature columns into canonical long-form rows."""

    layout = MeasurementTableRowLayout.WIDE

    def long_rows(self, row: object) -> tuple[Mapping[str, object], ...]:
        row_mapping = measurement_row_mapping(row)
        axis_fields = measurement_row_axis_field_names()
        axis_values = {
            str(field_name): value
            for field_name, value in row_mapping.items()
            if str(field_name) in axis_fields
        }
        long_rows: list[Mapping[str, object]] = []
        for field_name, value in row_mapping.items():
            field_text = str(field_name)
            if field_text in axis_fields:
                continue
            long_row = dict(axis_values)
            long_row[MeasurementRowAxisField.FEATURE_NAME.value] = field_text
            long_row[MeasurementRowValueField.RESULT_VALUE.value] = value
            long_rows.append(long_row)
        return tuple(long_rows)


def measurement_row_feature_field_names() -> frozenset[str]:
    """Return row fields that name long-form measurement features."""
    return frozenset(
        (
            MeasurementRowAxisField.FEATURE_NAME.value,
            MeasurementRowAxisField.MEASUREMENT_NAME.value,
            MeasurementRowAxisField.OUTPUT_NAME.value,
        )
    )


def measurement_row_value_field_names() -> frozenset[str]:
    """Return row fields that carry long-form measurement values."""
    return frozenset(field.value for field in MeasurementRowValueField)


def measurement_row_mapping(row: object) -> Mapping[str, object]:
    """Return a mapping view for a supported measurement row payload."""
    if isinstance(row, Mapping):
        return row
    if is_dataclass(row):
        return MeasurementRowMappingCache.process_cache().mapping(row)
    try:
        return vars(row)
    except TypeError as exc:
        raise TypeError(
            f"Unsupported measurement row type {type(row).__name__}."
        ) from exc


@dataclass(slots=True)
class MeasurementRowMappingCache(
    ProcessLocalBoundedCache[int, tuple[object, Mapping[str, object]]]
):
    """Bounded process-local cache for immutable dataclass measurement rows."""

    max_entries: int = 262_144

    def mapping(self, row: object) -> Mapping[str, object]:
        row_id = id(row)
        cached = self.cached_value(row_id)
        if cached is not None:
            cached_row, row_mapping = cached
            if cached_row is row:
                return row_mapping
            del self.entries[row_id]

        row_mapping = {
            field_name: getattr(row, field_name)
            for field_name in _dataclass_field_names(type(row))
        }
        self.store_value(row_id, (row, row_mapping))
        return row_mapping


def measurement_table_row_layout(rows: object) -> MeasurementTableRowLayout:
    """Return the declared layout implied by a table row payload."""
    observed_layouts = measurement_table_row_layouts(rows)
    if not observed_layouts:
        return MeasurementTableRowLayout.EMPTY
    if len(observed_layouts) != 1:
        raise ValueError(
            "MeasurementTable rows must not mix long-form and wide-form layouts; "
            f"got {sorted(layout.value for layout in observed_layouts)!r}."
        )
    return next(iter(observed_layouts))


def measurement_table_row_layout_from_fields(
    fields: Iterable[FieldSpec],
) -> MeasurementTableRowLayout | None:
    """Return row layout declared by table fields when fields are authoritative."""
    return _measurement_table_row_layout_from_field_names(
        tuple(field.name for field in fields)
    )


@lru_cache(maxsize=256)
def _measurement_table_row_layout_from_field_names(
    field_names_tuple: tuple[str, ...],
) -> MeasurementTableRowLayout | None:
    """Return row layout declared by field names."""
    field_names = frozenset(field_names_tuple)
    if not field_names:
        return None
    has_feature_field = bool(field_names & measurement_row_feature_field_names())
    has_value_field = bool(field_names & measurement_row_value_field_names())
    if has_feature_field and not has_value_field:
        raise ValueError(
            "Long-form measurement table fields must declare both a feature field "
            f"and a value field, got fields {sorted(field_names)!r}."
        )
    return (
        MeasurementTableRowLayout.LONG
        if has_feature_field
        else MeasurementTableRowLayout.WIDE
    )


def measurement_table_row_layouts(rows: object) -> frozenset[MeasurementTableRowLayout]:
    """Return every nominal row layout observed in a measurement payload."""
    if rows is None:
        return frozenset()
    row_sequence = rows if isinstance(rows, list | tuple) else (rows,)
    if not row_sequence:
        return frozenset()
    if isinstance(row_sequence[0], ObjectMeasurementValueRow) and all(
        isinstance(row, ObjectMeasurementValueRow) for row in row_sequence
    ):
        return frozenset((MeasurementTableRowLayout.LONG,))
    return frozenset(MeasurementRowLayoutAuthority(row).layout() for row in row_sequence)


def normalize_measurement_table_rows(
    rows: object,
    *,
    fields: Iterable[FieldSpec] = (),
) -> object:
    """Return homogeneous measurement rows, canonicalizing mixed tables to long form."""
    declared_layout = measurement_table_row_layout_from_fields(fields)
    if declared_layout is not None:
        return rows
    observed_layouts = measurement_table_row_layouts(rows)
    if len(observed_layouts) <= 1:
        return rows
    return measurement_rows_as_layout(rows, MeasurementTableRowLayout.LONG)


def measurement_rows_as_layout(
    rows: object,
    layout: MeasurementTableRowLayout,
) -> object:
    """Project measurement rows into a declared table layout."""
    if layout is not MeasurementTableRowLayout.LONG:
        raise ValueError(f"Unsupported measurement row layout projection: {layout.value}.")
    row_sequence = rows if isinstance(rows, list | tuple) else (rows,)
    return [
        projected_row
        for row in row_sequence
        for projected_row in MeasurementRowLayoutProjectionStrategy.for_enum_member(
            MeasurementRowLayoutAuthority(row).layout()
        ).long_rows(row)
    ]


@dataclass(frozen=True, slots=True)
class MeasurementRowLayoutAuthority:
    """Classify measurement rows by their declared feature/value fields."""

    row: object

    def layout(self) -> MeasurementTableRowLayout:
        field_names = frozenset(
            str(field_name) for field_name in measurement_row_mapping(self.row)
        )
        has_feature_field = bool(field_names & measurement_row_feature_field_names())
        has_value_field = bool(field_names & measurement_row_value_field_names())
        if has_feature_field and not has_value_field:
            raise ValueError(
                "Long-form measurement rows must declare both a feature field and a "
                f"value field, got fields {sorted(field_names)!r}."
            )
        return (
            MeasurementTableRowLayout.LONG
            if has_feature_field
            else MeasurementTableRowLayout.WIDE
        )


@lru_cache(maxsize=256)
def _dataclass_field_names(row_type: type[object]) -> tuple[str, ...]:
    """Return dataclass field names without per-row reflection overhead."""
    return tuple(field.name for field in fields(row_type))


class MeasurementObjectRowIdentity(str, Enum):
    """How object-scoped measurement rows identify their measured object."""

    LABEL_ID = "label_id"
    ROW_ORDINAL = "row_ordinal"


def measurement_row_axis_field_names() -> frozenset[str]:
    """Return fields that identify a measurement row axis, not a result value."""
    return frozenset(field.value for field in MeasurementRowAxisField)


def indexed_measurement_feature_name(
    feature: ObjectShapeMeasurementFeature | str,
    *indices: int,
) -> str:
    """Return a stable runtime field name for indexed measurement features."""
    feature = coerce_enum(
        ObjectShapeMeasurementFeature,
        feature,
        "indexed_measurement_feature_name.feature",
    )
    if not indices:
        return feature.value
    return "_".join((feature.value, *(str(int(index)) for index in indices)))


@dataclass(frozen=True, slots=True)
class IndexedObjectZernikeDescriptor:
    """Parsed identity for an indexed object Zernike descriptor feature."""

    family: ObjectZernikeDescriptorFeature
    degree: int
    repetition: int

    @classmethod
    def from_feature_name(
        cls,
        feature_name: str,
    ) -> "IndexedObjectZernikeDescriptor | None":
        normalized_parts = tuple(
            part
            for part in str(feature_name).strip().lower().replace("-", "_").split("_")
            if part
        )
        for family in ObjectZernikeDescriptorFeature:
            family_parts = tuple(
                part
                for part in family.value.split("_")
                if part
            )
            family_prefixes = (family_parts, ("".join(family_parts),))
            for family_prefix in family_prefixes:
                if len(normalized_parts) != len(family_prefix) + 2:
                    continue
                if normalized_parts[: len(family_prefix)] != family_prefix:
                    continue
                degree_text, repetition_text = normalized_parts[-2:]
                if not degree_text.isdecimal() or not repetition_text.isdecimal():
                    continue
                return cls(
                    family=family,
                    degree=int(degree_text),
                    repetition=int(repetition_text),
                )
        return None


def object_shape_measurement_field_names(
    *,
    dimensions: int = 2,
    calculate_advanced: bool = True,
    calculate_zernikes: bool = True,
    object_id_field: str = "object_label",
    slice_index_field: str = "slice_index",
) -> tuple[str, ...]:
    """Return canonical table fields for object shape measurements."""
    if dimensions not in (2, 3):
        raise ValueError(f"Object shape measurements support 2D/3D, got {dimensions}D.")

    fields: list[str] = [slice_index_field, object_id_field]
    if dimensions == 2:
        fields.extend(feature.value for feature in _OBJECT_SHAPE_STANDARD_2D_FIELDS)
        fields.append(ObjectShapeMeasurementFeature.CENTER_Z.value)
        if calculate_advanced:
            fields.extend(_indexed_object_shape_fields(_OBJECT_SHAPE_ADVANCED_2D_SPECS))
        if calculate_zernikes:
            fields.extend(zernike_shape_feature_names(max_order=9))
    else:
        fields.extend(feature.value for feature in _OBJECT_SHAPE_STANDARD_3D_FIELDS)
        if calculate_advanced:
            fields.append(ObjectShapeMeasurementFeature.SOLIDITY.value)
    return tuple(dict.fromkeys(fields))


def object_shape_measurement_all_field_names(
    *,
    calculate_advanced: bool = True,
    calculate_zernikes: bool = True,
    object_id_field: str = "object_label",
    slice_index_field: str = "slice_index",
) -> tuple[str, ...]:
    """Return the union schema for object-shape tables that may emit 2D or 3D rows."""

    return tuple(
        dict.fromkeys(
            (
                *object_shape_measurement_field_names(
                    dimensions=2,
                    calculate_advanced=calculate_advanced,
                    calculate_zernikes=calculate_zernikes,
                    object_id_field=object_id_field,
                    slice_index_field=slice_index_field,
                ),
                *object_shape_measurement_field_names(
                    dimensions=3,
                    calculate_advanced=calculate_advanced,
                    calculate_zernikes=calculate_zernikes,
                    object_id_field=object_id_field,
                    slice_index_field=slice_index_field,
                ),
            )
        )
    )


_OBJECT_SHAPE_STANDARD_2D_FIELDS = (
    ObjectShapeMeasurementFeature.AREA,
    ObjectShapeMeasurementFeature.PERIMETER,
    ObjectShapeMeasurementFeature.MAJOR_AXIS_LENGTH,
    ObjectShapeMeasurementFeature.MINOR_AXIS_LENGTH,
    ObjectShapeMeasurementFeature.ECCENTRICITY,
    ObjectShapeMeasurementFeature.ORIENTATION,
    ObjectShapeMeasurementFeature.CENTER_X,
    ObjectShapeMeasurementFeature.CENTER_Y,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_AREA,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_X,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_X,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_Y,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_Y,
    ObjectShapeMeasurementFeature.FORM_FACTOR,
    ObjectShapeMeasurementFeature.EXTENT,
    ObjectShapeMeasurementFeature.SOLIDITY,
    ObjectShapeMeasurementFeature.COMPACTNESS,
    ObjectShapeMeasurementFeature.EULER_NUMBER,
    ObjectShapeMeasurementFeature.MAXIMUM_RADIUS,
    ObjectShapeMeasurementFeature.MEAN_RADIUS,
    ObjectShapeMeasurementFeature.MEDIAN_RADIUS,
    ObjectShapeMeasurementFeature.CONVEX_AREA,
    ObjectShapeMeasurementFeature.MIN_FERET_DIAMETER,
    ObjectShapeMeasurementFeature.MAX_FERET_DIAMETER,
    ObjectShapeMeasurementFeature.EQUIVALENT_DIAMETER,
)


_OBJECT_SHAPE_STANDARD_3D_FIELDS = (
    ObjectShapeMeasurementFeature.VOLUME,
    ObjectShapeMeasurementFeature.SURFACE_AREA,
    ObjectShapeMeasurementFeature.MAJOR_AXIS_LENGTH,
    ObjectShapeMeasurementFeature.MINOR_AXIS_LENGTH,
    ObjectShapeMeasurementFeature.CENTER_X,
    ObjectShapeMeasurementFeature.CENTER_Y,
    ObjectShapeMeasurementFeature.CENTER_Z,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_VOLUME,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_X,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_X,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_Y,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_Y,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_MINIMUM_Z,
    ObjectShapeMeasurementFeature.BOUNDING_BOX_MAXIMUM_Z,
    ObjectShapeMeasurementFeature.EXTENT,
    ObjectShapeMeasurementFeature.EULER_NUMBER,
    ObjectShapeMeasurementFeature.EQUIVALENT_DIAMETER,
)


_OBJECT_SHAPE_ADVANCED_2D_SPECS = (
    (ObjectShapeMeasurementFeature.SPATIAL_MOMENT, range(3), range(4)),
    (ObjectShapeMeasurementFeature.CENTRAL_MOMENT, range(3), range(4)),
    (ObjectShapeMeasurementFeature.NORMALIZED_MOMENT, range(4), range(4)),
    (ObjectShapeMeasurementFeature.HU_MOMENT, range(7), None),
    (ObjectShapeMeasurementFeature.INERTIA_TENSOR, range(2), range(2)),
    (ObjectShapeMeasurementFeature.INERTIA_TENSOR_EIGENVALUES, range(2), None),
)


def _indexed_object_shape_fields(
    specs: tuple[
        tuple[ObjectShapeMeasurementFeature, range, range | None],
        ...,
    ],
) -> tuple[str, ...]:
    fields: list[str] = []
    for feature, rows, columns in specs:
        if columns is None:
            fields.extend(indexed_measurement_feature_name(feature, row) for row in rows)
            continue
        fields.extend(
            indexed_measurement_feature_name(feature, row, column)
            for row in rows
            for column in columns
        )
    return tuple(fields)


@dataclass(frozen=True, slots=True)
class MeasurementSubject:
    """Entity measured by a measurement table."""

    scope: MeasurementScope
    name: str | None = None
    id_field: str | None = None

    def __post_init__(self) -> None:
        scope = coerce_enum(MeasurementScope, self.scope, "MeasurementSubject.scope")
        object.__setattr__(self, "scope", scope)

        if self.name == "":
            raise ValueError("MeasurementSubject.name cannot be empty.")
        if self.id_field == "":
            raise ValueError("MeasurementSubject.id_field cannot be empty.")
        if scope.requires_subject_name and self.name is None:
            raise ValueError(
                f"MeasurementSubject.name is required for {scope.value} scope."
            )

    @property
    def source_image_name(self) -> str | None:
        """Return the concrete source image represented by this subject, if any."""
        if self.scope is not MeasurementScope.IMAGE or self.name is None:
            return None
        if self.name.casefold() == MeasurementScope.IMAGE.value:
            return None
        return self.name


@dataclass(frozen=True, slots=True)
class RelationshipEndpoint:
    """One endpoint in a directed relationship."""

    name: str
    role: str
    id_field: str
    kind: ArtifactKind = ArtifactKind.OBJECT_LABELS

    def __post_init__(self) -> None:
        _require_name(self.name, "RelationshipEndpoint.name")
        _require_name(self.role, "RelationshipEndpoint.role")
        _require_name(self.id_field, "RelationshipEndpoint.id_field")
        object.__setattr__(
            self,
            "kind",
            coerce_enum(ArtifactKind, self.kind, "RelationshipEndpoint.kind"),
        )


PARENT_RELATIONSHIP_ROLE = "parent"
CHILD_RELATIONSHIP_ROLE = "child"
PARENT_RELATIONSHIP_ID_FIELD = "parent_id"
CHILD_RELATIONSHIP_ID_FIELD = "child_id"
PARENT_CHILD_RELATIONSHIP_TYPE = "parent_child"
PARENT_CHILD_RELATIONSHIP_ARTIFACT_SUFFIX = "relationships"


def parent_child_relationship_artifact_name(parent_name: str, child_name: str) -> str:
    """Return the canonical artifact name for a directed parent-child relation."""
    _require_name(parent_name, "parent_name")
    _require_name(child_name, "child_name")
    return f"{parent_name}_{child_name}_{PARENT_CHILD_RELATIONSHIP_ARTIFACT_SUFFIX}"


def parent_child_relationship_artifact_endpoints(
    artifact_name: str,
    *,
    parent_candidates: tuple[str, ...],
) -> tuple[str, str] | None:
    """Return parent/child names encoded by a canonical relationship artifact.

    Child object outputs can be pruned from a runtime contract while their
    relationship artifact is retained. Reconstructing the child endpoint from
    the same canonical naming schema keeps relationship recording typed without
    depending on module-local string conventions.
    """
    _require_name(artifact_name, "artifact_name")
    suffix = f"_{PARENT_CHILD_RELATIONSHIP_ARTIFACT_SUFFIX}"
    if not artifact_name.endswith(suffix):
        return None
    body = artifact_name[: -len(suffix)]
    for parent_name in parent_candidates:
        _require_name(parent_name, "parent_candidate")
        prefix = f"{parent_name}_"
        if not body.startswith(prefix):
            continue
        child_name = body[len(prefix):]
        if child_name:
            return parent_name, child_name
    return None


@dataclass(frozen=True, slots=True)
class ParentChildRelationshipPayload:
    """Generic parent-child id pairs emitted before endpoint names are bound."""

    parent_ids: tuple[int, ...]
    child_ids: tuple[int, ...]
    slice_indices: tuple[int, ...] = ()
    slice_count: int | None = None

    def __post_init__(self) -> None:
        parent_ids = tuple(int(parent_id) for parent_id in self.parent_ids)
        child_ids = tuple(int(child_id) for child_id in self.child_ids)
        if len(parent_ids) != len(child_ids):
            raise ValueError(
                "ParentChildRelationshipPayload parent_ids and child_ids must "
                f"have equal length, got {len(parent_ids)} and {len(child_ids)}."
            )
        slice_indices = tuple(int(slice_index) for slice_index in self.slice_indices)
        if slice_indices and len(slice_indices) != len(parent_ids):
            raise ValueError(
                "ParentChildRelationshipPayload slice_indices must be empty or "
                "match parent_ids/child_ids length, got "
                f"{len(slice_indices)} for {len(parent_ids)} relationships."
            )
        if any(slice_index < 0 for slice_index in slice_indices):
            raise ValueError("ParentChildRelationshipPayload slice_indices cannot be negative.")
        slice_count = None if self.slice_count is None else int(self.slice_count)
        if slice_count is not None and slice_count < 0:
            raise ValueError("ParentChildRelationshipPayload slice_count cannot be negative.")
        if (
            slice_count is not None
            and slice_indices
            and max(slice_indices) >= slice_count
        ):
            raise ValueError(
                "ParentChildRelationshipPayload slice_indices must be smaller "
                f"than slice_count {slice_count}."
            )
        object.__setattr__(self, "parent_ids", parent_ids)
        object.__setattr__(self, "child_ids", child_ids)
        object.__setattr__(self, "slice_indices", slice_indices)
        object.__setattr__(self, "slice_count", slice_count)


class ObjectRelationshipPayloadKernel(ABC):
    """Kernel contract used by semantic relationship payload policies."""

    def dominant_parent_ids_by_child(
        self,
        parent_array: Any,
        child_array: Any,
        context_array: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return child ids with their dominant parent ids by positive overlap."""
        children = np.asarray(child_array, dtype=np.int64)
        parents = np.asarray(parent_array, dtype=np.int64)
        context = np.asarray(context_array, dtype=np.int64)

        child_ids = np.unique(children[children > 0])
        if child_ids.size == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty

        max_parent = int(np.max(parents)) if parents.size else 0
        parent_ids = np.zeros(child_ids.size, dtype=np.int64)
        valid = (context > 0) & (parents > 0)
        if not np.any(valid) or max_parent <= 0:
            return child_ids, parent_ids

        stride = max_parent + 1
        pair_keys = context[valid] * stride + parents[valid]
        counts = np.bincount(pair_keys)
        child_to_index = np.full(int(child_ids[-1]) + 1, -1, dtype=np.int64)
        child_to_index[child_ids] = np.arange(child_ids.size, dtype=np.int64)

        nonzero_keys = np.flatnonzero(counts)
        best_counts = np.zeros(child_ids.size, dtype=np.int64)
        for key in nonzero_keys:
            child_id = key // stride
            if child_id >= child_to_index.size:
                continue
            output_index = child_to_index[child_id]
            if output_index < 0:
                continue
            count = counts[key]
            parent_id = key % stride
            if count > best_counts[output_index]:
                best_counts[output_index] = count
                parent_ids[output_index] = parent_id
        return child_ids, parent_ids

    @abstractmethod
    def relate_children_to_parents(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        child_count: int,
    ) -> np.ndarray:
        """Assign each child label id to its dominant parent label id."""

    @abstractmethod
    def relate_sparse_ijv_children_to_parents(
        self,
        parent_rows: np.ndarray,
        child_rows: np.ndarray,
        child_count: int,
        parent_count: int,
    ) -> np.ndarray:
        """Assign sparse IJV child ids to parent ids."""


class DefaultObjectRelationshipPayloadKernel(ObjectRelationshipPayloadKernel):
    """Core NumPy relationship kernel used outside backend-specific execution."""

    def relate_children_to_parents(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        child_count: int,
    ) -> np.ndarray:
        del child_count
        child_ids, parent_ids = self.dominant_parent_ids_by_child(
            parent_labels,
            child_labels,
            child_labels,
        )
        parents_of = np.zeros(
            int(child_labels.max()) if child_labels.size else 0,
            dtype=np.int32,
        )
        for child_id, parent_id in zip(child_ids, parent_ids, strict=True):
            if 0 < child_id <= parents_of.size:
                parents_of[int(child_id) - 1] = int(parent_id)
        return parents_of

    def relate_sparse_ijv_children_to_parents(
        self,
        parent_rows: np.ndarray,
        child_rows: np.ndarray,
        child_count: int,
        parent_count: int,
    ) -> np.ndarray:
        del parent_count
        parents_of = np.zeros(child_count, dtype=np.int32)
        if parent_rows.size == 0 or child_rows.size == 0:
            return parents_of
        parent_by_yx = {
            (int(row[0]), int(row[1])): int(row[2])
            for row in np.asarray(parent_rows, dtype=np.int64)
            if int(row[2]) > 0
        }
        votes: dict[tuple[int, int], int] = {}
        for row in np.asarray(child_rows, dtype=np.int64):
            child_id = int(row[2])
            if child_id <= 0:
                continue
            parent_id = parent_by_yx.get((int(row[0]), int(row[1])), 0)
            if parent_id <= 0:
                continue
            votes[(child_id, parent_id)] = votes.get((child_id, parent_id), 0) + 1
        for child_id in range(1, child_count + 1):
            child_votes = tuple(
                (parent_id, count)
                for (candidate_child, parent_id), count in votes.items()
                if candidate_child == child_id
            )
            if child_votes:
                parents_of[child_id - 1] = min(
                    child_votes,
                    key=lambda item: (-item[1], item[0]),
                )[0]
        return parents_of


DEFAULT_OBJECT_RELATIONSHIP_PAYLOAD_KERNEL = DefaultObjectRelationshipPayloadKernel()


@dataclass(frozen=True, slots=True)
class ObjectRelationshipPayloadRequest:
    """Semantic request for deriving parent-child payloads from label values."""

    parent_labels: Any
    child_labels: Any
    kernel: ObjectRelationshipPayloadKernel = DEFAULT_OBJECT_RELATIONSHIP_PAYLOAD_KERNEL


class ObjectRelationshipPayloadStrategy(
    MostDerivedContextStrategyMixin[ObjectRelationshipPayloadRequest],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Derive parent-child payloads by nominal object-label representation."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[str | None] = None

    @abstractmethod
    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        """Return whether this strategy owns the label representation pair."""

    @abstractmethod
    def payload(
        self,
        context: ObjectRelationshipPayloadRequest,
    ) -> ParentChildRelationshipPayload:
        """Return parent-child ids for the strategy's representation contract."""

    @staticmethod
    def related_payload_from_parents_of(
        parents_of: np.ndarray,
        child_ids: np.ndarray,
    ) -> ParentChildRelationshipPayload:
        parent_ids: list[int] = []
        related_child_ids: list[int] = []
        for child_id in child_ids:
            if 0 < child_id <= len(parents_of):
                parent_id = int(parents_of[child_id - 1])
                if parent_id > 0:
                    parent_ids.append(parent_id)
                    related_child_ids.append(int(child_id))
        return ParentChildRelationshipPayload(
            parent_ids=tuple(parent_ids),
            child_ids=tuple(related_child_ids),
        )


class DenseObjectRelationshipPayloadStrategy(ObjectRelationshipPayloadStrategy):
    """Dense label images use maximum positive-pixel overlap."""

    strategy_key = "dense"

    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        del context
        return True

    def payload(
        self,
        context: ObjectRelationshipPayloadRequest,
    ) -> ParentChildRelationshipPayload:
        from openhcs.core.runtime_values import object_label_dense_array

        slice_count = self.relationship_slice_count(context)
        if slice_count is not None:
            aligned_stacks = DenseObjectLabelPairAligner(
                context.parent_labels,
                context.child_labels,
            ).aligned_stacks(slice_count)
            if aligned_stacks is not None:
                return self.stack_payload(context, *aligned_stacks)

        parent_array, child_array = (
            object_label_dense_array(labels, dtype=np.int32)
            for labels in DenseObjectLabelPairAligner(
                context.parent_labels,
                context.child_labels,
            ).aligned()
        )
        child_count = int(child_array.max()) if child_array.size else 0
        if child_count <= 0:
            return ParentChildRelationshipPayload(parent_ids=(), child_ids=())
        parents_of = context.kernel.relate_children_to_parents(
            parent_array,
            child_array,
            child_count,
        )
        present_children = np.unique(child_array[child_array > 0]).astype(
            np.int32,
            copy=False,
        )
        return self.related_payload_from_parents_of(parents_of, present_children)

    def relationship_slice_count(
        self,
        context: ObjectRelationshipPayloadRequest,
    ) -> int | None:
        """Return the plane count for plane-scoped label relationships."""
        domains = tuple(
            ObjectLabelDomainMetadataStrategy.for_value(labels).object_label_domain(
                labels,
            )
            for labels in (context.parent_labels, context.child_labels)
        )
        if (
            ObjectLabelDomainScope.common(domain.scope for domain in domains)
            is not ObjectLabelDomainScope.PLANE
        ):
            return None
        leading_plane_counts = tuple(
            int(label_array.shape[0])
            for label_array in (
                np.asarray(context.parent_labels),
                np.asarray(context.child_labels),
            )
            if label_array.ndim == 3
        )
        if not leading_plane_counts:
            return None
        return max(leading_plane_counts)

    def stack_payload(
        self,
        context: ObjectRelationshipPayloadRequest,
        parent_stack: np.ndarray,
        child_stack: np.ndarray,
    ) -> ParentChildRelationshipPayload:
        """Return parent-child ids with explicit runtime-slice identity."""
        parent_ids: list[int] = []
        child_ids: list[int] = []
        slice_indices: list[int] = []
        for slice_index, (parent_plane, child_plane) in enumerate(
            zip(parent_stack, child_stack, strict=True)
        ):
            child_count = int(child_plane.max()) if child_plane.size else 0
            if child_count <= 0:
                continue
            parents_of = context.kernel.relate_children_to_parents(
                parent_plane,
                child_plane,
                child_count,
            )
            present_children = np.unique(child_plane[child_plane > 0]).astype(
                np.int32,
                copy=False,
            )
            payload = self.related_payload_from_parents_of(
                parents_of,
                present_children,
            )
            parent_ids.extend(payload.parent_ids)
            child_ids.extend(payload.child_ids)
            slice_indices.extend(slice_index for _child_id in payload.child_ids)
        return ParentChildRelationshipPayload(
            parent_ids=tuple(parent_ids),
            child_ids=tuple(child_ids),
            slice_indices=tuple(slice_indices),
            slice_count=int(parent_stack.shape[0]),
        )


class SparseIJVObjectRelationshipPayloadStrategy(DenseObjectRelationshipPayloadStrategy):
    """Sparse IJV labels derive parent-child ids through sparse rows."""

    strategy_key = "sparse_ijv"

    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        return self.is_sparse_ijv(context.parent_labels) or self.is_sparse_ijv(
            context.child_labels
        )

    def payload(
        self,
        context: ObjectRelationshipPayloadRequest,
    ) -> ParentChildRelationshipPayload:
        parent_rows = self.sparse_rows(context.parent_labels)
        child_rows = self.sparse_rows(context.child_labels)
        parent_array = parent_rows.as_yx_label_array()
        child_array = child_rows.as_yx_label_array()
        parent_count = self.label_count(parent_array, parent_rows)
        child_count = self.label_count(child_array, child_rows)
        if parent_count <= 0 or child_count <= 0:
            return ParentChildRelationshipPayload(parent_ids=(), child_ids=())
        parents_of = context.kernel.relate_sparse_ijv_children_to_parents(
            np.asarray(parent_array, dtype=np.int64),
            np.asarray(child_array, dtype=np.int64),
            child_count,
            parent_count,
        )
        return self.related_payload_from_parents_of(
            parents_of,
            self.present_sparse_child_ids(child_array, child_rows),
        )

    @classmethod
    def is_sparse_ijv(cls, labels: Any) -> bool:
        from openhcs.core.runtime_values import (
            ObjectLabelRepresentation,
            ObjectLabelSet,
            SparseIJVLabelRows,
        )

        if isinstance(labels, SparseIJVLabelRows):
            return True
        return (
            isinstance(labels, ObjectLabelSet)
            and labels.representation is ObjectLabelRepresentation.SPARSE_IJV
        )

    @classmethod
    def sparse_rows(cls, labels: Any) -> Any:
        from openhcs.core.runtime_values import (
            ObjectLabelRepresentation,
            ObjectLabelSet,
            SparseIJVLabelRows,
        )

        if isinstance(labels, ObjectLabelSet):
            if labels.representation is not ObjectLabelRepresentation.SPARSE_IJV:
                return SparseIJVLabelRows.from_dense_labels(labels.labels)
            labels = labels.labels
        if isinstance(labels, SparseIJVLabelRows):
            return labels
        return SparseIJVLabelRows.from_dense_labels(labels)

    @staticmethod
    def label_count(
        array: np.ndarray,
        rows: Any,
    ) -> int:
        if array.size == 0:
            return 0
        return int(np.max(array[:, rows.label_column]))

    @staticmethod
    def present_sparse_child_ids(
        child_array: np.ndarray,
        child_rows: Any,
    ) -> np.ndarray:
        if child_array.size == 0:
            return np.empty(0, dtype=np.int32)
        return np.unique(child_array[:, child_rows.label_column]).astype(
            np.int32,
            copy=False,
        )


class ObjectLabelLineageGeometry(str, Enum):
    """Geometry relation used to derive parent-child label lineage."""

    SHARED_GEOMETRY = "shared_geometry"
    IDENTITY_DOMAIN = "identity_domain"


class ObjectLabelLineageStrategy(
    EnumKeyedStrategyMixin[ObjectLabelLineageGeometry],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Derive parent-child object lineage from two dense label artifacts."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[ObjectLabelLineageGeometry | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def payload(
        self,
        parent_labels: Any,
        child_labels: Any,
    ) -> ParentChildRelationshipPayload:
        """Return parent-child ids for the strategy's geometry contract."""


class SharedGeometryObjectLabelLineageStrategy(ObjectLabelLineageStrategy):
    """Use spatial overlap when parent and child labels share a geometry."""

    strategy_key = ObjectLabelLineageGeometry.SHARED_GEOMETRY

    def payload(
        self,
        parent_labels: Any,
        child_labels: Any,
    ) -> ParentChildRelationshipPayload:
        return object_label_parent_child_payload(parent_labels, child_labels)


class IdentityDomainObjectLabelLineageStrategy(ObjectLabelLineageStrategy):
    """Use preserved label ids when a transform changes label geometry."""

    strategy_key = ObjectLabelLineageGeometry.IDENTITY_DOMAIN

    def payload(
        self,
        parent_labels: Any,
        child_labels: Any,
    ) -> ParentChildRelationshipPayload:
        parent_ids = set(dense_object_label_id_domain(parent_labels))
        child_ids = tuple(dense_object_label_id_domain(child_labels))
        related_ids = tuple(object_id for object_id in child_ids if object_id in parent_ids)
        return ParentChildRelationshipPayload(
            parent_ids=related_ids,
            child_ids=related_ids,
        )


@dataclass(frozen=True, slots=True)
class ObjectInstanceKey:
    """Typed identity for one object label inside an optional measurement plane."""

    object_id: int
    slice_index: int | None = None

    def __post_init__(self) -> None:
        object_id = int(self.object_id)
        if object_id <= 0:
            raise ValueError("ObjectInstanceKey.object_id must be positive.")
        slice_index = None if self.slice_index is None else int(self.slice_index)
        if slice_index is not None and slice_index < 0:
            raise ValueError("ObjectInstanceKey.slice_index cannot be negative.")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "slice_index", slice_index)

    @classmethod
    def from_measurement_row(
        cls,
        row: Mapping[str, Any],
        object_id: int,
        *,
        image_number_offset: float = 1.0,
        image_number_field: MeasurementRowAxisField = MeasurementRowAxisField.IMAGE_NUMBER,
        slice_index_field: MeasurementRowAxisField = MeasurementRowAxisField.SLICE_INDEX,
    ) -> "ObjectInstanceKey":
        """Build object identity from the row's nominal axis fields."""
        raw_image_number = row.get(image_number_field.value)
        if raw_image_number is not None and str(raw_image_number).strip() != "":
            slice_index = int(float(raw_image_number) - float(image_number_offset) - 1)
            if slice_index >= 0:
                return cls(object_id, slice_index=slice_index)
        raw_slice_index = row.get(slice_index_field.value)
        if raw_slice_index is None or str(raw_slice_index).strip() == "":
            return cls(object_id)
        return cls(object_id, slice_index=int(raw_slice_index))

    @classmethod
    def domain(
        cls,
        object_ids: Iterable[int],
        *,
        slice_index: int | None = None,
    ) -> tuple["ObjectInstanceKey", ...]:
        """Return typed object identities for one optional measurement plane."""
        return tuple(cls(object_id, slice_index=slice_index) for object_id in object_ids)


@dataclass(frozen=True, slots=True)
class ObjectInstanceRelationship:
    """Parent-child relationships keyed by typed object instance identity."""

    source_keys: tuple[ObjectInstanceKey, ...]
    target_keys: tuple[ObjectInstanceKey, ...]
    slice_count: int | None = None

    @classmethod
    def from_id_columns(
        cls,
        source_ids: Iterable[int],
        target_ids: Iterable[int],
        *,
        slice_indices: Iterable[int] = (),
        slice_count: int | None = None,
    ) -> "ObjectInstanceRelationship":
        """Build typed relationship identity from id columns plus optional slices."""
        source_id_tuple = tuple(int(value) for value in source_ids)
        target_id_tuple = tuple(int(value) for value in target_ids)
        if len(source_id_tuple) != len(target_id_tuple):
            raise ValueError(
                "ObjectInstanceRelationship source_ids and target_ids must have "
                f"equal length, got {len(source_id_tuple)} and {len(target_id_tuple)}."
            )
        slice_index_tuple = tuple(int(value) for value in slice_indices)
        if slice_index_tuple and len(slice_index_tuple) != len(source_id_tuple):
            raise ValueError(
                "ObjectInstanceRelationship slice_indices must be empty or match "
                f"id columns, got {len(slice_index_tuple)} for {len(source_id_tuple)}."
            )
        resolved_slice_count = None if slice_count is None else int(slice_count)
        if resolved_slice_count is not None and resolved_slice_count < 0:
            raise ValueError("ObjectInstanceRelationship.slice_count cannot be negative.")
        source_keys: list[ObjectInstanceKey] = []
        target_keys: list[ObjectInstanceKey] = []
        for index, (source_id, target_id) in enumerate(zip(source_id_tuple, target_id_tuple, strict=True)):
            slice_index = slice_index_tuple[index] if slice_index_tuple else None
            if source_id > 0 and target_id > 0:
                source_keys.append(ObjectInstanceKey(source_id, slice_index))
                target_keys.append(ObjectInstanceKey(target_id, slice_index))
        return cls(
            source_keys=tuple(source_keys),
            target_keys=tuple(target_keys),
            slice_count=resolved_slice_count,
        )

    def source_domain(
        self,
        object_count: int = 0,
        *,
        declared_keys: Iterable[ObjectInstanceKey] = (),
    ) -> tuple[ObjectInstanceKey, ...]:
        """Return all source identities represented by this relationship domain."""
        return self._domain(
            self.source_keys,
            object_count=object_count,
            slice_count=self.slice_count,
            declared_keys=declared_keys,
        )

    def target_domain(
        self,
        object_count: int = 0,
        *,
        declared_keys: Iterable[ObjectInstanceKey] = (),
    ) -> tuple[ObjectInstanceKey, ...]:
        """Return all target identities represented by this relationship domain."""
        return self._domain(
            self.target_keys,
            object_count=object_count,
            slice_count=self.slice_count,
            declared_keys=declared_keys,
        )

    def child_keys_by_parent(
        self,
        *,
        source_object_count: int = 0,
        declared_source_keys: Iterable[ObjectInstanceKey] = (),
    ) -> dict[ObjectInstanceKey, tuple[ObjectInstanceKey, ...]]:
        """Return target identities grouped by source identity."""
        children: dict[ObjectInstanceKey, list[ObjectInstanceKey]] = {
            source_key: []
            for source_key in self.source_domain(
                source_object_count,
                declared_keys=declared_source_keys,
            )
        }
        for source_key, target_key in zip(self.source_keys, self.target_keys, strict=True):
            children.setdefault(source_key, []).append(target_key)
        return {
            source_key: tuple(sorted(child_keys, key=lambda key: (key.slice_index is None, key.slice_index or -1, key.object_id)))
            for source_key, child_keys in sorted(
                children.items(),
                key=lambda item: (
                    item[0].slice_index is None,
                    item[0].slice_index or -1,
                    item[0].object_id,
                ),
            )
        }

    def parent_key_by_child(self) -> dict[ObjectInstanceKey, ObjectInstanceKey]:
        """Return source identity for each target identity."""
        return dict(zip(self.target_keys, self.source_keys, strict=True))

    @staticmethod
    def _domain(
        keys: tuple[ObjectInstanceKey, ...],
        *,
        object_count: int,
        slice_count: int | None,
        declared_keys: Iterable[ObjectInstanceKey],
    ) -> tuple[ObjectInstanceKey, ...]:
        declared_key_tuple = tuple(declared_keys)
        if declared_key_tuple:
            merged = {key: None for key in declared_key_tuple}
            merged.update({key: None for key in keys})
            return tuple(
                sorted(
                    merged,
                    key=lambda key: (
                        key.slice_index is None,
                        key.slice_index or -1,
                        key.object_id,
                    ),
                )
            )
        if object_count <= 0 and keys:
            return tuple(
                sorted(
                    dict.fromkeys(keys),
                    key=lambda key: (
                        key.slice_index is None,
                        key.slice_index or -1,
                        key.object_id,
                    ),
                )
            )
        max_object_id = max((key.object_id for key in keys), default=0)
        max_object_id = max(max_object_id, int(object_count))
        if max_object_id <= 0:
            return ()
        slice_indexes = (
            tuple(range(slice_count))
            if slice_count is not None and slice_count > 1
            else ()
        ) or tuple(
            dict.fromkeys(key.slice_index for key in keys if key.slice_index is not None)
        )
        if not slice_indexes:
            return ObjectInstanceKey.domain(range(1, max_object_id + 1))
        return tuple(
            key
            for slice_index in slice_indexes
            for key in ObjectInstanceKey.domain(
                range(1, max_object_id + 1),
                slice_index=slice_index,
            )
        )


@dataclass(frozen=True, slots=True)
class ObjectLabelInstanceDomains:
    """Typed object-instance domains keyed by object-label artifact name."""

    domains_by_name: Mapping[str, tuple[ObjectInstanceKey, ...]]

    @classmethod
    def from_named_plane_domains(
        cls,
        named_plane_domains: Iterable[tuple[str, tuple[tuple[int, ...], ...]]],
    ) -> "ObjectLabelInstanceDomains":
        """Build domains from per-plane object-id domains."""
        domains: dict[str, dict[ObjectInstanceKey, None]] = {}
        for object_name, plane_domains in named_plane_domains:
            slice_indexes = (
                (None,)
                if len(plane_domains) <= 1
                else tuple(range(len(plane_domains)))
            )
            domain = domains.setdefault(str(object_name), {})
            for slice_index, object_ids in zip(slice_indexes, plane_domains, strict=True):
                for object_id in object_ids:
                    domain[ObjectInstanceKey(object_id, slice_index=slice_index)] = None
        return cls(
            {
                object_name: cls._ordered_keys(domain)
                for object_name, domain in domains.items()
            }
        )

    def for_name(self, object_name: str) -> tuple[ObjectInstanceKey, ...]:
        """Return the typed object-instance domain for one object-label name."""
        return self.domains_by_name.get(object_name, ())

    @staticmethod
    def _ordered_keys(
        keys: Mapping[ObjectInstanceKey, None],
    ) -> tuple[ObjectInstanceKey, ...]:
        return tuple(
            sorted(
                keys,
                key=lambda key: (
                    key.slice_index is None,
                    key.slice_index or -1,
                    key.object_id,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class SourceSpatialDomainProjection:
    """Projection from source-image dense coordinates back to native XY shape."""

    domain: SourceSpatialDomain
    shape_yx: tuple[int, int]

    @classmethod
    def from_adapter(
        cls,
        adapter: SourceSpatialDomainAdapter,
    ) -> "SourceSpatialDomainProjection":
        import numpy as np

        array = np.asarray(adapter.array)
        if array.ndim < 2:
            raise ValueError(
                "Source-spatial object-label projection requires at least two "
                f"dimensions, got {array.ndim}."
            )
        return cls(adapter.domain, tuple(int(axis) for axis in array.shape[-2:]))

    def restore(self, value: Any) -> Any:
        import numpy as np

        array = np.asarray(value)
        if array.ndim < 2 or tuple(array.shape[-2:]) == self.shape_yx:
            return value
        if self.domain.origin_yx is None:
            return value
        source_shape_yx = self.domain.source_shape_yx
        if source_shape_yx is None or tuple(array.shape[-2:]) != tuple(source_shape_yx):
            raise ValueError(
                "Cannot restore source-domain object labels with shape "
                f"{array.shape[-2:]} to native shape {self.shape_yx}."
            )
        origin_y, origin_x = self.domain.origin_yx
        height, width = self.shape_yx
        return array[..., origin_y : origin_y + height, origin_x : origin_x + width]


@dataclass(frozen=True, slots=True)
class DenseObjectLabelProjectionAlignment(metaclass=AutoRegisterMeta):
    """Shared source-to-native projection metadata for dense label alignment."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True

    first_projection: SourceSpatialDomainProjection | None
    second_projection: SourceSpatialDomainProjection | None


@dataclass(frozen=True, slots=True)
class DenseObjectLabelPairAlignment(DenseObjectLabelProjectionAlignment):
    """Aligned dense label arrays with source-to-native projection metadata."""

    first: Any
    second: Any


@dataclass(frozen=True, slots=True)
class SourceSpatialAlignmentValue:
    """One dense label value with optional source-spatial semantics."""

    value: Any
    adapter: SourceSpatialDomainAdapter | None = None

    @classmethod
    def from_value(cls, value: Any) -> "SourceSpatialAlignmentValue":
        return cls(value, SourceSpatialDomainAdapter.for_value(value))

    @property
    def array(self) -> Any:
        if self.adapter is not None:
            return self.adapter.array
        return self.value

    @property
    def native_shape_yx(self) -> tuple[int, int] | None:
        import numpy as np

        array = np.asarray(self.array)
        if array.ndim < 2:
            return None
        return tuple(int(axis) for axis in array.shape[-2:])

    @property
    def source_domain(self) -> SourceSpatialDomain | None:
        if self.adapter is None:
            return None
        if self.adapter.domain.source_shape_yx is None:
            return None
        return self.adapter.domain

    def shares_native_shape(self, other: "SourceSpatialAlignmentValue") -> bool:
        return (
            self.native_shape_yx is not None
            and other.native_shape_yx is not None
            and self.native_shape_yx == other.native_shape_yx
        )

    def materialize(self) -> Any:
        if self.adapter is not None:
            return self.adapter.materialize()
        return self.value

    def materialize_in_domain(self, domain: SourceSpatialDomain) -> Any:
        return domain.materialize(self.array)

    def projection(self) -> SourceSpatialDomainProjection | None:
        if self.adapter is None:
            return None
        return SourceSpatialDomainProjection.from_adapter(self.adapter)

    def projection_in_domain(
        self,
        domain: SourceSpatialDomain,
    ) -> SourceSpatialDomainProjection | None:
        shape_yx = self.native_shape_yx
        if shape_yx is None:
            return None
        return SourceSpatialDomainProjection(domain, shape_yx)


@dataclass(frozen=True, slots=True)
class SourceSpatialAlignmentPair:
    """Pairwise source-domain alignment for dense label arrays."""

    values: tuple[SourceSpatialAlignmentValue, SourceSpatialAlignmentValue]

    @classmethod
    def from_values(
        cls,
        first: Any,
        second: Any,
    ) -> "SourceSpatialAlignmentPair":
        return cls(
            (
                SourceSpatialAlignmentValue.from_value(first),
                SourceSpatialAlignmentValue.from_value(second),
            )
        )

    def aligned(self) -> DenseObjectLabelPairAlignment:
        source_domain = self.shared_source_domain_for_native_pair()
        if source_domain is not None:
            aligned_values = tuple(
                value.materialize_in_domain(source_domain) for value in self.values
            )
            projections = tuple(
                value.projection_in_domain(source_domain) for value in self.values
            )
        else:
            aligned_values = tuple(value.materialize() for value in self.values)
            projections = tuple(value.projection() for value in self.values)
        return DenseObjectLabelPairAlignment(
            first=aligned_values[0],
            second=aligned_values[1],
            first_projection=projections[0],
            second_projection=projections[1],
        )

    def shared_source_domain_for_native_pair(self) -> SourceSpatialDomain | None:
        source_domains = tuple(
            value.source_domain
            for value in self.values
            if value.source_domain is not None
        )
        if len(source_domains) != 1:
            return None
        first, second = self.values
        if not first.shares_native_shape(second):
            return None
        return source_domains[0]


@dataclass(frozen=True, slots=True)
class DenseObjectLabelStackAlignment(DenseObjectLabelProjectionAlignment):
    """Aligned dense label stacks with native-domain restoration hooks."""

    first_stack: Any
    second_stack: Any

    def restore_first_stack(self, labels: Any) -> Any:
        if self.first_projection is None:
            return labels
        return self.first_projection.restore(labels)

    def restore_second_stack(self, labels: Any) -> Any:
        if self.second_projection is None:
            return labels
        return self.second_projection.restore(labels)


@dataclass(frozen=True, slots=True)
class DenseObjectLabelMaskStackAlignment:
    """Aligned dense object-label and mask stacks over the runtime-slice axis."""

    label_stack: Any
    mask_stack: Any
    label_projection: SourceSpatialDomainProjection | None = None

    def restore_label_stack(self, labels: Any) -> Any:
        if self.label_projection is None:
            return labels
        return self.label_projection.restore(labels)


@dataclass(frozen=True, slots=True)
class DenseObjectLabelPairAligner:
    """Align two dense object-label payloads before pairwise semantics."""

    first_labels: Any
    second_labels: Any

    def aligned(self) -> tuple[Any, Any]:
        alignment = self.alignment()
        return alignment.first, alignment.second

    def alignment(self) -> DenseObjectLabelPairAlignment:
        alignment = SourceSpatialAlignmentPair.from_values(
            self.first_labels,
            self.second_labels,
        ).aligned()
        first, second = alignment.first, alignment.second
        first = DenseObjectLabelStack.from_labels(first).collapse_singleton_plane()
        second = DenseObjectLabelStack.from_labels(second).collapse_singleton_plane()
        if first.shape == second.shape:
            return DenseObjectLabelPairAlignment(
                first=first,
                second=second,
                first_projection=alignment.first_projection,
                second_projection=alignment.second_projection,
            )

        if first.ndim == 3 and second.ndim == 2 and first.shape[1:] == second.shape:
            first = DenseObjectLabelStack.from_labels(
                first
            ).project_xy_plane_without_relabeling()
        if second.ndim == 3 and first.ndim == 2 and second.shape[1:] == first.shape:
            second = DenseObjectLabelStack.from_labels(
                second
            ).project_xy_plane_without_relabeling()
        if first.shape != second.shape:
            raise ValueError(
                "Dense object-label payloads must share a common geometry after "
                f"alignment; got {first.shape} and {second.shape}."
            )
        return DenseObjectLabelPairAlignment(
            first=first,
            second=second,
            first_projection=alignment.first_projection,
            second_projection=alignment.second_projection,
        )

    def aligned_stacks(self, slice_count: int) -> tuple[Any, Any] | None:
        alignment = self.aligned_stack_context(slice_count)
        if alignment is None:
            return None
        return alignment.first_stack, alignment.second_stack

    def aligned_stack_context(
        self,
        slice_count: int,
    ) -> DenseObjectLabelStackAlignment | None:
        alignment = self.alignment()
        first_stack = self._stack_view(alignment.first, slice_count)
        second_stack = self._stack_view(alignment.second, slice_count)
        if first_stack is None or second_stack is None:
            return None
        if first_stack.shape != second_stack.shape:
            raise ValueError(
                "Dense object-label stacks must share a common geometry after "
                f"alignment; got {first_stack.shape} and {second_stack.shape}."
            )
        return DenseObjectLabelStackAlignment(
            first_stack=first_stack,
            second_stack=second_stack,
            first_projection=alignment.first_projection,
            second_projection=alignment.second_projection,
        )

    @staticmethod
    def _stack_view(value: Any, slice_count: int) -> Any | None:
        import numpy as np

        array = np.asarray(value, dtype=np.int32)
        if array.ndim == 3 and array.shape[0] == slice_count:
            return np.ascontiguousarray(array)
        if array.ndim == 2:
            return np.ascontiguousarray(np.broadcast_to(array, (slice_count, *array.shape)))
        return None


@dataclass(frozen=True, slots=True)
class DenseObjectLabelMaskAligner:
    """Align dense object labels with an image/binary mask in source geometry."""

    labels: Any
    mask: Any

    def aligned(self) -> tuple[Any, Any]:
        alignment = SourceSpatialAlignmentPair.from_values(
            self.labels,
            self.mask,
        ).aligned()
        labels = DenseObjectLabelStack.from_labels(
            alignment.first,
        ).collapse_singleton_plane()
        mask = self._collapse_singleton_mask_plane(alignment.second)

        if labels.shape == mask.shape:
            return labels, mask

        if labels.ndim == 3 and mask.ndim == 2 and labels.shape[1:] == mask.shape:
            labels = DenseObjectLabelStack.from_labels(
                labels
            ).project_xy_plane_without_relabeling()
        if mask.ndim == 3 and labels.ndim == 2 and mask.shape[1:] == labels.shape:
            mask = self._project_mask_stack(mask)
        if labels.shape != mask.shape:
            raise ValueError(
                "Dense object labels and mask must share a common geometry after "
                f"alignment; got {labels.shape} and {mask.shape}."
            )
        return labels, mask

    def aligned_stack_context(
        self,
        slice_count: int,
    ) -> DenseObjectLabelMaskStackAlignment | None:
        alignment = SourceSpatialAlignmentPair.from_values(
            self.labels,
            self.mask,
        ).aligned()
        label_stack = self._stack_view(alignment.first, slice_count)
        mask_stack = self._stack_view(alignment.second, slice_count)
        if label_stack is None or mask_stack is None:
            return None
        if label_stack.shape != mask_stack.shape:
            raise ValueError(
                "Dense object-label and mask stacks must share a common geometry "
                f"after alignment; got {label_stack.shape} and {mask_stack.shape}."
            )
        return DenseObjectLabelMaskStackAlignment(
            label_stack,
            mask_stack,
            alignment.first_projection,
        )

    @staticmethod
    def _collapse_singleton_mask_plane(mask: Any) -> Any:
        import numpy as np

        array = np.asarray(mask)
        if array.ndim == 3 and array.shape[0] == 1:
            return array[0]
        return mask

    @staticmethod
    def _project_mask_stack(mask: Any) -> Any:
        import numpy as np

        array = np.asarray(mask)
        positive = array != 0
        conflicts = int(np.count_nonzero(np.count_nonzero(positive, axis=0) > 1))
        if conflicts:
            raise ValueError(
                "Mask stack cannot be projected to one XY plane because "
                f"{conflicts} pixels are positive in multiple planes."
            )
        return np.max(array, axis=0)

    @staticmethod
    def _stack_view(value: Any, slice_count: int) -> Any | None:
        import numpy as np

        array = np.asarray(value)
        if array.ndim == 3 and array.shape[0] == slice_count:
            return np.ascontiguousarray(array)
        if array.ndim == 2:
            return np.ascontiguousarray(np.broadcast_to(array, (slice_count, *array.shape)))
        return None


@dataclass(frozen=True, slots=True)
class DenseObjectLabelStack:
    """Dense object-label stack semantics independent of payload wrapper type."""

    array: np.ndarray

    @classmethod
    def from_labels(cls, labels: Any) -> "DenseObjectLabelStack":
        return cls(np.asarray(labels, dtype=np.int32))

    def collapse_singleton_plane(self) -> np.ndarray:
        if self.array.ndim == 3 and self.array.shape[0] == 1:
            return self.array[0]
        return self.array

    def project_xy_plane_without_relabeling(self) -> np.ndarray:
        stack = self.array
        if stack.ndim != 3:
            return stack
        if stack.shape[0] == 1:
            return stack[0]

        positive = stack > 0
        if not np.any(positive):
            return np.zeros(stack.shape[1:], dtype=np.int32)

        max_label = np.where(positive, stack, 0).max(axis=0)
        sentinel = np.iinfo(np.int32).max
        min_positive = np.where(positive, stack, sentinel).min(axis=0)
        positive_count = np.count_nonzero(positive, axis=0)
        conflicts = (positive_count > 1) & (min_positive != max_label)
        if np.any(conflicts):
            raise ValueError(
                "Cannot project dense object-label stack with conflicting positive "
                "labels at the same XY coordinate."
            )
        return max_label.astype(np.int32, copy=False)


@dataclass(frozen=True, slots=True)
class ConsecutiveObjectLabelIdProjection:
    """Projection from arbitrary positive object IDs to consecutive IDs."""

    positive_label_ids: Any

    @classmethod
    def from_dense_array(cls, labels: np.ndarray) -> "ConsecutiveObjectLabelIdProjection":
        label_array = np.asarray(labels)
        positive_ids = np.unique(label_array[label_array > 0])
        return cls(positive_ids.astype(np.int64, copy=False))

    @property
    def object_count(self) -> int:
        return int(len(self.positive_label_ids))

    @property
    def has_objects(self) -> bool:
        return self.object_count > 0

    def relabel_numpy_array(
        self,
        labels: np.ndarray,
        *,
        dtype: Any | None = None,
    ) -> np.ndarray:
        """Apply this ID projection to one dense NumPy label array."""
        label_array = np.asarray(labels)
        output_dtype = np.dtype(dtype or label_array.dtype)
        if not self.has_objects:
            return np.zeros_like(label_array, dtype=output_dtype)
        if self.labels_are_already_consecutive:
            return label_array.astype(output_dtype, copy=False)
        if self.lookup_table_is_bounded(label_array):
            return self.lookup_table_relabel(label_array, output_dtype)
        return self.searchsorted_relabel(label_array, output_dtype)

    @property
    def labels_are_already_consecutive(self) -> bool:
        return bool(
            np.array_equal(
                self.positive_label_ids,
                np.arange(1, self.object_count + 1, dtype=np.int64),
            )
        )

    def lookup_table_is_bounded(self, labels: Any) -> bool:
        label_array = np.asarray(labels)
        max_label = int(self.positive_label_ids[-1])
        return max_label <= max(label_array.size * 2, self.object_count * 16)

    def lookup_table_relabel(self, labels: np.ndarray, dtype: Any) -> np.ndarray:
        label_array = np.asarray(labels)
        lookup = np.zeros(int(self.positive_label_ids[-1]) + 1, dtype=dtype)
        lookup[self.positive_label_ids] = np.arange(
            1,
            self.object_count + 1,
            dtype=dtype,
        )
        return lookup[label_array]

    def searchsorted_relabel(self, labels: np.ndarray, dtype: Any) -> np.ndarray:
        label_array = np.asarray(labels)
        flat = label_array.reshape(-1)
        remapped = np.zeros(flat.shape, dtype=dtype)
        foreground = flat > 0
        positions = np.searchsorted(self.positive_label_ids, flat[foreground])
        remapped[foreground] = positions.astype(dtype, copy=False) + 1
        return remapped.reshape(label_array.shape)


class DenseObjectLabelConsecutiveRelabelingStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered backend strategy for consecutive dense object-label IDs."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def for_labels(
        cls,
        labels: object,
    ) -> "DenseObjectLabelConsecutiveRelabelingStrategy":
        strategy = cls.for_nominal_value(labels)
        return (
            strategy
            if strategy is not None
            else RawDenseObjectLabelConsecutiveRelabelingStrategy()
        )

    @abstractmethod
    def relabel(self, labels: object, *, dtype: Any | None = None) -> object:
        """Return labels with materially present positive IDs remapped to 1..N."""


class NumpyDenseObjectLabelConsecutiveRelabelingStrategy(
    DenseObjectLabelConsecutiveRelabelingStrategy
):
    """Consecutive-ID relabeling for dense NumPy object-label arrays."""

    value_type = np.ndarray

    def relabel(self, labels: object, *, dtype: Any | None = None) -> np.ndarray:
        if not isinstance(labels, np.ndarray):
            raise TypeError(
                "NumpyDenseObjectLabelConsecutiveRelabelingStrategy requires ndarray, "
                f"got {type(labels).__name__}."
            )
        projection = ConsecutiveObjectLabelIdProjection.from_dense_array(labels)
        return projection.relabel_numpy_array(
            labels,
            dtype=dtype,
        )


class RawDenseObjectLabelConsecutiveRelabelingStrategy(
    DenseObjectLabelConsecutiveRelabelingStrategy
):
    """Compatibility relabeling for legacy dense array-like object labels."""

    def relabel(self, labels: object, *, dtype: Any | None = None) -> np.ndarray:
        label_array = np.asarray(labels)
        return ConsecutiveObjectLabelIdProjection.from_dense_array(
            label_array
        ).relabel_numpy_array(label_array, dtype=dtype)


def dense_object_label_id_domain(
    labels: Any,
    *,
    declared_object_count: int | None = None,
    declared_object_ids: DeclaredObjectIds = None,
) -> tuple[int, ...]:
    """Return the semantic object-id domain represented by dense labels.

    Producers that need object identities without current pixels must declare
    them explicitly. Undeclared dense/sparse payloads use materially present
    positive IDs so sparse or non-contiguous labels do not fabricate phantom
    measurement rows.
    """
    payload_domain = ObjectLabelDomainMetadataStrategy.for_value(
        labels
    ).object_label_domain(labels)
    payload_ids = payload_domain.declared_object_ids
    payload_count = payload_domain.declared_object_count
    resolved_ids = declared_object_ids if declared_object_ids is not None else payload_ids
    if resolved_ids:
        ids = tuple(int(object_id) for object_id in resolved_ids)
        if any(object_id <= 0 for object_id in ids):
            raise ValueError("Object label IDs must be positive integers.")
        return tuple(sorted(dict.fromkeys(ids)))

    resolved_count = (
        declared_object_count
        if declared_object_count is not None
        else payload_count
    )
    if resolved_count is not None:
        count = int(resolved_count)
        if count < 0:
            raise ValueError("declared_object_count cannot be negative.")
        return tuple(range(1, count + 1))
    return ObjectLabelIdDomainStrategy.for_value(labels).present_ids(labels)


def dense_object_label_extent_id_domain(labels: Any) -> tuple[int, ...]:
    """Return the dense positive ID extent materially represented by labels.

    Declared domains describe semantic object identity. Dense measurement
    producers that need CellProfiler-style missing rows should instead use the
    material label extent so gaps inside ``1..max(label)`` become explicit
    missing measurements without fabricating rows for an external grid domain.
    """
    max_present_id = ObjectLabelIdDomainStrategy.for_value(labels).max_present_id(labels)
    return tuple(range(1, max_present_id + 1))


class ObjectLabelDomainDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Nominal declaration for transformed object-label identity domains."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True

    @abstractmethod
    def declared_domain(self, source: Any, labels: Any) -> ObjectLabelDomain:
        """Return the object-label identity domain for transformed labels."""


@dataclass(frozen=True, slots=True)
class ExplicitObjectLabelDomainDeclaration(ObjectLabelDomainDeclaration):
    """Use an explicitly supplied object-label domain."""

    domain: ObjectLabelDomain

    def declared_domain(self, source: Any, labels: Any) -> ObjectLabelDomain:
        del source, labels
        return self.domain


@dataclass(frozen=True, slots=True)
class PreserveSourceObjectLabelDomainDeclaration(ObjectLabelDomainDeclaration):
    """Preserve source object-label declarations across shape-preserving transforms."""

    def declared_domain(self, source: Any, labels: Any) -> ObjectLabelDomain:
        del labels
        return ObjectLabelDomainMetadataStrategy.for_value(source).object_label_domain(
            source
        )


@dataclass(frozen=True, slots=True)
class PresentObjectLabelIdsDomainDeclaration(ObjectLabelDomainDeclaration):
    """Declare only object IDs materially present in transformed labels."""

    def declared_domain(self, source: Any, labels: Any) -> ObjectLabelDomain:
        del source
        return ObjectLabelDomain(
            declared_object_ids=(
                ObjectLabelIdDomainStrategy.for_value(labels).present_ids(labels)
            )
        )


@dataclass(frozen=True, slots=True)
class DenseObjectLabelExtentDomainDeclaration(ObjectLabelDomainDeclaration):
    """Declare the dense positive extent represented by transformed labels."""

    def declared_domain(self, source: Any, labels: Any) -> ObjectLabelDomain:
        del source
        return ObjectLabelDomain(
            declared_object_count=(
                ObjectLabelIdDomainStrategy.for_value(labels).max_present_id(labels)
            )
        )


def dense_object_label_plane_id_domains(
    labels: Any,
    *,
    declared_object_count: int | None = None,
    declared_object_ids: DeclaredObjectIds = None,
    declared_object_id_domains: tuple[tuple[int, ...], ...] = (),
    domain_scope: ObjectLabelDomainScope | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return object-id domains for each dense object-label measurement plane.

    Whole-payload declared domains describe the object-label artifact as a
    whole. For slice stacks, exported object tables are plane-local: each plane
    contributes its own dense label domain rather than repeating the global
    declaration for every slice.
    """
    payload_domain = ObjectLabelDomainMetadataStrategy.for_value(
        labels
    ).object_label_domain(labels).with_runtime_declaration_overrides(
        declared_object_count=declared_object_count,
        declared_object_ids=declared_object_ids,
        declared_object_id_domains=declared_object_id_domains,
    )
    resolved_scope = domain_scope or payload_domain.scope
    return ObjectLabelPlaneDomainStrategy.for_enum_member(
        resolved_scope,
    ).plane_domains(
        labels,
        declared_object_count=(
            declared_object_count
            if declared_object_count is not None
            else payload_domain.declared_object_count
        ),
        declared_object_ids=(
            declared_object_ids
            if declared_object_ids is not None
            else payload_domain.declared_object_ids
        ),
        declared_object_id_domains=payload_domain.declared_object_id_domains,
    )


def dense_object_label_identity_domains(
    labels: Any,
    *,
    declared_object_count: int | None = None,
    declared_object_ids: DeclaredObjectIds = None,
    declared_object_id_domains: tuple[tuple[int, ...], ...] = (),
    domain_scope: ObjectLabelDomainScope | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return object-id domains for object identity rows represented by labels."""
    payload_domain = ObjectLabelDomainMetadataStrategy.for_value(
        labels
    ).object_label_domain(labels).with_runtime_declaration_overrides(
        declared_object_count=declared_object_count,
        declared_object_ids=declared_object_ids,
        declared_object_id_domains=declared_object_id_domains,
    )
    resolved_scope = domain_scope or payload_domain.scope
    return ObjectLabelPlaneDomainStrategy.for_enum_member(
        resolved_scope,
    ).identity_domains(
        labels,
        declared_object_count=(
            declared_object_count
            if declared_object_count is not None
            else payload_domain.declared_object_count
        ),
        declared_object_ids=(
            declared_object_ids
            if declared_object_ids is not None
            else payload_domain.declared_object_ids
        ),
        declared_object_id_domains=payload_domain.declared_object_id_domains,
    )


def dense_array_in_source_spatial_domain(
    value: Any,
    *,
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
    if label_array.shape[-2:] == (source_y, source_x) and origin == (0, 0):
        return label_array

    if label_array.ndim < 2:
        raise ValueError(
            f"{value_name} spatial domains require at least 2D arrays; got "
            f"shape {label_array.shape!r}."
        )
    if origin_y + label_array.shape[-2] > source_y or origin_x + label_array.shape[-1] > source_x:
        raise ValueError(
            f"{value_name} crop exceeds its declared source domain; got array "
            f"{label_array.shape!r}, source={source_shape!r}, origin={origin!r}."
        )

    expanded_shape = (*label_array.shape[:-2], source_y, source_x)
    expanded = np.full(expanded_shape, fill_value, dtype=label_array.dtype)
    expanded[
        ...,
        origin_y : origin_y + label_array.shape[-2],
        origin_x : origin_x + label_array.shape[-1],
    ] = label_array
    return expanded


def object_label_parent_child_payload(
    parent_labels: Any,
    child_labels: Any,
    *,
    child_region_labels: Any | None = None,
    kernel: ObjectRelationshipPayloadKernel = DEFAULT_OBJECT_RELATIONSHIP_PAYLOAD_KERNEL,
) -> ParentChildRelationshipPayload:
    """Derive parent-child ids from nominal object-label representations.

    ``child_region_labels`` lets callers use one label image to enumerate child
    ids while selecting the pixels that define each child's parent context.
    """
    import numpy as np

    if child_region_labels is None:
        request = ObjectRelationshipPayloadRequest(
            parent_labels=parent_labels,
            child_labels=child_labels,
            kernel=kernel,
        )
        return ObjectRelationshipPayloadStrategy.for_context(request).payload(request)
    else:
        parent_array, context_array = DenseObjectLabelPairAligner(
            parent_labels,
            child_region_labels,
        ).aligned()
        child_array, context_array = DenseObjectLabelPairAligner(
            child_labels,
            context_array,
        ).aligned()

    child_ids_array, parent_ids_array = kernel.dominant_parent_ids_by_child(
        parent_array,
        child_array,
        context_array,
    )
    return ParentChildRelationshipPayload(
        parent_ids=tuple(int(parent_id) for parent_id in parent_ids_array),
        child_ids=tuple(int(child_id) for child_id in child_ids_array),
    )


def object_label_lineage_payload(
    parent_labels: Any,
    child_labels: Any,
) -> ParentChildRelationshipPayload:
    """Derive typed parent-child lineage for object-label transforms.

    Shared-geometry transforms use spatial dominance. Geometry-changing
    transforms use preserved object ids, which is the only nominal identity that
    survives nearest-neighbor label resizing without inventing spatial overlap.
    """
    geometry = object_label_lineage_geometry(parent_labels, child_labels)
    return ObjectLabelLineageStrategy.for_enum_member(geometry).payload(
        parent_labels,
        child_labels,
    )


def object_label_lineage_geometry(
    parent_labels: Any,
    child_labels: Any,
) -> ObjectLabelLineageGeometry:
    """Classify the geometry contract for object-label lineage derivation."""
    try:
        DenseObjectLabelPairAligner(parent_labels, child_labels).aligned()
    except ValueError:
        return ObjectLabelLineageGeometry.IDENTITY_DOMAIN
    return ObjectLabelLineageGeometry.SHARED_GEOMETRY


@dataclass(frozen=True, slots=True)
class RelationshipSemantics:
    """Directed relationship semantics between two named runtime entities."""

    source: RelationshipEndpoint
    target: RelationshipEndpoint
    relationship_type: str = "related"

    def __post_init__(self) -> None:
        _require_name(
            self.relationship_type,
            "RelationshipSemantics.relationship_type",
        )
        if not isinstance(self.source, RelationshipEndpoint):
            raise TypeError(
                "RelationshipSemantics.source must be RelationshipEndpoint, "
                f"got {type(self.source).__name__}."
            )
        if not isinstance(self.target, RelationshipEndpoint):
            raise TypeError(
                "RelationshipSemantics.target must be RelationshipEndpoint, "
                f"got {type(self.target).__name__}."
            )

    @classmethod
    def parent_child(
        cls,
        parent_name: str,
        child_name: str,
        *,
        parent_kind: ArtifactKind = ArtifactKind.OBJECT_LABELS,
        child_kind: ArtifactKind = ArtifactKind.OBJECT_LABELS,
    ) -> "RelationshipSemantics":
        """Return standard parent-child semantics between two runtime entities."""
        return cls(
            source=RelationshipEndpoint(
                parent_name,
                role=PARENT_RELATIONSHIP_ROLE,
                id_field=PARENT_RELATIONSHIP_ID_FIELD,
                kind=parent_kind,
            ),
            target=RelationshipEndpoint(
                child_name,
                role=CHILD_RELATIONSHIP_ROLE,
                id_field=CHILD_RELATIONSHIP_ID_FIELD,
                kind=child_kind,
            ),
            relationship_type=PARENT_CHILD_RELATIONSHIP_TYPE,
        )


def coerce_enum(enum_type: type[Enum], value: Any, field_name: str) -> Any:
    """Normalize string-backed enum inputs while keeping validation centralized."""
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of "
            f"{', '.join(member.value for member in enum_type)}; got {value!r}."
        ) from exc


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
