"""Generic semantic contracts for typed runtime artifacts."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import (
    asdict,
    dataclass,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from enum import Enum
from functools import lru_cache
import math
import re
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast
from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
from openhcs.core.alias_property import AliasProperty
import numpy as np
from openhcs.core.artifacts import (
    ArtifactPayloadShape,
    ArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.source_spatial_domain import (
    SourceSpatialDomainAdapter,
    SourceSpatialDomain,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    GeneratedLeafClassSpec,
    MostDerivedContextStrategyMixin,
    NominalTypeKeyedStrategyMixin,
    str_enum_member_with_payload,
)
from openhcs.core.process_local_cache import RegisteredProcessLocalBoundedCache

if TYPE_CHECKING:
    from openhcs.core.runtime_values import (
        ObjectLabelMeasurementSource,
        ObjectLabelPayload,
        ObjectLabelSet,
        ObjectLabelValue,
        RuntimeArrayData,
        SparseIJVLabelRows,
    )
DeclaredObjectIds = tuple[int, ...] | list[int] | None


class RuntimeSliceProjectableValue(ABC):
    """Nominal contract for values that own runtime-slice row projection."""

    @abstractmethod
    def project_runtime_slice(self, slice_index: int) -> object:
        """Return the value represented by one runtime-slice index."""


class RuntimeSliceIdentityProjectableValue(ABC):
    """Nominal contract for values that can be stamped with execution-slice identity."""

    @abstractmethod
    def with_runtime_slice_identity(
        self, *, slice_index: int, slice_count: int
    ) -> Self:
        """Return the value with execution-slice identity applied."""


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
            cls, value, payload_attribute="_payload_shape", payload=payload_shape
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
        unique_scopes = tuple(
            dict.fromkeys(
                (coerce_enum(cls, scope, "ObjectLabelDomain.scope") for scope in scopes)
            )
        )
        if len(unique_scopes) == 1:
            return unique_scopes[0]
        return cls.PAYLOAD


RuntimePlaneAxisPlaneIndexResolver = Callable[
    ["RuntimePlaneAxisProjector", tuple[str, ...]], int | None
]
RuntimePlaneAxisSizeResolver = Callable[
    ["RuntimePlaneAxisProjector", tuple[str, ...]], int | None
]


def runtime_slice_axis_plane_index(
    projector: "RuntimePlaneAxisProjector", source_aliases: tuple[str, ...]
) -> int | None:
    """Resolve a runtime-slice axis through the projector."""
    return projector.runtime_slice_plane_index()


def runtime_slice_axis_size(
    projector: "RuntimePlaneAxisProjector", source_aliases: tuple[str, ...]
) -> int | None:
    """Resolve runtime-slice axis cardinality through the projector."""
    del source_aliases
    return projector.runtime_slice_axis_size()


def source_binding_axis_plane_index(
    projector: "RuntimePlaneAxisProjector", source_aliases: tuple[str, ...]
) -> int | None:
    """Resolve a source-binding axis through the projector."""
    return projector.source_binding_axis_plane_index(source_aliases)


def source_binding_axis_size(
    projector: "RuntimePlaneAxisProjector", source_aliases: tuple[str, ...]
) -> int | None:
    """Resolve source-binding axis cardinality through the projector."""
    return projector.source_binding_axis_size(source_aliases)


class RuntimePlaneAxis(str, Enum):
    """Semantic meaning of the leading plane axis on runtime array stacks."""

    def __new__(
        cls,
        value: str,
        plane_index_resolver: RuntimePlaneAxisPlaneIndexResolver,
        axis_size_resolver: RuntimePlaneAxisSizeResolver,
    ):
        member = str_enum_member_with_payload(
            cls,
            value,
            payload_attribute="_plane_index_resolver",
            payload=plane_index_resolver,
        )
        member.__dict__["_axis_size_resolver"] = axis_size_resolver
        return member

    RUNTIME_SLICE = (
        "runtime_slice",
        runtime_slice_axis_plane_index,
        runtime_slice_axis_size,
    )
    SOURCE_BINDING = (
        "source_binding",
        source_binding_axis_plane_index,
        source_binding_axis_size,
    )
    plane_index_resolver = AliasProperty[RuntimePlaneAxisPlaneIndexResolver](
        "_plane_index_resolver"
    )
    axis_size_resolver = AliasProperty[RuntimePlaneAxisSizeResolver](
        "_axis_size_resolver"
    )

    @classmethod
    def common(cls, axes: Any) -> "RuntimePlaneAxis":
        """Return the common plane axis for merged labels."""
        unique_axes = tuple(
            dict.fromkeys((coerce_enum(cls, axis, "RuntimePlaneAxis") for axis in axes))
        )
        if len(unique_axes) != 1:
            raise ValueError(
                f"Cannot merge object-label stacks with different plane-axis semantics: {unique_axes!r}."
            )
        return unique_axes[0]

    def plane_index(
        self, projector: "RuntimePlaneAxisProjector", *, source_aliases: tuple[str, ...]
    ) -> int | None:
        """Resolve this semantic axis against the execution-local projector."""
        return self.plane_index_resolver(projector, source_aliases)

    def axis_size(
        self, projector: "RuntimePlaneAxisProjector", *, source_aliases: tuple[str, ...]
    ) -> int | None:
        """Resolve this semantic axis cardinality against the projector."""
        return self.axis_size_resolver(projector, source_aliases)


class RuntimePlaneAxisSliceProjectionPolicy(
    EnumKeyedStrategyMixin[RuntimePlaneAxis], ABC, metaclass=AutoRegisterMeta
):
    """Policy for axes that can be selected by a runtime slice index."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    __enum_member_attr__ = "axis"
    axis: ClassVar[RuntimePlaneAxis]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def supports_slice_projection(self) -> bool:
        """Return whether this plane axis is addressable by slice index."""


class RuntimeSlicePlaneAxisSliceProjectionPolicy(RuntimePlaneAxisSliceProjectionPolicy):
    """Runtime-slice planes are directly selected by the runtime slice index."""

    axis = RuntimePlaneAxis.RUNTIME_SLICE

    def supports_slice_projection(self) -> bool:
        return True


class SourceBindingPlaneAxisSliceProjectionPolicy(
    RuntimePlaneAxisSliceProjectionPolicy
):
    """Source-binding planes are selected by slice index during source-bound execution."""

    axis = RuntimePlaneAxis.SOURCE_BINDING

    def supports_slice_projection(self) -> bool:
        return True


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
    plane_count: int | None = None

    def __post_init__(self) -> None:
        scope = coerce_enum(
            RuntimePlaneProjectionScope, self.scope, "RuntimePlaneProjection.scope"
        )
        object.__setattr__(self, "scope", scope)
        if scope is RuntimePlaneProjectionScope.STACK:
            if self.plane_index is not None:
                raise ValueError(
                    "Stack runtime-plane projection cannot carry a plane index."
                )
            if self.plane_count is not None:
                raise ValueError(
                    "Stack runtime-plane projection cannot carry a plane count."
                )
            return
        if self.plane_index is None:
            raise ValueError("Grouped runtime-plane projection requires a plane index.")
        plane_index = int(self.plane_index)
        if plane_index < 0:
            raise ValueError(
                "Grouped runtime-plane projection plane_index cannot be negative."
            )
        object.__setattr__(self, "plane_index", plane_index)
        if self.plane_count is None:
            return
        plane_count = int(self.plane_count)
        if plane_count <= 0:
            raise ValueError(
                "Grouped runtime-plane projection plane_count must be positive."
            )
        if plane_index >= plane_count:
            raise ValueError(
                f"Grouped runtime-plane projection plane_index must be within plane_count: index {plane_index}, count {plane_count}."
            )
        object.__setattr__(self, "plane_count", plane_count)

    @classmethod
    def stack(cls) -> "RuntimePlaneProjection":
        """Preserve runtime-slice stacks for stack-scoped execution."""
        return cls(RuntimePlaneProjectionScope.STACK)

    @classmethod
    def group(
        cls, plane_index: int, plane_count: int | None = None
    ) -> "RuntimePlaneProjection":
        """Select one runtime-slice plane for grouped execution."""
        return cls(RuntimePlaneProjectionScope.GROUP, plane_index, plane_count)

    @classmethod
    def for_execution_group(
        cls,
        group_key: str | None,
        *,
        plane_index: int | None,
        plane_count: int | None = None,
        projects_runtime_plane: bool,
    ) -> "RuntimePlaneProjection":
        """Derive validated projection semantics from compiled group identity."""
        if group_key is None:
            if plane_index is not None:
                raise ValueError(
                    "Ungrouped runtime execution cannot carry a plane index."
                )
            if plane_count is not None:
                raise ValueError(
                    "Ungrouped runtime execution cannot carry a plane count."
                )
            if projects_runtime_plane:
                raise ValueError(
                    "Runtime-plane projection requires grouped execution identity."
                )
            return cls.stack()
        if not projects_runtime_plane:
            return cls.stack()
        if plane_index is None:
            raise ValueError(
                "Runtime-plane grouped execution requires the OpenHCS component plane index."
            )
        return cls.group(plane_index, plane_count)

    def runtime_slice_plane_index(self) -> int | None:
        """Return selected runtime-slice plane, or None when stacks are preserved."""
        return self.plane_index

    def runtime_slice_axis_size(self) -> int | None:
        """Return the grouped runtime-slice axis size when known."""
        return self.plane_count


StackRuntimePlaneProjection = RuntimePlaneProjection
GroupRuntimePlaneProjection = RuntimePlaneProjection


class RuntimePlaneAxisProjector(ABC):
    """Nominal provider for execution-local runtime plane selection."""

    @abstractmethod
    def runtime_slice_plane_index(self) -> int | None:
        """Return the execution-local runtime-slice plane index."""

    def runtime_slice_axis_size(self) -> int | None:
        """Return the runtime-slice axis size for the current execution scope."""
        return None

    def source_binding_axis_plane_index(
        self, source_aliases: tuple[str, ...]
    ) -> int | None:
        """Return the execution-local source-binding plane index."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide source-binding plane projection."
        )

    def source_binding_axis_size(self, source_aliases: tuple[str, ...]) -> int | None:
        """Return the source-binding axis size for this execution scope."""
        return None

    def plane_index_for_axis(
        self, request: "RuntimePlaneAxisProjectionRequest"
    ) -> int | None:
        """Return the execution-local plane index for a nominal runtime axis."""
        return request.resolve(self)


@dataclass(frozen=True, slots=True)
class RuntimePlaneAxisProjectionRequest:
    """Nominal runtime-plane axis lookup against a projector."""

    axis: RuntimePlaneAxis
    source_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "axis",
            coerce_enum(
                RuntimePlaneAxis, self.axis, "RuntimePlaneAxisProjectionRequest.axis"
            ),
        )
        object.__setattr__(self, "source_aliases", tuple(self.source_aliases))

    def resolve(self, projector: RuntimePlaneAxisProjector) -> int | None:
        """Resolve this request through the typed projector contract."""
        return self.axis.plane_index(projector, source_aliases=self.source_aliases)


def bounded_runtime_plane_index(
    plane_count: int, plane_index: int | None
) -> int | None:
    """Return a selected runtime plane index only when it is in range."""
    if plane_index is None:
        return None
    if plane_index >= plane_count:
        return None
    return plane_index


@dataclass(frozen=True, slots=True)
class RuntimePlaneAxisValueProjection:
    """Projection of values that explicitly carry a declared runtime plane axis."""

    axis: RuntimePlaneAxis
    source_aliases: tuple[str, ...]
    plane_index: int | None
    axis_size: int

    @classmethod
    def from_projector(
        cls,
        projector: RuntimePlaneAxisProjector | None,
        axis: RuntimePlaneAxis,
        source_aliases: tuple[str, ...],
    ) -> "RuntimePlaneAxisValueProjection | None":
        """Return the runtime-axis projection declared by a runtime projector."""
        if projector is None:
            return None
        if not isinstance(projector, RuntimePlaneAxisProjector):
            raise TypeError(
                f"Runtime plane-axis projection requires RuntimePlaneAxisProjector, got {type(projector).__name__}."
            )
        source_aliases = tuple(source_aliases)
        axis_size = axis.axis_size(projector, source_aliases=source_aliases)
        if axis_size is None:
            return None
        return cls(
            axis=axis,
            source_aliases=source_aliases,
            plane_index=projector.plane_index_for_axis(
                RuntimePlaneAxisProjectionRequest(
                    axis=axis, source_aliases=source_aliases
                )
            ),
            axis_size=axis_size,
        )

    @classmethod
    def from_selected_plane(
        cls, *, axis: RuntimePlaneAxis, plane_index: int, axis_size: int
    ) -> "RuntimePlaneAxisValueProjection":
        """Return a projection whose explicit plane proof is already resolved."""
        return cls(
            axis=axis, source_aliases=(), plane_index=plane_index, axis_size=axis_size
        )

    def project(
        self, value: "RuntimeArrayData | ObjectLabelMeasurementSource"
    ) -> "RuntimeArrayData | ObjectLabelMeasurementSource":
        """Return the selected plane when the value carries this runtime axis."""
        plane_index = self.plane_index_for_value(value)
        if plane_index is None:
            return value
        from openhcs.core.runtime_values import ObjectLabelPayload, ObjectLabelSet

        if isinstance(value, (ObjectLabelPayload, ObjectLabelSet)):
            return self.project_object_labels(value, plane_index)
        return self.project_array_payload(value, plane_index)

    def plane_index_for_value(
        self, value: "RuntimeArrayData | ObjectLabelMeasurementSource"
    ) -> int | None:
        """Return the selected runtime-axis plane for this value."""
        alias_plane_index = self.source_alias_plane_index(value)
        if alias_plane_index is not None:
            return alias_plane_index
        return self.plane_index

    def source_alias_plane_index(
        self, value: "RuntimeArrayData | ObjectLabelMeasurementSource"
    ) -> int | None:
        """Return a source-alias plane index only for a complete alias axis."""
        if self.axis is RuntimePlaneAxis.SOURCE_BINDING:
            from openhcs.core.runtime_values import source_image_context_plane_index

            return source_image_context_plane_index(
                value, self.source_aliases, self.axis_size
            )
        return None

    def project_array_payload(
        self, value: "RuntimeArrayData | ObjectLabelMeasurementSource", plane_index: int
    ) -> "RuntimeArrayData | ObjectLabelMeasurementSource":
        """Project image-like values carrying this leading runtime axis."""
        from openhcs.core.runtime_values import (
            image_payload_data,
            image_payload_slice_context,
        )

        data = image_payload_data(value)
        if not isinstance(data, np.ndarray):
            return value
        if not self.data_carries_axis(data):
            return value
        self.validate_plane_index(plane_index, data.shape)
        return image_payload_slice_context(value, data[plane_index], plane_index)

    def project_object_labels(
        self, value: "ObjectLabelValue", plane_index: int
    ) -> "ObjectLabelValue":
        """Project object labels while preserving their nominal metadata."""
        if value.plane_axis is not self.axis:
            return value
        from openhcs.core.runtime_values import (
            ObjectLabelMeasurementPayloadStrategy,
            ObjectLabelSourcePlaneProjectionRequest,
            object_label_dense_array,
        )

        labels = object_label_dense_array(value)
        if not self.data_carries_axis(labels):
            return value
        self.validate_plane_index(plane_index, labels.shape)
        return ObjectLabelMeasurementPayloadStrategy.for_source(value).materialize(
            value,
            ObjectLabelSourcePlaneProjectionRequest(labels[plane_index], plane_index),
        )

    def data_carries_axis(self, data: np.ndarray) -> bool:
        """Return whether dense data explicitly carries this runtime axis."""
        return data.ndim >= 3 and data.shape[0] == self.axis_size

    @staticmethod
    def validate_plane_index(plane_index: int, shape: tuple[int, ...]) -> None:
        """Validate a selected source-binding plane against dense data shape."""
        if plane_index < 0 or plane_index >= shape[0]:
            raise RuntimeError(
                f"Runtime plane-axis projection produced an out-of-range plane index {plane_index} for shape {shape!r}."
            )


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
                raise ValueError(
                    "ObjectLabelDomain.declared_object_count cannot be negative."
                )
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
                (
                    self._normalize_ids(domain, "declared_object_id_domains")
                    for domain in self.declared_object_id_domains
                )
            ),
        )
        object.__setattr__(
            self,
            "scope",
            coerce_enum(ObjectLabelDomainScope, self.scope, "ObjectLabelDomain.scope"),
        )

    @staticmethod
    def _normalize_ids(
        ids: tuple[int, ...] | list[int], field_name: str
    ) -> tuple[int, ...]:
        normalized = tuple((int(object_id) for object_id in ids))
        if any((object_id <= 0 for object_id in normalized)):
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
                tuple(declared_object_ids) if declared_object_ids is not None else ()
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

    def with_missing_declarations_from(
        self, fallback: "ObjectLabelDomain"
    ) -> "ObjectLabelDomain":
        """Return this domain with absent declarations filled from another domain."""
        return ObjectLabelDomain(
            declared_object_count=(
                self.declared_object_count
                if self.declared_object_count is not None
                else fallback.declared_object_count
            ),
            declared_object_ids=(
                self.declared_object_ids
                if self.declared_object_ids
                else fallback.declared_object_ids
            ),
            declared_object_id_domains=(
                self.declared_object_id_domains
                if self.declared_object_id_domains
                else fallback.declared_object_id_domains
            ),
            scope=fallback.scope,
        )

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
            declared_object_id_domains=self.declared_object_id_domains
            or declared_object_id_domains,
            scope=self.scope,
        )

    def with_scope(self, scope: ObjectLabelDomainScope) -> "ObjectLabelDomain":
        """Return this object-label declaration with the requested domain scope."""
        normalized_scope = coerce_enum(
            ObjectLabelDomainScope, scope, "ObjectLabelDomain.scope"
        )
        if (
            normalized_scope is ObjectLabelDomainScope.PAYLOAD
            and self.scope is ObjectLabelDomainScope.PLANE
            and self.declared_object_id_domains
        ):
            declared_object_ids = tuple(
                sorted(
                    {
                        object_id
                        for domain in self.declared_object_id_domains
                        for object_id in domain
                    }
                )
            )
            return ObjectLabelDomain(
                declared_object_count=self.declared_object_count,
                declared_object_ids=declared_object_ids,
                scope=normalized_scope,
            )
        return ObjectLabelDomain(
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            scope=normalized_scope,
        )

    @classmethod
    def explicit_plane_id_domains(
        cls, domains: Iterable["ObjectLabelDomain"]
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
                        "Cannot combine declared and undeclared object-label plane domains."
                    )
                plane_domains.append(())
                continue
            saw_declared = True
            plane_domains.append(id_domain)
        if not saw_declared:
            return ()
        if any((not domain for domain in plane_domains)):
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
                f"Object-label slice_index {normalized_index} is outside slice_count {normalized_count}."
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
                f"Plane-scoped object-label domains must match PURE_2D slice count: {len(self.declared_object_id_domains)} domains for {normalized_count} slices."
            )
        return self.project_planes((normalized_index,))

    def project_planes(self, plane_indices: Iterable[int]) -> "ObjectLabelDomain":
        """Return the object-label domain carried by selected plane indexes."""
        normalized_indices = tuple((int(index) for index in plane_indices))
        if not self.declared_object_id_domains:
            return self
        if not normalized_indices:
            domains: tuple[tuple[int, ...], ...] = ()
        elif len(self.declared_object_id_domains) == 1:
            domains = (self.declared_object_id_domains[0],)
        elif any(
            (
                index < 0 or index >= len(self.declared_object_id_domains)
                for index in normalized_indices
            )
        ):
            raise ValueError(
                f"Object-label plane projection index is outside declared domain count {len(self.declared_object_id_domains)}: {normalized_indices!r}."
            )
        else:
            domains = tuple(
                (self.declared_object_id_domains[index] for index in normalized_indices)
            )
        if len(domains) == 1:
            return ObjectLabelDomain.declared(
                scope=ObjectLabelDomainScope.PLANE, declared_object_ids=domains[0]
            )
        return ObjectLabelDomain.declared(
            scope=ObjectLabelDomainScope.PLANE, declared_object_id_domains=domains
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeObjectMeasurementQuery(ABC):
    """Store-stable identity for object measurement queries."""

    group_key: str | None
    object_name: str
    feature_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "object_name", self.required_name("object_name", self.object_name)
        )
        object.__setattr__(
            self, "feature_name", self.required_name("feature_name", self.feature_name)
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
            ObjectLabelDomain._normalize_ids(
                tuple(self.object_domain), "object_domain"
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeObjectImageMeasurementQuery(RuntimeObjectMeasurementQuery):
    """Object measurement query scoped by CellProfiler image number."""

    image_number: int | None = None

    def __post_init__(self) -> None:
        RuntimeObjectMeasurementQuery.__post_init__(self)
        if self.image_number is not None:
            object.__setattr__(self, "image_number", int(self.image_number))


@dataclass(frozen=True, slots=True)
class RuntimeObjectLabelMeasurementQuery(RuntimeObjectImageMeasurementQuery):
    """Store-stable identity for label-aligned object measurement queries."""

    axis_id: str
    label_domain: tuple[int, ...]

    def __post_init__(self) -> None:
        RuntimeObjectImageMeasurementQuery.__post_init__(self)
        object.__setattr__(self, "axis_id", str(self.axis_id))
        object.__setattr__(
            self,
            "label_domain",
            ObjectLabelDomain._normalize_ids(tuple(self.label_domain), "label_domain"),
        )


class ObjectLabelDomainMetadata(ABC, metaclass=AutoRegisterMeta):
    """Nominal provider for object-label ID domain metadata."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True

    @abstractmethod
    def object_label_domain(self) -> ObjectLabelDomain:
        """Return the declared object-label identity domain."""


class ObjectLabelDomainMetadataStrategy(
    NominalTypeKeyedStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Registered extractor for nominal object-label domain metadata."""

    value_type: ClassVar[type[object] | tuple[type[object], ...] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def for_value(cls, value: object) -> "ObjectLabelDomainMetadataStrategy":
        strategy = cls.for_nominal_value(value)
        return (
            strategy if strategy is not None else RawObjectLabelDomainMetadataStrategy()
        )

    @abstractmethod
    def object_label_domain(self, value: object) -> ObjectLabelDomain:
        """Return the declared object-label identity domain for ``value``."""


class NominalObjectLabelDomainMetadataStrategy(ObjectLabelDomainMetadataStrategy):
    """Use the domain declared by a nominal object-label domain provider."""

    value_type = ObjectLabelDomainMetadata

    def object_label_domain(self, value: object) -> ObjectLabelDomain:
        if not isinstance(value, ObjectLabelDomainMetadata):
            raise TypeError(
                f"NominalObjectLabelDomainMetadataStrategy requires ObjectLabelDomainMetadata, got {type(value).__name__}."
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
            self.object_ids, "ObjectLabelMeasurementValues.object_ids"
        )
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if len(object_ids) != values.size:
            raise ValueError(
                f"ObjectLabelMeasurementValues requires one value per object ID, got {len(object_ids)} IDs and {values.size} values."
            )
        object.__setattr__(self, "object_ids", object_ids)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_label_indexed_values(
        cls, object_ids: Iterable[int], values: Any
    ) -> "ObjectLabelMeasurementValues":
        """Bind dense label-indexed values where index ``label_id - 1``."""
        normalized_ids = ObjectLabelDomain._normalize_ids(
            tuple(object_ids), "ObjectLabelMeasurementValues.object_ids"
        )
        source_values = np.asarray(values, dtype=np.float64).reshape(-1)
        bound_values = np.array(
            [
                (
                    source_values[object_id - 1]
                    if object_id - 1 < source_values.size
                    else np.nan
                )
                for object_id in normalized_ids
            ],
            dtype=np.float64,
        )
        return cls(normalized_ids, bound_values)

    @classmethod
    def from_positional_values(
        cls, object_ids: Iterable[int], values: Any
    ) -> "ObjectLabelMeasurementValues":
        """Bind values that are already ordered like ``object_ids``."""
        normalized_ids = ObjectLabelDomain._normalize_ids(
            tuple(object_ids), "ObjectLabelMeasurementValues.object_ids"
        )
        source_values = np.asarray(values, dtype=np.float64).reshape(-1)
        bound_values = np.full(len(normalized_ids), np.nan, dtype=np.float64)
        copied = min(source_values.size, bound_values.size)
        if copied:
            bound_values[:copied] = source_values[:copied]
        return cls(normalized_ids, bound_values)

    @classmethod
    def from_value_mapping(
        cls, object_ids: Iterable[int], values_by_object_id: Mapping[int, float]
    ) -> "ObjectLabelMeasurementValues":
        """Bind sparse object-id keyed values to an explicit object domain."""
        normalized_ids = ObjectLabelDomain._normalize_ids(
            tuple(object_ids), "ObjectLabelMeasurementValues.object_ids"
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
            (
                object_id
                for object_id, hit in zip(self.object_ids, hits, strict=True)
                if bool(hit)
            )
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
        self, *, max_label: int | None = None, fill_value: float = np.nan
    ) -> np.ndarray:
        """Return values as a dense ``label_id - 1`` indexed vector."""
        largest_id = max(self.object_ids, default=0)
        output_size = max(largest_id, int(max_label or 0))
        output = np.full(output_size, fill_value, dtype=np.float64)
        for object_id, value in zip(self.object_ids, self.values, strict=True):
            output[object_id - 1] = value
        return output


@dataclass(frozen=True, slots=True)
class ObjectMeasurementVectorDomain:
    """Bind object-measurement vectors to the current object-label domain."""

    labels: Any
    value_slices: tuple[Any, ...]

    @property
    def label_slices(self) -> tuple[Any, ...]:
        label_array = np.asarray(self.labels)
        if label_array.ndim <= 2:
            return (self.labels,)
        from openhcs.core.runtime_slice_projection import (
            RuntimeProjectionAxis,
            RuntimeSliceProjection,
        )

        slice_count = int(label_array.shape[0])
        return tuple(
            (
                RuntimeSliceProjection.value_for_slice(
                    self.labels, RuntimeProjectionAxis(index, slice_count)
                )
                for index in range(slice_count)
            )
        )

    @property
    def aligned_value_slices(self) -> tuple[np.ndarray, ...]:
        label_slices = self.label_slices
        value_slices = tuple(self.value_slices)
        if len(value_slices) != len(label_slices):
            return tuple((np.asarray(values) for values in value_slices))
        return tuple(
            (
                self.aligned_values(values, label_slice)
                for values, label_slice in zip(value_slices, label_slices, strict=True)
            )
        )

    @staticmethod
    def aligned_values(values: Any, label_slice: Any) -> np.ndarray:
        value_array = np.asarray(values, dtype=np.float64).reshape(-1)
        object_ids = dense_object_label_id_domain(label_slice)
        if value_array.size == len(object_ids):
            return value_array
        return ObjectLabelMeasurementValues.from_positional_values(
            object_ids, value_array
        ).values


class ObjectLabelIdDomainStrategy(
    NominalTypeKeyedStrategyMixin, ABC, metaclass=AutoRegisterMeta
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
        dense_integer_domain = DenseIntegerObjectLabelIdDomain.from_array(label_array)
        if dense_integer_domain is not None:
            return dense_integer_domain.present_ids()
        return tuple(
            (int(object_id) for object_id in np.unique(label_array) if object_id > 0)
        )


@dataclass(frozen=True, slots=True)
class DenseIntegerObjectLabelIdDomain:
    """Exact present-ID extractor for bounded nonnegative dense integer labels."""

    labels: np.ndarray
    max_label: int

    @classmethod
    def from_array(cls, labels: np.ndarray) -> "DenseIntegerObjectLabelIdDomain | None":
        if not (
            np.issubdtype(labels.dtype, np.integer)
            or np.issubdtype(labels.dtype, np.bool_)
        ):
            return None
        if labels.size == 0:
            return None
        min_label = int(labels.min())
        if min_label < 0:
            return None
        max_label = int(labels.max())
        if max_label > labels.size:
            return None
        return cls(labels=labels, max_label=max_label)

    def present_ids(self) -> tuple[int, ...]:
        if self.max_label <= 0:
            return ()
        counts = np.bincount(
            np.asarray(self.labels, dtype=np.int64).ravel(),
            minlength=self.max_label + 1,
        )
        present_ids = np.flatnonzero(counts)
        return tuple((int(label_id) for label_id in present_ids if label_id > 0))


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
    EnumKeyedStrategyMixin[ObjectLabelDomainScope], ABC, metaclass=AutoRegisterMeta
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
            if len(declared_object_id_domains) != plane_count:
                raise ValueError(
                    f"Plane-scoped object-label domains must match dense label plane count: {len(declared_object_id_domains)} domains for {plane_count} planes."
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
        return tuple((dense_object_label_id_domain(plane) for plane in label_array))


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


@dataclass(frozen=True, slots=True)
class MeasurementScopeSelection:
    """A closed set of measurement scopes selected for one runtime operation."""

    scopes: frozenset[MeasurementScope]

    def __post_init__(self) -> None:
        scopes = frozenset(
            (
                coerce_enum(MeasurementScope, scope, "MeasurementScopeSelection.scopes")
                for scope in self.scopes
            )
        )
        if not scopes:
            raise ValueError("MeasurementScopeSelection.scopes cannot be empty.")
        object.__setattr__(self, "scopes", scopes)

    @classmethod
    def of(cls, *scopes: MeasurementScope | str) -> "MeasurementScopeSelection":
        """Return a selection for one or more measurement scopes."""
        return cls(
            frozenset(
                (
                    coerce_enum(MeasurementScope, scope, "MeasurementScope")
                    for scope in scopes
                )
            )
        )

    def includes(self, scope: MeasurementScope | str) -> bool:
        """Return whether this selection includes one semantic scope."""
        return coerce_enum(MeasurementScope, scope, "MeasurementScope") in self.scopes

    def includes_all(self, *scopes: MeasurementScope | str) -> bool:
        """Return whether this selection includes every supplied semantic scope."""
        return all((self.includes(scope) for scope in scopes))


class RuntimeMeasurementFeatureRelation(ABC):
    """Polymorphic relation declared by a runtime measurement feature member."""

    @abstractmethod
    def source_family_names(
        self, source_feature: "RuntimeMeasurementFeature"
    ) -> tuple[str, ...]:
        """Return feature-family names that select the relation source."""

    @abstractmethod
    def target_family_name(
        self,
        source_feature: "RuntimeMeasurementFeature",
        source_family_name: str,
        feature_type: type["RuntimeMeasurementFeature"],
    ) -> str | None:
        """Return the target family for one selected source family."""


class RuntimeMeasurementFeatureSemanticMarker(ABC):
    """Nominal marker carried by a runtime measurement feature member."""

    family_qualifier: ClassVar[str | None] = None

    @classmethod
    def matches_feature(cls, feature: "RuntimeMeasurementFeature") -> bool:
        """Return whether ``feature`` carries this semantic marker."""
        return any(
            issubclass(marker_type, cls) for marker_type in feature.semantic_markers
        )

    @classmethod
    def qualified_family(cls, feature: "RuntimeMeasurementFeature") -> str:
        """Return a marker-qualified feature family name."""
        if cls.family_qualifier is None:
            raise ValueError(f"{cls.__name__} does not declare family_qualifier.")
        return normalize_runtime_identifier(f"{cls.family_qualifier}_{feature.value}")

    @classmethod
    def requires_sparse_boundary_object_count_stability(cls) -> bool:
        """Return whether sparse-boundary comparison is gated by object count."""
        return True


class ObjectCountFeatureMarker(RuntimeMeasurementFeatureSemanticMarker):
    """Generic marker for object-count measurement features."""

    family_qualifier = "count"


class ObjectIdentifierFeatureMarker(RuntimeMeasurementFeatureSemanticMarker):
    """Generic marker for object-identifier measurement features."""

    family_qualifier = "identifier"


class MeasuredObjectAnchorFeatureMarker(RuntimeMeasurementFeatureSemanticMarker):
    """Generic marker for features proving that an object row was measured."""

    family_qualifier = "object"


class ObjectLocationFeatureMarker(RuntimeMeasurementFeatureSemanticMarker):
    """Generic marker for object-location measurement features."""

    family_qualifier = "location"


class ObjectIntensityFeatureMarker(RuntimeMeasurementFeatureSemanticMarker):
    """Generic marker for object-intensity measurement features."""

    family_qualifier = "intensity"


class ObjectCalculatedFeatureMarker(RuntimeMeasurementFeatureSemanticMarker):
    """Generic marker for calculated object measurement features."""

    family_qualifier = "calculated"

    @classmethod
    def requires_sparse_boundary_object_count_stability(cls) -> bool:
        """Calculated object aggregates may gain or lose missing boundary rows."""
        return False


class ObjectShapeDescriptorFeatureMarker(RuntimeMeasurementFeatureSemanticMarker):
    """Generic marker for object shape-descriptor measurement features."""

    family_qualifier = "shape"


class RuntimeMeasurementIndexedDescriptorDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Registered parser/renderer declaration for indexed feature names."""

    __registry_key__ = "descriptor_key"
    __skip_if_no_key__ = True
    descriptor_key: ClassVar[str | None] = None

    @classmethod
    def require_registered(
        cls,
        declaration_type: type["RuntimeMeasurementIndexedDescriptorDeclaration"],
    ) -> type["RuntimeMeasurementIndexedDescriptorDeclaration"]:
        """Return a registered indexed-descriptor declaration type."""
        if not isinstance(declaration_type, type) or not issubclass(
            declaration_type,
            RuntimeMeasurementIndexedDescriptorDeclaration,
        ):
            raise TypeError(
                "Indexed descriptor declaration must inherit "
                "RuntimeMeasurementIndexedDescriptorDeclaration."
            )
        if declaration_type not in cls.__registry__.values():
            raise TypeError(
                f"{declaration_type.__name__} is not registered in {cls.__name__}."
            )
        return declaration_type

    @classmethod
    def matching_declarations(
        cls,
        feature_name: str,
    ) -> tuple[tuple[type["RuntimeMeasurementIndexedDescriptorDeclaration"], object], ...]:
        """Return registered descriptor declarations that parse ``feature_name``."""
        return tuple(
            (declaration_type, descriptor)
            for declaration_type in cls.__registry__.values()
            for descriptor in (declaration_type.from_feature_name(feature_name),)
            if descriptor is not None
        )

    @classmethod
    def indexed_suffix_token_width_for(
        cls,
        feature_tokens: tuple[str, ...],
    ) -> int | None:
        """Return the unique registered descriptor suffix width for feature tokens."""
        suffix_widths = frozenset(
            suffix_width
            for declaration_type in cls.__registry__.values()
            for suffix_width in (
                declaration_type.indexed_suffix_token_width(feature_tokens),
            )
            if suffix_width is not None
        )
        if not suffix_widths:
            return None
        if len(suffix_widths) != 1:
            raise ValueError(
                "Indexed descriptor declarations disagree on suffix width for "
                f"{feature_tokens!r}: {tuple(sorted(suffix_widths))!r}."
            )
        return next(iter(suffix_widths))

    @classmethod
    @abstractmethod
    def from_feature_name(
        cls,
        feature_name: str,
    ) -> object | None:
        """Parse ``feature_name`` into this declaration's descriptor identity."""

    @classmethod
    @abstractmethod
    def feature_name(
        cls,
        descriptor: object,
    ) -> str:
        """Render one descriptor identity."""

    @classmethod
    @abstractmethod
    def indexed_suffix_token_width(
        cls,
        feature_tokens: tuple[str, ...],
    ) -> int | None:
        """Return trailing token width owned by the descriptor index, if any."""


class RuntimeMeasurementFeature(str, Enum):
    """Base for generated runtime measurement feature enums."""

    def __new__(
        cls,
        value: str,
        relations: Iterable[RuntimeMeasurementFeatureRelation] = (),
        semantic_markers: Iterable[type[RuntimeMeasurementFeatureSemanticMarker]] = (),
        indexed_descriptor_declarations: Iterable[
            type[RuntimeMeasurementIndexedDescriptorDeclaration]
        ] = (),
    ):
        descriptor_declarations = tuple(
            RuntimeMeasurementIndexedDescriptorDeclaration.require_registered(
                declaration_type
            )
            for declaration_type in indexed_descriptor_declarations
        )
        member = str_enum_member_with_payload(
            cls, value, payload_attribute="_relations", payload=tuple(relations)
        )
        member.__dict__["_semantic_markers"] = tuple(semantic_markers)
        member.__dict__["_indexed_descriptor_declarations"] = descriptor_declarations
        return member

    feature_name = AliasProperty[str]("value")
    relations = AliasProperty[tuple[RuntimeMeasurementFeatureRelation, ...]](
        "_relations"
    )
    semantic_markers = AliasProperty[
        tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]
    ]("_semantic_markers")
    _indexed_descriptor_declaration_types = AliasProperty[
        tuple[type[RuntimeMeasurementIndexedDescriptorDeclaration], ...]
    ]("_indexed_descriptor_declarations")

    def feature_family(self) -> str:
        """Return this feature's normalized runtime family."""
        return normalize_runtime_identifier(self.value)

    def relation_declarations(
        self,
    ) -> tuple[RuntimeMeasurementFeatureRelationDeclaration, ...]:
        """Return relation declarations owned by this feature member."""
        return tuple(
            (
                RuntimeMeasurementFeatureRelationDeclaration(self, relation)
                for relation in self.relations
            )
        )

    def indexed_descriptor_declarations(
        self,
    ) -> tuple[type[RuntimeMeasurementIndexedDescriptorDeclaration], ...]:
        """Return parser/render declarations owned by this feature member."""
        return self._indexed_descriptor_declaration_types


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureRelationDeclaration:
    """One producer-owned relation declared between measurement feature families."""

    source_feature: RuntimeMeasurementFeature
    relation: RuntimeMeasurementFeatureRelation

    def source_family_names(
        self, relation_type: type[RuntimeMeasurementFeatureRelation]
    ) -> tuple[str, ...]:
        """Return source families when this declaration belongs to ``relation_type``."""
        if not isinstance(self.relation, relation_type):
            return ()
        return self.relation.source_family_names(self.source_feature)

    def target_family_name(
        self,
        relation_type: type[RuntimeMeasurementFeatureRelation],
        source_family_name: str,
    ) -> str | None:
        """Return the target family for one source family and relation type."""
        if not isinstance(self.relation, relation_type):
            return None
        return self.relation.target_family_name(
            self.source_feature, source_family_name, type(self.source_feature)
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureRelationDeclarationCollection:
    """Blind query surface over producer-declared feature relations."""

    declarations: tuple[RuntimeMeasurementFeatureRelationDeclaration, ...]

    def __init__(
        self, declarations: Iterable[RuntimeMeasurementFeatureRelationDeclaration]
    ) -> None:
        normalized = tuple(declarations)
        for declaration in normalized:
            if not isinstance(
                declaration, RuntimeMeasurementFeatureRelationDeclaration
            ):
                raise TypeError(
                    "RuntimeMeasurementFeatureRelationDeclarationCollection requires RuntimeMeasurementFeatureRelationDeclaration values."
                )
        object.__setattr__(self, "declarations", normalized)

    def source_family_names(
        self, relation_type: type[RuntimeMeasurementFeatureRelation]
    ) -> tuple[str, ...]:
        """Return all declared source families for one relation type."""
        return tuple(
            (
                family_name
                for declaration in self.declarations
                for family_name in declaration.source_family_names(relation_type)
            )
        )

    def target_family_name(
        self,
        relation_type: type[RuntimeMeasurementFeatureRelation],
        source_family_name: str,
    ) -> str | None:
        """Return the declared target family for one source family."""
        for declaration in self.declarations:
            target_family = declaration.target_family_name(
                relation_type, source_family_name
            )
            if target_family is not None:
                return target_family
        return None


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureFamilyRelation(RuntimeMeasurementFeatureRelation):
    """Relation from one feature family to another family in the same enum."""

    target_member_name: str

    def target_feature(
        self, feature_type: type[RuntimeMeasurementFeature]
    ) -> RuntimeMeasurementFeature:
        """Return the target feature member named by this relation."""
        try:
            return feature_type.__members__[self.target_member_name]
        except KeyError as exc:
            raise ValueError(
                f"{feature_type.__name__} relation targets unknown member {self.target_member_name!r}."
            ) from exc

    def source_family_names(
        self, source_feature: RuntimeMeasurementFeature
    ) -> tuple[str, ...]:
        """Return feature-family names that select the relation source."""
        return (source_feature.feature_family(),)

    def target_family_name(
        self,
        source_feature: RuntimeMeasurementFeature,
        source_family_name: str,
        feature_type: type[RuntimeMeasurementFeature],
    ) -> str | None:
        """Return the target family when ``source_family_name`` selects source."""
        if normalize_runtime_identifier(source_family_name) not in (
            normalize_runtime_identifier(family)
            for family in self.source_family_names(source_feature)
        ):
            return None
        return self.target_feature(feature_type).feature_family()


class MeasurementStatistic(str, Enum):
    """Canonical runtime measurement statistic labels."""

    VALUE = "value"
    COUNT = "count"
    MEAN = "mean"


class ObjectCoreMeasurementFeature(RuntimeMeasurementFeature):
    """Core object-measurement feature families."""

    OBJECT_COUNT = "object_count"
    OBJECT_NUMBER = "object_number"
    CENTER_X = "center_x"
    CENTER_Y = "center_y"
    CENTER_Z = "center_z"


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
    __enum_member_attr__ = "axis_feature"
    axis_feature: ClassVar[ObjectCoreMeasurementFeature]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def coordinate_values(
        self, axis_centers: Sequence[Any], counts: Any
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
    absent_axis_missing_for_unlabeled_objects: ClassVar[bool] = True

    def coordinate_values(
        self, axis_centers: Sequence[Any], counts: Any
    ) -> ObjectLocationCoordinateValues:
        import numpy as np

        if len(axis_centers) >= type(self).required_ndim:
            return ObjectLocationCoordinateValues(
                axis_centers[type(self).axis_offset], include_missing=False
            )
        values = np.zeros(len(counts))
        if type(self).absent_axis_missing_for_unlabeled_objects:
            values = self.missing_for_absent_labels(values, counts)
        return ObjectLocationCoordinateValues(values, include_missing=False)


for _coordinate_projection_spec in (
    GeneratedLeafClassSpec(
        class_name="CenterXObjectLocationCoordinateProjectionStrategy",
        base_type=AxisBackedObjectLocationCoordinateProjectionStrategy,
        attributes={
            "axis_feature": ObjectCoreMeasurementFeature.CENTER_X,
            "required_ndim": 1,
            "axis_offset": -1,
        },
    ),
    GeneratedLeafClassSpec(
        class_name="CenterYObjectLocationCoordinateProjectionStrategy",
        base_type=AxisBackedObjectLocationCoordinateProjectionStrategy,
        attributes={
            "axis_feature": ObjectCoreMeasurementFeature.CENTER_Y,
            "required_ndim": 2,
            "axis_offset": -2,
        },
    ),
    GeneratedLeafClassSpec(
        class_name="CenterZObjectLocationCoordinateProjectionStrategy",
        base_type=AxisBackedObjectLocationCoordinateProjectionStrategy,
        attributes={
            "axis_feature": ObjectCoreMeasurementFeature.CENTER_Z,
            "required_ndim": 3,
            "axis_offset": -3,
            "absent_axis_missing_for_unlabeled_objects": False,
        },
    ),
):
    _coordinate_projection_spec.declare_in(globals())


def object_location_coordinate_arrays(
    axis_centers: Sequence[Any], counts: Any
) -> tuple[tuple[str, ObjectLocationCoordinateValues], ...]:
    """Return nominal object-location coordinate arrays in core feature order."""
    return tuple(
        (
            strategy_type.axis_feature.value,
            strategy_type().coordinate_values(axis_centers, counts),
        )
        for strategy_type in (
            ObjectLocationCoordinateProjectionStrategy.registered_strategy_types()
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


@dataclass(frozen=True, slots=True)
class ObjectMeasurementValueRow:
    """Nominal long-form object measurement row."""

    object_label: int
    feature_name: str
    result_value: float


@dataclass(frozen=True, slots=True)
class ObjectMeasurementSliceValueRow(ObjectMeasurementValueRow):
    """Long-form object measurement row scoped to a runtime slice."""

    slice_index: int


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
    OBJECT_ROW_IDENTITY = "openhcs_object_row_identity"
    SOURCE_IMAGE_NAME = "source_image_name"
    BIN_INDEX = "bin_index"
    BIN_COUNT = "bin_count"
    SCALE = "scale"
    DIRECTION = "direction"
    GRAY_LEVELS = "gray_levels"
    ZERNIKE_N = "n"
    ZERNIKE_M = "m"

    @classmethod
    def field_names(cls) -> frozenset[str]:
        """Return every canonical row-axis field name."""
        return frozenset((field.value for field in cls))

    @classmethod
    def object_id_fields(cls) -> tuple["MeasurementRowAxisField", ...]:
        """Return axis fields that can identify an object row."""
        return (
            cls.OBJECT_LABEL,
            cls.OBJECT_NUMBER,
            cls.OBJECT_ID,
            cls.LABEL,
        )

    @classmethod
    def object_id_field_names(cls) -> tuple[str, ...]:
        """Return canonical object-row identity field names."""
        return tuple(field.value for field in cls.object_id_fields())

    @classmethod
    def normalized_object_id_field_names(cls) -> frozenset[str]:
        """Return normalized object-row identity field names."""
        return frozenset(
            normalize_runtime_identifier(field_name)
            for field_name in (
                *cls.object_id_field_names(),
                cls.OBJECT_ROW_IDENTITY.value,
            )
        )

    @classmethod
    def object_ownership_fields(cls) -> tuple["MeasurementRowAxisField", ...]:
        """Return row-axis fields that declare row ownership."""
        return (cls.OBJECT_NAME, cls.SOURCE_IMAGE_NAME)

    @classmethod
    def object_ownership_field_names(cls) -> tuple[str, ...]:
        """Return canonical row ownership field names."""
        return tuple(field.value for field in cls.object_ownership_fields())

    @classmethod
    def feature_name_fields(cls) -> tuple["MeasurementRowAxisField", ...]:
        """Return axis fields that name long-form measurement features."""
        return (cls.FEATURE_NAME, cls.MEASUREMENT_NAME, cls.OUTPUT_NAME)

    @classmethod
    def feature_name_field_names_ordered(cls) -> tuple[str, ...]:
        """Return long-form feature-name fields in semantic priority order."""
        return tuple(field.value for field in cls.feature_name_fields())

    @classmethod
    def feature_name_field_names(cls) -> frozenset[str]:
        """Return canonical long-form feature-name field names."""
        return frozenset(cls.feature_name_field_names_ordered())

    @classmethod
    def normalized_feature_name_field_names(cls) -> frozenset[str]:
        """Return normalized long-form feature-name field names."""
        return frozenset(
            normalize_runtime_identifier(field_name)
            for field_name in cls.feature_name_field_names()
        )


class MeasurementRowValueField(str, Enum):
    """Canonical scalar value fields for long/tall measurement rows."""

    RESULT_VALUE = "result_value"
    MEASUREMENT_VALUE = "measurement_value"
    VALUE = "value"
    MEAN_VALUE = "mean_value"

    @classmethod
    def fields(cls) -> tuple["MeasurementRowValueField", ...]:
        """Return scalar measurement value fields in semantic priority order."""
        return (cls.RESULT_VALUE, cls.MEASUREMENT_VALUE, cls.VALUE, cls.MEAN_VALUE)

    @classmethod
    def field_names_ordered(cls) -> tuple[str, ...]:
        """Return scalar measurement value field names in semantic priority order."""
        return tuple(field.value for field in cls.fields())

    @classmethod
    def field_names(cls) -> frozenset[str]:
        """Return every canonical scalar measurement value field name."""
        return frozenset(cls.field_names_ordered())

    @classmethod
    def normalized_field_names(cls) -> frozenset[str]:
        """Return normalized scalar measurement value field names."""
        return frozenset(
            normalize_runtime_identifier(field_name) for field_name in cls.field_names()
        )


@dataclass(frozen=True, slots=True)
class MeasurementScalarLiteral:
    """Scalar classification shared by measurement row and setting policies."""

    raw_value: object
    _NUMERIC_LITERAL_RE: ClassVar[re.Pattern[str]] = re.compile(
        "^[+-]?(?:(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?|nan|inf|infinity)$",
        re.IGNORECASE,
    )

    @property
    def token(self) -> str | None:
        if self.raw_value in (None, ""):
            return None
        if isinstance(self.raw_value, bool):
            return str(int(self.raw_value))
        if isinstance(self.raw_value, (int, float, np.integer, np.floating)):
            return str(self.raw_value)
        if isinstance(self.raw_value, str):
            stripped = self.raw_value.strip()
            return stripped or None
        return None

    @property
    def is_absent(self) -> bool:
        return self.token is None

    @property
    def is_numeric(self) -> bool:
        token = self.token
        return token is not None and self._NUMERIC_LITERAL_RE.match(token) is not None

    @property
    def numeric_value(self) -> float | None:
        token = self.token
        if token is None or self._NUMERIC_LITERAL_RE.match(token) is None:
            return None
        return float(token)

    @property
    def is_finite_numeric(self) -> bool:
        value = self.numeric_value
        return value is not None and math.isfinite(value)

    @property
    def is_nonfinite_numeric(self) -> bool:
        value = self.numeric_value
        return value is not None and (not math.isfinite(value))

    @property
    def finite_numeric_value(self) -> float | None:
        value = self.numeric_value
        return value if value is not None and math.isfinite(value) else None

    @property
    def integer_value(self) -> int | None:
        value = self.finite_numeric_value
        if value is None:
            return None
        integer = int(value)
        return integer if float(integer) == value else None

    @property
    def is_present_axis_value(self) -> bool:
        if self.is_absent:
            return False
        return self.is_finite_numeric if self.is_numeric else True

    @property
    def is_present_measurement_value(self) -> bool:
        if self.is_absent:
            return False
        value = self.numeric_value
        if value is None:
            return True
        return not math.isnan(value)

    @property
    def is_padding_measurement_value(self) -> bool:
        return not self.is_present_measurement_value


def measurement_axis_integer_value(
    value: object, axis: MeasurementRowAxisField
) -> int | None:
    """Return one present integer axis value, or ``None`` for absent values."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        literal = MeasurementScalarLiteral(stripped)
        if not literal.is_present_axis_value:
            return None
        integer_value = literal.integer_value
        if integer_value is None:
            raise ValueError(
                f"Measurement axis field {axis.value!r} requires integer-compatible values, got {value!r}."
            )
        return integer_value
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        integer = int(value)
        if float(integer) == float(value):
            return integer
        raise ValueError(
            f"Measurement axis field {axis.value!r} requires integer-compatible values, got {value!r}."
        )
    literal = MeasurementScalarLiteral(value)
    if not literal.is_present_axis_value:
        return None
    integer_value = literal.integer_value
    if integer_value is None:
        raise ValueError(
            f"Measurement axis field {axis.value!r} requires integer-compatible values, got {value!r}."
        )
    return integer_value


def measurement_axis_integer_domain(
    values: Sequence[object], axis: MeasurementRowAxisField
) -> tuple[int, ...]:
    """Return the present integer domain for one row-axis value vector."""
    if isinstance(values, np.ndarray) and values.size == 0:
        return ()
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.bool_):
        return tuple((int(value) for value in np.unique(values)))
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.integer):
        return tuple((int(value) for value in np.unique(values)))
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.floating):
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return ()
        integer_values = finite_values.astype(np.int64)
        if not bool(np.all(finite_values == integer_values)):
            invalid_value = finite_values[finite_values != integer_values][0]
            raise ValueError(
                f"Measurement axis field {axis.value!r} requires integer-compatible values, got {invalid_value!r}."
            )
        return tuple((int(value) for value in np.unique(integer_values)))
    return tuple(
        dict.fromkeys(
            (
                integer_value
                for value in values
                for integer_value in (measurement_axis_integer_value(value, axis),)
                if integer_value is not None
            )
        )
    )


class MeasurementRowAxisState(str, Enum):
    """Whether measurement rows are runtime-axis keyed or image-number keyed."""

    RUNTIME_AXES = "runtime_axes"
    IMAGE_NUMBER = "image_number"

    @classmethod
    def for_field_names(cls, field_names: Iterable[str]) -> "MeasurementRowAxisState":
        """Return row-axis state from declared measurement field names."""
        if MeasurementRowAxisField.IMAGE_NUMBER.value in frozenset(field_names):
            return cls.IMAGE_NUMBER
        return cls.RUNTIME_AXES

    @classmethod
    def for_image_number_presence(
        cls, *, has_image_number: bool
    ) -> "MeasurementRowAxisState":
        """Return row-axis state from precomputed ImageNumber presence."""
        return cls.IMAGE_NUMBER if has_image_number else cls.RUNTIME_AXES


class ObjectFeatureArrayDomain(str, Enum):
    """How a feature array indexes values for an object-feature table."""

    MEASURED_OBJECT_ID = "measured_object_id"
    LABEL_ID = "label_id"
    ROW_ORDINAL = "row_ordinal"


class ObjectFeatureMissingValue(str, Enum):
    """How an object-feature table represents unmeasured feature values."""

    NAN = "nan"
    ZERO = "zero"


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
    EnumKeyedStrategyMixin[ObjectFeatureArrayDomain], ABC, metaclass=AutoRegisterMeta
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
        self, context: ObjectFeatureArrayDomainContext
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


class OrdinalObjectFeatureArrayDomainStrategy(ObjectFeatureArrayDomainStrategy):
    """Feature arrays indexed by an ordered object-ID axis."""

    @abstractmethod
    def ordinal_axis(self, context: ObjectFeatureArrayDomainContext) -> tuple[int, ...]:
        """Return the object-ID axis that defines feature-array order."""

    def value_index(self, context: ObjectFeatureArrayDomainContext) -> int | None:
        axis = self.ordinal_axis(context)
        if context.object_id not in axis:
            return None
        value_index = axis.index(context.object_id)
        return value_index if value_index < context.value_count else None

    def value_indexes(
        self, context: ObjectFeatureArrayDomainContext
    ) -> Mapping[int, int]:
        return {
            object_id: index
            for index, object_id in enumerate(self.ordinal_axis(context))
            if index < context.value_count
        }


class MeasuredObjectFeatureArrayDomainStrategy(OrdinalObjectFeatureArrayDomainStrategy):
    """Feature arrays indexed by compact measured-object IDs."""

    domain = ObjectFeatureArrayDomain.MEASURED_OBJECT_ID

    def ordinal_axis(self, context: ObjectFeatureArrayDomainContext) -> tuple[int, ...]:
        return context.measured_object_ids

    def accepts(self, context: ObjectFeatureArrayDomainContext) -> bool:
        return context.value_count == context.measured_object_count


class LabelIdFeatureArrayDomainStrategy(ObjectFeatureArrayDomainStrategy):
    """Feature arrays indexed by dense label ID minus one."""

    domain = ObjectFeatureArrayDomain.LABEL_ID

    def value_index(self, context: ObjectFeatureArrayDomainContext) -> int | None:
        value_index = context.object_id - 1
        return value_index if 0 <= value_index < context.value_count else None

    def accepts(self, context: ObjectFeatureArrayDomainContext) -> bool:
        return context.value_count >= context.measured_object_max

    def value_indexes(
        self, context: ObjectFeatureArrayDomainContext
    ) -> Mapping[int, int]:
        return {
            object_id: object_id - 1
            for object_id in context.object_domain
            if 0 <= object_id - 1 < context.value_count
        }


class RowOrdinalFeatureArrayDomainStrategy(OrdinalObjectFeatureArrayDomainStrategy):
    """Feature arrays indexed by the emitted row ordinal."""

    domain = ObjectFeatureArrayDomain.ROW_ORDINAL

    def ordinal_axis(self, context: ObjectFeatureArrayDomainContext) -> tuple[int, ...]:
        return context.object_domain

    def accepts(self, context: ObjectFeatureArrayDomainContext) -> bool:
        return context.value_count <= len(context.object_domain)


class ObjectFeatureMissingValueStrategy(
    EnumKeyedStrategyMixin[ObjectFeatureMissingValue], ABC, metaclass=AutoRegisterMeta
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
    NominalTypeKeyedStrategyMixin, ABC, metaclass=AutoRegisterMeta
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
                self.measured_object_ids, "ObjectFeatureValueTable.measured_object_ids"
            ),
        )
        object.__setattr__(
            self,
            "object_domain",
            ObjectLabelDomain._normalize_ids(
                self.object_domain, "ObjectFeatureValueTable.object_domain"
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
            measured_object_ids=tuple(
                (int(object_id) for object_id in measured_object_ids)
            ),
            object_domain=tuple((int(object_id) for object_id in object_domain)),
            **kwargs,
        )

    def rows(self) -> list[dict[str, float | int]]:
        """Return wide rows ordered by the declared object domain."""
        feature_items = tuple(
            (
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
        self, feature_name: str, values: np.ndarray
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
        self, feature_name: str, object_id: int, *, values: np.ndarray
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
            feature_name, ObjectFeatureArrayDomain.MEASURED_OBJECT_ID
        )

    def validate_feature_value_domain(
        self, feature_name: str, values: np.ndarray
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
            f"{type(self).__name__} feature {feature_name!r} has {context.value_count} values for {context.measured_object_count} measured objects. Feature arrays must align to measured_object_ids unless the table declares another feature-array domain."
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
            feature_name, ObjectFeatureMissingValue.NAN
        )


class GenericObjectFeatureValueTable(ObjectFeatureValueTable):
    """Generic object feature table with NaN missing values."""

    table_label = "generic"


class MeasurementTableRowLayout(str, Enum):
    """Nominal row layout for measurement tables."""

    EMPTY = "empty"
    LONG = "long"
    WIDE = "wide"


class MeasurementRowLayoutProjectionStrategy(
    EnumKeyedStrategyMixin, ABC, metaclass=AutoRegisterMeta
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
    return MeasurementRowAxisField.feature_name_field_names()


def measurement_row_feature_field_names_ordered() -> tuple[str, ...]:
    """Return long-form feature-name fields in semantic priority order."""
    return MeasurementRowAxisField.feature_name_field_names_ordered()


def measurement_row_value_field_names() -> frozenset[str]:
    """Return row fields that carry long-form measurement values."""
    return MeasurementRowValueField.field_names()


def measurement_row_value_field_names_ordered() -> tuple[str, ...]:
    """Return long-form value fields in semantic priority order."""
    return MeasurementRowValueField.field_names_ordered()


def measurement_row_semantic_field_names() -> frozenset[str]:
    """Return fields that identify a payload as a measurement row."""
    return measurement_row_axis_field_names() | measurement_row_value_field_names()


def carries_measurement_row_semantics(row: object) -> bool:
    """Return whether a row-like object declares measurement-row fields."""
    semantic_fields = measurement_row_semantic_field_names()
    if isinstance(row, Mapping):
        field_names = frozenset((str(field_name) for field_name in row.keys()))
    elif is_dataclass(row):
        field_names = frozenset((field.name for field in dataclass_fields(row)))
    elif type(row).__dictoffset__ != 0:
        field_names = frozenset((str(field_name) for field_name in vars(row).keys()))
    else:
        return False
    return bool(field_names & semantic_fields)


def supports_measurement_row_mapping(row: object) -> bool:
    """Return whether ``measurement_row_mapping`` can project this row."""
    return (
        isinstance(row, Mapping) or is_dataclass(row) or type(row).__dictoffset__ != 0
    )


def measurement_row_mapping(row: object) -> Mapping[str, object]:
    """Return a mapping view for a supported measurement row payload."""
    if isinstance(row, Mapping):
        return row
    if is_dataclass(row):
        return MeasurementRowMappingCache.process_cache().mapping(row)
    if type(row).__dictoffset__ != 0:
        return vars(row)
    raise TypeError(f"Unsupported measurement row type {type(row).__name__}.")


@dataclass(slots=True)
class MeasurementRowMappingCache(
    RegisteredProcessLocalBoundedCache[int, tuple[object, Mapping[str, object]]]
):
    """Bounded process-local cache for immutable dataclass measurement rows."""

    max_entries: int = 262144

    def mapping(self, row: object) -> Mapping[str, object]:
        row_id = id(row)
        cached = self.cached_value(row_id)
        if cached is not None:
            cached_row, row_mapping = cached
            if cached_row is row:
                return row_mapping
            del self.entries[row_id]
        row_mapping = asdict(row)
        row_mapping.update(MeasurementRowDescriptorFields.mapping(row, row_mapping))
        self.store_value(row_id, (row, row_mapping))
        return row_mapping


class MeasurementRowDescriptorFields:
    """Project class-declared descriptor columns for dataclass measurement rows."""

    @classmethod
    def mapping(
        cls, row: object, dataclass_mapping: Mapping[str, object]
    ) -> Mapping[str, object]:
        field_names = frozenset(dataclass_mapping)
        descriptor_values: dict[str, object] = {}
        for owner in reversed(type(row).__mro__):
            annotations = vars(owner).get("__annotations__", {})
            for field_name in annotations:
                if field_name in field_names or field_name in descriptor_values:
                    continue
                descriptor = vars(owner).get(field_name)
                if not cls.is_descriptor_column(descriptor):
                    continue
                descriptor_values[field_name] = cls.value(row, descriptor, field_name)
        return descriptor_values

    @staticmethod
    def is_descriptor_column(descriptor: object) -> bool:
        return descriptor is not None and callable(
            vars(type(descriptor)).get("__get__")
        )

    @staticmethod
    def value(row: object, descriptor: object, field_name: str) -> object:
        try:
            return descriptor.__get__(row, type(row))
        except Exception as exc:
            raise ValueError(
                f"Measurement row descriptor field {type(row).__name__}.{field_name} could not be projected."
            ) from exc


def measurement_table_row_layout(rows: object) -> MeasurementTableRowLayout:
    """Return the declared layout implied by a table row payload."""
    observed_layouts = measurement_table_row_layouts(rows)
    if not observed_layouts:
        return MeasurementTableRowLayout.EMPTY
    if len(observed_layouts) != 1:
        raise ValueError(
            f"MeasurementTable rows must not mix long-form and wide-form layouts; got {sorted((layout.value for layout in observed_layouts))!r}."
        )
    return next(iter(observed_layouts))


def measurement_table_row_layout_from_fields(
    fields: Iterable[FieldSpec],
) -> MeasurementTableRowLayout | None:
    """Return row layout declared by table fields when fields are authoritative."""
    return _measurement_table_row_layout_from_field_names(
        tuple((field.name for field in fields))
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
    if has_feature_field and (not has_value_field):
        raise ValueError(
            f"Long-form measurement table fields must declare both a feature field and a value field, got fields {sorted(field_names)!r}."
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
        (isinstance(row, ObjectMeasurementValueRow) for row in row_sequence)
    ):
        return frozenset((MeasurementTableRowLayout.LONG,))
    return frozenset(
        (MeasurementRowLayoutAuthority(row).layout() for row in row_sequence)
    )


def normalize_measurement_table_rows(
    rows: object, *, fields: Iterable[FieldSpec] = ()
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
    rows: object, layout: MeasurementTableRowLayout
) -> object:
    """Project measurement rows into a declared table layout."""
    if layout is not MeasurementTableRowLayout.LONG:
        raise ValueError(
            f"Unsupported measurement row layout projection: {layout.value}."
        )
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
            (str(field_name) for field_name in measurement_row_mapping(self.row))
        )
        has_feature_field = bool(field_names & measurement_row_feature_field_names())
        has_value_field = bool(field_names & measurement_row_value_field_names())
        if has_feature_field and (not has_value_field):
            raise ValueError(
                f"Long-form measurement rows must declare both a feature field and a value field, got fields {sorted(field_names)!r}."
            )
        return (
            MeasurementTableRowLayout.LONG
            if has_feature_field
            else MeasurementTableRowLayout.WIDE
        )


class MeasurementObjectRowIdentity(str, Enum):
    """How object-scoped measurement rows identify their measured object."""

    LABEL_ID = "label_id"
    ROW_ORDINAL = "row_ordinal"
    ROW_SEQUENCE = "row_sequence"


def measurement_row_axis_field_names() -> frozenset[str]:
    """Return fields that identify a measurement row axis, not a result value."""
    return MeasurementRowAxisField.field_names()


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
    kind: ArtifactType = ObjectLabelsArtifactType

    def __post_init__(self) -> None:
        _require_name(self.name, "RelationshipEndpoint.name")
        _require_name(self.role, "RelationshipEndpoint.role")
        _require_name(self.id_field, "RelationshipEndpoint.id_field")
        object.__setattr__(self, "kind", ArtifactType.coerce(self.kind))


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
    artifact_name: str, *, parent_candidates: tuple[str, ...]
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
        child_name = body[len(prefix) :]
        if child_name:
            return (parent_name, child_name)
    return None


@dataclass(frozen=True, slots=True)
class ParentChildRelationshipPayload(
    RuntimeSliceProjectableValue, RuntimeSliceIdentityProjectableValue
):
    """Generic parent-child id pairs emitted before endpoint names are bound."""

    parent_ids: tuple[int, ...]
    child_ids: tuple[int, ...]
    slice_indices: tuple[int, ...] = ()
    slice_count: int | None = None

    def __post_init__(self) -> None:
        parent_ids = tuple((int(parent_id) for parent_id in self.parent_ids))
        child_ids = tuple((int(child_id) for child_id in self.child_ids))
        if len(parent_ids) != len(child_ids):
            raise ValueError(
                f"ParentChildRelationshipPayload parent_ids and child_ids must have equal length, got {len(parent_ids)} and {len(child_ids)}."
            )
        slice_indices = tuple((int(slice_index) for slice_index in self.slice_indices))
        if slice_indices and len(slice_indices) != len(parent_ids):
            raise ValueError(
                f"ParentChildRelationshipPayload slice_indices must be empty or match parent_ids/child_ids length, got {len(slice_indices)} for {len(parent_ids)} relationships."
            )
        if any((slice_index < 0 for slice_index in slice_indices)):
            raise ValueError(
                "ParentChildRelationshipPayload slice_indices cannot be negative."
            )
        slice_count = None if self.slice_count is None else int(self.slice_count)
        if slice_count is not None and slice_count < 0:
            raise ValueError(
                "ParentChildRelationshipPayload slice_count cannot be negative."
            )
        if (
            slice_count is not None
            and slice_indices
            and (max(slice_indices) >= slice_count)
        ):
            raise ValueError(
                f"ParentChildRelationshipPayload slice_indices must be smaller than slice_count {slice_count}."
            )
        object.__setattr__(self, "parent_ids", parent_ids)
        object.__setattr__(self, "child_ids", child_ids)
        object.__setattr__(self, "slice_indices", slice_indices)
        object.__setattr__(self, "slice_count", slice_count)

    def with_runtime_slice_identity(
        self, *, slice_index: int, slice_count: int
    ) -> "ParentChildRelationshipPayload":
        """Return relationships stamped with the execution runtime-slice identity."""
        return ParentChildRelationshipPayload(
            parent_ids=self.parent_ids,
            child_ids=self.child_ids,
            slice_indices=tuple((int(slice_index) for _child_id in self.child_ids)),
            slice_count=int(slice_count),
        )

    def explicit_slice_indices(self) -> tuple[int, ...]:
        """Return per-relationship slice ids for persisted relationship records."""
        if self.slice_indices:
            return self.slice_indices
        return tuple((0 for _child_id in self.child_ids))

    def project_runtime_slice(
        self, slice_index: int
    ) -> "ParentChildRelationshipPayload":
        """Return only relationship rows belonging to one runtime slice."""
        if not self.slice_indices:
            if (
                self.slice_count is not None
                and self.slice_count > 1
                and self.parent_ids
            ):
                raise ValueError(
                    "Cannot slice multi-plane ParentChildRelationshipPayload without slice_indices."
                )
            return self
        parent_ids: list[int] = []
        child_ids: list[int] = []
        for parent_id, child_id, relationship_slice_index in zip(
            self.parent_ids, self.child_ids, self.slice_indices, strict=True
        ):
            if relationship_slice_index != int(slice_index):
                continue
            parent_ids.append(parent_id)
            child_ids.append(child_id)
        return ParentChildRelationshipPayload(
            parent_ids=tuple(parent_ids), child_ids=tuple(child_ids), slice_count=1
        )


class ObjectRelationshipPayloadKernel(ABC):
    """Kernel contract used by semantic relationship payload policies."""

    def dominant_parent_ids_by_child(
        self, parent_array: Any, child_array: Any, context_array: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return child ids with their dominant parent ids by positive overlap."""
        children = np.asarray(child_array, dtype=np.int64)
        parents = np.asarray(parent_array, dtype=np.int64)
        context = np.asarray(context_array, dtype=np.int64)
        child_ids = np.unique(children[children > 0])
        if child_ids.size == 0:
            empty = np.zeros(0, dtype=np.int64)
            return (empty, empty)
        max_parent = int(np.max(parents)) if parents.size else 0
        parent_ids = np.zeros(child_ids.size, dtype=np.int64)
        valid = (context > 0) & (parents > 0)
        if not np.any(valid) or max_parent <= 0:
            return (child_ids, parent_ids)
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
        return (child_ids, parent_ids)

    @abstractmethod
    def relate_children_to_parents(
        self, parent_labels: np.ndarray, child_labels: np.ndarray, child_count: int
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
        self, parent_labels: np.ndarray, child_labels: np.ndarray, child_count: int
    ) -> np.ndarray:
        del child_count
        child_ids, parent_ids = self.dominant_parent_ids_by_child(
            parent_labels, child_labels, child_labels
        )
        parents_of = np.zeros(
            int(child_labels.max()) if child_labels.size else 0, dtype=np.int32
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
            votes[child_id, parent_id] = votes.get((child_id, parent_id), 0) + 1
        for child_id in range(1, child_count + 1):
            child_votes = tuple(
                (
                    (parent_id, count)
                    for (candidate_child, parent_id), count in votes.items()
                    if candidate_child == child_id
                )
            )
            if child_votes:
                parents_of[child_id - 1] = min(
                    child_votes, key=lambda item: (-item[1], item[0])
                )[0]
        return parents_of


DEFAULT_OBJECT_RELATIONSHIP_PAYLOAD_KERNEL = DefaultObjectRelationshipPayloadKernel()


@dataclass(frozen=True, slots=True)
class ObjectRelationshipPayloadRequest:
    """Semantic request for deriving parent-child payloads from label values."""

    parent_labels: ObjectLabelMeasurementSource
    child_labels: ObjectLabelMeasurementSource
    kernel: ObjectRelationshipPayloadKernel = DEFAULT_OBJECT_RELATIONSHIP_PAYLOAD_KERNEL


class ObjectRelationshipPayloadStrategy(
    MostDerivedContextStrategyMixin[ObjectRelationshipPayloadRequest], ABC
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
        self, context: ObjectRelationshipPayloadRequest
    ) -> ParentChildRelationshipPayload:
        """Return parent-child ids for the strategy's representation contract."""

    @staticmethod
    def related_payload_from_parents_of(
        parents_of: np.ndarray, child_ids: np.ndarray
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
            parent_ids=tuple(parent_ids), child_ids=tuple(related_child_ids)
        )

    @staticmethod
    def related_payload_from_dense_parent_vector(
        parents_of: np.ndarray,
    ) -> ParentChildRelationshipPayload:
        """Return related dense child ids directly from a 1-indexed parent vector."""
        parent_vector = np.asarray(parents_of, dtype=np.int64)
        related_child_indexes = np.flatnonzero(parent_vector > 0)
        return ParentChildRelationshipPayload(
            parent_ids=tuple(
                (int(parent_vector[index]) for index in related_child_indexes)
            ),
            child_ids=tuple((int(index + 1) for index in related_child_indexes)),
        )


class DenseObjectRelationshipPayloadStrategy(ObjectRelationshipPayloadStrategy):
    """Dense label images use maximum positive-pixel overlap."""

    strategy_key = "dense"

    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        del context
        return True

    def payload(
        self, context: ObjectRelationshipPayloadRequest
    ) -> ParentChildRelationshipPayload:
        from openhcs.core.runtime_values import object_label_dense_array

        slice_count = self.relationship_slice_count(context)
        if slice_count is not None:
            aligned_stacks = DenseObjectLabelPairAligner(
                context.parent_labels, context.child_labels
            ).aligned_stacks(slice_count)
            if aligned_stacks is not None:
                return self.stack_payload(context, *aligned_stacks)
        parent_array, child_array = (
            object_label_dense_array(labels, dtype=np.int32)
            for labels in DenseObjectLabelPairAligner(
                context.parent_labels, context.child_labels
            ).aligned()
        )
        child_count = int(child_array.max()) if child_array.size else 0
        if child_count <= 0:
            return ParentChildRelationshipPayload(parent_ids=(), child_ids=())
        parents_of = context.kernel.relate_children_to_parents(
            parent_array, child_array, child_count
        )
        return self.related_payload_from_dense_parent_vector(parents_of)

    def relationship_slice_count(
        self, context: ObjectRelationshipPayloadRequest
    ) -> int | None:
        """Return the plane count for plane-scoped label relationships."""
        domains = tuple(
            (
                ObjectLabelDomainMetadataStrategy.for_value(labels).object_label_domain(
                    labels
                )
                for labels in (context.parent_labels, context.child_labels)
            )
        )
        if (
            ObjectLabelDomainScope.common((domain.scope for domain in domains))
            is not ObjectLabelDomainScope.PLANE
        ):
            return None
        leading_plane_counts = tuple(
            (
                int(label_array.shape[0])
                for label_array in (
                    np.asarray(context.parent_labels),
                    np.asarray(context.child_labels),
                )
                if label_array.ndim == 3
            )
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
                parent_plane, child_plane, child_count
            )
            payload = self.related_payload_from_dense_parent_vector(parents_of)
            parent_ids.extend(payload.parent_ids)
            child_ids.extend(payload.child_ids)
            slice_indices.extend((slice_index for _child_id in payload.child_ids))
        return ParentChildRelationshipPayload(
            parent_ids=tuple(parent_ids),
            child_ids=tuple(child_ids),
            slice_indices=tuple(slice_indices),
            slice_count=int(parent_stack.shape[0]),
        )


class SparseIJVObjectRelationshipPayloadStrategy(
    DenseObjectRelationshipPayloadStrategy
):
    """Sparse IJV labels derive parent-child ids through sparse rows."""

    strategy_key = "sparse_ijv"

    def matches(self, context: ObjectRelationshipPayloadRequest) -> bool:
        from openhcs.core.runtime_values import SparseIJVReplacementLabelsStrategy

        return any(
            (
                SparseIJVReplacementLabelsStrategy.source_is_sparse_ijv(labels)
                for labels in (context.parent_labels, context.child_labels)
            )
        )

    def payload(
        self, context: ObjectRelationshipPayloadRequest
    ) -> ParentChildRelationshipPayload:
        from openhcs.core.runtime_values import SparseIJVReplacementLabelsStrategy

        parent_rows = SparseIJVReplacementLabelsStrategy.replacement_for(
            context.parent_labels
        )
        child_rows = SparseIJVReplacementLabelsStrategy.replacement_for(
            context.child_labels
        )
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
            parents_of, self.present_sparse_child_ids(child_array, child_rows)
        )

    @staticmethod
    def label_count(array: np.ndarray, rows: SparseIJVLabelRows) -> int:
        if array.size == 0:
            return 0
        return int(np.max(array[:, rows.label_column]))

    @staticmethod
    def present_sparse_child_ids(
        child_array: np.ndarray, child_rows: SparseIJVLabelRows
    ) -> np.ndarray:
        if child_array.size == 0:
            return np.empty(0, dtype=np.int32)
        return np.unique(child_array[:, child_rows.label_column]).astype(
            np.int32, copy=False
        )


class ObjectLabelLineageGeometry(str, Enum):
    """Geometry relation used to derive parent-child label lineage."""

    SHARED_GEOMETRY = "shared_geometry"
    IDENTITY_DOMAIN = "identity_domain"


class ObjectLabelLineageStrategy(
    EnumKeyedStrategyMixin[ObjectLabelLineageGeometry], ABC, metaclass=AutoRegisterMeta
):
    """Derive parent-child object lineage from two dense label artifacts."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[ObjectLabelLineageGeometry | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def payload(
        self, parent_labels: Any, child_labels: Any
    ) -> ParentChildRelationshipPayload:
        """Return parent-child ids for the strategy's geometry contract."""


class SharedGeometryObjectLabelLineageStrategy(ObjectLabelLineageStrategy):
    """Use spatial overlap when parent and child labels share a geometry."""

    strategy_key = ObjectLabelLineageGeometry.SHARED_GEOMETRY

    def payload(
        self, parent_labels: Any, child_labels: Any
    ) -> ParentChildRelationshipPayload:
        return object_label_parent_child_payload(parent_labels, child_labels)


class IdentityDomainObjectLabelLineageStrategy(ObjectLabelLineageStrategy):
    """Use preserved label ids when a transform changes label geometry."""

    strategy_key = ObjectLabelLineageGeometry.IDENTITY_DOMAIN

    def payload(
        self, parent_labels: Any, child_labels: Any
    ) -> ParentChildRelationshipPayload:
        parent_ids = set(dense_object_label_id_domain(parent_labels))
        child_ids = tuple(dense_object_label_id_domain(child_labels))
        related_ids = tuple(
            (object_id for object_id in child_ids if object_id in parent_ids)
        )
        return ParentChildRelationshipPayload(
            parent_ids=related_ids, child_ids=related_ids
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
        raw_slice_index = row.get(slice_index_field.value)
        if raw_slice_index is not None and str(raw_slice_index).strip() != "":
            return cls(object_id, slice_index=int(raw_slice_index))
        raw_image_number = row.get(image_number_field.value)
        if raw_image_number is not None and str(raw_image_number).strip() != "":
            slice_index = int(float(raw_image_number) - float(image_number_offset) - 1)
            if slice_index >= 0:
                return cls(object_id, slice_index=slice_index)
        return cls(object_id)

    @classmethod
    def domain(
        cls, object_ids: Iterable[int], *, slice_index: int | None = None
    ) -> tuple["ObjectInstanceKey", ...]:
        """Return typed object identities for one optional measurement plane."""
        return tuple(
            (cls(object_id, slice_index=slice_index) for object_id in object_ids)
        )


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
        source_id_tuple = tuple((int(value) for value in source_ids))
        target_id_tuple = tuple((int(value) for value in target_ids))
        if len(source_id_tuple) != len(target_id_tuple):
            raise ValueError(
                f"ObjectInstanceRelationship source_ids and target_ids must have equal length, got {len(source_id_tuple)} and {len(target_id_tuple)}."
            )
        slice_index_tuple = tuple((int(value) for value in slice_indices))
        if slice_index_tuple and len(slice_index_tuple) != len(source_id_tuple):
            raise ValueError(
                f"ObjectInstanceRelationship slice_indices must be empty or match id columns, got {len(slice_index_tuple)} for {len(source_id_tuple)}."
            )
        resolved_slice_count = None if slice_count is None else int(slice_count)
        if resolved_slice_count is not None and resolved_slice_count < 0:
            raise ValueError(
                "ObjectInstanceRelationship.slice_count cannot be negative."
            )
        source_keys: list[ObjectInstanceKey] = []
        target_keys: list[ObjectInstanceKey] = []
        for index, (source_id, target_id) in enumerate(
            zip(source_id_tuple, target_id_tuple, strict=True)
        ):
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
        self, object_count: int = 0, *, declared_keys: Iterable[ObjectInstanceKey] = ()
    ) -> tuple[ObjectInstanceKey, ...]:
        """Return all source identities represented by this relationship domain."""
        return self._domain(
            self.source_keys,
            object_count=object_count,
            slice_count=self.slice_count,
            declared_keys=declared_keys,
        )

    def target_domain(
        self, object_count: int = 0, *, declared_keys: Iterable[ObjectInstanceKey] = ()
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
                source_object_count, declared_keys=declared_source_keys
            )
        }
        for source_key, target_key in zip(
            self.source_keys, self.target_keys, strict=True
        ):
            children.setdefault(source_key, []).append(target_key)
        return {
            source_key: tuple(
                sorted(
                    child_keys,
                    key=lambda key: (
                        key.slice_index is None,
                        key.slice_index or -1,
                        key.object_id,
                    ),
                )
            )
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
            dict.fromkeys(
                (key.slice_index for key in keys if key.slice_index is not None)
            )
        )
        if not slice_indexes:
            return ObjectInstanceKey.domain(range(1, max_object_id + 1))
        return tuple(
            (
                key
                for slice_index in slice_indexes
                for key in ObjectInstanceKey.domain(
                    range(1, max_object_id + 1), slice_index=slice_index
                )
            )
        )


@dataclass(frozen=True, slots=True)
class ObjectLabelInstanceDomains:
    """Typed object-instance domains keyed by object-label artifact name."""

    domains_by_name: Mapping[str, tuple[ObjectInstanceKey, ...]]

    @classmethod
    def from_named_plane_domains(
        cls, named_plane_domains: Iterable[tuple[str, tuple[tuple[int, ...], ...]]]
    ) -> "ObjectLabelInstanceDomains":
        """Build domains from per-plane object-id domains."""
        domains: dict[str, dict[ObjectInstanceKey, None]] = {}
        for object_name, plane_domains in named_plane_domains:
            slice_indexes = (
                (None,) if len(plane_domains) <= 1 else tuple(range(len(plane_domains)))
            )
            domain = domains.setdefault(str(object_name), {})
            for slice_index, object_ids in zip(
                slice_indexes, plane_domains, strict=True
            ):
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
        cls, adapter: SourceSpatialDomainAdapter
    ) -> "SourceSpatialDomainProjection":
        import numpy as np

        array = np.asarray(adapter.array)
        if array.ndim < 2:
            raise ValueError(
                f"Source-spatial object-label projection requires at least two dimensions, got {array.ndim}."
            )
        return cls(adapter.domain, tuple((int(axis) for axis in array.shape[-2:])))

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
                f"Cannot restore source-domain object labels with shape {array.shape[-2:]} to native shape {self.shape_yx}."
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
        return tuple((int(axis) for axis in array.shape[-2:]))

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
            and (self.native_shape_yx == other.native_shape_yx)
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
        self, domain: SourceSpatialDomain
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
    def from_values(cls, first: Any, second: Any) -> "SourceSpatialAlignmentPair":
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
                (value.materialize_in_domain(source_domain) for value in self.values)
            )
            projections = tuple(
                (value.projection_in_domain(source_domain) for value in self.values)
            )
        else:
            aligned_values = tuple((value.materialize() for value in self.values))
            projections = tuple((value.projection() for value in self.values))
        return DenseObjectLabelPairAlignment(
            first=aligned_values[0],
            second=aligned_values[1],
            first_projection=projections[0],
            second_projection=projections[1],
        )

    def shared_source_domain_for_native_pair(self) -> SourceSpatialDomain | None:
        source_domains = tuple(
            (
                value.source_domain
                for value in self.values
                if value.source_domain is not None
            )
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
        return (alignment.first, alignment.second)

    def alignment(self) -> DenseObjectLabelPairAlignment:
        alignment = self.shared_geometry_alignment()
        if alignment is None:
            first = DenseObjectLabelStack.from_labels(
                self.first_labels
            ).collapse_singleton_plane()
            second = DenseObjectLabelStack.from_labels(
                self.second_labels
            ).collapse_singleton_plane()
            raise ValueError(
                f"Dense object-label payloads must share a common geometry after alignment; got {first.shape} and {second.shape}."
            )
        return alignment

    def shared_geometry_alignment(self) -> DenseObjectLabelPairAlignment | None:
        """Return aligned labels when the pair has a declared shared geometry."""
        alignment = SourceSpatialAlignmentPair.from_values(
            self.first_labels, self.second_labels
        ).aligned()
        first, second = (alignment.first, alignment.second)
        first = DenseObjectLabelStack.from_labels(first).collapse_singleton_plane()
        second = DenseObjectLabelStack.from_labels(second).collapse_singleton_plane()
        if first.shape == second.shape:
            return DenseObjectLabelPairAlignment(
                first=first,
                second=second,
                first_projection=alignment.first_projection,
                second_projection=alignment.second_projection,
            )
        factorized = self._factorized_pair(first, second)
        if factorized is not None:
            first, second = factorized
            return DenseObjectLabelPairAlignment(
                first=first,
                second=second,
                first_projection=alignment.first_projection,
                second_projection=alignment.second_projection,
            )
        if first.ndim == 3 and second.ndim == 2 and (first.shape[1:] == second.shape):
            first = DenseObjectLabelStack.from_labels(
                first
            ).project_xy_plane_without_relabeling()
        if second.ndim == 3 and first.ndim == 2 and (second.shape[1:] == first.shape):
            second = DenseObjectLabelStack.from_labels(
                second
            ).project_xy_plane_without_relabeling()
        if first.shape != second.shape:
            return None
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
        return (alignment.first_stack, alignment.second_stack)

    def aligned_stack_context(
        self, slice_count: int
    ) -> DenseObjectLabelStackAlignment | None:
        alignment = SourceSpatialAlignmentPair.from_values(
            self.first_labels, self.second_labels
        ).aligned()
        first_stack = self._stack_view(alignment.first, slice_count)
        second_stack = self._stack_view(alignment.second, slice_count)
        if first_stack is None or second_stack is None:
            return None
        if first_stack.shape != second_stack.shape:
            raise ValueError(
                f"Dense object-label stacks must share a common geometry after alignment; got {first_stack.shape} and {second_stack.shape}."
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
        if (
            array.ndim == 3
            and array.shape[0] > 0
            and (slice_count % array.shape[0] == 0)
        ):
            indexes = np.arange(slice_count) % array.shape[0]
            return np.ascontiguousarray(array[indexes])
        if array.ndim == 2:
            return np.ascontiguousarray(
                np.broadcast_to(array, (slice_count, *array.shape))
            )
        return None

    @classmethod
    def _factorized_pair(cls, first: Any, second: Any) -> tuple[Any, Any] | None:
        import numpy as np

        first_array = np.asarray(first, dtype=np.int32)
        second_array = np.asarray(second, dtype=np.int32)
        if (
            first_array.ndim != 3
            or second_array.ndim != 3
            or first_array.shape[1:] != second_array.shape[1:]
        ):
            return None
        max_count = max(first_array.shape[0], second_array.shape[0])
        first_stack = cls._stack_view(first_array, max_count)
        second_stack = cls._stack_view(second_array, max_count)
        if first_stack is None or second_stack is None:
            return None
        if first_stack.shape != second_stack.shape:
            return None
        return (first_stack, second_stack)


@dataclass(frozen=True, slots=True)
class DenseObjectLabelMaskAligner:
    """Align dense object labels with an image/binary mask in source geometry."""

    labels: Any
    mask: Any

    def aligned(self) -> tuple[Any, Any]:
        alignment = SourceSpatialAlignmentPair.from_values(
            self.labels, self.mask
        ).aligned()
        labels = DenseObjectLabelStack.from_labels(
            alignment.first
        ).collapse_singleton_plane()
        mask = self._collapse_singleton_mask_plane(alignment.second)
        if labels.shape == mask.shape:
            return (labels, mask)
        if labels.ndim == 3 and mask.ndim == 2 and (labels.shape[1:] == mask.shape):
            labels = DenseObjectLabelStack.from_labels(
                labels
            ).project_xy_plane_without_relabeling()
        if mask.ndim == 3 and labels.ndim == 2 and (mask.shape[1:] == labels.shape):
            mask = self._project_mask_stack(mask)
        if labels.shape != mask.shape:
            raise ValueError(
                f"Dense object labels and mask must share a common geometry after alignment; got {labels.shape} and {mask.shape}."
            )
        return (labels, mask)

    def aligned_stack_context(
        self, slice_count: int
    ) -> DenseObjectLabelMaskStackAlignment | None:
        alignment = SourceSpatialAlignmentPair.from_values(
            self.labels, self.mask
        ).aligned()
        label_stack = self._stack_view(alignment.first, slice_count)
        mask_stack = self._stack_view(alignment.second, slice_count)
        if label_stack is None or mask_stack is None:
            return None
        if label_stack.shape != mask_stack.shape:
            raise ValueError(
                f"Dense object-label and mask stacks must share a common geometry after alignment; got {label_stack.shape} and {mask_stack.shape}."
            )
        return DenseObjectLabelMaskStackAlignment(
            label_stack, mask_stack, alignment.first_projection
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
                f"Mask stack cannot be projected to one XY plane because {conflicts} pixels are positive in multiple planes."
            )
        return np.max(array, axis=0)

    @staticmethod
    def _stack_view(value: Any, slice_count: int) -> Any | None:
        import numpy as np

        array = np.asarray(value)
        if array.ndim == 3 and array.shape[0] == slice_count:
            return np.ascontiguousarray(array)
        if array.ndim == 2:
            return np.ascontiguousarray(
                np.broadcast_to(array, (slice_count, *array.shape))
            )
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
                "Cannot project dense object-label stack with conflicting positive labels at the same XY coordinate."
            )
        return max_label.astype(np.int32, copy=False)


@dataclass(frozen=True, slots=True)
class ConsecutiveObjectLabelIdProjection:
    """Projection from arbitrary positive object IDs to consecutive IDs."""

    positive_label_ids: Any

    @classmethod
    def from_dense_array(
        cls, labels: np.ndarray
    ) -> "ConsecutiveObjectLabelIdProjection":
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
        self, labels: np.ndarray, *, dtype: Any | None = None
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
            1, self.object_count + 1, dtype=dtype
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
    NominalTypeKeyedStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Registered backend strategy for consecutive dense object-label IDs."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_family__ = RegistryFamily(RegistryKeyAttribute.VALUE_TYPE_LABEL)

    @classmethod
    def for_labels(
        cls, labels: object
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
                f"NumpyDenseObjectLabelConsecutiveRelabelingStrategy requires ndarray, got {type(labels).__name__}."
            )
        projection = ConsecutiveObjectLabelIdProjection.from_dense_array(labels)
        return projection.relabel_numpy_array(labels, dtype=dtype)


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
    resolved_ids = (
        declared_object_ids if declared_object_ids is not None else payload_ids
    )
    if resolved_ids:
        ids = tuple((int(object_id) for object_id in resolved_ids))
        if any((object_id <= 0 for object_id in ids)):
            raise ValueError("Object label IDs must be positive integers.")
        return tuple(sorted(dict.fromkeys(ids)))
    resolved_count = (
        declared_object_count if declared_object_count is not None else payload_count
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
    max_present_id = ObjectLabelIdDomainStrategy.for_value(labels).max_present_id(
        labels
    )
    return tuple(range(1, max_present_id + 1))


def dense_object_label_declared_or_extent_id_domain(labels: Any) -> tuple[int, ...]:
    """Return declared object IDs when present, otherwise the dense material extent."""
    declared_domain = (
        ObjectLabelDomainMetadataStrategy.for_value(labels)
        .object_label_domain(labels)
        .explicit_id_domain()
    )
    if declared_domain is not None:
        return declared_domain
    return dense_object_label_extent_id_domain(labels)


def dense_object_label_measurement_row_domain(
    labels: Any, dense_labels: Any
) -> tuple[int, ...]:
    """Return the direct object-measurement row domain for dense labels.

    Undeclared labels use materially present IDs. Declared counts use the
    measured dense extent so positive gaps become explicit missing rows.
    Declared ID sets remain explicit unless the measured/declared labels show
    the compact dense extent emitted by CellProfiler object measurement arrays.
    """
    payload_domain = ObjectLabelDomainMetadataStrategy.for_value(
        labels
    ).object_label_domain(labels)
    if payload_domain.declared_object_count is not None:
        return dense_object_label_extent_id_domain(dense_labels)
    declared_ids = payload_domain.declared_object_ids
    if not declared_ids:
        return dense_object_label_id_domain(labels)
    if len(declared_ids) == 1:
        return declared_ids
    dense_domain = dense_object_label_extent_id_domain(dense_labels)
    if declared_ids == tuple(range(1, len(declared_ids) + 1)):
        return dense_domain
    max_label = dense_domain[-1] if dense_domain else 0
    max_declared = max(declared_ids)
    compact_limit = max(len(declared_ids) * 4, len(declared_ids) + 16)
    if max_declared > len(declared_ids) and max_declared >= max_label:
        if max_declared <= compact_limit:
            return tuple(range(1, max_declared + 1))
    if max_label > len(declared_ids) and max_label <= len(declared_ids) + 16:
        return dense_domain
    return dense_object_label_id_domain(labels)


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
            declared_object_ids=ObjectLabelIdDomainStrategy.for_value(
                labels
            ).present_ids(labels)
        )


@dataclass(frozen=True, slots=True)
class DenseObjectLabelExtentDomainDeclaration(ObjectLabelDomainDeclaration):
    """Declare the dense positive extent represented by transformed labels."""

    def declared_domain(self, source: Any, labels: Any) -> ObjectLabelDomain:
        del source
        return ObjectLabelDomain(
            declared_object_count=ObjectLabelIdDomainStrategy.for_value(
                labels
            ).max_present_id(labels)
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
    payload_domain = (
        ObjectLabelDomainMetadataStrategy.for_value(labels)
        .object_label_domain(labels)
        .with_runtime_declaration_overrides(
            declared_object_count=declared_object_count,
            declared_object_ids=declared_object_ids,
            declared_object_id_domains=declared_object_id_domains,
        )
    )
    resolved_scope = domain_scope or payload_domain.scope
    return ObjectLabelPlaneDomainStrategy.for_enum_member(resolved_scope).plane_domains(
        labels,
        declared_object_count=payload_domain.declared_object_count,
        declared_object_ids=payload_domain.declared_object_ids,
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
    payload_domain = (
        ObjectLabelDomainMetadataStrategy.for_value(labels)
        .object_label_domain(labels)
        .with_runtime_declaration_overrides(
            declared_object_count=declared_object_count,
            declared_object_ids=declared_object_ids,
            declared_object_id_domains=declared_object_id_domains,
        )
    )
    resolved_scope = domain_scope or payload_domain.scope
    return ObjectLabelPlaneDomainStrategy.for_enum_member(
        resolved_scope
    ).identity_domains(
        labels,
        declared_object_count=payload_domain.declared_object_count,
        declared_object_ids=payload_domain.declared_object_ids,
        declared_object_id_domains=payload_domain.declared_object_id_domains,
    )


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
            parent_labels=parent_labels, child_labels=child_labels, kernel=kernel
        )
        return ObjectRelationshipPayloadStrategy.for_context(request).payload(request)
    else:
        parent_array, context_array = DenseObjectLabelPairAligner(
            parent_labels, child_region_labels
        ).aligned()
        child_array, context_array = DenseObjectLabelPairAligner(
            child_labels, context_array
        ).aligned()
    child_ids_array, parent_ids_array = kernel.dominant_parent_ids_by_child(
        parent_array, child_array, context_array
    )
    return ParentChildRelationshipPayload(
        parent_ids=tuple((int(parent_id) for parent_id in parent_ids_array)),
        child_ids=tuple((int(child_id) for child_id in child_ids_array)),
    )


def object_label_lineage_payload(
    parent_labels: Any, child_labels: Any
) -> ParentChildRelationshipPayload:
    """Derive typed parent-child lineage for object-label transforms.

    Shared-geometry transforms use spatial dominance. Geometry-changing
    transforms use preserved object ids, which is the only nominal identity that
    survives nearest-neighbor label resizing without inventing spatial overlap.
    """
    geometry = object_label_lineage_geometry(parent_labels, child_labels)
    return ObjectLabelLineageStrategy.for_enum_member(geometry).payload(
        parent_labels, child_labels
    )


def object_label_lineage_geometry(
    parent_labels: Any, child_labels: Any
) -> ObjectLabelLineageGeometry:
    """Classify the geometry contract for object-label lineage derivation."""
    alignment = SourceSpatialAlignmentPair.from_values(
        parent_labels, child_labels
    ).aligned()
    parent_stack = DenseObjectLabelStack.from_labels(
        alignment.first
    ).collapse_singleton_plane()
    child_stack = DenseObjectLabelStack.from_labels(
        alignment.second
    ).collapse_singleton_plane()
    if parent_stack.shape == child_stack.shape:
        return ObjectLabelLineageGeometry.SHARED_GEOMETRY
    return ObjectLabelLineageGeometry.IDENTITY_DOMAIN


@dataclass(frozen=True, slots=True)
class RelationshipSemantics:
    """Directed relationship semantics between two named runtime entities."""

    source: RelationshipEndpoint
    target: RelationshipEndpoint
    relationship_type: str = "related"

    def __post_init__(self) -> None:
        _require_name(self.relationship_type, "RelationshipSemantics.relationship_type")
        if not isinstance(self.source, RelationshipEndpoint):
            raise TypeError(
                f"RelationshipSemantics.source must be RelationshipEndpoint, got {type(self.source).__name__}."
            )
        if not isinstance(self.target, RelationshipEndpoint):
            raise TypeError(
                f"RelationshipSemantics.target must be RelationshipEndpoint, got {type(self.target).__name__}."
            )

    @classmethod
    def parent_child(
        cls,
        parent_name: str,
        child_name: str,
        *,
        parent_kind: ArtifactType = ObjectLabelsArtifactType,
        child_kind: ArtifactType = ObjectLabelsArtifactType,
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
            f"{field_name} must be one of {', '.join((member.value for member in enum_type))}; got {value!r}."
        ) from exc


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
