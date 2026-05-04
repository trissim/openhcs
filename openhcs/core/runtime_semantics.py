"""Generic semantic contracts for typed runtime artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openhcs.core.artifacts import ArtifactKind, ArtifactPayloadShape


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
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._payload_shape = payload_shape
        return obj

    DENSE_LABELS = ("dense_labels", ArtifactPayloadShape.ARRAY)
    SPARSE_IJV = ("sparse_ijv", ArtifactPayloadShape.TABLE)

    @property
    def payload_shape(self) -> ArtifactPayloadShape:
        return self._payload_shape


class ObjectLabelVariant(str, Enum):
    """Named semantic variants carried by an object-label artifact."""

    FINAL = "final"
    UNEDITED = "unedited"
    SMALL_REMOVED = "small_removed"


@dataclass(frozen=True, slots=True)
class ObjectLabelDomain:
    """Declared object-label identity domain metadata."""

    declared_object_count: int | None = None
    declared_object_ids: tuple[int, ...] = ()


class ObjectLabelDomainMetadata(ABC):
    """Nominal provider for object-label ID domain metadata."""

    @abstractmethod
    def object_label_domain(self) -> ObjectLabelDomain:
        """Return the declared object-label identity domain."""


class SpatialGridOrdering(str, Enum):
    """Primary axis used when numbering positions in a spatial grid."""

    BY_ROWS = "rows"
    BY_COLUMNS = "columns"


class MeasurementScope(str, Enum):
    """Semantic entity scope for measurement rows."""

    def __new__(cls, value: str, requires_subject_name: bool = False):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._requires_subject_name = requires_subject_name
        return obj

    ARTIFACT = ("artifact", False)
    IMAGE = ("image", True)
    OBJECT = ("object", True)
    RELATIONSHIP = ("relationship", True)
    EXPERIMENT = ("experiment", False)

    @property
    def requires_subject_name(self) -> bool:
        return self._requires_subject_name


class PairMeasurementFeature(str, Enum):
    """Generic pairwise measurement features with direction semantics."""

    CORRELATION = "correlation"
    REGRESSION_SLOPE = "slope"
    OVERLAP = "overlap"
    COSTES_MANDERS = "costes"
    MANDERS = "manders"
    RANK_WEIGHTED_COLOCALIZATION = "rwc"
    OVERLAP_K = "k"


class ObjectShapeMeasurementFeature(str, Enum):
    """Canonical object shape measurement field names.

    The values are the runtime table field names used by object-shape producers.
    Keeping them in core avoids module-local string schemas and lets dialect
    layers focus on external naming projection.
    """

    AREA = "Area"
    PERIMETER = "Perimeter"
    VOLUME = "Volume"
    SURFACE_AREA = "SurfaceArea"
    ECCENTRICITY = "Eccentricity"
    SOLIDITY = "Solidity"
    CONVEX_AREA = "ConvexArea"
    EXTENT = "Extent"
    CENTER_X = "Center_X"
    CENTER_Y = "Center_Y"
    CENTER_Z = "Center_Z"
    BOUNDING_BOX_AREA = "BoundingBoxArea"
    BOUNDING_BOX_VOLUME = "BoundingBoxVolume"
    BOUNDING_BOX_MINIMUM_X = "BoundingBoxMinimum_X"
    BOUNDING_BOX_MAXIMUM_X = "BoundingBoxMaximum_X"
    BOUNDING_BOX_MINIMUM_Y = "BoundingBoxMinimum_Y"
    BOUNDING_BOX_MAXIMUM_Y = "BoundingBoxMaximum_Y"
    BOUNDING_BOX_MINIMUM_Z = "BoundingBoxMinimum_Z"
    BOUNDING_BOX_MAXIMUM_Z = "BoundingBoxMaximum_Z"
    EULER_NUMBER = "EulerNumber"
    FORM_FACTOR = "FormFactor"
    MAJOR_AXIS_LENGTH = "MajorAxisLength"
    MINOR_AXIS_LENGTH = "MinorAxisLength"
    ORIENTATION = "Orientation"
    COMPACTNESS = "Compactness"
    MAXIMUM_RADIUS = "MaximumRadius"
    MEDIAN_RADIUS = "MedianRadius"
    MEAN_RADIUS = "MeanRadius"
    MIN_FERET_DIAMETER = "MinFeretDiameter"
    MAX_FERET_DIAMETER = "MaxFeretDiameter"
    EQUIVALENT_DIAMETER = "EquivalentDiameter"
    SPATIAL_MOMENT = "SpatialMoment"
    CENTRAL_MOMENT = "CentralMoment"
    NORMALIZED_MOMENT = "NormalizedMoment"
    HU_MOMENT = "HuMoment"
    INERTIA_TENSOR = "InertiaTensor"
    INERTIA_TENSOR_EIGENVALUES = "InertiaTensorEigenvalues"
    ZERNIKE = "Zernike"


class MeasurementRowAxisField(str, Enum):
    """Canonical row-axis fields for long/tall measurement tables."""

    SLICE_INDEX = "slice_index"
    FEATURE_NAME = "feature_name"
    MEASUREMENT_NAME = "measurement_name"
    OUTPUT_NAME = "output_name"
    OBJECT_NAME = "object_name"
    SOURCE_IMAGE_NAME = "source_image_name"
    BIN_INDEX = "bin_index"
    BIN_COUNT = "bin_count"
    SCALE = "scale"
    DIRECTION = "direction"
    GRAY_LEVELS = "gray_levels"
    ZERNIKE_N = "n"
    ZERNIKE_M = "m"


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
            fields.extend(_zernike_feature_names(max_order=9))
    else:
        fields.extend(feature.value for feature in _OBJECT_SHAPE_STANDARD_3D_FIELDS)
        if calculate_advanced:
            fields.append(ObjectShapeMeasurementFeature.SOLIDITY.value)
    return tuple(dict.fromkeys(fields))


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


def _zernike_feature_names(*, max_order: int) -> tuple[str, ...]:
    return tuple(
        indexed_measurement_feature_name(ObjectShapeMeasurementFeature.ZERNIKE, n, m)
        for n in range(max_order + 1)
        for m in range(n % 2, n + 1, 2)
    )


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


def aligned_dense_object_label_arrays(
    first_labels: Any,
    second_labels: Any,
) -> tuple[Any, Any]:
    """Align two dense object-label payloads to a common label geometry.

    OpenHCS can carry labels as stacks, while compatibility shims and many
    module semantics operate on one dense XY label plane. Projection is only
    allowed when it is deterministic: singleton stacks collapse, identical
    shapes pass through, and stack-to-plane projection rejects conflicting
    positive labels at the same XY coordinate.
    """
    import numpy as np

    first = _collapse_singleton_dense_label_stack(
        np.asarray(first_labels).astype(np.int32, copy=False)
    )
    second = _collapse_singleton_dense_label_stack(
        np.asarray(second_labels).astype(np.int32, copy=False)
    )
    if first.shape == second.shape:
        return first, second

    if first.ndim == 3 and second.ndim == 2 and first.shape[1:] == second.shape:
        first = project_dense_object_label_stack(first)
    if second.ndim == 3 and first.ndim == 2 and second.shape[1:] == first.shape:
        second = project_dense_object_label_stack(second)
    if first.shape != second.shape:
        raise ValueError(
            "Dense object-label payloads must share a common geometry after "
            f"alignment; got {first.shape} and {second.shape}."
        )
    return first, second


def dense_object_label_id_domain(
    labels: Any,
    *,
    declared_object_count: int | None = None,
    declared_object_ids: tuple[int, ...] | list[int] | None = None,
) -> tuple[int, ...]:
    """Return the semantic object-id domain represented by dense labels.

    Dense label images commonly encode object identity as labels 1..N. Some
    producers also declare object identities that have no pixels in the current
    image. The returned domain preserves those declared IDs and otherwise uses
    the dense label convention up to the maximum positive label ID.
    """
    import numpy as np

    payload_ids: tuple[int, ...] = ()
    payload_count: int | None = None
    if isinstance(labels, ObjectLabelDomainMetadata):
        payload_domain = labels.object_label_domain()
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
    else:
        count = 0

    label_array = np.asarray(labels)
    if label_array.size and (
        np.issubdtype(label_array.dtype, np.number)
        or np.issubdtype(label_array.dtype, np.bool_)
    ):
        positive_labels = label_array[label_array > 0]
        if positive_labels.size:
            count = max(count, int(np.max(positive_labels)))
    return tuple(range(1, count + 1))


def project_dense_object_label_stack(labels: Any) -> Any:
    """Project a dense label stack to one XY plane without relabeling."""
    import numpy as np

    stack = np.asarray(labels).astype(np.int32, copy=False)
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


def _collapse_singleton_dense_label_stack(labels: Any) -> Any:
    if hasattr(labels, "ndim") and labels.ndim == 3 and labels.shape[0] == 1:
        return labels[0]
    return labels


def object_label_parent_child_payload(
    parent_labels: Any,
    child_labels: Any,
    *,
    child_region_labels: Any | None = None,
) -> ParentChildRelationshipPayload:
    """Derive parent-child ids from dense object-label images.

    ``child_region_labels`` lets callers use one label image to enumerate child
    ids while selecting the pixels that define each child's parent context.
    """
    import numpy as np

    if child_region_labels is None:
        parent_array, child_array = aligned_dense_object_label_arrays(
            parent_labels,
            child_labels,
        )
        context_array = child_array
    else:
        parent_array, context_array = aligned_dense_object_label_arrays(
            parent_labels,
            child_region_labels,
        )
        child_array, context_array = aligned_dense_object_label_arrays(
            child_labels,
            context_array,
        )

    child_ids_array, parent_ids_array = _dominant_parent_ids_by_child(
        parent_array,
        child_array,
        context_array,
    )
    return ParentChildRelationshipPayload(
        parent_ids=tuple(int(parent_id) for parent_id in parent_ids_array),
        child_ids=tuple(int(child_id) for child_id in child_ids_array),
    )


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


def _dominant_positive_label(labels: Any) -> int:
    import numpy as np

    positive_labels = np.asarray(labels)[np.asarray(labels) > 0].astype(np.int64)
    if positive_labels.size == 0:
        return 0
    counts = np.bincount(positive_labels)
    return int(np.argmax(counts))


def _dominant_parent_ids_by_child(
    parent_array: Any,
    child_array: Any,
    context_array: Any,
) -> tuple[Any, Any]:
    import numpy as np

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


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
