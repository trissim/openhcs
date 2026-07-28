"""Nominal runtime measurement table values."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    NamedArtifactPayload,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.source_image_provenance import (
    SourceImageProvenance,
    SourceImageProvenanceFields,
)

from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import replace
from enum import Enum
from metaclass_registry import AutoRegisterMeta
from metaclass_registry import RegistryFamily
from metaclass_registry import RegistryKeyAttribute
from openhcs.core.alias_property import AliasProperty
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.registry_strategies import GeneratedLeafClassSpec
from openhcs.core.registry_strategies import str_enum_member_with_payload
from openhcs.core.runtime_identifier import normalize_runtime_identifier
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
from typing import Any
from typing import ClassVar
import math
import numpy as np
import re


@dataclass(slots=True, kw_only=True)
class MeasurementTable(
    SourceImageProvenanceFields,
    NamedArtifactPayload,
):
    """Native OpenHCS measurement table value."""

    name: str
    rows: ColumnarRows
    source_image_name: str | None = None
    subject: MeasurementSubject
    measurement_feature_owner: type[RuntimeMeasurementFeatureOwner] | None = None

    def __post_init__(self, *source_provenance_values: object) -> None:
        self.absorb_explicit_source_provenance(
            SourceImageProvenance.from_init_values(source_provenance_values)
        )
        self.normalize_source_provenance_fields()
        self.validate_artifact_name()
        if self.source_image_name == "":
            raise ValueError("MeasurementTable.source_image_name cannot be empty.")
        if not isinstance(self.subject, MeasurementSubject):
            raise TypeError("MeasurementTable.subject requires MeasurementSubject.")
        if self.measurement_feature_owner is not None and (
            not isinstance(self.measurement_feature_owner, type)
            or not issubclass(
                self.measurement_feature_owner,
                RuntimeMeasurementFeatureOwner,
            )
        ):
            raise TypeError(
                "MeasurementTable.measurement_feature_owner requires a "
                "RuntimeMeasurementFeatureOwner type."
            )
        if not isinstance(self.rows, ColumnarRows):
            raise TypeError(
                f"MeasurementTable {self.name!r} requires schema-bearing "
                "ColumnarRows, "
                f"got {type(self.rows).__name__}."
            )
        self.rows.validate_fields()
        self.validate_runtime_slice_axis()

    def validate_runtime_slice_axis(self) -> None:
        """Require non-negative values on the canonical runtime slice axis."""
        axis = MeasurementRowAxisField.SLICE_INDEX
        slice_field = axis.value
        if slice_field not in {field.name for field in self.rows.fields}:
            return
        for slice_index in measurement_axis_integer_domain(
            self.rows.column_values(slice_field),
            axis,
        ):
            if slice_index < 0:
                raise ValueError(
                    f"MeasurementTable {self.name!r} has negative slice_index "
                    f"{slice_index}."
                )

    @property
    def runtime_semantic_id(self) -> str:
        """Return the exact semantic partition represented by this table."""

        feature_owner = self.measurement_feature_owner
        feature_owner_name = (
            None
            if feature_owner is None
            else f"{feature_owner.__module__}.{feature_owner.__qualname__}"
        )
        return repr(
            (
                self.subject.identity_token,
                feature_owner_name,
                self.source_image_name,
            )
        )


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


class MeasurementScope(str, Enum):
    """Semantic entity scope for measurement rows."""

    def __new__(
        cls,
        value: str,
        requires_subject_name: bool = False,
    ):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._requires_subject_name = requires_subject_name
        return obj

    ARTIFACT = ("artifact", False)
    IMAGE = ("image", True)
    OBJECT = ("object", True)
    RELATIONSHIP = ("relationship", True)
    EXPERIMENT = ("experiment", False)
    requires_subject_name = AliasProperty[bool]("_requires_subject_name")


@dataclass(frozen=True, slots=True)
class MeasurementScopeSelection:
    """A closed set of measurement scopes selected for one runtime operation."""

    scopes: frozenset[MeasurementScope]

    def __post_init__(self) -> None:
        scopes = frozenset(
            (
                MeasurementScope(
                    scope,
                )
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
                    MeasurementScope(
                        scope,
                    )
                    for scope in scopes
                )
            )
        )

    def includes(self, scope: MeasurementScope | str) -> bool:
        """Return whether this selection includes one semantic scope."""
        return (
            MeasurementScope(
                scope,
            )
            in self.scopes
        )

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


class RuntimeMeasurementFeatureOwner(ABC):
    """Nominal owner of one runtime measurement feature vocabulary."""

    @classmethod
    @abstractmethod
    def owns_primary_measurement_feature_name(cls, feature_name: str) -> bool:
        """Return whether this owner canonically emits the raw feature name."""


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
    measurement_dtype: ClassVar[type[object]] = int


class ObjectGroupInvariantFeatureMarker(RuntimeMeasurementFeatureSemanticMarker):
    """Generic marker for object features invariant under invocation grouping."""


class ObjectReferenceFeatureMarker(ObjectGroupInvariantFeatureMarker):
    """Generic marker for measurements whose values reference object identities."""


class ObjectIdentifierFeatureMarker(ObjectReferenceFeatureMarker):
    """Generic marker for an object's own identifier measurement."""

    family_qualifier = "identifier"


class MeasuredObjectAnchorFeatureMarker(RuntimeMeasurementFeatureSemanticMarker):
    """Generic marker for features proving that an object row was measured."""

    family_qualifier = "object"


class ObjectLocationFeatureMarker(ObjectGroupInvariantFeatureMarker):
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


class RuntimeMeasurementFeatureDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Registered parser, renderer, and semantic owner for dynamic feature names."""

    __registry_key__ = "declaration_key"
    __skip_if_no_key__ = True
    declaration_key: ClassVar[str | None] = None
    semantic_marker_types: ClassVar[
        tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]
    ] = ()

    @classmethod
    def require_registered(
        cls,
        declaration_type: type["RuntimeMeasurementFeatureDeclaration"],
    ) -> type["RuntimeMeasurementFeatureDeclaration"]:
        """Return a declaration registered with the dynamic-feature authority."""
        if not isinstance(declaration_type, type) or not issubclass(
            declaration_type,
            RuntimeMeasurementFeatureDeclaration,
        ):
            raise TypeError(
                "Measurement feature declaration must inherit "
                "RuntimeMeasurementFeatureDeclaration."
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
    ) -> tuple[tuple[type["RuntimeMeasurementFeatureDeclaration"], object], ...]:
        """Return registered declarations that parse ``feature_name`` exactly."""
        return tuple(
            (declaration_type, identity)
            for declaration_type in cls.__registry__.values()
            for identity in (declaration_type.from_feature_name(feature_name),)
            if identity is not None
        )

    @classmethod
    def semantic_marker_types_for(
        cls,
        feature_name: str,
    ) -> tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]:
        """Return semantic markers declared by exact dynamic-name owners."""
        return tuple(
            dict.fromkeys(
                marker_type
                for declaration_type, _identity in cls.matching_declarations(
                    feature_name
                )
                for marker_type in declaration_type.semantic_marker_types
            )
        )

    @classmethod
    def feature_has_semantic_marker(
        cls,
        feature_name: str,
        marker_type: type[RuntimeMeasurementFeatureSemanticMarker],
    ) -> bool:
        """Return whether an exact dynamic-name declaration carries ``marker_type``."""
        if not isinstance(marker_type, type) or not issubclass(
            marker_type,
            RuntimeMeasurementFeatureSemanticMarker,
        ):
            raise TypeError(
                "marker_type must inherit RuntimeMeasurementFeatureSemanticMarker."
            )
        return any(
            issubclass(declared_marker_type, marker_type)
            for declared_marker_type in cls.semantic_marker_types_for(feature_name)
        )

    @classmethod
    def indexed_suffix_token_width_for(
        cls,
        feature_tokens: tuple[str, ...],
    ) -> int | None:
        """Return the unique declared trailing descriptor width, when present."""
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
                "Measurement feature declarations disagree on indexed suffix width "
                f"for {feature_tokens!r}: {tuple(sorted(suffix_widths))!r}."
            )
        return next(iter(suffix_widths))

    @classmethod
    @abstractmethod
    def from_feature_name(
        cls,
        feature_name: str,
    ) -> object | None:
        """Parse ``feature_name`` into this declaration's nominal identity."""

    @classmethod
    @abstractmethod
    def feature_name(
        cls,
        identity: object,
    ) -> str:
        """Render one nominal feature identity."""

    @classmethod
    def indexed_suffix_token_width(
        cls,
        feature_tokens: tuple[str, ...],
    ) -> int | None:
        """Return a trailing descriptor width, or none for non-indexed features."""
        del feature_tokens
        return None


class RuntimeMeasurementIndexedDescriptorDeclaration(
    RuntimeMeasurementFeatureDeclaration,
    ABC,
):
    """Dynamic feature declaration whose identity contains an indexed descriptor."""

    @classmethod
    def is_declaration_type(
        cls,
        declaration_type: type[RuntimeMeasurementFeatureDeclaration],
    ) -> bool:
        """Return whether a dynamic declaration owns an indexed descriptor."""
        return issubclass(declaration_type, cls)

    @classmethod
    def require_registered(
        cls,
        declaration_type: type[RuntimeMeasurementFeatureDeclaration],
    ) -> type["RuntimeMeasurementIndexedDescriptorDeclaration"]:
        """Return a registered indexed-descriptor declaration type."""
        registered = super().require_registered(declaration_type)
        if not cls.is_declaration_type(registered):
            raise TypeError(
                "Indexed descriptor declaration must inherit "
                "RuntimeMeasurementIndexedDescriptorDeclaration."
            )
        return registered

    @classmethod
    def matching_declarations(
        cls,
        feature_name: str,
    ) -> tuple[
        tuple[type["RuntimeMeasurementIndexedDescriptorDeclaration"], object], ...
    ]:
        """Return indexed declarations that parse ``feature_name`` exactly."""
        return tuple(
            (declaration_type, identity)
            for declaration_type, identity in super().matching_declarations(
                feature_name
            )
            if cls.is_declaration_type(declaration_type)
        )

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
        measurement_row_field_name: str | None = None,
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
        member.__dict__["_measurement_row_field_name"] = (
            normalize_runtime_identifier(value)
            if measurement_row_field_name is None
            else str(measurement_row_field_name)
        )
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
    measurement_row_field_name = AliasProperty[str]("_measurement_row_field_name")

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
        return frozenset(
            (
                *(field.value for field in cls),
                *(component.value for component in AllComponents),
            )
        )

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


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowIdentityContract:
    """Declarative identity-field precedence for measurement table rows."""

    primary_image_fields: frozenset[str] = frozenset({"slice_index"})
    fallback_image_fields: frozenset[str] = frozenset({"image_number", "image_id"})
    object_identity_fields: tuple[str, ...] = (
        MeasurementRowAxisField.object_id_field_names()
    )

    def __post_init__(self) -> None:
        primary_image_fields = frozenset(
            normalize_runtime_identifier(field_name)
            for field_name in self.primary_image_fields
            if str(field_name).strip()
        )
        fallback_image_fields = frozenset(
            normalize_runtime_identifier(field_name)
            for field_name in self.fallback_image_fields
            if str(field_name).strip()
        )
        object_identity_fields = tuple(
            dict.fromkeys(
                normalize_runtime_identifier(field_name)
                for field_name in self.object_identity_fields
                if str(field_name).strip()
            )
        )
        overlap = primary_image_fields & fallback_image_fields
        if overlap:
            raise ValueError(
                "RuntimeMeasurementRowIdentityContract fields must be disjoint: "
                f"{sorted(overlap)!r}."
            )
        if not object_identity_fields:
            raise ValueError(
                "RuntimeMeasurementRowIdentityContract.object_identity_fields "
                "cannot be empty."
            )
        object.__setattr__(self, "primary_image_fields", primary_image_fields)
        object.__setattr__(self, "fallback_image_fields", fallback_image_fields)
        object.__setattr__(self, "object_identity_fields", object_identity_fields)

    @property
    def image_identity_fields(self) -> frozenset[str]:
        """Return every field that can identify an image row."""
        return self.primary_image_fields | self.fallback_image_fields

    def selected_image_identity_fields(
        self,
        normalized_present_fields: frozenset[str],
    ) -> frozenset[str]:
        """Return the identity fields that own a row under this contract."""
        primary_fields = normalized_present_fields & self.primary_image_fields
        if primary_fields:
            return primary_fields
        return normalized_present_fields & self.fallback_image_fields

    def selected_object_identity_field(
        self,
        normalized_present_fields: frozenset[str],
    ) -> str | None:
        """Return the first declared object-identity field present in a row."""
        return next(
            (
                field_name
                for field_name in self.object_identity_fields
                if field_name in normalized_present_fields
            ),
            None,
        )


DEFAULT_RUNTIME_MEASUREMENT_ROW_IDENTITY_CONTRACT = (
    RuntimeMeasurementRowIdentityContract()
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


class ObjectFeatureArrayDomain(str, Enum):
    """How a feature array indexes values for an object-feature table."""

    MEASURED_OBJECT_ID = "measured_object_id"
    LABEL_ID = "label_id"
    ROW_ORDINAL = "row_ordinal"


class ObjectFeatureMissingValue(str, Enum):
    """How an object-feature table represents unmeasured feature values."""

    def __new__(cls, value: str, scalar: float):
        member = str.__new__(cls, value)
        member._value_ = value
        member.scalar = scalar
        return member

    NAN = ("nan", float(np.nan))
    ZERO = ("zero", 0.0)
    scalar: float


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


@dataclass(frozen=True, slots=True)
class ObjectFeatureValueTable:
    """Wide object-feature values aligned onto a declared object-id domain."""

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
                    self.feature_missing_value(feature_name).scalar,
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


@dataclass(frozen=True, slots=True)
class MeasurementSubject:
    """Entity measured by a measurement table."""

    scope: MeasurementScope
    name: str | None = None
    id_field: str | None = None

    def __post_init__(self) -> None:
        scope = MeasurementScope(
            self.scope,
        )
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

    @property
    def object_name(self) -> str | None:
        """Return the concrete object set represented by this subject, if any."""
        return self.name if self.scope is MeasurementScope.OBJECT else None

    @property
    def object_id_field(self) -> str | None:
        """Return the object identifier field represented by this subject."""
        return self.id_field if self.scope is MeasurementScope.OBJECT else None

    @property
    def identity_token(self) -> str:
        """Return the stable identity of this measurement subject."""
        return ":".join(
            (
                "measurement_subject",
                self.scope.value,
                self.name or "",
                self.id_field or "",
            )
        )
