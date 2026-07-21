"""Nominal measurement features authorities for CellProfiler modules."""

from __future__ import annotations
from abc import ABC
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    TypeVar,
)
from openhcs.core.equivalence.policy import normalize_runtime_identifier
from openhcs.core.runtime_measurements import (
    MeasuredObjectAnchorFeatureMarker,
    MeasurementScope,
    MeasurementStatistic,
    ObjectCalculatedFeatureMarker,
    ObjectCountFeatureMarker,
    ObjectIdentifierFeatureMarker,
    ObjectIntensityFeatureMarker,
    ObjectLocationFeatureMarker,
    ObjectShapeDescriptorFeatureMarker,
    RuntimeMeasurementFeature,
    RuntimeMeasurementFeatureOwner,
    RuntimeMeasurementFeatureRelationDeclaration,
    RuntimeMeasurementFeatureSemanticMarker,
)
from openhcs.core.runtime_tabular_values import FieldSpec

if TYPE_CHECKING:
    from openhcs.core.artifacts import ArtifactSpec
    from openhcs.core.callable_contract import CallableContract
    from openhcs.core.runtime_measurements import MeasurementTable
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )
    from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
        RelationshipMeasurementRows,
    )


class CellProfilerModuleAuthority:
    """Nominal semantic capability inherited by CellProfiler module declarations."""


AuthorityT = TypeVar("AuthorityT", bound=CellProfilerModuleAuthority)


class CellProfilerMeasurementFeatureMarker(
    RuntimeMeasurementFeatureSemanticMarker,
    CellProfilerModuleAuthority,
    ABC,
):
    """Semantic marker authority for module-owned measurement features."""

    @classmethod
    def feature_members(
        cls,
        module_type: type["CellProfilerModule"],
    ) -> tuple[RuntimeMeasurementFeature, ...]:
        """Return module feature members carrying this marker."""
        return tuple(
            feature
            for feature_type in module_type.measurement_feature_types()
            for feature in feature_type
            if cls.matches_feature(feature)
        )

    @classmethod
    def matches_feature_key(
        cls,
        module_type: type["CellProfilerModule"],
        key: Any,
        dialect: Any,
    ) -> bool:
        """Return whether ``key`` is an object value feature owned by this marker."""
        del dialect
        if key.subject.scope is not MeasurementScope.OBJECT:
            return False
        if key.statistic != MeasurementStatistic.VALUE.value:
            return False
        feature_name = normalize_runtime_identifier(key.feature_name)
        for feature in cls.feature_members(module_type):
            descriptor_declarations = feature.indexed_descriptor_declarations()
            if descriptor_declarations:
                if any(
                    declaration_type.from_feature_name(key.feature_name) is not None
                    for declaration_type in descriptor_declarations
                ):
                    return True
                continue
            family = feature.feature_family()
            if feature_name == family or feature_name.startswith(f"{family}_"):
                return True
        return False


class ObjectCountFeature(
    CellProfilerMeasurementFeatureMarker, ObjectCountFeatureMarker
):
    """Object-count feature marker."""


class ObjectIdentifierFeature(
    CellProfilerMeasurementFeatureMarker,
    ObjectIdentifierFeatureMarker,
):
    """Object-identifier feature marker."""


class MeasuredObjectAnchorFeature(
    CellProfilerMeasurementFeatureMarker,
    MeasuredObjectAnchorFeatureMarker,
):
    """Feature marker proving that an object row is measured."""


class ObjectLocationFeature(
    CellProfilerMeasurementFeatureMarker,
    ObjectLocationFeatureMarker,
):
    """Object-location feature marker."""


class IntensityFeature(
    CellProfilerMeasurementFeatureMarker, ObjectIntensityFeatureMarker
):
    """Object-intensity feature marker."""


class ObjectCalculatedFeature(
    CellProfilerMeasurementFeatureMarker,
    ObjectCalculatedFeatureMarker,
):
    """Calculated object-feature marker."""


class ShapeDescriptorFeature(
    CellProfilerMeasurementFeatureMarker,
    ObjectShapeDescriptorFeatureMarker,
):
    """Object shape-descriptor feature marker."""


class CellProfilerMeasurementFeatureOwner(RuntimeMeasurementFeatureOwner):
    measurement_feature_part_aliases: ClassVar[
        Mapping[tuple[str, ...], tuple[tuple[str, ...], ...]]
    ] = {}
    measurement_feature_part_rewrites: ClassVar[
        Mapping[tuple[str, ...], tuple[str, ...]]
    ] = {}
    measurement_category_prefixes: ClassVar[tuple[tuple[str, ...], ...]] = ()
    measurement_source_feature_prefixes: ClassVar[tuple[tuple[str, ...], ...]] = ()
    calculated_measurement_feature_prefixes: ClassVar[tuple[tuple[str, ...], ...]] = ()
    numbered_measurement_feature_prefix_aliases: ClassVar[
        Mapping[str, tuple[str, ...]]
    ] = {}
    directional_pair_feature_aliases: ClassVar[Mapping[str, tuple[str, int]]] = {}
    scale_qualified_measurement_feature_prefixes: ClassVar[
        tuple[tuple[str, ...], ...]
    ] = ()
    pair_correlation_feature_name: ClassVar[str | None] = None
    pair_regression_slope_feature_name: ClassVar[str | None] = None
    undirected_pair_feature_names: ClassVar[frozenset[str]] = frozenset()
    threshold_sensitive_pair_feature_names: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def database_measurement_field(cls, field: FieldSpec) -> FieldSpec:
        """Project a runtime field through its module-owned database declaration."""

        from openhcs.interop.cellprofiler.runtime.measurement_rows import (
            FormattingMeasurementFeatureTemplate,
        )

        matches = tuple(
            feature
            for feature_type in cls.declared_authority_types(
                FormattingMeasurementFeatureTemplate
            )
            for feature in feature_type
            if feature.matches_feature_name(field.name)
            and type(feature).database_measurement_dtype() is not None
        )
        if len(matches) > 1:
            raise ValueError(
                f"{cls.__name__} declares overlapping database measurement "
                f"templates for {field.name!r}."
            )
        if not matches:
            return field
        return matches[0].database_field_spec(
            field.name,
            required=field.required,
        )

    @classmethod
    def declared_authority_types(
        cls,
        authority_root: type[AuthorityT],
    ) -> tuple[type[AuthorityT], ...]:
        """Return most-derived authority types declared by this module's MRO."""
        if not isinstance(authority_root, type) or not issubclass(
            authority_root,
            CellProfilerModuleAuthority,
        ):
            raise TypeError(
                f"{cls.__name__} authority must inherit CellProfilerModuleAuthority."
            )
        matching_authority_types = tuple(
            dict.fromkeys(
                candidate_type
                for candidate_type in cls.__mro__
                if candidate_type is not cls
                and candidate_type is not authority_root
                and candidate_type is not CellProfilerModuleAuthority
                and issubclass(candidate_type, authority_root)
            )
        ) + tuple(
            dict.fromkeys(
                nested_type
                for owner_type in cls.__mro__
                for nested_type in owner_type.__dict__.values()
                if isinstance(nested_type, type)
                and nested_type is not authority_root
                and nested_type is not CellProfilerModuleAuthority
                and issubclass(nested_type, authority_root)
            )
        )
        return tuple(
            candidate_type
            for candidate_type in matching_authority_types
            if not any(
                other_type is not candidate_type
                and issubclass(other_type, candidate_type)
                for other_type in matching_authority_types
            )
        )

    @classmethod
    def derived_measurement_feature_relation_declarations(
        cls,
    ) -> tuple[RuntimeMeasurementFeatureRelationDeclaration, ...]:
        """Return measurement-feature relations derived from module-owned markers."""
        return ()

    @classmethod
    def measurement_feature_relation_declarations(
        cls,
    ) -> tuple[RuntimeMeasurementFeatureRelationDeclaration, ...]:
        """Return producer-owned measurement-feature relations from modules."""
        return tuple(
            (
                relation
                for module_type in cls.__registry__.values()
                for feature_type in module_type.measurement_feature_types()
                for feature in feature_type
                for relation in feature.relation_declarations()
            )
        ) + tuple(
            relation
            for module_type in cls.__registry__.values()
            for relation in module_type.derived_measurement_feature_relation_declarations()
        )

    @classmethod
    def declared_measurement_feature_family_parts(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return feature families owned by one module declaration."""

        return tuple(
            dict.fromkeys(
                tuple(part for part in feature.feature_family().split("_") if part)
                for feature_type in cls.measurement_feature_types()
                for feature in feature_type
            )
        )

    @classmethod
    def source_qualified_measurement_feature_family_parts(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return module-owned feature families that carry source names."""
        return tuple(
            family
            for module_type in cls.__registry__.values()
            for family in module_type.declared_source_qualified_measurement_feature_family_parts()
        )

    @classmethod
    def declared_source_qualified_measurement_feature_family_parts(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return source-qualified families owned by one module declaration."""

        return ()

    @classmethod
    def source_qualified_measurement_feature_types(
        cls,
    ) -> tuple[type[RuntimeMeasurementFeature], ...]:
        """Return measurement feature enums whose rows are source-qualified."""
        return ()

    @classmethod
    def measurement_feature_marker_types_for_key(
        cls,
        key: Any,
        dialect: Any,
    ) -> tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]:
        """Return module-owned semantic marker authorities matching ``key``."""
        from openhcs.core.equivalence.keys import RuntimeMeasurementFeatureKey

        del dialect
        if not isinstance(key, RuntimeMeasurementFeatureKey):
            raise TypeError(
                "CellProfilerModule.measurement_feature_marker_types_for_key "
                "requires RuntimeMeasurementFeatureKey."
            )
        return cls._measurement_feature_marker_types_for_key_payload(
            key.to_cache_payload()
        )

    @classmethod
    @lru_cache(maxsize=4096)
    def _measurement_feature_marker_types_for_key_payload(
        cls,
        key_payload: object,
    ) -> tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]:
        """Return module-owned marker authorities keyed by semantic feature payload."""
        from openhcs.core.equivalence.keys import RuntimeMeasurementFeatureKey

        key = RuntimeMeasurementFeatureKey.from_cache_payload(key_payload)
        return tuple(
            dict.fromkeys(
                authority_type
                for module_type in cls.__registry__.values()
                for authority_type in module_type.__mro__
                if authority_type is not CellProfilerMeasurementFeatureMarker
                and authority_type is not RuntimeMeasurementFeatureSemanticMarker
                and isinstance(authority_type, type)
                and issubclass(authority_type, CellProfilerMeasurementFeatureMarker)
                if authority_type.matches_feature_key(module_type, key, None)
            )
        )

    @classmethod
    @lru_cache(maxsize=None)
    def measurement_feature_types(
        cls,
    ) -> tuple[type[RuntimeMeasurementFeature], ...]:
        """Return measurement feature enums declared on this module's MRO."""
        return tuple(
            dict.fromkeys(
                feature_type
                for owner_type in cls.__mro__
                for feature_type in owner_type.__dict__.values()
                if isinstance(feature_type, type)
                and issubclass(feature_type, RuntimeMeasurementFeature)
                and feature_type is not RuntimeMeasurementFeature
            )
        )

    @classmethod
    def owns_measurement_feature_name(cls, feature_name: str) -> bool:
        """Return whether this module declaration emits the feature family."""

        feature_parts = tuple(
            part
            for part in normalize_runtime_identifier(feature_name).split("_")
            if part
        )
        family_suffixes = tuple(
            feature_parts[len(prefix) :]
            for prefix in cls.measurement_category_prefixes
            if len(feature_parts) > len(prefix)
            and feature_parts[: len(prefix)] == prefix
        )
        if not family_suffixes:
            return False
        return any(
            cls.owns_measurement_feature_parts(suffix) for suffix in family_suffixes
        )

    @classmethod
    def owns_measurement_feature_parts(cls, feature_parts: tuple[str, ...]) -> bool:
        """Return whether normalized parts name one declared feature family."""

        feature_types = cls.measurement_feature_types()
        if not feature_types:
            return True
        family_parts = tuple(
            tuple(
                part
                for part in normalize_runtime_identifier(
                    feature.feature_family()
                ).split("_")
                if part
            )
            for feature_type in feature_types
            for feature in feature_type
        )
        return any(feature_parts[: len(family)] == family for family in family_parts)

    @classmethod
    def owns_primary_measurement_feature_name(cls, feature_name: str) -> bool:
        """Return whether this module canonically emits the raw feature name."""

        if not cls.measurement_category_prefixes:
            return False
        feature_parts = tuple(
            part
            for part in normalize_runtime_identifier(feature_name).split("_")
            if part
        )
        primary_prefix = cls.measurement_category_prefixes[0]
        if (
            len(feature_parts) <= len(primary_prefix)
            or feature_parts[: len(primary_prefix)] != primary_prefix
        ):
            return False
        return cls.owns_measurement_feature_parts(feature_parts[len(primary_prefix) :])

    @classmethod
    def experiment_measurement_tables(
        cls,
        tables: Sequence["MeasurementTable"],
    ) -> tuple["MeasurementTable", ...]:
        """Derive module-owned experiment measurements from complete plate tables."""

        del cls, tables
        return ()

    @classmethod
    def derive_experiment_measurement_tables(
        cls,
        tables: Sequence["MeasurementTable"],
    ) -> tuple["MeasurementTable", ...]:
        """Dispatch plate reductions to the exact recorded measurement owners."""

        table_sequence = tuple(tables)
        owners = tuple(
            dict.fromkeys(
                module_type
                for table in table_sequence
                for module_type in cls.__registry__.values()
                if table.measurement_feature_owner is module_type
            )
        )
        return tuple(
            experiment_table
            for owner in owners
            for experiment_table in owner.experiment_measurement_tables(
                tuple(
                    table
                    for table in table_sequence
                    if table.measurement_feature_owner is owner
                )
            )
        )

    @classmethod
    @lru_cache(maxsize=1)
    def alternative_measurement_feature_part_aliases(
        cls,
    ) -> Mapping[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        """Return module-owned alternative feature-family aliases."""
        aliases: dict[tuple[str, ...], tuple[tuple[str, ...], ...]] = {}
        for module_type in cls.__registry__.values():
            for (
                source,
                alternatives,
            ) in module_type.measurement_feature_part_aliases.items():
                aliases[tuple(source)] = tuple((tuple(alias) for alias in alternatives))
        return aliases

    @classmethod
    @lru_cache(maxsize=1)
    def measurement_feature_part_rewrite_declarations(
        cls,
    ) -> Mapping[tuple[str, ...], tuple[str, ...]]:
        """Return module-owned direct feature-family rewrites."""
        aliases: dict[tuple[str, ...], tuple[str, ...]] = {}
        declarations = (
            (
                tuple(source),
                tuple(target),
                module_type,
            )
            for module_type in cls.__registry__.values()
            for source, target in module_type.measurement_feature_part_rewrites.items()
        )
        derived_declarations = (
            (
                tuple(
                    part
                    for part in normalize_runtime_identifier(
                        feature.measurement_row_field_name
                    ).split("_")
                    if part
                ),
                tuple(part for part in feature.feature_family().split("_") if part),
                module_type,
            )
            for module_type in cls.__registry__.values()
            for feature_type in module_type.measurement_feature_types()
            for feature in feature_type
            if not feature.relations
            if normalize_runtime_identifier(feature.measurement_row_field_name)
            != feature.feature_family()
        )
        for source, target, owner in (*declarations, *derived_declarations):
            existing = aliases.get(source)
            if existing is not None and existing != target:
                raise ValueError(
                    "CellProfiler measurement feature declarations disagree for "
                    f"{source!r}: {existing!r} versus {target!r} on "
                    f"{owner.__name__}."
                )
            aliases[source] = target
        return aliases

    @classmethod
    @lru_cache(maxsize=1)
    def measurement_category_prefix_declarations(cls) -> tuple[tuple[str, ...], ...]:
        """Return module-owned CellProfiler measurement category prefixes."""
        return tuple(
            dict.fromkeys(
                prefix
                for module_type in cls.__registry__.values()
                for prefix in module_type.measurement_category_prefixes
            )
        )

    @classmethod
    @lru_cache(maxsize=1)
    def primary_measurement_category_prefix_declarations(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return each module's canonical measurement category prefix."""
        return tuple(
            dict.fromkeys(
                module_type.measurement_category_prefixes[0]
                for module_type in cls.__registry__.values()
                if module_type.measurement_category_prefixes
            )
        )

    @classmethod
    @lru_cache(maxsize=1)
    def measurement_source_feature_prefix_declarations(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return module-owned source-qualified feature prefixes."""
        return tuple(
            dict.fromkeys(
                prefix
                for module_type in cls.__registry__.values()
                for prefix in module_type.measurement_source_feature_prefixes
            )
        )

    @classmethod
    @lru_cache(maxsize=1)
    def calculated_measurement_feature_prefix_declarations(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return module-owned calculated feature prefixes."""
        return tuple(
            dict.fromkeys(
                prefix
                for module_type in cls.__registry__.values()
                for prefix in module_type.calculated_measurement_feature_prefixes
            )
        )

    @classmethod
    @lru_cache(maxsize=1)
    def numbered_measurement_feature_prefix_alias_declarations(
        cls,
    ) -> Mapping[str, tuple[str, ...]]:
        """Return module-owned numbered feature-prefix aliases."""
        aliases: dict[str, tuple[str, ...]] = {}
        for module_type in cls.__registry__.values():
            aliases.update(module_type.numbered_measurement_feature_prefix_aliases)
        return aliases

    @classmethod
    @lru_cache(maxsize=1)
    def scale_qualified_measurement_feature_prefix_declarations(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return module-owned scale-qualified feature prefixes."""
        return tuple(
            dict.fromkeys(
                prefix
                for module_type in cls.__registry__.values()
                for prefix in module_type.scale_qualified_measurement_feature_prefixes
            )
        )

    @classmethod
    def directional_pair_feature_alias_declarations(
        cls,
    ) -> Mapping[str, tuple[str, int]]:
        """Return module-owned directional pair aliases."""
        aliases: dict[str, tuple[str, int]] = {}
        for module_type in cls.__registry__.values():
            aliases.update(module_type.directional_pair_feature_aliases)
        return aliases

    @classmethod
    def pair_correlation_feature_name_declaration(cls) -> str | None:
        """Return the module-owned pair correlation feature name."""
        names = tuple(
            module_type.pair_correlation_feature_name
            for module_type in cls.__registry__.values()
            if module_type.pair_correlation_feature_name is not None
        )
        if len(set(names)) > 1:
            raise ValueError(f"Conflicting pair correlation feature names: {names!r}.")
        return names[0] if names else None

    @classmethod
    def pair_regression_slope_feature_name_declaration(cls) -> str | None:
        """Return the module-owned pair regression-slope feature name."""
        names = tuple(
            module_type.pair_regression_slope_feature_name
            for module_type in cls.__registry__.values()
            if module_type.pair_regression_slope_feature_name is not None
        )
        if len(set(names)) > 1:
            raise ValueError(f"Conflicting pair regression feature names: {names!r}.")
        return names[0] if names else None

    @classmethod
    def undirected_pair_feature_name_declarations(cls) -> frozenset[str]:
        """Return module-owned pair features that have no direction."""
        return frozenset(
            feature_name
            for module_type in cls.__registry__.values()
            for feature_name in module_type.undirected_pair_feature_names
        )

    @classmethod
    def threshold_sensitive_pair_feature_name_declarations(cls) -> frozenset[str]:
        """Return module-owned threshold-sensitive pair feature names."""
        return frozenset(
            feature_name
            for module_type in cls.__registry__.values()
            for feature_name in module_type.threshold_sensitive_pair_feature_names
        )

    @classmethod
    def runtime_object_measurement_row_policy(cls):
        """Return the object-measurement row policy declared by this module."""
        from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
            CellProfilerObjectMeasurementRowPolicy,
        )

        return CellProfilerObjectMeasurementRowPolicy()

    @classmethod
    def relationship_measurement_rows(
        cls,
        request: "CellProfilerOutputRecordRequest",
    ) -> "RelationshipMeasurementRows":
        """Return the relationship-row projector owned by this module declaration."""
        from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
            RelationshipMeasurementRows,
        )

        return RelationshipMeasurementRows(request)

    @classmethod
    def relationship_distance_measurements_apply(
        cls,
        callable_contract: "CallableContract",
        relationship_spec: "ArtifactSpec",
    ) -> bool:
        """Return whether this relationship output owns distance measurement rows."""
        del callable_contract, relationship_spec
        return False
