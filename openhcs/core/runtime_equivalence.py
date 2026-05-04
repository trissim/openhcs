"""Semantic equivalence checks for runtime outputs."""

from __future__ import annotations

import hashlib
import inspect
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Generic, TypeVar

import numpy as np

from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.record_algebra import product_record

import openhcs.core.runtime_artifact_queries as runtime_artifact_queries
import openhcs.core.runtime_semantics as runtime_semantics
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.runtime_artifact_queries import (
    MEASUREMENT_FEATURE_NAME_FIELDS,
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    MEASUREMENT_VALUE_FIELDS,
    measurement_row_mapping,
    measurement_object_label,
    measurement_row_object_name,
    measurement_row_source_image_name,
    measurement_rows,
    measurement_table_object_id_field,
)
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionObservation,
)
from openhcs.core.runtime_exports import RuntimeImageExportSpec
from openhcs.core.runtime_semantics import (
    MeasurementScope,
    MeasurementSubject,
    PairMeasurementFeature,
    dense_object_label_id_domain,
)
from openhcs.core.runtime_values import MeasurementTable
from openhcs.core.runtime_values import ObjectLabelSet
from openhcs.core.runtime_values import ObjectRelationship
from openhcs.core.runtime_values import RuntimeValue
from openhcs.core.runtime_values import SpatialGrid
from openhcs.core.equivalence.policy import (
    DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    RuntimeEquivalencePolicy,
    RuntimeMeasurementDialect,
    RuntimeMeasurementFeatureNameMode,
    RuntimeMeasurementFeatureNumericTolerance,
    RuntimeMeasurementQualifierValueMode,
    RuntimeMeasurementRowQualifier,
    normalize_runtime_identifier as _normalize_identifier,
    normalize_runtime_source_name as _normalize_source_name,
)
from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.cells import (
    RuntimeCellMissingStrategy,
    RuntimeCellSignature,
    RuntimeCellValueKind,
    absolute_numeric_counters_equivalent as _absolute_numeric_counters_equivalent,
    finite_signature_number as _finite_signature_number,
    runtime_cell_signature as _cell_signature,
    runtime_cell_signature_counters_equivalent as _cell_signature_counters_equivalent,
    sparse_absolute_numeric_counters_equivalent as _sparse_absolute_numeric_counters_equivalent,
    sparse_numeric_counters_equivalent as _sparse_numeric_counters_equivalent,
)
from openhcs.core.equivalence.tables import (
    CSV_HEADER_CONTEXT_STOPWORDS as _CSV_HEADER_CONTEXT_STOPWORDS,
    MEASUREMENT_IDENTITY_FIELDS as _MEASUREMENT_IDENTITY_FIELDS,
    RuntimeTableSnapshot,
)
from openhcs.core.equivalence.images import RuntimeImageSnapshot
from openhcs.core.equivalence.outputs import (
    RuntimeOutputSnapshot,
    image_paths,
    table_paths,
)
from openhcs.core.equivalence.report import (
    RuntimeEquivalenceDifference,
    RuntimeEquivalenceDifferenceKind,
    RuntimeEquivalenceReport,
)
from openhcs.core.equivalence.comparison import (
    runtime_image_differences as _image_differences,
    runtime_table_differences as _table_differences,
)


BENCHMARK_CACHE_DOMAINS = frozenset({"parity"})
_RUNTIME_MEASUREMENT_PROJECTION_MODULES = (
    sys.modules[__name__],
    runtime_artifact_queries,
    runtime_semantics,
)
_RUNTIME_AGGREGATE_TABLE_IDENTITY_FIELDS = frozenset(
    {"image_id", "image_number", "slice_index"}
)
_CACHE_MISS = object()


def runtime_measurement_projection_cache_identity() -> tuple[tuple[str, str], ...]:
    """Return the core semantic-projection identity for measurement caches."""
    return tuple(
        (module.__name__, _module_source_digest(module))
        for module in _RUNTIME_MEASUREMENT_PROJECTION_MODULES
    )


def _module_source_digest(module: ModuleType) -> str:
    try:
        source = inspect.getsource(module).encode("utf-8")
    except (OSError, TypeError):
        source = (
            Path(module.__file__).read_bytes()
            if module.__file__ is not None
            else repr(module).encode("utf-8")
        )
    return hashlib.sha256(source).hexdigest()


_RuntimeMeasurementIndexedQualifier = tuple[
    RuntimeMeasurementRowQualifier,
    tuple[int | None, ...],
]
_RuntimeMeasurementFact = tuple[
    RuntimeMeasurementFeatureKey,
    RuntimeCellSignature,
]
_RuntimeMeasurementFacts = tuple[_RuntimeMeasurementFact, ...]
_RuntimeMeasurementFactList = list[_RuntimeMeasurementFact]
_RuntimeMeasurementFactCounters = dict[
    RuntimeMeasurementFeatureKey,
    Counter[RuntimeCellSignature],
]
_RuntimeMeasurementKeySet = frozenset[RuntimeMeasurementFeatureKey]
_RuntimeRequiredMeasurementKeys = _RuntimeMeasurementKeySet | None
_RuntimeObjectValuesByLabel = dict[RuntimeMeasurementFeatureKey, dict[int, float]]
_RuntimeObjectValuesByObject = dict[
    tuple[str, _RuntimeRequiredMeasurementKeys],
    _RuntimeObjectValuesByLabel,
]
_RuntimeLongFormMeasurementFact = _RuntimeMeasurementFact | None
_RuntimeNumericMeasurementValue = tuple[RuntimeMeasurementFeatureKey, float]
_RuntimeNumericMeasurementValues = tuple[_RuntimeNumericMeasurementValue, ...]
_RuntimeMeasurementRowIdentity = tuple[tuple[str, object], ...]
_RuntimeMeasurementRowIdentityOrMissing = _RuntimeMeasurementRowIdentity | None
_RuntimeMeasurementNameParts = tuple[tuple[str, ...], tuple[str, ...]]
_RuntimeSourceTokenGroups = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class _RuntimeMeasurementRowSchema:
    feature_indexes: tuple[int, ...]
    qualifiers_by_index: dict[int, tuple[_RuntimeMeasurementIndexedQualifier, ...]]
    long_form_feature_indexes: tuple[int, ...]
    long_form_value_indexes: tuple[int, ...]


_RuntimeMeasurementQualifierCacheKey = tuple[
    RuntimeMeasurementRowQualifier,
    tuple[str | None, ...],
]
_RuntimeMeasurementRowSubjectSchema = tuple[
    int | None,
    int | None,
    tuple[int, ...],
    tuple[int, ...],
]
_ContextualMeasurementPaddingGroup = tuple[str, tuple[str, ...], str | None]
_RuntimeMeasurementPaddingGroup = tuple[
    RuntimeMeasurementSubjectKey,
    str | None,
    tuple[str, ...],
]
_RuntimeMeasurementRowSchemaCache = dict[tuple[str, ...], _RuntimeMeasurementRowSchema]
_RuntimeMeasurementFeatureKeyCache = dict[
    tuple[RuntimeMeasurementSubjectKey, str | None, str, tuple[str, ...]],
    RuntimeMeasurementFeatureKey | None,
]
_RuntimeMeasurementLongFormKeyCache = dict[
    tuple[RuntimeMeasurementSubjectKey, str | None, str],
    RuntimeMeasurementFeatureKey | None,
]
_RuntimeMeasurementQualifierRenderCache = dict[
    _RuntimeMeasurementQualifierCacheKey,
    str | None,
]
_RuntimeMeasurementPaddingGroupCache = dict[
    tuple[str, RuntimeMeasurementFeatureKey],
    _RuntimeMeasurementPaddingGroup,
]
_RuntimeMeasurementIndexedQualifierCache = dict[
    tuple[_RuntimeMeasurementIndexedQualifier, ...],
    tuple[str, ...],
]
_RuntimeRowQualifierResolutionCache = dict[
    tuple[_RuntimeMeasurementIndexedQualifier, ...],
    tuple[str, ...],
]
_RuntimeMeasurementPaddingGroupPresence = dict[_RuntimeMeasurementPaddingGroup, bool]
_StaticWideRuntimeKeyCache = dict[
    tuple[RuntimeMeasurementSubjectKey, str | None, int, tuple[str, ...]],
    RuntimeMeasurementFeatureKey | None,
]
_StaticWideRuntimeQualifiersByIndex = dict[
    int,
    tuple[_RuntimeMeasurementIndexedQualifier, ...],
]
_AggregateValuesByFeature = dict[
    tuple[RuntimeMeasurementFeatureKey, tuple[tuple[str, object], ...]],
    list[float],
]
_AggregateMeanKeyCache = dict[
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementFeatureKey | None,
]
_RuntimeRowProjectionValueT = TypeVar("_RuntimeRowProjectionValueT")
_RuntimeRowProjectionRecord = tuple[
    _RuntimeMeasurementPaddingGroup,
    RuntimeMeasurementFeatureKey,
    _RuntimeRowProjectionValueT,
]
_RuntimeRowProjectionRecords = tuple[
    _RuntimeRowProjectionRecord[_RuntimeRowProjectionValueT],
    ...,
]
_RuntimeProjectedCell = tuple[
    RuntimeMeasurementFeatureKey,
    _RuntimeRowProjectionValueT,
]
_RuntimeRowValueProjector = Callable[
    [
        RuntimeMeasurementFeatureKey,
        object,
        RuntimeEquivalencePolicy,
    ],
    tuple[_RuntimeProjectedCell[_RuntimeRowProjectionValueT], ...],
]
_RuntimeRowLongFormProjector = Callable[
    [_RuntimeMeasurementFact],
    tuple[_RuntimeProjectedCell[_RuntimeRowProjectionValueT], ...],
]


@dataclass(frozen=True, slots=True)
class _RuntimeRowProjection(Generic[_RuntimeRowProjectionValueT]):
    records: _RuntimeRowProjectionRecords[_RuntimeRowProjectionValueT]
    long_form: bool = False


@dataclass(frozen=True, slots=True)
class _RuntimeRowProjectionContext:
    row: Mapping[str, object]
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    source_name: str | None
    known_source_names: tuple[str, ...]
    required_keys: _RuntimeRequiredMeasurementKeys
    table_padding_group: str
    schema_cache: _RuntimeMeasurementRowSchemaCache
    key_cache: _RuntimeMeasurementFeatureKeyCache
    long_form_key_cache: _RuntimeMeasurementLongFormKeyCache
    qualifier_render_cache: _RuntimeMeasurementQualifierRenderCache
    padding_group_cache: _RuntimeMeasurementPaddingGroupCache

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, object],
        subject: RuntimeMeasurementSubjectKey,
        policy: RuntimeEquivalencePolicy,
        *,
        source_name: str | None,
        known_source_names: tuple[str, ...],
        required_keys: _RuntimeRequiredMeasurementKeys,
        table_padding_group: str,
        schema_cache: _RuntimeMeasurementRowSchemaCache,
        key_cache: _RuntimeMeasurementFeatureKeyCache,
        long_form_key_cache: _RuntimeMeasurementLongFormKeyCache,
        qualifier_render_cache: _RuntimeMeasurementQualifierRenderCache,
        padding_group_cache: _RuntimeMeasurementPaddingGroupCache,
    ) -> "_RuntimeRowProjectionContext":
        return cls(
            row=row,
            subject=subject,
            policy=policy,
            source_name=source_name,
            known_source_names=known_source_names,
            required_keys=required_keys,
            table_padding_group=table_padding_group,
            schema_cache=schema_cache,
            key_cache=key_cache,
            long_form_key_cache=long_form_key_cache,
            qualifier_render_cache=qualifier_render_cache,
            padding_group_cache=padding_group_cache,
        )


@dataclass(frozen=True, slots=True)
class _LongFormMeasurementContext:
    row: Mapping[str, object]
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    source_name: str | None
    known_source_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CachedLongFormMeasurementContext:
    row_values: tuple[object, ...]
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    source_name: str | None
    known_source_names: tuple[str, ...]
    feature_indexes: tuple[int, ...]
    value_indexes: tuple[int, ...]
    key_cache: _RuntimeMeasurementLongFormKeyCache

    @classmethod
    def from_runtime_row_projection(
        cls,
        context: _RuntimeRowProjectionContext,
        row_values: tuple[object, ...],
        feature_indexes: tuple[int, ...],
        value_indexes: tuple[int, ...],
    ) -> "_CachedLongFormMeasurementContext":
        return cls(
            row_values,
            context.subject,
            context.policy,
            context.source_name,
            context.known_source_names,
            feature_indexes,
            value_indexes,
            context.long_form_key_cache,
        )


@dataclass(frozen=True, slots=True)
class _AggregateInputRecordingContext:
    values_by_feature: _AggregateValuesByFeature
    row_mapping: Mapping[str, object]
    axis_key: object | None
    required_keys: _RuntimeRequiredMeasurementKeys
    key_cache: _AggregateMeanKeyCache


@dataclass(frozen=True, slots=True)
class _RuntimeMeasurementFactRecordingContext:
    values_by_feature: _RuntimeMeasurementFactCounters
    explicit_measurement_keys: set[RuntimeMeasurementFeatureKey]
    required_keys: _RuntimeRequiredMeasurementKeys


@dataclass(frozen=True, slots=True)
class _StaticWideRuntimeRowProjectionContext:
    header: tuple[str, ...]
    policy: RuntimeEquivalencePolicy
    known_source_names: tuple[str, ...]
    input_keys: _RuntimeRequiredMeasurementKeys
    feature_column_indexes: tuple[int, ...]
    aggregate_reference_indexes: frozenset[int]
    qualifiers_by_index: _StaticWideRuntimeQualifiersByIndex
    qualifier_render_cache: _RuntimeMeasurementQualifierRenderCache
    key_cache: _StaticWideRuntimeKeyCache
    padding_group_cache: _RuntimeMeasurementPaddingGroupCache
    table_padding_group: str


@dataclass(frozen=True, slots=True)
class _MeasurementFeatureKeySourceContext:
    field_name: str
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    qualifiers: tuple[str, ...]
    source_name: str | None
    known_source_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RuntimeMeasurementTableProjectionContext:
    table: MeasurementTable
    policy: RuntimeEquivalencePolicy
    axis_key: object | None
    known_source_names: tuple[str, ...]
    required_keys: _RuntimeRequiredMeasurementKeys


@dataclass(frozen=True, slots=True)
class _ObjectLabelMeasurementContext:
    labels: object
    object_name: str | None
    policy: RuntimeEquivalencePolicy
    object_identifier_subjects: frozenset[RuntimeMeasurementSubjectKey]
    object_location_subjects: frozenset[RuntimeMeasurementSubjectKey]
    required_keys: _RuntimeRequiredMeasurementKeys
    declared_object_count: int | None
    declared_object_ids: tuple[int, ...]

    @classmethod
    def from_runtime_value(
        cls,
        value: RuntimeValue,
        policy: RuntimeEquivalencePolicy,
        object_identifier_subjects: frozenset[RuntimeMeasurementSubjectKey],
        object_location_subjects: frozenset[RuntimeMeasurementSubjectKey],
        required_keys: _RuntimeRequiredMeasurementKeys,
    ) -> "_ObjectLabelMeasurementContext":
        label_set = ObjectLabelSet.from_runtime_value(value)
        return cls(
            label_set.labels,
            value.schema.object_name,
            policy,
            object_identifier_subjects,
            object_location_subjects,
            required_keys,
            label_set.declared_object_count,
            label_set.declared_object_ids,
        )


_MeasurementScopeAggregatePolicy = product_record(
    "_MeasurementScopeAggregatePolicy",
    "scope: MeasurementScope; accepts_aggregate_feature_key: bool",
    doc="Aggregate-feature parsing policy for one closed measurement scope.",
)


def _measurement_scope_aggregate_policy_by_scope(
    rows: tuple[_MeasurementScopeAggregatePolicy, ...],
) -> Mapping[MeasurementScope, _MeasurementScopeAggregatePolicy]:
    by_scope = {row.scope: row for row in rows}
    if set(by_scope) != set(MeasurementScope):
        missing = sorted_tuple(scope.value for scope in set(MeasurementScope) - set(by_scope))
        extra = sorted_tuple(scope.value for scope in set(by_scope) - set(MeasurementScope))
        raise ValueError(
            "Measurement scope aggregate policy must cover MeasurementScope "
            f"exactly: missing={missing!r}, extra={extra!r}."
        )
    return MappingProxyType(by_scope)


_MEASUREMENT_SCOPE_AGGREGATE_POLICY_BY_SCOPE = (
    _measurement_scope_aggregate_policy_by_scope(
        (
            _MeasurementScopeAggregatePolicy(MeasurementScope.ARTIFACT, False),
            _MeasurementScopeAggregatePolicy(MeasurementScope.IMAGE, True),
            _MeasurementScopeAggregatePolicy(MeasurementScope.OBJECT, False),
            _MeasurementScopeAggregatePolicy(MeasurementScope.RELATIONSHIP, False),
            _MeasurementScopeAggregatePolicy(MeasurementScope.EXPERIMENT, True),
        )
    )
)
_MeasurementQualifierValueRenderer = Callable[[tuple[object, ...]], str | None]


def _two_digit_integer_measurement_qualifier_value(
    values: tuple[object, ...],
) -> str | None:
    return str(_measurement_qualifier_integer(values[0])).zfill(2)


def _fraction_of_count_measurement_qualifier_value(
    values: tuple[object, ...],
) -> str | None:
    if len(values) != 2:
        return None
    return (
        f"{_measurement_qualifier_integer(values[0])}"
        f"of{_measurement_qualifier_integer(values[1])}"
    )


def _identifier_measurement_qualifier_value(
    values: tuple[object, ...],
) -> str | None:
    return "_".join(_measurement_qualifier_identifier(value) for value in values)


def _measurement_qualifier_value_renderers(
    renderers: Mapping[
        RuntimeMeasurementQualifierValueMode,
        _MeasurementQualifierValueRenderer,
    ],
) -> Mapping[
    RuntimeMeasurementQualifierValueMode,
    _MeasurementQualifierValueRenderer,
]:
    renderer_modes = set(renderers)
    value_modes = set(RuntimeMeasurementQualifierValueMode)
    if renderer_modes != value_modes:
        missing = sorted_tuple(mode.value for mode in value_modes - renderer_modes)
        extra = sorted_tuple(mode.value for mode in renderer_modes - value_modes)
        raise ValueError(
            "Measurement qualifier value renderers must cover "
            f"RuntimeMeasurementQualifierValueMode exactly: "
            f"missing={missing!r}, extra={extra!r}."
        )
    return MappingProxyType(dict(renderers))


_MEASUREMENT_QUALIFIER_VALUE_RENDERERS = _measurement_qualifier_value_renderers(
    {
        RuntimeMeasurementQualifierValueMode.IDENTIFIER: _identifier_measurement_qualifier_value,
        RuntimeMeasurementQualifierValueMode.TWO_DIGIT_INTEGER: _two_digit_integer_measurement_qualifier_value,
        RuntimeMeasurementQualifierValueMode.FRACTION_OF_COUNT: _fraction_of_count_measurement_qualifier_value,
    }
)


def _aggregate_mean_key(
    key: RuntimeMeasurementFeatureKey,
    *,
    required_keys: _RuntimeRequiredMeasurementKeys,
    key_cache: _AggregateMeanKeyCache,
) -> RuntimeMeasurementFeatureKey | None:
    mean_key = key_cache.get(key, _CACHE_MISS)
    if mean_key is _CACHE_MISS:
        mean_key = _runtime_measurement_feature_key(
            key.subject,
            key.feature_name,
            "mean",
            source_name=key.source_name,
        )
        if required_keys is not None and mean_key not in required_keys:
            mean_key = None
        elif _is_image_number_reference_feature(key):
            mean_key = None
        key_cache[key] = mean_key
    return mean_key


def _finite_numeric_runtime_cell_value(value: RuntimeCellSignature) -> float | None:
    if value.kind is not RuntimeCellValueKind.NUMBER:
        return None
    numeric_value = float(value.value)
    return numeric_value if math.isfinite(numeric_value) else None


_MEASUREMENT_DIALECT_QUALIFIER_FIELD_NAMES_CACHE: dict[
    int,
    tuple[RuntimeMeasurementDialect, frozenset[str]],
] = {}


_TIE_SENSITIVE_LOCATION_FEATURES = frozenset(
    ("max_intensity_x", "max_intensity_y", "max_intensity_z")
)
_OBJECT_LOCATION_FEATURES = frozenset(("center_x", "center_y", "center_z"))
_LOCATION_VALUE_FEATURE_BY_NAME = MappingProxyType(
    {
        "max_intensity_x": "max_intensity",
        "max_intensity_y": "max_intensity",
        "max_intensity_z": "max_intensity",
    }
)
_ORIENTATION_FEATURES = frozenset(("orientation",))
_SHAPE_DESCRIPTOR_GATING_FEATURES = (
    "area",
    "center_x",
    "center_y",
    "center_z",
    "maximum_radius",
    "mean_radius",
    "median_radius",
    "major_axis_length",
    "minor_axis_length",
    "compactness",
    "form_factor",
    "min_feret_diameter",
    "max_feret_diameter",
)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementSnapshot:
    """Semantic measurement facts independent of table layout."""

    values_by_feature: Mapping[RuntimeMeasurementFeatureKey, Counter[RuntimeCellSignature]]

    @classmethod
    def from_output_snapshot(
        cls,
        snapshot: "RuntimeOutputSnapshot",
        *,
        policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
        known_source_names: tuple[str, ...] = (),
    ) -> "RuntimeMeasurementSnapshot":
        """Project exported tables into semantic measurement facts."""
        values_by_feature: _RuntimeMeasurementFactCounters = {}
        for table in snapshot.tables:
            if _record_static_wide_measurement_table_snapshot(
                values_by_feature,
                table,
                policy,
                known_source_names=known_source_names,
            ):
                continue
            _record_measurement_facts(
                values_by_feature,
                _measurement_facts_from_table_snapshot(
                    table,
                    policy,
                    known_source_names=known_source_names,
                ),
            )
        return cls(values_by_feature=values_by_feature)

    @classmethod
    def from_artifact_execution_observation(
        cls,
        observation: RuntimeArtifactExecutionObservation,
        *,
        policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
        known_source_names: tuple[str, ...] = (),
        required_measurement_keys: _RuntimeRequiredMeasurementKeys = None,
    ) -> "RuntimeMeasurementSnapshot":
        """Project typed runtime measurement artifacts into semantic facts."""
        values_by_feature: _RuntimeMeasurementFactCounters = {}
        object_label_records = []
        object_label_records_by_axis: dict[object, list[object]] = {}
        relationship_records_by_axis: dict[object, list[object]] = {}
        measurement_tables_by_axis: dict[object, list[MeasurementTable]] = {}
        for axis_key, records in observation.records_by_axis.items():
            seen_aggregate_measurement_tables: set[tuple[object, ...]] = set()
            for record in records:
                if record.key.kind is ArtifactKind.SPATIAL_GRID:
                    _record_measurement_facts(
                        values_by_feature,
                        _spatial_grid_measurement_facts(record.value, policy),
                        required_keys=required_measurement_keys,
                    )
                    continue
                if record.key.kind is ArtifactKind.MEASUREMENTS:
                    table = MeasurementTable.from_runtime_value(record.value)
                    aggregate_table_key = _aggregate_measurement_table_key(table)
                    if aggregate_table_key is not None:
                        if aggregate_table_key in seen_aggregate_measurement_tables:
                            continue
                        seen_aggregate_measurement_tables.add(aggregate_table_key)
                    measurement_tables_by_axis.setdefault(axis_key, []).append(table)
                    table_projection_context = _RuntimeMeasurementTableProjectionContext(
                        table,
                        policy,
                        axis_key,
                        known_source_names,
                        required_measurement_keys,
                    )
                    if _record_static_wide_runtime_measurement_table(
                        values_by_feature,
                        table_projection_context,
                    ):
                        continue
                    _record_measurement_facts(
                        values_by_feature,
                        _measurement_facts_from_runtime_table(
                            table_projection_context,
                        ),
                        required_keys=required_measurement_keys,
                    )
                    continue
                if record.key.kind is ArtifactKind.OBJECT_LABELS:
                    object_label_records.append(record)
                    object_label_records_by_axis.setdefault(axis_key, []).append(record)
                    continue
                if record.key.kind is ArtifactKind.RELATIONSHIPS:
                    relationship_records_by_axis.setdefault(axis_key, []).append(record)
        object_identifier_subjects = _object_identifier_subjects(values_by_feature)
        object_location_subjects = _object_location_subjects(values_by_feature)
        for record in object_label_records:
            _record_measurement_facts(
                values_by_feature,
                _object_label_measurement_facts(
                    _ObjectLabelMeasurementContext.from_runtime_value(
                        record.value,
                        policy,
                        object_identifier_subjects,
                        object_location_subjects,
                        required_measurement_keys,
                    ),
                ),
                required_keys=required_measurement_keys,
            )
        explicit_measurement_keys = frozenset(values_by_feature)
        for axis_key, relationship_records in relationship_records_by_axis.items():
            object_label_counts = _object_label_counts_by_name(
                object_label_records_by_axis.get(axis_key, ())
            )
            axis_object_label_records = tuple(
                object_label_records_by_axis.get(axis_key, ())
            )
            measurement_tables = tuple(measurement_tables_by_axis.get(axis_key, ()))
            child_measurement_values_by_object: _RuntimeObjectValuesByObject = {}
            object_label_values_by_object: _RuntimeObjectValuesByObject = {}

            def object_label_measurement_values(
                object_name: str,
                *,
                required_child_keys: _RuntimeRequiredMeasurementKeys = None,
            ) -> _RuntimeObjectValuesByLabel:
                normalized_object_name = _normalize_identifier(object_name)
                cache_key = (normalized_object_name, required_child_keys)
                cached = object_label_values_by_object.get(cache_key)
                if cached is not None:
                    return cached
                values = _object_label_measurement_values_for_name(
                    axis_object_label_records,
                    object_name,
                    required_keys=required_child_keys,
                )
                object_label_values_by_object[cache_key] = values
                return values

            def child_measurement_values(
                object_name: str,
                *,
                required_child_keys: _RuntimeRequiredMeasurementKeys = None,
            ) -> _RuntimeObjectValuesByLabel:
                normalized_object_name = _normalize_identifier(object_name)
                cache_key = (normalized_object_name, required_child_keys)
                cached = child_measurement_values_by_object.get(cache_key)
                if cached is not None:
                    return cached

                values = _object_measurement_values_by_label(
                    measurement_tables,
                    object_name,
                    policy,
                    known_source_names=known_source_names,
                    required_keys=required_child_keys,
                )
                for key, values_by_child_id in object_label_measurement_values(
                    normalized_object_name,
                    required_child_keys=required_child_keys,
                ).items():
                    if required_child_keys is not None and key not in required_child_keys:
                        continue
                    values.setdefault(key, {}).update(values_by_child_id)
                child_measurement_values_by_object[cache_key] = values
                return values

            for record in relationship_records:
                relationship = ObjectRelationship.from_runtime_value(record.value)
                relationship_facts = _relationship_measurement_facts(
                    relationship,
                    policy,
                    object_label_counts=object_label_counts,
                )
                _record_measurement_facts(
                    values_by_feature,
                    (
                        (key, value)
                        for key, value in relationship_facts
                        if key not in explicit_measurement_keys
                    ),
                    required_keys=required_measurement_keys,
                )
                required_child_keys = _relationship_required_child_measurement_keys(
                    relationship,
                    required_measurement_keys,
                )
                if required_measurement_keys is not None and not required_child_keys:
                    continue
                aggregate_facts = _relationship_aggregate_measurement_facts(
                    relationship,
                    child_measurement_values(
                        relationship.target.name,
                        required_child_keys=required_child_keys,
                    ),
                    policy,
                    object_label_counts=object_label_counts,
                    existing_measurement_keys=explicit_measurement_keys,
                    required_measurement_keys=required_measurement_keys,
                )
                _record_measurement_facts(
                    values_by_feature,
                    (
                        (key, value)
                        for key, value in aggregate_facts
                        if key not in explicit_measurement_keys
                    ),
                    required_keys=required_measurement_keys,
                )
        return cls(values_by_feature=values_by_feature)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values_by_feature",
            MappingProxyType(
                {
                    key: Counter(values)
                    for key, values in self.values_by_feature.items()
                }
            ),
        )

    @property
    def is_empty(self) -> bool:
        return not self.values_by_feature

    def to_cache_payload(
        self,
    ) -> tuple[
        tuple[
            tuple[tuple[str, str | None], str, str, str | None],
            tuple[tuple[tuple[str, str], int], ...],
        ],
        ...,
    ]:
        """Return a stable semantic cache payload for repeated equivalence checks."""
        return tuple(
            (
                feature.to_cache_payload(),
                tuple(
                    (value.to_cache_payload(), int(count))
                    for value, count in sorted(
                        values.items(),
                        key=lambda item: item[0].sort_key,
                    )
                ),
            )
            for feature, values in sorted(
                self.values_by_feature.items(),
                key=lambda item: item[0].sort_key,
            )
        )

    @classmethod
    def from_cache_payload(
        cls,
        payload: object,
    ) -> "RuntimeMeasurementSnapshot":
        """Rebuild a semantic measurement snapshot from cache payload data."""
        values_by_feature: _RuntimeMeasurementFactCounters = {}
        for feature_payload, values_payload in payload:  # type: ignore[union-attr]
            counter: Counter[RuntimeCellSignature] = Counter()
            for value_payload, count in values_payload:
                counter[
                    RuntimeCellSignature.from_cache_payload(value_payload)
                ] = int(count)
            values_by_feature[
                RuntimeMeasurementFeatureKey.from_cache_payload(feature_payload)
            ] = counter
        return cls(values_by_feature=values_by_feature)


def _aggregate_measurement_table_key(table: MeasurementTable) -> tuple[object, ...] | None:
    """Return a semantic key for duplicate full-axis measurement tables.

    Grouped execution can materialize an already-aggregated measurement table
    once per group key. Row-local image identity fields carry the actual
    measurement scope, so exact duplicate full-axis tables should only
    contribute once. Group-local tables remain count-preserving.
    """
    rows = tuple(measurement_rows((table,)))
    if not rows:
        return None

    row_mappings = tuple(measurement_row_mapping(row) for row in rows)
    normalized_field_cache: dict[str, str] = {}

    def normalized_field(field_name: str) -> str:
        cached = normalized_field_cache.get(field_name)
        if cached is None:
            cached = _normalize_identifier(field_name)
            normalized_field_cache[field_name] = cached
        return cached

    row_identity_values: set[tuple[str, object]] = set()
    for row_mapping in row_mappings:
        for field_name, value in row_mapping.items():
            normalized_field_name = normalized_field(str(field_name))
            if normalized_field_name in _RUNTIME_AGGREGATE_TABLE_IDENTITY_FIELDS:
                row_identity_values.add(
                    (
                        normalized_field_name,
                        _measurement_table_cell_payload(value),
                    )
                )
        if len(row_identity_values) > 1:
            break

    if len(row_identity_values) <= 1:
        return None

    row_payloads: list[tuple[tuple[str, object], ...]] = []
    for row_mapping in row_mappings:
        row_payloads.append(
            tuple(
                (
                    normalized_field(str(field_name)),
                    _measurement_table_cell_payload(value),
                )
                for field_name, value in row_mapping.items()
            )
        )

    field_payloads = tuple(
        (field.name, field.dtype, field.required)
        for field in table.fields
    )
    return (
        table.name,
        repr(table.subject),
        table.object_name,
        table.object_id_field,
        table.source_image_name,
        field_payloads,
        tuple(row_payloads),
    )


def _measurement_table_cell_payload(value: object) -> object:
    """Return a hashable exact payload for measurement-table dedupe."""
    if isinstance(value, np.generic):
        return _measurement_table_cell_payload(value.item())
    if value is None:
        return None
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float) and math.isnan(value):
        return ("float", "nan")
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (
                    _measurement_table_cell_payload(key),
                    _measurement_table_cell_payload(nested_value),
                )
                for key, nested_value in value.items()
            ),
        )
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            tuple(_measurement_table_cell_payload(item) for item in value),
        )
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            str(value.dtype),
            tuple(value.shape),
            hashlib.sha256(value.tobytes()).hexdigest(),
        )
    return (type(value).__name__, repr(value))


def runtime_output_equivalence(
    reference: RuntimeOutputSnapshot,
    candidate: RuntimeOutputSnapshot,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare two runtime output snapshots for semantic equivalence."""
    return RuntimeEquivalenceReport(
        differences=(
            *_table_differences(reference.tables, candidate.tables, policy),
            *_image_differences(reference.images, candidate.images, policy),
        )
    )


def runtime_output_root_equivalence(
    reference_output_root: Path,
    candidate_output_root: Path,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare two runtime output directories for semantic equivalence."""
    return runtime_output_equivalence(
        RuntimeOutputSnapshot.from_output_root(reference_output_root),
        RuntimeOutputSnapshot.from_output_root(candidate_output_root),
        policy=policy,
    )


def runtime_measurement_equivalence(
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare precomputed semantic measurement snapshots."""
    return RuntimeEquivalenceReport(
        differences=_measurement_differences(reference, candidate, policy)
    )


def runtime_artifact_measurement_source_names(
    observation: RuntimeArtifactExecutionObservation,
) -> tuple[str, ...]:
    """Return source-image names carried by typed runtime artifacts."""
    return _measurement_source_names_from_artifact_execution(observation)


def runtime_reference_artifact_equivalence(
    reference: RuntimeOutputSnapshot,
    candidate: RuntimeArtifactExecutionObservation,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
    candidate_image_artifact_names: frozenset[str] = frozenset(),
    candidate_image_export_specs: tuple[RuntimeImageExportSpec, ...] = (),
    candidate_image_snapshots: tuple[RuntimeImageSnapshot, ...] | None = None,
) -> RuntimeEquivalenceReport:
    """Compare an external output reference to typed runtime artifact execution."""
    known_source_names = runtime_artifact_measurement_source_names(candidate)
    reference_measurements = RuntimeMeasurementSnapshot.from_output_snapshot(
        reference,
        policy=policy,
        known_source_names=known_source_names,
    )
    candidate_measurements = RuntimeMeasurementSnapshot.from_artifact_execution_observation(
        candidate,
        policy=policy,
        known_source_names=known_source_names,
        required_measurement_keys=frozenset(reference_measurements.values_by_feature),
    )
    if reference_measurements.is_empty and candidate_measurements.is_empty:
        table_differences = (
            ()
            if not reference.tables
            else _table_differences(
                reference.tables,
                RuntimeOutputSnapshot.from_artifact_execution_observation(
                    candidate
                ).tables,
                policy,
            )
        )
    else:
        table_differences = runtime_measurement_equivalence(
            reference_measurements,
            candidate_measurements,
            policy=policy,
        ).differences

    image_differences = ()
    if reference.images:
        candidate_images = (
            candidate_image_snapshots
            if candidate_image_snapshots is not None
            else RuntimeOutputSnapshot.from_artifact_execution_observation(
                candidate,
                image_artifact_names=candidate_image_artifact_names,
                image_export_specs=candidate_image_export_specs,
            ).images
        )
        image_differences = _image_differences(
            reference.images,
            candidate_images,
            policy,
        )

    return RuntimeEquivalenceReport(
        differences=(
            *table_differences,
            *image_differences,
        )
    )


def runtime_artifact_execution_equivalence(
    reference: RuntimeArtifactExecutionObservation,
    candidate: RuntimeArtifactExecutionObservation,
    *,
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy(),
) -> RuntimeEquivalenceReport:
    """Compare runtime artifact state and file outputs for semantic equivalence."""
    return RuntimeEquivalenceReport(
        differences=(
            *_runtime_artifact_count_differences(reference, candidate),
            *runtime_output_equivalence(
                RuntimeOutputSnapshot.from_artifact_execution_observation(reference),
                RuntimeOutputSnapshot.from_artifact_execution_observation(candidate),
                policy=policy,
            ).differences,
        )
    )


def _runtime_artifact_count_differences(
    reference: RuntimeArtifactExecutionObservation,
    candidate: RuntimeArtifactExecutionObservation,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    reference_counts = _total_record_counts(reference)
    candidate_counts = _total_record_counts(candidate)
    if reference_counts == candidate_counts:
        return ()
    return (
        RuntimeEquivalenceDifference(
            RuntimeEquivalenceDifferenceKind.RUNTIME_ARTIFACT_COUNTS,
            "runtime artifact counts differ: "
            f"reference={dict(reference_counts)!r}, "
            f"candidate={dict(candidate_counts)!r}",
        ),
    )


def _total_record_counts(
    observation: RuntimeArtifactExecutionObservation,
) -> Counter[ArtifactKind]:
    counts: Counter[ArtifactKind] = Counter()
    for axis_counts in observation.record_counts_by_axis.values():
        counts.update(axis_counts)
    return counts


def _measurement_differences(
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> tuple[RuntimeEquivalenceDifference, ...]:
    differences: list[RuntimeEquivalenceDifference] = []
    reference_features = set(reference.values_by_feature)
    candidate_features = set(candidate.values_by_feature)
    for feature in sorted(
        reference_features - candidate_features,
        key=lambda key: key.sort_key,
    ):
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.MEASUREMENT_FEATURE,
                f"candidate is missing measurement feature {_feature_label(feature)}",
            )
        )
    if not policy.allow_extra_candidate_measurements:
        for feature in sorted(
            candidate_features - reference_features,
            key=lambda key: key.sort_key,
        ):
            differences.append(
                RuntimeEquivalenceDifference(
                    RuntimeEquivalenceDifferenceKind.MEASUREMENT_FEATURE,
                    f"candidate has extra measurement feature {_feature_label(feature)}",
                )
            )
    for feature in sorted(
        reference_features & candidate_features,
        key=lambda key: key.sort_key,
    ):
        if _measurement_feature_values_equivalent(
            feature,
            reference,
            candidate,
            policy,
        ):
            continue
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.MEASUREMENT_CONTENT,
                f"measurement feature {_feature_label(feature)} values differ",
            )
        )
    return tuple(differences)


def _measurement_feature_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    reference_values = reference.values_by_feature[feature]
    candidate_values = candidate.values_by_feature[feature]
    if _cell_signature_counters_equivalent(reference_values, candidate_values, policy):
        return True
    if _tie_sensitive_location_values_equivalent(
        feature,
        reference,
        candidate,
        policy,
    ):
        return True
    if _threshold_entropy_values_equivalent(feature, reference, candidate, policy):
        return True
    if _sparse_object_boundary_values_equivalent(
        feature,
        reference,
        candidate,
        policy,
    ):
        return True
    if _threshold_sensitive_pair_values_equivalent(
        feature,
        reference,
        candidate,
        policy,
    ):
        return True
    if _feature_numeric_tolerance_values_equivalent(
        feature,
        reference,
        candidate,
        policy,
    ):
        return True
    return _unstable_shape_descriptor_values_equivalent(
        feature,
        reference,
        candidate,
        policy,
    )


def _feature_numeric_tolerance_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    for tolerance in policy.feature_numeric_tolerances:
        if not _feature_numeric_tolerance_matches(feature, tolerance):
            continue
        if (
            tolerance.require_object_count_stability
            and not _object_count_values_stable(feature, reference, candidate, policy)
        ):
            continue
        feature_policy = RuntimeEquivalencePolicy(
            numeric_decimal_places=policy.numeric_decimal_places,
            numeric_abs_tolerance=tolerance.numeric_abs_tolerance,
            numeric_rel_tolerance=tolerance.numeric_rel_tolerance,
            measurement_feature_name_mode=policy.measurement_feature_name_mode,
            measurement_dialect=policy.measurement_dialect,
        )
        if _cell_signature_counters_equivalent(
            reference.values_by_feature[feature],
            candidate.values_by_feature[feature],
            feature_policy,
        ):
            return True
    return False


def _feature_numeric_tolerance_matches(
    feature: RuntimeMeasurementFeatureKey,
    tolerance: RuntimeMeasurementFeatureNumericTolerance,
) -> bool:
    if (
        tolerance.subject_scope is not None
        and feature.subject.scope is not tolerance.subject_scope
    ):
        return False
    if tolerance.statistic is not None and feature.statistic != tolerance.statistic:
        return False
    if feature.feature_name in tolerance.feature_names:
        return True
    return any(
        feature.feature_name.startswith(prefix)
        for prefix in tolerance.feature_name_prefixes
    )


def _tie_sensitive_location_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if not policy.allow_tie_sensitive_location_mismatches:
        return False
    if feature.feature_name not in _TIE_SENSITIVE_LOCATION_FEATURES:
        return False
    value_feature = RuntimeMeasurementFeatureKey(
        subject=feature.subject,
        feature_name=_LOCATION_VALUE_FEATURE_BY_NAME[feature.feature_name],
        statistic=feature.statistic,
        source_name=feature.source_name,
    )
    reference_values = reference.values_by_feature.get(value_feature)
    candidate_values = candidate.values_by_feature.get(value_feature)
    if reference_values is None or candidate_values is None:
        return False
    if _cell_signature_counters_equivalent(
        reference_values,
        candidate_values,
        policy,
    ):
        return True
    return _sparse_object_boundary_values_equivalent(
        value_feature,
        reference,
        candidate,
        policy,
    )


def _threshold_entropy_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if policy.threshold_entropy_abs_tolerance == 0:
        return False
    if not feature.feature_name.startswith("sum_of_entropies"):
        return False

    entropy_policy = RuntimeEquivalencePolicy(
        numeric_decimal_places=policy.numeric_decimal_places,
        numeric_abs_tolerance=policy.threshold_entropy_abs_tolerance,
        numeric_rel_tolerance=policy.numeric_rel_tolerance,
        measurement_feature_name_mode=policy.measurement_feature_name_mode,
    )
    return _cell_signature_counters_equivalent(
        reference.values_by_feature[feature],
        candidate.values_by_feature[feature],
        entropy_policy,
    )


def _threshold_sensitive_pair_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if (
        policy.threshold_sensitive_pair_abs_tolerance == 0
        and policy.threshold_sensitive_pair_rel_tolerance == 0
    ):
        return False
    if feature.feature_name not in _THRESHOLD_SENSITIVE_PAIR_FEATURES:
        return False

    pair_policy = RuntimeEquivalencePolicy(
        numeric_decimal_places=policy.numeric_decimal_places,
        numeric_abs_tolerance=policy.threshold_sensitive_pair_abs_tolerance,
        numeric_rel_tolerance=policy.threshold_sensitive_pair_rel_tolerance,
        measurement_feature_name_mode=policy.measurement_feature_name_mode,
    )
    if not _cell_signature_counters_equivalent(
        reference.values_by_feature[feature],
        candidate.values_by_feature[feature],
        pair_policy,
    ):
        return False

    companion_features = _threshold_sensitive_pair_companion_features(
        feature,
        reference,
        candidate,
    )
    if not companion_features:
        return False
    return any(
        _cell_signature_counters_equivalent(
            reference.values_by_feature[companion],
            candidate.values_by_feature[companion],
            pair_policy,
        )
        for companion in companion_features
    )


def _threshold_sensitive_pair_companion_features(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
) -> tuple[RuntimeMeasurementFeatureKey, ...]:
    source_tokens = _pair_feature_source_tokens(feature)
    if source_tokens is None:
        return ()

    comparable_features = set(reference.values_by_feature) & set(
        candidate.values_by_feature
    )
    return sorted_tuple(
        (
            other
            for other in comparable_features
            if _is_pair_companion_feature(
                feature,
                other,
                source_tokens=source_tokens,
            )
        ),
        key=lambda key: key.sort_key,
    )


def _is_pair_companion_feature(
    feature: RuntimeMeasurementFeatureKey,
    other: RuntimeMeasurementFeatureKey,
    *,
    source_tokens: Counter[str],
) -> bool:
    if other == feature:
        return False
    if other.feature_name != feature.feature_name:
        return False
    if other.statistic != feature.statistic:
        return False
    if other.subject.scope is not feature.subject.scope:
        return False
    if (feature.source_name is not None or other.source_name is not None) and (
        other.subject != feature.subject
    ):
        return False
    return _pair_feature_source_tokens(other) == source_tokens


def _pair_feature_source_tokens(
    feature: RuntimeMeasurementFeatureKey,
) -> Counter[str] | None:
    source_name = feature.source_name or feature.subject.name
    if source_name is None:
        return None
    tokens = _source_name_tokens(
        _normalize_source_name(source_name) or _normalize_identifier(source_name)
    )
    if len(tokens) < 2:
        return None
    return Counter(tokens)


def _sparse_object_boundary_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if not policy.allow_sparse_object_boundary_jitter:
        return False
    if feature.subject.scope is not MeasurementScope.OBJECT:
        return False
    equivalence_rule = _SPARSE_OBJECT_BOUNDARY_EQUIVALENCE_BY_STATISTIC.get(
        feature.statistic
    )
    if equivalence_rule is None:
        return False
    return equivalence_rule(feature, reference, candidate, policy)


def _sparse_object_boundary_value_feature_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if not _object_count_values_stable(feature, reference, candidate, policy):
        return False
    if feature.feature_name in _ORIENTATION_FEATURES:
        return _object_boundary_jitter_sparse_absolute_numeric_counters_equivalent(
            reference.values_by_feature[feature],
            candidate.values_by_feature[feature],
            policy,
        )
    if _is_object_identifier_feature(feature.feature_name):
        return _sparse_object_identifier_counters_equivalent(
            reference.values_by_feature[feature],
            candidate.values_by_feature[feature],
            policy,
        )
    if _numeric_counters_are_binary(
        reference.values_by_feature[feature],
        candidate.values_by_feature[feature],
    ):
        return _sparse_numeric_counters_equivalent(
            reference.values_by_feature[feature],
            candidate.values_by_feature[feature],
            policy,
            abs_tolerance=policy.numeric_abs_tolerance,
            rel_tolerance=policy.numeric_rel_tolerance,
            max_unstable_values=policy.object_boundary_jitter_max_unstable_values,
            max_unstable_fraction=policy.object_boundary_jitter_max_unstable_fraction,
        )
    return _sparse_numeric_counters_equivalent(
        reference.values_by_feature[feature],
        candidate.values_by_feature[feature],
        policy,
        abs_tolerance=policy.object_boundary_jitter_abs_tolerance,
        rel_tolerance=policy.object_boundary_jitter_rel_tolerance,
        max_unstable_values=policy.object_boundary_jitter_max_unstable_values,
        max_unstable_fraction=policy.object_boundary_jitter_max_unstable_fraction,
    )


def _sparse_object_boundary_mean_feature_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    value_feature = RuntimeMeasurementFeatureKey(
        subject=feature.subject,
        feature_name=feature.feature_name,
        statistic="value",
        source_name=feature.source_name,
    )
    if value_feature not in reference.values_by_feature:
        return False
    if value_feature not in candidate.values_by_feature:
        return False
    if not _sparse_object_boundary_value_feature_equivalent(
        value_feature,
        reference,
        candidate,
        policy,
    ):
        return False

    mean_policy = RuntimeEquivalencePolicy(
        numeric_decimal_places=policy.numeric_decimal_places,
        numeric_abs_tolerance=policy.object_boundary_jitter_aggregate_abs_tolerance,
        numeric_rel_tolerance=policy.object_boundary_jitter_aggregate_rel_tolerance,
        measurement_feature_name_mode=policy.measurement_feature_name_mode,
    )
    return _cell_signature_counters_equivalent(
        reference.values_by_feature[feature],
        candidate.values_by_feature[feature],
        mean_policy,
    )


def _object_boundary_jitter_sparse_absolute_numeric_counters_equivalent(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    return _sparse_absolute_numeric_counters_equivalent(
        reference,
        candidate,
        policy,
        abs_tolerance=policy.object_boundary_jitter_abs_tolerance,
        rel_tolerance=policy.object_boundary_jitter_rel_tolerance,
        max_unstable_values=policy.object_boundary_jitter_max_unstable_values,
        max_unstable_fraction=policy.object_boundary_jitter_max_unstable_fraction,
    )


_SPARSE_OBJECT_BOUNDARY_EQUIVALENCE_BY_STATISTIC = MappingProxyType(
    {
        "count": lambda feature, reference, candidate, policy: (
            feature.feature_name == "object_count"
            and _object_count_counters_sparse_equivalent(
                reference.values_by_feature[feature],
                candidate.values_by_feature[feature],
                policy,
            )
        ),
        "value": _sparse_object_boundary_value_feature_equivalent,
        "mean": _sparse_object_boundary_mean_feature_equivalent,
    }
)


def _object_count_values_stable(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    count_feature = RuntimeMeasurementFeatureKey(
        subject=feature.subject,
        feature_name="object_count",
        statistic="count",
    )
    reference_counts = reference.values_by_feature.get(count_feature)
    candidate_counts = candidate.values_by_feature.get(count_feature)
    if reference_counts is None or candidate_counts is None:
        return False
    return _cell_signature_counters_equivalent(
        reference_counts,
        candidate_counts,
        policy,
    )


def _object_count_counters_sparse_equivalent(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if reference == candidate:
        return True
    if any(signature.kind is not RuntimeCellValueKind.NUMBER for signature in reference):
        return False
    if any(signature.kind is not RuntimeCellValueKind.NUMBER for signature in candidate):
        return False

    unstable_cap = max(
        policy.object_boundary_jitter_max_unstable_values,
        math.ceil(
            sum(reference.values())
            * policy.object_boundary_jitter_max_unstable_fraction
        ),
    )
    missing = sum((reference - candidate).values())
    extra = sum((candidate - reference).values())
    return max(missing, extra) <= unstable_cap


def _is_object_identifier_feature(feature_name: str) -> bool:
    return feature_name == "object_number" or "_object_number" in feature_name


def _numeric_counters_are_binary(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
) -> bool:
    numbers: set[float] = set()
    for counter in (reference, candidate):
        for signature in counter:
            numeric = _finite_signature_number(signature)
            if numeric is None:
                return False
            numbers.add(numeric)
    return numbers.issubset({0.0, 1.0})


def _sparse_object_identifier_counters_equivalent(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if reference == candidate:
        return True
    if any(signature.kind is not RuntimeCellValueKind.NUMBER for signature in reference):
        return False
    if any(signature.kind is not RuntimeCellValueKind.NUMBER for signature in candidate):
        return False

    unstable_cap = max(
        policy.object_boundary_jitter_max_unstable_values,
        math.ceil(sum(reference.values()) * policy.object_boundary_jitter_max_unstable_fraction),
    )
    missing = sum((reference - candidate).values())
    extra = sum((candidate - reference).values())
    return max(missing, extra) <= unstable_cap


def _unstable_shape_descriptor_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if not policy.allow_unstable_shape_descriptors:
        return False
    if feature.subject.scope is not MeasurementScope.OBJECT:
        return False
    if feature.statistic != "value":
        return False
    if not _shape_descriptor_geometry_is_stable(feature, reference, candidate, policy):
        return False

    reference_values = reference.values_by_feature[feature]
    candidate_values = candidate.values_by_feature[feature]
    if _is_zernike_feature(feature.feature_name):
        return _sparse_numeric_counters_equivalent(
            reference_values,
            candidate_values,
            policy,
            abs_tolerance=policy.shape_descriptor_abs_tolerance,
            rel_tolerance=policy.shape_descriptor_rel_tolerance,
            max_unstable_values=policy.shape_descriptor_max_unstable_values,
            max_unstable_fraction=policy.shape_descriptor_max_unstable_fraction,
        )
    if feature.feature_name in _ORIENTATION_FEATURES:
        if policy.allow_sparse_object_boundary_jitter:
            return _object_boundary_jitter_sparse_absolute_numeric_counters_equivalent(
                reference_values,
                candidate_values,
                policy,
            )
        return _absolute_numeric_counters_equivalent(
            reference_values,
            candidate_values,
            policy,
        )
    return False


def _shape_descriptor_geometry_is_stable(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    matched_features = 0
    for feature_name in _SHAPE_DESCRIPTOR_GATING_FEATURES:
        geometry_feature = RuntimeMeasurementFeatureKey(
            subject=feature.subject,
            feature_name=feature_name,
            statistic=feature.statistic,
            source_name=feature.source_name,
        )
        reference_values = reference.values_by_feature.get(geometry_feature)
        candidate_values = candidate.values_by_feature.get(geometry_feature)
        if reference_values is None and candidate_values is None:
            continue
        if reference_values is None or candidate_values is None:
            continue
        if _cell_signature_counters_equivalent(
            reference_values,
            candidate_values,
            policy,
        ):
            matched_features += 1
            continue
        if policy.allow_sparse_object_boundary_jitter and (
            _sparse_object_boundary_value_feature_equivalent(
                geometry_feature,
                reference,
                candidate,
                policy,
            )
        ):
            matched_features += 1
            continue
        else:
            return False
    return matched_features >= 3


def _is_zernike_feature(feature_name: str) -> bool:
    return bool(re.fullmatch(r"zernike_\d+_\d+", feature_name))


def _feature_label(feature: RuntimeMeasurementFeatureKey) -> str:
    subject = feature.subject
    if subject.name is None:
        subject_label = subject.scope.value
    else:
        subject_label = f"{subject.scope.value}:{subject.name}"
    feature_label = feature.feature_name
    if feature.source_name is not None:
        feature_label = f"{feature_label}@{feature.source_name}"
    if feature.statistic == "value":
        return f"{subject_label}/{feature_label}"
    return f"{subject_label}/{feature.statistic}({feature_label})"


def _record_measurement_facts(
    values_by_feature: _RuntimeMeasurementFactCounters,
    facts: Iterable[_RuntimeMeasurementFact],
    *,
    required_keys: _RuntimeRequiredMeasurementKeys = None,
) -> None:
    for key, value in facts:
        if required_keys is not None and key not in required_keys:
            continue
        values_by_feature.setdefault(key, Counter()).update((value,))


def _spatial_grid_measurement_facts(
    value: RuntimeValue,
    policy: RuntimeEquivalencePolicy,
) -> _RuntimeMeasurementFacts:
    """Project a typed spatial-grid artifact to CellProfiler-style image facts."""
    grid = SpatialGrid.from_runtime_value(value)
    grid_name = _normalize_identifier(value.name or grid.name)
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")
    fields = (
        ("columns", grid.columns),
        ("rows", grid.rows),
        ("x_location_of_lowest_x_spot", grid.x_location_of_lowest_x_spot),
        ("x_spacing", grid.x_spacing),
        ("y_location_of_lowest_y_spot", grid.y_location_of_lowest_y_spot),
        ("y_spacing", grid.y_spacing),
    )
    return tuple(
        (
            _runtime_measurement_feature_key(
                subject,
                f"defined_grid_{grid_name}_{field_name}",
            ),
            _cell_signature(str(field_value), policy),
        )
        for field_name, field_value in fields
    )


def _record_static_wide_measurement_table_snapshot(
    values_by_feature: _RuntimeMeasurementFactCounters,
    table: RuntimeTableSnapshot,
    policy: RuntimeEquivalencePolicy,
    *,
    known_source_names: tuple[str, ...],
) -> bool:
    """Record wide exported measurement tables without per-row key rebuilding."""
    if not table.rows:
        return True

    first_row = dict(zip(table.header, table.rows[0], strict=True))
    if not _is_wide_measurement_table(first_row, policy.measurement_dialect):
        return False

    first_subject = _measurement_subject_from_export_row(table.path, first_row)
    if _is_metadata_map_row(first_subject, first_row):
        return True

    normalized_fields = tuple(_normalize_identifier(field) for field in table.header)
    identity_indexes = {
        index
        for index, field_name in enumerate(normalized_fields)
        if _is_measurement_identity_field(field_name, policy.measurement_dialect)
    }
    feature_column_indexes = tuple(
        index
        for index in range(len(table.header))
        if index not in identity_indexes
    )
    if not feature_column_indexes:
        return True
    feature_column_indexes = tuple(
        index
        for index in feature_column_indexes
        if not _is_aggregate_image_number_reference_measurement_field(
            table.header[index]
        )
    )
    if not feature_column_indexes:
        return True
    column_subject_contexts = {
        index: _export_column_subject_context(table, index)
        for index in feature_column_indexes
    }
    padding_groups_by_index = _contextual_measurement_padding_groups(
        table.column_context,
        table.header,
        feature_column_indexes,
        policy.measurement_dialect,
        known_source_names=known_source_names,
    )
    if _wide_measurement_table_needs_row_derivation(
        table,
        feature_column_indexes,
        policy,
        source_name=measurement_row_source_image_name(first_row),
        known_source_names=known_source_names,
    ):
        return False
    image_number_offset = _table_image_number_offset(table.header, table.rows)

    key_cache: dict[
        tuple[RuntimeMeasurementSubjectKey, str | None, int, tuple[str, ...]],
        RuntimeMeasurementFeatureKey | None,
    ] = {}
    row_qualifier_columns = _row_qualifier_columns(
        normalized_fields,
        policy.measurement_dialect,
    )
    uses_row_qualifiers = bool(row_qualifier_columns)
    collapsed_numeric_qualifier_by_index = {
        index: _measurement_field_has_collapsed_numeric_qualifier(
            table.header[index],
            policy.measurement_dialect,
            known_source_names=known_source_names,
        )
        for index in feature_column_indexes
    }

    for row in table.rows:
        row_mapping = dict(zip(table.header, row, strict=True))
        row_subject = _measurement_subject_from_export_row(table.path, row_mapping)
        source_name = measurement_row_source_image_name(row_mapping)
        row_object_name = measurement_row_object_name(row_mapping)
        normalized_row_object_name = (
            _normalize_identifier(row_object_name)
            if row_object_name is not None
            else None
        )
        padding_indexes = _contextual_measurement_padding_indexes(
            table.column_context,
            table.header,
            row,
            feature_column_indexes,
            policy.measurement_dialect,
            known_source_names=known_source_names,
            padding_groups_by_index=padding_groups_by_index,
        )
        qualifier_values = (
            _row_qualifier_values(row, row_qualifier_columns)
            if uses_row_qualifiers
            else ()
        )
        row_facts: list[
            tuple[RuntimeMeasurementFeatureKey, RuntimeCellSignature, bool]
        ] = []
        for index in feature_column_indexes:
            if index in padding_indexes:
                continue
            context, normalized_context = column_subject_contexts[index]
            subject = _measurement_subject_from_column_context(
                context,
                normalized_context,
                row_object_name,
                normalized_row_object_name,
                fallback_subject=row_subject,
            )
            qualifiers = (
                _measurement_row_qualifiers_from_values(
                    qualifier_values,
                    policy.measurement_dialect,
                    table.header[index],
                )
                if uses_row_qualifiers
                else ()
            )
            cache_key = (subject, source_name, index, qualifiers)
            key = key_cache.get(cache_key, _CACHE_MISS)
            if key is _CACHE_MISS:
                key = _measurement_feature_key_from_source_context(
                    _MeasurementFeatureKeySourceContext(
                        table.header[index],
                        subject,
                        policy,
                        qualifiers,
                        source_name,
                        known_source_names,
                    )
                )
                key_cache[cache_key] = key
            if key is None:
                continue
            field_name = table.header[index]
            row_facts.append(
                (
                    key,
                    _cell_signature(
                        str(
                            _normalize_image_number_reference_measurement_value(
                                field_name,
                                row[index],
                                image_number_offset,
                            )
                        ),
                        policy,
                    ),
                    collapsed_numeric_qualifier_by_index[index],
                )
            )
        for key, value in _dedupe_measurement_fact_records(row_facts):
            values_by_feature.setdefault(key, Counter())[value] += 1
    return True


def _record_static_wide_runtime_measurement_table(
    values_by_feature: _RuntimeMeasurementFactCounters,
    context: _RuntimeMeasurementTableProjectionContext,
) -> bool:
    """Record wide runtime measurement tables without per-row key rebuilding."""
    table = context.table
    policy = context.policy
    all_rows = measurement_rows((table,))
    if not all_rows:
        return True

    first_mapping = measurement_row_mapping(all_rows[0])
    if not _is_wide_measurement_table(first_mapping, policy.measurement_dialect):
        return False
    header = tuple(first_mapping)
    for row in all_rows[1:]:
        row_mapping = measurement_row_mapping(row)
        if tuple(row_mapping) != header:
            return False

    table_subject = RuntimeMeasurementSubjectKey.from_subject(table.subject)
    subject_schema = _runtime_measurement_row_subject_schema(header)
    first_row_values = tuple(first_mapping.get(field_name) for field_name in header)
    first_subject = _measurement_subject_from_runtime_row_values(
        table_subject,
        first_row_values,
        subject_schema,
    )
    if _is_metadata_map_row(first_subject, first_mapping):
        return True

    normalized_fields = tuple(_normalize_identifier(field) for field in header)
    normalized_field_indexes = {
        field_name: index
        for index, field_name in enumerate(normalized_fields)
    }
    identity_indexes = {
        index
        for index, field_name in enumerate(normalized_fields)
        if _is_normalized_measurement_identity_field(
            field_name,
            policy.measurement_dialect,
        )
    }
    feature_column_indexes = tuple(
        index for index in range(len(header)) if index not in identity_indexes
    )
    if not feature_column_indexes:
        return True
    if _wide_measurement_table_needs_row_derivation(
        header,
        feature_column_indexes,
        policy,
        source_name=_measurement_source_name_from_runtime_row_values(
            table.source_image_name,
            first_row_values,
            subject_schema,
        ),
        known_source_names=context.known_source_names,
    ):
        return False

    input_keys = _required_measurement_input_keys(
        context.required_keys,
        known_source_names=context.known_source_names,
    )
    required_subjects = _required_measurement_subjects(input_keys)
    qualifier_indexes = {
        qualifier: tuple(
            normalized_field_indexes.get(field_name)
            for field_name in qualifier.field_names
        )
        for qualifier in policy.measurement_dialect.row_qualifiers
    }
    qualifiers_by_index = {
        index: tuple(
            (qualifier, qualifier_indexes[qualifier])
            for qualifier in policy.measurement_dialect.row_qualifiers
            if _row_qualifier_applies_to_field(
                qualifier,
                tuple(part for part in normalized_fields[index].split("_") if part),
            )
        )
        for index in feature_column_indexes
    }
    qualifier_render_cache: _RuntimeMeasurementQualifierRenderCache = {}
    padding_group_cache: _RuntimeMeasurementPaddingGroupCache = {}
    key_cache: _StaticWideRuntimeKeyCache = {}
    aggregate_reference_indexes = frozenset(
        index
        for index in feature_column_indexes
        if _is_aggregate_image_number_reference_measurement_field(header[index])
    )
    aggregate_values_by_feature: _AggregateValuesByFeature = {}
    aggregate_input_key_cache: _AggregateMeanKeyCache = {}
    table_padding_group = _normalize_identifier(table.name) or "measurements"
    table_explicit_measurement_keys: set[RuntimeMeasurementFeatureKey] = set()
    fact_recording_context = _RuntimeMeasurementFactRecordingContext(
        values_by_feature,
        table_explicit_measurement_keys,
        context.required_keys,
    )
    row_projection_context = _StaticWideRuntimeRowProjectionContext(
        header,
        policy,
        context.known_source_names,
        input_keys,
        feature_column_indexes,
        aggregate_reference_indexes,
        qualifiers_by_index,
        qualifier_render_cache,
        key_cache,
        padding_group_cache,
        table_padding_group,
    )

    for row in all_rows:
        row_mapping = measurement_row_mapping(row)
        row_values = tuple(row_mapping.get(field_name) for field_name in header)
        subject = _measurement_subject_from_runtime_row_values(
            table_subject,
            row_values,
            subject_schema,
        )
        if (
            required_subjects is not None
            and subject.scope is MeasurementScope.OBJECT
            and subject not in required_subjects
        ):
            continue
        source_name = _measurement_source_name_from_runtime_row_values(
            table.source_image_name,
            row_values,
            subject_schema,
        )
        derived_row_facts = _static_wide_runtime_measurement_row_facts(
            row_projection_context,
            row_values,
            subject,
            source_name,
        )
        if not derived_row_facts:
            continue
        aggregate_input_context = _AggregateInputRecordingContext(
            aggregate_values_by_feature,
            row_mapping,
            context.axis_key,
            context.required_keys,
            aggregate_input_key_cache,
        )
        _record_runtime_measurement_facts_for_row(
            fact_recording_context,
            aggregate_input_context,
            derived_row_facts,
        )

    explicit_measurement_keys = frozenset(table_explicit_measurement_keys)
    _record_runtime_aggregate_mean_facts(
        values_by_feature,
        aggregate_values_by_feature,
        explicit_measurement_keys,
        policy,
        required_keys=context.required_keys,
    )
    return True


def _static_wide_runtime_measurement_row_facts(
    context: _StaticWideRuntimeRowProjectionContext,
    row_values: tuple[object, ...],
    subject: RuntimeMeasurementSubjectKey,
    source_name: str | None,
) -> _RuntimeMeasurementFacts:
    row_fact_records: list[_RuntimeRowProjectionRecord[RuntimeCellSignature]] = []
    padding_group_presence: dict[
        tuple[RuntimeMeasurementSubjectKey, str | None, tuple[str, ...]],
        bool,
    ] = {}
    row_qualifier_cache: _RuntimeMeasurementIndexedQualifierCache = {}
    for index in context.feature_column_indexes:
        if index in context.aggregate_reference_indexes:
            continue
        field_name = context.header[index]
        value = row_values[index]
        indexed_qualifiers = context.qualifiers_by_index[index]
        if not indexed_qualifiers:
            qualifiers = ()
        else:
            qualifiers = row_qualifier_cache.get(indexed_qualifiers)
            if qualifiers is None:
                qualifiers = _measurement_row_qualifiers_from_indexed_values_cached(
                    row_values,
                    indexed_qualifiers,
                    context.qualifier_render_cache,
                )
                row_qualifier_cache[indexed_qualifiers] = qualifiers
        cache_key = (subject, source_name, index, qualifiers)
        key = context.key_cache.get(cache_key, _CACHE_MISS)
        if key is _CACHE_MISS:
            key = _measurement_feature_key_from_source_context(
                _MeasurementFeatureKeySourceContext(
                    field_name,
                    subject,
                    context.policy,
                    qualifiers,
                    source_name,
                    context.known_source_names,
                )
            )
            context.key_cache[cache_key] = key
        if key is None:
            continue
        padding_group_cache_key = (field_name, key)
        padding_group = context.padding_group_cache.get(padding_group_cache_key)
        if padding_group is None:
            padding_group = _runtime_measurement_padding_group(
                context.table_padding_group,
                field_name,
                key,
                context.policy.measurement_dialect,
            )
            context.padding_group_cache[padding_group_cache_key] = padding_group
        padding_group_presence[padding_group] = (
            padding_group_presence.get(padding_group, False)
            or _measurement_value_is_present(value)
        )
        if (
            context.input_keys is not None
            and key not in context.input_keys
            and not isinstance(value, Mapping)
        ):
            continue
        cell_facts = _cell_measurement_facts(key, value, context.policy)
        if context.input_keys is not None:
            cell_facts = tuple(
                (cell_key, cell_value)
                for cell_key, cell_value in cell_facts
                if cell_key in context.input_keys
            )
        row_fact_records.extend(
            (padding_group, cell_key, cell_value)
            for cell_key, cell_value in cell_facts
        )

    row_facts = tuple(
        (key, value)
        for padding_group, key, value in row_fact_records
        if padding_group_presence.get(padding_group, True)
    )
    return _derive_pair_measurement_facts(
        _dedupe_measurement_facts(row_facts),
        context.policy,
        known_source_names=context.known_source_names,
    )


def _record_row_aggregate_input_value(
    context: _AggregateInputRecordingContext,
    key: RuntimeMeasurementFeatureKey,
    value: RuntimeCellSignature,
    *,
    row_identity: _RuntimeMeasurementRowIdentityOrMissing,
) -> _RuntimeMeasurementRowIdentityOrMissing:
    """Record object values needed to derive row-identity-scoped means."""
    if key.subject.scope is not MeasurementScope.OBJECT:
        return row_identity
    if key.statistic != "value":
        return row_identity
    mean_key = _aggregate_mean_key(
        key,
        required_keys=context.required_keys,
        key_cache=context.key_cache,
    )
    if mean_key is None:
        return row_identity
    numeric_value = _finite_numeric_runtime_cell_value(value)
    if numeric_value is None:
        return row_identity
    if row_identity is None:
        row_identity = _axis_scoped_measurement_row_identity(
            context.row_mapping,
            context.axis_key,
        )
    context.values_by_feature.setdefault((mean_key, row_identity), []).append(
        numeric_value
    )
    return row_identity


def _record_runtime_measurement_facts_for_row(
    fact_context: _RuntimeMeasurementFactRecordingContext,
    aggregate_context: _AggregateInputRecordingContext,
    facts: Iterable[_RuntimeMeasurementFact],
) -> None:
    row_identity: _RuntimeMeasurementRowIdentityOrMissing = None
    for key, value in facts:
        fact_context.explicit_measurement_keys.add(key)
        if fact_context.required_keys is None or key in fact_context.required_keys:
            fact_context.values_by_feature.setdefault(key, Counter())[value] += 1
        row_identity = _record_row_aggregate_input_value(
            aggregate_context,
            key,
            value,
            row_identity=row_identity,
        )


def _record_runtime_aggregate_mean_facts(
    values_by_feature: _RuntimeMeasurementFactCounters,
    aggregate_values_by_feature: _AggregateValuesByFeature,
    explicit_measurement_keys: _RuntimeMeasurementKeySet,
    policy: RuntimeEquivalencePolicy,
    *,
    required_keys: _RuntimeRequiredMeasurementKeys,
) -> None:
    for (mean_key, _row_identity), values in aggregate_values_by_feature.items():
        if not values:
            continue
        if mean_key in explicit_measurement_keys:
            continue
        if required_keys is not None and mean_key not in required_keys:
            continue
        values_by_feature.setdefault(mean_key, Counter())[
            _cell_signature(str(sum(values) / len(values)), policy)
        ] += 1


def _table_image_number_offset(
    header: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
) -> float:
    image_number_indexes = tuple(
        index
        for index, field_name in enumerate(header)
        if _normalize_identifier(field_name) == "image_number"
    )
    if not image_number_indexes:
        return 0.0
    image_number_index = image_number_indexes[0]
    image_numbers: list[float] = []
    for row in rows:
        if image_number_index >= len(row):
            continue
        try:
            image_number = float(str(row[image_number_index]).strip())
        except ValueError:
            continue
        if math.isfinite(image_number) and image_number > 0:
            image_numbers.append(image_number)
    if not image_numbers:
        return 0.0
    return min(image_numbers) - 1.0


def _normalize_image_number_reference_measurement_value(
    field_name: str,
    value: object,
    image_number_offset: float,
) -> object:
    if image_number_offset == 0:
        return value
    if not _is_image_number_reference_measurement_field(field_name):
        return value
    if isinstance(value, Mapping):
        return {
            key: _normalize_image_number_reference_measurement_value(
                field_name,
                nested_value,
                image_number_offset,
            )
            for key, nested_value in value.items()
        }
    try:
        numeric_value = float(str(value).strip())
    except ValueError:
        return value
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        return value
    normalized = numeric_value - image_number_offset
    return int(normalized) if normalized.is_integer() else normalized


@lru_cache(maxsize=32768)
def _is_aggregate_image_number_reference_measurement_field(field_name: str) -> bool:
    parts = tuple(
        part for part in _normalize_identifier(field_name).split("_") if part
    )
    return (
        bool(parts)
        and parts[0] in _MEASUREMENT_AGGREGATE_PREFIXES
        and _is_image_number_reference_measurement_field(field_name)
    )


def _is_image_number_reference_measurement_field(field_name: str) -> bool:
    normalized = _normalize_identifier(field_name)
    if normalized in _IMAGE_IDENTITY_FIELDS:
        return False
    parts = tuple(part for part in normalized.split("_") if part)
    return _parts_contain_adjacent_image_number(parts)


def _is_image_number_reference_feature(key: RuntimeMeasurementFeatureKey) -> bool:
    parts = tuple(part for part in key.feature_name.split("_") if part)
    if _parts_contain_adjacent_image_number(parts):
        return True
    return (
        key.source_name == "image"
        and "parent" in parts
        and "number" in parts
    )


def _parts_contain_adjacent_image_number(parts: tuple[str, ...]) -> bool:
    return any(
        parts[index] == "image" and parts[index + 1] == "number"
        for index in range(len(parts) - 1)
    )


def _measurement_row_image_identity_key(
    row: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """Return the image identity carried by a measurement row."""
    identity_values: list[tuple[str, object]] = []
    for field_name, value in row.items():
        normalized_field_name = _normalize_identifier(field_name)
        if normalized_field_name not in _IMAGE_IDENTITY_FIELDS:
            continue
        if value is None or not str(value).strip():
            continue
        identity_values.append(
            (
                normalized_field_name,
                _measurement_table_cell_payload(value),
            )
        )
    return sorted_tuple(identity_values)


def _axis_scoped_measurement_row_identity(
    row: Mapping[str, object],
    axis_key: object | None,
) -> tuple[tuple[str, object], ...]:
    """Return row identity scoped by runtime axis for local image numbering."""
    row_identity = _measurement_row_image_identity_key(row)
    if axis_key is None:
        return row_identity
    return (
        ("_runtime_axis", _measurement_table_cell_payload(axis_key)),
        *row_identity,
    )


def _object_location_subjects(
    values_by_feature: Mapping[RuntimeMeasurementFeatureKey, object],
) -> frozenset[RuntimeMeasurementSubjectKey]:
    return frozenset(
        key.subject
        for key in values_by_feature
        if key.subject.scope is MeasurementScope.OBJECT
        and key.statistic == "value"
        and key.feature_name in _OBJECT_LOCATION_FEATURES
    )


def _object_identifier_subjects(
    values_by_feature: Mapping[RuntimeMeasurementFeatureKey, object],
) -> frozenset[RuntimeMeasurementSubjectKey]:
    return frozenset(
        key.subject
        for key in values_by_feature
        if key.subject.scope is MeasurementScope.OBJECT
        and key.statistic == "value"
        and key.source_name is None
        and _is_object_identifier_feature(key.feature_name)
    )


def _runtime_measurement_feature_key(
    subject: RuntimeMeasurementSubjectKey,
    feature_name: str,
    statistic: str = "value",
    source_name: str | None = None,
) -> RuntimeMeasurementFeatureKey:
    """Build a feature key using OpenHCS image subjects for image sources."""
    if subject.scope is MeasurementScope.IMAGE and source_name is not None:
        return RuntimeMeasurementFeatureKey(
            RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, source_name),
            feature_name,
            statistic,
        )
    return RuntimeMeasurementFeatureKey(
        subject,
        feature_name,
        statistic,
        source_name,
    )


def _measurement_feature_key_from_source_context(
    context: _MeasurementFeatureKeySourceContext,
) -> RuntimeMeasurementFeatureKey | None:
    return _measurement_feature_key_for_field(
        context.field_name,
        context.subject,
        context.policy,
        qualifiers=context.qualifiers,
        source_name=context.source_name,
        known_source_names=context.known_source_names,
    )


def _measurement_facts_from_table_snapshot(
    table: RuntimeTableSnapshot,
    policy: RuntimeEquivalencePolicy,
    *,
    known_source_names: tuple[str, ...],
) -> _RuntimeMeasurementFacts:
    static_facts = _static_wide_measurement_facts_from_table_snapshot(
        table,
        policy,
        known_source_names=known_source_names,
    )
    if static_facts is not None:
        return static_facts

    feature_indexes = tuple(
        index
        for index, field_name in enumerate(table.header)
        if not _is_measurement_identity_field(
            field_name,
            policy.measurement_dialect,
        )
    )
    padding_groups_by_index = _contextual_measurement_padding_groups(
        table.column_context,
        table.header,
        feature_indexes,
        policy.measurement_dialect,
        known_source_names=known_source_names,
    )

    facts: _RuntimeMeasurementFactList = []
    for row in table.rows:
        row_mapping = dict(zip(table.header, row, strict=True))
        row_subject = _measurement_subject_from_export_row(table.path, row_mapping)
        if _is_metadata_map_row(row_subject, row_mapping):
            continue
        source_name = measurement_row_source_image_name(row_mapping)
        padding_indexes = _contextual_measurement_padding_indexes(
            table.column_context,
            table.header,
            row,
            feature_indexes,
            policy.measurement_dialect,
            known_source_names=known_source_names,
            padding_groups_by_index=padding_groups_by_index,
        )
        long_form_fact = _long_form_measurement_fact(
            _LongFormMeasurementContext(
                row_mapping,
                row_subject,
                policy,
                source_name,
                known_source_names,
            )
        )
        if long_form_fact is not None:
            facts.append(long_form_fact)
            continue
        for index, field_name in enumerate(table.header):
            if index in padding_indexes:
                continue
            if _is_measurement_identity_field(field_name, policy.measurement_dialect):
                continue
            subject = _measurement_subject_from_export_column(
                table,
                row_mapping,
                index,
                fallback_subject=row_subject,
            )
            key = _measurement_feature_key_from_source_context(
                _MeasurementFeatureKeySourceContext(
                    field_name,
                    subject,
                    policy,
                    _measurement_row_qualifiers(
                        row_mapping,
                        policy.measurement_dialect,
                        field_name,
                    ),
                    source_name,
                    known_source_names,
                )
            )
            if key is None:
                continue
            if _is_aggregate_image_number_reference_measurement_field(field_name):
                continue
            facts.extend(
                _cell_measurement_facts(
                    key,
                    _normalize_image_number_reference_measurement_value(
                        field_name,
                        row_mapping[field_name],
                        _table_image_number_offset(table.header, table.rows),
                    ),
                    policy,
                )
            )
    return tuple(facts)


def _static_wide_measurement_facts_from_table_snapshot(
    table: RuntimeTableSnapshot,
    policy: RuntimeEquivalencePolicy,
    *,
    known_source_names: tuple[str, ...],
) -> _RuntimeMeasurementFacts | None:
    """Project wide exported tables by caching column-level feature semantics."""
    if not table.rows:
        return ()

    first_row = dict(zip(table.header, table.rows[0], strict=True))
    if not _is_static_wide_measurement_table(first_row, policy.measurement_dialect):
        return None

    subject = _measurement_subject_from_export_row(table.path, first_row)
    if _is_metadata_map_row(subject, first_row):
        return ()

    source_name = measurement_row_source_image_name(first_row)
    feature_columns = tuple(
        (index, key)
        for index, field_name in enumerate(table.header)
        if not _is_measurement_identity_field(field_name, policy.measurement_dialect)
        and not _is_aggregate_image_number_reference_measurement_field(field_name)
        for column_subject in (
            _measurement_subject_from_export_column(
                table,
                first_row,
                index,
                fallback_subject=subject,
            ),
        )
        for key in (
            _measurement_feature_key_from_source_context(
                _MeasurementFeatureKeySourceContext(
                    field_name,
                    column_subject,
                    policy,
                    (),
                    source_name,
                    known_source_names,
                )
            ),
        )
        if key is not None
    )
    if not feature_columns:
        return ()
    image_number_offset = _table_image_number_offset(table.header, table.rows)
    padding_groups_by_index = _contextual_measurement_padding_groups(
        table.column_context,
        table.header,
        tuple(index for index, _key in feature_columns),
        policy.measurement_dialect,
        known_source_names=known_source_names,
    )

    facts: _RuntimeMeasurementFactList = []
    for row in table.rows:
        padding_indexes = _contextual_measurement_padding_indexes(
            table.column_context,
            table.header,
            row,
            tuple(index for index, _key in feature_columns),
            policy.measurement_dialect,
            known_source_names=known_source_names,
            padding_groups_by_index=padding_groups_by_index,
        )
        row_facts = tuple(
            (
                key,
                _cell_signature(
                    str(
                        _normalize_image_number_reference_measurement_value(
                            table.header[index],
                            row[index],
                            image_number_offset,
                        )
                    ),
                    policy,
                ),
            )
            for index, key in feature_columns
            if index not in padding_indexes
        )
        facts.extend(
            _derive_pair_measurement_facts(
                _dedupe_measurement_facts(row_facts),
                policy,
                known_source_names=known_source_names,
            )
        )
    return tuple(facts)


def _is_static_wide_measurement_table(
    row: Mapping[str, object],
    dialect: RuntimeMeasurementDialect,
) -> bool:
    normalized_fields = {_normalize_identifier(field_name) for field_name in row}
    if normalized_fields & frozenset(_MEASUREMENT_FEATURE_NAME_FIELDS):
        return False
    if normalized_fields & frozenset(_MEASUREMENT_VALUE_FIELDS):
        return False
    if normalized_fields & _measurement_qualifier_field_names(dialect):
        return False
    return (
        MEASUREMENT_OBJECT_NAME_FIELD not in normalized_fields
        and MEASUREMENT_SOURCE_IMAGE_NAME_FIELD not in normalized_fields
    )


def _is_wide_measurement_table(
    row: Mapping[str, object],
    dialect: RuntimeMeasurementDialect,
) -> bool:
    """Return whether a table encodes measurements as feature columns."""
    normalized_fields = {_normalize_identifier(field_name) for field_name in row}
    if normalized_fields & frozenset(_MEASUREMENT_FEATURE_NAME_FIELDS):
        return False
    if normalized_fields & frozenset(_MEASUREMENT_VALUE_FIELDS):
        return False
    return True


def _wide_measurement_table_needs_row_derivation(
    table: RuntimeTableSnapshot | tuple[str, ...],
    feature_column_indexes: tuple[int, ...],
    policy: RuntimeEquivalencePolicy,
    *,
    source_name: str | None,
    known_source_names: tuple[str, ...],
) -> bool:
    """Return whether row-local derived facts require the slower projection."""
    header = table.header if isinstance(table, RuntimeTableSnapshot) else table
    first_row = (
        {}
        if not isinstance(table, RuntimeTableSnapshot) or not table.rows
        else dict(zip(table.header, table.rows[0], strict=True))
    )
    fallback_subject = (
        RuntimeMeasurementSubjectKey(MeasurementScope.ARTIFACT, None)
        if not isinstance(table, RuntimeTableSnapshot)
        else _measurement_subject_from_export_row(table.path, first_row)
    )
    for index in feature_column_indexes:
        subject = (
            fallback_subject
            if not isinstance(table, RuntimeTableSnapshot)
            else _measurement_subject_from_export_column(
                table,
                first_row,
                index,
                fallback_subject=fallback_subject,
            )
        )
        key = _measurement_feature_key_from_source_context(
            _MeasurementFeatureKeySourceContext(
                header[index],
                subject,
                policy,
                (),
                source_name,
                known_source_names,
            )
        )
        if key is not None and key.feature_name == _PAIR_REGRESSION_SLOPE_FEATURE:
            return True
    return False


def _measurement_facts_from_runtime_table(
    context: _RuntimeMeasurementTableProjectionContext,
) -> _RuntimeMeasurementFacts:
    table = context.table
    policy = context.policy
    facts: _RuntimeMeasurementFactList = []
    row_required_keys = _required_measurement_input_keys(
        context.required_keys,
        known_source_names=context.known_source_names,
    )
    row_required_subjects = _required_measurement_subjects(row_required_keys)
    schema_cache: _RuntimeMeasurementRowSchemaCache = {}
    key_cache: _RuntimeMeasurementFeatureKeyCache = {}
    long_form_key_cache: _RuntimeMeasurementLongFormKeyCache = {}
    qualifier_render_cache: _RuntimeMeasurementQualifierRenderCache = {}
    padding_group_cache: _RuntimeMeasurementPaddingGroupCache = {}
    subject_schema_cache: dict[
        tuple[str, ...],
        _RuntimeMeasurementRowSubjectSchema,
    ] = {}
    aggregate_values_by_feature: _AggregateValuesByFeature = {}
    aggregate_input_key_cache: _AggregateMeanKeyCache = {}
    table_subject = RuntimeMeasurementSubjectKey.from_subject(table.subject)
    table_padding_group = _normalize_identifier(table.name) or "measurements"
    for row in measurement_rows((table,)):
        row_mapping = measurement_row_mapping(row)
        header = tuple(row_mapping)
        row_values = tuple(row_mapping.get(field_name) for field_name in header)
        subject_schema = subject_schema_cache.get(header)
        if subject_schema is None:
            subject_schema = _runtime_measurement_row_subject_schema(header)
            subject_schema_cache[header] = subject_schema
        subject = _measurement_subject_from_runtime_row_values(
            table_subject,
            row_values,
            subject_schema,
        )
        if (
            row_required_subjects is not None
            and subject.scope is MeasurementScope.OBJECT
            and subject not in row_required_subjects
        ):
            continue
        source_name = _measurement_source_name_from_runtime_row_values(
            table.source_image_name,
            row_values,
            subject_schema,
        )
        row_context = _RuntimeRowProjectionContext.from_row(
            row_mapping,
            subject,
            policy,
            source_name=source_name,
            known_source_names=context.known_source_names,
            required_keys=row_required_keys,
            table_padding_group=table_padding_group,
            schema_cache=schema_cache,
            key_cache=key_cache,
            long_form_key_cache=long_form_key_cache,
            qualifier_render_cache=qualifier_render_cache,
            padding_group_cache=padding_group_cache,
        )
        row_facts = _measurement_facts_from_runtime_row_cached(row_context)
        facts.extend(row_facts)
        if not row_facts:
            continue
        row_identity: _RuntimeMeasurementRowIdentityOrMissing = None
        aggregate_input_context = _AggregateInputRecordingContext(
            aggregate_values_by_feature,
            row_mapping,
            context.axis_key,
            context.required_keys,
            aggregate_input_key_cache,
        )
        for key, value in row_facts:
            row_identity = _record_row_aggregate_input_value(
                aggregate_input_context,
                key,
                value,
                row_identity=row_identity,
            )

    explicit_keys = frozenset(key for key, _value in facts)
    derived_facts: _RuntimeMeasurementFactList = []
    for (mean_key, _row_identity), values in aggregate_values_by_feature.items():
        if not values or mean_key in explicit_keys:
            continue
        if context.required_keys is not None and mean_key not in context.required_keys:
            continue
        derived_facts.append(
            (
                mean_key,
                _cell_signature(str(sum(values) / len(values)), policy),
            )
        )
    facts.extend(derived_facts)
    return tuple(facts)


@dataclass(frozen=True, slots=True)
class _RuntimeRowProjectionEngine(Generic[_RuntimeRowProjectionValueT]):
    context: _RuntimeRowProjectionContext
    value_projector: _RuntimeRowValueProjector[_RuntimeRowProjectionValueT]
    long_form_projector: _RuntimeRowLongFormProjector[_RuntimeRowProjectionValueT]

    def project(self) -> _RuntimeRowProjection[_RuntimeRowProjectionValueT]:
        """Project one runtime row through shared schema/key/padding caches."""
        context = self.context
        if _is_metadata_map_row(context.subject, context.row):
            return _RuntimeRowProjection(())

        header = tuple(context.row)
        row_schema = _runtime_measurement_row_schema_cached(context, header)
        row_values = tuple(context.row.get(field_name) for field_name in header)
        long_form_projection = self._long_form_projection(row_schema, row_values)
        if long_form_projection is not None:
            return long_form_projection
        return self._wide_projection(header, row_schema, row_values)

    def _long_form_projection(
        self,
        row_schema: _RuntimeMeasurementRowSchema,
        row_values: tuple[object, ...],
    ) -> _RuntimeRowProjection[_RuntimeRowProjectionValueT] | None:
        context = self.context
        if (
            not row_schema.long_form_feature_indexes
            or not row_schema.long_form_value_indexes
        ):
            return None
        long_form_fact = _long_form_measurement_fact_cached(
            _CachedLongFormMeasurementContext.from_runtime_row_projection(
                context,
                row_values,
                row_schema.long_form_feature_indexes,
                row_schema.long_form_value_indexes,
            )
        )
        if long_form_fact is None:
            return None
        if (
            context.required_keys is not None
            and long_form_fact[0] not in context.required_keys
        ):
            return _RuntimeRowProjection((), long_form=True)
        return _RuntimeRowProjection(
            tuple(
                ((context.subject, context.source_name, ()), key, value)
                for key, value in self.long_form_projector(long_form_fact)
                if context.required_keys is None or key in context.required_keys
            ),
            long_form=True,
        )

    def _wide_projection(
        self,
        header: tuple[str, ...],
        row_schema: _RuntimeMeasurementRowSchema,
        row_values: tuple[object, ...],
    ) -> _RuntimeRowProjection[_RuntimeRowProjectionValueT]:
        records: list[_RuntimeRowProjectionRecord[_RuntimeRowProjectionValueT]] = []
        padding_group_presence: _RuntimeMeasurementPaddingGroupPresence = {}
        row_qualifier_cache: _RuntimeRowQualifierResolutionCache = {}
        for index in row_schema.feature_indexes:
            records.extend(
                self._wide_projection_records_for_index(
                    header,
                    row_schema,
                    row_values,
                    index,
                    row_qualifier_cache,
                    padding_group_presence,
                )
            )
        return _RuntimeRowProjection(
            tuple(
                (padding_group, key, value)
                for padding_group, key, value in records
                if padding_group_presence.get(padding_group, True)
            )
        )

    def _wide_projection_records_for_index(
        self,
        header: tuple[str, ...],
        row_schema: _RuntimeMeasurementRowSchema,
        row_values: tuple[object, ...],
        index: int,
        row_qualifier_cache: _RuntimeRowQualifierResolutionCache,
        padding_group_presence: _RuntimeMeasurementPaddingGroupPresence,
    ) -> _RuntimeRowProjectionRecords[_RuntimeRowProjectionValueT]:
        context = self.context
        field_name = header[index]
        value = row_values[index]
        qualifiers = self._qualifiers_for_index(
            row_schema,
            row_values,
            index,
            row_qualifier_cache,
        )
        key = self._feature_key(field_name, qualifiers)
        if key is None:
            return ()
        padding_group = self._padding_group(field_name, key)
        padding_group_presence[padding_group] = (
            padding_group_presence.get(padding_group, False)
            or _measurement_value_is_present(value)
        )
        if (
            context.required_keys is not None
            and key not in context.required_keys
            and not isinstance(value, Mapping)
        ):
            return ()
        projected_values = self.value_projector(key, value, context.policy)
        if context.required_keys is not None:
            projected_values = tuple(
                (cell_key, cell_value)
                for cell_key, cell_value in projected_values
                if cell_key in context.required_keys
            )
        return tuple(
            (padding_group, cell_key, cell_value)
            for cell_key, cell_value in projected_values
        )

    def _qualifiers_for_index(
        self,
        row_schema: _RuntimeMeasurementRowSchema,
        row_values: tuple[object, ...],
        index: int,
        row_qualifier_cache: _RuntimeRowQualifierResolutionCache,
    ) -> tuple[str, ...]:
        indexed_qualifiers = row_schema.qualifiers_by_index[index]
        if not indexed_qualifiers:
            return ()
        qualifiers = row_qualifier_cache.get(indexed_qualifiers)
        if qualifiers is None:
            qualifiers = _measurement_row_qualifiers_from_indexed_values_cached(
                row_values,
                indexed_qualifiers,
                self.context.qualifier_render_cache,
            )
            row_qualifier_cache[indexed_qualifiers] = qualifiers
        return qualifiers

    def _feature_key(
        self,
        field_name: str,
        qualifiers: tuple[str, ...],
    ) -> RuntimeMeasurementFeatureKey | None:
        context = self.context
        cache_key = (context.subject, context.source_name, field_name, qualifiers)
        key = context.key_cache.get(cache_key, _CACHE_MISS)
        if key is _CACHE_MISS:
            key = _measurement_feature_key_from_source_context(
                _MeasurementFeatureKeySourceContext(
                    field_name,
                    context.subject,
                    context.policy,
                    qualifiers,
                    context.source_name,
                    context.known_source_names,
                )
            )
            context.key_cache[cache_key] = key
        return key

    def _padding_group(
        self,
        field_name: str,
        key: RuntimeMeasurementFeatureKey,
    ) -> _RuntimeMeasurementPaddingGroup:
        context = self.context
        cache_key = (field_name, key)
        padding_group = context.padding_group_cache.get(cache_key)
        if padding_group is None:
            padding_group = _runtime_measurement_padding_group(
                context.table_padding_group,
                field_name,
                key,
                context.policy.measurement_dialect,
            )
            context.padding_group_cache[cache_key] = padding_group
        return padding_group


def _runtime_row_projection_cached(
    context: _RuntimeRowProjectionContext,
    *,
    value_projector: _RuntimeRowValueProjector[_RuntimeRowProjectionValueT],
    long_form_projector: _RuntimeRowLongFormProjector[_RuntimeRowProjectionValueT],
) -> _RuntimeRowProjection[_RuntimeRowProjectionValueT]:
    return _RuntimeRowProjectionEngine(
        context,
        value_projector,
        long_form_projector,
    ).project()


def _runtime_measurement_row_schema_cached(
    context: _RuntimeRowProjectionContext,
    header: tuple[str, ...],
) -> _RuntimeMeasurementRowSchema:
    cached_schema = context.schema_cache.get(header)
    if cached_schema is not None:
        return cached_schema

    normalized_fields = tuple(_normalize_identifier(field) for field in header)
    aggregate_reference_indexes = frozenset(
        index
        for index, field_name in enumerate(header)
        if _is_aggregate_image_number_reference_measurement_field(field_name)
    )
    normalized_field_indexes = {
        field_name: index
        for index, field_name in enumerate(normalized_fields)
    }
    feature_indexes = tuple(
        index
        for index, field_name in enumerate(normalized_fields)
        if not _is_normalized_measurement_identity_field(
            field_name,
            context.policy.measurement_dialect,
        )
        and index not in aggregate_reference_indexes
    )
    qualifier_indexes = {
        qualifier: tuple(
            normalized_field_indexes.get(field_name)
            for field_name in qualifier.field_names
        )
        for qualifier in context.policy.measurement_dialect.row_qualifiers
    }
    qualifiers_by_index = {
        index: tuple(
            (qualifier, qualifier_indexes[qualifier])
            for qualifier in context.policy.measurement_dialect.row_qualifiers
            if _row_qualifier_applies_to_field(
                qualifier,
                tuple(
                    part
                    for part in normalized_fields[index].split("_")
                    if part
                ),
            )
        )
        for index in feature_indexes
    }
    cached_schema = _RuntimeMeasurementRowSchema(
        feature_indexes,
        qualifiers_by_index,
        _field_indexes_for_names(
            normalized_field_indexes,
            _MEASUREMENT_FEATURE_NAME_FIELDS,
        ),
        _field_indexes_for_names(
            normalized_field_indexes,
            _MEASUREMENT_VALUE_FIELDS,
        ),
    )
    context.schema_cache[header] = cached_schema
    return cached_schema


def _measurement_facts_from_runtime_row_cached(
    context: _RuntimeRowProjectionContext,
) -> _RuntimeMeasurementFacts:
    """Project one runtime row using table-local schema/key caches."""
    projection = _runtime_row_projection_cached(
        context,
        value_projector=_cell_measurement_facts,
        long_form_projector=lambda fact: (fact,),
    )
    row_facts = tuple(
        (key, value)
        for _padding_group, key, value in projection.records
    )
    if projection.long_form:
        return row_facts
    derived_facts = _derive_pair_measurement_facts(
        _dedupe_measurement_facts(row_facts),
        context.policy,
        known_source_names=context.known_source_names,
    )
    if context.required_keys is not None:
        return tuple(
            (key, value)
            for key, value in derived_facts
            if key in context.required_keys
        )
    return derived_facts


def _required_measurement_input_keys(
    required_keys: _RuntimeRequiredMeasurementKeys,
    *,
    known_source_names: tuple[str, ...] = (),
) -> _RuntimeRequiredMeasurementKeys:
    if required_keys is None:
        return None
    keys: set[RuntimeMeasurementFeatureKey] = set(required_keys)
    for key in required_keys:
        if key.statistic == "mean":
            keys.add(
                RuntimeMeasurementFeatureKey(
                    key.subject,
                    key.feature_name,
                    "value",
                    key.source_name,
                )
            )
        keys.update(_source_orientation_input_keys(key, known_source_names))
    return frozenset(keys)


def _source_orientation_input_keys(
    key: RuntimeMeasurementFeatureKey,
    known_source_names: tuple[str, ...],
) -> tuple[RuntimeMeasurementFeatureKey, ...]:
    """Return same-feature keys for source-name orientations needed for derivation."""
    if key.subject.scope is not MeasurementScope.IMAGE or key.subject.name is None:
        return ()
    input_keys: list[RuntimeMeasurementFeatureKey] = []
    for source_name in known_source_names:
        source_parts = tuple(part for part in source_name.split("__") if part)
        if len(source_parts) != 2:
            continue
        forward_subject = RuntimeMeasurementSubjectKey(
            MeasurementScope.IMAGE,
            "__".join(source_parts),
        )
        reverse_subject = RuntimeMeasurementSubjectKey(
            MeasurementScope.IMAGE,
            "__".join(reversed(source_parts)),
        )
        if key.subject == forward_subject:
            input_keys.append(
                RuntimeMeasurementFeatureKey(
                    reverse_subject,
                    key.feature_name,
                    key.statistic,
                    key.source_name,
                )
            )
        if key.subject == reverse_subject:
            input_keys.append(
                RuntimeMeasurementFeatureKey(
                    forward_subject,
                    key.feature_name,
                    key.statistic,
                    key.source_name,
                )
            )
    return tuple(input_keys)


def _required_measurement_subjects(
    required_keys: _RuntimeRequiredMeasurementKeys,
) -> frozenset[RuntimeMeasurementSubjectKey] | None:
    if required_keys is None:
        return None
    subjects: set[RuntimeMeasurementSubjectKey] = set()
    for key in required_keys:
        subjects.add(key.subject)
        if (
            key.subject.scope is MeasurementScope.IMAGE
            and key.subject.name is not None
        ):
            source_parts = tuple(
                part for part in key.subject.name.split("__") if part
            )
            if len(source_parts) == 2:
                subjects.add(
                    RuntimeMeasurementSubjectKey(
                        MeasurementScope.IMAGE,
                        "__".join(reversed(source_parts)),
                    )
                )
    return frozenset(subjects)


def _is_metadata_map_row(
    subject: RuntimeMeasurementSubjectKey,
    row: Mapping[str, object],
) -> bool:
    if subject.scope is not MeasurementScope.EXPERIMENT:
        return False
    normalized_fields = frozenset(
        _normalize_identifier(field_name) for field_name in row
    )
    return normalized_fields == frozenset(("key", "value"))


def _dedupe_measurement_facts(
    facts: Iterable[_RuntimeMeasurementFact],
) -> _RuntimeMeasurementFacts:
    """Collapse same-row aliases that map to the same semantic feature key."""
    values_by_key: dict[RuntimeMeasurementFeatureKey, RuntimeCellSignature] = {}
    for key, value in facts:
        current = values_by_key.get(key)
        if current is None or (
            RuntimeCellMissingStrategy.for_kind(current.kind).is_missing(current)
            and not RuntimeCellMissingStrategy.for_kind(value.kind).is_missing(value)
        ):
            values_by_key[key] = value
    return tuple(values_by_key.items())


def _dedupe_measurement_fact_records(
    facts: Iterable[
        tuple[RuntimeMeasurementFeatureKey, RuntimeCellSignature, bool]
    ],
) -> _RuntimeMeasurementFacts:
    """Collapse aliases unless field normalization intentionally dropped a qualifier."""
    values_by_key: dict[RuntimeMeasurementFeatureKey, list[RuntimeCellSignature]] = {}
    qualified_by_key: dict[RuntimeMeasurementFeatureKey, bool] = {}
    for key, value, qualified_observation in facts:
        current_values = values_by_key.get(key)
        if current_values is None:
            values_by_key[key] = [value]
            qualified_by_key[key] = qualified_observation
            continue

        if RuntimeCellMissingStrategy.for_kind(value.kind).is_missing(value):
            continue
        if any(
            RuntimeCellMissingStrategy.for_kind(current.kind).is_missing(current)
            for current in current_values
        ):
            values_by_key[key] = [value]
            qualified_by_key[key] = qualified_observation
            continue
        if value in current_values:
            continue
        if qualified_by_key.get(key, False) or qualified_observation:
            current_values.append(value)

    return tuple(
        (key, value)
        for key, values in values_by_key.items()
        for value in values
    )


def _measurement_field_has_collapsed_numeric_qualifier(
    field_name: str,
    dialect: RuntimeMeasurementDialect,
    *,
    known_source_names: tuple[str, ...],
) -> bool:
    """Return true when semantic normalization drops a numeric feature qualifier."""
    parts = tuple(part for part in _normalize_identifier(field_name).split("_") if part)
    category_prefix = _measurement_category_prefix(parts, dialect)
    if category_prefix:
        parts = parts[len(category_prefix) :]
    parts, _source_names = _extract_source_qualifier_tokens(
        parts,
        known_source_names=known_source_names,
        dialect=dialect,
    )
    return _semantic_core_measurement_feature_parts(parts, dialect) != parts


def _contextual_measurement_padding_indexes(
    column_context: tuple[str | None, ...],
    header: tuple[str, ...],
    row_values: tuple[object, ...],
    feature_indexes: tuple[int, ...],
    dialect: RuntimeMeasurementDialect,
    *,
    known_source_names: tuple[str, ...],
    padding_groups_by_index: Mapping[
        int,
        _ContextualMeasurementPaddingGroup | None,
    ] | None = None,
) -> frozenset[int]:
    """Return contextual wide-table cells that are row padding, not facts."""
    if not column_context:
        return frozenset()

    if padding_groups_by_index is None:
        padding_groups_by_index = _contextual_measurement_padding_groups(
            column_context,
            header,
            feature_indexes,
            dialect,
            known_source_names=known_source_names,
        )

    indexes_by_group: dict[_ContextualMeasurementPaddingGroup, list[int]] = {}
    for index in feature_indexes:
        group = padding_groups_by_index.get(index)
        if group is None:
            continue
        indexes_by_group.setdefault(group, []).append(index)

    padding_indexes: set[int] = set()
    for indexes in indexes_by_group.values():
        if any(_measurement_cell_is_present(row_values[index]) for index in indexes):
            continue
        padding_indexes.update(indexes)
    return frozenset(padding_indexes)


def _contextual_measurement_padding_groups(
    column_context: tuple[str | None, ...],
    header: tuple[str, ...],
    feature_indexes: tuple[int, ...],
    dialect: RuntimeMeasurementDialect,
    *,
    known_source_names: tuple[str, ...],
) -> Mapping[int, _ContextualMeasurementPaddingGroup | None]:
    """Return reusable contextual wide-table padding groups by column index."""
    return MappingProxyType(
        {
            index: (
                _contextual_measurement_padding_group(
                    column_context[index],
                    header[index],
                    dialect,
                    known_source_names=known_source_names,
                )
                if index < len(column_context)
                else None
            )
            for index in feature_indexes
        }
    )


def _contextual_measurement_padding_group(
    context: str | None,
    field_name: str,
    dialect: RuntimeMeasurementDialect,
    *,
    known_source_names: tuple[str, ...],
) -> _ContextualMeasurementPaddingGroup | None:
    """Return the contextual wide-table domain implied by a measurement column."""
    if context is None:
        return None
    normalized_context = _normalize_identifier(context)
    if not normalized_context or normalized_context in _CSV_HEADER_CONTEXT_STOPWORDS:
        return None
    if _is_measurement_identity_field(field_name, dialect):
        return None

    normalized_field = _normalize_identifier(field_name)
    parts = tuple(part for part in normalized_field.split("_") if part)
    if not parts:
        return None
    feature_group = _measurement_category_prefix(parts, dialect) or parts[:1]
    _feature_name, source_name = _semantic_core_feature_name_and_source(
        normalized_field,
        known_source_names=known_source_names,
        dialect=dialect,
    )
    return normalized_context, feature_group, source_name


def _measurement_category_prefix(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> tuple[str, ...]:
    """Return the longest dialect category prefix matched by a feature name."""
    matches = tuple(
        prefix
        for prefix in dialect.category_prefixes
        if len(parts) >= len(prefix) and parts[: len(prefix)] == prefix
    )
    if not matches:
        return ()
    return max(matches, key=len)


def _measurement_cell_is_present(value: object) -> bool:
    text = str(value).strip()
    if not text:
        return False
    try:
        numeric = float(text)
    except ValueError:
        return True
    return not math.isnan(numeric)


def _measurement_value_is_present(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_measurement_value_is_present(nested) for nested in value.values())
    return _measurement_cell_is_present(value)


def _runtime_measurement_padding_group(
    table_group: str,
    field_name: str,
    key: RuntimeMeasurementFeatureKey,
    dialect: RuntimeMeasurementDialect,
) -> tuple[RuntimeMeasurementSubjectKey, str | None, tuple[str, ...]]:
    """Return the row-padding family for a runtime measurement field."""
    normalized_field = _normalize_identifier(field_name)
    parts = tuple(part for part in normalized_field.split("_") if part)
    feature_group = _measurement_category_prefix(parts, dialect) or (table_group,)
    return key.subject, key.source_name, feature_group


def _derive_pair_measurement_facts(
    facts: _RuntimeMeasurementFacts,
    policy: RuntimeEquivalencePolicy,
    *,
    known_source_names: tuple[str, ...],
) -> _RuntimeMeasurementFacts:
    """Derive mathematically equivalent facts for directional pair measurements."""
    derived: _RuntimeMeasurementFactList = []
    values_by_key = dict(facts)
    for key, slope_value in facts:
        if key.feature_name != _PAIR_REGRESSION_SLOPE_FEATURE:
            continue
        source_parts = _pair_source_parts_from_feature_key(
            key,
            known_source_names=known_source_names,
        )
        if source_parts is None:
            continue
        correlation_key = RuntimeMeasurementFeatureKey(
            key.subject,
            _PAIR_CORRELATION_FEATURE,
            key.statistic,
            key.source_name,
        )
        correlation_value = values_by_key.get(correlation_key)
        if correlation_value is None:
            continue
        reverse_slope = _reverse_regression_slope(
            correlation_value,
            slope_value,
        )
        if reverse_slope is None:
            continue
        reversed_key = _reversed_pair_feature_key(key, source_parts)
        if reversed_key is None:
            continue
        derived.append(
            (
                reversed_key,
                _cell_signature(repr(reverse_slope), policy),
            )
        )
    if not derived:
        return facts
    return _dedupe_measurement_facts((*facts, *derived))


def _pair_source_parts_from_feature_key(
    key: RuntimeMeasurementFeatureKey,
    *,
    known_source_names: tuple[str, ...],
) -> tuple[str, str] | None:
    source_name = _pair_source_name_from_feature_key(
        key,
        known_source_names=known_source_names,
    )
    if source_name is None:
        return None
    parts = tuple(part for part in source_name.split("__") if part)
    if len(parts) != 2:
        return None
    return (parts[0], parts[1])


def _pair_source_name_from_feature_key(
    key: RuntimeMeasurementFeatureKey,
    *,
    known_source_names: tuple[str, ...],
) -> str | None:
    if key.source_name is not None:
        return key.source_name
    if (
        key.subject.scope is MeasurementScope.IMAGE
        and key.subject.name is not None
    ):
        for source_name in known_source_names:
            normalized_source_name = _normalize_source_name(source_name)
            if normalized_source_name is None:
                continue
            if key.subject.name == _normalize_identifier(normalized_source_name):
                return normalized_source_name
        if "__" in key.subject.name:
            return key.subject.name
    return None


def _reversed_pair_feature_key(
    key: RuntimeMeasurementFeatureKey,
    source_parts: tuple[str, str],
) -> RuntimeMeasurementFeatureKey | None:
    reversed_source_name = "__".join(reversed(source_parts))
    if key.source_name is not None:
        return RuntimeMeasurementFeatureKey(
            key.subject,
            key.feature_name,
            key.statistic,
            reversed_source_name,
        )
    if (
        key.subject.scope is MeasurementScope.IMAGE
        and key.subject.name is not None
    ):
        return RuntimeMeasurementFeatureKey(
            RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, reversed_source_name),
            key.feature_name,
            key.statistic,
        )
    return None


def _reverse_regression_slope(
    correlation_value: RuntimeCellSignature,
    slope_value: RuntimeCellSignature,
) -> float | None:
    correlation = _finite_signature_number(correlation_value)
    slope = _finite_signature_number(slope_value)
    if correlation is None or slope is None or slope == 0:
        return None
    return (correlation * correlation) / slope


def _long_form_measurement_fact(
    context: _LongFormMeasurementContext,
) -> _RuntimeLongFormMeasurementFact:
    feature_name = _first_row_value(context.row, _MEASUREMENT_FEATURE_NAME_FIELDS)
    value = _first_row_value(context.row, _MEASUREMENT_VALUE_FIELDS)
    if feature_name is None or value is None:
        return None
    if _is_aggregate_image_number_reference_measurement_field(str(feature_name)):
        return None
    aggregate_key = _aggregate_measurement_feature_key(
        str(feature_name),
        context.subject,
        context.policy,
        known_source_names=context.known_source_names,
    )
    if aggregate_key is not None:
        return aggregate_key, _cell_signature(str(value), context.policy)
    canonical_feature_name, canonical_source_name = (
        _canonical_measurement_feature_name_and_source(
            str(feature_name),
            context.policy,
            source_name=context.source_name,
            known_source_names=context.known_source_names,
        )
    )
    if not canonical_feature_name:
        return None
    return (
        _runtime_measurement_feature_key(
            context.subject,
            canonical_feature_name,
            source_name=canonical_source_name,
        ),
        _cell_signature(str(value), context.policy),
    )


def _long_form_measurement_fact_cached(
    context: _CachedLongFormMeasurementContext,
) -> _RuntimeLongFormMeasurementFact:
    feature_name = _first_indexed_row_value(
        context.row_values,
        context.feature_indexes,
    )
    value = _first_indexed_row_value(context.row_values, context.value_indexes)
    if feature_name is None or value is None:
        return None
    feature_text = str(feature_name)
    if _is_aggregate_image_number_reference_measurement_field(feature_text):
        return None
    cache_key = (context.subject, context.source_name, feature_text)
    key = context.key_cache.get(cache_key, _CACHE_MISS)
    if key is _CACHE_MISS:
        aggregate_key = _aggregate_measurement_feature_key(
            feature_text,
            context.subject,
            context.policy,
            known_source_names=context.known_source_names,
        )
        if aggregate_key is not None:
            context.key_cache[cache_key] = aggregate_key
            return aggregate_key, _cell_signature(str(value), context.policy)
        canonical_feature_name, canonical_source_name = (
            _canonical_measurement_feature_name_and_source(
                feature_text,
                context.policy,
                source_name=context.source_name,
                known_source_names=context.known_source_names,
            )
        )
        key = (
            _runtime_measurement_feature_key(
                context.subject,
                canonical_feature_name,
                source_name=canonical_source_name,
            )
            if canonical_feature_name
            else None
        )
        context.key_cache[cache_key] = key
    if key is None:
        return None
    return key, _cell_signature(str(value), context.policy)


def _field_indexes_for_names(
    normalized_field_indexes: Mapping[str, int],
    field_names: tuple[str, ...],
) -> tuple[int, ...]:
    return tuple(
        index
        for field_name in field_names
        if (index := normalized_field_indexes.get(field_name)) is not None
    )


def _first_indexed_row_value(
    row_values: tuple[object, ...],
    indexes: tuple[int, ...],
) -> object | None:
    if not indexes:
        return None
    return row_values[indexes[0]]


def _cell_measurement_facts(
    key: RuntimeMeasurementFeatureKey,
    value: object,
    policy: RuntimeEquivalencePolicy,
) -> _RuntimeMeasurementFacts:
    if isinstance(value, Mapping):
        return tuple(
            (
                _runtime_measurement_feature_key(
                    key.subject,
                    f"{key.feature_name}_{_canonical_measurement_feature_name(str(name), policy)}",
                    key.statistic,
                    source_name=key.source_name,
                ),
                _cell_signature(str(nested_value), policy),
            )
            for name, nested_value in value.items()
        )
    return ((key, _cell_signature(str(value), policy)),)


def _measurement_feature_key_for_field(
    field_name: str,
    subject: RuntimeMeasurementSubjectKey,
    policy: RuntimeEquivalencePolicy,
    *,
    qualifiers: tuple[str, ...],
    source_name: str | None,
    known_source_names: tuple[str, ...],
) -> RuntimeMeasurementFeatureKey | None:
    aggregate_key = _aggregate_measurement_feature_key(
        field_name,
        subject,
        policy,
        known_source_names=known_source_names,
    )
    if aggregate_key is not None:
        return aggregate_key
    feature_name, feature_source_name = _canonical_measurement_feature_name_and_source(
        field_name,
        policy,
        source_name=source_name,
        known_source_names=known_source_names,
    )
    if qualifiers:
        qualified_feature_name, feature_source_name = (
            _canonical_measurement_feature_name_and_source(
                "_".join((feature_name, *qualifiers)),
                policy,
                source_name=feature_source_name,
                known_source_names=known_source_names,
            )
        )
        feature_name = qualified_feature_name
    feature_name = _strip_subject_suffix_feature_name(feature_name, subject)
    if not feature_name:
        return None
    return _runtime_measurement_feature_key(
        subject,
        feature_name,
        source_name=feature_source_name,
    )


def _aggregate_measurement_feature_key(
    field_name: str,
    subject: RuntimeMeasurementSubjectKey,
    policy: RuntimeEquivalencePolicy,
    *,
    known_source_names: tuple[str, ...],
) -> RuntimeMeasurementFeatureKey | None:
    scope_policy = _MEASUREMENT_SCOPE_AGGREGATE_POLICY_BY_SCOPE[subject.scope]
    if not scope_policy.accepts_aggregate_feature_key:
        return None
    parts = tuple(part for part in _normalize_identifier(field_name).split("_") if part)
    if len(parts) >= 2 and parts[0] == "count":
        return _runtime_measurement_feature_key(
            RuntimeMeasurementSubjectKey(
                MeasurementScope.OBJECT,
                "_".join(parts[1:]),
            ),
            "object_count",
            "count",
        )
    if len(parts) < 3 or parts[0] not in _MEASUREMENT_AGGREGATE_PREFIXES:
        return None
    object_name_parts, feature_parts = _aggregate_object_and_feature_parts(
        parts[1:],
        policy.measurement_dialect,
    )
    if not object_name_parts or not feature_parts:
        return None
    feature_name, source_name = _canonical_measurement_feature_name_and_source(
        "_".join(feature_parts),
        policy,
        source_name=None,
        known_source_names=known_source_names,
    )
    return _runtime_measurement_feature_key(
        RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            "_".join(object_name_parts),
        ),
        feature_name,
        parts[0],
        source_name,
    )


def _aggregate_object_and_feature_parts(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> _RuntimeMeasurementNameParts:
    for index in range(1, len(parts)):
        if _starts_aggregate_feature_parts(parts[index:], dialect):
            return parts[:index], parts[index:]
    return parts[:1], parts[1:]


def _starts_aggregate_feature_parts(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> bool:
    return (
        _starts_with_measurement_category(parts, dialect)
        or parts in dialect.feature_part_aliases
    )


def _starts_with_measurement_category(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> bool:
    return any(
        len(parts) >= len(prefix) and parts[: len(prefix)] == prefix
        for prefix in dialect.category_prefixes
    )


def _required_mean_measurement_keys(
    required_keys: _RuntimeRequiredMeasurementKeys,
) -> _RuntimeRequiredMeasurementKeys:
    if required_keys is None:
        return None
    return frozenset(key for key in required_keys if key.statistic == "mean")


def _object_label_measurement_facts(
    context: _ObjectLabelMeasurementContext,
) -> _RuntimeMeasurementFacts:
    if not _object_label_measurements_required(
        context.object_name,
        context.required_keys,
    ):
        return ()
    return (
        *_object_count_measurement_facts(
            context,
        ),
        *_object_identifier_measurement_facts(
            context,
        ),
        *_object_location_measurement_facts(
            context,
        ),
    )


def _object_label_measurements_required(
    object_name: str | None,
    required_keys: _RuntimeRequiredMeasurementKeys,
) -> bool:
    if required_keys is None:
        return True
    if object_name is None:
        return False
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)
    return any(key.subject == subject for key in required_keys)


def _runtime_object_label_array(labels: object) -> np.ndarray | None:
    try:
        label_array = np.asarray(labels)
    except (TypeError, ValueError):
        return None
    if label_array.ndim == 0:
        return None
    return label_array


def _object_identifier_measurement_facts(
    context: _ObjectLabelMeasurementContext,
) -> _RuntimeMeasurementFacts:
    if context.object_name is None:
        return ()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, context.object_name)
    if subject in context.object_identifier_subjects:
        return ()
    keys = _required_object_identifier_keys(subject, context.required_keys)
    if not keys:
        return ()
    label_array = _runtime_object_label_array(context.labels)
    if label_array is None:
        return ()

    planes = (label_array,) if label_array.ndim <= 2 else tuple(label_array)
    facts: _RuntimeMeasurementFactList = []
    for plane in planes:
        object_ids = dense_object_label_id_domain(
            plane,
            declared_object_count=context.declared_object_count,
            declared_object_ids=context.declared_object_ids,
        )
        for key in keys:
            facts.extend(
                (key, _cell_signature(str(object_id), context.policy))
                for object_id in object_ids
            )
    return tuple(facts)


def _required_object_identifier_keys(
    subject: RuntimeMeasurementSubjectKey,
    required_keys: _RuntimeRequiredMeasurementKeys,
) -> tuple[RuntimeMeasurementFeatureKey, ...]:
    if required_keys is None:
        return (RuntimeMeasurementFeatureKey(subject, "object_number"),)
    return tuple(
        key
        for key in sorted(required_keys, key=lambda item: item.sort_key)
        if key.subject == subject
        and key.statistic == "value"
        and key.source_name is None
        and _is_object_identifier_feature(key.feature_name)
    )


def _object_count_measurement_facts(
    context: _ObjectLabelMeasurementContext,
) -> _RuntimeMeasurementFacts:
    if context.object_name is None:
        return ()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, context.object_name)
    key = RuntimeMeasurementFeatureKey(subject, "object_count", "count")
    if context.required_keys is not None and key not in context.required_keys:
        return ()
    label_array = _runtime_object_label_array(context.labels)
    if label_array is None:
        return ()
    planes = (label_array,) if label_array.ndim <= 2 else tuple(label_array)
    return tuple(
        (
            key,
            _cell_signature(
                str(
                    len(
                        dense_object_label_id_domain(
                            plane,
                            declared_object_count=context.declared_object_count,
                            declared_object_ids=context.declared_object_ids,
                        )
                    )
                ),
                context.policy,
            ),
        )
        for plane in planes
    )


def _object_location_measurement_facts(
    context: _ObjectLabelMeasurementContext,
) -> _RuntimeMeasurementFacts:
    if context.object_name is None:
        return ()
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, context.object_name)
    if subject in context.object_location_subjects:
        return ()
    required_feature_names = _required_object_location_feature_names(
        subject,
        context.required_keys,
        statistic="value",
    )
    required_mean_feature_names = _required_object_location_feature_names(
        subject,
        context.required_keys,
        statistic="mean",
    )
    if (
        required_feature_names is not None
        and not required_feature_names
        and required_mean_feature_names is not None
        and not required_mean_feature_names
    ):
        return ()
    label_array = _runtime_object_label_array(context.labels)
    if label_array is None:
        return ()
    object_ids = dense_object_label_id_domain(context.labels)
    if not object_ids:
        return ()

    include_missing_locations = context.declared_object_count is not None
    facts: _RuntimeMeasurementFactList = []
    planes = (label_array,) if label_array.ndim <= 2 else tuple(label_array)
    for plane in planes:
        facts.extend(
            _object_location_measurement_facts_for_plane(
                plane,
                subject,
                context.policy,
                required_feature_names=required_feature_names,
                required_mean_feature_names=required_mean_feature_names,
                object_ids=object_ids,
                include_missing=include_missing_locations,
            )
        )
    return tuple(facts)


def _required_object_location_feature_names(
    subject: RuntimeMeasurementSubjectKey,
    required_keys: _RuntimeRequiredMeasurementKeys,
    *,
    statistic: str,
) -> frozenset[str] | None:
    if required_keys is None:
        return None
    return frozenset(
        key.feature_name
        for key in required_keys
        if key.subject == subject
        and key.statistic == statistic
        and key.source_name is None
        and key.feature_name in _OBJECT_LOCATION_FEATURES
    )


def _object_location_measurement_facts_for_plane(
    labels: np.ndarray,
    subject: RuntimeMeasurementSubjectKey,
    policy: RuntimeEquivalencePolicy,
    *,
    required_feature_names: frozenset[str] | None = None,
    required_mean_feature_names: frozenset[str] | None = None,
    object_ids: tuple[int, ...] | None = None,
    include_missing: bool = True,
) -> _RuntimeMeasurementFacts:
    if (
        required_feature_names is not None
        and not required_feature_names
        and required_mean_feature_names is not None
        and not required_mean_feature_names
    ):
        return ()
    if labels.size == 0:
        return ()
    integer_labels = np.asarray(labels)
    resolved_object_ids = (
        object_ids
        if object_ids is not None
        else dense_object_label_id_domain(integer_labels)
    )
    if not resolved_object_ids:
        return ()

    max_object_id = max(resolved_object_ids)
    if max_object_id <= 0:
        return ()
    object_id_indexes = np.asarray(resolved_object_ids, dtype=np.intp)

    flat_labels = integer_labels.ravel()
    valid = (flat_labels > 0) & (flat_labels <= max_object_id)
    valid_labels = flat_labels[valid].astype(np.intp, copy=False)
    counts = np.bincount(valid_labels, minlength=max_object_id + 1).astype(float)
    axis_centers: list[np.ndarray] = []
    for coordinates in np.indices(integer_labels.shape, sparse=False):
        sums = np.bincount(
            valid_labels,
            weights=coordinates.ravel()[valid],
            minlength=max_object_id + 1,
        )
        centers = np.full(max_object_id + 1, np.nan, dtype=float)
        np.divide(sums, counts, out=centers, where=counts > 0)
        axis_centers.append(centers)

    center_x = axis_centers[-1][object_id_indexes]
    center_y = (
        axis_centers[-2][object_id_indexes]
        if len(axis_centers) >= 2
        else np.zeros(len(resolved_object_ids))
    )
    center_z = (
        axis_centers[-3][object_id_indexes]
        if len(axis_centers) >= 3
        else np.zeros(len(resolved_object_ids))
    )
    raw_facts = tuple(
        fact
        for feature_name, values in (
            ("center_x", center_x),
            ("center_y", center_y),
            ("center_z", center_z),
        )
        if required_feature_names is None or feature_name in required_feature_names
        for fact in _object_location_feature_facts(
            subject,
            feature_name,
            values,
            policy,
            include_missing=include_missing,
        )
    )
    mean_facts = tuple(
        fact
        for feature_name, values in (
            ("center_x", center_x),
            ("center_y", center_y),
            ("center_z", center_z),
        )
        if required_mean_feature_names is None
        or feature_name in required_mean_feature_names
        for fact in _object_location_mean_feature_fact(
            subject,
            feature_name,
            values,
            policy,
        )
    )
    return (*raw_facts, *mean_facts)


def _object_location_feature_facts(
    subject: RuntimeMeasurementSubjectKey,
    feature_name: str,
    values: np.ndarray,
    policy: RuntimeEquivalencePolicy,
    *,
    include_missing: bool = True,
) -> _RuntimeMeasurementFacts:
    key = RuntimeMeasurementFeatureKey(subject, feature_name)
    return tuple(
        (key, _cell_signature(str(value), policy))
        for value in values
        if include_missing or np.isfinite(value)
    )


def _object_location_mean_feature_fact(
    subject: RuntimeMeasurementSubjectKey,
    feature_name: str,
    values: np.ndarray,
    policy: RuntimeEquivalencePolicy,
) -> _RuntimeMeasurementFacts:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return ()
    key = RuntimeMeasurementFeatureKey(subject, feature_name, "mean")
    return ((key, _cell_signature(str(float(np.mean(finite_values))), policy)),)


def _positive_label_count(labels: np.ndarray) -> int:
    values = np.unique(np.asarray(labels))
    return int(np.count_nonzero(values > 0))


def _object_label_counts_by_name(
    records: Iterable[object],
) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        object_labels = ObjectLabelSet.from_runtime_value(record.value)
        counts[object_labels.name] = max(
            counts.get(object_labels.name, 0),
            _max_positive_label_id(object_labels.labels),
        )
    return counts


def _object_label_measurement_values_for_name(
    records: Iterable[object],
    object_name: str,
    *,
    required_keys: _RuntimeRequiredMeasurementKeys = None,
) -> _RuntimeObjectValuesByLabel:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)
    if subject.name is None:
        return {}
    required_feature_names = _required_object_location_feature_names(
        subject,
        required_keys,
        statistic="value",
    )
    if required_feature_names is not None and not required_feature_names:
        return {}

    values_by_feature: _RuntimeObjectValuesByLabel = {}
    for record in records:
        object_labels = ObjectLabelSet.from_runtime_value(record.value)
        if _normalize_identifier(object_labels.name) != subject.name:
            continue
        for key, values in _object_label_location_values_by_label(
            object_labels.labels,
            subject,
            required_feature_names=required_feature_names,
        ).items():
            values_by_feature.setdefault(key, {}).update(values)
    return values_by_feature


def _object_label_location_values_by_label(
    labels: object,
    subject: RuntimeMeasurementSubjectKey,
    *,
    required_feature_names: frozenset[str] | None = None,
) -> _RuntimeObjectValuesByLabel:
    if required_feature_names is not None and not required_feature_names:
        return {}
    label_array = _runtime_object_label_array(labels)
    if label_array is None:
        return {}

    object_ids = dense_object_label_id_domain(labels)
    if not object_ids:
        return {}

    values_by_feature: _RuntimeObjectValuesByLabel = {}
    planes = (label_array,) if label_array.ndim <= 2 else tuple(label_array)
    for plane in planes:
        for key, values in _object_label_location_values_by_label_for_plane(
            plane,
            subject,
            required_feature_names=required_feature_names,
            object_ids=object_ids,
        ).items():
            values_by_feature.setdefault(key, {}).update(values)
    return values_by_feature


def _object_label_location_values_by_label_for_plane(
    labels: np.ndarray,
    subject: RuntimeMeasurementSubjectKey,
    *,
    required_feature_names: frozenset[str] | None = None,
    object_ids: tuple[int, ...] | None = None,
) -> _RuntimeObjectValuesByLabel:
    if required_feature_names is not None and not required_feature_names:
        return {}
    if labels.size == 0:
        return {}
    integer_labels = np.asarray(labels)
    resolved_object_ids = (
        object_ids
        if object_ids is not None
        else dense_object_label_id_domain(integer_labels)
    )
    if not resolved_object_ids:
        return {}

    max_object_id = max(resolved_object_ids)
    if max_object_id <= 0:
        return {}

    flat_labels = integer_labels.ravel()
    valid = (flat_labels > 0) & (flat_labels <= max_object_id)
    valid_labels = flat_labels[valid].astype(np.intp, copy=False)
    counts = np.bincount(valid_labels, minlength=max_object_id + 1).astype(float)
    axis_centers: list[np.ndarray] = []
    for coordinates in np.indices(integer_labels.shape, sparse=False):
        sums = np.bincount(
            valid_labels,
            weights=coordinates.ravel()[valid],
            minlength=max_object_id + 1,
        )
        centers = np.full(max_object_id + 1, np.nan, dtype=float)
        np.divide(sums, counts, out=centers, where=counts > 0)
        axis_centers.append(centers)

    center_x = axis_centers[-1]
    center_y = (
        axis_centers[-2] if len(axis_centers) >= 2 else np.zeros(max_object_id + 1)
    )
    center_z = (
        axis_centers[-3] if len(axis_centers) >= 3 else np.zeros(max_object_id + 1)
    )
    return {
        RuntimeMeasurementFeatureKey(subject, feature_name): {
            label: float(values[label])
            for label in resolved_object_ids
        }
        for feature_name, values in (
            ("center_x", center_x),
            ("center_y", center_y),
            ("center_z", center_z),
        )
        if required_feature_names is None or feature_name in required_feature_names
    }


def _max_positive_label_id(labels: object) -> int:
    values = np.asarray(labels)
    if values.size == 0:
        return 0
    positive = values[values > 0]
    if positive.size == 0:
        return 0
    return int(np.max(positive))


def _relationship_measurement_facts(
    relationship: ObjectRelationship,
    policy: RuntimeEquivalencePolicy,
    *,
    object_label_counts: Mapping[str, int] | None = None,
) -> _RuntimeMeasurementFacts:
    source_ids = tuple(int(value) for value in np.asarray(relationship.source_ids).ravel())
    target_ids = tuple(int(value) for value in np.asarray(relationship.target_ids).ravel())
    if len(source_ids) != len(target_ids):
        raise ValueError(
            f"Relationship '{relationship.name}' has {len(source_ids)} source ids "
            f"and {len(target_ids)} target ids."
        )

    source_subject = RuntimeMeasurementSubjectKey(
        MeasurementScope.OBJECT,
        relationship.source.name,
    )
    target_subject = RuntimeMeasurementSubjectKey(
        MeasurementScope.OBJECT,
        relationship.target.name,
    )
    child_count_key = RuntimeMeasurementFeatureKey(
        source_subject,
        f"{_normalize_identifier(relationship.target.name)}_count",
    )
    parent_key = RuntimeMeasurementFeatureKey(
        target_subject,
        _normalize_identifier(relationship.source.name),
    )

    object_label_counts = object_label_counts or {}
    source_count = max(
        max(source_ids, default=0),
        object_label_counts.get(relationship.source.name, 0),
    )
    target_count = max(
        max(target_ids, default=0),
        object_label_counts.get(relationship.target.name, 0),
    )
    counts = {source_id: 0 for source_id in range(1, source_count + 1)}
    parent_by_target: dict[int, int] = {}
    for source_id, target_id in zip(source_ids, target_ids, strict=True):
        if source_id > 0:
            counts[source_id] = counts.get(source_id, 0) + 1
        if target_id > 0:
            parent_by_target[target_id] = source_id

    return (
        *(
            (child_count_key, _cell_signature(str(count), policy))
            for _, count in sorted(counts.items())
        ),
        *(
            (
                parent_key,
                _cell_signature(str(parent_by_target.get(target_id, 0)), policy),
            )
            for target_id in range(1, target_count + 1)
        ),
    )


def _relationship_required_child_measurement_keys(
    relationship: ObjectRelationship,
    required_measurement_keys: _RuntimeRequiredMeasurementKeys,
) -> _RuntimeRequiredMeasurementKeys:
    """Return child measurements needed to synthesize required relationship aggregates."""
    if required_measurement_keys is None:
        return None

    parent_subject = RuntimeMeasurementSubjectKey(
        MeasurementScope.OBJECT,
        relationship.source.name,
    )
    child_subject = RuntimeMeasurementSubjectKey(
        MeasurementScope.OBJECT,
        relationship.target.name,
    )
    aggregate_prefix = (
        f"mean_{_normalize_identifier(relationship.target.name)}_"
    )
    return frozenset(
        _runtime_measurement_feature_key(
            child_subject,
            key.feature_name.removeprefix(aggregate_prefix),
            source_name=key.source_name,
        )
        for key in required_measurement_keys
        if key.subject == parent_subject
        and key.statistic == "value"
        and key.feature_name.startswith(aggregate_prefix)
        and key.feature_name != aggregate_prefix
    )


def _relationship_aggregate_measurement_facts(
    relationship: ObjectRelationship,
    child_values_by_feature: Mapping[
        RuntimeMeasurementFeatureKey,
        Mapping[int, float],
    ],
    policy: RuntimeEquivalencePolicy,
    *,
    object_label_counts: Mapping[str, int] | None = None,
    existing_measurement_keys: _RuntimeMeasurementKeySet = frozenset(),
    required_measurement_keys: _RuntimeRequiredMeasurementKeys = None,
) -> _RuntimeMeasurementFacts:
    child_values = {
        key: dict(values_by_child_id)
        for key, values_by_child_id in child_values_by_feature.items()
    }
    child_values.setdefault(
        _runtime_measurement_feature_key(
            RuntimeMeasurementSubjectKey(
                MeasurementScope.OBJECT,
                relationship.target.name,
            ),
            "object_number",
        ),
        _relationship_target_object_number_values(
            relationship,
            object_label_counts=object_label_counts,
        ),
    )
    if not child_values:
        return ()

    child_ids_by_parent = _relationship_child_ids_by_parent(
        relationship,
        object_label_counts=object_label_counts,
    )
    parent_subject = RuntimeMeasurementSubjectKey(
        MeasurementScope.OBJECT,
        relationship.source.name,
    )
    aggregate_facts: _RuntimeMeasurementFactList = []
    for child_key, values_by_child_id in child_values.items():
        if child_key.subject.scope is not MeasurementScope.OBJECT:
            continue
        if child_key.subject != RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            relationship.target.name,
        ):
            continue
        aggregate_key = _runtime_measurement_feature_key(
            parent_subject,
            _relationship_aggregate_feature_name(
                "mean",
                relationship.target.name,
                child_key.feature_name,
            ),
            source_name=child_key.source_name,
        )
        if aggregate_key in existing_measurement_keys:
            continue
        if required_measurement_keys is not None and aggregate_key not in required_measurement_keys:
            continue
        aggregate_values_by_parent = _mean_child_values_by_parent(
            child_ids_by_parent,
            values_by_child_id,
        )
        for _parent_id, child_ids in sorted(child_ids_by_parent.items()):
            aggregate_facts.append(
                (
                    aggregate_key,
                    _cell_signature(
                        str(aggregate_values_by_parent[child_ids]),
                        policy,
                    ),
                )
            )
    return tuple(aggregate_facts)


def _object_measurement_values_by_label(
    measurement_tables: tuple[MeasurementTable, ...],
    object_name: str,
    policy: RuntimeEquivalencePolicy,
    *,
    known_source_names: tuple[str, ...],
    required_keys: _RuntimeRequiredMeasurementKeys = None,
) -> _RuntimeObjectValuesByLabel:
    object_subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)
    values_by_feature: _RuntimeObjectValuesByLabel = {}
    row_required_keys = _required_measurement_input_keys(
        required_keys,
        known_source_names=known_source_names,
    )
    schema_cache: _RuntimeMeasurementRowSchemaCache = {}
    key_cache: _RuntimeMeasurementFeatureKeyCache = {}
    long_form_key_cache: _RuntimeMeasurementLongFormKeyCache = {}
    qualifier_render_cache: _RuntimeMeasurementQualifierRenderCache = {}
    padding_group_cache: _RuntimeMeasurementPaddingGroupCache = {}
    for table in measurement_tables:
        table_subject = RuntimeMeasurementSubjectKey.from_subject(table.subject)
        table_object_subject = (
            table_subject if table_subject.scope is MeasurementScope.OBJECT else None
        )
        if table_object_subject is not None and table_object_subject != object_subject:
            continue
        object_id_field = measurement_table_object_id_field(table)
        table_source_name = table.source_image_name
        table_padding_group = _normalize_identifier(table.name) or "measurements"
        for row in measurement_rows((table,)):
            row_mapping = measurement_row_mapping(row)
            try:
                object_label = measurement_object_label(
                    row_mapping,
                    object_id_field=object_id_field,
                )
            except (TypeError, ValueError):
                continue
            if object_label is None:
                continue
            if table_object_subject is None:
                row_object_name = measurement_row_object_name(row_mapping)
                if row_object_name is None:
                    continue
                subject = RuntimeMeasurementSubjectKey(
                    MeasurementScope.OBJECT,
                    row_object_name,
                )
                if subject != object_subject:
                    continue
            else:
                subject = table_object_subject
            row_context = _RuntimeRowProjectionContext.from_row(
                row_mapping,
                subject,
                policy,
                source_name=measurement_row_source_image_name(row_mapping)
                or table_source_name,
                known_source_names=known_source_names,
                required_keys=row_required_keys,
                table_padding_group=table_padding_group,
                schema_cache=schema_cache,
                key_cache=key_cache,
                long_form_key_cache=long_form_key_cache,
                qualifier_render_cache=qualifier_render_cache,
                padding_group_cache=padding_group_cache,
            )
            for key, value in _numeric_measurement_values_from_runtime_row_cached(
                row_context
            ):
                if row_required_keys is not None and key not in row_required_keys:
                    continue
                if key.statistic != "value":
                    continue
                values_by_feature.setdefault(key, {})[object_label] = value
    return values_by_feature


def _numeric_measurement_values_from_runtime_row_cached(
    context: _RuntimeRowProjectionContext,
) -> _RuntimeNumericMeasurementValues:
    """Project numeric runtime row values without building cell signatures."""
    projection = _runtime_row_projection_cached(
        context,
        value_projector=_numeric_cell_measurement_values,
        long_form_projector=_numeric_long_form_measurement_values,
    )
    row_values_by_key = _dedupe_numeric_measurement_values(
        (key, value)
        for _padding_group, key, value in projection.records
    )
    if projection.long_form:
        return row_values_by_key
    if not any(
        key.feature_name == _PAIR_REGRESSION_SLOPE_FEATURE
        for key, _value in row_values_by_key
    ):
        return row_values_by_key

    derived_facts = _derive_pair_measurement_facts(
        tuple(
            (key, _cell_signature(repr(value), context.policy))
            for key, value in row_values_by_key
        ),
        context.policy,
        known_source_names=context.known_source_names,
    )
    if context.required_keys is not None:
        derived_facts = tuple(
            (key, value)
            for key, value in derived_facts
            if key in context.required_keys
        )
    return tuple(
        (key, numeric_value)
        for key, value in derived_facts
        if (numeric_value := _cell_signature_numeric_value(value)) is not None
    )


def _numeric_long_form_measurement_values(
    fact: _RuntimeMeasurementFact,
) -> _RuntimeNumericMeasurementValues:
    key, cell_value = fact
    numeric_value = _cell_signature_numeric_value(cell_value)
    if numeric_value is None:
        return ()
    return ((key, numeric_value),)


def _numeric_cell_measurement_values(
    key: RuntimeMeasurementFeatureKey,
    value: object,
    policy: RuntimeEquivalencePolicy,
) -> _RuntimeNumericMeasurementValues:
    if isinstance(value, Mapping):
        return tuple(
            (
                _runtime_measurement_feature_key(
                    key.subject,
                    f"{key.feature_name}_{_canonical_measurement_feature_name(str(name), policy)}",
                    key.statistic,
                    source_name=key.source_name,
                ),
                numeric_value,
            )
            for name, nested_value in value.items()
            if (
                numeric_value := _measurement_numeric_runtime_value(
                    nested_value,
                    policy,
                )
            )
            is not None
        )
    numeric_value = _measurement_numeric_runtime_value(value, policy)
    if numeric_value is None:
        return ()
    return ((key, numeric_value),)


def _measurement_numeric_runtime_value(
    value: object,
    policy: RuntimeEquivalencePolicy,
) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric_value = float(text)
    except ValueError:
        return None
    if math.isnan(numeric_value):
        return None
    if math.isfinite(numeric_value):
        return float(round(numeric_value, policy.numeric_decimal_places))
    return numeric_value


def _cell_signature_numeric_value(value: RuntimeCellSignature) -> float | None:
    if value.kind is not RuntimeCellValueKind.NUMBER:
        return None
    numeric_value = float(value.value)
    if math.isnan(numeric_value):
        return None
    return numeric_value


def _dedupe_numeric_measurement_values(
    values: Iterable[tuple[RuntimeMeasurementFeatureKey, float]],
) -> _RuntimeNumericMeasurementValues:
    values_by_key: dict[RuntimeMeasurementFeatureKey, float] = {}
    for key, value in values:
        values_by_key.setdefault(key, value)
    return tuple(values_by_key.items())


def _relationship_target_object_number_values(
    relationship: ObjectRelationship,
    *,
    object_label_counts: Mapping[str, int] | None,
) -> dict[int, float]:
    target_ids = tuple(int(value) for value in np.asarray(relationship.target_ids).ravel())
    target_count = max(
        max(target_ids, default=0),
        (object_label_counts or {}).get(relationship.target.name, 0),
    )
    return {target_id: float(target_id) for target_id in range(1, target_count + 1)}


def _relationship_child_ids_by_parent(
    relationship: ObjectRelationship,
    *,
    object_label_counts: Mapping[str, int] | None,
) -> dict[int, tuple[int, ...]]:
    source_ids = tuple(int(value) for value in np.asarray(relationship.source_ids).ravel())
    target_ids = tuple(int(value) for value in np.asarray(relationship.target_ids).ravel())
    source_count = max(
        max(source_ids, default=0),
        (object_label_counts or {}).get(relationship.source.name, 0),
    )
    children: dict[int, list[int]] = {
        source_id: [] for source_id in range(1, source_count + 1)
    }
    for source_id, target_id in zip(source_ids, target_ids, strict=True):
        if source_id > 0 and target_id > 0:
            children.setdefault(source_id, []).append(target_id)
    return {
        source_id: sorted_tuple(child_ids)
        for source_id, child_ids in sorted(children.items())
    }


def _relationship_aggregate_feature_name(
    aggregate: str,
    child_object_name: str,
    child_feature_name: str,
) -> str:
    return "_".join(
        part
        for part in (
            _normalize_identifier(aggregate),
            _normalize_identifier(child_object_name),
            child_feature_name,
        )
        if part
    )


def _mean_child_value(
    child_ids: tuple[int, ...],
    values_by_child_id: Mapping[int, float],
) -> float:
    values = tuple(values_by_child_id[child_id] for child_id in child_ids if child_id in values_by_child_id)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _mean_child_values_by_parent(
    child_ids_by_parent: Mapping[int, tuple[int, ...]],
    values_by_child_id: Mapping[int, float],
) -> dict[tuple[int, ...], float]:
    means: dict[tuple[int, ...], float] = {}
    for child_ids in child_ids_by_parent.values():
        if child_ids in means:
            continue
        means[child_ids] = _mean_child_value(child_ids, values_by_child_id)
    return means


def _measurement_source_names_from_artifact_execution(
    observation: RuntimeArtifactExecutionObservation,
) -> tuple[str, ...]:
    """Return source-image names carried by typed runtime artifacts."""
    source_names: set[str] = set()
    for records in observation.records_by_axis.values():
        for record in records:
            schema_source_name = record.value.schema.source_image_name
            if schema_source_name is not None:
                source_names.update(_source_name_aliases(str(schema_source_name)))
            if record.key.kind is not ArtifactKind.MEASUREMENTS:
                continue
            table = MeasurementTable.from_runtime_value(record.value)
            if table.source_image_name is not None:
                source_names.update(_source_name_aliases(table.source_image_name))
            for row in measurement_rows((table,)):
                row_source_name = measurement_row_source_image_name(
                    measurement_row_mapping(row)
                )
                if row_source_name is not None:
                    source_names.update(_source_name_aliases(row_source_name))
    return sorted_tuple(source_names, key=_normalize_identifier)


def _source_name_aliases(source_name: str) -> tuple[str, ...]:
    names = tuple(part for part in str(source_name).split("__") if part)
    if len(names) <= 1:
        return names
    return (source_name, *names)


def _runtime_measurement_row_subject_schema(
    header: tuple[str, ...],
) -> _RuntimeMeasurementRowSubjectSchema:
    normalized_fields = tuple(_normalize_identifier(field) for field in header)
    normalized_field_indexes = {
        field_name: index
        for index, field_name in enumerate(normalized_fields)
    }
    return (
        normalized_field_indexes.get(MEASUREMENT_OBJECT_NAME_FIELD),
        normalized_field_indexes.get(MEASUREMENT_SOURCE_IMAGE_NAME_FIELD),
        _field_indexes_for_names(
            normalized_field_indexes,
            MEASUREMENT_OBJECT_ID_FIELDS,
        ),
        _field_indexes_for_names(
            normalized_field_indexes,
            sorted_tuple(_IMAGE_IDENTITY_FIELDS),
        ),
    )


def _measurement_source_name_from_runtime_row_values(
    table_source_name: str | None,
    row_values: tuple[object, ...],
    subject_schema: _RuntimeMeasurementRowSubjectSchema,
) -> str | None:
    row_source_name = _indexed_row_text(row_values, subject_schema[1])
    if row_source_name is not None:
        return row_source_name
    return table_source_name


def _measurement_subject_from_runtime_row_values(
    table_subject: RuntimeMeasurementSubjectKey,
    row_values: tuple[object, ...],
    subject_schema: _RuntimeMeasurementRowSubjectSchema,
) -> RuntimeMeasurementSubjectKey:
    object_name_index, source_name_index, object_identity_indexes, image_identity_indexes = (
        subject_schema
    )
    object_name = _indexed_row_text(row_values, object_name_index)
    has_object_identity = _indexed_row_has_value(row_values, object_identity_indexes)
    if object_name is not None and has_object_identity:
        return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)
    row_source_name = _indexed_row_text(row_values, source_name_index)
    if row_source_name is not None:
        return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, row_source_name)
    if _indexed_row_has_value(row_values, image_identity_indexes) and not has_object_identity:
        return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")
    return table_subject


def _indexed_row_text(
    row_values: tuple[object, ...],
    index: int | None,
) -> str | None:
    if index is None:
        return None
    value = row_values[index]
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _indexed_row_has_value(
    row_values: tuple[object, ...],
    indexes: tuple[int, ...],
) -> bool:
    for index in indexes:
        value = row_values[index]
        if value is None:
            continue
        if str(value).strip():
            return True
    return False


def _measurement_subject_from_export_row(
    path: Path,
    row: Mapping[str, object],
) -> RuntimeMeasurementSubjectKey:
    object_name = measurement_row_object_name(row)
    if object_name is not None:
        return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)

    table_name = path.stem
    normalized_table_name = _normalize_identifier(table_name)
    if _row_has_object_identity(row) and normalized_table_name != "image":
        return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, table_name)
    if normalized_table_name == "experiment":
        return RuntimeMeasurementSubjectKey(MeasurementScope.EXPERIMENT, None)
    return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")


def _measurement_subject_from_export_column(
    table: RuntimeTableSnapshot,
    row: Mapping[str, object],
    index: int,
    *,
    fallback_subject: RuntimeMeasurementSubjectKey,
) -> RuntimeMeasurementSubjectKey:
    """Return a column-scoped subject when exported tables carry one."""
    if not table.column_context or index >= len(table.column_context):
        return fallback_subject

    field_name = table.header[index]
    if _is_measurement_identity_field(field_name, DEFAULT_RUNTIME_MEASUREMENT_DIALECT):
        return fallback_subject

    context, normalized_context = _export_column_subject_context(table, index)
    row_object_name = measurement_row_object_name(row)
    normalized_row_object_name = (
        _normalize_identifier(row_object_name) if row_object_name is not None else None
    )
    return _measurement_subject_from_column_context(
        context,
        normalized_context,
        row_object_name,
        normalized_row_object_name,
        fallback_subject=fallback_subject,
    )


def _export_column_subject_context(
    table: RuntimeTableSnapshot,
    index: int,
) -> tuple[str | None, str | None]:
    """Return raw and normalized context for a contextual measurement column."""
    if not table.column_context or index >= len(table.column_context):
        return None, None
    context = table.column_context[index]
    if context is None:
        return None, None
    normalized_context = _normalize_identifier(context)
    if not normalized_context:
        return None, None
    return context, normalized_context


def _measurement_subject_from_column_context(
    context: str | None,
    normalized_context: str | None,
    row_object_name: str | None,
    normalized_row_object_name: str | None,
    *,
    fallback_subject: RuntimeMeasurementSubjectKey,
) -> RuntimeMeasurementSubjectKey:
    """Return a subject implied by a contextual wide-table column."""
    if context is None or normalized_context is None:
        return fallback_subject
    if normalized_context in _CSV_HEADER_CONTEXT_STOPWORDS:
        if normalized_context == "image":
            return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")
        return fallback_subject

    if (
        row_object_name is not None
        and normalized_row_object_name == normalized_context
    ):
        return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, row_object_name)
    return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, context)


def _row_has_image_identity(row: Mapping[str, object]) -> bool:
    return _row_has_identity_value(row, _IMAGE_IDENTITY_FIELDS)


def _row_has_object_identity(row: Mapping[str, object]) -> bool:
    return _row_has_identity_value(row, _OBJECT_IDENTITY_FIELDS)


def _row_has_identity_value(
    row: Mapping[str, object],
    field_names: frozenset[str],
) -> bool:
    normalized_fields = {_normalize_identifier(field): field for field in row}
    for field_name in field_names:
        field = normalized_fields.get(field_name)
        if field is None:
            continue
        value = row[field]
        if value is None:
            continue
        if str(value).strip():
            return True
    return False


def _first_row_value(
    row: Mapping[str, object],
    field_names: tuple[str, ...],
) -> object | None:
    normalized_fields = {_normalize_identifier(field): field for field in row}
    for field_name in field_names:
        field = normalized_fields.get(field_name)
        if field is not None:
            return row[field]
    return None


def _canonical_measurement_feature_name(
    feature_name: str,
    policy: RuntimeEquivalencePolicy,
) -> str:
    return _canonical_measurement_feature_name_and_source(
        feature_name,
        policy,
        source_name=None,
        known_source_names=(),
    )[0]


def _canonical_measurement_feature_name_and_source(
    feature_name: str,
    policy: RuntimeEquivalencePolicy,
    *,
    source_name: str | None,
    known_source_names: tuple[str, ...],
) -> tuple[str, str | None]:
    normalized = _normalize_identifier(feature_name)
    normalized_source_name = _normalize_source_name(source_name)
    if (
        policy.measurement_feature_name_mode
        is RuntimeMeasurementFeatureNameMode.FULL
    ):
        return normalized, normalized_source_name
    core_feature_name, field_source_name = _semantic_core_feature_name_and_source(
        normalized,
        known_source_names=known_source_names,
        dialect=policy.measurement_dialect,
    )
    return _directional_pair_feature_name_and_source(
        core_feature_name,
        field_source_name or normalized_source_name,
        policy.measurement_dialect,
    )


def _semantic_core_feature_name(feature_name: str) -> str:
    return _semantic_core_feature_name_and_source(
        feature_name,
        known_source_names=(),
        dialect=DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    )[0]


def _semantic_core_feature_name_and_source(
    feature_name: str,
    *,
    known_source_names: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> tuple[str, str | None]:
    parts = tuple(part for part in feature_name.split("_") if part)
    aggregate_feature = _aggregate_prefixed_feature_name_and_source(
        parts,
        known_source_names=known_source_names,
        dialect=dialect,
    )
    if aggregate_feature is not None:
        return aggregate_feature
    for prefix in dialect.category_prefixes:
        if _should_strip_measurement_category_prefix(parts, prefix):
            parts = parts[len(prefix) :]
            break

    direct_alias = dialect.feature_part_aliases.get(parts)
    if direct_alias is not None:
        return "_".join(direct_alias), None

    source_feature = _source_feature_name_and_source(parts, dialect)
    if source_feature is not None:
        return source_feature

    parts, source_names = _extract_source_qualifier_tokens(
        parts,
        known_source_names=known_source_names,
        dialect=dialect,
    )
    parts = _semantic_core_measurement_feature_parts(parts, dialect)
    source_name = "__".join(source_names) if source_names else None
    return "_".join(parts), source_name


def _should_strip_measurement_category_prefix(
    parts: tuple[str, ...],
    prefix: tuple[str, ...],
) -> bool:
    if parts[: len(prefix)] != prefix or len(parts) <= len(prefix):
        return False
    suffix = parts[len(prefix) :]
    if prefix == (PairMeasurementFeature.CORRELATION.value,):
        return not _measurement_qualifier_parts_only(suffix)
    return True


def _measurement_qualifier_parts_only(parts: tuple[str, ...]) -> bool:
    return bool(parts) and all(part.isdigit() for part in parts)


def _aggregate_prefixed_feature_name_and_source(
    parts: tuple[str, ...],
    *,
    known_source_names: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> tuple[str, str | None] | None:
    if len(parts) < 3 or parts[0] not in _MEASUREMENT_AGGREGATE_PREFIXES:
        return None
    object_name_parts, feature_parts = _aggregate_object_and_feature_parts(
        parts[1:],
        dialect,
    )
    if not object_name_parts or not feature_parts:
        return None
    feature_name, source_name = _semantic_core_feature_name_and_source(
        "_".join(feature_parts),
        known_source_names=known_source_names,
        dialect=dialect,
    )
    feature_name_parts = tuple(part for part in feature_name.split("_") if part)
    if not feature_name_parts:
        return None
    return (
        "_".join((parts[0], *object_name_parts, *feature_name_parts)),
        source_name,
    )


def _semantic_core_measurement_feature_parts(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> tuple[str, ...]:
    aliased = dialect.feature_part_aliases.get(parts)
    if aliased is not None:
        return aliased
    numbered_alias = _numbered_feature_parts_alias(parts, dialect)
    if numbered_alias is not None:
        return numbered_alias
    for prefix in dialect.scale_qualified_feature_prefixes:
        if (
            len(parts) == len(prefix) + 1
            and parts[: len(prefix)] == prefix
            and parts[-1].isdigit()
        ):
            return prefix
    if (
        len(parts) > 2
        and parts[:2] == ("threshold", "otsu")
        and all(
            part.isdigit() or part in dialect.threshold_qualifier_tokens
            for part in parts[2:]
        )
    ):
        return parts[:2]
    if len(parts) == 3 and parts[:2] == ("center", "mass"):
        return ("center", "mass", "intensity", parts[2])
    if len(parts) == 2 and parts[0] == "center" and parts[1] in {"x", "y", "z"}:
        return ("center", parts[1])
    return parts


def _numbered_feature_parts_alias(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> tuple[str, ...] | None:
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    prefix_alias = dialect.numbered_feature_prefix_aliases.get(parts[0])
    if prefix_alias is None:
        return None
    return (*prefix_alias, str(int(parts[1])))


def _source_feature_name_and_source(
    parts: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> tuple[str, str | None] | None:
    """Protect dialect-defined source feature phrases from source-name extraction."""
    for prefix in dialect.source_feature_prefixes:
        if parts[: len(prefix)] != prefix:
            continue
        source_parts = parts[len(prefix) :]
        source_name = "_".join(source_parts) if source_parts else None
        return "_".join(prefix), source_name
    return None


def _directional_pair_feature_name_and_source(
    feature_name: str,
    source_name: str | None,
    dialect: RuntimeMeasurementDialect,
) -> tuple[str, str | None]:
    alias = dialect.directional_pair_feature_aliases.get(feature_name)
    if source_name is not None and feature_name in _UNDIRECTED_PAIR_FEATURES:
        return feature_name, _canonical_pair_source_name(source_name)
    if alias is None or source_name is None:
        return feature_name, source_name

    source_parts = tuple(part for part in source_name.split("__") if part)
    if len(source_parts) != 2:
        return feature_name, source_name

    canonical_feature_name, direction_index = alias
    directed_source_name = (
        "__".join(reversed(source_parts))
        if direction_index == 2
        else source_name
    )
    return canonical_feature_name, directed_source_name


def _canonical_pair_source_name(source_name: str) -> str:
    source_parts = tuple(part for part in source_name.split("__") if part)
    if len(source_parts) != 2:
        return source_name
    return "__".join(sorted(source_parts))


def _strip_source_qualifier_tokens(parts: tuple[str, ...]) -> tuple[str, ...]:
    return _extract_source_qualifier_tokens(
        parts,
        known_source_names=(),
        dialect=DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    )[0]


def _extract_source_qualifier_tokens(
    parts: tuple[str, ...],
    *,
    known_source_names: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> _RuntimeMeasurementNameParts:
    source_token_groups = _source_name_token_groups(known_source_names)
    stripped: list[str] = []
    source_names: list[str] = []
    index = 0
    while index < len(parts):
        matched_source_name = _matching_source_name_at(parts, index, source_token_groups)
        if matched_source_name is not None:
            source_names.append(matched_source_name)
            index += len(matched_source_name.split("_"))
            continue
        if (
            index + 1 < len(parts)
            and parts[index] in dialect.source_qualifier_prefix_tokens
            and parts[index + 1] in dialect.source_qualifier_suffix_tokens
        ):
            source_names.append(f"{parts[index]}_{parts[index + 1]}")
            index += 2
            continue
        stripped.append(parts[index])
        index += 1
    return tuple(stripped), tuple(source_names)


def _source_name_token_groups(
    known_source_names: tuple[str, ...],
) -> _RuntimeSourceTokenGroups:
    groups = tuple(
        (normalized, _source_name_tokens(normalized))
        for normalized in (
            _normalize_source_name(source_name) for source_name in known_source_names
        )
        if normalized
    )
    return sorted_tuple(groups, key=lambda group: (-len(group[1]), group[0]))


def _matching_source_name_at(
    parts: tuple[str, ...],
    index: int,
    source_token_groups: _RuntimeSourceTokenGroups,
) -> str | None:
    for source_name, source_parts in source_token_groups:
        if parts[index : index + len(source_parts)] == source_parts:
            return source_name
    return None


def _source_name_tokens(source_name: str) -> tuple[str, ...]:
    return tuple(
        token
        for part in source_name.split("__")
        for token in part.split("_")
        if token
    )


def _strip_subject_suffix_feature_name(
    feature_name: str,
    subject: RuntimeMeasurementSubjectKey,
) -> str:
    """Remove redundant object-table suffixes from object-scoped features."""
    if subject.scope is not MeasurementScope.OBJECT or subject.name is None:
        return feature_name
    subject_parts = tuple(part for part in subject.name.split("_") if part)
    feature_parts = tuple(part for part in feature_name.split("_") if part)
    if (
        subject_parts
        and len(feature_parts) > len(subject_parts)
        and feature_parts[-len(subject_parts) :] == subject_parts
    ):
        return "_".join(feature_parts[: -len(subject_parts)])
    return feature_name


def _measurement_row_qualifiers(
    row: Mapping[str, object],
    dialect: RuntimeMeasurementDialect,
    field_name: str,
) -> tuple[str, ...]:
    return _measurement_row_qualifiers_for_field(
        dialect,
        field_name,
        lambda qualifier: _render_measurement_row_qualifier(row, qualifier),
    )


def _measurement_row_qualifiers_for_field(
    dialect: RuntimeMeasurementDialect,
    field_name: str,
    render: Callable[[RuntimeMeasurementRowQualifier], str | None],
) -> tuple[str, ...]:
    qualifiers: list[str] = []
    field_parts = tuple(part for part in _normalize_identifier(field_name).split("_") if part)
    for qualifier in dialect.row_qualifiers:
        if not _row_qualifier_applies_to_field(qualifier, field_parts):
            continue
        rendered = render(qualifier)
        if rendered is None:
            continue
        qualifiers.append(rendered)
    return tuple(qualifiers)


def _measurement_row_qualifiers_from_values(
    row_values: Mapping[str, object],
    dialect: RuntimeMeasurementDialect,
    field_name: str,
) -> tuple[str, ...]:
    return _measurement_row_qualifiers_for_field(
        dialect,
        field_name,
        lambda qualifier: _render_measurement_row_qualifier_from_values(
            row_values,
            qualifier,
        ),
    )


def _measurement_row_qualifiers_from_indexed_values_cached(
    row_values: tuple[object, ...],
    qualifiers: tuple[_RuntimeMeasurementIndexedQualifier, ...],
    cache: dict[_RuntimeMeasurementQualifierCacheKey, str | None],
) -> tuple[str, ...]:
    rendered_values: list[str] = []
    for qualifier, indexes in qualifiers:
        values = tuple(
            None if index is None else row_values[index]
            for index in indexes
        )
        cache_key = (
            qualifier,
            tuple(None if value is None else str(value) for value in values),
        )
        rendered = cache.get(cache_key)
        if cache_key not in cache:
            rendered = _render_measurement_row_qualifier_value_tuple(
                values,
                qualifier,
            )
            cache[cache_key] = rendered
        if rendered is not None:
            rendered_values.append(rendered)
    return tuple(rendered_values)


def _row_qualifier_columns(
    normalized_fields: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> tuple[tuple[str, int], ...]:
    qualifier_fields = _measurement_qualifier_field_names(dialect)
    return tuple(
        (field_name, index)
        for index, field_name in enumerate(normalized_fields)
        if field_name in qualifier_fields
    )


def _row_qualifier_values(
    row: tuple[object, ...],
    columns: tuple[tuple[str, int], ...],
) -> Mapping[str, object]:
    return MappingProxyType(
        {field_name: row[index] for field_name, index in columns}
    )


def _row_qualifier_applies_to_field(
    qualifier: RuntimeMeasurementRowQualifier,
    field_parts: tuple[str, ...],
) -> bool:
    if not qualifier.feature_prefixes:
        return True
    return any(
        len(field_parts) >= len(prefix) and field_parts[: len(prefix)] == prefix
        for prefix in qualifier.feature_prefixes
    )


def _render_measurement_row_qualifier(
    row: Mapping[str, object],
    qualifier: RuntimeMeasurementRowQualifier,
) -> str | None:
    values = tuple(_first_row_value(row, (field_name,)) for field_name in qualifier.field_names)
    return _render_measurement_row_qualifier_value_tuple(values, qualifier)


def _render_measurement_row_qualifier_from_values(
    row_values: Mapping[str, object],
    qualifier: RuntimeMeasurementRowQualifier,
) -> str | None:
    values = tuple(row_values.get(field_name) for field_name in qualifier.field_names)
    return _render_measurement_row_qualifier_value_tuple(values, qualifier)


def _render_measurement_row_qualifier_value_tuple(
    values: tuple[object, ...],
    qualifier: RuntimeMeasurementRowQualifier,
) -> str | None:
    if any(_is_missing_measurement_qualifier_value(value) for value in values):
        return None
    return _MEASUREMENT_QUALIFIER_VALUE_RENDERERS[qualifier.value_mode](values)


def _measurement_qualifier_identifier(value: object) -> str:
    integer_value = _optional_measurement_qualifier_integer(value)
    if integer_value is not None:
        return str(integer_value)
    return _normalize_identifier(value)


def _is_missing_measurement_qualifier_value(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    try:
        return math.isnan(float(text))
    except (TypeError, ValueError):
        return False


def _measurement_qualifier_integer(value: object) -> int:
    integer_value = _optional_measurement_qualifier_integer(value)
    if integer_value is None:
        raise ValueError(f"Measurement qualifier {value!r} is not integer-like.")
    return integer_value


def _optional_measurement_qualifier_integer(value: object) -> int | None:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _measurement_qualifier_field_names(
    dialect: RuntimeMeasurementDialect,
) -> frozenset[str]:
    cached = _MEASUREMENT_DIALECT_QUALIFIER_FIELD_NAMES_CACHE.get(id(dialect))
    if cached is not None and cached[0] is dialect:
        return cached[1]
    field_names = frozenset(
        field_name
        for qualifier in dialect.row_qualifiers
        for field_name in qualifier.field_names
    )
    _MEASUREMENT_DIALECT_QUALIFIER_FIELD_NAMES_CACHE[id(dialect)] = (
        dialect,
        field_names,
    )
    return field_names


def _is_measurement_identity_field(
    field_name: str,
    dialect: RuntimeMeasurementDialect,
) -> bool:
    normalized = _normalize_identifier(field_name)
    return _is_normalized_measurement_identity_field(normalized, dialect)


def _is_normalized_measurement_identity_field(
    normalized: str,
    dialect: RuntimeMeasurementDialect,
) -> bool:
    if normalized in _MEASUREMENT_IDENTITY_FIELDS:
        return True
    if normalized in _measurement_qualifier_field_names(dialect):
        return True
    if normalized.startswith(_NON_MEASUREMENT_FIELD_PREFIXES):
        return True
    return normalized.startswith("metadata_")


_MEASUREMENT_FEATURE_NAME_FIELDS = MEASUREMENT_FEATURE_NAME_FIELDS
_MEASUREMENT_VALUE_FIELDS = MEASUREMENT_VALUE_FIELDS
_MEASUREMENT_AGGREGATE_PREFIXES = frozenset({"mean"})
_MEASUREMENT_QUALIFIER_FIELDS = (
    "scale",
    "direction",
    "gray_levels",
)
_MEASUREMENT_QUALIFIER_FIELD_SET = frozenset(_MEASUREMENT_QUALIFIER_FIELDS)
_OBJECT_IDENTITY_FIELDS = frozenset(MEASUREMENT_OBJECT_ID_FIELDS)
_IMAGE_IDENTITY_FIELDS = frozenset({"image_number", "image_id", "slice_index"})
_NON_MEASUREMENT_FIELD_PREFIXES = (
    "channel_",
    "execution_time_",
    "file_name_",
    "frame_",
    "group_",
    "height_",
    "image_quality_scaling_",
    "image_set_",
    "md_5_digest_",
    "md5_digest_",
    "module_error_",
    "path_name_",
    "scaling_",
    "series_",
    "url_",
    "width_",
)
_PAIR_CORRELATION_FEATURE = PairMeasurementFeature.CORRELATION.value
_PAIR_REGRESSION_SLOPE_FEATURE = PairMeasurementFeature.REGRESSION_SLOPE.value
_PAIR_OVERLAP_FEATURE = PairMeasurementFeature.OVERLAP.value
_PAIR_COSTES_MANDERS_FEATURE = PairMeasurementFeature.COSTES_MANDERS.value
_UNDIRECTED_PAIR_FEATURES = frozenset(
    (_PAIR_CORRELATION_FEATURE, _PAIR_OVERLAP_FEATURE)
)
_THRESHOLD_SENSITIVE_PAIR_FEATURES = frozenset((_PAIR_COSTES_MANDERS_FEATURE,))
