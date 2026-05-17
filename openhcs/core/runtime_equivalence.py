"""Semantic equivalence checks for runtime outputs."""

from __future__ import annotations

import hashlib
import inspect
import math
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, ClassVar, Generic, TypeVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

import openhcs.core.runtime_artifact_queries as runtime_artifact_queries
import openhcs.core.equivalence.measurement_features as measurement_features
import openhcs.core.runtime_semantics as runtime_semantics
from openhcs.core.artifacts import ArtifactKind, ArtifactScope
from openhcs.core.runtime_artifact_queries import (
    MEASUREMENT_FEATURE_NAME_FIELDS,
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    MEASUREMENT_VALUE_FIELDS,
    iter_measurement_rows,
    measurement_object_label,
    measurement_row_has_object_identity,
    measurement_row_mapping,
    measurement_row_object_name,
    measurement_row_source_image_name,
    measurement_table_object_id_field,
)
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactExecutionObservation,
)
from openhcs.core.runtime_exports import RuntimeImageExportSpec
from openhcs.core.runtime_semantics import (
    IndexedObjectZernikeDescriptor,
    MeasurementStatistic,
    MeasurementScope,
    MeasurementSubject,
    ObjectInstanceKey,
    ObjectInstanceRelationship,
    ObjectLabelInstanceDomains,
    ObjectLabelDomainScope,
    ObjectCoreMeasurementFeature,
    ObjectLocationCoordinateValues,
    ObjectMeasurementFeatureRole,
    ObjectZernikeDescriptorFeature,
    PairMeasurementFeature,
    dense_object_label_id_domain,
    dense_object_label_identity_domains,
    ObjectLabelIdDomainStrategy,
    dense_object_label_plane_id_domains,
    object_location_coordinate_arrays,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    MostDerivedContextStrategyMixin,
)
from openhcs.core.runtime_values import MeasurementTable
from openhcs.core.runtime_values import ObjectLabelSet
from openhcs.core.runtime_values import ObjectRelationship
from openhcs.core.runtime_values import RuntimeValue
from openhcs.core.equivalence.policy import (
    DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    RuntimeEquivalencePolicy,
    RuntimeMeasurementDialect,
    RuntimeMeasurementFeatureNameMode,
    RuntimeMeasurementFeatureNumericTolerance,
    RuntimeMeasurementRowQualifier,
    normalize_runtime_identifier as _normalize_identifier,
    normalize_runtime_source_name as _normalize_source_name,
    runtime_source_name_tokens as _source_name_tokens,
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
from openhcs.core.equivalence.arrays import (
    canonical_scalar,
    semantic_array_payload,
)
from openhcs.core.equivalence.tables import (
    CSV_HEADER_CONTEXT_STOPWORDS as _CSV_HEADER_CONTEXT_STOPWORDS,
    MEASUREMENT_IDENTITY_FIELDS as _MEASUREMENT_IDENTITY_FIELDS,
    RuntimeTableSnapshot,
    RuntimeMeasurementRowFingerprintBuilder,
    RuntimeMeasurementTableIdentity,
    aggregate_measurement_table_key,
    exact_measurement_table_key,
    is_static_wide_measurement_table,
    is_wide_measurement_table,
    measurement_table_cell_payload,
    update_measurement_table_cell_hash,
)
from openhcs.core.equivalence.measurement_rows import (
    IMAGE_IDENTITY_FIELDS as _IMAGE_IDENTITY_FIELDS,
    axis_scoped_measurement_row_identity,
    measurement_qualifier_field_names,
    measurement_row_qualifiers,
    measurement_row_qualifiers_from_indexed_values_cached,
    measurement_row_qualifiers_from_values,
    row_qualifier_applies_to_field,
    row_qualifier_columns,
    row_qualifier_values,
)
from openhcs.core.equivalence.measurement_facts import (
    record_measurement_facts,
    spatial_grid_measurement_facts,
)
from openhcs.core.equivalence.measurement_features import (
    object_measurement_feature_has_role,
    object_measurement_feature_requires_sparse_boundary_object_count_stability,
    object_measurement_subject_row_identities_with_role,
    object_measurement_subjects_with_role,
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
    measurement_features,
    runtime_artifact_queries,
    runtime_semantics,
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
_RuntimeMeasurementFactCounterMapping = Mapping[
    RuntimeMeasurementFeatureKey,
    Counter[RuntimeCellSignature],
]
_RuntimeMeasurementKeySet = frozenset[RuntimeMeasurementFeatureKey]
_RuntimeRequiredMeasurementKeys = _RuntimeMeasurementKeySet | None
_RuntimeObjectValuesByLabel = dict[
    RuntimeMeasurementFeatureKey,
    dict[ObjectInstanceKey, float],
]
_RuntimeObjectValuesByObject = dict[
    tuple[str, _RuntimeRequiredMeasurementKeys],
    _RuntimeObjectValuesByLabel,
]
_RuntimeLongFormMeasurementFact = _RuntimeMeasurementFact | None
_RuntimeNumericMeasurementValue = tuple[RuntimeMeasurementFeatureKey, float]
_RuntimeNumericMeasurementValues = tuple[_RuntimeNumericMeasurementValue, ...]
_RuntimeMeasurementRowIdentity = tuple[tuple[str, object], ...]
_RuntimeMeasurementRowIdentityOrMissing = _RuntimeMeasurementRowIdentity | None
_RuntimeMeasurementRowMergeKey = tuple[
    RuntimeMeasurementFeatureKey,
    _RuntimeMeasurementRowIdentity,
]
_RuntimeMeasurementRowMergeValue = tuple[int, int, RuntimeCellSignature]
_RuntimeMeasurementRowMergeCache = dict[
    _RuntimeMeasurementRowMergeKey,
    _RuntimeMeasurementRowMergeValue,
]
_ObjectInstanceChildrenByParent = Mapping[
    ObjectInstanceKey,
    tuple[ObjectInstanceKey, ...],
]
_ObjectInstanceAggregateValues = dict[tuple[ObjectInstanceKey, ...], float]
@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowPriorityCacheKey:
    """Identity for row-to-feature priority resolution within one equivalence pass."""

    row_fields: tuple[str, ...]
    long_form_feature: str | None
    feature_key: RuntimeMeasurementFeatureKey


_RuntimeMeasurementRowPriorityCache = dict[RuntimeMeasurementRowPriorityCacheKey, int]
_RuntimeMeasurementPrimaryRowKey = tuple[
    RuntimeMeasurementSubjectKey,
    _RuntimeMeasurementRowIdentity,
]
_RuntimeMeasurementPrimaryRowSet = set[_RuntimeMeasurementPrimaryRowKey]
_OBJECT_LABEL_ROW_IDENTITY_FIELD = "object_label"
_RuntimeMeasurementObjectSubtableKey = RuntimeMeasurementTableIdentity
_RuntimeMeasurementObjectSubtableSet = set[_RuntimeMeasurementObjectSubtableKey]
_RuntimeMeasurementNameParts = tuple[tuple[str, ...], tuple[str, ...]]
_RuntimeSourceTokenGroups = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class _RuntimeMeasurementRowSchema:
    """Schema indexes for one runtime measurement table row shape."""

    feature_indexes: tuple[int, ...]
    qualifiers_by_index: dict[int, tuple[_RuntimeMeasurementIndexedQualifier, ...]]
    long_form_feature_indexes: tuple[int, ...]
    long_form_value_indexes: tuple[int, ...]


_RuntimeMeasurementQualifierCacheKey = tuple[
    RuntimeMeasurementRowQualifier,
    tuple[object | None, ...],
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
_RuntimeMeasurementIndexedQualifierCache = dict[int, tuple[str, ...]]
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


@dataclass(slots=True)
class RuntimeAggregateMeanAccumulator:
    """Running aggregate mean state without retaining per-row values."""

    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def has_values(self) -> bool:
        return self.count > 0

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ValueError("Cannot compute mean without values.")
        return self.total / self.count


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFactCounters:
    """Mutable counter map for runtime measurement facts."""

    values_by_feature: _RuntimeMeasurementFactCounters

    def counter(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> Counter[RuntimeCellSignature]:
        counter = self.values_by_feature.get(key)
        if counter is None:
            counter = Counter()
            self.values_by_feature[key] = counter
        return counter


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellSignatureProjection:
    """Project runtime measurement payloads into comparison cell signatures."""

    value: object
    policy: RuntimeEquivalencePolicy

    def signature(self) -> RuntimeCellSignature:
        value = canonical_scalar(self.value)
        if value is None or isinstance(value, (str, bool, int, float)):
            return _cached_runtime_cell_signature(
                str(value),
                self.policy.numeric_decimal_places,
            )
        array_payload = semantic_array_payload(value)
        if array_payload is not None:
            dtype, shape, digest = array_payload[1:]
            return RuntimeCellSignature(
                RuntimeCellValueKind.TEXT,
                f"array:{dtype}:{'x'.join(str(axis) for axis in shape)}:{digest}",
            )
        if _runtime_value_is_mapping(value) or isinstance(value, (tuple, list)):
            value_digest = hashlib.blake2b(digest_size=32)
            update_measurement_table_cell_hash(value_digest, value)
            return RuntimeCellSignature(
                RuntimeCellValueKind.TEXT,
                f"{type(value).__name__}:{value_digest.hexdigest()}",
            )
        return _cell_signature(str(value), self.policy)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellPresence:
    """Presence semantics for runtime measurement cell payloads."""

    value: object

    def is_present(self) -> bool:
        value = canonical_scalar(self.value)
        if value is None:
            return False
        array_payload = semantic_array_payload(value)
        if array_payload is not None:
            return any(axis > 0 for axis in array_payload[2])
        if _runtime_value_is_mapping(value) or isinstance(value, (tuple, list)):
            return bool(value)
        text = str(value).strip()
        if not text:
            return False
        try:
            numeric = float(text)
        except ValueError:
            return True
        return not math.isnan(numeric)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementValuePresence:
    """Presence semantics for scalar or nested runtime measurement values."""

    value: object

    def is_present(self) -> bool:
        if _runtime_value_is_mapping(self.value):
            return any(
                RuntimeMeasurementValuePresence(nested).is_present()
                for nested in self.value.values()
            )
        return RuntimeMeasurementCellPresence(self.value).is_present()


@dataclass(frozen=True, slots=True)
class RuntimeImageNumberOffset:
    """Compute image-number offset for table rows."""

    @classmethod
    def from_table_rows(
        cls,
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
        return cls._offset_from_values(
            row[image_number_index]
            for row in rows
            if image_number_index < len(row)
        )

    @classmethod
    def from_runtime_rows(cls, rows: Iterable[object]) -> float:
        return cls._offset_from_values(
            image_number
            for row in rows
            for image_number in (
                RuntimeMeasurementRowMapping(
                    measurement_row_mapping(row)
                ).first_value(("image_number",)),
            )
            if image_number is not None
        )

    @classmethod
    def _offset_from_values(cls, values: Iterable[object]) -> float:
        image_numbers: list[float] = []
        for value in values:
            try:
                image_number = float(str(value).strip())
            except ValueError:
                continue
            if math.isfinite(image_number) and image_number > 0:
                image_numbers.append(image_number)
        if not image_numbers:
            return 0.0
        return min(image_numbers) - 1.0


@dataclass(frozen=True, slots=True)
class RuntimeImageNumberReferenceValue:
    """Normalize image-number reference fields to axis-local numbering."""

    field_name: str
    value: object
    image_number_offset: float

    def normalized(self) -> object:
        if self.image_number_offset == 0:
            return self.value
        if not _is_image_number_reference_measurement_field(self.field_name):
            return self.value
        if isinstance(self.value, Mapping):
            return {
                key: RuntimeImageNumberReferenceValue(
                    self.field_name,
                    nested_value,
                    self.image_number_offset,
                ).normalized()
                for key, nested_value in self.value.items()
            }
        try:
            numeric_value = float(str(self.value).strip())
        except ValueError:
            return self.value
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            return self.value
        normalized = numeric_value - self.image_number_offset
        return int(normalized) if normalized.is_integer() else normalized


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFieldIndexMap:
    """Normalized field-name lookup for runtime measurement rows."""

    normalized_field_indexes: Mapping[str, int]

    def indexes_for(self, field_names: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(
            index
            for field_name in field_names
            if (index := self.normalized_field_indexes.get(field_name)) is not None
        )


@dataclass(frozen=True, slots=True)
class RuntimeIndexedRowValues:
    """Typed accessors for row values indexed by schema positions."""

    row_values: tuple[object, ...]

    def first_at(self, indexes: tuple[int, ...]) -> object | None:
        if not indexes:
            return None
        return self.row_values[indexes[0]]

    def text_at(self, index: int | None) -> str | None:
        if index is None:
            return None
        value = self.row_values[index]
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def has_text_at_any(self, indexes: tuple[int, ...]) -> bool:
        for index in indexes:
            if self.text_at(index) is not None:
                return True
        return False


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowMapping:
    """Normalized field-name lookup for runtime measurement row mappings."""

    row: Mapping[str, object]

    def first_value(self, field_names: tuple[str, ...]) -> object | None:
        normalized_fields = {_normalize_identifier(field): field for field in self.row}
        for field_name in field_names:
            field = normalized_fields.get(field_name)
            if field is not None:
                return self.row[field]
        return None

    def has_identity_value(self, field_names: frozenset[str]) -> bool:
        normalized_fields = {_normalize_identifier(field): field for field in self.row}
        for field_name in field_names:
            field = normalized_fields.get(field_name)
            if field is None:
                continue
            value = self.row[field]
            if value is None:
                continue
            if str(value).strip():
                return True
        return False

    def has_image_identity(self) -> bool:
        return self.has_identity_value(_IMAGE_IDENTITY_FIELDS)

    def has_object_identity(self) -> bool:
        return self.has_identity_value(_OBJECT_IDENTITY_FIELDS)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureKeyFactory:
    """Factory for runtime measurement keys with image-source subject folding."""

    subject: RuntimeMeasurementSubjectKey
    feature_name: str
    statistic: str = MeasurementStatistic.VALUE.value
    source_name: str | None = None

    def key(self) -> RuntimeMeasurementFeatureKey:
        if self.subject.scope is MeasurementScope.IMAGE and self.source_name is not None:
            return RuntimeMeasurementFeatureKey(
                RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, self.source_name),
                self.feature_name,
                self.statistic,
            )
        return RuntimeMeasurementFeatureKey(
            self.subject,
            self.feature_name,
            self.statistic,
            self.source_name,
        )


@dataclass(frozen=True, slots=True)
class RuntimeMetadataMapRow:
    """Experiment metadata key/value row predicate."""

    subject: RuntimeMeasurementSubjectKey
    row: Mapping[str, object]

    def matches(self) -> bool:
        if self.subject.scope is not MeasurementScope.EXPERIMENT:
            return False
        normalized_fields = frozenset(
            _normalize_identifier(field_name) for field_name in self.row
        )
        return normalized_fields == frozenset(("key", "value"))


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementIdentityField:
    """Runtime measurement identity-field predicate for a dialect."""

    dialect: RuntimeMeasurementDialect

    def field_matches(self, field_name: str) -> bool:
        return self.normalized_field_matches(_normalize_identifier(field_name))

    def normalized_field_matches(self, normalized: str) -> bool:
        if normalized in _MEASUREMENT_IDENTITY_FIELDS:
            return True
        if normalized in measurement_qualifier_field_names(self.dialect):
            return True
        if normalized.startswith(_NON_MEASUREMENT_FIELD_PREFIXES):
            return True
        return normalized.startswith("metadata_")


_AggregateValuesByFeature = dict[
    tuple[RuntimeMeasurementFeatureKey, tuple[tuple[str, object], ...]],
    RuntimeAggregateMeanAccumulator,
]
_AggregateMeanKeyCache = dict[
    tuple[MeasurementScope, str | None, str, str, str | None],
    RuntimeMeasurementFeatureKey | None,
]
_RuntimeRowProjectionValueT = TypeVar("_RuntimeRowProjectionValueT")
_RuntimeRowProjectionRecord = tuple[
    _RuntimeMeasurementPaddingGroup,
    RuntimeMeasurementFeatureKey,
    _RuntimeRowProjectionValueT,
]


@dataclass(frozen=True, slots=True)
class RuntimeObjectMeasurementRowIdentity:
    """Nominal identity for one object measurement row within a runtime axis."""

    row_identity: _RuntimeMeasurementRowIdentity

    @classmethod
    def from_row_mapping(
        cls,
        row_mapping: Mapping[str, object],
        axis_key: object | None,
        policy: RuntimeEquivalencePolicy,
    ) -> "RuntimeObjectMeasurementRowIdentity | None":
        try:
            object_label = measurement_object_label(row_mapping)
        except (TypeError, ValueError):
            return None
        if object_label is None:
            return None
        return cls(
            (
                *axis_scoped_measurement_row_identity(
                    row_mapping,
                    axis_key,
                    policy.measurement_dialect,
                ),
                (
                    _OBJECT_LABEL_ROW_IDENTITY_FIELD,
                    RuntimeMeasurementCellSignatureProjection(object_label, policy).signature(),
                ),
            )
        )

    @property
    def image_identity(self) -> _RuntimeMeasurementRowIdentity:
        return tuple(
            field
            for field in self.row_identity
            if field[0] != _OBJECT_LABEL_ROW_IDENTITY_FIELD
        )

    @property
    def has_image_identity(self) -> bool:
        return any(field[0] in _IMAGE_IDENTITY_FIELDS for field in self.row_identity)

    @property
    def object_label_signature(self) -> RuntimeCellSignature | None:
        return next(
            (
                field[1]
                for field in self.row_identity
                if (
                    field[0] == _OBJECT_LABEL_ROW_IDENTITY_FIELD
                    and isinstance(field[1], RuntimeCellSignature)
                )
            ),
            None,
        )


@dataclass(slots=True)
class RuntimeObjectMeasurementFactRowDomain:
    """Object row identities proven by emitted measurement facts."""

    identities_by_subject: dict[
        RuntimeMeasurementSubjectKey,
        set[RuntimeObjectMeasurementRowIdentity],
    ] = field(default_factory=dict)

    def record_row_facts(
        self,
        row_mapping: Mapping[str, object],
        axis_key: object | None,
        policy: RuntimeEquivalencePolicy,
        facts: Iterable[_RuntimeMeasurementFact],
    ) -> None:
        identity = RuntimeObjectMeasurementRowIdentity.from_row_mapping(
            row_mapping,
            axis_key,
            policy,
        )
        if identity is None:
            return
        subjects = frozenset(
            key.subject
            for key, _value in facts
            if key.subject.scope is MeasurementScope.OBJECT
        )
        for subject in subjects:
            self.identities_by_subject.setdefault(subject, set()).add(identity)

    def record_subject_row_identity(
        self,
        subject: RuntimeMeasurementSubjectKey,
        row_identity: _RuntimeMeasurementRowIdentity,
    ) -> None:
        if subject.scope is not MeasurementScope.OBJECT:
            return
        self.identities_by_subject.setdefault(subject, set()).add(
            RuntimeObjectMeasurementRowIdentity(row_identity)
        )

    def primary_row_keys(self) -> frozenset[_RuntimeMeasurementPrimaryRowKey]:
        return frozenset(
            (subject, identity.row_identity)
            for subject, identities in self.identities_by_subject.items()
            for identity in identities
            if identity.has_image_identity
        )


def _runtime_aggregate_mean_accumulator(
    values_by_feature: _AggregateValuesByFeature,
    key: RuntimeMeasurementFeatureKey,
    row_identity: _RuntimeMeasurementRowIdentity,
) -> RuntimeAggregateMeanAccumulator:
    accumulator_key = (key, row_identity)
    accumulator = values_by_feature.get(accumulator_key)
    if accumulator is None:
        accumulator = RuntimeAggregateMeanAccumulator()
        values_by_feature[accumulator_key] = accumulator
    return accumulator


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementProjectionCachePlan:
    """Which auxiliary row caches are needed for a required-key projection."""

    required_keys: _RuntimeRequiredMeasurementKeys
    policy: RuntimeEquivalencePolicy

    @property
    def needs_primary_row_identities(self) -> bool:
        if self.required_keys is None:
            return True
        return any(
            self._required_key_needs_primary_row_identity_set(key)
            for key in self.required_keys
        )

    def _required_key_needs_primary_row_identity_set(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> bool:
        if key.subject.scope is not MeasurementScope.OBJECT:
            return False
        if (
            key.statistic == MeasurementStatistic.COUNT.value
            and key.feature_name == ObjectCoreMeasurementFeature.OBJECT_COUNT.value
        ):
            return True
        row_merge_contract = RuntimeObjectLocationRowMergeContract(self.policy)
        return any(
            row_merge_contract.owns_key(input_key)
            for input_key in RuntimeMeasurementStatisticDependencyStrategy.for_enum_member(
                MeasurementStatistic(key.statistic)
            ).required_input_keys(key)
        )

    def primary_row_identities(self) -> _RuntimeMeasurementPrimaryRowSet | None:
        return set() if self.needs_primary_row_identities else None

    def row_merge_cache(self) -> _RuntimeMeasurementRowMergeCache:
        return {}


class RuntimeMeasurementStatisticDependencyStrategy(
    EnumKeyedStrategyMixin[MeasurementStatistic],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declare input measurement keys required by one output statistic."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "statistic"

    statistic: ClassVar[MeasurementStatistic]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def required_input_keys(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        """Return semantic input keys needed to produce ``key``."""


class ValueMeasurementStatisticDependencyStrategy(
    RuntimeMeasurementStatisticDependencyStrategy
):
    """Value facts are their own projection input."""

    statistic = MeasurementStatistic.VALUE

    def required_input_keys(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        return (key,)


class MeanMeasurementStatisticDependencyStrategy(
    RuntimeMeasurementStatisticDependencyStrategy
):
    """Mean facts depend on row-identifiable value facts."""

    statistic = MeasurementStatistic.MEAN

    def required_input_keys(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        return (
            RuntimeMeasurementFeatureKey(
                key.subject,
                key.feature_name,
                MeasurementStatistic.VALUE.value,
                key.source_name,
            ),
        )


class CountMeasurementStatisticDependencyStrategy(
    RuntimeMeasurementStatisticDependencyStrategy
):
    """Count facts are explicit facts unless higher-level object-count logic applies."""

    statistic = MeasurementStatistic.COUNT

    def required_input_keys(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        return (key,)


class RuntimeObjectLocationRowMergeProjectionKey(Enum):
    """Registered object-location row-merge projection identities."""

    LOCATION = "location"
    AGGREGATE_LOCATION = "aggregate_location"


_RuntimeRowProjectionRecords = tuple[
    _RuntimeRowProjectionRecord[_RuntimeRowProjectionValueT],
    ...,
]
_RuntimeProjectedCell = tuple[
    RuntimeMeasurementFeatureKey,
    _RuntimeRowProjectionValueT,
]
_RuntimeProjectedCells = tuple[
    _RuntimeProjectedCell[_RuntimeRowProjectionValueT],
    ...,
]


@dataclass(frozen=True, slots=True)
class RuntimeRowProjection(Generic[_RuntimeRowProjectionValueT]):
    records: _RuntimeRowProjectionRecords[_RuntimeRowProjectionValueT]
    long_form: bool = False


def runtime_row_projection(
    records: Iterable[_RuntimeRowProjectionRecord[_RuntimeRowProjectionValueT]] = (),
    *,
    long_form: bool = False,
) -> RuntimeRowProjection[_RuntimeRowProjectionValueT]:
    """Build a row projection through one normalized record boundary."""
    return RuntimeRowProjection(tuple(records), long_form=long_form)


@dataclass(frozen=True, slots=True)
class RuntimeRowProjectionContext:
    row: Mapping[str, object]
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    source_name: str | None
    known_source_names: tuple[str, ...]
    required_keys: _RuntimeRequiredMeasurementKeys
    table_padding_group: str
    image_number_offset: float
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
        image_number_offset: float,
        schema_cache: _RuntimeMeasurementRowSchemaCache,
        key_cache: _RuntimeMeasurementFeatureKeyCache,
        long_form_key_cache: _RuntimeMeasurementLongFormKeyCache,
        qualifier_render_cache: _RuntimeMeasurementQualifierRenderCache,
        padding_group_cache: _RuntimeMeasurementPaddingGroupCache,
    ) -> "RuntimeRowProjectionContext":
        return cls(
            row=row,
            subject=subject,
            policy=policy,
            source_name=source_name,
            known_source_names=known_source_names,
            required_keys=required_keys,
            table_padding_group=table_padding_group,
            image_number_offset=image_number_offset,
            schema_cache=schema_cache,
            key_cache=key_cache,
            long_form_key_cache=long_form_key_cache,
            qualifier_render_cache=qualifier_render_cache,
            padding_group_cache=padding_group_cache,
        )


class RuntimeRowValueProjection(
    ABC,
    Generic[_RuntimeRowProjectionValueT],
):
    """Project wide-form runtime measurement values for row fact extraction."""

    @abstractmethod
    def project(
        self,
        key: RuntimeMeasurementFeatureKey,
        value: object,
        policy: RuntimeEquivalencePolicy,
    ) -> _RuntimeProjectedCells[_RuntimeRowProjectionValueT]:
        """Project one wide-form cell into semantic values."""


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellValue:
    """Runtime measurement cell plus nested value expansion policy."""

    key: RuntimeMeasurementFeatureKey
    value: object
    policy: RuntimeEquivalencePolicy
    required_keys: frozenset[RuntimeMeasurementFeatureKey] | None = None

    def iter_key_values(
        self,
    ) -> Iterable[tuple[RuntimeMeasurementFeatureKey, object]]:
        if not _runtime_value_is_mapping(self.value):
            if self.required_keys is not None and self.key not in self.required_keys:
                return ()
            return ((self.key, self.value),)
        return tuple(
            (nested_key, nested_value)
            for name, nested_value in self.value.items()
            for nested_key in (self.nested_key(name),)
            if self.required_keys is None or nested_key in self.required_keys
        )

    def nested_key(self, name: object) -> RuntimeMeasurementFeatureKey:
        return RuntimeMeasurementFeatureKeyFactory(
            self.key.subject,
            f"{self.key.feature_name}_{_canonical_measurement_feature_name(str(name), self.policy)}",
            self.key.statistic,
            source_name=self.key.source_name,
        ).key()


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellFactProjection(
    RuntimeRowValueProjection[RuntimeCellSignature],
):
    """Project runtime measurement cells into cell-signature facts."""

    def project(
        self,
        key: RuntimeMeasurementFeatureKey,
        value: object,
        policy: RuntimeEquivalencePolicy,
    ) -> _RuntimeMeasurementFacts:
        cell = RuntimeMeasurementCellValue(key, value, policy)
        return tuple(
            (cell_key, RuntimeMeasurementCellSignatureProjection(cell_value, policy).signature())
            for cell_key, cell_value in cell.iter_key_values()
        )

    def project_cell(
        self,
        cell: RuntimeMeasurementCellValue,
    ) -> _RuntimeMeasurementFacts:
        return tuple(
            (
                cell_key,
                RuntimeMeasurementCellSignatureProjection(cell_value, cell.policy).signature(),
            )
            for cell_key, cell_value in cell.iter_key_values()
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellNumericProjection(
    RuntimeRowValueProjection[float],
):
    """Project runtime measurement cells into numeric values."""

    def project(
        self,
        key: RuntimeMeasurementFeatureKey,
        value: object,
        policy: RuntimeEquivalencePolicy,
    ) -> _RuntimeNumericMeasurementValues:
        cell = RuntimeMeasurementCellValue(key, value, policy)
        return self.project_cell(cell)

    def project_cell(
        self,
        cell: RuntimeMeasurementCellValue,
    ) -> _RuntimeNumericMeasurementValues:
        return tuple(
            (cell_key, numeric_value)
            for cell_key, cell_value in cell.iter_key_values()
            if (
                numeric_value := _measurement_numeric_runtime_value(
                    cell_value,
                    cell.policy,
                )
            )
            is not None
        )


class RuntimeRowLongFormProjection(
    ABC,
    Generic[_RuntimeRowProjectionValueT],
):
    """Project normalized long-form measurement facts for row extraction."""

    @abstractmethod
    def project(
        self,
        fact: _RuntimeMeasurementFact,
    ) -> _RuntimeProjectedCells[_RuntimeRowProjectionValueT]:
        """Project one normalized long-form fact into semantic values."""


@dataclass(frozen=True, slots=True)
class RuntimeRowLongFormFactProjection(
    RuntimeRowLongFormProjection[RuntimeCellSignature],
):
    """Preserve long-form cell-signature facts as row facts."""

    def project(self, fact: _RuntimeMeasurementFact) -> _RuntimeMeasurementFacts:
        return (fact,)


@dataclass(frozen=True, slots=True)
class RuntimeRowLongFormNumericProjection(
    RuntimeRowLongFormProjection[float],
):
    """Project long-form cell-signature facts into numeric values."""

    def project(self, fact: _RuntimeMeasurementFact) -> _RuntimeNumericMeasurementValues:
        return _numeric_long_form_measurement_values(fact)


@dataclass(frozen=True, slots=True)
class _LongFormMeasurementContext:
    """Runtime row context for long-form measurement extraction."""

    row: Mapping[str, object]
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    source_name: str | None
    known_source_names: tuple[str, ...]
    image_number_offset: float


@dataclass(frozen=True, slots=True)
class _CachedLongFormMeasurementContext:
    row_values: tuple[object, ...]
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    source_name: str | None
    known_source_names: tuple[str, ...]
    image_number_offset: float
    feature_indexes: tuple[int, ...]
    value_indexes: tuple[int, ...]
    key_cache: _RuntimeMeasurementLongFormKeyCache

    @classmethod
    def from_runtime_row_projection(
        cls,
        context: RuntimeRowProjectionContext,
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
            context.image_number_offset,
            feature_indexes,
            value_indexes,
            context.long_form_key_cache,
        )


@dataclass(frozen=True, slots=True)
class _AggregateInputRecordingContext:
    """Context for recording row-local aggregate input values."""

    values_by_feature: _AggregateValuesByFeature
    row_mapping: Mapping[str, object]
    axis_key: object | None
    required_keys: _RuntimeRequiredMeasurementKeys
    key_cache: _AggregateMeanKeyCache
    measurement_dialect: RuntimeMeasurementDialect


@dataclass(frozen=True, slots=True)
class _RuntimeMeasurementFactRecordingContext:
    """Context for recording runtime measurement row facts."""

    values_by_feature: _RuntimeMeasurementFactCounters
    explicit_measurement_keys: set[RuntimeMeasurementFeatureKey]
    object_row_domain: RuntimeObjectMeasurementFactRowDomain
    required_keys: _RuntimeRequiredMeasurementKeys
    row_priority_cache: _RuntimeMeasurementRowPriorityCache


@dataclass(frozen=True, slots=True)
class _StaticWideRuntimeRowProjectionContext:
    """Context for projecting static-wide runtime measurement rows."""

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
    """Source context for deriving a runtime measurement feature key."""

    field_name: str
    subject: RuntimeMeasurementSubjectKey
    policy: RuntimeEquivalencePolicy
    qualifiers: tuple[str, ...]
    source_name: str | None
    known_source_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RuntimeMeasurementTableProjectionContext:
    """Context for projecting one runtime measurement table."""

    table: MeasurementTable
    policy: RuntimeEquivalencePolicy
    axis_key: object | None
    known_source_names: tuple[str, ...]
    required_keys: _RuntimeRequiredMeasurementKeys


@dataclass(frozen=True, slots=True)
class RuntimeRecordPlaneIdentity:
    """Plane identity contributed by runtime record scope rather than row payload."""

    slice_index: int
    authority: "RuntimeRecordPlaneIdentityAuthority"

    def object_instance_key(
        self,
        row: Mapping[str, Any],
        object_id: int,
        *,
        image_number_offset: float,
    ) -> ObjectInstanceKey:
        if (
            self.authority
            is RuntimeRecordPlaneIdentityAuthority.OVERRIDE_ROW_IDENTITY
        ):
            return ObjectInstanceKey(object_id, slice_index=self.slice_index)
        key = ObjectInstanceKey.from_measurement_row(
            row,
            object_id,
            image_number_offset=image_number_offset,
        )
        if key.slice_index is None:
            return ObjectInstanceKey(object_id, slice_index=self.slice_index)
        return key

    def relationship_for_projection(
        self,
        relationship: ObjectRelationship,
    ) -> ObjectRelationship:
        if (
            relationship.slice_indices
            and self.authority
            is RuntimeRecordPlaneIdentityAuthority.FILL_MISSING_ROW_IDENTITY
        ):
            return relationship
        source_ids = tuple(
            int(value) for value in np.asarray(relationship.source_ids).ravel()
        )
        slice_count = max(
            relationship.slice_count or 0,
            self.slice_index + 1,
        )
        return ObjectRelationship(
            name=relationship.name,
            source=relationship.source,
            target=relationship.target,
            source_ids=relationship.source_ids,
            target_ids=relationship.target_ids,
            relationship_type=relationship.relationship_type,
            slice_indices=tuple(self.slice_index for _ in source_ids),
            slice_count=slice_count,
        )


@dataclass(frozen=True, slots=True)
class RuntimeScopedMeasurementTable:
    """Measurement table plus runtime record-plane identity used for joins."""

    table: MeasurementTable
    plane_identity: RuntimeRecordPlaneIdentity | None = None

    def object_instance_key(
        self,
        row: Mapping[str, Any],
        object_id: int,
        *,
        image_number_offset: float,
    ) -> ObjectInstanceKey:
        if self.plane_identity is None:
            return ObjectInstanceKey.from_measurement_row(
                row,
                object_id,
                image_number_offset=image_number_offset,
            )
        return self.plane_identity.object_instance_key(
            row,
            object_id,
            image_number_offset=image_number_offset,
        )


class RuntimeRecordPlaneIdentityAuthority(Enum):
    """How runtime record identity should interact with row-local identity."""

    FILL_MISSING_ROW_IDENTITY = auto()
    OVERRIDE_ROW_IDENTITY = auto()


@dataclass(slots=True)
class RuntimeAxisGroupPlaneIndex:
    """Stable per-axis mapping from runtime group scope to plane identity."""

    indices_by_group_key: dict[object, int] = field(default_factory=dict)

    def slice_index_for_scope(self, scope: ArtifactScope) -> int | None:
        if scope.group_key is None:
            return None
        return self.indices_by_group_key.setdefault(
            scope.group_key,
            len(self.indices_by_group_key),
        )


@dataclass(slots=True)
class RuntimeAxisRepeatedArtifactPlaneIndex:
    """Assign plane identity to repeated records without distinct runtime scope."""

    next_index_by_artifact_key: dict[tuple[ArtifactKind, str], int] = field(
        default_factory=dict
    )

    def plane_identity_for_record(
        self,
        *,
        kind: ArtifactKind,
        name: str,
    ) -> RuntimeRecordPlaneIdentity:
        key = (kind, name)
        slice_index = self.next_index_by_artifact_key.get(key, 0)
        self.next_index_by_artifact_key[key] = slice_index + 1
        return RuntimeRecordPlaneIdentity(
            slice_index,
            RuntimeRecordPlaneIdentityAuthority.OVERRIDE_ROW_IDENTITY,
        )


@dataclass(slots=True)
class RuntimeAxisRecordPlaneIdentityResolver:
    """Resolve runtime record plane identity from scope and repeated artifacts."""

    repeated_artifact_counts: Counter[tuple[ArtifactKind, str]]
    group_plane_index: RuntimeAxisGroupPlaneIndex = field(
        default_factory=RuntimeAxisGroupPlaneIndex
    )
    repeated_artifact_plane_index: RuntimeAxisRepeatedArtifactPlaneIndex = field(
        default_factory=RuntimeAxisRepeatedArtifactPlaneIndex
    )

    @classmethod
    def from_records(
        cls,
        records: Iterable[object],
    ) -> "RuntimeAxisRecordPlaneIdentityResolver":
        return cls(
            Counter(
                (record.key.kind, record.key.name)
                for record in records
                if record.key.kind.participates_in_axis_plane_identity
            )
        )

    def plane_identity_for_record(
        self,
        *,
        kind: ArtifactKind,
        name: str,
        scope: ArtifactScope,
    ) -> RuntimeRecordPlaneIdentity | None:
        artifact_key = (kind, name)
        if self.repeated_artifact_counts[artifact_key] > 1:
            return self.repeated_artifact_plane_index.plane_identity_for_record(
                kind=kind,
                name=name,
            )
        slice_index = self.group_plane_index.slice_index_for_scope(scope)
        if slice_index is None:
            return None
        return RuntimeRecordPlaneIdentity(
            slice_index,
            RuntimeRecordPlaneIdentityAuthority.FILL_MISSING_ROW_IDENTITY,
        )

    def plane_identity_for_runtime_record(
        self,
        record: object,
    ) -> RuntimeRecordPlaneIdentity | None:
        """Resolve plane identity directly from a runtime artifact record."""
        return self.plane_identity_for_record(
            kind=record.key.kind,
            name=record.key.name,
            scope=record.key.scope,
        )


@dataclass(frozen=True, slots=True)
class RuntimeScopedObjectRelationship:
    """Object relationship plus runtime scope identity used for relationship joins."""

    relationship: ObjectRelationship
    plane_identity: RuntimeRecordPlaneIdentity | None = None

    def relationship_for_projection(self) -> ObjectRelationship:
        if self.plane_identity is None:
            return self.relationship
        return self.plane_identity.relationship_for_projection(self.relationship)


@dataclass(frozen=True, slots=True)
class _ObjectLabelMeasurementContext:
    labels: object
    object_name: str | None
    policy: RuntimeEquivalencePolicy
    values_by_feature: _RuntimeMeasurementFactCounterMapping
    object_identifier_subjects: frozenset[RuntimeMeasurementSubjectKey]
    object_location_subjects: frozenset[RuntimeMeasurementSubjectKey]
    object_count_subjects: frozenset[RuntimeMeasurementSubjectKey]
    required_keys: _RuntimeRequiredMeasurementKeys
    declared_object_count: int | None
    declared_object_ids: tuple[int, ...]
    declared_object_id_domains: tuple[tuple[int, ...], ...]
    domain_scope: ObjectLabelDomainScope
    object_location_aggregate_subjects: frozenset[
        RuntimeMeasurementSubjectKey
    ] = frozenset()

    @classmethod
    def from_runtime_value(
        cls,
        value: RuntimeValue,
        policy: RuntimeEquivalencePolicy,
        values_by_feature: _RuntimeMeasurementFactCounterMapping,
        object_identifier_subjects: frozenset[RuntimeMeasurementSubjectKey],
        object_location_subjects: frozenset[RuntimeMeasurementSubjectKey],
        object_count_subjects: frozenset[RuntimeMeasurementSubjectKey],
        required_keys: _RuntimeRequiredMeasurementKeys,
        object_location_aggregate_subjects: frozenset[
            RuntimeMeasurementSubjectKey
        ] = frozenset(),
    ) -> "_ObjectLabelMeasurementContext":
        label_set = ObjectLabelSet.from_runtime_value(value)
        return cls(
            label_set.labels,
            value.schema.object_name or label_set.name,
            policy,
            values_by_feature,
            object_identifier_subjects,
            object_location_subjects,
            object_count_subjects,
            required_keys,
            label_set.declared_object_count,
            label_set.declared_object_ids,
            label_set.declared_object_id_domains,
            label_set.domain_scope,
            object_location_aggregate_subjects,
        )

    def dense_identity_domains(self) -> tuple[tuple[int, ...], ...]:
        """Return dense object identity domains for this object-label payload."""
        return dense_object_label_identity_domains(
            self.labels,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
        )

    def dense_plane_id_domains(self) -> tuple[tuple[int, ...], ...]:
        """Return dense object ID domains aligned to measurement planes."""
        return dense_object_label_plane_id_domains(
            self.labels,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
        )


@dataclass(frozen=True, slots=True)
class ObjectLabelMeasurementProjection:
    """One semantic object-label measurement domain."""

    labels: np.ndarray
    object_ids: tuple[int, ...]
    slice_index: int | None = None


@dataclass(frozen=True, slots=True)
class RuntimeObjectCountAuthority:
    """Nominal ownership of object-count facts before row fallback is allowed."""

    declared_subjects: frozenset[RuntimeMeasurementSubjectKey] = frozenset()

    @classmethod
    def from_object_label_records(
        cls,
        records: Iterable[object],
    ) -> "RuntimeObjectCountAuthority":
        subjects: set[RuntimeMeasurementSubjectKey] = set()
        for record in records:
            object_labels = ObjectLabelSet.from_runtime_value(record.value)
            domain = object_labels.object_label_domain()
            if (
                domain.declared_object_count is None
                and not domain.declared_object_ids
                and not domain.declared_object_id_domains
            ):
                continue
            subjects.add(
                RuntimeMeasurementSubjectKey(
                    MeasurementScope.OBJECT,
                    object_labels.name,
                )
            )
        return cls(frozenset(subjects))

    def primary_row_reserved_subjects(
        self,
        explicit_subjects: frozenset[RuntimeMeasurementSubjectKey],
    ) -> frozenset[RuntimeMeasurementSubjectKey]:
        """Return subjects unavailable to primary-row object-count fallback."""

        return explicit_subjects | self.declared_subjects


class ObjectLabelMeasurementProjectionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Project object-label payloads into measurement domains by declared scope."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True

    scope: ClassVar[ObjectLabelDomainScope]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_scope(
        cls,
        scope: ObjectLabelDomainScope,
    ) -> "ObjectLabelMeasurementProjectionStrategy":
        matches = tuple(
            strategy_type()
            for strategy_type in cls.__registry__.values()
            if strategy_type.scope is scope
        )
        if len(matches) != 1:
            names = tuple(strategy.strategy_label for strategy in matches)
            raise ValueError(
                "Object-label measurement projection requires exactly one "
                f"strategy for {scope.value!r}, got {names!r}."
            )
        return matches[0]

    @abstractmethod
    def projections(
        self,
        context: _ObjectLabelMeasurementContext,
    ) -> tuple[ObjectLabelMeasurementProjection, ...]:
        """Return semantic label domains used for count/location facts."""

    @staticmethod
    def label_array(context: _ObjectLabelMeasurementContext) -> np.ndarray | None:
        return _runtime_object_label_array(context.labels)


class PayloadObjectLabelMeasurementProjectionStrategy(
    ObjectLabelMeasurementProjectionStrategy
):
    """Payload-scoped labels measure the dense object payload as one domain."""

    scope = ObjectLabelDomainScope.PAYLOAD
    strategy_label = ObjectLabelDomainScope.PAYLOAD.value

    def projections(
        self,
        context: _ObjectLabelMeasurementContext,
    ) -> tuple[ObjectLabelMeasurementProjection, ...]:
        label_array = self.label_array(context)
        if label_array is None:
            return ()
        domains = context.dense_identity_domains()
        if len(domains) != 1:
            raise ValueError(
                "Payload-scoped object labels must project to exactly one "
                f"measurement domain, got {len(domains)}."
            )
        return (ObjectLabelMeasurementProjection(label_array, domains[0]),)


class PlaneObjectLabelMeasurementProjectionStrategy(
    ObjectLabelMeasurementProjectionStrategy
):
    """Plane-scoped labels measure each dense XY plane independently."""

    scope = ObjectLabelDomainScope.PLANE
    strategy_label = ObjectLabelDomainScope.PLANE.value

    def projections(
        self,
        context: _ObjectLabelMeasurementContext,
    ) -> tuple[ObjectLabelMeasurementProjection, ...]:
        label_array = self.label_array(context)
        if label_array is None:
            return ()
        planes = (label_array,) if label_array.ndim <= 2 else tuple(label_array)
        plane_indexes = (None,) if label_array.ndim <= 2 else tuple(range(len(planes)))
        domains = context.dense_plane_id_domains()
        return tuple(
            ObjectLabelMeasurementProjection(plane, object_ids, plane_index)
            for plane, object_ids, plane_index in zip(
                planes,
                domains,
                plane_indexes,
                strict=True,
            )
        )


class ObjectIdentifierDomainProjectionStrategy(
    MostDerivedContextStrategyMixin[_ObjectLabelMeasurementContext],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project object-label domains into ObjectNumber measurement rows."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True

    strategy_key: ClassVar[str | None] = None

    @abstractmethod
    def domains(
        self,
        context: _ObjectLabelMeasurementContext,
    ) -> tuple[tuple[int, ...], ...]:
        """Return ObjectNumber domains to emit for ``context``."""


class PayloadObjectIdentifierDomainProjectionStrategy(
    ObjectIdentifierDomainProjectionStrategy
):
    """Undeclared payload labels export one identity row per represented object."""

    strategy_key = "payload_identity_domains"

    def matches(self, context: _ObjectLabelMeasurementContext) -> bool:
        del context
        return True

    def domains(
        self,
        context: _ObjectLabelMeasurementContext,
    ) -> tuple[tuple[int, ...], ...]:
        return context.dense_identity_domains()


class DeclaredObjectIdentifierPlaneDomainProjectionStrategy(
    PayloadObjectIdentifierDomainProjectionStrategy
):
    """Declared object domains are exported once for each measurement plane."""

    strategy_key = "declared_plane_domains"

    def matches(self, context: _ObjectLabelMeasurementContext) -> bool:
        return (
            context.declared_object_count is not None
            or bool(context.declared_object_ids)
            or bool(context.declared_object_id_domains)
        )

    def domains(
        self,
        context: _ObjectLabelMeasurementContext,
    ) -> tuple[tuple[int, ...], ...]:
        return context.dense_plane_id_domains()


@dataclass(frozen=True, slots=True)
class RuntimeExpectedMeasurementFactCompletion:
    """Materialize expected facts missing from explicit runtime measurements."""

    expected_by_key: _RuntimeMeasurementFactCounters
    values_by_feature: _RuntimeMeasurementFactCounterMapping

    def missing_facts(self) -> _RuntimeMeasurementFacts:
        facts: _RuntimeMeasurementFactList = []
        for key, expected_counter in self.expected_by_key.items():
            explicit_counter = self.values_by_feature.get(key, Counter())
            for signature, expected_count in expected_counter.items():
                missing_count = expected_count - explicit_counter[signature]
                if missing_count <= 0:
                    continue
                facts.extend((key, signature) for _index in range(missing_count))
        return tuple(facts)


@dataclass(frozen=True, slots=True)
class ObjectLabelIdentifierMeasurementCompletion:
    """Complete ObjectNumber facts from all nominal object-label domains."""

    policy: RuntimeEquivalencePolicy
    values_by_feature: _RuntimeMeasurementFactCounterMapping
    object_identifier_subjects: frozenset[RuntimeMeasurementSubjectKey]
    object_location_subjects: frozenset[RuntimeMeasurementSubjectKey]
    object_count_subjects: frozenset[RuntimeMeasurementSubjectKey]
    required_keys: _RuntimeRequiredMeasurementKeys

    def facts_for_records(
        self,
        records: Sequence[Any],
    ) -> _RuntimeMeasurementFacts:
        expected_by_key: _RuntimeMeasurementFactCounters = {}
        for record in records:
            self._add_expected_record_counts(expected_by_key, record)
        return RuntimeExpectedMeasurementFactCompletion(
            expected_by_key,
            self.values_by_feature,
        ).missing_facts()

    def _add_expected_record_counts(
        self,
        expected_by_key: _RuntimeMeasurementFactCounters,
        record: Any,
    ) -> None:
        context = _ObjectLabelMeasurementContext.from_runtime_value(
            record.value,
            self.policy,
            self.values_by_feature,
            self.object_identifier_subjects,
            self.object_location_subjects,
            self.object_count_subjects,
            self.required_keys,
        )
        if context.object_name is None:
            return
        subject = RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            context.object_name,
        )
        keys = RequiredRuntimeMeasurementProjection(
            context.required_keys,
            context.policy,
        ).object_identifier_keys(subject)
        if not keys:
            return
        if _runtime_object_label_array(context.labels) is None:
            return
        object_number_domains = ObjectIdentifierDomainProjectionStrategy.for_context(
            context
        ).domains(context)
        for object_ids in object_number_domains:
            for key in keys:
                counter = expected_by_key.setdefault(key, Counter())
                for object_id in object_ids:
                    counter[_cell_signature(str(object_id), context.policy)] += 1


@dataclass(frozen=True, slots=True)
class PrimaryRowObjectIdentifierMeasurementCompletion:
    """Complete ObjectNumber facts from explicit primary object measurement rows."""

    policy: RuntimeEquivalencePolicy
    values_by_feature: _RuntimeMeasurementFactCounterMapping
    object_identifier_subjects: frozenset[RuntimeMeasurementSubjectKey]
    required_keys: _RuntimeRequiredMeasurementKeys

    def facts_for_rows(
        self,
        primary_row_identities: Iterable[_RuntimeMeasurementPrimaryRowKey],
    ) -> _RuntimeMeasurementFacts:
        expected_by_key: _RuntimeMeasurementFactCounters = {}
        for subject, row_identity in primary_row_identities:
            self.add_expected_row_counts(expected_by_key, subject, row_identity)
        return RuntimeExpectedMeasurementFactCompletion(
            expected_by_key,
            self.values_by_feature,
        ).missing_facts()

    def add_expected_row_counts(
        self,
        expected_by_key: _RuntimeMeasurementFactCounters,
        subject: RuntimeMeasurementSubjectKey,
        row_identity: _RuntimeMeasurementRowIdentity,
    ) -> None:
        if subject in self.object_identifier_subjects:
            return
        object_label = RuntimeObjectMeasurementRowIdentity(
            row_identity
        ).object_label_signature
        if object_label is None:
            return
        for key in RequiredRuntimeMeasurementProjection(
            self.required_keys,
            self.policy,
        ).object_identifier_keys(subject):
            expected_by_key.setdefault(key, Counter())[object_label] += 1


@dataclass(frozen=True, slots=True)
class _MeasurementScopeAggregatePolicy:
    """Aggregate-feature parsing policy for one closed measurement scope."""

    scope: MeasurementScope
    accepts_aggregate_feature_key: bool


def _measurement_scope_aggregate_policy_by_scope(
    rows: tuple[_MeasurementScopeAggregatePolicy, ...],
) -> Mapping[MeasurementScope, _MeasurementScopeAggregatePolicy]:
    by_scope = {row.scope: row for row in rows}
    if set(by_scope) != set(MeasurementScope):
        missing = tuple(
            sorted(scope.value for scope in set(MeasurementScope) - set(by_scope))
        )
        extra = tuple(
            sorted(scope.value for scope in set(by_scope) - set(MeasurementScope))
        )
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
def _aggregate_mean_key(
    key: RuntimeMeasurementFeatureKey,
    *,
    required_keys: _RuntimeRequiredMeasurementKeys,
    key_cache: _AggregateMeanKeyCache,
) -> RuntimeMeasurementFeatureKey | None:
    cache_key = (
        key.subject.scope,
        key.subject.name,
        key.feature_name,
        key.statistic,
        key.source_name,
    )
    mean_key = key_cache.get(cache_key, _CACHE_MISS)
    if mean_key is _CACHE_MISS:
        mean_key = RuntimeMeasurementFeatureKeyFactory(
            key.subject,
            key.feature_name,
            MeasurementStatistic.MEAN.value,
            source_name=key.source_name,
        ).key()
        if required_keys is not None and mean_key not in required_keys:
            mean_key = None
        elif _is_image_number_reference_feature(key):
            mean_key = None
        key_cache[cache_key] = mean_key
    return mean_key


def _finite_numeric_runtime_cell_value(value: RuntimeCellSignature) -> float | None:
    if value.kind is not RuntimeCellValueKind.NUMBER:
        return None
    numeric_value = float(value.value)
    return numeric_value if math.isfinite(numeric_value) else None

_TIE_SENSITIVE_LOCATION_FEATURES = frozenset(
    ("max_intensity_x", "max_intensity_y", "max_intensity_z")
)
_LOCATION_VALUE_FEATURE_BY_NAME = MappingProxyType(
    {
        "max_intensity_x": "max_intensity",
        "max_intensity_y": "max_intensity",
        "max_intensity_z": "max_intensity",
    }
)


@dataclass(frozen=True, slots=True)
class TieSensitiveLocationValueFeatureContext:
    """Typed context for resolving max-location dependency features."""

    feature: RuntimeMeasurementFeatureKey
    policy: RuntimeEquivalencePolicy

    @property
    def feature_parts(self) -> tuple[str, ...]:
        return tuple(
            part
            for part in _normalize_identifier(self.feature.feature_name).split("_")
            if part
        )

    def location_feature_family(
        self,
        feature_name: str,
    ) -> Any | None:
        """Return the source-qualified location family for one feature name."""
        return self.policy.measurement_dialect.source_qualified_feature_family(
            feature_name,
            self.feature.source_name,
            self.feature.subject.scope,
            _TIE_SENSITIVE_LOCATION_FEATURES,
        )

    def value_key_from_location_family(
        self,
        feature_family: Any,
    ) -> RuntimeMeasurementFeatureKey:
        """Return direct value key for one source-qualified location family."""
        value_family = _LOCATION_VALUE_FEATURE_BY_NAME[feature_family.feature_name]
        value_identity = self.policy.measurement_dialect.encode_source_qualified_feature(
            value_family,
            feature_family.source_name,
            self.feature.subject.scope,
        )
        return RuntimeMeasurementFeatureKey(
            subject=self.feature.subject,
            feature_name=value_identity.feature_name,
            statistic=self.feature.statistic,
            source_name=value_identity.source_name,
        )


@dataclass(frozen=True, slots=True)
class AggregateTieSensitiveLocationFeature:
    """Parsed aggregate max-location feature identity."""

    aggregate: str
    object_name_parts: tuple[str, ...]
    feature_parts: tuple[str, ...]
    feature_family: Any

    @classmethod
    def from_context(
        cls,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> "AggregateTieSensitiveLocationFeature | None":
        """Parse an aggregate location feature, if the context declares one."""
        for identity in RuntimeAggregateFeatureIdentity.candidates_from_parts(
            context.feature_parts
        ):
            feature_family = context.location_feature_family(identity.feature_name)
            if feature_family is None:
                continue
            return cls(
                aggregate=identity.aggregate,
                object_name_parts=identity.object_name_parts,
                feature_parts=identity.feature_parts,
                feature_family=feature_family,
            )
        return None

    @property
    def feature_name(self) -> str:
        return "_".join(self.feature_parts)

    def value_key_from_location_family(
        self,
        context: TieSensitiveLocationValueFeatureContext,
        feature_family: Any,
    ) -> RuntimeMeasurementFeatureKey:
        """Return aggregate value key corresponding to a max-location aggregate."""
        direct_value_key = context.value_key_from_location_family(feature_family)
        return RuntimeMeasurementFeatureKey(
            subject=context.feature.subject,
            feature_name="_".join(
                (
                    self.aggregate,
                    *self.object_name_parts,
                    direct_value_key.feature_name,
                )
            ),
            statistic=context.feature.statistic,
            source_name=direct_value_key.source_name,
        )


class TieSensitiveLocationValueFeatureStrategy(
    MostDerivedContextStrategyMixin[TieSensitiveLocationValueFeatureContext],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered resolver for location-feature value dependencies."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True

    strategy_key: ClassVar[str | None] = None

    @classmethod
    def value_feature_key(
        cls,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> RuntimeMeasurementFeatureKey | None:
        """Resolve value dependency through most-derived registered strategy."""
        strategy = cls.for_context(
            context,
            required=False,
            error_subject="Tie-sensitive location value feature resolution",
        )
        return None if strategy is None else strategy.resolve(context)

    @abstractmethod
    def resolve(
        self,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> RuntimeMeasurementFeatureKey:
        """Return the value feature that gates location tie mismatches."""


class DirectTieSensitiveLocationValueFeatureStrategy(
    TieSensitiveLocationValueFeatureStrategy
):
    """Resolve direct max-location features to direct max-intensity values."""

    strategy_key = "direct"

    def matches(
        self,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> bool:
        return context.location_feature_family(context.feature.feature_name) is not None

    def resolve(
        self,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> RuntimeMeasurementFeatureKey:
        feature_family = context.location_feature_family(context.feature.feature_name)
        if feature_family is None:
            raise ValueError("Direct tie-sensitive location strategy lost ownership.")
        return context.value_key_from_location_family(feature_family)


class AggregateTieSensitiveLocationValueFeatureStrategy(
    TieSensitiveLocationValueFeatureStrategy
):
    """Resolve aggregate max-location features to aggregate max-intensity values."""

    strategy_key = "aggregate"

    def matches(
        self,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> bool:
        return AggregateTieSensitiveLocationFeature.from_context(context) is not None

    def resolve(
        self,
        context: TieSensitiveLocationValueFeatureContext,
    ) -> RuntimeMeasurementFeatureKey:
        aggregate_feature = AggregateTieSensitiveLocationFeature.from_context(context)
        if aggregate_feature is None:
            raise ValueError("Aggregate tie-sensitive location strategy lost ownership.")
        return aggregate_feature.value_key_from_location_family(
            context,
            aggregate_feature.feature_family,
        )


@dataclass(frozen=True, slots=True)
class TieSensitiveLocationFeatureContract:
    """Semantic value dependency for tie-sensitive max-location features."""

    feature: RuntimeMeasurementFeatureKey
    policy: RuntimeEquivalencePolicy

    def value_feature_key(self) -> RuntimeMeasurementFeatureKey | None:
        """Return the value feature that makes this location tie-insensitive."""
        return TieSensitiveLocationValueFeatureStrategy.value_feature_key(
            TieSensitiveLocationValueFeatureContext(
                feature=self.feature,
                policy=self.policy,
            )
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
class ShapeDescriptorFeatureContext:
    """Measurement feature context for unstable shape descriptor equivalence."""

    feature: RuntimeMeasurementFeatureKey
    policy: RuntimeEquivalencePolicy

    def semantic_context(self) -> "ShapeDescriptorFeatureContext":
        """Return the descriptor context that owns comparison semantics."""
        child_feature_name = (
            RelationshipAggregateFeatureSemantics.aggregate_child_feature_name_from_key(
                self.feature,
                self.policy.measurement_dialect,
            )
        )
        if child_feature_name is None:
            return self
        return ShapeDescriptorFeatureContext(
            RuntimeMeasurementFeatureKey(
                subject=self.feature.subject,
                feature_name=child_feature_name,
                statistic=self.feature.statistic,
                source_name=self.feature.source_name,
            ),
            self.policy,
        )


class ShapeDescriptorFeatureSemantics(
    MostDerivedContextStrategyMixin[ShapeDescriptorFeatureContext],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Classify direct and derived shape descriptor measurement features."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True

    strategy_key: ClassVar[str | None] = None

    @abstractmethod
    def matches(self, context: ShapeDescriptorFeatureContext) -> bool:
        """Return whether this strategy owns the feature."""

    @abstractmethod
    def descriptor_feature_name(self, context: ShapeDescriptorFeatureContext) -> str:
        """Return the underlying child/object descriptor feature name."""

    @abstractmethod
    def values_equivalent(
        self,
        context: ShapeDescriptorFeatureContext,
        reference_values: Counter[RuntimeCellSignature],
        candidate_values: Counter[RuntimeCellSignature],
    ) -> bool:
        """Return whether descriptor values are equivalent under policy."""

    def boundary_jitter_values_equivalent(
        self,
        context: ShapeDescriptorFeatureContext,
        reference_values: Counter[RuntimeCellSignature],
        candidate_values: Counter[RuntimeCellSignature],
    ) -> bool:
        """Return sparse-boundary equivalence for descriptor values."""
        return _sparse_numeric_counters_equivalent(
            reference_values,
            candidate_values,
            context.policy,
            abs_tolerance=context.policy.object_boundary_jitter_abs_tolerance,
            rel_tolerance=context.policy.object_boundary_jitter_rel_tolerance,
            max_unstable_values=context.policy.object_boundary_jitter_max_unstable_values,
            max_unstable_fraction=context.policy.object_boundary_jitter_max_unstable_fraction,
        )


class OrientationShapeDescriptorFeatureSemantics(ShapeDescriptorFeatureSemantics):
    """Orientation descriptors compare as angular boundary-sensitive values."""

    strategy_key = "orientation_descriptor"

    def matches(self, context: ShapeDescriptorFeatureContext) -> bool:
        return context.feature.feature_name in _ORIENTATION_FEATURES

    def descriptor_feature_name(self, context: ShapeDescriptorFeatureContext) -> str:
        return context.feature.feature_name

    def values_equivalent(
        self,
        context: ShapeDescriptorFeatureContext,
        reference_values: Counter[RuntimeCellSignature],
        candidate_values: Counter[RuntimeCellSignature],
    ) -> bool:
        if context.policy.allow_sparse_object_boundary_jitter:
            return _sparse_absolute_numeric_counters_equivalent(
                reference_values,
                candidate_values,
                context.policy,
                abs_tolerance=context.policy.object_boundary_jitter_abs_tolerance,
                rel_tolerance=context.policy.object_boundary_jitter_rel_tolerance,
                max_unstable_values=context.policy.object_boundary_jitter_max_unstable_values,
                max_unstable_fraction=context.policy.object_boundary_jitter_max_unstable_fraction,
            )
        return _absolute_numeric_counters_equivalent(
            reference_values,
            candidate_values,
            context.policy,
        )

class ShapeZernikeDescriptorFeatureSemantics(ShapeDescriptorFeatureSemantics):
    """Shape Zernike descriptors compare with shape-descriptor tolerance policy."""

    strategy_key = "shape_zernike_descriptor"

    def matches(self, context: ShapeDescriptorFeatureContext) -> bool:
        descriptor = IndexedObjectZernikeDescriptor.from_feature_name(
            context.feature.feature_name
        )
        return (
            descriptor is not None
            and descriptor.family is ObjectZernikeDescriptorFeature.SHAPE
        )

    def descriptor_feature_name(self, context: ShapeDescriptorFeatureContext) -> str:
        return context.feature.feature_name

    def values_equivalent(
        self,
        context: ShapeDescriptorFeatureContext,
        reference_values: Counter[RuntimeCellSignature],
        candidate_values: Counter[RuntimeCellSignature],
    ) -> bool:
        return _sparse_numeric_counters_equivalent(
            reference_values,
            candidate_values,
            context.policy,
            abs_tolerance=context.policy.shape_descriptor_abs_tolerance,
            rel_tolerance=context.policy.shape_descriptor_rel_tolerance,
            max_unstable_values=context.policy.shape_descriptor_max_unstable_values,
            max_unstable_fraction=context.policy.shape_descriptor_max_unstable_fraction,
        )

class ObjectZernikeDescriptorStabilityContract(
    EnumKeyedStrategyMixin[ObjectZernikeDescriptorFeature],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declare which object-domain facts make a Zernike descriptor comparable."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "descriptor_family"

    descriptor_family: ClassVar[ObjectZernikeDescriptorFeature]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def is_stable(
        self,
        feature: RuntimeMeasurementFeatureKey,
        descriptor: IndexedObjectZernikeDescriptor,
        reference: "RuntimeMeasurementSnapshot",
        candidate: "RuntimeMeasurementSnapshot",
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return whether the descriptor's supporting object domain is stable."""


class ShapeObjectZernikeDescriptorStabilityContract(
    ObjectZernikeDescriptorStabilityContract
):
    """Shape Zernikes are comparable only when shape geometry is stable."""

    descriptor_family = ObjectZernikeDescriptorFeature.SHAPE

    def is_stable(
        self,
        feature: RuntimeMeasurementFeatureKey,
        descriptor: IndexedObjectZernikeDescriptor,
        reference: "RuntimeMeasurementSnapshot",
        candidate: "RuntimeMeasurementSnapshot",
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        return MeasurementFeatureStabilityPolicy(
            feature,
            reference,
            candidate,
            policy,
        ).shape_descriptor_geometry_is_stable()


class IntensityObjectZernikeDescriptorStabilityContract(
    ObjectZernikeDescriptorStabilityContract
):
    """Intensity Zernikes are comparable when object identity and centers are stable."""

    descriptor_family: ClassVar[ObjectZernikeDescriptorFeature]
    strategy_label = None

    def is_stable(
        self,
        feature: RuntimeMeasurementFeatureKey,
        descriptor: IndexedObjectZernikeDescriptor,
        reference: "RuntimeMeasurementSnapshot",
        candidate: "RuntimeMeasurementSnapshot",
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        return (
            MeasurementFeatureStabilityPolicy(
                feature,
                reference,
                candidate,
                policy,
            ).object_count_values_stable()
            and MeasurementFeatureStabilityPolicy(
                feature,
                reference,
                candidate,
                policy,
            ).object_measurement_role_values_stable(
                ObjectMeasurementFeatureRole.IDENTIFIER,
                min_stable_features=1,
            )
            and MeasurementFeatureStabilityPolicy(
                feature,
                reference,
                candidate,
                policy,
            ).object_measurement_role_values_stable(
                ObjectMeasurementFeatureRole.LOCATION,
                min_stable_features=2,
            )
        )


class IntensityMagnitudeObjectZernikeDescriptorStabilityContract(
    IntensityObjectZernikeDescriptorStabilityContract
):
    """Magnitude intensity Zernikes share the intensity object-domain contract."""

    descriptor_family = ObjectZernikeDescriptorFeature.INTENSITY_MAGNITUDE


class IntensityPhaseObjectZernikeDescriptorStabilityContract(
    IntensityObjectZernikeDescriptorStabilityContract
):
    """Phase intensity Zernikes share the intensity object-domain contract."""

    descriptor_family = ObjectZernikeDescriptorFeature.INTENSITY_PHASE


@dataclass(slots=True)
class RuntimeMeasurementObservationProjector:
    """Project runtime artifact observations into mutable measurement facts."""

    observation: RuntimeArtifactExecutionObservation
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy()
    known_source_names: tuple[str, ...] = ()
    required_measurement_keys: _RuntimeRequiredMeasurementKeys = None
    values_by_feature: _RuntimeMeasurementFactCounters = field(init=False)
    row_merge_cache: _RuntimeMeasurementRowMergeCache = field(init=False)
    primary_row_identities: _RuntimeMeasurementPrimaryRowSet | None = field(init=False)
    object_row_domain: RuntimeObjectMeasurementFactRowDomain = field(init=False)
    object_label_records: list[object] = field(init=False)
    object_label_records_by_axis: dict[object, list[object]] = field(init=False)
    relationship_records_by_axis: dict[
        object,
        list[RuntimeScopedObjectRelationship],
    ] = field(init=False)
    measurement_tables_by_axis: dict[
        object,
        list[RuntimeScopedMeasurementTable],
    ] = field(init=False)
    _seen_aggregate_measurement_tables: set[RuntimeMeasurementTableIdentity] = field(
        init=False
    )

    def __post_init__(self) -> None:
        cache_plan = RuntimeMeasurementProjectionCachePlan(
            self.required_measurement_keys,
            self.policy,
        )
        self.values_by_feature = {}
        self.row_merge_cache = cache_plan.row_merge_cache()
        self.primary_row_identities = cache_plan.primary_row_identities()
        self.object_row_domain = RuntimeObjectMeasurementFactRowDomain()
        self.object_label_records = []
        self.object_label_records_by_axis = {}
        self.relationship_records_by_axis = {}
        self.measurement_tables_by_axis = {}
        self._seen_aggregate_measurement_tables = set()

    def record_artifacts(self) -> None:
        for axis_key, records in self.observation.records_by_axis.items():
            self._record_axis(axis_key, tuple(records))

    def _record_axis(
        self,
        axis_key: object,
        axis_records: tuple[object, ...],
    ) -> None:
        plane_identity_resolver = RuntimeAxisRecordPlaneIdentityResolver.from_records(
            axis_records
        )
        seen_exact_measurement_tables: set[tuple[object, ...]] = set()
        seen_object_subtables: _RuntimeMeasurementObjectSubtableSet = set()
        for record in axis_records:
            if record.key.kind is ArtifactKind.SPATIAL_GRID:
                self._record_spatial_grid(record)
                continue
            if record.key.kind is ArtifactKind.MEASUREMENTS:
                self._record_measurement_table(
                    axis_key,
                    record,
                    plane_identity_resolver,
                    seen_exact_measurement_tables,
                    seen_object_subtables,
                )
                continue
            if record.key.kind is ArtifactKind.OBJECT_LABELS:
                self.object_label_records.append(record)
                self.object_label_records_by_axis.setdefault(axis_key, []).append(
                    record
                )
                continue
            if record.key.kind is ArtifactKind.RELATIONSHIPS:
                self._record_relationship(
                    axis_key,
                    record,
                    plane_identity_resolver,
                )

    def _record_spatial_grid(self, record: object) -> None:
        record_measurement_facts(
            self.values_by_feature,
            spatial_grid_measurement_facts(record.value, self.policy),
            required_keys=self.required_measurement_keys,
        )

    def _record_measurement_table(
        self,
        axis_key: object,
        record: object,
        plane_identity_resolver: RuntimeAxisRecordPlaneIdentityResolver,
        seen_exact_measurement_tables: set[tuple[object, ...]],
        seen_object_subtables: _RuntimeMeasurementObjectSubtableSet,
    ) -> None:
        table = MeasurementTable.from_runtime_value(record.value)
        if record.key.scope.group_key is None:
            exact_table_key = exact_measurement_table_key(table)
            if exact_table_key in seen_exact_measurement_tables:
                return
            seen_exact_measurement_tables.add(exact_table_key)
            table = _dedupe_runtime_measurement_table_object_subtable(
                table,
                seen_object_subtables,
            )
        if record.key.scope.group_key is not None:
            aggregate_table_key = aggregate_measurement_table_key(table)
            if aggregate_table_key is not None:
                if aggregate_table_key in self._seen_aggregate_measurement_tables:
                    return
                self._seen_aggregate_measurement_tables.add(aggregate_table_key)
        plane_identity = plane_identity_resolver.plane_identity_for_runtime_record(
            record
        )
        self.measurement_tables_by_axis.setdefault(axis_key, []).append(
            RuntimeScopedMeasurementTable(
                table,
                plane_identity=plane_identity,
            )
        )
        table_projection_context = _RuntimeMeasurementTableProjectionContext(
            table,
            self.policy,
            axis_key,
            self.known_source_names,
            self.required_measurement_keys,
        )
        if _record_static_wide_runtime_measurement_table(
            self.values_by_feature,
            table_projection_context,
            row_merge_cache=self.row_merge_cache,
            primary_row_identities=self.primary_row_identities,
            object_row_domain=self.object_row_domain,
        ):
            return
        RuntimeMeasurementTableFactRecorder(
            self.values_by_feature,
            table_projection_context,
            row_merge_cache=self.row_merge_cache,
            primary_row_identities=self.primary_row_identities,
            object_row_domain=self.object_row_domain,
        ).record()

    def _record_relationship(
        self,
        axis_key: object,
        record: object,
        plane_identity_resolver: RuntimeAxisRecordPlaneIdentityResolver,
    ) -> None:
        plane_identity = plane_identity_resolver.plane_identity_for_runtime_record(
            record
        )
        self.relationship_records_by_axis.setdefault(axis_key, []).append(
            RuntimeScopedObjectRelationship(
                ObjectRelationship.from_runtime_value(record.value),
                plane_identity=plane_identity,
            )
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementSnapshot:
    """Semantic measurement facts independent of table layout."""

    values_by_feature: _RuntimeMeasurementFactCounterMapping

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
            record_measurement_facts(
                values_by_feature,
                RuntimeTableSnapshotFactExtractor(
                    table,
                    known_source_names=known_source_names,
                    policy=policy,
                ).measurement_facts(),
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
        projector = RuntimeMeasurementObservationProjector(
            observation,
            policy=policy,
            known_source_names=known_source_names,
            required_measurement_keys=required_measurement_keys,
        )
        projector.record_artifacts()
        values_by_feature = projector.values_by_feature
        row_merge_cache = projector.row_merge_cache
        primary_row_identities = projector.primary_row_identities
        object_row_domain = projector.object_row_domain
        object_label_records = projector.object_label_records
        object_label_records_by_axis = projector.object_label_records_by_axis
        relationship_records_by_axis = projector.relationship_records_by_axis
        measurement_tables_by_axis = projector.measurement_tables_by_axis
        object_identifier_subjects = object_measurement_subjects_with_role(
            values_by_feature,
            ObjectMeasurementFeatureRole.IDENTIFIER,
            policy.measurement_dialect,
        )
        object_location_aggregate_subjects = (
            RuntimeObjectLocationRowMergeContract.registered_projection(
                RuntimeObjectLocationRowMergeProjectionKey.AGGREGATE_LOCATION,
                policy,
            ).subjects(row_merge_cache)
        )
        object_location_subjects = object_measurement_subjects_with_role(
            values_by_feature,
            ObjectMeasurementFeatureRole.LOCATION,
            policy.measurement_dialect,
        ) | RuntimeObjectLocationRowMergeContract.registered_projection(
            RuntimeObjectLocationRowMergeProjectionKey.LOCATION,
            policy,
        ).subjects(row_merge_cache)
        explicit_object_count_subjects = object_measurement_subjects_with_role(
            values_by_feature,
            ObjectMeasurementFeatureRole.COUNT,
            policy.measurement_dialect,
        )
        object_count_authority = RuntimeObjectCountAuthority.from_object_label_records(
            object_label_records
        )
        _record_runtime_row_merge_facts(
            values_by_feature,
            row_merge_cache,
            required_keys=required_measurement_keys,
            policy=policy,
            primary_row_identities=primary_row_identities or set(),
            object_row_domain=object_row_domain,
        )
        object_location_subjects = object_measurement_subjects_with_role(
            values_by_feature,
            ObjectMeasurementFeatureRole.LOCATION,
            policy.measurement_dialect,
        ) | RuntimeObjectLocationRowMergeContract.registered_projection(
            RuntimeObjectLocationRowMergeProjectionKey.LOCATION,
            policy,
        ).subjects(row_merge_cache)
        record_measurement_facts(
            values_by_feature,
            _primary_row_object_count_measurement_facts(
                primary_row_identities,
                row_merge_cache,
                policy,
                existing_subjects=object_count_authority.primary_row_reserved_subjects(
                    explicit_object_count_subjects
                ),
                required_keys=required_measurement_keys,
            ) if primary_row_identities is not None else (),
            required_keys=required_measurement_keys,
        )
        object_count_subjects = explicit_object_count_subjects | object_measurement_subjects_with_role(
            values_by_feature,
            ObjectMeasurementFeatureRole.COUNT,
            policy.measurement_dialect,
        )
        record_measurement_facts(
            values_by_feature,
            PrimaryRowObjectIdentifierMeasurementCompletion(
                policy=policy,
                values_by_feature=values_by_feature,
                object_identifier_subjects=object_identifier_subjects,
                required_keys=required_measurement_keys,
            ).facts_for_rows(object_row_domain.primary_row_keys()),
            required_keys=required_measurement_keys,
        )
        object_identifier_subjects = object_measurement_subjects_with_role(
            values_by_feature,
            ObjectMeasurementFeatureRole.IDENTIFIER,
            policy.measurement_dialect,
        )
        record_measurement_facts(
            values_by_feature,
            ObjectLabelIdentifierMeasurementCompletion(
                policy=policy,
                values_by_feature=values_by_feature,
                object_identifier_subjects=object_identifier_subjects,
                object_location_subjects=object_location_subjects,
                object_count_subjects=object_count_subjects,
                required_keys=required_measurement_keys,
            ).facts_for_records(object_label_records),
            required_keys=required_measurement_keys,
        )
        for record in object_label_records:
            record_measurement_facts(
                values_by_feature,
                RuntimeObjectLabelMeasurementFactProjector(
                    _ObjectLabelMeasurementContext.from_runtime_value(
                        record.value,
                        policy,
                        values_by_feature,
                        object_identifier_subjects,
                        object_location_subjects,
                        object_count_subjects,
                        required_measurement_keys,
                        object_location_aggregate_subjects,
                    ),
                ).facts(),
                required_keys=required_measurement_keys,
            )
        explicit_measurement_keys = frozenset(values_by_feature)
        for axis_key, relationship_records in relationship_records_by_axis.items():
            axis_object_label_records = tuple(
                object_label_records_by_axis.get(axis_key, ())
            )
            object_label_catalog = RuntimeObjectLabelInstanceCatalog.from_records(
                axis_object_label_records
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
                    policy,
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

            for scoped_relationship in relationship_records:
                relationship = scoped_relationship.relationship_for_projection()
                relationship_measurement = RelationshipMeasurementSemantics(
                    relationship
                )
                relationship_facts = relationship_measurement.measurement_facts(
                    policy,
                    object_label_catalog=object_label_catalog,
                )
                record_measurement_facts(
                    values_by_feature,
                    (
                        (key, value)
                        for key, value in relationship_facts
                        if key not in explicit_measurement_keys
                    ),
                    required_keys=required_measurement_keys,
                )
                required_child_keys = (
                    relationship_measurement.required_child_measurement_keys(
                        required_measurement_keys
                    )
                )
                if required_measurement_keys is not None and not required_child_keys:
                    continue
                aggregate_facts = relationship_measurement.aggregate_measurement_facts(
                    child_measurement_values(
                        relationship_measurement.target_name,
                        required_child_keys=required_child_keys,
                    ),
                    policy,
                    object_label_catalog=object_label_catalog,
                    existing_measurement_keys=explicit_measurement_keys,
                    required_measurement_keys=required_measurement_keys,
                )
                relationship_object_number_aggregate_key = RuntimeMeasurementFeatureKeyFactory(
                    relationship_measurement.source_subject,
                    relationship_measurement.aggregate_feature_name(
                        ObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
                    ),
                ).key()
                if any(
                    key == relationship_object_number_aggregate_key
                    for key, _value in aggregate_facts
                ):
                    values_by_feature.pop(relationship_object_number_aggregate_key, None)
                record_measurement_facts(
                    values_by_feature,
                    (
                        (key, value)
                        for key, value in aggregate_facts
                        if key not in explicit_measurement_keys
                        or key == relationship_object_number_aggregate_key
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


@dataclass(slots=True)
class RuntimeMeasurementSnapshotAccumulator:
    """Accumulate semantic measurement facts from independently executed windows."""

    _values_by_feature: _RuntimeMeasurementFactCounters = field(default_factory=dict)

    def add(self, snapshot: RuntimeMeasurementSnapshot) -> None:
        """Merge one projected runtime window into this semantic accumulator."""
        for feature, values in snapshot.values_by_feature.items():
            RuntimeMeasurementFactCounters(
                self._values_by_feature
            ).counter(feature).update(values)

    def snapshot(self) -> RuntimeMeasurementSnapshot:
        """Freeze the accumulated semantic facts for equivalence comparison."""
        return RuntimeMeasurementSnapshot(values_by_feature=self._values_by_feature)


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
        feature_label = RuntimeMeasurementFeatureSemantics(feature, policy).label
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.MEASUREMENT_FEATURE,
                f"candidate is missing measurement feature {feature_label}",
            )
        )
    if not policy.allow_extra_candidate_measurements:
        for feature in sorted(
            candidate_features - reference_features,
            key=lambda key: key.sort_key,
        ):
            feature_label = RuntimeMeasurementFeatureSemantics(feature, policy).label
            differences.append(
                RuntimeEquivalenceDifference(
                    RuntimeEquivalenceDifferenceKind.MEASUREMENT_FEATURE,
                    f"candidate has extra measurement feature {feature_label}",
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
        feature_label = RuntimeMeasurementFeatureSemantics(feature, policy).label
        differences.append(
            RuntimeEquivalenceDifference(
                RuntimeEquivalenceDifferenceKind.MEASUREMENT_CONTENT,
                f"measurement feature {feature_label} values differ",
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
    if _zernike_descriptor_values_equivalent(feature, reference, candidate, policy):
        return True
    if SparseObjectBoundaryEquivalence(
        feature,
        reference,
        candidate,
        policy,
    ).values_equivalent():
        return True
    if RuntimeThresholdSensitivePairToleranceContract(policy).values_equivalent(
        feature,
        reference,
        candidate,
    ):
        return True
    if _feature_numeric_tolerance_values_equivalent(
        feature,
        reference,
        candidate,
        policy,
    ):
        return True
    return RuntimeMeasurementFeatureSemantics(
        feature,
        policy,
    ).unstable_shape_descriptor_values_equivalent(reference, candidate)


def _feature_numeric_tolerance_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    feature_semantics = RuntimeMeasurementFeatureSemantics(feature, policy)
    for tolerance in policy.feature_numeric_tolerances:
        if not feature_semantics.matches_numeric_tolerance(tolerance):
            continue
        if (
            tolerance.require_object_count_stability
            and not MeasurementFeatureStabilityPolicy(
                feature,
                reference,
                candidate,
                policy,
            ).object_count_values_stable()
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


def _tie_sensitive_location_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if not policy.allow_tie_sensitive_location_mismatches:
        return False
    value_feature = TieSensitiveLocationFeatureContract(
        feature,
        policy,
    ).value_feature_key()
    if value_feature is None:
        return False
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
    return SparseObjectBoundaryEquivalence(
        value_feature,
        reference,
        candidate,
        policy,
    ).values_equivalent()


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


@dataclass(frozen=True, slots=True)
class RuntimeThresholdSensitivePairToleranceContract:
    """SSOT for pair measurements allowed to drift with threshold placement."""

    policy: RuntimeEquivalencePolicy

    def values_equivalent(
        self,
        feature: RuntimeMeasurementFeatureKey,
        reference: RuntimeMeasurementSnapshot,
        candidate: RuntimeMeasurementSnapshot,
    ) -> bool:
        if (
            self.policy.threshold_sensitive_pair_abs_tolerance == 0
            and self.policy.threshold_sensitive_pair_rel_tolerance == 0
        ):
            return False
        if not self.owns_key(feature):
            return False

        pair_policy = RuntimeEquivalencePolicy(
            numeric_decimal_places=self.policy.numeric_decimal_places,
            numeric_abs_tolerance=self.policy.threshold_sensitive_pair_abs_tolerance,
            numeric_rel_tolerance=self.policy.threshold_sensitive_pair_rel_tolerance,
            measurement_feature_name_mode=self.policy.measurement_feature_name_mode,
        )
        if not _cell_signature_counters_equivalent(
            reference.values_by_feature[feature],
            candidate.values_by_feature[feature],
            pair_policy,
        ):
            return False

        return any(
            _cell_signature_counters_equivalent(
                reference.values_by_feature[companion],
                candidate.values_by_feature[companion],
                pair_policy,
            )
            for companion in self.companion_features(feature, reference, candidate)
        )

    def owns_key(self, feature: RuntimeMeasurementFeatureKey) -> bool:
        """Return whether this feature is a threshold-sensitive pair family."""
        return feature.belongs_to_source_qualified_feature_family(
            self.policy.measurement_dialect,
            _THRESHOLD_SENSITIVE_PAIR_FEATURES,
        )

    def companion_features(
        self,
        feature: RuntimeMeasurementFeatureKey,
        reference: RuntimeMeasurementSnapshot,
        candidate: RuntimeMeasurementSnapshot,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        """Return comparable pair-orientation companions for ``feature``."""
        source_tokens = feature.source_token_counter(
            self.policy.measurement_dialect,
            _THRESHOLD_SENSITIVE_PAIR_FEATURES,
        )
        if source_tokens is None:
            return ()

        comparable_features = set(reference.values_by_feature) & set(
            candidate.values_by_feature
        )
        return tuple(sorted(
            (
                other
                for other in comparable_features
                if self.is_companion_feature(
                    feature,
                    other,
                    source_tokens=source_tokens,
                )
            ),
            key=lambda key: key.sort_key,
        ))

    def is_companion_feature(
        self,
        feature: RuntimeMeasurementFeatureKey,
        other: RuntimeMeasurementFeatureKey,
        *,
        source_tokens: Counter[str],
    ) -> bool:
        """Return whether ``other`` is the opposite orientation for ``feature``."""
        if other == feature:
            return False
        if other.statistic != feature.statistic:
            return False
        if other.subject.scope is not feature.subject.scope:
            return False
        if feature.subject.scope is not MeasurementScope.IMAGE and other.subject != feature.subject:
            return False
        if (feature.source_name is not None or other.source_name is not None) and (
            other.subject != feature.subject
        ):
            return False

        feature_family = feature.source_qualified_feature_family(
            self.policy.measurement_dialect,
            _THRESHOLD_SENSITIVE_PAIR_FEATURES,
        )
        other_family = other.source_qualified_feature_family(
            self.policy.measurement_dialect,
            _THRESHOLD_SENSITIVE_PAIR_FEATURES,
        )
        if feature_family is None or other_family is None:
            return False
        if other_family.feature_name != feature_family.feature_name:
            return False
        return (
            other.source_token_counter(
                self.policy.measurement_dialect,
                _THRESHOLD_SENSITIVE_PAIR_FEATURES,
            )
            == source_tokens
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureSemantics:
    """Feature-local runtime measurement semantics used during equivalence."""

    feature: RuntimeMeasurementFeatureKey
    policy: RuntimeEquivalencePolicy

    @property
    def label(self) -> str:
        """Return a stable human-readable feature label."""
        subject = self.feature.subject
        if subject.name is None:
            subject_label = subject.scope.value
        else:
            subject_label = f"{subject.scope.value}:{subject.name}"
        feature_label = self.feature.feature_name
        if self.feature.source_name is not None:
            feature_label = f"{feature_label}@{self.feature.source_name}"
        if self.feature.statistic == MeasurementStatistic.VALUE.value:
            return f"{subject_label}/{feature_label}"
        return f"{subject_label}/{self.feature.statistic}({feature_label})"

    def matches_numeric_tolerance(
        self,
        tolerance: RuntimeMeasurementFeatureNumericTolerance,
    ) -> bool:
        """Return whether ``tolerance`` applies to this feature."""
        if (
            tolerance.subject_scope is not None
            and self.feature.subject.scope is not tolerance.subject_scope
        ):
            return False
        if (
            tolerance.statistic is not None
            and self.feature.statistic != tolerance.statistic
        ):
            return False
        if self._feature_name_matches_numeric_tolerance(
            self.feature.feature_name,
            tolerance,
        ):
            return True
        aggregate_child_feature_name = (
            RelationshipAggregateFeatureSemantics.aggregate_child_feature_name_from_key(
                self.feature,
                self.policy.measurement_dialect,
            )
        )
        return (
            aggregate_child_feature_name is not None
            and self._feature_name_matches_numeric_tolerance(
                aggregate_child_feature_name,
                tolerance,
            )
        )

    def _feature_name_matches_numeric_tolerance(
        self,
        feature_name: str,
        tolerance: RuntimeMeasurementFeatureNumericTolerance,
    ) -> bool:
        if feature_name in tolerance.feature_names:
            return True
        if (
            tolerance.feature_names
            and self.policy.measurement_dialect.source_qualified_feature_family(
                feature_name,
                self.feature.source_name,
                self.feature.subject.scope,
                tolerance.feature_names,
            )
            is not None
        ):
            return True
        return any(
            feature_name.startswith(prefix)
            for prefix in tolerance.feature_name_prefixes
        )

    def unstable_shape_descriptor_values_equivalent(
        self,
        reference: RuntimeMeasurementSnapshot,
        candidate: RuntimeMeasurementSnapshot,
    ) -> bool:
        """Return whether unstable shape-descriptor values are equivalent."""
        if not self.policy.allow_unstable_shape_descriptors:
            return False
        if self.feature.subject.scope is not MeasurementScope.OBJECT:
            return False
        if self.feature.statistic != "value":
            return False
        shape_descriptor_context = ShapeDescriptorFeatureContext(
            self.feature,
            self.policy,
        ).semantic_context()
        semantics = ShapeDescriptorFeatureSemantics.for_context(
            shape_descriptor_context,
            required=False,
        )
        if semantics is None:
            return False
        if not MeasurementFeatureStabilityPolicy(
            self.feature,
            reference,
            candidate,
            self.policy,
        ).shape_descriptor_geometry_is_stable():
            return False

        reference_values = reference.values_by_feature[self.feature]
        candidate_values = candidate.values_by_feature[self.feature]
        return semantics.values_equivalent(
            shape_descriptor_context,
            reference_values,
            candidate_values,
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementNamePartsProjection:
    """Dialect-aware semantic projection for runtime measurement feature parts."""

    parts: tuple[str, ...]
    dialect: RuntimeMeasurementDialect
    known_source_names: tuple[str, ...] = ()

    @classmethod
    def from_feature_name(
        cls,
        feature_name: str,
        dialect: RuntimeMeasurementDialect,
        *,
        known_source_names: tuple[str, ...] = (),
    ) -> "RuntimeMeasurementNamePartsProjection":
        return cls(
            tuple(part for part in feature_name.split("_") if part),
            dialect,
            known_source_names,
        )

    def category_prefix(self) -> tuple[str, ...]:
        """Return the longest dialect category prefix matched by these parts."""
        matches = tuple(
            prefix
            for prefix in self.dialect.category_prefixes
            if len(self.parts) >= len(prefix) and self.parts[: len(prefix)] == prefix
        )
        if not matches:
            return ()
        return max(matches, key=len)

    def strip_category_prefix_for_core(self) -> "RuntimeMeasurementNamePartsProjection":
        for prefix in self.dialect.category_prefixes:
            if prefix in self.dialect.calculated_feature_prefixes:
                continue
            if self.should_strip_category_prefix(prefix):
                return RuntimeMeasurementNamePartsProjection(
                    self.parts[len(prefix) :],
                    self.dialect,
                    self.known_source_names,
                )
        return self

    def should_strip_category_prefix(self, prefix: tuple[str, ...]) -> bool:
        if self.parts[: len(prefix)] != prefix or len(self.parts) <= len(prefix):
            return False
        suffix = self.parts[len(prefix) :]
        if prefix == (PairMeasurementFeature.CORRELATION.value,):
            return not _measurement_qualifier_parts_only(suffix)
        return True

    def source_qualifier_tokens(self) -> _RuntimeMeasurementNameParts:
        source_token_groups = _source_name_token_groups(self.known_source_names)
        stripped: list[str] = []
        source_names: list[str] = []
        index = 0
        while index < len(self.parts):
            matched_source_name = _matching_source_name_at(
                self.parts,
                index,
                source_token_groups,
            )
            if matched_source_name is not None:
                source_names.append(matched_source_name)
                index += len(matched_source_name.split("_"))
                continue
            if (
                index + 1 < len(self.parts)
                and self.parts[index] in self.dialect.source_qualifier_prefix_tokens
                and self.parts[index + 1] in self.dialect.source_qualifier_suffix_tokens
            ):
                source_names.append(f"{self.parts[index]}_{self.parts[index + 1]}")
                index += 2
                continue
            stripped.append(self.parts[index])
            index += 1
        return tuple(stripped), tuple(source_names)

    def semantic_core_parts(self) -> tuple[str, ...]:
        aliased = self.dialect.feature_part_aliases.get(self.parts)
        if aliased is not None:
            return aliased
        numbered_alias = _numbered_feature_parts_alias(self.parts, self.dialect)
        if numbered_alias is not None:
            return numbered_alias
        for prefix in self.dialect.scale_qualified_feature_prefixes:
            if (
                len(self.parts) == len(prefix) + 1
                and self.parts[: len(prefix)] == prefix
                and self.parts[-1].isdigit()
            ):
                return prefix
        if (
            len(self.parts) > 2
            and self.parts[:2] == ("threshold", "otsu")
            and all(
                part.isdigit() or part in self.dialect.threshold_qualifier_tokens
                for part in self.parts[2:]
            )
        ):
            return self.parts[:2]
        if len(self.parts) == 3 and self.parts[:2] == ("center", "mass"):
            return ("center", "mass", "intensity", self.parts[2])
        if (
            len(self.parts) == 2
            and self.parts[0] == "center"
            and self.parts[1] in {"x", "y", "z"}
        ):
            return ("center", self.parts[1])
        return self.parts

    def source_feature_name_and_source(self) -> tuple[str, str | None] | None:
        """Protect dialect-defined source feature phrases from source-name extraction."""
        for prefix in self.dialect.source_feature_prefixes:
            if self.parts[: len(prefix)] != prefix:
                continue
            source_parts = self.parts[len(prefix) :]
            source_name = "_".join(source_parts) if source_parts else None
            return "_".join(prefix), source_name
        return None


@dataclass(frozen=True, slots=True)
class SemanticCoreFeatureAndSourceNameProjection:
    """Project a runtime feature name to its semantic core and source qualifier."""

    feature_name: str
    dialect: RuntimeMeasurementDialect
    known_source_names: tuple[str, ...] = ()

    def project(self) -> tuple[str, str | None]:
        parts_projection = RuntimeMeasurementNamePartsProjection.from_feature_name(
            self.feature_name,
            self.dialect,
            known_source_names=self.known_source_names,
        )
        aggregate_feature = self.aggregate_prefixed_feature_name_and_source(
            parts_projection.parts
        )
        if aggregate_feature is not None:
            return aggregate_feature
        parts_projection = parts_projection.strip_category_prefix_for_core()

        direct_alias = self.dialect.feature_part_aliases.get(parts_projection.parts)
        if direct_alias is not None:
            return "_".join(direct_alias), None

        source_feature = parts_projection.source_feature_name_and_source()
        if source_feature is not None:
            return source_feature

        parts, source_names = parts_projection.source_qualifier_tokens()
        core_parts = RuntimeMeasurementNamePartsProjection(
            parts,
            self.dialect,
            self.known_source_names,
        ).semantic_core_parts()
        source_name = "__".join(source_names) if source_names else None
        return "_".join(core_parts), source_name

    def aggregate_prefixed_feature_name_and_source(
        self,
        parts: tuple[str, ...],
    ) -> tuple[str, str | None] | None:
        aggregate_identity = RuntimeAggregateFeatureIdentity.from_parts(
            parts,
            self.dialect,
        )
        if aggregate_identity is None:
            return None
        feature_name, source_name = SemanticCoreFeatureAndSourceNameProjection(
            aggregate_identity.feature_name,
            self.dialect,
            self.known_source_names,
        ).project()
        feature_name_parts = tuple(part for part in feature_name.split("_") if part)
        if not feature_name_parts:
            return None
        return (
            "_".join(
                (
                    aggregate_identity.aggregate,
                    *aggregate_identity.object_name_parts,
                    *feature_name_parts,
                )
            ),
            source_name,
        )


@dataclass(frozen=True, slots=True)
class SparseObjectBoundaryEquivalence:
    """Sparse object-boundary equivalence for object measurement features."""

    feature: RuntimeMeasurementFeatureKey
    reference: RuntimeMeasurementSnapshot
    candidate: RuntimeMeasurementSnapshot
    policy: RuntimeEquivalencePolicy

    def values_equivalent(self) -> bool:
        if not self.policy.allow_sparse_object_boundary_jitter:
            return False
        if self.feature.subject.scope is not MeasurementScope.OBJECT:
            return False
        try:
            statistic = MeasurementStatistic(self.feature.statistic)
        except ValueError:
            return False
        return SparseObjectBoundaryStatisticEquivalence.for_enum_member(
            statistic
        ).values_equivalent(self)

    def boundary_numeric_counters_equivalent(self) -> bool:
        return _sparse_numeric_counters_equivalent(
            self.reference.values_by_feature[self.feature],
            self.candidate.values_by_feature[self.feature],
            self.policy,
            abs_tolerance=self.policy.object_boundary_jitter_abs_tolerance,
            rel_tolerance=self.policy.object_boundary_jitter_rel_tolerance,
            max_unstable_values=self.policy.object_boundary_jitter_max_unstable_values,
            max_unstable_fraction=self.policy.object_boundary_jitter_max_unstable_fraction,
        )

    def shape_descriptor_values_equivalent(self) -> bool:
        shape_descriptor_context = ShapeDescriptorFeatureContext(
            self.feature,
            self.policy,
        ).semantic_context()
        shape_descriptor_semantics = ShapeDescriptorFeatureSemantics.for_context(
            shape_descriptor_context,
            required=False,
        )
        if shape_descriptor_semantics is not None:
            return shape_descriptor_semantics.boundary_jitter_values_equivalent(
                shape_descriptor_context,
                self.reference.values_by_feature[self.feature],
                self.candidate.values_by_feature[self.feature],
            )
        if _numeric_counters_are_binary(
            self.reference.values_by_feature[self.feature],
            self.candidate.values_by_feature[self.feature],
        ):
            return _sparse_numeric_counters_equivalent(
                self.reference.values_by_feature[self.feature],
                self.candidate.values_by_feature[self.feature],
                self.policy,
                abs_tolerance=self.policy.numeric_abs_tolerance,
                rel_tolerance=self.policy.numeric_rel_tolerance,
                max_unstable_values=self.policy.object_boundary_jitter_max_unstable_values,
                max_unstable_fraction=self.policy.object_boundary_jitter_max_unstable_fraction,
            )
        return self.boundary_numeric_counters_equivalent()

    def identifier_counters_equivalent(
        self,
        reference: Counter[RuntimeCellSignature],
        candidate: Counter[RuntimeCellSignature],
    ) -> bool:
        if reference == candidate:
            return True
        if any(signature.kind is not RuntimeCellValueKind.NUMBER for signature in reference):
            return False
        if any(signature.kind is not RuntimeCellValueKind.NUMBER for signature in candidate):
            return False

        unstable_cap = max(
            self.policy.object_boundary_jitter_max_unstable_values,
            math.ceil(
                sum(reference.values())
                * self.policy.object_boundary_jitter_max_unstable_fraction
            ),
        )
        missing = sum((reference - candidate).values())
        extra = sum((candidate - reference).values())
        return max(missing, extra) <= unstable_cap


class SparseObjectBoundaryStatisticEquivalence(
    EnumKeyedStrategyMixin[MeasurementStatistic],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Statistic-specific sparse object-boundary equivalence."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "statistic"

    statistic: ClassVar[MeasurementStatistic]
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def values_equivalent(
        self,
        context: SparseObjectBoundaryEquivalence,
    ) -> bool:
        """Return whether the statistic-specific sparse boundary values match."""


class SparseObjectBoundaryCountEquivalence(SparseObjectBoundaryStatisticEquivalence):
    """Sparse boundary equivalence for object count facts."""

    statistic = MeasurementStatistic.COUNT

    def values_equivalent(
        self,
        context: SparseObjectBoundaryEquivalence,
    ) -> bool:
        return (
            object_measurement_feature_has_role(
                context.feature,
                ObjectMeasurementFeatureRole.COUNT,
                context.policy.measurement_dialect,
            )
            and _object_count_counters_sparse_equivalent(
                context.reference.values_by_feature[context.feature],
                context.candidate.values_by_feature[context.feature],
                context.policy,
            )
        )


class SparseObjectBoundaryValueEquivalence(SparseObjectBoundaryStatisticEquivalence):
    """Sparse boundary equivalence for object value facts."""

    statistic = MeasurementStatistic.VALUE

    def values_equivalent(
        self,
        context: SparseObjectBoundaryEquivalence,
    ) -> bool:
        if (
            object_measurement_feature_requires_sparse_boundary_object_count_stability(
                context.feature,
                context.policy.measurement_dialect,
            )
            and not MeasurementFeatureStabilityPolicy(
                context.feature,
                context.reference,
                context.candidate,
                context.policy,
            ).object_count_values_stable()
        ):
            return False
        if object_measurement_feature_has_role(
            context.feature,
            ObjectMeasurementFeatureRole.IDENTIFIER,
            context.policy.measurement_dialect,
        ):
            return context.identifier_counters_equivalent(
                context.reference.values_by_feature[context.feature],
                context.candidate.values_by_feature[context.feature],
            )
        if any(
            object_measurement_feature_has_role(
                context.feature,
                role,
                context.policy.measurement_dialect,
            )
            for role in (
                ObjectMeasurementFeatureRole.LOCATION,
                ObjectMeasurementFeatureRole.INTENSITY,
                ObjectMeasurementFeatureRole.CALCULATED,
            )
        ):
            return context.boundary_numeric_counters_equivalent()
        if not object_measurement_feature_has_role(
            context.feature,
            ObjectMeasurementFeatureRole.SHAPE_DESCRIPTOR,
            context.policy.measurement_dialect,
        ):
            return False
        return context.shape_descriptor_values_equivalent()


class SparseObjectBoundaryMeanEquivalence(SparseObjectBoundaryStatisticEquivalence):
    """Sparse boundary equivalence for object mean facts."""

    statistic = MeasurementStatistic.MEAN

    def values_equivalent(
        self,
        context: SparseObjectBoundaryEquivalence,
    ) -> bool:
        value_feature = RuntimeMeasurementFeatureKey(
            subject=context.feature.subject,
            feature_name=context.feature.feature_name,
            statistic=MeasurementStatistic.VALUE.value,
            source_name=context.feature.source_name,
        )
        if value_feature not in context.reference.values_by_feature:
            return False
        if value_feature not in context.candidate.values_by_feature:
            return False
        if not SparseObjectBoundaryStatisticEquivalence.for_enum_member(
            MeasurementStatistic.VALUE
        ).values_equivalent(
            SparseObjectBoundaryEquivalence(
                value_feature,
                context.reference,
                context.candidate,
                context.policy,
            )
        ):
            return False

        mean_policy = RuntimeEquivalencePolicy(
            numeric_decimal_places=context.policy.numeric_decimal_places,
            numeric_abs_tolerance=context.policy.object_boundary_jitter_aggregate_abs_tolerance,
            numeric_rel_tolerance=context.policy.object_boundary_jitter_aggregate_rel_tolerance,
            measurement_feature_name_mode=context.policy.measurement_feature_name_mode,
        )
        return _cell_signature_counters_equivalent(
            context.reference.values_by_feature[context.feature],
            context.candidate.values_by_feature[context.feature],
            mean_policy,
        )


@dataclass(frozen=True, slots=True)
class MeasurementFeatureStabilityPolicy:
    """Evaluate supporting measurement stability for feature equivalence."""

    feature: RuntimeMeasurementFeatureKey
    reference: RuntimeMeasurementSnapshot
    candidate: RuntimeMeasurementSnapshot
    policy: RuntimeEquivalencePolicy

    def object_count_values_stable(self) -> bool:
        count_feature = RuntimeMeasurementFeatureKey(
            subject=self.feature.subject,
            feature_name=ObjectCoreMeasurementFeature.OBJECT_COUNT.value,
            statistic=MeasurementStatistic.COUNT.value,
        )
        reference_counts = self.reference.values_by_feature.get(count_feature)
        candidate_counts = self.candidate.values_by_feature.get(count_feature)
        if reference_counts is None or candidate_counts is None:
            reference_values = self.reference.values_by_feature.get(self.feature)
            candidate_values = self.candidate.values_by_feature.get(self.feature)
            return (
                reference_values is not None
                and candidate_values is not None
                and sum(reference_values.values()) == sum(candidate_values.values())
            )
        return _cell_signature_counters_equivalent(
            reference_counts,
            candidate_counts,
            self.policy,
        )

    def shape_descriptor_geometry_is_stable(self) -> bool:
        matched_features = 0
        for feature_name in _SHAPE_DESCRIPTOR_GATING_FEATURES:
            geometry_feature, reference_values, candidate_values = (
                self._shape_descriptor_geometry_feature_values(feature_name)
            )
            if reference_values is None and candidate_values is None:
                continue
            if reference_values is None or candidate_values is None:
                continue
            if _cell_signature_counters_equivalent(
                reference_values,
                candidate_values,
                self.policy,
            ):
                matched_features += 1
                continue
            if self.policy.allow_sparse_object_boundary_jitter and (
                SparseObjectBoundaryStatisticEquivalence.for_enum_member(
                    MeasurementStatistic.VALUE
                ).values_equivalent(
                    SparseObjectBoundaryEquivalence(
                        geometry_feature,
                        self.reference,
                        self.candidate,
                        self.policy,
                    )
                )
            ):
                matched_features += 1
                continue
            else:
                return False
        return matched_features >= 3

    def object_measurement_role_values_stable(
        self,
        role: ObjectMeasurementFeatureRole,
        *,
        min_stable_features: int,
    ) -> bool:
        matched_features = 0
        candidate_keys = (
            self.reference.values_by_feature.keys()
            | self.candidate.values_by_feature.keys()
        )
        for candidate_key in candidate_keys:
            if not self._candidate_key_has_role(candidate_key, role):
                continue
            reference_values = self.reference.values_by_feature.get(candidate_key)
            candidate_values = self.candidate.values_by_feature.get(candidate_key)
            if reference_values is None or candidate_values is None:
                continue
            if not _cell_signature_counters_equivalent(
                reference_values,
                candidate_values,
                self.policy,
            ):
                continue
            matched_features += 1
        return matched_features >= min_stable_features

    def _shape_descriptor_geometry_feature_values(
        self,
        feature_name: str,
    ) -> tuple[
        RuntimeMeasurementFeatureKey | None,
        Counter[RuntimeCellSignature] | None,
        Counter[RuntimeCellSignature] | None,
    ]:
        geometry_source_names = (
            (self.feature.source_name, None)
            if self.feature.source_name is not None
            else (None,)
        )
        for source_name in geometry_source_names:
            candidate_geometry_feature = RuntimeMeasurementFeatureKey(
                subject=self.feature.subject,
                feature_name=feature_name,
                statistic=self.feature.statistic,
                source_name=source_name,
            )
            candidate_reference_values = self.reference.values_by_feature.get(
                candidate_geometry_feature
            )
            candidate_candidate_values = self.candidate.values_by_feature.get(
                candidate_geometry_feature
            )
            if (
                candidate_reference_values is None
                and candidate_candidate_values is None
            ):
                continue
            return (
                candidate_geometry_feature,
                candidate_reference_values,
                candidate_candidate_values,
            )
        return None, None, None

    def _candidate_key_has_role(
        self,
        candidate_key: RuntimeMeasurementFeatureKey,
        role: ObjectMeasurementFeatureRole,
    ) -> bool:
        if candidate_key.subject != self.feature.subject:
            return False
        if candidate_key.source_name is not None:
            return False
        if candidate_key.statistic != MeasurementStatistic.VALUE.value:
            return False
        return object_measurement_feature_has_role(
            candidate_key,
            role,
            self.policy.measurement_dialect,
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


def _zernike_descriptor_values_equivalent(
    feature: RuntimeMeasurementFeatureKey,
    reference: RuntimeMeasurementSnapshot,
    candidate: RuntimeMeasurementSnapshot,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    if not policy.allow_unstable_zernike_descriptors:
        return False
    descriptor = IndexedObjectZernikeDescriptor.from_feature_name(feature.feature_name)
    if descriptor is None:
        return False
    if not object_measurement_feature_has_role(
        feature,
        ObjectMeasurementFeatureRole.ZERNIKE_DESCRIPTOR,
        policy.measurement_dialect,
    ):
        return False
    if not ObjectZernikeDescriptorStabilityContract.for_enum_member(
        descriptor.family
    ).is_stable(
        feature,
        descriptor,
        reference,
        candidate,
        policy,
    ):
        return False

    abs_tolerance = (
        policy.zernike_descriptor_phase_abs_tolerance
        if descriptor.family is ObjectZernikeDescriptorFeature.INTENSITY_PHASE
        else policy.zernike_descriptor_magnitude_abs_tolerance
    )
    return _sparse_numeric_counters_equivalent(
        reference.values_by_feature[feature],
        candidate.values_by_feature[feature],
        policy,
        abs_tolerance=abs_tolerance,
        rel_tolerance=policy.zernike_descriptor_rel_tolerance,
        max_unstable_values=policy.object_boundary_jitter_max_unstable_values,
        max_unstable_fraction=policy.object_boundary_jitter_max_unstable_fraction,
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
    if not is_wide_measurement_table(first_row):
        return False

    first_subject = RuntimeExportRowSubject(table.path, first_row).subject()
    if RuntimeMetadataMapRow(first_subject, first_row).matches():
        return True

    normalized_fields = tuple(_normalize_identifier(field) for field in table.header)
    identity_indexes = {
        index
        for index, field_name in enumerate(normalized_fields)
        if RuntimeMeasurementIdentityField(
            policy.measurement_dialect
        ).normalized_field_matches(field_name)
    }
    feature_column_indexes = tuple(
        index
        for index in range(len(table.header))
        if index not in identity_indexes
    )
    if not feature_column_indexes:
        record_measurement_facts(
            values_by_feature,
            RuntimeTableSnapshotFactExtractor(
                table,
                policy=policy,
                known_source_names=known_source_names,
            ).identity_facts(),
        )
        return True
    feature_column_indexes = tuple(
        index
        for index in feature_column_indexes
        if not _is_aggregate_image_number_reference_measurement_field(
            table.header[index]
        )
    )
    if not feature_column_indexes:
        record_measurement_facts(
            values_by_feature,
            RuntimeTableSnapshotFactExtractor(
                table,
                policy=policy,
                known_source_names=known_source_names,
            ).identity_facts(),
        )
        return True
    column_subject_contexts = {
        index: RuntimeExportColumnSubject(
            table,
            first_row,
            index,
            first_subject,
        ).context_pair()
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
    image_number_offset = RuntimeImageNumberOffset.from_table_rows(table.header, table.rows)

    key_cache: dict[
        tuple[RuntimeMeasurementSubjectKey, str | None, int, tuple[str, ...]],
        RuntimeMeasurementFeatureKey | None,
    ] = {}
    qualifier_columns = row_qualifier_columns(
        normalized_fields,
        policy.measurement_dialect,
    )
    uses_row_qualifiers = bool(qualifier_columns)
    collapsed_numeric_qualifier_by_index = {
        index: _measurement_field_has_collapsed_numeric_qualifier(
            table.header[index],
            policy.measurement_dialect,
            known_source_names=known_source_names,
        )
        for index in feature_column_indexes
    }

    image_number_offset = RuntimeImageNumberOffset.from_table_rows(table.header, table.rows)
    for row in table.rows:
        row_mapping = dict(zip(table.header, row, strict=True))
        row_subject = RuntimeExportRowSubject(table.path, row_mapping).subject()
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
            row_qualifier_values(row, qualifier_columns)
            if uses_row_qualifiers
            else ()
        )
        row_fact_records: list[
            tuple[
                _RuntimeMeasurementPaddingGroup,
                RuntimeMeasurementFeatureKey,
                RuntimeCellSignature,
                bool,
            ]
        ] = []
        for index in feature_column_indexes:
            if index in padding_indexes:
                continue
            context, normalized_context = column_subject_contexts[index]
            subject = RuntimeColumnContextSubject(
                context,
                normalized_context,
                row_object_name,
                normalized_row_object_name,
                fallback_subject=row_subject,
            ).subject()
            qualifiers = (
                measurement_row_qualifiers_from_values(
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
            row_fact_records.append(
                (
                    RuntimeMeasurementFactProjectionContract.padding_group(
                        _normalize_identifier(table.path.stem) or "measurements",
                        field_name,
                        key,
                        policy.measurement_dialect,
                    ),
                    key,
                    _cell_signature(
                        str(
                            RuntimeImageNumberReferenceValue(
                                field_name,
                                row[index],
                                image_number_offset,
                            ).normalized()
                        ),
                        policy,
                    ),
                    collapsed_numeric_qualifier_by_index[index],
                )
            )
        for key, value in RuntimeMeasurementFactProjectionContract.dedupe_observed_qualified_records(
            row_fact_records,
            policy,
        ):
            RuntimeMeasurementFactCounters(values_by_feature).counter(key)[value] += 1
    return True


def _record_static_wide_runtime_measurement_table(
    values_by_feature: _RuntimeMeasurementFactCounters,
    context: _RuntimeMeasurementTableProjectionContext,
    *,
    row_merge_cache: _RuntimeMeasurementRowMergeCache | None = None,
    primary_row_identities: _RuntimeMeasurementPrimaryRowSet | None = None,
    object_row_domain: RuntimeObjectMeasurementFactRowDomain | None = None,
) -> bool:
    """Record wide runtime measurement tables without per-row key rebuilding."""
    table = context.table
    policy = context.policy
    row_iterator = iter_measurement_rows((table,))
    first_row = next(row_iterator, None)
    if first_row is None:
        return True

    first_mapping = measurement_row_mapping(first_row)
    if not is_wide_measurement_table(first_mapping):
        return False
    header = tuple(first_mapping)
    for row in row_iterator:
        row_mapping = measurement_row_mapping(row)
        if tuple(row_mapping) != header:
            return False

    table_subject = RuntimeMeasurementSubjectKey.from_table_subject(table.subject)
    subject_schema = _runtime_measurement_row_subject_schema(header)
    first_row_values = tuple(first_mapping.get(field_name) for field_name in header)
    first_subject_projection = RuntimeMeasurementRowSubjectProjection(
        table_subject,
        table.source_image_name,
        first_row_values,
        subject_schema,
    )
    first_subject = first_subject_projection.subject()
    if RuntimeMetadataMapRow(first_subject, first_mapping).matches():
        return True

    normalized_fields = tuple(_normalize_identifier(field) for field in header)
    normalized_field_indexes = {
        field_name: index
        for index, field_name in enumerate(normalized_fields)
    }
    identity_indexes = {
        index
        for index, field_name in enumerate(normalized_fields)
        if RuntimeMeasurementIdentityField(
            policy.measurement_dialect
        ).normalized_field_matches(field_name)
    }
    feature_column_indexes = tuple(
        index for index in range(len(header)) if index not in identity_indexes
    )
    if not feature_column_indexes:
        return True
    first_source_qualification = first_subject.bind_row_source_identity(
        first_subject_projection.source_name()
    )
    if _wide_measurement_table_needs_row_derivation(
        header,
        feature_column_indexes,
        policy,
        source_name=first_source_qualification.feature_source_name,
        known_source_names=context.known_source_names,
    ):
        return False

    required_projection = RequiredRuntimeMeasurementProjection(
        context.required_keys,
        policy,
        known_source_names=context.known_source_names,
    )
    input_keys = required_projection.input_keys()
    required_subjects = required_projection.subjects()
    qualifier_indexes = {
        qualifier: tuple(
            (
                normalized_field_indexes[field_name]
                if normalized_field_indexes.get(field_name) in identity_indexes
                else None
            )
            for field_name in qualifier.field_names
        )
        for qualifier in policy.measurement_dialect.row_qualifiers
    }
    qualifiers_by_index = {
        index: tuple(
            (qualifier, qualifier_indexes[qualifier])
            for qualifier in policy.measurement_dialect.row_qualifiers
            if row_qualifier_applies_to_field(
                qualifier,
                tuple(part for part in normalized_fields[index].split("_") if part),
            )
            and any(axis_index is not None for axis_index in qualifier_indexes[qualifier])
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
        object_row_domain or RuntimeObjectMeasurementFactRowDomain(),
        context.required_keys,
        {},
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

    for row in iter_measurement_rows((table,)):
        row_mapping = measurement_row_mapping(row)
        row_values = tuple(row_mapping.get(field_name) for field_name in header)
        row_subject_projection = RuntimeMeasurementRowSubjectProjection(
            table_subject,
            table.source_image_name,
            row_values,
            subject_schema,
        )
        subject = row_subject_projection.subject()
        if (
            required_subjects is not None
            and subject.scope is MeasurementScope.OBJECT
            and subject not in required_subjects
        ):
            continue
        source_qualification = subject.bind_row_source_identity(
            row_subject_projection.source_name()
        )
        _record_runtime_primary_measurement_row_identity(
            primary_row_identities,
            row_mapping,
            context.axis_key,
            subject,
            policy,
        )
        derived_row_facts = StaticWideRuntimeRowProjector(
            row_projection_context,
            row_values,
            subject,
            source_qualification.feature_source_name,
        ).facts()
        if not derived_row_facts:
            continue
        aggregate_input_context = _AggregateInputRecordingContext(
            aggregate_values_by_feature,
            row_mapping,
            context.axis_key,
            context.required_keys,
            aggregate_input_key_cache,
            policy.measurement_dialect,
        )
        RuntimeRowMeasurementFactRecorder(
            fact_recording_context,
            aggregate_input_context,
            row_merge_cache=row_merge_cache,
            policy=policy,
        ).record(
            derived_row_facts,
            row_mapping=row_mapping,
            axis_key=context.axis_key,
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


@dataclass(frozen=True, slots=True)
class StaticWideRuntimeRowProjector:
    """Project one static-wide runtime measurement row into semantic facts."""

    context: _StaticWideRuntimeRowProjectionContext
    row_values: tuple[object, ...]
    subject: RuntimeMeasurementSubjectKey
    source_name: str | None

    def facts(self) -> _RuntimeMeasurementFacts:
        row_fact_records: list[_RuntimeRowProjectionRecord[RuntimeCellSignature]] = []
        padding_group_presence: _RuntimeMeasurementPaddingGroupPresence = {}
        row_qualifier_cache: _RuntimeMeasurementIndexedQualifierCache = {}
        derives_directional_pair_facts = False
        for index in self.context.feature_column_indexes:
            if index in self.context.aggregate_reference_indexes:
                continue
            field_name = self.context.header[index]
            value = self.row_values[index]
            qualifiers = self.qualifiers(index, row_qualifier_cache)
            key = self.feature_key(index, qualifiers)
            if key is None:
                continue
            derives_directional_pair_facts = (
                derives_directional_pair_facts
                or self.derives_directional_pair_facts(key)
            )
            padding_group = self.padding_group(field_name, key, qualifiers)
            padding_group_presence[padding_group] = (
                padding_group_presence.get(padding_group, False)
                or RuntimeMeasurementValuePresence(value).is_present()
            )
            if (
                self.context.input_keys is not None
                and key not in self.context.input_keys
                and not _runtime_value_is_mapping(value)
            ):
                continue
            for fact_key, fact_value in RuntimeMeasurementCellFactProjection().project_cell(
                RuntimeMeasurementCellValue(
                    key,
                    value,
                    self.context.policy,
                    required_keys=self.context.input_keys,
                )
            ):
                derives_directional_pair_facts = (
                    derives_directional_pair_facts
                    or self.derives_directional_pair_facts(fact_key)
                )
                row_fact_records.append((padding_group, fact_key, fact_value))

        facts = RuntimeMeasurementFactProjectionContract.dedupe_observed_alias_records(
            (
                record
                for record in row_fact_records
                if padding_group_presence.get(record[0], True)
            ),
            self.context.policy,
        )
        if not derives_directional_pair_facts:
            return facts
        return RuntimeDirectionalPairMeasurementDerivationContract(
            self.context.policy,
            self.context.known_source_names,
        ).derive(facts)

    def qualifiers(
        self,
        index: int,
        row_qualifier_cache: _RuntimeMeasurementIndexedQualifierCache,
    ) -> tuple[str, ...]:
        indexed_qualifiers = self.context.qualifiers_by_index[index]
        if not indexed_qualifiers:
            return ()
        qualifier_cache_key = id(indexed_qualifiers)
        qualifiers = row_qualifier_cache.get(qualifier_cache_key)
        if qualifiers is None:
            qualifiers = measurement_row_qualifiers_from_indexed_values_cached(
                self.row_values,
                indexed_qualifiers,
                self.context.qualifier_render_cache,
            )
            row_qualifier_cache[qualifier_cache_key] = qualifiers
        return qualifiers

    def feature_key(
        self,
        index: int,
        qualifiers: tuple[str, ...],
    ) -> RuntimeMeasurementFeatureKey | None:
        if qualifiers:
            return self._feature_key(index, qualifiers)
        cache_key = (self.subject, self.source_name, index, qualifiers)
        key = self.context.key_cache.get(cache_key, _CACHE_MISS)
        if key is _CACHE_MISS:
            key = self._feature_key(index, qualifiers)
            self.context.key_cache[cache_key] = key
        return key

    def padding_group(
        self,
        field_name: str,
        key: RuntimeMeasurementFeatureKey,
        qualifiers: tuple[str, ...],
    ) -> _RuntimeMeasurementPaddingGroup:
        if qualifiers:
            return self._padding_group(field_name, key)
        padding_group_cache_key = (field_name, key)
        padding_group = self.context.padding_group_cache.get(padding_group_cache_key)
        if padding_group is None:
            padding_group = self._padding_group(field_name, key)
            self.context.padding_group_cache[padding_group_cache_key] = padding_group
        return padding_group

    def derives_directional_pair_facts(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> bool:
        return key.belongs_to_source_qualified_feature_family(
            self.context.policy.measurement_dialect,
            (_PAIR_REGRESSION_SLOPE_FEATURE,),
        )

    def _feature_key(
        self,
        index: int,
        qualifiers: tuple[str, ...],
    ) -> RuntimeMeasurementFeatureKey | None:
        return _measurement_feature_key_from_source_context(
            _MeasurementFeatureKeySourceContext(
                self.context.header[index],
                self.subject,
                self.context.policy,
                qualifiers,
                self.source_name,
                self.context.known_source_names,
            )
        )

    def _padding_group(
        self,
        field_name: str,
        key: RuntimeMeasurementFeatureKey,
    ) -> _RuntimeMeasurementPaddingGroup:
        return RuntimeMeasurementFactProjectionContract.padding_group(
            self.context.table_padding_group,
            field_name,
            key,
            self.context.policy.measurement_dialect,
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
        row_identity = axis_scoped_measurement_row_identity(
            context.row_mapping,
            context.axis_key,
            context.measurement_dialect,
        )
    _runtime_aggregate_mean_accumulator(
        context.values_by_feature,
        mean_key,
        row_identity,
    ).add(numeric_value)
    return row_identity


@dataclass(frozen=True, slots=True)
class RuntimeRowMeasurementFactRecorder:
    """Record one runtime measurement row with merge and aggregate semantics."""

    fact_context: _RuntimeMeasurementFactRecordingContext
    aggregate_context: _AggregateInputRecordingContext
    row_merge_cache: _RuntimeMeasurementRowMergeCache | None = None
    policy: RuntimeEquivalencePolicy = RuntimeEquivalencePolicy()

    def record(
        self,
        facts: Iterable[_RuntimeMeasurementFact],
        *,
        row_mapping: Mapping[str, object] | None = None,
        axis_key: object | None = None,
    ) -> None:
        row_identity: _RuntimeMeasurementRowIdentityOrMissing = None
        emitted_row_facts: _RuntimeMeasurementFactList = []
        for key, value in self.merged_facts(facts, row_mapping, axis_key):
            self.fact_context.explicit_measurement_keys.add(key)
            if (
                self.fact_context.required_keys is None
                or key in self.fact_context.required_keys
            ):
                RuntimeMeasurementFactCounters(
                    self.fact_context.values_by_feature
                ).counter(key)[value] += 1
                emitted_row_facts.append((key, value))
            row_identity = _record_row_aggregate_input_value(
                self.aggregate_context,
                key,
                value,
                row_identity=row_identity,
            )
        if row_mapping is not None and emitted_row_facts:
            self.fact_context.object_row_domain.record_row_facts(
                row_mapping,
                axis_key,
                self.policy,
                emitted_row_facts,
            )

    def merged_facts(
        self,
        facts: Iterable[_RuntimeMeasurementFact],
        row_mapping: Mapping[str, object] | None,
        axis_key: object | None,
    ) -> _RuntimeMeasurementFacts:
        row_facts = tuple(facts)
        if self.row_merge_cache is None or row_mapping is None:
            return row_facts
        return self._merge_runtime_row_measurement_facts(
            row_mapping,
            axis_key,
            row_facts,
        )

    def _merge_runtime_row_measurement_facts(
        self,
        row_mapping: Mapping[str, object],
        axis_key: object | None,
        facts: Iterable[_RuntimeMeasurementFact],
    ) -> _RuntimeMeasurementFacts:
        """Defer row-identifiable alias families until all runtime tables are seen."""
        if self.row_merge_cache is None:
            return tuple(facts)
        remaining_facts: _RuntimeMeasurementFactList = []
        row_merge_contract = RuntimeObjectLocationRowMergeContract(self.policy)
        for key, value in facts:
            if not row_merge_contract.owns_key(key):
                remaining_facts.append((key, value))
                continue
            identity = RuntimeObjectMeasurementRowIdentity.from_row_mapping(
                row_mapping,
                axis_key,
                self.policy,
            )
            if identity is None:
                remaining_facts.append((key, value))
                continue
            merge_key = (key, identity.row_identity)
            priority = _runtime_row_measurement_fact_priority(
                row_mapping,
                key,
                self.policy,
                self.fact_context.row_priority_cache,
            )
            candidate = (
                priority,
                priority,
                value,
            )
            current = self.row_merge_cache.get(merge_key)
            if current is None or _runtime_row_merge_candidate_preferred(
                candidate,
                current,
            ):
                self.row_merge_cache[merge_key] = (
                    candidate[0],
                    candidate[1] if current is None else min(candidate[1], current[1]),
                    candidate[2],
                )
            elif current is not None:
                self.row_merge_cache[merge_key] = (
                    current[0],
                    min(current[1], priority),
                    current[2],
                )
        return tuple(remaining_facts)


def _dedupe_runtime_measurement_table_object_subtable(
    table: MeasurementTable,
    seen_object_subtables: _RuntimeMeasurementObjectSubtableSet,
) -> MeasurementTable:
    non_object_rows: list[object] = []
    object_row_fingerprint = RuntimeMeasurementRowFingerprintBuilder()
    object_row_count = 0
    total_row_count = 0
    for row in iter_measurement_rows((table,)):
        total_row_count += 1
        row_mapping = measurement_row_mapping(row)
        if RuntimeMeasurementRowMapping(row_mapping).has_object_identity():
            object_row_fingerprint.add_row_mapping(row_mapping)
            object_row_count += 1
        else:
            non_object_rows.append(row)
    if total_row_count == 0 or object_row_count == 0:
        return table

    subtable_key = RuntimeMeasurementTableIdentity.from_table_row_fingerprint(
        table,
        object_row_fingerprint.finish(),
    )
    if subtable_key not in seen_object_subtables:
        seen_object_subtables.add(subtable_key)
        return table
    if not non_object_rows:
        return MeasurementTable(
            name=table.name,
            rows=(),
            object_name=table.object_name,
            fields=table.fields,
            object_id_field=table.object_id_field,
            source_image_name=table.source_image_name,
            subject=table.subject,
        )
    return MeasurementTable(
        name=table.name,
        rows=tuple(non_object_rows),
        object_name=table.object_name,
        fields=table.fields,
        object_id_field=table.object_id_field,
        source_image_name=table.source_image_name,
        subject=table.subject,
    )


@dataclass(frozen=True, slots=True)
class RuntimeObjectLocationRowMergeContract(metaclass=AutoRegisterMeta):
    """SSOT for object-location value facts merged by runtime row identity."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None
    policy: RuntimeEquivalencePolicy

    @classmethod
    def registered_projection(
        cls,
        registry_key: RuntimeObjectLocationRowMergeProjectionKey,
        policy: RuntimeEquivalencePolicy,
    ) -> "RuntimeObjectLocationRowMergeContract":
        """Return the registered row-merge projection for ``registry_key``."""
        try:
            projection_type = cls.__registry__[registry_key.value]
        except KeyError as exc:
            registered = tuple(cls.__registry__)
            raise ValueError(
                "Unknown runtime object-location row-merge projection "
                f"{registry_key.value!r}; registered projections: {registered!r}."
            ) from exc
        return projection_type(policy)

    def owns_key(self, key: RuntimeMeasurementFeatureKey) -> bool:
        return (
            key.subject.scope is MeasurementScope.OBJECT
            and key.statistic == MeasurementStatistic.VALUE.value
            and key.source_name is None
            and object_measurement_feature_has_role(
                key,
                ObjectMeasurementFeatureRole.LOCATION,
                self.policy.measurement_dialect,
            )
        )

    def subjects(
        self,
        row_merge_cache: _RuntimeMeasurementRowMergeCache,
    ) -> frozenset[RuntimeMeasurementSubjectKey]:
        """Return subjects owned by this row-merge projection."""
        return frozenset(
            key.subject
            for key, row_identity in row_merge_cache
            if self.owns_row_identity(key, row_identity)
        )

    def owns_row_identity(
        self,
        key: RuntimeMeasurementFeatureKey,
        row_identity: _RuntimeMeasurementRowIdentity,
    ) -> bool:
        """Return whether this projection owns a row identity."""
        del row_identity
        return self.owns_key(key)


def _runtime_row_measurement_fact_priority(
    row_mapping: Mapping[str, object],
    key: RuntimeMeasurementFeatureKey,
    policy: RuntimeEquivalencePolicy,
    row_priority_cache: _RuntimeMeasurementRowPriorityCache,
) -> int:
    """Return dialect category priority for the row field that produced ``key``."""
    candidates: list[str] = []
    long_form_feature = RuntimeMeasurementRowMapping(row_mapping).first_value(_MEASUREMENT_FEATURE_NAME_FIELDS)
    cache_key = RuntimeMeasurementRowPriorityCacheKey(
        row_fields=tuple(row_mapping),
        long_form_feature=str(long_form_feature) if long_form_feature is not None else None,
        feature_key=key,
    )
    cached = row_priority_cache.get(cache_key)
    if cached is not None:
        return cached
    if long_form_feature is not None:
        candidates.append(str(long_form_feature))
    else:
        candidates.extend(str(field_name) for field_name in row_mapping)

    priorities = tuple(
        priority
        for feature_name in candidates
        if (
            priority := _measurement_feature_source_priority(
                feature_name,
                key,
                policy,
            )
        )
        is not None
    )
    priority = min(priorities, default=sys.maxsize)
    row_priority_cache[cache_key] = priority
    return priority


def _measurement_feature_source_priority(
    feature_name: str,
    key: RuntimeMeasurementFeatureKey,
    policy: RuntimeEquivalencePolicy,
) -> int | None:
    priority = RuntimeMeasurementFeatureCategoryPriority(
        feature_name,
        policy,
    ).priority()
    if priority is None:
        return None
    canonical_feature_name, canonical_source_name = (
        _canonical_measurement_feature_name_and_source(
            feature_name,
            policy,
            source_name=None,
            known_source_names=(),
        )
    )
    if canonical_feature_name != key.feature_name or canonical_source_name != key.source_name:
        return None
    return priority if canonical_feature_name else None


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureCategoryPriority:
    """Dialect category-prefix priority for runtime measurement feature names."""

    feature_name: str
    policy: RuntimeEquivalencePolicy

    def priority(self) -> int | None:
        normalized = _normalize_identifier(self.feature_name)
        if RuntimeMeasurementIdentityField(
            self.policy.measurement_dialect
        ).normalized_field_matches(normalized):
            return None
        parts = tuple(part for part in normalized.split("_") if part)
        parts_projection = RuntimeMeasurementNamePartsProjection(
            parts,
            self.policy.measurement_dialect,
        )
        for index, prefix in enumerate(
            self.policy.measurement_dialect.category_prefixes
        ):
            if parts_projection.should_strip_category_prefix(prefix):
                return index
        return -1


def _runtime_row_merge_candidate_preferred(
    candidate: _RuntimeMeasurementRowMergeValue,
    current: _RuntimeMeasurementRowMergeValue,
) -> bool:
    candidate_priority, _candidate_row_priority, candidate_value = candidate
    current_priority, _current_row_priority, current_value = current
    candidate_missing = RuntimeCellMissingStrategy.for_kind(
        candidate_value.kind
    ).is_missing(candidate_value)
    current_missing = RuntimeCellMissingStrategy.for_kind(
        current_value.kind
    ).is_missing(current_value)
    if current_missing and not candidate_missing:
        return True
    if candidate_missing and not current_missing:
        return False
    return candidate_priority < current_priority


def _record_runtime_primary_measurement_row_identity(
    primary_row_identities: _RuntimeMeasurementPrimaryRowSet | None,
    row_mapping: Mapping[str, object],
    axis_key: object | None,
    subject: RuntimeMeasurementSubjectKey,
    policy: RuntimeEquivalencePolicy,
) -> None:
    if primary_row_identities is None:
        return
    if subject.scope is not MeasurementScope.OBJECT:
        return
    if not _runtime_measurement_row_has_primary_object_features(row_mapping, policy):
        return
    identity = RuntimeObjectMeasurementRowIdentity.from_row_mapping(
        row_mapping,
        axis_key,
        policy,
    )
    if identity is None:
        return
    primary_row_identities.add((subject, identity.row_identity))


def _runtime_measurement_row_has_primary_object_features(
    row_mapping: Mapping[str, object],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    primary_location_priority = _object_location_primary_row_priority(policy)
    long_form_feature = RuntimeMeasurementRowMapping(row_mapping).first_value(_MEASUREMENT_FEATURE_NAME_FIELDS)
    feature_names = (
        (str(long_form_feature),)
        if long_form_feature is not None
        else tuple(str(field_name) for field_name in row_mapping)
    )
    for feature_name in feature_names:
        priority = RuntimeMeasurementFeatureCategoryPriority(
            feature_name,
            policy,
        ).priority()
        if priority is not None and priority <= primary_location_priority:
            return True
    return False


class RuntimeRowMergeLocationSubjectProjection(RuntimeObjectLocationRowMergeContract):
    """Project subjects with measured location rows from the row-merge cache."""

    registry_key: ClassVar[str | None] = (
        RuntimeObjectLocationRowMergeProjectionKey.LOCATION.value
    )


class RuntimeRowMergeAggregateLocationSubjectProjection(
    RuntimeRowMergeLocationSubjectProjection
):
    """Project measured-location subjects that can derive image-scoped aggregates."""

    registry_key: ClassVar[str | None] = (
        RuntimeObjectLocationRowMergeProjectionKey.AGGREGATE_LOCATION.value
    )

    def owns_row_identity(
        self,
        key: RuntimeMeasurementFeatureKey,
        row_identity: _RuntimeMeasurementRowIdentity,
    ) -> bool:
        return super().owns_row_identity(
            key,
            row_identity,
        ) and RuntimeObjectMeasurementRowIdentity(row_identity).has_image_identity


def _record_runtime_row_merge_facts(
    values_by_feature: _RuntimeMeasurementFactCounters,
    row_merge_cache: _RuntimeMeasurementRowMergeCache,
    *,
    required_keys: _RuntimeRequiredMeasurementKeys,
    policy: RuntimeEquivalencePolicy,
    primary_row_identities: _RuntimeMeasurementPrimaryRowSet,
    object_row_domain: RuntimeObjectMeasurementFactRowDomain | None = None,
) -> None:
    aggregate_values_by_identity: dict[
        tuple[RuntimeMeasurementFeatureKey, _RuntimeMeasurementRowIdentity],
        RuntimeAggregateMeanAccumulator,
    ] = {}
    aggregate_key_cache: _AggregateMeanKeyCache = {}
    for (key, row_identity), (_priority, _row_priority, value) in row_merge_cache.items():
        if (key.subject, row_identity) not in primary_row_identities:
            continue
        mean_key = _aggregate_mean_key(
            key,
            required_keys=required_keys,
            key_cache=aggregate_key_cache,
        )
        value_required = required_keys is None or key in required_keys
        if not value_required and mean_key is None:
            continue
        if value_required:
            RuntimeMeasurementFactCounters(values_by_feature).counter(key)[value] += 1
            if object_row_domain is not None:
                object_row_domain.record_subject_row_identity(key.subject, row_identity)
        if mean_key is None or not RuntimeObjectMeasurementRowIdentity(
            row_identity
        ).has_image_identity:
            continue
        numeric_value = _finite_numeric_runtime_cell_value(value)
        if numeric_value is None:
            continue
        image_row_identity = tuple(
            field
            for field in row_identity
            if field[0] != _OBJECT_LABEL_ROW_IDENTITY_FIELD
        )
        aggregate_identity = (mean_key, image_row_identity)
        accumulator = aggregate_values_by_identity.get(aggregate_identity)
        if accumulator is None:
            accumulator = RuntimeAggregateMeanAccumulator()
            aggregate_values_by_identity[aggregate_identity] = accumulator
        accumulator.add(numeric_value)

    mean_keys = frozenset(
        mean_key for mean_key, _row_identity in aggregate_values_by_identity
    )
    for mean_key in mean_keys:
        values_by_feature.pop(mean_key, None)
    for (mean_key, _row_identity), accumulator in aggregate_values_by_identity.items():
        if not accumulator.has_values:
            continue
        RuntimeMeasurementFactCounters(values_by_feature).counter(mean_key)[
            _cell_signature(str(accumulator.mean), policy)
        ] += 1


def _primary_row_object_count_measurement_facts(
    primary_row_identities: _RuntimeMeasurementPrimaryRowSet,
    row_merge_cache: _RuntimeMeasurementRowMergeCache,
    policy: RuntimeEquivalencePolicy,
    *,
    existing_subjects: frozenset[RuntimeMeasurementSubjectKey],
    required_keys: _RuntimeRequiredMeasurementKeys,
) -> _RuntimeMeasurementFacts:
    counts_by_image: dict[
        tuple[RuntimeMeasurementSubjectKey, _RuntimeMeasurementRowIdentity],
        set[object],
    ] = {}
    source_row_identities = (
        object_measurement_subject_row_identities_with_role(
            row_merge_cache,
            ObjectMeasurementFeatureRole.LOCATION,
            policy.measurement_dialect,
        )
        or primary_row_identities
    )
    for subject, row_identity in source_row_identities:
        identity = RuntimeObjectMeasurementRowIdentity(row_identity)
        image_identity = identity.image_identity
        object_label = identity.object_label_signature
        if object_label is None:
            continue
        counts_by_image.setdefault((subject, image_identity), set()).add(object_label)

    facts: _RuntimeMeasurementFactList = []
    for (subject, _image_identity), object_labels in counts_by_image.items():
        if subject in existing_subjects:
            continue
        key = RuntimeMeasurementFeatureKey(
            subject,
            ObjectCoreMeasurementFeature.OBJECT_COUNT.value,
            MeasurementStatistic.COUNT.value,
        )
        if required_keys is not None and key not in required_keys:
            continue
        facts.append((key, _cell_signature(str(len(object_labels)), policy)))
    return tuple(facts)


def _object_location_primary_row_priority(policy: RuntimeEquivalencePolicy) -> int:
    location_priority = _measurement_category_priority(
        ("location",),
        policy.measurement_dialect,
    )
    return sys.maxsize if location_priority is None else location_priority - 1


def _measurement_category_priority(
    prefix: tuple[str, ...],
    dialect: RuntimeMeasurementDialect,
) -> int | None:
    try:
        return dialect.category_prefixes.index(prefix)
    except ValueError:
        return None


def _record_runtime_aggregate_mean_facts(
    values_by_feature: _RuntimeMeasurementFactCounters,
    aggregate_values_by_feature: _AggregateValuesByFeature,
    explicit_measurement_keys: _RuntimeMeasurementKeySet,
    policy: RuntimeEquivalencePolicy,
    *,
    required_keys: _RuntimeRequiredMeasurementKeys,
) -> None:
    for (mean_key, _row_identity), accumulator in aggregate_values_by_feature.items():
        if not accumulator.has_values:
            continue
        if mean_key in explicit_measurement_keys:
            continue
        if required_keys is not None and mean_key not in required_keys:
            continue
        RuntimeMeasurementFactCounters(values_by_feature).counter(mean_key)[
            _cell_signature(str(accumulator.mean), policy)
        ] += 1


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


@dataclass(frozen=True, slots=True)
class RuntimeTableSnapshotFactExtractor:
    """Project exported measurement table snapshots into semantic facts."""

    table: RuntimeTableSnapshot
    policy: RuntimeEquivalencePolicy
    known_source_names: tuple[str, ...] = ()

    def measurement_facts(self) -> _RuntimeMeasurementFacts:
        static_facts = self.static_wide_measurement_facts()
        if static_facts is not None:
            return static_facts

        feature_indexes = tuple(
            index
            for index, field_name in enumerate(self.table.header)
            if not RuntimeMeasurementIdentityField(
                self.policy.measurement_dialect
            ).field_matches(field_name)
        )
        if not feature_indexes:
            return self.identity_facts()
        padding_groups_by_index = _contextual_measurement_padding_groups(
            self.table.column_context,
            self.table.header,
            feature_indexes,
            self.policy.measurement_dialect,
            known_source_names=self.known_source_names,
        )

        image_number_offset = RuntimeImageNumberOffset.from_table_rows(
            self.table.header,
            self.table.rows,
        )
        facts: _RuntimeMeasurementFactList = []
        for row in self.table.rows:
            row_mapping = dict(zip(self.table.header, row, strict=True))
            row_subject = RuntimeExportRowSubject(
                self.table.path,
                row_mapping,
            ).subject()
            if RuntimeMetadataMapRow(row_subject, row_mapping).matches():
                continue
            source_name = measurement_row_source_image_name(row_mapping)
            padding_indexes = _contextual_measurement_padding_indexes(
                self.table.column_context,
                self.table.header,
                row,
                feature_indexes,
                self.policy.measurement_dialect,
                known_source_names=self.known_source_names,
                padding_groups_by_index=padding_groups_by_index,
            )
            long_form_fact = _long_form_measurement_fact(
                _LongFormMeasurementContext(
                    row_mapping,
                    row_subject,
                    self.policy,
                    source_name,
                    self.known_source_names,
                    image_number_offset,
                )
            )
            if long_form_fact is not None:
                facts.append(long_form_fact)
                continue
            for index, field_name in enumerate(self.table.header):
                if index in padding_indexes:
                    continue
                if RuntimeMeasurementIdentityField(
                    self.policy.measurement_dialect
                ).field_matches(field_name):
                    continue
                subject = RuntimeExportColumnSubject(
                    self.table,
                    row_mapping,
                    index,
                    row_subject,
                ).subject()
                key = _measurement_feature_key_from_source_context(
                    _MeasurementFeatureKeySourceContext(
                        field_name,
                        subject,
                        self.policy,
                        measurement_row_qualifiers(
                            row_mapping,
                            self.policy.measurement_dialect,
                            field_name,
                        ),
                        source_name,
                        self.known_source_names,
                    )
                )
                if key is None:
                    continue
                if _is_aggregate_image_number_reference_measurement_field(field_name):
                    continue
                facts.extend(
                    RuntimeMeasurementCellFactProjection().project_cell(
                        RuntimeMeasurementCellValue(
                            key,
                            RuntimeImageNumberReferenceValue(
                                field_name,
                                row_mapping[field_name],
                                image_number_offset,
                            ).normalized(),
                            self.policy,
                        )
                    )
                )
        return tuple(facts)

    def identity_facts(self) -> _RuntimeMeasurementFacts:
        """Project object identity-only exports into semantic object-number facts."""
        facts: _RuntimeMeasurementFactList = []
        for row in self.table.rows:
            row_mapping = dict(zip(self.table.header, row, strict=True))
            row_subject = RuntimeExportRowSubject(
                self.table.path,
                row_mapping,
            ).subject()
            if row_subject.scope is not MeasurementScope.OBJECT:
                continue
            normalized_row_mapping = {
                _normalize_identifier(field_name): value
                for field_name, value in row_mapping.items()
            }
            object_number = measurement_object_label(normalized_row_mapping)
            if object_number is None:
                continue
            facts.append(
                (
                    RuntimeMeasurementFeatureKey(
                        row_subject,
                        ObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
                    ),
                    _cell_signature(str(object_number), self.policy),
                )
            )
        return tuple(facts)

    def static_wide_measurement_facts(self) -> _RuntimeMeasurementFacts | None:
        """Project wide exported tables by caching column-level feature semantics."""
        if not self.table.rows:
            return ()

        first_row = dict(zip(self.table.header, self.table.rows[0], strict=True))
        if not is_static_wide_measurement_table(
            first_row,
            measurement_qualifier_field_names(self.policy.measurement_dialect),
        ):
            return None

        subject = RuntimeExportRowSubject(self.table.path, first_row).subject()
        if RuntimeMetadataMapRow(subject, first_row).matches():
            return ()

        source_name = measurement_row_source_image_name(first_row)
        feature_columns = tuple(
            (index, key)
            for index, field_name in enumerate(self.table.header)
            if not RuntimeMeasurementIdentityField(
                self.policy.measurement_dialect
            ).field_matches(field_name)
            and not _is_aggregate_image_number_reference_measurement_field(field_name)
            for column_subject in (
                RuntimeExportColumnSubject(
                    self.table,
                    first_row,
                    index,
                    subject,
                ).subject(),
            )
            for key in (
                _measurement_feature_key_from_source_context(
                    _MeasurementFeatureKeySourceContext(
                        field_name,
                        column_subject,
                        self.policy,
                        (),
                        source_name,
                        self.known_source_names,
                    )
                ),
            )
            if key is not None
        )
        if not feature_columns:
            return ()
        image_number_offset = RuntimeImageNumberOffset.from_table_rows(
            self.table.header,
            self.table.rows,
        )
        feature_indexes = tuple(index for index, _key in feature_columns)
        padding_groups_by_index = _contextual_measurement_padding_groups(
            self.table.column_context,
            self.table.header,
            feature_indexes,
            self.policy.measurement_dialect,
            known_source_names=self.known_source_names,
        )

        facts: _RuntimeMeasurementFactList = []
        for row in self.table.rows:
            padding_indexes = _contextual_measurement_padding_indexes(
                self.table.column_context,
                self.table.header,
                row,
                feature_indexes,
                self.policy.measurement_dialect,
                known_source_names=self.known_source_names,
                padding_groups_by_index=padding_groups_by_index,
            )
            row_fact_records = tuple(
                (
                    RuntimeMeasurementFactProjectionContract.padding_group(
                        _normalize_identifier(self.table.path.stem) or "measurements",
                        self.table.header[index],
                        key,
                        self.policy.measurement_dialect,
                    ),
                    key,
                    _cell_signature(
                        str(
                            RuntimeImageNumberReferenceValue(
                                self.table.header[index],
                                row[index],
                                image_number_offset,
                            ).normalized()
                        ),
                        self.policy,
                    ),
                )
                for index, key in feature_columns
                if index not in padding_indexes
            )
            facts.extend(
                RuntimeDirectionalPairMeasurementDerivationContract(
                    self.policy,
                    self.known_source_names,
                ).derive(
                    RuntimeMeasurementFactProjectionContract.dedupe_observed_alias_records(
                        row_fact_records,
                        self.policy,
                    )
                )
            )
        return tuple(facts)


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
        else RuntimeExportRowSubject(table.path, first_row).subject()
    )
    for index in feature_column_indexes:
        subject = (
            fallback_subject
            if not isinstance(table, RuntimeTableSnapshot)
            else RuntimeExportColumnSubject(
                table,
                first_row,
                index,
                fallback_subject,
            ).subject()
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
        if (
            key is not None
            and key.belongs_to_source_qualified_feature_family(
                policy.measurement_dialect,
                (_PAIR_REGRESSION_SLOPE_FEATURE,),
            )
        ):
            return True
    return False


@dataclass
class RuntimeMeasurementTableFactRecorder:
    """Stream one runtime measurement table into semantic fact counters."""

    values_by_feature: _RuntimeMeasurementFactCounters
    context: _RuntimeMeasurementTableProjectionContext
    row_merge_cache: _RuntimeMeasurementRowMergeCache | None = None
    primary_row_identities: _RuntimeMeasurementPrimaryRowSet | None = None
    object_row_domain: RuntimeObjectMeasurementFactRowDomain | None = None

    def __post_init__(self) -> None:
        self._schema_cache: _RuntimeMeasurementRowSchemaCache = {}
        self._key_cache: _RuntimeMeasurementFeatureKeyCache = {}
        self._long_form_key_cache: _RuntimeMeasurementLongFormKeyCache = {}
        self._qualifier_render_cache: _RuntimeMeasurementQualifierRenderCache = {}
        self._padding_group_cache: _RuntimeMeasurementPaddingGroupCache = {}
        self._subject_schema_cache: dict[
            tuple[str, ...],
            _RuntimeMeasurementRowSubjectSchema,
        ] = {}
        self._aggregate_values_by_feature: _AggregateValuesByFeature = {}
        self._aggregate_input_key_cache: _AggregateMeanKeyCache = {}
        self._explicit_measurement_keys: set[RuntimeMeasurementFeatureKey] = set()
        required_projection = RequiredRuntimeMeasurementProjection(
            self.context.required_keys,
            self.context.policy,
            known_source_names=self.context.known_source_names,
        )
        self._row_required_keys = required_projection.input_keys()
        self._row_required_subjects = required_projection.subjects()

    def record(self) -> None:
        table = self.context.table
        table_subject = RuntimeMeasurementSubjectKey.from_table_subject(table.subject)
        table_padding_group = _normalize_identifier(table.name) or "measurements"
        image_number_offset = RuntimeImageNumberOffset.from_runtime_rows(
            iter_measurement_rows((table,))
        )
        fact_context = _RuntimeMeasurementFactRecordingContext(
            self.values_by_feature,
            self._explicit_measurement_keys,
            self.object_row_domain or RuntimeObjectMeasurementFactRowDomain(),
            self.context.required_keys,
            {},
        )
        for row in iter_measurement_rows((table,)):
            self._record_row(
                row,
                table_subject,
                table_padding_group,
                image_number_offset,
                fact_context,
            )
        self._record_derived_aggregate_facts()

    def _record_row(
        self,
        row: object,
        table_subject: RuntimeMeasurementSubjectKey,
        table_padding_group: str,
        image_number_offset: float,
        fact_context: _RuntimeMeasurementFactRecordingContext,
    ) -> None:
        table = self.context.table
        row_mapping = measurement_row_mapping(row)
        header = tuple(row_mapping)
        row_values = tuple(row_mapping.get(field_name) for field_name in header)
        subject_schema = self._subject_schema(header)
        row_subject_projection = RuntimeMeasurementRowSubjectProjection(
            table_subject,
            table.source_image_name,
            row_values,
            subject_schema,
        )
        subject = row_subject_projection.subject()
        if (
            self._row_required_subjects is not None
            and subject.scope is MeasurementScope.OBJECT
            and subject not in self._row_required_subjects
        ):
            return
        source_qualification = subject.bind_row_source_identity(
            row_subject_projection.source_name()
        )
        _record_runtime_primary_measurement_row_identity(
            self.primary_row_identities,
            row_mapping,
            self.context.axis_key,
            subject,
            self.context.policy,
        )
        row_context = RuntimeRowProjectionContext.from_row(
            row_mapping,
            subject,
            self.context.policy,
            source_name=source_qualification.feature_source_name,
            known_source_names=self.context.known_source_names,
            required_keys=self._row_required_keys,
            table_padding_group=table_padding_group,
            image_number_offset=image_number_offset,
            schema_cache=self._schema_cache,
            key_cache=self._key_cache,
            long_form_key_cache=self._long_form_key_cache,
            qualifier_render_cache=self._qualifier_render_cache,
            padding_group_cache=self._padding_group_cache,
        )
        row_facts = RuntimeMeasurementRowFactProjector(row_context).facts()
        if not row_facts:
            return
        aggregate_input_context = _AggregateInputRecordingContext(
            self._aggregate_values_by_feature,
            row_mapping,
            self.context.axis_key,
            self.context.required_keys,
            self._aggregate_input_key_cache,
            self.context.policy.measurement_dialect,
        )
        RuntimeRowMeasurementFactRecorder(
            fact_context,
            aggregate_input_context,
            row_merge_cache=self.row_merge_cache,
            policy=self.context.policy,
        ).record(
            row_facts,
            row_mapping=row_mapping,
            axis_key=self.context.axis_key,
        )

    def _subject_schema(
        self,
        header: tuple[str, ...],
    ) -> _RuntimeMeasurementRowSubjectSchema:
        subject_schema = self._subject_schema_cache.get(header)
        if subject_schema is None:
            subject_schema = _runtime_measurement_row_subject_schema(header)
            self._subject_schema_cache[header] = subject_schema
        return subject_schema

    def _record_derived_aggregate_facts(self) -> None:
        policy = self.context.policy
        for (
            mean_key,
            _row_identity,
        ), accumulator in self._aggregate_values_by_feature.items():
            if (
                not accumulator.has_values
                or mean_key in self._explicit_measurement_keys
            ):
                continue
            if (
                self.context.required_keys is not None
                and mean_key not in self.context.required_keys
            ):
                continue
            RuntimeMeasurementFactCounters(self.values_by_feature).counter(mean_key)[
                _cell_signature(str(accumulator.mean), policy)
            ] += 1


@dataclass(frozen=True, slots=True)
class RuntimeRowProjectionEngine(Generic[_RuntimeRowProjectionValueT]):
    context: RuntimeRowProjectionContext
    value_projector: RuntimeRowValueProjection[_RuntimeRowProjectionValueT]
    long_form_projector: RuntimeRowLongFormProjection[_RuntimeRowProjectionValueT]

    def project(self) -> RuntimeRowProjection[_RuntimeRowProjectionValueT]:
        """Project one runtime row through shared schema/key/padding caches."""
        context = self.context
        if RuntimeMetadataMapRow(context.subject, context.row).matches():
            return runtime_row_projection()

        header = tuple(context.row)
        row_schema = RuntimeMeasurementRowSchemaProjector(context, header).schema()
        row_values = tuple(context.row.get(field_name) for field_name in header)
        long_form_projection = self._long_form_projection(row_schema, row_values)
        if long_form_projection is not None:
            return long_form_projection
        return self._wide_projection(header, row_schema, row_values)

    def _long_form_projection(
        self,
        row_schema: _RuntimeMeasurementRowSchema,
        row_values: tuple[object, ...],
    ) -> RuntimeRowProjection[_RuntimeRowProjectionValueT] | None:
        context = self.context
        if (
            not row_schema.long_form_feature_indexes
            or not row_schema.long_form_value_indexes
        ):
            return None
        long_form_fact = RuntimeMeasurementLongFormFactProjector(
            _CachedLongFormMeasurementContext.from_runtime_row_projection(
                context,
                row_values,
                row_schema.long_form_feature_indexes,
                row_schema.long_form_value_indexes,
            )
        ).fact()
        if long_form_fact is None:
            return None
        if (
            context.required_keys is not None
            and long_form_fact[0] not in context.required_keys
        ):
            return runtime_row_projection(long_form=True)
        return runtime_row_projection(
            (
                ((context.subject, context.source_name, ()), key, value)
                for key, value in self.long_form_projector.project(long_form_fact)
                if context.required_keys is None or key in context.required_keys
            ),
            long_form=True,
        )

    def _wide_projection(
        self,
        header: tuple[str, ...],
        row_schema: _RuntimeMeasurementRowSchema,
        row_values: tuple[object, ...],
    ) -> RuntimeRowProjection[_RuntimeRowProjectionValueT]:
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
        return runtime_row_projection(
            (
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
            or RuntimeMeasurementValuePresence(value).is_present()
        )
        if (
            context.required_keys is not None
            and key not in context.required_keys
            and not _runtime_value_is_mapping(value)
        ):
            return ()
        projected_values = self.value_projector.project(key, value, context.policy)
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
            qualifiers = measurement_row_qualifiers_from_indexed_values_cached(
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
            padding_group = RuntimeMeasurementFactProjectionContract.padding_group(
                context.table_padding_group,
                field_name,
                key,
                context.policy.measurement_dialect,
            )
            context.padding_group_cache[cache_key] = padding_group
        return padding_group


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowFactProjector:
    """Project one runtime measurement row into semantic fact views."""

    context: RuntimeRowProjectionContext

    def facts(self) -> _RuntimeMeasurementFacts:
        """Project one runtime row using table-local schema/key caches."""
        projection = RuntimeRowProjectionEngine(
            self.context,
            RuntimeMeasurementCellFactProjection(),
            RuntimeRowLongFormFactProjection(),
        ).project()
        row_facts = tuple(
            (key, value)
            for _padding_group, key, value in RuntimeMeasurementFactProjectionContract.observed_records(
                projection.records,
                self.context.policy,
            )
        )
        if projection.long_form:
            return row_facts
        derived_facts = RuntimeDirectionalPairMeasurementDerivationContract(
            self.context.policy,
            self.context.known_source_names,
        ).derive(
            RuntimeMeasurementFactProjectionContract.dedupe_alias_facts(row_facts),
        )
        if self.context.required_keys is not None:
            return tuple(
                (key, value)
                for key, value in derived_facts
                if key in self.context.required_keys
            )
        return derived_facts

    def numeric_values(self) -> _RuntimeNumericMeasurementValues:
        """Project numeric runtime row values without building cell signatures."""
        projection = RuntimeRowProjectionEngine(
            self.context,
            RuntimeMeasurementCellNumericProjection(),
            RuntimeRowLongFormNumericProjection(),
        ).project()
        row_values_by_key = _dedupe_numeric_measurement_values(
            (key, value)
            for _padding_group, key, value in projection.records
        )
        if projection.long_form:
            return row_values_by_key
        if not any(
            key.belongs_to_source_qualified_feature_family(
                self.context.policy.measurement_dialect,
                (_PAIR_REGRESSION_SLOPE_FEATURE,),
            )
            for key, _value in row_values_by_key
        ):
            return row_values_by_key

        derived_facts = RuntimeDirectionalPairMeasurementDerivationContract(
            self.context.policy,
            self.context.known_source_names,
        ).derive(
            tuple(
                (key, _cell_signature(repr(value), self.context.policy))
                for key, value in row_values_by_key
            ),
        )
        if self.context.required_keys is not None:
            derived_facts = tuple(
                (key, value)
                for key, value in derived_facts
                if key in self.context.required_keys
            )
        return tuple(
            (key, numeric_value)
            for key, value in derived_facts
            if (numeric_value := _cell_signature_numeric_value(value)) is not None
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowSchemaProjector:
    """Project a runtime measurement row header into cached row schema."""

    context: RuntimeRowProjectionContext
    header: tuple[str, ...]

    def schema(self) -> _RuntimeMeasurementRowSchema:
        cached_schema = self.context.schema_cache.get(self.header)
        if cached_schema is not None:
            return cached_schema

        normalized_fields = tuple(
            _normalize_identifier(field) for field in self.header
        )
        aggregate_reference_indexes = frozenset(
            index
            for index, field_name in enumerate(self.header)
            if _is_aggregate_image_number_reference_measurement_field(field_name)
        )
        normalized_field_indexes = {
            field_name: index
            for index, field_name in enumerate(normalized_fields)
        }
        feature_indexes = tuple(
            index
            for index, field_name in enumerate(normalized_fields)
            if not RuntimeMeasurementIdentityField(
                self.context.policy.measurement_dialect
            ).normalized_field_matches(field_name)
            and index not in aggregate_reference_indexes
        )
        qualifier_indexes = {
            qualifier: tuple(
                normalized_field_indexes.get(field_name)
                for field_name in qualifier.field_names
            )
            for qualifier in self.context.policy.measurement_dialect.row_qualifiers
        }
        qualifiers_by_index = {
            index: tuple(
                (qualifier, qualifier_indexes[qualifier])
                for qualifier in self.context.policy.measurement_dialect.row_qualifiers
                if row_qualifier_applies_to_field(
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
            RuntimeMeasurementFieldIndexMap(normalized_field_indexes).indexes_for(
                _MEASUREMENT_FEATURE_NAME_FIELDS
            ),
            RuntimeMeasurementFieldIndexMap(normalized_field_indexes).indexes_for(
                _MEASUREMENT_VALUE_FIELDS
            ),
        )
        self.context.schema_cache[self.header] = cached_schema
        return cached_schema


@dataclass(frozen=True, slots=True)
class RuntimeDirectionalPairMeasurementDerivationContract:
    """SSOT for directional pair facts derivable from equivalent orientations."""

    policy: RuntimeEquivalencePolicy
    known_source_names: tuple[str, ...] = ()

    @property
    def feature_families(self) -> tuple[str, ...]:
        """Return pair feature families participating in orientation derivation."""
        return (_PAIR_REGRESSION_SLOPE_FEATURE, _PAIR_CORRELATION_FEATURE)

    def required_input_keys(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        """Return orientation inputs needed to satisfy a required pair key."""
        input_keys: list[RuntimeMeasurementFeatureKey] = []
        reversed_key = key.reversed_source_pair_feature_key(
            self.policy.measurement_dialect,
            self.feature_families,
            self.known_source_names,
        )
        if reversed_key is not None:
            input_keys.append(reversed_key)

        if key.belongs_to_source_qualified_feature_family(
            self.policy.measurement_dialect,
            (_PAIR_REGRESSION_SLOPE_FEATURE,),
        ):
            pair = key.source_pair(
                self.policy.measurement_dialect,
                (_PAIR_REGRESSION_SLOPE_FEATURE,),
                self.known_source_names,
            )
            if pair is not None:
                input_keys.extend(
                    self.source_pair_feature_key(
                        key,
                        _PAIR_CORRELATION_FEATURE,
                        source_name,
                    )
                    for source_name in (pair.source_name, pair.reversed_source_name)
                )
        return tuple(dict.fromkeys(input_keys))

    def source_pair_feature_key(
        self,
        key: RuntimeMeasurementFeatureKey,
        feature_name: str,
        source_name: str,
    ) -> RuntimeMeasurementFeatureKey:
        """Return ``feature_name`` encoded for ``key``'s pair source orientation."""
        return RuntimeMeasurementFeatureKey.from_source_qualified_feature(
            key.subject,
            feature_name,
            source_name,
            self.policy.measurement_dialect,
            key.statistic,
        )

    def derive(
        self,
        facts: _RuntimeMeasurementFacts,
    ) -> _RuntimeMeasurementFacts:
        """Derive mathematically equivalent directional pair facts."""
        slope_facts = tuple(
            (key, value)
            for key, value in facts
            if key.belongs_to_source_qualified_feature_family(
                self.policy.measurement_dialect,
                (_PAIR_REGRESSION_SLOPE_FEATURE,),
            )
        )
        if not slope_facts:
            return facts

        derived: _RuntimeMeasurementFactList = []
        values_by_key = dict(facts)
        for key, slope_value in slope_facts:
            pair = key.source_pair(
                self.policy.measurement_dialect,
                (_PAIR_REGRESSION_SLOPE_FEATURE,),
                self.known_source_names,
            )
            if pair is None:
                continue
            correlation_value = self.source_pair_correlation_value(
                key,
                pair.source_name,
                pair.reversed_source_name,
                values_by_key,
            )
            if correlation_value is None:
                continue
            reverse_slope = _reverse_regression_slope(
                correlation_value,
                slope_value,
            )
            if reverse_slope is None:
                continue
            reversed_key = key.reversed_source_pair_feature_key(
                self.policy.measurement_dialect,
                (_PAIR_REGRESSION_SLOPE_FEATURE,),
                self.known_source_names,
            )
            if reversed_key is None:
                continue
            derived.append(
                (
                    reversed_key,
                    _cell_signature(repr(reverse_slope), self.policy),
                )
            )
        if not derived:
            return facts
        return RuntimeMeasurementFactProjectionContract.dedupe_alias_facts((*facts, *derived))

    def source_pair_correlation_value(
        self,
        key: RuntimeMeasurementFeatureKey,
        source_name: str,
        reversed_source_name: str,
        values_by_key: Mapping[RuntimeMeasurementFeatureKey, RuntimeCellSignature],
    ) -> RuntimeCellSignature | None:
        """Return correlation for either orientation of ``key``'s source pair."""
        for candidate_key in (
            self.source_pair_feature_key(
                key,
                _PAIR_CORRELATION_FEATURE,
                source_name,
            ),
            self.source_pair_feature_key(
                key,
                _PAIR_CORRELATION_FEATURE,
                reversed_source_name,
            ),
        ):
            correlation_value = values_by_key.get(candidate_key)
            if correlation_value is not None:
                return correlation_value
        return None


class RuntimeMeasurementFactProjectionContract:
    """Nominal contract for projecting runtime cells into measurement facts."""

    @classmethod
    def is_observed_value(cls, value: RuntimeCellSignature) -> bool:
        """Return whether ``value`` represents an observed measurement."""
        return not RuntimeCellMissingStrategy.for_kind(value.kind).is_missing(value)

    @classmethod
    def observed_padding_groups(
        cls,
        records: Iterable[_RuntimeRowProjectionRecord[RuntimeCellSignature]],
        policy: RuntimeEquivalencePolicy,
    ) -> frozenset[_RuntimeMeasurementPaddingGroup]:
        """Return padding groups that carry observed measurement facts."""
        records_by_group: dict[
            _RuntimeMeasurementPaddingGroup,
            list[tuple[RuntimeMeasurementFeatureKey, RuntimeCellSignature]],
        ] = {}
        for padding_group, key, value in records:
            records_by_group.setdefault(padding_group, []).append((key, value))

        observed_groups: set[_RuntimeMeasurementPaddingGroup] = set()
        for padding_group, group_records in records_by_group.items():
            anchor_values = tuple(
                value
                for key, value in group_records
                if object_measurement_feature_has_role(
                    key,
                    ObjectMeasurementFeatureRole.MEASURED_OBJECT_ANCHOR,
                    policy.measurement_dialect,
                )
            )
            if anchor_values:
                if any(cls.is_observed_value(value) for value in anchor_values):
                    observed_groups.add(padding_group)
                continue
            if any(cls.is_observed_value(value) for _key, value in group_records):
                observed_groups.add(padding_group)
        return frozenset(observed_groups)

    @classmethod
    def padding_group(
        cls,
        table_group: str,
        field_name: str,
        key: RuntimeMeasurementFeatureKey,
        dialect: RuntimeMeasurementDialect,
    ) -> _RuntimeMeasurementPaddingGroup:
        """Return the row-padding family for a measurement field."""
        normalized_field = _normalize_identifier(field_name)
        parts = tuple(part for part in normalized_field.split("_") if part)
        feature_group = RuntimeMeasurementNamePartsProjection(
            parts,
            dialect,
        ).category_prefix() or (table_group,)
        return key.subject, key.source_name, feature_group

    @classmethod
    def observed_records(
        cls,
        records: Iterable[_RuntimeRowProjectionRecord[RuntimeCellSignature]],
        policy: RuntimeEquivalencePolicy,
    ) -> _RuntimeRowProjectionRecords[RuntimeCellSignature]:
        """Return records whose padding family has an observed anchor/value."""
        materialized = tuple(records)
        observed_padding_groups = cls.observed_padding_groups(materialized, policy)
        return tuple(
            record
            for record in materialized
            if record[0] in observed_padding_groups
        )

    @classmethod
    def dedupe_observed_alias_records(
        cls,
        records: Iterable[_RuntimeRowProjectionRecord[RuntimeCellSignature]],
        policy: RuntimeEquivalencePolicy,
    ) -> _RuntimeMeasurementFacts:
        """Filter unobserved padding groups and collapse same-row aliases."""
        return cls.dedupe_alias_facts(
            (key, value)
            for _padding_group, key, value in cls.observed_records(records, policy)
        )

    @classmethod
    def dedupe_observed_qualified_records(
        cls,
        records: Iterable[
            tuple[
                _RuntimeMeasurementPaddingGroup,
                RuntimeMeasurementFeatureKey,
                RuntimeCellSignature,
                bool,
            ]
        ],
        policy: RuntimeEquivalencePolicy,
    ) -> _RuntimeMeasurementFacts:
        """Filter unobserved padding groups and collapse qualified aliases."""
        materialized = tuple(records)
        observed_padding_groups = cls.observed_padding_groups(
            (
                (padding_group, key, value)
                for padding_group, key, value, _qualified_observation in materialized
            ),
            policy,
        )
        return cls.dedupe_qualified_records(
            (key, value, qualified_observation)
            for padding_group, key, value, qualified_observation in materialized
            if padding_group in observed_padding_groups
        )

    @classmethod
    def dedupe_alias_facts(
        cls,
        facts: Iterable[_RuntimeMeasurementFact],
    ) -> _RuntimeMeasurementFacts:
        """Collapse same-row aliases that map to the same semantic feature key."""
        values_by_key: dict[RuntimeMeasurementFeatureKey, RuntimeCellSignature] = {}
        for key, value in facts:
            if not cls.is_observed_value(value):
                continue
            current = values_by_key.get(key)
            if current is None or (
                RuntimeCellMissingStrategy.for_kind(current.kind).is_missing(current)
                and not RuntimeCellMissingStrategy.for_kind(value.kind).is_missing(value)
            ):
                values_by_key[key] = value
        return tuple(values_by_key.items())

    @classmethod
    def dedupe_qualified_records(
        cls,
        facts: Iterable[
            tuple[RuntimeMeasurementFeatureKey, RuntimeCellSignature, bool]
        ],
    ) -> _RuntimeMeasurementFacts:
        """Collapse aliases unless field normalization intentionally dropped a qualifier."""
        values_by_key: dict[RuntimeMeasurementFeatureKey, list[RuntimeCellSignature]] = {}
        qualified_by_key: dict[RuntimeMeasurementFeatureKey, bool] = {}
        for key, value, qualified_observation in facts:
            if not cls.is_observed_value(value):
                continue
            current_values = values_by_key.get(key)
            if current_values is None:
                values_by_key[key] = [value]
                qualified_by_key[key] = qualified_observation
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
    category_prefix = RuntimeMeasurementNamePartsProjection(
        parts,
        dialect,
    ).category_prefix()
    if category_prefix:
        parts = parts[len(category_prefix) :]
    parts, _source_names = RuntimeMeasurementNamePartsProjection(
        parts,
        dialect,
        known_source_names,
    ).source_qualifier_tokens()
    return (
        RuntimeMeasurementNamePartsProjection(parts, dialect).semantic_core_parts()
        != parts
    )


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
        if any(RuntimeMeasurementCellPresence(row_values[index]).is_present() for index in indexes):
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
    if RuntimeMeasurementIdentityField(dialect).field_matches(field_name):
        return None

    normalized_field = _normalize_identifier(field_name)
    parts = tuple(part for part in normalized_field.split("_") if part)
    if not parts:
        return None
    feature_group = RuntimeMeasurementNamePartsProjection(
        parts,
        dialect,
    ).category_prefix() or parts[:1]
    _feature_name, source_name = SemanticCoreFeatureAndSourceNameProjection(
        normalized_field,
        dialect,
        known_source_names,
    ).project()
    return normalized_context, feature_group, source_name


@lru_cache(maxsize=256)
def _runtime_value_type_is_mapping(value_type: type[object]) -> bool:
    return issubclass(value_type, Mapping)


def _runtime_value_is_mapping(value: object) -> bool:
    return _runtime_value_type_is_mapping(type(value))


@lru_cache(maxsize=131072)
def _cached_runtime_cell_signature(
    text: str,
    numeric_decimal_places: int,
) -> RuntimeCellSignature:
    stripped = text.strip()
    if not stripped:
        return RuntimeCellSignature(RuntimeCellValueKind.EMPTY, "")
    try:
        numeric = float(stripped)
    except ValueError:
        return RuntimeCellSignature(RuntimeCellValueKind.TEXT, stripped)
    if math.isnan(numeric):
        canonical = "nan"
    elif math.isinf(numeric):
        canonical = "inf" if numeric > 0 else "-inf"
    else:
        rounded = round(numeric, numeric_decimal_places)
        canonical = repr(0.0 if rounded == 0 else rounded)
    return RuntimeCellSignature(RuntimeCellValueKind.NUMBER, canonical)


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
    feature_name = RuntimeMeasurementRowMapping(context.row).first_value(_MEASUREMENT_FEATURE_NAME_FIELDS)
    value = RuntimeMeasurementRowMapping(context.row).first_value(_MEASUREMENT_VALUE_FIELDS)
    if feature_name is None or value is None:
        return None
    if _is_aggregate_image_number_reference_measurement_field(str(feature_name)):
        return None
    value = RuntimeImageNumberReferenceValue(
        str(feature_name),
        value,
        context.image_number_offset,
    ).normalized()
    aggregate_key = _aggregate_measurement_feature_key(
        str(feature_name),
        context.subject,
        context.policy,
        known_source_names=context.known_source_names,
    )
    if aggregate_key is not None:
        return aggregate_key, RuntimeMeasurementCellSignatureProjection(value, context.policy).signature()
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
        RuntimeMeasurementFeatureKey.from_source_qualified_feature(
            context.subject,
            canonical_feature_name,
            canonical_source_name,
            context.policy.measurement_dialect,
        ),
        RuntimeMeasurementCellSignatureProjection(value, context.policy).signature(),
    )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementLongFormFactProjector:
    """Project one cached long-form measurement row into a semantic fact."""

    context: _CachedLongFormMeasurementContext

    def fact(self) -> _RuntimeLongFormMeasurementFact:
        feature_name = RuntimeIndexedRowValues(self.context.row_values).first_at(
            self.context.feature_indexes
        )
        value = RuntimeIndexedRowValues(self.context.row_values).first_at(
            self.context.value_indexes
        )
        if feature_name is None or value is None:
            return None
        feature_text = str(feature_name)
        if _is_aggregate_image_number_reference_measurement_field(feature_text):
            return None
        value = RuntimeImageNumberReferenceValue(
            feature_text,
            value,
            self.context.image_number_offset,
        ).normalized()
        cache_key = (
            self.context.subject,
            self.context.source_name,
            feature_text,
        )
        key = self.context.key_cache.get(cache_key, _CACHE_MISS)
        if key is _CACHE_MISS:
            key = self._feature_key(feature_text)
            self.context.key_cache[cache_key] = key
        if key is None:
            return None
        return key, RuntimeMeasurementCellSignatureProjection(value, self.context.policy).signature()

    def _feature_key(
        self,
        feature_text: str,
    ) -> RuntimeMeasurementFeatureKey | None:
        aggregate_key = _aggregate_measurement_feature_key(
            feature_text,
            self.context.subject,
            self.context.policy,
            known_source_names=self.context.known_source_names,
        )
        if aggregate_key is not None:
            return aggregate_key
        canonical_feature_name, canonical_source_name = (
            _canonical_measurement_feature_name_and_source(
                feature_text,
                self.context.policy,
                source_name=self.context.source_name,
                known_source_names=self.context.known_source_names,
            )
        )
        if not canonical_feature_name:
            return None
        return RuntimeMeasurementFeatureKey.from_source_qualified_feature(
            self.context.subject,
            canonical_feature_name,
            canonical_source_name,
            self.context.policy.measurement_dialect,
        )


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
    return RuntimeMeasurementFeatureKey.from_source_qualified_feature(
        subject,
        feature_name,
        feature_source_name,
        policy.measurement_dialect,
        qualifiers=qualifiers,
    )


@dataclass(frozen=True, slots=True)
class RequiredRuntimeMeasurementProjection:
    """Project user-required measurement keys into runtime input domains."""

    required_keys: _RuntimeRequiredMeasurementKeys
    policy: RuntimeEquivalencePolicy
    known_source_names: tuple[str, ...] = ()

    def input_keys(self) -> _RuntimeRequiredMeasurementKeys:
        if self.required_keys is None:
            return None
        keys: set[RuntimeMeasurementFeatureKey] = set(self.required_keys)
        pair_derivation = RuntimeDirectionalPairMeasurementDerivationContract(
            self.policy,
            self.known_source_names,
        )
        for key in self.required_keys:
            if key.statistic == MeasurementStatistic.MEAN.value:
                keys.add(
                    RuntimeMeasurementFeatureKey(
                        key.subject,
                        key.feature_name,
                        MeasurementStatistic.VALUE.value,
                        key.source_name,
                    )
                )
            keys.update(pair_derivation.required_input_keys(key))
        return frozenset(keys)

    def subjects(self) -> frozenset[RuntimeMeasurementSubjectKey] | None:
        input_keys = self.input_keys()
        if input_keys is None:
            return None
        subjects: set[RuntimeMeasurementSubjectKey] = set()
        for key in input_keys:
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

    def object_identifier_keys(
        self,
        subject: RuntimeMeasurementSubjectKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        if self.required_keys is None:
            return (
                RuntimeMeasurementFeatureKey(
                    subject,
                    ObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
                ),
            )
        return tuple(
            key
            for key in sorted(self.required_keys, key=lambda item: item.sort_key)
            if key.subject == subject
            and object_measurement_feature_has_role(
                key,
                ObjectMeasurementFeatureRole.IDENTIFIER,
                self.policy.measurement_dialect,
            )
        )

    def object_location_feature_names(
        self,
        subject: RuntimeMeasurementSubjectKey,
        *,
        statistic: MeasurementStatistic,
    ) -> frozenset[str] | None:
        if self.required_keys is None:
            return None
        return frozenset(
            key.feature_name
            for key in self.required_keys
            if key.subject == subject
            and key.statistic == statistic.value
            and key.source_name is None
            and object_measurement_feature_has_role(
                RuntimeMeasurementFeatureKey(
                    key.subject,
                    key.feature_name,
                    MeasurementStatistic.VALUE.value,
                    source_name=key.source_name,
                ),
                ObjectMeasurementFeatureRole.LOCATION,
                self.policy.measurement_dialect,
            )
        )


@dataclass(frozen=True, slots=True)
class RuntimeAggregateFeatureIdentity:
    """Parsed aggregate measurement feature identity."""

    aggregate: str
    object_name_parts: tuple[str, ...]
    feature_parts: tuple[str, ...]

    @classmethod
    def from_parts(
        cls,
        parts: tuple[str, ...],
        dialect: RuntimeMeasurementDialect,
    ) -> "RuntimeAggregateFeatureIdentity | None":
        """Parse an aggregate feature using the runtime measurement dialect."""
        if len(parts) < 3 or parts[0] not in _MEASUREMENT_AGGREGATE_PREFIXES:
            return None
        object_name_parts, feature_parts = _aggregate_object_and_feature_parts(
            parts[1:],
            dialect,
        )
        if not object_name_parts or not feature_parts:
            return None
        return cls(
            aggregate=parts[0],
            object_name_parts=object_name_parts,
            feature_parts=feature_parts,
        )

    @classmethod
    def candidates_from_parts(
        cls,
        parts: tuple[str, ...],
    ) -> tuple["RuntimeAggregateFeatureIdentity", ...]:
        """Return all aggregate object/feature splits for family-owned suffixes."""
        if len(parts) < 3 or parts[0] not in _MEASUREMENT_AGGREGATE_PREFIXES:
            return ()
        return tuple(
            cls(
                aggregate=parts[0],
                object_name_parts=parts[1:feature_start_index],
                feature_parts=parts[feature_start_index:],
            )
            for feature_start_index in range(2, len(parts))
        )

    @property
    def object_name(self) -> str:
        return "_".join(self.object_name_parts)

    @property
    def feature_name(self) -> str:
        return "_".join(self.feature_parts)

    def canonical_feature_name_and_source(
        self,
        policy: RuntimeEquivalencePolicy,
        *,
        known_source_names: tuple[str, ...],
    ) -> tuple[str, str | None]:
        """Return the canonical runtime feature/source for the aggregate child feature."""
        return _canonical_measurement_feature_name_and_source(
            self.feature_name,
            policy,
            source_name=None,
            known_source_names=known_source_names,
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
    if len(parts) >= 2 and parts[0] == MeasurementStatistic.COUNT.value:
        return RuntimeMeasurementFeatureKeyFactory(
            RuntimeMeasurementSubjectKey(
                MeasurementScope.OBJECT,
                "_".join(parts[1:]),
            ),
            ObjectCoreMeasurementFeature.OBJECT_COUNT.value,
            MeasurementStatistic.COUNT.value,
        ).key()
    aggregate_identity = RuntimeAggregateFeatureIdentity.from_parts(
        parts,
        policy.measurement_dialect,
    )
    if aggregate_identity is None:
        return None
    feature_name, source_name = aggregate_identity.canonical_feature_name_and_source(
        policy,
        known_source_names=known_source_names,
    )
    return RuntimeMeasurementFeatureKey.from_source_qualified_feature(
        RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            aggregate_identity.object_name,
        ),
        feature_name,
        source_name,
        policy.measurement_dialect,
        aggregate_identity.aggregate,
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


@dataclass(frozen=True, slots=True)
class RuntimeObjectLabelMeasurementFactProjector:
    """Project object-label runtime values into implicit object measurement facts."""

    context: _ObjectLabelMeasurementContext

    def facts(self) -> _RuntimeMeasurementFacts:
        if not _object_label_measurements_required(
            self.context.object_name,
            self.context.required_keys,
        ):
            return ()
        return (
            *self.count_facts(),
            *self.location_facts(),
        )

    def identifier_facts(self) -> _RuntimeMeasurementFacts:
        if self.context.object_name is None:
            return ()
        subject = RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            self.context.object_name,
        )
        keys = RequiredRuntimeMeasurementProjection(
            self.context.required_keys,
            self.context.policy,
        ).object_identifier_keys(subject)
        if not keys:
            return ()
        label_array = _runtime_object_label_array(self.context.labels)
        if label_array is None:
            return ()

        object_number_domains = ObjectIdentifierDomainProjectionStrategy.for_context(
            self.context
        ).domains(self.context)
        facts: _RuntimeMeasurementFactList = []
        emitted_counts_by_key: dict[
            RuntimeMeasurementFeatureKey,
            Counter[RuntimeCellSignature],
        ] = {key: Counter() for key in keys}
        for object_ids in object_number_domains:
            for key in keys:
                explicit_counter = (
                    self.context.values_by_feature.get(key, Counter())
                    if subject in self.context.object_identifier_subjects
                    else Counter()
                )
                emitted_counter = emitted_counts_by_key[key]
                for object_id in object_ids:
                    signature = _cell_signature(str(object_id), self.context.policy)
                    if emitted_counter[signature] < explicit_counter[signature]:
                        emitted_counter[signature] += 1
                        continue
                    emitted_counter[signature] += 1
                    facts.append((key, signature))
        return tuple(facts)

    def count_facts(self) -> _RuntimeMeasurementFacts:
        if self.context.object_name is None:
            return ()
        subject = RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            self.context.object_name,
        )
        if subject in self.context.object_count_subjects:
            return ()
        key = RuntimeMeasurementFeatureKey(
            subject,
            ObjectCoreMeasurementFeature.OBJECT_COUNT.value,
            MeasurementStatistic.COUNT.value,
        )
        if self.context.required_keys is not None and key not in self.context.required_keys:
            return ()
        label_array = _runtime_object_label_array(self.context.labels)
        if label_array is None:
            return ()
        projections = ObjectLabelMeasurementProjectionStrategy.for_scope(
            self.context.domain_scope
        ).projections(self.context)
        return tuple(
            (
                key,
                _cell_signature(
                    str(len(projection.object_ids)),
                    self.context.policy,
                ),
            )
            for projection in projections
        )

    def location_facts(self) -> _RuntimeMeasurementFacts:
        if self.context.object_name is None:
            return ()
        subject = RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            self.context.object_name,
        )
        required_projection = RequiredRuntimeMeasurementProjection(
            self.context.required_keys,
            self.context.policy,
        )
        required_feature_names = required_projection.object_location_feature_names(
            subject,
            statistic=MeasurementStatistic.VALUE,
        )
        required_mean_feature_names = required_projection.object_location_feature_names(
            subject,
            statistic=MeasurementStatistic.MEAN,
        )
        if subject in self.context.object_location_subjects:
            required_feature_names = frozenset()
        if subject in self.context.object_location_aggregate_subjects:
            required_mean_feature_names = frozenset()
        if (
            required_feature_names is not None
            and not required_feature_names
            and required_mean_feature_names is not None
            and not required_mean_feature_names
        ):
            return ()
        label_array = _runtime_object_label_array(self.context.labels)
        if label_array is None:
            return ()
        projections = ObjectLabelMeasurementProjectionStrategy.for_scope(
            self.context.domain_scope
        ).projections(self.context)
        if not any(projection.object_ids for projection in projections):
            return ()

        facts: _RuntimeMeasurementFactList = []
        include_missing_locations = (
            label_array.ndim <= 2 and self.context.declared_object_count is not None
        )
        for projection in projections:
            facts.extend(
                _object_location_measurement_facts_for_plane(
                    projection.labels,
                    subject,
                    self.context.policy,
                    required_feature_names=required_feature_names,
                    required_mean_feature_names=required_mean_feature_names,
                    object_ids=projection.object_ids,
                    include_missing=include_missing_locations,
                )
            )
        return tuple(facts)


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

    coordinate_arrays = tuple(
        (
            feature_name,
            ObjectLocationCoordinateValues(
                coordinate.values[object_id_indexes],
                coordinate.include_missing,
            ),
        )
        for feature_name, coordinate in object_location_coordinate_arrays(
            axis_centers,
            counts,
        )
    )
    raw_facts = tuple(
        fact
        for feature_name, coordinate in coordinate_arrays
        if required_feature_names is None or feature_name in required_feature_names
        for fact in _object_location_feature_facts(
            subject,
            feature_name,
            coordinate.values,
            policy,
            include_missing=include_missing and coordinate.include_missing,
        )
    )
    mean_facts = tuple(
        fact
        for feature_name, coordinate in coordinate_arrays
        if required_mean_feature_names is None
        or feature_name in required_mean_feature_names
        for fact in _object_location_mean_feature_fact(
            subject,
            feature_name,
            coordinate.values,
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


@dataclass(frozen=True, slots=True)
class RuntimeObjectLabelInstanceCatalog:
    """Canonical object-label instance domains available to measurement projection."""

    counts_by_subject: Mapping[RuntimeMeasurementSubjectKey, int]
    domains_by_subject: Mapping[
        RuntimeMeasurementSubjectKey,
        tuple[ObjectInstanceKey, ...],
    ]

    @classmethod
    def from_records(
        cls,
        records: Iterable[object],
    ) -> "RuntimeObjectLabelInstanceCatalog":
        """Build the catalog from runtime object-label artifacts."""
        counts: dict[RuntimeMeasurementSubjectKey, int] = {}
        named_plane_domains: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
        for record in records:
            object_labels = ObjectLabelSet.from_runtime_value(record.value)
            subject = RuntimeMeasurementSubjectKey(
                MeasurementScope.OBJECT,
                object_labels.name,
            )
            counts[subject] = max(
                counts.get(subject, 0),
                ObjectLabelIdDomainStrategy.for_value(
                    object_labels.labels
                ).max_present_id(object_labels.labels),
            )
            named_plane_domains.append(
                (
                    object_labels.name,
                    dense_object_label_identity_domains(
                        object_labels.labels,
                        domain_scope=ObjectLabelDomainScope.PLANE,
                    ),
                )
            )
        domains = ObjectLabelInstanceDomains.from_named_plane_domains(
            named_plane_domains
        )
        domains_by_subject = {
            RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name): domain
            for object_name, domain in domains.domains_by_name.items()
        }
        return cls(
            counts_by_subject=MappingProxyType(counts),
            domains_by_subject=MappingProxyType(domains_by_subject),
        )

    @classmethod
    def empty(cls) -> "RuntimeObjectLabelInstanceCatalog":
        """Return an empty typed catalog."""
        return cls(
            counts_by_subject=MappingProxyType({}),
            domains_by_subject=MappingProxyType({}),
        )

    def count_for_subject(
        self,
        subject: RuntimeMeasurementSubjectKey,
    ) -> int:
        """Return the declared object-count extent for a canonical subject."""
        return self.counts_by_subject.get(subject, 0)

    def domain_for_subject(
        self,
        subject: RuntimeMeasurementSubjectKey,
    ) -> tuple[ObjectInstanceKey, ...]:
        """Return the declared object-instance domain for a canonical subject."""
        return self.domains_by_subject.get(subject, ())


@dataclass(frozen=True, slots=True)
class ObjectInstanceKeyPlaneAlignmentContext:
    """Object-instance relationship keys plus measured child value identity."""

    child_ids_by_parent: _ObjectInstanceChildrenByParent
    values_by_child_id: Mapping[ObjectInstanceKey, float]

    @property
    def child_slice_indices(self) -> frozenset[int | None]:
        return frozenset(
            child_id.slice_index
            for child_ids in self.child_ids_by_parent.values()
            for child_id in child_ids
        )

    @property
    def value_slice_indices(self) -> frozenset[int | None]:
        return frozenset(child_id.slice_index for child_id in self.values_by_child_id)

    @property
    def single_value_slice_index(self) -> int | None:
        slice_indices = self.value_slice_indices
        if len(slice_indices) != 1:
            return None
        slice_index = next(iter(slice_indices))
        return slice_index if slice_index is not None else None


class ObjectInstanceKeyPlaneAlignmentStrategy(
    MostDerivedContextStrategyMixin[ObjectInstanceKeyPlaneAlignmentContext],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal projection between relationship and measurement-row plane identity."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True

    strategy_key: ClassVar[str | None] = None

    @classmethod
    def align_child_ids_by_parent(
        cls,
        child_ids_by_parent: _ObjectInstanceChildrenByParent,
        values_by_child_id: Mapping[ObjectInstanceKey, float],
    ) -> _ObjectInstanceChildrenByParent:
        """Align relationship child identities to the measured value domain."""
        context = ObjectInstanceKeyPlaneAlignmentContext(
            child_ids_by_parent=child_ids_by_parent,
            values_by_child_id=values_by_child_id,
        )
        strategy = cls.for_context(
            context,
            error_subject="Object instance plane alignment",
        )
        if strategy is None:
            raise ValueError("Object instance plane alignment requires a strategy.")
        return strategy.align(context)

    @abstractmethod
    def align(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> _ObjectInstanceChildrenByParent:
        """Return relationship child identities projected into value identity."""


class UnmodifiedObjectInstanceKeyPlaneAlignmentStrategy(
    ObjectInstanceKeyPlaneAlignmentStrategy
):
    """Preserve relationship identity when no plane projection is required."""

    strategy_key = "unmodified"

    def matches(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> bool:
        del context
        return True

    def align(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> _ObjectInstanceChildrenByParent:
        return context.child_ids_by_parent


class SingleSliceValueObjectInstanceKeyPlaneAlignmentStrategy(
    UnmodifiedObjectInstanceKeyPlaneAlignmentStrategy
):
    """Inject the single measured plane into unscoped relationship identities."""

    strategy_key = "single_slice_value"

    def matches(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> bool:
        return (
            context.child_slice_indices == frozenset({None})
            and context.single_value_slice_index is not None
        )

    def align(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> _ObjectInstanceChildrenByParent:
        slice_index = context.single_value_slice_index
        if slice_index is None:
            raise ValueError("Single-slice alignment requires a value slice index.")
        return {
            ObjectInstanceKey(parent_id.object_id, slice_index=slice_index): tuple(
                ObjectInstanceKey(child_id.object_id, slice_index=slice_index)
                for child_id in child_ids
            )
            for parent_id, child_ids in context.child_ids_by_parent.items()
        }


class MultiSliceValueObjectInstanceKeyPlaneAlignmentStrategy(
    UnmodifiedObjectInstanceKeyPlaneAlignmentStrategy
):
    """Expand unscoped relationship identities across measured child planes."""

    strategy_key = "multi_slice_value"

    def matches(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> bool:
        return (
            context.child_slice_indices == frozenset({None})
            and None not in context.value_slice_indices
            and len(context.value_slice_indices) > 1
        )

    def align(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> _ObjectInstanceChildrenByParent:
        value_keys_by_object_id: dict[int, list[ObjectInstanceKey]] = {}
        for value_key in context.values_by_child_id:
            value_keys_by_object_id.setdefault(value_key.object_id, []).append(value_key)
        return {
            parent_id: tuple(
                value_key
                for child_id in child_ids
                for value_key in value_keys_by_object_id.get(child_id.object_id, ())
            )
            for parent_id, child_ids in context.child_ids_by_parent.items()
        }


class UnscopedValueObjectInstanceKeyPlaneAlignmentStrategy(
    UnmodifiedObjectInstanceKeyPlaneAlignmentStrategy
):
    """Drop relationship plane identity when measured values are unscoped."""

    strategy_key = "unscoped_value"

    def matches(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> bool:
        return (
            context.value_slice_indices == frozenset({None})
            and None not in context.child_slice_indices
        )

    def align(
        self,
        context: ObjectInstanceKeyPlaneAlignmentContext,
    ) -> _ObjectInstanceChildrenByParent:
        unscoped_children_by_parent: dict[ObjectInstanceKey, list[ObjectInstanceKey]] = {}
        for parent_id, child_ids in context.child_ids_by_parent.items():
            parent_key = ObjectInstanceKey(parent_id.object_id)
            unscoped_children_by_parent.setdefault(parent_key, []).extend(
                ObjectInstanceKey(child_id.object_id) for child_id in child_ids
            )
        return {
            parent_id: tuple(child_ids)
            for parent_id, child_ids in unscoped_children_by_parent.items()
        }


def _object_label_measurement_values_for_name(
    records: Iterable[object],
    object_name: str,
    policy: RuntimeEquivalencePolicy,
    *,
    required_keys: _RuntimeRequiredMeasurementKeys = None,
) -> _RuntimeObjectValuesByLabel:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)
    if subject.name is None:
        return {}
    required_feature_names = RequiredRuntimeMeasurementProjection(
        required_keys,
        policy,
    ).object_location_feature_names(
        subject,
        statistic=MeasurementStatistic.VALUE,
    )
    if required_feature_names is not None and not required_feature_names:
        return {}

    values_by_feature: _RuntimeObjectValuesByLabel = {}
    for record in records:
        object_labels = ObjectLabelSet.from_runtime_value(record.value)
        if _normalize_identifier(object_labels.name) != subject.name:
            continue
        for key, values in _object_label_location_values_by_label(
            object_labels,
            subject,
            policy,
            required_feature_names=required_feature_names,
        ).items():
            values_by_feature.setdefault(key, {}).update(values)
    return values_by_feature


def _object_label_location_values_by_label(
    labels: object,
    subject: RuntimeMeasurementSubjectKey,
    policy: RuntimeEquivalencePolicy,
    *,
    required_feature_names: frozenset[str] | None = None,
) -> _RuntimeObjectValuesByLabel:
    if required_feature_names is not None and not required_feature_names:
        return {}
    context = _ObjectLabelMeasurementContext(
        labels=labels,
        object_name=subject.name,
        policy=policy,
        values_by_feature={},
        object_identifier_subjects=frozenset(),
        object_location_subjects=frozenset(),
        object_count_subjects=frozenset(),
        required_keys=None,
        declared_object_count=(
            labels.object_label_domain().declared_object_count
            if isinstance(labels, runtime_semantics.ObjectLabelDomainMetadata)
            else None
        ),
        declared_object_ids=(
            labels.object_label_domain().declared_object_ids
            if isinstance(labels, runtime_semantics.ObjectLabelDomainMetadata)
            else ()
        ),
        declared_object_id_domains=(
            labels.object_label_domain().declared_object_id_domains
            if isinstance(labels, runtime_semantics.ObjectLabelDomainMetadata)
            else ()
        ),
        domain_scope=(
            labels.object_label_domain().scope
            if isinstance(labels, runtime_semantics.ObjectLabelDomainMetadata)
            else ObjectLabelDomainScope.PAYLOAD
        ),
    )
    projections = ObjectLabelMeasurementProjectionStrategy.for_scope(
        context.domain_scope
    ).projections(context)
    if not projections:
        return {}

    values_by_feature: _RuntimeObjectValuesByLabel = {}
    for projection in projections:
        for key, values in _object_label_location_values_by_label_for_plane(
            projection.labels,
            subject,
            required_feature_names=required_feature_names,
            object_ids=projection.object_ids,
            slice_index=projection.slice_index,
        ).items():
            values_by_feature.setdefault(key, {}).update(values)
    return values_by_feature


def _object_label_location_values_by_label_for_plane(
    labels: np.ndarray,
    subject: RuntimeMeasurementSubjectKey,
    *,
    required_feature_names: frozenset[str] | None = None,
    object_ids: tuple[int, ...] | None = None,
    slice_index: int | None = None,
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

    return {
        RuntimeMeasurementFeatureKey(subject, feature_name): {
            ObjectInstanceKey(label, slice_index=slice_index): float(
                coordinate.values[label]
            )
            for label in resolved_object_ids
            if coordinate.include_missing or np.isfinite(coordinate.values[label])
        }
        for feature_name, coordinate in object_location_coordinate_arrays(
            axis_centers,
            counts,
        )
        if required_feature_names is None or feature_name in required_feature_names
    }


RELATIONSHIP_DISTANCE_FEATURE_NAMES = frozenset(
    (
        "distance_centroid",
        "distance_minimum",
    )
)


@dataclass(frozen=True, slots=True)
class RelationshipAggregateFeatureContext:
    """Semantic context for deriving parent-row aggregates from child rows."""

    source_name: str
    target_name: str
    feature_name: str


class RelationshipAggregateFeatureSemantics(
    MostDerivedContextStrategyMixin[RelationshipAggregateFeatureContext],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Map child measurement features onto relationship aggregate features."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True

    strategy_key: ClassVar[str | None] = None

    @abstractmethod
    def matches(self, context: RelationshipAggregateFeatureContext) -> bool:
        """Return whether this semantic strategy owns the feature context."""

    @abstractmethod
    def required_child_feature_names(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> tuple[str, ...]:
        """Return child-row features needed to synthesize the aggregate."""

    @abstractmethod
    def aggregate_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        """Return the source-row aggregate feature emitted for a child feature."""

    def aggregate_child_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> str:
        """Return the child feature semantically represented by ``context``."""
        return _normalize_identifier(context.feature_name)

    @staticmethod
    def parent_qualified_feature_name(feature_name: str, parent_name: str) -> str:
        return "_".join(
            part for part in (feature_name, _normalize_identifier(parent_name)) if part
        )

    @staticmethod
    def parent_unqualified_feature_name(feature_name: str, parent_name: str) -> str:
        parent_suffix = f"_{_normalize_identifier(parent_name)}"
        if feature_name.endswith(parent_suffix):
            return feature_name[: -len(parent_suffix)]
        return feature_name

    @staticmethod
    def target_aggregate_feature_name(
        target_name: str,
        child_feature_name: str,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        parts = (
            _normalize_identifier(aggregate),
            _normalize_identifier(target_name),
            _normalize_identifier(child_feature_name),
        )
        return "_".join(part for part in parts if part)

    @classmethod
    def aggregate_child_feature_name_from_key(
        cls,
        feature: RuntimeMeasurementFeatureKey,
        dialect: RuntimeMeasurementDialect,
    ) -> str | None:
        """Return child feature represented by a relationship aggregate key."""
        parts = tuple(part for part in feature.feature_name.split("_") if part)
        aggregate_identity = RuntimeAggregateFeatureIdentity.from_parts(
            parts,
            dialect,
        )
        if aggregate_identity is None:
            return None
        context = RelationshipAggregateFeatureContext(
            source_name=feature.subject.name or "",
            target_name=aggregate_identity.object_name,
            feature_name=aggregate_identity.feature_name,
        )
        semantics = cls.for_context(
            context,
            required=False,
        )
        if semantics is None:
            return None
        return semantics.aggregate_child_feature_name(context)


class GenericRelationshipAggregateFeatureSemantics(
    RelationshipAggregateFeatureSemantics
):
    """Default relationship aggregates preserve the child feature identity."""

    strategy_key = "generic"

    def matches(self, context: RelationshipAggregateFeatureContext) -> bool:
        del context
        return True

    def required_child_feature_names(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> tuple[str, ...]:
        return (_normalize_identifier(context.feature_name),)

    def aggregate_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        return self.target_aggregate_feature_name(
            context.target_name,
            context.feature_name,
            aggregate=aggregate,
        )

    def aggregate_child_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> str:
        return _normalize_identifier(context.feature_name)


class ParentQualifiedDistanceAggregateFeatureSemantics(
    GenericRelationshipAggregateFeatureSemantics
):
    """CellProfiler relationship distance rows qualify child features by parent."""

    strategy_key = "parent_qualified_distance"

    def matches(self, context: RelationshipAggregateFeatureContext) -> bool:
        feature_name = _normalize_identifier(context.feature_name)
        return (
            feature_name in RELATIONSHIP_DISTANCE_FEATURE_NAMES
            or self.parent_unqualified_feature_name(
                feature_name,
                context.source_name,
            )
            in RELATIONSHIP_DISTANCE_FEATURE_NAMES
        )

    def required_child_feature_names(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> tuple[str, ...]:
        feature_name = _normalize_identifier(context.feature_name)
        return (
            feature_name,
            self.parent_qualified_feature_name(feature_name, context.source_name),
        )

    def aggregate_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        return self.target_aggregate_feature_name(
            context.target_name,
            self.parent_unqualified_feature_name(
                _normalize_identifier(context.feature_name),
                context.source_name,
            ),
            aggregate=aggregate,
        )

    def aggregate_child_feature_name(
        self,
        context: RelationshipAggregateFeatureContext,
    ) -> str:
        return self.parent_unqualified_feature_name(
            _normalize_identifier(context.feature_name),
            context.source_name,
        )


@dataclass(frozen=True, slots=True)
class RelationshipMeasurementSemantics:
    """Measurement identity contract for a directed object relationship."""

    relationship: ObjectRelationship

    @property
    def source_name(self) -> str:
        return _normalize_identifier(self.relationship.source.name)

    @property
    def target_name(self) -> str:
        return _normalize_identifier(self.relationship.target.name)

    @property
    def source_subject(self) -> RuntimeMeasurementSubjectKey:
        return RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            self.source_name,
        )

    @property
    def target_subject(self) -> RuntimeMeasurementSubjectKey:
        return RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            self.target_name,
        )

    @property
    def instance_relationship(self) -> ObjectInstanceRelationship:
        return ObjectInstanceRelationship.from_id_columns(
            tuple(int(value) for value in np.asarray(self.relationship.source_ids).ravel()),
            tuple(int(value) for value in np.asarray(self.relationship.target_ids).ravel()),
            slice_indices=self.relationship.slice_indices,
            slice_count=self.relationship.slice_count,
        )

    @property
    def aggregate_prefix(self) -> str:
        return f"{MeasurementStatistic.MEAN.value}_{self.target_name}_"

    @property
    def child_count_key(self) -> RuntimeMeasurementFeatureKey:
        return RuntimeMeasurementFeatureKey(
            self.source_subject,
            f"{self.target_name}_count",
        )

    @property
    def parent_key(self) -> RuntimeMeasurementFeatureKey:
        return RuntimeMeasurementFeatureKey(
            self.target_subject,
            self.source_name,
        )

    @property
    def target_object_number_key(self) -> RuntimeMeasurementFeatureKey:
        return RuntimeMeasurementFeatureKeyFactory(
            self.target_subject,
            ObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
        ).key()

    def aggregate_feature_name(
        self,
        child_feature_name: str,
        *,
        aggregate: str = MeasurementStatistic.MEAN.value,
    ) -> str:
        context = RelationshipAggregateFeatureContext(
            source_name=self.source_name,
            target_name=self.target_name,
            feature_name=child_feature_name,
        )
        return RelationshipAggregateFeatureSemantics.for_context(
            context,
            error_subject="relationship aggregate feature",
        ).aggregate_feature_name(context, aggregate=aggregate)

    def required_child_measurement_keys(
        self,
        required_measurement_keys: _RuntimeRequiredMeasurementKeys,
    ) -> _RuntimeRequiredMeasurementKeys:
        """Return child measurements needed to synthesize required aggregates."""
        if required_measurement_keys is None:
            return None
        child_keys: set[RuntimeMeasurementFeatureKey] = set()
        for key in required_measurement_keys:
            if (
                key.subject != self.source_subject
                or key.statistic != MeasurementStatistic.VALUE.value
                or not key.feature_name.startswith(self.aggregate_prefix)
                or key.feature_name == self.aggregate_prefix
            ):
                continue
            aggregate_child_feature_name = key.feature_name.removeprefix(
                self.aggregate_prefix
            )
            context = RelationshipAggregateFeatureContext(
                source_name=self.source_name,
                target_name=self.target_name,
                feature_name=aggregate_child_feature_name,
            )
            semantics = RelationshipAggregateFeatureSemantics.for_context(
                context,
                error_subject="relationship aggregate child feature",
            )
            child_keys.update(
                RuntimeMeasurementFeatureKeyFactory(
                    self.target_subject,
                    child_feature_name,
                    source_name=key.source_name,
                ).key()
                for child_feature_name in semantics.required_child_feature_names(
                    context
                )
            )
        return frozenset(child_keys)

    def measurement_facts(
        self,
        policy: RuntimeEquivalencePolicy,
        *,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog | None = None,
    ) -> _RuntimeMeasurementFacts:
        """Return direct relationship measurements under canonical object identity."""
        catalog = self.object_label_catalog(object_label_catalog)
        child_keys_by_parent = self.child_keys_by_parent(catalog)
        parent_key_by_child = self.parent_key_by_child()
        return (
            *(
                (
                    self.child_count_key,
                    _cell_signature(
                        str(len(child_keys_by_parent.get(source_key, ()))),
                        policy,
                    ),
                )
                for source_key in self.source_domain(catalog)
            ),
            *(
                (
                    self.parent_key,
                    _cell_signature(
                        str(
                            parent_key_by_child[target_key].object_id
                            if target_key in parent_key_by_child
                            else 0
                        ),
                        policy,
                    ),
                )
                for target_key in self.target_domain(catalog)
            ),
        )

    def aggregate_measurement_facts(
        self,
        child_values_by_feature: Mapping[
            RuntimeMeasurementFeatureKey,
            Mapping[ObjectInstanceKey, float],
        ],
        policy: RuntimeEquivalencePolicy,
        *,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog | None = None,
        existing_measurement_keys: _RuntimeMeasurementKeySet = frozenset(),
        required_measurement_keys: _RuntimeRequiredMeasurementKeys = None,
    ) -> _RuntimeMeasurementFacts:
        """Return source-row aggregate measurements derived from target rows."""
        catalog = self.object_label_catalog(object_label_catalog)
        child_values = {
            key: dict(values_by_child_id)
            for key, values_by_child_id in child_values_by_feature.items()
        }
        child_values[self.target_object_number_key] = self.target_object_number_values(
            catalog
        )
        if not child_values:
            return ()

        child_ids_by_parent = self.child_keys_by_parent(catalog)
        aggregate_facts: _RuntimeMeasurementFactList = []
        for child_key, values_by_child_id in child_values.items():
            if child_key.subject.scope is not MeasurementScope.OBJECT:
                continue
            if child_key.subject != self.target_subject:
                continue
            aggregate_key = RuntimeMeasurementFeatureKeyFactory(
                self.source_subject,
                self.aggregate_feature_name(child_key.feature_name),
                source_name=child_key.source_name,
            ).key()
            if (
                aggregate_key in existing_measurement_keys
                and child_key.feature_name
                != ObjectCoreMeasurementFeature.OBJECT_NUMBER.value
            ):
                continue
            if (
                required_measurement_keys is not None
                and aggregate_key not in required_measurement_keys
            ):
                continue
            aligned_child_ids_by_parent = (
                ObjectInstanceKeyPlaneAlignmentStrategy.align_child_ids_by_parent(
                    child_ids_by_parent,
                    values_by_child_id,
                )
            )
            aggregate_values_by_parent = self.aggregate_values_by_parent(
                aligned_child_ids_by_parent,
                values_by_child_id,
            )
            for _parent_id, child_ids in aligned_child_ids_by_parent.items():
                aggregate_value = aggregate_values_by_parent[child_ids]
                if not math.isfinite(aggregate_value):
                    continue
                aggregate_facts.append(
                    (
                        aggregate_key,
                        _cell_signature(
                            str(aggregate_value),
                            policy,
                        ),
                    )
                )
        return tuple(aggregate_facts)

    def source_domain(
        self,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog,
    ) -> tuple[ObjectInstanceKey, ...]:
        """Return source-object identities represented by this relationship."""
        return self.instance_relationship.source_domain(
            object_label_catalog.count_for_subject(self.source_subject),
            declared_keys=object_label_catalog.domain_for_subject(self.source_subject),
        )

    def target_domain(
        self,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog,
    ) -> tuple[ObjectInstanceKey, ...]:
        """Return target-object identities represented by this relationship."""
        return self.instance_relationship.target_domain(
            object_label_catalog.count_for_subject(self.target_subject),
            declared_keys=object_label_catalog.domain_for_subject(self.target_subject),
        )

    def child_keys_by_parent(
        self,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog,
    ) -> dict[ObjectInstanceKey, tuple[ObjectInstanceKey, ...]]:
        """Return target identities grouped by source identity."""
        return self.instance_relationship.child_keys_by_parent(
            source_object_count=object_label_catalog.count_for_subject(
                self.source_subject
            ),
            declared_source_keys=object_label_catalog.domain_for_subject(
                self.source_subject
            ),
        )

    def parent_key_by_child(self) -> dict[ObjectInstanceKey, ObjectInstanceKey]:
        """Return source identity for each target identity."""
        return self.instance_relationship.parent_key_by_child()

    def target_object_number_values(
        self,
        object_label_catalog: RuntimeObjectLabelInstanceCatalog,
    ) -> dict[ObjectInstanceKey, float]:
        """Return target-object-number values keyed by target identity."""
        return {
            target_key: float(target_key.object_id)
            for target_key in self.target_domain(object_label_catalog)
        }

    def aggregate_values_by_parent(
        self,
        child_ids_by_parent: _ObjectInstanceChildrenByParent,
        values_by_child_id: Mapping[ObjectInstanceKey, float],
    ) -> _ObjectInstanceAggregateValues:
        """Return aggregate target values keyed by each parent child-domain."""
        means: _ObjectInstanceAggregateValues = {}
        for child_ids in child_ids_by_parent.values():
            if child_ids in means:
                continue
            means[child_ids] = self.mean_child_value(child_ids, values_by_child_id)
        return means

    def mean_child_value(
        self,
        child_ids: tuple[ObjectInstanceKey, ...],
        values_by_child_id: Mapping[ObjectInstanceKey, float],
    ) -> float:
        """Return the mean over child identities with available values."""
        del self
        values = tuple(
            values_by_child_id[child_id]
            for child_id in child_ids
            if child_id in values_by_child_id
        )
        if not values:
            return float("nan")
        return float(sum(values) / len(values))

    @staticmethod
    def object_label_catalog(
        object_label_catalog: RuntimeObjectLabelInstanceCatalog | None,
    ) -> RuntimeObjectLabelInstanceCatalog:
        """Return a typed object-label catalog for relationship projection."""
        if object_label_catalog is None:
            return RuntimeObjectLabelInstanceCatalog.empty()
        return object_label_catalog


def _object_measurement_values_by_label(
    measurement_tables: tuple[RuntimeScopedMeasurementTable, ...],
    object_name: str,
    policy: RuntimeEquivalencePolicy,
    *,
    known_source_names: tuple[str, ...],
    required_keys: _RuntimeRequiredMeasurementKeys = None,
) -> _RuntimeObjectValuesByLabel:
    object_subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)
    values_by_feature: _RuntimeObjectValuesByLabel = {}
    row_required_keys = RequiredRuntimeMeasurementProjection(
        required_keys,
        policy,
        known_source_names=known_source_names,
    ).input_keys()
    schema_cache: _RuntimeMeasurementRowSchemaCache = {}
    key_cache: _RuntimeMeasurementFeatureKeyCache = {}
    long_form_key_cache: _RuntimeMeasurementLongFormKeyCache = {}
    qualifier_render_cache: _RuntimeMeasurementQualifierRenderCache = {}
    padding_group_cache: _RuntimeMeasurementPaddingGroupCache = {}
    for scoped_table in measurement_tables:
        table = scoped_table.table
        table_subject = RuntimeMeasurementSubjectKey.from_table_subject(table.subject)
        table_object_subject = (
            table_subject if table_subject.scope is MeasurementScope.OBJECT else None
        )
        if table_object_subject is not None and table_object_subject != object_subject:
            continue
        object_id_field = measurement_table_object_id_field(table)
        table_source_name = table.source_image_name
        table_padding_group = _normalize_identifier(table.name) or "measurements"
        image_number_offset = RuntimeImageNumberOffset.from_runtime_rows(
            iter_measurement_rows((table,))
        )
        for row in iter_measurement_rows((table,)):
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
            row_context = RuntimeRowProjectionContext.from_row(
                row_mapping,
                subject,
                policy,
                source_name=measurement_row_source_image_name(row_mapping)
                or table_source_name,
                known_source_names=known_source_names,
                required_keys=row_required_keys,
                table_padding_group=table_padding_group,
                image_number_offset=image_number_offset,
                schema_cache=schema_cache,
                key_cache=key_cache,
                long_form_key_cache=long_form_key_cache,
                qualifier_render_cache=qualifier_render_cache,
                padding_group_cache=padding_group_cache,
            )
            for key, value in RuntimeMeasurementRowFactProjector(
                row_context
            ).numeric_values():
                if row_required_keys is not None and key not in row_required_keys:
                    continue
                if key.statistic != "value":
                    continue
                values_by_feature.setdefault(key, {})[
                    scoped_table.object_instance_key(
                        row_mapping,
                        object_label,
                        image_number_offset=image_number_offset,
                    )
                ] = value
    return values_by_feature


def _numeric_long_form_measurement_values(
    fact: _RuntimeMeasurementFact,
) -> _RuntimeNumericMeasurementValues:
    key, cell_value = fact
    numeric_value = _cell_signature_numeric_value(cell_value)
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
            for row in iter_measurement_rows((table,)):
                row_source_name = measurement_row_source_image_name(
                    measurement_row_mapping(row)
                )
                if row_source_name is not None:
                    source_names.update(_source_name_aliases(row_source_name))
    return tuple(sorted(source_names, key=_normalize_identifier))


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
        RuntimeMeasurementFieldIndexMap(normalized_field_indexes).indexes_for(
            MEASUREMENT_OBJECT_ID_FIELDS
        ),
        RuntimeMeasurementFieldIndexMap(normalized_field_indexes).indexes_for(
            tuple(sorted(_IMAGE_IDENTITY_FIELDS))
        ),
    )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowSubjectResolutionContext:
    """Typed subject-resolution facts for one measurement row."""

    table_subject: RuntimeMeasurementSubjectKey
    object_name: str | None
    row_source_name: str | None
    has_object_identity: bool
    has_image_identity: bool


class RuntimeMeasurementRowSubjectResolutionStrategy(
    MostDerivedContextStrategyMixin[RuntimeMeasurementRowSubjectResolutionContext],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal row-subject resolver for runtime measurement tables."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True

    strategy_key: ClassVar[str | None] = None

    @classmethod
    def resolve(
        cls,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        strategy = cls.for_context(
            context,
            error_subject="Runtime measurement row subject resolution",
        )
        if strategy is None:
            raise ValueError(
                "Runtime measurement row subject resolution requires a strategy."
            )
        return strategy.subject(context)

    @abstractmethod
    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        """Return the semantic measurement subject for this row."""


class FallbackTableSubjectResolutionStrategy(
    RuntimeMeasurementRowSubjectResolutionStrategy
):
    """Fallback to the table's declared subject."""

    strategy_key = "fallback_table_subject"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return True

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return context.table_subject


class SourceImageRowSubjectResolutionStrategy(FallbackTableSubjectResolutionStrategy):
    """Rows declaring only source-image identity remain image-scoped."""

    strategy_key = "source_image_row"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return context.row_source_name is not None

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")


class ImageTableSubjectResolutionStrategy(SourceImageRowSubjectResolutionStrategy):
    """Image tables own rows that do not carry a stronger row identity."""

    strategy_key = "image_table_subject"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return context.table_subject.scope is MeasurementScope.IMAGE

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return context.table_subject


class ObjectTableSubjectResolutionStrategy(ImageTableSubjectResolutionStrategy):
    """Object tables own rows that do not carry a stronger row identity."""

    strategy_key = "object_table_subject"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return context.table_subject.scope is MeasurementScope.OBJECT

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return context.table_subject


class ImageIdentityRowSubjectResolutionStrategy(ObjectTableSubjectResolutionStrategy):
    """Rows with image identity and no object identity are image-scoped."""

    strategy_key = "image_identity"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return context.has_image_identity and not context.has_object_identity

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")


class ObjectIdentityRowSubjectResolutionStrategy(ImageIdentityRowSubjectResolutionStrategy):
    """Rows with object identity are object-scoped."""

    strategy_key = "object_identity"

    def matches(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> bool:
        return context.object_name is not None and context.has_object_identity

    def subject(
        self,
        context: RuntimeMeasurementRowSubjectResolutionContext,
    ) -> RuntimeMeasurementSubjectKey:
        return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, context.object_name)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementRowSubjectProjection:
    """Resolve runtime-row source and measurement subject from row values."""

    table_subject: RuntimeMeasurementSubjectKey
    table_source_name: str | None
    row_values: tuple[object, ...]
    subject_schema: _RuntimeMeasurementRowSubjectSchema

    @property
    def indexed_values(self) -> RuntimeIndexedRowValues:
        return RuntimeIndexedRowValues(self.row_values)

    def source_name(self) -> str | None:
        row_source_name = self.indexed_values.text_at(self.subject_schema[1])
        if row_source_name is not None:
            return row_source_name
        return self.table_source_name

    def subject(self) -> RuntimeMeasurementSubjectKey:
        (
            object_name_index,
            source_name_index,
            object_identity_indexes,
            image_identity_indexes,
        ) = self.subject_schema
        return RuntimeMeasurementRowSubjectResolutionStrategy.resolve(
            RuntimeMeasurementRowSubjectResolutionContext(
                table_subject=self.table_subject,
                object_name=self.indexed_values.text_at(object_name_index),
                row_source_name=self.indexed_values.text_at(source_name_index),
                has_object_identity=self.indexed_values.has_text_at_any(
                    object_identity_indexes
                ),
                has_image_identity=self.indexed_values.has_text_at_any(
                    image_identity_indexes
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class RuntimeExportRowSubject:
    """Resolve table-export row subject from table path and row identity."""

    path: Path
    row: Mapping[str, object]

    def subject(self) -> RuntimeMeasurementSubjectKey:
        object_name = measurement_row_object_name(self.row)
        if object_name is not None:
            return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)

        table_name = self.path.stem
        normalized_table_name = _normalize_identifier(table_name)
        if RuntimeMeasurementRowMapping(self.row).has_object_identity() and normalized_table_name != "image":
            return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, table_name)
        if normalized_table_name == "experiment":
            return RuntimeMeasurementSubjectKey(MeasurementScope.EXPERIMENT, None)
        return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")


@dataclass(frozen=True, slots=True)
class RuntimeExportColumnSubject:
    """Resolve table-export column subject from contextual column metadata."""

    table: RuntimeTableSnapshot
    row: Mapping[str, object]
    index: int
    fallback_subject: RuntimeMeasurementSubjectKey

    def subject(self) -> RuntimeMeasurementSubjectKey:
        if not self.table.column_context or self.index >= len(self.table.column_context):
            return self.fallback_subject

        field_name = self.table.header[self.index]
        if RuntimeMeasurementIdentityField(
            DEFAULT_RUNTIME_MEASUREMENT_DIALECT
        ).field_matches(field_name):
            return self.fallback_subject

        context, normalized_context = self.context_pair()
        row_object_name = measurement_row_object_name(self.row)
        normalized_row_object_name = (
            _normalize_identifier(row_object_name)
            if row_object_name is not None
            else None
        )
        return RuntimeColumnContextSubject(
            context,
            normalized_context,
            row_object_name,
            normalized_row_object_name,
            fallback_subject=self.fallback_subject,
        ).subject()

    def context_pair(self) -> tuple[str | None, str | None]:
        if not self.table.column_context or self.index >= len(self.table.column_context):
            return None, None
        context = self.table.column_context[self.index]
        if context is None:
            return None, None
        normalized_context = _normalize_identifier(context)
        if not normalized_context:
            return None, None
        return context, normalized_context


@dataclass(frozen=True, slots=True)
class RuntimeColumnContextSubject:
    """Resolve a subject implied by contextual wide-table column metadata."""

    context: str | None
    normalized_context: str | None
    row_object_name: str | None
    normalized_row_object_name: str | None
    fallback_subject: RuntimeMeasurementSubjectKey

    def subject(self) -> RuntimeMeasurementSubjectKey:
        if self.context is None or self.normalized_context is None:
            return self.fallback_subject
        if self.normalized_context in _CSV_HEADER_CONTEXT_STOPWORDS:
            if self.normalized_context == "image":
                return RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")
            return self.fallback_subject

        if (
            self.row_object_name is not None
            and self.normalized_row_object_name == self.normalized_context
        ):
            return RuntimeMeasurementSubjectKey(
                MeasurementScope.OBJECT,
                self.row_object_name,
            )
        return RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, self.context)


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
    core_feature_name, field_source_name = SemanticCoreFeatureAndSourceNameProjection(
        normalized,
        policy.measurement_dialect,
        known_source_names,
    ).project()
    return _directional_pair_feature_name_and_source(
        core_feature_name,
        field_source_name or normalized_source_name,
        policy.measurement_dialect,
    )


def _semantic_core_feature_name(feature_name: str) -> str:
    return SemanticCoreFeatureAndSourceNameProjection(
        feature_name,
        DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    ).project()[0]


def _measurement_qualifier_parts_only(parts: tuple[str, ...]) -> bool:
    return bool(parts) and all(part.isdigit() for part in parts)


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
    return tuple(sorted(groups, key=lambda group: (-len(group[1]), group[0])))


def _matching_source_name_at(
    parts: tuple[str, ...],
    index: int,
    source_token_groups: _RuntimeSourceTokenGroups,
) -> str | None:
    for source_name, source_parts in source_token_groups:
        if parts[index : index + len(source_parts)] == source_parts:
            return source_name
    return None


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
_PAIR_MANDERS_FEATURE = PairMeasurementFeature.MANDERS.value
_PAIR_RANK_WEIGHTED_COLOCALIZATION_FEATURE = (
    PairMeasurementFeature.RANK_WEIGHTED_COLOCALIZATION.value
)
_PAIR_OVERLAP_K_FEATURE = PairMeasurementFeature.OVERLAP_K.value
_UNDIRECTED_PAIR_FEATURES = frozenset(
    (_PAIR_CORRELATION_FEATURE, _PAIR_OVERLAP_FEATURE)
)
_THRESHOLD_SENSITIVE_PAIR_FEATURES = frozenset(
    (
        _PAIR_COSTES_MANDERS_FEATURE,
        _PAIR_MANDERS_FEATURE,
        _PAIR_RANK_WEIGHTED_COLOCALIZATION_FEATURE,
        _PAIR_OVERLAP_K_FEATURE,
    )
)
