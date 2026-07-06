"""Object-label derived runtime measurement projection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
import numpy as np

from openhcs.core.equivalence.cells import runtime_cell_signature
from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.measurement_facts import (
    RuntimeExpectedMeasurementFactCompletion,
    RuntimeMeasurementFactCounterMap,
    RuntimeMeasurementFactCounterMapping,
    RuntimeMeasurementFactList,
    RuntimeMeasurementFacts,
    RuntimeRequiredMeasurementKeys,
)
from openhcs.core.equivalence.measurement_requirements import (
    RequiredRuntimeMeasurementProjection,
)
from openhcs.core.equivalence.measurement_features import (
    object_measurement_subjects_matching_marker,
)
from openhcs.core.equivalence.measurement_rows import (
    RuntimeObjectMeasurementRowIdentity,
)
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    normalize_runtime_identifier,
)
from openhcs.core.runtime_semantics import (
    MeasurementScope,
    MeasurementStatistic,
    ObjectCoreMeasurementFeature,
    ObjectCountFeatureMarker,
    ObjectIdentifierFeatureMarker,
    ObjectLocationFeatureMarker,
    ObjectInstanceKey,
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    ObjectLabelIdDomainStrategy,
    ObjectLabelInstanceDomains,
    ObjectLocationCoordinateValues,
    RuntimePlaneAxis,
    dense_object_label_id_domain,
    dense_object_label_identity_domains,
    dense_object_label_plane_id_domains,
    object_location_coordinate_arrays,
)
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.core.runtime_values import (
    DenseObjectLabelPlaneDomainStackRequest,
    ObjectLabelData,
    ObjectLabelSet,
    ObjectLabelValue,
    RuntimeValue,
    object_label_dense_array,
)


RuntimeObjectValuesByLabel = dict[
    RuntimeMeasurementFeatureKey,
    dict[ObjectInstanceKey, float],
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectLabelMeasurementState:
    """Shared object-label measurement completion state."""

    policy: RuntimeEquivalencePolicy
    object_identifier_subjects: frozenset[RuntimeMeasurementSubjectKey]
    object_location_subjects: frozenset[RuntimeMeasurementSubjectKey]
    object_count_subjects: frozenset[RuntimeMeasurementSubjectKey]
    required_keys: RuntimeRequiredMeasurementKeys
    object_location_aggregate_subjects: frozenset[
        RuntimeMeasurementSubjectKey
    ] = frozenset()


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectLabelMeasurementContext(ObjectLabelMeasurementState):
    object_labels: ObjectLabelValue
    object_name: str | None
    domain: ObjectLabelDomain

    @classmethod
    def from_runtime_value(
        cls,
        value: RuntimeValue,
        policy: RuntimeEquivalencePolicy,
        object_identifier_subjects: frozenset[RuntimeMeasurementSubjectKey],
        object_location_subjects: frozenset[RuntimeMeasurementSubjectKey],
        object_count_subjects: frozenset[RuntimeMeasurementSubjectKey],
        required_keys: RuntimeRequiredMeasurementKeys,
        object_location_aggregate_subjects: frozenset[
            RuntimeMeasurementSubjectKey
        ] = frozenset(),
    ) -> "ObjectLabelMeasurementContext":
        label_set = ObjectLabelSet.from_runtime_value(value)
        return cls.from_object_labels(
            label_set,
            value.schema.object_name or label_set.name,
            policy,
            object_identifier_subjects,
            object_location_subjects,
            object_count_subjects,
            required_keys,
            object_location_aggregate_subjects=object_location_aggregate_subjects,
        )

    @classmethod
    def from_object_labels(
        cls,
        object_labels: ObjectLabelValue,
        object_name: str | None,
        policy: RuntimeEquivalencePolicy,
        object_identifier_subjects: frozenset[RuntimeMeasurementSubjectKey],
        object_location_subjects: frozenset[RuntimeMeasurementSubjectKey],
        object_count_subjects: frozenset[RuntimeMeasurementSubjectKey],
        required_keys: RuntimeRequiredMeasurementKeys,
        object_location_aggregate_subjects: frozenset[
            RuntimeMeasurementSubjectKey
        ] = frozenset(),
    ) -> "ObjectLabelMeasurementContext":
        return cls(
            object_labels=object_labels,
            object_name=object_name,
            policy=policy,
            object_identifier_subjects=object_identifier_subjects,
            object_location_subjects=object_location_subjects,
            object_count_subjects=object_count_subjects,
            required_keys=required_keys,
            domain=object_labels.object_label_domain(),
            object_location_aggregate_subjects=object_location_aggregate_subjects,
        )

    @property
    def labels(self) -> ObjectLabelData:
        return self.object_labels.labels

    def label_array(self) -> np.ndarray | None:
        label_array = np.asarray(self.labels)
        if label_array.ndim == 0:
            return None
        return label_array

    def dense_identity_domains(self) -> tuple[tuple[int, ...], ...]:
        """Return dense object identity domains for this object-label payload."""
        return dense_object_label_identity_domains(
            self.labels,
            declared_object_count=self.domain.declared_object_count,
            declared_object_ids=self.domain.declared_object_ids,
            declared_object_id_domains=self.domain.declared_object_id_domains,
            domain_scope=self.domain.scope,
        )

    def dense_plane_id_domains(self) -> tuple[tuple[int, ...], ...]:
        """Return dense object ID domains aligned to measurement planes."""
        return dense_object_label_plane_id_domains(
            self.labels,
            declared_object_count=self.domain.declared_object_count,
            declared_object_ids=self.domain.declared_object_ids,
            declared_object_id_domains=self.domain.declared_object_id_domains,
            domain_scope=self.domain.scope,
        )

    def has_declared_object_domain(self) -> bool:
        """Return whether this context carries explicit object-domain metadata."""
        return (
            self.domain.declared_object_count is not None
            or bool(self.domain.declared_object_ids)
            or bool(self.domain.declared_object_id_domains)
        )

    def owns_payload_spatial_location(self) -> bool:
        """Return whether this payload is authoritative for dimensional locations."""
        if self.domain.scope is not ObjectLabelDomainScope.PAYLOAD:
            return False
        projections = ObjectLabelMeasurementProjectionStrategy.for_scope(
            self.domain.scope
        ).projections(self)
        return any(projection.labels.ndim >= 3 for projection in projections)

    def measurement_facts(self) -> RuntimeMeasurementFacts:
        """Project this object-label payload into implicit object measurement facts."""
        if not object_label_measurements_required(
            self.object_name,
            self.required_keys,
        ):
            return ()
        return (
            *self.count_facts(),
            *self.location_facts(),
        )

    def count_facts(self) -> RuntimeMeasurementFacts:
        """Return Object_Count facts implied by this object-label payload."""
        if self.object_name is None:
            return ()
        subject = RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            self.object_name,
        )
        if subject in self.object_count_subjects:
            return ()
        key = RuntimeMeasurementFeatureKey(
            subject,
            ObjectCoreMeasurementFeature.OBJECT_COUNT.value,
            MeasurementStatistic.COUNT.value,
        )
        if self.required_keys is not None and key not in self.required_keys:
            return ()
        label_array = self.label_array()
        if label_array is None:
            return ()
        projections = ObjectLabelMeasurementProjectionStrategy.for_scope(
            self.domain.scope
        ).projections(self)
        return tuple(
            (
                key,
                runtime_cell_signature(
                    str(len(projection.object_ids)),
                    self.policy,
                ),
            )
            for projection in projections
        )

    def location_facts(self) -> RuntimeMeasurementFacts:
        """Return Location_* facts implied by this object-label payload."""
        if self.object_name is None:
            return ()
        subject = RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            self.object_name,
        )
        required_projection = RequiredRuntimeMeasurementProjection(
            self.required_keys,
            self.policy,
        )
        required_feature_names = required_projection.object_location_feature_names(
            subject,
            statistic=MeasurementStatistic.VALUE,
        )
        required_mean_feature_names = required_projection.object_location_feature_names(
            subject,
            statistic=MeasurementStatistic.MEAN,
        )
        if subject in self.object_location_subjects:
            required_feature_names = frozenset()
        if subject in self.object_location_aggregate_subjects:
            required_mean_feature_names = frozenset()
        if (
            required_feature_names is not None
            and not required_feature_names
            and required_mean_feature_names is not None
            and not required_mean_feature_names
        ):
            return ()
        label_array = self.label_array()
        if label_array is None:
            return ()
        projections = ObjectLabelMeasurementProjectionStrategy.for_scope(
            self.domain.scope
        ).projections(self)
        if not any(projection.object_ids for projection in projections):
            return ()

        facts: RuntimeMeasurementFactList = []
        include_missing_locations = (
            label_array.ndim <= 2 and self.domain.declared_object_count is not None
        )
        for projection in projections:
            facts.extend(
                object_location_measurement_facts_for_plane(
                    projection.labels,
                    subject,
                    self.policy,
                    required_feature_names=required_feature_names,
                    required_mean_feature_names=required_mean_feature_names,
                    object_ids=projection.object_ids,
                    include_missing=include_missing_locations,
                )
            )
        return tuple(facts)


@dataclass(frozen=True, slots=True)
class ObjectLabelMeasurementProjection:
    """One semantic object-label measurement domain."""

    labels: np.ndarray
    object_ids: tuple[int, ...]
    slice_index: int | None = None


@dataclass(frozen=True, slots=True)
class RuntimeObjectLabelMeasurementAuthority:
    """Nominal object-label ownership before primary-row fallback is allowed."""

    count_subjects: frozenset[RuntimeMeasurementSubjectKey] = frozenset()
    identifier_subjects: frozenset[RuntimeMeasurementSubjectKey] = frozenset()
    location_subjects: frozenset[RuntimeMeasurementSubjectKey] = frozenset()

    @classmethod
    def from_object_label_records(
        cls,
        records: Iterable[StoredRuntimeValue],
        policy: RuntimeEquivalencePolicy,
    ) -> "RuntimeObjectLabelMeasurementAuthority":
        count_subjects: set[RuntimeMeasurementSubjectKey] = set()
        identifier_subjects: set[RuntimeMeasurementSubjectKey] = set()
        location_subjects: set[RuntimeMeasurementSubjectKey] = set()
        for record in records:
            object_labels = ObjectLabelSet.from_runtime_value(record.value)
            subject = RuntimeMeasurementSubjectKey(
                MeasurementScope.OBJECT,
                object_labels.name,
            )
            domain = object_labels.object_label_domain()
            if (
                domain.declared_object_count is not None
                or domain.declared_object_ids
                or domain.declared_object_id_domains
            ):
                count_subjects.add(subject)
            context = ObjectLabelMeasurementContext.from_object_labels(
                object_labels,
                object_labels.name,
                policy,
                frozenset(),
                frozenset(),
                frozenset(),
                None,
            )
            if ObjectLabelMeasurementProjectionStrategy.for_scope(
                context.domain.scope
            ).exports_identifier_domains(context):
                identifier_subjects.add(subject)
            if context.owns_payload_spatial_location():
                location_subjects.add(subject)
        return cls(
            frozenset(count_subjects),
            frozenset(identifier_subjects),
            frozenset(location_subjects),
        )

    def primary_row_reserved_count_subjects(
        self,
        explicit_subjects: frozenset[RuntimeMeasurementSubjectKey],
    ) -> frozenset[RuntimeMeasurementSubjectKey]:
        """Return subjects unavailable to primary-row object-count fallback."""

        return explicit_subjects | self.count_subjects

    def primary_row_reserved_count_subjects_from_features(
        self,
        measurement_fact_counts: RuntimeMeasurementFactCounterMapping,
        policy: RuntimeEquivalencePolicy,
    ) -> frozenset[RuntimeMeasurementSubjectKey]:
        """Return object-count subjects unavailable to primary-row fallback."""
        explicit_subjects = object_measurement_subjects_matching_marker(
            measurement_fact_counts,
            ObjectCountFeatureMarker,
            policy,
        )
        return self.primary_row_reserved_count_subjects(explicit_subjects)

    def primary_row_reserved_identifier_subjects_from_features(
        self,
        measurement_fact_counts: RuntimeMeasurementFactCounterMapping,
        policy: RuntimeEquivalencePolicy,
    ) -> frozenset[RuntimeMeasurementSubjectKey]:
        """Return ObjectNumber subjects unavailable to primary-row fallback."""
        explicit_subjects = object_measurement_subjects_matching_marker(
            measurement_fact_counts,
            ObjectIdentifierFeatureMarker,
            policy,
        )
        return explicit_subjects | self.identifier_subjects


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
        records: Iterable[StoredRuntimeValue],
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


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectLabelMeasurementCompletion(ObjectLabelMeasurementState):
    """Complete implicit facts from all nominal object-label domains."""

    measurement_fact_counts: RuntimeMeasurementFactCounterMapping

    @classmethod
    def from_feature_state(
        cls,
        *,
        policy: RuntimeEquivalencePolicy,
        measurement_fact_counts: RuntimeMeasurementFactCounterMapping,
        required_keys: RuntimeRequiredMeasurementKeys,
        object_identifier_subjects: frozenset[RuntimeMeasurementSubjectKey]
        | None = None,
        object_location_subjects: frozenset[
            RuntimeMeasurementSubjectKey
        ] | None = None,
        object_location_aggregate_subjects: frozenset[
            RuntimeMeasurementSubjectKey
        ] = frozenset(),
    ) -> "ObjectLabelMeasurementCompletion":
        """Build object-label completion from explicit measurement facts."""
        explicit_object_location_subjects = object_measurement_subjects_matching_marker(
            measurement_fact_counts,
            ObjectLocationFeatureMarker,
            policy,
        )
        return cls(
            policy=policy,
            measurement_fact_counts=measurement_fact_counts,
            object_identifier_subjects=(
                object_measurement_subjects_matching_marker(
                    measurement_fact_counts,
                    ObjectIdentifierFeatureMarker,
                    policy,
                )
                if object_identifier_subjects is None
                else object_identifier_subjects
            ),
            object_location_subjects=(
                explicit_object_location_subjects
                if object_location_subjects is None
                else object_location_subjects
            ),
            object_count_subjects=object_measurement_subjects_matching_marker(
                measurement_fact_counts,
                ObjectCountFeatureMarker,
                policy,
            ),
            required_keys=required_keys,
            object_location_aggregate_subjects=object_location_aggregate_subjects,
        )

    def facts_for_records(
        self,
        records: Sequence[StoredRuntimeValue],
    ) -> RuntimeMeasurementFacts:
        expected_by_key: RuntimeMeasurementFactCounterMap = {}
        facts: RuntimeMeasurementFactList = []
        for record in records:
            context = ObjectLabelMeasurementContext.from_runtime_value(
                record.value,
                self.policy,
                self.object_identifier_subjects,
                self.object_location_subjects,
                self.object_count_subjects,
                self.required_keys,
                self.object_location_aggregate_subjects,
            )
            self._add_expected_context_counts(expected_by_key, context)
            facts.extend(context.measurement_facts())
        return (
            *RuntimeExpectedMeasurementFactCompletion(
                expected_by_key,
                self.measurement_fact_counts,
            ).missing_facts(),
            *facts,
        )

    def facts_for_primary_rows(
        self,
        primary_row_identities: Iterable[
            tuple[RuntimeMeasurementSubjectKey, RuntimeObjectMeasurementRowIdentity]
        ],
    ) -> RuntimeMeasurementFacts:
        """Complete ObjectNumber facts from explicit primary object measurement rows."""
        expected_by_key: RuntimeMeasurementFactCounterMap = {}
        for subject, identity in primary_row_identities:
            self._add_expected_primary_row_counts(
                expected_by_key,
                subject,
                identity,
            )
        return RuntimeExpectedMeasurementFactCompletion(
            expected_by_key,
            self.measurement_fact_counts,
        ).missing_facts()

    def _add_expected_primary_row_counts(
        self,
        expected_by_key: RuntimeMeasurementFactCounterMap,
        subject: RuntimeMeasurementSubjectKey,
        identity: RuntimeObjectMeasurementRowIdentity,
    ) -> None:
        if subject in self.object_identifier_subjects:
            return
        object_label = identity.object_label_signature
        if object_label is None:
            return
        keys = RequiredRuntimeMeasurementProjection(
            self.required_keys,
            self.policy,
        ).object_identifier_keys(subject)
        for key in keys:
            if key not in expected_by_key:
                expected_by_key[key] = Counter()
            expected_by_key[key][object_label] += 1

    def _add_expected_context_counts(
        self,
        expected_by_key: RuntimeMeasurementFactCounterMap,
        context: ObjectLabelMeasurementContext,
    ) -> None:
        if context.object_name is None:
            return
        subject = RuntimeMeasurementSubjectKey(
            MeasurementScope.OBJECT,
            context.object_name,
        )
        if subject in self.object_identifier_subjects:
            return
        keys = RequiredRuntimeMeasurementProjection(
            context.required_keys,
            context.policy,
        ).object_identifier_keys(subject)
        if not keys:
            return
        if context.label_array() is None:
            return
        projection_strategy = ObjectLabelMeasurementProjectionStrategy.for_scope(
            context.domain.scope
        )
        if not projection_strategy.exports_identifier_domains(context):
            return
        object_number_domains = projection_strategy.identifier_domains(context)
        for object_ids in object_number_domains:
            for key in keys:
                if key not in expected_by_key:
                    expected_by_key[key] = Counter()
                counter = expected_by_key[key]
                for object_id in object_ids:
                    counter[runtime_cell_signature(str(object_id), context.policy)] += 1


class ObjectLabelMeasurementProjectionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Project object-label payloads into measurement domains by declared scope."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)

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
        context: ObjectLabelMeasurementContext,
    ) -> tuple[ObjectLabelMeasurementProjection, ...]:
        """Return semantic label domains used for count/location facts."""

    def exports_identifier_domains(
        self,
        context: ObjectLabelMeasurementContext,
    ) -> bool:
        """Return whether this label domain represents semantic ObjectNumber IDs."""
        if context.object_labels.domain.scope is not ObjectLabelDomainScope.PLANE:
            return True
        if context.object_labels.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return True
        if len(context.object_labels.source_image_names) <= 1:
            return True
        return object_label_dense_array(context.object_labels).ndim < 4

    def identifier_domains(
        self,
        context: ObjectLabelMeasurementContext,
    ) -> tuple[tuple[int, ...], ...]:
        """Return ObjectNumber domains to emit for an object-label context."""
        if context.has_declared_object_domain():
            return self.declared_identifier_domains(context)
        return context.dense_identity_domains()

    def declared_identifier_domains(
        self,
        context: ObjectLabelMeasurementContext,
    ) -> tuple[tuple[int, ...], ...]:
        """Return declared ObjectNumber domains aligned to measurement planes."""
        domain_stack = DenseObjectLabelPlaneDomainStackRequest(
            context.object_labels,
            allow_single_plane=True,
            collapse_repeated=True,
        ).stack()
        if domain_stack is not None:
            return domain_stack.object_id_domains
        return context.dense_plane_id_domains()


class PayloadObjectLabelMeasurementProjectionStrategy(
    ObjectLabelMeasurementProjectionStrategy
):
    """Payload-scoped labels measure the dense object payload as one domain."""

    scope = ObjectLabelDomainScope.PAYLOAD
    strategy_label = ObjectLabelDomainScope.PAYLOAD.value

    def projections(
        self,
        context: ObjectLabelMeasurementContext,
    ) -> tuple[ObjectLabelMeasurementProjection, ...]:
        label_array = context.label_array()
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
        context: ObjectLabelMeasurementContext,
    ) -> tuple[ObjectLabelMeasurementProjection, ...]:
        domain_stack = DenseObjectLabelPlaneDomainStackRequest(
            context.object_labels,
            allow_single_plane=True,
            collapse_repeated=True,
        ).stack()
        if domain_stack is not None:
            return tuple(
                ObjectLabelMeasurementProjection(
                    domain_stack.labels[index],
                    object_ids,
                    index,
                )
                for index, object_ids in enumerate(domain_stack.object_id_domains)
            )
        label_array = context.label_array()
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


def object_label_measurements_required(
    object_name: str | None,
    required_keys: RuntimeRequiredMeasurementKeys,
) -> bool:
    if required_keys is None:
        return True
    if object_name is None:
        return False
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.OBJECT, object_name)
    return any(key.subject == subject for key in required_keys)


def object_location_measurement_facts_for_plane(
    labels: np.ndarray,
    subject: RuntimeMeasurementSubjectKey,
    policy: RuntimeEquivalencePolicy,
    *,
    required_feature_names: frozenset[str] | None = None,
    required_mean_feature_names: frozenset[str] | None = None,
    object_ids: tuple[int, ...] | None = None,
    include_missing: bool = True,
) -> RuntimeMeasurementFacts:
    if (
        required_feature_names is not None
        and not required_feature_names
        and required_mean_feature_names is not None
        and not required_mean_feature_names
    ):
        return ()
    _resolved_object_ids, coordinate_arrays = _object_label_location_coordinate_arrays(
        labels,
        object_ids=object_ids,
    )
    if not coordinate_arrays:
        return ()
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


def _object_label_location_coordinate_arrays(
    labels: np.ndarray,
    *,
    object_ids: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[tuple[str, ObjectLocationCoordinateValues], ...]]:
    if labels.size == 0:
        return (), ()
    integer_labels = np.asarray(labels)
    resolved_object_ids = (
        object_ids if object_ids is not None else dense_object_label_id_domain(integer_labels)
    )
    if not resolved_object_ids:
        return (), ()

    max_object_id = max(resolved_object_ids)
    if max_object_id <= 0:
        return (), ()
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

    return (
        resolved_object_ids,
        tuple(
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
        ),
    )


def _object_location_feature_facts(
    subject: RuntimeMeasurementSubjectKey,
    feature_name: str,
    values: np.ndarray,
    policy: RuntimeEquivalencePolicy,
    *,
    include_missing: bool = True,
) -> RuntimeMeasurementFacts:
    key = RuntimeMeasurementFeatureKey(subject, feature_name)
    return tuple(
        (key, runtime_cell_signature(str(value), policy))
        for value in values
        if include_missing or np.isfinite(value)
    )


def _object_location_mean_feature_fact(
    subject: RuntimeMeasurementSubjectKey,
    feature_name: str,
    values: np.ndarray,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeMeasurementFacts:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return ()
    key = RuntimeMeasurementFeatureKey(subject, feature_name, "mean")
    return ((key, runtime_cell_signature(str(float(np.mean(finite_values))), policy)),)


def object_label_measurement_values_for_name(
    records: Iterable[StoredRuntimeValue],
    object_name: str,
    policy: RuntimeEquivalencePolicy,
    *,
    required_keys: RuntimeRequiredMeasurementKeys = None,
) -> RuntimeObjectValuesByLabel:
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

    values_by_feature: RuntimeObjectValuesByLabel = {}
    for record in records:
        object_labels = ObjectLabelSet.from_runtime_value(record.value)
        if normalize_runtime_identifier(object_labels.name) != subject.name:
            continue
        for key, values in object_label_location_values_by_label(
            object_labels,
            subject,
            policy,
            required_feature_names=required_feature_names,
        ).items():
            if key not in values_by_feature:
                values_by_feature[key] = {}
            values_by_feature[key].update(values)
    return values_by_feature


def object_label_location_values_by_label(
    object_labels: ObjectLabelValue,
    subject: RuntimeMeasurementSubjectKey,
    policy: RuntimeEquivalencePolicy,
    *,
    required_feature_names: frozenset[str] | None = None,
) -> RuntimeObjectValuesByLabel:
    if required_feature_names is not None and not required_feature_names:
        return {}
    context = ObjectLabelMeasurementContext.from_object_labels(
        object_labels=object_labels,
        object_name=subject.name,
        policy=policy,
        object_identifier_subjects=frozenset(),
        object_location_subjects=frozenset(),
        object_count_subjects=frozenset(),
        required_keys=None,
    )
    projections = ObjectLabelMeasurementProjectionStrategy.for_scope(
        context.domain.scope
    ).projections(context)
    if not projections:
        return {}

    values_by_feature: RuntimeObjectValuesByLabel = {}
    for projection in projections:
        for key, values in _object_label_location_values_by_label_for_plane(
            projection.labels,
            subject,
            required_feature_names=required_feature_names,
            object_ids=projection.object_ids,
            slice_index=projection.slice_index,
        ).items():
            if key not in values_by_feature:
                values_by_feature[key] = {}
            values_by_feature[key].update(values)
    return values_by_feature


def _object_label_location_values_by_label_for_plane(
    labels: np.ndarray,
    subject: RuntimeMeasurementSubjectKey,
    *,
    required_feature_names: frozenset[str] | None = None,
    object_ids: tuple[int, ...] | None = None,
    slice_index: int | None = None,
) -> RuntimeObjectValuesByLabel:
    if required_feature_names is not None and not required_feature_names:
        return {}
    resolved_object_ids, coordinate_arrays = _object_label_location_coordinate_arrays(
        labels,
        object_ids=object_ids,
    )
    if not coordinate_arrays:
        return {}

    return {
        RuntimeMeasurementFeatureKey(subject, feature_name): {
            ObjectInstanceKey(label, slice_index=slice_index): float(value)
            for label, value in zip(resolved_object_ids, coordinate.values, strict=True)
            if coordinate.include_missing or np.isfinite(value)
        }
        for feature_name, coordinate in coordinate_arrays
        if required_feature_names is None or feature_name in required_feature_names
    }
