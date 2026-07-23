"""Measurement fact primitives for runtime equivalence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast, Generic, TypeVar

from openhcs.core.equivalence.cells import (
    RuntimeCellMissingStrategy,
    RuntimeCellSignature,
    finite_signature_number,
    runtime_cell_signature,
)
from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementNamePartsProjection,
    RuntimeMeasurementSourcePair,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.measurement_features import (
    object_measurement_feature_matches_marker,
)
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    RuntimeMeasurementDialect,
    normalize_runtime_identifier,
)
from openhcs.core.runtime_measurements import (
    MeasuredObjectAnchorFeatureMarker,
    MeasurementScope,
)
from openhcs.core.runtime_artifact_values import (
    RuntimeValue,
)
from openhcs.core.runtime_spatial_grid import (
    SpatialGrid,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet

RuntimeMeasurementFact = tuple[
    RuntimeMeasurementFeatureKey,
    RuntimeCellSignature,
]
RuntimeMeasurementFacts = tuple[RuntimeMeasurementFact, ...]
RuntimeMeasurementFactList = list[RuntimeMeasurementFact]
RuntimeMeasurementFactCounterMap = dict[
    RuntimeMeasurementFeatureKey,
    Counter[RuntimeCellSignature],
]
RuntimeMeasurementFactCounterMapping = Mapping[
    RuntimeMeasurementFeatureKey,
    Counter[RuntimeCellSignature],
]
RuntimeMeasurementKeySet = frozenset[RuntimeMeasurementFeatureKey]
RuntimeRequiredMeasurementKeys = frozenset[RuntimeMeasurementFeatureKey] | None
RuntimeRowProjectionValueT = TypeVar("RuntimeRowProjectionValueT")
RuntimeMeasurementPaddingGroup = tuple[
    RuntimeMeasurementSubjectKey,
    str | None,
    tuple[str, ...],
]
@dataclass(frozen=True, slots=True)
class RuntimeRowProjectionRecord(Generic[RuntimeRowProjectionValueT]):
    """One exact projected cell before same-row alias consolidation."""

    padding_group: RuntimeMeasurementPaddingGroup
    key: RuntimeMeasurementFeatureKey
    value: RuntimeRowProjectionValueT
    measured_object_anchor: bool = False
    producer_owned_feature: bool = False


RuntimeRowProjectionRecords = tuple[
    RuntimeRowProjectionRecord[RuntimeRowProjectionValueT],
    ...,
]


def runtime_measurement_fact_counter(
    measurement_fact_counts: RuntimeMeasurementFactCounterMap,
    key: RuntimeMeasurementFeatureKey,
) -> Counter[RuntimeCellSignature]:
    """Return the mutable counter for one measurement feature key."""
    counter = measurement_fact_counts.get(key)
    if counter is None:
        counter = Counter()
        measurement_fact_counts[key] = counter
    return counter


@dataclass(frozen=True, slots=True)
class RuntimeExpectedMeasurementFactCompletion:
    """Materialize expected facts missing from explicit runtime measurements."""

    expected_by_key: RuntimeMeasurementFactCounterMap
    measurement_fact_counts: RuntimeMeasurementFactCounterMapping

    def missing_facts(self) -> RuntimeMeasurementFacts:
        facts: RuntimeMeasurementFactList = []
        for key, expected_counter in self.expected_by_key.items():
            if key in self.measurement_fact_counts:
                explicit_counter = self.measurement_fact_counts[key]
                for signature, expected_count in expected_counter.items():
                    missing_count = expected_count - explicit_counter[signature]
                    if missing_count <= 0:
                        continue
                    facts.extend((key, signature) for _index in range(missing_count))
                continue
            for signature, expected_count in expected_counter.items():
                missing_count = expected_count
                if missing_count <= 0:
                    continue
                facts.extend((key, signature) for _index in range(missing_count))
        return tuple(facts)


def record_measurement_facts(
    measurement_fact_counts: RuntimeMeasurementFactCounterMap,
    facts: Iterable[RuntimeMeasurementFact],
    *,
    required_keys: RuntimeRequiredMeasurementKeys = None,
) -> None:
    for key, value in facts:
        if required_keys is not None and key not in required_keys:
            continue
        runtime_measurement_fact_counter(measurement_fact_counts, key)[value] += 1


def _reverse_regression_slope(
    correlation_value: RuntimeCellSignature,
    slope_value: RuntimeCellSignature,
) -> float | None:
    correlation = finite_signature_number(correlation_value)
    slope = finite_signature_number(slope_value)
    if correlation is None or slope is None or slope == 0:
        return None
    return (correlation * correlation) / slope


@dataclass(frozen=True, slots=True)
class RuntimeDirectionalPairMeasurementDerivationContract:
    """SSOT for directional pair facts derivable from equivalent orientations."""

    policy: RuntimeEquivalencePolicy
    known_source_names: tuple[str, ...] = ()

    @property
    def regression_slope_feature(self) -> str | None:
        """Return the dialect-declared pair regression-slope family."""
        return self.policy.measurement_dialect.resolved_pair_regression_slope_feature_name()

    @property
    def correlation_feature(self) -> str | None:
        """Return the dialect-declared pair correlation family."""
        return self.policy.measurement_dialect.resolved_pair_correlation_feature_name()

    @property
    def regression_slope_family(self) -> tuple[str, ...]:
        """Return the source-qualified family tuple for pair slopes."""
        feature = self.regression_slope_feature
        return () if feature is None else (feature,)

    @property
    def feature_families(self) -> tuple[str, ...]:
        """Return pair feature families participating in orientation derivation."""
        return tuple(
            feature
            for feature in (self.regression_slope_feature, self.correlation_feature)
            if feature is not None
        )

    def required_keys_need_derivation(
        self,
        required_keys: RuntimeRequiredMeasurementKeys,
    ) -> bool:
        """Return whether required output keys can consume derived pair facts."""
        if not self.regression_slope_family:
            return False
        if required_keys is None:
            return False
        return any(
            key.belongs_to_source_qualified_feature_family(
                self.policy.measurement_dialect,
                self.regression_slope_family,
            )
            for key in required_keys
        )

    def required_input_keys(
        self,
        key: RuntimeMeasurementFeatureKey,
    ) -> tuple[RuntimeMeasurementFeatureKey, ...]:
        """Return orientation inputs needed to satisfy a required pair key."""
        input_keys: list[RuntimeMeasurementFeatureKey] = []
        pair = self.declared_source_pair_for_key(
            key,
            self.feature_families,
        )
        if pair is not None:
            input_keys.append(
                key.source_pair_feature_key(
                    pair.reversed_source_name,
                    self.policy.measurement_dialect,
                    self.feature_families,
                )
            )

        if key.belongs_to_source_qualified_feature_family(
            self.policy.measurement_dialect,
            self.regression_slope_family,
        ):
            pair = self.declared_source_pair_for_key(
                key,
                self.regression_slope_family,
            )
            if pair is not None and self.correlation_feature is not None:
                input_keys.extend(
                    self.source_pair_feature_key(
                        key,
                        self.correlation_feature,
                        source_name,
                    )
                    for source_name in (pair.source_name, pair.reversed_source_name)
                )
        return tuple(dict.fromkeys(input_keys))

    def declared_source_pair_for_key(
        self,
        key: RuntimeMeasurementFeatureKey,
        feature_families: tuple[str, ...],
    ) -> RuntimeMeasurementSourcePair | None:
        """Resolve a key against canonical source-pair declarations only."""
        direct_pair = key.source_pair(
            self.policy.measurement_dialect,
            feature_families,
        )
        if direct_pair is not None:
            return direct_pair

        declared_pairs = tuple(
            pair
            for source_name in self.known_source_names
            if (pair := RuntimeMeasurementSourcePair.from_source_name(source_name))
            is not None
        )
        for declared_pair in dict.fromkeys(declared_pairs):
            for candidate in (
                declared_pair,
                RuntimeMeasurementSourcePair(
                    declared_pair.second,
                    declared_pair.first,
                ),
            ):
                if (
                    key.source_pair_feature_key(
                        candidate.source_name,
                        self.policy.measurement_dialect,
                        feature_families,
                    )
                    == key
                ):
                    return candidate
        return None

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
        facts: RuntimeMeasurementFacts,
    ) -> RuntimeMeasurementFacts:
        """Derive mathematically equivalent directional pair facts."""
        if not self.regression_slope_family or self.correlation_feature is None:
            return facts
        slope_facts = tuple(
            (key, value)
            for key, value in facts
            if key.belongs_to_source_qualified_feature_family(
                self.policy.measurement_dialect,
                self.regression_slope_family,
            )
        )
        if not slope_facts:
            return facts

        derived: RuntimeMeasurementFactList = []
        values_by_key = dict(facts)
        for key, slope_value in slope_facts:
            pair = self.declared_source_pair_for_key(
                key,
                self.regression_slope_family,
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
            reversed_key = key.source_pair_feature_key(
                pair.reversed_source_name,
                self.policy.measurement_dialect,
                self.regression_slope_family,
            )
            derived.append(
                (
                    reversed_key,
                    runtime_cell_signature(repr(reverse_slope), self.policy),
                )
            )
        if not derived:
            return facts
        return RuntimeMeasurementFactProjectionContract.dedupe_alias_facts(
            (*facts, *derived)
        )

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
                self.correlation_feature,
                source_name,
            ),
            self.source_pair_feature_key(
                key,
                self.correlation_feature,
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
        records: Iterable[RuntimeRowProjectionRecord[RuntimeCellSignature]],
        policy: RuntimeEquivalencePolicy,
        *,
        declared_anchor_groups: frozenset[RuntimeMeasurementPaddingGroup] = frozenset(),
    ) -> frozenset[RuntimeMeasurementPaddingGroup]:
        """Return padding groups that carry observed measurement facts."""
        anchor_key_cache: dict[RuntimeMeasurementFeatureKey, bool] = {}
        has_anchor = set(declared_anchor_groups)
        observed_anchors: set[RuntimeMeasurementPaddingGroup] = set()
        observed_values: set[RuntimeMeasurementPaddingGroup] = set()
        for record in records:
            observed = cls.is_observed_value(record.value)
            if observed:
                observed_values.add(record.padding_group)
            is_anchor = anchor_key_cache.get(record.key)
            if is_anchor is None:
                is_anchor = object_measurement_feature_matches_marker(
                    record.key,
                    MeasuredObjectAnchorFeatureMarker,
                    policy,
                )
                anchor_key_cache[record.key] = is_anchor
            if not is_anchor:
                continue
            has_anchor.add(record.padding_group)
            if observed:
                observed_anchors.add(record.padding_group)
        return frozenset(observed_anchors | (observed_values - has_anchor))

    @classmethod
    def padding_group(
        cls,
        table_group: str,
        field_name: str,
        key: RuntimeMeasurementFeatureKey,
        dialect: RuntimeMeasurementDialect,
    ) -> RuntimeMeasurementPaddingGroup:
        """Return the row-padding family for a measurement field."""
        normalized_field = normalize_runtime_identifier(field_name)
        parts = tuple(part for part in normalized_field.split("_") if part)
        feature_group = RuntimeMeasurementNamePartsProjection(
            parts,
            dialect,
        ).category_prefix() or (table_group,)
        return key.subject, key.source_name, feature_group

    @classmethod
    def observed_records(
        cls,
        records: Iterable[RuntimeRowProjectionRecord[RuntimeCellSignature]],
        policy: RuntimeEquivalencePolicy,
        *,
        declared_anchor_groups: frozenset[RuntimeMeasurementPaddingGroup] = frozenset(),
    ) -> RuntimeRowProjectionRecords[RuntimeCellSignature]:
        """Return records whose padding family has an observed anchor/value."""
        materialized = tuple(records)
        observed_padding_groups = cls.observed_padding_groups(
            materialized,
            policy,
            declared_anchor_groups=declared_anchor_groups,
        )
        return tuple(
            record
            for record in materialized
            if record.padding_group in observed_padding_groups
        )

    @classmethod
    def dedupe_observed_alias_records(
        cls,
        records: Iterable[RuntimeRowProjectionRecord[RuntimeCellSignature]],
        policy: RuntimeEquivalencePolicy,
    ) -> RuntimeMeasurementFacts:
        """Filter unobserved padding groups and collapse same-row aliases."""
        return cls.dedupe_alias_facts(
            (record.key, record.value)
            for record in cls.observed_records(records, policy)
        )

    @classmethod
    def dedupe_observed_records(
        cls,
        records: Iterable[RuntimeRowProjectionRecord[RuntimeCellSignature]],
        policy: RuntimeEquivalencePolicy,
    ) -> RuntimeMeasurementFacts:
        """Filter unobserved padding groups and collapse declared aliases."""
        materialized = tuple(records)
        observed_padding_groups = cls.observed_padding_groups(
            materialized,
            policy,
        )
        return cls.dedupe_records(
            record
            for record in materialized
            if record.padding_group in observed_padding_groups
        )

    @classmethod
    def dedupe_alias_facts(
        cls,
        facts: Iterable[RuntimeMeasurementFact],
    ) -> RuntimeMeasurementFacts:
        """Collapse same-row aliases that map to the same semantic feature key."""
        values_by_key: dict[RuntimeMeasurementFeatureKey, RuntimeCellSignature] = {}
        for key, value in facts:
            if not cls.is_observed_value(value):
                continue
            current = values_by_key.get(key)
            if current is None or (
                RuntimeCellMissingStrategy.for_kind(current.kind).is_missing(current)
                and not RuntimeCellMissingStrategy.for_kind(value.kind).is_missing(
                    value
                )
            ):
                values_by_key[key] = value
        return tuple(values_by_key.items())

    @classmethod
    def dedupe_records(
        cls,
        records: Iterable[RuntimeRowProjectionRecord[RuntimeCellSignature]],
    ) -> RuntimeMeasurementFacts:
        """Collapse aliases using producer-declared raw feature ownership."""
        values_by_key: dict[
            RuntimeMeasurementFeatureKey, list[RuntimeCellSignature]
        ] = {}
        producer_owned_by_key: dict[RuntimeMeasurementFeatureKey, bool] = {}
        for record in records:
            key = record.key
            value = record.value
            if not cls.is_observed_value(value):
                continue
            current_values = values_by_key.get(key)
            if current_values is None:
                values_by_key[key] = [value]
                producer_owned_by_key[key] = record.producer_owned_feature
                continue

            current_is_producer_owned = producer_owned_by_key[key]
            if record.producer_owned_feature and not current_is_producer_owned:
                values_by_key[key] = [value]
                producer_owned_by_key[key] = True
                continue
            if current_is_producer_owned and not record.producer_owned_feature:
                continue
            if value in current_values:
                continue
            current_values.append(value)

        return tuple(
            (key, value) for key, values in values_by_key.items() for value in values
        )


def spatial_grid_measurement_facts(
    value: RuntimeValue,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeMeasurementFacts:
    """Project a typed spatial-grid artifact to dialect-rendered image facts."""
    return tuple(
        fact
        for grid in _spatial_grids_from_runtime_value(value)
        for fact in _single_spatial_grid_measurement_facts(value, grid, policy)
    )


def _single_spatial_grid_measurement_facts(
    value: RuntimeValue,
    grid: SpatialGrid,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeMeasurementFacts:
    grid_name = normalize_runtime_identifier(value.name or grid.name)
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "Image")
    fields = (
        ("columns", grid.columns),
        ("rows", grid.rows),
        ("x_origin", grid.x_origin),
        ("x_spacing", grid.x_spacing),
        ("y_origin", grid.y_origin),
        ("y_spacing", grid.y_spacing),
    )
    return tuple(
        (
            RuntimeMeasurementFeatureKey(
                subject,
                policy.measurement_dialect.spatial_grid_measurement_feature_name(
                    grid_name,
                    field_name,
                ),
            ),
            runtime_cell_signature(str(field_value), policy),
        )
        for field_name, field_value in fields
    )


def _spatial_grids_from_runtime_value(value: RuntimeValue) -> tuple[SpatialGrid, ...]:
    if isinstance(value.data, RuntimeSliceAlignedValueSet):
        return tuple(
            cast(SpatialGrid, value.data.value_for_slice(index))
            for index in range(value.data.slice_count)
        )
    return (cast(SpatialGrid, value.data),)
