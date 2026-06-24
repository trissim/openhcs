"""Measurement fact primitives for runtime equivalence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from openhcs.core.equivalence.cells import (
    RuntimeCellMissingStrategy,
    RuntimeCellSignature,
    finite_signature_number,
    runtime_cell_signature,
)
from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementNamePartsProjection,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.measurement_features import (
    object_measurement_feature_has_role,
)
from openhcs.core.equivalence.policy import (
    RuntimeEquivalencePolicy,
    RuntimeMeasurementDialect,
    normalize_runtime_identifier,
)
from openhcs.core.runtime_semantics import (
    MeasurementScope,
    ObjectMeasurementFeatureRole,
    PairMeasurementFeature,
)
from openhcs.core.runtime_values import (
    RuntimeValue,
    SpatialGrid,
)

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
RuntimeRowProjectionRecord = tuple[
    RuntimeMeasurementPaddingGroup,
    RuntimeMeasurementFeatureKey,
    RuntimeRowProjectionValueT,
    bool,
]
RuntimeRowProjectionRecords = tuple[
    RuntimeRowProjectionRecord[RuntimeRowProjectionValueT],
    ...,
]
PAIR_CORRELATION_FEATURE = PairMeasurementFeature.CORRELATION.value
PAIR_REGRESSION_SLOPE_FEATURE = PairMeasurementFeature.REGRESSION_SLOPE.value


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
    def feature_families(self) -> tuple[str, ...]:
        """Return pair feature families participating in orientation derivation."""
        return (PAIR_REGRESSION_SLOPE_FEATURE, PAIR_CORRELATION_FEATURE)

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
            (PAIR_REGRESSION_SLOPE_FEATURE,),
        ):
            pair = key.source_pair(
                self.policy.measurement_dialect,
                (PAIR_REGRESSION_SLOPE_FEATURE,),
                self.known_source_names,
            )
            if pair is not None:
                input_keys.extend(
                    self.source_pair_feature_key(
                        key,
                        PAIR_CORRELATION_FEATURE,
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
        facts: RuntimeMeasurementFacts,
    ) -> RuntimeMeasurementFacts:
        """Derive mathematically equivalent directional pair facts."""
        slope_facts = tuple(
            (key, value)
            for key, value in facts
            if key.belongs_to_source_qualified_feature_family(
                self.policy.measurement_dialect,
                (PAIR_REGRESSION_SLOPE_FEATURE,),
            )
        )
        if not slope_facts:
            return facts

        derived: RuntimeMeasurementFactList = []
        values_by_key = dict(facts)
        for key, slope_value in slope_facts:
            pair = key.source_pair(
                self.policy.measurement_dialect,
                (PAIR_REGRESSION_SLOPE_FEATURE,),
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
                (PAIR_REGRESSION_SLOPE_FEATURE,),
                self.known_source_names,
            )
            if reversed_key is None:
                continue
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
                PAIR_CORRELATION_FEATURE,
                source_name,
            ),
            self.source_pair_feature_key(
                key,
                PAIR_CORRELATION_FEATURE,
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
    ) -> frozenset[RuntimeMeasurementPaddingGroup]:
        """Return padding groups that carry observed measurement facts."""
        records_by_group: dict[
            RuntimeMeasurementPaddingGroup,
            list[tuple[RuntimeMeasurementFeatureKey, RuntimeCellSignature]],
        ] = {}
        for padding_group, key, value, _qualified_observation in records:
            records_by_group.setdefault(padding_group, []).append((key, value))

        observed_groups: set[RuntimeMeasurementPaddingGroup] = set()
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
    ) -> RuntimeRowProjectionRecords[RuntimeCellSignature]:
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
        records: Iterable[RuntimeRowProjectionRecord[RuntimeCellSignature]],
        policy: RuntimeEquivalencePolicy,
    ) -> RuntimeMeasurementFacts:
        """Filter unobserved padding groups and collapse same-row aliases."""
        return cls.dedupe_alias_facts(
            (key, value)
            for (
                _padding_group,
                key,
                value,
                _qualified_observation,
            ) in cls.observed_records(records, policy)
        )

    @classmethod
    def dedupe_observed_qualified_records(
        cls,
        records: Iterable[RuntimeRowProjectionRecord[RuntimeCellSignature]],
        policy: RuntimeEquivalencePolicy,
    ) -> RuntimeMeasurementFacts:
        """Filter unobserved padding groups and collapse qualified aliases."""
        materialized = tuple(records)
        observed_padding_groups = cls.observed_padding_groups(
            (
                (padding_group, key, value, qualified_observation)
                for padding_group, key, value, qualified_observation in materialized
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
    ) -> RuntimeMeasurementFacts:
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


def spatial_grid_measurement_facts(
    value: RuntimeValue,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeMeasurementFacts:
    """Project a typed spatial-grid artifact to CellProfiler-style image facts."""
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
        ("x_location_of_lowest_x_spot", grid.x_location_of_lowest_x_spot),
        ("x_spacing", grid.x_spacing),
        ("y_location_of_lowest_y_spot", grid.y_location_of_lowest_y_spot),
        ("y_spacing", grid.y_spacing),
    )
    return tuple(
        (
            RuntimeMeasurementFeatureKey(
                subject,
                f"defined_grid_{grid_name}_{field_name}",
            ),
            runtime_cell_signature(str(field_value), policy),
        )
        for field_name, field_value in fields
    )


def _spatial_grids_from_runtime_value(value: RuntimeValue) -> tuple[SpatialGrid, ...]:
    if value.schema.slice_aligned:
        if not isinstance(value.data, tuple | list):
            raise TypeError(
                f"Slice-aligned spatial grid '{value.name}' payload must be a "
                f"sequence of mappings, got {type(value.data).__name__}."
            )
        return tuple(
            SpatialGrid.from_mapping(value.name, item)
            for item in value.data
            if isinstance(item, Mapping)
        )
    return (SpatialGrid.from_runtime_value(value),)
