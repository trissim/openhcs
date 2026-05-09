"""Staged runtime equivalence APIs."""

from nominal_refactor_advisor.collection_algebra import sorted_tuple

from openhcs.core.equivalence.report import (
    RuntimeEquivalenceDifference,
    RuntimeEquivalenceDifferenceKind,
    RuntimeEquivalenceReport,
)
from openhcs.core.equivalence.policy import (
    DEFAULT_RUNTIME_MEASUREMENT_DIALECT,
    RuntimeEquivalencePolicy,
    RuntimeMeasurementDialect,
    RuntimeMeasurementFeatureNameMode,
    RuntimeMeasurementFeatureNumericTolerance,
    RuntimeMeasurementQualifierValueMode,
    RuntimeMeasurementRowQualifier,
    RuntimeMeasurementRowQualifierSequence,
    RuntimeMeasurementSourceQualifiedFeature,
    RuntimeMeasurementSourceNameEncoding,
    normalize_runtime_identifier,
    normalize_runtime_source_name,
    runtime_source_name_tokens,
)
from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.cells import (
    RuntimeCellMissingStrategy,
    RuntimeCellSignature,
    RuntimeCellValueKind,
    absolute_numeric_counters_equivalent,
    finite_signature_number,
    runtime_cell_signature,
    runtime_cell_signature_counters_equivalent,
    sparse_absolute_numeric_counters_equivalent,
    sparse_numeric_counters_equivalent,
)
from openhcs.core.equivalence.images import RuntimeImageSnapshot
from openhcs.core.equivalence.outputs import RuntimeOutputSnapshot
from openhcs.core.equivalence.tables import RuntimeTableSnapshot
from openhcs.core.equivalence.measurement_rows import (
    IMAGE_IDENTITY_FIELDS,
    axis_scoped_measurement_row_identity,
    measurement_qualifier_field_names,
    measurement_row_image_identity_key,
    measurement_row_qualifiers,
    measurement_row_qualifiers_from_indexed_values_cached,
    measurement_row_qualifiers_from_values,
    row_qualifier_applies_to_field,
    row_qualifier_columns,
    row_qualifier_values,
)
from openhcs.core.equivalence.measurement_facts import (
    RuntimeMeasurementFact,
    RuntimeMeasurementFactCounters,
    RuntimeMeasurementFacts,
    RuntimeRequiredMeasurementKeys,
    record_measurement_facts,
    spatial_grid_measurement_facts,
)
from openhcs.core.equivalence.comparison import (
    runtime_image_differences,
    runtime_table_differences,
)

__all__ = sorted_tuple(name for name in globals() if not name.startswith("_"))
