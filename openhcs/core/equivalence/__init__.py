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
    normalize_runtime_identifier,
    normalize_runtime_source_name,
)
from openhcs.core.equivalence.keys import (
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementSubjectKey,
)
from openhcs.core.equivalence.cells import (
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
from openhcs.core.equivalence.comparison import (
    runtime_image_differences,
    runtime_table_differences,
)

__all__ = sorted_tuple(name for name in globals() if not name.startswith("_"))
