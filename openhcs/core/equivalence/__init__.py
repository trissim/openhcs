"""Staged runtime equivalence APIs."""

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

__all__ = (
    "DEFAULT_RUNTIME_MEASUREMENT_DIALECT",
    "RuntimeCellSignature",
    "RuntimeCellValueKind",
    "RuntimeEquivalenceDifference",
    "RuntimeEquivalenceDifferenceKind",
    "RuntimeEquivalencePolicy",
    "RuntimeEquivalenceReport",
    "RuntimeImageSnapshot",
    "RuntimeMeasurementDialect",
    "RuntimeMeasurementFeatureKey",
    "RuntimeMeasurementFeatureNameMode",
    "RuntimeMeasurementFeatureNumericTolerance",
    "RuntimeMeasurementQualifierValueMode",
    "RuntimeMeasurementRowQualifier",
    "RuntimeMeasurementSubjectKey",
    "RuntimeOutputSnapshot",
    "RuntimeTableSnapshot",
    "absolute_numeric_counters_equivalent",
    "finite_signature_number",
    "runtime_image_differences",
    "runtime_table_differences",
    "runtime_cell_signature",
    "runtime_cell_signature_counters_equivalent",
    "sparse_absolute_numeric_counters_equivalent",
    "sparse_numeric_counters_equivalent",
    "normalize_runtime_identifier",
    "normalize_runtime_source_name",
)
