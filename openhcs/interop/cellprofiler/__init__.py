"""CellProfiler dialect support for OpenHCS.

This package owns product-facing CellProfiler semantics. Benchmark modules may
use these APIs to run parity checks, but benchmark code should not be the
canonical owner of `.cppipe` import or CellProfiler measurement semantics.
"""

from openhcs.core.public_api import exported_public_names
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_DIALECT,
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    cellprofiler_runtime_equivalence_policy,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    CellProfilerChildCountFeatureParser,
    CellProfilerMeasurementFeature,
    CellProfilerMeasurementFeatureKind,
    CellProfilerMeasurementFeatureParser,
    CellProfilerObjectCountFeatureParser,
    child_count_feature_child_name,
    count_feature_object_name,
)
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline
from openhcs.interop.cellprofiler.parser import (
    CPPipeParser,
    ModuleBlock,
    ModuleSetting,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
    CellProfilerMeasurementImage,
    CellProfilerMeasurementImageDomain,
    CellProfilerSliceAlignedValues,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.worm_measurements import (
    WormControlPointAxis,
    WormControlPointMeasurementField,
    WormControlPointMeasurementSchema,
)
from openhcs.interop.cellprofiler.measurement_scope import (
    CellProfilerMeasurementTargetScope,
    coerce_cellprofiler_measurement_target_scope,
)

# Importing the interop boundary registers the compiler provider for the
# declaration-owned CellProfiler callables exposed by the processing package.
from openhcs.interop.cellprofiler import compile_time_contracts as _compile_time_contracts

__all__ = exported_public_names(
    globals(),
    excluded_names=("exported_public_names", "_compile_time_contracts"),
)
