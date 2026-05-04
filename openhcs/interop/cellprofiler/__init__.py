"""CellProfiler dialect support for OpenHCS.

This package owns product-facing CellProfiler semantics. Benchmark modules may
use these APIs to run parity checks, but benchmark code should not be the
canonical owner of `.cppipe` import or CellProfiler measurement semantics.
"""

from openhcs.interop.cellprofiler.measurement_dialect import (
    BENCHMARK_CACHE_DOMAINS,
    CELLPROFILER_FEATURE_NUMERIC_TOLERANCES,
    CELLPROFILER_MEASUREMENT_DIALECT,
    cellprofiler_runtime_equivalence_policy,
)
from openhcs.interop.cellprofiler.import_records import (
    CellProfilerModuleReference,
    CellProfilerPipelineImportResult,
    CellProfilerPipelineProvenance,
)
from openhcs.interop.cellprofiler.import_service import (
    CellProfilerPipelineImporter,
    CellProfilerPipelineImportRequest,
)
from openhcs.interop.cellprofiler.compiler_registry import (
    clear_cellprofiler_dialect_compiler,
    get_cellprofiler_dialect_compiler,
    register_cellprofiler_dialect_compiler,
)
from openhcs.interop.cellprofiler.module_roles import (
    INFRASTRUCTURE_MODULE_NAMES,
    INFRASTRUCTURE_MODULE_NAMES_BY_KEY,
    CellProfilerModuleRole,
    CellProfilerModuleRoleSpec,
    cellprofiler_module_role,
)
from openhcs.interop.cellprofiler.parser import (
    CPPipeParser,
    ModuleBlock,
    ModuleSetting,
)
from openhcs.interop.cellprofiler.pipeline_compiler import (
    CellProfilerDialectCompiler,
)
from openhcs.interop.cellprofiler.runtime import (
    CellProfilerImageExecutionContext,
    CellProfilerImageRequest,
    CellProfilerInvocationRequest,
    CellProfilerMeasurementImage,
    CellProfilerMeasurementImageDomain,
    CellProfilerResolvedInputRequest,
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
from openhcs.interop.cellprofiler.source_schema import (
    SetupModuleCompiler,
    compile_image_schema,
)
from openhcs.interop.cellprofiler.measurement_scope import (
    CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG,
    CellProfilerMeasurementTargetScope,
    coerce_cellprofiler_measurement_target_scope,
)

__all__ = (
    "BENCHMARK_CACHE_DOMAINS",
    "CELLPROFILER_FEATURE_NUMERIC_TOLERANCES",
    "CELLPROFILER_MEASUREMENT_DIALECT",
    "CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG",
    "CPPipeParser",
    "CellProfilerModuleReference",
    "CellProfilerModuleRole",
    "CellProfilerModuleRoleSpec",
    "CellProfilerImageExecutionContext",
    "CellProfilerImageRequest",
    "CellProfilerInvocationRequest",
    "CellProfilerMeasurementImage",
    "CellProfilerMeasurementImageDomain",
    "CellProfilerPipelineImportResult",
    "CellProfilerPipelineImporter",
    "CellProfilerPipelineImportRequest",
    "CellProfilerPipelineProvenance",
    "CellProfilerResolvedInputRequest",
    "CellProfilerSliceAlignedValues",
    "CellProfilerMeasurementTargetScope",
    "CellProfilerDialectCompiler",
    "INFRASTRUCTURE_MODULE_NAMES",
    "INFRASTRUCTURE_MODULE_NAMES_BY_KEY",
    "ModuleBlock",
    "ModuleSetting",
    "SettingNameFamily",
    "SettingToKeywordBinding",
    "SettingsBinder",
    "SetupModuleCompiler",
    "cellprofiler_runtime_equivalence_policy",
    "cellprofiler_module_role",
    "clear_cellprofiler_dialect_compiler",
    "compile_image_schema",
    "coerce_cellprofiler_measurement_target_scope",
    "get_cellprofiler_dialect_compiler",
    "normalize_cellprofiler_setting_name",
    "optional_setting_value",
    "register_cellprofiler_dialect_compiler",
    "required_setting_value",
    "setting_values",
)
