"""CellProfiler dialect support for OpenHCS.

This package owns product-facing CellProfiler semantics. Benchmark modules may
use these APIs to run parity checks, but benchmark code should not be the
canonical owner of `.cppipe` import or CellProfiler measurement semantics.
"""

from openhcs.core.public_api import exported_public_names
from openhcs.interop.cellprofiler.measurement_dialect import (
    BENCHMARK_CACHE_DOMAINS,
    CELLPROFILER_FEATURE_NUMERIC_TOLERANCES,
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
from openhcs.interop.cellprofiler.import_records import (
    CellProfilerModuleReference,
    CellProfilerPipelineImportResult,
    CellProfilerPipelineProvenance,
)
from openhcs.interop.cellprofiler.import_service import (
    CellProfilerPipelineImporter,
    CellProfilerPipelineImportRequest,
)
from openhcs.processing.backends.cellprofiler.illumination import (
    IlluminationCalculationScope,
    IlluminationCorrectionMethod,
    IlluminationFilterSizeMethod,
    IlluminationIntensityChoice,
    IlluminationRescaleOption,
    IlluminationSmoothingMethod,
    IlluminationSplineBackgroundMode,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    CellProfilerExpandShrinkOperation,
    ExpandShrinkMode,
)
from openhcs.processing.backends.cellprofiler.intensity_distribution import (
    IntensityDistributionCenterChoice,
    IntensityDistributionZernikeMode,
    parse_intensity_distribution_center_choice,
    parse_intensity_distribution_zernike_mode,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    CombineObjectsMethod,
)
from openhcs.processing.backends.cellprofiler.object_images import (
    ConvertObjectsToImageMode,
)
from openhcs.processing.backends.cellprofiler.image_quality import (
    ImageQualityThresholdMethod,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    MaskImageSource,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    RescaleIntensityAutomaticHigh,
    RescaleIntensityAutomaticLow,
    RescaleIntensityMethod,
)
from openhcs.processing.backends.cellprofiler.watershed import (
    WatershedDeclumpMethod,
    WatershedInputKeyword,
    WatershedMethod,
)
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathOperation,
    parse_image_math_operation,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    MaskObjectsNumberingChoice,
    MaskObjectsOverlapHandling,
)
from openhcs.interop.cellprofiler.artifact_semantics import (
    ArtifactSettingDirection,
    ArtifactSettingRole,
    ArtifactSettingSymbol,
    FunctionSpecialOutput,
    artifact_setting_symbols,
    function_special_outputs,
)
from openhcs.interop.cellprofiler.compiler_registry import (
    clear_cellprofiler_dialect_compiler,
    get_cellprofiler_dialect_compiler,
    register_cellprofiler_dialect_compiler,
)
from openhcs.interop.cellprofiler.module_roles import (
    ArtifactSpecKey,
    CellProfilerModuleRole,
    CellProfilerModuleRoleSpec,
    cellprofiler_infrastructure_import_note,
    cellprofiler_infrastructure_retained_artifacts,
    cellprofiler_module_role,
)
from openhcs.interop.cellprofiler.module_semantics import (
    CellProfilerModuleCategory,
    CellProfilerModuleDimensionality,
    CellProfilerModuleSemanticFamily,
    CellProfilerModuleSemantics,
    cellprofiler_module_semantics_family,
    cellprofiler_module_semantics,
)
from openhcs.interop.cellprofiler.parser import (
    CPPipeParser,
    ModuleBlock,
    ModuleSetting,
)
from openhcs.interop.cellprofiler.pipeline_compiler import (
    CellProfilerDialectCompiler,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    RelateObjectsDistanceMethod,
)
from openhcs.interop.cellprofiler.relationship_measurements import (
    RelationshipMeasurements,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    InterpolationMethod as ResizeInterpolationMethod,
    ResizeMethod,
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
from openhcs.processing.backends.cellprofiler.module_classes import (
    DEFAULT_STRUCTURING_ELEMENT_SETTING,
    STRUCTURING_ELEMENT_SETTING_NAME,
    CellProfilerStructuringElement,
    StructuringElementSetting,
    StructuringElementSettingBinding,
    structuring_element_bound_kwargs,
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

__all__ = exported_public_names(globals(), excluded_names=("exported_public_names",))
