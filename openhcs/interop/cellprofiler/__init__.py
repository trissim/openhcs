"""CellProfiler dialect support for OpenHCS.

This package owns product-facing CellProfiler semantics. Benchmark modules may
use these APIs to run parity checks, but benchmark code should not be the
canonical owner of `.cppipe` import or CellProfiler measurement semantics.
"""

from openhcs.interop.cellprofiler.measurement_dialect import (
    BENCHMARK_CACHE_DOMAINS,
    CELLPROFILER_FEATURE_NUMERIC_TOLERANCES,
    CELLPROFILER_MEASUREMENT_DIALECT,
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
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
from openhcs.interop.cellprofiler.illumination_settings import (
    CORRECT_ILLUMINATION_APPLY_SETTINGS,
    CORRECT_ILLUMINATION_CALCULATE_SETTINGS,
    IlluminationCalculationScope,
    IlluminationCorrectionMethod,
    IlluminationFilterSizeMethod,
    IlluminationIntensityChoice,
    IlluminationRescaleOption,
    IlluminationSmoothingMethod,
    IlluminationSplineBackgroundMode,
)
from openhcs.interop.cellprofiler.intensity_distribution_settings import (
    IntensityDistributionCenterChoice,
    IntensityDistributionZernikeMode,
    parse_intensity_distribution_center_choice,
    parse_intensity_distribution_zernike_mode,
)
from openhcs.interop.cellprofiler.image_module_settings import (
    ImageQualityThresholdMethod,
    MaskImageSource,
    RescaleIntensityAutomaticHigh,
    RescaleIntensityAutomaticLow,
    RescaleIntensityMethod,
)
from openhcs.interop.cellprofiler.mask_objects_settings import (
    MASK_OBJECTS_SETTINGS,
    MaskObjectsNumberingChoice,
    MaskObjectsOverlapHandling,
)
from openhcs.interop.cellprofiler.artifact_semantics import (
    ArtifactSettingClassifier,
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
    INFRASTRUCTURE_MODULE_NAMES,
    INFRASTRUCTURE_MODULE_NAMES_BY_KEY,
    CellProfilerModuleRole,
    CellProfilerModuleRoleSpec,
    cellprofiler_module_role,
)
from openhcs.interop.cellprofiler.module_runtime_semantics import (
    CellProfilerWatershedRuntimeFamily,
    ModuleRevisionRange,
    ModuleRuntimeSemanticsBinding,
    WatershedRuntimeSemanticsBinding,
)
from openhcs.interop.cellprofiler.module_semantics import (
    CELLPROFILER_MODULE_SEMANTICS,
    CELLPROFILER_MODULE_SEMANTICS_BY_KEY,
    CellProfilerModuleCategory,
    CellProfilerModuleDimensionality,
    CellProfilerModuleSemantics,
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
from openhcs.interop.cellprofiler.relate_objects_settings import (
    RELATE_OBJECTS_DISTANCE_SETTING,
    RELATE_OBJECTS_PER_PARENT_MEANS_SETTING,
    RELATE_OBJECTS_SAVE_CHILDREN_SETTING,
    RelateObjectsDistanceMethod,
    parse_relate_objects_distance_method,
)
from openhcs.interop.cellprofiler.resize_settings import (
    RESIZE_FACTOR_SETTING,
    RESIZE_FACTOR_X_SETTING,
    RESIZE_FACTOR_Y_SETTING,
    RESIZE_FACTOR_Z_SETTING,
    RESIZE_HEIGHT_SETTING,
    RESIZE_INTERPOLATION_SETTING,
    RESIZE_METHOD_SETTING,
    RESIZE_PLANES_SETTING,
    RESIZE_WIDTH_SETTING,
    ResizeInterpolationMethod,
    ResizeMethod,
    resize_bound_kwargs,
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
from openhcs.interop.cellprofiler.straighten_worms_settings import (
    STRAIGHTEN_WORMS_INPUT_IMAGE_SETTING,
    STRAIGHTEN_WORMS_INPUT_OBJECTS_SETTING,
    STRAIGHTEN_WORMS_OUTPUT_IMAGE_SETTING,
    STRAIGHTEN_WORMS_OUTPUT_OBJECTS_SETTING,
    StraightenWormsFlipMode,
    StraightenWormsImageBinding,
    straighten_worms_bound_kwargs,
    straighten_worms_image_bindings,
    straighten_worms_input_objects_name,
    straighten_worms_output_objects_name,
)
from openhcs.interop.cellprofiler.structuring_element_settings import (
    DEFAULT_STRUCTURING_ELEMENT_SETTING,
    STRUCTURING_ELEMENT_SETTING_NAME,
    CellProfilerStructuringElement,
    StructuringElementSetting,
    StructuringElementSettingBinding,
    structuring_element_bound_kwargs,
)
from openhcs.interop.cellprofiler.untangle_worms_settings import (
    UNTANGLE_WORMS_INPUT_IMAGE_SETTING,
    UNTANGLE_WORMS_NONOVERLAPPING_OBJECTS_SETTING,
    UNTANGLE_WORMS_OVERLAPPING_OBJECTS_SETTING,
    UNTANGLE_WORMS_TRAINING_FILE_NAME_SETTING,
    UntangleWormsOverlapStyle,
    untangle_worms_bound_kwargs,
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
    "CELLPROFILER_MODULE_SEMANTICS",
    "CELLPROFILER_MODULE_SEMANTICS_BY_KEY",
    "CORRECT_ILLUMINATION_APPLY_SETTINGS",
    "CORRECT_ILLUMINATION_CALCULATE_SETTINGS",
    "CPPipeParser",
    "ArtifactSettingClassifier",
    "ArtifactSettingDirection",
    "ArtifactSettingRole",
    "ArtifactSettingSymbol",
    "CellProfilerModuleCategory",
    "CellProfilerModuleDimensionality",
    "CellProfilerModuleReference",
    "CellProfilerModuleRole",
    "CellProfilerModuleRoleSpec",
    "CellProfilerModuleSemantics",
    "CellProfilerWatershedRuntimeFamily",
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
    "CellProfilerStructuringElement",
    "CellProfilerMeasurementTargetScope",
    "DEFAULT_STRUCTURING_ELEMENT_SETTING",
    "CellProfilerDialectCompiler",
    "FunctionSpecialOutput",
    "INFRASTRUCTURE_MODULE_NAMES",
    "INFRASTRUCTURE_MODULE_NAMES_BY_KEY",
    "IlluminationCalculationScope",
    "IlluminationCorrectionMethod",
    "IlluminationFilterSizeMethod",
    "IlluminationIntensityChoice",
    "IlluminationRescaleOption",
    "IlluminationSmoothingMethod",
    "IlluminationSplineBackgroundMode",
    "ImageQualityThresholdMethod",
    "IntensityDistributionCenterChoice",
    "IntensityDistributionZernikeMode",
    "MASK_OBJECTS_SETTINGS",
    "MaskImageSource",
    "MaskObjectsNumberingChoice",
    "MaskObjectsOverlapHandling",
    "ModuleBlock",
    "ModuleRevisionRange",
    "ModuleRuntimeSemanticsBinding",
    "ModuleSetting",
    "RELATE_OBJECTS_DISTANCE_SETTING",
    "RELATE_OBJECTS_PER_PARENT_MEANS_SETTING",
    "RELATE_OBJECTS_SAVE_CHILDREN_SETTING",
    "RelateObjectsDistanceMethod",
    "RESIZE_FACTOR_SETTING",
    "RESIZE_FACTOR_X_SETTING",
    "RESIZE_FACTOR_Y_SETTING",
    "RESIZE_FACTOR_Z_SETTING",
    "RESIZE_HEIGHT_SETTING",
    "RESIZE_INTERPOLATION_SETTING",
    "RESIZE_METHOD_SETTING",
    "RESIZE_PLANES_SETTING",
    "RESIZE_WIDTH_SETTING",
    "RescaleIntensityAutomaticHigh",
    "RescaleIntensityAutomaticLow",
    "RescaleIntensityMethod",
    "ResizeInterpolationMethod",
    "ResizeMethod",
    "SettingNameFamily",
    "SettingToKeywordBinding",
    "SettingsBinder",
    "SetupModuleCompiler",
    "STRUCTURING_ELEMENT_SETTING_NAME",
    "STRAIGHTEN_WORMS_INPUT_IMAGE_SETTING",
    "STRAIGHTEN_WORMS_INPUT_OBJECTS_SETTING",
    "STRAIGHTEN_WORMS_OUTPUT_IMAGE_SETTING",
    "STRAIGHTEN_WORMS_OUTPUT_OBJECTS_SETTING",
    "StraightenWormsFlipMode",
    "StraightenWormsImageBinding",
    "StructuringElementSetting",
    "StructuringElementSettingBinding",
    "UNTANGLE_WORMS_INPUT_IMAGE_SETTING",
    "UNTANGLE_WORMS_NONOVERLAPPING_OBJECTS_SETTING",
    "UNTANGLE_WORMS_OVERLAPPING_OBJECTS_SETTING",
    "UNTANGLE_WORMS_TRAINING_FILE_NAME_SETTING",
    "UntangleWormsOverlapStyle",
    "WatershedRuntimeSemanticsBinding",
    "artifact_setting_symbols",
    "cellprofiler_runtime_equivalence_policy",
    "cellprofiler_module_role",
    "cellprofiler_module_semantics",
    "clear_cellprofiler_dialect_compiler",
    "compile_image_schema",
    "coerce_cellprofiler_measurement_target_scope",
    "function_special_outputs",
    "get_cellprofiler_dialect_compiler",
    "normalize_cellprofiler_setting_name",
    "optional_setting_value",
    "parse_intensity_distribution_center_choice",
    "parse_intensity_distribution_zernike_mode",
    "parse_relate_objects_distance_method",
    "register_cellprofiler_dialect_compiler",
    "required_setting_value",
    "resize_bound_kwargs",
    "setting_values",
    "straighten_worms_bound_kwargs",
    "straighten_worms_image_bindings",
    "straighten_worms_input_objects_name",
    "straighten_worms_output_objects_name",
    "structuring_element_bound_kwargs",
    "untangle_worms_bound_kwargs",
)
