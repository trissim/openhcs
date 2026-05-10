"""Module-level settings-to-kwargs translation for generated CellProfiler steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, final

from metaclass_registry import AutoRegisterMeta

from openhcs.interop.cellprofiler.artifact_semantics import artifact_setting_symbols
from openhcs.interop.cellprofiler.measurement_scope import (
    CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG,
)
from openhcs.processing.backends.cellprofiler.library import canonical_module_name
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.interop.cellprofiler.align_settings import align_bound_kwargs
from openhcs.interop.cellprofiler.area_occupied_settings import (
    area_occupied_bound_kwargs,
)
from .calculate_math_settings import calculate_math_bound_kwargs
from openhcs.interop.cellprofiler.classify_objects_settings import (
    classify_objects_bound_kwargs,
)
from openhcs.interop.cellprofiler.color_to_gray_settings import (
    color_to_gray_bound_kwargs,
)
from .crop_settings import crop_bound_kwargs
from .display_data_settings import display_data_on_image_bound_kwargs
from .enhance_edges_settings import ENHANCE_EDGES_SETTINGS
from openhcs.interop.cellprofiler.expand_or_shrink_settings import (
    expand_or_shrink_bound_kwargs,
)
from .filter_objects_settings import filter_objects_bound_kwargs
from openhcs.interop.cellprofiler.grid_settings import (
    define_grid_bound_kwargs,
    define_grid_invocation_options,
    identify_objects_in_grid_bound_kwargs,
)
from .gray_to_color_settings import (
    GRAY_TO_COLOR_CMYK_IMAGE_SETTINGS,
    GRAY_TO_COLOR_CMYK_WEIGHT_SETTINGS,
    GRAY_TO_COLOR_RGB_IMAGE_SETTINGS,
    GRAY_TO_COLOR_RGB_WEIGHT_SETTINGS,
    GRAY_TO_COLOR_RESCALE_SETTING,
    GrayToColorScheme,
    coerce_gray_to_color_scheme,
    gray_to_color_rescale_default,
    gray_to_color_stack_channels,
    is_blank_gray_to_color_source,
)
from openhcs.interop.cellprofiler.illumination_settings import (
    CORRECT_ILLUMINATION_APPLY_SETTINGS,
    CORRECT_ILLUMINATION_CALCULATE_SETTINGS,
    IlluminationCorrectionMethod,
)
from openhcs.interop.cellprofiler.intensity_distribution_settings import (
    parse_intensity_distribution_center_choice,
    parse_intensity_distribution_zernike_mode,
)
from openhcs.interop.cellprofiler.image_module_settings import (
    CombineObjectsMethod,
    ConvertObjectsToImageMode,
    ImageQualityThresholdMethod,
    MaskImageSource,
    RescaleIntensityAutomaticHigh,
    RescaleIntensityAutomaticLow,
    RescaleIntensityMethod,
    WatershedDeclumpMethod,
    WatershedMethod,
)
from openhcs.interop.cellprofiler.image_math_settings import image_math_bound_kwargs
from openhcs.interop.cellprofiler.mask_objects_settings import MASK_OBJECTS_SETTINGS
from openhcs.interop.cellprofiler.module_function_resolution import (
    measurement_target_scope,
)
from openhcs.interop.cellprofiler.module_runtime_semantics import (
    ModuleRuntimeSemanticsBinding,
)
from openhcs.interop.cellprofiler.overlay_outlines_settings import (
    overlay_outlines_bound_kwargs,
)
from .parser import ModuleBlock
from openhcs.interop.cellprofiler.resize_objects_settings import (
    resize_objects_bound_kwargs,
)
from openhcs.interop.cellprofiler.resize_settings import resize_bound_kwargs
from openhcs.interop.cellprofiler.relate_objects_settings import (
    RELATE_OBJECTS_CHILD_OBJECTS_SETTING,
    RELATE_OBJECTS_DISTANCE_SETTING,
    RELATE_OBJECTS_PARENT_OBJECTS_SETTING,
    RELATE_OBJECTS_PER_PARENT_MEANS_SETTING,
    RELATE_OBJECTS_SAVE_CHILDREN_SETTING,
    parse_relate_objects_distance_method,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from .settings_binder import (
    cellprofiler_enum_value_setting_parser,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
    SettingToKeywordBinding,
    SettingsBinder,
    normalize_cellprofiler_setting_name,
)
from .symbol_table import (
    IDENTIFY_PRIMARY_OUTPUT_OBJECTS_SETTING,
    INPUT_IMAGE_SETTING,
    INPUT_OBJECTS_SETTING,
    OUTPUT_IMAGE_SETTING,
    OUTPUT_OBJECTS_SETTING,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    is_blank_symbol_name,
    optional_setting_value,
    setting_name_matches,
    setting_names,
    setting_values,
)
from .smooth_settings import SMOOTH_SETTINGS
from openhcs.interop.cellprofiler.straighten_worms_settings import (
    straighten_worms_bound_kwargs,
)
from openhcs.interop.cellprofiler.structuring_element_settings import (
    StructuringElementSettingBinding,
    structuring_element_bound_kwargs,
)
from openhcs.interop.cellprofiler.untangle_worms_settings import (
    untangle_worms_bound_kwargs,
)
from .unmix_colors_settings import unmix_colors_bound_kwargs
from .watershed_settings import (
    WATERSHED_BORDER_EXCLUSION_SETTING,
    WATERSHED_COMPACTNESS_SETTING,
    WATERSHED_CONNECTIVITY_SETTING,
    WATERSHED_DECLUMP_METHOD_SETTING,
    WATERSHED_DOWNSAMPLE_SETTING,
    WATERSHED_FOOTPRINT_SETTING,
    WATERSHED_INTENSITY_IMAGE_SETTING,
    WATERSHED_LABEL_SEPARATION_SETTING,
    WATERSHED_MAX_SEEDS_SETTING,
    WATERSHED_MARKERS_SETTING,
    WATERSHED_METHOD_SETTING,
    WATERSHED_MINIMUM_INTERNAL_DISTANCE_SETTING,
    WATERSHED_MINIMUM_SEED_DISTANCE_SETTING,
    WATERSHED_MASK_SETTING,
    WATERSHED_SMOOTHING_FACTOR_SETTING,
    WATERSHED_STRUCTURING_ELEMENT_SETTING,
    WATERSHED_USE_ADVANCED_SETTINGS_SETTING,
    parse_watershed_border_exclusion,
)


@dataclass(frozen=True, slots=True)
class BoundModuleSettings:
    """Typed module-setting translation result."""

    kwargs: Mapping[str, Any]
    unmapped_kwargs: Mapping[str, Any] = field(default_factory=dict)
    invocation_options: RuntimeInvocationOptions | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "unmapped_kwargs", dict(self.unmapped_kwargs))
        if (
            self.invocation_options is not None
            and not isinstance(self.invocation_options, RuntimeInvocationOptions)
        ):
            raise TypeError(
                "BoundModuleSettings.invocation_options must inherit "
                "RuntimeInvocationOptions."
            )


@dataclass(frozen=True, slots=True)
class UnmappedModuleSetting:
    """A CellProfiler setting that no registered binding strategy consumed."""

    module_name: str
    module_num: int
    setting_name: str
    value: Any


class UnmappedModuleSettingsError(ValueError):
    """Raised when enabled module settings are not mapped or explicitly ignored."""

    def __init__(self, settings: tuple[UnmappedModuleSetting, ...]) -> None:
        self.settings = settings
        rendered = "; ".join(
            f"{setting.module_name}({setting.module_num})."
            f"{setting.setting_name}={setting.value!r}"
            for setting in settings
        )
        super().__init__(
            "Enabled CellProfiler modules have unmapped settings. "
            "Add a ModuleSettingsBindingStrategy hook or an explicit typed ignore: "
            f"{rendered}"
        )


class ModuleUnmappedSettingIgnore(ABC, metaclass=AutoRegisterMeta):
    """Auto-registered typed ignore list for semantically dead CP settings."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True

    module_name: ClassVar[str | None] = None
    ignored_settings: ClassVar[tuple[str | SettingNameFamily, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        module_name = cls.__dict__.get("module_name")
        if isinstance(module_name, str):
            cls.module_name = canonical_module_name(module_name)

    @classmethod
    def ignored_setting_names_for(cls, module_name: str) -> frozenset[str]:
        ignore_type = cls.__registry__.get(canonical_module_name(module_name))
        if ignore_type is None:
            return frozenset()
        return ignore_type._normalized_setting_names(ignore_type.ignored_settings)

    @classmethod
    def ignored_setting_names_for_module(cls, module: ModuleBlock) -> frozenset[str]:
        ignore_type = cls.__registry__.get(canonical_module_name(module.name))
        if ignore_type is None:
            return frozenset()
        return ignore_type._normalized_setting_names(
            ignore_type.ignored_settings_for(module)
        )

    @classmethod
    def ignored_settings_for(
        cls,
        module: ModuleBlock,
    ) -> tuple[str | SettingNameFamily, ...]:
        return cls.ignored_settings

    @classmethod
    def _normalized_setting_names(
        cls,
        settings: tuple[str | SettingNameFamily, ...],
    ) -> frozenset[str]:
        return frozenset(
            normalize_cellprofiler_setting_name(concrete_name)
            for setting_name in settings
            for concrete_name in setting_names(setting_name)
        )


class ConditionalModuleUnmappedSettingIgnore(ModuleUnmappedSettingIgnore):
    """Nominal policy for CP settings inactive under a controlling UI choice."""

    controlling_setting: ClassVar[str | SettingNameFamily]
    inactive_when_values: ClassVar[tuple[str, ...]]
    inactive_settings: ClassVar[tuple[str | SettingNameFamily, ...]]

    @classmethod
    def ignored_settings_for(
        cls,
        module: ModuleBlock,
    ) -> tuple[str | SettingNameFamily, ...]:
        value = cls.controlling_setting_value(module)
        if value is not None and cls.setting_value_is_inactive(value):
            return (*cls.ignored_settings, *cls.inactive_settings)
        return cls.ignored_settings

    @classmethod
    def controlling_setting_value(cls, module: ModuleBlock) -> str | None:
        """Return the controlling setting value used by this inactive policy."""
        return optional_setting_value(module, cls.controlling_setting)

    @classmethod
    def setting_value_is_inactive(cls, value: str) -> bool:
        return value in cls.inactive_when_values


class BlankSymbolModuleUnmappedSettingIgnore(ConditionalModuleUnmappedSettingIgnore):
    """Nominal policy for CP settings inactive when their selector is blank."""

    inactive_when_values = ()

    @classmethod
    def controlling_setting_value(cls, module: ModuleBlock) -> str | None:
        """Return blank values too, because blank is the inactive signal."""
        for setting in module.iter_settings():
            if setting_name_matches(setting.name, cls.controlling_setting):
                return setting.value.strip()
        for setting_name, value in module.settings.items():
            if setting_name_matches(setting_name, cls.controlling_setting):
                return value.strip()
        return None

    @classmethod
    def setting_value_is_inactive(cls, value: str) -> bool:
        return not value.strip() or is_blank_symbol_name(value)


class CorrectIlluminationCalculateUnmappedSettingIgnore(ModuleUnmappedSettingIgnore):
    """CP UI output toggles ignored when averaged/dilated images are disabled."""

    module_name = "CorrectIlluminationCalculate"
    ignored_settings = (
        "Retain the averaged image?",
        "Name the averaged image",
        "Retain the dilated image?",
        "Name the dilated image",
    )


class IdentifyPrimaryObjectsUnmappedSettingIgnore(ModuleUnmappedSettingIgnore):
    """Display-only maxima visualization settings are not runtime semantics."""

    module_name = "IdentifyPrimaryObjects"
    ignored_settings = (
        "Display accepted local maxima?",
        "Select maxima color",
    )


class MeasureObjectIntensityUnmappedSettingIgnore(ModuleUnmappedSettingIgnore):
    """Legacy hidden group-count setting is parser metadata, not runtime input."""

    module_name = "MeasureObjectIntensity"
    ignored_settings = ("Hidden",)


class MeasureImageIntensityUnmappedSettingIgnore(BlankSymbolModuleUnmappedSettingIgnore):
    """Empty object-set selector is consumed by symbol-table scope selection."""

    module_name = "MeasureImageIntensity"
    controlling_setting = INPUT_OBJECTS_SETTING
    ignored_settings = (
        "Measure the intensity only from areas enclosed by objects?",
        "calculate_custom_percentiles",
        "specify_percentiles_to_measure",
    )
    inactive_settings = (INPUT_OBJECTS_SETTING,)


class MeasureColocalizationUnmappedSettingIgnore(
    BlankSymbolModuleUnmappedSettingIgnore
):
    """Empty object selector is consumed by colocalization scope binding."""

    module_name = "MeasureColocalization"
    controlling_setting = SettingNameFamily("Select an object to measure")
    ignored_settings = ("Select objects to measure", "Hidden")
    inactive_settings = (SettingNameFamily("Select an object to measure"),)


class MeasureGranularityUnmappedSettingIgnore(ModuleUnmappedSettingIgnore):
    """Granularity object mask routing is consumed by measurement scope binding."""

    module_name = "MeasureGranularity"
    ignored_settings = (
        "Measure within objects?",
        "image_count",
        "object_count",
    )


class MeasureImageQualityUnmappedSettingIgnore(ConditionalModuleUnmappedSettingIgnore):
    """Image selector is inactive when image quality measures all loaded images."""

    module_name = "MeasureImageQuality"
    controlling_setting = "Calculate metrics for which images?"
    inactive_when_values = ("All loaded images",)
    inactive_settings = ("Select the images to measure",)


class RelateObjectsUnmappedSettingIgnore(ModuleUnmappedSettingIgnore):
    """Object routing and disabled relationship outputs are contract semantics."""

    module_name = "RelateObjects"
    ignored_settings = (
        RELATE_OBJECTS_PARENT_OBJECTS_SETTING,
        RELATE_OBJECTS_CHILD_OBJECTS_SETTING,
        RELATE_OBJECTS_PER_PARENT_MEANS_SETTING,
        "Calculate distances to other parents?",
        "Parent name",
        RELATE_OBJECTS_SAVE_CHILDREN_SETTING,
        "Name the output object",
    )


class MaskObjectsUnmappedSettingIgnore(ConditionalModuleUnmappedSettingIgnore):
    """MaskObjects object/image routing is consumed by the symbol table."""

    module_name = "MaskObjects"
    controlling_setting = "Retain outlines of the resulting objects?"
    inactive_when_values = ("No",)
    ignored_settings = (
        "Mask using a region defined by other objects or by binary image",
        "Select the masking image",
        "Retain outlines of the resulting objects?",
    )
    inactive_settings = ("Name the outline image",)


class MeasureTextureUnmappedSettingIgnore(ModuleUnmappedSettingIgnore):
    """Legacy hidden group-count setting is parser metadata, not runtime input."""

    module_name = "MeasureTexture"
    ignored_settings = (
        "Hidden",
        "Angles to measure",
        "Measure Gabor features?",
        "Number of angles to compute for Gabor",
    )


class EnhanceOrSuppressFeaturesUnmappedSettingIgnore(ModuleUnmappedSettingIgnore):
    """Output display scaling does not affect absorbed numeric feature output."""

    module_name = "EnhanceOrSuppressFeatures"
    ignored_settings = ("Rescale result image",)


class TrackObjectsUnmappedSettingIgnore(ModuleUnmappedSettingIgnore):
    """Unsupported LAP/display knobs are intentionally outside overlap tracking."""

    module_name = "TrackObjects"
    ignored_settings = (
        "Average cell diameter in pixels",
        "Cost of cell to empty matching",
        "Filter objects by lifetime?",
        "Filter using a maximum lifetime?",
        "Filter using a minimum lifetime?",
        "Gap closing cost",
        "Maximum gap displacement in pixel units",
        "Maximum lifetime",
        "Maximum merge score",
        "Maximum mitosis distance in pixel units",
        "Maximum split score",
        "Maximum temporal gap in frames",
        "Merge alternative cost",
        "Minimum lifetime",
        "Mitosis alternative cost",
        "Number of standard deviations for search radius",
        "Run the second phase of the LAP algorithm?",
        "Save color-coded image?",
        "Search radius limit, in pixel units",
        "Select display option",
        "Select object measurement to use for tracking",
        "Select the movement model",
        "Split alternative cost",
        "Use advanced configuration parameters",
        "Weight of area difference in function matching cost",
    )


class RepeatedSettingValuePolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal resolver for CellProfiler settings that reuse the same label."""

    __registry_key__ = "policy_key"
    __skip_if_no_key__ = True
    setting_name: ClassVar[str | None] = None
    policy_key: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("policy_key") is not None:
            return
        setting_name = cls.__dict__.get("setting_name")
        if isinstance(setting_name, str):
            cls.policy_key = normalize_cellprofiler_setting_name(setting_name)

    @classmethod
    def for_setting(
        cls,
        setting_name: str,
    ) -> "RepeatedSettingValuePolicy":
        strategy_type = cls.__registry__.get(
            normalize_cellprofiler_setting_name(setting_name),
            LastRepeatedSettingValuePolicy,
        )
        return strategy_type()

    @final
    def value(
        self,
        module: ModuleBlock,
        setting_name: str | SettingNameFamily,
    ) -> str | None:
        values = setting_values(module, setting_name)
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return self._resolve_repeated_value(module, setting_name, tuple(values))

    @abstractmethod
    def _resolve_repeated_value(
        self,
        module: ModuleBlock,
        setting_name: str | SettingNameFamily,
        values: tuple[str, ...],
    ) -> str:
        """Return the semantically active value for a repeated setting label."""


class LastRepeatedSettingValuePolicy(RepeatedSettingValuePolicy):
    """Default CellProfiler scalar behavior: the later row is authoritative."""

    def _resolve_repeated_value(
        self,
        module: ModuleBlock,
        setting_name: str | SettingNameFamily,
        values: tuple[str, ...],
    ) -> str:
        return values[-1]


class ModuleSettingsBindingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for converting one module's settings into function kwargs."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        module_name = cls.__dict__.get("module_name")
        if isinstance(module_name, str):
            cls.module_name = canonical_module_name(module_name)

    @classmethod
    def for_module(cls, module_name: str) -> "ModuleSettingsBindingStrategy":
        strategy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            GenericModuleSettingsBindingStrategy,
        )
        return strategy_type()

    @final
    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> BoundModuleSettings:
        """Bind and require complete setting coverage for a live module."""
        bound = self._bind(module, binder=binder, param_mapping=param_mapping)
        runtime_semantics = ModuleRuntimeSemanticsBinding.for_module(module.name)
        if runtime_semantics is not None:
            bound = BoundModuleSettings(
                {**bound.kwargs, **runtime_semantics.kwargs(module)},
                bound.unmapped_kwargs,
                bound.invocation_options,
            )
        unmapped_kwargs = {
            setting_name: value
            for setting_name, value in bound.unmapped_kwargs.items()
            if setting_name not in ignored_unmapped_settings
            and setting_name not in self._artifact_setting_names(module)
            and setting_name
            not in ModuleUnmappedSettingIgnore.ignored_setting_names_for_module(module)
        }
        self._validate_mapped_module_settings(module, unmapped_kwargs)
        if len(unmapped_kwargs) != len(bound.unmapped_kwargs):
            return BoundModuleSettings(
                bound.kwargs,
                unmapped_kwargs,
                bound.invocation_options,
            )
        return bound

    @abstractmethod
    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        """Bind one parsed module into generated function kwargs."""

    @classmethod
    def _validate_mapped_module_settings(
        cls,
        module: ModuleBlock,
        unmapped_kwargs: Mapping[str, Any],
    ) -> None:
        if not unmapped_kwargs:
            return
        raise UnmappedModuleSettingsError(
            tuple(
                UnmappedModuleSetting(
                    module_name=module.name,
                    module_num=module.module_num,
                    setting_name=setting_name,
                    value=value,
                )
                for setting_name, value in sorted(unmapped_kwargs.items())
            )
        )

    @classmethod
    def _artifact_setting_names(cls, module: ModuleBlock) -> frozenset[str]:
        """Return settings consumed by the symbol-table artifact boundary."""
        return frozenset(
            normalize_cellprofiler_setting_name(symbol.setting_name)
            for symbol in artifact_setting_symbols(module)
        )


class GenericModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Default docstring-mapped module-setting binder."""

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        return _translate_bound_kwargs(
            binder.bind(module.settings),
            param_mapping,
        )


_CELLPROFILER_THRESHOLD_SETTINGS: Mapping[str, str] = {
    "Threshold strategy": "threshold_scope",
    "Thresholding method": "threshold_method",
    "Threshold smoothing scale": "threshold_smoothing_scale",
    "Threshold correction factor": "threshold_correction_factor",
    "Two-class or three-class thresholding?": "otsu_class_count",
    "Assign pixels in the middle intensity class to the foreground or the background?": (
        "assign_middle_to_foreground"
    ),
    "Log transform before thresholding?": "log_transform",
    "Size of adaptive window": "adaptive_window_size",
    "Lower outlier fraction": "lower_outlier_fraction",
    "Upper outlier fraction": "upper_outlier_fraction",
    "Averaging method": "averaging_method",
    "Variance method": "variance_method",
    "# of deviations": "number_of_deviations",
    "Manual threshold": "manual_threshold",
}
_IGNORED_CELLPROFILER_THRESHOLD_SETTINGS: tuple[str, ...] = (
    "Threshold setting version",
    "Select the measurement to threshold with",
)
_CELLPROFILER_THRESHOLD_SETTING_VERSION = "Threshold setting version"
_CELLPROFILER_THRESHOLD_METHOD_SETTING = "Thresholding method"
_CELLPROFILER_OTSU_METHOD = "otsu"
_CELLPROFILER_THREE_CLASS_OTSU = "three classes"


class CellProfilerThresholdScope(Enum):
    """Serialized threshold strategy names used to select active method rows."""

    GLOBAL = "global"
    ADAPTIVE = "adaptive"


class ThresholdMethodRepeatedSettingValuePolicy(RepeatedSettingValuePolicy):
    """Resolve CP's global/local threshold method rows from threshold scope."""

    setting_name = _CELLPROFILER_THRESHOLD_METHOD_SETTING

    def _resolve_repeated_value(
        self,
        module: ModuleBlock,
        setting_name: str | SettingNameFamily,
        values: tuple[str, ...],
    ) -> str:
        scope = _threshold_scope(module)
        if scope is CellProfilerThresholdScope.GLOBAL:
            return values[0]
        if scope is CellProfilerThresholdScope.ADAPTIVE:
            return values[-1]
        raise ValueError(
            f"{module.name}({module.module_num}) has repeated "
            f"{setting_name!r} rows but no supported threshold strategy."
        )


_LEGACY_CELLPROFILER_THRESHOLD_METHOD_NAMES: Mapping[str, str] = {
    "robustbackground": "Robust Background",
    "minimum cross entropy": "Minimum Cross-Entropy",
}
_FLOAT_CELLPROFILER_THRESHOLD_SETTINGS: frozenset[str] = frozenset(
    {
        "Threshold smoothing scale",
        "Threshold correction factor",
        "Lower outlier fraction",
        "Upper outlier fraction",
        "# of deviations",
        "Manual threshold",
    }
)
_INT_CELLPROFILER_THRESHOLD_SETTINGS: frozenset[str] = frozenset(
    {
        "Size of adaptive window",
    }
)
_BOOL_CELLPROFILER_THRESHOLD_SETTINGS: frozenset[str] = frozenset(
    {
        "Log transform before thresholding?",
    }
)


def _parse_cellprofiler_threshold_setting(
    binder: SettingsBinder,
    setting_name: str,
    value: str,
) -> Any:
    """Parse threshold settings by semantic field, not generic literal shape."""
    if setting_name in _FLOAT_CELLPROFILER_THRESHOLD_SETTINGS:
        return parse_cellprofiler_float(value)
    if setting_name in _INT_CELLPROFILER_THRESHOLD_SETTINGS:
        return parse_cellprofiler_int(value)
    if setting_name in _BOOL_CELLPROFILER_THRESHOLD_SETTINGS:
        return parse_cellprofiler_bool(value)
    return binder.parse_value(setting_name, value)


def _last_optional_setting_value(
    module: ModuleBlock,
    setting_name: str | SettingNameFamily,
) -> str | None:
    """Return the last ordered value for legacy scalar settings."""
    return LastRepeatedSettingValuePolicy().value(module, setting_name)


def _active_setting_value(
    module: ModuleBlock,
    setting_name: str,
) -> str | None:
    """Return the active ordered value through registered repeated-row policy."""
    return RepeatedSettingValuePolicy.for_setting(setting_name).value(
        module,
        setting_name,
    )


def _threshold_scope(module: ModuleBlock) -> CellProfilerThresholdScope | None:
    value = _last_optional_setting_value(module, "Threshold strategy")
    token = _cellprofiler_threshold_setting_token(value or "")
    for scope in CellProfilerThresholdScope:
        if token == scope.value:
            return scope
    return None


def _active_threshold_setting_value(
    module: ModuleBlock,
    setting_name: str,
) -> str | None:
    """Return the active ordered value for threshold settings.

    CellProfiler threshold settings can carry both the global and adaptive
    threshold method rows in one module block. The active method is selected by
    the threshold strategy: global uses the first method row, adaptive uses the
    last method row.
    """
    return _active_setting_value(module, setting_name)


def _cellprofiler_threshold_setting_token(value: Any) -> str:
    """Return a stable comparison token for parsed CellProfiler settings."""
    if isinstance(value, Enum) and isinstance(value.value, str):
        value = value.value
    return " ".join(str(value).strip().lower().replace("-", " ").split())


def _legacy_cellprofiler_threshold_version(module: ModuleBlock) -> int | None:
    value = optional_setting_value(module, _CELLPROFILER_THRESHOLD_SETTING_VERSION)
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _legacy_cellprofiler_log_transform_default(
    module: ModuleBlock,
    kwargs: Mapping[str, Any],
) -> bool | None:
    """Return CP's synthesized log-transform default for old threshold blocks.

    Native CellProfiler inserts ``Log transform before thresholding?`` while
    upgrading threshold setting version 10 to version 11. The inserted value is
    true only for three-class Otsu thresholding.
    """
    version = _legacy_cellprofiler_threshold_version(module)
    if version is None or version > 10:
        return None
    threshold_method = _cellprofiler_threshold_setting_token(
        kwargs.get("threshold_method", "")
    )
    otsu_class_count = _cellprofiler_threshold_setting_token(
        kwargs.get("otsu_class_count", "")
    )
    return (
        threshold_method == _CELLPROFILER_OTSU_METHOD
        and otsu_class_count == _CELLPROFILER_THREE_CLASS_OTSU
    )


def _upgrade_legacy_cellprofiler_threshold_kwargs(
    module: ModuleBlock,
    kwargs: dict[str, Any],
) -> None:
    version = _legacy_cellprofiler_threshold_version(module)
    if version is None or version > 10:
        return

    threshold_method = kwargs.get("threshold_method")
    if threshold_method is not None:
        method_token = _cellprofiler_threshold_setting_token(threshold_method)
        kwargs["threshold_method"] = _LEGACY_CELLPROFILER_THRESHOLD_METHOD_NAMES.get(
            method_token,
            threshold_method,
        )

    if "log_transform" not in kwargs:
        log_transform_default = _legacy_cellprofiler_log_transform_default(
            module,
            kwargs,
        )
        if log_transform_default is not None:
            kwargs["log_transform"] = log_transform_default


def _bind_cellprofiler_threshold_settings(
    *,
    module: ModuleBlock,
    binder: SettingsBinder,
    kwargs: dict[str, Any],
    unmapped_kwargs: dict[str, Any],
    include_advanced_setting: bool,
) -> None:
    if include_advanced_setting:
        value = _last_optional_setting_value(module, "Use advanced settings?")
        if value is not None:
            kwargs["use_advanced_settings"] = binder.parse_value(
                "Use advanced settings?",
                value,
            )
        unmapped_kwargs.pop(
            normalize_cellprofiler_setting_name("Use advanced settings?"),
            None,
        )

    for setting_name, parameter_name in _CELLPROFILER_THRESHOLD_SETTINGS.items():
        value = _active_threshold_setting_value(module, setting_name)
        if value is not None:
            kwargs[parameter_name] = _parse_cellprofiler_threshold_setting(
                binder,
                setting_name,
                value,
            )
        unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

    _upgrade_legacy_cellprofiler_threshold_kwargs(module, kwargs)

    bounds = _last_optional_setting_value(module, "Lower and upper bounds on threshold")
    if bounds is not None:
        parsed_bounds = binder.parse_value(
            "Lower and upper bounds on threshold",
            bounds,
        )
        if not isinstance(parsed_bounds, tuple) or len(parsed_bounds) != 2:
            raise ValueError(
                f"{module.name} threshold bounds must contain two values, "
                f"got {bounds!r}."
            )
        kwargs["threshold_min"], kwargs["threshold_max"] = parsed_bounds
    unmapped_kwargs.pop(
        normalize_cellprofiler_setting_name("Lower and upper bounds on threshold"),
        None,
    )

    for setting_name in _IGNORED_CELLPROFILER_THRESHOLD_SETTINGS:
        unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)


class IdentifyPrimaryObjectsModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind primary-object threshold settings using ordered CellProfiler settings."""

    module_name = "IdentifyPrimaryObjects"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        _bind_cellprofiler_threshold_settings(
            module=module,
            binder=binder,
            kwargs=kwargs,
            unmapped_kwargs=unmapped_kwargs,
            include_advanced_setting=True,
        )
        for setting_name in (
            *setting_names(INPUT_IMAGE_SETTING),
            *setting_names(IDENTIFY_PRIMARY_OUTPUT_OBJECTS_SETTING),
        ):
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class ThresholdModuleSettingsBindingStrategy(GenericModuleSettingsBindingStrategy):
    """Bind standalone Threshold settings through shared threshold semantics."""

    module_name = "Threshold"
    parameter_aliases: ClassVar[Mapping[str, str]] = {
        "threshold_smoothing_scale": "smoothing",
        "adaptive_window_size": "window_size",
    }
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Select the input image",
        "Name the output image",
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        _bind_cellprofiler_threshold_settings(
            module=module,
            binder=binder,
            kwargs=kwargs,
            unmapped_kwargs=unmapped_kwargs,
            include_advanced_setting=False,
        )
        for source_name, target_name in type(self).parameter_aliases.items():
            if source_name in kwargs:
                kwargs[target_name] = kwargs.pop(source_name)
        manual_threshold = kwargs.pop("manual_threshold", None)
        if (
            manual_threshold is not None
            and _cellprofiler_threshold_setting_token(kwargs.get("threshold_method", ""))
            == "manual"
        ):
            kwargs["predefined_threshold"] = manual_threshold
        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)
        return BoundModuleSettings(kwargs, unmapped_kwargs)


class RescaleIntensityModuleSettingsBindingStrategy(GenericModuleSettingsBindingStrategy):
    """Bind RescaleIntensity settings to the absorbed function's nominal enums."""

    module_name = "RescaleIntensity"
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Select the input image",
        "Name the output image",
        "Select image to match in maximum intensity",
        "Divisor measurement",
    )
    explicit_settings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        SettingToKeywordBinding(
            "Rescaling method",
            "rescale_method",
            cellprofiler_enum_value_setting_parser(RescaleIntensityMethod),
        ),
        SettingToKeywordBinding(
            "Method to calculate the minimum intensity",
            "automatic_low",
            cellprofiler_enum_value_setting_parser(RescaleIntensityAutomaticLow),
        ),
        SettingToKeywordBinding(
            "Method to calculate the maximum intensity",
            "automatic_high",
            cellprofiler_enum_value_setting_parser(RescaleIntensityAutomaticHigh),
        ),
        SettingToKeywordBinding(
            "Lower intensity limit for the input image",
            "source_low",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            "Upper intensity limit for the input image",
            "source_high",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            "Divisor value",
            "divisor_value",
            parse_cellprofiler_float,
        ),
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        kwargs.update(binder.bind_declared(module, type(self).explicit_settings))
        input_range = optional_setting_value(module, "Intensity range for the input image")
        if input_range is not None:
            parsed_input_range = binder.parse_value(
                "Intensity range for the input image",
                input_range,
            )
            if not isinstance(parsed_input_range, tuple) or len(parsed_input_range) != 2:
                raise ValueError(
                    f"{module.name} input intensity range must contain two values, "
                    f"got {input_range!r}."
                )
            kwargs["source_low"], kwargs["source_high"] = parsed_input_range
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(
                    "Intensity range for the input image"
                ),
                None,
            )
        output_range = optional_setting_value(module, "Intensity range for the output image")
        if output_range is not None:
            parsed_output_range = binder.parse_value(
                "Intensity range for the output image",
                output_range,
            )
            if not isinstance(parsed_output_range, tuple) or len(parsed_output_range) != 2:
                raise ValueError(
                    f"{module.name} output intensity range must contain two values, "
                    f"got {output_range!r}."
                )
            kwargs["dest_low"], kwargs["dest_high"] = parsed_output_range
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(
                    "Intensity range for the output image"
                ),
                None,
            )
        for binding in type(self).explicit_settings:
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(binding.setting_name),
                None,
            )
        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)
        return BoundModuleSettings(kwargs, unmapped_kwargs)


class MaskImageModuleSettingsBindingStrategy(GenericModuleSettingsBindingStrategy):
    """Bind MaskImage routing settings to mask semantics."""

    module_name = "MaskImage"
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Select the input image",
        "Name the output image",
        "Select object for mask",
        "Select image for mask",
    )
    explicit_settings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        SettingToKeywordBinding(
            "Use objects or an image as a mask?",
            "mask_source",
            cellprofiler_enum_value_setting_parser(MaskImageSource),
        ),
        SettingToKeywordBinding(
            "Invert the mask?",
            "invert_mask",
            parse_cellprofiler_bool,
        ),
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        kwargs.update(binder.bind_declared(module, type(self).explicit_settings))
        for binding in type(self).explicit_settings:
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(binding.setting_name),
                None,
            )
        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)
        return BoundModuleSettings(kwargs, unmapped_kwargs)


class EnhanceOrSuppressFeaturesModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind EnhanceOrSuppressFeatures settings to native library semantics."""

    module_name = "EnhanceOrSuppressFeatures"
    explicit_settings: ClassVar[Mapping[str, str]] = {
        "Select the operation": "method",
        "Feature type": "enhance_method",
        "Smoothing scale": "smoothing_value",
        "Shear angle": "dic_angle",
        "Decay": "dic_decay",
        "Enhancement method": "neurite_method",
        "Speed and accuracy": "speckle_accuracy",
    }
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Select the input image",
        "Name the output image",
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, parameter_name in type(self).explicit_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is not None:
                kwargs[parameter_name] = binder.parse_value(
                    setting_names(setting_name)[0],
                    value,
                )
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        hole_sizes = optional_setting_value(module, "Range of hole sizes")
        if hole_sizes is not None:
            parsed_hole_sizes = binder.parse_value("Range of hole sizes", hole_sizes)
            if (
                not isinstance(parsed_hole_sizes, tuple)
                or len(parsed_hole_sizes) != 2
            ):
                raise ValueError(
                    f"{module.name} hole size range must contain two values, "
                    f"got {hole_sizes!r}."
                )
            kwargs["dark_hole_radius_min"], kwargs["dark_hole_radius_max"] = (
                parsed_hole_sizes
            )
        unmapped_kwargs.pop(
            normalize_cellprofiler_setting_name("Range of hole sizes"),
            None,
        )

        feature_size = optional_setting_value(module, "Feature size")
        if feature_size is not None:
            kwargs["radius"] = binder.parse_value("Feature size", feature_size) / 2
        unmapped_kwargs.pop(
            normalize_cellprofiler_setting_name("Feature size"),
            None,
        )

        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class IdentifySecondaryObjectsModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind secondary-object method and shared threshold settings."""

    module_name = "IdentifySecondaryObjects"

    explicit_settings: ClassVar[Mapping[str, str]] = {
        "Select the method to identify the secondary objects": "method",
        "Number of pixels by which to expand the primary objects": (
            "distance_to_dilate"
        ),
        "Regularization factor": "regularization_factor",
        "Fill holes in identified objects?": "fill_holes",
        "Discard secondary objects touching the border of the image?": (
            "discard_edge_objects"
        ),
    }
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Select the input objects",
        "Name the objects to be identified",
        "Select the input image",
        "Discard the associated primary objects?",
        "Name the new primary objects",
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, parameter_name in type(self).explicit_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is not None:
                kwargs[parameter_name] = binder.parse_value(setting_name, value)
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        _bind_cellprofiler_threshold_settings(
            module=module,
            binder=binder,
            kwargs=kwargs,
            unmapped_kwargs=unmapped_kwargs,
            include_advanced_setting=False,
        )

        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class ScopedMeasurementModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind generic measurement-scope intent alongside normal kwargs."""

    scope_setting_name: ClassVar[SettingNameFamily | None] = None
    default_scope_value: ClassVar[str | None] = None

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        kwargs[CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG] = (
            measurement_target_scope(
                module,
                setting=_required_class_attr(
                    type(self).scope_setting_name,
                    "scope_setting_name",
                ),
                default=_required_class_attr(
                    type(self).default_scope_value,
                    "default_scope_value",
                ),
            ).value
        )
        for setting_name in setting_names(
            _required_class_attr(type(self).scope_setting_name, "scope_setting_name")
        ):
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)
        return BoundModuleSettings(kwargs, unmapped_kwargs)


class MeasureTextureModuleSettingsBindingStrategy(
    ScopedMeasurementModuleSettingsBindingStrategy
):
    """Bind MeasureTexture's image/object target scope."""

    module_name = "MeasureTexture"
    scope_setting_name = SettingNameFamily(
        "Measure images or objects?",
        aliases=("Measure whole images or objects?",),
    )
    default_scope_value = "Images"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        texture_scales = setting_values(module, "Texture scale to measure")
        if texture_scales:
            parsed_scales = tuple(
                parse_cellprofiler_int(value)
                for value in texture_scales
            )
            kwargs["scale"] = (
                parsed_scales[0] if len(parsed_scales) == 1 else parsed_scales
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name("Texture scale to measure"),
                None,
            )

        gray_levels = optional_setting_value(
            module,
            "Enter how many gray levels to measure the texture at",
        )
        if gray_levels is not None:
            kwargs["gray_levels"] = parse_cellprofiler_int(gray_levels)
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(
                    "Enter how many gray levels to measure the texture at"
                ),
                None,
            )

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class MeasureObjectSizeShapeModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind MeasureObjectSizeShape semantic toggles and routing-only settings."""

    module_name = "MeasureObjectSizeShape"
    explicit_settings: ClassVar[Mapping[str, str]] = {
        "Calculate the Zernike features?": "calculate_zernikes",
        "Calculate the advanced features?": "calculate_advanced",
    }
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Select objects to measure",
        "Select object sets to measure",
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, parameter_name in type(self).explicit_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is not None:
                kwargs[parameter_name] = parse_cellprofiler_bool(value)
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class MeasureObjectIntensityModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind MeasureObjectIntensity routing settings consumed by contracts."""

    module_name = "MeasureObjectIntensity"
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Select images to measure",
        "Select objects to measure",
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)
        return BoundModuleSettings(bound.kwargs, unmapped_kwargs)


class MeasureObjectIntensityDistributionModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind radial-distribution settings not named like function parameters."""

    module_name = "MeasureObjectIntensityDistribution"
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Hidden",
        "Select objects to use as centers",
    )
    zernike_setting_name: ClassVar[str] = "Calculate intensity Zernikes?"
    zernike_degree_setting_name: ClassVar[str | SettingNameFamily] = SettingNameFamily(
        "Maximum Zernike moment",
        aliases=("Maximum zernike moment",),
    )
    scalar_settings: ClassVar[Mapping[str, tuple[str, Any]]] = {
        "Scale the bins?": ("wants_scaled", parse_cellprofiler_bool),
        "Number of bins": ("bin_count", parse_cellprofiler_int),
        "Maximum radius": ("maximum_radius", parse_cellprofiler_int),
    }

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        zernike_value = optional_setting_value(module, type(self).zernike_setting_name)
        if zernike_value is not None:
            kwargs["wants_zernikes"] = parse_intensity_distribution_zernike_mode(
                zernike_value
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(type(self).zernike_setting_name),
                None,
            )

        zernike_degree = optional_setting_value(
            module,
            type(self).zernike_degree_setting_name,
        )
        if zernike_degree is not None:
            kwargs["zernike_degree"] = parse_cellprofiler_int(zernike_degree)
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(
                    setting_names(type(self).zernike_degree_setting_name)[0]
                ),
                None,
            )

        for setting_name, (parameter_name, parse) in type(self).scalar_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is None:
                continue
            kwargs[parameter_name] = parse(value)
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        center_choice = optional_setting_value(module, "Object to use as center?")
        if center_choice is not None:
            kwargs["center_choice"] = parse_intensity_distribution_center_choice(
                center_choice
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name("Object to use as center?"),
                None,
            )

        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class MeasureImageQualityModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind MeasureImageQuality settings to absorbed image-quality parameters."""

    module_name = "MeasureImageQuality"
    scalar_settings: ClassVar[Mapping[str, tuple[str, Any]]] = {
        "Calculate blur metrics?": ("calculate_blur", parse_cellprofiler_bool),
        "Calculate saturation metrics?": (
            "calculate_saturation",
            parse_cellprofiler_bool,
        ),
        "Calculate intensity metrics?": ("calculate_intensity", parse_cellprofiler_bool),
        "Calculate thresholds?": ("calculate_threshold", parse_cellprofiler_bool),
        "Spatial scale for blur measurements": ("blur_scale", parse_cellprofiler_int),
            "Select a thresholding method": (
                "threshold_method",
                cellprofiler_enum_value_setting_parser(ImageQualityThresholdMethod),
            ),
    }
    unsupported_settings: ClassVar[tuple[str, ...]] = (
        "Image count",
        "Calculate metrics for which images?",
        "Include the image rescaling value?",
        "Threshold count",
        "Two-class or three-class thresholding?",
        "Assign pixels in the middle intensity class to the foreground or the background?",
        "Minimize the weighted variance or the entropy?",
        "Typical fraction of the image covered by objects",
        "Use all thresholding methods?",
        "Scale count",
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, (parameter_name, parse) in type(self).scalar_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is None:
                continue
            kwargs[parameter_name] = parse(value)
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        for setting_name in type(self).unsupported_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)

class MeasureGranularityModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind granularity spectrum settings shared across image/object rows."""

    module_name = "MeasureGranularity"
    scalar_settings: ClassVar[Mapping[str, tuple[str, Any]]] = {
        "Subsampling factor for granularity measurements": (
            "subsample_size",
            parse_cellprofiler_float,
        ),
        "Subsampling factor for background reduction": (
            "background_subsample_size",
            parse_cellprofiler_float,
        ),
        "Radius of structuring element": (
            "element_radius",
            parse_cellprofiler_int,
        ),
        "Range of the granular spectrum": (
            "spectrum_length",
            parse_cellprofiler_int,
        ),
    }

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, (parameter_name, parse) in type(self).scalar_settings.items():
            values = setting_values(module, setting_name)
            if not values:
                continue
            parsed_values = tuple(parse(value) for value in values)
            first_value = parsed_values[0]
            if any(value != first_value for value in parsed_values[1:]):
                raise ValueError(
                    f"Module {module.name}({module.module_num}) has per-row "
                    f"{setting_name!r} values {parsed_values!r}; OpenHCS "
                    "currently binds one granularity setting set per module."
                )
            kwargs[parameter_name] = first_value
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class MeasureColocalizationModuleSettingsBindingStrategy(
    ScopedMeasurementModuleSettingsBindingStrategy
):
    """Bind MeasureColocalization's image/object target scope."""

    module_name = "MeasureColocalization"
    scope_setting_name = SettingNameFamily("Select where to measure correlation")
    default_scope_value = "Across entire image"
    metric_settings: ClassVar[Mapping[str | SettingNameFamily, str]] = {
        "Set threshold as percentage of maximum intensity for the images": (
            "threshold_percent"
        ),
        "Calculate correlation and slope metrics?": "do_correlation",
        "Calculate the Manders coefficients?": "do_manders",
        SettingNameFamily(
            "Calculate the Rank Weighted Colocalization coefficients?",
            aliases=("Calculate the Rank Weighted Coloalization coefficients?",),
        ): "do_rwc",
        "Calculate the Overlap coefficients?": "do_overlap",
        "Calculate the Manders coefficients using Costes auto threshold?": (
            "do_costes"
        ),
    }
    metric_flags: ClassVar[tuple[str, ...]] = (
        "do_correlation",
        "do_manders",
        "do_rwc",
        "do_overlap",
        "do_costes",
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, parameter_name in type(self).metric_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is not None:
                kwargs[parameter_name] = binder.parse_value(setting_name, value)
            for concrete_name in setting_names(setting_name):
                unmapped_kwargs.pop(
                    normalize_cellprofiler_setting_name(concrete_name),
                    None,
                )

        run_all_value = optional_setting_value(module, "Run all metrics?")
        if run_all_value is not None:
            if _parse_colocalization_run_all_metrics(run_all_value):
                kwargs.update({flag: True for flag in type(self).metric_flags})
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name("Run all metrics?"),
                None,
            )

        costes_method = optional_setting_value(module, "Method for Costes thresholding")
        if costes_method is not None:
            kwargs["costes_method"] = _parse_colocalization_costes_method(
                costes_method
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name("Method for Costes thresholding"),
                None,
            )

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class DeclarativeModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind modules described by explicit setting-to-kwarg declarations."""

    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = ()
    ignored_settings: ClassVar[tuple[str | SettingNameFamily, ...]] = ()

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        bound_details = binder.bind_with_details(module.settings)
        kwargs = binder.bind_declared(module, type(self).setting_bindings)
        mapped_settings = {
            normalize_cellprofiler_setting_name(setting_name)
            for binding in type(self).setting_bindings
            for setting_name in setting_names(binding.setting_name)
        }
        mapped_settings.update(
            normalize_cellprofiler_setting_name(concrete_setting_name)
            for setting_name in type(self).ignored_settings
            for concrete_setting_name in setting_names(setting_name)
        )
        unmapped_kwargs = {
            detail.name: detail.original_value
            for detail in bound_details
            if detail.name not in mapped_settings
        }
        return BoundModuleSettings(
            kwargs,
            unmapped_kwargs,
        )


class ConvertObjectsToImageModuleSettingsBindingStrategy(
    DeclarativeModuleSettingsBindingStrategy
):
    """Bind object-label rendering mode into the absorbed image renderer."""

    module_name = "ConvertObjectsToImage"
    setting_bindings = (
        SettingToKeywordBinding(
            "Select the color format",
            "image_mode",
            cellprofiler_enum_value_setting_parser(ConvertObjectsToImageMode),
        ),
        SettingToKeywordBinding("Select the colormap", "colormap_value"),
    )


class CombineObjectsModuleSettingsBindingStrategy(DeclarativeModuleSettingsBindingStrategy):
    """Bind object-overlap policy into CombineObjects' nominal method enum."""

    module_name = "CombineObjects"
    setting_bindings = (
        SettingToKeywordBinding(
            "Select how to handle overlapping objects",
            "method",
            cellprofiler_enum_value_setting_parser(CombineObjectsMethod),
        ),
    )


class WatershedModuleSettingsBindingStrategy(DeclarativeModuleSettingsBindingStrategy):
    """Bind Watershed settings that control object-domain semantics."""

    module_name = "Watershed"
    structuring_element_binding = StructuringElementSettingBinding(
        setting_name=WATERSHED_STRUCTURING_ELEMENT_SETTING,
        default_value="Disk,1",
        shape_keyword="structuring_element",
        size_keyword="structuring_element_size",
    )
    ignored_settings: ClassVar[tuple[str | SettingNameFamily, ...]] = (
        INPUT_IMAGE_SETTING,
        OUTPUT_OBJECTS_SETTING,
        WATERSHED_MARKERS_SETTING,
        WATERSHED_MASK_SETTING,
        WATERSHED_INTENSITY_IMAGE_SETTING,
        "Display watershed seeds?",
    )
    setting_bindings = (
        SettingToKeywordBinding(
            WATERSHED_USE_ADVANCED_SETTINGS_SETTING,
            "use_advanced_settings",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            WATERSHED_METHOD_SETTING,
            "watershed_method",
            cellprofiler_enum_value_setting_parser(WatershedMethod),
        ),
        SettingToKeywordBinding(
            WATERSHED_DECLUMP_METHOD_SETTING,
            "declump_method",
            cellprofiler_enum_value_setting_parser(WatershedDeclumpMethod),
        ),
        SettingToKeywordBinding(
            WATERSHED_CONNECTIVITY_SETTING, "connectivity", parse_cellprofiler_int
        ),
        SettingToKeywordBinding(
            WATERSHED_COMPACTNESS_SETTING, "compactness", parse_cellprofiler_float
        ),
        SettingToKeywordBinding(
            WATERSHED_FOOTPRINT_SETTING, "footprint", parse_cellprofiler_int
        ),
        SettingToKeywordBinding(
            WATERSHED_DOWNSAMPLE_SETTING, "downsample", parse_cellprofiler_int
        ),
        SettingToKeywordBinding(
            WATERSHED_LABEL_SEPARATION_SETTING,
            "watershed_line",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            WATERSHED_BORDER_EXCLUSION_SETTING,
            "exclude_border",
            parse_watershed_border_exclusion,
        ),
        SettingToKeywordBinding(
            WATERSHED_MINIMUM_SEED_DISTANCE_SETTING,
            "min_distance",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            WATERSHED_MINIMUM_INTERNAL_DISTANCE_SETTING,
            "min_intensity",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            WATERSHED_SMOOTHING_FACTOR_SETTING,
            "gaussian_sigma",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            WATERSHED_MAX_SEEDS_SETTING,
            "max_seeds",
            parse_cellprofiler_int,
        ),
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        kwargs.update(
            structuring_element_bound_kwargs(
                module,
                binder,
                type(self).structuring_element_binding,
            )
        )
        for setting_name in type(self).structuring_element_binding.normalized_setting_names:
            unmapped_kwargs.pop(setting_name, None)
        return BoundModuleSettings(kwargs, unmapped_kwargs)


class GrayToColorModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Scheme-aware binder for GrayToColor's closed family of input layouts."""

    module_name = "GrayToColor"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        scheme = coerce_gray_to_color_scheme(
            module.get_setting("Select a color scheme", GrayToColorScheme.RGB.value)
        )
        return GrayToColorSchemeBindingStrategy.for_scheme(scheme).bind(
            module,
            binder=binder,
        )


class UnmixColorsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind UnmixColors repeated output rows into one multi-output call."""

    module_name = "UnmixColors"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(unmix_colors_bound_kwargs(module))


class ColorToGrayModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ColorToGray's mode-dependent channel plan."""

    module_name = "ColorToGray"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(color_to_gray_bound_kwargs(module, binder))


class MeasureImageAreaOccupiedModuleSettingsBindingStrategy(
    ModuleSettingsBindingStrategy
):
    """Bind ordered area-occupied rows into one generic multi-row call."""

    module_name = "MeasureImageAreaOccupiedBinary"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(area_occupied_bound_kwargs(module))


class AlignModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind legacy Align settings into the absorbed registration function."""

    module_name = "Align"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(align_bound_kwargs(module))


class OverlayOutlinesModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ordered OverlayOutlines rows into one generic overlay call."""

    module_name = "OverlayOutlines"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(overlay_outlines_bound_kwargs(module))


class OverlayObjectsModuleSettingsBindingStrategy(GenericModuleSettingsBindingStrategy):
    """Bind OverlayObjects visual settings; artifact routing lives in contracts."""

    module_name = "OverlayObjects"
    explicit_settings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        SettingToKeywordBinding("Opacity", "opacity", parse_cellprofiler_float),
    )
    ignored_settings: ClassVar[tuple[str | SettingNameFamily, ...]] = (
        INPUT_IMAGE_SETTING,
        INPUT_OBJECTS_SETTING,
        OUTPUT_IMAGE_SETTING,
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        kwargs.update(binder.bind_declared(module, type(self).explicit_settings))
        for binding in type(self).explicit_settings:
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(binding.setting_name),
                None,
            )
        for setting_name in type(self).ignored_settings:
            for name in setting_names(setting_name):
                unmapped_kwargs.pop(normalize_cellprofiler_setting_name(name), None)
        return BoundModuleSettings(kwargs, unmapped_kwargs)


class TileModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind legacy Tile assembly modes into explicit montage geometry."""

    module_name = "Tile"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        assembly_method = module.get_setting(
            "Tile assembly method",
            "Within cycles",
        ).strip()
        normalized_method = normalize_cellprofiler_setting_name(assembly_method)
        if normalized_method != "within_cycles":
            raise NotImplementedError(
                "Tile assembly method is not supported by the converter: "
                f"{assembly_method!r}"
            )
        return BoundModuleSettings(
            {
                "rows": 1,
                "columns": 1,
                "place_first": "top_left",
                "tile_style": "row",
                "meander": False,
                "auto_rows": False,
                "auto_columns": True,
            }
        )


class TrackObjectsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind TrackObjects settings that define tracking identity and feature names."""

    module_name = "TrackObjects"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        generic = GenericModuleSettingsBindingStrategy()._bind(
            module,
            binder=binder,
            param_mapping=param_mapping,
        )
        method = module.get_setting("Choose a tracking method", "Overlap").strip()
        normalized_method = normalize_cellprofiler_setting_name(method)
        if normalized_method not in {"overlap", "distance"}:
            raise NotImplementedError(
                "TrackObjects tracking method is not supported by the converter: "
                f"{method!r}"
            )
        object_name = module.get_setting("Select the objects to track", "Objects").strip()
        if not object_name:
            raise ValueError("TrackObjects requires a non-empty tracked object name.")
        pixel_radius = parse_cellprofiler_int(
            module.get_setting("Maximum pixel distance to consider matches", "50")
        )
        unmapped_kwargs = dict(generic.unmapped_kwargs)
        for setting_name in (
            "Choose a tracking method",
            "Select the objects to track",
            "Maximum pixel distance to consider matches",
        ):
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)
        return BoundModuleSettings(
            {
                **dict(generic.kwargs),
                "object_name": object_name,
                "tracking_method": normalized_method,
                "pixel_radius": pixel_radius,
            },
            unmapped_kwargs,
        )


class ImageMathModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ImageMath operation and arithmetic modifiers."""

    module_name = "ImageMath"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(image_math_bound_kwargs(module, binder))


class ResizeModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind Resize's factor/size/interpolation settings."""

    module_name = "Resize"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(resize_bound_kwargs(module, binder))


class ResizeObjectsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ResizeObjects' factor/size settings."""

    module_name = "ResizeObjects"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(resize_objects_bound_kwargs(module, binder))


class MedianFilterModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind MedianFilter's spatial window setting."""

    module_name = "Medianfilter"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(
            binder.bind_declared(
                module,
                (
                    SettingToKeywordBinding("Window", "window_size", parse_cellprofiler_int),
                ),
            )
        )


class GaussianFilterModuleSettingsBindingStrategy(
    DeclarativeModuleSettingsBindingStrategy
):
    """Bind GaussianFilter's smoothing sigma setting."""

    module_name = "GaussianFilter"
    setting_bindings = (
        SettingToKeywordBinding("Sigma", "sigma", parse_cellprofiler_float),
    )


class RemoveHolesModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind RemoveHoles' diameter setting."""

    module_name = "RemoveHoles"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(
            binder.bind_declared(
                module,
                (
                    SettingToKeywordBinding(
                        "Size of holes to fill",
                        "diameter",
                        parse_cellprofiler_float,
                    ),
                ),
            )
        )


class MeasureObjectNeighborsModuleSettingsBindingStrategy(
    ModuleSettingsBindingStrategy
):
    """Bind retained neighbor-image settings alongside topology measurements."""

    module_name = "MeasureObjectNeighbors"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = GenericModuleSettingsBindingStrategy()._bind(
            module,
            binder=binder,
            param_mapping=param_mapping,
        )
        colormaps = module.get_setting_values("Select colormap")
        kwargs = dict(bound.kwargs)
        kwargs.update(
            {
                "retain_neighbor_count_image": parse_cellprofiler_bool(
                    module.get_setting(
                        "Retain the image of objects colored by numbers of neighbors?",
                        "No",
                    )
                ),
                "neighbor_count_colormap": colormaps[0] if colormaps else "Default",
                "retain_percent_touching_image": parse_cellprofiler_bool(
                    module.get_setting(
                        "Retain the image of objects colored by percent of touching pixels?",
                        "No",
                    )
                ),
                "percent_touching_colormap": (
                    colormaps[1]
                    if len(colormaps) > 1
                    else colormaps[0] if colormaps else "Default"
                ),
            }
        )
        unmapped = {
            name: value
            for name, value in bound.unmapped_kwargs.items()
            if name
            not in {
                "retain_the_image_of_objects_colored_by_numbers_of_neighbors",
                "retain_the_image_of_objects_colored_by_percent_of_touching_pixels",
                "name_the_output_image",
                "select_colormap",
            }
        }
        return BoundModuleSettings(kwargs, unmapped)


class FilterObjectsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind FilterObjects rows into one generic multi-output object filter."""

    module_name = "FilterObjects"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(filter_objects_bound_kwargs(module))


class RelateObjectsModuleSettingsBindingStrategy(DeclarativeModuleSettingsBindingStrategy):
    """Bind RelateObjects distance settings instead of using expensive defaults."""

    module_name = "RelateObjects"
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        SettingToKeywordBinding(
            RELATE_OBJECTS_DISTANCE_SETTING,
            "calculate_distances",
            parse_relate_objects_distance_method,
        ),
        SettingToKeywordBinding(
            RELATE_OBJECTS_PER_PARENT_MEANS_SETTING,
            "calculate_per_parent_means",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            RELATE_OBJECTS_SAVE_CHILDREN_SETTING,
            "save_children_with_parents",
            parse_cellprofiler_bool,
        ),
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        generic_bound = GenericModuleSettingsBindingStrategy._bind(
            self,
            module,
            binder=binder,
            param_mapping={},
        )
        declared_bound = DeclarativeModuleSettingsBindingStrategy._bind(
            self,
            module,
            binder=binder,
            param_mapping={},
        )
        return BoundModuleSettings(
            {**generic_bound.kwargs, **declared_bound.kwargs},
            {
                name: value
                for name, value in generic_bound.unmapped_kwargs.items()
                if name
                not in {
                    normalize_cellprofiler_setting_name(setting_name)
                    for binding in type(self).setting_bindings
                    for setting_name in setting_names(binding.setting_name)
                }
            },
        )


class DisplayDataOnImageModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind DisplayDataOnImage measurement-selection settings."""

    module_name = "DisplayDataOnImage"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(display_data_on_image_bound_kwargs(module))


class CalculateMathModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind CalculateMath operand and arithmetic settings."""

    module_name = "CalculateMath"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(calculate_math_bound_kwargs(module, binder))


class ClassifyObjectsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ClassifyObjects settings into absorbed classification kwargs."""

    module_name = "ClassifyObjectsSingleMeasurement"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(classify_objects_bound_kwargs(module, binder))


class CropModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind Crop's coordinate/mask mode settings into absorbed Crop kwargs."""

    module_name = "Crop"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(crop_bound_kwargs(module, binder))


class CorrectIlluminationCalculateModuleSettingsBindingStrategy(
    DeclarativeModuleSettingsBindingStrategy
):
    """Bind illumination-function calculation settings without bool/enum loss."""

    module_name = "CorrectIlluminationCalculate"
    setting_bindings = CORRECT_ILLUMINATION_CALCULATE_SETTINGS


class CorrectIlluminationApplyModuleSettingsBindingStrategy(
    DeclarativeModuleSettingsBindingStrategy
):
    """Bind illumination application settings for image+function pairs."""

    module_name = "CorrectIlluminationApply"
    setting_bindings = CORRECT_ILLUMINATION_APPLY_SETTINGS

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        kwargs = binder.bind_declared(module, type(self).setting_bindings)
        _bind_repeated_correct_illumination_apply_settings(module, kwargs)
        return BoundModuleSettings(kwargs)


def _bind_repeated_correct_illumination_apply_settings(
    module: ModuleBlock,
    kwargs: dict[str, Any],
) -> None:
    """Preserve per-image-pair settings when one CP module has several pairs."""
    method_values = setting_values(
        module,
        "Select how the illumination function is applied",
    )
    if len(method_values) > 1:
        kwargs["method"] = tuple(
            coerce_cellprofiler_enum(IlluminationCorrectionMethod, value).value
            for value in method_values
        )
    _bind_repeated_bool_setting(
        module,
        "Set output image values less than 0 equal to 0?",
        "truncate_low",
        kwargs,
    )
    _bind_repeated_bool_setting(
        module,
        "Set output image values greater than 1 equal to 1?",
        "truncate_high",
        kwargs,
    )


def _bind_repeated_bool_setting(
    module: ModuleBlock,
    setting_name: str,
    parameter_name: str,
    kwargs: dict[str, Any],
) -> None:
    values = setting_values(module, setting_name)
    if len(values) > 1:
        kwargs[parameter_name] = tuple(parse_cellprofiler_bool(value) for value in values)


class SmoothModuleSettingsBindingStrategy(DeclarativeModuleSettingsBindingStrategy):
    """Bind Smooth's filter and scale settings into absorbed image smoothing."""

    module_name = "Smooth"
    setting_bindings = SMOOTH_SETTINGS


class EnhanceEdgesModuleSettingsBindingStrategy(
    DeclarativeModuleSettingsBindingStrategy
):
    """Bind EnhanceEdges method and threshold settings into edge filtering."""

    module_name = "EnhanceEdges"
    setting_bindings = ENHANCE_EDGES_SETTINGS


class ExpandOrShrinkObjectsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ExpandOrShrinkObjects operation settings into morphology kwargs."""

    module_name = "ExpandOrShrinkObjects"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(expand_or_shrink_bound_kwargs(module, binder))


class MaskObjectsModuleSettingsBindingStrategy(
    DeclarativeModuleSettingsBindingStrategy
):
    """Bind object-mask policy settings into absorbed MaskObjects kwargs."""

    module_name = "MaskObjects"
    setting_bindings = MASK_OBJECTS_SETTINGS


class StructuringElementModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind shared morphology structuring-element settings."""

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(structuring_element_bound_kwargs(module, binder))


class OpeningModuleSettingsBindingStrategy(StructuringElementModuleSettingsBindingStrategy):
    """Bind Opening's CellProfiler structuring-element setting."""

    module_name = "Opening"


class ClosingModuleSettingsBindingStrategy(StructuringElementModuleSettingsBindingStrategy):
    """Bind Closing's CellProfiler structuring-element setting."""

    module_name = "Closing"


class ErodeImageModuleSettingsBindingStrategy(StructuringElementModuleSettingsBindingStrategy):
    """Bind ErodeImage's CellProfiler structuring-element setting."""

    module_name = "ErodeImage"


class DilateImageModuleSettingsBindingStrategy(StructuringElementModuleSettingsBindingStrategy):
    """Bind DilateImage's CellProfiler structuring-element setting."""

    module_name = "DilateImage"


class ErodeObjectsModuleSettingsBindingStrategy(StructuringElementModuleSettingsBindingStrategy):
    """Bind ErodeObjects structuring-element and object-preservation semantics."""

    module_name = "ErodeObjects"
    setting_bindings = (
        SettingToKeywordBinding(
            "Prevent object removal",
            "preserve_midpoints",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Relabel resulting objects",
            "relabel_objects",
            parse_cellprofiler_bool,
        ),
    )

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super()._bind(module, binder=binder, param_mapping=param_mapping)
        declarative_bound = binder.bind_declared(module, type(self).setting_bindings)
        return BoundModuleSettings({**bound.kwargs, **declarative_bound})


class DefineGridModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind DefineGrid settings into absorbed grid-definition kwargs."""

    module_name = "DefineGridManual"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(
            define_grid_bound_kwargs(module, binder),
            invocation_options=define_grid_invocation_options(module),
        )


class IdentifyObjectsInGridModuleSettingsBindingStrategy(
    ModuleSettingsBindingStrategy
):
    """Bind IdentifyObjectsInGrid settings into absorbed grid-object kwargs."""

    module_name = "IdentifyObjectsInGrid"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(
            identify_objects_in_grid_bound_kwargs(module, binder)
        )


class UntangleWormsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind UntangleWorms output-mode settings into typed runtime kwargs."""

    module_name = "UntangleWorms"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(untangle_worms_bound_kwargs(module))


class StraightenWormsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind StraightenWorms geometry and measurement settings."""

    module_name = "StraightenWorms"

    def _bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(straighten_worms_bound_kwargs(module))


class GrayToColorSchemeBindingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Closed nominal family for GrayToColor scheme-specific kwarg lowering."""

    __registry_key__ = "scheme_literal"
    __skip_if_no_key__ = True
    scheme_literal: ClassVar[str | None] = None

    @classmethod
    def for_scheme(
        cls,
        scheme: GrayToColorScheme,
    ) -> "GrayToColorSchemeBindingStrategy":
        strategy_type = cls.__registry__.get(scheme.value)
        if strategy_type is None:
            raise ValueError(f"Unsupported GrayToColor scheme: {scheme.value!r}")
        return strategy_type()

    @abstractmethod
    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        """Bind one GrayToColor module for a specific color scheme."""


class _GrayToColorIndexedSchemeBindingStrategy(GrayToColorSchemeBindingStrategy):
    """Base class for schemes whose payload is indexed into ordered image planes."""

    image_settings: ClassVar[tuple[tuple[str, str], ...]] = ()
    weight_settings: ClassVar[tuple[tuple[str, str], ...]] = ()

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        kwargs: dict[str, Any] = {
            "color_scheme": type(self).scheme_literal,
            "rescale_intensity": _typed_setting_value(
                binder,
                GRAY_TO_COLOR_RESCALE_SETTING,
                module.get_setting(
                    GRAY_TO_COLOR_RESCALE_SETTING,
                    gray_to_color_rescale_default(module),
                ),
            ),
        }
        channel_index = 0
        for setting_name, parameter_name in type(self).image_settings:
            kwargs[parameter_name] = -1
            image_name = module.get_setting(setting_name, "").strip()
            if is_blank_gray_to_color_source(image_name):
                continue
            kwargs[parameter_name] = channel_index
            channel_index += 1
        for setting_name, parameter_name in type(self).weight_settings:
            kwargs[parameter_name] = _typed_setting_value(
                binder,
                setting_name,
                module.get_setting(setting_name, "1.0"),
            )
        return BoundModuleSettings(kwargs)


class GrayToColorRgbBindingStrategy(_GrayToColorIndexedSchemeBindingStrategy):
    scheme_literal = GrayToColorScheme.RGB.value
    image_settings = tuple(
        zip(
            GRAY_TO_COLOR_RGB_IMAGE_SETTINGS,
            ("red_channel", "green_channel", "blue_channel"),
            strict=True,
        )
    )
    weight_settings = tuple(
        zip(
            GRAY_TO_COLOR_RGB_WEIGHT_SETTINGS,
            ("red_weight", "green_weight", "blue_weight"),
            strict=True,
        )
    )


class GrayToColorCmykBindingStrategy(_GrayToColorIndexedSchemeBindingStrategy):
    scheme_literal = GrayToColorScheme.CMYK.value
    image_settings = tuple(
        zip(
            GRAY_TO_COLOR_CMYK_IMAGE_SETTINGS,
            ("cyan_channel", "magenta_channel", "yellow_channel", "gray_channel"),
            strict=True,
        )
    )
    weight_settings = tuple(
        zip(
            GRAY_TO_COLOR_CMYK_WEIGHT_SETTINGS,
            ("cyan_weight", "magenta_weight", "yellow_weight", "gray_weight"),
            strict=True,
        )
    )


class _GrayToColorStackFamilyBindingStrategy(GrayToColorSchemeBindingStrategy):
    """Base class for Stack/Composite repeated per-image settings."""

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        channels = gray_to_color_stack_channels(module)
        kwargs: dict[str, Any] = {
            "color_scheme": type(self).scheme_literal,
            "rescale_intensity": _typed_setting_value(
                binder,
                GRAY_TO_COLOR_RESCALE_SETTING,
                module.get_setting(
                    GRAY_TO_COLOR_RESCALE_SETTING,
                    gray_to_color_rescale_default(module),
                ),
            ),
            "channel_weights": tuple(
                float(_typed_setting_value(binder, "Weight", channel.weight))
                for channel in channels
            ),
        }
        if type(self).scheme_literal == GrayToColorScheme.COMPOSITE.value:
            kwargs["channel_colors"] = tuple(
                channel.color for channel in channels
            )
        return BoundModuleSettings(kwargs)


class GrayToColorStackBindingStrategy(_GrayToColorStackFamilyBindingStrategy):
    scheme_literal = GrayToColorScheme.STACK.value


class GrayToColorCompositeBindingStrategy(_GrayToColorStackFamilyBindingStrategy):
    scheme_literal = GrayToColorScheme.COMPOSITE.value


def _translate_bound_kwargs(
    kwargs: Mapping[str, Any],
    param_mapping: Mapping[str, Any],
) -> BoundModuleSettings:
    translated_kwargs: dict[str, Any] = {}
    unmapped_kwargs: dict[str, Any] = {}

    for cp_setting, value in kwargs.items():
        if cp_setting in param_mapping:
            py_param = param_mapping[cp_setting]
            if py_param is None:
                continue
            if isinstance(py_param, list):
                if isinstance(value, tuple) and len(value) == len(py_param):
                    for index, param_name in enumerate(py_param):
                        translated_kwargs[param_name] = value[index]
                else:
                    translated_kwargs[py_param[0]] = value
            else:
                translated_kwargs[py_param] = value
            continue
        unmapped_kwargs[cp_setting] = value

    return BoundModuleSettings(
        kwargs=translated_kwargs,
        unmapped_kwargs=unmapped_kwargs,
    )


def _parse_colocalization_run_all_metrics(value: str) -> bool:
    """Parse modern and legacy MeasureColocalization all-metric settings."""
    try:
        return parse_cellprofiler_bool(value)
    except ValueError:
        # Older pipelines may store legacy accuracy choices such as "Accurate"
        # in this slot. Those choices do not disable any metric family.
        return bool(value.strip())


def _parse_colocalization_costes_method(value: str) -> str:
    """Normalize CellProfiler Costes method choices to absorbed-function values."""
    normalized = value.strip().lower()
    if normalized not in {"faster", "fast", "accurate"}:
        raise ValueError(f"Unsupported Costes thresholding method: {value!r}.")
    return normalized


def _typed_setting_value(
    binder: SettingsBinder,
    key: str,
    value: str,
) -> Any:
    return binder.parse_value(key, value)


def _required_class_attr[T](value: T | None, name: str) -> T:
    if value is None:
        raise TypeError(f"Measurement settings strategy must define {name}.")
    return value
