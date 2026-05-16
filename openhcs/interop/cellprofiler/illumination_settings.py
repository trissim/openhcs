"""Typed lowering for CellProfiler illumination-module settings."""

from __future__ import annotations

from enum import Enum

from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.interop.cellprofiler.settings_binder import (
    SettingParser,
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)


class IlluminationIntensityChoice(Enum):
    REGULAR = "regular"
    BACKGROUND = "background"


class IlluminationSmoothingMethod(Enum):
    NONE = "none"
    CONVEX_HULL = "convex_hull"
    FIT_POLYNOMIAL = "fit_polynomial"
    MEDIAN_FILTER = "median_filter"
    GAUSSIAN_FILTER = "gaussian_filter"
    TO_AVERAGE = "to_average"
    SPLINES = "splines"


IlluminationSmoothingMethod.NONE.cellprofiler_literals = ("No smoothing",)


class IlluminationFilterSizeMethod(Enum):
    AUTOMATIC = "automatic"
    OBJECT_SIZE = "object_size"
    MANUALLY = "manually"


class IlluminationRescaleOption(Enum):
    YES = "yes"
    NO = "no"
    MEDIAN = "median"


class IlluminationSplineBackgroundMode(Enum):
    AUTO = "auto"
    DARK = "dark"
    BRIGHT = "bright"
    GRAY = "gray"


class IlluminationCalculationScope(Enum):
    EACH = "each"
    ALL_FIRST_CYCLE = "all_first_cycle"
    ALL_ACROSS_CYCLES = "all_across_cycles"

    @property
    def requires_channel_grouping(self) -> bool:
        return self is not IlluminationCalculationScope.EACH


class IlluminationCorrectionMethod(Enum):
    DIVIDE = "divide"
    SUBTRACT = "subtract"


def _enum_literal(enum_type: type[Enum]) -> SettingParser:
    def parse(value: str) -> str:
        member = coerce_cellprofiler_enum(enum_type, value)
        if not isinstance(member.value, str):
            raise TypeError(f"{enum_type.__name__}.{member.name} must have a string value.")
        return member.value

    return parse


def _calculation_scope_literal(value: str) -> str:
    cleaned = (
        value.replace("\x00", "")
        .replace("\ufeff", "")
        .replace("ÿþ", "")
        .replace("þÿ", "")
    )
    return coerce_cellprofiler_enum(IlluminationCalculationScope, cleaned).value


CORRECT_ILLUMINATION_CALCULATE_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding(
        "Select how the illumination function is calculated",
        "intensity_choice",
        _enum_literal(IlluminationIntensityChoice),
    ),
    SettingToKeywordBinding(
        "Dilate objects in the final averaged image?",
        "dilate_objects",
        parse_cellprofiler_bool,
    ),
    SettingToKeywordBinding(
        "Dilation radius",
        "object_dilation_radius",
        parse_cellprofiler_int,
    ),
    SettingToKeywordBinding("Block size", "block_size", parse_cellprofiler_int),
    SettingToKeywordBinding(
        "Rescale the illumination function?",
        "rescale_option",
        _enum_literal(IlluminationRescaleOption),
    ),
    SettingToKeywordBinding(
        "Calculate function for each image individually, or based on all images?",
        "calculation_scope",
        _calculation_scope_literal,
    ),
    SettingToKeywordBinding(
        "Smoothing method",
        "smoothing_method",
        _enum_literal(IlluminationSmoothingMethod),
    ),
    SettingToKeywordBinding(
        "Method to calculate smoothing filter size",
        "filter_size_method",
        _enum_literal(IlluminationFilterSizeMethod),
    ),
    SettingToKeywordBinding(
        SettingNameFamily(
            "Approximate object diameter",
            aliases=("Approximate object size",),
        ),
        "object_width",
        parse_cellprofiler_int,
    ),
    SettingToKeywordBinding(
        "Smoothing filter size",
        "manual_filter_size",
        parse_cellprofiler_int,
    ),
    SettingToKeywordBinding(
        "Automatically calculate spline parameters?",
        "automatic_splines",
        parse_cellprofiler_bool,
    ),
    SettingToKeywordBinding(
        "Background mode",
        "spline_bg_mode",
        _enum_literal(IlluminationSplineBackgroundMode),
    ),
    SettingToKeywordBinding(
        "Number of spline points",
        "spline_points",
        parse_cellprofiler_int,
    ),
    SettingToKeywordBinding(
        "Background threshold",
        "spline_threshold",
        parse_cellprofiler_float,
    ),
    SettingToKeywordBinding(
        "Image resampling factor",
        "spline_rescale",
        parse_cellprofiler_float,
    ),
    SettingToKeywordBinding(
        "Maximum number of iterations",
        "spline_max_iterations",
        parse_cellprofiler_int,
    ),
    SettingToKeywordBinding(
        "Residual value for convergence",
        "spline_convergence",
        parse_cellprofiler_float,
    ),
)


CORRECT_ILLUMINATION_APPLY_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding(
        "Select how the illumination function is applied",
        "method",
        _enum_literal(IlluminationCorrectionMethod),
    ),
    SettingToKeywordBinding(
        "Set output image values less than 0 equal to 0?",
        "truncate_low",
        parse_cellprofiler_bool,
    ),
    SettingToKeywordBinding(
        "Set output image values greater than 1 equal to 1?",
        "truncate_high",
        parse_cellprofiler_bool,
    ),
)
