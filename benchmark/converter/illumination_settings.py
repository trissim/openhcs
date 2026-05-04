"""Typed lowering for illumination-module settings."""

from __future__ import annotations

from enum import Enum

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from benchmark.cellprofiler_library.functions.correctilluminationapply import (
    IlluminationCorrectionMethod,
)
from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
    CalculationScope,
    FilterSizeMethod,
    IntensityChoice,
    RescaleOption,
    SmoothingMethod,
    SplineBgMode,
)

from .settings_binder import (
    SettingParser,
    SettingToKeywordBinding,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)


def _enum_literal(enum_type: type[Enum]) -> SettingParser:
    def parse(value: str) -> str:
        return _coerce_function_enum(enum_type, value).value

    return parse


def _calculation_scope_literal(value: str) -> str:
    cleaned = (
        value.replace("\x00", "")
        .replace("\ufeff", "")
        .replace("ÿþ", "")
        .replace("þÿ", "")
    )
    return _coerce_function_enum(CalculationScope, cleaned).value


CORRECT_ILLUMINATION_CALCULATE_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding(
        "Select how the illumination function is calculated",
        "intensity_choice",
        _enum_literal(IntensityChoice),
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
        _enum_literal(RescaleOption),
    ),
    SettingToKeywordBinding(
        "Calculate function for each image individually, or based on all images?",
        "calculation_scope",
        _calculation_scope_literal,
    ),
    SettingToKeywordBinding(
        "Smoothing method",
        "smoothing_method",
        _enum_literal(SmoothingMethod),
    ),
    SettingToKeywordBinding(
        "Method to calculate smoothing filter size",
        "filter_size_method",
        _enum_literal(FilterSizeMethod),
    ),
    SettingToKeywordBinding(
        "Approximate object diameter",
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
        _enum_literal(SplineBgMode),
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
