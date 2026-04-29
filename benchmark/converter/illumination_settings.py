"""Typed lowering for CellProfiler illumination-module settings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from benchmark.cellprofiler_library.functions.correctilluminationapply import (
    IlluminationCorrectionMethod,
)
from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
    FilterSizeMethod,
    IntensityChoice,
    RescaleOption,
    SmoothingMethod,
    SplineBgMode,
)

from .parser import ModuleBlock
from .setting_names import optional_setting_value


SettingParser = Callable[[str], object]


@dataclass(frozen=True, slots=True)
class CellProfilerSettingBinding:
    """Declarative mapping from one CellProfiler setting to one function kwarg."""

    setting_name: str
    parameter_name: str
    parse: SettingParser

    def bind(self, module: ModuleBlock, kwargs: dict[str, Any]) -> None:
        value = optional_setting_value(module, self.setting_name)
        if value is None:
            return
        kwargs[self.parameter_name] = self.parse(value)


def _enum_literal(enum_type: type[Enum]) -> SettingParser:
    def parse(value: str) -> str:
        return _coerce_function_enum(enum_type, value).value

    return parse


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"yes", "true", "1", "on"}:
        return True
    if normalized in {"no", "false", "0", "off"}:
        return False
    raise ValueError(f"CellProfiler boolean setting must be Yes/No, got {value!r}.")


def _int(value: str) -> int:
    return int(float(value))


def _float(value: str) -> float:
    return float(value)


CORRECT_ILLUMINATION_CALCULATE_SETTINGS: tuple[CellProfilerSettingBinding, ...] = (
    CellProfilerSettingBinding(
        "Select how the illumination function is calculated",
        "intensity_choice",
        _enum_literal(IntensityChoice),
    ),
    CellProfilerSettingBinding(
        "Dilate objects in the final averaged image?",
        "dilate_objects",
        _bool,
    ),
    CellProfilerSettingBinding("Dilation radius", "object_dilation_radius", _int),
    CellProfilerSettingBinding("Block size", "block_size", _int),
    CellProfilerSettingBinding(
        "Rescale the illumination function?",
        "rescale_option",
        _enum_literal(RescaleOption),
    ),
    CellProfilerSettingBinding(
        "Smoothing method",
        "smoothing_method",
        _enum_literal(SmoothingMethod),
    ),
    CellProfilerSettingBinding(
        "Method to calculate smoothing filter size",
        "filter_size_method",
        _enum_literal(FilterSizeMethod),
    ),
    CellProfilerSettingBinding("Approximate object diameter", "object_width", _int),
    CellProfilerSettingBinding("Smoothing filter size", "manual_filter_size", _int),
    CellProfilerSettingBinding(
        "Automatically calculate spline parameters?",
        "automatic_splines",
        _bool,
    ),
    CellProfilerSettingBinding(
        "Background mode",
        "spline_bg_mode",
        _enum_literal(SplineBgMode),
    ),
    CellProfilerSettingBinding("Number of spline points", "spline_points", _int),
    CellProfilerSettingBinding("Background threshold", "spline_threshold", _float),
    CellProfilerSettingBinding("Image resampling factor", "spline_rescale", _float),
    CellProfilerSettingBinding(
        "Maximum number of iterations",
        "spline_max_iterations",
        _int,
    ),
    CellProfilerSettingBinding(
        "Residual value for convergence",
        "spline_convergence",
        _float,
    ),
)

CORRECT_ILLUMINATION_APPLY_SETTINGS: tuple[CellProfilerSettingBinding, ...] = (
    CellProfilerSettingBinding(
        "Select how the illumination function is applied",
        "method",
        _enum_literal(IlluminationCorrectionMethod),
    ),
    CellProfilerSettingBinding(
        "Set output image values less than 0 equal to 0?",
        "truncate_low",
        _bool,
    ),
    CellProfilerSettingBinding(
        "Set output image values greater than 1 equal to 1?",
        "truncate_high",
        _bool,
    ),
)


def correct_illumination_calculate_bound_kwargs(
    module: ModuleBlock,
) -> dict[str, Any]:
    """Return absorbed-function kwargs for CorrectIlluminationCalculate."""

    return _bound_kwargs(module, CORRECT_ILLUMINATION_CALCULATE_SETTINGS)


def correct_illumination_apply_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    """Return absorbed-function kwargs for CorrectIlluminationApply."""

    return _bound_kwargs(module, CORRECT_ILLUMINATION_APPLY_SETTINGS)


def _bound_kwargs(
    module: ModuleBlock,
    bindings: tuple[CellProfilerSettingBinding, ...],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for binding in bindings:
        binding.bind(module, kwargs)
    return kwargs
