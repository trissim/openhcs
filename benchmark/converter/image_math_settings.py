"""Typed lowering for CellProfiler ImageMath settings."""

from __future__ import annotations

from typing import Any

from .parser import ModuleBlock
from .setting_names import optional_setting_value
from .settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
)


IMAGE_MATH_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding("Operation", "operation"),
    SettingToKeywordBinding(
        "Raise the power of the result by",
        "exponent",
        parse_cellprofiler_float,
    ),
    SettingToKeywordBinding(
        "Multiply the result by",
        "after_factor",
        parse_cellprofiler_float,
    ),
    SettingToKeywordBinding("Add to result", "addend", parse_cellprofiler_float),
    SettingToKeywordBinding(
        "Set values less than 0 equal to 0?",
        "truncate_low",
        parse_cellprofiler_bool,
    ),
    SettingToKeywordBinding(
        "Set values greater than 1 equal to 1?",
        "truncate_high",
        parse_cellprofiler_bool,
    ),
    SettingToKeywordBinding(
        "Replace invalid values with 0?",
        "replace_nan",
        parse_cellprofiler_bool,
    ),
    SettingToKeywordBinding(
        "Ignore the image masks?",
        "ignore_masks",
        parse_cellprofiler_bool,
    ),
)


def image_math_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return absorbed-function kwargs for CellProfiler ImageMath."""
    kwargs = binder.bind_declared(module, IMAGE_MATH_SETTINGS)
    factors = _image_math_factors(module)
    if factors:
        kwargs["factors"] = factors
    return kwargs


def _image_math_factors(module: ModuleBlock) -> tuple[float, ...]:
    factors: list[float] = []
    for setting_name in (
        "Multiply the first image by",
        "Multiply the second image by",
    ):
        value = optional_setting_value(module, setting_name)
        if value is not None:
            factors.append(parse_cellprofiler_float(value))
    return tuple(factors)
