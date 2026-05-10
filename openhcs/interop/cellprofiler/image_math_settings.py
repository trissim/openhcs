"""Typed lowering for CellProfiler ImageMath settings."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any

from metaclass_registry import AutoRegisterMeta

from openhcs.core.registry_strategies import EnumKeyedStrategyMixin

from .parser import ModuleBlock
from .setting_names import optional_setting_value
from .settings_binder import (
    SettingToKeywordBinding,
    SettingsBinder,
    coerce_cellprofiler_enum,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
)


class ImageMathOperation(Enum):
    """ImageMath operation literals exposed by CellProfiler settings."""

    ADD = "add"
    SUBTRACT = "subtract"
    DIFFERENCE = "absolute_difference"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    STDEV = "standard_deviation"
    INVERT = "invert"
    COMPLEMENT = "complement"
    LOG_TRANSFORM = "log_transform_base2"
    LOG_TRANSFORM_LEGACY = "log_transform_legacy"
    NONE = "none"
    OR = "or"
    AND = "and"
    NOT = "not"
    EQUALS = "equals"


def parse_image_math_operation(value: str) -> str:
    """Return the absorbed-function operation literal for a CP setting."""
    normalized = normalize_cellprofiler_setting_name(value)
    aliases = {
        "difference": ImageMathOperation.DIFFERENCE,
        "stdev": ImageMathOperation.STDEV,
        "log_transform": ImageMathOperation.LOG_TRANSFORM,
        "log_transform_base_2": ImageMathOperation.LOG_TRANSFORM,
    }
    if normalized in aliases:
        return aliases[normalized].value
    return coerce_cellprofiler_enum(ImageMathOperation, value).value


IMAGE_MATH_SETTINGS: tuple[SettingToKeywordBinding, ...] = (
    SettingToKeywordBinding(
        "Operation",
        "operation",
        parse_image_math_operation,
    ),
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
    factors = ImageMathOperandFactorSettingResolver.bound_factors(module)
    if factors:
        kwargs["factors"] = factors
    return kwargs


class ImageMathOperandOrdinal(Enum):
    """Ordered ImageMath operand slots represented in CellProfiler settings."""

    FIRST = 1
    SECOND = 2


@dataclass(frozen=True, slots=True)
class ImageMathOperandFactorSetting:
    """Typed repeated ImageMath factor setting for one operand slot."""

    ordinal: ImageMathOperandOrdinal
    setting_name: str

    def value_for_module(self, module: ModuleBlock) -> float | None:
        value = optional_setting_value(module, self.setting_name)
        if value is None:
            return None
        return parse_cellprofiler_float(value)


class ImageMathOperandFactorSettingResolver(
    EnumKeyedStrategyMixin[ImageMathOperandOrdinal],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered resolver for ImageMath operand-factor settings."""

    __registry_key__ = "ordinal_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "ordinal"
    __enum_label_attr__ = "ordinal_label"
    ordinal: ImageMathOperandOrdinal | None = None
    ordinal_label: str | None = None
    setting_name: str

    @classmethod
    def registered_settings(cls) -> tuple[ImageMathOperandFactorSetting, ...]:
        return tuple(
            ImageMathOperandFactorSetting(
                resolver_type.ordinal_member(),
                resolver_type.setting_name,
            )
            for resolver_type in sorted(
                cls.registered_strategy_types(),
                key=lambda registered_type: registered_type.ordinal_member().value,
            )
        )

    @classmethod
    def bound_factors(cls, module: ModuleBlock) -> tuple[float, ...]:
        factors: list[float] = []
        for setting in cls.registered_settings():
            value = setting.value_for_module(module)
            if value is not None:
                factors.append(value)
        return tuple(factors)

    @classmethod
    def ordinal_member(cls) -> ImageMathOperandOrdinal:
        if not isinstance(cls.ordinal, ImageMathOperandOrdinal):
            raise TypeError(f"{cls.__name__} must declare an ImageMath operand ordinal.")
        return cls.ordinal


class FirstImageMathOperandFactorSetting(ImageMathOperandFactorSettingResolver):
    ordinal = ImageMathOperandOrdinal.FIRST
    setting_name = "Multiply the first image by"


class SecondImageMathOperandFactorSetting(ImageMathOperandFactorSettingResolver):
    ordinal = ImageMathOperandOrdinal.SECOND
    setting_name = "Multiply the second image by"
