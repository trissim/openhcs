"""Typed lowering for CellProfiler CalculateMath settings."""

from __future__ import annotations

from typing import Any

from benchmark.cellprofiler_compat.measurement_lookup import count_feature_object_name

from .parser import ModuleBlock
from .setting_names import SettingNameFamily, optional_setting_value
from .settings_binder import SettingsBinder


OUTPUT_MEASUREMENT_SETTING = SettingNameFamily("Name the output measurement")
OPERATION_SETTING = SettingNameFamily("Operation")
NUMERATOR_OBJECTS_SETTING = SettingNameFamily("Select the numerator objects")
NUMERATOR_MEASUREMENT_SETTING = SettingNameFamily("Select the numerator measurement")
DENOMINATOR_OBJECTS_SETTING = SettingNameFamily("Select the denominator objects")
DENOMINATOR_MEASUREMENT_SETTING = SettingNameFamily(
    "Select the denominator measurement"
)


def calculate_math_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return absorbed-function kwargs for runtime CalculateMath operands."""

    return {
        "output_name": _setting_value(
            module,
            OUTPUT_MEASUREMENT_SETTING,
            default="Measurement",
        ),
        "operation": _setting_value(module, OPERATION_SETTING, default="None"),
        "operand1_feature": _setting_value(module, NUMERATOR_MEASUREMENT_SETTING),
        "operand2_feature": _setting_value(module, DENOMINATOR_MEASUREMENT_SETTING),
        "operand1_object_name": _optional_object_name(
            module,
            NUMERATOR_OBJECTS_SETTING,
        ),
        "operand2_object_name": _optional_object_name(
            module,
            DENOMINATOR_OBJECTS_SETTING,
        ),
        "operand1_multiplicand": _typed_setting(
            module,
            binder,
            "Multiply the above operand by",
            index=0,
            default="1.0",
        ),
        "operand1_exponent": _typed_setting(
            module,
            binder,
            "Raise the power of above operand by",
            index=0,
            default="1.0",
        ),
        "operand2_multiplicand": _typed_setting(
            module,
            binder,
            "Multiply the above operand by",
            index=1,
            default="1.0",
        ),
        "operand2_exponent": _typed_setting(
            module,
            binder,
            "Raise the power of above operand by",
            index=1,
            default="1.0",
        ),
        "take_log10": _typed_setting(
            module,
            binder,
            "Take log10 of result?",
            default="No",
        ),
        "final_multiplicand": _typed_setting(
            module,
            binder,
            "Multiply the result by",
            default="1.0",
        ),
        "final_exponent": _typed_setting(
            module,
            binder,
            "Raise the power of result by",
            default="1.0",
        ),
        "final_addend": _typed_setting(
            module,
            binder,
            "Add to the result",
            default="0.0",
        ),
        "rounding": _setting_value(
            module,
            "How should the output value be rounded?",
            default="Not rounded",
        ),
        "rounding_digits": _typed_setting(
            module,
            binder,
            "Enter how many decimal places the value should be rounded to",
            default="0",
        ),
        "constrain_lower_bound": _typed_setting(
            module,
            binder,
            "Constrain the result to a lower bound?",
            default="No",
        ),
        "lower_bound": _typed_setting(
            module,
            binder,
            "Enter the lower bound",
            default="0.0",
        ),
        "constrain_upper_bound": _typed_setting(
            module,
            binder,
            "Constrain the result to an upper bound?",
            default="No",
        ),
        "upper_bound": _typed_setting(
            module,
            binder,
            "Enter the upper bound",
            default="1.0",
        ),
    }


def calculate_math_object_dependencies(module: ModuleBlock) -> tuple[str, ...]:
    """Return object names referenced by CalculateMath measurement operands."""

    names = (
        _optional_object_name(module, NUMERATOR_OBJECTS_SETTING),
        _optional_object_name(module, DENOMINATOR_OBJECTS_SETTING),
        count_feature_object_name(
            optional_setting_value(module, NUMERATOR_MEASUREMENT_SETTING)
        ),
        count_feature_object_name(
            optional_setting_value(module, DENOMINATOR_MEASUREMENT_SETTING)
        ),
    )
    return tuple(dict.fromkeys(name for name in names if name is not None))


def _typed_setting(
    module: ModuleBlock,
    binder: SettingsBinder,
    setting_name: str,
    *,
    default: str,
    index: int = 0,
) -> Any:
    return binder.parse_value(
        setting_name,
        _indexed_setting_value(module, setting_name, index=index, default=default),
    )


def _indexed_setting_value(
    module: ModuleBlock,
    setting_name: str,
    *,
    index: int,
    default: str,
) -> str:
    values = module.get_setting_values(setting_name)
    if index < len(values):
        return values[index]
    return default


def _setting_value(
    module: ModuleBlock,
    setting_name: str | SettingNameFamily,
    *,
    default: str | None = None,
) -> str:
    value = optional_setting_value(module, setting_name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ValueError(f"CalculateMath requires setting {setting_name!r}.")


def _optional_object_name(
    module: ModuleBlock,
    setting_name: SettingNameFamily,
) -> str | None:
    value = optional_setting_value(module, setting_name)
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "do not use"}:
        return None
    return normalized

