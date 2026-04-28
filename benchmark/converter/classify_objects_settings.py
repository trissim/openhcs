"""Typed lowering for CellProfiler ClassifyObjects variants."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .parser import ModuleBlock
from .setting_names import SettingNameFamily, setting_values
from .settings_binder import SettingsBinder


CLASSIFICATION_DECISION_COUNT_SETTING = SettingNameFamily(
    "Make each classification decision on how many measurements?"
)


class ClassifyObjectsVariant(Enum):
    """Absorbed ClassifyObjects function variants."""

    SINGLE_MEASUREMENT = "classify_objects_single_measurement"
    TWO_MEASUREMENTS = "classify_objects_two_measurements"

    @classmethod
    def from_module(cls, module: ModuleBlock) -> "ClassifyObjectsVariant":
        value = _first_setting_value(
            module,
            CLASSIFICATION_DECISION_COUNT_SETTING,
            default="Single measurement",
        ).lower()
        if "two" in value:
            return cls.TWO_MEASUREMENTS
        if "single" in value:
            return cls.SINGLE_MEASUREMENT
        raise ValueError(
            f"Unsupported ClassifyObjects measurement count setting: {value!r}."
        )

    @property
    def function_name(self) -> str:
        return self.value


def classify_objects_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return kwargs for the absorbed ClassifyObjects variant."""

    variant = ClassifyObjectsVariant.from_module(module)
    if variant is ClassifyObjectsVariant.TWO_MEASUREMENTS:
        return _two_measurement_kwargs(module, binder)
    return _single_measurement_kwargs(module, binder)


def _single_measurement_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    return {
        "bin_choice": _bin_choice(
            _first_setting_value(
                module,
                "Select bin spacing",
                default="Evenly spaced bins",
            )
        ),
        "bin_count": _typed_setting_value(
            module,
            binder,
            "Number of bins",
            default="3",
        ),
        "low_threshold": _typed_setting_value(
            module,
            binder,
            "Lower threshold",
            default="0.0",
        ),
        "high_threshold": _typed_setting_value(
            module,
            binder,
            "Upper threshold",
            default="1.0",
        ),
        "wants_low_bin": _typed_setting_value(
            module,
            binder,
            "Use a bin for objects below the threshold?",
            default="No",
        ),
        "wants_high_bin": _typed_setting_value(
            module,
            binder,
            "Use a bin for objects above the threshold?",
            default="No",
        ),
        "custom_thresholds": _first_setting_value(
            module,
            "Enter the custom thresholds separating the values between bins",
            default="0,1",
        ),
        "bin_names": _optional_setting_value(
            module,
            "Enter the bin names separated by commas",
        ),
    }


def _two_measurement_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    return {
        "threshold1_method": _threshold_method(
            _first_setting_value(
                module,
                "Method to select the cutoff",
                default="Mean",
            )
        ),
        "threshold1_value": _typed_setting_value(
            module,
            binder,
            "Enter the cutoff value",
            default="0.5",
        ),
        "threshold2_method": _threshold_method(
            _last_setting_value(
                module,
                "Method to select the cutoff",
                default="Mean",
            )
        ),
        "threshold2_value": _typed_setting_value(
            module,
            binder,
            "Enter the cutoff value",
            default="0.5",
            value_index=-1,
        ),
        "low_low_name": _first_setting_value(
            module,
            "Enter the low-low bin name",
            default="low_low",
        ),
        "low_high_name": _first_setting_value(
            module,
            "Enter the low-high bin name",
            default="low_high",
        ),
        "high_low_name": _first_setting_value(
            module,
            "Enter the high-low bin name",
            default="high_low",
        ),
        "high_high_name": _first_setting_value(
            module,
            "Enter the high-high bin name",
            default="high_high",
        ),
    }


def _typed_setting_value(
    module: ModuleBlock,
    binder: SettingsBinder,
    setting_name: str,
    *,
    default: str,
    value_index: int = 0,
) -> Any:
    values = setting_values(module, setting_name)
    value = values[value_index] if values else default
    return binder.parse_value(setting_name, value)


def _optional_setting_value(
    module: ModuleBlock,
    setting_name: str,
) -> str | None:
    value = _first_setting_value(module, setting_name, default="")
    return value or None


def _first_setting_value(
    module: ModuleBlock,
    setting_name: str | SettingNameFamily,
    *,
    default: str,
) -> str:
    values = setting_values(module, setting_name)
    return values[0] if values else default


def _last_setting_value(
    module: ModuleBlock,
    setting_name: str | SettingNameFamily,
    *,
    default: str,
) -> str:
    values = setting_values(module, setting_name)
    return values[-1] if values else default


def _bin_choice(value: str) -> str:
    normalized = value.strip().lower()
    if "custom" in normalized:
        return "custom"
    if "even" in normalized:
        return "even"
    raise ValueError(f"Unsupported ClassifyObjects bin spacing: {value!r}.")


def _threshold_method(value: str) -> str:
    normalized = value.strip().lower()
    if "median" in normalized:
        return "median"
    if "mean" in normalized:
        return "mean"
    if "custom" in normalized:
        return "custom"
    raise ValueError(f"Unsupported ClassifyObjects threshold method: {value!r}.")
