"""Typed lowering for CellProfiler ClassifyObjects variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .parser import ModuleBlock
from .setting_names import SettingNameFamily, setting_values
from .settings_binder import SettingsBinder


CLASSIFICATION_DECISION_COUNT_SETTING = SettingNameFamily(
    "Make each classification decision on how many measurements?"
)
SINGLE_MEASUREMENT_FEATURE_SETTING = SettingNameFamily(
    "Select the measurement to classify by"
)
FIRST_MEASUREMENT_FEATURE_SETTING = SettingNameFamily("Select the first measurement")
SECOND_MEASUREMENT_FEATURE_SETTING = SettingNameFamily("Select the second measurement")


class ClassifyObjectsVariant(str, Enum):
    """Absorbed ClassifyObjects function variants."""

    SINGLE_MEASUREMENT = "classify_objects_single_measurement"
    TWO_MEASUREMENTS = "classify_objects_two_measurements"

    @classmethod
    def from_module(cls, module: ModuleBlock) -> "ClassifyObjectsVariant":
        value = IndexedClassifySetting(
            module,
            CLASSIFICATION_DECISION_COUNT_SETTING,
            default="Single measurement",
        ).first.lower()
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


@dataclass(frozen=True, slots=True)
class IndexedClassifySetting:
    """CellProfiler ClassifyObjects repeated setting with fallback semantics."""

    module: ModuleBlock
    setting_name: str | SettingNameFamily
    default: str

    def at(self, value_index: int) -> str:
        values = setting_values(self.module, self.setting_name)
        if not values:
            return self.default
        if value_index < len(values):
            return values[value_index]
        return values[-1]

    @property
    def first(self) -> str:
        return self.at(0)

    @property
    def last(self) -> str:
        values = setting_values(self.module, self.setting_name)
        return values[-1] if values else self.default

    def optional_at(self, value_index: int) -> str | None:
        value = self.at(value_index)
        return value or None

    def required_at(self, value_index: int) -> str:
        value = self.at(value_index).strip()
        if not value:
            raise ValueError(f"ClassifyObjects requires setting {self.setting_name!r}.")
        return value

    @property
    def required(self) -> str:
        return self.required_at(0)


@dataclass(frozen=True, slots=True)
class TypedClassifySetting:
    """Typed parser request for one ClassifyObjects setting."""

    module: ModuleBlock
    binder: SettingsBinder
    setting_name: str
    default: str
    value_index: int = 0

    @property
    def value(self) -> Any:
        raw_value = IndexedClassifySetting(
            self.module,
            self.setting_name,
            default=self.default,
        ).at(self.value_index)
        return self.binder.parse_value(self.setting_name, raw_value)


def _single_measurement_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    measurement_features = setting_values(module, SINGLE_MEASUREMENT_FEATURE_SETTING)
    if len(measurement_features) > 1:
        return {
            "classification_rules": tuple(
                _single_measurement_rule_kwargs(module, binder, index)
                for index in range(len(measurement_features))
            ),
        }
    return _single_measurement_rule_kwargs(module, binder, 0)


def _single_measurement_rule_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
    value_index: int,
) -> dict[str, Any]:
    return {
        "measurement_feature": IndexedClassifySetting(
            module,
            SINGLE_MEASUREMENT_FEATURE_SETTING,
            default="",
        ).required_at(value_index),
        "bin_choice": _bin_choice(
            IndexedClassifySetting(
                module,
                "Select bin spacing",
                default="Evenly spaced bins",
            ).at(value_index)
        ),
        "bin_count": TypedClassifySetting(
            module,
            binder,
            "Number of bins",
            default="3",
            value_index=value_index,
        ).value,
        "low_threshold": TypedClassifySetting(
            module,
            binder,
            "Lower threshold",
            default="0.0",
            value_index=value_index,
        ).value,
        "high_threshold": TypedClassifySetting(
            module,
            binder,
            "Upper threshold",
            default="1.0",
            value_index=value_index,
        ).value,
        "wants_low_bin": TypedClassifySetting(
            module,
            binder,
            "Use a bin for objects below the threshold?",
            default="No",
            value_index=value_index,
        ).value,
        "wants_high_bin": TypedClassifySetting(
            module,
            binder,
            "Use a bin for objects above the threshold?",
            default="No",
            value_index=value_index,
        ).value,
        "custom_thresholds": IndexedClassifySetting(
            module,
            "Enter the custom thresholds separating the values between bins",
            default="0,1",
        ).at(value_index),
        "bin_names": IndexedClassifySetting(
            module,
            "Enter the bin names separated by commas",
            default="",
        ).optional_at(value_index),
    }


def _two_measurement_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    return {
        "measurement1_feature": IndexedClassifySetting(
            module,
            FIRST_MEASUREMENT_FEATURE_SETTING,
            default="",
        ).required,
        "measurement2_feature": IndexedClassifySetting(
            module,
            SECOND_MEASUREMENT_FEATURE_SETTING,
            default="",
        ).required,
        "threshold1_method": _threshold_method(
            IndexedClassifySetting(
                module,
                "Method to select the cutoff",
                default="Mean",
            ).first
        ),
        "threshold1_value": TypedClassifySetting(
            module,
            binder,
            "Enter the cutoff value",
            default="0.5",
        ).value,
        "threshold2_method": _threshold_method(
            IndexedClassifySetting(
                module,
                "Method to select the cutoff",
                default="Mean",
            ).last
        ),
        "threshold2_value": TypedClassifySetting(
            module,
            binder,
            "Enter the cutoff value",
            default="0.5",
            value_index=-1,
        ).value,
        "low_low_name": IndexedClassifySetting(
            module,
            "Enter the low-low bin name",
            default="low_low",
        ).first,
        "low_high_name": IndexedClassifySetting(
            module,
            "Enter the low-high bin name",
            default="low_high",
        ).first,
        "high_low_name": IndexedClassifySetting(
            module,
            "Enter the high-low bin name",
            default="high_low",
        ).first,
        "high_high_name": IndexedClassifySetting(
            module,
            "Enter the high-high bin name",
            default="high_high",
        ).first,
    }


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
