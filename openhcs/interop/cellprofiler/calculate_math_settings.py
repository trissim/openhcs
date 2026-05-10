"""Typed lowering for CellProfiler CalculateMath settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .measurement_lookup import count_feature_object_name
from .parser import ModuleBlock
from .setting_names import (
    SettingNameFamily,
    normalized_symbol_name,
    optional_setting_value,
)
from .settings_binder import SettingsBinder


OUTPUT_MEASUREMENT_SETTING = SettingNameFamily("Name the output measurement")
OPERATION_SETTING = SettingNameFamily("Operation")
NUMERATOR_OBJECTS_SETTING = SettingNameFamily("Select the numerator objects")
NUMERATOR_MEASUREMENT_SETTING = SettingNameFamily("Select the numerator measurement")
DENOMINATOR_OBJECTS_SETTING = SettingNameFamily("Select the denominator objects")
DENOMINATOR_MEASUREMENT_SETTING = SettingNameFamily(
    "Select the denominator measurement"
)


@dataclass(frozen=True, slots=True)
class CalculateMathSettingValue:
    """One CalculateMath setting with default and required-setting semantics."""

    module: ModuleBlock
    setting_name: str | SettingNameFamily
    default: str | None = None

    @property
    def value(self) -> str:
        value = optional_setting_value(self.module, self.setting_name)
        if value is not None:
            return value
        if self.default is not None:
            return self.default
        raise ValueError(f"CalculateMath requires setting {self.setting_name!r}.")


@dataclass(frozen=True, slots=True)
class IndexedCalculateMathSettingValue:
    """Repeated CalculateMath setting value selected by operand index."""

    module: ModuleBlock
    setting_name: str
    index: int
    default: str

    @property
    def value(self) -> str:
        values = self.module.get_setting_values(self.setting_name)
        if self.index < len(values):
            return values[self.index]
        return self.default


@dataclass(frozen=True, slots=True)
class TypedCalculateMathSettingValue:
    """CalculateMath setting parsed through the shared settings binder."""

    module: ModuleBlock
    binder: SettingsBinder
    setting_name: str
    default: str
    index: int = 0

    @property
    def value(self) -> Any:
        return self.binder.parse_value(
            self.setting_name,
            IndexedCalculateMathSettingValue(
                self.module,
                self.setting_name,
                self.index,
                self.default,
            ).value,
        )


@dataclass(frozen=True, slots=True)
class CalculateMathObjectSetting:
    """Optional CalculateMath object selector normalized as an artifact symbol."""

    module: ModuleBlock
    setting_name: SettingNameFamily

    @property
    def object_name(self) -> str | None:
        value = optional_setting_value(self.module, self.setting_name)
        if value is None:
            return None
        return normalized_symbol_name(value)


@dataclass(frozen=True, slots=True)
class CalculateMathOperandSettings:
    """One CalculateMath operand settings row."""

    module: ModuleBlock
    binder: SettingsBinder
    object_setting: SettingNameFamily
    measurement_setting: SettingNameFamily
    operand_index: int

    @property
    def feature_name(self) -> str:
        return CalculateMathSettingValue(
            self.module,
            self.measurement_setting,
        ).value

    @property
    def object_name(self) -> str | None:
        return CalculateMathObjectSetting(
            self.module,
            self.object_setting,
        ).object_name

    @property
    def multiplicand(self) -> Any:
        return TypedCalculateMathSettingValue(
            self.module,
            self.binder,
            "Multiply the above operand by",
            "1.0",
            self.operand_index,
        ).value

    @property
    def exponent(self) -> Any:
        return TypedCalculateMathSettingValue(
            self.module,
            self.binder,
            "Raise the power of above operand by",
            "1.0",
            self.operand_index,
        ).value


@dataclass(frozen=True, slots=True)
class CalculateMathBoundSettings:
    """Runtime kwargs for absorbed CalculateMath execution."""

    module: ModuleBlock
    binder: SettingsBinder

    @property
    def operand1(self) -> CalculateMathOperandSettings:
        return CalculateMathOperandSettings(
            module=self.module,
            binder=self.binder,
            object_setting=NUMERATOR_OBJECTS_SETTING,
            measurement_setting=NUMERATOR_MEASUREMENT_SETTING,
            operand_index=0,
        )

    @property
    def operand2(self) -> CalculateMathOperandSettings:
        return CalculateMathOperandSettings(
            module=self.module,
            binder=self.binder,
            object_setting=DENOMINATOR_OBJECTS_SETTING,
            measurement_setting=DENOMINATOR_MEASUREMENT_SETTING,
            operand_index=1,
        )

    def typed_setting(self, setting_name: str, default: str) -> Any:
        return TypedCalculateMathSettingValue(
            self.module,
            self.binder,
            setting_name,
            default,
        ).value

    @property
    def kwargs(self) -> dict[str, Any]:
        return {
            "output_name": CalculateMathSettingValue(
                self.module,
                OUTPUT_MEASUREMENT_SETTING,
                default="Measurement",
            ).value,
            "operation": CalculateMathSettingValue(
                self.module,
                OPERATION_SETTING,
                default="None",
            ).value,
            "operand1_feature": self.operand1.feature_name,
            "operand2_feature": self.operand2.feature_name,
            "operand1_object_name": self.operand1.object_name,
            "operand2_object_name": self.operand2.object_name,
            "operand1_multiplicand": self.operand1.multiplicand,
            "operand1_exponent": self.operand1.exponent,
            "operand2_multiplicand": self.operand2.multiplicand,
            "operand2_exponent": self.operand2.exponent,
            "take_log10": self.typed_setting("Take log10 of result?", "No"),
            "final_multiplicand": self.typed_setting("Multiply the result by", "1.0"),
            "final_exponent": self.typed_setting(
                "Raise the power of result by",
                "1.0",
            ),
            "final_addend": self.typed_setting("Add to the result", "0.0"),
            "rounding": CalculateMathSettingValue(
                self.module,
                "How should the output value be rounded?",
                default="Not rounded",
            ).value,
            "rounding_digits": self.typed_setting(
                "Enter how many decimal places the value should be rounded to",
                "0",
            ),
            "constrain_lower_bound": self.typed_setting(
                "Constrain the result to a lower bound?",
                "No",
            ),
            "lower_bound": self.typed_setting("Enter the lower bound", "0.0"),
            "constrain_upper_bound": self.typed_setting(
                "Constrain the result to an upper bound?",
                "No",
            ),
            "upper_bound": self.typed_setting("Enter the upper bound", "1.0"),
        }


@dataclass(frozen=True, slots=True)
class CalculateMathObjectDependencies:
    """Object dependencies referenced by CalculateMath measurement operands."""

    module: ModuleBlock

    @property
    def object_names(self) -> tuple[str, ...]:
        names = (
            CalculateMathObjectSetting(
                self.module,
                NUMERATOR_OBJECTS_SETTING,
            ).object_name,
            CalculateMathObjectSetting(
                self.module,
                DENOMINATOR_OBJECTS_SETTING,
            ).object_name,
            count_feature_object_name(
                optional_setting_value(self.module, NUMERATOR_MEASUREMENT_SETTING)
            ),
            count_feature_object_name(
                optional_setting_value(self.module, DENOMINATOR_MEASUREMENT_SETTING)
            ),
        )
        return tuple(dict.fromkeys(name for name in names if name is not None))


def calculate_math_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return absorbed-function kwargs for runtime CalculateMath operands."""
    return CalculateMathBoundSettings(module=module, binder=binder).kwargs


def calculate_math_object_dependencies(module: ModuleBlock) -> tuple[str, ...]:
    """Return object names referenced by CalculateMath measurement operands."""
    return CalculateMathObjectDependencies(module).object_names

