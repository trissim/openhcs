"""Typed ColorToGray setting semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from .cellprofiler_literals import cellprofiler_enum_from_literal
from .parser import ModuleBlock
from .setting_names import (
    SettingNameFamily,
    required_setting_value,
    setting_values,
)
from .settings_binder import SettingsBinder


COLOR_TO_GRAY_INPUT_IMAGE_SETTING = SettingNameFamily("Select the input image")
COLOR_TO_GRAY_OUTPUT_IMAGE_SETTING = SettingNameFamily("Name the output image")
COLOR_TO_GRAY_CHANNEL_OUTPUT_IMAGE_SETTING = SettingNameFamily("Image name")
COLOR_TO_GRAY_CONVERSION_METHOD_SETTING = "Conversion method"
COLOR_TO_GRAY_IMAGE_TYPE_SETTING = "Image type"
COLOR_TO_GRAY_CHANNEL_NUMBER_SETTING = "Channel number"
COLOR_TO_GRAY_CHANNEL_WEIGHT_SETTING = "Relative weight of the channel"
COLOR_TO_GRAY_RGB_WEIGHT_SETTINGS = (
    "Relative weight of the red channel",
    "Relative weight of the green channel",
    "Relative weight of the blue channel",
)


class ColorToGrayConversionMethod(str, Enum):
    """CellProfiler ColorToGray conversion families."""

    COMBINE = "combine"
    SPLIT = "split"

    @classmethod
    def from_literal(cls, value: str) -> "ColorToGrayConversionMethod":
        return cellprofiler_enum_from_literal(cls, value)


class ColorToGrayImageType(str, Enum):
    """CellProfiler ColorToGray input color interpretations."""

    RGB = "rgb"
    HSV = "hsv"
    CHANNELS = "channels"

    @classmethod
    def from_literal(cls, value: str) -> "ColorToGrayImageType":
        return cellprofiler_enum_from_literal(cls, value)


@dataclass(frozen=True, slots=True)
class ColorToGrayChannelNumber:
    """One CellProfiler channel-number setting lowered to a zero-based index."""

    literal: str

    @property
    def index(self) -> int:
        match = re.search(r"([0-9]+)$", self.literal.strip())
        if match is None:
            raise ValueError(
                "ColorToGray channel number lacks an integer suffix: "
                f"{self.literal!r}"
            )
        return int(match.group(1)) - 1


@dataclass(frozen=True, slots=True)
class ColorToGrayPlan:
    """Compiled ColorToGray settings used by contracts and generated kwargs."""

    input_image_name: str
    output_image_names: tuple[str, ...]
    mode: ColorToGrayConversionMethod
    image_type: ColorToGrayImageType
    channel_indices: tuple[int, ...]
    contributions: tuple[float, ...]

    @classmethod
    def from_module(
        cls,
        module: ModuleBlock,
        binder: SettingsBinder,
    ) -> "ColorToGrayPlan":
        mode = ColorToGrayConversionMethod.from_literal(
            required_setting_value(module, COLOR_TO_GRAY_CONVERSION_METHOD_SETTING)
        )
        image_type = ColorToGrayImageType.from_literal(
            required_setting_value(module, COLOR_TO_GRAY_IMAGE_TYPE_SETTING)
        )
        input_image_name = required_setting_value(
            module,
            COLOR_TO_GRAY_INPUT_IMAGE_SETTING,
        )
        image_type_settings = ColorToGrayImageTypeSettingsStrategy.for_image_type(
            image_type
        )
        output_names = ColorToGrayOutputPlan(
            module=module,
            mode=mode,
            image_type_settings=image_type_settings,
            binder=binder,
        ).output_image_names
        channel_indices = image_type_settings.channel_indices(module, mode, binder)
        return cls(
            input_image_name=input_image_name,
            output_image_names=output_names,
            mode=mode,
            image_type=image_type,
            channel_indices=channel_indices,
            contributions=ColorToGrayContributionPlan(
                module=module,
                mode=mode,
                channel_indices=channel_indices,
                binder=binder,
            ).contributions,
        )

    @property
    def kwargs(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "image_type": self.image_type.value,
            "channel_indices": self.channel_indices,
            "contributions": self.contributions,
        }


@dataclass(frozen=True, slots=True)
class ColorToGrayModuleBinding(ABC, metaclass=AutoRegisterMeta):
    """Shared CellProfiler module/binder context for ColorToGray settings."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None
    module: ModuleBlock
    binder: SettingsBinder

    @classmethod
    def registered_binding_types(cls) -> tuple[type["ColorToGrayModuleBinding"], ...]:
        return tuple(cls.__registry__.values())


@dataclass(frozen=True, slots=True)
class ColorToGrayModePlan(ColorToGrayModuleBinding, ABC):
    """Shared mode-aware context for ColorToGray setting plans."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None
    mode: ColorToGrayConversionMethod

    @classmethod
    def registered_mode_plan_types(cls) -> tuple[type["ColorToGrayModePlan"], ...]:
        return tuple(cls.__registry__.values())


@dataclass(frozen=True, slots=True)
class ColorToGrayOutputPlan(ColorToGrayModePlan):
    """Mode-dependent ColorToGray output-image semantics."""

    registry_key: ClassVar[str] = "output_plan"
    image_type_settings: "ColorToGrayImageTypeSettingsStrategy"

    @property
    def output_image_names(self) -> tuple[str, ...]:
        if self.mode is ColorToGrayConversionMethod.COMBINE:
            return (
                required_setting_value(
                    self.module,
                    COLOR_TO_GRAY_OUTPUT_IMAGE_SETTING,
                ),
            )
        return self.image_type_settings.split_output_names(
            self.module,
            self.binder,
        )


@dataclass(frozen=True, slots=True)
class ColorToGrayContributionPlan(ColorToGrayModePlan):
    """Mode- and channel-dependent ColorToGray contribution weights."""

    registry_key: ClassVar[str] = "contribution_plan"
    channel_indices: tuple[int, ...]

    @property
    def contributions(self) -> tuple[float, ...]:
        if self.mode is ColorToGrayConversionMethod.SPLIT:
            return tuple(1.0 for _index in self.channel_indices)
        if len(self.channel_indices) == 3:
            return tuple(
                float(
                    self.binder.parse_value(
                        setting,
                        required_setting_value(self.module, setting),
                    )
                )
                for setting in COLOR_TO_GRAY_RGB_WEIGHT_SETTINGS
            )
        return tuple(
            float(
                self.binder.parse_value(
                    COLOR_TO_GRAY_CHANNEL_WEIGHT_SETTING,
                    value,
                )
            )
            for value in setting_values(
                self.module,
                COLOR_TO_GRAY_CHANNEL_WEIGHT_SETTING,
            )
        )


def color_to_gray_plan(module: ModuleBlock, binder: SettingsBinder) -> ColorToGrayPlan:
    """Compile ColorToGray's mode-dependent live outputs and function kwargs."""
    return ColorToGrayPlan.from_module(module, binder)


def color_to_gray_input_name(module: ModuleBlock) -> str:
    return required_setting_value(module, COLOR_TO_GRAY_INPUT_IMAGE_SETTING)


def color_to_gray_output_names(module: ModuleBlock) -> tuple[str, ...]:
    return color_to_gray_plan(module, SettingsBinder()).output_image_names


def color_to_gray_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, object]:
    return color_to_gray_plan(module, binder).kwargs


class ColorToGrayImageTypeSettingsStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal ColorToGray setting semantics for one input image type."""

    __registry_key__ = "image_type_literal"
    __skip_if_no_key__ = True
    image_type_literal: ClassVar[str | None] = None

    @classmethod
    def for_image_type(
        cls,
        image_type: ColorToGrayImageType,
    ) -> "ColorToGrayImageTypeSettingsStrategy":
        strategy_type = cls.__registry__.get(image_type.value)
        if strategy_type is None:
            raise ValueError(f"Unsupported ColorToGray image type: {image_type.value!r}")
        return strategy_type()

    @abstractmethod
    def split_output_names(
        self,
        module: ModuleBlock,
        binder: SettingsBinder,
    ) -> tuple[str, ...]:
        """Return the live split-mode outputs for this image type."""

    @abstractmethod
    def channel_indices(
        self,
        module: ModuleBlock,
        mode: ColorToGrayConversionMethod,
        binder: SettingsBinder,
    ) -> tuple[int, ...]:
        """Return input channel indices for this image type."""


class FixedChannelColorToGraySettingsStrategy(ColorToGrayImageTypeSettingsStrategy):
    """Shared RGB/HSV settings where channels are fixed and outputs are flagged."""

    output_offset: ClassVar[int]
    output_flags: ClassVar[tuple[str, ...]]

    def split_output_names(
        self,
        module: ModuleBlock,
        binder: SettingsBinder,
    ) -> tuple[str, ...]:
        output_names = setting_values(module, COLOR_TO_GRAY_OUTPUT_IMAGE_SETTING)
        selected = tuple(
            output_names[self.output_offset + index]
            for index, flag in enumerate(self.output_flags)
            if ColorToGrayFlag(module=module, binder=binder, setting=flag).enabled
        )
        if not selected:
            raise ValueError(
                f"ColorToGray({module.module_num}) split mode must declare at "
                "least one enabled output channel."
            )
        return selected

    def channel_indices(
        self,
        module: ModuleBlock,
        mode: ColorToGrayConversionMethod,
        binder: SettingsBinder,
    ) -> tuple[int, ...]:
        if mode is ColorToGrayConversionMethod.COMBINE:
            del module, binder
            return (0, 1, 2)
        return tuple(
            index
            for index, flag in enumerate(self.output_flags)
            if ColorToGrayFlag(module=module, binder=binder, setting=flag).enabled
        )


class RgbColorToGraySettingsStrategy(FixedChannelColorToGraySettingsStrategy):
    """RGB ColorToGray settings."""

    image_type_literal = ColorToGrayImageType.RGB.value
    output_offset = 1
    output_flags = (
        "Convert red to gray?",
        "Convert green to gray?",
        "Convert blue to gray?",
    )


class HsvColorToGraySettingsStrategy(FixedChannelColorToGraySettingsStrategy):
    """HSV ColorToGray settings."""

    image_type_literal = ColorToGrayImageType.HSV.value
    output_offset = 4
    output_flags = (
        "Convert hue to gray?",
        "Convert saturation to gray?",
        "Convert value to gray?",
    )


class ChannelsColorToGraySettingsStrategy(ColorToGrayImageTypeSettingsStrategy):
    """Arbitrary-channel ColorToGray settings."""

    image_type_literal = ColorToGrayImageType.CHANNELS.value

    def split_output_names(
        self,
        module: ModuleBlock,
        binder: SettingsBinder,
    ) -> tuple[str, ...]:
        del binder
        return setting_values(module, COLOR_TO_GRAY_CHANNEL_OUTPUT_IMAGE_SETTING)

    def channel_indices(
        self,
        module: ModuleBlock,
        mode: ColorToGrayConversionMethod,
        binder: SettingsBinder,
    ) -> tuple[int, ...]:
        del binder
        channel_numbers = setting_values(module, COLOR_TO_GRAY_CHANNEL_NUMBER_SETTING)
        if not channel_numbers:
            return (0,)
        indices = tuple(
            ColorToGrayChannelNumber(channel_number).index
            for channel_number in channel_numbers
        )
        if mode is ColorToGrayConversionMethod.SPLIT:
            output_count = len(
                setting_values(module, COLOR_TO_GRAY_CHANNEL_OUTPUT_IMAGE_SETTING)
            )
            return indices[:output_count]
        return indices


@dataclass(frozen=True, slots=True)
class ColorToGrayFlag(ColorToGrayModuleBinding):
    """Typed boolean CellProfiler flag lookup for ColorToGray settings."""

    registry_key: ClassVar[str] = "flag"
    setting: str

    @property
    def enabled(self) -> bool:
        return bool(
            self.binder.parse_value(
                self.setting,
                required_setting_value(self.module, self.setting),
            )
        )
