"""Typed ColorToGray setting semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

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


class ColorToGrayConversionMethod(str, Enum):
    """CellProfiler ColorToGray conversion families."""

    COMBINE = "combine"
    SPLIT = "split"


class ColorToGrayImageType(str, Enum):
    """CellProfiler ColorToGray input color interpretations."""

    RGB = "rgb"
    HSV = "hsv"
    CHANNELS = "channels"


@dataclass(frozen=True, slots=True)
class ColorToGrayPlan:
    """Compiled ColorToGray settings used by contracts and generated kwargs."""

    input_image_name: str
    output_image_names: tuple[str, ...]
    mode: ColorToGrayConversionMethod
    image_type: ColorToGrayImageType
    channel_indices: tuple[int, ...]
    contributions: tuple[float, ...]

    @property
    def kwargs(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "image_type": self.image_type.value,
            "channel_indices": self.channel_indices,
            "contributions": self.contributions,
        }


def color_to_gray_plan(module: ModuleBlock, binder: SettingsBinder) -> ColorToGrayPlan:
    """Compile ColorToGray's mode-dependent live outputs and function kwargs."""

    mode = _conversion_method(module)
    image_type = _image_type(module)
    input_image_name = required_setting_value(module, COLOR_TO_GRAY_INPUT_IMAGE_SETTING)
    output_names = _output_image_names(module, mode, image_type, binder)
    channel_indices = ColorToGrayImageTypeSettingsStrategy.for_image_type(
        image_type
    ).channel_indices(module, mode, binder)
    return ColorToGrayPlan(
        input_image_name=input_image_name,
        output_image_names=output_names,
        mode=mode,
        image_type=image_type,
        channel_indices=channel_indices,
        contributions=_contributions(module, mode, channel_indices, binder),
    )


def color_to_gray_input_name(module: ModuleBlock) -> str:
    return required_setting_value(module, COLOR_TO_GRAY_INPUT_IMAGE_SETTING)


def color_to_gray_output_names(module: ModuleBlock) -> tuple[str, ...]:
    return color_to_gray_plan(module, SettingsBinder()).output_image_names


def color_to_gray_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, object]:
    return color_to_gray_plan(module, binder).kwargs


def _conversion_method(module: ModuleBlock) -> ColorToGrayConversionMethod:
    value = required_setting_value(module, "Conversion method")
    return _coerce_enum(ColorToGrayConversionMethod, value, "Conversion method")


def _image_type(module: ModuleBlock) -> ColorToGrayImageType:
    value = required_setting_value(module, "Image type")
    return _coerce_enum(ColorToGrayImageType, value, "Image type")


def _output_image_names(
    module: ModuleBlock,
    mode: ColorToGrayConversionMethod,
    image_type: ColorToGrayImageType,
    binder: SettingsBinder,
) -> tuple[str, ...]:
    if mode is ColorToGrayConversionMethod.COMBINE:
        return (
            required_setting_value(module, COLOR_TO_GRAY_OUTPUT_IMAGE_SETTING),
        )
    return ColorToGrayImageTypeSettingsStrategy.for_image_type(
        image_type
    ).split_output_names(module, binder)


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
        return _flagged_outputs(
            module,
            binder,
            output_offset=self.output_offset,
            flags=self.output_flags,
        )

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
            if _truthy(module, flag, binder)
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
        channel_numbers = setting_values(module, "Channel number")
        if not channel_numbers:
            return (0,)
        indices = tuple(_channel_index(channel_number) for channel_number in channel_numbers)
        if mode is ColorToGrayConversionMethod.SPLIT:
            output_count = len(
                setting_values(module, COLOR_TO_GRAY_CHANNEL_OUTPUT_IMAGE_SETTING)
            )
            return indices[:output_count]
        return indices


def _flagged_outputs(
    module: ModuleBlock,
    binder: SettingsBinder,
    *,
    output_offset: int,
    flags: tuple[str, ...],
) -> tuple[str, ...]:
    output_names = setting_values(module, COLOR_TO_GRAY_OUTPUT_IMAGE_SETTING)
    selected = tuple(
        output_names[output_offset + index]
        for index, flag in enumerate(flags)
        if _truthy(module, flag, binder)
    )
    if not selected:
        raise ValueError(
            f"ColorToGray({module.module_num}) split mode must declare at least "
            "one enabled output channel."
        )
    return selected


def _contributions(
    module: ModuleBlock,
    mode: ColorToGrayConversionMethod,
    channel_indices: tuple[int, ...],
    binder: SettingsBinder,
) -> tuple[float, ...]:
    if mode is ColorToGrayConversionMethod.SPLIT:
        return tuple(1.0 for _index in channel_indices)
    if len(channel_indices) == 3:
        return tuple(
            float(binder.parse_value(setting, required_setting_value(module, setting)))
            for setting in (
                "Relative weight of the red channel",
                "Relative weight of the green channel",
                "Relative weight of the blue channel",
            )
        )
    return tuple(
        float(binder.parse_value("Relative weight of the channel", value))
        for value in setting_values(module, "Relative weight of the channel")
    )


def _truthy(module: ModuleBlock, setting: str, binder: SettingsBinder) -> bool:
    return bool(binder.parse_value(setting, required_setting_value(module, setting)))


def _channel_index(value: str) -> int:
    match = re.search(r"([0-9]+)$", value.strip())
    if match is None:
        raise ValueError(f"ColorToGray channel number lacks an integer suffix: {value!r}")
    return int(match.group(1)) - 1


def _coerce_enum[T: Enum](
    enum_type: type[T],
    value: str,
    setting_name: str,
) -> T:
    normalized = value.strip().lower()
    for option in enum_type:
        if normalized in {option.name.lower(), str(option.value).lower()}:
            return option
    raise ValueError(f"Unsupported ColorToGray {setting_name}: {value!r}")
