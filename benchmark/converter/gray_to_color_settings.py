"""Shared GrayToColor setting semantics for converter compilation/generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from .parser import ModuleBlock


class GrayToColorScheme(str, Enum):
    """Closed family of GrayToColor scheme literals preserved from CellProfiler."""

    RGB = "RGB"
    CMYK = "CMYK"
    STACK = "Stack"
    COMPOSITE = "Composite"


@dataclass(frozen=True, slots=True)
class GrayToColorStackChannelSetting:
    """One repeated stack/composite channel block from CellProfiler settings."""

    image_name: str
    color: str
    weight: str


GRAY_TO_COLOR_RGB_IMAGE_SETTINGS = (
    "Select the image to be colored red",
    "Select the image to be colored green",
    "Select the image to be colored blue",
)
GRAY_TO_COLOR_RGB_WEIGHT_SETTINGS = (
    "Relative weight for the red image",
    "Relative weight for the green image",
    "Relative weight for the blue image",
)
GRAY_TO_COLOR_CMYK_IMAGE_SETTINGS = (
    "Select the image to be colored cyan",
    "Select the image to be colored magenta",
    "Select the image to be colored yellow",
    "Select the image that determines brightness",
)
GRAY_TO_COLOR_CMYK_WEIGHT_SETTINGS = (
    "Relative weight for the cyan image",
    "Relative weight for the magenta image",
    "Relative weight for the yellow image",
    "Relative weight for the brightness image",
)
GRAY_TO_COLOR_RESCALE_SETTING = "Rescale intensity"
GRAY_TO_COLOR_CURRENT_RESCALE_DEFAULT = "Yes"
GRAY_TO_COLOR_REVISION_3_UPGRADED_RESCALE_DEFAULT = "No"


def coerce_gray_to_color_scheme(
    value: GrayToColorScheme | str,
) -> GrayToColorScheme:
    """Parse one stored CellProfiler scheme literal into the closed enum."""
    if isinstance(value, GrayToColorScheme):
        return value
    normalized = value.strip()
    for scheme in GrayToColorScheme:
        if scheme.value == normalized:
            return scheme
    raise ValueError(f"Unsupported GrayToColor scheme: {value!r}")


def gray_to_color_scheme(module: ModuleBlock) -> GrayToColorScheme:
    """Return the declared GrayToColor scheme for one parsed module."""
    return coerce_gray_to_color_scheme(
        module.get_setting("Select a color scheme", GrayToColorScheme.RGB.value)
    )


def gray_to_color_rescale_default(module: ModuleBlock) -> str:
    """Return the CP-upgraded default for a missing GrayToColor rescale setting."""
    if (
        module.variable_revision_number is not None
        and module.variable_revision_number <= 3
    ):
        return GRAY_TO_COLOR_REVISION_3_UPGRADED_RESCALE_DEFAULT
    return GRAY_TO_COLOR_CURRENT_RESCALE_DEFAULT


class GrayToColorInputNameResolver(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for GrayToColor image-input discovery by scheme."""

    __registry_key__ = "scheme_literal"
    __skip_if_no_key__ = True
    scheme_literal: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module: ModuleBlock) -> "GrayToColorInputNameResolver":
        scheme = gray_to_color_scheme(module)
        resolver_type = cls.__registry__.get(scheme.value)
        if resolver_type is None:
            raise ValueError(f"Unsupported GrayToColor scheme: {scheme.value!r}")
        return resolver_type()

    @abstractmethod
    def input_names(self, module: ModuleBlock) -> tuple[str, ...]:
        """Return ordered nonblank source image names for one GrayToColor module."""


class _FixedSettingGrayToColorInputNameResolver(GrayToColorInputNameResolver):
    """Scheme resolver backed by a fixed ordered setting family."""

    image_settings: ClassVar[tuple[str, ...]] = ()

    def input_names(self, module: ModuleBlock) -> tuple[str, ...]:
        return tuple(
            image_name
            for setting_name in type(self).image_settings
            if (
                image_name := _normalized_source_name(
                    module.get_setting(setting_name, "")
                )
            )
            is not None
        )


class GrayToColorRgbInputNameResolver(_FixedSettingGrayToColorInputNameResolver):
    scheme_literal = GrayToColorScheme.RGB.value
    image_settings = GRAY_TO_COLOR_RGB_IMAGE_SETTINGS


class GrayToColorCmykInputNameResolver(_FixedSettingGrayToColorInputNameResolver):
    scheme_literal = GrayToColorScheme.CMYK.value
    image_settings = GRAY_TO_COLOR_CMYK_IMAGE_SETTINGS


class _RepeatedImageNameGrayToColorInputNameResolver(GrayToColorInputNameResolver):
    """Base resolver for Stack/Composite repeated channel settings."""

    def input_names(self, module: ModuleBlock) -> tuple[str, ...]:
        return tuple(
            channel.image_name
            for channel in gray_to_color_stack_channels(module)
        )


class GrayToColorStackInputNameResolver(
    _RepeatedImageNameGrayToColorInputNameResolver
):
    scheme_literal = GrayToColorScheme.STACK.value


class GrayToColorCompositeInputNameResolver(
    _RepeatedImageNameGrayToColorInputNameResolver
):
    scheme_literal = GrayToColorScheme.COMPOSITE.value


def gray_to_color_stack_channels(
    module: ModuleBlock,
) -> tuple[GrayToColorStackChannelSetting, ...]:
    """Parse repeated stack/composite channel groups from ordered .cppipe settings."""
    channels: list[GrayToColorStackChannelSetting] = []
    image_name: str | None = None
    color = "#ff0000"
    weight = "1.0"
    for setting in module.iter_settings():
        if setting.name == "Image name":
            if image_name is not None and not is_blank_gray_to_color_source(image_name):
                channels.append(
                    GrayToColorStackChannelSetting(
                        image_name=image_name,
                        color=color,
                        weight=weight,
                    )
                )
            image_name = setting.value.strip()
            color = "#ff0000"
            weight = "1.0"
            continue
        if image_name is None:
            continue
        if setting.name == "Color":
            color = setting.value.strip()
            continue
        if setting.name == "Weight":
            weight = setting.value.strip()
    if image_name is not None and not is_blank_gray_to_color_source(image_name):
        channels.append(
            GrayToColorStackChannelSetting(
                image_name=image_name,
                color=color,
                weight=weight,
            )
        )
    return tuple(channels)


def is_blank_gray_to_color_source(value: str) -> bool:
    """Whether one GrayToColor input literal means 'unused'."""
    return value.strip().lower() in {"", "leave this black", "none", "do not use"}


def _normalized_source_name(value: str) -> str | None:
    normalized = value.strip()
    if is_blank_gray_to_color_source(normalized):
        return None
    if not normalized:
        return None
    return normalized
