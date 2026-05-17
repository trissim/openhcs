"""Shared GrayToColor setting semantics for CellProfiler interop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.public_api import declared_public_names

from .parser import ModuleBlock
from .setting_names import is_blank_symbol_name, normalized_symbol_name


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


@dataclass(frozen=True, slots=True)
class GrayToColorModuleRevision:
    """Versioned GrayToColor defaults derived from the parsed module revision."""

    module: ModuleBlock

    @property
    def rescale_default(self) -> str:
        revision = self.module.variable_revision_number
        if revision is not None and revision <= 3:
            return GRAY_TO_COLOR_REVISION_3_UPGRADED_RESCALE_DEFAULT
        return GRAY_TO_COLOR_CURRENT_RESCALE_DEFAULT


def gray_to_color_rescale_default(module: ModuleBlock) -> str:
    """Return the CP-upgraded default for a missing GrayToColor rescale setting."""
    return GrayToColorModuleRevision(module).rescale_default


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


class FixedSettingGrayToColorInputNameResolver(GrayToColorInputNameResolver):
    """Scheme resolver backed by a fixed ordered setting family."""

    image_settings: ClassVar[tuple[str, ...]] = ()

    def input_names(self, module: ModuleBlock) -> tuple[str, ...]:
        return tuple(
            image_name
            for setting_name in type(self).image_settings
            if (
                image_name := normalized_symbol_name(
                    module.get_setting(setting_name, "")
                )
            )
            is not None
        )


class RepeatedImageNameGrayToColorInputNameResolver(GrayToColorInputNameResolver):
    """Base resolver for Stack/Composite repeated channel settings."""

    def input_names(self, module: ModuleBlock) -> tuple[str, ...]:
        return tuple(
            channel.image_name
            for channel in gray_to_color_stack_channels(module)
        )


@dataclass(frozen=True, slots=True)
class GrayToColorInputNameResolverDeclaration:
    """Typed declaration for one GrayToColor input-name resolver class."""

    class_name: str
    scheme: GrayToColorScheme
    base: type[GrayToColorInputNameResolver]
    image_settings: tuple[str, ...] = ()

    def materialize(self) -> type[GrayToColorInputNameResolver]:
        return type(
            self.class_name,
            (self.base,),
            {
                "scheme_literal": self.scheme.value,
                "image_settings": self.image_settings,
                "__module__": __name__,
            },
        )


GRAY_TO_COLOR_INPUT_NAME_RESOLVER_DECLARATIONS: tuple[
    GrayToColorInputNameResolverDeclaration,
    ...,
] = (
    GrayToColorInputNameResolverDeclaration(
        "GrayToColorRgbInputNameResolver",
        GrayToColorScheme.RGB,
        FixedSettingGrayToColorInputNameResolver,
        GRAY_TO_COLOR_RGB_IMAGE_SETTINGS,
    ),
    GrayToColorInputNameResolverDeclaration(
        "GrayToColorCmykInputNameResolver",
        GrayToColorScheme.CMYK,
        FixedSettingGrayToColorInputNameResolver,
        GRAY_TO_COLOR_CMYK_IMAGE_SETTINGS,
    ),
    GrayToColorInputNameResolverDeclaration(
        "GrayToColorStackInputNameResolver",
        GrayToColorScheme.STACK,
        RepeatedImageNameGrayToColorInputNameResolver,
    ),
    GrayToColorInputNameResolverDeclaration(
        "GrayToColorCompositeInputNameResolver",
        GrayToColorScheme.COMPOSITE,
        RepeatedImageNameGrayToColorInputNameResolver,
    ),
)

globals().update(
    {
        declaration.class_name: declaration.materialize()
        for declaration in GRAY_TO_COLOR_INPUT_NAME_RESOLVER_DECLARATIONS
    }
)


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
            if (
                normalized_image_name := normalized_symbol_name(image_name or "")
            ) is not None:
                channels.append(
                    GrayToColorStackChannelSetting(
                        image_name=normalized_image_name,
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
    if (
        normalized_image_name := normalized_symbol_name(image_name or "")
    ) is not None:
        channels.append(
            GrayToColorStackChannelSetting(
                image_name=normalized_image_name,
                color=color,
                weight=weight,
            )
        )
    return tuple(channels)


def is_blank_gray_to_color_source(value: str) -> bool:
    """Whether one GrayToColor input literal means 'unused'."""
    return is_blank_symbol_name(value)


__all__ = declared_public_names(globals(), constant_prefixes=("GRAY_TO_COLOR_",))
