"""Module-level settings-to-kwargs translation for generated CellProfiler steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from .gray_to_color_settings import (
    GRAY_TO_COLOR_CMYK_IMAGE_SETTINGS,
    GRAY_TO_COLOR_CMYK_WEIGHT_SETTINGS,
    GRAY_TO_COLOR_RGB_IMAGE_SETTINGS,
    GRAY_TO_COLOR_RGB_WEIGHT_SETTINGS,
    GrayToColorScheme,
    coerce_gray_to_color_scheme,
    gray_to_color_stack_channels,
    is_blank_gray_to_color_source,
)
from .parser import ModuleBlock
from .settings_binder import SettingsBinder


@dataclass(frozen=True, slots=True)
class BoundModuleSettings:
    """Typed module-setting translation result."""

    kwargs: Mapping[str, Any]
    unmapped_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "unmapped_kwargs", dict(self.unmapped_kwargs))


class ModuleSettingsBindingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for converting one module's settings into function kwargs."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "ModuleSettingsBindingStrategy":
        strategy_type = cls.__registry__.get(
            module_name,
            GenericModuleSettingsBindingStrategy,
        )
        return strategy_type()

    @abstractmethod
    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        """Bind one parsed module into generated function kwargs."""


class GenericModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Default docstring-mapped module-setting binder."""

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        return _translate_bound_kwargs(
            binder.bind(module.settings),
            param_mapping,
        )


class GrayToColorModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Scheme-aware binder for GrayToColor's closed family of input layouts."""

    module_name = "GrayToColor"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        scheme = coerce_gray_to_color_scheme(
            module.get_setting("Select a color scheme", GrayToColorScheme.RGB.value)
        )
        return GrayToColorSchemeBindingStrategy.for_scheme(scheme).bind(
            module,
            binder=binder,
        )


class GrayToColorSchemeBindingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Closed nominal family for GrayToColor scheme-specific kwarg lowering."""

    __registry_key__ = "scheme_literal"
    __skip_if_no_key__ = True
    scheme_literal: ClassVar[str | None] = None

    @classmethod
    def for_scheme(
        cls,
        scheme: GrayToColorScheme,
    ) -> "GrayToColorSchemeBindingStrategy":
        strategy_type = cls.__registry__.get(scheme.value)
        if strategy_type is None:
            raise ValueError(f"Unsupported GrayToColor scheme: {scheme.value!r}")
        return strategy_type()

    @abstractmethod
    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        """Bind one GrayToColor module for a specific color scheme."""


class _GrayToColorIndexedSchemeBindingStrategy(GrayToColorSchemeBindingStrategy):
    """Base class for schemes whose payload is indexed into ordered image planes."""

    image_settings: ClassVar[tuple[tuple[str, str], ...]] = ()
    weight_settings: ClassVar[tuple[tuple[str, str], ...]] = ()

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        kwargs: dict[str, Any] = {
            "color_scheme": type(self).scheme_literal,
            "rescale_intensity": _typed_setting_value(
                binder,
                "Rescale intensity",
                module.get_setting("Rescale intensity", "Yes"),
            ),
        }
        channel_index = 0
        for setting_name, parameter_name in type(self).image_settings:
            kwargs[parameter_name] = -1
            image_name = module.get_setting(setting_name, "").strip()
            if is_blank_gray_to_color_source(image_name):
                continue
            kwargs[parameter_name] = channel_index
            channel_index += 1
        for setting_name, parameter_name in type(self).weight_settings:
            kwargs[parameter_name] = _typed_setting_value(
                binder,
                setting_name,
                module.get_setting(setting_name, "1.0"),
            )
        return BoundModuleSettings(kwargs)


class GrayToColorRgbBindingStrategy(_GrayToColorIndexedSchemeBindingStrategy):
    scheme_literal = GrayToColorScheme.RGB.value
    image_settings = tuple(
        zip(
            GRAY_TO_COLOR_RGB_IMAGE_SETTINGS,
            ("red_channel", "green_channel", "blue_channel"),
            strict=True,
        )
    )
    weight_settings = tuple(
        zip(
            GRAY_TO_COLOR_RGB_WEIGHT_SETTINGS,
            ("red_weight", "green_weight", "blue_weight"),
            strict=True,
        )
    )


class GrayToColorCmykBindingStrategy(_GrayToColorIndexedSchemeBindingStrategy):
    scheme_literal = GrayToColorScheme.CMYK.value
    image_settings = tuple(
        zip(
            GRAY_TO_COLOR_CMYK_IMAGE_SETTINGS,
            ("cyan_channel", "magenta_channel", "yellow_channel", "gray_channel"),
            strict=True,
        )
    )
    weight_settings = tuple(
        zip(
            GRAY_TO_COLOR_CMYK_WEIGHT_SETTINGS,
            ("cyan_weight", "magenta_weight", "yellow_weight", "gray_weight"),
            strict=True,
        )
    )


class _GrayToColorStackFamilyBindingStrategy(GrayToColorSchemeBindingStrategy):
    """Base class for Stack/Composite repeated per-image settings."""

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        channels = gray_to_color_stack_channels(module)
        kwargs: dict[str, Any] = {
            "color_scheme": type(self).scheme_literal,
            "rescale_intensity": _typed_setting_value(
                binder,
                "Rescale intensity",
                module.get_setting("Rescale intensity", "Yes"),
            ),
            "channel_weights": tuple(
                float(_typed_setting_value(binder, "Weight", channel.weight))
                for channel in channels
            ),
        }
        if type(self).scheme_literal == GrayToColorScheme.COMPOSITE.value:
            kwargs["channel_colors"] = tuple(
                channel.color for channel in channels
            )
        return BoundModuleSettings(kwargs)


class GrayToColorStackBindingStrategy(_GrayToColorStackFamilyBindingStrategy):
    scheme_literal = GrayToColorScheme.STACK.value


class GrayToColorCompositeBindingStrategy(_GrayToColorStackFamilyBindingStrategy):
    scheme_literal = GrayToColorScheme.COMPOSITE.value


def _translate_bound_kwargs(
    kwargs: Mapping[str, Any],
    param_mapping: Mapping[str, Any],
) -> BoundModuleSettings:
    translated_kwargs: dict[str, Any] = {}
    unmapped_kwargs: dict[str, Any] = {}

    for cp_setting, value in kwargs.items():
        if cp_setting in param_mapping:
            py_param = param_mapping[cp_setting]
            if py_param is None:
                continue
            if isinstance(py_param, list):
                if isinstance(value, tuple) and len(value) == len(py_param):
                    for index, param_name in enumerate(py_param):
                        translated_kwargs[param_name] = value[index]
                else:
                    translated_kwargs[py_param[0]] = value
            else:
                translated_kwargs[py_param] = value
            continue
        unmapped_kwargs[cp_setting] = value

    return BoundModuleSettings(
        kwargs=translated_kwargs,
        unmapped_kwargs=unmapped_kwargs,
    )


def _typed_setting_value(
    binder: SettingsBinder,
    key: str,
    value: str,
) -> Any:
    return binder.parse_value(key, value)
