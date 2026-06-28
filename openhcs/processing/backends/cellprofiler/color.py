"""Shared CellProfiler color literal semantics."""

from __future__ import annotations

from openhcs.interop.cellprofiler.setting_names import normalized_symbol_name

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.image_shapes import (
    is_channel_last_image_slice,
    is_channel_last_image_stack,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.processing.backends.cellprofiler.module_classes import (
    ArtifactContractModule,
    BinderSettingsSourceModule,
    CellProfilerModule,
    ImageProcessingDebugViewModule,
    ModuleSettingsSourceModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    block_setting_value,
    optional_setting_value,
    repeating_setting_blocks,
    required_setting_value,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.interop.cellprofiler.runtime.execution_mode_policies import (
    CellProfilerInvocationExecutionModePolicyMixin,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
)
class ChannelCompositeExecutionModePolicy(CellProfilerInvocationExecutionModePolicyMixin):
    """Consume a channel composite instead of independent channel slices."""

    def execution_mode(
        self,
        default: ImagePayloadExecutionMode,
        *,
        image: CellProfilerRuntimeValue,
        kwargs: CellProfilerKwargs,
        invocation_options: RuntimeInvocationOptions | None = None,
    ) -> ImagePayloadExecutionMode:
        del default, image, kwargs, invocation_options
        return ImagePayloadExecutionMode.FULL_STACK


class ColorToGrayModule(ChannelCompositeExecutionModePolicy, ImageProcessingDebugViewModule, BinderSettingsSourceModule):
    module_name = 'ColorToGray'
    function_name = 'color_to_gray'
    validated = True
    contract = 'flexible'
    category = 'channel_operation'
    confidence = 1.0
    input_image_setting = SettingNameFamily("Select the input image")
    output_image_setting = SettingNameFamily("Name the output image")
    channel_output_image_setting = SettingNameFamily("Image name")
    conversion_method_setting = "Conversion method"
    image_type_setting = "Image type"
    channel_number_setting = "Channel number"
    channel_weight_setting = "Relative weight of the channel"
    rgb_weight_settings = (
        "Relative weight of the red channel",
        "Relative weight of the green channel",
        "Relative weight of the blue channel",
    )
    rgb_output_offset = 1
    rgb_output_flags = (
        "Convert red to gray?",
        "Convert green to gray?",
        "Convert blue to gray?",
    )
    hsv_output_offset = 4
    hsv_output_flags = (
        "Convert hue to gray?",
        "Convert saturation to gray?",
        "Convert value to gray?",
    )

    class ConversionMethod(str, Enum):
        COMBINE = "combine"
        SPLIT = "split"

    class ImageType(str, Enum):
        RGB = "rgb"
        HSV = "hsv"
        CHANNELS = "channels"

    @dataclass(frozen=True, slots=True)
    class Plan:
        input_image_name: str
        output_image_names: tuple[str, ...]
        mode: "ColorToGrayModule.ConversionMethod"
        image_type: "ColorToGrayModule.ImageType"
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

    @classmethod
    def settings_source(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
    ) -> "CellProfilerKwargs":
        return cls.plan(module, binder).kwargs

    @classmethod
    def plan(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
    ) -> "ColorToGrayModule.Plan":
        mode = cls.conversion_method(module)
        image_type = cls.image_type(module)
        channel_indices = cls.channel_indices(module, mode, image_type, binder)
        return cls.Plan(
            input_image_name=cls.input_name(module),
            output_image_names=cls.output_image_names(module, mode, image_type, binder),
            mode=mode,
            image_type=image_type,
            channel_indices=channel_indices,
            contributions=cls.contributions(module, mode, channel_indices, binder),
        )

    @classmethod
    def input_name(cls, module: "ModuleBlock") -> str:
        return required_setting_value(module, cls.input_image_setting)

    @classmethod
    def output_image_names(
        cls,
        module: "ModuleBlock",
        mode: "ColorToGrayModule.ConversionMethod | None" = None,
        image_type: "ColorToGrayModule.ImageType | None" = None,
        binder: "SettingsBinder | None" = None,
    ) -> tuple[str, ...]:
        if binder is None:
            from openhcs.interop.cellprofiler.settings_binder import SettingsBinder

            binder = SettingsBinder()
        if mode is None:
            mode = cls.conversion_method(module)
        if image_type is None:
            image_type = cls.image_type(module)
        if mode is cls.ConversionMethod.COMBINE:
            return (required_setting_value(module, cls.output_image_setting),)
        if image_type is cls.ImageType.CHANNELS:
            return setting_values(module, cls.channel_output_image_setting)
        output_offset, output_flags = cls.fixed_channel_output_settings(image_type)
        output_names = setting_values(module, cls.output_image_setting)
        selected = tuple(
            output_names[output_offset + index]
            for index, flag in enumerate(output_flags)
            if cls.flag_enabled(module, binder, flag)
        )
        if not selected:
            raise ValueError(
                f"ColorToGray({module.module_num}) split mode must declare at "
                "least one enabled output channel."
            )
        return selected

    @classmethod
    def conversion_method(
        cls,
        module: "ModuleBlock",
    ) -> "ColorToGrayModule.ConversionMethod":
        return coerce_cellprofiler_enum(
            cls.ConversionMethod,
            required_setting_value(module, cls.conversion_method_setting),
        )

    @classmethod
    def image_type(cls, module: "ModuleBlock") -> "ColorToGrayModule.ImageType":
        return coerce_cellprofiler_enum(
            cls.ImageType,
            required_setting_value(module, cls.image_type_setting),
        )

    @classmethod
    def channel_indices(
        cls,
        module: "ModuleBlock",
        mode: "ColorToGrayModule.ConversionMethod",
        image_type: "ColorToGrayModule.ImageType",
        binder: "SettingsBinder",
    ) -> tuple[int, ...]:
        if image_type is cls.ImageType.CHANNELS:
            channel_numbers = setting_values(module, cls.channel_number_setting)
            if not channel_numbers:
                return (0,)
            indices = tuple(cls.channel_number_index(value) for value in channel_numbers)
            if mode is cls.ConversionMethod.SPLIT:
                output_count = len(
                    setting_values(module, cls.channel_output_image_setting)
                )
                return indices[:output_count]
            return indices

        if mode is cls.ConversionMethod.COMBINE:
            return (0, 1, 2)

        _output_offset, output_flags = cls.fixed_channel_output_settings(image_type)
        return tuple(
            index
            for index, flag in enumerate(output_flags)
            if cls.flag_enabled(module, binder, flag)
        )

    @classmethod
    def contributions(
        cls,
        module: "ModuleBlock",
        mode: "ColorToGrayModule.ConversionMethod",
        channel_indices: tuple[int, ...],
        binder: "SettingsBinder",
    ) -> tuple[float, ...]:
        if mode is cls.ConversionMethod.SPLIT:
            return tuple(1.0 for _index in channel_indices)
        if len(channel_indices) == 3:
            return tuple(
                float(
                    binder.parse_value(
                        setting,
                        required_setting_value(module, setting),
                    )
                )
                for setting in cls.rgb_weight_settings
            )
        return tuple(
            float(
                binder.parse_value(
                    cls.channel_weight_setting,
                    value,
                )
            )
            for value in setting_values(module, cls.channel_weight_setting)
        )

    @classmethod
    def fixed_channel_output_settings(
        cls,
        image_type: "ColorToGrayModule.ImageType",
    ) -> tuple[int, tuple[str, ...]]:
        if image_type is cls.ImageType.RGB:
            return cls.rgb_output_offset, cls.rgb_output_flags
        if image_type is cls.ImageType.HSV:
            return cls.hsv_output_offset, cls.hsv_output_flags
        raise ValueError(f"{image_type.value!r} is not a fixed-channel image type.")

    @classmethod
    def flag_enabled(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
        setting_name: str,
    ) -> bool:
        return bool(
            binder.parse_value(
                setting_name,
                required_setting_value(module, setting_name),
            )
        )

    @staticmethod
    def channel_number_index(literal: str) -> int:
        match = re.search(r"([0-9]+)$", literal.strip())
        if match is None:
            raise ValueError(
                "ColorToGray channel number lacks an integer suffix: "
                f"{literal!r}"
            )
        return int(match.group(1)) - 1

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        from openhcs.core.artifacts import ArtifactKind, ArtifactSpec

        image = builder.require_artifact(ArtifactSpec(cls.input_name(module), ArtifactKind.IMAGE), module)
        outputs = [
            builder.declare_artifact(ArtifactSpec(output_name, ArtifactKind.IMAGE), module)
            for output_name in cls.output_image_names(module)
        ]
        return assembler.assemble_contract(module, builder, inputs=[image], outputs=outputs)


class GrayToColorModule(ImageProcessingDebugViewModule, BinderSettingsSourceModule):
    module_name = 'GrayToColor'
    function_name = 'gray_to_color'
    validated = True
    category = 'channel_operation'
    confidence = 1.0
    color_scheme_setting = SettingNameFamily("Select a color scheme")
    rescale_setting = SettingNameFamily("Rescale intensity")
    current_rescale_default = "Yes"
    revision_3_upgraded_rescale_default = "No"
    stack_image_setting = "Image name"
    stack_color_setting = "Color"
    stack_weight_setting = "Weight"
    stack_default_color = "#ff0000"
    stack_default_weight = "1.0"

    class Scheme(str, Enum):
        RGB = "RGB"
        CMYK = "CMYK"
        STACK = "Stack"
        COMPOSITE = "Composite"

    rgb_image_settings = (
        "Select the image to be colored red",
        "Select the image to be colored green",
        "Select the image to be colored blue",
    )
    rgb_weight_settings = (
        "Relative weight for the red image",
        "Relative weight for the green image",
        "Relative weight for the blue image",
    )
    rgb_channel_parameters = ("red_channel", "green_channel", "blue_channel")
    rgb_weight_parameters = ("red_weight", "green_weight", "blue_weight")
    cmyk_image_settings = (
        "Select the image to be colored cyan",
        "Select the image to be colored magenta",
        "Select the image to be colored yellow",
        "Select the image that determines brightness",
    )
    cmyk_weight_settings = (
        "Relative weight for the cyan image",
        "Relative weight for the magenta image",
        "Relative weight for the yellow image",
        "Relative weight for the brightness image",
    )
    cmyk_channel_parameters = (
        "cyan_channel",
        "magenta_channel",
        "yellow_channel",
        "gray_channel",
    )
    cmyk_weight_parameters = (
        "cyan_weight",
        "magenta_weight",
        "yellow_weight",
        "gray_weight",
    )

    @dataclass(frozen=True, slots=True)
    class StackChannel:
        """One repeated Stack/Composite source channel row."""

        image_name: str
        color: str
        weight: str

    @classmethod
    def settings_source(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
    ) -> "CellProfilerKwargs":
        scheme = cls.scheme(module)
        if scheme is GrayToColorModule.Scheme.RGB:
            return cls.indexed_scheme_kwargs(
                module,
                binder,
                scheme=scheme,
                image_settings=cls.rgb_image_settings,
                channel_parameters=cls.rgb_channel_parameters,
                weight_settings=cls.rgb_weight_settings,
                weight_parameters=cls.rgb_weight_parameters,
            )
        if scheme is GrayToColorModule.Scheme.CMYK:
            return cls.indexed_scheme_kwargs(
                module,
                binder,
                scheme=scheme,
                image_settings=cls.cmyk_image_settings,
                channel_parameters=cls.cmyk_channel_parameters,
                weight_settings=cls.cmyk_weight_settings,
                weight_parameters=cls.cmyk_weight_parameters,
            )
        if scheme in {GrayToColorModule.Scheme.STACK, GrayToColorModule.Scheme.COMPOSITE}:
            return cls.stack_scheme_kwargs(module, binder, scheme=scheme)
        raise ValueError(f"Unsupported GrayToColor scheme: {scheme.value!r}")

    @classmethod
    def scheme(cls, module: "ModuleBlock") -> GrayToColorModule.Scheme:
        return cls.coerce_scheme(
            module.get_setting(
                cls.color_scheme_setting.canonical,
                cls.Scheme.RGB.value,
            )
        )

    @classmethod
    def coerce_scheme(
        cls,
        value: "GrayToColorModule.Scheme | str",
    ) -> "GrayToColorModule.Scheme":
        if isinstance(value, cls.Scheme):
            return value
        normalized = value.strip()
        for scheme in cls.Scheme:
            if scheme.value == normalized:
                return scheme
        raise ValueError(f"Unsupported GrayToColor scheme: {value!r}.")

    @classmethod
    def input_names(cls, module: "ModuleBlock") -> tuple[str, ...]:
        scheme = cls.scheme(module)
        if scheme is GrayToColorModule.Scheme.RGB:
            return cls.indexed_input_names(module, cls.rgb_image_settings)
        if scheme is GrayToColorModule.Scheme.CMYK:
            return cls.indexed_input_names(module, cls.cmyk_image_settings)
        if scheme in {GrayToColorModule.Scheme.STACK, GrayToColorModule.Scheme.COMPOSITE}:
            return tuple(channel.image_name for channel in cls.stack_channels(module))
        raise ValueError(f"Unsupported GrayToColor scheme: {scheme.value!r}")

    @staticmethod
    def indexed_input_names(
        module: "ModuleBlock",
        image_settings: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            image_name
            for setting_name in image_settings
            if (
                image_name := normalized_symbol_name(
                    module.get_setting(setting_name, "")
                )
            )
            is not None
        )

    @classmethod
    def indexed_scheme_kwargs(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
        *,
        scheme: GrayToColorModule.Scheme,
        image_settings: tuple[str, ...],
        channel_parameters: tuple[str, ...],
        weight_settings: tuple[str, ...],
        weight_parameters: tuple[str, ...],
    ) -> "CellProfilerKwargs":
        kwargs: dict[str, Any] = cls.base_kwargs(module, binder, scheme)
        channel_index = 0
        for setting_name, parameter_name in zip(
            image_settings,
            channel_parameters,
            strict=True,
        ):
            kwargs[parameter_name] = -1
            image_name = normalized_symbol_name(module.get_setting(setting_name, ""))
            if image_name is None:
                continue
            kwargs[parameter_name] = channel_index
            channel_index += 1
        for setting_name, parameter_name in zip(
            weight_settings,
            weight_parameters,
            strict=True,
        ):
            kwargs[parameter_name] = float(
                binder.parse_value(
                    setting_name,
                    module.get_setting(setting_name, "1.0"),
                )
            )
        return kwargs

    @classmethod
    def stack_scheme_kwargs(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
        *,
        scheme: GrayToColorModule.Scheme,
    ) -> "CellProfilerKwargs":
        channels = cls.stack_channels(module)
        kwargs: dict[str, Any] = cls.base_kwargs(module, binder, scheme)
        kwargs["channel_weights"] = tuple(
            float(
                binder.parse_value(
                    cls.stack_weight_setting,
                    channel.weight,
                )
            )
            for channel in channels
        )
        if scheme is GrayToColorModule.Scheme.COMPOSITE:
            kwargs["channel_colors"] = tuple(channel.color for channel in channels)
        return kwargs

    @classmethod
    def base_kwargs(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
        scheme: GrayToColorModule.Scheme,
    ) -> dict[str, Any]:
        return {
            "color_scheme": scheme.value,
            "rescale_intensity": bool(
                binder.parse_value(
                    cls.rescale_setting.canonical,
                    module.get_setting(
                        cls.rescale_setting.canonical,
                        cls.rescale_default(module),
                    ),
                )
            ),
        }

    @classmethod
    def rescale_default(cls, module: "ModuleBlock") -> str:
        revision = module.variable_revision_number
        if revision is not None and revision <= 3:
            return cls.revision_3_upgraded_rescale_default
        return cls.current_rescale_default

    @classmethod
    def stack_channels(
        cls,
        module: "ModuleBlock",
    ) -> tuple["GrayToColorModule.StackChannel", ...]:
        channels: list[GrayToColorModule.StackChannel] = []
        image_name: str | None = None
        color = cls.stack_default_color
        weight = cls.stack_default_weight
        for setting in module.iter_settings():
            if setting.name == cls.stack_image_setting:
                cls.append_stack_channel(
                    channels,
                    image_name=image_name,
                    color=color,
                    weight=weight,
                )
                image_name = setting.value.strip()
                color = cls.stack_default_color
                weight = cls.stack_default_weight
                continue
            if image_name is None:
                continue
            if setting.name == cls.stack_color_setting:
                color = setting.value.strip()
                continue
            if setting.name == cls.stack_weight_setting:
                weight = setting.value.strip()
        cls.append_stack_channel(
            channels,
            image_name=image_name,
            color=color,
            weight=weight,
        )
        return tuple(channels)

    @classmethod
    def append_stack_channel(
        cls,
        channels: list["GrayToColorModule.StackChannel"],
        *,
        image_name: str | None,
        color: str,
        weight: str,
    ) -> None:
        normalized_image_name = normalized_symbol_name(image_name or "")
        if normalized_image_name is None:
            return
        channels.append(
            cls.StackChannel(
                image_name=normalized_image_name,
                color=color,
                weight=weight,
            )
        )

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        from openhcs.core.artifacts import ArtifactKind, ArtifactSpec

        inputs = [
            builder.require_artifact(ArtifactSpec(name, ArtifactKind.IMAGE), module)
            for name in cls.input_names(module)
        ]
        output = builder.declare_artifact(
            ArtifactSpec(required_setting_value(module, "Name the output image"), ArtifactKind.IMAGE),
            module,
        )
        return assembler.assemble_contract(module, builder, inputs=inputs, outputs=[output])


class UnmixColorsModule(ImageProcessingDebugViewModule, ModuleSettingsSourceModule):
    module_name = 'UnmixColors'
    function_name = 'unmix_colors'
    validated = True
    contract = 'flexible'
    confidence = 1.0
    input_image_setting = SettingNameFamily(
        "Select the input color image",
        aliases=("Color image",),
    )
    output_image_setting = SettingNameFamily(
        "Name the output image",
        aliases=("Image name",),
    )
    stain_setting = "Stain"
    red_absorbance_setting = "Red absorbance"
    green_absorbance_setting = "Green absorbance"
    blue_absorbance_setting = "Blue absorbance"
    stain_count_setting = "Stain count"

    @dataclass(frozen=True, slots=True)
    class OutputRow:
        image_name: str
        stain_name: str
        custom_absorbance: tuple[float, float, float]

    @classmethod
    def settings_source(cls, module: "ModuleBlock") -> "CellProfilerKwargs":
        rows = cls.output_rows(module)
        return {
            "stain_names": tuple(row.stain_name for row in rows),
            "custom_absorbances": tuple(row.custom_absorbance for row in rows),
        }

    @classmethod
    def input_name(cls, module: "ModuleBlock") -> str:
        return required_setting_value(module, cls.input_image_setting)

    @classmethod
    def output_rows(
        cls,
        module: "ModuleBlock",
    ) -> tuple["UnmixColorsModule.OutputRow", ...]:
        rows = tuple(
            cls.output_row_from_block(module, block)
            for block in repeating_setting_blocks(
                module.iter_settings(),
                start_name=cls.output_image_setting,
            )
        )
        cls.validate_output_rows(module, rows)
        return rows

    @classmethod
    def output_row_from_block(
        cls,
        module: "ModuleBlock",
        block: tuple["ModuleSetting", ...],
    ) -> "UnmixColorsModule.OutputRow":
        image_name = cls.symbol_name(
            block_setting_value(block, cls.output_image_setting)
        )
        stain_name = block_setting_value(block, cls.stain_setting)
        if not stain_name.strip():
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an UnmixColors "
                f"output row for {image_name!r} without a stain."
            )
        return cls.OutputRow(
            image_name=image_name,
            stain_name=stain_name,
            custom_absorbance=(
                float(
                    block_setting_value(
                        block,
                        cls.red_absorbance_setting,
                        default="0.5",
                    )
                ),
                float(
                    block_setting_value(
                        block,
                        cls.green_absorbance_setting,
                        default="0.5",
                    )
                ),
                float(
                    block_setting_value(
                        block,
                        cls.blue_absorbance_setting,
                        default="0.5",
                    )
                ),
            ),
        )

    @classmethod
    def validate_output_rows(
        cls,
        module: "ModuleBlock",
        rows: tuple["UnmixColorsModule.OutputRow", ...],
    ) -> None:
        if not rows:
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares no "
                "UnmixColors output rows."
            )
        expected_count = cls.expected_output_count(module)
        if expected_count is not None and expected_count != len(rows):
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares "
                f"stain count {expected_count}, but {len(rows)} "
                "UnmixColors output rows were parsed."
            )

    @classmethod
    def expected_output_count(cls, module: "ModuleBlock") -> int | None:
        value = optional_setting_value(module, cls.stain_count_setting)
        if value is None:
            return None
        return int(value)

    @staticmethod
    def symbol_name(raw_value: str) -> str:
        normalized = raw_value.strip()
        if not normalized:
            raise ValueError("CellProfiler symbol names cannot be empty.")
        return normalized

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        from openhcs.core.artifacts import ArtifactKind, ArtifactSpec

        image = builder.require_artifact(ArtifactSpec(cls.input_name(module), ArtifactKind.IMAGE), module)
        outputs = [
            builder.declare_artifact(ArtifactSpec(row.image_name, ArtifactKind.IMAGE), module)
            for row in cls.output_rows(module)
        ]
        return assembler.assemble_contract(module, builder, inputs=[image], outputs=outputs)



class CellProfilerColorFormat(ABC, metaclass=AutoRegisterMeta):
    """Nominal parser family for CellProfiler RGB color literals."""

    __registry_key__ = "format_key"
    __skip_if_no_key__ = True
    format_key: ClassVar[str | None] = None

    @classmethod
    def for_value(cls, value: str | Sequence[float]) -> "CellProfilerColorFormat":
        for format_type in cls.__registry__.values():
            parser = format_type()
            if parser.matches(value):
                return parser
        raise ValueError(f"Unsupported CellProfiler color literal: {value!r}")

    @abstractmethod
    def matches(self, value: str | Sequence[float]) -> bool:
        """Return whether this parser owns the color literal."""

    @abstractmethod
    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        """Return RGB channel values, possibly in 0-255 space."""


class NamedCellProfilerColorFormat(CellProfilerColorFormat):
    """Named CellProfiler colors."""

    format_key = "named"

    def matches(self, value: str | Sequence[float]) -> bool:
        return isinstance(value, str) and value.strip().lower() in _COLOR_BY_NAME

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return _COLOR_BY_NAME[str(value).strip().lower()]


class HexCellProfilerColorFormat(CellProfilerColorFormat):
    """Hex CellProfiler colors such as #0800F7."""

    format_key = "hex"

    def matches(self, value: str | Sequence[float]) -> bool:
        if not isinstance(value, str):
            return False
        literal = value.strip()
        return literal.startswith("#") and len(literal) in {4, 7}

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        literal = str(value).strip().lstrip("#")
        if len(literal) == 3:
            literal = "".join(channel * 2 for channel in literal)
        return (
            int(literal[0:2], 16),
            int(literal[2:4], 16),
            int(literal[4:6], 16),
        )


class DelimitedCellProfilerColorFormat(CellProfilerColorFormat):
    """Comma-delimited RGB triples."""

    format_key = "delimited"

    def matches(self, value: str | Sequence[float]) -> bool:
        return isinstance(value, str) and "," in value

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return tuple(float(part.strip()) for part in str(value).split(","))  # type: ignore[return-value]


class SequenceCellProfilerColorFormat(CellProfilerColorFormat):
    """Already-structured RGB channel sequences."""

    format_key = "sequence"

    def matches(self, value: str | Sequence[float]) -> bool:
        return not isinstance(value, str) and isinstance(value, Sequence)

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return tuple(float(part) for part in value)  # type: ignore[arg-type, return-value]


def coerce_rgb_color(value: str | Sequence[float]) -> tuple[float, float, float]:
    """Parse a CellProfiler color literal into an RGB tuple in 0-1 space."""
    parts = CellProfilerColorFormat.for_value(value).color_parts(value)
    if len(parts) != 3:
        raise ValueError(f"CellProfiler color must have three channels, got {parts!r}.")
    scale = 255.0 if max(parts) > 1.0 else 1.0
    return parts[0] / scale, parts[1] / scale, parts[2] / scale


class StainType(Enum):
    """Closed family of CellProfiler UnmixColors stain choices."""

    HEMATOXYLIN = ("Hematoxylin", (0.644, 0.717, 0.267))
    EOSIN = ("Eosin", (0.093, 0.954, 0.283))
    DAB = ("DAB", (0.268, 0.570, 0.776))
    FAST_RED = ("Fast red", (0.214, 0.851, 0.478))
    FAST_BLUE = ("Fast blue", (0.749, 0.606, 0.267))
    METHYL_BLUE = ("Methyl blue", (0.799, 0.591, 0.105))
    METHYL_GREEN = ("Methyl green", (0.980, 0.144, 0.133))
    AEC = ("AEC", (0.274, 0.679, 0.680))
    ANILINE_BLUE = ("Aniline blue", (0.853, 0.509, 0.113))
    AZOCARMINE = ("Azocarmine", (0.071, 0.977, 0.198))
    ALCIAN_BLUE = ("Alcian blue", (0.875, 0.458, 0.158))
    PAS = ("PAS", (0.175, 0.972, 0.155))
    HEMATOXYLIN_AND_PAS = ("Hematoxylin and PAS", (0.553, 0.754, 0.354))
    FEULGEN = ("Feulgen", (0.464, 0.830, 0.308))
    METHYLENE_BLUE = ("Methylene blue", (0.553, 0.754, 0.354))
    ORANGE_G = ("Orange-G", (0.107, 0.368, 0.923))
    PONCEAU_FUCHSIN = ("Ponceau-fuchsin", (0.100, 0.737, 0.668))
    CUSTOM = ("Custom", None)

    @property
    def display_name(self) -> str:
        return self.value[0]

    @property
    def calibrated_absorbance(self) -> tuple[float, float, float]:
        absorbance = self.value[1]
        if absorbance is None:
            raise ValueError("Custom stains require explicit absorbance values.")
        return absorbance


@dataclass(frozen=True, slots=True)
class StainDefinition:
    """One stain row participating in CellProfiler color deconvolution."""

    stain: StainType
    custom_absorbance: tuple[float, float, float] | None = None

    @property
    def absorbance(self) -> np.ndarray:
        if self.stain is StainType.CUSTOM:
            if self.custom_absorbance is None:
                raise ValueError("Custom UnmixColors rows require absorbance values.")
            absorbance = self.custom_absorbance
        else:
            absorbance = self.stain.calibrated_absorbance
        return _normalized_absorbance(absorbance)


class ImageChannelType(Enum):
    RGB = "rgb"
    HSV = "hsv"
    CHANNELS = "channels"


class ColorToGrayMode(Enum):
    COMBINE = "combine"
    SPLIT = "split"


class OutputMode(Enum):
    """InvertForPrinting output layout."""

    COLOR = "color"
    GRAYSCALE = "grayscale"


_COLOR_BY_NAME: dict[str, tuple[float, float, float]] = {
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


@dataclass(frozen=True, slots=True)
class GrayToColorRequest:
    """Typed request record for one GrayToColor dispatch."""

    image: np.ndarray
    rescale_intensity: bool = True
    red_channel: int = -1
    green_channel: int = -1
    blue_channel: int = -1
    cyan_channel: int = -1
    magenta_channel: int = -1
    yellow_channel: int = -1
    gray_channel: int = -1
    red_weight: float = 1.0
    green_weight: float = 1.0
    blue_weight: float = 1.0
    cyan_weight: float = 1.0
    magenta_weight: float = 1.0
    yellow_weight: float = 1.0
    gray_weight: float = 1.0
    channel_colors: Sequence[str] = ()
    channel_weights: Sequence[float] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_colors", tuple(self.channel_colors))
        object.__setattr__(self, "channel_weights", tuple(self.channel_weights))


class GrayToColorSchemeRunner(ABC, metaclass=AutoRegisterMeta):
    """Nominal closed family for GrayToColor scheme dispatch."""

    __registry_key__ = "scheme_literal"
    __skip_if_no_key__ = True
    scheme_literal: ClassVar[str | None] = None

    @classmethod
    def for_scheme(
        cls,
        scheme: GrayToColorModule.Scheme,
    ) -> "GrayToColorSchemeRunner":
        runner_type = cls.__registry__.get(scheme.value)
        if runner_type is None:
            raise ValueError(f"Unsupported GrayToColor scheme: {scheme.value!r}")
        return runner_type()

    @abstractmethod
    def run(self, request: GrayToColorRequest) -> np.ndarray:
        """Execute one GrayToColor request for the scheme owned by this runner."""

    def channel_or_black(
        self,
        image: np.ndarray,
        channel_index: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        if channel_index < 0 or channel_index >= image.shape[0]:
            return np.zeros((height, width), dtype=np.float64)
        return image[channel_index].astype(np.float64)

    def rescale_positive_channel(self, channel: np.ndarray) -> np.ndarray:
        maximum = np.max(channel)
        if maximum > 0:
            return channel / maximum
        return channel

    def final_rgb(
        self,
        rgb_image: np.ndarray,
        request: GrayToColorRequest,
    ) -> np.ndarray:
        if request.rescale_intensity:
            rgb_image = np.clip(rgb_image, 0, 1)
        return rgb_image.astype(np.float32)


class RGBGrayToColorRunner(GrayToColorSchemeRunner):
    scheme_literal = GrayToColorModule.Scheme.RGB.value

    def run(self, request: GrayToColorRequest) -> np.ndarray:
        image = request.image
        height, width = image.shape[1], image.shape[2]

        red_img = self.channel_or_black(image, request.red_channel, height, width)
        green_img = self.channel_or_black(image, request.green_channel, height, width)
        blue_img = self.channel_or_black(image, request.blue_channel, height, width)

        if request.rescale_intensity:
            red_img = self.rescale_positive_channel(red_img)
            green_img = self.rescale_positive_channel(green_img)
            blue_img = self.rescale_positive_channel(blue_img)

        rgb_image = np.dstack(
            [
                red_img * request.red_weight,
                green_img * request.green_weight,
                blue_img * request.blue_weight,
            ]
        )
        return self.final_rgb(rgb_image, request)


class CMYKGrayToColorRunner(GrayToColorSchemeRunner):
    scheme_literal = GrayToColorModule.Scheme.CMYK.value

    def run(self, request: GrayToColorRequest) -> np.ndarray:
        image = request.image
        height, width = image.shape[1], image.shape[2]

        cyan_img = self.channel_or_black(image, request.cyan_channel, height, width)
        magenta_img = self.channel_or_black(
            image,
            request.magenta_channel,
            height,
            width,
        )
        yellow_img = self.channel_or_black(image, request.yellow_channel, height, width)
        gray_img = self.channel_or_black(image, request.gray_channel, height, width)

        if request.rescale_intensity:
            cyan_img = self.rescale_positive_channel(cyan_img)
            magenta_img = self.rescale_positive_channel(magenta_img)
            yellow_img = self.rescale_positive_channel(yellow_img)
            gray_img = self.rescale_positive_channel(gray_img)

        rgb_image = np.zeros((height, width, 3), dtype=np.float64)
        rgb_image[:, :, 1] += cyan_img * request.cyan_weight * 0.5
        rgb_image[:, :, 2] += cyan_img * request.cyan_weight * 0.5
        rgb_image[:, :, 0] += magenta_img * request.magenta_weight * 0.5
        rgb_image[:, :, 2] += magenta_img * request.magenta_weight * 0.5
        rgb_image[:, :, 0] += yellow_img * request.yellow_weight * 0.5
        rgb_image[:, :, 1] += yellow_img * request.yellow_weight * 0.5
        rgb_image[:, :, 0] += gray_img * request.gray_weight * (1.0 / 3.0)
        rgb_image[:, :, 1] += gray_img * request.gray_weight * (1.0 / 3.0)
        rgb_image[:, :, 2] += gray_img * request.gray_weight * (1.0 / 3.0)
        return self.final_rgb(rgb_image, request)


class StackGrayToColorRunner(GrayToColorSchemeRunner):
    scheme_literal = GrayToColorModule.Scheme.STACK.value

    def run(self, request: GrayToColorRequest) -> np.ndarray:
        return np.transpose(request.image, (1, 2, 0)).astype(np.float32)


class CompositeGrayToColorRunner(GrayToColorSchemeRunner):
    scheme_literal = GrayToColorModule.Scheme.COMPOSITE.value
    default_colors: ClassVar[tuple[str, ...]] = (
        "#ff0000",
        "#00ff00",
        "#0000ff",
        "#808000",
        "#800080",
        "#008080",
    )

    def run(self, request: GrayToColorRequest) -> np.ndarray:
        image = request.image
        colors = list(request.channel_colors) or [
            self.default_colors[index % len(self.default_colors)]
            for index in range(image.shape[0])
        ]
        weights = list(request.channel_weights) or [1.0] * image.shape[0]
        height, width = image.shape[1], image.shape[2]
        rgb_image = np.zeros((height, width, 3), dtype=np.float64)

        for index in range(image.shape[0]):
            channel_img = image[index].astype(np.float64)
            if request.rescale_intensity:
                channel_img = self.rescale_positive_channel(channel_img)
            red, green, blue = coerce_rgb_color(colors[index])
            weight = weights[index]
            rgb_image[:, :, 0] += channel_img * red * weight
            rgb_image[:, :, 1] += channel_img * green * weight
            rgb_image[:, :, 2] += channel_img * blue * weight

        return self.final_rgb(rgb_image, request)


@numpy
def gray_to_color(
    image: np.ndarray,
    color_scheme: GrayToColorModule.Scheme | str = GrayToColorModule.Scheme.RGB.value,
    rescale_intensity: bool = True,
    red_channel: int = -1,
    green_channel: int = -1,
    blue_channel: int = -1,
    cyan_channel: int = -1,
    magenta_channel: int = -1,
    yellow_channel: int = -1,
    gray_channel: int = -1,
    red_weight: float = 1.0,
    green_weight: float = 1.0,
    blue_weight: float = 1.0,
    cyan_weight: float = 1.0,
    magenta_weight: float = 1.0,
    yellow_weight: float = 1.0,
    gray_weight: float = 1.0,
    channel_colors: Sequence[str] = (),
    channel_weights: Sequence[float] = (),
) -> np.ndarray:
    """Dispatch GrayToColor across its RGB, CMYK, Stack, and Composite variants."""
    scheme = GrayToColorModule.coerce_scheme(color_scheme)
    request = GrayToColorRequest(
        image=image,
        rescale_intensity=rescale_intensity,
        red_channel=red_channel,
        green_channel=green_channel,
        blue_channel=blue_channel,
        cyan_channel=cyan_channel,
        magenta_channel=magenta_channel,
        yellow_channel=yellow_channel,
        gray_channel=gray_channel,
        red_weight=red_weight,
        green_weight=green_weight,
        blue_weight=blue_weight,
        cyan_weight=cyan_weight,
        magenta_weight=magenta_weight,
        yellow_weight=yellow_weight,
        gray_weight=gray_weight,
        channel_colors=channel_colors,
        channel_weights=channel_weights,
    )
    return GrayToColorSchemeRunner.for_scheme(scheme).run(request)


@numpy
def color_to_gray(
    image: np.ndarray,
    mode: ColorToGrayMode | str = ColorToGrayMode.SPLIT,
    image_type: ImageChannelType | str = ImageChannelType.RGB,
    channel_indices: tuple[int, ...] = (0, 1, 2),
    contributions: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Convert a CellProfiler channel-last color payload to grayscale outputs."""
    resolved_mode = coerce_cellprofiler_enum(ColorToGrayMode, mode)
    resolved_image_type = coerce_cellprofiler_enum(ImageChannelType, image_type)
    if resolved_mode is ColorToGrayMode.COMBINE:
        output = combine_color_to_gray(image, channel_indices, contributions)
        return with_image_payload_data(
            image,
            output,
            metadata=image_payload_metadata(image).without_unit_interval_intensity_scale(),
        )
    return tuple(
        with_image_payload_data(image, output)
        for output in split_color_to_gray(image, resolved_image_type, channel_indices)
    )


@numpy
def invert_for_printing(
    image: np.ndarray,
    output_mode: OutputMode = OutputMode.COLOR,
    output_red: bool = True,
    output_green: bool = True,
    output_blue: bool = True,
) -> np.ndarray:
    """Invert fluorescent channels into CellProfiler brightfield-print colors."""
    output_mode = coerce_cellprofiler_enum(OutputMode, output_mode)
    image_data = np.asarray(image)
    if image_data.ndim == 2:
        image_data = image_data[np.newaxis, :, :]

    channel_count = image_data.shape[0]
    height, width = image_data.shape[1], image_data.shape[2]
    red_image = (
        image_data[0]
        if channel_count >= 1
        else np.zeros((height, width), dtype=image_data.dtype)
    )
    green_image = (
        image_data[1]
        if channel_count >= 2
        else np.zeros((height, width), dtype=image_data.dtype)
    )
    blue_image = (
        image_data[2]
        if channel_count >= 3
        else np.zeros((height, width), dtype=image_data.dtype)
    )

    inverted_red = (1.0 - green_image) * (1.0 - blue_image)
    inverted_green = (1.0 - red_image) * (1.0 - blue_image)
    inverted_blue = (1.0 - red_image) * (1.0 - green_image)

    if output_mode is OutputMode.COLOR:
        return np.stack(
            [inverted_red, inverted_green, inverted_blue],
            axis=0,
        ).astype(np.float32)

    output_channels = []
    if output_red:
        output_channels.append(inverted_red)
    if output_green:
        output_channels.append(inverted_green)
    if output_blue:
        output_channels.append(inverted_blue)
    if not output_channels:
        return np.zeros((1, height, width), dtype=np.float32)
    return np.stack(output_channels, axis=0).astype(np.float32)


def combine_color_to_gray(
    image: np.ndarray,
    channel_indices: tuple[int, ...],
    contributions: tuple[float, ...],
) -> np.ndarray:
    image_data = image_payload_data(image)
    if len(channel_indices) != len(contributions):
        raise ValueError("channel_indices and contributions must have same length.")
    weights = normalized_color_to_gray_weights(contributions)
    color_stack = nhwc_color_stack(image_data)
    result = np.zeros(color_stack.shape[:3], dtype=np.float32)
    for channel_index, weight in zip(channel_indices, weights, strict=True):
        result += color_to_gray_channel(color_stack, channel_index).astype(np.float32) * weight
    return restore_color_to_gray_shape(image_data, result)


def split_color_to_gray(
    image: np.ndarray,
    image_type: ImageChannelType,
    channel_indices: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    image_data = image_payload_data(image)
    color_stack = nhwc_color_stack(image_data).astype(np.float32)
    source_stack = (
        rgb_to_hsv_stack(color_stack)
        if image_type is ImageChannelType.HSV
        else color_stack
    )
    return tuple(
        restore_color_to_gray_shape(
            image_data,
            color_to_gray_channel(source_stack, index),
        )
        for index in channel_indices
    )


def color_to_gray_channel(
    color_stack: np.ndarray,
    channel_index: int,
) -> np.ndarray:
    if channel_index >= color_stack.shape[-1]:
        raise ValueError(
            f"ColorToGray channel index {channel_index} is outside payload "
            f"with {color_stack.shape[-1]} channels."
        )
    return color_stack[..., channel_index]


def nhwc_color_stack(image: np.ndarray) -> np.ndarray:
    if is_channel_last_image_stack(image):
        return image
    if is_channel_last_image_slice(image):
        return image[np.newaxis, ...]
    raise ValueError(
        "ColorToGray requires a channel-last image shaped (H, W, C) or "
        f"(N, H, W, C), got {getattr(image, 'shape', 'unknown')}."
    )


def restore_color_to_gray_shape(
    original: np.ndarray,
    stack: np.ndarray,
) -> np.ndarray:
    if is_channel_last_image_slice(original):
        return stack[0]
    return stack


def normalized_color_to_gray_weights(
    contributions: tuple[float, ...],
) -> tuple[float, ...]:
    total = sum(contributions)
    if total == 0:
        raise ValueError("Contributions cannot all be zero.")
    return tuple(float(contribution) / total for contribution in contributions)


def rgb_to_hsv_stack(rgb_stack: np.ndarray) -> np.ndarray:
    if rgb_stack.shape[-1] < 3:
        raise ValueError("HSV conversion requires at least three RGB channels.")
    rgb = rgb_stack[..., :3]
    if rgb.size and np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)
    delta = max_channel - min_channel
    value = max_channel
    saturation = np.divide(
        delta,
        max_channel,
        out=np.zeros_like(delta),
        where=max_channel != 0,
    )
    hue = np.zeros_like(red)
    nonzero_delta = delta != 0
    red_is_max = (max_channel == red) & nonzero_delta
    green_is_max = (max_channel == green) & nonzero_delta
    blue_is_max = (max_channel == blue) & nonzero_delta
    hue[red_is_max] = ((green[red_is_max] - blue[red_is_max]) / delta[red_is_max]) % 6
    hue[green_is_max] = (
        (blue[green_is_max] - red[green_is_max]) / delta[green_is_max]
    ) + 2
    hue[blue_is_max] = (
        (red[blue_is_max] - green[blue_is_max]) / delta[blue_is_max]
    ) + 4
    hue = hue / 6.0
    return np.stack((hue, saturation, value), axis=-1).astype(np.float32)


@numpy(contract=ProcessingContract.FLEXIBLE)
def unmix_colors(
    image: np.ndarray,
    stain_names: Sequence[StainType | str] = (),
    custom_absorbances: Sequence[Sequence[float] | None] = (),
    stain1: StainType | str = StainType.HEMATOXYLIN,
    stain2: StainType | str = StainType.EOSIN,
    stain3: StainType | str | None = None,
    output_stain_index: int = 0,
    custom_red_absorbance_1: float = 0.5,
    custom_green_absorbance_1: float = 0.5,
    custom_blue_absorbance_1: float = 0.5,
    custom_red_absorbance_2: float = 0.5,
    custom_green_absorbance_2: float = 0.5,
    custom_blue_absorbance_2: float = 0.5,
    custom_red_absorbance_3: float = 0.5,
    custom_green_absorbance_3: float = 0.5,
    custom_blue_absorbance_3: float = 0.5,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Unmix one RGB image into one image per configured CellProfiler stain row."""
    rgb_image = _as_rgb_image(image)
    if stain_names:
        return _unmix_stain_outputs(
            rgb_image,
            _stain_definitions(stain_names, custom_absorbances),
        )

    definitions = _legacy_stain_definitions(
        stain1=stain1,
        stain2=stain2,
        stain3=stain3,
        custom_absorbances=(
            (
                custom_red_absorbance_1,
                custom_green_absorbance_1,
                custom_blue_absorbance_1,
            ),
            (
                custom_red_absorbance_2,
                custom_green_absorbance_2,
                custom_blue_absorbance_2,
            ),
            (
                custom_red_absorbance_3,
                custom_green_absorbance_3,
                custom_blue_absorbance_3,
            ),
        ),
    )
    outputs = _unmix_stain_outputs(rgb_image, definitions)
    if output_stain_index < 0 or output_stain_index >= len(outputs):
        raise ValueError(
            f"output_stain_index must be in [0, {len(outputs) - 1}], "
            f"got {output_stain_index}."
        )
    return outputs[output_stain_index]


def _stain_definitions(
    stain_names: Sequence[StainType | str],
    custom_absorbances: Sequence[Sequence[float] | None],
) -> tuple[StainDefinition, ...]:
    if len(stain_names) != len(custom_absorbances):
        raise ValueError(
            "UnmixColors stain_names and custom_absorbances must have the "
            "same length."
        )
    return tuple(
        StainDefinition(
            stain=coerce_cellprofiler_enum(StainType, stain_name),
            custom_absorbance=_coerce_custom_absorbance(custom_absorbance),
        )
        for stain_name, custom_absorbance in zip(
            stain_names,
            custom_absorbances,
            strict=True,
        )
    )


def _legacy_stain_definitions(
    *,
    stain1: StainType | str,
    stain2: StainType | str,
    stain3: StainType | str | None,
    custom_absorbances: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[StainDefinition, ...]:
    stains = (stain1, stain2, stain3)
    return tuple(
        StainDefinition(
            stain=coerce_cellprofiler_enum(StainType, stain),
            custom_absorbance=custom_absorbances[index],
        )
        for index, stain in enumerate(stains)
        if stain is not None
    )


def _unmix_stain_outputs(
    image: np.ndarray,
    definitions: tuple[StainDefinition, ...],
) -> tuple[np.ndarray, ...]:
    if not definitions:
        raise ValueError("UnmixColors requires at least one stain definition.")
    inverse_matrix = np.linalg.pinv(
        np.asarray([definition.absorbance for definition in definitions])
    )
    return tuple(
        _run_unmix_output(image, inverse_matrix[:, index])
        for index in range(len(definitions))
    )


def _run_unmix_output(
    image: np.ndarray,
    inverse_absorbances: np.ndarray,
) -> np.ndarray:
    eps = 1.0 / 256.0 / 2.0
    log_image = np.log(image + eps)
    broadcast_shape = (1,) * (log_image.ndim - 1) + (3,)
    scaled_image = log_image * inverse_absorbances.reshape(broadcast_shape)
    result = np.exp(np.sum(scaled_image, axis=-1)) - eps
    return (1.0 - np.clip(result, 0.0, 1.0)).astype(np.float32)


def _as_rgb_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return np.stack((array, array, array), axis=-1)
    if array.ndim >= 3 and array.shape[-1] == 3:
        return array
    if array.ndim == 3 and array.shape[0] == 3:
        return np.moveaxis(array, 0, -1)
    if array.ndim >= 4 and array.shape[1] == 3:
        return np.moveaxis(array, 1, -1)
    raise ValueError(
        "UnmixColors expects an RGB image with three color channels on the "
        f"first or last channel axis, got shape {array.shape}."
    )


def _normalized_absorbance(absorbance: Sequence[float]) -> np.ndarray:
    vector = np.asarray(tuple(float(channel) for channel in absorbance))
    if vector.shape != (3,):
        raise ValueError(
            f"UnmixColors absorbance vectors must have three channels, got {vector}."
        )
    norm = np.sqrt(np.sum(vector**2))
    if norm <= 0:
        raise ValueError("UnmixColors absorbance vectors cannot be zero.")
    return vector / norm


def _coerce_custom_absorbance(
    absorbance: Sequence[float] | None,
) -> tuple[float, float, float] | None:
    if absorbance is None:
        return None
    red, green, blue = absorbance
    return float(red), float(green), float(blue)


class InvertForPrintingModule(ImageProcessingDebugViewModule, CellProfilerModule):
    module_name = 'InvertForPrinting'
    function_name = 'invert_for_printing'
    validated = True
    contract = 'flexible'
    confidence = 1.0
