"""Shared CellProfiler color literal semantics."""

from __future__ import annotations
from openhcs.core.artifacts import ArtifactInputPlan

import re

from openhcs.interop.cellprofiler.setting_names import normalized_symbol_name
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar
from metaclass_registry import AutoRegisterMeta
import numpy as np
from matplotlib.colors import CSS4_COLORS, to_rgb
from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
    pack_aligned_image_outputs,
)
from openhcs.core.artifacts import (
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    GroupLineageSourceRelation,
    ImageArtifactType,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.pipeline.function_contracts import (
    composed_image_payload,
    required_variable_components,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    block_setting_value,
    optional_setting_value,
    repeating_setting_blocks,
    required_setting_value,
    setting_names,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_setting_parser,
    coerce_cellprofiler_enum,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
)

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
    from openhcs.core.steps.function_runtime import RuntimeCallableKwargs
    from openhcs.interop.cellprofiler.settings_binder import SettingsBinder


class ImageChannelType(Enum):
    RGB = "rgb"
    HSV = "hsv"
    CHANNELS = "channels"


class ColorToGrayMode(Enum):
    COMBINE = "combine"
    SPLIT = "split"


class ColorToGrayModule(
    CellProfilerModule,
):
    module_name = "ColorToGray"
    function_name = "color_to_gray"
    validated = True
    confidence = 1.0
    input_image_setting = SettingNameFamily("Select the input image")
    output_image_setting = SettingNameFamily("Name the output image")
    channel_output_image_setting = SettingNameFamily("Image name")
    conversion_method_setting = "Conversion method"
    image_type_setting = "Image type"
    channel_number_setting = "Channel number"
    channel_weight_setting = "Relative weight of the channel"
    channel_count_setting = "Channel count"

    ConversionMethod = ColorToGrayMode
    ImageType = ImageChannelType

    @dataclass(frozen=True, slots=True)
    class FixedChannel:
        output_flag: str
        output_suffix: str
        weight_setting: str | None = None

    @dataclass(frozen=True, slots=True)
    class FixedImageType:
        output_offset: int
        channels: tuple["ColorToGrayModule.FixedChannel", ...]

    rgb_fixed_image_type = FixedImageType(
        output_offset=1,
        channels=(
            FixedChannel(
                "Convert red to gray?",
                "Red",
                "Relative weight of the red channel",
            ),
            FixedChannel(
                "Convert green to gray?",
                "Green",
                "Relative weight of the green channel",
            ),
            FixedChannel(
                "Convert blue to gray?",
                "Blue",
                "Relative weight of the blue channel",
            ),
        ),
    )
    hsv_fixed_image_type = FixedImageType(
        output_offset=4,
        channels=(
            FixedChannel("Convert hue to gray?", "Hue"),
            FixedChannel("Convert saturation to gray?", "Saturation"),
            FixedChannel("Convert value to gray?", "Value"),
        ),
    )
    default_channel_indices = tuple(range(len(rgb_fixed_image_type.channels)))

    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting, ImageArtifactType
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )
    channel_output_image_binding = SettingToKeywordBinding.output(
        channel_output_image_setting,
        ImageArtifactType,
        repeated=True,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        input_image_binding,
        output_image_binding,
        channel_output_image_binding,
        SettingToKeywordBinding(
            conversion_method_setting,
            "mode",
            cellprofiler_enum_setting_parser(ConversionMethod),
        ),
        SettingToKeywordBinding(
            image_type_setting,
            "image_type",
            cellprofiler_enum_setting_parser(ImageType),
        ),
    )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        split_channels = (
            cls.conversion_method(module) is cls.ConversionMethod.SPLIT
            and cls.image_type(module) is cls.ImageType.CHANNELS
        )
        inactive = (
            cls.output_image_binding
            if split_channels
            else cls.channel_output_image_binding
        )
        return tuple(binding for binding in bindings if binding is not inactive)

    @classmethod
    def artifact_names_for_binding(cls, module, binding):
        """Project only output rows active under the selected conversion mode."""

        if binding in (
            cls.output_image_binding,
            cls.channel_output_image_binding,
        ):
            return cls.output_image_names(module)
        return super().artifact_names_for_binding(module, binding)

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
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
    ) -> BoundModuleSettings:
        """Bind behavior and artifact identities from the same module rows."""

        bound = cls._bind_declared_settings(module, binder=binder)
        plan = cls.plan(module, binder)
        kwargs = {**dict(bound.kwargs), **plan.kwargs}
        output_binding = (
            cls.channel_output_image_binding
            if plan.mode is cls.ConversionMethod.SPLIT
            and plan.image_type is cls.ImageType.CHANNELS
            else cls.output_image_binding
        )
        kwargs[output_binding.require_parameter_name()] = (
            plan.output_image_names[0]
            if len(plan.output_image_names) == 1
            else plan.output_image_names
        )
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting in cls._compound_setting_names():
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting), None)
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(kwargs, unmapped_kwargs),
        )

    @classmethod
    def _derived_identity_setting_records(
        cls,
        *,
        invocation,
        block_position,
        existing_records,
        step_context,
    ):
        """Rebuild sparse ColorToGray image-output rows from public settings."""
        from openhcs.interop.cellprofiler.parser import ModuleSetting

        if cls._setting_values(
            existing_records, cls.output_image_setting
        ) or cls._setting_values(existing_records, cls.channel_output_image_setting):
            own_records = ()
        else:
            input_name = cls._single_setting_value(
                existing_records, cls.input_image_setting
            )
            mode_literal = cls._single_setting_value(
                existing_records, cls.conversion_method_setting
            )
            image_type_literal = cls._single_setting_value(
                existing_records, cls.image_type_setting
            )
            if input_name is None or mode_literal is None or image_type_literal is None:
                own_records = ()
            else:
                mode = coerce_cellprofiler_enum(cls.ConversionMethod, mode_literal)
                image_type = coerce_cellprofiler_enum(cls.ImageType, image_type_literal)
                channel_indices = tuple(
                    int(index)
                    for index in invocation.kwargs_dict.get(
                        "channel_indices",
                        cls.default_channel_indices,
                    )
                )
                output_names = cls._canonical_output_names(
                    input_name,
                    mode=mode,
                    image_type=image_type,
                    channel_indices=channel_indices,
                )
                setting_name = (
                    cls.channel_output_image_setting.canonical
                    if image_type is cls.ImageType.CHANNELS
                    and mode is cls.ConversionMethod.SPLIT
                    else cls.output_image_setting.canonical
                )
                own_records = tuple(
                    ModuleSetting(setting_name, name) for name in output_names
                )
        return (
            *own_records,
            *super()._derived_identity_setting_records(
                invocation=invocation,
                block_position=block_position,
                existing_records=(*existing_records, *own_records),
                step_context=step_context,
            ),
        )

    @classmethod
    def _canonical_output_names(
        cls,
        input_name: str,
        *,
        mode: "ColorToGrayModule.ConversionMethod",
        image_type: "ColorToGrayModule.ImageType",
        channel_indices: tuple[int, ...],
    ) -> tuple[str, ...]:
        """Return declaration-owned ColorToGray output names for sparse source."""
        base_name = cls._grayscale_base_name(input_name)
        if mode is cls.ConversionMethod.COMBINE:
            return (f"{base_name}Gray",)
        if image_type is cls.ImageType.CHANNELS:
            return tuple(f"{base_name}Channel{index + 1}" for index in channel_indices)
        declaration = cls.fixed_image_type(image_type)
        if any(
            index < 0 or index >= len(declaration.channels) for index in channel_indices
        ):
            raise ValueError(
                f"ColorToGray channel indices must address {image_type.value} "
                f"channels, got {channel_indices!r}."
            )
        return tuple(
            f"{base_name}{declaration.channels[index].output_suffix}"
            for index in channel_indices
        )

    @classmethod
    def _compound_setting_names(cls) -> tuple[str, ...]:
        return (
            cls.channel_count_setting,
            cls.channel_number_setting,
            cls.channel_weight_setting,
            *(
                channel.weight_setting
                for channel in cls.rgb_fixed_image_type.channels
                if channel.weight_setting is not None
            ),
            *(
                channel.output_flag
                for declaration in (
                    cls.rgb_fixed_image_type,
                    cls.hsv_fixed_image_type,
                )
                for channel in declaration.channels
            ),
        )

    @classmethod
    def _setting_values(cls, records, setting_name) -> tuple[str, ...]:
        """Return sparse transient setting values owned by this module."""
        from openhcs.interop.cellprofiler.setting_names import setting_name_matches

        return tuple(
            record.value
            for record in records
            if setting_name_matches(record.name, setting_name)
        )

    @classmethod
    def _single_setting_value(cls, records, setting_name) -> str | None:
        """Return one sparse compile-time setting value for this module."""
        values = cls._setting_values(records, setting_name)
        if not values:
            return None
        if len(values) != 1:
            raise ValueError(
                "Expected one ColorToGray transient setting row for "
                f"{setting_name!r}, got {values!r}."
            )
        return values[0]

    @staticmethod
    def _grayscale_base_name(input_name: str) -> str:
        """Return the source-name stem used for canonical grayscale outputs."""
        for suffix in ("Color", "Colour"):
            if input_name.endswith(suffix) and len(input_name) > len(suffix):
                return input_name[: -len(suffix)]
        return input_name

    @classmethod
    def plan(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
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
        declaration = cls.fixed_image_type(image_type)
        output_names = setting_values(module, cls.output_image_setting)
        output_flag_values = tuple(
            optional_setting_value(module, channel.output_flag)
            for channel in declaration.channels
        )
        if all(value is None for value in output_flag_values):
            if not output_names:
                raise ValueError(
                    f"ColorToGray({module.module_num}) sparse split mode declares "
                    "no output images."
                )
            return output_names
        enabled_indices = tuple(
            index
            for index, (channel, value) in enumerate(
                zip(declaration.channels, output_flag_values, strict=True)
            )
            if value is not None
            and cls.flag_enabled(module, binder, channel.output_flag)
        )
        if not enabled_indices:
            raise ValueError(
                f"ColorToGray({module.module_num}) split mode must declare at least one enabled output channel."
            )
        full_cellprofiler_row_count = declaration.output_offset + len(
            declaration.channels
        )
        if len(output_names) >= full_cellprofiler_row_count:
            return tuple(
                output_names[declaration.output_offset + index]
                for index in enabled_indices
            )
        if len(output_names) == len(enabled_indices):
            return output_names
        raise ValueError(
            f"ColorToGray({module.module_num}) split mode expected either "
            f"{full_cellprofiler_row_count} CellProfiler output rows or "
            f"{len(enabled_indices)} sparse public output rows, got "
            f"{len(output_names)}."
        )

    @classmethod
    def conversion_method(
        cls, module: "ModuleBlock"
    ) -> "ColorToGrayModule.ConversionMethod":
        return coerce_cellprofiler_enum(
            cls.ConversionMethod,
            required_setting_value(module, cls.conversion_method_setting),
        )

    @classmethod
    def image_type(cls, module: "ModuleBlock") -> "ColorToGrayModule.ImageType":
        return coerce_cellprofiler_enum(
            cls.ImageType, required_setting_value(module, cls.image_type_setting)
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
            indices = tuple(
                (cls.channel_number_index(value) for value in channel_numbers)
            )
            if mode is cls.ConversionMethod.SPLIT:
                output_count = len(
                    setting_values(module, cls.channel_output_image_setting)
                )
                return indices[:output_count]
            return indices
        if mode is cls.ConversionMethod.COMBINE:
            return (0, 1, 2)
        declaration = cls.fixed_image_type(image_type)
        return tuple(
            index
            for index, channel in enumerate(declaration.channels)
            if cls.flag_enabled(module, binder, channel.output_flag)
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
            return tuple((1.0 for _index in channel_indices))
        if len(channel_indices) == 3:
            return tuple(
                (
                    float(
                        binder.parse_value(
                            setting, required_setting_value(module, setting)
                        )
                    )
                    for channel in cls.rgb_fixed_image_type.channels
                    if (setting := channel.weight_setting) is not None
                )
            )
        return tuple(
            (
                float(binder.parse_value(cls.channel_weight_setting, value))
                for value in setting_values(module, cls.channel_weight_setting)
            )
        )

    @classmethod
    def fixed_image_type(
        cls, image_type: "ColorToGrayModule.ImageType"
    ) -> "ColorToGrayModule.FixedImageType":
        if image_type is cls.ImageType.RGB:
            return cls.rgb_fixed_image_type
        if image_type is cls.ImageType.HSV:
            return cls.hsv_fixed_image_type
        raise ValueError(f"{image_type.value!r} is not a fixed-channel image type.")

    @classmethod
    def flag_enabled(
        cls, module: "ModuleBlock", binder: "SettingsBinder", setting_name: str
    ) -> bool:
        return bool(
            binder.parse_value(
                setting_name, required_setting_value(module, setting_name)
            )
        )

    @staticmethod
    def channel_number_index(literal: str) -> int:
        match = re.search("([0-9]+)$", literal.strip())
        if match is None:
            raise ValueError(
                f"ColorToGray channel number lacks an integer suffix: {literal!r}"
            )
        return int(match.group(1)) - 1


class GrayToColorModule(
    CellProfilerModule,
):
    module_name = "GrayToColor"
    function_name = "gray_to_color"
    validated = True
    group_by = GroupBy.SITE
    confidence = 1.0
    color_scheme_setting = SettingNameFamily("Select a color scheme")
    rescale_setting = SettingNameFamily("Rescale intensity")
    current_rescale_default = "Yes"
    revision_3_upgraded_rescale_default = "No"
    output_image_setting = "Name the output image"
    ignored_settings = ("Hidden",)

    class Scheme(str, Enum):
        RGB = "RGB"
        CMYK = "CMYK"
        STACK = "Stack"
        COMPOSITE = "Composite"

    @dataclass(frozen=True, slots=True)
    class IndexedChannel:
        image_binding: SettingToKeywordBinding
        channel_parameter: str
        weight_binding: SettingToKeywordBinding

    @dataclass(frozen=True, slots=True)
    class StackRows:
        image_binding: SettingToKeywordBinding
        color_setting: str
        weight_setting: str
        default_color: str
        default_weight: str

    rgb_channels = (
        IndexedChannel(
            SettingToKeywordBinding.input(
                "Select the image to be colored red", ImageArtifactType
            ),
            "red_channel",
            SettingToKeywordBinding(
                "Relative weight for the red image",
                "red_weight",
                float,
            ),
        ),
        IndexedChannel(
            SettingToKeywordBinding.input(
                "Select the image to be colored green", ImageArtifactType
            ),
            "green_channel",
            SettingToKeywordBinding(
                "Relative weight for the green image",
                "green_weight",
                float,
            ),
        ),
        IndexedChannel(
            SettingToKeywordBinding.input(
                "Select the image to be colored blue", ImageArtifactType
            ),
            "blue_channel",
            SettingToKeywordBinding(
                "Relative weight for the blue image",
                "blue_weight",
                float,
            ),
        ),
    )
    cmyk_channels = (
        IndexedChannel(
            SettingToKeywordBinding.input(
                "Select the image to be colored cyan", ImageArtifactType
            ),
            "cyan_channel",
            SettingToKeywordBinding(
                "Relative weight for the cyan image",
                "cyan_weight",
                float,
            ),
        ),
        IndexedChannel(
            SettingToKeywordBinding.input(
                "Select the image to be colored magenta", ImageArtifactType
            ),
            "magenta_channel",
            SettingToKeywordBinding(
                "Relative weight for the magenta image",
                "magenta_weight",
                float,
            ),
        ),
        IndexedChannel(
            SettingToKeywordBinding.input(
                "Select the image to be colored yellow", ImageArtifactType
            ),
            "yellow_channel",
            SettingToKeywordBinding(
                "Relative weight for the yellow image",
                "yellow_weight",
                float,
            ),
        ),
        IndexedChannel(
            SettingToKeywordBinding.input(
                "Select the image that determines brightness", ImageArtifactType
            ),
            "gray_channel",
            SettingToKeywordBinding(
                "Relative weight for the brightness image",
                "gray_weight",
                float,
            ),
        ),
    )
    stack_rows = StackRows(
        image_binding=SettingToKeywordBinding.input(
            "Image name", ImageArtifactType, repeated=True
        ),
        color_setting="Color",
        weight_setting="Weight",
        default_color="#ff0000",
        default_weight="1.0",
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        *(channel.image_binding for channel in (*rgb_channels, *cmyk_channels)),
        stack_rows.image_binding,
        output_image_binding,
        SettingToKeywordBinding(
            color_scheme_setting,
            "color_scheme",
            cellprofiler_enum_setting_parser(Scheme),
        ),
        SettingToKeywordBinding(
            rescale_setting,
            "rescale_intensity",
            parse_cellprofiler_bool,
        ),
        *(channel.weight_binding for channel in (*rgb_channels, *cmyk_channels)),
    )

    @classmethod
    def artifact_output_relations(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        binding: SettingToKeywordBinding,
        name: str,
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Use the first composed channel as the RGB image's parent context."""

        del cls, module, invocation_key, step_context, binding, name, output_position
        image_inputs = artifact_inputs.for_artifact_type(ImageArtifactType).specs
        if not image_inputs:
            raise ValueError("GrayToColor requires at least one image input.")
        return (GroupLineageSourceRelation(source=image_inputs[0].ref()),)

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        scheme = cls.scheme(module)
        if scheme in (cls.Scheme.RGB, cls.Scheme.CMYK):
            active_inputs = tuple(
                channel.image_binding for channel in cls.indexed_channels(scheme)
            )
        else:
            active_inputs = (cls.stack_rows.image_binding,)
        declared_inputs = frozenset(
            cls.declared_artifact_bindings(
                plan_type=ArtifactInputPlan,
                artifact_type=ImageArtifactType,
            )
        )
        return tuple(
            binding
            for binding in bindings
            if binding not in declared_inputs or binding in active_inputs
        )

    @classmethod
    def module_blocks_for_invocation(cls, *, invocation, step_context):
        """Project explicit callable channel choices into CellProfiler rows."""

        explicit_kwargs = invocation.kwargs_dict
        scheme = cls.coerce_scheme(
            explicit_kwargs.get("color_scheme", cls.Scheme.RGB.value)
        )
        if scheme not in (cls.Scheme.RGB, cls.Scheme.CMYK):
            return super().module_blocks_for_invocation(
                invocation=invocation,
                step_context=step_context,
            )
        channels = cls.indexed_channels(scheme)
        if not any(
            channel.channel_parameter in explicit_kwargs for channel in channels
        ):
            return super().module_blocks_for_invocation(
                invocation=invocation,
                step_context=step_context,
            )

        reconstructed_kwargs = dict(explicit_kwargs)
        for channel in channels:
            if explicit_kwargs.get(channel.channel_parameter, -1) < 0:
                reconstructed_kwargs[channel.image_binding.require_parameter_name()] = (
                    None
                )
        blocks, consumed = super().module_blocks_for_invocation(
            invocation=replace(
                invocation,
                kwargs=tuple(reconstructed_kwargs.items()),
            ),
            step_context=step_context,
        )
        explicit_names = frozenset(explicit_kwargs)
        return blocks, tuple(name for name in consumed if name in explicit_names)

    @dataclass(frozen=True, slots=True)
    class StackChannel:
        """One repeated Stack/Composite source channel row."""

        image_name: str
        color: str
        weight: str

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
    ) -> "BoundModuleSettings":
        """Bind indexed and repeated channel rows through this declaration."""

        bound = cls._bind_declared_settings(module, binder=binder)
        kwargs = {**dict(bound.kwargs), **cls._gray_to_color_kwargs(module, binder)}
        stack_images = tuple(
            channel.image_name for channel in cls.stack_channels(module)
        )
        if stack_images:
            kwargs[cls.stack_rows.image_binding.require_parameter_name()] = stack_images
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting in (
            cls.stack_rows.color_setting,
            cls.stack_rows.weight_setting,
        ):
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting), None)
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(kwargs, unmapped_kwargs),
        )

    @classmethod
    def _gray_to_color_kwargs(
        cls, module: ModuleBlock, binder: SettingsBinder
    ) -> RuntimeCallableKwargs:
        scheme = cls.scheme(module)
        if scheme in (
            GrayToColorModule.Scheme.RGB,
            GrayToColorModule.Scheme.CMYK,
        ):
            return cls.indexed_scheme_kwargs(
                module,
                binder,
                scheme=scheme,
                channels=cls.indexed_channels(scheme),
            )
        if scheme in {
            GrayToColorModule.Scheme.STACK,
            GrayToColorModule.Scheme.COMPOSITE,
        }:
            return cls.stack_scheme_kwargs(module, binder, scheme=scheme)
        raise ValueError(f"Unsupported GrayToColor scheme: {scheme.value!r}")

    @classmethod
    def scheme(cls, module: ModuleBlock) -> GrayToColorModule.Scheme:
        return cls.coerce_scheme(
            module.get_setting(cls.color_scheme_setting.canonical, cls.Scheme.RGB.value)
        )

    @classmethod
    def coerce_scheme(
        cls, value: GrayToColorModule.Scheme | str
    ) -> GrayToColorModule.Scheme:
        if isinstance(value, cls.Scheme):
            return value
        normalized = value.strip()
        for scheme in cls.Scheme:
            if scheme.value == normalized:
                return scheme
        raise ValueError(f"Unsupported GrayToColor scheme: {value!r}.")

    @classmethod
    def indexed_scheme_kwargs(
        cls,
        module: ModuleBlock,
        binder: SettingsBinder,
        *,
        scheme: GrayToColorModule.Scheme,
        channels: tuple["GrayToColorModule.IndexedChannel", ...],
    ) -> RuntimeCallableKwargs:
        kwargs: dict[str, Any] = cls.base_kwargs(module, binder, scheme)
        channel_index = 0
        for channel in channels:
            setting_name = setting_names(channel.image_binding.setting_name)[0]
            kwargs[channel.channel_parameter] = -1
            image_name = normalized_symbol_name(module.get_setting(setting_name, ""))
            if image_name is None:
                continue
            kwargs[channel.channel_parameter] = channel_index
            channel_index += 1
        for channel in channels:
            setting_name = setting_names(channel.weight_binding.setting_name)[0]
            kwargs[channel.weight_binding.require_parameter_name()] = float(
                binder.parse_value(
                    setting_name, module.get_setting(setting_name, "1.0")
                )
            )
        return kwargs

    @classmethod
    def indexed_channels(
        cls,
        scheme: GrayToColorModule.Scheme,
    ) -> tuple[GrayToColorModule.IndexedChannel, ...]:
        if scheme is cls.Scheme.RGB:
            return cls.rgb_channels
        if scheme is cls.Scheme.CMYK:
            return cls.cmyk_channels
        raise ValueError(f"{scheme.value!r} is not an indexed color scheme.")

    @classmethod
    def stack_scheme_kwargs(
        cls,
        module: ModuleBlock,
        binder: SettingsBinder,
        *,
        scheme: GrayToColorModule.Scheme,
    ) -> RuntimeCallableKwargs:
        channels = cls.stack_channels(module)
        kwargs: dict[str, Any] = cls.base_kwargs(module, binder, scheme)
        kwargs["channel_weights"] = tuple(
            float(binder.parse_value(cls.stack_rows.weight_setting, channel.weight))
            for channel in channels
        )
        if scheme is GrayToColorModule.Scheme.COMPOSITE:
            kwargs["channel_colors"] = tuple((channel.color for channel in channels))
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
                        cls.rescale_setting.canonical, cls.rescale_default(module)
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
        cls, module: "ModuleBlock"
    ) -> tuple["GrayToColorModule.StackChannel", ...]:
        channels: list[GrayToColorModule.StackChannel] = []
        image_name: str | None = None
        color = cls.stack_rows.default_color
        weight = cls.stack_rows.default_weight
        image_setting = setting_names(cls.stack_rows.image_binding.setting_name)[0]
        for setting in module.iter_settings():
            if setting.name == image_setting:
                cls.append_stack_channel(
                    channels, image_name=image_name, color=color, weight=weight
                )
                image_name = setting.value.strip()
                color = cls.stack_rows.default_color
                weight = cls.stack_rows.default_weight
                continue
            if image_name is None:
                continue
            if setting.name == cls.stack_rows.color_setting:
                color = setting.value.strip()
                continue
            if setting.name == cls.stack_rows.weight_setting:
                weight = setting.value.strip()
        cls.append_stack_channel(
            channels, image_name=image_name, color=color, weight=weight
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
                image_name=normalized_image_name, color=color, weight=weight
            )
        )


class UnmixColorsModule(
    CellProfilerModule,
):
    module_name = "UnmixColors"
    function_name = "unmix_colors"
    validated = True
    confidence = 1.0
    input_image_setting = SettingNameFamily(
        "Select the input color image", aliases=("Color image",)
    )
    output_image_setting = SettingNameFamily(
        "Name the output image", aliases=("Image name",)
    )
    stain_setting = "Stain"
    red_absorbance_setting = "Red absorbance"
    green_absorbance_setting = "Green absorbance"
    blue_absorbance_setting = "Blue absorbance"
    stain_count_setting = "Stain count"
    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting, ImageArtifactType
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        input_image_binding,
        output_image_binding,
    )

    @dataclass(frozen=True, slots=True)
    class OutputRow:
        image_name: str
        stain_name: str
        custom_absorbance: tuple[float, float, float]

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: "ModuleBlock",
        bound: BoundModuleSettings,
    ) -> BoundModuleSettings:
        rows = cls.output_rows(module)
        return bound.with_kwargs(
            {
                "stain_names": tuple((row.stain_name for row in rows)),
                "custom_absorbances": tuple((row.custom_absorbance for row in rows)),
            }
        ).with_consumed_settings(
            cls.stain_count_setting,
            cls.stain_setting,
            cls.red_absorbance_setting,
            cls.green_absorbance_setting,
            cls.blue_absorbance_setting,
        )

    @classmethod
    def output_rows(
        cls, module: "ModuleBlock"
    ) -> tuple["UnmixColorsModule.OutputRow", ...]:
        rows = tuple(
            (
                cls.output_row_from_block(module, block)
                for block in repeating_setting_blocks(
                    module.iter_settings(), start_name=cls.output_image_setting
                )
            )
        )
        cls.validate_output_rows(module, rows)
        return rows

    @classmethod
    def output_row_from_block(
        cls, module: "ModuleBlock", block: tuple["ModuleSetting", ...]
    ) -> "UnmixColorsModule.OutputRow":
        image_name = cls.symbol_name(
            block_setting_value(block, cls.output_image_setting)
        )
        stain_name = block_setting_value(block, cls.stain_setting)
        if not stain_name.strip():
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an UnmixColors output row for {image_name!r} without a stain."
            )
        return cls.OutputRow(
            image_name=image_name,
            stain_name=stain_name,
            custom_absorbance=(
                float(
                    block_setting_value(
                        block, cls.red_absorbance_setting, default="0.5"
                    )
                ),
                float(
                    block_setting_value(
                        block, cls.green_absorbance_setting, default="0.5"
                    )
                ),
                float(
                    block_setting_value(
                        block, cls.blue_absorbance_setting, default="0.5"
                    )
                ),
            ),
        )

    @classmethod
    def validate_output_rows(
        cls, module: "ModuleBlock", rows: tuple["UnmixColorsModule.OutputRow", ...]
    ) -> None:
        if not rows:
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares no UnmixColors output rows."
            )
        expected_count = cls.expected_output_count(module)
        if expected_count is not None and expected_count != len(rows):
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares stain count {expected_count}, but {len(rows)} UnmixColors output rows were parsed."
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
        return isinstance(value, str) and value.strip().lower() in CSS4_COLORS

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return to_rgb(str(value).strip().lower())


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
            literal = "".join((channel * 2 for channel in literal))
        return (int(literal[0:2], 16), int(literal[2:4], 16), int(literal[4:6], 16))


class DelimitedCellProfilerColorFormat(CellProfilerColorFormat):
    """Comma-delimited RGB triples."""

    format_key = "delimited"

    def matches(self, value: str | Sequence[float]) -> bool:
        return isinstance(value, str) and "," in value

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return tuple((float(part.strip()) for part in str(value).split(",")))


class SequenceCellProfilerColorFormat(CellProfilerColorFormat):
    """Already-structured RGB channel sequences."""

    format_key = "sequence"

    def matches(self, value: str | Sequence[float]) -> bool:
        return not isinstance(value, str) and isinstance(value, Sequence)

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return tuple((float(part) for part in value))


def coerce_rgb_color(value: str | Sequence[float]) -> tuple[float, float, float]:
    """Parse a CellProfiler color literal into an RGB tuple in 0-1 space."""
    parts = CellProfilerColorFormat.for_value(value).color_parts(value)
    if len(parts) != 3:
        raise ValueError(f"CellProfiler color must have three channels, got {parts!r}.")
    scale = 255.0 if max(parts) > 1.0 else 1.0
    return (parts[0] / scale, parts[1] / scale, parts[2] / scale)


class StainType(Enum):
    """Closed family of CellProfiler UnmixColors stain choices."""

    HEMATOXYLIN = ("Hematoxylin", (0.644, 0.717, 0.267))
    EOSIN = ("Eosin", (0.093, 0.954, 0.283))
    DAB = ("DAB", (0.268, 0.57, 0.776))
    FAST_RED = ("Fast red", (0.214, 0.851, 0.478))
    FAST_BLUE = ("Fast blue", (0.749, 0.606, 0.267))
    METHYL_BLUE = ("Methyl blue", (0.799, 0.591, 0.105))
    METHYL_GREEN = ("Methyl green", (0.98, 0.144, 0.133))
    AEC = ("AEC", (0.274, 0.679, 0.68))
    ANILINE_BLUE = ("Aniline blue", (0.853, 0.509, 0.113))
    AZOCARMINE = ("Azocarmine", (0.071, 0.977, 0.198))
    ALCIAN_BLUE = ("Alcian blue", (0.875, 0.458, 0.158))
    PAS = ("PAS", (0.175, 0.972, 0.155))
    HEMATOXYLIN_AND_PAS = ("Hematoxylin and PAS", (0.553, 0.754, 0.354))
    FEULGEN = ("Feulgen", (0.464, 0.83, 0.308))
    METHYLENE_BLUE = ("Methylene blue", (0.553, 0.754, 0.354))
    ORANGE_G = ("Orange-G", (0.107, 0.368, 0.923))
    PONCEAU_FUCHSIN = ("Ponceau-fuchsin", (0.1, 0.737, 0.668))
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


class OutputMode(Enum):
    """InvertForPrinting output layout."""

    COLOR = "color"
    GRAYSCALE = "grayscale"


class InvertInputMode(Enum):
    """InvertForPrinting input layout."""

    COLOR = "color"
    GRAYSCALE = "grayscale"


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
    def for_scheme(cls, scheme: GrayToColorModule.Scheme) -> "GrayToColorSchemeRunner":
        runner_type = cls.__registry__.get(scheme.value)
        if runner_type is None:
            raise ValueError(f"Unsupported GrayToColor scheme: {scheme.value!r}")
        return runner_type()

    @abstractmethod
    def run(self, request: GrayToColorRequest) -> np.ndarray:
        """Execute one GrayToColor request for the scheme owned by this runner."""

    def channel_or_black(
        self, image: np.ndarray, channel_index: int, height: int, width: int
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
        self, rgb_image: np.ndarray, request: GrayToColorRequest
    ) -> np.ndarray:
        if request.rescale_intensity:
            rgb_image = np.clip(rgb_image, 0, 1)
        return rgb_image.astype(np.float32)


class RGBGrayToColorRunner(GrayToColorSchemeRunner):
    scheme_literal = GrayToColorModule.Scheme.RGB.value

    def run(self, request: GrayToColorRequest) -> np.ndarray:
        image = request.image
        height, width = (image.shape[1], image.shape[2])
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
        height, width = (image.shape[1], image.shape[2])
        cyan_img = self.channel_or_black(image, request.cyan_channel, height, width)
        magenta_img = self.channel_or_black(
            image, request.magenta_channel, height, width
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
        height, width = (image.shape[1], image.shape[2])
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


@required_variable_components(VariableComponents.CHANNEL)
@composed_image_payload
@numpy(contract=ProcessingContract.PURE_3D)
def gray_to_color(
    image: np.ndarray,
    color_scheme: GrayToColorModule.Scheme = GrayToColorModule.Scheme.RGB,
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
    """Dispatch GrayToColor across its RGB, CMYK, Stack, and Composite variants.

    Args:
        red_channel: Zero-based source channel mapped to red; ``-1`` omits it.
        green_channel: Zero-based source channel mapped to green; ``-1`` omits it.
        blue_channel: Zero-based source channel mapped to blue; ``-1`` omits it.
        cyan_channel: Zero-based source channel mapped to cyan; ``-1`` omits it.
        magenta_channel: Zero-based source channel mapped to magenta; ``-1`` omits
            it.
        yellow_channel: Zero-based source channel mapped to yellow; ``-1`` omits
            it.
        gray_channel: Zero-based source channel added equally to RGB; ``-1`` omits
            it.
        channel_colors: Per-channel color names or RGB specifications for composite
            mode; the sequence must cover each input channel.
        channel_weights: Per-channel intensity multipliers for composite mode; an
            empty sequence uses unit weights.
    """
    scheme = color_scheme
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
    output = GrayToColorSchemeRunner.for_scheme(scheme).run(request)
    metadata = image_payload_metadata(image)
    if metadata.plane_axis is not None:
        if metadata.plane_axis is not RuntimePlaneAxis.SOURCE_BINDING:
            raise ValueError(
                "GrayToColor image metadata requires a source-binding plane "
                f"axis, got {metadata.plane_axis!r}."
            )
        source_plane_count = metadata.source_provenance.source_plane_count
        if source_plane_count < 1:
            raise ValueError(
                "GrayToColor source-binding payload has no declared source planes."
            )
        parent_image = RuntimeSliceProjection.value_for_slice(
            image,
            RuntimePlaneAxisValueProjection.from_selected_plane(
                axis=RuntimePlaneAxis.SOURCE_BINDING,
                plane_index=0,
                axis_size=source_plane_count,
                source_aliases=metadata.source_image_names,
            ),
        )
    else:
        parent_image = image
    output_metadata = (
        metadata.without_leading_plane_axis()
        if metadata.plane_axis is not None
        else metadata
    )
    return replace(output_metadata, source_channel_axis=-1).payload_with(
        output,
        image_payload_mask(parent_image),
    )


@numpy(contract=ProcessingContract.FLEXIBLE)
def color_to_gray(
    image: np.ndarray,
    mode: ColorToGrayMode = ColorToGrayMode.SPLIT,
    image_type: ImageChannelType = ImageChannelType.RGB,
    channel_indices: tuple[int, ...] = ColorToGrayModule.default_channel_indices,
    contributions: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> np.ndarray | AlignedImageStack:
    """Convert a channel-last color image to selected grayscale outputs.

    Args:
        channel_indices: Zero-based color-channel positions to split or combine.
        contributions: Relative weights for the selected channels in combine mode;
            values cannot all be zero.
    """
    if mode is ColorToGrayMode.COMBINE:
        output = combine_color_to_gray(image, channel_indices, contributions)
        return with_image_payload_data(
            image,
            output,
            metadata=color_to_gray_combine_output_metadata(image),
        )
    image_data = image_payload_data(image)
    outputs = split_color_to_gray(image, image_type, channel_indices)
    if image_type is ImageChannelType.RGB:
        return pack_aligned_image_outputs(
            tuple(
                image_payload_metadata(image).project_channel_payload(
                    source_payload=image,
                    source_data=image_data,
                    channel_index=channel_index,
                    channel_data=output,
                    channel_axis=-1,
                )
                for channel_index, output in zip(channel_indices, outputs, strict=True)
            )
        )
    return pack_aligned_image_outputs(
        tuple(with_image_payload_data(image, output) for output in outputs)
    )


def _invert_for_printing_channels(
    image: np.ndarray,
    *,
    input_mode: InvertInputMode,
    use_red_input: bool,
    use_green_input: bool,
    use_blue_input: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CellProfiler red, green, and blue input planes."""

    image_data = np.asarray(image_payload_data(image))
    if input_mode is InvertInputMode.COLOR:
        if image_data.ndim != 3 or image_data.shape[-1] != 3:
            raise ValueError(
                "InvertForPrinting color input requires an HxWx3 image, got "
                f"shape {image_data.shape!r}."
            )
        return tuple(image_data[..., index] for index in range(3))

    enabled_inputs = (use_red_input, use_green_input, use_blue_input)
    enabled_count = sum(enabled_inputs)
    if enabled_count == 0:
        raise ValueError(
            "InvertForPrinting grayscale input requires at least one enabled image."
        )
    grayscale_stack = (
        image_data[np.newaxis, ...]
        if enabled_count == 1 and image_data.ndim == 2
        else image_data
    )
    if grayscale_stack.ndim != 3 or grayscale_stack.shape[0] != enabled_count:
        raise ValueError(
            "InvertForPrinting grayscale input requires one leading plane per "
            f"enabled channel; expected {enabled_count}, got shape "
            f"{grayscale_stack.shape!r}."
        )
    height, width = grayscale_stack.shape[1:]
    zero = np.zeros((height, width), dtype=grayscale_stack.dtype)
    planes = iter(grayscale_stack)
    return tuple(next(planes) if enabled else zero for enabled in enabled_inputs)


def _invert_for_printing_result(
    image: np.ndarray,
    *,
    input_mode: InvertInputMode,
    use_red_input: bool,
    use_green_input: bool,
    use_blue_input: bool,
    output_mode: OutputMode,
    output_red: bool,
    output_green: bool,
    output_blue: bool,
    red_output_name: str,
    green_output_name: str,
    blue_output_name: str,
    color_output_name: str,
) -> RuntimeArrayData | AlignedImageStack | None:
    red_image, green_image, blue_image = _invert_for_printing_channels(
        image,
        input_mode=input_mode,
        use_red_input=use_red_input,
        use_green_input=use_green_input,
        use_blue_input=use_blue_input,
    )
    inverted_channels = (
        ((1.0 - green_image) * (1.0 - blue_image)).astype(np.float32),
        ((1.0 - red_image) * (1.0 - blue_image)).astype(np.float32),
        ((1.0 - red_image) * (1.0 - green_image)).astype(np.float32),
    )
    if output_mode is OutputMode.COLOR:
        outputs = (
            with_image_payload_data(
                image,
                np.stack(inverted_channels, axis=-1),
                metadata=image_payload_metadata(image).replace_fields(
                    source_channel_axis=-1
                ),
            ),
        )
        output_names = (color_output_name,)
    else:
        enabled_outputs = (output_red, output_green, output_blue)
        output_names = tuple(
            name
            for enabled, name in zip(
                enabled_outputs,
                (red_output_name, green_output_name, blue_output_name),
                strict=True,
            )
            if enabled
        )
        output_metadata = image_payload_metadata(image).without_source_channel_axis()
        outputs = tuple(
            with_image_payload_data(
                image,
                output,
                metadata=output_metadata,
            )
            for enabled, output in zip(
                enabled_outputs,
                inverted_channels,
                strict=True,
            )
            if enabled
        )
    if not outputs:
        return None
    if any(not name.strip() for name in output_names):
        raise ValueError("InvertForPrinting output image names cannot be blank.")
    return pack_aligned_image_outputs(
        outputs,
        slice_contexts=tuple(
            AlignedImageSliceContext.independent_main_flow(
                name,
                artifact_kind=ImageArtifactType.value,
            )
            for name in output_names
        ),
    )


@composed_image_payload
@numpy(contract=ProcessingContract.PURE_3D)
def invert_for_printing(
    image: np.ndarray,
    input_mode: InvertInputMode = InvertInputMode.COLOR,
    use_red_input: bool = True,
    use_green_input: bool = True,
    use_blue_input: bool = True,
    output_mode: OutputMode = OutputMode.COLOR,
    output_red: bool = True,
    output_green: bool = True,
    output_blue: bool = True,
    red_output_name: str = "InvertedRed",
    green_output_name: str = "InvertedGreen",
    blue_output_name: str = "InvertedBlue",
    color_output_name: str = "InvertedColor",
) -> np.ndarray | AlignedImageStack | None:
    """Invert fluorescent channels into the exact selected named outputs."""

    return _invert_for_printing_result(
        image,
        input_mode=input_mode,
        use_red_input=use_red_input,
        use_green_input=use_green_input,
        use_blue_input=use_blue_input,
        output_mode=output_mode,
        output_red=output_red,
        output_green=output_green,
        output_blue=output_blue,
        red_output_name=red_output_name,
        green_output_name=green_output_name,
        blue_output_name=blue_output_name,
        color_output_name=color_output_name,
    )


@composed_image_payload
@numpy(contract=ProcessingContract.PURE_3D)
def invert_for_printing_grayscale(
    image: np.ndarray,
    input_mode: InvertInputMode = InvertInputMode.COLOR,
    use_red_input: bool = True,
    use_green_input: bool = True,
    use_blue_input: bool = True,
    output_mode: OutputMode = OutputMode.GRAYSCALE,
    output_red: bool = True,
    output_green: bool = True,
    output_blue: bool = True,
    red_output_name: str = "InvertedRed",
    green_output_name: str = "InvertedGreen",
    blue_output_name: str = "InvertedBlue",
    color_output_name: str = "InvertedColor",
) -> np.ndarray | AlignedImageStack:
    """Invert into one or more enabled grayscale output channels."""

    if output_mode is not OutputMode.GRAYSCALE:
        raise ValueError("invert_for_printing_grayscale requires grayscale output.")
    result = _invert_for_printing_result(
        image,
        input_mode=input_mode,
        use_red_input=use_red_input,
        use_green_input=use_green_input,
        use_blue_input=use_blue_input,
        output_mode=output_mode,
        output_red=output_red,
        output_green=output_green,
        output_blue=output_blue,
        red_output_name=red_output_name,
        green_output_name=green_output_name,
        blue_output_name=blue_output_name,
        color_output_name=color_output_name,
    )
    if result is None:
        raise ValueError(
            "invert_for_printing_grayscale requires at least one enabled output."
        )
    return result


@composed_image_payload
@numpy(contract=ProcessingContract.PURE_3D)
def invert_for_printing_without_output(
    image: np.ndarray,
    input_mode: InvertInputMode = InvertInputMode.COLOR,
    use_red_input: bool = True,
    use_green_input: bool = True,
    use_blue_input: bool = True,
    output_mode: OutputMode = OutputMode.GRAYSCALE,
    output_red: bool = False,
    output_green: bool = False,
    output_blue: bool = False,
    red_output_name: str = "InvertedRed",
    green_output_name: str = "InvertedGreen",
    blue_output_name: str = "InvertedBlue",
    color_output_name: str = "InvertedColor",
) -> None:
    """Execute the valid grayscale topology that retains no output images."""

    result = _invert_for_printing_result(
        image,
        input_mode=input_mode,
        use_red_input=use_red_input,
        use_green_input=use_green_input,
        use_blue_input=use_blue_input,
        output_mode=output_mode,
        output_red=output_red,
        output_green=output_green,
        output_blue=output_blue,
        red_output_name=red_output_name,
        green_output_name=green_output_name,
        blue_output_name=blue_output_name,
        color_output_name=color_output_name,
    )
    if result is not None:
        raise ValueError(
            "invert_for_printing_without_output requires all grayscale outputs disabled."
        )
    return None


def combine_color_to_gray(
    image: np.ndarray,
    channel_indices: tuple[int, ...],
    contributions: tuple[float, ...],
) -> np.ndarray:
    if len(channel_indices) != len(contributions):
        raise ValueError("channel_indices and contributions must have same length.")
    color_stack = nhwc_color_stack(image)
    channels = np.asarray(channel_indices, dtype=int)
    weights = np.asarray(contributions, dtype=float) / float(sum(contributions))
    result = np.sum(
        color_stack[..., channels] * weights[np.newaxis, np.newaxis, np.newaxis, :],
        axis=3,
    )
    return restore_color_to_gray_shape(image, result)


def color_to_gray_combine_output_metadata(image: np.ndarray):
    """Return metadata for a color-to-grayscale semantic collapse."""
    return (
        image_payload_metadata(image)
        .without_unit_interval_intensity_scale()
        .without_source_channel_axis()
    )


def split_color_to_gray(
    image: np.ndarray, image_type: ImageChannelType, channel_indices: tuple[int, ...]
) -> tuple[np.ndarray, ...]:
    color_stack = nhwc_color_stack(image).astype(np.float32)
    source_stack = (
        rgb_to_hsv_stack(color_stack)
        if image_type is ImageChannelType.HSV
        else color_stack
    )
    return tuple(
        (
            restore_color_to_gray_shape(
                image, color_to_gray_channel(source_stack, index)
            )
            for index in channel_indices
        )
    )


def color_to_gray_channel(color_stack: np.ndarray, channel_index: int) -> np.ndarray:
    if channel_index >= color_stack.shape[-1]:
        raise ValueError(
            f"ColorToGray channel index {channel_index} is outside payload with {color_stack.shape[-1]} channels."
        )
    return color_stack[..., channel_index]


def nhwc_color_stack(image: np.ndarray) -> np.ndarray:
    """Return NHWC pixels from explicitly declared image layout metadata."""
    image_data = np.asarray(image_payload_data(image))
    metadata = image_payload_metadata(image)
    channel_axis = metadata.normalized_source_channel_axis(image_data)
    if channel_axis is None:
        raise ValueError(
            "ColorToGray requires ImagePayloadMetadata.source_channel_axis; "
            "array shape cannot declare color semantics."
        )
    channel_last = np.moveaxis(image_data, channel_axis, -1)
    if metadata.is_declared_source_channel_plane(image_data):
        if channel_last.ndim != 3:
            raise ValueError(
                "ColorToGray image-plane storage must have one channel and two "
                f"spatial axes, got shape {image_data.shape!r}."
            )
        return channel_last[np.newaxis, ...]
    if channel_axis == 0:
        raise ValueError(
            "ColorToGray metadata cannot declare the leading plane axis as its "
            "source channel axis."
        )
    if channel_last.ndim != 4:
        raise ValueError(
            "ColorToGray image-stack storage must have one plane, one channel, "
            f"and two spatial axes, got shape {image_data.shape!r}."
        )
    return channel_last


def restore_color_to_gray_shape(original: np.ndarray, stack: np.ndarray) -> np.ndarray:
    metadata = image_payload_metadata(original)
    if metadata.is_declared_source_channel_plane(original):
        if stack.shape[0] != 1:
            raise ValueError(
                "ColorToGray plane output must contain exactly one projected "
                f"plane, got shape {stack.shape!r}."
            )
        return stack[0]
    if metadata.is_declared_source_channel_stack(original):
        return stack
    raise ValueError(
        "ColorToGray output restoration requires declared source channel and "
        "runtime plane-axis metadata."
    )


def normalized_color_to_gray_weights(
    contributions: tuple[float, ...],
) -> tuple[float, ...]:
    total = sum(contributions)
    if total == 0:
        raise ValueError("Contributions cannot all be zero.")
    return tuple((float(contribution) / total for contribution in contributions))


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
        delta, max_channel, out=np.zeros_like(delta), where=max_channel != 0
    )
    hue = np.zeros_like(red)
    nonzero_delta = delta != 0
    red_is_max = (max_channel == red) & nonzero_delta
    green_is_max = (max_channel == green) & nonzero_delta
    blue_is_max = (max_channel == blue) & nonzero_delta
    hue[red_is_max] = (green[red_is_max] - blue[red_is_max]) / delta[red_is_max] % 6
    hue[green_is_max] = (blue[green_is_max] - red[green_is_max]) / delta[
        green_is_max
    ] + 2
    hue[blue_is_max] = (red[blue_is_max] - green[blue_is_max]) / delta[blue_is_max] + 4
    hue = hue / 6.0
    return np.stack((hue, saturation, value), axis=-1).astype(np.float32)


@numpy(contract=ProcessingContract.FLEXIBLE)
def unmix_colors(
    image: np.ndarray,
    stain_names: Sequence[StainType] = (),
    custom_absorbances: Sequence[Sequence[float] | None] = (),
    stain1: StainType = StainType.HEMATOXYLIN,
    stain2: StainType = StainType.EOSIN,
    stain3: StainType | None = None,
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
) -> RuntimeArrayData | AlignedImageStack:
    """Unmix one RGB image into one image per configured CellProfiler stain row.

    Args:
        stain_names: Ordered stain types to unmix into separate output images.
        custom_absorbances: Optional RGB optical-density vector corresponding to
            each stain in ``stain_names``.
        stain1: First stain used by the single-output legacy form.
        stain2: Second stain used by the single-output legacy form.
        stain3: Optional third stain included in the legacy unmixing matrix.
        output_stain_index: Zero-based legacy stain output to return.
        custom_red_absorbance_1: Red optical-density component for custom stain 1.
        custom_green_absorbance_1: Green optical-density component for custom
            stain 1.
        custom_blue_absorbance_1: Blue optical-density component for custom stain 1.
        custom_red_absorbance_2: Red optical-density component for custom stain 2.
        custom_green_absorbance_2: Green optical-density component for custom
            stain 2.
        custom_blue_absorbance_2: Blue optical-density component for custom stain 2.
        custom_red_absorbance_3: Red optical-density component for custom stain 3.
        custom_green_absorbance_3: Green optical-density component for custom
            stain 3.
        custom_blue_absorbance_3: Blue optical-density component for custom stain 3.
    """
    rgb_image = _as_rgb_image(image)
    output_metadata = image_payload_metadata(image).without_source_channel_axis()
    if stain_names:
        return pack_aligned_image_outputs(
            tuple(
                with_image_payload_data(
                    image,
                    output,
                    metadata=output_metadata,
                )
                for output in _unmix_stain_outputs(
                    rgb_image,
                    _stain_definitions(stain_names, custom_absorbances),
                )
            )
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
            f"output_stain_index must be in [0, {len(outputs) - 1}], got {output_stain_index}."
        )
    return with_image_payload_data(
        image,
        outputs[output_stain_index],
        metadata=output_metadata,
    )


def _stain_definitions(
    stain_names: Sequence[StainType],
    custom_absorbances: Sequence[Sequence[float] | None],
) -> tuple[StainDefinition, ...]:
    if len(stain_names) != len(custom_absorbances):
        raise ValueError(
            "UnmixColors stain_names and custom_absorbances must have the same length."
        )
    return tuple(
        (
            StainDefinition(
                stain=stain_name,
                custom_absorbance=_coerce_custom_absorbance(custom_absorbance),
            )
            for stain_name, custom_absorbance in zip(
                stain_names, custom_absorbances, strict=True
            )
        )
    )


def _legacy_stain_definitions(
    *,
    stain1: StainType,
    stain2: StainType,
    stain3: StainType | None,
    custom_absorbances: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[StainDefinition, ...]:
    stains = (stain1, stain2, stain3)
    return tuple(
        (
            StainDefinition(
                stain=stain,
                custom_absorbance=custom_absorbances[index],
            )
            for index, stain in enumerate(stains)
            if stain is not None
        )
    )


def _unmix_stain_outputs(
    image: np.ndarray, definitions: tuple[StainDefinition, ...]
) -> tuple[np.ndarray, ...]:
    if not definitions:
        raise ValueError("UnmixColors requires at least one stain definition.")
    inverse_matrix = np.linalg.pinv(
        np.asarray([definition.absorbance for definition in definitions])
    )
    return tuple(
        (
            _run_unmix_output(image, inverse_matrix[:, index])
            for index in range(len(definitions))
        )
    )


def _run_unmix_output(image: np.ndarray, inverse_absorbances: np.ndarray) -> np.ndarray:
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
        f"UnmixColors expects an RGB image with three color channels on the first or last channel axis, got shape {array.shape}."
    )


def _normalized_absorbance(absorbance: Sequence[float]) -> np.ndarray:
    vector = np.asarray(tuple((float(channel) for channel in absorbance)))
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
    return (float(red), float(green), float(blue))


class InvertForPrintingModule(
    CellProfilerModule,
):
    module_name = "InvertForPrinting"
    function_name = "invert_for_printing"
    function_variants = (
        "invert_for_printing_grayscale",
        "invert_for_printing_without_output",
    )
    validated = True
    group_by = GroupBy.SITE
    confidence = 1.0

    input_mode_setting = "Input image type"
    red_input_flag_setting = "Use a red image?"
    red_input_setting = "Select the red image"
    green_input_flag_setting = "Use a green image?"
    green_input_setting = "Select the green image"
    blue_input_flag_setting = "Use a blue image?"
    blue_input_setting = "Select the blue image"
    color_input_setting = "Select the color image"
    output_mode_setting = "Output image type"
    red_output_flag_setting = 'Select "*Yes*" to produce a red image.'
    red_output_setting = "Name the red image"
    green_output_flag_setting = 'Select "*Yes*" to produce a green image.'
    green_output_setting = "Name the green image"
    blue_output_flag_setting = 'Select "*Yes*" to produce a blue image.'
    blue_output_setting = "Name the blue image"
    color_output_setting = "Name the inverted color image"

    @dataclass(frozen=True, slots=True)
    class Channel:
        """One exact grayscale input/output channel declaration."""

        input_flag_setting: str
        input_binding: SettingToKeywordBinding
        output_flag_setting: str
        output_binding: SettingToKeywordBinding

    red_channel = Channel(
        red_input_flag_setting,
        SettingToKeywordBinding.input(
            red_input_setting,
            ImageArtifactType,
        ),
        red_output_flag_setting,
        SettingToKeywordBinding.output(
            red_output_setting,
            ImageArtifactType,
            "red_output_name",
        ),
    )
    green_channel = Channel(
        green_input_flag_setting,
        SettingToKeywordBinding.input(
            green_input_setting,
            ImageArtifactType,
        ),
        green_output_flag_setting,
        SettingToKeywordBinding.output(
            green_output_setting,
            ImageArtifactType,
            "green_output_name",
        ),
    )
    blue_channel = Channel(
        blue_input_flag_setting,
        SettingToKeywordBinding.input(
            blue_input_setting,
            ImageArtifactType,
        ),
        blue_output_flag_setting,
        SettingToKeywordBinding.output(
            blue_output_setting,
            ImageArtifactType,
            "blue_output_name",
        ),
    )
    channels = (red_channel, green_channel, blue_channel)
    color_input_binding = SettingToKeywordBinding.input(
        color_input_setting,
        ImageArtifactType,
    )
    color_output_binding = SettingToKeywordBinding.output(
        color_output_setting,
        ImageArtifactType,
        "color_output_name",
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        *(channel.input_binding for channel in channels),
        color_input_binding,
        *(channel.output_binding for channel in channels),
        color_output_binding,
        SettingToKeywordBinding(
            input_mode_setting,
            "input_mode",
            cellprofiler_enum_setting_parser(InvertInputMode),
        ),
        *(
            SettingToKeywordBinding(
                channel.input_flag_setting,
                parameter_name,
                parse_cellprofiler_bool,
            )
            for channel, parameter_name in zip(
                channels,
                ("use_red_input", "use_green_input", "use_blue_input"),
                strict=True,
            )
        ),
        SettingToKeywordBinding(
            output_mode_setting,
            "output_mode",
            cellprofiler_enum_setting_parser(OutputMode),
        ),
        *(
            SettingToKeywordBinding(
                channel.output_flag_setting,
                parameter_name,
                parse_cellprofiler_bool,
            )
            for channel, parameter_name in zip(
                channels,
                ("output_red", "output_green", "output_blue"),
                strict=True,
            )
        ),
    )

    @classmethod
    def input_mode(cls, module: "ModuleBlock") -> InvertInputMode:
        return coerce_cellprofiler_enum(
            InvertInputMode,
            required_setting_value(module, cls.input_mode_setting),
        )

    @classmethod
    def output_mode(cls, module: "ModuleBlock") -> OutputMode:
        return coerce_cellprofiler_enum(
            OutputMode,
            required_setting_value(module, cls.output_mode_setting),
        )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        if cls.input_mode(module) is InvertInputMode.COLOR:
            active_inputs = (cls.color_input_binding,)
        else:
            active_inputs = tuple(
                channel.input_binding
                for channel in cls.channels
                if parse_cellprofiler_bool(
                    required_setting_value(module, channel.input_flag_setting)
                )
            )
        if not active_inputs:
            raise ValueError(
                f"InvertForPrinting({module.module_num}) grayscale input requires "
                "at least one enabled image."
            )
        if cls.output_mode(module) is OutputMode.COLOR:
            active_outputs = (cls.color_output_binding,)
        else:
            active_outputs = tuple(
                channel.output_binding
                for channel in cls.channels
                if parse_cellprofiler_bool(
                    required_setting_value(module, channel.output_flag_setting)
                )
            )
        selected = frozenset((*active_inputs, *active_outputs))
        declared_images = frozenset(
            cls.declared_artifact_bindings(artifact_type=ImageArtifactType)
        )
        return tuple(
            binding
            for binding in bindings
            if binding not in declared_images or binding in selected
        )

    @classmethod
    def artifact_output_relations(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        binding,
        name,
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Use the first active input as the composed output's parent context."""

        del cls, module, invocation_key, step_context, binding, name, output_position
        image_inputs = artifact_inputs.for_artifact_type(ImageArtifactType).specs
        if not image_inputs:
            raise ValueError("InvertForPrinting requires at least one image input.")
        return (GroupLineageSourceRelation(source=image_inputs[0].ref()),)

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract,
        source_bindings,
    ) -> Callable[..., object]:
        """Select the callable from the declared output mode and channel flags."""

        del contract, source_bindings
        mode = cls.output_mode(module)
        if mode is OutputMode.COLOR:
            return cls.require_callable()
        has_grayscale_output = any(
            parse_cellprofiler_bool(
                required_setting_value(module, channel.output_flag_setting)
            )
            for channel in cls.channels
        )
        return cls.require_callable(
            cls.function_variants[0]
            if has_grayscale_output
            else cls.function_variants[1]
        )
