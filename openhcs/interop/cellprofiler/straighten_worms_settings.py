"""Typed lowering for CellProfiler StraightenWorms settings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.setting_names import (
    block_setting_value,
    optional_setting_value,
    repeating_setting_blocks,
    required_setting_value,
)
from openhcs.interop.cellprofiler.settings_binder import (
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
)


class StraightenWormsFlipMode(Enum):
    """CellProfiler StraightenWorms alignment mode."""

    NONE = "do_not_align"
    TOP = "top_brightest"
    BOTTOM = "bottom_brightest"
    MANUAL = "flip_manually"


STRAIGHTEN_WORMS_INPUT_OBJECTS_SETTING = "Select the input untangled worm objects"
STRAIGHTEN_WORMS_OUTPUT_OBJECTS_SETTING = "Name the output straightened worm objects"
STRAIGHTEN_WORMS_INPUT_IMAGE_SETTING = "Select an input image to straighten"
STRAIGHTEN_WORMS_OUTPUT_IMAGE_SETTING = "Name the output straightened image"


@dataclass(frozen=True, slots=True)
class StraightenWormsImageBinding:
    """One input image and its corresponding straightened image artifact."""

    input_image_name: str
    output_image_name: str


def straighten_worms_input_objects_name(module: ModuleBlock) -> str:
    return required_setting_value(module, STRAIGHTEN_WORMS_INPUT_OBJECTS_SETTING)


def straighten_worms_output_objects_name(module: ModuleBlock) -> str:
    return required_setting_value(module, STRAIGHTEN_WORMS_OUTPUT_OBJECTS_SETTING)


def straighten_worms_image_bindings(
    module: ModuleBlock,
) -> tuple[StraightenWormsImageBinding, ...]:
    return tuple(
        StraightenWormsImageBinding(
            input_image_name=block_setting_value(
                block,
                STRAIGHTEN_WORMS_INPUT_IMAGE_SETTING,
            ),
            output_image_name=block_setting_value(
                block,
                STRAIGHTEN_WORMS_OUTPUT_IMAGE_SETTING,
            ),
        )
        for block in repeating_setting_blocks(
            module.iter_settings(),
            start_name=STRAIGHTEN_WORMS_INPUT_IMAGE_SETTING,
        )
        if block_setting_value(block, STRAIGHTEN_WORMS_INPUT_IMAGE_SETTING)
        and block_setting_value(block, STRAIGHTEN_WORMS_OUTPUT_IMAGE_SETTING)
    )


def straighten_worms_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    _bind_optional_int(module, "Worm width", "worm_width", kwargs)
    _bind_optional_bool(
        module,
        "Measure intensity distribution?",
        "measure_intensity",
        kwargs,
    )
    _bind_optional_int(
        module,
        "Number of transverse segments",
        "number_of_segments",
        kwargs,
    )
    _bind_optional_int(
        module,
        "Number of longitudinal stripes",
        "number_of_stripes",
        kwargs,
    )
    alignment = optional_setting_value(module, "Align worms?")
    if alignment is not None:
        kwargs["flip_mode"] = coerce_cellprofiler_enum(
            StraightenWormsFlipMode,
            alignment,
        ).value
    return kwargs


def _bind_optional_int(
    module: ModuleBlock,
    setting_name: str,
    parameter_name: str,
    kwargs: dict[str, Any],
) -> None:
    value = optional_setting_value(module, setting_name)
    if value is not None:
        kwargs[parameter_name] = int(float(value))


def _bind_optional_bool(
    module: ModuleBlock,
    setting_name: str,
    parameter_name: str,
    kwargs: dict[str, Any],
) -> None:
    value = optional_setting_value(module, setting_name)
    if value is not None:
        kwargs[parameter_name] = parse_cellprofiler_bool(value)
