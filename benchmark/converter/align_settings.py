"""Typed lowering for legacy CellProfiler Align settings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .parser import ModuleBlock
from .setting_names import setting_values


ALIGN_METHOD_SETTING = "Select the alignment method"
ALIGN_V2_CROP_SETTING = "Crop output images to retain just the aligned regions?"
ALIGN_CROP_MODE_SETTING = "Crop mode"
ALIGN_FIRST_INPUT_SETTING = "Select the first input image"
ALIGN_FIRST_OUTPUT_SETTING = "Name the first output image"
ALIGN_SECOND_INPUT_SETTING = "Select the second input image"
ALIGN_SECOND_OUTPUT_SETTING = "Name the second output image"
ALIGN_ADDITIONAL_INPUT_SETTING = "Select the additional image"
ALIGN_ADDITIONAL_OUTPUT_SETTING = "Name the output image"
ALIGN_ADDITIONAL_MODE_SETTING = "Select how the alignment is to be applied"


class AlignAdditionalMode(str, Enum):
    """Closed modes for applying a primary Align transform to extra images."""

    SIMILARLY = "Similarly"

    @classmethod
    def from_literal(cls, value: str) -> "AlignAdditionalMode":
        normalized = value.strip().lower()
        for mode in cls:
            if normalized == mode.value.lower():
                return mode
        raise ValueError(f"Unsupported Align additional-image mode {value!r}.")


class AlignCropMode(str, Enum):
    """Closed crop modes from legacy CellProfiler Align."""

    KEEP_SIZE = "Keep size"
    CROP_TO_ALIGNED_REGION = "Crop to aligned region"
    PAD_IMAGES = "Pad images"

    @classmethod
    def from_literal(cls, value: str) -> "AlignCropMode":
        normalized = value.strip().lower()
        if normalized in {"yes", "true"}:
            return cls.CROP_TO_ALIGNED_REGION
        if normalized in {"no", "false"}:
            return cls.KEEP_SIZE
        for mode in cls:
            if normalized == mode.value.lower():
                return mode
        raise ValueError(f"Unsupported Align crop mode {value!r}.")


@dataclass(frozen=True, slots=True)
class AlignImagePlan:
    """Image names consumed and produced by one Align module."""

    first_input_name: str
    first_output_name: str
    second_input_name: str
    second_output_name: str
    additional_input_names: tuple[str, ...] = ()
    additional_output_names: tuple[str, ...] = ()
    additional_alignment_modes: tuple[AlignAdditionalMode, ...] = ()

    @property
    def input_names(self) -> tuple[str, ...]:
        return (
            self.first_input_name,
            self.second_input_name,
            *self.additional_input_names,
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return (
            self.first_output_name,
            self.second_output_name,
            *self.additional_output_names,
        )


def align_image_plan(module: ModuleBlock) -> AlignImagePlan:
    """Return typed image IO names for a parsed Align module."""
    additional_input_names = _additional_image_inputs(module)
    additional_output_names = _additional_image_outputs(module)
    additional_alignment_modes = _additional_alignment_modes(module)
    if len(additional_input_names) != len(additional_output_names):
        raise ValueError(
            f"Module Align({module.module_num}) has "
            f"{len(additional_input_names)} additional inputs but "
            f"{len(additional_output_names)} additional outputs."
        )
    if len(additional_alignment_modes) not in (0, len(additional_input_names)):
        raise ValueError(
            f"Module Align({module.module_num}) has "
            f"{len(additional_alignment_modes)} additional alignment modes for "
            f"{len(additional_input_names)} additional images."
        )
    if not additional_alignment_modes:
        additional_alignment_modes = (
            AlignAdditionalMode.SIMILARLY,
        ) * len(additional_input_names)
    return AlignImagePlan(
        first_input_name=_required_setting(module, ALIGN_FIRST_INPUT_SETTING),
        first_output_name=_required_setting(module, ALIGN_FIRST_OUTPUT_SETTING),
        second_input_name=_required_setting(module, ALIGN_SECOND_INPUT_SETTING),
        second_output_name=_required_setting(module, ALIGN_SECOND_OUTPUT_SETTING),
        additional_input_names=additional_input_names,
        additional_output_names=additional_output_names,
        additional_alignment_modes=additional_alignment_modes,
    )


def align_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    """Return literal kwargs for the absorbed Align function."""
    image_plan = align_image_plan(module)
    kwargs: dict[str, Any] = {
        "method": module.get_setting(ALIGN_METHOD_SETTING, "Mutual Information"),
        "crop_mode": _align_crop_mode(module).value,
    }
    if image_plan.additional_alignment_modes:
        kwargs["additional_alignment_modes"] = tuple(
            mode.value
            for mode in image_plan.additional_alignment_modes
        )
    return kwargs


def _additional_image_inputs(module: ModuleBlock) -> tuple[str, ...]:
    return setting_values(module, ALIGN_ADDITIONAL_INPUT_SETTING)


def _additional_image_outputs(module: ModuleBlock) -> tuple[str, ...]:
    return setting_values(module, ALIGN_ADDITIONAL_OUTPUT_SETTING)


def _additional_alignment_modes(module: ModuleBlock) -> tuple[AlignAdditionalMode, ...]:
    return tuple(
        AlignAdditionalMode.from_literal(value)
        for value in setting_values(module, ALIGN_ADDITIONAL_MODE_SETTING)
    )


def _align_crop_mode(module: ModuleBlock) -> AlignCropMode:
    if (value := module.get_setting(ALIGN_CROP_MODE_SETTING)) and value.strip():
        return AlignCropMode.from_literal(value)
    return AlignCropMode.from_literal(
        module.get_setting(ALIGN_V2_CROP_SETTING, "No")
    )


def _required_setting(module: ModuleBlock, name: str) -> str:
    value = module.get_setting(name)
    if value is None or not value.strip():
        raise ValueError(
            f"Module {module.name}({module.module_num}) missing setting {name!r}."
        )
    return value.strip()
