"""Typed lowering for legacy CellProfiler Align settings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .cellprofiler_literals import cellprofiler_enum_from_literal
from .parser import ModuleBlock
from .setting_names import optional_setting_value, required_setting_value, setting_values


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
        return cellprofiler_enum_from_literal(cls, value)


class AlignCropMode(str, Enum):
    """Closed crop modes from legacy CellProfiler Align."""

    KEEP_SIZE = "Keep size"
    CROP_TO_ALIGNED_REGION = "Crop to aligned region"
    PAD_IMAGES = "Pad images"

    @classmethod
    def from_literal(cls, value: str) -> "AlignCropMode":
        return cellprofiler_enum_from_literal(
            cls,
            value,
            aliases={
                "yes": cls.CROP_TO_ALIGNED_REGION,
                "true": cls.CROP_TO_ALIGNED_REGION,
                "no": cls.KEEP_SIZE,
                "false": cls.KEEP_SIZE,
            },
        )


@dataclass(frozen=True, slots=True)
class AlignAdditionalImagePlan:
    """Additional-image IO rows for one Align module."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    alignment_modes: tuple[AlignAdditionalMode, ...]

    @classmethod
    def from_module(cls, module: ModuleBlock) -> "AlignAdditionalImagePlan":
        input_names = setting_values(module, ALIGN_ADDITIONAL_INPUT_SETTING)
        output_names = setting_values(module, ALIGN_ADDITIONAL_OUTPUT_SETTING)
        alignment_modes = tuple(
            AlignAdditionalMode.from_literal(value)
            for value in setting_values(module, ALIGN_ADDITIONAL_MODE_SETTING)
        )
        plan = cls(
            input_names=input_names,
            output_names=output_names,
            alignment_modes=alignment_modes
            or (AlignAdditionalMode.SIMILARLY,) * len(input_names),
        )
        plan.validate(module)
        return plan

    def validate(self, module: ModuleBlock) -> None:
        if len(self.input_names) != len(self.output_names):
            raise ValueError(
                f"Module Align({module.module_num}) has {len(self.input_names)} "
                f"additional inputs but {len(self.output_names)} additional outputs."
            )
        if len(self.alignment_modes) != len(self.input_names):
            raise ValueError(
                f"Module Align({module.module_num}) has {len(self.alignment_modes)} "
                f"additional alignment modes for {len(self.input_names)} "
                "additional images."
            )


@dataclass(frozen=True, slots=True)
class AlignCropModeSetting:
    """Legacy Align crop mode with CellProfiler revision fallback."""

    crop_mode_literal: str | None
    v2_crop_literal: str | None

    @classmethod
    def from_module(cls, module: ModuleBlock) -> "AlignCropModeSetting":
        return cls(
            crop_mode_literal=optional_setting_value(module, ALIGN_CROP_MODE_SETTING),
            v2_crop_literal=optional_setting_value(module, ALIGN_V2_CROP_SETTING),
        )

    @property
    def crop_mode(self) -> AlignCropMode:
        return AlignCropMode.from_literal(
            self.crop_mode_literal or self.v2_crop_literal or "No"
        )


@dataclass(frozen=True, slots=True)
class AlignImagePlan:
    """Image names consumed and produced by one Align module."""

    first_input_name: str
    first_output_name: str
    second_input_name: str
    second_output_name: str
    additional_images: AlignAdditionalImagePlan

    @classmethod
    def from_module(cls, module: ModuleBlock) -> "AlignImagePlan":
        return cls(
            first_input_name=required_setting_value(module, ALIGN_FIRST_INPUT_SETTING),
            first_output_name=required_setting_value(module, ALIGN_FIRST_OUTPUT_SETTING),
            second_input_name=required_setting_value(module, ALIGN_SECOND_INPUT_SETTING),
            second_output_name=required_setting_value(module, ALIGN_SECOND_OUTPUT_SETTING),
            additional_images=AlignAdditionalImagePlan.from_module(module),
        )

    @property
    def additional_input_names(self) -> tuple[str, ...]:
        return self.additional_images.input_names

    @property
    def additional_output_names(self) -> tuple[str, ...]:
        return self.additional_images.output_names

    @property
    def additional_alignment_modes(self) -> tuple[AlignAdditionalMode, ...]:
        return self.additional_images.alignment_modes

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
    return AlignImagePlan.from_module(module)


def align_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    """Return literal kwargs for the absorbed Align function."""
    image_plan = align_image_plan(module)
    kwargs: dict[str, Any] = {
        "method": optional_setting_value(module, ALIGN_METHOD_SETTING)
        or "Mutual Information",
        "crop_mode": AlignCropModeSetting.from_module(module).crop_mode.value,
    }
    if image_plan.additional_alignment_modes:
        kwargs["additional_alignment_modes"] = tuple(
            mode.value
            for mode in image_plan.additional_alignment_modes
        )
    return kwargs
