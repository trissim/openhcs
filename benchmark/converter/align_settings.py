"""Typed lowering for legacy CellProfiler Align settings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .parser import ModuleBlock


ALIGN_METHOD_SETTING = "Select the alignment method"
ALIGN_V2_CROP_SETTING = "Crop output images to retain just the aligned regions?"
ALIGN_CROP_MODE_SETTING = "Crop mode"
ALIGN_FIRST_INPUT_SETTING = "Select the first input image"
ALIGN_FIRST_OUTPUT_SETTING = "Name the first output image"
ALIGN_SECOND_INPUT_SETTING = "Select the second input image"
ALIGN_SECOND_OUTPUT_SETTING = "Name the second output image"


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

    @property
    def input_names(self) -> tuple[str, str]:
        return (self.first_input_name, self.second_input_name)

    @property
    def output_names(self) -> tuple[str, str]:
        return (self.first_output_name, self.second_output_name)


def align_image_plan(module: ModuleBlock) -> AlignImagePlan:
    """Return typed image IO names for a parsed Align module."""
    return AlignImagePlan(
        first_input_name=_required_setting(module, ALIGN_FIRST_INPUT_SETTING),
        first_output_name=_required_setting(module, ALIGN_FIRST_OUTPUT_SETTING),
        second_input_name=_required_setting(module, ALIGN_SECOND_INPUT_SETTING),
        second_output_name=_required_setting(module, ALIGN_SECOND_OUTPUT_SETTING),
    )


def align_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    """Return literal kwargs for the absorbed Align function."""
    return {
        "method": module.get_setting(ALIGN_METHOD_SETTING, "Mutual Information"),
        "crop_mode": _align_crop_mode(module).value,
    }


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
