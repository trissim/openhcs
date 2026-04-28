"""Typed lowering for CellProfiler MeasureImageAreaOccupied settings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .parser import ModuleBlock, ModuleSetting
from .setting_names import block_setting_value, repeating_setting_blocks


AREA_OCCUPIED_MODE_SETTING = (
    "Measure the area occupied in a binary image, or in objects?"
)
AREA_OCCUPIED_BINARY_IMAGE_SETTING = "Select a binary image to measure"
AREA_OCCUPIED_OBJECTS_SETTING = "Select objects to measure"
AREA_OCCUPIED_RETAIN_IMAGE_SETTING = (
    "Retain a binary image of the object regions?"
)
AREA_OCCUPIED_OUTPUT_IMAGE_SETTING = "Name the output binary image"


class AreaOccupiedOperand(str, Enum):
    """Closed CellProfiler operand family for area-occupied rows."""

    BINARY_IMAGE = "binary_image"
    OBJECTS = "objects"

    @classmethod
    def from_literal(cls, value: str) -> "AreaOccupiedOperand":
        normalized = value.strip().lower()
        if "binary" in normalized:
            return cls.BINARY_IMAGE
        if "object" in normalized:
            return cls.OBJECTS
        raise ValueError(f"Unsupported MeasureImageAreaOccupied mode {value!r}.")


@dataclass(frozen=True, slots=True)
class AreaOccupiedMeasurementRow:
    """One ordered MeasureImageAreaOccupied row lowered from CellProfiler settings."""

    operand: AreaOccupiedOperand
    binary_image_name: str | None
    objects_name: str | None
    retained_image_name: str | None

    @classmethod
    def from_block(
        cls,
        module: ModuleBlock,
        block: Sequence[ModuleSetting],
    ) -> "AreaOccupiedMeasurementRow":
        row = cls(
            operand=AreaOccupiedOperand.from_literal(
                block_setting_value(block, AREA_OCCUPIED_MODE_SETTING)
            ),
            binary_image_name=_optional_symbol_value(
                block_setting_value(block, AREA_OCCUPIED_BINARY_IMAGE_SETTING)
            ),
            objects_name=_optional_symbol_value(
                block_setting_value(block, AREA_OCCUPIED_OBJECTS_SETTING)
            ),
            retained_image_name=_retained_area_occupied_image_name(block),
        )
        row.validate(module)
        return row

    @property
    def input_name(self) -> str:
        if self.operand is AreaOccupiedOperand.BINARY_IMAGE:
            if self.binary_image_name is None:
                raise RuntimeError("Binary area-occupied row has no image input.")
            return self.binary_image_name
        if self.objects_name is None:
            raise RuntimeError("Object area-occupied row has no object input.")
        return self.objects_name

    def validate(self, module: ModuleBlock) -> None:
        if self.operand is AreaOccupiedOperand.BINARY_IMAGE:
            if self.binary_image_name is None:
                raise ValueError(
                    f"Module {module.name}({module.module_num}) has a binary "
                    "area-occupied row with no binary image input."
                )
            return
        if self.objects_name is None:
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an object "
                "area-occupied row with no object input."
            )


def area_occupied_rows(module: ModuleBlock) -> tuple[AreaOccupiedMeasurementRow, ...]:
    """Return ordered MeasureImageAreaOccupied rows from a parsed module."""
    return tuple(
        AreaOccupiedMeasurementRow.from_block(module, block)
        for block in repeating_setting_blocks(
            module.iter_settings(),
            start_name=AREA_OCCUPIED_MODE_SETTING,
        )
    )


def area_occupied_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    """Return literal kwargs for the generic absorbed area-occupied function."""
    rows = area_occupied_rows(module)
    return {
        "operand_choices": tuple(row.operand.value for row in rows),
        "input_names": tuple(row.input_name for row in rows),
        "retained_image_names": tuple(row.retained_image_name for row in rows),
    }


def _retained_area_occupied_image_name(
    block: Sequence[ModuleSetting],
) -> str | None:
    retain = block_setting_value(block, AREA_OCCUPIED_RETAIN_IMAGE_SETTING)
    if retain.strip().lower() != "yes":
        return None
    return _optional_symbol_value(
        block_setting_value(block, AREA_OCCUPIED_OUTPUT_IMAGE_SETTING)
    )


def _optional_symbol_value(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"leave this black", "none", "do not use"}:
        return None
    return normalized
