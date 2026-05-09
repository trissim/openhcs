"""Typed lowering for CellProfiler MeasureImageAreaOccupied settings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openhcs.interop.cellprofiler.setting_names import (
    normalized_symbol_name,
    split_symbol_names,
)

from .parser import ModuleBlock, ModuleSetting
from .setting_names import (
    SettingNameFamily,
    block_setting_value,
    repeating_setting_blocks,
)


AREA_OCCUPIED_MODE_SETTING = SettingNameFamily(
    "Measure the area occupied in a binary image, or in objects?",
    aliases=("Measure the area occupied by",),
)
AREA_OCCUPIED_BINARY_IMAGE_SETTING = SettingNameFamily(
    "Select a binary image to measure",
    aliases=("Select binary images to measure",),
)
AREA_OCCUPIED_OBJECTS_SETTING = SettingNameFamily(
    "Select objects to measure",
    aliases=("Select object sets to measure",),
)
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
            binary_image_name=normalized_symbol_name(
                block_setting_value(block, AREA_OCCUPIED_BINARY_IMAGE_SETTING)
            ),
            objects_name=normalized_symbol_name(
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
    rows: list[AreaOccupiedMeasurementRow] = []
    for block in repeating_setting_blocks(
        module.iter_settings(),
        start_name=AREA_OCCUPIED_MODE_SETTING,
    ):
        row = AreaOccupiedMeasurementRow.from_block(module, block)
        rows.extend(
            _expanded_area_occupied_rows(
                module,
                row,
                binary_image_names=_split_symbol_values(
                    block_setting_value(block, AREA_OCCUPIED_BINARY_IMAGE_SETTING)
                ),
                object_names=_split_symbol_values(
                    block_setting_value(block, AREA_OCCUPIED_OBJECTS_SETTING)
                ),
            )
        )
    return tuple(rows)


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
    return normalized_symbol_name(
        block_setting_value(block, AREA_OCCUPIED_OUTPUT_IMAGE_SETTING)
    )


def _expanded_area_occupied_rows(
    module: ModuleBlock,
    row: AreaOccupiedMeasurementRow,
    *,
    binary_image_names: tuple[str, ...],
    object_names: tuple[str, ...],
) -> tuple[AreaOccupiedMeasurementRow, ...]:
    if row.operand is AreaOccupiedOperand.BINARY_IMAGE:
        names = binary_image_names or _required_single_name(
            module,
            row.binary_image_name,
            "binary image",
        )
        return tuple(
            AreaOccupiedMeasurementRow(
                operand=row.operand,
                binary_image_name=name,
                objects_name=None,
                retained_image_name=row.retained_image_name,
            )
            for name in names
        )
    names = object_names or _required_single_name(
        module,
        row.objects_name,
        "object",
    )
    return tuple(
        AreaOccupiedMeasurementRow(
            operand=row.operand,
            binary_image_name=None,
            objects_name=name,
            retained_image_name=row.retained_image_name,
        )
        for name in names
    )


def _split_symbol_values(value: str) -> tuple[str, ...]:
    return split_symbol_names(value)


def _required_single_name(
    module: ModuleBlock,
    value: str | None,
    role: str,
) -> tuple[str, ...]:
    if value is None:
        raise ValueError(
            f"Module {module.name}({module.module_num}) has an area-occupied "
            f"row with no {role} input."
        )
    return (value,)
