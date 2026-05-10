"""Typed lowering for CellProfiler MeasureImageAreaOccupied settings."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from .parser import ModuleBlock, ModuleSetting
from .setting_names import (
    SettingNameFamily,
    block_setting_value,
    normalized_symbol_name,
    repeating_setting_blocks,
    split_symbol_names,
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
class AreaOccupiedRepeatedSymbols:
    """CellProfiler repeated symbol values for one area-occupied row."""

    binary_image_names: tuple[str, ...]
    object_names: tuple[str, ...]

    @classmethod
    def from_block(
        cls,
        block: Sequence[ModuleSetting],
    ) -> "AreaOccupiedRepeatedSymbols":
        return cls(
            binary_image_names=split_symbol_names(
                block_setting_value(block, AREA_OCCUPIED_BINARY_IMAGE_SETTING)
            ),
            object_names=split_symbol_names(
                block_setting_value(block, AREA_OCCUPIED_OBJECTS_SETTING)
            ),
        )


@dataclass(frozen=True, slots=True)
class AreaOccupiedInputSelection(ABC, metaclass=AutoRegisterMeta):
    """Nominal source-selection semantics for one MeasureImageAreaOccupied row."""

    __registry_key__ = "operand"
    __skip_if_no_key__ = True

    operand: ClassVar[AreaOccupiedOperand | None] = None
    role: ClassVar[str | None] = None
    input_name: str | None

    @classmethod
    def from_operand(
        cls,
        module: ModuleBlock,
        operand: AreaOccupiedOperand,
        *,
        binary_image_name: str | None,
        objects_name: str | None,
    ) -> "AreaOccupiedInputSelection":
        for selection_type in cls.registered_selection_types():
            if selection_type.operand is operand:
                selection = selection_type.from_field_names(
                    binary_image_name=binary_image_name,
                    objects_name=objects_name,
                )
                selection.validate(module)
                return selection
        raise ValueError(f"Unsupported MeasureImageAreaOccupied operand {operand!r}.")

    @classmethod
    def registered_selection_types(cls) -> tuple[type["AreaOccupiedInputSelection"], ...]:
        return tuple(cls.__registry__.values())

    @classmethod
    def from_field_names(
        cls,
        *,
        binary_image_name: str | None,
        objects_name: str | None,
    ) -> "AreaOccupiedInputSelection":
        raise NotImplementedError

    def expanded_names(
        self,
        module: ModuleBlock,
        repeated_symbols: AreaOccupiedRepeatedSymbols,
    ) -> tuple[str, ...]:
        raise NotImplementedError

    def require_input_name(self, module: ModuleBlock) -> str:
        if self.input_name is None:
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an area-occupied "
                f"row with no {self.role} input."
            )
        return self.input_name

    def validate(self, module: ModuleBlock) -> None:
        self.require_input_name(module)


@dataclass(frozen=True, slots=True)
class BinaryImageAreaOccupiedInput(AreaOccupiedInputSelection):
    """Binary-image operand source for one MeasureImageAreaOccupied row."""

    operand: ClassVar[AreaOccupiedOperand] = AreaOccupiedOperand.BINARY_IMAGE
    role: ClassVar[str] = "binary image"

    @classmethod
    def from_field_names(
        cls,
        *,
        binary_image_name: str | None,
        objects_name: str | None,
    ) -> "BinaryImageAreaOccupiedInput":
        return cls(input_name=binary_image_name)

    def expanded_names(
        self,
        module: ModuleBlock,
        repeated_symbols: AreaOccupiedRepeatedSymbols,
    ) -> tuple[str, ...]:
        return repeated_symbols.binary_image_names or (self.require_input_name(module),)


@dataclass(frozen=True, slots=True)
class ObjectAreaOccupiedInput(AreaOccupiedInputSelection):
    """Object-label operand source for one MeasureImageAreaOccupied row."""

    operand: ClassVar[AreaOccupiedOperand] = AreaOccupiedOperand.OBJECTS
    role: ClassVar[str] = "object"

    @classmethod
    def from_field_names(
        cls,
        *,
        binary_image_name: str | None,
        objects_name: str | None,
    ) -> "ObjectAreaOccupiedInput":
        return cls(input_name=objects_name)

    def expanded_names(
        self,
        module: ModuleBlock,
        repeated_symbols: AreaOccupiedRepeatedSymbols,
    ) -> tuple[str, ...]:
        return repeated_symbols.object_names or (self.require_input_name(module),)


@dataclass(frozen=True, slots=True)
class AreaOccupiedRetainedImagePolicy:
    """Retained-output image policy for one MeasureImageAreaOccupied row."""

    retain_literal: str
    output_image_name: str

    @classmethod
    def from_block(
        cls,
        block: Sequence[ModuleSetting],
    ) -> "AreaOccupiedRetainedImagePolicy":
        return cls(
            retain_literal=block_setting_value(block, AREA_OCCUPIED_RETAIN_IMAGE_SETTING),
            output_image_name=block_setting_value(block, AREA_OCCUPIED_OUTPUT_IMAGE_SETTING),
        )

    @property
    def retained_image_name(self) -> str | None:
        if self.retain_literal.strip().lower() != "yes":
            return None
        return normalized_symbol_name(self.output_image_name)


@dataclass(frozen=True, slots=True)
class AreaOccupiedMeasurementRow:
    """One ordered MeasureImageAreaOccupied row lowered from CellProfiler settings."""

    selection: AreaOccupiedInputSelection
    retained_image_name: str | None

    @classmethod
    def from_block(
        cls,
        module: ModuleBlock,
        block: Sequence[ModuleSetting],
    ) -> "AreaOccupiedMeasurementRow":
        return cls(
            selection=AreaOccupiedInputSelection.from_operand(
                module,
                AreaOccupiedOperand.from_literal(
                    block_setting_value(block, AREA_OCCUPIED_MODE_SETTING)
                ),
                binary_image_name=normalized_symbol_name(
                    block_setting_value(block, AREA_OCCUPIED_BINARY_IMAGE_SETTING)
                ),
                objects_name=normalized_symbol_name(
                    block_setting_value(block, AREA_OCCUPIED_OBJECTS_SETTING)
                ),
            ),
            retained_image_name=AreaOccupiedRetainedImagePolicy.from_block(
                block
            ).retained_image_name,
        )

    @property
    def operand(self) -> AreaOccupiedOperand:
        operand = self.selection.operand
        if operand is None:
            raise RuntimeError("Area-occupied row has no operand.")
        return operand

    @property
    def binary_image_name(self) -> str | None:
        if self.selection.operand is AreaOccupiedOperand.BINARY_IMAGE:
            return self.selection.input_name
        return None

    @property
    def objects_name(self) -> str | None:
        if self.selection.operand is AreaOccupiedOperand.OBJECTS:
            return self.selection.input_name
        return None

    @property
    def input_name(self) -> str:
        if self.selection.input_name is None:
            raise RuntimeError("Area-occupied row has no input.")
        return self.selection.input_name

    def validate(self, module: ModuleBlock) -> None:
        self.selection.validate(module)

    def expanded(
        self,
        module: ModuleBlock,
        repeated_symbols: AreaOccupiedRepeatedSymbols,
    ) -> tuple["AreaOccupiedMeasurementRow", ...]:
        return tuple(
            AreaOccupiedMeasurementRow(
                selection=type(self.selection)(input_name=input_name),
                retained_image_name=self.retained_image_name,
            )
            for input_name in self.selection.expanded_names(module, repeated_symbols)
        )


def area_occupied_rows(module: ModuleBlock) -> tuple[AreaOccupiedMeasurementRow, ...]:
    """Return ordered MeasureImageAreaOccupied rows from a parsed module."""
    rows: list[AreaOccupiedMeasurementRow] = []
    for block in repeating_setting_blocks(
        module.iter_settings(),
        start_name=AREA_OCCUPIED_MODE_SETTING,
    ):
        row = AreaOccupiedMeasurementRow.from_block(module, block)
        rows.extend(row.expanded(module, AreaOccupiedRepeatedSymbols.from_block(block)))
    return tuple(rows)


def area_occupied_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    """Return literal kwargs for the generic absorbed area-occupied function."""
    rows = area_occupied_rows(module)
    return {
        "operand_choices": tuple(row.operand.value for row in rows),
        "input_names": tuple(row.input_name for row in rows),
        "retained_image_names": tuple(row.retained_image_name for row in rows),
    }
