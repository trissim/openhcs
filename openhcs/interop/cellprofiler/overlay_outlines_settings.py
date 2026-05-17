"""Typed lowering for CellProfiler OverlayOutlines settings."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from .parser import ModuleBlock, ModuleSetting
from .setting_names import (
    RepeatedSettingSequence,
    SettingNameFamily,
    block_setting_value,
    normalized_symbol_name,
    optional_setting_value,
    repeating_setting_blocks,
    required_setting_value,
    setting_values,
)


OVERLAY_BLANK_IMAGE_SETTING = "Display outlines on a blank image?"
OVERLAY_BASE_IMAGE_SETTING = "Select image on which to display outlines"
OVERLAY_OUTPUT_IMAGE_SETTING = "Name the output image"
OVERLAY_DISPLAY_MODE_SETTING = SettingNameFamily(
    "Outline display mode",
    aliases=("Select outline display mode",),
)
OVERLAY_MAX_TYPE_SETTING = "Select method to determine brightness of outlines"
OVERLAY_LINE_MODE_SETTING = "How to outline"
OVERLAY_OUTLINE_IMAGE_SETTING = SettingNameFamily(
    "Select outlines to display",
    aliases=("Select outline to display",),
)
OVERLAY_OBJECTS_SETTING = SettingNameFamily(
    "Select objects to display",
    aliases=("Select object to display",),
)
OVERLAY_SOURCE_KIND_SETTING = "Load outlines from an image or objects?"
OVERLAY_COLOR_SETTING = "Select outline color"


class OverlayOutlineSourceKind(str, Enum):
    """Closed CellProfiler family for one OverlayOutlines row input source."""

    IMAGE = "image"
    OBJECTS = "objects"

    @classmethod
    def from_literal(cls, value: str) -> "OverlayOutlineSourceKind":
        normalized = value.strip().lower()
        if normalized.startswith("image"):
            return cls.IMAGE
        if normalized.startswith("object"):
            return cls.OBJECTS
        raise ValueError(f"Unsupported OverlayOutlines source kind {value!r}.")


@dataclass(frozen=True, slots=True)
class OverlayOutlineSymbolPair(ABC, metaclass=AutoRegisterMeta):
    """Shared image/object symbol pair for one OverlayOutlines row."""

    __registry_key__ = "pair_role"
    __skip_if_no_key__ = True

    pair_role: ClassVar[str | None] = None
    image_name: str | None
    objects_name: str | None

    @classmethod
    def registered_pair_types(cls) -> tuple[type["OverlayOutlineSymbolPair"], ...]:
        return tuple(cls.__registry__.values())


@dataclass(frozen=True, slots=True)
class OverlayOutlineSourceFields(OverlayOutlineSymbolPair):
    """Raw source-selector fields for one OverlayOutlines row."""

    pair_role: ClassVar[str] = "source_fields"
    source_kind_literal: str

    @classmethod
    def from_literals(
        cls,
        image_name: str,
        objects_name: str,
        source_kind: str,
    ) -> "OverlayOutlineSourceFields":
        return cls(
            image_name=normalized_symbol_name(image_name),
            objects_name=normalized_symbol_name(objects_name),
            source_kind_literal=source_kind,
        )

    @property
    def source_kind(self) -> OverlayOutlineSourceKind:
        if self.source_kind_literal.strip():
            return OverlayOutlineSourceKind.from_literal(self.source_kind_literal)
        if self.image_name is not None and self.objects_name is None:
            return OverlayOutlineSourceKind.IMAGE
        return OverlayOutlineSourceKind.OBJECTS


@dataclass(frozen=True, slots=True)
class OverlayOutlineRow(OverlayOutlineSymbolPair):
    """One ordered OverlayOutlines row lowered from CellProfiler settings."""

    pair_role: ClassVar[str] = "row"
    source_kind: OverlayOutlineSourceKind
    color: str

    @classmethod
    def from_block(
        cls,
        module: ModuleBlock,
        block: Sequence[ModuleSetting],
    ) -> "OverlayOutlineRow":
        return cls.from_source_fields(
            module,
            OverlayOutlineSourceFields.from_literals(
                block_setting_value(block, OVERLAY_OUTLINE_IMAGE_SETTING),
                block_setting_value(block, OVERLAY_OBJECTS_SETTING),
                block_setting_value(block, OVERLAY_SOURCE_KIND_SETTING),
            ),
            color=block_setting_value(block, OVERLAY_COLOR_SETTING, default="Red"),
        )

    @classmethod
    def from_source_fields(
        cls,
        module: ModuleBlock,
        source_fields: OverlayOutlineSourceFields,
        *,
        color: str,
    ) -> "OverlayOutlineRow":
        row = cls(
            source_kind=source_fields.source_kind,
            image_name=source_fields.image_name,
            objects_name=source_fields.objects_name,
            color=color,
        )
        row.validate(module)
        return row

    @property
    def input_name(self) -> str:
        if self.source_kind is OverlayOutlineSourceKind.IMAGE:
            if self.image_name is None:
                raise RuntimeError("Image outline row has no image input.")
            return self.image_name
        if self.objects_name is None:
            raise RuntimeError("Object outline row has no object input.")
        return self.objects_name

    def validate(self, module: ModuleBlock) -> None:
        if self.source_kind is OverlayOutlineSourceKind.IMAGE:
            if self.image_name is None:
                raise ValueError(
                    f"Module {module.name}({module.module_num}) has an image "
                    "outline row with no outline image input."
                )
            return
        if self.objects_name is None:
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an object "
                "outline row with no object input."
            )


@dataclass(frozen=True, slots=True)
class RepeatedOverlaySetting(RepeatedSettingSequence):
    """CellProfiler repeated OverlayOutlines setting with last-value fallback."""


def overlay_outlines_uses_blank_image(module: ModuleBlock) -> bool:
    """Return whether OverlayOutlines should render on a generated blank image."""
    value = optional_setting_value(module, OVERLAY_BLANK_IMAGE_SETTING)
    return value is not None and value.strip().lower() == "yes"


def overlay_outlines_base_image_name(module: ModuleBlock) -> str | None:
    """Return the required base image symbol, unless blank-image mode is active."""
    if overlay_outlines_uses_blank_image(module):
        return None
    return required_setting_value(module, OVERLAY_BASE_IMAGE_SETTING)


def overlay_outlines_output_image_name(module: ModuleBlock) -> str:
    """Return the required OverlayOutlines output image symbol."""
    return required_setting_value(module, OVERLAY_OUTPUT_IMAGE_SETTING)


def overlay_outline_rows(module: ModuleBlock) -> tuple[OverlayOutlineRow, ...]:
    """Return ordered OverlayOutlines rows from a parsed module."""
    if module.iter_settings():
        rows = _ordered_overlay_rows(module)
    else:
        rows = OverlayOutlineRowsFromMapping(module).rows
    if not rows:
        raise ValueError(
            f"Module {module.name}({module.module_num}) declares no "
            "OverlayOutlines rows."
        )
    return rows


def overlay_outlines_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    """Return absorbed-function kwargs for ordered OverlayOutlines rows."""
    rows = overlay_outline_rows(module)
    return {
        "blank_image": overlay_outlines_uses_blank_image(module),
        "display_mode": optional_setting_value(module, OVERLAY_DISPLAY_MODE_SETTING)
        or "Color",
        "line_mode": optional_setting_value(module, OVERLAY_LINE_MODE_SETTING)
        or "Inner",
        "max_type": optional_setting_value(module, OVERLAY_MAX_TYPE_SETTING)
        or "Max of image",
        "outline_source_kinds": tuple(row.source_kind.value for row in rows),
        "outline_colors": tuple(row.color for row in rows),
    }


def _ordered_overlay_rows(module: ModuleBlock) -> tuple[OverlayOutlineRow, ...]:
    image_blocks = repeating_setting_blocks(
        module.iter_settings(),
        start_name=OVERLAY_OUTLINE_IMAGE_SETTING,
    )
    if image_blocks:
        return tuple(OverlayOutlineRow.from_block(module, block) for block in image_blocks)
    object_blocks = repeating_setting_blocks(
        module.iter_settings(),
        start_name=OVERLAY_OBJECTS_SETTING,
    )
    if object_blocks:
        return OverlayOutlineRowsFromMapping(module).rows
    return ()


@dataclass(frozen=True, slots=True)
class OverlayOutlineRowsFromMapping:
    """OverlayOutlines rows from legacy mapping-shaped parsed module settings."""

    module: ModuleBlock

    @property
    def rows(self) -> tuple[OverlayOutlineRow, ...]:
        image_names = setting_values(self.module, OVERLAY_OUTLINE_IMAGE_SETTING)
        object_names = setting_values(self.module, OVERLAY_OBJECTS_SETTING)
        source_kind_values = setting_values(self.module, OVERLAY_SOURCE_KIND_SETTING)
        colors = setting_values(self.module, OVERLAY_COLOR_SETTING)
        row_count = max(
            len(image_names),
            len(object_names),
            len(source_kind_values),
            1 if object_names or image_names else 0,
        )
        return tuple(
            OverlayOutlineRow.from_source_fields(
                self.module,
                OverlayOutlineSourceFields.from_literals(
                    RepeatedOverlaySetting(image_names).at(index),
                    RepeatedOverlaySetting(object_names).at(index),
                    RepeatedOverlaySetting(source_kind_values).at(index),
                ),
                color=RepeatedOverlaySetting(colors, default="Red").at(index),
            )
            for index in range(row_count)
        )
