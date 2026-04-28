"""Typed lowering of CellProfiler UnmixColors repeated output rows."""

from __future__ import annotations

from dataclasses import dataclass

from .parser import ModuleBlock, ModuleSetting
from .setting_names import (
    SettingNameFamily,
    block_setting_value,
    optional_setting_value,
    repeating_setting_blocks,
    required_setting_value,
)


UNMIX_COLORS_INPUT_IMAGE_SETTING = SettingNameFamily(
    "Select the input color image",
    aliases=("Color image",),
)
UNMIX_COLORS_OUTPUT_IMAGE_SETTING = SettingNameFamily(
    "Name the output image",
    aliases=("Image name",),
)
UNMIX_COLORS_STAIN_SETTING = "Stain"
UNMIX_COLORS_RED_ABSORBANCE_SETTING = "Red absorbance"
UNMIX_COLORS_GREEN_ABSORBANCE_SETTING = "Green absorbance"
UNMIX_COLORS_BLUE_ABSORBANCE_SETTING = "Blue absorbance"


@dataclass(frozen=True, slots=True)
class UnmixColorsOutputRow:
    """One UnmixColors output row lowered from ordered CellProfiler settings."""

    image_name: str
    stain_name: str
    custom_absorbance: tuple[float, float, float]

    @classmethod
    def from_block(
        cls,
        module: ModuleBlock,
        block: tuple[ModuleSetting, ...],
    ) -> "UnmixColorsOutputRow":
        row = cls(
            image_name=_required_symbol_name(
                block_setting_value(block, UNMIX_COLORS_OUTPUT_IMAGE_SETTING)
            ),
            stain_name=block_setting_value(block, UNMIX_COLORS_STAIN_SETTING),
            custom_absorbance=(
                _float_block_value(block, UNMIX_COLORS_RED_ABSORBANCE_SETTING),
                _float_block_value(block, UNMIX_COLORS_GREEN_ABSORBANCE_SETTING),
                _float_block_value(block, UNMIX_COLORS_BLUE_ABSORBANCE_SETTING),
            ),
        )
        row._validate(module)
        return row

    def _validate(self, module: ModuleBlock) -> None:
        if not self.stain_name.strip():
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an UnmixColors "
                f"output row for {self.image_name!r} without a stain."
            )


def unmix_colors_input_name(module: ModuleBlock) -> str:
    """Return the required input color image symbol name."""
    return required_setting_value(module, UNMIX_COLORS_INPUT_IMAGE_SETTING)


def unmix_colors_output_rows(
    module: ModuleBlock,
) -> tuple[UnmixColorsOutputRow, ...]:
    """Return validated UnmixColors output rows in CellProfiler order."""
    rows = tuple(
        UnmixColorsOutputRow.from_block(module, block)
        for block in repeating_setting_blocks(
            module.iter_settings(),
            start_name=UNMIX_COLORS_OUTPUT_IMAGE_SETTING,
        )
    )
    if not rows:
        raise ValueError(
            f"Module {module.name}({module.module_num}) declares no "
            "UnmixColors output rows."
        )
    expected_count = _optional_unmix_colors_row_count(module)
    if expected_count is not None and expected_count != len(rows):
        raise ValueError(
            f"Module {module.name}({module.module_num}) declares stain count "
            f"{expected_count}, but {len(rows)} UnmixColors output rows were "
            "parsed."
        )
    return rows


def unmix_colors_bound_kwargs(module: ModuleBlock) -> dict[str, object]:
    """Return absorbed-function kwargs for UnmixColors repeated rows."""
    rows = unmix_colors_output_rows(module)
    return {
        "stain_names": tuple(row.stain_name for row in rows),
        "custom_absorbances": tuple(row.custom_absorbance for row in rows),
    }


def _optional_unmix_colors_row_count(module: ModuleBlock) -> int | None:
    value = optional_setting_value(module, "Stain count")
    if value is None:
        return None
    return int(value)


def _float_block_value(
    block: tuple[ModuleSetting, ...],
    setting_name: str,
) -> float:
    return float(block_setting_value(block, setting_name, default="0.5"))


def _required_symbol_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("CellProfiler symbol names cannot be empty.")
    return normalized
