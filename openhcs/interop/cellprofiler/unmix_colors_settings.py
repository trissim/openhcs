"""Typed lowering of CellProfiler UnmixColors repeated output rows."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.public_api import declared_public_names

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
UNMIX_COLORS_STAIN_COUNT_SETTING = "Stain count"


@dataclass(frozen=True, slots=True)
class UnmixColorsSymbolName:
    """Validated CellProfiler symbol name used by UnmixColors rows."""

    raw_value: str

    @property
    def value(self) -> str:
        normalized = self.raw_value.strip()
        if not normalized:
            raise ValueError("CellProfiler symbol names cannot be empty.")
        return normalized


@dataclass(frozen=True, slots=True)
class UnmixColorsAbsorbance:
    """One UnmixColors custom RGB absorbance triplet."""

    red: float
    green: float
    blue: float

    @classmethod
    def from_block(
        cls,
        block: tuple[ModuleSetting, ...],
    ) -> "UnmixColorsAbsorbance":
        return cls(
            red=float(
                block_setting_value(
                    block,
                    UNMIX_COLORS_RED_ABSORBANCE_SETTING,
                    default="0.5",
                )
            ),
            green=float(
                block_setting_value(
                    block,
                    UNMIX_COLORS_GREEN_ABSORBANCE_SETTING,
                    default="0.5",
                )
            ),
            blue=float(
                block_setting_value(
                    block,
                    UNMIX_COLORS_BLUE_ABSORBANCE_SETTING,
                    default="0.5",
                )
            ),
        )

    @property
    def values(self) -> tuple[float, float, float]:
        return (self.red, self.green, self.blue)


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
        return cls(
            image_name=UnmixColorsSymbolName(
                block_setting_value(block, UNMIX_COLORS_OUTPUT_IMAGE_SETTING)
            ).value,
            stain_name=block_setting_value(block, UNMIX_COLORS_STAIN_SETTING),
            custom_absorbance=UnmixColorsAbsorbance.from_block(block).values,
        ).validated(module)

    def validated(self, module: ModuleBlock) -> "UnmixColorsOutputRow":
        if self.stain_name.strip():
            return self
        raise ValueError(
            f"Module {module.name}({module.module_num}) has an UnmixColors "
            f"output row for {self.image_name!r} without a stain."
        )


@dataclass(frozen=True, slots=True)
class UnmixColorsOutputRows:
    """Validated ordered UnmixColors output-row collection."""

    module: ModuleBlock
    rows: tuple[UnmixColorsOutputRow, ...]

    @classmethod
    def from_module(cls, module: ModuleBlock) -> "UnmixColorsOutputRows":
        return cls(
            module=module,
            rows=tuple(
                UnmixColorsOutputRow.from_block(module, block)
                for block in repeating_setting_blocks(
                    module.iter_settings(),
                    start_name=UNMIX_COLORS_OUTPUT_IMAGE_SETTING,
                )
            ),
        ).validated()

    @property
    def expected_count(self) -> int | None:
        value = optional_setting_value(self.module, UNMIX_COLORS_STAIN_COUNT_SETTING)
        if value is None:
            return None
        return int(value)

    def validated(self) -> "UnmixColorsOutputRows":
        if not self.rows:
            raise ValueError(
                f"Module {self.module.name}({self.module.module_num}) declares no "
                "UnmixColors output rows."
            )
        expected_count = self.expected_count
        if expected_count is not None and expected_count != len(self.rows):
            raise ValueError(
                f"Module {self.module.name}({self.module.module_num}) declares "
                f"stain count {expected_count}, but {len(self.rows)} "
                "UnmixColors output rows were parsed."
            )
        return self


def unmix_colors_input_name(module: ModuleBlock) -> str:
    """Return the required input color image symbol name."""
    return required_setting_value(module, UNMIX_COLORS_INPUT_IMAGE_SETTING)


def unmix_colors_output_rows(
    module: ModuleBlock,
) -> tuple[UnmixColorsOutputRow, ...]:
    """Return validated UnmixColors output rows in CellProfiler order."""
    return UnmixColorsOutputRows.from_module(module).rows


def unmix_colors_bound_kwargs(module: ModuleBlock) -> dict[str, object]:
    """Return absorbed-function kwargs for UnmixColors repeated rows."""
    rows = unmix_colors_output_rows(module)
    return {
        "stain_names": tuple(row.stain_name for row in rows),
        "custom_absorbances": tuple(row.custom_absorbance for row in rows),
    }


__all__ = declared_public_names(globals(), constant_prefixes=("UNMIX_COLORS_",))
