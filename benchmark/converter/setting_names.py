"""Typed CellProfiler setting-name families and lookup helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .parser import ModuleBlock, ModuleSetting


@dataclass(frozen=True, slots=True)
class SettingNameFamily:
    """Canonical CellProfiler setting plus accepted schema aliases."""

    canonical: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases)


IMAGE_MEASUREMENT_SETTING = SettingNameFamily(
    "Select images to measure",
    aliases=("Select an image to measure",),
)
OBJECT_MEASUREMENT_SETTING = SettingNameFamily(
    "Select object sets to measure",
    aliases=("Select objects to measure", "Select an object to measure"),
)


def optional_setting_value(
    module: ModuleBlock,
    name: str | SettingNameFamily,
) -> str | None:
    """Return the first non-empty module setting matching a name family."""
    for setting_name in setting_names(name):
        value = module.settings.get(setting_name)
        if value is not None and value.strip():
            return value.strip()
    return None


def required_setting_value(
    module: ModuleBlock,
    name: str | SettingNameFamily,
) -> str:
    """Return a required setting value or fail with schema context."""
    value = optional_setting_value(module, name)
    if value is None:
        raise ValueError(
            f"Module {module.name}({module.module_num}) missing setting "
            f"{setting_names(name)}."
        )
    return value


def setting_values(
    module: ModuleBlock,
    name: str | SettingNameFamily,
) -> tuple[str, ...]:
    """Return all non-empty ordered values matching a setting name family."""
    values: list[str] = []
    for setting_name in setting_names(name):
        record_values = tuple(
            value.strip()
            for value in module.get_setting_values(setting_name)
            if value.strip()
        )
        if record_values:
            values.extend(record_values)
            continue
        value = module.settings.get(setting_name)
        if value is not None and value.strip():
            values.append(value.strip())
    return tuple(values)


def setting_names(name: str | SettingNameFamily) -> tuple[str, ...]:
    """Return the concrete setting labels accepted by this lookup."""
    if isinstance(name, SettingNameFamily):
        return name.names
    return (name,)


def repeating_setting_blocks(
    settings: Sequence[ModuleSetting],
    *,
    start_name: str,
) -> tuple[tuple[ModuleSetting, ...], ...]:
    """Group ordered CellProfiler settings into repeated semantic blocks."""
    blocks: list[list[ModuleSetting]] = []
    current_block: list[ModuleSetting] = []
    started = False
    for setting in settings:
        if setting.name == start_name:
            if started and current_block:
                blocks.append(current_block)
                current_block = []
            started = True
        if started:
            current_block.append(setting)
    if current_block:
        blocks.append(current_block)
    return tuple(tuple(block) for block in blocks)
