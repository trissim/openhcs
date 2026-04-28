"""Typed CellProfiler setting-name families and lookup helpers."""

from __future__ import annotations

import warnings
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
    setting_records = module.iter_settings()
    for setting in setting_records:
        if setting_name_matches(setting.name, name) and setting.value.strip():
            return setting.value.strip()
    if setting_records:
        return None
    for setting_name, value in module.settings.items():
        if setting_name_matches(setting_name, name) and value.strip():
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
    setting_records = module.iter_settings()
    if not setting_records:
        return tuple(
            value.strip()
            for setting_name, value in module.settings.items()
            if setting_name_matches(setting_name, name) and value.strip()
        )
    return tuple(
        setting.value.strip()
        for setting in setting_records
        if setting_name_matches(setting.name, name) and setting.value.strip()
    )


def setting_names(name: str | SettingNameFamily) -> tuple[str, ...]:
    """Return the concrete setting labels accepted by this lookup."""
    if isinstance(name, SettingNameFamily):
        return name.names
    return (name,)


def setting_name_matches(
    actual: str,
    expected: str | SettingNameFamily,
) -> bool:
    """Return whether a parsed CellProfiler setting label matches a family."""
    decoded_actual = (
        decode_cellprofiler_setting_literal(actual).strip().rstrip(":").strip()
    )
    return any(
        decoded_actual
        == decode_cellprofiler_setting_literal(name).strip().rstrip(":").strip()
        for name in setting_names(expected)
    )


def setting_name_startswith(actual: str, prefix: str | SettingNameFamily) -> bool:
    """Return whether a parsed CellProfiler setting label starts with a family."""
    decoded_actual = (
        decode_cellprofiler_setting_literal(actual).strip().rstrip(":").strip()
    )
    return any(
        decoded_actual.startswith(
            decode_cellprofiler_setting_literal(name).strip().rstrip(":").strip()
        )
        for name in setting_names(prefix)
    )


def block_setting_value(
    block: Sequence[ModuleSetting],
    name: str | SettingNameFamily,
    *,
    default: str = "",
) -> str:
    """Return a setting value from an ordered repeated setting block."""
    for setting in block:
        if setting_name_matches(setting.name, name):
            return setting.value
    return default


def block_setting_value_by_prefix(
    block: Sequence[ModuleSetting],
    prefix: str | SettingNameFamily,
    *,
    default: str = "",
) -> str:
    """Return a setting value by decoded CellProfiler label prefix."""
    for setting in block:
        if setting_name_startswith(setting.name, prefix):
            return setting.value
    return default


def repeating_setting_blocks(
    settings: Sequence[ModuleSetting],
    *,
    start_name: str | SettingNameFamily,
) -> tuple[tuple[ModuleSetting, ...], ...]:
    """Group ordered CellProfiler settings into repeated semantic blocks."""
    blocks: list[list[ModuleSetting]] = []
    current_block: list[ModuleSetting] = []
    started = False
    for setting in settings:
        if setting_name_matches(setting.name, start_name):
            if started and current_block:
                blocks.append(current_block)
                current_block = []
            started = True
        if started:
            current_block.append(setting)
    if current_block:
        blocks.append(current_block)
    return tuple(tuple(block) for block in blocks)


def decode_cellprofiler_setting_literal(value: str) -> str:
    """Decode CellProfiler's escaped setting-name/value literals."""
    if "\\x" not in value and "\\\\\\\\" not in value:
        return value
    decoded = value
    for _ in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            decoded = bytes(decoded, "utf-8").decode("unicode_escape")
    return decoded
