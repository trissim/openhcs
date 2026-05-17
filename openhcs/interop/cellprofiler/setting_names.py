"""Typed CellProfiler setting-name families and lookup helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .cellprofiler_literals import decode_cellprofiler_setting_literal
from .parser import ModuleBlock, ModuleSetting


@dataclass(frozen=True, slots=True)
class SettingNameFamily:
    """Canonical CellProfiler setting plus accepted schema aliases."""

    canonical: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases)


@dataclass(frozen=True, slots=True)
class SettingNameFamilySpec:
    """Declarative source row for a CellProfiler setting-name family."""

    canonical: str
    aliases: tuple[str, ...] = ()

    def materialize(self) -> SettingNameFamily:
        """Build the runtime setting-name family from this declaration."""
        return SettingNameFamily(self.canonical, aliases=self.aliases)


@dataclass(frozen=True, slots=True)
class OptionalSettingSymbol:
    """Optional CellProfiler artifact symbol selected by one setting family."""

    module: ModuleBlock
    setting_name: str | SettingNameFamily

    @property
    def value(self) -> str | None:
        setting_value = optional_setting_value(self.module, self.setting_name)
        if setting_value is None:
            return None
        return normalized_symbol_name(setting_value)


@dataclass(frozen=True, slots=True)
class RepeatedSettingSequence:
    """Repeated CellProfiler setting values with last-value fallback semantics."""

    values: tuple[str, ...]
    default: str = ""

    def at(self, index: int) -> str:
        if not self.values:
            return self.default
        if index < len(self.values):
            return self.values[index]
        return self.values[-1]


class CellProfilerSettingLiteralNormalizer:
    """Normalize CellProfiler UI labels and blank-literal sentinels."""

    @staticmethod
    def blank_literal(value: str) -> str:
        """Return the decoded lowercase token form used for blank sentinels."""
        return "_".join(
            decode_cellprofiler_setting_literal(value).strip().lower().split()
        )


class BlockSettingLookupPolicy:
    """Shared lookup policy for ordered repeated CellProfiler setting blocks."""

    @staticmethod
    def value(
        block: Sequence[ModuleSetting],
        setting_name: str | SettingNameFamily,
        matcher: Callable[[str, str | SettingNameFamily], bool],
        *,
        default: str = "",
    ) -> str:
        """Return the first value whose setting name satisfies the matcher."""
        for setting in block:
            if matcher(setting.name, setting_name):
                return setting.value
        return default


IMAGE_MEASUREMENT_SETTING = SettingNameFamily(
    "Select images to measure",
    aliases=("Select an image to measure", "Select the image to measure"),
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


def split_symbol_names(value: str) -> tuple[str, ...]:
    """Split a CellProfiler symbol setting while dropping blank sentinels."""
    return tuple(
        normalized
        for part in value.split(",")
        if (normalized := normalized_symbol_name(part)) is not None
    )


def normalized_symbol_name(value: str) -> str | None:
    """Normalize one CellProfiler artifact symbol value."""
    normalized = value.strip()
    if not normalized or is_blank_symbol_name(normalized):
        return None
    return normalized


def is_blank_symbol_name(value: str) -> bool:
    """Return whether a CellProfiler setting value means no artifact symbol."""
    return CellProfilerSettingLiteralNormalizer.blank_literal(value) in {
        "leave_blank",
        "leave_this_black",
        "leave_this_blank",
        "none",
        "do_not_use",
        "no",
        "not_using",
    }


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
    decoded_actual = decode_cellprofiler_setting_literal(actual).strip().rstrip(":").strip()
    return any(
        decoded_actual
        == decode_cellprofiler_setting_literal(name).strip().rstrip(":").strip()
        for name in setting_names(expected)
    )


def setting_name_startswith(actual: str, prefix: str | SettingNameFamily) -> bool:
    """Return whether a parsed CellProfiler setting label starts with a family."""
    decoded_actual = decode_cellprofiler_setting_literal(actual).strip().rstrip(":").strip()
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
    return BlockSettingLookupPolicy.value(
        block,
        name,
        setting_name_matches,
        default=default,
    )


def block_setting_value_by_prefix(
    block: Sequence[ModuleSetting],
    prefix: str | SettingNameFamily,
    *,
    default: str = "",
) -> str:
    """Return a setting value by decoded CellProfiler label prefix."""
    return BlockSettingLookupPolicy.value(
        block,
        prefix,
        setting_name_startswith,
        default=default,
    )


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
