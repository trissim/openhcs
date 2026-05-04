"""Convert CellProfiler .cppipe settings to absorbed-function kwargs."""

import logging
import re
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .parser import ModuleBlock
from .setting_names import SettingNameFamily, optional_setting_value, setting_names

logger = logging.getLogger(__name__)

SettingParser = Callable[[str], object]


def parse_cellprofiler_bool(value: str) -> bool:
    """Parse a CellProfiler boolean literal."""
    normalized = value.strip().lower()
    if normalized in SettingsBinder.BOOL_TRUE:
        return True
    if normalized in SettingsBinder.BOOL_FALSE:
        return False
    raise ValueError(f"CellProfiler boolean setting must be Yes/No, got {value!r}.")


def parse_cellprofiler_float(value: str) -> float:
    """Parse a numeric CellProfiler setting as float."""
    return float(value)


def parse_cellprofiler_int(value: str) -> int:
    """Parse a numeric CellProfiler setting as int, accepting decimal spelling."""
    return int(float(value))


def normalize_cellprofiler_setting_name(name: str) -> str:
    """Normalize a CellProfiler setting label into a snake_case key."""
    without_parentheses = re.sub(r"\([^)]*\)", "", name)
    without_questions = without_parentheses.replace("?", "")
    words = re.sub(r"[^\w\s]", " ", without_questions).lower().split()
    return "_".join(words)


@dataclass(frozen=True, slots=True)
class BoundParameter:
    """A parameter with its bound value."""

    name: str
    value: Any
    original_key: str
    original_value: str


@dataclass(frozen=True, slots=True)
class SettingToKeywordBinding:
    """Declarative mapping from one parsed setting to one function kwarg."""

    setting_name: str | SettingNameFamily
    parameter_name: str
    parse: SettingParser | None = None

    def bind(
        self,
        module: ModuleBlock,
        kwargs: dict[str, Any],
        binder: "SettingsBinder",
    ) -> None:
        value = optional_setting_value(module, self.setting_name)
        if value is None:
            return
        setting_name = setting_names(self.setting_name)[0]
        kwargs[self.parameter_name] = (
            binder.parse_value(setting_name, value)
            if self.parse is None
            else self.parse(value)
        )


class SettingsBinder:
    """Bind parsed .cppipe setting strings to typed Python kwargs."""

    BOOL_TRUE = {"yes", "true", "1", "on"}
    BOOL_FALSE = {"no", "false", "0", "off"}
    GENERIC_BOOL_TRUE = {"yes", "true", "on"}
    GENERIC_BOOL_FALSE = {"no", "false", "off"}

    SKIP_SETTINGS = {
        "show_window",
        "notes",
        "batch_state",
        "wants_pause",
        "module_num",
        "svn_version",
        "variable_revision_number",
    }
    
    def __init__(
        self,
        enum_mappings: Mapping[str, type[Enum]] | None = None,
    ) -> None:
        self.enum_mappings = dict(enum_mappings or {})
    
    def bind(self, settings: Mapping[str, str]) -> dict[str, Any]:
        """Bind a settings mapping into normalized kwargs."""
        kwargs: dict[str, Any] = {}
        for key, value in settings.items():
            normalized_key = normalize_cellprofiler_setting_name(key)
            if normalized_key in self.SKIP_SETTINGS:
                continue
            kwargs[normalized_key] = self.parse_value(key, value)
        return kwargs

    def bind_declared(
        self,
        module: ModuleBlock,
        bindings: tuple[SettingToKeywordBinding, ...],
    ) -> dict[str, Any]:
        """Bind an explicit setting-to-kwarg declaration for one module."""
        kwargs: dict[str, Any] = {}
        for binding in bindings:
            binding.bind(module, kwargs, self)
        return kwargs
    
    def bind_with_details(self, settings: Mapping[str, str]) -> list[BoundParameter]:
        """Bind settings and preserve original CellProfiler key/value provenance."""
        result: list[BoundParameter] = []
        for key, value in settings.items():
            normalized_key = normalize_cellprofiler_setting_name(key)
            if normalized_key in self.SKIP_SETTINGS:
                continue
            result.append(
                BoundParameter(
                    name=normalized_key,
                    value=self.parse_value(key, value),
                    original_key=key,
                    original_value=value,
                )
            )
        return result
    
    def parse_value(self, key: str, value: str) -> Any:
        """Parse one CellProfiler setting value into a Python value."""
        value = value.strip()

        if value.lower() in self.GENERIC_BOOL_TRUE:
            return True
        if value.lower() in self.GENERIC_BOOL_FALSE:
            return False

        normalized_key = normalize_cellprofiler_setting_name(key)
        if normalized_key in self.enum_mappings:
            enum_type = self.enum_mappings[normalized_key]
            try:
                return enum_type[value.upper().replace(" ", "_")]
            except KeyError:
                logger.warning(f"Unknown enum value '{value}' for {normalized_key}")
                return value

        if "," in value:
            return _parse_cellprofiler_csv_value(value)

        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value


def _parse_cellprofiler_csv_value(value: str) -> tuple[int | float, ...] | list[str]:
    parts = [part.strip() for part in value.split(",")]
    try:
        return tuple(
            float(part) if "." in part else int(part)
            for part in parts
        )
    except ValueError:
        return parts
