"""Convert CellProfiler .cppipe settings to absorbed-function kwargs."""

import logging
import re
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from .parser import ModuleBlock
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    setting_names,
)

logger = logging.getLogger(__name__)

_EnumT = TypeVar("_EnumT", bound=Enum)
_NEGATED_ENUM_LITERALS = frozenset(("none", "no", "false", "disabled", "disable"))
_ENUM_DOMAIN_SUFFIXES = (
    "method",
    "choice",
    "option",
    "mode",
    "type",
    "style",
)

CellProfilerSettingValue = (
    bool
    | int
    | float
    | str
    | tuple[int | float, ...]
    | list[str]
    | Enum
)
SettingParser = Callable[[str], CellProfilerSettingValue]


def coerce_cellprofiler_enum(
    enum_type: type[_EnumT],
    value: _EnumT | str,
) -> _EnumT:
    """Coerce a CellProfiler literal into a nominal enum member."""
    if isinstance(value, enum_type):
        return value
    normalized_value = _normalized_enum_literal(str(value))
    for member in enum_type:
        if normalized_value in _member_literals(enum_type, member):
            return member
    prefix_matches = [
        member
        for member in enum_type
        if any(
            normalized_value.startswith(candidate)
            or candidate.startswith(normalized_value)
            for candidate in _member_literals(enum_type, member)
        )
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    raise ValueError(
        f"{enum_type.__name__} cannot be coerced from {value!r}."
    )


def cellprofiler_enum_setting_parser(
    enum_type: type[_EnumT],
) -> Callable[[str], _EnumT]:
    """Return a typed parser for a CellProfiler setting enum."""

    def parse(value: str) -> _EnumT:
        return coerce_cellprofiler_enum(enum_type, value)

    return parse


def cellprofiler_enum_value_setting_parser(
    enum_type: type[_EnumT],
) -> Callable[[str], str]:
    """Return a typed parser that emits an enum member's serialized value."""

    def parse(value: str) -> str:
        member = coerce_cellprofiler_enum(enum_type, value)
        if not isinstance(member.value, str):
            raise TypeError(
                f"{enum_type.__name__}.{member.name} must have a string value."
            )
        return member.value

    return parse


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


def _member_literals(enum_type: type[Enum], member: Enum) -> frozenset[str]:
    literals = [member.name]
    if isinstance(member.value, str):
        literals.append(member.value)
    literals.extend(
        literal
        for literal in getattr(member, "cellprofiler_literals", ())
        if isinstance(literal, str)
    )
    normalized_literals = {
        _normalized_enum_literal(literal)
        for literal in literals
    }
    if normalized_literals & _NEGATED_ENUM_LITERALS:
        domain = _enum_domain_literal(enum_type)
        normalized_literals.add(f"no_{domain}")
    return frozenset(normalized_literals)


def _enum_domain_literal(enum_type: type[Enum]) -> str:
    literal = _normalized_enum_literal(enum_type.__name__)
    for suffix in _ENUM_DOMAIN_SUFFIXES:
        suffix_literal = f"_{suffix}"
        if literal.endswith(suffix_literal):
            return literal.removesuffix(suffix_literal)
    return literal


def _normalized_enum_literal(value: str) -> str:
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value.strip())
    return re.sub(r"[^a-z0-9]+", "_", words.lower()).strip("_")


@dataclass(frozen=True, slots=True)
class BoundParameter:
    """A parameter with its bound value."""

    name: str
    value: CellProfilerSettingValue
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
        kwargs: dict[str, CellProfilerSettingValue],
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
    
    def bind(self, settings: Mapping[str, str]) -> dict[str, CellProfilerSettingValue]:
        """Bind a settings mapping into normalized kwargs."""
        kwargs: dict[str, CellProfilerSettingValue] = {}
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
    ) -> dict[str, CellProfilerSettingValue]:
        """Bind an explicit setting-to-kwarg declaration for one module."""
        kwargs: dict[str, CellProfilerSettingValue] = {}
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
    
    def parse_value(self, key: str, value: str) -> CellProfilerSettingValue:
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
