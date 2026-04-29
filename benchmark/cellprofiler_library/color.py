"""Shared CellProfiler color literal semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta


class CellProfilerColorFormat(ABC, metaclass=AutoRegisterMeta):
    """Nominal parser family for CellProfiler RGB color literals."""

    __registry_key__ = "format_key"
    __skip_if_no_key__ = True
    format_key: ClassVar[str | None] = None

    @classmethod
    def for_value(cls, value: str | Sequence[float]) -> "CellProfilerColorFormat":
        for format_type in cls.__registry__.values():
            parser = format_type()
            if parser.matches(value):
                return parser
        raise ValueError(f"Unsupported CellProfiler color literal: {value!r}")

    @abstractmethod
    def matches(self, value: str | Sequence[float]) -> bool:
        """Return whether this parser owns the color literal."""

    @abstractmethod
    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        """Return RGB channel values, possibly in 0-255 space."""


class NamedCellProfilerColorFormat(CellProfilerColorFormat):
    """Named CellProfiler colors."""

    format_key = "named"

    def matches(self, value: str | Sequence[float]) -> bool:
        return isinstance(value, str) and value.strip().lower() in _COLOR_BY_NAME

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return _COLOR_BY_NAME[str(value).strip().lower()]


class HexCellProfilerColorFormat(CellProfilerColorFormat):
    """Hex CellProfiler colors such as #0800F7."""

    format_key = "hex"

    def matches(self, value: str | Sequence[float]) -> bool:
        if not isinstance(value, str):
            return False
        literal = value.strip()
        return literal.startswith("#") and len(literal) in {4, 7}

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        literal = str(value).strip().lstrip("#")
        if len(literal) == 3:
            literal = "".join(channel * 2 for channel in literal)
        return (
            int(literal[0:2], 16),
            int(literal[2:4], 16),
            int(literal[4:6], 16),
        )


class DelimitedCellProfilerColorFormat(CellProfilerColorFormat):
    """Comma-delimited RGB triples."""

    format_key = "delimited"

    def matches(self, value: str | Sequence[float]) -> bool:
        return isinstance(value, str) and "," in value

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return tuple(float(part.strip()) for part in str(value).split(","))  # type: ignore[return-value]


class SequenceCellProfilerColorFormat(CellProfilerColorFormat):
    """Already-structured RGB channel sequences."""

    format_key = "sequence"

    def matches(self, value: str | Sequence[float]) -> bool:
        return not isinstance(value, str) and isinstance(value, Sequence)

    def color_parts(self, value: str | Sequence[float]) -> tuple[float, float, float]:
        return tuple(float(part) for part in value)  # type: ignore[arg-type, return-value]


def coerce_rgb_color(value: str | Sequence[float]) -> tuple[float, float, float]:
    """Parse a CellProfiler color literal into an RGB tuple in 0-1 space."""
    parts = CellProfilerColorFormat.for_value(value).color_parts(value)
    if len(parts) != 3:
        raise ValueError(f"CellProfiler color must have three channels, got {parts!r}.")
    scale = 255.0 if max(parts) > 1.0 else 1.0
    return parts[0] / scale, parts[1] / scale, parts[2] / scale


_COLOR_BY_NAME: dict[str, tuple[float, float, float]] = {
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}
