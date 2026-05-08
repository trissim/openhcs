"""Compatibility aliases for absorbed-function enum coercion."""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum

_EnumT = TypeVar("_EnumT", bound=Enum)


def _coerce_function_enum(enum_type: type[_EnumT], value: _EnumT | str) -> _EnumT:
    return coerce_cellprofiler_enum(enum_type, value)
