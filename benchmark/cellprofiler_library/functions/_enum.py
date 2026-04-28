"""Shared absorbed-function enum coercion helpers."""

from __future__ import annotations

import re
from enum import Enum
from typing import TypeVar


_EnumT = TypeVar("_EnumT", bound=Enum)


def _coerce_function_enum(enum_type: type[_EnumT], value: _EnumT | str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    normalized_value = _normalized_enum_literal(str(value))
    for member in enum_type:
        if normalized_value in _member_literals(member):
            return member
    raise ValueError(
        f"{enum_type.__name__} cannot be coerced from {value!r}."
    )


def _member_literals(member: Enum) -> frozenset[str]:
    literals = [member.name]
    if isinstance(member.value, str):
        literals.append(member.value)
    return frozenset(_normalized_enum_literal(literal) for literal in literals)


def _normalized_enum_literal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
