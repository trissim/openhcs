"""Shared absorbed-function enum coercion helpers."""

from __future__ import annotations

import re
from enum import Enum
from typing import TypeVar


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


def _coerce_function_enum(enum_type: type[_EnumT], value: _EnumT | str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    normalized_value = _normalized_enum_literal(str(value))
    for member in enum_type:
        if normalized_value in _member_literals(enum_type, member):
            return member
    raise ValueError(
        f"{enum_type.__name__} cannot be coerced from {value!r}."
    )


def _member_literals(enum_type: type[Enum], member: Enum) -> frozenset[str]:
    literals = [member.name]
    if isinstance(member.value, str):
        literals.append(member.value)
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
