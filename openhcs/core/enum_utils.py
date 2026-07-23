"""Shared exact enum value normalization."""

from __future__ import annotations

from enum import Enum
from typing import TypeVar


EnumT = TypeVar("EnumT", bound=Enum)


def coerce_enum(
    enum_type: type[EnumT],
    value: EnumT | object,
    field_name: str,
) -> EnumT:
    """Return one member of the declared enum type or reject the value."""

    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(str(member.value) for member in enum_type)
        raise ValueError(
            f"{field_name} must be one of {choices}; got {value!r}."
        ) from exc
