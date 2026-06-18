"""Serialization helpers for OpenHCS agent DTOs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from functools import singledispatch
from pathlib import Path

from openhcs.agent.dto.common import JsonValue


@singledispatch
def to_jsonable(value) -> JsonValue:
    """Convert dataclasses and common OpenHCS values into JSON-safe objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    raise TypeError(f"Value is not JSON-serializable through the OpenHCS agent API: {type(value).__name__}")


@to_jsonable.register(Mapping)
def _jsonable_mapping(value: Mapping) -> JsonValue:
    return {
        str(to_jsonable(key)): to_jsonable(item)
        for key, item in value.items()
    }


@to_jsonable.register(tuple)
@to_jsonable.register(list)
@to_jsonable.register(set)
@to_jsonable.register(frozenset)
def _jsonable_sequence(value) -> JsonValue:
    return [to_jsonable(item) for item in value]


@to_jsonable.register(Enum)
def _jsonable_enum(value: Enum) -> JsonValue:
    return value.value


@to_jsonable.register(Path)
def _jsonable_path(value: Path) -> JsonValue:
    return str(value)


@to_jsonable.register(type(None))
@to_jsonable.register(str)
@to_jsonable.register(int)
@to_jsonable.register(float)
@to_jsonable.register(bool)
def _jsonable_scalar(value) -> JsonValue:
    return value
