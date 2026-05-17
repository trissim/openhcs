"""Helpers for deriving module public API surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from inspect import isclass, isfunction


def declared_public_names(
    module_globals: Mapping[str, object],
    *,
    constant_prefixes: Iterable[str] = (),
    excluded_names: Iterable[str] = (),
    extra_names: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return public names declared by the module represented by globals()."""
    module_name = module_globals["__name__"]
    prefixes = tuple(constant_prefixes)
    excluded = set(excluded_names)
    declared_names = tuple(
        name
        for name, value in module_globals.items()
        if name not in excluded
        if is_declared_public_name(
            module_name,
            name,
            value,
            constant_prefixes=prefixes,
        )
    )
    return declared_names + tuple(name for name in extra_names if name not in excluded)


def is_declared_public_name(
    module_name: str,
    name: str,
    value: object,
    *,
    constant_prefixes: tuple[str, ...] = (),
) -> bool:
    """Return whether a global is a public module declaration."""
    if name.startswith("_"):
        return False
    if name.isupper():
        return any(name.startswith(prefix) for prefix in constant_prefixes)
    return (isclass(value) or isfunction(value)) and value.__module__ == module_name
