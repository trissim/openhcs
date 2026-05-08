"""Nominal measurement-feature lookup dialects for runtime artifact queries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator


_EMPTY_FEATURE_ALIASES: Mapping[tuple[str, ...], tuple[str, ...]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementLookupDialect:
    """Dialect used to resolve external measurement names to runtime fields."""

    category_prefixes: tuple[tuple[str, ...], ...] = ()
    feature_part_aliases: Mapping[tuple[str, ...], tuple[str, ...]] = (
        _EMPTY_FEATURE_ALIASES
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "category_prefixes",
            tuple(
                tuple(part for part in prefix if part)
                for prefix in self.category_prefixes
            ),
        )
        object.__setattr__(
            self,
            "feature_part_aliases",
            MappingProxyType(
                {
                    tuple(part for part in parts if part): tuple(
                        part for part in alias if part
                    )
                    for parts, alias in self.feature_part_aliases.items()
                }
            ),
        )


DEFAULT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT = RuntimeMeasurementLookupDialect()
_CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT: ContextVar[RuntimeMeasurementLookupDialect] = (
    ContextVar(
        "current_runtime_measurement_lookup_dialect",
        default=DEFAULT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT,
    )
)


class RuntimeMeasurementLookupDialectReference(ABC):
    """Nominal reference to a runtime measurement lookup dialect."""

    @abstractmethod
    def resolve(self) -> RuntimeMeasurementLookupDialect:
        """Return the concrete lookup dialect for the active runtime scope."""


@dataclass(frozen=True, slots=True)
class CurrentRuntimeMeasurementLookupDialect(RuntimeMeasurementLookupDialectReference):
    """Reference that resolves through the active runtime lookup dialect context."""

    def resolve(self) -> RuntimeMeasurementLookupDialect:
        """Return the current runtime measurement lookup dialect."""
        return _CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT.get()


CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT = CurrentRuntimeMeasurementLookupDialect()
RuntimeMeasurementLookupDialectLike = (
    RuntimeMeasurementLookupDialect | RuntimeMeasurementLookupDialectReference
)


def resolve_runtime_measurement_lookup_dialect(
    dialect: RuntimeMeasurementLookupDialectLike,
) -> RuntimeMeasurementLookupDialect:
    """Resolve a nominal lookup dialect or dialect reference."""
    if isinstance(dialect, RuntimeMeasurementLookupDialect):
        return dialect
    if isinstance(dialect, RuntimeMeasurementLookupDialectReference):
        return dialect.resolve()
    raise TypeError(
        "Expected RuntimeMeasurementLookupDialect or "
        "RuntimeMeasurementLookupDialectReference, "
        f"got {type(dialect).__name__}."
    )


@contextmanager
def runtime_measurement_lookup_dialect(
    dialect: RuntimeMeasurementLookupDialect,
) -> Iterator[None]:
    """Temporarily bind the runtime measurement lookup dialect."""
    if not isinstance(dialect, RuntimeMeasurementLookupDialect):
        raise TypeError(
            "runtime_measurement_lookup_dialect requires "
            f"RuntimeMeasurementLookupDialect, got {type(dialect).__name__}."
        )
    token = _CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT.set(dialect)
    try:
        yield
    finally:
        _CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT.reset(token)


__all__ = (
    "CURRENT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT",
    "CurrentRuntimeMeasurementLookupDialect",
    "DEFAULT_RUNTIME_MEASUREMENT_LOOKUP_DIALECT",
    "RuntimeMeasurementLookupDialect",
    "RuntimeMeasurementLookupDialectLike",
    "RuntimeMeasurementLookupDialectReference",
    "resolve_runtime_measurement_lookup_dialect",
    "runtime_measurement_lookup_dialect",
)
