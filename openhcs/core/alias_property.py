"""Descriptor for read-only aliases of existing attributes."""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import attrgetter
from typing import Callable, Generic, TypeVar, cast, overload

ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class AliasProperty(Generic[ValueT]):
    """Property-like descriptor that projects another attribute."""

    source_name: str
    _project: Callable[[object], ValueT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_project",
            cast(Callable[[object], ValueT], attrgetter(self.source_name)),
        )

    @overload
    def __get__(
        self, instance: None, owner: type[object] | None = None
    ) -> "AliasProperty[ValueT]": ...

    @overload
    def __get__(
        self, instance: object, owner: type[object] | None = None
    ) -> ValueT: ...

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> ValueT | "AliasProperty[ValueT]":
        del owner
        if instance is None:
            return self
        return self._project(instance)
