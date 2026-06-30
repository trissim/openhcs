"""Lightweight plate file inventory declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class PlateFileKind(str, Enum):
    """Kind of file exposed by a plate inventory."""

    IMAGE = "image"
    RESULT = "result"


@dataclass(frozen=True, slots=True)
class PlateFileInventoryQuery:
    """Bounded filter over unified plate file records."""

    ALL_KIND_VALUE: ClassVar[str] = "all"

    kinds: tuple[PlateFileKind, ...] = ()
    path_contains: str | None = None
    well: str | None = None
    offset: int = 0
    limit: int = 50

    @classmethod
    def kinds_for(cls, kind: PlateFileKind | str | None) -> tuple[PlateFileKind, ...]:
        if kind is None:
            return ()
        if isinstance(kind, str) and kind == cls.ALL_KIND_VALUE:
            return ()
        return (PlateFileKind(kind),)

    @classmethod
    def kind_choices(cls) -> tuple[str, ...]:
        return (cls.ALL_KIND_VALUE, *(kind.value for kind in PlateFileKind))

    @classmethod
    def kind_from_value(cls, value: PlateFileKind | str | None) -> PlateFileKind | None:
        if value is None:
            return None
        if isinstance(value, str) and value == cls.ALL_KIND_VALUE:
            return None
        return PlateFileKind(value)

    def normalized(self) -> "PlateFileInventoryQuery":
        return PlateFileInventoryQuery(
            kinds=self.kinds,
            path_contains=self.path_contains,
            well=self.well,
            offset=max(0, int(self.offset)),
            limit=max(0, int(self.limit)),
        )
