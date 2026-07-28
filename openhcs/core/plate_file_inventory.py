"""Lightweight plate file inventory declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal, TypeAlias


class PlateFileKind(str, Enum):
    """Kind of file exposed by a plate inventory."""

    IMAGE = "image"
    RESULT = "result"


ALL_PLATE_FILE_KINDS: Final[Literal["all"]] = "all"
PlateFileKindSelection: TypeAlias = PlateFileKind | Literal["all"]


@dataclass(frozen=True, slots=True)
class PlateFileInventoryQuery:
    """Bounded filter over unified plate file records."""

    kinds: tuple[PlateFileKind, ...] = ()
    path_contains: str | None = None
    well: str | None = None
    offset: int = 0
    limit: int = 50

    @staticmethod
    def kinds_for(
        kind: PlateFileKind | None,
    ) -> tuple[PlateFileKind, ...]:
        return () if kind is None else (kind,)

    @classmethod
    def kind_choices(cls) -> tuple[str, ...]:
        return (ALL_PLATE_FILE_KINDS, *(kind.value for kind in PlateFileKind))

    @classmethod
    def kind_from_value(
        cls,
        value: PlateFileKindSelection,
    ) -> PlateFileKind | None:
        if value == ALL_PLATE_FILE_KINDS:
            return None
        try:
            return PlateFileKind(value)
        except ValueError as exc:
            choices = ", ".join(repr(choice) for choice in cls.kind_choices())
            raise ValueError(
                f"Plate file kind must be one of: {choices}; received {value!r}."
            ) from exc

    @staticmethod
    def kind_value(kind: PlateFileKind | None) -> str:
        """Return the public query value for one normalized kind selection."""
        return ALL_PLATE_FILE_KINDS if kind is None else kind.value

    def normalized(self) -> "PlateFileInventoryQuery":
        return PlateFileInventoryQuery(
            kinds=self.kinds,
            path_contains=self.path_contains,
            well=self.well,
            offset=max(0, int(self.offset)),
            limit=max(0, int(self.limit)),
        )
