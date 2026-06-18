"""Shared selection carriers for UI and agent-facing projections."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class SelectedAllSelectionMode(str, Enum):
    SELECTED = "selected"
    ALL = "all"


@dataclass(frozen=True, kw_only=True, slots=True)
class SelectionModeCarrier:
    selection_mode: str | None = None

    def resolved_selection_mode(
        self,
        default: str | SelectedAllSelectionMode,
    ) -> str:
        if self.selection_mode is None:
            if isinstance(default, SelectedAllSelectionMode):
                return default.value
            return default
        return self.selection_mode


@dataclass(frozen=True, kw_only=True, slots=True)
class SelectedScopeIdsCarrier(SelectionModeCarrier):
    selected_scope_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectedScopeIdsArgument(SelectedScopeIdsCarrier):
    """Nominal adapter for optional external selected-scope collections."""

    @classmethod
    def from_optional_iterable(
        cls,
        value: Iterable[str] | None,
    ) -> "SelectedScopeIdsArgument":
        if value is None:
            return cls(selected_scope_ids=())
        return cls(selected_scope_ids=tuple(value))
