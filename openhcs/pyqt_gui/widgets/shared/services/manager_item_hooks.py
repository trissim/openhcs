"""Typed manager-item hook declarations for the legacy PyQt manager base."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ManagerItemHooks:
    """Typed source for the hook mapping consumed by AbstractManagerWidget."""

    id_accessor: str | tuple[str, str]
    backing_attr: str
    selection_attr: str
    selection_signal: str
    selection_emit_id: bool
    selection_clear_value: Any
    items_changed_signal: str | None
    preserve_selection_pred: Callable[[Any], bool]
    list_item_data: str
    scope_item_type: Any
    scope_id_builder: Callable[[Any, int, Any], str] | None = None
    scope_id_attr: str | None = None

    def to_legacy_mapping(self) -> dict[str, Any]:
        """Project typed hooks to the mapping shape required by the base widget."""

        hooks = {
            "id_accessor": self.id_accessor,
            "backing_attr": self.backing_attr,
            "selection_attr": self.selection_attr,
            "selection_signal": self.selection_signal,
            "selection_emit_id": self.selection_emit_id,
            "selection_clear_value": self.selection_clear_value,
            "items_changed_signal": self.items_changed_signal,
            "preserve_selection_pred": self.preserve_selection_pred,
            "list_item_data": self.list_item_data,
            "scope_item_type": self.scope_item_type,
        }
        if self.scope_id_builder is not None:
            hooks["scope_id_builder"] = self.scope_id_builder
        if self.scope_id_attr is not None:
            hooks["scope_id_attr"] = self.scope_id_attr
        return hooks


def is_manager_item_hooks_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_manager_item_hooks_export(name, value)
)
