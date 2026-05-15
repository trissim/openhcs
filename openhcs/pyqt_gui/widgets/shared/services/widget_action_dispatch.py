"""Enum-keyed action dispatch for manager widgets."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


_ActionT = TypeVar("_ActionT", bound=Enum)
_WidgetT = TypeVar("_WidgetT")


@dataclass(frozen=True, slots=True)
class WidgetActionRoute(Generic[_ActionT, _WidgetT]):
    """Nominal route from a closed widget action id to executable behavior."""

    action: _ActionT
    resolve_callable: Callable[[_WidgetT], Callable[[], object]]

    def dispatch(self, widget: _WidgetT) -> None:
        action_callable = self.resolve_callable(widget)
        if inspect.iscoroutinefunction(action_callable):
            widget.run_async_action(action_callable)
            return
        action_callable()


def dispatch_widget_action(
    *,
    widget: _WidgetT,
    action_id: str,
    action_enum: type[_ActionT],
    routes: Mapping[_ActionT, WidgetActionRoute[_ActionT, _WidgetT]],
) -> bool:
    """Dispatch one string UI action through a nominal enum-keyed route map."""

    try:
        action = action_enum(action_id)
    except ValueError:
        return False

    route = routes.get(action)
    if route is None:
        return False

    route.dispatch(widget)
    return True


def is_widget_action_dispatch_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    ) or name == "dispatch_widget_action"


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_widget_action_dispatch_export(name, value)
)
