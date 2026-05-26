"""Enum-keyed action dispatch for manager widgets."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


_ActionT = TypeVar("_ActionT", bound=Enum)
_WidgetT = TypeVar("_WidgetT")
AsyncActionRunner = Callable[[Callable[[], object]], None]


class WidgetActionDispatchError(ValueError):
    """Raised when a widget action id is outside the declared route set."""


@dataclass(frozen=True, slots=True)
class WidgetActionRoute(Generic[_ActionT, _WidgetT]):
    """Nominal route from a closed widget action id to executable behavior."""

    action: _ActionT
    resolve_callable: Callable[[_WidgetT], Callable[[], object]]

    def dispatch(
        self,
        *,
        widget: _WidgetT,
        async_runner: AsyncActionRunner,
    ) -> None:
        action_callable = self.resolve_callable(widget)
        if inspect.iscoroutinefunction(action_callable):
            async_runner(action_callable)
            return
        action_callable()


def dispatch_widget_action(
    *,
    widget: _WidgetT,
    action_id: str,
    action_enum: type[_ActionT],
    routes: Mapping[_ActionT, WidgetActionRoute[_ActionT, _WidgetT]],
    async_runner: AsyncActionRunner,
) -> None:
    """Dispatch one string UI action through a nominal enum-keyed route map."""

    try:
        action = action_enum(action_id)
    except ValueError as error:
        raise WidgetActionDispatchError(
            f"Unknown {action_enum.__name__} action id: {action_id!r}"
        ) from error

    route = routes.get(action)
    if route is None:
        raise WidgetActionDispatchError(
            f"No {action_enum.__name__} route registered for {action.value!r}"
        )

    route.dispatch(widget=widget, async_runner=async_runner)


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
