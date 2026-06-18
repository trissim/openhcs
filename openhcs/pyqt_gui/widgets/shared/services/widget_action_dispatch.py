"""Enum-keyed action dispatch for manager widgets."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


_ActionT = TypeVar("_ActionT", bound=Enum)
_WidgetT = TypeVar("_WidgetT")
WidgetActionSyncCallable = Callable[[], None]
WidgetActionAsyncCallable = Callable[[], Awaitable[None]]
WidgetActionCallable = WidgetActionSyncCallable | WidgetActionAsyncCallable
AsyncActionRunner = Callable[[WidgetActionAsyncCallable], None]


class WidgetActionDispatchError(ValueError):
    """Raised when a widget action id is outside the declared route set."""


@dataclass(frozen=True, slots=True)
class WidgetActionDispatchResult(Generic[_ActionT]):
    """Result of handing one widget action to its route."""

    action: _ActionT
    invocation_mode: str


@dataclass(frozen=True, slots=True)
class WidgetActionRoute(Generic[_ActionT, _WidgetT]):
    """Nominal route from a closed widget action id to executable behavior."""

    action: _ActionT
    resolve_callable: Callable[[_WidgetT], WidgetActionCallable]

    def dispatch(
        self,
        *,
        widget: _WidgetT,
        async_runner: AsyncActionRunner,
    ) -> WidgetActionDispatchResult[_ActionT]:
        action_callable = self.resolve_callable(widget)
        if inspect.iscoroutinefunction(action_callable):
            async_runner(action_callable)
            return WidgetActionDispatchResult(
                action=self.action,
                invocation_mode="async",
            )
        action_callable()
        return WidgetActionDispatchResult(
            action=self.action,
            invocation_mode="sync",
        )


def dispatch_widget_action(
    *,
    widget: _WidgetT,
    action_id: str,
    action_enum: type[_ActionT],
    routes: Mapping[_ActionT, WidgetActionRoute[_ActionT, _WidgetT]],
    async_runner: AsyncActionRunner,
) -> WidgetActionDispatchResult[_ActionT]:
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

    return route.dispatch(widget=widget, async_runner=async_runner)
