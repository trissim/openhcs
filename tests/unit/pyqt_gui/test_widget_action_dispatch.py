from __future__ import annotations

from enum import Enum

from openhcs.pyqt_gui.widgets.shared.services.widget_action_dispatch import (
    WidgetActionRoute,
    dispatch_widget_action,
)


class DemoAction(str, Enum):
    SYNC = "sync"
    ASYNC = "async"


class ActionDispatchHarness:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.async_calls = []

    def sync_action(self) -> None:
        self.sync_calls += 1

    async def async_action(self) -> None:
        pass

    def run_async_action(self, action_callable) -> None:
        self.async_calls.append(action_callable)


ROUTES = {
    route.action: route
    for route in (
        WidgetActionRoute(DemoAction.SYNC, lambda widget: widget.sync_action),
        WidgetActionRoute(DemoAction.ASYNC, lambda widget: widget.async_action),
    )
}


def test_dispatch_widget_action_invokes_sync_route() -> None:
    harness = ActionDispatchHarness()

    handled = dispatch_widget_action(
        widget=harness,
        action_id="sync",
        action_enum=DemoAction,
        routes=ROUTES,
    )

    assert handled is True
    assert harness.sync_calls == 1


def test_dispatch_widget_action_routes_async_to_widget_runner() -> None:
    harness = ActionDispatchHarness()

    handled = dispatch_widget_action(
        widget=harness,
        action_id="async",
        action_enum=DemoAction,
        routes=ROUTES,
    )

    assert handled is True
    assert harness.async_calls == [harness.async_action]


def test_dispatch_widget_action_reports_unknown_action() -> None:
    harness = ActionDispatchHarness()

    assert (
        dispatch_widget_action(
            widget=harness,
            action_id="missing",
            action_enum=DemoAction,
            routes=ROUTES,
        )
        is False
    )
