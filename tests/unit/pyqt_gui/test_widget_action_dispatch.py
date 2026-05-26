from __future__ import annotations

from enum import Enum

from openhcs.pyqt_gui.widgets.shared.services.widget_action_dispatch import (
    WidgetActionDispatchError,
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


ROUTES = {
    route.action: route
    for route in (
        WidgetActionRoute(DemoAction.SYNC, lambda widget: widget.sync_action),
        WidgetActionRoute(DemoAction.ASYNC, lambda widget: widget.async_action),
    )
}


def test_dispatch_widget_action_invokes_sync_route() -> None:
    harness = ActionDispatchHarness()

    dispatch_widget_action(
        widget=harness,
        action_id="sync",
        action_enum=DemoAction,
        routes=ROUTES,
        async_runner=harness.async_calls.append,
    )

    assert harness.sync_calls == 1


def test_dispatch_widget_action_routes_async_to_widget_runner() -> None:
    harness = ActionDispatchHarness()

    dispatch_widget_action(
        widget=harness,
        action_id="async",
        action_enum=DemoAction,
        routes=ROUTES,
        async_runner=harness.async_calls.append,
    )

    assert harness.async_calls == [harness.async_action]


def test_dispatch_widget_action_rejects_unknown_action() -> None:
    harness = ActionDispatchHarness()

    import pytest

    with pytest.raises(WidgetActionDispatchError, match="Unknown DemoAction"):
        dispatch_widget_action(
            widget=harness,
            action_id="missing",
            action_enum=DemoAction,
            routes=ROUTES,
            async_runner=harness.async_calls.append,
        )


def test_dispatch_widget_action_rejects_missing_route() -> None:
    harness = ActionDispatchHarness()

    import pytest

    with pytest.raises(WidgetActionDispatchError, match="No DemoAction route"):
        dispatch_widget_action(
            widget=harness,
            action_id="sync",
            action_enum=DemoAction,
            routes={},
            async_runner=harness.async_calls.append,
        )
