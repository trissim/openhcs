from pyqt_reactive.widgets.shared import KillOperationKind
from zmqruntime import EndpointShutdownMode, EndpointShutdownResult

from openhcs.pyqt_gui.widgets.shared.server_browser import (
    ServerKillService,
)


class _FakeRegistry:
    def __init__(self):
        self.removed: list[int] = []

    def remove_tracker(self, port: int) -> None:
        self.removed.append(port)


def test_server_kill_service_strict_failures_returns_error():
    registry = _FakeRegistry()
    killed_ports: list[int] = []

    def shutdown_endpoint(
        port: int,
        mode: EndpointShutdownMode,
        config,
    ) -> EndpointShutdownResult:
        del mode, config
        return EndpointShutdownResult(
            succeeded=port != 7778,
            endpoint_terminated=port != 7778,
        )

    service = ServerKillService(
        shutdown_endpoint_fn=shutdown_endpoint,
        queue_tracker_registry_factory=lambda: registry,
        config=object(),
    )
    success, message = service.kill_ports(
        ports=[7777, 7778],
        kind=KillOperationKind.GRACEFUL,
        on_endpoint_terminated=lambda port: killed_ports.append(port),
        log_info=lambda *_args, **_kwargs: None,
        log_warning=lambda *_args, **_kwargs: None,
        log_error=lambda *_args, **_kwargs: None,
    )

    assert not success
    assert "7778" in message
    assert killed_ports == [7777]
    assert registry.removed == [7777]


def test_server_kill_service_never_reports_failed_endpoint_as_terminated():
    registry = _FakeRegistry()
    killed_ports: list[int] = []

    def shutdown_endpoint(
        port: int,
        mode: EndpointShutdownMode,
        config,
    ) -> EndpointShutdownResult:
        del port, mode, config
        return EndpointShutdownResult(
            succeeded=False,
            endpoint_terminated=False,
        )

    service = ServerKillService(
        shutdown_endpoint_fn=shutdown_endpoint,
        queue_tracker_registry_factory=lambda: registry,
        config=object(),
    )
    success, message = service.kill_ports(
        ports=[8888, 9999],
        kind=KillOperationKind.FORCE,
        on_endpoint_terminated=lambda port: killed_ports.append(port),
        log_info=lambda *_args, **_kwargs: None,
        log_warning=lambda *_args, **_kwargs: None,
        log_error=lambda *_args, **_kwargs: None,
    )

    assert not success
    assert "8888" in message
    assert "9999" in message
    assert killed_ports == []
    assert registry.removed == []


def test_server_kill_service_separates_operation_success_from_endpoint_termination():
    registry = _FakeRegistry()
    operations: list[int] = []
    terminated: list[int] = []

    service = ServerKillService(
        shutdown_endpoint_fn=lambda _port, _mode, _config: EndpointShutdownResult(
            succeeded=True,
            endpoint_terminated=False,
        ),
        queue_tracker_registry_factory=lambda: registry,
        config=object(),
    )

    success, _message = service.kill_ports(
        ports=[7777],
        kind=KillOperationKind.GRACEFUL,
        on_operation_succeeded=operations.append,
        on_endpoint_terminated=terminated.append,
        log_info=lambda *_args, **_kwargs: None,
        log_warning=lambda *_args, **_kwargs: None,
        log_error=lambda *_args, **_kwargs: None,
    )

    assert success
    assert operations == [7777]
    assert terminated == []
    assert registry.removed == [7777]
