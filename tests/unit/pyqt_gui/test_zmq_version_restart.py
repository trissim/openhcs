"""State-preserving ZMQ version replacement tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from PyQt6.QtWidgets import QDialog
from pyqt_reactive.services.async_operation_executor import (
    AsyncOperationExecutorClosedError,
)

from openhcs.core.execution_state import ManagerExecutionState
from openhcs.pyqt_gui.services.desktop_update import DesktopUpdateError
from openhcs.pyqt_gui.services.zmq_version_restart import ZMQVersionRestartWorkflow


class _WindowProbe(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> bool:
        self.close_calls += 1
        return super().close()


class _SessionProbe:
    def __init__(
        self,
        *,
        start_result: bool = True,
        start_error: DesktopUpdateError | None = None,
    ) -> None:
        self.start_result = start_result
        self.start_error = start_error
        self.start_calls = 0
        self.discard_calls = 0

    def start(self) -> bool:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error
        return self.start_result

    def discard(self) -> None:
        self.discard_calls += 1


class _ClientProbe:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    async def restart_endpoint(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error


class _PresenterProbe:
    def __init__(self, *, confirmed: bool = True) -> None:
        self.confirmed = confirmed
        self.confirmations = []
        self.failures = []

    def confirm_restart(self, compatibility) -> bool:
        self.confirmations.append(compatibility)
        return self.confirmed

    def show_failure(self, message: str) -> None:
        self.failures.append(message)


def _workflow(
    *,
    window: _WindowProbe,
    client: _ClientProbe,
    execute_async,
    presenter: _PresenterProbe,
    statuses: list[str],
) -> ZMQVersionRestartWorkflow:
    return ZMQVersionRestartWorkflow(
        main_window=window,
        client_service=client,
        execution_state=lambda: ManagerExecutionState.IDLE,
        execute_async=execute_async,
        publish_status=statuses.append,
        presenter=presenter,
    )


def test_matching_endpoint_does_not_begin_replacement(qapp) -> None:
    scheduled = []
    presenter = _PresenterProbe()
    workflow = _workflow(
        window=_WindowProbe(),
        client=_ClientProbe(),
        execute_async=lambda *args: scheduled.append(args),
        presenter=presenter,
        statuses=[],
    )

    workflow.observe_compatibility(SimpleNamespace(matches=True))

    assert presenter.confirmations == []
    assert scheduled == []


def test_successful_replacement_starts_saved_session_and_closes_ui(
    qapp,
    monkeypatch,
) -> None:
    compatibility = SimpleNamespace(matches=False)
    session = _SessionProbe()
    scheduled = []
    statuses = []
    client = _ClientProbe()
    window = _WindowProbe()
    presenter = _PresenterProbe()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.zmq_version_restart.DesktopSessionRestart.capture",
        lambda _window: session,
    )
    workflow = _workflow(
        window=window,
        client=client,
        execute_async=lambda operation, *args: scheduled.append((operation, args)),
        presenter=presenter,
        statuses=statuses,
    )

    workflow.observe_compatibility(compatibility)
    operation, arguments = scheduled.pop()
    asyncio.run(operation(*arguments))

    assert client.calls == [
        {"expected_compatibility": compatibility, "persistent": True}
    ]
    assert session.start_calls == 1
    assert session.discard_calls == 0
    assert window.close_calls == 1
    assert statuses == [
        "Replacing the mismatched ZMQ execution server…",
        "ZMQ server matched; restarting OpenHCS…",
    ]
    assert presenter.failures == []


def test_endpoint_replacement_failure_discards_saved_session(
    qapp,
    monkeypatch,
) -> None:
    compatibility = SimpleNamespace(matches=False)
    session = _SessionProbe()
    scheduled = []
    statuses = []
    presenter = _PresenterProbe()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.zmq_version_restart.DesktopSessionRestart.capture",
        lambda _window: session,
    )
    workflow = _workflow(
        window=_WindowProbe(),
        client=_ClientProbe(RuntimeError("replacement failed")),
        execute_async=lambda operation, *args: scheduled.append((operation, args)),
        presenter=presenter,
        statuses=statuses,
    )

    workflow.observe_compatibility(compatibility)
    operation, arguments = scheduled.pop()
    asyncio.run(operation(*arguments))

    assert session.start_calls == 0
    assert session.discard_calls == 1
    assert statuses[-1] == "ZMQ server restart failed"
    assert "replacement failed" in presenter.failures[-1]


def test_synchronous_executor_rejection_discards_saved_session(
    qapp,
    monkeypatch,
) -> None:
    session = _SessionProbe()
    statuses = []
    presenter = _PresenterProbe()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.zmq_version_restart.DesktopSessionRestart.capture",
        lambda _window: session,
    )

    def reject_submission(*_args) -> None:
        raise AsyncOperationExecutorClosedError("executor is closed")

    workflow = _workflow(
        window=_WindowProbe(),
        client=_ClientProbe(),
        execute_async=reject_submission,
        presenter=presenter,
        statuses=statuses,
    )

    workflow.observe_compatibility(SimpleNamespace(matches=False))

    assert session.discard_calls == 1
    assert statuses[-1] == "ZMQ server restart failed"
    assert "executor is closed" in presenter.failures[-1]


def test_capture_failure_does_not_schedule_endpoint_replacement(
    qapp,
    monkeypatch,
) -> None:
    scheduled = []
    presenter = _PresenterProbe()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.zmq_version_restart.DesktopSessionRestart.capture",
        lambda _window: (_ for _ in ()).throw(OSError("session could not be saved")),
    )
    workflow = _workflow(
        window=_WindowProbe(),
        client=_ClientProbe(),
        execute_async=lambda *args: scheduled.append(args),
        presenter=presenter,
        statuses=[],
    )

    workflow.observe_compatibility(SimpleNamespace(matches=False))

    assert scheduled == []
    assert "session could not be saved" in presenter.failures[-1]


def test_pending_replacement_owns_subsequent_mismatch_observations(
    qapp,
    monkeypatch,
) -> None:
    session = _SessionProbe()
    scheduled = []
    presenter = _PresenterProbe()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.zmq_version_restart.DesktopSessionRestart.capture",
        lambda _window: session,
    )
    workflow = _workflow(
        window=_WindowProbe(),
        client=_ClientProbe(),
        execute_async=lambda operation, *args: scheduled.append((operation, args)),
        presenter=presenter,
        statuses=[],
    )

    workflow.observe_compatibility(SimpleNamespace(matches=False))
    workflow.observe_compatibility(SimpleNamespace(matches=False))

    assert len(presenter.confirmations) == 1
    assert len(scheduled) == 1


def test_restart_worker_validation_failure_preserves_running_ui(
    qapp,
    monkeypatch,
) -> None:
    compatibility = SimpleNamespace(matches=False)
    session = _SessionProbe(
        start_error=DesktopUpdateError("restart worker is unavailable")
    )
    scheduled = []
    presenter = _PresenterProbe()
    window = _WindowProbe()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.zmq_version_restart.DesktopSessionRestart.capture",
        lambda _window: session,
    )
    workflow = _workflow(
        window=window,
        client=_ClientProbe(),
        execute_async=lambda operation, *args: scheduled.append((operation, args)),
        presenter=presenter,
        statuses=[],
    )

    workflow.observe_compatibility(compatibility)
    operation, arguments = scheduled.pop()
    asyncio.run(operation(*arguments))

    assert session.start_calls == 1
    assert session.discard_calls == 1
    assert window.close_calls == 0
    assert "restart worker is unavailable" in presenter.failures[-1]


def test_failed_restart_worker_launch_preserves_running_ui(
    qapp,
    monkeypatch,
) -> None:
    session = _SessionProbe(start_result=False)
    scheduled = []
    presenter = _PresenterProbe()
    window = _WindowProbe()
    monkeypatch.setattr(
        "openhcs.pyqt_gui.services.zmq_version_restart.DesktopSessionRestart.capture",
        lambda _window: session,
    )
    workflow = _workflow(
        window=window,
        client=_ClientProbe(),
        execute_async=lambda operation, *args: scheduled.append((operation, args)),
        presenter=presenter,
        statuses=[],
    )

    workflow.observe_compatibility(SimpleNamespace(matches=False))
    operation, arguments = scheduled.pop()
    asyncio.run(operation(*arguments))

    assert session.discard_calls == 1
    assert window.close_calls == 0
    assert "current application remains open" in presenter.failures[-1]
