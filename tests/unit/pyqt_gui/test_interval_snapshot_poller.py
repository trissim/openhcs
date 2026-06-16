import threading
import time

from pyqt_reactive.services.interval_snapshot_poller import (
    CallbackIntervalSnapshotPollerPolicy,
    IntervalSnapshotPoller,
)


def _wait_until(predicate, timeout_seconds: float = 1.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


def test_interval_snapshot_poller_reports_only_changed_snapshots():
    snapshots = [{"value": 1}, {"value": 1}, {"value": 2}]
    changed: list[dict[str, int]] = []

    poller = IntervalSnapshotPoller[dict[str, int]](
        CallbackIntervalSnapshotPollerPolicy(
            fetch_snapshot_fn=lambda: snapshots.pop(0),
            clone_snapshot_fn=lambda snapshot: dict(snapshot),
            poll_interval_seconds_value=0.0,
            on_snapshot_changed_fn=lambda snapshot: changed.append(snapshot),
        )
    )

    poller.tick()
    _wait_until(lambda: not poller.is_poll_inflight())
    poller.tick()
    _wait_until(lambda: not poller.is_poll_inflight())
    poller.tick()
    _wait_until(lambda: not poller.is_poll_inflight())

    assert changed == [{"value": 1}, {"value": 2}]
    assert poller.get_snapshot_copy() == {"value": 2}


def test_interval_snapshot_poller_reset_drops_stale_inflight_result():
    release_fetch = threading.Event()
    changed: list[dict[str, int]] = []
    fetch_values = [{"value": 1}, {"value": 2}]

    def _fetch() -> dict[str, int]:
        release_fetch.wait(timeout=1.0)
        return fetch_values.pop(0)

    poller = IntervalSnapshotPoller[dict[str, int]](
        CallbackIntervalSnapshotPollerPolicy(
            fetch_snapshot_fn=_fetch,
            clone_snapshot_fn=lambda snapshot: dict(snapshot),
            poll_interval_seconds_value=0.0,
            on_snapshot_changed_fn=lambda snapshot: changed.append(snapshot),
        )
    )

    poller.tick()
    _wait_until(lambda: poller.is_poll_inflight())
    poller.reset()
    release_fetch.set()
    _wait_until(lambda: not poller.is_poll_inflight())

    assert poller.get_snapshot_copy() is None
    assert changed == []

    poller.tick()
    _wait_until(lambda: not poller.is_poll_inflight())
    assert changed == [{"value": 2}]
    assert poller.get_snapshot_copy() == {"value": 2}
