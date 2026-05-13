from __future__ import annotations

import time

from benchmark.metrics import memory as memory_module
from benchmark.metrics.memory import MemoryMetric


def test_memory_metric_reenforces_limit_while_rss_remains_high(monkeypatch) -> None:
    callbacks: list[float] = []
    metric = MemoryMetric(
        interval_seconds=0.01,
        max_memory_mb=1.0,
        on_limit_exceeded=lambda peak_mb, _children: callbacks.append(peak_mb),
        limit_callback_interval_seconds=0.01,
    )
    monkeypatch.setattr(
        metric,
        "_sample_process_tree_rss",
        lambda: (2 * 1024 * 1024, ()),
    )

    with metric:
        deadline = time.monotonic() + 0.5
        while len(callbacks) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

    assert metric.limit_exceeded
    assert len(callbacks) >= 2


def test_memory_metric_can_interrupt_main_once_on_limit(monkeypatch) -> None:
    interrupts: list[None] = []
    metric = MemoryMetric(
        interval_seconds=0.01,
        max_memory_mb=1.0,
        interrupt_main_on_limit=True,
    )
    monkeypatch.setattr(
        metric,
        "_sample_process_tree_rss",
        lambda: (2 * 1024 * 1024, ()),
    )
    monkeypatch.setattr(
        memory_module._thread,
        "interrupt_main",
        lambda: interrupts.append(None),
    )

    with metric:
        deadline = time.monotonic() + 0.5
        while not interrupts and time.monotonic() < deadline:
            time.sleep(0.01)

    assert metric.limit_exceeded
    assert interrupts == [None]


def test_memory_metric_suppresses_own_interrupt_during_cleanup(monkeypatch) -> None:
    metric = MemoryMetric(max_memory_mb=1.0, interrupt_main_on_limit=True)
    metric._thread = type(
        "InterruptingThread",
        (),
        {"join": lambda self, timeout=None: (_ for _ in ()).throw(KeyboardInterrupt())},
    )()
    metric._limit_exceeded = True

    metric.__exit__(None, None, None)
