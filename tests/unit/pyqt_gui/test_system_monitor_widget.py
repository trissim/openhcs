from __future__ import annotations

from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PyQt6.QtWidgets import QSizePolicy

import pyqt_reactive.widgets.system_monitor as system_monitor
from pyqt_reactive.services.system_metrics_sampler import SystemMetricsSamplerConfig
from pyqt_reactive.services.system_monitor_config import (
    PerformanceMonitorColors,
    PerformanceMonitorConfig,
)
from pyqt_reactive.services.system_metrics_sampler import SystemMetrics
from pyqt_reactive.widgets.system_monitor import SystemMonitorWidget


class FakeCurve:
    def __init__(self) -> None:
        self.calls = []

    def setData(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class FakePlot:
    def __init__(self, width: int = 300, fail_opengl: bool = False) -> None:
        self.ranges = []
        self.titles = []
        self.opengl_calls = []
        self._width = width
        self._fail_opengl = fail_opengl

    def setXRange(self, left: float, right: float, padding=0) -> None:
        self.ranges.append((left, right, padding))

    def setTitle(self, title: str) -> None:
        self.titles.append(title)

    def width(self) -> int:
        return self._width

    def useOpenGL(self, enabled: bool) -> None:
        if enabled and self._fail_opengl:
            raise RuntimeError("OpenGL unavailable")
        self.opengl_calls.append(enabled)


def make_widget(**attrs):
    methods = {
        name: method
        for name, method in vars(SystemMonitorWidget).items()
        if callable(method) and not name.startswith("__")
    }
    widget_type = type("FakeSystemMonitorWidget", (SimpleNamespace,), methods)
    widget = widget_type()
    for name, value in attrs.items():
        setattr(widget, name, value)
    return widget


class FakePersistentMonitor:
    created = []

    def __init__(self, update_interval, history_length, *, sampler_config=None) -> None:
        self.update_interval = update_interval
        self.history_length = history_length
        self.sampler_config = sampler_config
        self.started = False
        self.stopped = False
        self.connected = False
        self.__class__.created.append(self)

    def start_monitoring(self) -> None:
        self.started = True

    def stop_monitoring(self) -> None:
        self.stopped = True

    def connect_signals(self, metrics_callback, error_callback) -> None:
        self.metrics_callback = metrics_callback
        self.error_callback = error_callback
        self.connected = True


def monitor_config(
    *,
    update_fps: float = 5.0,
    history_duration_seconds: float = 60.0,
    enable_gpu_monitoring: bool = True,
    gpu_temperature_monitoring: bool = True,
    cpu_frequency_monitoring: bool = True,
    gpu_refresh_seconds: float = 1.0,
    cpu_frequency_refresh_seconds: float = 5.0,
    antialiasing: bool = True,
    use_opengl: bool = True,
):
    return PerformanceMonitorConfig(
        update_fps=update_fps,
        history_duration_seconds=history_duration_seconds,
        antialiasing=antialiasing,
        use_opengl=use_opengl,
        show_grid=True,
        line_width=2.0,
        sampler_config=SystemMetricsSamplerConfig(
            enable_gpu_monitoring=enable_gpu_monitoring,
            gpu_temperature_monitoring=gpu_temperature_monitoring,
            cpu_frequency_monitoring=cpu_frequency_monitoring,
            gpu_refresh_seconds=gpu_refresh_seconds,
            cpu_frequency_refresh_seconds=cpu_frequency_refresh_seconds,
        ),
        colors=PerformanceMonitorColors(),
    )


def test_system_monitor_uses_compact_shared_manager_header(qapp, monkeypatch) -> None:
    monkeypatch.setattr(system_monitor, "PersistentSystemMonitor", FakePersistentMonitor)
    monkeypatch.setattr(SystemMonitorWidget, "_load_pyqtgraph_async", lambda self: None)

    widget = SystemMonitorWidget()

    assert widget.manager_header.title_layout is widget.title_layout
    assert widget.manager_header.status_label is widget.status_label
    assert (
        widget.manager_header.header.sizePolicy().verticalPolicy()
        is QSizePolicy.Policy.Fixed
    )
    assert widget.minimumHeight() == 80
    widget.cleanup()


def test_system_monitor_enables_opengl_plot_acceleration(monkeypatch) -> None:
    config_calls = []
    monkeypatch.setattr(
        system_monitor,
        "pg",
        SimpleNamespace(setConfigOption=lambda name, value: config_calls.append((name, value))),
    )
    fake_widget = make_widget(
        monitor_config=SimpleNamespace(use_opengl=True, antialiasing=True),
        cpu_gpu_plot=FakePlot(),
        ram_vram_plot=FakePlot(),
    )

    assert SystemMonitorWidget._configure_plot_acceleration(fake_widget) is True
    assert fake_widget.cpu_gpu_plot.opengl_calls == [True]
    assert fake_widget.ram_vram_plot.opengl_calls == [True]
    assert ("enableExperimental", True) in config_calls

    fake_widget._plot_opengl_enabled = True
    assert SystemMonitorWidget._effective_plot_antialiasing(fake_widget) is True


def test_system_monitor_disables_raster_antialiasing_when_opengl_fails(monkeypatch) -> None:
    config_calls = []
    monkeypatch.setattr(
        system_monitor,
        "pg",
        SimpleNamespace(setConfigOption=lambda name, value: config_calls.append((name, value))),
    )
    fake_widget = make_widget(
        monitor_config=SimpleNamespace(use_opengl=True, antialiasing=True),
        cpu_gpu_plot=FakePlot(fail_opengl=True),
        ram_vram_plot=FakePlot(),
    )

    assert SystemMonitorWidget._configure_plot_acceleration(fake_widget) is False
    assert ("enableExperimental", False) in config_calls

    fake_widget._plot_opengl_enabled = False
    assert SystemMonitorWidget._effective_plot_antialiasing(fake_widget) is False


def test_metrics_update_queues_full_plot_data_on_visual_frame(monkeypatch) -> None:
    monkeypatch.setattr(system_monitor, "PYQTGRAPH_AVAILABLE", True)
    metrics = SystemMetrics(cpu_percent=12.5, ram_percent=34.5)
    fake_widget = SimpleNamespace(
        update_system_info=Mock(),
        _queue_pyqtgraph_plot_update=Mock(),
        update_pyqtgraph_plots=Mock(),
        update_fallback_display=Mock(),
    )

    SystemMonitorWidget.update_display(fake_widget, metrics)

    fake_widget.update_system_info.assert_called_once_with(metrics)
    fake_widget._queue_pyqtgraph_plot_update.assert_called_once_with(metrics)
    fake_widget.update_pyqtgraph_plots.assert_not_called()
    fake_widget.update_fallback_display.assert_not_called()


def test_pyqtgraph_plot_update_uses_visual_frame_coordinator(monkeypatch) -> None:
    queued = []

    def queue_callback(owner, callback) -> None:
        queued.append((owner, callback))

    monkeypatch.setattr(system_monitor, "queue_visual_frame_callback", queue_callback)
    metrics = SystemMetrics(cpu_percent=12.5, ram_percent=34.5)
    fake_widget = make_widget(update_pyqtgraph_plots=Mock())

    SystemMonitorWidget._queue_pyqtgraph_plot_update(fake_widget, metrics)

    assert len(queued) == 1
    owner, callback = queued[0]
    assert owner is fake_widget

    callback()

    fake_widget.update_pyqtgraph_plots.assert_called_once_with()


def test_pyqtgraph_update_uses_persistent_history_arrays() -> None:
    fake_widget = make_widget(
        monitor=SimpleNamespace(
            cpu_history=deque([10.0, 20.0], maxlen=2),
            ram_history=deque([30.0, 40.0], maxlen=2),
            gpu_history=deque([0.0, 0.0], maxlen=2),
            vram_history=deque([0.0, 0.0], maxlen=2),
            time_stamps=deque([100.0, 101.0], maxlen=2),
        ),
        monitor_config=SimpleNamespace(update_interval_seconds=0.2, history_duration_seconds=60.0),
        _history_length=0,
        _history_x=None,
        _history_cpu=None,
        _history_ram=None,
        _history_gpu=None,
        _history_vram=None,
        _history_update_interval=None,
        _last_plot_history=None,
        _plot_point_budget=2000,
        _gpu_series_visible=False,
        _vram_series_visible=False,
        cpu_gpu_plot=FakePlot(),
        ram_vram_plot=FakePlot(),
        cpu_curve=FakeCurve(),
        gpu_curve=FakeCurve(),
        ram_curve=FakeCurve(),
        vram_curve=FakeCurve(),
    )

    SystemMonitorWidget.update_pyqtgraph_plots(fake_widget)

    assert np.array_equal(fake_widget._history_x, np.array([-0.2, 0.0]))
    assert np.array_equal(fake_widget._history_cpu, np.array([10.0, 20.0], dtype=np.float32))
    assert np.array_equal(fake_widget._history_ram, np.array([30.0, 40.0], dtype=np.float32))
    assert len(fake_widget.cpu_curve.calls) == 1
    assert len(fake_widget.ram_curve.calls) == 1
    assert len(fake_widget.gpu_curve.calls) == 0
    assert len(fake_widget.vram_curve.calls) == 0
    assert fake_widget.cpu_gpu_plot.ranges == [(-60.0, 0.0, 0)]
    assert fake_widget.ram_vram_plot.ranges == [(-60.0, 0.0, 0)]
    assert fake_widget.cpu_gpu_plot.titles == []
    assert fake_widget.ram_vram_plot.titles == []


def test_pyqtgraph_update_downsamples_curve_views() -> None:
    fake_widget = make_widget(
        monitor=SimpleNamespace(
            cpu_history=deque(range(10), maxlen=10),
            ram_history=deque(range(10, 20), maxlen=10),
            gpu_history=deque([0.0] * 10, maxlen=10),
            vram_history=deque([0.0] * 10, maxlen=10),
            time_stamps=deque(range(10), maxlen=10),
        ),
        monitor_config=SimpleNamespace(update_interval_seconds=1.0, history_duration_seconds=10.0),
        _history_length=0,
        _history_x=None,
        _history_cpu=None,
        _history_ram=None,
        _history_gpu=None,
        _history_vram=None,
        _history_update_interval=None,
        _last_plot_history=None,
        _plot_point_budget=4,
        _gpu_series_visible=False,
        _vram_series_visible=False,
        cpu_gpu_plot=FakePlot(),
        ram_vram_plot=FakePlot(),
        cpu_curve=FakeCurve(),
        gpu_curve=FakeCurve(),
        ram_curve=FakeCurve(),
        vram_curve=FakeCurve(),
    )

    SystemMonitorWidget.update_pyqtgraph_plots(fake_widget)

    cpu_args = fake_widget.cpu_curve.calls[0][0]
    ram_args = fake_widget.ram_curve.calls[0][0]
    assert np.array_equal(cpu_args[0], np.array([-9.0, -6.0, -3.0, 0.0]))
    assert np.array_equal(cpu_args[1], np.array([0.0, 3.0, 6.0, 9.0], dtype=np.float32))
    assert np.array_equal(ram_args[0], np.array([-9.0, -6.0, -3.0, 0.0]))
    assert np.array_equal(ram_args[1], np.array([10.0, 13.0, 16.0, 19.0], dtype=np.float32))
    assert len(fake_widget.gpu_curve.calls) == 0
    assert len(fake_widget.vram_curve.calls) == 0


def test_pyqtgraph_update_keeps_default_history_resolution_on_narrow_plot() -> None:
    data_length = 300
    fake_widget = make_widget(
        monitor=SimpleNamespace(
            cpu_history=deque(range(data_length), maxlen=data_length),
            ram_history=deque(range(data_length), maxlen=data_length),
            gpu_history=deque([0.0] * data_length, maxlen=data_length),
            vram_history=deque([0.0] * data_length, maxlen=data_length),
            time_stamps=deque(range(data_length), maxlen=data_length),
        ),
        monitor_config=SimpleNamespace(update_interval_seconds=0.2, history_duration_seconds=60.0),
        _history_length=0,
        _history_x=None,
        _history_cpu=None,
        _history_ram=None,
        _history_gpu=None,
        _history_vram=None,
        _history_update_interval=None,
        _last_plot_history=None,
        _plot_point_budget=2000,
        _gpu_series_visible=False,
        _vram_series_visible=False,
        cpu_gpu_plot=FakePlot(width=120),
        ram_vram_plot=FakePlot(width=120),
        cpu_curve=FakeCurve(),
        gpu_curve=FakeCurve(),
        ram_curve=FakeCurve(),
        vram_curve=FakeCurve(),
    )

    SystemMonitorWidget.update_pyqtgraph_plots(fake_widget)

    cpu_args = fake_widget.cpu_curve.calls[0][0]
    assert len(cpu_args[0]) == data_length
    assert len(cpu_args[1]) == data_length
    assert cpu_args[0][-1] == 0.0
    assert cpu_args[1][-1] == np.float32(data_length - 1)


def test_update_config_rebuilds_core_monitor_and_resets_plot_buffers(monkeypatch) -> None:
    FakePersistentMonitor.created.clear()
    monkeypatch.setattr(system_monitor, "PersistentSystemMonitor", FakePersistentMonitor)

    old_config = monitor_config(update_fps=5.0, history_duration_seconds=60.0)
    new_config = monitor_config(update_fps=10.0, history_duration_seconds=30.0)
    old_persistent_monitor = FakePersistentMonitor(update_interval=0.2, history_length=300)
    fake_widget = make_widget(
        monitor_config=old_config,
        monitor=object(),
        persistent_monitor=old_persistent_monitor,
        _history_length=2,
        _history_x=np.array([100.0, 101.0]),
        _history_cpu=np.array([10.0, 20.0], dtype=np.float32),
        _history_ram=np.array([30.0, 40.0], dtype=np.float32),
        _history_gpu=np.array([1.0, 2.0], dtype=np.float32),
        _history_vram=np.array([3.0, 4.0], dtype=np.float32),
        _history_update_interval=0.2,
        _plot_point_budget=2000,
        _gpu_series_visible=True,
        _vram_series_visible=True,
        _last_plot_history=60.0,
        on_metrics_updated=Mock(),
        on_metrics_error=Mock(),
    )

    SystemMonitorWidget.update_config(fake_widget, new_config)

    assert old_persistent_monitor.stopped is True
    assert isinstance(fake_widget.monitor, system_monitor.SystemMonitorCore)
    assert fake_widget.monitor.cpu_history.maxlen == 300
    assert fake_widget.persistent_monitor is FakePersistentMonitor.created[-1]
    assert fake_widget.persistent_monitor.update_interval == 0.1
    assert fake_widget.persistent_monitor.history_length == 300
    assert fake_widget.persistent_monitor.sampler_config == SystemMetricsSamplerConfig(
        enable_gpu_monitoring=True,
        gpu_temperature_monitoring=True,
        cpu_frequency_monitoring=True,
        gpu_refresh_seconds=1.0,
        cpu_frequency_refresh_seconds=5.0,
    )
    assert fake_widget.persistent_monitor.connected is True
    assert fake_widget.persistent_monitor.started is True
    assert fake_widget._history_length == 0
    assert fake_widget._history_x is None
    assert fake_widget._history_update_interval is None
    assert fake_widget._last_plot_history is None


def test_update_config_restarts_when_sampler_policy_changes(monkeypatch) -> None:
    FakePersistentMonitor.created.clear()
    monkeypatch.setattr(system_monitor, "PersistentSystemMonitor", FakePersistentMonitor)

    old_config = monitor_config(enable_gpu_monitoring=True)
    new_config = replace(
        old_config,
        sampler_config=replace(
            old_config.sampler_config,
            enable_gpu_monitoring=False,
        ),
    )
    old_persistent_monitor = FakePersistentMonitor(update_interval=0.2, history_length=300)
    fake_widget = make_widget(
        monitor_config=old_config,
        monitor=object(),
        persistent_monitor=old_persistent_monitor,
        on_metrics_updated=Mock(),
        on_metrics_error=Mock(),
    )

    SystemMonitorWidget.update_config(fake_widget, new_config)

    assert old_persistent_monitor.stopped is True
    assert fake_widget.persistent_monitor.sampler_config.enable_gpu_monitoring is False
