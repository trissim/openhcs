"""Stable contracts for the native main-window docking workspace."""

from __future__ import annotations

import inspect

import pytest
from PyQt6.QtCore import QByteArray, QSettings, Qt
from PyQt6.QtWidgets import QDockWidget, QMainWindow, QWidget

from openhcs.pyqt_gui.services.main_window_workflows import (
    MainWindowDockLayoutStore,
    MainWindowDockPane,
    MainWindowEmbeddedWidgets,
)
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId


PANE_ROWS = (
    (OpenHCSUiWindowId.system_monitor, "System Monitor"),
    (OpenHCSUiWindowId.plate_manager, "Plate Manager"),
    (OpenHCSUiWindowId.zmq_server_manager, "ZMQ Server Manager"),
    (OpenHCSUiWindowId.pipeline_editor, "Pipeline Editor"),
)


def _workspace(qapp) -> tuple[QMainWindow, MainWindowEmbeddedWidgets]:
    main_window = QMainWindow()
    main_window.setDockNestingEnabled(True)
    main_window.setDockOptions(
        QMainWindow.DockOption.AllowNestedDocks
        | QMainWindow.DockOption.AllowTabbedDocks
        | QMainWindow.DockOption.AnimatedDocks
        | QMainWindow.DockOption.GroupedDragging
    )
    embedded = MainWindowEmbeddedWidgets()

    panes = []
    for window_id, title in PANE_ROWS:
        pane = MainWindowDockPane.create(
            main_window=main_window,
            window_id=window_id,
            title=title,
            widget=QWidget(),
        )
        embedded.register(pane)
        panes.append(pane)

    system_monitor, plate_manager, zmq_manager, pipeline_editor = panes
    main_window.addDockWidget(
        Qt.DockWidgetArea.TopDockWidgetArea,
        system_monitor.dock_widget,
    )
    main_window.addDockWidget(
        Qt.DockWidgetArea.LeftDockWidgetArea,
        plate_manager.dock_widget,
    )
    main_window.splitDockWidget(
        plate_manager.dock_widget,
        zmq_manager.dock_widget,
        Qt.Orientation.Vertical,
    )
    main_window.addDockWidget(
        Qt.DockWidgetArea.RightDockWidgetArea,
        pipeline_editor.dock_widget,
    )
    main_window.resize(1000, 700)
    main_window.show()
    qapp.processEvents()
    return main_window, embedded


@pytest.mark.parametrize(
    "window_id",
    tuple(window_id for window_id, _title in PANE_ROWS),
)
def test_showing_dock_pane_preserves_user_geometry(qapp, window_id: str) -> None:
    """Focus/show routes must not replace the user's pane proportions."""

    main_window, embedded = _workspace(qapp)
    pane_geometries_before = {
        pane.window_id: pane.dock_widget.geometry() for pane in embedded.panes()
    }
    pane_areas_before = {
        pane.window_id: main_window.dockWidgetArea(pane.dock_widget)
        for pane in embedded.panes()
    }

    embedded.require_pane(window_id).show()
    qapp.processEvents()

    assert {
        pane.window_id: pane.dock_widget.geometry() for pane in embedded.panes()
    } == pane_geometries_before
    assert {
        pane.window_id: main_window.dockWidgetArea(pane.dock_widget)
        for pane in embedded.panes()
    } == pane_areas_before
    main_window.close()


def test_dock_panes_expose_native_float_move_close_and_all_drop_areas(qapp) -> None:
    main_window, embedded = _workspace(qapp)
    required_features = (
        QDockWidget.DockWidgetFeature.DockWidgetClosable
        | QDockWidget.DockWidgetFeature.DockWidgetMovable
        | QDockWidget.DockWidgetFeature.DockWidgetFloatable
    )

    for pane in embedded.panes():
        assert pane.dock_widget.objectName() == pane.window_id
        assert pane.dock_widget.features() == required_features
        assert pane.dock_widget.allowedAreas() == Qt.DockWidgetArea.AllDockWidgetAreas

    pipeline_pane = embedded.require_pane(OpenHCSUiWindowId.pipeline_editor)
    pipeline_pane.dock_widget.setFloating(True)
    qapp.processEvents()
    assert pipeline_pane.dock_widget.isFloating()

    pipeline_pane.show()
    qapp.processEvents()
    assert pipeline_pane.dock_widget.isFloating()

    plate_pane = embedded.require_pane(OpenHCSUiWindowId.plate_manager)
    plate_area = main_window.dockWidgetArea(plate_pane.dock_widget)
    plate_pane.dock_widget.close()
    qapp.processEvents()
    assert not plate_pane.dock_widget.isVisible()
    plate_pane.show()
    qapp.processEvents()
    assert plate_pane.dock_widget.isVisible()
    assert main_window.dockWidgetArea(plate_pane.dock_widget) == plate_area
    main_window.close()


def test_dock_layout_round_trips_tabs_floating_and_visibility(qapp, tmp_path) -> None:
    settings = QSettings(
        str(tmp_path / "dock-layout.ini"),
        QSettings.Format.IniFormat,
    )
    store = MainWindowDockLayoutStore(settings)
    source_window, source_embedded = _workspace(qapp)
    source_plate = source_embedded.require_pane(OpenHCSUiWindowId.plate_manager)
    source_pipeline = source_embedded.require_pane(OpenHCSUiWindowId.pipeline_editor)
    source_zmq = source_embedded.require_pane(OpenHCSUiWindowId.zmq_server_manager)
    source_system = source_embedded.require_pane(OpenHCSUiWindowId.system_monitor)

    source_window.tabifyDockWidget(
        source_plate.dock_widget,
        source_pipeline.dock_widget,
    )
    source_zmq.dock_widget.setFloating(True)
    source_system.dock_widget.hide()
    qapp.processEvents()
    store.save(source_window)

    restored_store = MainWindowDockLayoutStore(
        QSettings(
            str(tmp_path / "dock-layout.ini"),
            QSettings.Format.IniFormat,
        )
    )
    restored_window, restored_embedded = _workspace(qapp)
    assert restored_store.restore(restored_window)
    qapp.processEvents()

    restored_plate = restored_embedded.require_pane(OpenHCSUiWindowId.plate_manager)
    restored_pipeline = restored_embedded.require_pane(OpenHCSUiWindowId.pipeline_editor)
    restored_zmq = restored_embedded.require_pane(OpenHCSUiWindowId.zmq_server_manager)
    restored_system = restored_embedded.require_pane(OpenHCSUiWindowId.system_monitor)
    assert restored_pipeline.dock_widget in restored_window.tabifiedDockWidgets(
        restored_plate.dock_widget
    )
    assert restored_zmq.dock_widget.isFloating()
    assert not restored_system.dock_widget.isVisible()

    source_window.close()
    restored_window.close()


@pytest.mark.parametrize(
    "invalid_state",
    ("not Qt state", QByteArray(b"not Qt state")),
)
def test_invalid_dock_layout_is_discarded_without_changing_default(
    qapp,
    tmp_path,
    invalid_state,
) -> None:
    settings = QSettings(
        str(tmp_path / "invalid-dock-layout.ini"),
        QSettings.Format.IniFormat,
    )
    settings.setValue(MainWindowDockLayoutStore.STATE_KEY, invalid_state)
    store = MainWindowDockLayoutStore(settings)
    main_window, embedded = _workspace(qapp)
    default_areas = {
        pane.window_id: main_window.dockWidgetArea(pane.dock_widget)
        for pane in embedded.panes()
    }

    assert not store.restore(main_window)
    assert not settings.contains(MainWindowDockLayoutStore.STATE_KEY)
    assert {
        pane.window_id: main_window.dockWidgetArea(pane.dock_widget)
        for pane in embedded.panes()
    } == default_areas
    main_window.close()


def test_dock_layout_store_does_not_depend_on_object_state() -> None:
    source = inspect.getsource(MainWindowDockLayoutStore)
    assert "ObjectState" not in source
