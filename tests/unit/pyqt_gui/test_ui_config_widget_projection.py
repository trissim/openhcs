"""Acceptance coverage for typed widgets in the real nested UI configuration form."""

from dataclasses import fields
from typing import get_type_hints

from PyQt6.QtGui import QColor, QKeySequence
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtTest import QTest


def _wait_until(qapp, predicate, *, attempts: int = 400) -> None:
    for _ in range(attempts):
        qapp.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    raise AssertionError("Nested UI configuration form did not finish materializing")


def test_nested_ui_config_projects_color_enums_and_shortcut_capture(qapp) -> None:
    from objectstate import (
        ObjectState,
        ObjectStateRegistry,
        get_base_config_type,
        set_base_config_type,
    )
    from pyqt_reactive.forms.parameter_form_manager import (
        FormManagerConfig,
        ParameterFormManager,
    )
    from pyqt_reactive.protocols import KeySequenceEditAdapter
    from pyqt_reactive.services.system_monitor_config import (
        PerformanceGraphColor,
        PerformanceMonitorColors,
    )
    from pyqt_reactive.theming import ColorScheme
    from pyqt_reactive.widgets.no_scroll_spinbox import NoScrollComboBox

    from openhcs.pyqt_gui.config import UIConfig

    previous_base_type = get_base_config_type()
    set_base_config_type(UIConfig)
    ObjectStateRegistry.clear()
    state = ObjectState(UIConfig(), scope_id="typed-ui-config-widget-projection")
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    try:
        _wait_until(
            qapp,
            lambda: (
                "check_for_updates_on_startup" in manager.widgets
                and "performance_monitor" in manager.nested_managers
                and "colors"
                in manager.nested_managers["performance_monitor"].nested_managers
                and "shortcuts" in manager.nested_managers
                and "show_help" in manager.nested_managers["shortcuts"].widgets
                and "logging" in manager.nested_managers
                and "level" in manager.nested_managers["logging"].widgets
            ),
        )
        colors = manager.nested_managers["performance_monitor"].nested_managers[
            "colors"
        ]
        shortcuts = manager.nested_managers["shortcuts"]
        logging_config = manager.nested_managers["logging"]
        assert isinstance(manager.widgets["check_for_updates_on_startup"], QCheckBox)
        assert isinstance(logging_config.widgets["level"], NoScrollComboBox)

        color_hints = get_type_hints(PerformanceMonitorColors, include_extras=True)
        assert all(
            color_hints[declared_field.name] is PerformanceGraphColor
            for declared_field in fields(PerformanceMonitorColors)
        )
        assert all(QColor.isValidColor(color.value) for color in PerformanceGraphColor)
        for widget in colors.widgets.values():
            assert isinstance(widget, NoScrollComboBox)
            assert tuple(
                widget.itemData(index) for index in range(widget.count())
            ) == tuple(PerformanceGraphColor)

        help_widget = shortcuts.widgets["show_help"]
        assert isinstance(help_widget, KeySequenceEditAdapter)

        flashes: list[str] = []
        shortcuts.queue_field_flash = flashes.append
        help_widget.setKeySequence(QKeySequence("Ctrl+Shift+H"))
        qapp.processEvents()
        assert state.parameters["shortcuts.show_help"] == "F1"

        help_widget.editingFinished.emit()
        qapp.processEvents()
        assert state.parameters["shortcuts.show_help"] == "Ctrl+Shift+H"
        assert flashes == ["shortcuts.show_help"]
    finally:
        manager.close()
        manager.deleteLater()
        qapp.processEvents()
        ObjectStateRegistry.clear()
        set_base_config_type(previous_base_type)
