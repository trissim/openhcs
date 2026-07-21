from __future__ import annotations

from functools import wraps
import inspect

from PyQt6.QtTest import QTest

from openhcs.config_framework.global_config import set_global_config_for_editing
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.steps.function_step import FunctionStep
from pyqt_reactive.forms.parameter_form_manager import (
    FormManagerConfig,
    ParameterFormManager,
)
from pyqt_reactive.theming.color_scheme import ColorScheme
from pyqt_reactive.widgets.function_pane import FunctionPaneWidget
from python_introspect import Enableable, mark_enableable


def _registered_callable_pair():
    def declared(image):
        return image

    @wraps(declared)
    def registered(image, *, enabled: bool = True):
        del enabled
        return declared(image)

    registered.__signature__ = inspect.signature(declared).replace(
        parameters=(
            *inspect.signature(declared).parameters.values(),
            Enableable.parameter(),
        )
    )
    mark_enableable(registered)
    return declared, registered


def test_registered_function_pane_moves_only_enabled_control_to_title(qapp) -> None:
    """The canonical registered callable drives title-only enableable chrome."""
    ObjectStateRegistry.clear()
    _declared, registered = _registered_callable_pair()
    pane = FunctionPaneWidget(
        (registered, {}),
        0,
        None,
        scope_id="enableable-function",
        func_scope_token="function-1",
    )
    pane.show()

    try:
        for _ in range(100):
            qapp.processEvents()
            if pane._enabled_widget_moved:
                break
            QTest.qWait(5)

        enabled_widget = pane.form_manager.widgets["enabled"]
        enabled_label = pane.form_manager.labels["enabled"]

        assert pane._enabled_widget_moved
        assert pane.title_layout.isAncestorOf(enabled_widget)
        assert enabled_label.isHidden()
        assert not pane.isAncestorOf(enabled_label.parentWidget())
    finally:
        pane.close()
        pane.deleteLater()
        ObjectStateRegistry.clear()


def test_step_lazy_enableable_contents_stay_dimmed_after_async_build(qapp) -> None:
    """Parent refreshes must not overwrite a nested Enableable's resolved state."""
    ObjectStateRegistry.clear()
    set_global_config_for_editing(GlobalPipelineConfig, GlobalPipelineConfig())
    plate_state = ObjectState(PipelineConfig(), scope_id="enableable-plate")
    step_state = ObjectState(
        FunctionStep(func=lambda image: image),
        scope_id="enableable-plate::step",
        parent_state=plate_state,
    )
    ObjectStateRegistry.register(plate_state, _skip_snapshot=True)
    ObjectStateRegistry.register(step_state, _skip_snapshot=True)
    manager = ParameterFormManager(
        step_state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="enableable-plate::step",
        ),
    )

    try:
        for _ in range(300):
            qapp.processEvents()
            if all(
                field_name in manager.nested_managers
                for field_name in (
                    "step_materialization_config",
                    "streaming_defaults",
                    "napari_streaming_config",
                    "fiji_streaming_config",
                )
            ):
                break
            QTest.qWait(5)

        # Allow later parent batches to run: the regression was an order-dependent
        # parent refresh that undimmed already-built nested forms.
        for _ in range(40):
            qapp.processEvents()
            QTest.qWait(5)

        disabled_managers = []
        for nested_manager in manager.nested_managers.values():
            if "enabled" not in nested_manager.widgets:
                continue
            enabled_path = f"{nested_manager.field_id}.enabled"
            if (
                step_state.parameters.get(enabled_path) is None
                and step_state.get_resolved_value(enabled_path) is False
            ):
                disabled_managers.append(nested_manager)

        assert disabled_managers
        for nested_manager in disabled_managers:
            groupbox = nested_manager.form_tree.owning_groupbox(nested_manager)
            assert groupbox is not None
            enabled_widget = nested_manager.widgets["enabled"]
            content_widgets = [
                widget
                for widget in nested_manager._enabled_field_styling_service._get_value_widgets(
                    groupbox
                )
                if widget is not enabled_widget
            ]
            assert content_widgets
            assert all(
                widget.property("enabled_field_dimmed") is True
                for widget in content_widgets
            )
            assert all(
                widget.graphicsEffect() is not None for widget in content_widgets
            )
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()
