from __future__ import annotations

from PyQt6.QtWidgets import QApplication, QScrollArea, QWidget
from objectstate import ObjectState, ObjectStateRegistry

from openhcs.core.config import PipelineConfig
from openhcs.core.source_bindings import (
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
)
from openhcs.pyqt_gui.widgets.source_bindings_editor import (
    SourceBindingsEditorWidget,
)
from pyqt_reactive.forms.parameter_form_manager import (
    FormManagerConfig,
    ParameterFormManager,
)
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.widgets.shared.clickable_help_components import (
    InlineDataclassGroupBox,
)
from pyqt_reactive.widgets.shared.scrollable_form_mixin import ScrollableFormMixin


class _ScrollableFormHarness(QWidget, ScrollableFormMixin):
    def __init__(
        self,
        form_manager: ParameterFormManager,
        scroll_area: QScrollArea,
    ) -> None:
        super().__init__()
        self.form_manager = form_manager
        self.scroll_area = scroll_area


def _process_events(count: int = 20) -> None:
    for _ in range(count):
        QApplication.processEvents()


def test_removed_structural_row_time_travel_preserves_visible_viewport() -> None:
    app = QApplication.instance() or QApplication([])
    ObjectStateRegistry.clear()
    state = ObjectState(PipelineConfig(), scope_id="plate")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="plate",
        ),
    )
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(manager)
    scroll_area.resize(700, 360)
    scroll_area.show()
    owner = _ScrollableFormHarness(manager, scroll_area)

    try:
        for _ in range(200):
            app.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        source_bindings = container._inline_value_widget
        assert isinstance(source_bindings, SourceBindingsEditorWidget)

        def source_filter(value: str) -> SourceFilterClause:
            return SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.EQUALS,
                value,
            )

        source_bindings.add_source_filter_row(source_filter("first"))
        _process_events()
        state.mark_saved()
        ObjectStateRegistry.increment_token(notify=True)

        source_bindings.add_source_filter_row(source_filter("second"))
        _process_events()
        changed_path = state.last_changed_field
        assert changed_path is not None
        assert "[1]" in changed_path

        exact_target = owner._resolve_scroll_target(changed_path)
        assert exact_target is not None
        viewport = owner._scroll_viewport()
        scroll_area.verticalScrollBar().setValue(
            owner._target_scroll_position(exact_target, viewport)
        )
        app.processEvents()
        visible_value = scroll_area.verticalScrollBar().value()
        assert visible_value > 0
        assert owner._target_is_fully_visible(exact_target, owner._scroll_viewport())

        assert ObjectStateRegistry.time_travel_back()
        _process_events()
        assert source_bindings.source_filters_table is not None
        assert source_bindings.source_filters_table.rowCount() == 1
        assert scroll_area.verticalScrollBar().value() == visible_value
        assert owner._resolve_scroll_target(changed_path, warn_missing=False) is None
        fallback = owner._resolve_nearest_ancestor_scroll_target(changed_path)
        assert fallback is not None

        owner.select_and_scroll_to_field(changed_path)
        app.processEvents()
        assert scroll_area.verticalScrollBar().value() == visible_value

        assert ObjectStateRegistry.time_travel_forward()
        _process_events()
        assert source_bindings.source_filters_table.rowCount() == 2
        owner.select_and_scroll_to_field(changed_path)
        app.processEvents()
        assert scroll_area.verticalScrollBar().value() == visible_value

        assert ObjectStateRegistry.time_travel_back()
        _process_events()
        owner.select_and_scroll_to_field(changed_path)
        app.processEvents()
        assert scroll_area.verticalScrollBar().value() == visible_value
    finally:
        scroll_area.close()
        manager.deleteLater()
        ObjectStateRegistry.clear()
