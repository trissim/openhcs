from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import QEvent, QEventLoop, QPoint, QPointF, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QEnterEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QStyle,
    QTableWidgetItem,
    QWidget,
)
from python_introspect import is_enableable

from openhcs.core.source_bindings import (
    ComponentSelector,
    MetadataExtractionRule,
    MetadataSource,
    MetadataSelector,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
    SourceBindingOrigin,
    SourceProjectionRole,
    SourceSetRole,
    NamedSourceBinding,
)
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazySourceBindingsConfig,
    LazyStepSourceBindingsConfig,
    PipelineConfig,
    SourceBindingsConfig,
    StepSourceBindingsConfig,
)
from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.source_bindings_view import SourceInventory
from openhcs.pyqt_gui.widgets.source_bindings_editor import (
    MatchPlanColumn,
    MetadataRuleColumn,
    EditableTableLayout,
    FreeFormCellEditorKind,
    SourceBindingColumn,
    SourceBindingsEditorWidget,
    SourceFilterColumn,
    StructuredSelectorCellWidget,
    StructuredSelectorDialog,
)
from openhcs.config_framework.object_state import ObjectState
from objectstate import (
    DataclassFieldAccess,
    LazyDefaultPlaceholderService,
    ObjectStateRegistry,
    get_base_type_for_lazy,
    replace_raw,
)
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from pyqt_reactive.forms.parameter_form_manager import (
    FormManagerConfig,
    ParameterFormManager,
)
from pyqt_reactive.forms.widget_strategies import PlaceholderConfig, PyQt6WidgetEnhancer
from pyqt_reactive.forms.parameter_info_types import (
    InlineDataclassWidgetInfo,
    create_parameter_info,
)
from pyqt_reactive.widgets.shared.scrollable_form_mixin import ScrollableFormMixin
from pyqt_reactive.widgets.shared.scrollable_form_mixin import ScrollViewport
from pyqt_reactive.widgets.shared.reflowing_vertical_scroll_area import (
    ReflowingVerticalScrollArea,
)
from pyqt_reactive.widgets.structural_table import (
    StructuralDescendantMaskTarget,
    StructuralMaskedContainerTarget,
    StructuralTableCellTarget,
)
from pyqt_reactive.services.window_navigation import (
    NavigationWaitReason,
    RegisteredWindowNavigationRequest,
)
from pyqt_reactive.theming import ColorScheme, StyleSheetGenerator
from pyqt_reactive.animation.flash_mixin import create_groupbox_element
from pyqt_reactive.widgets.shared.clickable_help_components import HelpButton
from pyqt_reactive.widgets.shared.clickable_help_components import HelpContext
from pyqt_reactive.widgets.shared.clickable_help_components import HelpIndicator
from pyqt_reactive.widgets.shared.clickable_help_components import InlineDataclassGroupBox
from pyqt_reactive.widgets.shared.clickable_help_components import ProvenanceLabel
from pyqt_reactive.widgets.no_scroll_spinbox import NoScrollComboBox, NoneAwareCheckBox
from pyqt_reactive.widgets.shared.scoped_table_widget import ScopedTableWidget
from pyqt_reactive.widgets.shared.scope_color_utils import get_scope_color_scheme
class QtApplicationHarness:
    """Nominal owner for the QApplication singleton used by GUI smoke tests."""

    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


def set_combo_cell_text(table, row: int, column: int, text: str) -> None:
    widget = table.cellWidget(row, column)
    assert isinstance(widget, QComboBox)
    index = widget.findText(text)
    assert index >= 0
    widget.setCurrentIndex(index)


def set_editable_cell_text(table, row: int, column: int, text: str) -> None:
    widget = table.cellWidget(row, column)
    if isinstance(widget, StructuredSelectorCellWidget):
        widget.set_text(text)
        return
    if isinstance(widget, QComboBox):
        widget.setCurrentText(text)
        return
    item = table.item(row, column)
    assert item is not None
    item.setText(text)


def table_cell_text(table, row: int, column: int) -> str:
    widget = table.cellWidget(row, column)
    if isinstance(widget, StructuredSelectorCellWidget):
        return widget.text()
    if isinstance(widget, QComboBox):
        value = widget.currentData()
        if isinstance(value, Enum):
            return str(value.value)
        return widget.currentText().strip()
    item = table.item(row, column)
    if item is None:
        return ""
    logical_value = item.data(Qt.ItemDataRole.UserRole)
    if isinstance(logical_value, str):
        return logical_value.strip()
    return item.text().strip()


def binding_cell_position(
    binding_index: int,
    field: SourceBindingColumn,
) -> tuple[int, int]:
    return int(field), binding_index


def set_binding_cell_text(
    table,
    binding_index: int,
    field: SourceBindingColumn,
    text: str,
) -> None:
    set_editable_cell_text(table, *binding_cell_position(binding_index, field), text)


def set_binding_combo_cell_text(
    table,
    binding_index: int,
    field: SourceBindingColumn,
    text: str,
) -> None:
    set_combo_cell_text(table, *binding_cell_position(binding_index, field), text)


def binding_cell_widget(table, binding_index: int, field: SourceBindingColumn):
    return table.cellWidget(*binding_cell_position(binding_index, field))


def test_structured_selector_cell_widget_uses_semantic_editor_kind() -> None:
    QtApplicationHarness.app()

    widget = StructuredSelectorCellWidget(
        values=("file:contains:DNA",),
        value="",
        editor_kind=FreeFormCellEditorKind.FILTER_CLAUSES,
        apply_changes=lambda: None,
    )

    assert widget.editor_kind is FreeFormCellEditorKind.FILTER_CLAUSES


def test_structured_selector_dialog_edits_filter_rows_without_text_area() -> None:
    QtApplicationHarness.app()

    dialog = StructuredSelectorDialog(
        editor_kind=FreeFormCellEditorKind.FILTER_CLAUSES,
        suggestions=("metadata:equals:A01",),
        value="file:contains:DNA",
    )

    assert dialog.table.columnCount() == 3
    assert dialog.value() == "file:contains:DNA"

    dialog._append_suggestion("metadata:equals:A01")

    assert dialog.value() == "file:contains:DNA;metadata:equals:A01"


def test_structured_selector_dialog_uses_closed_domain_combo_cells() -> None:
    QtApplicationHarness.app()

    dialog = StructuredSelectorDialog(
        editor_kind=FreeFormCellEditorKind.COMPONENT_SELECTORS,
        suggestions=("channel=DNA",),
        value="channel=DNA",
    )

    component_widget = dialog.table.cellWidget(0, 0)
    assert isinstance(component_widget, QComboBox)
    assert component_widget.findText(AllComponents.CHANNEL.value) >= 0


def test_structured_selector_dialog_reports_incomplete_rows() -> None:
    QtApplicationHarness.app()

    dialog = StructuredSelectorDialog(
        editor_kind=FreeFormCellEditorKind.METADATA_SELECTORS,
        suggestions=(),
        value="",
    )

    dialog._append_row(("Well", ""))

    assert "Incomplete rows ignored: 1" == dialog.validation_label.text()


def test_source_bindings_config_uses_inline_dataclass_widget_info() -> None:
    info = create_parameter_info(
        "source_bindings",
        StepSourceBindingsConfig,
        StepSourceBindingsConfig(),
    )

    assert isinstance(info, InlineDataclassWidgetInfo)
    assert info.widget_creation_type == "INLINE_DATACLASS"


def test_pipeline_source_bindings_config_uses_inline_dataclass_widget_info() -> None:
    info = create_parameter_info(
        "source_bindings_config",
        SourceBindingsConfig,
        SourceBindingsConfig(),
    )

    assert isinstance(info, InlineDataclassWidgetInfo)
    assert info.widget_creation_type == "INLINE_DATACLASS"


def test_lazy_source_bindings_config_uses_nominal_inline_widget_info() -> None:
    info = create_parameter_info(
        "source_bindings",
        LazyStepSourceBindingsConfig,
        LazyStepSourceBindingsConfig(),
    )

    assert isinstance(info, InlineDataclassWidgetInfo)
    assert info.widget_creation_type == "INLINE_DATACLASS"


def test_lazy_pipeline_source_bindings_config_uses_nominal_inline_widget_info() -> None:
    info = create_parameter_info(
        "source_bindings_config",
        LazySourceBindingsConfig,
        LazySourceBindingsConfig(),
    )

    assert isinstance(info, InlineDataclassWidgetInfo)
    assert info.widget_creation_type == "INLINE_DATACLASS"


def test_source_bindings_editor_builds_from_bindings() -> None:
    QtApplicationHarness.app()

    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    assert widget.layout.count() > 0


def test_source_bindings_editor_renders_pipeline_source_filter_table() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(
        SourceBindingsConfig(
            source_filters=(
                SourceFilterClause(
                    SourceFilterSubject.FILE,
                    SourceFilterMatchType.CONTAINS,
                    "DNA",
                ),
            ),
        )
    )

    assert widget.source_filters_table is not None
    assert widget.source_filters_table.rowCount() == 1

    section_titles = {label.text() for label in widget.findChildren(QLabel)}
    assert "Source Filters" in section_titles


def test_source_bindings_editor_edits_pipeline_source_filters() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(SourceBindingsConfig())

    widget.add_source_filter_row()
    assert widget.source_filters_table is not None
    set_combo_cell_text(
        widget.source_filters_table,
        0,
        int(SourceFilterColumn.SUBJECT),
        "extension",
    )
    set_combo_cell_text(
        widget.source_filters_table,
        0,
        int(SourceFilterColumn.MATCH_TYPE),
        "equals",
    )
    set_editable_cell_text(
        widget.source_filters_table,
        0,
        int(SourceFilterColumn.VALUE),
        ".tif",
    )

    assert widget.get_value().source_filters == (
        SourceFilterClause(
            SourceFilterSubject.EXTENSION,
            SourceFilterMatchType.EQUALS,
            ".tif",
        ),
    )


def test_source_bindings_source_filter_dropdown_ignores_wheel() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(SourceBindingsConfig())

    widget.add_source_filter_row(
        SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "DNA",
        )
    )
    assert widget.source_filters_table is not None
    combo = widget.source_filters_table.cellWidget(
        0,
        int(SourceFilterColumn.MATCH_TYPE),
    )
    assert isinstance(combo, NoScrollComboBox)
    assert combo.currentData() is SourceFilterMatchType.CONTAINS

    wheel_event = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(combo, wheel_event)

    assert combo.currentData() is SourceFilterMatchType.CONTAINS
    assert widget.get_value().source_filters == (
        SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "DNA",
        ),
    )


def test_source_bindings_editor_accepts_lazy_pipeline_source_bindings() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(LazySourceBindingsConfig())

    widget.add_source_filter_row()

    edited = widget.get_value()
    assert get_base_type_for_lazy(type(edited)) is SourceBindingsConfig
    assert edited.source_filters == (
        SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.IS_IMAGE,
        ),
    )


def test_inline_pipeline_source_bindings_widget_accepts_lazy_config_value() -> None:
    QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    for _ in range(80):
        QApplication.processEvents()
        if "source_bindings_config" in manager.widgets:
            break

    container = manager.widgets["source_bindings_config"]
    assert isinstance(container, InlineDataclassGroupBox)
    widget = container._inline_value_widget
    assert isinstance(widget, SourceBindingsEditorWidget)
    assert get_base_type_for_lazy(type(widget.get_value())) is SourceBindingsConfig
    assert len(widget.findChildren(HelpIndicator)) >= 4


def test_inline_source_bindings_root_reset_all_resets_child_fields() -> None:
    QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        root_reset_buttons = [
            button
            for button in container.findChildren(QPushButton)
            if button.text() == "Reset All"
        ]
        assert len(root_reset_buttons) == 1

        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)
        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            )
        )
        for _ in range(10):
            QApplication.processEvents()
        assert DataclassFieldAccess.raw_value(
            state.parameters["source_bindings_config"],
            "source_filters",
        ) is not None

        root_reset_buttons[0].click()
        for _ in range(10):
            QApplication.processEvents()

        assert DataclassFieldAccess.raw_value(
            state.parameters["source_bindings_config"],
            "source_filters",
        ) is None
        assert DataclassFieldAccess.raw_value(
            widget.get_value(),
            "source_filters",
        ) is None
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_pipeline_source_bindings_edit_updates_flat_children_and_persists() -> None:
    QtApplicationHarness.app()
    binding = NamedSourceBinding(alias="DNA")
    state = ObjectState(PipelineConfig())
    source_config = state.parameters["source_bindings_config"]

    state.update_parameter(
        "source_bindings_config",
        replace_raw(
            source_config,
            bindings=(binding,),
        ),
    )

    saved_config = state.to_object(update_delegate=False)
    saved_source_bindings = DataclassFieldAccess.raw_value(
        saved_config,
        "source_bindings_config",
    )

    assert state.parameters["source_bindings_config.bindings"] == (binding,)
    assert DataclassFieldAccess.raw_value(
        saved_source_bindings,
        "bindings",
    ) == (binding,)


def test_pipeline_source_bindings_table_edits_recreate_container_and_children() -> None:
    QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )
    expected_filter = SourceFilterClause(
        SourceFilterSubject.FILE,
        SourceFilterMatchType.CONTAINS,
        "DNA",
    )
    expected_rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=r"(?P<well>A\d{2})_(?P<channel>DNA)\.tif",
    )
    expected_match_plan = SourceBindingMatchPlan(
        method=SourceBindingMatchMethod.METADATA,
        dimensions=(
            SourceBindingMatchDimension(
                fields=(SourceBindingMatchField("DNA", "well"),),
            ),
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        source_container = manager.widgets["source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)

        source_widget.add_source_filter_row()
        assert source_widget.source_filters_table is not None
        set_combo_cell_text(
            source_widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
            "contains",
        )
        set_editable_cell_text(
            source_widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
            "DNA",
        )

        source_widget.add_metadata_rule_row(expected_rule)
        source_widget.add_match_plan_row()
        assert source_widget.match_plan_table is not None
        set_editable_cell_text(
            source_widget.match_plan_table,
            0,
            int(MatchPlanColumn.FIELDS),
            "DNA=well",
        )

        for _ in range(10):
            QApplication.processEvents()

        edited_config = state.parameters["source_bindings_config"]
        assert DataclassFieldAccess.raw_value(
            edited_config,
            "source_filters",
        ) == (expected_filter,)
        assert DataclassFieldAccess.raw_value(
            edited_config,
            "metadata_rules",
        ) == (expected_rule,)
        assert DataclassFieldAccess.raw_value(
            edited_config,
            "match_plan",
        ) == expected_match_plan
        assert state.parameters["source_bindings_config.source_filters"] == (
            expected_filter,
        )
        assert state.parameters["source_bindings_config.metadata_rules"] == (
            expected_rule,
        )
        assert state.parameters["source_bindings_config.match_plan"] == (
            expected_match_plan
        )
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_pipeline_source_bindings_edit_refreshes_step_lazy_preview() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    binding = NamedSourceBinding(alias="DNA")
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if (
                "source_bindings_config" in manager.widgets
                and "step_source_bindings_config" in manager.widgets
            ):
                break

        source_container = manager.widgets["source_bindings_config"]
        step_container = manager.widgets["step_source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        assert isinstance(step_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        step_widget = step_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)
        assert isinstance(step_widget, SourceBindingsEditorWidget)

        queued_flashes: list[str] = []
        manager.queue_flash_local_batch = queued_flashes.extend
        source_widget.add_binding_row(binding)
        for _ in range(10):
            QApplication.processEvents()

        assert state.parameters["source_bindings_config.bindings"] == (binding,)
        assert state.parameters["step_source_bindings_config.bindings"] is None
        assert state.get_resolved_value(
            "step_source_bindings_config.bindings"
        ) == (binding,)
        assert DataclassFieldAccess.raw_value(
            step_widget.get_value(),
            "bindings",
        ) is None
        assert tuple(
            inherited.alias
            for inherited in step_widget._create_step_bindings_dialog().bindings()
        ) == ("DNA",)
        assert source_container._title_label.text().startswith("* ")
        assert step_container._title_label.text().startswith("* ")
        assert "step_source_bindings_config.bindings" in queued_flashes
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_pipeline_step_source_bindings_refresh_open_step_editor_preview() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    binding = NamedSourceBinding(alias="DNA")
    plate_state = ObjectState(PipelineConfig(), scope_id="plate")
    step_state = ObjectState(
        FunctionStep(func=lambda image: image),
        scope_id="plate::functionstep_0",
        parent_state=plate_state,
    )
    ObjectStateRegistry.register(plate_state, _skip_snapshot=True)
    ObjectStateRegistry.register(step_state, _skip_snapshot=True)
    pipeline_manager = ParameterFormManager(
        plate_state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="plate",
        ),
    )
    step_manager = ParameterFormManager(
        step_state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="plate::functionstep_0",
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if (
                "step_source_bindings_config" in pipeline_manager.widgets
                and "source_bindings" in step_manager.widgets
            ):
                break

        pipeline_container = pipeline_manager.widgets["step_source_bindings_config"]
        step_container = step_manager.widgets["source_bindings"]
        assert isinstance(pipeline_container, InlineDataclassGroupBox)
        assert isinstance(step_container, InlineDataclassGroupBox)
        pipeline_widget = pipeline_container._inline_value_widget
        step_widget = step_container._inline_value_widget
        assert isinstance(pipeline_widget, SourceBindingsEditorWidget)
        assert isinstance(step_widget, SourceBindingsEditorWidget)

        assert tuple(step_widget._create_step_bindings_dialog().bindings()) == ()

        pipeline_widget.add_binding_row(binding)
        loop = QEventLoop()
        QTimer.singleShot(260, loop.quit)
        loop.exec()
        QApplication.processEvents()

        assert plate_state.parameters["step_source_bindings_config.bindings"] == (
            binding,
        )
        assert step_state.parameters["source_bindings.bindings"] is None
        assert step_state.get_resolved_value("source_bindings.bindings") == (
            binding,
        )
        assert DataclassFieldAccess.raw_value(
            step_widget.get_value(),
            "bindings",
        ) is None
        assert tuple(
            inherited.alias
            for inherited in step_widget._create_step_bindings_dialog().bindings()
        ) == ("DNA",)
    finally:
        pipeline_manager.deleteLater()
        step_manager.deleteLater()
        ObjectStateRegistry.clear()


def test_pipeline_step_source_filter_time_travel_refreshes_open_step_editor() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    plate_state = ObjectState(PipelineConfig(), scope_id="plate")
    step_state = ObjectState(
        FunctionStep(func=lambda image: image),
        scope_id="plate::functionstep_0",
        parent_state=plate_state,
    )
    ObjectStateRegistry.register(plate_state, _skip_snapshot=True)
    ObjectStateRegistry.register(step_state, _skip_snapshot=True)
    pipeline_manager = ParameterFormManager(
        plate_state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="plate",
        ),
    )
    step_manager = ParameterFormManager(
        step_state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="plate::functionstep_0",
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if (
                "step_source_bindings_config" in pipeline_manager.widgets
                and "source_bindings" in step_manager.widgets
            ):
                break

        pipeline_container = pipeline_manager.widgets["step_source_bindings_config"]
        step_container = step_manager.widgets["source_bindings"]
        assert isinstance(pipeline_container, InlineDataclassGroupBox)
        assert isinstance(step_container, InlineDataclassGroupBox)
        pipeline_widget = pipeline_container._inline_value_widget
        step_widget = step_container._inline_value_widget
        assert isinstance(pipeline_widget, SourceBindingsEditorWidget)
        assert isinstance(step_widget, SourceBindingsEditorWidget)

        first_filter = SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "DNA",
        )
        pipeline_widget.add_source_filter_row(first_filter)
        loop = QEventLoop()
        QTimer.singleShot(260, loop.quit)
        loop.exec()
        QApplication.processEvents()

        assert step_state.parameters["source_bindings.source_filters"] is None
        assert step_state.get_resolved_value("source_bindings.source_filters") == (
            first_filter,
        )
        assert step_widget.source_filters_table is not None
        assert table_cell_text(
            step_widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "contains"

        assert pipeline_widget.source_filters_table is not None
        set_combo_cell_text(
            pipeline_widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
            "equals",
        )
        loop = QEventLoop()
        QTimer.singleShot(260, loop.quit)
        loop.exec()
        QApplication.processEvents()

        second_filter = SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.EQUALS,
            "DNA",
        )
        assert plate_state.parameters["step_source_bindings_config.source_filters"] == (
            second_filter,
        )
        assert step_state.get_resolved_value("source_bindings.source_filters") == (
            second_filter,
        )
        assert table_cell_text(
            step_widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "equals"

        assert ObjectStateRegistry.time_travel_back()
        QApplication.processEvents()
        assert step_state.get_resolved_value("source_bindings.source_filters") == (
            first_filter,
        )
        assert table_cell_text(
            step_widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "contains"

        assert ObjectStateRegistry.time_travel_forward()
        QApplication.processEvents()
        assert step_state.get_resolved_value("source_bindings.source_filters") == (
            second_filter,
        )
        assert table_cell_text(
            step_widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "equals"
    finally:
        pipeline_manager.deleteLater()
        step_manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_enableable_chrome_uses_nominal_step_config() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if (
                "source_bindings_config" in manager.widgets
                and "step_source_bindings_config" in manager.widgets
            ):
                break

        source_container = manager.widgets["source_bindings_config"]
        step_container = manager.widgets["step_source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        assert isinstance(step_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        step_widget = step_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)
        assert isinstance(step_widget, SourceBindingsEditorWidget)

        assert not is_enableable(SourceBindingsConfig)
        assert is_enableable(StepSourceBindingsConfig)
        assert is_enableable(LazyStepSourceBindingsConfig)
        assert source_container.findChildren(NoneAwareCheckBox) == []
        step_checkboxes = step_container.findChildren(NoneAwareCheckBox)
        assert len(step_checkboxes) == 1
        assert not step_checkboxes[0].isChecked()
        assert step_widget.graphicsEffect() is not None

        registered = []
        full_groupbox_registrations = []
        manager.register_flash_masked_container = (
            lambda key, container, mask_rects, *, label_widget=None, layout_watch_widgets=(): registered.append(
                (key, container, mask_rects, label_widget)
            )
        )
        manager.register_flash_groupbox_full = (
            lambda *args, **kwargs: full_groupbox_registrations.append((args, kwargs))
        )
        manager._queue_leaf_flash_for_path(
            "step_source_bindings_config.enabled",
            queue_flash=False,
        )

        assert len(registered) == 1
        key, container_widget, mask_rects, label_widget = registered[0]
        assert key == "step_source_bindings_config.enabled"
        assert container_widget is step_container
        assert label_widget is None
        assert full_groupbox_registrations == []
        masks = tuple(mask_rects(manager))
        assert any(
            needs_square and rect.width() == rect.height()
            for rect, needs_square in masks
        )

        step_container._title_label.mousePressEvent(None)
        for _ in range(10):
            QApplication.processEvents()

        assert state.parameters["step_source_bindings_config.enabled"] is True
        assert step_checkboxes[0].isChecked()
        assert step_widget.graphicsEffect() is None

        refresh_calls = []
        step_widget.refresh = lambda: refresh_calls.append("refresh")  # type: ignore[method-assign]
        original_source_filters_table = step_widget.source_filters_table
        assert step_widget._enabled_reset_button is not None

        step_widget._enabled_reset_button.click()
        for _ in range(20):
            QApplication.processEvents()

        assert state.parameters["step_source_bindings_config.enabled"] is None
        assert DataclassFieldAccess.raw_value(step_widget.get_value(), "enabled") is None
        assert step_widget.source_filters_table is original_source_filters_table
        assert refresh_calls == []
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_enableable_reset_to_inherited_true_is_path_scoped() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(
        GlobalPipelineConfig,
        GlobalPipelineConfig(
            step_source_bindings_config=StepSourceBindingsConfig(enabled=True),
        ),
    )
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "step_source_bindings_config" in manager.widgets:
                break

        step_container = manager.widgets["step_source_bindings_config"]
        assert isinstance(step_container, InlineDataclassGroupBox)
        step_widget = step_container._inline_value_widget
        assert isinstance(step_widget, SourceBindingsEditorWidget)

        state.update_parameter(
            "step_source_bindings_config",
            replace_raw(
                state.parameters["step_source_bindings_config"],
                enabled=True,
            ),
        )
        for _ in range(20):
            QApplication.processEvents()

        full_marker_refreshes = []
        marker_paths = []
        original_markers = step_widget.refresh_section_label_markers

        def record_markers(owner_field_paths=None):
            if owner_field_paths is None:
                full_marker_refreshes.append(None)
            else:
                marker_paths.append(tuple(path.value for path in owner_field_paths))
            return original_markers(owner_field_paths)

        set_value_calls = []
        original_set_value = step_widget.set_value

        def record_set_value(value):
            set_value_calls.append(value)
            return original_set_value(value)

        refresh_calls = []
        step_widget.refresh = lambda: refresh_calls.append("refresh")  # type: ignore[method-assign]
        step_widget.refresh_section_label_markers = record_markers  # type: ignore[method-assign]
        step_widget.set_value = record_set_value  # type: ignore[method-assign]

        assert step_widget._enabled_reset_button is not None
        step_widget._enabled_reset_button.click()
        for _ in range(30):
            QApplication.processEvents()

        assert state.parameters["step_source_bindings_config.enabled"] is None
        assert state.get_resolved_value("step_source_bindings_config.enabled") is True
        assert DataclassFieldAccess.raw_value(step_widget.get_value(), "enabled") is None
        assert step_widget.graphicsEffect() is None
        assert refresh_calls == []
        assert set_value_calls == []
        assert full_marker_refreshes == []
        assert marker_paths
        assert set(marker_paths) == {
            ("step_source_bindings_config.enabled",),
        }
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_enableable_reset_from_false_to_inherited_value_is_path_scoped() -> None:
    QtApplicationHarness.app()

    for inherited_enabled in (True, False):
        ObjectStateRegistry.clear()
        ensure_global_config_context(
            GlobalPipelineConfig,
            GlobalPipelineConfig(
                step_source_bindings_config=StepSourceBindingsConfig(
                    enabled=inherited_enabled,
                ),
            ),
        )
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

        try:
            for _ in range(80):
                QApplication.processEvents()
                if "step_source_bindings_config" in manager.widgets:
                    break

            step_container = manager.widgets["step_source_bindings_config"]
            assert isinstance(step_container, InlineDataclassGroupBox)
            step_widget = step_container._inline_value_widget
            assert isinstance(step_widget, SourceBindingsEditorWidget)

            state.update_parameter(
                "step_source_bindings_config",
                replace_raw(
                    state.parameters["step_source_bindings_config"],
                    enabled=False,
                ),
            )
            for _ in range(20):
                QApplication.processEvents()

            refresh_calls = []
            set_value_calls = []
            original_set_value = step_widget.set_value

            def record_set_value(value):
                set_value_calls.append(value)
                return original_set_value(value)

            step_widget.refresh = lambda: refresh_calls.append("refresh")  # type: ignore[method-assign]
            step_widget.set_value = record_set_value  # type: ignore[method-assign]

            assert step_widget._enabled_reset_button is not None
            step_widget._enabled_reset_button.click()
            for _ in range(30):
                QApplication.processEvents()

            assert state.parameters["step_source_bindings_config.enabled"] is None
            assert (
                state.get_resolved_value("step_source_bindings_config.enabled")
                is inherited_enabled
            )
            assert DataclassFieldAccess.raw_value(step_widget.get_value(), "enabled") is None
            if inherited_enabled:
                assert step_widget.graphicsEffect() is None
            else:
                assert step_widget.graphicsEffect() is not None
            assert refresh_calls == []
            assert set_value_calls == []
        finally:
            manager.deleteLater()
            ObjectStateRegistry.clear()


def test_pipeline_source_bindings_preview_preserves_inherited_table_rows() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if (
                "source_bindings_config" in manager.widgets
                and "step_source_bindings_config" in manager.widgets
            ):
                break

        source_container = manager.widgets["source_bindings_config"]
        step_container = manager.widgets["step_source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        assert isinstance(step_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        step_widget = step_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)
        assert isinstance(step_widget, SourceBindingsEditorWidget)
        assert step_widget.child_field_section_group("bindings").graphicsEffect() is None
        assert step_widget.child_field_section_group("match_plan").graphicsEffect() is None

        source_widget.add_source_filter_row()
        assert source_widget.source_filters_table is not None
        set_combo_cell_text(
            source_widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
            "contains",
        )
        QApplication.processEvents()
        assert source_widget.source_filters_table.rowCount() == 1

        set_editable_cell_text(
            source_widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
            "DNA",
        )
        for _ in range(10):
            QApplication.processEvents()

        assert state.get_resolved_value(
            "step_source_bindings_config.source_filters"
        ) == (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            ),
        )
        assert step_widget.source_filters_table is not None
        assert step_widget.source_filters_table.rowCount() == 1
        assert (
            table_cell_text(
                step_widget.source_filters_table,
                0,
                int(SourceFilterColumn.MATCH_TYPE),
            )
            == "contains"
        )
        assert step_widget.child_field_section_group("source_filters").graphicsEffect() is None
        inherited_match_type_widget = step_widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(inherited_match_type_widget, QComboBox)
        assert (
            PlaceholderConfig.text_color_name()
            in inherited_match_type_widget.styleSheet()
        )
        inherited_value_item = step_widget.source_filters_table.item(
            0,
            int(SourceFilterColumn.VALUE),
        )
        assert inherited_value_item is not None
        assert inherited_value_item.font().italic()

        source_widget.add_match_plan_row()
        assert source_widget.match_plan_table is not None
        set_editable_cell_text(
            source_widget.match_plan_table,
            0,
            int(MatchPlanColumn.FIELDS),
            "DNA=well",
        )
        for _ in range(10):
            QApplication.processEvents()

        source_widget.add_match_plan_row()
        QApplication.processEvents()
        assert source_widget.match_plan_table is not None
        assert source_widget.match_plan_table.rowCount() == 2

        set_editable_cell_text(
            source_widget.match_plan_table,
            1,
            int(MatchPlanColumn.FIELDS),
            "GFP=well",
        )
        for _ in range(10):
            QApplication.processEvents()

        expected_plan = SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(SourceBindingMatchField("DNA", "well"),),
                ),
                SourceBindingMatchDimension(
                    fields=(SourceBindingMatchField("GFP", "well"),),
                ),
            ),
        )
        assert state.get_resolved_value(
            "step_source_bindings_config.match_plan"
        ) == expected_plan
        assert step_widget.match_plan_table is not None
        assert step_widget.match_plan_table.rowCount() == 2
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_enableable_only_updates_skip_table_rebuild() -> None:
    QtApplicationHarness.app()
    inherited_filter = SourceFilterClause(
        SourceFilterSubject.FILE,
        SourceFilterMatchType.CONTAINS,
        "DNA",
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        LazyStepSourceBindingsConfig(
            enabled=True,
            source_filters=(inherited_filter,),
        ),
    )
    refresh_calls = []
    widget.refresh = lambda: refresh_calls.append("refresh")  # type: ignore[method-assign]

    raw_reset = LazyStepSourceBindingsConfig(
        enabled=None,
        source_filters=(inherited_filter,),
    )
    resolved_reset = StepSourceBindingsConfig(
        enabled=False,
        source_filters=(inherited_filter,),
    )

    try:
        widget.set_value(raw_reset)
        widget.set_raw_value_with_resolved_preview(raw_reset, resolved_reset)

        assert refresh_calls == []
        assert DataclassFieldAccess.raw_value(widget.get_value(), "enabled") is None
        assert widget.source_filters_table is not None
        assert widget.source_filters_table.rowCount() == 1
    finally:
        widget.deleteLater()


def test_source_bindings_refresh_detaches_obsolete_subtrees_before_flash_replay() -> None:
    from pyqt_reactive.animation.flash_mixin import WindowFlashOverlay
    from PyQt6.QtWidgets import QDialog, QVBoxLayout

    QtApplicationHarness.app()
    dialog = QDialog()
    dialog_layout = QVBoxLayout(dialog)
    container = InlineDataclassGroupBox(
        title="Source Bindings",
        help_target=StepSourceBindingsConfig,
        color_scheme=ColorScheme(),
        flash_key="source_bindings",
        parent=dialog,
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(),
        parent=container,
    )
    container.set_value_widget(widget)
    dialog_layout.addWidget(container)
    dialog.show()
    QApplication.processEvents()
    container.register_flash_groupbox("source_bindings", container)
    obsolete_widgets = tuple(
        item_widget
        for index in range(widget.layout.count())
        if (item_widget := widget.layout.itemAt(index).widget()) is not None
    )

    try:
        widget.refresh()

        descendants = set(container.findChildren(QWidget))
        assert obsolete_widgets
        assert all(obsolete not in descendants for obsolete in obsolete_widgets)
        assert all(obsolete.parent() is None for obsolete in obsolete_widgets)

        WindowFlashOverlay.cleanup_window(dialog)
        container.reregister_flash_elements()
        assert WindowFlashOverlay.get_for_window(container) is not None
    finally:
        WindowFlashOverlay.cleanup_window(dialog)
        dialog.deleteLater()


def test_source_bindings_table_child_reset_restores_inherited_placeholder_rows() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if (
                "source_bindings_config" in manager.widgets
                and "step_source_bindings_config" in manager.widgets
            ):
                break

        source_container = manager.widgets["source_bindings_config"]
        step_container = manager.widgets["step_source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        assert isinstance(step_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        step_widget = step_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)
        assert isinstance(step_widget, SourceBindingsEditorWidget)

        inherited_filter = SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "DNA",
        )
        source_widget.add_source_filter_row(inherited_filter)
        for _ in range(10):
            QApplication.processEvents()

        assert state.parameters["step_source_bindings_config.source_filters"] is None
        assert state.get_resolved_value(
            "step_source_bindings_config.source_filters"
        ) == (inherited_filter,)
        assert step_widget.source_filters_table is not None
        assert step_widget.source_filters_table.rowCount() == 1
        assert (
            table_cell_text(
                step_widget.source_filters_table,
                0,
                int(SourceFilterColumn.VALUE),
            )
            == "DNA"
        )

        set_editable_cell_text(
            step_widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
            "RNA",
        )
        for _ in range(10):
            QApplication.processEvents()

        local_filter = SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "RNA",
        )
        assert state.parameters["step_source_bindings_config.source_filters"] == (
            local_filter,
        )
        assert (
            table_cell_text(
                step_widget.source_filters_table,
                0,
                int(SourceFilterColumn.VALUE),
            )
            == "RNA"
        )

        step_widget.child_field_reset_button("source_filters").click()
        for _ in range(10):
            QApplication.processEvents()

        assert state.parameters["step_source_bindings_config.source_filters"] is None
        assert state.get_resolved_value(
            "step_source_bindings_config.source_filters"
        ) == (inherited_filter,)
        assert (
            DataclassFieldAccess.raw_value(step_widget.get_value(), "source_filters")
            is None
        )
        assert step_widget.source_filters_table.rowCount() == 1
        assert (
            table_cell_text(
                step_widget.source_filters_table,
                0,
                int(SourceFilterColumn.VALUE),
            )
            == "DNA"
        )
        inherited_value_item = step_widget.source_filters_table.item(
            0,
            int(SourceFilterColumn.VALUE),
        )
        assert inherited_value_item is not None
        assert inherited_value_item.font().italic()
        inherited_match_type_widget = step_widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(inherited_match_type_widget, QComboBox)
        assert (
            PlaceholderConfig.text_color_name()
            in inherited_match_type_widget.styleSheet()
        )
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_inherited_table_combo_activation_materializes_child() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if (
                "source_bindings_config" in manager.widgets
                and "step_source_bindings_config" in manager.widgets
            ):
                break

        source_container = manager.widgets["source_bindings_config"]
        step_container = manager.widgets["step_source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        assert isinstance(step_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        step_widget = step_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)
        assert isinstance(step_widget, SourceBindingsEditorWidget)

        inherited_filter = SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "DNA",
        )
        source_widget.add_source_filter_row(inherited_filter)
        for _ in range(10):
            QApplication.processEvents()

        assert state.parameters["step_source_bindings_config.source_filters"] is None
        history_len_before_activation = len(ObjectStateRegistry.get_branch_history())
        assert step_widget.source_filters_table is not None
        match_type_widget = step_widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(match_type_widget, QComboBox)
        assert (
            table_cell_text(
                step_widget.source_filters_table,
                0,
                int(SourceFilterColumn.MATCH_TYPE),
            )
            == "contains"
        )

        match_type_widget.activated.emit(match_type_widget.currentIndex())
        for _ in range(10):
            QApplication.processEvents()

        assert state.parameters["step_source_bindings_config.source_filters"] == (
            inherited_filter,
        )
        activation_history = ObjectStateRegistry.get_branch_history()[
            history_len_before_activation:
        ]
        assert len(activation_history) == 1
        assert (
            "step_source_bindings_config.source_filters"
            in activation_history[0].label
        )
        assert (
            DataclassFieldAccess.raw_value(step_widget.get_value(), "source_filters")
            == (inherited_filter,)
        )
        assert step_widget.child_field_label("source_filters")._dirty_label_state.is_dirty

        assert ObjectStateRegistry.time_travel_back()
        QApplication.processEvents()
        assert state.parameters["step_source_bindings_config.source_filters"] is None
        assert (
            DataclassFieldAccess.raw_value(step_widget.get_value(), "source_filters")
            is None
        )
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_inherited_table_value_edit_undo_restores_lazy_child() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if (
                "source_bindings_config" in manager.widgets
                and "step_source_bindings_config" in manager.widgets
            ):
                break

        source_container = manager.widgets["source_bindings_config"]
        step_container = manager.widgets["step_source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        assert isinstance(step_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        step_widget = step_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)
        assert isinstance(step_widget, SourceBindingsEditorWidget)

        inherited_filter = SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "DNA",
        )
        source_widget.add_source_filter_row(inherited_filter)
        for _ in range(10):
            QApplication.processEvents()

        assert state.parameters["step_source_bindings_config.source_filters"] is None
        history_len_before_edit = len(ObjectStateRegistry.get_branch_history())
        assert step_widget.source_filters_table is not None
        set_editable_cell_text(
            step_widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
            "RNA",
        )
        for _ in range(10):
            QApplication.processEvents()

        edited_filter = SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "RNA",
        )
        assert state.parameters["step_source_bindings_config.source_filters"] == (
            edited_filter,
        )
        edit_history = ObjectStateRegistry.get_branch_history()[
            history_len_before_edit:
        ]
        assert len(edit_history) == 1
        assert "step_source_bindings_config.source_filters" in edit_history[0].label

        assert ObjectStateRegistry.time_travel_back()
        QApplication.processEvents()
        assert state.parameters["step_source_bindings_config.source_filters"] is None
        assert (
            DataclassFieldAccess.raw_value(step_widget.get_value(), "source_filters")
            is None
        )
        assert table_cell_text(
            step_widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
        ) == "DNA"
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_inherited_table_first_edit_materializes_second_edit_flashes_cell() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if (
                "source_bindings_config" in manager.widgets
                and "step_source_bindings_config" in manager.widgets
            ):
                break

        source_container = manager.widgets["source_bindings_config"]
        step_container = manager.widgets["step_source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        assert isinstance(step_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        step_widget = step_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)
        assert isinstance(step_widget, SourceBindingsEditorWidget)

        inherited_filter = SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "DNA",
        )
        source_widget.add_source_filter_row(inherited_filter)
        for _ in range(10):
            QApplication.processEvents()

        assert state.parameters["step_source_bindings_config.source_filters"] is None
        assert step_widget.source_filters_table is not None

        set_editable_cell_text(
            step_widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
            "RNA",
        )
        for _ in range(10):
            QApplication.processEvents()

        materialized_filter = SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "RNA",
        )
        assert state.parameters["step_source_bindings_config.source_filters"] == (
            materialized_filter,
        )
        value_item = step_widget.source_filters_table.item(
            0,
            int(SourceFilterColumn.VALUE),
        )
        assert value_item is not None
        assert value_item.data(Qt.ItemDataRole.UserRole) == "RNA"
        assert value_item.data(Qt.ItemDataRole.EditRole) == "RNA"
        assert value_item.text() == '*_RNA'
        assert table_cell_text(
            step_widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
        ) == "RNA"

        queued: list[str] = []
        registered = []
        manager.queue_flash_local_batch = queued.extend
        manager.register_flash_masked_container = (
            lambda key, container, mask_rects, *, label_widget=None, layout_watch_widgets=(): registered.append(
                (key, container, mask_rects, label_widget)
            )
        )

        set_editable_cell_text(
            step_widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
            "RNA2",
        )
        for _ in range(10):
            QApplication.processEvents()

        edited_filter = SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "RNA2",
        )
        assert state.parameters["step_source_bindings_config.source_filters"] == (
            edited_filter,
        )
        value_item = step_widget.source_filters_table.item(
            0,
            int(SourceFilterColumn.VALUE),
        )
        assert value_item is not None
        assert value_item.data(Qt.ItemDataRole.UserRole) == "RNA2"
        assert value_item.data(Qt.ItemDataRole.EditRole) == "RNA2"
        assert value_item.text() == '*_RNA2'

        flash_key = "step_source_bindings_config.source_filters[0].value"
        assert flash_key in queued
        assert "step_source_bindings_config" not in queued
        assert any(key == flash_key for key, *_ in registered)
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_child_chrome_and_reset_use_flat_state_paths() -> None:
    QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    ObjectStateRegistry.register(state)
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        source_container = manager.widgets["source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)

        source_widget.add_binding_row(NamedSourceBinding(alias="DNA"))
        loop = QEventLoop()
        QTimer.singleShot(0, loop.quit)
        loop.exec()
        QApplication.processEvents()

        bindings_label = source_widget.child_field_label("bindings")
        reset_button = source_widget.child_field_reset_button("bindings")
        assert bindings_label._label.font().bold()
        assert bindings_label._label.alignment() & Qt.AlignmentFlag.AlignLeft
        assert bindings_label._dirty_label_state.is_dirty
        assert bindings_label._label.font().underline()
        assert reset_button.text().startswith("*")
        assert reset_button.font().underline()

        reset_button.click()
        for _ in range(10):
            QApplication.processEvents()

        assert state.parameters["source_bindings_config.bindings"] is None
        assert not source_widget.child_field_label("bindings")._dirty_label_state.is_dirty
        assert not source_widget.child_field_reset_button("bindings").font().underline()
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_child_provenance_label_grows_on_hover() -> None:
    QtApplicationHarness.app()

    class FakeState:
        def get_provenance(self, dotted_path: str):
            assert dotted_path == "step_source_bindings_config.bindings"
            return ("plate", SourceBindingsConfig)

    label = ProvenanceLabel(
        "Bindings",
        state=FakeState(),
        dotted_path="step_source_bindings_config.bindings",
    )
    base_size = label.font().pointSizeF()
    enter_event = QEnterEvent(QPointF(1, 1), QPointF(1, 1), QPointF(1, 1))
    leave_event = QEvent(QEvent.Type.Leave)

    label.enterEvent(enter_event)
    assert label.font().pointSizeF() > base_size

    label.leaveEvent(leave_event)
    assert label.font().pointSizeF() == base_size


def test_source_bindings_child_path_resolves_inline_scroll_target() -> None:
    QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    class Owner(ScrollableFormMixin):
        def __init__(self, form_manager: ParameterFormManager) -> None:
            self.form_manager = form_manager

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        source_container = manager.widgets["source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)

        target = Owner(manager)._resolve_scroll_target(
            "source_bindings_config.bindings"
        )

        assert target is not None
        assert target.section_path == "source_bindings_config"
        assert target.leaf_name == "bindings"
        assert isinstance(target.structural_flash_target, StructuralMaskedContainerTarget)
        assert target.target_widget is source_widget.child_field_label("bindings")
        assert target.structural_flash_target.scroll_widget() is target.target_widget
        assert not target.is_field
    finally:
        manager.deleteLater()


def test_source_bindings_owner_path_resolves_structural_container_target() -> None:
    QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    class Owner(ScrollableFormMixin):
        def __init__(self, form_manager: ParameterFormManager) -> None:
            self.form_manager = form_manager

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        source_container = manager.widgets["source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)

        target = Owner(manager)._resolve_scroll_target("source_bindings_config")

        assert target is not None
        assert target.field_name == "source_bindings_config"
        assert target.section_path == "source_bindings_config"
        assert target.target_widget is source_container
        assert isinstance(target.structural_flash_target, StructuralDescendantMaskTarget)
        assert target.structural_flash_target.container is source_container
        assert not target.is_field
    finally:
        manager.deleteLater()


def test_source_bindings_child_path_waits_until_inline_child_target_exists() -> None:
    QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    class Owner(ScrollableFormMixin):
        def __init__(self, form_manager: ParameterFormManager) -> None:
            self.form_manager = form_manager

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        source_container = manager.widgets["source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)

        original_target = source_widget.child_field_navigation_target
        source_widget.child_field_navigation_target = lambda field_name: None
        try:
            target = Owner(manager)._resolve_scroll_target(
                "source_bindings_config.source_filters"
            )
        finally:
            source_widget.child_field_navigation_target = original_target

        assert target is None
    finally:
        manager.deleteLater()


def test_source_bindings_navigation_falls_back_to_visible_owner_section() -> None:
    app = QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(manager)
    scroll_area.resize(640, 360)
    scroll_area.show()

    class Owner(QWidget, ScrollableFormMixin):
        def __init__(
            self,
            form_manager: ParameterFormManager,
            scroll: QScrollArea,
        ) -> None:
            super().__init__()
            self.form_manager = form_manager
            self.scroll_area = scroll
            self.flashed_targets = []

        def _flash_scroll_target(self, target) -> None:
            self.flashed_targets.append(target)

    try:
        for _ in range(80):
            app.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        source_container = manager.widgets["source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)

        original_target = source_widget.child_field_navigation_target
        source_widget.child_field_navigation_target = lambda field_name: None
        try:
            owner = Owner(manager, scroll_area)
            request = RegisteredWindowNavigationRequest(
                window=scroll_area.window(),
                field_path="source_bindings_config.source_filters",
            )
            driver = owner.window_navigation_driver()

            first = driver.readiness(request)
            second = driver.readiness(request)
            owner.select_and_scroll_to_field("source_bindings_config.source_filters")
        finally:
            source_widget.child_field_navigation_target = original_target

        assert first.wait_reason is NavigationWaitReason.LAYOUT
        assert not second.needs_wait
        assert len(owner.flashed_targets) == 1
        fallback_target = owner.flashed_targets[0]
        assert fallback_target.field_name == "source_bindings_config"
        assert fallback_target.target_widget is source_container
    finally:
        scroll_area.deleteLater()
        manager.deleteLater()


def test_source_bindings_child_navigation_waits_for_stable_geometry() -> None:
    app = QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(manager)
    scroll_area.resize(640, 360)
    scroll_area.show()

    class Owner(ScrollableFormMixin):
        def __init__(
            self,
            form_manager: ParameterFormManager,
            scroll: QScrollArea,
        ) -> None:
            self.form_manager = form_manager
            self.scroll_area = scroll

    try:
        for _ in range(80):
            app.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        owner = Owner(manager, scroll_area)
        driver = owner.window_navigation_driver()
        request = RegisteredWindowNavigationRequest(
            window=scroll_area.window(),
            field_path="source_bindings_config.source_filters",
        )

        first = driver.readiness(request)
        settled = first
        for _ in range(80):
            app.processEvents()
            settled = driver.readiness(request)
            if not settled.needs_wait:
                break

        assert first.wait_reason is NavigationWaitReason.LAYOUT
        assert not settled.needs_wait
    finally:
        scroll_area.deleteLater()
        manager.deleteLater()


def test_source_bindings_editor_uses_resolved_placeholder_tables() -> None:
    QtApplicationHarness.app()
    inherited_binding = NamedSourceBinding(alias="InheritedDNA")
    inherited_rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=r"(?P<well>A\d{2})_(?P<channel>DNA)\.tif",
    )
    inherited_filter = SourceFilterClause(
        SourceFilterSubject.FILE,
        SourceFilterMatchType.CONTAINS,
        "DNA",
    )
    local_raw = LazyStepSourceBindingsConfig()
    resolved_display = StepSourceBindingsConfig(
        bindings=(inherited_binding,),
        metadata_rules=(inherited_rule,),
        source_filters=(inherited_filter,),
        match_plan=None,
        enabled=True,
    )

    widget = SourceBindingsEditorWidget.from_bindings(
        local_raw,
        display_bindings=resolved_display,
    )
    dialog = widget._create_step_bindings_dialog()

    assert tuple(binding.alias for binding in dialog.bindings()) == ("InheritedDNA",)
    assert widget.metadata_rules_table is not None
    assert widget.metadata_rules_table.rowCount() == 1
    assert widget.source_filters_table is not None
    assert widget.source_filters_table.rowCount() == 1

    match_type_widget = widget.source_filters_table.cellWidget(
        0,
        int(SourceFilterColumn.MATCH_TYPE),
    )
    assert isinstance(match_type_widget, QComboBox)
    assert match_type_widget.currentData() is SourceFilterMatchType.CONTAINS
    assert match_type_widget.currentText() == "contains"
    assert PlaceholderConfig.text_color_name() in match_type_widget.styleSheet()

    value_item = widget.source_filters_table.item(0, int(SourceFilterColumn.VALUE))
    assert value_item is not None
    assert value_item.font().italic()
    assert value_item.foreground().color().name() == PlaceholderConfig.text_color_name()
    assert DataclassFieldAccess.raw_value(widget.get_value(), "bindings") is None
    assert DataclassFieldAccess.raw_value(widget.get_value(), "metadata_rules") is None
    assert DataclassFieldAccess.raw_value(widget.get_value(), "source_filters") is None


def test_source_bindings_table_row_value_read_does_not_emit_item_changed() -> None:
    QtApplicationHarness.app()
    inherited_filter = SourceFilterClause(
        SourceFilterSubject.FILE,
        SourceFilterMatchType.CONTAINS,
        "DNA",
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        LazyStepSourceBindingsConfig(),
        display_bindings=StepSourceBindingsConfig(
            source_filters=(inherited_filter,),
            enabled=True,
        ),
    )

    assert widget.source_filters_table is not None
    assert widget.source_filters_controller is not None
    value_item = widget.source_filters_table.item(0, int(SourceFilterColumn.VALUE))
    assert value_item is not None
    value_item.setText("*DNA")
    value_item.setData(Qt.ItemDataRole.UserRole, "DNA")

    emitted: list[object] = []
    widget.source_filters_table.itemChanged.connect(emitted.append)

    assert widget.source_filters_controller.row_values() == (
        ("file", "contains", "DNA", ""),
    )
    assert emitted == []


def test_source_bindings_editor_editing_inherited_table_makes_lazy_child_concrete() -> None:
    QtApplicationHarness.app()
    inherited_filter = SourceFilterClause(
        SourceFilterSubject.FILE,
        SourceFilterMatchType.CONTAINS,
        "DNA",
    )
    local_raw = LazyStepSourceBindingsConfig()
    resolved_display = StepSourceBindingsConfig(
        source_filters=(inherited_filter,),
        enabled=True,
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        local_raw,
        display_bindings=resolved_display,
    )

    assert widget.source_filters_table is not None
    set_editable_cell_text(
        widget.source_filters_table,
        0,
        int(SourceFilterColumn.VALUE),
        "RNA",
    )
    QApplication.processEvents()

    edited = widget.get_value()
    assert type(edited) is LazyStepSourceBindingsConfig
    assert DataclassFieldAccess.raw_value(edited, "enabled") is None
    assert DataclassFieldAccess.raw_value(edited, "bindings") is None
    assert DataclassFieldAccess.raw_value(edited, "source_filters") == (
        SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "RNA",
        ),
    )


def test_source_bindings_child_reset_restores_lazy_inheritance_slot() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    state = ObjectState(FunctionStep(func=lambda image: image), scope_id="step")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="step",
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings" in manager.widgets:
                break

        container = manager.widgets["source_bindings"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)

        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            )
        )
        QApplication.processEvents()
        assert state.parameters["source_bindings.source_filters"] == (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            ),
        )
        assert widget.child_field_label("source_filters")._dirty_label_state.is_dirty

        widget.child_field_reset_button("source_filters").click()
        QApplication.processEvents()

        assert state.parameters["source_bindings.source_filters"] is None
        assert DataclassFieldAccess.raw_value(
            state.parameters["source_bindings"],
            "source_filters",
        ) is None
        assert not widget.child_field_label("source_filters")._dirty_label_state.is_dirty
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_child_reset_noops_for_already_inherited_table_preview() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    inherited_filter = SourceFilterClause(
        SourceFilterSubject.FILE,
        SourceFilterMatchType.CONTAINS,
        "DNA",
    )
    plate_state = ObjectState(PipelineConfig(), scope_id="plate")
    step_state = ObjectState(
        FunctionStep(func=lambda image: image),
        scope_id="plate::functionstep_0",
        parent_state=plate_state,
    )
    ObjectStateRegistry.register(plate_state, _skip_snapshot=True)
    ObjectStateRegistry.register(step_state, _skip_snapshot=True)
    pipeline_manager = ParameterFormManager(
        plate_state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="plate",
        ),
    )
    step_manager = ParameterFormManager(
        step_state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="plate::functionstep_0",
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if (
                "step_source_bindings_config" in pipeline_manager.widgets
                and "source_bindings" in step_manager.widgets
            ):
                break

        pipeline_container = pipeline_manager.widgets["step_source_bindings_config"]
        step_container = step_manager.widgets["source_bindings"]
        assert isinstance(pipeline_container, InlineDataclassGroupBox)
        assert isinstance(step_container, InlineDataclassGroupBox)
        pipeline_widget = pipeline_container._inline_value_widget
        step_widget = step_container._inline_value_widget
        assert isinstance(pipeline_widget, SourceBindingsEditorWidget)
        assert isinstance(step_widget, SourceBindingsEditorWidget)

        pipeline_widget.add_source_filter_row(inherited_filter)
        loop = QEventLoop()
        QTimer.singleShot(260, loop.quit)
        loop.exec()
        QApplication.processEvents()

        assert step_state.parameters["source_bindings.source_filters"] is None
        assert step_state.get_resolved_value("source_bindings.source_filters") == (
            inherited_filter,
        )
        assert step_widget.source_filters_table is not None
        assert step_widget.source_filters_table.rowCount() == 1

        step_widget.child_field_reset_button("source_filters").click()
        QApplication.processEvents()

        assert step_state.parameters["source_bindings.source_filters"] is None
        assert step_state.get_resolved_value("source_bindings.source_filters") == (
            inherited_filter,
        )
        assert DataclassFieldAccess.raw_value(
            step_widget.get_value(),
            "source_filters",
        ) is None
        assert step_widget.source_filters_table.rowCount() == 1
        assert (
            table_cell_text(
                step_widget.source_filters_table,
                0,
                int(SourceFilterColumn.MATCH_TYPE),
            )
            == "contains"
        )
    finally:
        pipeline_manager.deleteLater()
        step_manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_editor_preserves_unedited_lazy_inheritance_slots() -> None:
    QtApplicationHarness.app()
    inherited_binding = NamedSourceBinding(alias="InheritedDNA")
    inherited_rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=r"(?P<well>A\d{2})_(?P<channel>DNA)\.tif",
    )
    local_raw = LazyStepSourceBindingsConfig()
    resolved_display = StepSourceBindingsConfig(
        bindings=(inherited_binding,),
        metadata_rules=(inherited_rule,),
        source_filters=(),
        match_plan=None,
        enabled=False,
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        local_raw,
        display_bindings=resolved_display,
    )

    widget.add_metadata_rule_row(
        MetadataExtractionRule(
            source=MetadataSource.FILE_NAME,
            pattern=r"(?P<site>s\d+)",
        )
    )

    edited = widget.get_value()
    assert DataclassFieldAccess.raw_value(edited, "bindings") is None
    assert DataclassFieldAccess.raw_value(edited, "source_filters") is None
    assert edited.metadata_rules == (
        inherited_rule,
        MetadataExtractionRule(
            source=MetadataSource.FILE_NAME,
            pattern=r"(?P<site>s\d+)",
        ),
    )


def test_source_bindings_editor_uses_compact_inline_step_binding_summary() -> None:
    QtApplicationHarness.app()

    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA"),),
        )
    )

    button_labels = tuple(
        button.text() for button in widget.findChildren(QPushButton)
    )
    assert widget.step_bindings_table is None
    assert "Edit bindings..." in button_labels


def test_source_bindings_editor_tables_expand_without_vertical_scrollbars() -> None:
    QtApplicationHarness.app()

    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA"),),
        )
    )
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table

    assert table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert not table.verticalHeader().isHidden()
    assert table.verticalHeaderItem(int(SourceBindingColumn.ALIAS)).text() == "Alias"
    expected_minimum_height = table.horizontalHeader().height() + sum(
        table.rowHeight(row)
        for row in range(table.rowCount())
    )
    assert table.height() >= expected_minimum_height
    assert table.viewport().height() >= sum(
        table.rowHeight(row) for row in range(table.rowCount())
    )
    final_index = table.model().index(table.rowCount() - 1, 0)
    assert table.visualRect(final_index).bottom() <= table.viewport().rect().bottom()


def test_editable_table_layout_keeps_final_row_visible_with_or_without_scrollbar() -> None:
    app = QtApplicationHarness.app()
    table = ScopedTableWidget(2, 2)
    table.setItem(0, 0, QTableWidgetItem("field"))
    table.setItem(0, 1, QTableWidgetItem("short"))
    table.setItem(1, 0, QTableWidgetItem("imported metadata tables"))
    table.setItem(1, 1, QTableWidgetItem("a value wide enough to require scrolling"))
    EditableTableLayout.configure(table)
    EditableTableLayout.fit_to_rows(table)
    table.resize(180, table.height())
    table.show()
    for _ in range(4):
        app.processEvents()
    EditableTableLayout.fit_to_rows(table)
    app.processEvents()

    expected_rows_height = sum(
        table.rowHeight(row) for row in range(table.rowCount())
    )
    final_index = table.model().index(table.rowCount() - 1, 0)
    assert table.horizontalScrollBar().isVisible()
    assert table.viewport().height() >= expected_rows_height
    assert table.visualRect(final_index).bottom() <= table.viewport().rect().bottom()
    viewport_rect = QRect(
        table.viewport().mapTo(table, table.viewport().rect().topLeft()),
        table.viewport().size(),
    )
    horizontal_bar_rect = QRect(
        table.horizontalScrollBar().mapTo(
            table,
            table.horizontalScrollBar().rect().topLeft(),
        ),
        table.horizontalScrollBar().size(),
    )
    assert not viewport_rect.intersects(horizontal_bar_rect)

    table.resize(1200, table.height())
    for _ in range(4):
        app.processEvents()
    EditableTableLayout.fit_to_rows(table)
    app.processEvents()

    assert not table.horizontalScrollBar().isVisible()
    assert table.viewport().height() >= expected_rows_height
    assert table.visualRect(final_index).bottom() <= table.viewport().rect().bottom()


def test_pipeline_sources_preview_fits_rows_and_bar_in_reflowing_config_body() -> None:
    app = QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())
    widget.set_preview_context(source_bindings=SourceBindingsConfig())
    scroll_area = ReflowingVerticalScrollArea()
    scroll_area.setStyleSheet(
        StyleSheetGenerator(ColorScheme()).generate_config_window_style()
    )
    scroll_area.setWidget(widget)
    scroll_area.show()

    try:
        pipeline_sources_table = widget.findChildren(ScopedTableWidget)[0]
        for width in (300, 150, 70, 300):
            scroll_area.resize(width, 600)
            for _ in range(8):
                app.processEvents()

            table = pipeline_sources_table
            rows_height = sum(
                table.rowHeight(row) for row in range(table.rowCount())
            )
            viewport_vertical_margin = 2 * table.style().pixelMetric(
                QStyle.PixelMetric.PM_FocusFrameVMargin,
                None,
                table,
            )
            expected_height = (
                max(
                    table.horizontalHeader().height(),
                    table.horizontalHeader().sizeHint().height(),
                )
                + rows_height
                + table.horizontalScrollBar().sizeHint().height()
                + 2 * table.frameWidth()
                + viewport_vertical_margin
            )
            final_index = table.model().index(table.rowCount() - 1, 0)

            assert table.height() >= expected_height
            assert table.viewport().height() >= (
                rows_height + viewport_vertical_margin
            )
            assert (
                table.visualRect(final_index).bottom()
                <= table.viewport().rect().bottom()
            )
            if table.horizontalScrollBar().isVisible():
                viewport_rect = QRect(
                    table.viewport().mapTo(
                        table,
                        table.viewport().rect().topLeft(),
                    ),
                    table.viewport().size(),
                )
                horizontal_bar_rect = QRect(
                    table.horizontalScrollBar().mapTo(
                        table,
                        table.horizontalScrollBar().rect().topLeft(),
                    ),
                    table.horizontalScrollBar().size(),
                )
                assert not viewport_rect.intersects(horizontal_bar_rect)
    finally:
        scroll_area.close()
        widget.deleteLater()


def test_source_filters_fit_complete_rows_and_bar_inside_scoped_section() -> None:
    app = QtApplicationHarness.app()
    filters = tuple(
        SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.IS_IMAGE,
            f"source-{index}",
        )
        for index in range(3)
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(source_filters=filters)
    )
    scroll_area = ReflowingVerticalScrollArea()
    scroll_area.setStyleSheet(
        StyleSheetGenerator(ColorScheme()).generate_config_window_style()
    )
    scroll_area.setWidget(widget)
    widget.set_scope_color_scheme(
        get_scope_color_scheme("plate::step_0", step_index=0)
    )
    scroll_area.show()

    try:
        table = widget.source_filters_table
        assert table is not None
        section = table.parentWidget()
        assert section is not None

        for width in (360, 150, 70, 360):
            scroll_area.resize(width, 900)
            for _ in range(8):
                app.processEvents()

            rows_height = sum(
                table.rowHeight(row) for row in range(table.rowCount())
            )
            viewport_vertical_margin = 2 * table.style().pixelMetric(
                QStyle.PixelMetric.PM_FocusFrameVMargin,
                None,
                table,
            )
            viewport_rect = QRect(
                table.viewport().mapTo(
                    table,
                    table.viewport().rect().topLeft(),
                ),
                table.viewport().size(),
            )
            horizontal_bar = table.horizontalScrollBar()
            horizontal_bar_rect = QRect(
                horizontal_bar.mapTo(
                    table,
                    horizontal_bar.rect().topLeft(),
                ),
                horizontal_bar.size(),
            )

            assert table.rowCount() == len(filters)
            assert horizontal_bar.isVisible()
            assert table.viewport().height() >= (
                rows_height + viewport_vertical_margin
            )
            assert all(
                table.visualRect(table.model().index(row, 0)).bottom()
                <= table.viewport().rect().bottom()
                for row in range(table.rowCount())
            )
            assert table.rect().contains(viewport_rect)
            assert table.rect().contains(horizontal_bar_rect)
            assert not viewport_rect.intersects(horizontal_bar_rect)
            assert section.contentsRect().contains(table.geometry())
            assert table._border_overlay.geometry() == table.rect()
    finally:
        scroll_area.close()
        widget.deleteLater()


def test_step_metadata_rule_cells_show_inherited_and_local_edit_markers() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    inherited_rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=r"(?P<well>A\d{2})_(?P<channel>DNA)\.tif",
    )
    state = ObjectState(
        PipelineConfig(
            source_bindings_config=SourceBindingsConfig(
                metadata_rules=(inherited_rule,),
            )
        ),
        scope_id="plate",
    )
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="plate",
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "step_source_bindings_config" in manager.widgets:
                break

        step_container = manager.widgets["step_source_bindings_config"]
        assert isinstance(step_container, InlineDataclassGroupBox)
        step_widget = step_container._inline_value_widget
        assert isinstance(step_widget, SourceBindingsEditorWidget)
        table = step_widget.metadata_rules_table
        assert table is not None
        assert table.rowCount() == 1
        assert state.parameters["step_source_bindings_config.metadata_rules"] is None

        source_widget = table.cellWidget(0, int(MetadataRuleColumn.SOURCE))
        pattern_item = table.item(0, int(MetadataRuleColumn.PATTERN))
        assert isinstance(source_widget, QComboBox)
        assert pattern_item is not None
        assert source_widget.currentText() == "_file_name"
        assert pattern_item.text() == f"_{inherited_rule.pattern}"
        assert table_cell_text(table, 0, int(MetadataRuleColumn.SOURCE)) == "file_name"
        assert table_cell_text(table, 0, int(MetadataRuleColumn.PATTERN)) == inherited_rule.pattern

        local_pattern = r"(?P<well>B\d{2})_(?P<channel>RNA)\.tif"
        set_editable_cell_text(
            table,
            0,
            int(MetadataRuleColumn.PATTERN),
            local_pattern,
        )
        for _ in range(10):
            QApplication.processEvents()

        pattern_item = table.item(0, int(MetadataRuleColumn.PATTERN))
        assert pattern_item is not None
        assert pattern_item.text() == f"*_{local_pattern}"
        assert pattern_item.data(Qt.ItemDataRole.UserRole) == local_pattern
        assert table_cell_text(table, 0, int(MetadataRuleColumn.PATTERN)) == local_pattern
        assert state.parameters["step_source_bindings_config.metadata_rules"] == (
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=local_pattern,
            ),
        )
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_editor_explains_binding_selector_and_roles() -> None:
    QtApplicationHarness.app()

    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA"),),
        )
    )
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table

    select_axes = table.verticalHeaderItem(int(SourceBindingColumn.COMPONENTS))
    select_metadata = table.verticalHeaderItem(int(SourceBindingColumn.METADATA))
    assign_axes = table.verticalHeaderItem(int(SourceBindingColumn.IDENTITY))
    set_role = table.verticalHeaderItem(int(SourceBindingColumn.SET_ROLE))
    projection_role = table.verticalHeaderItem(
        int(SourceBindingColumn.PROJECTION_ROLE)
    )

    assert select_axes.text() == "Select Axes"
    assert "choose sources" in select_axes.toolTip()
    assert "does not assign" in select_axes.toolTip()
    assert select_metadata.text() == "Select Metadata"
    assert "filters candidates" in select_metadata.toolTip()
    assert "Source Set Pairing" in select_metadata.toolTip()
    assert assign_axes.text() == "Assign Axes"
    assert "attached after selection" in assign_axes.toolTip()
    assert set_role.text() == "Set Role"
    assert "broadcast" in set_role.toolTip()
    assert projection_role.text() == "Projection Role"
    assert "typed source artifact" in projection_role.toolTip()


def test_source_bindings_editor_explains_image_set_pairing_table() -> None:
    QtApplicationHarness.app()

    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())
    assert widget.match_plan_table is not None

    method_header = widget.match_plan_table.horizontalHeaderItem(
        int(MatchPlanColumn.METHOD)
    )
    fields_header = widget.match_plan_table.horizontalHeaderItem(
        int(MatchPlanColumn.FIELDS)
    )
    button_labels = tuple(
        button.text() for button in widget.findChildren(QPushButton)
    )
    section_titles = tuple(label.text() for label in widget.findChildren(QLabel))

    assert method_header.text() == "Pairing Method"
    assert "grouped into one source set" in method_header.toolTip()
    assert fields_header.text() == "Pairing Keys"
    assert "DNA=Well;GFP=Well" in fields_header.toolTip()
    assert "Add pairing key" in button_labels
    assert "Source Set Pairing" in section_titles


def test_source_bindings_editor_tables_use_scoped_table_abstraction() -> None:
    QtApplicationHarness.app()

    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA"),),
        )
    )
    scheme = get_scope_color_scheme("plate::step_0", step_index=0)

    widget.set_scope_color_scheme(scheme)
    tables = widget.findChildren(ScopedTableWidget)
    dialog = widget._create_step_bindings_dialog()

    assert tables
    assert all(table._scope_color_scheme is scheme for table in tables)
    assert isinstance(dialog.editor.table, ScopedTableWidget)
    assert dialog.editor.table._scope_color_scheme is scheme


def test_inline_groupbox_propagates_scope_to_source_binding_tables() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA"),),
        )
    )
    container = InlineDataclassGroupBox(
        title="Source Bindings",
        help_target=StepSourceBindingsConfig,
        color_scheme=ColorScheme(),
        flash_key="source_bindings",
    )
    scheme = get_scope_color_scheme("plate::step_0", step_index=0)

    container.set_scope_color_scheme(scheme)
    container.set_value_widget(widget)

    tables = widget.findChildren(ScopedTableWidget)
    assert tables
    assert widget._scope_color_scheme is scheme
    assert all(table._scope_color_scheme is scheme for table in tables)


def test_global_scope_color_scheme_is_white_and_layered() -> None:
    scheme = get_scope_color_scheme("")

    assert scheme.scope_id == ""
    assert scheme.step_border_layers == [(3, 1, "solid")]
    assert scheme.step_border_width == 3
    assert scheme.accent_qcolor().name().lower() == "#ffffff"
    assert scheme.border_layer_qcolor(scheme.step_border_layers[0]).name().lower() == (
        "#ffffff"
    )


def test_empty_root_scope_form_manager_receives_global_white_scheme() -> None:
    QtApplicationHarness.app()
    manager = ParameterFormManager(
        ObjectState(FunctionStep(func=lambda image: image), scope_id=""),
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="",
        ),
    )

    try:
        assert manager._scope_color_scheme is not None
        assert manager._scope_color_scheme.accent_qcolor().name().lower() == "#ffffff"
    finally:
        manager.deleteLater()


def test_white_scope_help_controls_use_neutral_outline_chrome() -> None:
    QtApplicationHarness.app()
    context = HelpContext(
        color_scheme=ColorScheme(),
        scope_accent_color=QColor("#ffffff"),
    )
    button = HelpButton(context, text="?")
    indicator = HelpIndicator(context)

    try:
        button_style = button.styleSheet()
        indicator_style = indicator.styleSheet()

        assert "background-color: #555555" in button_style
        assert "border: 1px solid #ffffff" in button_style
        assert "background-color: #555555" in indicator_style
        assert "border: 1px solid #ffffff" in indicator_style
    finally:
        button.deleteLater()
        indicator.deleteLater()


def test_source_bindings_placeholder_preview_updates_for_same_summary_text() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(LazyStepSourceBindingsConfig())
    container = InlineDataclassGroupBox(
        title="Source Bindings",
        help_target=StepSourceBindingsConfig,
        color_scheme=ColorScheme(),
        flash_key="source_bindings",
    )
    container.set_value_widget(widget)

    first = StepSourceBindingsConfig(
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(SourceBindingMatchField("DNA", "well"),),
                ),
            ),
        ),
    )
    second = StepSourceBindingsConfig(
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(SourceBindingMatchField("DNA", "well"),),
                ),
                SourceBindingMatchDimension(
                    fields=(SourceBindingMatchField("GFP", "site"),),
                ),
            ),
        ),
    )
    first_placeholder = LazyDefaultPlaceholderService._format_placeholder_text(
        first,
        "Pipeline default",
    )
    second_placeholder = LazyDefaultPlaceholderService._format_placeholder_text(
        second,
        "Pipeline default",
    )
    assert first_placeholder == second_placeholder

    PyQt6WidgetEnhancer.apply_placeholder_with_value(
        container,
        first,
        first_placeholder,
    )
    assert widget.match_plan_table is not None
    assert widget.match_plan_table.rowCount() == 1

    PyQt6WidgetEnhancer.apply_placeholder_with_value(
        container,
        second,
        second_placeholder,
    )
    assert widget.match_plan_table.rowCount() == 2
    assert (
        table_cell_text(widget.match_plan_table, 1, int(MatchPlanColumn.FIELDS))
        == "GFP=site"
    )


def test_inline_source_bindings_widget_updates_object_state() -> None:
    QtApplicationHarness.app()
    step = FunctionStep(func=lambda image: image)
    state = ObjectState(step)
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    for _ in range(20):
        QApplication.processEvents()
        widget = manager.findChild(SourceBindingsEditorWidget)
        if widget is not None:
            break
    else:
        widget = None
    assert widget is not None

    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    QApplication.processEvents()

    edited = state.parameters["source_bindings"]
    assert isinstance(edited, StepSourceBindingsConfig)
    assert edited.bindings[0].alias == "DNA"


def test_inline_step_source_bindings_time_travel_preserves_dirty_marker() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    state = ObjectState(FunctionStep(func=lambda image: image), scope_id="step")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="step",
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings" in manager.widgets:
                break

        container = manager.widgets["source_bindings"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)

        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.EQUALS,
                "DNA",
            )
        )
        QApplication.processEvents()
        assert widget.source_filters_table is not None
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "equals"

        set_combo_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
            "contains",
        )
        QApplication.processEvents()
        match_type_widget = widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(match_type_widget, QComboBox)
        assert match_type_widget.currentText() == '*_contains'
        assert match_type_widget.property("objectstate_dirty") is True

        assert ObjectStateRegistry.time_travel_back()
        QApplication.processEvents()
        assert state.parameters["source_bindings.source_filters"] == (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.EQUALS,
                "DNA",
            ),
        )
        match_type_widget = widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(match_type_widget, QComboBox)
        assert match_type_widget.currentData() is SourceFilterMatchType.EQUALS
        assert match_type_widget.currentText() == '*_equals'
        assert match_type_widget.property("objectstate_dirty") is True
        assert "source_bindings.source_filters" in state.dirty_fields
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_inline_step_source_bindings_undo_one_of_two_cell_edits_keeps_owner_dirty() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    saved_filter = SourceFilterClause(
        SourceFilterSubject.FILE,
        SourceFilterMatchType.EQUALS,
        "DNA",
    )
    step = FunctionStep(
        func=lambda image: image,
        source_bindings=StepSourceBindingsConfig(
            source_filters=(saved_filter,),
        ),
    )
    state = ObjectState(step, scope_id="step")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="step",
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings" in manager.widgets:
                break

        container = manager.widgets["source_bindings"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)
        assert widget.source_filters_table is not None

        set_combo_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
            "contains",
        )
        QApplication.processEvents()
        match_type_widget = widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(match_type_widget, QComboBox)
        assert match_type_widget.currentText() == '*_contains'

        set_editable_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
            "RNA",
        )
        QApplication.processEvents()
        value_item = widget.source_filters_table.item(0, int(SourceFilterColumn.VALUE))
        assert value_item is not None
        assert value_item.text() == '*_RNA'
        assert "source_bindings.source_filters" in state.dirty_fields

        assert ObjectStateRegistry.time_travel_back()
        QApplication.processEvents()
        assert state.parameters["source_bindings.source_filters"] == (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            ),
        )

        match_type_widget = widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(match_type_widget, QComboBox)
        assert match_type_widget.currentData() is SourceFilterMatchType.CONTAINS
        assert match_type_widget.currentText() == '*_contains'
        assert match_type_widget.property("objectstate_dirty") is True

        value_item = widget.source_filters_table.item(0, int(SourceFilterColumn.VALUE))
        assert value_item is not None
        assert value_item.data(Qt.ItemDataRole.UserRole) == "DNA"
        assert value_item.text() == "_DNA"
        assert "source_bindings.source_filters" in state.dirty_fields
        assert widget.child_field_label("source_filters")._dirty_label_state.is_dirty
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_inline_source_bindings_edit_queues_child_field_flash() -> None:
    QtApplicationHarness.app()
    step = FunctionStep(func=lambda image: image)
    state = ObjectState(step)
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    for _ in range(20):
        QApplication.processEvents()
        widget = manager.findChild(SourceBindingsEditorWidget)
        if widget is not None:
            break
    else:
        widget = None
    assert widget is not None

    queued: list[str] = []
    manager.queue_flash_local_batch = queued.extend
    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    QApplication.processEvents()

    assert "source_bindings.bindings" in queued
    assert "source_bindings" not in queued


def test_inline_source_bindings_dropdown_edit_queues_child_section_flash() -> None:
    QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)

        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.EQUALS,
                "DNA",
            )
        )
        QApplication.processEvents()

        queued: list[str] = []
        registered = []
        manager.queue_flash_local_batch = queued.extend
        manager.register_flash_masked_container = (
            lambda key, container, mask_rects, *, label_widget=None, layout_watch_widgets=(): registered.append(
                (key, container, mask_rects, label_widget)
            )
        )
        assert widget.source_filters_table is not None
        set_combo_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
            "contains",
        )
        QApplication.processEvents()

        assert len(registered) == 1
        key, section, _, label_widget = registered[0]
        assert key == "source_bindings_config.source_filters[0].match_type"
        assert section is widget.child_field_section_group("source_filters")
        assert label_widget is widget.child_field_label("source_filters")
        assert "source_bindings_config.source_filters[0].match_type" in queued
        assert "source_bindings_config" not in queued
        assert state.parameters["source_bindings_config.source_filters"] == (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            ),
        )
    finally:
        manager.deleteLater()


def test_inline_source_bindings_provenance_navigation_masks_child_section() -> None:
    QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    class ScrollHarness(QWidget, ScrollableFormMixin):
        def __init__(self, form_manager: ParameterFormManager) -> None:
            super().__init__()
            self.form_manager = form_manager
            self.scroll_area = None

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)

        harness = ScrollHarness(manager)
        target = harness._resolve_scroll_target("source_bindings_config.source_filters")
        assert target is not None
        assert isinstance(
            target.structural_flash_target,
            StructuralMaskedContainerTarget,
        )
        assert isinstance(
            target.structural_flash_target.masked_target,
            StructuralDescendantMaskTarget,
        )
        assert target.target_widget is widget.child_field_label("source_filters")
        label_rect = target.structural_flash_target.scroll_rect_in(manager)
        label_widget = widget.child_field_label("source_filters")
        assert label_rect is not None
        label_window_pos = manager.mapFromGlobal(
            label_widget.mapToGlobal(label_widget.rect().topLeft())
        )
        assert label_rect.topLeft() == label_window_pos
        assert label_rect.size() == label_widget.size()

        class FakeScrollBar:
            def maximum(self) -> int:
                return 10000

        viewport = ScrollViewport(
            content_widget=manager,
            viewport_height=10000,
            viewport_top=0,
            viewport_bottom=10000,
            vertical_scroll_bar=FakeScrollBar(),
        )
        assert harness._target_is_fully_visible(target, viewport)

        queued: list[str] = []
        registered = []
        manager.queue_flash_local = queued.append
        manager.register_flash_masked_container = (
            lambda key, container, mask_rects, *, label_widget=None, layout_watch_widgets=(): registered.append(
                (key, container, mask_rects, label_widget)
            )
        )

        harness._flash_scroll_target(target)

        assert len(registered) == 1
        key, container_widget, mask_rects, label_widget = registered[0]
        assert key == "source_bindings_config.source_filters"
        assert container_widget is widget
        assert label_widget is widget.child_field_label("source_filters")
        masks = tuple(mask_rects(manager))
        section = widget.child_field_section_group("source_filters")
        section_window_pos = manager.mapFromGlobal(
            section.mapToGlobal(section.rect().topLeft())
        )
        assert widget.source_filters_table is not None
        table_window_pos = manager.mapFromGlobal(
            widget.source_filters_table.mapToGlobal(
                widget.source_filters_table.rect().topLeft()
            )
        )
        label_widget = widget.child_field_label("source_filters")
        label_window_pos = manager.mapFromGlobal(
            label_widget.mapToGlobal(label_widget.rect().topLeft())
        )
        section_rect = QRect(section_window_pos, section.size())
        label_rect = QRect(label_window_pos, label_widget.size())
        if section_rect != label_rect:
            assert (section_rect, False) not in masks
        assert (
            QRect(table_window_pos, widget.source_filters_table.size()),
            False,
        ) in masks
        assert (label_rect, False) in masks
        assert "source_bindings_config.source_filters" in queued
        assert "source_bindings_config" not in queued
    finally:
        manager.deleteLater()


def test_inline_source_bindings_initial_source_filter_cells_show_signature_diff() -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    step = FunctionStep(
        func=lambda image: image,
        source_bindings=LazyStepSourceBindingsConfig(
            source_filters=(
                SourceFilterClause(
                    SourceFilterSubject.EXTENSION,
                    SourceFilterMatchType.IS_IMAGE,
                ),
                SourceFilterClause(
                    SourceFilterSubject.DIRECTORY,
                    SourceFilterMatchType.DOES_NOT_CONTAIN_REGEX,
                    value=r"[\\/]\.",
                ),
            ),
        ),
    )
    state = ObjectState(step, scope_id="step")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
            scope_id="step",
        ),
    )

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings" in manager.widgets:
                break

        container = manager.widgets["source_bindings"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)
        assert widget.source_filters_table is not None
        assert "source_bindings.source_filters" in state.signature_diff_fields
        assert widget.child_field_label("source_filters")._label.font().underline()

        subject_widget = widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.SUBJECT),
        )
        assert isinstance(subject_widget, QComboBox)
        assert subject_widget.property("objectstate_signature_diff") is True
        assert subject_widget.font().underline()

        match_type_widget = widget.source_filters_table.cellWidget(
            1,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(match_type_widget, QComboBox)
        assert match_type_widget.property("objectstate_signature_diff") is True
        assert match_type_widget.font().underline()

        value_item = widget.source_filters_table.item(
            1,
            int(SourceFilterColumn.VALUE),
        )
        assert value_item is not None
        assert value_item.font().underline()
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_inline_source_bindings_structural_path_flash_targets_table_cell() -> None:
    app = QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )
    manager.resize(900, 700)
    manager.show()

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)
        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            )
        )
        app.processEvents()

        queued: list[str] = []
        registered = []
        manager.queue_flash_local = queued.append
        manager.register_flash_masked_container = (
            lambda key, container, mask_rects, *, label_widget=None, layout_watch_widgets=(): registered.append(
                (key, container, mask_rects, label_widget)
            )
        )

        manager._queue_leaf_flash_for_path(
            "source_bindings_config.source_filters[0].match_type"
        )

        assert len(registered) == 1
        key, container_widget, mask_rects, label_widget = registered[0]
        assert key == "source_bindings_config.source_filters[0].match_type"
        assert container_widget is widget.child_field_section_group("source_filters")
        assert label_widget is widget.child_field_label("source_filters")
        assert widget.source_filters_table is not None
        cell_rect = widget.source_filters_table.visualRect(
            widget.source_filters_table.model().index(
                0,
                int(SourceFilterColumn.MATCH_TYPE),
            )
        )
        cell_window_pos = manager.mapFromGlobal(
            widget.source_filters_table.viewport().mapToGlobal(cell_rect.topLeft())
        )
        label_widget = widget.child_field_label("source_filters")
        label_window_pos = manager.mapFromGlobal(
            label_widget.mapToGlobal(label_widget.rect().topLeft())
        )
        masks = tuple(mask_rects(manager))
        assert (QRect(cell_window_pos, cell_rect.size()), False) in masks
        assert (QRect(label_window_pos, label_widget.size()), False) in masks
        assert queued == ["source_bindings_config.source_filters[0].match_type"]
    finally:
        manager.deleteLater()


def test_inline_source_bindings_structural_provenance_navigation_flashes_table_cell() -> None:
    app = QtApplicationHarness.app()
    state = ObjectState(PipelineConfig())
    manager = ParameterFormManager(
        state,
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )
    manager.resize(900, 700)
    manager.show()

    class ScrollHarness(QWidget, ScrollableFormMixin):
        def __init__(self, form_manager: ParameterFormManager) -> None:
            super().__init__()
            self.form_manager = form_manager
            self.scroll_area = None

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)
        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            )
        )
        app.processEvents()

        harness = ScrollHarness(manager)
        target = harness._resolve_scroll_target(
            "source_bindings_config.source_filters[0].match_type"
        )
        assert target is not None
        assert isinstance(
            target.structural_flash_target,
            StructuralMaskedContainerTarget,
        )
        assert isinstance(
            target.structural_flash_target.masked_target,
            StructuralTableCellTarget,
        )
        viewport = ScrollViewport(
            content_widget=manager,
            viewport_height=120,
            viewport_top=0,
            viewport_bottom=120,
            vertical_scroll_bar=None,
        )
        target_bounds = harness._target_visual_bounds(target, viewport)
        assert target.structural_flash_target is not None
        target_rect = target.structural_flash_target.scroll_rect_in(manager)
        assert target_rect is not None
        assert target_bounds == (
            target_rect.y(),
            target_rect.height(),
            target_rect.y() + target_rect.height(),
        )
        assert widget.source_filters_table is not None
        cell_rect = widget.source_filters_table.visualRect(
            widget.source_filters_table.model().index(
                0,
                int(SourceFilterColumn.MATCH_TYPE),
            )
        )
        cell_window_pos = manager.mapFromGlobal(
            widget.source_filters_table.viewport().mapToGlobal(cell_rect.topLeft())
        )
        assert target_rect == QRect(cell_window_pos, cell_rect.size())
        assert target.target_widget is target.structural_flash_target.scroll_widget()
        table_top = widget.source_filters_table.mapTo(
            manager,
            widget.source_filters_table.rect().topLeft(),
        ).y()
        assert target_bounds[0] > table_top
        target_scroll = harness._target_scroll_position(target, viewport)
        expected_cell_scroll = max(
            0,
            target_rect.y() + target_rect.height() // 2 - viewport.viewport_height // 2,
        )
        table_center_scroll = max(
            0,
            table_top
            + widget.source_filters_table.height() // 2
            - viewport.viewport_height // 2,
        )
        assert target_scroll == expected_cell_scroll
        assert target_scroll != table_center_scroll

        queued: list[str] = []
        registered = []
        manager.queue_flash_local = queued.append
        manager.register_flash_masked_container = (
            lambda key, container, mask_rects, *, label_widget=None, layout_watch_widgets=(): registered.append(
                (key, container, mask_rects, label_widget)
            )
        )

        harness._flash_scroll_target(target)

        assert len(registered) == 1
        key, container_widget, _, label_widget = registered[0]
        assert key == "source_bindings_config.source_filters[0].match_type"
        assert container_widget is widget.child_field_section_group("source_filters")
        assert label_widget is widget.child_field_label("source_filters")
        assert queued == ["source_bindings_config.source_filters[0].match_type"]
    finally:
        manager.deleteLater()


def test_source_bindings_cell_flash_element_masks_cell_and_child_label() -> None:
    app = QtApplicationHarness.app()
    manager = ParameterFormManager(
        ObjectState(PipelineConfig()),
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )
    manager.resize(900, 700)
    manager.show()

    try:
        for _ in range(80):
            app.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)
        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            )
        )
        app.processEvents()

        flash_key = "source_bindings_config.source_filters[0].match_type"
        manager._queue_leaf_flash_for_path(flash_key)

        registrations = [
            registration
            for registration in manager._flash_registrations
            if registration[0] == flash_key
        ]
        assert len(registrations) == 1
        element_factory = registrations[0][1]
        element = element_factory(flash_key)
        assert element.get_child_rects is not None

        mask_rects = tuple(element.get_child_rects(manager))
        assert widget.source_filters_table is not None
        cell_rect = widget.source_filters_table.visualRect(
            widget.source_filters_table.model().index(
                0,
                int(SourceFilterColumn.MATCH_TYPE),
            )
        )
        cell_window_pos = manager.mapFromGlobal(
            widget.source_filters_table.viewport().mapToGlobal(cell_rect.topLeft())
        )
        expected_cell_rect = cell_rect.translated(
            cell_window_pos - cell_rect.topLeft()
        )
        label_widget = widget.child_field_label("source_filters")
        label_window_pos = manager.mapFromGlobal(
            label_widget.mapToGlobal(label_widget.rect().topLeft())
        )
        expected_label_rect = QRect(label_window_pos, label_widget.size())

        assert (expected_cell_rect, False) in mask_rects
        assert (expected_label_rect, False) in mask_rects
    finally:
        manager.deleteLater()


def test_source_bindings_child_section_flash_masks_changed_child_section() -> None:
    app = QtApplicationHarness.app()
    manager = ParameterFormManager(
        ObjectState(PipelineConfig()),
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )
    manager.resize(900, 700)
    manager.show()

    try:
        for _ in range(80):
            app.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)
        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            )
        )
        app.processEvents()

        queued: list[str] = []
        registered = []
        manager.queue_flash_local = queued.append
        manager.register_flash_masked_container = (
            lambda key, container, mask_rects, *, label_widget=None, layout_watch_widgets=(): registered.append(
                (key, container, mask_rects, label_widget)
            )
        )

        manager._queue_leaf_flash_for_path("source_bindings_config.source_filters")

        assert len(registered) == 1
        key, container_widget, mask_rects, label_widget = registered[0]
        assert key == "source_bindings_config.source_filters"
        assert container_widget is widget
        assert label_widget is widget.child_field_label("source_filters")
        section = widget.child_field_section_group("source_filters")
        section_window_pos = manager.mapFromGlobal(
            section.mapToGlobal(section.rect().topLeft())
        )
        label_widget = widget.child_field_label("source_filters")
        label_window_pos = manager.mapFromGlobal(
            label_widget.mapToGlobal(label_widget.rect().topLeft())
        )
        masks = tuple(mask_rects(manager))
        assert widget.source_filters_table is not None
        table_window_pos = manager.mapFromGlobal(
            widget.source_filters_table.mapToGlobal(
                widget.source_filters_table.rect().topLeft()
            )
        )
        section_rect = QRect(section_window_pos, section.size())
        label_rect = QRect(label_window_pos, label_widget.size())
        if section_rect != label_rect:
            assert (section_rect, False) not in masks
        assert (
            QRect(table_window_pos, widget.source_filters_table.size()),
            False,
        ) in masks
        assert (label_rect, False) in masks
        assert queued == ["source_bindings_config.source_filters"]

    finally:
        manager.deleteLater()


def test_source_bindings_owner_flash_masks_descendant_fields() -> None:
    app = QtApplicationHarness.app()
    manager = ParameterFormManager(
        ObjectState(PipelineConfig()),
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )
    manager.resize(900, 700)
    manager.show()

    try:
        for _ in range(80):
            app.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)
        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            )
        )
        app.processEvents()

        queued: list[str] = []
        registered = []
        groupbox_calls = []
        original_register_flash_groupbox = manager.register_flash_groupbox
        manager.queue_flash_local = queued.append

        def record_groupbox_flash(*args, **kwargs):
            groupbox_calls.append(args)
            return original_register_flash_groupbox(*args, **kwargs)

        manager.register_flash_groupbox = record_groupbox_flash
        manager.register_flash_masked_container = (
            lambda key, container, mask_rects, *, label_widget=None, layout_watch_widgets=(): registered.append(
                (key, container, mask_rects, label_widget)
            )
        )

        groupbox_call_count = len(groupbox_calls)
        manager._queue_leaf_flash_for_path("source_bindings_config")

        assert len(groupbox_calls) == groupbox_call_count
        assert len(registered) == 1
        key, container_widget, mask_rects, label_widget = registered[0]
        assert key == "source_bindings_config"
        assert container_widget is container
        assert label_widget is None
        masks = tuple(mask_rects(manager))
        assert widget.source_filters_table is not None
        table_window_pos = manager.mapFromGlobal(
            widget.source_filters_table.mapToGlobal(
                widget.source_filters_table.rect().topLeft()
            )
        )
        assert (
            QRect(table_window_pos, widget.source_filters_table.size()),
            False,
        ) in masks
        assert queued == ["source_bindings_config"]
    finally:
        manager.deleteLater()


def test_source_bindings_dropdown_time_travel_restores_widget_value() -> None:
    QtApplicationHarness.app()
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)

        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.EQUALS,
                "DNA",
            )
        )
        QApplication.processEvents()
        assert widget.source_filters_table is not None
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "equals"

        set_combo_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
            "contains",
        )
        QApplication.processEvents()
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "contains"
        match_type_widget = widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(match_type_widget, QComboBox)
        assert match_type_widget.currentData() is SourceFilterMatchType.CONTAINS
        assert match_type_widget.currentText() == '*_contains'
        assert match_type_widget.property("objectstate_dirty") is True
        assert (
            state.last_changed_field
            == "source_bindings_config.source_filters[0].match_type"
        )

        assert ObjectStateRegistry.time_travel_back()
        QApplication.processEvents()
        assert state.parameters["source_bindings_config.source_filters"] == (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.EQUALS,
                "DNA",
            ),
        )
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "equals"
        match_type_widget = widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(match_type_widget, QComboBox)
        assert match_type_widget.currentData() is SourceFilterMatchType.EQUALS
        assert match_type_widget.currentText() == '*_equals'
        assert match_type_widget.property("objectstate_dirty") is True
        assert (
            state.last_changed_field
            == "source_bindings_config.source_filters[0].match_type"
        )

        assert ObjectStateRegistry.time_travel_forward()
        QApplication.processEvents()
        assert state.parameters["source_bindings_config.source_filters"] == (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            ),
        )
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "contains"
        match_type_widget = widget.source_filters_table.cellWidget(
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        )
        assert isinstance(match_type_widget, QComboBox)
        assert match_type_widget.currentData() is SourceFilterMatchType.CONTAINS
        assert match_type_widget.currentText() == '*_contains'
        assert match_type_widget.property("objectstate_dirty") is True
        assert (
            state.last_changed_field
            == "source_bindings_config.source_filters[0].match_type"
        )
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_text_time_travel_restores_widget_value() -> None:
    QtApplicationHarness.app()
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)

        widget.add_source_filter_row(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            )
        )
        QApplication.processEvents()
        assert widget.source_filters_table is not None
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
        ) == "DNA"

        set_editable_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
            "RNA",
        )
        QApplication.processEvents()
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
        ) == "RNA"
        item = widget.source_filters_table.item(0, int(SourceFilterColumn.VALUE))
        assert item is not None
        assert item.data(Qt.ItemDataRole.UserRole) == "RNA"
        assert item.text() == '*_RNA'

        assert ObjectStateRegistry.time_travel_back()
        QApplication.processEvents()
        assert state.parameters["source_bindings_config.source_filters"] == (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            ),
        )
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
        ) == "DNA"
        item = widget.source_filters_table.item(0, int(SourceFilterColumn.VALUE))
        assert item is not None
        assert item.data(Qt.ItemDataRole.UserRole) == "DNA"
        assert item.text() == '*_DNA'

        assert ObjectStateRegistry.time_travel_forward()
        QApplication.processEvents()
        assert state.parameters["source_bindings_config.source_filters"] == (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "RNA",
            ),
        )
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
        ) == "RNA"
        item = widget.source_filters_table.item(0, int(SourceFilterColumn.VALUE))
        assert item is not None
        assert item.data(Qt.ItemDataRole.UserRole) == "RNA"
        assert item.text() == '*_RNA'
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_source_bindings_child_state_update_refreshes_widget_value() -> None:
    QtApplicationHarness.app()
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

    try:
        for _ in range(80):
            QApplication.processEvents()
            if "source_bindings_config" in manager.widgets:
                break

        container = manager.widgets["source_bindings_config"]
        assert isinstance(container, InlineDataclassGroupBox)
        widget = container._inline_value_widget
        assert isinstance(widget, SourceBindingsEditorWidget)

        first_filters = (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.EQUALS,
                "DNA",
            ),
        )
        state.update_parameter(
            "source_bindings_config.source_filters",
            first_filters,
        )
        loop = QEventLoop()
        QTimer.singleShot(0, loop.quit)
        loop.exec()
        QApplication.processEvents()

        source_config = state.parameters["source_bindings_config"]
        assert DataclassFieldAccess.raw_value(
            source_config,
            "source_filters",
        ) == first_filters
        assert widget.get_value().source_filters == first_filters
        assert widget.source_filters_table is not None
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "equals"
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
        ) == "DNA"
        history_len_before_second_update = len(ObjectStateRegistry.get_branch_history())

        second_filters = (
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "RNA",
            ),
        )
        state.update_parameter(
            "source_bindings_config.source_filters",
            second_filters,
        )
        loop = QEventLoop()
        QTimer.singleShot(0, loop.quit)
        loop.exec()
        QApplication.processEvents()

        source_config = state.parameters["source_bindings_config"]
        assert DataclassFieldAccess.raw_value(
            source_config,
            "source_filters",
        ) == second_filters
        assert widget.get_value().source_filters == second_filters
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "contains"
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
        ) == "RNA"

        new_history = ObjectStateRegistry.get_branch_history()[
            history_len_before_second_update:
        ]
        assert len(new_history) == 1
        assert "source_bindings_config.source_filters" in new_history[0].label

        assert ObjectStateRegistry.time_travel_back()
        QApplication.processEvents()
        assert state.parameters["source_bindings_config.source_filters"] == first_filters
        assert widget.get_value().source_filters == first_filters
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.MATCH_TYPE),
        ) == "equals"
        assert table_cell_text(
            widget.source_filters_table,
            0,
            int(SourceFilterColumn.VALUE),
        ) == "DNA"
    finally:
        manager.deleteLater()
        ObjectStateRegistry.clear()


def test_inline_source_bindings_uses_dataclass_groupbox_chrome() -> None:
    QtApplicationHarness.app()
    step = FunctionStep(func=lambda image: image)
    manager = ParameterFormManager(
        ObjectState(step),
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    for _ in range(20):
        QApplication.processEvents()
        if "source_bindings" in manager.widgets:
            break

    container = manager.widgets["source_bindings"]
    assert isinstance(container, InlineDataclassGroupBox)
    assert container._inline_value_widget is not None
    assert container._help_button is not None
    assert container._flash_key == "source_bindings"

    queued: list[str] = []
    manager.queue_flash_local = queued.append
    manager._queue_leaf_flash_for_path("source_bindings")

    assert queued == ["source_bindings"]


def test_nested_form_flash_delegates_to_root_manager() -> None:
    QtApplicationHarness.app()
    step = FunctionStep(func=lambda image: image)
    manager = ParameterFormManager(
        ObjectState(step),
        FormManagerConfig(
            color_scheme=ColorScheme(),
            use_scroll_area=False,
        ),
    )

    for _ in range(20):
        QApplication.processEvents()
        if "processing_config" in manager.nested_managers:
            break

    nested_manager = manager.nested_managers["processing_config"]
    queued: list[str] = []

    def record_flash(path: str, *, queue_flash: bool = True) -> str:
        queued.append(path)
        return path

    manager._queue_leaf_flash_for_path = record_flash

    nested_manager.queue_field_flash("processing_config.group_by")
    nested_manager._queue_leaf_flash_for_path("processing_config.input_source")

    assert queued == [
        "processing_config.group_by",
        "processing_config.input_source",
    ]


def test_source_bindings_flash_masks_nested_section_titles() -> None:
    app = QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA"),),
        )
    )
    container = InlineDataclassGroupBox(
        title="Source Bindings",
        help_target=StepSourceBindingsConfig,
        color_scheme=ColorScheme(),
        flash_key="source_bindings",
    )
    container.content_layout.addWidget(widget)
    container.resize(900, 700)
    container.show()

    for _ in range(5):
        app.processEvents()

    element = create_groupbox_element("source_bindings", container)
    assert element.get_child_rects is not None
    child_rects = element.get_child_rects(container)
    section_title_labels = [
        label
        for label in widget.findChildren(QLabel)
        if label.text() in {
            "Bindings",
            "Source Filters",
            "Metadata Rules",
            "Source Set Pairing",
        }
    ]

    title_rects = [
        rect
        for rect, _ in child_rects
        if any(
            label.mapTo(container, label.rect().topLeft()).y()
            <= rect.y()
            <= label.mapTo(container, label.rect().topLeft()).y()
            + label.fontMetrics().height()
            + 4
            for label in section_title_labels
        )
    ]

    assert len(section_title_labels) == 4
    assert len(title_rects) >= len(section_title_labels)


def test_source_bindings_editor_round_trips_form_value() -> None:
    QtApplicationHarness.app()
    binding_config = StepSourceBindingsConfig(
        bindings=(NamedSourceBinding(alias="DNA"),),
    )
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())
    changed_count = 0

    def record_change() -> None:
        nonlocal changed_count
        changed_count += 1

    widget.changed.connect(record_change)

    widget.set_value(binding_config)

    assert widget.get_value() == binding_config
    assert widget.value == binding_config
    assert changed_count == 0


def test_source_bindings_editor_set_value_resets_stale_preview_table() -> None:
    QtApplicationHarness.app()
    raw_config = StepSourceBindingsConfig(
        source_filters=(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            ),
        ),
    )
    preview_config = StepSourceBindingsConfig(
        source_filters=(
            SourceFilterClause(
                SourceFilterSubject.EXTENSION,
                SourceFilterMatchType.CONTAINS,
                ".tif",
            ),
        ),
    )
    widget = SourceBindingsEditorWidget.from_bindings(raw_config)
    assert widget.source_filters_table is not None

    widget.set_resolved_value_preview(preview_config)
    assert table_cell_text(
        widget.source_filters_table,
        0,
        int(SourceFilterColumn.SUBJECT),
    ) == "extension"

    widget.set_value(raw_config)

    assert table_cell_text(
        widget.source_filters_table,
        0,
        int(SourceFilterColumn.SUBJECT),
    ) == "file"


def test_source_bindings_editor_renders_preview_context(tmp_path) -> None:
    QtApplicationHarness.app()
    source_path = tmp_path / "A01_DNA.tif"
    source_path.write_text("placeholder", encoding="utf-8")
    source_bindings = SourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                origin=SourceBindingOrigin.PIPELINE_START,
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "DNA",
                        ),
                    ),
                ),
            ),
        ),
    )
    inventory = SourceInventory.from_paths(
        (source_path,),
        source_root=tmp_path,
        source_backend=Backend.DISK,
        source_bindings=source_bindings,
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(),
        source_bindings=source_bindings,
    )

    widget.set_preview_context(
        source_bindings=source_bindings,
        inventory=inventory,
    )

    section_titles = {label.text() for label in widget.findChildren(QLabel)}
    assert "Preview Matches" in section_titles
    assert "Source Sets" in section_titles


def test_source_bindings_editor_edits_step_binding_table() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    set_binding_cell_text(table, 0, SourceBindingColumn.ALIAS, "OrigDNA")
    widget._apply_step_bindings(dialog.bindings())

    edited = widget.get_value()
    assert edited.bindings[0].alias == "OrigDNA"


def test_source_bindings_editor_preserves_selector_on_basic_edits() -> None:
    QtApplicationHarness.app()
    selector = SourceSelector(
        components=(ComponentSelector(AllComponents.CHANNEL, "DNA"),),
        metadata=(MetadataSelector("Well", "A01"),),
        filters=(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            ),
        ),
        inherit_current_scope=False,
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA", selector=selector),),
        )
    )
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    set_binding_cell_text(table, 0, SourceBindingColumn.ALIAS, "OrigDNA")
    widget._apply_step_bindings(dialog.bindings())

    edited = widget.get_value().bindings[0]
    assert edited.alias == "OrigDNA"
    assert edited.selector == selector


def test_source_bindings_editor_preserves_binding_identity_on_basic_edits() -> None:
    QtApplicationHarness.app()
    binding = NamedSourceBinding(
        alias="DNA",
        component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
        projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(bindings=(binding,))
    )
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    set_binding_cell_text(table, 0, SourceBindingColumn.ALIAS, "OrigDNA")
    widget._apply_step_bindings(dialog.bindings())

    edited = widget.get_value().bindings[0]
    assert edited.alias == "OrigDNA"
    assert edited.component_identity == (
        ComponentSelector(AllComponents.CHANNEL, "1"),
    )
    assert edited.projection_role is SourceProjectionRole.SOURCE_ARTIFACT


def test_source_bindings_editor_edits_binding_identity_columns() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    set_editable_cell_text(
        table,
        *binding_cell_position(0, SourceBindingColumn.IDENTITY),
        "channel=2;site=1",
    )
    set_binding_cell_text(
        table,
        0,
        SourceBindingColumn.SET_ROLE,
        SourceSetRole.BROADCAST.value,
    )
    set_binding_cell_text(
        table,
        0,
        SourceBindingColumn.PROJECTION_ROLE,
        SourceProjectionRole.SOURCE_ARTIFACT.value,
    )
    widget._apply_step_bindings(dialog.bindings())

    binding = widget.get_value().bindings[0]
    assert binding.component_identity == (
        ComponentSelector(AllComponents.CHANNEL, "2"),
        ComponentSelector(AllComponents.SITE, "1"),
    )
    assert binding.source_set_role is SourceSetRole.BROADCAST
    assert binding.projection_role is SourceProjectionRole.SOURCE_ARTIFACT


def test_source_bindings_editor_edits_selector_columns() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    set_editable_cell_text(
        table,
        *binding_cell_position(0, SourceBindingColumn.COMPONENTS),
        "channel=DNA;site=1",
    )
    set_editable_cell_text(
        table,
        *binding_cell_position(0, SourceBindingColumn.METADATA),
        "Well=A01",
    )
    set_editable_cell_text(
        table,
        *binding_cell_position(0, SourceBindingColumn.FILTERS),
        "file:contains:DNA",
    )
    set_binding_cell_text(table, 0, SourceBindingColumn.INHERIT, "False")
    widget._apply_step_bindings(dialog.bindings())

    selector = widget.get_value().bindings[0].selector
    assert selector.components == (
        ComponentSelector(AllComponents.CHANNEL, "DNA"),
        ComponentSelector(AllComponents.SITE, "1"),
    )
    assert selector.metadata == (MetadataSelector("Well", "A01"),)
    assert selector.filters == (
        SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "DNA",
        ),
    )
    assert selector.inherit_current_scope is False


def test_source_bindings_editor_uses_free_form_selector_pickers(tmp_path) -> None:
    QtApplicationHarness.app()
    source_path = tmp_path / "A01_DNA.tif"
    source_path.write_text("placeholder", encoding="utf-8")
    source_bindings = SourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                origin=SourceBindingOrigin.PIPELINE_START,
                selector=SourceSelector(),
            ),
        ),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"(?P<Well>A\d{2})_(?P<Channel>DNA)\.tif",
            ),
        ),
    )
    inventory = SourceInventory.from_paths(
        (source_path,),
        source_root=tmp_path,
        source_backend=Backend.DISK,
        source_bindings=source_bindings,
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(),
        source_bindings=source_bindings,
        inventory=inventory,
    )

    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    components_widget = binding_cell_widget(table, 0, SourceBindingColumn.COMPONENTS)
    metadata_widget = binding_cell_widget(table, 0, SourceBindingColumn.METADATA)

    assert isinstance(components_widget, StructuredSelectorCellWidget)
    assert isinstance(metadata_widget, StructuredSelectorCellWidget)
    assert "channel=" in components_widget.values
    assert "Well=A01" in metadata_widget.values


def test_source_bindings_editor_removes_selected_binding_row() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(
            bindings=(
                NamedSourceBinding(alias="DNA"),
                NamedSourceBinding(alias="GFP"),
            ),
        )
    )
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    table.selectColumn(0)

    dialog.editor.remove_selected_binding_rows()
    widget._apply_step_bindings(dialog.bindings())

    remaining_aliases = tuple(
        binding.alias
        for binding in widget.get_value().bindings
    )
    assert remaining_aliases == ("GFP",)


def test_source_bindings_editor_edits_metadata_rules() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    widget.add_metadata_rule_row()
    assert widget.metadata_rules_table is not None
    set_combo_cell_text(
        widget.metadata_rules_table,
        0,
        int(MetadataRuleColumn.SOURCE),
        "file_name",
    )
    widget.metadata_rules_table.item(
        0,
        int(MetadataRuleColumn.PATTERN),
    ).setText(r"(?P<well>A\d{2})_(?P<channel>DNA)\.tif")
    set_editable_cell_text(
        widget.metadata_rules_table,
        0,
        int(MetadataRuleColumn.FILTERS),
        "file:contains:DNA",
    )

    rules = widget.get_value().metadata_rules
    assert rules == (
        MetadataExtractionRule(
            source=MetadataSource.FILE_NAME,
            pattern=r"(?P<well>A\d{2})_(?P<channel>DNA)\.tif",
            filters=(
                SourceFilterClause(
                    SourceFilterSubject.FILE,
                    SourceFilterMatchType.CONTAINS,
                    "DNA",
                ),
            ),
        ),
    )


def test_source_bindings_editor_edits_match_plan() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    widget.add_match_plan_row()
    assert widget.match_plan_table is not None
    set_combo_cell_text(
        widget.match_plan_table,
        0,
        int(MatchPlanColumn.METHOD),
        "metadata",
    )
    set_editable_cell_text(
        widget.match_plan_table,
        0,
        int(MatchPlanColumn.FIELDS),
        "DNA=well;GFP=well",
    )

    assert widget.get_value().match_plan == SourceBindingMatchPlan(
        method=SourceBindingMatchMethod.METADATA,
        dimensions=(
            SourceBindingMatchDimension(
                fields=(
                    SourceBindingMatchField("DNA", "well"),
                    SourceBindingMatchField("GFP", "well"),
                ),
            ),
        ),
    )


def test_source_bindings_editor_enum_columns_use_typed_combos() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    set_combo_cell_text(
        table,
        *binding_cell_position(0, SourceBindingColumn.KIND),
        "object_labels",
    )
    set_combo_cell_text(
        table,
        *binding_cell_position(0, SourceBindingColumn.ORIGIN),
        "pipeline_start",
    )
    set_combo_cell_text(
        table,
        *binding_cell_position(0, SourceBindingColumn.PROJECTION_ROLE),
        "source_artifact",
    )
    widget._apply_step_bindings(dialog.bindings())

    binding = widget.get_value().bindings[0]
    assert binding.artifact_kind.value == "object_labels"
    assert binding.origin is SourceBindingOrigin.PIPELINE_START
    assert binding.projection_role is SourceProjectionRole.SOURCE_ARTIFACT
