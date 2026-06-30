from __future__ import annotations

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QColor, QEnterEvent
from PyQt6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton
from python_introspect import is_enableable

from openhcs.core.pipeline_image_schema import ImageAssignment, PipelineImageSchema
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
from openhcs.constants.constants import AllComponents
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.source_bindings_view import SourceInventory
from openhcs.pyqt_gui.widgets.source_bindings_editor import (
    MatchPlanColumn,
    MetadataRuleColumn,
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
from pyqt_reactive.forms.widget_strategies import PyQt6WidgetEnhancer
from pyqt_reactive.forms.parameter_info_types import (
    InlineDataclassWidgetInfo,
    create_parameter_info,
)
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.animation.flash_mixin import create_groupbox_element
from pyqt_reactive.widgets.shared.clickable_help_components import HelpButton
from pyqt_reactive.widgets.shared.clickable_help_components import HelpContext
from pyqt_reactive.widgets.shared.clickable_help_components import HelpIndicator
from pyqt_reactive.widgets.shared.clickable_help_components import InlineDataclassGroupBox
from pyqt_reactive.widgets.shared.clickable_help_components import ProvenanceLabel
from pyqt_reactive.widgets.no_scroll_spinbox import NoneAwareCheckBox
from pyqt_reactive.widgets.shared.scoped_table_widget import ScopedTableWidget
from pyqt_reactive.widgets.shared.scope_color_utils import get_scope_color_scheme
from pyqt_reactive.widgets.shared.scrollable_form_mixin import ScrollableFormMixin


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
        return widget.currentText().strip()
    item = table.item(row, column)
    return "" if item is None else item.text().strip()


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
        manager.queue_flash_local = queued_flashes.append
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
        assert "step_source_bindings_config" in queued_flashes
    finally:
        manager.deleteLater()
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

        step_container._title_label.mousePressEvent(None)
        for _ in range(10):
            QApplication.processEvents()

        assert state.parameters["step_source_bindings_config.enabled"] is True
        assert step_checkboxes[0].isChecked()
        assert step_widget.graphicsEffect() is None
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
        assert step_widget.section_groups["bindings"].graphicsEffect() is not None
        assert step_widget.section_groups["match_plan"].graphicsEffect() is None

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


def test_source_bindings_child_chrome_and_reset_use_flat_state_paths() -> None:
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

        source_container = manager.widgets["source_bindings_config"]
        assert isinstance(source_container, InlineDataclassGroupBox)
        source_widget = source_container._inline_value_widget
        assert isinstance(source_widget, SourceBindingsEditorWidget)

        source_widget.add_binding_row(NamedSourceBinding(alias="DNA"))
        for _ in range(10):
            QApplication.processEvents()

        bindings_label = source_widget.section_labels["bindings"]
        reset_button = source_widget.section_reset_buttons["bindings"]
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
        assert not source_widget.section_labels["bindings"]._dirty_label_state.is_dirty
        assert not source_widget.section_reset_buttons["bindings"].font().underline()
    finally:
        manager.deleteLater()


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
        assert target.target_widget is source_widget.section_groups["bindings"]
        assert not target.is_field
    finally:
        manager.deleteLater()


def test_source_bindings_editor_uses_resolved_placeholder_tables() -> None:
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
    dialog = widget._create_step_bindings_dialog()

    assert tuple(binding.alias for binding in dialog.bindings()) == ("InheritedDNA",)
    assert widget.metadata_rules_table is not None
    assert widget.metadata_rules_table.rowCount() == 1
    assert DataclassFieldAccess.raw_value(widget.get_value(), "bindings") is None
    assert DataclassFieldAccess.raw_value(widget.get_value(), "metadata_rules") is None


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
    assert table.verticalHeader().isHidden()
    assert table.height() >= table.horizontalHeader().height() + table.rowHeight(0)


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


def test_inline_source_bindings_edit_queues_groupbox_flash() -> None:
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
    manager.queue_flash_local = queued.append
    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    QApplication.processEvents()

    assert "source_bindings" in queued


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
    manager._queue_leaf_flash_for_path = queued.append

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
        if label.text() in {"Bindings", "Source Filters", "Metadata Rules", "Match Plan"}
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


def test_source_bindings_editor_renders_preview_context(tmp_path) -> None:
    QtApplicationHarness.app()
    source_path = tmp_path / "A01_DNA.tif"
    source_path.write_text("placeholder", encoding="utf-8")
    schema = PipelineImageSchema(
        assignments_by_alias={
            "DNA": ImageAssignment(
                alias="DNA",
                image_type="Grayscale image",
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
        },
    )
    inventory = SourceInventory.from_paths(
        (source_path,),
        schema=schema,
        source_root=tmp_path,
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(),
        schema=schema,
    )

    widget.set_preview_context(schema=schema, inventory=inventory)

    section_titles = {label.text() for label in widget.findChildren(QLabel)}
    assert "Preview Matches" in section_titles
    assert "Image Sets" in section_titles


def test_source_bindings_editor_edits_step_binding_table() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    table.item(
        0,
        int(SourceBindingColumn.ALIAS),
    ).setText("OrigDNA")
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
    table.item(0, int(SourceBindingColumn.ALIAS)).setText("OrigDNA")
    widget._apply_step_bindings(dialog.bindings())

    edited = widget.get_value().bindings[0]
    assert edited.alias == "OrigDNA"
    assert edited.selector == selector


def test_source_bindings_editor_edits_selector_columns() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    set_editable_cell_text(
        table,
        0,
        int(SourceBindingColumn.COMPONENTS),
        "channel=DNA;site=1",
    )
    set_editable_cell_text(
        table,
        0,
        int(SourceBindingColumn.METADATA),
        "Well=A01",
    )
    set_editable_cell_text(
        table,
        0,
        int(SourceBindingColumn.FILTERS),
        "file:contains:DNA",
    )
    table.item(
        0,
        int(SourceBindingColumn.INHERIT),
    ).setText("False")
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
    schema = PipelineImageSchema(
        assignments_by_alias={
            "DNA": ImageAssignment(
                alias="DNA",
                image_type="Grayscale image",
                origin=SourceBindingOrigin.PIPELINE_START,
                selector=SourceSelector(),
            ),
        },
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"(?P<Well>A\d{2})_(?P<Channel>DNA)\.tif",
            ),
        ),
    )
    inventory = SourceInventory.from_paths(
        (source_path,),
        schema=schema,
        source_root=tmp_path,
    )
    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(),
        schema=schema,
        inventory=inventory,
    )

    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    components_widget = table.cellWidget(
        0,
        int(SourceBindingColumn.COMPONENTS),
    )
    metadata_widget = table.cellWidget(
        0,
        int(SourceBindingColumn.METADATA),
    )

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
    table.selectRow(0)

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
        0,
        int(SourceBindingColumn.KIND),
        "object_labels",
    )
    set_combo_cell_text(
        table,
        0,
        int(SourceBindingColumn.ORIGIN),
        "pipeline_start",
    )
    widget._apply_step_bindings(dialog.bindings())

    binding = widget.get_value().bindings[0]
    assert binding.artifact_kind.value == "object_labels"
    assert binding.origin is SourceBindingOrigin.PIPELINE_START
