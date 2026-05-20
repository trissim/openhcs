from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QComboBox, QGroupBox, QPushButton

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
    GroupedSourceBindings,
    NamedSourceBinding,
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
    StructuredSelectorCellWidget,
    StructuredSelectorDialog,
)
from openhcs.config_framework.object_state import ObjectState
from pyqt_reactive.forms import (
    FormManagerConfig,
    InlineDataclassWidgetInfo,
    ParameterFormManager,
    create_parameter_info,
)
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.animation.flash_mixin import create_groupbox_element
from pyqt_reactive.widgets.shared.clickable_help_components import InlineDataclassGroupBox


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


def test_source_bindings_editor_builds_from_bindings() -> None:
    QtApplicationHarness.app()

    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    assert widget.layout.count() > 0


def test_source_bindings_editor_uses_compact_inline_step_binding_summary() -> None:
    QtApplicationHarness.app()

    widget = SourceBindingsEditorWidget.from_bindings(
        StepSourceBindingsConfig(
            groups=(
                GroupedSourceBindings(
                    bindings=(NamedSourceBinding(alias="DNA"),),
                ),
            ),
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
            groups=(
                GroupedSourceBindings(
                    bindings=(NamedSourceBinding(alias="DNA"),),
                ),
            ),
        )
    )
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table

    assert table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert table.height() >= table.horizontalHeader().height() + table.rowHeight(0)


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
    assert edited.groups[0].bindings[0].alias == "DNA"


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
            groups=(
                GroupedSourceBindings(
                    bindings=(NamedSourceBinding(alias="DNA"),),
                ),
            ),
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
    titled_sections = [section for section in widget.findChildren(QGroupBox) if section.title()]

    title_rects = [
        rect
        for rect, _ in child_rects
        if any(
            section.mapTo(container, section.rect().topLeft()).y()
            <= rect.y()
            <= section.mapTo(container, section.rect().topLeft()).y()
            + section.fontMetrics().height()
            + 4
            for section in titled_sections
        )
    ]

    assert len(title_rects) >= len(titled_sections)


def test_source_bindings_editor_round_trips_form_value() -> None:
    QtApplicationHarness.app()
    binding_config = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                bindings=(NamedSourceBinding(alias="DNA"),),
            ),
        ),
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
    assert changed_count == 1


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

    group_titles = [
        widget.layout.itemAt(index).widget().title()
        for index in range(widget.layout.count())
        if hasattr(widget.layout.itemAt(index).widget(), "title")
    ]
    assert "Preview Matches" in group_titles
    assert "Image Sets" in group_titles


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
    widget._apply_step_binding_groups(dialog.groups())

    edited = widget.get_value()
    assert edited.groups[0].bindings[0].alias == "OrigDNA"


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
            groups=(
                GroupedSourceBindings(
                    bindings=(NamedSourceBinding(alias="DNA", selector=selector),),
                ),
            ),
        )
    )
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    table.item(0, int(SourceBindingColumn.ALIAS)).setText("OrigDNA")
    widget._apply_step_binding_groups(dialog.groups())

    edited = widget.get_value().groups[0].bindings[0]
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
    widget._apply_step_binding_groups(dialog.groups())

    selector = widget.get_value().groups[0].bindings[0].selector
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
            groups=(
                GroupedSourceBindings(
                    bindings=(
                        NamedSourceBinding(alias="DNA"),
                        NamedSourceBinding(alias="GFP"),
                    ),
                ),
            ),
        )
    )
    dialog = widget._create_step_bindings_dialog()
    table = dialog.editor.table
    table.selectRow(0)

    dialog.editor.remove_selected_binding_rows()
    widget._apply_step_binding_groups(dialog.groups())

    remaining_aliases = tuple(
        binding.alias
        for group in widget.get_value().groups
        for binding in group.bindings
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
    widget._apply_step_binding_groups(dialog.groups())

    binding = widget.get_value().groups[0].bindings[0]
    assert binding.artifact_kind.value == "object_labels"
    assert binding.origin is SourceBindingOrigin.PIPELINE_START
