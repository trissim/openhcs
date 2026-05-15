from __future__ import annotations

from PyQt6.QtWidgets import QApplication, QComboBox

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
from pyqt_reactive.forms import InlineDataclassWidgetInfo, create_parameter_info


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
    assert widget.step_bindings_table is not None
    widget.step_bindings_table.item(
        0,
        int(SourceBindingColumn.ALIAS),
    ).setText("OrigDNA")

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
    assert widget.step_bindings_table is not None

    widget.step_bindings_table.item(0, int(SourceBindingColumn.ALIAS)).setText("OrigDNA")

    edited = widget.get_value().groups[0].bindings[0]
    assert edited.alias == "OrigDNA"
    assert edited.selector == selector


def test_source_bindings_editor_edits_selector_columns() -> None:
    QtApplicationHarness.app()
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())

    widget.add_binding_row(NamedSourceBinding(alias="DNA"))
    assert widget.step_bindings_table is not None
    set_editable_cell_text(
        widget.step_bindings_table,
        0,
        int(SourceBindingColumn.COMPONENTS),
        "channel=DNA;site=1",
    )
    set_editable_cell_text(
        widget.step_bindings_table,
        0,
        int(SourceBindingColumn.METADATA),
        "Well=A01",
    )
    set_editable_cell_text(
        widget.step_bindings_table,
        0,
        int(SourceBindingColumn.FILTERS),
        "file:contains:DNA",
    )
    widget.step_bindings_table.item(
        0,
        int(SourceBindingColumn.INHERIT),
    ).setText("False")

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
    assert widget.step_bindings_table is not None
    components_widget = widget.step_bindings_table.cellWidget(
        0,
        int(SourceBindingColumn.COMPONENTS),
    )
    metadata_widget = widget.step_bindings_table.cellWidget(
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
    assert widget.step_bindings_table is not None
    widget.step_bindings_table.selectRow(0)

    widget.remove_selected_binding_rows()

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
    assert widget.step_bindings_table is not None
    set_combo_cell_text(
        widget.step_bindings_table,
        0,
        int(SourceBindingColumn.KIND),
        "object_labels",
    )
    set_combo_cell_text(
        widget.step_bindings_table,
        0,
        int(SourceBindingColumn.ORIGIN),
        "pipeline_start",
    )

    binding = widget.get_value().groups[0].bindings[0]
    assert binding.artifact_kind.value == "object_labels"
    assert binding.origin is SourceBindingOrigin.PIPELINE_START
