"""PyQt source-bindings editor over the typed source-binding view model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Callable, Generic, Mapping, TypeVar

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyqt_reactive.protocols import (
    ChangeSignalEmitter,
    PyQtWidgetMeta,
    ValueGettable,
    ValueSettable,
)
from pyqt_reactive.widgets.shared.scoped_table_widget import ScopedTableWidget
from pyqt_reactive.widgets.shared.scope_color_receiver import ScopeColorSchemeReceiver

from openhcs.constants.constants import AllComponents
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.source_bindings import (
    ComponentSelector,
    GroupedSourceBindings,
    MetadataExtractionRule,
    MetadataSource,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.source_bindings_view import (
    SourceBindingsPreview,
    SourceBindingsViewModel,
    SourceInventory,
)

EMPTY_PIPELINE_IMAGE_SCHEMA = PipelineImageSchema.empty()
EditableRowT = TypeVar("EditableRowT")


class SourceBindingColumn(IntEnum):
    """Editable table columns for one named source binding."""

    GROUP = 0
    ALIAS = 1
    KIND = 2
    ORIGIN = 3
    REQUIRED = 4
    COMPONENTS = 5
    METADATA = 6
    FILTERS = 7
    INHERIT = 8


class MetadataRuleColumn(IntEnum):
    """Editable table columns for one metadata extraction rule."""

    SOURCE = 0
    PATTERN = 1
    FILTERS = 2


class MatchPlanColumn(IntEnum):
    """Editable table columns for one match-plan dimension."""

    METHOD = 0
    FIELDS = 1


@dataclass(frozen=True, slots=True)
class EnumCellSpec:
    """Typed enum editor specification for one editable table column."""

    enum_type: type[Enum]

    @property
    def values(self) -> tuple[Enum, ...]:
        return tuple(self.enum_type)

    def text_for_value(self, value: Enum) -> str:
        return str(value.value)


class FreeFormCellEditorKind(Enum):
    """Semantic dialog family for structured free-form source-binding cells."""

    SELECTOR_LIST = "selector_list"
    COMPONENT_SELECTORS = "component_selectors"
    METADATA_SELECTORS = "metadata_selectors"
    FILTER_CLAUSES = "filter_clauses"
    MATCH_DIMENSIONS = "match_dimensions"


SelectorDialogRowParser = Callable[[str], tuple[str | None, ...]]
SelectorDialogRowFormatter = Callable[[tuple[str, ...]], str]


def parse_key_value_dialog_row(item: str) -> tuple[str, str]:
    return SelectorListCodec.key_value_parts(item)


def parse_filter_dialog_row(item: str) -> tuple[str, str, str | None]:
    return SelectorListCodec.filter_parts(item)


def format_key_value_dialog_row(values: tuple[str, ...]) -> str:
    key, value = values
    if not key or not value:
        return ""
    return SelectorListCodec.KEY_VALUE_SEPARATOR.join((key, value))


def format_filter_dialog_row(values: tuple[str, ...]) -> str:
    subject, match_type, value = values
    if not subject or not match_type:
        return ""
    return SelectorListCodec.FILTER_SEPARATOR.join((subject, match_type, value))


@dataclass(frozen=True, slots=True)
class StructuredSelectorEditorSpec:
    """Authoritative behavior for one structured source-binding cell editor."""

    editor_kind: FreeFormCellEditorKind
    columns: tuple[str, ...]
    hint: str
    row_parser: SelectorDialogRowParser
    row_formatter: SelectorDialogRowFormatter
    column_options: Mapping[int, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )


STRUCTURED_SELECTOR_EDITOR_SPECS: Mapping[
    FreeFormCellEditorKind,
    StructuredSelectorEditorSpec,
] = MappingProxyType(
    {
        FreeFormCellEditorKind.SELECTOR_LIST: StructuredSelectorEditorSpec(
            editor_kind=FreeFormCellEditorKind.SELECTOR_LIST,
            columns=("Key", "Value"),
            hint="Use key=value entries separated by semicolons.",
            row_parser=parse_key_value_dialog_row,
            row_formatter=format_key_value_dialog_row,
        ),
        FreeFormCellEditorKind.COMPONENT_SELECTORS: StructuredSelectorEditorSpec(
            editor_kind=FreeFormCellEditorKind.COMPONENT_SELECTORS,
            columns=("Component", "Value"),
            hint="Use key=value entries separated by semicolons.",
            row_parser=parse_key_value_dialog_row,
            row_formatter=format_key_value_dialog_row,
            column_options=MappingProxyType(
                {0: tuple(component.value for component in AllComponents)}
            ),
        ),
        FreeFormCellEditorKind.METADATA_SELECTORS: StructuredSelectorEditorSpec(
            editor_kind=FreeFormCellEditorKind.METADATA_SELECTORS,
            columns=("Metadata field", "Value"),
            hint="Use key=value entries separated by semicolons.",
            row_parser=parse_key_value_dialog_row,
            row_formatter=format_key_value_dialog_row,
        ),
        FreeFormCellEditorKind.FILTER_CLAUSES: StructuredSelectorEditorSpec(
            editor_kind=FreeFormCellEditorKind.FILTER_CLAUSES,
            columns=("Subject", "Match type", "Value"),
            hint="Use subject:match_type:value entries separated by semicolons.",
            row_parser=parse_filter_dialog_row,
            row_formatter=format_filter_dialog_row,
            column_options=MappingProxyType(
                {
                    0: tuple(subject.value for subject in SourceFilterSubject),
                    1: tuple(match_type.value for match_type in SourceFilterMatchType),
                }
            ),
        ),
        FreeFormCellEditorKind.MATCH_DIMENSIONS: StructuredSelectorEditorSpec(
            editor_kind=FreeFormCellEditorKind.MATCH_DIMENSIONS,
            columns=("Alias", "Metadata field"),
            hint="Use alias=metadata_field entries separated by semicolons.",
            row_parser=parse_key_value_dialog_row,
            row_formatter=format_key_value_dialog_row,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class FreeFormCellSpec:
    """Editable suggestions and semantic dialog type for selector cells."""

    values: tuple[str, ...]
    editor_kind: FreeFormCellEditorKind = FreeFormCellEditorKind.SELECTOR_LIST


EnumCellSpecMap = Mapping[tuple[type[IntEnum], IntEnum], EnumCellSpec]
FreeFormCellSpecMap = Mapping[tuple[type[IntEnum], IntEnum], FreeFormCellSpec]


class StructuredSelectorCellWidget(QWidget):
    """Mini-editor for structured selector cells with suggestions and free text."""

    def __init__(
        self,
        *,
        values: tuple[str, ...],
        value: str,
        editor_kind: FreeFormCellEditorKind,
        apply_changes: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.values = values
        self.editor_kind = editor_kind
        self._apply_changes = apply_changes
        self.line_edit = QLineEdit(value, self)
        self.line_edit.editingFinished.connect(self._apply_changes)
        picker_button = QPushButton("...", self)
        picker_button.setFixedWidth(28)
        picker_button.clicked.connect(self._open_picker)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(picker_button)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def set_text(self, value: str) -> None:
        self.line_edit.setText(value)
        self._apply_changes()

    def _open_picker(self) -> None:
        dialog = StructuredSelectorDialog(
            editor_kind=self.editor_kind,
            suggestions=self.values,
            value=self.text(),
            parent=self,
        )
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        if not accepted:
            return
        self.set_text(dialog.value())


class StructuredSelectorDialog(QDialog):
    """Semantic source-binding picker for selector/filter/match list cells."""

    TITLES = {
        FreeFormCellEditorKind.SELECTOR_LIST: "Edit selector list",
        FreeFormCellEditorKind.COMPONENT_SELECTORS: "Edit component selectors",
        FreeFormCellEditorKind.METADATA_SELECTORS: "Edit metadata selectors",
        FreeFormCellEditorKind.FILTER_CLAUSES: "Edit source filters",
        FreeFormCellEditorKind.MATCH_DIMENSIONS: "Edit match dimensions",
    }

    def __init__(
        self,
        *,
        editor_kind: FreeFormCellEditorKind,
        suggestions: tuple[str, ...],
        value: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.TITLES[editor_kind])
        self.editor_spec = STRUCTURED_SELECTOR_EDITOR_SPECS[editor_kind]
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(self.editor_spec.columns))
        self.table.setHorizontalHeaderLabels(self.editor_spec.columns)
        self.validation_label = QLabel(self)
        self.table.itemChanged.connect(lambda _item: self._update_validation_hint())
        for item in SelectorListCodec.items(value):
            self._append_structured_item(item)
        self.suggestions = QListWidget(self)
        self.suggestions.addItems(list(suggestions))
        self.suggestions.itemDoubleClicked.connect(
            lambda item: self._append_suggestion(item.text())
        )
        add_button = QPushButton("Add selected", self)
        add_button.clicked.connect(self._append_selected)
        add_row_button = QPushButton("Add row", self)
        add_row_button.clicked.connect(lambda: self._append_row(("",) * self.table.columnCount()))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self.editor_spec.hint, self))
        layout.addWidget(self.table)
        layout.addWidget(self.validation_label)
        layout.addWidget(add_row_button)
        layout.addWidget(QLabel("Suggestions", self))
        layout.addWidget(self.suggestions)
        layout.addWidget(add_button)
        layout.addWidget(buttons)
        self._update_validation_hint()

    def value(self) -> str:
        items = tuple(
            self._item_from_row(row_index)
            for row_index in range(self.table.rowCount())
        )
        return SelectorListCodec.ITEM_SEPARATOR.join(item for item in items if item)

    def _append_selected(self) -> None:
        for item in self.suggestions.selectedItems():
            self._append_suggestion(item.text())

    def _append_suggestion(self, text: str) -> None:
        self._append_structured_item(text)

    def _append_structured_item(self, text: str) -> None:
        self._append_row(self.editor_spec.row_parser(text))

    def _append_row(self, values: tuple[str | None, ...]) -> None:
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)
        for column_index in range(self.table.columnCount()):
            value = "" if column_index >= len(values) else values[column_index] or ""
            options = self.editor_spec.column_options.get(column_index)
            if options is not None:
                combo = QComboBox(self.table)
                combo.setEditable(True)
                combo.addItems(list(options))
                combo.setCurrentText(value)
                combo.currentTextChanged.connect(lambda _text: self._update_validation_hint())
                self.table.setCellWidget(row_index, column_index, combo)
                continue
            self.table.setItem(row_index, column_index, QTableWidgetItem(value))

    def _item_from_row(self, row_index: int) -> str:
        values = tuple(
            self._cell_text(row_index, column_index)
            for column_index in range(self.table.columnCount())
        )
        if not any(values):
            return ""
        return self.editor_spec.row_formatter(values)

    def _cell_text(self, row_index: int, column_index: int) -> str:
        widget = self.table.cellWidget(row_index, column_index)
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()
        item = self.table.item(row_index, column_index)
        return "" if item is None else item.text().strip()

    def _update_validation_hint(self) -> None:
        invalid_rows = tuple(
            row_index + 1
            for row_index in range(self.table.rowCount())
            if self._row_is_incomplete(row_index)
        )
        if invalid_rows:
            joined_rows = ", ".join(str(row) for row in invalid_rows)
            self.validation_label.setText(f"Incomplete rows ignored: {joined_rows}")
            return
        self.validation_label.setText("All rows are structurally valid.")

    def _row_is_incomplete(self, row_index: int) -> bool:
        values = tuple(
            self._cell_text(row_index, column_index)
            for column_index in range(self.table.columnCount())
        )
        return any(values) and not self.editor_spec.row_formatter(values)


@dataclass(frozen=True, slots=True)
class SourceBindingSuggestionSet:
    """Nominal source-binding editor suggestions derived from schema/inventory."""

    component_selectors: tuple[str, ...] = ()
    metadata_selectors: tuple[str, ...] = ()
    filter_clauses: tuple[str, ...] = ()
    match_fields: tuple[str, ...] = ()

    @classmethod
    def from_context(
        cls,
        *,
        schema: PipelineImageSchema,
        inventory: SourceInventory | None,
    ) -> "SourceBindingSuggestionSet":
        metadata_fields = cls.metadata_fields(schema=schema, inventory=inventory)
        aliases = tuple(sorted(schema.assignments_by_alias))
        return cls(
            component_selectors=tuple(
                f"{component.value}="
                for component in AllComponents
            ),
            metadata_selectors=tuple(f"{field}=" for field in metadata_fields)
            + cls.inventory_metadata_selectors(inventory),
            filter_clauses=tuple(
                f"{subject.value}:{match_type.value}:"
                for subject in SourceFilterSubject
                for match_type in SourceFilterMatchType
            ),
            match_fields=tuple(
                f"{alias}{SelectorListCodec.KEY_VALUE_SEPARATOR}{field}"
                for alias in aliases
                for field in metadata_fields
            ),
        )

    @staticmethod
    def metadata_fields(
        *,
        schema: PipelineImageSchema,
        inventory: SourceInventory | None,
    ) -> tuple[str, ...]:
        fields: set[str] = set()
        if schema.grouping is not None:
            fields.update(schema.grouping.metadata_fields)
        for rule in schema.metadata_rules:
            fields.update(re.compile(rule.pattern).groupindex)
        if inventory is not None:
            for candidate in inventory.candidates:
                fields.update(candidate.metadata)
        return tuple(sorted(fields))

    @staticmethod
    def inventory_metadata_selectors(
        inventory: SourceInventory | None,
    ) -> tuple[str, ...]:
        if inventory is None:
            return ()
        selectors: set[str] = set()
        for candidate in inventory.candidates:
            for field, value in candidate.metadata.items():
                selectors.add(f"{field}{SelectorListCodec.KEY_VALUE_SEPARATOR}{value}")
        return tuple(sorted(selectors))


@dataclass(frozen=True, slots=True)
class EditableTableController(Generic[EditableRowT]):
    """Own editable Qt table mechanics for one typed row model."""

    table: QTableWidget
    columns: tuple[IntEnum, ...]
    enum_cell_specs: EnumCellSpecMap
    free_form_cell_specs: FreeFormCellSpecMap
    row_cells: Callable[[EditableRowT], tuple[str, ...]]
    row_from_cells: Callable[[tuple[str, ...]], EditableRowT | None]
    apply_changes: Callable[[], None]

    def append(self, row_model: EditableRowT) -> None:
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)
        for column, value in zip(
            self.columns,
            self.row_cells(row_model),
            strict=True,
        ):
            self._set_cell(row_index, column, value)

    def rows(self) -> tuple[EditableRowT, ...]:
        rows: list[EditableRowT] = []
        for row_index in range(self.table.rowCount()):
            row_model = self.row_from_cells(
                tuple(self._cell_text(row_index, column) for column in self.columns)
            )
            if row_model is not None:
                rows.append(row_model)
        return tuple(rows)

    def remove_selected(self) -> bool:
        selected_rows = {index.row() for index in self.table.selectedIndexes()}
        if not selected_rows:
            return False
        for row_index in sorted(selected_rows, reverse=True):
            self.table.removeRow(row_index)
        return True

    def _cell_text(self, row_index: int, column: IntEnum) -> str:
        widget = self.table.cellWidget(row_index, int(column))
        if isinstance(widget, StructuredSelectorCellWidget):
            return widget.text()
        if isinstance(widget, QComboBox):
            value = widget.currentData()
            if value is None:
                return widget.currentText().strip()
            if not isinstance(value, Enum):
                raise TypeError(
                    "Editable enum cell must store an Enum value, "
                    f"got {type(value).__name__}."
                )
            return str(value.value)
        item = self.table.item(row_index, int(column))
        return "" if item is None else item.text()

    def _set_cell(
        self,
        row_index: int,
        column: IntEnum,
        value: str,
    ) -> None:
        spec = self.enum_cell_specs.get((type(column), column))
        free_form_spec = self.free_form_cell_specs.get((type(column), column))
        if spec is None and free_form_spec is None:
            self.table.setItem(row_index, int(column), QTableWidgetItem(value))
            return
        if free_form_spec is not None:
            self.table.setCellWidget(
                row_index,
                int(column),
                StructuredSelectorCellWidget(
                    values=free_form_spec.values,
                    value=value,
                    editor_kind=free_form_spec.editor_kind,
                    apply_changes=self.apply_changes,
                    parent=self.table,
                ),
            )
            return
        combo = QComboBox(self.table)
        for enum_value in spec.values:
            combo.addItem(spec.text_for_value(enum_value), enum_value)
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(lambda _: self.apply_changes())
        self.table.setCellWidget(row_index, int(column), combo)


class EditableTableLayout:
    """Shared layout policy for compact source-binding tables."""

    @staticmethod
    def configure(table: QTableWidget) -> None:
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

    @staticmethod
    def fit_to_rows(table: QTableWidget) -> None:
        table.resizeRowsToContents()
        header_height = table.horizontalHeader().height()
        row_height = sum(table.rowHeight(row) for row in range(table.rowCount()))
        if table.rowCount() == 0:
            row_height = table.verticalHeader().defaultSectionSize()
        scrollbar_height = table.horizontalScrollBar().sizeHint().height()
        frame = table.frameWidth() * 2
        table.setFixedHeight(header_height + row_height + scrollbar_height + frame + 8)


class StepBindingsTableEditor(QWidget):
    """Typed table editor for step-local source bindings."""

    changed = pyqtSignal()

    def __init__(
        self,
        *,
        groups: tuple[GroupedSourceBindings, ...],
        enum_cell_specs: EnumCellSpecMap,
        free_form_cell_specs: FreeFormCellSpecMap,
        scope_color_scheme: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._updating_ui = False
        self.table = ScopedTableWidget(0, len(SourceBindingColumn), self)
        self.table.set_scope_color_scheme(scope_color_scheme)
        self.table.setHorizontalHeaderLabels(
            tuple(column.name.title() for column in SourceBindingColumn)
        )
        self.controller = EditableTableController(
            table=self.table,
            columns=tuple(SourceBindingColumn),
            enum_cell_specs=enum_cell_specs,
            free_form_cell_specs=free_form_cell_specs,
            row_cells=EditableSourceBindingRow.cells,
            row_from_cells=EditableSourceBindingRow.from_cells,
            apply_changes=self._emit_changed,
        )
        for binding_group in groups:
            for binding in binding_group.bindings:
                self.controller.append(
                    EditableSourceBindingRow.from_binding(
                        group_key=binding_group.group_key,
                        binding=binding,
                    )
                )
        self.table.itemChanged.connect(lambda _: self._emit_changed())
        EditableTableLayout.configure(self.table)
        EditableTableLayout.fit_to_rows(self.table)

        buttons = QHBoxLayout()
        add_button = QPushButton("Add binding", self)
        add_button.clicked.connect(self.add_binding_row)
        remove_button = QPushButton("Remove selected", self)
        remove_button.clicked.connect(self.remove_selected_binding_rows)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    def add_binding_row(self, binding: NamedSourceBinding | None = None) -> None:
        row_model = EditableSourceBindingRow.from_binding(
            group_key=None,
            binding=binding or NamedSourceBinding(alias="NewSource"),
        )
        self._updating_ui = True
        try:
            self.controller.append(row_model)
        finally:
            self._updating_ui = False
        EditableTableLayout.fit_to_rows(self.table)
        self.changed.emit()

    def remove_selected_binding_rows(self) -> None:
        if not self.controller.remove_selected():
            return
        EditableTableLayout.fit_to_rows(self.table)
        self.changed.emit()

    def groups(self) -> tuple[GroupedSourceBindings, ...]:
        grouped_bindings: dict[str | None, list[NamedSourceBinding]] = {}
        for row in self.controller.rows():
            grouped_bindings.setdefault(row.group_key, []).append(row.binding)
        return tuple(
            GroupedSourceBindings(
                group_key=group_key,
                bindings=tuple(bindings),
            )
            for group_key, bindings in grouped_bindings.items()
        )

    def _emit_changed(self) -> None:
        if self._updating_ui:
            return
        self.changed.emit()


class StepBindingsDialog(QDialog):
    """Modal editor for the large step-bindings table."""

    def __init__(
        self,
        *,
        groups: tuple[GroupedSourceBindings, ...],
        enum_cell_specs: EnumCellSpecMap,
        free_form_cell_specs: FreeFormCellSpecMap,
        scope_color_scheme: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit step source bindings")
        self.editor = StepBindingsTableEditor(
            groups=groups,
            enum_cell_specs=enum_cell_specs,
            free_form_cell_specs=free_form_cell_specs,
            scope_color_scheme=scope_color_scheme,
            parent=self,
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)
        layout.addWidget(buttons)
        self.resize(1100, 520)

    def groups(self) -> tuple[GroupedSourceBindings, ...]:
        return self.editor.groups()


class SelectorListCodec:
    """Compact table-cell codec for typed source selectors."""

    ITEM_SEPARATOR = ";"
    KEY_VALUE_SEPARATOR = "="
    FILTER_SEPARATOR = ":"

    @classmethod
    def component_cells(cls, selector: SourceSelector) -> str:
        return cls.ITEM_SEPARATOR.join(
            f"{component_selector.component.value}{cls.KEY_VALUE_SEPARATOR}{component_selector.value}"
            for component_selector in selector.components
        )

    @classmethod
    def metadata_cells(cls, selector: SourceSelector) -> str:
        return cls.ITEM_SEPARATOR.join(
            f"{metadata_selector.field}{cls.KEY_VALUE_SEPARATOR}{metadata_selector.value}"
            for metadata_selector in selector.metadata
        )

    @classmethod
    def filter_cells(cls, selector: SourceSelector) -> str:
        return cls.ITEM_SEPARATOR.join(
            cls.FILTER_SEPARATOR.join(
                (
                    clause.subject.value,
                    clause.match_type.value,
                    clause.value or "",
                )
            )
            for clause in selector.filters
        )

    @classmethod
    def parse_components(cls, text: str) -> tuple[ComponentSelector, ...]:
        selectors: list[ComponentSelector] = []
        for item in cls.items(text):
            key, value = cls.key_value_parts(item)
            selectors.append(
                ComponentSelector(
                    component=AllComponents(key),
                    value=value,
                )
            )
        return tuple(selectors)

    @classmethod
    def parse_metadata(cls, text: str) -> tuple[MetadataSelector, ...]:
        selectors: list[MetadataSelector] = []
        for item in cls.items(text):
            key, value = cls.key_value_parts(item)
            selectors.append(MetadataSelector(field=key, value=value))
        return tuple(selectors)

    @classmethod
    def parse_filters(cls, text: str) -> tuple[SourceFilterClause, ...]:
        filters: list[SourceFilterClause] = []
        for item in cls.items(text):
            subject, match_type, value = cls.filter_parts(item)
            filters.append(
                SourceFilterClause(
                    subject=SourceFilterSubject(subject),
                    match_type=SourceFilterMatchType(match_type),
                    value=value or None,
                )
            )
        return tuple(filters)

    @classmethod
    def match_field_cells(cls, dimension: SourceBindingMatchDimension) -> str:
        return cls.ITEM_SEPARATOR.join(
            f"{field.alias}{cls.KEY_VALUE_SEPARATOR}{field.metadata_field}"
            for field in dimension.fields
        )

    @classmethod
    def parse_match_fields(cls, text: str) -> tuple[SourceBindingMatchField, ...]:
        fields: list[SourceBindingMatchField] = []
        for item in cls.items(text):
            alias, metadata_field = cls.key_value_parts(item)
            fields.append(
                SourceBindingMatchField(
                    alias=alias,
                    metadata_field=metadata_field,
                )
            )
        return tuple(fields)

    @classmethod
    def items(cls, text: str) -> tuple[str, ...]:
        return tuple(
            item.strip()
            for item in text.split(cls.ITEM_SEPARATOR)
            if item.strip()
        )

    @classmethod
    def key_value_parts(cls, item: str) -> tuple[str, str]:
        key, separator, value = item.partition(cls.KEY_VALUE_SEPARATOR)
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"Expected selector item as key=value, got {item!r}.")
        return key.strip(), value.strip()

    @classmethod
    def filter_parts(cls, item: str) -> tuple[str, str, str | None]:
        parts = tuple(part.strip() for part in item.split(cls.FILTER_SEPARATOR, 2))
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "Expected filter item as subject:match_type[:value], "
                f"got {item!r}."
            )
        return parts[0], parts[1], parts[2] if len(parts) == 3 else None


@dataclass(frozen=True, slots=True)
class EditableSourceBindingRow:
    """Nominal row model for editing one source-binding declaration."""

    group_key: str | None
    binding: NamedSourceBinding

    @classmethod
    def from_cells(cls, values: tuple[str, ...]) -> "EditableSourceBindingRow | None":
        (
            group_key,
            alias,
            artifact_kind,
            origin,
            required,
            components,
            metadata,
            filters,
            inherit_current_scope,
        ) = (
            value.strip() for value in values
        )
        if not alias:
            return None
        selector = SourceSelector(
            components=SelectorListCodec.parse_components(components),
            metadata=SelectorListCodec.parse_metadata(metadata),
            filters=SelectorListCodec.parse_filters(filters),
            inherit_current_scope=inherit_current_scope.lower()
            not in {"false", "0", "no", "n"},
        )
        return cls(
            group_key=group_key or None,
            binding=NamedSourceBinding(
                alias=alias,
                artifact_kind=ArtifactKind(artifact_kind or ArtifactKind.IMAGE.value),
                selector=selector,
                origin=SourceBindingOrigin(origin or SourceBindingOrigin.STEP_INPUT.value),
                required=required.lower() not in {"false", "0", "no", "n"},
            ),
        )

    @classmethod
    def from_binding(
        cls,
        *,
        group_key: str | None,
        binding: NamedSourceBinding,
    ) -> "EditableSourceBindingRow":
        return cls(group_key=group_key, binding=binding)

    def cells(self) -> tuple[str, ...]:
        return (
            self.group_key or "",
            self.binding.alias,
            self.binding.artifact_kind.value,
            self.binding.origin.value,
            str(self.binding.required),
            SelectorListCodec.component_cells(self.binding.selector),
            SelectorListCodec.metadata_cells(self.binding.selector),
            SelectorListCodec.filter_cells(self.binding.selector),
            str(self.binding.selector.inherit_current_scope),
        )


@dataclass(frozen=True, slots=True)
class EditableMetadataRuleRow:
    """Nominal row model for editing one metadata extraction rule."""

    rule: MetadataExtractionRule

    @classmethod
    def from_cells(cls, values: tuple[str, ...]) -> "EditableMetadataRuleRow | None":
        source, pattern, filters = (value.strip() for value in values)
        if not pattern:
            return None
        return cls(
            rule=MetadataExtractionRule(
                source=MetadataSource(source or MetadataSource.FILE_NAME.value),
                pattern=pattern,
                filters=SelectorListCodec.parse_filters(filters),
            ),
        )

    @classmethod
    def from_rule(cls, rule: MetadataExtractionRule) -> "EditableMetadataRuleRow":
        return cls(rule=rule)

    def cells(self) -> tuple[str, ...]:
        return (
            self.rule.source.value,
            self.rule.pattern,
            SelectorListCodec.filter_cells(SourceSelector(filters=self.rule.filters)),
        )


@dataclass(frozen=True, slots=True)
class EditableMatchPlanRow:
    """Nominal row model for editing one match-plan dimension."""

    method: SourceBindingMatchMethod
    dimension: SourceBindingMatchDimension | None = None

    @classmethod
    def from_cells(cls, values: tuple[str, ...]) -> "EditableMatchPlanRow | None":
        method, fields = (value.strip() for value in values)
        if not method and not fields:
            return None
        return cls(
            method=SourceBindingMatchMethod(
                method or SourceBindingMatchMethod.ORDER.value
            ),
            dimension=SourceBindingMatchDimension(
                fields=SelectorListCodec.parse_match_fields(fields),
            )
            if fields
            else None,
        )

    @classmethod
    def from_plan(
        cls,
        plan: SourceBindingMatchPlan | None,
    ) -> tuple["EditableMatchPlanRow", ...]:
        if plan is None:
            return ()
        if not plan.dimensions:
            return (cls(method=plan.method),)
        return tuple(
            cls(method=plan.method, dimension=dimension)
            for dimension in plan.dimensions
        )

    def cells(self) -> tuple[str, ...]:
        return (
            self.method.value,
            ""
            if self.dimension is None
            else SelectorListCodec.match_field_cells(self.dimension),
        )


class SourceBindingsEditorWidget(
    QWidget,
    ValueGettable,
    ValueSettable,
    ChangeSignalEmitter,
    ScopeColorSchemeReceiver,
    metaclass=PyQtWidgetMeta,
):
    """Inline form widget for typed source-binding semantics."""

    changed = pyqtSignal()

    def __init__(
        self,
        view_model: SourceBindingsViewModel | None = None,
        *,
        schema: PipelineImageSchema = EMPTY_PIPELINE_IMAGE_SCHEMA,
        bindings: StepSourceBindingsConfig | None = None,
        inventory: SourceInventory | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._schema = schema
        self._bindings = bindings or StepSourceBindingsConfig()
        self._inventory = inventory
        self._updating_ui = False
        self.step_bindings_table: QTableWidget | None = None
        self.step_bindings_editor: StepBindingsTableEditor | None = None
        self.metadata_rules_table: QTableWidget | None = None
        self.match_plan_table: QTableWidget | None = None
        self.metadata_rules_controller: (
            EditableTableController[EditableMetadataRuleRow] | None
        ) = None
        self.match_plan_controller: (
            EditableTableController[EditableMatchPlanRow] | None
        ) = None
        self._scope_color_scheme = None
        self._enum_cell_specs = {
            (SourceBindingColumn, SourceBindingColumn.KIND): EnumCellSpec(ArtifactKind),
            (SourceBindingColumn, SourceBindingColumn.ORIGIN): EnumCellSpec(SourceBindingOrigin),
            (MetadataRuleColumn, MetadataRuleColumn.SOURCE): EnumCellSpec(MetadataSource),
            (MatchPlanColumn, MatchPlanColumn.METHOD): EnumCellSpec(SourceBindingMatchMethod),
        }
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)
        self.empty_label = QLabel("No source bindings loaded")
        self.layout.addWidget(self.empty_label)
        if view_model is not None:
            self.set_view_model(view_model)

    @classmethod
    def from_bindings(
        cls,
        bindings: StepSourceBindingsConfig,
        *,
        schema: PipelineImageSchema = EMPTY_PIPELINE_IMAGE_SCHEMA,
        inventory: SourceInventory | None = None,
        parent: QWidget | None = None,
    ) -> "SourceBindingsEditorWidget":
        """Create an editor from typed source bindings and an optional schema."""

        return cls(
            SourceBindingsViewModel.from_schema_and_bindings(
                schema=schema,
                bindings=bindings,
            ),
            schema=schema,
            bindings=bindings,
            inventory=inventory,
            parent=parent,
        )

    @property
    def value(self) -> StepSourceBindingsConfig:
        """Value property used by pyqt-reactive signal adapters."""

        return self.get_value()

    def get_value(self) -> StepSourceBindingsConfig:
        """Return the current typed source-bindings config."""

        return self._bindings

    def set_value(self, value: StepSourceBindingsConfig | None) -> None:
        """Update the widget from a typed source-bindings config."""

        bindings = value or StepSourceBindingsConfig()
        if not isinstance(bindings, StepSourceBindingsConfig):
            raise TypeError(
                "SourceBindingsEditorWidget value must be StepSourceBindingsConfig, "
                f"got {type(bindings).__name__}."
            )
        if bindings == self._bindings:
            return
        self._bindings = bindings
        self.refresh()
        self.changed.emit()

    def set_preview_context(
        self,
        *,
        schema: PipelineImageSchema,
        inventory: SourceInventory | None = None,
    ) -> None:
        """Set pipeline-level schema and optional inventory for source preview."""

        self._schema = schema
        self._inventory = inventory
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the rendered view from current bindings and preview context."""

        self.set_view_model(
            SourceBindingsViewModel.from_schema_and_bindings(
                schema=self._schema,
                bindings=self._bindings,
            )
        )

    def set_view_model(self, view_model: SourceBindingsViewModel) -> None:
        self._updating_ui = True
        self.clear()
        self.layout.addWidget(
            self._table_group(
                "Pipeline Sources",
                ("Field", "Value"),
                (
                    (
                        "image-plane sources",
                        str(view_model.pipeline_sources.image_plane_source_count),
                    ),
                    (
                        "imported metadata tables",
                        str(len(view_model.pipeline_sources.imported_metadata_tables)),
                    ),
                ),
            ),
        )
        self.layout.addWidget(
            self._table_group(
                "Pipeline Bindings",
                ("Alias", "Kind", "Origin", "Payload"),
                tuple(
                    (
                        row.alias,
                        row.artifact_kind,
                        row.origin,
                        row.payload_type or "",
                    )
                    for row in view_model.pipeline_bindings
                ),
            ),
        )
        self.layout.addWidget(self._step_bindings_group(view_model))
        self.layout.addWidget(self._metadata_rules_group())
        self.layout.addWidget(self._match_plan_group())
        self.layout.addWidget(
            self._table_group(
                "Match Plans",
                ("Scope", "Method", "Dimensions"),
                tuple(
                    (
                        plan.declaration_scope,
                        plan.method,
                        str(len(plan.dimensions)),
                    )
                    for plan in view_model.match_plans
                ),
            )
        )
        if self._inventory is not None:
            preview = SourceBindingsPreview.from_schema_and_bindings(
                schema=self._schema,
                bindings=self._bindings,
                inventory=self._inventory,
            )
            if preview.diagnostics:
                self.layout.addWidget(
                    self._table_group(
                        "Diagnostics",
                        ("Severity", "Code", "Alias", "Message"),
                        tuple(
                            (
                                diagnostic.severity.value,
                                diagnostic.code,
                                diagnostic.alias or "",
                                diagnostic.message,
                            )
                            for diagnostic in preview.diagnostics
                        ),
                    )
                )
            self.layout.addWidget(
                self._table_group(
                    "Preview Matches",
                    ("Alias", "Scope", "Matched", "Samples"),
                    tuple(
                        (
                            row.alias,
                            row.declaration_scope,
                            str(row.matched_source_count),
                            ", ".join(row.sample_paths),
                        )
                        for row in preview.binding_rows
                    ),
                )
            )
            self.layout.addWidget(
                self._table_group(
                    "Image Sets",
                    ("Index", "Aliases", "Metadata"),
                    tuple(
                        (
                            str(row.index),
                            ", ".join(
                                f"{alias}:{path}"
                                for alias, path in row.paths_by_alias
                            ),
                            ", ".join(
                                f"{field}={value}"
                                for field, value in row.metadata
                            ),
                        )
                        for row in preview.image_set_rows
                    ),
                )
            )
        self.layout.addStretch(1)
        self._updating_ui = False

    def connect_change_signal(self, callback: Callable[[StepSourceBindingsConfig], None]) -> None:
        """Implement ChangeSignalEmitter for pyqt-reactive inline dataclass forms."""

        self.changed.connect(lambda: callback(self.get_value()))

    def disconnect_change_signal(self, callback: Callable[[StepSourceBindingsConfig], None]) -> None:
        """Disconnect a previously registered change callback when Qt can match it."""

        try:
            self.changed.disconnect(callback)
        except TypeError:
            pass

    def set_scope_color_scheme(self, scheme) -> None:
        """Apply scope styling to every source-binding table."""

        self._scope_color_scheme = scheme
        for table in self.findChildren(ScopedTableWidget):
            table.set_scope_color_scheme(scheme)

    def add_binding_row(self, binding: NamedSourceBinding | None = None) -> None:
        """Append one editable source binding row to the step-binding table."""

        if self.step_bindings_table is None:
            self._append_step_binding(binding or NamedSourceBinding(alias="NewSource"))
            return
        if self.step_bindings_editor is None:
            self._append_step_binding(binding or NamedSourceBinding(alias="NewSource"))
            return
        self.step_bindings_editor.add_binding_row(binding)
        self._apply_step_binding_groups(self.step_bindings_editor.groups())

    def remove_selected_binding_rows(self) -> None:
        """Remove selected source binding rows from the open dialog editor."""

        if self.step_bindings_editor is None:
            return
        self.step_bindings_editor.remove_selected_binding_rows()
        self._apply_step_binding_groups(self.step_bindings_editor.groups())

    def add_metadata_rule_row(
        self,
        rule: MetadataExtractionRule | None = None,
    ) -> None:
        """Append one editable metadata extraction rule row."""

        if self.metadata_rules_table is None:
            return
        if self.metadata_rules_controller is None:
            raise RuntimeError("Metadata rules table controller is not initialized.")
        self._updating_ui = True
        try:
            self.metadata_rules_controller.append(
                EditableMetadataRuleRow.from_rule(
                    rule
                    or MetadataExtractionRule(
                        source=MetadataSource.FILE_NAME,
                        pattern=r"(?P<field>.+)",
                    )
                )
            )
        finally:
            self._updating_ui = False
        self._fit_table_to_rows(self.metadata_rules_table)
        self._apply_metadata_rules_table()

    def remove_selected_metadata_rule_rows(self) -> None:
        """Remove selected metadata extraction rule rows."""

        if self.metadata_rules_table is None:
            return
        if self.metadata_rules_controller is None:
            raise RuntimeError("Metadata rules table controller is not initialized.")
        if not self.metadata_rules_controller.remove_selected():
            return
        self._fit_table_to_rows(self.metadata_rules_table)
        self._apply_metadata_rules_table()

    def add_match_plan_row(
        self,
        row: EditableMatchPlanRow | None = None,
    ) -> None:
        """Append one editable match-plan dimension row."""

        if self.match_plan_table is None:
            return
        if self.match_plan_controller is None:
            raise RuntimeError("Match plan table controller is not initialized.")
        self._updating_ui = True
        try:
            self.match_plan_controller.append(
                row
                or EditableMatchPlanRow(method=SourceBindingMatchMethod.METADATA)
            )
        finally:
            self._updating_ui = False
        self._fit_table_to_rows(self.match_plan_table)
        self._apply_match_plan_table()

    def remove_selected_match_plan_rows(self) -> None:
        """Remove selected match-plan dimension rows."""

        if self.match_plan_table is None:
            return
        if self.match_plan_controller is None:
            raise RuntimeError("Match plan table controller is not initialized.")
        if not self.match_plan_controller.remove_selected():
            return
        self._fit_table_to_rows(self.match_plan_table)
        self._apply_match_plan_table()

    def clear(self) -> None:
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.step_bindings_table = None
        self.step_bindings_editor = None
        self.metadata_rules_table = None
        self.match_plan_table = None
        self.metadata_rules_controller = None
        self.match_plan_controller = None

    def _free_form_cell_specs(
        self,
    ) -> FreeFormCellSpecMap:
        suggestions = SourceBindingSuggestionSet.from_context(
            schema=self._schema,
            inventory=self._inventory,
        )
        return {
            (SourceBindingColumn, SourceBindingColumn.COMPONENTS): FreeFormCellSpec(
                suggestions.component_selectors,
                FreeFormCellEditorKind.COMPONENT_SELECTORS,
            ),
            (SourceBindingColumn, SourceBindingColumn.METADATA): FreeFormCellSpec(
                suggestions.metadata_selectors,
                FreeFormCellEditorKind.METADATA_SELECTORS,
            ),
            (SourceBindingColumn, SourceBindingColumn.FILTERS): FreeFormCellSpec(
                suggestions.filter_clauses,
                FreeFormCellEditorKind.FILTER_CLAUSES,
            ),
            (MetadataRuleColumn, MetadataRuleColumn.FILTERS): FreeFormCellSpec(
                suggestions.filter_clauses,
                FreeFormCellEditorKind.FILTER_CLAUSES,
            ),
            (MatchPlanColumn, MatchPlanColumn.FIELDS): FreeFormCellSpec(
                suggestions.match_fields,
                FreeFormCellEditorKind.MATCH_DIMENSIONS,
            ),
        }

    def _step_bindings_group(self, view_model: SourceBindingsViewModel) -> QGroupBox:
        group = self._section_group("Step Bindings")
        layout = QVBoxLayout(group)
        summary_table = self._create_table(0, 4)
        summary_table.setHorizontalHeaderLabels(("Group", "Bindings", "Aliases", "Origins"))
        for row_index, row in enumerate(self._binding_summary_rows(view_model)):
            summary_table.insertRow(row_index)
            for column_index, value in enumerate(row):
                summary_table.setItem(row_index, column_index, QTableWidgetItem(value))
        summary_table.resizeColumnsToContents()
        self._configure_table(summary_table)
        self._fit_table_to_rows(summary_table)
        layout.addWidget(summary_table)

        buttons = QHBoxLayout()
        edit_button = QPushButton("Edit bindings...", group)
        edit_button.clicked.connect(self._open_step_bindings_dialog)
        add_button = QPushButton("Add binding", group)
        add_button.clicked.connect(lambda: self.add_binding_row())
        buttons.addWidget(edit_button)
        buttons.addWidget(add_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return group

    def _create_step_bindings_dialog(self) -> StepBindingsDialog:
        return StepBindingsDialog(
            groups=self._bindings.groups,
            enum_cell_specs=self._enum_cell_specs,
            free_form_cell_specs=self._free_form_cell_specs(),
            scope_color_scheme=self._scope_color_scheme,
            parent=self,
        )

    def _open_step_bindings_dialog(self) -> None:
        dialog = self._create_step_bindings_dialog()
        self.step_bindings_editor = dialog.editor
        self.step_bindings_table = dialog.editor.table
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        if accepted:
            self._apply_step_binding_groups(dialog.groups())
        self.step_bindings_editor = None
        self.step_bindings_table = None

    def _append_step_binding(self, binding: NamedSourceBinding) -> None:
        groups = list(self._bindings.groups)
        if groups:
            first_group = groups[0]
            groups[0] = GroupedSourceBindings(
                group_key=first_group.group_key,
                bindings=first_group.bindings + (binding,),
            )
        else:
            groups.append(
                GroupedSourceBindings(
                    group_key=None,
                    bindings=(binding,),
                )
            )
        self._apply_step_binding_groups(tuple(groups))

    def _binding_summary_rows(
        self,
        view_model: SourceBindingsViewModel,
    ) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(
            (
                group.group_key or "default",
                str(len(group.bindings)),
                ", ".join(binding.alias for binding in group.bindings),
                ", ".join(sorted({binding.origin for binding in group.bindings})),
            )
            for group in view_model.step_binding_groups
        )

    def _metadata_rules_group(self) -> QGroupBox:
        group = self._section_group("Step Metadata Rules")
        layout = QVBoxLayout(group)
        table = self._create_table(0, len(MetadataRuleColumn))
        table.setHorizontalHeaderLabels(
            tuple(column.name.title() for column in MetadataRuleColumn)
        )
        self.metadata_rules_table = table
        self.metadata_rules_controller = EditableTableController(
            table=table,
            columns=tuple(MetadataRuleColumn),
            enum_cell_specs=self._enum_cell_specs,
            free_form_cell_specs=self._free_form_cell_specs(),
            row_cells=EditableMetadataRuleRow.cells,
            row_from_cells=EditableMetadataRuleRow.from_cells,
            apply_changes=self._apply_metadata_rules_table,
        )
        for rule in self._bindings.metadata_rules:
            self.metadata_rules_controller.append(EditableMetadataRuleRow.from_rule(rule))
        table.itemChanged.connect(lambda _: self._apply_metadata_rules_table())
        table.resizeColumnsToContents()
        self._configure_table(table)
        self._fit_table_to_rows(table)
        layout.addWidget(table)

        buttons = QHBoxLayout()
        add_button = QPushButton("Add metadata rule")
        add_button.clicked.connect(self.add_metadata_rule_row)
        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self.remove_selected_metadata_rule_rows)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return group

    def _match_plan_group(self) -> QGroupBox:
        group = self._section_group("Step Match Plan")
        layout = QVBoxLayout(group)
        table = self._create_table(0, len(MatchPlanColumn))
        table.setHorizontalHeaderLabels(
            tuple(column.name.title() for column in MatchPlanColumn)
        )
        self.match_plan_table = table
        self.match_plan_controller = EditableTableController(
            table=table,
            columns=tuple(MatchPlanColumn),
            enum_cell_specs=self._enum_cell_specs,
            free_form_cell_specs=self._free_form_cell_specs(),
            row_cells=EditableMatchPlanRow.cells,
            row_from_cells=EditableMatchPlanRow.from_cells,
            apply_changes=self._apply_match_plan_table,
        )
        for row in EditableMatchPlanRow.from_plan(self._bindings.match_plan):
            self.match_plan_controller.append(row)
        table.itemChanged.connect(lambda _: self._apply_match_plan_table())
        table.resizeColumnsToContents()
        self._configure_table(table)
        self._fit_table_to_rows(table)
        layout.addWidget(table)

        buttons = QHBoxLayout()
        add_button = QPushButton("Add match dimension")
        add_button.clicked.connect(self.add_match_plan_row)
        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self.remove_selected_match_plan_rows)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return group

    def _apply_step_binding_groups(
        self,
        groups: tuple[GroupedSourceBindings, ...],
    ) -> None:
        if self._updating_ui:
            return
        self._bindings = StepSourceBindingsConfig(
            groups=groups,
            metadata_rules=self._bindings.metadata_rules,
            match_plan=self._bindings.match_plan,
        )
        self.refresh()
        self.changed.emit()

    def _apply_metadata_rules_table(self) -> None:
        if self._updating_ui or self.metadata_rules_table is None:
            return
        if self.metadata_rules_controller is None:
            raise RuntimeError("Metadata rules table controller is not initialized.")
        self._bindings = StepSourceBindingsConfig(
            groups=self._bindings.groups,
            metadata_rules=tuple(
                row.rule for row in self.metadata_rules_controller.rows()
            ),
            match_plan=self._bindings.match_plan,
        )
        self.changed.emit()

    def _apply_match_plan_table(self) -> None:
        if self._updating_ui or self.match_plan_table is None:
            return
        if self.match_plan_controller is None:
            raise RuntimeError("Match plan table controller is not initialized.")
        rows = self.match_plan_controller.rows()
        self._bindings = StepSourceBindingsConfig(
            groups=self._bindings.groups,
            metadata_rules=self._bindings.metadata_rules,
            match_plan=self._match_plan_from_rows(rows),
        )
        self.changed.emit()

    @staticmethod
    def _match_plan_from_rows(
        rows: tuple[EditableMatchPlanRow, ...],
    ) -> SourceBindingMatchPlan | None:
        if not rows:
            return None
        method = rows[0].method
        dimensions = tuple(
            row.dimension for row in rows if row.dimension is not None
        )
        return SourceBindingMatchPlan(method=method, dimensions=dimensions)

    def _table_group(
        self,
        title: str,
        columns: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
    ) -> QGroupBox:
        group = self._section_group(title)
        layout = QVBoxLayout(group)
        table = self._create_table(len(rows), len(columns))
        table.setHorizontalHeaderLabels(columns)
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                table.setItem(row_index, column_index, QTableWidgetItem(value))
        table.resizeColumnsToContents()
        self._configure_table(table)
        self._fit_table_to_rows(table)
        layout.addWidget(table)
        return group

    def _create_table(self, rows: int, columns: int) -> ScopedTableWidget:
        table = ScopedTableWidget(rows, columns)
        table.set_scope_color_scheme(self._scope_color_scheme)
        return table

    @staticmethod
    def _section_group(title: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setStyleSheet(
            """
            QGroupBox {
                margin-top: 18px;
            }
            QGroupBox::title {
                color: #f0f0f0;
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 3px;
            }
            """
        )
        return group

    @staticmethod
    def _configure_table(table: QTableWidget) -> None:
        EditableTableLayout.configure(table)

    @staticmethod
    def _fit_table_to_rows(table: QTableWidget) -> None:
        EditableTableLayout.fit_to_rows(table)


def create_source_bindings_editor_widget(
    *,
    current_value: StepSourceBindingsConfig,
    parent: QWidget | None = None,
    **_: object,
) -> SourceBindingsEditorWidget:
    """pyqt-reactive inline dataclass widget factory for source bindings."""

    if not isinstance(current_value, StepSourceBindingsConfig):
        raise TypeError(
            "SourceBindingsEditorWidget requires StepSourceBindingsConfig, "
            f"got {type(current_value).__name__}."
        )
    return SourceBindingsEditorWidget.from_bindings(current_value, parent=parent)


def register_source_bindings_editor_widget() -> None:
    """Register the typed source-binding editor with pyqt-reactive forms."""

    from pyqt_reactive.forms import register_inline_dataclass_widget

    register_inline_dataclass_widget(
        StepSourceBindingsConfig,
        create_source_bindings_editor_widget,
    )


register_source_bindings_editor_widget()


__all__ = (
    "SourceBindingsEditorWidget",
    "create_source_bindings_editor_widget",
    "register_source_bindings_editor_widget",
)
