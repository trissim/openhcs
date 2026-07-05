"""PyQt source-bindings editor over the typed source-binding view model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Generic, Mapping, TypeVar, cast

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
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
    ChildFieldChromeRefreshable,
    ChildFieldIdentityProvider,
    ChildFieldNavigationTargetProvider,
    ChildFieldSemanticChromeRefreshable,
    ChildSubfieldNavigationTargetProvider,
    ChangeSignalEmitter,
    InlineDataclassGroupBoxChromeProvider,
    InlineDataclassRootResettable,
    PyQtWidgetMeta,
    RawResolvedValueSettable,
    ResolvedValuePreviewSettable,
    ValueGettable,
    ValueSettable,
)
from pyqt_reactive.forms.layout_constants import CURRENT_LAYOUT
from pyqt_reactive.forms.inline_dataclass_chrome import InlineDataclassChildChrome
from pyqt_reactive.forms.inline_dataclass_context import (
    InlineDataclassChildFieldIdentity,
    InlineDataclassFormContext,
)
from pyqt_reactive.forms.widget_creation_config import (
    EnabledTitleWidgetMoveAuthority,
)
from pyqt_reactive.forms.widget_strategies import PlaceholderConfig
from pyqt_reactive.widgets.shared.scoped_table_widget import ScopedTableWidget
from pyqt_reactive.widgets.shared.scope_color_receiver import ScopeColorSchemeReceiver
from pyqt_reactive.widgets.structural_table import (
    EditableTableSemanticBinding,
    IsomorphicDataclassRowPathPolicy,
    StructuralDescendantMaskTarget,
    StructuralFlashTarget,
    StructuralMaskedContainerTarget,
    StructuralTableCellTarget,
    StructuralWidgetTarget,
)
from pyqt_reactive.widgets.shared.clickable_help_components import (
    InlineDataclassGroupBox,
)
from pyqt_reactive.widgets.no_scroll_spinbox import NoScrollComboBox, NoneAwareCheckBox
from python_introspect import Enableable, is_enableable

from openhcs.constants.constants import AllComponents
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.source_bindings import (
    ComponentSelector,
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
    SourceBindingsConfig,
    StepSourceBindingsConfig,
)
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.source_bindings_view import (
    SourceBindingsPreview,
    SourceBindingsViewModel,
    SourceInventory,
)
from objectstate import (
    DataclassFieldAccess,
    DottedFieldPath,
    ObjectStateSubfieldSemanticIndex,
    StructuralValuePath,
)
from objectstate.lazy_factory import LazyDataclass, get_base_type_for_lazy, replace_raw

if TYPE_CHECKING:
    from objectstate import ObjectState
    from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager
    from pyqt_reactive.forms.parameter_info_types import InlineDataclassWidgetInfo

EMPTY_PIPELINE_IMAGE_SCHEMA = PipelineImageSchema.empty()
EditableRowT = TypeVar("EditableRowT")
SourceBindingsEditorRawValue = SourceBindingsConfig | LazyDataclass


class EditableTableColumn(Enum):
    """Qt table column declaration with optional enum-editor authority."""

    def __new__(
        cls,
        index: int,
        enum_type: type[Enum] | None = None,
    ) -> "EditableTableColumn":
        member = object.__new__(cls)
        member._value_ = index
        member.index = index
        member.enum_type = enum_type
        return member

    def __int__(self) -> int:
        return self.index

    def __index__(self) -> int:
        return self.index

    def enum_cell_spec(self) -> "EnumCellSpec | None":
        if self.enum_type is None:
            return None
        return EnumCellSpec(self.enum_type)


class SourceBindingColumn(EditableTableColumn):
    """Editable table columns for one named source binding."""

    ALIAS = (0, None)
    KIND = (1, ArtifactKind)
    ORIGIN = (2, SourceBindingOrigin)
    REQUIRED = (3, None)
    COMPONENTS = (4, None)
    METADATA = (5, None)
    FILTERS = (6, None)
    INHERIT = (7, None)


class MetadataRuleColumn(EditableTableColumn):
    """Editable table columns for one metadata extraction rule."""

    SOURCE = (0, MetadataSource)
    PATTERN = (1, None)
    FILTERS = (2, None)


class SourceFilterColumn(EditableTableColumn):
    """Editable table columns for one source-universe filter clause."""

    SUBJECT = (0, SourceFilterSubject)
    MATCH_TYPE = (1, SourceFilterMatchType)
    VALUE = (2, None)


class MatchPlanColumn(EditableTableColumn):
    """Editable table columns for one match-plan dimension."""

    METHOD = (0, SourceBindingMatchMethod)
    FIELDS = (1, None)


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
    title: str
    columns: tuple[str, ...]
    hint: str
    row_parser: SelectorDialogRowParser
    row_formatter: SelectorDialogRowFormatter
    column_options: Mapping[int, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )


STRUCTURED_SELECTOR_EDITOR_SPEC_ITEMS: tuple[StructuredSelectorEditorSpec, ...] = (
    StructuredSelectorEditorSpec(
        editor_kind=FreeFormCellEditorKind.SELECTOR_LIST,
        title="Edit selector list",
        columns=("Key", "Value"),
        hint="Use key=value entries separated by semicolons.",
        row_parser=parse_key_value_dialog_row,
        row_formatter=format_key_value_dialog_row,
    ),
    StructuredSelectorEditorSpec(
        editor_kind=FreeFormCellEditorKind.COMPONENT_SELECTORS,
        title="Edit component selectors",
        columns=("Component", "Value"),
        hint="Use key=value entries separated by semicolons.",
        row_parser=parse_key_value_dialog_row,
        row_formatter=format_key_value_dialog_row,
        column_options=MappingProxyType(
            {0: tuple(component.value for component in AllComponents)}
        ),
    ),
    StructuredSelectorEditorSpec(
        editor_kind=FreeFormCellEditorKind.METADATA_SELECTORS,
        title="Edit metadata selectors",
        columns=("Metadata field", "Value"),
        hint="Use key=value entries separated by semicolons.",
        row_parser=parse_key_value_dialog_row,
        row_formatter=format_key_value_dialog_row,
    ),
    StructuredSelectorEditorSpec(
        editor_kind=FreeFormCellEditorKind.FILTER_CLAUSES,
        title="Edit source filters",
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
    StructuredSelectorEditorSpec(
        editor_kind=FreeFormCellEditorKind.MATCH_DIMENSIONS,
        title="Edit match dimensions",
        columns=("Alias", "Metadata field"),
        hint="Use alias=metadata_field entries separated by semicolons.",
        row_parser=parse_key_value_dialog_row,
        row_formatter=format_key_value_dialog_row,
    ),
)
STRUCTURED_SELECTOR_EDITOR_SPECS: Mapping[
    FreeFormCellEditorKind,
    StructuredSelectorEditorSpec,
] = MappingProxyType(
    {
        spec.editor_kind: spec
        for spec in STRUCTURED_SELECTOR_EDITOR_SPEC_ITEMS
    }
)


@dataclass(frozen=True, slots=True)
class FreeFormCellSpec:
    """Editable suggestions and semantic dialog type for selector cells."""

    values: tuple[str, ...]
    editor_kind: FreeFormCellEditorKind = FreeFormCellEditorKind.SELECTOR_LIST


FreeFormCellSpecMap = Mapping[
    tuple[type[EditableTableColumn], EditableTableColumn],
    FreeFormCellSpec,
]


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

    def __init__(
        self,
        *,
        editor_kind: FreeFormCellEditorKind,
        suggestions: tuple[str, ...],
        value: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.editor_spec = STRUCTURED_SELECTOR_EDITOR_SPECS[editor_kind]
        self.setWindowTitle(self.editor_spec.title)
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
                combo = NoScrollComboBox(self.table)
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


@dataclass(slots=True, weakref_slot=True)
class EditableTableProgrammaticUpdateGuard:
    """Suppress table-driven change callbacks while the model updates the view."""

    depth: int = 0
    pending_release: bool = False
    programmatic_row_values: tuple[tuple[str, ...], ...] | None = None

    def __enter__(self) -> "EditableTableProgrammaticUpdateGuard":
        self.depth += 1
        self.pending_release = False
        self.programmatic_row_values = None
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.depth > 0:
            self.depth -= 1
        self.pending_release = True
        QTimer.singleShot(0, lambda: self._release_pending())

    @property
    def active(self) -> bool:
        return self.depth > 0

    def remember_rows(self, row_values: tuple[tuple[str, ...], ...]) -> None:
        """Record the table rows produced by the current programmatic update."""

        self.programmatic_row_values = row_values

    def suppress_pending_rows(
        self,
        row_values: tuple[tuple[str, ...], ...],
    ) -> None:
        """Suppress delayed callbacks that still reflect programmatic rows."""

        self.pending_release = True
        self.remember_rows(row_values)
        QTimer.singleShot(0, lambda: self._release_pending())

    def should_suppress_pending_rows(
        self,
        row_values: tuple[tuple[str, ...], ...],
    ) -> bool:
        """Whether a pending callback still reflects the programmatic rows."""

        return self.pending_release and self.programmatic_row_values == row_values

    def release(self) -> None:
        self.pending_release = False
        self.programmatic_row_values = None

    def _release_pending(self) -> None:
        if self.depth == 0:
            self.release()


class EditableTableItem(QTableWidgetItem):
    """QTableWidgetItem with separate logical/edit value and rendered chrome."""

    LOGICAL_VALUE_ROLE = Qt.ItemDataRole.UserRole
    RENDERED_TEXT_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(self, value: str) -> None:
        super().__init__()
        self._logical_value = value
        self._rendered_text = value
        super().setData(Qt.ItemDataRole.DisplayRole, value)

    def data(self, role):
        if role == Qt.ItemDataRole.DisplayRole:
            return self._rendered_text
        if role == Qt.ItemDataRole.EditRole:
            return self._logical_value
        if role == self.LOGICAL_VALUE_ROLE:
            return self._logical_value
        if role == self.RENDERED_TEXT_ROLE:
            return self._rendered_text
        return super().data(role)

    def setData(self, role, value) -> None:
        text = "" if value is None else str(value)
        if role == Qt.ItemDataRole.EditRole:
            self.set_logical_text(text)
            return
        if role == Qt.ItemDataRole.DisplayRole:
            self.set_rendered_text(text)
            return
        if role == self.LOGICAL_VALUE_ROLE:
            self._logical_value = text
            super().setData(self.LOGICAL_VALUE_ROLE, text)
            return
        if role == self.RENDERED_TEXT_ROLE:
            self.set_rendered_text(text)
            return
        super().setData(role, value)

    def setText(self, text: str) -> None:
        self.set_logical_text(text)

    def set_logical_text(self, value: str) -> None:
        self._logical_value = value
        self._rendered_text = value
        super().setData(self.LOGICAL_VALUE_ROLE, value)
        super().setData(self.RENDERED_TEXT_ROLE, value)
        super().setData(Qt.ItemDataRole.DisplayRole, value)

    def set_rendered_text(self, value: str) -> None:
        self._rendered_text = value
        super().setData(self.RENDERED_TEXT_ROLE, value)
        super().setData(Qt.ItemDataRole.DisplayRole, value)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class EditableTableController(Generic[EditableRowT]):
    """Own editable Qt table mechanics for one typed row model."""

    LOGICAL_VALUE_ROLE = EditableTableItem.LOGICAL_VALUE_ROLE
    RENDERED_TEXT_ROLE = EditableTableItem.RENDERED_TEXT_ROLE

    table: QTableWidget
    columns: tuple[EditableTableColumn, ...]
    free_form_cell_specs: FreeFormCellSpecMap
    row_cells: Callable[[EditableRowT], tuple[str, ...]]
    row_from_cells: Callable[[tuple[str, ...]], EditableRowT | None]
    apply_changes: Callable[[], None]
    semantic_binding: EditableTableSemanticBinding | None = None
    update_guard: EditableTableProgrammaticUpdateGuard = field(
        default_factory=EditableTableProgrammaticUpdateGuard,
    )

    def append(self, row_model: EditableRowT) -> None:
        table_signals_blocked = self.table.blockSignals(True)
        try:
            row_index = self.table.rowCount()
            self.table.insertRow(row_index)
            for column, value in zip(
                self.columns,
                self.row_cells(row_model),
                strict=True,
            ):
                self._set_cell(row_index, column, value)
        finally:
            self.table.blockSignals(table_signals_blocked)

    def replace_all(self, row_models: tuple[EditableRowT, ...]) -> bool:
        """Replace table contents; return whether table structure changed."""

        with self.update_guard:
            table_signals_blocked = self.table.blockSignals(True)
            try:
                if self.table.rowCount() == len(row_models):
                    for row_index, row_model in enumerate(row_models):
                        for column, value in zip(
                            self.columns,
                            self.row_cells(row_model),
                            strict=True,
                        ):
                            self._update_cell(row_index, column, value)
                    return False

                self.table.setRowCount(0)
                for row_model in row_models:
                    self.append(row_model)
                return True
            finally:
                self.table.blockSignals(table_signals_blocked)
                self.update_guard.remember_rows(self.row_values())

    def request_apply_changes(self) -> None:
        """Apply edits unless they are fallout from a programmatic table update."""

        if self.update_guard.active:
            return
        self._sync_item_logical_values_from_edit_role()
        if self.update_guard.pending_release:
            if self.update_guard.should_suppress_pending_rows(self.row_values()):
                return
            self.update_guard.release()
        self.apply_changes()

    def rows(self) -> tuple[EditableRowT, ...]:
        rows: list[EditableRowT] = []
        for values in self.row_values():
            row_model = self.row_from_cells(values)
            if row_model is not None:
                rows.append(row_model)
        return tuple(rows)

    def row_values(self) -> tuple[tuple[str, ...], ...]:
        """Return current table cell text values for every row."""

        return tuple(
            tuple(self._cell_text(row_index, column) for column in self.columns)
            for row_index in range(self.table.rowCount())
        )

    def suppress_current_rows_until_idle(self) -> None:
        """Suppress delayed callbacks for the currently rendered row values."""

        self.update_guard.suppress_pending_rows(self.row_values())

    def has_incomplete_rows(self) -> bool:
        """Whether any non-empty row is not yet valid for the row model."""

        for values in self.row_values():
            if any(value.strip() for value in values) and self.row_from_cells(values) is None:
                return True
        return False

    def _sync_item_logical_values_from_edit_role(self) -> None:
        """Update item backing values for a real table edit without re-emitting."""

        table_signals_blocked = self.table.blockSignals(True)
        try:
            for row_index in range(self.table.rowCount()):
                for column in self.columns:
                    column_index = int(column)
                    if self.table.cellWidget(row_index, column_index) is not None:
                        continue
                    item = self.table.item(row_index, column_index)
                    if item is None:
                        continue
                    edit_value = item.data(Qt.ItemDataRole.EditRole)
                    if not isinstance(edit_value, str):
                        edit_value = item.text()
                    if item.data(self.LOGICAL_VALUE_ROLE) == edit_value:
                        continue
                    self._set_item_logical_text(item, edit_value)
        finally:
            self.table.blockSignals(table_signals_blocked)

    def remove_selected(self) -> bool:
        selected_rows = {index.row() for index in self.table.selectedIndexes()}
        if not selected_rows:
            return False
        table_signals_blocked = self.table.blockSignals(True)
        try:
            for row_index in sorted(selected_rows, reverse=True):
                self.table.removeRow(row_index)
        finally:
            self.table.blockSignals(table_signals_blocked)
        return True

    def _cell_text(self, row_index: int, column: EditableTableColumn) -> str:
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
        if item is None:
            return ""
        return self._editable_item(item).data(self.LOGICAL_VALUE_ROLE)

    def _set_cell(
        self,
        row_index: int,
        column: EditableTableColumn,
        value: str,
    ) -> None:
        spec = column.enum_cell_spec()
        free_form_spec = self.free_form_cell_specs.get((type(column), column))
        if spec is None and free_form_spec is None:
            item = EditableTableItem(value)
            self.table.setItem(row_index, int(column), item)
            self._apply_current_placeholder_text_style_to_cell(row_index, column)
            return
        if free_form_spec is not None:
            widget = StructuredSelectorCellWidget(
                values=free_form_spec.values,
                value=value,
                editor_kind=free_form_spec.editor_kind,
                apply_changes=self.request_apply_changes,
                parent=self.table,
            )
            self.table.setCellWidget(
                row_index,
                int(column),
                widget,
            )
            self._apply_current_placeholder_text_style_to_cell(row_index, column)
            return
        combo = NoScrollComboBox(self.table)
        for enum_value in spec.values:
            combo.addItem(spec.text_for_value(enum_value), enum_value)
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(lambda _: self.request_apply_changes())
        combo.activated.connect(lambda _: self.request_apply_changes())
        self.table.setCellWidget(row_index, int(column), combo)
        self._apply_current_placeholder_text_style_to_cell(row_index, column)

    def _update_cell(
        self,
        row_index: int,
        column: EditableTableColumn,
        value: str,
    ) -> None:
        """Update an existing cell in place, rebuilding only on shape mismatch."""

        spec = column.enum_cell_spec()
        free_form_spec = self.free_form_cell_specs.get((type(column), column))
        column_index = int(column)

        if free_form_spec is not None:
            widget = self.table.cellWidget(row_index, column_index)
            if isinstance(widget, StructuredSelectorCellWidget):
                line_edit_signals_blocked = widget.line_edit.blockSignals(True)
                try:
                    widget.line_edit.setText(value)
                finally:
                    widget.line_edit.blockSignals(line_edit_signals_blocked)
                return
            self._set_cell(row_index, column, value)
            return

        if spec is not None:
            widget = self.table.cellWidget(row_index, column_index)
            if isinstance(widget, QComboBox):
                index = self._combo_index_for_cell_value(widget, value)
                combo_signals_blocked = widget.blockSignals(True)
                try:
                    if index >= 0:
                        widget.setCurrentIndex(index)
                    else:
                        self._set_cell(row_index, column, value)
                finally:
                    widget.blockSignals(combo_signals_blocked)
                return
            self._set_cell(row_index, column, value)
            return

        item = self.table.item(row_index, column_index)
        if item is None:
            self._set_cell(row_index, column, value)
            return
        self._set_item_logical_text(item, value)

    @classmethod
    def _set_item_logical_text(cls, item: EditableTableItem, value: str) -> None:
        cls._editable_item(item)
        item.set_logical_text(value)

    @staticmethod
    def _editable_item(item: QTableWidgetItem) -> EditableTableItem:
        if not isinstance(item, EditableTableItem):
            raise TypeError(
                "Editable table text cells must be EditableTableItem instances "
                f"for semantic chrome isolation, got {type(item).__name__}."
            )
        return item

    def _apply_current_placeholder_text_style_to_cell(
        self,
        row_index: int,
        column: EditableTableColumn,
    ) -> None:
        """Apply the table's current inherited-preview style to a newly built cell."""

        active = self.table.property("placeholder_text_style_active") is True
        column_index = int(column)
        widget = self.table.cellWidget(row_index, column_index)
        if widget is not None:
            self._apply_widget_placeholder_text_style(widget, active)
            return
        item = self.table.item(row_index, column_index)
        if item is not None:
            self._apply_item_placeholder_text_style(item, active)

    @staticmethod
    def _combo_index_for_cell_value(combo: QComboBox, value: str) -> int:
        """Return the row whose logical enum/text value matches ``value``."""

        for index in range(combo.count()):
            payload = combo.itemData(index)
            if isinstance(payload, Enum):
                if str(payload.value) == value:
                    return index
                continue
            if combo.itemText(index) == value:
                return index
        return -1

    def apply_semantic_index(
        self,
        semantic_index: ObjectStateSubfieldSemanticIndex,
    ) -> None:
        """Apply ObjectState structural semantics to table cell chrome."""

        from objectstate.time_travel_profile import TimeTravelProfiler

        with TimeTravelProfiler.phase(
            "openhcs.editable_table.apply_semantic_index",
            rows=self.table.rowCount(),
            columns=self.table.columnCount(),
        ):
            with self.update_guard:
                table_signals_blocked = self.table.blockSignals(True)
                try:
                    for (row_index, column_index), relative_path in self.semantic_paths().items():
                        semantic = semantic_index.leaf_for(relative_path)
                        if semantic is None:
                            continue
                        widget = self.table.cellWidget(row_index, column_index)
                        if widget is not None:
                            self._apply_widget_semantic(widget, semantic)
                            continue
                        item = self.table.item(row_index, column_index)
                        if item is not None:
                            self._apply_item_semantic(item, semantic)
                finally:
                    self.table.blockSignals(table_signals_blocked)
                    self.update_guard.remember_rows(self.row_values())

    def set_placeholder_text_style(self, active: bool) -> None:
        """Apply inherited-preview text styling to every rendered table cell."""

        with self.update_guard:
            self.apply_placeholder_text_style_to_table(self.table, active)
            self.update_guard.remember_rows(self.row_values())

    @classmethod
    def apply_placeholder_text_style_to_table(
        cls,
        table: QTableWidget,
        active: bool,
    ) -> None:
        """Apply or clear placeholder text styling without dimming table controls."""

        table_signals_blocked = table.blockSignals(True)
        try:
            table.setProperty("placeholder_text_style_active", active)
            for row_index in range(table.rowCount()):
                for column_index in range(table.columnCount()):
                    widget = table.cellWidget(row_index, column_index)
                    if widget is not None:
                        cls._apply_widget_placeholder_text_style(widget, active)
                        continue
                    item = table.item(row_index, column_index)
                    if item is not None:
                        cls._apply_item_placeholder_text_style(item, active)
        finally:
            table.blockSignals(table_signals_blocked)

    def semantic_paths(self) -> Mapping[tuple[int, int], StructuralValuePath]:
        """Return structural paths for all currently rendered cells."""

        if self.semantic_binding is None:
            return MappingProxyType({})
        return MappingProxyType(
            {
                (row_index, int(column)): self.semantic_binding.relative_path_for_cell(
                    row_index,
                    column_index,
                )
                for row_index in range(self.table.rowCount())
                for column_index, column in enumerate(self.columns)
            }
        )

    def cell_target_for_semantic_path(
        self,
        relative_path: StructuralValuePath,
    ) -> StructuralTableCellTarget | None:
        """Return a concrete target for a structural cell path."""

        for (row_index, column_index), candidate in self.semantic_paths().items():
            if candidate != relative_path:
                continue
            return StructuralTableCellTarget(
                table=self.table,
                row_index=row_index,
                column_index=column_index,
                cell_widget=self.table.cellWidget(row_index, column_index),
            )
        return None

    @staticmethod
    def _semantic_tooltip(semantic) -> str:
        reasons: list[str] = [semantic.display_path]
        if semantic.dirty:
            reasons.append("* differs from saved value")
        if semantic.signature_diff:
            reasons.append("_ differs from signature default")
        if semantic.inherited_value:
            reasons.append("inherited resolved value")
        return "\n".join(reasons)

    @classmethod
    def _apply_item_semantic(cls, item: QTableWidgetItem, semantic) -> None:
        item = cls._editable_item(item)
        logical_value = item.data(cls.LOGICAL_VALUE_ROLE)
        if not isinstance(logical_value, str):
            edit_value = item.data(Qt.ItemDataRole.EditRole)
            logical_value = edit_value if isinstance(edit_value, str) else item.text()
            item.setData(cls.LOGICAL_VALUE_ROLE, logical_value)
        target_text = f"*{logical_value}" if semantic.dirty else logical_value
        target_tooltip = cls._semantic_tooltip(semantic)
        font = item.font()
        if (
            item.text() == target_text
            and item.data(Qt.ItemDataRole.EditRole) == logical_value
            and item.data(cls.RENDERED_TEXT_ROLE) == target_text
            and font.underline() == semantic.signature_diff
            and item.toolTip() == target_tooltip
        ):
            return
        item.set_rendered_text(target_text)
        font.setUnderline(semantic.signature_diff)
        item.setFont(font)
        item.setToolTip(target_tooltip)

    @classmethod
    def _apply_widget_semantic(cls, widget: QWidget, semantic) -> None:
        target_tooltip = cls._semantic_tooltip(semantic)
        font = widget.font()
        combo_dirty_matches = (
            not isinstance(widget, QComboBox)
            or cls._combo_current_dirty_marker_matches(widget, semantic.dirty)
        )
        if (
            widget.property("objectstate_dirty") == semantic.dirty
            and widget.property("objectstate_signature_diff") == semantic.signature_diff
            and widget.property("objectstate_inherited") == semantic.inherited_value
            and widget.toolTip() == target_tooltip
            and font.underline() == semantic.signature_diff
            and combo_dirty_matches
        ):
            return
        if isinstance(widget, QComboBox):
            cls._apply_combo_dirty_marker(widget, semantic.dirty)
        widget.setProperty("objectstate_dirty", semantic.dirty)
        widget.setProperty("objectstate_signature_diff", semantic.signature_diff)
        widget.setProperty("objectstate_inherited", semantic.inherited_value)
        widget.setToolTip(target_tooltip)
        font.setUnderline(semantic.signature_diff)
        widget.setFont(font)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    @staticmethod
    def _combo_current_dirty_marker_matches(combo: QComboBox, dirty: bool) -> bool:
        rendered_index = combo.currentIndex() if dirty else -1
        return (
            combo.property("objectstate_rendered_dirty") == dirty
            and combo.property("objectstate_rendered_dirty_index") == rendered_index
        )

    @classmethod
    def _apply_combo_dirty_marker(cls, combo: QComboBox, dirty: bool) -> None:
        was_blocked = combo.blockSignals(True)
        try:
            cls._restore_combo_item_texts(combo)
            if dirty and combo.currentIndex() >= 0:
                current_index = combo.currentIndex()
                combo.setItemText(
                    current_index,
                    f"*{cls._combo_logical_text(combo, current_index)}",
                )
                combo.setProperty("objectstate_rendered_dirty_index", current_index)
            else:
                combo.setProperty("objectstate_rendered_dirty_index", -1)
            combo.setProperty("objectstate_rendered_dirty", dirty)
        finally:
            combo.blockSignals(was_blocked)

    @classmethod
    def _restore_combo_item_texts(cls, combo: QComboBox) -> None:
        for index in range(combo.count()):
            combo.setItemText(index, cls._combo_logical_text(combo, index))

    @staticmethod
    def _combo_logical_text(combo: QComboBox, index: int) -> str:
        value = combo.itemData(index)
        if isinstance(value, Enum):
            return str(value.value)
        return combo.itemText(index)

    @classmethod
    def _apply_item_placeholder_text_style(
        cls,
        item: QTableWidgetItem,
        active: bool,
    ) -> None:
        font = item.font()
        font.setItalic(active)
        item.setFont(font)
        item.setForeground(
            QBrush(QColor(PlaceholderConfig.text_color_name()))
            if active
            else QBrush()
        )

    @classmethod
    def _apply_widget_placeholder_text_style(
        cls,
        widget: QWidget,
        active: bool,
    ) -> None:
        if isinstance(widget, StructuredSelectorCellWidget):
            cls._apply_line_edit_placeholder_text_style(widget.line_edit, active)
            return
        if isinstance(widget, QLineEdit):
            cls._apply_line_edit_placeholder_text_style(widget, active)
            return
        if isinstance(widget, QComboBox):
            widget.setStyleSheet(
                f"QComboBox {{ {PlaceholderConfig.text_style(include_opacity=False)} }}"
                if active
                else ""
            )

    @staticmethod
    def _apply_line_edit_placeholder_text_style(
        line_edit: QLineEdit,
        active: bool,
    ) -> None:
        line_edit.setStyleSheet(
            f"QLineEdit {{ {PlaceholderConfig.text_style(include_opacity=False)} }}"
            if active
            else ""
        )


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
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(22)
        table.verticalHeader().setMinimumSectionSize(18)
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
        table.setFixedHeight(header_height + row_height + scrollbar_height + frame + 4)


class StepBindingsTableEditor(QWidget):
    """Typed table editor for step-local source bindings."""

    changed = pyqtSignal()

    def __init__(
        self,
        *,
        bindings: tuple[NamedSourceBinding, ...],
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
            free_form_cell_specs=free_form_cell_specs,
            row_cells=EditableSourceBindingRow.cells,
            row_from_cells=EditableSourceBindingRow.from_cells,
            apply_changes=self._emit_changed,
        )
        for binding in bindings:
            self.controller.append(EditableSourceBindingRow.from_binding(binding))
        self.table.itemChanged.connect(lambda _: self.controller.request_apply_changes())
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
            binding or NamedSourceBinding(alias="NewSource"),
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

    def bindings(self) -> tuple[NamedSourceBinding, ...]:
        return tuple(row.binding for row in self.controller.rows())

    def _emit_changed(self) -> None:
        if self._updating_ui:
            return
        self.changed.emit()


class StepBindingsDialog(QDialog):
    """Modal editor for the large step-bindings table."""

    def __init__(
        self,
        *,
        bindings: tuple[NamedSourceBinding, ...],
        free_form_cell_specs: FreeFormCellSpecMap,
        scope_color_scheme: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit step source bindings")
        self.editor = StepBindingsTableEditor(
            bindings=bindings,
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

    def bindings(self) -> tuple[NamedSourceBinding, ...]:
        return self.editor.bindings()


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

    binding: NamedSourceBinding

    @classmethod
    def from_cells(cls, values: tuple[str, ...]) -> "EditableSourceBindingRow | None":
        (
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
        binding: NamedSourceBinding,
    ) -> "EditableSourceBindingRow":
        return cls(binding=binding)

    def cells(self) -> tuple[str, ...]:
        return (
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
class EditableSourceFilterRow:
    """Nominal row model for editing one source filter clause."""

    clause: SourceFilterClause

    @classmethod
    def from_cells(cls, values: tuple[str, ...]) -> "EditableSourceFilterRow | None":
        subject, match_type, value = (cell.strip() for cell in values)
        if not subject and not match_type and not value:
            return None
        match_type_value = SourceFilterMatchType(
            match_type or SourceFilterMatchType.IS_IMAGE.value
        )
        value_or_none = value or None
        if match_type_value.requires_value and value_or_none is None:
            return None
        return cls(
            clause=SourceFilterClause(
                subject=SourceFilterSubject(subject or SourceFilterSubject.FILE.value),
                match_type=match_type_value,
                value=value_or_none,
            )
        )

    @classmethod
    def from_clause(cls, clause: SourceFilterClause) -> "EditableSourceFilterRow":
        return cls(clause=clause)

    def cells(self) -> tuple[str, ...]:
        return (
            self.clause.subject.value,
            self.clause.match_type.value,
            self.clause.value or "",
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


@dataclass(frozen=True, slots=True)
class SourceBindingsEditorValue:
    """Validated editor bridge between raw lazy configs and concrete source-binding views."""

    raw: SourceBindingsEditorRawValue

    def __post_init__(self) -> None:
        self.base_type()

    def base_type(self) -> type[SourceBindingsConfig]:
        value_type = type(self.raw)
        base_type = get_base_type_for_lazy(value_type) or value_type
        if not issubclass(base_type, SourceBindingsConfig):
            raise TypeError(
                "SourceBindingsEditorWidget requires SourceBindingsConfig, "
                f"got {value_type.__name__}."
            )
        return base_type

    def raw_field_value(self, field_name: str) -> object:
        """Return one source-binding field without triggering lazy resolution."""

        return DataclassFieldAccess.raw_value(self.raw, field_name)

    @property
    def is_enableable(self) -> bool:
        return is_enableable(self.base_type())

    @property
    def enabled_value(self) -> bool | None:
        if not self.is_enableable:
            return None
        return cast(
            bool | None,
            self.raw_field_value(Enableable.require_parameter_name()),
        )

    @property
    def source_filter_declarations(self) -> tuple[SourceFilterClause, ...]:
        return tuple(self.raw_field_value("source_filters") or ())

    @property
    def binding_declarations(self) -> tuple[NamedSourceBinding, ...]:
        return tuple(self.raw_field_value("bindings") or ())

    @property
    def metadata_rule_declarations(self) -> tuple[MetadataExtractionRule, ...]:
        return tuple(self.raw_field_value("metadata_rules") or ())

    @property
    def match_plan(self) -> SourceBindingMatchPlan | None:
        return cast(
            SourceBindingMatchPlan | None,
            self.raw_field_value("match_plan"),
        )

    def concrete_view(self) -> SourceBindingsConfig:
        if isinstance(self.raw, SourceBindingsConfig):
            return self.raw
        config_type = self.base_type()
        return config_type(
            **DataclassFieldAccess.raw_init_values(self.raw, config_type)
        )


class SourceBindingsEditorWidget(
    QWidget,
    ValueGettable,
    ValueSettable,
    ResolvedValuePreviewSettable,
    RawResolvedValueSettable,
    ChildFieldChromeRefreshable,
    ChildFieldIdentityProvider,
    ChildFieldNavigationTargetProvider,
    ChildFieldSemanticChromeRefreshable,
    ChildSubfieldNavigationTargetProvider,
    InlineDataclassGroupBoxChromeProvider,
    InlineDataclassRootResettable,
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
        bindings: SourceBindingsEditorRawValue | None = None,
        display_bindings: SourceBindingsEditorRawValue | None = None,
        inventory: SourceInventory | None = None,
        form_context: InlineDataclassFormContext | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._schema = schema
        self._bindings = bindings if bindings is not None else StepSourceBindingsConfig()
        self._display_bindings = (
            display_bindings if display_bindings is not None else self._bindings
        )
        self._inventory = inventory
        self._form_context = form_context
        self._child_chrome = (
            InlineDataclassChildChrome(form_context)
            if form_context is not None
            else None
        )
        self._change_signal_callbacks: dict[
            Callable[[SourceBindingsEditorRawValue], None],
            Callable[[], None],
        ] = {}
        self._updating_ui = False
        self.step_bindings_summary_table: QTableWidget | None = None
        self.step_bindings_table: QTableWidget | None = None
        self.step_bindings_editor: StepBindingsTableEditor | None = None
        self.source_filters_table: QTableWidget | None = None
        self.metadata_rules_table: QTableWidget | None = None
        self.match_plan_table: QTableWidget | None = None
        self._inline_groupbox: object | None = None
        self._enabled_checkbox: NoneAwareCheckBox | None = None
        self._enabled_title_widget: QWidget | None = None
        self._enabled_reset_button: QPushButton | None = None
        self._enabled_provenance_button: QWidget | None = None
        self._updating_enableable_chrome = False
        self.source_filters_controller: (
            EditableTableController[EditableSourceFilterRow] | None
        ) = None
        self.metadata_rules_controller: (
            EditableTableController[EditableMetadataRuleRow] | None
        ) = None
        self.match_plan_controller: (
            EditableTableController[EditableMatchPlanRow] | None
        ) = None
        self._editable_table_controllers: list[EditableTableController] = []
        self._structural_table_controllers: list[EditableTableController] = []
        self._table_placeholder_chrome_state: dict[str, bool] = {}
        self._scope_color_scheme = None
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(3)
        self.empty_label = QLabel("No source bindings loaded")
        self.layout.addWidget(self.empty_label)
        if view_model is not None:
            self.set_view_model(view_model)

    @classmethod
    def from_bindings(
        cls,
        bindings: SourceBindingsEditorRawValue,
        *,
        display_bindings: SourceBindingsEditorRawValue | None = None,
        schema: PipelineImageSchema = EMPTY_PIPELINE_IMAGE_SCHEMA,
        inventory: SourceInventory | None = None,
        form_context: InlineDataclassFormContext | None = None,
        parent: QWidget | None = None,
    ) -> "SourceBindingsEditorWidget":
        """Create an editor from typed source bindings and an optional schema."""

        table_bindings = display_bindings if display_bindings is not None else bindings
        return cls(
            SourceBindingsViewModel.from_schema_and_bindings(
                schema=schema,
                bindings=SourceBindingsEditorValue(table_bindings).concrete_view(),
            ),
            schema=schema,
            bindings=bindings,
            display_bindings=table_bindings,
            inventory=inventory,
            form_context=form_context,
            parent=parent,
        )

    @property
    def value(self) -> SourceBindingsEditorRawValue:
        """Value property used by pyqt-reactive signal adapters."""

        return self.get_value()

    def get_value(self) -> SourceBindingsEditorRawValue:
        """Return the current typed source-bindings config."""

        return self._bindings

    def set_value(self, value: SourceBindingsEditorRawValue | None) -> None:
        """Update the widget from a typed source-bindings config."""

        from objectstate.time_travel_profile import TimeTravelProfiler

        bindings = value if value is not None else type(self._bindings)()
        SourceBindingsEditorValue(bindings)
        with TimeTravelProfiler.phase(
            "openhcs.source_bindings.set_value",
            widget=type(self).__qualname__,
        ):
            if bindings == self._bindings:
                if bindings != self._display_bindings:
                    if self._rendered_bindings_equal(bindings, self._display_bindings):
                        self._display_bindings = bindings
                        self.refresh_child_field_chrome()
                        return
                    self._display_bindings = bindings
                    self.refresh()
                    return
                self.refresh_child_field_chrome()
                return
            if self._refresh_enableable_only(bindings, bindings):
                return
            if self._refresh_source_filters_only(bindings):
                return
            self._bindings = bindings
            self._display_bindings = bindings
            self.refresh()

    def set_resolved_value_preview(self, value: SourceBindingsEditorRawValue) -> None:
        """Update inherited/resolved display without changing the raw edit value."""

        SourceBindingsEditorValue(value)
        if value == self._display_bindings:
            self.refresh_child_field_chrome()
            return
        if self._rendered_bindings_equal(value, self._display_bindings):
            self._display_bindings = value
            self.refresh_child_field_chrome()
            return
        if self._refresh_enableable_only(self._bindings, value):
            return
        if self._refresh_source_filters_only(value, update_raw=False):
            return
        self._display_bindings = value
        self.refresh()

    def set_raw_value_with_resolved_preview(
        self,
        raw_value: SourceBindingsEditorRawValue | None,
        resolved_value: SourceBindingsEditorRawValue | None,
    ) -> None:
        """Update raw edit value and resolved display in one render pass."""

        raw_bindings = (
            raw_value if raw_value is not None else type(self._bindings)()
        )
        display_bindings = (
            resolved_value if resolved_value is not None else raw_bindings
        )
        SourceBindingsEditorValue(raw_bindings)
        SourceBindingsEditorValue(display_bindings)

        if self._refresh_enableable_only(raw_bindings, display_bindings):
            return

        if self._refresh_source_filters_only(
            display_bindings,
            raw_bindings=raw_bindings,
            refresh_chrome=False,
        ):
            return

        self._bindings = raw_bindings
        self._display_bindings = display_bindings
        self.refresh()

    def refresh_child_field_chrome(
        self,
        owner_field_paths: tuple[DottedFieldPath, ...] | None = None,
    ) -> None:
        """Refresh child labels and reset buttons after ObjectState changes."""

        self.refresh_section_label_markers(owner_field_paths)

    def reset_inline_dataclass_fields(self) -> None:
        """Reset SourceBindings child fields through the inline form context."""

        if self._form_context is None:
            raise RuntimeError(
                "SourceBindingsEditorWidget requires an InlineDataclassFormContext "
                "for root reset-all behavior."
            )
        self._form_context.reset_children()

    def child_field_semantic_owner_paths(self) -> tuple[DottedFieldPath, ...]:
        """Return SourceBindings child fields with structural table semantics."""

        if self._form_context is None:
            return ()
        return tuple(
            self._form_context.child_path(binding.owner_field_name)
            for binding in self._structural_table_bindings()
        )

    def refresh_child_field_semantics(
        self,
        owner_field_path: DottedFieldPath,
        semantic_index: ObjectStateSubfieldSemanticIndex,
    ) -> None:
        """Apply ObjectState-derived child structural semantics."""

        if self._form_context is None:
            return
        for controller, binding in self._structural_table_controller_bindings():
            if owner_field_path != self._form_context.child_path(binding.owner_field_name):
                continue
            controller.apply_semantic_index(semantic_index)
            return

    def _refresh_source_filters_only(
        self,
        bindings: SourceBindingsEditorRawValue,
        *,
        update_raw: bool = True,
        raw_bindings: SourceBindingsEditorRawValue | None = None,
        refresh_chrome: bool = True,
    ) -> bool:
        """Update the source-filter table without rebuilding sibling sections."""

        from objectstate.time_travel_profile import TimeTravelProfiler

        if self.source_filters_table is None or self.source_filters_controller is None:
            return False

        with TimeTravelProfiler.phase(
            "openhcs.source_bindings.refresh_source_filters_only",
            update_raw=update_raw,
        ):
            current = SourceBindingsEditorValue(self._display_bindings)
            incoming = SourceBindingsEditorValue(bindings)
            if (
                current.source_filter_declarations
                == incoming.source_filter_declarations
            ):
                return False
            if (
                current.binding_declarations != incoming.binding_declarations
                or current.metadata_rule_declarations != incoming.metadata_rule_declarations
                or current.match_plan != incoming.match_plan
            ):
                return False

            if update_raw:
                self._bindings = raw_bindings if raw_bindings is not None else bindings
            self._display_bindings = bindings
            self._updating_ui = True
            try:
                structure_changed = self.source_filters_controller.replace_all(
                    tuple(
                        EditableSourceFilterRow.from_clause(clause)
                        for clause in incoming.source_filter_declarations
                    )
                )
            finally:
                self._updating_ui = False
            if structure_changed:
                self._fit_table_to_rows(self.source_filters_table)
            if refresh_chrome:
                self._refresh_single_child_field_chrome("source_filters")
            return True

    def _refresh_enableable_only(
        self,
        raw_bindings: SourceBindingsEditorRawValue,
        display_bindings: SourceBindingsEditorRawValue,
    ) -> bool:
        """Update enableable chrome without rebuilding source-binding tables."""

        if not is_enableable(SourceBindingsEditorValue(raw_bindings).base_type()):
            return False
        if not self._rendered_sections_equal(display_bindings, self._display_bindings):
            return False

        self._bindings = raw_bindings
        self._display_bindings = display_bindings
        self._sync_enableable_chrome()
        self._refresh_single_child_field_chrome(self._enabled_field_name())
        return True

    def _refresh_single_child_field_chrome(self, field_name: str) -> None:
        """Refresh child chrome for one source-binding field."""

        if self._form_context is None:
            self.refresh_child_field_chrome()
            return
        self.refresh_child_field_chrome(
            (self._form_context.child_path(field_name),),
        )

    def _rendered_bindings_equal(
        self,
        left: SourceBindingsEditorRawValue,
        right: SourceBindingsEditorRawValue,
    ) -> bool:
        """Return whether two configs render the same source-binding editor rows."""

        return self._rendered_sections_equal(
            left,
            right,
        ) and (
            SourceBindingsEditorValue(left).enabled_value
            == SourceBindingsEditorValue(right).enabled_value
        )

    def _rendered_sections_equal(
        self,
        left: SourceBindingsEditorRawValue,
        right: SourceBindingsEditorRawValue,
    ) -> bool:
        """Return whether two configs render the same non-enableable sections."""

        left_value = SourceBindingsEditorValue(left)
        right_value = SourceBindingsEditorValue(right)
        return (
            left_value.binding_declarations == right_value.binding_declarations
            and left_value.source_filter_declarations
            == right_value.source_filter_declarations
            and left_value.metadata_rule_declarations
            == right_value.metadata_rule_declarations
            and left_value.match_plan == right_value.match_plan
        )

    def child_field_navigation_target(self, field_name: str) -> QWidget | None:
        """Return the rendered section widget for a source-binding child field."""

        if self._form_context is None:
            return None
        if self._child_chrome is None:
            return None
        return self._child_chrome.navigation_target(field_name)

    def child_subfield_navigation_target(
        self,
        child_identity: InlineDataclassChildFieldIdentity,
        relative_path: StructuralValuePath,
    ) -> StructuralFlashTarget | None:
        """Return the rendered target for one structural child cell."""

        if self._form_context is None:
            return None
        if (
            child_identity.field_name == self._enabled_field_name()
            and not relative_path.segments
        ):
            return self._enabled_structural_target()
        section_group = self.child_field_navigation_target(child_identity.field_name)
        if section_group is None:
            return None
        if not relative_path.segments:
            label_widget = self.child_field_label(child_identity.field_name)
            return StructuralMaskedContainerTarget(
                container=self,
                masked_target=StructuralDescendantMaskTarget(section_group),
                label_widget=label_widget,
                scroll_target=StructuralWidgetTarget(label_widget),
            )
        controller = self._structural_table_controller_for_field(
            child_identity.field_name
        )
        if controller is None:
            return None
        cell_target = controller.cell_target_for_semantic_path(relative_path)
        if cell_target is None:
            return None
        return StructuralMaskedContainerTarget(
            container=section_group,
            masked_target=cell_target,
            label_widget=self.child_field_label(child_identity.field_name),
        )

    def child_field_identity(self, field_name: str) -> InlineDataclassChildFieldIdentity:
        """Return the nominal ObjectState identity for one source-binding child."""

        if self._child_chrome is None:
            raise RuntimeError("Source binding child identity requires form context.")
        return self._child_chrome.child_identity(field_name)

    def child_field_label(self, field_name: str):
        """Return the rendered label for one source-binding child field."""

        if self._child_chrome is None:
            raise RuntimeError("Source binding child label requires form context.")
        return self._child_chrome.labels[self.child_field_identity(field_name)]

    def child_field_reset_button(self, field_name: str) -> QPushButton:
        """Return the rendered reset button for one source-binding child field."""

        if self._child_chrome is None:
            raise RuntimeError("Source binding child reset button requires form context.")
        return self._child_chrome.reset_buttons[self.child_field_identity(field_name)]

    def child_field_section_group(self, field_name: str) -> QWidget:
        """Return the rendered section group for one source-binding child field."""

        target = self.child_field_navigation_target(field_name)
        if target is None:
            raise RuntimeError(f"Source binding child section {field_name!r} is not rendered.")
        return target

    def _enabled_structural_target(self) -> StructuralFlashTarget | None:
        """Return the structural target for inline enableable title chrome."""

        if not isinstance(self._inline_groupbox, InlineDataclassGroupBox):
            return None
        if self._enabled_checkbox is None:
            return None
        return EnabledTitleWidgetMoveAuthority.title_field_flash_target(
            container=self._inline_groupbox,
            title_widget=self._enabled_title_widget or self._enabled_checkbox,
            checkbox_widget=self._enabled_checkbox,
            reset_button=self._enabled_reset_button,
            provenance_button=self._enabled_provenance_button,
        )

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
                bindings=SourceBindingsEditorValue(self._display_bindings).concrete_view(),
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
                "Pipeline Image Schema Bindings",
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
        self.layout.addWidget(self._source_filters_group())
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
                bindings=self._display_bindings,
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
        self.refresh_section_label_markers()
        for controller in self._editable_table_controllers:
            controller.suppress_current_rows_until_idle()
        self._updating_ui = False

    def connect_change_signal(self, callback: Callable[[SourceBindingsEditorRawValue], None]) -> None:
        """Implement ChangeSignalEmitter for pyqt-reactive inline dataclass forms."""

        def delegated_callback() -> None:
            callback(self.get_value())

        self._change_signal_callbacks[callback] = delegated_callback
        self.changed.connect(delegated_callback)

    def disconnect_change_signal(self, callback: Callable[[SourceBindingsEditorRawValue], None]) -> None:
        """Disconnect a previously registered change callback."""

        delegated_callback = self._change_signal_callbacks.pop(callback, None)
        if delegated_callback is None:
            return
        try:
            self.changed.disconnect(delegated_callback)
        except TypeError:
            pass

    def set_scope_color_scheme(self, scheme) -> None:
        """Apply scope styling to every source-binding table."""

        self._scope_color_scheme = scheme
        for table in self.findChildren(ScopedTableWidget):
            table.set_scope_color_scheme(scheme)

    def configure_inline_dataclass_groupbox(
        self,
        groupbox: InlineDataclassGroupBox,
    ) -> None:
        """Attach source-binding enableable chrome to the inline groupbox title."""

        self._inline_groupbox = groupbox
        if not self._config_type_is_enableable():
            return
        if self._enabled_checkbox is not None:
            return

        enableable_title_authority = EnabledTitleWidgetMoveAuthority()
        checkbox = NoneAwareCheckBox(groupbox)
        checkbox.setToolTip("Enable step source bindings")
        checkbox.toggled.connect(self._on_enabled_checkbox_toggled)
        title_widget = enableable_title_authority.wrap_checkbox_widget_for_title(
            checkbox,
            groupbox.color_scheme,
        )
        enableable_title_authority.bind_title_label_to_checkbox(groupbox, checkbox)

        reset_button = QPushButton("Reset", groupbox)
        reset_button.setToolTip("Reset enabled to default")
        enableable_title_authority.prepare_title_reset_button(
            reset_button,
            groupbox.color_scheme,
        )
        reset_button.clicked.connect(self._reset_enabled_field)

        provenance_button = None
        if self._form_context is not None:
            provenance_button = enableable_title_authority.create_title_provenance_button(
                state=self._form_context.state,
                dotted_path=self._form_context.child_path(self._enabled_field_name()).value,
                color_scheme=groupbox.color_scheme,
            )

        self._enabled_checkbox = checkbox
        self._enabled_title_widget = title_widget
        self._enabled_reset_button = reset_button
        self._enabled_provenance_button = provenance_button
        groupbox.addEnableableWidgets(title_widget, reset_button, provenance_button)
        self._sync_enableable_chrome()

    def _config_type_is_enableable(self) -> bool:
        return is_enableable(SourceBindingsEditorValue(self._bindings).base_type())

    def _enabled_field_name(self) -> str:
        return Enableable.require_parameter_name()

    def _enabled_value(
        self,
        value: SourceBindingsEditorRawValue,
    ) -> bool | None:
        if not is_enableable(value):
            raise TypeError(
                f"{type(value).__name__} is not an Enableable source-binding config."
            )
        enabled = cast(Enableable, value).enabled
        return None if enabled is None else bool(enabled)

    def _sync_enableable_chrome(self) -> None:
        if not self._config_type_is_enableable():
            InlineDataclassChildChrome.set_widget_dimmed(self, False)
            return

        effective_enabled = bool(self._enabled_value(self._display_bindings))
        raw_enabled = self._enabled_value(self._bindings)
        if self._enabled_checkbox is not None:
            self._updating_enableable_chrome = True
            try:
                if raw_enabled is None:
                    self._enabled_checkbox.set_value(None)
                    self._enabled_checkbox.set_placeholder_preview(effective_enabled)
                else:
                    self._enabled_checkbox.set_value(raw_enabled)
            finally:
                self._updating_enableable_chrome = False
        if self._enabled_reset_button is not None and self._form_context is not None:
            self._form_context.update_reset_button_styling(
                self._enabled_reset_button,
                self._enabled_field_name(),
            )
        InlineDataclassChildChrome.set_widget_dimmed(self, not effective_enabled, 0.4)

    def _on_enabled_checkbox_toggled(self, checked: bool) -> None:
        if self._updating_enableable_chrome:
            return
        self._set_enabled_value(bool(checked))

    def _set_enabled_value(self, enabled: bool) -> None:
        field_name = self._enabled_field_name()
        self._bindings = replace_raw(self._bindings, **{field_name: enabled})
        self._display_bindings = replace_raw(
            self._display_bindings,
            **{field_name: enabled},
        )
        self._sync_enableable_chrome()
        self.changed.emit()

    def _reset_enabled_field(self) -> None:
        field_name = self._enabled_field_name()
        if self._form_context is not None:
            self._form_context.reset_child(field_name)
            return
        default_value = cast(
            Enableable,
            SourceBindingsEditorValue(self._bindings).base_type()(),
        ).enabled
        self._bindings = replace_raw(self._bindings, **{field_name: default_value})
        self._display_bindings = replace_raw(
            self._display_bindings,
            **{field_name: default_value},
        )
        self._sync_enableable_chrome()
        self.changed.emit()

    def add_binding_row(self, binding: NamedSourceBinding | None = None) -> None:
        """Append one editable source binding row to the active binding table."""

        if self.step_bindings_table is None:
            self._append_step_binding(binding or NamedSourceBinding(alias="NewSource"))
            return
        if self.step_bindings_editor is None:
            self._append_step_binding(binding or NamedSourceBinding(alias="NewSource"))
            return
        self.step_bindings_editor.add_binding_row(binding)
        self._apply_step_bindings(self.step_bindings_editor.bindings())

    def add_source_filter_row(
        self,
        clause: SourceFilterClause | None = None,
    ) -> None:
        """Append one editable source-universe filter clause row."""

        if self.source_filters_table is None:
            return
        if self.source_filters_controller is None:
            raise RuntimeError("Source filters table controller is not initialized.")
        self._updating_ui = True
        try:
            self.source_filters_controller.append(
                EditableSourceFilterRow.from_clause(
                    clause
                    or SourceFilterClause(
                        SourceFilterSubject.FILE,
                        SourceFilterMatchType.IS_IMAGE,
                    )
                )
            )
        finally:
            self._updating_ui = False
        self._fit_table_to_rows(self.source_filters_table)
        self._apply_source_filters_table()

    def remove_selected_source_filter_rows(self) -> None:
        """Remove selected source-universe filter clause rows."""

        if self.source_filters_table is None:
            return
        if self.source_filters_controller is None:
            raise RuntimeError("Source filters table controller is not initialized.")
        if not self.source_filters_controller.remove_selected():
            return
        self._fit_table_to_rows(self.source_filters_table)
        self._apply_source_filters_table()

    def remove_selected_binding_rows(self) -> None:
        """Remove selected source binding rows from the open dialog editor."""

        if self.step_bindings_editor is None:
            return
        self.step_bindings_editor.remove_selected_binding_rows()
        self._apply_step_bindings(self.step_bindings_editor.bindings())

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
        self._table_placeholder_chrome_state.clear()
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.step_bindings_summary_table = None
        self.step_bindings_table = None
        self.step_bindings_editor = None
        self.source_filters_table = None
        self.metadata_rules_table = None
        self.match_plan_table = None
        self.source_filters_controller = None
        self.metadata_rules_controller = None
        self.match_plan_controller = None
        self._editable_table_controllers.clear()
        self._structural_table_controllers.clear()
        if self._form_context is not None:
            self._child_chrome = InlineDataclassChildChrome(self._form_context)

    def _register_editable_table_controller(
        self,
        controller: EditableTableController[EditableRowT],
    ) -> EditableTableController[EditableRowT]:
        """Track a rendered editable table controller for lifecycle guards."""

        self._editable_table_controllers.append(controller)
        return controller

    def _register_structural_table_controller(
        self,
        controller: EditableTableController[EditableRowT],
    ) -> EditableTableController[EditableRowT]:
        """Track a rendered table controller that owns structural child semantics."""

        if controller.semantic_binding is None:
            raise ValueError(
                "Structural table controllers must declare an "
                "EditableTableSemanticBinding."
            )
        controller = self._register_editable_table_controller(controller)
        self._structural_table_controllers.append(controller)
        return controller

    def _structural_table_bindings(
        self,
    ) -> tuple[EditableTableSemanticBinding, ...]:
        return tuple(
            binding
            for _, binding in self._structural_table_controller_bindings()
        )

    def _structural_table_controller_bindings(
        self,
    ) -> tuple[tuple[EditableTableController, EditableTableSemanticBinding], ...]:
        return tuple(
            (controller, binding)
            for controller in self._structural_table_controllers
            if (binding := controller.semantic_binding) is not None
        )

    def _structural_table_controller_for_field(
        self,
        field_name: str,
    ) -> EditableTableController | None:
        for controller, binding in self._structural_table_controller_bindings():
            if binding.owner_field_name == field_name:
                return controller
        return None

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

    @staticmethod
    def _compact_button(button: QPushButton) -> QPushButton:
        button.setFixedHeight(min(CURRENT_LAYOUT.button_height, 24))
        return button

    def _step_bindings_group(self, view_model: SourceBindingsViewModel) -> QGroupBox:
        group, layout = self._section_group("Bindings", "bindings")
        summary_table = self._create_table(0, 3)
        self.step_bindings_summary_table = summary_table
        summary_table.setHorizontalHeaderLabels(("Bindings", "Aliases", "Origins"))
        for row_index, row in enumerate(self._binding_summary_rows(view_model)):
            summary_table.insertRow(row_index)
            for column_index, value in enumerate(row):
                summary_table.setItem(row_index, column_index, QTableWidgetItem(value))
        summary_table.resizeColumnsToContents()
        self._configure_table(summary_table)
        self._fit_table_to_rows(summary_table)
        layout.addWidget(summary_table)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(3)
        edit_button = self._compact_button(QPushButton("Edit bindings...", group))
        edit_button.clicked.connect(self._open_step_bindings_dialog)
        add_button = self._compact_button(QPushButton("Add binding", group))
        add_button.clicked.connect(lambda: self.add_binding_row())
        buttons.addWidget(edit_button)
        buttons.addWidget(add_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return group

    def _create_step_bindings_dialog(self) -> StepBindingsDialog:
        return StepBindingsDialog(
            bindings=SourceBindingsEditorValue(self._display_bindings).binding_declarations,
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
            self._apply_step_bindings(dialog.bindings())
        self.step_bindings_editor = None
        self.step_bindings_table = None

    def _append_step_binding(self, binding: NamedSourceBinding) -> None:
        self._apply_step_bindings(
            SourceBindingsEditorValue(self._display_bindings).binding_declarations
            + (binding,)
        )

    def _binding_summary_rows(
        self,
        view_model: SourceBindingsViewModel,
    ) -> tuple[tuple[str, str, str], ...]:
        if not view_model.step_bindings:
            return ()
        return (
            (
                str(len(view_model.step_bindings)),
                ", ".join(binding.alias for binding in view_model.step_bindings),
                ", ".join(sorted({binding.origin for binding in view_model.step_bindings})),
            ),
        )

    def _source_filters_group(self) -> QGroupBox:
        group, layout = self._section_group("Source Filters", "source_filters")
        table = self._create_table(0, len(SourceFilterColumn))
        table.setHorizontalHeaderLabels(
            tuple(column.name.title() for column in SourceFilterColumn)
        )
        self.source_filters_table = table
        self.source_filters_controller = self._register_structural_table_controller(
            EditableTableController(
                table=table,
                columns=tuple(SourceFilterColumn),
                free_form_cell_specs=self._free_form_cell_specs(),
                row_cells=EditableSourceFilterRow.cells,
                row_from_cells=EditableSourceFilterRow.from_cells,
                apply_changes=self._apply_source_filters_table,
                semantic_binding=EditableTableSemanticBinding(
                    owner_field_name="source_filters",
                    row_path_policy=IsomorphicDataclassRowPathPolicy(
                        row_value_type=SourceFilterClause,
                        column_count=len(SourceFilterColumn),
                    ),
                ),
            ),
        )
        for clause in SourceBindingsEditorValue(
            self._display_bindings
        ).source_filter_declarations:
            self.source_filters_controller.append(EditableSourceFilterRow.from_clause(clause))
        table.itemChanged.connect(
            lambda _: self.source_filters_controller.request_apply_changes()
        )
        table.resizeColumnsToContents()
        self._configure_table(table)
        self._fit_table_to_rows(table)
        layout.addWidget(table)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(3)
        add_button = self._compact_button(QPushButton("Add source filter"))
        add_button.clicked.connect(self.add_source_filter_row)
        remove_button = self._compact_button(QPushButton("Remove selected"))
        remove_button.clicked.connect(self.remove_selected_source_filter_rows)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return group

    def _metadata_rules_group(self) -> QGroupBox:
        group, layout = self._section_group("Metadata Rules", "metadata_rules")
        table = self._create_table(0, len(MetadataRuleColumn))
        table.setHorizontalHeaderLabels(
            tuple(column.name.title() for column in MetadataRuleColumn)
        )
        self.metadata_rules_table = table
        self.metadata_rules_controller = self._register_editable_table_controller(
            EditableTableController(
                table=table,
                columns=tuple(MetadataRuleColumn),
                free_form_cell_specs=self._free_form_cell_specs(),
                row_cells=EditableMetadataRuleRow.cells,
                row_from_cells=EditableMetadataRuleRow.from_cells,
                apply_changes=self._apply_metadata_rules_table,
            )
        )
        for rule in SourceBindingsEditorValue(
            self._display_bindings
        ).metadata_rule_declarations:
            self.metadata_rules_controller.append(EditableMetadataRuleRow.from_rule(rule))
        table.itemChanged.connect(
            lambda _: self.metadata_rules_controller.request_apply_changes()
        )
        table.resizeColumnsToContents()
        self._configure_table(table)
        self._fit_table_to_rows(table)
        layout.addWidget(table)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(3)
        add_button = self._compact_button(QPushButton("Add metadata rule"))
        add_button.clicked.connect(self.add_metadata_rule_row)
        remove_button = self._compact_button(QPushButton("Remove selected"))
        remove_button.clicked.connect(self.remove_selected_metadata_rule_rows)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return group

    def _match_plan_group(self) -> QGroupBox:
        group, layout = self._section_group("Match Plan", "match_plan")
        table = self._create_table(0, len(MatchPlanColumn))
        table.setHorizontalHeaderLabels(
            tuple(column.name.title() for column in MatchPlanColumn)
        )
        self.match_plan_table = table
        self.match_plan_controller = self._register_editable_table_controller(
            EditableTableController(
                table=table,
                columns=tuple(MatchPlanColumn),
                free_form_cell_specs=self._free_form_cell_specs(),
                row_cells=EditableMatchPlanRow.cells,
                row_from_cells=EditableMatchPlanRow.from_cells,
                apply_changes=self._apply_match_plan_table,
            )
        )
        for row in EditableMatchPlanRow.from_plan(
            SourceBindingsEditorValue(self._display_bindings).match_plan
        ):
            self.match_plan_controller.append(row)
        table.itemChanged.connect(
            lambda _: self.match_plan_controller.request_apply_changes()
        )
        table.resizeColumnsToContents()
        self._configure_table(table)
        self._fit_table_to_rows(table)
        layout.addWidget(table)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(3)
        add_button = self._compact_button(QPushButton("Add match dimension"))
        add_button.clicked.connect(self.add_match_plan_row)
        remove_button = self._compact_button(QPushButton("Remove selected"))
        remove_button.clicked.connect(self.remove_selected_match_plan_rows)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return group

    def _apply_step_bindings(
        self,
        bindings: tuple[NamedSourceBinding, ...],
    ) -> None:
        if self._updating_ui:
            return
        self._bindings = replace_raw(self._bindings, bindings=bindings)
        self._display_bindings = replace_raw(self._display_bindings, bindings=bindings)
        self.refresh()
        self.changed.emit()

    def _apply_source_filters_table(self) -> None:
        if self._updating_ui or self.source_filters_table is None:
            return
        if self.source_filters_controller is None:
            raise RuntimeError("Source filters table controller is not initialized.")
        if self.source_filters_controller.has_incomplete_rows():
            return
        source_filters = tuple(
            row.clause for row in self.source_filters_controller.rows()
        )
        self._bindings = replace_raw(
            self._bindings,
            source_filters=source_filters,
        )
        self._display_bindings = replace_raw(
            self._display_bindings,
            source_filters=source_filters,
        )
        self.changed.emit()

    def _apply_metadata_rules_table(self) -> None:
        if self._updating_ui or self.metadata_rules_table is None:
            return
        if self.metadata_rules_controller is None:
            raise RuntimeError("Metadata rules table controller is not initialized.")
        metadata_rules = tuple(
            row.rule for row in self.metadata_rules_controller.rows()
        )
        self._bindings = replace_raw(
            self._bindings,
            metadata_rules=metadata_rules,
        )
        self._display_bindings = replace_raw(
            self._display_bindings,
            metadata_rules=metadata_rules,
        )
        self.changed.emit()

    def _apply_match_plan_table(self) -> None:
        if self._updating_ui or self.match_plan_table is None:
            return
        if self.match_plan_controller is None:
            raise RuntimeError("Match plan table controller is not initialized.")
        if self.match_plan_controller.has_incomplete_rows():
            return
        if self._match_plan_table_has_incomplete_dimension_rows():
            return
        rows = self.match_plan_controller.rows()
        match_plan = self._match_plan_from_rows(rows)
        self._bindings = replace_raw(
            self._bindings,
            match_plan=match_plan,
        )
        self._display_bindings = replace_raw(
            self._display_bindings,
            match_plan=match_plan,
        )
        self.changed.emit()

    def _match_plan_table_has_incomplete_dimension_rows(self) -> bool:
        if self.match_plan_controller is None:
            return False
        row_values = self.match_plan_controller.row_values()
        rows_with_fields = 0
        rows_without_fields = 0
        for method, fields in row_values:
            if not method.strip() and not fields.strip():
                continue
            if fields.strip():
                rows_with_fields += 1
            else:
                rows_without_fields += 1
        return rows_with_fields > 0 and rows_without_fields > 0

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
        group, layout = self._section_group(title)
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

    def _section_group(
        self,
        title: str,
        field_name: str | None = None,
    ) -> tuple[QGroupBox, QVBoxLayout]:
        group = QGroupBox("")
        group.setStyleSheet(
            """
            QGroupBox {
                margin-top: 6px;
            }
            """
        )
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        if field_name and self._child_chrome:
            self._child_chrome.register_section_group(field_name, group)
            title_layout = self._child_chrome.create_section_header(
                title=title,
                field_name=field_name,
            )
            layout.addLayout(title_layout)
        else:
            label = QLabel(title, group)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            layout.addWidget(label)
        return group, layout

    def refresh_section_label_markers(
        self,
        owner_field_paths: tuple[DottedFieldPath, ...] | None = None,
    ) -> None:
        from objectstate.time_travel_profile import TimeTravelProfiler

        with TimeTravelProfiler.phase("openhcs.source_bindings.refresh_section_label_markers"):
            field_names = self._field_names_for_owner_paths(owner_field_paths)
            if self._child_chrome is not None:
                self._child_chrome.refresh_markers(field_names)
            self._refresh_table_placeholder_chrome(field_names)
            if field_names is None or self._enabled_field_name() in field_names:
                self._sync_enableable_chrome()

    def _field_names_for_owner_paths(
        self,
        owner_field_paths: tuple[DottedFieldPath, ...] | None,
    ) -> tuple[str, ...] | None:
        if owner_field_paths is None:
            return None
        if self._form_context is None:
            return ()

        owner_parts = self._form_context.owner_path.parts
        field_names: list[str] = []
        seen: set[str] = set()
        for owner_field_path in owner_field_paths:
            path_parts = owner_field_path.parts
            if path_parts[: len(owner_parts)] != owner_parts:
                continue
            if len(path_parts) != len(owner_parts) + 1:
                continue
            field_name = path_parts[-1]
            if field_name not in seen:
                seen.add(field_name)
                field_names.append(field_name)
        return tuple(field_names)

    def _refresh_table_placeholder_chrome(
        self,
        field_names: tuple[str, ...] | None = None,
    ) -> None:
        target_field_names = set(field_names) if field_names is not None else None
        for field_name, table in self._table_placeholder_targets():
            if target_field_names is not None and field_name not in target_field_names:
                continue
            active = self._field_has_table_placeholder_preview(field_name)
            if active and self._child_chrome is not None:
                group = self._child_chrome.navigation_target(field_name)
                if group is not None:
                    InlineDataclassChildChrome.set_widget_dimmed(group, False)
            if table is not None:
                if self._table_placeholder_chrome_state.get(field_name) != active:
                    EditableTableController.apply_placeholder_text_style_to_table(
                        table,
                        active,
                    )
                    self._table_placeholder_chrome_state[field_name] = active

    def _table_placeholder_targets(self) -> tuple[tuple[str, QTableWidget | None], ...]:
        return (
            ("bindings", self.step_bindings_summary_table),
            ("source_filters", self.source_filters_table),
            ("metadata_rules", self.metadata_rules_table),
            ("match_plan", self.match_plan_table),
        )

    def _field_has_table_placeholder_preview(self, field_name: str) -> bool:
        if (
            self._form_context is not None
            and self._form_context.child_has_inherited_preview(field_name)
        ):
            return True
        raw_value = SourceBindingsEditorValue(self._bindings).raw_field_value(field_name)
        display_value = SourceBindingsEditorValue(
            self._display_bindings
        ).raw_field_value(field_name)
        return raw_value is None and display_value is not None

    @staticmethod
    def _configure_table(table: QTableWidget) -> None:
        EditableTableLayout.configure(table)

    @staticmethod
    def _fit_table_to_rows(table: QTableWidget) -> None:
        EditableTableLayout.fit_to_rows(table)


def create_source_bindings_editor_widget(
    *,
    current_value: SourceBindingsEditorRawValue,
    manager: ParameterFormManager | None = None,
    param_info: InlineDataclassWidgetInfo | None = None,
    parent: QWidget | None = None,
    **_: object,
) -> SourceBindingsEditorWidget:
    """pyqt-reactive inline dataclass widget factory for source bindings."""

    SourceBindingsEditorValue(current_value)
    display_value = resolved_source_bindings_value(
        current_value=current_value,
        manager=manager,
        param_info=param_info,
    )
    return SourceBindingsEditorWidget.from_bindings(
        current_value,
        display_bindings=display_value,
        form_context=source_bindings_form_context(
            current_value=current_value,
            manager=manager,
            param_info=param_info,
        ),
        parent=parent,
    )


def source_bindings_form_context(
    *,
    current_value: SourceBindingsEditorRawValue,
    manager: ParameterFormManager | None,
    param_info: InlineDataclassWidgetInfo | None,
) -> InlineDataclassFormContext | None:
    """Build the semantic form context for source-binding child fields."""

    if manager is None or param_info is None:
        return None
    SourceBindingsEditorValue(current_value)
    return InlineDataclassFormContext.from_inline_widget(
        manager=manager,
        param_info=param_info,
        current_value=current_value,
    )


def resolved_source_bindings_value(
    *,
    current_value: SourceBindingsEditorRawValue,
    manager: ParameterFormManager | None,
    param_info: InlineDataclassWidgetInfo | None,
) -> SourceBindingsEditorRawValue:
    """Return the live inherited value used to seed placeholder tables."""

    if manager is None or param_info is None:
        return current_value
    field_path = (
        DottedFieldPath(manager.field_id).child(param_info.name)
        if manager.field_id
        else DottedFieldPath(param_info.name)
    )
    resolved_value = manager.state.get_resolved_value(field_path.value)
    if resolved_value is None:
        return current_value
    expected_base_type = SourceBindingsEditorValue(current_value).base_type()
    resolved_editor_value = SourceBindingsEditorValue(resolved_value)
    if resolved_editor_value.base_type() is not expected_base_type:
        raise TypeError(
            f"Resolved source-bindings value must be {expected_base_type.__name__}, "
            f"got {type(resolved_value).__name__}."
        )
    return resolved_value


def register_source_bindings_editor_widget() -> None:
    """Register the typed source-binding editor with pyqt-reactive forms."""

    from pyqt_reactive.forms.parameter_info_types import register_inline_dataclass_widget

    for config_type in SourceBindingsConfig.registered_plan_types():
        if issubclass(config_type, SourceBindingsConfig):
            register_inline_dataclass_widget(
                config_type,
                create_source_bindings_editor_widget,
            )


register_source_bindings_editor_widget()


__all__ = (
    "SourceBindingsEditorWidget",
    "SourceFilterColumn",
    "create_source_bindings_editor_widget",
    "resolved_source_bindings_value",
    "register_source_bindings_editor_widget",
)
