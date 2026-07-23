"""Live measurement table window for running OpenHCS executions."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from openhcs.core.progress.live_measurements import LiveMeasurementTablePreview
from openhcs.pyqt_gui.widgets.shared.services.live_measurement_progress_service import (
    LiveMeasurementAvailableNotification,
)
from pyqt_reactive.theming import ColorScheme, StyleSheetGenerator
from openhcs.runtime.zmq_config import OpenHCSZMQConfig
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope


@dataclass(frozen=True, slots=True)
class LiveMeasurementTableEntry:
    """Presentation row for one live measurement preview."""

    sequence_id: int
    execution_id: str
    plate_id: str
    axis_id: str
    step_name: str
    preview: LiveMeasurementTablePreview
    truncated_preview_group: bool

    @property
    def label(self) -> str:
        address = self.preview.address
        scope_text = _scope_text(address.key.scope)
        object_text = (
            f" [{self.preview.object_name}]" if self.preview.object_name else ""
        )
        return f"{self.step_name}: {address.key.name}{object_text} ({scope_text})"

    @property
    def semantic_sort_key(self) -> tuple:
        address = self.preview.address
        return (
            _semantic_sort_atom(self.plate_id),
            *(_semantic_sort_atom(part) for part in astuple(address.key.scope)),
            _semantic_sort_atom(self.step_name),
            _semantic_sort_atom(address.key.artifact_type.value),
            _semantic_sort_atom(address.key.name),
            _semantic_sort_atom(self.preview.object_name),
            self.sequence_id,
        )


class LiveMeasurementTableModel:
    """Retained UI-side projection of live measurement preview notifications."""

    def __init__(self, *, max_entries: int = 500) -> None:
        self._max_entries = max_entries
        self._entries: list[LiveMeasurementTableEntry] = []
        self._next_sequence_id = 0

    def clear(self) -> None:
        self._entries.clear()
        self._next_sequence_id = 0

    def add_notification(
        self,
        notification: LiveMeasurementAvailableNotification,
    ) -> None:
        event = notification.event
        for preview in notification.payload.previews:
            self._entries.append(
                LiveMeasurementTableEntry(
                    sequence_id=self._next_sequence_id,
                    execution_id=event.execution_id,
                    plate_id=event.plate_id,
                    axis_id=event.axis_id,
                    step_name=event.step_name,
                    preview=preview,
                    truncated_preview_group=notification.payload.truncated_previews,
                )
            )
            self._next_sequence_id += 1
        if len(self._entries) > self._max_entries:
            del self._entries[: len(self._entries) - self._max_entries]

    @property
    def entries(self) -> tuple[LiveMeasurementTableEntry, ...]:
        return tuple(self._entries)

    def latest_sequence_id(self) -> int | None:
        if not self._entries:
            return None
        return self._entries[-1].sequence_id

    def entry_by_sequence_id(
        self,
        sequence_id: int | None,
    ) -> LiveMeasurementTableEntry | None:
        if sequence_id is None:
            return None
        for entry in self._entries:
            if entry.sequence_id == sequence_id:
                return entry
        return None

    def semantic_entries(self) -> tuple[LiveMeasurementTableEntry, ...]:
        return tuple(sorted(self._entries, key=lambda entry: entry.semantic_sort_key))


class LiveMeasurementsWindow(QDialog):
    """Read-only window showing live measurement previews."""

    def __init__(
        self,
        model: LiveMeasurementTableModel,
        *,
        orchestrator: object | None = None,
        color_scheme: ColorScheme | None = None,
        zmq_config: OpenHCSZMQConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._orchestrator = orchestrator
        self._zmq_config = zmq_config
        self.color_scheme = color_scheme or ColorScheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)
        self._image_browser = None
        self._image_browser_placeholder: QWidget | None = None
        self._image_browser_tab_index: int | None = None
        self.setWindowTitle("Live Results")
        self.setModal(False)
        self.resize(1200, 700)
        self._apply_theme()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("LiveResultsTabs")
        self.tabs.tabBar().setObjectName("LiveResultsTabBar")
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._activate_tab)
        layout.addWidget(self.tabs, 1)

        measurements_tab = QWidget(self)
        measurements_tab.setObjectName("LiveResultsMeasurementsTab")
        measurements_layout = QVBoxLayout(measurements_tab)
        measurements_layout.setContentsMargins(0, 0, 0, 0)
        measurements_layout.setSpacing(6)
        header = QHBoxLayout()
        header.setContentsMargins(0, 4, 0, 0)
        header.setSpacing(6)

        self.view_artifact_button = QPushButton("Show Artifact", self)
        self.view_artifact_button.clicked.connect(self._show_selected_artifact)
        self.view_artifact_button.setStyleSheet(
            self.style_generator.generate_button_style()
        )
        header.addStretch(1)
        header.addWidget(self.view_artifact_button)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self._clear)
        self.clear_button.setStyleSheet(self.style_generator.generate_button_style())
        header.addWidget(self.clear_button)

        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)
        close_button.setStyleSheet(self.style_generator.generate_button_style())
        header.addWidget(close_button)
        measurements_layout.addLayout(header)

        measurement_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        measurement_splitter.setObjectName("LiveResultsSplitter")
        measurement_splitter.setChildrenCollapsible(False)

        nav_panel = QWidget(self)
        nav_panel.setObjectName("LiveResultsNavPanel")
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(8, 8, 8, 8)
        nav_layout.setSpacing(6)
        nav_label = QLabel("Measurement snapshots", self)
        nav_label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)};"
        )
        nav_layout.addWidget(nav_label)

        self.entry_list = QListWidget(self)
        self.entry_list.setObjectName("LiveResultsEntryList")
        self.entry_list.setFrameShape(QFrame.Shape.NoFrame)
        self.entry_list.setAlternatingRowColors(True)
        self.entry_list.currentRowChanged.connect(self._render_selected_entry)
        nav_layout.addWidget(self.entry_list, 1)
        measurement_splitter.addWidget(nav_panel)

        table_panel = QWidget(self)
        table_panel.setObjectName("LiveResultsTablePanel")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(8, 8, 8, 8)
        table_layout.setSpacing(6)

        self.status_label = QLabel("No live measurements yet", self)
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.status_label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)};"
        )
        table_layout.addWidget(self.status_label)

        self.table = QTableWidget(0, 0, self)
        self.table.setObjectName("LiveResultsTable")
        self.table.setFrameShape(QFrame.Shape.NoFrame)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.table, 1)
        measurement_splitter.addWidget(table_panel)
        measurement_splitter.setSizes([320, 880])

        measurements_layout.addWidget(measurement_splitter, 1)
        self.tabs.addTab(measurements_tab, "Measurements")

        self._reset_image_browser_tab()

        self.refresh(select_latest=True)

    def set_orchestrator(self, orchestrator: object | None) -> None:
        """Set the plate context used by the embedded image/results viewer."""
        if orchestrator is self._orchestrator:
            return
        self._orchestrator = orchestrator
        self._reset_image_browser_tab()

    def refresh(self, *, select_latest: bool = False) -> None:
        entries = self._model.semantic_entries()
        selected_sequence_id = (
            self._model.latest_sequence_id()
            if select_latest
            else self._selected_sequence_id()
        )
        self.entry_list.blockSignals(True)
        self.entry_list.clear()
        for entry in entries:
            item = QListWidgetItem(_navigation_label(entry))
            item.setToolTip(entry.label)
            item.setData(Qt.ItemDataRole.UserRole, entry.sequence_id)
            self.entry_list.addItem(item)
        selected_row = _row_for_sequence_id(self.entry_list, selected_sequence_id)
        if selected_row is None and entries:
            selected_row = _row_for_sequence_id(
                self.entry_list,
                self._model.latest_sequence_id(),
            )
        if selected_row is not None:
            self.entry_list.setCurrentRow(selected_row)
        self.entry_list.blockSignals(False)
        self._render_selected_entry()

    def _clear(self) -> None:
        self._model.clear()
        self.refresh()

    def _render_selected_entry(self, *_args: object) -> None:
        entry = self._selected_entry()
        if entry is None:
            self.status_label.setText("No live measurements yet")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.view_artifact_button.setEnabled(False)
            return

        preview = entry.preview
        self.view_artifact_button.setEnabled(
            self._orchestrator is not None and bool(preview.address.location.path)
        )
        status_parts = [
            f"{preview.row_count} row(s)",
            f"{len(preview.columns)} column(s)",
            f"axis {preview.address.key.scope.axis_id}",
            f"backend {preview.address.location.backend}",
        ]
        if preview.truncated_rows:
            status_parts.append("rows truncated")
        if preview.truncated_columns:
            status_parts.append("columns truncated")
        if entry.truncated_preview_group:
            status_parts.append("additional tables omitted from event")
        self.status_label.setText(" | ".join(status_parts))

        self.table.setColumnCount(len(preview.columns))
        self.table.setRowCount(len(preview.rows))
        self.table.setHorizontalHeaderLabels(preview.columns)
        for row_index, row in enumerate(preview.rows):
            for column_index, column in enumerate(preview.columns):
                item = QTableWidgetItem(_cell_text(row.get(column)))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row_index, column_index, item)
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()

    def _selected_sequence_id(self) -> int | None:
        item = self.entry_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, int) else None

    def _selected_entry(self) -> LiveMeasurementTableEntry | None:
        return self._model.entry_by_sequence_id(self._selected_sequence_id())

    def _show_selected_artifact(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        self._ensure_image_browser()
        if self._image_browser is None:
            return
        self.tabs.setCurrentWidget(self._image_browser)
        self._image_browser.focus_file_by_path(entry.preview.address.location.path)

    def _activate_tab(self, index: int) -> None:
        if index == self._image_browser_tab_index:
            self._ensure_image_browser()

    def _reset_image_browser_tab(self) -> None:
        if self._image_browser_tab_index is not None:
            self.tabs.removeTab(self._image_browser_tab_index)
            self._image_browser_tab_index = None
            self._image_browser = None
            self._image_browser_placeholder = None

        if self._orchestrator is None:
            return

        self._image_browser_placeholder = QWidget(self)
        self._image_browser_placeholder.setObjectName("LiveResultsImageBrowserPending")
        self._image_browser_tab_index = self.tabs.addTab(
            self._image_browser_placeholder,
            "Images / Viewers",
        )

    def _ensure_image_browser(self) -> None:
        if self._image_browser is not None:
            return
        if self._orchestrator is None or self._image_browser_tab_index is None:
            return

        from openhcs.pyqt_gui.widgets.image_browser import ImageBrowserWidget

        image_browser = ImageBrowserWidget(
            orchestrator=self._orchestrator,
            color_scheme=self.color_scheme,
            zmq_config=self._zmq_config,
            parent=self,
        )
        tab_index = self._image_browser_tab_index
        self.tabs.removeTab(tab_index)
        self.tabs.insertTab(
            tab_index,
            image_browser,
            "Images / Viewers",
        )
        self._image_browser = image_browser
        self._image_browser_placeholder = None
        self.tabs.setCurrentIndex(tab_index)
        self._render_selected_entry()

    def _apply_theme(self) -> None:
        cs = self.color_scheme
        self.setStyleSheet(
            self.style_generator.generate_dialog_style()
            + self.style_generator.generate_list_widget_style()
            + self.style_generator.generate_table_widget_style()
            + self.style_generator.generate_button_style()
            + f"""
            QTabWidget#LiveResultsTabs::pane {{
                border: none;
                background-color: {cs.to_hex(cs.window_bg)};
                margin: 0;
                padding: 0;
            }}
            QTabBar#LiveResultsTabBar::tab {{
                background-color: {cs.to_hex(cs.button_normal_bg)};
                color: {cs.to_hex(cs.text_secondary)};
                border: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 8px 16px;
                margin-right: 2px;
            }}
            QTabBar#LiveResultsTabBar::tab:selected {{
                background-color: {cs.to_hex(cs.panel_bg)};
                color: {cs.to_hex(cs.text_primary)};
                font-weight: bold;
                border-bottom: 2px solid {cs.to_hex(cs.text_accent)};
            }}
            QTabBar#LiveResultsTabBar::tab:hover {{
                background-color: {cs.to_hex(cs.button_hover_bg)};
            }}
            QWidget#LiveResultsMeasurementsTab {{
                background-color: {cs.to_hex(cs.window_bg)};
            }}
            QWidget#LiveResultsNavPanel,
            QWidget#LiveResultsTablePanel {{
                background-color: {cs.to_hex(cs.panel_bg)};
                border: none;
                border-radius: 3px;
            }}
            QSplitter#LiveResultsSplitter::handle {{
                background-color: {cs.to_hex(cs.border_color)};
            }}
            QSplitter#LiveResultsSplitter::handle:horizontal {{
                width: 1px;
                margin: 8px 4px;
            }}
            QListWidget#LiveResultsEntryList,
            QTableWidget#LiveResultsTable {{
                background-color: {cs.to_hex(cs.window_bg)};
                border: none;
                border-radius: 3px;
            }}
            QListWidget#LiveResultsEntryList::item {{
                padding: 6px;
                border: none;
            }}
            QListWidget#LiveResultsEntryList::item:selected {{
                background-color: {cs.to_hex(cs.selection_bg)};
                color: {cs.to_hex(cs.selection_text)};
            }}
            QListWidget#LiveResultsEntryList::item:hover {{
                background-color: {cs.to_hex(cs.hover_bg)};
            }}
            """
        )


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _semantic_sort_atom(value: Any) -> tuple[int, int | str]:
    if value in (None, ""):
        return (0, "")
    text = str(value)
    if text.isdecimal():
        return (1, int(text))
    return (2, text.casefold())


def _row_for_sequence_id(
    list_widget: QListWidget,
    sequence_id: int | None,
) -> int | None:
    if sequence_id is None:
        return None
    for row in range(list_widget.count()):
        item = list_widget.item(row)
        if item.data(Qt.ItemDataRole.UserRole) == sequence_id:
            return row
    return None


def _navigation_label(entry: LiveMeasurementTableEntry) -> str:
    address = entry.preview.address
    scope_text = _scope_text(address.key.scope)
    object_text = f" [{entry.preview.object_name}]" if entry.preview.object_name else ""
    return (
        f"{entry.step_name}\n"
        f"{address.key.name}{object_text}\n"
        f"{scope_text or entry.plate_id}"
    )


def _scope_text(scope: RuntimeExecutionAxisScope) -> str:
    return " / ".join(str(part) for part in astuple(scope) if part not in (None, ""))
