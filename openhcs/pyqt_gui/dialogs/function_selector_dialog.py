"""Function selector for the endpoint-owned callable catalog."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Self

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.widgets.shared.function_table_browser import (
    FunctionTableBrowser,
    FunctionTableRow,
)

from openhcs.agent.dto.functions import FunctionCatalogEntry, FunctionCatalogPage
from openhcs.processing.custom_functions.signals import custom_function_signals
from openhcs.pyqt_gui.services.function_catalog_projection import (
    EndpointFunctionUnavailableError,
    ZMQFunctionCatalogProjectionService,
)

logger = logging.getLogger(__name__)


FunctionCatalogEntryMap = Mapping[str, FunctionCatalogEntry]


def function_table_rows(
    entries: FunctionCatalogEntryMap,
) -> dict[str, FunctionTableRow]:
    """Project endpoint-owned catalog entries into generic table rows."""
    return {
        function_id: FunctionTableRow(
            name=entry.name,
            module=entry.module,
            library=entry.library,
            backend_tags=entry.backend_tags,
            summary=entry.summary,
        )
        for function_id, entry in entries.items()
    }


@dataclass(slots=True)
class FunctionModuleTreeNode:
    """One module declaration and every endpoint function below it."""

    name: str
    full_path: str
    direct_function_ids: list[str] = field(default_factory=list)
    children: dict[str, Self] = field(default_factory=dict)

    @classmethod
    def forest(cls, entries: FunctionCatalogEntryMap) -> tuple[Self, ...]:
        """Build the module forest directly from endpoint catalog declarations."""
        roots: dict[str, Self] = {}
        for function_id, entry in entries.items():
            path_parts = tuple(filter(None, (entry.module or "unknown").split(".")))
            if not path_parts:
                path_parts = ("unknown",)

            siblings = roots
            full_path_parts: list[str] = []
            for part in path_parts:
                full_path_parts.append(part)
                node = siblings.get(part)
                if node is None:
                    node = cls(name=part, full_path=".".join(full_path_parts))
                    siblings[part] = node
                siblings = node.children
            node.direct_function_ids.append(function_id)
        return tuple(roots.values())

    @property
    def function_ids(self) -> tuple[str, ...]:
        """Return every endpoint function owned by this module subtree."""
        return (
            *self.direct_function_ids,
            *(
                function_id
                for child in self.children.values()
                for function_id in child.function_ids
            ),
        )

    @property
    def function_count(self) -> int:
        """Return the number of endpoint functions in this module subtree."""
        return len(self.direct_function_ids) + sum(
            child.function_count for child in self.children.values()
        )

    def entries_for_subtree(
        self,
        catalog_entries: FunctionCatalogEntryMap,
    ) -> dict[str, FunctionCatalogEntry]:
        function_ids = frozenset(self.function_ids)
        return {
            function_id: metadata
            for function_id, metadata in catalog_entries.items()
            if function_id in function_ids
        }

    def filter_description(self) -> str:
        """Return the module filter description shown beside the row count."""
        return f"filtered by module: {self.full_path}"


class FunctionSelectorDialog(QDialog):
    """
    Enhanced function selector dialog with table-based interface and rich metadata.

    Uses the connected execution endpoint's typed catalog projection, so remote
    backend and custom-function availability determine the displayed functions.
    """

    # UI Constants (RST principle: eliminate magic numbers)
    DEFAULT_WIDTH = 1200
    DEFAULT_HEIGHT = 700
    MIN_WIDTH = 800
    MIN_HEIGHT = 500
    MODULE_COLUMN_WIDTH = 250
    DESCRIPTION_COLUMN_WIDTH = 200
    TREE_PROPORTION = 180  # Reduced to give more space to function table
    TABLE_PROPORTION = 820  # Increased for better function table visibility

    # Signals
    function_selected = pyqtSignal(object)  # Selected function
    catalog_prepared = pyqtSignal(object)

    def __init__(
        self,
        catalog_service: ZMQFunctionCatalogProjectionService,
        current_function: Callable | None = None,
        parent=None,
    ):
        """
        Initialize function selector dialog.

        Args:
            current_function: Currently selected function (for highlighting)
            parent: Parent widget
        """
        super().__init__(parent)

        self.catalog_service = catalog_service
        self.current_function = current_function
        self.selected_function: Callable | None = None
        self.selected_function_id: str | None = None

        # Initialize color scheme and style generator
        self.color_scheme = ColorScheme()

        # Endpoint metadata is populated asynchronously after the widgets exist.
        self.catalog_entries: dict[str, FunctionCatalogEntry] = {}
        self._catalog_future: Future[FunctionCatalogPage] | None = None

        self.setup_ui()
        self.setup_connections()
        self.populate_module_tree()
        self.populate_function_table()

        # Connect to custom function signals for auto-refresh
        custom_function_signals.functions_changed.connect(self._on_functions_changed)
        self.catalog_prepared.connect(self._apply_function_data)
        self._request_function_data()

        logger.debug(
            f"Function selector initialized with {len(self.catalog_entries)} functions"
        )

    def _request_function_data(self) -> None:
        """Request the shared endpoint catalog and return to Qt immediately."""

        logger.info("Requesting functions from the connected execution endpoint")
        future = self.catalog_service.prepare(
            compact_signatures=True,
        )
        self._catalog_future = future
        self._set_selection_state(None, False)
        self.function_table_browser.status_label.setText(
            "Loading function catalog from the execution server..."
        )
        future.add_done_callback(self.catalog_prepared.emit)

    def _apply_function_data(
        self,
        future: Future[FunctionCatalogPage],
    ) -> None:
        """Apply a completed catalog future on the Qt thread."""

        if future is not self._catalog_future:
            return
        try:
            page = future.result()
        except Exception as error:
            logger.exception("Failed to load the endpoint function catalog")
            self.function_table_browser.status_label.setText(
                f"Function catalog unavailable: {error}"
            )
            return

        self.catalog_entries = {entry.function_id: entry for entry in page.items}
        logger.info(
            "Loaded %d functions from endpoint catalog revision %s",
            len(self.catalog_entries),
            page.revision,
        )

        self.populate_module_tree()
        self.populate_function_table()

    def _on_functions_changed(self):
        """Handle custom function changes by reloading and refreshing the view."""
        logger.info("Custom functions changed - refreshing function selector")

        self.catalog_service.invalidate()

        self._request_function_data()

    def populate_module_tree(self):
        """Project the endpoint catalog's module declarations into the Qt tree."""
        self.module_tree.clear()
        for module_node in FunctionModuleTreeNode.forest(self.catalog_entries):
            self._append_module_tree_item(self.module_tree, module_node)

    def _update_filtered_view(
        self,
        catalog_entries: FunctionCatalogEntryMap,
        filter_description: str = "",
    ):
        """Update filtered view using table browser."""
        self.function_table_browser.set_filtered_items(
            function_table_rows(catalog_entries)
        )

        # Create unified count display in the browser's status label
        total_count = len(self.catalog_entries)
        filtered_count = len(self.function_table_browser.filtered_items)
        count_text = f"Functions: {filtered_count}/{total_count}"
        if filter_description:
            count_text += f" ({filter_description})"

        self.function_table_browser.status_label.setText(count_text)

        # Clear selection when filtering
        self._set_selection_state(None, False)

    def _set_selection_state(self, function_id: str | None, enabled: bool):
        """Set button state based on selection."""
        self.selected_function = None
        self.selected_function_id = function_id
        self.select_btn.setEnabled(enabled)

    def _create_pane_widget(self, title: str, main_widget) -> QWidget:
        """Mathematical simplification: factor out common pane setup pattern (RST principle)."""
        pane_widget = QWidget()
        layout = QVBoxLayout(pane_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create title with consistent styling using color scheme
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-weight: bold;
            background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
            color: {self.color_scheme.to_hex(self.color_scheme.text_primary)};
            padding: 5px;
        """)
        layout.addWidget(title_label)

        # Add main widget
        layout.addWidget(main_widget)

        return pane_widget

    def _append_module_tree_item(
        self,
        parent_container,
        module_node: FunctionModuleTreeNode,
    ) -> None:
        """Append one typed module subtree to the Qt tree projection."""
        module_item = QTreeWidgetItem(parent_container)
        module_item.setText(
            0,
            f"{module_node.name} ({module_node.function_count} functions)",
        )
        module_item.setData(0, Qt.ItemDataRole.UserRole, module_node)
        module_item.setExpanded(False)
        for child in module_node.children.values():
            self._append_module_tree_item(module_item, child)

    def setup_ui(self):
        """Setup the dual-pane user interface with tree, filters, and table."""
        self.setWindowTitle("Select Function - Dual Pane View")
        self.setModal(True)
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)

        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("Select Function - Dual Pane View")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)};"
        )
        layout.addWidget(title_label)

        # Create main horizontal splitter (left panel | right table)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_splitter.setHandleWidth(5)

        # === LEFT PANEL: Tree + Filters ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Module tree
        self.module_tree = QTreeWidget()
        self.module_tree.setHeaderHidden(True)
        self.module_tree.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.module_tree.mousePressEvent = self._tree_mouse_press_event
        left_layout.addWidget(
            self._create_pane_widget("Module Structure", self.module_tree), 1
        )

        main_splitter.addWidget(left_panel)

        # === RIGHT PANEL: Function Table Browser ===
        self.function_table_browser = FunctionTableBrowser(
            color_scheme=self.color_scheme, parent=self
        )
        self.function_table_browser.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        right_widget = self._create_pane_widget(
            "Function Details", self.function_table_browser
        )
        main_splitter.addWidget(right_widget)

        # Set splitter proportions
        main_splitter.setSizes([self.TREE_PROPORTION, self.TABLE_PROPORTION])
        layout.addWidget(main_splitter, 1)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.select_btn = QPushButton("Select")
        self.select_btn.setEnabled(False)
        self.select_btn.setDefault(True)
        button_layout.addWidget(self.select_btn)

        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(cancel_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Apply centralized styling
        self.setStyleSheet(
            self.color_scheme.styles.generate_dialog_style()
            + "\n"
            + self.color_scheme.styles.generate_tree_widget_style()
            + "\n"
            + self.color_scheme.styles.generate_table_widget_style()
            + "\n"
            + self.color_scheme.styles.generate_button_style()
        )

        # Connect buttons
        self.select_btn.clicked.connect(self.accept_selection)
        cancel_btn.clicked.connect(self.reject)

    def setup_connections(self):
        """Setup signal/slot connections."""
        # Tree selection for filtering
        self.module_tree.itemSelectionChanged.connect(self.on_tree_selection_changed)

        # Table browser signals
        self.function_table_browser.item_selected.connect(self._on_function_selected)
        self.function_table_browser.item_double_clicked.connect(
            self._on_function_double_clicked
        )

    def populate_function_table(self):
        """Populate function table using FunctionTableBrowser."""
        self.function_table_browser.set_items(function_table_rows(self.catalog_entries))

        # Update count label
        total = len(self.catalog_entries)
        filtered = len(self.function_table_browser.filtered_items)
        self.function_table_browser.status_label.setText(
            f"Functions: {filtered}/{total}"
        )

    def on_tree_selection_changed(self):
        """Handle tree selection using mathematical simplification (RST principle)."""
        selected_items = self.module_tree.selectedItems()

        # If no items selected, show all functions
        if not selected_items:
            self._update_filtered_view(self.catalog_entries, "showing all functions")
            return

        item = selected_items[0]
        data = item.data(0, Qt.ItemDataRole.UserRole)

        if isinstance(data, FunctionModuleTreeNode):
            entries = data.entries_for_subtree(self.catalog_entries)
            self._update_filtered_view(entries, data.filter_description())
        else:
            # No data means show all functions
            self._update_filtered_view(self.catalog_entries, "showing all functions")

    def _tree_mouse_press_event(self, event):
        """Handle mouse press events on the tree to allow deselection."""
        # Get the item at the click position
        item = self.module_tree.itemAt(event.pos())

        if item is None:
            # Clicked in empty space - clear selection
            self.module_tree.clearSelection()
        else:
            # Clicked on an item - use default behavior
            QTreeWidget.mousePressEvent(self.module_tree, event)

    def _on_function_selected(self, key: str, _item: FunctionTableRow):
        """Handle function selection from table browser."""
        self._set_selection_state(key, True)

    def _on_function_double_clicked(self, key: str, _item: FunctionTableRow):
        """Handle function double-click from table browser."""
        self._set_selection_state(key, True)
        self.accept_selection()

    def accept_selection(self):
        """Accept the selected function."""
        if self.selected_function_id is None:
            return
        try:
            self.selected_function = self.catalog_service.import_selected_callable(
                self.selected_function_id
            )
        except EndpointFunctionUnavailableError as exc:
            self.function_table_browser.status_label.setText(str(exc))
            self.select_btn.setEnabled(False)
            return
        self.function_selected.emit(self.selected_function)
        self.accept()

    def get_selected_function(self) -> Callable | None:
        """Get the selected function."""
        return self.selected_function

    @staticmethod
    def select_function(
        catalog_service: ZMQFunctionCatalogProjectionService,
        current_function: Callable | None = None,
        parent=None,
    ) -> Callable | None:
        """
        Static method to show function selector and return selected function.

        Args:
            current_function: Currently selected function (for highlighting)
            parent: Parent widget

        Returns:
            Selected function or None if cancelled
        """
        dialog = FunctionSelectorDialog(catalog_service, current_function, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_selected_function()
        return None
