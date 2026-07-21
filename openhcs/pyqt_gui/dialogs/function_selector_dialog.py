"""
Function Selector Dialog for PyQt6 GUI.

Mirrors the Textual TUI FunctionSelectorWindow functionality using the same
FunctionRegistryService and business logic.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

from metaclass_registry import AutoRegisterMeta
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
from pyqt_reactive.theming import ColorScheme, StyleSheetGenerator
from pyqt_reactive.widgets.shared.function_table_browser import FunctionTableBrowser

# Use the registry service from correct location
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.lib_registry.unified_registry import FunctionMetadata
from openhcs.processing.custom_functions.signals import custom_function_signals

logger = logging.getLogger(__name__)


FunctionMetadataMap = Mapping[str, FunctionMetadata]


class FunctionTreeNode(ABC, metaclass=AutoRegisterMeta):
    """Domain object stored in the module tree instead of stringly item data."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True

    @abstractmethod
    def filtered_functions(
        self,
        all_functions_metadata: FunctionMetadataMap,
        module_path_for: Callable[[FunctionMetadata], str],
    ) -> Dict[str, FunctionMetadata]:
        """Return functions selected by this tree node."""

    @abstractmethod
    def filter_description(self) -> str:
        """Human-readable status suffix for the active tree filter."""


@dataclass(frozen=True)
class ModuleFunctionTreeNode(FunctionTreeNode):
    module_path: str
    function_names: tuple[str, ...]

    def filtered_functions(
        self,
        all_functions_metadata: FunctionMetadataMap,
        module_path_for: Callable[[FunctionMetadata], str],
    ) -> Dict[str, FunctionMetadata]:
        function_names = frozenset(self.function_names)
        return {
            name: metadata
            for name, metadata in all_functions_metadata.items()
            if name in function_names
        }

    def filter_description(self) -> str:
        return "filtered by module"


@dataclass(frozen=True)
class ModulePartTreeNode(FunctionTreeNode):
    full_path: str

    def filtered_functions(
        self,
        all_functions_metadata: FunctionMetadataMap,
        module_path_for: Callable[[FunctionMetadata], str],
    ) -> Dict[str, FunctionMetadata]:
        return {
            name: metadata
            for name, metadata in all_functions_metadata.items()
            if module_path_for(metadata).startswith(self.full_path)
        }

    def filter_description(self) -> str:
        return f"filtered by module part: {self.full_path}"


class FunctionSelectorDialog(QDialog):
    """
    Enhanced function selector dialog with table-based interface and rich metadata.

    Uses unified metadata from FunctionRegistryService for consistent display
    of both OpenHCS and external library functions.
    """

    # Class-level cache for expensive metadata discovery
    _metadata_cache: Optional[Dict[str, FunctionMetadata]] = None

    @classmethod
    def clear_cache(cls):
        """Clear the function metadata cache. Call this when function registry is reloaded."""
        cls._metadata_cache = None
        logger.debug("Function selector dialog cache cleared")

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

    def __init__(self, current_function: Optional[Callable] = None, parent=None):
        """
        Initialize function selector dialog.

        Args:
            current_function: Currently selected function (for highlighting)
            parent: Parent widget
        """
        super().__init__(parent)

        self.current_function = current_function
        self.selected_function = None

        # Initialize color scheme and style generator
        self.color_scheme = ColorScheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)

        # Load enhanced function metadata
        self.all_functions_metadata: Dict[str, FunctionMetadata] = {}
        self.filtered_functions: Dict[str, FunctionMetadata] = {}
        self._load_function_data()

        self.setup_ui()
        self.setup_connections()
        self.populate_module_tree()
        self.populate_function_table()

        # Connect to custom function signals for auto-refresh
        custom_function_signals.functions_changed.connect(self._on_functions_changed)

        logger.debug(
            f"Function selector initialized with {len(self.all_functions_metadata)} functions"
        )

    def _load_function_data(self) -> None:
        """Load ALL functions from registries (not just FUNC_REGISTRY subset)."""
        # Check if we have cached metadata
        if FunctionSelectorDialog._metadata_cache is not None:
            logger.debug("Using cached function metadata")
            self.all_functions_metadata = FunctionSelectorDialog._metadata_cache
            self.filtered_functions = self.all_functions_metadata.copy()
            return

        logger.info("Loading ALL functions from registries")
        self.all_functions_metadata = dict(
            RegistryService.get_all_functions_with_metadata()
        )

        # Cache the results for future use
        FunctionSelectorDialog._metadata_cache = self.all_functions_metadata
        logger.info(
            f"Loaded {len(self.all_functions_metadata)} functions from all registries"
        )

        self.filtered_functions = self.all_functions_metadata.copy()

    def _on_functions_changed(self):
        """Handle custom function changes by reloading and refreshing the view."""
        logger.info("Custom functions changed - refreshing function selector")

        # Clear caches to force reload
        FunctionSelectorDialog.clear_metadata_cache()

        # Reload function data
        self._load_function_data()

        # Refresh both views
        self.populate_module_tree()
        self.populate_function_table()

        logger.debug(
            f"Function selector refreshed with {len(self.all_functions_metadata)} functions"
        )

    def populate_module_tree(self):
        """Populate the module tree with hierarchical function organization based purely on module paths."""
        self.module_tree.clear()

        # Build hierarchical structure directly from module paths
        module_hierarchy = {}
        for func_name, metadata in self.all_functions_metadata.items():
            module_path = self._extract_module_path(metadata)
            # Build hierarchical structure by splitting module path on '.'
            self._add_function_to_hierarchy(module_hierarchy, module_path, func_name)

        # Build tree structure directly from module hierarchy (no library grouping)
        self._build_module_hierarchy_tree(
            self.module_tree, module_hierarchy, [], is_root=True
        )

    def _update_filtered_view(
        self, filtered_functions: Dict[str, FunctionMetadata], filter_description: str = ""
    ):
        """Update filtered view using table browser."""
        self.filtered_functions = filtered_functions
        self.function_table_browser.set_filtered_items(self.filtered_functions)

        # Create unified count display in the browser's status label
        total_count = len(self.all_functions_metadata)
        filtered_count = len(self.function_table_browser.filtered_items)
        count_text = f"Functions: {filtered_count}/{total_count}"
        if filter_description:
            count_text += f" ({filter_description})"

        self.function_table_browser.status_label.setText(count_text)

        # Clear selection when filtering
        self._set_selection_state(None, False)

    def _set_selection_state(self, function: Optional[Callable], enabled: bool):
        """Set button state based on selection."""
        self.selected_function = function
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

    def _extract_module_path(self, metadata: FunctionMetadata) -> str:
        """Extract full module path from metadata for hierarchical tree building."""
        if not metadata.module:
            return "unknown"

        # Return the full module path for hierarchical tree building
        return metadata.module

    def _add_function_to_hierarchy(
        self, hierarchy: dict, module_path: str, func_name: str
    ):
        """Add a function to the hierarchical module structure."""
        if module_path == "unknown":
            # Handle unknown modules
            if "functions" not in hierarchy:
                hierarchy["functions"] = []
            hierarchy["functions"].append(func_name)
            return

        # Split module path and build hierarchy
        parts = module_path.split(".")
        current_level = hierarchy

        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

        # Add function to the deepest level
        if "functions" not in current_level:
            current_level["functions"] = []
        current_level["functions"].append(func_name)

    def _count_functions_in_hierarchy(self, hierarchy: dict) -> int:
        """Count total functions in a hierarchical structure."""
        count = 0
        for key, value in hierarchy.items():
            if key == "functions":
                count += len(value)
            elif isinstance(value, dict):
                count += self._count_functions_in_hierarchy(value)
        return count

    def _build_module_hierarchy_tree(
        self,
        parent_container,
        hierarchy: dict,
        module_path_parts: list,
        is_root: bool = False,
    ):
        """Recursively build the hierarchical module tree."""
        for key, value in hierarchy.items():
            if key == "functions":
                # This level has functions - create a module node if there are functions
                if value:  # Only create node if there are functions
                    current_path = (
                        ".".join(module_path_parts) if module_path_parts else "unknown"
                    )
                    if is_root:
                        # For root level, add directly to tree widget
                        module_item = QTreeWidgetItem(parent_container)
                    else:
                        # For nested levels, add to parent item
                        module_item = QTreeWidgetItem(parent_container)
                    module_item.setText(0, f"{current_path} ({len(value)} functions)")
                    module_item.setData(
                        0,
                        Qt.ItemDataRole.UserRole,
                        ModuleFunctionTreeNode(current_path, tuple(value)),
                    )
            elif isinstance(value, dict):
                # This is a module part - create a tree node and recurse
                new_path_parts = module_path_parts + [key]

                # Count functions in this subtree
                subtree_function_count = self._count_functions_in_hierarchy(value)

                if is_root:
                    # For root level, add directly to tree widget
                    module_part_item = QTreeWidgetItem(parent_container)
                else:
                    # For nested levels, add to parent item
                    module_part_item = QTreeWidgetItem(parent_container)

                module_part_item.setText(
                    0, f"{key} ({subtree_function_count} functions)"
                )
                module_part_item.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    ModulePartTreeNode(".".join(new_path_parts)),
                )
                # Start collapsed - users can expand as needed
                module_part_item.setExpanded(False)

                # Recursively build subtree
                self._build_module_hierarchy_tree(
                    module_part_item, value, new_path_parts, is_root=False
                )

    @classmethod
    def clear_metadata_cache(cls) -> None:
        """Clear the cached metadata to force re-discovery."""
        cls._metadata_cache = None
        RegistryService.clear_metadata_cache()
        logger.info("Function metadata cache cleared")

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
            self.style_generator.generate_dialog_style()
            + "\n"
            + self.style_generator.generate_tree_widget_style()
            + "\n"
            + self.style_generator.generate_table_widget_style()
            + "\n"
            + self.style_generator.generate_button_style()
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
        self.function_table_browser.set_items(self.all_functions_metadata)

        # Update count label
        total = len(self.all_functions_metadata)
        filtered = len(self.function_table_browser.filtered_items)
        self.function_table_browser.status_label.setText(
            f"Functions: {filtered}/{total}"
        )

    def on_tree_selection_changed(self):
        """Handle tree selection using mathematical simplification (RST principle)."""
        selected_items = self.module_tree.selectedItems()

        # If no items selected, show all functions
        if not selected_items:
            self._update_filtered_view(
                self.all_functions_metadata, "showing all functions"
            )
            return

        item = selected_items[0]
        data = item.data(0, Qt.ItemDataRole.UserRole)

        if isinstance(data, FunctionTreeNode):
            filtered = data.filtered_functions(
                self.all_functions_metadata, self._extract_module_path
            )
            self._update_filtered_view(filtered, data.filter_description())
        else:
            # No data means show all functions
            self._update_filtered_view(
                self.all_functions_metadata, "showing all functions"
            )

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

    def _on_function_selected(self, key: str, item: FunctionMetadata):
        """Handle function selection from table browser."""
        self._set_selection_state(item.func, True)

    def _on_function_double_clicked(self, key: str, item: FunctionMetadata):
        """Handle function double-click from table browser."""
        self.selected_function = item.func
        self.accept_selection()

    def accept_selection(self):
        """Accept the selected function."""
        if self.selected_function:
            self.function_selected.emit(self.selected_function)
            self.accept()

    def get_selected_function(self) -> Optional[Callable]:
        """Get the selected function."""
        return self.selected_function

    @staticmethod
    def select_function(
        current_function: Optional[Callable] = None, parent=None
    ) -> Optional[Callable]:
        """
        Static method to show function selector and return selected function.

        Args:
            current_function: Currently selected function (for highlighting)
            parent: Parent widget

        Returns:
            Selected function or None if cancelled
        """
        dialog = FunctionSelectorDialog(current_function, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_selected_function()
        return None
