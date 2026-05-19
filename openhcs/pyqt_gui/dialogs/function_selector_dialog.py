"""
Function Selector Dialog for PyQt6 GUI.

Mirrors the Textual TUI FunctionSelectorWindow functionality using the same
FunctionRegistryService and business logic.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Dict, Any, Iterable, Mapping

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
    QWidget,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from metaclass_registry import AutoRegisterMeta

# Use the registry service from correct location
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.lib_registry.unified_registry import FunctionMetadata
from openhcs.processing.custom_functions.signals import custom_function_signals
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.theming import StyleSheetGenerator
from pyqt_reactive.widgets.shared.function_table_browser import FunctionTableBrowser
from pyqt_reactive.widgets.shared.column_filter_widget import MultiColumnFilterPanel

logger = logging.getLogger(__name__)


FunctionMetadataMap = Mapping[str, Dict[str, Any]]


def _contract_display_value(contract: object) -> str:
    if contract is None:
        return "unknown"
    if isinstance(contract, Enum):
        return contract.name
    return str(contract)


@dataclass(frozen=True)
class FunctionFilterColumn:
    """Column-specific extraction and match semantics for the filter panel."""

    label: str
    values_for: Callable[[Dict[str, Any]], Iterable[str]]

    def display_values(self, metadata: Dict[str, Any]) -> tuple[str, ...]:
        return tuple(value for value in self.values_for(metadata) if value)

    def matches(self, metadata: Dict[str, Any], allowed_values: set[str]) -> bool:
        return any(value in allowed_values for value in self.display_values(metadata))


FUNCTION_FILTER_COLUMNS = (
    FunctionFilterColumn(
        "Registry", lambda metadata: (metadata.get("registry", "unknown").title(),)
    ),
    FunctionFilterColumn(
        "Backend", lambda metadata: (metadata.get("backend", "unknown").title(),)
    ),
    FunctionFilterColumn(
        "Contract", lambda metadata: (_contract_display_value(metadata.get("contract")),)
    ),
    FunctionFilterColumn("Tags", lambda metadata: tuple(metadata.get("tags", ()))),
)
FUNCTION_FILTER_COLUMNS_BY_LABEL = {
    column.label: column for column in FUNCTION_FILTER_COLUMNS
}


class FunctionTreeNode(ABC, metaclass=AutoRegisterMeta):
    """Domain object stored in the module tree instead of stringly item data."""

    __registry_key__ = "__name__"
    __skip_if_no_key__ = True

    @abstractmethod
    def filtered_functions(
        self,
        all_functions_metadata: FunctionMetadataMap,
        module_path_for: Callable[[Dict[str, Any]], str],
    ) -> Dict[str, Dict[str, Any]]:
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
        module_path_for: Callable[[Dict[str, Any]], str],
    ) -> Dict[str, Dict[str, Any]]:
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
        module_path_for: Callable[[Dict[str, Any]], str],
    ) -> Dict[str, Dict[str, Any]]:
        return {
            name: metadata
            for name, metadata in all_functions_metadata.items()
            if module_path_for(metadata).startswith(self.full_path)
        }

    def filter_description(self) -> str:
        return f"filtered by module part: {self.full_path}"


# Direct registry-based library detection (robust, no pattern matching)
class LibraryDetector:
    """Direct library detection using registry ownership data."""

    # Class-level cache for efficient lookup (OpenHCS caching pattern)
    _registry_functions: Optional[Dict[str, str]] = None
    _registry_display_names: Optional[Dict[str, str]] = None

    @classmethod
    def get_function_library(cls, func_name: str, module_path: str) -> str:
        """Get library for a function using direct registry ownership (robust approach)."""
        if cls._registry_functions is None:
            cls._build_registry_ownership_cache()

        # Direct lookup by function name (most reliable)
        if func_name in cls._registry_functions:
            return cls._registry_functions[func_name]

        # Fallback: check by registry library name in module path
        for registry_name, display_name in cls._registry_display_names.items():
            if registry_name.lower() in module_path.lower():
                return display_name

        return "Unknown"

    @classmethod
    def _build_registry_ownership_cache(cls):
        """Build cache of function ownership using FunctionRegistryService."""
        cls._registry_functions = {}
        cls._registry_display_names = {}

        # Use existing RegistryService (already has discovery and caching)
        unified_functions = RegistryService.get_all_functions_with_metadata()

        # Build ownership cache from unified metadata
        for func_name, metadata in unified_functions.items():
            # Extract library name from tags or module
            library_name = cls._extract_library_name(metadata)
            cls._registry_functions[func_name] = library_name

    @classmethod
    def _extract_library_name(cls, metadata) -> str:
        """Extract library name from function metadata."""
        # Use tags first (most reliable)
        if metadata.tags:
            primary_tag = metadata.tags[0]
            return primary_tag.title()

        # Fallback to module path analysis
        module_path = metadata.module.lower()
        if "openhcs" in module_path:
            return "OpenHCS"
        elif "cupy" in module_path:
            return "CuPy"
        elif "pyclesperanto" in module_path or "cle" in module_path:
            return "Pyclesperanto"
        elif "skimage" in module_path:
            return "scikit-image"

        return "Unknown"


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
        self._build_column_filters()
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

        # Load ALL functions from registries directly
        logger.info("Loading ALL functions from registries")

        # Use existing RegistryService (already has caching and discovery)
        unified_functions = RegistryService.get_all_functions_with_metadata()

        self.all_functions_metadata = {}

        # Convert FunctionMetadata to dict format for UI compatibility
        # Handle composite keys (backend:function_name) from registry service
        for composite_key, metadata in unified_functions.items():
            # Extract backend and function name from composite key
            if ":" in composite_key:
                backend, func_name = composite_key.split(":", 1)
            else:
                # Fallback for non-composite keys
                backend = (
                    metadata.registry.library_name if metadata.registry else "unknown"
                )
                func_name = composite_key

            self.all_functions_metadata[composite_key] = {
                "name": metadata.name,
                "func": metadata.func,
                "module": metadata.module,
                "contract": metadata.contract,
                "tags": metadata.tags,
                "doc": metadata.doc,
                "backend": metadata.get_memory_type(),  # Actual memory type (cupy, numpy, etc.)
                "registry": metadata.get_registry_name(),  # Registry source (openhcs, skimage, etc.)
                "metadata": metadata,  # Store full metadata for access to new methods
            }

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
        self, filtered_functions: Dict[str, Any], filter_description: str = ""
    ):
        """Update filtered view using table browser."""
        self.filtered_functions = filtered_functions
        self.populate_function_table(self.filtered_functions)

        # Create unified count display in the browser's status label
        total_count = len(self.all_functions_metadata)
        filtered_count = len(self.filtered_functions)
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

    def _determine_library(self, metadata) -> str:
        """Direct library determination using registry ownership (robust approach)."""
        func_name = metadata.get("name", "")
        module = metadata.get("module", "")

        # Use direct registry ownership lookup (eliminates pattern matching fragility)
        return LibraryDetector.get_function_library(func_name, module)

    def _extract_module_path(self, metadata) -> str:
        """Extract full module path from metadata for hierarchical tree building."""
        module = metadata.get("module", "")
        if not module:
            return "unknown"

        # Return the full module path for hierarchical tree building
        return module

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

        # Column filter panel (Library, Backend, Contract, Tags)
        self.column_filter_panel = MultiColumnFilterPanel(
            color_scheme=self.color_scheme
        )
        self.column_filter_panel.setVisible(False)  # Hidden until populated
        left_layout.addWidget(self.column_filter_panel)

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

        # Column filter panel
        self.column_filter_panel.filters_changed.connect(
            self._on_column_filters_changed
        )

    def populate_function_table(
        self, functions_metadata: Optional[Dict[str, FunctionMetadata]] = None
    ):
        """Populate function table using FunctionTableBrowser."""
        if functions_metadata is None:
            functions_metadata = self.filtered_functions

        self.function_table_browser.set_items(functions_metadata)

        # Update count label
        total = len(self.all_functions_metadata)
        filtered = len(functions_metadata)
        self.function_table_browser.status_label.setText(
            f"Functions: {filtered}/{total}"
        )

    def _build_column_filters(self):
        """Build column filter widgets from function metadata."""
        if not self.all_functions_metadata:
            return

        self.column_filter_panel.clear_all_filters()

        for column in FUNCTION_FILTER_COLUMNS:
            unique_values = set()

            for metadata in self.all_functions_metadata.values():
                unique_values.update(column.display_values(metadata))

            if unique_values:
                self.column_filter_panel.add_column_filter(
                    column.label, sorted(list(unique_values))
                )

        if self.column_filter_panel.column_filters:
            self.column_filter_panel.setVisible(True)

    def _on_column_filters_changed(self):
        """Handle column filter checkbox changes."""
        active_filters = self.column_filter_panel.get_active_filters()

        if not active_filters:
            # No filters active - show all
            self._update_filtered_view(self.all_functions_metadata)
            return

        # Apply filters with AND logic across columns
        filtered = {}
        for key, metadata in self.all_functions_metadata.items():
            matches = True

            for column_name, allowed_values in active_filters.items():
                column = FUNCTION_FILTER_COLUMNS_BY_LABEL.get(column_name)
                if column is None:
                    continue

                if not column.matches(metadata, set(allowed_values)):
                    matches = False
                    break

            if matches:
                filtered[key] = metadata

        self._update_filtered_view(filtered, "filtered by column")

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

    def _on_function_selected(self, key: str, item: Dict[str, Any]):
        """Handle function selection from table browser."""
        func = item.get("func")
        self._set_selection_state(func, func is not None)

    def _on_function_double_clicked(self, key: str, item: Dict[str, Any]):
        """Handle function double-click from table browser."""
        func = item.get("func")
        if func:
            self.selected_function = func
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
