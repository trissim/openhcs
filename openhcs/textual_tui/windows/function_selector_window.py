"""Function selector window for selecting functions from the registry."""

from typing import Callable, Optional, Dict
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Input, DataTable, Button, Static, Tree

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.lib_registry.unified_registry import FunctionMetadata


class FunctionSelectorWindow(BaseOpenHCSWindow):
    """Window for selecting functions from the registry."""

    DEFAULT_CSS = """
    FunctionSelectorWindow {
        width: 90; height: 30;
        min-width: 90; min-height: 30;
    }

    .left-pane {
        width: 30%;  /* Reduced from 40% to take up less width */
        border-right: solid $primary;
    }

    .right-pane {
        width: 70%;  /* Increased from 60% to give more space to table */
    }

    .pane-title {
        text-align: center;
        text-style: bold;
        background: $primary;
        color: $text;
        height: 1;
    }
    """

    def __init__(self, current_function: Optional[Callable] = None, on_result_callback: Optional[Callable] = None, **kwargs):
        """Initialize function selector window.

        Args:
            current_function: Currently selected function (for highlighting)
            on_result_callback: Callback function to handle the result
        """
        self.current_function = current_function
        self.selected_function = None
        self.all_functions_metadata: Dict[str, FunctionMetadata] = {}
        self.filtered_functions: Dict[str, FunctionMetadata] = {}
        self.on_result_callback = on_result_callback

        # Load function data with enhanced metadata
        self._load_function_data()

        super().__init__(
            window_id="function_selector",
            title="Select Function - Dual Pane View",
            mode="temporary",
            **kwargs
        )

    def _load_function_data(self) -> None:
        """Load function data with enhanced metadata from registry."""
        self.all_functions_metadata = dict(
            RegistryService.get_all_functions_with_metadata()
        )
        self.filtered_functions = self.all_functions_metadata.copy()

    def compose(self) -> ComposeResult:
        """Compose the dual-pane function selector content."""
        with Vertical():
            # Search input
            yield Input(
                placeholder="Search functions by name, module, contract, or tags...",
                id="search_input"
            )

            # Function count display
            yield Static(f"Functions: {len(self.all_functions_metadata)}", id="function_count")

            # Dual-pane layout
            with Horizontal():
                # Left pane: Hierarchical tree view
                with Vertical(classes="left-pane"):
                    yield self._build_module_tree()

                # Right pane: Enhanced table view
                with Vertical(classes="right-pane"):
                    yield Static("Function Details", classes="pane-title")
                    yield self._build_function_table()

            # Buttons - use unified dialog-buttons class for centered alignment
            with Horizontal(classes="dialog-buttons"):
                yield Button("Select", id="select_btn", variant="primary", compact=True, disabled=True)
                yield Button("Cancel", id="cancel_btn", compact=True)

    def _build_function_table(self) -> DataTable:
        """Build table widget with enhanced function metadata."""
        table = DataTable(id="function_table", cursor_type="row")

        # Add columns with sorting support - Backend shows memory type, Registry shows source
        table.add_column("Name", key="name")
        table.add_column("Module", key="module")
        table.add_column("Backend", key="backend")
        table.add_column("Registry", key="registry")
        table.add_column("Contract", key="contract")
        table.add_column("Tags", key="tags")
        table.add_column("Description", key="description")

        # Populate table with function data
        self._populate_table(table, self.filtered_functions)

        return table

    def _build_module_tree(self) -> Tree:
        """Build hierarchical tree widget showing module structure based purely on module paths."""
        tree = Tree("Module Structure", id="module_tree")
        # Start with tree collapsed - users can expand as needed
        tree.root.collapse()

        # Build hierarchical structure directly from module paths
        module_hierarchy = {}
        for func_name, metadata in self.all_functions_metadata.items():
            module_path = self._extract_module_path(metadata)
            # Build hierarchical structure by splitting module path on '.'
            self._add_function_to_hierarchy(module_hierarchy, module_path, func_name)

        # Build tree structure directly from module hierarchy
        self._build_module_hierarchy_tree(tree.root, module_hierarchy, [])

        return tree

    def _extract_module_path(self, metadata: FunctionMetadata) -> str:
        """Extract meaningful module path for display."""
        module = metadata.module

        # For OpenHCS functions, show the backend structure
        if 'openhcs' in metadata.tags:
            parts = module.split('.')
            # Find the backends part and show from there
            try:
                backends_idx = parts.index('backends')
                return '.'.join(parts[backends_idx+1:])  # Skip 'backends'
            except ValueError:
                return module.split('.')[-1]  # Just the last part

        # For external libraries, show the meaningful part
        elif 'skimage' in module:
            parts = module.split('.')
            try:
                skimage_idx = parts.index('skimage')
                return '.'.join(parts[skimage_idx+1:]) or 'core'
            except ValueError:
                return module.split('.')[-1]

        elif 'pyclesperanto' in module or 'cle' in module:
            return 'pyclesperanto_prototype'

        elif 'cupy' in module.lower():
            parts = module.split('.')
            try:
                cupy_idx = next(i for i, part in enumerate(parts) if 'cupy' in part.lower())
                return '.'.join(parts[cupy_idx+1:]) or 'core'
            except (StopIteration, IndexError):
                return module.split('.')[-1]

        return module.split('.')[-1]

    def _populate_table(self, table: DataTable, functions_metadata: Dict[str, FunctionMetadata]) -> None:
        """Populate table with function metadata."""
        table.clear()

        for composite_key, metadata in functions_metadata.items():
            # Get actual memory type (backend) and registry name separately
            memory_type = metadata.get_memory_type()
            registry_name = metadata.get_registry_name()

            # Format tags as comma-separated string
            tags_str = ", ".join(metadata.tags) if metadata.tags else ""

            # Truncate description for table display
            description = metadata.doc[:50] + "..." if len(metadata.doc) > 50 else metadata.doc

            row_key = table.add_row(
                metadata.display_name,
                metadata.module.split('.')[-1] if metadata.module else "unknown",  # Show only last part of module
                (memory_type or "").title(),
                registry_name.title(),  # Show registry source (openhcs, skimage, etc.)
                metadata.contract.name if metadata.contract else "unknown",
                tags_str,
                description,
                key=composite_key  # Use composite key for row identification
            )

            # Store function reference for selection
            table.get_row(row_key).metadata = {"func": metadata.func, "metadata": metadata}

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "search_input":
            self._filter_functions(event.value)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection to filter table."""
        if event.node.data:
            node_type = event.node.data.get("type")

            if node_type == "module":
                # Filter table to show only functions from this module
                module_functions = event.node.data["functions"]
                self.filtered_functions = {
                    name: metadata for name, metadata in self.all_functions_metadata.items()
                    if name in module_functions
                }

                # Update table and count
                table = self.query_one("#function_table", DataTable)
                self._populate_table(table, self.filtered_functions)

                count_label = self.query_one("#function_count", Static)
                count_label.update(f"Functions: {len(self.filtered_functions)}/{len(self.all_functions_metadata)} (filtered by module)")

        # Clear function selection when tree selection changes
        self.selected_function = None
        select_btn = self.query_one("#select_btn", Button)
        select_btn.disabled = True

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle table row selection."""
        table = self.query_one("#function_table", DataTable)

        if event.row_key:
            # Get function from filtered metadata using composite key
            composite_key = str(event.row_key.value)
            if composite_key in self.filtered_functions:
                metadata = self.filtered_functions[composite_key]
                self.selected_function = metadata.func

                # Enable select button
                select_btn = self.query_one("#select_btn", Button)
                select_btn.disabled = False
            else:
                self.selected_function = None
                select_btn = self.query_one("#select_btn", Button)
                select_btn.disabled = True
        else:
            self.selected_function = None
            select_btn = self.query_one("#select_btn", Button)
            select_btn.disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "select_btn" and self.selected_function:
            if self.on_result_callback:
                self.on_result_callback(self.selected_function)
            self.close_window()
        elif event.button.id == "cancel_btn":
            if self.on_result_callback:
                self.on_result_callback(None)
            self.close_window()

    def _filter_functions(self, search_term: str) -> None:
        """Filter functions based on search term across multiple fields."""
        table = self.query_one("#function_table", DataTable)
        count_label = self.query_one("#function_count", Static)

        search_term = search_term.strip()

        if not search_term or len(search_term) < 2:
            # Show all functions if empty or less than 2 characters (performance optimization)
            self.filtered_functions = self.all_functions_metadata.copy()
        else:
            # Filter functions across multiple fields
            search_lower = search_term.lower()
            self.filtered_functions = {}

            for func_name, metadata in self.all_functions_metadata.items():
                # Search in name, module, contract, tags, and description
                searchable_text = " ".join([
                    metadata.name.lower(),
                    metadata.module.lower(),
                    metadata.contract.name.lower() if metadata.contract else "",
                    " ".join(metadata.tags).lower(),
                    metadata.doc.lower()
                ])

                if search_lower in searchable_text:
                    self.filtered_functions[func_name] = metadata

        # Update table and count
        self._populate_table(table, self.filtered_functions)
        count_label.update(f"Functions: {len(self.filtered_functions)}/{len(self.all_functions_metadata)}")

        # Clear selection when filtering
        self.selected_function = None
        select_btn = self.query_one("#select_btn", Button)
        select_btn.disabled = True
