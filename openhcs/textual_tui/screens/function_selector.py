"""Function selector modal screen."""

from typing import Callable, Optional, List, Tuple
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Input, Tree, Button, Static
from textual.widgets.tree import TreeNode

from openhcs.textual_tui.widgets.floating_window import BaseFloatingWindow
from openhcs.textual_tui.services.function_registry_service import FunctionRegistryService


class FunctionSelectorScreen(BaseFloatingWindow):
    """Modal screen for selecting functions from the registry."""

    def __init__(self, current_function: Optional[Callable] = None, **kwargs):
        """Initialize function selector.
        
        Args:
            current_function: Currently selected function (for highlighting)
        """
        self.current_function = current_function
        self.selected_function = None
        self.functions_by_backend = {}
        self.all_functions = []
        
        # Load function data
        self._load_function_data()
        
        super().__init__(title="Select Function", **kwargs)

    def _load_function_data(self) -> None:
        """Load function data from registry."""
        registry_service = FunctionRegistryService()
        self.functions_by_backend = registry_service.get_functions_by_backend()
        
        # Flatten for search
        self.all_functions = []
        for backend, functions in self.functions_by_backend.items():
            for func, display_name in functions:
                self.all_functions.append((func, display_name, backend))

    def compose_content(self) -> ComposeResult:
        """Compose the function selector content."""
        with Vertical():
            # Search input
            yield Input(
                placeholder="Search functions...",
                id="search_input"
            )
            
            # Function tree
            yield self._build_function_tree()

    def compose_buttons(self) -> ComposeResult:
        """Compose dialog buttons."""
        yield Button("Select", id="select_btn", variant="primary", compact=True, disabled=True)
        yield Button("Cancel", id="cancel_btn", compact=True)

    def _build_function_tree(self) -> Tree:
        """Build tree widget with functions grouped by backend."""
        tree = Tree("Functions", id="function_tree")
        
        # Add backend nodes
        for backend, functions in self.functions_by_backend.items():
            backend_node = tree.root.add(f"{backend} ({len(functions)} functions)")
            backend_node.data = {"type": "backend", "name": backend}
            
            # Add function nodes
            for func, display_name in functions:
                func_node = backend_node.add(display_name)
                func_node.data = {"type": "function", "func": func, "name": display_name}
                
                # Highlight current function
                if func == self.current_function:
                    func_node.expand()
                    backend_node.expand()
        
        return tree

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "search_input":
            self._filter_functions(event.value)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        if event.node.data and event.node.data.get("type") == "function":
            self.selected_function = event.node.data["func"]
            # Enable select button
            select_btn = self.query_one("#select_btn", Button)
            select_btn.disabled = False
        else:
            self.selected_function = None
            # Disable select button
            select_btn = self.query_one("#select_btn", Button)
            select_btn.disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "select_btn" and self.selected_function:
            self.dismiss(self.selected_function)
        elif event.button.id == "cancel_btn":
            self.dismiss(None)

    def _filter_functions(self, search_term: str) -> None:
        """Filter functions based on search term."""
        tree = self.query_one("#function_tree", Tree)
        
        if not search_term.strip():
            # Show all functions
            tree.clear()
            tree.root.label = "Functions"
            self._populate_tree(tree, self.functions_by_backend)
        else:
            # Filter functions
            search_lower = search_term.lower()
            filtered_functions = {}
            
            for backend, functions in self.functions_by_backend.items():
                matching_functions = [
                    (func, display_name) for func, display_name in functions
                    if search_lower in display_name.lower()
                ]
                if matching_functions:
                    filtered_functions[backend] = matching_functions
            
            tree.clear()
            tree.root.label = f"Functions (filtered: {search_term})"
            self._populate_tree(tree, filtered_functions)

    def _populate_tree(self, tree: Tree, functions_by_backend: dict) -> None:
        """Populate tree with function data."""
        for backend, functions in functions_by_backend.items():
            backend_node = tree.root.add(f"{backend} ({len(functions)} functions)")
            backend_node.data = {"type": "backend", "name": backend}
            
            for func, display_name in functions:
                func_node = backend_node.add(display_name)
                func_node.data = {"type": "function", "func": func, "name": display_name}
                
                # Highlight current function
                if func == self.current_function:
                    func_node.expand()
                    backend_node.expand()
