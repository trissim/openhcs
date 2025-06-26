"""Dynamic Visual AST Inspector for OpenHCS debugging."""

import ast
import inspect
import importlib
import pkgutil
from typing import Dict, List, Any, Optional, Type, Union
from pathlib import Path
from textual.app import ComposeResult
from textual.widgets import Static, Tree, Button, Select, TextArea, Input, Label
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow


class ASTNodeInfo:
    """Information about an AST node."""

    def __init__(self, node: ast.AST, parent: Optional['ASTNodeInfo'] = None):
        self.node = node
        self.parent = parent
        self.children = []
        self.node_type = type(node).__name__
        self.description = self._get_description()
        self.details = self._get_details()

    def _get_description(self) -> str:
        """Get a human-readable description of the node."""
        if isinstance(self.node, ast.ClassDef):
            bases = [self._get_name(base) for base in self.node.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            return f"Class: {self.node.name}{base_str}"
        elif isinstance(self.node, ast.FunctionDef):
            args = [arg.arg for arg in self.node.args.args]
            return f"Function: {self.node.name}({', '.join(args)})"
        elif isinstance(self.node, ast.AsyncFunctionDef):
            args = [arg.arg for arg in self.node.args.args]
            return f"Async Function: {self.node.name}({', '.join(args)})"
        elif isinstance(self.node, ast.Import):
            names = [alias.name for alias in self.node.names]
            return f"Import: {', '.join(names)}"
        elif isinstance(self.node, ast.ImportFrom):
            names = [alias.name for alias in self.node.names]
            return f"From {self.node.module}: {', '.join(names)}"
        elif isinstance(self.node, ast.Assign):
            targets = [self._get_name(target) for target in self.node.targets]
            return f"Assignment: {', '.join(targets)}"
        elif isinstance(self.node, ast.AnnAssign):
            target = self._get_name(self.node.target)
            return f"Annotated Assignment: {target}"
        elif isinstance(self.node, ast.Expr):
            return f"Expression: {self._get_name(self.node.value)}"
        else:
            return f"{self.node_type}"

    def _get_details(self) -> Dict[str, Any]:
        """Get detailed information about the node."""
        details = {'type': self.node_type}

        if isinstance(self.node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            details['name'] = self.node.name
            details['docstring'] = ast.get_docstring(self.node)

        if isinstance(self.node, ast.ClassDef):
            details['bases'] = [self._get_name(base) for base in self.node.bases]
            details['decorators'] = [self._get_name(dec) for dec in self.node.decorator_list]

        if isinstance(self.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            details['args'] = [arg.arg for arg in self.node.args.args]
            details['decorators'] = [self._get_name(dec) for dec in self.node.decorator_list]
            details['returns'] = self._get_name(self.node.returns) if self.node.returns else None

        if hasattr(self.node, 'lineno'):
            details['line'] = self.node.lineno

        return details

    def _get_name(self, node: ast.AST) -> str:
        """Get the name of a node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Call):
            func_name = self._get_name(node.func)
            return f"{func_name}(...)"
        else:
            return f"<{type(node).__name__}>"


class DynamicASTAnalyzer:
    """Dynamic AST analyzer that can inspect any Python module or file."""

    @staticmethod
    def get_inspection_categories() -> Dict[str, Dict[str, str]]:
        """Get predefined categories of things to inspect."""
        return {
            "Config Classes": {
                "config_form": "openhcs.textual_tui.widgets.config_form",
                "config_dialog": "openhcs.textual_tui.screens.config_dialog",
                "config_window": "openhcs.textual_tui.windows.config_window",
                "config_form_screen": "openhcs.textual_tui.screens.config_form"
            },
            "Core Components": {
                "pipeline": "openhcs.core.pipeline",
                "orchestrator": "openhcs.core.orchestrator.orchestrator",
                "step_base": "openhcs.core.step_base",
                "config": "openhcs.core.config"
            },
            "TUI Widgets": {
                "main_content": "openhcs.textual_tui.widgets.main_content",
                "plate_manager": "openhcs.textual_tui.widgets.plate_manager",
                "function_pane": "openhcs.textual_tui.widgets.function_pane",
                "pipeline_editor": "openhcs.textual_tui.widgets.pipeline_editor"
            },
            "Processing Backends": {
                "cupy_processor": "openhcs.processing.backends.enhance.basic_processor_cupy",
                "numpy_processor": "openhcs.processing.backends.enhance.basic_processor_numpy",
                "torch_processor": "openhcs.processing.backends.enhance.n2v2_processor_torch"
            },
            "Window System": {
                "base_window": "openhcs.textual_tui.windows.base_window",
                "help_window": "openhcs.textual_tui.windows.help_window",
                "file_browser": "openhcs.textual_tui.windows.file_browser_window"
            }
        }

    @staticmethod
    def analyze_module(module_name: str) -> Optional[ASTNodeInfo]:
        """Analyze a Python module and return AST tree."""
        try:
            # Import the module to get its file path
            module = importlib.import_module(module_name)
            file_path = inspect.getfile(module)

            # Read and parse the source
            with open(file_path, 'r') as f:
                source = f.read()

            tree = ast.parse(source)
            return DynamicASTAnalyzer._build_ast_tree(tree)

        except Exception as e:
            # Create error node
            error_node = ast.Module(body=[], type_ignores=[])
            error_info = ASTNodeInfo(error_node)
            error_info.description = f"Error: {e}"
            return error_info

    @staticmethod
    def analyze_file(file_path: str) -> Optional[ASTNodeInfo]:
        """Analyze a Python file and return AST tree."""
        try:
            with open(file_path, 'r') as f:
                source = f.read()

            tree = ast.parse(source)
            return DynamicASTAnalyzer._build_ast_tree(tree)

        except Exception as e:
            # Create error node
            error_node = ast.Module(body=[], type_ignores=[])
            error_info = ASTNodeInfo(error_node)
            error_info.description = f"Error: {e}"
            return error_info

    @staticmethod
    def _build_ast_tree(node: ast.AST, parent: Optional[ASTNodeInfo] = None) -> ASTNodeInfo:
        """Recursively build AST tree."""
        node_info = ASTNodeInfo(node, parent)

        # Add children
        for child in ast.iter_child_nodes(node):
            child_info = DynamicASTAnalyzer._build_ast_tree(child, node_info)
            node_info.children.append(child_info)

        return node_info


class DebugClassExplorerWindow(BaseOpenHCSWindow):
    """Dynamic Visual AST Inspector for OpenHCS debugging."""

    DEFAULT_CSS = """
    DebugClassExplorerWindow {
        width: 95; height: 35;
        min-width: 80; min-height: 25;
    }

    .category-buttons {
        height: auto;
        width: 1fr;
    }

    .custom-input {
        height: auto;
        width: 1fr;
    }

    .ast-tree {
        width: 1fr;
        height: 1fr;
    }

    .ast-details {
        width: 1fr;
        height: 1fr;
    }

    .detail-text {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    .breadcrumbs {
        height: auto;
        width: 1fr;
        margin: 1 0;
    }

    #breadcrumb_path {
        margin: 0 1;
        color: $text-muted;
    }

    /* Use standard button spacing from global styles */
    .category-buttons Button {
        margin: 0 1;
    }

    .custom-input Button {
        margin: 0 1;
    }

    .breadcrumbs Button {
        margin: 0 1;
    }
    """

    selected_node = reactive(None)
    current_ast_tree = reactive(None)
    navigation_stack = []  # Stack for breadcrumb navigation

    def __init__(self, **kwargs):
        super().__init__(
            window_id="debug_ast_inspector",
            title="Dynamic Visual AST Inspector",
            mode="temporary",
            **kwargs
        )

        # Get inspection categories
        self.categories = DynamicASTAnalyzer.get_inspection_categories()

    def on_mount(self) -> None:
        """Initialize the window with a default view."""
        # Show available categories in the tree initially
        self._show_initial_state()
    

    
    def compose(self) -> ComposeResult:
        """Compose the dynamic AST inspector."""
        with Vertical():
            # Top panel - Category buttons and custom input
            with Horizontal(classes="category-buttons"):
                yield Static("[bold]Inspect:[/bold]")
                for category_name in self.categories.keys():
                    yield Button(category_name, id=f"cat_{category_name.lower().replace(' ', '_')}", compact=True)

            # Custom module input
            with Horizontal(classes="custom-input"):
                yield Label("Custom Module:")
                yield Input(placeholder="e.g., openhcs.core.pipeline", id="custom_module_input")
                yield Button("Analyze", id="analyze_custom", compact=True)
                yield Button("Browse File", id="browse_file", compact=True)

            # Navigation breadcrumbs
            with Horizontal(classes="breadcrumbs"):
                yield Button("🏠 Home", id="nav_home", compact=True)
                yield Button("⬅️ Back", id="nav_back", compact=True, disabled=True)
                yield Static("", id="breadcrumb_path")

            # Main content - AST tree and details
            with Horizontal():
                # Left panel - AST tree
                with Vertical(classes="ast-tree"):
                    yield Static("[bold]AST Structure[/bold]")
                    tree = Tree("🔍 Click a category button or enter a module name")
                    tree.id = "ast_tree"
                    tree.show_root = True
                    yield tree

                # Right panel - node details
                with Vertical(classes="ast-details"):
                    yield Static("[bold]Node Details[/bold]", id="details_title")

                    with ScrollableContainer(classes="detail-text"):
                        yield Static("Select a node to see details", id="node_details")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "close":
            self.close_window()
        elif button_id.startswith("cat_"):
            # Category button pressed
            category_name = button_id[4:].replace('_', ' ').title()
            self._analyze_category(category_name)
        elif button_id == "analyze_custom":
            # Custom module analysis
            self._analyze_custom_module()
        elif button_id == "browse_file":
            # File browser (placeholder for now)
            self._show_file_browser()
        elif button_id == "nav_home":
            # Navigate back to home
            self._navigate_home()
        elif button_id == "nav_back":
            # Navigate back one level
            self._navigate_back()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "custom_module_input":
            self._analyze_custom_module()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle AST tree node selection."""
        if hasattr(event.node, 'data'):
            if isinstance(event.node.data, ASTNodeInfo):
                self.selected_node = event.node.data
                self._update_node_details()
            elif isinstance(event.node.data, dict):
                data = event.node.data
                if data.get("type") == "expandable":
                    # Expand this node
                    self._expand_node(event.node, data["parent_node"])
                elif data.get("type") == "module_placeholder":
                    # Analyze this specific module
                    self._analyze_single_module(data["module_name"], data["label"])

    def _expand_node(self, tree_node, ast_node_info: ASTNodeInfo):
        """Expand a node that was previously collapsed."""
        # Remove the placeholder
        tree_node.remove()

        # Add the actual children
        parent_tree_node = tree_node.parent
        self._populate_ast_tree(parent_tree_node, ast_node_info, max_depth=3, current_depth=0)

    def _analyze_single_module(self, module_name: str, module_label: str):
        """Analyze a single module from the category view."""
        # Add to navigation stack
        self.navigation_stack.append({
            "type": "module",
            "module_name": module_name,
            "label": module_label
        })
        self._update_navigation_ui()

        tree = self.query_one("#ast_tree", Tree)
        tree.clear()
        tree.root.label = f"📦 {module_label} ({module_name})"

        try:
            # Show loading message
            try:
                details_widget = self.query_one("#node_details", Static)
                details_widget.update(f"[yellow]Analyzing {module_label}...[/yellow]")
            except:
                pass

            ast_tree = DynamicASTAnalyzer.analyze_module(module_name)
            if ast_tree:
                tree.root.data = ast_tree
                self._populate_ast_tree(tree.root, ast_tree, max_depth=3)  # Deeper analysis for single module
                tree.root.expand()

                # Update details
                try:
                    details_widget = self.query_one("#node_details", Static)
                    details_widget.update(
                        f"[bold cyan]Module: {module_label}[/bold cyan]\n"
                        f"[dim]({module_name})[/dim]\n\n"
                        f"[green]✅ Analysis complete![/green]\n"
                        f"[yellow]AST nodes:[/yellow] {len(ast_tree.children)}\n\n"
                        "[blue]Click any node to see details[/blue]\n"
                        "[green]Use navigation buttons to go back[/green]"
                    )
                except:
                    pass
            else:
                error_node = tree.root.add("❌ Failed to analyze")

        except Exception as e:
            error_node = tree.root.add(f"❌ Error: {str(e)[:50]}")
            try:
                details_widget = self.query_one("#node_details", Static)
                details_widget.update(f"[red]❌ Error analyzing {module_label}: {e}[/red]")
            except:
                pass

    def _navigate_home(self):
        """Navigate back to the home screen."""
        self.navigation_stack = []
        self._update_navigation_ui()
        self._show_initial_state()

    def _navigate_back(self):
        """Navigate back one level."""
        if not self.navigation_stack:
            return

        # Remove current level
        self.navigation_stack.pop()
        self._update_navigation_ui()

        if not self.navigation_stack:
            # Back to home
            self._show_initial_state()
        else:
            # Recreate the previous level (but don't add to stack again)
            previous = self.navigation_stack[-1]
            if previous["type"] == "category":
                # Temporarily remove from stack to avoid double-adding
                temp_item = self.navigation_stack.pop()
                self._analyze_category(previous["name"])
            elif previous["type"] == "module":
                temp_item = self.navigation_stack.pop()
                self._analyze_single_module(previous["module_name"], previous["label"])
            elif previous["type"] == "custom_module":
                temp_item = self.navigation_stack.pop()
                # Set the input field and analyze
                try:
                    input_widget = self.query_one("#custom_module_input", Input)
                    input_widget.value = previous["module_name"]
                    self._analyze_custom_module()
                except:
                    pass

    def _update_navigation_ui(self):
        """Update the navigation breadcrumbs and button states."""
        try:
            # Update back button state
            back_button = self.query_one("#nav_back", Button)
            back_button.disabled = len(self.navigation_stack) == 0

            # Update breadcrumb path
            breadcrumb = self.query_one("#breadcrumb_path", Static)
            if not self.navigation_stack:
                breadcrumb.update("🏠 Home")
            else:
                path_parts = ["🏠"] + [item["label"] for item in self.navigation_stack]
                breadcrumb.update(" → ".join(path_parts))
        except:
            pass

    def _show_initial_state(self):
        """Show initial state with available categories."""
        # Clear navigation stack
        self.navigation_stack = []
        self._update_navigation_ui()

        tree = self.query_one("#ast_tree", Tree)
        tree.clear()
        tree.root.label = "🔍 Available Analysis Categories"

        for category_name, modules in self.categories.items():
            category_node = tree.root.add(f"📂 {category_name} ({len(modules)} modules)")
            for module_label, module_name in modules.items():
                module_node = category_node.add(f"📄 {module_label}")
                module_node.data = {"type": "module_placeholder", "module_name": module_name, "label": module_label}

        tree.root.expand()

        # Update details
        try:
            details_widget = self.query_one("#node_details", Static)
            details_widget.update(
                "[bold cyan]Dynamic AST Inspector[/bold cyan]\n\n"
                "[yellow]Instructions:[/yellow]\n"
                "• Click a category button to analyze all modules in that category\n"
                "• Enter a custom module name and click 'Analyze'\n"
                "• Click any node in the AST tree to see detailed information\n"
                "• Use 🏠 Home or ⬅️ Back buttons to navigate\n\n"
                f"[green]Available Categories:[/green]\n" +
                "\n".join(f"• {name}: {len(modules)} modules" for name, modules in self.categories.items())
            )
        except:
            pass

    def _analyze_category(self, category_name: str):
        """Analyze all modules in a category."""
        if category_name not in self.categories:
            return

        # Add to navigation stack
        self.navigation_stack.append({
            "type": "category",
            "name": category_name,
            "label": f"Category: {category_name}"
        })
        self._update_navigation_ui()

        tree = self.query_one("#ast_tree", Tree)
        tree.clear()
        tree.root.label = f"📂 {category_name} - AST Analysis"

        modules = self.categories[category_name]
        for module_label, module_name in modules.items():
            try:
                ast_tree = DynamicASTAnalyzer.analyze_module(module_name)
                if ast_tree:
                    module_node = tree.root.add(f"📦 {module_label}")
                    module_node.data = ast_tree
                    self._populate_ast_tree(module_node, ast_tree, max_depth=1)  # Shallow depth for category view
                else:
                    error_node = tree.root.add(f"❌ {module_label} (failed to analyze)")
            except Exception as e:
                error_node = tree.root.add(f"❌ {module_label} (error: {str(e)[:50]})")

        tree.root.expand()

        # Update details to show category info
        try:
            details_widget = self.query_one("#node_details", Static)
            details_widget.update(
                f"[bold cyan]Category: {category_name}[/bold cyan]\n\n"
                f"[yellow]Modules analyzed:[/yellow] {len(modules)}\n\n"
                "[green]Click any module to see detailed AST[/green]\n"
                "[blue]Use navigation buttons to go back[/blue]"
            )
        except:
            pass

    def _analyze_custom_module(self):
        """Analyze a custom module entered by user."""
        try:
            input_widget = self.query_one("#custom_module_input", Input)
            module_name = input_widget.value.strip()

            if not module_name:
                # Show help message
                try:
                    details_widget = self.query_one("#node_details", Static)
                    details_widget.update("[yellow]Please enter a module name (e.g., openhcs.core.config)[/yellow]")
                except:
                    pass
                return

            # Add to navigation stack
            self.navigation_stack.append({
                "type": "custom_module",
                "module_name": module_name,
                "label": f"Custom: {module_name}"
            })
            self._update_navigation_ui()

            tree = self.query_one("#ast_tree", Tree)
            tree.clear()
            tree.root.label = f"📦 Custom: {module_name}"

            # Show loading message
            try:
                details_widget = self.query_one("#node_details", Static)
                details_widget.update(f"[yellow]Analyzing module: {module_name}...[/yellow]")
            except:
                pass

            ast_tree = DynamicASTAnalyzer.analyze_module(module_name)
            if ast_tree:
                tree.root.data = ast_tree
                self._populate_ast_tree(tree.root, ast_tree, max_depth=3)
                tree.root.expand()

                # Update details with success message
                try:
                    details_widget = self.query_one("#node_details", Static)
                    details_widget.update(
                        f"[bold cyan]Custom Module: {module_name}[/bold cyan]\n\n"
                        f"[green]✅ Analysis complete![/green]\n"
                        f"[yellow]AST nodes found:[/yellow] {len(ast_tree.children)}\n\n"
                        "[blue]Click any node in the tree to see details[/blue]\n"
                        "[green]Use navigation buttons to go back[/green]"
                    )
                except:
                    pass
            else:
                # Show error in tree
                error_node = tree.root.add("❌ Failed to analyze module")
                try:
                    details_widget = self.query_one("#node_details", Static)
                    details_widget.update(f"[red]❌ Failed to analyze module: {module_name}[/red]")
                except:
                    pass

        except Exception as e:
            # Show error in details
            try:
                details_widget = self.query_one("#node_details", Static)
                details_widget.update(f"[red]❌ Error analyzing module: {e}[/red]")
            except:
                pass

            # Show error in tree
            tree = self.query_one("#ast_tree", Tree)
            tree.clear()
            tree.root.label = f"❌ Error: {module_name}"
            error_node = tree.root.add(f"Error: {str(e)[:100]}")

    def _show_file_browser(self):
        """Show file browser (placeholder)."""
        try:
            details_widget = self.query_one("#node_details", Static)
            details_widget.update("[yellow]File browser not implemented yet. Use custom module input instead.[/yellow]")
        except:
            pass

    def _populate_ast_tree(self, parent_node, ast_node_info: ASTNodeInfo, max_depth: int = 2, current_depth: int = 0):
        """Recursively populate the AST tree with depth control."""
        if current_depth >= max_depth:
            return

        for child in ast_node_info.children:
            # Choose appropriate icon based on node type
            icon = self._get_node_icon(child.node_type)
            child_node = parent_node.add(f"{icon} {child.description}")
            child_node.data = child

            # Recursively add children up to max depth
            if child.children and current_depth < max_depth - 1:
                self._populate_ast_tree(child_node, child, max_depth, current_depth + 1)
            elif child.children:
                # Add a placeholder to show there are more children
                placeholder = child_node.add(f"📁 ... {len(child.children)} more nodes (click to expand)")
                placeholder.data = {"type": "expandable", "parent_node": child}

    def _get_node_icon(self, node_type: str) -> str:
        """Get appropriate icon for AST node type."""
        icons = {
            "ClassDef": "🏛️",
            "FunctionDef": "⚙️",
            "AsyncFunctionDef": "⚡",
            "Import": "📥",
            "ImportFrom": "📦",
            "Assign": "📝",
            "AnnAssign": "🏷️",
            "Expr": "💭",
            "If": "🔀",
            "For": "🔄",
            "While": "🔁",
            "Try": "🛡️",
            "With": "🔒",
            "Return": "↩️",
            "Yield": "⤴️"
        }
        return icons.get(node_type, "🔹")

    def _update_node_details(self):
        """Update the node details panel."""
        if not self.selected_node:
            return

        node_info = self.selected_node
        details = []

        details.append(f"[bold cyan]AST Node: {node_info.node_type}[/bold cyan]")
        details.append(f"[dim]Description: {node_info.description}[/dim]")
        details.append("")

        # Show node details
        for key, value in node_info.details.items():
            if value is not None:
                if isinstance(value, list) and value:
                    details.append(f"[bold yellow]{key.title()}:[/bold yellow]")
                    for item in value:
                        details.append(f"  • {item}")
                elif not isinstance(value, list):
                    details.append(f"[bold yellow]{key.title()}:[/bold yellow] {value}")

        # Show children count
        if node_info.children:
            details.append("")
            details.append(f"[bold blue]Children:[/bold blue] {len(node_info.children)}")
            for child in node_info.children[:5]:
                details.append(f"  • {child.description}")
            if len(node_info.children) > 5:
                details.append(f"  ... and {len(node_info.children) - 5} more")

        # Update the details widget
        try:
            details_widget = self.query_one("#node_details", Static)
            details_widget.update("\n".join(details))
        except Exception:
            pass
