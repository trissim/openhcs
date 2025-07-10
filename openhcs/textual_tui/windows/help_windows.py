"""Unified help system for displaying docstring and parameter information."""

from typing import Union, Callable, Optional
from textual import on
from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Horizontal
from textual.widgets import Button, Static, Markdown
from textual.css.query import NoMatches

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
from openhcs.textual_tui.widgets.shared.signature_analyzer import DocstringExtractor


class BaseHelpWindow(BaseOpenHCSWindow):
    """Base class for all help windows with unified button handling."""

    @on(Button.Pressed, "#close")
    def close_help(self) -> None:
        """Handle close button press."""
        self.close_window()


class DocstringHelpWindow(BaseHelpWindow):
    """Window for displaying docstring information."""

    DEFAULT_CSS = """
    DocstringHelpWindow ScrollableContainer {
        text-align: left;
        align: left top;
    }

    DocstringHelpWindow Static {
        text-align: left;
    }

    .help-summary {
        margin: 1 0;
        padding: 1;
        background: $surface;
        border: solid $primary;
        text-align: left;
    }

    .help-description {
        margin: 1 0;
        padding: 1;
        text-align: left;
    }

    .help-section-header {
        margin: 1 0 0 0;
        text-style: bold;
        color: $accent;
        text-align: left;
    }

    .help-parameter {
        margin: 0 0 0 2;
        color: $text;
        text-align: left;
    }

    .help-returns {
        margin: 0 0 0 2;
        color: $text;
        text-align: left;
    }

    .help-examples {
        margin: 1 0;
        padding: 1;
        background: $surface;
        border: solid $accent;
        text-align: left;
    }
    """

    def __init__(self, target: Union[Callable, type], title: Optional[str] = None, **kwargs):
        self.target = target
        self.docstring_info = DocstringExtractor.extract(target)

        # Generate title from target if not provided
        if title is None:
            if hasattr(target, '__name__'):
                title = f"Help: {target.__name__}"
            else:
                title = "Help"

        super().__init__(
            window_id="docstring_help",
            title=title,
            mode="temporary",
            **kwargs
        )

        # Calculate dynamic minimum size based on content
        self._calculate_dynamic_size()

    def compose(self) -> ComposeResult:
        """Compose the help window content with scrollable area."""
        # Scrollable content area
        with ScrollableContainer():
            # Function/class summary
            if self.docstring_info.summary:
                yield Static(f"[bold]{self.docstring_info.summary}[/bold]", classes="help-summary")

            # Full description
            if self.docstring_info.description:
                yield Markdown(self.docstring_info.description, classes="help-description")

            # Parameters section
            if self.docstring_info.parameters:
                yield Static("[bold]Parameters[/bold]", classes="help-section-header")
                for param_name, param_desc in self.docstring_info.parameters.items():
                    yield Static(f"• [bold]{param_name}[/bold]: {param_desc}", classes="help-parameter")
                    # Add spacing between parameters for better readability
                    yield Static("", classes="help-parameter-spacer")

            # Returns section
            if self.docstring_info.returns:
                yield Static("[bold]Returns:[/bold]", classes="help-section-header")
                yield Static(f"• {self.docstring_info.returns}", classes="help-returns")

            # Examples section
            if self.docstring_info.examples:
                yield Static("[bold]Examples:[/bold]", classes="help-section-header")
                yield Markdown(f"```python\n{self.docstring_info.examples}\n```", classes="help-examples")

        # Close button at bottom - wrapped in Horizontal for automatic centering
        with Horizontal():
            yield Button("Close", id="close", compact=True)

    def _calculate_dynamic_size(self) -> None:
        """Calculate and set dynamic minimum window size based on content."""
        try:
            # Calculate width based on longest line in content
            max_width = 40  # Base minimum width

            # Check summary length
            if self.docstring_info.summary:
                max_width = max(max_width, len(self.docstring_info.summary) + 10)

            # Check parameter descriptions
            if self.docstring_info.parameters:
                for param_name, param_desc in self.docstring_info.parameters.items():
                    line_length = len(f"• {param_name}: {param_desc}")
                    max_width = max(max_width, line_length + 10)

            # Check returns description
            if self.docstring_info.returns:
                max_width = max(max_width, len(self.docstring_info.returns) + 10)

            # Calculate height based on content sections
            min_height = 10  # Base minimum height
            content_lines = 0

            if self.docstring_info.summary:
                content_lines += 2  # Summary + margin
            if self.docstring_info.description:
                # Estimate lines for description (rough approximation)
                content_lines += max(3, len(self.docstring_info.description) // 60)
            if self.docstring_info.parameters:
                content_lines += 1 + len(self.docstring_info.parameters)  # Header + params
            if self.docstring_info.returns:
                content_lines += 2  # Header + returns
            if self.docstring_info.examples:
                content_lines += 5  # Header + example block

            content_lines += 3  # Close button + margins
            min_height = max(min_height, content_lines)

            # Cap maximum size to reasonable limits
            max_width = min(max_width, 120)
            min_height = min(min_height, 40)

            # Set dynamic window size
            self.styles.width = max_width
            self.styles.height = min_height

        except Exception:
            # Fallback to default sizes if calculation fails
            self.styles.width = 60
            self.styles.height = 20


class ParameterHelpWindow(BaseHelpWindow):
    """Window for displaying individual parameter documentation."""

    DEFAULT_CSS = """
    ParameterHelpWindow ScrollableContainer {
        text-align: left;
        align: left top;
    }

    ParameterHelpWindow Static {
        text-align: left;
    }

    .param-header {
        margin: 0 0 1 0;
        text-style: bold;
        color: $primary;
        text-align: left;
    }

    .param-description {
        margin: 1 0;
        text-align: left;
    }

    .param-no-desc {
        margin: 1 0;
        color: $text-muted;
        text-style: italic;
        text-align: left;
    }
    """

    def __init__(self, param_name: str, param_description: str, param_type: type = None, **kwargs):
        self.param_name = param_name
        self.param_description = param_description
        self.param_type = param_type

        title = f"Help: {param_name}"

        super().__init__(
            window_id="parameter_help",
            title=title,
            mode="temporary",
            **kwargs
        )

        # Calculate dynamic minimum size based on content
        self._calculate_dynamic_size()

    def compose(self) -> ComposeResult:
        """Compose the parameter help content with scrollable area."""
        # Scrollable content area
        with ScrollableContainer():
            # Parameter name and type
            type_info = f" ({self._format_type_info(self.param_type)})" if self.param_type else ""
            yield Static(f"[bold]{self.param_name}[/bold]{type_info}", classes="param-header")

            # Parameter description
            if self.param_description:
                yield Markdown(self.param_description.rstrip(), classes="param-description")
            else:
                yield Static("[dim]No description available[/dim]", classes="param-no-desc")

        # Close button at bottom - wrapped in Horizontal for automatic centering
        with Horizontal():
            yield Button("Close", id="close", compact=True)

    def _calculate_dynamic_size(self) -> None:
        """Calculate and set dynamic minimum window size based on parameter content."""
        try:
            # Calculate width based on parameter content
            max_width = 30  # Base minimum width for parameter windows

            # Check parameter name length
            max_width = max(max_width, len(self.param_name) + 15)

            # Check parameter description length
            if self.param_description:
                # Split description into lines and find longest
                desc_lines = self.param_description.split('\n')
                for line in desc_lines:
                    max_width = max(max_width, len(line) + 10)

            # Check parameter type length
            if self.param_type:
                type_str = str(self.param_type)
                max_width = max(max_width, len(f"Type: {type_str}") + 10)

            # Calculate height based on content
            min_height = 8  # Base minimum height for parameter windows
            content_lines = 2  # Parameter name + margin

            if self.param_description:
                # Estimate lines for description
                desc_lines = self.param_description.split('\n')
                content_lines += len(desc_lines) + 1  # Description lines + margin

            if self.param_type:
                content_lines += 1  # Type line

            content_lines += 3  # Close button + margins
            min_height = max(min_height, content_lines)

            # Cap maximum size to reasonable limits
            max_width = min(max_width, 100)
            min_height = min(min_height, 30)

            # Set dynamic minimum size
            self.styles.min_width = max_width
            self.styles.min_height = min_height

        except Exception:
            # Fallback to default sizes if calculation fails
            self.styles.min_width = 30
            self.styles.min_height = 8

    def _format_type_info(self, param_type) -> str:
        """Format type information for display, showing full Union types."""
        if not param_type:
            return "Unknown"

        try:
            from typing import get_origin, get_args, Union

            # Handle Union types (including Optional)
            origin = get_origin(param_type)
            if origin is Union:
                args = get_args(param_type)
                # Filter out NoneType for cleaner display
                non_none_args = [arg for arg in args if arg is not type(None)]

                if len(non_none_args) == 1 and type(None) in args:
                    # This is Optional[T] - show as "T (optional)"
                    return f"{self._format_single_type(non_none_args[0])} (optional)"
                else:
                    # This is a true Union - show all types
                    type_names = [self._format_single_type(arg) for arg in args]
                    return f"Union[{', '.join(type_names)}]"
            else:
                # Regular type
                return self._format_single_type(param_type)

        except Exception:
            # Fallback to simple name if anything goes wrong
            return getattr(param_type, '__name__', str(param_type))

    def _format_single_type(self, type_obj) -> str:
        """Format a single type for display."""
        try:
            from typing import get_origin, get_args

            origin = get_origin(type_obj)
            if origin:
                # Handle generic types like List[str], Dict[str, int]
                args = get_args(type_obj)
                if args:
                    arg_names = [self._format_single_type(arg) for arg in args]
                    return f"{origin.__name__}[{', '.join(arg_names)}]"
                else:
                    return origin.__name__
            else:
                # Simple type
                return getattr(type_obj, '__name__', str(type_obj))
        except Exception:
            return str(type_obj)


class HelpWindowManager:
    """Unified help window management system - consolidates all help window logic."""

    @staticmethod
    async def show_docstring_help(app, target: Union[Callable, type], title: Optional[str] = None):
        """Show help for a function or class using the window system."""
        try:
            window = app.query_one(DocstringHelpWindow)
            # Window exists, update it and open
            window.target = target
            window.docstring_info = DocstringExtractor.extract(target)
            window.open_state = True
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = DocstringHelpWindow(target, title)
            await app.mount(window)
            window.open_state = True

    @staticmethod
    async def show_parameter_help(app, param_name: str, param_description: str, param_type: type = None):
        """Show help for a parameter using the window system."""
        try:
            window = app.query_one(ParameterHelpWindow)
            # Window exists, update it and open
            window.param_name = param_name
            window.param_description = param_description
            window.param_type = param_type
            window.open_state = True
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = ParameterHelpWindow(param_name, param_description, param_type)
            await app.mount(window)
            window.open_state = True


class HelpableWidget:
    """Mixin class to add help functionality to widgets - uses unified manager."""

    async def show_function_help(self, target: Union[Callable, type]) -> None:
        """Show help window for a function or class."""
        if hasattr(self, 'app'):
            await HelpWindowManager.show_docstring_help(self.app, target)

    async def show_parameter_help(self, param_name: str, param_description: str, param_type: type = None) -> None:
        """Show help window for a parameter."""
        if hasattr(self, 'app'):
            await HelpWindowManager.show_parameter_help(self.app, param_name, param_description, param_type)
