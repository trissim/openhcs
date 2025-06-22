"""Help windows for displaying docstring and parameter information."""

from typing import Union, Callable, Optional
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static, Markdown

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
from openhcs.textual_tui.widgets.shared.signature_analyzer import DocstringExtractor


class DocstringHelpWindow(BaseOpenHCSWindow):
    """Window for displaying docstring information."""

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

    def compose(self) -> ComposeResult:
        """Compose the help window content."""
        with Vertical():
            # Function/class summary
            if self.docstring_info.summary:
                yield Static(f"[bold]{self.docstring_info.summary}[/bold]", classes="help-summary")
            
            # Full description
            if self.docstring_info.description:
                yield Markdown(self.docstring_info.description, classes="help-description")
            
            # Parameters section
            if self.docstring_info.parameters:
                yield Static("[bold]Parameters:[/bold]", classes="help-section-header")
                for param_name, param_desc in self.docstring_info.parameters.items():
                    yield Static(f"• [bold]{param_name}[/bold]: {param_desc}", classes="help-parameter")
            
            # Returns section
            if self.docstring_info.returns:
                yield Static("[bold]Returns:[/bold]", classes="help-section-header")
                yield Static(f"• {self.docstring_info.returns}", classes="help-returns")
            
            # Examples section
            if self.docstring_info.examples:
                yield Static("[bold]Examples:[/bold]", classes="help-section-header")
                yield Markdown(f"```python\n{self.docstring_info.examples}\n```", classes="help-examples")
            
            # Close button
            yield Button("Close", id="close", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close":
            self.close_window()


class ParameterHelpWindow(BaseOpenHCSWindow):
    """Window for displaying individual parameter documentation."""

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

    def compose(self) -> ComposeResult:
        """Compose the parameter help content."""
        with Vertical():
            # Parameter name and type
            type_info = f" ({self.param_type.__name__})" if self.param_type else ""
            yield Static(f"[bold]{self.param_name}[/bold]{type_info}", classes="param-header")
            
            # Parameter description
            if self.param_description:
                yield Markdown(self.param_description, classes="param-description")
            else:
                yield Static("[dim]No description available[/dim]", classes="param-no-desc")
            
            # Close button
            yield Button("Close", id="close", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close":
            self.close_window()


class HelpableWidget:
    """Mixin class to add help functionality to widgets using windows."""
    
    async def show_function_help(self, target: Union[Callable, type]) -> None:
        """Show help window for a function or class."""
        if not hasattr(self, 'app'):
            return
        
        from textual.css.query import NoMatches
        
        # Use window-based help (follows ConfigWindow pattern)
        try:
            window = self.app.query_one(DocstringHelpWindow)
            # Window exists, update it and open
            window.target = target
            window.docstring_info = DocstringExtractor.extract(target)
            window.open_state = True
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = DocstringHelpWindow(target)
            await self.app.mount(window)
            window.open_state = True
    
    async def show_parameter_help(self, param_name: str, param_description: str, param_type: type = None) -> None:
        """Show help window for a parameter."""
        if not hasattr(self, 'app'):
            return
        
        from textual.css.query import NoMatches
        
        # Use window-based help (follows ConfigWindow pattern)
        try:
            window = self.app.query_one(ParameterHelpWindow)
            # Window exists, update it and open
            window.param_name = param_name
            window.param_description = param_description
            window.param_type = param_type
            window.open_state = True
        except NoMatches:
            # Expected case: window doesn't exist yet, create new one
            window = ParameterHelpWindow(param_name, param_description, param_type)
            await self.app.mount(window)
            window.open_state = True
