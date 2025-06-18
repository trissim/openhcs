"""Docstring help dialog for displaying function and parameter documentation."""

from typing import Union, Callable, Optional
from textual.app import ComposeResult
from textual.widgets import Static, Button, Markdown
from textual.containers import Vertical, Horizontal

from openhcs.textual_tui.widgets.floating_window import BaseFloatingWindow
from .signature_analyzer import DocstringExtractor, DocstringInfo


class DocstringHelpDialog(BaseFloatingWindow):
    """Help dialog for displaying docstring information."""

    def __init__(self, target: Union[Callable, type], title: Optional[str] = None, **kwargs):
        self.target = target
        self.docstring_info = DocstringExtractor.extract(target)
        
        # Generate title
        if title:
            dialog_title = title
        else:
            target_name = getattr(target, '__name__', 'Unknown')
            dialog_title = f"Help: {target_name}"
        
        super().__init__(title=dialog_title, **kwargs)

    def compose_content(self) -> ComposeResult:
        """Compose the help dialog content."""
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

    def compose_buttons(self) -> ComposeResult:
        """Provide Close button."""
        yield Button("Close", id="close", compact=True)

    def handle_button_action(self, button_id: str, button_text: str):
        """Handle button actions - Close button dismisses dialog."""
        return None  # Dismiss with None result


class ParameterHelpDialog(BaseFloatingWindow):
    """Help dialog for displaying individual parameter documentation."""

    def __init__(self, param_name: str, param_description: str, param_type: type = None, **kwargs):
        self.param_name = param_name
        self.param_description = param_description
        self.param_type = param_type
        
        super().__init__(title=f"Parameter: {param_name}", **kwargs)

    def compose_content(self) -> ComposeResult:
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

    def compose_buttons(self) -> ComposeResult:
        """Provide Close button."""
        yield Button("Close", id="close", compact=True)

    def handle_button_action(self, button_id: str, button_text: str):
        """Handle button actions - Close button dismisses dialog."""
        return None  # Dismiss with None result


class HelpableWidget:
    """Mixin class to add help functionality to widgets."""
    
    def show_function_help(self, target: Union[Callable, type]) -> None:
        """Show help dialog for a function or class."""
        if not hasattr(self, 'app'):
            return
            
        help_dialog = DocstringHelpDialog(target)
        self.app.push_screen(help_dialog)
    
    def show_parameter_help(self, param_name: str, param_description: str, param_type: type = None) -> None:
        """Show help dialog for a parameter."""
        if not hasattr(self, 'app'):
            return
            
        help_dialog = ParameterHelpDialog(param_name, param_description, param_type)
        self.app.push_screen(help_dialog)
