"""Unified clickable help components using consolidated help system."""

from typing import Union, Callable, Optional
from textual.widgets import Static
from textual.events import Click
from textual.message import Message


class ClickableHelpLabel(Static):
    """A clickable label that shows help information when clicked."""

    def __init__(self, text: str, help_target: Union[Callable, type] = None, 
                 param_name: str = None, param_description: str = None, 
                 param_type: type = None, **kwargs):
        """Initialize clickable help label.
        
        Args:
            text: Display text for the label
            help_target: Function or class to show help for (for function help)
            param_name: Parameter name (for parameter help)
            param_description: Parameter description (for parameter help)
            param_type: Parameter type (for parameter help)
        """
        # Add help indicator to text
        display_text = f"{text} [dim](?)[/dim]"
        super().__init__(display_text, **kwargs)
        
        self.help_target = help_target
        self.param_name = param_name
        self.param_description = param_description
        self.param_type = param_type
        
        # Add CSS classes for styling
        self.add_class("clickable-help")
        
    async def on_click(self, event: Click) -> None:
        """Handle click events to show help window using unified manager."""
        event.stop()  # Prevent event bubbling

        from openhcs.textual_tui.windows.help_windows import HelpWindowManager

        if self.help_target:
            # Show function/class help using unified manager
            await HelpWindowManager.show_docstring_help(self.app, self.help_target)
        elif self.param_name and self.param_description:
            # Show parameter help using unified manager
            await HelpWindowManager.show_parameter_help(
                self.app, self.param_name, self.param_description, self.param_type
            )


class ClickableFunctionTitle(ClickableHelpLabel):
    """Clickable function title that shows function documentation."""
    
    def __init__(self, func: Callable, index: int = None, **kwargs):
        func_name = getattr(func, '__name__', 'Unknown Function')
        module_name = getattr(func, '__module__', '').split('.')[-1] if func else ''
        
        # Build title text
        title = f"{index + 1}: {func_name}" if index is not None else func_name
        if module_name:
            title += f" ({module_name})"
            
        super().__init__(
            text=f"[bold]{title}[/bold]",
            help_target=func,
            **kwargs
        )


class ClickableParameterLabel(ClickableHelpLabel):
    """Clickable parameter label that shows parameter documentation."""
    
    def __init__(self, param_name: str, param_description: str = None, 
                 param_type: type = None, **kwargs):
        # Format parameter name nicely
        display_name = param_name.replace('_', ' ').title()
        
        super().__init__(
            text=display_name,
            param_name=param_name,
            param_description=param_description or "No description available",
            param_type=param_type,
            **kwargs
        )


class HelpIndicator(Static):
    """Simple help indicator that can be added next to any widget."""
    
    def __init__(self, help_target: Union[Callable, type] = None,
                 param_name: str = None, param_description: str = None,
                 param_type: type = None, **kwargs):
        super().__init__("[dim](?)[/dim]", **kwargs)
        
        self.help_target = help_target
        self.param_name = param_name
        self.param_description = param_description
        self.param_type = param_type
        
        self.add_class("help-indicator")
        
    async def on_click(self, event: Click) -> None:
        """Handle click events to show help window using unified manager."""
        event.stop()

        from openhcs.textual_tui.windows.help_windows import HelpWindowManager

        if self.help_target:
            # Show function/class help using unified manager
            await HelpWindowManager.show_docstring_help(self.app, self.help_target)
        elif self.param_name and self.param_description:
            # Show parameter help using unified manager
            await HelpWindowManager.show_parameter_help(
                self.app, self.param_name, self.param_description, self.param_type
            )
