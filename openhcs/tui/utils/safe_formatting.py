"""
Safe text formatting utilities for TUI components.

This module provides utilities to safely format text for prompt_toolkit widgets
without triggering string formatting errors when the content contains curly braces.
"""

from typing import Any, Callable, Union, Optional
from prompt_toolkit.widgets import Label
from prompt_toolkit.formatted_text import FormattedText, AnyFormattedText
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window, Dimension
from prompt_toolkit.mouse_events import MouseEventType


def safe_format(template: str, **kwargs) -> str:
    """
    Safely format a template string with values that may contain curly braces.

    This function escapes any curly braces in the values before formatting,
    preventing KeyError exceptions when the values contain unmatched braces.

    Args:
        template: Template string with format placeholders
        **kwargs: Values to substitute into the template

    Returns:
        Safely formatted string

    Example:
        >>> safe_format("Error: {message}", message="Invalid {config}")
        "Error: Invalid {{config}}"
    """
    escaped_kwargs = {}
    for key, value in kwargs.items():
        if value is None:
            escaped_kwargs[key] = "None"
        else:
            # Convert to string and escape curly braces
            str_value = str(value)
            escaped_kwargs[key] = str_value.replace('{', '{{').replace('}', '}}')

    return template.format(**escaped_kwargs)


def safe_text(content: Any) -> str:
    """
    Convert any content to a safe string representation.

    Args:
        content: Any object to convert to string

    Returns:
        String with escaped curly braces and other problematic characters
    """
    if content is None:
        return "None"

    str_content = str(content)
    # Escape curly braces and other problematic format characters
    safe_content = (str_content
                   .replace('{', '{{')
                   .replace('}', '}}')
                   .replace('%', '%%')  # Escape percent signs
                   .replace(':', ' ')   # Replace colons that might cause format issues
                   )
    return safe_content


class SafeLabel(Window):
    """
    A completely safe text widget that bypasses prompt_toolkit's Label formatting.

    This widget uses FormattedTextControl directly to avoid any string formatting
    issues that can occur with Label widgets.
    """

    def __init__(self, text: Union[str, Callable[[], str], AnyFormattedText] = "", style: str = "", **kwargs):
        """
        Initialize SafeLabel with completely safe text handling.

        Args:
            text: Text content (string, callable, or FormattedText)
            style: Style string for the text
            **kwargs: Additional arguments passed to Window
        """
        # Store the original text
        self._original_text = text
        self._style = style

        # Create a safe text function
        if callable(text):
            safe_text_func = lambda: self._make_safe_formatted_text(text())
        elif isinstance(text, str):
            safe_text_func = lambda: self._make_safe_formatted_text(text)
        else:
            safe_text_func = lambda: self._make_safe_formatted_text(str(text))

        # Create a FormattedTextControl that bypasses Label's formatting
        control = FormattedTextControl(
            text=safe_text_func,
            show_cursor=False
        )

        super().__init__(content=control, **kwargs)

    def _make_safe_formatted_text(self, content: Any) -> FormattedText:
        """Make text completely safe for display as FormattedText."""
        if content is None:
            safe_content = "None"
        else:
            str_content = str(content)
            # Keep only safe characters - be very conservative
            safe_content = ""
            for char in str_content:
                if char.isalnum() or char in ' .-_/\\':
                    safe_content += char
                else:
                    safe_content += " "  # Replace problematic chars with space

        # Return as FormattedText with style
        if self._style:
            return FormattedText([(self._style, safe_content)])
        else:
            return FormattedText([('', safe_content)])


def safe_error_label(message: str, details: str = "") -> SafeLabel:
    """
    Create a SafeLabel for displaying error messages.

    Args:
        message: Main error message
        details: Optional error details

    Returns:
        SafeLabel configured for error display
    """
    if details:
        text = safe_format("Error: {message} - {details}", message=message, details=details)
    else:
        text = safe_format("Error: {message}", message=message)

    return SafeLabel(text, style="class:error-text")


def safe_info_label(message: str, **kwargs) -> SafeLabel:
    """
    Create a SafeLabel for displaying informational messages.

    Args:
        message: Information message
        **kwargs: Additional format values

    Returns:
        SafeLabel configured for info display
    """
    if kwargs:
        text = safe_format(message, **kwargs)
    else:
        text = safe_text(message)

    return SafeLabel(text, style="class:info-text")


def safe_status_label(status: str, item_name: str = "") -> SafeLabel:
    """
    Create a SafeLabel for displaying status messages.

    Args:
        status: Status text
        item_name: Optional item name

    Returns:
        SafeLabel configured for status display
    """
    if item_name:
        text = safe_format("{item_name}: {status}", item_name=item_name, status=status)
    else:
        text = safe_text(status)

    return SafeLabel(text, style="class:status-text")


# Convenience functions for common patterns
def error_building_component(component_name: str, exception: Exception) -> SafeLabel:
    """Create error label for component building failures."""
    return safe_error_label(
        f"Error building {component_name}",
        str(exception)
    )


def unsupported_type_label(value: Any, type_info: Any) -> SafeLabel:
    """Create label for unsupported type display."""
    return SafeLabel(safe_format(
        "{value} (Unsupported type: {type_info})",
        value=value,
        type_info=type_info
    ))


def field_label(field_name: str, required: bool = False, special: bool = False) -> SafeLabel:
    """Create label for form fields."""
    safe_name = safe_text(field_name)
    label_text = f"{safe_name}{'*' if required else ''}"
    if special:
        label_text = f"{label_text} [S]"
    label_text += ": "

    return SafeLabel(label_text)


class SafeButton(Window):
    """
    A completely safe button widget that bypasses prompt_toolkit's Button formatting.

    This widget uses FormattedTextControl directly to avoid string formatting
    issues that can occur with Button widgets when text contains problematic characters.
    """

    def __init__(self, text: str, handler: Optional[Callable[[], None]] = None,
                 width: Optional[int] = None, style: str = "class:button", **kwargs):
        """
        Initialize SafeButton with completely safe text handling.

        Args:
            text: Button text (will be made safe)
            handler: Click handler function
            width: Button width (None for auto-width)
            style: Style string for the button
            **kwargs: Additional arguments passed to Window
        """
        self.original_text = text
        self.handler = handler
        self._style = style

        # Make text completely safe
        safe_text_content = safe_text(text)

        # Create button-like display with brackets
        button_text = f"[{safe_text_content}]"

        # Create a FormattedTextControl that bypasses Button's formatting
        control = FormattedTextControl(
            text=lambda: FormattedText([(self._style, button_text)]),
            show_cursor=False
        )

        # Set up mouse handler for clicks
        def mouse_handler(mouse_event):
            if (mouse_event.event_type == MouseEventType.MOUSE_UP and
                self.handler is not None):
                self.handler()
                return True
            return False

        # Store the handler for the window
        self._mouse_handler = mouse_handler

        # Set width if specified
        if width is not None:
            kwargs['width'] = Dimension.exact(width)

        super().__init__(content=control, **kwargs)

    def mouse_handler(self, mouse_event):
        """Handle mouse events for button clicks."""
        return self._mouse_handler(mouse_event)


def safe_button(text: str, handler: Optional[Callable[[], None]] = None,
                width: Optional[int] = None, style: str = "class:button") -> SafeButton:
    """
    Create a SafeButton that won't cause formatting errors.

    Args:
        text: Button text
        handler: Click handler function
        width: Button width (None for auto-width)
        style: Style string for the button

    Returns:
        SafeButton that handles problematic text safely
    """
    return SafeButton(text, handler, width, style)
