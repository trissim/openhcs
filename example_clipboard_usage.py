#!/usr/bin/env python3
"""
Example of how to use the universal clipboard functionality in OpenHCS TUI.

This demonstrates three ways to add clipboard support to widgets:
1. Using the utility functions directly
2. Using the decorator for automatic binding
3. Manual integration for custom widgets
"""

from textual.app import App, ComposeResult
from textual.widgets import TextArea, Header, Footer, Static
from textual.containers import Vertical
from textual.binding import Binding

# Import the universal clipboard utilities
from openhcs.textual_tui.utils.clipboard_utils import (
    copy_text_with_notification,
    copy_selected_text,
    add_copy_binding_to_widget
)


# Method 1: Using the decorator (simplest) - uses Ctrl+Shift+C by default to avoid conflicts
@add_copy_binding_to_widget
class AutoClipboardTextArea(TextArea):
    """TextArea with automatic Ctrl+Shift+C copy binding."""
    pass


# Method 2: Manual integration (most flexible)
class ManualClipboardTextArea(TextArea):
    """TextArea with manual clipboard integration."""
    
    BINDINGS = [
        Binding("ctrl+c", "copy_selection", "Copy selection"),
        Binding("ctrl+shift+c", "copy_all", "Copy all"),
    ]
    
    def action_copy_selection(self):
        """Copy selected text to clipboard."""
        copy_selected_text(self)
    
    def action_copy_all(self):
        """Copy all text to clipboard."""
        copy_text_with_notification(self, self.text, "content")


# Method 3: Custom widget with clipboard support
class ClipboardDemo(Static):
    """Custom widget demonstrating clipboard functionality."""
    
    BINDINGS = [
        Binding("c", "copy_demo_text", "Copy demo text"),
    ]
    
    def action_copy_demo_text(self):
        """Copy some demo text to clipboard."""
        demo_text = "Hello from OpenHCS TUI clipboard system!"
        copy_text_with_notification(self, demo_text, "demo text")


class ClipboardExampleApp(App):
    """Example app showing clipboard functionality."""
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Vertical():
            yield Static("OpenHCS Universal Clipboard Demo", classes="title")
            yield Static("1. Auto-binding TextArea (Ctrl+Shift+C to copy selection):")
            yield AutoClipboardTextArea("Select this text and press Ctrl+Shift+C", id="auto_area")
            
            yield Static("2. Manual integration TextArea (Ctrl+C for selection, Ctrl+Shift+C for all):")
            yield ManualClipboardTextArea("This area has custom copy bindings", id="manual_area")
            
            yield Static("3. Custom widget (Press 'c' to copy demo text):")
            yield ClipboardDemo("Press 'c' to copy demo text", id="demo_widget")
        
        yield Footer()


if __name__ == "__main__":
    app = ClipboardExampleApp()
    app.run()
