#!/usr/bin/env python3
"""
Test script to verify terminal paste functionality works.
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.containers import Vertical

# Import terminal window
from openhcs.textual_tui.windows.terminal_window import TerminalWindow


class TerminalPasteTestApp(App):
    """Test app for terminal paste functionality."""
    
    # No global Ctrl+C binding to avoid conflicts
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("q", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Vertical():
            yield Static("Terminal Paste Test")
            yield Static("1. Copy some text from outside the terminal")
            yield Static("2. Try pasting with Alt+C (your terminal binding)")
            yield Static("3. Paste should work in the terminal below")
        
        yield Footer()
    
    def on_mount(self):
        """Open terminal window for testing."""
        # Open a terminal window to test paste functionality
        terminal_window = TerminalWindow(
            window_id="test_terminal",
            title="Test Terminal - Try Alt+C paste"
        )
        terminal_window.open_state = True


if __name__ == "__main__":
    app = TerminalPasteTestApp()
    app.run()
