#!/usr/bin/env python3
"""
Test script to verify Alt+V paste functionality works in terminal.
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.containers import Vertical

# Import terminal widget directly
try:
    from textual_terminal import Terminal
    TERMINAL_AVAILABLE = True
except ImportError:
    TERMINAL_AVAILABLE = False
    from textual.widgets import Static as Terminal


class AltVPasteTestApp(App):
    """Test app for Alt+V paste functionality."""
    
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("q", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Vertical():
            yield Static("Alt+V Paste Test")
            yield Static("1. Copy some text from a GUI app (Ctrl+C)")
            yield Static("2. Focus the terminal below")
            yield Static("3. Try pasting with Alt+V")
            yield Static("4. The paste should work now!")
            
            if TERMINAL_AVAILABLE:
                yield Terminal(id="test_terminal")
            else:
                yield Static("Terminal not available - install textual-terminal")
        
        yield Footer()
    
    def on_mount(self):
        """Focus the terminal for testing."""
        if TERMINAL_AVAILABLE:
            terminal = self.query_one("#test_terminal", Terminal)
            terminal.focus()


if __name__ == "__main__":
    app = AltVPasteTestApp()
    app.run()
