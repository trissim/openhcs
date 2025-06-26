"""Terminal window for OpenHCS Textual TUI."""

from textual.app import ComposeResult
from textual.widgets import Button, Static
from textual.containers import Container, Horizontal

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow

# Fix textual-terminal compatibility with Textual 3.5.0+
import textual.app
from textual.color import ANSI_COLORS
if not hasattr(textual.app, 'DEFAULT_COLORS'):
    textual.app.DEFAULT_COLORS = ANSI_COLORS

# Import textual-terminal with compatibility fix
try:
    from textual_terminal import Terminal
    TERMINAL_AVAILABLE = True
except ImportError:
    TERMINAL_AVAILABLE = False
    # Create a placeholder Terminal class
    class Terminal(Static):
        def __init__(self, command=None, **kwargs):
            super().__init__("Terminal not available\n\nInstall textual-terminal:\npip install textual-terminal", **kwargs)
            self.command = command

        def clear(self):
            pass

        def write(self, text):
            pass


class TerminalWindow(BaseOpenHCSWindow):
    """Terminal window using textual-window system with embedded terminal."""

    DEFAULT_CSS = """
    TerminalWindow {
        width: 80; height: 24;
        min-width: 60; min-height: 20;
    }
    """

    def __init__(self, shell_command: str = None, **kwargs):
        """
        Initialize terminal window.
        
        Args:
            shell_command: Optional command to run (defaults to system shell)
        """
        super().__init__(
            window_id="terminal",
            title="Terminal",
            mode="temporary",
            **kwargs
        )
        self.shell_command = shell_command or "/bin/bash"

    def compose(self) -> ComposeResult:
        """Compose the terminal window content."""
        with Container():
            if TERMINAL_AVAILABLE:
                yield Terminal(
                    command=self.shell_command,
                    id="terminal"
                )
            else:
                yield Terminal(id="terminal")
        
        with Horizontal(classes="dialog-buttons"):
            yield Button("Clear", id="clear", compact=True)
            yield Button("Close", id="close", compact=True)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "clear":
            if TERMINAL_AVAILABLE:
                await self.send_command("clear")
        elif event.button.id == "close":
            self.close_window()

    async def send_command(self, command: str):
        """Send a command to the terminal."""
        terminal = self.query_one("#terminal", Terminal)
        if TERMINAL_AVAILABLE and hasattr(terminal, 'send_queue') and terminal.send_queue:
            # Send each character of the command
            for char in command:
                await terminal.send_queue.put(["stdin", char])
            # Send enter key
            await terminal.send_queue.put(["stdin", "\n"])

    def on_mount(self) -> None:
        """Called when terminal window is mounted."""
        terminal = self.query_one("#terminal", Terminal)
        if TERMINAL_AVAILABLE:
            terminal.start()  # Start the terminal emulator
        terminal.focus()