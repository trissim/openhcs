"""
Advanced Terminal Window for OpenHCS Textual TUI

Uses the Gate One terminal emulator for advanced terminal functionality.
This supersedes the basic terminal window with full VT-* emulation support.
"""

import asyncio
import logging
import os
import pty
import subprocess
import threading
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow

# Import the Gate One terminal emulator and callback constants
from openhcs.textual_tui.services.terminal import (
    Terminal as GateOneTerminal,
    CALLBACK_CHANGED,
    CALLBACK_CURSOR_POS,
    CALLBACK_TITLE,
    CALLBACK_BELL
)

logger = logging.getLogger(__name__)


class AdvancedTerminalWidget(Widget):
    """
    Advanced terminal widget using Gate One terminal emulator.
    
    Provides full VT-* terminal emulation with:
    - Complete ECMA-48/ANSI X3.64 support
    - VT-52, VT-100, VT-220, VT-320, VT-420, VT-520 emulation
    - Linux console emulation
    - Image support (with PIL)
    - Scrollback buffer
    - HTML output capabilities
    """

    DEFAULT_CSS = """
    AdvancedTerminalWidget {
        width: 100%;
        height: 100%;
    }
    """

    # Reactive properties
    cursor_visible = reactive(True)
    terminal_title = reactive("Advanced Terminal")
    
    def __init__(self, command: Optional[str] = None, rows: int = 24, cols: int = 80, **kwargs):
        """
        Initialize the advanced terminal widget.
        
        Args:
            command: Shell command to run (defaults to current shell)
            rows: Terminal height in rows
            cols: Terminal width in columns
        """
        super().__init__(**kwargs)
        
        self.command = command or self._get_default_shell()
        self.rows = rows
        self.cols = cols
        
        # Initialize Gate One terminal
        self.terminal = GateOneTerminal(rows=rows, cols=cols)
        
        # Setup terminal callbacks
        self._setup_terminal_callbacks()
        
        # Process management
        self.process = None
        self.master_fd = None
        self.reader_thread = None
        self.running = False
        
        logger.info(f"AdvancedTerminalWidget initialized: {rows}x{cols}, command: {self.command}")

    def _get_default_shell(self) -> str:
        """Get the default shell for the current user."""
        # Try to get user's shell from environment or passwd
        shell = os.environ.get('SHELL')
        if shell and os.path.exists(shell):
            return shell
        
        # Fallback to common shells
        for shell_path in ['/bin/bash', '/bin/zsh', '/bin/sh']:
            if os.path.exists(shell_path):
                return shell_path
        
        return '/bin/sh'  # Ultimate fallback

    def _setup_terminal_callbacks(self):
        """Setup callbacks for the Gate One terminal."""
        # Callback when screen changes - trigger refresh
        def on_screen_changed():
            self.call_after_refresh(self._update_display)
        
        # Callback when cursor position changes
        def on_cursor_changed():
            self.call_after_refresh(self._update_cursor)
        
        # Callback when title changes
        def on_title_changed():
            title = getattr(self.terminal, 'title', 'Advanced Terminal')
            self.terminal_title = title
        
        # Callback for bell
        def on_bell():
            # Could implement visual bell here
            logger.debug("Terminal bell")
        
        # Set up the callbacks using the proper add_callback method
        self.terminal.add_callback(CALLBACK_CHANGED, on_screen_changed, identifier="widget_changed")
        self.terminal.add_callback(CALLBACK_CURSOR_POS, on_cursor_changed, identifier="widget_cursor")
        self.terminal.add_callback(CALLBACK_TITLE, on_title_changed, identifier="widget_title")
        self.terminal.add_callback(CALLBACK_BELL, on_bell, identifier="widget_bell")

    def compose(self) -> ComposeResult:
        """Compose the terminal widget layout."""
        # Just the terminal screen display, no control buttons
        yield Static("", id="terminal_screen")

    def on_mount(self) -> None:
        """Start the terminal when widget is mounted."""
        logger.info("AdvancedTerminalWidget mounting - starting terminal process")
        self.start_terminal()

    def on_unmount(self) -> None:
        """Clean up when widget is unmounted."""
        logger.info("AdvancedTerminalWidget unmounting - stopping terminal process")
        self.stop_terminal()

    def start_terminal(self):
        """Start the terminal process and reader thread."""
        try:
            # Create a pseudo-terminal
            self.master_fd, slave_fd = pty.openpty()
            
            # Start the shell process
            self.process = subprocess.Popen(
                self.command,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                shell=True,
                preexec_fn=os.setsid  # Create new session
            )
            
            # Close slave fd in parent process
            os.close(slave_fd)
            
            # Start reader thread
            self.running = True
            self.reader_thread = threading.Thread(target=self._read_from_terminal, daemon=True)
            self.reader_thread.start()
            
            logger.info(f"Terminal process started: PID {self.process.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start terminal process: {e}")
            self._show_error(f"Failed to start terminal: {e}")

    def stop_terminal(self):
        """Stop the terminal process and cleanup."""
        self.running = False
        
        if self.process:
            try:
                # Terminate the process group
                os.killpg(os.getpgid(self.process.pid), 15)  # SIGTERM
                self.process.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    # Force kill if needed
                    os.killpg(os.getpgid(self.process.pid), 9)  # SIGKILL
                except ProcessLookupError:
                    pass
            self.process = None
        
        if self.master_fd:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None
        
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1)

    def _read_from_terminal(self):
        """Read output from terminal in background thread."""
        while self.running and self.master_fd:
            try:
                # Read from terminal with timeout
                import select
                ready, _, _ = select.select([self.master_fd], [], [], 0.1)
                
                if ready:
                    data = os.read(self.master_fd, 1024)
                    if data:
                        # Decode and write to Gate One terminal
                        text = data.decode('utf-8', errors='replace')
                        self.terminal.write(text)
                    else:
                        # EOF - process ended
                        break
                        
            except (OSError, ValueError) as e:
                if self.running:
                    logger.error(f"Error reading from terminal: {e}")
                break
        
        logger.info("Terminal reader thread stopped")

    def _update_display(self):
        """Update the terminal display with current screen content."""
        try:
            # Get screen content from Gate One terminal
            screen_lines = self.terminal.dump()
            
            # Join lines and update display
            screen_content = '\n'.join(screen_lines)
            
            # Update the display widget
            screen_widget = self.query_one("#terminal_screen", Static)
            screen_widget.update(screen_content)
            
        except Exception as e:
            logger.error(f"Error updating terminal display: {e}")

    def _update_cursor(self):
        """Update cursor position display."""
        # Could implement cursor position indicator here
        pass

    def _show_error(self, message: str):
        """Show error message in terminal display."""
        try:
            screen_widget = self.query_one("#terminal_screen", Static)
            screen_widget.update(f"[red]Error: {message}[/red]")
        except Exception:
            pass



    async def clear_terminal(self):
        """Clear the terminal screen."""
        if self.master_fd:
            try:
                # Send clear command
                os.write(self.master_fd, b'\x0c')  # Form feed (clear)
            except OSError as e:
                logger.error(f"Error clearing terminal: {e}")

    async def reset_terminal(self):
        """Reset the terminal."""
        if self.master_fd:
            try:
                # Send reset sequence
                os.write(self.master_fd, b'\x1bc')  # ESC c (reset)
            except OSError as e:
                logger.error(f"Error resetting terminal: {e}")

    async def send_input(self, text: str):
        """Send input to the terminal."""
        if self.master_fd:
            try:
                os.write(self.master_fd, text.encode('utf-8'))
            except OSError as e:
                logger.error(f"Error sending input to terminal: {e}")

    async def on_key(self, event) -> None:
        """Handle key input and send to terminal."""
        if self.master_fd:
            try:
                # Convert key to appropriate bytes
                key_bytes = self._key_to_bytes(event.key)
                if key_bytes:
                    os.write(self.master_fd, key_bytes)
            except OSError as e:
                logger.error(f"Error sending key to terminal: {e}")

    def _key_to_bytes(self, key: str) -> bytes:
        """Convert Textual key to terminal bytes."""
        # Handle special keys
        key_map = {
            'enter': b'\r',
            'escape': b'\x1b',
            'backspace': b'\x7f',
            'tab': b'\t',
            'up': b'\x1b[A',
            'down': b'\x1b[B',
            'right': b'\x1b[C',
            'left': b'\x1b[D',
            'home': b'\x1b[H',
            'end': b'\x1b[F',
            'page_up': b'\x1b[5~',
            'page_down': b'\x1b[6~',
            'delete': b'\x1b[3~',
            'insert': b'\x1b[2~',
        }
        
        # Handle ctrl combinations
        if key.startswith('ctrl+'):
            char = key[5:]
            if len(char) == 1 and 'a' <= char <= 'z':
                return bytes([ord(char) - ord('a') + 1])
        
        # Handle regular keys
        if key in key_map:
            return key_map[key]
        elif len(key) == 1:
            return key.encode('utf-8')
        
        return b''


class AdvancedTerminalWindow(BaseOpenHCSWindow):
    """
    Advanced Terminal Window using Gate One terminal emulator.
    
    This supersedes the basic TerminalWindow with advanced features:
    - Full VT-* terminal emulation
    - Complete ANSI escape sequence support
    - Image display capabilities
    - Advanced scrollback buffer
    - HTML output support
    """

    DEFAULT_CSS = """
    AdvancedTerminalWindow {
        width: 100; height: 30;
        min-width: 80; min-height: 24;
    }
    """

    def __init__(self, shell_command: str = None, **kwargs):
        """
        Initialize advanced terminal window.

        Args:
            shell_command: Optional command to run (defaults to current shell)
        """
        # Get shell before calling super() so we can use it in title
        self.shell_command = shell_command or self._get_current_shell()

        # Extract shell name for title
        shell_name = os.path.basename(self.shell_command)

        logger.info(f"AdvancedTerminal: Initializing with shell: {self.shell_command}")

        super().__init__(
            window_id="advanced_terminal",
            title=f"Advanced Terminal ({shell_name})",
            mode="temporary",
            **kwargs
        )

    def _get_current_shell(self) -> str:
        """Get the current shell from environment."""
        return os.environ.get('SHELL', '/bin/bash')

    def compose(self) -> ComposeResult:
        """Compose the advanced terminal window content."""
        yield AdvancedTerminalWidget(
            command=self.shell_command,
            rows=24,
            cols=80,
            id="advanced_terminal"
        )

    async def on_key(self, event) -> None:
        """Handle key presses for terminal shortcuts."""
        # Ctrl+Shift+C to copy (if we implement clipboard)
        if event.key == "ctrl+shift+c":
            # Could implement copy functionality here
            pass
        # Ctrl+Shift+V to paste (if we implement clipboard)
        elif event.key == "ctrl+shift+v":
            # Could implement paste functionality here
            pass
        # Ctrl+Shift+T to open new terminal
        elif event.key == "ctrl+shift+t":
            # Could open new terminal window
            pass
        else:
            # Forward all other keys to the terminal widget
            terminal = self.query_one("#advanced_terminal", AdvancedTerminalWidget)
            await terminal.on_key(event)

    async def send_command(self, command: str):
        """Send a command to the terminal."""
        terminal = self.query_one("#advanced_terminal", AdvancedTerminalWidget)
        await terminal.send_input(command + '\n')

    def on_mount(self) -> None:
        """Called when advanced terminal window is mounted."""
        terminal = self.query_one("#advanced_terminal", AdvancedTerminalWidget)
        terminal.focus()
        logger.info("AdvancedTerminalWindow mounted and focused")
