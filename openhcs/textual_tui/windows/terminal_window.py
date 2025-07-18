"""Terminal window for OpenHCS Textual TUI."""

from pathlib import Path
from textual.app import ComposeResult
from textual.widgets import Button, Static
from textual.containers import Container, Horizontal, Vertical

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow

# Fix textual-terminal compatibility with Textual 3.5.0+
import textual.app
from textual.color import ANSI_COLORS
if not hasattr(textual.app, 'DEFAULT_COLORS'):
    textual.app.DEFAULT_COLORS = ANSI_COLORS

# Import textual-terminal with compatibility fix
try:
    from textual_terminal import Terminal
    from textual_terminal._terminal import TerminalEmulator
    TERMINAL_AVAILABLE = True

    # Import our extracted terminal enhancements
    from openhcs.textual_tui.services.terminal_enhancements import terminal_enhancements

    # Import Gate One terminal for enhanced features
    from openhcs.textual_tui.services.terminal import Terminal as GateOneTerminal

    # Monkey-patch Terminal to track cursor styles
    _original_terminal_recv = Terminal.recv

    async def _patched_recv(self):
        """Patched recv that tracks cursor style and renders appropriately."""
        import re
        import asyncio

        # Initialize cursor_style if not present
        if not hasattr(self, 'cursor_style'):
            self.cursor_style = 1  # Default block cursor

        try:
            while True:
                message = await self.recv_queue.get()
                cmd = message[0]
                if cmd == "setup":
                    await self.send_queue.put(["set_size", self.nrow, self.ncol])
                elif cmd == "stdout":
                    chars = message[1]

                    # Track cursor style sequences - comprehensive detection

                    # Debug: Look for cursor sequences specifically
                    if ' q' in chars or 'q' in chars:
                        print(f"DEBUG: Raw chars containing 'q': {repr(chars)}")

                    # Standard DECSCUSR sequences: ESC [ Ps q (vim style)
                    cursor_style_pattern = re.compile(r'\x1b\[([0-6]) q')
                    for match in cursor_style_pattern.finditer(chars):
                        style_num = int(match.group(1))
                        old_style = self.cursor_style
                        if style_num in [0, 1, 2]:  # 0=default, 1=blinking block, 2=steady block
                            self.cursor_style = 1  # block
                        elif style_num in [3, 4]:  # 3=blinking underline, 4=steady underline
                            self.cursor_style = 3  # underline
                        elif style_num in [5, 6]:  # 5=blinking bar, 6=steady bar
                            self.cursor_style = 5  # bar
                        print(f"DEBUG: VIM cursor style changed from {old_style} to {self.cursor_style} (sequence: {repr(match.group(0))})")

                    # Linux terminal sequences: ESC [ ? Ps c
                    linux_cursor_pattern = re.compile(r'\x1b\[\?([0-9]+)c')
                    for match in linux_cursor_pattern.finditer(chars):
                        style_num = int(match.group(1))
                        old_style = self.cursor_style
                        if style_num == 0:
                            self.cursor_style = 1  # normal block
                        elif style_num == 1:
                            self.cursor_style = 0  # invisible
                        elif style_num == 8:
                            self.cursor_style = 3  # very visible (underline)
                        print(f"DEBUG: Linux cursor style changed from {old_style} to {self.cursor_style} (sequence: {match.group(0)})")

                    # Vim-style cursor sequences: ESC ] 50 ; CursorShape=N BEL
                    vim_cursor_pattern = re.compile(r'\x1b\]50;CursorShape=([0-2])\x07')
                    for match in vim_cursor_pattern.finditer(chars):
                        shape_num = int(match.group(1))
                        old_style = self.cursor_style
                        if shape_num == 0:
                            self.cursor_style = 1  # block
                        elif shape_num == 1:
                            self.cursor_style = 5  # bar
                        elif shape_num == 2:
                            self.cursor_style = 3  # underline
                        print(f"DEBUG: Vim cursor style changed from {old_style} to {self.cursor_style} (sequence: {match.group(0)})")

                    # Alternative vim sequences: ESC ] 12 ; color BEL (cursor color)
                    # We don't change style but log it
                    vim_color_pattern = re.compile(r'\x1b\]12;[^\\x07]*\x07')
                    for match in vim_color_pattern.finditer(chars):
                        print(f"DEBUG: Vim cursor color sequence detected: {match.group(0)}")

                    # Cursor visibility sequences
                    cursor_visibility_pattern = re.compile(r'\x1b\[\?25([hl])')
                    for match in cursor_visibility_pattern.finditer(chars):
                        visibility = match.group(1)
                        if visibility == 'l':  # hide cursor
                            print(f"DEBUG: Cursor hidden")
                        elif visibility == 'h':  # show cursor
                            print(f"DEBUG: Cursor shown")

                    # Handle mouse tracking (from original)
                    _re_ansi_sequence = re.compile(r"(\x1b\[\??[\d;]*[a-zA-Z])")
                    DECSET_PREFIX = "\x1b[?"

                    for sep_match in re.finditer(_re_ansi_sequence, chars):
                        sequence = sep_match.group(0)
                        if sequence.startswith(DECSET_PREFIX):
                            parameters = sequence.removeprefix(DECSET_PREFIX).split(";")
                            if "1000h" in parameters:
                                self.mouse_tracking = True
                            if "1000l" in parameters:
                                self.mouse_tracking = False

                    try:
                        self.stream.feed(chars)
                    except TypeError as error:
                        from textual import log
                        log.warning("could not feed:", error)

                    # Custom display building with cursor styles
                    from rich.text import Text
                    lines = []
                    for y in range(self._screen.lines):
                        line_text = Text()
                        line = self._screen.buffer[y]
                        style_change_pos: int = 0
                        for x in range(self._screen.columns):
                            char = line[x]

                            # Check if this is the cursor position
                            is_cursor = (
                                self._screen.cursor.x == x
                                and self._screen.cursor.y == y
                                and not self._screen.cursor.hidden
                            )

                            # Modify character for cursor styles
                            char_data = char.data
                            if is_cursor:
                                cursor_style = getattr(self, 'cursor_style', 1)
                                if cursor_style == 5:  # bar cursor
                                    # Use a thin left-aligned vertical bar
                                    char_data = "▎"  # Left one-quarter block (slightly thicker but still thin)
                                elif cursor_style == 3:  # underline cursor
                                    char_data = char.data  # Keep original character

                            line_text.append(char_data)

                            # Handle styling (from original)
                            if x > 0:
                                last_char = line[x - 1]
                                if not self.char_style_cmp(char, last_char) or x == self._screen.columns - 1:
                                    last_style = self.char_rich_style(last_char)
                                    line_text.stylize(last_style, style_change_pos, x + 1)
                                    style_change_pos = x

                            # Apply cursor styling
                            if is_cursor:
                                cursor_style = getattr(self, 'cursor_style', 1)
                                if cursor_style == 1:  # block cursor
                                    line_text.stylize("reverse blink", x, x + 1)
                                elif cursor_style == 3:  # underline cursor
                                    line_text.stylize("underline bold blink", x, x + 1)
                                elif cursor_style == 5:  # bar cursor
                                    # Make the thin bar visible and blinking
                                    line_text.stylize("bold blink", x, x + 1)

                        lines.append(line_text)

                    from textual_terminal._terminal import TerminalDisplay
                    self._display = TerminalDisplay(lines)
                    self.refresh()

                elif cmd == "disconnect":
                    self.stop()
        except asyncio.CancelledError:
            pass

    # Apply the monkey patch
    Terminal.recv = _patched_recv

    # Monkey-patch TerminalEmulator for better environment
    _original_open_terminal = TerminalEmulator.open_terminal

    def _patched_open_terminal(self, command: str):
        """Patched version that uses linux terminal type for proper Unicode box-drawing."""
        import pty
        import shlex
        import os
        from pathlib import Path

        self.pid, fd = pty.fork()
        if self.pid == 0:
            argv = shlex.split(command)
            # Use linux terminal type - this makes programs output Unicode directly
            # instead of VT100 line-drawing sequences that need translation
            env = dict(
                TERM="linux",  # Critical: linux terminal outputs Unicode box-drawing directly
                LC_ALL="en_US.UTF-8",
                LC_CTYPE="en_US.UTF-8",
                LANG="en_US.UTF-8",
                HOME=str(Path.home()),
            )
            # Add current PATH and other important vars
            for key in ['PATH', 'USER', 'SHELL']:
                if key in os.environ:
                    env[key] = os.environ[key]
            os.execvpe(argv[0], argv, env)
        return fd

    # Apply the monkey patch
    TerminalEmulator.open_terminal = _patched_open_terminal

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
        min-width: 80; min-height: 20;
    }
    TerminalWindow #content_pane {
        padding: 0;
    }
    TerminalWindow #terminal {
        height: 1fr;
        width: 100%;
    }
    """

    def __init__(self, shell_command: str = None, **kwargs):
        """
        Initialize terminal window.

        Args:
            shell_command: Optional command to run (defaults to current shell)
        """
        # Get shell before calling super() so we can use it in title
        self.shell_command = shell_command or self._get_current_shell()

        # Extract shell name for title
        import os
        import logging
        shell_name = os.path.basename(self.shell_command)

        logger = logging.getLogger(__name__)
        logger.info(f"Terminal: Initializing with shell: {self.shell_command}")

        super().__init__(
            window_id="terminal",
            title=f"Terminal ({shell_name})",
            mode="temporary",
            **kwargs
        )

        # Track wrapper script for cleanup
        self.wrapper_script_path = None

    def _get_current_shell(self) -> str:
        """Get the current shell command with login flag."""
        import os
        import logging

        logger = logging.getLogger(__name__)

        # Method 1: Check SHELL environment variable
        shell_env = os.environ.get('SHELL')
        if shell_env and os.path.exists(shell_env):
            logger.debug(f"Terminal: Using shell from SHELL env var: {shell_env}")
            return f"{shell_env} -l"  # Login shell

        # Method 2: Check parent process (what launched the TUI)
        try:
            import psutil
            current_process = psutil.Process()
            parent_process = current_process.parent()
            if parent_process and parent_process.name() in ['bash', 'zsh', 'fish', 'tcsh', 'csh', 'sh']:
                # Try to get the full path
                try:
                    shell_path = parent_process.exe()
                    logger.debug(f"Terminal: Using shell from parent process: {shell_path}")
                    return f"{shell_path} -l"  # Login shell
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    # Fall back to name-based lookup
                    shell_name = parent_process.name()
                    shell_path = f"/bin/{shell_name}"
                    logger.debug(f"Terminal: Using shell from parent process name: {shell_path}")
                    return f"{shell_path} -l"  # Login shell
        except (ImportError, psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Method 3: Check what shell the user is actually using
        try:
            # Get the shell from /etc/passwd for current user
            import pwd
            user_shell = pwd.getpwuid(os.getuid()).pw_shell
            if user_shell and os.path.exists(user_shell):
                return f"{user_shell} -l"  # Login shell
        except (ImportError, KeyError, OSError):
            pass

        # Method 4: Try to detect common shells in order of preference
        common_shells = ['/bin/zsh', '/usr/bin/zsh', '/bin/bash', '/usr/bin/bash', '/bin/fish', '/usr/bin/fish', '/bin/sh']
        for shell_path in common_shells:
            if os.path.exists(shell_path):
                return f"{shell_path} -l"  # Login shell

        # Fallback: bash (original behavior)
        logger.debug("Terminal: Using fallback shell: /bin/bash -l")
        return "/bin/bash -l"

    def _create_environment_wrapper(self) -> str:
        """Create a wrapper command that exports current environment variables."""
        import os
        import tempfile
        import stat

        # Export key environment variables that the monkey-patch doesn't handle
        important_vars = ['PATH', 'HOME', 'USER', 'SHELL', 'PWD']
        env_exports = []

        for key in important_vars:
            if key in os.environ:
                escaped_value = os.environ[key].replace('"', '\\"')
                env_exports.append(f'export {key}="{escaped_value}"')

        script_content = f'''#!/bin/bash
# Export important environment variables
{chr(10).join(env_exports)}

# Launch the shell (terminal capabilities handled by monkey-patch)
exec {self.shell_command}
'''

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        # Make executable
        os.chmod(script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

        # Store for cleanup
        self.wrapper_script_path = script_path

        return script_path

    def compose(self) -> ComposeResult:
        """Compose the terminal window content - full window, no buttons."""
        with Vertical():
            if TERMINAL_AVAILABLE:
                # Create wrapper script that exports all environment variables
                wrapper_command = self._create_environment_wrapper()

                yield Terminal(
                    command=wrapper_command,
                    id="terminal"
                )
            else:
                yield Terminal(id="terminal")

    async def on_key(self, event) -> None:
        """Handle key presses for terminal shortcuts."""
        # Ctrl+L to clear terminal (like most terminals)
        if event.key == "ctrl+l":
            if TERMINAL_AVAILABLE:
                await self.send_command("clear")
        # Ctrl+D to close terminal (like most terminals)
        elif event.key == "ctrl+d":
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

    def analyze_terminal_output(self, text: str) -> dict:
        """
        Analyze terminal output using enhanced escape sequence parsing.

        Returns:
            Dictionary with parsed information about colors, styles, etc.
        """
        if not TERMINAL_AVAILABLE:
            return {}

        try:
            parts = terminal_enhancements.parse_enhanced_escape_sequences(text)

            analysis = {
                'has_colors': False,
                'has_styles': False,
                'title_changes': [],
                'color_sequences': [],
                'text_parts': [],
            }

            for text_part, seq_type, params in parts:
                if seq_type == 'text':
                    analysis['text_parts'].append(text_part)
                elif seq_type == 'csi' and params.get('command') == 'm':
                    # Color/style sequence
                    color_info = terminal_enhancements.parse_color_sequence(params.get('params', []))
                    analysis['color_sequences'].append(color_info)
                    if color_info.get('fg_color') or color_info.get('bg_color'):
                        analysis['has_colors'] = True
                    if any(color_info.get(k) for k in ['bold', 'italic', 'underline']):
                        analysis['has_styles'] = True
                elif seq_type == 'title':
                    analysis['title_changes'].append(params.get('title', ''))

            return analysis

        except Exception as e:
            # Fallback to empty analysis if parsing fails
            return {}

    def on_mount(self) -> None:
        """Called when terminal window is mounted."""
        terminal = self.query_one("#terminal", Terminal)
        if TERMINAL_AVAILABLE:
            terminal.start()  # Start the terminal emulator
        terminal.focus()

    def on_unmount(self) -> None:
        """Called when terminal window is unmounted - cleanup wrapper script."""
        import os
        if self.wrapper_script_path and os.path.exists(self.wrapper_script_path):
            try:
                os.unlink(self.wrapper_script_path)
            except OSError:
                pass  # Ignore cleanup errors