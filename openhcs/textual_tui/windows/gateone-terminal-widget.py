"""
Gate One Terminal Widget for Textual
A terminal emulator widget using Gate One's terminal.py instead of pyte
"""

import os
import pty
import asyncio
import struct
import fcntl
import termios
import logging
from typing import Optional, Tuple, List, Callable
from pathlib import Path

from rich.text import Text
from rich.style import Style
from rich.console import Console
from rich.segment import Segment
from rich.color import Color as RichColor

from textual.widget import Widget
from textual.reactive import reactive
from textual.geometry import Size
from textual import events
from textual.strip import Strip

# You'll need to extract terminal.py from Gate One
# Download from: https://github.com/liftoff/GateOne
try:
    from terminal import Terminal as GateOneTerminal
except ImportError:
    raise ImportError(
        "Please extract terminal.py from Gate One:\n"
        "1. git clone https://github.com/liftoff/GateOne\n"
        "2. Copy GateOne/gateone/terminal.py to your project"
    )


class GateOneTextualTerminal(Widget):
    """A Textual widget that uses Gate One's terminal emulator."""
    
    DEFAULT_CSS = """
    GateOneTextualTerminal {
        width: 100%;
        height: 100%;
    }
    """
    
    # Reactive properties
    cursor_visible = reactive(True)
    cursor_style = reactive(1)  # 1=block, 3=underline, 5=bar
    
    def __init__(
        self,
        command: Optional[str] = None,
        size: Tuple[int, int] = (80, 24),
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Terminal dimensions
        self.cols, self.rows = size
        
        # Command to run
        self.command = command or os.environ.get('SHELL', '/bin/bash')
        
        # Gate One terminal instance
        self.terminal = GateOneTerminal(rows=self.rows, cols=self.cols)
        
        # Set up callbacks
        self._setup_callbacks()
        
        # PTY file descriptors
        self.pty_master = None
        self.pty_pid = None
        
        # Async reader task
        self._reader_task = None
        
        # Color palette (will be populated from terminal)
        self.colors = self._default_colors()
        
        # For handling OSC sequences (like pywal)
        self.osc_buffer = ""
        
        # Logger
        self.log = logging.getLogger(__name__)
        
    def _default_colors(self) -> dict:
        """Default 16-color palette."""
        return {
            0: "#000000",   # Black
            1: "#cd0000",   # Red
            2: "#00cd00",   # Green
            3: "#cdcd00",   # Yellow
            4: "#0000ee",   # Blue
            5: "#cd00cd",   # Magenta
            6: "#00cdcd",   # Cyan
            7: "#e5e5e5",   # White
            8: "#7f7f7f",   # Bright Black
            9: "#ff0000",   # Bright Red
            10: "#00ff00",  # Bright Green
            11: "#ffff00",  # Bright Yellow
            12: "#5c5cff",  # Bright Blue
            13: "#ff00ff",  # Bright Magenta
            14: "#00ffff",  # Bright Cyan
            15: "#ffffff",  # Bright White
        }
        
    def _setup_callbacks(self):
        """Set up Gate One terminal callbacks."""
        t = self.terminal
        
        # Screen content changed
        t.callbacks[t.CALLBACK_CHANGED] = self._on_terminal_change
        
        # Cursor position changed
        t.callbacks[t.CALLBACK_CURSOR_POS] = self._on_cursor_move
        
        # Mode changes (for cursor styles, etc)
        t.callbacks[t.CALLBACK_MODE] = self._on_mode_change
        
        # Title changes
        t.callbacks[t.CALLBACK_TITLE] = self._on_title_change
        
        # Bell
        t.callbacks[t.CALLBACK_BELL] = self._on_bell
        
        # For handling special escape sequences
        t.callbacks[t.CALLBACK_ESC] = self._on_escape_sequence
        
        # OSC sequences (for colors)
        t.callbacks[t.CALLBACK_OSC] = self._on_osc_sequence
        
    def _on_terminal_change(self):
        """Called when terminal content changes."""
        self.refresh()
        
    def _on_cursor_move(self):
        """Called when cursor moves."""
        self.refresh()
        
    def _on_mode_change(self, mode, value):
        """Called when terminal mode changes."""
        self.log.debug(f"Mode change: {mode} = {value}")
        
        # Handle cursor visibility
        if mode == 'cursor':
            self.cursor_visible = value
            
        self.refresh()
        
    def _on_title_change(self, title):
        """Called when terminal title changes."""
        # You could emit an event here to update window title
        self.log.debug(f"Title changed: {title}")
        
    def _on_bell(self):
        """Called when bell character is received."""
        # You could flash the screen or play a sound
        self.log.debug("Bell!")
        
    def _on_escape_sequence(self, seq):
        """Handle special escape sequences like cursor style."""
        self.log.debug(f"Escape sequence: {repr(seq)}")
        
        # Check for DECSCUSR (cursor style) - ESC [ Ps SP q
        if seq.endswith(' q'):
            try:
                # Extract parameter
                param = seq[2:-2].strip()
                if param.isdigit():
                    style = int(param)
                    if style in [0, 1, 2]:
                        self.cursor_style = 1  # Block
                    elif style in [3, 4]:
                        self.cursor_style = 3  # Underline
                    elif style in [5, 6]:
                        self.cursor_style = 5  # Bar
                    self.log.debug(f"Cursor style changed to: {self.cursor_style}")
            except:
                pass
                
    def _on_osc_sequence(self, command, text):
        """Handle OSC sequences (for pywal colors)."""
        self.log.debug(f"OSC: command={command}, text={text}")
        
        # OSC 4 - Change color palette
        if command == 4:
            # Format: "index;rgb:rr/gg/bb"
            parts = text.split(';', 1)
            if len(parts) == 2:
                try:
                    index = int(parts[0])
                    color = parts[1]
                    if color.startswith('rgb:'):
                        # Convert rgb:rr/gg/bb to #rrggbb
                        rgb = color[4:].split('/')
                        if len(rgb) == 3:
                            # Handle both 8-bit and 16-bit formats
                            r = rgb[0][:2]
                            g = rgb[1][:2]
                            b = rgb[2][:2]
                            hex_color = f"#{r}{g}{b}"
                            self.colors[index] = hex_color
                            self.log.debug(f"Color {index} = {hex_color}")
                except:
                    pass
                    
        # OSC 10 - Change foreground color
        elif command == 10:
            self.log.debug(f"Foreground color: {text}")
            
        # OSC 11 - Change background color
        elif command == 11:
            self.log.debug(f"Background color: {text}")
            
    async def on_mount(self):
        """Start the terminal when widget is mounted."""
        await self.start_terminal()
        
    async def on_unmount(self):
        """Clean up when widget is unmounted."""
        await self.stop_terminal()
        
    async def start_terminal(self):
        """Start the PTY and terminal process."""
        # Create PTY
        self.pty_master, pty_slave = pty.openpty()
        
        # Get terminal size
        cols, rows = self.size.width, self.size.height
        if cols > 0 and rows > 0:
            self.resize_terminal(cols, rows)
        
        # Fork and exec
        self.pty_pid = os.fork()
        
        if self.pty_pid == 0:  # Child process
            os.close(self.pty_master)
            
            # Make slave the controlling terminal
            os.setsid()
            os.dup2(pty_slave, 0)  # stdin
            os.dup2(pty_slave, 1)  # stdout
            os.dup2(pty_slave, 2)  # stderr
            os.close(pty_slave)
            
            # Set environment
            env = os.environ.copy()
            env['TERM'] = 'xterm-256color'  # Gate One supports this well
            env['COLORTERM'] = 'truecolor'
            
            # Load pywal colors if available
            pywal_sequence_file = Path.home() / '.cache' / 'wal' / 'sequences'
            if pywal_sequence_file.exists():
                # This will be sent to our terminal on startup
                env['PYWAL_SEQUENCES'] = str(pywal_sequence_file)
            
            # Execute shell
            os.execve(self.command, [self.command], env)
        
        else:  # Parent process
            os.close(pty_slave)
            
            # Make non-blocking
            import fcntl
            flags = fcntl.fcntl(self.pty_master, fcntl.F_GETFL)
            fcntl.fcntl(self.pty_master, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            
            # Start reader task
            self._reader_task = asyncio.create_task(self._read_pty())
            
            # Send pywal sequences if available
            await self._apply_pywal_colors()
            
    async def _apply_pywal_colors(self):
        """Apply pywal color sequences if available."""
        pywal_sequence_file = Path.home() / '.cache' / 'wal' / 'sequences'
        if pywal_sequence_file.exists():
            try:
                sequences = pywal_sequence_file.read_bytes()
                os.write(self.pty_master, sequences)
                self.log.debug("Applied pywal color sequences")
            except Exception as e:
                self.log.error(f"Failed to apply pywal colors: {e}")
                
    async def stop_terminal(self):
        """Stop the terminal process."""
        if self._reader_task:
            self._reader_task.cancel()
            
        if self.pty_master:
            os.close(self.pty_master)
            
        if self.pty_pid:
            try:
                os.kill(self.pty_pid, 9)
                os.waitpid(self.pty_pid, 0)
            except:
                pass
                
    async def _read_pty(self):
        """Read data from PTY and feed to terminal."""
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                # Read available data
                data = await loop.run_in_executor(None, os.read, self.pty_master, 4096)
                if data:
                    # Decode and feed to terminal
                    text = data.decode('utf-8', errors='replace')
                    self.terminal.write(text)
                else:
                    break
            except OSError:
                await asyncio.sleep(0.01)
            except Exception as e:
                self.log.error(f"PTY read error: {e}")
                break
                
    def resize_terminal(self, cols: int, rows: int):
        """Resize the terminal."""
        self.cols = cols
        self.rows = rows
        
        # Resize Gate One terminal
        self.terminal.resize(rows, cols)
        
        # Resize PTY
        if self.pty_master:
            size = struct.pack('HHHH', rows, cols, 0, 0)
            fcntl.ioctl(self.pty_master, termios.TIOCSWINSZ, size)
            
    def on_resize(self, event):
        """Handle widget resize."""
        # Calculate character dimensions
        cols = self.size.width
        rows = self.size.height
        
        if cols > 0 and rows > 0:
            self.resize_terminal(cols, rows)
            
    def render_line(self, y: int) -> Strip:
        """Render a single line of the terminal."""
        segments = []
        
        # Get line from terminal
        if y < len(self.terminal.screen):
            line = self.terminal.screen[y]
            
            for x, char in enumerate(line):
                # Get character data
                char_data = char.data if hasattr(char, 'data') else ' '
                if not char_data:
                    char_data = ' '
                    
                # Build style from character attributes
                style = Style()
                
                # Foreground color
                if hasattr(char, 'fg') and char.fg != 'default':
                    if isinstance(char.fg, int) and char.fg < 16:
                        style = style + Style(color=self.colors.get(char.fg, "#ffffff"))
                    else:
                        style = style + Style(color=str(char.fg))
                        
                # Background color
                if hasattr(char, 'bg') and char.bg != 'default':
                    if isinstance(char.bg, int) and char.bg < 16:
                        style = style + Style(bgcolor=self.colors.get(char.bg, "#000000"))
                    else:
                        style = style + Style(bgcolor=str(char.bg))
                        
                # Text attributes
                if hasattr(char, 'bold') and char.bold:
                    style = style + Style(bold=True)
                if hasattr(char, 'italic') and char.italic:
                    style = style + Style(italic=True)
                if hasattr(char, 'underscore') and char.underscore:
                    style = style + Style(underline=True)
                if hasattr(char, 'strikethrough') and char.strikethrough:
                    style = style + Style(strike=True)
                    
                # IMPORTANT: Handle reverse video (inversion)
                if hasattr(char, 'reverse') and char.reverse:
                    # Swap foreground and background
                    fg = style.color
                    bg = style.bgcolor
                    style = Style(
                        color=bg or "#000000",
                        bgcolor=fg or "#ffffff",
                        bold=style.bold,
                        italic=style.italic,
                        underline=style.underline,
                        strike=style.strike
                    )
                    
                # Handle cursor
                is_cursor = (x == self.terminal.cursor.x and 
                           y == self.terminal.cursor.y and
                           self.cursor_visible and
                           not self.terminal.cursor.hidden)
                           
                if is_cursor:
                    if self.cursor_style == 1:  # Block cursor
                        style = style + Style(reverse=True, blink=True)
                    elif self.cursor_style == 3:  # Underline cursor  
                        style = style + Style(underline=True, bold=True, blink=True)
                    elif self.cursor_style == 5:  # Bar cursor
                        # For bar cursor, we modify the character
                        if char_data == ' ':
                            char_data = '▎'  # Left one-eighth block
                        style = style + Style(bold=True, blink=True)
                        
                segments.append(Segment(char_data, style))
                
        return Strip(segments)
        
    def on_key(self, event: events.Key) -> None:
        """Handle keyboard input."""
        if not self.pty_master:
            return
            
        # Map key to escape sequence
        key_map = {
            "up": "\x1b[A",
            "down": "\x1b[B",
            "right": "\x1b[C",
            "left": "\x1b[D",
            "home": "\x1b[H",
            "end": "\x1b[F",
            "pageup": "\x1b[5~",
            "pagedown": "\x1b[6~",
            "insert": "\x1b[2~",
            "delete": "\x1b[3~",
            "f1": "\x1bOP",
            "f2": "\x1bOQ",
            "f3": "\x1bOR",
            "f4": "\x1bOS",
            "f5": "\x1b[15~",
            "f6": "\x1b[17~",
            "f7": "\x1b[18~",
            "f8": "\x1b[19~",
            "f9": "\x1b[20~",
            "f10": "\x1b[21~",
            "f11": "\x1b[23~",
            "f12": "\x1b[24~",
            "escape": "\x1b",
            "enter": "\r",
            "tab": "\t",
            "backspace": "\x7f",
        }
        
        # Check for special keys
        if event.key in key_map:
            data = key_map[event.key]
        elif event.character:
            # Regular character
            data = event.character
            
            # Handle Ctrl combinations
            if event.key.startswith("ctrl+"):
                char = event.key[5:]
                if len(char) == 1 and char.isalpha():
                    # Ctrl+A = 1, Ctrl+B = 2, etc.
                    data = chr(ord(char.upper()) - ord('A') + 1)
        else:
            return
            
        # Write to PTY
        try:
            os.write(self.pty_master, data.encode('utf-8'))
        except:
            pass
            
    def on_paste(self, event: events.Paste) -> None:
        """Handle paste events."""
        if self.pty_master and event.text:
            try:
                # Send bracketed paste sequences
                os.write(self.pty_master, b'\x1b[200~')
                os.write(self.pty_master, event.text.encode('utf-8'))
                os.write(self.pty_master, b'\x1b[201~')
            except:
                pass


# Example usage with your window system
if __name__ == "__main__":
    from textual.app import App
    
    class TerminalApp(App):
        def compose(self):
            yield GateOneTextualTerminal()
            
    app = TerminalApp()
    app.run()
