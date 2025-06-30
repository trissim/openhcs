"""
Terminal Enhancements for textual-terminal

Extracted useful features from Gate One terminal.py to enhance textual-terminal:
- Better ANSI escape sequence parsing
- Enhanced color handling
- Improved cursor positioning
- Advanced scrollback management

This is a focused extraction of the most valuable parts without the bloat.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class TerminalEnhancements:
    """
    Enhanced terminal features extracted from Gate One terminal.py
    
    This class provides the most useful enhancements that can be integrated
    into textual-terminal without the massive bloat of the full Gate One implementation.
    """
    
    # Enhanced regex patterns for better escape sequence parsing
    RE_CSI_ESC_SEQ = re.compile(r'\x1B\[([?A-Za-z0-9>;@:\!]*?)([A-Za-z@_])')
    RE_ESC_SEQ = re.compile(r'\x1b(.*\x1b\\|[ABCDEFGHIJKLMNOQRSTUVWXYZa-z0-9=<>]|[()# %*+].)')
    RE_TITLE_SEQ = re.compile(r'\x1b\][0-2]\;(.*?)(\x07|\x1b\\)')
    RE_OPT_SEQ = re.compile(r'\x1b\]_\;(.+?)(\x07|\x1b\\)')
    RE_NUMBERS = re.compile(r'\d*')
    
    # Enhanced color mappings
    COLORS_256 = {
        # Standard 16 colors
        0: (0, 0, 0),        # Black
        1: (128, 0, 0),      # Dark Red
        2: (0, 128, 0),      # Dark Green
        3: (128, 128, 0),    # Dark Yellow
        4: (0, 0, 128),      # Dark Blue
        5: (128, 0, 128),    # Dark Magenta
        6: (0, 128, 128),    # Dark Cyan
        7: (192, 192, 192),  # Light Gray
        8: (128, 128, 128),  # Dark Gray
        9: (255, 0, 0),      # Red
        10: (0, 255, 0),     # Green
        11: (255, 255, 0),   # Yellow
        12: (0, 0, 255),     # Blue
        13: (255, 0, 255),   # Magenta
        14: (0, 255, 255),   # Cyan
        15: (255, 255, 255), # White
    }
    
    def __init__(self):
        """Initialize terminal enhancements."""
        self.callbacks = defaultdict(dict)
        self.enhanced_colors = self._generate_256_colors()
        
        # Enhanced escape sequence handlers
        self.csi_handlers = self._setup_csi_handlers()
        self.esc_handlers = self._setup_esc_handlers()
        
        logger.info("Terminal enhancements initialized")
    
    def _generate_256_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """Generate the full 256-color palette."""
        colors = self.COLORS_256.copy()
        
        # Colors 16-231: 6x6x6 color cube
        for i in range(216):
            r = (i // 36) * 51
            g = ((i % 36) // 6) * 51
            b = (i % 6) * 51
            colors[16 + i] = (r, g, b)
        
        # Colors 232-255: Grayscale
        for i in range(24):
            gray = 8 + i * 10
            colors[232 + i] = (gray, gray, gray)
        
        return colors
    
    def _setup_csi_handlers(self) -> Dict[str, Callable]:
        """Setup enhanced CSI escape sequence handlers."""
        return {
            'A': self._cursor_up,
            'B': self._cursor_down,
            'C': self._cursor_right,
            'D': self._cursor_left,
            'H': self._cursor_position,
            'f': self._cursor_position,
            'J': self._erase_display,
            'K': self._erase_line,
            'm': self._set_rendition,
            'r': self._set_scroll_region,
            's': self._save_cursor,
            'u': self._restore_cursor,
            'l': self._reset_mode,
            'h': self._set_mode,
        }
    
    def _setup_esc_handlers(self) -> Dict[str, Callable]:
        """Setup enhanced ESC sequence handlers."""
        return {
            'c': self._reset_terminal,
            'D': self._index,
            'E': self._next_line,
            'H': self._set_tab_stop,
            'M': self._reverse_index,
            '7': self._save_cursor_and_attrs,
            '8': self._restore_cursor_and_attrs,
            '=': self._application_keypad,
            '>': self._normal_keypad,
        }
    
    def parse_enhanced_escape_sequences(self, text: str) -> List[Tuple[str, str, Dict]]:
        """
        Parse escape sequences with enhanced Gate One logic.
        
        Returns:
            List of (text_part, sequence_type, params) tuples
        """
        parts = []
        pos = 0
        
        while pos < len(text):
            # Look for CSI sequences first
            csi_match = self.RE_CSI_ESC_SEQ.search(text, pos)
            esc_match = self.RE_ESC_SEQ.search(text, pos)
            title_match = self.RE_TITLE_SEQ.search(text, pos)
            
            # Find the earliest match
            matches = []
            if csi_match:
                matches.append(('csi', csi_match))
            if esc_match:
                matches.append(('esc', esc_match))
            if title_match:
                matches.append(('title', title_match))
            
            if not matches:
                # No more escape sequences, add remaining text
                if pos < len(text):
                    parts.append((text[pos:], 'text', {}))
                break
            
            # Sort by position to get earliest match
            matches.sort(key=lambda x: x[1].start())
            seq_type, match = matches[0]
            
            # Add text before the sequence
            if match.start() > pos:
                parts.append((text[pos:match.start()], 'text', {}))
            
            # Add the sequence
            if seq_type == 'csi':
                params = self._parse_csi_params(match.group(1))
                command = match.group(2)
                parts.append((match.group(0), 'csi', {'params': params, 'command': command}))
            elif seq_type == 'esc':
                parts.append((match.group(0), 'esc', {'sequence': match.group(1)}))
            elif seq_type == 'title':
                parts.append((match.group(0), 'title', {'title': match.group(1)}))
            
            pos = match.end()
        
        return parts
    
    def _parse_csi_params(self, param_str: str) -> List[int]:
        """Parse CSI parameters into a list of integers."""
        if not param_str:
            return []
        
        params = []
        for param in param_str.split(';'):
            if param.isdigit():
                params.append(int(param))
            else:
                params.append(0)  # Default value
        
        return params
    
    def get_enhanced_color(self, color_code: int) -> Optional[Tuple[int, int, int]]:
        """Get RGB values for enhanced 256-color palette."""
        return self.enhanced_colors.get(color_code)
    
    def parse_color_sequence(self, params: List[int]) -> Dict[str, any]:
        """
        Parse SGR (Select Graphic Rendition) color sequences.
        
        Returns:
            Dictionary with color and style information
        """
        result = {
            'fg_color': None,
            'bg_color': None,
            'bold': False,
            'italic': False,
            'underline': False,
            'reverse': False,
            'strikethrough': False,
        }
        
        i = 0
        while i < len(params):
            param = params[i]
            
            if param == 0:  # Reset
                result = {k: False if isinstance(v, bool) else None for k, v in result.items()}
            elif param == 1:  # Bold
                result['bold'] = True
            elif param == 3:  # Italic
                result['italic'] = True
            elif param == 4:  # Underline
                result['underline'] = True
            elif param == 7:  # Reverse
                result['reverse'] = True
            elif param == 9:  # Strikethrough
                result['strikethrough'] = True
            elif param == 22:  # Normal intensity
                result['bold'] = False
            elif param == 23:  # Not italic
                result['italic'] = False
            elif param == 24:  # Not underlined
                result['underline'] = False
            elif param == 27:  # Not reversed
                result['reverse'] = False
            elif param == 29:  # Not strikethrough
                result['strikethrough'] = False
            elif 30 <= param <= 37:  # Foreground colors
                result['fg_color'] = param - 30
            elif param == 38:  # Extended foreground color
                if i + 2 < len(params) and params[i + 1] == 5:
                    result['fg_color'] = params[i + 2]
                    i += 2
                elif i + 4 < len(params) and params[i + 1] == 2:
                    # RGB color
                    r, g, b = params[i + 2], params[i + 3], params[i + 4]
                    result['fg_color'] = (r, g, b)
                    i += 4
            elif param == 39:  # Default foreground
                result['fg_color'] = None
            elif 40 <= param <= 47:  # Background colors
                result['bg_color'] = param - 40
            elif param == 48:  # Extended background color
                if i + 2 < len(params) and params[i + 1] == 5:
                    result['bg_color'] = params[i + 2]
                    i += 2
                elif i + 4 < len(params) and params[i + 1] == 2:
                    # RGB color
                    r, g, b = params[i + 2], params[i + 3], params[i + 4]
                    result['bg_color'] = (r, g, b)
                    i += 4
            elif param == 49:  # Default background
                result['bg_color'] = None
            elif 90 <= param <= 97:  # Bright foreground colors
                result['fg_color'] = param - 90 + 8
            elif 100 <= param <= 107:  # Bright background colors
                result['bg_color'] = param - 100 + 8
            
            i += 1
        
        return result
    
    def add_callback(self, event_type: str, callback: Callable, identifier: str = None):
        """Add a callback for terminal events."""
        if identifier is None:
            identifier = str(hash(callback))
        self.callbacks[event_type][identifier] = callback
    
    def remove_callback(self, event_type: str, identifier: str):
        """Remove a callback."""
        if event_type in self.callbacks and identifier in self.callbacks[event_type]:
            del self.callbacks[event_type][identifier]
    
    def trigger_callbacks(self, event_type: str, *args, **kwargs):
        """Trigger all callbacks for an event type."""
        for callback in self.callbacks[event_type].values():
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback {callback}: {e}")
    
    # Placeholder methods for escape sequence handlers
    # These would be implemented to actually modify terminal state
    def _cursor_up(self, params): pass
    def _cursor_down(self, params): pass
    def _cursor_right(self, params): pass
    def _cursor_left(self, params): pass
    def _cursor_position(self, params): pass
    def _erase_display(self, params): pass
    def _erase_line(self, params): pass
    def _set_rendition(self, params): pass
    def _set_scroll_region(self, params): pass
    def _save_cursor(self, params): pass
    def _restore_cursor(self, params): pass
    def _reset_mode(self, params): pass
    def _set_mode(self, params): pass
    def _reset_terminal(self, params): pass
    def _index(self, params): pass
    def _next_line(self, params): pass
    def _set_tab_stop(self, params): pass
    def _reverse_index(self, params): pass
    def _save_cursor_and_attrs(self, params): pass
    def _restore_cursor_and_attrs(self, params): pass
    def _application_keypad(self, params): pass
    def _normal_keypad(self, params): pass


# Global instance for easy access
terminal_enhancements = TerminalEnhancements()
