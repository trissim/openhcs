"""
UI Constants for consistent TUI styling and dimensions.

Centralizes all magic numbers and style strings for predictable UI.
"""
from typing import Dict


# Dimensions
UI_SPACER_WIDTH = 1
UI_SPACER_WIDTH_LARGE = 2
UI_LINE_HEIGHT = 1
UI_BUTTON_PADDING = 2
UI_BUTTON_PADDING_COMPACT = 1
UI_BUTTON_PADDING_PROMINENT = 4

# Dialog dimensions
DIALOG_WIDTH_SMALL = 50
DIALOG_WIDTH_MEDIUM = 70
DIALOG_WIDTH_LARGE = 80
DIALOG_WIDTH_XLARGE = 100

# Style strings - centralized for consistency
class STYLES:
    """UI style constants."""
    
    # Button styles
    BUTTON = "class:button"
    BUTTON_FOCUSED = "class:button.focused"
    BUTTON_DISABLED = "class:button.disabled"
    
    # Frame styles
    FRAME = "class:frame"
    FRAME_TITLE = "class:frame.title"
    FRAME_BORDER = "class:frame.border"
    
    # Text styles
    TEXT_NORMAL = ""
    TEXT_ERROR = "class:error-text"
    TEXT_INFO = "class:info-text"
    TEXT_WARNING = "class:warning-text"
    TEXT_SUCCESS = "class:success-text"
    TEXT_TITLE = "class:title"
    
    # List styles
    LIST_ITEM = "class:list.item"
    LIST_ITEM_FOCUSED = "class:list.item.focused"
    LIST_ITEM_SELECTED = "class:list.item.selected"
    
    # File browser styles
    FILEBROWSER_ITEM = "class:filebrowser.item"
    FILEBROWSER_ITEM_FOCUSED = "class:filebrowser.item.focused"
    FILEBROWSER_ITEM_SELECTED = "class:filebrowser.item.selected"
    
    # Move button styles
    MOVE_BUTTON = "class:move-button"
    MOVE_BUTTON_DISABLED = "class:disabled-button"
    
    # Main content
    MAIN_CONTENT = "class:main-content"


# Common character constants
class CHARS:
    """UI character constants."""
    SPACER = ' '
    HORIZONTAL_LINE = '─'
    VERTICAL_LINE = '│'
    
    # Status symbols
    STATUS_NOT_INITIALIZED = '?'
    STATUS_INITIALIZED = '-'
    STATUS_READY = 'o'
    STATUS_RUNNING = '!'
    
    # Navigation
    ARROW_UP = '↑'
    ARROW_DOWN = '↓'
    ARROW_LEFT = '←'
    ARROW_RIGHT = '→'
    
    # File browser
    FOLDER_ICON = '📁'
    FILE_ICON = '📄'


# Status symbol mappings
STATUS_SYMBOLS: Dict[str, str] = {
    '?': CHARS.STATUS_NOT_INITIALIZED,
    'not_initialized': CHARS.STATUS_NOT_INITIALIZED,
    'created': CHARS.STATUS_NOT_INITIALIZED,
    
    '-': CHARS.STATUS_INITIALIZED,
    'initialized': CHARS.STATUS_INITIALIZED,
    'yellow': CHARS.STATUS_INITIALIZED,
    
    'o': CHARS.STATUS_READY,
    'ready': CHARS.STATUS_READY,
    'compiled': CHARS.STATUS_READY,
    'green': CHARS.STATUS_READY,
    
    '!': CHARS.STATUS_RUNNING,
    'running': CHARS.STATUS_RUNNING,
    'error': CHARS.STATUS_RUNNING,
    'red': CHARS.STATUS_RUNNING,
}


def get_status_symbol(status: str) -> str:
    """Get status symbol for a given status string."""
    return STATUS_SYMBOLS.get(status, CHARS.STATUS_NOT_INITIALIZED)
