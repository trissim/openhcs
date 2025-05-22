"""
Status Bar and Log Drawer for OpenHCS TUI.

This module implements the status bar and expandable log drawer for the OpenHCS TUI,
providing real-time feedback on operations and detailed logging information.

Based on plan: plan_06_status_bar.md

🔒 Clause 3: Declarative Primacy
Status indicators follow a consistent visual language.
LogFormatter separates formatting logic.

🔒 Clause 245: Declarative Enforcement
Drawer expansion/collapse is explicitly controlled.
Status messages have explicit priority.

🔒 Clause 246: Statelessness Mandate / Bounded Log Buffer
Log buffer has a declared maximum size.
StatusBarState is an immutable state container.

🔒 Clause 317: Runtime Correctness / Thread Safety
All status updates are protected by a lock.
"""
import asyncio
import logging # For TUIStatusBarLogHandler
import time # For log timestamps
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Deque, ClassVar, Union

# Renamed to avoid conflict with standard library 'Lock' if used directly
from asyncio import Lock as AsyncioLock 

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout import ConditionalContainer, Container, HSplit
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.widgets import Box, Frame, Label

# Assuming TUIState will be imported where StatusBar is used, or passed in.
# For now, type hint as string if direct import is an issue here.
# from .tui_architecture import TUIState # Example if in same package


# Status message priority levels
class Priority(Enum):
    """Priority levels for status messages."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

# Log levels
class LogLevel(Enum):
    """Log levels for log entries."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        """Convert string to LogLevel enum."""
        try:
            return cls(level_str.upper())
        except ValueError:
            raise ValueError(f"Unknown log level: {level_str}. Valid levels are: {', '.join([l.value for l in cls])}")

    @classmethod
    def from_logging_level(cls, level_no: int) -> 'LogLevel':
        """Convert Python logging level number to LogLevel enum."""
        if level_no >= logging.CRITICAL: return cls.CRITICAL
        if level_no >= logging.ERROR: return cls.ERROR
        if level_no >= logging.WARNING: return cls.WARNING
        if level_no >= logging.INFO: return cls.INFO
        if level_no >= logging.DEBUG: return cls.DEBUG
        return cls.DEBUG # Default for custom levels lower than DEBUG


# Status icons for different states
STATUS_ICONS = {
    'idle': '○',
    'pending': '⋯', # Or '...'
    'running': '◔', # Or a spinner
    'success': '●', # Or checkmark ✓ ✔
    'error': '✗'   # Or '❗'
}


@dataclass(frozen=True)
class StatusBarSchema:
    """Schema for validating StatusBarState."""
    max_log_entries: ClassVar[int] = 1000 # Default, can be overridden in StatusBar init
    
    @staticmethod
    def validate_priority(priority: Priority) -> None:
        """Validate priority is a valid Priority enum value."""
        if not isinstance(priority, Priority):
            raise ValueError(f"Priority must be a Priority enum value, got {type(priority)}")
    
    @staticmethod
    def validate_log_entry(entry: Dict[str, Any]) -> None:
        """Validate log entry has required fields."""
        required_fields = ['timestamp', 'level', 'message']
        for field_name in required_fields:
            if field_name not in entry:
                raise ValueError(f"Log entry missing required field: {field_name}")
        
        if 'level' in entry:
            try:
                LogLevel(entry['level'].upper()) # Ensure level is valid LogLevel string
            except ValueError:
                raise ValueError(f"Invalid log level in entry: {entry['level']}. Valid levels are: {', '.join([l.value for l in LogLevel])}")


@dataclass(frozen=True) # Immutable state object
class StatusBarState:
    """
    Immutable state container for StatusBar.
    All state transitions result in a new StatusBarState instance.
    """
    status_message: str = ""
    status_priority: Priority = Priority.INFO
    operation_status: str = 'idle' # e.g., 'idle', 'running', 'success', 'error'
    drawer_expanded: bool = False
    log_buffer: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=StatusBarSchema.max_log_entries))
    
    def __post_init__(self):
        """Validate state after initialization."""
        StatusBarSchema.validate_priority(self.status_priority)
        if self.operation_status not in STATUS_ICONS:
            raise ValueError(f"Invalid operation_status: {self.operation_status}. Valid statuses are: {', '.join(STATUS_ICONS.keys())}")

    def with_status_message(self, message: str, priority: Priority) -> 'StatusBarState':
        """Create a new state with updated status message, respecting priority."""
        StatusBarSchema.validate_priority(priority)
        # Only update if new message has higher or equal priority
        if priority.value >= self.status_priority.value:
            return dataclasses.replace(self, status_message=message, status_priority=priority)
        return self

    def with_operation_status(self, operation_status: str) -> 'StatusBarState':
        """Create a new state with updated operation status."""
        if operation_status not in STATUS_ICONS:
            raise ValueError(f"Invalid operation_status: {operation_status}")
        return dataclasses.replace(self, operation_status=operation_status)

    def with_drawer_expanded(self, expanded: bool) -> 'StatusBarState':
        """Create a new state with updated drawer expansion state."""
        return dataclasses.replace(self, drawer_expanded=expanded)
    
    def with_log_entry(self, entry: Dict[str, Any]) -> 'StatusBarState':
        """Create a new state with an additional log entry."""
        StatusBarSchema.validate_log_entry(entry)
        new_buffer = deque(self.log_buffer, maxlen=self.log_buffer.maxlen) # Create new deque
        new_buffer.append(entry)
        return dataclasses.replace(self, log_buffer=new_buffer)


class LogFormatter:
    """Formats log entries for display."""
    LEVEL_STYLES: ClassVar[Dict[LogLevel, Tuple[str, str]]] = {
        LogLevel.DEBUG: ("class:log.debug", "#888888"), # Dim
        LogLevel.INFO: ("class:log.info", ""), # Default fg
        LogLevel.WARNING: ("class:log.warning", "ansiyellow"),
        LogLevel.ERROR: ("class:log.error", "ansired"),
        LogLevel.CRITICAL: ("class:log.critical", "ansibrightred bg:ansiblack"),
    }
    
    @classmethod
    def format_log_entry(cls, entry: Dict[str, Any]) -> FormattedText:
        """Format a single log entry as FormattedText."""
        ts = entry.get('timestamp', time.strftime('%H:%M:%S'))
        level_str = entry.get('level', LogLevel.INFO.value)
        msg = entry.get('message', '')
        src = entry.get('source', '')
        
        try:
            level_enum = LogLevel(level_str.upper())
        except ValueError:
            level_enum = LogLevel.INFO # Fallback
        
        style_class, _ = cls.LEVEL_STYLES.get(level_enum, cls.LEVEL_STYLES[LogLevel.INFO])
        
        segments = [
            ("", f"[{ts}] "),
            (style_class, f"{level_enum.value:<8}"), # Padded level
            ("", f": {msg}")
        ]
        if src:
            segments.append(("", f" ({src})"))
        
        return FormattedText(segments)
    
    @classmethod
    def format_log_entries(cls, entries: Deque[Dict[str, Any]]) -> FormattedText:
        """Format multiple log entries."""
        result_fragments: List[Tuple[str, str]] = []
        for entry in entries: # Iterate from oldest to newest
            formatted_entry_tuples = cls.format_log_entry(entry)
            result_fragments.extend(formatted_entry_tuples)
            result_fragments.append(("", "\n"))
        if not result_fragments and entries: # Handle case where entries exist but formatting yields nothing (should not happen)
             result_fragments.append(("", "Error formatting logs or no logs.\n"))
        elif not entries:
            result_fragments.append(("", "No log entries.\n"))

        return FormattedText(result_fragments[:-1]) # Remove last newline


class StatusBar(Container):
    """Status bar and expandable log drawer for the OpenHCS TUI."""
    
    def __init__(self, tui_state: Any, max_log_entries: int = 1000):
        """Initialize the status bar."""
        self.tui_state = tui_state # General TUI state
        
        # Internal, immutable state for the status bar itself
        self._status_bar_state = StatusBarState(
            log_buffer=deque(maxlen=max_log_entries)
        )
        
        self.status_lock = AsyncioLock() # For thread-safe updates to _status_bar_state

        # Create UI components
        self.status_label = self._create_status_label()
        self.log_drawer_content = self._create_log_drawer_content()
        self.log_drawer_container = ConditionalContainer(
            Frame(self.log_drawer_content, title="Log History (Click Status Bar to toggle)", height=10),
            filter=Condition(lambda: self._status_bar_state.drawer_expanded)
        )

        self.container = HSplit([
            self.status_label,
            self.log_drawer_container
        ])

        # Register for TUIState events
        self.tui_state.add_observer('operation_status_changed', self._on_operation_status_changed)
        self.tui_state.add_observer('error', self._on_error_event) # Distinguish from internal _on_error
        self.tui_state.add_observer('tui_log_level_changed', self._on_tui_log_level_changed) # Already present from draft
        
        # Listen for requests to toggle the log drawer (e.g., from MenuBar)
        self.tui_state.add_observer('ui_request_toggle_log_drawer',
                                    lambda data=None: get_app().create_background_task(self._toggle_drawer()))

        self._setup_logging_handler()

    def _create_status_label(self) -> Label:
        """Create the clickable status label."""
        def get_display_text() -> FormattedText:
            op_status_icon = STATUS_ICONS.get(self._status_bar_state.operation_status, '?')
            message = self._status_bar_state.status_message
            # Combine icon and message
            return FormattedText([
                ("", f"{op_status_icon} "),
                ("", message) # Style for message can be added if priority implies it
            ])

        label = Label(
            text=get_display_text,
            dont_extend_height=True,
            style="class:status-bar" # Base style
        )

        original_mouse_handler = label.mouse_handler
        def status_bar_mouse_handler(mouse_event):
            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                get_app().create_background_task(self._toggle_drawer())
                return True # Event handled
            return original_mouse_handler(mouse_event)
        label.mouse_handler = status_bar_mouse_handler
        return label

    def _create_log_drawer_content(self) -> Label:
        """Create the content area for the log drawer."""
        return Label(
            text=lambda: LogFormatter.format_log_entries(self._status_bar_state.log_buffer),
            dont_extend_width=True, # Allow horizontal scrolling if Frame enables it
            style="class:log-drawer"
        )

    async def _toggle_drawer(self) -> None:
        """Toggle the log drawer's expansion state."""
        async with self.status_lock:
            self._status_bar_state = self._status_bar_state.with_drawer_expanded(
                not self._status_bar_state.drawer_expanded
            )
        get_app().invalidate()

    async def set_status_message(self, message: str, priority: Priority = Priority.INFO, operation_status: Optional[str] = None) -> None:
        """Update the status message and/or operation status icon."""
        async with self.status_lock:
            new_state = self._status_bar_state.with_status_message(message, priority)
            if operation_status is not None:
                new_state = new_state.with_operation_status(operation_status)
            self._status_bar_state = new_state
        get_app().invalidate()

    async def add_log_entry(self, message: str, level: Union[str, LogLevel] = LogLevel.INFO, source: Optional[str] = None) -> None:
        """Add a new entry to the log buffer."""
        if isinstance(level, LogLevel):
            level_str = level.value
        elif isinstance(level, str):
            try:
                level_str = LogLevel(level.upper()).value
            except ValueError:
                level_str = LogLevel.INFO.value # Fallback
        else:
            level_str = LogLevel.INFO.value # Fallback

        entry = {
            'timestamp': time.strftime('%H:%M:%S'),
            'level': level_str,
            'message': message,
            'source': source or ''
        }
        async with self.status_lock:
            self._status_bar_state = self._status_bar_state.with_log_entry(entry)
        
        # If drawer is open, invalidate to refresh its content
        if self._status_bar_state.drawer_expanded:
            get_app().invalidate()

    def _setup_logging_handler(self):
        """Set up a custom logging handler to route logs to this status bar."""
        handler = TUIStatusBarLogHandler(self)
        # Configure formatter for the handler if needed, or rely on LogFormatter
        # Example:
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # handler.setFormatter(formatter)
        
        # Add handler to the root logger or specific loggers
        # Adding to root logger to capture all openhcs logs.
        # Be mindful of log levels to avoid flooding.
        root_logger = logging.getLogger("openhcs") # Or just logging.getLogger() for everything
        root_logger.addHandler(handler)
        # Ensure the logger's level allows messages to pass to this handler
        # For example, if TUI log level is INFO, set logger to INFO.
        # This might be configured globally elsewhere.
        # if root_logger.level == logging.NOTSET or root_logger.level > logging.DEBUG:
        #     root_logger.setLevel(logging.DEBUG) # Capture all, filter in handler or StatusBar
        self.tui_state.add_observer('tui_log_level_changed', self._on_tui_log_level_changed)


    async def _on_operation_status_changed(self, data: Dict[str, Any]):
        """Handle 'operation_status_changed' TUIState event."""
        message = data.get('message', self._status_bar_state.status_message)
        priority = data.get('priority', self._status_bar_state.status_priority)
        operation_status = data.get('status', 'idle') # 'idle', 'running', 'success', 'error'
        
        # Convert priority string from event to Priority enum if necessary
        if isinstance(priority, str):
            try:
                priority = Priority[priority.upper()]
            except KeyError:
                priority = Priority.INFO

        await self.set_status_message(message, priority, operation_status)
        
        log_level_str = data.get('log_level', 'INFO').upper()
        log_message = data.get('log_message', message)
        
        if log_message: # Log the operation status change if a log message is provided
            try:
                log_level = LogLevel(log_level_str)
            except ValueError:
                log_level = LogLevel.INFO
            await self.add_log_entry(log_message, level=log_level, source=data.get('source', 'TUIState'))


    async def _on_error_event(self, error_data: Dict[str, Any]):
        """Handle 'error' TUIState event by adding to log and updating status."""
        message = error_data.get('message', 'An unspecified error occurred.')
        details = error_data.get('details')
        source = error_data.get('source', 'Application')

        await self.set_status_message(message, Priority.ERROR, operation_status='error')
        
        log_msg = message
        if details:
            log_msg += f" | Details: {str(details)[:200]}" # Truncate details for log line
        await self.add_log_entry(log_msg, level=LogLevel.ERROR, source=source)

    def _on_tui_log_level_changed(self, new_level_str: str):
        """Responds to TUI log level changes from TUIState."""
        # This is primarily for the Python logging system, not the StatusBar's internal buffer filtering.
        # Filtering of the displayed log_buffer would be a separate feature.
        try:
            level_enum = LogLevel(new_level_str.upper())
            logging_level = getattr(logging, level_enum.value, logging.INFO)
            
            # Example: Adjust level of the logger that TUIStatusBarLogHandler is attached to
            logger_to_adjust = logging.getLogger("openhcs") # Or logging.getLogger()
            logger_to_adjust.setLevel(logging_level)
            asyncio.create_task(self.add_log_entry(f"TUI log capture level set to {level_enum.value}", LogLevel.INFO, "StatusBar"))
        except ValueError:
            asyncio.create_task(self.add_log_entry(f"Invalid TUI log level received: {new_level_str}", LogLevel.WARNING, "StatusBar"))


    def __pt_container__(self):
        return self.container

class TUIStatusBarLogHandler(logging.Handler):
    """A logging handler that directs log records to the TUI StatusBar."""
    def __init__(self, status_bar_instance: StatusBar, level=logging.NOTSET):
        super().__init__(level=level)
        self.status_bar = status_bar_instance

    def emit(self, record: logging.LogRecord):
        """Emit a log record to the status bar."""
        # This method can be called from any thread, so adding to log buffer
        # must be thread-safe. StatusBar.add_log_entry is async and uses a lock.
        # We need to schedule this async method on the TUI's event loop.
        
        log_level_enum = LogLevel.from_logging_level(record.levelno)
        
        # Basic filtering: only process if record level is >= handler level
        # More advanced filtering (e.g., based on TUIState.tui_log_level) could be added here
        # or within StatusBar.add_log_entry itself.
        # For now, let TUIState.tui_log_level control the Python logger's level.
        
        # Formatting can be done here or delegated to StatusBar's LogFormatter
        # For simplicity, pass raw message and let StatusBar format.
        message = self.format(record) # Use handler's formatter if set, else record.getMessage()
        
        # Get app and schedule the async task
        try:
            app = get_app()
            # Schedule add_log_entry as a background task on the app's event loop
            app.create_background_task(
                self.status_bar.add_log_entry(
                    message=message, # Or record.getMessage() if not using handler's formatter
                    level=log_level_enum,
                    source=record.name
                )
            )
        except RuntimeError: 
            # get_app() fails if no event loop is running or app not fully set up.
            # This can happen during early startup or late shutdown.
            # Fallback to stderr or a temporary buffer if critical.
            # For now, we might lose these logs, or print to stderr.
            print(f"TUI Log Handler Error: Could not get_app(). Log: {log_level_enum.value} - {message}", file=sys.stderr)