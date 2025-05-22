# plan_06_status_bar.md
## Component: Status Bar and Log Drawer

### Objective
Implement the status bar and expandable log drawer for the OpenHCS TUI, providing real-time feedback on operations and detailed logging information.

### Plan

1. **Status Bar Core Implementation**
   - Create a `StatusBar` class with proper initialization
   - Implement status message display with priority levels
   - Support for operation status indicators (idle, running, success, error)
   - Implement click-to-expand functionality for log drawer
   - Ensure thread-safe status updates with asyncio.Lock

2. **Log Drawer Implementation**
   - Create an expandable drawer for detailed log display
   - Implement scrollable log history with proper formatting
   - Support for log filtering by severity
   - Implement proper drawer lifecycle management
   - Ensure thread-safe log updates

3. **Integration with Logging System**
   - Create a custom log handler for the TUI
   - Capture log messages from OpenHCS components
   - Format log messages with timestamp, severity, and source
   - Implement log buffer with bounded size
   - Support for log persistence across sessions

4. **Visual Feedback System**
   - Implement consistent visual indicators for operations
   - Support for progress indicators during long-running operations
   - Implement status message prioritization
   - Provide clear visual distinction between status types
   - Ensure accessibility with color and symbol redundancy

5. **Thread-Safe State Management**
   - Implement lock-based state updates for status messages
   - Ensure all state mutations are protected by the lock
   - Prevent race conditions in UI updates
   - Maintain consistent state across async operations
   - Implement proper event propagation

### Findings

#### Key Considerations for Status Bar Implementation

1. **🔒 Thread Safety (Clause 317)**
   - All status updates must be protected by a lock
   - Log buffer access must be thread-safe
   - Implementation: `async with self.status_lock: ...`
   - Rationale: Prevents race conditions during concurrent operations

2. **🔒 Message Prioritization (Clause 245)**
   - Status messages must have explicit priority levels
   - Higher priority messages must override lower priority ones
   - Implementation: `self._update_status(message, priority=Priority.INFO)`
   - Rationale: Ensures critical messages are always visible

3. **🔒 Bounded Log Buffer (Clause 246)**
   - Log buffer must have a declared maximum size
   - Oldest logs must be removed when buffer is full
   - Implementation: `self.log_buffer = deque(maxlen=self.max_log_entries)`
   - Rationale: Prevents unbounded memory growth

4. **🔒 Visual Consistency (Clause 3)**
   - Status indicators must follow a consistent visual language
   - Same status types must have the same visual representation
   - Implementation: `STATUS_ICONS = {'idle': '○', 'running': '◔', 'success': '●', 'error': '✗'}`
   - Rationale: Maintains declarative UI principles

5. **🔒 Drawer Lifecycle Management (Clause 245)**
   - Drawer expansion/collapse must be explicitly controlled
   - No implicit state changes based on other UI operations
   - Implementation: `self._set_drawer_expanded(expanded)`
   - Rationale: Ensures declarative state management

### Implementation Draft

```python
"""
Status Bar and Log Drawer for OpenHCS TUI.

This module implements the status bar and expandable log drawer for the OpenHCS TUI,
providing real-time feedback on operations and detailed logging information.

🔒 Clause 3: Declarative Primacy
Status indicators follow a consistent visual language.

🔒 Clause 245: Declarative Enforcement
Drawer expansion/collapse is explicitly controlled.

🔒 Clause 246: Statelessness Mandate
Log buffer has a declared maximum size.

🔒 Clause 317: Runtime Correctness
All status updates are protected by a lock.
"""
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Deque, ClassVar, Union
from asyncio import AbstractEventLoop, Lock

from prompt_toolkit.layout import HSplit, VSplit, Container, ConditionalContainer
from prompt_toolkit.widgets import Label, Box, Frame
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.filters import Condition
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.application import get_app


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
        """
        Convert string to LogLevel enum.
        
        Args:
            level_str: String representation of log level
            
        Returns:
            LogLevel enum value
            
        Raises:
            ValueError: If level_str is not a valid log level
        """
        try:
            return cls(level_str)
        except ValueError:
            raise ValueError(f"Unknown log level: {level_str}. Valid levels are: {', '.join([l.value for l in cls])}")


# Status icons for different states
STATUS_ICONS = {
    'idle': '○',
    'pending': '⋯',
    'running': '◔',
    'success': '●',
    'error': '✗'
}


@dataclass(frozen=True)
class StatusBarSchema:
    """Schema for validating StatusBarState."""
    max_log_entries: ClassVar[int] = 1000
    
    @staticmethod
    def validate_priority(priority: Priority) -> None:
        """Validate priority is a valid Priority enum value."""
        if not isinstance(priority, Priority):
            raise ValueError(f"Priority must be a Priority enum value, got {type(priority)}")
    
    @staticmethod
    def validate_log_entry(entry: Dict[str, Any]) -> None:
        """Validate log entry has required fields."""
        required_fields = ['timestamp', 'level', 'message']
        for field in required_fields:
            if field not in entry:
                raise ValueError(f"Log entry missing required field: {field}")
        
        # Validate level is a valid LogLevel
        if 'level' in entry:
            try:
                LogLevel(entry['level'])
            except ValueError:
                raise ValueError(f"Invalid log level: {entry['level']}. Valid levels are: {', '.join([l.value for l in LogLevel])}")


@dataclass
class StatusBarState:
    """
    Immutable state container for StatusBar.
    
    🔒 Clause 246: Statelessness Mandate
    All state transitions are explicit and traceable.
    """
    status_message: str = ""
    status_priority: Priority = Priority.INFO
    drawer_expanded: bool = False
    log_buffer: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=StatusBarSchema.max_log_entries))
    
    def __post_init__(self):
        """Validate state after initialization."""
        StatusBarSchema.validate_priority(self.status_priority)
    
    def with_status_message(self, message: str, priority: Priority) -> 'StatusBarState':
        """
        Create a new state with updated status message.
        
        Args:
            message: The new status message
            priority: The priority level of the message
            
        Returns:
            A new StatusBarState instance with updated status message
        """
        StatusBarSchema.validate_priority(priority)
        
        # Only update if new message has higher or equal priority
        if priority.value >= self.status_priority.value:
            return StatusBarState(
                status_message=message,
                status_priority=priority,
                drawer_expanded=self.drawer_expanded,
                log_buffer=self.log_buffer
            )
        return self
    
    def with_drawer_expanded(self, expanded: bool) -> 'StatusBarState':
        """
        Create a new state with updated drawer expansion state.
        
        Args:
            expanded: Whether the drawer should be expanded
            
        Returns:
            A new StatusBarState instance with updated drawer expansion state
        """
        return StatusBarState(
            status_message=self.status_message,
            status_priority=self.status_priority,
            drawer_expanded=expanded,
            log_buffer=self.log_buffer
        )
    
    def with_log_entry(self, entry: Dict[str, Any]) -> 'StatusBarState':
        """
        Create a new state with an additional log entry.
        
        Args:
            entry: The log entry to add
            
        Returns:
            A new StatusBarState instance with the new log entry
        """
        StatusBarSchema.validate_log_entry(entry)
        
        # Create a new deque with the same maxlen
        new_buffer = deque(self.log_buffer, maxlen=self.log_buffer.maxlen)
        new_buffer.append(entry)
        
        return StatusBarState(
            status_message=self.status_message,
            status_priority=self.status_priority,
            drawer_expanded=self.drawer_expanded,
            log_buffer=new_buffer
        )


class LogFormatter:
    """
    Formats log entries for display.
    
    🔒 Clause 3: Declarative Primacy
    Separates formatting logic from data representation.
    """
    # Color mapping for log levels
    LEVEL_STYLES = {
        LogLevel.DEBUG: ("ansiblue", "#0000ff"),
        LogLevel.INFO: ("ansiblue", "#0000ff"),
        LogLevel.WARNING: ("ansiyellow", "#ffff00"),
        LogLevel.ERROR: ("ansired", "#ff0000"),
        LogLevel.CRITICAL: ("ansired", "#ff0000"),
    }
    
    @classmethod
    def format_log_entry(cls, entry: Dict[str, Any]) -> FormattedText:
        """
        Format a log entry as FormattedText.
        
        Args:
            entry: Log entry dictionary
            
        Returns:
            FormattedText for display
        """
        timestamp = entry.get('timestamp', '')
        level_str = entry.get('level', LogLevel.INFO.value)
        message = entry.get('message', '')
        source = entry.get('source', '')
        
        # Get level enum
        try:
            level = LogLevel(level_str)
        except ValueError:
            # This should never happen due to validation, but just in case
            level = LogLevel.INFO
        
        # Get style for level
        style_class, _ = cls.LEVEL_STYLES.get(level, cls.LEVEL_STYLES[LogLevel.INFO])
        
        # Create formatted text segments
        segments = [
            ("", f"[{timestamp}] "),
            (f"class:{style_class}", f"{level_str}"),
            ("", f": {message}")
        ]
        
        if source:
            segments.append(("", f" ({source})"))
        
        return FormattedText(segments)
    
    @classmethod
    def format_log_entries(cls, entries: Deque[Dict[str, Any]]) -> FormattedText:
        """
        Format multiple log entries as FormattedText.
        
        Args:
            entries: Deque of log entry dictionaries
            
        Returns:
            FormattedText for display
        """
        result = []
        
        for entry in entries:
            formatted = cls.format_log_entry(entry)
            result.extend(formatted)
            result.append(("", "\n"))
        
        return FormattedText(result)


class StatusBar(Container):
    """
    Status bar and expandable log drawer for the OpenHCS TUI.

    Displays current status messages and operation indicators,
    with an expandable drawer for detailed log history.
    
    🔒 Clause 3: Declarative Primacy
    Separates presentation from data with LogFormatter.
    
    🔒 Clause 88: No Inferred Capabilities
    Explicitly declares event loop dependency.
    
    🔒 Clause 246: Statelessness Mandate
    All state is externalized in an immutable StatusBarState instance.
    """
    __slots__ = ('state', 'status_state', 'status_lock', 'status_label', 'log_drawer', 'container', 'event_loop')
    
    def __init__(self, state, event_loop: Optional[AbstractEventLoop] = None, max_log_entries: int = 1000):
        """
        Initialize the status bar.

        Args:
            state: The TUI state manager
            max_log_entries: Maximum number of log entries to keep in buffer
        """
        self.state = state
        
        # Initialize immutable state
        self.status_state = StatusBarState(
            log_buffer=deque(maxlen=max_log_entries)
        )
        
        # Store event loop
        self.event_loop = event_loop
        
        # Thread safety lock
        self.status_lock = Lock()

        # Create UI components
        self.status_label = self._create_status_label()
        self.log_drawer = self._create_log_drawer()

        # Create container
        self.container = HSplit([
            # Status bar (always visible)
            self.status_label,
            # Log drawer (conditionally visible)
            ConditionalContainer(
                self.log_drawer,
                filter=Condition(lambda: self.status_state.drawer_expanded)
            )
        ])

        # Register for events
        self.state.add_observer('operation_status_changed', self._on_operation_status_changed)
        self.state.add_observer('error', self._on_error)

        # Set up logging integration
        self._setup_logging()

    def _create_status_label(self) -> Label:
        """
        Create the status label component.

        Returns:
            A Label for displaying status messages
        """
        label = Label(
            text=lambda: self.status_state.status_message,
            dont_extend_height=True,
            style="class:status-bar"
        )

        # Add mouse handler for expanding/collapsing log drawer
        original_mouse_handler = label.mouse_handler

        def status_bar_mouse_handler(mouse_event):
            """Handle mouse events on the status bar."""
            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                # Use prompt_toolkit's create_background_task to avoid event loop issues
                get_app().create_background_task(self._toggle_drawer())
                return True
            return original_mouse_handler(mouse_event)

        label.mouse_handler = status_bar_mouse_handler
        return label

    def _create_log_drawer(self) -> Container:
        """
        Create the log drawer component.

        Returns:
            A Container for displaying log entries
        """
        # Create log content label with dynamic text
        log_content = Label(
            text=lambda: self._format_log_entries(),
            dont_extend_width=True
        )

        # Create scrollable frame
        return Frame(
            Box(
                log_content,
                padding=1,
                height=10,  # Fixed height for log drawer
                scrollbar=True
            ),
            title="Log History (Click to collapse)"
        )

    def _format_log_entries(self) -> FormattedText:
        """
        Format log entries for display.

        Returns:
            FormattedText for display
        """
        return LogFormatter.format_log_entries(self.status_state.log_buffer)

    async def _toggle_drawer(self) -> None:
        """Toggle the log drawer expansion state."""
        async with self.status_lock:
            self.status_state = self.status_state.with_drawer_expanded(
                not self.status_state.drawer_expanded
            )
            # Force UI refresh
            get_app().invalidate()

    async def _set_drawer_expanded(self, expanded: bool) -> None:
        """
        Set the log drawer expansion state.

        Args:
            expanded: Whether the drawer should be expanded
        """
        async with self.status_lock:
            self.status_state = self.status_state.with_drawer_expanded(expanded)
            # Force UI refresh
            get_app().invalidate()

    async def update_status(self, message: str, priority: Priority = Priority.INFO) -> None:
        """
        Update the status message with thread safety.

        Args:
            message: The status message to display
            priority: The priority level of the message
        """
        async with self.status_lock:
            self.status_state = self.status_state.with_status_message(message, priority)
            # Force UI refresh
            get_app().invalidate()

    async def add_log_entry(self, message: str, level: Union[str, LogLevel] = LogLevel.INFO, source: str = '') -> None:
        """
        Add a log entry to the buffer with thread safety.

        Args:
            message: The log message
            level: The log level (LogLevel enum or string)
            source: The source of the log message
            
        Raises:
            ValueError: If level string is not a valid LogLevel
        """
        # Convert string level to enum if needed
        if isinstance(level, str):
            level = LogLevel.from_string(level)
        
        async with self.status_lock:
            # Create timestamp
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            # Create log entry
            entry = {
                'timestamp': timestamp,
                'level': level.value,
                'message': message,
                'source': source
            }
            
            # Add to state
            self.status_state = self.status_state.with_log_entry(entry)

            # Update status message based on log level
            if level in (LogLevel.ERROR, LogLevel.CRITICAL):
                await self.update_status(
                    f"{STATUS_ICONS['error']} {message}",
                    Priority.ERROR
                )
            elif level == LogLevel.WARNING:
                await self.update_status(
                    f"{STATUS_ICONS['warning']} {message}",
                    Priority.WARNING
                )

            # Force UI refresh
            get_app().invalidate()

    def _setup_logging(self) -> None:
        """Set up integration with Python's logging system."""
        # Create custom log handler
        class TUILogHandler(logging.Handler):
            """Custom log handler that sends logs to the status bar."""
            def __init__(self, status_bar):
                super().__init__()
                self.status_bar = status_bar

            def emit(self, record):
                """Send log record to status bar."""
                # Format message
                message = self.format(record)

                # Map log level
                level_map = {
                    logging.DEBUG: LogLevel.DEBUG,
                    logging.INFO: LogLevel.INFO,
                    logging.WARNING: LogLevel.WARNING,
                    logging.ERROR: LogLevel.ERROR,
                    logging.CRITICAL: LogLevel.CRITICAL
                }
                
                # Get level, raising error for unknown levels
                if record.levelno not in level_map:
                    raise ValueError(f"Unknown log level number: {record.levelno}")
                
                level = level_map[record.levelno]

                # Get source
                source = record.name

                # Add log entry asynchronously
                get_app().create_background_task(
                    self.status_bar.add_log_entry(message, level, source)
                )

        # Create and configure handler
        handler = TUILogHandler(self)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)

        # Add handler to root logger
        logging.getLogger().addHandler(handler)

    async def _on_operation_status_changed(self, data) -> None:
        """
        Handle operation status change event.

        Args:
            data: Dictionary with operation and status
        """
        if not data or 'operation' not in data or 'status' not in data:
            return

        # Extract data
        operation = data['operation']
        status = data['status']
        error = data.get('error')

        # Get status icon
        icon = STATUS_ICONS.get(status, '?')

        # Create status message
        if status == 'running':
            message = f"{icon} {operation.capitalize()} in progress..."
            priority = Priority.INFO
        elif status == 'success':
            message = f"{icon} {operation.capitalize()} completed successfully"
            priority = Priority.INFO
        elif status == 'error':
            message = f"{icon} {operation.capitalize()} failed"
            if error:
                message += f": {error}"
            priority = Priority.ERROR
        elif status == 'pending':
            message = f"{icon} {operation.capitalize()} pending..."
            priority = Priority.INFO
        else:
            message = f"{icon} {operation.capitalize()}: {status}"
            priority = Priority.INFO

        # Update status
        await self.update_status(message, priority)

        # Add to log
        await self.add_log_entry(
            message,
            level='ERROR' if status == 'error' else 'INFO',
            source=f"Operation:{operation}"
        )

    async def _on_error(self, data) -> None:
        """
        Handle error event.

        Args:
            data: Dictionary with error information
        """
        if not data or 'message' not in data:
            return

        # Extract data
        message = data['message']
        source = data.get('source', '')
        details = data.get('details')

        # Update status
        await self.update_status(
            f"{STATUS_ICONS['error']} {message}",
            Priority.ERROR
        )

        # Add to log
        log_message = message
        if details:
            log_message += f"\nDetails: {details}"

        await self.add_log_entry(
            log_message,
            level='ERROR',
            source=source
        )

        # Expand drawer for errors
        await self._set_drawer_expanded(True)

    def __pt_container__(self) -> Container:
        """Return the container to render."""
        return self.container
```

### Usage Example

```python
# Create status bar
status_bar = StatusBar(state)

# Add to layout
layout = Layout(
    HSplit([
        # ... other components ...
        status_bar
    ])
)

# Update status
await status_bar.update_status("Compiling pipeline...", Priority.INFO)

# Add log entry
await status_bar.add_log_entry("Pipeline compilation started", "INFO", "Compiler")

# Handle operation status change
await status_bar._on_operation_status_changed({
    'operation': 'compile',
    'status': 'running'
})

# Handle error
await status_bar._on_error({
    'message': "Compilation failed: Invalid memory type",
    'source': "Compiler",
    'details': "Function 'stack_func' has incompatible memory types"
})
```

### Integration with TUI State

The status bar integrates with the TUI state through the observer pattern, listening for the following events:

1. `operation_status_changed` - Updates the status bar with operation status
2. `error` - Displays error messages and expands the log drawer

It also integrates with Python's logging system through a custom log handler, capturing log messages from OpenHCS components and displaying them in the log drawer.

The status bar maintains its own thread-safe state for status messages and log entries, ensuring consistent updates during concurrent operations.