"""
FileManagerBrowser Component for OpenHCS TUI - Declarative Architecture

Clean, declarative file browser that uses FileManager like os module.
No async constructors, no scattered task management, no defensive UI calls.
"""
import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Coroutine
import datetime
import logging
import time
from dataclasses import dataclass
from enum import Enum

from prompt_toolkit.mouse_events import MouseEventType

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import (
    Dimension,
    HSplit,
    VSplit,
    DynamicContainer,
    FormattedTextControl,
    Window,
    Container,
    ConditionalContainer,
    ScrollablePane
)
from prompt_toolkit.layout.controls import FormattedTextControl

from prompt_toolkit.filters import Condition
from prompt_toolkit.widgets import Box, Button, Label
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from openhcs.tui.utils.button_utils import action_button, nav_button
from openhcs.tui.constants.ui_constants import STYLES

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from openhcs.io.filemanager import FileManager
    from openhcs.constants.constants import Backend

try:
    from openhcs.io.exceptions import StorageResolutionError
except ImportError:
    class StorageResolutionError(Exception):
        pass

logger = logging.getLogger(__name__)

class SelectionMode(Enum):
    FILES_ONLY = "files"
    DIRECTORIES_ONLY = "directories"
    BOTH = "both"

@dataclass
class FileItem:
    """File or directory item."""
    name: str
    path: Path
    is_dir: bool
    size: Optional[int] = None
    mtime: Optional[float] = None
    
    @property
    def display_size(self) -> str:
        if self.size is None or self.is_dir:
            return ""
        
        size = self.size
        if size < 1024:
            return f"{size} B"
        
        for unit in ['KB', 'MB', 'GB', 'TB']:
            size /= 1024.0
            if size < 1024.0:
                return f"{size:.1f} {unit}"
        return f"{size:.1f} PB"
    
    @property
    def display_mtime(self) -> str:
        if self.mtime is None:
            return ""
        try:
            return datetime.datetime.fromtimestamp(self.mtime).strftime('%Y-%m-%d %H:%M')
        except (ValueError, OSError):
            return ""

class FileManagerBrowser:
    """Declarative file browser - no async constructor, clean state management."""

    def __init__(
        self,
        file_manager: "FileManager",
        backend: "Backend", 
        on_path_selected: Callable[[List[Path]], Coroutine[Any, Any, None]],
        on_cancel: Callable[[], Coroutine[Any, Any, None]],
        initial_path: Path,
        selection_mode: SelectionMode = SelectionMode.FILES_ONLY,
        allow_multiple: bool = False,
        show_hidden_files: bool = False,
        filter_extensions: Optional[List[str]] = None
    ):
        # Core dependencies - all required, no optionals
        self.file_manager = file_manager
        self.backend_str = backend.value
        self.on_path_selected = on_path_selected
        self.on_cancel = on_cancel
        
        # Configuration - all explicit
        self.selection_mode = selection_mode
        self.allow_multiple = allow_multiple
        self.show_hidden_files = show_hidden_files
        self.filter_extensions = [ext.lower() for ext in filter_extensions] if filter_extensions else None

        # State - single source of truth
        self.current_path = initial_path
        self.listing: List[FileItem] = []
        self.focused_index = 0
        self.selected_paths: set[Path] = set()  # Simpler than tracking indices
        self.error_message: Optional[str] = None
        self.loading = False

        # Click tracking for double-click detection
        self.last_click_time: float = 0
        self.last_click_index: int = -1

        # Task management
        self.load_task: Optional[asyncio.Task] = None
        
        # UI components
        self._build_ui()

    def _build_ui(self) -> None:
        """Build UI components once."""
        # Controls
        self.path_display = FormattedTextControl(text=self._get_path_text)
        # Note: item_list_control will be built dynamically as individual Windows


        
        # Buttons with dynamic width calculation
        self.select_btn = action_button("Select", handler=self._on_select)
        self.cancel_btn = action_button("Cancel", handler=self._on_cancel)
        self.up_btn = nav_button("Up", handler=self._on_up)
        self.refresh_btn = action_button("Refresh", handler=self._on_refresh)
        
        # Layout
        path_window = Window(self.path_display, height=1)
        
        error_label = Label(lambda: HTML(f"<ansired>{self.error_message}</ansired>") if self.error_message else "")
        error_box = ConditionalContainer(
            Box(error_label, padding_left=1, height=1),
            filter=Condition(lambda: bool(self.error_message))
        )
        
        # Create a simple text-based scrollable area
        # This should work better than trying to scroll individual Windows
        self.item_list_control = FormattedTextControl(
            text=self._get_item_list_text,
            focusable=True,
            key_bindings=self._create_key_bindings()  # Attach key bindings to the control
        )

        self.scrollable_pane = ScrollablePane(
            Window(content=self.item_list_control),
            height=Dimension(min=10, max=30),  # Reasonable height with scrolling capability
            show_scrollbar=True,  # Ensure scrollbar is visible
            display_arrows=True,  # Show up/down arrows on scrollbar
        )

        # Mouse wheel will be handled in FormattedTextControl mouse handlers

        # Let Dialog handle padding - don't add extra Box
        file_area = self.scrollable_pane
        
        buttons = VSplit([
            VSplit([self.up_btn, self.refresh_btn], padding=1),
            Window(width=Dimension(weight=1)),
            VSplit([self.select_btn, self.cancel_btn], padding=1)
        ])
        
        self.container = HSplit([
            path_window,
            file_area,
            error_box,
            buttons,
        ])

    # Removed _setup_scrollbar_click_handler - ScrollablePane handles this natively

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        
        @kb.add('up')
        def _(event):
            logger.info("Up arrow key pressed!")
            self._move_focus(-1)

        @kb.add('down')
        def _(event):
            logger.info("Down arrow key pressed!")
            self._move_focus(1)
        
        @kb.add('pageup')
        def _(event):
            logger.info("Page Up key pressed!")
            self._move_focus(-10)

        @kb.add('pagedown')
        def _(event):
            logger.info("Page Down key pressed!")
            self._move_focus(10)
        
        @kb.add('home')
        def _(event): self._set_focus(0)
        
        @kb.add('end')
        def _(event): self._set_focus(len(self.listing) - 1)
        
        @kb.add('enter')
        def _(event): self._on_enter_key()

        @kb.add('space')
        def _(event):
            if self.allow_multiple:
                self._toggle_selection()
        
        @kb.add('escape')
        def _(event): self._on_cancel()
        
        @kb.add('backspace')
        def _(event): self._on_up()
        
        @kb.add('f5')
        def _(event): self._on_refresh()

        # Mouse wheel support for smooth scrolling
        @kb.add('<scroll-up>')
        def _(event):
            self._move_focus(-3)  # Scroll 3 lines up

        @kb.add('<scroll-down>')
        def _(event):
            self._move_focus(3)   # Scroll 3 lines down

        # Test with regular keys to verify key bindings work
        @kb.add('j')
        def _(event):
            logger.info("🔤 J key pressed - moving down!")
            self._move_focus(1)

        @kb.add('k')
        def _(event):
            logger.info("🔤 K key pressed - moving up!")
            self._move_focus(-1)

        return kb

    # State management - clean, synchronous
    def _move_focus(self, delta: int) -> None:
        if not self.listing:
            return
        new_index = max(0, min(len(self.listing) - 1, self.focused_index + delta))
        self._set_focus(new_index)

    def _set_focus(self, index: int) -> None:
        if 0 <= index < len(self.listing):
            self.focused_index = index
            self._ensure_focused_visible()
            self._update_ui()

    def _ensure_focused_visible(self) -> None:
        """Ensure the focused item is visible by making FormattedTextControl report cursor position."""
        try:
            # Make the FormattedTextControl show a cursor at the focused line
            # This will trigger ScrollablePane's automatic scrolling
            from prompt_toolkit.data_structures import Point

            # Validate cursor position against current content
            # When listing is empty, we have 1 line of content ("Loading..." or "(empty directory)")
            # When listing has items, we have len(self.listing) lines (plus newlines, but cursor should be on item lines)
            if not self.listing:
                # Empty/loading state - cursor should be at line 0
                cursor_y = 0
            else:
                # Ensure focused_index is within bounds of the listing
                cursor_y = max(0, min(self.focused_index, len(self.listing) - 1))

            cursor_pos = Point(x=0, y=cursor_y)

            # Override the get_cursor_position method of FormattedTextControl
            def get_cursor_position():
                return cursor_pos

            # Monkey patch the method
            self.item_list_control.get_cursor_position = get_cursor_position

            # Also make the control show cursor
            self.item_list_control.show_cursor = True

        except Exception as e:
            pass  # Silently handle cursor positioning errors

    # Removed broken mouse wheel handler - ScrollablePane doesn't work that way

    def _toggle_selection(self) -> None:
        if not self._valid_focus():
            return

        item = self.listing[self.focused_index]
        if not self._can_select(item):
            return

        if item.path in self.selected_paths:
            self.selected_paths.remove(item.path)
        else:
            self.selected_paths.add(item.path)

        self._update_ui()

    def _can_select(self, item: FileItem) -> bool:
        if self.selection_mode == SelectionMode.FILES_ONLY:
            return not item.is_dir
        elif self.selection_mode == SelectionMode.DIRECTORIES_ONLY:
            return item.is_dir
        else:
            return True

    def _valid_focus(self) -> bool:
        return 0 <= self.focused_index < len(self.listing)

    def _clear_error(self) -> None:
        self.error_message = None

    def _set_error(self, msg: str) -> None:
        self.error_message = msg
        self._update_ui()

    def _update_ui(self) -> None:
        """Single UI update point."""
        try:
            get_app().invalidate()
        except Exception:
            pass





    # Event handlers - all sync, delegate async work
    def _on_select(self) -> None:
        self._run_async(self._handle_selection())

    def _on_cancel(self) -> None:
        self._run_async(self.on_cancel())

    def _on_up(self) -> None:
        parent = self.current_path.parent
        if parent != self.current_path:
            self._navigate_to(parent)

    def _on_refresh(self) -> None:
        self._navigate_to(self.current_path)

    def _on_activate(self) -> None:
        if not self._valid_focus():
            return

        item = self.listing[self.focused_index]

        if self.allow_multiple:
            # In multi-selection mode, clicking only toggles selection
            # Navigation requires double-click or Enter key
            self._toggle_selection()
        else:
            # In single-selection mode, clicking navigates or selects
            if item.is_dir:
                self._navigate_to(item.path)
            elif self._can_select(item):
                self._run_async(self._handle_selection())

    def _on_enter_key(self) -> None:
        """Handle Enter key - navigate into directories or finalize selection."""
        if not self._valid_focus():
            return

        item = self.listing[self.focused_index]

        # Enter key always navigates into directories (regardless of selection mode)
        if item.is_dir and self.selection_mode != SelectionMode.FILES_ONLY:
            self._navigate_to(item.path)
        else:
            # For files, treat as selection
            self._run_async(self._handle_selection())

    def _run_async(self, coro: Coroutine) -> None:
        """Centralized async task management."""
        get_app().create_background_task(coro)

    def _navigate_to(self, path: Path) -> None:
        """Navigate to path - cancel previous load, start new one."""
        if self.load_task and not self.load_task.done():
            self.load_task.cancel()

        self.current_path = path
        self.focused_index = 0
        self.selected_paths.clear()
        self._clear_error()

        # CRITICAL: Reset cursor position immediately to prevent race condition
        # This ensures cursor position is valid for the current listing state
        self._ensure_focused_visible()

        # Create and schedule the task - no need for _run_async since create_task already schedules it
        self.load_task = asyncio.create_task(self._load_directory())

    # Async operations - clean, focused
    async def _load_directory(self) -> None:
        """Load directory contents."""
        self.loading = True
        self._update_ui()
        
        try:
            loop = asyncio.get_event_loop()
            
            # Get listing
            raw_items = await loop.run_in_executor(
                None, self.file_manager.list_dir, self.current_path, self.backend_str
            )
            
            # Process in parallel
            tasks = [self._process_item(name) for name in raw_items if self._should_include(name)]
            items = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            self.listing = [item for item in items if isinstance(item, FileItem)]
            self.listing.sort(key=lambda x: (not x.is_dir, x.name.lower()))
            
        except StorageResolutionError as e:
            self._set_error(f"Storage error: {e}")
            self.listing = []
        except Exception as e:
            self._set_error(f"Error: {e}")
            self.listing = []
        finally:
            self.loading = False
            self._update_ui()

    def _should_include(self, name: str) -> bool:
        """Check if item should be included in listing."""
        if not self.show_hidden_files and name.startswith('.'):
            return False
        return True

    async def _process_item(self, name: str) -> FileItem:
        """Process single directory item."""
        item_path = self.current_path / name
        loop = asyncio.get_event_loop()
        
        # Check type
        is_dir = await loop.run_in_executor(
            None, self.file_manager.is_dir, item_path, self.backend_str
        )
        
        # Apply extension filter
        if self.filter_extensions and not is_dir:
            if not any(name.lower().endswith(ext) for ext in self.filter_extensions):
                raise ValueError("Filtered out by extension")
        
        return FileItem(name=name, path=item_path, is_dir=is_dir)

    async def _handle_selection(self) -> None:
        """Handle final selection."""
        try:
            if self.allow_multiple:
                paths = list(self.selected_paths) if self.selected_paths else []
            else:
                paths = self._get_single_selection()
            
            if not paths:
                self._set_error("No valid selection")
                return
                
            await self.on_path_selected(paths)
            
        except Exception as e:
            self._set_error(f"Selection error: {e}")

    def _get_single_selection(self) -> List[Path]:
        """Get single selection paths."""
        # Empty directory case
        if not self.listing and self.selection_mode == SelectionMode.DIRECTORIES_ONLY:
            return [self.current_path]
        
        # Focused item
        if self._valid_focus():
            item = self.listing[self.focused_index]
            if self._can_select(item):
                return [item.path]
            elif self.selection_mode == SelectionMode.DIRECTORIES_ONLY:
                return [self.current_path]
        
        # Fallback for directory selection
        if self.selection_mode == SelectionMode.DIRECTORIES_ONLY:
            return [self.current_path]
        
        return []

    # UI builders
    def _get_path_text(self):
        status = " (loading...)" if self.loading else ""
        return HTML(f"<b>Path:</b> {self.current_path}{status}")

    def _get_item_list_text(self):
        """Generate formatted text with mouse handlers for FormattedTextControl."""
        if not self.listing:
            # Return properly formatted text tuples, not plain strings
            message = "Loading..." if self.loading else "(empty directory)"
            return [("class:file-browser.empty", message)]

        lines = []
        max_name_width = min(60, max(20, max(len(item.name) for item in self.listing) + 6))

        for i, item in enumerate(self.listing):
            is_selected = item.path in self.selected_paths
            is_focused = i == self.focused_index

            # Build display text
            prefix = ""
            if self.allow_multiple:
                prefix = "[x] " if is_selected else "[ ] "

            icon = "📁" if item.is_dir else "📄"
            name_part = f"{prefix}{icon} {item.name}"

            display = f"{name_part:<{max_name_width}}"
            if not item.is_dir:
                display += f"{item.display_size:>10}  "
            else:
                display += f"{'':>10}  "
            display += item.display_mtime

            # Add selection/focus styling
            style = ""
            if is_focused:
                style = "reverse"
                display = f"> {display}"
            else:
                display = f"  {display}"

            # Create mouse handler for this line
            def make_handler(index):
                def handler(mouse_event):
                    # Handle mouse wheel events FIRST
                    if mouse_event.event_type == MouseEventType.SCROLL_UP:
                        self._move_focus(-3)  # Scroll 3 lines up
                        return True
                    elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                        self._move_focus(3)   # Scroll 3 lines down
                        return True

                    # Handle regular mouse clicks
                    elif mouse_event.event_type == MouseEventType.MOUSE_UP:
                        current_time = time.time()

                        # Check for double-click
                        is_double_click = (
                            index == self.last_click_index and
                            current_time - self.last_click_time < 0.5
                        )

                        # Update click tracking
                        self.last_click_time = current_time
                        self.last_click_index = index

                        # Handle click based on position and type
                        x_pos = mouse_event.position.x

                        if self.allow_multiple and x_pos <= 4:  # Checkbox area
                            self._set_focus(index)
                            self._toggle_selection()
                        elif is_double_click:
                            # Double click - navigate or select
                            item = self.listing[index]
                            if item.is_dir:
                                self._navigate_to(item.path)
                            else:
                                self._set_focus(index)
                                self._on_activate()
                        else:
                            # Single click - just focus
                            self._set_focus(index)

                        return True  # Event handled
                    return False  # Event not handled
                return handler

            # Add line with mouse handler - format: (style, text, mouse_handler)
            lines.append((style, display, make_handler(i)))
            if i < len(self.listing) - 1:  # Add newline except for last item
                lines.append(("", "\n"))

        return lines

    # Removed _handle_list_mouse_event - using FormattedTextControl's built-in mouse support

    # Manual scroll methods removed - ScrollablePane handles scrolling automatically

    # Removed _build_item_list() and _item_clicked() - using FormattedTextControl approach

    # Public interface
    def start_load(self) -> None:
        """Start initial directory load - call this after UI is ready."""
        self._navigate_to(self.current_path)

    def get_key_bindings(self) -> KeyBindings:
        return self.item_list_control.key_bindings

    def __pt_container__(self) -> Container:
        return self.container
