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
            focusable=True
        )
        self.item_list_control.mouse_handler = self._handle_list_mouse_event

        self.scrollable_pane = ScrollablePane(
            Window(content=self.item_list_control),
            height=Dimension(min=10, max=30),  # Reasonable height with scrolling capability
            show_scrollbar=True,  # Ensure scrollbar is visible
            display_arrows=True,  # Show up/down arrows on scrollbar
        )

        # Patch the existing scrollbar window for click-to-jump functionality
        self._setup_scrollbar_click_handler()

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
        
        self.container.key_bindings = self._create_key_bindings()

    def _setup_scrollbar_click_handler(self) -> None:
        """Setup click-to-jump functionality on the existing scrollbar."""
        try:
            # Access the internal scrollbar window
            sb_window = self.scrollable_pane._scrollbar_window

            def scrollbar_click(mouse_event):
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    info = self.scrollable_pane.render_info
                    if not info:
                        return False

                    track_height = info.window_height
                    click_row = mouse_event.position.y
                    max_scroll = max(0, len(self.listing) - track_height)

                    # Proportional jump
                    self.scrollable_pane.vertical_scroll = int(
                        click_row / track_height * max_scroll
                    )
                    get_app().invalidate()
                    return True
                return False

            sb_window.content.mouse_handler = scrollbar_click
        except AttributeError:
            # Fallback if _scrollbar_window doesn't exist
            logger.warning("Could not access scrollbar window for click handling")

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

        # ScrollablePane will handle scrolling automatically with individual Windows
        # No manual scroll manipulation needed

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
            self._update_ui()

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
        """Get formatted text for the entire item list."""
        if not self.listing:
            return "Loading..." if self.loading else "(empty directory)"

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
            if is_focused:
                display = f"> {display}"
            else:
                display = f"  {display}"

            lines.append(display)

        return "\n".join(lines)

    def _handle_list_mouse_event(self, mouse_event):
        """Handle mouse events on the item list."""
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            # Calculate which item was clicked based on Y position
            if self.listing and mouse_event.position.y < len(self.listing):
                clicked_index = mouse_event.position.y
                current_time = time.time()

                # Check for double-click (within 500ms of same item)
                is_double_click = (
                    clicked_index == self.last_click_index and
                    current_time - self.last_click_time < 0.5
                )

                # Update click tracking
                self.last_click_time = current_time
                self.last_click_index = clicked_index

                # Determine click area based on X position
                x_pos = mouse_event.position.x

                if self.allow_multiple and x_pos <= 4:  # Checkbox area "[x] " or "[ ] "
                    # Single click on checkbox - toggle selection
                    self._set_focus(clicked_index)
                    self._toggle_selection()
                elif is_double_click:
                    # Double click on folder name - navigate into folder
                    item = self.listing[clicked_index]
                    if item.is_dir:
                        self._navigate_to(item.path)
                    else:
                        # Double click on file - select and close dialog
                        self._set_focus(clicked_index)
                        self._on_activate()
                else:
                    # Single click on folder/file name - just focus
                    self._set_focus(clicked_index)

                return True
        return False

    # Manual scroll methods removed - ScrollablePane handles scrolling automatically

    def _build_item_list(self) -> Container:
        if not self.listing:
            return Label("Loading..." if self.loading else "(empty directory)")

        max_name_width = min(60, max(20, max(len(item.name) for item in self.listing) + 6))

        items = []
        for i, item in enumerate(self.listing):
            is_selected = item.path in self.selected_paths

            # Build display text (exact same logic as before)
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

            # Create control with mouse handler (FramedButton pattern)
            control = FormattedTextControl(display, focusable=True)

            # Closure capture fix
            def make_handler(index):
                def handler(mouse_event):
                    if mouse_event.event_type == MouseEventType.MOUSE_UP:
                        self._item_clicked(index)
                        return True
                    return False
                return handler

            control.mouse_handler = make_handler(i)

            # Direct Window (no duck typing)
            item_window = Window(
                control,
                style="reverse" if is_selected else "",
                height=1
            )

            items.append(item_window)

        return HSplit(items)

    def _item_clicked(self, index: int) -> None:
        """Handle item click."""
        self._set_focus(index)
        self._on_activate()

    # Public interface
    def start_load(self) -> None:
        """Start initial directory load - call this after UI is ready."""
        self._navigate_to(self.current_path)

    def get_key_bindings(self) -> KeyBindings:
        return self.container.key_bindings

    def __pt_container__(self) -> Container:
        return self.container
