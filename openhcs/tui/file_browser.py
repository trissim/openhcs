"""
FileManagerBrowser Component for OpenHCS TUI.

This module provides an interactive file browser widget that uses the
OpenHCS FileManager to navigate and select files/directories across
different storage backends.
"""
import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Coroutine # Added Coroutine
import datetime # For formatting mtime

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import (
    Dimension,
    HSplit,
    VSplit,
    DynamicContainer,
    FormattedTextControl,
    ScrollablePane,
    Window,
    Container,
    ConditionalContainer # Added ConditionalContainer
)
from prompt_toolkit.filters import Condition # Added Condition
from prompt_toolkit.widgets import Box, Button, Label, TextArea
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML, FormattedText, to_formatted_text

from openhcs.io.filemanager import FileManager
from openhcs.constants.constants import Backend
# from openhcs.tui.components import InteractiveListItem # Not used in current basic version

# For logging
import logging
logger = logging.getLogger(__name__)

# Define SafeButton locally to avoid circular imports
class SafeButton(Button):
    """Safe wrapper around Button that handles formatting errors."""
    
    def __init__(self, text="", handler=None, width=None, **kwargs):
        # Sanitize text before passing to parent
        if text is not None:
            text = str(text).replace('{', '{{').replace('}', '}}').replace(':', ' ')
        super().__init__(text=text, handler=handler, width=width, **kwargs)
    
    def _get_text_fragments(self):
        """Safe version that handles formatting errors gracefully."""
        try:
            return super()._get_text_fragments()
        except (ValueError, TypeError, AttributeError):
            # Fallback to simple text formatting without centering
            text = str(self.text) if self.text is not None else ""
            safe_text = text.replace('{', '{{').replace('}', '}}')
            return [("class:button", f" {safe_text} ")]

class FileManagerBrowser:
    """
    An interactive file browser TUI component.

    This component allows users to navigate a filesystem (local or remote via
    FileManager backends) and select a file or directory.
    """

    def __init__(
        self,
        file_manager: FileManager,
        on_path_selected: Callable[[List[Path]], Coroutine[Any, Any, None]], # Expects a List of Paths
        on_cancel: Callable[[], Coroutine[Any, Any, None]],
        initial_path: Optional[Union[str, Path]] = None,
        backend: Optional[Backend] = None, # Backend for FileManager operations
        select_files: bool = True, # True to select files, False to select directories
        select_multiple: bool = False, # TODO: Implement multi-select later
        show_hidden_files: bool = False,
        filter_extensions: Optional[List[str]] = None # e.g., [".h5", ".zarr"]
    ):
        self.file_manager = file_manager
        self.on_path_selected = on_path_selected
        self.on_cancel = on_cancel
        self.backend = backend
        self.select_files = select_files
        self.select_multiple = select_multiple # Now to be implemented
        self.show_hidden_files = show_hidden_files
        self.filter_extensions = [ext.lower() for ext in filter_extensions] if filter_extensions else None

        self.current_path: Path = self._resolve_initial_path(initial_path)
        self.current_listing: List[Dict[str, Any]] = [] # List of {'name': str, 'path': Path, 'is_dir': bool}
        self.focused_item_index: int = 0 # For keyboard navigation focus
        self.selected_item_indices: List[int] = [] # For multi-selection
        self.error_message: Optional[str] = None

        # UI Components
        self.path_display = FormattedTextControl(text=self._get_path_display_text)
        self.item_list_container = DynamicContainer(lambda: self._build_item_list_ui())

        self.ok_button = SafeButton("Select", handler=lambda: get_app().create_background_task(self._handle_ok()))
        self.cancel_button = SafeButton("Cancel", handler=lambda: get_app().create_background_task(self._handle_cancel()))
        self.up_button = SafeButton("Up ..", handler=lambda: get_app().create_background_task(self._handle_up_directory()))
        self.refresh_button = SafeButton("Refresh", handler=lambda: get_app().create_background_task(self._handle_refresh()))
        # TODO: toggle hidden files button

        self.container = self._build_ui()
        # Initial load is now async, schedule it
        get_app().create_background_task(self._load_directory_listing())


    def get_initial_focus_target(self) -> Optional[Button]:
        """Returns the element that should receive initial focus."""
        # Try to focus the first item if list is not empty,
        # otherwise fallback to a button like 'Up' or 'Select'.
        # For now, let's consistently focus the 'Up' button as it's always present.
        return self.up_button

    def _resolve_initial_path(self, initial_path: Optional[Union[str, Path]]) -> Path:
        """Resolves the initial path to a valid, absolute directory."""
        path_to_check = Path.cwd() # Default to CWD
        if initial_path:
            path_to_check = Path(initial_path).resolve()

        # Ensure it's a directory; if not, go to parent
        # This requires a FileManager call, which might be slow for remote.
        # For now, assume initial_path is generally valid or local.
        # A more robust solution would involve an async check here.
        try:
            if self.file_manager.exists(str(path_to_check), backend=self.backend):
                if not self.file_manager.isdir(str(path_to_check), backend=self.backend):
                    path_to_check = path_to_check.parent
            else: # Path doesn't exist, default to CWD or root
                path_to_check = Path.cwd() # Or perhaps file_manager.get_root_for_backend(self.backend)
        except Exception as e:
            logger.warning(f"Error resolving initial path '{initial_path}': {e}. Defaulting to CWD.")
            path_to_check = Path.cwd()
        return path_to_check.resolve()

    def _get_path_display_text(self) -> FormattedText:
        """Returns the formatted text for the current path display."""
        return HTML(f"<b>Path:</b> {self.current_path}")

    def _build_ui(self) -> Container:
        """Builds the main container for the file browser."""
        path_window = Window(content=self.path_display, height=1, style="class:filebrowser.path")

        action_buttons = VSplit([
            self.ok_button,
            self.cancel_button,
        ], padding=1, align="RIGHT")

        nav_buttons = VSplit([
            self.up_button,
            self.refresh_button,
            # TODO: Add toggle hidden button here
        ], padding=1, align="LEFT")

        bottom_bar = VSplit([
            nav_buttons,
            # This will push nav_buttons left and action_buttons right
            Box(style="class:filebrowser.bottombar.spacer", width=Dimension(weight=1)),
            action_buttons
        ])


        # Error display area
        self.error_label = Label(lambda: HTML(f"<ansired>{self.error_message}</ansired>") if self.error_message else "")
        error_container = ConditionalContainer(
            Box(self.error_label, padding_left=1, height=1),
            filter=Condition(lambda: bool(self.error_message))
        )

        return HSplit([
            path_window,
            Window(height=1, char='─', style="class:filebrowser.separator"), # Separator
            ScrollablePane(self.item_list_container),
            error_container,
            Window(height=1, char='─', style="class:filebrowser.separator"), # Separator
            bottom_bar, # Use the new VSplit for buttons
        ])

    def _format_size(self, size_bytes: Optional[int]) -> str:
        """Formats size in bytes to a human-readable string."""
        if size_bytes is None:
            return ""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        for unit in ['KB', 'MB', 'GB', 'TB']:
            size_bytes /= 1024.0
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
        return f"{size_bytes:.1f} PB" # Should be rare

    def _format_mtime(self, timestamp: Optional[float]) -> str:
        """Formats a Unix timestamp to a human-readable date string."""
        if timestamp is None:
            return ""
        try:
            return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
        except Exception:
            return ""


    def _build_item_list_ui(self) -> Container:
        """Builds the container for the list of files and directories."""
        if not self.current_listing:
            return Label(" (empty directory) " if not self.error_message else "")

        items_ui = []
        max_name_width = 0
        if self.current_listing: # Calculate max name width for alignment
            # Consider only a portion if list is very long, for performance
            sample_size = min(len(self.current_listing), 50)
            max_name_width = max(len(item['name']) for item in self.current_listing[:sample_size]) + 4 # +4 for icon and spaces
            max_name_width = max(20, min(max_name_width, 60)) # Clamp width


        for i, item_info in enumerate(self.current_listing):
            is_focused = (i == self.focused_item_index) # For focus highlight
            is_selected_for_multi = i in self.selected_item_indices and self.select_multiple

            prefix = " "
            if self.select_multiple:
                prefix = "[x] " if is_selected_for_multi else "[ ] "

            icon = "📁" if item_info['is_dir'] else "📄"
            name_part = f"{prefix}{icon} {item_info['name']}" # Add selection prefix

            size_part = self._format_size(item_info.get('size')) if not item_info['is_dir'] else ""
            mtime_part = self._format_mtime(item_info.get('mtime'))

            # Construct display text with fixed-width columns for alignment (using spaces)
            # This is a basic way; for perfect columns, FormattedText with (width, text) tuples is better.
            # For simplicity with Button, we'll use string formatting.
            # Max width for name can be estimated or fixed.

            # Using FormattedText for better control with Button text
            text_fragments = []
            text_fragments.append(('', f"{name_part:<{max_name_width}}")) # Left align name
            if not item_info['is_dir']:
                 text_fragments.append(('', f"{size_part:>10}  ")) # Right align size
            else:
                 text_fragments.append(('', f"{'':>10}  ")) # Spacer for dirs
            text_fragments.append(('', f"{mtime_part}"))

            display_formatted_text = to_formatted_text(HTML("".join(t[1] for t in text_fragments)))


            item_button = SafeButton(text=display_formatted_text, # Use FormattedText
                handler=lambda idx=i: get_app().create_background_task(self._handle_item_activated(idx)),
                width=None
            )
            if is_focused: # Style based on focus for navigation
                item_button.style = "class:filebrowser.item.focused"
            elif is_selected_for_multi: # Different style for multi-selected items
                item_button.style = "class:filebrowser.item.selected"
            else:
                item_button.style = "class:filebrowser.item"

            items_ui.append(item_button)

        return HSplit(items_ui)

    async def _load_directory_listing(self): # Made async
        """Loads and displays the content of the current_path."""
        self.error_message = None
        try:
            # Ensure file_manager methods are awaitable if they do I/O
            # For now, assuming they are synchronous or wrapped by run_in_executor if needed by FileManager
            # If FileManager methods become async, await them here.
            # Example: raw_items = await self.file_manager.listdir(...)

            # To make this non-blocking for TUI, run sync FM calls in executor
            loop = asyncio.get_event_loop()
            raw_items = await loop.run_in_executor(None, self.file_manager.listdir, str(self.current_path), self.backend)

            processed_listing = []
            for item_name in raw_items:
                if not self.show_hidden_files and item_name.startswith('.'):
                    continue

                item_path = self.current_path / item_name

                # Fetch stats for each item
                try:
                    stats = await loop.run_in_executor(None, self.file_manager.stat, str(item_path), self.backend)
                    is_dir = stats.get('is_dir', False)
                    size = stats.get('size')
                    mtime = stats.get('mtime')
                except Exception as stat_exc:
                    logger.warning(f"Could not stat {item_path}: {stat_exc}")
                    # Fallback if stat fails, try isdir at least
                    try:
                        is_dir = await loop.run_in_executor(None, self.file_manager.isdir, str(item_path), self.backend)
                    except Exception:
                        is_dir = False # Assume not a dir if isdir also fails
                    size = None
                    mtime = None

                if self.filter_extensions and not is_dir:
                    if not any(item_name.lower().endswith(ext) for ext in self.filter_extensions):
                        continue

                processed_listing.append({
                    'name': item_name,
                    'path': item_path,
                    'is_dir': is_dir,
                    'size': size,
                    'mtime': mtime
                })

            self.current_listing = sorted(
                processed_listing,
                key=lambda x: (not x['is_dir'], x['name'].lower())
            )
            self.focused_item_index = 0
            self.selected_item_indices = [] # Clear previous multi-selections on new dir load

        except Exception as e:
            logger.error(f"Error listing directory {self.current_path}: {e}", exc_info=True)
            self.error_message = f"Error: {e}"
            self.current_listing = []

        get_app().invalidate()

    async def _handle_item_activated(self, index: int):
        """Handles activation (click or Enter) of a list item."""
        if not self._is_valid_index(index):
            return

        self.focused_item_index = index
        selected_item_info = self.current_listing[index]

        if self.select_multiple:
            await self._handle_multiple_selection_activation(index, selected_item_info)
        else:
            await self._handle_single_selection_activation(selected_item_info)

        get_app().invalidate()

    def _is_valid_index(self, index: int) -> bool:
        """Check if index is valid for current listing."""
        return 0 <= index < len(self.current_listing)

    async def _handle_multiple_selection_activation(self, index: int, item_info: dict):
        """Handle activation in multiple selection mode."""
        await self._toggle_item_selection(index, item_info)
        await self._handle_directory_navigation_in_multi_select(item_info)

    async def _toggle_item_selection(self, index: int, item_info: dict):
        """Toggle selection state for an item."""
        if index in self.selected_item_indices:
            self.selected_item_indices.remove(index)
        else:
            self._add_item_to_selection_if_valid(index, item_info)

    def _add_item_to_selection_if_valid(self, index: int, item_info: dict):
        """Add item to selection if it matches selection criteria."""
        if self.select_files and not item_info['is_dir']:
            self.selected_item_indices.append(index)
        elif not self.select_files and item_info['is_dir']:
            self.selected_item_indices.append(index)
        # Silently ignore invalid selections (files in dir mode, etc.)

    async def _handle_directory_navigation_in_multi_select(self, item_info: dict):
        """Handle directory navigation in multi-select mode."""
        should_navigate = (
            item_info['is_dir'] and
            (self.select_files or (not self.select_files and not self.select_multiple))
        )

        if should_navigate:
            self.current_path = item_info['path']
            await self._load_directory_listing()

    async def _handle_single_selection_activation(self, item_info: dict):
        """Handle activation in single selection mode."""
        if item_info['is_dir']:
            await self._navigate_to_directory(item_info['path'])
        elif self.select_files:
            await self._handle_ok()

    async def _navigate_to_directory(self, path: Path):
        """Navigate to a directory."""
        self.current_path = path
        await self._load_directory_listing()


    async def _handle_ok(self):
        """Handles the OK/Select button press."""
        if self.select_multiple:
            selected_paths = await self._handle_multiple_selection()
        else:
            selected_paths = await self._handle_single_selection()

        if selected_paths:
            self.error_message = None
            await self.on_path_selected(selected_paths)

    async def _handle_multiple_selection(self) -> List[Path]:
        """Handle multiple selection mode."""
        if not self.selected_item_indices:
            self._set_error("No items selected.")
            return []

        selected_paths = self._collect_selected_paths()
        if not selected_paths:
            self._set_error("Selected items do not match selection type (files/directories).")
            return []

        return selected_paths

    def _collect_selected_paths(self) -> List[Path]:
        """Collect paths from selected indices."""
        selected_paths = []
        for index in self.selected_item_indices:
            if 0 <= index < len(self.current_listing):
                item_info = self.current_listing[index]
                if self._is_valid_selection(item_info):
                    selected_paths.append(item_info['path'])
        return selected_paths

    def _is_valid_selection(self, item_info: dict) -> bool:
        """Check if item matches selection criteria."""
        if self.select_files and not item_info['is_dir']:
            return True
        elif not self.select_files and item_info['is_dir']:
            return True
        return False

    async def _handle_single_selection(self) -> List[Path]:
        """Handle single selection mode."""
        # Empty directory case
        if not self.current_listing and not self.select_files:
            return [self.current_path]

        # Valid focused item case
        if 0 <= self.focused_item_index < len(self.current_listing):
            return self._handle_focused_item_selection()

        # No focused item in directory selection mode
        if not self.select_files:
            return [self.current_path]

        self._set_error("No valid item focused for selection.")
        return []

    def _handle_focused_item_selection(self) -> List[Path]:
        """Handle selection of the currently focused item."""
        selected_item_info = self.current_listing[self.focused_item_index]

        if self.select_files:
            return self._handle_file_selection(selected_item_info)
        else:
            return self._handle_directory_selection(selected_item_info)

    def _handle_file_selection(self, item_info: dict) -> List[Path]:
        """Handle file selection logic."""
        if not item_info['is_dir']:
            return [item_info['path']]
        else:
            self._set_error("Please select a file, not a directory.")
            return []

    def _handle_directory_selection(self, item_info: dict) -> List[Path]:
        """Handle directory selection logic."""
        if item_info['is_dir']:
            return [item_info['path']]
        else:
            # Select current_path if a file is highlighted in dir selection mode
            return [self.current_path]

    def _set_error(self, message: str) -> None:
        """Set error message and invalidate display."""
        self.error_message = message
        get_app().invalidate()


    async def _handle_cancel(self):
        """Handles the Cancel button press."""
        await self.on_cancel()

    async def _handle_up_directory(self):
        """Handles the 'Up' button to navigate to the parent directory."""
        parent_path = self.current_path.parent
        if parent_path != self.current_path:
            self.current_path = parent_path
            await self._load_directory_listing() # Await async load
        get_app().invalidate()

    async def _handle_refresh(self):
        """Handles the 'Refresh' button."""
        await self._load_directory_listing() # Await async load
        get_app().invalidate()

    def get_key_bindings(self) -> KeyBindings:
        """Returns key bindings for the file browser."""
        kb = KeyBindings()

        @kb.add('up')
        def _move_focus_up(event):
            if self.focused_item_index > 0:
                self.focused_item_index -= 1
                get_app().invalidate()

        @kb.add('down')
        def _move_focus_down(event):
            if self.focused_item_index < len(self.current_listing) - 1:
                self.focused_item_index += 1
                get_app().invalidate()

        @kb.add('pageup')
        def _page_up(event):
            self.focused_item_index = max(0, self.focused_item_index - 10)
            get_app().invalidate()

        @kb.add('pagedown')
        def _page_down(event):
            self.focused_item_index = min(len(self.current_listing) - 1, self.focused_item_index + 10)
            get_app().invalidate()

        @kb.add('space')
        def _toggle_selection_space(event):
            if self.select_multiple and self.current_listing and \
               0 <= self.focused_item_index < len(self.current_listing):
                event.app.create_background_task(self._handle_item_activated(self.focused_item_index))


        @kb.add('enter')
        def _activate_item_enter(event):
            # In multi-select mode, Enter triggers OK action with current selections
            if self.select_multiple:
                event.app.create_background_task(self._handle_ok())
            # In single-select mode, Enter activates the focused item (navigates dir or selects file)
            elif self.current_listing and 0 <= self.focused_item_index < len(self.current_listing):
                event.app.create_background_task(self._handle_item_activated(self.focused_item_index))
            # Enter on empty dir in single dir selection mode
            elif not self.select_files and not self.current_listing:
                event.app.create_background_task(self._handle_ok())


        @kb.add('escape')
        def _cancel_dialog(event):
            event.app.create_background_task(self._handle_cancel())

        @kb.add('backspace')
        def _go_up_directory_key(event):
            """Handles backspace key to navigate up a directory."""
            event.app.create_background_task(self._handle_up_directory())

        # TODO: Add keybindings for home/end, refresh (e.g. F5)

        return kb

    def __pt_container__(self):
        return self.container

if __name__ == '__main__':
    # Example Usage (requires a running prompt_toolkit application)
    # This is a placeholder for how it might be tested or integrated.
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout.layout import Layout
    from openhcs.io.filemanager import FileManager
    from openhcs.constants.constants import Backend

    logging.basicConfig(level=logging.DEBUG)

    async def main_async():
        # Dummy callbacks
        async def on_path_selected_cb(path: Path):
            print(f"Path selected: {path}")
            get_app().exit(result=path)

        async def on_cancel_cb():
            print("Cancelled.")
            get_app().exit(result=None)

        # Create a FileManager instance (assuming local backend for example)
        # In a real app, this would come from a StorageRegistry
        fm = FileManager()

        # Create the browser
        file_browser = FileManagerBrowser(
            file_manager=fm,
            on_path_selected=on_path_selected_cb,
            on_cancel=on_cancel_cb,
            initial_path=Path.home(), # Start at home directory
            backend=Backend.DISK, # Specify backend
            select_files=True
        )

        # Create a simple layout for testing
        layout = Layout(
            container=Dialog(title="File Browser Test", body=file_browser, modal=True, width=Dimension(preferred=80)),
            focused_element=file_browser.ok_button # Focus an element within the browser
        )

        # Create and run the application
        app = Application(layout=layout, full_screen=True, key_bindings=file_browser.get_key_bindings())

        selected_path = await app.run_async()
        if selected_path:
            print(f"Application exited with selected path: {selected_path}")
        else:
            print("Application exited without selection.")

    asyncio.run(main_async())