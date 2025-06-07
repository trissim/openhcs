"""
Function List Manager - UI component for managing function lists.

This component handles function list display, editing, and management
with composition-based UI and observer pattern for clean separation.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
import time

from prompt_toolkit.layout.containers import HSplit, VSplit, Window, Container, DynamicContainer, FloatContainer, Float
from prompt_toolkit.layout import ScrollablePane
from prompt_toolkit.widgets import Button, Frame, Box, Label
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.data_structures import Point
from prompt_toolkit.application import get_app

from openhcs.tui.components.parameter_editor import ParameterEditor
from openhcs.tui.services.function_registry_service import FunctionRegistryService
from openhcs.tui.utils.unified_task_manager import get_task_manager

logger = logging.getLogger(__name__)


class FocusableList(DynamicContainer):
    """
    Makes complex function panes navigable using FileManager pattern.

    ARCHITECTURE:
    - Visual Layer: Complex function panes (Frame + HSplit + editors)
    - Selection Layer: FormattedTextControl with text representation
    - Navigation Layer: FileManager-style focus management

    NO FOCUS STEALING: Only one FormattedTextControl is focusable.
    """

    def __init__(self, function_items, parent_manager=None):
        """Show actual function panes with FileManagerBrowser-style navigation."""
        self._function_items = function_items
        self._parent_manager = parent_manager
        self.focused_index = 0  # Like FileManagerBrowser.focused_index

        # Create FormattedTextControl for navigation (like FileManagerBrowser)
        self.item_list_control = FormattedTextControl(
            text=self._get_navigation_text,
            focusable=True,
            key_bindings=self._create_key_bindings()
        )

        # Create ScrollablePane with navigation control (like FileManagerBrowser)
        self.scrollable_pane = ScrollablePane(
            Window(content=self.item_list_control),
            show_scrollbar=True,
            display_arrows=True
        )

        # Create visual overlay with actual function panes
        self.combined_container = self._create_combined_container()

        # Initialize DynamicContainer to return combined container
        super().__init__(self._get_child)

        logger.info(f"DEBUG: FocusableList created with {len(function_items)} function items")

    def _get_child(self):
        """Return the combined container (visual panes + navigation)."""
        return self.combined_container

    def _create_visual_pane(self):
        """Create ScrollablePane with actual function panes (visual layer)."""
        from prompt_toolkit.layout import HSplit, ScrollablePane

        if not self._function_items:
            from prompt_toolkit.widgets import Label
            return ScrollablePane(Label("No functions"))

        # Create HSplit with actual function panes - this shows the real UI
        inner_list = HSplit(self._function_items, padding=0)

        # Wrap in ScrollablePane for scrolling
        return ScrollablePane(
            inner_list,
            show_scrollbar=True,
            display_arrows=True
        )

    def _create_combined_container(self):
        """Overlay visual panes on top of navigation ScrollablePane."""
        from prompt_toolkit.layout import FloatContainer, Float

        # Base: ScrollablePane with FormattedTextControl (like FileManagerBrowser)
        # Overlay: Actual function panes positioned to match navigation text
        return FloatContainer(
            content=self.scrollable_pane,  # Navigation layer (invisible text)
            floats=[
                Float(
                    content=self._create_visual_pane(),  # Visual layer (actual panes)
                    left=0,
                    top=0,
                    width=lambda: None,  # Full width
                    height=lambda: None  # Full height
                )
            ]
        )

    def _move_focus(self, delta: int) -> None:
        """Move focus by delta items (exactly like FileManagerBrowser._move_focus)."""
        if not self._function_items:
            return
        new_index = max(0, min(len(self._function_items) - 1, self.focused_index + delta))
        self._set_focus(new_index)

    def _set_focus(self, index: int) -> None:
        """Set focus to specific index (exactly like FileManagerBrowser._set_focus)."""
        if 0 <= index < len(self._function_items):
            self.focused_index = index
            self._ensure_focused_visible()
            self._update_ui()

    def _ensure_focused_visible(self) -> None:
        """Ensure focused item visible (exactly like FileManagerBrowser._ensure_focused_visible)."""
        try:
            from prompt_toolkit.data_structures import Point

            # Position cursor at the focused function pane (like FileManagerBrowser)
            # Each function pane corresponds to one "line" in the navigation text
            cursor_y = max(0, min(self.focused_index, len(self._function_items) - 1))
            cursor_pos = Point(x=0, y=cursor_y)

            # Use the EXACT same monkey patch pattern as FileManagerBrowser
            self.item_list_control.get_cursor_position = lambda: cursor_pos
            self.item_list_control.show_cursor = True

        except Exception:
            pass  # Silently handle cursor positioning errors

    def _get_navigation_text(self):
        """Generate invisible navigation text (like FileManagerBrowser._get_item_list_text)."""
        if not self._function_items:
            return [("", "")]

        # Create one invisible line per function pane for cursor positioning
        lines = []
        for i, func_item in enumerate(self._function_items):
            # Each function pane = 1 line in navigation text (like FileManagerBrowser)
            lines.append(("", ""))  # Invisible line
            if i < len(self._function_items) - 1:
                lines.append(("", "\n"))

        return lines

    def _update_ui(self) -> None:
        """Update UI (exactly like FileManagerBrowser._update_ui)."""
        try:
            from prompt_toolkit.application import get_app
            get_app().invalidate()
        except Exception:
            pass





    def _create_key_bindings(self):
        """Create navigation key bindings (FileManager pattern)."""
        from prompt_toolkit.key_binding import KeyBindings
        kb = KeyBindings()

        @kb.add('up')
        def _(event):
            self._move_focus(-1)

        @kb.add('down')
        def _(event):
            self._move_focus(1)

        @kb.add('<scroll-up>')
        def _(event):
            self._move_focus(-3)

        @kb.add('<scroll-down>')
        def _(event):
            self._move_focus(3)

        @kb.add('home')
        def _(event):
            self._set_focus(0)

        @kb.add('end')
        def _(event):
            self._set_focus(len(self._function_items) - 1)

        @kb.add('enter')
        def _(event):
            # Trigger function dialog for focused item
            if hasattr(self._parent_manager, '_open_function_dialog'):
                from openhcs.tui.utils.unified_task_manager import get_task_manager
                get_task_manager().fire_and_forget(
                    self._parent_manager._open_function_dialog(self._focused_index),
                    f"function_enter_{self._focused_index}"
                )

        return kb

    def handle_initial_focus(self):
        """Handle initial focus (exactly like FileManagerBrowser pattern)."""
        if self._function_items:
            logger.info("DEBUG: Handling initial focus for FocusableList")
            # Focus the FormattedTextControl for navigation
            try:
                from prompt_toolkit.application import get_app
                get_app().layout.focus(self.item_list_control)
                # Set initial focus and cursor position
                self._set_focus(0)
                logger.info("DEBUG: Successfully focused item_list_control")
            except Exception as e:
                logger.warning(f"Could not focus item_list_control: {e}")

    def get_focused_pane(self):
        """Get the currently focused function pane."""
        if 0 <= self._focused_index < len(self._function_items):
            return self._function_items[self._focused_index]
        return None




# FunctionSelector class removed - using dialog-based selection instead


class FunctionListManager:
    """
    UI component for managing function lists.

    Handles function display, editing, reordering, and parameter management
    with composition-based UI following established TUI patterns.
    """

    def __init__(self, functions: List,
                 on_function_change: Optional[Callable] = None,
                 on_add_function: Optional[Callable] = None,
                 on_delete_function: Optional[Callable] = None,
                 on_move_function: Optional[Callable] = None,
                 on_parameter_change: Optional[Callable] = None,
                 app_state: Optional[Any] = None):
        """
        Initialize the function list manager.

        Args:
            functions: List of function items (callables or (callable, kwargs) tuples)
            on_function_change: Callback for function selection changes
            on_add_function: Callback for adding new functions
            on_delete_function: Callback for deleting functions
            on_move_function: Callback for moving functions
            on_parameter_change: Callback for parameter changes
            app_state: TUI state for dialog operations
        """
        self.functions = functions
        self.on_function_change = on_function_change
        self.on_add_function = on_add_function
        self.on_delete_function = on_delete_function
        self.on_move_function = on_move_function
        self.on_parameter_change = on_parameter_change
        self.app_state = app_state

        # Cache function registry data
        self.functions_by_backend = FunctionRegistryService.get_functions_by_backend()
        # Use functions_by_backend directly - it's already in the correct Dict format for GroupedDropdown

        self._container = self._build_function_list()
    
    @property
    def container(self):
        """Return the main container for the function list."""
        return self._container

    def get_focus_window(self):
        """Return the focusable control for key bindings."""
        # EMPTY STATE GUARD: Don't provide focus target when no functions exist
        if not self.functions:
            logger.info("DEBUG: No functions available - no focus target provided")
            raise RuntimeError("FunctionListManager: No functions available for focus")

        # Return the scroll control (like FileManagerBrowser)
        if hasattr(self, 'scroll_control') and self.scroll_control:
            logger.info("DEBUG: get_focus_window returning scroll control")
            return self.scroll_control
        else:
            logger.info("DEBUG: get_focus_window returning container (fallback)")
            return self._container

    def force_focus_restoration(self):
        """Force focus restoration - simplified for ScrollablePane."""
        # No complex restoration needed - just ensure focus is set
        logger.info("DEBUG: Force focus restoration called - simple ScrollablePane focus")



    def _build_function_list(self) -> Container:
        """
        Build the complete function list UI with simple scrolling.

        Returns:
            ScrollablePane with FormattedTextControl for keyboard input (FileManagerBrowser pattern)
        """
        function_items = []

        # Build individual function items
        for i, func_item in enumerate(self.functions):
            func, kwargs = self._extract_func_and_kwargs(func_item)
            function_items.append(self._create_function_item(i, func, kwargs))

        from prompt_toolkit.layout.dimension import Dimension

        if not function_items:
            function_items = [Label("No functions defined")]

        logger.info(f"DEBUG: Built {len(function_items)} function items for display")

        # Create FormattedTextControl with invisible text but working key bindings
        self.scroll_control = FormattedTextControl(
            text=lambda: [("", "\n") for _ in range(len(function_items) * 8)],  # Invisible spacer lines
            focusable=True,
            key_bindings=self._create_simple_scroll_bindings(),
            show_cursor=False
        )

        # Create combined content: invisible control + visual function panes
        combined_content = HSplit([
            Window(content=self.scroll_control, height=0),  # Invisible keyboard handler
            HSplit(function_items, padding=0)  # Actual function panes
        ])

        # Put combined content in ScrollablePane
        pane = ScrollablePane(
            combined_content,
            show_scrollbar=True,
            display_arrows=True
        )

        # Store references
        self.scrollable_pane = pane
        self.focused_index = 0

        return pane

    def _create_simple_scroll_bindings(self):
        """Simple scroll bindings that definitely work."""
        from prompt_toolkit.key_binding import KeyBindings
        kb = KeyBindings()

        @kb.add('up')
        def _(event):
            if hasattr(self, 'scrollable_pane'):
                current = getattr(self.scrollable_pane, 'vertical_scroll', 0)
                self.scrollable_pane.vertical_scroll = max(0, current - 1)
                get_app().invalidate()

        @kb.add('down')
        def _(event):
            if hasattr(self, 'scrollable_pane'):
                current = getattr(self.scrollable_pane, 'vertical_scroll', 0)
                self.scrollable_pane.vertical_scroll = current + 1
                get_app().invalidate()

        return kb

    def _get_scroll_text(self) -> List[Tuple[str, str]]:
        """Generate text for FormattedTextControl showing function names."""
        if not self.functions:
            return [("", "No functions defined")]

        lines = []
        for i, func_item in enumerate(self.functions):
            func, kwargs = self._extract_func_and_kwargs(func_item)

            # Style based on focus
            if i == getattr(self, 'focused_index', 0):
                style = "class:function-focused"
                prefix = "► "
            else:
                style = "class:function-normal"
                prefix = "  "

            # Function name with index
            func_name = getattr(func, '__name__', str(func))
            display_text = f"{prefix}{i+1}: {func_name}"

            lines.append((style, display_text))
            if i < len(self.functions) - 1:
                lines.append(("", "\n"))

        return lines

    def _get_navigation_text(self) -> List[Tuple[str, str]]:
        """Generate invisible navigation text for cursor positioning."""
        if not hasattr(self, 'function_items') or not self.functions:
            return [("", "")]

        lines = []
        for i, func_item in enumerate(self.functions):
            # Create invisible lines that correspond to visual function panes
            # Each function pane gets multiple invisible lines to match its visual height

            # Estimate function pane height (title + controls + parameters)
            # This should roughly match the visual height of each function pane
            pane_height = 8  # Approximate lines per function pane

            for line_in_pane in range(pane_height):
                if line_in_pane == 0 and i == self.focused_index:
                    # First line of focused pane - this is where cursor appears
                    lines.append(("", " "))  # Single space for cursor positioning
                else:
                    lines.append(("", ""))  # Empty invisible line

                if line_in_pane < pane_height - 1:
                    lines.append(("", "\n"))

            # Add separator between function panes
            if i < len(self.functions) - 1:
                lines.append(("", "\n"))

        return lines

    def _create_key_bindings(self):
        """Create navigation key bindings (like FileManagerBrowser)."""
        from prompt_toolkit.key_binding import KeyBindings
        kb = KeyBindings()

        @kb.add('up')
        def move_up(event):
            self._move_focus(-1)

        @kb.add('down')
        def move_down(event):
            self._move_focus(1)

        @kb.add('pageup')
        def page_up(event):
            self._move_focus(-5)

        @kb.add('pagedown')
        def page_down(event):
            self._move_focus(5)

        @kb.add('home')
        def go_home(event):
            self._set_focus(0)

        @kb.add('end')
        def go_end(event):
            self._set_focus(len(self.functions) - 1)

        @kb.add('<scroll-up>')
        def scroll_up(event):
            self._move_focus(-3)

        @kb.add('<scroll-down>')
        def scroll_down(event):
            self._move_focus(3)

        @kb.add('enter')
        def select_function(event):
            # Open function dialog for focused item
            if hasattr(self, '_open_function_dialog'):
                get_task_manager().fire_and_forget(
                    self._open_function_dialog(self.focused_index),
                    f"function_enter_{self.focused_index}"
                )

        return kb

    def _create_scroll_key_bindings(self):
        """Create simple scroll key bindings using cursor positioning (FileManagerBrowser pattern)."""
        from prompt_toolkit.key_binding import KeyBindings
        kb = KeyBindings()

        @kb.add('up')
        def scroll_up(event):
            self._move_selection(-1)

        @kb.add('down')
        def scroll_down(event):
            self._move_selection(1)

        @kb.add('pageup')
        def page_up(event):
            self._move_selection(-5)

        @kb.add('pagedown')
        def page_down(event):
            self._move_selection(5)

        @kb.add('<scroll-up>')
        def mouse_scroll_up(event):
            self._move_selection(-3)

        @kb.add('<scroll-down>')
        def mouse_scroll_down(event):
            self._move_selection(3)

        return kb

    def _move_selection(self, delta: int):
        """Scroll the visual function panes directly."""
        if hasattr(self, 'scrollable_pane') and self.scrollable_pane:
            # Get current scroll position
            current_scroll = getattr(self.scrollable_pane, 'vertical_scroll', 0)

            # Calculate new scroll position (delta lines)
            new_scroll = max(0, current_scroll + delta)

            # Apply scroll directly to ScrollablePane
            self.scrollable_pane.vertical_scroll = new_scroll

            # Invalidate to trigger redraw
            get_app().invalidate()

    def _move_focus(self, delta: int) -> None:
        """Move focus by delta (like FileManagerBrowser)."""
        if not self.functions:
            return
        new_index = max(0, min(len(self.functions) - 1, self.focused_index + delta))
        self._set_focus(new_index)

    def _set_focus(self, index: int) -> None:
        """Set focus to specific index (like FileManagerBrowser)."""
        if 0 <= index < len(self.functions):
            self.focused_index = index
            self._ensure_focused_visible()
            get_app().invalidate()

    def _ensure_focused_visible(self) -> None:
        """Ensure focused item visible with multi-line function pane positioning."""
        try:
            from prompt_toolkit.data_structures import Point

            # Calculate cursor position for multi-line function panes
            # Each function pane has ~8 lines, cursor goes to first line of focused pane
            pane_height = 8
            cursor_y = self.focused_index * (pane_height + 1)  # +1 for separator
            cursor_pos = Point(x=0, y=cursor_y)

            # Use the EXACT same monkey patch pattern as FileManagerBrowser
            self.list_control.get_cursor_position = lambda: cursor_pos
            self.list_control.show_cursor = True  # Show cursor for navigation

        except Exception:
            pass  # Silently handle cursor positioning errors

    def _create_function_item(self, index: int, func: Callable, kwargs: Dict) -> Frame:
        """
        Create UI component for a single function item.

        Args:
            index: Function index in the list
            func: Function callable
            kwargs: Function keyword arguments

        Returns:
            Frame containing function controls and parameter editor
        """
        # Get function info for display
        func_info = self._get_function_info_safe(func)

        # Function controls
        controls = self._create_function_controls(index)

        # Parameter editor for this function
        param_editor = self._create_parameter_editor(index, func, kwargs)

        # Create Frame with regular title (visual only - not focusable)
        title_text = f"{index+1}: {func_info['name']} ({func_info['backend']})"

        from prompt_toolkit.layout.dimension import Dimension

        frame = Frame(
            HSplit([
                # Function controls
                VSplit(controls, height=1),  # Fixed height for button row
                # Parameter editor - let it size itself based on content
                param_editor.container if param_editor else HSplit([])
            ], width=Dimension(weight=1)),
            title=title_text,  # Regular Frame title (red styling from CSS)
            width=Dimension(weight=1)
        )

        # Store function index for reference
        frame._function_index = index

        return frame



    async def _open_function_dialog(self, index: int):
        """Open function selection dialog for the given function index."""
        from openhcs.tui.dialogs.function_selector_dialog import FunctionSelectorDialog

        # Get current function
        func_item = self.functions[index]
        current_func, _ = self._extract_func_and_kwargs(func_item)

        # Create async callback for function selection
        async def handle_selection(new_func):
            await self._handle_function_change(index, new_func)

        dialog = FunctionSelectorDialog(
            functions_by_backend=self.functions_by_backend,
            current_func=current_func,
            on_selection=handle_selection
        )

        # Show dialog with app_state
        if self.app_state:
            await dialog.show(self.app_state)
        else:
            logger.error("No app_state available for function dialog - function title clicks will not work")
    
    def _create_function_controls(self, index: int) -> List[Box]:
        """
        Create function control buttons.

        Args:
            index: Function index

        Returns:
            List of Box-wrapped control buttons
        """
        move_up = Button(
            "↑",
            handler=lambda: get_task_manager().fire_and_forget(
                self._handle_move_up(index), f"move_up_{index}"
            ),
            width=3
        )

        move_down = Button(
            "↓",
            handler=lambda: get_task_manager().fire_and_forget(
                self._handle_move_down(index), f"move_down_{index}"
            ),
            width=3
        )

        add_button = Button(
            "Add",
            handler=lambda: get_task_manager().fire_and_forget(
                self._handle_add_function_at(index), f"add_at_{index}"
            ),
            width=5
        )

        delete_button = Button(
            "Delete",
            handler=lambda: get_task_manager().fire_and_forget(
                self._handle_delete(index), f"delete_{index}"
            ),
            width=8
        )

        # Return list of buttons for unpacking in VSplit
        return [
            move_up,
            move_down,
            add_button,
            delete_button
        ]
    
    def _create_parameter_editor(self, index: int, func: Callable, kwargs: Dict) -> Optional[ParameterEditor]:
        """
        Create parameter editor for function.
        
        Args:
            index: Function index
            func: Function callable
            kwargs: Current keyword arguments
            
        Returns:
            ParameterEditor instance or None if function is None
        """
        if func is None:
            return None
        
        return ParameterEditor(
            func=func,
            current_kwargs=kwargs,
            on_parameter_change=lambda p_name, p_val_str, idx=index: get_task_manager().fire_and_forget(
                self._handle_parameter_change(p_name, p_val_str, idx), f"param_change_{idx}"
            ),
            on_reset_parameter=lambda p_name, idx=index: get_task_manager().fire_and_forget(
                self._handle_reset_parameter(p_name, idx), f"reset_param_{idx}"
            ),
            on_reset_all_parameters=lambda idx=index: get_task_manager().fire_and_forget(
                self._handle_reset_all_parameters(idx), f"reset_all_{idx}"
            )
        )
    
    def _extract_func_and_kwargs(self, func_item) -> Tuple[Optional[Callable], Dict]:
        """
        Extract function and kwargs from function item.
        
        Args:
            func_item: Either (callable, kwargs) tuple or bare callable
            
        Returns:
            Tuple of (callable, kwargs_dict)
        """
        if isinstance(func_item, tuple) and len(func_item) == 2 and callable(func_item[0]):
            return func_item[0], func_item[1]
        elif callable(func_item):
            return func_item, {}
        else:
            return None, {}
    
    def _get_function_info_safe(self, func: Callable) -> Dict[str, str]:
        """
        Get function info with safe fallback.
        
        Args:
            func: Function to get info for
            
        Returns:
            Dict with function metadata
        """
        if func is None:
            return {"name": "None", "backend": "unknown"}
        
        try:
            return FunctionRegistryService.get_enhanced_function_metadata(func)
        except Exception as e:
            logger.warning(f"Failed to get function info for {func}: {e}")
            return {"name": getattr(func, '__name__', 'unknown'), "backend": "unknown"}
    
    # Event handlers
    async def _handle_function_change(self, index: int, new_func: Callable):
        """Handle function selection change."""
        if self.on_function_change:
            await self.on_function_change(index, new_func)
    
    async def _handle_add_function(self):
        """Handle add function button click."""
        logger.info("DEBUG: FunctionListManager._handle_add_function called")
        try:
            import asyncio
            logger.info(f"DEBUG: Event loop running: {asyncio.get_running_loop()}")
        except RuntimeError as e:
            logger.error(f"DEBUG: No event loop: {e}")

        if self.on_add_function:
            logger.info("DEBUG: Calling on_add_function callback")
            await self.on_add_function()
        else:
            logger.warning("DEBUG: No on_add_function callback set")

    async def _handle_add_function_at(self, index: int):
        """Handle add function at specific position."""
        if self.on_add_function:
            # Add function after the current index
            await self.on_add_function(index + 1)
    
    async def _handle_delete(self, index: int):
        """Handle delete function button click."""
        if self.on_delete_function:
            await self.on_delete_function(index)
    
    async def _handle_move_up(self, index: int):
        """Handle move up button click."""
        if self.on_move_function:
            await self.on_move_function(index, -1)  # Move up = -1
    
    async def _handle_move_down(self, index: int):
        """Handle move down button click."""
        if self.on_move_function:
            await self.on_move_function(index, 1)  # Move down = +1
    
    async def _handle_parameter_change(self, param_name: str, param_value: str, index: int):
        """Handle parameter change."""
        if self.on_parameter_change:
            await self.on_parameter_change(param_name, param_value, index)
    
    async def _handle_reset_parameter(self, param_name: str, index: int):
        """Handle parameter reset."""
        if self.on_parameter_change:
            # Reset to default value - implementation depends on parent
            await self.on_parameter_change(param_name, None, index)
    
    async def _handle_reset_all_parameters(self, index: int):
        """Handle reset all parameters."""
        if self.on_parameter_change:
            # Reset all parameters - implementation depends on parent
            await self.on_parameter_change(None, None, index)
    
    def update_functions(self, functions: List):
        """
        Update the function list with new data and recalculate height.

        Args:
            functions: New function list
        """
        logger.info(f"DEBUG: FunctionListManager.update_functions called with {len(functions)} functions")
        try:
            self.functions = functions
            logger.info("DEBUG: About to call _build_function_list with direct ScrollablePane pattern")
            self._container = self._build_function_list()

            logger.info("DEBUG: FunctionListManager.update_functions completed with direct ScrollablePane")
        except Exception as e:
            logger.error(f"DEBUG: Exception in update_functions: {e}", exc_info=True)
    
    @staticmethod
    def create_function_list_manager(functions: List,
                                   on_function_change: Optional[Callable] = None,
                                   on_add_function: Optional[Callable] = None,
                                   on_delete_function: Optional[Callable] = None,
                                   on_move_function: Optional[Callable] = None,
                                   on_parameter_change: Optional[Callable] = None,
                                   app_state: Optional[Any] = None) -> 'FunctionListManager':
        """
        Factory method for creating function list manager instances.

        Args:
            functions: Function list
            on_function_change: Function change callback
            on_add_function: Add function callback
            on_delete_function: Delete function callback
            on_move_function: Move function callback
            on_parameter_change: Parameter change callback
            app_state: TUI state for dialog operations

        Returns:
            FunctionListManager instance
        """
        return FunctionListManager(
            functions=functions,
            on_function_change=on_function_change,
            on_add_function=on_add_function,
            on_delete_function=on_delete_function,
            on_move_function=on_move_function,
            on_parameter_change=on_parameter_change,
            app_state=app_state
        )
