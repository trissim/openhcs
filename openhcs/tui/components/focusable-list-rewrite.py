"""
Function List Manager - UI component for managing function lists.

This component handles function list display, editing, and management
with composition-based UI and observer pattern for clean separation.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
import time

from prompt_toolkit.layout.containers import HSplit, VSplit, Window, Container, DynamicContainer, ConditionalContainer
from prompt_toolkit.layout import ScrollablePane
from prompt_toolkit.widgets import Button, Frame, Box, Label
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition

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
        """Initialize with simplified approach: visual panes + navigation control."""
        self._function_items = function_items
        self._parent_manager = parent_manager
        self._focused_index = 0
        
        # Store wrapped items for selection styling
        self._wrapped_items = []
        
        # Create navigation control with actual text
        self.navigation_control = FormattedTextControl(
            text=self._get_navigation_text,
            focusable=True,
            key_bindings=self._create_key_bindings(),
            show_cursor=True
        )
        
        # Create the main container
        self.container = self._create_container()
        
        # Initialize DynamicContainer
        super().__init__(lambda: self.container)
        
        logger.info(f"DEBUG: FocusableList created with {len(function_items)} function items")

    def _create_container(self):
        """Create the main container with visual panes and navigation."""
        if not self._function_items:
            return Label("No functions available")
        
        # Wrap each function item in ConditionalContainer for selection styling
        self._wrapped_items = []
        for i, item in enumerate(self._function_items):
            # Create wrapper that changes style based on selection
            wrapped = ConditionalContainer(
                content=item,
                filter=Condition(lambda i=i: True)  # Always visible
            )
            # Store reference to update style later
            wrapped._original_item = item
            wrapped._index = i
            self._wrapped_items.append(wrapped)
        
        # Create visual pane with wrapped items
        visual_content = HSplit(self._wrapped_items, padding=0)
        visual_pane = ScrollablePane(
            visual_content,
            show_scrollbar=True,
            display_arrows=True,
            keep_cursor_visible=True
        )
        
        # Store reference for scrolling
        self._visual_pane = visual_pane
        
        # Create navigation window (minimal height, but visible)
        nav_window = Window(
            content=self.navigation_control,
            height=1,  # Single line navigation indicator
            style="class:function-nav"
        )
        
        # Combine visual and navigation
        return HSplit([
            visual_pane,
            Window(height=1, char='-'),  # Separator
            nav_window
        ])

    def _get_navigation_text(self):
        """Get navigation text showing current selection."""
        if not self._function_items:
            return [("", "No functions")]
        
        # Show current selection in navigation bar
        func_info = self._get_function_info(self._focused_index)
        text = f"▶ {self._focused_index + 1}/{len(self._function_items)}: {func_info['name']} ({func_info['backend']})"
        
        # Return styled text
        return [("class:function-nav.focused", text)]

    def _get_function_info(self, index):
        """Extract function info from parent manager."""
        if (self._parent_manager and 
            hasattr(self._parent_manager, 'functions') and 
            0 <= index < len(self._parent_manager.functions)):
            func_item = self._parent_manager.functions[index]
            func, kwargs = self._parent_manager._extract_func_and_kwargs(func_item)
            if func:
                info = self._parent_manager._get_function_info_safe(func)
                return info
        return {'name': f'Function {index + 1}', 'backend': 'Unknown'}

    def _move_focus(self, delta: int) -> None:
        """Move focus by delta items."""
        if not self._function_items:
            return
        new_index = max(0, min(len(self._function_items) - 1, self._focused_index + delta))
        self._set_focus(new_index)

    def _set_focus(self, index: int) -> None:
        """Set focus to specific index and update visual styling."""
        if 0 <= index < len(self._function_items):
            old_index = self._focused_index
            self._focused_index = index
            
            # Update visual selection
            self._update_visual_selection(old_index, index)
            
            # Update cursor position for scrolling
            self._update_cursor_position()
            
            # Ensure visible
            self._ensure_focused_visible()
            
            # Update UI
            self._update_ui()

    def _update_visual_selection(self, old_index: int, new_index: int):
        """Update visual selection by recreating frames with different styles."""
        # Since Frame doesn't support dynamic styling, we need a different approach
        # We'll update the wrapped items to show selection
        
        # For now, we'll rely on the navigation bar to show selection
        # In a full implementation, you'd recreate the Frames with different styles
        # or use a different container that supports dynamic styling
        pass

    def _update_cursor_position(self):
        """Update cursor position to trigger automatic scrolling."""
        try:
            # Set cursor position to match focused index
            # This triggers ScrollablePane's keep_cursor_visible
            cursor_pos = Point(x=0, y=self._focused_index)
            
            # Monkey patch cursor position
            self.navigation_control.get_cursor_position = lambda: cursor_pos
            
        except Exception as e:
            logger.warning(f"Could not update cursor position: {e}")

    def _ensure_focused_visible(self):
        """Ensure focused item is visible in the scrollable pane."""
        try:
            if hasattr(self._visual_pane, 'vertical_scroll'):
                # Get viewport height
                viewport_height = getattr(self._visual_pane, 'height', 10) or 10
                
                # Estimate item heights (this is still approximate but better)
                # Each function pane is roughly 5-7 lines
                item_height = 6
                item_top = self._focused_index * item_height
                item_bottom = item_top + item_height
                
                current_scroll = self._visual_pane.vertical_scroll
                viewport_bottom = current_scroll + viewport_height
                
                # Scroll if item is outside viewport
                if item_top < current_scroll:
                    # Item is above viewport
                    self._visual_pane.vertical_scroll = item_top
                elif item_bottom > viewport_bottom:
                    # Item is below viewport
                    self._visual_pane.vertical_scroll = item_bottom - viewport_height
                    
        except Exception as e:
            logger.warning(f"Could not ensure visible: {e}")

    def _update_ui(self):
        """Update UI."""
        try:
            from prompt_toolkit.application import get_app
            get_app().invalidate()
        except Exception:
            pass

    def _create_key_bindings(self):
        """Create navigation key bindings."""
        kb = KeyBindings()

        @kb.add('up')
        def _(event):
            self._move_focus(-1)

        @kb.add('down')
        def _(event):
            self._move_focus(1)

        @kb.add('pageup')
        def _(event):
            self._move_focus(-5)

        @kb.add('pagedown')
        def _(event):
            self._move_focus(5)

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
                get_task_manager().fire_and_forget(
                    self._parent_manager._open_function_dialog(self._focused_index),
                    f"function_enter_{self._focused_index}"
                )

        return kb

    def handle_initial_focus(self):
        """Handle initial focus when layout is ready."""
        if self._function_items:
            logger.info("DEBUG: Handling initial focus for FocusableList")
            try:
                from prompt_toolkit.application import get_app
                app = get_app()
                
                # Focus the navigation control
                app.layout.focus(self.navigation_control)
                
                # Set initial selection
                self._set_focus(0)
                
                logger.info("DEBUG: Successfully focused navigation control")
            except Exception as e:
                logger.warning(f"Could not focus navigation control: {e}")

    def get_focused_pane(self):
        """Get the currently focused function pane."""
        if 0 <= self._focused_index < len(self._function_items):
            return self._function_items[self._focused_index]
        return None

    @property
    def focused_index(self):
        """Current focused index."""
        return self._focused_index


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

        # Handle initial focus if pending
        if hasattr(self._container, 'handle_initial_focus'):
            self._container.handle_initial_focus()

        # Return the navigation control for key bindings
        if hasattr(self._container, 'navigation_control'):
            logger.info(f"DEBUG: get_focus_window returning navigation control")
            return self._container.navigation_control

        logger.warning("DEBUG: No navigation control available, returning container")
        return self._container

    def force_focus_restoration(self):
        """Force focus restoration - can be called when returning from dialogs or switching tabs."""
        if hasattr(self._container, 'handle_initial_focus'):
            logger.info("DEBUG: Force focus restoration called")
            self._container.handle_initial_focus()

    def _build_function_list(self) -> Container:
        """
        Build the complete function list UI.

        Returns:
            Container with scrollable function list
        """
        if not self.functions:
            return Label("No functions defined")

        function_items = []

        # Build individual function items
        for i, func_item in enumerate(self.functions):
            func, kwargs = self._extract_func_and_kwargs(func_item)
            function_items.append(self._create_function_item(i, func, kwargs))

        # Create FocusableList with simplified pattern
        focusable_list = FocusableList(function_items, parent_manager=self)

        # Return the focusable wrapper
        return focusable_list
    
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

        # Create Frame with regular title
        title_text = f"{index+1}: {func_info['name']} ({func_info['backend']})"

        from prompt_toolkit.layout.dimension import Dimension

        frame = Frame(
            HSplit([
                # Function controls
                VSplit(controls, height=1),  # Fixed height for button row
                # Parameter editor
                param_editor.container if param_editor else HSplit([])
            ], width=Dimension(weight=1)),
            title=title_text,
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
            logger.error("No app_state available for function dialog")
    
    def _create_function_controls(self, index: int) -> List[Box]:
        """
        Create function control buttons.

        Args:
            index: Function index

        Returns:
            List of control buttons
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

        return [move_up, move_down, add_button, delete_button]
    
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
    
    # Event handlers remain the same...
    async def _handle_function_change(self, index: int, new_func: Callable):
        """Handle function selection change."""
        if self.on_function_change:
            await self.on_function_change(index, new_func)
    
    async def _handle_add_function(self):
        """Handle add function button click."""
        logger.info("DEBUG: FunctionListManager._handle_add_function called")
        if self.on_add_function:
            logger.info("DEBUG: Calling on_add_function callback")
            await self.on_add_function()
        else:
            logger.warning("DEBUG: No on_add_function callback set")

    async def _handle_add_function_at(self, index: int):
        """Handle add function at specific position."""
        if self.on_add_function:
            await self.on_add_function(index + 1)
    
    async def _handle_delete(self, index: int):
        """Handle delete function button click."""
        if self.on_delete_function:
            await self.on_delete_function(index)
    
    async def _handle_move_up(self, index: int):
        """Handle move up button click."""
        if self.on_move_function:
            await self.on_move_function(index, -1)
    
    async def _handle_move_down(self, index: int):
        """Handle move down button click."""
        if self.on_move_function:
            await self.on_move_function(index, 1)
    
    async def _handle_parameter_change(self, param_name: str, param_value: str, index: int):
        """Handle parameter change."""
        if self.on_parameter_change:
            await self.on_parameter_change(param_name, param_value, index)
    
    async def _handle_reset_parameter(self, param_name: str, index: int):
        """Handle parameter reset."""
        if self.on_parameter_change:
            await self.on_parameter_change(param_name, None, index)
    
    async def _handle_reset_all_parameters(self, index: int):
        """Handle reset all parameters."""
        if self.on_parameter_change:
            await self.on_parameter_change(None, None, index)
    
    def update_functions(self, functions: List):
        """
        Update the function list with new data.

        Args:
            functions: New function list
        """
        logger.info(f"DEBUG: FunctionListManager.update_functions called with {len(functions)} functions")
        try:
            self.functions = functions
            self._container = self._build_function_list()

            # Trigger initial focus when functions are added
            if functions and hasattr(self._container, 'handle_initial_focus'):
                logger.info("DEBUG: Triggering initial focus after adding functions")
                self._container.handle_initial_focus()

            logger.info("DEBUG: FunctionListManager.update_functions completed")
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