"""
Function List Manager - UI component for managing function lists.

This component handles function list display, editing, and management
with composition-based UI and observer pattern for clean separation.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
import time

from prompt_toolkit.layout.containers import HSplit, VSplit, Window, Container
from prompt_toolkit.layout import ScrollablePane
from prompt_toolkit.widgets import Button, Frame, Box, Label
from prompt_toolkit.mouse_events import MouseEventType

from openhcs.tui.components.parameter_editor import ParameterEditor
from openhcs.tui.services.function_registry_service import FunctionRegistryService
from openhcs.tui.utils.unified_task_manager import get_task_manager

logger = logging.getLogger(__name__)


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
    
    def _build_function_list(self) -> Container:
        """
        Build the complete function list UI.

        Returns:
            Frame containing ScrollablePane with all function items and controls
        """
        function_items = []

        # Build individual function items
        for i, func_item in enumerate(self.functions):
            func, kwargs = self._extract_func_and_kwargs(func_item)
            function_items.append(self._create_function_item(i, func, kwargs))

        # ScrollablePane will now correctly receive height=None from registry pattern
        from prompt_toolkit.layout.dimension import Dimension

        if not function_items:
            function_items = [Label("No functions defined")]

        # Calculate intelligent height based on content and terminal size
        intelligent_height = self._calculate_content_based_height(len(function_items))

        from prompt_toolkit.layout.dimension import Dimension
        scrollable_content = ScrollablePane(
            HSplit(function_items, padding=0, width=Dimension(weight=1)),  # Add width to HSplit
            height=intelligent_height,  # Add height constraint back to ScrollablePane
            width=Dimension(weight=1)   # Add width expansion to ScrollablePane
        )

        # Return ScrollablePane directly - no redundant Frame wrapper
        return scrollable_content
    
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

        # Create clickable frame title (restore red title appearance)
        title_text = f"{index+1}: {func_info['name']} ({func_info['backend']})"

        # Combine components with Frame title (this shows the red title)
        from prompt_toolkit.layout.dimension import Dimension
        frame = Frame(
            HSplit([
                # Function controls
                VSplit(controls, height=1),  # Fixed height for button row
                # Parameter editor - let it size itself based on content
                param_editor.container if param_editor else HSplit([])
            ], width=Dimension(weight=1)),  # Remove height=weight=1 from HSplit
            title=title_text,  # This creates the red clickable title
            width=Dimension(weight=1)   # Force Frame to expand width, but let height be content-based
            # Remove height=Dimension(weight=1) to allow content-based sizing
        )

        # Make frame title clickable using the working async handler pattern
        async def title_click_handler(mouse_event):
            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                # Use unified task manager for async operation
                from openhcs.tui.utils.unified_task_manager import get_task_manager
                get_task_manager().fire_and_forget(
                    self._open_function_dialog(index),
                    f"function_title_click_{index}"
                )
                return True
            return False

        # Add mouse handler to frame (this should work with our async handler patching)
        frame.mouse_handler = title_click_handler

        return frame

    def _calculate_content_based_height(self, num_functions: int) -> 'Dimension':
        """Calculate intelligent height based on content and terminal constraints."""
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.application import get_app

        # Base height for empty/minimal content
        min_height = 5

        # Estimate lines per function (function title + parameters)
        # Each function typically has: title (1) + 3-5 parameters (3-5) + spacing (1) = ~5-7 lines
        lines_per_function = 6
        content_height = min_height + (num_functions * lines_per_function)

        # Get terminal size for max calculation
        try:
            app = get_app()
            terminal_height = app.output.get_size().rows if app.output else 40
        except:
            terminal_height = 40  # Fallback

        # Calculate max height (terminal - padding for header, tabs, buttons, status)
        # Reserve space for: dialog title (3) + tab bar (3) + header buttons (3) + status (3) = 12 lines
        padding = 12
        max_height = max(min_height, terminal_height - padding)

        # Use content height but cap at max to enable scrolling
        preferred_height = min(content_height, max_height)

        return Dimension(min=min_height, preferred=preferred_height, max=max_height)

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
            logger.info("DEBUG: About to call _build_function_list with content-based height")
            self._container = self._build_function_list()
            logger.info("DEBUG: FunctionListManager.update_functions completed with intelligent height")
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
