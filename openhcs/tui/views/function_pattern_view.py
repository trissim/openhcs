"""
Function Pattern View - Minimal coordinator replacing FunctionPatternEditor god class.

This view coordinates between services and UI components with clean separation
of concerns and exact interface preservation for DualEditorPane compatibility.
"""

from typing import Union, List, Dict, Any, Optional, Callable
import logging

from prompt_toolkit.layout.containers import HSplit, VSplit, Container
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import Button, Label, Box, Frame
from prompt_toolkit.formatted_text import HTML

from openhcs.tui.services.pattern_data_manager import PatternDataManager
from openhcs.tui.services.function_registry_service import FunctionRegistryService
from openhcs.tui.services.pattern_file_service import PatternFileService
from openhcs.tui.components.pattern_key_selector import PatternKeySelector
from openhcs.tui.components.function_list_manager import FunctionListManager
from openhcs.tui.utils.unified_task_manager import get_task_manager
from openhcs.tui.utils.dialog_helpers import prompt_for_file_dialog, prompt_for_save_file_dialog
from openhcs.tui.utils import show_error_dialog

logger = logging.getLogger(__name__)


class FunctionPatternView:
    """
    Minimal coordinator for function pattern editing.
    
    EXACT INTERFACE PRESERVATION: Maintains same constructor signature and public methods
    as original FunctionPatternEditor for DualEditorPane compatibility.
    """
    
    def __init__(self, state: Any, initial_pattern: Union[List, Dict, None] = None,
                 change_callback: Optional[Callable] = None, step_context: Optional[Dict] = None):
        """
        Initialize the function pattern view.

        Args:
            state: TUIState instance
            initial_pattern: Initial function pattern to edit
            change_callback: Callback to notify when pattern changes
            step_context: Optional step context with group_by and variable_components
        """
        self.state = state
        self.change_callback = change_callback
        self.step_context = step_context or {}
        
        # Initialize services
        self.data_manager = PatternDataManager()
        self.registry_service = FunctionRegistryService()
        self.file_service = PatternFileService(state)
        
        # Initialize pattern data
        self.current_pattern = self.data_manager.clone_pattern(
            initial_pattern if initial_pattern is not None else []
        )
        
        # Pattern state
        self.is_dict = isinstance(self.current_pattern, dict)
        self.current_key = None if not self.is_dict else next(iter(self.current_pattern), None)
        
        # Initialize UI components
        self.header = self._create_header()
        self.key_selector = self._create_key_selector()
        self.function_list = self._create_function_list()
        
        # Create main container - function_list.container is already a Frame with proper height
        self._container = HSplit([
            self.header,
            self.key_selector.container,
            self.function_list.container  # Already has height set at creation time
        ])
    
    @property
    def container(self) -> Container:
        """
        Return the main container for the view.
        
        EXACT INTERFACE: Preserves original property for DualEditorPane compatibility.
        """
        return self._container
    
    def get_pattern(self) -> Union[List, Dict]:
        """
        Return the current pattern state.
        
        EXACT INTERFACE: Preserves original method for DualEditorPane compatibility.
        """
        return self.data_manager.clone_pattern(self.current_pattern)
    
    def _create_header(self) -> VSplit:
        """Create the header with title and file operation buttons."""
        title = Label(HTML("<b>Function Pattern Editor</b>"))

        add_func_button = Button(
            "Add Function",
            handler=lambda: get_task_manager().fire_and_forget(
                self._handle_add_function(), "add_function"
            ),
            width=15
        )

        load_func_button = Button(
            "Load .func",
            handler=lambda: get_task_manager().fire_and_forget(
                self._load_func_pattern_from_file(), "load_func"
            ),
            width=12
        )

        save_as_func_button = Button(
            "Save .func As",
            handler=lambda: get_task_manager().fire_and_forget(
                self._save_func_pattern_as_file(), "save_func"
            ),
            width=15
        )

        edit_in_vim_button = Button(
            "Edit in Vim",
            handler=lambda: get_task_manager().fire_and_forget(
                self._edit_in_vim(), "edit_vim"
            ),
            width=12
        )

        return VSplit([
            title,
            Box(add_func_button, padding_left=2),
            Box(load_func_button, padding_left=1),
            Box(save_as_func_button, padding_left=1),
            Box(edit_in_vim_button, padding_left=1)
        ], height=1, padding=0)
    
    def _create_key_selector(self) -> PatternKeySelector:
        """Create the key selector component."""
        return PatternKeySelector(
            pattern=self.current_pattern,
            current_key=self.current_key,
            is_dict=self.is_dict,
            on_key_change=self._handle_key_change,
            on_add_key=self._handle_add_key,
            on_remove_key=self._handle_remove_key,
            on_convert_to_dict=self._handle_convert_to_dict
        )
    
    def _create_function_list(self) -> FunctionListManager:
        """Create the function list component."""
        current_functions = self.data_manager.get_current_functions(
            self.current_pattern, self.current_key, self.is_dict
        )
        
        return FunctionListManager(
            functions=current_functions,
            on_function_change=self._handle_function_change,
            on_add_function=self._handle_add_function,
            on_delete_function=self._handle_delete_function,
            on_move_function=self._handle_move_function,
            on_parameter_change=self._handle_parameter_change,
            app_state=self.state
        )


    
    def _notify_change(self):
        """Notify parent component of pattern changes."""
        if self.change_callback:
            self.change_callback()
    
    def _refresh_ui(self):
        """Refresh all UI components after pattern changes."""
        logger.info("DEBUG: _refresh_ui called")

        try:
            logger.info("DEBUG: Updating key selector")
            # Update key selector
            self.key_selector.update_pattern(self.current_pattern, self.current_key, self.is_dict)

            logger.info("DEBUG: Getting current functions for UI refresh")
            # Update function list
            current_functions = self.data_manager.get_current_functions(
                self.current_pattern, self.current_key, self.is_dict
            )
            logger.info(f"DEBUG: About to update function list with {len(current_functions)} functions")

            self.function_list.update_functions(current_functions)
            logger.info("DEBUG: Function list updated")

            logger.info("DEBUG: Rebuilding main container")
            # Rebuild main container - let prompt-toolkit size naturally
            self._container = HSplit([
                self.header,
                self.key_selector.container,
                self.function_list.container
            ])

            # Force parent layout to re-render with new container
            from prompt_toolkit.application import get_app
            get_app().invalidate()
            logger.info("DEBUG: _refresh_ui completed successfully - app invalidated")
        except Exception as e:
            logger.error(f"DEBUG: Exception in _refresh_ui: {e}", exc_info=True)
    
    # Event handlers
    async def _handle_key_change(self, new_key: Any):
        """Handle key selection change."""
        self.current_key = new_key
        self._refresh_ui()
        self._notify_change()
    
    async def _handle_add_key(self):
        """Handle add key request."""
        # Prompt user for experimental component identifier
        from openhcs.tui.utils.dialog_helpers import prompt_for_text_input

        new_key = await prompt_for_text_input(
            title="Add Component Identifier",
            message="Enter experimental component ID (e.g., '1', '2', 'DAPI', 'GFP'):",
            app_state=self.state
        )

        if new_key and isinstance(self.current_pattern, dict):
            self.current_pattern = self.data_manager.add_new_key(self.current_pattern, new_key)
            self.current_key = new_key
            self._refresh_ui()
            self._notify_change()
    
    async def _handle_remove_key(self):
        """Handle remove key request."""
        if isinstance(self.current_pattern, dict) and self.current_key is not None:
            self.current_pattern = self.data_manager.remove_key(self.current_pattern, self.current_key)
            
            # Update pattern type if converted back to list
            self.is_dict = isinstance(self.current_pattern, dict)
            self.current_key = None if not self.is_dict else next(iter(self.current_pattern), None)
            
            self._refresh_ui()
            self._notify_change()
    
    async def _handle_convert_to_dict(self):
        """Handle convert to dict request."""
        if isinstance(self.current_pattern, list):
            self.current_pattern = self.data_manager.convert_list_to_dict(self.current_pattern)
            self.is_dict = True
            self.current_key = None  # No keys initially - user will add them
            self._refresh_ui()
            self._notify_change()
    
    async def _handle_function_change(self, index: int, new_func: Callable):
        """Handle function selection change."""
        current_functions = self.data_manager.get_current_functions(
            self.current_pattern, self.current_key, self.is_dict
        )
        
        if 0 <= index < len(current_functions):
            # Preserve existing kwargs when changing function
            _, existing_kwargs = self.data_manager.extract_func_and_kwargs(current_functions[index])
            current_functions[index] = (new_func, existing_kwargs)
            
            self.current_pattern = self.data_manager.update_pattern_functions(
                self.current_pattern, self.current_key, self.is_dict, current_functions
            )
            
            self._refresh_ui()
            self._notify_change()
    
    async def _handle_add_function(self, index: Optional[int] = None):
        """Handle add function request by showing function selection dialog.

        Args:
            index: Optional position to insert function. If None, appends to end.
        """
        logger.info(f"DEBUG: _handle_add_function called with index: {index}")

        try:
            # Show function selection dialog instead of auto-selecting default
            from openhcs.tui.dialogs.function_selector_dialog import FunctionSelectorDialog
            from openhcs.tui.services.function_registry_service import FunctionRegistryService

            logger.info("DEBUG: Opening function selection dialog for new function")

            # Get functions by backend for dialog
            functions_by_backend = FunctionRegistryService.get_functions_by_backend()

            # Create dialog with callback to handle selection
            def handle_function_selection(selected_func):
                logger.info(f"DEBUG: Function selected from dialog: {selected_func}")

                # Add the selected function
                current_functions = self.data_manager.get_current_functions(
                    self.current_pattern, self.current_key, self.is_dict
                )

                if index is not None:
                    logger.info(f"DEBUG: Inserting selected function at index {index}")
                    current_functions.insert(index, (selected_func, {}))
                else:
                    logger.info("DEBUG: Appending selected function to end")
                    current_functions.append((selected_func, {}))

                # Update pattern
                self.current_pattern = self.data_manager.update_pattern_functions(
                    self.current_pattern, self.current_key, self.is_dict, current_functions
                )

                # Refresh UI
                self._refresh_ui()
                self._notify_change()
                logger.info("DEBUG: Function added successfully")

            dialog = FunctionSelectorDialog(
                functions_by_backend=functions_by_backend,
                current_func=None,  # No current function for new additions
                on_selection=handle_function_selection
            )

            # Show dialog using proper async interface
            await dialog.show(self.state)
            logger.info("DEBUG: Function selection dialog completed")

        except Exception as e:
            logger.error(f"DEBUG: Exception in _handle_add_function: {e}", exc_info=True)
    
    async def _handle_delete_function(self, index: int):
        """Handle delete function request."""
        current_functions = self.data_manager.get_current_functions(
            self.current_pattern, self.current_key, self.is_dict
        )
        
        if 0 <= index < len(current_functions):
            current_functions.pop(index)
            
            self.current_pattern = self.data_manager.update_pattern_functions(
                self.current_pattern, self.current_key, self.is_dict, current_functions
            )
            
            self._refresh_ui()
            self._notify_change()
    
    async def _handle_move_function(self, index: int, direction: int):
        """Handle move function request."""
        current_functions = self.data_manager.get_current_functions(
            self.current_pattern, self.current_key, self.is_dict
        )
        
        new_index = index + direction
        if 0 <= new_index < len(current_functions):
            # Swap functions
            current_functions[index], current_functions[new_index] = \
                current_functions[new_index], current_functions[index]
            
            self.current_pattern = self.data_manager.update_pattern_functions(
                self.current_pattern, self.current_key, self.is_dict, current_functions
            )
            
            self._refresh_ui()
            self._notify_change()
    
    async def _handle_parameter_change(self, param_name: str, param_value: str, index: int):
        """Handle parameter change request."""
        # Implementation depends on specific parameter change logic
        # This is a simplified version - full implementation would parse values
        logger.info(f"Parameter change: {param_name}={param_value} for function {index}")
        self._notify_change()
    
    # File operations (preserved interface)
    # REMOVED: _add_function - No longer needed, FunctionListManager handles add function
    
    async def _load_func_pattern_from_file(self):
        """Load pattern from file (preserved interface)."""
        file_path_str = await prompt_for_file_dialog(
            title="Load .func Pattern File",
            prompt_message="Select .func pattern file to load:",
            app_state=self.state,
            filemanager=getattr(self.state, 'filemanager', None)
        )
        
        if file_path_str:
            try:
                from pathlib import Path
                loaded_pattern = await self.file_service.load_pattern_from_file(Path(file_path_str))
                
                if self.data_manager.validate_pattern_structure(loaded_pattern):
                    self.current_pattern = loaded_pattern
                    self.is_dict = isinstance(self.current_pattern, dict)
                    self.current_key = None if not self.is_dict else next(iter(self.current_pattern), None)
                    self._refresh_ui()
                    self._notify_change()
                else:
                    await show_error_dialog("Load Error", "Invalid pattern structure.", app_state=self.state)
                    
            except Exception as e:
                logger.error(f"Failed to load pattern: {e}")
                await show_error_dialog("Load Error", f"Failed to load pattern: {e}", app_state=self.state)
    
    async def _save_func_pattern_as_file(self):
        """Save pattern to file (preserved interface)."""
        file_path_str = await prompt_for_save_file_dialog(
            title="Save .func Pattern As",
            message="Enter path to save .func pattern file:",
            state=self.state,
            initial_path="pattern.func"
        )
        
        if file_path_str:
            try:
                from pathlib import Path
                file_path = Path(self.file_service.ensure_func_extension(file_path_str))
                await self.file_service.save_pattern_to_file(self.current_pattern, file_path)
                logger.info(f"Pattern saved to {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save pattern: {e}")
                await show_error_dialog("Save Error", f"Failed to save pattern: {e}", app_state=self.state)
    
    async def _edit_in_vim(self):
        """Edit pattern in external editor (preserved interface)."""
        try:
            success, new_pattern, error_message = await self.file_service.edit_pattern_externally(self.current_pattern)
            
            if success and self.data_manager.validate_pattern_structure(new_pattern):
                self.current_pattern = new_pattern
                self.is_dict = isinstance(self.current_pattern, dict)
                self.current_key = None if not self.is_dict else next(iter(self.current_pattern), None)
                self._refresh_ui()
                self._notify_change()
            elif error_message:
                await show_error_dialog("Edit Error", error_message, app_state=self.state)
                
        except Exception as e:
            logger.error(f"External editor failed: {e}")
            await show_error_dialog("Edit Error", f"External editor failed: {e}", app_state=self.state)
