"""
Pattern Editor Controller - UI Coordination for Pattern Editing.

Coordinates between pattern editing UI and business logic services.
Handles user interactions and state synchronization.

🔒 Clause 295: Component Boundaries
Clear separation between controller, service, and view layers.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable

from prompt_toolkit.application import get_app

from openhcs.tui.services.pattern_editing_service import PatternEditingService
from openhcs.tui.services.external_editor_service import ExternalEditorService

logger = logging.getLogger(__name__)


class PatternEditorController:
    """
    Controller for pattern editor UI operations.
    
    Coordinates between:
    - PatternEditingService (business logic)
    - ExternalEditorService (external editing)
    - PatternEditorView (UI rendering)
    - TUIState (application state)
    """
    
    def __init__(self, state, initial_pattern: Union[List, Dict, None] = None, change_callback: Optional[Callable] = None):
        self.state = state
        self.change_callback = change_callback
        
        # Initialize services
        self.pattern_service = PatternEditingService(state)
        self.external_editor_service = ExternalEditorService(state)
        
        # Pattern state
        self.original_pattern = self.pattern_service.clone_pattern(initial_pattern)
        self.current_pattern = self.pattern_service.clone_pattern(self.original_pattern)
        self.current_key = None
        
        # UI state
        self.is_dict = self.pattern_service.is_dict_pattern(self.current_pattern)
        self.available_functions = self.pattern_service.get_available_functions()
        
        # Initialize current key for dict patterns
        if self.is_dict and self.current_pattern:
            self.current_key = next(iter(self.current_pattern.keys()))
    
    def get_pattern(self) -> Union[List, Dict]:
        """Get the current pattern."""
        return self.current_pattern
    
    def get_original_pattern(self) -> Union[List, Dict]:
        """Get the original pattern."""
        return self.original_pattern
    
    def has_changes(self) -> bool:
        """Check if the pattern has been modified."""
        return self.current_pattern != self.original_pattern
    
    def get_pattern_keys(self) -> List[str]:
        """Get available pattern keys."""
        return self.pattern_service.get_pattern_keys(self.current_pattern)
    
    def get_current_key(self) -> Optional[str]:
        """Get the currently selected key."""
        return self.current_key
    
    def set_current_key(self, key: Optional[str]):
        """Set the currently selected key."""
        self.current_key = key
        self._notify_ui_update()
    
    def get_current_functions(self) -> List[tuple]:
        """Get functions for the current key/pattern."""
        return self.pattern_service.get_pattern_functions(self.current_pattern, self.current_key)
    
    def get_available_functions(self) -> List[Callable]:
        """Get available functions from the registry."""
        return self.available_functions
    
    def get_function_info(self, func: Callable) -> Dict[str, Any]:
        """Get information about a function."""
        return self.pattern_service.get_function_info(func)
    
    async def add_pattern_key(self, key: str) -> bool:
        """
        Add a new key to the pattern.
        
        Args:
            key: Key to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_dict:
                await self._show_error("Cannot add key to list pattern. Convert to dict first.")
                return False
            
            self.pattern_service.add_pattern_key(self.current_pattern, key)
            self.current_key = key
            self._notify_change()
            self._notify_ui_update()
            return True
            
        except ValueError as e:
            await self._show_error(str(e))
            return False
        except Exception as e:
            logger.error(f"Error adding pattern key: {e}", exc_info=True)
            await self._show_error(f"Failed to add key: {str(e)}")
            return False
    
    async def remove_pattern_key(self, key: str) -> bool:
        """
        Remove a key from the pattern.
        
        Args:
            key: Key to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_dict:
                await self._show_error("Cannot remove key from list pattern.")
                return False
            
            self.pattern_service.remove_pattern_key(self.current_pattern, key)
            
            # Update current key if removed
            if self.current_key == key:
                remaining_keys = self.get_pattern_keys()
                self.current_key = remaining_keys[0] if remaining_keys else None
            
            self._notify_change()
            self._notify_ui_update()
            return True
            
        except ValueError as e:
            await self._show_error(str(e))
            return False
        except Exception as e:
            logger.error(f"Error removing pattern key: {e}", exc_info=True)
            await self._show_error(f"Failed to remove key: {str(e)}")
            return False
    
    async def convert_to_dict_pattern(self) -> bool:
        """Convert the current pattern to a dictionary pattern."""
        try:
            if self.is_dict:
                await self._show_info("Pattern is already a dictionary.")
                return True
            
            self.current_pattern = self.pattern_service.convert_list_to_dict_pattern(self.current_pattern)
            self.is_dict = True
            self.current_key = None  # Use None key for converted pattern
            
            self._notify_change()
            self._notify_ui_update()
            return True
            
        except Exception as e:
            logger.error(f"Error converting to dict pattern: {e}", exc_info=True)
            await self._show_error(f"Failed to convert pattern: {str(e)}")
            return False
    
    async def add_function(self, func: Callable, kwargs: Dict = None) -> bool:
        """
        Add a function to the current pattern.
        
        Args:
            func: Function to add
            kwargs: Function arguments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if kwargs is None:
                kwargs = {}
            
            self.pattern_service.add_function_to_pattern(
                self.current_pattern, func, kwargs, self.current_key
            )
            
            self._notify_change()
            self._notify_ui_update()
            return True
            
        except ValueError as e:
            await self._show_error(str(e))
            return False
        except Exception as e:
            logger.error(f"Error adding function: {e}", exc_info=True)
            await self._show_error(f"Failed to add function: {str(e)}")
            return False
    
    async def remove_function(self, index: int) -> bool:
        """
        Remove a function from the current pattern.
        
        Args:
            index: Index of function to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.pattern_service.remove_function_from_pattern(
                self.current_pattern, index, self.current_key
            )
            
            self._notify_change()
            self._notify_ui_update()
            return True
            
        except ValueError as e:
            await self._show_error(str(e))
            return False
        except Exception as e:
            logger.error(f"Error removing function: {e}", exc_info=True)
            await self._show_error(f"Failed to remove function: {str(e)}")
            return False
    
    async def move_function(self, from_index: int, to_index: int) -> bool:
        """
        Move a function within the pattern.
        
        Args:
            from_index: Current index
            to_index: Target index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.pattern_service.move_function_in_pattern(
                self.current_pattern, from_index, to_index, self.current_key
            )
            
            self._notify_change()
            self._notify_ui_update()
            return True
            
        except ValueError as e:
            await self._show_error(str(e))
            return False
        except Exception as e:
            logger.error(f"Error moving function: {e}", exc_info=True)
            await self._show_error(f"Failed to move function: {str(e)}")
            return False
    
    async def update_function_kwargs(self, index: int, kwargs: Dict) -> bool:
        """
        Update function arguments.
        
        Args:
            index: Index of function to update
            kwargs: New arguments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.pattern_service.update_function_kwargs(
                self.current_pattern, index, kwargs, self.current_key
            )
            
            self._notify_change()
            self._notify_ui_update()
            return True
            
        except ValueError as e:
            await self._show_error(str(e))
            return False
        except Exception as e:
            logger.error(f"Error updating function kwargs: {e}", exc_info=True)
            await self._show_error(f"Failed to update function: {str(e)}")
            return False
    
    async def validate_pattern(self) -> bool:
        """
        Validate the current pattern.
        
        Returns:
            True if valid, False otherwise
        """
        is_valid, error_message = self.pattern_service.validate_pattern(self.current_pattern)
        
        if not is_valid:
            await self._show_error(f"Pattern validation failed: {error_message}")
        
        return is_valid
    
    async def edit_in_external_editor(self) -> bool:
        """
        Edit the pattern in an external editor.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            initial_content = f"pattern = {repr(self.current_pattern)}"
            
            success, new_pattern, error_message = await self.external_editor_service.edit_pattern_in_external_editor(initial_content)
            
            if success and new_pattern is not None:
                self.current_pattern = new_pattern
                self.is_dict = self.pattern_service.is_dict_pattern(self.current_pattern)
                
                # Update current key for dict patterns
                if self.is_dict and self.current_pattern:
                    keys = self.get_pattern_keys()
                    self.current_key = keys[0] if keys else None
                
                self._notify_change()
                self._notify_ui_update()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error editing in external editor: {e}", exc_info=True)
            await self._show_error(f"Failed to edit in external editor: {str(e)}")
            return False
    
    def _notify_change(self):
        """Notify that the pattern has changed."""
        if self.change_callback:
            try:
                self.change_callback(self.current_pattern)
            except Exception as e:
                logger.error(f"Error in change callback: {e}", exc_info=True)
    
    def _notify_ui_update(self):
        """Notify that the UI should be updated."""
        app = get_app()
        app.invalidate()
    
    async def _show_error(self, message: str):
        """Show an error message."""
        await self.state.notify('show_dialog_requested', {
            'type': 'error',
            'data': {
                'title': 'Pattern Editor Error',
                'message': message
            }
        })
    
    async def _show_info(self, message: str):
        """Show an information message."""
        await self.state.notify('show_dialog_requested', {
            'type': 'info',
            'data': {
                'title': 'Pattern Editor',
                'message': message
            }
        })
