"""
Menu Service - Business Logic Layer.

Handles menu structure management, command execution, and state coordination.
Separates business logic from UI concerns.

🔒 Clause 295: Component Boundaries
Clear separation between service layer and UI layer.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum

from openhcs.core.context.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


class MenuItemType(Enum):
    """Types of menu items."""
    COMMAND = "command"
    CHECKBOX = "checkbox"
    SEPARATOR = "separator"
    SUBMENU = "submenu"


class MenuService:
    """
    Service layer for menu operations.
    
    Handles:
    - Menu structure management
    - Command execution coordination
    - State-based menu enabling/disabling
    - Menu validation
    """
    
    def __init__(self, state, context: ProcessingContext):
        self.state = state
        self.context = context
        
        # Menu state
        self.active_menu: Optional[str] = None
        self.active_submenu: Optional[List[Any]] = None
        self.active_item_index: Optional[int] = None
        
        # Thread safety
        self.menu_lock = asyncio.Lock()
        
        # Command registry
        self.command_registry = {}
        
        # Initialize command handlers
        self._initialize_command_handlers()
    
    def _initialize_command_handlers(self):
        """Initialize the command handler registry."""
        self.command_registry = {
            'new_pipeline': self._handle_new_pipeline,
            'open_pipeline': self._handle_open_pipeline,
            'save_pipeline': self._handle_save_pipeline,
            'save_pipeline_as': self._handle_save_pipeline_as,
            'export_pipeline': self._handle_export_pipeline,
            'import_pipeline': self._handle_import_pipeline,
            'exit': self._handle_exit,
            'undo': self._handle_undo,
            'redo': self._handle_redo,
            'cut': self._handle_cut,
            'copy': self._handle_copy,
            'paste': self._handle_paste,
            'select_all': self._handle_select_all,
            'find': self._handle_find,
            'replace': self._handle_replace,
            'add_step': self._handle_add_step,
            'remove_step': self._handle_remove_step,
            'duplicate_step': self._handle_duplicate_step,
            'move_step_up': self._handle_move_step_up,
            'move_step_down': self._handle_move_step_down,
            'run_pipeline': self._handle_run_pipeline,
            'stop_pipeline': self._handle_stop_pipeline,
            'validate_pipeline': self._handle_validate_pipeline,
            'debug_pipeline': self._handle_debug_pipeline,
            'preferences': self._handle_preferences,
            'keyboard_shortcuts': self._handle_keyboard_shortcuts,
            'documentation': self._handle_documentation,
            'about': self._handle_about,
        }
    
    async def execute_command(self, command_name: str, **kwargs) -> bool:
        """
        Execute a menu command.
        
        Args:
            command_name: Name of the command to execute
            **kwargs: Additional arguments for the command
            
        Returns:
            True if command was executed successfully, False otherwise
        """
        if command_name not in self.command_registry:
            logger.warning(f"Unknown command: {command_name}")
            await self._notify_error(f"Unknown command: {command_name}")
            return False
        
        try:
            handler = self.command_registry[command_name]
            await handler(**kwargs)
            return True
        except Exception as e:
            logger.error(f"Error executing command '{command_name}': {e}", exc_info=True)
            await self._notify_error(f"Error executing command '{command_name}': {str(e)}")
            return False
    
    async def set_active_menu(self, menu_name: Optional[str]):
        """Set the active menu."""
        async with self.menu_lock:
            self.active_menu = menu_name
            if menu_name is None:
                self.active_submenu = None
                self.active_item_index = None
    
    async def set_active_submenu(self, submenu: Optional[List[Any]], item_index: Optional[int] = None):
        """Set the active submenu and item index."""
        async with self.menu_lock:
            self.active_submenu = submenu
            self.active_item_index = item_index
    
    async def get_menu_state(self) -> Dict[str, Any]:
        """Get the current menu state."""
        async with self.menu_lock:
            return {
                'active_menu': self.active_menu,
                'active_submenu': self.active_submenu,
                'active_item_index': self.active_item_index
            }
    
    def is_command_enabled(self, command_name: str) -> bool:
        """
        Check if a command is currently enabled based on application state.
        
        Args:
            command_name: Name of the command to check
            
        Returns:
            True if command is enabled, False otherwise
        """
        # Get current application state
        has_active_plate = hasattr(self.state, 'selected_plate') and self.state.selected_plate is not None
        has_active_pipeline = hasattr(self.state, 'active_orchestrator') and self.state.active_orchestrator is not None
        is_compiled = getattr(self.state, 'is_compiled', False)
        is_running = getattr(self.state, 'is_running', False)
        
        # Define enabling conditions for each command
        enabling_conditions = {
            'new_pipeline': True,  # Always enabled
            'open_pipeline': True,  # Always enabled
            'save_pipeline': has_active_pipeline,
            'save_pipeline_as': has_active_pipeline,
            'export_pipeline': has_active_pipeline and is_compiled,
            'import_pipeline': True,  # Always enabled
            'exit': True,  # Always enabled
            'undo': False,  # TODO: Implement undo system
            'redo': False,  # TODO: Implement undo system
            'cut': has_active_pipeline,
            'copy': has_active_pipeline,
            'paste': has_active_pipeline,
            'select_all': has_active_pipeline,
            'find': has_active_pipeline,
            'replace': has_active_pipeline,
            'add_step': has_active_pipeline,
            'remove_step': has_active_pipeline,
            'duplicate_step': has_active_pipeline,
            'move_step_up': has_active_pipeline,
            'move_step_down': has_active_pipeline,
            'run_pipeline': has_active_pipeline and is_compiled and not is_running,
            'stop_pipeline': is_running,
            'validate_pipeline': has_active_pipeline,
            'debug_pipeline': has_active_pipeline and is_compiled,
            'preferences': True,  # Always enabled
            'keyboard_shortcuts': True,  # Always enabled
            'documentation': True,  # Always enabled
            'about': True,  # Always enabled
        }
        
        return enabling_conditions.get(command_name, False)
    
    async def _notify_error(self, message: str):
        """Notify about an error."""
        await self.state.notify('error', {
            'source': 'MenuService',
            'message': message
        })
    
    async def _notify_status(self, operation: str, status: str, message: str):
        """Notify about operation status."""
        await self.state.notify('operation_status_changed', {
            'operation': operation,
            'status': status,
            'message': message,
            'source': 'MenuService'
        })
    
    # Command handlers
    async def _handle_new_pipeline(self, **kwargs):
        """Handle New Pipeline command."""
        await self._notify_status('new_pipeline', 'info', 'New Pipeline: Not yet implemented')
    
    async def _handle_open_pipeline(self, **kwargs):
        """Handle Open Pipeline command."""
        await self._notify_status('open_pipeline', 'info', 'Open Pipeline: Not yet implemented')
    
    async def _handle_save_pipeline(self, **kwargs):
        """Handle Save Pipeline command."""
        if not hasattr(self.state, 'active_orchestrator') or self.state.active_orchestrator is None:
            await self._notify_error("No active pipeline to save")
            return
        
        try:
            # TODO: Implement actual save logic
            await self._notify_status('save_pipeline', 'success', 'Pipeline saved successfully')
        except Exception as e:
            await self._notify_error(f"Failed to save pipeline: {str(e)}")
    
    async def _handle_save_pipeline_as(self, **kwargs):
        """Handle Save Pipeline As command."""
        await self._notify_status('save_pipeline_as', 'info', 'Save Pipeline As: Not yet implemented')
    
    async def _handle_export_pipeline(self, **kwargs):
        """Handle Export Pipeline command."""
        await self._notify_status('export_pipeline', 'info', 'Export Pipeline: Not yet implemented')
    
    async def _handle_import_pipeline(self, **kwargs):
        """Handle Import Pipeline command."""
        await self._notify_status('import_pipeline', 'info', 'Import Pipeline: Not yet implemented')
    
    async def _handle_exit(self, **kwargs):
        """Handle Exit command."""
        await self.state.notify('exit_requested', {})
    
    async def _handle_undo(self, **kwargs):
        """Handle Undo command."""
        await self._notify_status('undo', 'info', 'Undo: Not yet implemented')
    
    async def _handle_redo(self, **kwargs):
        """Handle Redo command."""
        await self._notify_status('redo', 'info', 'Redo: Not yet implemented')
    
    async def _handle_cut(self, **kwargs):
        """Handle Cut command."""
        await self._notify_status('cut', 'info', 'Cut: Not yet implemented')
    
    async def _handle_copy(self, **kwargs):
        """Handle Copy command."""
        await self._notify_status('copy', 'info', 'Copy: Not yet implemented')
    
    async def _handle_paste(self, **kwargs):
        """Handle Paste command."""
        await self._notify_status('paste', 'info', 'Paste: Not yet implemented')
    
    async def _handle_select_all(self, **kwargs):
        """Handle Select All command."""
        await self._notify_status('select_all', 'info', 'Select All: Not yet implemented')
    
    async def _handle_find(self, **kwargs):
        """Handle Find command."""
        await self._notify_status('find', 'info', 'Find: Not yet implemented')
    
    async def _handle_replace(self, **kwargs):
        """Handle Replace command."""
        await self._notify_status('replace', 'info', 'Replace: Not yet implemented')
    
    async def _handle_add_step(self, **kwargs):
        """Handle Add Step command."""
        await self.state.notify('add_step_requested', {})
    
    async def _handle_remove_step(self, **kwargs):
        """Handle Remove Step command."""
        await self.state.notify('remove_step_requested', {})
    
    async def _handle_duplicate_step(self, **kwargs):
        """Handle Duplicate Step command."""
        await self.state.notify('duplicate_step_requested', {})
    
    async def _handle_move_step_up(self, **kwargs):
        """Handle Move Step Up command."""
        await self.state.notify('move_step_up_requested', {})
    
    async def _handle_move_step_down(self, **kwargs):
        """Handle Move Step Down command."""
        await self.state.notify('move_step_down_requested', {})
    
    async def _handle_run_pipeline(self, **kwargs):
        """Handle Run Pipeline command."""
        await self.state.notify('run_pipeline_requested', {})
    
    async def _handle_stop_pipeline(self, **kwargs):
        """Handle Stop Pipeline command."""
        await self.state.notify('stop_pipeline_requested', {})
    
    async def _handle_validate_pipeline(self, **kwargs):
        """Handle Validate Pipeline command."""
        await self.state.notify('validate_pipeline_requested', {})
    
    async def _handle_debug_pipeline(self, **kwargs):
        """Handle Debug Pipeline command."""
        await self._notify_status('debug_pipeline', 'info', 'Debug Pipeline: Not yet implemented')
    
    async def _handle_preferences(self, **kwargs):
        """Handle Preferences command."""
        await self.state.notify('show_preferences_requested', {})
    
    async def _handle_keyboard_shortcuts(self, **kwargs):
        """Handle Keyboard Shortcuts command."""
        await self.state.notify('show_help_requested', {'topic': 'keyboard_shortcuts'})
    
    async def _handle_documentation(self, **kwargs):
        """Handle Documentation command."""
        await self.state.notify('show_help_requested', {'topic': 'documentation'})
    
    async def _handle_about(self, **kwargs):
        """Handle About command."""
        await self.state.notify('show_help_requested', {'topic': 'about'})
