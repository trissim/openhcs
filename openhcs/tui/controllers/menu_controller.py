"""
Menu Controller - UI Coordination Layer.

Coordinates between menu UI components and business logic services.
Handles user interactions and state synchronization.

🔒 Clause 295: Component Boundaries
Clear separation between controller, service, and view layers.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from prompt_toolkit.application import get_app

from openhcs.tui.services.menu_service import MenuService, MenuItemType

logger = logging.getLogger(__name__)


class MenuController:
    """
    Controller for menu UI operations.
    
    Coordinates between:
    - MenuService (business logic)
    - MenuView (UI rendering)
    - TUIState (application state)
    """
    
    def __init__(self, state, service: MenuService):
        self.state = state
        self.service = service
        
        # UI state
        self.is_menu_open = False
        self.current_menu_name = None
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register event handlers with the state manager."""
        self.state.add_observer('operation_status_changed', self._handle_operation_status_changed)
        self.state.add_observer('plate_selected', self._handle_plate_selected)
        self.state.add_observer('is_compiled_changed', self._handle_is_compiled_changed)
    
    async def _handle_operation_status_changed(self, data):
        """Handle operation status change events."""
        # Force UI refresh to update menu item enabling
        app = get_app()
        app.invalidate()
    
    async def _handle_plate_selected(self, data):
        """Handle plate selection events."""
        # Force UI refresh to update menu item enabling
        app = get_app()
        app.invalidate()
    
    async def _handle_is_compiled_changed(self, data):
        """Handle compilation status change events."""
        # Force UI refresh to update menu item enabling
        app = get_app()
        app.invalidate()
    
    async def activate_menu(self, menu_name: str):
        """
        Activate a menu category.
        
        Args:
            menu_name: Name of the menu to activate
        """
        try:
            await self.service.set_active_menu(menu_name)
            self.is_menu_open = True
            self.current_menu_name = menu_name
            
            # Notify UI to show submenu
            await self.state.notify('menu_activated', {
                'menu_name': menu_name,
                'is_open': True
            })
            
            # Force UI refresh
            app = get_app()
            app.invalidate()
            
        except Exception as e:
            logger.error(f"Error activating menu '{menu_name}': {e}", exc_info=True)
            await self._handle_error("Error activating menu", str(e))
    
    async def close_menu(self):
        """Close the currently active menu."""
        try:
            await self.service.set_active_menu(None)
            self.is_menu_open = False
            self.current_menu_name = None
            
            # Notify UI to hide submenu
            await self.state.notify('menu_closed', {})
            
            # Force UI refresh
            app = get_app()
            app.invalidate()
            
        except Exception as e:
            logger.error(f"Error closing menu: {e}", exc_info=True)
            await self._handle_error("Error closing menu", str(e))
    
    async def navigate_menu(self, delta: int):
        """
        Navigate between top-level menus.
        
        Args:
            delta: Direction to navigate (-1 for left, 1 for right)
        """
        if not self.is_menu_open or not self.current_menu_name:
            return
        
        # This would need access to the menu structure
        # For now, just notify the UI to handle navigation
        await self.state.notify('menu_navigation_requested', {
            'direction': delta,
            'current_menu': self.current_menu_name
        })
    
    async def navigate_submenu(self, delta: int):
        """
        Navigate within a submenu.
        
        Args:
            delta: Direction to navigate (-1 for up, 1 for down)
        """
        if not self.is_menu_open:
            return
        
        # Get current menu state
        menu_state = await self.service.get_menu_state()
        current_index = menu_state.get('active_item_index', 0)
        
        # This would need access to the submenu items
        # For now, just notify the UI to handle navigation
        await self.state.notify('submenu_navigation_requested', {
            'direction': delta,
            'current_index': current_index
        })
    
    async def select_menu_item(self, command_name: str):
        """
        Select a menu item and execute its command.
        
        Args:
            command_name: Name of the command to execute
        """
        try:
            # Close menu first
            await self.close_menu()
            
            # Execute command through service
            success = await self.service.execute_command(command_name)
            
            if success:
                logger.info(f"Successfully executed command: {command_name}")
            else:
                logger.warning(f"Failed to execute command: {command_name}")
                
        except Exception as e:
            logger.error(f"Error selecting menu item '{command_name}': {e}", exc_info=True)
            await self._handle_error(f"Error executing command '{command_name}'", str(e))
    
    def is_command_enabled(self, command_name: str) -> bool:
        """
        Check if a command is currently enabled.
        
        Args:
            command_name: Name of the command to check
            
        Returns:
            True if command is enabled, False otherwise
        """
        return self.service.is_command_enabled(command_name)
    
    async def handle_key_binding(self, key_sequence: str):
        """
        Handle keyboard shortcuts for menu activation.
        
        Args:
            key_sequence: The key sequence that was pressed
        """
        # Map key sequences to menu names
        key_to_menu = {
            'f': 'File',
            'e': 'Edit',
            'p': 'Pipeline',
            'r': 'Run',
            't': 'Tools',
            'h': 'Help'
        }
        
        menu_name = key_to_menu.get(key_sequence.lower())
        if menu_name:
            await self.activate_menu(menu_name)
    
    async def handle_mouse_click(self, menu_name: str):
        """
        Handle mouse clicks on menu items.
        
        Args:
            menu_name: Name of the menu that was clicked
        """
        if self.is_menu_open and self.current_menu_name == menu_name:
            # If same menu is already open, close it
            await self.close_menu()
        else:
            # Open the clicked menu
            await self.activate_menu(menu_name)
    
    async def _handle_error(self, message: str, details: str = ""):
        """Handle errors by notifying the state."""
        logger.error(f"MenuController error: {message} - {details}")
        await self.state.notify('error', {
            'source': 'MenuController',
            'message': message,
            'details': details
        })
    
    def get_menu_state(self) -> Dict[str, Any]:
        """Get the current menu UI state."""
        return {
            'is_menu_open': self.is_menu_open,
            'current_menu_name': self.current_menu_name
        }
    
    async def shutdown(self):
        """Clean up resources."""
        logger.info("MenuController: Shutting down...")
        
        # Unregister event handlers
        self.state.remove_observer('operation_status_changed', self._handle_operation_status_changed)
        self.state.remove_observer('plate_selected', self._handle_plate_selected)
        self.state.remove_observer('is_compiled_changed', self._handle_is_compiled_changed)
        
        # Close any open menus
        if self.is_menu_open:
            await self.close_menu()
        
        logger.info("MenuController: Shutdown complete")
