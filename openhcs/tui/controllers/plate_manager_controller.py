"""
Plate Manager UI Controller.

Coordinates between UI components and business logic services.
Handles user interactions and state synchronization.

🔒 Clause 295: Component Boundaries
Clear separation between controller, service, and view layers.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from prompt_toolkit.application import get_app

from openhcs.tui.services.plate_manager_service import PlateManagerService
from openhcs.tui.services.plate_validation import PlateValidationService
from openhcs.tui.dialogs.plate_dialog_manager import PlateDialogManager

logger = logging.getLogger(__name__)


class PlateManagerController:
    """
    Controller for plate manager UI operations.
    
    Coordinates between:
    - PlateManagerService (business logic)
    - PlateValidationService (validation)
    - PlateDialogManager (UI dialogs)
    - TUIState (application state)
    """
    
    def __init__(self, state, service: PlateManagerService, validation_service: PlateValidationService):
        self.state = state
        self.service = service
        self.validation_service = validation_service
        
        # UI state
        self.selected_index = 0
        self.is_loading = False
        
        # Initialize dialog manager
        self.dialog_manager = PlateDialogManager(state)
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register event handlers with the state manager."""
        self.state.add_observer('refresh_plates', self._handle_refresh_request)
        self.state.add_observer('plate_status_changed', self._handle_plate_status_changed)
        self.state.add_observer('ui_request_show_add_plate_dialog', self._handle_show_add_dialog_request)
        self.state.add_observer('add_predefined_plate', self._handle_add_predefined_plate)
    
    async def _handle_refresh_request(self, data=None):
        """Handle plate refresh requests."""
        app = get_app()
        app.create_background_task(self.refresh_plates())
    
    async def _handle_plate_status_changed(self, data):
        """Handle plate status change events."""
        plate_id = data.get('plate_id')
        new_status = data.get('status')
        error_message = data.get('error_message')
        
        if not plate_id or not new_status:
            return
            
        await self.service.update_plate_status(plate_id, new_status, error_message)
        await self._notify_ui_update()
    
    async def _handle_show_add_dialog_request(self, data=None):
        """Handle requests to show the add plate dialog."""
        app = get_app()
        app.create_background_task(self.show_add_plate_dialog())
    
    async def _handle_add_predefined_plate(self, data=None):
        """Handle adding a predefined plate."""
        if not data or 'path' not in data or 'backend' not in data:
            logger.error("Received 'add_predefined_plate' event with missing data.")
            await self._handle_error("Invalid data for predefined plate.", f"Received: {data}")
            return
            
        path = data['path']
        backend = data['backend']
        
        try:
            await self.validation_service.validate_plate(path, backend)
            logger.info(f"Validation initiated for predefined plate '{path}'.")
        except Exception as e:
            await self._handle_error(f"Error validating predefined plate '{path}'", str(e))
    
    async def show_add_plate_dialog(self):
        """Show the add plate dialog."""
        try:
            result = await self.dialog_manager.show_add_plate_dialog()
            if result:
                await self._handle_add_dialog_result(result)
        except Exception as e:
            await self._handle_error("Error showing add plate dialog", str(e))
    
    async def _handle_add_dialog_result(self, result: Dict[str, Any]):
        """Handle the result of the add plate dialog."""
        if not result or not result.get('paths'):
            logger.info("Add plate dialog cancelled or no paths provided.")
            return

        paths = result.get('paths', [])
        backend = result.get('backend', 'disk')

        if not paths:
            logger.info("No paths provided in add plate dialog result.")
            return

        # Validate each path
        for path in paths:
            try:
                await self.validation_service.validate_plate(path, backend)
            except Exception as e:
                await self._handle_error(f"Error validating plate '{path}'", str(e))

        logger.info(f"Submitted {len(paths)} paths for validation.")
    
    async def remove_selected_plates(self, plate_ids: List[str]):
        """Remove the specified plates."""
        try:
            num_removed = await self.service.remove_plates(plate_ids)
            if num_removed > 0:
                await self._notify_ui_update()
                await self.state.notify('plates_removed', {
                    'plate_ids': plate_ids,
                    'count': num_removed
                })
        except Exception as e:
            await self._handle_error("Error removing plates", str(e))
    
    async def refresh_plates(self):
        """Refresh the plate list."""
        logger.info("Refreshing plates...")
        await self._set_loading_state(True)
        
        try:
            common_output_directory = getattr(self.service.context, 'common_output_directory', None)
            if not common_output_directory:
                await self._handle_error("Output directory not set", "No common_output_directory in context")
                return
            
            await self.service.refresh_plates_from_directory(common_output_directory)
            await self._notify_ui_update()
            
        except Exception as e:
            await self._handle_error("Failed to refresh plates", str(e))
        finally:
            await self._set_loading_state(False)
    
    async def _set_loading_state(self, is_loading: bool):
        """Set loading state and notify UI."""
        self.is_loading = is_loading
        await self._notify_ui_update()
    
    async def _notify_ui_update(self):
        """Notify that the UI should be updated."""
        await self.state.notify('plate_manager_ui_update', {
            'plates': await self.service.get_plates(),
            'selected_index': self.selected_index,
            'is_loading': self.is_loading
        })
    
    async def _handle_error(self, message: str, details: str = ""):
        """Handle errors by notifying the state."""
        logger.error(f"PlateManagerController error: {message} - {details}")
        await self.state.notify('error', {
            'source': 'PlateManagerController',
            'message': message,
            'details': details
        })
    
    async def select_plate(self, index: int):
        """Select a plate by index."""
        plates = await self.service.get_plates()
        if 0 <= index < len(plates):
            self.selected_index = index
            selected_plate = plates[index]
            
            # Notify state about selection
            await self.state.notify('plate_selected', {
                'plate': selected_plate,
                'index': index
            })
            
            await self._notify_ui_update()
    
    async def get_selected_plate(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected plate."""
        plates = await self.service.get_plates()
        if 0 <= self.selected_index < len(plates):
            return plates[self.selected_index]
        return None
    
    async def shutdown(self):
        """Clean up resources."""
        logger.info("PlateManagerController: Shutting down...")
        await self.service.shutdown()
        logger.info("PlateManagerController: Shutdown complete.")
