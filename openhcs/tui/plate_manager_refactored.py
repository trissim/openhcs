"""
Refactored Plate Manager Pane - Clean Architecture.

This module provides a facade for the plate manager functionality,
using the new MVC architecture with separated concerns.

🔒 Clause 295: Component Boundaries
Clear separation between service, controller, and view layers.
"""
import asyncio
import logging
from typing import Any

from prompt_toolkit.layout import Container

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.tui.services.plate_manager_service import PlateManagerService
from openhcs.tui.services.plate_validation import PlateValidationService
from openhcs.tui.controllers.plate_manager_controller import PlateManagerController
from openhcs.tui.views.plate_manager_view import PlateManagerView

logger = logging.getLogger(__name__)


class PlateManagerPane:
    """
    Refactored Plate Manager Pane using clean architecture.
    
    This class serves as a facade that coordinates the MVC components:
    - PlateManagerService: Business logic and data management
    - PlateManagerController: User interaction coordination
    - PlateManagerView: UI rendering and input handling
    """
    
    def __init__(self, state, context: ProcessingContext, storage_registry: Any):
        """
        Initialize the plate manager with clean architecture.
        
        Args:
            state: TUI state manager
            context: Processing context with filemanager
            storage_registry: Storage registry for orchestrator creation
        """
        self.state = state
        self.context = context
        self.storage_registry = storage_registry
        
        # Initialize service layer
        self.service = PlateManagerService(context, storage_registry)
        
        # Initialize validation service
        self.validation_service = PlateValidationService(state, context)
        
        # Initialize controller
        self.controller = PlateManagerController(state, self.service, self.validation_service)
        
        # Initialize view
        self.view = PlateManagerView(self.controller)
        
        # Set up validation result handling
        self.validation_service.set_result_handler(self._handle_validation_result)
        
        logger.info("PlateManagerPane: Initialized with clean architecture")
    
    async def _handle_validation_result(self, data):
        """Handle validation results from the validation service."""
        is_valid = data.get('is_valid', False)
        path = data.get('path')
        backend = data.get('backend', 'disk')
        
        if is_valid:
            try:
                # Add the validated plate
                plate = await self.service.add_plate(path, self.state.global_config)
                
                # Notify state about the new plate
                await self.state.notify('plate_added', plate)
                
                logger.info(f"Successfully added validated plate: {plate['name']}")
                
            except Exception as e:
                logger.error(f"Error adding validated plate '{path}': {e}", exc_info=True)
                await self.controller._handle_error(f"Error adding plate '{path}'", str(e))
        else:
            error_details = data.get('error_details', "Unknown validation error")
            await self.controller._handle_error(f"Failed to validate plate: {path}", error_details)
    
    def get_container(self) -> Container:
        """Get the UI container for this pane."""
        return self.view.get_container()
    
    def register_with_app(self):
        """Register event handlers with the application."""
        # The controller already registers its own event handlers
        logger.info("PlateManagerPane: Event handlers registered via controller")
    
    async def initialize_and_refresh(self):
        """Initialize the pane and perform initial refresh."""
        try:
            await self.controller.refresh_plates()
            logger.info("PlateManagerPane: Initial refresh completed")
        except Exception as e:
            logger.error(f"PlateManagerPane: Error during initial refresh: {e}", exc_info=True)
            await self.controller._handle_error("Error during initialization", str(e))
    
    def handle_key(self, key_event):
        """Handle keyboard input."""
        self.view.handle_key(key_event)
    
    async def shutdown(self):
        """Clean up resources."""
        logger.info("PlateManagerPane: Shutting down...")
        await self.controller.shutdown()
        logger.info("PlateManagerPane: Shutdown complete")
    
    # Legacy compatibility methods for existing code
    @property
    def plates(self):
        """Legacy property for backward compatibility."""
        # This should be replaced with async calls in the future
        return []
    
    @property
    def selected_index(self):
        """Legacy property for backward compatibility."""
        return self.controller.selected_index
    
    async def get_selected_plate(self):
        """Get the currently selected plate."""
        return await self.controller.get_selected_plate()
    
    async def refresh_plates(self):
        """Refresh the plate list."""
        await self.controller.refresh_plates()
    
    async def add_plate_from_path(self, path: str, backend: str = 'disk'):
        """Add a plate from a specific path."""
        try:
            await self.validation_service.validate_plate(path, backend)
        except Exception as e:
            await self.controller._handle_error(f"Error adding plate from path '{path}'", str(e))
    
    async def remove_selected_plates(self, plate_ids=None):
        """Remove selected plates."""
        if plate_ids is None:
            # Remove currently selected plate
            selected_plate = await self.get_selected_plate()
            if selected_plate:
                plate_ids = [selected_plate.get('id')]
            else:
                return
        
        await self.controller.remove_selected_plates(plate_ids)


# For backward compatibility, create an alias
PlateManagerCore = PlateManagerPane
