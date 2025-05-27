"""
Plate Manager Controller for OpenHCS TUI.

This module defines the PlateManagerController class, which orchestrates
the plate list view and plate actions toolbar. It responds to TUIState
changes and manages interactions with services like dialogs and validation.
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, Dimension, Window, Container
from prompt_toolkit.widgets import Box, Label, Frame

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.tui.interfaces import CoreApplicationAdapterInterface, CoreOrchestratorAdapterInterface, CorePlateData
    from ..components.plate_list_view import PlateListView
    from ..components.plate_actions_toolbar import PlateActionsToolbar
    from openhcs.tui.dialogs.plate_dialog_manager import PlateDialogManager # For type hinting
    from openhcs.tui.services.plate_validation import PlateValidationService # For type hinting


logger = logging.getLogger(__name__)

class PlateManagerController:
    """
    Orchestrates PlateListView and PlateActionsToolbar.
    Manages plate-related state and interactions.
    """
    def __init__(self,
                 ui_state: 'TUIState',
                 app_adapter: 'CoreApplicationAdapterInterface',
                 async_ui_manager: 'AsyncUIManager', # Added AsyncUIManager
                 plate_list_view_class: type['PlateListView'], 
                 plate_actions_toolbar_class: type['PlateActionsToolbar']
                ):
        self.ui_state = ui_state
        self.app_adapter = app_adapter
        self.async_ui_manager = async_ui_manager # Store AsyncUIManager
        self._lock = asyncio.Lock()

        # Instantiate View and Toolbar
        self.plate_list_view = plate_list_view_class(
            on_plate_selected=self._handle_plate_list_selection, # Controller handles selection from view
            on_plate_activated=self._handle_plate_list_activation # Controller handles activation
        )
        self.plate_actions_toolbar = plate_actions_toolbar_class(
            ui_state=self.ui_state,
            app_adapter=self.app_adapter,
            get_current_plate_adapter=self._get_current_plate_adapter_for_toolbar
        )

        # Services (DialogManager, ValidationService)
        # These were in the old PlateManagerPane, now managed by controller
        self.dialog_manager: Optional['PlateDialogManager'] = None
        self.validation_service: Optional['PlateValidationService'] = None
        self._initialize_services() # Initialize services

        # Main container for this controller's view components
        # This is what OpenHCSTUI (via LayoutManager) will display for the "Plate Manager" area.
        self.container = Frame(
            HSplit([
                self.plate_actions_toolbar.container, # Toolbar at the top
                Window(height=1, char='-'), # Separator
                self.plate_list_view.container  # List view below
            ]),
            title="Plate Manager" # Title for the frame
        )
        
        self._ui_initialized = False # To track if initial data load has occurred.

        # Register observers for TUIState changes
        self.ui_state.add_observer('available_plates_updated', self._on_available_plates_updated)
        self.ui_state.add_observer('active_plate_id_changed', self._on_active_plate_id_changed) # New event from TUIState
        self.ui_state.add_observer('plate_status_changed', self._on_plate_status_changed)
        # For dialog requests that might be triggered globally but handled here
        self.ui_state.add_observer('ui_request_show_add_plate_dialog', self._handle_global_add_plate_dialog_request)


    async def initialize_controller(self):
        """
        Perform any asynchronous initialization needed for the controller,
        including fetching initial data.
        """
        logger.info("PlateManagerController: Initializing...")
        # _refresh_plates_from_core is awaited, so it blocks initialize_controller.
        # If it were truly fire-and-forget at this stage, we'd use:
        # self.async_ui_manager.fire_and_forget(self._refresh_plates_from_core(), name="InitialPlateRefresh")
        await self._refresh_plates_from_core()
        self._ui_initialized = True
        logger.info("PlateManagerController: Initialization complete.")

    def _initialize_services(self):
        """Initialize dialog manager and validation service."""
        from openhcs.tui.dialogs.plate_dialog_manager import PlateDialogManager
        from openhcs.tui.services.plate_validation import PlateValidationService
        
        self.dialog_manager = PlateDialogManager(
            on_add_dialog_result=self._handle_add_dialog_result,
            on_remove_dialog_result=self._handle_remove_dialog_result,
            on_error=self._handle_service_error,
            app_adapter=self.app_adapter
            # If dialog_manager needs to run its own async tasks not tied to user interaction:
            # async_ui_manager=self.async_ui_manager 
        )
        self.validation_service = PlateValidationService(
            app_adapter=self.app_adapter,
            on_validation_result=self._handle_validation_result,
            on_error=self._handle_service_error,
            io_executor=None, # This service might use its own executor or could use async_ui_manager for its tasks
            # async_ui_manager=self.async_ui_manager # If it needs to manage asyncio tasks
        )
        logger.info("PlateManagerController: Services initialized.")

    async def _get_current_plate_adapter_for_toolbar(self) -> Optional['CoreOrchestratorAdapterInterface']:
        """Provides the current plate adapter to the toolbar."""
        if self.ui_state.active_plate_id:
            try:
                return await self.app_adapter.get_orchestrator_adapter(self.ui_state.active_plate_id)
            except Exception as e:
                logger.error(f"Error getting plate adapter for toolbar: {e}")
                await self.ui_state.notify("error", {"message": f"Plate functions unavailable: {e}"})
        return None

    # --- TUIState Observer Handlers ---
    async def _on_available_plates_updated(self, plates_core_data: List['CorePlateData']):
        """Handles updates to the list of available plates from TUIState."""
        logger.debug(f"PlateManagerController: Received {len(plates_core_data)} plates from TUIState.")
        # Pass CorePlateData (as dicts) to the view
        await self.plate_list_view.update_plate_list([dict(p) for p in plates_core_data])
        self.plate_actions_toolbar.update_button_states() # Update button enable/disable

    async def _on_active_plate_id_changed(self, active_plate_id: Optional[str]):
        """Handles changes to TUIState.active_plate_id."""
        logger.debug(f"PlateManagerController: Active plate ID changed to: {active_plate_id}")
        await self.plate_list_view.set_selected_plate_by_id(active_plate_id)
        self.plate_actions_toolbar.update_button_states()

    async def _on_plate_status_changed(self, plate_status_data: Dict[str, Any]):
        """Handles plate status changes from TUIState."""
        # The PlateListView itself might not need to know about every status detail,
        # but its display text function uses it. Rebuilding the list or specific item is needed.
        # For simplicity, if available_plates_data from TUIState includes updated status,
        # then _on_available_plates_updated will refresh the view.
        # If more granular update is needed, PlateListView could have an update_item_status method.
        logger.debug(f"PlateManagerController: Plate status changed: {plate_status_data.get('id')}")
        # Re-fetch the full list from TUIState to ensure consistency, or update specific item
        # This assumes TUIState.available_plates_data is the source of truth and is updated by the core interaction.
        if self.ui_state.available_plates_data: # Type hint as List[CorePlateData]
            await self.plate_list_view.update_plate_list([dict(p) for p in self.ui_state.available_plates_data])
        self.plate_actions_toolbar.update_button_states()


    # --- View Event Handlers (Callbacks from PlateListView) ---
    async def _handle_plate_list_selection(self, selected_plate_data: Optional[Dict[str, Any]]):
        """Handles a plate being selected in the PlateListView."""
        new_active_plate_id = selected_plate_data.get('id') if selected_plate_data else None
        logger.debug(f"PlateManagerController: Selection from view, plate ID: {new_active_plate_id}")
        
        # Update TUIState.active_plate_id. This will trigger _on_active_plate_id_changed.
        # It will also trigger PipelineEditorPane to load steps for this plate.
        if self.ui_state.active_plate_id != new_active_plate_id:
            await self.ui_state.set_active_plate_id(new_active_plate_id) # Assumed method on TUIState
            # If set_active_plate_id doesn't notify 'active_plate_id_changed', do it manually:
            # await self.ui_state.notify('active_plate_id_changed', new_active_plate_id)
        
        # Update toolbar button states based on new selection
        self.plate_actions_toolbar.update_button_states()


    async def _handle_plate_list_activation(self, activated_plate_data: Dict[str, Any]):
        """Handles a plate being "activated" (e.g., Enter pressed) in the PlateListView."""
        # Currently, activation might be the same as selection, or could trigger
        # a default action like "Initialize" if not initialized, or "Edit Config".
        logger.info(f"PlateManagerController: Plate activated: {activated_plate_data.get('name')}")
        # For now, ensure it's selected. Future: trigger default action.
        await self._handle_plate_list_selection(activated_plate_data)
        # Example: Trigger "Edit Config" if already selected and activated
        # from ..commands import ShowEditPlateConfigDialogCommand # Local import
        # cmd = ShowEditPlateConfigDialogCommand()
        # if cmd.can_execute(self.ui_state):
        #    plate_adapter = await self._get_current_plate_adapter_for_toolbar()
        #    await cmd.execute(self.app_adapter, plate_adapter, self.ui_state)


    # --- Service Callback Handlers (DialogManager, ValidationService) ---
    async def _handle_add_dialog_result(self, result: Dict[str, Any]):
        """Handles paths from PlateDialogManager by dispatching AddPlateByPathCommand."""
        if not result or not result.get('paths'):
            logger.info("PlateManagerController: Add plate dialog cancelled or no paths.")
            return
        
        from ..commands import AddPlateByPathCommand # Local import for command
        
        paths_to_add = result.get('paths', [])
        backend_identifier = result.get('backend', 'default_storage') # Dialog should provide this

        for path_str in paths_to_add:
            logger.info(f"PlateManagerController: Requesting to add plate from path: {path_str}")
            cmd = AddPlateByPathCommand(path_str=path_str, backend_identifier=backend_identifier)
            try:
                await cmd.execute(app_adapter=self.app_adapter, plate_adapter=None, ui_state=self.ui_state)
                # Successful execution of command should lead to core updating data,
                # TUIState being updated, and then _on_available_plates_updated being called.
            except Exception as e:
                logger.error(f"Error executing AddPlateByPathCommand for {path_str}: {e}", exc_info=True)
                await self._handle_service_error(f"Failed to process plate addition for: {Path(path_str).name}", str(e))

    async def _handle_remove_dialog_result(self, result: Dict[str, Any]):
        """Handles plate ID from PlateDialogManager for removal by dispatching DeletePlateByIdCommand."""
        # This is for single-item removal via dialog. Multi-delete is handled by DeleteSelectedPlatesCommand.
        if not result or 'plate_id' not in result:
            logger.info("PlateManagerController: Remove plate dialog cancelled or no plate ID.")
            return
        
        from ..commands import DeletePlateByIdCommand # Local import for command
        plate_id_to_remove = result['plate_id']
        logger.info(f"PlateManagerController: Requesting to remove plate ID: {plate_id_to_remove}")
        cmd = DeletePlateByIdCommand(plate_id=plate_id_to_remove)
        try:
            await cmd.execute(app_adapter=self.app_adapter, plate_adapter=None, ui_state=self.ui_state)
        except Exception as e:
            logger.error(f"Error executing DeletePlateByIdCommand for {plate_id_to_remove}: {e}", exc_info=True)
            await self._handle_service_error(f"Failed to delete plate ID: {plate_id_to_remove}", str(e))


    async def _handle_validation_result(self, validation_data: Dict[str, Any]):
        """Handles validation result from PlateValidationService."""
        if not validation_data:
            logger.error("PlateManagerController: Validation result is empty.")
            return

        plate_path = validation_data.get('path')
        if not validation_data.get('is_valid', False):
            error_details = validation_data.get('error_details', f"Plate at {plate_path} failed validation.")
            await self._handle_service_error(f"Plate validation failed: {Path(plate_path).name if plate_path else 'Unknown'}", error_details)
            return

        logger.info(f"PlateManagerController: Plate at '{plate_path}' validated. Confirming with core.")
        from ..commands import ConfirmValidatedPlateCommand # Local import
        cmd = ConfirmValidatedPlateCommand(validated_plate_info=validation_data)
        try:
            await cmd.execute(app_adapter=self.app_adapter, plate_adapter=None, ui_state=self.ui_state)
        except Exception as e:
            logger.error(f"Error executing ConfirmValidatedPlateCommand for {plate_path}: {e}", exc_info=True)
            await self._handle_service_error(f"Failed to confirm plate: {Path(plate_path).name if plate_path else 'Unknown'}", str(e))

    async def _handle_service_error(self, message: str, details: str = ""):
        """Handles errors reported by services."""
        logger.error(f"PlateManagerController Service Error: {message} - Details: {details}")
        await self.ui_state.notify('error', {'source': 'PlateManagerController.Service', 'message': message, 'details': details})

    # --- Global Event Handlers (e.g., from Menu or other global actions) ---
    async def _handle_global_add_plate_dialog_request(self, data: Optional[Dict[str, Any]] = None):
        """Handles global request to show the add plate dialog."""
        logger.info("PlateManagerController: Global request to show Add Plate dialog.")
        if self.dialog_manager:
            await self.dialog_manager.show_add_plate_dialog() # Assumes dialog_manager uses app_adapter for file listing
        else:
            logger.error("PlateManagerController: DialogManager not available for Add Plate request.")


    # --- Core Data Refresh ---
    async def _refresh_plates_from_core(self):
        """Fetches available plates from core and updates TUIState."""
        logger.info("PlateManagerController: Refreshing plates from core...")
        if not self.ui_state: return # Should not happen

        # Indicate loading state (e.g., for PlateListView)
        # await self.ui_state.notify('plate_list_loading_started')
        # For now, PlateListView can show its own loading message if list is empty.
        try:
            plates_core_data: List['CorePlateData'] = await self.app_adapter.list_available_plates()
            await self.ui_state.set_available_plates_data(plates_core_data) # This will trigger observers
        except Exception as e:
            logger.error(f"PlateManagerController: Error refreshing plates from core: {e}", exc_info=True)
            await self._handle_service_error("Failed to load plates from core.", str(e))
            await self.ui_state.set_available_plates_data([]) # Clear list on error
        # finally:
            # await self.ui_state.notify('plate_list_loading_finished')
            
    async def shutdown(self):
        """Performs cleanup of resources."""
        logger.info("PlateManagerController: Shutting down...")
        # Unregister observers
        self.ui_state.remove_observer('available_plates_updated', self._on_available_plates_updated)
        self.ui_state.remove_observer('active_plate_id_changed', self._on_active_plate_id_changed)
        self.ui_state.remove_observer('plate_status_changed', self._on_plate_status_changed)
        self.ui_state.remove_observer('ui_request_show_add_plate_dialog', self._handle_global_add_plate_dialog_request)

        if hasattr(self.validation_service, 'close') and self.validation_service:
             if asyncio.iscoroutinefunction(self.validation_service.close): # type: ignore
                await self.validation_service.close() # type: ignore
             else:
                self.validation_service.close() # type: ignore
        logger.info("PlateManagerController: Shutdown complete.")

    def get_container(self) -> Container:
        """Returns the root container for this controller's managed view."""
        return self.container
