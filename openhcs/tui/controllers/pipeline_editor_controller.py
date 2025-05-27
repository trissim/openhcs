"""
Pipeline Editor Controller for OpenHCS TUI.

This module defines the PipelineEditorController class, which orchestrates
the StepListView and PipelineActionsToolbar. It responds to TUIState
changes and manages interactions related to pipeline editing.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, Dimension, Window, Container
from prompt_toolkit.widgets import Box, Label, Frame

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.tui.interfaces import CoreApplicationAdapterInterface, CoreOrchestratorAdapterInterface, CoreStepData
    from ..components.step_list_view import StepListView
    from ..components.pipeline_actions_toolbar import PipelineActionsToolbar
    from openhcs.tui.async_manager import AsyncUIManager # Import AsyncUIManager

logger = logging.getLogger(__name__)

class PipelineEditorController:
    """
    Orchestrates StepListView and PipelineActionsToolbar.
    Manages pipeline-related state and interactions.
    """
    def __init__(self,
                 ui_state: 'TUIState',
                 app_adapter: 'CoreApplicationAdapterInterface',
                 async_ui_manager: 'AsyncUIManager', # Added AsyncUIManager
                 step_list_view_class: type['StepListView'],
                 pipeline_actions_toolbar_class: type['PipelineActionsToolbar']
                ):
        self.ui_state = ui_state
        self.app_adapter = app_adapter
        self.async_ui_manager = async_ui_manager # Store AsyncUIManager
        self._lock = asyncio.Lock()

        self.step_list_view = step_list_view_class(
            on_step_selected=self._handle_step_list_selection,
            on_step_activated=self._handle_step_list_activation,
            on_step_reorder_requested=self._handle_step_reorder_request
        )
        self.pipeline_actions_toolbar = pipeline_actions_toolbar_class(
            ui_state=self.ui_state,
            app_adapter=self.app_adapter,
            get_current_plate_adapter=self._get_current_plate_adapter_for_toolbar
        )

        self.container = Frame(
            HSplit([
                self.pipeline_actions_toolbar.container,
                Window(height=1, char='-'), # Separator
                self.step_list_view.container
            ]),
            title="Pipeline Editor" # Or make dynamic if needed
        )
        
        self._ui_initialized = False

        # Register observers for TUIState changes
        self.ui_state.add_observer('current_pipeline_definition_updated', self._on_pipeline_definition_updated)
        # Observe 'active_step_data_changed' to update local selection highlight
        self.ui_state.add_observer('active_step_data_changed', self._on_active_step_data_changed_external) # Renamed event
        # Observe 'active_plate_id_changed' to reload pipeline for new plate
        self.ui_state.add_observer('active_plate_id_changed', self._on_active_plate_changed)
        # Listen to generic UI context changes to refresh button states
        self.ui_state.add_observer('ui_context_changed', self._on_ui_context_changed)


    async def initialize_controller(self):
        """Perform async initialization, including fetching initial pipeline if a plate is active."""
        logger.info("PipelineEditorController: Initializing...")
        # _load_pipeline_for_active_plate is awaited. If it were to be fire-and-forget:
        # self.async_ui_manager.fire_and_forget(self._load_pipeline_for_active_plate(), name="InitialPipelineLoad")
        await self._load_pipeline_for_active_plate() 
        self.pipeline_actions_toolbar.update_button_states() 
        self._ui_initialized = True
        logger.info("PipelineEditorController: Initialization complete.")

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
    async def _on_pipeline_definition_updated(self, pipeline_core_data: Optional[List['CoreStepData']]):
        """Handles updates to TUIState.current_pipeline_definition."""
        logger.debug(f"PipelineEditorController: Received pipeline update with {len(pipeline_core_data) if pipeline_core_data else 0} steps.")
        steps_to_display = [dict(s) for s in pipeline_core_data] if pipeline_core_data else []
        await self.step_list_view.update_step_list(steps_to_display)
        self.pipeline_actions_toolbar.update_button_states()

    async def _on_active_step_data_changed_external(self, active_step_data: Optional['CoreStepData']): # Renamed from _on_selected_step_changed_externally
        """Handles changes to TUIState.active_step_data if changed by external source."""
        logger.debug(f"PipelineEditorController: External active step data changed to: {active_step_data.get('id') if active_step_data else 'None'}")
        await self.step_list_view.set_selected_step_by_id(active_step_data.get('id') if active_step_data else None)
        self.pipeline_actions_toolbar.update_button_states() # Ensure buttons reflect new active step context

    async def _on_active_plate_changed(self, active_plate_id: Optional[str]): # Name clarifies it's about the plate ID
        """Handles TUIState.active_plate_id changes by reloading the pipeline."""
        logger.debug(f"PipelineEditorController: Active plate ID changed to: {active_plate_id}. Reloading pipeline.")
        await self._load_pipeline_for_active_plate()
        # Button states update will be triggered by _on_pipeline_definition_updated or _on_ui_context_changed
    
    async def _on_ui_context_changed(self, data: Any = None):
        """Handles generic UI context changes to refresh button states."""
        self.pipeline_actions_toolbar.update_button_states()


    # --- View Event Handlers (Callbacks from StepListView) ---
    async def _handle_step_list_selection(self, selected_step_data: Optional[Dict[str, Any]]): # CoreStepData-like dict
        """Handles a step being selected in the StepListView by user interaction."""
        new_selected_step_id = selected_step_data.get('id') if selected_step_data else None
        logger.debug(f"PipelineEditorController: User selected step ID: {new_selected_step_id} from view.")

        current_tui_state_step_id = self.ui_state.active_step_data.get('id') if self.ui_state.active_step_data else None
        if current_tui_state_step_id != new_selected_step_id:
            # This will set TUIState.active_step_data and notify 'active_step_data_changed'.
            # The 'active_step_data_changed' will be caught by _on_active_step_data_changed_external.
            await self.ui_state.set_active_step_data(selected_step_data) # Pass CoreStepData-like dict
        # Toolbar update will happen via _on_ui_context_changed or _on_active_step_data_changed_external


    async def _handle_step_list_activation(self, activated_step_data: Dict[str, Any]): # CoreStepData-like dict
        """Handles a step being "activated" (e.g., Enter pressed) in the StepListView."""
        logger.info(f"PipelineEditorController: Step activated: {activated_step_data.get('name')}")
        # Ensure it's selected first, which also updates TUIState.active_step_data
        await self._handle_step_list_selection(activated_step_data)
        
        # Request AppController to show the step editor
        # This will call TUIState.set_active_editor("STEP_EDITOR", activated_step_data)
        await self.ui_state.notify('request_step_editor', {'step_data': activated_step_data})


    async def _handle_step_reorder_request(self, index: int, direction: str):
        """Handles reorder request from StepListView by dispatching ReorderStepCommand."""
        logger.info(f"PipelineEditorController: Reorder request for step at index {index}, direction {direction}.")
        async with self._lock: # Accessing steps_display_data
            if not (0 <= index < len(self.step_list_view.steps_display_data)):
                logger.warning("Invalid index for step reorder.")
                return
            step_to_move = self.step_list_view.steps_display_data[index]
            step_id = step_to_move.get('id')
            if not step_id:
                logger.warning("Step to reorder has no ID.")
                return

        from ..commands import ReorderStepCommand # Local import
        cmd = ReorderStepCommand(step_id_to_move=step_id, direction=direction)
        plate_adapter = await self._get_current_plate_adapter_for_toolbar()
        if plate_adapter: # Reordering requires a plate context
            try:
                await cmd.execute(self.app_adapter, plate_adapter, self.ui_state)
                # Successful execution should update core, then TUIState, then _on_pipeline_definition_updated.
            except Exception as e:
                logger.error(f"Error executing ReorderStepCommand: {e}", exc_info=True)
                await self.ui_state.notify("error", {"message": f"Failed to reorder step: {e}"})
        else:
            logger.warning("Cannot reorder step: No active plate adapter.")


    # --- Core Data Interaction ---
    async def _load_pipeline_for_active_plate(self):
        """Fetches pipeline for current active plate and updates TUIState."""
        if not self.ui_state.active_plate_id:
            logger.debug("PipelineEditorController: No active plate, clearing pipeline definition.")
            await self.ui_state.set_current_pipeline_definition(None) # Assumed method on TUIState
            return

        logger.info(f"PipelineEditorController: Loading pipeline for plate {self.ui_state.active_plate_id}")
        try:
            plate_adapter = await self.app_adapter.get_orchestrator_adapter(self.ui_state.active_plate_id)
            if plate_adapter:
                pipeline_core_data: Optional[List['CoreStepData']] = await plate_adapter.get_pipeline_definition()
                await self.ui_state.set_current_pipeline_definition(pipeline_core_data)
            else:
                logger.warning(f"No plate adapter available for {self.ui_state.active_plate_id}")
                await self.ui_state.set_current_pipeline_definition(None)
        except Exception as e:
            logger.error(f"Error loading pipeline for plate {self.ui_state.active_plate_id}: {e}", exc_info=True)
            await self.ui_state.set_current_pipeline_definition(None) # Clear on error
            await self.ui_state.notify("error", {"message": f"Failed to load pipeline: {e}"})

    async def shutdown(self):
        """Performs cleanup."""
        logger.info("PipelineEditorController: Shutting down...")
        self.ui_state.remove_observer('current_pipeline_definition_updated', self._on_pipeline_definition_updated)
        self.ui_state.remove_observer('active_step_data_changed', self._on_active_step_data_changed_external) # Renamed
        self.ui_state.remove_observer('active_plate_id_changed', self._on_active_plate_changed)
        self.ui_state.remove_observer('ui_context_changed', self._on_ui_context_changed)
        logger.info("PipelineEditorController: Shutdown complete.")

    def get_container(self) -> Container:
        """Returns the root container for this controller's managed view."""
        return self.container
