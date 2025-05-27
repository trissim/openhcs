"""
Application Controller for the OpenHCS TUI.

This module defines the AppController class, which is responsible for
managing the lifecycle of primary UI components (panes/controllers),
handling global application events, and orchestrating high-level UI state
changes (e.g., which editor pane is active in a dynamic slot).
"""
import asyncio
import logging
from typing import Optional, Any, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.tui.interfaces import CoreApplicationAdapterInterface, CoreStepData
    from openhcs.tui.controllers.plate_manager_controller import PlateManagerController
    from openhcs.tui.controllers.pipeline_editor_controller import PipelineEditorController
    from openhcs.tui.status_bar import StatusBar
    from openhcs.tui.components.loading_screen import LoadingScreen
    from openhcs.tui.layout_manager import LayoutManager
    from openhcs.tui.controllers.dual_editor_controller import DualEditorController
    from openhcs.tui.dialogs.plate_config_editor import PlateConfigEditorPane
    from openhcs.tui.async_manager import AsyncUIManager # Import AsyncUIManager
    GlobalConfigDataType = Dict[str, Any]


logger = logging.getLogger(__name__)

class AppController:
    """
    Manages UI component lifecycle, global events, and high-level UI state.
    """
    def __init__(self,
                 ui_state: 'TUIState',
                 app_core_adapter: 'CoreApplicationAdapterInterface',
                 layout_manager: 'LayoutManager', # LayoutManager is passed in
                 async_ui_manager: 'AsyncUIManager' # Added AsyncUIManager
                ):
        self.ui_state = ui_state
        self.app_core_adapter = app_core_adapter
        self.layout_manager = layout_manager
        self.async_ui_manager = async_ui_manager # Store AsyncUIManager

        # Component instances - will be initialized
        self.plate_manager_controller: Optional['PlateManagerController'] = None
        self.pipeline_editor_controller: Optional['PipelineEditorController'] = None # Changed from pane to controller
        self.status_bar: Optional['StatusBar'] = None
        self.loading_screen: Optional['LoadingScreen'] = None
        
        # Dynamic editor panes - instantiated on demand
        self.dual_editor_controller: Optional['DualEditorController'] = None # Changed from dual_step_func_editor
        self.plate_config_editor: Optional['PlateConfigEditorPane'] = None

        self.components_initialized = False
        self._initialization_lock = asyncio.Lock()

    async def initialize_components(self,
                                    plate_manager_controller_class: type['PlateManagerController'],
                                    pipeline_editor_controller_class: type['PipelineEditorController'], # Changed type
                                    status_bar_class: type['StatusBar'],
                                    loading_screen_class: type['LoadingScreen'],
                                    # Component classes for PlateManagerController
                                    plate_list_view_class: type,
                                    plate_actions_toolbar_class: type,
                                    step_list_view_class: type,
                                    pipeline_actions_toolbar_class: type,
                                    # Classes needed for on-demand instantiation of DualEditorController's views
                                    step_settings_editor_view_class: type, 
                                    func_pattern_view_class: type
                                    ):
        """
        Asynchronously instantiates and sets up core UI components.
        This method replaces component instantiation from OpenHCSTUI.__init__.
        """
        async with self._initialization_lock:
            if self.components_initialized:
                logger.info("AppController: Components already initialized.")
                return

            logger.info("AppController: Initializing core UI components...")

            # Store view classes needed for on-demand editor creation
            self.step_settings_editor_view_class = step_settings_editor_view_class
            self.func_pattern_view_class = func_pattern_view_class
            
            self.loading_screen = loading_screen_class(message="Initializing OpenHCS TUI...")
            self.status_bar = status_bar_class(ui_state=self.ui_state)
            
            self.plate_manager_controller = plate_manager_controller_class(
                ui_state=self.ui_state,
                app_adapter=self.app_core_adapter,
                async_ui_manager=self.async_ui_manager, # Pass AsyncUIManager
                plate_list_view_class=plate_list_view_class,
                plate_actions_toolbar_class=plate_actions_toolbar_class
            )
            if hasattr(self.plate_manager_controller, 'initialize_controller'):
                await self.plate_manager_controller.initialize_controller()

            self.pipeline_editor_controller = pipeline_editor_controller_class(
                ui_state=self.ui_state,
                app_adapter=self.app_core_adapter,
                async_ui_manager=self.async_ui_manager, # Pass AsyncUIManager
                step_list_view_class=step_list_view_class,
                pipeline_actions_toolbar_class=pipeline_actions_toolbar_class
            )
            if hasattr(self.pipeline_editor_controller, 'initialize_controller'):
                await self.pipeline_editor_controller.initialize_controller()
            
            # Set components on LayoutManager now that they are instantiated
            if self.layout_manager:
                self.layout_manager.loading_screen = self.loading_screen
                self.layout_manager.status_bar = self.status_bar
                self.layout_manager.plate_manager_controller = self.plate_manager_controller
                self.layout_manager.pipeline_editor_controller = self.pipeline_editor_controller
            
            self._register_global_event_observers()

            self.components_initialized = True
            logger.info("AppController: Core UI components initialized.")
            
            # Start loading screen if it has such a method
            if hasattr(self.loading_screen, 'start') and callable(self.loading_screen.start):
                self.loading_screen.start() # type: ignore

            # Trigger initial data load or checks, now managed by AsyncUIManager
            await self._post_initialization_tasks()


    def _register_global_event_observers(self):
        """Registers observers for events that AppController needs to handle."""
        self.ui_state.add_observer('launcher_core_config_rebound', self._on_launcher_config_rebound)
        
        # Listen to requests to show specific editors
        self.ui_state.add_observer('request_plate_config_editor', self._handle_request_plate_config_editor)
        self.ui_state.add_observer('request_step_editor', self._handle_request_step_editor)

        # Listen to events indicating an editor is finished or cancelled
        self.ui_state.add_observer('plate_config_editor_closed', self._handle_editor_closed) # Generic handler
        self.ui_state.add_observer('step_editor_closed', self._handle_editor_closed) # Generic handler
        
        # This handles the actual switch based on TUIState.active_editor_type
        self.ui_state.add_observer('active_editor_changed', self._on_active_editor_changed)


    async def _post_initialization_tasks(self):
        """Tasks to run after all main components are initialized."""
        # PlateManagerController and PipelineEditorController now have initialize_controller methods
        # which are called during initialize_components.
        # If those methods need to run background tasks themselves, they should use the passed async_ui_manager.

        # The check for components ready is still valid and can be managed.
        self.async_ui_manager.fire_and_forget(
            self._check_components_ready_and_hide_loading(),
            name="CheckComponentsReady"
        )


    async def _check_components_ready_and_hide_loading(self):
        """
        Waits for components to signal readiness (e.g., initial data loaded),
        then hides the loading screen.
        """
        # This is a simplified check. Real implementation might involve components
        # setting flags in TUIState or emitting 'ready' events.
        max_wait_time = 30  # seconds
        wait_interval = 0.5 # seconds
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            # Define what "ready" means.
            plate_manager_ready = (self.plate_manager_controller and
                                   getattr(self.plate_manager_controller, '_ui_initialized', False))
            pipeline_editor_ready = (self.pipeline_editor_controller and # Changed
                                     getattr(self.pipeline_editor_controller, '_ui_initialized', False))

            if pipeline_editor_ready and plate_manager_ready: # Both controllers report ready
                logger.info("AppController: All main components report ready.")
                if self.loading_screen and hasattr(self.loading_screen, 'complete'):
                    self.loading_screen.complete() # type: ignore
                return
            
            await asyncio.sleep(wait_interval)
            elapsed_time += wait_interval
        
        logger.warning("AppController: Timed out waiting for components to be fully ready. Hiding loading screen anyway.")
        if self.loading_screen and hasattr(self.loading_screen, 'complete'):
            self.loading_screen.complete() # type: ignore


    async def _on_launcher_config_rebound(self, new_core_config_data: 'GlobalConfigDataType') -> None:
        """Handles core config updates from the launcher."""
        logger.info(f"AppController: Global config data rebound from launcher: {new_core_config_data}")
        self.ui_state.global_config = new_core_config_data # Update TUIState
        
        if hasattr(self.app_core_adapter, 'update_global_config'):
             await self.app_core_adapter.update_global_config(new_core_config_data)

        await self.ui_state.notify('global_config_updated', new_core_config_data)
        # Application invalidation should be handled by OpenHCSTUI or TUIState observer if needed globally

    # --- Event Handlers for Editor Requests ---
    async def _handle_request_plate_config_editor(self, data: Dict[str, Any]):
        """Handles 'request_plate_config_editor' from TUIState (triggered by a command)."""
        plate_id = data.get('plate_id')
        if plate_id:
            await self.ui_state.set_active_editor("PLATE_CONFIG_EDITOR", plate_id)
        else:
            logger.warning("AppController: Request for plate config editor missing plate_id.")

    async def _handle_request_step_editor(self, data: Dict[str, Any]):
        """Handles 'request_step_editor' from TUIState (triggered by a command)."""
        step_data: Optional['CoreStepData'] = data.get('step_data')
        if step_data:
            await self.ui_state.set_active_editor("STEP_EDITOR", step_data)
        else:
            logger.warning("AppController: Request for step editor missing step_data.")

    # --- Event Handler for TUIState.active_editor_changed ---
    async def _on_active_editor_changed(self, event_data: Dict[str, Any]):
        """
        Handles changes to TUIState.active_editor_type.
        Instantiates/clears editor controllers and informs LayoutManager.
        """
        editor_type = event_data.get("editor_type")
        context_data = event_data.get("context_data") # plate_id or step_data

        logger.info(f"AppController: Active editor changed to: {editor_type}")

        # Clear any existing dynamic editors first
        if self.dual_editor_controller and hasattr(self.dual_editor_controller, 'shutdown'):
            await self.dual_editor_controller.shutdown()
        self.dual_editor_controller = None
        if self.plate_config_editor and hasattr(self.plate_config_editor, 'shutdown'): # Assuming sync shutdown
            if asyncio.iscoroutinefunction(self.plate_config_editor.shutdown): # type: ignore
                await self.plate_config_editor.shutdown() # type: ignore
            elif callable(self.plate_config_editor.shutdown): # type: ignore
                 self.plate_config_editor.shutdown() # type: ignore
        self.plate_config_editor = None
        
        active_editor_instance = None

        if editor_type == "STEP_EDITOR" and context_data:
            from openhcs.tui.controllers.dual_editor_controller import DualEditorController
            # Ensure self has access to these view classes, stored from initialize_components
            if not hasattr(self, 'step_settings_editor_view_class') or not hasattr(self, 'func_pattern_view_class'):
                logger.error("AppController: View classes for DualEditorController not available.")
                return # Or raise error

            self.dual_editor_controller = DualEditorController(
                ui_state=self.ui_state,
                app_adapter=self.app_core_adapter,
                step_data=context_data, 
                step_settings_editor_view_class=self.step_settings_editor_view_class, # Use stored class
                func_pattern_view_class=self.func_pattern_view_class         # Use stored class
            )
            if hasattr(self.dual_editor_controller, 'initialize_controller'): # Call its init if it has one
                await self.dual_editor_controller.initialize_controller()
            active_editor_instance = self.dual_editor_controller

        elif editor_type == "PLATE_CONFIG_EDITOR" and context_data:
            from openhcs.tui.dialogs.plate_config_editor import PlateConfigEditorPane 
            self.plate_config_editor = PlateConfigEditorPane(
                state=self.ui_state, # PlateConfigEditorPane might still expect 'state'
                app_core_adapter=self.app_core_adapter,
                plate_id=context_data # This is plate_id
            )
            active_editor_instance = self.plate_config_editor
        elif editor_type == "PLATE_MANAGER":
            # No specific editor instance, LayoutManager defaults to PlateManagerController's view
            pass
        else: # Unknown type or None
             logger.warning(f"AppController: Unknown or no editor type specified: {editor_type}. Defaulting to Plate Manager.")
             await self.ui_state.set_active_editor("PLATE_MANAGER") # Reset to default

        if self.layout_manager:
            self.layout_manager.set_active_left_pane_editor(active_editor_instance)


    async def _handle_editor_closed(self, data: Optional[Dict[str, Any]] = None):
        """
        Generic handler for when any editor signals it's closed (saved or cancelled).
        This will set the active editor back to the default (PlateManager).
        """
        logger.debug(f"AppController: Editor closed event received: {data}")
        await self.ui_state.set_active_editor("PLATE_MANAGER")


    # Removed: _handle_plate_config_editing_cancelled, _handle_plate_config_saved,
    # _handle_step_editing_cancelled, _handle_step_editing_state_changed,
    # _handle_plate_editing_state_changed, _handle_step_editing_finished.
    # These are now covered by commands notifying TUIState to change active_editor_type
    # (e.g. via 'step_editor_closed', 'plate_config_editor_closed'), and _on_active_editor_changed reacting.


    async def shutdown_all_components(self):
        """Gracefully shut down all managed UI components."""
        logger.info("AppController: Initiating shutdown sequence for all components...")
        
        # Include the dynamic editors in the shutdown list
        # They are already cleared by _clear_dynamic_editors if they were active,
        # but if shutdown is called abruptly, they might still exist.
        components_to_shutdown = [
            self.plate_manager_controller, 
            self.pipeline_editor_controller,
            self.status_bar,
            self.dual_editor_controller, 
            self.plate_config_editor    
        ]

        # Call shutdown on UI components first
        for component in components_to_shutdown:
            if component and hasattr(component, 'shutdown') and callable(component.shutdown):
                try:
                    comp_name = component.__class__.__name__
                    logger.info(f"Attempting to shut down {comp_name}...")
                    if asyncio.iscoroutinefunction(component.shutdown): # type: ignore
                        await component.shutdown() # type: ignore
                    else:
                        component.shutdown() # type: ignore
                    logger.info(f"Successfully shut down {comp_name}.")
                except Exception as e:
                    logger.error(f"Error during shutdown of {comp_name}: {e}", exc_info=True)
        
        # Unregister observers this controller set on TUIState
        # Unregister TUIState observers
        self.ui_state.remove_observer('launcher_core_config_rebound', self._on_launcher_config_rebound)
        self.ui_state.remove_observer('request_plate_config_editor', self._handle_request_plate_config_editor)
        self.ui_state.remove_observer('request_step_editor', self._handle_request_step_editor)
        self.ui_state.remove_observer('plate_config_editor_closed', self._handle_editor_closed)
        self.ui_state.remove_observer('step_editor_closed', self._handle_editor_closed)
        self.ui_state.remove_observer('active_editor_changed', self._on_active_editor_changed)
        
        logger.info("AppController: Component and TUIState observer shutdown sequence complete.")
        # AsyncUIManager shutdown is handled by OpenHCSTUI after AppController shutdown.
