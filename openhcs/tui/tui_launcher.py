"""
OpenHCS TUI Launcher.

Creates and manages the TUI application, initializing all required components
and creating per-plate orchestrators. Integrates the GlobalPipelineConfig.
"""
import asyncio
import logging
import sys # For fallback print
import pickle # Added for loading pipeline definitions
from pathlib import Path
from typing import Dict, Optional, Any, List # Added List

# Core OpenHCS components
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.steps.abstract import AbstractStep # Added for pipeline definition typing
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.io.filemanager import FileManager
from openhcs.io.base import storage_registry # Import storage_registry

# TUI components from tui_architecture.py
from openhcs.tui.tui_architecture import OpenHCSTUI, TUIState
from prompt_toolkit.application.current import get_app


logger = logging.getLogger(__name__)


class OpenHCSTUILauncher:
    """
    Launcher for the OpenHCS TUI.
    Manages shared components, orchestrators, and the main TUI application lifecycle.
    """
    DEFAULT_PIPELINE_FILENAME = "pipeline_definition.openhcs" # Added class variable

    def __init__(self,
                 core_global_config: GlobalPipelineConfig,
                 common_output_directory: Optional[str] = None, # Renamed parameter
                 tui_config_path: Optional[str] = None): # For any TUI-specific settings
        """
        Initialize the launcher.

        Args:
            core_global_config: The main configuration object for OpenHCS core.
            common_output_directory: Optional root path for all TUI-managed plate outputs.
            tui_config_path: Optional path to a TUI-specific configuration file.
        """
        self.logger = logger
        self.core_global_config = core_global_config
        self.common_output_root = Path(common_output_directory) if common_output_directory else Path("./openhcs_tui_outputs")
        self.tui_config_path = Path(tui_config_path) if tui_config_path else None # Not used yet, placeholder

        try:
            self.common_output_root.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Common output root for TUI operations: {self.common_output_root.resolve()}")
        except Exception as e:
            self.logger.critical(f"Failed to create common output root directory {self.common_output_root}: {e}", exc_info=True)
            # Depending on desired robustness, might raise e or fallback to a temp dir. For now, logs critical.

        self.orchestrators: Dict[str, PipelineOrchestrator] = {}
        self.orchestrators_lock = asyncio.Lock()
        self.tui_app_instance: Optional[OpenHCSTUI] = None # To store the TUI app instance

        # Create shared instances
        self.shared_storage_registry = storage_registry() # Create registry once
        self.filemanager = FileManager(self.shared_storage_registry) # Pass shared registry
        # FileManager is intentionally config-agnostic.
        # Clients of FileManager (e.g., Orchestrator, Pipeline Steps) use VFSConfig
        # from GlobalPipelineConfig to determine which backend to specify for operations
        # or how to construct paths, rather than FileManager itself holding VFSConfig.
        self.state = TUIState()

        # Notify that the filemanager is available, for components like PlateManagerPane
        # which might defer parts of their initialization until FileManager is ready.
        # We'll notify components about filemanager availability after the application is created
        # This will be done in the run() method

        # Create an initial ProcessingContext for the TUI itself.
        # This context holds shared objects like filemanager and global_config.
        # It's not for a specific plate/pipeline run but for TUI's general use.
        self.initial_tui_context = ProcessingContext(
            global_config=self.core_global_config,
            filemanager=self.filemanager,
            well_id="TUI_GlobalContext" # A placeholder well_id
        )
        # Note: ProcessingContext.freeze() is not called on this initial context
        # as TUI might need to modify parts of it or its sub-objects if not careful.
        # However, global_config itself is frozen.

        self.logger.info(
            f"OpenHCSTUILauncher initialized. Common Output Root: {self.common_output_root}. "
            f"Using GlobalPipelineConfig (num_workers={self.core_global_config.num_workers})."
        )

        # Register event observers with TUIState
        self.state.add_observer('plate_added', self._on_plate_added)
        self.state.add_observer('plate_removed', self._on_plate_removed)
        self.state.add_observer('plate_selected', self._on_plate_selected)
        # Register observer for global config updates from the settings pane
        self.state.add_observer('global_config_needs_update', self._handle_global_config_update)

    async def _handle_global_config_update(self, new_config: GlobalPipelineConfig):
        """
        Handles updates to the GlobalPipelineConfig triggered from the TUI settings.
        Propagates the new config to relevant components.
        """
        if not isinstance(new_config, GlobalPipelineConfig):
            self.logger.error(f"Invalid data received for global_config_needs_update: {type(new_config)}. Expected GlobalPipelineConfig.")
            return

        self.logger.info(f"GlobalPipelineConfig update received. New num_workers: {new_config.num_workers}")
        self.core_global_config = new_config # Update launcher's master copy

        # Notify OpenHCSTUI that its direct reference to core_global_config should be updated.
        # OpenHCSTUI will need to observe 'launcher_core_config_rebound'.
        self.state.notify('launcher_core_config_rebound', self.core_global_config)
        self.logger.info("Notified TUI components about core_global_config rebound.")

        # Update all active PipelineOrchestrator instances
        async with self.orchestrators_lock:
            if not self.orchestrators:
                self.logger.info("No active orchestrators to update with new global config.")
            for plate_id, orchestrator in self.orchestrators.items():
                try:
                    if hasattr(orchestrator, 'apply_new_global_config'):
                        await orchestrator.apply_new_global_config(self.core_global_config)
                        self.logger.info(f"Applied new global config to orchestrator for plate '{plate_id}'.")
                    else:
                        # Fallback: Re-initialize orchestrator if apply_new_global_config is not yet implemented
                        # This is more disruptive and assumes orchestrator can be re-initialized safely.
                        # For now, we'll log a warning, as apply_new_global_config is the preferred path (Task 2.5)
                        self.logger.warning(
                            f"PipelineOrchestrator for plate '{plate_id}' does not have "
                            f"'apply_new_global_config' method. Config not applied live to this orchestrator."
                        )
                        # As a more forceful alternative (if apply_new_global_config is hard):
                        # self.orchestrators[plate_id] = PipelineOrchestrator(
                        #     plate_path=orchestrator.plate_path, # Assuming these are stored
                        #     workspace_path=orchestrator.workspace_path,
                        #     global_config=self.core_global_config
                        # )
                        # self.orchestrators[plate_id].initialize()
                        # logger.info(f"Re-initialized orchestrator for plate '{plate_id}' with new global config.")
                except Exception as e:
                    self.logger.error(f"Error applying new global config to orchestrator for plate '{plate_id}': {e}", exc_info=True)

        # The initial_tui_context's global_config reference will be stale.
        # Components should rely on OpenHCSTUI.global_config or new contexts from orchestrators.
        self.logger.info("Global config propagation complete in launcher.")

    async def _on_plate_added(self, plate_info: Dict[str, Any]):
        """Handles 'plate_added' event: Creates and initializes a PipelineOrchestrator."""
        plate_id = plate_info.get('id')
        plate_path_str = plate_info.get('path')

        if not plate_id or not plate_path_str:
            self.logger.error(f"Cannot add plate: plate_id or path missing in plate_info: {plate_info}")
            return

        self.logger.info(f"Attempting to add plate: id='{plate_id}', path='{plate_path_str}'")

        async with self.orchestrators_lock:
            if plate_id in self.orchestrators:
                self.logger.warning(f"Plate '{plate_id}' already has an orchestrator. Ignoring add request.")
                return

            # Construct plate-specific workspace path within the common output root
            safe_plate_id_for_path = plate_id.replace(':', '_').replace('/', '_').replace('\\', '_')
            workspace_path_for_plate = self.common_output_root / f"plate_{safe_plate_id_for_path}"

            try:
                self.logger.debug(f"Creating PipelineOrchestrator for plate '{plate_id}'.")
                orchestrator = PipelineOrchestrator(
                    plate_path=plate_path_str,
                    workspace_path=workspace_path_for_plate,
                    global_config=self.core_global_config, # Crucial: Pass the global config
                    filemanager=self.filemanager # Pass shared filemanager
                )

                # orchestrator.initialize() # DO NOT initialize here. Initialization is now explicit via "Pre-compile" button.
                self.orchestrators[plate_id] = orchestrator
                self.logger.info(f"PipelineOrchestrator instance created for plate '{plate_id}'. Initialization pending user action.")

                # Notify that plate is added but not yet fully initialized/ready for compilation.
                self.state.notify('plate_status_changed', {'plate_id': plate_id, 'status': 'added'})
            except Exception as e:
                self.logger.error(f"Failed to create/initialize orchestrator for plate '{plate_id}': {e}", exc_info=True)
                self.state.notify('error', {
                    'source': 'OpenHCSTUILauncher._on_plate_added',
                    'message': f"Error initializing plate {plate_id}",
                    'details': str(e)
                })
                self.state.notify('plate_status_changed', {'plate_id': plate_id, 'status': 'error', 'message': str(e)})

    async def _on_plate_removed(self, plate_info: Dict[str, Any]):
        """Handles 'plate_removed' event: Cleans up the orchestrator."""
        plate_id = plate_info.get('id')
        if not plate_id: return

        self.logger.info(f"Attempting to remove plate: id='{plate_id}'")
        removed_orchestrator_instance = None
        async with self.orchestrators_lock:
            if plate_id in self.orchestrators:
                removed_orchestrator_instance = self.orchestrators.pop(plate_id)
                self.logger.info(f"Orchestrator for plate '{plate_id}' removed from launcher's active list.")
                # Actual orchestrator shutdown/cleanup might be handled by its own methods if needed,
                # or rely on garbage collection if it doesn't hold persistent resources like open files/threads.
                # For now, we just remove it from the dict.
            else:
                self.logger.warning(f"Attempted to remove plate '{plate_id}', but no orchestrator found in launcher.")
                return # Exit if no orchestrator was found to remove

        # If the removed plate was the active one, clear relevant TUIState.
        if hasattr(self.state, 'active_orchestrator') and self.state.active_orchestrator == removed_orchestrator_instance:
            self.logger.info(f"Removed plate '{plate_id}' was active. Clearing active plate context in TUIState.")
            if hasattr(self.state, 'clear_active_plate_context') and callable(self.state.clear_active_plate_context):
                self.state.clear_active_plate_context() # This will set active_orchestrator, selected_plate, etc. to None
            else: # Fallback if method is missing (should not happen with prior changes)
                self.state.active_orchestrator = None
                self.state.selected_plate = None # Ensure selected_plate is also cleared
                self.state.current_pipeline_definition = None
            
            # Notify components that the active context (including pipeline) has changed.
            # PipelineEditorPane would observe 'pipeline_definition_loaded' to clear its view.
            await self.state.notify('active_orchestrator_changed', {'orchestrator': None, 'plate_id': None})
            await self.state.notify('pipeline_definition_loaded', {'plate_id': None, 'pipeline': []})
            self.logger.info(f"Notified UI about active context change for removed plate '{plate_id}'.")
        
        # Notify about the removal itself, e.g., for PlateManagerPane to update its list.
        # This should ideally happen after TUIState context is cleared if it was active.
        await self.state.notify('plate_status_changed', {'plate_id': plate_id, 'status': 'removed'})


    async def _on_plate_selected(self, plate_info: Optional[Dict[str, Any]]): # plate_info can be None
        """Handles 'plate_selected' event: Sets active orchestrator and loads its pipeline."""
        if plate_info is None or plate_info.get('id') is None:
            self.logger.info("Plate selection cleared or invalid. Clearing active plate context.")
            if hasattr(self.state, 'clear_active_plate_context') and callable(self.state.clear_active_plate_context):
                self.state.clear_active_plate_context()
            else: # Fallback
                self.state.active_orchestrator = None
                self.state.selected_plate = None
                self.state.current_pipeline_definition = None
            await self.state.notify('active_orchestrator_changed', {'orchestrator': None, 'plate_id': None})
            await self.state.notify('pipeline_definition_loaded', {'plate_id': None, 'pipeline': []})
            return

        plate_id = plate_info.get('id')
        self.logger.info(f"Plate selected: id='{plate_id}'")

        orchestrator = None
        async with self.orchestrators_lock:
            orchestrator = self.orchestrators.get(plate_id)

        if orchestrator:
            self.state.active_orchestrator = orchestrator
            self.state.selected_plate = plate_info # Store the full plate_info for context
            self.logger.debug(f"Active orchestrator set in TUIState for plate '{plate_id}'.")
            await self.state.notify('active_orchestrator_changed', {'orchestrator': orchestrator, 'plate_id': plate_id})

            # Load pipeline definition
            pipeline_loaded: List[AbstractStep] = []
            pipeline_file_path = orchestrator.workspace_path / self.DEFAULT_PIPELINE_FILENAME
            
            if await asyncio.to_thread(pipeline_file_path.exists): # Run sync Path.exists in thread
                try:
                    with open(pipeline_file_path, "rb") as f:
                        # Run sync pickle.load in thread
                        pipeline_loaded = await asyncio.to_thread(pickle.load, f) 
                    if not isinstance(pipeline_loaded, list) or \
                       not all(isinstance(step, AbstractStep) for step in pipeline_loaded):
                        self.logger.error(f"Pipeline file '{pipeline_file_path}' for plate '{plate_id}' is corrupted or not a list of AbstractStep. Treating as empty.")
                        pipeline_loaded = []
                    else:
                        self.logger.info(f"Pipeline definition loaded for plate '{plate_id}' from '{pipeline_file_path}'. Steps: {len(pipeline_loaded)}")
                except FileNotFoundError: # Should be caught by .exists(), but defensive
                    self.logger.info(f"Pipeline file not found for plate '{plate_id}' at '{pipeline_file_path}'. Starting with empty pipeline.")
                    pipeline_loaded = []
                except pickle.UnpicklingError as e:
                    self.logger.error(f"Error unpickling pipeline for plate '{plate_id}' from '{pipeline_file_path}': {e}. Treating as empty.")
                    pipeline_loaded = []
                except Exception as e: # Catch other potential errors during load
                    self.logger.error(f"Unexpected error loading pipeline for plate '{plate_id}': {e}", exc_info=True)
                    pipeline_loaded = []
            else:
                self.logger.info(f"No pipeline definition file found for plate '{plate_id}' at '{pipeline_file_path}'. Starting with empty pipeline.")
                pipeline_loaded = []
            
            self.state.current_pipeline_definition = pipeline_loaded
            await self.state.notify('pipeline_definition_loaded', {'plate_id': plate_id, 'pipeline': pipeline_loaded})

        else:
            self.logger.warning(f"Selected plate '{plate_id}' has no orchestrator instance in launcher. Clearing active context.")
            if hasattr(self.state, 'clear_active_plate_context') and callable(self.state.clear_active_plate_context):
                self.state.clear_active_plate_context()
            else: # Fallback
                self.state.active_orchestrator = None
                self.state.selected_plate = None
                self.state.current_pipeline_definition = None
            await self.state.notify('active_orchestrator_changed', {'orchestrator': None, 'plate_id': None})
            await self.state.notify('pipeline_definition_loaded', {'plate_id': None, 'pipeline': []})


    async def run(self):
        """Initializes and runs the TUI application."""
        self.logger.info("Initializing OpenHCSTUI application...")

        # Instantiate OpenHCSTUI, passing the shared state, filemanager,
        # the initial_tui_context (which contains global_config), and global_config directly.
        # OpenHCSTUI.__init__ was updated to accept these.
        tui_app = OpenHCSTUI(
            initial_context=self.initial_tui_context,
            state=self.state,
            global_config=self.core_global_config
        )
        self.tui_app_instance = tui_app # Store the instance

        # Now that the application is created, notify components about filemanager availability
        await self.state.notify('filemanager_available', {'filemanager': self.filemanager})
        self.logger.info("Notified 'filemanager_available'.")

        self.logger.info("Starting OpenHCS TUI event loop.")
        try:
            if hasattr(tui_app, 'application') and hasattr(tui_app.application, 'run_async'):
                await tui_app.application.run_async()
            elif hasattr(tui_app, 'run_async'): # If OpenHCSTUI itself is runnable
                 await tui_app.run_async()
            else:
                self.logger.critical("OpenHCSTUI object or its 'application' attribute does not have a 'run_async' method. TUI cannot start.")
                print("CRITICAL ERROR: TUI application cannot be started.", file=sys.stderr)
                return # Exit if TUI cannot run
        except Exception as e:
            self.logger.critical(f"Unhandled exception during TUI run: {e}", exc_info=True)
        finally:
            self.logger.info("OpenHCS TUI run loop finished or exited.")
            await self._cleanup()

    async def _cleanup(self):
        """Cleans up resources when the application exits."""
        self.logger.info("Cleaning up TUI launcher resources...")

        # Shutdown TUI components first
        if self.tui_app_instance:
            if hasattr(self.tui_app_instance, 'shutdown_components') and callable(self.tui_app_instance.shutdown_components):
                self.logger.info("Calling OpenHCSTUI.shutdown_components()...")
                try:
                    await self.tui_app_instance.shutdown_components()
                except Exception as e:
                    self.logger.error(f"Error during OpenHCSTUI.shutdown_components(): {e}", exc_info=True)
            else:
                self.logger.warning("OpenHCSTUI instance does not have a callable 'shutdown_components' method.")

        async with self.orchestrators_lock:
            # Note: PipelineOrchestrator instances manage resources like ThreadPoolExecutor
            # within method scopes (e.g., using `with` statements), generally not requiring
            # an explicit shutdown call from TuiLauncher for their own direct, long-lived resources.
            # The loop below provides a generic mechanism to call a `shutdown()` method
            # if an orchestrator instance defines one (e.g., for custom resource cleanup).
            # For now, just clearing the orchestrators list after attempting shutdown.
            for orchestrator in self.orchestrators.values():
                if hasattr(orchestrator, 'shutdown') and callable(orchestrator.shutdown):
                    try:
                        # Assuming orchestrator.shutdown() might be async or sync
                        if asyncio.iscoroutinefunction(orchestrator.shutdown):
                            await orchestrator.shutdown()
                        else:
                            orchestrator.shutdown()
                        self.logger.info(f"Shut down orchestrator for plate: {orchestrator.plate_path}") # Assuming plate_path attr
                    except Exception as e:
                        self.logger.error(f"Error shutting down orchestrator for {orchestrator.plate_path}: {e}", exc_info=True)
            self.orchestrators.clear()
        self.logger.info("Orchestrators cleared/shut down.")

        # Any other cleanup (e.g., for filemanager if it holds resources)
        if hasattr(self.filemanager, 'close') and callable(self.filemanager.close):
            try:
                self.logger.info("Closing FileManager...")
                # Assuming close might be async or sync
                if asyncio.iscoroutinefunction(self.filemanager.close):
                    await self.filemanager.close()
                else:
                    self.filemanager.close()
                self.logger.info("FileManager closed.")
            except Exception as e:
                self.logger.error(f"Error closing FileManager: {e}", exc_info=True)

        self.logger.info("TUI Launcher cleanup complete.")