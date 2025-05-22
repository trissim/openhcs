"""
Action Menu Pane for OpenHCS TUI.

This module implements the right pane of the OpenHCS TUI, which displays
buttons for key operations and handles pipeline compilation, execution,
and configuration settings.
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path # For Path.home()
import dataclasses # For asdict
import yaml # For YAML serialization
import json # ADDED for saving pipeline definition

from openhcs.core.steps.abstract import AbstractStep # ADDED
from openhcs.core.context.processing_context import ProcessingContext # ADDED
# Forward reference for TUIState to avoid circular import if it's in tui_architecture
if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator


from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout import HSplit, VSplit, Container, DynamicContainer, Window
from prompt_toolkit.widgets import Button, Label, Box, Dialog, TextArea, Checkbox, RadioList
from prompt_toolkit.layout.dimension import D

# Import config classes for creating new instances
from openhcs.core.config import GlobalPipelineConfig, VFSConfig, PathPlanningConfig
# Assuming TUIState and ProcessingContext will be imported from their respective locations
# For now, using Any as placeholder if direct import causes issues during this step.
# from .tui_architecture import TUIState # Correct path once file structure is set
# from openhcs.core.context.processing_context import ProcessingContext
# from openhcs.core.config import GlobalPipelineConfig

logger = logging.getLogger(__name__)

class ActionMenuPane:
    """
    Right pane for action menu in the OpenHCS TUI.

    Displays buttons for key operations, with state-dependent enabling/disabling
    and visual feedback for long-running operations.
    """

    def __init__(self, state: 'TUIState', initial_tui_context: 'ProcessingContext'):
        """
        Initialize the Action Menu pane.

        Args:
            state: The global TUIState instance.
            initial_tui_context: The initial ProcessingContext, containing GlobalPipelineConfig and FileManager.
        """
        self.state = state
        # Assuming initial_tui_context has global_config and filemanager attributes
        self.global_config = getattr(initial_tui_context, 'global_config', None)
        self.filemanager = getattr(initial_tui_context, 'filemanager', None)
        self.active_settings_dialog_float: Optional[Float] = None
        self._current_dialog_future: Optional[asyncio.Future] = None


        # Dictionaries to hold references to input widgets in the settings dialog
        self.tui_settings_widgets: Dict[str, Any] = {}
        self.global_config_widgets: Dict[str, Any] = {}
        
        # Internal component state
        self.error_message_text: Optional[str] = None # For displaying errors from this pane's operations
        self.status_text: Optional[str] = None # For displaying status from this pane's operations

        # Create UI components
        self.error_banner = Label(lambda: self.error_message_text or "")
        self.status_indicator = Label(lambda: self.status_text or "")
        self.buttons_container = HSplit(self._create_buttons())

        self.container = HSplit([
            self.error_banner,
            self.status_indicator,
            self.buttons_container
        ])

        # Register to handle requests from other components (e.g., MenuBar)
        if hasattr(self.state, 'add_observer'):
            self.state.add_observer('ui_request_show_settings_dialog',
                                    lambda data=None: get_app().create_background_task(self._settings_handler()))

    def _show_error(self, message: str):
        """Displays an error message in this pane's error banner."""
        self.error_message_text = f"Error: {message}"
        # In a real app, this might also clear after a timeout or be logged
        logger.error(message)
        # Force redraw if necessary, though prompt_toolkit often handles it.
        get_app().invalidate()

    def _clear_error(self):
        self.error_message_text = None
        get_app().invalidate()

    def _set_status(self, message: str):
        self.status_text = message
        get_app().invalidate()
    
    def _clear_status(self):
        self.status_text = None
        get_app().invalidate()

    def _create_buttons(self) -> List[Container]:
        """
        Create the action buttons.
        """
        # Button handlers - most are stubs for now
        async def _add_handler():
            logger.info("ActionMenuPane: 'Add Plate' button clicked. Emitting 'ui_request_show_add_plate_dialog' event.")
            self.state.notify('ui_request_show_add_plate_dialog')
            # The PlateManagerPane should be listening for this event to show the dialog.
            # self._show_error("Add: Not yet implemented.") # Remove stub message

        async def _pre_compile_handler(): # This is "Initialize Plate"
            self._clear_error()
            self._set_status("Initialize Plate command received...")
            logger.info("ActionMenuPane: 'Pre-compile' (Initialize Plate) button clicked.")

            active_orchestrator: Optional['PipelineOrchestrator'] = getattr(self.state, 'active_orchestrator', None)
            selected_plate: Optional[Dict[str, Any]] = getattr(self.state, 'selected_plate', None)

            if not active_orchestrator or not selected_plate:
                err_msg = "No active plate selected for initialization."
                self._show_error(err_msg)
                logger.warning(err_msg)
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {'operation': 'initialize_plate', 'status': 'error', 'message': err_msg, 'source': 'ActionMenuPane'})
                self._clear_status()
                return

            plate_id = selected_plate.get('id')
            plate_name = selected_plate.get('name', 'Unknown')

            # Check if orchestrator is initialized by calling its method
            if hasattr(active_orchestrator, 'is_initialized') and callable(active_orchestrator.is_initialized) and active_orchestrator.is_initialized():
                status_msg = f"Plate '{plate_name}' is already initialized."
                self._set_status(status_msg)
                logger.info(status_msg)
                return

            init_start_msg = f"Initializing plate '{plate_name}'..."
            setattr(self.state, 'is_running', True)
            if hasattr(self.state, 'notify'):
                self.state.notify('operation_status_changed', {
                    'operation': 'initialize_plate',
                    'status': 'running',
                    'message': init_start_msg,
                    'source': 'ActionMenuPane'
                })
            self._set_status(init_start_msg)

            try:
                logger.info(f"Calling initialize() for orchestrator of plate '{plate_id}'...")
                await active_orchestrator.initialize() # This should set orchestrator.is_initialized = True

                init_success_msg = f"Plate '{plate_name}' initialized successfully."
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {
                        'operation': 'initialize_plate',
                        'status': 'success',
                        'message': init_success_msg,
                        'source': 'ActionMenuPane'
                    })
                    # Notify PlateManagerPane to update the plate's display status
                    self.state.notify('plate_status_changed', {'plate_id': plate_id, 'status': 'initialized'})
                self._set_status(init_success_msg)
                logger.info(init_success_msg)

            except Exception as e:
                logger.error(f"Initialization failed for plate '{plate_name}': {e}", exc_info=True)
                init_fail_msg = f"Initialization Error: {str(e)}"
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {
                        'operation': 'initialize_plate',
                        'status': 'error',
                        'message': init_fail_msg,
                        'source': 'ActionMenuPane'
                    })
                    self.state.notify('plate_status_changed', {'plate_id': plate_id, 'status': 'init_error'})
                self._show_error(init_fail_msg)
                self._clear_status()
            finally:
                setattr(self.state, 'is_running', False)

        async def _compile_handler():
            self._clear_error()
            self._set_status("Compile command received...")
            logger.info("ActionMenuPane: Compile button clicked.")

            active_orchestrator: Optional['PipelineOrchestrator'] = getattr(self.state, 'active_orchestrator', None)
            selected_plate: Optional[Dict[str, Any]] = getattr(self.state, 'selected_plate', None)

            if not active_orchestrator or not selected_plate:
                err_msg = "No active plate selected for compilation."
                self._show_error(err_msg)
                logger.warning(err_msg)
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {'operation': 'compile', 'status': 'error', 'message': err_msg, 'source': 'ActionMenuPane'})
                self._clear_status()
                return

            # Check if orchestrator is initialized
            if not (hasattr(active_orchestrator, 'is_initialized') and active_orchestrator.is_initialized()):
                err_msg = f"Plate '{selected_plate.get('name', 'Unknown')}' is not initialized. Please Initialize Plate first."
                self._show_error(err_msg)
                logger.warning(err_msg)
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {'operation': 'compile', 'status': 'error', 'message': err_msg, 'source': 'ActionMenuPane'})
                self._clear_status()
                return

            pipeline_to_compile: Optional[List[AbstractStep]] = getattr(active_orchestrator, 'pipeline_definition', None)

            if not pipeline_to_compile:
                err_msg = f"No pipeline definition found for plate '{selected_plate.get('name', 'Unknown')}'. Load or define a pipeline first."
                self._show_error(err_msg)
                logger.warning(err_msg)
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {'operation': 'compile', 'status': 'error', 'message': err_msg, 'source': 'ActionMenuPane'})
                self._clear_status()
                return

            # Update TUIState for compilation start
            setattr(self.state, 'is_compiled', False)
            setattr(self.state, 'compiled_contexts', None)
            setattr(self.state, 'is_running', True)
            comp_start_msg = f"Compiling: {selected_plate.get('name', 'Unknown')}..."
            setattr(self.state, 'compilation_status', comp_start_msg)
            if hasattr(self.state, 'notify'):
                self.state.notify('operation_status_changed', {
                    'operation': 'compile',
                    'status': 'running',
                    'message': comp_start_msg,
                    'source': 'ActionMenuPane'
                })
            self._set_status(comp_start_msg)

            try:
                logger.info(f"Calling compile_pipelines for orchestrator of plate '{selected_plate.get('id')}'...")
                compiled_contexts: Optional[Dict[str, ProcessingContext]] = await active_orchestrator.compile_pipelines(
                    well_filter=None
                )
                
                if compiled_contexts is None:
                    raise RuntimeError("Compilation returned None, indicating an internal failure.")

                setattr(self.state, 'is_compiled', True)
                setattr(self.state, 'compiled_contexts', compiled_contexts)
                comp_success_msg = f"Compiled: {selected_plate.get('name', 'Unknown')}"
                setattr(self.state, 'compilation_status', comp_success_msg)
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {
                        'operation': 'compile',
                        'status': 'success',
                        'message': comp_success_msg,
                        'source': 'ActionMenuPane'
                    })
                self._set_status(comp_success_msg)
                logger.info(comp_success_msg)

            except Exception as e:
                logger.error(f"Compilation failed for plate '{selected_plate.get('name', 'Unknown')}': {e}", exc_info=True)
                setattr(self.state, 'is_compiled', False)
                comp_fail_msg = f"Compile Error: {str(e)}"
                setattr(self.state, 'compilation_status', comp_fail_msg)
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {
                        'operation': 'compile',
                        'status': 'error',
                        'message': comp_fail_msg,
                        'source': 'ActionMenuPane'
                    })
                self._show_error(comp_fail_msg)
                self._clear_status()
            finally:
                setattr(self.state, 'is_running', False)

        async def _run_handler():
            self._clear_error()
            self._set_status("Run command received...")
            logger.info("ActionMenuPane: Run button clicked.")

            active_orchestrator: Optional['PipelineOrchestrator'] = getattr(self.state, 'active_orchestrator', None)
            selected_plate: Optional[Dict[str, Any]] = getattr(self.state, 'selected_plate', None)
            is_compiled: bool = getattr(self.state, 'is_compiled', False)
            compiled_contexts: Optional[Dict[str, ProcessingContext]] = getattr(self.state, 'compiled_contexts', None)

            if not active_orchestrator or not selected_plate:
                err_msg = "No active plate selected for execution."
                self._show_error(err_msg)
                logger.warning(err_msg)
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {'operation': 'run', 'status': 'error', 'message': err_msg, 'source': 'ActionMenuPane'})
                self._clear_status()
                return

            if not is_compiled or not compiled_contexts:
                err_msg = f"Pipeline for plate '{selected_plate.get('name', 'Unknown')}' is not compiled. Please compile first."
                self._show_error(err_msg)
                logger.warning(err_msg)
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {'operation': 'run', 'status': 'error', 'message': err_msg, 'source': 'ActionMenuPane'})
                self._clear_status()
                return

            # Update TUIState for execution start
            setattr(self.state, 'is_running', True)
            exec_start_msg = f"Running pipeline for {selected_plate.get('name', 'Unknown')}..."
            setattr(self.state, 'execution_status', exec_start_msg) # New TUIState attribute
            if hasattr(self.state, 'notify'):
                self.state.notify('operation_status_changed', {
                    'operation': 'run',
                    'status': 'running',
                    'message': exec_start_msg,
                    'source': 'ActionMenuPane'
                })
            self._set_status(exec_start_msg)

            try:
                logger.info(f"Calling execute_compiled_plate for orchestrator of plate '{selected_plate.get('id')}'...")
                # Assuming execute_compiled_plate returns some results or None on success
                # The actual return type of execute_compiled_plate might need to be handled (e.g., List of results or exceptions)
                execution_results = await active_orchestrator.execute_compiled_plate(
                    compiled_contexts=compiled_contexts,
                    well_ids_to_execute=None # For now, execute all wells in the compiled_contexts
                )
                
                # Orchestrator's execute_compiled_plate indicates overall success by not raising an exception.
                # Detailed per-well/per-step status and outputs are written to VFS and can be inspected separately.
                # The `execution_results` variable captures the orchestrator's direct return,
                # but per user guidance, it's not used here for detailed TUI status.
                # Number of steps remaining/processed would be part of a more detailed VFS-based status view.

                exec_success_msg = f"Execution command completed for {selected_plate.get('name', 'Unknown')}. Check VFS for outputs."
                setattr(self.state, 'execution_status', exec_success_msg)
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {
                        'operation': 'run',
                        'status': 'success', # Indicates the command itself finished without top-level error
                        'message': exec_success_msg,
                        'source': 'ActionMenuPane'
                    })
                self._set_status(exec_success_msg) # Update pane's local status
                logger.info(exec_success_msg)

            except Exception as e:
                logger.error(f"Execution failed for plate '{selected_plate.get('name', 'Unknown')}': {e}", exc_info=True)
                exec_fail_msg = f"Run Error: {str(e)}"
                setattr(self.state, 'execution_status', exec_fail_msg)
                if hasattr(self.state, 'notify'):
                    self.state.notify('operation_status_changed', {
                        'operation': 'run',
                        'status': 'error',
                        'message': exec_fail_msg,
                        'source': 'ActionMenuPane'
                    })
                self._show_error(exec_fail_msg)
                self._clear_status()
            finally:
                setattr(self.state, 'is_running', False)
        
        async def _save_handler():
            self._clear_error()
            self._set_status("Save Pipeline command received...")
            logger.info("ActionMenuPane: Save Pipeline button clicked.")

            active_orchestrator: Optional['PipelineOrchestrator'] = getattr(self.state, 'active_orchestrator', None)
            selected_plate: Optional[Dict[str, Any]] = getattr(self.state, 'selected_plate', None)

            if not active_orchestrator or not selected_plate:
                err_msg = "No active plate selected to save pipeline for."
                self._show_error(err_msg)
                logger.warning(err_msg)
                self._clear_status()
                return

            pipeline_definition: Optional[List[AbstractStep]] = getattr(active_orchestrator, 'pipeline_definition', None)
            if not pipeline_definition:
                err_msg = f"No pipeline definition loaded for plate '{selected_plate.get('name', 'Unknown')}' to save."
                self._show_error(err_msg)
                logger.warning(err_msg)
                self._clear_status()
                return

            if not self.filemanager:
                err_msg = "FileManager not available to save pipeline."
                self._show_error(err_msg)
                logger.error(err_msg)
                self._clear_status()
                return

            try:
                plate_dir_path = selected_plate.get('path')
                if not plate_dir_path:
                    raise ValueError("Selected plate information is missing the 'path' attribute.")
                
                # Use DEFAULT_PIPELINE_FILENAME from orchestrator if available, otherwise fallback
                filename = getattr(active_orchestrator, 'DEFAULT_PIPELINE_FILENAME', 'pipeline_definition.json')
                save_path = Path(plate_dir_path) / filename
                
                pipeline_dicts = [step.to_dict() for step in pipeline_definition]
                json_content = json.dumps(pipeline_dicts, indent=2)

                self._set_status(f"Saving pipeline to {save_path}...")
                logger.info(f"Attempting to save pipeline definition for plate '{selected_plate.get('id')}' to {save_path}")

                # Assuming filemanager can handle Path objects or convert to string
                # And that write_file_async exists and works with 'disk' backend for absolute paths.
                # If filemanager is strictly VFS, direct file I/O might be needed for user-space saves.
                # For now, proceeding with filemanager.
                # The backend for saving a pipeline definition to a plate's local directory should be 'disk'.
                await self.filemanager.write_file_async(str(save_path), json_content, backend="disk")

                save_success_msg = f"Pipeline saved to {save_path}"
                self._set_status(save_success_msg)
                logger.info(save_success_msg)
                # Optionally notify TUIState or other components
                # self.state.notify('pipeline_saved', {'plate_id': selected_plate.get('id'), 'path': str(save_path)})

            except Exception as e:
                logger.error(f"Failed to save pipeline for plate '{selected_plate.get('name', 'Unknown')}': {e}", exc_info=True)
                save_fail_msg = f"Save Error: {str(e)}"
                self._show_error(save_fail_msg)
                self._clear_status()

        async def _test_handler():
            self._clear_error()
            logger.info("ActionMenuPane: 'Test Pipeline' button clicked. Adding predefined test plate.")
            
            # Define the test plate details
            # Based on tests/integration/tests_data/opera_phenix_pipeline/test_main_3d[OperaPhenix]/zstack_plate
            test_plate_relative_path = "tests/integration/tests_data/opera_phenix_pipeline/test_main_3d[OperaPhenix]/zstack_plate"
            # Construct absolute path from workspace root (assuming TUI runs from project root)
            # For robustness, this might need to be configurable or resolved via a helper if TUI CWD changes.
            # For now, direct relative path from assumed project root.
            # test_plate_path = Path.cwd() / test_plate_relative_path
            # Simpler: assume relative path is fine for FileManager if it resolves from workspace root.
            test_plate_path_str = test_plate_relative_path
            test_plate_backend = "OperaPhenix"

            status_msg = f"Attempting to add test plate: {test_plate_path_str} (Backend: {test_plate_backend})"
            self._set_status(status_msg)
            logger.info(status_msg)

            if hasattr(self.state, 'notify'):
                self.state.notify('add_predefined_plate', {
                    'path': test_plate_path_str,
                    'backend': test_plate_backend
                })
                # Also notify operation status for immediate feedback via StatusBar
                self.state.notify('operation_status_changed', {
                    'operation': 'add_test_plate',
                    'status': 'pending',
                    'message': f"Adding test plate '{Path(test_plate_path_str).name}'...",
                    'source': 'ActionMenuPane'
                })
            else:
                err_msg = "Cannot add test plate: TUIState.notify not available."
                self._show_error(err_msg)
                logger.error(err_msg)

        # Settings handler will call _create_main_settings_dialog
        # _settings_handler is defined as a method of the class

        buttons_layout = [
            Box(Button("Add Plate", handler=lambda: get_app().create_background_task(_add_handler())), padding_left=1, padding_right=1),
            Label("─" * 20), # Visual separator
            Box(Button("Pre-compile", handler=lambda: get_app().create_background_task(_pre_compile_handler())), padding_left=1, padding_right=1),
            Box(Button("Compile", handler=lambda: get_app().create_background_task(_compile_handler())), padding_left=1, padding_right=1),
            Box(Button("Run", handler=lambda: get_app().create_background_task(_run_handler())), padding_left=1, padding_right=1),
            Label("─" * 20), # Visual separator
            Box(Button("Save Pipeline", handler=lambda: get_app().create_background_task(_save_handler())), padding_left=1, padding_right=1),
            Box(Button("Test Pipeline", handler=lambda: get_app().create_background_task(_test_handler())), padding_left=1, padding_right=1),
            Label("─" * 20), # Visual separator
            Box(Button("Settings", handler=lambda: get_app().create_background_task(self._settings_handler())), padding_left=1, padding_right=1),
        ]
        return buttons_layout

    async def _settings_handler(self):
        """
        Handle Settings button click. Shows the main settings dialog.
        """
        self._clear_error()
        logger.info("ActionMenuPane: Settings button clicked. Creating and showing settings dialog.")

        if self.active_settings_dialog_float is not None:
            logger.warning("ActionMenuPane: Settings dialog is already active.")
            self._show_error("Settings dialog is already open.")
            return

        dialog = self._create_main_settings_dialog()
        app = get_app()

        if not (hasattr(app, 'layout') and hasattr(app.layout, 'container') and hasattr(app.layout.container, 'floats')):
            self._show_error("Cannot display dialog: App layout not configured for floats.")
            logger.error("ActionMenuPane: App layout not configured for floats. Cannot show settings dialog.")
            return

        self._current_dialog_future = asyncio.Future()
        
        # Create a Float for the dialog. Adjust width/height as needed, or let dialog control.
        # The Dialog widget itself has width/height properties.
        dialog_float = Float(content=dialog, width=D(preferred=80, max=100), height=D(preferred=30, max=40))
        self.active_settings_dialog_float = dialog_float

        try:
            app.layout.container.floats.append(self.active_settings_dialog_float)
            app.layout.focus(dialog) # Focus the dialog itself, not the float
            self._set_status("Settings dialog opened. Use dialog buttons to save or close.")

            # Wait for the dialog to be closed (future resolved by dialog's buttons)
            result = await self._current_dialog_future
            
            if result:
                if result.get('tui_settings_saved'):
                    logger.info("TUI settings were saved via dialog.")
                    # Status already set by save_tui_settings_handler
                if result.get('global_config_saved'):
                    logger.info("Global config settings were saved via dialog.")
                    # Status already set by _apply_and_save_global_config_settings
            else: # Dialog was cancelled (future resolved with None)
                logger.info("Settings dialog was cancelled or closed without saving.")
                self._set_status("Settings dialog closed.")

        except Exception as e:
            logger.error(f"Error during settings dialog lifecycle: {e}", exc_info=True)
            self._show_error(f"Error with settings dialog: {e}")
            if self._current_dialog_future and not self._current_dialog_future.done():
                self._current_dialog_future.set_exception(e) # Propagate error
        finally:
            if self.active_settings_dialog_float and app.layout.container.floats and self.active_settings_dialog_float in app.layout.container.floats:
                app.layout.container.floats.remove(self.active_settings_dialog_float)
            self.active_settings_dialog_float = None
            self._current_dialog_future = None # Clear the future
            
            # Attempt to focus the main application area or last focused element
            # This might need adjustment based on overall TUI focus management
            if hasattr(app.layout, 'focus_last'):
                 app.layout.focus_last()
            else: # Fallback: focus the action menu pane itself if possible
                try:
                    app.layout.focus(self.container)
                except Exception:
                    logger.warning("Could not determine element to focus after closing settings dialog.")

            self._clear_status() # Clear any specific dialog status
            get_app().invalidate() # Ensure UI redraws correctly


    def _create_main_settings_dialog(self) -> Dialog:
        """
        Creates the main settings dialog.
        Allows editing TUI settings and GlobalPipelineConfig.
        """
        from prompt_toolkit.widgets import ScrollablePane # Ensure import

        self.tui_settings_widgets = {} # Clear previous widget refs
        self.global_config_widgets = {} # Clear previous widget refs

        # --- Section 1: TUI-Specific Settings (Editable) ---
        tui_settings_content = [Label("TUI Specific Settings:", style="class:dialog.title")]

        current_vim_mode = getattr(self.state, 'vim_mode', False)
        vim_mode_checkbox = Checkbox(text="Enable Vim keybindings", checked=current_vim_mode)
        self.tui_settings_widgets['vim_mode'] = vim_mode_checkbox
        tui_settings_content.append(vim_mode_checkbox)

        current_tui_log_level = getattr(self.state, 'tui_log_level', "INFO")
        log_levels = [("DEBUG", "DEBUG"), ("INFO", "INFO"), ("WARNING", "WARNING"), ("ERROR", "ERROR"), ("CRITICAL", "CRITICAL")]
        tui_log_level_radios = RadioList(values=log_levels, default_value=current_tui_log_level)
        self.tui_settings_widgets['tui_log_level'] = tui_log_level_radios
        tui_settings_content.extend([Label("TUI Log Level:"), tui_log_level_radios])
        
        current_editor_path = getattr(self.state, 'editor_path', os.environ.get('EDITOR', 'vim'))
        editor_path_area = TextArea(text=current_editor_path, multiline=False, height=1, prompt="Path to editor: ")
        self.tui_settings_widgets['editor_path'] = editor_path_area
        tui_settings_content.append(VSplit([Label("Editor Path:", width=20), editor_path_area]))

        tui_settings_content.append(Window(height=1, char='─'))

        # --- Section 2: Global Pipeline Configuration (Editable) ---
        global_config_content = [Label("Global Pipeline Configuration:", style="class:dialog.title")]
        
        if not self.global_config: # Should be set by OpenHCSTUILauncher
            global_config_content.append(Label("  GlobalPipelineConfig not available."))
        else:
            global_config_content.append(Label(" General:", style="class:dialog.subtitle"))
            num_workers_area = TextArea(text=str(self.global_config.num_workers), multiline=False, height=1)
            self.global_config_widgets['num_workers'] = num_workers_area
            global_config_content.append(VSplit([Label("  Num Workers:", width=30), num_workers_area]))

            global_config_content.append(Label(" VFS Configuration:", style="class:dialog.subtitle"))
            vfs_intermediate_backends = [("memory", "memory"), ("disk", "disk"), ("zarr", "zarr")] # Ensure these match VFSConfig options
            vfs_intermediate_backend_radios = RadioList(values=vfs_intermediate_backends, default_value=self.global_config.vfs.default_intermediate_backend)
            self.global_config_widgets['vfs_default_intermediate_backend'] = vfs_intermediate_backend_radios
            global_config_content.extend([Label("  Default Intermediate Backend:"), vfs_intermediate_backend_radios])

            vfs_materialization_backends = [("disk", "disk"), ("zarr", "zarr")] # Ensure these match VFSConfig options
            vfs_materialization_backend_radios = RadioList(values=vfs_materialization_backends, default_value=self.global_config.vfs.default_materialization_backend)
            self.global_config_widgets['vfs_default_materialization_backend'] = vfs_materialization_backend_radios
            global_config_content.extend([Label("  Default Materialization Backend:"), vfs_materialization_backend_radios])
            
            vfs_root_path_area = TextArea(text=self.global_config.vfs.persistent_storage_root_path or "", multiline=False, height=1, prompt="Path or empty: ")
            self.global_config_widgets['vfs_persistent_storage_root_path'] = vfs_root_path_area
            global_config_content.append(VSplit([Label("  Persistent Storage Root Path:", width=30), vfs_root_path_area]))

            global_config_content.append(Label(" Path Planning Configuration:", style="class:dialog.subtitle"))
            path_output_suffix_area = TextArea(text=self.global_config.path_planning.output_dir_suffix, multiline=False, height=1)
            self.global_config_widgets['path_planning_output_dir_suffix'] = path_output_suffix_area
            global_config_content.append(VSplit([Label("  Output Dir Suffix:", width=30), path_output_suffix_area]))

            path_pos_suffix_area = TextArea(text=self.global_config.path_planning.positions_dir_suffix, multiline=False, height=1)
            self.global_config_widgets['path_planning_positions_dir_suffix'] = path_pos_suffix_area
            global_config_content.append(VSplit([Label("  Positions Dir Suffix:", width=30), path_pos_suffix_area]))

            path_stitched_suffix_area = TextArea(text=self.global_config.path_planning.stitched_dir_suffix, multiline=False, height=1)
            self.global_config_widgets['path_planning_stitched_dir_suffix'] = path_stitched_suffix_area
            global_config_content.append(VSplit([Label("  Stitched Dir Suffix:", width=30), path_stitched_suffix_area]))
        
        # --- Dialog Buttons ---
        def save_tui_settings_handler():
            logger.info("ActionMenuPane: 'Save TUI Settings' button clicked.")
            self._clear_error()
            try:
                new_vim_mode = self.tui_settings_widgets['vim_mode'].checked
                new_tui_log_level = self.tui_settings_widgets['tui_log_level'].current_value
                new_editor_path = self.tui_settings_widgets['editor_path'].text.strip()

                self.state.vim_mode = new_vim_mode
                self.state.tui_log_level = new_tui_log_level
                self.state.editor_path = new_editor_path
                
                logger.info(f"TUI settings updated in state: vim_mode={new_vim_mode}, log_level={new_tui_log_level}, editor='{new_editor_path}'")

                tui_settings_data = {
                    'vim_mode': new_vim_mode,
                    'tui_log_level': new_tui_log_level,
                    'editor_path': new_editor_path
                }
                config_file_dir = Path.home() / ".config" / "openhcs"
                config_file_path = config_file_dir / "tui_settings.yaml"
                
                config_file_dir.mkdir(parents=True, exist_ok=True)
                with open(config_file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(tui_settings_data, f, indent=2, sort_keys=False)
                
                logger.info(f"TUI settings persisted to {config_file_path}")
                self._set_status(f"TUI settings saved to {config_file_path}")
                self.state.notify('tui_settings_changed', tui_settings_data)
                
                if hasattr(self, '_current_dialog_future') and self._current_dialog_future and not self._current_dialog_future.done():
                    self._current_dialog_future.set_result({'tui_settings_saved': True})
                
            except KeyError as e:
                err_msg = f"Error accessing settings widget: {e}."
                logger.error(err_msg, exc_info=True)
                self._show_error(err_msg)
                if hasattr(self, '_current_dialog_future') and self._current_dialog_future and not self._current_dialog_future.done():
                    self._current_dialog_future.set_exception(RuntimeError(err_msg))
            except Exception as e:
                err_msg = f"Error saving TUI settings: {e}"
                logger.error(err_msg, exc_info=True)
                self._show_error(err_msg)
                if hasattr(self, '_current_dialog_future') and self._current_dialog_future and not self._current_dialog_future.done():
                    self._current_dialog_future.set_exception(e)

        def close_dialog_handler():
            logger.info("Settings dialog Close/Cancel button clicked.")
            if hasattr(self, '_current_dialog_future') and self._current_dialog_future and not self._current_dialog_future.done():
                self._current_dialog_future.set_result(None) # Indicate cancellation
            # else:
            #      self._show_error("Dialog close is a placeholder (no future).") # No longer needed if future is always created

        dialog_buttons = [
            Button(text="Save TUI Settings", handler=lambda: get_app().create_background_task(save_tui_settings_handler())),
            Button(text="Save Global Config", handler=lambda: get_app().create_background_task(self._apply_and_save_global_config_settings())),
            Button(text="Close", handler=close_dialog_handler),
        ]

        dialog_body_content = tui_settings_content + [Window(height=1, char='─')] + global_config_content
        dialog_body = HSplit(dialog_body_content, padding=D(preferred=1, max=1))
        
        return Dialog(
            title="Settings",
            body=ScrollablePane(content=dialog_body),
            buttons=dialog_buttons,
            width=D(preferred=80, max=100),
            modal=True
        )

    async def _apply_and_save_global_config_settings(self):
        """
        Collects data from global config widgets, validates, creates new config,
        notifies for propagation, and calls persistence.
        """
        logger.info("Attempting to apply and save global config settings.")
        self._clear_error() # Clear previous dialog/pane errors
        errors = []
        
        # --- Collect Data & Validate ---
        try:
            # Num Workers
            num_workers_str = self.global_config_widgets['num_workers'].text
            if not num_workers_str.isdigit() or int(num_workers_str) <= 0:
                errors.append("Num Workers must be a positive integer.")
            num_workers = int(num_workers_str) if num_workers_str.isdigit() else 0 # Temp for creation

            # VFS
            vfs_intermediate_backend = self.global_config_widgets['vfs_default_intermediate_backend'].current_value
            vfs_materialization_backend = self.global_config_widgets['vfs_default_materialization_backend'].current_value
            vfs_root_path_str = self.global_config_widgets['vfs_persistent_storage_root_path'].text
            vfs_persistent_storage_root_path = vfs_root_path_str.strip() if vfs_root_path_str.strip() else None

            # Path Planning
            path_output_suffix = self.global_config_widgets['path_planning_output_dir_suffix'].text
            if not path_output_suffix.strip(): errors.append("Output Dir Suffix cannot be empty.")
            
            path_pos_suffix = self.global_config_widgets['path_planning_positions_dir_suffix'].text
            if not path_pos_suffix.strip(): errors.append("Positions Dir Suffix cannot be empty.")

            path_stitched_suffix = self.global_config_widgets['path_planning_stitched_dir_suffix'].text
            if not path_stitched_suffix.strip(): errors.append("Stitched Dir Suffix cannot be empty.")

        except KeyError as e:
            errors.append(f"A settings widget is missing: {e}. Dialog might not be fully initialized.")
            logger.error(f"Missing settings widget: {e}", exc_info=True)
        except ValueError: # Catches int conversion error for num_workers
             errors.append("Invalid numeric value for Num Workers.")
        except Exception as e: # Catch any other unexpected errors during data collection
            errors.append(f"Unexpected error collecting settings: {str(e)}")
            logger.error(f"Unexpected error collecting settings: {e}", exc_info=True)


        if errors:
            # TODO: Enhance UX by displaying validation errors directly within the settings dialog.
            # This would involve:
            # 1. Modifying `_create_main_settings_dialog` to include an error message Label/TextArea within the Dialog's layout.
            # 2. Updating this handler to set the text of that error Label/TextArea when `errors` is populated.
            # 3. Ensuring the dialog content refreshes to show the message.
            # For now, errors are shown in the ActionMenuPane's main error banner.
            error_summary = "Validation Errors: " + "; ".join(errors)
            self._show_error(error_summary)
            logger.warning(f"Global config validation failed: {error_summary}")
            # Do not close the dialog or resolve the future if validation fails
            return

        # --- Create New Config Objects ---
        try:
            new_vfs_config = VFSConfig(
                default_intermediate_backend=vfs_intermediate_backend,
                default_materialization_backend=vfs_materialization_backend,
                persistent_storage_root_path=vfs_persistent_storage_root_path
            )
            new_path_planning_config = PathPlanningConfig(
                output_dir_suffix=path_output_suffix.strip(), # Use stripped values
                positions_dir_suffix=path_pos_suffix.strip(),
                stitched_dir_suffix=path_stitched_suffix.strip()
            )
            new_global_config = GlobalPipelineConfig(
                num_workers=num_workers, # Already validated as int > 0
                path_planning=new_path_planning_config,
                vfs=new_vfs_config
            )
        except Exception as e:
            self._show_error(f"Error creating new config objects: {e}")
            logger.error(f"Error creating new config objects: {e}", exc_info=True)
            return # Do not proceed if config creation fails

        # --- Notify for Propagation & Persist ---
        logger.info(f"New GlobalPipelineConfig created: {new_global_config}")
        self.state.notify('global_config_needs_update', new_global_config) # Event for TUI Launcher
        
        await self._persist_global_config_to_file(new_global_config) # Stub for now (Task 2.3)

        self._set_status("Global config updated and change notified for propagation.")
        
        # Close the dialog by resolving the future with a success indicator
        if hasattr(self, '_current_dialog_future') and self._current_dialog_future and not self._current_dialog_future.done():
            self._current_dialog_future.set_result({'global_config_saved': True})

    async def _persist_global_config_to_file(self, config_to_save: GlobalPipelineConfig):
        """
        Placeholder for persisting the GlobalPipelineConfig to a file.
        This corresponds to Task 2.3 in plan02_phase2_editable_global_config.md.
        """
        config_file_dir = Path.home() / ".config" / "openhcs"
        config_file_path = config_file_dir / "global_pipeline_config.yaml"

        try:
            logger.info(f"Attempting to persist GlobalPipelineConfig to {config_file_path}")
            
            # Ensure the directory exists
            # FileManager might handle this, but explicit creation is safer for user config dirs.
            # However, self.filemanager.write_file should ideally handle directory creation if it's part of its contract.
            # For simplicity here, we'll assume self.filemanager.write_file can create parent dirs or expect them.
            # A more robust approach would be:
            # if not self.filemanager.exists(config_file_dir):
            # self.filemanager.mkdir(config_file_dir, recursive=True, backend='disk') # Assuming mkdir supports backend
            
            # For user-facing config files, direct os-level path operations for directory creation are common.
            # Let's use os.makedirs for ensuring config directory exists, as FileManager might be VFS-focused.
            # This is a bit of a gray area if FileManager is strictly for VFS.
            # For user config, direct path ops are often clearer.
            config_file_dir.mkdir(parents=True, exist_ok=True)


            config_dict = dataclasses.asdict(config_to_save)
            yaml_str = yaml.dump(config_dict, indent=2, sort_keys=False)

            # Using self.filemanager to write, assuming it can handle absolute paths
            # and the 'disk' backend correctly writes to the local filesystem.
            # If self.filemanager is strictly for VFS paths relative to a root, this needs adjustment.
            # For user config files, direct file I/O might be more straightforward if FileManager adds complexity.
            # Let's assume self.filemanager is versatile enough for this.
            
            # await self.filemanager.write_file_async(str(config_file_path), yaml_str, backend="disk")
            # Since filemanager operations are not consistently async, using synchronous for now.
            # If filemanager has async write, use it. Otherwise, consider running in executor.
            
            # Using standard Python file I/O for user config as it's simpler and direct.
            with open(config_file_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)

            logger.info(f"GlobalPipelineConfig successfully persisted to {config_file_path}")
            self._set_status("Global config saved to file.") # Update status

        except Exception as e:
            error_msg = f"Failed to persist GlobalPipelineConfig to {config_file_path}: {e}"
            logger.error(error_msg, exc_info=True)
            self._show_error(error_msg) # Show error in the TUI

    def __pt_container__(self): # Special method for prompt_toolkit to get the container
        return self.container