"""
Core Command Infrastructure for the OpenHCS TUI.

This module defines the base `Command` protocol that all TUI actions
will implement. Commands encapsulate the logic for user-triggered actions,
promoting separation of concerns and making UI components leaner.
"""
from typing import Protocol, Any, TYPE_CHECKING, List, Optional, Dict
import uuid
# import pickle # Replaced by adapter methods for load/save pipeline
# from pathlib import Path # Path might still be used for user input, but core interaction via adapter
from pathlib import Path

# TUI specific imports
from openhcs.tui.utils import show_error_dialog, prompt_for_path_dialog
from prompt_toolkit.shortcuts import message_dialog # Keep for UI feedback
from prompt_toolkit.application import get_app # Keep for UI feedback

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    # Import adapter interfaces and TUI data types
    from .interfaces import CoreApplicationAdapterInterface, CoreOrchestratorAdapterInterface, CoreStepData, CorePlateData
    # Removed core type imports like ProcessingContext, PipelineOrchestrator, AbstractStep, FunctionStep

# Removed: Actual core imports like FunctionStep, AbstractStep, FUNC_REGISTRY
# Removed: SHARED_EXECUTOR (should be in core_adapters.py)
import asyncio # Still needed for async command execution
import logging # Keep for logging

logger = logging.getLogger(__name__) # Ensure logger is defined


class Command(Protocol):
    """
    Protocol for a TUI command.

    Commands are responsible for executing an action, potentially interacting
    with TUIState and core adapters. They can also determine if they are
    currently executable.
    """

    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"], # Can be None if no plate is active
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        """
        Execute the command's action.

        Args:
            app_adapter: Adapter for core application-level interactions.
            plate_adapter: Adapter for active plate/orchestrator interactions. None if no plate context.
            ui_state: The current TUIState instance.
            **kwargs: Additional arguments specific to the command.
        """
        ...

    def can_execute(self, ui_state: "TUIState") -> bool:
        """
        Determine if the command can currently be executed.
        Defaults to True. UI elements can use this to enable/disable themselves.

        Args:
            ui_state: The current TUIState instance.

        Returns:
            True if the command can be executed, False otherwise.
        """
        return True

# Example of a concrete command (for illustration, will be expanded later)
# class NoOpCommand(Command):
#     async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
#         print(f"NoOpCommand executed with state: {state}, context: {context}, kwargs: {kwargs}")
#
#     def can_execute(self, state: "TUIState") -> bool:
#         return True
from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit.application import get_app

# Assuming GlobalSettingsEditorDialog is refactored to take app_adapter if it needs to save.
from openhcs.tui.dialogs.global_settings_editor import GlobalSettingsEditorDialog
# Removed: from openhcs.core.config import GlobalPipelineConfig (GlobalPipelineConfig is now opaque data)


class ShowGlobalSettingsDialogCommand(Command):
    """
    Command to show the Global Settings Editor dialog.
    Uses CoreApplicationAdapterInterface to fetch/update settings.
    """
    def __init__(self, ui_state: "TUIState", app_adapter: "CoreApplicationAdapterInterface"):
        """
        Initialize the command.
        Args:
            ui_state: The TUIState instance.
            app_adapter: The core application adapter.
        """
        # Store references if needed by the command itself, or rely on them being passed during execute.
        # For commands instantiated once and reused, storing is fine.
        self.ui_state = ui_state
        self.app_adapter = app_adapter

    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        
        current_global_config = await app_adapter.get_global_config() # Assumed method
        if current_global_config is None:
            await message_dialog(title="Error", text="Global configuration is not available from core.").run_async()
            return

        dialog = GlobalSettingsEditorDialog(
            initial_config=current_global_config, # Pass opaque config data
            app_adapter=app_adapter, # Dialog needs adapter to save changes
            ui_state=ui_state
        )
        
        # The dialog's show method returns the new config data if saved, or None.
        # The dialog itself, upon saving, should call app_adapter.update_global_config()
        # and then ui_state.notify('global_config_changed', new_config_data).
        # This command then notifies 'global_config_needs_update' for the launcher.
        new_config_data: Optional[Dict[str, Any]] = await dialog.show()

        if new_config_data:
            # This notification is for the launcher to potentially re-bind the config.
            await ui_state.notify('global_config_needs_update', new_config_data)
            # ui_state.global_config should have been updated by the 'global_config_changed' event
            # that the dialog (or its save command) should have triggered.

    def can_execute(self, ui_state: "TUIState") -> bool:
        # Depends on whether global_config is considered always available or if adapter can be unavailable
        return True # Assuming app_adapter is always available if TUI is running.


class ShowHelpCommand(Command):
    """
    Command to show a simple help message/dialog.
    """
    # No specific __init__ needed if it doesn't store custom state.
    # It will receive ui_state and adapters during execute.

    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        help_text = (
            "OpenHCS TUI - Help\n\n"
            "Controls:\n"
            "- Use mouse or Tab/Shift-Tab to navigate.\n"
            "- Arrow keys for lists and menus.\n"
            "- Enter to activate buttons/menu items.\n"
            "- Ctrl-C to Exit.\n\n"
            "Workflow:\n"
            "1. Add Plate(s) using Plate Manager [add] button.\n"
            "2. Select a plate, then [init] it.\n"
            "3. Add steps to its pipeline using Pipeline Editor [add] button.\n"
            "4. [edit] steps to configure them.\n"
            "5. [compile] the plate's pipeline.\n"
            "6. [run] the compiled pipeline.\n\n"
            "Global Settings: Access via top menu to change default configurations."
        )
        await message_dialog(
            title="OpenHCS Help",
            text=help_text
        ).run_async()

    def can_execute(self, state: "TUIState") -> bool:
        return True
# Forward references are handled by TYPE_CHECKING block at the top for adapter interfaces.

# --- PlateManagerPane Commands ---

class ShowAddPlateDialogCommand(Command):
    """Command to show the 'Add Plate' dialog."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        # The dialog itself will likely use app_adapter to list files or interact with storage.
        # This command primarily signals the TUI to display the relevant dialog.
        logger.info("ShowAddPlateDialogCommand: Triggered.")
        # The actual dialog showing logic is often part of the UI component (PlateManagerPane)
        # that owns the button. This command might just notify the TUIState.
        # Or, if PlateDialogManager is a TUI service, it could be accessed via app_adapter or ui_state.
        await ui_state.notify("show_add_plate_dialog_requested", {"app_adapter": app_adapter})


class DeleteSelectedPlatesCommand(Command):
    """Command to delete selected plate(s) using the app_adapter."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        logger.info("DeleteSelectedPlatesCommand: Triggered.")
        # Assuming 'selected_plates_for_action' in ui_state now holds List[CorePlateData] or List[Dict]
        selected_plates_data: Optional[List[Dict]] = getattr(ui_state, 'selected_plates_for_action', None)
        if not selected_plates_data:
            await message_dialog(title="Info", text="No plates selected for deletion.").run_async()
            return

        plate_ids_to_delete = [p.get('plate_id') for p in selected_plates_data if p.get('plate_id')]
        plate_names = [p.get('name', p.get('plate_id', 'Unknown Plate')) for p in selected_plates_data]

        if not plate_ids_to_delete:
            await message_dialog(title="Error", text="Selected items are missing plate IDs.").run_async()
            return

        confirm_dialog = message_dialog(
            title="Confirm Delete",
            text=f"Are you sure you want to delete selected plate(s):\n{', '.join(plate_names)}?",
            buttons=[("Yes", True), ("No", False)]
        )
        result = await confirm_dialog.run_async()
        if result:
            try:
                # Assuming app_adapter has a method to delete plates by ID
                success_flags = await app_adapter.delete_plates_by_id(plate_ids_to_delete)
                # Process success_flags, notify ui_state about changes
                # For simplicity, assume it raises error on failure or returns overall status
                await ui_state.notify("plates_deleted_operation_finished", {"deleted_ids": plate_ids_to_delete, "success": True}) # Or based on success_flags
                logger.info(f"Deletion requested for plate IDs: {plate_ids_to_delete}")
            except Exception as e:
                logger.error(f"Error deleting plates: {e}", exc_info=True)
                await show_error_dialog(title="Deletion Error", message=f"Failed to delete plates: {e}", app_state=ui_state)
                await ui_state.notify("plates_deleted_operation_finished", {"deleted_ids": plate_ids_to_delete, "success": False, "error": str(e)})


    def can_execute(self, ui_state: "TUIState") -> bool:
        return getattr(ui_state, 'selected_plates_for_action', None) is not None


class ShowEditPlateConfigDialogCommand(Command):
    """Command to show the 'Edit Plate Config' dialog for the selected plate."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        logger.info("ShowEditPlateConfigDialogCommand: Triggered.")
        active_plate_id = getattr(ui_state, 'active_plate_id', None)
        if not active_plate_id: 
            await message_dialog(title="Error", text="No active plate selected to edit configuration.").run_async()
            return
        # Notify AppController to handle showing the editor via TUIState.set_active_editor
        await ui_state.notify("request_plate_config_editor", {"plate_id": active_plate_id})

    def can_execute(self, ui_state: "TUIState") -> bool:
        return getattr(ui_state, 'active_plate_id', None) is not None


class InitializePlatesCommand(Command):
    """Command to initialize the active plate using the plate_adapter."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        logger.info("InitializePlatesCommand: Triggered.")
        if not plate_adapter:
            await ui_state.notify('error', {'message': 'No active plate selected to initialize.', 'source': self.__class__.__name__})
            await message_dialog(title="Error", text="No active plate selected to initialize.").run_async()
            return

        plate_id = ui_state.active_plate_id # Should be set if plate_adapter is available
        plate_name = getattr(ui_state.selected_plate, 'name', plate_id) # Get name from selected_plate data

        try:
            await ui_state.notify('operation_status_changed', {'status': 'running', 'message': f'Initializing plate {plate_name}...', 'source': self.__class__.__name__})
            
            init_success = await plate_adapter.initialize_plate() # Assumed adapter method

            if init_success:
                logger.info(f"Plate '{plate_name}' (ID: {plate_id}) initialized successfully via adapter.")
                await ui_state.notify('plate_status_changed', {
                    'plate_id': plate_id, 'status': 'initialized', 'message': 'Plate initialized successfully.'
                })
                await ui_state.notify('operation_status_changed', {'status': 'idle', 'message': f'Plate {plate_name} initialization complete.', 'source': self.__class__.__name__})
            else:
                raise Exception("Core adapter reported initialization failure.")

        except Exception as e:
            logger.error(f"Error initializing plate '{plate_name}' (ID: {plate_id}): {e}", exc_info=True)
            await ui_state.notify('plate_status_changed', {
                'plate_id': plate_id, 'status': 'error_init', 'message': f'Error initializing plate: {str(e)}'
            })
            await ui_state.notify('operation_status_changed', {'status': 'idle', 'message': f'Plate {plate_name} initialization failed.', 'source': self.__class__.__name__})
            await show_error_dialog(title="Initialization Error", message=f"Failed to initialize plate '{plate_name}':\n{e}", app_state=ui_state)

    def can_execute(self, ui_state: "TUIState") -> bool:
        # Can execute if there's an active plate and it's not already initialized (status check needed in TUIState or via adapter)
        if ui_state.active_plate_id and ui_state.selected_plate:
            # Example: Check status from ui_state.selected_plate which should be CorePlateData
            plate_status = ui_state.selected_plate.get('status')
            return plate_status not in ['initialized', 'compiling', 'running', 'compiled_ok', 'run_completed'] # Simplified
        return False


class CompilePlatesCommand(Command):
    """Command to compile pipeline for the active plate using plate_adapter."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        logger.info("CompilePlatesCommand: Triggered.")
        if not plate_adapter:
            await message_dialog(title="Info", text="No active plate selected to compile.").run_async()
            return

        plate_id = ui_state.active_plate_id
        # Assuming TUIState.current_pipeline_definition holds the List[CoreStepData]
        pipeline_to_compile: Optional[List["CoreStepData"]] = ui_state.current_pipeline_definition

        if not pipeline_to_compile:
            logger.error(f"CompilePlatesCommand: Pipeline definition missing for plate '{plate_id}'.")
            await message_dialog(title="Error", text=f"Pipeline definition missing for plate '{plate_id}'.").run_async()
            return
        
        # Check current plate status from ui_state.selected_plate
        current_plate_status = getattr(ui_state.selected_plate, 'status', None)
        if current_plate_status != 'initialized': # Or a more robust check
             await message_dialog(title="Info", text=f"Plate '{plate_id}' must be initialized before compiling. Current status: {current_plate_status}").run_async()
             return

        try:
            await ui_state.notify("plate_operation_started", {"plate_id": plate_id, "operation": "compile"})
            
            # The adapter method takes the pipeline definition (list of CoreStepData)
            # and returns some representation of the compiled pipeline or status.
            compiled_data_representation = await plate_adapter.compile_pipeline(pipeline_to_compile)
            
            logger.info(f"Plate '{plate_id}' compiled successfully via adapter.")
            # Store compiled representation in TUIState if needed for 'run'
            ui_state.compiled_pipeline_representation = compiled_data_representation # Store opaque compiled data
            ui_state.is_compiled = True # Update TUIState flag
            await ui_state.notify("plate_status_changed", {"plate_id": plate_id, "status": "compiled_ok"})
            await ui_state.notify("is_compiled_changed", True)

        except Exception as e:
            logger.error(f"Error compiling plate '{plate_id}': {e}", exc_info=True)
            ui_state.is_compiled = False
            await ui_state.notify("plate_status_changed", {"plate_id": plate_id, "status": "error_compile", "message": str(e)})
            await ui_state.notify("is_compiled_changed", False)
            await message_dialog(title="Compilation Error", text=f"Failed to compile plate '{plate_id}':\n{e}").run_async()
        finally:
            await ui_state.notify("plate_operation_finished", {"plate_id": plate_id, "operation": "compile"})

    def can_execute(self, ui_state: "TUIState") -> bool:
        if ui_state.active_plate_id and ui_state.selected_plate:
            plate_status = ui_state.selected_plate.get('status')
            # Example: Can compile if initialized or if compilation previously failed
            return plate_status in ['initialized', 'error_compile'] and bool(ui_state.current_pipeline_definition)
        return False


class RunPlatesCommand(Command):
    """Command to run compiled pipeline for the active plate using plate_adapter."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        logger.info("RunPlatesCommand: Triggered.")
        if not plate_adapter:
            await message_dialog(title="Info", text="No active plate selected to run.").run_async()
            return

        plate_id = ui_state.active_plate_id
        
        # Check status from ui_state or a dedicated flag like ui_state.is_compiled
        if not ui_state.is_compiled or (ui_state.selected_plate and ui_state.selected_plate.get('status') != 'compiled_ok'):
             await message_dialog(title="Info", text=f"Pipeline for plate '{plate_id}' must be compiled before running.").run_async()
             return
        
        # The compiled_pipeline_representation might be needed by the adapter
        compiled_representation = ui_state.compiled_pipeline_representation

        try:
            await ui_state.notify("plate_operation_started", {"plate_id": plate_id, "operation": "run"})
            ui_state.is_running = True
            await ui_state.notify("run_status_changed", {"plate_id": plate_id, "running": True})

            # Adapter's run method might need the compiled representation if core is stateless here
            await plate_adapter.run_pipeline(compiled_representation) 
            
            logger.info(f"Plate '{plate_id}' run completed successfully via adapter.")
            await ui_state.notify("plate_status_changed", {"plate_id": plate_id, "status": "run_completed"})
        except Exception as e:
            logger.error(f"Error running plate '{plate_id}': {e}", exc_info=True)
            await ui_state.notify("plate_status_changed", {"plate_id": plate_id, "status": "error_run", "message": str(e)})
            await message_dialog(title="Run Error", text=f"Failed to run plate '{plate_id}':\n{e}").run_async()
        finally:
            ui_state.is_running = False
            await ui_state.notify("run_status_changed", {"plate_id": plate_id, "running": False})
            await ui_state.notify("plate_operation_finished", {"plate_id": plate_id, "operation": "run"})

    def can_execute(self, ui_state: "TUIState") -> bool:
        if ui_state.active_plate_id and ui_state.selected_plate and not ui_state.is_running:
            return ui_state.is_compiled and ui_state.selected_plate.get('status') == 'compiled_ok'
        return False

# Removed old logger definition here

# --- PipelineEditorPane Commands ---

class AddStepCommand(Command):
    """Command to add a new step to the current pipeline using plate_adapter."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        logger.info("AddStepCommand: Triggered.")
        if not plate_adapter:
            await ui_state.notify('error', {'message': 'No active plate/pipeline to add a step to.', 'source': self.__class__.__name__})
            await show_error_dialog(title="Error", message="No active plate/pipeline to add a step to.", app_state=ui_state)
            return

        # The adapter will be responsible for creating a default step structure.
        # The TUI no longer knows about FunctionStep or FUNC_REGISTRY.
        try:
            # Adapter method to create and add a new default step.
            # It returns the new step data (CoreStepData) or the full updated pipeline.
            # Let's assume it returns the new step_data.
            new_step_data: Optional["CoreStepData"] = await plate_adapter.add_new_step_to_pipeline()

            if new_step_data:
                # Update TUIState's current_pipeline_definition.
                # The adapter might have already updated the canonical source,
                # so TUIState might need to re-fetch or append.
                # For now, assume add_new_step_to_pipeline returns the step and TUI appends.
                if ui_state.current_pipeline_definition is None:
                    ui_state.current_pipeline_definition = []
                ui_state.current_pipeline_definition.append(new_step_data)
                
                await ui_state.notify('steps_updated', {
                    'pipeline_definition': ui_state.current_pipeline_definition, # Send full list
                    'action': 'add',
                    'added_step_id': new_step_data.get('id')
                })
                await ui_state.notify('operation_status_changed', {'status': 'idle', 'message': f"Added step: {new_step_data.get('name', 'New Step')}", 'source': self.__class__.__name__})
                logger.info(f"Added new step '{new_step_data.get('name')}' (ID: {new_step_data.get('id')}) via adapter.")
            else:
                await show_error_dialog(title="Error", message="Failed to add new step via adapter.", app_state=ui_state)

        except Exception as e:
            logger.error(f"Error adding new step: {e}", exc_info=True)
            await show_error_dialog(title="Add Step Error", message=f"Failed to add step: {e}", app_state=ui_state)


    def can_execute(self, ui_state: "TUIState") -> bool:
        return ui_state.active_plate_id is not None


class DeleteSelectedStepsCommand(Command):
    """Command to delete selected step(s) from the current pipeline using plate_adapter."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        logger.info("DeleteSelectedStepsCommand: Triggered.")
        if not plate_adapter:
            await show_error_dialog(title="Error", message="No active pipeline to delete steps from.", app_state=ui_state)
            return

        # steps_to_delete_data is List[CoreStepData] from PipelineEditorPane or TUIState.selected_step
        steps_to_delete_data: List["CoreStepData"] = kwargs.get('steps_to_delete', [])
        if not steps_to_delete_data and ui_state.selected_step:
            steps_to_delete_data = [ui_state.selected_step]

        if not steps_to_delete_data:
            await message_dialog(title="Info", text="No steps selected for deletion.").run_async()
            return

        step_names = [s.get('name', s.get('id', 'Unknown Step')) for s in steps_to_delete_data]
        step_ids_to_delete = [s.get('id') for s in steps_to_delete_data if s.get('id')]

        if not step_ids_to_delete:
            logger.warning("DeleteSelectedStepsCommand: No valid step IDs found for deletion.")
            return
            
        confirm_dialog = message_dialog(
            title="Confirm Delete Step(s)",
            text=f"Are you sure you want to delete selected step(s):\n{', '.join(step_names)}?",
            buttons=[("Yes", True), ("No", False)]
        )
        result = await confirm_dialog.run_async()
        if not result:
            logger.info("DeleteSelectedStepsCommand: Deletion cancelled by user.")
            return

        try:
            # Adapter method to delete steps by their IDs
            updated_pipeline: Optional[List["CoreStepData"]] = await plate_adapter.delete_steps_from_pipeline(step_ids_to_delete)

            if updated_pipeline is not None:
                ui_state.current_pipeline_definition = updated_pipeline
                await ui_state.notify('steps_updated', {'action': 'delete', 'deleted_ids': step_ids_to_delete, 'pipeline_definition': updated_pipeline})
                logger.info(f"Deleted steps with IDs: {step_ids_to_delete} via adapter.")
                # Clear selected step if it was deleted
                if ui_state.selected_step and ui_state.selected_step.get('id') in step_ids_to_delete:
                    await ui_state.set_selected_step_data(None)
            else:
                # This case implies an error or that the adapter couldn't perform the action
                await show_error_dialog(title="Error", message="Failed to delete steps or update pipeline via adapter.", app_state=ui_state)

        except Exception as e:
            logger.error(f"Error deleting steps {step_ids_to_delete}: {e}", exc_info=True)
            await show_error_dialog(title="Delete Step Error", message=f"Failed to delete steps: {e}", app_state=ui_state)

    def can_execute(self, ui_state: "TUIState") -> bool:
        # Requires an active plate and either a selected step in ui_state or steps_to_delete in kwargs
        # The kwargs part is tricky for a generic can_execute. Focus on ui_state.
        return ui_state.active_plate_id is not None and ui_state.selected_step is not None


class ShowEditStepDialogCommand(Command):
    """Command to trigger editing of the selected step."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        logger.info("ShowEditStepDialogCommand: Triggered.")
        # Use active_step_data from TUIState, which should be CoreStepData
        active_step_data: Optional["CoreStepData"] = getattr(ui_state, 'active_step_data', None)

        if not ui_state.active_plate_id or not active_step_data: 
            await message_dialog(title="Error", text="No step selected or no active pipeline to edit a step from.").run_async()
            return
        
        # Notify AppController to handle showing the editor via TUIState.set_active_editor
        await ui_state.notify("request_step_editor", {"step_data": active_step_data})

    def can_execute(self, ui_state: "TUIState") -> bool:
        # Check active_step_data instead of selected_step
        return getattr(ui_state, 'active_step_data', None) is not None and \
               getattr(ui_state, 'active_plate_id', None) is not None


class LoadPipelineCommand(Command):
    """Command to load a pipeline definition for the active plate using plate_adapter."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        logger.info("LoadPipelineCommand: Triggered.")
        if not plate_adapter:
            await message_dialog(title="Error", text="No active plate to load a pipeline into.").run_async()
            return

        # Path prompt remains the same
        file_path_str = await prompt_for_path_dialog(
            title="Load Pipeline", prompt_message="Enter path to .pipeline file:", app_state=ui_state
        )
        if not file_path_str:
            logger.info("LoadPipelineCommand: Load operation cancelled by user.")
            await ui_state.notify('info', {'message': "Load pipeline operation cancelled.", 'source': self.__class__.__name__})
            return

        file_path = Path(file_path_str)
        if not file_path.exists() or not file_path.is_file():
            await show_error_dialog("Load Pipeline Error", f"File not found or is not a file: {file_path}", app_state=ui_state)
            return

        try:
            # Adapter method handles loading from storage, deserialization, and validation.
            # It should return List[CoreStepData] or raise an error.
            loaded_pipeline: Optional[List["CoreStepData"]] = await plate_adapter.load_pipeline_from_storage(str(file_path))

            if loaded_pipeline is not None:
                ui_state.current_pipeline_definition = loaded_pipeline
                await ui_state.notify('steps_updated', {'action': 'load_pipeline', 'pipeline_definition': loaded_pipeline})
                await ui_state.notify('operation_status_changed', {'message': f"Pipeline loaded from {file_path}", 'status': 'success', 'source': self.__class__.__name__})
                logger.info(f"Pipeline loaded from {file_path} via adapter.")
            else:
                # This case implies an error handled by the adapter but returned None (e.g. validation failed)
                await show_error_dialog("Load Pipeline Error", f"Failed to load a valid pipeline from {file_path}. Adapter returned no data.", app_state=ui_state)

        except Exception as e: # Catch errors from adapter call
            logger.error(f"Failed to load pipeline from {file_path} via adapter: {e}", exc_info=True)
            await show_error_dialog("Load Pipeline Error", f"Failed to load pipeline: {e}", app_state=ui_state)

    def can_execute(self, ui_state: "TUIState") -> bool:
        return ui_state.active_plate_id is not None


class SavePipelineCommand(Command):
    """Command to save the current pipeline definition using plate_adapter."""
    async def execute(self,
                      app_adapter: "CoreApplicationAdapterInterface",
                      plate_adapter: Optional["CoreOrchestratorAdapterInterface"],
                      ui_state: "TUIState",
                      **kwargs: Any) -> None:
        logger.info("SavePipelineCommand: Triggered.")
        if not plate_adapter:
            await message_dialog(title="Error", text="No active pipeline to save.").run_async()
            return

        current_pipeline: Optional[List["CoreStepData"]] = ui_state.current_pipeline_definition
        if not current_pipeline:
            await message_dialog(title="Info", text="Pipeline is empty. Nothing to save.").run_async()
            return

        # Path prompt remains similar, or adapter might use a default path
        file_path_str = await prompt_for_path_dialog(
            title="Save Pipeline As", prompt_message="Enter path to save .pipeline file:", app_state=ui_state,
            default_path=kwargs.get('default_save_path', f"{ui_state.active_plate_id}_pipeline.json") # Example default
        )
        if not file_path_str:
            logger.info("SavePipelineCommand: Save operation cancelled by user.")
            return
        
        save_path = Path(file_path_str)

        try:
            # Adapter method handles serialization and saving to storage.
            success = await plate_adapter.save_pipeline_to_storage(current_pipeline, str(save_path))

            if success:
                await message_dialog(title="Success", text=f"Pipeline saved to:\n{save_path}").run_async()
                logger.info(f"Pipeline saved to {save_path} via adapter.")
                await ui_state.notify("pipeline_saved", {"path": str(save_path)})
            else:
                await show_error_dialog("Save Pipeline Error", f"Adapter failed to save pipeline to {save_path}.", app_state=ui_state)

        except Exception as e: # Catch errors from adapter call
            logger.error(f"SavePipelineCommand: Error saving pipeline to {save_path} via adapter: {e}", exc_info=True)
            await show_error_dialog("Save Pipeline Error", f"Could not save pipeline: {e}", app_state=ui_state)

    def can_execute(self, ui_state: "TUIState") -> bool:
        return ui_state.active_plate_id is not None and bool(ui_state.current_pipeline_definition)


# Command Registry (remains the same conceptually)
# Ensure all command instances are updated if their __init__ changes.
# For now, assuming commands are instantiated where they are used, or registry is populated
# with types and instantiated with new adapter args.

command_registry: Dict[str, Command] = {}

def register_command(name: str, command_instance: Command) -> None:
    """Registers a command instance."""
    if name in command_registry:
        logger.warning(f"Command '{name}' is already registered. Overwriting.")
    command_registry[name] = command_instance

def get_command(name: str) -> Optional[Command]:
    """Retrieves a command instance by name."""
    return command_registry.get(name)

# Example: Re-registering commands if they were instantiated globally (not typical for commands with state)
# This part is illustrative; actual registration depends on how commands are managed.
# If commands are stateless and only receive adapters/state via execute(), then type registration is fine.
# If commands have their own state via __init__ (like ShowGlobalSettingsCommand was), they need careful instantiation.

# For commands that used to take (state, context) in __init__ and are now (ui_state, app_adapter),
# their instantiation point (e.g., in OpenHCSTUI._initialize_components) needs to be updated.
# This diff focuses on the command definitions themselves.

# Removed old imports like AbstractStep, json, etc. as they are handled by adapters or not used.
# Path and asyncio are still used.