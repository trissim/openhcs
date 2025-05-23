"""
Core Command Infrastructure for the OpenHCS TUI.

This module defines the base `Command` protocol that all TUI actions
will implement. Commands encapsulate the logic for user-triggered actions,
promoting separation of concerns and making UI components leaner.
"""
from typing import Protocol, Any, TYPE_CHECKING, List, Optional # Added List and Optional
import uuid # For AddStepCommand
import pickle # For Load/Save Pipeline
from pathlib import Path # For Load/Save Pipeline

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator # For type hint
    from openhcs.core.steps.abstract import AbstractStep # For type hint
    from openhcs.core.steps.function_step import FunctionStep # For AddStepCommand

# Actual imports needed at runtime by commands
import asyncio # For running sync code in executor
from concurrent.futures import ThreadPoolExecutor # For InitializePlatesCommand
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.abstract import AbstractStep
from openhcs.tui.utils import show_error_dialog, prompt_for_path_dialog # For Load/Save
from openhcs.processing.func_registry import FUNC_REGISTRY # For AddStepCommand

# Shared ThreadPoolExecutor for running synchronous orchestrator methods
SHARED_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="tui-cmd-executor")


class Command(Protocol):
    """
    Protocol for a TUI command.

    Commands are responsible for executing an action, potentially interacting
    with TUIState, ProcessingContext, or other services. They can also
    determine if they are currently executable.
    """

    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        """
        Execute the command's action.

        Args:
            state: The current TUIState instance.
            context: The current ProcessingContext instance.
            **kwargs: Additional arguments specific to the command.
        """
        ...

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can currently be executed.
        Defaults to True. UI elements can use this to enable/disable themselves.

        Args:
            state: The current TUIState instance.

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

# Attempt to import dialogs and config, handle potential circularity if commands are also imported by them
# This might require restructuring imports if circular dependencies arise during full implementation.
try:
    from openhcs.tui.dialogs.global_settings_editor import GlobalSettingsEditorDialog
    from openhcs.core.config import GlobalPipelineConfig
except ImportError:
    # This might happen if commands.py is imported by one of these modules before they are fully defined.
    # For now, we'll assume this will resolve or use forward references if truly needed.
    GlobalSettingsEditorDialog = Any
    GlobalPipelineConfig = Any


class ShowGlobalSettingsDialogCommand(Command):
    """
    Command to show the Global Settings Editor dialog.
    """
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        if state.global_config is None:
            # This should ideally not happen if TUIState is properly initialized
            await message_dialog(
                title="Error",
                text="Global configuration is not available."
            ).run_async()
            return

        dialog = GlobalSettingsEditorDialog(
            initial_config=state.global_config,
            state=state # Pass state if dialog needs to notify on changes directly or access other state
        )
        # The dialog's show() method should be async and return the result
        # or handle notifications internally.
        # For now, assuming it updates state.global_config if changes are saved
        # and notifies 'global_config_needs_update'
        result_config: GlobalPipelineConfig | None = await dialog.show()

        if result_config and isinstance(result_config, GlobalPipelineConfig):
            # If dialog returns the new config, update state and notify
            # This logic aligns with V4 plan (Phase 4.1)
            # where the dialog's save handler notifies 'global_config_changed' and 'global_config_needs_update'.
            # If the dialog itself handles notifications, this command might just show it.
            # For now, let's assume the command is responsible for the notification if dialog returns data.
            # This part might be refined when GlobalSettingsEditorDialog is fully implemented with commands.

            # As per plan 4.1: "MenuBar._on_settings handler correctly instantiates and shows this dialog,
            # then passes the result to OpenHCSTUILauncher via self.state.notify('global_config_needs_update', ...)"
            # The command now replaces the _on_settings handler.

            # The dialog itself (its _save_settings) should notify 'global_config_changed'.
            # This command ensures 'global_config_needs_update' is notified for the launcher.
            await state.notify('global_config_needs_update', result_config)
            # The TUIState's global_config reference should also be updated by the dialog or its save command.
            # state.global_config = result_config # This should be handled by the dialog's save logic / command

    def can_execute(self, state: "TUIState") -> bool:
        return state.global_config is not None


class ShowHelpCommand(Command):
    """
    Command to show a simple help message/dialog.
    """
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        # For now, a simple message dialog. This can be expanded later.
        # The content of the help message should be defined elsewhere.
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
# Forward references for type hints if TUIState or other complex types are defined later
# For now, using Any or string literals if full import causes issues.
if TYPE_CHECKING:
    from openhcs.tui.dialogs.plate_dialog_manager import PlateDialogManager
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    # from openhcs.tui.dialogs.plate_config_editor import PlateConfigEditorDialog # Define later

# --- PlateManagerPane Commands ---

class ShowAddPlateDialogCommand(Command):
    """Command to show the 'Add Plate' dialog."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        # PlateDialogManager should be available, perhaps via TUIState or passed via kwargs
        # For now, assuming it can be instantiated or accessed.
        # This command will trigger the dialog that uses the interactive browser (Phase 2)
        # The dialog manager itself will handle showing the dialog.
        # This command's primary role is to initiate that process.

        # Placeholder: Actual dialog invocation will depend on PlateDialogManager's final API
        # and how it's made available.
        # Example:
        # plate_dialog_manager = kwargs.get('plate_dialog_manager') # or state.plate_dialog_manager
        # if plate_dialog_manager:
        #     await plate_dialog_manager.show_add_plate_dialog(context.filemanager) # Pass FileManager
        # else:
        #     logger.error("ShowAddPlateDialogCommand: PlateDialogManager not available.")
        #     await message_dialog(title="Error", text="Cannot open Add Plate dialog.").run_async()

        # For now, let's just log and notify. The actual dialog showing will be part of Phase 2.
        # The PlateManagerPane's button handler will call this command.
        # The command then might interact with PlateDialogManager.
        logger.info("ShowAddPlateDialogCommand: Triggered.")
        await state.notify("show_add_plate_dialog_requested", {"file_manager": context.filemanager})


class DeleteSelectedPlatesCommand(Command):
    """Command to delete selected plate(s)."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        logger.info("DeleteSelectedPlatesCommand: Triggered.")
        selected_plates_data = getattr(state, 'selected_plates_for_action', None) # Assume state tracks this
        if not selected_plates_data:
            await message_dialog(title="Info", text="No plates selected for deletion.").run_async()
            return

        plate_names = [p.get('name', p.get('id', 'Unknown Plate')) for p in selected_plates_data]
        confirm_dialog = message_dialog(
            title="Confirm Delete",
            text=f"Are you sure you want to delete selected plate(s):\n{', '.join(plate_names)}?",
            buttons=[("Yes", True), ("No", False)]
        )
        result = await confirm_dialog.run_async()
        if result:
            await state.notify("delete_plates_requested", {"plates_to_delete": selected_plates_data})

    def can_execute(self, state: "TUIState") -> bool:
        return getattr(state, 'selected_plates_for_action', None) is not None


class ShowEditPlateConfigDialogCommand(Command):
    """Command to show the 'Edit Plate Config' dialog for the selected plate."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        logger.info("ShowEditPlateConfigDialogCommand: Triggered.")
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)
        if not active_orchestrator:
            await message_dialog(title="Error", text="No active plate selected to edit configuration.").run_async()
            return

        # This will eventually show PlateConfigEditorDialog (Phase 4.2)
        # For now, notify state.
        await state.notify("show_edit_plate_config_requested", {"orchestrator": active_orchestrator})

    def can_execute(self, state: "TUIState") -> bool:
        return getattr(state, 'active_orchestrator', None) is not None

class InitializePlatesCommand(Command):
    """Command to initialize selected plate(s)."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        logger.info("InitializePlatesCommand: Triggered.")
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)

        if not active_orchestrator:
            await state.notify('error', {'message': 'No active plate selected to initialize.', 'source': self.__class__.__name__})
            await message_dialog(title="Error", text="No active plate selected to initialize.").run_async() # User feedback
            return

        plate_name = active_orchestrator.plate_path.name
        plate_id = getattr(active_orchestrator, 'plate_id', plate_name) # Use plate_id if available

        try:
            await state.notify('operation_status_changed', {'status': 'running', 'message': f'Initializing plate {plate_name}...', 'source': self.__class__.__name__})
            
            loop = asyncio.get_event_loop()
            # Run the synchronous orchestrator.initialize() in the shared thread pool
            await loop.run_in_executor(SHARED_EXECUTOR, active_orchestrator.initialize)
            
            logger.info(f"Plate '{plate_name}' (ID: {plate_id}) initialized successfully by command.")
            await state.notify('plate_status_changed', {
                'plate_id': plate_id, 
                'status': 'initialized', 
                'message': 'Plate initialized successfully.'
            })
            await state.notify('operation_status_changed', {'status': 'idle', 'message': f'Plate {plate_name} initialization complete.', 'source': self.__class__.__name__})

        except Exception as e:
            logger.error(f"Error initializing plate '{plate_name}' (ID: {plate_id}): {e}", exc_info=True)
            await state.notify('plate_status_changed', {
                'plate_id': plate_id, 
                'status': 'error_init', 
                'message': f'Error initializing plate: {str(e)}'
            })
            await state.notify('operation_status_changed', {'status': 'idle', 'message': f'Plate {plate_name} initialization failed.', 'source': self.__class__.__name__})
            # Also show a user-facing dialog for the error
            await show_error_dialog(title="Initialization Error", message=f"Failed to initialize plate '{plate_name}':\n{e}", app_state=state)


    def can_execute(self, state: "TUIState") -> bool:
        # Can execute if there's an active orchestrator
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)
        if active_orchestrator:
            # Add more specific checks if needed, e.g., plate not already initialized
            # current_plate_status = getattr(active_orchestrator, 'status', None) # Assuming orchestrator has a status
            # return current_plate_status != 'initialized' 
            return True # For now, allow if orchestrator is present
        return False


class CompilePlatesCommand(Command):
    """Command to compile pipelines for selected plate(s)."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        logger.info("CompilePlatesCommand: Triggered.")
        # Similar to InitializePlatesCommand, iterates selected plates.
        # Calls orchestrator.compile_pipelines()
        # Replaces logic from ActionMenuPane._compile_handler()
        selected_orchestrators = kwargs.get('orchestrators_to_compile', [])
        if not selected_orchestrators:
             active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)
             if active_orchestrator:
                 selected_orchestrators = [active_orchestrator]

        if not selected_orchestrators:
            await message_dialog(title="Info", text="No plates selected to compile.").run_async()
            return

        for orchestrator in selected_orchestrators:
            plate_id = getattr(orchestrator, 'plate_id', 'Unknown Plate')
            if getattr(orchestrator, 'status', '') != 'initialized': # Or a more robust check
                 await message_dialog(title="Info", text=f"Plate '{plate_id}' must be initialized before compiling.").run_async()
                 continue
            try:
                await state.notify("plate_operation_started", {"plate_id": plate_id, "operation": "compile"})
                loop = asyncio.get_event_loop()

                if not hasattr(orchestrator, 'pipeline_definition') or not orchestrator.pipeline_definition:
                    logger.error(f"CompilePlatesCommand: Pipeline definition missing for plate '{plate_id}'.")
                    await message_dialog(title="Error", text=f"Pipeline definition missing for plate '{plate_id}'.").run_async()
                    await state.notify("plate_status_changed", {"plate_id": plate_id, "status": "error_compile", "message": "Pipeline definition missing."})
                    # Ensure finally block still runs for this orchestrator
                    raise ValueError("Pipeline definition missing") # Raise to be caught by outer try-except

                # Pass the orchestrator's pipeline_definition to its compile_pipelines method
                compiled_pipeline_data = await loop.run_in_executor(
                    None,
                    orchestrator.compile_pipelines,
                    orchestrator.pipeline_definition
                )
                logger.info(f"Plate '{plate_id}' compiled successfully.")
                # Store compiled_contexts on the orchestrator instance for later use by RunPlatesCommand
                orchestrator.last_compiled_contexts = compiled_pipeline_data

                await state.notify("plate_status_changed", {"plate_id": plate_id, "status": "compiled_ok"}) # No need to pass compiled_data in notification if stored on orchestrator
                state.is_compiled = True
                await state.notify("is_compiled_changed", True)

            except Exception as e:
                logger.error(f"Error compiling plate '{plate_id}': {e}", exc_info=True)
                await state.notify("plate_status_changed", {"plate_id": plate_id, "status": "error_compile", "message": str(e)})
                state.is_compiled = False
                await state.notify("is_compiled_changed", False)
                await message_dialog(title="Compilation Error", text=f"Failed to compile plate '{plate_id}':\n{e}").run_async()
            finally:
                await state.notify("plate_operation_finished", {"plate_id": plate_id, "operation": "compile"})

    def can_execute(self, state: "TUIState") -> bool:
        # Example: Can execute if active plate is initialized but not yet compiled
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)
        if active_orchestrator:
            # This status check needs to align with actual orchestrator status values
            return getattr(active_orchestrator, 'status', '') == 'initialized'
        return False


class RunPlatesCommand(Command):
    """Command to run compiled pipelines for selected plate(s)."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        logger.info("RunPlatesCommand: Triggered.")
        # Similar to CompilePlatesCommand.
        # Calls orchestrator.execute_compiled_plate()
        # Replaces logic from ActionMenuPane._run_handler()
        selected_orchestrators = kwargs.get('orchestrators_to_run', [])
        if not selected_orchestrators:
             active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)
             if active_orchestrator:
                 selected_orchestrators = [active_orchestrator]

        if not selected_orchestrators:
            await message_dialog(title="Info", text="No plates selected to run.").run_async()
            return

        for orchestrator in selected_orchestrators:
            plate_id = getattr(orchestrator, 'plate_id', 'Unknown Plate')
            # Add more robust status check from orchestrator if it has one (e.g., orchestrator.is_compiled)
            if not state.is_compiled and getattr(orchestrator, 'status', '') != 'compiled_ok':
                 await message_dialog(title="Info", text=f"Pipeline for plate '{plate_id}' must be compiled before running.").run_async()
                 continue
            try:
                await state.notify("plate_operation_started", {"plate_id": plate_id, "operation": "run"})
                state.is_running = True
                await state.notify("run_status_changed", {"plate_id": plate_id, "running": True})

                loop = asyncio.get_event_loop()

                if not hasattr(orchestrator, 'pipeline_definition') or not orchestrator.pipeline_definition:
                    logger.error(f"RunPlatesCommand: Stateless pipeline definition missing for plate '{plate_id}'.")
                    await message_dialog(title="Error", text=f"Stateless pipeline definition missing for plate '{plate_id}'. Was it compiled?").run_async()
                    raise ValueError("Stateless pipeline definition missing")

                if not hasattr(orchestrator, 'last_compiled_contexts') or not orchestrator.last_compiled_contexts:
                    logger.error(f"RunPlatesCommand: Compiled contexts missing for plate '{plate_id}'.")
                    await message_dialog(title="Error", text=f"Compiled contexts missing for plate '{plate_id}'. Was it compiled?").run_async()
                    raise ValueError("Compiled contexts missing")

                await loop.run_in_executor(
                    None,
                    orchestrator.execute_compiled_plate,
                    orchestrator.pipeline_definition, # The stateless pipeline
                    orchestrator.last_compiled_contexts # The compiled contexts
                )
                logger.info(f"Plate '{plate_id}' run completed successfully.")
                await state.notify("plate_status_changed", {"plate_id": plate_id, "status": "run_completed"})
            except Exception as e:
                logger.error(f"Error running plate '{plate_id}': {e}", exc_info=True)
                await state.notify("plate_status_changed", {"plate_id": plate_id, "status": "error_run", "message": str(e)})
                await message_dialog(title="Run Error", text=f"Failed to run plate '{plate_id}':\n{e}").run_async()
            finally:
                state.is_running = False
                await state.notify("run_status_changed", {"plate_id": plate_id, "running": False})
                await state.notify("plate_operation_finished", {"plate_id": plate_id, "operation": "run"})

    def can_execute(self, state: "TUIState") -> bool:
        # Example: Can execute if active plate is compiled and not currently running
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)
        if active_orchestrator and not state.is_running:
             # This status check needs to align with actual orchestrator status values
            return state.is_compiled or getattr(active_orchestrator, 'status', '') == 'compiled_ok'
        return False

# Need to import logger if not already at top of commands.py
import logging
logger = logging.getLogger(__name__)
# --- PipelineEditorPane Commands ---

class AddStepCommand(Command):
    """Command to add a new step to the current pipeline."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        logger.info("AddStepCommand: Triggered.")
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)

        if not active_orchestrator:
            await state.notify('error', {'message': 'No active plate/pipeline to add a step to.', 'source': self.__class__.__name__})
            await show_error_dialog(title="Error", message="No active plate/pipeline to add a step to.", app_state=state)
            return

        # Ensure current_pipeline_definition is a list in TUIState
        if state.current_pipeline_definition is None:
            logger.warning("AddStepCommand: state.current_pipeline_definition was None. Initializing to empty list.")
            state.current_pipeline_definition = []
        
        # Attempt to find a default function
        default_func_pattern = None
        default_func_name = "Default Func" # Fallback name
        
        # Try to find a simple, known function from the registry (example placeholder)
        # This part is illustrative; a robust solution needs a well-defined default or specific UI to choose.
        if FUNC_REGISTRY:
            for backend_name, funcs_in_backend in FUNC_REGISTRY.items():
                if backend_name == "numpy" and "noop_numpy" in funcs_in_backend: # Example
                    default_func_pattern = funcs_in_backend["noop_numpy"]['pattern']
                    default_func_name = "NumPy No-Op"
                    break
                elif funcs_in_backend: # Fallback: take the first function from any backend
                    first_func_name = next(iter(funcs_in_backend))
                    default_func_pattern = funcs_in_backend[first_func_name]['pattern']
                    default_func_name = f"{first_func_name} (default)"
                    break
        
        if default_func_pattern is None:
            logger.warning("AddStepCommand: No suitable default function found in FUNC_REGISTRY. New step will have func=None.")
            # Using func=None is acceptable if DualStepFuncEditor can handle it for selection.

        new_step = FunctionStep(
            func=default_func_pattern, # Can be None if no default found
            name=f"New Step - {default_func_name}" 
            # Other parameters will use defaults from FunctionStep.__init__
        )
        # Ensure step_id is unique if FunctionStep doesn't auto-generate a sufficiently unique one
        # new_step.step_id = str(uuid.uuid4()) # FunctionStep already generates a UUID

        state.current_pipeline_definition.append(new_step)
        
        # Update the orchestrator's view of the pipeline definition directly
        # This assumes that TUIState.current_pipeline_definition is THE source of truth for the active pipeline
        active_orchestrator.pipeline_definition = state.current_pipeline_definition

        await state.notify('steps_updated', {'pipeline_definition': state.current_pipeline_definition, 'action': 'add', 'added_step_id': new_step.step_id})
        await state.notify('operation_status_changed', {'status': 'idle', 'message': f'Added step: {new_step.name}', 'source': self.__class__.__name__})
        logger.info(f"Added new step '{new_step.name}' (ID: {new_step.step_id}) to current_pipeline_definition and orchestrator.")

    def can_execute(self, state: "TUIState") -> bool:
        # Can execute if there's an active orchestrator, as a pipeline can always be started or added to.
        return getattr(state, 'active_orchestrator', None) is not None


class DeleteSelectedStepsCommand(Command):
    """Command to delete selected step(s) from the current pipeline."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        logger.info("DeleteSelectedStepsCommand: Triggered.")
        selected_steps_data = kwargs.get('steps_to_delete', []) # Expect list of step dicts/objects
        if not selected_steps_data:
            # Fallback to TUIState.selected_step if available (for single selection context)
            single_selected_step = getattr(state, 'selected_step', None)
            if single_selected_step:
                selected_steps_data = [single_selected_step]

        if not selected_steps_data:
            await message_dialog(title="Info", text="No steps selected for deletion.").run_async()
            return

        step_names = [s.get('name', s.get('id', 'Unknown Step')) for s in selected_steps_data]
        confirm_dialog = message_dialog(
            title="Confirm Delete Step(s)",
            text=f"Are you sure you want to delete selected step(s):\n{', '.join(step_names)}?",
            buttons=[("Yes", True), ("No", False)]
        )
        result = await confirm_dialog.run_async()
        if not result:
            logger.info("DeleteSelectedStepsCommand: Deletion cancelled by user.")
            return

        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)
        if not active_orchestrator or not hasattr(active_orchestrator, 'pipeline_definition') or active_orchestrator.pipeline_definition is None:
            await show_error_dialog(title="Error", message="No active pipeline definition to delete steps from.", app_state=state)
            return

        ids_to_delete = {step_data.get('id') for step_data in selected_steps_data if step_data.get('id')}
        if not ids_to_delete:
            logger.warning("DeleteSelectedStepsCommand: No valid step IDs found for deletion.")
            return

        original_pipeline = active_orchestrator.pipeline_definition
        new_pipeline = [step for step in original_pipeline if getattr(step, 'step_id', getattr(step, 'id', None)) not in ids_to_delete]

        if len(new_pipeline) < len(original_pipeline):
            active_orchestrator.pipeline_definition = new_pipeline
            await state.notify('steps_updated', {'action': 'delete', 'deleted_ids': list(ids_to_delete)})
            logger.info(f"Deleted steps with IDs: {ids_to_delete}")
        else:
            logger.warning(f"DeleteSelectedStepsCommand: No steps found matching IDs {ids_to_delete} in the active pipeline.")
            await show_error_dialog(title="Info", message="No matching steps found for deletion in the active pipeline.", app_state=state)


    def can_execute(self, state: "TUIState") -> bool:
        # Can execute if there's a selected step in TUIState or if steps_to_delete is provided
        return getattr(state, 'selected_step', None) is not None or \
               kwargs.get('steps_to_delete') # This needs to be passed if used


class ShowEditStepDialogCommand(Command):
    """Command to trigger editing of the selected step (shows DualStepFuncEditorPane)."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        logger.info("ShowEditStepDialogCommand: Triggered.")
        selected_step_data = getattr(state, 'selected_step', None)
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)

        if not active_orchestrator or not selected_step_data:
            await message_dialog(title="Error", text="No step selected or no active pipeline to edit a step from.").run_async()
            return

        # The PipelineEditorPane's handler for this command will set:
        # state.step_to_edit_config = actual_FunctionStep_instance
        # state.editing_step_config = True
        # Then TUIArchitecture._get_left_pane() will show the DualStepFuncEditorPane.
        # This command just signals the intent.
        await state.notify("edit_step_dialog_requested", {"step_data": selected_step_data})

    def can_execute(self, state: "TUIState") -> bool:
        return getattr(state, 'selected_step', None) is not None and \
               getattr(state, 'active_orchestrator', None) is not None


class LoadPipelineCommand(Command):
    """Command to load a pipeline definition for the active plate."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        logger.info("LoadPipelineCommand: Triggered.")
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)
        if not active_orchestrator:
            await message_dialog(title="Error", text="No active plate to load a pipeline into.").run_async()
            return

        file_path_str = await prompt_for_path_dialog(
            title="Load Pipeline",
            prompt_message="Enter path to .pipeline file:",
            app_state=state
        )

        if not file_path_str:
            logger.info("LoadPipelineCommand: Load operation cancelled by user.")
            await state.notify('info', {'message': "Load pipeline operation cancelled.", 'source': self.__class__.__name__})
            return

        file_path = Path(file_path_str)

        if not file_path.exists() or not file_path.is_file():
            await show_error_dialog("Load Pipeline Error", f"File not found or is not a file: {file_path}", app_state=state)
            return

        try:
            with open(file_path, "rb") as f:
                loaded_pipeline = pickle.load(f)

            if not isinstance(loaded_pipeline, list):
                await show_error_dialog("Load Pipeline Error", "Invalid pipeline file: content is not a list.", app_state=state)
                return

            valid_pipeline = True
            for item in loaded_pipeline:
                if not isinstance(item, AbstractStep):
                    await show_error_dialog("Load Pipeline Error", f"Invalid pipeline file: contains non-Step object: {type(item)}.", app_state=state)
                    valid_pipeline = False
                    break
            if not valid_pipeline:
                return

            active_orchestrator.pipeline_definition = loaded_pipeline

            await state.notify('steps_updated', {'action': 'load_pipeline'})
            await state.notify('operation_status_changed', {'message': f"Pipeline loaded from {file_path}", 'status': 'success', 'source': self.__class__.__name__})
            logger.info(f"Pipeline loaded from {file_path} into orchestrator.")

        except pickle.UnpicklingError as e:
            logger.error(f"Error unpickling pipeline from {file_path}: {e}", exc_info=True)
            await show_error_dialog("Load Pipeline Error", f"Error unpickling pipeline file: {e}", app_state=state)
        except Exception as e:
            logger.error(f"Failed to load pipeline from {file_path}: {e}", exc_info=True)
            await show_error_dialog("Load Pipeline Error", f"Failed to load pipeline: {e}", app_state=state)

    def can_execute(self, state: "TUIState") -> bool:
        return getattr(state, 'active_orchestrator', None) is not None


class SavePipelineCommand(Command):
    """Command to save the current pipeline definition for the active plate."""
    async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        logger.info("SavePipelineCommand: Triggered.")
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)
        if not active_orchestrator:
            await message_dialog(title="Error", text="No active pipeline to save.").run_async()
            return

        if not active_orchestrator.pipeline_definition:
            await message_dialog(title="Info", text="Pipeline is empty. Nothing to save.").run_async()
            return

        # This would involve a file dialog or use a default path.
        # For now, just notify. PipelineEditorPane would handle dialog/saving.
        # Placeholder for file dialog/saving logic:
        # save_path = await show_file_save_dialog_for_pipeline(default_name=...)
        # if save_path:
        #    await state.notify("save_pipeline_requested", {"orchestrator": active_orchestrator, "path": save_path})

        # Simplified for now, directly calling the orchestrator's save method (which needs to exist)
        # or using the logic from MenuBar._on_save_pipeline
        try:
            # Re-using logic similar to MenuBar._on_save_pipeline for now
            # This logic should ideally be in a shared service or the orchestrator itself.
            default_filename = "pipeline.json"
            if hasattr(active_orchestrator, 'config') and hasattr(active_orchestrator.config, 'pipeline_filename'):
                default_filename = active_orchestrator.config.pipeline_filename
            elif hasattr(active_orchestrator, 'DEFAULT_PIPELINE_FILENAME'):
                default_filename = active_orchestrator.DEFAULT_PIPELINE_FILENAME

            plate_path = Path(getattr(active_orchestrator, 'plate_path', '.'))
            save_path = plate_path / default_filename

            pipeline_data_to_save = []
            for step_obj in active_orchestrator.pipeline_definition:
                if isinstance(step_obj, AbstractStep): # Assuming AbstractStep is imported
                    pipeline_data_to_save.append(step_obj.to_dict())
                elif isinstance(step_obj, dict):
                    pipeline_data_to_save.append(step_obj)
                else:
                    logger.warning(f"SavePipelineCommand: Cannot serialize step of type {type(step_obj)}.")

            # Assuming context.filemanager for VFS operations if available, else direct write
            fm = getattr(context, 'filemanager', None)
            if fm:
                # Assuming Backend.DISK for now for pipeline definition files
                # This might need to be configurable or inferred.
                from openhcs.constants.constants import Backend
                await fm.write_json(save_path, pipeline_data_to_save, backend=Backend.DISK)
            else: # Fallback to direct write if no filemanager on context
                with open(save_path, 'w') as f:
                    json.dump(pipeline_data_to_save, f, indent=4)

            await message_dialog(title="Success", text=f"Pipeline saved to:\n{save_path}").run_async()
            logger.info(f"Pipeline saved to {save_path}")
            await state.notify("pipeline_saved", {"path": str(save_path)})

        except Exception as e:
            logger.error(f"SavePipelineCommand: Error saving pipeline: {e}", exc_info=True)
            await message_dialog(title="Error", text=f"Could not save pipeline:\n{e}").run_async()


    def can_execute(self, state: "TUIState") -> bool:
        active_orchestrator: Optional["PipelineOrchestrator"] = getattr(state, 'active_orchestrator', None)
        return active_orchestrator is not None and bool(active_orchestrator.pipeline_definition)

# Ensure AbstractStep, Path, json, logging, asyncio are imported if not already at the top
from openhcs.core.steps.abstract import AbstractStep # Already present
from pathlib import Path # Already present
import json # Already present
import logging # Already present or added by previous diffs
import asyncio # Already present or added by previous diffs

# Ensure PipelineOrchestrator is available for type hint if not already
if TYPE_CHECKING:
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator