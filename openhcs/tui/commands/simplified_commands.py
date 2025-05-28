"""
Simplified Command Implementations.

Clean, focused command implementations using the new architecture.
Each command has a single responsibility and clear dependencies.

🔒 Clause 295: Component Boundaries
Clear separation between command types and their responsibilities.
"""
import logging
from typing import Any, List, Optional, TYPE_CHECKING

from openhcs.tui.commands.base_command import (
    BaseCommand, ServiceCommand, DialogCommand, PlateCommand, PipelineCommand
)

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.core.context.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


class ShowHelpCommand(DialogCommand):
    """Command to show help information."""
    
    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """Execute the show help command."""
        state = self._get_state(state)
        
        help_text = (
            "OpenHCS TUI - Help\n\n"
            "Controls:\n"
            "- Use mouse or Tab/Shift-Tab to navigate.\n"
            "- Arrow keys for lists and menus.\n"
            "- Enter to activate buttons/menu items.\n"
            "- Ctrl-Q to Exit.\n\n"
            "Workflow:\n"
            "1. Add Plate(s) using Plate Manager [Add] button.\n"
            "2. Select a plate, then initialize it.\n"
            "3. Add steps to create a pipeline.\n"
            "4. Compile and run the pipeline.\n\n"
            "For more information, see the documentation."
        )
        
        await self._show_info_dialog("OpenHCS TUI Help", help_text)


class ShowSettingsCommand(DialogCommand):
    """Command to show global settings."""
    
    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """Execute the show settings command."""
        state = self._get_state(state)
        
        # For now, show a placeholder dialog
        await self._show_info_dialog(
            "Global Settings",
            "Settings dialog not yet implemented.\n\n"
            "Future settings will include:\n"
            "- Editor preferences\n"
            "- Default file paths\n"
            "- UI themes\n"
            "- Logging levels"
        )


class InitializePlatesCommand(ServiceCommand):
    """Command to initialize selected plates."""
    
    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """Execute the initialize plates command."""
        state = self._get_state(state)
        service = self._get_service()
        
        # Get orchestrators to initialize
        orchestrators = kwargs.get('orchestrators_to_initialize', [])
        if not orchestrators:
            await self._notify_info(state, 'initialize_plates', 'No plates selected to initialize.')
            return
        
        try:
            # Initialize plates using the service
            results = await service.initialize_plates(orchestrators)
            
            # Report results
            if results['successful']:
                success_msg = f"Successfully initialized {len(results['successful'])} plates."
                await self._notify_success(state, 'initialize_plates', success_msg)
            
            if results['failed']:
                error_msg = f"Failed to initialize {len(results['failed'])} plates."
                details = '\n'.join(results['errors'])
                await self._notify_error(state, error_msg, details)
                
        except Exception as e:
            logger.error(f"InitializePlatesCommand: Unexpected error: {e}", exc_info=True)
            await self._notify_error(state, "Failed to initialize plates", str(e))


class CompilePlatesCommand(ServiceCommand):
    """Command to compile selected plates."""
    
    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """Execute the compile plates command."""
        state = self._get_state(state)
        service = self._get_service()
        
        # Get orchestrators to compile
        orchestrators = kwargs.get('orchestrators_to_compile', [])
        if not orchestrators:
            await self._notify_info(state, 'compile_plates', 'No plates selected to compile.')
            return
        
        try:
            # Compile plates using the service
            results = await service.compile_plates(orchestrators)
            
            # Store compiled contexts in state
            if results['compiled_contexts']:
                if not hasattr(state, 'compiled_contexts'):
                    state.compiled_contexts = {}
                state.compiled_contexts.update(results['compiled_contexts'])
                state.is_compiled = True
            
            # Report results
            if results['successful']:
                success_msg = f"Successfully compiled {len(results['successful'])} plates."
                await self._notify_success(state, 'compile_plates', success_msg)
            
            if results['failed']:
                error_msg = f"Failed to compile {len(results['failed'])} plates."
                details = '\n'.join(results['errors'])
                await self._notify_error(state, error_msg, details)
                
        except Exception as e:
            logger.error(f"CompilePlatesCommand: Unexpected error: {e}", exc_info=True)
            await self._notify_error(state, "Failed to compile plates", str(e))


class RunPlatesCommand(ServiceCommand):
    """Command to run compiled plates."""
    
    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """Execute the run plates command."""
        state = self._get_state(state)
        service = self._get_service()
        
        # Get orchestrators to run
        orchestrators = kwargs.get('orchestrators_to_run', [])
        if not orchestrators:
            await self._notify_info(state, 'run_plates', 'No plates selected to run.')
            return
        
        # Check if plates are compiled
        compiled_contexts = getattr(state, 'compiled_contexts', {})
        if not compiled_contexts:
            await self._notify_error(state, "No compiled plates available", "Please compile plates first.")
            return
        
        try:
            # Set running state
            state.is_running = True
            
            # Run plates using the service
            results = await service.run_plates(orchestrators, compiled_contexts)
            
            # Report results
            if results['successful']:
                success_msg = f"Successfully ran {len(results['successful'])} plates."
                await self._notify_success(state, 'run_plates', success_msg)
            
            if results['failed']:
                error_msg = f"Failed to run {len(results['failed'])} plates."
                details = '\n'.join(results['errors'])
                await self._notify_error(state, error_msg, details)
                
        except Exception as e:
            logger.error(f"RunPlatesCommand: Unexpected error: {e}", exc_info=True)
            await self._notify_error(state, "Failed to run plates", str(e))
        finally:
            # Clear running state
            state.is_running = False


class DeletePlatesCommand(ServiceCommand):
    """Command to delete selected plates."""
    
    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """Execute the delete plates command."""
        state = self._get_state(state)
        service = self._get_service()
        
        # Get plate IDs to delete
        plate_ids = kwargs.get('plate_ids', [])
        if not plate_ids:
            await self._notify_info(state, 'delete_plates', 'No plates selected to delete.')
            return
        
        try:
            # Delete plates using the service
            results = await service.delete_plates(plate_ids)
            
            # Report results
            if results['successful']:
                success_msg = f"Successfully deleted {len(results['successful'])} plates."
                await self._notify_success(state, 'delete_plates', success_msg)
            
            if results['failed']:
                error_msg = f"Failed to delete {len(results['failed'])} plates."
                details = '\n'.join(results['errors'])
                await self._notify_error(state, error_msg, details)
                
        except Exception as e:
            logger.error(f"DeletePlatesCommand: Unexpected error: {e}", exc_info=True)
            await self._notify_error(state, "Failed to delete plates", str(e))


class AddStepCommand(PipelineCommand):
    """Command to add a step to the pipeline."""
    
    def can_execute(self, state: "TUIState") -> bool:
        """Check if command can be executed."""
        return self._validate_orchestrator_active(state)
    
    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """Execute the add step command."""
        state = self._get_state(state)
        
        if not self._validate_orchestrator_active(state):
            await self._handle_no_orchestrator_active(state)
            return
        
        # Notify that step addition was requested
        await state.notify('add_step_requested', {
            'orchestrator': state.active_orchestrator
        })


class RemoveStepCommand(PipelineCommand):
    """Command to remove a step from the pipeline."""
    
    def can_execute(self, state: "TUIState") -> bool:
        """Check if command can be executed."""
        return (
            self._validate_orchestrator_active(state) and
            getattr(state, 'selected_step', None) is not None
        )
    
    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """Execute the remove step command."""
        state = self._get_state(state)
        
        if not self._validate_orchestrator_active(state):
            await self._handle_no_orchestrator_active(state)
            return
        
        selected_step = getattr(state, 'selected_step', None)
        if not selected_step:
            await self._show_error_dialog(
                title="No Step Selected",
                message="Please select a step to remove."
            )
            return
        
        # Notify that step removal was requested
        await state.notify('remove_step_requested', {
            'orchestrator': state.active_orchestrator,
            'step': selected_step
        })


class ValidatePipelineCommand(PipelineCommand):
    """Command to validate the current pipeline."""
    
    def can_execute(self, state: "TUIState") -> bool:
        """Check if command can be executed."""
        orchestrator = self._get_active_orchestrator(state)
        return (
            orchestrator is not None and
            self._validate_pipeline_exists(orchestrator)
        )
    
    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """Execute the validate pipeline command."""
        state = self._get_state(state)
        
        orchestrator = self._get_active_orchestrator(state)
        if not orchestrator:
            await self._handle_no_orchestrator_active(state)
            return
        
        if not self._validate_pipeline_exists(orchestrator):
            await self._handle_no_pipeline(state)
            return
        
        try:
            # Perform basic validation
            pipeline_def = self._get_pipeline_definition(orchestrator)
            
            # Check for empty pipeline
            if not pipeline_def or len(pipeline_def) == 0:
                await self._show_error_dialog(
                    title="Empty Pipeline",
                    message="Pipeline has no steps. Please add steps first."
                )
                return
            
            # Basic validation passed
            await self._notify_success(
                state, 
                'validate_pipeline', 
                f"Pipeline validation passed. {len(pipeline_def)} steps found."
            )
            
        except Exception as e:
            logger.error(f"ValidatePipelineCommand: Validation error: {e}", exc_info=True)
            await self._notify_error(state, "Pipeline validation failed", str(e))
