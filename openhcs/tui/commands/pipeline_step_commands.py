"""
Pipeline Step Commands - Extracted from simplified_commands.py.

These commands handle pipeline step management operations.
They use notification-based architecture for loose coupling.
"""
import logging
from typing import Any, TYPE_CHECKING

from openhcs.tui.commands.base_command import PipelineCommand

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.core.context.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


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
