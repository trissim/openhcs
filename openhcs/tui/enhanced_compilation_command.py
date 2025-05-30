"""
Enhanced Pipeline Compilation Command.

This module provides an enhanced compilation command that uses the
PipelineCompilationBridge for improved validation, error handling,
and multi-plate coordination.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from prompt_toolkit.shortcuts import message_dialog

if TYPE_CHECKING:
    from openhcs.tui.simple_launcher import SimpleTUIState
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


class EnhancedCompilePlatesCommand:
    """
    Enhanced command for compiling pipelines with validation and coordination.
    
    This command replaces or enhances the basic CompilePlatesCommand with:
    - Pipeline validation before compilation
    - Multi-plate compilation coordination
    - Enhanced error reporting and user feedback
    - Compilation state management
    """
    
    def __init__(self, orchestrator_manager=None):
        """
        Initialize the enhanced compilation command.
        
        Args:
            orchestrator_manager: OrchestratorManager instance (optional)
        """
        self.orchestrator_manager = orchestrator_manager
        self.compilation_bridge = None
    
    async def execute(self, state: "SimpleTUIState", context: "ProcessingContext", **kwargs: Any) -> None:
        """
        Execute the enhanced compilation command.
        
        Args:
            state: TUI state manager
            context: Processing context
            **kwargs: Additional arguments including:
                - plate_ids: List of plate IDs to compile (optional)
                - orchestrators_to_compile: List of orchestrators (optional)
        """
        try:
            # Validate parameters
            self._validate_parameters(state, context)
            
            # Initialize compilation bridge if needed
            await self._ensure_compilation_bridge(state)
            
            # Get plates to compile
            plate_ids = await self._get_plates_to_compile(state, kwargs)
            if not plate_ids:
                await self._show_no_plates_message()
                return
            
            # Show compilation start message
            await self._notify_compilation_start(state, plate_ids)
            
            # Validate and compile pipelines
            results = await self.compilation_bridge.validate_and_compile_pipeline(plate_ids)
            
            # Process and display results
            await self._process_compilation_results(state, results)
            
        except Exception as e:
            logger.error(f"Error in enhanced compilation command: {e}", exc_info=True)
            await self._handle_command_error(state, e)
    
    def _validate_parameters(self, state: "SimpleTUIState", context: "ProcessingContext") -> None:
        """Validate required parameters."""
        if state is None:
            raise ValueError("state parameter is required")
        if context is None:
            raise ValueError("context parameter is required")
    
    async def _ensure_compilation_bridge(self, state: "SimpleTUIState") -> None:
        """Ensure compilation bridge is initialized."""
        if self.compilation_bridge is None:
            # Get orchestrator manager from state or use provided one
            orchestrator_manager = self.orchestrator_manager
            if not orchestrator_manager:
                orchestrator_manager = getattr(state, 'orchestrator_manager', None)
            
            if not orchestrator_manager:
                raise RuntimeError("No orchestrator manager available for compilation")
            
            # Import and create compilation bridge
            from openhcs.tui.pipeline_compilation_bridge import PipelineCompilationBridge
            self.compilation_bridge = PipelineCompilationBridge(
                orchestrator_manager=orchestrator_manager,
                tui_state=state
            )
            
            logger.info("EnhancedCompilePlatesCommand: Compilation bridge initialized")
    
    async def _get_plates_to_compile(self, state: "SimpleTUIState", kwargs: Dict[str, Any]) -> List[str]:
        """
        Get the list of plate IDs to compile.
        
        Args:
            state: TUI state manager
            kwargs: Command arguments
            
        Returns:
            List of plate IDs to compile
        """
        # Check for explicit plate IDs
        plate_ids = kwargs.get('plate_ids', [])
        if plate_ids:
            return plate_ids
        
        # Check for orchestrators to compile
        orchestrators = kwargs.get('orchestrators_to_compile', [])
        if orchestrators:
            return [getattr(orch, 'plate_id', 'unknown') for orch in orchestrators]
        
        # Fall back to active orchestrator
        active_orchestrator = getattr(state, 'active_orchestrator', None)
        if active_orchestrator:
            plate_id = getattr(active_orchestrator, 'plate_id', None)
            if plate_id:
                return [plate_id]
        
        # Check selected plate
        selected_plate = getattr(state, 'selected_plate', None)
        if selected_plate and 'id' in selected_plate:
            return [selected_plate['id']]
        
        return []
    
    async def _show_no_plates_message(self) -> None:
        """Show message when no plates are available for compilation."""
        await message_dialog(
            title="No Plates to Compile",
            text="No plates selected or available for compilation.\n\n"
                 "Please select a plate and ensure it has a pipeline defined."
        ).run_async()
    
    async def _notify_compilation_start(self, state: "SimpleTUIState", plate_ids: List[str]) -> None:
        """Notify that compilation is starting."""
        plate_names = ", ".join(plate_ids)
        message = f"Starting compilation for plates: {plate_names}"
        
        await state.notify('operation_status_changed', {
            'status': 'running',
            'message': message,
            'source': self.__class__.__name__
        })
        
        logger.info(f"EnhancedCompilePlatesCommand: {message}")
    
    async def _process_compilation_results(self, state: "SimpleTUIState", results: Dict[str, Dict[str, Any]]) -> None:
        """
        Process and display compilation results.
        
        Args:
            state: TUI state manager
            results: Compilation results dictionary
        """
        successful_plates = []
        failed_plates = []
        validation_failures = []
        
        # Categorize results
        for plate_id, result in results.items():
            if result.get('valid', True) and result.get('success', False):
                successful_plates.append(plate_id)
            elif not result.get('valid', True):
                validation_failures.append((plate_id, result.get('error', 'Unknown validation error')))
            else:
                failed_plates.append((plate_id, result.get('error', 'Unknown compilation error')))
        
        # Update operation status
        await state.notify('operation_status_changed', {
            'status': 'idle',
            'message': f"Compilation complete: {len(successful_plates)} successful, {len(failed_plates + validation_failures)} failed",
            'source': self.__class__.__name__
        })
        
        # Show results dialog
        await self._show_results_dialog(successful_plates, failed_plates, validation_failures)
        
        # Update compilation state if any succeeded
        if successful_plates:
            state.is_compiled = True
            await state.notify('is_compiled_changed', True)
    
    async def _show_results_dialog(self, successful: List[str], failed: List[tuple], validation_failed: List[tuple]) -> None:
        """
        Show compilation results dialog.
        
        Args:
            successful: List of successful plate IDs
            failed: List of (plate_id, error) tuples for compilation failures
            validation_failed: List of (plate_id, error) tuples for validation failures
        """
        # Build results message
        message_parts = []
        
        if successful:
            message_parts.append(f"✅ Successfully compiled {len(successful)} plate(s):")
            for plate_id in successful:
                message_parts.append(f"  • {plate_id}")
        
        if validation_failed:
            message_parts.append(f"\n❌ Validation failed for {len(validation_failed)} plate(s):")
            for plate_id, error in validation_failed:
                message_parts.append(f"  • {plate_id}: {error}")
        
        if failed:
            message_parts.append(f"\n❌ Compilation failed for {len(failed)} plate(s):")
            for plate_id, error in failed:
                message_parts.append(f"  • {plate_id}: {error}")
        
        if not message_parts:
            message_parts.append("No plates were processed.")
        
        # Determine dialog title
        if successful and not (failed or validation_failed):
            title = "Compilation Successful"
        elif successful:
            title = "Compilation Partially Successful"
        else:
            title = "Compilation Failed"
        
        # Show dialog
        await message_dialog(
            title=title,
            text="\n".join(message_parts)
        ).run_async()
    
    async def _handle_command_error(self, state: "SimpleTUIState", error: Exception) -> None:
        """Handle command-level errors."""
        error_message = f"Compilation command error: {str(error)}"
        
        await state.notify('operation_status_changed', {
            'status': 'idle',
            'message': error_message,
            'source': self.__class__.__name__
        })
        
        await message_dialog(
            title="Compilation Error",
            text=f"An error occurred during compilation:\n\n{error_message}"
        ).run_async()
    
    def can_execute(self, state: "SimpleTUIState") -> bool:
        """
        Check if the command can be executed.
        
        Args:
            state: TUI state manager
            
        Returns:
            True if command can be executed
        """
        # Check if there's an active orchestrator or selected plate
        active_orchestrator = getattr(state, 'active_orchestrator', None)
        selected_plate = getattr(state, 'selected_plate', None)
        
        return active_orchestrator is not None or selected_plate is not None
    
    async def shutdown(self):
        """Shutdown the command and cleanup resources."""
        if self.compilation_bridge:
            await self.compilation_bridge.shutdown()
            self.compilation_bridge = None
        
        logger.info("EnhancedCompilePlatesCommand: Shutdown complete")
