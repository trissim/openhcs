"""
Pipeline-related commands for the TUI.

This module contains commands for pipeline management in the TUI.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from prompt_toolkit.application import get_app
from prompt_toolkit.shortcuts import message_dialog

if TYPE_CHECKING:
    from openhcs.tui.tui_architecture import TUIState
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

# Legacy commands moved to archive/legacy_commands.py:
# - DeleteSelectedPlatesCommand
# - ShowEditPlateConfigDialogCommand


class InitializePlatesCommand:
    """
    Command to initialize plates.
    """
    def __init__(self, state: "TUIState" = None, context: "ProcessingContext" = None):
        """
        Initialize the command.

        Args:
            state: The TUIState instance (optional, can be provided during execute)
            context: The ProcessingContext instance (optional, can be provided during execute)
        """
        self.state = state
        self.context = context

    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """
        Execute the command to initialize plates.

        Args:
            state: The TUIState instance
            context: The ProcessingContext instance
            **kwargs: Additional arguments
                orchestrators_to_init: List of orchestrators to initialize
        """
        # Use provided state/context or the ones from initialization
        state = state or self.state
        context = context or self.context
        
        # Get the orchestrators to initialize from kwargs
        orchestrators_to_init = kwargs.get('orchestrators_to_init', [])
        
        if not orchestrators_to_init:
            await message_dialog(
                title="Error",
                text="No plates selected to initialize."
            ).run_async()
            return
            
        # Initialize each orchestrator
        for orchestrator in orchestrators_to_init:
            try:
                # Initialize the orchestrator
                await orchestrator.initialize()
                
                # Update the plate status
                await state.notify('plate_status_changed', {
                    'plate_id': orchestrator.plate_id,
                    'status': 'initialized'
                })
            except Exception as e:
                logger.error(f"Error initializing plate {orchestrator.plate_id}: {e}", exc_info=True)
                await state.notify('error', {
                    'source': 'InitializePlatesCommand',
                    'message': f"Failed to initialize plate {orchestrator.plate_id}",
                    'details': str(e)
                })
                
                # Update the plate status to error
                await state.notify('plate_status_changed', {
                    'plate_id': orchestrator.plate_id,
                    'status': 'error_init',
                    'error_details': str(e)
                })

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can be executed.

        Args:
            state: The TUIState instance

        Returns:
            True if the command can be executed, False otherwise
        """
        return state.selected_plate is not None


class CompilePlatesCommand:
    """
    Command to compile plates.
    """
    def __init__(self, state: "TUIState" = None, context: "ProcessingContext" = None):
        """
        Initialize the command.

        Args:
            state: The TUIState instance (optional, can be provided during execute)
            context: The ProcessingContext instance (optional, can be provided during execute)
        """
        self.state = state
        self.context = context

    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """
        Execute the command to compile plates.

        Args:
            state: The TUIState instance
            context: The ProcessingContext instance
            **kwargs: Additional arguments
                orchestrators_to_compile: List of orchestrators to compile
        """
        # Use provided state/context or the ones from initialization
        state = state or self.state
        context = context or self.context
        
        # Get the orchestrators to compile from kwargs
        orchestrators_to_compile = kwargs.get('orchestrators_to_compile', [])
        
        if not orchestrators_to_compile:
            await message_dialog(
                title="Error",
                text="No plates selected to compile."
            ).run_async()
            return
            
        # Compile each orchestrator
        for orchestrator in orchestrators_to_compile:
            try:
                # Compile the orchestrator
                compiled_pipeline_data = await orchestrator.compile()
                
                # Update the plate status
                await state.notify('plate_status_changed', {
                    'plate_id': orchestrator.plate_id,
                    'status': 'compiled_ok'
                })
                
                # Store the compiled pipeline data
                if not hasattr(state, 'compiled_contexts'):
                    state.compiled_contexts = {}
                state.compiled_contexts[orchestrator.plate_id] = compiled_pipeline_data
                
            except Exception as e:
                logger.error(f"Error compiling plate {orchestrator.plate_id}: {e}", exc_info=True)
                await state.notify('error', {
                    'source': 'CompilePlatesCommand',
                    'message': f"Failed to compile plate {orchestrator.plate_id}",
                    'details': str(e)
                })
                
                # Update the plate status to error
                await state.notify('plate_status_changed', {
                    'plate_id': orchestrator.plate_id,
                    'status': 'error_compile',
                    'error_details': str(e)
                })

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can be executed.

        Args:
            state: The TUIState instance

        Returns:
            True if the command can be executed, False otherwise
        """
        return state.selected_plate is not None and state.active_orchestrator is not None


class RunPlatesCommand:
    """
    Command to run plates.
    """
    def __init__(self, state: "TUIState" = None, context: "ProcessingContext" = None):
        """
        Initialize the command.

        Args:
            state: The TUIState instance (optional, can be provided during execute)
            context: The ProcessingContext instance (optional, can be provided during execute)
        """
        self.state = state
        self.context = context

    async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:
        """
        Execute the command to run plates.

        Args:
            state: The TUIState instance
            context: The ProcessingContext instance
            **kwargs: Additional arguments
                orchestrators_to_run: List of orchestrators to run
        """
        # Use provided state/context or the ones from initialization
        state = state or self.state
        context = context or self.context
        
        # Get the orchestrators to run from kwargs
        orchestrators_to_run = kwargs.get('orchestrators_to_run', [])
        
        if not orchestrators_to_run:
            await message_dialog(
                title="Error",
                text="No plates selected to run."
            ).run_async()
            return
            
        # Run each orchestrator
        for orchestrator in orchestrators_to_run:
            try:
                # Check if the plate has been compiled
                if not hasattr(state, 'compiled_contexts') or orchestrator.plate_id not in state.compiled_contexts:
                    # Try to compile it first
                    try:
                        compiled_pipeline_data = await orchestrator.compile()
                        if not hasattr(state, 'compiled_contexts'):
                            state.compiled_contexts = {}
                        state.compiled_contexts[orchestrator.plate_id] = compiled_pipeline_data
                    except Exception as e:
                        logger.error(f"Error compiling plate {orchestrator.plate_id} before run: {e}", exc_info=True)
                        await state.notify('error', {
                            'source': 'RunPlatesCommand',
                            'message': f"Failed to compile plate {orchestrator.plate_id} before run",
                            'details': str(e)
                        })
                        continue
                
                # Run the orchestrator
                state.is_running = True
                await state.notify('plate_status_changed', {
                    'plate_id': orchestrator.plate_id,
                    'status': 'running'
                })
                
                # Get the compiled context
                context = state.compiled_contexts.get(orchestrator.plate_id)
                if not context:
                    raise ValueError(f"No compiled context found for plate {orchestrator.plate_id}")
                
                # Run the pipeline
                await orchestrator.run(context)
                
                # Update the plate status
                await state.notify('plate_status_changed', {
                    'plate_id': orchestrator.plate_id,
                    'status': 'run_complete'
                })
                
            except Exception as e:
                logger.error(f"Error running plate {orchestrator.plate_id}: {e}", exc_info=True)
                await state.notify('error', {
                    'source': 'RunPlatesCommand',
                    'message': f"Failed to run plate {orchestrator.plate_id}",
                    'details': str(e)
                })
                
                # Update the plate status to error
                await state.notify('plate_status_changed', {
                    'plate_id': orchestrator.plate_id,
                    'status': 'error_run',
                    'error_details': str(e)
                })
            finally:
                state.is_running = False

    def can_execute(self, state: "TUIState") -> bool:
        """
        Determine if the command can be executed.

        Args:
            state: The TUIState instance

        Returns:
            True if the command can be executed, False otherwise
        """
        return (state.selected_plate is not None and 
                state.active_orchestrator is not None and 
                not state.is_running)
