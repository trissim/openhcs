"""
Command Service - Business Logic for Command Execution.

Handles the business logic for command execution, separating it from UI concerns.
Provides a clean API for command operations.

🔒 Clause 295: Component Boundaries
Clear separation between command business logic and UI interaction.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from openhcs.core.context.processing_context import ProcessingContext

logger = logging.getLogger(__name__)


class CommandService:
    """
    Service layer for command execution business logic.
    
    Handles:
    - Command validation
    - Business logic execution
    - Resource management
    - Error handling coordination
    """
    
    def __init__(self, state, context: ProcessingContext):
        self.state = state
        self.context = context
        
        # Shared executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="command-service-")
        
        # Command execution state
        self.active_operations = {}
        self.operation_lock = asyncio.Lock()
    
    async def initialize_plates(self, orchestrators: List[Any]) -> Dict[str, Any]:
        """
        Initialize plates using their orchestrators.
        
        Args:
            orchestrators: List of orchestrators to initialize
            
        Returns:
            Dictionary with initialization results
        """
        results = {
            'successful': [],
            'failed': [],
            'errors': []
        }
        
        for orchestrator in orchestrators:
            try:
                plate_id = getattr(orchestrator, 'plate_id', 'Unknown')
                
                # Mark operation as started
                await self._mark_operation_started(plate_id, 'initialize')
                
                # Perform initialization
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    orchestrator.initialize_plate
                )
                
                results['successful'].append(plate_id)
                await self._mark_operation_completed(plate_id, 'initialize', 'success')
                
            except Exception as e:
                error_msg = f"Failed to initialize plate {plate_id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                results['failed'].append(plate_id)
                results['errors'].append(error_msg)
                await self._mark_operation_completed(plate_id, 'initialize', 'error', str(e))
        
        return results
    
    async def compile_plates(self, orchestrators: List[Any]) -> Dict[str, Any]:
        """
        Compile pipelines for the given orchestrators.
        
        Args:
            orchestrators: List of orchestrators to compile
            
        Returns:
            Dictionary with compilation results
        """
        results = {
            'successful': [],
            'failed': [],
            'errors': [],
            'compiled_contexts': {}
        }
        
        for orchestrator in orchestrators:
            try:
                plate_id = getattr(orchestrator, 'plate_id', 'Unknown')
                
                # Validate orchestrator is ready for compilation
                if not self._is_orchestrator_ready_for_compilation(orchestrator):
                    error_msg = f"Orchestrator {plate_id} not ready for compilation"
                    results['failed'].append(plate_id)
                    results['errors'].append(error_msg)
                    continue
                
                # Mark operation as started
                await self._mark_operation_started(plate_id, 'compile')
                
                # Perform compilation
                loop = asyncio.get_event_loop()
                compiled_contexts = await loop.run_in_executor(
                    self.executor,
                    orchestrator.compile_pipelines,
                    orchestrator.pipeline_definition
                )
                
                # Store compiled contexts
                results['compiled_contexts'][plate_id] = compiled_contexts
                results['successful'].append(plate_id)
                await self._mark_operation_completed(plate_id, 'compile', 'success')
                
            except Exception as e:
                error_msg = f"Failed to compile plate {plate_id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                results['failed'].append(plate_id)
                results['errors'].append(error_msg)
                await self._mark_operation_completed(plate_id, 'compile', 'error', str(e))
        
        return results
    
    async def run_plates(self, orchestrators: List[Any], compiled_contexts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run compiled pipelines for the given orchestrators.
        
        Args:
            orchestrators: List of orchestrators to run
            compiled_contexts: Dictionary of compiled contexts by plate ID
            
        Returns:
            Dictionary with execution results
        """
        results = {
            'successful': [],
            'failed': [],
            'errors': []
        }
        
        for orchestrator in orchestrators:
            try:
                plate_id = getattr(orchestrator, 'plate_id', 'Unknown')
                
                # Validate orchestrator is ready for execution
                if not self._is_orchestrator_ready_for_execution(orchestrator, compiled_contexts):
                    error_msg = f"Orchestrator {plate_id} not ready for execution"
                    results['failed'].append(plate_id)
                    results['errors'].append(error_msg)
                    continue
                
                # Mark operation as started
                await self._mark_operation_started(plate_id, 'run')
                
                # Get compiled context
                context = compiled_contexts.get(plate_id)
                if not context:
                    raise ValueError(f"No compiled context found for plate {plate_id}")
                
                # Perform execution
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    orchestrator.execute_compiled_plate,
                    orchestrator.pipeline_definition,
                    context
                )
                
                results['successful'].append(plate_id)
                await self._mark_operation_completed(plate_id, 'run', 'success')
                
            except Exception as e:
                error_msg = f"Failed to run plate {plate_id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                results['failed'].append(plate_id)
                results['errors'].append(error_msg)
                await self._mark_operation_completed(plate_id, 'run', 'error', str(e))
        
        return results
    
    async def delete_plates(self, plate_ids: List[str]) -> Dict[str, Any]:
        """
        Delete plates by their IDs.
        
        Args:
            plate_ids: List of plate IDs to delete
            
        Returns:
            Dictionary with deletion results
        """
        results = {
            'successful': [],
            'failed': [],
            'errors': []
        }
        
        for plate_id in plate_ids:
            try:
                # Mark operation as started
                await self._mark_operation_started(plate_id, 'delete')
                
                # Notify about deletion
                await self.state.notify('plate_deletion_requested', {'plate_id': plate_id})
                
                results['successful'].append(plate_id)
                await self._mark_operation_completed(plate_id, 'delete', 'success')
                
            except Exception as e:
                error_msg = f"Failed to delete plate {plate_id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                results['failed'].append(plate_id)
                results['errors'].append(error_msg)
                await self._mark_operation_completed(plate_id, 'delete', 'error', str(e))
        
        return results
    
    def _is_orchestrator_ready_for_compilation(self, orchestrator: Any) -> bool:
        """Check if an orchestrator is ready for compilation."""
        return (
            hasattr(orchestrator, 'pipeline_definition') and
            orchestrator.pipeline_definition is not None and
            len(orchestrator.pipeline_definition) > 0
        )
    
    def _is_orchestrator_ready_for_execution(self, orchestrator: Any, compiled_contexts: Dict[str, Any]) -> bool:
        """Check if an orchestrator is ready for execution."""
        plate_id = getattr(orchestrator, 'plate_id', 'Unknown')
        return (
            self._is_orchestrator_ready_for_compilation(orchestrator) and
            plate_id in compiled_contexts and
            compiled_contexts[plate_id] is not None
        )
    
    async def _mark_operation_started(self, plate_id: str, operation: str):
        """Mark an operation as started."""
        async with self.operation_lock:
            self.active_operations[f"{plate_id}_{operation}"] = {
                'plate_id': plate_id,
                'operation': operation,
                'status': 'running',
                'start_time': asyncio.get_event_loop().time()
            }
        
        await self.state.notify('plate_operation_started', {
            'plate_id': plate_id,
            'operation': operation
        })
    
    async def _mark_operation_completed(self, plate_id: str, operation: str, status: str, error_message: str = None):
        """Mark an operation as completed."""
        async with self.operation_lock:
            operation_key = f"{plate_id}_{operation}"
            if operation_key in self.active_operations:
                self.active_operations[operation_key]['status'] = status
                self.active_operations[operation_key]['end_time'] = asyncio.get_event_loop().time()
                if error_message:
                    self.active_operations[operation_key]['error'] = error_message
        
        await self.state.notify('plate_operation_finished', {
            'plate_id': plate_id,
            'operation': operation,
            'status': status,
            'error_message': error_message
        })
    
    def get_active_operations(self) -> Dict[str, Any]:
        """Get currently active operations."""
        return dict(self.active_operations)
    
    async def cancel_operation(self, plate_id: str, operation: str) -> bool:
        """
        Cancel an active operation.
        
        Args:
            plate_id: ID of the plate
            operation: Type of operation to cancel
            
        Returns:
            True if operation was cancelled, False otherwise
        """
        async with self.operation_lock:
            operation_key = f"{plate_id}_{operation}"
            if operation_key in self.active_operations:
                self.active_operations[operation_key]['status'] = 'cancelled'
                await self.state.notify('plate_operation_cancelled', {
                    'plate_id': plate_id,
                    'operation': operation
                })
                return True
        return False
    
    async def shutdown(self):
        """Clean up the command service."""
        logger.info("CommandService: Shutting down...")
        
        # Cancel all active operations
        async with self.operation_lock:
            for operation_key in list(self.active_operations.keys()):
                operation = self.active_operations[operation_key]
                if operation['status'] == 'running':
                    await self.cancel_operation(operation['plate_id'], operation['operation'])
        
        # Shutdown executor
        if self.executor:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.executor.shutdown, True)
            self.executor = None
        
        logger.info("CommandService: Shutdown complete")
