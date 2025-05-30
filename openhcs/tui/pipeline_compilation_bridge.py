"""
Pipeline Compilation Enhancement Bridge.

This module enhances the existing pipeline compilation flow with:
- Pipeline validation before compilation
- Multi-plate compilation coordination  
- Enhanced error handling and user feedback
- Compilation state management

🔍 KEY DISCOVERY: Basic compilation integration already exists in CompilePlatesCommand.
This bridge enhances the existing flow rather than replacing it.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.abstract import AbstractStep
from openhcs.processing.func_registry import FUNC_REGISTRY

logger = logging.getLogger(__name__)


class PipelineValidationError(Exception):
    """Exception raised when pipeline validation fails."""
    pass


class PipelineCompilationBridge:
    """
    Enhanced pipeline compilation bridge.
    
    Provides validation, coordination, and enhanced error handling
    for the existing compilation flow.
    """
    
    def __init__(self, orchestrator_manager, tui_state):
        """
        Initialize the compilation bridge.
        
        Args:
            orchestrator_manager: OrchestratorManager instance
            tui_state: TUI state manager
        """
        self.orchestrator_manager = orchestrator_manager
        self.tui_state = tui_state
        self.compilation_executor = ThreadPoolExecutor(
            max_workers=2, 
            thread_name_prefix="pipeline-compilation-"
        )
        
        logger.info("PipelineCompilationBridge: Initialized")
    
    async def validate_and_compile_pipeline(self, plate_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Validate and compile pipelines for specified plates.
        
        Args:
            plate_ids: List of plate IDs to compile
            
        Returns:
            Dictionary mapping plate_id to compilation results
        """
        if not plate_ids:
            logger.warning("No plate IDs provided for compilation")
            return {}
        
        logger.info(f"Starting validation and compilation for plates: {plate_ids}")
        
        # Validate all pipelines first
        validation_results = await self._validate_pipelines(plate_ids)
        
        # Only compile plates that passed validation
        valid_plate_ids = [
            plate_id for plate_id, result in validation_results.items()
            if result.get('valid', False)
        ]
        
        if not valid_plate_ids:
            logger.error("No valid pipelines found for compilation")
            await self.tui_state.notify('compilation_error', {
                'error': 'No valid pipelines found',
                'validation_results': validation_results
            })
            return validation_results
        
        # Compile valid pipelines
        compilation_results = await self._compile_pipelines(valid_plate_ids)
        
        # Merge validation and compilation results
        final_results = {}
        for plate_id in plate_ids:
            final_results[plate_id] = {
                **validation_results.get(plate_id, {}),
                **compilation_results.get(plate_id, {})
            }
        
        return final_results
    
    async def _validate_pipelines(self, plate_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Validate pipelines for specified plates.
        
        Args:
            plate_ids: List of plate IDs to validate
            
        Returns:
            Dictionary mapping plate_id to validation results
        """
        validation_results = {}
        
        for plate_id in plate_ids:
            try:
                orchestrator = self.orchestrator_manager.get_orchestrator(plate_id)
                if not orchestrator:
                    validation_results[plate_id] = {
                        'valid': False,
                        'error': f'No orchestrator found for plate {plate_id}'
                    }
                    continue
                
                # Validate pipeline definition
                validation_result = await self._validate_single_pipeline(orchestrator, plate_id)
                validation_results[plate_id] = validation_result
                
            except Exception as e:
                logger.error(f"Error validating pipeline for plate {plate_id}: {e}", exc_info=True)
                validation_results[plate_id] = {
                    'valid': False,
                    'error': f'Validation error: {str(e)}'
                }
        
        return validation_results
    
    async def _validate_single_pipeline(self, orchestrator, plate_id: str) -> Dict[str, Any]:
        """
        Validate a single pipeline.
        
        Args:
            orchestrator: PipelineOrchestrator instance
            plate_id: Plate ID for error reporting
            
        Returns:
            Validation result dictionary
        """
        try:
            # Check if orchestrator is initialized
            if not orchestrator.is_initialized():
                return {
                    'valid': False,
                    'error': f'Orchestrator for plate {plate_id} is not initialized'
                }
            
            # Check if pipeline definition exists
            if not hasattr(orchestrator, 'pipeline_definition') or not orchestrator.pipeline_definition:
                return {
                    'valid': False,
                    'error': f'No pipeline definition found for plate {plate_id}'
                }
            
            pipeline = orchestrator.pipeline_definition
            
            # Validate pipeline structure
            if not isinstance(pipeline, list):
                return {
                    'valid': False,
                    'error': f'Pipeline definition must be a list, got {type(pipeline)}'
                }
            
            if len(pipeline) == 0:
                return {
                    'valid': False,
                    'error': 'Pipeline is empty - no steps defined'
                }
            
            # Validate individual steps
            step_validation_errors = []
            for i, step in enumerate(pipeline):
                step_error = await self._validate_step(step, i)
                if step_error:
                    step_validation_errors.append(step_error)
            
            if step_validation_errors:
                return {
                    'valid': False,
                    'error': 'Step validation failed',
                    'step_errors': step_validation_errors
                }
            
            return {
                'valid': True,
                'step_count': len(pipeline),
                'message': f'Pipeline validation passed for plate {plate_id}'
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline validation for plate {plate_id}: {e}", exc_info=True)
            return {
                'valid': False,
                'error': f'Validation exception: {str(e)}'
            }
    
    async def _validate_step(self, step: AbstractStep, step_index: int) -> Optional[str]:
        """
        Validate a single pipeline step.
        
        Args:
            step: Step to validate
            step_index: Index of step in pipeline
            
        Returns:
            Error message if validation fails, None if valid
        """
        try:
            # Check if step is an AbstractStep
            if not isinstance(step, AbstractStep):
                return f"Step {step_index}: Not an AbstractStep instance, got {type(step)}"
            
            # For FunctionStep, validate function pattern
            if isinstance(step, FunctionStep):
                func_error = await self._validate_function_pattern(step.func, step_index)
                if func_error:
                    return func_error
            
            # Validate step has required attributes
            if not hasattr(step, 'name') or not step.name:
                return f"Step {step_index}: Missing or empty name"
            
            if not hasattr(step, 'step_id') or not step.step_id:
                return f"Step {step_index}: Missing or empty step_id"
            
            return None
            
        except Exception as e:
            return f"Step {step_index}: Validation exception - {str(e)}"
    
    async def _validate_function_pattern(self, func_pattern: Any, step_index: int) -> Optional[str]:
        """
        Validate a function pattern.
        
        Args:
            func_pattern: Function pattern to validate
            step_index: Step index for error reporting
            
        Returns:
            Error message if validation fails, None if valid
        """
        if func_pattern is None:
            return f"Step {step_index}: Function pattern is None"
        
        try:
            # Validate based on pattern type
            if callable(func_pattern):
                # Single function - check if it's in registry
                return await self._validate_single_function(func_pattern, step_index)
            
            elif isinstance(func_pattern, tuple) and len(func_pattern) == 2:
                # (function, parameters) tuple
                func, params = func_pattern
                if not callable(func):
                    return f"Step {step_index}: First element of tuple must be callable"
                if not isinstance(params, dict):
                    return f"Step {step_index}: Second element of tuple must be a dict"
                return await self._validate_single_function(func, step_index)
            
            elif isinstance(func_pattern, list):
                # Sequential functions
                if len(func_pattern) == 0:
                    return f"Step {step_index}: Function list is empty"
                for i, func in enumerate(func_pattern):
                    if not callable(func):
                        return f"Step {step_index}: List element {i} is not callable"
                return None
            
            elif isinstance(func_pattern, dict):
                # Component-specific functions
                if len(func_pattern) == 0:
                    return f"Step {step_index}: Function dict is empty"
                for comp, func in func_pattern.items():
                    if not callable(func):
                        return f"Step {step_index}: Dict value for '{comp}' is not callable"
                return None
            
            else:
                return f"Step {step_index}: Invalid function pattern type: {type(func_pattern)}"
                
        except Exception as e:
            return f"Step {step_index}: Function pattern validation error - {str(e)}"
    
    async def _validate_single_function(self, func: callable, step_index: int) -> Optional[str]:
        """
        Validate a single function against the registry.
        
        Args:
            func: Function to validate
            step_index: Step index for error reporting
            
        Returns:
            Error message if validation fails, None if valid
        """
        # This is a basic validation - could be enhanced to check FUNC_REGISTRY
        if not hasattr(func, '__name__'):
            return f"Step {step_index}: Function has no __name__ attribute"
        
        # Could add more sophisticated validation here:
        # - Check if function is in FUNC_REGISTRY
        # - Validate function signature
        # - Check memory type compatibility
        
        return None
    
    async def _compile_pipelines(self, plate_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compile pipelines for validated plates.
        
        Args:
            plate_ids: List of validated plate IDs to compile
            
        Returns:
            Dictionary mapping plate_id to compilation results
        """
        compilation_results = {}
        
        # Compile each plate
        for plate_id in plate_ids:
            try:
                result = await self._compile_single_plate(plate_id)
                compilation_results[plate_id] = result
                
            except Exception as e:
                logger.error(f"Error compiling plate {plate_id}: {e}", exc_info=True)
                compilation_results[plate_id] = {
                    'success': False,
                    'error': f'Compilation error: {str(e)}'
                }
        
        return compilation_results
    
    async def _compile_single_plate(self, plate_id: str) -> Dict[str, Any]:
        """
        Compile a single plate's pipeline.
        
        Args:
            plate_id: Plate ID to compile
            
        Returns:
            Compilation result dictionary
        """
        orchestrator = self.orchestrator_manager.get_orchestrator(plate_id)
        if not orchestrator:
            return {
                'success': False,
                'error': f'No orchestrator found for plate {plate_id}'
            }
        
        try:
            # Notify compilation start
            await self.tui_state.notify('plate_operation_started', {
                'plate_id': plate_id,
                'operation': 'compile'
            })
            
            # Run compilation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            compiled_contexts = await loop.run_in_executor(
                self.compilation_executor,
                orchestrator.compile_pipelines,
                orchestrator.pipeline_definition
            )
            
            # Store compiled contexts
            orchestrator.last_compiled_contexts = compiled_contexts
            
            # Notify success
            await self.tui_state.notify('plate_status_changed', {
                'plate_id': plate_id,
                'status': 'compiled_ok'
            })
            
            await self.tui_state.notify('plate_operation_finished', {
                'plate_id': plate_id,
                'operation': 'compile'
            })
            
            logger.info(f"Successfully compiled pipeline for plate {plate_id}")
            
            return {
                'success': True,
                'compiled_contexts': len(compiled_contexts) if compiled_contexts else 0,
                'message': f'Pipeline compiled successfully for plate {plate_id}'
            }
            
        except Exception as e:
            # Notify error
            await self.tui_state.notify('plate_status_changed', {
                'plate_id': plate_id,
                'status': 'error_compile',
                'message': str(e)
            })
            
            await self.tui_state.notify('plate_operation_finished', {
                'plate_id': plate_id,
                'operation': 'compile'
            })
            
            logger.error(f"Compilation failed for plate {plate_id}: {e}", exc_info=True)
            
            return {
                'success': False,
                'error': str(e)
            }
    
    async def shutdown(self):
        """Shutdown the compilation bridge."""
        logger.info("PipelineCompilationBridge: Shutting down...")
        
        if hasattr(self, 'compilation_executor'):
            self.compilation_executor.shutdown(wait=True)
            
        logger.info("PipelineCompilationBridge: Shutdown complete")
