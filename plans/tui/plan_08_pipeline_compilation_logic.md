# plan_08_pipeline_compilation_logic.md
## Component: Pipeline Compilation and Execution Logic

### Objective
Implement the core pipeline compilation and execution logic for the TUI, integrating with the existing OpenHCS pipeline system to provide proper validation, error handling, and status tracking.

### Plan

1. **Pipeline Compilation Interface**
   - Create a `PipelineCompilationManager` class to handle all compilation-related operations
   - Implement methods for pre-compilation, compilation, and validation
   - Integrate with `PipelineCompiler` from `ezstitcher/core/pipeline/compiler.py`
   - Ensure proper error handling and status updates
   - Validate all required parameters before compilation

2. **Pipeline Execution Interface**
   - Create a `PipelineExecutionManager` class to handle all execution-related operations
   - Implement methods for execution, monitoring, and result handling
   - Integrate with `PipelineExecutor` from `ezstitcher/core/pipeline/executor.py`
   - Support both sequential and parallel execution
   - Provide real-time status updates during execution

3. **Status Tracking System**
   - Implement a `OperationStatusTracker` class to centralize status tracking
   - Define clear status states (idle, pending, running, success, error)
   - Provide methods for updating and querying status
   - Integrate with TUI state management
   - Ensure thread-safety for concurrent operations

4. **Error Handling and Logging**
   - Implement comprehensive error handling for all pipeline operations
   - Create detailed error messages with context information
   - Integrate with TUI error display system
   - Provide error recovery mechanisms where possible
   - Implement proper logging for debugging

5. **Validation Framework**
   - Implement a validation framework for pipeline structure
   - Integrate with `FuncStepContractValidator` from `ezstitcher/core/pipeline/funcstep_contract_validator.py`
   - Validate all required parameters before operations
   - Provide clear validation error messages
   - Ensure validation follows OpenHCS principles

### Findings

#### Key Components for Pipeline Integration

1. **🔒 PipelineCompiler Integration (Clause 92)**
   - PipelineCompiler handles compilation through a sequence of planners and validators
   - Key method: `extract_path_overrides()`, `_prepare_step_paths()`, `_prepare_materialization_flags()`
   - Requires input_dir, well_id, and path_overrides as parameters
   - Returns step_plans dictionary with all necessary information for execution
   - Implementation: `await self._compile_pipeline(plate_id, input_dir, well_id)`

2. **🔒 PipelineExecutor Integration (Clause 297)**
   - PipelineExecutor handles execution of compiled pipelines
   - Key method: `execute()` which takes steps, context, and max_workers
   - Returns updated context with results
   - Supports both sequential and parallel execution
   - Implementation: `await self._execute_pipeline(plate_id, step_plans, context)`

3. **🔒 FuncStepContractValidator Integration (Clause 101)**
   - Validates memory type contracts for FunctionStep instances
   - Key method: `validate_pipeline()` which takes steps and pipeline_context
   - Returns dictionary mapping step UIDs to memory type dictionaries
   - Must run after path planner and materialization planner
   - Implementation: `memory_types = FuncStepContractValidator.validate_pipeline(steps, pipeline_context)`

4. **🔒 GPUMemoryTypeValidator Integration (Clause 293)**
   - Validates GPU memory types and assigns GPU device IDs
   - Key method: `validate_step_plans()` which takes step_plans
   - Returns dictionary mapping step UIDs to GPU assignment information
   - Fails loudly if no suitable GPU is available
   - Implementation: `gpu_assignments = GPUMemoryTypeValidator.validate_step_plans(step_plans)`

5. **🔒 Status Tracking Requirements (Clause 503)**
   - Status must be tracked centrally and reflect actual backend state
   - Status updates must be thread-safe and observable
   - Status must include detailed information for UI feedback
   - Implementation: `self.state.operation_status[operation] = status`

#### Implementation Considerations

1. **🔒 Error Handling (Clause 65)**
   - All errors must be caught and properly handled
   - Error messages must include context information
   - No fallback logic or silent failures
   - Implementation: `try/except` blocks with detailed error messages

2. **🔒 Validation (Clause 88)**
   - All parameters must be validated before use
   - No inferred capabilities or defaults
   - Validation must fail loudly with clear error messages
   - Implementation: Explicit validation methods for all parameters

3. **🔒 Thread Safety (Clause 295)**
   - All operations must be thread-safe
   - Status updates must be atomic
   - Concurrent operations must not interfere with each other
   - Implementation: Use of `asyncio` primitives for synchronization

4. **🔒 UI Integration (Clause 503)**
   - UI must reflect actual backend state
   - Status updates must be propagated to UI
   - Error messages must be displayed in UI
   - Implementation: Observer pattern for status updates

### Implementation Draft

```python
"""
Pipeline compilation and execution logic for OpenHCS TUI.

This module provides the core pipeline compilation and execution logic for the TUI,
integrating with the existing OpenHCS pipeline system to provide proper validation,
error handling, and status tracking.

🔒 Clause 65: No Fallback Logic
🔒 Clause 88: No Inferred Capabilities
🔒 Clause 92: Structural Validation First
🔒 Clause 101: Memory Type Declaration
🔒 Clause 293: GPU Pre-Declaration Enforcement
🔒 Clause 295: GPU Scheduling Affinity
🔒 Clause 297: Immutable Result Enforcement
🔒 Clause 503: Load Transfer
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from ezstitcher.core.pipeline.pipeline import PipelineCompiler, PipelineExecutor
from ezstitcher.core.pipeline.funcstep_contract_validator import FuncStepContractValidator
from ezstitcher.core.pipeline.gpu_memory_validator import GPUMemoryTypeValidator
from ezstitcher.core.context.processing_context import ProcessingContext
from ezstitcher.core.steps.abstract import AbstractStep

logger = logging.getLogger(__name__)


class PipelineCompilationManager:
    """
    Manages pipeline compilation operations for the TUI.

    This class provides methods for pre-compilation, compilation, and validation
    of pipelines, integrating with the existing OpenHCS pipeline system.

    Key responsibilities:
    1. Pre-compilation initialization
    2. Pipeline compilation
    3. Validation of pipeline structure
    4. Error handling and status updates
    """

    def __init__(self, state, context: ProcessingContext):
        """
        Initialize the pipeline compilation manager.

        Args:
            state: The TUI state manager
            context: The OpenHCS ProcessingContext
        """
        self.state = state
        self.context = context
        self.compiler = PipelineCompiler()

    async def initialize_orchestrator(self, plate_id: str) -> Tuple[bool, Optional[str]]:
        """
        Initialize the orchestrator for a plate.

        This is the pre-compilation step that prepares the orchestrator
        for compilation by loading plate information.

        Args:
            plate_id: The ID of the plate to initialize

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Update operation status
            self.state.operation_status['compile'] = 'pending'
            self.state.notify('operation_status_changed', {
                'operation': 'compile',
                'status': 'pending'
            })

            # Get plate information
            plate_info = self._get_plate_info(plate_id)
            if not plate_info:
                return False, f"Plate not found: {plate_id}"

            # Initialize orchestrator with plate information
            await asyncio.to_thread(
                self.context.initialize_plate,
                plate_id=plate_id,
                plate_dir=plate_info['path']
            )

            return True, None

        except Exception as e:
            logger.error(f"Error initializing orchestrator for plate {plate_id}: {e}")
            return False, str(e)

    async def compile_pipeline(self, plate_id: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Compile a pipeline for a plate.

        This method compiles a pipeline for the specified plate,
        validating the pipeline structure and preparing it for execution.

        Args:
            plate_id: The ID of the plate to compile

        Returns:
            Tuple of (success, error_message, step_plans)
        """
        try:
            # Update operation status
            self.state.operation_status['compile'] = 'running'
            self.state.notify('operation_status_changed', {
                'operation': 'compile',
                'status': 'running'
            })

            # Get plate information
            plate_info = self._get_plate_info(plate_id)
            if not plate_info:
                return False, f"Plate not found: {plate_id}", None

            # Get input directory
            input_dir = Path(plate_info['path'])

            # Get pipeline steps
            steps = await asyncio.to_thread(
                self.context.get_pipeline_steps,
                plate_id=plate_id
            )

            if not steps:
                return False, "No steps found in pipeline", None

            # Compile pipeline
            step_plans = await self._compile_pipeline(
                steps=steps,
                input_dir=input_dir,
                well_id=plate_id
            )

            if not step_plans:
                return False, "Pipeline compilation failed", None

            # Store step plans in context
            self.context.step_plans = step_plans

            # Set compilation flag
            self.state.is_compiled = True

            return True, None, step_plans

        except Exception as e:
            logger.error(f"Error compiling pipeline for plate {plate_id}: {e}")
            return False, str(e), None

        finally:
            # Update operation status
            if self.state.is_compiled:
                self.state.operation_status['compile'] = 'success'
                self.state.notify('operation_status_changed', {
                    'operation': 'compile',
                    'status': 'success'
                })
            else:
                self.state.operation_status['compile'] = 'error'
                self.state.notify('operation_status_changed', {
                    'operation': 'compile',
                    'status': 'error'
                })

    async def _compile_pipeline(
        self,
        steps: List[AbstractStep],
        input_dir: Path,
        well_id: str
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Compile a pipeline using the PipelineCompiler.

        This method handles the actual compilation process,
        calling the PipelineCompiler and handling the results.

        Args:
            steps: The pipeline steps
            input_dir: The input directory
            well_id: The well ID

        Returns:
            Dictionary mapping step UIDs to step plans, or None if compilation fails
        """
        try:
            # Extract path overrides from steps
            path_overrides = await asyncio.to_thread(
                self.compiler.extract_path_overrides,
                steps
            )

            # Initialize pipeline context
            pipeline_context = {
                "path_planner_done": False,
                "materialization_planner_done": False,
                "memory_contract_validator_done": False,
                "gpu_memory_validator_done": False,
                "attribute_stripper_done": False
            }

            # Prepare step paths
            step_paths = await asyncio.to_thread(
                self.compiler._prepare_step_paths,
                steps=steps,
                input_dir=input_dir,
                well_id=well_id,
                path_overrides=path_overrides,
                pipeline_context=pipeline_context
            )

            # Prepare materialization flags
            step_flags = await asyncio.to_thread(
                self.compiler._prepare_materialization_flags,
                steps=steps,
                well_id=well_id,
                pipeline_context=pipeline_context
            )

            # Create initial step plans
            step_plans = await asyncio.to_thread(
                self.compiler._create_initial_step_plans,
                steps=steps,
                step_paths=step_paths,
                step_flags=step_flags,
                step_contracts={},  # Empty dict for backward compatibility
                well_id=well_id
            )

            # Validate memory contracts
            step_memory_types = await asyncio.to_thread(
                self.compiler._validate_memory_contracts,
                steps=steps,
                pipeline_context=pipeline_context
            )

            # Add memory types to step plans
            await asyncio.to_thread(
                self.compiler._add_memory_types_to_step_plans,
                step_plans=step_plans,
                step_memory_types=step_memory_types
            )

            # Validate GPU memory types
            gpu_assignments = await asyncio.to_thread(
                self.compiler._validate_gpu_memory_types,
                step_plans=step_plans,
                pipeline_context=pipeline_context
            )

            # Add GPU device IDs to step plans
            await asyncio.to_thread(
                self.compiler._add_gpu_device_ids_to_step_plans,
                step_plans=step_plans,
                gpu_assignments=gpu_assignments
            )

            # Strip step attributes
            await asyncio.to_thread(
                self.compiler._strip_step_attributes,
                steps=steps,
                step_plans=step_plans,
                pipeline_context=pipeline_context
            )

            return step_plans

        except Exception as e:
            logger.error(f"Error in _compile_pipeline: {e}")
            raise

    def _get_plate_info(self, plate_id: str) -> Optional[Dict[str, Any]]:
        """
        Get plate information from the TUI state.

        Args:
            plate_id: The ID of the plate

        Returns:
            Dictionary with plate information, or None if not found
        """
        # Find plate in state.plates
        for plate in getattr(self.state, 'plates', []):
            if plate.get('id') == plate_id:
                return plate
        return None


class PipelineExecutionManager:
    """
    Manages pipeline execution operations for the TUI.

    This class provides methods for executing compiled pipelines,
    monitoring execution progress, and handling results.

    Key responsibilities:
    1. Pipeline execution
    2. Execution monitoring
    3. Result handling
    4. Error handling and status updates
    """

    def __init__(self, state, context: ProcessingContext):
        """
        Initialize the pipeline execution manager.

        Args:
            state: The TUI state manager
            context: The OpenHCS ProcessingContext
        """
        self.state = state
        self.context = context
        self.executor = PipelineExecutor()

    async def execute_pipeline(self, plate_id: str) -> Tuple[bool, Optional[str]]:
        """
        Execute a compiled pipeline for a plate.

        This method executes a compiled pipeline for the specified plate,
        monitoring execution progress and handling results.

        Args:
            plate_id: The ID of the plate to execute

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check if pipeline is compiled
            if not self.state.is_compiled:
                return False, "Pipeline not compiled. Please compile first."

            # Update operation status
            self.state.operation_status['run'] = 'running'
            self.state.notify('operation_status_changed', {
                'operation': 'run',
                'status': 'running'
            })

            # Get pipeline steps
            steps = await asyncio.to_thread(
                self.context.get_pipeline_steps,
                plate_id=plate_id
            )

            if not steps:
                return False, "No steps found in pipeline", None

            # Execute pipeline
            updated_context = await self._execute_pipeline(
                steps=steps,
                context=self.context
            )

            # Update context with results
            self.context = updated_context

            return True, None

        except Exception as e:
            logger.error(f"Error executing pipeline for plate {plate_id}: {e}")
            return False, str(e)

        finally:
            # Update operation status
            if self.state.operation_status['run'] == 'running':
                self.state.operation_status['run'] = 'success'
                self.state.notify('operation_status_changed', {
                    'operation': 'run',
                    'status': 'success'
                })
            else:
                self.state.operation_status['run'] = 'error'
                self.state.notify('operation_status_changed', {
                    'operation': 'run',
                    'status': 'error'
                })

    async def _execute_pipeline(
        self,
        steps: List[AbstractStep],
        context: ProcessingContext
    ) -> ProcessingContext:
        """
        Execute a pipeline using the PipelineExecutor.

        This method handles the actual execution process,
        calling the PipelineExecutor and handling the results.

        Args:
            steps: The pipeline steps
            context: The processing context

        Returns:
            Updated processing context with results
        """
        try:
            # Execute pipeline
            updated_context = await asyncio.to_thread(
                self.executor.execute,
                steps=steps,
                context=context,
                max_workers=1,  # Sequential execution for now
                visualizer=None  # No visualization for now
            )

            return updated_context

        except Exception as e:
            logger.error(f"Error in _execute_pipeline: {e}")
            raise


class PipelineSaveManager:
    """
    Manages pipeline saving operations for the TUI.

    This class provides methods for saving compiled pipelines,
    handling save results, and managing pipeline configurations.

    Key responsibilities:
    1. Pipeline saving
    2. Configuration management
    3. Error handling and status updates
    """

    def __init__(self, state, context: ProcessingContext):
        """
        Initialize the pipeline save manager.

        Args:
            state: The TUI state manager
            context: The OpenHCS ProcessingContext
        """
        self.state = state
        self.context = context

    async def save_pipeline(self, plate_id: str) -> Tuple[bool, Optional[str]]:
        """
        Save a compiled pipeline for a plate.

        This method saves a compiled pipeline for the specified plate,
        including step plans and configuration.

        Args:
            plate_id: The ID of the plate to save

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check if pipeline is compiled
            if not self.state.is_compiled:
                return False, "Pipeline not compiled. Please compile first."

            # Update operation status
            self.state.operation_status['save'] = 'running'
            self.state.notify('operation_status_changed', {
                'operation': 'save',
                'status': 'running'
            })

            # Get plate information
            plate_info = self._get_plate_info(plate_id)
            if not plate_info:
                return False, f"Plate not found: {plate_id}"

            # Save pipeline configuration
            success, error_message = await self._save_pipeline_config(
                plate_id=plate_id,
                plate_path=plate_info['path']
            )

            if not success:
                return False, error_message

            return True, None

        except Exception as e:
            logger.error(f"Error saving pipeline for plate {plate_id}: {e}")
            return False, str(e)

        finally:
            # Update operation status
            if self.state.operation_status['save'] == 'running':
                self.state.operation_status['save'] = 'success'
                self.state.notify('operation_status_changed', {
                    'operation': 'save',
                    'status': 'success'
                })
            else:
                self.state.operation_status['save'] = 'error'
                self.state.notify('operation_status_changed', {
                    'operation': 'save',
                    'status': 'error'
                })

    async def _save_pipeline_config(
        self,
        plate_id: str,
        plate_path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Save pipeline configuration to disk.

        This method saves the pipeline configuration to disk,
        including step plans and other settings.

        Args:
            plate_id: The ID of the plate
            plate_path: The path to the plate directory

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Get step plans from context
            step_plans = getattr(self.context, 'step_plans', {})
            if not step_plans:
                return False, "No step plans found in context"

            # Save step plans to disk
            await asyncio.to_thread(
                self.context.save_pipeline_config,
                plate_id=plate_id,
                config_path=Path(plate_path) / "pipeline_config.json"
            )

            return True, None

        except Exception as e:
            logger.error(f"Error saving pipeline configuration: {e}")
            return False, str(e)

    def _get_plate_info(self, plate_id: str) -> Optional[Dict[str, Any]]:
        """
        Get plate information from the TUI state.

        Args:
            plate_id: The ID of the plate

        Returns:
            Dictionary with plate information, or None if not found
        """
        # Find plate in state.plates
        for plate in getattr(self.state, 'plates', []):
            if plate.get('id') == plate_id:
                return plate
        return None


class OperationStatusTracker:
    """
    Centralized tracker for operation status in the TUI.

    This class provides methods for tracking and updating operation status,
    ensuring thread-safety and consistent status reporting.

    Key responsibilities:
    1. Status tracking
    2. Status updates
    3. Status queries
    4. Thread-safety
    """

    def __init__(self, state):
        """
        Initialize the operation status tracker.

        Args:
            state: The TUI state manager
        """
        self.state = state

        # Initialize operation status if not exists
        if not hasattr(self.state, 'operation_status'):
            self.state.operation_status = {
                'compile': 'idle',  # idle, pending, running, success, error
                'run': 'idle',
                'save': 'idle',
                'test': 'idle'
            }

    def get_status(self, operation: str) -> str:
        """
        Get the current status of an operation.

        Args:
            operation: The operation to get status for

        Returns:
            The current status of the operation
        """
        return self.state.operation_status.get(operation, 'idle')

    def update_status(self, operation: str, status: str) -> None:
        """
        Update the status of an operation.

        Args:
            operation: The operation to update
            status: The new status
        """
        # Validate operation
        if operation not in self.state.operation_status:
            logger.warning(f"Unknown operation: {operation}")
            return

        # Validate status
        valid_statuses = ['idle', 'pending', 'running', 'success', 'error']
        if status not in valid_statuses:
            logger.warning(f"Invalid status: {status}")
            return

        # Update status
        self.state.operation_status[operation] = status

        # Notify observers
        self.state.notify('operation_status_changed', {
            'operation': operation,
            'status': status
        })

    def is_running(self, operation: str) -> bool:
        """
        Check if an operation is running.

        Args:
            operation: The operation to check

        Returns:
            True if the operation is running, False otherwise
        """
        return self.get_status(operation) == 'running'

    def is_success(self, operation: str) -> bool:
        """
        Check if an operation completed successfully.

        Args:
            operation: The operation to check

        Returns:
            True if the operation completed successfully, False otherwise
        """
        return self.get_status(operation) == 'success'

    def is_error(self, operation: str) -> bool:
        """
        Check if an operation failed with an error.

        Args:
            operation: The operation to check

        Returns:
            True if the operation failed with an error, False otherwise
        """
        return self.get_status(operation) == 'error'
```
