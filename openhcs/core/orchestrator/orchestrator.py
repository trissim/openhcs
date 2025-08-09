"""
Consolidated orchestrator module for OpenHCS.

This module provides a unified PipelineOrchestrator class that implements
a two-phase (compile-all-then-execute-all) pipeline execution model.

Doctrinal Clauses:
- Clause 12 — Absolute Clean Execution
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 293 — GPU Pre-Declaration Enforcement
- Clause 295 — GPU Scheduling Affinity
"""

import logging
import concurrent.futures
import multiprocessing
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Set

from openhcs.constants.constants import Backend, DEFAULT_WORKSPACE_DIR_SUFFIX, DEFAULT_IMAGE_EXTENSIONS, GroupBy, OrchestratorState
from openhcs.constants import Microscope
from openhcs.core.config import GlobalPipelineConfig, get_default_global_config
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.pipeline.compiler import PipelineCompiler
from openhcs.core.pipeline.step_attribute_stripper import StepAttributeStripper
from openhcs.core.steps.abstract import AbstractStep, get_step_id
from openhcs.io.exceptions import StorageWriteError
from openhcs.io.filemanager import FileManager
from openhcs.io.base import storage_registry
from openhcs.microscopes import create_microscope_handler
from openhcs.microscopes.microscope_base import MicroscopeHandler

# Optional napari import for visualization
try:
    from openhcs.runtime.napari_stream_visualizer import NapariStreamVisualizer
    NapariVisualizerType = NapariStreamVisualizer
except ImportError:
    # Create a placeholder type for type hints when napari is not available
    NapariStreamVisualizer = None
    NapariVisualizerType = Any  # Use Any for type hints when napari is not available


logger = logging.getLogger(__name__)


def _configure_worker_logging(log_file_base: str):
    """
    Configure logging and import hook for worker process.

    This function is called once per worker process when it starts.
    Each worker will get its own log file with a unique identifier.

    Args:
        log_file_base: Base path for worker log files
    """
    import os
    import logging
    import time

    # CRITICAL: Skip function registry initialization for fast worker startup
    # The environment variable is inherited from the subprocess runner
    # Note: We don't log this yet because logging isn't configured

    # CRITICAL: Install import hook for auto-discovered functions
    # Worker processes are fresh Python processes that need the import hook
    try:
        from openhcs.processing.func_registry import _install_import_hook
        _install_import_hook()
        # Note: We don't log this yet because logging isn't configured
    except Exception:
        # Can't log yet, but this is critical - the worker will fail later
        pass

    # Create unique worker identifier using PID and timestamp
    worker_pid = os.getpid()
    worker_timestamp = int(time.time() * 1000000)  # Microsecond precision for uniqueness
    worker_id = f"{worker_pid}_{worker_timestamp}"
    worker_log_file = f"{log_file_base}_worker_{worker_id}.log"

    # Configure root logger to capture ALL logs from worker process
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear any inherited handlers

    # Create file handler for worker logs
    file_handler = logging.FileHandler(worker_log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    # Ensure all OpenHCS module logs are captured
    logging.getLogger("openhcs").setLevel(logging.INFO)

    # Get worker logger
    worker_logger = logging.getLogger("openhcs.worker")
    worker_logger.info(f"🔥 WORKER: Process {worker_pid} (ID: {worker_id}) logging configured")
    worker_logger.info(f"🔥 WORKER: All logs writing to: {worker_log_file}")

    # Log import hook installation status
    worker_logger.info(f"🔥 WORKER: Import hook installed for auto-discovered functions")


# Global variable to store log file base for worker processes
_worker_log_file_base = None


def _ensure_step_ids_for_multiprocessing(
    frozen_context: ProcessingContext,
    pipeline_definition: List[AbstractStep],
    well_id: str
) -> None:
    """
    Helper function to update step IDs after multiprocessing pickle/unpickle.
    
    When contexts are pickled/unpickled for multiprocessing, step objects get
    new memory addresses, changing their IDs. This remaps the step_plans.
    """
    from openhcs.core.pipeline.compiler import PipelineCompiler
    try:
        logger.debug(f"🔥 MULTIPROCESSING: Updating step IDs for well {well_id}")
        PipelineCompiler.update_step_ids_for_multiprocessing(frozen_context, pipeline_definition)
        logger.debug(f"🔥 MULTIPROCESSING: Step IDs updated successfully for well {well_id}")
    except Exception as remap_error:
        error_msg = f"🔥 MULTIPROCESSING ERROR: Failed to remap step IDs for well {well_id}: {remap_error}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from remap_error


class PipelineOrchestrator:
    """
    Unified orchestrator for a two-phase pipeline execution model.

    The orchestrator first compiles the pipeline for all specified wells,
    creating frozen, immutable ProcessingContexts using `compile_plate_for_processing()`.
    Then, it executes the (now stateless) pipeline definition against these contexts,
    potentially in parallel, using `execute_compiled_plate()`.
    """

    def __init__(
        self,
        plate_path: Union[str, Path],
        workspace_path: Optional[Union[str, Path]] = None,
        *,
        global_config: Optional[GlobalPipelineConfig] = None,
        storage_registry: Optional[Any] = None, # Optional StorageRegistry instance
    ):
        # Lock removed - was orphaned code never used
        
        if global_config is None:
            self.global_config = get_default_global_config()
            logger.info("PipelineOrchestrator using default global configuration.")
        else:
            self.global_config = global_config

        if plate_path is None:
            # This case should ideally be prevented by TUI logic if plate_path is mandatory
            # for an orchestrator instance tied to a specific plate.
            # If workspace_path is also None, this will be caught later.
            pass
        elif isinstance(plate_path, str):
            plate_path = Path(plate_path)
        elif not isinstance(plate_path, Path):
            raise ValueError(f"Invalid plate_path type: {type(plate_path)}. Must be str or Path.")
        
        if plate_path:
            if not plate_path.is_absolute():
                raise ValueError(f"Plate path must be absolute: {plate_path}")
            if not plate_path.exists():
                raise FileNotFoundError(f"Plate path does not exist: {plate_path}")
            if not plate_path.is_dir():
                raise NotADirectoryError(f"Plate path is not a directory: {plate_path}")

        # Initialize _plate_path_frozen first to allow plate_path to be set during initialization
        object.__setattr__(self, '_plate_path_frozen', False)

        self.plate_path = plate_path
        self.workspace_path = workspace_path

        if self.plate_path is None and self.workspace_path is None:
            raise ValueError("Either plate_path or workspace_path must be provided for PipelineOrchestrator.")

        # Freeze plate_path immediately after setting it to prove immutability
        object.__setattr__(self, '_plate_path_frozen', True)
        logger.info(f"🔒 PLATE_PATH FROZEN: {self.plate_path} is now immutable")

        if storage_registry:
            self.registry = storage_registry
            logger.info("PipelineOrchestrator using provided StorageRegistry instance.")
        else:
            from openhcs.io.base import storage_registry as global_registry
            # Create a copy of the global registry to avoid modifying shared state
            self.registry = global_registry.copy()
            logger.info("PipelineOrchestrator created its own StorageRegistry instance (copy of global).")

        # Override zarr backend with orchestrator's config
        from openhcs.io.zarr import ZarrStorageBackend
        from openhcs.constants.constants import Backend

        zarr_backend_with_config = ZarrStorageBackend(self.global_config.zarr)
        self.registry[Backend.ZARR.value] = zarr_backend_with_config
        logger.info(f"Orchestrator zarr backend configured with {self.global_config.zarr.compressor.value} compression")

        # Orchestrator always creates its own FileManager, using the determined registry
        self.filemanager = FileManager(self.registry)
        self.input_dir: Optional[Path] = None
        self.microscope_handler: Optional[MicroscopeHandler] = None
        self.default_pipeline_definition: Optional[List[AbstractStep]] = None
        self._initialized: bool = False
        self._state: OrchestratorState = OrchestratorState.CREATED

        # Component keys cache for fast access
        self._component_keys_cache: Dict[GroupBy, List[str]] = {}

        # Metadata cache for component key→name mappings
        self._metadata_cache: Dict[GroupBy, Dict[str, Optional[str]]] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute, preventing modification of plate_path after it's frozen.

        This proves that plate_path is truly immutable after initialization.
        """
        if name == 'plate_path' and getattr(self, '_plate_path_frozen', False):
            import traceback
            stack_trace = ''.join(traceback.format_stack())
            error_msg = (
                f"🚫 IMMUTABLE PLATE_PATH VIOLATION: Cannot modify plate_path after freezing!\n"
                f"Current value: {getattr(self, 'plate_path', 'UNSET')}\n"
                f"Attempted new value: {value}\n"
                f"Stack trace:\n{stack_trace}"
            )
            logger.error(error_msg)
            raise AttributeError(error_msg)
        super().__setattr__(name, value)

    @property
    def state(self) -> OrchestratorState:
        """Get the current orchestrator state."""
        return self._state

    def initialize_microscope_handler(self):
        """Initializes the microscope handler."""
        if self.microscope_handler is not None:
            logger.debug("Microscope handler already initialized.")
            return
#        if self.input_dir is None:
#            raise RuntimeError("Workspace (and input_dir) must be initialized before microscope handler.")

        logger.info(f"Initializing microscope handler using input directory: {self.input_dir}...")
        try:
            # Use configured microscope type or auto-detect
            microscope_type = self.global_config.microscope.value if self.global_config.microscope != Microscope.AUTO else 'auto'
            self.microscope_handler = create_microscope_handler(
                plate_folder=str(self.plate_path),
                filemanager=self.filemanager,
                microscope_type=microscope_type,
            )
            logger.info(f"Initialized microscope handler: {type(self.microscope_handler).__name__}")
        except Exception as e:
            error_msg = f"Failed to create microscope handler: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def initialize(self, workspace_path: Optional[Union[str, Path]] = None) -> 'PipelineOrchestrator':
        """
        Initializes all required components for the orchestrator.
        Must be called before other processing methods.
        Returns self for chaining.
        """
        if self._initialized:
            logger.info("Orchestrator already initialized.")
            return self

        try:
            self.initialize_microscope_handler()

            # Delegate workspace initialization to microscope handler
            logger.info("Initializing workspace with microscope handler...")
            actual_image_dir = self.microscope_handler.initialize_workspace(
                self.plate_path, workspace_path, self.filemanager
            )

            # Use the actual image directory returned by the microscope handler
            self.input_dir = Path(actual_image_dir)
            logger.info(f"Set input directory to: {self.input_dir}")

            # Set workspace_path based on what the handler returned
            if actual_image_dir != self.plate_path:
                # Handler created a workspace
                self.workspace_path = Path(actual_image_dir).parent if Path(actual_image_dir).name != "workspace" else Path(actual_image_dir)
            else:
                # Handler used plate directly (like OpenHCS)
                self.workspace_path = None

            # Mark as initialized BEFORE caching to avoid chicken-and-egg problem
            self._initialized = True
            self._state = OrchestratorState.READY

            # Auto-cache component keys and metadata for instant access
            logger.info("Caching component keys and metadata...")
            self.cache_component_keys()
            self.cache_metadata()

            logger.info("PipelineOrchestrator fully initialized with cached component keys and metadata.")
            return self
        except Exception as e:
            self._state = OrchestratorState.INIT_FAILED
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    def is_initialized(self) -> bool:
        return self._initialized

    def create_context(self, well_id: str) -> ProcessingContext:
        """Creates a ProcessingContext for a given well."""
        if not self.is_initialized():
            raise RuntimeError("Orchestrator must be initialized before calling create_context().")
        if not well_id:
            raise ValueError("Well identifier must be provided.")
        if self.input_dir is None:
             raise RuntimeError("Orchestrator input_dir is not set; initialize orchestrator first.")

        context = ProcessingContext(
            global_config=self.global_config,
            well_id=well_id,
            filemanager=self.filemanager
        )
        # Orchestrator reference removed - was orphaned and unpickleable
        context.microscope_handler = self.microscope_handler
        context.input_dir = self.input_dir
        context.workspace_path = self.workspace_path
        context.plate_path = self.plate_path  # Add plate_path for path planner
        # Pass metadata cache for OpenHCS metadata creation
        context.metadata_cache = dict(self._metadata_cache)  # Copy to avoid pickling issues
        return context

    def compile_pipelines(
        self,
        pipeline_definition: List[AbstractStep],
        well_filter: Optional[List[str]] = None,
        enable_visualizer_override: bool = False
    ) -> Dict[str, ProcessingContext]:
        """
        Compile-all phase: Prepares frozen ProcessingContexts for each well.

        This method iterates through the specified wells, creates a ProcessingContext
        for each, and invokes the various phases of the PipelineCompiler to populate
        the context's step_plans. After all compilation phases for a well are complete,
        its context is frozen. Finally, attributes are stripped from the pipeline_definition,
        making the step objects stateless for the execution phase.

        Args:
            pipeline_definition: The list of AbstractStep objects defining the pipeline.
            well_filter: Optional list of well IDs to process. If None, processes all found wells.
            enable_visualizer_override: If True, all steps in all compiled contexts
                                        will have their 'visualize' flag set to True.

        Returns:
            A dictionary mapping well IDs to their compiled and frozen ProcessingContexts.
            The input `pipeline_definition` list (of step objects) is modified in-place
            to become stateless.
        """
        if not self.is_initialized():
            raise RuntimeError("PipelineOrchestrator must be explicitly initialized before calling compile_pipelines().")

        if not pipeline_definition:
            raise ValueError("A valid pipeline definition (List[AbstractStep]) must be provided.")

        try:
            compiled_contexts: Dict[str, ProcessingContext] = {}
            wells_to_process = self.get_component_keys(GroupBy.WELL, well_filter)

            if not wells_to_process:
                logger.warning("No wells found to process based on filter.")
                return {}

            logger.info(f"Starting compilation for wells: {', '.join(wells_to_process)}")

            # Determine responsible well for metadata creation (lexicographically first)
            responsible_well = sorted(wells_to_process)[0] if wells_to_process else None
            logger.debug(f"Designated responsible well for metadata creation: {responsible_well}")

            for well_id in wells_to_process:
                logger.debug(f"Compiling for well: {well_id}")
                context = self.create_context(well_id)

                # Determine if this well is responsible for metadata creation
                is_responsible = (well_id == responsible_well)
                logger.debug(f"Well {well_id} metadata responsibility: {is_responsible}")

                PipelineCompiler.initialize_step_plans_for_context(context, pipeline_definition, metadata_writer=is_responsible, plate_path=self.plate_path)
                PipelineCompiler.declare_zarr_stores_for_context(context, pipeline_definition, self)
                PipelineCompiler.plan_materialization_flags_for_context(context, pipeline_definition, self)
                PipelineCompiler.validate_memory_contracts_for_context(context, pipeline_definition, self)
                PipelineCompiler.assign_gpu_resources_for_context(context)

                if enable_visualizer_override:
                    PipelineCompiler.apply_global_visualizer_override_for_context(context, True)

                context.freeze()
                compiled_contexts[well_id] = context
                logger.debug(f"Compilation finished for well: {well_id}")

            # After processing all wells, strip attributes and finalize
            logger.info("Stripping attributes from pipeline definition steps.")
            StepAttributeStripper.strip_step_attributes(pipeline_definition, {})

            self._state = OrchestratorState.COMPILED
            logger.info(f"Plate compilation finished for {len(compiled_contexts)} wells.")
            return compiled_contexts
        except Exception as e:
            self._state = OrchestratorState.COMPILE_FAILED
            logger.error(f"Failed to compile pipelines: {e}")
            raise

    def _execute_single_well(
        self,
        pipeline_definition: List[AbstractStep],
        frozen_context: ProcessingContext,
        visualizer: Optional[NapariVisualizerType]
    ) -> Dict[str, Any]:
        """Executes the pipeline for a single well using its frozen context."""
        well_id = frozen_context.well_id
        logger.info(f"🔥 SINGLE_WELL: Starting execution for well {well_id}")

        # NUCLEAR VALIDATION
        if not frozen_context.is_frozen():
            error_msg = f"🔥 SINGLE_WELL ERROR: Context for well {well_id} is not frozen before execution"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not pipeline_definition:
            error_msg = f"🔥 SINGLE_WELL ERROR: Empty pipeline_definition for well {well_id}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # MULTIPROCESSING FIX: Update step IDs after pickle/unpickle
        _ensure_step_ids_for_multiprocessing(frozen_context, pipeline_definition, well_id)

        logger.info(f"🔥 SINGLE_WELL: Processing {len(pipeline_definition)} steps for well {well_id}")

        for step_index, step in enumerate(pipeline_definition):
            # Generate step_id from object reference (elegant stateless approach)
            step_id = get_step_id(step)
            step_name = getattr(step, 'name', 'N/A') if hasattr(step, 'name') else 'N/A'

            logger.info(f"🔥 SINGLE_WELL: Executing step {step_index+1}/{len(pipeline_definition)} - {step_id} ({step_name}) for well {well_id}")

            if not hasattr(step, 'process'):
                error_msg = f"🔥 SINGLE_WELL ERROR: Step {step_id} missing process method for well {well_id}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            step.process(frozen_context)
            logger.info(f"🔥 SINGLE_WELL: Step {step_index+1}/{len(pipeline_definition)} - {step_id} completed for well {well_id}")

    #        except Exception as step_error:
    #            import traceback
    #            full_traceback = traceback.format_exc()
    #            error_msg = f"🔥 SINGLE_WELL ERROR: Step {step_index+1} ({step_id}) failed for well {well_id}: {step_error}"
    #            logger.error(error_msg, exc_info=True)
    #            logger.error(f"🔥 SINGLE_WELL TRACEBACK for well {well_id}, step {step_index+1} ({step_id}):\n{full_traceback}")
    #            raise RuntimeError(error_msg) from step_error

            if visualizer:
                step_plan = frozen_context.step_plans[step_id]
                if step_plan['visualize']:
                    output_dir = step_plan['output_dir']
                    write_backend = step_plan['write_backend']
                    if output_dir:
                        logger.debug(f"Visualizing output for step {step_id} from path {output_dir} (backend: {write_backend}) for well {well_id}")
                        visualizer.visualize_path(
                            step_id=step_id,
                            path=str(output_dir),
                            backend=write_backend,
                            well_id=well_id
                        )
                    else:
                        logger.warning(f"Step {step_id} in well {well_id} flagged for visualization but 'output_dir' is missing in its plan.")
        
        logger.info(f"🔥 SINGLE_WELL: Pipeline execution completed successfully for well {well_id}")
        return {"status": "success", "well_id": well_id}

    def execute_compiled_plate(
        self,
        pipeline_definition: List[AbstractStep],
        compiled_contexts: Dict[str, ProcessingContext],
        max_workers: Optional[int] = None,
        visualizer: Optional[NapariVisualizerType] = None,
        log_file_base: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute-all phase: Runs the stateless pipeline against compiled contexts.

        Args:
            pipeline_definition: The stateless list of AbstractStep objects.
            compiled_contexts: Dict of well_id to its compiled, frozen ProcessingContext.
                               Obtained from `compile_plate_for_processing`.
            max_workers: Maximum number of worker threads for parallel execution.
            visualizer: Optional instance of NapariStreamVisualizer for real-time visualization
                        (requires napari to be installed; must be initialized with orchestrator's filemanager by the caller).
            log_file_base: Base path for worker process log files (without extension).
                          Each worker will create its own log file: {log_file_base}_worker_{pid}.log

        Returns:
            A dictionary mapping well IDs to their execution status (success/error and details).
        """
        if not self.is_initialized():
             raise RuntimeError("Orchestrator must be initialized before executing.")
        if not pipeline_definition:
            raise ValueError("A valid (stateless) pipeline definition must be provided.")
        if not compiled_contexts:
            logger.warning("No compiled contexts provided for execution.")
            return {}
        
        actual_max_workers = max_workers if max_workers is not None else self.global_config.num_workers
        if actual_max_workers <= 0: # Ensure positive number of workers
            actual_max_workers = 1

        self._state = OrchestratorState.EXECUTING
        logger.info(f"Starting execution for {len(compiled_contexts)} wells with max_workers={actual_max_workers}.")

        # 🔍 VRAM TRACKING: Log initial memory state
        try:
            from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
            log_gpu_memory_usage("plate execution start")
        except Exception:
            pass

        try:
            execution_results: Dict[str, Dict[str, Any]] = {}

            # CUDA COMPATIBILITY: Set spawn method for multiprocessing to support CUDA
            try:
                # Check if spawn method is available and set it if not already set
                current_method = multiprocessing.get_start_method(allow_none=True)
                if current_method != 'spawn':
                    logger.info(f"🔥 CUDA: Setting multiprocessing start method from '{current_method}' to 'spawn' for CUDA compatibility")
                    multiprocessing.set_start_method('spawn', force=True)
                else:
                    logger.debug("🔥 CUDA: Multiprocessing start method already set to 'spawn'")
            except RuntimeError as e:
                # Start method may already be set, which is fine
                logger.debug(f"🔥 CUDA: Start method already configured: {e}")

            # Choose executor type based on global config for debugging support
            executor_type = "ThreadPoolExecutor" if self.global_config.use_threading else "ProcessPoolExecutor"
            logger.info(f"🔥 ORCHESTRATOR: Creating {executor_type} with {actual_max_workers} workers")

            # DEATH DETECTION: Mark executor creation
            logger.info(f"🔥 DEATH_MARKER: BEFORE_{executor_type.upper()}_CREATION")

            # Choose appropriate executor class and configure worker logging
            if self.global_config.use_threading:
                logger.info("🔥 DEBUG MODE: Using ThreadPoolExecutor for easier debugging")
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=actual_max_workers)
            else:
                logger.info("🔥 PRODUCTION MODE: Using ProcessPoolExecutor for true parallelism")
                if log_file_base:
                    logger.info(f"🔥 WORKER LOGGING: Configuring worker processes with log base: {log_file_base}")
                    executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=actual_max_workers,
                        initializer=_configure_worker_logging,
                        initargs=(log_file_base,)
                    )
                else:
                    logger.info("🔥 WORKER LOGGING: No log base provided, workers will inherit logging")
                    executor = concurrent.futures.ProcessPoolExecutor(max_workers=actual_max_workers)

            logger.info(f"🔥 DEATH_MARKER: ENTERING_{executor_type.upper()}_CONTEXT")
            with executor:
                logger.info(f"🔥 DEATH_MARKER: {executor_type.upper()}_CREATED_SUCCESSFULLY")
                logger.info(f"🔥 ORCHESTRATOR: {executor_type} created, submitting {len(compiled_contexts)} tasks")

                # NUCLEAR ERROR TRACING: Create snapshot of compiled_contexts to prevent iteration issues
                contexts_snapshot = dict(compiled_contexts.items())
                logger.info(f"🔥 ORCHESTRATOR: Created contexts snapshot with {len(contexts_snapshot)} items")

                logger.info("🔥 DEATH_MARKER: BEFORE_TASK_SUBMISSION_LOOP")
                future_to_well_id = {}
                for well_id, context in contexts_snapshot.items():
                    try:
                        logger.info(f"🔥 DEATH_MARKER: SUBMITTING_TASK_FOR_WELL_{well_id}")
                        logger.info(f"🔥 ORCHESTRATOR: Submitting task for well {well_id}")
                        future = executor.submit(self._execute_single_well, pipeline_definition, context, visualizer)
                        future_to_well_id[future] = well_id
                        logger.info(f"🔥 ORCHESTRATOR: Task submitted for well {well_id}")
                        logger.info(f"🔥 DEATH_MARKER: TASK_SUBMITTED_FOR_WELL_{well_id}")
                    except Exception as submit_error:
                        error_msg = f"🔥 ORCHESTRATOR ERROR: Failed to submit task for well {well_id}: {submit_error}"
                        logger.error(error_msg, exc_info=True)
                        # FAIL-FAST: Re-raise task submission errors immediately
                        raise

                logger.info("🔥 DEATH_MARKER: TASK_SUBMISSION_LOOP_COMPLETED")

                logger.info(f"🔥 ORCHESTRATOR: All {len(future_to_well_id)} tasks submitted, waiting for completion")
                logger.info("🔥 DEATH_MARKER: BEFORE_COMPLETION_LOOP")

                completed_count = 0
                logger.info("🔥 DEATH_MARKER: ENTERING_AS_COMPLETED_LOOP")
                for future in concurrent.futures.as_completed(future_to_well_id):
                    well_id = future_to_well_id[future]
                    completed_count += 1
                    logger.info(f"🔥 DEATH_MARKER: PROCESSING_COMPLETED_TASK_{completed_count}_WELL_{well_id}")
                    logger.info(f"🔥 ORCHESTRATOR: Task {completed_count}/{len(future_to_well_id)} completed for well {well_id}")

                    try:
                        logger.info(f"🔥 DEATH_MARKER: CALLING_FUTURE_RESULT_FOR_WELL_{well_id}")
                        result = future.result()
                        logger.info(f"🔥 DEATH_MARKER: FUTURE_RESULT_SUCCESS_FOR_WELL_{well_id}")
                        logger.info(f"🔥 ORCHESTRATOR: Well {well_id} result: {result}")
                        execution_results[well_id] = result
                        logger.info(f"🔥 DEATH_MARKER: RESULT_STORED_FOR_WELL_{well_id}")
                    except Exception as exc:
                        import traceback
                        full_traceback = traceback.format_exc()
                        error_msg = f"Well {well_id} generated an exception during execution: {exc}"
                        logger.error(f"🔥 ORCHESTRATOR ERROR: {error_msg}", exc_info=True)
                        logger.error(f"🔥 ORCHESTRATOR FULL TRACEBACK for well {well_id}:\n{full_traceback}")
                        # FAIL-FAST: Re-raise immediately instead of storing error
                        raise

                logger.info("🔥 DEATH_MARKER: COMPLETION_LOOP_FINISHED")

                logger.info(f"🔥 ORCHESTRATOR: All tasks completed, {len(execution_results)} results collected")


            # 🔥 GPU CLEANUP: Clear GPU memory after plate execution
            try:
                from openhcs.core.memory.gpu_cleanup import cleanup_all_gpu_frameworks
                cleanup_all_gpu_frameworks()
                logger.debug("🔥 GPU CLEANUP: Cleared all GPU frameworks after plate execution")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup GPU memory after plate execution: {cleanup_error}")



            logger.info(f"🔥 ORCHESTRATOR: Plate execution completed, checking for analysis consolidation")
            # Run automatic analysis consolidation if enabled
            if self.global_config.analysis_consolidation.enabled:
                try:
                    from openhcs.processing.backends.analysis.consolidate_analysis_results import consolidate_analysis_results

                    # Get results directory from compiled contexts (Option 2: use existing paths)
                    results_dir = None
                    for well_id, context in compiled_contexts.items():
                        # Look for any step that has an output_dir - this is where materialization happens
                        for step_id, step_plan in context.step_plans.items():
                            if 'output_dir' in step_plan:
                                # Found an output directory, check if it has a results subdirectory
                                potential_results_dir = Path(step_plan['output_dir']) / self.global_config.path_planning.materialization_results_path
                                if potential_results_dir.exists():
                                    results_dir = potential_results_dir
                                    logger.info(f"🔍 CONSOLIDATION: Found results directory from step {step_id}: {results_dir}")
                                    break
                        if results_dir:
                            break

                    if results_dir and results_dir.exists():
                        # Check if there are actually CSV files (materialized results)
                        csv_files = list(results_dir.glob("*.csv"))
                        if csv_files:
                            logger.info(f"🔄 CONSOLIDATION: Found {len(csv_files)} CSV files, running consolidation")
                            # Get well IDs from compiled contexts
                            well_ids = list(compiled_contexts.keys())
                            logger.info(f"🔄 CONSOLIDATION: Using well IDs: {well_ids}")

                            consolidate_analysis_results(
                                results_directory=str(results_dir),
                                well_ids=well_ids,
                                consolidation_config=self.global_config.analysis_consolidation,
                                plate_metadata_config=self.global_config.plate_metadata
                            )
                            logger.info("✅ CONSOLIDATION: Completed successfully")
                        else:
                            logger.info(f"⏭️ CONSOLIDATION: No CSV files found in {results_dir}, skipping")
                    else:
                        logger.info(f"⏭️ CONSOLIDATION: No results directory found in compiled contexts")
                except Exception as e:
                    logger.error(f"❌ CONSOLIDATION: Failed: {e}")
            
            # Update state based on execution results
            if all(result.get("status") == "success" for result in execution_results.values()):
                self._state = OrchestratorState.COMPLETED
            else:
                self._state = OrchestratorState.EXEC_FAILED

            logger.info(f"🔥 ORCHESTRATOR: Plate execution finished. Results: {execution_results}")

            return execution_results
        except Exception as e:
            self._state = OrchestratorState.EXEC_FAILED
            logger.error(f"Failed to execute compiled plate: {e}")
            raise

    def get_component_keys(self, group_by: GroupBy, component_filter: Optional[List[Union[str, int]]] = None) -> List[str]:
        """
        Generic method to get component keys for any group_by type.

        This method works with any GroupBy enum value (CHANNEL, SITE, Z_INDEX, WELL)
        and returns the discovered component values as strings to match the pattern
        detection system format.

        Tries metadata cache first, falls back to filename parsing cache if metadata is empty.

        Args:
            group_by: GroupBy enum specifying which component to extract
            component_filter: Optional list of component values to filter by

        Returns:
            List of component values as strings, sorted

        Raises:
            RuntimeError: If orchestrator is not initialized
        """
        if not self.is_initialized():
            raise RuntimeError("Orchestrator must be initialized before getting component keys.")

        # Try metadata cache first (preferred source)
        if group_by in self._metadata_cache and self._metadata_cache[group_by]:
            all_components = list(self._metadata_cache[group_by].keys())
            logger.debug(f"Using metadata cache for {group_by.value}: {len(all_components)} components")
        else:
            # Fall back to filename parsing cache
            if group_by not in self._component_keys_cache:
                raise RuntimeError(f"Component keys cache is empty for {group_by.value}. "
                                 f"Ensure cache_component_keys() was called during initialization.")

            all_components = self._component_keys_cache[group_by]

            if not all_components:
                component_name = group_by.value
                logger.warning(f"No {component_name} values found in input directory: {self.input_dir}")
                return []

            logger.debug(f"Using filename parsing cache for {group_by.value}: {len(all_components)} components")

        if component_filter:
            str_component_filter = {str(c) for c in component_filter}
            selected_components = [comp for comp in all_components if comp in str_component_filter]
            if not selected_components:
                component_name = group_by.value
                logger.warning(f"No {component_name} values from {all_components} match the filter: {component_filter}")
            return selected_components
        else:
            return all_components

    def cache_component_keys(self, components: Optional[List[GroupBy]] = None) -> None:
        """
        Pre-compute and cache component keys for fast access using single-pass parsing.

        This method performs expensive file listing and parsing operations once,
        extracting all component types in a single pass for maximum efficiency.

        Args:
            components: Optional list of GroupBy components to cache.
                       If None, caches all components in the GroupBy enum.
        """
        if not self.is_initialized():
            raise RuntimeError("Orchestrator must be initialized before caching component keys.")

        if components is None:
            components = list(GroupBy)  # Cache all enum values

        logger.info(f"Caching component keys for: {[comp.value for comp in components]}")

        # Initialize component sets for all requested components
        component_sets: Dict[GroupBy, Set[Union[str, int]]] = {}
        for group_by in components:
            if group_by != GroupBy.NONE:  # Skip the empty enum
                component_sets[group_by] = set()

        # Single pass through all filenames - extract all components at once
        try:
            filenames = self.filemanager.list_files(str(self.input_dir), Backend.DISK.value, extensions=DEFAULT_IMAGE_EXTENSIONS)
            logger.debug(f"Parsing {len(filenames)} filenames in single pass...")

            for filename in filenames:
                parsed_info = self.microscope_handler.parser.parse_filename(str(filename))
                if parsed_info:
                    # Extract all requested components from this filename
                    for group_by in component_sets:
                        component_name = group_by.value
                        if component_name in parsed_info and parsed_info[component_name] is not None:
                            component_sets[group_by].add(parsed_info[component_name])
                else:
                    logger.warning(f"Could not parse filename: {filename}")

        except Exception as e:
            logger.error(f"Error listing files or parsing filenames from {self.input_dir}: {e}", exc_info=True)
            # Initialize empty sets for failed parsing
            for group_by in component_sets:
                component_sets[group_by] = set()

        # Convert sets to sorted lists and store in cache
        for group_by, component_set in component_sets.items():
            sorted_components = [str(comp) for comp in sorted(list(component_set))]
            self._component_keys_cache[group_by] = sorted_components
            logger.debug(f"Cached {len(sorted_components)} {group_by.value} keys")

            if not sorted_components:
                logger.warning(f"No {group_by.value} values found in input directory: {self.input_dir}")

        logger.info(f"Component key caching complete. Cached {len(component_sets)} component types in single pass.")

    def clear_component_cache(self, components: Optional[List[GroupBy]] = None) -> None:
        """
        Clear cached component keys to force recomputation.

        Use this when the input directory contents have changed and you need
        to refresh the component key cache.

        Args:
            components: Optional list of GroupBy components to clear from cache.
                       If None, clears entire cache.
        """
        if components is None:
            self._component_keys_cache.clear()
            logger.info("Cleared entire component keys cache")
        else:
            for group_by in components:
                if group_by in self._component_keys_cache:
                    del self._component_keys_cache[group_by]
                    logger.debug(f"Cleared cache for {group_by.value}")
            logger.info(f"Cleared cache for {len(components)} component types")

    def cache_metadata(self) -> None:
        """
        Cache all metadata from metadata handler for fast access.

        This method calls the metadata handler's parse_metadata() method once
        and stores the results for instant access to component key→name mappings.
        Call this after orchestrator initialization to enable metadata-based
        component names.
        """
        if not self.is_initialized() or self.input_dir is None or self.microscope_handler is None:
            raise RuntimeError("Orchestrator must be initialized before caching metadata.")

        try:
            # Parse all metadata once using enum→method mapping
            # Use plate_path for metadata loading since metadata files are in plate root
            metadata = self.microscope_handler.metadata_handler.parse_metadata(self.plate_path)

            # Initialize all GroupBy components with component keys mapped to None
            for group_by in [GroupBy.CHANNEL, GroupBy.WELL, GroupBy.SITE, GroupBy.Z_INDEX]:
                # Get all component keys for this GroupBy from filename parsing
                component_keys = self.get_component_keys(group_by)
                # Create dict mapping each key to None (no metadata available)
                self._metadata_cache[group_by] = {key: None for key in component_keys}

            # Update with actual metadata from metadata handler where available
            for component_name, mapping in metadata.items():
                try:
                    group_by = GroupBy(component_name)  # Convert string to enum
                    # For OpenHCS plates, metadata keys might be the only source of component keys
                    # Merge metadata keys with any existing component keys from filename parsing
                    if group_by in self._metadata_cache:
                        # Start with existing component keys (from filename parsing)
                        combined_cache = self._metadata_cache[group_by].copy()
                        # Add any metadata keys that weren't found in filename parsing
                        for metadata_key in mapping.keys():
                            if metadata_key not in combined_cache:
                                combined_cache[metadata_key] = None
                        # Update with actual metadata values
                        combined_cache.update(mapping)
                        self._metadata_cache[group_by] = combined_cache
                    else:
                        self._metadata_cache[group_by] = mapping
                    logger.debug(f"Updated metadata for {group_by.value}: {len(mapping)} entries with real data")
                except ValueError:
                    logger.warning(f"Unknown component type in metadata: {component_name}")

            # Log what we have for each component
            for group_by in [GroupBy.CHANNEL, GroupBy.WELL, GroupBy.SITE, GroupBy.Z_INDEX]:
                mapping = self._metadata_cache[group_by]
                real_metadata_count = sum(1 for v in mapping.values() if v is not None)
                total_keys = len(mapping)
                logger.debug(f"Cached {group_by.value}: {total_keys} keys, {real_metadata_count} with metadata")

            logger.info(f"Metadata caching complete. All {len(self._metadata_cache)} component types populated.")

        except Exception as e:
            logger.warning(f"Could not cache metadata: {e}")
            # Don't fail - metadata is optional enhancement

    def get_component_metadata(self, group_by: GroupBy, key: str) -> Optional[str]:
        """
        Get metadata display name for a specific component key.

        Args:
            group_by: GroupBy enum specifying component type
            key: Component key (e.g., "1", "2", "A01")

        Returns:
            Display name from metadata, or None if not available
            Example: get_component_metadata(GroupBy.CHANNEL, "1") → "HOECHST 33342"
        """
        if group_by in self._metadata_cache:
            return self._metadata_cache[group_by].get(key)
        return None

    def clear_metadata_cache(self) -> None:
        """
        Clear cached metadata to force recomputation.

        Use this when the input directory contents have changed and you need
        to refresh the metadata cache.
        """
        self._metadata_cache.clear()
        logger.info("Cleared metadata cache")

    async def apply_new_global_config(self, new_config: GlobalPipelineConfig):
        """
        Applies a new GlobalPipelineConfig to this orchestrator instance.

        This updates the internal global_config reference. Subsequent operations,
        especially new context creation and pipeline compilations, will use this
        new configuration.

        Args:
            new_config: The new GlobalPipelineConfig object.
        """
        if not isinstance(new_config, GlobalPipelineConfig):
            logger.error(
                f"Attempted to apply invalid config type {type(new_config)} to PipelineOrchestrator. Expected GlobalPipelineConfig."
            )
            return

        logger.info(
            f"PipelineOrchestrator (plate: {self.plate_path}, workspace: {self.workspace_path}) "
            f"is applying new GlobalPipelineConfig. Old num_workers: {self.global_config.num_workers}, "
            f"New num_workers: {new_config.num_workers}"
        )
        self.global_config = new_config
        # Re-initialization of components like path_planner or materialization_flag_planner
        # is implicitly handled if they are created fresh during compilation using contexts
        # that are generated with the new self.global_config.
        # If any long-lived orchestrator components directly cache parts of global_config
        # and need explicit updating, that would be done here. For now, updating the
        # reference is the primary action.
        logger.info("New GlobalPipelineConfig applied to orchestrator.")
