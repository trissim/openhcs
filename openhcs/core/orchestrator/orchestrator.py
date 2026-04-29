"""
Consolidated orchestrator module for OpenHCS.

This module provides a unified PipelineOrchestrator class that implements
a two-phase (compile-all-then-execute-all) pipeline execution model.
"""

import logging
import concurrent.futures
import multiprocessing
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Set

from openhcs.constants.constants import (
    Backend,
    DEFAULT_IMAGE_EXTENSIONS,
    GroupBy,
    OrchestratorState,
    get_openhcs_config,
    AllComponents,
    VariableComponents,
)
from openhcs.constants import Microscope
from openhcs.core.config import GlobalPipelineConfig
from openhcs.config_framework.global_config import get_current_global_config


from openhcs.core.metadata_cache import get_metadata_cache, MetadataCache
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.pipeline.compiler import PipelineCompiler
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.components.validation import convert_enum_by_value
from openhcs.core.orchestrator.execution_result import ExecutionResult, ExecutionStatus
from openhcs.core.progress import (
    emit,
    set_progress_queue,
    ProgressPhase,
    ProgressStatus,
    create_event,
)
from polystore.filemanager import FileManager

# Zarr backend is CPU-only; always import it (even in subprocess/no-GPU mode)
import os
from polystore.zarr import ZarrStorageBackend

# PipelineConfig now imported directly above
from openhcs.config_framework.lazy_factory import (
    resolve_lazy_configurations_for_serialization,
)
from openhcs.microscopes import create_microscope_handler
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.processing.backends.analysis.consolidate_analysis_results import (
    consolidate_results_directories,
)


# Import generic component system - required for orchestrator functionality

# Optional napari import for visualization
try:
    from openhcs.runtime.napari_stream_visualizer import NapariStreamVisualizer

    NapariVisualizerType = NapariStreamVisualizer
except ImportError:
    # Create a placeholder type for type hints when napari is not available
    NapariStreamVisualizer = None
    NapariVisualizerType = Any  # Use Any for type hints when napari is not available

# Optional GPU memory management imports
try:
    from openhcs.core.memory import cleanup_all_gpu_frameworks
except ImportError:
    cleanup_all_gpu_frameworks = None


logger = logging.getLogger(__name__)


def _merge_nested_dataclass(pipeline_value, global_value):
    """
    Recursively merge nested dataclass configs.

    For each field in the dataclass:
    - If pipeline value is not None, use it
    - Otherwise, use global value

    This ensures that None values in nested configs resolve to global config values.
    """
    from dataclasses import is_dataclass, fields as dataclass_fields

    if not is_dataclass(pipeline_value) or not is_dataclass(global_value):
        # Not dataclasses, return pipeline value as-is
        return pipeline_value

    # Both are dataclasses - merge field by field
    merged_values = {}
    for field in dataclass_fields(type(pipeline_value)):
        # CRITICAL FIX: Use __dict__.get() to get RAW stored value, not getattr()
        # For lazy dataclasses, getattr() triggers resolution which falls back to class defaults
        # We need the actual None value to know if it should inherit from global config
        raw_pipeline_field = pipeline_value.__dict__.get(field.name)
        global_field_value = getattr(global_value, field.name)

        if raw_pipeline_field is not None:
            # Pipeline has an explicitly set value - check if it's a nested dataclass that needs merging
            if is_dataclass(raw_pipeline_field) and is_dataclass(global_field_value):
                merged_values[field.name] = _merge_nested_dataclass(
                    raw_pipeline_field, global_field_value
                )
            else:
                merged_values[field.name] = raw_pipeline_field
        else:
            # Pipeline value is None - use global value
            merged_values[field.name] = global_field_value

    # Create new instance with merged values
    return type(pipeline_value)(**merged_values)


def _create_merged_config(
    pipeline_config: "PipelineConfig", global_config: GlobalPipelineConfig
) -> GlobalPipelineConfig:
    """
    Pure function for creating merged config that preserves None values for sibling inheritance.

    Follows OpenHCS stateless architecture principles - no side effects, explicit dependencies.
    Extracted from apply_pipeline_config to eliminate code duplication.
    """
    logger.debug(
        f"Starting merge with pipeline_config={type(pipeline_config)} and global_config={type(global_config)}"
    )

    merged_config_values = {}
    for field in fields(GlobalPipelineConfig):
        # CRITICAL: Access raw stored value from __dict__ to avoid lazy resolution fallback to MRO defaults
        # For lazy dataclasses, getattr() triggers resolution which falls back to GlobalPipelineConfig defaults
        # We need the actual None value to know if it should inherit from global config
        pipeline_value = pipeline_config.__dict__.get(field.name)

        if pipeline_value is not None:
            # CRITICAL FIX: For lazy configs, merge with global config BEFORE converting to base
            # This ensures None values in lazy configs resolve to global values
            # Then convert to base config to store in thread-local context
            if hasattr(pipeline_value, "to_base_config"):
                # This is a lazy config - merge with global config first
                global_value = getattr(global_config, field.name)
                from dataclasses import is_dataclass

                if is_dataclass(global_value):
                    # Merge lazy config with global config to resolve None values
                    merged_lazy = _merge_nested_dataclass(pipeline_value, global_value)
                    # Now convert merged result to base config
                    converted_value = (
                        merged_lazy.to_base_config()
                        if hasattr(merged_lazy, "to_base_config")
                        else merged_lazy
                    )
                    merged_config_values[field.name] = converted_value
                else:
                    # No global value to merge with, just convert to base
                    converted_value = pipeline_value.to_base_config()
                    merged_config_values[field.name] = converted_value
            else:
                # CRITICAL FIX: For base dataclass configs, merge nested fields
                # This ensures None values in nested configs resolve to global values
                global_value = getattr(global_config, field.name)
                from dataclasses import is_dataclass

                if is_dataclass(pipeline_value) and is_dataclass(global_value):
                    merged_config_values[field.name] = _merge_nested_dataclass(
                        pipeline_value, global_value
                    )
                else:
                    # Regular value - use as-is
                    merged_config_values[field.name] = pipeline_value
        else:
            global_value = getattr(global_config, field.name)
            merged_config_values[field.name] = global_value

    result = GlobalPipelineConfig(**merged_config_values)
    return result


def _execute_axis_with_sequential_combinations(
    pipeline_definition: List[AbstractStep],
    axis_contexts: List[tuple],  # List of (context_key, frozen_context) tuples
    visualizer: Optional["NapariVisualizerType"],
    execution_id: str,
    plate_id: str,
    worker_slot: str,
    owned_wells: List[str],
) -> ExecutionResult:
    """
    Execute all sequential combinations for a single axis in order.

    This function runs in a worker process and handles VFS clearing between combinations.
    Multiple axes can run in parallel, but combinations within an axis are sequential.

    Args:
        pipeline_definition: List of pipeline steps to execute
        axis_contexts: List of (context_key, frozen_context) tuples for this axis
        visualizer: Optional Napari visualizer (not used in multiprocessing)
        execution_id: Unique identifier for this execution
        plate_id: Path or identifier for plate being processed

    Returns:
        ExecutionResult with status for the entire axis (all combinations)
    """
    # Precondition: axis_contexts must not be empty
    if not axis_contexts:
        raise ValueError(
            "axis_contexts cannot be empty - this indicates a bug in the caller"
        )

    # Extract axis_id from first context
    first_context_key, first_context = axis_contexts[0]
    axis_id = first_context.axis_id
    total_steps = len(pipeline_definition)

    emit(
        execution_id=execution_id,
        plate_id=plate_id,
        axis_id=axis_id,
        step_name="pipeline",
        phase=ProgressPhase.AXIS_STARTED,
        status=ProgressStatus.STARTED,
        completed=0,
        total=total_steps,
        percent=0.0,
        worker_slot=worker_slot,
        owned_wells=owned_wells,
    )

    for combo_idx, (context_key, frozen_context) in enumerate(axis_contexts):
        # Execute this combination
        result = _execute_single_axis_static(
            pipeline_definition,
            frozen_context,
            visualizer,
            execution_id,
            plate_id,
            worker_slot,
            owned_wells,
        )

        # Clear VFS after each combination to prevent memory accumulation
        # This must happen REGARDLESS of success/failure to prevent memory leaks
        # when worker processes handle multiple wells sequentially
        from polystore.base import reset_memory_backend
        from openhcs.core.memory import cleanup_all_gpu_frameworks

        reset_memory_backend()
        if cleanup_all_gpu_frameworks:
            cleanup_all_gpu_frameworks()

        # Check if this combination failed (after cleanup to prevent memory leaks)
        if not result.is_success():
            logger.error(
                f"🔄 WORKER: Combination {context_key} failed for axis {axis_id}"
            )
            emit(
                execution_id=execution_id,
                plate_id=plate_id,
                axis_id=axis_id,
                step_name="pipeline",
                phase=ProgressPhase.AXIS_ERROR,
                status=ProgressStatus.ERROR,
                completed=0,
                total=total_steps,
                percent=0.0,
                message=result.error_message,
                worker_slot=worker_slot,
                owned_wells=owned_wells,
            )
            return ExecutionResult.error(
                axis_id=axis_id,
                failed_combination=context_key,
                error_message=result.error_message,
            )

    emit(
        execution_id=execution_id,
        plate_id=plate_id,
        axis_id=axis_id,
        step_name="pipeline",
        phase=ProgressPhase.AXIS_COMPLETED,
        status=ProgressStatus.SUCCESS,
        completed=total_steps,
        total=total_steps,
        percent=100.0,
        worker_slot=worker_slot,
        owned_wells=owned_wells,
    )
    return ExecutionResult.success(axis_id=axis_id)


def _execute_single_axis_static(
    pipeline_definition: List[AbstractStep],
    frozen_context: "ProcessingContext",
    visualizer: Optional["NapariVisualizerType"],
    execution_id: str,
    plate_id: str,
    worker_slot: str,
    owned_wells: List[str],
) -> ExecutionResult:
    """
    Static version of _execute_single_axis for multiprocessing compatibility.

    This function is identical to PipelineOrchestrator._execute_single_axis but doesn't
    require an orchestrator instance, making it safe for pickling in ProcessPoolExecutor.

    Args:
        pipeline_definition: List of pipeline steps to execute
        frozen_context: Frozen processing context for this axis
        visualizer: Optional Napari visualizer (not used in multiprocessing)
        execution_id: Unique identifier for this execution
        plate_id: Path or identifier for plate being processed

    Returns:
        ExecutionResult with status for this axis
    """
    axis_id = frozen_context.axis_id
    total_steps = len(pipeline_definition)

    # NUCLEAR VALIDATION
    if not frozen_context.is_frozen():
        error_msg = f"Context for axis {axis_id} is not frozen before execution"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if not pipeline_definition:
        error_msg = f"Empty pipeline_definition for axis {axis_id}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Set execution tracking fields on context (these are allowed even when frozen)
    # These fields are set at execution time, not compilation time
    object.__setattr__(frozen_context, "execution_id", execution_id)
    object.__setattr__(frozen_context, "plate_id", plate_id)
    object.__setattr__(frozen_context, "worker_slot", worker_slot)
    object.__setattr__(frozen_context, "owned_wells", owned_wells)

    # Execute each step in the pipeline
    for step_index, step in enumerate(pipeline_definition):
        step_name = frozen_context.step_plans[step_index]["step_name"]

        emit(
            execution_id=execution_id,
            plate_id=plate_id,
            axis_id=axis_id,
            step_name=step_name,
            phase=ProgressPhase.STEP_STARTED,
            status=ProgressStatus.STARTED,
            completed=step_index,
            total=total_steps,
            percent=(step_index / total_steps) * 100.0,
            worker_slot=worker_slot,
            owned_wells=owned_wells,
        )

        # Call process method on step instance
        step.process(frozen_context, step_index)

        emit(
            execution_id=execution_id,
            plate_id=plate_id,
            axis_id=axis_id,
            step_name=step_name,
            phase=ProgressPhase.STEP_COMPLETED,
            status=ProgressStatus.SUCCESS,
            completed=step_index + 1,
            total=total_steps,
            percent=((step_index + 1) / total_steps) * 100.0,
            worker_slot=worker_slot,
            owned_wells=owned_wells,
        )

        # Handle visualization if requested
        if visualizer:
            step_plan = frozen_context.step_plans[step_index]
            if step_plan["visualize"]:
                output_dir = step_plan["output_dir"]
                write_backend = step_plan["write_backend"]
                if output_dir:
                    logger.debug(
                        f"Visualizing output for step {step_index} from path {output_dir} (backend: {write_backend}) for axis {axis_id}"
                    )
                    visualizer.visualize_path(
                        step_id=f"step_{step_index}",
                        path=str(output_dir),
                        backend=write_backend,
                        axis_id=axis_id,
                    )
                else:
                    logger.warning(
                        f"Step {step_index} in axis {axis_id} flagged for visualization but 'output_dir' is missing in its plan."
                    )

    return ExecutionResult.success(axis_id=axis_id)


def _execute_worker_lane_static(
    pipeline_definition: List[AbstractStep],
    lane_axis_contexts: List[tuple[str, List[tuple]]],
    visualizer: Optional["NapariVisualizerType"],
    execution_id: str,
    plate_id: str,
    worker_slot: str,
    owned_wells: List[str],
) -> Dict[str, ExecutionResult]:
    """Execute a deterministic worker lane: wells sequentially within one slot."""
    lane_results: Dict[str, ExecutionResult] = {}
    for axis_id, axis_contexts in lane_axis_contexts:
        lane_results[axis_id] = _execute_axis_with_sequential_combinations(
            pipeline_definition=pipeline_definition,
            axis_contexts=axis_contexts,
            visualizer=visualizer,
            execution_id=execution_id,
            plate_id=plate_id,
            worker_slot=worker_slot,
            owned_wells=owned_wells,
        )
    return lane_results


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

    # Note: Import hook system was removed - using existing comprehensive registries

    # Create unique worker identifier using PID and timestamp
    worker_pid = os.getpid()
    worker_timestamp = int(
        time.time() * 1000000
    )  # Microsecond precision for uniqueness
    worker_id = f"{worker_pid}_{worker_timestamp}"
    worker_log_file = f"{log_file_base}_worker_{worker_id}.log"

    # Configure root logger to capture ALL logs from worker process
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear any inherited handlers

    # Create file handler for worker logs
    file_handler = logging.FileHandler(worker_log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    # Ensure all OpenHCS module logs are captured
    logging.getLogger("openhcs").setLevel(logging.INFO)

    # Get worker logger
    worker_logger = logging.getLogger("openhcs.worker")


def _configure_worker_with_gpu(
    log_file_base: str,
    global_config_dict: dict,
    progress_queue=None,
    progress_context=None,
):
    """
    Configure logging, function registry, and GPU registry for worker process.

    This function is called once per worker process when it starts.
    It sets up logging, function registry, and GPU registry initialization.

    Args:
        log_file_base: Base path for worker log files (empty string if no logging)
        global_config_dict: Serialized global configuration for GPU registry setup
    """
    import logging
    import os

    # Workers should be allowed to import GPU libs if available.
    # The parent subprocess runner may set OPENHCS_SUBPROCESS_NO_GPU=1 to stay lean,
    # but that flag must not leak into worker processes.
    os.environ.pop("OPENHCS_SUBPROCESS_NO_GPU", None)
    os.environ.pop("POLYSTORE_SUBPROCESS_NO_GPU", None)

    # Configure logging only if log_file_base is provided
    if log_file_base:
        _configure_worker_logging(log_file_base)
        worker_logger = logging.getLogger("openhcs.worker")
    else:
        # Set up basic logging for worker messages
        logging.basicConfig(level=logging.INFO)
        worker_logger = logging.getLogger("openhcs.worker")

    # Initialize function registry for this worker process
    try:
        # Import and initialize function registry (will auto-discover all libraries)
        import openhcs.processing.func_registry as func_registry_module

        # Force initialization if not already done (workers need full registry)
        with func_registry_module._registry_lock:
            if not func_registry_module._registry_initialized:
                func_registry_module._auto_initialize_registry()

    except Exception as e:
        # Don't raise - let worker continue, registry will auto-init on first function call
        pass

    # Initialize GPU registry for this worker process
    try:
        # Reconstruct global config from dict
        from openhcs.core.config import GlobalPipelineConfig

        global_config = GlobalPipelineConfig(**global_config_dict)

        # Initialize GPU registry for this worker
        from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry

        setup_global_gpu_registry(global_config)

    except Exception as e:
        # Don't raise - let worker continue without GPU if needed
        pass

    if progress_queue is not None and progress_context is not None:
        from openhcs.core.progress import set_progress_queue

        # Set progress queue for worker processes
        set_progress_queue(progress_queue)


# Global variable to store log file base for worker processes
_worker_log_file_base = None


class PipelineOrchestrator:
    """
    Updated orchestrator supporting both global and per-orchestrator configuration.

    Global configuration: Updates all orchestrators (existing behavior)
    Per-orchestrator configuration: Affects only this orchestrator instance

    The orchestrator first compiles the pipeline for all specified axis values,
    creating frozen, immutable ProcessingContexts using `compile_plate_for_processing()`.
    Then, it executes the (now stateless) pipeline definition against these contexts,
    potentially in parallel, using `execute_compiled_plate()`.
    """

    # ObjectState delegation: when ObjectState stores this orchestrator, extract
    # editable parameters from pipeline_config (a dataclass) instead of the orchestrator.
    # This enables time-travel to track the orchestrator lifecycle while forms edit the config.
    __objectstate_delegate__ = "pipeline_config"

    def __init__(
        self,
        plate_path: Union[str, Path],
        workspace_path: Optional[Union[str, Path]] = None,
        *,
        pipeline_config: Optional["PipelineConfig"] = None,
        storage_registry: Optional[Any] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        # Lock removed - was orphaned code never used

        # Validate shared global context exists
        if get_current_global_config(GlobalPipelineConfig) is None:
            raise RuntimeError(
                "No global configuration context found. "
                "Ensure application startup has called ensure_global_config_context()."
            )

        # Track executor for cancellation support
        self._executor = None
        self._cancelled = False  # Track cancellation requests
        self.execution_id = f"local::{plate_path}"

        # Initialize auto-sync control for pipeline config
        self._pipeline_config = None
        self._auto_sync_enabled = True

        # Context management now handled by contextvars-based system

        # Initialize per-orchestrator configuration
        # DUAL-AXIS FIX: Always create a PipelineConfig instance to make orchestrator detectable as context provider
        # This ensures the orchestrator has a dataclass attribute for stack introspection
        # PipelineConfig is already the lazy version of GlobalPipelineConfig
        from openhcs.core.config import PipelineConfig

        if pipeline_config is None:
            # CRITICAL FIX: Create pipeline config that inherits from global config
            # This ensures the orchestrator's pipeline_config has the global values for resolution
            pipeline_config = PipelineConfig()

        # CRITICAL FIX: Do NOT apply global config inheritance during initialization
        # PipelineConfig should always have None values that resolve through lazy resolution
        # Copying concrete values breaks the placeholder system and makes all fields appear "explicitly set"

        self.pipeline_config = pipeline_config

        # CRITICAL FIX: Expose pipeline config as public attribute for dual-axis resolver discovery
        # The resolver's _is_context_provider method only finds public attributes (skips _private)
        # This allows the resolver to discover the orchestrator's pipeline config during context resolution
        self.pipeline_config = pipeline_config
        logger.info(
            "PipelineOrchestrator initialized with PipelineConfig for context discovery."
        )

        # REMOVED: Unnecessary thread-local modification
        # The orchestrator should not modify thread-local storage during initialization
        # Global config is already available through the dual-axis resolver fallback

        # Convert to Path and validate
        if plate_path:
            plate_path = Path(plate_path)

            # Validate filesystem paths (skip for OMERO virtual paths)
            if not str(plate_path).startswith("/omero/"):
                if not plate_path.is_absolute():
                    raise ValueError(f"Plate path must be absolute: {plate_path}")
                if not plate_path.exists():
                    raise FileNotFoundError(f"Plate path does not exist: {plate_path}")
                if not plate_path.is_dir():
                    raise NotADirectoryError(
                        f"Plate path is not a directory: {plate_path}"
                    )

        # Initialize _plate_path_frozen first to allow plate_path to be set during initialization
        object.__setattr__(self, "_plate_path_frozen", False)

        self.plate_path = plate_path
        self.workspace_path = workspace_path

        if self.plate_path is None and self.workspace_path is None:
            raise ValueError(
                "Either plate_path or workspace_path must be provided for PipelineOrchestrator."
            )

        # Freeze plate_path immediately after setting it to prove immutability
        object.__setattr__(self, "_plate_path_frozen", True)
        logger.info(f"🔒 PLATE_PATH FROZEN: {self.plate_path} is now immutable")

        if storage_registry:
            self.registry = storage_registry
            logger.info("PipelineOrchestrator using provided StorageRegistry instance.")
        else:
            # Use the global registry directly (don't copy) so that reset_memory_backend() works correctly
            # The global registry is a singleton, and VFS clearing needs to clear the same instance
            from polystore.base import (
                storage_registry as global_storage_registry,
                ensure_storage_registry,
            )

            # Ensure registry is initialized
            ensure_storage_registry()
            self.registry = global_storage_registry
            logger.info("PipelineOrchestrator using global StorageRegistry instance.")

        # Override zarr backend with orchestrator's config
        shared_context = get_current_global_config(GlobalPipelineConfig)
        zarr_backend_with_config = ZarrStorageBackend(shared_context.zarr_config)
        self.registry[Backend.ZARR.value] = zarr_backend_with_config
        logger.info(
            f"Orchestrator zarr backend configured with {shared_context.zarr_config.compressor.value} compression"
        )

        # Orchestrator always creates its own FileManager, using the determined registry
        self.filemanager = FileManager(self.registry)
        self.input_dir: Optional[Path] = None
        self.microscope_handler: Optional[MicroscopeHandler] = None
        self.default_pipeline_definition: Optional[List[AbstractStep]] = None
        self._initialized: bool = False
        self._state: OrchestratorState = OrchestratorState.CREATED

        # Progress callback for real-time execution updates
        self.progress_callback = progress_callback
        if progress_callback:
            logger.info("PipelineOrchestrator initialized with progress callback")

        # Component keys cache for fast access - uses AllComponents (includes multiprocessing axis)
        self._component_keys_cache: Dict["AllComponents", List[str]] = {}

        # Metadata cache service - per-orchestrator instance (not global singleton)
        from openhcs.core.metadata_cache import MetadataCache

        self._metadata_cache_service = MetadataCache()

        # Viewer management - shared between pipeline execution and image browser
        self._visualizers = {}  # Dict[(backend_name, port)] -> visualizer instance

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute, preventing modification of plate_path after it's frozen.

        This proves that plate_path is truly immutable after initialization.
        """
        if name == "plate_path" and getattr(self, "_plate_path_frozen", False):
            import traceback

            stack_trace = "".join(traceback.format_stack())
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

    def get_or_create_visualizer(self, config, vis_config=None):
        """
        Get existing visualizer or create a new one for the given config.

        This method is shared between pipeline execution and image browser to avoid
        duplicating viewer instances. Viewers are tracked by (backend_name, port) key.

        Args:
            config: Streaming config (any StreamingConfig subclass)
            vis_config: Optional visualizer config (can be None for image browser)

        Returns:
            Visualizer instance
        """
        from openhcs.core.config import StreamingConfig

        # Streaming configs should be managed by the centralized ViewerStateManager
        if isinstance(config, StreamingConfig):
            # Ensure a queue tracker exists for this viewer
            from zmqruntime.queue_tracker import GlobalQueueTrackerRegistry

            registry = GlobalQueueTrackerRegistry()
            registry.get_or_create_tracker(config.port, config.viewer_type)

            key = (config.viewer_type, config.port)

            # Use zmqruntime's atomic get_or_create to avoid races that spawned duplicate viewers
            from zmqruntime import get_or_create_viewer

            viewer, created = get_or_create_viewer(
                viewer_type=config.viewer_type,
                port=config.port,
                factory=lambda: config.create_visualizer(self.filemanager, vis_config),
                wait_for_ready=True,
                ready_timeout=30.0,
            )

            # Keep a reference for backward compatibility
            self._visualizers[key] = viewer
            return viewer

        # Non-streaming (local) visualizers: create and start synchronously
        vis = config.create_visualizer(self.filemanager, vis_config)
        try:
            vis.start_viewer()
        except Exception:
            # Some visualizers expose start() instead of start_viewer()
            if hasattr(vis, "start"):
                vis.start()
            else:
                logger.exception("Failed to start non-streaming visualizer")

        # Store for compatibility
        backend_name = config.backend.name if hasattr(config, "backend") else "unknown"
        self._visualizers[(backend_name,)] = vis
        return vis

    def initialize_microscope_handler(self):
        """Initializes the microscope handler."""
        if self.microscope_handler is not None:
            logger.debug("Microscope handler already initialized.")
            return
        #        if self.input_dir is None:
        #            raise RuntimeError("Workspace (and input_dir) must be initialized before microscope handler.")

        logger.info(
            f"Initializing microscope handler using input directory: {self.input_dir}..."
        )
        try:
            # Use configured microscope type or auto-detect
            # Use SAVED global config (not live edits) for orchestrator initialization
            shared_context = get_current_global_config(
                GlobalPipelineConfig, use_live=False
            )
            microscope_type = (
                shared_context.microscope.value
                if shared_context.microscope != Microscope.AUTO
                else "auto"
            )
            self.microscope_handler = create_microscope_handler(
                plate_folder=str(self.plate_path),
                filemanager=self.filemanager,
                microscope_type=microscope_type,
            )
            logger.info(
                f"Initialized microscope handler: {type(self.microscope_handler).__name__}"
            )
        except Exception as e:
            error_msg = f"Failed to create microscope handler: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def initialize(
        self, workspace_path: Optional[Union[str, Path]] = None
    ) -> "PipelineOrchestrator":
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
                self.plate_path, self.filemanager
            )

            # Use the actual image directory returned by the microscope handler
            # All handlers now return Path (including OMERO with virtual paths)
            self.input_dir = Path(actual_image_dir)
            logger.info(f"Set input directory to: {self.input_dir}")

            # Log effective backend intent early for debugging test/UI differences
            try:
                vfs_cfg = (
                    self.get_effective_config().vfs_config
                    if self.pipeline_config
                    else None
                )
                if vfs_cfg is not None:
                    logger.info(
                        "VFS config at init: read_backend=%s intermediate_backend=%s materialization_backend=%s",
                        getattr(vfs_cfg, "read_backend", None),
                        getattr(vfs_cfg, "intermediate_backend", None),
                        getattr(vfs_cfg, "materialization_backend", None),
                    )
            except Exception:
                logger.debug("Could not log VFS config at init", exc_info=True)

            # Set workspace_path based on what the handler returned
            if actual_image_dir != self.plate_path:
                # Handler created a workspace (or virtual path for OMERO)
                self.workspace_path = (
                    Path(actual_image_dir).parent
                    if Path(actual_image_dir).name != "workspace"
                    else Path(actual_image_dir)
                )
            else:
                # Handler used plate directly (like OpenHCS)
                self.workspace_path = None

            # Mark as initialized BEFORE caching to avoid chicken-and-egg problem
            self._initialized = True
            self._state = OrchestratorState.READY

            # Auto-cache component keys and metadata for instant access
            logger.info("Caching component keys and metadata...")
            self.cache_component_keys()
            self._metadata_cache_service.cache_metadata(
                self.microscope_handler, self.plate_path, self._component_keys_cache
            )

            # Ensure complete OpenHCS metadata exists
            self._ensure_openhcs_metadata()

            logger.info(
                "PipelineOrchestrator fully initialized with cached component keys and metadata."
            )
            return self
        except Exception as e:
            self._state = OrchestratorState.INIT_FAILED
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    def is_initialized(self) -> bool:
        return self._initialized

    def _ensure_openhcs_metadata(self) -> None:
        """Ensure complete OpenHCS metadata exists for the plate.

        Uses the same context creation logic as pipeline execution to get full metadata
        with channel names from metadata files (HTD, Index.xml, etc).

        Skips OMERO and other non-disk-based microscope handlers since they don't have
        real disk directories.
        """
        from openhcs.microscopes.openhcs import OpenHCSMetadataGenerator

        # Skip metadata creation for OMERO and other non-disk-based handlers
        # OMERO uses virtual paths like /omero/plate_1 which are not real directories
        if self.microscope_handler.microscope_type == "omero":
            logger.debug(
                "Skipping metadata creation for OMERO plate (uses virtual paths)"
            )
            return

        # For plates with virtual workspace, metadata is already created by _build_virtual_mapping()
        # We just need to add the component metadata to the existing "." subdirectory
        from polystore.metadata_writer import get_subdirectory_name

        subdir_name = get_subdirectory_name(self.input_dir, self.plate_path)

        # Create context using SAME logic as create_context() to get full metadata
        context = self.create_context(axis_id="metadata_init")

        # Determine correct backend using handler's logic (virtual_workspace for ImageXpress/Opera, disk for others)
        backend = self.microscope_handler.get_primary_backend(
            self.plate_path, self.filemanager
        )
        logger.debug(f"Using backend '{backend}' for metadata extraction")

        # Create metadata (will skip if already complete)
        generator = OpenHCSMetadataGenerator(self.filemanager)
        generator.create_metadata(
            context,
            str(self.input_dir),
            backend,
            is_main=True,
            plate_root=str(self.plate_path),
            sub_dir=subdir_name,
            skip_if_complete=True,
        )

    def get_results_path(self) -> Path:
        """Get the results directory path for this orchestrator's plate.

        Uses the same logic as PathPlanner._get_results_path() to ensure consistency.
        This is the single source of truth for where results are stored.

        Returns:
            Path to results directory (absolute or relative to output plate root)
        """
        from openhcs.core.pipeline.path_planner import PipelinePathPlanner

        # Get materialization_results_path from global config
        materialization_path = self.global_config.materialization_results_path

        # If absolute, use as-is
        if Path(materialization_path).is_absolute():
            return Path(materialization_path)

        # If relative, resolve relative to output plate root
        # Use path_planning_config from global config
        path_config = self.global_config.path_planning_config
        output_plate_root = PipelinePathPlanner.build_output_plate_root(
            self.plate_path, path_config, is_per_step_materialization=False
        )

        return output_plate_root / materialization_path

    def create_context(self, axis_id: str) -> ProcessingContext:
        """Creates a ProcessingContext for a given multiprocessing axis value."""
        if not self.is_initialized():
            raise RuntimeError(
                "Orchestrator must be initialized before calling create_context()."
            )
        if not axis_id:
            raise ValueError("Axis identifier must be provided.")
        if self.input_dir is None:
            raise RuntimeError(
                "Orchestrator input_dir is not set; initialize orchestrator first."
            )

        context = ProcessingContext(
            global_config=self.get_effective_config(),
            axis_id=axis_id,
            filemanager=self.filemanager,
        )
        # Orchestrator reference removed - was orphaned and unpickleable
        context.microscope_handler = self.microscope_handler
        context.input_dir = self.input_dir
        context.workspace_path = self.workspace_path
        context.plate_path = self.plate_path  # Add plate_path for path planner

        # CRITICAL: Pass metadata cache for OpenHCS metadata creation
        # Extract cached metadata from service and convert to dict format expected by OpenHCSMetadataGenerator
        metadata_dict = {}
        for component in AllComponents:
            cached_metadata = self._metadata_cache_service.get_cached_metadata(
                component
            )
            if cached_metadata:
                metadata_dict[component] = cached_metadata
        context.metadata_cache = metadata_dict

        return context

    def compile_pipelines(
        self,
        pipeline_definition: List[AbstractStep],
        well_filter: Optional[List[str]] = None,
        enable_visualizer_override: bool = False,
        is_zmq_execution: bool = False,
    ) -> Dict[str, ProcessingContext]:
        """Compile pipelines for axis values (well_filter name preserved for UI compatibility)."""
        return PipelineCompiler.compile_pipelines(
            orchestrator=self,
            pipeline_definition=pipeline_definition,
            axis_filter=well_filter,  # Translate well_filter to axis_filter for generic backend
            enable_visualizer_override=enable_visualizer_override,
            is_zmq_execution=is_zmq_execution,
        )

    def cancel_execution(self):
        """
        Cancel ongoing execution by shutting down the executor.

        This gracefully cancels pending futures and shuts down worker processes
        without killing all child processes (preserving Napari viewers, etc.).
        """
        # Set cancellation flag so that any waiting loops can detect cancellation
        self._cancelled = True

        if self._executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.warning(f"🔥 ORCHESTRATOR: Failed to cancel executor: {e}")

    def execute_compiled_plate(
        self,
        pipeline_definition: List[AbstractStep],
        compiled_contexts: Dict[str, ProcessingContext],
        max_workers: Optional[int] = None,
        visualizer: Optional[NapariVisualizerType] = None,
        log_file_base: Optional[str] = None,
        progress_queue=None,
        progress_context=None,
        worker_assignments: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute-all phase: Runs the stateless pipeline against compiled contexts.

        Args:
            pipeline_definition: The stateless list of AbstractStep objects.
            compiled_contexts: Dict of axis_id to its compiled, frozen ProcessingContext.
                               Obtained from `compile_plate_for_processing`.
            max_workers: Maximum number of worker threads for parallel execution.
            visualizer: Optional instance of NapariStreamVisualizer for real-time visualization
                        (requires napari to be installed; must be initialized with orchestrator's filemanager by the caller).
            log_file_base: Base path for worker process log files (without extension).
                          Each worker will create its own log file: {log_file_base}_worker_{pid}.log

        Returns:
            A dictionary mapping well IDs to their execution status (success/error and details).
        """

        # CRITICAL FIX: Use resolved pipeline definition from compilation if available
        # For subprocess runner, use the parameter directly since it receives pre-compiled contexts
        resolved_pipeline = self.__dict__.get("_resolved_pipeline_definition")
        if resolved_pipeline is not None:
            pipeline_definition = resolved_pipeline

        if not self.is_initialized():
            raise RuntimeError("Orchestrator must be initialized before executing.")
        if not pipeline_definition:
            raise ValueError(
                "A valid (stateless) pipeline definition must be provided."
            )
        if not compiled_contexts:
            logger.warning("No compiled contexts provided for execution.")
            return {}
        if progress_queue is None:
            raise ValueError(
                "progress_queue is required for execute_compiled_plate invariant path"
            )
        if progress_context is None:
            raise ValueError(
                "progress_context is required for execute_compiled_plate invariant path"
            )
        execution_id = progress_context["execution_id"]
        plate_id = progress_context["plate_id"]

        # Access num_workers from effective config (merged pipeline + global config)
        actual_max_workers = max_workers or self.get_effective_config().num_workers
        if actual_max_workers <= 0:  # Ensure positive number of workers
            actual_max_workers = 1

        # 🔬 AUTOMATIC VISUALIZER CREATION: Create visualizers if compiler detected streaming
        visualizers = []
        if visualizer is None:
            from openhcs.core.config import StreamingConfig

            # Collect unique configs (deduplicate by viewer_type + port)
            unique_configs = {}
            for ctx in compiled_contexts.values():
                for visualizer_info in ctx.required_visualizers:
                    config = visualizer_info["config"]
                    key = (
                        (config.viewer_type, config.port)
                        if isinstance(config, StreamingConfig)
                        else (config.backend.name,)
                    )
                    if key not in unique_configs:
                        unique_configs[key] = (config, ctx.visualizer_config)

            # Create visualizers — emit progress for each launch
            for key, (config, vis_config) in unique_configs.items():
                viewer_name = key[0] if key else type(config).__name__
                viewer_port = key[1] if len(key) > 1 else None
                port_info = f" on port {viewer_port}" if viewer_port else ""
                progress_queue.put(
                    create_event(
                        execution_id=execution_id,
                        plate_id=plate_id,
                        axis_id="",
                        step_name="",
                        phase=ProgressPhase.INIT,
                        status=ProgressStatus.STARTED,
                        percent=0.0,
                        message=f"Launching {viewer_name} viewer{port_info}",
                    ).to_dict()
                )
                visualizers.append(self.get_or_create_visualizer(config, vis_config))

            # Wait for all streaming viewers to be ready before starting pipeline
            # This ensures viewers are available to receive images
            if visualizers:
                import time

                max_wait = 30.0  # Maximum wait time in seconds
                start_time = time.time()

                while time.time() - start_time < max_wait:
                    # Check if execution was cancelled
                    if self._cancelled:
                        raise RuntimeError("Execution cancelled by user")

                    all_ready = all(v.is_running for v in visualizers)
                    if all_ready:
                        progress_queue.put(
                            create_event(
                                execution_id=execution_id,
                                plate_id=plate_id,
                                axis_id="",
                                step_name="",
                                phase=ProgressPhase.INIT,
                                status=ProgressStatus.RUNNING,
                                percent=0.0,
                                message="All streaming viewers ready",
                            ).to_dict()
                        )
                        break
                    time.sleep(0.2)  # Check every 200ms
                else:
                    # Timeout - log which viewers aren't ready (use generic port attribute)
                    not_ready = [v.port for v in visualizers if not v.is_running]
                    logger.warning(
                        f"🔬 ORCHESTRATOR: Timeout waiting for streaming viewers. Not ready: {not_ready}"
                    )
                    progress_queue.put(
                        create_event(
                            execution_id=execution_id,
                            plate_id=plate_id,
                            axis_id="",
                            step_name="",
                            phase=ProgressPhase.INIT,
                            status=ProgressStatus.RUNNING,
                            percent=0.0,
                            message=f"Timeout waiting for streaming viewers. Not ready: {not_ready}",
                        ).to_dict()
                    )

                # Clear viewer state for new pipeline run to prevent accumulation
                for vis in visualizers:
                    if hasattr(vis, "clear_viewer_state"):
                        success = vis.clear_viewer_state()
                        if not success:
                            logger.warning(
                                f"🔬 ORCHESTRATOR: Failed to clear state for viewer on port {vis.port}"
                            )

            # For backwards compatibility, set visualizer to the first one
            visualizer = visualizers[0] if visualizers else None

        set_progress_queue(progress_queue)
        try:
            # Reset cancellation flag at start of execution
            self._cancelled = False

            self._state = OrchestratorState.EXECUTING
            logger.info(
                f"Starting execution for {len(compiled_contexts)} axis values with max_workers={actual_max_workers}."
            )

            execution_results: Dict[str, ExecutionResult] = {}

            # CUDA COMPATIBILITY: Set spawn method for multiprocessing to support CUDA
            try:
                # Check if spawn method is available and set it if not already set
                current_method = multiprocessing.get_start_method(allow_none=True)
                if current_method != "spawn":
                    multiprocessing.set_start_method("spawn", force=True)
            except RuntimeError as e:
                # Start method may already be set, which is fine
                pass

            # Choose executor type based on effective config for debugging support
            effective_config = self.get_effective_config()
            executor_type = (
                "ThreadPoolExecutor"
                if effective_config.use_threading
                else "ProcessPoolExecutor"
            )

            # DEATH DETECTION: Mark executor creation

            # Choose appropriate executor class and configure worker logging
            if effective_config.use_threading:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=actual_max_workers
                )
            else:
                # CRITICAL FIX: Use _configure_worker_with_gpu to ensure workers have function registry
                # Workers need the function registry to access decorated functions with memory types
                global_config = get_current_global_config(GlobalPipelineConfig)
                global_config_dict = global_config.__dict__ if global_config else {}

                if log_file_base:
                    executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=actual_max_workers,
                        initializer=_configure_worker_with_gpu,
                        initargs=(
                            log_file_base,
                            global_config_dict,
                            progress_queue,
                            progress_context,
                        ),
                    )
                else:
                    executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=actual_max_workers,
                        initializer=_configure_worker_with_gpu,
                        initargs=(
                            "",
                            global_config_dict,
                            progress_queue,
                            progress_context,
                        ),  # Empty string for no logging
                    )

            # Store executor for cancellation support
            self._executor = executor

            # Wrap executor context in try/except to handle BrokenProcessPool during exit
            # This can happen if workers are killed externally (e.g., during cancellation)
            try:
                with executor:
                    # NUCLEAR ERROR TRACING: Create snapshot of compiled_contexts to prevent iteration issues
                    contexts_snapshot = dict(compiled_contexts.items())

                    # CRITICAL FIX: Resolve all lazy dataclass instances before multiprocessing
                    # This ensures that the contexts are safe for pickling in ProcessPoolExecutor
                    # Note: Don't resolve pipeline_definition as it may overwrite collision-resolved configs
                    contexts_snapshot = resolve_lazy_configurations_for_serialization(
                        contexts_snapshot
                    )

                    future_to_worker_slot = {}
                    config = get_openhcs_config()
                    if not config:
                        raise RuntimeError(
                            "Component configuration is required for orchestrator execution"
                        )
                    axis_name = config.multiprocessing_axis.value

                    # Group contexts by axis to detect sequential combinations
                    from collections import defaultdict

                    contexts_by_axis = defaultdict(list)
                    for context_key, context in contexts_snapshot.items():
                        # Extract axis_id from context_key (either "axis_id" or "axis_id__combo_N")
                        if "__combo_" in context_key:
                            axis_id = context_key.split("__combo_")[0]
                            contexts_by_axis[axis_id].append((context_key, context))
                        else:
                            contexts_by_axis[context_key].append((context_key, context))

                    if worker_assignments is None:
                        generated: Dict[str, List[str]] = {
                            f"worker_{idx}": [] for idx in range(actual_max_workers)
                        }
                        for idx, axis_id in enumerate(sorted(contexts_by_axis.keys())):
                            generated[f"worker_{idx % actual_max_workers}"].append(
                                axis_id
                            )
                        worker_assignments = {
                            worker_slot: owned
                            for worker_slot, owned in generated.items()
                            if owned
                        }

                    # Validate worker ownership map against compiled axis ids.
                    expected_axis_ids = set(contexts_by_axis.keys())
                    all_assigned_axis_ids: list[str] = []
                    for owned in worker_assignments.values():
                        all_assigned_axis_ids.extend(owned)
                    if len(all_assigned_axis_ids) != len(set(all_assigned_axis_ids)):
                        raise RuntimeError(
                            f"Duplicate axis ownership detected in worker_assignments: {worker_assignments}"
                        )
                    if set(all_assigned_axis_ids) != expected_axis_ids:
                        raise RuntimeError(
                            f"worker_assignments mismatch. expected={sorted(expected_axis_ids)}, got={sorted(all_assigned_axis_ids)}"
                        )

                    axis_to_worker: Dict[str, str] = {}
                    for worker_slot, owned in worker_assignments.items():
                        for axis_id in owned:
                            axis_to_worker[axis_id] = worker_slot

                    lane_axis_contexts: Dict[str, List[tuple[str, List[tuple]]]] = {
                        worker_slot: [] for worker_slot in worker_assignments.keys()
                    }

                    for axis_id, axis_contexts in contexts_by_axis.items():
                        try:
                            # Resolve all contexts for this axis
                            resolved_axis_contexts = [
                                (
                                    context_key,
                                    resolve_lazy_configurations_for_serialization(
                                        context
                                    ),
                                )
                                for context_key, context in axis_contexts
                            ]

                            worker_slot = axis_to_worker[axis_id]
                            lane_axis_contexts[worker_slot].append(
                                (axis_id, resolved_axis_contexts)
                            )
                        except Exception as submit_error:
                            error_msg = f"🔥 ORCHESTRATOR ERROR: Failed to submit task for {axis_name} {axis_id}: {submit_error}"
                            logger.error(error_msg, exc_info=True)
                            # FAIL-FAST: Re-raise task submission errors immediately
                            raise

                    for worker_slot, lane_contexts in lane_axis_contexts.items():
                        if not lane_contexts:
                            continue
                        owned_wells = list(worker_assignments[worker_slot])

                        try:
                            future = executor.submit(
                                _execute_worker_lane_static,
                                pipeline_definition,
                                lane_contexts,
                                None,  # visualizer
                                execution_id,
                                plate_id,
                                worker_slot,
                                owned_wells,
                            )
                            future_to_worker_slot[future] = (worker_slot, owned_wells)
                        except Exception as submit_error:
                            error_msg = f"🔥 ORCHESTRATOR ERROR: Failed to submit lane {worker_slot}: {submit_error}"
                            logger.error(error_msg, exc_info=True)
                            raise

                    for future in concurrent.futures.as_completed(
                        future_to_worker_slot
                    ):
                        worker_slot, owned_wells = future_to_worker_slot[future]

                        try:
                            lane_results = future.result()
                            execution_results.update(lane_results)
                        except Exception as exc:
                            import traceback

                            full_traceback = traceback.format_exc()
                            error_msg = f"Worker lane {worker_slot} generated an exception during execution: {exc}"
                            logger.error(
                                f"🔥 ORCHESTRATOR ERROR: {error_msg}", exc_info=True
                            )
                            logger.error(
                                f"🔥 ORCHESTRATOR FULL TRACEBACK for worker lane {worker_slot}:\n{full_traceback}"
                            )
                            failing_axis_id = owned_wells[0] if owned_wells else ""

                            emit(
                                execution_id=execution_id,
                                plate_id=plate_id,
                                axis_id=failing_axis_id,
                                step_name="pipeline",
                                phase=ProgressPhase.AXIS_ERROR,
                                status=ProgressStatus.ERROR,
                                completed=0,
                                total=len(pipeline_definition),
                                percent=0.0,
                                error=str(exc),
                                traceback=full_traceback,
                                worker_slot=worker_slot,
                                owned_wells=owned_wells,
                            )

                            # FAIL-FAST: Re-raise immediately instead of storing error
                            raise

                    # Explicitly shutdown executor INSIDE the with block to avoid hang on context exit
                    # Handle BrokenProcessPool in case workers were killed externally (e.g., during cancellation)
                    try:
                        executor.shutdown(wait=True, cancel_futures=False)
                    except concurrent.futures.process.BrokenProcessPool as e:
                        logger.warning(
                            f"🔥 ORCHESTRATOR: Executor shutdown failed due to broken process pool (workers were killed externally): {e}"
                        )
                        # Don't wait for broken workers - they're already dead
                        # The with block exit will handle cleanup
                    except Exception as e:
                        logger.warning(
                            f"🔥 ORCHESTRATOR: Executor shutdown failed: {e}"
                        )

            except concurrent.futures.process.BrokenProcessPool as e:
                # Workers were killed externally (e.g., during cancellation)
                # This is expected behavior when cancellation happens
                logger.warning(
                    f"🔥 ORCHESTRATOR: Executor context exit failed due to broken process pool (workers were killed externally): {e}"
                )
                # Continue execution - the workers are already dead, no cleanup needed

            # Determine if we're using multiprocessing (ProcessPoolExecutor) or threading
            effective_config = self.get_effective_config()
            use_multiprocessing = not effective_config.use_threading

            # 🔥 GPU CLEANUP: Skip in multiprocessing mode - workers handle their own cleanup
            # In multiprocessing mode, GPU cleanup in the main process can hang because
            # GPU contexts are owned by worker processes, not the orchestrator process
            try:
                if cleanup_all_gpu_frameworks and not use_multiprocessing:
                    cleanup_all_gpu_frameworks()
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to cleanup GPU memory after plate execution: {cleanup_error}"
                )

            # Run automatic analysis consolidation if enabled
            # Get the consolidation config from the first compiled context (captured at compile time)
            first_context = next(iter(compiled_contexts.values()))
            analysis_consolidation_config = getattr(
                first_context, "analysis_consolidation_config", None
            )

            # Debug logging for consolidation troubleshooting
            if not analysis_consolidation_config.enabled:
                logger.info("⏭️ CONSOLIDATION: Disabled")
            else:
                try:
                    # Collect all unique results directories from step plans
                    results_dirs = set()
                    for context in compiled_contexts.values():
                        for step_plan in context.step_plans.values():
                            if "analysis_results_dir" in step_plan:
                                results_dirs.add(
                                    Path(step_plan["analysis_results_dir"])
                                )
                            if "materialized_analysis_results_dir" in step_plan:
                                results_dirs.add(
                                    Path(step_plan["materialized_analysis_results_dir"])
                                )

                    if results_dirs:
                        successful_dirs, failed_dirs = consolidate_results_directories(
                            results_dirs=list(results_dirs),
                            plate_path=Path(first_context.plate_path),
                            analysis_consolidation_config=analysis_consolidation_config,
                            plate_metadata_config=first_context.plate_metadata_config,
                            filename_parser=self.microscope_handler.parser,
                        )

                        if successful_dirs:
                            logger.info(
                                f"✅ CONSOLIDATION: {len(successful_dirs)} directories consolidated"
                            )
                        if failed_dirs:
                            logger.warning(
                                f"⚠️ CONSOLIDATION: {len(failed_dirs)} directories failed"
                            )
                except Exception as e:
                    logger.error(
                        f"❌ CONSOLIDATION: Failed with error: {e}", exc_info=True
                    )

            # Update state based on execution results
            if all(result.is_success() for result in execution_results.values()):
                self._state = OrchestratorState.COMPLETED
            else:
                self._state = OrchestratorState.EXEC_FAILED

            # 🔬 VISUALIZER CLEANUP: Stop all visualizers if they were auto-created and not persistent
            for idx, vis in enumerate(visualizers):
                try:
                    if not vis.persistent:
                        vis.stop_viewer()
                except Exception as e:
                    logger.warning(
                        f"🔬 ORCHESTRATOR: Failed to cleanup visualizer {idx + 1}: {e}"
                    )

            return execution_results
        except Exception as e:
            self._state = OrchestratorState.EXEC_FAILED
            logger.error(f"Failed to execute compiled plate: {e}")
            raise
        finally:
            set_progress_queue(None)

    def get_component_keys(
        self,
        component: Union["AllComponents", "VariableComponents"],
        component_filter: Optional[List[Union[str, int]]] = None,
    ) -> List[str]:
        """
        Generic method to get component keys using VariableComponents directly.

        Returns the discovered component values as strings to match the pattern
        detection system format.

        Tries metadata cache first, falls back to filename parsing cache if metadata is empty.

        Args:
            component: AllComponents or VariableComponents enum specifying which component to extract
                      (also accepts GroupBy enum which will be converted to AllComponents)
            component_filter: Optional list of component values to filter by

        Returns:
            List of component values as strings, sorted

        Raises:
            RuntimeError: If orchestrator is not initialized
        """
        if not self.is_initialized():
            raise RuntimeError(
                "Orchestrator must be initialized before getting component keys."
            )

        # Convert GroupBy to AllComponents using OpenHCS generic utility
        if isinstance(component, GroupBy) and component.value is None:
            raise ValueError("Cannot get component keys for GroupBy.NONE")

        # Convert to AllComponents for cache lookup (includes multiprocessing axis)
        component = convert_enum_by_value(component, AllComponents) or component

        # Use component directly - let natural errors occur for wrong types
        component_name = component.value

        # Try metadata cache first (preferred source)
        cached_metadata = self._metadata_cache_service.get_cached_metadata(component)
        if cached_metadata:
            all_components = list(cached_metadata.keys())
            logger.debug(
                f"Using metadata cache for {component_name}: {len(all_components)} components"
            )
        else:
            # Fall back to filename parsing cache
            all_components = self._component_keys_cache[
                component
            ]  # Let KeyError bubble up naturally

            if not all_components:
                logger.warning(
                    f"No {component_name} values found in input directory: {self.input_dir}"
                )
                return []

            logger.debug(
                f"Using filename parsing cache for {component.value}: {len(all_components)} components"
            )

        if component_filter:
            str_component_filter = {str(c) for c in component_filter}
            selected_components = [
                comp for comp in all_components if comp in str_component_filter
            ]
            if not selected_components:
                component_name = group_by.value
                logger.warning(
                    f"No {component_name} values from {all_components} match the filter: {component_filter}"
                )
            return selected_components
        else:
            return all_components

    def cache_component_keys(
        self, components: Optional[List["AllComponents"]] = None
    ) -> None:
        """
        Pre-compute and cache component keys for fast access using single-pass parsing.

        This method performs expensive file listing and parsing operations once,
        extracting all component types in a single pass for maximum efficiency.

        Args:
            components: Optional list of AllComponents to cache.
                       If None, caches all components in the AllComponents enum.
        """
        if not self.is_initialized():
            raise RuntimeError(
                "Orchestrator must be initialized before caching component keys."
            )

        if components is None:
            components = list(
                AllComponents
            )  # Cache all enum values including multiprocessing axis

        logger.info(
            f"Caching component keys for: {[comp.value for comp in components]}"
        )

        # Initialize component sets for all requested components
        component_sets: Dict["AllComponents", Set[Union[str, int]]] = {}
        for component in components:
            component_sets[component] = set()

        # Single pass through all filenames - extract all components at once
        try:
            # Use primary backend from microscope handler
            backend_to_use = self.microscope_handler.get_primary_backend(
                self.input_dir, self.filemanager
            )
            logger.info(
                "Component key discovery: input_dir=%s backend_to_use=%s microscope=%s parser=%s",
                self.input_dir,
                backend_to_use,
                getattr(self.microscope_handler, "microscope_type", None),
                getattr(
                    getattr(self.microscope_handler, "parser", None),
                    "__class__",
                    type(None),
                ).__name__,
            )

            filenames = self.filemanager.list_files(
                str(self.input_dir), backend_to_use, extensions=DEFAULT_IMAGE_EXTENSIONS
            )
            logger.info(
                "Component key discovery: listed %d files (extensions=%s)",
                len(filenames),
                DEFAULT_IMAGE_EXTENSIONS,
            )
            if filenames:
                preview = [str(p) for p in filenames[:10]]
                logger.debug(
                    "Component key discovery: first %d files: %s",
                    len(preview),
                    preview,
                )

            for filename in filenames:
                parsed_info = self.microscope_handler.parser.parse_filename(
                    str(filename)
                )
                if parsed_info:
                    # Extract all requested components from this filename
                    for component in component_sets:
                        component_name = component.value
                        if (
                            component_name in parsed_info
                            and parsed_info[component_name] is not None
                        ):
                            component_sets[component].add(parsed_info[component_name])
                else:
                    logger.warning(
                        "Could not parse filename: %s (backend=%s input_dir=%s)",
                        filename,
                        backend_to_use,
                        self.input_dir,
                    )

        except Exception as e:
            logger.error(
                f"Error listing files or parsing filenames from {self.input_dir}: {e}",
                exc_info=True,
            )
            # Initialize empty sets for failed parsing
            for component in component_sets:
                component_sets[component] = set()

        # Convert sets to sorted lists and store in cache
        for component, component_set in component_sets.items():
            sorted_components = [str(comp) for comp in sorted(list(component_set))]
            self._component_keys_cache[component] = sorted_components
            logger.debug(f"Cached {len(sorted_components)} {component.value} keys")

            if not sorted_components:
                logger.warning(
                    f"No {component.value} values found in input directory: {self.input_dir}"
                )

        logger.info(
            f"Component key caching complete. Cached {len(component_sets)} component types in single pass."
        )

    def clear_component_cache(
        self, components: Optional[List["AllComponents"]] = None
    ) -> None:
        """
        Clear cached component keys to force recomputation.

        Use this when the input directory contents have changed and you need
        to refresh the component key cache.

        Args:
            components: Optional list of AllComponents to clear from cache.
                       If None, clears entire cache.
        """
        if components is None:
            self._component_keys_cache.clear()
            logger.info("Cleared entire component keys cache")
        else:
            for component in components:
                if component in self._component_keys_cache:
                    del self._component_keys_cache[component]
                    logger.debug(f"Cleared cache for {component.value}")
            logger.info(f"Cleared cache for {len(components)} component types")

    @property
    def metadata_cache(self) -> MetadataCache:
        """Access to metadata cache service."""
        return self._metadata_cache_service

    # Global config management removed - handled by UI layer

    @property
    def pipeline_config(self) -> Optional["PipelineConfig"]:
        """Get current pipeline configuration."""
        return self._pipeline_config

    @pipeline_config.setter
    def pipeline_config(self, value: Optional["PipelineConfig"]) -> None:
        """Set pipeline configuration with auto-sync to thread-local context."""
        self._pipeline_config = value
        # CRITICAL FIX: Also update public attribute for dual-axis resolver discovery
        # This ensures the resolver can always find the current pipeline config
        if hasattr(self, "__dict__"):  # Avoid issues during __init__
            self.__dict__["pipeline_config"] = value
        if self._auto_sync_enabled and value is not None:
            self._sync_to_thread_local()

    def _sync_to_thread_local(self) -> None:
        """Internal method to sync current pipeline_config to thread-local context."""
        if self._pipeline_config and hasattr(self, "plate_path"):
            self.apply_pipeline_config(self._pipeline_config)

    def apply_pipeline_config(self, pipeline_config: "PipelineConfig") -> None:
        """
        Apply per-orchestrator configuration using thread-local storage.

        This method sets the orchestrator's effective config in thread-local storage
        for step-level lazy configurations to resolve against.
        """
        # Import PipelineConfig at runtime for isinstance check
        from openhcs.core.config import PipelineConfig

        if not isinstance(pipeline_config, PipelineConfig):
            raise TypeError(f"Expected PipelineConfig, got {type(pipeline_config)}")

        # Temporarily disable auto-sync to prevent recursion
        self._auto_sync_enabled = False
        try:
            self._pipeline_config = pipeline_config
        finally:
            self._auto_sync_enabled = True

        # CRITICAL FIX: Do NOT contaminate thread-local context during PipelineConfig editing
        # The orchestrator should maintain its own internal context without modifying
        # the global thread-local context. This prevents reset operations from showing
        # orchestrator's saved values instead of original thread-local defaults.
        #
        # The merged config is computed internally and used by get_effective_config()
        # but should NOT be set as the global thread-local context.

        logger.info(f"Applied orchestrator config for plate: {self.plate_path}")

    def get_effective_config(
        self, *, for_serialization: bool = False
    ) -> GlobalPipelineConfig:
        """
        Get effective configuration for this orchestrator.

        Args:
            for_serialization: If True, resolves all values for pickling/storage.
                              If False, preserves None values for sibling inheritance.
        """

        if for_serialization:
            result = self.pipeline_config.to_base_config()
            return result
        else:
            # Reuse existing merged config logic from apply_pipeline_config
            shared_context = get_current_global_config(GlobalPipelineConfig)
            if not shared_context:
                raise RuntimeError(
                    "No global configuration context available for merging"
                )

            result = _create_merged_config(self.pipeline_config, shared_context)
            return result

    def clear_pipeline_config(self) -> None:
        """Clear per-orchestrator configuration."""
        # REMOVED: Thread-local modification - dual-axis resolver handles context automatically
        # No need to modify thread-local storage when clearing orchestrator config
        self.pipeline_config = None
        # Clear metadata cache for this orchestrator
        if hasattr(self, "_metadata_cache_service") and self._metadata_cache_service:
            self._metadata_cache_service.clear_cache()
        logger.info(f"Cleared per-orchestrator config for plate: {self.plate_path}")

    def cleanup_pipeline_config(self) -> None:
        """Clean up orchestrator context when done (for backward compatibility)."""
        self.clear_pipeline_config()

    def __del__(self):
        """Ensure config cleanup on orchestrator destruction."""
        try:
            # Clear any stored configuration references
            self.clear_pipeline_config()
        except Exception:
            # Ignore errors during cleanup in destructor to prevent cascading failures
            pass
