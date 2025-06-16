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
import threading
import concurrent.futures
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Set

from openhcs.constants.constants import Backend, DEFAULT_WORKSPACE_DIR_SUFFIX, DEFAULT_IMAGE_EXTENSIONS # DEFAULT_NUM_WORKERS removed
from openhcs.constants import Microscope
from openhcs.core.config import GlobalPipelineConfig, get_default_global_config
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.pipeline.compiler import PipelineCompiler
from openhcs.core.pipeline.step_attribute_stripper import StepAttributeStripper
from openhcs.core.steps.abstract import AbstractStep, get_step_id
from openhcs.io.exceptions import StorageWriteError
from openhcs.io.filemanager import FileManager
from openhcs.io.base import storage_registry
from openhcs.microscopes.microscope_interfaces import (
    MicroscopeHandler, create_microscope_handler)
from openhcs.runtime.napari_stream_visualizer import NapariStreamVisualizer


logger = logging.getLogger(__name__)


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
        self._lock = threading.RLock()
        
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

        self.plate_path = plate_path
        self.workspace_path = workspace_path

        if self.plate_path is None and self.workspace_path is None:
            raise ValueError("Either plate_path or workspace_path must be provided for PipelineOrchestrator.")

        if storage_registry:
            self.registry = storage_registry
            logger.info("PipelineOrchestrator using provided StorageRegistry instance.")
        else:
            from openhcs.io.base import storage_registry as global_registry
            self.registry = global_registry  # Use the global singleton registry
            logger.info("PipelineOrchestrator created its own StorageRegistry instance (global singleton).")
        
        # Orchestrator always creates its own FileManager, using the determined registry
        self.filemanager = FileManager(self.registry)
        self.input_dir: Optional[Path] = None
        self.microscope_handler: Optional[MicroscopeHandler] = None
        self.default_pipeline_definition: Optional[List[AbstractStep]] = None
        self._initialized: bool = False

    def initialize_workspace(self, workspace_path: Optional[Union[str, Path]] = None):
        """Initializes workspace path and mirrors plate directory if needed."""
        if workspace_path:
            self.workspace_path = Path(workspace_path) if isinstance(workspace_path, str) else workspace_path

        if self.workspace_path is None and self.plate_path:
            # Check if global output folder is configured
            global_output_folder = self.global_config.path_planning.global_output_folder
            if global_output_folder:
                # Use global output folder: {global_folder}/{plate_name}_workspace
                global_folder = Path(global_output_folder)
                plate_name = self.plate_path.name
                self.workspace_path = global_folder / f"{plate_name}{DEFAULT_WORKSPACE_DIR_SUFFIX}"
            else:
                # Use current behavior: {plate_path}_workspace
                self.workspace_path = Path(str(self.plate_path) + DEFAULT_WORKSPACE_DIR_SUFFIX)
        elif self.workspace_path is None:
             raise ValueError("Cannot initialize workspace without either plate_path or a specified workspace_path.")

        self.filemanager.ensure_directory(str(self.workspace_path), Backend.DISK.value)

#        if self.microscope_handler is not None and self.input_dir is not None:
#            logger.debug("Workspace already initialized.")
#            return


        if self.plate_path and self.workspace_path:
            logger.info(f"Mirroring plate directory {self.plate_path} to workspace {self.workspace_path}...")
            try:
                num_links = self.filemanager.mirror_directory_with_symlinks(
                    source_dir=str(self.plate_path),
                    target_dir=str(self.workspace_path),
                    backend=Backend.DISK.value,
                    recursive=True,
                    overwrite_symlinks_only=True,
                )
                logger.info(f"Created {num_links} symlinks in workspace.")
                self.input_dir = Path(self.workspace_path)
            except Exception as e:
                error_msg = f"Failed to mirror plate directory to workspace: {e}"
                logger.error(error_msg)
                raise StorageWriteError(error_msg) from e
        elif self.plate_path:
            self.input_dir = self.plate_path
            logger.info(f"Using plate path as input directory: {self.input_dir}")
        elif self.workspace_path:
            self.input_dir = self.workspace_path
            logger.info(f"Using workspace path as input directory: {self.input_dir}")
        else:
            raise RuntimeError("Cannot determine input_dir due to missing plate_path and workspace_path.")

        #set microscope handler's plate_folder to workspace path
        try:
            self.microscope_handler.plate_folder = self.workspace_path
        except Exception as e:
            error_msg = f"Failed to set plate_folder on microscope handler: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

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

        self.initialize_microscope_handler()
        self.initialize_workspace(workspace_path)

        # Process workspace with microscope-specific logic
        logger.info("Processing workspace with microscope handler...")
        actual_image_dir = self.microscope_handler.post_workspace(self.workspace_path, self.filemanager)

        # Use the actual image directory returned by the microscope handler
        self.input_dir = Path(actual_image_dir)
        logger.info(f"Set input directory to: {self.input_dir}")

        self._initialized = True
        logger.info("PipelineOrchestrator fully initialized.")
        return self

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
        context.orchestrator = self
        context.microscope_handler = self.microscope_handler
        context.input_dir = self.input_dir
        context.workspace_path = self.workspace_path
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

        compiled_contexts: Dict[str, ProcessingContext] = {}
        wells_to_process = self.get_wells(well_filter)

        if not wells_to_process:
            logger.warning("No wells found to process based on filter.")
            return {}

        logger.info(f"Starting compilation for wells: {', '.join(wells_to_process)}")

        for well_id in wells_to_process:
            logger.debug(f"Compiling for well: {well_id}")
            context = self.create_context(well_id)
            
            PipelineCompiler.initialize_step_plans_for_context(context, pipeline_definition)
            PipelineCompiler.plan_materialization_flags_for_context(context, pipeline_definition)
            PipelineCompiler.validate_memory_contracts_for_context(context, pipeline_definition)
            PipelineCompiler.assign_gpu_resources_for_context(context)

            if enable_visualizer_override:
                PipelineCompiler.apply_global_visualizer_override_for_context(context, True)
            
            context.freeze()
            compiled_contexts[well_id] = context
            logger.debug(f"Compilation finished for well: {well_id}")

        logger.info("Stripping attributes from pipeline definition steps.")
        StepAttributeStripper.strip_step_attributes(pipeline_definition, {}) 

        logger.info(f"Plate compilation finished for {len(compiled_contexts)} wells.")
        return compiled_contexts

    def _execute_single_well(
        self,
        pipeline_definition: List[AbstractStep],
        frozen_context: ProcessingContext,
        visualizer: Optional[NapariStreamVisualizer]
    ) -> Dict[str, Any]:
        """Executes the pipeline for a single well using its frozen context."""
        well_id = frozen_context.well_id
        logger.info(f"Executing pipeline for well {well_id}")
        if not frozen_context.is_frozen():
            logger.error(f"Attempted to execute with a non-frozen context for well {well_id}.")
            raise RuntimeError(f"Context for well {well_id} is not frozen before execution.")

        try:
            for step in pipeline_definition:
                # Generate step_id from object reference (elegant stateless approach)
                step_id = get_step_id(step)
                step_name = getattr(step, 'name', 'N/A') if hasattr(step, 'name') else 'N/A'

                logger.info(f"Executing step {step_id} ({step_name}) for well {well_id}")
                step.process(frozen_context)

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
            
            logger.info(f"Pipeline execution completed successfully for well {well_id}")
            return {"status": "success", "well_id": well_id}
        except Exception as e:
            logger.error(f"Error during pipeline execution for well {well_id}: {e}", exc_info=True)
            return {"status": "error", "well_id": well_id, "error_message": str(e), "details": repr(e)}

    def execute_compiled_plate(
        self,
        pipeline_definition: List[AbstractStep],
        compiled_contexts: Dict[str, ProcessingContext],
        max_workers: Optional[int] = None, # Changed from DEFAULT_NUM_WORKERS
        visualizer: Optional[NapariStreamVisualizer] = None,
        executor_callback: Optional[Callable[[Any], None]] = None,
        cancellation_flag: Optional[Any] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute-all phase: Runs the stateless pipeline against compiled contexts.

        Args:
            pipeline_definition: The stateless list of AbstractStep objects.
            compiled_contexts: Dict of well_id to its compiled, frozen ProcessingContext.
                               Obtained from `compile_plate_for_processing`.
            max_workers: Maximum number of worker threads for parallel execution.
            visualizer: Optional instance of NapariStreamVisualizer (must be
                        initialized with orchestrator's filemanager by the caller).
            executor_callback: Optional callback to receive the ThreadPoolExecutor reference
                              for cancellation purposes. Called with executor when parallel execution starts.

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

        logger.info(f"Starting execution for {len(compiled_contexts)} wells with max_workers={actual_max_workers}.")

        # 🔍 VRAM TRACKING: Log initial memory state
        try:
            from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
            log_gpu_memory_usage("plate execution start")
        except Exception:
            pass

        execution_results: Dict[str, Dict[str, Any]] = {}

        if actual_max_workers > 1 and len(compiled_contexts) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
                # Provide executor reference to caller for cancellation
                if executor_callback:
                    executor_callback(executor)

                future_to_well_id = {
                    executor.submit(self._execute_single_well, pipeline_definition, context, visualizer): well_id
                    for well_id, context in compiled_contexts.items()
                }
                for future in concurrent.futures.as_completed(future_to_well_id):
                    well_id = future_to_well_id[future]
                    try:
                        result = future.result()
                        execution_results[well_id] = result
                    except Exception as exc:
                        logger.error(f"Well {well_id} generated an exception during parallel execution: {exc}", exc_info=True)
                        execution_results[well_id] = {"status": "error", "well_id": well_id, "error_message": str(exc), "details": repr(exc)}

                    # 🔥 GPU CLEANUP: Clear GPU memory after each well completion (parallel execution)
                    try:
                        from openhcs.core.memory.gpu_cleanup import cleanup_all_gpu_frameworks
                        cleanup_all_gpu_frameworks()
                        logger.debug(f"🔥 GPU CLEANUP: Cleared all GPU frameworks after well {well_id} (parallel)")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup GPU memory after well {well_id} (parallel): {cleanup_error}")
        else:
            logger.info("Executing wells sequentially.")
            for well_id, context in compiled_contexts.items():
                # 🔍 VRAM TRACKING: Log memory before well execution
                try:
                    from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
                    log_gpu_memory_usage(f"before well {well_id}")
                except Exception:
                    pass

                execution_results[well_id] = self._execute_single_well(pipeline_definition, context, visualizer)

                # 🔍 VRAM TRACKING: Log memory after well execution (before cleanup)
                try:
                    from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
                    log_gpu_memory_usage(f"after well {well_id} (before cleanup)")
                except Exception:
                    pass

                # 🔥 GPU CLEANUP: Clear GPU memory after each well to prevent accumulation
                try:
                    from openhcs.core.memory.gpu_cleanup import cleanup_all_gpu_frameworks
                    cleanup_all_gpu_frameworks()
                    logger.debug(f"🔥 GPU CLEANUP: Cleared all GPU frameworks after well {well_id}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup GPU memory after well {well_id}: {cleanup_error}")

                # 🔍 VRAM TRACKING: Log memory after cleanup
                try:
                    from openhcs.core.memory.gpu_cleanup import log_gpu_memory_usage
                    log_gpu_memory_usage(f"after well {well_id} (after cleanup)")
                except Exception:
                    pass

                # 🔥 MEMORY BACKEND CLEANUP: Clear memory backend after each well to prevent accumulation
                try:
                    from openhcs.io.base import reset_memory_backend
                    reset_memory_backend()
                    logger.debug(f"🔥 MEMORY BACKEND CLEANUP: Reset memory backend after well {well_id}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to reset memory backend after well {well_id}: {cleanup_error}")

        logger.info(f"Plate execution finished. Results: {execution_results}")
        return execution_results

    def get_wells(self, well_filter: Optional[List[str]] = None) -> List[str]:
        """
        Get the wells to process based on the filter.
        """
        if not self.is_initialized() or self.input_dir is None or self.microscope_handler is None:
            raise RuntimeError("Orchestrator must be initialized with input_dir and microscope_handler to get wells.")

        all_wells_set: Set[str] = set()
        try:
            filenames = self.filemanager.list_files(str(self.input_dir), Backend.DISK.value, extensions=DEFAULT_IMAGE_EXTENSIONS)
            for filename in filenames:
                parsed_info = self.microscope_handler.parser.parse_filename(str(filename))
                if parsed_info and 'well' in parsed_info:
                    all_wells_set.add(parsed_info['well'])
                else:
                    logger.warning(f"Could not parse well information from filename: {filename}")
        except Exception as e:
            logger.error(f"Error listing files or parsing well names from {self.input_dir}: {e}", exc_info=True)
            return []

        all_wells = sorted(list(all_wells_set))

        if not all_wells:
            logger.warning(f"No wells found in input directory: {self.input_dir} with extensions {DEFAULT_IMAGE_EXTENSIONS}")
            return []

        if well_filter:
            str_well_filter = {str(w) for w in well_filter}
            selected_wells = [well for well in all_wells if well in str_well_filter]
            if not selected_wells:
                logger.warning(f"No wells from {all_wells} match the filter: {well_filter}")
            return selected_wells
        else:
            return all_wells

    def get_channels(self, channel_filter: Optional[List[Union[str, int]]] = None) -> List[int]:
        """
        Get the channels to process based on the filter.
        """
        if not self.is_initialized() or self.input_dir is None or self.microscope_handler is None:
            raise RuntimeError("Orchestrator must be initialized with input_dir and microscope_handler to get channels.")

        all_channels_set: Set[int] = set()
        try:
            filenames = self.filemanager.list_files(str(self.input_dir), Backend.DISK.value, extensions=DEFAULT_IMAGE_EXTENSIONS)
            for filename in filenames:
                parsed_info = self.microscope_handler.parser.parse_filename(str(filename))
                if parsed_info and 'channel' in parsed_info and parsed_info['channel'] is not None:
                    all_channels_set.add(int(parsed_info['channel']))
                else:
                    logger.warning(f"Could not parse channel information from filename: {filename}")
        except Exception as e:
            logger.error(f"Error listing files or parsing channel names from {self.input_dir}: {e}", exc_info=True)
            return []

        all_channels = sorted(list(all_channels_set))

        if not all_channels:
            logger.warning(f"No channels found in input directory: {self.input_dir} with extensions {DEFAULT_IMAGE_EXTENSIONS}")
            return []

        if channel_filter:
            int_channel_filter = {int(c) for c in channel_filter}
            selected_channels = [channel for channel in all_channels if channel in int_channel_filter]
            if not selected_channels:
                logger.warning(f"No channels from {all_channels} match the filter: {channel_filter}")
            return selected_channels
        else:
            return all_channels

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
