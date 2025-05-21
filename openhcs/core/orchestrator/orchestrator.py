"""
Consolidated orchestrator module for OpenHCS.

This module provides a unified PipelineOrchestrator class that combines
the functionality of the previous orchestrator and dispatcher components.

Doctrinal Clauses:
- Clause 12 — Absolute Clean Execution
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 293 — GPU Pre-Declaration Enforcement
- Clause 295 — GPU Scheduling Affinity
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from openhcs.constants.constants import DEFAULT_NUM_WORKERS, Backend, DEFAULT_WORKSPACE_DIR_SUFFIX, DEFAULT_IMAGE_EXTENSIONS
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.pipeline.pipeline import (PipelineCompiler,
                                               PipelineExecutor)
from openhcs.io.exceptions import StorageWriteError
from openhcs.io.filemanager import FileManager
from openhcs.io.base import storage_registry 
from openhcs.microscopes.microscope_interfaces import (
    MicroscopeHandler, create_microscope_handler)
from openhcs.runtime.napari_stream_visualizer import NapariStreamVisualizer

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Unified orchestrator for pipeline execution.

    This class combines the responsibilities of the previous orchestrator and
    dispatcher components, providing a simpler, more cohesive API for pipeline
    execution.

    Thread Safety:
        This class is thread-safe and can be used to execute multiple pipelines
        concurrently.
    """

    def __init__(
        self,
        plate_path: Union[str, Path],
        workspace_path: Optional[Union[str, Path]] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            plate_path: Path to the plate directory
            workspace_path: Path to the workspace directory (optional)
            config: Configuration parameters (optional)
        """
        self._lock = threading.RLock()
        self._config = config or {}

        if plate_path is not None:
            if isinstance(plate_path, str):
                plate_path = Path(plate_path)
            elif not isinstance(plate_path, Path):
                raise ValueError(f"Invalid plate_path type: {type(plate_path)}")

        # Store plate_path as a regular path (string or Path)
        self.plate_path = plate_path

        # Set workspace_path if provided, otherwise it will be set in initialize()
        self.workspace_path = workspace_path

        if self.plate_path is None and self.workspace_path is None:
            raise ValueError("Either plate_path or workspace_path must be provided")
      #  elif self.workspace_path is None and self.plate_path:
      #      self.workspace_path = str(self.plate_path) + DEFAULT_WORKSPACE_DIR_SUFFIX

        # Initialize registry
        self.registry = storage_registry()

        # Initialize file manager with registry
        self.filemanager = FileManager(self.registry)
        self.input_dir = None
        self.microscope_handler = None

        # Pipeline storage
        self.pipeline = None

        # Initialization flag
        self._initialized: bool = False



    def initialize_workspace(self,workspace_path: Optional[Union[str, Path]] = None):
        """
        Initialize workspace path and mirror plate directory if needed.

        # Clause 17 — VFS Exclusivity
        # Clause 77 — Rot Intolerance
        # Clause 245 — Declarative Enforcement
        """
        #if self.workspace_path is not None:
        #    logger.debug("Workspace path already set")
        #    return
        self.workspace_path = workspace_path
        if self.workspace_path is None and self.plate_path:
            self.workspace_path = str(self.plate_path) + DEFAULT_WORKSPACE_DIR_SUFFIX

        # Create workspace directory if it doesn't exist
        self.filemanager.ensure_directory(self.workspace_path, Backend.DISK.value)

        # CRITICAL: Mirror plate directory to workspace with symlinks
        if self.plate_path and self.workspace_path:
            logger.info("Mirroring plate directory to workspace...")
            # Clause 245: Workspace operations are disk-only by design

            # Pass regular paths to FileManager, which will handle VirtualPath conversion internally
            try:
                # Clause 245: Workspace operations are disk-only by design
                num_links = self.filemanager.mirror_directory_with_symlinks(
                    self.plate_path,
                    self.workspace_path,
                    Backend.DISK.value,
                    recursive=True,
                    overwrite=True
                )
                logger.info("Created %d symlinks in workspace", num_links)
                self.input_dir = self.workspace_path
            except Exception as e:
                error_msg = f"Failed to mirror plate directory to workspace: {e}"
                logger.error(error_msg)
                raise StorageWriteError(error_msg) from e
        else:
            # Set input directory to plate path if no mirroring was done
            self.input_dir = self.plate_path
            logger.info("Set input directory to plate path: %s", self.input_dir)

    def initialize_microscope_handler(self,):
        """
        Initialize the microscope handler.

        This method creates a microscope handler based on the input directory
        and initializes it with the file manager.
        """
        if self.microscope_handler is not None:
            logger.debug("Microscope handler already initialized")
            return

        logger.info("Initializing microscope handler using workspace...")

        # CRITICAL: Initialize the microscope handler with the disk-based file manager
        try:
            # Pass the filemanager as a positional argument as required by Clause 306
            # (Backend Positional Parameters)
            self.microscope_handler = create_microscope_handler(
                self.workspace_path,  # plate_folder as positional arg
                self.filemanager,  # filemanager as positional arg
                microscope_type='auto',  # Use auto-detection
            )
        except Exception as e:
            error_msg = f"Failed to create microscope handler: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # This is necessary for pattern matching to work correctly
        logger.info("Processing workspace directory with microscope handler...")

        # The microscope handler will use the FileManager to convert to VirtualPath internally
        try:
            # Pass the physical workspace path and filemanager to post_workspace
            self.input_dir = self.microscope_handler.post_workspace(self.workspace_path, self.filemanager)
            logger.info("Workspace directory processed")
        except Exception as e:
            error_msg = f"Failed to process workspace directory: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        logger.info(
            "Initialized microscope handler: %s",
            type(self.microscope_handler).__name__
        )

    def initialize(self):
        """
        Initialize the orchestrator.

        This method initializes all required components for the orchestrator.
        It must be called before any other methods are used.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            logger.info("Orchestrator already initialized")
            return self

        self.initialize_workspace()
        self.initialize_microscope_handler()

        self._initialized = True
        logger.info("PipelineOrchestrator fully initialized.")

    def is_initialized(self) -> bool:
        """
        Check if the orchestrator is initialized.

        Returns:
            True if the orchestrator is initialized, False otherwise
        """
        return self._initialized

    def create_context(self, well: str) -> ProcessingContext:
        """
        Create a ProcessingContext for a given well.

        Args:
            well: The well identifier to create a context for

        Returns:
            A new ProcessingContext instance

        Raises:
            RuntimeError: If the orchestrator is not initialized
            ValueError: If the well is not valid
        """
        if not self.is_initialized():
            raise RuntimeError("Orchestrator must be initialized before calling create_context()")

        if not well:
            raise ValueError("Well identifier must be provided")

        # Create a new context with the well ID
        context = ProcessingContext(well_id=well)

        # Add orchestrator to context
        context.orchestrator = self

        # Add file manager to context
        context.filemanager = self.filemanager

        # Add microscope handler to context
        context.microscope_handler = self.microscope_handler

        # Add input directory to context
        context.input_dir = self.input_dir

        # Add workspace path to context
        context.workspace_path = self.workspace_path

        return context

    def create_pipeline(self, steps: List[Any], well_id: str, context: Optional[ProcessingContext] = None) -> List[Any]:
        """
        Create a pipeline with automatic memory type conversion.

        Args:
            steps: List of steps to include in the pipeline
            well_id: Identifier of the well being processed
            context: Optional ProcessingContext to inject step plans into

        Returns:
            List of steps with conversion steps inserted where needed

        Raises:
            ValueError: If memory types are incompatible and no conversion is available
            ValueError: If well_id is not provided
        """
        if not steps:
            raise ValueError("Steps must be provided")

        if not well_id:
            raise ValueError("Well identifier must be provided")

        # Compile the pipeline with automatic memory type conversion
        pipeline = PipelineCompiler.compile(steps, self.input_dir, well_id)

        # Inject step plans into context if provided
        if context is not None:
            PipelineCompiler.inject_step_plans(pipeline, context)

        return pipeline

    def execute_pipeline(self, context: ProcessingContext, pipeline: List[Any]) -> ProcessingContext:
        """
        Execute a pipeline with the given context.

        Args:
            context: Fully prepared ProcessingContext
            pipeline: List of steps to execute

        Returns:
            Updated context after pipeline execution

        Raises:
            ValueError: If context or pipeline is invalid
        """
        # Validate inputs
        if context is None:
            raise ValueError("Context cannot be None")
        if pipeline is None:
            raise ValueError("Pipeline cannot be None")

        # Log dispatch with structured logging
        logger.info("Dispatching pipeline execution for well: %s", context.well_id)

        # Execute pipeline using PipelineExecutor
        updated_context = PipelineExecutor.execute(
            steps=pipeline,
            context=context
        )

        # Log completion with structured logging
        logger.info("Pipeline execution completed for well: %s", context.well_id)

        return updated_context

    def run(
        self,
        pipeline=None,
        well_filter: Optional[List[str]] = None,
        max_workers: int = DEFAULT_NUM_WORKERS,
        enable_visualizer_override: bool = False
    ) -> Dict[str, ProcessingContext]:
        """
        Process all wells using the given pipeline.

        Args:
            pipeline: List of steps to execute (optional)
            well_filter: Optional list of wells to process
            max_workers: Maximum number of worker threads (1 for sequential)
            enable_visualizer_override: Whether to enable visualization regardless of step settings

        Returns:
            Dict mapping wells to their updated contexts

        Raises:
            RuntimeError: If the orchestrator is not initialized
            ValueError: If no pipeline is provided and no default pipeline is set
            ValueError: If context creation fails
        """
        with self._lock:
            # Validate orchestrator is initialized
            if not self.is_initialized():
                raise RuntimeError("Orchestrator must be initialized before calling run()")

            # Use provided pipeline or default
            pipeline_to_use = pipeline or self.pipeline
            if not pipeline_to_use:
                raise ValueError("No pipeline provided and no default pipeline set")

            # Get wells to process
            wells = self.get_wells(well_filter)

            if not wells:
                logger.warning("No wells found to process")
                return {}

            # Log the start of processing with structured logging
            logger.info("Processing %d wells: %s", len(wells), ", ".join(wells))

            # Prepare contexts and pipelines for all wells
            contexts = []
            #pipelines = []

            for well in wells:

                compiled = PipelineCompiler.compile(pipeline, input_dir=self.input_dir, well_id=well)
                PipelineCompiler.inject_step_plans(compiled, context)
               # # Create pipeline with automatic memory type conversion
               # pipeline_with_conversions = self.create_pipeline(
               #     pipeline_to_use,
               #     well,
               #     context
               # )

                # Add to lists
                contexts.append(context)
                pipelines.append(pipeline_with_conversions)

            # Initialize visualizer if needed
            visualizer_instance = None
            if enable_visualizer_override or any(plan.get('visualize', False) for context in contexts for plan in context.step_plans.values()):
                logger.info("Visualization requested. Initializing NapariStreamVisualizer.")
                visualizer_instance = NapariStreamVisualizer()
                # Visualizer will be started on first tensor push

            # Execute pipelines (parallel if max_workers > 1 and multiple wells)
            updated_contexts = PipelineExecutor.execute_parallel(
                pipelines=pipelines,
                contexts=contexts,
                max_workers=max_workers,
                visualizer=visualizer_instance
            )

            # Stop visualizer if it was created
            if visualizer_instance:
                logger.info("All pipeline executions finished. Stopping visualizer.")
                visualizer_instance.stop_viewer()

            # Create a dictionary mapping wells to their updated contexts
            results = {}
            for i, well in enumerate(wells):
                results[well] = updated_contexts[i]

            return results

    def get_wells(self, well_filter: Optional[List[str]] = None) -> List[str]:
        """
        Get the wells to process based on the filter.

        Args:
            well_filter: Optional list of wells to process

        Returns:
            List of wells to process
        """
        # Get all available wells
        all_wells = []
        filenames = self.filemanager.list_files(self.input_dir, Backend.DISK.value,extensions=DEFAULT_IMAGE_EXTENSIONS)
        for filename in filenames:
            all_wells.append(self.microscope_handler.parser.parse_filename(filename)['well'])

        # Filter wells if a filter is provided
        if well_filter:
            wells = [well for well in all_wells if well in well_filter]
            if not wells:
                logger.warning("No wells match the filter: %s", well_filter)
        else:
            wells = all_wells

        return wells
