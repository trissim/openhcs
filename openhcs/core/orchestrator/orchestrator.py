"""
Consolidated orchestrator module for OpenHCS.

This module provides a unified PipelineOrchestrator class that implements
a two-phase (compile-all-then-execute-all) pipeline execution model.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, Set

from openhcs.constants.constants import (
    Backend,
    LOADABLE_IMAGE_EXTENSIONS,
    GroupBy,
    OrchestratorState,
    AllComponents,
    VariableComponents,
)
from openhcs.constants import Microscope
from openhcs.core.compiled_execution import CompiledExecutionBundle
from openhcs.core.config import GlobalPipelineConfig
from objectstate.object_state import ObjectState
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig


from openhcs.core.metadata_cache import MetadataCache
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.input_workspace import InputWorkspacePreparationResult
from openhcs.core.source_binding_context import SourceBindingContext
from openhcs.core.source_bindings import source_bindings_defaults_to_base
from openhcs.core.pipeline.compiler import PipelineCompiler
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.components.validation import convert_enum_by_value
from openhcs.core.orchestrator.execution_result import (
    ExecutionResult,
    RuntimeObservationMode,
)
from openhcs.core.orchestrator.cancellation import ExecutionCancellationAuthority
from openhcs.core.orchestrator.compiled_plate_execution import (
    CompiledPlateExecutionRequest,
    execute_compiled_plate_request,
)
from openhcs.core.debug import (
    DebugExecutionPolicy,
    NoOpDebugExecutionPolicy,
)
from openhcs.core.progress import ProgressExecutionContext
from polystore.filemanager import FileManager

if TYPE_CHECKING:
    from openhcs.core.config import PipelineConfig

# Zarr backend is CPU-only; always import it (even in subprocess/no-GPU mode)
from polystore.zarr import ZarrStorageBackend

# PipelineConfig now imported directly above
from openhcs.microscopes import create_microscope_handler
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.core.alias_property import AliasProperty

# Import generic component system - required for orchestrator functionality

# Optional napari import for visualization
try:
    from openhcs.runtime.napari_stream_visualizer import NapariStreamVisualizer

    NapariVisualizerType = NapariStreamVisualizer
except ImportError:
    # Create a placeholder type for type hints when napari is not available
    NapariStreamVisualizer = None
    NapariVisualizerType = Any  # Use Any for type hints when napari is not available

logger = logging.getLogger(__name__)


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
    _plate_path: Optional[Path] = None
    _plate_path_frozen: bool = False
    _metadata_cache_service: Optional["MetadataCache"] = None
    state: AliasProperty[OrchestratorState] = AliasProperty("_state")
    metadata_cache: AliasProperty[MetadataCache] = AliasProperty(
        "_metadata_cache_service"
    )

    def __init__(
        self,
        plate_path: Union[str, Path],
        workspace_path: Optional[Union[str, Path]] = None,
        *,
        pipeline_config: Optional["PipelineConfig"] = None,
        storage_registry: Optional[Any] = None,
        selected_pipeline_path: Union[str, Path, None] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        transport_config: OpenHCSZMQConfig = OPENHCS_ZMQ_CONFIG,
    ):
        # Lock removed - was orphaned code never used

        # Track executor for cancellation support
        self._executor = None
        self._execution_cancellation = ExecutionCancellationAuthority()
        self.execution_id = f"local::{plate_path}"
        self.transport_config = transport_config

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

        # Convert to the immutable execution identity. Source availability is a
        # runtime initialization precondition so declarations can be loaded and
        # edited while their external source is temporarily unavailable.
        if plate_path:
            plate_path = Path(plate_path)
            if not plate_path.is_absolute():
                raise ValueError(f"Plate path must be absolute: {plate_path}")

        self._plate_path_frozen = False

        self.plate_path = plate_path
        self.workspace_path = workspace_path
        self.source_plate_path = plate_path
        self.input_workspace_preparation_result: (
            InputWorkspacePreparationResult | None
        ) = None
        self.selected_pipeline_path = (
            Path(selected_pipeline_path) if selected_pipeline_path is not None else None
        )

        if self.plate_path is None and self.workspace_path is None:
            raise ValueError(
                "Either plate_path or workspace_path must be provided for PipelineOrchestrator."
            )

        # Freeze plate_path immediately after setting it to prove immutability
        self._plate_path_frozen = True
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

        # Override zarr backend with orchestrator's resolved config.
        effective_config = self.get_effective_config()
        zarr_backend_with_config = ZarrStorageBackend(effective_config.zarr_config)
        self.registry[Backend.ZARR.value] = zarr_backend_with_config
        logger.info(
            f"Orchestrator zarr backend configured with {effective_config.zarr_config.compressor.value} compression"
        )

        # Orchestrator always creates its own FileManager, using the determined registry
        self.filemanager = FileManager(self.registry)
        self.input_dir: Optional[Path] = None
        self.microscope_handler: Optional[MicroscopeHandler] = None
        self._microscope_handler_rebuild_type: type[MicroscopeHandler] | None = None
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

    @property
    def plate_path(self) -> Optional[Path]:
        """Execution plate path for this orchestrator."""

        return self._plate_path

    @plate_path.setter
    def plate_path(self, value: Optional[Path]) -> None:
        """Set plate path until the orchestrator freezes execution identity."""

        if self._plate_path_frozen:
            import traceback

            stack_trace = "".join(traceback.format_stack())
            error_msg = (
                f"🚫 IMMUTABLE PLATE_PATH VIOLATION: Cannot modify plate_path after freezing!\n"
                f"Current value: {self._plate_path}\n"
                f"Attempted new value: {value}\n"
                f"Stack trace:\n{stack_trace}"
            )
            logger.error(error_msg)
            raise AttributeError(error_msg)
        self._plate_path = value

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
            key = (config.viewer_type, config.port)
            from openhcs.core.viewer_streaming_service import StreamingViewerLifecycle

            viewer = StreamingViewerLifecycle.get_or_create_visualizer(
                filemanager=self.filemanager,
                config=config,
                visualizer_config=vis_config,
                transport_config=self.transport_config,
                fresh=True,
                ready_timeout=30.0,
            )

            # Keep a reference for backward compatibility
            self._visualizers[key] = viewer
            return viewer

        # Non-streaming (local) visualizers: create and start synchronously
        vis = config.create_visualizer(
            self.filemanager,
            vis_config,
            self.transport_config,
        )
        vis.start_viewer()

        # Store for compatibility
        backend_name = config.backend.name
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
            shared_context = self.get_effective_config()
            microscope_type = (
                shared_context.microscope.value
                if shared_context.microscope != Microscope.AUTO
                else "auto"
            )
            if self._microscope_handler_rebuild_type is None:
                self.microscope_handler = create_microscope_handler(
                    plate_folder=str(self.plate_path),
                    filemanager=self.filemanager,
                    microscope_type=microscope_type,
                    source_bindings_config=shared_context.source_bindings_config,
                )
            else:
                self.microscope_handler = self._microscope_handler_rebuild_type.create(
                    filemanager=self.filemanager,
                    source_bindings_config=shared_context.source_bindings_config,
                )
                self.microscope_handler.plate_folder = Path(self.plate_path)
                self._microscope_handler_rebuild_type = None
            logger.info(
                f"Initialized microscope handler: {type(self.microscope_handler).__name__}"
            )
        except Exception as e:
            error_msg = f"Failed to create microscope handler: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def bind_input_workspace(
        self,
        result: InputWorkspacePreparationResult,
    ) -> None:
        """Bind a caller-prepared generic input workspace before initialization."""

        if self._initialized or self.state is not OrchestratorState.CREATED:
            raise RuntimeError(
                "An input workspace can only be bound before initialization."
            )
        self.input_workspace_preparation_result = result
        self.source_plate_path = Path(result.original_source_root)
        if Path(result.execution_plate_path) == Path(self.plate_path):
            return
        self._rebind_plate_path_for_prepared_workspace(result.execution_plate_path)

    def source_binding_context(self, logical_plate_id: str) -> SourceBindingContext:
        """Project the current source declaration and workspace owned by this plate."""

        if not logical_plate_id:
            raise ValueError("Source-binding context requires a logical plate id.")
        if self.source_plate_path is None or self.plate_path is None:
            raise RuntimeError("Source-binding context requires a bound plate path.")
        if self.pipeline_config is None:
            raise RuntimeError("Source-binding context requires a PipelineConfig.")
        return SourceBindingContext(
            logical_plate_id=logical_plate_id,
            display_plate_root=self.source_plate_path,
            execution_plate_path=self.plate_path,
            source_bindings=source_bindings_defaults_to_base(
                self.pipeline_config.source_bindings_config
            ),
            filemanager=self.filemanager,
            source_backend=Backend.DISK.value,
        )

    def _rebind_plate_path_for_prepared_workspace(
        self,
        execution_plate_path: Path,
    ) -> None:
        """Bind CREATED orchestrator execution to its prepared input workspace."""

        if self._initialized or self.state is not OrchestratorState.CREATED:
            raise RuntimeError(
                "Prepared input workspace can only rebind a CREATED orchestrator."
            )
        self._rebind_created_plate_path(Path(execution_plate_path))
        self.execution_id = f"local::{self.plate_path}"
        logger.info(
            "Prepared input workspace rebound orchestrator execution path to %s",
            self.plate_path,
        )

    def _rebind_created_plate_path(self, execution_plate_path: Path) -> None:
        self._plate_path_frozen = False
        try:
            self.plate_path = execution_plate_path
        finally:
            self._plate_path_frozen = True

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
            self.microscope_handler.source_selection_role().require_available_source(
                self.plate_path
            )

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
                vfs_cfg = self.get_effective_config().vfs_config
                logger.info(
                    "VFS config at init: read_backend=%s intermediate_backend=%s materialization_backend=%s",
                    vfs_cfg.read_backend,
                    vfs_cfg.intermediate_backend,
                    vfs_cfg.materialization_backend,
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

        Skips remote-service handlers because they do not have local source
        directories.
        """
        from openhcs.microscopes.openhcs import (
            OpenHCSMetadataGenerator,
            get_subdirectory_name,
        )

        source_role = self.microscope_handler.source_selection_role()
        if not source_role.requires_local_directory:
            logger.debug("Skipping local metadata creation for %s", source_role.value)
            return

        # For plates with virtual workspace, metadata is already created by _build_virtual_mapping()
        # We just need to add the component metadata to the existing "." subdirectory
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

        effective_config = self.get_effective_config()
        materialization_path = effective_config.materialization_results_path

        # If absolute, use as-is
        if Path(materialization_path).is_absolute():
            return Path(materialization_path)

        # If relative, resolve relative to output plate root
        path_config = effective_config.path_planning_config
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

        effective_config = self.get_effective_config()
        context = ProcessingContext(
            axis_id=axis_id,
            filemanager=self.filemanager,
            analysis_consolidation_config=(
                effective_config.analysis_consolidation_config
            ),
            plate_metadata_config=effective_config.plate_metadata_config,
            auto_add_output_plate_to_plate_manager=(
                effective_config.auto_add_output_plate_to_plate_manager
            ),
            transport_config=self.transport_config,
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

    def source_workspace_projection(self):
        """Return the canonical resolved source state for the initialized plate.

        The projection carries every named alias, sample/well, site, channel, Z,
        timepoint, and backend-owned pixel reference used by compilation, runtime,
        and UI inspection. Consumers must query this view rather than constructing
        a second metadata model.
        """
        if not self.is_initialized():
            raise RuntimeError(
                "Orchestrator must be initialized before source workspace inspection."
            )
        if self.plate_path is None or self.microscope_handler is None:
            raise RuntimeError("Orchestrator source workspace is not available.")

        plate_path = Path(self.plate_path)
        from openhcs.core.source_workspace_projection import (
            VirtualWorkspaceSourceProjection,
            VirtualWorkspaceSourceProjectionAuthority,
        )

        projection = VirtualWorkspaceSourceProjectionAuthority.from_plate_metadata(
            plate_path=plate_path,
            metadata_handler=self.microscope_handler.metadata_handler,
            filemanager=self.filemanager,
        ).projection_if_available()
        if projection is not None:
            return projection
        return VirtualWorkspaceSourceProjection.empty(plate_path)

    def source_workspace_files(self, axis_id: str | None = None) -> tuple[str, ...]:
        """Return VFS-visible virtual source paths for one axis or all axes."""
        return self.source_workspace_projection().pipeline_start_files(axis_id=axis_id)

    def compile_pipelines(
        self,
        pipeline_definition: List[AbstractStep],
        well_filter: Optional[List[str]] = None,
        enable_visualizer_override: bool = False,
        is_zmq_execution: bool = False,
        debug_execution_policy: DebugExecutionPolicy = NoOpDebugExecutionPolicy(),
    ) -> CompiledExecutionBundle:
        """Compile the selected axes into one typed execution bundle."""
        return PipelineCompiler.compile_pipelines(
            orchestrator=self,
            pipeline_definition=pipeline_definition,
            axis_filter=well_filter,  # Translate well_filter to axis_filter for generic backend
            enable_visualizer_override=enable_visualizer_override,
            is_zmq_execution=is_zmq_execution,
            debug_execution_policy=debug_execution_policy,
        )

    def cancel_execution(self):
        """
        Cancel ongoing execution by shutting down the executor.

        This gracefully cancels pending futures and shuts down worker processes
        without killing all child processes (preserving Napari viewers, etc.).
        """
        self._execution_cancellation.request()

        if self._executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.warning(f"🔥 ORCHESTRATOR: Failed to cancel executor: {e}")

    def execute_compiled_plate(
        self,
        execution_bundle: CompiledExecutionBundle,
        max_workers: Optional[int] = None,
        visualizer: Optional[NapariVisualizerType] = None,
        log_file_base: Optional[str] = None,
        progress_queue=None,
        progress_context=None,
        runtime_observation_mode: RuntimeObservationMode = RuntimeObservationMode.MERGE_INTO_PARENT,
        debug_execution_policy: DebugExecutionPolicy = NoOpDebugExecutionPolicy(),
    ) -> Dict[str, ExecutionResult]:
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
        if progress_context is None:
            raise ValueError("progress_context is required for execute_compiled_plate.")
        execution_progress_context = ProgressExecutionContext.from_value(
            progress_context
        )

        return execute_compiled_plate_request(
            self,
            CompiledPlateExecutionRequest(
                execution_id=execution_progress_context.execution_id,
                plate_id=execution_progress_context.plate_id,
                execution_bundle=execution_bundle,
                max_workers=max_workers,
                visualizer=visualizer,
                log_file_base=log_file_base,
                progress_queue=progress_queue,
                runtime_observation_mode=runtime_observation_mode,
                debug_execution_policy=debug_execution_policy,
            ),
        )

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
                self.microscope_handler.microscope_type,
                self.microscope_handler.parser.__class__.__name__,
            )

            filenames = self.filemanager.list_files(
                str(self.input_dir),
                backend_to_use,
                extensions=LOADABLE_IMAGE_EXTENSIONS,
            )
            logger.info(
                "Component key discovery: listed %d files (extensions=%s)",
                len(filenames),
                LOADABLE_IMAGE_EXTENSIONS,
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
                        component_value = parsed_info.value_for(component)
                        if component_value is not None:
                            component_sets[component].add(component_value)
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

    # Global config management removed - handled by UI layer

    @property
    def pipeline_config(self) -> Optional["PipelineConfig"]:
        """Get current pipeline configuration."""
        return self._pipeline_config

    @pipeline_config.setter
    def pipeline_config(self, value: Optional["PipelineConfig"]) -> None:
        """Set pipeline configuration with auto-sync to thread-local context."""
        self._pipeline_config = value
        if self._auto_sync_enabled and value is not None:
            self._sync_to_thread_local()

    def _sync_to_thread_local(self) -> None:
        """Internal method to sync current pipeline_config to thread-local context."""
        if self._pipeline_config and self.plate_path is not None:
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

        previous_config = self._pipeline_config
        previous_source_bindings = None
        if previous_config is not None:
            previous_source_bindings = (
                ObjectState(previous_config)
                .to_saved_resolved_object()
                .source_bindings_config
            )
        current_source_bindings = (
            ObjectState(pipeline_config)
            .to_saved_resolved_object()
            .source_bindings_config
        )
        source_bindings_changed = (
            previous_source_bindings is not None
            and previous_source_bindings != current_source_bindings
        )
        if source_bindings_changed and self.state is OrchestratorState.EXECUTING:
            raise RuntimeError(
                "Source bindings cannot change while the plate is executing."
            )

        # Temporarily disable auto-sync to prevent recursion
        self._auto_sync_enabled = False
        try:
            self._pipeline_config = pipeline_config
        finally:
            self._auto_sync_enabled = True

        if source_bindings_changed:
            self._invalidate_source_projection()

        # CRITICAL FIX: Do NOT contaminate thread-local context during PipelineConfig editing
        # The orchestrator should maintain its own internal context without modifying
        # the global thread-local context. This prevents reset operations from showing
        # orchestrator's saved values instead of original thread-local defaults.
        #
        # The merged config is computed internally and used by get_effective_config()
        # but should NOT be set as the global thread-local context.

        logger.info(f"Applied orchestrator config for plate: {self.plate_path}")

    def _invalidate_source_projection(self) -> None:
        """Require normal initialization to rebuild source-owned plate state."""

        if (
            self.microscope_handler is not None
            and type(self.microscope_handler).projects_declared_source_bindings()
        ):
            self._microscope_handler_rebuild_type = type(self.microscope_handler)
        self.microscope_handler = None
        self.input_dir = None
        self._initialized = False
        self._state = OrchestratorState.CREATED
        self._component_keys_cache.clear()
        self._metadata_cache_service.clear_cache()

    def get_effective_config(
        self, *, for_serialization: bool = False
    ) -> GlobalPipelineConfig:
        """
        Get effective configuration for this orchestrator.

        Args:
            for_serialization: Retained for compatibility; the returned config is
                always the saved ObjectState-resolved concrete configuration.
        """

        if self.pipeline_config is None:
            raise RuntimeError("No pipeline configuration available for resolution")

        result = ObjectState(self.pipeline_config).to_saved_resolved_object()
        if not isinstance(result, GlobalPipelineConfig):
            raise TypeError(
                "Resolved pipeline configuration must be GlobalPipelineConfig, "
                f"got {type(result).__name__}."
            )
        return result

    def clear_pipeline_config(self) -> None:
        """Clear per-orchestrator configuration."""
        # REMOVED: Thread-local modification - dual-axis resolver handles context automatically
        # No need to modify thread-local storage when clearing orchestrator config
        self.pipeline_config = None
        # Clear metadata cache for this orchestrator
        if self._metadata_cache_service is not None:
            self._metadata_cache_service.clear_cache()
        logger.info(f"Cleared per-orchestrator config for plate: {self.plate_path}")

    def cleanup_pipeline_config(self) -> None:
        """Clean up orchestrator context when done (for backward compatibility)."""
        self.clear_pipeline_config()
