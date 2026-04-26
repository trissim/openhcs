"""
Processing Context for OpenHCS.

This module defines the ProcessingContext class, which maintains state during pipeline execution.
"""

from typing import Any, Dict, Optional

from openhcs.core.config import GlobalPipelineConfig, VFSConfig, PathPlanningConfig
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.runtime_stores import RuntimeValueStore


class ProcessingContext:
    """
    Maintains state during pipeline execution.

    The ProcessingContext is the canonical owner of all state during pipeline execution.
    After compilation and freezing, it should be treated as immutable by processing steps.

    OWNERSHIP: This class may ONLY be instantiated by PipelineOrchestrator.
    All other components must receive a context instance, never create one.

    Attributes:
        step_plans: Dictionary mapping step indices to compiled execution plans.
        outputs: Dictionary for step outputs (usage may change with VFS-centric model).
        intermediates: Dictionary for intermediate results (usage may change).
        current_step: Current executing step ID (usage may change).
        axis_id: Identifier of the multiprocessing axis value being processed.
        filemanager: Instance of FileManager for VFS operations.
        global_config: GlobalPipelineConfig holding system-wide configurations.
        pipeline_sequential_mode: Flag indicating pipeline-wide vs step-wide sequential processing.
        pipeline_sequential_combinations: Pre-computed sequential combinations for pipeline-wide mode.
        current_sequential_combination: Active combination during pipeline-wide sequential execution.
        _is_frozen: Internal flag indicating if the context is immutable.
    """

    def __init__(
        self,
        global_config: GlobalPipelineConfig, # Made a required argument
        step_plans: Optional[Dict[int, CompiledStepPlan]] = None,
        axis_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the processing context.

        Args:
            global_config: The global pipeline configuration object.
            step_plans: Dictionary mapping step indices to compiled execution plans.
            axis_id: Identifier of the multiprocessing axis value being processed.
            **kwargs: Additional context attributes (e.g., filemanager, microscope_handler).
        """
        # Initialize _is_frozen first to allow other attributes to be set by __setattr__
        # This direct assignment bypasses the custom __setattr__ during initialization.
        object.__setattr__(self, '_is_frozen', False)

        self.step_plans = step_plans or {}
        self.outputs = {}  # Future use TBD, primary data flow via VFS
        self.intermediates = {} # Future use TBD, primary data flow via VFS
        self.runtime_value_store = RuntimeValueStore()
        self.current_step = None # Future use TBD
        self.axis_id = axis_id
        self.global_config = global_config # Store the global config
        self.filemanager = None # Expected to be set by Orchestrator via kwargs or direct assignment
        self.required_visualizers: list[Any] = []
        self.step_axis_filters: dict[int, dict[str, Any]] = {}

        # Execution tracking fields (set at execution time)
        self.execution_id = None  # Set by worker before execution
        self.plate_id = None  # Set by worker before execution (same as plate_path)
        self.worker_slot = None  # Logical worker slot for deterministic ownership
        self.owned_wells = None  # Full well set owned by this worker slot

        # Pipeline-wide sequential processing fields
        self.pipeline_sequential_mode = False
        self.pipeline_sequential_combinations = None  # Precomputed at compile time from metadata
        self.current_sequential_combination = None  # Set by compiler for each combination iteration

        # Add any additional attributes from kwargs
        # Note: 'filemanager' is often passed via kwargs by PipelineOrchestrator.create_context
        for key, value in kwargs.items():
            setattr(self, key, value) # This will now go through our __setattr__

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute, preventing modification if the context is frozen.

        All fields are immutable once frozen - no exceptions.
        """
        if self._is_frozen and name != '_is_frozen':
            raise AttributeError(f"Cannot modify attribute '{name}' of a frozen ProcessingContext.")
        super().__setattr__(name, value)

    def inject_plan(self, step_id: int, plan: CompiledStepPlan) -> None:
        """
        Inject a step plan into the context.

        This method is the canonical way to add step plans to the context during compilation.
        All step configuration must be injected into the context using this method.

        Args:
            step_id: The step index used as the compiled-plan key.
            plan: The step execution plan

        Raises:
            AttributeError: If the context is frozen.
        """
        if self._is_frozen:
            raise AttributeError("Cannot inject plan into a frozen ProcessingContext.")
        self.step_plans[step_id] = plan

    def freeze(self) -> None:
        """
        Freezes the context, making its attributes immutable.

        This should be called after all compilation and plan injection is complete.
        Essential attributes like step_plans, filemanager, and axis_id must be set.

        Raises:
            RuntimeError: If essential attributes are not set before freezing.
        """
        if not self.axis_id:
            raise RuntimeError("Cannot freeze ProcessingContext: 'axis_id' is not set.")
        if self.filemanager is None:
            raise RuntimeError("Cannot freeze ProcessingContext: 'filemanager' is not set.")
        # step_plans can be empty if the pipeline is empty, but it must exist.
        # Trust dataclass contract - if step_plans doesn't exist, that's a bug
        if self.step_plans is None:
             raise RuntimeError("Cannot freeze ProcessingContext: 'step_plans' is not set.")

        self._is_frozen = True # This assignment is allowed by __setattr__

    def is_frozen(self) -> bool:
        """
        Check if the context is frozen.

        Returns:
            True if the context is frozen, False otherwise.
        """
        return self._is_frozen



    # update_from_step_result method is removed as per plan.

    # --- Config Getters ---
    # NOTE: These are only used outside compilation (e.g., in workers after context is frozen)
    # During compilation, code should access orchestrator.pipeline_config directly

    def get_vfs_config(self) -> VFSConfig:
        """Returns the VFSConfig part of the global configuration."""
        if self.global_config is None:
            raise RuntimeError("GlobalPipelineConfig not set on ProcessingContext.")
        return self.global_config.vfs_config

    def get_path_planning_config(self) -> PathPlanningConfig:
        """Returns the PathPlanningConfig part of the global configuration."""
        if self.global_config is None:
            raise RuntimeError("GlobalPipelineConfig not set on ProcessingContext.")
        return self.global_config.path_planning_config

    def get_num_workers(self) -> int:
        """Returns the number of workers from the global configuration."""
        if self.global_config is None:
            raise RuntimeError("GlobalPipelineConfig not set on ProcessingContext.")
        return self.global_config.num_workers

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare context for pickling (e.g., for multiprocessing).

        Excludes the filemanager from pickling to avoid copying the storage registry
        across process boundaries. The filemanager will be recreated in the worker
        process using the worker's local global registry.

        Uses self-describing backend pickling: iterates over all backends and preserves
        connection params for any that implement PicklableBackend, storing their class
        info for dynamic recreation.

        Returns:
            Dictionary of state to pickle
        """
        from openhcs.constants.constants import Backend
        from polystore.base import PicklableBackend

        state = self.__dict__.copy()

        # Preserve zarr config from global_config for filemanager recreation
        # Trust dataclass contract - if global_config doesn't exist, that's a bug
        state['_zarr_config'] = self.global_config.zarr_config if self.global_config else None

        # Preserve plate_path for virtual_workspace backend recreation
        # Trust dataclass contract - plate_path is a defined field
        state['_plate_path'] = self.plate_path

        # Self-describing backend pickling: iterate over all backends and preserve
        # connection params for any that implement PicklableBackend
        state['_picklable_backends'] = {}

        if self.filemanager is not None:
            # Track virtual_workspace separately for backward compatibility
            state['_has_virtual_workspace'] = Backend.VIRTUAL_WORKSPACE.value in self.filemanager.registry

            # Iterate over all registered backends and preserve picklable ones
            for backend_key, backend_instance in self.filemanager.registry.items():
                if isinstance(backend_instance, PicklableBackend):
                    params = backend_instance.get_connection_params()
                    if params is not None:
                        # Store backend class info for dynamic recreation
                        state['_picklable_backends'][backend_key] = {
                            'class_name': type(backend_instance).__name__,
                            'module_name': type(backend_instance).__module__,
                            'params': params
                        }
        else:
            state['_has_virtual_workspace'] = False

        # Remove filemanager - will be recreated in worker process
        state.pop('filemanager', None)

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore context after unpickling (e.g., in worker process).

        Recreates the filemanager using the worker's local global registry,
        ensuring that all processes share the same memory backend instance
        within their own process space.

        Uses self-describing backend recreation: dynamically recreates all
        picklable backends based on stored class info and connection params.

        Args:
            state: Dictionary of state from __getstate__
        """
        import logging
        import os
        import importlib
        from pathlib import Path

        from polystore.base import storage_registry as global_storage_registry, ensure_storage_registry
        from polystore.filemanager import FileManager
        from polystore.zarr import ZarrStorageBackend
        from openhcs.constants.constants import Backend

        logger = logging.getLogger(__name__)

        # Extract preserved state
        zarr_config = state.pop('_zarr_config', None)
        plate_path = state.pop('_plate_path', None)
        has_virtual_workspace = state.pop('_has_virtual_workspace', False)
        picklable_backends = state.pop('_picklable_backends', {})

        # Restore all other attributes
        self.__dict__.update(state)

        # Ensure worker's registry is initialized
        ensure_storage_registry()

        # Override zarr backend with preserved config (same as orchestrator does)
        if zarr_config is not None:
            zarr_backend_with_config = ZarrStorageBackend(zarr_config)
            global_storage_registry[Backend.ZARR.value] = zarr_backend_with_config

        # Recreate virtual_workspace backend if it was registered in main process
        if has_virtual_workspace and plate_path is not None:
            try:
                from polystore.virtual_workspace import VirtualWorkspaceBackend
            except ImportError:
                logger.debug("VirtualWorkspaceBackend module not available in worker")
            else:
                try:
                    virtual_backend = VirtualWorkspaceBackend(plate_root=Path(plate_path))
                    global_storage_registry[Backend.VIRTUAL_WORKSPACE.value] = virtual_backend
                except (ValueError, TypeError, OSError) as e:
                    # Expected: Invalid config or missing directory
                    logger.warning(f"Failed to recreate virtual_workspace backend: {e}")
                except Exception as e:
                    # Unexpected: This is a bug
                    logger.error(f"BUG: Unexpected error recreating virtual_workspace backend: {e}", exc_info=True)

        # Self-describing backend recreation: dynamically recreate all picklable backends
        for backend_key, backend_info in picklable_backends.items():
            try:
                # Dynamically import the backend class
                module = importlib.import_module(backend_info['module_name'])
                backend_class = getattr(module, backend_info['class_name'])

                # Create backend instance
                backend_instance = backend_class()

                # Restore connection parameters
                backend_instance.set_connection_params(backend_info['params'])

                # Backend-specific connection logic
                if backend_info['class_name'] == 'OMEROLocalBackend':
                    # OMERO requires establishing connection
                    try:
                        from omero.gateway import BlitzGateway
                        password = os.getenv('OMERO_PASSWORD', 'openhcs')
                        params = backend_info['params']
                        conn = BlitzGateway(
                            params['username'],
                            password,
                            host=params['host'],
                            port=params['port']
                        )
                        if conn.connect():
                            backend_instance._initial_conn = conn
                            global_storage_registry[backend_key] = backend_instance
                            logger.info(f"✓ Recreated {backend_info['class_name']} in worker (connected to {params['host']}:{params['port']})")
                        else:
                            logger.warning(f"Failed to connect {backend_info['class_name']} in worker - backend not registered")
                    except Exception as e:
                        logger.warning(f"Failed to establish connection for {backend_info['class_name']}: {e}")
                else:
                    # Generic picklable backend - just register it
                    global_storage_registry[backend_key] = backend_instance
                    logger.info(f"✓ Recreated {backend_info['class_name']} in worker")

            except (ImportError, AttributeError, KeyError, ValueError, TypeError) as e:
                # Expected: Module not available, class not found, invalid params
                logger.warning(f"Failed to recreate backend '{backend_key}' ({backend_info.get('class_name', 'unknown')}): {e}")
            except Exception as e:
                # Unexpected: This is a bug
                logger.error(f"BUG: Unexpected error recreating backend '{backend_key}': {e}", exc_info=True)

        # Create filemanager using worker's local global registry
        # This ensures the worker uses its own memory backend instance
        # Use __dict__ directly to bypass frozen check
        self.__dict__['filemanager'] = FileManager(global_storage_registry)
