"""
Pipeline module for OpenHCS.

This module provides core pipeline compilation components for OpenHCS.
The PipelineCompiler is responsible for preparing step_plans within a ProcessingContext.

CONFIGURATION ACCESS PATTERN:
============================
The compiler uses ObjectState pattern for all configuration access:

✅ CORRECT (SAVED VALUES FOR COMPILATION):
    # Steps are registered in ObjectState with parent hierarchy: step → orchestrator → global
    step_state = ObjectState(object_instance=step, scope_id=scope_id, parent_state=orch_state)
    ObjectStateRegistry.register(step_state)

    # For compilation: use get_saved_resolved_value() to get saved values with inheritance
    # This ensures unsaved UI edits don't affect the compiled pipeline
    enabled = step_state.get_saved_resolved_value('streaming_defaults.enabled')
    var_comps = step_state.get_saved_resolved_value('processing_config.variable_components')

✅ CORRECT (LIVE VALUES FOR UI):
    # For UI: use get_resolved_value() to get current values with unsaved edits
    enabled = step_state.get_resolved_value('streaming_defaults.enabled')

❌ INCORRECT (LEGACY - REMOVED):
    with config_context(orchestrator.pipeline_config):  # REMOVED
        resolved_step = resolve_lazy_configurations_for_serialization(step)  # REMOVED

    # Using .parameters.get() doesn't get inheritance
    enabled = step_state.parameters.get('streaming_defaults.enabled')  # WRONG - no inheritance

    if hasattr(step, 'config_name'):  # REMOVED - use isinstance checks only
        config = getattr(step, 'config_name')  # REMOVED - use ObjectState.get_saved_resolved_value()

WHY:
- get_saved_resolved_value() provides saved baseline with inheritance (for compilation)
- get_resolved_value() provides live state with unsaved edits (for UI)
- parameters.get() returns raw local value only, NO inheritance
- No cross-step pollution - each step only sees its own config hierarchy
- isinstance checks are the only type checking pattern (no hasattr)
"""

from __future__ import annotations

import logging
import dataclasses
import time
from pathlib import Path
from typing import (
    Annotated,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Union,
    get_args,
    get_origin,
)

from openhcs.constants.constants import (
    get_multiprocessing_axis,
    OrchestratorState,
    VALID_GPU_MEMORY_TYPES,
    READ_BACKEND,
    WRITE_BACKEND,
    Backend,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.config import (
    MaterializationBackend,
    StreamingConfig,
    VFSConfig,
)
from openhcs.core.pipeline.funcstep_contract_validator import FuncStepContractValidator
from openhcs.core.pipeline.materialization_flag_planner import (
    MaterializationFlagPlanner,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan, InputConversionPlan
from openhcs.core.pipeline.path_planner import PipelinePathPlanner
from openhcs.core.pipeline.step_snapshot import (
    StepSnapshot,
    build_step_snapshots,
)
from openhcs.core.pipeline.gpu_memory_validator import GPUMemoryTypeValidator
from openhcs.core.pipeline.step_attribute_stripper import StepAttributeStripper
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.utils import WellFilterProcessor
from objectstate import ObjectState, ObjectStateRegistry
from objectstate.lazy_factory import get_base_type_for_lazy
from openhcs.core.steps.function_step import FunctionStep  # Used for isinstance check
from openhcs.core.progress import emit, ProgressPhase, ProgressStatus
from dataclasses import dataclass

if TYPE_CHECKING:
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


def _register_object_state(
    object_instance,
    scope_id: str,
    parent_state: Optional["ObjectState"],
) -> "ObjectState":
    """Create and register an ObjectState with the compiler's snapshot policy."""
    state = ObjectState(
        object_instance=object_instance,
        scope_id=scope_id,
        parent_state=parent_state,
    )
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    return state


def _get_or_register_object_state(
    scope_id: str,
    object_instance,
    parent_state: Optional["ObjectState"],
    *,
    force_fresh: bool = False,
) -> "ObjectState":
    """Return an existing ObjectState unless a fresh compiler state is required."""
    state = None if force_fresh else ObjectStateRegistry.get_by_scope(scope_id)
    if state is not None:
        return state
    return _register_object_state(object_instance, scope_id, parent_state)


_FUNCTION_REFERENCE_ATTRIBUTE_FIELDS = {
    "__name__": "function_name",
    "__module__": "original_module",
}


@dataclass(frozen=True)
class FunctionReference:
    """
    A picklable reference to a function in the registry.

    This replaces raw function objects in compiled step definitions to ensure
    picklability while allowing workers to resolve functions from their registry.

    Preserves all dunder attributes from the original function so they can be
    accessed during compilation (e.g., __artifact_inputs__, __artifact_outputs__).
    """

    function_name: str
    registry_name: str
    memory_type: str  # The memory type for get_function_by_name() (e.g., "numpy", "pyclesperanto")
    composite_key: str  # The full registry key (e.g., "pyclesperanto:gaussian_blur")
    original_module: str  # The original module path (e.g., "skimage.filters.edges")
    preserved_attrs: dict  # All dunder attributes from the original function (except __name__ and __module__)

    def __getattr__(self, name: str):
        """Allow access to preserved dunder attributes as if they were on the function."""
        # Use object.__getattribute__ to avoid infinite recursion
        preserved = object.__getattribute__(self, "preserved_attrs")

        if name in _FUNCTION_REFERENCE_ATTRIBUTE_FIELDS:
            return object.__getattribute__(
                self,
                _FUNCTION_REFERENCE_ATTRIBUTE_FIELDS[name],
            )

        if name in preserved:
            return preserved[name]
        raise AttributeError(f"FunctionReference has no attribute '{name}'")

    def resolve(self) -> Callable:
        """Resolve this reference to the actual decorated function from the registry.

        Always resolves through RegistryService to get the fully decorated function
        with all wrapper layers (memory type, slice-by-slice, dtype conversion, etc.).
        This should only be called in worker processes during execution, never during
        compilation (use preserved_attrs via __getattr__ instead).
        """
        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        all_functions = RegistryService.get_all_functions_with_metadata()
        if self.composite_key in all_functions:
            return all_functions[self.composite_key].func
        raise RuntimeError(
            f"Function {self.composite_key} not found in registry. "
            f"Ensure the function registry is initialized in this process."
        )


def _refresh_function_objects_in_steps(pipeline_definition: List[AbstractStep]) -> None:
    """
    Refresh all function objects in pipeline steps to ensure they're picklable.

    This recreates function objects by importing them fresh from their original modules,
    similar to how code mode works, which avoids unpicklable closures from registry wrapping.
    """
    logger.debug(f"🔄 FUNCTION REFRESH: Processing {len(pipeline_definition)} steps")
    for step_idx, step in enumerate(pipeline_definition):
        if isinstance(step, FunctionStep):
            if hasattr(step, "func") and step.func is not None:
                old_type = type(step.func).__name__
                step.func = _refresh_function_object(step.func)
                new_type = type(step.func).__name__

                # Log what's inside containers
                if isinstance(step.func, list) and step.func:
                    first_item = step.func[0]
                    first_item_type = type(first_item).__name__
                    if isinstance(first_item, tuple) and len(first_item) == 2:
                        inner_func_type = type(first_item[0]).__name__
                        logger.debug(
                            f"🔄 FUNCTION REFRESH: Step {step_idx} ({step.name}): {old_type} → {new_type} (first item: {first_item_type}, inner func: {inner_func_type})"
                        )
                    else:
                        logger.debug(
                            f"🔄 FUNCTION REFRESH: Step {step_idx} ({step.name}): {old_type} → {new_type} (first item: {first_item_type})"
                        )
                elif isinstance(step.func, tuple) and len(step.func) == 2:
                    func_type = type(step.func[0]).__name__
                    logger.debug(
                        f"🔄 FUNCTION REFRESH: Step {step_idx} ({step.name}): {old_type} → {new_type} (func: {func_type})"
                    )
                else:
                    logger.debug(
                        f"🔄 FUNCTION REFRESH: Step {step_idx} ({step.name}): {old_type} → {new_type}"
                    )
            else:
                logger.debug(
                    f"🔄 FUNCTION REFRESH: Step {step_idx} ({step.name}): No func attribute"
                )


def _refresh_function_object(func_value):
    """Convert function objects to picklable FunctionReference objects.

    Also filters out functions with enabled=False at compile time.
    """
    if callable(func_value) and hasattr(func_value, "__module__"):
        return _get_function_reference(func_value)

    elif isinstance(func_value, tuple) and len(func_value) == 2:
        func, params = func_value

        if isinstance(params, dict) and params.get("enabled", True) is False:
            return None

        if isinstance(params, dict) and "enabled" in params:
            params = {k: v for k, v in params.items() if k != "enabled"}

        if isinstance(params, dict) and "dtype_config" in params:
            params = {k: v for k, v in params.items() if k != "dtype_config"}

        if callable(func):
            func_ref = _refresh_function_object(func)
            return (func_ref, params)
        else:
            return (func, params)

    elif isinstance(func_value, list):
        refreshed = [_refresh_function_object(item) for item in func_value]
        return [item for item in refreshed if item is not None]

    elif isinstance(func_value, dict):
        refreshed = {
            key: _refresh_function_object(value) for key, value in func_value.items()
        }
        return {key: value for key, value in refreshed.items() if value is not None}

    return func_value


def _get_function_reference(func):
    """Convert a function to a picklable FunctionReference.

    Preserves custom attributes (like __artifact_inputs__, __artifact_outputs__)
    so they can be accessed during compilation without resolving the function.

    Compares unwrapped original functions to handle wrapper functions that may be
    different Python objects but wrap the same underlying callable.
    """
    from openhcs.processing.backends.lib_registry.registry_service import (
        RegistryService,
    )

    def _get_original_func(f):
        unwrapped = getattr(f, "__wrapped__", None)
        if unwrapped is not None:
            return _get_original_func(unwrapped)
        return f

    original_func = _get_original_func(func)
    original_name = original_func.__name__
    original_module = getattr(original_func, "__module__", "")

    all_functions = RegistryService.get_all_functions_with_metadata()

    for composite_key, metadata in all_functions.items():
        registry_original = _get_original_func(metadata.func)
        if (
            registry_original.__name__ == original_name
            and getattr(registry_original, "__module__", "") == original_module
        ):
            preserved_attrs = {}
            for attr in [
                "__artifact_inputs__",
                "__artifact_outputs__",
                "input_memory_type",
                "output_memory_type",
            ]:
                if hasattr(func, attr):
                    try:
                        preserved_attrs[attr] = getattr(func, attr)
                    except Exception:
                        pass

            return FunctionReference(
                function_name=original_name,
                registry_name=metadata.registry.library_name,
                memory_type=metadata.registry.MEMORY_TYPE,
                composite_key=composite_key,
                original_module=original_module,
                preserved_attrs=preserved_attrs,
            )

    raise RuntimeError(
        f"Function {original_name} (module: {original_module}) not found in registry - cannot create reference"
    )


class PipelineCompiler:
    """
    Compiles a pipeline by populating step plans within a ProcessingContext.

    This class provides static methods that are called sequentially by the
    PipelineOrchestrator for each well's ProcessingContext. Each method
    is responsible for a specific part of the compilation process, such as
    path planning, artifact I/O resolution, materialization flag setting,
    memory contract validation, and GPU resource assignment.
    """

    @staticmethod
    def initialize_step_plans_for_context(
        context: ProcessingContext,
        steps_definition: List[AbstractStep],
        orchestrator,
        metadata_writer: bool = False,
        plate_path: Optional[Path] = None,
        step_state_map: Dict[int, "ObjectState"] = None,
        step_snapshots: tuple[StepSnapshot, ...] | None = None,
        steps_already_resolved: bool = True,
        is_zmq_execution: bool = False,
        # base_input_dir and axis_id parameters removed, will use from context
    ) -> Tuple[List[AbstractStep], Dict[int, "ObjectState"]]:
        """
        Initializes step_plans by calling PipelinePathPlanner.prepare_pipeline_paths,
        which handles primary paths, artifact I/O path planning and linking, and chainbreaker status.
        Then, this method supplements the plans with non-I/O FunctionStep-specific attributes.

        Args:
            context: ProcessingContext to initialize step plans for
            steps_definition: List of AbstractStep objects defining the pipeline
            orchestrator: Orchestrator instance for well filter resolution
            metadata_writer: If True, this well is responsible for creating OpenHCS metadata files
            plate_path: Path to plate root for zarr conversion detection
            step_state_map: Pre-resolved ObjectState mapping from compile_pipelines one-time resolution
            steps_already_resolved: If True, steps are pre-resolved (default for performance)

        Returns:
            Tuple of (resolved steps, step_state_map)
        """
        # NOTE: This method is called within config_context() wrapper in compile_pipelines()
        if context.is_frozen():
            raise AttributeError(
                "Cannot initialize step plans in a frozen ProcessingContext."
            )

        if not hasattr(context, "step_plans") or context.step_plans is None:
            context.step_plans = {}  # Ensure step_plans dict exists

        # Visualizers read display configuration from display_config, not this field.
        context.visualizer_config = None

        # Steps are filtered in compile_pipelines() using ObjectState pattern
        # All steps must be properly registered in ObjectState for config resolution

        if step_snapshots is None and step_state_map is not None:
            step_snapshots = build_step_snapshots(steps_definition, step_state_map)

        # Pre-initialize step_plans with basic entries for each step
        # Use step index as key instead of step_id for multiprocessing compatibility
        for step_index, step in enumerate(steps_definition):
            snapshot = (
                step_snapshots[step_index]
                if step_snapshots is not None
                else None
            )
            if step_index not in context.step_plans:
                context.step_plans[step_index] = CompiledStepPlan(
                    step_index=step_index,
                    step_name=snapshot.name if snapshot else step.name,
                    step_type=snapshot.step_type if snapshot else step.__class__.__name__,
                    axis_id=context.axis_id,
                )

        # Resolve steps here when the caller did not provide a pre-resolved state map.
        if not steps_already_resolved or step_state_map is None:
            compilation_id = f"compile_{int(time.time() * 1000)}"

            # === IPC FIX: Register global config for cross-process inheritance ===
            from objectstate import get_current_global_config
            from openhcs.core.config import GlobalPipelineConfig

            global_config_state = ObjectStateRegistry.get_by_scope("")
            if global_config_state is None:
                global_config = get_current_global_config(GlobalPipelineConfig)
                if global_config:
                    global_config_state = _register_object_state(
                        global_config,
                        "",
                        None,
                    )
                    logger.info(
                        "Registered global config at scope '' (initialize_step_plans)"
                    )

            # Register orchestrator with PipelineConfig as parent for config inheritance
            # The orchestrator provides pipeline-level config (streaming_defaults, etc.)
            orch_scope_id = f"{compilation_id}::orchestrator"
            orch_state = _register_object_state(
                orchestrator,
                orch_scope_id,
                global_config_state,
            )
            logger.info(
                f"Registered orchestrator at scope: {orch_scope_id}"
            )

            # Register each step with orchestrator as parent
            # Each step only sees: itself → orchestrator → global (NOT other steps)
            step_state_map = {}
            for step_index, step in enumerate(steps_definition):
                step_scope_id = f"{compilation_id}::{context.plate_path or 'plate'}::step_{step_index}"
                step_state = _register_object_state(
                    step,
                    step_scope_id,
                    orch_state,
                )
                step_state_map[step_index] = step_state

            # Now resolve all steps using their ObjectStates
            resolved_steps = []
            for step_index, step in enumerate(steps_definition):
                step_state = step_state_map[step_index]
                logger.info(
                    f"Resolving step {step_index} ('{step.name}') from ObjectState."
                )
                resolved_step = step_state.to_object()
                resolved_steps.append(resolved_step)

            # Cleanup compiler-created ObjectStates.
            # IMPORTANT:
            # - UI/editor mode: do NOT unregister (GUI relies on these registered states).
            # - ZMQ execution server: DO unregister to free RAM.
            if is_zmq_execution:
                ObjectStateRegistry.unregister(orch_state, _skip_snapshot=True)
                for step_index, step_state in step_state_map.items():
                    ObjectStateRegistry.unregister(step_state, _skip_snapshot=True)

            steps_definition = resolved_steps
            _refresh_function_objects_in_steps(steps_definition)
            step_snapshots = build_step_snapshots(steps_definition, step_state_map)
            logger.info(
                f"Resolved {len(resolved_steps)} steps under scope: {compilation_id}"
            )
        else:
            # Steps already resolved in compile_pipelines - just use them directly
            logger.debug(
                f"Using pre-resolved steps for context {context.axis_id}"
            )
            if step_snapshots is None:
                step_snapshots = build_step_snapshots(
                    steps_definition,
                    step_state_map,
                )

        # === INPUT CONVERSION DETECTION ===
        # Check if first step needs zarr conversion
        if steps_definition and plate_path:
            first_step = steps_definition[0]
            # Access config from merged config (pipeline + global) for proper inheritance
            vfs_config = orchestrator.get_effective_config().vfs_config

            # Only convert if default materialization backend is ZARR
            wants_zarr_conversion = (
                vfs_config.materialization_backend == MaterializationBackend.ZARR
            )

            if wants_zarr_conversion:
                # Check if input plate is already zarr format
                available_backends = context.microscope_handler.get_available_backends(
                    plate_path
                )
                already_zarr = Backend.ZARR in available_backends

                if not already_zarr:
                    # Determine if input uses virtual workspace
                    from openhcs.microscopes.openhcs import OpenHCSMetadataHandler
                    from polystore.metadata_writer import get_subdirectory_name

                    openhcs_metadata_handler = OpenHCSMetadataHandler(
                        context.filemanager
                    )
                    metadata = openhcs_metadata_handler._load_metadata_dict(plate_path)
                    subdirs = metadata["subdirectories"]

                    # Get actual subdirectory from input_dir
                    original_subdir = get_subdirectory_name(
                        context.input_dir, plate_path
                    )
                    uses_virtual_workspace = (
                        Backend.VIRTUAL_WORKSPACE.value
                        in subdirs[original_subdir]["available_backends"]
                    )

                    zarr_subdir = "zarr" if uses_virtual_workspace else original_subdir
                    conversion_dir = plate_path / zarr_subdir

                    context.step_plans[0].input_conversion = InputConversionPlan(
                        output_dir=conversion_dir,
                        backend=MaterializationBackend.ZARR.value,
                        uses_virtual_workspace=uses_virtual_workspace,
                        original_subdir=original_subdir,
                    )
                    logger.debug(
                        f"Input conversion to zarr enabled for first step: {first_step.name}"
                    )

        # The axis_id and base_input_dir are available from the context object.

        # === PATH PLANNING ===
        # CRITICAL: Pass merged config (not raw pipeline_config) for proper global config inheritance
        # This ensures path_planning_config and vfs_config inherit from global config
        # CRITICAL: Pass step_state_map so path planner can resolve lazy dataclass attributes via ObjectState
        PipelinePathPlanner.prepare_pipeline_paths(
            context,
            steps_definition,
            context.global_config,  # Use merged config from context instead of raw pipeline_config
            orchestrator=orchestrator,
            step_state_map=step_state_map,
            step_snapshots=step_snapshots,
        )

        # NOTE: Function object refresh is now done ONCE at the top level after resolving steps
        # (see compile_pipelines_for_plate() line ~1310)
        # This ensures ObjectState.to_object() restored functions are converted to FunctionReference
        # before any per-well compilation, avoiding redundant conversions

        # Loop to supplement step_plans with non-I/O, non-path attributes
        # after PipelinePathPlanner has fully populated them with I/O info.
        for step_index, step in enumerate(steps_definition):
            snapshot = step_snapshots[step_index]
            if step_index not in context.step_plans:
                logger.error(
                    f"Critical error: Step {snapshot.name} (index: {step_index}) "
                    f"not found in step_plans after path planning phase."
                )
                # Create a minimal error plan
                context.step_plans[step_index] = CompiledStepPlan(
                    step_index=step_index,
                    step_name=snapshot.name,
                    step_type=snapshot.step_type,
                    axis_id=context.axis_id,
                    error="Missing from path planning phase by PipelinePathPlanner",
                    create_openhcs_metadata=metadata_writer,
                )
                continue

            current_plan = context.step_plans[step_index]

            # Ensure basic metadata (PathPlanner should set most of this)
            current_plan.step_name = snapshot.name
            current_plan.step_type = snapshot.step_type
            current_plan.axis_id = context.axis_id
            current_plan.create_openhcs_metadata = metadata_writer

            current_plan.variable_components = snapshot.variable_components
            current_plan.group_by = snapshot.group_by
            current_plan.input_source = snapshot.input_source
            current_plan.sequential_processing = snapshot.processing_config

        # === STREAMING CONFIG COLLECTION ===
        # Discover streaming configs attached to each step via dataclass field types.
        # For compilation: read ONLY from ObjectState.get_saved_resolved_value('<dotted.path>').

        if not hasattr(context, "required_visualizers"):
            context.required_visualizers = []

        # Compiler policy: access all attributes via ObjectState.get_saved_resolved_value
        # Minimal, deterministic access pattern: read every required nested attribute
        # directly from the ObjectState flattened snapshot using dotted paths.

        # Helper: reconstruct dataclass instance from ObjectState using dotted-path reads only
        def _rebuild_dataclass_from_objectstate(
            config_cls, step_state, root_field_name
        ):
            kwargs = {}
            for f in dataclasses.fields(config_cls):
                dotted = f"{root_field_name}.{f.name}"
                val = step_state.get_saved_resolved_value(dotted)

                # If value is None, but the field type is a dataclass (or Optional[...] of dataclass),
                # attempt recursive reconstruction from nested dotted paths.
                candidate = None
                origin = get_origin(f.type)
                if origin is Annotated:
                    candidate = get_args(f.type)[0]
                elif origin is Union:
                    for a in get_args(f.type):
                        if a is type(None):
                            continue
                        if dataclasses.is_dataclass(a):
                            candidate = a
                            break
                else:
                    candidate = f.type

                if (
                    val is None
                    and candidate is not None
                    and dataclasses.is_dataclass(candidate)
                ):
                    val = _rebuild_dataclass_from_objectstate(
                        candidate, step_state, dotted
                    )

                kwargs[f.name] = val

            return config_cls(**kwargs)

        registry_keys = list(StreamingConfig.__registry__.keys())
        for step_index, step_state in step_state_map.items():
            step_plan = context.step_plans[step_index]
            for field_name in registry_keys:
                # Enable semantics:
                # - If streaming_defaults.enabled is True, enable all streaming configs for the step
                # - Otherwise use the per-stream config enabled flag

                defaults_enabled = step_state.get_saved_resolved_value(
                    "streaming_defaults.enabled"
                )
                per_stream_enabled = step_state.get_saved_resolved_value(
                    f"{field_name}.enabled"
                )
                enabled = True if defaults_enabled is True else per_stream_enabled
                if is_zmq_execution:
                    logger.info(
                        "Streaming resolution: step=%s field=%s defaults_enabled=%r per_stream_enabled=%r effective_enabled=%r",
                        step_index,
                        field_name,
                        defaults_enabled,
                        per_stream_enabled,
                        enabled,
                    )

                if enabled is True:
                    base_cls = get_base_type_for_lazy(
                        StreamingConfig.__registry__[field_name]
                    )
                    config_obj = _rebuild_dataclass_from_objectstate(
                        base_cls, step_state, field_name
                    )
                    backend_name = step_state.get_saved_resolved_value(
                        f"{field_name}.backend"
                    )
                    visualizer_info = {"backend": backend_name, "config": config_obj}
                    if visualizer_info not in context.required_visualizers:
                        context.required_visualizers.append(visualizer_info)
                        logger.info(
                            f"Streaming enabled for step {step_index}, field {field_name} (backend={backend_name})"
                        )

                    # IMPORTANT: FunctionStep streams by scanning step_plan for StreamingConfig instances.
                    # Inject the reconstructed StreamingConfig instance into the step plan so workers
                    # can execute streaming via filemanager.save_batch(..., backend='napari_stream'/'fiji_stream').
                    step_plan.streaming_configs[field_name] = config_obj

        # Return resolved steps and step_state_map for use by subsequent compiler methods
        return steps_definition, step_state_map

    # _prepare_materialization_flags is removed as MaterializationFlagPlanner.prepare_pipeline_flags
    # now modifies context.step_plans in-place and takes context directly.

    @staticmethod
    def declare_zarr_stores_for_context(
        context: ProcessingContext, steps_definition: List[AbstractStep], orchestrator
    ) -> None:
        """
        Declare zarr store creation functions for runtime execution.

        This method runs after path planning but before materialization flag planning
        to declare which steps need zarr stores and provide the metadata needed
        for runtime store creation.

        Args:
            context: ProcessingContext for current well
            steps_definition: List of AbstractStep objects
            orchestrator: Orchestrator instance for accessing all wells
        """

        all_wells = orchestrator.get_component_keys(get_multiprocessing_axis())

        # Access config from merged config (pipeline + global) for proper inheritance
        vfs_config = orchestrator.get_effective_config().vfs_config

        for step_index, step in enumerate(steps_definition):
            step_plan = context.step_plans[step_index]

            will_use_zarr = (
                vfs_config.materialization_backend == MaterializationBackend.ZARR
                and step_index == len(steps_definition) - 1
            )

            if will_use_zarr:
                step_plan.zarr_config = {
                    "all_wells": all_wells,
                    "needs_initialization": True,
                }
                logger.debug(
                    f"Step '{step.name}' will use zarr backend for axis {context.axis_id}"
                )
            else:
                step_plan.zarr_config = None

    @staticmethod
    def plan_materialization_flags_for_context(
        context: ProcessingContext, steps_definition: List[AbstractStep], orchestrator
    ) -> None:
        """
        Plans and injects materialization flags into context.step_plans
        by calling MaterializationFlagPlanner.
        """
        if context.is_frozen():
            raise AttributeError(
                "Cannot plan materialization flags in a frozen ProcessingContext."
            )
        if not context.step_plans:
            logger.warning(
                "step_plans is empty in context for materialization planning. This may be valid if pipeline is empty."
            )
            return

        # MaterializationFlagPlanner.prepare_pipeline_flags now takes context and pipeline_definition
        # and modifies context.step_plans in-place.
        # CRITICAL: Pass merged config (not raw pipeline_config) for proper global config inheritance
        MaterializationFlagPlanner.prepare_pipeline_flags(
            context,
            steps_definition,
            orchestrator.plate_path,
            context.global_config,  # Use merged config from context instead of raw pipeline_config
        )

        # Post-check (optional, but good for ensuring contracts are met by the planner)
        for step_index, step in enumerate(steps_definition):
            if step_index not in context.step_plans:
                # This should not happen if prepare_pipeline_flags guarantees plans for all steps
                logger.error(
                    f"Step {step.name} (index: {step_index}) missing from step_plans after materialization planning."
                )
                continue

            plan = context.step_plans[step_index]
            # Check for keys that FunctionStep actually uses during execution
            missing_keys = [
                name
                for name in [READ_BACKEND, WRITE_BACKEND]
                if getattr(plan, name) is None
            ]
            if missing_keys:
                logger.error(
                    f"Materialization flag planning incomplete for step {step.name} (index: {step_index}). "
                    f"Missing required keys: {missing_keys}."
                )

    @staticmethod
    def validate_sequential_components_compatibility(
        steps_definition: List[AbstractStep],
        sequential_components: List,
        step_state_map: Dict[int, "ObjectState"],
    ) -> None:
        """
        Validate that no step's variable_components overlap with pipeline's sequential_components.

        Args:
            steps_definition: List of AbstractStep objects
            sequential_components: List of SequentialComponents from pipeline config
            step_state_map: Map of step index to ObjectState for accessing config values

        Raises:
            ValueError: If any step has variable_components that overlap with sequential_components
        """
        if not sequential_components:
            return

        seq_comp_values = {sc.value for sc in sequential_components}

        for step_index, step in enumerate(steps_definition):
            if isinstance(step, FunctionStep):
                step_objectstate = step_state_map.get(step_index)
                if step_objectstate is None:
                    raise ValueError(
                        f"Step {step_index} ('{step.name}') not found in step_state_map"
                    )

                var_comps = step_objectstate.get_saved_resolved_value(
                    "processing_config.variable_components"
                )
                if var_comps:
                    var_comp_values = {vc.value for vc in var_comps}
                    overlap = seq_comp_values & var_comp_values

                    if overlap:
                        raise ValueError(
                            f"Step '{step.name}' has variable_components {sorted(overlap)} that conflict with "
                            f"pipeline's sequential_components {sorted(seq_comp_values)}. "
                            f"A component cannot be both sequential (pipeline-level) and variable (step-level). "
                            f"Either remove {sorted(overlap)} from step's variable_components or from "
                            f"pipeline's sequential_components."
                        )

    @staticmethod
    def analyze_pipeline_sequential_mode(
        context: ProcessingContext,
        global_config: GlobalPipelineConfig,
        orchestrator: PipelineOrchestrator,
    ) -> None:
        """
        Configure pipeline-wide sequential processing mode from pipeline-level config.
        Precomputes sequential combinations at compile time.

        Args:
            context: ProcessingContext to configure
            global_config: GlobalPipelineConfig containing SequentialProcessingConfig
            orchestrator: PipelineOrchestrator with microscope handler for pattern discovery
        """
        if context.is_frozen():
            raise AttributeError(
                "Cannot analyze pipeline sequential mode in a frozen ProcessingContext."
            )

        # Get pipeline-level sequential processing config
        seq_config = global_config.sequential_processing_config

        if seq_config and seq_config.sequential_components:
            # Enable pipeline-wide sequential mode
            context.pipeline_sequential_mode = True
            seq_comps = tuple(sc.value for sc in seq_config.sequential_components)

            # Precompute combinations from orchestrator's component keys cache
            # This cache is populated from filename parsing during init, so it's always available
            from openhcs.constants import AllComponents
            import itertools

            # Extract component values from orchestrator's cache for each sequential component
            # Filter out components with only 1 value (no point in sequential processing)
            component_values_lists = []
            filtered_seq_comps = []

            for seq_comp in seq_comps:
                # Convert component name to AllComponents enum
                component_enum = AllComponents(seq_comp)

                # Get component values from orchestrator's cache (populated from filename parsing)
                component_values = orchestrator.get_component_keys(component_enum)

                if not component_values:
                    logger.warning(f"No {seq_comp} values found in orchestrator cache")
                    component_values_lists.append([])
                elif len(component_values) == 1:
                    logger.info(
                        f"Sequential component '{seq_comp}' has only 1 value - ignoring for sequential processing"
                    )
                else:
                    # Only include components with multiple values
                    component_values_lists.append(component_values)
                    filtered_seq_comps.append(seq_comp)
                    logger.debug(
                        f"Sequential component '{seq_comp}': {len(component_values)} values from cache"
                    )

            # Generate all combinations using Cartesian product
            if component_values_lists and all(component_values_lists):
                combinations = list(itertools.product(*component_values_lists))
                context.pipeline_sequential_combinations = combinations
                logger.info(
                    f"Pipeline sequential mode: ENABLED (components: {tuple(filtered_seq_comps)}, "
                    f"combinations: {len(combinations)})"
                )
            else:
                # No components with multiple values - disable sequential mode
                context.pipeline_sequential_mode = False
                context.pipeline_sequential_combinations = None
                logger.info(
                    "Pipeline sequential mode: DISABLED (all sequential components have ≤1 value)"
                )
        else:
            # No sequential processing configured
            context.pipeline_sequential_mode = False
            context.pipeline_sequential_combinations = None
            logger.debug(
                "Pipeline sequential mode: DISABLED (no sequential components configured)"
            )

    @staticmethod
    def validate_memory_contracts_for_context(
        context: ProcessingContext,
        steps_definition: List[AbstractStep],
        step_state_map: Dict[int, "ObjectState"],
        orchestrator=None,
    ) -> None:
        """
        Validates FunctionStep memory contracts, dict patterns, and adds memory type info to context.step_plans.

        Args:
            context: ProcessingContext to validate
            steps_definition: List of AbstractStep objects
            step_state_map: Map of step index to ObjectState for accessing config values
            orchestrator: Optional orchestrator for dict pattern key validation
        """
        if context.is_frozen():
            raise AttributeError(
                "Cannot validate memory contracts in a frozen ProcessingContext."
            )

        FuncStepContractValidator.validate_pipeline(
            steps=steps_definition,
            pipeline_context=context,  # Pass context so validator can access step plans for memory type overrides
            step_state_map=step_state_map,  # Pass step_state_map for accessing config via ObjectState
            orchestrator=orchestrator,  # Pass orchestrator for dict pattern key validation
        )

        for step_index, step in enumerate(steps_definition):
            if not isinstance(step, FunctionStep):
                continue
            if step_index not in context.step_plans:
                raise AssertionError(
                    f"Memory validation requires a compiled plan for FunctionStep {step.name} (index: {step_index})."
                )
            step_plan = context.step_plans[step_index]
            missing_fields = [
                field_name
                for field_name in ["input_memory_type", "output_memory_type", "func"]
                if getattr(step_plan, field_name) is None
            ]
            if missing_fields:
                raise AssertionError(
                    f"Memory validation must set {missing_fields} for FunctionStep {step_plan.step_name} (index: {step_index})."
                )

        # Apply memory type override: Any step with disk output must use numpy for disk writing
        for step_index, step in enumerate(steps_definition):
            if isinstance(step, FunctionStep):
                if step_index in context.step_plans:
                    step_plan = context.step_plans[step_index]
                    write_backend = step_plan.write_backend

                    if write_backend == "disk":
                        logger.debug(
                            f"Step {step.name} has disk output, overriding output_memory_type to numpy"
                        )
                        step_plan.output_memory_type = "numpy"

    @staticmethod
    def assign_gpu_resources_for_context(context: ProcessingContext) -> None:
        """
        Validates GPU memory types from context.step_plans and assigns GPU device IDs.
        """
        if context.is_frozen():
            raise AttributeError(
                "Cannot assign GPU resources in a frozen ProcessingContext."
            )

        GPUMemoryTypeValidator.validate_step_plans(context.step_plans)

        for step_index, step_plan_val in context.step_plans.items():
            is_gpu_step = False
            input_type = step_plan_val.input_memory_type
            if input_type in VALID_GPU_MEMORY_TYPES:
                is_gpu_step = True

            output_type = step_plan_val.output_memory_type
            if output_type in VALID_GPU_MEMORY_TYPES:
                is_gpu_step = True

            if is_gpu_step:
                if step_plan_val.gpu_id is None:
                    step_name = step_plan_val.step_name
                    raise AssertionError(
                        f"GPU validation must assign gpu_id for step {step_name} (index: {step_index}) "
                        f"with GPU memory types."
                    )

    @staticmethod
    def apply_global_visualizer_override_for_context(
        context: ProcessingContext, global_enable_visualizer: bool
    ) -> None:
        """
        Applies global visualizer override to all step_plans in the context.
        """
        if context.is_frozen():
            raise AttributeError(
                "Cannot apply visualizer override in a frozen ProcessingContext."
            )

        if global_enable_visualizer:
            if not context.step_plans:
                return  # Guard against empty step_plans
            for step_index, plan in context.step_plans.items():
                plan.visualize = True
                logger.info(
                    f"Global visualizer override: Step '{plan.step_name}' marked for visualization."
                )

    @staticmethod
    def resolve_lazy_dataclasses_for_context(
        context: ProcessingContext,
        orchestrator,
        steps_definition: List[AbstractStep],
        step_state_map: Dict[int, "ObjectState"] = None,
    ) -> None:
        """
        Resolve all lazy dataclass instances in step plans to their base configurations.

        This method uses ObjectState for resolution.
        All configs are already resolved via ObjectState.to_object() during compilation.
        This method now just ensures step plans reference the resolved configs.

        Args:
            context: ProcessingContext to process
            orchestrator: PipelineOrchestrator (unused - kept for API compatibility)
            steps_definition: List of resolved step objects
            step_state_map: Map of step_index to ObjectState for parameter access
        """
        # Configs are already resolved via ObjectState.to_object() in initialize_step_plans_for_context
        # No additional resolution needed - step plans already contain resolved configs
        logger.debug(
            f"Step plans already resolved via ObjectState for {len(steps_definition)} steps"
        )

    @staticmethod
    def validate_backend_compatibility(orchestrator) -> None:
        """
        Validate configured read backend against microscope support.

        Materialization backend selection is always allowed at compile time (e.g. materialize
        to Zarr even when source data is read from disk). What must be compatible with the
        selected microscope is the backend used for reading input images.

        Args:
            orchestrator: PipelineOrchestrator instance with initialized microscope_handler
        """

        microscope_handler = orchestrator.microscope_handler

        # Read saved resolved vfs_config.read_backend from ObjectState (not live UI edits)
        plate_scope_id = str(orchestrator.plate_path)
        pipeline_config_state = ObjectStateRegistry.get_by_scope(plate_scope_id)
        if pipeline_config_state is not None:
            configured_read_backend = pipeline_config_state.get_saved_resolved_value(
                "vfs_config.read_backend"
            )
        else:
            # Fallback: if no ObjectState exists (unexpected in compiler path),
            # use the effective merged config.
            vfs_config = orchestrator.get_effective_config().vfs_config or VFSConfig()
            configured_read_backend = vfs_config.read_backend

        # AUTO/None means "let the microscope handler decide".
        if configured_read_backend in (None, Backend.AUTO):
            return

        # Normalize to Backend enum
        if isinstance(configured_read_backend, Backend):
            read_backend = configured_read_backend
        else:
            try:
                read_backend = Backend(str(configured_read_backend))
            except Exception:
                raise ValueError(
                    f"Invalid vfs_config.read_backend={configured_read_backend!r}. "
                    f"Expected one of: {[b.value for b in Backend]}."
                )

        available_backends = microscope_handler.get_available_backends(
            orchestrator.input_dir or orchestrator.plate_path
        )
        if read_backend not in available_backends:
            raise ValueError(
                f"{microscope_handler.microscope_type} does not support read_backend={read_backend.value}. "
                f"Supported backends for this plate: {[b.value for b in available_backends]}. "
                "Update vfs_config.read_backend (or set it to 'auto') and recompile."
            )

    @staticmethod
    def compile_pipelines(
        orchestrator,
        pipeline_definition: List[AbstractStep],
        axis_filter: Optional[List[str]] = None,
        enable_visualizer_override: bool = False,
        is_zmq_execution: bool = False,
    ) -> Dict[str, ProcessingContext]:
        """
        Compile-all phase: Prepares frozen ProcessingContexts for each axis value.

        This method iterates through specified axis values, creates a ProcessingContext
        for each, and invokes various phases of PipelineCompiler to populate
        context's step_plans. After all compilation phases for an axis value are complete,
        its context is frozen. Finally, attributes are stripped from the pipeline_definition,
        making step objects stateless for execution phase.

        Args:
            orchestrator: The PipelineOrchestrator instance to use for compilation
            pipeline_definition: The list of AbstractStep objects defining the pipeline.
            axis_filter: Optional list of axis values to process. If None, processes all found axis values.
            enable_visualizer_override: If True, all steps in all compiled contexts
                                        will have their 'visualize' flag set to True.
            is_zmq_execution: If True, compiler-created ObjectStates will be unregistered
                              after resolution to free RAM (for ZMQ server mode).

        Returns:
            A dictionary mapping axis values to their compiled and frozen ProcessingContexts.
            The input `pipeline_definition` list (of step objects) is modified in-place
            to become stateless.
        """

        if not orchestrator.is_initialized():
            raise RuntimeError(
                "PipelineOrchestrator must be explicitly initialized before calling compile_pipelines()."
            )

        if not pipeline_definition:
            raise ValueError(
                "A valid pipeline definition (List[AbstractStep]) must be provided."
            )

        try:
            compiled_contexts: Dict[str, ProcessingContext] = {}
            # Get multiprocessing axis values dynamically from configuration

            # CRITICAL: Resolve well_filter_config from merged config (pipeline + global)
            # This allows global-level well filtering to work (e.g., well_filter_config.well_filter = 1)
            # Must use get_effective_config() to get merged config, not raw pipeline_config
            resolved_axis_filter = axis_filter
            effective_config = orchestrator.get_effective_config()
            if effective_config and hasattr(effective_config, "well_filter_config"):
                well_filter_config = effective_config.well_filter_config
                if (
                    well_filter_config
                    and hasattr(well_filter_config, "well_filter")
                    and well_filter_config.well_filter is not None
                ):
                    available_wells = orchestrator.get_component_keys(
                        get_multiprocessing_axis()
                    )
                    resolved_wells = WellFilterProcessor.resolve_filter_with_mode(
                        well_filter_config.well_filter,
                        well_filter_config.well_filter_mode,
                        available_wells,
                    )
                    logger.info(
                        f"Well filter: {well_filter_config.well_filter} (mode={well_filter_config.well_filter_mode.value}) "
                        f"→ {len(resolved_wells)} wells to process: {resolved_wells}"
                    )

                    # If axis_filter was also provided, intersect them
                    if axis_filter:
                        resolved_axis_filter = [
                            w for w in resolved_wells if w in axis_filter
                        ]
                        logger.info(
                            f"Intersected with axis_filter: {len(resolved_axis_filter)} wells remain"
                        )
                    else:
                        resolved_axis_filter = resolved_wells

            axis_values_to_process = orchestrator.get_component_keys(
                get_multiprocessing_axis(), resolved_axis_filter
            )

            if not axis_values_to_process:
                logger.warning("No axis values found to process based on filter.")
                return {
                    "pipeline_definition": pipeline_definition,
                    "compiled_contexts": {},
                }

            logger.info(
                f"Starting compilation for axis values: {', '.join(axis_values_to_process)}"
            )

            # === ONE-TIME STEP RESOLUTION ===
            # Resolve steps ONCE per pipeline, not once per well.
            # Register persistent ObjectStates for the entire compilation.

            # === IPC FIX: Register persistent ObjectStates for cross-process inheritance ===
            from objectstate import get_current_global_config
            from openhcs.core.config import GlobalPipelineConfig

            # In ZMQ execution mode, always overwrite compile-time ObjectStates from the
            # current request to prevent stale pipeline/config reuse across requests.
            force_fresh_compile_states = bool(is_zmq_execution)

            global_config_state = ObjectStateRegistry.get_by_scope("")
            if force_fresh_compile_states or global_config_state is None:
                global_config = get_current_global_config(
                    GlobalPipelineConfig, use_live=False
                )
                if global_config:
                    global_config_state = _register_object_state(
                        global_config,
                        "",
                        None,
                    )
                    logger.debug("Registered global config at scope ''")

            # Register the orchestrator's pipeline_config at plate_path scope
            plate_path_str = str(orchestrator.plate_path)
            plate_orch_state = ObjectStateRegistry.get_by_scope(plate_path_str)
            if orchestrator.pipeline_config:
                plate_orch_state = _get_or_register_object_state(
                    plate_path_str,
                    orchestrator.pipeline_config,
                    global_config_state,
                    force_fresh=force_fresh_compile_states,
                )
                logger.debug(f"Registered pipeline_config at scope '{plate_path_str}'")

            # Register orchestrator ObjectState (for delegation pattern)
            # Use proper scope hierarchy: plate_path::orchestrator
            orch_scope_id = f"{plate_path_str}::orchestrator"
            orch_state = _get_or_register_object_state(
                orch_scope_id,
                orchestrator,
                plate_orch_state,
                force_fresh=force_fresh_compile_states,
            )
            logger.debug(f"Registered orchestrator at scope: {orch_scope_id}")

            # Register step ObjectStates (persistent for entire compilation)
            step_state_map = {}
            for step_index, step in enumerate(pipeline_definition):
                step_scope_id = f"{plate_path_str}::step_{step_index}"
                step_state = _get_or_register_object_state(
                    step_scope_id,
                    step,
                    orch_state,
                    force_fresh=force_fresh_compile_states,
                )
                step_state_map[step_index] = step_state

            # Resolve steps ONCE using their ObjectStates
            # ARCHITECTURAL FIX: Replace pipeline_definition in-place with resolved steps
            # This ensures there's only ONE list of steps used throughout compilation
            pipeline_definition.clear()
            for step_index, step_state in step_state_map.items():
                resolved_step = step_state.to_object()
                pipeline_definition.append(resolved_step)

            logger.debug(
                f"Resolved {len(pipeline_definition)} steps once per pipeline (replaced original list in-place)"
            )

            # CRITICAL: Refresh function objects immediately after resolving steps
            # ObjectState.to_object() restores original .func attributes (raw functions)
            # We must convert them to FunctionReference BEFORE any per-well compilation
            _refresh_function_objects_in_steps(pipeline_definition)
            logger.debug(
                f"Refreshed function objects in {len(pipeline_definition)} steps (converted to FunctionReference)"
            )

            step_snapshots = build_step_snapshots(
                pipeline_definition,
                step_state_map,
            )
            enabled_pairs = [
                (step, step_state_map[snapshot.index])
                for snapshot, step in zip(step_snapshots, pipeline_definition)
                if snapshot.enabled
            ]

            pipeline_definition.clear()
            pipeline_definition.extend(step for step, _state in enabled_pairs)
            step_state_map = {
                new_index: state
                for new_index, (_step, state) in enumerate(enabled_pairs)
            }
            if not pipeline_definition:
                logger.warning(
                    "All steps were disabled. Pipeline is empty after filtering."
                )
                return {
                    "pipeline_definition": pipeline_definition,
                    "compiled_contexts": {},
                }
            step_snapshots = build_step_snapshots(
                pipeline_definition,
                step_state_map,
            )

            # === END ONE-TIME STEP RESOLUTION ===
            # NOTE: ObjectStates remain registered for use by streaming config resolution

            # Capture config values at compile time from PipelineConfig scope
            pipeline_config_state = ObjectStateRegistry.get_by_scope(plate_path_str)
            if pipeline_config_state is None:
                raise RuntimeError(
                    "Missing ObjectState for plate; cannot resolve pipeline config."
                )

            # Get the complete resolved AnalysisConsolidationConfig with all fields populated
            # get_saved_resolved_value() automatically reconstructs dataclass containers
            lazy_analysis_config = pipeline_config_state.get_saved_resolved_value(
                "analysis_consolidation_config"
            )
            # Convert lazy config to base type for pickling in multiprocessing
            from objectstate.lazy_factory import LazyDataclass

            analysis_consolidation_config = (
                lazy_analysis_config.to_base_config()
                if isinstance(lazy_analysis_config, LazyDataclass)
                else lazy_analysis_config
            )

            # Get auto_add_output_plate flag directly (it's a top-level field, not a dataclass)
            auto_add_output_plate = pipeline_config_state.get_saved_resolved_value(
                "auto_add_output_plate_to_plate_manager"
            )

            # === BACKEND COMPATIBILITY VALIDATION ===
            # Validate that configured backend is compatible with microscope
            # For microscopes with only one compatible backend (e.g., OMERO), auto-set it
            PipelineCompiler.validate_backend_compatibility(orchestrator)

            # === GLOBAL AXIS FILTER RESOLUTION ===
            # Use ObjectState pattern to resolve axis filters
            # Steps will be registered in ObjectState during initialize_step_plans_for_context
            # For now, create a temporary registration to resolve filters before compilation

            # Create a temporary context to store the global axis filters
            temp_context = orchestrator.create_context("temp")

            # Resolve axis filters from the one-time StepSnapshot tuple.
            _resolve_step_axis_filters(
                step_snapshots,
                temp_context,
                orchestrator,
            )
            global_step_axis_filters = getattr(temp_context, "step_axis_filters", {})

            # Determine responsible axis value for metadata creation (lexicographically first)
            responsible_axis_value = (
                sorted(axis_values_to_process)[0] if axis_values_to_process else None
            )

            # Track compilation progress
            total_axis_values = len(axis_values_to_process)
            completed_axis_values = 0

            for axis_id in axis_values_to_process:
                # Determine if this axis value is responsible for metadata creation
                is_responsible = axis_id == responsible_axis_value

                # Create a temporary context to check if sequential mode is enabled
                temp_context = orchestrator.create_context(axis_id)
                temp_context.step_axis_filters = global_step_axis_filters

                # Initialize step plans first to get step_state_map for validation
                # Use pre-resolved steps and step_state_map for performance
                resolved_steps, step_state_map = (
                    PipelineCompiler.initialize_step_plans_for_context(
                        temp_context,
                        pipeline_definition,  # Now using the in-place replaced list
                        orchestrator,
                        metadata_writer=is_responsible,
                        plate_path=orchestrator.plate_path,
                        step_state_map=step_state_map,
                        step_snapshots=step_snapshots,
                        steps_already_resolved=True,
                        is_zmq_execution=is_zmq_execution,
                    )
                )

                # Validate sequential components compatibility BEFORE analyzing sequential mode
                seq_config = temp_context.global_config.sequential_processing_config
                if seq_config and seq_config.sequential_components:
                    PipelineCompiler.validate_sequential_components_compatibility(
                        resolved_steps, seq_config.sequential_components, step_state_map
                    )

                # Analyze sequential mode to get combinations (doesn't freeze context)
                PipelineCompiler.analyze_pipeline_sequential_mode(
                    temp_context, temp_context.global_config, orchestrator
                )

                # Check if sequential mode is enabled
                if (
                    temp_context.pipeline_sequential_mode
                    and temp_context.pipeline_sequential_combinations
                ):
                    # Compile separate context for each sequential combination
                    combinations = temp_context.pipeline_sequential_combinations

                    for combo_idx, combo in enumerate(combinations):
                        context = orchestrator.create_context(axis_id)
                        context.step_axis_filters = global_step_axis_filters

                        # Store compile-time captured config values in context
                        context.analysis_consolidation_config = (
                            analysis_consolidation_config
                        )
                        context.auto_add_output_plate_to_plate_manager = (
                            auto_add_output_plate
                        )

                        # Set the current combination BEFORE freezing
                        context.pipeline_sequential_mode = True
                        context.pipeline_sequential_combinations = combinations
                        context.current_sequential_combination = combo

                        # Use pre-resolved steps and step_state_map for performance
                        resolved_steps, step_state_map = (
                            PipelineCompiler.initialize_step_plans_for_context(
                                context,
                                pipeline_definition,  # Now using the in-place replaced list
                                orchestrator,
                                metadata_writer=is_responsible,
                                plate_path=orchestrator.plate_path,
                                step_state_map=step_state_map,
                                step_snapshots=step_snapshots,
                                steps_already_resolved=True,
                                is_zmq_execution=is_zmq_execution,
                            )
                        )
                        PipelineCompiler.declare_zarr_stores_for_context(
                            context, resolved_steps, orchestrator
                        )
                        PipelineCompiler.plan_materialization_flags_for_context(
                            context, resolved_steps, orchestrator
                        )
                        PipelineCompiler.validate_memory_contracts_for_context(
                            context, resolved_steps, step_state_map, orchestrator
                        )
                        PipelineCompiler.assign_gpu_resources_for_context(context)

                        if enable_visualizer_override:
                            PipelineCompiler.apply_global_visualizer_override_for_context(
                                context, True
                            )

                        PipelineCompiler.resolve_lazy_dataclasses_for_context(
                            context, orchestrator, resolved_steps, step_state_map
                        )

                        context.freeze()
                        # Use composite key: (axis_id, combo_idx)
                        context_key = f"{axis_id}__combo_{combo_idx}"
                        compiled_contexts[context_key] = context
                else:
                    # No sequential mode - compile single context as before
                    context = orchestrator.create_context(axis_id)
                    context.step_axis_filters = global_step_axis_filters

                    # Store compile-time captured config values in context
                    context.analysis_consolidation_config = (
                        analysis_consolidation_config
                    )
                    context.auto_add_output_plate_to_plate_manager = (
                        auto_add_output_plate
                    )

                    # Use pre-resolved steps and step_state_map for performance
                    resolved_steps, step_state_map = (
                        PipelineCompiler.initialize_step_plans_for_context(
                            context,
                            pipeline_definition,  # Now using the in-place replaced list
                            orchestrator,
                            metadata_writer=is_responsible,
                            plate_path=orchestrator.plate_path,
                            step_state_map=step_state_map,
                            step_snapshots=step_snapshots,
                            steps_already_resolved=True,
                            is_zmq_execution=is_zmq_execution,
                        )
                    )
                    PipelineCompiler.declare_zarr_stores_for_context(
                        context, resolved_steps, orchestrator
                    )
                    PipelineCompiler.plan_materialization_flags_for_context(
                        context, resolved_steps, orchestrator
                    )

                    # Validate sequential components compatibility BEFORE analyzing sequential mode
                    seq_config = context.global_config.sequential_processing_config
                    if seq_config and seq_config.sequential_components:
                        PipelineCompiler.validate_sequential_components_compatibility(
                            pipeline_definition,
                            seq_config.sequential_components,
                            step_state_map,
                        )

                    PipelineCompiler.analyze_pipeline_sequential_mode(
                        context, context.global_config, orchestrator
                    )
                    PipelineCompiler.validate_memory_contracts_for_context(
                        context, resolved_steps, step_state_map, orchestrator
                    )
                    PipelineCompiler.assign_gpu_resources_for_context(context)

                    if enable_visualizer_override:
                        PipelineCompiler.apply_global_visualizer_override_for_context(
                            context, True
                        )

                    PipelineCompiler.resolve_lazy_dataclasses_for_context(
                        context, orchestrator, resolved_steps, step_state_map
                    )

                    context.freeze()
                    compiled_contexts[axis_id] = context

                # Emit progress after each axis is compiled (applies to both sequential and non-sequential)
                completed_axis_values += 1
                emit(
                    execution_id=orchestrator.execution_id,
                    plate_id=str(orchestrator.plate_path),
                    axis_id=axis_id,
                    step_name="compilation",
                    phase=ProgressPhase.COMPILE,
                    status=ProgressStatus.RUNNING,
                    completed=completed_axis_values,
                    total=total_axis_values,
                    percent=(completed_axis_values / total_axis_values) * 100.0,
                )

            # Log path planning summary once per plate
            if compiled_contexts:
                first_context = next(iter(compiled_contexts.values()))
                logger.info("📁 PATH PLANNING SUMMARY:")
                logger.info(
                    f"   Main pipeline output: {first_context.output_plate_root}"
                )

                # Check for materialization steps in first context
                materialization_steps = []
                for step_id, plan in first_context.step_plans.items():
                    if plan.materialized_output is not None:
                        step_name = plan.step_name or f"step_{step_id}"
                        mat_path = plan.materialized_output.output_dir
                        materialization_steps.append((step_name, mat_path))

                for step_name, mat_path in materialization_steps:
                    logger.info(f"   Materialization {step_name}: {mat_path}")

            # After processing all wells, cleanup ObjectStates and finalize
            # Cleanup persistent ObjectStates created for compilation
            # IMPORTANT: Only unregister orchestrator and steps, NOT the pipeline_config at plate_path
            plate_path_str = str(orchestrator.plate_path)
            orch_scope_id = f"{plate_path_str}::orchestrator"
            ObjectStateRegistry.unregister_scope_and_descendants(
                orch_scope_id, _skip_snapshot=True
            )
            logger.debug(
                f"Cleaned up compilation ObjectStates for scope: {orch_scope_id}"
            )

            logger.info("Stripping attributes from pipeline definition steps.")
            StepAttributeStripper.strip_step_attributes(pipeline_definition, {})

            orchestrator._state = OrchestratorState.COMPILED

            # Log worker configuration for execution planning
            effective_config = orchestrator.get_effective_config()
            logger.info(
                f"Execution config: {effective_config.num_workers} workers configured for pipeline execution"
            )

            logger.info(
                f"Compilation complete: {len(compiled_contexts)} wells compiled successfully"
            )

            logger.debug(
                "Checking pipeline_definition before compiler return"
            )
            for i, step in enumerate(pipeline_definition):
                func_attr = getattr(step, "func", None)
                func_type = type(func_attr).__name__ if func_attr else "None"
                logger.debug(f"📦 COMPILER RETURN: step[{i}].func = {func_type}")

            # Return expected structure with both pipeline_definition and compiled_contexts
            return {
                "pipeline_definition": pipeline_definition,
                "compiled_contexts": compiled_contexts,
            }
        except Exception as e:
            orchestrator._state = OrchestratorState.COMPILE_FAILED
            logger.error(f"Failed to compile pipelines: {e}")
            raise


# The monolithic compile() method is removed.
# Orchestrator will call the static methods above in sequence.
# _strip_step_attributes is also removed as StepAttributeStripper is called by Orchestrator.


def _resolve_step_axis_filters(
    step_snapshots: tuple[StepSnapshot, ...],
    context,
    orchestrator,
):
    """
    Resolve axis filters for steps with any WellFilterConfig instances.

    This function handles step-level axis filtering by resolving patterns like
    "row:A", ["A01", "B02"], or max counts against the available axis values for the plate.
    It processes ALL WellFilterConfig instances (materialization, streaming, etc.) uniformly.

    Args:
        step_snapshots: ObjectState-resolved compiler snapshots
        context: Processing context for the current axis value
        orchestrator: Orchestrator instance with access to available axis values
    """

    # Get available axis values from orchestrator using multiprocessing axis

    available_axis_values = orchestrator.get_component_keys(get_multiprocessing_axis())
    if not available_axis_values:
        logger.warning("No available axis values found for axis filter resolution")
        return

    # Initialize step_axis_filters in context if not present
    if not hasattr(context, "step_axis_filters"):
        context.step_axis_filters = {}

    for snapshot in step_snapshots:
        step_filters = {}
        for well_filter in snapshot.well_filters:
            resolved_axis_values = WellFilterProcessor.resolve_filter_with_mode(
                well_filter.well_filter,
                well_filter.well_filter_mode,
                available_axis_values,
            )
            step_filters[well_filter.root] = {
                "resolved_axis_values": set(resolved_axis_values),
                "filter_mode": well_filter.well_filter_mode,
                "original_filter": well_filter.well_filter,
            }

        if step_filters:
            context.step_axis_filters[snapshot.index] = step_filters

    total_filters = sum(len(filters) for filters in context.step_axis_filters.values())
    logger.debug(
        f"Axis filter resolution complete. {len(context.step_axis_filters)} steps have axis filters, {total_filters} total filters."
    )
