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
import inspect
import time
from pathlib import Path
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
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
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.materialization_flag_planner import (
    MaterializationFlagPlanner,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan, InputConversionPlan
from openhcs.core.pipeline.path_planner import PipelinePathPlanner
from openhcs.core.pipeline.step_snapshot import (
    StepSnapshot,
    build_step_snapshots,
)
from openhcs.core.source_bindings import CompiledSourceBindingPlan
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


def _step_scope_token(step: "AbstractStep", step_index: int) -> str:
    """Return the existing step scope token when available."""
    token = getattr(step, "_scope_token", None)
    if isinstance(token, str) and token:
        return token
    return f"step_{step_index}"


def _compiler_step_scope_id(
    compilation_scope: str,
    step: "AbstractStep",
    step_index: int,
) -> str:
    """Build a compiler ObjectState scope id that preserves stable step tokens."""
    return f"{compilation_scope}::{_step_scope_token(step, step_index)}"


_FUNCTION_REFERENCE_ATTRIBUTE_FIELDS = MappingProxyType({
    "__name__": "function_name",
    "__module__": "original_module",
})

_FUNCTION_REFERENCE_PRESERVED_ATTRS = (
    "__artifact_inputs__",
    "__artifact_outputs__",
    "__runtime_adapter__",
    "input_memory_type",
    "output_memory_type",
)


MATERIALIZATION_PLAN_REQUIREMENTS = (
    (READ_BACKEND, lambda plan: plan.read_backend),
    (WRITE_BACKEND, lambda plan: plan.write_backend),
)

FUNCTION_MEMORY_PLAN_REQUIREMENTS = (
    ("input_memory_type", lambda plan: plan.input_memory_type),
    ("output_memory_type", lambda plan: plan.output_memory_type),
    ("func", lambda plan: plan.func),
)


@dataclass(frozen=True, slots=True)
class StepPlanInputSource:
    """Authoritative source record for resolving per-context step-plan inputs."""

    context: ProcessingContext
    steps_definition: List[AbstractStep]
    orchestrator: Any
    step_state_map: Dict[int, "ObjectState"] | None
    step_snapshots: tuple[StepSnapshot, ...] | None
    steps_already_resolved: bool
    is_zmq_execution: bool


@dataclass(slots=True)
class ResolvedStepPlanInputs:
    """Resolved step inputs required by path planning and downstream validation."""

    steps: List[AbstractStep]
    step_state_map: Dict[int, "ObjectState"]
    snapshots: tuple[StepSnapshot, ...]

    @classmethod
    def from_source(cls, source: StepPlanInputSource) -> "ResolvedStepPlanInputs":
        if not source.steps_already_resolved or source.step_state_map is None:
            return PipelineCompiler._resolve_steps_for_context(source)

        logger.debug("Using pre-resolved steps for context %s", source.context.axis_id)
        snapshots = source.step_snapshots or build_step_snapshots(
            source.steps_definition,
            source.step_state_map,
        )
        return cls(
            steps=source.steps_definition,
            step_state_map=source.step_state_map,
            snapshots=snapshots,
        )

    @classmethod
    def from_resolved(
        cls,
        steps: List[AbstractStep],
        step_state_map: Dict[int, "ObjectState"],
    ) -> "ResolvedStepPlanInputs":
        return cls(
            steps=steps,
            step_state_map=step_state_map,
            snapshots=build_step_snapshots(steps, step_state_map),
        )


@dataclass(frozen=True, slots=True)
class AxisCompilationRequest:
    """Authoritative context record for axis-level compilation fanout."""

    orchestrator: Any
    pipeline_definition: List[AbstractStep]
    step_state_map: Mapping[int, "ObjectState"]
    step_snapshots: tuple[StepSnapshot, ...]
    analysis_consolidation_config: Any
    plate_metadata_config: Any
    auto_add_output_plate: Any
    global_step_axis_filters: dict[int, dict[str, Any]]
    enable_visualizer_override: bool
    is_zmq_execution: bool

    def context_for(self, axis_id: str) -> ProcessingContext:
        context = self.orchestrator.create_context(axis_id)
        context.step_axis_filters = self.global_step_axis_filters
        context.analysis_consolidation_config = (
            self.analysis_consolidation_config
        )
        context.plate_metadata_config = self.plate_metadata_config
        context.auto_add_output_plate_to_plate_manager = (
            self.auto_add_output_plate
        )
        return context


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


def _missing_plan_fields(
    plan: CompiledStepPlan,
    requirements: Sequence[tuple[str, Callable[[CompiledStepPlan], object | None]]],
) -> list[str]:
    return [
        name
        for name, read_field in requirements
        if read_field(plan) is None
    ]


def _refresh_function_objects_in_steps(pipeline_definition: List[AbstractStep]) -> None:
    """
    Refresh all function objects in pipeline steps to ensure they're picklable.

    This recreates function objects by importing them fresh from their original modules,
    similar to how code mode works, which avoids unpicklable closures from registry wrapping.
    """
    logger.debug(f"🔄 FUNCTION REFRESH: Processing {len(pipeline_definition)} steps")
    for step_idx, step in enumerate(pipeline_definition):
        if isinstance(step, FunctionStep):
            if step.func is not None:
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
                    f"🔄 FUNCTION REFRESH: Step {step_idx} ({step.name}): No function pattern"
                )


def _refresh_function_object(func_value):
    """Convert function objects to picklable FunctionReference objects.

    Also filters out functions with enabled=False at compile time.
    """
    if callable(func_value):
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
        return inspect.unwrap(f)

    original_func = _get_original_func(func)
    original_name = original_func.__name__
    original_module = original_func.__module__

    all_functions = RegistryService.get_all_functions_with_metadata()

    for composite_key, metadata in all_functions.items():
        registry_original = _get_original_func(metadata.func)
        registry_module = registry_original.__module__
        if (
            registry_original.__name__ == original_name
            and registry_module == original_module
        ):
            function_attrs = func.__dict__
            preserved_attrs = {
                attr: function_attrs[attr]
                for attr in _FUNCTION_REFERENCE_PRESERVED_ATTRS
                if attr in function_attrs
            }

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


def _dataclass_field_candidate(field_type: Any) -> Any:
    origin = get_origin(field_type)
    if origin is Annotated:
        return get_args(field_type)[0]
    if origin is Union:
        for arg in get_args(field_type):
            if arg is type(None):
                continue
            if dataclasses.is_dataclass(arg):
                return arg
        return None
    return field_type


def _rebuild_dataclass_from_objectstate(
    config_cls,
    step_state,
    root_field_name,
):
    """Reconstruct a dataclass from saved ObjectState dotted-path values only."""
    kwargs = {}
    for field in dataclasses.fields(config_cls):
        dotted = f"{root_field_name}.{field.name}"
        value = step_state.get_saved_resolved_value(dotted)
        candidate = _dataclass_field_candidate(field.type)

        if (
            value is None
            and candidate is not None
            and dataclasses.is_dataclass(candidate)
        ):
            value = _rebuild_dataclass_from_objectstate(
                candidate,
                step_state,
                dotted,
            )

        kwargs[field.name] = value

    return config_cls(**kwargs)


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
        PipelineCompiler._assert_context_mutable_for_planning(context)
        context.visualizer_config = None

        resolved_inputs = ResolvedStepPlanInputs.from_source(
            StepPlanInputSource(
                context,
                steps_definition,
                orchestrator,
                step_state_map,
                step_snapshots,
                steps_already_resolved,
                is_zmq_execution,
            )
        )
        PipelineCompiler._ensure_initial_step_plans(context, resolved_inputs)
        PipelineCompiler._configure_input_conversion_if_needed(
            context,
            resolved_inputs.steps,
            orchestrator,
            plate_path,
        )
        PipelineCompiler._plan_context_paths(
            context,
            resolved_inputs,
            orchestrator,
        )

        session = CompilationSession.from_context(
            context=context,
            steps=resolved_inputs.steps,
            orchestrator=orchestrator,
            step_state_map=resolved_inputs.step_state_map,
            snapshots=resolved_inputs.snapshots,
            metadata_writer=metadata_writer,
            plate_path=plate_path,
        )
        PipelineCompiler._supplement_step_plans(session)
        PipelineCompiler._collect_streaming_configs(
            session,
            is_zmq_execution=is_zmq_execution,
        )
        return resolved_inputs.steps, resolved_inputs.step_state_map

    @staticmethod
    def _assert_context_mutable_for_planning(context: ProcessingContext) -> None:
        if context.is_frozen():
            raise AttributeError(
                "Cannot initialize step plans in a frozen ProcessingContext."
            )
        if context.step_plans is None:
            context.step_plans = {}

    @staticmethod
    def _resolve_steps_for_context(
        source: StepPlanInputSource,
    ) -> ResolvedStepPlanInputs:
        compilation_id = f"compile_{int(time.time() * 1000)}"

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

        orch_scope_id = f"{compilation_id}::orchestrator"
        orch_state = _register_object_state(
            source.orchestrator,
            orch_scope_id,
            global_config_state,
        )
        logger.info("Registered orchestrator at scope: %s", orch_scope_id)

        step_state_map = PipelineCompiler._register_context_step_states(
            source.context,
            source.steps_definition,
            compilation_id,
            orch_state,
        )
        resolved_steps = PipelineCompiler._resolve_registered_steps(
            source.steps_definition,
            step_state_map,
        )

        if source.is_zmq_execution:
            ObjectStateRegistry.unregister(orch_state, _skip_snapshot=True)
            for step_state in step_state_map.values():
                ObjectStateRegistry.unregister(step_state, _skip_snapshot=True)

        _refresh_function_objects_in_steps(resolved_steps)
        logger.info(
            "Resolved %s steps under scope: %s",
            len(resolved_steps),
            compilation_id,
        )
        return ResolvedStepPlanInputs.from_resolved(resolved_steps, step_state_map)

    @staticmethod
    def _register_context_step_states(
        context: ProcessingContext,
        steps_definition: Sequence[AbstractStep],
        compilation_id: str,
        orch_state: "ObjectState",
    ) -> Dict[int, "ObjectState"]:
        plate_scope = context.plate_path or "plate"
        step_state_map: Dict[int, "ObjectState"] = {}
        for step_index, step in enumerate(steps_definition):
            step_scope_id = _compiler_step_scope_id(
                f"{compilation_id}::{plate_scope}",
                step,
                step_index,
            )
            step_state_map[step_index] = _register_object_state(
                step,
                step_scope_id,
                orch_state,
            )
        return step_state_map

    @staticmethod
    def _resolve_registered_steps(
        steps_definition: Sequence[AbstractStep],
        step_state_map: Mapping[int, "ObjectState"],
    ) -> List[AbstractStep]:
        resolved_steps: List[AbstractStep] = []
        for step_index, step in enumerate(steps_definition):
            logger.info(
                "Resolving step %s ('%s') from ObjectState.",
                step_index,
                step.name,
            )
            resolved_steps.append(step_state_map[step_index].to_object())
        return resolved_steps

    @staticmethod
    def _ensure_initial_step_plans(
        context: ProcessingContext,
        resolved_inputs: ResolvedStepPlanInputs,
    ) -> None:
        for step_index, snapshot in enumerate(resolved_inputs.snapshots):
            if step_index not in context.step_plans:
                context.step_plans[step_index] = CompiledStepPlan(
                    step_index=step_index,
                    step_name=snapshot.name,
                    step_type=snapshot.step_type,
                    axis_id=context.axis_id,
                )

    @staticmethod
    def _configure_input_conversion_if_needed(
        context: ProcessingContext,
        steps: Sequence[AbstractStep],
        orchestrator,
        plate_path: Path | None,
    ) -> None:
        if not steps or plate_path is None:
            return

        vfs_config = orchestrator.get_effective_config().vfs_config
        if vfs_config.materialization_backend != MaterializationBackend.ZARR:
            return

        available_backends = context.microscope_handler.get_available_backends(
            plate_path
        )
        if Backend.ZARR in available_backends:
            return

        from openhcs.microscopes.openhcs import OpenHCSMetadataHandler
        from polystore.metadata_writer import get_subdirectory_name

        openhcs_metadata_handler = OpenHCSMetadataHandler(context.filemanager)
        metadata = openhcs_metadata_handler._load_metadata_dict(plate_path)
        subdirs = metadata["subdirectories"]
        original_subdir = get_subdirectory_name(context.input_dir, plate_path)
        uses_virtual_workspace = (
            Backend.VIRTUAL_WORKSPACE.value
            in subdirs[original_subdir]["available_backends"]
        )

        zarr_subdir = "zarr" if uses_virtual_workspace else original_subdir
        context.step_plans[0].input_conversion = InputConversionPlan(
            output_dir=plate_path / zarr_subdir,
            backend=MaterializationBackend.ZARR.value,
            uses_virtual_workspace=uses_virtual_workspace,
            original_subdir=original_subdir,
        )
        logger.debug(
            "Input conversion to zarr enabled for first step: %s",
            steps[0].name,
        )

    @staticmethod
    def _plan_context_paths(
        context: ProcessingContext,
        resolved_inputs: ResolvedStepPlanInputs,
        orchestrator,
    ) -> None:
        PipelinePathPlanner.prepare_pipeline_paths(
            context,
            resolved_inputs.steps,
            context.global_config,
            orchestrator=orchestrator,
            step_state_map=resolved_inputs.step_state_map,
            step_snapshots=resolved_inputs.snapshots,
        )

    @staticmethod
    def _supplement_step_plans(session: CompilationSession) -> None:
        for step_index, snapshot in enumerate(session.snapshots):
            if step_index not in session.plans:
                logger.error(
                    "Critical error: Step %s (index: %s) not found in step_plans after path planning phase.",
                    snapshot.name,
                    step_index,
                )
                session.plans[step_index] = CompiledStepPlan(
                    step_index=step_index,
                    step_name=snapshot.name,
                    step_type=snapshot.step_type,
                    axis_id=session.axis_id,
                    error="Missing from path planning phase by PipelinePathPlanner",
                    create_openhcs_metadata=session.metadata_writer,
                )
                continue

            current_plan = session.plans[step_index]
            current_plan.step_scope_id = snapshot.scope_id
            current_plan.step_name = snapshot.name
            current_plan.step_type = snapshot.step_type
            current_plan.axis_id = session.axis_id
            current_plan.create_openhcs_metadata = session.metadata_writer
            current_plan.variable_components = snapshot.variable_components
            current_plan.group_by = FuncStepContractValidator.normalized_group_by(
                snapshot.group_by,
                snapshot.variable_components,
                snapshot.name,
            )
            current_plan.input_source = snapshot.input_source
            current_plan.sequential_processing = snapshot.processing_config
            current_plan.source_binding_plan = CompiledSourceBindingPlan.from_config(
                snapshot.source_bindings
            )

    @staticmethod
    def _collect_streaming_configs(
        session: CompilationSession,
        *,
        is_zmq_execution: bool,
    ) -> None:
        registry_keys = list(StreamingConfig.__registry__.keys())
        for step_index, step_state in session.step_state_map.items():
            step_plan = session.plans[step_index]
            for field_name in registry_keys:
                PipelineCompiler._collect_streaming_config(
                    session,
                    step_index,
                    step_state,
                    step_plan,
                    field_name,
                    is_zmq_execution=is_zmq_execution,
                )

    @staticmethod
    def _collect_streaming_config(
        session: CompilationSession,
        step_index: int,
        step_state: "ObjectState",
        step_plan: CompiledStepPlan,
        field_name: str,
        *,
        is_zmq_execution: bool,
    ) -> None:
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
        if enabled is not True:
            return

        base_cls = get_base_type_for_lazy(StreamingConfig.__registry__[field_name])
        config_obj = _rebuild_dataclass_from_objectstate(
            base_cls,
            step_state,
            field_name,
        )
        backend_name = step_state.get_saved_resolved_value(f"{field_name}.backend")
        visualizer_info = {"backend": backend_name, "config": config_obj}
        if visualizer_info not in session.context.required_visualizers:
            session.context.required_visualizers.append(visualizer_info)
            logger.info(
                "Streaming enabled for step %s, field %s (backend=%s)",
                step_index,
                field_name,
                backend_name,
            )

        step_plan.streaming_configs[field_name] = config_obj

    # _prepare_materialization_flags is removed as MaterializationFlagPlanner.prepare_pipeline_flags
    # now modifies context.step_plans in-place and takes context directly.

    @staticmethod
    def declare_zarr_stores(session: CompilationSession) -> None:
        """
        Declare zarr store creation functions for runtime execution.

        This method runs after path planning but before materialization flag planning
        to declare which steps need zarr stores and provide the metadata needed
        for runtime store creation.

        Args:
            session: Axis-scoped compiler session.
        """

        context = session.context
        orchestrator = session.orchestrator
        all_wells = orchestrator.get_component_keys(get_multiprocessing_axis())

        # Access config from merged config (pipeline + global) for proper inheritance
        vfs_config = orchestrator.get_effective_config().vfs_config

        for step_index, snapshot in enumerate(session.snapshots):
            step_plan = session.plan(step_index)

            will_use_zarr = (
                vfs_config.materialization_backend == MaterializationBackend.ZARR
                and step_index == len(session.steps) - 1
            )

            if will_use_zarr:
                step_plan.zarr_config = {
                    "all_wells": all_wells,
                    "needs_initialization": True,
                }
                logger.debug(
                    f"Step '{snapshot.name}' will use zarr backend for axis {context.axis_id}"
                )
            else:
                step_plan.zarr_config = None

    @staticmethod
    def plan_materialization_flags(session: CompilationSession) -> None:
        """
        Plans and injects materialization flags into context.step_plans
        by calling MaterializationFlagPlanner.
        """
        context = session.context
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
            session.steps,
            session.orchestrator.plate_path,
            context.global_config,  # Use merged config from context instead of raw pipeline_config
        )

        # Post-check (optional, but good for ensuring contracts are met by the planner)
        for step_index, snapshot in enumerate(session.snapshots):
            if step_index not in context.step_plans:
                # This should not happen if prepare_pipeline_flags guarantees plans for all steps
                logger.error(
                    f"Step {snapshot.name} (index: {step_index}) missing from step_plans after materialization planning."
                )
                continue

            plan = context.step_plans[step_index]
            # Check for keys that FunctionStep actually uses during execution
            missing_keys = _missing_plan_fields(
                plan,
                MATERIALIZATION_PLAN_REQUIREMENTS,
            )
            if missing_keys:
                logger.error(
                    f"Materialization flag planning incomplete for step {snapshot.name} (index: {step_index}). "
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
    def validate_memory_contracts(session: CompilationSession) -> None:
        """
        Validates FunctionStep memory contracts, dict patterns, and adds memory type info to context.step_plans.

        Args:
            context: ProcessingContext to validate
            steps_definition: List of AbstractStep objects
            step_state_map: Map of step index to ObjectState for accessing config values
            orchestrator: Optional orchestrator for dict pattern key validation
        """
        context = session.context
        if context.is_frozen():
            raise AttributeError(
                "Cannot validate memory contracts in a frozen ProcessingContext."
            )

        FuncStepContractValidator.validate_pipeline(
            steps=session.steps,
            pipeline_context=context,  # Pass context so validator can access step plans for memory type overrides
            step_state_map=session.step_state_map,  # Pass step_state_map for accessing config via ObjectState
            orchestrator=session.orchestrator,  # Pass orchestrator for dict pattern key validation
        )

        for step_index, step in enumerate(session.steps):
            if not isinstance(step, FunctionStep):
                continue
            if step_index not in context.step_plans:
                raise AssertionError(
                    f"Memory validation requires a compiled plan for FunctionStep {session.snapshot(step_index).name} (index: {step_index})."
                )
            step_plan = context.step_plans[step_index]
            missing_fields = _missing_plan_fields(
                step_plan,
                FUNCTION_MEMORY_PLAN_REQUIREMENTS,
            )
            if missing_fields:
                raise AssertionError(
                    f"Memory validation must set {missing_fields} for FunctionStep {step_plan.step_name} (index: {step_index})."
                )

        # Apply memory type override: Any step with disk output must use numpy for disk writing
        for step_index, step in enumerate(session.steps):
            if isinstance(step, FunctionStep):
                if step_index in context.step_plans:
                    step_plan = context.step_plans[step_index]
                    write_backend = step_plan.write_backend

                    if write_backend == "disk":
                        logger.debug(
                            f"Step {session.snapshot(step_index).name} has disk output, overriding output_memory_type to numpy"
                        )
                        step_plan.output_memory_type = "numpy"

    @staticmethod
    def assign_gpu_resources(session: CompilationSession) -> None:
        """
        Validates GPU memory types from context.step_plans and assigns GPU device IDs.
        """
        context = session.context
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
    def apply_global_visualizer_override(
        session: CompilationSession, global_enable_visualizer: bool
    ) -> None:
        """
        Applies global visualizer override to all step_plans in the context.
        """
        context = session.context
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
    def resolve_lazy_dataclasses(session: CompilationSession) -> None:
        """
        Resolve all lazy dataclass instances in step plans to their base configurations.

        This method uses ObjectState for resolution.
        All configs are already resolved via ObjectState.to_object() during compilation.
        This method now just ensures step plans reference the resolved configs.

        Args:
            session: Axis-scoped compiler session.
        """
        # Configs are already resolved via ObjectState.to_object() in initialize_step_plans_for_context
        # No additional resolution needed - step plans already contain resolved configs
        logger.debug(
            f"Step plans already resolved via ObjectState for {len(session.steps)} steps"
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
    def _validate_compile_request(
        orchestrator,
        pipeline_definition: Sequence[AbstractStep],
    ) -> None:
        if not orchestrator.is_initialized():
            raise RuntimeError(
                "PipelineOrchestrator must be explicitly initialized before calling compile_pipelines()."
            )
        if not pipeline_definition:
            raise ValueError(
                "A valid pipeline definition (List[AbstractStep]) must be provided."
            )

    @staticmethod
    def _axis_values_to_process(
        orchestrator,
        axis_filter: Optional[List[str]],
    ) -> List[str]:
        resolved_axis_filter = axis_filter
        effective_config = orchestrator.get_effective_config()
        well_filter_config = (
            effective_config.well_filter_config
            if effective_config
            else None
        )
        if well_filter_config and well_filter_config.well_filter is not None:
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
            resolved_axis_filter = (
                [well for well in resolved_wells if well in axis_filter]
                if axis_filter
                else resolved_wells
            )
            if axis_filter:
                logger.info(
                    f"Intersected with axis_filter: {len(resolved_axis_filter)} wells remain"
                )

        return orchestrator.get_component_keys(
            get_multiprocessing_axis(),
            resolved_axis_filter,
        )

    @staticmethod
    def _register_and_resolve_pipeline_once(
        orchestrator,
        pipeline_definition: List[AbstractStep],
        *,
        is_zmq_execution: bool,
    ) -> tuple["ObjectState", ResolvedStepPlanInputs]:
        # Compile from the submitted pipeline definition, not from any stale UI
        # ObjectState that may point at post-compile stripped step shells.
        force_fresh = True
        global_config_state = PipelineCompiler._compile_global_config_state(
            force_fresh=force_fresh
        )
        plate_path_str = str(orchestrator.plate_path)
        plate_orch_state = PipelineCompiler._pipeline_config_state(
            orchestrator,
            plate_path_str,
            global_config_state,
            force_fresh=force_fresh,
        )
        orchestrator_scope_id = f"{plate_path_str}::orchestrator"
        orch_state = _get_or_register_object_state(
            orchestrator_scope_id,
            orchestrator,
            plate_orch_state,
            force_fresh=force_fresh,
        )
        logger.debug("Registered orchestrator at scope: %s", orchestrator_scope_id)

        step_state_map = PipelineCompiler._register_pipeline_step_states(
            pipeline_definition,
            plate_path_str,
            orch_state,
            force_fresh=force_fresh,
        )
        PipelineCompiler._replace_pipeline_with_resolved_steps(
            pipeline_definition,
            step_state_map,
        )
        _refresh_function_objects_in_steps(pipeline_definition)
        logger.debug(
            "Refreshed function objects in %s steps (converted to FunctionReference)",
            len(pipeline_definition),
        )

        step_state_map, snapshots = PipelineCompiler._filter_enabled_steps(
            pipeline_definition,
            step_state_map,
        )
        pipeline_config_state = ObjectStateRegistry.get_by_scope(plate_path_str)
        if pipeline_config_state is None:
            raise RuntimeError(
                "Missing ObjectState for plate; cannot resolve pipeline config."
            )
        return (
            pipeline_config_state,
            ResolvedStepPlanInputs(
                steps=pipeline_definition,
                step_state_map=step_state_map,
                snapshots=snapshots,
            ),
        )

    @staticmethod
    def _compile_global_config_state(*, force_fresh: bool) -> "ObjectState" | None:
        from objectstate import get_current_global_config
        from openhcs.core.config import GlobalPipelineConfig

        global_config_state = ObjectStateRegistry.get_by_scope("")
        if force_fresh or global_config_state is None:
            global_config = get_current_global_config(
                GlobalPipelineConfig,
                use_live=False,
            )
            if global_config:
                global_config_state = _register_object_state(
                    global_config,
                    "",
                    None,
                )
                logger.debug("Registered global config at scope ''")
        return global_config_state

    @staticmethod
    def _pipeline_config_state(
        orchestrator,
        plate_path_str: str,
        global_config_state: "ObjectState" | None,
        *,
        force_fresh: bool,
    ) -> "ObjectState" | None:
        plate_orch_state = ObjectStateRegistry.get_by_scope(plate_path_str)
        if orchestrator.pipeline_config:
            plate_orch_state = _get_or_register_object_state(
                plate_path_str,
                orchestrator.pipeline_config,
                global_config_state,
                force_fresh=force_fresh,
            )
            logger.debug("Registered pipeline_config at scope '%s'", plate_path_str)
        return plate_orch_state

    @staticmethod
    def _register_pipeline_step_states(
        pipeline_definition: Sequence[AbstractStep],
        plate_path_str: str,
        orch_state: "ObjectState",
        *,
        force_fresh: bool,
    ) -> Dict[int, "ObjectState"]:
        step_state_map: Dict[int, "ObjectState"] = {}
        for step_index, step in enumerate(pipeline_definition):
            step_scope_id = _compiler_step_scope_id(
                plate_path_str,
                step,
                step_index,
            )
            step_state_map[step_index] = _get_or_register_object_state(
                step_scope_id,
                step,
                orch_state,
                force_fresh=force_fresh,
            )
        return step_state_map

    @staticmethod
    def _replace_pipeline_with_resolved_steps(
        pipeline_definition: List[AbstractStep],
        step_state_map: Mapping[int, "ObjectState"],
    ) -> None:
        pipeline_definition.clear()
        pipeline_definition.extend(
            step_state.to_object() for step_state in step_state_map.values()
        )
        logger.debug(
            "Resolved %s steps once per pipeline (replaced original list in-place)",
            len(pipeline_definition),
        )

    @staticmethod
    def _filter_enabled_steps(
        pipeline_definition: List[AbstractStep],
        step_state_map: Mapping[int, "ObjectState"],
    ) -> tuple[Dict[int, "ObjectState"], tuple[StepSnapshot, ...]]:
        snapshots = build_step_snapshots(pipeline_definition, step_state_map)
        enabled_pairs = [
            (step, step_state_map[snapshot.index])
            for snapshot, step in zip(snapshots, pipeline_definition)
            if snapshot.enabled
        ]
        pipeline_definition.clear()
        pipeline_definition.extend(step for step, _state in enabled_pairs)
        enabled_state_map = {
            new_index: state
            for new_index, (_step, state) in enumerate(enabled_pairs)
        }
        if not pipeline_definition:
            return enabled_state_map, ()
        return (
            enabled_state_map,
            build_step_snapshots(pipeline_definition, enabled_state_map),
        )

    @staticmethod
    def _capture_pipeline_config(
        pipeline_config_state: "ObjectState",
    ) -> tuple[Any, Any, Any, int]:
        from objectstate.lazy_factory import LazyDataclass

        lazy_analysis_config = pipeline_config_state.get_saved_resolved_value(
            "analysis_consolidation_config"
        )
        analysis_consolidation_config = (
            lazy_analysis_config.to_base_config()
            if isinstance(lazy_analysis_config, LazyDataclass)
            else lazy_analysis_config
        )
        return (
            analysis_consolidation_config,
            pipeline_config_state.get_saved_resolved_value(
                "plate_metadata_config",
            ),
            pipeline_config_state.get_saved_resolved_value(
                "auto_add_output_plate_to_plate_manager",
            ),
            pipeline_config_state.get_saved_resolved_value("num_workers"),
        )

    @staticmethod
    def _resolve_global_step_axis_filters(
        orchestrator,
        step_snapshots: tuple[StepSnapshot, ...],
    ) -> dict[int, dict[str, Any]]:
        temp_context = orchestrator.create_context("temp")
        _resolve_step_axis_filters(step_snapshots, temp_context, orchestrator)
        return temp_context.step_axis_filters

    @staticmethod
    def _compile_axis_values(
        request: AxisCompilationRequest,
        axis_values: Sequence[str],
    ) -> Dict[str, ProcessingContext]:
        compiled_contexts: Dict[str, ProcessingContext] = {}
        responsible_axis_value = sorted(axis_values)[0]
        total_axis_values = len(axis_values)
        for completed, axis_id in enumerate(axis_values, start=1):
            compiled_contexts.update(
                PipelineCompiler._compile_axis_value(
                    request=request,
                    axis_id=axis_id,
                    metadata_writer=axis_id == responsible_axis_value,
                )
            )
            PipelineCompiler._emit_axis_compile_progress(
                request.orchestrator,
                axis_id,
                completed,
                total_axis_values,
            )
        return compiled_contexts

    @staticmethod
    def _compile_axis_value(
        *,
        request: AxisCompilationRequest,
        axis_id: str,
        metadata_writer: bool,
    ) -> Dict[str, ProcessingContext]:
        temp_context = request.context_for(axis_id)
        temp_session = PipelineCompiler.build_initialize_axis_session(
            request,
            temp_context,
            metadata_writer,
        )
        PipelineCompiler._validate_sequential_components_for_session(temp_session)
        PipelineCompiler.analyze_pipeline_sequential_mode(
            temp_context,
            temp_context.global_config,
            request.orchestrator,
        )
        if (
            temp_context.pipeline_sequential_mode
            and temp_context.pipeline_sequential_combinations
        ):
            return PipelineCompiler._compile_sequential_axis_contexts(
                request=request,
                temp_context=temp_context,
                axis_id=axis_id,
                metadata_writer=metadata_writer,
            )
        context = PipelineCompiler._compile_single_axis_context(
            request=request,
            axis_id=axis_id,
            metadata_writer=metadata_writer,
        )
        return {axis_id: context}

    @staticmethod
    def build_initialize_axis_session(
        request: AxisCompilationRequest,
        context: ProcessingContext,
        metadata_writer: bool,
    ) -> CompilationSession:
        resolved_steps, resolved_state_map = (
            PipelineCompiler.initialize_step_plans_for_context(
                context,
                request.pipeline_definition,
                request.orchestrator,
                metadata_writer=metadata_writer,
                plate_path=request.orchestrator.plate_path,
                step_state_map=dict(request.step_state_map),
                step_snapshots=request.step_snapshots,
                steps_already_resolved=True,
                is_zmq_execution=request.is_zmq_execution,
            )
        )
        return CompilationSession.from_context(
            context=context,
            steps=resolved_steps,
            orchestrator=request.orchestrator,
            step_state_map=resolved_state_map,
            snapshots=request.step_snapshots,
            metadata_writer=metadata_writer,
            plate_path=request.orchestrator.plate_path,
        )

    @staticmethod
    def _compile_sequential_axis_contexts(
        *,
        request: AxisCompilationRequest,
        temp_context: ProcessingContext,
        axis_id: str,
        metadata_writer: bool,
    ) -> Dict[str, ProcessingContext]:
        compiled_contexts: Dict[str, ProcessingContext] = {}
        combinations = temp_context.pipeline_sequential_combinations
        for combo_idx, combo in enumerate(combinations):
            context = request.context_for(axis_id)
            context.pipeline_sequential_mode = True
            context.pipeline_sequential_combinations = combinations
            context.current_sequential_combination = combo
            session = PipelineCompiler.build_initialize_axis_session(
                request,
                context,
                metadata_writer,
            )
            PipelineCompiler.declare_zarr_stores(session)
            PipelineCompiler.plan_materialization_flags(session)
            PipelineCompiler._run_post_plan_compile_stages(
                session,
                enable_visualizer_override=request.enable_visualizer_override,
            )
            context.freeze()
            compiled_contexts[f"{axis_id}__combo_{combo_idx}"] = context
        return compiled_contexts

    @staticmethod
    def _compile_single_axis_context(
        *,
        request: AxisCompilationRequest,
        axis_id: str,
        metadata_writer: bool,
    ) -> ProcessingContext:
        context = request.context_for(axis_id)
        session = PipelineCompiler.build_initialize_axis_session(
            request,
            context,
            metadata_writer,
        )
        PipelineCompiler.declare_zarr_stores(session)
        PipelineCompiler.plan_materialization_flags(session)
        PipelineCompiler._validate_sequential_components_for_session(session)
        PipelineCompiler.analyze_pipeline_sequential_mode(
            context,
            context.global_config,
            request.orchestrator,
        )
        PipelineCompiler._run_post_plan_compile_stages(
            session,
            enable_visualizer_override=request.enable_visualizer_override,
        )
        context.freeze()
        return context

    @staticmethod
    def _run_post_plan_compile_stages(
        session: CompilationSession,
        *,
        enable_visualizer_override: bool,
    ) -> None:
        PipelineCompiler.validate_memory_contracts(session)
        PipelineCompiler.assign_gpu_resources(session)
        if enable_visualizer_override:
            PipelineCompiler.apply_global_visualizer_override(
                session,
                True,
            )
        PipelineCompiler.resolve_lazy_dataclasses(session)

    @staticmethod
    def _validate_sequential_components_for_session(
        session: CompilationSession,
    ) -> None:
        seq_config = session.context.global_config.sequential_processing_config
        if seq_config and seq_config.sequential_components:
            PipelineCompiler.validate_sequential_components_compatibility(
                session.steps,
                seq_config.sequential_components,
                session.step_state_map,
            )

    @staticmethod
    def _emit_axis_compile_progress(
        orchestrator,
        axis_id: str,
        completed: int,
        total: int,
    ) -> None:
        emit(
            execution_id=orchestrator.execution_id,
            plate_id=str(orchestrator.plate_path),
            axis_id=axis_id,
            step_name="compilation",
            phase=ProgressPhase.COMPILE,
            status=ProgressStatus.RUNNING,
            completed=completed,
            total=total,
            percent=(completed / total) * 100.0,
        )

    @staticmethod
    def _finalize_compilation(
        orchestrator,
        pipeline_definition: List[AbstractStep],
        compiled_contexts: Mapping[str, ProcessingContext],
    ) -> None:
        PipelineCompiler._log_path_planning_summary(compiled_contexts)
        PipelineCompiler._cleanup_compilation_object_states(orchestrator)
        logger.info("Stripping attributes from pipeline definition steps.")
        StepAttributeStripper.strip_step_attributes(pipeline_definition, {})
        orchestrator._state = OrchestratorState.COMPILED
        effective_config = orchestrator.get_effective_config()
        logger.info(
            f"Execution config: {effective_config.num_workers} workers configured for pipeline execution"
        )
        logger.info(
            f"Compilation complete: {len(compiled_contexts)} wells compiled successfully"
        )
        logger.debug(
            "Pipeline definition stripped before compiler return: %s steps",
            len(pipeline_definition),
        )

    @staticmethod
    def _log_path_planning_summary(
        compiled_contexts: Mapping[str, ProcessingContext],
    ) -> None:
        if not compiled_contexts:
            return
        first_context = next(iter(compiled_contexts.values()))
        logger.info("📁 PATH PLANNING SUMMARY:")
        logger.info("   Main pipeline output: %s", first_context.output_plate_root)
        for step_id, plan in first_context.step_plans.items():
            if plan.materialized_output is not None:
                step_name = plan.step_name or f"step_{step_id}"
                logger.info(
                    "   Materialization %s: %s",
                    step_name,
                    plan.materialized_output.output_dir,
                )

    @staticmethod
    def _cleanup_compilation_object_states(orchestrator) -> None:
        orch_scope_id = f"{orchestrator.plate_path}::orchestrator"
        ObjectStateRegistry.unregister_scope_and_descendants(
            orch_scope_id,
            _skip_snapshot=True,
        )
        logger.debug(
            "Cleaned up compilation ObjectStates for scope: %s",
            orch_scope_id,
        )

    @staticmethod
    def _calculate_worker_assignments(
        wells: list[str],
        num_workers: int,
    ) -> dict[str, list[str]]:
        """Assign compiled context keys to stable worker slots."""
        if num_workers <= 0:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if len(set(wells)) != len(wells):
            raise ValueError(f"Duplicate well IDs: {wells}")

        slots = {f"worker_{idx}": [] for idx in range(num_workers)}
        for idx, axis_id in enumerate(sorted(wells)):
            slot = f"worker_{idx % num_workers}"
            slots[slot].append(axis_id)
        return {slot: owned for slot, owned in slots.items() if owned}

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
        PipelineCompiler._validate_compile_request(orchestrator, pipeline_definition)
        try:
            axis_values_to_process = PipelineCompiler._axis_values_to_process(
                orchestrator,
                axis_filter,
            )
            if not axis_values_to_process:
                logger.warning("No axis values found to process based on filter.")
                return {
                    "pipeline_definition": pipeline_definition,
                    "compiled_contexts": {},
                    "worker_assignments": {},
                }

            logger.info(
                f"Starting compilation for axis values: {', '.join(axis_values_to_process)}"
            )

            pipeline_config_state, pipeline_inputs = (
                PipelineCompiler._register_and_resolve_pipeline_once(
                    orchestrator,
                    pipeline_definition,
                    is_zmq_execution=is_zmq_execution,
                )
            )
            if not pipeline_definition:
                logger.warning(
                    "All steps were disabled. Pipeline is empty after filtering."
                )
                PipelineCompiler._cleanup_compilation_object_states(orchestrator)
                return {
                    "pipeline_definition": pipeline_definition,
                    "compiled_contexts": {},
                    "worker_assignments": {},
                }
            (
                analysis_config,
                plate_metadata_config,
                auto_add_output_plate,
                num_workers,
            ) = PipelineCompiler._capture_pipeline_config(pipeline_config_state)
            PipelineCompiler.validate_backend_compatibility(orchestrator)
            global_step_axis_filters = PipelineCompiler._resolve_global_step_axis_filters(
                orchestrator,
                pipeline_inputs.snapshots,
            )
            axis_request = AxisCompilationRequest(
                orchestrator=orchestrator,
                pipeline_definition=pipeline_definition,
                step_state_map=pipeline_inputs.step_state_map,
                step_snapshots=pipeline_inputs.snapshots,
                analysis_consolidation_config=analysis_config,
                plate_metadata_config=plate_metadata_config,
                auto_add_output_plate=auto_add_output_plate,
                global_step_axis_filters=global_step_axis_filters,
                enable_visualizer_override=enable_visualizer_override,
                is_zmq_execution=is_zmq_execution,
            )
            compiled_contexts = PipelineCompiler._compile_axis_values(
                axis_request,
                axis_values_to_process,
            )
            worker_assignments = PipelineCompiler._calculate_worker_assignments(
                list(compiled_contexts.keys()),
                num_workers,
            )
            PipelineCompiler._finalize_compilation(
                orchestrator,
                pipeline_definition,
                compiled_contexts,
            )
            return {
                "pipeline_definition": pipeline_definition,
                "compiled_contexts": compiled_contexts,
            }
        except Exception as e:
            orchestrator._state = OrchestratorState.COMPILE_FAILED
            logger.error(f"Failed to compile pipelines: {e}")
            raise

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
