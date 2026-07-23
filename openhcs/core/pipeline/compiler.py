"""
Pipeline module for OpenHCS.

This module provides core pipeline compilation components for OpenHCS.
The PipelineCompiler is responsible for preparing step_plans within a ProcessingContext.

CONFIGURATION ACCESS PATTERN:
============================
The compiler resolves ObjectState once and replaces the submitted pipeline with
the resolved steps before creating compiler snapshots:

CORRECT:
    # Steps are registered in ObjectState with parent hierarchy: step → orchestrator → global
    step_state = ObjectState(object_instance=step, scope_id=scope_id, parent_state=orch_state)
    ObjectStateRegistry.register(step_state)

    resolved_step = step_state.to_saved_resolved_object()
    snapshot = StepSnapshot(index=0, scope_id=step_state.scope_id, step=resolved_step)
    var_comps = snapshot.step.processing_config.variable_components

✅ CORRECT (LIVE VALUES FOR UI):
    # For UI: use get_resolved_value() to get current values with unsaved edits
    current_value = step_state.get_resolved_value(field_path)

REMOVED:
    with config_context(orchestrator.pipeline_config):  # REMOVED
        resolved_step = resolve_lazy_configurations_for_serialization(step)  # REMOVED

    # Using .parameters.get() doesn't get inheritance
    current_value = step_state.parameters.get(field_path)  # WRONG - no inheritance

    Compiler consumers must read semantics from the resolved step, not rebuild a
    projection from ObjectState.

WHY:
- ObjectState.to_saved_resolved_object() provides the resolved compiler step
- StepSnapshot carries only compiler index, scope identity, and that resolved step
- get_resolved_value() provides live state with unsaved edits (for UI)
- parameters.get() returns raw local value only, NO inheritance
- No cross-step pollution - each step only sees its own config hierarchy
- isinstance checks are the only type checking pattern (no hasattr)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

import dill as pickle

from openhcs.constants.constants import (
    AllComponents,
    get_multiprocessing_axis,
    OrchestratorState,
    VALID_GPU_MEMORY_TYPES,
    READ_BACKEND,
    WRITE_BACKEND,
    Backend,
)

from openhcs.core.compiled_execution import (
    CompiledExecutionBundle,
    CompiledGpuRegistryPlan,
    CompiledRuntimeEnvironmentPlan,
)
from openhcs.core.callable_contract import FunctionStepExecutionScope
from openhcs.core.context.processing_context import (
    ProcessingContext,
    RequiredVisualizer,
)
from openhcs.core.config import (
    MaterializationBackend,
    PipelineConfig,
    StreamingConfig,
    VFSConfig,
    WellFilterConfig,
)
from openhcs.core.debug import DebugExecutionPolicy, NoOpDebugExecutionPolicy
from openhcs.core.axis_filter import (
    StepAxisFilterMap,
    StepAxisFilterResolution,
    StepAxisFilterSet,
)
from openhcs.core.pipeline.funcstep_contract_validator import FuncStepContractValidator
from openhcs.core.pipeline.compilation_session import (
    CompilationPathResolver,
    CompilationPlateScope,
    CompilationSession,
    ResolvedPipelineDefinition,
    resolve_declared_dataclass_paths,
)
from openhcs.core.pipeline.materialization_flag_planner import (
    MaterializationFlagPlanner,
)
from openhcs.core.compiled_step_plan import (
    CompiledStepPlan,
    InputConversionPlan,
    RuntimeArtifactMaterializationPlan,
    SequentialRuntimeFilter,
    SequentialRuntimeFilterPlan,
)
from openhcs.core.pipeline.path_planner import (
    PathPlannerExecutionGroups,
    PipelinePathPlanner,
)
from openhcs.core.pipeline.step_snapshot import (
    StepSnapshot,
    build_step_snapshots,
)
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.core.source_workspace_projection import (
    VirtualWorkspaceSourceProjectionAuthority,
)
from openhcs.core.source_load_plan import SourceLoadPlan
from openhcs.core.invocation_artifacts import (
    PipelineInvocationContractProviderAuthority,
)
from openhcs.core.pipeline.gpu_memory_validator import GPUMemoryTypeValidator
from openhcs.core.pipeline.step_attribute_stripper import StepAttributeStripper
from openhcs.core.function_reference import FunctionReferenceTransportAuthority
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.utils import WellFilterProcessor
from objectstate import ObjectState, ObjectStateRegistry, get_base_type_for_lazy
from openhcs.core.steps.function_step import FunctionStep  # Used for isinstance check
from openhcs.core.progress import emit, ProgressPhase, ProgressStatus
from dataclasses import dataclass, replace

if TYPE_CHECKING:
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


def _step_scope_token(step: "AbstractStep", step_index: int) -> str:
    """Return the existing step scope token when available."""
    token = step._scope_token
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


def _compiler_pipeline_scope_id(
    plate_path_str: str,
    pipeline_definition: Sequence["AbstractStep"],
) -> str:
    """Build the compiler-owned ObjectState root for one submitted pipeline."""
    return f"{plate_path_str}::pipeline::submission_{id(pipeline_definition):x}"


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
class AxisCompilationRequest:
    """Authoritative context record for axis-level compilation fanout."""

    orchestrator: "PipelineOrchestrator"
    global_config: "GlobalPipelineConfig"
    pipeline_config: PipelineConfig
    pipeline: ResolvedPipelineDefinition
    path_resolver: CompilationPathResolver
    global_step_axis_filters: StepAxisFilterMap
    enable_visualizer_override: bool
    is_zmq_execution: bool

    def context_for(self, axis_id: str) -> ProcessingContext:
        context = self.orchestrator.create_context(axis_id)
        context.source_image_set_identity_policy = (
            SourceImageSetIdentityPolicy.from_pipeline_config(self.pipeline_config)
        )
        context.step_axis_filters = self.global_step_axis_filters
        context.analysis_consolidation_config = (
            self.global_config.analysis_consolidation_config
        )
        context.plate_metadata_config = self.global_config.plate_metadata_config
        return context


def _refresh_function_objects_in_steps(pipeline_definition: List[AbstractStep]) -> None:
    """Refresh all function objects in pipeline steps for picklable transport."""
    FunctionReferenceTransportAuthority.reference_pipeline_in_place(pipeline_definition)


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
        global_config: "GlobalPipelineConfig",
        step_state_map: Mapping[int, "ObjectState"],
        step_snapshots: tuple[StepSnapshot, ...],
        metadata_writer: bool = False,
        plate_path: Optional[Path] = None,
        path_resolver: CompilationPathResolver | None = None,
        is_zmq_execution: bool = False,
        # base_input_dir and axis_id parameters removed, will use from context
    ) -> CompilationSession:
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
            global_config: Resolved global compiler configuration
            step_state_map: ObjectState mapping from pipeline-level resolution
            step_snapshots: Snapshots of the same resolved pipeline steps

        Returns:
            The axis-scoped compilation session used by all downstream stages.
        """
        PipelineCompiler._assert_context_mutable_for_planning(context)
        context.visualizer_config = None
        logger.debug("Using pipeline-resolved steps for context %s", context.axis_id)
        session = CompilationSession.from_context(
            context=context,
            steps=steps_definition,
            orchestrator=orchestrator,
            global_config=global_config,
            step_state_map=step_state_map,
            snapshots=step_snapshots,
            source_workspace_projection=(
                orchestrator.source_workspace_projection().filtered_by_axis(
                    axis_id=context.axis_id,
                )
            ),
            metadata_writer=metadata_writer,
            plate_path=plate_path,
            path_resolver=path_resolver,
            is_zmq_execution=is_zmq_execution,
        )

        PipelineCompiler._ensure_initial_step_plans(session)
        PipelineCompiler._configure_input_conversion_if_needed(
            context,
            session.steps,
            orchestrator,
            plate_path,
        )
        PipelineCompiler._plan_context_paths(session)
        PipelineCompiler._supplement_step_plans(session)
        PipelineCompiler._collect_streaming_configs(session)
        return session

    @staticmethod
    def _assert_context_mutable_for_planning(context: ProcessingContext) -> None:
        if context.is_frozen():
            raise AttributeError(
                "Cannot initialize step plans in a frozen ProcessingContext."
            )
        if context.step_plans is None:
            context.step_plans = {}

    @staticmethod
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

    @staticmethod
    def _get_or_register_object_state(
        scope_id: str,
        object_instance,
        parent_state: Optional["ObjectState"],
        *,
        force_fresh: bool = False,
    ) -> "ObjectState":
        """Return an existing ObjectState unless a fresh compiler state is required."""
        state = None
        if not force_fresh:
            state = ObjectStateRegistry.get_by_scope(scope_id)
        if state is not None:
            return state
        return PipelineCompiler._register_object_state(
            object_instance,
            scope_id,
            parent_state,
        )

    @staticmethod
    def _missing_plan_fields(
        plan: CompiledStepPlan,
        requirements: Sequence[tuple[str, Callable[[CompiledStepPlan], object | None]]],
    ) -> list[str]:
        return [name for name, read_field in requirements if read_field(plan) is None]

    @staticmethod
    def _ensure_initial_step_plans(
        session: CompilationSession,
    ) -> None:
        for step_index, snapshot in enumerate(session.snapshots):
            if step_index not in session.plans:
                session.plans[step_index] = CompiledStepPlan(
                    step_index=step_index,
                    step_name=snapshot.step.name,
                    step_type=type(snapshot.step).__name__,
                    axis_id=session.axis_id,
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

        from openhcs.microscopes.openhcs import (
            OpenHCSMetadataHandler,
            get_subdirectory_name,
        )

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
        session: CompilationSession,
    ) -> None:
        PipelinePathPlanner.prepare_pipeline_paths(
            session,
            invocation_contract_provider=(
                PipelineInvocationContractProviderAuthority.provider_for_session(
                    session
                )
            ),
        )

    @staticmethod
    def _supplement_step_plans(session: CompilationSession) -> None:
        for step_index, snapshot in enumerate(session.snapshots):
            if step_index not in session.plans:
                logger.error(
                    "Critical error: Step %s (index: %s) not found in step_plans after path planning phase.",
                    snapshot.step.name,
                    step_index,
                )
                session.plans[step_index] = CompiledStepPlan(
                    step_index=step_index,
                    step_name=snapshot.step.name,
                    step_type=type(snapshot.step).__name__,
                    axis_id=session.axis_id,
                    error="Missing from path planning phase by PipelinePathPlanner",
                    create_openhcs_metadata=session.metadata_writer,
                )
                continue

            current_plan = session.plans[step_index]
            current_plan.step_scope_id = snapshot.scope_id
            current_plan.step_name = snapshot.step.name
            current_plan.step_type = type(snapshot.step).__name__
            current_plan.axis_id = session.axis_id
            current_plan.create_openhcs_metadata = session.metadata_writer
            current_plan.variable_components = (
                snapshot.step.processing_config.variable_components
            )
            current_plan.group_by = PathPlannerExecutionGroups.normalized_group_by(
                snapshot,
            )
            declared_group_by = snapshot.step.processing_config.group_by
            if current_plan.group_by is not declared_group_by:
                logger.warning(
                    "Step %r uses a non-routed function pattern with group_by=%s "
                    "also present in variable_components; compiled group_by is "
                    "GroupBy.NONE because variable_components owns the runtime "
                    "stack axis.",
                    snapshot.step.name,
                    declared_group_by.name,
                )
            current_plan.input_source = snapshot.step.processing_config.input_source
            current_plan.sequential_processing = snapshot.step.processing_config
            current_plan.sequential_filter_plan = (
                PipelineCompiler._compile_sequential_runtime_filter_plan(session)
            )
            current_plan.source_load_plan = SourceLoadPlan(
                zarr_config=session.global_config.zarr_config
            )

    @staticmethod
    def _compile_sequential_runtime_filter_plan(
        session: CompilationSession,
    ) -> SequentialRuntimeFilterPlan:
        current_combination = session.context.current_sequential_combination
        if not current_combination:
            return SequentialRuntimeFilterPlan.disabled()

        seq_config = session.global_config.sequential_processing_config
        if not seq_config or not seq_config.sequential_components:
            return SequentialRuntimeFilterPlan.disabled()

        filtered_components = tuple(
            seq_component
            for seq_component in seq_config.sequential_components
            if len(
                session.orchestrator.get_component_keys(
                    AllComponents(seq_component.value)
                )
            )
            > 1
        )
        if len(filtered_components) != len(current_combination):
            raise ValueError(
                "Sequential runtime filter cardinality mismatch: "
                f"{len(filtered_components)} components for "
                f"{len(current_combination)} active values."
            )
        return SequentialRuntimeFilterPlan(
            tuple(
                SequentialRuntimeFilter(seq_component, str(value))
                for seq_component, value in zip(
                    filtered_components,
                    current_combination,
                    strict=True,
                )
            )
        )

    @staticmethod
    def _collect_streaming_configs(
        session: CompilationSession,
    ) -> None:
        streaming_config_types = tuple(StreamingConfig.__registry__.values())
        for step_index, snapshot in enumerate(session.snapshots):
            step_plan = session.plans[step_index]
            for config_type in streaming_config_types:
                PipelineCompiler._collect_streaming_config(
                    session,
                    step_index,
                    snapshot,
                    step_plan,
                    config_type,
                )

    @staticmethod
    def _collect_streaming_config(
        session: CompilationSession,
        step_index: int,
        snapshot: StepSnapshot,
        step_plan: CompiledStepPlan,
        config_type: type[StreamingConfig],
    ) -> None:
        resolved_config_type = get_base_type_for_lazy(config_type) or config_type
        configs = tuple(
            value
            for value in vars(snapshot.step).values()
            if isinstance(value, resolved_config_type)
        )
        if not configs:
            return
        defaults_enabled = snapshot.step.streaming_defaults.enabled
        for config_obj in configs:
            per_stream_enabled = config_obj.enabled
            enabled = True if defaults_enabled is True else per_stream_enabled
            compiled_pattern = step_plan.compiled_function_pattern
            runtime_required = bool(
                compiled_pattern is not None
                and any(
                    binding.value is config_obj
                    for invocation in compiled_pattern.iter_invocations()
                    for binding in invocation.runtime_parameter_bindings
                )
            )
            config_key = config_obj.streaming_config_key
            if session.is_zmq_execution:
                logger.info(
                    "Streaming resolution: step=%s field=%s defaults_enabled=%r per_stream_enabled=%r effective_enabled=%r",
                    step_index,
                    config_key,
                    defaults_enabled,
                    per_stream_enabled,
                    enabled,
                )
            if enabled is not True and not runtime_required:
                continue

            backend_name = config_obj.backend.name
            required_visualizer = RequiredVisualizer(
                backend_name=backend_name,
                config=replace(config_obj, enabled=enabled is True),
            )
            if required_visualizer not in session.context.required_visualizers:
                session.context.required_visualizers.append(required_visualizer)
                logger.info(
                    "Streaming enabled for step %s, field %s (backend=%s)",
                    step_index,
                    config_key,
                    backend_name,
                )

            if enabled is True:
                step_plan.streaming_configs[config_key] = config_obj

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
                    f"Step '{snapshot.step.name}' will use zarr backend for axis {context.axis_id}"
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
            session.global_config,  # Use merged config instead of raw pipeline_config
            available_axis_values=session.orchestrator.get_component_keys(
                get_multiprocessing_axis()
            ),
        )

        # Post-check (optional, but good for ensuring contracts are met by the planner)
        for step_index, snapshot in enumerate(session.snapshots):
            if step_index not in context.step_plans:
                # This should not happen if prepare_pipeline_flags guarantees plans for all steps
                logger.error(
                    f"Step {snapshot.step.name} (index: {step_index}) missing from step_plans after materialization planning."
                )
                continue

            plan = context.step_plans[step_index]
            # Check for keys that FunctionStep actually uses during execution
            missing_keys = PipelineCompiler._missing_plan_fields(
                plan,
                MATERIALIZATION_PLAN_REQUIREMENTS,
            )
            if missing_keys:
                logger.error(
                    f"Materialization flag planning incomplete for step {snapshot.step.name} (index: {step_index}). "
                    f"Missing required keys: {missing_keys}."
                )
        PipelineCompiler._compile_runtime_artifact_materialization_plans(session)

    @staticmethod
    def _compile_runtime_artifact_materialization_plans(
        session: CompilationSession,
    ) -> None:
        globally_enabled = bool(session.global_config.materialize_runtime_artifacts)
        persistent_backend = None
        persistent_step_indexes: set[int] = set()
        for step_index, step_plan in session.plans.items():
            has_explicit_output = any(
                output.materialization is not None
                and output.materialization.participates_in_persistent_materialization()
                for output in step_plan.artifact_outputs.values()
            )
            if globally_enabled or has_explicit_output:
                persistent_step_indexes.add(step_index)

        if persistent_step_indexes:
            persistent_backend = (
                MaterializationFlagPlanner._resolve_materialization_backend(
                    session.context,
                    session.global_config.vfs_config,
                )
            )
        for step_index, step_plan in session.plans.items():
            persistent_enabled = step_index in persistent_step_indexes
            step_plan.runtime_artifact_materialization = (
                RuntimeArtifactMaterializationPlan(
                    persistent_enabled=persistent_enabled,
                    persistent_backend=(
                        persistent_backend if persistent_enabled else None
                    ),
                )
            )

    @staticmethod
    def validate_sequential_components_compatibility(
        step_snapshots: Sequence[StepSnapshot],
        sequential_components: List,
    ) -> None:
        """
        Validate that no step's variable_components overlap with pipeline's sequential_components.

        Args:
            step_snapshots: ObjectState-resolved compiler snapshots
            sequential_components: List of SequentialComponents from pipeline config

        Raises:
            ValueError: If any step has variable_components that overlap with sequential_components
        """
        if not sequential_components:
            return

        seq_comp_values = {sc.value for sc in sequential_components}

        for snapshot in step_snapshots:
            if not isinstance(snapshot.step, FunctionStep):
                continue
            var_comps = snapshot.step.processing_config.variable_components
            if not var_comps:
                continue
            var_comp_values = {vc.value for vc in var_comps}
            overlap = seq_comp_values & var_comp_values

            if overlap:
                raise ValueError(
                    f"Step '{snapshot.step.name}' has variable_components {sorted(overlap)} that conflict with "
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
            orchestrator=session.orchestrator,  # Pass orchestrator for dict pattern key validation
        )

        for step_index, step in enumerate(session.steps):
            if not isinstance(step, FunctionStep):
                continue
            if step_index not in context.step_plans:
                raise AssertionError(
                    f"Memory validation requires a compiled plan for FunctionStep {session.snapshot(step_index).step.name} (index: {step_index})."
                )
            step_plan = context.step_plans[step_index]
            compiled_pattern = step_plan.compiled_function_pattern
            if (
                compiled_pattern is not None
                and compiled_pattern.execution_scope is FunctionStepExecutionScope.PLATE
            ):
                continue
            missing_fields = PipelineCompiler._missing_plan_fields(
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
                    compiled_pattern = step_plan.compiled_function_pattern
                    if (
                        compiled_pattern is not None
                        and compiled_pattern.execution_scope
                        is FunctionStepExecutionScope.PLATE
                    ):
                        continue
                    write_backend = step_plan.write_backend

                    if write_backend == "disk":
                        logger.debug(
                            f"Step {session.snapshot(step_index).step.name} has disk output, overriding output_memory_type to numpy"
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
    def validate_backend_compatibility(
        orchestrator,
        vfs_config: VFSConfig,
    ) -> None:
        """
        Validate configured read backend against microscope support.

        Materialization backend selection is always allowed at compile time (e.g. materialize
        to Zarr even when source data is read from disk). What must be compatible with the
        selected microscope is the backend used for reading input images.

        Args:
            orchestrator: PipelineOrchestrator instance with initialized microscope_handler
        """

        microscope_handler = orchestrator.microscope_handler

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
        if pipeline_definition is None:
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
            effective_config.well_filter_config if effective_config else None
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
    ) -> tuple[str, "ObjectState", ResolvedPipelineDefinition]:
        # Compile from the submitted pipeline definition, not from any stale UI
        # ObjectState that may point at post-compile stripped step shells.
        force_fresh = True
        global_config_state = PipelineCompiler._compile_global_config_state(
            force_fresh=force_fresh
        )
        plate_path_str = str(orchestrator.plate_path)
        compiler_scope_id = _compiler_pipeline_scope_id(
            plate_path_str,
            pipeline_definition,
        )
        plate_orch_state = PipelineCompiler._pipeline_config_state(
            orchestrator,
            compiler_scope_id,
            global_config_state,
            force_fresh=force_fresh,
        )
        orchestrator_scope_id = f"{compiler_scope_id}::orchestrator"
        orch_state = PipelineCompiler._get_or_register_object_state(
            orchestrator_scope_id,
            orchestrator,
            plate_orch_state,
            force_fresh=force_fresh,
        )
        logger.debug("Registered orchestrator at scope: %s", orchestrator_scope_id)

        step_state_map = PipelineCompiler._register_pipeline_step_states(
            pipeline_definition,
            compiler_scope_id,
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
        pipeline_config_state = ObjectStateRegistry.get_by_scope(compiler_scope_id)
        if pipeline_config_state is None:
            raise RuntimeError(
                "Missing ObjectState for plate; cannot resolve pipeline config."
            )
        return (
            compiler_scope_id,
            pipeline_config_state,
            ResolvedPipelineDefinition(
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
                global_config_state = PipelineCompiler._register_object_state(
                    global_config,
                    "",
                    None,
                )
                logger.debug("Registered global config at scope ''")
        return global_config_state

    @staticmethod
    def _pipeline_config_state(
        orchestrator,
        compiler_scope_id: str,
        global_config_state: "ObjectState" | None,
        *,
        force_fresh: bool,
    ) -> "ObjectState" | None:
        plate_orch_state = ObjectStateRegistry.get_by_scope(compiler_scope_id)
        if orchestrator.pipeline_config:
            plate_orch_state = PipelineCompiler._get_or_register_object_state(
                compiler_scope_id,
                orchestrator.pipeline_config,
                global_config_state,
                force_fresh=force_fresh,
            )
            logger.debug("Registered pipeline_config at scope '%s'", compiler_scope_id)
        return plate_orch_state

    @staticmethod
    def _register_pipeline_step_states(
        pipeline_definition: Sequence[AbstractStep],
        compiler_scope_id: str,
        orch_state: "ObjectState",
        *,
        force_fresh: bool,
    ) -> Dict[int, "ObjectState"]:
        step_state_map: Dict[int, "ObjectState"] = {}
        for step_index, step in enumerate(pipeline_definition):
            step_scope_id = _compiler_step_scope_id(
                compiler_scope_id,
                step,
                step_index,
            )
            step_state_map[step_index] = PipelineCompiler._get_or_register_object_state(
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
            step_state.to_saved_resolved_object()
            for step_state in step_state_map.values()
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
            if step.enabled
        ]
        pipeline_definition.clear()
        pipeline_definition.extend(step for step, _state in enabled_pairs)
        enabled_state_map = {
            new_index: state for new_index, (_step, state) in enumerate(enabled_pairs)
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
    ) -> PipelineConfig:
        pipeline_config = pipeline_config_state.to_object(update_delegate=False)
        if not isinstance(pipeline_config, PipelineConfig):
            raise TypeError(
                "Compiler pipeline ObjectState must reconstruct PipelineConfig; "
                f"got {type(pipeline_config).__name__}."
            )
        return pipeline_config

    @staticmethod
    def _resolve_global_step_axis_filters(
        orchestrator,
        step_snapshots: tuple[StepSnapshot, ...],
        step_state_map: Mapping[int, "ObjectState"],
    ) -> StepAxisFilterMap:
        temp_context = orchestrator.create_context("temp")
        _resolve_step_axis_filters(
            step_snapshots,
            step_state_map,
            temp_context,
            orchestrator,
        )
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
            temp_session.global_config,
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
        return PipelineCompiler.initialize_step_plans_for_context(
            context,
            list(request.pipeline.steps),
            request.orchestrator,
            request.global_config,
            dict(request.pipeline.step_state_map),
            request.pipeline.snapshots,
            metadata_writer=metadata_writer,
            plate_path=request.orchestrator.plate_path,
            path_resolver=request.path_resolver,
            is_zmq_execution=request.is_zmq_execution,
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
            session.global_config,
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
        PipelineCompiler.validate_source_workspace_projection(session)
        PipelineCompiler.assign_gpu_resources(session)
        if enable_visualizer_override:
            PipelineCompiler.apply_global_visualizer_override(
                session,
                True,
            )
        PipelineCompiler.resolve_lazy_dataclasses(session)

    @staticmethod
    def validate_source_workspace_projection(session: CompilationSession) -> None:
        """Validate source-workspace metadata before runtime image loading."""

        projection = VirtualWorkspaceSourceProjectionAuthority.from_context(
            session.context,
            cache=session.context.runtime_source_workspace_projection_cache,
        ).projection_if_available()
        if projection is None:
            return
        projection.validate_runtime_metadata_projection(axis_id=session.axis_id)

    @staticmethod
    def _validate_sequential_components_for_session(
        session: CompilationSession,
    ) -> None:
        seq_config = session.global_config.sequential_processing_config
        if seq_config and seq_config.sequential_components:
            PipelineCompiler.validate_sequential_components_compatibility(
                session.snapshots,
                seq_config.sequential_components,
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
        compiler_scope_id: str,
    ) -> None:
        PipelineCompiler._log_path_planning_summary(compiled_contexts)
        PipelineCompiler._cleanup_compilation_object_states(
            orchestrator,
            compiler_scope_id,
        )
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
    def _cleanup_compilation_object_states(
        orchestrator,
        compiler_scope_id: str | None = None,
    ) -> None:
        orch_scope_id = (
            compiler_scope_id
            if compiler_scope_id is not None
            else f"{orchestrator.plate_path}::orchestrator"
        )
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
        debug_execution_policy: DebugExecutionPolicy = NoOpDebugExecutionPolicy(),
    ) -> Dict[str, Any]:
        """
        Compile-all phase: prepares execution artifacts for each axis value.

        This method iterates through specified axis values, creates a ProcessingContext
        for each, and invokes various phases of PipelineCompiler to populate
        context's step_plans. After all compilation phases for an axis value are complete,
        its context is frozen. It also builds the worker-transport contexts and worker
        ownership map that execution consumes without re-resolving lazy configuration.

        Args:
            orchestrator: The PipelineOrchestrator instance to use for compilation
            pipeline_definition: The list of AbstractStep objects defining the pipeline.
            axis_filter: Optional list of axis values to process. If None, processes all found axis values.
            enable_visualizer_override: If True, all steps in all compiled contexts
                                        will have their 'visualize' flag set to True.
            is_zmq_execution: If True, compiler-created ObjectStates will be unregistered
                              after resolution to free RAM (for ZMQ server mode).

        Returns:
            A compilation result containing the compiler-owned
            CompiledExecutionBundle plus compatibility projections for compiled
            contexts, worker assignments, and the stateless pipeline definition.
        """
        PipelineCompiler._validate_compile_request(orchestrator, pipeline_definition)
        compiler_scope_id: str | None = None
        try:
            axis_values_to_process = PipelineCompiler._axis_values_to_process(
                orchestrator,
                axis_filter,
            )
            if not axis_values_to_process:
                logger.warning("No axis values found to process based on filter.")
                return CompiledExecutionBundle.from_runtime_contexts(
                    pipeline_definition=pipeline_definition,
                    runtime_contexts={},
                    worker_assignments={},
                    runtime_environment=CompiledRuntimeEnvironmentPlan.from_global_config(
                        orchestrator.get_effective_config(),
                        compiled_contexts={},
                        server_mode=is_zmq_execution,
                    ),
                ).as_compilation_result()

            logger.info(
                f"Starting compilation for axis values: {', '.join(axis_values_to_process)}"
            )

            compiler_scope_id, pipeline_config_state, pipeline_inputs = (
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
                PipelineCompiler._cleanup_compilation_object_states(
                    orchestrator,
                    compiler_scope_id,
                )
                return CompiledExecutionBundle.from_runtime_contexts(
                    pipeline_definition=pipeline_definition,
                    runtime_contexts={},
                    worker_assignments={},
                    runtime_environment=CompiledRuntimeEnvironmentPlan.from_global_config(
                        orchestrator.get_effective_config(),
                        compiled_contexts={},
                        server_mode=is_zmq_execution,
                    ),
                ).as_compilation_result()
            pipeline_config = PipelineCompiler._capture_pipeline_config(
                pipeline_config_state
            )
            path_resolver = CompilationPathResolver(
                plate_scope=CompilationPlateScope.from_path(
                    orchestrator.plate_path
                ),
                filemanager=orchestrator.filemanager,
                backend=Backend.DISK.value,
            )
            pipeline_config = resolve_declared_dataclass_paths(
                pipeline_config,
                path_resolver,
                owner="PipelineConfig",
            )
            num_workers = pipeline_config.num_workers
            num_workers = debug_execution_policy.compile_worker_count(num_workers)
            effective_config = resolve_declared_dataclass_paths(
                orchestrator.get_effective_config(),
                path_resolver,
                owner="GlobalPipelineConfig",
            )
            resolved_steps = tuple(
                resolve_declared_dataclass_paths(
                    step,
                    path_resolver,
                    owner=f"pipeline step {index}",
                )
                for index, step in enumerate(pipeline_inputs.steps)
            )
            pipeline_inputs = replace(
                pipeline_inputs,
                steps=resolved_steps,
                snapshots=build_step_snapshots(
                    resolved_steps,
                    pipeline_inputs.step_state_map,
                ),
            )
            CompiledGpuRegistryPlan(
                configured_num_workers=num_workers
            ).setup_global_registry()
            PipelineCompiler.validate_backend_compatibility(
                orchestrator,
                pipeline_config.vfs_config,
            )
            global_step_axis_filters = (
                PipelineCompiler._resolve_global_step_axis_filters(
                    orchestrator,
                    pipeline_inputs.snapshots,
                    pipeline_inputs.step_state_map,
                )
            )
            axis_request = AxisCompilationRequest(
                orchestrator=orchestrator,
                global_config=effective_config,
                pipeline_config=pipeline_config,
                pipeline=pipeline_inputs,
                path_resolver=path_resolver,
                global_step_axis_filters=global_step_axis_filters,
                enable_visualizer_override=enable_visualizer_override,
                is_zmq_execution=is_zmq_execution,
            )
            compiled_contexts = PipelineCompiler._compile_axis_values(
                axis_request,
                axis_values_to_process,
            )
            from openhcs.core.steps.function_runtime import (
                prepare_compiled_context_callables,
            )

            prepare_compiled_context_callables(compiled_contexts)
            worker_assignments = PipelineCompiler._calculate_worker_assignments(
                list(compiled_contexts.keys()),
                num_workers,
            )
            runtime_environment = debug_execution_policy.compiled_runtime_environment(
                CompiledRuntimeEnvironmentPlan.from_global_config(
                    effective_config,
                    compiled_contexts=compiled_contexts,
                    server_mode=is_zmq_execution,
                )
            )
            PipelineCompiler._finalize_compilation(
                orchestrator,
                pipeline_definition,
                compiled_contexts,
                compiler_scope_id,
            )
            execution_bundle = CompiledExecutionBundle.from_runtime_contexts(
                pipeline_definition=pipeline_definition,
                runtime_contexts=compiled_contexts,
                worker_assignments=worker_assignments,
                runtime_environment=runtime_environment,
            )
            PipelineCompiler._write_compilation_debug_bundle_if_configured(
                pipeline_config,
                execution_bundle,
            )
            return execution_bundle.as_compilation_result()
        except Exception as e:
            if compiler_scope_id is not None:
                PipelineCompiler._cleanup_compilation_object_states(
                    orchestrator,
                    compiler_scope_id,
                )
            orchestrator._state = OrchestratorState.COMPILE_FAILED
            logger.error(f"Failed to compile pipelines: {e}")
            raise

    @staticmethod
    def _write_compilation_debug_bundle_if_configured(
        pipeline_config: PipelineConfig,
        execution_bundle: CompiledExecutionBundle,
    ) -> None:
        compilation_debug_config = pipeline_config.compilation_debug_config
        if not compilation_debug_config.enabled:
            return
        dump_path = compilation_debug_config.compiled_execution_bundle_path
        if dump_path is None:
            logger.warning(
                "CompilationDebugConfig enabled but no "
                "compiled_execution_bundle_path was set."
            )
            return
        dump_path = Path(dump_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        transport_bundle = execution_bundle.for_transport_serialization()
        with dump_path.open("wb") as handle:
            pickle.dump(
                transport_bundle.transport_contexts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def _resolve_step_axis_filters(
    step_snapshots: tuple[StepSnapshot, ...],
    step_state_map: Mapping[int, "ObjectState"],
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
        step_state_map: ObjectState owners retaining inherited-field provenance
        context: Processing context for the current axis value
        orchestrator: Orchestrator instance with access to available axis values
    """

    # Get available axis values from orchestrator using multiprocessing axis

    available_axis_values = orchestrator.get_component_keys(get_multiprocessing_axis())
    if not available_axis_values:
        logger.warning("No available axis values found for axis filter resolution")
        return

    for snapshot in step_snapshots:
        step_filters: dict[type[WellFilterConfig], StepAxisFilterResolution] = {}
        step_state = step_state_map[snapshot.index]
        for field_name, config in vars(snapshot.step).items():
            if not isinstance(config, WellFilterConfig) or config.well_filter is None:
                continue
            provenance = step_state.get_provenance(f"{field_name}.well_filter")
            source_type = provenance[1] if provenance is not None else None
            if not config.accepts_well_filter_provenance(source_type):
                logger.debug(
                    "Ignoring %s.%s inherited from sibling policy %s",
                    type(config).__name__,
                    "well_filter",
                    source_type.__name__ if source_type is not None else "unknown",
                )
                continue
            resolved_axis_values = WellFilterProcessor.resolve_filter_with_mode(
                config.well_filter,
                config.well_filter_mode,
                available_axis_values,
            )
            step_filters[type(config)] = StepAxisFilterResolution(
                resolved_axis_values=frozenset(
                    str(value) for value in resolved_axis_values
                ),
                filter_mode=config.well_filter_mode,
                original_filter=config.well_filter,
            )

        if step_filters:
            context.step_axis_filters[snapshot.index] = StepAxisFilterSet(step_filters)

    total_filters = sum(len(filters) for filters in context.step_axis_filters.values())
    logger.debug(
        f"Axis filter resolution complete. {len(context.step_axis_filters)} steps have axis filters, {total_filters} total filters."
    )
