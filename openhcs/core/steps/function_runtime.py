"""Runtime execution helpers for FunctionStep.

This module owns callable invocation, artifact routing, and pattern-group stack
execution. FunctionStep remains responsible for step-level orchestration.
"""

from abc import ABC, abstractmethod
import inspect
import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence
from weakref import WeakKeyDictionary

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import (
    ArtifactKind,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    StepResult,
)
from openhcs.core.callable_contract import prepare_processing_callable
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.debug import (
    DebugCursor,
    DebugEvent,
    DebugEventType,
    DebugArtifactRefProjection,
    DebugInvocationParameter,
    debug_event_sink_from_context,
)
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    DEFAULT_GROUP_KEY,
)
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import (
    convert_memory,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactGroupTarget,
    RuntimeArtifactLocation,
    RuntimeArtifactLocationTarget,
    RuntimeArtifactQuery,
    require_runtime_value_store,
    replace_runtime_artifact_payload,
)
from openhcs.core.runtime_adapters import RuntimeAdapterRequest, RuntimeAdapterSpec
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.core.source_image_semantics import apply_source_image_loading_semantics
from openhcs.core.source_schema_workspace import (
    source_schema_metadata_with_virtual_components,
)
from openhcs.core.source_matching import (
    source_component_metadata_values,
    source_metadata_values_equal,
)
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    SourceBindingOrigin,
    SourceBindingRuntimeContext,
)
from openhcs.core.runtime_values import (
    DerivedImagePayloadContext,
    ImagePayloadMetadata,
    ObjectLabelPayload,
    ObjectLabelSet,
    SourceImageObjectLabelBuildRequest,
    image_payload_data,
    image_payload_metadata,
    image_payload_mask,
    image_payload_with_context,
    is_array_payload,
    normalize_artifact_value,
    with_image_payload_data,
)
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_semantics import RuntimePlaneProjection
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan

logger = logging.getLogger(__name__)


class FunctionInvocationCallableResolver:
    """Process-local resolver for compiled invocation callables.

    The compiler stores picklable ``FunctionReference`` objects in compiled
    invocations. Runtime execution needs actual callables. Resolving them during
    compiler preparation lets fork workers inherit the resolved callable cache,
    while spawn workers still resolve lazily in their own process.
    """

    _lock = Lock()
    _cache: dict[object, Callable] = {}

    @classmethod
    def prepare(cls, invocation: CompiledFunctionInvocation) -> None:
        """Resolve and cache one invocation callable before timed execution."""
        prepare_processing_callable(cls.resolve(invocation))

    @classmethod
    def resolve(cls, invocation: CompiledFunctionInvocation) -> Callable:
        """Return the callable for a compiled invocation."""
        cache_key = cls.cache_key(invocation)
        with cls._lock:
            cached = cls._cache.get(cache_key)
        if cached is not None:
            return cached

        from openhcs.core.pipeline.compiler import FunctionReference

        if isinstance(invocation.func, FunctionReference):
            from openhcs.core.function_reference_rehydration import (
                FunctionReferenceRehydrationRequest,
                FunctionReferenceRehydrator,
            )

            resolved = FunctionReferenceRehydrator.rehydrate_reference(
                FunctionReferenceRehydrationRequest(
                    reference=invocation.func,
                    contract=invocation.contract,
                    resolved_callable=invocation.func.resolve(),
                )
            )
        elif callable(invocation.func):
            resolved = invocation.func
        else:
            raise TypeError(f"Invalid compiled invocation function: {invocation.func}")

        with cls._lock:
            cls._cache[cache_key] = resolved
        return resolved

    @classmethod
    def cache_key(cls, invocation: CompiledFunctionInvocation) -> object:
        """Return process-local callable cache key for one compiled invocation."""
        from openhcs.core.pipeline.compiler import FunctionReference

        if isinstance(invocation.func, FunctionReference):
            return (
                invocation.func.composite_key,
                cls.contract_cache_key(invocation.contract),
            )
        if callable(invocation.func):
            return id(invocation.func)
        raise TypeError(f"Invalid compiled invocation function: {invocation.func}")

    @staticmethod
    def contract_cache_key(contract: Any) -> object:
        """Return semantic callable-contract identity for contextual rehydration.

        Registry references can point many generated steps at the same importable
        function. The module artifact contract is the runtime context that
        distinguishes those steps, so it must be part of the cache identity.
        """
        return (
            contract.module_artifact_contract,
            contract.declared_processing_contract,
            contract.processing_contract,
            contract.runtime_adapter,
            contract.runtime_image_execution_mode,
        )
_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
_PROFILE_RUNTIME_PATH_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME_PATH"
PROCESSING_CONTEXT_OWNER_NAME = ProcessingContext.__name__
ArtifactInputPlans = Mapping[str, ArtifactInputPlan]
ArtifactOutputPlans = Mapping[str, ArtifactOutputPlan]
_CALLABLE_PARAMETER_NAMES: WeakKeyDictionary[Callable, frozenset[str]] = (
    WeakKeyDictionary()
)


@dataclass(frozen=True, slots=True)
class RuntimeArtifactInvocationGroupKey:
    """Resolve artifact group identity for one runtime invocation."""

    invocation_group_key: str
    component_value: Any

    def artifact_group_key(self) -> str:
        if (
            self.invocation_group_key == DEFAULT_GROUP_KEY
            and self.component_value is not None
        ):
            return str(self.component_value)
        return self.invocation_group_key


def _runtime_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def _log_runtime_profile(label: str, seconds: float, **fields: Any) -> None:
    if not _runtime_profile_enabled():
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)
    if profile_path := os.environ.get(_PROFILE_RUNTIME_PATH_ENV):
        with open(profile_path, "a", encoding="utf-8") as handle:
            handle.write(f"RUNTIME_PROFILE {label} {seconds:.6f}s {field_text}\n")


@dataclass(frozen=True, slots=True)
class RuntimeProfileRecorder:
    """Function-scoped runtime profile field authority."""

    function_name: str

    def record_elapsed(
        self,
        label: str,
        *,
        started_at: float,
        **fields: Any,
    ) -> None:
        _log_runtime_profile(
            label,
            time.perf_counter() - started_at,
            function=self.function_name,
            **fields,
        )

    def record_artifact_elapsed(
        self,
        label: str,
        *,
        started_at: float,
        artifact_name: str,
        artifact_kind: str,
    ) -> None:
        self.record_elapsed(
            label,
            started_at=started_at,
            artifact=artifact_name,
            kind=artifact_kind,
        )

    def record_adapter_elapsed(
        self,
        *,
        started_at: float,
        adapter_name: str,
    ) -> None:
        self.record_elapsed(
            "runtime_adapter_factory",
            started_at=started_at,
            adapter=adapter_name,
        )


def _callable_parameter_names(func: Callable) -> frozenset[str]:
    """Return cached callable parameter names for runtime adapter injection."""
    names = _CALLABLE_PARAMETER_NAMES.get(func)
    if names is None:
        names = frozenset(inspect.signature(func).parameters)
        _CALLABLE_PARAMETER_NAMES[func] = names
    return names


@dataclass(frozen=True)
class FunctionExecutionRequest:
    """Nominal request for one callable invocation."""

    func_callable: Callable
    main_data_arg: Any
    base_kwargs: Mapping[str, Any]
    context: ProcessingContext
    artifact_inputs: ArtifactInputPlans
    artifact_outputs: ArtifactOutputPlans
    runtime_adapter: RuntimeAdapterSpec | None = None
    invocation_options: RuntimeInvocationOptions | None = None
    source_binding_plan: CompiledSourceBindingPlan = CompiledSourceBindingPlan.empty()
    source_binding_context: SourceBindingRuntimeContext = (
        SourceBindingRuntimeContext.empty()
    )
    group_key: str | None = None
    axis_component: str | None = None
    axis_component_value: str | None = None
    plane_projection: RuntimePlaneProjection = field(
        default_factory=RuntimePlaneProjection.stack
    )


@dataclass(frozen=True)
class FunctionChainExecutionRequest:
    """Nominal request for a chain of callables over one image stack."""

    initial_data_stack: Any
    invocations: Sequence[CompiledFunctionInvocation]
    context: ProcessingContext
    execution_plan: FunctionStepExecutionPlan
    artifact_inputs: ArtifactInputPlans
    artifact_outputs: ArtifactOutputPlans
    runtime_plane_index: int
    component_value: Any = None
    source_binding_context: SourceBindingRuntimeContext = (
        SourceBindingRuntimeContext.empty()
    )

@dataclass(frozen=True, slots=True)
class FunctionOutputContextRequest:
    """Source/output pair for runtime output context preservation."""

    source_payload: Any
    output_value: Any
    output_plan: ArtifactOutputPlan | None = None

    @property
    def kind(self) -> ArtifactKind:
        if self.output_plan is None:
            return ArtifactKind.IMAGE
        return self.output_plan.kind


class FunctionOutputContextStrategy(
    EnumKeyedStrategyMixin[ArtifactKind],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered normalization for function outputs before chaining or storage."""

    __registry_key__ = "kind_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "kind"
    __enum_label_attr__ = "kind_label"

    kind: ClassVar[ArtifactKind | None] = None
    kind_label: ClassVar[str | None] = None

    @classmethod
    def for_output(
        cls,
        request: FunctionOutputContextRequest,
    ) -> "FunctionOutputContextStrategy":
        strategy_type = cls.__registry__.get(request.kind.value)
        if strategy_type is None:
            return UnchangedFunctionOutputContextStrategy()
        return strategy_type()

    @abstractmethod
    def contextualize(self, request: FunctionOutputContextRequest) -> Any:
        """Return output with source context preserved where semantics allow it."""


class UnchangedFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Leave outputs unchanged when no contextual image semantics are declared."""

    kind = ArtifactKind.SPECIAL

    def contextualize(self, request: FunctionOutputContextRequest) -> Any:
        return request.output_value


class ImageFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Preserve source-image metadata for image outputs derived from the main input."""

    kind = ArtifactKind.IMAGE

    def contextualize(self, request: FunctionOutputContextRequest) -> Any:
        if image_payload_metadata(request.output_value).has_values:
            return request.output_value
        return DerivedImagePayloadContext(
            request.source_payload,
            request.output_value,
        ).payload()


class ObjectLabelsFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Preserve source-image metadata for object-label outputs."""

    kind = ArtifactKind.OBJECT_LABELS

    def contextualize(self, request: FunctionOutputContextRequest) -> Any:
        if isinstance(request.output_value, (ObjectLabelPayload, ObjectLabelSet)):
            return request.output_value
        if not is_array_payload(request.output_value):
            return request.output_value
        return SourceImageObjectLabelBuildRequest(
            image=request.source_payload,
            labels=request.output_value,
        ).payload()


@dataclass(frozen=True)
class ComponentArtifactPlans:
    """Artifact plans selected for one grouped component execution."""

    inputs: ArtifactInputPlans
    outputs: ArtifactOutputPlans


@dataclass(frozen=True)
class PatternGroupExecutionRequest:
    """All runtime data needed to process one pattern group."""

    context: ProcessingContext
    execution_plan: FunctionStepExecutionPlan
    pattern_group_info: Any
    compiled_group: CompiledFunctionGroup
    component_value: Any
    component_index: int


@dataclass(frozen=True)
class PatternGroupData:
    """Loaded image data for one pattern group."""

    matching_files: list[str]
    main_data_stack: Any
    source_slice_shapes: tuple[tuple[int, ...], ...]
    source_binding_context: SourceBindingRuntimeContext


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceSourceProjection:
    """Source-binding projection derived from OpenHCS virtual-workspace metadata."""

    source_paths_by_virtual_path: Mapping[str, str]
    source_metadata_by_path: Mapping[str, Mapping[str, str]]
    workspace_root: str | None = None

    def virtual_path_candidates(
        self,
        *,
        virtual_path: str,
        full_virtual_path: str,
    ) -> tuple[str, str]:
        """Return virtual path lookup keys in most-specific order."""
        return (str(virtual_path), str(full_virtual_path))

    def first_virtual_path_value(
        self,
        mapping: Mapping[str, Any],
        *,
        virtual_path: str,
        full_virtual_path: str,
    ) -> Any | None:
        """Return the first mapped value for a virtual/full path pair."""
        for key in self.virtual_path_candidates(
            virtual_path=virtual_path,
            full_virtual_path=full_virtual_path,
        ):
            value = mapping.get(key)
            if value is not None:
                return value
        return None

    def source_path_for(
        self,
        *,
        virtual_path: str,
        full_virtual_path: str,
        fallback_path: str,
    ) -> str:
        """Return the physical source path represented by a virtual workspace path."""
        source_path = self.first_virtual_path_value(
            self.source_paths_by_virtual_path,
            virtual_path=virtual_path,
            full_virtual_path=full_virtual_path,
        )
        return fallback_path if source_path is None else str(source_path)

    def source_metadata_for(
        self,
        *,
        virtual_path: str,
        full_virtual_path: str,
    ) -> Mapping[str, str] | None:
        """Return source metadata represented by a virtual workspace path."""
        metadata = self.first_virtual_path_value(
            self.source_metadata_by_path,
            virtual_path=virtual_path,
            full_virtual_path=full_virtual_path,
        )
        if metadata is not None:
            return metadata
        source_path = self.source_path_for(
            virtual_path=virtual_path,
            full_virtual_path=full_virtual_path,
            fallback_path=full_virtual_path,
        )
        return self.source_metadata_by_path.get(source_path)

    def pipeline_start_files(self, *, axis_id: str | None = None) -> tuple[str, ...]:
        """Return loadable virtual source paths for one runtime source universe."""
        relative_virtual_paths = tuple(
            virtual_path
            for virtual_path in self.source_paths_by_virtual_path
            if not Path(virtual_path).is_absolute()
        )
        if not relative_virtual_paths:
            relative_virtual_paths = tuple(self.source_paths_by_virtual_path)

        selected = tuple(
            virtual_path
            for virtual_path in relative_virtual_paths
            if self._path_belongs_to_axis(virtual_path, axis_id)
        )
        return tuple(
            dict.fromkeys(
                self._loadable_virtual_path(virtual_path)
                for virtual_path in selected
            )
        )

    def _path_belongs_to_axis(
        self,
        virtual_path: str,
        axis_id: str | None,
    ) -> bool:
        if axis_id is None:
            return True
        metadata = self.source_metadata_by_path.get(virtual_path)
        if metadata is None:
            return True
        from openhcs.constants import MULTIPROCESSING_AXIS

        values = source_component_metadata_values(metadata, MULTIPROCESSING_AXIS)
        if not values:
            return True
        return any(source_metadata_values_equal(value, axis_id) for value in values)

    def _loadable_virtual_path(self, virtual_path: str) -> str:
        if Path(virtual_path).is_absolute():
            return virtual_path
        if self.workspace_root is not None:
            return str(Path(self.workspace_root) / virtual_path)
        return virtual_path


@dataclass(slots=True)
class SourceBindingExecutionCache:
    """Process-local cache for source-binding metadata shared by step runtimes."""

    virtual_workspace_projections: dict[str, VirtualWorkspaceSourceProjection]
    physical_source_files: dict[tuple[str, tuple[str, ...]], tuple[str, ...]]

    @classmethod
    def empty(cls) -> "SourceBindingExecutionCache":
        return cls(
            virtual_workspace_projections={},
            physical_source_files={},
        )


_SOURCE_BINDING_EXECUTION_CACHES: WeakKeyDictionary[
    ProcessingContext,
    SourceBindingExecutionCache,
] = WeakKeyDictionary()


def _save_artifact_value(
    context: ProcessingContext,
    output_plan: ArtifactOutputPlan,
    value: Any,
    source_payload: Any,
    *,
    group_key: str | None,
) -> None:
    """Validate and save one planned artifact value to the memory VFS."""
    resolved_output_plan = output_plan.for_group(group_key)
    vfs_path = resolved_output_plan.path
    axis_id = _require_axis_id(context)
    output_context_request = FunctionOutputContextRequest(
        source_payload=source_payload,
        output_value=value,
        output_plan=resolved_output_plan,
    )
    contextualized_value = FunctionOutputContextStrategy.for_output(
        output_context_request
    ).contextualize(output_context_request)
    runtime_value = normalize_artifact_value(
        resolved_output_plan,
        contextualized_value,
        axis_id=axis_id,
    )

    location = RuntimeArtifactLocation(
        path=vfs_path,
        backend=Backend.MEMORY.value,
    )
    runtime_value_store = require_runtime_value_store(
        context,
        owner_name=PROCESSING_CONTEXT_OWNER_NAME,
    )
    runtime_value_store.replace(
        runtime_value,
        path=location.path,
        backend=location.backend,
    )
    replace_runtime_artifact_payload(
        context.filemanager,
        runtime_value.data,
        location,
    )


def _contextualize_main_output(source_payload: Any, output_value: Any) -> Any:
    """Preserve runtime image context for the main image-flow output."""
    request = FunctionOutputContextRequest(
        source_payload=source_payload,
        output_value=output_value,
    )
    return FunctionOutputContextStrategy.for_output(request).contextualize(request)


def _require_axis_id(context: ProcessingContext) -> str:
    axis_id = context.axis_id
    if not axis_id:
        raise RuntimeError(
            f"{PROCESSING_CONTEXT_OWNER_NAME}.axis_id is required for artifact values."
        )
    return str(axis_id)


def _load_artifact_input_value(
    context: ProcessingContext,
    input_plan: ArtifactInputPlan,
) -> Any:
    """Load an artifact input from VFS through its typed runtime store record."""
    store = require_runtime_value_store(
        context,
        owner_name=PROCESSING_CONTEXT_OWNER_NAME,
    )
    axis_id = _require_axis_id(context)
    query = _artifact_input_query(
        input_plan=input_plan,
        axis_id=axis_id,
    )
    try:
        record = store.resolve(
            query,
            purpose="planned artifact input",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc} Refusing direct VFS fallback because this indicates a lost "
            "typed runtime contract or an artifact not produced through the runtime."
        ) from exc
    return context.filemanager.load(record.path, record.backend)


def _artifact_input_query(
    *,
    input_plan: ArtifactInputPlan,
    axis_id: str,
) -> RuntimeArtifactQuery:
    if input_plan.path != "self":
        return RuntimeArtifactQuery(
            name=input_plan.name,
            kind=input_plan.kind,
            axis_id=axis_id,
            target=RuntimeArtifactLocationTarget(
                RuntimeArtifactLocation(
                    path=input_plan.path,
                    backend=Backend.MEMORY.value,
                )
            ),
        )

    return RuntimeArtifactQuery(
        name=input_plan.name,
        kind=input_plan.kind,
        axis_id=axis_id,
        target=RuntimeArtifactGroupTarget(_single_input_group_key(input_plan)),
    )


def _single_input_group_key(input_plan: ArtifactInputPlan) -> str | None:
    group_keys = input_plan.group_keys or (None,)
    if len(group_keys) == 1:
        return group_keys[0]
    return None


def _select_artifact_plan_for_component(
    plan_by_group: Optional[Mapping[Any, ArtifactOutputPlans | ArtifactInputPlans]],
    component_key: Optional[str],
    default_plan: ArtifactOutputPlans | ArtifactInputPlans,
) -> ArtifactOutputPlans | ArtifactInputPlans:
    """Select precompiled artifact I/O plan for a component."""
    if not plan_by_group:
        return default_plan

    global_plan = plan_by_group.get(None, {})
    if component_key in plan_by_group:
        return {
            **global_plan,
            **plan_by_group[component_key],
        }
    if global_plan:
        return global_plan
    return default_plan


def _select_component_artifact_plans(
    plan: FunctionStepExecutionPlan,
    component_key: Optional[str],
    compiled_group: CompiledFunctionGroup,
) -> ComponentArtifactPlans:
    """Select artifact plans and invocation identity for one component."""
    return ComponentArtifactPlans(
        inputs=_select_artifact_plan_for_component(
            plan.artifact_inputs_by_group,
            component_key,
            plan.artifact_inputs,
        ),
        outputs=_select_artifact_plan_for_component(
            plan.artifact_outputs_by_group,
            component_key,
            plan.artifact_outputs,
        ),
    )


def _resolve_invocation_callable(invocation: CompiledFunctionInvocation) -> Callable:
    """Resolve one compiled invocation to the callable used in this worker."""
    return FunctionInvocationCallableResolver.resolve(invocation)


def prepare_compiled_function_group(group: CompiledFunctionGroup) -> None:
    """Run optional preparation hooks for each callable in a compiled group."""
    for invocation in group.invocations:
        FunctionInvocationCallableResolver.prepare(invocation)


def prepare_compiled_context_callables(
    compiled_contexts: Mapping[str, ProcessingContext],
) -> None:
    """Prepare every compiled callable visible in a set of execution contexts."""
    prepared_group_keys: set[tuple[str, int, str]] = set()
    prepared_invocation_count = 0
    for context_key, context in compiled_contexts.items():
        step_plans = context.step_plans
        if not step_plans:
            continue
        for step_plan in step_plans.values():
            compiled_pattern = step_plan.compiled_function_pattern
            if compiled_pattern is None:
                continue
            for group in compiled_pattern.groups:
                prepare_key = (
                    str(context_key),
                    int(step_plan.step_index),
                    group.group_key,
                )
                if prepare_key in prepared_group_keys:
                    continue
                prepare_compiled_function_group(group)
                prepared_invocation_count += len(group.invocations)
                prepared_group_keys.add(prepare_key)
    logger.info(
        "Prepared %d compiled callable invocations across %d groups.",
        prepared_invocation_count,
        len(prepared_group_keys),
    )


def _execute_function_core(request: FunctionExecutionRequest) -> Any:
    """Execute one callable and route declared artifact I/O."""
    func_callable = request.func_callable
    function_name = func_callable.__name__
    profile = RuntimeProfileRecorder(function_name=function_name)
    context = request.context
    artifact_outputs = request.artifact_outputs
    final_kwargs = dict(request.base_kwargs)

    adapter_manages_artifact_inputs = (
        request.runtime_adapter is not None
        and request.runtime_adapter.manages_artifact_inputs
    )

    if request.artifact_inputs and not adapter_manages_artifact_inputs:
        logger.info(
            f"Artifact inputs for {func_callable.__name__}: {request.artifact_inputs}"
        )
        for arg_name, input_plan in request.artifact_inputs.items():
            logger.info(
                f"Loading artifact input '{arg_name}' from path '{input_plan.path}' (memory backend)"
            )
            load_started_at = time.perf_counter()
            try:
                final_kwargs[arg_name] = _load_artifact_input_value(
                    context,
                    input_plan,
                )
            except Exception as e:
                logger.error(
                    f"Failed to load artifact input '{arg_name}' from '{input_plan.path}': {e}",
                    exc_info=True,
                )
                raise
            profile.record_artifact_elapsed(
                "artifact_input_load",
                started_at=load_started_at,
                artifact_name=arg_name,
                artifact_kind=input_plan.kind.value,
            )

    parameter_names = _callable_parameter_names(func_callable)
    if "context" in parameter_names:
        final_kwargs["context"] = context
    if (
        request.invocation_options is not None
        and "runtime_invocation_options" in parameter_names
    ):
        final_kwargs["runtime_invocation_options"] = request.invocation_options

    if request.runtime_adapter is not None:
        adapter_parameter = request.runtime_adapter.parameter_name
        if adapter_parameter not in parameter_names:
            raise TypeError(
                f"{func_callable.__name__} declares runtime adapter parameter "
                f"'{adapter_parameter}', but its signature does not accept it."
            )
        adapter_started_at = time.perf_counter()
        final_kwargs[adapter_parameter] = request.runtime_adapter.factory(
            RuntimeAdapterRequest(
                context=context,
                artifact_inputs=request.artifact_inputs,
                artifact_outputs=artifact_outputs,
                source_binding_plan=request.source_binding_plan,
                source_binding_context=request.source_binding_context,
                group_key=request.group_key,
                axis_component=request.axis_component,
                axis_component_value=request.axis_component_value,
                plane_projection=request.plane_projection,
            )
        )
        profile.record_adapter_elapsed(
            started_at=adapter_started_at,
            adapter_name=adapter_parameter,
        )

    logger.info(f"Executing function: {func_callable.__name__}")
    call_started_at = time.perf_counter()
    raw_function_output = func_callable(request.main_data_arg, **final_kwargs)
    profile.record_elapsed(
        "function_call",
        started_at=call_started_at,
    )

    if isinstance(raw_function_output, StepResult):
        main_output_data = raw_function_output.image
        if artifact_outputs:
            for output_key, output_plan in artifact_outputs.items():
                logger.info(
                    f"Saving artifact output '{output_key}' to VFS path '{output_plan.path}' (memory backend)"
                )
                if output_key not in raw_function_output.artifacts:
                    raise ValueError(
                        f"Function returned StepResult without planned artifact '{output_key}'."
                    )
                save_started_at = time.perf_counter()
                _save_artifact_value(
                    context,
                    output_plan,
                    raw_function_output.artifacts[output_key],
                    request.main_data_arg,
                    group_key=request.group_key,
                )
                profile.record_artifact_elapsed(
                    "artifact_output_save",
                    started_at=save_started_at,
                    artifact_name=output_key,
                    artifact_kind=output_plan.kind.value,
                )
    elif isinstance(raw_function_output, tuple):
        main_output_data = raw_function_output[0]
        returned_artifact_values_tuple = raw_function_output[1:]

        if artifact_outputs:
            for i, (output_key, output_plan) in enumerate(artifact_outputs.items()):
                logger.info(
                    f"Saving artifact output '{output_key}' to VFS path '{output_plan.path}' (memory backend)"
                )
                if i < len(returned_artifact_values_tuple):
                    save_started_at = time.perf_counter()
                    _save_artifact_value(
                        context,
                        output_plan,
                        returned_artifact_values_tuple[i],
                        request.main_data_arg,
                        group_key=request.group_key,
                    )
                    profile.record_artifact_elapsed(
                        "artifact_output_save",
                        started_at=save_started_at,
                        artifact_name=output_key,
                        artifact_kind=output_plan.kind.value,
                    )
                else:
                    logger.error(
                        f"Artifact output plan wants to save '{output_key}', but function only returned {len(returned_artifact_values_tuple)} artifact values."
                    )
                    raise ValueError(
                        f"Function did not return enough values for all planned artifact outputs. Missing value for '{output_key}'."
                    )
    else:
        main_output_data = raw_function_output

    return _contextualize_main_output(request.main_data_arg, main_output_data)


def execute_function_chain(request: FunctionChainExecutionRequest) -> Any:
    """Execute compiled invocations over one image stack."""
    plan = request.execution_plan
    current_stack = request.initial_data_stack
    current_memory_type = plan.input_memory_type
    debug_sink = debug_event_sink_from_context(request.context)
    capture_debug_events = debug_sink.captures_invocation_events()

    for invocation in request.invocations:
        actual_callable = _resolve_invocation_callable(invocation)
        callable_name = invocation.key.function_name
        artifact_group_key = invocation.key.group_key
        runtime_artifact_group_key = RuntimeArtifactInvocationGroupKey(
            invocation_group_key=artifact_group_key,
            component_value=request.component_value,
        ).artifact_group_key()
        axis_component_value = (
            None if request.component_value is None else str(request.component_value)
        )
        debug_cursor = None
        if capture_debug_events:
            debug_cursor = DebugCursor.from_invocation(
                step_index=plan.step_index,
                step_scope_id=plan.step_scope_id,
                invocation=invocation,
                pattern_group_identity=str(request.runtime_plane_index),
            )
            if debug_sink.should_skip_invocation(debug_cursor):
                continue
        invocation_input_type = invocation.input_memory_type
        invocation_output_type = invocation.output_memory_type
        if invocation_input_type is None or invocation_output_type is None:
            raise ValueError(
                f"Compiled invocation {invocation.key} is missing memory types."
            )
        invocation_artifact_inputs = invocation.select_inputs(request.artifact_inputs)
        invocation_artifact_outputs = invocation.select_outputs(request.artifact_outputs)
        invocation_parameters = None
        input_debug_refs = None
        output_debug_refs = None
        if capture_debug_events:
            if debug_cursor is None:
                raise RuntimeError("Debug cursor missing while debug events are active.")
            invocation_parameters = DebugInvocationParameter.from_kwargs(
                invocation.kwargs_dict
            )
            input_debug_refs = DebugArtifactRefProjection.from_artifact_plans(
                artifact_plans=invocation_artifact_inputs,
                cursor=debug_cursor,
            )
            output_debug_refs = DebugArtifactRefProjection.from_artifact_plans(
                artifact_plans=invocation_artifact_outputs,
                cursor=debug_cursor,
            )

        current_stack = _convert_main_flow_memory(
            current_stack,
            source_type=current_memory_type,
            target_type=invocation_input_type,
            gpu_id=plan.device_id,
        )

        invocation_started_at = time.perf_counter()
        if capture_debug_events:
            if debug_cursor is None or input_debug_refs is None:
                raise RuntimeError("Debug state missing while debug events are active.")
            debug_sink.record(
                DebugEvent(
                    event_type=DebugEventType.BEFORE_INVOCATION,
                    cursor=debug_cursor,
                    step_name=plan.step_name,
                    callable_name=callable_name,
                    axis_id=plan.axis_id,
                    input_artifact_refs=input_debug_refs.refs,
                    measurement_refs=input_debug_refs.measurement_refs,
                    relationship_refs=input_debug_refs.relationship_refs,
                    invocation_parameters=invocation_parameters,
                )
            )
        try:
            current_stack = _execute_function_core(
                FunctionExecutionRequest(
                    func_callable=actual_callable,
                    main_data_arg=current_stack,
                    base_kwargs=invocation.kwargs_dict,
                    context=request.context,
                    artifact_inputs=invocation_artifact_inputs,
                    artifact_outputs=invocation_artifact_outputs,
                    runtime_adapter=invocation.contract.runtime_adapter,
                    invocation_options=invocation.invocation_options,
                    source_binding_plan=plan.source_binding_plan,
                    source_binding_context=request.source_binding_context,
                    group_key=runtime_artifact_group_key,
                    axis_component=plan.group_by_value,
                    axis_component_value=axis_component_value,
                    plane_projection=RuntimePlaneProjection.for_group_key(
                        runtime_artifact_group_key,
                        plane_index=(
                            request.runtime_plane_index
                            if runtime_artifact_group_key is not None
                            else None
                        ),
                    ),
                )
            )
        except Exception as exc:
            if capture_debug_events:
                if (
                    debug_cursor is None
                    or input_debug_refs is None
                    or output_debug_refs is None
                ):
                    raise RuntimeError(
                        "Debug state missing while debug events are active."
                    ) from exc
                debug_sink.record(
                    DebugEvent.for_exception(
                        cursor=debug_cursor,
                        step_name=plan.step_name,
                        callable_name=callable_name,
                        axis_id=plan.axis_id,
                        exception=exc,
                        input_artifact_refs=input_debug_refs.refs,
                        output_artifact_refs=output_debug_refs.refs,
                        measurement_refs=output_debug_refs.measurement_refs,
                        relationship_refs=output_debug_refs.relationship_refs,
                        invocation_parameters=invocation_parameters,
                    )
                )
            raise
        invocation_seconds = time.perf_counter() - invocation_started_at
        after_event = None
        if capture_debug_events:
            if (
                debug_cursor is None
                or input_debug_refs is None
                or output_debug_refs is None
            ):
                raise RuntimeError("Debug state missing while debug events are active.")
            after_event = DebugEvent(
                event_type=DebugEventType.AFTER_INVOCATION,
                cursor=debug_cursor,
                step_name=plan.step_name,
                callable_name=callable_name,
                axis_id=plan.axis_id,
                timing_seconds=invocation_seconds,
                input_artifact_refs=input_debug_refs.refs,
                output_artifact_refs=output_debug_refs.refs,
                measurement_refs=output_debug_refs.measurement_refs,
                relationship_refs=output_debug_refs.relationship_refs,
                invocation_parameters=invocation_parameters,
            )
            debug_sink.record(after_event)
        _log_runtime_profile(
            "invocation_total",
            invocation_seconds,
            function=callable_name,
            group=invocation.key.group_key,
            position=invocation.key.position,
        )
        current_memory_type = invocation_output_type
        if after_event is not None and debug_sink.should_stop_after_invocation(after_event):
            break

    return current_stack


def _convert_main_flow_memory(
    payload: Any,
    *,
    source_type: str,
    target_type: str,
    gpu_id: int,
) -> Any:
    """Convert main-flow image pixels while preserving image context."""
    data = image_payload_data(payload)
    converted = convert_memory(
        data=data,
        source_type=source_type,
        target_type=target_type,
        gpu_id=gpu_id,
    )
    return with_image_payload_data(payload, converted)


def _stack_payload_context(raw_slices: Sequence[Any], stack: Any) -> Any:
    """Attach per-slice image context to a freshly loaded stack."""
    metadata = _stack_payload_metadata(raw_slices)
    mask = _stack_payload_mask(raw_slices)
    return image_payload_with_context(stack, mask=mask, metadata=metadata)


def _stack_payload_metadata(raw_slices: Sequence[Any]) -> ImagePayloadMetadata:
    slice_metadata = tuple(image_payload_metadata(slice_data) for slice_data in raw_slices)
    if not any(metadata.has_values for metadata in slice_metadata):
        return ImagePayloadMetadata()
    return ImagePayloadMetadata(
        channel_intensity_scales=tuple(
            metadata.intensity_scale_for_channel(0)
            for metadata in slice_metadata
        ),
        channel_source_dtypes=tuple(
            metadata.source_dtype
            for metadata in slice_metadata
        ),
        channel_source_paths=tuple(
            metadata.source_path
            for metadata in slice_metadata
        ),
        channel_source_component_metadata=tuple(
            metadata.source_component_metadata
            for metadata in slice_metadata
        ),
        channel_unit_interval_intensity_scales=tuple(
            metadata.unit_interval_intensity_scale_for_channel(0)
            for metadata in slice_metadata
        ),
    )


def _stack_payload_mask(raw_slices: Sequence[Any]) -> Any | None:
    masks = tuple(image_payload_mask(slice_data) for slice_data in raw_slices)
    if not any(mask is not None for mask in masks):
        return None
    data_slices = tuple(image_payload_data(slice_data) for slice_data in raw_slices)
    resolved_masks = tuple(
        np.ones(np.asarray(data_slice).shape[:2], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
        for data_slice, mask in zip(data_slices, masks)
    )
    return np.stack(resolved_masks)


def _unstack_payload_context(payload: Any, slices: Sequence[Any]) -> list[Any]:
    """Attach per-slice image context after unstacking a runtime stack."""
    mask = image_payload_mask(payload)
    metadata = image_payload_metadata(payload)
    if mask is None and not metadata.has_values:
        return list(slices)
    return [
        image_payload_with_context(
            data=slice_data,
            mask=(
                None
                if mask is None
                else _payload_mask_slice(mask, index, slice_count=len(slices))
            ),
            metadata=metadata.for_channel(index),
        )
        for index, slice_data in enumerate(slices)
    ]


def _payload_mask_slice(mask: Any, index: int, *, slice_count: int) -> Any:
    mask_array = np.asarray(mask)
    if mask_array.ndim >= 3 and mask_array.shape[0] == slice_count:
        return mask_array[index]
    return mask


class PatternGroupRuntime:
    """Staged runtime for one pattern group."""

    def __init__(self, request: PatternGroupExecutionRequest) -> None:
        self.request = request
        self.pattern_repr = str(request.pattern_group_info)[:100]

    @property
    def context(self) -> ProcessingContext:
        return self.request.context

    @property
    def plan(self) -> FunctionStepExecutionPlan:
        return self.request.execution_plan

    def source_binding_execution_cache(self) -> SourceBindingExecutionCache:
        """Return the per-context source-binding execution cache."""
        cache = _SOURCE_BINDING_EXECUTION_CACHES.get(self.context)
        if cache is None:
            cache = SourceBindingExecutionCache.empty()
            _SOURCE_BINDING_EXECUTION_CACHES[self.context] = cache
        return cache

    def run(self) -> None:
        start_time = time.time()
        logger.debug(
            f"Processing pattern {self.pattern_repr} for axis {self.plan.axis_id}"
        )

        try:
            load_started_at = time.perf_counter()
            loaded = self._load_input_stack()
            _log_runtime_profile(
                "pattern_load_stack",
                time.perf_counter() - load_started_at,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                pattern=self.pattern_repr,
            )
            execute_started_at = time.perf_counter()
            processed_stack = self._execute_pattern(loaded)
            _log_runtime_profile(
                "pattern_execute_chain",
                time.perf_counter() - execute_started_at,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                pattern=self.pattern_repr,
            )
            unstack_started_at = time.perf_counter()
            output_slices = self._validate_and_unstack(processed_stack, loaded)
            _log_runtime_profile(
                "pattern_validate_unstack",
                time.perf_counter() - unstack_started_at,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                pattern=self.pattern_repr,
            )
            save_started_at = time.perf_counter()
            self._save_outputs(output_slices, loaded.matching_files)
            _log_runtime_profile(
                "pattern_save_outputs",
                time.perf_counter() - save_started_at,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                pattern=self.pattern_repr,
            )
            cleanup_started_at = time.perf_counter()
            self._cleanup_collapsed_domains(output_slices, loaded.matching_files)
            _log_runtime_profile(
                "pattern_cleanup",
                time.perf_counter() - cleanup_started_at,
                step=self.plan.step_index,
                step_name=self.plan.step_name,
                pattern=self.pattern_repr,
            )
            logger.debug(
                f"Finished pattern group {self.pattern_repr} in {(time.time() - start_time):.2f}s."
            )
        except Exception as e:
            import traceback

            full_traceback = traceback.format_exc()
            logger.error(
                f"Error processing pattern group {self.pattern_repr}: {e}",
                exc_info=True,
            )
            logger.error(
                f"Full traceback for pattern group {self.pattern_repr}:\n{full_traceback}"
            )
            raise ValueError(
                f"Failed to process pattern group {self.pattern_repr}: {e}"
            ) from e

    def _load_input_stack(self) -> PatternGroupData:
        context = self.context
        request = self.request
        if not context.microscope_handler:
            raise RuntimeError("MicroscopeHandler not available in context.")

        matching_files = context.microscope_handler.path_list_from_pattern(
            str(self.plan.input_dir),
            request.pattern_group_info,
            context.filemanager,
            Backend.MEMORY.value,
            [vc.value for vc in self.plan.variable_components]
            if self.plan.variable_components
            else None,
        )

        if not matching_files:
            raise ValueError(
                f"No matching files found for pattern group {self.pattern_repr} in {self.plan.input_dir}. "
                f"This indicates either: (1) no image files exist in the directory, "
                f"(2) files don't match the pattern, or (3) pattern parsing failed. "
                f"Check that input files exist and match the expected naming convention."
            )

        matching_files = self._filter_matching_files_for_group(matching_files)

        logger.debug(
            f"Pattern {self.pattern_repr} matched {len(matching_files)} files: {[Path(f).name for f in matching_files]}"
        )

        matching_files.sort()
        logger.debug(
            f"Pattern {self.pattern_repr} sorted files: {[Path(f).name for f in matching_files]}"
        )

        full_file_paths = [str(self.plan.input_dir / f) for f in matching_files]
        raw_slices = context.filemanager.load_batch(
            full_file_paths,
            Backend.MEMORY.value,
        )
        raw_slices = self._apply_source_image_loading_semantics(
            raw_slices,
            matching_files,
            full_file_paths,
        )

        if not raw_slices:
            raise ValueError(
                f"No valid images loaded for pattern group {self.pattern_repr} in {self.plan.input_dir}. "
                f"Found {len(matching_files)} matching files but failed to load any valid images. "
                f"This indicates corrupted image files, unsupported formats, or I/O errors. "
                f"Check file integrity and format compatibility."
            )

        raw_slice_data = tuple(image_payload_data(slice_data) for slice_data in raw_slices)
        main_data_stack = ImageStackLayout.for_slices(raw_slice_data).stack(
            slices=raw_slice_data,
            memory_type=self.plan.input_memory_type,
            gpu_id=self.plan.device_id,
        )
        main_data_stack = _stack_payload_context(raw_slices, main_data_stack)

        return PatternGroupData(
            matching_files=matching_files,
            main_data_stack=main_data_stack,
            source_slice_shapes=tuple(
                tuple(slice_data.shape)
                for slice_data in raw_slice_data
            ),
            source_binding_context=self._source_binding_context(matching_files),
        )

    def _filter_matching_files_for_group(
        self,
        matching_files: list[str],
    ) -> list[str]:
        """Constrain grouped executions to files from the current component."""
        group_component = self.plan.group_by_value
        component_value = self.request.component_value
        if group_component is None or component_value is None:
            return matching_files

        parser = self.context.microscope_handler.parser
        filtered = [
            filename
            for filename in matching_files
            if (
                metadata := parser.parse_filename(Path(filename).name)
            )
            and str(metadata.get(group_component)) == str(component_value)
        ]
        if not filtered:
            raise ValueError(
                f"Pattern group {self.pattern_repr} for {group_component}="
                f"{component_value!r} matched files, but none carried the "
                f"expected grouped component. Matched files: {matching_files}"
            )
        return filtered

    def _apply_source_image_loading_semantics(
        self,
        raw_slices: Sequence[Any],
        matching_files: Sequence[str],
        full_file_paths: Sequence[str],
    ) -> list[Any]:
        source_projection = self._source_schema_workspace_projection()
        if source_projection is None:
            return list(raw_slices)
        return [
            apply_source_image_loading_semantics(
                payload,
                source_metadata=source_projection.source_metadata_for(
                    virtual_path=virtual_path,
                    full_virtual_path=full_virtual_path,
                ),
                source_path=source_projection.source_path_for(
                    virtual_path=virtual_path,
                    full_virtual_path=full_virtual_path,
                    fallback_path=full_virtual_path,
                ),
                read_backend=Backend.DISK.value,
                filemanager=self.context.filemanager,
            )
            for payload, virtual_path, full_virtual_path in zip(
                raw_slices,
                matching_files,
                full_file_paths,
            )
        ]

    def _source_binding_context(
        self,
        matching_files: list[str],
    ) -> SourceBindingRuntimeContext:
        source_backend = self.context.microscope_handler.get_primary_backend(
            self.context.input_dir,
            self.context.filemanager,
        )
        source_projection = self._source_schema_workspace_projection()
        step_input_source_paths = (
            source_projection.source_paths_by_virtual_path
            if source_projection is not None
            else {}
        )
        source_metadata_by_path = (
            source_projection.source_metadata_by_path
            if source_projection is not None
            else {}
        )
        pipeline_input_files, pipeline_input_backend = (
            self._pipeline_start_source_universe(
                source_backend,
                source_projection=source_projection,
            )
        )
        step_input_files = self._step_input_source_universe(
            matching_files,
            source_backend,
            source_projection=source_projection,
        )
        return SourceBindingRuntimeContext(
            step_input_files=step_input_files,
            current_step_input_files=tuple(matching_files),
            step_input_dir=str(self.plan.input_dir),
            step_input_backend=self.plan.read_backend,
            step_input_source_paths=step_input_source_paths,
            source_metadata_by_path=source_metadata_by_path,
            pipeline_input_files=pipeline_input_files,
            pipeline_input_backend=pipeline_input_backend,
        )

    def _step_input_source_universe(
        self,
        matching_files: list[str],
        source_backend: str,
        *,
        source_projection: VirtualWorkspaceSourceProjection | None,
    ) -> tuple[str, ...]:
        """Return the source universe needed for step-input selector bindings."""

        if not self.plan.source_binding_plan.requires_step_input_selector_resolution:
            return tuple(matching_files)
        if (
            source_backend == Backend.VIRTUAL_WORKSPACE.value
            and source_projection is not None
        ):
            return source_projection.pipeline_start_files(axis_id=self.plan.axis_id)
        return tuple(self.plan.get_paths_for_axis(self.context.input_dir, source_backend))

    def _pipeline_start_source_universe(
        self,
        source_backend: str,
        *,
        source_projection: VirtualWorkspaceSourceProjection | None,
    ) -> tuple[tuple[str, ...], str]:
        if not self._requires_full_pipeline_source_universe():
            return (
                tuple(self.plan.get_paths_for_axis(self.context.input_dir, source_backend)),
                source_backend,
            )

        if source_projection is not None:
            disk_files = tuple(
                str(path)
                for path in self.context.filemanager.list_files(
                    str(self.context.input_dir),
                    Backend.DISK.value,
                    recursive=True,
                )
            )
            return (
                tuple(
                    dict.fromkeys(
                        (
                            *(
                                str(path)
                                for path in source_projection.source_paths_by_virtual_path.values()
                            ),
                            *disk_files,
                        )
                    )
                ),
                Backend.DISK.value,
            )

        universe_backend = (
            Backend.DISK.value
            if source_backend
            in (Backend.VIRTUAL_WORKSPACE.value, Backend.MEMORY.value)
            else source_backend
        )
        return (
            tuple(
                str(path)
                for path in self.context.filemanager.list_files(
                    str(self.context.input_dir),
                    universe_backend,
                    recursive=True,
                )
            ),
            universe_backend,
        )

    def _requires_full_pipeline_source_universe(self) -> bool:
        if any(
            invocation.contract.runtime_adapter is not None
            for invocation in self.plan.compiled_function_pattern.iter_invocations()
        ):
            return True
        plan = self.plan.source_binding_plan
        if plan.metadata_rules:
            return True
        return any(
            binding.origin is SourceBindingOrigin.PIPELINE_START
            for bindings in plan.bindings_by_group.values()
            for binding in bindings
        )

    def _virtual_workspace_source_paths_by_virtual_path(self) -> Mapping[str, str]:
        return self._virtual_workspace_source_projection().source_paths_by_virtual_path

    def _virtual_workspace_source_metadata_by_path(
        self,
    ) -> Mapping[str, Mapping[str, str]]:
        return self._virtual_workspace_source_projection().source_metadata_by_path

    def _source_schema_workspace_projection(
        self,
    ) -> VirtualWorkspaceSourceProjection | None:
        """Return source-schema metadata projection when OpenHCS metadata declares one."""

        metadata = self._openhcs_metadata_dict()
        if not self._declares_source_schema_workspace_projection(metadata):
            return None
        return self._virtual_workspace_source_projection_from_metadata(metadata)

    @staticmethod
    def _declares_source_schema_workspace_projection(
        metadata: Mapping[str, Any],
    ) -> bool:
        from openhcs.microscopes.openhcs import FIELDS

        return any(
            bool(subdirectory.get("workspace_mapping"))
            for subdirectory in metadata.get(FIELDS.SUBDIRECTORIES, {}).values()
            if isinstance(subdirectory, Mapping)
        )

    def _virtual_workspace_source_projection(self) -> VirtualWorkspaceSourceProjection:
        """Return cached virtual-workspace source-binding projection for this plate."""
        return self._virtual_workspace_source_projection_from_metadata(
            self._openhcs_metadata_dict()
        )

    def _virtual_workspace_source_projection_from_metadata(
        self,
        metadata: Mapping[str, Any],
    ) -> VirtualWorkspaceSourceProjection:
        """Return cached source-schema projection for this plate metadata."""
        plate_path = str(Path(self.context.plate_path))
        cache = self.source_binding_execution_cache()
        projection = cache.virtual_workspace_projections.get(plate_path)
        if projection is not None:
            return projection

        from openhcs.microscopes.openhcs import FIELDS

        workspace_source_paths: dict[str, str] = {}
        source_metadata_by_path: dict[str, Mapping[str, str]] = {}
        source_metadata_by_real_path: dict[str, Mapping[str, str] | None] = {}
        for subdirectory in metadata.get(FIELDS.SUBDIRECTORIES, {}).values():
            workspace_mapping = subdirectory.get("workspace_mapping", {})
            for virtual_relative, real_relative in workspace_mapping.items():
                real_path = str(Path(self.context.plate_path) / real_relative)
                virtual_path = str(virtual_relative)
                workspace_source_paths[virtual_path] = real_path
                workspace_source_paths[
                    str(Path(self.context.plate_path) / virtual_path)
                ] = real_path

            source_metadata = subdirectory.get(FIELDS.SOURCE_METADATA, {})
            if not isinstance(source_metadata, Mapping):
                raise RuntimeError(
                    "virtual_workspace source metadata must be a path-keyed mapping."
                )
            for virtual_relative, metadata_fields in source_metadata.items():
                if not isinstance(metadata_fields, Mapping):
                    raise RuntimeError(
                        "virtual_workspace source metadata values must be mappings."
                    )
                normalized_metadata = MappingProxyType(
                    {
                        str(key): str(value)
                        for key, value in metadata_fields.items()
                    }
                )
                virtual_path = str(virtual_relative)
                full_virtual_path = str(Path(self.context.plate_path) / virtual_path)
                normalized_metadata = source_schema_metadata_with_virtual_components(
                    virtual_path,
                    normalized_metadata,
                )
                source_metadata_by_path[virtual_path] = normalized_metadata
                source_metadata_by_path[full_virtual_path] = normalized_metadata
                real_relative = workspace_mapping.get(virtual_path)
                if real_relative is not None:
                    real_path = str(Path(self.context.plate_path) / real_relative)
                    existing_metadata = source_metadata_by_real_path.get(real_path)
                    if existing_metadata is None and real_path in source_metadata_by_real_path:
                        continue
                    if existing_metadata is None:
                        source_metadata_by_real_path[real_path] = normalized_metadata
                    elif dict(existing_metadata) != dict(normalized_metadata):
                        source_metadata_by_real_path[real_path] = None

        for real_path, metadata_fields in source_metadata_by_real_path.items():
            if metadata_fields is not None:
                source_metadata_by_path[real_path] = metadata_fields

        if not workspace_source_paths:
            raise RuntimeError(
                "virtual_workspace source binding resolution requires "
                "workspace_mapping entries in OpenHCS metadata."
            )

        projection = VirtualWorkspaceSourceProjection(
            source_paths_by_virtual_path=MappingProxyType(workspace_source_paths),
            source_metadata_by_path=MappingProxyType(source_metadata_by_path),
            workspace_root=plate_path,
        )
        cache.virtual_workspace_projections[plate_path] = projection
        return projection

    def _openhcs_metadata_dict(self) -> Mapping[str, Any]:
        from openhcs.microscopes.openhcs import OpenHCSMetadataHandler

        metadata_handler = self.context.microscope_handler.metadata_handler
        if not isinstance(metadata_handler, OpenHCSMetadataHandler):
            metadata_handler = OpenHCSMetadataHandler(self.context.filemanager)
        return metadata_handler._load_metadata_dict(self.context.plate_path)

    def _component_artifact_plans(self) -> ComponentArtifactPlans:
        request = self.request
        component_key = (
            None if request.component_value is None else str(request.component_value)
        )
        component_artifacts = _select_component_artifact_plans(
            self.plan,
            component_key,
            request.compiled_group,
        )

        logger.debug(
            "Selected artifact outputs for component %s: %s",
            component_key,
            component_artifacts.outputs,
        )

        return component_artifacts

    def _execute_pattern(
        self,
        loaded: PatternGroupData,
    ) -> Any:
        request = self.request
        component_artifacts = self._component_artifact_plans()

        if not request.compiled_group.invocations:
            raise ValueError(
                f"Compiled function group {request.compiled_group.group_key} has no invocations."
            )

        return execute_function_chain(
            FunctionChainExecutionRequest(
                initial_data_stack=loaded.main_data_stack,
                invocations=request.compiled_group.invocations,
                context=self.context,
                execution_plan=self.plan,
                artifact_inputs=component_artifacts.inputs,
                artifact_outputs=component_artifacts.outputs,
                source_binding_context=loaded.source_binding_context,
                runtime_plane_index=request.component_index,
                component_value=request.component_value,
            )
        )

    def _validate_and_unstack(
        self,
        processed_stack: Any,
        loaded: PatternGroupData,
    ) -> list[Any]:
        processed_data = image_payload_data(processed_stack)
        try:
            output_slices = ImageStackLayout.unstack_result_for_source_slices(
                processed_data,
                source_slice_shapes=loaded.source_slice_shapes,
                memory_type=self.plan.output_memory_type,
                gpu_id=self.plan.device_id,
            )
        except ValueError as exc:
            logger.error("Function output is not an OpenHCS image stack.")
            logger.error(f"Output type: {type(processed_stack)}")
            logger.error(
                f"Output shape: {getattr(processed_data, 'shape', 'no shape attr')}"
            )
            logger.error(
                f"Output exposes ndim: {hasattr(processed_data, 'ndim')}"
            )
            if hasattr(processed_data, "ndim"):
                logger.error(f"Output ndim: {processed_data.ndim}")
            raise ValueError(
                "Main processing must result in an image stack shaped "
                f"(N, H, W) or (N, H, W, C), got "
                f"{getattr(processed_data, 'shape', 'unknown')}"
            ) from exc

        return _unstack_payload_context(processed_stack, output_slices)

    def _save_outputs(self, output_slices: list[Any], matching_files: list[str]) -> None:
        context = self.context
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs < num_inputs:
            logger.debug(
                f"Function returned {num_outputs} images from {num_inputs} inputs - likely flattening operation"
            )
        elif num_outputs > num_inputs:
            logger.warning(
                f"Function returned more images ({num_outputs}) than inputs ({num_inputs}) - unexpected"
            )

        output_data = []
        output_paths_batch = []

        for i, img_slice in enumerate(output_slices):
            if i >= len(matching_files):
                raise ValueError(
                    f"Function returned {num_outputs} output slices but only {num_inputs} input files available. "
                    f"Cannot generate filename for output slice {i}. This indicates a bug in the function or "
                    f"unstacking logic - functions should return same or fewer images than inputs."
                )

            input_filename = matching_files[i]
            output_filename = Path(input_filename).name
            output_path = self.plan.output_dir / output_filename

            if context.filemanager.exists(str(output_path), Backend.MEMORY.value):
                context.filemanager.delete(str(output_path), Backend.MEMORY.value)

            output_data.append(img_slice)
            output_paths_batch.append(str(output_path))

        context.filemanager.ensure_directory(
            str(self.plan.output_dir),
            Backend.MEMORY.value,
        )
        context.filemanager.save_batch(
            output_data,
            output_paths_batch,
            Backend.MEMORY.value,
        )

    def _cleanup_collapsed_domains(
        self,
        output_slices: list[Any],
        matching_files: list[str],
    ) -> None:
        context = self.context
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs >= num_inputs:
            return

        for j in range(num_outputs, num_inputs):
            unused_filename = matching_files[j]
            for cleanup_dir in (self.plan.input_dir, self.plan.output_dir):
                unused_path = cleanup_dir / unused_filename
                if context.filemanager.exists(
                    str(unused_path),
                    Backend.MEMORY.value,
                ):
                    context.filemanager.delete(
                        str(unused_path),
                        Backend.MEMORY.value,
                    )
                    logger.debug(
                        "Deleted unused collapsed-domain file after reduced "
                        "output cardinality: %s",
                        unused_path,
                    )


def _process_single_pattern_group(request: PatternGroupExecutionRequest) -> None:
    """Process one image pattern group through its assigned callable pattern."""
    PatternGroupRuntime(request).run()
