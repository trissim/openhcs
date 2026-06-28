"""Runtime execution helpers for FunctionStep.

This module owns callable invocation, artifact routing, and pattern-group stack
execution. FunctionStep remains responsible for step-level orchestration.
"""

from abc import ABC, abstractmethod
import inspect
import logging
import os
import time
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from threading import Lock
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Iterator, Mapping, Optional, Sequence
from weakref import WeakKeyDictionary

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.constants.constants import Backend, VariableComponents
from openhcs.core.artifacts import (
    ArtifactKind,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    StepResult,
)
from openhcs.core.callable_contract import (
    CallableContract,
    CallableRuntimeCacheKey,
    prepare_processing_callable,
)
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
    RuntimeComponentValue,
)
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    AlignedImageSliceContext,
    flatten_aligned_image_slice_contexts,
    flatten_aligned_image_payload_slices,
    stack_image_payload_context,
    stack_image_payload_context_from_metadata,
    unstack_image_payload_context,
)
from openhcs.core.image_stack_layout import ImageStackLayout, SourceSliceUnstackRequest
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.memory import (
    convert_memory,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactLocation,
    RuntimeArtifactQuery,
    replace_runtime_artifact_payload,
)
from openhcs.core.runtime_adapters import (
    RuntimeAdapterRequest,
    RuntimeAdapterSpec,
    RuntimeExecutionAxisScope,
)
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.core.runtime_slice_alignment import (
    RuntimeSliceAlignedValueSet,
    RuntimeSliceAlignedValues,
)
from openhcs.core.source_image_semantics import SourceImagePayloadSemantics
from openhcs.core.source_workspace_projection import (
    VirtualWorkspacePathLookup,
    VirtualWorkspaceSourceProjection,
    VirtualWorkspaceSourceProjectionAuthority,
    VirtualWorkspaceSourceProjectionCache,
)
from openhcs.core.source_binding_selection import (
    SourceBindingCandidateMatcher,
    SourceBindingMatchedImageSet,
    SourceBindingRuntimeContextRequest,
    SourceUniverseRequest,
    SourcePatternResolutionContext,
)
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    SourceBindingRuntimeContext,
)
from openhcs.core.runtime_values import (
    DerivedImagePayloadContext,
    ImagePayloadMetadata,
    ObjectLabelValue,
    RuntimeArrayData,
    RuntimeImageSourceIdentityCompleteness,
    SourceImageObjectLabelBuildRequest,
    image_payload_data,
    image_payload_metadata,
    image_payload_slice_context,
    is_array_payload,
    normalize_artifact_value,
    with_image_payload_data,
    with_image_payload_metadata,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    NominalTypeKeyedStrategyMixin,
)
from openhcs.core.runtime_semantics import RuntimePlaneProjection
from openhcs.core.step_dependencies import StepInputDependencyKind
from openhcs.core.steps.function_output_manifest import (
    NoStepOutputManifestMatch,
    ProducedOutputSemantics,
    step_output_manifest,
)
from openhcs.core.steps.function_output_identity import (
    FunctionOutputIdentityAuthority,
    FunctionOutputPathAuthority,
    FunctionOutputPathRequest,
)
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan

logger = logging.getLogger(__name__)

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
_PROFILE_RUNTIME_PATH_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME_PATH"
RUNTIME_CONTEXT_PARAMETER_NAME = "context"
RUNTIME_INVOCATION_OPTIONS_PARAMETER_NAME = "runtime_invocation_options"
PROCESSING_CONTEXT_OWNER_NAME = ProcessingContext.__name__
ArtifactInputPlans = Mapping[str, ArtifactInputPlan]
ArtifactOutputPlans = Mapping[str, ArtifactOutputPlan]
JsonValue = RuntimeComponentValue | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
ObjectLabelContextualizedOutput = (
    ObjectLabelValue | RuntimeSliceAlignedValueSet[ObjectLabelValue]
)
ObjectLabelContextualizableOutput = (
    RuntimeArrayData | ObjectLabelValue | RuntimeSliceAlignedValueSet
)
RuntimePayload = RuntimeArrayData | ObjectLabelValue | RuntimeSliceAlignedValueSet
RuntimeFunctionOutput = RuntimePayload | StepResult | tuple[RuntimePayload, ...]
RuntimeCallableArgument = (
    JsonValue | RuntimePayload | ProcessingContext | RuntimeInvocationOptions
)
RuntimeCallableKwargs = Mapping[str, RuntimeCallableArgument]
RuntimeProfileFieldValue = str | int | float | bool | None
EMPTY_ARTIFACT_PLANS: ArtifactOutputPlans = MappingProxyType({})
_CALLABLE_PARAMETER_NAMES: WeakKeyDictionary[Callable, frozenset[str]] = (
    WeakKeyDictionary()
)


class FunctionInvocationCallableResolver:
    """Process-local resolver for compiled invocation callables.

    The compiler stores picklable ``FunctionReference`` objects in compiled
    invocations. Runtime execution needs actual callables. Resolving them during
    compiler preparation lets fork workers inherit the resolved callable cache,
    while spawn workers still resolve lazily in their own process.
    """

    _lock = Lock()
    _cache: dict[CallableRuntimeCacheKey, Callable] = {}

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

        resolved = invocation.contract.resolve_runtime_callable()

        with cls._lock:
            cls._cache[cache_key] = resolved
        return resolved

    @classmethod
    def cache_key(
        cls,
        invocation: CompiledFunctionInvocation,
    ) -> CallableRuntimeCacheKey:
        """Return process-local callable cache key for one compiled invocation."""
        return invocation.contract.runtime_callable_cache_identity()


class RuntimeProfileSink:
    """Runtime-profile output authority backed by explicit environment settings."""

    @classmethod
    def enabled(cls) -> bool:
        raw_value = cls.environment_value(_PROFILE_RUNTIME_ENV)
        if raw_value is None:
            return False
        return raw_value.lower() in {"1", "true", "yes"}

    @staticmethod
    def environment_value(name: str) -> str | None:
        if name not in os.environ:
            return None
        return os.environ[name]

    @classmethod
    def record(
        cls,
        label: str,
        seconds: float,
        **fields: RuntimeProfileFieldValue,
    ) -> None:
        if not cls.enabled():
            return
        field_text = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)
        profile_path = cls.environment_value(_PROFILE_RUNTIME_PATH_ENV)
        if profile_path is not None:
            with open(profile_path, "a", encoding="utf-8") as handle:
                handle.write(f"RUNTIME_PROFILE {label} {seconds:.6f}s {field_text}\n")


def _callable_parameter_names(func: Callable) -> frozenset[str]:
    """Return cached callable parameter names for runtime adapter injection."""
    names = _CALLABLE_PARAMETER_NAMES.get(func)
    if names is None:
        names = frozenset(inspect.signature(func).parameters)
        _CALLABLE_PARAMETER_NAMES[func] = names
    return names


@dataclass(frozen=True, slots=True)
class SourceImagePayloadSlice:
    """One source-image slice used to contextualize slice-aligned outputs."""

    source_payload: RuntimePayload
    slice_index: int

    def payload(self) -> RuntimeArrayData:
        source_data = image_payload_data(self.source_payload)
        source_array = np.asarray(source_data)
        if source_array.ndim >= 3:
            return image_payload_slice_context(
                self.source_payload,
                source_array[self.slice_index],
                self.slice_index,
            )
        if self.slice_index == 0:
            return self.source_payload
        raise ValueError(
            "Slice-aligned object-label output has more slices than its source "
            "image context."
        )


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
    def for_output_plan(
        cls,
        output_plan: ArtifactOutputPlan | None,
    ) -> "FunctionOutputContextStrategy":
        output_kind = ArtifactKind.IMAGE if output_plan is None else output_plan.kind
        strategy_type = cls.__registry__.get(output_kind.value)
        if strategy_type is None:
            return UnchangedFunctionOutputContextStrategy()
        return strategy_type()

    @abstractmethod
    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
    ) -> ObjectLabelContextualizedOutput:
        """Return output with source context preserved where semantics allow it."""


class UnchangedFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Leave outputs unchanged when no contextual image semantics are declared."""

    kind = ArtifactKind.SPECIAL

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
    ) -> ObjectLabelContextualizedOutput:
        del source_payload, output_plan
        return output_value


class ImageFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Preserve source-image metadata for image outputs derived from the main input."""

    kind = ArtifactKind.IMAGE

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
    ) -> ObjectLabelContextualizedOutput:
        del output_plan
        return ImageOutputSourceContextStrategy.for_source_payload(
            source_payload,
        ).contextualize(source_payload, output_value)


class ImageOutputSourceContextStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered image-output contextualization by semantic source payload type."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def for_source_payload(
        cls,
        source_payload: RuntimePayload,
    ) -> "ImageOutputSourceContextStrategy":
        strategy = cls.for_nominal_value(source_payload)
        if strategy is None:
            return DefaultImageOutputSourceContextStrategy()
        return strategy

    @abstractmethod
    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
    ) -> RuntimePayload:
        """Return image output with source semantics attached."""


class DefaultImageOutputSourceContextStrategy(ImageOutputSourceContextStrategy):
    """Attach scalar source-image context to a derived image output."""

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
    ) -> RuntimePayload:
        if isinstance(output_value, AlignedImageStack):
            return output_value
        return DerivedImagePayloadContext(
            source_payload,
            output_value,
        ).payload()


class AlignedImageStackOutputSourceContextStrategy(ImageOutputSourceContextStrategy):
    """Preserve aligned multi-source image payloads as their own source context."""

    value_type = AlignedImageStack

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
    ) -> RuntimePayload:
        del source_payload
        return output_value


class RuntimeSliceAlignedImageOutputSourceContextStrategy(
    ImageOutputSourceContextStrategy
):
    """Attach per-runtime-slice source context to derived image outputs."""

    value_type = RuntimeSliceAlignedValueSet

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
    ) -> RuntimePayload:
        source_values = source_payload
        if not isinstance(source_values, RuntimeSliceAlignedValueSet):
            raise TypeError(
                "Runtime-slice-aligned image output strategy requires "
                f"RuntimeSliceAlignedValueSet, got {type(source_values).__name__}."
            )
        return RuntimeSliceAlignedImageOutputContext(
            source_values=source_values,
            output_value=output_value,
        ).payload()


@dataclass(frozen=True, slots=True)
class RuntimeSliceAlignedImageOutputContext:
    """Compose stack metadata for image output derived from aligned source values."""

    source_values: RuntimeSliceAlignedValueSet
    output_value: RuntimePayload

    def payload(self) -> RuntimePayload:
        output_data = image_payload_data(self.output_value)
        output_slices = self.output_slices(output_data)
        contextualized_slices = tuple(
            DerivedImagePayloadContext(
                self.source_values.value_for_aligned_slice(
                    slice_index,
                    len(output_slices),
                ),
                output_slice,
            ).payload()
            for slice_index, output_slice in enumerate(output_slices)
        )
        return stack_image_payload_context(contextualized_slices, output_data)

    @staticmethod
    def output_slices(output_data: RuntimeArrayData) -> tuple[RuntimeArrayData, ...]:
        if is_color_image_slice(output_data):
            return (output_data,)
        output_shape = np.shape(output_data)
        if len(output_shape) < 3:
            return (output_data,)
        return tuple(output_data[slice_index] for slice_index in range(output_shape[0]))


class ObjectLabelsFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Preserve source-image metadata for object-label outputs."""

    kind = ArtifactKind.OBJECT_LABELS

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
    ) -> ObjectLabelContextualizedOutput:
        del output_plan
        return ObjectLabelOutputValueContextStrategy.for_output_value(
            output_value,
        ).contextualize(source_payload, output_value)


class ObjectLabelOutputValueContextStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered object-label output contextualization by nominal value type."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def for_output_value(
        cls,
        output_value: ObjectLabelContextualizableOutput,
    ) -> "ObjectLabelOutputValueContextStrategy":
        strategy = cls.for_nominal_value(output_value)
        if strategy is None:
            return RawObjectLabelOutputValueContextStrategy()
        return strategy

    @abstractmethod
    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: ObjectLabelContextualizableOutput,
    ) -> ObjectLabelContextualizedOutput:
        """Return the output with source-image context attached when possible."""


class RuntimeSliceAlignedObjectLabelOutputValueContextStrategy(
    ObjectLabelOutputValueContextStrategy
):
    """Contextualize each runtime-slice-aligned object-label output slice."""

    value_type = RuntimeSliceAlignedValueSet

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: ObjectLabelContextualizableOutput,
    ) -> ObjectLabelContextualizedOutput:
        aligned_values = output_value
        if not isinstance(aligned_values, RuntimeSliceAlignedValueSet):
            raise TypeError(
                "Runtime-slice-aligned object-label output strategy requires "
                f"RuntimeSliceAlignedValueSet, got {type(aligned_values).__name__}."
            )
        return RuntimeSliceAlignedValues(
            tuple(
                ObjectLabelOutputValueContextStrategy.for_output_value(
                    aligned_values.value_for_slice(slice_index)
                ).contextualize(
                    SourceImagePayloadSlice(
                        source_payload,
                        slice_index,
                    ).payload(),
                    aligned_values.value_for_slice(slice_index),
                )
                for slice_index in range(aligned_values.slice_count)
            )
        )


class ContextualObjectLabelOutputValueContextStrategy(
    ObjectLabelOutputValueContextStrategy
):
    """Preserve object-label domain while filling missing source-image context."""

    value_type = ObjectLabelValue

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: ObjectLabelContextualizableOutput,
    ) -> ObjectLabelValue:
        if not isinstance(output_value, ObjectLabelValue):
            raise TypeError(
                "Contextual object-label output strategy requires "
                f"ObjectLabelValue, got {type(output_value).__name__}."
            )
        return output_value.with_source_image_context(source_payload)


class RawObjectLabelOutputValueContextStrategy(ObjectLabelOutputValueContextStrategy):
    """Build object-label payload context for raw array-like label outputs."""

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: ObjectLabelContextualizableOutput,
    ) -> ObjectLabelValue:
        if not is_array_payload(output_value):
            raise TypeError(
                "Object-label output must be an OpenHCS object-label value, "
                "runtime-slice-aligned value, or array payload; got "
                f"{type(output_value).__name__}."
            )
        return SourceImageObjectLabelBuildRequest(
            image=source_payload,
            labels=output_value,
        ).payload()


@dataclass(frozen=True)
class ComponentArtifactPlans:
    """Artifact plans selected for one grouped component execution."""

    inputs: ArtifactInputPlans
    outputs: ArtifactOutputPlans

    @classmethod
    def from_step_component(
        cls,
        plan: FunctionStepExecutionPlan,
        component_key: str | None,
    ) -> "ComponentArtifactPlans":
        return cls(
            inputs=cls._select_plan_for_component(
                plan.artifact_inputs_by_group,
                component_key,
                plan.artifact_inputs,
            ),
            outputs=cls._select_plan_for_component(
                plan.artifact_outputs_by_group,
                component_key,
                plan.artifact_outputs,
            ),
        )

    def select_for_invocation(
        self,
        invocation: CompiledFunctionInvocation,
    ) -> "ComponentArtifactPlans":
        return ComponentArtifactPlans(
            inputs=invocation.select_inputs(self.inputs),
            outputs=invocation.select_outputs(self.outputs),
        )

    @staticmethod
    def _select_plan_for_component(
        plan_by_group: Optional[Mapping[str | None, ArtifactOutputPlans | ArtifactInputPlans]],
        component_key: Optional[str],
        default_plan: ArtifactOutputPlans | ArtifactInputPlans,
    ) -> ArtifactOutputPlans | ArtifactInputPlans:
        if not plan_by_group:
            return default_plan

        global_plan = (
            plan_by_group[None]
            if None in plan_by_group
            else EMPTY_ARTIFACT_PLANS
        )
        if component_key in plan_by_group:
            return {
                **global_plan,
                **plan_by_group[component_key],
            }
        if global_plan:
            return global_plan
        return default_plan


@dataclass(frozen=True, slots=True, kw_only=True)
class PatternGroupExecutionScope:
    """Shared pattern-group execution coordinates."""

    context: ProcessingContext
    execution_plan: FunctionStepExecutionPlan
    compiled_group: CompiledFunctionGroup
    component_value: RuntimeComponentValue = None

    @property
    def component_key(self) -> str | None:
        if self.component_value is None:
            return None
        return str(self.component_value)

    @property
    def source_binding_plan(self) -> CompiledSourceBindingPlan:
        return self.execution_plan.source_binding_plan

    @property
    def axis_component(self) -> str | None:
        return self.execution_plan.group_by_value

    @property
    def axis_component_value(self) -> str | None:
        return self.component_key

    @property
    def axis_scope(self) -> RuntimeExecutionAxisScope:
        return RuntimeExecutionAxisScope.from_raw(
            self.execution_plan.axis_id,
            component=self.axis_component,
            value=self.axis_component_value,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FunctionRuntimeScope(PatternGroupExecutionScope, SourceBindingRuntimeContext):
    """Generic runtime scope shared by chain, invocation, adapter, and debug code."""

    artifacts: ComponentArtifactPlans
    runtime_plane_index: int
    runtime_plane_count: int

    @classmethod
    def from_pattern_group(
        cls,
        request: "PatternGroupExecutionRequest",
        loaded: "PatternGroupData",
    ) -> "FunctionRuntimeScope":
        artifacts = ComponentArtifactPlans.from_step_component(
            request.execution_plan,
            request.component_key,
        )
        logger.debug(
            "Selected artifact outputs for component %s: %s",
            request.component_key,
            artifacts.outputs,
        )
        return cls(
            context=request.context,
            execution_plan=request.execution_plan,
            compiled_group=request.compiled_group,
            artifacts=artifacts,
            source_binding_context=loaded,
            runtime_plane_index=request.component_index,
            runtime_plane_count=request.component_count,
            component_value=request.component_value,
        )

    def require_invocations(self) -> None:
        if self.compiled_group.invocations:
            return
        raise ValueError(
            f"Compiled function group {self.compiled_group.group_key} has no invocations."
        )

    def execute_chain(self, initial_data_stack: RuntimeArrayData) -> RuntimeArrayData:
        self.require_invocations()
        current_stack = initial_data_stack
        current_memory_type = self.execution_plan.input_memory_type
        debug_sink = debug_event_sink_from_context(self.context)
        for invocation in self.compiled_group.invocations:
            group_key = invocation.key.runtime_group_key(self.component_value)
            plane_index = None
            projects_runtime_plane = self.execution_plan.group_projects_runtime_plane
            if group_key is not None and projects_runtime_plane:
                plane_index = self.runtime_plane_index
            executor = FunctionCoreExecutor(
                main_data_arg=current_stack,
                source_memory_type=current_memory_type,
                runtime_scope=self,
                invocation=invocation,
                artifacts=self.artifacts.select_for_invocation(invocation),
                group_key=group_key,
                plane_projection=RuntimePlaneProjection.for_execution_group(
                    group_key,
                    plane_index=plane_index,
                    plane_count=(
                        self.runtime_plane_count if projects_runtime_plane else None
                    ),
                    projects_runtime_plane=projects_runtime_plane,
                ),
            )
            captures_debug = debug_sink.captures_invocation_events()
            if captures_debug and debug_sink.should_skip_invocation(
                executor.debug_cursor()
            ):
                continue

            invocation_started_at = time.perf_counter()
            if captures_debug:
                debug_sink.record(executor.debug_event(DebugEventType.BEFORE_INVOCATION))
            try:
                current_stack = executor.execute()
            except Exception as exc:
                if captures_debug:
                    debug_sink.record(
                        executor.debug_event(
                            DebugEventType.EXCEPTION,
                            exception=exc,
                        )
                    )
                raise
            invocation_seconds = time.perf_counter() - invocation_started_at
            if captures_debug:
                after_event = executor.debug_event(
                    DebugEventType.AFTER_INVOCATION,
                    timing_seconds=invocation_seconds,
                )
                debug_sink.record(after_event)
                if debug_sink.should_stop_after_invocation(after_event):
                    break
            RuntimeProfileSink.record(
                "invocation_total",
                invocation_seconds,
                function=invocation.key.function_name,
                group=invocation.key.group_key,
                position=invocation.key.position,
            )
            current_memory_type = executor.memory_types().output_type
        return current_stack

@dataclass(frozen=True, slots=True, kw_only=True)
class PatternGroupExecutionRequest(PatternGroupExecutionScope):
    """All runtime data needed to process one pattern group."""

    pattern_group_info: JsonValue
    component_index: int
    component_count: int


@dataclass(frozen=True, kw_only=True)
class PatternGroupData(SourceBindingRuntimeContext):
    """Loaded image data for one pattern group."""

    matching_files: list[str]
    main_data_stack: RuntimeArrayData
    source_slice_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class OutputPathBatchUniqueness:
    """Validate that one runtime output batch has unique destination paths."""

    output_paths: Sequence[str]
    input_paths: Sequence[str]
    step_name: str
    pattern_repr: str

    def validate(self) -> None:
        counts: dict[str, int] = {}
        for path in self.output_paths:
            if path not in counts:
                counts[path] = 1
                continue
            counts[path] += 1
        duplicates = tuple(path for path, count in counts.items() if count > 1)
        if not duplicates:
            return
        raise ValueError(
            f"Step {self.step_name!r} produced duplicate output path(s) "
            f"for pattern {self.pattern_repr}: {duplicates!r}. Input files: "
            f"{tuple(self.input_paths)!r}."
        )


def _save_artifact_value(
    context: ProcessingContext,
    output_plan: ArtifactOutputPlan,
    value: RuntimePayload,
    source_payload: RuntimePayload,
    *,
    group_key: str | None,
) -> None:
    """Validate and save one planned artifact value to the memory VFS."""
    resolved_output_plan = output_plan.for_group(group_key)
    vfs_path = resolved_output_plan.path
    axis_id = _require_axis_id(context)
    contextualized_value = FunctionOutputContextStrategy.for_output_plan(
        resolved_output_plan
    ).contextualize(source_payload, value, resolved_output_plan)
    runtime_value = normalize_artifact_value(
        resolved_output_plan,
        contextualized_value,
        axis_id=axis_id,
    )

    location = RuntimeArtifactLocation(
        path=vfs_path,
        backend=Backend.MEMORY.value,
    )
    runtime_value_store = context.runtime_value_store
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
) -> RuntimePayload:
    """Load an artifact input from VFS through its typed runtime store record."""
    store = context.runtime_value_store
    axis_id = _require_axis_id(context)
    query = RuntimeArtifactQuery.from_input_plan(
        input_plan=input_plan,
        axis_id=axis_id,
        backend=Backend.MEMORY.value,
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


def prepare_compiled_function_group(group: CompiledFunctionGroup) -> None:
    """Run optional preparation hooks for each callable in a compiled group."""
    for invocation in group.invocations:
        FunctionInvocationCallableResolver.prepare(invocation)


def prepare_compiled_context_callables(
    compiled_contexts: Mapping[str, ProcessingContext],
) -> None:
    """Prepare every compiled callable and runtime adapter visible in contexts."""
    prepared_group_keys: set[tuple[str, int, str]] = set()
    prepared_invocation_count = 0
    prepared_adapter_count = 0
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
                prepared_adapter_count += prepare_compiled_runtime_adapters(
                    context,
                    step_plan,
                    group,
                )
                prepared_group_keys.add(prepare_key)
    logger.info(
        "Prepared %d compiled callable invocations and %d runtime adapters across %d groups.",
        prepared_invocation_count,
        prepared_adapter_count,
        len(prepared_group_keys),
    )


def prepare_compiled_runtime_adapters(
    context: ProcessingContext,
    compiled_plan: "CompiledStepPlan",
    group: CompiledFunctionGroup,
) -> int:
    """Run compile-time preparation hooks for adapter-backed invocations."""
    execution_plan = FunctionStepExecutionPlan.from_context(
        context,
        compiled_plan.step_index,
    )
    source_context = compiled_source_binding_context(context, execution_plan)
    prepared_count = 0
    for invocation in group.invocations:
        runtime_adapter = invocation.contract.runtime_adapter
        if runtime_adapter is None:
            continue
        runtime_adapter.prepare_request(
            compile_runtime_adapter_request(
                context,
                execution_plan,
                group,
                invocation,
                source_context,
            )
        )
        prepared_count += 1
    return prepared_count


def compiled_source_binding_context(
    context: ProcessingContext,
    execution_plan: FunctionStepExecutionPlan,
) -> SourceBindingRuntimeContext:
    """Build the compile-owned source universe for runtime-adapter preparation."""
    source_projection = (
        VirtualWorkspaceSourceProjectionAuthority.from_context(
            context,
            cache=context.runtime_source_workspace_projection_cache,
        ).projection_if_available()
    )
    return SourceBindingRuntimeContextRequest.from_context(
        context=context,
        plan=execution_plan,
        matching_files=(),
        source_projection=source_projection,
    ).runtime_context()


def compile_runtime_adapter_request(
    context: ProcessingContext,
    execution_plan: FunctionStepExecutionPlan,
    group: CompiledFunctionGroup,
    invocation: CompiledFunctionInvocation,
    source_context: SourceBindingRuntimeContext,
) -> RuntimeAdapterRequest:
    """Return the typed adapter request available at compile preparation time."""
    del group
    artifacts = ComponentArtifactPlans.from_step_component(
        execution_plan,
        None,
    ).select_for_invocation(invocation)
    return RuntimeAdapterRequest.from_source_context(
        context=context,
        artifact_inputs=artifacts.inputs,
        artifact_outputs=artifacts.outputs,
        source_binding_plan=execution_plan.source_binding_plan,
        source_binding_context=source_context,
        plane_projection=RuntimePlaneProjection.stack(),
        variable_components=tuple(execution_plan.variable_components),
    )


@dataclass(frozen=True, slots=True)
class FunctionCoreExecutor:
    """Execute one scoped callable invocation and route declared artifact I/O."""

    runtime_scope: FunctionRuntimeScope
    invocation: CompiledFunctionInvocation
    artifacts: ComponentArtifactPlans
    group_key: str | None
    plane_projection: RuntimePlaneProjection
    main_data_arg: RuntimeArrayData
    source_memory_type: str

    def runtime_adapter_request(self) -> RuntimeAdapterRequest:
        return RuntimeAdapterRequest.from_runtime_scope(
            runtime_scope=self.runtime_scope,
            artifact_inputs=self.artifacts.inputs,
            artifact_outputs=self.artifacts.outputs,
            group_key=self.group_key,
            plane_projection=self.plane_projection,
        )

    def debug_cursor(self) -> DebugCursor:
        return DebugCursor.from_invocation(
            step_index=self.runtime_scope.execution_plan.step_index,
            step_scope_id=self.runtime_scope.execution_plan.step_scope_id,
            invocation=self.invocation,
            pattern_group_identity=str(self.runtime_scope.runtime_plane_index),
        )

    def debug_artifacts(
        self,
        artifact_plans: ArtifactInputPlans | ArtifactOutputPlans,
    ) -> DebugArtifactRefProjection:
        return DebugArtifactRefProjection.from_artifact_plans(
            artifact_plans=artifact_plans,
            cursor=self.debug_cursor(),
        )

    def debug_event(
        self,
        event_type: DebugEventType,
        *,
        exception: Exception | None = None,
        timing_seconds: float | None = None,
    ) -> DebugEvent:
        return DebugEvent.for_invocation(
            event_type=event_type,
            cursor=self.debug_cursor(),
            step_name=self.runtime_scope.execution_plan.step_name,
            callable_name=self.invocation.key.function_name,
            axis_id=self.runtime_scope.execution_plan.axis_id,
            input_artifacts=self.debug_artifacts(self.artifacts.inputs),
            output_artifacts=self.debug_artifacts(self.artifacts.outputs),
            exception=exception,
            timing_seconds=timing_seconds,
            invocation_parameters=DebugInvocationParameter.from_kwargs(
                self.invocation.kwargs_dict
            ),
        )

    @property
    def func_callable(self) -> Callable:
        return FunctionInvocationCallableResolver.resolve(self.invocation)

    @property
    def base_kwargs(self) -> RuntimeCallableKwargs:
        return self.invocation.kwargs_dict

    @property
    def function_name(self) -> str:
        return self.func_callable.__name__

    def execute(self) -> RuntimePayload:
        memory_types = self.memory_types()
        main_data_arg = MainFlowMemoryConversion(
            payload=self.main_data_arg,
            source_type=self.source_memory_type,
            target_type=memory_types.input_type,
            gpu_id=self.runtime_scope.execution_plan.device_id,
        ).converted_payload()
        final_kwargs = dict(self.base_kwargs)
        parameter_names = _callable_parameter_names(self.func_callable)
        self.bind_compiled_runtime_parameters(final_kwargs)
        loads_artifact_inputs = self.should_load_artifact_inputs()
        loaded_artifact_payloads: dict[str, RuntimePayload] = {}
        if loads_artifact_inputs:
            loaded_artifact_payloads = self.load_artifact_inputs(final_kwargs)
        self.bind_runtime_owned_parameters(final_kwargs, parameter_names)
        self.bind_runtime_adapter(final_kwargs, parameter_names)
        raw_output = self.invoke(main_data_arg, final_kwargs)
        main_output = self.save_artifact_outputs(
            raw_output,
            main_data_arg,
            loaded_artifact_payloads=loaded_artifact_payloads,
        )
        if RuntimeImageSourceIdentityCompleteness(main_output).complete():
            return main_output
        main_output_source = self.main_output_source_payload(
            main_data_arg,
            loaded_artifact_payloads=loaded_artifact_payloads,
        )
        return FunctionOutputContextStrategy.for_output_plan(None).contextualize(
            main_output_source,
            main_output,
            None,
        )

    def memory_types(self) -> "FunctionChainInvocationMemoryTypes":
        return FunctionChainInvocationMemoryTypes.from_invocation(self.invocation)

    def main_output_source_payload(
        self,
        primary_source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[str, RuntimePayload],
    ) -> RuntimePayload:
        if RuntimeImageSourceIdentityCompleteness(primary_source_payload).complete():
            return primary_source_payload
        object_label_sources = self.object_label_source_payloads(
            loaded_artifact_payloads,
        )
        if not object_label_sources:
            return primary_source_payload
        if self.has_non_object_artifact_inputs(loaded_artifact_payloads):
            return primary_source_payload
        if len(object_label_sources) != 1:
            raise NotImplementedError(
                "Main-flow image output source context is ambiguous for multiple "
                "object-label artifact inputs with no image artifact input."
            )
        return object_label_sources[0]

    def object_label_source_payloads(
        self,
        loaded_artifact_payloads: Mapping[str, RuntimePayload],
    ) -> tuple[RuntimePayload, ...]:
        return tuple(
            self.require_artifact_runtime_payload(parameter_name, payload)
            for parameter_name, payload in loaded_artifact_payloads.items()
            if self.artifacts.inputs[parameter_name].kind is ArtifactKind.OBJECT_LABELS
        )

    def has_non_object_artifact_inputs(
        self,
        loaded_artifact_payloads: Mapping[str, RuntimePayload],
    ) -> bool:
        return any(
            self.artifacts.inputs[parameter_name].kind is not ArtifactKind.OBJECT_LABELS
            for parameter_name in loaded_artifact_payloads
        )

    @staticmethod
    def require_artifact_runtime_payload(
        parameter_name: str,
        value: RuntimePayload,
    ) -> RuntimePayload:
        if isinstance(value, (ObjectLabelValue, RuntimeSliceAlignedValueSet)):
            return value
        if is_array_payload(value):
            return value
        raise TypeError(
            f"Artifact input {parameter_name!r} must be a runtime payload, "
            f"got {type(value).__name__}."
        )

    def load_artifact_inputs(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
    ) -> dict[str, RuntimePayload]:
        if not self.should_load_artifact_inputs():
            return {}
        logger.info(
            f"Artifact inputs for {self.function_name}: {self.artifacts.inputs}"
        )
        loaded_artifact_payloads: dict[str, RuntimePayload] = {}
        for arg_name, input_plan in self.artifacts.inputs.items():
            loaded_value = self.load_artifact_input(arg_name, input_plan)
            final_kwargs[arg_name] = loaded_value
            loaded_artifact_payloads[arg_name] = loaded_value
        return loaded_artifact_payloads

    def should_load_artifact_inputs(self) -> bool:
        return bool(
            self.artifacts.inputs
            and not self.invocation.runtime_domain.adapter_manages_artifact_inputs
        )

    def load_artifact_input(
        self,
        arg_name: str,
        input_plan: ArtifactInputPlan,
    ) -> RuntimePayload:
        logger.info(
            f"Loading artifact input '{arg_name}' from path '{input_plan.path}' "
            "(memory backend)"
        )
        load_started_at = time.perf_counter()
        try:
            loaded_value = _load_artifact_input_value(
                self.runtime_scope.context,
                input_plan,
            )
        except Exception as exc:
            logger.error(
                f"Failed to load artifact input '{arg_name}' from "
                f"'{input_plan.path}': {exc}",
                exc_info=True,
            )
            raise
        RuntimeProfileSink.record(
            "artifact_input_load",
            time.perf_counter() - load_started_at,
            function=self.function_name,
            artifact=arg_name,
            kind=input_plan.kind.value,
        )
        return loaded_value

    def bind_runtime_owned_parameters(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
        parameter_names: frozenset[str],
    ) -> None:
        if RUNTIME_CONTEXT_PARAMETER_NAME in parameter_names:
            final_kwargs[RUNTIME_CONTEXT_PARAMETER_NAME] = self.runtime_scope.context
        if (
            RUNTIME_INVOCATION_OPTIONS_PARAMETER_NAME in parameter_names
            and self.invocation.invocation_options is not None
        ):
            final_kwargs[RUNTIME_INVOCATION_OPTIONS_PARAMETER_NAME] = (
                self.invocation.invocation_options
            )

    def bind_compiled_runtime_parameters(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
    ) -> None:
        for binding in self.invocation.runtime_parameter_bindings:
            final_kwargs[binding.parameter_name] = binding.value

    def bind_runtime_adapter(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
        parameter_names: frozenset[str],
    ) -> None:
        runtime_adapter = self.invocation.contract.runtime_adapter
        if runtime_adapter is None:
            return
        adapter_parameter = runtime_adapter.parameter_name
        if adapter_parameter not in parameter_names:
            raise TypeError(
                f"{self.function_name} declares runtime adapter parameter "
                f"'{adapter_parameter}', but its signature does not accept it."
            )
        adapter_started_at = time.perf_counter()
        final_kwargs[adapter_parameter] = runtime_adapter.factory(
            self.runtime_adapter_request()
        )
        RuntimeProfileSink.record(
            "runtime_adapter_factory",
            time.perf_counter() - adapter_started_at,
            function=self.function_name,
            adapter=adapter_parameter,
        )

    def invoke(
        self,
        main_data_arg: RuntimeArrayData,
        final_kwargs: dict[str, RuntimeCallableArgument],
    ) -> RuntimeFunctionOutput:
        logger.info(f"Executing function: {self.function_name}")
        call_started_at = time.perf_counter()
        raw_output = self.func_callable(
            image_payload_data(main_data_arg),
            **final_kwargs,
        )
        RuntimeProfileSink.record(
            "function_call",
            time.perf_counter() - call_started_at,
            function=self.function_name,
        )
        return raw_output

    def save_artifact_outputs(
        self,
        raw_output: RuntimeFunctionOutput,
        source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[str, RuntimePayload],
    ) -> RuntimePayload:
        if isinstance(raw_output, StepResult):
            self.save_step_result_artifacts(
                raw_output,
                source_payload,
                loaded_artifact_payloads=loaded_artifact_payloads,
            )
            return raw_output.image
        if isinstance(raw_output, tuple):
            return self.save_tuple_output(
                raw_output,
                source_payload,
                loaded_artifact_payloads=loaded_artifact_payloads,
            )
        return raw_output

    def save_tuple_output(
        self,
        raw_output: tuple[RuntimePayload, ...],
        source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[str, RuntimePayload],
    ) -> RuntimePayload:
        artifact_outputs = tuple(self.artifacts.outputs.values())
        artifact_count = len(artifact_outputs)
        if artifact_count == 0:
            return AlignedImageStack(raw_output)
        if len(raw_output) == artifact_count:
            self.save_tuple_artifacts(
                raw_output,
                source_payload,
                loaded_artifact_payloads=loaded_artifact_payloads,
            )
            return self.main_flow_output_from_artifacts(raw_output, artifact_outputs)
        if len(raw_output) == artifact_count + 1:
            self.save_tuple_artifacts(
                raw_output[1:],
                source_payload,
                loaded_artifact_payloads=loaded_artifact_payloads,
            )
            return raw_output[0]
        raise ValueError(
            f"Function returned {len(raw_output)} tuple values for "
            f"{artifact_count} planned artifact output(s). Tuple outputs must either "
            "match the declared artifact outputs exactly or include one primary "
            "main-flow value followed by all declared artifacts."
        )

    @staticmethod
    def main_flow_output_from_artifacts(
        raw_output: tuple[RuntimePayload, ...],
        artifact_outputs: tuple[ArtifactOutputPlan, ...],
    ) -> RuntimePayload:
        main_flow_output_items = tuple(
            (value, output_plan)
            for value, output_plan in zip(raw_output, artifact_outputs, strict=True)
            if output_plan.kind.participates_in_main_flow_output
        )
        if not main_flow_output_items:
            raise ValueError(
                "Function returned only declared artifact outputs, but none of the "
                "planned artifact kinds participates in main-flow output."
            )
        return AlignedImageStack(
            slices=tuple(value for value, _output_plan in main_flow_output_items),
            slice_contexts=tuple(
                AlignedImageSliceContext.main_flow(
                    output_key=output_plan.name,
                    artifact_kind=output_plan.kind.value,
                )
                for _value, output_plan in main_flow_output_items
            ),
        )

    def save_step_result_artifacts(
        self,
        step_result: StepResult,
        source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[str, RuntimePayload],
    ) -> None:
        for output_key, output_plan in self.artifacts.outputs.items():
            if output_key not in step_result.artifacts:
                raise ValueError(
                    f"Function returned StepResult without planned artifact "
                    f"'{output_key}'."
                )
            self.save_artifact_output(
                output_key,
                output_plan,
                step_result.artifacts[output_key],
                source_payload,
                loaded_artifact_payloads=loaded_artifact_payloads,
            )

    def save_tuple_artifacts(
        self,
        returned_artifact_values: tuple[RuntimePayload, ...],
        source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[str, RuntimePayload],
    ) -> None:
        for index, (output_key, output_plan) in enumerate(
            self.artifacts.outputs.items()
        ):
            if index >= len(returned_artifact_values):
                logger.error(
                    f"Artifact output plan wants to save '{output_key}', but function "
                    f"only returned {len(returned_artifact_values)} artifact values."
                )
                raise ValueError(
                    "Function did not return enough values for all planned artifact "
                    f"outputs. Missing value for '{output_key}'."
                )
            self.save_artifact_output(
                output_key,
                output_plan,
                returned_artifact_values[index],
                source_payload,
                loaded_artifact_payloads=loaded_artifact_payloads,
            )

    def save_artifact_output(
        self,
        output_key: str,
        output_plan: ArtifactOutputPlan,
        value: RuntimePayload,
        source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[str, RuntimePayload],
    ) -> None:
        logger.info(
            f"Saving artifact output '{output_key}' to VFS path '{output_plan.path}' "
            "(memory backend)"
        )
        save_started_at = time.perf_counter()
        _save_artifact_value(
            self.runtime_scope.context,
            output_plan,
            value,
            self.artifact_output_source_payload(
                output_plan,
                value,
                source_payload,
                loaded_artifact_payloads=loaded_artifact_payloads,
            ),
            group_key=self.group_key,
        )
        RuntimeProfileSink.record(
            "artifact_output_save",
            time.perf_counter() - save_started_at,
            function=self.function_name,
            artifact=output_key,
            kind=output_plan.kind.value,
        )

    def artifact_output_source_payload(
        self,
        output_plan: ArtifactOutputPlan,
        output_value: RuntimePayload,
        primary_source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[str, RuntimePayload],
    ) -> RuntimePayload:
        if (
            output_plan.kind is ArtifactKind.IMAGE
            and not RuntimeImageSourceIdentityCompleteness(output_value).complete()
        ):
            object_label_sources = self.object_label_source_payloads(
                loaded_artifact_payloads,
            )
            if object_label_sources and not self.has_non_object_artifact_inputs(
                loaded_artifact_payloads
            ):
                if len(object_label_sources) != 1:
                    raise NotImplementedError(
                        "Image artifact output source context is ambiguous for "
                        "multiple object-label artifact inputs with no image "
                        "artifact input."
                    )
                return object_label_sources[0]
        return primary_source_payload


@dataclass(frozen=True, slots=True)
class FunctionChainInvocationMemoryTypes:
    """Validated memory types for one compiled invocation."""

    input_type: str
    output_type: str

    @classmethod
    def from_invocation(
        cls,
        invocation: CompiledFunctionInvocation,
    ) -> "FunctionChainInvocationMemoryTypes":
        if invocation.input_memory_type is None or invocation.output_memory_type is None:
            raise ValueError(
                f"Compiled invocation {invocation.key} is missing memory types."
            )
        return cls(invocation.input_memory_type, invocation.output_memory_type)


@dataclass(frozen=True, slots=True)
class VariableComponentNames:
    """Microscope parser variable-component names for pattern lookup."""

    components: Sequence[VariableComponents]

    @property
    def value(self) -> list[str] | None:
        if not self.components:
            return None
        return [component.value for component in self.components]


@dataclass(frozen=True, slots=True)
class MainFlowMemoryConversion:
    """Main-flow image memory conversion preserving payload context."""

    payload: RuntimeArrayData
    source_type: str
    target_type: str
    gpu_id: int

    def converted_payload(self) -> RuntimeArrayData:
        data = image_payload_data(self.payload)
        converted = convert_memory(
            data=data,
            source_type=self.source_type,
            target_type=self.target_type,
            gpu_id=self.gpu_id,
        )
        return with_image_payload_data(self.payload, converted)


@dataclass(slots=True)
class PatternGroupOutputData:
    """Unstacked output slices plus declared per-slice output semantics."""

    slices: list[RuntimeArrayData]
    slice_contexts: tuple[AlignedImageSliceContext, ...] = ()
    stack_payload: RuntimeArrayData | None = None

    def __post_init__(self) -> None:
        if not self.slice_contexts:
            self.slice_contexts = tuple(
                AlignedImageSliceContext.anonymous_main_flow()
                for _slice in self.slices
            )
        if len(self.slice_contexts) != len(self.slices):
            raise ValueError(
                "PatternGroupOutputData.slice_contexts must match slices; "
                f"got {len(self.slice_contexts)} context(s) for {len(self.slices)} slice(s)."
            )

    def __iter__(self) -> Iterator[RuntimeArrayData]:
        return iter(self.slices)

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> RuntimeArrayData:
        return self.slices[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PatternGroupOutputData):
            return (
                self.slices == other.slices
                and self.slice_contexts == other.slice_contexts
            )
        if isinstance(other, SequenceABC):
            return self.slices == list(other)
        return NotImplemented


class PatternGroupRuntime:
    """Staged runtime for one pattern group."""

    def __init__(self, request: PatternGroupExecutionRequest) -> None:
        self.request = request
        self.pattern_repr = str(request.pattern_group_info)[:100]

    def source_workspace_projection_cache(self) -> VirtualWorkspaceSourceProjectionCache:
        """Return the per-context source-workspace projection cache."""
        return self.request.context.runtime_source_workspace_projection_cache

    def source_workspace_projection_authority(
        self,
    ) -> VirtualWorkspaceSourceProjectionAuthority:
        return VirtualWorkspaceSourceProjectionAuthority.from_context(
            self.request.context,
            cache=self.source_workspace_projection_cache(),
        )

    def run(self) -> None:
        start_time = time.time()
        plan = self.request.execution_plan
        logger.debug(
            f"Processing pattern {self.pattern_repr} for axis {plan.axis_id}"
        )

        try:
            load_started_at = time.perf_counter()
            loaded = self._load_input_stack()
        except NoStepOutputManifestMatch:
            logger.debug(
                "Skipping stale pattern group %s for step %s (%s); no files "
                "belong to producer manifest.",
                self.pattern_repr,
                plan.step_index,
                plan.step_name,
            )
            return
        try:
            RuntimeProfileSink.record(
                "pattern_load_stack",
                time.perf_counter() - load_started_at,
                step=plan.step_index,
                step_name=plan.step_name,
                pattern=self.pattern_repr,
            )
            execute_started_at = time.perf_counter()
            processed_stack = self._execute_pattern(loaded)
            RuntimeProfileSink.record(
                "pattern_execute_chain",
                time.perf_counter() - execute_started_at,
                step=plan.step_index,
                step_name=plan.step_name,
                pattern=self.pattern_repr,
            )
            unstack_started_at = time.perf_counter()
            output_data = self._validate_and_unstack(processed_stack, loaded)
            RuntimeProfileSink.record(
                "pattern_validate_unstack",
                time.perf_counter() - unstack_started_at,
                step=plan.step_index,
                step_name=plan.step_name,
                pattern=self.pattern_repr,
            )
            save_started_at = time.perf_counter()
            output_records = self._save_outputs(output_data, loaded.matching_files)
            output_paths = [record.output_path for record in output_records]
            RuntimeProfileSink.record(
                "pattern_save_outputs",
                time.perf_counter() - save_started_at,
                step=plan.step_index,
                step_name=plan.step_name,
                pattern=self.pattern_repr,
            )
            cleanup_started_at = time.perf_counter()
            self._cleanup_collapsed_domains(
                output_data.slices,
                loaded.matching_files,
                output_paths,
            )
            step_output_manifest(self.request.context).record_outputs(
                plan,
                output_records,
            )
            RuntimeProfileSink.record(
                "pattern_cleanup",
                time.perf_counter() - cleanup_started_at,
                step=plan.step_index,
                step_name=plan.step_name,
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
        context = self.request.context
        plan = self.request.execution_plan
        request = self.request
        if not context.microscope_handler:
            raise RuntimeError("MicroscopeHandler not available in context.")

        output_manifest = step_output_manifest(context)
        producer_matching_files = output_manifest.producer_paths_matching_pattern(
            plan,
            str(request.pattern_group_info),
            context.microscope_handler.parser,
        )
        matching_files = list(producer_matching_files)
        if not matching_files:
            matching_files = context.microscope_handler.path_list_from_pattern(
                str(plan.input_dir),
                request.pattern_group_info,
                context.filemanager,
                Backend.MEMORY.value,
                VariableComponentNames(plan.variable_components).value,
            )
        matching_files = output_manifest.filter_to_producer_paths(
            plan,
            matching_files,
            context.microscope_handler.parser,
        )

        if not matching_files:
            raise ValueError(
                f"No matching files found for pattern group {self.pattern_repr} "
                f"in {plan.input_dir}. "
                f"This indicates either: (1) no image files exist in the directory, "
                f"(2) files don't match the pattern, or (3) pattern parsing failed. "
                f"Check that input files exist and match the expected naming convention."
            )

        matching_files = self._filter_matching_files_for_group(matching_files)
        matching_files = self._filter_matching_files_for_source_bindings(
            matching_files
        )

        logger.debug(
            "Pattern %s matched %d files: %s",
            self.pattern_repr,
            len(matching_files),
            [Path(f).name for f in matching_files],
        )

        matching_files.sort()
        logger.debug(
            f"Pattern {self.pattern_repr} sorted files: {[Path(f).name for f in matching_files]}"
        )

        full_file_paths = [str(plan.input_dir / f) for f in matching_files]
        source_projection = (
            self.source_workspace_projection_authority().projection_if_available()
        )
        source_binding_context = SourceBindingRuntimeContextRequest.from_context(
            context=self.request.context,
            plan=self.request.execution_plan,
            matching_files=matching_files,
            source_projection=source_projection,
        ).runtime_context()
        cached_stack = context.runtime_image_stack_cache.get(
            tuple(full_file_paths),
            memory_type=plan.input_memory_type,
        )
        RuntimeProfileSink.record(
            "runtime_stack_cache_get",
            0.0,
            step=plan.step_index,
            step_name=plan.step_name,
            hit=cached_stack is not None,
            paths=len(full_file_paths),
            memory_type=plan.input_memory_type,
        )
        if cached_stack is None:
            raw_slices = context.filemanager.load_batch(
                full_file_paths,
                Backend.MEMORY.value,
            )
            raw_slices = self._apply_source_image_loading_semantics(
                raw_slices,
                matching_files,
                full_file_paths,
                source_binding_context,
                source_projection,
            )

            if not raw_slices:
                raise ValueError(
                    f"No valid images loaded for pattern group {self.pattern_repr} "
                    f"in {plan.input_dir}. "
                    f"Found {len(matching_files)} matching files but failed to load any valid images. "
                    f"This indicates corrupted image files, unsupported formats, or I/O errors. "
                    f"Check file integrity and format compatibility."
                )

            raw_slice_data = tuple(image_payload_data(slice_data) for slice_data in raw_slices)
            main_data_stack = ImageStackLayout.for_slices(raw_slice_data).stack(
                slices=raw_slice_data,
                memory_type=plan.input_memory_type,
                gpu_id=plan.device_id,
            )
            main_data_stack = stack_image_payload_context(raw_slices, main_data_stack)
            source_slice_shapes = tuple(
                tuple(slice_data.shape)
                for slice_data in raw_slice_data
            )
        else:
            main_data_stack = cached_stack.stack
            source_slice_shapes = cached_stack.source_slice_shapes

        return PatternGroupData(
            matching_files=matching_files,
            main_data_stack=main_data_stack,
            source_slice_shapes=source_slice_shapes,
            source_binding_context=source_binding_context,
        )

    def _filter_matching_files_for_group(
        self,
        matching_files: list[str],
    ) -> list[str]:
        """Constrain grouped executions to files from the current component."""
        group_component = self.request.execution_plan.group_by_value
        component_value = self.request.component_value
        if group_component is None or component_value is None:
            return matching_files

        parser = self.request.context.microscope_handler.parser
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

    def _filter_matching_files_for_source_bindings(
        self,
        matching_files: list[str],
    ) -> list[str]:
        """Constrain the loaded main stack to declared image source bindings."""

        if (
            self.request.execution_plan.main_input_dependency.kind
            is StepInputDependencyKind.STEP_OUTPUT
        ):
            return matching_files

        bindings = tuple(
            binding
            for binding in self.request.execution_plan.source_binding_plan.bindings
            if binding.participates_in_execution_anchoring
        )
        selector_bindings = SourceBindingCandidateMatcher.selector_bindings(bindings)
        if not selector_bindings:
            return matching_files

        source_context = self._source_binding_candidate_context()
        compatible = list(
            SourceBindingMatchedImageSet.from_plan(
                bindings=selector_bindings,
                match_plan=self.request.execution_plan.source_binding_plan.match_plan,
                source_context=source_context,
                plane_member_fields=frozenset(
                    self.request.execution_plan.variable_component_values
                ),
            ).expand(
                matching_files,
                source_universe=self._source_binding_load_universe(),
            )
        )
        if compatible:
            return compatible

        raise ValueError(
            f"Source-bound step {self.request.execution_plan.step_name!r} resolved no files for "
            f"selector-bearing image bindings "
            f"{[binding.alias for binding in selector_bindings]!r} in pattern "
            f"{self.pattern_repr}. Matched files before source filtering: "
            f"{matching_files!r}."
        )

    def _source_binding_load_universe(self) -> tuple[str, ...]:
        """Return loadable files available for source image-set expansion."""
        source_projection = (
            self.source_workspace_projection_authority().projection_if_available()
        )
        request = SourceBindingRuntimeContextRequest.from_context(
            context=self.request.context,
            plan=self.request.execution_plan,
            matching_files=(),
            source_projection=source_projection,
        )
        return SourceUniverseRequest.runtime_state(request).require_load_universe().files

    def _source_binding_candidate_context(self) -> SourcePatternResolutionContext:
        projection = self.source_workspace_projection_authority().projection_or_empty()
        return SourcePatternResolutionContext.from_projection(
            parser=self.request.context.microscope_handler.parser,
            projection=self.source_workspace_projection_cache().filtered_by_axis(
                projection,
                axis_id=self.request.execution_plan.axis_id,
            ),
            metadata_rules=self.request.execution_plan.source_binding_plan.metadata_rules,
        )

    def _apply_source_image_loading_semantics(
        self,
        raw_slices: Sequence[RuntimeArrayData],
        matching_files: Sequence[str],
        full_file_paths: Sequence[str],
        source_binding_context: SourceBindingRuntimeContext,
        source_projection: VirtualWorkspaceSourceProjection | None,
    ) -> list[RuntimeArrayData]:
        if source_projection is not None:
            return [
                SourceImagePayloadSemantics.from_source_metadata(
                    source_projection.source_metadata_for(
                        VirtualWorkspacePathLookup.from_paths(
                            virtual_path,
                            full_virtual_path,
                        )
                    ),
                    source_projection.source_path_for(
                        VirtualWorkspacePathLookup.from_paths(
                            virtual_path,
                            full_virtual_path,
                        )
                    ),
                    Backend.DISK.value,
                    self.request.context.filemanager,
                ).apply(payload)
                for payload, virtual_path, full_virtual_path in zip(
                    raw_slices,
                    matching_files,
                    full_file_paths,
                )
            ]

        source_context = SourcePatternResolutionContext.from_sources(
            parser=self.request.context.microscope_handler.parser,
            source_paths_by_virtual_path=source_binding_context.step_input_source_paths,
            source_metadata_by_path=source_binding_context.source_metadata_by_path,
            metadata_rules=self.request.execution_plan.source_binding_plan.metadata_rules,
        )
        return [
            SourceImagePayloadSemantics.from_source_metadata(
                source_context.merged_metadata_for_paths(
                    (
                        virtual_path,
                        full_virtual_path,
                    )
                ),
                source_context.source_path_for(full_virtual_path),
                source_binding_context.step_input_source_backend,
                self.request.context.filemanager,
            ).apply(payload)
            for payload, virtual_path, full_virtual_path in zip(
                raw_slices,
                matching_files,
                full_file_paths,
            )
        ]

    def _execute_pattern(
        self,
        loaded: PatternGroupData,
    ) -> RuntimeArrayData:
        request = self.request
        runtime_scope = FunctionRuntimeScope.from_pattern_group(request, loaded)
        return runtime_scope.execute_chain(loaded.main_data_stack)

    def _validate_and_unstack(
        self,
        processed_stack: RuntimeArrayData,
        loaded: PatternGroupData,
    ) -> PatternGroupOutputData:
        if isinstance(processed_stack, AlignedImageStack):
            return PatternGroupOutputData(
                slices=list(flatten_aligned_image_payload_slices(processed_stack)),
                slice_contexts=flatten_aligned_image_slice_contexts(processed_stack),
            )
        processed_data = image_payload_data(processed_stack)
        try:
            unstack_started_at = time.perf_counter()
            output_slices = SourceSliceUnstackRequest(
                array=processed_data,
                source_slice_shapes=loaded.source_slice_shapes,
                memory_type=self.request.execution_plan.output_memory_type,
                gpu_id=self.request.execution_plan.device_id,
            ).slices()
            RuntimeProfileSink.record(
                "pattern_source_unstack",
                time.perf_counter() - unstack_started_at,
                step=self.request.execution_plan.step_index,
                step_name=self.request.execution_plan.step_name,
                slices=len(output_slices),
            )
        except ValueError as exc:
            output_shape = np.shape(processed_data)
            output_ndim = np.ndim(processed_data)
            logger.error("Function output is not an OpenHCS image stack.")
            logger.error(f"Output type: {type(processed_stack)}")
            logger.error("Output shape: %s", output_shape)
            logger.error("Output ndim: %s", output_ndim)
            raise ValueError(
                "Main processing must result in an image stack shaped "
                f"(N, H, W) or (N, H, W, C), got "
                f"{output_shape}"
            ) from exc

        context_started_at = time.perf_counter()
        output_payloads = unstack_image_payload_context(processed_stack, output_slices)
        RuntimeProfileSink.record(
            "pattern_payload_context_unstack",
            time.perf_counter() - context_started_at,
            step=self.request.execution_plan.step_index,
            step_name=self.request.execution_plan.step_name,
            slices=len(output_payloads),
        )
        return PatternGroupOutputData(
            slices=output_payloads,
            stack_payload=processed_stack,
        )

    def _save_outputs(
        self,
        output_data: PatternGroupOutputData,
        matching_files: list[str],
    ) -> list[ProducedOutputSemantics]:
        context = self.request.context
        output_slices = output_data.slices
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs < num_inputs:
            logger.debug(
                "Function returned %d images from %d inputs - likely "
                "flattening operation",
                num_outputs,
                num_inputs,
            )
        elif num_outputs > num_inputs:
            logger.debug(
                "Function returned %s output slices from %s positional input "
                "files; extra slices must carry payload component identity.",
                num_outputs,
                num_inputs,
            )

        output_payloads = []
        output_payload_metadata = []
        output_paths_batch = []
        output_records = []

        overwritten_output_paths: list[str] = []
        output_directory_exists = context.filemanager.exists(
            str(self.request.execution_plan.output_dir),
            Backend.MEMORY.value,
        )
        for i, img_slice in enumerate(output_slices):
            input_filename = None
            if i < len(matching_files):
                input_filename = matching_files[i]
            output_path_request = FunctionOutputPathRequest(
                parser=context.microscope_handler.parser,
                output_dir=self.request.execution_plan.output_dir,
                output_payload=img_slice,
                input_path=input_filename,
                variable_components=self.request.execution_plan.variable_components,
                identity_cache=context.runtime_function_output_identity_cache,
            )
            try:
                output_identity = FunctionOutputIdentityAuthority.identity(
                    output_path_request
                )
            except ValueError as exc:
                if input_filename is None:
                    raise ValueError(
                        f"Function returned {num_outputs} output slices but only "
                        f"{num_inputs} input files were available, and output slice "
                        f"{i} does not carry payload component identity."
                    ) from exc
                raise
            output_context = output_data.slice_contexts[i]
            if not output_context.is_anonymous_main_flow:
                output_identity = output_identity.with_filename_qualifier(
                    output_context.output_key
                )
            output_path = FunctionOutputPathAuthority.output_path_for_identity(
                output_path_request,
                output_identity,
            )
            output_path_text = str(output_path)
            output_metadata = image_payload_metadata(img_slice)
            output_component_metadata = output_identity.component_metadata()
            if output_metadata.source_component_metadata != output_component_metadata:
                output_metadata = output_metadata.with_source_component_metadata(
                    output_component_metadata
                )
                img_slice = with_image_payload_metadata(
                    img_slice,
                    metadata=output_metadata,
                )
            output_record = ProducedOutputSemantics.from_output(
                self.request.execution_plan,
                output_path_text,
                output_identity,
                output_context=output_context,
            )

            if output_directory_exists and context.filemanager.exists(
                output_path_text,
                Backend.MEMORY.value,
            ):
                overwritten_output_paths.append(output_path_text)

            output_payloads.append(img_slice)
            output_payload_metadata.append(output_metadata)
            output_paths_batch.append(output_path_text)
            output_records.append(output_record)

        OutputPathBatchUniqueness(
            output_paths=output_paths_batch,
            input_paths=matching_files,
            step_name=self.request.execution_plan.step_name,
            pattern_repr=self.pattern_repr,
        ).validate()

        if overwritten_output_paths:
            for output_path_text in overwritten_output_paths:
                context.filemanager.delete(output_path_text, Backend.MEMORY.value)
            context.runtime_image_stack_cache.discard_paths(
                tuple(overwritten_output_paths)
            )

        context.filemanager.ensure_directory(
            str(self.request.execution_plan.output_dir),
            Backend.MEMORY.value,
        )
        context.filemanager.save_batch(
            output_payloads,
            output_paths_batch,
            Backend.MEMORY.value,
        )
        stack_payload_data = (
            image_payload_data(output_data.stack_payload)
            if output_data.stack_payload is not None
            else None
        )
        if (
            output_data.stack_payload is not None
            and np.shape(stack_payload_data)[:1] == (len(output_payloads),)
        ):
            source_slice_shapes = tuple(
                tuple(image_payload_data(payload).shape)
                for payload in output_payloads
            )
            stack_payload = stack_image_payload_context_from_metadata(
                output_payloads,
                stack_payload_data,
                output_payload_metadata,
            )
            context.runtime_image_stack_cache.store(
                tuple(output_paths_batch),
                memory_type=self.request.execution_plan.output_memory_type,
                stack=stack_payload,
                source_slice_shapes=source_slice_shapes,
            )
            RuntimeProfileSink.record(
                "runtime_stack_cache_store",
                0.0,
                step=self.request.execution_plan.step_index,
                step_name=self.request.execution_plan.step_name,
                paths=len(output_paths_batch),
                memory_type=self.request.execution_plan.output_memory_type,
            )
        return output_records

    def _cleanup_collapsed_domains(
        self,
        output_slices: list[RuntimeArrayData],
        matching_files: list[str],
        output_paths: Sequence[str],
    ) -> None:
        context = self.request.context
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs >= num_inputs:
            return

        retained_paths = {Path(path).as_posix() for path in output_paths}
        for j in range(num_outputs, num_inputs):
            unused_filename = matching_files[j]
            for cleanup_dir in (
                self.request.execution_plan.input_dir,
                self.request.execution_plan.output_dir,
            ):
                unused_path = cleanup_dir / unused_filename
                if unused_path.as_posix() in retained_paths:
                    continue
                if context.filemanager.exists(
                    str(unused_path),
                    Backend.MEMORY.value,
                ):
                    context.runtime_image_stack_cache.discard_paths((str(unused_path),))
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
