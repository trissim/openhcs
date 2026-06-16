"""Runtime execution helpers for FunctionStep.

This module owns callable invocation, artifact routing, and pattern-group stack
execution. FunctionStep remains responsible for step-level orchestration.
"""

from abc import ABC, abstractmethod
from enum import Enum
import inspect
import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Hashable, Mapping, Optional, Sequence, TypeVar
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
from openhcs.core.callable_contract import CallableContract, prepare_processing_callable
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.debug import (
    DebugCursor,
    DebugEvent,
    DebugEventType,
    DebugArtifactRefProjection,
    DebugInvocationParameter,
    DebugEventSink,
    debug_event_sink_from_context,
)
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    DEFAULT_GROUP_KEY,
)
from openhcs.core.image_stack_layout import ImageStackLayout, SourceSliceUnstackRequest
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
from openhcs.core.runtime_slice_alignment import (
    RuntimeSliceAlignedValueSet,
    RuntimeSliceAlignedValues,
)
from openhcs.core.source_image_semantics import SourceImagePayloadSemantics
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
    ImagePayloadMetadataCompositionRequest,
    ImageMetadataPayload,
    MaskedImagePayload,
    ObjectLabelValue,
    RuntimeArrayData,
    SourceImageObjectLabelBuildRequest,
    image_payload_data,
    image_payload_metadata,
    image_payload_mask,
    image_payload_slice_context,
    is_array_payload,
    normalize_artifact_value,
    with_image_payload_data,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    MostDerivedContextStrategyMixin,
    NominalTypeKeyedStrategyMixin,
)
from openhcs.core.runtime_semantics import RuntimePlaneProjection
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan

logger = logging.getLogger(__name__)

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
_PROFILE_RUNTIME_PATH_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME_PATH"
PROCESSING_CONTEXT_OWNER_NAME = ProcessingContext.__name__
ArtifactInputPlans = Mapping[str, ArtifactInputPlan]
ArtifactOutputPlans = Mapping[str, ArtifactOutputPlan]
JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
OpenHCSMetadataPayload = Mapping[str, JsonValue]
OpenHCSSubdirectoryPayload = Mapping[str, JsonValue]
WorkspaceSourceRef = JsonValue
RuntimeComponentValue = JsonScalar
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
CallableContractCacheKey = tuple[Hashable | None, ...]
CallableCacheKey = int | tuple[str, CallableContractCacheKey]
LookupValueT = TypeVar("LookupValueT")
RuntimeProfileFieldValue = str | int | float | bool | None
RuntimeProfileFieldItems = tuple[tuple[str, RuntimeProfileFieldValue], ...]
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
    _cache: dict[CallableCacheKey, Callable] = {}

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
    def cache_key(cls, invocation: CompiledFunctionInvocation) -> CallableCacheKey:
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
    def contract_cache_key(contract: CallableContract) -> CallableContractCacheKey:
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
@dataclass(frozen=True, slots=True)
class RuntimeArtifactInvocationGroupKey:
    """Resolve artifact group identity for one runtime invocation."""

    invocation_group_key: str
    component_value: RuntimeComponentValue

    def artifact_group_key(self) -> str:
        if (
            self.invocation_group_key == DEFAULT_GROUP_KEY
            and self.component_value is not None
        ):
            return str(self.component_value)
        return self.invocation_group_key


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
        fields: "RuntimeProfileFieldSet",
    ) -> None:
        if not cls.enabled():
            return
        field_text = " ".join(f"{key}={value}" for key, value in fields.field_items())
        logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)
        profile_path = cls.environment_value(_PROFILE_RUNTIME_PATH_ENV)
        if profile_path is not None:
            with open(profile_path, "a", encoding="utf-8") as handle:
                handle.write(f"RUNTIME_PROFILE {label} {seconds:.6f}s {field_text}\n")


class RuntimeProfileFieldSet(ABC, metaclass=AutoRegisterMeta):
    """Nominal runtime-profile field payload."""

    __registry_key__ = "__name__"

    @abstractmethod
    def field_items(self) -> RuntimeProfileFieldItems:
        """Return ordered profile fields."""


@dataclass(frozen=True, slots=True)
class FunctionRuntimeProfileFieldSet(RuntimeProfileFieldSet):
    """Profile fields for one callable invocation."""

    function_name: str

    def field_items(self) -> RuntimeProfileFieldItems:
        return (("function", self.function_name),)


@dataclass(frozen=True, slots=True)
class ArtifactRuntimeProfileFieldSet(RuntimeProfileFieldSet):
    """Profile fields for artifact load/save work inside one callable."""

    function_name: str
    artifact_name: str
    artifact_kind: str

    def field_items(self) -> RuntimeProfileFieldItems:
        return (
            ("function", self.function_name),
            ("artifact", self.artifact_name),
            ("kind", self.artifact_kind),
        )


@dataclass(frozen=True, slots=True)
class AdapterRuntimeProfileFieldSet(RuntimeProfileFieldSet):
    """Profile fields for runtime-adapter construction."""

    function_name: str
    adapter_name: str

    def field_items(self) -> RuntimeProfileFieldItems:
        return (
            ("function", self.function_name),
            ("adapter", self.adapter_name),
        )


@dataclass(frozen=True, slots=True)
class InvocationRuntimeProfileFieldSet(RuntimeProfileFieldSet):
    """Profile fields for one compiled invocation."""

    function_name: str
    group_key: str
    position: int

    def field_items(self) -> RuntimeProfileFieldItems:
        return (
            ("function", self.function_name),
            ("group", self.group_key),
            ("position", self.position),
        )


@dataclass(frozen=True, slots=True)
class PatternRuntimeProfileFieldSet(RuntimeProfileFieldSet):
    """Profile fields for one pattern group."""

    step: int
    step_name: str
    pattern: str

    @classmethod
    def from_plan(
        cls,
        plan: FunctionStepExecutionPlan,
        pattern: str,
    ) -> "PatternRuntimeProfileFieldSet":
        return cls(
            step=plan.step_index,
            step_name=plan.step_name,
            pattern=pattern,
        )

    def field_items(self) -> RuntimeProfileFieldItems:
        return (
            ("step", self.step),
            ("step_name", self.step_name),
            ("pattern", self.pattern),
        )


@dataclass(frozen=True, slots=True)
class RuntimeProfileRecorder:
    """Function-scoped runtime profile field authority."""

    function_name: str

    def record_elapsed(
        self,
        label: str,
        *,
        started_at: float,
    ) -> None:
        RuntimeProfileSink.record(
            label,
            time.perf_counter() - started_at,
            FunctionRuntimeProfileFieldSet(self.function_name),
        )

    def record_artifact_elapsed(
        self,
        label: str,
        *,
        started_at: float,
        artifact_name: str,
        artifact_kind: str,
    ) -> None:
        RuntimeProfileSink.record(
            label,
            time.perf_counter() - started_at,
            ArtifactRuntimeProfileFieldSet(
                self.function_name,
                artifact_name,
                artifact_kind,
            ),
        )

    def record_adapter_elapsed(
        self,
        *,
        started_at: float,
        adapter_name: str,
    ) -> None:
        RuntimeProfileSink.record(
            "runtime_adapter_factory",
            time.perf_counter() - started_at,
            AdapterRuntimeProfileFieldSet(
                self.function_name,
                adapter_name,
            ),
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
    main_data_arg: RuntimeArrayData
    base_kwargs: RuntimeCallableKwargs
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


class RuntimeInjectedParameter(str, Enum):
    """Callable parameter names injected by the FunctionStep runtime."""

    CONTEXT = "context"
    RUNTIME_INVOCATION_OPTIONS = "runtime_invocation_options"


class SourceUniverseScope(str, Enum):
    """Source-binding universe requested by one runtime boundary."""

    STEP_INPUT = "step_input"
    PIPELINE_START = "pipeline_start"


class RuntimeInjectedParameterBinding(
    EnumKeyedStrategyMixin[RuntimeInjectedParameter],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered binder for runtime-owned callable parameters."""

    __registry_key__ = "parameter_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "parameter"
    __enum_label_attr__ = "parameter_label"

    parameter: ClassVar[RuntimeInjectedParameter | None] = None
    parameter_label: ClassVar[str | None] = None

    @classmethod
    def bind_all(
        cls,
        request: FunctionExecutionRequest,
        final_kwargs: dict[str, RuntimeCallableArgument],
        parameter_names: frozenset[str],
    ) -> None:
        for strategy_type in cls.registered_strategy_types():
            strategy_type().bind_if_accepted(
                request,
                final_kwargs,
                parameter_names,
            )

    def bind_if_accepted(
        self,
        request: FunctionExecutionRequest,
        final_kwargs: dict[str, RuntimeCallableArgument],
        parameter_names: frozenset[str],
    ) -> None:
        parameter_name = self.parameter_name()
        if parameter_name not in parameter_names:
            return
        self.bind(request, final_kwargs, parameter_name)

    def parameter_name(self) -> str:
        if self.parameter is None:
            raise TypeError(f"{type(self).__name__} must declare a parameter.")
        return self.parameter.value

    @abstractmethod
    def bind(
        self,
        request: FunctionExecutionRequest,
        final_kwargs: dict[str, RuntimeCallableArgument],
        parameter_name: str,
    ) -> None:
        """Bind one accepted runtime-owned callable parameter."""


class ContextRuntimeInjectedParameterBinding(RuntimeInjectedParameterBinding):
    """Inject ProcessingContext when a callable declares it."""

    parameter = RuntimeInjectedParameter.CONTEXT

    def bind(
        self,
        request: FunctionExecutionRequest,
        final_kwargs: dict[str, RuntimeCallableArgument],
        parameter_name: str,
    ) -> None:
        final_kwargs[parameter_name] = request.context


class InvocationOptionsRuntimeInjectedParameterBinding(
    RuntimeInjectedParameterBinding
):
    """Inject invocation options when a callable declares them."""

    parameter = RuntimeInjectedParameter.RUNTIME_INVOCATION_OPTIONS

    def bind(
        self,
        request: FunctionExecutionRequest,
        final_kwargs: dict[str, RuntimeCallableArgument],
        parameter_name: str,
    ) -> None:
        if request.invocation_options is None:
            return
        final_kwargs[parameter_name] = request.invocation_options


@dataclass(frozen=True)
class FunctionChainExecutionRequest:
    """Nominal request for a chain of callables over one image stack."""

    initial_data_stack: RuntimeArrayData
    invocations: Sequence[CompiledFunctionInvocation]
    context: ProcessingContext
    execution_plan: FunctionStepExecutionPlan
    artifact_inputs: ArtifactInputPlans
    artifact_outputs: ArtifactOutputPlans
    runtime_plane_index: int
    component_value: RuntimeComponentValue = None
    source_binding_context: SourceBindingRuntimeContext = (
        SourceBindingRuntimeContext.empty()
    )

@dataclass(frozen=True, slots=True)
class FunctionOutputContextRequest:
    """Source/output pair for runtime output context preservation."""

    source_payload: RuntimeArrayData
    output_value: RuntimePayload
    output_plan: ArtifactOutputPlan | None = None

    @property
    def kind(self) -> ArtifactKind:
        if self.output_plan is None:
            return ArtifactKind.IMAGE
        return self.output_plan.kind


@dataclass(frozen=True, slots=True)
class SourceImagePayloadSlice:
    """One source-image slice used to contextualize slice-aligned outputs."""

    source_payload: RuntimeArrayData
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
    def for_output(
        cls,
        request: FunctionOutputContextRequest,
    ) -> "FunctionOutputContextStrategy":
        strategy_type = cls.__registry__.get(request.kind.value)
        if strategy_type is None:
            return UnchangedFunctionOutputContextStrategy()
        return strategy_type()

    @abstractmethod
    def contextualize(
        self,
        request: FunctionOutputContextRequest,
    ) -> ObjectLabelContextualizedOutput:
        """Return output with source context preserved where semantics allow it."""


class UnchangedFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Leave outputs unchanged when no contextual image semantics are declared."""

    kind = ArtifactKind.SPECIAL

    def contextualize(
        self,
        request: FunctionOutputContextRequest,
    ) -> ObjectLabelContextualizedOutput:
        return request.output_value


class ImageFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Preserve source-image metadata for image outputs derived from the main input."""

    kind = ArtifactKind.IMAGE

    def contextualize(
        self,
        request: FunctionOutputContextRequest,
    ) -> ObjectLabelContextualizedOutput:
        return DerivedImagePayloadContext(
            request.source_payload,
            request.output_value,
        ).payload()


class ObjectLabelsFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Preserve source-image metadata for object-label outputs."""

    kind = ArtifactKind.OBJECT_LABELS

    def contextualize(
        self,
        request: FunctionOutputContextRequest,
    ) -> ObjectLabelContextualizedOutput:
        return ObjectLabelOutputValueContextStrategy.for_output_value(
            request.output_value,
        ).contextualize(request)


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
        request: FunctionOutputContextRequest,
    ) -> ObjectLabelContextualizedOutput:
        """Return the output with source-image context attached when possible."""


class RuntimeSliceAlignedObjectLabelOutputValueContextStrategy(
    ObjectLabelOutputValueContextStrategy
):
    """Contextualize each runtime-slice-aligned object-label output slice."""

    value_type = RuntimeSliceAlignedValueSet

    def contextualize(
        self,
        request: FunctionOutputContextRequest,
    ) -> ObjectLabelContextualizedOutput:
        aligned_values = request.output_value
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
                    FunctionOutputContextRequest(
                        source_payload=SourceImagePayloadSlice(
                            request.source_payload,
                            slice_index,
                        ).payload(),
                        output_value=aligned_values.value_for_slice(slice_index),
                        output_plan=request.output_plan,
                    )
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
        request: FunctionOutputContextRequest,
    ) -> ObjectLabelValue:
        output_value = request.output_value
        if not isinstance(output_value, ObjectLabelValue):
            raise TypeError(
                "Contextual object-label output strategy requires "
                f"ObjectLabelValue, got {type(output_value).__name__}."
            )
        return output_value.with_source_image_context(request.source_payload)


class RawObjectLabelOutputValueContextStrategy(ObjectLabelOutputValueContextStrategy):
    """Build object-label payload context for raw array-like label outputs."""

    def contextualize(
        self,
        request: FunctionOutputContextRequest,
    ) -> ObjectLabelValue:
        if not is_array_payload(request.output_value):
            raise TypeError(
                "Object-label output must be an OpenHCS object-label value, "
                "runtime-slice-aligned value, or array payload; got "
                f"{type(request.output_value).__name__}."
            )
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
    pattern_group_info: JsonValue
    compiled_group: CompiledFunctionGroup
    component_value: RuntimeComponentValue
    component_index: int


@dataclass(frozen=True)
class PatternGroupData:
    """Loaded image data for one pattern group."""

    matching_files: list[str]
    main_data_stack: RuntimeArrayData
    source_slice_shapes: tuple[tuple[int, ...], ...]
    source_binding_context: SourceBindingRuntimeContext


@dataclass(frozen=True, slots=True)
class VirtualWorkspacePathLookup:
    """Virtual workspace path identity for source path and metadata lookup."""

    virtual_path: str
    full_virtual_path: str

    @classmethod
    def from_paths(
        cls,
        virtual_path: str,
        full_virtual_path: str,
    ) -> "VirtualWorkspacePathLookup":
        return cls(str(virtual_path), str(full_virtual_path))

    def candidates(self) -> tuple[str, str]:
        return (self.virtual_path, self.full_virtual_path)


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceSourceProjection:
    """Source-binding projection derived from OpenHCS virtual-workspace metadata."""

    source_paths_by_virtual_path: Mapping[str, str]
    source_metadata_by_path: Mapping[str, Mapping[str, str]]
    workspace_root: str | None = None

    def first_virtual_path_value(
        self,
        mapping: Mapping[str, LookupValueT],
        lookup: VirtualWorkspacePathLookup,
    ) -> LookupValueT | None:
        """Return the first mapped value for a virtual/full path pair."""
        for key in lookup.candidates():
            value = mapping.get(key)
            if value is not None:
                return value
        return None

    def source_path_for(
        self,
        lookup: VirtualWorkspacePathLookup,
    ) -> str:
        """Return the physical source path represented by a virtual workspace path."""
        source_path = self.first_virtual_path_value(
            self.source_paths_by_virtual_path,
            lookup,
        )
        return lookup.full_virtual_path if source_path is None else str(source_path)

    def source_metadata_for(
        self,
        lookup: VirtualWorkspacePathLookup,
    ) -> Mapping[str, str] | None:
        """Return source metadata represented by a virtual workspace path."""
        metadata = self.first_virtual_path_value(
            self.source_metadata_by_path,
            lookup,
        )
        if metadata is not None:
            return metadata
        source_path = self.source_path_for(lookup)
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


@dataclass(frozen=True, slots=True)
class OpenHCSMetadataSubdirectories:
    """Typed view over OpenHCS metadata subdirectory payloads."""

    metadata: OpenHCSMetadataPayload

    def values(self) -> tuple[OpenHCSSubdirectoryPayload, ...]:
        from openhcs.microscopes.openhcs import FIELDS

        subdirectories = self.metadata.get(FIELDS.SUBDIRECTORIES)
        if subdirectories is None:
            return ()
        if not isinstance(subdirectories, Mapping):
            raise RuntimeError("OpenHCS metadata subdirectories must be a mapping.")
        return tuple(
            subdirectory
            for subdirectory in subdirectories.values()
            if isinstance(subdirectory, Mapping)
        )

    def has_workspace_mapping(self) -> bool:
        return any(
            VirtualWorkspaceMapping.from_subdirectory(subdirectory).has_entries
            for subdirectory in self.values()
        )


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceMapping:
    """Validated virtual-workspace mapping entries for one subdirectory."""

    entries: Mapping[str, WorkspaceSourceRef]

    @classmethod
    def from_subdirectory(
        cls,
        subdirectory: OpenHCSSubdirectoryPayload,
    ) -> "VirtualWorkspaceMapping":
        from openhcs.microscopes.openhcs import FIELDS

        mapping = subdirectory.get(FIELDS.WORKSPACE_MAPPING)
        if mapping is None:
            return cls(MappingProxyType({}))
        if not isinstance(mapping, Mapping):
            raise RuntimeError("virtual_workspace workspace_mapping must be a mapping.")
        return cls(MappingProxyType({str(key): value for key, value in mapping.items()}))

    @property
    def has_entries(self) -> bool:
        return bool(self.entries)

    def source_ref_for(self, virtual_path: str) -> WorkspaceSourceRef | None:
        return self.entries.get(virtual_path)


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceSourceMetadataEntries:
    """Validated source metadata entries for one virtual-workspace subdirectory."""

    entries: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_subdirectory(
        cls,
        subdirectory: OpenHCSSubdirectoryPayload,
    ) -> "VirtualWorkspaceSourceMetadataEntries":
        from openhcs.microscopes.openhcs import FIELDS

        source_metadata = subdirectory.get(FIELDS.SOURCE_METADATA)
        if source_metadata is None:
            return cls(MappingProxyType({}))
        if not isinstance(source_metadata, Mapping):
            raise RuntimeError(
                "virtual_workspace source metadata must be a path-keyed mapping."
            )
        return cls(
            MappingProxyType(
                {
                    str(virtual_path): cls.normalize_metadata_fields(metadata_fields)
                    for virtual_path, metadata_fields in source_metadata.items()
                }
            )
        )

    @staticmethod
    def normalize_metadata_fields(metadata_fields: JsonValue) -> Mapping[str, str]:
        if not isinstance(metadata_fields, Mapping):
            raise RuntimeError("virtual_workspace source metadata values must be mappings.")
        return MappingProxyType(
            {str(key): str(value) for key, value in metadata_fields.items()}
        )


@dataclass(slots=True)
class RealPathSourceMetadataIndex:
    """Track real-path source metadata while discarding ambiguous conflicts."""

    metadata_by_real_path: dict[str, Mapping[str, str]] = field(default_factory=dict)
    conflicted_real_paths: set[str] = field(default_factory=set)

    def record(self, real_path: str, metadata_fields: Mapping[str, str]) -> None:
        if real_path in self.conflicted_real_paths:
            return
        existing_metadata = self.metadata_by_real_path.get(real_path)
        if existing_metadata is None:
            self.metadata_by_real_path[real_path] = metadata_fields
            return
        if dict(existing_metadata) != dict(metadata_fields):
            del self.metadata_by_real_path[real_path]
            self.conflicted_real_paths.add(real_path)


@dataclass(slots=True)
class VirtualWorkspaceSourceProjectionBuilder:
    """Build source-binding projection data from OpenHCS virtual-workspace metadata."""

    plate_path: Path
    workspace_source_paths: dict[str, str] = field(default_factory=dict)
    source_metadata_by_path: dict[str, Mapping[str, str]] = field(default_factory=dict)
    real_path_metadata: RealPathSourceMetadataIndex = field(
        default_factory=RealPathSourceMetadataIndex
    )

    def ingest_subdirectory(self, subdirectory: OpenHCSSubdirectoryPayload) -> None:
        workspace_mapping = VirtualWorkspaceMapping.from_subdirectory(subdirectory)
        self.ingest_workspace_mapping(workspace_mapping)
        self.ingest_source_metadata(
            VirtualWorkspaceSourceMetadataEntries.from_subdirectory(subdirectory),
            workspace_mapping,
        )

    def ingest_workspace_mapping(self, workspace_mapping: VirtualWorkspaceMapping) -> None:
        for virtual_path, source_ref in workspace_mapping.entries.items():
            self.record_workspace_source_path(virtual_path, source_ref)

    def record_workspace_source_path(
        self,
        virtual_path: str,
        source_ref: WorkspaceSourceRef,
    ) -> None:
        real_path = self.source_path(source_ref)
        self.workspace_source_paths[virtual_path] = real_path
        self.workspace_source_paths[str(self.plate_path / virtual_path)] = real_path

    def ingest_source_metadata(
        self,
        source_metadata: VirtualWorkspaceSourceMetadataEntries,
        workspace_mapping: VirtualWorkspaceMapping,
    ) -> None:
        for virtual_path, metadata_fields in source_metadata.entries.items():
            self.record_source_metadata(virtual_path, metadata_fields, workspace_mapping)

    def record_source_metadata(
        self,
        virtual_path: str,
        metadata_fields: Mapping[str, str],
        workspace_mapping: VirtualWorkspaceMapping,
    ) -> None:
        normalized_metadata = source_schema_metadata_with_virtual_components(
            virtual_path,
            metadata_fields,
        )
        self.source_metadata_by_path[virtual_path] = normalized_metadata
        self.source_metadata_by_path[str(self.plate_path / virtual_path)] = (
            normalized_metadata
        )
        source_ref = workspace_mapping.source_ref_for(virtual_path)
        if source_ref is not None:
            self.real_path_metadata.record(
                self.source_path(source_ref),
                normalized_metadata,
            )

    def projection(self) -> VirtualWorkspaceSourceProjection:
        for real_path, metadata_fields in self.real_path_metadata.metadata_by_real_path.items():
            self.source_metadata_by_path[real_path] = metadata_fields
        if not self.workspace_source_paths:
            raise RuntimeError(
                "virtual_workspace source binding resolution requires "
                "workspace_mapping entries in OpenHCS metadata."
            )
        return VirtualWorkspaceSourceProjection(
            source_paths_by_virtual_path=MappingProxyType(self.workspace_source_paths),
            source_metadata_by_path=MappingProxyType(self.source_metadata_by_path),
            workspace_root=str(self.plate_path),
        )

    def source_path(self, source_ref: WorkspaceSourceRef) -> str:
        from openhcs.microscopes.openhcs import workspace_mapping_source_path

        return str(workspace_mapping_source_path(self.plate_path, source_ref))


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


@dataclass(frozen=True, slots=True)
class SourceFileUniverse:
    """Concrete file universe plus the backend that names those files."""

    files: tuple[str, ...]
    backend: Backend


@dataclass(frozen=True, slots=True)
class SourceUniverseRequest:
    """Source-file universe request for source-binding runtime resolution."""

    scope: SourceUniverseScope
    context: ProcessingContext
    plan: FunctionStepExecutionPlan
    matching_files: tuple[str, ...]
    source_backend: Backend
    source_projection: VirtualWorkspaceSourceProjection | None

    @property
    def requires_step_input_selector_resolution(self) -> bool:
        return self.plan.source_binding_plan.requires_step_input_selector_resolution

    @property
    def uses_virtual_workspace_projection(self) -> bool:
        return (
            self.source_backend is Backend.VIRTUAL_WORKSPACE
            and self.source_projection is not None
        )

    @property
    def requires_full_pipeline_source_universe(self) -> bool:
        if any(
            invocation.contract.runtime_adapter is not None
            for invocation in self.plan.compiled_function_pattern.iter_invocations()
        ):
            return True
        source_binding_plan = self.plan.source_binding_plan
        if source_binding_plan.metadata_rules:
            return True
        return any(
            binding.origin is SourceBindingOrigin.PIPELINE_START
            for bindings in source_binding_plan.bindings_by_group.values()
            for binding in bindings
        )

    @property
    def step_input_source_paths(self) -> Mapping[str, str]:
        projection = self.source_projection
        if projection is None:
            return MappingProxyType({})
        return projection.source_paths_by_virtual_path

    @property
    def source_metadata_by_path(self) -> Mapping[str, Mapping[str, str]]:
        projection = self.source_projection
        if projection is None:
            return MappingProxyType({})
        return projection.source_metadata_by_path

    def require_source_projection(self) -> VirtualWorkspaceSourceProjection:
        projection = self.source_projection
        if projection is None:
            raise RuntimeError("Virtual workspace source universe requires projection metadata.")
        return projection

    def axis_files(self) -> tuple[str, ...]:
        return tuple(
            self.plan.get_paths_for_axis(
                self.context.input_dir,
                self.source_backend.value,
            )
        )

    def disk_files(self) -> tuple[str, ...]:
        return tuple(
            str(path)
            for path in self.context.filemanager.list_files(
                str(self.context.input_dir),
                Backend.DISK.value,
                recursive=True,
            )
        )

    def physical_full_universe_backend(self) -> Backend:
        return PipelineStartListingBackendPolicy.backend_for(self.source_backend)


class PipelineStartListingBackendPolicy(
    EnumKeyedStrategyMixin[Backend],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Backend policy for full pipeline-start file listing."""

    __registry_key__ = "backend_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "source_backend"
    __enum_label_attr__ = "backend_label"

    source_backend: ClassVar[Backend | None] = None
    backend_label: ClassVar[str | None] = None

    @classmethod
    def backend_for(cls, source_backend: Backend) -> Backend:
        strategy_type = cls.__registry__.get(source_backend.value)
        if strategy_type is None:
            return source_backend
        return strategy_type().listing_backend()

    @abstractmethod
    def listing_backend(self) -> Backend:
        """Return the backend used for recursive full-universe listing."""


class DiskPipelineStartListingBackendPolicy(PipelineStartListingBackendPolicy):
    """Pipeline-start fan-out policy that lists disk files."""

    def listing_backend(self) -> Backend:
        return Backend.DISK


class MemoryPipelineStartListingBackendPolicy(DiskPipelineStartListingBackendPolicy):
    """Memory-backed pipeline-start fan-out lists disk files."""

    source_backend = Backend.MEMORY


class VirtualWorkspacePipelineStartListingBackendPolicy(
    DiskPipelineStartListingBackendPolicy
):
    """Virtual-workspace pipeline-start fan-out lists disk files."""

    source_backend = Backend.VIRTUAL_WORKSPACE


class SourceUniverseStrategy(
    MostDerivedContextStrategyMixin[SourceUniverseRequest],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered source-universe selection for source-binding runtime scopes."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True

    strategy_key: ClassVar[str | None] = None

    @classmethod
    def universe(cls, request: SourceUniverseRequest) -> SourceFileUniverse:
        strategy = cls.for_context(
            request,
            error_subject="Source universe",
        )
        if strategy is None:
            raise ValueError("Source universe requires a strategy.")
        return strategy.source_universe(request)

    @abstractmethod
    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        """Return source files and backend for pipeline-start bindings."""


class AxisFilesSourceUniverseStrategy(SourceUniverseStrategy):
    """Source-universe strategy that uses current-axis files from the source backend."""

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        return SourceFileUniverse(
            files=request.axis_files(),
            backend=request.source_backend,
        )


class CurrentPatternStepInputSourceUniverseStrategy(SourceUniverseStrategy):
    """Use the already-loaded pattern files when selectors do not need fan-out."""

    strategy_key = "step_input_current_pattern"

    def matches(self, request: SourceUniverseRequest) -> bool:
        return (
            request.scope is SourceUniverseScope.STEP_INPUT
            and not request.requires_step_input_selector_resolution
        )

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        return SourceFileUniverse(
            files=request.matching_files,
            backend=request.source_backend,
        )


class VirtualWorkspaceStepInputSourceUniverseStrategy(SourceUniverseStrategy):
    """Use source-schema virtual files when selector resolution must span sources."""

    strategy_key = "step_input_virtual_workspace_source_projection"

    def matches(self, request: SourceUniverseRequest) -> bool:
        return (
            request.scope is SourceUniverseScope.STEP_INPUT
            and request.requires_step_input_selector_resolution
            and request.uses_virtual_workspace_projection
        )

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        return SourceFileUniverse(
            files=request.require_source_projection().pipeline_start_files(
                axis_id=request.plan.axis_id
            ),
            backend=request.source_backend,
        )


class PhysicalAxisStepInputSourceUniverseStrategy(AxisFilesSourceUniverseStrategy):
    """Use physical axis files when source selectors need fan-out outside VWS."""

    strategy_key = "step_input_physical_axis"

    def matches(self, request: SourceUniverseRequest) -> bool:
        return (
            request.scope is SourceUniverseScope.STEP_INPUT
            and request.requires_step_input_selector_resolution
            and not request.uses_virtual_workspace_projection
        )


class AxisScopedPipelineStartSourceUniverseStrategy(AxisFilesSourceUniverseStrategy):
    """Use the current axis source files when full pipeline fan-out is unnecessary."""

    strategy_key = "axis_scoped"

    def matches(self, request: SourceUniverseRequest) -> bool:
        return (
            request.scope is SourceUniverseScope.PIPELINE_START
            and not request.requires_full_pipeline_source_universe
        )


class VirtualWorkspacePipelineStartSourceUniverseStrategy(SourceUniverseStrategy):
    """Use physical source paths plus disk files for full virtual-workspace fan-out."""

    strategy_key = "virtual_workspace_source_projection"

    def matches(self, request: SourceUniverseRequest) -> bool:
        return (
            request.scope is SourceUniverseScope.PIPELINE_START
            and
            request.requires_full_pipeline_source_universe
            and request.source_projection is not None
        )

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        projection = request.require_source_projection()
        return SourceFileUniverse(
            files=tuple(
                dict.fromkeys(
                    (
                        *(
                            str(path)
                            for path in projection.source_paths_by_virtual_path.values()
                        ),
                        *request.disk_files(),
                    )
                )
            ),
            backend=Backend.DISK,
        )


class PhysicalPipelineStartSourceUniverseStrategy(SourceUniverseStrategy):
    """Use a file listing backend for full pipeline fan-out outside VWS."""

    strategy_key = "physical_full_universe"

    def matches(self, request: SourceUniverseRequest) -> bool:
        return (
            request.scope is SourceUniverseScope.PIPELINE_START
            and
            request.requires_full_pipeline_source_universe
            and request.source_projection is None
        )

    def source_universe(self, request: SourceUniverseRequest) -> SourceFileUniverse:
        universe_backend = request.physical_full_universe_backend()
        return SourceFileUniverse(
            files=tuple(
                str(path)
                for path in request.context.filemanager.list_files(
                    str(request.context.input_dir),
                    universe_backend.value,
                    recursive=True,
                )
            ),
            backend=universe_backend,
        )


_SOURCE_BINDING_EXECUTION_CACHES: WeakKeyDictionary[
    ProcessingContext,
    SourceBindingExecutionCache,
] = WeakKeyDictionary()


def _save_artifact_value(
    context: ProcessingContext,
    output_plan: ArtifactOutputPlan,
    value: RuntimePayload,
    source_payload: RuntimeArrayData,
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


def _contextualize_main_output(
    source_payload: RuntimeArrayData,
    output_value: RuntimePayload,
) -> RuntimePayload:
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
) -> RuntimePayload:
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
    plan_by_group: Optional[Mapping[str | None, ArtifactOutputPlans | ArtifactInputPlans]],
    component_key: Optional[str],
    default_plan: ArtifactOutputPlans | ArtifactInputPlans,
) -> ArtifactOutputPlans | ArtifactInputPlans:
    """Select precompiled artifact I/O plan for a component."""
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


@dataclass(slots=True)
class FunctionCoreExecutor:
    """Execute one callable and route declared artifact I/O."""

    request: FunctionExecutionRequest

    @property
    def func_callable(self) -> Callable:
        return self.request.func_callable

    @property
    def function_name(self) -> str:
        return self.func_callable.__name__

    @property
    def profile(self) -> RuntimeProfileRecorder:
        return RuntimeProfileRecorder(function_name=self.function_name)

    def execute(self) -> RuntimePayload:
        final_kwargs = dict(self.request.base_kwargs)
        self.load_artifact_inputs(final_kwargs)
        parameter_names = _callable_parameter_names(self.func_callable)
        self.bind_runtime_context(final_kwargs, parameter_names)
        self.bind_runtime_adapter(final_kwargs, parameter_names)
        raw_output = self.invoke(final_kwargs)
        main_output = self.save_artifact_outputs(raw_output)
        return _contextualize_main_output(self.request.main_data_arg, main_output)

    def load_artifact_inputs(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
    ) -> None:
        if not self.should_load_artifact_inputs():
            return
        logger.info(
            f"Artifact inputs for {self.function_name}: {self.request.artifact_inputs}"
        )
        for arg_name, input_plan in self.request.artifact_inputs.items():
            self.load_artifact_input(arg_name, input_plan, final_kwargs)

    def should_load_artifact_inputs(self) -> bool:
        runtime_adapter = self.request.runtime_adapter
        return bool(
            self.request.artifact_inputs
            and not (
                runtime_adapter is not None
                and runtime_adapter.manages_artifact_inputs
            )
        )

    def load_artifact_input(
        self,
        arg_name: str,
        input_plan: ArtifactInputPlan,
        final_kwargs: dict[str, RuntimeCallableArgument],
    ) -> None:
        logger.info(
            f"Loading artifact input '{arg_name}' from path '{input_plan.path}' "
            "(memory backend)"
        )
        load_started_at = time.perf_counter()
        try:
            final_kwargs[arg_name] = _load_artifact_input_value(
                self.request.context,
                input_plan,
            )
        except Exception as exc:
            logger.error(
                f"Failed to load artifact input '{arg_name}' from "
                f"'{input_plan.path}': {exc}",
                exc_info=True,
            )
            raise
        self.profile.record_artifact_elapsed(
            "artifact_input_load",
            started_at=load_started_at,
            artifact_name=arg_name,
            artifact_kind=input_plan.kind.value,
        )

    def bind_runtime_context(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
        parameter_names: frozenset[str],
    ) -> None:
        RuntimeInjectedParameterBinding.bind_all(
            self.request,
            final_kwargs,
            parameter_names,
        )

    def bind_runtime_adapter(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
        parameter_names: frozenset[str],
    ) -> None:
        runtime_adapter = self.request.runtime_adapter
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
        self.profile.record_adapter_elapsed(
            started_at=adapter_started_at,
            adapter_name=adapter_parameter,
        )

    def runtime_adapter_request(self) -> RuntimeAdapterRequest:
        return RuntimeAdapterRequest(
            context=self.request.context,
            artifact_inputs=self.request.artifact_inputs,
            artifact_outputs=self.request.artifact_outputs,
            source_binding_plan=self.request.source_binding_plan,
            source_binding_context=self.request.source_binding_context,
            group_key=self.request.group_key,
            axis_component=self.request.axis_component,
            axis_component_value=self.request.axis_component_value,
            plane_projection=self.request.plane_projection,
        )

    def invoke(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
    ) -> RuntimeFunctionOutput:
        logger.info(f"Executing function: {self.function_name}")
        call_started_at = time.perf_counter()
        raw_output = self.func_callable(self.request.main_data_arg, **final_kwargs)
        self.profile.record_elapsed(
            "function_call",
            started_at=call_started_at,
        )
        return raw_output

    def save_artifact_outputs(
        self,
        raw_output: RuntimeFunctionOutput,
    ) -> RuntimePayload:
        if isinstance(raw_output, StepResult):
            self.save_step_result_artifacts(raw_output)
            return raw_output.image
        if isinstance(raw_output, tuple):
            self.save_tuple_artifacts(raw_output[1:])
            return raw_output[0]
        return raw_output

    def save_step_result_artifacts(self, step_result: StepResult) -> None:
        for output_key, output_plan in self.request.artifact_outputs.items():
            if output_key not in step_result.artifacts:
                raise ValueError(
                    f"Function returned StepResult without planned artifact "
                    f"'{output_key}'."
                )
            self.save_artifact_output(
                output_key,
                output_plan,
                step_result.artifacts[output_key],
            )

    def save_tuple_artifacts(
        self,
        returned_artifact_values: tuple[RuntimePayload, ...],
    ) -> None:
        for index, (output_key, output_plan) in enumerate(
            self.request.artifact_outputs.items()
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
            )

    def save_artifact_output(
        self,
        output_key: str,
        output_plan: ArtifactOutputPlan,
        value: RuntimePayload,
    ) -> None:
        logger.info(
            f"Saving artifact output '{output_key}' to VFS path '{output_plan.path}' "
            "(memory backend)"
        )
        save_started_at = time.perf_counter()
        _save_artifact_value(
            self.request.context,
            output_plan,
            value,
            self.request.main_data_arg,
            group_key=self.request.group_key,
        )
        self.profile.record_artifact_elapsed(
            "artifact_output_save",
            started_at=save_started_at,
            artifact_name=output_key,
            artifact_kind=output_plan.kind.value,
        )


@dataclass(frozen=True, slots=True)
class FunctionChainInvocationIdentity:
    """Runtime identity for one compiled invocation inside a chain."""

    group_key: str | None
    axis_component_value: str | None
    plane_projection: RuntimePlaneProjection

    @classmethod
    def from_request(
        cls,
        request: FunctionChainExecutionRequest,
        invocation: CompiledFunctionInvocation,
    ) -> "FunctionChainInvocationIdentity":
        group_key = RuntimeArtifactInvocationGroupKey(
            invocation_group_key=invocation.key.group_key,
            component_value=request.component_value,
        ).artifact_group_key()
        return cls(
            group_key=group_key,
            axis_component_value=ComponentValueString.from_value(
                request.component_value
            ).value,
            plane_projection=RuntimePlaneProjection.for_group_key(
                group_key,
                plane_index=RuntimeArtifactPlaneIndex(
                    group_key,
                    request.runtime_plane_index,
                ).value,
            ),
        )


@dataclass(frozen=True, slots=True)
class ComponentValueString:
    """String form of an optional runtime component value."""

    value: str | None

    @classmethod
    def from_value(cls, component_value: RuntimeComponentValue) -> "ComponentValueString":
        if component_value is None:
            return cls(None)
        return cls(str(component_value))


@dataclass(frozen=True, slots=True)
class RuntimeArtifactPlaneIndex:
    """Plane index semantics for invocation-scoped runtime artifacts."""

    group_key: str | None
    runtime_plane_index: int

    @property
    def value(self) -> int | None:
        if self.group_key is None:
            return None
        return self.runtime_plane_index


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
class FunctionChainInvocationArtifacts:
    """Artifact I/O plans selected for one invocation."""

    inputs: ArtifactInputPlans
    outputs: ArtifactOutputPlans

    @classmethod
    def from_request(
        cls,
        request: FunctionChainExecutionRequest,
        invocation: CompiledFunctionInvocation,
    ) -> "FunctionChainInvocationArtifacts":
        return cls(
            inputs=invocation.select_inputs(request.artifact_inputs),
            outputs=invocation.select_outputs(request.artifact_outputs),
        )


@dataclass(frozen=True, slots=True)
class VariableComponentNames:
    """Microscope parser variable-component names for pattern lookup."""

    components: Sequence[VariableComponents]

    @property
    def value(self) -> list[str] | None:
        if not self.components:
            return None
        return [component.value for component in self.components]


class InvocationDebugGate(ABC, metaclass=AutoRegisterMeta):
    """Debug skip gate for one invocation."""

    __registry_key__ = "gate_key"
    __skip_if_no_key__ = True

    gate_key: ClassVar[str | None] = None

    @classmethod
    def for_invocation(
        cls,
        sink: DebugEventSink,
        request: FunctionChainExecutionRequest,
        invocation: CompiledFunctionInvocation,
    ) -> "InvocationDebugGate":
        if not sink.captures_invocation_events():
            return InactiveInvocationDebugGate()
        return ActiveInvocationDebugGate(
            sink=sink,
            cursor=DebugCursor.from_invocation(
                step_index=request.execution_plan.step_index,
                step_scope_id=request.execution_plan.step_scope_id,
                invocation=invocation,
                pattern_group_identity=str(request.runtime_plane_index),
            ),
        )

    @abstractmethod
    def should_skip(self) -> bool:
        """Return whether this invocation should be skipped."""

    @abstractmethod
    def trace(
        self,
        request: FunctionChainExecutionRequest,
        invocation: CompiledFunctionInvocation,
        artifacts: FunctionChainInvocationArtifacts,
    ) -> "InvocationDebugTrace":
        """Return debug trace hooks for a non-skipped invocation."""


class InactiveInvocationDebugGate(InvocationDebugGate):
    """No-op debug gate when invocation debug events are disabled."""

    gate_key = "inactive"

    def should_skip(self) -> bool:
        return False

    def trace(
        self,
        request: FunctionChainExecutionRequest,
        invocation: CompiledFunctionInvocation,
        artifacts: FunctionChainInvocationArtifacts,
    ) -> "InvocationDebugTrace":
        return InactiveInvocationDebugTrace()


@dataclass(frozen=True, slots=True)
class ActiveInvocationDebugGate(InvocationDebugGate):
    """Debug gate backed by a concrete invocation cursor."""

    gate_key = "active"

    sink: DebugEventSink
    cursor: DebugCursor

    def should_skip(self) -> bool:
        return self.sink.should_skip_invocation(self.cursor)

    def trace(
        self,
        request: FunctionChainExecutionRequest,
        invocation: CompiledFunctionInvocation,
        artifacts: FunctionChainInvocationArtifacts,
    ) -> "InvocationDebugTrace":
        return ActiveInvocationDebugTrace(
            sink=self.sink,
            cursor=self.cursor,
            step_name=request.execution_plan.step_name,
            callable_name=invocation.key.function_name,
            axis_id=request.execution_plan.axis_id,
            invocation_parameters=DebugInvocationParameter.from_kwargs(
                invocation.kwargs_dict
            ),
            input_debug_refs=DebugArtifactRefProjection.from_artifact_plans(
                artifact_plans=artifacts.inputs,
                cursor=self.cursor,
            ),
            output_debug_refs=DebugArtifactRefProjection.from_artifact_plans(
                artifact_plans=artifacts.outputs,
                cursor=self.cursor,
            ),
        )


class InvocationDebugTrace(ABC, metaclass=AutoRegisterMeta):
    """Debug event hooks for one non-skipped invocation."""

    __registry_key__ = "trace_key"
    __skip_if_no_key__ = True

    trace_key: ClassVar[str | None] = None

    @abstractmethod
    def record_before(self) -> None:
        """Record pre-invocation debug state."""

    @abstractmethod
    def record_exception(self, exc: Exception) -> None:
        """Record invocation exception state."""

    @abstractmethod
    def record_after(self, invocation_seconds: float) -> bool:
        """Record post-invocation state and return whether execution should stop."""


class InactiveInvocationDebugTrace(InvocationDebugTrace):
    """No-op invocation trace when debug events are disabled."""

    trace_key = "inactive"

    def record_before(self) -> None:
        return None

    def record_exception(self, exc: Exception) -> None:
        del exc

    def record_after(self, invocation_seconds: float) -> bool:
        del invocation_seconds
        return False


@dataclass(frozen=True, slots=True)
class ActiveInvocationDebugTrace(InvocationDebugTrace):
    """Invocation debug trace that emits before, after, and exception events."""

    trace_key = "active"

    sink: DebugEventSink
    cursor: DebugCursor
    step_name: str
    callable_name: str
    axis_id: str
    invocation_parameters: tuple[DebugInvocationParameter, ...]
    input_debug_refs: DebugArtifactRefProjection
    output_debug_refs: DebugArtifactRefProjection

    def record_before(self) -> None:
        self.sink.record(
            DebugEvent(
                event_type=DebugEventType.BEFORE_INVOCATION,
                cursor=self.cursor,
                step_name=self.step_name,
                callable_name=self.callable_name,
                axis_id=self.axis_id,
                input_artifact_refs=self.input_debug_refs.refs,
                measurement_refs=self.input_debug_refs.measurement_refs,
                relationship_refs=self.input_debug_refs.relationship_refs,
                invocation_parameters=self.invocation_parameters,
            )
        )

    def record_exception(self, exc: Exception) -> None:
        self.sink.record(
            DebugEvent.for_exception(
                cursor=self.cursor,
                step_name=self.step_name,
                callable_name=self.callable_name,
                axis_id=self.axis_id,
                exception=exc,
                input_artifact_refs=self.input_debug_refs.refs,
                output_artifact_refs=self.output_debug_refs.refs,
                measurement_refs=self.output_debug_refs.measurement_refs,
                relationship_refs=self.output_debug_refs.relationship_refs,
                invocation_parameters=self.invocation_parameters,
            )
        )

    def record_after(self, invocation_seconds: float) -> bool:
        event = DebugEvent(
            event_type=DebugEventType.AFTER_INVOCATION,
            cursor=self.cursor,
            step_name=self.step_name,
            callable_name=self.callable_name,
            axis_id=self.axis_id,
            timing_seconds=invocation_seconds,
            input_artifact_refs=self.input_debug_refs.refs,
            output_artifact_refs=self.output_debug_refs.refs,
            measurement_refs=self.output_debug_refs.measurement_refs,
            relationship_refs=self.output_debug_refs.relationship_refs,
            invocation_parameters=self.invocation_parameters,
        )
        self.sink.record(event)
        return self.sink.should_stop_after_invocation(event)


@dataclass(frozen=True, slots=True)
class FunctionChainInvocationResult:
    """Result of one invocation attempt in a function chain."""

    current_stack: RuntimeArrayData
    current_memory_type: str
    stop_chain: bool = False


@dataclass(slots=True)
class FunctionChainInvocationExecutor:
    """Execute one compiled invocation inside a function chain."""

    request: FunctionChainExecutionRequest
    invocation: CompiledFunctionInvocation
    current_stack: RuntimeArrayData
    current_memory_type: str
    debug_sink: DebugEventSink

    @property
    def plan(self) -> FunctionStepExecutionPlan:
        return self.request.execution_plan

    def execute(self) -> FunctionChainInvocationResult:
        debug_gate = InvocationDebugGate.for_invocation(
            self.debug_sink,
            self.request,
            self.invocation,
        )
        if debug_gate.should_skip():
            return FunctionChainInvocationResult(
                self.current_stack,
                self.current_memory_type,
            )

        memory_types = FunctionChainInvocationMemoryTypes.from_invocation(
            self.invocation
        )
        artifacts = FunctionChainInvocationArtifacts.from_request(
            self.request,
            self.invocation,
        )
        debug_trace = debug_gate.trace(self.request, self.invocation, artifacts)
        converted_stack = MainFlowMemoryConversion(
            payload=self.current_stack,
            source_type=self.current_memory_type,
            target_type=memory_types.input_type,
            gpu_id=self.plan.device_id,
        ).converted_payload()
        invocation_started_at = time.perf_counter()
        debug_trace.record_before()
        try:
            output_stack = self.execute_core(converted_stack, artifacts)
        except Exception as exc:
            debug_trace.record_exception(exc)
            raise
        invocation_seconds = time.perf_counter() - invocation_started_at
        stop_chain = debug_trace.record_after(invocation_seconds)
        self.record_profile(invocation_seconds)
        return FunctionChainInvocationResult(
            output_stack,
            memory_types.output_type,
            stop_chain,
        )

    def execute_core(
        self,
        converted_stack: RuntimeArrayData,
        artifacts: FunctionChainInvocationArtifacts,
    ) -> RuntimeArrayData:
        identity = FunctionChainInvocationIdentity.from_request(
            self.request,
            self.invocation,
        )
        return FunctionCoreExecutor(
            FunctionExecutionRequest(
                func_callable=FunctionInvocationCallableResolver.resolve(
                    self.invocation
                ),
                main_data_arg=converted_stack,
                base_kwargs=self.invocation.kwargs_dict,
                context=self.request.context,
                artifact_inputs=artifacts.inputs,
                artifact_outputs=artifacts.outputs,
                runtime_adapter=self.invocation.contract.runtime_adapter,
                invocation_options=self.invocation.invocation_options,
                source_binding_plan=self.plan.source_binding_plan,
                source_binding_context=self.request.source_binding_context,
                group_key=identity.group_key,
                axis_component=self.plan.group_by_value,
                axis_component_value=identity.axis_component_value,
                plane_projection=identity.plane_projection,
            )
        ).execute()

    def record_profile(self, invocation_seconds: float) -> None:
        RuntimeProfileSink.record(
            "invocation_total",
            invocation_seconds,
            InvocationRuntimeProfileFieldSet(
                self.invocation.key.function_name,
                self.invocation.key.group_key,
                self.invocation.key.position,
            ),
        )


@dataclass(slots=True)
class FunctionChainExecutor:
    """Execute compiled invocations over one image stack."""

    request: FunctionChainExecutionRequest
    current_stack: RuntimeArrayData = field(init=False)
    current_memory_type: str = field(init=False)
    debug_sink: DebugEventSink = field(init=False)

    def __post_init__(self) -> None:
        self.current_stack = self.request.initial_data_stack
        self.current_memory_type = self.request.execution_plan.input_memory_type
        self.debug_sink = debug_event_sink_from_context(self.request.context)

    def execute(self) -> RuntimeArrayData:
        for invocation in self.request.invocations:
            result = FunctionChainInvocationExecutor(
                request=self.request,
                invocation=invocation,
                current_stack=self.current_stack,
                current_memory_type=self.current_memory_type,
                debug_sink=self.debug_sink,
            ).execute()
            self.current_stack = result.current_stack
            self.current_memory_type = result.current_memory_type
            if result.stop_chain:
                break
        return self.current_stack


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


def _stack_payload_context(
    raw_slices: Sequence[RuntimeArrayData],
    stack: RuntimeArrayData,
) -> RuntimeArrayData:
    """Attach per-slice image context to a freshly loaded stack."""
    metadata = ImagePayloadMetadataCompositionRequest(raw_slices).metadata()
    mask = _stack_payload_mask(raw_slices)
    return ImagePayloadContextApplication(stack, mask, metadata).payload()


def _stack_payload_mask(raw_slices: Sequence[RuntimeArrayData]) -> RuntimeArrayData | None:
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


def _unstack_payload_context(
    payload: RuntimeArrayData,
    slices: Sequence[RuntimeArrayData],
) -> list[RuntimeArrayData]:
    """Attach per-slice image context after unstacking a runtime stack."""
    mask = image_payload_mask(payload)
    metadata = image_payload_metadata(payload)
    if mask is None and not metadata.has_values:
        return list(slices)
    mask_selector = PayloadMaskSliceSelector(mask=mask, slice_count=len(slices))
    return [
        ImagePayloadContextApplication(
            slice_data,
            mask_selector.for_slice(index),
            metadata.for_channel(index),
        ).payload()
        for index, slice_data in enumerate(slices)
    ]


@dataclass(frozen=True, slots=True)
class ImagePayloadContextApplication:
    """Attach optional image mask and metadata without keyword-bound field bags."""

    data: RuntimeArrayData
    mask: RuntimeArrayData | None
    metadata: ImagePayloadMetadata

    def payload(self) -> RuntimeArrayData:
        if self.mask is not None:
            return MaskedImagePayload(self.data, self.mask, self.metadata)
        if self.metadata.has_values:
            return ImageMetadataPayload(self.data, self.metadata)
        return self.data


@dataclass(frozen=True, slots=True)
class PayloadMaskSliceSelector:
    """Resolve optional stack masks for unstacked payload slices."""

    mask: RuntimeArrayData | None
    slice_count: int

    def for_slice(self, index: int) -> RuntimeArrayData | None:
        if self.mask is None:
            return None
        return _payload_mask_slice(self.mask, index, slice_count=self.slice_count)


def _payload_mask_slice(
    mask: RuntimeArrayData,
    index: int,
    *,
    slice_count: int,
) -> RuntimeArrayData:
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
            RuntimeProfileSink.record(
                "pattern_load_stack",
                time.perf_counter() - load_started_at,
                PatternRuntimeProfileFieldSet.from_plan(self.plan, self.pattern_repr),
            )
            execute_started_at = time.perf_counter()
            processed_stack = self._execute_pattern(loaded)
            RuntimeProfileSink.record(
                "pattern_execute_chain",
                time.perf_counter() - execute_started_at,
                PatternRuntimeProfileFieldSet.from_plan(self.plan, self.pattern_repr),
            )
            unstack_started_at = time.perf_counter()
            output_slices = self._validate_and_unstack(processed_stack, loaded)
            RuntimeProfileSink.record(
                "pattern_validate_unstack",
                time.perf_counter() - unstack_started_at,
                PatternRuntimeProfileFieldSet.from_plan(self.plan, self.pattern_repr),
            )
            save_started_at = time.perf_counter()
            self._save_outputs(output_slices, loaded.matching_files)
            RuntimeProfileSink.record(
                "pattern_save_outputs",
                time.perf_counter() - save_started_at,
                PatternRuntimeProfileFieldSet.from_plan(self.plan, self.pattern_repr),
            )
            cleanup_started_at = time.perf_counter()
            self._cleanup_collapsed_domains(output_slices, loaded.matching_files)
            RuntimeProfileSink.record(
                "pattern_cleanup",
                time.perf_counter() - cleanup_started_at,
                PatternRuntimeProfileFieldSet.from_plan(self.plan, self.pattern_repr),
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
            VariableComponentNames(self.plan.variable_components).value,
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
        raw_slices: Sequence[RuntimeArrayData],
        matching_files: Sequence[str],
        full_file_paths: Sequence[str],
    ) -> list[RuntimeArrayData]:
        source_projection = self._source_schema_workspace_projection()
        if source_projection is None:
            return list(raw_slices)
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
                self.context.filemanager,
            ).apply(payload)
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
        source_backend = Backend(
            self.context.microscope_handler.get_primary_backend(
                self.context.input_dir,
                self.context.filemanager,
            )
        )
        source_projection = self._source_schema_workspace_projection()
        step_input_universe_request = SourceUniverseRequest(
            scope=SourceUniverseScope.STEP_INPUT,
            context=self.context,
            plan=self.plan,
            matching_files=tuple(matching_files),
            source_backend=source_backend,
            source_projection=source_projection,
        )
        pipeline_start_universe_request = SourceUniverseRequest(
            scope=SourceUniverseScope.PIPELINE_START,
            context=self.context,
            plan=self.plan,
            matching_files=tuple(matching_files),
            source_backend=source_backend,
            source_projection=source_projection,
        )
        step_input_universe = SourceUniverseStrategy.universe(
            step_input_universe_request
        )
        pipeline_source_universe = SourceUniverseStrategy.universe(
            pipeline_start_universe_request
        )
        return SourceBindingRuntimeContext(
            step_input_files=step_input_universe.files,
            current_step_input_files=tuple(matching_files),
            step_input_dir=str(self.plan.input_dir),
            step_input_backend=self.plan.read_backend,
            step_input_source_paths=step_input_universe_request.step_input_source_paths,
            source_metadata_by_path=step_input_universe_request.source_metadata_by_path,
            pipeline_input_files=pipeline_source_universe.files,
            pipeline_input_backend=pipeline_source_universe.backend.value,
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
        if not OpenHCSMetadataSubdirectories(metadata).has_workspace_mapping():
            return None
        return self._virtual_workspace_source_projection_from_metadata(metadata)

    def _virtual_workspace_source_projection(self) -> VirtualWorkspaceSourceProjection:
        """Return cached virtual-workspace source-binding projection for this plate."""
        return self._virtual_workspace_source_projection_from_metadata(
            self._openhcs_metadata_dict()
        )

    def _virtual_workspace_source_projection_from_metadata(
        self,
        metadata: OpenHCSMetadataPayload,
    ) -> VirtualWorkspaceSourceProjection:
        """Return cached source-schema projection for this plate metadata."""
        plate_path = str(Path(self.context.plate_path))
        cache = self.source_binding_execution_cache()
        projection = cache.virtual_workspace_projections.get(plate_path)
        if projection is not None:
            return projection

        builder = VirtualWorkspaceSourceProjectionBuilder(Path(self.context.plate_path))
        for subdirectory in OpenHCSMetadataSubdirectories(metadata).values():
            builder.ingest_subdirectory(subdirectory)
        projection = builder.projection()
        cache.virtual_workspace_projections[plate_path] = projection
        return projection

    def _openhcs_metadata_dict(self) -> OpenHCSMetadataPayload:
        from openhcs.microscopes.openhcs import OpenHCSMetadataHandler

        metadata_handler = self.context.microscope_handler.metadata_handler
        if not isinstance(metadata_handler, OpenHCSMetadataHandler):
            metadata_handler = OpenHCSMetadataHandler(self.context.filemanager)
        return metadata_handler._load_metadata_dict(self.context.plate_path)

    def _component_artifact_plans(self) -> ComponentArtifactPlans:
        request = self.request
        component_key = ComponentValueString.from_value(request.component_value).value
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
    ) -> RuntimeArrayData:
        request = self.request
        component_artifacts = self._component_artifact_plans()

        if not request.compiled_group.invocations:
            raise ValueError(
                f"Compiled function group {request.compiled_group.group_key} has no invocations."
            )

        return FunctionChainExecutor(
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
        ).execute()

    def _validate_and_unstack(
        self,
        processed_stack: RuntimeArrayData,
        loaded: PatternGroupData,
    ) -> list[RuntimeArrayData]:
        processed_data = image_payload_data(processed_stack)
        try:
            output_slices = SourceSliceUnstackRequest(
                processed_data,
                loaded.source_slice_shapes,
                self.plan.output_memory_type,
                self.plan.device_id,
            ).slices()
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

        return _unstack_payload_context(processed_stack, output_slices)

    def _save_outputs(
        self,
        output_slices: list[RuntimeArrayData],
        matching_files: list[str],
    ) -> None:
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
        output_slices: list[RuntimeArrayData],
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
