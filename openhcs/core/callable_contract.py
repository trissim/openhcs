"""Typed callable contracts used by compiler phases.

This module centralizes metadata extraction from processing callables so the
compiler has one source of truth for memory and artifact declarations.
"""

from __future__ import annotations

import importlib
import inspect
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, MutableMapping
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from enum import Enum
from functools import lru_cache, wraps
from threading import Lock
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, get_type_hints

from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifact_key_selection import ArtifactPlanKeySelector
from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.module_artifact_contract import (
    MODULE_ARTIFACT_CONTRACT_ATTR,
    ModuleArtifactContract,
    module_artifact_contract_from_namespace,
)
from openhcs.core.runtime_adapters import (
    RuntimeAdapterSpec,
    runtime_adapter_spec_from_callable,
)
from openhcs.core.runtime_batch_contracts import RuntimeBatchCallableFamily


ArtifactSpecItems = tuple[tuple[str, ArtifactSpec], ...]
CallableNamespace = Mapping[str, Any]
PROCESSING_CONTRACT_ATTR = "__processing_contract__"
DECLARED_PROCESSING_CONTRACT_ATTR = "__openhcs_declared_processing_contract__"
RAW_PROCESSING_FUNCTION_ATTR = "__openhcs_raw_processing_function__"
PROCESSING_PREPARE_ATTR = "__openhcs_prepare__"
RUNTIME_IMAGE_EXECUTION_MODE_ATTR = "__openhcs_runtime_image_execution_mode__"
CALLABLE_REQUEST_BINDING_ATTR = "__openhcs_callable_request_binding__"
_PREPARED_CALLABLE_KEYS: set[tuple[str, str, int]] = set()
_PREPARED_CALLABLE_LOCK = Lock()

CallableContractCacheKey = tuple[Hashable | None, ...]
CallableRuntimeCacheKey = int | tuple[str, CallableContractCacheKey]


class CompilerPreparedAutoRegisterFamily(ABC):
    """AutoRegisterMeta family that can prepare runtime compiler substrates."""

    @classmethod
    @abstractmethod
    def prepare_registered_family(cls) -> None:
        """Prepare registered implementations before timed callable execution."""


@dataclass(frozen=True, slots=True)
class CallableMetadata:
    """Compiler-visible metadata declared by one processing callable."""

    input_memory_type: str | None = None
    output_memory_type: str | None = None
    artifact_inputs: ArtifactSpecItems = ()
    artifact_outputs: ArtifactSpecItems = ()
    runtime_adapter: RuntimeAdapterSpec | None = None
    processing_contract: Enum | None = None
    declared_processing_contract: str | None = None
    module_artifact_contract: ModuleArtifactContract | None = None
    raw_processing_function: Callable[..., object] | "FunctionReference" | None = None
    runtime_image_execution_mode: ImagePayloadExecutionMode | None = None
    request_binding: "CallableRequestBinding | None" = None
    prepare: Callable[..., object] | None = None

    @classmethod
    def from_callable(cls, func: Callable[..., object]) -> "CallableMetadata":
        """Build metadata from a callable or compiler function reference."""
        return cls.from_projection(CallableProjection.from_callable(func))

    @classmethod
    def from_projection(
        cls,
        projection: "CallableProjection",
    ) -> "CallableMetadata":
        """Build metadata from an already resolved callable projection."""
        namespace = projection.namespace
        reader = CallableMetadataReader(namespace, projection.name)
        return cls(
            input_memory_type=reader.optional_string("input_memory_type"),
            output_memory_type=reader.optional_string("output_memory_type"),
            artifact_inputs=_artifact_spec_items(
                namespace,
                projection.name,
                "__artifact_inputs__",
            ),
            artifact_outputs=_artifact_spec_items(
                namespace,
                projection.name,
                "__artifact_outputs__",
            ),
            runtime_adapter=runtime_adapter_spec_from_callable(projection.func),
            processing_contract=reader.optional_enum(PROCESSING_CONTRACT_ATTR),
            declared_processing_contract=reader.optional_string(
                DECLARED_PROCESSING_CONTRACT_ATTR,
            ),
            module_artifact_contract=module_artifact_contract_from_namespace(
                namespace,
                owner_name=projection.name,
            ),
            raw_processing_function=reader.optional_raw_processing_function(
                RAW_PROCESSING_FUNCTION_ATTR,
            ),
            runtime_image_execution_mode=reader.optional_execution_mode(
                RUNTIME_IMAGE_EXECUTION_MODE_ATTR,
            ),
            request_binding=reader.optional_request_binding(
                CALLABLE_REQUEST_BINDING_ATTR,
            ),
            prepare=reader.optional_callable(PROCESSING_PREPARE_ATTR),
        )

    def without_prepare(self) -> "CallableMetadata":
        """Return this metadata without a process-local prepare hook."""
        return dataclasses.replace(self, prepare=None)

    def with_raw_processing_function(
        self,
        raw_processing_function: Callable[..., object] | "FunctionReference" | None,
    ) -> "CallableMetadata":
        """Return this metadata with a normalized raw processing callable."""
        return dataclasses.replace(
            self,
            raw_processing_function=raw_processing_function,
        )

    def as_namespace(self) -> dict[str, object]:
        """Project typed metadata into callable declaration keys."""
        namespace: dict[str, object] = {}
        if self.input_memory_type is not None:
            namespace["input_memory_type"] = self.input_memory_type
        if self.output_memory_type is not None:
            namespace["output_memory_type"] = self.output_memory_type
        if self.artifact_inputs:
            namespace["__artifact_inputs__"] = dict(self.artifact_inputs)
        if self.artifact_outputs:
            namespace["__artifact_outputs__"] = dict(self.artifact_outputs)
        if self.runtime_adapter is not None:
            namespace["__runtime_adapter__"] = self.runtime_adapter
        if self.processing_contract is not None:
            namespace[PROCESSING_CONTRACT_ATTR] = self.processing_contract
        if self.declared_processing_contract is not None:
            namespace[DECLARED_PROCESSING_CONTRACT_ATTR] = (
                self.declared_processing_contract
            )
        if self.module_artifact_contract is not None:
            namespace[MODULE_ARTIFACT_CONTRACT_ATTR] = self.module_artifact_contract
        if self.raw_processing_function is not None:
            namespace[RAW_PROCESSING_FUNCTION_ATTR] = self.raw_processing_function
        if self.runtime_image_execution_mode is not None:
            namespace[RUNTIME_IMAGE_EXECUTION_MODE_ATTR] = (
                self.runtime_image_execution_mode
            )
        if self.request_binding is not None:
            namespace[CALLABLE_REQUEST_BINDING_ATTR] = self.request_binding
        if self.prepare is not None:
            namespace[PROCESSING_PREPARE_ATTR] = self.prepare
        return namespace


@dataclass(frozen=True, slots=True)
class CallableContract(ArtifactPlanKeySelector):
    """Compiler contract declared by one processing callable."""

    func: Callable[..., object] | "FunctionReference"
    function_name: str
    module_name: str | None
    metadata: CallableMetadata = dataclasses.field(default_factory=CallableMetadata)
    runtime_batch_executors: Mapping[object, object] | None = None

    def __reduce__(
        self,
    ) -> tuple[Callable[..., "CallableContract"], tuple[Any, ...]]:
        """Serialize immutable mapping-backed metadata across worker queues."""
        return (
            _rebuild_callable_contract,
            (
                self.func,
                self.function_name,
                self.module_name,
                self.metadata,
                (
                    dict(self.runtime_batch_executors)
                    if self.runtime_batch_executors is not None
                    else None
                ),
            ),
        )

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., object] | "FunctionReference",
    ) -> "CallableContract":
        """Build a contract from callable attributes once at compiler boundary."""
        projection = CallableProjection.from_callable(func)
        metadata = CallableMetadata.from_projection(projection)
        batch_raw_processing_function = (
            metadata.raw_processing_function
            if callable(metadata.raw_processing_function)
            else None
        )
        return cls(
            func=func,
            function_name=projection.name,
            module_name=projection.module_name,
            metadata=metadata,
            runtime_batch_executors=RuntimeBatchCallableFamily(
                func=func,
                raw_processing_function=batch_raw_processing_function,
            ).executors(),
        )

    @property
    def input_memory_type(self) -> str | None:
        """Declared input memory type."""
        return self.metadata.input_memory_type

    @property
    def output_memory_type(self) -> str | None:
        """Declared output memory type."""
        return self.metadata.output_memory_type

    @property
    def artifact_inputs(self) -> ArtifactSpecItems:
        """Declared artifact inputs."""
        return self.metadata.artifact_inputs

    @property
    def artifact_outputs(self) -> ArtifactSpecItems:
        """Declared artifact outputs."""
        return self.metadata.artifact_outputs

    @property
    def runtime_adapter(self) -> RuntimeAdapterSpec | None:
        """Declared runtime adapter."""
        return self.metadata.runtime_adapter

    @property
    def processing_contract(self) -> Enum | None:
        """Declared nominal processing contract."""
        return self.metadata.processing_contract

    @property
    def declared_processing_contract(self) -> str | None:
        """Declared processing contract name."""
        return self.metadata.declared_processing_contract

    @property
    def module_artifact_contract(self) -> ModuleArtifactContract | None:
        """Declared module artifact contract."""
        return self.metadata.module_artifact_contract

    @property
    def raw_processing_function(
        self,
    ) -> Callable[..., object] | "FunctionReference" | None:
        """Declared raw processing callable or transport reference."""
        return self.metadata.raw_processing_function

    @property
    def runtime_image_execution_mode(self) -> ImagePayloadExecutionMode | None:
        """Declared runtime image execution mode."""
        return self.metadata.runtime_image_execution_mode

    @property
    def request_binding(self) -> "CallableRequestBinding | None":
        """Declared callable request binding."""
        return self.metadata.request_binding

    artifact_input_names: ClassVar[AliasProperty[tuple[str, ...]]] = (
        AliasProperty("input_names")
    )

    @property
    def input_names(self) -> tuple[str, ...]:
        """Declared artifact input names in declaration order."""
        return tuple(name for name, _ in self.artifact_inputs)

    artifact_output_names: ClassVar[AliasProperty[tuple[str, ...]]] = (
        AliasProperty("output_names")
    )

    @property
    def output_names(self) -> tuple[str, ...]:
        """Declared artifact output names in declaration order."""
        return tuple(name for name, _ in self.artifact_outputs)

    @property
    def artifact_inputs_dict(self) -> dict[str, ArtifactSpec]:
        """Return declared artifact inputs as a runtime mapping."""
        return dict(self.artifact_inputs)

    @property
    def artifact_outputs_dict(self) -> dict[str, ArtifactSpec]:
        """Return declared artifact outputs as a runtime mapping."""
        return dict(self.artifact_outputs)

    def runtime_batch_executor(
        self,
        domain: Any,
    ) -> Any | None:
        """Return the declared runtime batch executor for one domain."""
        if self.runtime_batch_executors is None:
            return None
        return self.runtime_batch_executors.get(domain)

    def contract_cache_identity(self) -> CallableContractCacheKey:
        """Return the contract dimensions that affect reference rehydration."""
        return (
            self.module_artifact_contract,
            self.declared_processing_contract,
            self.processing_contract,
            self.runtime_adapter,
            self.runtime_image_execution_mode,
        )

    def runtime_callable_cache_identity(
        self,
    ) -> CallableRuntimeCacheKey:
        """Return the process-local cache identity for this callable contract."""
        if _is_function_reference(self.func):
            return (self.func.composite_key, self.contract_cache_identity())
        if callable(self.func):
            return id(self.func)
        raise TypeError(f"Invalid callable contract function: {self.func}")

    def resolve_runtime_callable(self) -> Callable[..., object]:
        """Return the executable callable for this contract."""
        if _is_function_reference(self.func):
            from openhcs.core.function_reference_rehydration import (
                FunctionReferenceRehydrationRequest,
                FunctionReferenceRehydrator,
            )

            return FunctionReferenceRehydrator.rehydrate_reference(
                FunctionReferenceRehydrationRequest(
                    reference=self.func,
                    contract=self,
                    resolved_callable=self.func.resolve(),
                )
            )
        if callable(self.func):
            return self.func
        raise TypeError(f"Invalid callable contract function: {self.func}")


def _rebuild_callable_contract(
    func: Callable[..., object] | "FunctionReference",
    function_name: str,
    module_name: str | None,
    metadata: CallableMetadata,
    runtime_batch_executors: Mapping[object, object] | None,
) -> CallableContract:
    """Rebuild a CallableContract with immutable executor metadata."""
    immutable_executors = (
        None
        if runtime_batch_executors is None
        else MappingProxyType(dict(runtime_batch_executors))
    )
    return CallableContract(
        func=func,
        function_name=function_name,
        module_name=module_name,
        metadata=metadata,
        runtime_batch_executors=immutable_executors,
    )


@dataclass(frozen=True, slots=True)
class CallableRequestBinding:
    """Typed declaration for public kwargs projected into a request record."""

    request_type: type[object]
    request_parameter: str
    public_fields: tuple[str, ...]
    public_defaults: tuple[tuple[str, Any], ...] = ()
    public_annotations: tuple[tuple[str, Any], ...] = ()

    @property
    def public_defaults_dict(self) -> dict[str, Any]:
        """Return request-field defaults as a runtime mapping."""
        return dict(self.public_defaults)

    @property
    def public_annotations_dict(self) -> dict[str, Any]:
        """Return request-field annotations as a runtime mapping."""
        return dict(self.public_annotations)

    @classmethod
    def from_dataclass(
        cls,
        request_type: type[object],
        *,
        request_parameter: str,
        public_fields: tuple[str, ...],
        public_defaults: Mapping[str, Any],
    ) -> "CallableRequestBinding":
        """Build a binding declaration from a request dataclass."""
        _validate_request_public_fields(request_type, public_fields)
        return cls(
            request_type=request_type,
            request_parameter=request_parameter,
            public_fields=public_fields,
            public_defaults=tuple(public_defaults.items()),
            public_annotations=tuple(get_type_hints(request_type).items()),
        )

    def request_from_bound_arguments(
        self,
        bound_arguments: Mapping[str, Any],
    ) -> object:
        """Build the request object from public bound call arguments."""
        defaults = self.public_defaults_dict
        request_kwargs: dict[str, Any] = {}
        missing: list[str] = []
        for field_name in self.public_fields:
            if field_name in bound_arguments:
                request_kwargs[field_name] = bound_arguments[field_name]
            elif field_name in defaults:
                request_kwargs[field_name] = defaults[field_name]
            else:
                missing.append(field_name)
        if missing:
            raise TypeError(
                f"{self.request_type.__name__} request is missing public "
                f"field(s): {', '.join(missing)}."
            )
        return self.request_type(**request_kwargs)

    def implementation_kwargs(
        self,
        bound_arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return kwargs for the implementation callable."""
        request_fields = set(self.public_fields)
        local_kwargs = {
            name: value
            for name, value in bound_arguments.items()
            if name not in request_fields
        }
        local_kwargs[self.request_parameter] = self.request_from_bound_arguments(
            bound_arguments,
        )
        return local_kwargs


@dataclass(frozen=True, slots=True)
class RequestParameterName:
    """Validated implementation parameter name for callable request binding."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("request_parameter must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class RequestPublicFieldSelection:
    """Validated public request field selection."""

    request_type: type[object]
    field_names: tuple[str, ...] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        if self.field_names is None:
            return tuple(field.name for field in fields(self.request_type))
        return tuple(self.field_names)


def callable_request(
    request_type: type[object],
    *,
    request_parameter: str = "request",
    public_fields: tuple[str, ...] | None = None,
    public_defaults: Mapping[str, Any] | object | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Expose request-record fields as the public callable signature."""
    if not is_dataclass(request_type):
        raise TypeError(
            "callable_request request_type must be a dataclass type, "
            f"got {request_type!r}."
        )
    binding = CallableRequestBinding.from_dataclass(
        request_type=request_type,
        request_parameter=RequestParameterName(request_parameter).value,
        public_fields=RequestPublicFieldSelection(
            request_type=request_type,
            field_names=public_fields,
        ).names,
        public_defaults=_public_defaults_mapping(public_defaults),
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        implementation_signature = inspect.signature(func)
        if request_parameter not in implementation_signature.parameters:
            raise ValueError(
                f"{func.__name__} must declare request parameter "
                f"{request_parameter!r}."
            )
        public_signature = _request_public_signature(
            func,
            binding,
            implementation_signature,
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = public_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return func(**binding.implementation_kwargs(bound.arguments))

        wrapper_namespace = _mutable_callable_namespace(wrapper)
        wrapper_namespace[CALLABLE_REQUEST_BINDING_ATTR] = binding
        wrapper_namespace["__signature__"] = public_signature
        return wrapper

    return decorator


def _public_defaults_mapping(
    public_defaults: Mapping[str, Any] | object | None,
) -> Mapping[str, Any]:
    """Return request public defaults as a mapping."""
    if public_defaults is None:
        return {}
    if isinstance(public_defaults, Mapping):
        return public_defaults
    if is_dataclass(public_defaults) and not isinstance(public_defaults, type):
        return asdict(public_defaults)
    raise TypeError(
        "public_defaults must be a mapping, dataclass instance, or None; "
        f"got {type(public_defaults).__name__}."
    )


def attach_callable_contract_metadata(
    func: Any,
    *,
    declared_processing_contract: str | None = None,
    raw_processing_function: Any | None = None,
    prepare: Any | None = None,
    runtime_image_execution_mode: ImagePayloadExecutionMode | None = None,
) -> None:
    """Attach OpenHCS callable metadata used by compiler/runtime phases."""
    if declared_processing_contract is not None:
        if (
            not isinstance(declared_processing_contract, str)
            or not declared_processing_contract.strip()
        ):
            raise ValueError(
                "declared_processing_contract must be a non-empty string."
            )
        namespace = _mutable_callable_namespace(func)
        namespace[DECLARED_PROCESSING_CONTRACT_ATTR] = declared_processing_contract
        _attach_nominal_processing_contract_if_supported(
            func,
            declared_processing_contract,
        )
    if raw_processing_function is not None:
        if not callable(raw_processing_function):
            raise TypeError(
                "raw_processing_function must be callable, "
                f"got {type(raw_processing_function).__name__}."
            )
        namespace = _mutable_callable_namespace(func)
        namespace[RAW_PROCESSING_FUNCTION_ATTR] = raw_processing_function
        raw_prepare = CallableMetadata.from_callable(raw_processing_function).prepare
        if raw_prepare is not None and PROCESSING_PREPARE_ATTR not in namespace:
            if not callable(raw_prepare):
                raise TypeError(
                    "raw_processing_function prepare hook must be callable, "
                    f"got {type(raw_prepare).__name__}."
                )
            namespace[PROCESSING_PREPARE_ATTR] = raw_prepare
    if prepare is not None:
        if not callable(prepare):
            raise TypeError(
                "prepare must be callable, "
                f"got {type(prepare).__name__}."
            )
        _mutable_callable_namespace(func)[PROCESSING_PREPARE_ATTR] = prepare
    if runtime_image_execution_mode is not None:
        if not isinstance(runtime_image_execution_mode, ImagePayloadExecutionMode):
            raise TypeError(
                "runtime_image_execution_mode must be ImagePayloadExecutionMode, "
                f"got {type(runtime_image_execution_mode).__name__}."
            )
        _mutable_callable_namespace(func)[RUNTIME_IMAGE_EXECUTION_MODE_ATTR] = (
            runtime_image_execution_mode
        )


def _attach_nominal_processing_contract_if_supported(
    func: Any,
    declared_processing_contract: str,
) -> None:
    """Coerce declared contract names to nominal metadata at the declaration boundary."""
    namespace = _mutable_callable_namespace(func)
    if PROCESSING_CONTRACT_ATTR in namespace:
        return

    from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

    contract = ProcessingContract.from_declared_name(declared_processing_contract)
    if contract is not None:
        namespace[PROCESSING_CONTRACT_ATTR] = contract


def processing_prepare(*targets: Any) -> Any:
    """Declare a preparation callable for one or more processing callables.

    This keeps preparation binding explicit and colocated with the prepare
    function definition instead of relying on tail-end attribute assignment.
    """
    if not targets:
        raise ValueError("processing_prepare requires at least one target callable.")
    for target in targets:
        if not callable(target):
            raise TypeError(
                "processing_prepare targets must be callable, "
                f"got {type(target).__name__}."
            )

    def decorator(prepare: Any) -> Any:
        if not callable(prepare):
            raise TypeError(
                "processing_prepare can only decorate callables, "
                f"got {type(prepare).__name__}."
            )
        for target in targets:
            attach_processing_prepare(target, prepare)
        return prepare

    return decorator


def attach_processing_prepare(func: Any, prepare: Any) -> None:
    """Attach preparation metadata across a decorated callable family."""
    if not callable(func):
        raise TypeError(
            "attach_processing_prepare target must be callable, "
            f"got {type(func).__name__}."
        )
    if not callable(prepare):
        raise TypeError(
            "attach_processing_prepare prepare must be callable, "
            f"got {type(prepare).__name__}."
        )
    for target in CallableProjection.from_callable(func).prepare_targets():
        _mutable_callable_namespace(target)[PROCESSING_PREPARE_ATTR] = prepare


def runtime_image_execution_mode(
    mode: ImagePayloadExecutionMode,
) -> Any:
    """Declare the image execution mode the compiler should preserve."""
    if not isinstance(mode, ImagePayloadExecutionMode):
        raise TypeError(
            "runtime_image_execution_mode mode must be ImagePayloadExecutionMode, "
            f"got {type(mode).__name__}."
        )

    def decorator(func: Any) -> Any:
        _mutable_callable_namespace(func)[RUNTIME_IMAGE_EXECUTION_MODE_ATTR] = mode
        return func

    return decorator


def prepare_processing_callable(func: Any) -> None:
    """Run an optional callable preparation hook before timed data processing."""
    projection = CallableProjection.from_callable(func)
    if projection.module_name is not None:
        _prepare_module_autoregister_families(projection.module_name)
        _prepare_processing_module(projection.module_name)

    prepare = projection.namespace.get(PROCESSING_PREPARE_ATTR)
    if prepare is None:
        return
    if not callable(prepare):
        raise TypeError(
            f"{projection.name!r}.{PROCESSING_PREPARE_ATTR} must be "
            f"callable, got {type(prepare).__name__}."
        )
    if projection.module_name is None:
        module_label = "<unknown>"
    else:
        module_label = projection.module_name
    prepare_key = (
        "callable",
        f"{module_label}.{projection.name}",
        _prepare_callable_identity(prepare),
    )
    with _PREPARED_CALLABLE_LOCK:
        if prepare_key in _PREPARED_CALLABLE_KEYS:
            return
    prepare()
    with _PREPARED_CALLABLE_LOCK:
        _PREPARED_CALLABLE_KEYS.add(prepare_key)


def _prepare_callable_identity(prepare: Callable[..., Any]) -> tuple[str, str]:
    """Return a stable identity for process-local prepare-hook caching."""
    return str(prepare.__module__), str(prepare.__qualname__)


def _prepare_processing_module(module_name: str) -> None:
    """Run an optional module-level preparation hook exactly once."""
    module = importlib.import_module(module_name)
    prepare = vars(module).get(PROCESSING_PREPARE_ATTR)
    if prepare is None:
        return
    if not callable(prepare):
        raise TypeError(
            f"Module {module_name!r}.{PROCESSING_PREPARE_ATTR} must be callable, "
            f"got {type(prepare).__name__}."
        )
    prepare_key = ("module", module_name, id(prepare))
    with _PREPARED_CALLABLE_LOCK:
        if prepare_key in _PREPARED_CALLABLE_KEYS:
            return
    prepare()
    with _PREPARED_CALLABLE_LOCK:
        _PREPARED_CALLABLE_KEYS.add(prepare_key)


@lru_cache(maxsize=None)
def _prepare_module_autoregister_families(module_name: str) -> None:
    """Prepare AutoRegisterMeta families imported by a callable module."""
    module = importlib.import_module(module_name)
    from openhcs.core.autoregister_preparation import AutoRegisterRegistryPreparation

    AutoRegisterRegistryPreparation.prepare_module_registered_families((module,))


def _is_function_reference(func: Any) -> bool:
    """Return whether func is the compiler's nominal picklable reference."""
    from openhcs.core.function_reference import FunctionReference

    return isinstance(func, FunctionReference)


def _callable_namespace(func: Any) -> CallableNamespace:
    """Return the readable metadata namespace for a callable-like object."""
    if _is_function_reference(func):
        return func.metadata.as_namespace()
    return vars(func)


def _mutable_callable_namespace(func: Any) -> MutableMapping[str, Any]:
    """Return the writable metadata namespace for a Python callable."""
    namespace = vars(func)
    if not isinstance(namespace, MutableMapping):
        raise TypeError(f"{func!r} does not expose a mutable metadata namespace.")
    return namespace


@dataclass(frozen=True, slots=True)
class CallableProjection:
    """Nominal view over callable metadata used at compiler/runtime boundaries."""

    func: Any
    name: str
    module_name: str | None
    namespace: CallableNamespace

    @classmethod
    def from_callable(cls, func: Any) -> "CallableProjection":
        """Project a callable or compiler function reference into stable metadata."""
        if _is_function_reference(func):
            name = func.function_name
            module_name = func.original_module
        else:
            name = func.__name__
            module_name = func.__module__
        namespace = _callable_namespace(func)
        if not isinstance(name, str):
            raise TypeError(
                f"Callable name must be a string, got {type(name).__name__}."
            )
        if module_name is not None and not isinstance(module_name, str):
            raise TypeError(
                f"{name!r}.__module__ must be a string or None, "
                f"got {type(module_name).__name__}."
            )
        return cls(func=func, name=name, module_name=module_name, namespace=namespace)

    def prepare_targets(self) -> tuple[Any, ...]:
        """Return wrapper/raw callables that may carry preparation metadata."""
        targets: list[Any] = []
        seen: set[int] = set()
        pending = [self.func]
        while pending:
            target = pending.pop()
            target_id = id(target)
            if target_id in seen:
                continue
            seen.add(target_id)
            targets.append(target)
            namespace = _callable_namespace(target)
            raw = namespace.get(RAW_PROCESSING_FUNCTION_ATTR)
            if callable(raw):
                pending.append(raw)
            wrapped = namespace.get("__wrapped__")
            if callable(wrapped):
                pending.append(wrapped)
        return tuple(targets)


@dataclass(frozen=True, slots=True)
class CallableMetadataReader:
    """Typed reader for user-declared callable metadata."""

    namespace: CallableNamespace
    function_name: str

    def optional_string(self, field_name: str) -> str | None:
        """Return an optional string metadata field."""
        value = self.namespace.get(field_name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be a string, "
                f"got {type(value).__name__}."
            )
        return value

    def optional_execution_mode(
        self,
        field_name: str,
    ) -> ImagePayloadExecutionMode | None:
        """Return an optional image execution-mode metadata field."""
        value = self.namespace.get(field_name)
        if value is None:
            return None
        if not isinstance(value, ImagePayloadExecutionMode):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be "
                "ImagePayloadExecutionMode, "
                f"got {type(value).__name__}."
            )
        return value

    def optional_enum(self, field_name: str) -> Enum | None:
        """Return an optional enum metadata field."""
        value = self.namespace.get(field_name)
        if value is None:
            return None
        if not isinstance(value, Enum):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be Enum, "
                f"got {type(value).__name__}."
            )
        return value

    def optional_callable(
        self,
        field_name: str,
    ) -> Callable[..., object] | None:
        """Return an optional callable metadata field."""
        value = self.namespace.get(field_name)
        if value is None:
            return None
        if not callable(value):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be callable, "
                f"got {type(value).__name__}."
            )
        return value

    def optional_raw_processing_function(
        self,
        field_name: str,
    ) -> Callable[..., object] | "FunctionReference" | None:
        """Return an optional raw callable or transport reference."""
        value = self.namespace.get(field_name)
        if value is None:
            return None
        if callable(value) or _is_function_reference(value):
            return value
        raise TypeError(
            f"{self.function_name!r}.{field_name} must be callable or "
            f"FunctionReference, got {type(value).__name__}."
        )

    def optional_request_binding(
        self,
        field_name: str,
    ) -> CallableRequestBinding | None:
        """Return an optional callable request-binding declaration."""
        value = self.namespace.get(field_name)
        if value is None:
            return None
        if not isinstance(value, CallableRequestBinding):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be "
                "CallableRequestBinding, "
                f"got {type(value).__name__}."
            )
        return value


def _validate_request_public_fields(
    request_type: type[object],
    public_fields: tuple[str, ...],
) -> None:
    """Validate declared public request fields against the dataclass type."""
    dataclass_field_names = {field.name for field in fields(request_type)}
    missing = [
        field_name
        for field_name in public_fields
        if field_name not in dataclass_field_names
    ]
    if missing:
        raise ValueError(
            f"{request_type.__name__} has no request field(s): "
            f"{', '.join(missing)}."
        )


def _request_public_signature(
    func: Callable[..., Any],
    binding: CallableRequestBinding,
    implementation_signature: inspect.Signature,
) -> inspect.Signature:
    """Return the public signature with request fields expanded."""
    defaults = binding.public_defaults_dict
    annotations = binding.public_annotations_dict
    request_parameters = tuple(
        inspect.Parameter(
            field_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=_request_field_default(binding.request_type, field_name, defaults),
            annotation=annotations.get(field_name, inspect.Parameter.empty),
        )
        for field_name in binding.public_fields
    )
    local_parameters = tuple(
        parameter
        for name, parameter in implementation_signature.parameters.items()
        if name != binding.request_parameter
    )
    return inspect.Signature(
        parameters=(*request_parameters, *local_parameters),
        return_annotation=implementation_signature.return_annotation,
    )


def _request_field_default(
    request_type: type[object],
    field_name: str,
    public_defaults: Mapping[str, Any],
) -> Any:
    """Return the public default for one request field."""
    if field_name in public_defaults:
        return public_defaults[field_name]
    field_by_name = {field.name: field for field in fields(request_type)}
    field = field_by_name[field_name]
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        return field.default_factory()  # type: ignore[misc]
    return inspect.Parameter.empty


def _artifact_spec_items(
    namespace: CallableNamespace,
    function_name: str,
    attr_name: str,
) -> ArtifactSpecItems:
    raw_specs = namespace.get(attr_name)
    if not raw_specs:
        return ()
    if not isinstance(raw_specs, Mapping):
        raise TypeError(
            f"{function_name!r}.{attr_name} must be a mapping, "
            f"got {type(raw_specs).__name__}."
        )

    items: list[tuple[str, ArtifactSpec]] = []
    for name, spec in raw_specs.items():
        if not isinstance(name, str):
            raise TypeError(
                f"{function_name!r}.{attr_name} contains a non-string "
                f"artifact name: {name!r}."
            )
        if not isinstance(spec, ArtifactSpec):
            raise TypeError(
                f"{function_name!r}.{attr_name}['{name}'] "
                f"must be ArtifactSpec, got {type(spec).__name__}."
            )
        if spec.name != name:
            raise ValueError(
                f"{function_name!r}.{attr_name} key '{name}' "
                f"does not match ArtifactSpec.name '{spec.name}'."
            )
        items.append((name, spec))
    return tuple(items)
