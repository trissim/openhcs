"""Typed callable contracts used by compiler phases.

This module centralizes metadata extraction from processing callables so the
compiler has one source of truth for memory and artifact declarations.
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifact_key_selection import ArtifactPlanKeySelector
from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.module_artifact_contract import (
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
_PREPARED_CALLABLE_KEYS: set[tuple[str, str, int]] = set()
_PREPARED_CALLABLE_LOCK = Lock()


class CompilerPreparedAutoRegisterFamily(ABC):
    """AutoRegisterMeta family that can prepare runtime compiler substrates."""

    @classmethod
    @abstractmethod
    def prepare_registered_family(cls) -> None:
        """Prepare registered implementations before timed callable execution."""


@dataclass(frozen=True, slots=True)
class CallableContract(ArtifactPlanKeySelector):
    """Compiler-visible contract declared by one processing callable."""

    func: Any
    function_name: str
    module_name: str | None
    input_memory_type: str | None
    output_memory_type: str | None
    artifact_inputs: ArtifactSpecItems = ()
    artifact_outputs: ArtifactSpecItems = ()
    runtime_adapter: RuntimeAdapterSpec | None = None
    processing_contract: Any | None = None
    declared_processing_contract: str | None = None
    module_artifact_contract: ModuleArtifactContract | None = None
    raw_processing_function: Any | None = None
    runtime_image_execution_mode: ImagePayloadExecutionMode | None = None
    runtime_batch_executors: Mapping[Any, Any] | None = None

    def __post_init__(self) -> None:
        if self.runtime_batch_executors is not None and not isinstance(
            self.runtime_batch_executors,
            MappingProxyType,
        ):
            object.__setattr__(
                self,
                "runtime_batch_executors",
                MappingProxyType(dict(self.runtime_batch_executors)),
            )

    def __reduce__(
        self,
    ) -> tuple[type["CallableContract"], tuple[Any, ...]]:
        """Serialize immutable mapping-backed metadata across worker queues."""
        return (
            self.__class__,
            (
                self.func,
                self.function_name,
                self.module_name,
                self.input_memory_type,
                self.output_memory_type,
                self.artifact_inputs,
                self.artifact_outputs,
                self.runtime_adapter,
                self.processing_contract,
                self.declared_processing_contract,
                self.module_artifact_contract,
                self.raw_processing_function,
                self.runtime_image_execution_mode,
                (
                    dict(self.runtime_batch_executors)
                    if self.runtime_batch_executors is not None
                    else None
                ),
            ),
        )

    @classmethod
    def from_callable(cls, func: Any) -> "CallableContract":
        """Build a contract from callable attributes once at compiler boundary."""
        namespace = _callable_namespace(func)
        function_name = _callable_name(func)
        metadata = CallableMetadataReader(namespace, function_name)
        raw_processing_function = namespace.get(RAW_PROCESSING_FUNCTION_ATTR)
        return cls(
            func=func,
            function_name=function_name,
            module_name=_callable_module(func),
            input_memory_type=metadata.optional_string("input_memory_type"),
            output_memory_type=metadata.optional_string("output_memory_type"),
            artifact_inputs=_artifact_spec_items(
                namespace,
                function_name,
                "__artifact_inputs__",
            ),
            artifact_outputs=_artifact_spec_items(
                namespace,
                function_name,
                "__artifact_outputs__",
            ),
            runtime_adapter=runtime_adapter_spec_from_callable(func),
            processing_contract=namespace.get(PROCESSING_CONTRACT_ATTR),
            declared_processing_contract=metadata.optional_string(
                DECLARED_PROCESSING_CONTRACT_ATTR,
            ),
            module_artifact_contract=module_artifact_contract_from_namespace(
                namespace,
                owner_name=function_name,
            ),
            raw_processing_function=raw_processing_function,
            runtime_image_execution_mode=metadata.optional_execution_mode(
                RUNTIME_IMAGE_EXECUTION_MODE_ATTR,
            ),
            runtime_batch_executors=RuntimeBatchCallableFamily(
                func=func,
                raw_processing_function=raw_processing_function,
            ).executors(),
        )

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

def _callable_namespace(func: Any) -> CallableNamespace:
    """Return user-declared callable metadata."""
    if _is_function_reference(func):
        return func.preserved_attrs
    return func.__dict__


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
        setattr(
            func,
            DECLARED_PROCESSING_CONTRACT_ATTR,
            declared_processing_contract,
        )
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
        setattr(func, RAW_PROCESSING_FUNCTION_ATTR, raw_processing_function)
        raw_prepare = getattr(raw_processing_function, PROCESSING_PREPARE_ATTR, None)
        if raw_prepare is not None and not hasattr(func, PROCESSING_PREPARE_ATTR):
            if not callable(raw_prepare):
                raise TypeError(
                    "raw_processing_function prepare hook must be callable, "
                    f"got {type(raw_prepare).__name__}."
                )
            setattr(func, PROCESSING_PREPARE_ATTR, raw_prepare)
    if prepare is not None:
        if not callable(prepare):
            raise TypeError(
                "prepare must be callable, "
                f"got {type(prepare).__name__}."
            )
        setattr(func, PROCESSING_PREPARE_ATTR, prepare)
    if runtime_image_execution_mode is not None:
        if not isinstance(runtime_image_execution_mode, ImagePayloadExecutionMode):
            raise TypeError(
                "runtime_image_execution_mode must be ImagePayloadExecutionMode, "
                f"got {type(runtime_image_execution_mode).__name__}."
            )
        setattr(
            func,
            RUNTIME_IMAGE_EXECUTION_MODE_ATTR,
            runtime_image_execution_mode,
        )


def _attach_nominal_processing_contract_if_supported(
    func: Any,
    declared_processing_contract: str,
) -> None:
    """Coerce declared contract names to nominal metadata at the declaration boundary."""
    if hasattr(func, PROCESSING_CONTRACT_ATTR):
        return

    from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

    contract = ProcessingContract.from_declared_name(declared_processing_contract)
    if contract is not None:
        setattr(func, PROCESSING_CONTRACT_ATTR, contract)


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
    for target in _callable_prepare_targets(func):
        setattr(target, PROCESSING_PREPARE_ATTR, prepare)


def _callable_prepare_targets(func: Any) -> tuple[Any, ...]:
    """Return wrapper/raw callables that may appear at compiler/runtime boundary."""
    targets: list[Any] = []
    seen: set[int] = set()
    pending = [func]
    while pending:
        target = pending.pop()
        target_id = id(target)
        if target_id in seen:
            continue
        seen.add(target_id)
        targets.append(target)
        raw = getattr(target, RAW_PROCESSING_FUNCTION_ATTR, None)
        if callable(raw):
            pending.append(raw)
        wrapped = getattr(target, "__wrapped__", None)
        if callable(wrapped):
            pending.append(wrapped)
    return tuple(targets)


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
        setattr(func, RUNTIME_IMAGE_EXECUTION_MODE_ATTR, mode)
        return func

    return decorator


def prepare_processing_callable(func: Any) -> None:
    """Run an optional callable preparation hook before timed data processing."""
    module_name = _callable_module(func)
    if module_name is not None:
        _prepare_module_autoregister_families(module_name)
        _prepare_processing_module(module_name)

    namespace = _callable_namespace(func)
    prepare = namespace.get(PROCESSING_PREPARE_ATTR)
    if prepare is None:
        return
    if not callable(prepare):
        raise TypeError(
            f"{_callable_name(func)!r}.{PROCESSING_PREPARE_ATTR} must be "
            f"callable, got {type(prepare).__name__}."
        )
    prepare_key = (
        "callable",
        f"{module_name or '<unknown>'}.{_callable_name(func)}",
        id(prepare),
    )
    with _PREPARED_CALLABLE_LOCK:
        if prepare_key in _PREPARED_CALLABLE_KEYS:
            return
    prepare()
    with _PREPARED_CALLABLE_LOCK:
        _PREPARED_CALLABLE_KEYS.add(prepare_key)


def _prepare_processing_module(module_name: str) -> None:
    """Run an optional module-level preparation hook exactly once."""
    module = importlib.import_module(module_name)
    prepare = getattr(module, PROCESSING_PREPARE_ATTR, None)
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
    for _name, candidate in inspect.getmembers(module, inspect.isclass):
        if not issubclass(candidate, CompilerPreparedAutoRegisterFamily):
            continue
        if candidate is CompilerPreparedAutoRegisterFamily:
            continue
        candidate.prepare_registered_family()


def _callable_name(func: Any) -> str:
    """Return the callable's nominal function name."""
    name = func.function_name if _is_function_reference(func) else func.__name__
    if not isinstance(name, str):
        raise TypeError(f"Callable name must be a string, got {type(name).__name__}.")
    return name


def _callable_module(func: Any) -> str | None:
    """Return the callable's declaring module when available."""
    module_name = (
        func.original_module
        if _is_function_reference(func)
        else func.__module__
    )
    if module_name is None or isinstance(module_name, str):
        return module_name
    raise TypeError(
        f"{_callable_name(func)!r}.__module__ must be a string or None, "
        f"got {type(module_name).__name__}."
    )


def _is_function_reference(func: Any) -> bool:
    """Return whether func is the compiler's nominal picklable reference."""
    from openhcs.core.pipeline.compiler import FunctionReference

    return isinstance(func, FunctionReference)


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
