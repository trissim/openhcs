"""Typed callable contracts used by compiler phases.

This module centralizes metadata extraction from processing callables so the
compiler has one source of truth for memory and artifact declarations.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, MutableMapping
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, get_type_hints

from arraybridge import MemoryContractAttribute, MemoryType
from python_introspect import (
    RuntimeParameterDeclarationABC,
    declared_enum_type,
    validate_annotation_value,
)

from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifact_key_selection import ArtifactPlanKeySelector
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    ImageArtifactType,
)
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.variable_component_stack_requirement import (
    VariableComponentStackRequirement,
)

if TYPE_CHECKING:
    from openhcs.core.function_reference import FunctionReference
    from openhcs.core.pipeline.compilation_session import CompilationPathResolver
    from openhcs.core.runtime_adapters import RuntimeAdapterSpec
    from openhcs.core.runtime_batch_contracts import RuntimeBatchExecutionDomain
    from openhcs.core.vfs_protocol import PlatePathDeclaration
    from openhcs.processing.backends.lib_registry.unified_registry import (
        ProcessingContract,
    )


CallableNamespace = Mapping[str, Any]
_prepared_callable_keys: set[tuple[str, str, Hashable]] = set()
_prepared_callable_lock = Lock()

CallableRuntimeCacheKey = int


@dataclass(frozen=True, slots=True)
class CallableImportIdentity:
    """Stable top-level import identity for one callable declaration."""

    module_name: str
    function_name: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("module_name", self.module_name),
            ("function_name", self.function_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"CallableImportIdentity.{field_name} must be a non-empty string."
                )

    @classmethod
    def from_callable(cls, func: Callable[..., object]) -> "CallableImportIdentity":
        """Return the import identity declared by a resolved callable object."""

        if not callable(func):
            raise TypeError(
                "Callable import identity requires a callable object, got "
                f"{type(func).__name__}."
            )
        return cls(
            module_name=func.__module__,
            function_name=func.__name__,
        )

    @property
    def import_path(self) -> str:
        """Return the complete import path for this callable."""

        return f"{self.module_name}.{self.function_name}"


class KeywordRuntimeParameter(RuntimeParameterDeclarationABC):
    """Reusable declaration for one runtime-owned keyword-only parameter."""

    parameter_name: ClassVar[str | None] = None
    annotation_type: ClassVar[Any] = inspect.Parameter.empty
    parameter_default: ClassVar[Any] = inspect.Parameter.empty

    @classmethod
    def require_parameter_name(cls) -> str:
        """Return the non-empty callable parameter name declared by the leaf."""

        parameter_name = cls.parameter_name
        if not isinstance(parameter_name, str) or not parameter_name.strip():
            raise ValueError(f"{cls.__name__} must declare parameter_name.")
        return parameter_name

    @classmethod
    def default_value(cls) -> Any:
        """Return the declared callable default."""

        return cls.parameter_default

    @classmethod
    def parameter(cls) -> inspect.Parameter:
        """Return the keyword-only signature parameter for this declaration."""

        return inspect.Parameter(
            cls.require_parameter_name(),
            inspect.Parameter.KEYWORD_ONLY,
            default=cls.default_value(),
            annotation=cls.annotation_type,
        )


class CompilerPreparedAutoRegisterFamily(ABC):
    """AutoRegisterMeta family that can prepare runtime compiler substrates."""

    @classmethod
    @abstractmethod
    def prepare_registered_family(cls) -> None:
        """Prepare registered implementations before timed callable execution."""


class FunctionStepExecutionScope(str, Enum):
    """Lifecycle scope for one compiled FunctionStep callable."""

    AXIS = "axis"
    PLATE = "plate"

    def context_owns_outputs(self, *, metadata_writer: bool) -> bool:
        """Return whether one compiled context owns this invocation's outputs."""
        return self is FunctionStepExecutionScope.AXIS or metadata_writer

    @classmethod
    def require_uniform(
        cls,
        contracts: Iterable["CallableContract"],
    ) -> "FunctionStepExecutionScope":
        """Return the one execution scope shared by all callable contracts."""
        scopes = tuple(contract.execution_scope for contract in contracts)
        if not scopes:
            return cls.AXIS
        first = scopes[0]
        if any(scope is not first for scope in scopes[1:]):
            raise ValueError(
                "FunctionStep callable pattern has mixed execution scopes; "
                "all invocation contracts must declare one uniform scope."
            )
        return first


class ImagePayloadConsumption(str, Enum):
    """How a callable consumes its primary image payload."""

    NATURAL = "natural"
    COMPOSED = "composed"


@dataclass(frozen=True, slots=True)
class CallableMetadata:
    """Compiler-visible metadata declared by one processing callable."""

    input_memory_type: str | None = None
    output_memory_type: str | None = None
    execution_memory_type: str | None = None
    artifact_inputs: tuple[ArtifactSpec, ...] = ()
    artifact_input_parameter_names: tuple[str, ...] = ()
    artifact_outputs: tuple[ArtifactSpec, ...] = ()
    runtime_bound_parameters: tuple[type[RuntimeParameterDeclarationABC], ...] = ()
    required_variable_components: tuple[VariableComponents, ...] = ()
    variable_component_stack_requirement: VariableComponentStackRequirement | None = (
        None
    )
    allowed_group_by: tuple[GroupBy, ...] = ()
    runtime_adapter: RuntimeAdapterSpec | None = None
    runtime_context_parameter: str | None = None
    execution_scope: FunctionStepExecutionScope = FunctionStepExecutionScope.AXIS
    processing_contract: Enum | None = None
    declared_processing_contract: str | None = None
    raw_processing_function: Callable[..., object] | "FunctionReference" | None = None
    runtime_image_execution_mode: ImagePayloadExecutionMode | None = None
    image_payload_consumption: ImagePayloadConsumption = ImagePayloadConsumption.NATURAL
    request_binding: "CallableRequestBinding | None" = None
    prepare: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        """Normalize the generic artifact-fed callable parameter declaration."""

        runtime_bound_parameters = (
            RuntimeParameterDeclarationABC.require_declaration_types(
                self.runtime_bound_parameters,
                boundary="CallableMetadata.runtime_bound_parameters",
            )
        )
        object.__setattr__(
            self,
            "runtime_bound_parameters",
            runtime_bound_parameters,
        )
        normalized = _unique_parameter_names(
            self.artifact_input_parameter_names,
            owner="CallableMetadata.artifact_input_parameter_names",
        )
        spec_parameter_names = _artifact_input_parameter_names(self.artifact_inputs)
        if (
            self.artifact_inputs
            and normalized
            and not (frozenset(spec_parameter_names) <= frozenset(normalized))
        ):
            raise ValueError(
                "Callable artifact-fed parameter declarations disagree: "
                f"normalized parameters {normalized!r}, ArtifactSpec.parameter_name "
                f"parameters {spec_parameter_names!r}."
            )
        if not normalized:
            normalized = spec_parameter_names
        object.__setattr__(self, "artifact_input_parameter_names", normalized)

    @property
    def declared_memory_types(self) -> frozenset[MemoryType]:
        """Return every array framework declared by this callable."""

        return frozenset(
            MemoryType(declaration)
            for declaration in (
                self.input_memory_type,
                self.output_memory_type,
                self.execution_memory_type,
            )
            if declaration is not None
        )

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
        from openhcs.core.runtime_adapters import runtime_adapter_spec_from_callable

        namespace = projection.namespace
        reader = CallableMetadataReader(namespace, projection.name)
        runtime_adapter = runtime_adapter_spec_from_callable(projection.func)
        if runtime_adapter is not None and callable(projection.func):
            runtime_adapter.validate_callable_signature(projection.func)
        artifact_inputs = _artifact_input_specs_from_projection(
            projection,
            _artifact_specs_from_namespace(
                namespace,
                projection.name,
                FunctionContractAttribute.artifact_inputs,
            ),
        )
        return cls(
            input_memory_type=reader.optional_string(
                MemoryContractAttribute.INPUT.value
            ),
            output_memory_type=reader.optional_string(
                MemoryContractAttribute.OUTPUT.value
            ),
            execution_memory_type=reader.optional_string(
                MemoryContractAttribute.EXECUTION.value
            ),
            artifact_inputs=artifact_inputs,
            artifact_input_parameter_names=_artifact_input_parameter_names_from_projection(
                projection,
                reader,
                artifact_inputs,
            ),
            artifact_outputs=_artifact_specs_from_namespace(
                namespace,
                projection.name,
                FunctionContractAttribute.artifact_outputs,
            ),
            runtime_bound_parameters=reader.optional_runtime_parameter_type_tuple(
                FunctionContractAttribute.runtime_bound_parameters,
            ),
            required_variable_components=reader.optional_variable_component_tuple(
                FunctionContractAttribute.required_variable_components,
            ),
            variable_component_stack_requirement=(
                reader.optional_variable_component_stack_requirement(
                    FunctionContractAttribute.variable_component_stack_requirement
                )
            ),
            allowed_group_by=reader.optional_group_by_tuple(
                FunctionContractAttribute.allowed_group_by,
            ),
            runtime_adapter=runtime_adapter,
            runtime_context_parameter=_runtime_context_parameter(projection, reader),
            execution_scope=reader.optional_execution_scope(
                FunctionContractAttribute.execution_scope,
            ),
            processing_contract=reader.optional_enum(
                FunctionContractAttribute.processing_contract
            ),
            declared_processing_contract=reader.optional_string(
                FunctionContractAttribute.declared_processing_contract,
            ),
            raw_processing_function=reader.optional_raw_processing_function(
                FunctionContractAttribute.raw_processing_function,
            ),
            runtime_image_execution_mode=reader.optional_execution_mode(
                FunctionContractAttribute.runtime_image_execution_mode,
            ),
            image_payload_consumption=reader.image_payload_consumption(),
            request_binding=reader.optional_request_binding(
                FunctionContractAttribute.callable_request_binding,
            ),
            prepare=reader.optional_callable(
                FunctionContractAttribute.processing_prepare
            ),
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
            MemoryContractAttribute.INPUT.write(namespace, self.input_memory_type)
        if self.output_memory_type is not None:
            MemoryContractAttribute.OUTPUT.write(namespace, self.output_memory_type)
        if self.execution_memory_type is not None:
            MemoryContractAttribute.EXECUTION.write(
                namespace,
                self.execution_memory_type,
            )
        if self.artifact_inputs:
            namespace[FunctionContractAttribute.artifact_inputs] = self.artifact_inputs
        if self.artifact_input_parameter_names:
            namespace[FunctionContractAttribute.artifact_input_parameter_names] = (
                self.artifact_input_parameter_names
            )
        if self.artifact_outputs:
            namespace[FunctionContractAttribute.artifact_outputs] = (
                self.artifact_outputs
            )
        if self.runtime_bound_parameters:
            namespace[FunctionContractAttribute.runtime_bound_parameters] = (
                self.runtime_bound_parameters
            )
        if self.required_variable_components:
            namespace[FunctionContractAttribute.required_variable_components] = (
                self.required_variable_components
            )
        if self.variable_component_stack_requirement is not None:
            namespace[
                FunctionContractAttribute.variable_component_stack_requirement
            ] = self.variable_component_stack_requirement
        if self.allowed_group_by:
            namespace[FunctionContractAttribute.allowed_group_by] = (
                self.allowed_group_by
            )
        if self.runtime_adapter is not None:
            namespace[FunctionContractAttribute.runtime_adapter] = self.runtime_adapter
        if self.runtime_context_parameter is not None:
            namespace[FunctionContractAttribute.runtime_context_parameter] = (
                self.runtime_context_parameter
            )
        namespace[FunctionContractAttribute.execution_scope] = self.execution_scope
        if self.processing_contract is not None:
            namespace[FunctionContractAttribute.processing_contract] = (
                self.processing_contract
            )
        if self.declared_processing_contract is not None:
            namespace[FunctionContractAttribute.declared_processing_contract] = (
                self.declared_processing_contract
            )
        if self.raw_processing_function is not None:
            namespace[FunctionContractAttribute.raw_processing_function] = (
                self.raw_processing_function
            )
        if self.runtime_image_execution_mode is not None:
            namespace[FunctionContractAttribute.runtime_image_execution_mode] = (
                self.runtime_image_execution_mode
            )
        if self.image_payload_consumption is not ImagePayloadConsumption.NATURAL:
            namespace[FunctionContractAttribute.image_payload_consumption] = (
                self.image_payload_consumption
            )
        if self.request_binding is not None:
            namespace[FunctionContractAttribute.callable_request_binding] = (
                self.request_binding
            )
        if self.prepare is not None:
            namespace[FunctionContractAttribute.processing_prepare] = self.prepare
        return namespace


@dataclass(frozen=True, slots=True)
class CallableContract(ArtifactPlanKeySelector):
    """Compiler contract declared by one processing callable."""

    func: Callable[..., object] | "FunctionReference"
    function_name: str
    module_name: str | None
    metadata: CallableMetadata = dataclasses.field(default_factory=CallableMetadata)
    runtime_batch_executors: Mapping[RuntimeBatchExecutionDomain, Callable] | None = (
        None
    )

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
        from openhcs.core.runtime_batch_contracts import RuntimeBatchCallableFamily

        projection = CallableProjection.from_callable(func)
        metadata = CallableMetadata.from_projection(projection)
        stack_requirement = metadata.variable_component_stack_requirement
        if stack_requirement is None and metadata.processing_contract is not None:
            stack_requirement = getattr(
                metadata.processing_contract,
                "variable_component_stack_requirement",
                None,
            )
        if stack_requirement is not None and callable(projection.func):
            metadata = dataclasses.replace(
                metadata,
                variable_component_stack_requirement=(
                    stack_requirement.bind_to_callable(projection.func)
                ),
            )
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
    def execution_memory_type(self) -> str | None:
        """Declared framework used while executing the callable."""
        return self.metadata.execution_memory_type

    @property
    def declared_memory_types(self) -> frozenset[MemoryType]:
        """Return every array framework declared by this callable."""

        return self.metadata.declared_memory_types

    @property
    def artifact_inputs(self) -> ArtifactSpecCollection:
        """Return effective compiled input declarations in contract order."""
        return ArtifactSpecCollection(self.metadata.artifact_inputs)

    @property
    def artifact_input_parameter_names(self) -> tuple[str, ...]:
        """Return callable parameters whose values come from artifact inputs."""

        return self.metadata.artifact_input_parameter_names

    @property
    def artifact_outputs(self) -> ArtifactSpecCollection:
        """Return effective compiled output declarations in contract order."""
        return ArtifactSpecCollection(self.metadata.artifact_outputs)

    @property
    def main_flow_outputs(self) -> ArtifactSpecCollection:
        """Return declared outputs eligible for the canonical image flow."""

        return ArtifactSpecCollection(
            spec for spec in self.artifact_outputs if spec.participates_in_main_flow
        )

    @property
    def canonical_return_output_specs(self) -> ArtifactSpecCollection:
        """Return named outputs carried by the callable's first return slot."""

        declared_outputs = self.artifact_outputs
        if self.execution_scope is FunctionStepExecutionScope.PLATE:
            return ArtifactSpecCollection(declared_outputs[:1])
        if not declared_outputs or not declared_outputs[0].participates_in_main_flow:
            return ArtifactSpecCollection(())
        canonical_count = 1
        if declared_outputs[0].artifact_type is ImageArtifactType:
            for spec in declared_outputs[1:]:
                if (
                    not spec.participates_in_main_flow
                    or spec.artifact_type is not ImageArtifactType
                ):
                    break
                canonical_count += 1
        return ArtifactSpecCollection(declared_outputs[:canonical_count])

    @property
    def trailing_return_output_specs(self) -> ArtifactSpecCollection:
        """Return named outputs carried by trailing positional return slots."""

        canonical_count = len(self.canonical_return_output_specs)
        return ArtifactSpecCollection(self.artifact_outputs[canonical_count:])

    @property
    def output_group_scope_sources(self) -> tuple[ArtifactSpecRef, ...]:
        """Return exact input identities owning declared output group scopes."""

        return tuple(
            dict.fromkeys(
                source_ref
                for output_spec in self.artifact_outputs
                for source_ref in output_spec.group_scope_sources()
            )
        )

    @property
    def group_scope_inputs(self) -> ArtifactSpecCollection:
        """Return inputs whose declared relations own this invocation's group scope."""

        input_specs = self.artifact_inputs
        source_refs = self.output_group_scope_sources
        if not source_refs:
            return input_specs
        sources = ArtifactSpecCollection(
            source_spec
            for source_ref in source_refs
            for source_spec in (input_specs.by_ref(source_ref),)
            if source_spec is not None
        )
        missing = tuple(
            source_ref
            for source_ref in source_refs
            if input_specs.by_ref(source_ref) is None
        )
        if missing:
            raise ValueError(
                f"Callable {self.function_name!r} output group-scope relations "
                f"reference undeclared inputs {missing!r}."
            )
        return sources

    def preserves_input_main_flow(self) -> bool:
        """Return whether declared artifact outputs leave main flow unchanged."""

        return bool(self.artifact_outputs) and not self.main_flow_outputs

    @property
    def runtime_adapter(self) -> RuntimeAdapterSpec | None:
        """Declared runtime adapter."""
        return self.metadata.runtime_adapter

    @property
    def runtime_context_parameter(self) -> str | None:
        """Compiled ABI name for runtime context injection."""
        return self.metadata.runtime_context_parameter

    @property
    def execution_scope(self) -> FunctionStepExecutionScope:
        """Declared lifecycle scope for this callable."""
        return self.metadata.execution_scope

    @property
    def runtime_bound_parameters(self) -> tuple[str, ...]:
        """Declared parameters supplied by runtime execution infrastructure."""
        return tuple(
            parameter_type.require_parameter_name()
            for parameter_type in self.runtime_bound_parameter_types
        )

    @property
    def runtime_bound_parameter_types(
        self,
    ) -> tuple[type[RuntimeParameterDeclarationABC], ...]:
        """Declared runtime-supplied parameter types."""
        return self.metadata.runtime_bound_parameters

    @property
    def config_bound_parameters(self) -> tuple[inspect.Parameter, ...]:
        """PipelineConfig-owned parameters declared by the callable signature."""

        from openhcs.core.config import runtime_config_parameter

        signature = inspect.signature(
            self.resolve_canonical_raw_callable(),
            eval_str=True,
        )
        return tuple(
            normalized
            for parameter in signature.parameters.values()
            for normalized in (runtime_config_parameter(parameter),)
            if normalized is not None
        )

    @property
    def config_bound_parameter_names(self) -> tuple[str, ...]:
        """Names of config values supplied by the compiled step scope."""

        return tuple(parameter.name for parameter in self.config_bound_parameters)

    @property
    def runtime_owned_parameter_names(self) -> frozenset[str]:
        """Parameters supplied by compiled artifact, config, or runtime state."""

        names = {
            *self.artifact_input_parameter_names,
            *self.runtime_bound_parameters,
            *self.config_bound_parameter_names,
        }
        if self.runtime_context_parameter is not None:
            names.add(self.runtime_context_parameter)
        if self.runtime_adapter is not None:
            names.add(self.runtime_adapter.require_parameter_name())
        return frozenset(names)

    @property
    def declared_path_parameters(self) -> Mapping[str, "PlatePathDeclaration"]:
        """Plate-relative path declarations keyed by public parameter name."""

        from openhcs.core.vfs_protocol import PlatePathDeclaration

        raw_callable = self.resolve_canonical_raw_callable()
        signature = inspect.signature(raw_callable)
        annotations = get_type_hints(raw_callable, include_extras=True)
        declarations = {
            parameter_name: declaration
            for parameter_name in signature.parameters
            for annotation in (annotations.get(parameter_name),)
            if annotation is not None
            for declaration in (PlatePathDeclaration.from_annotation(annotation),)
            if declaration is not None
        }
        return MappingProxyType(declarations)

    def declared_path_values(
        self,
        kwargs: Mapping[str, object],
    ) -> Mapping[str, tuple["PlatePathDeclaration", object]]:
        """Return authored or signature-default values for declared paths."""

        signature = inspect.signature(self.resolve_canonical_raw_callable())
        values: dict[str, tuple["PlatePathDeclaration", object]] = {}
        for parameter_name, declaration in self.declared_path_parameters.items():
            if parameter_name in kwargs:
                value = kwargs[parameter_name]
            else:
                value = signature.parameters[parameter_name].default
                if value is inspect.Parameter.empty:
                    continue
            if value is not None:
                values[parameter_name] = (declaration, value)
        return MappingProxyType(values)

    def resolve_declared_paths(
        self,
        kwargs: Mapping[str, object],
        resolver: "CompilationPathResolver",
    ) -> dict[str, object]:
        """Resolve only explicitly declared public path arguments."""

        resolved = dict(kwargs)
        for parameter_name, (declaration, value) in self.declared_path_values(
            kwargs
        ).items():
            if not isinstance(value, (str, Path)):
                raise TypeError(
                    f"Callable {self.function_name!r} path parameter "
                    f"{parameter_name!r} requires str or Path, got "
                    f"{type(value).__name__}."
                )
            resolved[parameter_name] = resolver.resolve(
                value,
                declaration,
                owner=f"callable {self.function_name}.{parameter_name}",
            )
        return resolved

    @property
    def required_variable_components(self) -> tuple[VariableComponents, ...]:
        """Declared FunctionStep variable axes required by this callable."""
        return self.metadata.required_variable_components

    @property
    def variable_component_stack_requirement(
        self,
    ) -> VariableComponentStackRequirement | None:
        """Declared requirement for a non-empty variable-component stack axis."""
        if self.metadata.variable_component_stack_requirement is not None:
            return self.metadata.variable_component_stack_requirement

        from openhcs.processing.backends.lib_registry.unified_registry import (
            ProcessingContract,
        )

        processing_contract = self.processing_contract
        if not isinstance(processing_contract, ProcessingContract):
            return None
        return processing_contract.variable_component_stack_requirement

    @property
    def allowed_group_by(self) -> tuple[GroupBy, ...]:
        """Declared FunctionStep group_by values allowed by this callable."""
        return self.metadata.allowed_group_by

    @property
    def processing_contract(self) -> Enum | None:
        """Declared nominal processing contract."""
        return self.metadata.processing_contract

    def require_processing_contract(self) -> "ProcessingContract":
        """Return the declared processing contract or fail at the contract boundary."""
        from openhcs.processing.backends.lib_registry.unified_registry import (
            ProcessingContract,
        )

        processing_contract = self.processing_contract
        if not isinstance(processing_contract, ProcessingContract):
            raise TypeError(
                f"Callable {self.function_name!r} must declare a ProcessingContract; "
                f"got {type(processing_contract).__name__}."
            )
        return processing_contract

    @property
    def declared_processing_contract(self) -> str | None:
        """Declared processing contract name."""
        return self.metadata.declared_processing_contract

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
    def image_payload_consumption(self) -> ImagePayloadConsumption:
        """Declared primary-image payload consumption."""
        return self.metadata.image_payload_consumption

    @property
    def request_binding(self) -> "CallableRequestBinding | None":
        """Declared callable request binding."""
        return self.metadata.request_binding

    @property
    def artifact_specs(self) -> ArtifactSpecCollection:
        """All artifact specs declared by this callable."""
        return ArtifactSpecCollection(
            (*self.metadata.artifact_inputs, *self.metadata.artifact_outputs)
        )

    @property
    def artifact_key_specs(self) -> ArtifactSpecCollection:
        """Return declarations owned by this callable's effective artifact contract."""

        return self.artifact_specs

    @property
    def primary_input_parameter_name(self) -> str | None:
        """FunctionStep input payload parameter declared by callable signature."""
        signature = inspect.signature(self.resolve_canonical_raw_callable())
        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return parameter.name
        return None

    @property
    def accepts_implicit_main_flow_input(self) -> bool:
        """Whether the callable accepts its primary input from unnamed main flow."""

        primary_input = self.primary_input_parameter_name
        return (
            primary_input is not None
            and primary_input not in self.runtime_owned_parameter_names
        )

    def runtime_batch_executor(
        self,
        domain: RuntimeBatchExecutionDomain,
    ) -> Callable | None:
        """Return the declared runtime batch executor for one domain."""
        if self.runtime_batch_executors is None:
            return None
        return self.runtime_batch_executors.get(domain)

    def runtime_callable_cache_identity(
        self,
    ) -> CallableRuntimeCacheKey:
        """Return the process-local cache identity for this callable contract."""
        if not _is_function_reference(self.func) and not callable(self.func):
            raise TypeError(f"Invalid callable contract function: {self.func}")
        return id(self)

    def resolve_runtime_callable(self) -> Callable[..., object]:
        """Return the executable callable for this contract."""
        resolved = _resolve_declared_callable(self.func)
        runtime_adapter = self.runtime_adapter
        if runtime_adapter is None:
            return resolved
        return runtime_adapter.executable_callable(resolved, self)

    def resolve_canonical_raw_callable(self) -> Callable[..., object]:
        """Return the declaration-owned callable whose signature defines behavior."""

        declared = self.raw_processing_function
        if declared is None:
            declared = self.func
        return _resolve_declared_callable(declared)

    def canonical_raw_import_identity(self) -> CallableImportIdentity:
        """Return the complete import identity of the canonical raw callable."""

        return CallableImportIdentity.from_callable(
            self.resolve_canonical_raw_callable()
        )

    def resolve_raw_runtime_callable(self) -> Callable[..., object]:
        """Remove runtime wrappers without crossing a semantic request binding."""

        canonical = self.resolve_canonical_raw_callable()
        if self.request_binding is None:
            return inspect.unwrap(canonical)

        binding_key = FunctionContractAttribute.callable_request_binding

        def is_request_boundary(candidate: Callable[..., object]) -> bool:
            namespace = _callable_namespace(candidate)
            if binding_key not in namespace:
                return False
            wrapped = namespace.get("__wrapped__")
            return not callable(wrapped) or binding_key not in _callable_namespace(
                wrapped
            )

        resolved = inspect.unwrap(canonical, stop=is_request_boundary)
        if binding_key not in _callable_namespace(resolved):
            raise RuntimeError(
                f"Callable {self.function_name!r} declares a request binding, but "
                "its canonical raw callable has no request-binding boundary."
            )
        return resolved

    def validate_public_kwargs(
        self,
        kwargs: Mapping,
        *,
        runtime_loaded_artifact_parameter_names: Iterable[str] = (),
    ) -> tuple[tuple[object, object], ...]:
        """Validate behavior kwargs against the canonical raw callable ABI."""

        if not isinstance(kwargs, Mapping):
            raise TypeError(
                "CallableContract.validate_public_kwargs requires a mapping."
            )
        raw_callable = self.resolve_canonical_raw_callable()
        signature = inspect.signature(raw_callable)
        runtime_owned_value = object()
        call_kwargs = dict(kwargs)
        runtime_loaded_parameters = frozenset(runtime_loaded_artifact_parameter_names)

        def bind_runtime_owned(parameter_name: str) -> None:
            if parameter_name not in signature.parameters:
                return
            if parameter_name in call_kwargs:
                if call_kwargs[parameter_name] is runtime_owned_value:
                    return
                raise TypeError(
                    f"Callable {self.function_name!r} public kwargs cannot set "
                    f"runtime-owned parameter {parameter_name!r}."
                )
            parameter = signature.parameters[parameter_name]
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise TypeError(
                    f"Callable {self.function_name!r} runtime-owned parameter "
                    f"{parameter_name!r} cannot be positional-only."
                )
            call_kwargs[parameter_name] = runtime_owned_value

        runtime_adapter = self.runtime_adapter
        for parameter_name in self.artifact_input_parameter_names:
            if (
                parameter_name in call_kwargs
                and parameter_name not in runtime_loaded_parameters
            ):
                continue
            bind_runtime_owned(parameter_name)
        if self.runtime_context_parameter is not None:
            bind_runtime_owned(self.runtime_context_parameter)
        if runtime_adapter is not None:
            bind_runtime_owned(runtime_adapter.require_parameter_name())
        for parameter_type in self.runtime_bound_parameter_types:
            bind_runtime_owned(parameter_type.require_parameter_name())
        for parameter_name in self.config_bound_parameter_names:
            bind_runtime_owned(parameter_name)

        primary_input = self.primary_input_parameter_name
        call_args = () if primary_input is None else (runtime_owned_value,)
        try:
            signature.bind(*call_args, **call_kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Callable {self.function_name!r} has invalid public kwargs for "
                f"canonical raw signature {signature}: {exc}"
            ) from exc
        resolved_annotations = get_type_hints(raw_callable, include_extras=True)
        for parameter_name, value in kwargs.items():
            parameter = signature.parameters.get(parameter_name)
            if parameter is None:
                continue
            annotation = resolved_annotations.get(
                parameter_name,
                parameter.annotation,
            )
            if declared_enum_type(annotation) is None:
                continue
            try:
                validate_annotation_value(
                    annotation,
                    value,
                    path=f"{self.function_name}.{parameter_name}",
                )
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Callable {self.function_name!r} has invalid public value "
                    f"for {parameter_name!r}: {exc}"
                ) from exc
        return tuple(kwargs.items())

    def validate_artifact_input_parameter_bindings(self) -> None:
        """Validate exact artifact occurrences against the normalized callable ABI."""

        from openhcs.core.pipeline.function_contracts import (
            resolved_callable_parameter,
            special_input_parameter_accepts_sequence,
        )

        raw_callable = self.resolve_canonical_raw_callable()
        artifact_parameters = {
            parameter_name: resolved_callable_parameter(raw_callable, parameter_name)
            for parameter_name in self.artifact_input_parameter_names
        }
        specs_by_parameter: dict[str, list[ArtifactSpec]] = {}
        for spec in self.artifact_inputs:
            parameter_name = spec.parameter_name
            if parameter_name is None:
                continue
            if parameter_name not in artifact_parameters:
                raise ValueError(
                    f"Callable {self.function_name!r} artifact input {spec.ref()!r} "
                    f"binds undeclared artifact-fed parameter {parameter_name!r}."
                )
            parameter = artifact_parameters[parameter_name]
            if (
                parameter.annotation is not inspect.Signature.empty
                and spec.artifact_type.runtime_parameter_types()
                and not spec.artifact_type.accepts_parameter_annotation(
                    parameter.annotation
                )
            ):
                raise TypeError(
                    f"Callable {self.function_name!r} parameter "
                    f"{parameter_name!r} does not accept "
                    f"{spec.artifact_type.require_value()} artifact payloads."
                )
            specs_by_parameter.setdefault(parameter_name, []).append(spec)

        for parameter_name, parameter in artifact_parameters.items():
            bound_specs = tuple(specs_by_parameter.get(parameter_name, ()))
            if not bound_specs:
                if parameter.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Callable {self.function_name!r} artifact-fed parameter "
                        f"{parameter_name!r} has no exact artifact declaration "
                        "binding."
                    )
                continue
            if (
                len(bound_specs) > 1
                and not special_input_parameter_accepts_sequence(parameter)
                and any(
                    spec.artifact_type is not ImageArtifactType for spec in bound_specs
                )
            ):
                raise ValueError(
                    f"Callable {self.function_name!r} scalar artifact-fed parameter "
                    f"{parameter_name!r} has multiple exact artifact occurrences "
                    f"{tuple(spec.ref() for spec in bound_specs)!r}."
                )


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
                f"{func.__name__} must declare request parameter {request_parameter!r}."
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
        wrapper_namespace[FunctionContractAttribute.callable_request_binding] = binding
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
    runtime_bound_parameters: tuple[type[RuntimeParameterDeclarationABC], ...] = (),
) -> None:
    """Attach OpenHCS callable metadata used by compiler/runtime phases."""
    if declared_processing_contract is not None:
        if (
            not isinstance(declared_processing_contract, str)
            or not declared_processing_contract.strip()
        ):
            raise ValueError("declared_processing_contract must be a non-empty string.")
        namespace = _mutable_callable_namespace(func)
        namespace[FunctionContractAttribute.declared_processing_contract] = (
            declared_processing_contract
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
        namespace = _mutable_callable_namespace(func)
        namespace[FunctionContractAttribute.raw_processing_function] = (
            raw_processing_function
        )
        raw_prepare = CallableMetadata.from_callable(raw_processing_function).prepare
        if (
            raw_prepare is not None
            and FunctionContractAttribute.processing_prepare not in namespace
        ):
            if not callable(raw_prepare):
                raise TypeError(
                    "raw_processing_function prepare hook must be callable, "
                    f"got {type(raw_prepare).__name__}."
                )
            namespace[FunctionContractAttribute.processing_prepare] = raw_prepare
    _attach_runtime_bound_parameter_metadata(
        func,
        runtime_bound_parameters,
        source=raw_processing_function,
    )
    if prepare is not None:
        if not callable(prepare):
            raise TypeError(f"prepare must be callable, got {type(prepare).__name__}.")
        _mutable_callable_namespace(func)[
            FunctionContractAttribute.processing_prepare
        ] = prepare
    if runtime_image_execution_mode is not None:
        if not isinstance(runtime_image_execution_mode, ImagePayloadExecutionMode):
            raise TypeError(
                "runtime_image_execution_mode must be ImagePayloadExecutionMode, "
                f"got {type(runtime_image_execution_mode).__name__}."
            )
        _mutable_callable_namespace(func)[
            FunctionContractAttribute.runtime_image_execution_mode
        ] = runtime_image_execution_mode


def _attach_nominal_processing_contract_if_supported(
    func: Any,
    declared_processing_contract: str,
) -> None:
    """Coerce declared contract names to nominal metadata at the declaration boundary."""
    namespace = _mutable_callable_namespace(func)
    if FunctionContractAttribute.processing_contract in namespace:
        return

    from openhcs.processing.backends.lib_registry.unified_registry import (
        ProcessingContract,
    )

    contract = ProcessingContract.from_declared_name(declared_processing_contract)
    if contract is not None:
        namespace[FunctionContractAttribute.processing_contract] = contract


def _attach_runtime_bound_parameter_metadata(
    func: Any,
    parameter_types: tuple[type[RuntimeParameterDeclarationABC], ...],
    *,
    source: Any | None,
) -> None:
    """Merge runtime-bound parameter declarations into callable metadata."""
    attribute = FunctionContractAttribute.runtime_bound_parameters
    ordered: list[type[RuntimeParameterDeclarationABC]] = []
    seen_names: set[str] = set()
    for owner in (source, func):
        if owner is None:
            continue
        for parameter_type in vars(owner).get(attribute, ()):
            _append_runtime_parameter_type(parameter_type, ordered, seen_names)
    for parameter_type in parameter_types:
        _append_runtime_parameter_type(parameter_type, ordered, seen_names)
    if ordered:
        _mutable_callable_namespace(func)[attribute] = tuple(ordered)


def _append_runtime_parameter_type(
    parameter_type: object,
    ordered: list[type[RuntimeParameterDeclarationABC]],
    seen_names: set[str],
) -> None:
    declaration = RuntimeParameterDeclarationABC.require_declaration_type(
        parameter_type,
        boundary="runtime_bound_parameters",
    )
    parameter_name = declaration.require_parameter_name()
    if parameter_name in seen_names:
        return
    ordered.append(declaration)
    seen_names.add(parameter_name)


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
        _mutable_callable_namespace(target)[
            FunctionContractAttribute.processing_prepare
        ] = prepare


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
        _mutable_callable_namespace(func)[
            FunctionContractAttribute.runtime_image_execution_mode
        ] = mode
        return func

    return decorator


def prepare_processing_callable(func: Any) -> None:
    """Run an optional callable preparation hook before timed data processing."""
    projection = CallableProjection.from_callable(func)
    if projection.module_name is not None:
        prepare_module_autoregister_families(projection.module_name)
        _prepare_processing_module(projection.module_name)

    prepare = projection.namespace.get(FunctionContractAttribute.processing_prepare)
    if prepare is None:
        return
    if not callable(prepare):
        raise TypeError(
            f"{projection.name!r}.{FunctionContractAttribute.processing_prepare} must be "
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
    with _prepared_callable_lock:
        if prepare_key in _prepared_callable_keys:
            return
    prepare()
    with _prepared_callable_lock:
        _prepared_callable_keys.add(prepare_key)


def reset_processing_callable_preparation_cache() -> None:
    """Clear process-local preparation caches for deterministic tests and tooling."""
    with _prepared_callable_lock:
        _prepared_callable_keys.clear()
    prepare_module_autoregister_families.cache_clear()
    from openhcs.core.autoregister_preparation import AutoRegisterRegistryPreparation

    AutoRegisterRegistryPreparation.cached_module_registry_families.cache_clear()


def _prepare_callable_identity(prepare: Callable[..., Any]) -> tuple[str, str]:
    """Return a stable identity for process-local prepare-hook caching."""
    return str(prepare.__module__), str(prepare.__qualname__)


def _prepare_processing_module(module_name: str) -> None:
    """Run an optional module-level preparation hook exactly once."""
    module = importlib.import_module(module_name)
    prepare = vars(module).get(FunctionContractAttribute.processing_prepare)
    if prepare is None:
        return
    if not callable(prepare):
        raise TypeError(
            f"Module {module_name!r}.{FunctionContractAttribute.processing_prepare} "
            "must be callable, "
            f"got {type(prepare).__name__}."
        )
    prepare_key = ("module", module_name, id(prepare))
    with _prepared_callable_lock:
        if prepare_key in _prepared_callable_keys:
            return
    prepare()
    with _prepared_callable_lock:
        _prepared_callable_keys.add(prepare_key)


@lru_cache(maxsize=None)
def prepare_module_autoregister_families(module_name: str) -> None:
    """Prepare AutoRegisterMeta families imported by a callable module."""
    module = importlib.import_module(module_name)
    from openhcs.core.autoregister_preparation import AutoRegisterRegistryPreparation

    AutoRegisterRegistryPreparation.prepare_module_registered_families((module,))


def _is_function_reference(func: Any) -> bool:
    """Return whether func is the compiler's nominal picklable reference."""
    from openhcs.core.function_reference import FunctionReference

    return isinstance(func, FunctionReference)


def _resolve_declared_callable(func: Any) -> Callable[..., object]:
    """Resolve one callable declaration without applying runtime adapters."""

    if _is_function_reference(func):
        return func.resolve()
    if callable(func):
        return func
    raise TypeError(f"Invalid callable contract function: {func}")


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
            raw = namespace.get(FunctionContractAttribute.raw_processing_function)
            if callable(raw):
                pending.append(raw)
            wrapped = namespace.get("__wrapped__")
            if callable(wrapped):
                pending.append(wrapped)
        return tuple(targets)


def _runtime_context_parameter(
    projection: CallableProjection,
    reader: "CallableMetadataReader",
) -> str | None:
    declared = reader.optional_string(
        FunctionContractAttribute.runtime_context_parameter
    )
    if declared is not None:
        return declared
    from openhcs.core.context.processing_context import ProcessingContext

    return _callable_signature_parameter(
        projection.func,
        ProcessingContext.require_parameter_name(),
    )


def _callable_signature_parameter(func: Any, parameter_name: str) -> str | None:
    """Return a callable parameter name accepted by live signatures only."""
    if not callable(func):
        return None
    if parameter_name not in inspect.signature(func).parameters:
        return None
    return parameter_name


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

    def optional_string_tuple(self, field_name: str) -> tuple[str, ...]:
        """Return an optional tuple of non-empty string metadata values."""
        value = self.namespace.get(field_name)
        if value is None:
            return ()
        if not isinstance(value, tuple):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be a tuple, "
                f"got {type(value).__name__}."
            )
        normalized = tuple(item.strip() for item in value if isinstance(item, str))
        if len(normalized) != len(value) or len(normalized) != len(set(normalized)):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must contain unique "
                "non-empty strings."
            )
        if any(not item for item in normalized):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must contain unique "
                "non-empty strings."
            )
        return normalized

    def optional_runtime_parameter_type_tuple(
        self,
        field_name: str,
    ) -> tuple[type[RuntimeParameterDeclarationABC], ...]:
        """Return runtime parameter declaration types from callable metadata."""
        value = self.namespace.get(field_name)
        if value is None:
            return ()
        if not isinstance(value, tuple):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be a tuple, "
                f"got {type(value).__name__}."
            )
        return RuntimeParameterDeclarationABC.require_declaration_types(
            value,
            boundary=f"{self.function_name!r}.{field_name}",
        )

    def optional_variable_component_tuple(
        self,
        field_name: str,
    ) -> tuple[VariableComponents, ...]:
        """Return optional required FunctionStep variable-component metadata."""
        value = self.namespace.get(field_name)
        if value is None:
            return ()
        if not isinstance(value, tuple):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be a tuple, "
                f"got {type(value).__name__}."
            )
        normalized = tuple(
            (
                component
                if isinstance(component, VariableComponents)
                else VariableComponents(component)
            )
            for component in value
        )
        if len(normalized) != len(set(normalized)):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must contain unique "
                "VariableComponents values."
            )
        return normalized

    def optional_variable_component_stack_requirement(
        self,
        field_name: str,
    ) -> VariableComponentStackRequirement | None:
        """Return an optional variable-component stack requirement declaration."""
        value = self.namespace.get(field_name)
        if value is None:
            return None
        if not isinstance(value, VariableComponentStackRequirement):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be "
                "VariableComponentStackRequirement, "
                f"got {type(value).__name__}."
            )
        return value

    def optional_group_by_tuple(
        self,
        field_name: str,
    ) -> tuple[GroupBy, ...]:
        """Return optional allowed FunctionStep group_by metadata."""
        value = self.namespace.get(field_name)
        if value is None:
            return ()
        if not isinstance(value, tuple):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be a tuple, "
                f"got {type(value).__name__}."
            )
        normalized = tuple(
            group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
            for group_by in value
        )
        if len(normalized) != len(set(normalized)):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must contain unique "
                "GroupBy values."
            )
        return normalized

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

    def image_payload_consumption(self) -> ImagePayloadConsumption:
        """Return declared image payload consumption with its natural default."""
        value = self.namespace.get(
            FunctionContractAttribute.image_payload_consumption,
            ImagePayloadConsumption.NATURAL,
        )
        if not isinstance(value, ImagePayloadConsumption):
            raise TypeError(
                f"{self.function_name!r}."
                f"{FunctionContractAttribute.image_payload_consumption} must be "
                "ImagePayloadConsumption, "
                f"got {type(value).__name__}."
            )
        return value

    def optional_execution_scope(
        self,
        field_name: str,
    ) -> FunctionStepExecutionScope:
        """Return a callable execution scope, defaulting to per-axis execution."""
        value = self.namespace.get(field_name)
        if value is None:
            return FunctionStepExecutionScope.AXIS
        if not isinstance(value, FunctionStepExecutionScope):
            raise TypeError(
                f"{self.function_name!r}.{field_name} must be "
                "FunctionStepExecutionScope, "
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
            f"{request_type.__name__} has no request field(s): {', '.join(missing)}."
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


def _unique_parameter_names(
    parameter_names: tuple[str, ...],
    *,
    owner: str,
) -> tuple[str, ...]:
    """Return a validated tuple of unique non-empty callable parameter names."""

    if not isinstance(parameter_names, tuple):
        raise TypeError(f"{owner} must be a tuple.")
    normalized = tuple(
        parameter_name.strip()
        for parameter_name in parameter_names
        if isinstance(parameter_name, str)
    )
    if (
        len(normalized) != len(parameter_names)
        or any(not parameter_name for parameter_name in normalized)
        or len(normalized) != len(set(normalized))
    ):
        raise TypeError(f"{owner} must contain unique non-empty strings.")
    return normalized


def _artifact_input_parameter_names(
    artifact_inputs: Iterable[ArtifactSpec],
) -> tuple[str, ...]:
    """Return unique artifact-fed parameter names in declaration order."""

    return tuple(
        dict.fromkeys(
            spec.parameter_name
            for spec in artifact_inputs
            if spec.parameter_name is not None
        )
    )


def _artifact_input_specs_from_projection(
    projection: CallableProjection,
    artifact_inputs: tuple[ArtifactSpec, ...],
) -> tuple[ArtifactSpec, ...]:
    """Bind exact same-name input declarations at the metadata boundary."""

    if not callable(projection.func):
        return artifact_inputs
    signature = inspect.signature(projection.func)
    primary_parameter_name = next(
        (
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ),
        None,
    )
    artifact_parameter_names = frozenset(signature.parameters) - {
        primary_parameter_name
    }
    return tuple(
        (
            dataclasses.replace(spec, parameter_name=spec.name)
            if spec.parameter_name is None and spec.name in artifact_parameter_names
            else spec
        )
        for spec in artifact_inputs
    )


def _artifact_input_parameter_names_from_projection(
    projection: CallableProjection,
    reader: CallableMetadataReader,
    artifact_inputs: tuple[ArtifactSpec, ...],
) -> tuple[str, ...]:
    """Normalize compatibility and exact artifact parameter declarations once."""

    namespace = projection.namespace
    parameter_declarations: list[tuple[str, tuple[str, ...]]] = []
    normalized_key = FunctionContractAttribute.artifact_input_parameter_names
    legacy_key = FunctionContractAttribute.special_inputs
    artifact_key = FunctionContractAttribute.artifact_inputs
    if normalized_key in namespace:
        parameter_declarations.append(
            (normalized_key, reader.optional_string_tuple(normalized_key))
        )
    if legacy_key in namespace:
        parameter_declarations.append(
            (legacy_key, reader.optional_string_tuple(legacy_key))
        )
    exact_parameter_names = _artifact_input_parameter_names(artifact_inputs)
    if not parameter_declarations:
        first_names = exact_parameter_names
    else:
        first_names = parameter_declarations[0][1]
    first_set = frozenset(first_names)
    if any(
        frozenset(parameter_names) != first_set
        for _source, parameter_names in parameter_declarations[1:]
    ):
        declarations = ", ".join(
            f"{source}={parameter_names!r}"
            for source, parameter_names in parameter_declarations
        )
        raise ValueError(
            f"Callable {projection.name!r} artifact-fed parameter declarations "
            f"disagree: {declarations}."
        )
    if (
        artifact_key in namespace
        and parameter_declarations
        and not frozenset(exact_parameter_names) <= first_set
    ):
        declarations = ", ".join(
            (
                *(
                    f"{source}={parameter_names!r}"
                    for source, parameter_names in parameter_declarations
                ),
                f"{artifact_key}={exact_parameter_names!r}",
            )
        )
        raise ValueError(
            f"Callable {projection.name!r} artifact-fed parameter declarations "
            f"disagree: {declarations}."
        )

    if not callable(projection.func):
        return first_names
    signature = inspect.signature(projection.func)
    ordered_names = tuple(name for name in signature.parameters if name in first_set)
    return (
        *ordered_names,
        *(name for name in first_names if name not in signature.parameters),
    )


def _artifact_specs_from_namespace(
    namespace: CallableNamespace,
    function_name: str,
    attr_name: str,
) -> tuple[ArtifactSpec, ...]:
    raw_specs = namespace.get(attr_name)
    if not raw_specs:
        return ()
    if not isinstance(raw_specs, tuple):
        raise TypeError(
            f"{function_name!r}.{attr_name} must be an ordered ArtifactSpec tuple, "
            f"got {type(raw_specs).__name__}."
        )
    for spec in raw_specs:
        if not isinstance(spec, ArtifactSpec):
            raise TypeError(
                f"{function_name!r}.{attr_name} must contain ArtifactSpec values, "
                f"got {type(spec).__name__}."
            )
    collection = ArtifactSpecCollection(raw_specs)
    if attr_name == FunctionContractAttribute.artifact_inputs:
        return collection.specs
    return collection.unique(conflict_context=f"{function_name}.{attr_name}")
