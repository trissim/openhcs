"""Picklable function references for FunctionStep transport."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

from openhcs.core.callable_contract import (
    CallableImportIdentity,
    CallableMetadata,
    FunctionStepExecutionScope,
)
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from python_introspect import Enableable

if TYPE_CHECKING:
    from openhcs.core.steps.abstract import AbstractStep
    from openhcs.processing.backends.lib_registry.unified_registry import (
        FunctionMetadata,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class FunctionReference(ABC):
    """Picklable callable identity plus explicit compiler metadata."""

    import_identity: CallableImportIdentity
    composite_key: str
    metadata: CallableMetadata = dataclasses.field(default_factory=CallableMetadata)
    declaration_revision: str | None = None

    @property
    def function_name(self) -> str:
        """Return the declaration name from the sole import identity owner."""

        return self.import_identity.function_name

    @property
    def original_module(self) -> str:
        """Return the declaration module from the sole import identity owner."""

        return self.import_identity.module_name

    @abstractmethod
    def resolve(self) -> Callable:
        """Resolve this reference through its nominal transport authority."""

    def require_current_declaration(self, resolved: Callable) -> Callable:
        """Return ``resolved`` only when its source proof matches compilation."""

        expected = self.declaration_revision
        if expected is None:
            return resolved
        current = vars(resolved).get(FunctionContractAttribute.declaration_revision)
        if current != expected:
            raise RuntimeError(
                f"Function declaration {self.original_module}.{self.function_name} "
                "changed after this reference was compiled; recompile the pipeline."
            )
        return resolved


@dataclass(frozen=True, kw_only=True)
class ImportableFunctionReference(FunctionReference):
    """Reference whose callable identity is owned by a Python module export."""

    def resolve(self) -> Callable:
        resolved = FunctionReferenceTransportAuthority.importable_function(
            self.original_module,
            self.function_name,
        )
        if callable(resolved):
            return self.require_current_declaration(resolved)
        raise RuntimeError(
            f"Python function {self.original_module}.{self.function_name} "
            "is not importable in this process."
        )


@dataclass(frozen=True, kw_only=True)
class RegistryFunctionReference(FunctionReference):
    """Reference whose callable identity is owned by a function registry."""

    def __post_init__(self) -> None:
        registry_name, separator, function_key = self.composite_key.partition(":")
        if not separator or not registry_name or not function_key:
            raise ValueError(
                "Registry function references require a '<registry>:<key>' "
                f"composite key, got {self.composite_key!r}."
            )

    @property
    def registry_name(self) -> str:
        """Return the registry owner derived from the canonical composite key."""

        return self.composite_key.partition(":")[0]

    def resolve(self) -> Callable:
        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        return RegistryService.resolve_function_reference(self)


class FunctionReferenceTransportAuthority:
    """Converts pipeline callables into picklable registry references."""

    @classmethod
    def reference_pipeline(
        cls,
        pipeline_definition: Sequence["AbstractStep"],
    ) -> list["AbstractStep"]:
        """Return a declaration copy whose function specs are FunctionReferences."""
        return [cls.reference_step(step) for step in pipeline_definition]

    @classmethod
    def reference_pipeline_in_place(
        cls,
        pipeline_definition: list["AbstractStep"],
    ) -> None:
        """Mutate FunctionStep callables into FunctionReferences for compilation."""
        logger.debug(
            "FUNCTION REFRESH: Processing %s steps",
            len(pipeline_definition),
        )
        from openhcs.core.steps.function_step import FunctionStep

        for step_idx, step in enumerate(pipeline_definition):
            if not isinstance(step, FunctionStep):
                continue

            func_spec = step.function_spec()
            if func_spec is None:
                logger.debug(
                    "FUNCTION REFRESH: Step %s (%s): No function pattern",
                    step_idx,
                    step.name,
                )
                continue

            old_type = type(func_spec).__name__
            step.func = cls.reference_function_spec(func_spec)
            new_type = type(step.func).__name__
            logger.debug(
                "FUNCTION REFRESH: Step %s (%s): %s -> %s",
                step_idx,
                step.name,
                old_type,
                new_type,
            )

    @classmethod
    def reference_step(cls, step: "AbstractStep") -> "AbstractStep":
        """Return a FunctionStep copy with referenced function specs."""
        from openhcs.core.steps.function_step import FunctionStep

        if not isinstance(step, FunctionStep):
            return step
        func_spec = step.function_spec()
        if func_spec is None:
            return step
        referenced = cls.reference_function_spec(func_spec)
        if referenced is func_spec:
            return step
        return step.with_function_spec(referenced)

    @classmethod
    def reference_function_spec(cls, func_value: object) -> object:
        """Convert callable function-pattern leaves to FunctionReference."""
        if callable(func_value):
            return cls.function_reference(func_value)

        if isinstance(func_value, tuple):
            if len(func_value) != 2:
                raise TypeError(
                    "Function-pattern tuple leaves must contain exactly two "
                    "members: (callable, kwargs)."
                )
            func, params = func_value

            if isinstance(params, dict) and Enableable.disabled_in(params):
                return None

            if isinstance(params, dict) and Enableable.parameter_in(params):
                params = Enableable.without_parameter(params)

            if callable(func):
                func_ref = cls.reference_function_spec(func)
                return (func_ref, params)
            return (func, params)

        if isinstance(func_value, list):
            referenced = [cls.reference_function_spec(item) for item in func_value]
            return [item for item in referenced if item is not None]

        if isinstance(func_value, dict):
            referenced = {
                key: cls.reference_function_spec(value)
                for key, value in func_value.items()
            }
            return {
                key: value for key, value in referenced.items() if value is not None
            }

        return func_value

    @classmethod
    def function_reference(cls, func: Callable) -> FunctionReference:
        """Convert a callable to a picklable FunctionReference."""
        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        registry_match = RegistryService.declared_metadata_for_callable(func)
        if registry_match is not None:
            return cls._registry_reference(
                registry_match,
                metadata=cls.callable_metadata(registry_match[1].func),
            )

        original_func = inspect.unwrap(func)
        imported = cls.importable_function(
            original_func.__module__,
            original_func.__name__,
        )
        if imported is func:
            return cls._python_reference(
                func,
                metadata=cls.callable_metadata(func),
            )

        registry_match = RegistryService.metadata_for_callable(func)
        if registry_match is not None:
            return cls._registry_reference(
                registry_match,
                metadata=cls.callable_metadata(registry_match[1].func),
            )

        raise RuntimeError(
            f"Function {original_func.__name__} "
            f"(module: {original_func.__module__}) not found in registry or "
            "importable module attribute - cannot create reference"
        )

    @classmethod
    def callable_metadata(cls, func: Callable) -> CallableMetadata:
        """Return compiler transport metadata declared by the callable."""
        from openhcs.core.callable_contract import CallableContract

        metadata = CallableContract.from_callable(func).metadata
        raw_processing_function = metadata.raw_processing_function
        if not callable(raw_processing_function):
            return metadata
        return metadata.with_raw_processing_function(
            cls.raw_processing_function_reference(raw_processing_function)
        )

    @classmethod
    def raw_processing_function_reference(cls, func: Callable) -> FunctionReference:
        """Reference a raw callable through its exact nominal owner.

        An importable declaration remains owned by its Python module. A wrapper
        displaced from that module namespace is owned by its registered callable.
        Raw-callable references carry no recursive compiler metadata.
        """
        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        original_func = inspect.unwrap(func)
        imported = cls.importable_function(
            original_func.__module__,
            original_func.__name__,
        )
        empty_metadata = CallableMetadata()
        if imported is func:
            return cls._python_reference(func, metadata=empty_metadata)

        registry_match = RegistryService.metadata_for_callable(func)
        if registry_match is not None:
            return cls._registry_reference(
                registry_match,
                metadata=empty_metadata,
            )

        raise RuntimeError(
            f"Raw processing function {original_func.__name__} "
            f"(module: {original_func.__module__}) has no importable or "
            "registered nominal owner."
        )

    @staticmethod
    def _registry_reference(
        registry_match: tuple[str, "FunctionMetadata"],
        *,
        metadata: CallableMetadata,
    ) -> FunctionReference:
        """Build one reference to a registry-owned callable."""
        composite_key, function_metadata = registry_match
        if (
            metadata.processing_contract is None
            and metadata.execution_scope is FunctionStepExecutionScope.AXIS
        ):
            metadata = dataclasses.replace(
                metadata,
                processing_contract=function_metadata.contract,
            )
        return RegistryFunctionReference(
            import_identity=function_metadata.import_identity,
            composite_key=composite_key,
            metadata=metadata,
            declaration_revision=FunctionReferenceTransportAuthority.declaration_revision(
                function_metadata.func
            ),
        )

    @staticmethod
    def _python_reference(
        func: Callable,
        *,
        metadata: CallableMetadata,
    ) -> FunctionReference:
        """Build one reference to an importable Python callable."""
        original_func = inspect.unwrap(func)
        return ImportableFunctionReference(
            import_identity=CallableImportIdentity.from_callable(original_func),
            composite_key=(
                f"python:{original_func.__module__}:{original_func.__name__}"
            ),
            metadata=metadata,
            declaration_revision=FunctionReferenceTransportAuthority.declaration_revision(
                func
            ),
        )

    @staticmethod
    def declaration_revision(func: Callable) -> str | None:
        """Return the optional declaration-owned source proof for ``func``."""

        revision = vars(func).get(FunctionContractAttribute.declaration_revision)
        if revision is None:
            return None
        if not isinstance(revision, str) or not revision:
            raise TypeError(
                f"{func.__module__}.{func.__name__} declares an invalid source "
                "revision proof."
            )
        return revision

    @staticmethod
    def importable_function(module_name: str, function_name: str) -> Callable | None:
        """Return a top-level importable function by explicit module namespace."""
        module = importlib.import_module(module_name)
        resolved = getattr(module, function_name, None)
        if callable(resolved):
            return resolved
        submodule = FunctionReferenceTransportAuthority.importable_submodule(
            module_name,
            function_name,
        )
        if submodule is None:
            return None
        resolved = vars(submodule).get(function_name)
        if callable(resolved):
            return resolved
        return None

    @staticmethod
    def importable_submodule(module_name: str, function_name: str) -> ModuleType | None:
        """Return a same-named function submodule when the package exposes one."""
        try:
            return importlib.import_module(f"{module_name}.{function_name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"{module_name}.{function_name}":
                return None
            raise
