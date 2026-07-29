"""Picklable function references for FunctionStep transport."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

from openhcs.core.callable_contract import CallableContract, CallableMetadata
from python_introspect import Enableable

if TYPE_CHECKING:
    from openhcs.core.steps.abstract import AbstractStep
    from openhcs.processing.backends.lib_registry.unified_registry import (
        FunctionMetadata,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionReference:
    """Picklable registry/import reference plus explicit callable metadata."""

    function_name: str
    registry_name: str
    memory_type: str
    composite_key: str
    original_module: str
    metadata: CallableMetadata = dataclasses.field(default_factory=CallableMetadata)

    def resolve(self) -> Callable:
        """Resolve this reference to the decorated callable for execution."""
        if self.registry_name == "python":
            resolved = FunctionReferenceTransportAuthority.importable_function(
                self.original_module,
                self.function_name,
            )
            if callable(resolved):
                return resolved
            raise RuntimeError(
                f"Python function {self.original_module}.{self.function_name} "
                "is not importable in this process."
            )

        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        all_functions = RegistryService.get_all_functions_with_metadata()
        if self.composite_key not in all_functions:
            raise RuntimeError(
                f"Function {self.composite_key} not found in registry. "
                "Ensure the function registry is initialized in this process."
            )
        return all_functions[self.composite_key].func


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

        registry_match = RegistryService.metadata_for_callable(func)
        if registry_match is not None:
            return cls._registry_reference(
                func,
                registry_match,
                metadata=cls.callable_metadata(registry_match[1].func),
            )

        original_func = inspect.unwrap(func)
        imported = cls.importable_function(
            original_func.__module__,
            original_func.__name__,
        )
        if callable(imported):
            return cls._python_reference(
                func,
                metadata=cls.callable_metadata(func),
            )

        raise RuntimeError(
            f"Function {original_func.__name__} "
            f"(module: {original_func.__module__}) not found in registry or "
            "importable module attribute - cannot create reference"
        )

    @classmethod
    def callable_metadata(cls, func: Callable) -> CallableMetadata:
        """Return compiler transport metadata declared by the callable."""
        metadata = CallableMetadata.from_callable(func)
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
                func,
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
        func: Callable,
        registry_match: tuple[str, "FunctionMetadata"],
        *,
        metadata: CallableMetadata,
    ) -> FunctionReference:
        """Build one reference to a registry-owned callable."""
        composite_key, function_metadata = registry_match
        original_func = inspect.unwrap(func)
        return FunctionReference(
            function_name=original_func.__name__,
            registry_name=function_metadata.registry.library_name,
            memory_type=function_metadata.registry.MEMORY_TYPE,
            composite_key=composite_key,
            original_module=original_func.__module__,
            metadata=metadata,
        )

    @staticmethod
    def _python_reference(
        func: Callable,
        *,
        metadata: CallableMetadata,
    ) -> FunctionReference:
        """Build one reference to an importable Python callable."""
        original_func = inspect.unwrap(func)
        contract = CallableContract.from_callable(func)
        memory_type = (
            "python"
            if contract.input_memory_type is None
            else contract.input_memory_type
        )
        return FunctionReference(
            function_name=original_func.__name__,
            registry_name="python",
            memory_type=memory_type,
            composite_key=(
                f"python:{original_func.__module__}:{original_func.__name__}"
            ),
            original_module=original_func.__module__,
            metadata=metadata,
        )

    @staticmethod
    def importable_function(module_name: str, function_name: str) -> Callable | None:
        """Return a top-level importable function by explicit module namespace."""
        module_namespace = vars(importlib.import_module(module_name))
        resolved = module_namespace.get(function_name)
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
