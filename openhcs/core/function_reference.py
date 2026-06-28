"""Picklable function references for FunctionStep transport."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, Any

from openhcs.core.callable_contract import CallableContract, CallableMetadata
from python_introspect import Enableable

if TYPE_CHECKING:
    from openhcs.core.steps.abstract import AbstractStep

logger = logging.getLogger(__name__)

CELLPROFILER_BACKEND_MODULE = "openhcs.processing.backends.cellprofiler"


@dataclass(frozen=True)
class FunctionReference:
    """Picklable registry/import reference plus explicit callable metadata."""

    function_name: str
    registry_name: str
    memory_type: str
    composite_key: str
    original_module: str
    metadata: CallableMetadata = dataclasses.field(
        default_factory=CallableMetadata
    )

    def resolve(self) -> Callable:
        """Resolve this reference to the decorated callable for execution."""
        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        all_functions = RegistryService.get_all_functions_with_metadata()
        if self.composite_key in all_functions:
            return all_functions[self.composite_key].func
        resolved = FunctionReferenceTransportAuthority.importable_function(
            self.original_module,
            self.function_name,
        )
        if callable(resolved):
            return resolved
        raise RuntimeError(
            f"Function {self.composite_key} not found in registry. "
            f"Ensure the function registry is initialized in this process."
        )


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

        if isinstance(func_value, tuple) and len(func_value) in {2, 3}:
            func, params, *invocation_options = func_value

            if (
                isinstance(params, dict)
                and Enableable.disabled_in(params)
            ):
                return None

            if isinstance(params, dict) and Enableable.parameter_in(params):
                params = Enableable.without_parameter(params)

            if callable(func):
                func_ref = cls.reference_function_spec(func)
                return (func_ref, params, *invocation_options)
            return (func, params, *invocation_options)

        if isinstance(func_value, list):
            referenced = [cls.reference_function_spec(item) for item in func_value]
            return [item for item in referenced if item is not None]

        if isinstance(func_value, dict):
            referenced = {
                key: cls.reference_function_spec(value)
                for key, value in func_value.items()
            }
            return {key: value for key, value in referenced.items() if value is not None}

        return func_value

    @staticmethod
    def function_reference(func: Callable) -> FunctionReference:
        """Convert a callable to a picklable FunctionReference."""
        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        cellprofiler_reference = (
            FunctionReferenceTransportAuthority.cellprofiler_catalog_reference(func)
        )
        if cellprofiler_reference is not None:
            return cellprofiler_reference

        original_func = inspect.unwrap(func)
        original_name = original_func.__name__
        original_module = original_func.__module__

        all_functions = RegistryService.get_all_functions_with_metadata()
        for composite_key, metadata in all_functions.items():
            registry_original = inspect.unwrap(metadata.func)
            registry_module = registry_original.__module__
            if (
                registry_original.__name__ == original_name
                and registry_module == original_module
            ):
                return FunctionReference(
                    function_name=original_name,
                    registry_name=metadata.registry.library_name,
                    memory_type=metadata.registry.MEMORY_TYPE,
                    composite_key=composite_key,
                    original_module=original_module,
                    metadata=FunctionReferenceTransportAuthority.callable_metadata(
                        func
                    ),
                )

        imported = FunctionReferenceTransportAuthority.importable_function(
            original_module,
            original_name,
        )
        if callable(imported):
            contract = CallableContract.from_callable(func)
            memory_type = (
                "python"
                if contract.input_memory_type is None
                else contract.input_memory_type
            )
            return FunctionReference(
                function_name=original_name,
                registry_name="python",
                memory_type=memory_type,
                composite_key=f"python:{original_module}:{original_name}",
                original_module=original_module,
                metadata=FunctionReferenceTransportAuthority.callable_metadata(func),
            )

        raise RuntimeError(
            f"Function {original_name} (module: {original_module}) not found in "
            "registry or importable module attribute - cannot create reference"
        )

    @staticmethod
    def cellprofiler_catalog_reference(func: Callable) -> FunctionReference | None:
        """Return a package-catalog reference for CellProfiler backend functions."""
        function_name = FunctionReferenceTransportAuthority.cellprofiler_function_name(
            func
        )
        if function_name is None:
            return None

        from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog

        try:
            CellProfilerFunctionCatalog.get_function(function_name)
        except KeyError:
            return None

        contract = CallableContract.from_callable(func)
        memory_type = (
            "python"
            if contract.input_memory_type is None
            else contract.input_memory_type
        )
        return FunctionReference(
            function_name=function_name,
            registry_name="cellprofiler",
            memory_type=memory_type,
            composite_key=f"cellprofiler:{function_name}",
            original_module=CELLPROFILER_BACKEND_MODULE,
            metadata=FunctionReferenceTransportAuthority.callable_metadata(func),
        )

    @staticmethod
    def callable_metadata(func: Callable) -> CallableMetadata:
        """Return compiler transport metadata with raw callables stabilized."""
        metadata = CallableMetadata.from_callable(func)
        raw_processing_function = metadata.raw_processing_function
        if not callable(raw_processing_function):
            return metadata
        stable_raw = FunctionReferenceTransportAuthority.cellprofiler_catalog_function(
            raw_processing_function
        )
        if stable_raw is None:
            return metadata
        return metadata.with_raw_processing_function(stable_raw)

    @staticmethod
    def cellprofiler_catalog_function(func: Callable) -> Callable | None:
        """Resolve a CellProfiler callable to the package-owned catalog function."""
        function_name = FunctionReferenceTransportAuthority.cellprofiler_function_name(
            func
        )
        if function_name is None:
            return None
        from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog

        try:
            return CellProfilerFunctionCatalog.get_function(function_name)
        except KeyError:
            return None

    @staticmethod
    def cellprofiler_catalog_module(module: ModuleType) -> Callable | None:
        """Resolve a CellProfiler function submodule to its catalog function."""
        function_name = FunctionReferenceTransportAuthority.cellprofiler_module_function_name(
            module
        )
        if function_name is None:
            return None
        from openhcs.processing.backends.cellprofiler import CellProfilerFunctionCatalog

        try:
            return CellProfilerFunctionCatalog.get_function(function_name)
        except KeyError:
            return None

    @staticmethod
    def cellprofiler_function_name(func: Callable) -> str | None:
        """Return the CellProfiler catalog name for a backend callable."""
        if not isinstance(func, FunctionType):
            return None
        module_name = func.__module__
        if (
            module_name != CELLPROFILER_BACKEND_MODULE
            and not module_name.startswith(f"{CELLPROFILER_BACKEND_MODULE}.")
        ):
            return None
        return func.__name__

    @staticmethod
    def cellprofiler_module_function_name(module: ModuleType) -> str | None:
        """Return the CellProfiler catalog name for a backend function submodule."""
        module_name = module.__name__
        module_prefix = f"{CELLPROFILER_BACKEND_MODULE}."
        if not module_name.startswith(module_prefix):
            return None
        return module_name.removeprefix(module_prefix)

    @staticmethod
    def importable_function(module_name: str, function_name: str) -> Callable | None:
        """Return a top-level importable function by explicit module namespace."""
        module_namespace = vars(importlib.import_module(module_name))
        resolved = module_namespace.get(function_name)
        if callable(resolved):
            return resolved
        catalog_resolved = (
            FunctionReferenceTransportAuthority.catalog_exported_function(
                module_namespace,
                function_name,
            )
        )
        if catalog_resolved is not None:
            return catalog_resolved
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

    @staticmethod
    def catalog_exported_function(
        module_namespace: Mapping[str, Any],
        function_name: str,
    ) -> Callable | None:
        """Return a function from a package-owned export catalog when present."""
        function_names = module_namespace.get("CELLPROFILER_FUNCTION_NAMES")
        catalog = module_namespace.get("CellProfilerFunctionCatalog")
        if (
            function_names is None
            or catalog is None
            or function_name not in function_names
        ):
            return None
        return catalog.get_function(function_name)
