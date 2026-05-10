"""Generated CellProfiler pipeline import and registry authorities."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from openhcs.core.callable_contract import CallableContract
from openhcs.core.pipeline import Pipeline
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.processing.backends.lib_registry.unified_registry import (
    FunctionMetadata,
    ProcessingContract,
)
from openhcs.processing.func_registry import register_function


@dataclass(frozen=True, slots=True)
class GeneratedPipelineModuleIdentity:
    """Stable import identity for one generated pipeline module."""

    module_path: Path
    code: str
    explicit_module_name: str | None = None

    @property
    def module_name(self) -> str:
        if self.explicit_module_name is not None:
            return self.explicit_module_name
        digest = hashlib.sha1(
            f"{self.module_path.resolve()}::{self.code}".encode("utf-8")
        ).hexdigest()[:12]
        stem = "".join(
            character if character.isalnum() else "_"
            for character in self.module_path.stem
        ).strip("_")
        return f"benchmark_generated_{stem or 'pipeline'}_{digest}"


class GeneratedPipelineRuntimeModule:
    """Nominal authority for generated pipeline module runtime integration."""

    @classmethod
    def for_generated_source(
        cls,
        *,
        module_path: Path,
        code: str,
    ) -> "GeneratedPipelineRuntimeModule":
        return cls(GeneratedPipelineModuleIdentity(module_path=module_path, code=code))

    def __init__(self, identity: GeneratedPipelineModuleIdentity) -> None:
        self.identity = identity

    @property
    def module_name(self) -> str:
        return self.identity.module_name

    def load_from_path(self, module_path: Path) -> ModuleType:
        """Import generated pipeline code from disk under the stable module name."""
        spec = importlib.util.spec_from_file_location(self.module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create module spec for {module_path}.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        spec.loader.exec_module(module)
        return module

    def load_from_source(self, *, filename: str) -> ModuleType:
        """Import generated pipeline code from source with the stable module name."""
        module = ModuleType(self.module_name)
        module.__file__ = filename
        sys.modules[self.module_name] = module
        exec(compile(self.identity.code, filename, "exec"), module.__dict__)
        return module

    def materialize_import_module(self, *, output_dir: Path) -> Path:
        """Write an importable module that restores registry visibility on import."""
        importable_path = output_dir / f"{self.module_name}.py"
        importable_source = (
            self.identity.code
            + "\n\n"
            + "if __name__ != '__main__':\n"
            + "    import sys as _openhcs_generated_sys\n"
            + "    from openhcs.interop.cellprofiler.runtime.generated_pipeline import "
            + "register_generated_pipeline_functions as _openhcs_register_generated\n"
            + "    _openhcs_register_generated(_openhcs_generated_sys.modules[__name__])\n"
        )
        if (
            not importable_path.exists()
            or importable_path.read_text(encoding="utf-8") != importable_source
        ):
            importable_path.write_text(importable_source, encoding="utf-8")
        output_dir_text = str(output_dir)
        if output_dir_text not in sys.path:
            sys.path.insert(0, output_dir_text)
        return importable_path

    def pipeline_from_module(self, module: ModuleType, *, pipeline_name: str) -> Pipeline:
        """Build a Pipeline object from generated module exports."""
        pipeline_steps = GeneratedPipelineModuleExports(module).pipeline_steps
        if isinstance(pipeline_steps, Pipeline):
            return pipeline_steps
        if not isinstance(pipeline_steps, list):
            raise TypeError(
                f"Generated module {module.__name__}.pipeline_steps must be list or "
                f"Pipeline, got {type(pipeline_steps).__name__}."
            )
        return Pipeline(steps=pipeline_steps, name=pipeline_name)


@dataclass(frozen=True, slots=True)
class GeneratedPipelineModuleExports:
    """Typed access to generated module exports."""

    module: ModuleType

    @property
    def pipeline_steps(self) -> Any:
        try:
            return self.module.pipeline_steps
        except AttributeError as exc:
            raise AttributeError(
                f"Generated module {self.module.__name__} does not define "
                "pipeline_steps."
            ) from exc

    @property
    def step_callables(self) -> tuple[Callable[..., Any], ...]:
        callables: list[Callable[..., Any]] = []
        seen: set[int] = set()
        for step in self.pipeline_steps:
            for func in GeneratedFunctionSpec(step.func).callables:
                func_id = id(func)
                if func_id in seen:
                    continue
                seen.add(func_id)
                callables.append(func)
        return tuple(callables)


@dataclass(frozen=True, slots=True)
class GeneratedFunctionSpec:
    """Callable extraction for generated FunctionStep function specifications."""

    func_spec: Any

    @property
    def callables(self) -> tuple[Callable[..., Any], ...]:
        if callable(self.func_spec):
            return (self.func_spec,)
        if (
            isinstance(self.func_spec, tuple)
            and len(self.func_spec) in {2, 3}
            and callable(self.func_spec[0])
        ):
            return (self.func_spec[0],)
        if isinstance(self.func_spec, list):
            callables: list[Callable[..., Any]] = []
            for item in self.func_spec:
                callables.extend(GeneratedFunctionSpec(item).callables)
            return tuple(callables)
        raise TypeError(
            "Unsupported generated FunctionStep func spec "
            f"{type(self.func_spec).__name__}."
        )


@dataclass(frozen=True, slots=True)
class GeneratedPipelineFunctionRegistration:
    """OpenHCS registry registration authority for generated runtime wrappers."""

    module: ModuleType

    def register(self) -> tuple[str, ...]:
        registry = OpenHCSRegistry()
        existing_references = {
            (
                inspect.unwrap(metadata.func).__module__,
                inspect.unwrap(metadata.func).__name__,
            )
            for metadata in RegistryService.get_all_functions_with_metadata().values()
        }
        registered_names: list[str] = []
        registered_new_function = False

        for func in GeneratedPipelineModuleExports(self.module).step_callables:
            reference = (
                inspect.unwrap(func).__module__,
                inspect.unwrap(func).__name__,
            )
            metadata_name = GeneratedPipelineFunction(func).metadata_name
            if reference in existing_references:
                registered_names.append(metadata_name)
                continue

            contract = GeneratedPipelineFunction(func).processing_contract
            func.__processing_contract__ = contract
            wrapped_func = registry.apply_contract_wrapper(func, contract)
            wrapped_func.__processing_contract__ = contract
            wrapped_func.__function_metadata__ = FunctionMetadata(
                name=metadata_name,
                func=wrapped_func,
                contract=contract,
                registry=registry,
                module=wrapped_func.__module__ or "",
                doc=wrapped_func.__doc__ or "",
                tags=["openhcs", "generated", "cellprofiler"],
                original_name=wrapped_func.__name__,
            )
            register_function(wrapped_func, backend="openhcs")
            existing_references.add(reference)
            registered_names.append(metadata_name)
            registered_new_function = True

        if registered_new_function:
            RegistryService.clear_metadata_cache()
        return tuple(registered_names)


@dataclass(frozen=True, slots=True)
class GeneratedPipelineFunction:
    """Typed metadata projection for one generated runtime function."""

    func: Callable[..., Any]

    @property
    def processing_contract(self) -> ProcessingContract:
        contract = CallableContract.from_callable(self.func)
        if isinstance(contract.processing_contract, ProcessingContract):
            return contract.processing_contract
        raise TypeError(
            f"Generated function {contract.function_name!r} has no nominal "
            "__processing_contract__ metadata. Coerce declared contracts during "
            "callable metadata attachment before registry registration."
        )

    @property
    def metadata_name(self) -> str:
        return f"{self.func.__module__}:{self.func.__name__}"


def generated_pipeline_module_name(module_path: Path, code: str) -> str:
    """Compatibility facade for generated module identity projection."""
    return GeneratedPipelineModuleIdentity(module_path=module_path, code=code).module_name


def load_generated_pipeline_module(
    module_path: Path,
    *,
    module_name: str,
) -> ModuleType:
    """Compatibility facade for importing generated code from disk."""
    return GeneratedPipelineRuntimeModule(
        GeneratedPipelineModuleIdentity(
            module_path=module_path,
            code="",
            explicit_module_name=module_name,
        )
    ).load_from_path(module_path)


def load_generated_pipeline_module_from_source(
    source: str,
    *,
    module_name: str,
    filename: str,
) -> ModuleType:
    """Compatibility facade for importing generated code from source."""
    module_path = Path(filename)
    return GeneratedPipelineRuntimeModule(
        GeneratedPipelineModuleIdentity(
            module_path=module_path,
            code=source,
            explicit_module_name=module_name,
        )
    ).load_from_source(filename=filename)


def materialize_generated_pipeline_import_module(
    source: str,
    *,
    module_name: str,
    output_dir: Path,
) -> Path:
    """Compatibility facade for generated module materialization."""
    module_path = output_dir / f"{module_name}.py"
    return GeneratedPipelineRuntimeModule(
        GeneratedPipelineModuleIdentity(
            module_path=module_path,
            code=source,
            explicit_module_name=module_name,
        )
    ).materialize_import_module(output_dir=output_dir)


def pipeline_from_generated_module(
    module: ModuleType,
    *,
    pipeline_name: str,
) -> Pipeline:
    """Compatibility facade for building a Pipeline from generated exports."""
    return GeneratedPipelineRuntimeModule(
        GeneratedPipelineModuleIdentity(module_path=Path(module.__file__ or ""), code="")
    ).pipeline_from_module(module, pipeline_name=pipeline_name)


def register_generated_pipeline_functions(module: ModuleType) -> tuple[str, ...]:
    """Register generated pipeline callables with the OpenHCS function registry."""
    return GeneratedPipelineFunctionRegistration(module).register()
