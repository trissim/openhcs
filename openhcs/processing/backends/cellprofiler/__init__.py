"""CellProfiler-compatible processing functions for OpenHCS.

The package exposes declaration-owned implementation callables lazily. Module
ownership remains exclusively on the registered ``CellProfilerModule`` class.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

from openhcs.processing.backends.lib_registry.openhcs_registry import (
    OpenHCSFunctionCatalogModule,
)

class CellProfilerBackendModule(OpenHCSFunctionCatalogModule):
    """Lazy package projection of the nominal CellProfiler module registry."""

    def __getattribute__(self, name: str) -> Any:
        try:
            value = ModuleType.__getattribute__(self, name)
        except AttributeError:
            return CellProfilerBackendModule.__getattr__(self, name)

        if not isinstance(value, ModuleType):
            return value

        from openhcs.interop.cellprofiler.module_declarations import (
            CellProfilerModule,
        )

        module_type = CellProfilerModule.for_backend_function_name(name)
        if module_type is not None:
            return module_type.require_callable(name)
        return value

    @staticmethod
    def declared_function_names() -> tuple[str, ...]:
        from openhcs.interop.cellprofiler.module_declarations import (
            CellProfilerModule,
        )

        return tuple(
            dict.fromkeys(
                function_name
                for module_type in CellProfilerModule.__registry__.values()
                for function_name in module_type.declared_function_names()
            )
        )

    @property
    def __all__(self) -> tuple[str, ...]:
        return tuple(sorted(self.declared_function_names()))

    def openhcs_registry_functions(self) -> tuple[Callable[..., Any], ...]:
        from openhcs.interop.cellprofiler.module_declarations import (
            CellProfilerModule,
        )

        return tuple(
            module_type.require_callable(function_name)
            for module_type in CellProfilerModule.__registry__.values()
            for function_name in module_type.declared_function_names()
        )

    def __getattr__(self, name: str) -> Any:
        from openhcs.interop.cellprofiler.module_declarations import (
            CellProfilerModule,
        )

        module_type = CellProfilerModule.for_backend_function_name(name)
        if module_type is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        return module_type.require_callable(name)


def __dir__() -> list[str]:
    module = sys.modules[__name__]
    if not isinstance(module, CellProfilerBackendModule):
        return sorted(globals())
    return sorted((*globals(), *module.declared_function_names()))


sys.modules[__name__].__class__ = CellProfilerBackendModule
