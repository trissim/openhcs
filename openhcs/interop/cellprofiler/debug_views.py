"""CellProfiler debug view hook over generic OpenHCS debug snapshots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.debug import DebugSnapshot
from openhcs.core.debug_views import DebugViewModel


class CellProfilerDebugView(ABC, metaclass=AutoRegisterMeta):
    """Registered CellProfiler renderer for generic debug snapshots."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str | None) -> "CellProfilerDebugView":
        if module_name is not None:
            renderer_type = cls.__registry__.get(module_name)
            if renderer_type is not None:
                return renderer_type()
        return DefaultCellProfilerDebugView()

    @abstractmethod
    def build_view_model(self, snapshot: DebugSnapshot) -> DebugViewModel:
        """Build a renderer-independent view model for one snapshot."""


class DefaultCellProfilerDebugView(CellProfilerDebugView):
    """Default renderer for CellProfiler modules."""

    module_name = "default"

    def build_view_model(self, snapshot: DebugSnapshot) -> DebugViewModel:
        return DebugViewModel.from_debug_snapshot(snapshot)


def is_cellprofiler_debug_view_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_cellprofiler_debug_view_export(name, value)
)
