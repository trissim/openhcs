"""Tool adapters."""

from __future__ import annotations

__all__ = ("CellProfilerAdapter", "OpenHCSAdapter")

_EXPORT_NAMES = frozenset(__all__)
_MISSING_EXPORT = object()


def _adapter_export_modules():
    import benchmark.adapters.openhcs as openhcs_adapter

    yield openhcs_adapter

    import benchmark.adapters.cellprofiler as cellprofiler_adapter

    yield cellprofiler_adapter


def resolve_adapter_export(name: str):
    """Resolve one public adapter export from its owning module."""
    if name not in _EXPORT_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    existing = globals().get(name, _MISSING_EXPORT)
    if existing is not _MISSING_EXPORT:
        return existing
    for module in _adapter_export_modules():
        namespace = vars(module)
        if name in namespace:
            value = namespace[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str):
    """Resolve public adapter re-exports from their owning modules on demand."""
    return resolve_adapter_export(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORT_NAMES)
