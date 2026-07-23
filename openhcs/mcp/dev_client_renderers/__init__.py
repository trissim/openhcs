"""Renderer modules for the OpenHCS MCP dev client."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules

_registered = False


def ensure_dev_client_renderers_registered() -> None:
    """Import renderer modules so AutoRegisterMeta sees their declarations."""
    global _registered
    if _registered:
        return
    for module_info in iter_modules(__path__):
        if module_info.ispkg:
            continue
        module_name = module_info.name
        import_module(f"{__name__}.{module_name}")
    _registered = True

__all__ = (
    "ensure_dev_client_renderers_registered",
)
