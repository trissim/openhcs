"""Default external payload registrations for runtime artifact validation."""

from __future__ import annotations

from functools import cache

from arraybridge.framework_config import _FRAMEWORK_CONFIG

from openhcs.core.runtime_values import (
    register_array_payload_predicate,
)


@cache
def register_runtime_payload_integrations() -> None:
    """Register runtime payload predicates from ArrayBridge backend metadata."""
    register_array_payload_predicate(_is_arraybridge_payload)


def _is_arraybridge_payload(data: object) -> bool:
    """Return whether data belongs to a configured ArrayBridge memory backend."""
    if not hasattr(data, "shape"):
        return False
    module_name = type(data).__module__
    return any(
        module_name.startswith(str(config["import_name"]))
        or str(config["import_name"]) in module_name
        for config in _FRAMEWORK_CONFIG.values()
    )
