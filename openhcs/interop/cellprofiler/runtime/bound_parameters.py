"""Runtime-bound callable parameter declarations for CellProfiler policies."""

from __future__ import annotations


class RuntimeBoundParameterName(str):
    """String marker for callable parameters supplied by runtime binding policies."""

    def __new__(cls, value: str) -> "RuntimeBoundParameterName":
        normalized = value.strip()
        if not normalized:
            raise ValueError("Runtime-bound parameter name cannot be empty.")
        return str.__new__(cls, normalized)


class RuntimeBoundParameterNames(tuple[RuntimeBoundParameterName, ...]):
    """Ordered marker tuple for a closed group of runtime-bound parameters."""

    def __new__(
        cls,
        *values: str | RuntimeBoundParameterName,
    ) -> "RuntimeBoundParameterNames":
        return tuple.__new__(
            cls,
            tuple(
                value
                if isinstance(value, RuntimeBoundParameterName)
                else RuntimeBoundParameterName(value)
                for value in values
            ),
        )


def declared_runtime_bound_parameter_names(policy_type: type) -> tuple[str, ...]:
    """Return runtime-bound parameter declarations from a policy MRO."""
    names: list[str] = []
    for base_type in reversed(policy_type.__mro__):
        for value in vars(base_type).values():
            if isinstance(value, RuntimeBoundParameterName):
                names.append(str(value))
            elif isinstance(value, RuntimeBoundParameterNames):
                names.extend(str(name) for name in value)
    return tuple(dict.fromkeys(names))
