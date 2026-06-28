"""Runtime-bound callable parameter declarations for CellProfiler policies."""

from __future__ import annotations


class RuntimeBoundParameterName(str):
    """String marker for callable parameters supplied by runtime binding policies."""

    def __new__(cls, value: str) -> "RuntimeBoundParameterName":
        normalized = value.strip()
        if not normalized:
            raise ValueError("Runtime-bound parameter name cannot be empty.")
        return str.__new__(cls, normalized)


class RuntimeSliceSequenceParameterName(RuntimeBoundParameterName):
    """Runtime-bound tuple parameter projected item-wise per pure-2D slice."""


class MeasurementTableCollectionParameterName(RuntimeBoundParameterName):
    """Runtime-bound parameter carrying a measurement-table collection."""


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
    return tuple(str(name) for name in declared_runtime_bound_parameters(policy_type))


def declared_runtime_slice_sequence_parameter_names(
    policy_type: type,
) -> tuple[str, ...]:
    """Return runtime-bound parameters that project tuple items per slice."""
    return tuple(
        str(name)
        for name in declared_runtime_bound_parameters(policy_type)
        if isinstance(name, RuntimeSliceSequenceParameterName)
    )


def declared_measurement_table_parameter_names(policy_type: type) -> tuple[str, ...]:
    """Return runtime-bound parameters carrying measurement-table collections."""
    return tuple(
        str(name)
        for name in declared_runtime_bound_parameters(policy_type)
        if isinstance(name, MeasurementTableCollectionParameterName)
    )


def declared_runtime_bound_parameters(
    policy_type: type,
) -> tuple[RuntimeBoundParameterName, ...]:
    """Return runtime-bound parameter declarations from a policy MRO."""
    names: list[str] = []
    parameters: list[RuntimeBoundParameterName] = []
    for base_type in reversed(policy_type.__mro__):
        for value in vars(base_type).values():
            if isinstance(value, RuntimeBoundParameterName):
                if str(value) not in names:
                    parameters.append(value)
                    names.append(str(value))
            elif isinstance(value, RuntimeBoundParameterNames):
                for name in value:
                    if str(name) not in names:
                        parameters.append(name)
                        names.append(str(name))
    return tuple(parameters)
