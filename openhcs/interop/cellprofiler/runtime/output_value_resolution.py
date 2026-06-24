"""Returned CellProfiler output value resolution."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import unwrap

from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.callable_contract import CallableContract
from openhcs.core.runtime_output_matching import RuntimeReturnedOutputMatcher
from openhcs.core.special_outputs import SpecialOutputKindClassifier, special_output_name
from openhcs.interop.cellprofiler.runtime.mapping_lookup import MappingValueLookup
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargDict,
    CellProfilerOptionalFunction,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
)


@dataclass(frozen=True, slots=True)
class CellProfilerCallableOutputSpecs:
    """CellProfiler special-output declarations lowered to artifact specs."""

    func: CellProfilerOptionalFunction = None

    def artifact_specs(self) -> tuple[ArtifactSpec, ...]:
        if self.func is None:
            return ()
        raw_outputs = self.callable_special_outputs(self.func)
        return tuple(
            ArtifactSpec(
                special_output_name(output_spec),
                SpecialOutputKindClassifier.kind_for(output_spec),
            )
            for output_spec in raw_outputs
        )

    @staticmethod
    def callable_special_outputs(func: CellProfilerFunction) -> CellProfilerRuntimeValues:
        contract = CallableContract.from_callable(func)
        candidates: list[CellProfilerRuntimeValue] = [func]
        raw_func = contract.raw_processing_function
        if callable(raw_func):
            candidates.append(raw_func)
        unwrapped = unwrap(func)
        if unwrapped not in candidates:
            candidates.append(unwrapped)
        for candidate in candidates:
            raw_outputs = MappingValueLookup(
                vars(candidate),
                "__special_outputs__",
            ).value_or(())
            if isinstance(raw_outputs, tuple) and raw_outputs:
                return raw_outputs
        return ()


@dataclass(frozen=True, slots=True)
class CellProfilerOutputValueResolutionRequest:
    """Resolve returned CellProfiler values against declared artifact outputs."""

    output_specs: tuple[ArtifactSpec, ...]
    main_output: CellProfilerRuntimeValue
    artifact_values: CellProfilerRuntimeValues
    func: CellProfilerOptionalFunction = None
    declared_output_specs: tuple[ArtifactSpec, ...] = ()

    def values_by_name(self) -> CellProfilerKwargDict:
        resolved = RuntimeReturnedOutputMatcher(
            retained_specs=self.output_specs,
            declared_specs=self.declared_output_specs,
            main_output=self.main_output,
            artifact_values=self.artifact_values,
            returned_specs=CellProfilerCallableOutputSpecs(self.func).artifact_specs(),
        ).resolve()
        if resolved is not None:
            return resolved
        raise ValueError(
            f"CellProfiler module declared {len(self.output_specs)} outputs but "
            f"returned {len(self.artifact_values)} artifact values."
        )


@dataclass(frozen=True, slots=True)
class CellProfilerResolvedOutputValues:
    """Resolved returned values for recorded outputs and full context outputs."""

    recorded_values: CellProfilerKwargDict
    context_values: CellProfilerKwargDict

    @classmethod
    def from_returned_outputs(
        cls,
        *,
        recorded_specs: tuple[ArtifactSpec, ...],
        context_specs: tuple[ArtifactSpec, ...],
        main_output: CellProfilerRuntimeValue,
        artifact_values: CellProfilerRuntimeValues,
        func: CellProfilerOptionalFunction,
        declared_output_specs: tuple[ArtifactSpec, ...],
    ) -> "CellProfilerResolvedOutputValues":
        return cls(
            recorded_values=CellProfilerOutputValueResolutionRequest(
                output_specs=recorded_specs,
                main_output=main_output,
                artifact_values=artifact_values,
                func=func,
                declared_output_specs=declared_output_specs,
            ).values_by_name(),
            context_values=CellProfilerOutputValueResolutionRequest(
                output_specs=context_specs,
                main_output=main_output,
                artifact_values=artifact_values,
                func=func,
                declared_output_specs=declared_output_specs,
            ).values_by_name(),
        )

    def recorded_value(self, spec: ArtifactSpec) -> CellProfilerRuntimeValue:
        """Return the resolved returned value for one output artifact spec."""
        return self.recorded_values[spec.name]
