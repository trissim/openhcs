"""Execution bridge from generated CellProfiler modules to OpenHCS runtime state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.runtime_adapters import RuntimeAdapterRequest
from openhcs.core.runtime_stores import require_runtime_value_store

from benchmark.cellprofiler_compat.runtime_adapter import CellProfilerRuntimeAdapter


def cellprofiler_runtime_adapter_factory(
    request: RuntimeAdapterRequest,
) -> CellProfilerRuntimeAdapter:
    """Build a CellProfiler adapter for one FunctionStep invocation."""
    axis_id = request.context.axis_id
    if not axis_id:
        raise RuntimeError(
            "ProcessingContext.axis_id is required for CellProfiler runtime."
        )
    return CellProfilerRuntimeAdapter(
        runtime_value_store=require_runtime_value_store(
            request.context,
            owner_name="ProcessingContext",
        ),
        axis_id=str(axis_id),
        artifact_outputs=request.artifact_outputs,
        filemanager=request.context.filemanager,
    )


@dataclass(frozen=True, slots=True)
class CellProfilerModuleExecutor:
    """Execute one generated CellProfiler module against a typed runtime adapter."""

    contract: Mapping[str, Any]

    @property
    def module_name(self) -> str:
        return str(self.contract["module_name"])

    @property
    def runtime_artifact_inputs(self) -> tuple[ArtifactSpec, ...]:
        return tuple(self.contract.get("runtime_artifact_inputs", ()))

    @property
    def external_image_inputs(self) -> tuple[str, ...]:
        return tuple(self.contract.get("external_image_inputs", ()))

    @property
    def outputs(self) -> tuple[ArtifactSpec, ...]:
        return tuple(self.contract.get("outputs", ()))

    def run(
        self,
        func: Callable[..., Any],
        image: Any,
        *,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        **kwargs: Any,
    ) -> Any:
        """Call the absorbed function and record declared outputs through the adapter."""
        if self._runs_per_object_measurement():
            return self._run_per_object_measurement(
                func,
                image,
                cellprofiler_runtime=cellprofiler_runtime,
                **kwargs,
            )

        bound_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(cellprofiler_runtime),
        }
        raw_output = func(image, **bound_kwargs)
        main_output, artifact_values = _split_cellprofiler_output(raw_output)
        self._record_outputs(cellprofiler_runtime, artifact_values)
        return main_output

    def _runs_per_object_measurement(self) -> bool:
        return CellProfilerPerObjectMeasurementPolicy.matches(
            self.module_name,
            self.runtime_artifact_inputs,
        )

    def _run_per_object_measurement(
        self,
        func: Callable[..., Any],
        image: Any,
        *,
        cellprofiler_runtime: CellProfilerRuntimeAdapter,
        **kwargs: Any,
    ) -> Any:
        object_inputs = _specs_of_kind(
            self.runtime_artifact_inputs,
            ArtifactKind.OBJECT_LABELS,
        )
        measurement_outputs = _specs_of_kind(self.outputs, ArtifactKind.MEASUREMENTS)
        if len(measurement_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} per-object execution requires exactly one "
                "measurement output."
            )

        main_output = image
        combined_rows: list[Any] = []
        for object_spec in object_inputs:
            labels = cellprofiler_runtime.get_objects(object_spec.name).labels
            raw_output = func(image, labels=labels, **kwargs)
            main_output, artifact_values = _split_cellprofiler_output(raw_output)
            combined_rows.extend(_measurement_rows_from_output(artifact_values))

        cellprofiler_runtime.add_measurements(
            measurement_outputs[0].name,
            combined_rows,
            source_image_name=_source_image_name(self.external_image_inputs),
        )
        return main_output

    def _runtime_input_kwargs(
        self,
        adapter: CellProfilerRuntimeAdapter,
    ) -> dict[str, Any]:
        object_inputs = _specs_of_kind(
            self.runtime_artifact_inputs,
            ArtifactKind.OBJECT_LABELS,
        )
        image_inputs = _specs_of_kind(
            self.runtime_artifact_inputs,
            ArtifactKind.IMAGE,
        )
        if image_inputs:
            raise NotImplementedError(
                f"{self.module_name} has produced-image inputs "
                f"{[spec.name for spec in image_inputs]}, but generated "
                "CellProfiler execution currently supports external image "
                "inputs and object-label runtime inputs."
            )
        return CellProfilerObjectInputPolicy.for_module(self.module_name).bind(
            self.module_name,
            object_inputs,
            adapter,
        )

    def _record_outputs(
        self,
        adapter: CellProfilerRuntimeAdapter,
        artifact_values: tuple[Any, ...],
    ) -> None:
        if not self.outputs:
            return

        output_values = _output_values_by_kind(self.outputs, artifact_values)
        for spec in self.outputs:
            CellProfilerOutputRecorder.for_kind(spec.kind).record(
                CellProfilerOutputRecordRequest(
                    executor=self,
                    adapter=adapter,
                    spec=spec,
                    value=output_values[spec.name],
                )
            )


class CellProfilerObjectInputPolicy(ABC):
    """Nominal binding policy for CellProfiler object-label inputs."""

    module_names: ClassVar[tuple[str, ...]] = ()
    _registry: ClassVar[dict[str, type["CellProfilerObjectInputPolicy"]]] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for module_name in cls.module_names:
            cls._registry[module_name] = cls

    @classmethod
    def for_module(cls, module_name: str) -> "CellProfilerObjectInputPolicy":
        policy_type = cls._registry.get(module_name, UnsupportedObjectInputPolicy)
        return policy_type()

    @abstractmethod
    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
    ) -> dict[str, Any]:
        """Return absorbed-function kwargs for object-label runtime inputs."""


class UnsupportedObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Reject undeclared object-input semantics instead of guessing."""

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
    ) -> dict[str, Any]:
        if not object_inputs:
            return {}
        raise NotImplementedError(
            f"{module_name} has object runtime inputs "
            f"{[spec.name for spec in object_inputs]}, but no nominal input binding "
            "policy has been declared for this CellProfiler module."
        )


class IdentifySecondaryObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Bind primary-object labels for secondary object identification."""

    module_names = ("IdentifySecondaryObjects",)

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
    ) -> dict[str, Any]:
        _require_exact_object_count(module_name, object_inputs, 1)
        return {"primary_labels": adapter.get_objects(object_inputs[0].name).labels}


class IdentifyTertiaryObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Bind smaller/larger labels to the absorbed tertiary-object signature."""

    module_names = ("IdentifyTertiaryObjects",)

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
    ) -> dict[str, Any]:
        _require_exact_object_count(module_name, object_inputs, 2)
        larger, smaller = object_inputs
        return {
            "primary_labels": adapter.get_objects(smaller.name).labels,
            "secondary_labels": adapter.get_objects(larger.name).labels,
        }


class SingleLabelMeasurementInputPolicy(CellProfilerObjectInputPolicy):
    """Bind one object-label input to measurement functions."""

    module_names = (
        "MeasureObjectSizeShape",
        "MeasureObjectIntensity",
        "MeasureTexture",
        "MeasureColocalization",
        "MeasureObjectNeighbors",
    )

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
    ) -> dict[str, Any]:
        _require_exact_object_count(module_name, object_inputs, 1)
        return {"labels": adapter.get_objects(object_inputs[0].name).labels}


class CellProfilerPerObjectMeasurementPolicy:
    """Predicate for modules that need one absorbed call per object set."""

    module_names: ClassVar[tuple[str, ...]] = (
        "MeasureObjectSizeShape",
        "MeasureObjectIntensity",
        "MeasureTexture",
        "MeasureColocalization",
    )

    @classmethod
    def matches(
        cls,
        module_name: str,
        runtime_artifact_inputs: tuple[ArtifactSpec, ...],
    ) -> bool:
        return module_name in cls.module_names and len(
            _specs_of_kind(runtime_artifact_inputs, ArtifactKind.OBJECT_LABELS)
        ) > 1


@dataclass(frozen=True, slots=True)
class CellProfilerOutputRecordRequest:
    """Inputs needed to record one declared CellProfiler artifact output."""

    executor: CellProfilerModuleExecutor
    adapter: CellProfilerRuntimeAdapter
    spec: ArtifactSpec
    value: Any


class CellProfilerOutputRecorder(ABC):
    """Nominal output writer selected by artifact kind."""

    kind: ClassVar[ArtifactKind]
    _registry: ClassVar[dict[ArtifactKind, type["CellProfilerOutputRecorder"]]] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._registry[cls.kind] = cls

    @classmethod
    def for_kind(cls, kind: ArtifactKind) -> "CellProfilerOutputRecorder":
        recorder_type = cls._registry.get(kind)
        if recorder_type is None:
            raise TypeError(f"Unsupported CellProfiler output kind {kind.value}.")
        return recorder_type()

    @abstractmethod
    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        """Record one output artifact through the runtime adapter."""


class ImageOutputRecorder(CellProfilerOutputRecorder):
    """Record image outputs."""

    kind = ArtifactKind.IMAGE

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_image(request.spec.name, request.value)


class ObjectLabelsOutputRecorder(CellProfilerOutputRecorder):
    """Record object-label outputs."""

    kind = ArtifactKind.OBJECT_LABELS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_objects(request.spec.name, request.value)


class MeasurementsOutputRecorder(CellProfilerOutputRecorder):
    """Record measurement outputs with inferred image/object ownership."""

    kind = ArtifactKind.MEASUREMENTS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_measurements(
            request.spec.name,
            request.value,
            object_name=_measurement_object_name(
                request.executor.runtime_artifact_inputs
            ),
            source_image_name=_source_image_name(
                request.executor.external_image_inputs
            ),
        )


class RelationshipsOutputRecorder(CellProfilerOutputRecorder):
    """Reject relationship outputs until the generated module exposes ids."""

    kind = ArtifactKind.RELATIONSHIPS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        raise NotImplementedError(
            f"{request.executor.module_name} declares relationship output "
            f"'{request.spec.name}', but generated relationship execution needs "
            "explicit parent/child id vectors before it can record RuntimeValue "
            "relationships."
        )


def _output_values_by_kind(
    output_specs: tuple[ArtifactSpec, ...],
    artifact_values: tuple[Any, ...],
) -> dict[str, Any]:
    if len(output_specs) == 1:
        return {
            output_specs[0].name: _single_output_value(
                output_specs[0],
                artifact_values,
            )
        }

    if len(output_specs) != len(artifact_values):
        raise ValueError(
            f"CellProfiler module declared {len(output_specs)} outputs but "
            f"returned {len(artifact_values)} artifact values."
        )
    return {
        spec.name: value
        for spec, value in zip(output_specs, artifact_values, strict=True)
    }


def _single_output_value(
    spec: ArtifactSpec,
    artifact_values: tuple[Any, ...],
) -> Any:
    if not artifact_values:
        raise ValueError(
            f"CellProfiler module did not return a value for output '{spec.name}'."
        )
    if spec.kind is ArtifactKind.OBJECT_LABELS:
        return artifact_values[-1]
    return artifact_values[0]


def _split_cellprofiler_output(raw_output: Any) -> tuple[Any, tuple[Any, ...]]:
    if isinstance(raw_output, tuple):
        return raw_output[0], tuple(raw_output[1:])
    return raw_output, ()


def _measurement_rows_from_output(artifact_values: tuple[Any, ...]) -> list[Any]:
    if not artifact_values:
        return []
    rows = artifact_values[0]
    if isinstance(rows, list):
        return rows
    if isinstance(rows, tuple):
        return list(rows)
    return [rows]


def _specs_of_kind(
    specs: Sequence[ArtifactSpec],
    kind: ArtifactKind,
) -> tuple[ArtifactSpec, ...]:
    return tuple(spec for spec in specs if spec.kind is kind)


def _require_exact_object_count(
    module_name: str,
    object_inputs: tuple[ArtifactSpec, ...],
    expected_count: int,
) -> None:
    if len(object_inputs) != expected_count:
        raise NotImplementedError(
            f"{module_name} requires {expected_count} object runtime input(s), "
            f"got {[spec.name for spec in object_inputs]}."
        )


def _measurement_object_name(
    runtime_inputs: tuple[ArtifactSpec, ...],
) -> str | None:
    object_inputs = _specs_of_kind(runtime_inputs, ArtifactKind.OBJECT_LABELS)
    if len(object_inputs) == 1:
        return object_inputs[0].name
    return None


def _source_image_name(external_image_inputs: tuple[str, ...]) -> str | None:
    if len(external_image_inputs) == 1:
        return external_image_inputs[0]
    return None
