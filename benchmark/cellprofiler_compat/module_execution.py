"""Execution bridge from generated CellProfiler modules to OpenHCS runtime state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.memory import detect_memory_type, stack_slices
from openhcs.core.pipeline.function_contracts import special_input_names_from_callable
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
    def inputs(self) -> tuple[ArtifactSpec, ...]:
        return tuple(self.contract.get("inputs", ()))

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
        image_request = self._image_request(
            image,
            cellprofiler_runtime,
        )
        if self._runs_per_object_measurement():
            return self._run_per_object_measurement(
                func,
                image_request.payload,
                cellprofiler_runtime=cellprofiler_runtime,
                source_image_name=image_request.source_image_name,
                **kwargs,
            )

        invocation = self._invocation_request(
            func,
            image_request=image_request,
            adapter=cellprofiler_runtime,
            kwargs=kwargs,
        )
        raw_output = func(invocation.image, **invocation.kwargs)
        main_output, artifact_values = _split_cellprofiler_output(raw_output)
        self._record_outputs(
            cellprofiler_runtime,
            main_output,
            artifact_values,
            source_image_name=invocation.source_image_name,
        )
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
        source_image_name: str | None,
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
            source_image_name=source_image_name,
        )
        return main_output

    def _runtime_input_kwargs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
    ) -> dict[str, Any]:
        runtime_non_image_inputs = tuple(
            spec
            for spec in self.runtime_artifact_inputs
            if spec.kind is not ArtifactKind.IMAGE
        )
        if not runtime_non_image_inputs:
            return {}

        special_input_names = special_input_names_from_callable(func)
        if special_input_names:
            return _bind_special_runtime_inputs(
                self.module_name,
                special_input_names,
                runtime_non_image_inputs,
                adapter,
            )

        unsupported_non_object_inputs = tuple(
            spec
            for spec in runtime_non_image_inputs
            if spec.kind is not ArtifactKind.OBJECT_LABELS
        )
        if unsupported_non_object_inputs:
            raise NotImplementedError(
                f"{self.module_name} has runtime inputs "
                f"{[spec.name for spec in unsupported_non_object_inputs]} with "
                "no declared special_inputs binding."
            )

        object_inputs = _specs_of_kind(
            runtime_non_image_inputs,
            ArtifactKind.OBJECT_LABELS,
        )
        return CellProfilerObjectInputPolicy.for_module(self.module_name).bind(
            self.module_name,
            object_inputs,
            adapter,
        )

    def _record_outputs(
        self,
        adapter: CellProfilerRuntimeAdapter,
        main_output: Any,
        artifact_values: tuple[Any, ...],
        *,
        source_image_name: str | None,
    ) -> None:
        if not self.outputs:
            return

        output_values = _output_values_by_kind(
            self.outputs,
            main_output,
            artifact_values,
        )
        for spec in self.outputs:
            CellProfilerOutputRecorder.for_kind(spec.kind).record(
                CellProfilerOutputRecordRequest(
                    executor=self,
                    adapter=adapter,
                    spec=spec,
                    value=output_values[spec.name],
                    source_image_name=source_image_name,
                )
            )

    def _image_request(
        self,
        fallback_image: Any,
        adapter: CellProfilerRuntimeAdapter,
    ) -> "CellProfilerImageRequest":
        image_inputs = _specs_of_kind(self.inputs, ArtifactKind.IMAGE)
        if not image_inputs:
            return CellProfilerImageRequest(
                payload=fallback_image,
                source_image_name=self._input_source_image_name(adapter),
            )

        runtime_image_names = {
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.IMAGE,
            )
        }
        external_images = _external_image_payloads(
            self.module_name,
            self.external_image_inputs,
            fallback_image,
        )
        payloads = []
        for spec in image_inputs:
            if spec.name in runtime_image_names:
                payloads.append(adapter.get_image(spec.name).data)
                continue
            try:
                payloads.append(external_images[spec.name])
            except KeyError as exc:
                raise NotImplementedError(
                    f"{self.module_name} declared image input '{spec.name}', but "
                    "it was neither a runtime image artifact nor a configured "
                    "external image input."
                ) from exc
        return CellProfilerImageRequest(
            payload=_compose_image_payload(self.module_name, tuple(payloads)),
            source_image_name=self._input_source_image_name(adapter),
        )

    def _input_source_image_name(
        self,
        adapter: CellProfilerRuntimeAdapter,
    ) -> str | None:
        source_names: list[str] = []
        runtime_image_names = frozenset(
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.IMAGE,
            )
        )
        external_image_names = frozenset(self.external_image_inputs)
        for spec in self.inputs:
            source_name = _artifact_kind_strategy(spec.kind).source_image_name(
                CellProfilerArtifactKindRequest(
                    spec=spec,
                    adapter=adapter,
                    external_image_names=external_image_names,
                    runtime_image_names=runtime_image_names,
                )
            )
            if source_name is not None:
                source_names.append(source_name)

        return _single_source_name(tuple(source_names))

    def _invocation_request(
        self,
        func: Callable[..., Any],
        *,
        image_request: "CellProfilerImageRequest",
        adapter: CellProfilerRuntimeAdapter,
        kwargs: Mapping[str, Any],
    ) -> "CellProfilerInvocationRequest":
        return CellProfilerInvocationRequest(
            image=image_request.payload,
            kwargs={
                **kwargs,
                **self._runtime_input_kwargs(func, adapter),
            },
            source_image_name=image_request.source_image_name,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerImageRequest:
    """Resolved image payload and source metadata for one module invocation."""

    payload: Any
    source_image_name: str | None


@dataclass(frozen=True, slots=True)
class CellProfilerInvocationRequest:
    """Resolved invocation inputs for one CellProfiler function call."""

    image: Any
    kwargs: Mapping[str, Any]
    source_image_name: str | None


@dataclass(frozen=True, slots=True)
class CellProfilerArtifactKindRequest:
    """One artifact-spec request dispatched through a nominal kind strategy."""

    spec: ArtifactSpec
    adapter: CellProfilerRuntimeAdapter
    external_image_names: frozenset[str] = frozenset()
    runtime_image_names: frozenset[str] = frozenset()


class CellProfilerArtifactKindStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy family for ArtifactKind-specific runtime semantics."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True
    kind: ClassVar[ArtifactKind | None] = None

    @classmethod
    def for_kind(cls, kind: ArtifactKind) -> "CellProfilerArtifactKindStrategy":
        return cls.__registry__[kind]()

    @abstractmethod
    def runtime_input_value(self, request: CellProfilerArtifactKindRequest) -> Any:
        """Return the runtime payload bound into absorbed function kwargs."""

    @abstractmethod
    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        """Return the transitive source image name for one artifact input."""


class ImageArtifactKindStrategy(CellProfilerArtifactKindStrategy):
    """Resolve image artifact payloads and source-image lineage."""

    kind = ArtifactKind.IMAGE

    def runtime_input_value(self, request: CellProfilerArtifactKindRequest) -> Any:
        return request.adapter.get_image(request.spec.name).data

    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        if request.spec.name in request.runtime_image_names:
            return request.adapter.get_image(request.spec.name).source_image_name
        if request.spec.name in request.external_image_names:
            return request.spec.name
        return None


class ObjectLabelsArtifactKindStrategy(CellProfilerArtifactKindStrategy):
    """Resolve object-label payloads and lineage."""

    kind = ArtifactKind.OBJECT_LABELS

    def runtime_input_value(self, request: CellProfilerArtifactKindRequest) -> Any:
        return request.adapter.get_objects(request.spec.name).labels

    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        return request.adapter.get_objects(request.spec.name).source_image_name


class MeasurementsArtifactKindStrategy(CellProfilerArtifactKindStrategy):
    """Resolve measurement payloads and lineage."""

    kind = ArtifactKind.MEASUREMENTS

    def runtime_input_value(self, request: CellProfilerArtifactKindRequest) -> Any:
        return request.adapter.get_measurements(request.spec.name).rows

    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        return request.adapter.get_measurements(request.spec.name).source_image_name


class RelationshipsArtifactKindStrategy(CellProfilerArtifactKindStrategy):
    """Resolve relationship payloads."""

    kind = ArtifactKind.RELATIONSHIPS

    def runtime_input_value(self, request: CellProfilerArtifactKindRequest) -> Any:
        raise NotImplementedError(
            f"Relationship runtime input '{request.spec.name}' needs an explicit "
            "binding contract before CellProfiler special_inputs can consume it."
        )

    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        return None


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
    source_image_name: str | None


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
        request.adapter.add_image(
            request.spec.name,
            request.value,
            source_image_name=request.source_image_name,
        )


class ObjectLabelsOutputRecorder(CellProfilerOutputRecorder):
    """Record object-label outputs."""

    kind = ArtifactKind.OBJECT_LABELS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        request.adapter.add_objects(
            request.spec.name,
            request.value,
            source_image_name=request.source_image_name,
        )


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
            source_image_name=request.source_image_name,
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
    main_output: Any,
    artifact_values: tuple[Any, ...],
) -> dict[str, Any]:
    if len(output_specs) == 1:
        return {
            output_specs[0].name: _single_output_value(
                output_specs[0],
                main_output,
                artifact_values,
            )
        }

    if (
        output_specs
        and output_specs[0].kind is ArtifactKind.IMAGE
        and len(output_specs) == len(artifact_values) + 1
    ):
        return {
            output_specs[0].name: main_output,
            **{
                spec.name: value
                for spec, value in zip(
                    output_specs[1:],
                    artifact_values,
                    strict=True,
                )
            },
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
    main_output: Any,
    artifact_values: tuple[Any, ...],
) -> Any:
    if spec.kind is ArtifactKind.IMAGE:
        return main_output
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


def _bind_special_runtime_inputs(
    module_name: str,
    parameter_names: tuple[str, ...],
    runtime_inputs: tuple[ArtifactSpec, ...],
    adapter: CellProfilerRuntimeAdapter,
) -> dict[str, Any]:
    if len(parameter_names) != len(runtime_inputs):
        raise NotImplementedError(
            f"{module_name} declares special_inputs {list(parameter_names)}, but "
            f"compiled runtime inputs are {[spec.name for spec in runtime_inputs]}."
        )
    return {
        parameter_name: _runtime_input_value(spec, adapter)
        for parameter_name, spec in zip(
            parameter_names,
            runtime_inputs,
            strict=True,
        )
    }


def _runtime_input_value(
    spec: ArtifactSpec,
    adapter: CellProfilerRuntimeAdapter,
) -> Any:
    try:
        return _artifact_kind_strategy(spec.kind).runtime_input_value(
            CellProfilerArtifactKindRequest(
                spec=spec,
                adapter=adapter,
            )
        )
    except KeyError as exc:
        raise TypeError(
            f"Unsupported special runtime input kind {spec.kind.value} for "
            f"'{spec.name}'."
        ) from exc


def _artifact_kind_strategy(
    kind: ArtifactKind,
) -> CellProfilerArtifactKindStrategy:
    try:
        return CellProfilerArtifactKindStrategy.for_kind(kind)
    except KeyError as exc:
        raise TypeError(
            f"No CellProfiler artifact kind strategy registered for {kind.value}."
        ) from exc


def _external_image_payloads(
    module_name: str,
    external_image_inputs: tuple[str, ...],
    fallback_image: Any,
) -> dict[str, Any]:
    if not external_image_inputs:
        return {}
    if len(external_image_inputs) > 1:
        raise NotImplementedError(
            f"{module_name} requires external images {list(external_image_inputs)}, "
            "but converted execution still needs a typed NamesAndTypes/Images "
            "source plan for multi-image external bindings."
        )
    return {external_image_inputs[0]: fallback_image}


def _compose_image_payload(
    module_name: str,
    image_payloads: tuple[Any, ...],
) -> Any:
    if not image_payloads:
        raise ValueError(f"{module_name} cannot compose an empty image input set.")
    if len(image_payloads) == 1:
        return image_payloads[0]

    memory_type = detect_memory_type(image_payloads[0])
    return stack_slices(list(image_payloads), memory_type=memory_type, gpu_id=0)


def _single_source_name(source_names: tuple[str, ...]) -> str | None:
    unique_names = tuple(dict.fromkeys(source_names))
    if len(unique_names) == 1:
        return unique_names[0]
    return None
