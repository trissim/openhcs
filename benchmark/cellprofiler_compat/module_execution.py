"""Execution bridge from generated CellProfiler modules to OpenHCS runtime state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from inspect import signature
import re
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.memory import detect_memory_type, stack_slices
from openhcs.core.pipeline.function_contracts import special_input_names_from_callable
from openhcs.core.runtime_adapters import RuntimeAdapterRequest
from openhcs.core.runtime_stores import require_runtime_value_store

from benchmark.cellprofiler_compat.module_contract import CellProfilerModuleContract
from benchmark.cellprofiler_compat.relationship_payload import (
    CellProfilerRelationshipPayload,
)
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
        source_binding_plan=request.source_binding_plan,
        source_binding_context=request.source_binding_context,
        group_key=request.group_key,
        processing_context=request.context,
        filemanager=request.context.filemanager,
    )


@dataclass(frozen=True, slots=True)
class CellProfilerModuleExecutor:
    """Execute one generated CellProfiler module against a typed runtime adapter."""

    contract: CellProfilerModuleContract

    def __post_init__(self) -> None:
        if not isinstance(self.contract, CellProfilerModuleContract):
            raise TypeError(
                "CellProfilerModuleExecutor.contract must be "
                "CellProfilerModuleContract, got "
                f"{type(self.contract).__name__}."
            )

    @property
    def module_name(self) -> str:
        return self.contract.module_name

    @property
    def inputs(self) -> tuple[ArtifactSpec, ...]:
        return self.contract.inputs

    @property
    def runtime_artifact_inputs(self) -> tuple[ArtifactSpec, ...]:
        return self.contract.runtime_artifact_inputs

    @property
    def outputs(self) -> tuple[ArtifactSpec, ...]:
        return self.contract.outputs

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
                input_image=image,
                measurement_source_image=image_request.payload,
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
        if not self._replaces_main_flow(
            input_image=image,
            output_image=main_output,
        ):
            return image
        return main_output

    def _runs_per_object_measurement(self) -> bool:
        return CellProfilerPerObjectMeasurementPolicy.matches(
            self.module_name,
            self.runtime_artifact_inputs,
        )

    def _produces_image_output(self) -> bool:
        return any(spec.kind is ArtifactKind.IMAGE for spec in self.outputs)

    def _replaces_main_flow(
        self,
        *,
        input_image: Any,
        output_image: Any,
    ) -> bool:
        if not self._produces_image_output():
            return False
        return _payload_slice_count(output_image) == _payload_slice_count(input_image)

    def _run_per_object_measurement(
        self,
        func: Callable[..., Any],
        *,
        input_image: Any,
        measurement_source_image: Any,
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

        combined_rows: list[Any] = []
        for object_spec in object_inputs:
            raw_labels = cellprofiler_runtime.get_objects(object_spec.name).labels
            measurement_labels = _measurement_labels(raw_labels)
            measurement_image = _measurement_image_for_labels(
                measurement_source_image,
                measurement_labels,
            )
            raw_output = func(measurement_image, labels=measurement_labels, **kwargs)
            _ignored_main_output, artifact_values = _split_cellprofiler_output(
                raw_output
            )
            combined_rows.extend(_measurement_rows_from_output(artifact_values))

        cellprofiler_runtime.add_measurements(
            measurement_outputs[0].name,
            combined_rows,
            source_image_name=source_image_name,
        )
        return input_image

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
        external_image_names = tuple(
            spec.name
            for spec in image_inputs
            if spec.name not in runtime_image_names
        )
        adapter.require_resolvable_source_aliases(external_image_names)
        payloads = []
        for spec in image_inputs:
            if spec.name in runtime_image_names:
                payloads.append(adapter.get_image(spec.name).data)
                continue
            payloads.append(adapter.resolve_source_image(spec.name, fallback_image))
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
        external_image_names = frozenset(self._external_source_image_names())
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
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(func, adapter),
        }
        return CellProfilerInvocationRequest(
            image=image_request.payload,
            kwargs=_coerce_invocation_kwargs(func, runtime_kwargs),
            source_image_name=image_request.source_image_name,
        )

    def _external_source_image_names(self) -> tuple[str, ...]:
        runtime_image_names = frozenset(
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.IMAGE,
            )
        )
        return tuple(
            spec.name
            for spec in _specs_of_kind(self.inputs, ArtifactKind.IMAGE)
            if spec.name not in runtime_image_names
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


def _coerce_invocation_kwargs(
    func: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    coerced_kwargs = dict(kwargs)
    parameters = signature(func).parameters
    for name, value in tuple(coerced_kwargs.items()):
        enum_type = _enum_annotation_type(parameters.get(name))
        if enum_type is None:
            continue
        coerced_kwargs[name] = _coerce_enum_argument(enum_type, value, name)
    return coerced_kwargs


def _enum_annotation_type(parameter: Any) -> type[Enum] | None:
    if parameter is None:
        return None
    annotation = parameter.annotation
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    return None


def _coerce_enum_argument(
    enum_type: type[Enum],
    value: Any,
    parameter_name: str,
) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError:
        pass
    if isinstance(value, str):
        normalized_value = _normalize_enum_literal(value)
        exact_matches = [
            member
            for member in enum_type
            if normalized_value in _normalized_member_literals(member)
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]

        prefix_matches = [
            member
            for member in enum_type
            if any(
                normalized_value.startswith(candidate)
                or candidate.startswith(normalized_value)
                for candidate in _normalized_member_literals(member)
            )
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

    raise ValueError(
        f"{parameter_name} must be coercible to {enum_type.__name__}; "
        f"got {value!r}."
    )


def _normalized_member_literals(member: Enum) -> tuple[str, ...]:
    return tuple(
        normalized
        for normalized in (
            _normalize_enum_literal(literal)
            for literal in _member_string_literals(member)
        )
        if normalized
    )


def _member_string_literals(member: Enum) -> tuple[str, ...]:
    literals = [member.name]
    if isinstance(member.value, str):
        literals.append(member.value)
    elif isinstance(member.value, tuple):
        literals.extend(
            item
            for item in member.value
            if isinstance(item, str)
        )
    return tuple(literals)


def _normalize_enum_literal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


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


class CellProfilerObjectInputPolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal binding policy for CellProfiler object-label inputs."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "CellProfilerObjectInputPolicy":
        policy_type = cls.__registry__.get(module_name, UnsupportedObjectInputPolicy)
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

    module_name = "IdentifySecondaryObjects"

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

    module_name = "IdentifyTertiaryObjects"

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

    module_name = "MeasureObjectSizeShape"

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
    ) -> dict[str, Any]:
        _require_exact_object_count(module_name, object_inputs, 1)
        return {"labels": adapter.get_objects(object_inputs[0].name).labels}


_SINGLE_LABEL_MEASUREMENT_POLICY_MODULES = (
    "MeasureObjectIntensity",
    "MeasureTexture",
    "MeasureColocalization",
    "MeasureObjectNeighbors",
)


def _declare_single_label_measurement_policy(module_name: str) -> None:
    type(
        f"{module_name}InputPolicy",
        (SingleLabelMeasurementInputPolicy,),
        {
            "__module__": __name__,
            "module_name": module_name,
        },
    )


for _module_name in _SINGLE_LABEL_MEASUREMENT_POLICY_MODULES:
    _declare_single_label_measurement_policy(_module_name)


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


class CellProfilerOutputRecorder(ABC, metaclass=AutoRegisterMeta):
    """Nominal output writer selected by artifact kind."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True
    kind: ClassVar[ArtifactKind | None] = None

    @classmethod
    def for_kind(cls, kind: ArtifactKind) -> "CellProfilerOutputRecorder":
        recorder_type = cls.__registry__.get(kind)
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
            _measurement_table_rows(request.value),
            object_name=_measurement_object_name(
                request.executor.runtime_artifact_inputs
            ),
            source_image_name=request.source_image_name,
        )


class RelationshipsOutputRecorder(CellProfilerOutputRecorder):
    """Record parent-child relationship artifacts."""

    kind = ArtifactKind.RELATIONSHIPS

    def record(self, request: CellProfilerOutputRecordRequest) -> None:
        if not isinstance(request.value, CellProfilerRelationshipPayload):
            raise TypeError(
                f"{request.executor.module_name} relationship output "
                f"'{request.spec.name}' must be CellProfilerRelationshipPayload, "
                f"got {type(request.value).__name__}."
            )
        object_inputs = _specs_of_kind(
            request.executor.runtime_artifact_inputs,
            ArtifactKind.OBJECT_LABELS,
        )
        if len(object_inputs) != 2:
            raise NotImplementedError(
                f"{request.executor.module_name} relationship output "
                f"'{request.spec.name}' requires exactly two object runtime "
                f"inputs, got {[spec.name for spec in object_inputs]}."
            )
        parent_spec, child_spec = object_inputs
        request.adapter.add_relationship(
            request.spec.name,
            parent_object_name=parent_spec.name,
            child_object_name=child_spec.name,
            parent_ids=request.value.parent_ids,
            child_ids=request.value.child_ids,
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
    return _measurement_table_rows(rows)


def _measurement_table_rows(rows: Any) -> list[Any]:
    if isinstance(rows, list):
        return rows
    if isinstance(rows, tuple):
        return list(rows)
    return [rows]


def _measurement_image_for_labels(image: Any, labels: Any) -> Any:
    """Align a measurement reference image to one object-label payload.

    Many absorbed CellProfiler measurement functions expect a 2D intensity image
    paired with one 2D object-label set. When the OpenHCS main flow is carrying a
    higher-level stack for the whole image set, use a single reference slice
    instead of handing the raw multi-slice stack to functions that require shape
    parity with the labels.
    """
    if not hasattr(image, "ndim") or not hasattr(labels, "ndim"):
        return image
    if image.ndim == labels.ndim:
        return image
    if image.ndim == labels.ndim + 1 and getattr(image, "shape", (0,))[0] >= 1:
        return image[0]
    return image


def _measurement_labels(labels: Any) -> Any:
    """Normalize singleton stack labels for absorbed 2D measurement functions."""
    if not hasattr(labels, "ndim"):
        return labels
    if labels.ndim == 3 and getattr(labels, "shape", (0,))[0] == 1:
        return labels[0]
    return labels


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


def _compose_image_payload(
    module_name: str,
    image_payloads: tuple[Any, ...],
) -> Any:
    if not image_payloads:
        raise ValueError(f"{module_name} cannot compose an empty image input set.")
    if len(image_payloads) == 1:
        return image_payloads[0]

    normalized_payloads: list[Any] = []
    for payload in image_payloads:
        if hasattr(payload, "ndim") and payload.ndim == 3:
            if getattr(payload, "shape", (0,))[0] != 1:
                raise NotImplementedError(
                    f"{module_name} cannot compose multi-image inputs from "
                    "payloads that each contain multiple stack slices."
                )
            normalized_payloads.append(payload[0])
            continue
        normalized_payloads.append(payload)

    memory_type = detect_memory_type(normalized_payloads[0])
    return stack_slices(normalized_payloads, memory_type=memory_type, gpu_id=0)


def _single_source_name(source_names: tuple[str, ...]) -> str | None:
    unique_names = tuple(dict.fromkeys(source_names))
    if len(unique_names) == 1:
        return unique_names[0]
    return None


def _payload_slice_count(payload: Any) -> int:
    if hasattr(payload, "ndim") and payload.ndim == 2:
        return 1
    if hasattr(payload, "shape") and payload.shape:
        return int(payload.shape[0])
    return 1
