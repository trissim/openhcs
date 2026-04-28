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
from openhcs.core.config import DtypeConfig
from openhcs.core.memory import detect_memory_type, stack_slices, unstack_slices
from openhcs.core.pipeline.function_contracts import special_input_names_from_callable
from openhcs.core.runtime_adapters import RuntimeAdapterRequest
from openhcs.core.runtime_stores import require_runtime_value_store
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    _aggregate_pure_2d_auxiliary_output,
    _pure_2d_slice_results,
    _rewrite_slice_index,
)

from benchmark.cellprofiler_library import canonical_module_name
from benchmark.cellprofiler_compat.module_contract import CellProfilerModuleContract
from benchmark.cellprofiler_compat.relationship_payload import (
    CellProfilerRelationshipPayload,
)
from benchmark.cellprofiler_compat.runtime_adapter import CellProfilerRuntimeAdapter
from benchmark.converter.contract_inference import InferredContract, infer_contract


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
            func,
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
            fallback_image=image,
            kwargs=kwargs,
        )
        raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
            func,
            invocation.image,
            invocation.kwargs,
            force_full_stack=invocation.image_count > 1,
        )
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
            self._object_input_specs(),
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
        object_inputs = self._object_input_specs()
        measurement_outputs = _specs_of_kind(self.outputs, ArtifactKind.MEASUREMENTS)
        if len(measurement_outputs) != 1:
            raise NotImplementedError(
                f"{self.module_name} per-object execution requires exactly one "
                "measurement output."
            )

        combined_rows: list[Any] = []
        for object_spec in object_inputs:
            raw_labels = self._object_labels(
                object_spec,
                cellprofiler_runtime,
                input_image,
            )
            measurement_labels = _measurement_labels_for_image(
                measurement_source_image,
                raw_labels,
            )
            measurement_image = _measurement_image_for_labels(
                measurement_source_image,
                measurement_labels,
            )
            raw_output = _CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR.execute(
                func,
                measurement_image,
                {**kwargs, "labels": measurement_labels},
            )
            _ignored_main_output, artifact_values = _split_cellprofiler_output(
                raw_output
            )
            combined_rows.extend(_measurement_rows_from_output(artifact_values))

        cellprofiler_runtime.add_measurements(
            measurement_outputs[0].name,
            combined_rows,
            object_name=object_inputs[0].name if len(object_inputs) == 1 else None,
            source_image_name=source_image_name,
        )
        return input_image

    def _object_input_specs(self) -> tuple[ArtifactSpec, ...]:
        return _specs_of_kind(
            self._declared_input_specs(),
            ArtifactKind.OBJECT_LABELS,
        )

    def _object_labels(
        self,
        spec: ArtifactSpec,
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
    ) -> Any:
        if spec.name in self._external_source_object_names():
            return adapter.resolve_source_objects(spec.name, fallback_image).labels
        return adapter.get_objects(spec.name).labels

    def _runtime_input_kwargs(
        self,
        func: Callable[..., Any],
        adapter: CellProfilerRuntimeAdapter,
        fallback_image: Any,
    ) -> dict[str, Any]:
        runtime_inputs = self._special_runtime_inputs(func)
        if not runtime_inputs:
            return {}

        special_input_names = special_input_names_from_callable(func)
        if special_input_names:
            return _bind_special_runtime_inputs(
                self.module_name,
                special_input_names,
                runtime_inputs,
                adapter,
                fallback_image=fallback_image,
                external_image_names=frozenset(self._external_source_image_names()),
                external_object_names=frozenset(
                    self._external_source_object_names()
                ),
                runtime_image_names=frozenset(self._runtime_image_names()),
            )

        unsupported_non_object_inputs = tuple(
            spec
            for spec in runtime_inputs
            if spec.kind is not ArtifactKind.OBJECT_LABELS
        )
        if unsupported_non_object_inputs:
            raise NotImplementedError(
                f"{self.module_name} has runtime inputs "
                f"{[spec.name for spec in unsupported_non_object_inputs]} with "
                "no declared special_inputs binding."
            )

        object_inputs = _specs_of_kind(
            runtime_inputs,
            ArtifactKind.OBJECT_LABELS,
        )
        return CellProfilerObjectInputPolicy.for_module(self.module_name).bind(
            self.module_name,
            object_inputs,
            adapter,
            fallback_image=fallback_image,
            external_object_names=frozenset(self._external_source_object_names()),
        )

    def _special_runtime_inputs(
        self,
        func: Callable[..., Any],
    ) -> tuple[ArtifactSpec, ...]:
        non_image_inputs = tuple(
            spec
            for spec in self._declared_input_specs()
            if spec.kind is not ArtifactKind.IMAGE
        )
        return (
            *non_image_inputs,
            *self._special_image_inputs(func),
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
        func: Callable[..., Any],
        fallback_image: Any,
        adapter: CellProfilerRuntimeAdapter,
    ) -> "CellProfilerImageRequest":
        image_inputs = self._primary_image_inputs(func)
        if not image_inputs:
            return CellProfilerImageRequest(
                payload=fallback_image,
                source_image_name=self._input_source_image_name(adapter),
                image_count=1,
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
            image_count=len(payloads),
        )

    def _primary_image_inputs(
        self,
        func: Callable[..., Any],
    ) -> tuple[ArtifactSpec, ...]:
        image_inputs = _specs_of_kind(
            self._declared_input_specs(),
            ArtifactKind.IMAGE,
        )
        special_image_count = len(self._special_image_inputs(func))
        if special_image_count == 0:
            return image_inputs
        return image_inputs[: len(image_inputs) - special_image_count]

    def _special_image_inputs(
        self,
        func: Callable[..., Any],
    ) -> tuple[ArtifactSpec, ...]:
        image_inputs = _specs_of_kind(
            self._declared_input_specs(),
            ArtifactKind.IMAGE,
        )
        special_input_count = len(special_input_names_from_callable(func))
        non_image_count = len(
            tuple(
                spec
                for spec in self._declared_input_specs()
                if spec.kind is not ArtifactKind.IMAGE
            )
        )
        special_image_count = max(0, special_input_count - non_image_count)
        if special_image_count == 0:
            return ()
        if special_image_count > len(image_inputs):
            raise NotImplementedError(
                f"{self.module_name} declares {special_image_count} image special "
                f"input(s), but only has image inputs {[spec.name for spec in image_inputs]}."
            )
        return image_inputs[-special_image_count:]

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
        for spec in self._declared_input_specs():
            source_name = _artifact_kind_strategy(spec.kind).source_image_name(
                CellProfilerArtifactKindRequest(
                    spec=spec,
                    adapter=adapter,
                    external_image_names=external_image_names,
                    external_object_names=frozenset(
                        self._external_source_object_names()
                    ),
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
        fallback_image: Any,
        kwargs: Mapping[str, Any],
    ) -> "CellProfilerInvocationRequest":
        runtime_kwargs = {
            **kwargs,
            **self._runtime_input_kwargs(func, adapter, fallback_image),
        }
        return CellProfilerInvocationRequest(
            image=image_request.payload,
            kwargs=_coerce_invocation_kwargs(func, runtime_kwargs),
            source_image_name=image_request.source_image_name,
            image_count=image_request.image_count,
        )

    def _external_source_image_names(self) -> tuple[str, ...]:
        runtime_image_names = frozenset(self._runtime_image_names())
        return tuple(
            spec.name
            for spec in _specs_of_kind(
                self._declared_input_specs(),
                ArtifactKind.IMAGE,
            )
            if spec.name not in runtime_image_names
        )

    def _external_source_object_names(self) -> tuple[str, ...]:
        runtime_object_names = frozenset(
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.OBJECT_LABELS,
            )
        )
        return tuple(
            spec.name
            for spec in _specs_of_kind(self.inputs, ArtifactKind.OBJECT_LABELS)
            if spec.name not in runtime_object_names
        )

    def _runtime_image_names(self) -> tuple[str, ...]:
        return tuple(
            spec.name
            for spec in _specs_of_kind(
                self.runtime_artifact_inputs,
                ArtifactKind.IMAGE,
            )
        )

    def _declared_input_specs(self) -> tuple[ArtifactSpec, ...]:
        return _unique_specs((*self.inputs, *self.runtime_artifact_inputs))


@dataclass(frozen=True, slots=True)
class CellProfilerResolvedInputRequest(ABC):
    """Shared source provenance for resolved CellProfiler invocation inputs."""

    source_image_name: str | None
    image_count: int


@dataclass(frozen=True, slots=True)
class CellProfilerImageRequest(CellProfilerResolvedInputRequest):
    """Resolved image payload and source metadata for one module invocation."""

    payload: Any


@dataclass(frozen=True, slots=True)
class CellProfilerInvocationRequest(CellProfilerResolvedInputRequest):
    """Resolved invocation inputs for one CellProfiler function call."""

    image: Any
    kwargs: Mapping[str, Any]


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
        normalized_value = re.sub(
            r"[^a-z0-9]+",
            "_",
            value.strip().lower(),
        ).strip("_")
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
        for literal in _member_string_literals(member)
        if (
            normalized := re.sub(
                r"[^a-z0-9]+",
                "_",
                literal.strip().lower(),
            ).strip("_")
        )
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


@dataclass(frozen=True, slots=True)
class CellProfilerArtifactKindRequest:
    """One artifact-spec request dispatched through a nominal kind strategy."""

    spec: ArtifactSpec
    adapter: CellProfilerRuntimeAdapter
    fallback_image: Any | None = None
    external_image_names: frozenset[str] = frozenset()
    external_object_names: frozenset[str] = frozenset()
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
        if request.spec.name in request.runtime_image_names:
            return request.adapter.get_image(request.spec.name).data
        if request.spec.name in request.external_image_names:
            if request.fallback_image is None:
                raise RuntimeError(
                    f"External image input '{request.spec.name}' requires a "
                    "fallback image payload for source-binding resolution."
                )
            return request.adapter.resolve_source_image(
                request.spec.name,
                request.fallback_image,
            )
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
        if request.spec.name in request.external_object_names:
            if request.fallback_image is None:
                raise RuntimeError(
                    f"External object input '{request.spec.name}' requires a "
                    "fallback image payload for source-binding resolution."
                )
            return _collapse_singleton_label_stack(
                request.adapter.resolve_source_objects(
                    request.spec.name,
                    request.fallback_image,
                ).labels
            )
        return _collapse_singleton_label_stack(
            request.adapter.get_objects(request.spec.name).labels
        )

    def source_image_name(
        self,
        request: CellProfilerArtifactKindRequest,
    ) -> str | None:
        if request.spec.name in request.external_object_names:
            return request.spec.name
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
        policy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            UnsupportedObjectInputPolicy,
        )
        return policy_type()

    @abstractmethod
    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        """Return absorbed-function kwargs for object-label runtime inputs."""


class UnsupportedObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Reject undeclared object-input semantics instead of guessing."""

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        fallback_image: Any,
        external_object_names: frozenset[str],
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
        *,
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        _require_exact_object_count(module_name, object_inputs, 1)
        return {
            "primary_labels": _object_input_labels(
                object_inputs[0],
                adapter,
                fallback_image=fallback_image,
                external_object_names=external_object_names,
            )
        }


class IdentifyTertiaryObjectInputPolicy(CellProfilerObjectInputPolicy):
    """Bind smaller/larger labels to the absorbed tertiary-object signature."""

    module_name = "IdentifyTertiaryObjects"

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        _require_exact_object_count(module_name, object_inputs, 2)
        larger, smaller = object_inputs
        return {
            "primary_labels": _object_input_labels(
                smaller,
                adapter,
                fallback_image=fallback_image,
                external_object_names=external_object_names,
            ),
            "secondary_labels": _object_input_labels(
                larger,
                adapter,
                fallback_image=fallback_image,
                external_object_names=external_object_names,
            ),
        }


class SingleLabelMeasurementInputPolicy(CellProfilerObjectInputPolicy):
    """Bind one object-label input to measurement functions."""

    module_name = "MeasureObjectSizeShape"

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        _require_exact_object_count(module_name, object_inputs, 1)
        return {
            "labels": _object_input_labels(
                object_inputs[0],
                adapter,
                fallback_image=fallback_image,
                external_object_names=external_object_names,
            )
        }


_SINGLE_LABEL_MEASUREMENT_POLICY_MODULES = (
    "MeasureObjectIntensity",
    "MeasureTexture",
    "MeasureColocalization",
    "MeasureGranularity",
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


class OverlayOutlinesInputPolicy(CellProfilerObjectInputPolicy):
    """Bind ordered object outline rows for the generic overlay runner."""

    module_name = "OverlayOutlines"

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        del module_name
        return {
            "object_labels": tuple(
                _object_input_labels(
                    spec,
                    adapter,
                    fallback_image=fallback_image,
                    external_object_names=external_object_names,
                )
                for spec in object_inputs
            )
        }


class ObjectRowsWithMeasurementsInputPolicy(CellProfilerObjectInputPolicy):
    """Bind ordered object rows plus prior measurements for the primary object."""

    def bind(
        self,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        *,
        fallback_image: Any,
        external_object_names: frozenset[str],
    ) -> dict[str, Any]:
        del module_name
        if not object_inputs:
            return {"object_labels": (), "measurement_tables": ()}
        primary_object = object_inputs[0]
        return {
            "object_labels": tuple(
                _object_input_labels(
                    spec,
                    adapter,
                    fallback_image=fallback_image,
                    external_object_names=external_object_names,
                )
                for spec in object_inputs
            ),
            "measurement_tables": adapter.measurement_tables_for_object(
                primary_object.name
            ),
        }


class MeasureImageAreaOccupiedInputPolicy(ObjectRowsWithMeasurementsInputPolicy):
    """Bind ordered object rows for the generic area-occupied runner."""

    module_name = "MeasureImageAreaOccupiedBinary"


class FilterObjectsInputPolicy(ObjectRowsWithMeasurementsInputPolicy):
    """Bind ordered primary/additional object rows for FilterObjects."""

    module_name = "FilterObjects"


class CellProfilerPerObjectMeasurementPolicy:
    """Predicate for modules that need one absorbed call per object set."""

    module_names: ClassVar[tuple[str, ...]] = (
        "MeasureObjectSizeShape",
        "MeasureObjectIntensity",
        "MeasureTexture",
        "MeasureColocalization",
        "MeasureGranularity",
    )

    @classmethod
    def matches(
        cls,
        module_name: str,
        object_inputs: tuple[ArtifactSpec, ...],
    ) -> bool:
        return canonical_module_name(module_name) in cls.module_names and len(
            object_inputs
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
                request.executor._declared_input_specs()
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
            request.executor._declared_input_specs(),
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
    return _collapse_singleton_label_stack(labels)


def _measurement_labels_for_image(image: Any, labels: Any) -> Any:
    """Align object-label payload rank to the selected measurement image."""
    labels = _measurement_labels(labels)
    if not hasattr(image, "ndim") or not hasattr(labels, "ndim"):
        return labels
    if labels.ndim == 3 and image.ndim == 2:
        return labels[0]
    if (
        labels.ndim == 3
        and image.ndim == 3
        and getattr(image, "shape", (0,))[0] == 1
        and labels.shape[1:] == image.shape[1:]
    ):
        return labels[0]
    return labels


def _collapse_singleton_label_stack(labels: Any) -> Any:
    """Normalize singleton OpenHCS label stacks to one CellProfiler label plane."""
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


def _unique_specs(specs: Sequence[ArtifactSpec]) -> tuple[ArtifactSpec, ...]:
    unique: dict[tuple[str, ArtifactKind], ArtifactSpec] = {}
    for spec in specs:
        key = (spec.name, spec.kind)
        existing = unique.get(key)
        if existing is not None and existing != spec:
            raise ValueError(
                f"Conflicting CellProfiler input spec declarations for "
                f"{spec.kind.value}:{spec.name}."
            )
        unique[key] = spec
    return tuple(unique.values())


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


def _object_input_labels(
    spec: ArtifactSpec,
    adapter: CellProfilerRuntimeAdapter,
    *,
    fallback_image: Any,
    external_object_names: frozenset[str],
) -> Any:
    if spec.name in external_object_names:
        return adapter.resolve_source_objects(spec.name, fallback_image).labels
    return adapter.get_objects(spec.name).labels


def _measurement_object_name(
    inputs: tuple[ArtifactSpec, ...],
) -> str | None:
    object_inputs = _specs_of_kind(inputs, ArtifactKind.OBJECT_LABELS)
    if len(object_inputs) == 1:
        return object_inputs[0].name
    return None


def _bind_special_runtime_inputs(
    module_name: str,
    parameter_names: tuple[str, ...],
    runtime_inputs: tuple[ArtifactSpec, ...],
    adapter: CellProfilerRuntimeAdapter,
    *,
    fallback_image: Any,
    external_image_names: frozenset[str],
    external_object_names: frozenset[str],
    runtime_image_names: frozenset[str],
) -> dict[str, Any]:
    if len(parameter_names) != len(runtime_inputs):
        raise NotImplementedError(
            f"{module_name} declares special_inputs {list(parameter_names)}, but "
            f"compiled runtime inputs are {[spec.name for spec in runtime_inputs]}."
        )
    return {
        parameter_name: _runtime_input_value(
            spec,
            adapter,
            fallback_image=fallback_image,
            external_image_names=external_image_names,
            external_object_names=external_object_names,
            runtime_image_names=runtime_image_names,
        )
        for parameter_name, spec in zip(
            parameter_names,
            runtime_inputs,
            strict=True,
        )
    }


def _runtime_input_value(
    spec: ArtifactSpec,
    adapter: CellProfilerRuntimeAdapter,
    *,
    fallback_image: Any,
    external_image_names: frozenset[str],
    external_object_names: frozenset[str],
    runtime_image_names: frozenset[str],
) -> Any:
    try:
        return _artifact_kind_strategy(spec.kind).runtime_input_value(
            CellProfilerArtifactKindRequest(
                spec=spec,
                adapter=adapter,
                fallback_image=fallback_image,
                external_image_names=external_image_names,
                external_object_names=external_object_names,
                runtime_image_names=runtime_image_names,
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


class CellProfilerFunctionContractExecutor:
    """Apply OpenHCS processing contracts after CellProfiler input resolution."""

    def execute(
        self,
        func: Callable[..., Any],
        image: Any,
        kwargs: Mapping[str, Any],
        *,
        force_full_stack: bool = False,
    ) -> Any:
        if force_full_stack:
            return self._execute_pure_3d(func, image, **dict(kwargs))
        return CellProfilerFunctionContractMetadata.from_callable(func).resolve(
            func
        ).execute(
            self,
            func,
            image,
            **dict(kwargs),
        )

    def _execute_pure_3d(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        return func(image, **kwargs)

    def _execute_pure_2d(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        if not hasattr(image, "ndim") or image.ndim == 2:
            return func(image, **kwargs)

        memory_type = detect_memory_type(image)
        slices_2d = unstack_slices(image, memory_type, 0)
        slice_count = len(slices_2d)
        slice_results = [
            _rewrite_slice_index(
                func(
                    slice_2d,
                    **_slice_pure_2d_kwargs(kwargs, slice_index, slice_count),
                ),
                slice_index,
            )
            for slice_index, slice_2d in enumerate(slices_2d)
        ]
        main_outputs, auxiliary_groups = _pure_2d_slice_results(slice_results)
        stacked_main_output = stack_slices(main_outputs, memory_type, 0)
        if not auxiliary_groups:
            return stacked_main_output
        return (
            stacked_main_output,
            *(
                _aggregate_pure_2d_auxiliary_output(values, memory_type)
                for values in auxiliary_groups
            ),
        )

    def _execute_flexible(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        slice_by_slice = bool(kwargs.pop("slice_by_slice", False))
        if slice_by_slice:
            return self._execute_pure_2d(func, image, **kwargs)
        return self._execute_pure_3d(func, image, **kwargs)

    def _execute_volumetric_to_slice(
        self,
        func: Callable[..., Any],
        image: Any,
        **kwargs: Any,
    ) -> Any:
        result_2d = func(image, **kwargs)
        memory_type = detect_memory_type(result_2d)
        return stack_slices([result_2d], memory_type, 0)


@dataclass(frozen=True, slots=True)
class CellProfilerFunctionContractMetadata:
    """Decorator-declared processing contract metadata for one absorbed callable."""

    explicit: ProcessingContract | None
    declared: str | None

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
    ) -> "CellProfilerFunctionContractMetadata":
        metadata = _callable_metadata(func)
        explicit = metadata.get("__processing_contract__")
        declared = metadata.get("__cellprofiler_declared_contract__")
        return cls(
            explicit=explicit if isinstance(explicit, ProcessingContract) else None,
            declared=declared if isinstance(declared, str) else None,
        )

    def resolve(self, func: Callable[..., Any]) -> ProcessingContract:
        if self.explicit is not None:
            return self.explicit
        if self.declared == "unknown":
            inferred = _infer_processing_contract(func)
            if inferred is not None:
                return inferred
        if self.declared is not None:
            declared = _declared_processing_contract(self.declared)
            if declared is not None:
                return declared
        return ProcessingContract.FLEXIBLE


def _callable_metadata(func: Callable[..., Any]) -> Mapping[str, Any]:
    try:
        return vars(func)
    except TypeError:
        return {}


def _infer_processing_contract(
    func: Callable[..., Any],
) -> ProcessingContract | None:
    inferred = infer_contract(func, dtype_config=DtypeConfig()).contract
    if inferred is InferredContract.UNKNOWN or inferred is InferredContract.ERROR:
        return None
    if inferred.name not in ProcessingContract.__members__:
        return None
    return ProcessingContract[inferred.name]


def _declared_processing_contract(
    contract_name: str,
) -> ProcessingContract | None:
    normalized = contract_name.upper()
    if normalized not in ProcessingContract.__members__:
        return None
    return ProcessingContract[normalized]


def _slice_pure_2d_kwargs(
    kwargs: Mapping[str, Any],
    slice_index: int,
    slice_count: int,
) -> dict[str, Any]:
    return {
        name: _slice_pure_2d_value(value, slice_index, slice_count)
        for name, value in kwargs.items()
    }


def _slice_pure_2d_value(value: Any, slice_index: int, slice_count: int) -> Any:
    if not hasattr(value, "ndim") or value.ndim != 3:
        return value
    if value.shape[0] == slice_count:
        return value[slice_index]
    if value.shape[0] == 1:
        return value[0]
    return value


_CELLPROFILER_FUNCTION_CONTRACT_EXECUTOR = CellProfilerFunctionContractExecutor()
