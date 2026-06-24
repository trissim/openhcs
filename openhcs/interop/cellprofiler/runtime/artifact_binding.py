"""Runtime artifact binding authorities for CellProfiler modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from inspect import Parameter, get_annotations, signature
from typing import ClassVar, get_args, get_origin, get_type_hints

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
import numpy as np

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.runtime_values import (
    ImagePayloadMetadataInput,
    ObjectLabelPayload,
    ObjectLabelRepresentation,
    ObjectLabelSet,
    ObjectLabelValue,
    SingletonObjectLabelStackCollapseStrategy,
    image_payload_metadata,
    normalize_image_payload_intensity,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.runtime.runtime_plane_kwargs import (
    CurrentPlaneObjectLabelProjection,
)
from openhcs.interop.cellprofiler.runtime.adapter import (
    CellProfilerRuntimeAdapter,
)
from openhcs.interop.cellprofiler.runtime.current_image_context import (
    CellProfilerOptionalCurrentImageContext,
    CellProfilerRequiredCurrentImageContext,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargs,
    CellProfilerOptionalFunction,
    CellProfilerRuntimeType,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    ArtifactKindRegistryMixin,
    EnumStrategyLabelRegistryMixin,
    NoSourceImageNameMixin,
)


def cellprofiler_image_payload(payload: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue:
    """Return payload in CellProfiler's float image intensity domain."""
    return normalize_image_payload_intensity(payload, dtype=np.float32)


@lru_cache(maxsize=256)
def _callable_parameters(func: CellProfilerFunction) -> Mapping[str, Parameter]:
    return signature(func).parameters


@lru_cache(maxsize=256)
def _callable_type_hints(func: CellProfilerFunction) -> CellProfilerKwargs:
    return get_type_hints(func)


class RuntimeImageInputOrigin(str, Enum):
    """Closed source categories for CellProfiler image artifact inputs."""

    RUNTIME = "runtime"
    EXTERNAL = "external"
    STORED = "stored"


@dataclass(frozen=True, slots=True)
class RuntimeArtifactBindingScope:
    """Contract-derived artifact-name scope for runtime input binding."""

    external_image_names: frozenset[str]
    external_object_names: frozenset[str]
    runtime_image_names: frozenset[str]

    @classmethod
    def from_contract(cls, contract: ModuleArtifactContract) -> "RuntimeArtifactBindingScope":
        """Return artifact binding scope declared by a module contract."""
        return cls(
            external_image_names=frozenset(
                contract.external_input_names(ArtifactKind.IMAGE)
            ),
            external_object_names=frozenset(
                contract.external_input_names(ArtifactKind.OBJECT_LABELS)
            ),
            runtime_image_names=contract.runtime_input_name_set(ArtifactKind.IMAGE),
        )

    def image_origin(self, spec: ArtifactSpec) -> RuntimeImageInputOrigin:
        """Classify one image artifact input within this binding scope."""
        if spec.name in self.runtime_image_names:
            return RuntimeImageInputOrigin.RUNTIME
        if spec.name in self.external_image_names:
            return RuntimeImageInputOrigin.EXTERNAL
        return RuntimeImageInputOrigin.STORED

    def is_external_object(self, spec: ArtifactSpec) -> bool:
        """Return whether an object-label input resolves from source bindings."""
        return spec.name in self.external_object_names


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeArtifactInputRequest(ArtifactSpec, CellProfilerOptionalCurrentImageContext):
    """One artifact-spec request dispatched through a nominal kind strategy."""

    adapter: CellProfilerRuntimeAdapter
    binding_scope: RuntimeArtifactBindingScope

    @classmethod
    def from_spec(
        cls,
        spec: ArtifactSpec,
        *,
        adapter: CellProfilerRuntimeAdapter,
        binding_scope: RuntimeArtifactBindingScope,
        current_image: ImagePayloadMetadataInput | None = None,
    ) -> "RuntimeArtifactInputRequest":
        return cls(
            name=spec.name,
            kind=spec.kind,
            materialization=spec.materialization,
            required=spec.required,
            sidecar_role=spec.sidecar_role,
            adapter=adapter,
            binding_scope=binding_scope,
            current_image=current_image,
        )


class RuntimeArtifactKindStrategy(
    ArtifactKindRegistryMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal strategy family for ArtifactKind-specific runtime semantics."""

    @classmethod
    @lru_cache(maxsize=None)
    def for_kind(cls, kind: ArtifactKind) -> "RuntimeArtifactKindStrategy":
        try:
            strategy_type = cls.__registry__[kind]
        except KeyError as exc:
            raise TypeError(
                f"No CellProfiler artifact kind strategy registered for {kind.value}."
            ) from exc
        return strategy_type()

    @abstractmethod
    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> CellProfilerRuntimeValue:
        """Return the runtime payload bound into absorbed function kwargs."""

    def raw_runtime_input_value(self, request: RuntimeArtifactInputRequest) -> CellProfilerRuntimeValue:
        """Return the runtime payload before CellProfiler intensity coercion."""
        return self.runtime_input_value(request)

    @abstractmethod
    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        """Return the transitive source image name for one artifact input."""

    def source_image_payload(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> CellProfilerRuntimeValue | None:
        """Return an image payload that carries this artifact's source paths."""
        return None

    def cellprofiler_image_number(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> int:
        """Return the CellProfiler ImageNumber associated with this artifact."""
        source_payload = self.source_image_payload(request)
        if source_payload is not None:
            image_number = request.adapter.cellprofiler_image_number_for_payload(
                source_payload
            )
            if image_number is not None:
                return image_number
        source_paths = image_payload_metadata(request.current_image).source_image_paths
        if not source_paths:
            source_paths = request.adapter.cellprofiler_source_paths_for_image_name(
                self.source_image_name(request)
            )
        return request.adapter.cellprofiler_image_number_start_for_source_paths(
            source_paths
        )


class NoSourceImageArtifactKindStrategy(
    NoSourceImageNameMixin,
    RuntimeArtifactKindStrategy,
):
    """Artifact-kind strategy for runtime payloads with no source image owner."""


class ImageArtifactInputOriginStrategy(
    EnumStrategyLabelRegistryMixin[RuntimeImageInputOrigin],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal image artifact resolution selected by binding origin."""

    __enum_member_attr__ = "origin"
    stable_key_axis: ClassVar[str] = "origin"
    origin: ClassVar[RuntimeImageInputOrigin]

    @abstractmethod
    def raw_runtime_input_value(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> CellProfilerRuntimeValue:
        """Return the runtime payload before CellProfiler intensity coercion."""

    @abstractmethod
    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        """Return the transitive source image name for the image artifact."""


class RuntimeImageArtifactInputOriginStrategy(ImageArtifactInputOriginStrategy):
    """Resolve runtime image inputs from the adapter's current-image scope."""

    origin = RuntimeImageInputOrigin.RUNTIME

    def raw_runtime_input_value(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> CellProfilerRuntimeValue:
        return request.adapter.get_image(
            request.name,
            current_image=request.current_image,
        ).data

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        return request.adapter.get_image(
            request.name,
            current_image=request.current_image,
        ).source_image_name


class ExternalImageArtifactInputOriginStrategy(ImageArtifactInputOriginStrategy):
    """Resolve source-bound external image inputs through the current image."""

    origin = RuntimeImageInputOrigin.EXTERNAL

    def raw_runtime_input_value(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> CellProfilerRuntimeValue:
        if request.current_image is None:
            raise RuntimeError(
                f"External image input '{request.name}' requires a "
                "current image payload for source-binding resolution."
            )
        return request.adapter.resolve_source_image(
            request.name,
            request.current_image,
        )

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        return request.name


class StoredImageArtifactInputOriginStrategy(
    NoSourceImageNameMixin,
    ImageArtifactInputOriginStrategy,
):
    """Resolve stored adapter image inputs without source-binding lineage."""

    origin = RuntimeImageInputOrigin.STORED

    def raw_runtime_input_value(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> CellProfilerRuntimeValue:
        return request.adapter.get_image(request.name).data


class ImageArtifactKindStrategy(RuntimeArtifactKindStrategy):
    """Resolve image artifact payloads and source-image lineage."""

    kind = ArtifactKind.IMAGE

    def raw_runtime_input_value(self, request: RuntimeArtifactInputRequest) -> CellProfilerRuntimeValue:
        return ImageArtifactInputOriginStrategy.for_enum_member(
            request.binding_scope.image_origin(request)
        ).raw_runtime_input_value(request)

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> CellProfilerRuntimeValue:
        return cellprofiler_image_payload(self.raw_runtime_input_value(request))

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        return ImageArtifactInputOriginStrategy.for_enum_member(
            request.binding_scope.image_origin(request)
        ).source_image_name(request)

    def source_image_payload(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> CellProfilerRuntimeValue | None:
        return self.raw_runtime_input_value(request)


class ObjectLabelsArtifactKindStrategy(RuntimeArtifactKindStrategy):
    """Resolve object-label payloads and lineage."""

    kind = ArtifactKind.OBJECT_LABELS

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> CellProfilerRuntimeValue:
        if request.binding_scope.is_external_object(request):
            if request.current_image is None:
                raise RuntimeError(
                    f"External object input '{request.name}' requires a "
                    "current image payload for source-binding resolution."
                )
            payload = _object_label_runtime_payload(
                request.adapter.resolve_source_objects(
                    request.name,
                    request.current_image,
                )
            )
            return SingletonObjectLabelStackCollapseStrategy.for_labels(payload).collapse(
                payload
            )
        payload = _object_label_runtime_payload(
            request.adapter.get_objects(
                request.name,
                current_image=request.current_image,
            )
        )
        return SingletonObjectLabelStackCollapseStrategy.for_labels(payload).collapse(
            payload
        )

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        if request.binding_scope.is_external_object(request):
            return request.name
        return request.adapter.get_objects(
            request.name,
            current_image=request.current_image,
        ).source_image_name

    def source_image_payload(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> CellProfilerRuntimeValue | None:
        source_image_name = self.source_image_name(request)
        if source_image_name is None:
            return None
        return request.adapter.source_image_payload_for_name(
            source_image_name,
            request.current_image,
        )


class MeasurementsArtifactKindStrategy(RuntimeArtifactKindStrategy):
    """Resolve measurement payloads and lineage."""

    kind = ArtifactKind.MEASUREMENTS

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> CellProfilerRuntimeValue:
        return request.adapter.get_measurements(
            request.name,
            current_image=request.current_image,
        ).rows

    def source_image_name(
        self,
        request: RuntimeArtifactInputRequest,
    ) -> str | None:
        return request.adapter.get_measurements(
            request.name,
            current_image=request.current_image,
        ).source_image_name


class RelationshipsArtifactKindStrategy(NoSourceImageArtifactKindStrategy):
    """Resolve relationship payloads."""

    kind = ArtifactKind.RELATIONSHIPS

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> CellProfilerRuntimeValue:
        return request.adapter.get_relationship(
            request.name,
            current_image=request.current_image,
        )

class SpatialGridArtifactKindStrategy(NoSourceImageArtifactKindStrategy):
    """Resolve spatial-grid payloads."""

    kind = ArtifactKind.SPATIAL_GRID

    def runtime_input_value(self, request: RuntimeArtifactInputRequest) -> CellProfilerRuntimeValue:
        return request.adapter.get_spatial_grid(request.name)

@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeInputBindingRequestBase(
    CellProfilerRequiredCurrentImageContext,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Shared runtime context for artifact-backed runtime-input binding."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    registry_key: ClassVar[str | None] = None

    module_name: str
    func: CellProfilerOptionalFunction = None
    adapter: CellProfilerRuntimeAdapter
    kwargs: CellProfilerKwargs
    binding_scope: RuntimeArtifactBindingScope
    project_object_labels_to_current_plane: bool = True

    def label_domain_payload_for(self, spec: ArtifactSpec) -> CellProfilerRuntimeValue:
        """Return object labels retaining nominal domain metadata."""
        return (
            self.current_plane_label_payload_for(spec)
            if self.project_object_labels_to_current_plane
            else self.label_payload_for(spec)
        )

    def labels_for(self, spec: ArtifactSpec) -> CellProfilerRuntimeValue:
        payload = self.label_domain_payload_for(spec)
        labels = (
            object_label_dense_array(payload)
            if isinstance(payload, (ObjectLabelPayload, ObjectLabelSet))
            else payload
        )
        return SingletonObjectLabelStackCollapseStrategy.for_labels(labels).collapse(
            labels
        )

    def label_payload_for(self, spec: ArtifactSpec) -> CellProfilerRuntimeValue:
        if self.binding_scope.is_external_object(spec):
            return self.adapter.resolve_source_objects(
                spec.name,
                self.current_image,
            )
        current_image = (
            self.current_image if self.project_object_labels_to_current_plane else None
        )
        return self.adapter.get_objects(
            spec.name,
            current_image=current_image,
        )

    def current_plane_label_payload_for(self, spec: ArtifactSpec) -> CellProfilerRuntimeValue:
        """Return object labels projected to the invocation's current plane."""
        payload = self.label_payload_for(spec)
        if not isinstance(payload, ObjectLabelValue):
            return payload
        plane_index = payload.plane_axis.plane_index(
            self.adapter,
            source_aliases=payload.source_alias_group,
        )
        if plane_index is None:
            return payload
        return CurrentPlaneObjectLabelProjection(
            value=payload,
            plane_index=plane_index,
            plane_axis=payload.plane_axis,
        ).projected_value()

    def label_argument_for(self, spec: ArtifactSpec, parameter_name: str) -> CellProfilerRuntimeValue:
        """Return semantic labels only when the callable declares that contract."""
        value = self.label_payload_for(spec)
        if (
            self.func is not None
            and CallableObjectLabelInputContract(
                self.func,
                parameter_name,
            ).accepts_native_value
        ):
            return value
        return self.labels_for(spec)

    def current_plane_relationship_for(self, spec: ArtifactSpec) -> CellProfilerRuntimeValue:
        """Return a relationship payload projected to the invocation plane."""
        relationship = self.adapter.get_relationship(
            spec.name,
            current_image=self.current_image,
        )
        if not self.project_object_labels_to_current_plane:
            return relationship
        plane_index = self.relationship_runtime_slice_index()
        if plane_index is None:
            return relationship
        return relationship.project_runtime_slice(plane_index)

    def relationship_runtime_slice_index(self) -> int | None:
        """Return the relationship row slice index for true runtime-slice groups."""
        plane_index = self.adapter.runtime_slice_plane_index()
        if plane_index is None:
            return None
        if self.object_inputs_projected_by_source_binding_axis():
            return None
        return plane_index

    def object_inputs_projected_by_source_binding_axis(self) -> bool:
        """Return whether object inputs are already scoped by source binding."""
        for spec in self.object_inputs:
            labels = self.adapter.get_objects(spec.name)
            aliases = labels.source_alias_group
            if labels.is_composed_source_axis_component:
                return True
            if not aliases:
                continue
            source_plane_index = self.adapter.source_binding_axis_plane_index(aliases)
            if source_plane_index is not None:
                return True
        return False

    def artifact_input_request(self, spec: ArtifactSpec) -> RuntimeArtifactInputRequest:
        """Return the nominal artifact request for this binding context."""
        return RuntimeArtifactInputRequest.from_spec(
            spec,
            adapter=self.adapter,
            current_image=self.current_image,
            binding_scope=self.binding_scope,
        )

def _object_label_runtime_payload(objects: ObjectLabelSet) -> CellProfilerRuntimeValue:
    if objects.representation is ObjectLabelRepresentation.SPARSE_IJV:
        return objects
    return objects.runtime_payload()


@dataclass(frozen=True, slots=True)
class CallableObjectLabelInputContract:
    """Callable-declared object-label input category for runtime binding."""

    func: CellProfilerFunction
    parameter_name: str

    @property
    def accepts_native_value(self) -> bool:
        annotation = self.parameter_annotation()
        return self.annotation_accepts_type(annotation, ObjectLabelValue)

    def parameter_annotation(self) -> CellProfilerRuntimeValue:
        try:
            hints = get_type_hints(self.func)
        except (NameError, TypeError):
            hints = get_annotations(self.func, eval_str=False)
        return hints.get(self.parameter_name)

    @classmethod
    def annotation_accepts_type(cls, annotation: CellProfilerRuntimeValue, value_type: CellProfilerRuntimeType) -> bool:
        if annotation is value_type:
            return True
        if isinstance(annotation, type) and issubclass(annotation, value_type):
            return True
        origin = get_origin(annotation)
        if origin is None:
            return False
        return any(
            cls.annotation_accepts_type(argument, value_type)
            for argument in get_args(annotation)
        )
