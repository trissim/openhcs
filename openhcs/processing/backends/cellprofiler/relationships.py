"""Relationship backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import ClassVar

from openhcs.core.alias_property import AliasProperty

from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.callable_contract import CallableContract
from openhcs.core.pipeline.function_contracts import runtime_bound_parameters
from openhcs.core.runtime_invocation import SliceIndexRuntimeParameter
from openhcs.interop.cellprofiler.runtime.bound_parameters import RuntimeBoundParameterName
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargDict
from openhcs.interop.cellprofiler.runtime.special_input_policies import (
    CellProfilerSpecialInputPolicyMixin,
    SpecialInputBindingRequest,
)
from openhcs.processing.backends.cellprofiler.module_classes import (
    ArtifactContractModule,
    BoundModuleSettings,
    CellProfilerModule,
    PlaneRuntimeArtifactModule,
    RelationshipDebugViewModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_value_setting_parser,
    parse_cellprofiler_bool,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import cellprofiler_enum_from_literal
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    ColumnarFieldsMeasurementRecordMixin,
    NoObjectNameMeasurementRecordMixin,
    RelationshipMeasurementRecordRowsMixin,
    SourceNameOnlyMeasurementRecordMixin,
    TableMeasurementRecordRowsMixin,
)
from openhcs.interop.cellprofiler.runtime.relationship_endpoints import (
    RelationshipEndpointContract,
    RelationshipEndpointResolver,
)


class RelateObjectsDistanceMethod(Enum):
    """CellProfiler RelateObjects child-parent distance calculation mode."""

    NONE = ("none", False, False)
    CENTROID = ("centroid", True, False)
    MINIMUM = ("minimum", False, True)
    BOTH = ("both", True, True)

    def __init__(
        self,
        label: str,
        calculates_centroid_distance: bool,
        calculates_minimum_distance: bool,
    ) -> None:
        self._value_ = label
        self._calculates_centroid_distance = calculates_centroid_distance
        self._calculates_minimum_distance = calculates_minimum_distance

    calculates_centroid_distance = AliasProperty[bool]("_calculates_centroid_distance")
    calculates_minimum_distance = AliasProperty[bool]("_calculates_minimum_distance")


DistanceMethod = RelateObjectsDistanceMethod


class PrimaryObjectInputRelationshipModule(ABC):
    """Relationship module declaration with indexed primary endpoints."""

    primary_relationship_object_input_indices: ClassVar[tuple[int, int]] = (0, 1)
    primary_relationship_output_index: ClassVar[int] = 0

    @classmethod
    def relationship_endpoint_contract(
        cls,
        resolver: RelationshipEndpointResolver,
        relationship_spec: ArtifactSpec,
    ) -> RelationshipEndpointContract | None:
        if relationship_spec != resolver.relationship_output_at(
            cls.primary_relationship_output_index
        ):
            return None
        return resolver.indexed_object_input_contract(
            cls.primary_relationship_object_input_indices
        )


class PrimaryObjectInputRelationshipDistanceModule(
    PrimaryObjectInputRelationshipModule,
    ABC,
):
    """Relationship module declaration whose primary relationship owns distances."""

    @classmethod
    def relationship_distance_measurements_apply(
        cls,
        resolver: RelationshipEndpointResolver,
        relationship_spec: ArtifactSpec,
    ) -> bool:
        return relationship_spec == resolver.relationship_output_at(
            cls.primary_relationship_output_index
        )

class RelateObjectsSpecialInputPolicy(CellProfilerSpecialInputPolicyMixin):
    """Bind parent/child object labels in the current runtime plane."""

    slice_index_parameter_type: ClassVar[type[SliceIndexRuntimeParameter]] = (
        SliceIndexRuntimeParameter
    )
    slice_index_kwarg: ClassVar[RuntimeBoundParameterName] = RuntimeBoundParameterName(
        SliceIndexRuntimeParameter.require_parameter_name()
    )

    def extra_bound_parameter_names(
        self,
        plan: CellProfilerModuleRuntimePlan,
    ) -> tuple[str, ...]:
        """Return the optional runtime slice index kwarg."""
        if self.slice_index_parameter_type in plan.callable_contract.runtime_bound_parameter_types:
            return (self.slice_index_kwarg,)
        return ()

    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerKwargDict:
        if len(request.parameter_names) != len(request.special_input_specs):
            raise NotImplementedError(
                f"{request.module_name} declares special_inputs "
                f"{list(request.parameter_names)}, but compiled runtime inputs are "
                f"{[spec.name for spec in request.special_input_specs]}."
            )
        bound = {
            parameter_name: (
                request.current_plane_object_label_runtime_value(spec)
                if spec.kind is ArtifactKind.OBJECT_LABELS
                else request.runtime_value(spec)
            )
            for parameter_name, spec in zip(
                request.parameter_names,
                request.special_input_specs,
                strict=True,
            )
        }
        plane_index = request.relationship_runtime_slice_index()
        if (
            plane_index is not None
            and request.func is not None
            and self.slice_index_parameter_type
            in CallableContract.from_callable(request.func).runtime_bound_parameter_types
        ):
            if self.slice_index_kwarg not in bound:
                bound[self.slice_index_kwarg] = plane_index
        return bound


class RelateObjectsModule(
    PlaneRuntimeArtifactModule,
    TableMeasurementRecordRowsMixin,
    RelationshipMeasurementRecordRowsMixin,
    NoObjectNameMeasurementRecordMixin,
    SourceNameOnlyMeasurementRecordMixin,
    ColumnarFieldsMeasurementRecordMixin,
    RelateObjectsSpecialInputPolicy,
    RelationshipDebugViewModule,
    PrimaryObjectInputRelationshipDistanceModule,
    CellProfilerModule,
):
    module_name = 'RelateObjects'
    function_name = 'relate_objects'
    validated = True
    confidence = 1.0
    distance_setting = SettingNameFamily("Calculate child-parent distances?")
    parent_objects_setting = SettingNameFamily(
        "Select the parent objects",
        aliases=("Parent objects",),
    )
    child_objects_setting = SettingNameFamily(
        "Select the child objects",
        aliases=("Child objects",),
    )
    per_parent_means_setting = SettingNameFamily(
        "Calculate per-parent means for all child measurements?"
    )
    save_children_setting = SettingNameFamily(
        "Do you want to save the children with parents as a new object set?"
    )

    @classmethod
    def relationship_measurement_rows(cls, request):
        """Return RelateObjects relationship rows including distance features."""
        return RelateObjectsRelationshipMeasurementRows(request)

    ignored_settings = (
        distance_setting,
        parent_objects_setting,
        child_objects_setting,
        per_parent_means_setting,
        "Calculate distances to other parents?",
        "Parent name",
        save_children_setting,
        "Name the output object",
    )
    setting_bindings = (
        SettingToKeywordBinding(
            distance_setting,
            "calculate_distances",
            cellprofiler_enum_value_setting_parser(RelateObjectsDistanceMethod),
        ),
        SettingToKeywordBinding(
            per_parent_means_setting,
            "calculate_per_parent_means",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            save_children_setting,
            "save_children_with_parents",
            parse_cellprofiler_bool,
        ),
    )

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
        from openhcs.core.runtime_semantics import parent_child_relationship_artifact_name

        parent = builder.require_artifact(
            ArtifactSpec(required_setting_value(module, cls.parent_objects_setting), ArtifactKind.OBJECT_LABELS),
            module,
        )
        child = builder.require_artifact(
            ArtifactSpec(required_setting_value(module, cls.child_objects_setting), ArtifactKind.OBJECT_LABELS),
            module,
        )
        outputs = [
            builder.declare_artifact(ArtifactSpec(parent_child_relationship_artifact_name(parent.name, child.name), ArtifactKind.RELATIONSHIPS), module),
            builder.declare_artifact(ArtifactSpec(cls.measurement_artifact_name(module), ArtifactKind.MEASUREMENTS), module),
        ]
        save_children = optional_setting_value(module, cls.save_children_setting)
        if save_children is not None and save_children.strip().lower() == "yes":
            output_objects = builder.declare_artifact(
                ArtifactSpec(required_setting_value(module, "Name the output object"), ArtifactKind.OBJECT_LABELS),
                module,
            )
            outputs.insert(0, output_objects)
            outputs.insert(
                2,
                builder.declare_artifact(
                    ArtifactSpec(parent_child_relationship_artifact_name(child.name, output_objects.name), ArtifactKind.RELATIONSHIPS),
                    module,
                ),
            )
        return assembler.assemble_contract(module, builder, inputs=[parent, child], outputs=outputs)



from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.memory import stack_slices
from openhcs.core.memory.decorators import numpy as numpy_decorator
from openhcs.core.pipeline.function_contracts import (
    special_inputs,
    special_outputs,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import GeneratedEnumClassSpec
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NAME_FIELD,
)
from openhcs.core.runtime_invocation import RuntimeOutputBundle
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.interop.cellprofiler.relationship_measurements import (
    RelationshipMeasurements,
)
from openhcs.interop.cellprofiler.runtime.mapping_lookup import MappingValueLookup
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    FormattingMeasurementFeatureTemplate,
)
from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
    CellProfilerRelationshipMeasurementPayloads,
    RelationshipDistanceRowTuple,
    RelationshipMeasurementRowList,
    RelationshipMeasurementRows,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.core.runtime_semantics import (
    DenseObjectLabelPairAligner,
    ObjectRelationshipPayloadKernel,
    ParentChildRelationshipPayload,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.core.runtime_values import object_label_value_with_dense_labels
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_dataclass_materializer


for _relate_objects_measurement_feature_template_spec in (
    GeneratedEnumClassSpec(
        class_name="RelateObjectsRelationshipMeasurementFeature",
        base_type=FormattingMeasurementFeatureTemplate,
        members={
            "DISTANCE_CENTROID": "Distance_Centroid_{parent_object_name}",
            "DISTANCE_MINIMUM": "Distance_Minimum_{parent_object_name}",
            "MEAN_CHILD": "Mean_{child_object_name}_{child_feature_name}",
        },
    ),
):
    _relate_objects_measurement_feature_template_spec.declare_in(globals())


class ObjectRelationshipBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ObjectRelationshipPayloadKernel,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Object relationship operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def relate_children_to_parents(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        child_count: int,
    ) -> np.ndarray:
        """Assign each child to its parent object."""

    @abstractmethod
    def centroid_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        """Return child-parent centroid distances."""

    @abstractmethod
    def minimum_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        """Return child-centroid to parent-boundary distances."""

    @abstractmethod
    def label_centers(self, labels: np.ndarray) -> np.ndarray:
        """Return row/column centers indexed by dense positive label id."""

    def parent_child_payload_from_labels(
        self,
        parent_labels: Any,
        child_labels: Any,
    ) -> ParentChildRelationshipPayload:
        """Return parent-child ids using the labels' nominal representation."""
        return object_label_parent_child_payload(
            parent_labels,
            child_labels,
            kernel=self,
        )

    def parents_of_from_payload(
        self,
        payload: ParentChildRelationshipPayload,
        child_count: int,
    ) -> np.ndarray:
        """Return a dense parents-of-child vector from a relationship payload."""
        parents_of = np.zeros(child_count, dtype=np.int32)
        for parent_id, child_id in zip(
            payload.parent_ids,
            payload.child_ids,
            strict=True,
        ):
            if 0 < child_id <= child_count:
                parents_of[child_id - 1] = int(parent_id)
        return parents_of


class RelateObjectsRelationshipMeasurementRows(RelationshipMeasurementRows):
    """RelateObjects additionally projects configured child-parent distances."""

    def rows(self) -> RelationshipMeasurementRowList:
        rows: RelationshipMeasurementRowList = list(super().rows())
        endpoint_resolver = RelationshipEndpointResolver.for_request(self.request)
        for relationship_spec, payload in self.output_entries():
            endpoint_contract = endpoint_resolver.endpoint_contract(
                relationship_spec
            )
            if not endpoint_resolver.distance_measurements_apply(relationship_spec):
                continue
            rows.extend(
                self.distance_rows(
                    parent_object_name=endpoint_contract.parent.name,
                    child_object_name=endpoint_contract.child.name,
                    payload=payload,
                )
            )
        return rows

    def distance_rows(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        payload: ParentChildRelationshipPayload,
    ) -> RelationshipDistanceRowTuple:
        if not self.distance_measurements_declared():
            return ()
        sliced_pairs = self.payload_pairs_by_slice(
            payload,
            child_object_name=child_object_name,
        )
        if sliced_pairs is not None:
            slice_count = len(sliced_pairs)
            rows: RelationshipMeasurementRowList = []
            for slice_index, pairs in sliced_pairs:
                rows.extend(
                    self.distance_rows_for_pairs(
                        parent_object_name=parent_object_name,
                        child_object_name=child_object_name,
                        pairs=pairs,
                        slice_index=slice_index,
                        slice_count=slice_count,
                    )
                )
            return tuple(rows)
        return self.distance_rows_for_pairs(
            parent_object_name=parent_object_name,
            child_object_name=child_object_name,
            pairs=tuple(
                (int(parent_id), int(child_id))
                for parent_id, child_id in zip(
                    payload.parent_ids,
                    payload.child_ids,
                    strict=True,
                )
            ),
            slice_index=None,
        )

    def distance_measurements_declared(self) -> bool:
        return (
            CellProfilerRelationshipMeasurementPayloads
            .from_value(self.request.output_value)
            .declares_distance_measurements
        )

    def per_parent_distance_means_enabled(self) -> bool:
        value = MappingValueLookup(
            self.request.call_kwargs,
            "calculate_per_parent_means",
        ).value_or(False)
        return bool(value)

    def distance_rows_for_pairs(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        pairs: tuple[tuple[int, int], ...],
        slice_index: int | None,
        slice_count: int | None = None,
    ) -> RelationshipDistanceRowTuple:
        if not pairs:
            return ()
        parent_labels = self.object_labels(
            parent_object_name,
            slice_index=slice_index,
            slice_count=slice_count,
        )
        child_labels = self.object_labels(
            child_object_name,
            slice_index=slice_index,
            slice_count=slice_count,
        )
        if parent_labels is None or child_labels is None:
            return ()
        parent_array = RuntimeSliceProjection.object_label_endpoint_dense_array(
            parent_labels,
            dtype=np.int32,
        )
        child_array = RuntimeSliceProjection.object_label_endpoint_dense_array(
            child_labels,
            dtype=np.int32,
        )
        parent_array, child_array = DenseObjectLabelPairAligner(
            parent_array,
            child_array,
        ).aligned()
        parent_count = 0
        if child_array.size:
            parent_count = int(child_array.max())
        parents_of = np.zeros(parent_count, dtype=np.int32)
        for parent_id, child_id in pairs:
            if 0 < child_id <= len(parents_of):
                parents_of[child_id - 1] = parent_id
        backend = ObjectRelationshipBackendStrategy.for_memory_type()
        centroid_distances = backend.centroid_distances(
            parent_array,
            child_array,
            parents_of,
        )
        minimum_distances = backend.minimum_distances(
            parent_array,
            child_array,
            parents_of,
        )
        centroid_feature = (
            RelateObjectsRelationshipMeasurementFeature.DISTANCE_CENTROID.feature_name(
                parent_object_name=parent_object_name
            )
        )
        minimum_feature = (
            RelateObjectsRelationshipMeasurementFeature.DISTANCE_MINIMUM.feature_name(
                parent_object_name=parent_object_name
            )
        )
        child_distance_rows = tuple(
            self.axis_qualified_row(
                {
                    MEASUREMENT_OBJECT_NAME_FIELD: child_object_name,
                    MEASUREMENT_OBJECT_LABEL_FIELD: child_id,
                    centroid_feature: float(centroid_distances[child_id - 1]),
                    minimum_feature: float(minimum_distances[child_id - 1]),
                },
                slice_index=slice_index,
            )
            for _parent_id, child_id in pairs
            if 0 < child_id <= len(parents_of)
        )
        if not self.per_parent_distance_means_enabled():
            return child_distance_rows
        return (
            *child_distance_rows,
            *self.parent_mean_distance_rows(
                parent_object_name=parent_object_name,
                child_object_name=child_object_name,
                pairs=pairs,
                centroid_distances=centroid_distances,
                minimum_distances=minimum_distances,
                slice_index=slice_index,
            ),
        )

    def parent_mean_distance_rows(
        self,
        *,
        parent_object_name: str,
        child_object_name: str,
        pairs: tuple[tuple[int, int], ...],
        centroid_distances: np.ndarray,
        minimum_distances: np.ndarray,
        slice_index: int | None,
    ) -> RelationshipDistanceRowTuple:
        distances_by_parent: dict[int, list[tuple[float, float]]] = {}
        for parent_id, child_id in pairs:
            if child_id <= 0 or child_id > len(centroid_distances):
                continue
            if parent_id not in distances_by_parent:
                distances_by_parent[parent_id] = []
            distances_by_parent[parent_id].append(
                (
                    float(centroid_distances[child_id - 1]),
                    float(minimum_distances[child_id - 1]),
                )
            )
        centroid_feature = (
            RelateObjectsRelationshipMeasurementFeature.MEAN_CHILD.feature_name(
                child_object_name=child_object_name,
                child_feature_name="Distance_Centroid",
            )
        )
        minimum_feature = (
            RelateObjectsRelationshipMeasurementFeature.MEAN_CHILD.feature_name(
                child_object_name=child_object_name,
                child_feature_name="Distance_Minimum",
            )
        )
        return tuple(
            self.axis_qualified_row(
                {
                    MEASUREMENT_OBJECT_NAME_FIELD: parent_object_name,
                    MEASUREMENT_OBJECT_LABEL_FIELD: parent_id,
                    centroid_feature: float(
                        np.mean([value[0] for value in distances])
                    ),
                    minimum_feature: float(
                        np.mean([value[1] for value in distances])
                    ),
                },
                slice_index=slice_index,
            )
            for parent_id, distances in sorted(distances_by_parent.items())
            if distances
        )


class NumbaNumpyObjectRelationshipBackendStrategy(
    ObjectRelationshipBackendStrategy
):
    """Numba-accelerated NumPy object relationship primitives."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        parent_labels = np.array([[1, 1, 0], [0, 2, 2], [0, 0, 0]], dtype=np.int32)
        child_labels = np.array([[1, 0, 0], [0, 2, 2], [0, 0, 3]], dtype=np.int32)
        parents_of = self.relate_children_to_parents(parent_labels, child_labels, 3)
        self.label_centers(parent_labels)
        self.centroid_distances(parent_labels, child_labels, parents_of)
        self.minimum_distances(parent_labels, child_labels, parents_of)

    def relate_children_to_parents(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        child_count: int,
    ) -> np.ndarray:
        parent_count = int(parent_labels.max()) if parent_labels.max() > 0 else 0
        parents_of = np.zeros(child_count, dtype=np.int32)
        if child_count == 0 or parent_count == 0:
            return parents_of

        return _relate_children_to_parents_numba(
            np.asarray(parent_labels),
            np.asarray(child_labels),
            child_count,
            parent_count,
        )

    def relate_sparse_ijv_children_to_parents(
        self,
        parent_rows: np.ndarray,
        child_rows: np.ndarray,
        child_count: int,
        parent_count: int,
    ) -> np.ndarray:
        if child_count == 0 or parent_count == 0:
            return np.zeros(child_count, dtype=np.int32)
        return _relate_sparse_ijv_children_to_parents_numba(
            np.asarray(parent_rows, dtype=np.int64),
            np.asarray(child_rows, dtype=np.int64),
            child_count,
            parent_count,
        )

    def centroid_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        parent_count = int(parent_labels.max())
        return _calculate_centroid_distances_numba(
            np.ascontiguousarray(parent_labels),
            np.ascontiguousarray(child_labels),
            np.asarray(parents_of, dtype=np.int32),
            parent_count,
        )

    def minimum_distances(
        self,
        parent_labels: np.ndarray,
        child_labels: np.ndarray,
        parents_of: np.ndarray,
    ) -> np.ndarray:
        parent_count = int(parent_labels.max())
        return _calculate_minimum_distances_numba(
            np.ascontiguousarray(parent_labels),
            np.ascontiguousarray(child_labels),
            np.asarray(parents_of, dtype=np.int32),
            parent_count,
        )

    def label_centers(self, labels: np.ndarray) -> np.ndarray:
        if labels.ndim > 2:
            labels = np.max(labels, axis=tuple(range(labels.ndim - 2)))
        label_count = int(labels.max())
        if label_count == 0:
            return np.empty((0, 2), dtype=np.float64)
        centroids = _label_centroids_numba(
            np.ascontiguousarray(labels),
            label_count,
        )
        return centroids[1:]


@dataclass(frozen=True, slots=True)
class RelateObjectsResult(RuntimeOutputBundle):
    """Nominal result bundle emitted by RelateObjects."""

    output_labels: np.ndarray
    parent_child_relationship: ParentChildRelationshipPayload
    relationship_measurements: RelationshipMeasurements
    saved_child_relationship: ParentChildRelationshipPayload | None = None

    def as_runtime_tuple(self) -> tuple[
        np.ndarray,
        ParentChildRelationshipPayload,
        RelationshipMeasurements,
    ] | tuple[
        np.ndarray,
        ParentChildRelationshipPayload,
        ParentChildRelationshipPayload,
        RelationshipMeasurements,
    ]:
        """Lower to the current positional function-contract ABI."""
        if self.saved_child_relationship is None:
            return (
                self.output_labels,
                self.parent_child_relationship,
                self.relationship_measurements,
            )
        return (
            self.output_labels,
            self.parent_child_relationship,
            self.saved_child_relationship,
            self.relationship_measurements,
        )

    def __iter__(self):
        """Preserve direct tuple-unpacking compatibility for function tests."""
        return iter(self.as_runtime_tuple())


def _combine_parent_child_payloads(
    payloads: tuple[ParentChildRelationshipPayload | None, ...],
) -> ParentChildRelationshipPayload | None:
    """Combine per-slice relationship payloads while preserving slice identity."""
    present_payloads = tuple(payload for payload in payloads if payload is not None)
    if not present_payloads:
        return None

    parent_ids: list[int] = []
    child_ids: list[int] = []
    slice_indices: list[int] = []
    for fallback_slice_index, payload in enumerate(payloads):
        if payload is None:
            continue
        parent_ids.extend(payload.parent_ids)
        child_ids.extend(payload.child_ids)
        if payload.slice_indices:
            slice_indices.extend(payload.slice_indices)
        else:
            slice_indices.extend([fallback_slice_index] * len(payload.parent_ids))

    return ParentChildRelationshipPayload(
        parent_ids=tuple(parent_ids),
        child_ids=tuple(child_ids),
        slice_indices=tuple(slice_indices),
        slice_count=len(payloads),
    )


def _aggregate_relate_objects_slice_results(
    slice_results: tuple[RelateObjectsResult, ...],
    *,
    memory_type: str,
) -> RelateObjectsResult:
    """Aggregate manual 2D RelateObjects slices for object-only stack inputs."""
    output_labels = stack_slices(
        [
            object_label_dense_array(result.output_labels, dtype=np.float32)
            for result in slice_results
        ],
        memory_type,
        0,
    )
    parent_child_relationship = _combine_parent_child_payloads(
        tuple(result.parent_child_relationship for result in slice_results)
    )
    if parent_child_relationship is None:
        parent_child_relationship = ParentChildRelationshipPayload(
            parent_ids=(),
            child_ids=(),
            slice_count=len(slice_results),
        )
    saved_child_relationship = _combine_parent_child_payloads(
        tuple(result.saved_child_relationship for result in slice_results)
    )
    return RelateObjectsResult(
        output_labels=output_labels,
        parent_child_relationship=parent_child_relationship,
        relationship_measurements=tuple(
            result.relationship_measurements for result in slice_results
        ),
        saved_child_relationship=saved_child_relationship,
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@runtime_bound_parameters(SliceIndexRuntimeParameter)
@special_inputs("parent_labels", "child_labels")
@special_outputs(
    (
        "relationship_measurements",
        csv_dataclass_materializer(
            RelationshipMeasurements,
            analysis_type="relate_objects",
        ),
    )
)
def relate_objects(
    image: np.ndarray,
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    calculate_distances: RelateObjectsDistanceMethod | str = RelateObjectsDistanceMethod.BOTH,
    calculate_per_parent_means: bool = False,
    save_children_with_parents: bool = False,
    relationship_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int | None = None,
) -> RelateObjectsResult:
    """Relate CellProfiler child objects to parent objects by spatial overlap."""
    parent_array = object_label_dense_array(parent_labels, dtype=np.int32)
    child_array = object_label_dense_array(child_labels, dtype=np.int32)
    if (
        slice_index is None
        and np.asarray(image).ndim == 2
        and parent_array.ndim == 3
        and child_array.ndim == 3
        and parent_array.shape[0] == child_array.shape[0]
        and parent_array.shape[-2:] == np.asarray(image).shape
        and child_array.shape[-2:] == np.asarray(image).shape
    ):
        slice_results = tuple(
            relate_objects(
                image,
                parent_array[index],
                child_array[index],
                calculate_distances=calculate_distances,
                calculate_per_parent_means=calculate_per_parent_means,
                save_children_with_parents=save_children_with_parents,
                relationship_backend_provider=relationship_backend_provider,
                slice_index=index,
            )
            for index in range(parent_array.shape[0])
        )
        return _aggregate_relate_objects_slice_results(
            slice_results,
            memory_type=MemoryType.NUMPY.value,
        )

    slice_index = 0 if slice_index is None else int(slice_index)
    raw_parent_labels = parent_labels
    raw_child_labels = child_labels
    calculate_distances = coerce_cellprofiler_enum(
        RelateObjectsDistanceMethod,
        calculate_distances,
    )
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
        backend_provider=relationship_backend_provider,
    )
    parent_child_relationship = relationship_backend.parent_child_payload_from_labels(
        raw_parent_labels,
        raw_child_labels,
    )

    parent_labels = object_label_dense_array(raw_parent_labels, dtype=np.int32)
    child_labels = object_label_dense_array(raw_child_labels, dtype=np.int32)
    parent_labels, child_labels = DenseObjectLabelPairAligner(
        parent_labels,
        child_labels,
    ).aligned()

    parent_count = int(parent_labels.max()) if parent_labels.max() > 0 else 0
    child_count = int(child_labels.max()) if child_labels.max() > 0 else 0

    parents_of = relationship_backend.parents_of_from_payload(
        parent_child_relationship,
        child_count,
    )

    child_counts_per_parent = np.zeros(parent_count, dtype=np.int32)
    for parent_idx in parents_of:
        if parent_idx > 0 and parent_idx <= parent_count:
            child_counts_per_parent[parent_idx - 1] += 1

    children_with_parents = np.sum(parents_of > 0)
    mean_children = np.mean(child_counts_per_parent) if parent_count > 0 else 0.0

    mean_centroid_dist = np.nan
    mean_minimum_dist = np.nan

    if calculate_distances.calculates_centroid_distance:
        centroid_distances = relationship_backend.centroid_distances(
            parent_labels,
            child_labels,
            parents_of,
        )
        valid_dists = centroid_distances[~np.isnan(centroid_distances)]
        mean_centroid_dist = (
            float(np.mean(valid_dists)) if len(valid_dists) > 0 else np.nan
        )

    if calculate_distances.calculates_minimum_distance:
        minimum_distances = relationship_backend.minimum_distances(
            parent_labels,
            child_labels,
            parents_of,
        )
        valid_dists = minimum_distances[~np.isnan(minimum_distances)]
        mean_minimum_dist = (
            float(np.mean(valid_dists)) if len(valid_dists) > 0 else np.nan
        )

    saved_child_relationship: ParentChildRelationshipPayload | None = None
    if save_children_with_parents:
        retained_child_ids = np.flatnonzero(
            np.concatenate((np.zeros(1, dtype=bool), parents_of > 0))
        ).astype(np.int32, copy=False)
        label_indexes = np.zeros(child_count + 1, dtype=np.int32)
        label_indexes[retained_child_ids] = np.arange(
            1,
            len(retained_child_ids) + 1,
            dtype=np.int32,
        )
        child_index = np.asarray(child_labels, dtype=np.intp)
        output_labels = label_indexes[child_index]
        saved_child_relationship = relationship_backend.parent_child_payload_from_labels(
            child_labels,
            output_labels,
        )
    else:
        output_labels = child_labels.copy()

    measurements = RelationshipMeasurements(
        slice_index=slice_index,
        parent_object_count=parent_count,
        child_object_count=child_count,
        children_with_parents_count=int(children_with_parents),
        mean_children_per_parent=float(mean_children),
        mean_centroid_distance=mean_centroid_dist,
        mean_minimum_distance=mean_minimum_dist,
    )

    related_child_ids = tuple(
        child_idx
        for child_idx, parent_idx in enumerate(parents_of, start=1)
        if parent_idx > 0
    )
    related_parent_ids = tuple(
        int(parent_idx)
        for parent_idx in parents_of
        if parent_idx > 0
    )

    if (
        parent_child_relationship.slice_indices
        or parent_child_relationship.slice_count is not None
    ):
        related_relationship = parent_child_relationship
    else:
        related_relationship = ParentChildRelationshipPayload(
            parent_ids=related_parent_ids,
            child_ids=related_child_ids,
            slice_indices=tuple(slice_index for _child_id in related_child_ids),
            slice_count=slice_index + 1,
        )
    output_labels = object_label_value_with_dense_labels(
        raw_child_labels,
        output_labels.astype(np.float32),
    )
    return RelateObjectsResult(
        output_labels,
        related_relationship,
        measurements,
        saved_child_relationship=saved_child_relationship,
    )


@njit(cache=True)
def _relate_sparse_ijv_children_to_parents_numba(
    parent_ijv: np.ndarray,
    child_ijv: np.ndarray,
    child_count: int,
    parent_count: int,
) -> np.ndarray:
    counts = np.zeros((child_count + 1, parent_count + 1), dtype=np.int32)
    parent_linear = _sparse_ijv_linear_coordinates(parent_ijv, child_ijv)
    child_linear = _sparse_ijv_linear_coordinates(child_ijv, parent_ijv)
    parent_order = np.argsort(parent_linear)
    child_order = np.argsort(child_linear)
    parent_position = 0
    child_position = 0
    while parent_position < parent_order.size and child_position < child_order.size:
        parent_index = parent_order[parent_position]
        child_index = child_order[child_position]
        parent_coordinate = parent_linear[parent_index]
        child_coordinate = child_linear[child_index]
        if parent_coordinate < child_coordinate:
            parent_position += 1
            continue
        if child_coordinate < parent_coordinate:
            child_position += 1
            continue

        parent_end = parent_position + 1
        while (
            parent_end < parent_order.size
            and parent_linear[parent_order[parent_end]] == parent_coordinate
        ):
            parent_end += 1
        child_end = child_position + 1
        while (
            child_end < child_order.size
            and child_linear[child_order[child_end]] == child_coordinate
        ):
            child_end += 1
        for grouped_parent_position in range(parent_position, parent_end):
            grouped_parent_index = parent_order[grouped_parent_position]
            parent_id = int(parent_ijv[grouped_parent_index, 2])
            if parent_id <= 0 or parent_id > parent_count:
                continue
            for grouped_child_position in range(child_position, child_end):
                grouped_child_index = child_order[grouped_child_position]
                child_id = int(child_ijv[grouped_child_index, 2])
                if child_id > 0 and child_id <= child_count:
                    counts[child_id, parent_id] += 1
        parent_position = parent_end
        child_position = child_end

    return _parents_of_from_overlap_counts_numba(counts, child_count, parent_count)


@njit(cache=True)
def _sparse_ijv_linear_coordinates(
    rows: np.ndarray,
    peer_rows: np.ndarray,
) -> np.ndarray:
    max_y = 0
    for index in range(rows.shape[0]):
        y = int(rows[index, 0])
        if y > max_y:
            max_y = y
    for index in range(peer_rows.shape[0]):
        y = int(peer_rows[index, 0])
        if y > max_y:
            max_y = y
    dim_y = max_y + 1
    linear = np.empty(rows.shape[0], dtype=np.int64)
    for index in range(rows.shape[0]):
        linear[index] = int(rows[index, 0]) + dim_y * int(rows[index, 1])
    return linear


@njit(cache=True)
def _relate_children_to_parents_numba(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    child_count: int,
    parent_count: int,
) -> np.ndarray:
    counts = np.zeros((child_count + 1, parent_count + 1), dtype=np.int32)
    height, width = child_labels.shape
    for row in range(height):
        for col in range(width):
            child_id = int(child_labels[row, col])
            parent_id = int(parent_labels[row, col])
            if (
                child_id > 0
                and child_id <= child_count
                and parent_id > 0
                and parent_id <= parent_count
            ):
                counts[child_id, parent_id] += 1

    return _parents_of_from_overlap_counts_numba(counts, child_count, parent_count)


@njit(cache=True)
def _parents_of_from_overlap_counts_numba(
    counts: np.ndarray,
    child_count: int,
    parent_count: int,
) -> np.ndarray:
    parents_of = np.zeros(child_count, dtype=np.int32)
    for child_id in range(1, child_count + 1):
        best_parent = 0
        best_count = 0
        for parent_id in range(1, parent_count + 1):
            overlap = counts[child_id, parent_id]
            if overlap > best_count:
                best_count = overlap
                best_parent = parent_id
        parents_of[child_id - 1] = best_parent
    return parents_of


@njit(cache=True)
def _label_centroids_numba(
    labels: np.ndarray,
    label_count: int,
) -> np.ndarray:
    sums = np.zeros((label_count + 1, 2), dtype=np.float64)
    counts = np.zeros(label_count + 1, dtype=np.int64)
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id > 0 and label_id <= label_count:
                sums[label_id, 0] += row
                sums[label_id, 1] += col
                counts[label_id] += 1

    centroids = np.empty((label_count + 1, 2), dtype=np.float64)
    for label_id in range(label_count + 1):
        if counts[label_id] == 0:
            centroids[label_id, 0] = np.nan
            centroids[label_id, 1] = np.nan
        else:
            centroids[label_id, 0] = sums[label_id, 0] / counts[label_id]
            centroids[label_id, 1] = sums[label_id, 1] / counts[label_id]
    return centroids


@njit(cache=True)
def _calculate_centroid_distances_numba(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    parents_of: np.ndarray,
    parent_count: int,
) -> np.ndarray:
    child_count = len(parents_of)
    distances = np.empty(child_count, dtype=np.float64)
    for child_idx in range(child_count):
        distances[child_idx] = np.nan

    if child_count == 0 or parent_count == 0:
        return distances

    parent_centroids = _label_centroids_numba(parent_labels, parent_count)
    child_centroids = _label_centroids_numba(child_labels, child_count)
    for child_idx in range(child_count):
        parent_id = int(parents_of[child_idx])
        child_id = child_idx + 1
        if parent_id > 0 and parent_id <= parent_count:
            child_row = child_centroids[child_id, 0]
            child_col = child_centroids[child_id, 1]
            parent_row = parent_centroids[parent_id, 0]
            parent_col = parent_centroids[parent_id, 1]
            if not (
                np.isnan(child_row)
                or np.isnan(child_col)
                or np.isnan(parent_row)
                or np.isnan(parent_col)
            ):
                row_delta = child_row - parent_row
                col_delta = child_col - parent_col
                distances[child_idx] = np.sqrt(
                    row_delta * row_delta + col_delta * col_delta
                )
    return distances


@njit(cache=True)
def _is_inner_boundary_pixel(
    labels: np.ndarray,
    row: int,
    col: int,
    label_id: int,
) -> bool:
    height, width = labels.shape
    if row > 0 and int(labels[row - 1, col]) != label_id:
        return True
    if row + 1 < height and int(labels[row + 1, col]) != label_id:
        return True
    if col > 0 and int(labels[row, col - 1]) != label_id:
        return True
    if col + 1 < width and int(labels[row, col + 1]) != label_id:
        return True
    return False


@njit(cache=True)
def _calculate_minimum_distances_numba(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    parents_of: np.ndarray,
    parent_count: int,
) -> np.ndarray:
    child_count = len(parents_of)
    distances = np.empty(child_count, dtype=np.float64)
    for child_idx in range(child_count):
        distances[child_idx] = np.nan

    if child_count == 0 or parent_count == 0:
        return distances

    child_centroids = _label_centroids_numba(child_labels, child_count)
    height, width = parent_labels.shape
    counts = np.zeros(parent_count + 1, dtype=np.int64)

    for row in range(height):
        for col in range(width):
            parent_id = int(parent_labels[row, col])
            if (
                parent_id > 0
                and parent_id <= parent_count
                and _is_inner_boundary_pixel(parent_labels, row, col, parent_id)
            ):
                counts[parent_id] += 1

    offsets = np.zeros(parent_count + 2, dtype=np.int64)
    for parent_id in range(1, parent_count + 1):
        offsets[parent_id + 1] = offsets[parent_id] + counts[parent_id]

    total = offsets[parent_count + 1]
    rows = np.empty(total, dtype=np.float64)
    cols = np.empty(total, dtype=np.float64)
    write_offsets = offsets.copy()
    for row in range(height):
        for col in range(width):
            parent_id = int(parent_labels[row, col])
            if (
                parent_id > 0
                and parent_id <= parent_count
                and _is_inner_boundary_pixel(parent_labels, row, col, parent_id)
            ):
                offset = write_offsets[parent_id]
                rows[offset] = row
                cols[offset] = col
                write_offsets[parent_id] += 1

    for child_idx in range(child_count):
        parent_id = int(parents_of[child_idx])
        child_id = child_idx + 1
        if parent_id <= 0 or parent_id > parent_count:
            continue
        child_row = child_centroids[child_id, 0]
        child_col = child_centroids[child_id, 1]
        if np.isnan(child_row) or np.isnan(child_col):
            continue
        start = offsets[parent_id]
        end = offsets[parent_id + 1]
        if start == end:
            continue
        min_distance_sq = np.inf
        for offset in range(start, end):
            row_delta = rows[offset] - child_row
            col_delta = cols[offset] - child_col
            distance_sq = row_delta * row_delta + col_delta * col_delta
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
        distances[child_idx] = np.sqrt(min_distance_sq)

    return distances


__all__ = public_names_from_objects(
    DistanceMethod,
    NumbaNumpyObjectRelationshipBackendStrategy,
    ObjectRelationshipBackendStrategy,
    RelationshipMeasurements,
    RelateObjectsResult,
    relate_objects,
)
