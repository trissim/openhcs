"""Thin CellProfiler-style view over OpenHCS runtime artifacts."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactType,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.runtime_adapters import RuntimeAdapterRequest
from openhcs.core.runtime_artifact_queries import (
    MeasurementTableUnion,
)
from openhcs.core.runtime_artifact_values import (
    RuntimeValue,
)
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_metadata,
    preserved_image_plane_projection,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelSet,
    ObjectLabelValue,
)
from openhcs.core.runtime_relationships import (
    ObjectRelationship,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
)
from openhcs.core.runtime_spatial_grid import (
    SpatialGrid,
)
from openhcs.core.runtime_stores import (
    RuntimeArtifactInput,
    RuntimeArtifactLocation,
    RuntimeArtifactQuery,
    StoredRuntimeValue,
    replace_runtime_artifact_payload,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)

RelationshipIdVector = np.ndarray | Sequence[int]


@dataclass(slots=True)
class CellProfilerRuntimeAdapter(RuntimePlaneAxisProjector):
    """CellProfiler-like API backed by typed OpenHCS runtime state.

    The adapter deliberately has no object/image/measurement dictionaries of its
    own. Writes require compiled output plans and a filemanager so the
    RuntimeValueStore record and VFS payload stay aligned with the normal
    FunctionStep runtime boundary.
    """

    @classmethod
    def require_parameter_name(cls) -> str:
        """Return the callable ABI name for CellProfiler adapter injection."""
        return "cellprofiler_runtime"

    @classmethod
    def runtime_adapter_spec(cls):
        """Return the sole compiled CellProfiler runtime-adapter declaration."""

        from openhcs.core.runtime_adapters import RuntimeAdapterSpec
        from openhcs.interop.cellprofiler.runtime.module_execution import (
            cellprofiler_runtime_adapter_factory,
            cellprofiler_runtime_callable_factory,
        )

        return RuntimeAdapterSpec(
            parameter_name=cls.require_parameter_name(),
            factory=cellprofiler_runtime_adapter_factory,
            manages_artifact_inputs=True,
            manages_artifact_outputs=True,
            runtime_callable_factory=cellprofiler_runtime_callable_factory,
        )

    request: RuntimeAdapterRequest
    backend: str = Backend.MEMORY.value

    def __post_init__(self) -> None:
        if not isinstance(self.request, RuntimeAdapterRequest):
            raise TypeError(
                "CellProfilerRuntimeAdapter.request must be RuntimeAdapterRequest, got "
                f"{type(self.request).__name__}."
            )
        if not self.backend:
            raise ValueError("CellProfilerRuntimeAdapter.backend cannot be empty.")

    def _artifact_input(
        self,
        name: str,
        artifact_type: type[ArtifactType],
    ) -> RuntimeArtifactInput:
        """Bind one exact compiled artifact input to this runtime axis."""
        spec = self.request.selected_artifact_input_specs().require_by_name_and_artifact_type(
            name,
            artifact_type,
        )
        edge_plan = self.request.require_artifact_input_edge(
            spec.ref()
        )
        storage_plan = edge_plan.storage_plan
        if storage_plan is None:
            raise RuntimeError("Compiled artifact input occurrence lost its storage plan.")
        if storage_plan.artifact_type is not artifact_type:
            raise TypeError(
                f"Compiled artifact input {name!r} has type "
                f"{storage_plan.artifact_type.value}, not {artifact_type.value}."
            )
        return RuntimeArtifactInput(
            edge_plan=edge_plan,
            axis_scope=self.request.axis_scope,
            backend=self.backend,
        )

    def artifact_input_records(
        self,
        name: str,
        artifact_type: type[ArtifactType],
    ) -> tuple[StoredRuntimeValue, ...]:
        """Project one compiled input through native component semantics."""
        return self._artifact_input(name, artifact_type).records(
            self.request.context.runtime_value_store
        )

    def scoped_artifact_input_records(
        self,
        name: str,
        artifact_type: type[ArtifactType],
        *,
        group_key: str | None = None,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Resolve compiled inputs for one explicit invocation group."""
        records = self.artifact_input_records(name, artifact_type)
        if group_key is not None:
            records = tuple(
                record for record in records if record.key.scope.value_text == group_key
            )
        return records

    def artifact_output_records(
        self,
        output_plan: ArtifactOutputPlan,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Return records written through one exact compiled output plan."""
        selected_plan = self.request.require_artifact_output_plan(output_plan.ref())
        if selected_plan != output_plan:
            raise RuntimeError(
                "CellProfiler output retrieval requires a plan selected by the "
                f"current compiled invocation, got {output_plan.ref()!r}."
            )
        query = RuntimeArtifactQuery.from_output_plan(
            output_plan,
            axis_id=self.request.axis_scope.axis_id,
            backend=self.backend,
            group_key=self.request.group_key,
        )
        records = self.request.context.runtime_value_store.find_matching(query)
        if not records:
            raise RuntimeError(
                "Missing CellProfiler invocation output "
                f"{output_plan.artifact_type.value}:{output_plan.name} on axis "
                f"{self.request.axis_scope.axis_id!r}."
            )
        return records

    def artifact_output_value(self, output_plan: ArtifactOutputPlan) -> object:
        """Compose one exact recorded output through its artifact-type owner."""

        records = self.artifact_output_records(output_plan)
        if len(records) != 1:
            raise RuntimeError(
                f"CellProfiler output {output_plan.ref()!r} requires exactly one "
                f"invocation record, got {len(records)}."
            )
        return RuntimeValue.compose((records[0].value,))

    def require_artifact_available(
        self,
        *,
        name: str,
        kind: type[ArtifactType],
    ) -> None:
        """Fail loudly unless an artifact is declared, bound, or resolvable."""
        declared_output = (
            self.request.require_callable_contract()
            .artifact_outputs.by_name_and_artifact_type(name, kind)
        )
        if (
            declared_output is not None
            and self.request.artifact_output_plan(declared_output.ref()) is not None
        ):
            return
        if self.has_source_binding(name, kind):
            return
        self._artifact_input(name, kind)

    def runtime_slice_plane_index(self) -> int | None:
        """Return the current axis-local runtime-slice plane index."""
        return self.request.plane_projection.runtime_slice_plane_index()

    def runtime_slice_axis_size(self) -> int | None:
        """Return the current runtime-slice axis size when known."""
        return self.request.plane_projection.runtime_slice_axis_size()

    def has_source_binding(
        self,
        alias: str,
        kind: ArtifactType | None = None,
    ) -> bool:
        binding = self.request.source_binding_plan.binding_for_alias(alias)
        return binding is not None and (kind is None or binding.artifact_kind is kind)

    def add_image(
        self,
        name: str,
        data: RuntimeArrayData,
        *,
        materialization_source_metadata: ImagePayloadMetadata | None = None,
    ) -> StoredRuntimeValue:
        return self._record_native_value(
            name,
            ImageArtifactType,
            data,
            materialization_source_metadata=materialization_source_metadata,
        )

    def get_image(
        self,
        name: str,
    ) -> RuntimeArrayData:
        records = self.scoped_artifact_input_records(
            name,
            ImageArtifactType,
        )
        edge = self._artifact_input(name, ImageArtifactType)
        data = edge.composed_value(records)
        projected: RuntimeArrayData = data
        source_metadata = image_payload_metadata(projected)
        if source_metadata.plane_axis is RuntimePlaneAxis.SOURCE_BINDING:
            source_projection = preserved_image_plane_projection(
                projected,
                self,
                source_metadata.source_image_names,
            )
            if source_projection.plane_index is not None:
                projected = cast(
                    RuntimeArrayData,
                    RuntimeSliceProjection.value_for_slice(
                        projected,
                        replace(
                            source_projection,
                            source_aliases=source_metadata.source_image_names,
                        ),
                    ),
                )
        runtime_projection = RuntimePlaneAxisValueProjection.from_projector(
            self,
            RuntimePlaneAxis.RUNTIME_SLICE,
            (),
        )
        if runtime_projection is None or runtime_projection.plane_index is None:
            return projected
        return cast(
            RuntimeArrayData,
            RuntimeSliceProjection.value_for_slice(projected, runtime_projection),
        )

    def add_objects(
        self,
        name: str,
        labels: ObjectLabelValue,
        *,
        source_image_name: str | None = None,
        source_image_names: tuple[str, ...] = (),
        source_image_payload: RuntimeArrayData | None = None,
        dimensions: tuple[str, ...] = (),
    ) -> StoredRuntimeValue:
        construct_started_at = time.perf_counter()
        if not isinstance(labels, ObjectLabelValue):
            raise TypeError(
                "CellProfiler object-label recording requires an ObjectLabelValue, "
                f"got {type(labels).__name__}."
            )
        provenance_source_names = labels.source_image_names or source_image_names
        payload = labels.with_variants(
            labels.variant_data,
            source_provenance=labels.source_provenance.with_source_image_names(
                provenance_source_names
            ),
        )
        object_labels = ObjectLabelSet.from_payload(
            name,
            payload,
            source_image_name=source_image_name or labels.source_image_name,
            dimensions=dimensions or labels.dimensions,
        )
        if source_image_payload is not None:
            object_labels = object_labels.with_source_image_context(
                source_image_payload
            )
            object_labels.validate_source_alignment(name)
        CellProfilerRuntimeProfileLogger.object_label_artifact(
            "adapter_construct_object_labels",
            time.perf_counter() - construct_started_at,
            artifact_name=name,
            payload_type=type(labels).__name__,
            labels=object_labels,
        )
        return self._record_native_value(
            name,
            ObjectLabelsArtifactType,
            object_labels,
        )

    def get_objects(
        self,
        name: str,
    ) -> ObjectLabelSet:
        records = self.scoped_artifact_input_records(
            name,
            ObjectLabelsArtifactType,
        )
        objects = cast(
            ObjectLabelSet,
            self._artifact_input(
                name,
                ObjectLabelsArtifactType,
            ).composed_value(records),
        )
        if objects.plane_axis is RuntimePlaneAxis.SOURCE_BINDING:
            declared_projection = objects.declared_plane_projection()
            if declared_projection is None:
                raise ValueError(
                    "Current-source object-label selection requires a declared "
                    "RuntimePlaneAxis.SOURCE_BINDING payload."
                )
            source_plane_index = objects.source_alias_plane_index(
                objects.source_image_names,
                declared_projection.axis_size,
            )
            if source_plane_index is not None:
                objects = cast(
                    ObjectLabelSet,
                    RuntimeSliceProjection.value_for_slice(
                        objects,
                        declared_projection.selected_plane(source_plane_index),
                    ),
                )
        return objects

    def add_measurements(
        self,
        table: MeasurementTable,
    ) -> StoredRuntimeValue:
        name = table.name
        validation_started_at = time.perf_counter()
        object_name = table.subject.object_name
        if object_name is not None:
            self.require_artifact_available(
                name=object_name,
                kind=ObjectLabelsArtifactType,
            )
        CellProfilerRuntimeProfileLogger.measurement_artifact(
            "adapter_measurement_subject_validation",
            time.perf_counter() - validation_started_at,
            artifact_name=name,
            object_name=object_name,
        )
        record_started_at = time.perf_counter()
        stored_value = self._record_native_value(
            name,
            MeasurementsArtifactType,
            table,
        )
        CellProfilerRuntimeProfileLogger.measurement_artifact(
            "adapter_measurement_record_native",
            time.perf_counter() - record_started_at,
            artifact_name=name,
            object_name=object_name,
        )
        return stored_value

    def get_measurements(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> MeasurementTable:
        records = self.scoped_artifact_input_records(
            name,
            MeasurementsArtifactType,
            group_key=group_key,
        )
        return MeasurementTableUnion(
            name,
            tuple(cast(MeasurementTable, record.value.data) for record in records),
        ).as_table()

    def measurement_tables(
        self,
        *,
        group_key: str | None = None,
        match_group: bool = True,
    ) -> tuple[MeasurementTable, ...]:
        """Return measurement tables selected by compiled artifact inputs."""
        records = self.declared_measurement_input_records(
            group_key=group_key,
            match_group=match_group,
        )
        return tuple(cast(MeasurementTable, record.value.data) for record in records)

    def declared_measurement_input_records(
        self,
        *,
        group_key: str | None = None,
        match_group: bool = True,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Return records for measurement artifacts declared by the runtime step."""
        records: list[StoredRuntimeValue] = []
        for edge_plan in self.request.artifact_inputs.values():
            storage_plan = edge_plan.storage_plan
            if storage_plan is None:
                continue
            if storage_plan.artifact_type is not MeasurementsArtifactType:
                continue
            records.extend(
                (
                    self.artifact_input_records(
                        storage_plan.name, MeasurementsArtifactType
                    )
                    if match_group
                    else self._artifact_input(
                        storage_plan.name,
                        MeasurementsArtifactType,
                    ).all_records(self.request.context.runtime_value_store)
                )
            )
        return tuple({id(record): record for record in records}.values())

    def add_relationship(
        self,
        relationship: ObjectRelationship,
        *,
        artifact_type: type[ObjectLineageArtifactType] = RelationshipsArtifactType,
    ) -> StoredRuntimeValue:
        """Record one canonical directed object relationship."""

        if not isinstance(relationship, ObjectRelationship):
            raise TypeError(
                "CellProfilerRuntimeAdapter.add_relationship requires "
                f"ObjectRelationship, got {type(relationship).__name__}."
            )
        if not issubclass(artifact_type, ObjectLineageArtifactType):
            raise TypeError(
                "CellProfilerRuntimeAdapter.add_relationship artifact_type must "
                f"belong to ObjectLineageArtifactType, got {artifact_type!r}."
            )
        return self._record_native_value(
            relationship.name,
            artifact_type,
            relationship,
        )

    def get_relationship(
        self,
        name: str,
        *,
        artifact_type: type[ObjectLineageArtifactType] = RelationshipsArtifactType,
        group_key: str | None = None,
    ) -> ObjectRelationship | RuntimeSliceAlignedValues[ObjectRelationship]:
        records = self.scoped_artifact_input_records(
            name,
            artifact_type,
            group_key=group_key,
        )
        return cast(
            ObjectRelationship | RuntimeSliceAlignedValues[ObjectRelationship],
            self._artifact_input(
                name,
                artifact_type,
            ).composed_value(records),
        )

    def add_spatial_grid(
        self,
        name: str,
        grid: object,
    ) -> StoredRuntimeValue:
        return self._record_native_value(
            name,
            SpatialGridArtifactType,
            grid,
        )

    def get_spatial_grid(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid]:
        records = self.artifact_input_records(name, SpatialGridArtifactType)
        return cast(
            SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid],
            self._artifact_input(
                name,
                SpatialGridArtifactType,
            ).composed_value(records),
        )

    def _record_native_value(
        self,
        name: str,
        expected_kind: ArtifactType,
        native_value: object,
        *,
        materialization_source_metadata: ImagePayloadMetadata | None = None,
    ) -> StoredRuntimeValue:
        total_started_at = time.perf_counter()
        plan_started_at = time.perf_counter()
        output_plan = self._require_output_plan(
            name,
            expected_kind,
        ).for_invocation_group(self.request.group_key)
        CellProfilerRuntimeProfileLogger.artifact(
            "adapter_require_output_plan",
            time.perf_counter() - plan_started_at,
            artifact_name=name,
            kind=expected_kind,
        )
        store_started_at = time.perf_counter()
        normalize_started_at = time.perf_counter()
        runtime_value = RuntimeValue.normalize_for_execution_scope(
            output_plan,
            native_value,
            execution_scope=self.request.axis_scope,
            materialization_source_metadata=materialization_source_metadata,
        )
        CellProfilerRuntimeProfileLogger.artifact(
            "adapter_normalize_artifact_value",
            time.perf_counter() - normalize_started_at,
            artifact_name=name,
            kind=expected_kind,
            payload_type=type(runtime_value.data).__name__,
            group_key=self.request.group_key,
        )
        save_started_at = time.perf_counter()
        runtime_path = output_plan.path
        self._save_payload(runtime_value, runtime_path)
        CellProfilerRuntimeProfileLogger.artifact(
            "adapter_save_payload",
            time.perf_counter() - save_started_at,
            artifact_name=name,
            kind=expected_kind,
            group_key=self.request.group_key,
        )
        replace_started_at = time.perf_counter()
        stored_value = self.request.context.runtime_value_store.replace(
            runtime_value,
            path=runtime_path,
            backend=self.backend,
        )
        CellProfilerRuntimeProfileLogger.artifact(
            "adapter_runtime_store_replace_only",
            time.perf_counter() - replace_started_at,
            artifact_name=name,
            kind=expected_kind,
            group_key=self.request.group_key,
        )
        CellProfilerRuntimeProfileLogger.artifact(
            "adapter_runtime_store_replace",
            time.perf_counter() - store_started_at,
            artifact_name=name,
            kind=expected_kind,
        )
        CellProfilerRuntimeProfileLogger.artifact(
            "adapter_record_native_value",
            time.perf_counter() - total_started_at,
            artifact_name=name,
            kind=expected_kind,
        )
        return stored_value

    def _require_output_plan(
        self,
        name: str,
        expected_kind: ArtifactType,
    ) -> ArtifactOutputPlan:
        declared_output = (
            self.request.require_callable_contract()
            .artifact_outputs.require_by_name_and_artifact_type(name, expected_kind)
        )
        return self.request.require_artifact_output_plan(declared_output.ref())

    def _save_payload(self, value: RuntimeValue, path: str) -> None:
        if self.request.context.filemanager is None:
            raise RuntimeError(
                "CellProfilerRuntimeAdapter.filemanager is required for writes; "
                "adapter writes must persist through the OpenHCS VFS boundary."
            )
        replace_runtime_artifact_payload(
            self.request.context.filemanager,
            value.data,
            RuntimeArtifactLocation(path=path, backend=self.backend),
        )
