"""Runtime stores for typed artifact values."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field, replace
import inspect
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactPlan,
    ArtifactSpec,
    ArtifactSpecRef,
    ArtifactType,
)
from openhcs.core.component_group_scope import (
    RuntimeExecutionAxisScope,
    ComponentGroupScope,
)
from openhcs.core.component_set import ComponentSet
from openhcs.core.function_patterns import InvocationArtifactInputEdgePlan
from openhcs.core.runtime_artifact_values import (
    ArtifactKey,
    RuntimeValue,
)
from openhcs.core.runtime_image_values import (
    image_payload_metadata,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis, RuntimePlaneAxisValueProjection
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.source_matching import (
    SourceAxisMetadataScope,
    SourceImageSetIdentityPolicy,
)
from openhcs.core.source_bindings import CompiledSourceBindingPlan
from openhcs.serialization.json import to_jsonable


@dataclass(frozen=True, slots=True)
class RuntimeArtifactLocation:
    """VFS location for one persisted runtime artifact payload."""

    path: str
    backend: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("RuntimeArtifactLocation.path cannot be empty.")
        if not self.backend:
            raise ValueError("RuntimeArtifactLocation.backend cannot be empty.")


@dataclass(frozen=True, slots=True)
class RuntimeStoreObservationCursor:
    """Cursor into the append-only runtime artifact observation stream."""

    index: int
    revision: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("RuntimeStoreObservationCursor.index cannot be negative.")
        if self.revision < 0:
            raise ValueError(
                "RuntimeStoreObservationCursor.revision cannot be negative."
            )


class RuntimeArtifactQueryTarget:
    """Nominal runtime-artifact address matched after semantic key fields."""

    def matches(self, record: "StoredRuntimeValue") -> bool:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RuntimeArtifactLocationTarget(RuntimeArtifactQueryTarget):
    """Runtime-artifact query target for one persisted VFS location."""

    location: RuntimeArtifactLocation

    def matches(self, record: "StoredRuntimeValue") -> bool:
        return record.location == self.location


@dataclass(frozen=True, slots=True)
class RuntimeArtifactDynamicComponentTarget(RuntimeArtifactQueryTarget):
    """Runtime-artifact query target for all discovered keys of one component."""

    component: AllComponents

    def __post_init__(self) -> None:
        if not isinstance(self.component, AllComponents):
            raise TypeError(
                "RuntimeArtifactDynamicComponentTarget.component must be an "
                "AllComponents value."
            )

    def matches(self, record: "StoredRuntimeValue") -> bool:
        scope = record.key.scope
        return scope.component is self.component and scope.value_text is not None


def replace_runtime_artifact_payload(
    filemanager: Any,
    data: Any,
    location: RuntimeArtifactLocation,
) -> None:
    """Persist the current payload for a latest-binding runtime artifact."""
    filemanager.ensure_directory(str(Path(location.path).parent), location.backend)
    if filemanager.exists(location.path, location.backend):
        filemanager.delete(location.path, location.backend)
    filemanager.save(data, location.path, location.backend)


@dataclass(frozen=True, slots=True)
class RuntimeArtifactQuery:
    """Typed lookup for one planned runtime artifact record."""

    name: str
    artifact_type: ArtifactType
    axis_id: str
    target: RuntimeArtifactQueryTarget

    @classmethod
    def from_input_plan(
        cls,
        input_plan: ArtifactInputPlan,
        *,
        axis_id: str,
        backend: str,
        group_key: str | None = None,
    ) -> "RuntimeArtifactQuery":
        """Build the runtime-store query for one compiled artifact input."""
        input_scope = ComponentGroupScope.from_raw(
            input_plan.group_keys,
            component=input_plan.group_component,
        )
        if input_scope.is_dynamic and group_key is None:
            return cls(
                name=input_plan.name,
                artifact_type=input_plan.artifact_type,
                axis_id=axis_id,
                target=RuntimeArtifactDynamicComponentTarget(
                    input_scope.component,
                ),
            )
        return cls(
            name=input_plan.name,
            artifact_type=input_plan.artifact_type,
            axis_id=axis_id,
            target=RuntimeArtifactLocationTarget(
                RuntimeArtifactLocation(
                    path=input_plan.path_for_runtime_query(group_key),
                    backend=backend,
                )
            ),
        )

    @classmethod
    def from_output_plan(
        cls,
        output_plan: ArtifactOutputPlan,
        *,
        axis_id: str,
        backend: str,
        group_key: str | None,
    ) -> "RuntimeArtifactQuery":
        """Build the exact runtime-store query for one compiled artifact output."""
        resolved_plan = output_plan.for_invocation_group(group_key)
        return cls(
            name=resolved_plan.name,
            artifact_type=resolved_plan.artifact_type,
            axis_id=axis_id,
            target=RuntimeArtifactLocationTarget(
                RuntimeArtifactLocation(
                    path=resolved_plan.path,
                    backend=backend,
                )
            ),
        )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RuntimeArtifactQuery.name cannot be empty.")
        if not self.axis_id:
            raise ValueError("RuntimeArtifactQuery.axis_id cannot be empty.")
        if not isinstance(self.target, RuntimeArtifactQueryTarget):
            raise TypeError(
                "RuntimeArtifactQuery.target must be RuntimeArtifactQueryTarget, "
                f"got {type(self.target).__name__}."
            )

    def matches(self, record: "StoredRuntimeValue") -> bool:
        key = record.key
        if key.name != self.name:
            return False
        if key.artifact_type is not self.artifact_type:
            return False
        if key.scope.axis_id != self.axis_id:
            return False
        if not self.target.matches(record):
            return False
        return True


@dataclass(frozen=True, slots=True)
class StoredRuntimeValue:
    """A validated runtime value with its persistence boundary."""

    value: RuntimeValue
    location: RuntimeArtifactLocation

    @property
    def key(self) -> ArtifactKey:
        return self.value.key

    @property
    def path(self) -> str:
        return self.location.path

    @property
    def backend(self) -> str:
        return self.location.backend


@dataclass(frozen=True, slots=True)
class RuntimeArtifactInput:
    """Bind one compiled artifact input to an exact runtime axis coordinate."""

    edge_plan: InvocationArtifactInputEdgePlan
    axis_scope: RuntimeExecutionAxisScope
    backend: str

    def __post_init__(self) -> None:
        if not isinstance(self.edge_plan, InvocationArtifactInputEdgePlan):
            raise TypeError(
                "RuntimeArtifactInput.edge_plan must be an "
                "InvocationArtifactInputEdgePlan, got "
                f"{type(self.edge_plan).__name__}."
            )
        if not isinstance(self.axis_scope, RuntimeExecutionAxisScope):
            raise TypeError(
                "RuntimeArtifactInput.axis_scope must be a "
                f"RuntimeExecutionAxisScope, got {type(self.axis_scope).__name__}."
            )
        if not self.backend:
            raise ValueError("RuntimeArtifactInput.backend cannot be empty.")
        if self.edge_plan.storage_plan is None or self.edge_plan.projection is None:
            raise ValueError(
                "RuntimeArtifactInput requires a storage-backed compiled input edge."
            )
        self.edge_plan.projection.validate_axis_projection(self.edge_plan.storage_plan)

    def records(
        self,
        store: "RuntimeValueStore",
    ) -> tuple[StoredRuntimeValue, ...]:
        """Resolve records selected by the compiled producer/consumer scopes."""

        storage_plan = self.edge_plan.storage_plan
        projection = self.edge_plan.projection
        producer_scope = storage_plan.producer_group_scope()
        if producer_scope.is_ungrouped:
            return self._records(store, producer_scope, None)
        selection_scope = projection.producer_selection_scope
        if storage_plan.composes_producer_groups(
            ComponentSet.coerce(projection.consumer_variable_components)
        ):
            return self.all_records(store)
        if (
            selection_scope == producer_scope
            and not selection_scope.is_dynamic
            and len(selection_scope.keys) > 1
        ):
            return self.all_records(store)

        runtime_key = self.axis_scope.value_text_for_component(
            selection_scope.component
        )
        selected_key = producer_scope.resolve_runtime_key(
            selection_scope.select_runtime_key(runtime_key)
        )
        return self._records(store, producer_scope, selected_key)

    def all_records(
        self,
        store: "RuntimeValueStore",
    ) -> tuple[StoredRuntimeValue, ...]:
        """Resolve every producer group explicitly declared by this plan."""

        storage_plan = self.edge_plan.storage_plan
        producer_scope = storage_plan.producer_group_scope()
        if not producer_scope.is_dynamic:
            return tuple(
                record
                for group_key in producer_scope.keys
                for record in self._records(store, producer_scope, group_key)
            )
        records = tuple(
            record
            for record in store.find(
                name=storage_plan.name,
                artifact_type=storage_plan.artifact_type,
                axis_id=self.axis_scope.axis_id,
                group_component=producer_scope.component,
                match_component=True,
            )
            if self._matches_execution_scope(record)
        )
        if not records:
            raise RuntimeError(
                f"Missing dynamic grouped artifact input {storage_plan.name!r} "
                f"({storage_plan.artifact_type.value}) on axis "
                f"{self.axis_scope.axis_id!r} for producer component "
                f"{producer_scope.component.value!r}."
            )
        records_by_group: OrderedDict[str, list[StoredRuntimeValue]] = OrderedDict()
        for record in records:
            group_key = record.key.scope.value_text
            if group_key is None:
                raise RuntimeError(
                    f"Dynamic grouped artifact input {storage_plan.name!r} has a "
                    "producer record without a concrete component value."
                )
            records_by_group.setdefault(group_key, []).append(record)
        return tuple(
            record
            for group_key, group_records in records_by_group.items()
            for record in self._validate_selected_records(
                tuple(group_records),
                producer_scope,
                group_key,
            )
        )

    def composed_value(
        self,
        records: tuple[StoredRuntimeValue, ...],
    ) -> Any:
        """Compose exact records after projecting the runtime-axis coordinate."""

        return RuntimeValue.compose(
            tuple(self._axis_value(record.value) for record in records),
            self._producer_group_composition_scope(),
        )

    def projected_values(
        self,
        store: "RuntimeValueStore",
    ) -> tuple[RuntimeValue, ...]:
        """Return exact producer values projected into this consumer scope."""

        return tuple(self._axis_value(record.value) for record in self.records(store))

    def resolve_value(self, store: "RuntimeValueStore") -> Any:
        """Resolve this compiled input to its invocation value."""

        return RuntimeValue.compose(
            self.projected_values(store),
            self._producer_group_composition_scope(),
        )

    def _producer_group_composition_scope(
        self,
    ) -> ComponentGroupScope | None:
        """Return declared group-axis reconstruction semantics for this input."""

        storage_plan = self.edge_plan.storage_plan
        consumer_components = ComponentSet.coerce(
            self.edge_plan.projection.consumer_variable_components
        )
        if not storage_plan.composes_producer_groups(consumer_components):
            return None
        return storage_plan.producer_group_scope()

    def _axis_value(self, value: RuntimeValue) -> RuntimeValue:
        projected_components = ComponentSet.coerce(
            self.edge_plan.projection.projected_variable_components(
                self.edge_plan.storage_plan
            )
        )
        if not projected_components:
            return value

        component_values = tuple(
            (
                component.value,
                self._consumer_component_value(component),
            )
            for component in projected_components
        )
        metadata_scope = SourceAxisMetadataScope.from_component_values(component_values)
        payload = value.data
        source_provenance = image_payload_metadata(payload).source_provenance
        plane_metadata = (
            source_provenance.source_image_provenance_planes.runtime_component_metadata
        )
        if not plane_metadata:
            scalar_metadata = source_provenance.source_component_metadata
            if scalar_metadata is None:
                raise RuntimeError(
                    "Artifact input "
                    f"{self.edge_plan.storage_plan.name!r} declares producer variable "
                    f"components {projected_components.as_tuple()!r} but carries no "
                    "source component metadata."
                )
            if not metadata_scope.matches_metadata(scalar_metadata):
                raise RuntimeError(
                    "Artifact input "
                    f"{self.edge_plan.storage_plan.name!r} scalar source metadata does "
                    f"not match consumer scope {metadata_scope.component_values!r}."
                )
            return value

        matching_indices = metadata_scope.matching_indices(plane_metadata)
        if not matching_indices:
            raise RuntimeError(
                "Artifact input "
                f"{self.edge_plan.storage_plan.name!r} has no producer plane for "
                f"consumer scope {metadata_scope.component_values!r}."
            )
        slice_count = RuntimeSliceProjection.slice_count_from_values((payload,))
        if slice_count is None:
            if len(plane_metadata) == 1 and matching_indices == (0,):
                return value
            raise RuntimeError(
                "Artifact input "
                f"{self.edge_plan.storage_plan.name!r} carries multiple "
                f"{projected_components.as_tuple()!r} source planes without a "
                "declared runtime-slice axis."
            )
        if len(plane_metadata) != slice_count:
            raise RuntimeError(
                "Artifact input "
                f"{self.edge_plan.storage_plan.name!r} source metadata count "
                f"{len(plane_metadata)} does not match its declared runtime-slice "
                f"count {slice_count}."
            )
        if len(matching_indices) != 1:
            raise RuntimeError(
                "Artifact input "
                f"{self.edge_plan.storage_plan.name!r} consumer scope "
                f"{metadata_scope.component_values!r} selects multiple producer "
                f"runtime slices {matching_indices!r}."
            )
        return replace(
            value,
            data=RuntimeSliceProjection.value_for_slice(
                payload,
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE,
                    plane_index=matching_indices[0],
                    axis_size=slice_count,
                ),
            ),
        )

    def _consumer_component_value(self, component: AllComponents) -> str:
        component_scope = self.edge_plan.projection.component_scope(component)
        if component_scope is None:
            raise RuntimeError(
                "Artifact input "
                f"{self.edge_plan.storage_plan.name!r} has no compiled coordinate "
                f"for component {component.value!r}."
            )
        runtime_value = self.axis_scope.value_text_for_component(component)
        return component_scope.select_runtime_key(runtime_value)

    def _records(
        self,
        store: "RuntimeValueStore",
        producer_scope: ComponentGroupScope,
        group_key: str | None,
    ) -> tuple[StoredRuntimeValue, ...]:
        query = RuntimeArtifactQuery.from_input_plan(
            input_plan=self.edge_plan.storage_plan,
            axis_id=self.axis_scope.axis_id,
            backend=self.backend,
            group_key=group_key,
        )
        address_records = tuple(store.find_matching(query))
        records = tuple(
            record
            for record in address_records
            if record.key.scope.component is producer_scope.component
            and record.key.scope.value_text == group_key
            and self._matches_execution_scope(record)
        )
        if not records:
            raise RuntimeError(
                f"Missing RuntimeValueStore record for planned artifact input "
                f"{self.edge_plan.storage_plan.name!r} at producer group "
                f"({producer_scope.component}, {group_key!r}) and execution scope "
                f"{self.axis_scope!r}; address-matched candidate scopes "
                f"{tuple(record.key.scope for record in address_records)!r}."
            )
        return self._validate_selected_records(records, producer_scope, group_key)

    def _validate_selected_records(
        self,
        records: tuple[StoredRuntimeValue, ...],
        producer_scope: ComponentGroupScope,
        group_key: str | None,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Preserve semantic partitions while rejecting unresolved scope ambiguity."""

        if len(records) <= 1:
            return records
        record_scopes = frozenset(record.key.scope for record in records)
        semantic_ids = tuple(record.key.semantic_id for record in records)
        if (
            len(record_scopes) != 1
            or any(semantic_id is None for semantic_id in semantic_ids)
            or len(frozenset(semantic_ids)) != len(semantic_ids)
        ):
            raise RuntimeError(
                f"Ambiguous RuntimeValueStore records for planned artifact input "
                f"{self.edge_plan.storage_plan.name!r} at producer group "
                f"({producer_scope.component}, {group_key!r}) and execution scope "
                f"{self.axis_scope!r}: {records!r}."
            )
        return records

    def _matches_execution_scope(
        self,
        record: StoredRuntimeValue,
    ) -> bool:
        """Match fixed consumer coordinates not carried on producer payload axes."""

        projected_components = ComponentSet.coerce(
            self.edge_plan.projection.projected_variable_components(
                self.edge_plan.storage_plan
            )
        )
        record_scope = record.key.scope
        for component, value in self.axis_scope.fixed_component_values:
            if component in projected_components:
                continue
            producer_value = record_scope.value_text_for_component(component)
            if producer_value is not None and producer_value != value:
                return False
        for component, value in record_scope.fixed_component_values:
            if component in projected_components:
                continue
            consumer_value = self.axis_scope.value_text_for_component(component)
            if consumer_value is not None and consumer_value != value:
                return False
        return True


@dataclass(frozen=True, slots=True)
class RuntimeArtifactBatch:
    """Immutable contract-selected runtime records for one plate invocation."""

    input_specs: tuple[ArtifactSpec, ...]
    records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]]
    source_image_set_identity_policy: SourceImageSetIdentityPolicy
    source_binding_plan: CompiledSourceBindingPlan = dataclass_field(
        default_factory=CompiledSourceBindingPlan.empty
    )

    def __post_init__(self) -> None:
        input_specs = tuple(self.input_specs)
        for spec in input_specs:
            if not isinstance(spec, ArtifactSpec):
                raise TypeError(
                    "RuntimeArtifactBatch.input_specs must contain ArtifactSpec "
                    f"values, got {type(spec).__name__}."
                )

        if not isinstance(
            self.source_image_set_identity_policy,
            SourceImageSetIdentityPolicy,
        ):
            raise TypeError(
                "RuntimeArtifactBatch.source_image_set_identity_policy must be "
                "SourceImageSetIdentityPolicy."
            )

        if not isinstance(self.source_binding_plan, CompiledSourceBindingPlan):
            raise TypeError(
                "RuntimeArtifactBatch.source_binding_plan must be "
                "CompiledSourceBindingPlan."
            )

        records_by_axis: dict[str, tuple[StoredRuntimeValue, ...]] = {}
        for axis_id, records in self.records_by_axis.items():
            if not isinstance(axis_id, str) or not axis_id:
                raise TypeError(
                    "RuntimeArtifactBatch.records_by_axis keys must be "
                    "non-empty strings."
                )
            frozen_records = tuple(records)
            for record in frozen_records:
                if not isinstance(record, StoredRuntimeValue):
                    raise TypeError(
                        "RuntimeArtifactBatch.records_by_axis must contain "
                        f"StoredRuntimeValue values, got {type(record).__name__}."
                    )
                if record.key.scope.axis_id != axis_id:
                    raise ValueError(
                        "RuntimeArtifactBatch record axis does not match its "
                        f"records_by_axis key {axis_id!r}."
                    )
            records_by_axis[axis_id] = frozen_records

        object.__setattr__(self, "input_specs", input_specs)
        object.__setattr__(
            self,
            "records_by_axis",
            MappingProxyType(records_by_axis),
        )

    @classmethod
    def require_parameter_name(cls) -> str:
        """Return the runtime-owned callable parameter name."""
        return "artifact_batch"

    @classmethod
    def parameter(cls) -> inspect.Parameter:
        """Return the required keyword-only runtime parameter declaration."""
        return inspect.Parameter(
            cls.require_parameter_name(),
            inspect.Parameter.KEYWORD_ONLY,
            annotation=cls,
        )

    def specs_of_type(
        self,
        artifact_type: type[ArtifactType],
    ) -> tuple[ArtifactSpec, ...]:
        """Return declared input specs of one exact artifact type."""
        return tuple(
            spec for spec in self.input_specs if spec.artifact_type is artifact_type
        )

    def records(
        self,
        ref: ArtifactSpecRef,
    ) -> Mapping[str, tuple[StoredRuntimeValue, ...]]:
        """Return selected records for one declared input reference."""
        if not isinstance(ref, ArtifactSpecRef):
            raise TypeError("RuntimeArtifactBatch.records requires ArtifactSpecRef.")
        if all(spec.ref() != ref for spec in self.input_specs):
            raise KeyError(
                "Undeclared runtime artifact reference: "
                f"{ref.artifact_type.value}:{ref.name}."
            )
        return MappingProxyType(
            {
                axis_id: tuple(
                    record
                    for record in records
                    if record.key.name == ref.name
                    and record.key.artifact_type is ref.artifact_type
                )
                for axis_id, records in self.records_by_axis.items()
            }
        )

    def records_of_type(
        self,
        artifact_type: type[ArtifactType],
    ) -> Mapping[str, tuple[StoredRuntimeValue, ...]]:
        """Return selected records for all declared inputs of one type."""
        declared_names = frozenset(
            spec.name for spec in self.specs_of_type(artifact_type)
        )
        return MappingProxyType(
            {
                axis_id: tuple(
                    record
                    for record in records
                    if record.key.artifact_type is artifact_type
                    and record.key.name in declared_names
                )
                for axis_id, records in self.records_by_axis.items()
            }
        )


@dataclass(frozen=True, slots=True)
class RuntimeArtifactAddress:
    """Runtime artifact address projected from a stored runtime value."""

    key: ArtifactKey
    location: RuntimeArtifactLocation
    value_type: str | None = None

    @classmethod
    def from_record(cls, record: StoredRuntimeValue) -> "RuntimeArtifactAddress":
        return cls(
            key=record.key,
            location=record.location,
            value_type=type(record.value.data).__qualname__,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeArtifactAddress":
        key = data["key"]
        location = data["location"]
        if not isinstance(key, Mapping):
            raise TypeError("RuntimeArtifactAddress.key must be a mapping.")
        if not isinstance(location, Mapping):
            raise TypeError("RuntimeArtifactAddress.location must be a mapping.")
        scope = key["scope"]
        if not isinstance(scope, Mapping):
            raise TypeError("RuntimeArtifactAddress.key.scope must be a mapping.")
        component = scope.get("component")
        value = scope.get("value")
        fixed_component_values = scope["fixed_component_values"]
        if not isinstance(fixed_component_values, Sequence) or isinstance(
            fixed_component_values,
            (str, bytes),
        ):
            raise TypeError(
                "RuntimeArtifactAddress.key.scope.fixed_component_values must be a "
                "sequence."
            )
        fixed_values: list[tuple[str, str]] = []
        for item in fixed_component_values:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) != 2
            ):
                raise TypeError(
                    "Runtime artifact fixed component coordinates must be "
                    "two-item sequences."
                )
            fixed_values.append((str(item[0]), str(item[1])))
        value_type = data.get("value_type")
        return cls(
            key=ArtifactKey(
                name=str(key["name"]),
                artifact_type=ArtifactType.coerce(str(key["artifact_type"])),
                scope=RuntimeExecutionAxisScope.from_raw(
                    str(scope["axis_id"]),
                    component=component,
                    value=None if value is None else str(value),
                    fixed_component_values=tuple(fixed_values),
                ),
                semantic_id=(
                    None if key.get("semantic_id") is None else str(key["semantic_id"])
                ),
            ),
            location=RuntimeArtifactLocation(
                path=str(location["path"]),
                backend=str(location["backend"]),
            ),
            value_type=None if value_type is None else str(value_type),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        if not isinstance(payload, Mapping):
            raise TypeError("RuntimeArtifactAddress transport must be a mapping.")
        return dict(payload)


class RuntimeValueStore:
    """Source of truth for validated runtime artifact values in one context."""

    def __init__(self) -> None:
        self._records_by_location: OrderedDict[
            tuple[ArtifactKey, RuntimeArtifactLocation],
            StoredRuntimeValue,
        ] = OrderedDict()
        self._observation_records: list[StoredRuntimeValue] = []
        self._current_location_by_key: dict[ArtifactKey, RuntimeArtifactLocation] = {}
        self._revision = 0
        self._find_cache: dict[
            tuple[
                int,
                str | None,
                ArtifactType | None,
                str | None,
                AllComponents | None,
                str | None,
                bool,
                bool,
            ],
            tuple[StoredRuntimeValue, ...],
        ] = {}
        self._find_matching_cache: dict[
            tuple[int, RuntimeArtifactQuery],
            tuple[StoredRuntimeValue, ...],
        ] = {}

    @staticmethod
    def address_matches_plan(
        address: RuntimeArtifactAddress,
        plan: ArtifactPlan,
        *,
        axis_id: str,
    ) -> bool:
        """Match a runtime address to one exact compiled artifact plan."""

        key = address.key
        if key.name != plan.name or key.artifact_type is not plan.artifact_type:
            return False
        if key.scope.axis_id != axis_id:
            return False
        if key.scope.component is not plan.group_component:
            return False
        if plan.group_component is None:
            resolved_plan = plan
        else:
            resolved_plan = plan.for_group(key.scope.value_text)
            if resolved_plan is None:
                return False
        return address.location.path == resolved_plan.path

    @property
    def revision(self) -> int:
        """Return the mutation revision for cache-safe runtime queries."""
        return self._revision

    def record(
        self,
        value: RuntimeValue,
        *,
        path: str,
        backend: str,
    ) -> StoredRuntimeValue:
        """Record a validated value and its persistence location."""
        record = StoredRuntimeValue(
            value=value,
            location=RuntimeArtifactLocation(path=path, backend=backend),
        )
        existing = self._current_record(value.key)
        if existing is not None:
            _validate_overwrite(existing, record)
        self._records_by_location[(value.key, record.location)] = record
        self._observation_records.append(record)
        self._current_location_by_key[value.key] = record.location
        self._mark_mutated()
        return record

    def replace(
        self,
        value: RuntimeValue,
        *,
        path: str,
        backend: str,
    ) -> StoredRuntimeValue:
        """Replace the current binding for a typed artifact key.

        Path planning treats repeated producers for the same artifact name as a
        new workspace binding. This method makes that replacement explicit while
        keeping record() strict for accidental duplicate writes.
        """
        record = StoredRuntimeValue(
            value=value,
            location=RuntimeArtifactLocation(path=path, backend=backend),
        )
        self._records_by_location[(value.key, record.location)] = record
        self._observation_records.append(record)
        self._current_location_by_key[value.key] = record.location
        self._mark_mutated()
        return record

    def resolve(
        self,
        query: RuntimeArtifactQuery,
        *,
        purpose: str,
    ) -> StoredRuntimeValue:
        """Resolve exactly one runtime artifact record for a planned operation."""
        records = self.find_matching(query)
        if not records:
            same_name_records = tuple(
                record
                for record in self._records_by_location.values()
                if record.key.name == query.name
                and record.key.artifact_type is query.artifact_type
                and record.key.scope.axis_id == query.axis_id
            )
            candidate_locations = tuple(
                (
                    record.key.scope.value_text,
                    record.location.backend,
                    record.location.path,
                )
                for record in same_name_records
            )
            raise RuntimeError(
                f"Missing RuntimeValueStore record for {purpose} "
                f"'{query.name}' ({query.artifact_type.value}) on axis "
                f"'{query.axis_id}' target {query.target!r}. "
                f"Candidate same-name records: {candidate_locations!r}."
            )
        if len(records) > 1:
            raise RuntimeError(
                f"Ambiguous RuntimeValueStore records for {purpose} "
                f"'{query.name}' ({query.artifact_type.value}) on axis '{query.axis_id}': "
                f"{records!r}."
            )
        return records[0]

    def find_matching(
        self,
        query: RuntimeArtifactQuery,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Return stored records matched by a typed runtime artifact query."""
        cache_key = (self._revision, query)
        cached = self._find_matching_cache.get(cache_key)
        if cached is not None:
            return cached
        result = tuple(
            record
            for record in self._records_by_location.values()
            if query.matches(record)
        )
        self._find_matching_cache[cache_key] = result
        return result

    def get(self, key: ArtifactKey) -> StoredRuntimeValue:
        """Return one stored value by exact typed artifact key."""
        record = self._current_record(key)
        if record is None:
            raise KeyError(f"Runtime artifact key not found: {key!r}")
        return record

    def find(
        self,
        *,
        name: str | None = None,
        artifact_type: ArtifactType | None = None,
        axis_id: str | None = None,
        group_component: AllComponents | None = None,
        group_key: str | None = None,
        match_component: bool = False,
        match_group: bool = False,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Find stored values by semantic identity fields."""
        cache_key = (
            self._revision,
            name,
            artifact_type,
            axis_id,
            group_component,
            group_key,
            match_component,
            match_group,
        )
        cached = self._find_cache.get(cache_key)
        if cached is not None:
            return cached
        records: list[StoredRuntimeValue] = []
        for record in self._records_by_location.values():
            key = record.key
            if name is not None and key.name != name:
                continue
            if artifact_type is not None and key.artifact_type is not artifact_type:
                continue
            if axis_id is not None and key.scope.axis_id != axis_id:
                continue
            if match_component and key.scope.component is not group_component:
                continue
            if match_group and key.scope.value_text != group_key:
                continue
            records.append(record)
        result = tuple(records)
        self._find_cache[cache_key] = result
        return result

    def find_by_location(
        self,
        *,
        path: str,
        backend: str,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Find stored values persisted at a VFS location."""
        location = RuntimeArtifactLocation(path=path, backend=backend)
        return tuple(
            record
            for record in self._records_by_location.values()
            if record.location == location
        )

    def keys(self) -> tuple[ArtifactKey, ...]:
        """Return stored keys in insertion order."""
        return tuple(record.key for record in self._records_by_location.values())

    def values(self) -> tuple[StoredRuntimeValue, ...]:
        """Return stored records in insertion order."""
        return tuple(self._records_by_location.values())

    @property
    def observed_values(self) -> tuple[StoredRuntimeValue, ...]:
        """Return every runtime artifact write in insertion order."""
        return tuple(self._observation_records)

    def observation_cursor(self) -> RuntimeStoreObservationCursor:
        """Return a cursor for future observation-delta queries."""
        return RuntimeStoreObservationCursor(
            index=len(self._observation_records),
            revision=self._revision,
        )

    def observed_values_after(
        self,
        cursor: RuntimeStoreObservationCursor,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Return runtime artifact writes recorded after ``cursor``."""
        if cursor.index > len(self._observation_records):
            raise ValueError(
                "RuntimeStoreObservationCursor.index is beyond the current "
                f"observation stream length: {cursor.index} > "
                f"{len(self._observation_records)}."
            )
        return tuple(self._observation_records[cursor.index :])

    def clear(self) -> None:
        """Release every runtime artifact record owned by this execution context."""
        self._records_by_location.clear()
        self._observation_records.clear()
        self._current_location_by_key.clear()
        self._mark_mutated()

    def merge_observed_values(
        self,
        records: tuple[StoredRuntimeValue, ...],
    ) -> None:
        """Merge observed records produced across an execution boundary."""
        if records and tuple(self._observation_records) == records:
            return
        for record in records:
            key = (record.key, record.location)
            self._records_by_location[key] = record
            self._current_location_by_key[record.key] = record.location
        if records:
            self._observation_records.extend(records)
            self._mark_mutated()

    def __len__(self) -> int:
        return len(self._records_by_location)

    def _current_record(self, key: ArtifactKey) -> StoredRuntimeValue | None:
        location = self._current_location_by_key.get(key)
        if location is None:
            return None
        return self._records_by_location.get((key, location))

    def _mark_mutated(self) -> None:
        self._revision += 1
        self._find_cache.clear()
        self._find_matching_cache.clear()


def _validate_overwrite(
    existing: StoredRuntimeValue,
    incoming: StoredRuntimeValue,
) -> None:
    if existing.location.backend != incoming.location.backend:
        raise ValueError(
            f"Runtime artifact '{incoming.key.name}' already exists in backend "
            f"'{existing.location.backend}', cannot overwrite from "
            f"'{incoming.location.backend}'."
        )
    if existing.location.path != incoming.location.path:
        raise ValueError(
            f"Runtime artifact '{incoming.key.name}' already exists at "
            f"'{existing.location.path}', cannot overwrite at "
            f"'{incoming.location.path}'."
        )
