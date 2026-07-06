"""Runtime artifact record resolution for the CellProfiler adapter."""

from __future__ import annotations

from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, TypeVar, cast

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSidecarRole,
    ArtifactType,
    ArtifactTypeStrategyMatchMixin,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.interop.cellprofiler.runtime.adapter_scope import (
    RuntimeGroupMatchScope,
)
from openhcs.interop.cellprofiler.runtime.source_identity import (
    CellProfilerCurrentImage,
    RuntimeRecordSourceImageSetSelector,
)

RuntimeArtifactRecordResolution = tuple[StoredRuntimeValue, ...] | None

class RuntimeArtifactSourceScopeStage(str, Enum):
    """Runtime record resolution stage for current-source scoping diagnostics."""

    GROUPED_INPUT = "grouped_input"
    MATCHING_INPUT = "matching_input"
    AXIS_FALLBACK = "axis_fallback"

@dataclass(frozen=True, slots=True)
class RuntimeArtifactSourceScopeStageSpec:
    """Diagnostic policy for one current-source resolution stage."""

    error_subject: str
    include_scope_diagnostics: bool

_SOURCE_SCOPE_STAGE_SPECS = MappingProxyType(
    {
        RuntimeArtifactSourceScopeStage.GROUPED_INPUT: RuntimeArtifactSourceScopeStageSpec(
            error_subject="Grouped runtime artifact input",
            include_scope_diagnostics=False,
        ),
        RuntimeArtifactSourceScopeStage.MATCHING_INPUT: RuntimeArtifactSourceScopeStageSpec(
            error_subject="Runtime artifact input",
            include_scope_diagnostics=False,
        ),
        RuntimeArtifactSourceScopeStage.AXIS_FALLBACK: RuntimeArtifactSourceScopeStageSpec(
            error_subject="Runtime artifact",
            include_scope_diagnostics=True,
        ),
    }
)

@dataclass(frozen=True, slots=True)
class CurrentSourceRuntimeInputGroupResolution:
    """Infer a producer group only when the current source scope is singular."""

    adapter: "CellProfilerRuntimeAdapter"
    current_image: CellProfilerCurrentImage | None
    group_keys: frozenset[str]

    def resolve(self) -> str | None:
        if self.is_multi_source_scope():
            return None
        return self.adapter.runtime_input_group_key_from_current_sources(self.group_keys)

    def is_multi_source_scope(self) -> bool:
        if self.current_image is None:
            return False
        identities = RuntimeRecordSourceImageSetSelector(
            self.adapter,
            self.current_image,
        ).current_source_identities()
        return len(identities) > 1

@dataclass(frozen=True, slots=True)
class RuntimeArtifactCurrentSourceScopeResolution:
    """Resolve candidate records against the current source image scope."""

    request: "RuntimeArtifactRecordResolver"
    records: tuple[StoredRuntimeValue, ...]
    stage_spec: RuntimeArtifactSourceScopeStageSpec

    def resolve(self) -> RuntimeArtifactRecordResolution:
        if self.request.current_image is None:
            return self.records
        selector = RuntimeRecordSourceImageSetSelector(
            self.request.adapter,
            self.request.current_image,
        )
        if not selector.has_current_source_scope():
            return self.records
        if selector.has_template_current_source_scope():
            return self.records
        if not selector.records_have_metadata_source_scope(self.records):
            return self.records
        scoped_records = selector.select(self.records)
        if scoped_records:
            return scoped_records
        message = (
            f"{self.stage_spec.error_subject} {self.request.name!r} "
            f"({self.request.artifact_type.value}) has records, but none match the "
            "current source image scope."
        )
        if self.stage_spec.include_scope_diagnostics:
            message = (
                f"{message} "
                f"current_paths={selector.current_source_paths()!r}; "
                f"current_identities={selector.current_source_identities()!r}; "
                f"record_identities="
                f"{tuple(selector.record_source_identities(record) for record in self.records)!r}."
            )
        raise RuntimeError(message)

RuntimeArtifactTypePolicyT = TypeVar(
    "RuntimeArtifactTypePolicyT",
    bound="RuntimeArtifactTypePolicyMixin",
)

class RuntimeArtifactTypePolicyMixin:
    """Shared ArtifactType registry lookup template for runtime artifact policies."""

    artifact_type: ClassVar[type[ArtifactType] | None] = None
    default_policy_type: ClassVar[type["RuntimeArtifactTypePolicyMixin"] | None] = None

    @classmethod
    def for_artifact_type(
        cls: type[RuntimeArtifactTypePolicyT],
        artifact_type: ArtifactType,
    ) -> RuntimeArtifactTypePolicyT:
        policy = cls.for_context(ArtifactType.coerce(artifact_type), required=False)
        if policy is not None:
            return cast(RuntimeArtifactTypePolicyT, policy)
        if cls.default_policy_type is None:
            raise LookupError(
                f"No runtime artifact policy registered for {artifact_type!r}."
            )
        return cast(type[RuntimeArtifactTypePolicyT], cls.default_policy_type)()

class RuntimeArtifactSourceScopePolicy(
    RuntimeArtifactTypePolicyMixin,
    ArtifactTypeStrategyMatchMixin,
    MostDerivedContextStrategyMixin[type[ArtifactType]],
):
    """Resolve runtime artifact records according to source-scope semantics."""

    def grouped_input_records(
        self,
        request: "RuntimeArtifactRecordResolver",
        records: tuple[StoredRuntimeValue, ...],
    ) -> RuntimeArtifactRecordResolution:
        return records

    def query_miss_records(
        self,
        request: "RuntimeArtifactRecordResolver",
        candidate_records: tuple[StoredRuntimeValue, ...],
    ) -> RuntimeArtifactRecordResolution:
        del request, candidate_records
        return None

    def matching_records(
        self,
        request: "RuntimeArtifactRecordResolver",
        records: tuple[StoredRuntimeValue, ...],
    ) -> RuntimeArtifactRecordResolution:
        return None

    def axis_records_after_query_miss(
        self,
        request: "RuntimeArtifactRecordResolver",
        records: tuple[StoredRuntimeValue, ...],
    ) -> RuntimeArtifactRecordResolution:
        return records

class RuntimeArtifactGlobalScopePolicy(RuntimeArtifactSourceScopePolicy):
    """Keep axis/group records intact for artifact kinds without source scoping."""


class RuntimeArtifactCurrentSourceScopePolicy(
    RuntimeArtifactGlobalScopePolicy,
    ABC,
):
    """Select records matching the current CellProfiler source image when needed."""

    def grouped_input_records(
        self,
        request: "RuntimeArtifactRecordResolver",
        records: tuple[StoredRuntimeValue, ...],
    ) -> RuntimeArtifactRecordResolution:
        return self.records_for_source_scope_stage(
            RuntimeArtifactSourceScopeStage.GROUPED_INPUT,
            request,
            records,
        )

    def query_miss_records(
        self,
        request: "RuntimeArtifactRecordResolver",
        candidate_records: tuple[StoredRuntimeValue, ...],
    ) -> RuntimeArtifactRecordResolution:
        if request.current_image is None:
            return None
        if not candidate_records:
            return None
        selector = RuntimeRecordSourceImageSetSelector(
            request.adapter,
            request.current_image,
        )
        if not selector.has_current_source_scope():
            return None
        scoped_records = selector.select(candidate_records)
        if scoped_records:
            return scoped_records
        return None

    def matching_records(
        self,
        request: "RuntimeArtifactRecordResolver",
        records: tuple[StoredRuntimeValue, ...],
    ) -> RuntimeArtifactRecordResolution:
        if len(records) <= 1:
            return None
        return self.records_for_source_scope_stage(
            RuntimeArtifactSourceScopeStage.MATCHING_INPUT,
            request,
            records,
        )

    def axis_records_after_query_miss(
        self,
        request: "RuntimeArtifactRecordResolver",
        records: tuple[StoredRuntimeValue, ...],
    ) -> RuntimeArtifactRecordResolution:
        return self.records_for_source_scope_stage(
            RuntimeArtifactSourceScopeStage.AXIS_FALLBACK,
            request,
            records,
        )

    @staticmethod
    def records_for_source_scope_stage(
        stage: RuntimeArtifactSourceScopeStage,
        request: "RuntimeArtifactRecordResolver",
        records: tuple[StoredRuntimeValue, ...],
    ) -> RuntimeArtifactRecordResolution:
        return RuntimeArtifactCurrentSourceScopeResolution(
            request=request,
            records=records,
            stage_spec=_SOURCE_SCOPE_STAGE_SPECS[stage],
        ).resolve()


RuntimeArtifactSourceScopePolicy.default_policy_type = RuntimeArtifactGlobalScopePolicy

class ImageRuntimeArtifactSourceScopePolicy(RuntimeArtifactCurrentSourceScopePolicy):
    """Image records are scoped to the current source image when possible."""

    artifact_type = ImageArtifactType

    def axis_records_after_query_miss(
        self,
        request: "RuntimeArtifactRecordResolver",
        records: tuple[StoredRuntimeValue, ...],
    ) -> RuntimeArtifactRecordResolution:
        if request.sidecar_role is ArtifactSidecarRole.CROP_MASK:
            return super().axis_records_after_query_miss(request, records)
        return records

class ObjectLabelRuntimeArtifactSourceScopePolicy(
    RuntimeArtifactCurrentSourceScopePolicy
):
    """Object-label records are scoped to the current source image when possible."""

    artifact_type = ObjectLabelsArtifactType

class MeasurementRuntimeArtifactSourceScopePolicy(
    RuntimeArtifactCurrentSourceScopePolicy
):
    """Measurement records are scoped to the current source image when possible."""

    artifact_type = MeasurementsArtifactType

@dataclass(frozen=True, slots=True)
class RuntimeArtifactRecordResolver:
    """Resolve runtime records for one adapter artifact lookup."""

    adapter: "CellProfilerRuntimeAdapter"
    group_key: str | None
    current_image: CellProfilerCurrentImage | None
    name: str
    artifact_type: type[ArtifactType]
    match_group: bool = True

    def resolve(self) -> tuple[StoredRuntimeValue, ...]:
        input_plan = self.adapter.artifact_inputs.get(self.name)
        resolved_group_key = (
            self.adapter.runtime_input_group_key(
                name=self.name,
                artifact_type=self.artifact_type,
                group_key=self.group_key,
                current_image=self.current_image,
            )
            if self.match_group
            else None
        )
        declared = RuntimeArtifactDeclaredInputResolution(
            request=self,
            input_plan=input_plan,
            resolved_group_key=resolved_group_key,
        ).resolve()
        if declared is not None:
            return declared
        return RuntimeArtifactUndeclaredInputResolution(
            request=self,
            resolved_group_key=resolved_group_key,
        ).resolve()

    @property
    def source_scope_policy(self) -> RuntimeArtifactSourceScopePolicy:
        return RuntimeArtifactSourceScopePolicy.for_artifact_type(
            self.artifact_type
        )

    @property
    def sidecar_role(self) -> ArtifactSidecarRole | None:
        input_plan = self.adapter.artifact_inputs.get(self.name)
        if input_plan is None:
            return None
        return input_plan.sidecar_role

    def validate_input_plan_artifact_type(
        self,
        input_plan: ArtifactInputPlan,
    ) -> None:
        if input_plan.artifact_type is self.artifact_type:
            return
        raise ValueError(
            f"CellProfiler artifact input '{self.name}' expected artifact type "
            f"{self.artifact_type.value}, got compiled artifact type "
            f"{input_plan.artifact_type.value}."
        )

@dataclass(frozen=True, slots=True)
class RuntimeArtifactDeclaredInputResolution:
    """Resolve records through a compiled artifact input plan."""

    request: RuntimeArtifactRecordResolver
    input_plan: ArtifactInputPlan | None
    resolved_group_key: str | None

    def resolve(self) -> RuntimeArtifactRecordResolution:
        input_plan = self.input_plan
        if input_plan is None:
            return None
        self.request.validate_input_plan_artifact_type(input_plan)
        grouped_resolution = self.grouped_resolution(input_plan)
        if grouped_resolution is not None:
            return grouped_resolution
        return self.query_resolution(input_plan)

    def grouped_resolution(
        self,
        input_plan: ArtifactInputPlan,
    ) -> RuntimeArtifactRecordResolution:
        if not _is_global_grouped_input_request(
            input_plan,
            self.resolved_group_key,
        ):
            return None
        group_keys = input_plan.group_keys
        if group_keys is None:
            return None
        records_by_group = tuple(
            self.request.adapter.runtime_value_store.find_matching(
                RuntimeGroupMatchScope(
                    group_key=input_group_key
                ).runtime_scope(self.request.adapter).input_plan_query(
                    input_plan,
                    group_key=input_group_key,
                    backend=self.request.adapter.backend,
                )
            )
            for input_group_key in group_keys
        )
        if all(records_by_group):
            records = RuntimeArtifactRecordDeduplication(
                tuple(record for records in records_by_group for record in records)
            ).unique_by_location()
            return self.request.source_scope_policy.grouped_input_records(
                self.request,
                records,
            )
        if any(records_by_group):
            realized_records = RuntimeArtifactRecordDeduplication(
                tuple(record for records in records_by_group for record in records)
            ).unique_by_location()
            return self.request.source_scope_policy.grouped_input_records(
                self.request,
                realized_records,
            )
        identity_records = self.identity_group_resolution(input_plan)
        if identity_records is not None:
            return identity_records

        first_group_key = group_keys[0]
        self.request.adapter.runtime_value_store.resolve(
            RuntimeGroupMatchScope(
                group_key=first_group_key
            ).runtime_scope(self.request.adapter).input_plan_query(
                input_plan,
                group_key=first_group_key,
                backend=self.request.adapter.backend,
            ),
            purpose="CellProfiler grouped runtime artifact input",
        )
        return None

    def identity_group_resolution(
        self,
        input_plan: ArtifactInputPlan,
    ) -> RuntimeArtifactRecordResolution:
        """Resolve a collapsed identity-scoped producer for grouped consumers."""
        records = self.request.adapter.runtime_value_store.find_matching(
            RuntimeGroupMatchScope(group_key=None)
            .runtime_scope(self.request.adapter)
            .input_plan_query(
                input_plan,
                group_key=None,
                backend=self.request.adapter.backend,
            )
        )
        identity_records = tuple(
            record for record in records if record.key.scope.group_key is None
        )
        if len(identity_records) != 1:
            return None
        return self.request.source_scope_policy.grouped_input_records(
            self.request,
            identity_records,
        )

    def query_resolution(
        self,
        input_plan: ArtifactInputPlan,
    ) -> tuple[StoredRuntimeValue, ...]:
        query = RuntimeGroupMatchScope(
            group_key=self.resolved_group_key
        ).runtime_scope(self.request.adapter).input_plan_query(
            self.group_input_plan(input_plan),
            group_key=self.resolved_group_key,
            backend=self.request.adapter.backend,
        )
        records = self.request.adapter.runtime_value_store.find_matching(query)
        ambiguous_records = self.ambiguous_records(records)
        if ambiguous_records:
            return ambiguous_records
        if not records:
            query_miss_records = self.query_miss_records()
            if query_miss_records is not None:
                return query_miss_records
        scoped_resolution = self.request.source_scope_policy.matching_records(
            self.request,
            records,
        )
        if scoped_resolution is not None:
            return scoped_resolution
        return (
            self.request.adapter.runtime_value_store.resolve(
                query,
                purpose="CellProfiler runtime artifact input",
            ),
        )

    def group_input_plan(
        self,
        input_plan: ArtifactInputPlan,
    ) -> ArtifactInputPlan:
        grouped_plan = input_plan.for_group(self.resolved_group_key)
        if grouped_plan is None:
            return input_plan
        return grouped_plan

    def ambiguous_records(
        self,
        records: tuple[StoredRuntimeValue, ...],
    ) -> tuple[StoredRuntimeValue, ...]:
        if len(records) <= 1:
            return ()
        group_keys = {
            str(record.key.scope.group_key)
            for record in records
            if record.key.scope.group_key is not None
        }
        if len(group_keys) <= 1:
            return ()

        if self.resolved_group_key is not None:
            selected_group = str(self.resolved_group_key)
            if selected_group in group_keys:
                return tuple(
                    record
                    for record in records
                    if str(record.key.scope.group_key) == selected_group
                )

        current_step_group = CurrentSourceRuntimeInputGroupResolution(
            adapter=self.request.adapter,
            group_keys=frozenset(group_keys),
            current_image=self.request.current_image,
        )
        current_step_group_key = current_step_group.resolve()
        if current_step_group_key is not None:
            return tuple(
                record
                for record in records
                if str(record.key.scope.group_key) == current_step_group_key
            )
        if self.request.current_image is None:
            return records

        selector = RuntimeRecordSourceImageSetSelector(
            self.request.adapter,
            self.request.current_image,
        )
        if not selector.has_current_source_scope():
            return records
        if selector.has_template_current_source_scope():
            return records
        if not selector.records_have_metadata_source_scope(records):
            return records
        scoped_records = selector.select(records)
        if scoped_records:
            return scoped_records
        raise RuntimeError(
            f"Runtime artifact input {self.request.name!r} "
            f"({self.request.artifact_type.value}) has producer-group records, but none "
            "match the current source image scope."
        )

    def query_miss_records(self) -> RuntimeArtifactRecordResolution:
        candidate_records = RuntimeGroupMatchScope(
            group_key=None,
            match_group=False,
        ).runtime_scope(
            self.request.adapter,
            current_image=self.request.current_image,
        ).artifact_query_context().find(
            name=self.request.name,
            artifact_type=self.request.artifact_type,
        )
        if len(candidate_records) == 1:
            return (candidate_records[0],)
        return self.request.source_scope_policy.query_miss_records(
            self.request,
            candidate_records,
        )

@dataclass(frozen=True, slots=True)
class RuntimeArtifactUndeclaredInputResolution:
    """Resolve records when no artifact input plan was compiled for the name."""

    request: RuntimeArtifactRecordResolver
    resolved_group_key: str | None

    def resolve(self) -> tuple[StoredRuntimeValue, ...]:
        try:
            return (
                RuntimeGroupMatchScope(
                    group_key=self.request.group_key
                ).runtime_scope(
                    self.request.adapter,
                    current_image=self.request.current_image,
                ).artifact_query_context().resolve(
                    name=self.request.name,
                    artifact_type=self.request.artifact_type,
                ),
            )
        except RuntimeError:
            records = RuntimeGroupMatchScope(
                group_key=None,
                match_group=False,
            ).runtime_scope(
                self.request.adapter,
                current_image=self.request.current_image,
            ).artifact_query_context().find(
                name=self.request.name,
                artifact_type=self.request.artifact_type,
            )
            if records and self.resolved_group_key is None:
                scoped_resolution = (
                    self.request.source_scope_policy.axis_records_after_query_miss(
                        self.request,
                        records,
                    )
                )
                if scoped_resolution is not None:
                    return scoped_resolution
            raise

@dataclass(frozen=True, slots=True)
class RuntimeArtifactRecordLocationIdentity:
    """Physical runtime-artifact location identity for duplicate-read detection."""

    path: str
    backend: str

    @classmethod
    def from_record(
        cls,
        record: StoredRuntimeValue,
    ) -> "RuntimeArtifactRecordLocationIdentity":
        return cls(path=record.path, backend=record.backend)

@dataclass(frozen=True, slots=True)
class RuntimeArtifactRecordDeduplication:
    """Collapse repeated reads of the same persisted runtime record."""

    records: tuple[StoredRuntimeValue, ...]

    def unique_by_location(self) -> tuple[StoredRuntimeValue, ...]:
        records_by_location: OrderedDict[
            RuntimeArtifactRecordLocationIdentity,
            StoredRuntimeValue,
        ] = OrderedDict()
        for record in self.records:
            location_identity = RuntimeArtifactRecordLocationIdentity.from_record(
                record
            )
            if location_identity not in records_by_location:
                records_by_location[location_identity] = record
        return tuple(records_by_location.values())

def _is_global_grouped_input_request(
    input_plan: ArtifactInputPlan,
    group_key: str | None,
) -> bool:
    input_group_keys = input_plan.group_keys
    if input_group_keys is None:
        group_keys = ()
    else:
        group_keys = tuple(input_group_keys)
    if len(group_keys) <= 1:
        return False
    paths_by_group = input_plan.paths_by_group
    if paths_by_group is None:
        paths_by_group = {}
    return group_key is None and group_key not in paths_by_group
