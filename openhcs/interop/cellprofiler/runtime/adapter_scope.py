"""Runtime query scope and cache-key contracts for the CellProfiler adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan
from openhcs.core.runtime_artifact_queries import RuntimeArtifactQueryContext
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    RuntimeObjectLabelMeasurementQuery,
)
from openhcs.core.runtime_stores import RuntimeArtifactQuery
from openhcs.core.runtime_values import (
    RuntimeValue,
    normalize_artifact_value,
)
from openhcs.core.source_matching import SourceImageSetIdentity
from openhcs.interop.cellprofiler.runtime.payload_types import RuntimeArtifactNormalizationInput
from openhcs.interop.cellprofiler.runtime.source_identity import (
    CellProfilerCurrentImage,
    RuntimeRecordSourceImageSetSelector,
)

@dataclass(frozen=True, slots=True)
class CurrentSourceIdentityCacheScope:
    """Cache-key component for current-source identity provenance."""

    identities: frozenset[SourceImageSetIdentity] = frozenset()

@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeGroupMatchScope:
    """Shared group/match coordinate for runtime cache scopes."""

    group_key: str | None
    match_group: bool = True

    def runtime_scope(
        self,
        adapter: "CellProfilerRuntimeAdapter",
        *,
        current_image: CellProfilerCurrentImage | None = None,
    ) -> "CellProfilerRuntimeScope":
        return CellProfilerRuntimeScope(
            adapter=adapter,
            group_key=self.group_key,
            match_group=self.match_group,
            current_image=current_image,
        )

@dataclass(frozen=True, slots=True)
class ObjectMeasurementTableCacheKey(RuntimeGroupMatchScope):
    """Semantic cache key for object-subject measurement table queries."""

    object_name: str
    source_scope: CurrentSourceIdentityCacheScope = field(
        default_factory=CurrentSourceIdentityCacheScope
    )

@dataclass(frozen=True, slots=True)
class ObjectMeasurementTableIndexCacheKey(RuntimeGroupMatchScope):
    """Semantic cache key for object-subject measurement table indexes."""

    source_scope: CurrentSourceIdentityCacheScope = field(
        default_factory=CurrentSourceIdentityCacheScope
    )

@dataclass(frozen=True, slots=True)
class MeasurementTableAxisProjectionCacheKey:
    """Semantic cache key for axis-projecting an immutable measurement table set."""

    revision: int
    axis: MeasurementRowAxisField
    value: int
    table_identities: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class MultiplaneObjectMeasurementTableCacheKey:
    """Semantic cache key for object-feature tables aligned to label stacks."""

    revision: int
    axis_identity: str
    object_name: str
    feature_name: str
    label_domain: tuple[int, ...]
    source_scope: CurrentSourceIdentityCacheScope

@dataclass(frozen=True, slots=True)
class CellProfilerRuntimeScope(RuntimeGroupMatchScope):
    """Adapter runtime scope projected into cache and query coordinates."""

    adapter: "CellProfilerRuntimeAdapter"
    current_image: CellProfilerCurrentImage | None = None

    @property
    def resolved_group_key(self) -> str | None:
        if self.group_key is None:
            return self.adapter.group_key
        return self.group_key

    @property
    def group_cache_component(self) -> str | None:
        if self.match_group:
            return self.resolved_group_key
        return None

    @property
    def current_image_cache_component(self) -> int | None:
        if self.current_image is None:
            return None
        return id(self.current_image)

    @property
    def source_identity_cache_scope(self) -> CurrentSourceIdentityCacheScope:
        return CurrentSourceIdentityCacheScope(
            self.current_source_identity_cache_key()
        )

    def current_source_identity_cache_key(self) -> frozenset[SourceImageSetIdentity]:
        """Return the semantic source-plane identity for scoped table caches."""
        if self.current_image is None:
            return frozenset()
        return RuntimeRecordSourceImageSetSelector(
            adapter=self.adapter,
            current_image=self.current_image,
        ).current_source_plane_identities()

    def artifact_query_context(self) -> RuntimeArtifactQueryContext:
        return RuntimeArtifactQueryContext(
            store=self.adapter.runtime_value_store,
            axis_id=self.adapter.axis_scope.axis_id,
            group_key=self.group_cache_component,
            match_group=self.match_group,
        )

    def object_label_measurement_query(
        self,
        *,
        object_name: str,
        feature_name: str,
        label_domain: tuple[int, ...],
        image_number: int | None,
    ) -> RuntimeObjectLabelMeasurementQuery:
        return RuntimeObjectLabelMeasurementQuery(
            axis_id=self.adapter.axis_scope.axis_id,
            group_key=self.resolved_group_key,
            object_name=object_name,
            feature_name=feature_name,
            label_domain=label_domain,
            image_number=image_number,
        )

    def artifact_group_key(
        self,
        input_plan: ArtifactInputPlan,
        *,
        requested_group_key: str | None,
    ) -> str | None:
        for group_key in self.artifact_group_key_candidates(
            input_plan,
            requested_group_key=requested_group_key,
        ):
            if input_plan.for_group(group_key) is not None:
                return group_key
        return None

    def artifact_group_key_candidates(
        self,
        input_plan: ArtifactInputPlan,
        *,
        requested_group_key: str | None,
    ) -> tuple[str | None, ...]:
        candidates: list[str | None] = []
        input_group_keys = set(input_plan.group_keys or ())
        if requested_group_key is not None and requested_group_key in input_group_keys:
            candidates.append(requested_group_key)
        scoped_group_key = self.adapter.axis_scope.value_text_for_component(
            input_plan.group_component
        )
        if requested_group_key is None and scoped_group_key in input_group_keys:
            candidates.append(scoped_group_key)
        single_group_key = input_plan.single_group_key
        if single_group_key not in candidates:
            candidates.append(single_group_key)
        return tuple(candidates)

    def input_plan_query(
        self,
        input_plan: ArtifactInputPlan,
        *,
        group_key: str | None,
        backend: str,
    ) -> RuntimeArtifactQuery:
        return RuntimeArtifactQuery.from_input_plan(
            input_plan=input_plan,
            axis_id=self.adapter.axis_scope.axis_id,
            backend=backend,
            group_key=group_key,
        )

    def normalize_artifact_value(
        self,
        output_plan: ArtifactOutputPlan,
        value: RuntimeArtifactNormalizationInput,
    ) -> RuntimeValue:
        return normalize_artifact_value(
            output_plan,
            value,
            axis_id=self.adapter.axis_scope.axis_id,
        )
