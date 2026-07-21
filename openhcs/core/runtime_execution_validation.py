"""Validation primitives for runtime artifact execution state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from polystore.streaming.identity import StreamProducerIdentity

from openhcs.core.artifacts import ArtifactSpec, ArtifactType
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.runtime_exports import (
    RuntimeExportExpectation,
    RuntimeExportObservation,
    runtime_export_failures,
)
from openhcs.core.runtime_stores import (
    StoredRuntimeValue,
)
from openhcs.core.source_matching import (
    SourceImageSetIdentityPolicy,
    source_component_metadata_items,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.steps.function_output_manifest import (
    FunctionStepOutputProducerIdentityAuthority,
    FunctionStepOutputProducerIdentityRequest,
)
from openhcs.processing.materialization import Output


RuntimeArtifactViewerComponentIdentity = tuple[tuple[str, str], ...]


def runtime_artifact_viewer_component_identity(
    metadata: Mapping[str, object],
) -> RuntimeArtifactViewerComponentIdentity:
    """Return nominal component coordinates from source or viewer metadata."""

    return tuple(
        (component.value, str(value))
        for component, value in source_component_metadata_items(metadata)
    )


@dataclass(frozen=True, slots=True)
class RuntimeArtifactViewerPayloadExpectation:
    """Exact component and source-domain identity of one viewer payload."""

    components: RuntimeArtifactViewerComponentIdentity
    source_spatial_domain: SourceSpatialDomain

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        if not isinstance(self.source_spatial_domain, SourceSpatialDomain):
            raise TypeError(
                "Runtime artifact viewer payload source_spatial_domain must be "
                "SourceSpatialDomain."
            )

    @property
    def identity_key(
        self,
    ) -> tuple[
        RuntimeArtifactViewerComponentIdentity,
        tuple[int, int] | None,
        tuple[int, int] | None,
    ]:
        domain = self.source_spatial_domain
        return self.components, domain.origin_yx, domain.source_shape_yx


@dataclass(frozen=True, slots=True)
class RuntimeArtifactViewerExpectation:
    """One compiled artifact producer and its exact runtime viewer payloads."""

    producer_identity: StreamProducerIdentity
    payloads: tuple[RuntimeArtifactViewerPayloadExpectation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.producer_identity, StreamProducerIdentity):
            raise TypeError(
                "RuntimeArtifactViewerExpectation.producer_identity must be "
                "StreamProducerIdentity."
            )
        payloads = tuple(self.payloads)
        if any(
            not isinstance(payload, RuntimeArtifactViewerPayloadExpectation)
            for payload in payloads
        ):
            raise TypeError(
                "RuntimeArtifactViewerExpectation.payloads must contain "
                "RuntimeArtifactViewerPayloadExpectation values."
            )
        identity_keys = tuple(payload.identity_key for payload in payloads)
        if len(identity_keys) != len(set(identity_keys)):
            raise ValueError(
                "Runtime artifact viewer payload identities must be unique per producer."
            )
        object.__setattr__(self, "payloads", payloads)


@dataclass(slots=True)
class RuntimeArtifactExecutionExpectation:
    """Runtime artifacts and file exports expected from one execution."""

    artifact_kinds: frozenset[type[ArtifactType]]
    exports: RuntimeExportExpectation
    artifact_viewer: tuple[RuntimeArtifactViewerExpectation, ...] = ()

    @classmethod
    def from_output_specs(
        cls,
        output_specs: Iterable[ArtifactSpec],
        *,
        exports: RuntimeExportExpectation,
    ) -> "RuntimeArtifactExecutionExpectation":
        return cls(
            artifact_kinds=frozenset(spec.artifact_type for spec in output_specs),
            exports=exports,
        )

    @classmethod
    def from_compiled_contexts(
        cls,
        compiled_contexts: Mapping[str, ProcessingContext],
    ) -> "RuntimeArtifactExecutionExpectation":
        """Derive execution expectations from compiler-owned output plans."""
        output_specs = tuple(
            dict.fromkeys(
                ArtifactSpec.output(
                    output.name,
                    output.artifact_type,
                    materialization=output.materialization,
                    sidecar_role=output.sidecar_role,
                )
                for context in compiled_contexts.values()
                for step_plan in context.step_plans.values()
                for output in step_plan.artifact_outputs.values()
            )
        )
        return cls(
            artifact_kinds=frozenset(spec.artifact_type for spec in output_specs),
            exports=RuntimeExportExpectation.from_output_specs(output_specs),
            artifact_viewer=runtime_artifact_viewer_expectations(compiled_contexts),
        )

    def __post_init__(self) -> None:
        self.artifact_kinds = frozenset(
            ArtifactType.coerce(kind)
            for kind in self.artifact_kinds
        )
        if not isinstance(self.exports, RuntimeExportExpectation):
            raise TypeError(
                "RuntimeArtifactExecutionExpectation.exports must be "
                f"RuntimeExportExpectation, got {type(self.exports).__name__}."
            )
        self.artifact_viewer = tuple(self.artifact_viewer)
        if any(
            not isinstance(item, RuntimeArtifactViewerExpectation)
            for item in self.artifact_viewer
        ):
            raise TypeError(
                "RuntimeArtifactExecutionExpectation.artifact_viewer must contain "
                "RuntimeArtifactViewerExpectation values."
            )
        producers = tuple(item.producer_identity for item in self.artifact_viewer)
        if len(producers) != len(set(producers)):
            raise ValueError(
                "Runtime artifact viewer expectations require one entry per producer."
            )


def runtime_artifact_viewer_expectations(
    compiled_contexts: Mapping[str, ProcessingContext],
) -> tuple[RuntimeArtifactViewerExpectation, ...]:
    """Derive artifact-only viewer expectations from compiled plans and records."""

    expected_by_producer: dict[
        StreamProducerIdentity,
        list[RuntimeArtifactViewerPayloadExpectation],
    ] = {}
    for context in compiled_contexts.values():
        for plan in context.step_plans.values():
            if not plan.streaming_configs or not plan.owns_runtime_outputs:
                continue
            from openhcs.core.steps.function_artifact_materialization import (
                observed_runtime_artifact_materializations,
            )

            for materialization in observed_runtime_artifact_materializations(
                plan,
                context,
            ):
                output_plan = materialization.output_plan
                if plan.compiled_function_pattern.publishes_output_to_main_flow(
                    output_plan,
                    materialization.record.key.scope.value_text,
                ):
                    continue
                viewer_outputs = materialization.viewer_outputs(plan, context)
                if not viewer_outputs:
                    continue
                producer = FunctionStepOutputProducerIdentityAuthority.build(
                    FunctionStepOutputProducerIdentityRequest.from_artifact(
                        plan,
                        output_plan,
                    )
                )
                payloads = expected_by_producer.setdefault(producer, [])
                payloads.extend(
                    _runtime_artifact_viewer_output_payloads(
                        viewer_outputs,
                    )
                )

    return tuple(
        RuntimeArtifactViewerExpectation(
            producer_identity=producer,
            payloads=tuple(
                {
                    payload.identity_key: payload
                    for payload in payloads
                }.values()
            ),
        )
        for producer, payloads in expected_by_producer.items()
    )


def _runtime_artifact_viewer_output_payloads(
    outputs: tuple[Output, ...],
) -> tuple[RuntimeArtifactViewerPayloadExpectation, ...]:
    payloads: list[RuntimeArtifactViewerPayloadExpectation] = []
    for output in outputs:
        metadata = output.metadata
        if metadata is None:
            raise ValueError(
                f"Viewer materialization output {output.path!r} has no image metadata."
            )
        source_identity = metadata.source_provenance.scalar_source_identity
        if not source_identity.addressable:
            raise ValueError(
                f"Viewer materialization output {output.path!r} has no addressable "
                "source identity."
            )
        payloads.append(
            RuntimeArtifactViewerPayloadExpectation(
                components=tuple(
                    (component.value, str(value))
                    for component, value in source_component_metadata_items(
                        source_identity.component_metadata or {}
                    )
                ),
                source_spatial_domain=metadata.source_spatial_domain,
            )
        )
    return tuple(payloads)


@dataclass(slots=True)
class RuntimeArtifactExecutionObservation:
    """Observed runtime artifacts and file exports from one execution."""

    records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]]
    exports: RuntimeExportObservation
    source_image_set_identity_policy: SourceImageSetIdentityPolicy = (
        SourceImageSetIdentityPolicy()
    )

    @classmethod
    def from_contexts(
        cls,
        execution_contexts: Mapping[str, ProcessingContext],
    ) -> "RuntimeArtifactExecutionObservation":
        from openhcs.core.steps.function_artifact_materialization import (
            materialized_artifact_output_paths,
        )

        identity_policies = frozenset(
            context.source_image_set_identity_policy
            for context in execution_contexts.values()
        )
        if len(identity_policies) > 1:
            raise ValueError(
                "Runtime artifact observation contexts disagree on source image-set "
                "identity policy."
            )
        return cls(
            records_by_axis=runtime_records_by_axis(execution_contexts),
            exports=RuntimeExportObservation.from_output_paths(
                tuple(
                    path
                    for context in execution_contexts.values()
                    for plan in context.step_plans.values()
                    if plan.owns_runtime_outputs
                    for path in materialized_artifact_output_paths(plan, context)
                )
            ),
            source_image_set_identity_policy=next(
                iter(identity_policies),
                SourceImageSetIdentityPolicy(),
            ),
        )

    def __post_init__(self) -> None:
        self.records_by_axis = MappingProxyType(
            {
                str(axis): tuple(records)
                for axis, records in self.records_by_axis.items()
            }
        )
        if not isinstance(self.exports, RuntimeExportObservation):
            raise TypeError(
                "RuntimeArtifactExecutionObservation.exports must be "
                f"RuntimeExportObservation, got {type(self.exports).__name__}."
            )
        if not isinstance(
            self.source_image_set_identity_policy,
            SourceImageSetIdentityPolicy,
        ):
            raise TypeError(
                "RuntimeArtifactExecutionObservation source identity policy must use "
                "SourceImageSetIdentityPolicy."
            )

    @property
    def record_counts_by_axis(self) -> Mapping[str, Mapping[type[ArtifactType], int]]:
        return MappingProxyType(
            {
                axis: MappingProxyType(Counter(record.key.artifact_type for record in records))
                for axis, records in self.records_by_axis.items()
            }
        )


def runtime_records_by_axis(
    execution_contexts: Mapping[str, ProcessingContext],
) -> Mapping[str, tuple[StoredRuntimeValue, ...]]:
    """Return stored runtime records from compiled execution contexts."""
    records_by_axis: dict[str, tuple[StoredRuntimeValue, ...]] = {}
    for axis_id, context in execution_contexts.items():
        store = context.runtime_value_store
        records_by_axis[str(axis_id)] = tuple(store.observed_values)
    return MappingProxyType(records_by_axis)


def runtime_output_roots(
    execution_contexts: Mapping[str, ProcessingContext],
    explicit_output_root: Path | None = None,
) -> tuple[Path, ...]:
    """Return authoritative runtime output roots for compiled contexts."""
    return _runtime_output_roots(execution_contexts, explicit_output_root)


def _runtime_output_roots(
    execution_contexts: Mapping[str, ProcessingContext],
    explicit_output_root: Path | None,
) -> tuple[Path, ...]:
    roots: list[Path] = []
    for context in execution_contexts.values():
        if not context.step_plans:
            continue
        for plan in context.step_plans.values():
            if plan.output_plate_root is not None:
                roots.append(Path(plan.output_plate_root))
            if plan.materialized_output is not None:
                roots.append(Path(plan.materialized_output.plate_root))

    if roots:
        return tuple(dict.fromkeys(roots))
    if explicit_output_root is None:
        return ()
    return (Path(explicit_output_root),)


def runtime_artifact_execution_failures(
    expectation: RuntimeArtifactExecutionExpectation,
    observation: RuntimeArtifactExecutionObservation,
) -> tuple[str, ...]:
    """Return validation failures for runtime artifacts and file exports."""
    return (
        *_runtime_artifact_failures(expectation, observation),
        *runtime_export_failures(
            expectation.exports,
            observation.exports,
            observation.records_by_axis,
        ),
    )


def _runtime_artifact_failures(
    expectation: RuntimeArtifactExecutionExpectation,
    observation: RuntimeArtifactExecutionObservation,
) -> tuple[str, ...]:
    failures: list[str] = []
    for axis_id, counts in observation.record_counts_by_axis.items():
        for kind in sorted(
            expectation.artifact_kinds,
            key=lambda artifact_kind: artifact_kind.value,
        ):
            if counts.get(kind, 0) == 0:
                failures.append(
                    f"axis {axis_id!r} produced no runtime records for "
                    f"declared artifact kind {kind.value!r}"
                )
    return tuple(failures)
