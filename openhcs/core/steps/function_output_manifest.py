"""Execution-local main-flow output lineage for FunctionStep execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Sequence
from weakref import WeakKeyDictionary

from polystore.streaming.identity import StreamProducerIdentity

from openhcs.core.artifacts import ArtifactOutputPlan, ArtifactType, ImageArtifactType
from openhcs.core.callable_contract import ImagePayloadConsumption
from openhcs.core.aligned_image_payload import AlignedImageSliceContext
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.path_pattern_matching import PathPatternTemplateMatcher
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_metadata,
)

from openhcs.core.step_dependencies import StepInputDependencyKind
from openhcs.core.steps.function_output_identity import FunctionOutputIdentity
from openhcs.core.steps.function_output_identity import FunctionOutputPathAuthority
from openhcs.microscopes.microscope_interfaces import FilenameParser
from openhcs.core.compiled_step_plan import CompiledStepPlan


@lru_cache(maxsize=65536)
def _cached_relative_output_path(output_path: str, output_dir: str) -> str:
    """Return output path relative to a step output directory."""

    return str(Path(output_path).relative_to(Path(output_dir)))


@lru_cache(maxsize=65536)
def _cached_posix_path(path: str) -> str:
    """Return normalized POSIX spelling for manifest path matching."""

    return Path(path).as_posix()


@lru_cache(maxsize=65536)
def _cached_path_name(path: str) -> str:
    """Return filename component for manifest path matching."""

    return Path(path).name


@dataclass(frozen=True, slots=True)
class StepOutputManifestKey:
    """Identity for main-flow files produced by one step for one axis."""

    step_scope_id: str
    axis_id: str


class NoStepOutputManifestMatch(RuntimeError):
    """Raised when a stale directory pattern is not in producer lineage."""


@dataclass(frozen=True, slots=True)
class FunctionStepOutputProducerIdentityRequest:
    """Declared producer identity facts for a FunctionStep output surface."""

    ARTIFACT_OUTPUT_KIND: ClassVar[str] = "artifact"

    plan: CompiledStepPlan
    output_kind: str
    output_key: str
    projection_key: str
    artifact_kind: str | None = None

    @classmethod
    def from_main_flow(
        cls,
        plan: CompiledStepPlan,
        output_context: AlignedImageSliceContext,
    ) -> "FunctionStepOutputProducerIdentityRequest":
        return cls(
            plan=plan,
            output_kind=output_context.output_kind,
            output_key=output_context.output_key,
            projection_key=output_context.output_kind,
            artifact_kind=output_context.artifact_kind,
        )

    @classmethod
    def from_artifact(
        cls,
        plan: CompiledStepPlan,
        output_plan: ArtifactOutputPlan,
    ) -> "FunctionStepOutputProducerIdentityRequest":
        return cls(
            plan=plan,
            output_kind=cls.ARTIFACT_OUTPUT_KIND,
            output_key=output_plan.name,
            projection_key=output_plan.name,
            artifact_kind=output_plan.artifact_type.value,
        )


class FunctionStepOutputProducerIdentityAuthority:
    """Build stable producer identities for all FunctionStep output surfaces."""

    @staticmethod
    def build(
        request: FunctionStepOutputProducerIdentityRequest,
    ) -> StreamProducerIdentity:
        return StreamProducerIdentity.pipeline_output(
            output_kind=request.output_kind,
            output_key=request.output_key,
            projection_key=request.projection_key,
            step_name=request.plan.step_name,
            pipeline_position=request.plan.pipeline_position,
            step_scope_id=request.plan.step_scope_id,
            artifact_kind=request.artifact_kind,
        )


@dataclass(frozen=True, slots=True)
class ProducedOutputSemantics(FunctionOutputIdentity):
    """Semantic record for one output file produced by a FunctionStep."""

    producer_identity: StreamProducerIdentity
    output_path: str
    relative_output_path: str
    image_metadata: ImagePayloadMetadata | None = None

    def contextualize_image_payload(
        self,
        payload: RuntimeArrayData,
    ) -> RuntimeArrayData:
        """Attach this exact produced-output identity to a reloaded payload."""

        metadata = self.image_metadata or image_payload_metadata(payload)
        return metadata.with_source_component_metadata(
            self.component_metadata(metadata.source_component_metadata)
        ).attach_to(payload)

    def path_under(self, output_dir: str | Path) -> str:
        """Project this output's manifest-owned relative path under a new root."""

        return str(Path(output_dir) / self.relative_output_path)

    @property
    def output_context(self) -> AlignedImageSliceContext:
        """Return the declared main-flow context for this produced output."""

        return AlignedImageSliceContext(
            output_kind=self.producer_identity.output_kind,
            output_key=self.producer_identity.output_key,
            artifact_kind=self.producer_identity.artifact_kind,
        )

    @property
    def main_flow_address(
        self,
    ) -> tuple[
        str,
        str,
        str | None,
        tuple[tuple[str, str | int], ...],
    ]:
        """Return the semantic main-flow slot occupied by this output."""

        return (
            self.producer_identity.output_kind,
            self.producer_identity.output_key,
            self.producer_identity.artifact_kind,
            tuple(sorted(self.component_values.items())),
        )

    @property
    def is_image_payload(self) -> bool:
        """Return whether image persistence and streaming own this output."""
        artifact_kind = self.producer_identity.artifact_kind
        if artifact_kind is None:
            return True
        return ArtifactType.coerce(artifact_kind) is ImageArtifactType

    @classmethod
    def from_output(
        cls,
        plan: CompiledStepPlan,
        output_path: str | Path,
        output_identity: FunctionOutputIdentity,
        output_context: AlignedImageSliceContext | None = None,
        image_metadata: ImagePayloadMetadata | None = None,
    ) -> "ProducedOutputSemantics":
        output_path_text = str(output_path)
        if output_context is None:
            output_context = AlignedImageSliceContext.anonymous_main_flow()
        return cls(
            producer_identity=FunctionStepOutputProducerIdentityAuthority.build(
                FunctionStepOutputProducerIdentityRequest.from_main_flow(
                    plan,
                    output_context,
                )
            ),
            component_values=output_identity.component_values,
            extension=output_identity.extension,
            source=output_identity.source,
            filename_component_values=output_identity.filename_component_values,
            filename_qualifier=output_identity.filename_qualifier,
            output_path=output_path_text,
            relative_output_path=StepOutputManifestStore.relative_output_path(
                output_path_text,
                Path(plan.output_dir),
            ),
            image_metadata=image_metadata,
        )

    @classmethod
    def from_existing_main_flow_path(
        cls,
        plan: CompiledStepPlan,
        path: str | Path,
        parser: FilenameParser,
        *,
        output_context: AlignedImageSliceContext | None = None,
    ) -> "ProducedOutputSemantics":
        """Return producer lineage for an existing image path passed through a step."""
        path = Path(path)
        if output_context is None:
            output_context = AlignedImageSliceContext.anonymous_main_flow()
        parsed = parser.parse_filename(path.name) or {}
        extension = parsed.pop("extension", path.suffix) or None
        identity = FunctionOutputIdentity(
            component_values={
                str(key): value
                for key, value in parsed.items()
                if isinstance(value, (str, int))
            },
            extension=extension,
            source="existing main-flow path",
        )
        return cls(
            producer_identity=FunctionStepOutputProducerIdentityAuthority.build(
                FunctionStepOutputProducerIdentityRequest.from_main_flow(
                    plan,
                    output_context,
                )
            ),
            component_values=identity.component_values,
            extension=identity.extension,
            source=identity.source,
            filename_component_values=identity.filename_component_values,
            filename_qualifier=identity.filename_qualifier,
            output_path=_cached_posix_path(str(path)),
            relative_output_path=_cached_path_name(str(path)),
        )


@dataclass(slots=True)
class StepOutputManifestStore:
    """Execution-local main-flow output lineage for shared VFS directories."""

    records_by_key: dict[StepOutputManifestKey, tuple[ProducedOutputSemantics, ...]] = (
        field(default_factory=dict)
    )
    records_revision: int = 0
    selected_records_by_plan: dict[
        tuple[int, int],
        tuple[ProducedOutputSemantics, ...] | None,
    ] = field(default_factory=dict)
    producer_paths_by_pattern: dict[
        tuple[int, int, str, int],
        tuple[str, ...],
    ] = field(default_factory=dict)
    filtered_paths_by_plan: dict[
        tuple[int, int, tuple[str, ...], int],
        tuple[str, ...],
    ] = field(default_factory=dict)

    def begin_step(
        self,
        plan: CompiledStepPlan,
        input_records: Sequence[ProducedOutputSemantics] = (),
    ) -> None:
        key = self.key_for_producer(plan)
        if key is None:
            return
        self.records_by_key[key] = tuple(input_records)
        self._invalidate_record_selection_caches()

    def record_outputs(
        self,
        plan: CompiledStepPlan,
        output_records: Sequence[ProducedOutputSemantics],
        *,
        collapsed_input_domain: bool = False,
    ) -> None:
        key = self.key_for_producer(plan)
        if key is None:
            return
        existing = self.records_for_key(key)
        current_outputs = tuple(output_records)
        has_current_step_output = any(
            record.producer_identity.step_scope_id == plan.step_scope_id
            for record in existing
        )
        if existing and current_outputs and not has_current_step_output:
            inherited_addresses = frozenset(
                record.main_flow_address for record in existing
            )
            output_addresses = frozenset(
                record.main_flow_address for record in current_outputs
            )
            if (
                collapsed_input_domain
                or any(
                    invocation.contract.image_payload_consumption
                    is ImagePayloadConsumption.COMPOSED
                    for invocation in plan.compiled_function_pattern.iter_invocations()
                )
                or plan.compiled_function_pattern.is_grouped
                or not output_addresses.issubset(inherited_addresses)
            ):
                existing = ()
        records_by_address = {
            record.main_flow_address: record for record in (*existing, *current_outputs)
        }
        self.records_by_key[key] = tuple(records_by_address.values())
        self._invalidate_record_selection_caches()

    def _invalidate_record_selection_caches(self) -> None:
        self.records_revision += 1
        self.selected_records_by_plan.clear()
        self.producer_paths_by_pattern.clear()
        self.filtered_paths_by_plan.clear()

    def producer_records_for(
        self,
        plan: CompiledStepPlan,
    ) -> tuple[ProducedOutputSemantics, ...] | None:
        key = self._main_input_producer_key(plan)
        if key is None:
            return None
        return self.records_for_key(key)

    @staticmethod
    def _main_input_producer_key(
        plan: CompiledStepPlan,
    ) -> StepOutputManifestKey | None:
        dependency = plan.main_input_dependency
        if dependency.kind is not StepInputDependencyKind.STEP_OUTPUT:
            return None
        if dependency.source_step_scope_id is None:
            return None
        return StepOutputManifestKey(dependency.source_step_scope_id, plan.axis_id)

    def producer_paths_for(
        self,
        plan: CompiledStepPlan,
    ) -> tuple[str, ...] | None:
        records = self.producer_records_for(plan)
        if records is None:
            return None
        records = self._unique_output_path_records(records)
        return tuple(record.relative_output_path for record in records)

    def produced_records_for(
        self,
        plan: CompiledStepPlan,
    ) -> tuple[ProducedOutputSemantics, ...]:
        key = self.key_for_producer(plan)
        if key is None:
            return ()
        return self.records_for_key(key)

    def produced_paths_for(
        self,
        plan: CompiledStepPlan,
    ) -> tuple[str, ...]:
        return tuple(
            record.relative_output_path for record in self.produced_records_for(plan)
        )

    def producer_output_contexts_for_paths(
        self,
        plan: CompiledStepPlan,
        paths: Sequence[str],
        parser: FilenameParser,
    ) -> tuple[AlignedImageSliceContext, ...]:
        """Return producer output contexts aligned to concrete input paths."""

        producer_records = self._selected_unique_producer_records_for(plan)
        if producer_records is None:
            return tuple(
                AlignedImageSliceContext.anonymous_main_flow() for _path in paths
            )

        contexts: list[AlignedImageSliceContext] = []
        for path in paths:
            matching_records = tuple(
                record
                for record in producer_records
                if ProducedPathSet.from_records((record,), parser).contains(path)
            )
            if len(matching_records) != 1:
                raise NoStepOutputManifestMatch(
                    "Expected one producer output context for input path "
                    f"{path!r}, found {len(matching_records)}."
                )
            contexts.append(matching_records[0].output_context)
        return tuple(contexts)

    def filter_to_producer_paths(
        self,
        plan: CompiledStepPlan,
        paths: Sequence[str],
        parser: FilenameParser,
    ) -> list[str]:
        cache_key = (
            self.records_revision,
            id(plan),
            tuple(str(path) for path in paths),
            id(parser),
        )
        cached = self.filtered_paths_by_plan.get(cache_key)
        if cached is not None:
            return list(cached)

        producer_records = self._selected_unique_producer_records_for(plan)
        if producer_records is None:
            return list(paths)
        allowed = ProducedPathSet.from_records(producer_records, parser)
        selected = [path for path in paths if allowed.contains(path)]
        if selected:
            self.filtered_paths_by_plan[cache_key] = tuple(selected)
            return selected
        if paths:
            raise NoStepOutputManifestMatch
        self.filtered_paths_by_plan[cache_key] = ()
        return []

    def producer_paths_matching_pattern(
        self,
        plan: CompiledStepPlan,
        pattern: str,
        parser: FilenameParser,
    ) -> list[str]:
        """Return requested producer paths addressed by a detected pattern."""
        cache_key = (self.records_revision, id(plan), str(pattern), id(parser))
        cached = self.producer_paths_by_pattern.get(cache_key)
        if cached is not None:
            return list(cached)

        producer_records = self._selected_unique_producer_records_for(plan)
        if producer_records is None:
            return []
        selector = ProducedPathPatternSelector.from_pattern(pattern)
        selected = tuple(
            record.output_path
            for record in producer_records
            if selector.matches(ProducedPathSet.from_records((record,), parser))
        )
        self.producer_paths_by_pattern[cache_key] = selected
        return list(selected)

    def _selected_unique_producer_records_for(
        self,
        plan: CompiledStepPlan,
    ) -> tuple[ProducedOutputSemantics, ...] | None:
        cache_key = (self.records_revision, id(plan))
        if cache_key in self.selected_records_by_plan:
            return self.selected_records_by_plan[cache_key]

        producer_records = self.producer_records_for(plan)
        if producer_records is None:
            self.selected_records_by_plan[cache_key] = None
            return None
        selected = self._select_requested_producer_records(plan, producer_records)
        selected = self._unique_output_path_records(selected)
        self.selected_records_by_plan[cache_key] = selected
        return selected

    @staticmethod
    def _unique_output_path_records(
        records: Sequence[ProducedOutputSemantics],
    ) -> tuple[ProducedOutputSemantics, ...]:
        records_by_path: dict[str, ProducedOutputSemantics] = {}
        for record in records:
            records_by_path.setdefault(record.output_path, record)
        return tuple(records_by_path.values())

    def _select_requested_producer_records(
        self,
        plan: CompiledStepPlan,
        producer_records: Sequence[ProducedOutputSemantics],
    ) -> tuple[ProducedOutputSemantics, ...]:
        requested = self._requested_producer_outputs(plan)
        if not requested:
            return tuple(producer_records)
        selected = tuple(
            record
            for record in producer_records
            if (
                record.producer_identity.output_kind,
                record.producer_identity.output_key,
                record.producer_identity.artifact_kind,
            )
            in requested
        )
        if selected:
            return selected
        if all(
            AlignedImageSliceContext(
                output_kind=record.producer_identity.output_kind,
                output_key=record.producer_identity.output_key,
                artifact_kind=record.producer_identity.artifact_kind,
            ).is_anonymous_main_flow
            for record in producer_records
        ):
            return tuple(producer_records)
        raise NoStepOutputManifestMatch

    @staticmethod
    def _requested_producer_outputs(
        plan: CompiledStepPlan,
    ) -> frozenset[tuple[str, str, str | None]]:
        dependency = plan.main_input_dependency
        if (
            dependency.kind is not StepInputDependencyKind.STEP_OUTPUT
            or dependency.source_step_scope_id is None
        ):
            return frozenset()
        producer_scope_id = dependency.source_step_scope_id
        return frozenset(
            (
                AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND,
                edge.spec.name,
                edge.spec.artifact_type.value,
            )
            for invocation in plan.compiled_function_pattern.iter_invocations()
            for edge in invocation.artifact_input_edges
            if (
                edge.consumes_main_flow
                or (
                    edge.spec.parameter_name is None
                    and edge.spec.ref()
                    in invocation.contract.output_group_scope_sources
                    and edge.storage_plan is not None
                    and edge.storage_plan.source_step_scope_id == producer_scope_id
                )
            )
        )

    def records_for_key(
        self,
        key: StepOutputManifestKey,
    ) -> tuple[ProducedOutputSemantics, ...]:
        if key not in self.records_by_key:
            return ()
        return self.records_by_key[key]

    @staticmethod
    def allowed_path_tokens(
        records: Sequence[ProducedOutputSemantics],
        parser: FilenameParser,
    ) -> set[str]:
        return ProducedPathSet.from_records(records, parser).tokens

    @staticmethod
    def key_for_producer(
        plan: CompiledStepPlan,
    ) -> StepOutputManifestKey | None:
        if not plan.step_scope_id:
            return None
        return StepOutputManifestKey(plan.step_scope_id, plan.axis_id)

    @staticmethod
    def relative_output_path(output_path: str, output_dir: Path) -> str:
        return _cached_relative_output_path(output_path, str(output_dir))


_STEP_OUTPUT_MANIFESTS: WeakKeyDictionary[
    ProcessingContext,
    StepOutputManifestStore,
] = WeakKeyDictionary()


def step_output_manifest(context: ProcessingContext) -> StepOutputManifestStore:
    """Return execution-local main-flow output lineage for a context."""
    if context in _STEP_OUTPUT_MANIFESTS:
        return _STEP_OUTPUT_MANIFESTS[context]
    manifest = StepOutputManifestStore()
    _STEP_OUTPUT_MANIFESTS[context] = manifest
    return manifest


@dataclass(frozen=True, slots=True)
class ProducedPathSet:
    """Path membership authority for concrete and template producer anchors."""

    tokens: frozenset[str]

    @classmethod
    def from_records(
        cls,
        records: Sequence[ProducedOutputSemantics],
        parser: FilenameParser,
    ) -> "ProducedPathSet":
        tokens: set[str] = set()
        for record in records:
            for value in (record.relative_output_path, record.output_path):
                tokens.add(_cached_posix_path(value))
                tokens.add(_cached_path_name(value))
            if record.filename_qualifier is not None:
                tokens.add(
                    FunctionOutputPathAuthority.filename_for_identity(
                        parser,
                        record.without_filename_qualifier(),
                    )
                )
        return cls(frozenset(tokens))

    def contains(self, path: str) -> bool:
        return ProducedPathPatternSelector.from_pattern(path).matches(self)


@dataclass(frozen=True, slots=True)
class ProducedPathPatternSelector:
    """Prepared matcher for concrete and template producer path membership."""

    path: str
    matcher: PathPatternTemplateMatcher | None

    @classmethod
    def from_pattern(cls, path: str) -> "ProducedPathPatternSelector":
        path_text = _cached_posix_path(path)
        return cls(path_text, PathPatternTemplateMatcher.from_pattern(path_text))

    def matches(self, path_set: ProducedPathSet) -> bool:
        tokens = path_set.tokens
        if self.path in tokens:
            return True
        if self.matcher is None:
            return False
        return any(self.matcher.matches(token) for token in tokens)
