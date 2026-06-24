"""Execution-local main-flow output lineage for FunctionStep execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from weakref import WeakKeyDictionary

from polystore.streaming.identity import StreamProducerIdentity

from openhcs.core.aligned_image_payload import AlignedImageSliceContext
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.path_pattern_matching import PathPatternTemplateMatcher

from openhcs.core.step_dependencies import StepInputDependencyKind
from openhcs.core.steps.function_output_identity import FunctionOutputIdentity
from openhcs.core.steps.function_output_identity import FunctionOutputPathAuthority
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
from openhcs.microscopes.microscope_interfaces import FilenameParser

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

    plan: FunctionStepExecutionPlan
    output_kind: str
    output_key: str
    artifact_kind: str | None = None

class FunctionStepOutputProducerIdentityAuthority:
    """Build stable producer identities for all FunctionStep output surfaces."""

    @staticmethod
    def build(
        request: FunctionStepOutputProducerIdentityRequest,
    ) -> StreamProducerIdentity:
        return StreamProducerIdentity.pipeline_output(
            output_kind=request.output_kind,
            output_key=request.output_key,
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

    @classmethod
    def from_output(
        cls,
        plan: FunctionStepExecutionPlan,
        output_path: str | Path,
        output_identity: FunctionOutputIdentity,
        output_context: AlignedImageSliceContext | None = None,
    ) -> "ProducedOutputSemantics":
        output_path_text = str(output_path)
        if output_context is None:
            output_context = AlignedImageSliceContext.anonymous_main_flow()
        return cls(
            producer_identity=FunctionStepOutputProducerIdentityAuthority.build(
                FunctionStepOutputProducerIdentityRequest(
                    plan=plan,
                    output_kind=output_context.output_kind,
                    output_key=output_context.output_key,
                    artifact_kind=output_context.artifact_kind,
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
        )

@dataclass(slots=True)
class StepOutputManifestStore:
    """Execution-local main-flow output lineage for shared VFS directories."""

    records_by_key: dict[StepOutputManifestKey, tuple[ProducedOutputSemantics, ...]] = field(
        default_factory=dict
    )

    def begin_step(self, plan: FunctionStepExecutionPlan) -> None:
        key = self.key_for_producer(plan)
        if key is None:
            return
        self.records_by_key[key] = ()

    def record_outputs(
        self,
        plan: FunctionStepExecutionPlan,
        output_records: Sequence[ProducedOutputSemantics],
    ) -> None:
        key = self.key_for_producer(plan)
        if key is None:
            return
        existing = self.records_for_key(key)
        records_by_path = {
            record.relative_output_path: record
            for record in (*existing, *tuple(output_records))
        }
        self.records_by_key[key] = tuple(records_by_path.values())

    def producer_records_for(
        self,
        plan: FunctionStepExecutionPlan,
    ) -> tuple[ProducedOutputSemantics, ...] | None:
        dependency = plan.main_input_dependency
        if dependency.kind is not StepInputDependencyKind.STEP_OUTPUT:
            return None
        if dependency.source_step_scope_id is None:
            return None
        return self.records_for_key(
            StepOutputManifestKey(dependency.source_step_scope_id, plan.axis_id)
        )

    def producer_paths_for(
        self,
        plan: FunctionStepExecutionPlan,
    ) -> tuple[str, ...] | None:
        records = self.producer_records_for(plan)
        if records is None:
            return None
        return tuple(record.relative_output_path for record in records)

    def produced_records_for(
        self,
        plan: FunctionStepExecutionPlan,
    ) -> tuple[ProducedOutputSemantics, ...]:
        key = self.key_for_producer(plan)
        if key is None:
            return ()
        return self.records_for_key(key)

    def produced_paths_for(
        self,
        plan: FunctionStepExecutionPlan,
    ) -> tuple[str, ...]:
        return tuple(
            record.relative_output_path
            for record in self.produced_records_for(plan)
        )

    def filter_to_producer_paths(
        self,
        plan: FunctionStepExecutionPlan,
        paths: Sequence[str],
        parser: FilenameParser,
    ) -> list[str]:
        producer_records = self.producer_records_for(plan)
        if producer_records is None:
            return list(paths)
        producer_records = self._select_requested_producer_records(
            plan,
            producer_records,
        )
        allowed = ProducedPathSet.from_records(producer_records, parser)
        selected = [
            path
            for path in paths
            if allowed.contains(path)
        ]
        if selected:
            return selected
        if paths:
            raise NoStepOutputManifestMatch
        return []

    def producer_paths_matching_pattern(
        self,
        plan: FunctionStepExecutionPlan,
        pattern: str,
        parser: FilenameParser,
    ) -> list[str]:
        """Return requested producer paths addressed by a detected pattern."""
        producer_records = self.producer_records_for(plan)
        if producer_records is None:
            return []
        producer_records = self._select_requested_producer_records(
            plan,
            producer_records,
        )
        return [
            record.relative_output_path
            for record in producer_records
            if ProducedPathSet.from_records((record,), parser).contains(pattern)
        ]

    def _select_requested_producer_records(
        self,
        plan: FunctionStepExecutionPlan,
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
        plan: FunctionStepExecutionPlan,
    ) -> frozenset[tuple[str, str, str | None]]:
        dependency = plan.main_input_dependency
        if dependency.kind is not StepInputDependencyKind.STEP_OUTPUT:
            return frozenset()
        return frozenset(
            (
                AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND,
                artifact_input.name,
                artifact_input.kind.value,
            )
            for artifact_input in plan.artifact_inputs.values()
            if (
                artifact_input.source_step_id in (
                    dependency.source_step_index,
                    "prev",
                )
                or artifact_input.source_step_scope_id == dependency.source_step_scope_id
            )
            and artifact_input.kind.participates_in_main_flow_output
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
        plan: FunctionStepExecutionPlan,
    ) -> StepOutputManifestKey | None:
        if not plan.step_scope_id:
            return None
        return StepOutputManifestKey(plan.step_scope_id, plan.axis_id)

    @staticmethod
    def relative_output_path(output_path: str, output_dir: Path) -> str:
        return str(Path(output_path).relative_to(output_dir))

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
                path = Path(value)
                tokens.add(path.as_posix())
                tokens.add(path.name)
            if record.filename_qualifier is not None:
                tokens.add(
                    FunctionOutputPathAuthority.filename_for_identity(
                        parser,
                        record.without_filename_qualifier(),
                    )
                )
        return cls(frozenset(tokens))

    def contains(self, path: str) -> bool:
        path_text = Path(path).as_posix()
        if path_text in self.tokens:
            return True
        matcher = PathPatternTemplateMatcher.from_pattern(path_text)
        if matcher is None:
            return False
        return any(matcher.matches(token) for token in self.tokens)
