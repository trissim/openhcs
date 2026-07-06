"""Source identity authorities for the CellProfiler runtime adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta, extract_key_from_class_name

from openhcs.core.artifacts import RelationshipsArtifactType
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.core.runtime_values import (
    ImagePayloadMetadataInput,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectRelationship,
    image_payload_metadata,
)
from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
    SourceImageProvenancePlanes,
)
from openhcs.core.source_bindings import SourceBindingRuntimeContext
from openhcs.core.source_matching import (
    SourceImageSetIdentity,
    SourceImageSetIdentityCompatibility,
    SourceImageSetIdentityExactMatch,
    SourceImageSetIdentityPairPredicate,
    SourceImageSetIdentityPolicy,
)
from openhcs.core.source_path_identity import source_path_identity_key
from openhcs.core.source_plane_alignment import (
    SourcePayloadPlaneIdentitySequence,
    SourcePlaneIdentitySequence,
    SourcePlaneIdentitySequenceAlignment,
)

CellProfilerParsedMetadataValue = str | int | float | bool | None

class ParsedSourceMetadata(
    dict[str, CellProfilerParsedMetadataValue | SourceComponentMetadata]
):
    """Nominal source-metadata mapping decoded from CellProfiler source files."""

class MutableParsedSourceMetadata(ParsedSourceMetadata):
    """Mutable source-metadata mapping assembled during parser ingestion."""

CellProfilerCurrentImage = ImagePayloadMetadataInput
SourceScopedPayload = ImagePayloadMetadataInput | ObjectLabelSet | ObjectLabelPayload
_SOURCE_PLANE_IDENTITY_POLICY = SourceImageSetIdentityPolicy(
    plane_member_components=frozenset()
)

class RuntimeRegistryKeyDeclarationMixin:
    """Shared class declaration surface for AutoRegisterMeta registry roots."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: ClassVar[str | None] = None

class ParsedSourceCandidateABC(ABC):
    """Nominal source-candidate contract required for identity matching."""

    path: str
    resolved_path: str
    metadata: ParsedSourceMetadata

class RuntimeSourceIdentityAdapterABC(ABC):
    """Nominal adapter contract required by source identity authorities."""

    source_binding_context: SourceBindingRuntimeContext
    can_resolve_source_candidates: bool

    @abstractmethod
    def source_candidates(
        self,
        source_paths: tuple[str, ...],
    ) -> Sequence[ParsedSourceCandidateABC]:
        """Return parser-backed source candidates for runtime source paths."""

    @abstractmethod
    def source_binding_runtime_path_identities(
        self,
    ) -> "SourceBindingRuntimePathIdentities":
        """Return path identities for the active runtime source-binding scope."""

@dataclass(frozen=True, slots=True)
class ParsedSourceCandidatePathIdentity:
    """Path identities for matching runtime virtual paths to source candidates."""

    values: frozenset[str]

    @classmethod
    def from_candidate(
        cls,
        candidate: ParsedSourceCandidateABC,
    ) -> "ParsedSourceCandidatePathIdentity":
        return cls.from_paths((candidate.path, candidate.resolved_path))

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[str | None],
    ) -> "ParsedSourceCandidatePathIdentity":
        return cls(
            frozenset(
                source_path_identity_key(str(path))
                for path in paths
                if path is not None and str(path)
            )
        )

    @property
    def is_empty(self) -> bool:
        return not self.values

    def intersects(self, other: "ParsedSourceCandidatePathIdentity") -> bool:
        return not self.values.isdisjoint(other.values)

@dataclass(frozen=True, slots=True)
class SourceBindingRuntimePathIdentities:
    """Path identities by which runtime source-binding planes may be referenced."""

    current_step_input: ParsedSourceCandidatePathIdentity
    virtual_step_input: ParsedSourceCandidatePathIdentity

@dataclass(frozen=True, slots=True)
class RuntimeSourceProvenancePayloadPlaneResolution:
    """Resolve a payload plane from runtime virtual source provenance."""

    path_identities: SourceBindingRuntimePathIdentities
    source_image_provenance_planes: SourceImageProvenancePlanes

    def plane_index(self) -> int | None:
        current_plane = self.plane_index_for_identity(
            self.path_identities.current_step_input
        )
        if current_plane is not None:
            return current_plane
        return self.plane_index_for_identity(
            self.path_identities.virtual_step_input
        )

    def plane_index_for_identity(
        self,
        runtime_identity: ParsedSourceCandidatePathIdentity,
    ) -> int | None:
        """Return the unique provenance plane matching one runtime path identity."""
        if runtime_identity.is_empty:
            return None
        candidate_indexes = tuple(
            index
            for index, source_path in enumerate(
                self.source_image_provenance_planes.paths
            )
            if ParsedSourceCandidatePathIdentity.from_paths(
                (source_path,)
            ).intersects(runtime_identity)
        )
        unique_indexes = tuple(dict.fromkeys(candidate_indexes))
        if len(unique_indexes) == 1:
            return unique_indexes[0]
        return None

@dataclass(frozen=True, slots=True)
class SourcePathTemplateScope:
    """Source-path scope represented by OpenHCS grouped placeholder syntax."""

    paths: tuple[str, ...]
    placeholder_token: ClassVar[str] = "{iii}"

    @classmethod
    def from_paths(cls, paths: Sequence[str]) -> "SourcePathTemplateScope":
        return cls(tuple(str(path) for path in paths if str(path)))

    @property
    def is_template(self) -> bool:
        return any(self.placeholder_token in path for path in self.paths)

@dataclass(frozen=True, slots=True)
class SourceImageSetIdentityQuality:
    """Classify whether identities contain semantic source metadata."""

    identities: frozenset[SourceImageSetIdentity]

    @classmethod
    def from_identities(
        cls,
        identities: frozenset[SourceImageSetIdentity],
    ) -> "SourceImageSetIdentityQuality":
        return cls(identities)

    @property
    def has_metadata_scope(self) -> bool:
        return any(
            key != "source_path"
            for identity in self.identities
            for key, _value in identity.components
        )

@dataclass(frozen=True, slots=True)
class RuntimeRecordSourceImageSetSelector:
    """Select ambiguous runtime records for the active source image set."""

    adapter: RuntimeSourceIdentityAdapterABC
    current_image: CellProfilerCurrentImage | None

    def select_runtime_scope(
        self,
        records: tuple[StoredRuntimeValue, ...],
    ) -> tuple[StoredRuntimeValue, ...]:
        """Select records matching the current source and group-component scope."""
        scoped_records = records
        if len(scoped_records) > 1 and self.current_image is not None:
            selected = self.select(scoped_records)
            if selected:
                scoped_records = selected
        if len(scoped_records) > 1:
            selected = self.select_current_group_component(scoped_records)
            if selected:
                scoped_records = selected
        return scoped_records

    def has_current_source_scope(self) -> bool:
        """Return whether the current payload identifies one source scope."""
        return bool(
            self.current_source_plane_identities()
            or self.current_source_identities()
        )

    def has_template_current_source_scope(self) -> bool:
        """Return whether the current scope is a grouped source-path template."""
        return SourcePathTemplateScope.from_paths(
            self.current_source_paths()
        ).is_template

    def records_have_metadata_source_scope(
        self,
        records: tuple[StoredRuntimeValue, ...],
    ) -> bool:
        """Return whether records carry source metadata beyond output paths."""
        return any(
            SourceImageSetIdentityQuality.from_identities(
                self.record_source_identities(record)
            ).has_metadata_scope
            for record in records
        )

    def current_source_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        paths.extend(self.adapter.source_binding_context.current_step_input_files)
        if self.current_image is not None:
            metadata = image_payload_metadata(self.current_image)
            paths.extend(
                str(path)
                for path in (*metadata.source_image_provenance_planes.paths, metadata.source_path)
                if path is not None and str(path)
            )
        return tuple(dict.fromkeys(paths))

    def current_source_path_tokens(self) -> tuple[str, ...]:
        """Return path tokens that can encode a producer group component."""
        tokens: list[str] = []
        if self.current_image is not None:
            tokens.extend(
                image_payload_metadata(self.current_image).source_image_path_tokens
            )
        tokens.extend(str(path) for path in self.adapter.source_binding_context.current_step_input_files)
        return self.path_token_variants(tokens)

    def select_current_group_component(
        self,
        records: tuple[StoredRuntimeValue, ...],
    ) -> tuple[StoredRuntimeValue, ...]:
        """Select records whose group component matches the current source path."""
        group_keys = {
            str(record.key.scope.group_key)
            for record in records
            if record.key.scope.group_key is not None
        }
        source_tokens = self.current_source_path_tokens()
        selected_group_keys = {
            group_key
            for group_key in group_keys
            if any(group_key in token for token in source_tokens)
        }
        if len(selected_group_keys) == 1:
            selected_group_key = next(iter(selected_group_keys))
            return tuple(
                record
                for record in records
                if record.key.scope.group_key == selected_group_key
            )

        current_components = self.path_group_components(source_tokens)
        if len(current_components) != 1:
            return ()
        current_component = next(iter(current_components))
        return tuple(
            record
            for record in records
            if current_component
            in self.path_group_components(self.record_path_tokens(record))
        )

    @staticmethod
    def record_path_tokens(record: StoredRuntimeValue) -> tuple[str, ...]:
        return RuntimeRecordSourceImageSetSelector.path_token_variants((record.path,))

    @staticmethod
    def path_token_variants(paths: Sequence[str]) -> tuple[str, ...]:
        tokens: list[str] = []
        for token in paths:
            path = Path(token)
            tokens.extend((token, path.name, path.stem))
        return tuple(dict.fromkeys(tokens))

    @staticmethod
    def path_group_components(tokens: Sequence[str]) -> frozenset[str]:
        components: set[str] = set()
        for token in tokens:
            components.update(re.findall(r"(?:(?<=_)|^)w\d+(?=_|$)", token))
        return frozenset(components)

    def select(
        self,
        records: tuple[StoredRuntimeValue, ...],
    ) -> tuple[StoredRuntimeValue, ...]:
        current_plane_identities = self.current_source_plane_identities()
        plane_selected = tuple(
            record
            for record in records
            if SourceImageSetIdentityCompatibility.any_match(
                self.record_source_plane_identities(record),
                current_plane_identities,
            )
        )
        if plane_selected:
            return plane_selected
        current_identities = self.current_source_identities()
        if not current_identities:
            return ()
        return tuple(
            record
            for record in records
            if SourceImageSetIdentityCompatibility.any_match(
                self.record_source_identities(record),
                current_identities,
            )
        )

    def current_source_plane_identities(self) -> frozenset[SourceImageSetIdentity]:
        return self.current_source_identities(policy=_SOURCE_PLANE_IDENTITY_POLICY)

    def current_source_identities(
        self,
        *,
        policy: SourceImageSetIdentityPolicy = SourceImageSetIdentity.DEFAULT_POLICY,
    ) -> frozenset[SourceImageSetIdentity]:
        if self.current_image is not None:
            payload_identities = self.payload_source_identities(
                self.current_image,
                policy=policy,
            )
            if payload_identities:
                return payload_identities
        file_identities = self.source_identities_for_paths(
            self.adapter.source_binding_context.current_step_input_files,
            policy=policy,
        )
        if file_identities:
            return file_identities
        return frozenset()

    def record_source_plane_identities(
        self,
        record: StoredRuntimeValue,
    ) -> frozenset[SourceImageSetIdentity]:
        return self.record_source_identities(
            record,
            policy=_SOURCE_PLANE_IDENTITY_POLICY,
        )

    def record_source_identities(
        self,
        record: StoredRuntimeValue,
        *,
        policy: SourceImageSetIdentityPolicy = SourceImageSetIdentity.DEFAULT_POLICY,
    ) -> frozenset[SourceImageSetIdentity]:
        identities = set(self.payload_source_identities(record.value.data, policy=policy))
        identities.update(
            self.source_identities_for_paths(
                self.record_source_candidate_paths(record),
                policy=policy,
            )
        )
        if record.value.artifact_type is RelationshipsArtifactType:
            identities.update(
                self.relationship_source_identities(
                    ObjectRelationship.from_runtime_value(record.value),
                    policy=policy,
                )
            )
        return frozenset(identities)

    def record_source_candidate_paths(
        self,
        record: StoredRuntimeValue,
    ) -> tuple[str, ...]:
        """Return parser-facing source path projections for an artifact record."""
        path = Path(record.path)
        name = path.name
        stem = path.stem
        delimiter = f"_{record.key.name}_"
        prefixes = [record.path, name, stem]
        if delimiter in name:
            prefixes.append(name.split(delimiter, 1)[0])
        if delimiter in stem:
            prefixes.append(stem.split(delimiter, 1)[0])
        current_inputs = tuple(self.adapter.source_binding_context.current_step_input_files)
        current_by_stem = {Path(input_path).stem: input_path for input_path in current_inputs}
        prefixes.extend(
            current_by_stem[prefix]
            for prefix in tuple(prefixes)
            if prefix in current_by_stem
        )
        return tuple(dict.fromkeys(prefixes))

    def relationship_source_identities(
        self,
        relationship: ObjectRelationship,
        *,
        policy: SourceImageSetIdentityPolicy,
    ) -> frozenset[SourceImageSetIdentity]:
        identities: set[SourceImageSetIdentity] = set()
        identities.update(
            self.metadata_identities(
                (relationship.source_component_metadata,),
                fallback_source_path=relationship.source_path,
                policy=policy,
            )
        )
        identities.update(
            self.metadata_identities(
                relationship.source_image_provenance_planes.component_metadata,
                fallback_source_path=None,
                policy=policy,
            )
        )
        source_paths = tuple(
            str(path)
            for path in (
                *relationship.source_image_provenance_planes.paths,
                relationship.source_path,
            )
            if path is not None and str(path)
        )
        identities.update(self.source_identities_for_paths(source_paths, policy=policy))
        return frozenset(identities)

    def payload_source_identities(
        self,
        payload: SourceScopedPayload,
        *,
        policy: SourceImageSetIdentityPolicy = SourceImageSetIdentity.DEFAULT_POLICY,
    ) -> frozenset[SourceImageSetIdentity]:
        metadata = image_payload_metadata(payload)
        identities: set[SourceImageSetIdentity] = set()
        identities.update(
            self.metadata_identities(
                (metadata.source_component_metadata,),
                fallback_source_path=metadata.source_path,
                policy=policy,
            )
        )
        identities.update(
            self.metadata_identities(
                metadata.source_image_provenance_planes.component_metadata,
                fallback_source_path=None,
                policy=policy,
            )
        )
        source_paths = tuple(
            str(path)
            for path in (*metadata.source_image_provenance_planes.paths, metadata.source_path)
            if path is not None and str(path)
        )
        identities.update(self.source_identities_for_paths(source_paths, policy=policy))
        return frozenset(identities)

    def metadata_identities(
        self,
        metadata_values: tuple[ParsedSourceMetadata | None, ...],
        *,
        fallback_source_path: str | None,
        policy: SourceImageSetIdentityPolicy,
    ) -> frozenset[SourceImageSetIdentity]:
        identity_fallback_source_path = fallback_source_path
        if identity_fallback_source_path is None:
            identity_fallback_source_path = ""
        identities = {
            SourceImageSetIdentity.from_metadata(
                metadata,
                fallback_source_path=identity_fallback_source_path,
                policy=policy,
            )
            for metadata in metadata_values
            if metadata is not None and metadata
        }
        return frozenset(
            identity
            for identity in identities
            if identity.components != (("source_path", ""),)
        )

    def source_identities_for_paths(
        self,
        source_paths: tuple[str, ...],
        *,
        policy: SourceImageSetIdentityPolicy,
    ) -> frozenset[SourceImageSetIdentity]:
        if not source_paths or not self.adapter.can_resolve_source_candidates:
            return frozenset()
        return frozenset(
            SourceImageSetIdentity.from_metadata(
                candidate.metadata,
                fallback_source_path=candidate.resolved_path,
                policy=policy,
            )
            for candidate in self.adapter.source_candidates(source_paths)
        )

@dataclass(frozen=True, slots=True)
class CurrentSourcePayloadPlaneSelection:
    """Typed result for current-source payload plane selection."""

    plane_index: int | None = None
    ambiguous_plane_indexes: tuple[int, ...] = ()

    @property
    def is_matched(self) -> bool:
        return self.plane_index is not None

    @property
    def is_ambiguous(self) -> bool:
        return bool(self.ambiguous_plane_indexes)

    def require_unambiguous(self) -> None:
        """Raise when the selection represents an unresolved exact ambiguity."""
        if self.is_ambiguous:
            raise RuntimeError(
                "Current source image projection requires exactly one matching "
                "runtime image plane, got "
                f"{self.ambiguous_plane_indexes}."
            )

@dataclass(frozen=True, slots=True)
class CurrentSourcePayloadPlaneSelectionRequest:
    """Coordinates for selecting one payload plane from current source identity."""

    adapter: RuntimeSourceIdentityAdapterABC
    current_image: CellProfilerCurrentImage
    payload: SourceScopedPayload

    @property
    def current_selector(self) -> "RuntimeRecordSourceImageSetSelector":
        return RuntimeRecordSourceImageSetSelector(
            self.adapter,
            self.current_image,
        )

@dataclass(frozen=True, slots=True)
class CurrentSourcePayloadPlaneIdentityMatch:
    """Match current-source identities against payload plane identities."""

    identity_predicate_type: ClassVar[type[SourceImageSetIdentityPairPredicate]] = (
        SourceImageSetIdentityCompatibility
    )

    current_identities: frozenset[SourceImageSetIdentity]
    payload_identities: SourcePlaneIdentitySequence

    def select(self) -> CurrentSourcePayloadPlaneSelection:
        if not self.current_identities:
            return CurrentSourcePayloadPlaneSelection()
        if len(self.current_identities) > 1:
            return CurrentSourcePayloadPlaneSelection()
        if not any(self.payload_identities):
            return CurrentSourcePayloadPlaneSelection()
        candidate_indexes = tuple(
            index
            for index, identities in enumerate(self.payload_identities)
            if self.identity_predicate_type.any_match(
                identities,
                self.current_identities,
            )
        )
        if not candidate_indexes:
            return CurrentSourcePayloadPlaneSelection()
        if len(candidate_indexes) == 1:
            return CurrentSourcePayloadPlaneSelection(candidate_indexes[0])
        return self.ambiguous_selection(candidate_indexes)

    def ambiguous_selection(
        self,
        candidate_indexes: tuple[int, ...],
    ) -> CurrentSourcePayloadPlaneSelection:
        del candidate_indexes
        return CurrentSourcePayloadPlaneSelection()

@dataclass(frozen=True, slots=True)
class CurrentSourcePayloadPlaneExactIdentityMatch(
    CurrentSourcePayloadPlaneIdentityMatch
):
    """Require complete source-plane identity equality before selecting a plane."""

    identity_predicate_type: ClassVar[type[SourceImageSetIdentityPairPredicate]] = (
        SourceImageSetIdentityExactMatch
    )

    def ambiguous_selection(
        self,
        candidate_indexes: tuple[int, ...],
    ) -> CurrentSourcePayloadPlaneSelection:
        return CurrentSourcePayloadPlaneSelection(
            ambiguous_plane_indexes=candidate_indexes,
        )

@dataclass(frozen=True, slots=True)
class CurrentSourcePayloadPlaneSelectionScan:
    """Accumulator for stage selection results."""

    ambiguous_exact_selection: CurrentSourcePayloadPlaneSelection = field(
        default_factory=CurrentSourcePayloadPlaneSelection
    )
    terminal_selection: CurrentSourcePayloadPlaneSelection | None = None

    @property
    def has_terminal_selection(self) -> bool:
        return self.terminal_selection is not None

    def terminal_or_raise(self) -> CurrentSourcePayloadPlaneSelection:
        if self.terminal_selection is None:
            raise RuntimeError("Selection scan has no terminal selection.")
        return self.terminal_selection

    def final_selection(self) -> CurrentSourcePayloadPlaneSelection:
        if self.ambiguous_exact_selection.is_ambiguous:
            return self.ambiguous_exact_selection
        return CurrentSourcePayloadPlaneSelection()

@dataclass(frozen=True, slots=True)
class CurrentSourcePayloadPlaneSelectionOutcome(
    RuntimeRegistryKeyDeclarationMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Closed outcome policy for one current-source selection stage."""

    __key_extractor__ = staticmethod(extract_key_from_class_name)

    @classmethod
    def registered_outcomes(
        cls,
    ) -> tuple["CurrentSourcePayloadPlaneSelectionOutcome", ...]:
        registered_types = frozenset(cls.__registry__.values())
        return tuple(
            outcome_type()
            for outcome_type in (
                CurrentSourcePayloadPlaneSelectionOutcomeOrder.__mro__
            )
            if outcome_type in registered_types
        )

    @classmethod
    def outcome_for(
        cls,
        selection: CurrentSourcePayloadPlaneSelection,
    ) -> "CurrentSourcePayloadPlaneSelectionOutcome":
        for outcome in cls.registered_outcomes():
            if outcome.matches(selection):
                return outcome
        raise RuntimeError("No current-source selection outcome matched.")

    @abstractmethod
    def matches(self, selection: CurrentSourcePayloadPlaneSelection) -> bool:
        """Return whether this outcome handles the stage selection."""

    @abstractmethod
    def accumulate(
        self,
        scan: CurrentSourcePayloadPlaneSelectionScan,
        selection: CurrentSourcePayloadPlaneSelection,
    ) -> CurrentSourcePayloadPlaneSelectionScan:
        """Return the accumulated scan after this stage outcome."""

class MatchedCurrentSourcePayloadPlaneSelectionOutcome(
    CurrentSourcePayloadPlaneSelectionOutcome
):
    """Terminal outcome for a uniquely selected payload plane."""

    def matches(self, selection: CurrentSourcePayloadPlaneSelection) -> bool:
        return selection.is_matched

    def accumulate(
        self,
        scan: CurrentSourcePayloadPlaneSelectionScan,
        selection: CurrentSourcePayloadPlaneSelection,
    ) -> CurrentSourcePayloadPlaneSelectionScan:
        return CurrentSourcePayloadPlaneSelectionScan(
            ambiguous_exact_selection=scan.ambiguous_exact_selection,
            terminal_selection=selection,
        )

class AmbiguousCurrentSourcePayloadPlaneSelectionOutcome(
    CurrentSourcePayloadPlaneSelectionOutcome
):
    """Deferred outcome for ambiguous exact source-plane matches."""

    def matches(self, selection: CurrentSourcePayloadPlaneSelection) -> bool:
        return selection.is_ambiguous

    def accumulate(
        self,
        scan: CurrentSourcePayloadPlaneSelectionScan,
        selection: CurrentSourcePayloadPlaneSelection,
    ) -> CurrentSourcePayloadPlaneSelectionScan:
        if scan.ambiguous_exact_selection.is_ambiguous:
            return scan
        return CurrentSourcePayloadPlaneSelectionScan(
            ambiguous_exact_selection=selection,
            terminal_selection=scan.terminal_selection,
        )

class UnmatchedCurrentSourcePayloadPlaneSelectionOutcome(
    CurrentSourcePayloadPlaneSelectionOutcome
):
    """No-op outcome for an unmatched stage."""

    def matches(self, selection: CurrentSourcePayloadPlaneSelection) -> bool:
        del selection
        return True

    def accumulate(
        self,
        scan: CurrentSourcePayloadPlaneSelectionScan,
        selection: CurrentSourcePayloadPlaneSelection,
    ) -> CurrentSourcePayloadPlaneSelectionScan:
        del selection
        return scan

class CurrentSourcePayloadPlaneSelectionOutcomeOrder(
    MatchedCurrentSourcePayloadPlaneSelectionOutcome,
    AmbiguousCurrentSourcePayloadPlaneSelectionOutcome,
    UnmatchedCurrentSourcePayloadPlaneSelectionOutcome,
):
    """MRO-declared current-source selection outcome order."""

    @abstractmethod
    def _ordering_carrier_only(self) -> None:
        """Keep this MRO carrier out of the executable outcome registry."""

@dataclass(frozen=True, slots=True)
class CurrentSourcePayloadPlaneSelectionStage(ABC, metaclass=AutoRegisterMeta):
    """One stage in current-source payload plane selection."""

    __registry_key__ = "stage_key"
    __skip_if_no_key__ = True
    stage_key: ClassVar[str | None] = None

    @classmethod
    def registered_stages(
        cls,
    ) -> tuple["CurrentSourcePayloadPlaneSelectionStage", ...]:
        registered_types = frozenset(cls.__registry__.values())
        return tuple(
            stage_type()
            for stage_type in CurrentSourcePayloadPlaneSelectionStageOrder.__mro__
            if stage_type in registered_types
        )

    @abstractmethod
    def select(
        self,
        request: CurrentSourcePayloadPlaneSelectionRequest,
    ) -> CurrentSourcePayloadPlaneSelection:
        """Return a selected plane or an unmatched result."""

class ExactSourcePayloadPlaneSelectionStage(CurrentSourcePayloadPlaneSelectionStage):
    """Select only when current and payload planes identify the same source image."""

    stage_key = "exact_source_plane"

    def select(
        self,
        request: CurrentSourcePayloadPlaneSelectionRequest,
    ) -> CurrentSourcePayloadPlaneSelection:
        return CurrentSourcePayloadPlaneExactIdentityMatch(
            current_identities=request.current_selector.current_source_plane_identities(),
            payload_identities=SourcePayloadPlaneIdentitySequence(
                request.payload,
                _SOURCE_PLANE_IDENTITY_POLICY,
            ).identities(),
        ).select()

class ImageSetSourcePayloadPlaneSelectionStage(CurrentSourcePayloadPlaneSelectionStage):
    """Select by image-set identity, ignoring declared plane-member components."""

    stage_key = "image_set"

    def select(
        self,
        request: CurrentSourcePayloadPlaneSelectionRequest,
    ) -> CurrentSourcePayloadPlaneSelection:
        return CurrentSourcePayloadPlaneIdentityMatch(
            current_identities=request.current_selector.current_source_identities(),
            payload_identities=SourcePayloadPlaneIdentitySequence(
                request.payload,
                SourceImageSetIdentity.DEFAULT_POLICY,
            ).identities(),
        ).select()

class RuntimePathSourcePayloadPlaneSelectionStage(CurrentSourcePayloadPlaneSelectionStage):
    """Select from runtime source path provenance when path identity is available."""

    stage_key = "runtime_path"

    def select(
        self,
        request: CurrentSourcePayloadPlaneSelectionRequest,
    ) -> CurrentSourcePayloadPlaneSelection:
        plane_index = RuntimeSourceProvenancePayloadPlaneResolution(
            path_identities=request.adapter.source_binding_runtime_path_identities(),
            source_image_provenance_planes=(
                image_payload_metadata(request.payload).source_image_provenance_planes
            ),
        ).plane_index()
        return CurrentSourcePayloadPlaneSelection(plane_index)

class CurrentSourcePayloadPlaneSelectionStageOrder(
    ExactSourcePayloadPlaneSelectionStage,
    ImageSetSourcePayloadPlaneSelectionStage,
    RuntimePathSourcePayloadPlaneSelectionStage,
):
    """MRO-declared proof strength for current-source payload plane selection."""

    @abstractmethod
    def _ordering_carrier_only(self) -> None:
        """Keep this MRO carrier out of the executable stage registry."""

class CurrentSourcePayloadPlaneSelectionAuthority:
    """Run current-source payload plane selection from strongest to weakest proof."""

    @classmethod
    def select(
        cls,
        request: CurrentSourcePayloadPlaneSelectionRequest,
    ) -> CurrentSourcePayloadPlaneSelection:
        if request.current_selector.has_template_current_source_scope():
            return CurrentSourcePayloadPlaneSelection()
        scan = CurrentSourcePayloadPlaneSelectionScan()
        for stage in CurrentSourcePayloadPlaneSelectionStage.registered_stages():
            selection = stage.select(request)
            scan = CurrentSourcePayloadPlaneSelectionOutcome.outcome_for(
                selection
            ).accumulate(scan, selection)
            if scan.has_terminal_selection:
                return scan.terminal_or_raise()
        return scan.final_selection()
