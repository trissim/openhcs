"""Source-candidate parsing and matching authorities for CellProfiler runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, replace
from operator import attrgetter
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import Backend
from openhcs.core.process_local_cache import ProcessLocalBoundedCache
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    MetadataSelector,
    NamedSourceBinding,
    SourceFilterClause,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceBindingRuntimeContext,
    SourceRuntimePathLookup,
)
from openhcs.core.source_load_plan import SourceLoadPlan
from openhcs.core.source_image_semantics import SourceImagePayloadSemantics
from openhcs.core.source_matching import (
    SourceAxisMetadataScope,
    SourceImageSetIdentity,
    is_image_path,
    merge_source_metadata,
    metadata_from_rules,
    semantic_source_metadata_value,
    source_component_metadata_values,
    source_filters_match,
    source_metadata_component,
    source_metadata_value,
    source_metadata_values_equal,
)
from openhcs.core.source_path_identity import (
    source_path_identity_key,
    source_paths_equal,
)
from openhcs.core.source_metadata import (
    SourceMetadataIdentityProjection,
    SourceMetadataRoleView,
)
from openhcs.core.runtime_values import (
    ImagePayloadSourceMetadataContext,
    image_payload_metadata,
)
from openhcs.core.source_image_provenance import SourceImageIdentity
from openhcs.interop.cellprofiler.runtime.adapter_profile import (
    AdapterProfileLog,
    SourceCandidateProfileEvent,
)
from openhcs.interop.cellprofiler.runtime.adapter_protocols import (
    CellProfilerFilenameParser,
    CellProfilerProcessingContext,
    RequireProcessingContextBoundaryPolicy,
)
from openhcs.interop.cellprofiler.runtime.payload_types import ImagePayloadValue
from openhcs.interop.cellprofiler.runtime.source_identity import (
    MutableParsedSourceMetadata,
    ParsedSourceCandidateABC,
    ParsedSourceCandidatePathIdentity,
    ParsedSourceMetadata,
    SourceBindingRuntimePathIdentities,
)

SourceOrderCacheValue = int | tuple[str, ...]

SOURCE_CANDIDATE_PROCESS_CACHE: ProcessLocalBoundedCache[
    tuple[Hashable, ...],
    tuple["ParsedSourceCandidate", ...],
] = ProcessLocalBoundedCache(max_entries=64)
SOURCE_CANDIDATE_METADATA_PROCESS_CACHE: ProcessLocalBoundedCache[
    tuple[Hashable, ...],
    tuple[ParsedSourceMetadata, ParsedSourceMetadata],
] = ProcessLocalBoundedCache(max_entries=8192)
CELLPROFILER_SOURCE_ORDER_PROCESS_CACHE: ProcessLocalBoundedCache[
    tuple[Hashable, ...],
    SourceOrderCacheValue,
] = ProcessLocalBoundedCache(max_entries=512)
SOURCE_CANDIDATE_MATCH_PROCESS_CACHE: ProcessLocalBoundedCache[
    tuple[Hashable, ...],
    tuple["ParsedSourceCandidate", ...],
] = ProcessLocalBoundedCache(max_entries=4096)

@dataclass(frozen=True, slots=True)
class SourceCandidateRuntimeCache:
    """Process-local cache authority for parsed source candidates."""

    adapter: "CellProfilerRuntimeAdapter"
    file_paths: tuple[str, ...]

    def candidates(self) -> tuple["ParsedSourceCandidate", ...]:
        cache_key = self.cache_key()
        cache = SOURCE_CANDIDATE_PROCESS_CACHE
        cached = cache.cached_value(cache_key)
        if cached is not None:
            return cached
        started_at = time.perf_counter()
        candidates = _parse_source_candidates(
            self.file_paths,
            self.adapter,
            universe=self.universe(),
        )
        cache.store_value(cache_key, candidates)
        AdapterProfileLog.source_candidates(
            SourceCandidateProfileEvent(
                label="source_candidates_parse",
                seconds=time.perf_counter() - started_at,
                count=len(candidates),
            )
        )
        return candidates

    def cache_key(self) -> tuple[Hashable, ...]:
        """Return the semantic cache key for this source-candidate projection."""
        context = self.adapter.source_binding_context
        parser = RequireProcessingContextBoundaryPolicy(
            self.adapter
        ).context.microscope_handler.parser
        universe = self.universe()
        path_identities = tuple(
            path_resolution.cache_identity(context)
            for file_path in self.file_paths
            for path_resolution in universe.path_projection(file_path).paths()
        )
        return (
            path_identities,
            universe.projects_virtual_paths,
            self.adapter.source_binding_plan.metadata_rules,
            parser.semantic_identity(),
        )

    def universe(self) -> "SourceCandidateRuntimeUniverse":
        """Return the path-projection universe for this source-candidate request."""
        return SourceCandidateRuntimeUniverse(
            adapter=self.adapter,
            file_paths=self.file_paths,
        )


@dataclass(frozen=True, slots=True)
class SourceCandidateRuntimeUniverse:
    """Path-projection semantics for one parsed source-candidate universe."""

    adapter: "CellProfilerRuntimeAdapter"
    file_paths: tuple[str, ...]

    @property
    def projects_virtual_paths(self) -> bool:
        context = self.adapter.source_binding_context
        return not (
            bool(context.pipeline_input_files)
            and self.file_paths == context.pipeline_input_files
        )

    def path_projection(self, file_path: str) -> "SourceCandidatePathProjection":
        return SourceCandidatePathProjection(
            file_path,
            self.adapter,
            include_virtual_paths=self.projects_virtual_paths,
        )

@dataclass(frozen=True, slots=True)
class SourceBindingAxisAliasResolution:
    """Resolve aliases that identify a source-binding plane request."""

    requested_aliases: tuple[str, ...]
    bindings: tuple[NamedSourceBinding, ...]

    def aliases(self) -> tuple[str, ...]:
        requested = tuple(str(alias) for alias in self.requested_aliases)
        if requested:
            return requested
        binding_aliases = tuple(
            str(binding.alias)
            for binding in self.bindings
            if binding.alias
        )
        unique_aliases = tuple(dict.fromkeys(binding_aliases))
        if len(unique_aliases) == 1:
            return unique_aliases
        return ()

@dataclass(frozen=True, slots=True)
class SourceBindingMatchedIndexSet:
    """Matched source-binding plane indexes with projection helpers."""

    indexes: tuple[int, ...]

    def __bool__(self) -> bool:
        return bool(self.indexes)

    @property
    def single_index(self) -> int | None:
        if len(self.indexes) == 1:
            return self.indexes[0]
        return None

    def plane_index_for_identity(
        self,
        plane_candidates: "ParsedSourceCandidateSet",
        path_identity: ParsedSourceCandidatePathIdentity,
    ) -> int | None:
        return plane_candidates.unique_index_for_identity(
            path_identity,
            allowed_indexes=self.indexes,
        )

    def covers_candidate_set(
        self,
        plane_candidates: "ParsedSourceCandidateSet",
    ) -> bool:
        """Return whether the matched indexes cover the whole candidate axis."""
        return bool(plane_candidates.candidates) and tuple(self.indexes) == tuple(
            range(len(plane_candidates.candidates))
        )

@dataclass(frozen=True, slots=True)
class SourceBindingPlaneMatch:
    """Matched source-binding axis candidates with their source alias."""

    alias: str
    universe: "SourceBindingPlaneCandidateUniverse"
    index_set: SourceBindingMatchedIndexSet

@dataclass(frozen=True, slots=True)
class SourceBindingPathIdentityPlaneResolution(ABC, metaclass=AutoRegisterMeta):
    """Base for source-binding plane selection from runtime path identities."""

    __registry_key__ = "scope_key"
    __skip_if_no_key__ = True
    scope_key: ClassVar[str | None] = None

    path_identities: SourceBindingRuntimePathIdentities
    match: SourceBindingPlaneMatch

    @abstractmethod
    def plane_index(self) -> int | None:
        """Return the matched source-binding plane index, when unique."""

@dataclass(frozen=True, slots=True)
class CurrentStepInputSourceBindingPlaneResolution(
    SourceBindingPathIdentityPlaneResolution
):
    """Resolve a source-binding plane from the current step input file identity."""

    scope_key = "current_step_input"

    def plane_index(self) -> int | None:
        return self.match.index_set.plane_index_for_identity(
            self.match.universe.plane_candidates,
            self.path_identities.current_step_input,
        )

@dataclass(frozen=True, slots=True)
class RuntimeSourceProvenanceSourceBindingPlaneResolution(
    SourceBindingPathIdentityPlaneResolution
):
    """Resolve a source-binding plane from runtime virtual source provenance."""

    scope_key = "runtime_source_provenance"

    def plane_index(self) -> int | None:
        return self.match.index_set.plane_index_for_identity(
            self.match.universe.plane_candidates,
            self.path_identities.virtual_step_input,
        )

@dataclass(frozen=True, slots=True)
class SourceBindingRequestBase(ABC):
    """Shared nominal fields for source-binding request records."""

    alias: str
    binding: NamedSourceBinding

@dataclass(frozen=True, slots=True)
class SourceBindingMatchPlanRequest:
    """Typed request for deriving target metadata from an image-set match plan."""

    alias: str
    binding_plan: "SourceBindingImageSetMatchScope"
    universe: "SourceBindingMatchCandidateUniverse"

@dataclass(frozen=True, slots=True)
class SourceAliasOrderIndexRequest(SourceBindingMatchPlanRequest):
    """Source candidate and match-plan context for source-order alignment."""

    candidate: "ParsedSourceCandidate"

@dataclass(frozen=True, slots=True)
class SourceBindingImageSetMatchScope:
    """Source-binding match plan and bindings for image-set alignment."""

    plan: SourceBindingMatchPlan | None
    bindings: tuple[NamedSourceBinding, ...]

    def binding_for_alias(self, alias: str) -> NamedSourceBinding | None:
        """Return the binding for ``alias`` within this image-set match scope."""
        for binding in self.bindings:
            if binding.alias == alias:
                return binding
        return None

@dataclass(frozen=True, slots=True)
class CellProfilerImageNumberCandidateContext:
    """Candidate universe for resolving one source path to a CP image number."""

    source_path: str
    candidates: tuple["ParsedSourceCandidate", ...]

@dataclass(frozen=True, slots=True)
class CellProfilerImageNumberMatchedContext:
    """Matched source candidate with its image-set candidate universe."""

    matched_candidate: "ParsedSourceCandidate"
    candidates: tuple["ParsedSourceCandidate", ...]

@dataclass(frozen=True, slots=True)
class CellProfilerImageNumberResolution:
    """CellProfiler ImageNumber lookup result with explicit absence."""

    value: int | None = None

@dataclass(frozen=True, slots=True)
class CellProfilerImageNumberMap:
    """Adapter-axis map between source-order paths and CP ImageNumber values."""

    by_source_order_path: Mapping[str, int]
    source_path_by_image_number: Mapping[int, str]

    def image_number_for_source_order_path(
        self,
        source_order_path: str,
    ) -> CellProfilerImageNumberResolution:
        return CellProfilerImageNumberResolution(
            self.by_source_order_path.get(source_order_path)
        )

    def source_path_for_image_number(self, image_number: int) -> str | None:
        return self.source_path_by_image_number.get(int(image_number))


CELLPROFILER_IMAGE_NUMBER_MAP_PROCESS_CACHE: ProcessLocalBoundedCache[
    tuple[Hashable, ...],
    CellProfilerImageNumberMap,
] = ProcessLocalBoundedCache(max_entries=512)


@dataclass(frozen=True, slots=True)
class CellProfilerImageNumberResolver:
    """Resolve source paths to CellProfiler image numbers with explicit absence flow."""

    adapter: CellProfilerRuntimeAdapter

    @classmethod
    def for_adapter(
        cls,
        adapter: CellProfilerRuntimeAdapter,
    ) -> "CellProfilerImageNumberResolver":
        return cls(adapter)

    def image_number_for_paths(self, source_paths: tuple[str, ...]) -> int | None:
        source_path = None
        if source_paths:
            source_path = source_paths[0]
        return self.image_number_for_path(source_path)

    def image_number_start_for_paths(
        self,
        source_paths: tuple[str, ...],
    ) -> int | None:
        """Return the first CP ImageNumber represented by a source-path group."""
        image_numbers = self.image_numbers_for_paths(source_paths)
        if not image_numbers:
            return None
        return min(image_numbers)

    def image_number_start_for_axis_scope(
        self,
        axis_scope: SourceAxisMetadataScope,
    ) -> int:
        """Return the first CP ImageNumber matching the adapter axis scope."""
        pipeline_paths = self.pipeline_paths()
        if axis_scope.has_component and self.adapter.can_resolve_source_candidates:
            candidates = self.adapter.source_candidates(pipeline_paths)
            image_numbers = self.image_numbers_by_set(candidates)
            for candidate in SourceCandidateMatcher.ordered_source_candidates(
                candidates
            ):
                if not axis_scope.matches_metadata(self.candidate_metadata(candidate)):
                    continue
                image_number = image_numbers.get(candidate.source_identity)
                if image_number is not None:
                    return image_number

        for index, path in enumerate(sorted(pipeline_paths), start=1):
            metadata = self.source_metadata_for_path(path)
            if metadata is not None and axis_scope.matches_metadata(metadata):
                return index

        if not self.adapter.can_resolve_source_candidates:
            return 1

        parser = RequireProcessingContextBoundaryPolicy(
            self.adapter
        ).context.microscope_handler.parser
        for index, path in enumerate(sorted(pipeline_paths), start=1):
            parsed = parser.parse_filename(Path(path).name)
            if parsed is None:
                parsed = {}
            if axis_scope.matches_metadata(parsed):
                return index
        return 1

    def source_path_for_image_number(self, image_number: int) -> str | None:
        """Return a representative source path for one CellProfiler ImageNumber."""
        if not self.adapter.can_resolve_source_candidates:
            return None
        return self.image_number_map().source_path_for_image_number(image_number)

    def image_numbers_for_paths(self, source_paths: tuple[str, ...]) -> tuple[int, ...]:
        """Return ordered CP ImageNumbers represented by source paths."""
        numbers_by_slice = self.image_numbers_by_source_path_index(source_paths)
        numbers: list[int] = []
        seen: set[int] = set()
        for image_number in numbers_by_slice.values():
            if image_number is None or image_number in seen:
                continue
            numbers.append(image_number)
            seen.add(image_number)
        return tuple(numbers)

    def image_numbers_by_source_path_index(
        self,
        source_paths: tuple[str, ...],
    ) -> Mapping[int, int]:
        """Return CP ImageNumbers keyed by source-path tuple index."""
        if not source_paths:
            return MappingProxyType({})
        image_number_map = self.image_number_map()
        image_numbers: dict[int, int] = {}
        for index, source_path in enumerate(source_paths):
            image_number = image_number_map.image_number_for_source_order_path(
                self.adapter.cellprofiler_source_order_path(source_path)
            ).value
            if image_number is not None:
                image_numbers[index] = int(image_number)
        return MappingProxyType(image_numbers)

    def image_number_for_path(self, source_path: str | None) -> int | None:
        return self.image_number_resolution(source_path).value

    def image_number_resolution(
        self,
        source_path: str | None,
    ) -> CellProfilerImageNumberResolution:
        if source_path is None:
            return CellProfilerImageNumberResolution()
        if not self.adapter.can_resolve_source_candidates:
            return CellProfilerImageNumberResolution()
        return self.image_number_map().image_number_for_source_order_path(
            self.adapter.cellprofiler_source_order_path(source_path)
        )

    def image_number_map(self) -> CellProfilerImageNumberMap:
        """Return the cached source-order path to CP ImageNumber map."""
        pipeline_paths = self.pipeline_paths()
        if not pipeline_paths:
            return CellProfilerImageNumberMap(
                MappingProxyType({}),
                MappingProxyType({}),
            )
        axis_scope = self.adapter.source_axis_metadata_scope()
        cache_key = (
            "image_number_map",
            SourceCandidateRuntimeCache(self.adapter, pipeline_paths).cache_key(),
            axis_scope.component_values,
            cellprofiler_source_order_identity(self.adapter),
        )
        cache = CELLPROFILER_IMAGE_NUMBER_MAP_PROCESS_CACHE
        cached = cache.cached_value(cache_key)
        if cached is not None:
            return cached

        candidates = SourceCandidateMatcher.axis_scoped_candidates(
            self.adapter.source_candidates(pipeline_paths),
            axis_scope,
            metadata_for_candidate=self.candidate_metadata,
        )
        image_numbers = self.image_numbers_by_set(candidates)
        by_source_order_path: dict[str, int] = {}
        source_path_by_image_number: dict[int, str] = {}
        for candidate in SourceCandidateMatcher.ordered_source_candidates(candidates):
            image_number = image_numbers.get(candidate.source_identity)
            if image_number is None:
                continue
            source_path_by_image_number.setdefault(int(image_number), candidate.path)
            for source_path in self.source_metadata_candidate_paths(
                candidate.path,
                extra_paths=(candidate.resolved_path, *candidate.paths),
            ):
                by_source_order_path[
                    self.adapter.cellprofiler_source_order_path(source_path)
                ] = int(image_number)
        return cache.store_value(
            cache_key,
            CellProfilerImageNumberMap(
                MappingProxyType(by_source_order_path),
                MappingProxyType(source_path_by_image_number),
            ),
        )

    def candidate_context(
        self,
        source_path: str,
    ) -> CellProfilerImageNumberCandidateContext | None:
        pipeline_paths = self.pipeline_paths()
        if not pipeline_paths:
            return None
        axis_scope = self.adapter.source_axis_metadata_scope()
        return CellProfilerImageNumberCandidateContext(
            source_path=source_path,
            candidates=SourceCandidateMatcher.axis_scoped_candidates(
                self.adapter.source_candidates(pipeline_paths),
                axis_scope,
                metadata_for_candidate=self.candidate_metadata,
            ),
        )

    def matched_context(
        self,
        context: CellProfilerImageNumberCandidateContext,
    ) -> CellProfilerImageNumberMatchedContext | None:
        matched_candidate = self.matched_source_candidate(
            context.source_path,
            context.candidates,
        )
        if matched_candidate is None:
            return None
        return CellProfilerImageNumberMatchedContext(
            matched_candidate=matched_candidate,
            candidates=context.candidates,
        )

    def pipeline_paths(self) -> tuple[str, ...]:
        return tuple(
            path
            for path in self.adapter.source_binding_context.pipeline_source_candidate_files
            if is_image_path(path)
        )

    def candidate_metadata(self, candidate: "ParsedSourceCandidate") -> ParsedSourceMetadata:
        """Return source metadata for a parsed candidate in adapter path space."""
        metadata = self.source_metadata_for_path(
            candidate.path,
            extra_paths=(candidate.resolved_path,),
        )
        if metadata is not None:
            return metadata
        return candidate.metadata

    def source_metadata_for_path(
        self,
        path: str,
        *,
        extra_paths: tuple[str, ...] = (),
    ) -> ParsedSourceMetadata | None:
        """Return source metadata for every path identity used by CP ordering."""
        metadata_by_path = self.adapter.source_binding_context.source_metadata_by_path
        for candidate_path in self.source_metadata_candidate_paths(
            path,
            extra_paths=extra_paths,
        ):
            metadata = metadata_by_path.get(candidate_path)
            if metadata is not None:
                return metadata
        return None

    def source_metadata_candidate_paths(
        self,
        path: str,
        *,
        extra_paths: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return path spellings that may key source metadata for one source."""
        return tuple(
            dict.fromkeys(
                (
                    path,
                    str(Path(path).resolve(strict=False)),
                    self.adapter.cellprofiler_source_order_path(path),
                    *extra_paths,
                )
            )
        )

    def matched_source_candidate(
        self,
        source_path: str,
        candidates: tuple["ParsedSourceCandidate", ...],
    ) -> "ParsedSourceCandidate | None":
        source_identity = self.source_order_path_identity((source_path,))
        for candidate in candidates:
            if self.source_order_path_identity(candidate.paths).intersects(source_identity):
                return candidate
        return None

    def source_order_path_identity(
        self,
        paths: tuple[str, ...],
    ) -> ParsedSourceCandidatePathIdentity:
        return ParsedSourceCandidatePathIdentity.from_paths(
            tuple(self.adapter.cellprofiler_source_order_path(path) for path in paths)
        )

    @staticmethod
    def image_numbers_by_set(
        candidates: tuple["ParsedSourceCandidate", ...],
    ) -> Mapping[SourceImageSetIdentity, int]:
        image_numbers: dict[SourceImageSetIdentity, int] = {}
        for candidate in SourceCandidateMatcher.ordered_source_candidates(candidates):
            if candidate.source_identity not in image_numbers:
                image_numbers[candidate.source_identity] = len(image_numbers) + 1
        return MappingProxyType(image_numbers)

@dataclass(frozen=True, slots=True)
class SourceBindingPlaneCandidateContext:
    """Candidate universe for resolving an alias to a current source plane."""

    request: SourceBindingRequestBase
    universe: "SourceBindingPlaneCandidateUniverse"
    match_scope: "SourceBindingImageSetMatchScope"

    @classmethod
    def from_adapter(
        cls,
        adapter: CellProfilerRuntimeAdapter,
        alias: str,
    ) -> "SourceBindingPlaneCandidateContext | None":
        binding = adapter.source_binding_plan.binding_for_alias(alias)
        if binding is None:
            return None
        source_context = adapter.source_binding_context
        if not source_context.step_input_files:
            return None

        step_candidates = adapter.source_candidates(source_context.step_input_files)
        pipeline_candidates = adapter.source_candidates(
            source_context.pipeline_source_candidate_files
        )
        candidate_universe = SourceBindingPlaneCandidateUniverse.from_binding(
            binding=binding,
            adapter=adapter,
            step_candidates=step_candidates,
            pipeline_candidates=pipeline_candidates,
        )
        return cls(
            request=SourceBindingRequestBase(alias=alias, binding=binding),
            universe=candidate_universe,
            match_scope=SourceBindingImageSetMatchScope(
                plan=adapter.source_binding_plan.match_plan,
                bindings=adapter.source_binding_plan.bindings,
            ),
        )

    def match(self) -> SourceBindingPlaneMatch | None:
        index_set = self.universe.matched_axis_indexes(
            self.request,
            self.match_scope,
        )
        if not index_set:
            return None
        return SourceBindingPlaneMatch(
            alias=self.request.alias,
            universe=self.universe,
            index_set=index_set,
        )

@dataclass(frozen=True, slots=True)
class SourceBindingPlaneCandidateUniverse:
    """Origin-aware source universe for resolving alias-to-plane indexes."""

    axis_scope: "SourceAxisMetadataScope"
    plane_candidates: "ParsedSourceCandidateSet"
    candidate_universe: "SourceBindingMatchCandidateUniverse"

    def matched_axis_indexes(
        self,
        request: SourceBindingRequestBase,
        match_scope: "SourceBindingImageSetMatchScope",
    ) -> SourceBindingMatchedIndexSet:
        target_candidates = SourceCandidateMatcher.match_candidates(
            candidates=self.candidate_universe.target_candidates,
            binding=request.binding,
            inherit_components={},
        )
        current_candidates = self.candidate_universe.with_target_candidates(
            target_candidates
        ).image_set_candidates(request.alias, match_scope)
        return SourceBindingMatchedIndexSet(
            self.plane_candidates.indexes_for_candidates(current_candidates)
        )

    @classmethod
    def from_binding(
        cls,
        *,
        binding: NamedSourceBinding,
        adapter: CellProfilerRuntimeAdapter,
        step_candidates: tuple["ParsedSourceCandidate", ...],
        pipeline_candidates: tuple["ParsedSourceCandidate", ...],
    ) -> "SourceBindingPlaneCandidateUniverse":
        candidate_source = (
            pipeline_candidates
            if binding.origin is SourceBindingOrigin.PIPELINE_START
            else step_candidates
        )
        ordered_candidates = SourceCandidateMatcher.ordered_binding_candidates(
            binding=binding,
            candidates=candidate_source,
        )
        plane_candidates = cls._prefer_virtual_plane_candidates(
            ordered_candidates,
            adapter.source_binding_context,
        )
        axis_scope = adapter.source_axis_metadata_scope()
        candidate_universe = SourceBindingMatchCandidateUniverse(
            step_input_candidates=step_candidates,
            target_candidates=candidate_source,
            pipeline_candidates=pipeline_candidates,
        )
        return cls(
            axis_scope=axis_scope,
            plane_candidates=ParsedSourceCandidateSet(
                candidate_universe.axis_scoped_candidates(
                    plane_candidates,
                    axis_scope,
                )
            ),
            candidate_universe=candidate_universe,
        )

    @staticmethod
    def _prefer_virtual_plane_candidates(
        candidates: tuple["ParsedSourceCandidate", ...],
        context: SourceBindingRuntimeContext,
    ) -> tuple["ParsedSourceCandidate", ...]:
        """Return plane candidates using virtual paths for known source identities."""
        preferred: list[ParsedSourceCandidate] = []
        for candidate in candidates:
            virtual_paths = context.virtual_source_paths_by_identity.get(
                source_path_identity_key(candidate.resolved_path),
                (),
            )
            if not virtual_paths:
                preferred.append(candidate)
                continue
            preferred.extend(
                replace(
                    candidate,
                    path=virtual_path,
                    virtual_path=virtual_path,
                )
                for virtual_path in virtual_paths
            )
        return tuple(preferred)

@dataclass(frozen=True, slots=True)
class SourceBindingMatchCandidateUniverse:
    """Source candidates needed for image-set match-plan filtering."""

    step_input_candidates: tuple["ParsedSourceCandidate", ...]
    target_candidates: tuple["ParsedSourceCandidate", ...]
    pipeline_candidates: tuple["ParsedSourceCandidate", ...]

    def with_target_candidates(
        self,
        target_candidates: tuple["ParsedSourceCandidate", ...],
    ) -> "SourceBindingMatchCandidateUniverse":
        return type(self)(
            step_input_candidates=self.step_input_candidates,
            target_candidates=target_candidates,
            pipeline_candidates=self.pipeline_candidates,
        )

    def match_plan_request(
        self,
        alias: str,
        match_scope: "SourceBindingImageSetMatchScope",
    ) -> SourceBindingMatchPlanRequest:
        return SourceBindingMatchPlanRequest(
            alias=alias,
            binding_plan=match_scope,
            universe=self,
        )

    def image_set_candidates(
        self,
        alias: str,
        match_scope: "SourceBindingImageSetMatchScope",
    ) -> tuple["ParsedSourceCandidate", ...]:
        if (
            match_scope.plan is None
            or not self.step_input_candidates
            or not self.target_candidates
        ):
            return self.target_candidates
        cache_key = self.image_set_candidates_cache_key(alias, match_scope)
        cache = SOURCE_CANDIDATE_MATCH_PROCESS_CACHE
        cached = cache.cached_value(cache_key)
        if cached is not None:
            return cached
        return cache.store_value(
            cache_key,
            SourceCandidateMatcher.match_plan_candidates(
                self.match_plan_request(alias, match_scope)
            ),
        )

    def image_set_candidates_cache_key(
        self,
        alias: str,
        match_scope: "SourceBindingImageSetMatchScope",
    ) -> tuple[Hashable, ...]:
        """Return the semantic identity for one image-set match-plan request."""
        return (
            "source_image_set_match",
            alias,
            match_scope,
            tuple(candidate.candidate_identity for candidate in self.step_input_candidates),
            tuple(candidate.candidate_identity for candidate in self.target_candidates),
            tuple(candidate.candidate_identity for candidate in self.pipeline_candidates),
        )

    def axis_scoped_candidates(
        self,
        candidates: tuple["ParsedSourceCandidate", ...],
        axis_scope: "SourceAxisMetadataScope",
    ) -> tuple["ParsedSourceCandidate", ...]:
        return SourceCandidateMatcher.axis_scoped_candidates(
            candidates,
            axis_scope,
            constraining_candidates=self.step_input_candidates,
        )

@dataclass(frozen=True, slots=True)
class CollectionAttributeProjection:
    """Descriptor projecting one member attribute from a tuple collection."""

    collection_attr: str
    member_attr: str

    def __get__(
        self,
        instance: "PipelineStartSourceLoadRequest | None",
        owner: type["PipelineStartSourceLoadRequest"] | None = None,
    ) -> tuple[str, ...]:
        del owner
        if instance is None:
            return ()
        collection = attrgetter(self.collection_attr)(instance)
        member = attrgetter(self.member_attr)
        return tuple(member(item) for item in collection)

@dataclass(frozen=True, slots=True)
class PipelineStartSourceLoadRequest:
    """Typed request for loading pipeline-start source payloads."""

    adapter: CellProfilerRuntimeAdapter
    selected_sources: tuple["ParsedSourceCandidate", ...]
    backend: str
    source_load_plan: SourceLoadPlan
    identity_paths = CollectionAttributeProjection("selected_sources", "path")

    @property
    def storage_paths(self) -> tuple[str, ...]:
        if Backend(self.backend) is Backend.VIRTUAL_WORKSPACE:
            return self.identity_paths
        return tuple(source.resolved_path for source in self.selected_sources)

@dataclass(frozen=True, slots=True)
class PipelineStartSourcePayloadRequest(PipelineStartSourceLoadRequest):
    """One loaded source payload with its source and storage identity."""

    payload: ImagePayloadValue
    source_path: str
    storage_path: str | None
    source_metadata: ParsedSourceMetadata

    @classmethod
    def from_candidate(
        cls,
        *,
        payload: ImagePayloadValue,
        source: "ParsedSourceCandidate",
        storage_path: str,
        load_request: PipelineStartSourceLoadRequest,
    ) -> "PipelineStartSourcePayloadRequest":
        return cls(
            adapter=load_request.adapter,
            selected_sources=load_request.selected_sources,
            backend=load_request.backend,
            source_load_plan=load_request.source_load_plan,
            payload=payload,
            source_path=source.path,
            storage_path=storage_path,
            source_metadata=source.metadata,
        )

    @property
    def metadata_source_path(self) -> str:
        if self.storage_path is not None:
            return self.storage_path
        return self.source_path

    @property
    def context(self) -> CellProfilerProcessingContext:
        return RequireProcessingContextBoundaryPolicy(self.adapter).context

    def with_payload(
        self,
        payload: ImagePayloadValue,
    ) -> "PipelineStartSourcePayloadRequest":
        return replace(self, payload=payload)

@dataclass(frozen=True, slots=True)
class SourceCandidatePathResolution:
    """Candidate, storage, and virtual workspace paths for one source file."""

    path: str
    resolved_path: str
    virtual_path: str | None = None

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                path
                for path in (self.path, self.resolved_path, self.virtual_path)
                if path is not None
            )
        )

    @property
    def path_identity(self) -> ParsedSourceCandidatePathIdentity:
        """Return path identities by which runtime context may reference this source."""
        return ParsedSourceCandidatePathIdentity.from_paths(self.paths)

    def metadata_paths(self, context: SourceBindingRuntimeContext) -> tuple[str, ...]:
        paths = [self.path, self.resolved_path]
        if context.step_input_dir is not None and not Path(self.path).is_absolute():
            paths.append(str(Path(context.step_input_dir) / self.path))
        if self.virtual_path is not None:
            paths.append(self.virtual_path)
        return tuple(dict.fromkeys(paths))

    def cache_identity(
        self,
        context: SourceBindingRuntimeContext,
    ) -> tuple[Hashable, ...]:
        """Return the source-context identity this candidate resolution depends on."""
        return (
            self.path,
            source_path_identity_key(self.resolved_path),
            context.metadata_identity_for_paths(self.metadata_paths(context)),
            self.virtual_path,
        )

@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedSourceCandidate(SourceCandidatePathResolution, ParsedSourceCandidateABC):
    """One parsed file candidate used for source-binding selector resolution."""

    filename: str
    metadata: ParsedSourceMetadata
    source_identity: SourceImageSetIdentity = field(init=False)
    candidate_identity: "ParsedSourceCandidateIdentity" = field(init=False)

    def __post_init__(self) -> None:
        source_identity = SourceImageSetIdentity.from_metadata(
            self.metadata,
            fallback_source_path=self.resolved_path,
        )
        object.__setattr__(self, "source_identity", source_identity)
        object.__setattr__(
            self,
            "candidate_identity",
            ParsedSourceCandidateIdentity(
                source_identity=source_identity,
                resolved_path_identity=source_path_identity_key(self.resolved_path),
            ),
        )

    def cache_identity(
        self,
        context: SourceBindingRuntimeContext,
    ) -> tuple[Hashable, ...]:
        """Return source-context identity, falling back to parsed metadata."""
        context_metadata_identity = context.metadata_identity_for_paths(
            self.metadata_paths(context)
        )
        if any(metadata for _path, metadata in context_metadata_identity):
            metadata_identity = context_metadata_identity
        else:
            metadata_identity = SourceMetadataIdentityProjection(self.metadata).items()
        return (
            self.path,
            source_path_identity_key(self.resolved_path),
            metadata_identity,
            self.virtual_path,
        )

@dataclass(frozen=True, slots=True)
class ParsedSourceCandidateSet:
    """Ordered candidate set with path-identity indexing semantics."""

    candidates: tuple[ParsedSourceCandidate, ...]

    def indexes_for_candidates(
        self,
        candidates: tuple[ParsedSourceCandidate, ...],
    ) -> tuple[int, ...]:
        path_identities = tuple(candidate.path_identity for candidate in candidates)
        return tuple(
            index
            for index, candidate in enumerate(self.candidates)
            if any(
                candidate.path_identity.intersects(path_identity)
                for path_identity in path_identities
            )
        )

    def unique_index_for_identity(
        self,
        path_identity: ParsedSourceCandidatePathIdentity,
        *,
        allowed_indexes: tuple[int, ...],
    ) -> int | None:
        if path_identity.is_empty:
            return None
        candidate_indexes = tuple(
            index
            for index in allowed_indexes
            if index < len(self.candidates)
            and self.candidates[index].path_identity.intersects(path_identity)
        )
        unique_indexes = tuple(dict.fromkeys(candidate_indexes))
        if len(unique_indexes) == 1:
            return unique_indexes[0]
        return None

    def candidate_at(self, index: int) -> ParsedSourceCandidate:
        return self.candidates[index]

@dataclass(slots=True)
class SourceMetadataMatchConstraint:
    """Allowed metadata values for one image-set matching field."""

    field_name: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        self.field_name = str(self.field_name)
        values = tuple(dict.fromkeys(str(value) for value in self.values))
        if not values:
            raise ValueError("SourceMetadataMatchConstraint.values cannot be empty.")
        self.values = values

    def intersect(
        self,
        values: tuple[str, ...],
    ) -> "SourceMetadataMatchConstraint":
        """Return this constraint restricted to another allowed-value set."""
        allowed = frozenset(values)
        intersection = tuple(value for value in self.values if value in allowed)
        if not intersection:
            raise RuntimeError(
                f"Conflicting image-set match values for field {self.field_name!r}: "
                f"{self.values!r} versus {values!r}."
            )
        return SourceMetadataMatchConstraint(self.field_name, intersection)

    def matches(self, candidate: "ParsedSourceCandidate") -> bool:
        """Return whether the candidate metadata satisfies this constraint."""
        metadata_value = semantic_source_metadata_value(
            candidate.metadata,
            self.field_name,
        )
        return metadata_value is not None and any(
            source_metadata_values_equal(metadata_value, value)
            for value in self.values
        )

@dataclass(frozen=True, slots=True)
class SourceCandidateSelectorResolution:
    """Effective source selector semantics for matching and cache identity."""

    component_selectors: Mapping[str, str]
    inherited_component_items: tuple[tuple[str, str], ...]

    @classmethod
    def from_binding(
        cls,
        binding: NamedSourceBinding,
        inherit_components: Mapping[str, str],
    ) -> "SourceCandidateSelectorResolution":
        component_selectors = MappingProxyType({
            selector.component.value: selector.value
            for selector in binding.selector.components
        })
        explicit_metadata_fields = {
            selector.field for selector in binding.selector.metadata
        }
        explicit_selector_components = {
            component
            for field_name in (
                *component_selectors,
                *explicit_metadata_fields,
            )
            for component in (source_metadata_component(field_name),)
            if component is not None
        }
        effective_components = (
            {
                **{
                    name: value
                    for name, value in inherit_components.items()
                    if name not in component_selectors
                    and name not in explicit_metadata_fields
                    and source_metadata_component(name)
                    not in explicit_selector_components
                },
                **component_selectors,
            }
            if binding.selector.inherit_current_scope
            else component_selectors
        )
        inherited_component_items = tuple(sorted(
            (str(name), str(value))
            for name, value in effective_components.items()
            if name not in component_selectors
        ))
        return cls(
            component_selectors=component_selectors,
            inherited_component_items=inherited_component_items,
        )

@dataclass(frozen=True, slots=True)
class ParsedSourceCandidateIdentity:
    """Identity for collapsing aliases without collapsing distinct source files."""

    source_identity: SourceImageSetIdentity
    resolved_path_identity: str

    @classmethod
    def from_candidate(
        cls,
        candidate: ParsedSourceCandidate,
    ) -> "ParsedSourceCandidateIdentity":
        return cls(
            source_identity=candidate.source_identity,
            resolved_path_identity=source_path_identity_key(candidate.resolved_path),
        )

@dataclass(frozen=True, slots=True)
class ParsedSourceCandidateCollection:
    """Ordered set semantics for parsed source-binding candidates."""

    candidates: tuple[ParsedSourceCandidate, ...]

    def deduplicated(self) -> tuple[ParsedSourceCandidate, ...]:
        deduplicated: list[ParsedSourceCandidate] = []
        seen: set[ParsedSourceCandidateIdentity] = set()
        for candidate in self.candidates:
            identity = candidate.candidate_identity
            if identity in seen:
                continue
            deduplicated.append(candidate)
            seen.add(identity)
        return tuple(deduplicated)

@dataclass(frozen=True, slots=True)
class MatchedSourceCandidatesRequest(SourceBindingRequestBase):
    """Typed request for fail-loud source-candidate selection."""

    matched: tuple[ParsedSourceCandidate, ...]
    candidates: tuple[ParsedSourceCandidate, ...]
    source_description: str

    @classmethod
    def from_resolution(
        cls,
        request: SourceBindingResolutionRequest,
        *,
        matched: tuple[ParsedSourceCandidate, ...],
        candidates: tuple[ParsedSourceCandidate, ...],
        source_description: str,
    ) -> "MatchedSourceCandidatesRequest":
        return cls(
            alias=request.alias,
            binding=request.binding,
            matched=matched,
            candidates=candidates,
            source_description=source_description,
        )

def _parse_source_candidates(
    file_paths: tuple[str, ...],
    adapter: CellProfilerRuntimeAdapter,
    *,
    universe: SourceCandidateRuntimeUniverse,
) -> tuple[ParsedSourceCandidate, ...]:
    parser = RequireProcessingContextBoundaryPolicy(
        adapter
    ).context.microscope_handler.parser
    metadata_resolver = SourceCandidateMetadataResolver(adapter=adapter, parser=parser)
    candidates: list[ParsedSourceCandidate] = []
    for file_path in file_paths:
        for path_resolution in universe.path_projection(file_path).paths():
            metadata = metadata_resolver.metadata(path_resolution)
            candidates.append(
                ParsedSourceCandidate(
                    path=path_resolution.path,
                    resolved_path=path_resolution.resolved_path,
                    virtual_path=path_resolution.virtual_path,
                    filename=Path(path_resolution.resolved_path).name,
                    metadata=MappingProxyType(dict(metadata)),
                )
            )
    return ParsedSourceCandidateCollection(tuple(candidates)).deduplicated()

@dataclass(frozen=True, slots=True)
class SourceCandidatePathProjection:
    """Project physical source paths into candidate identity paths.

    Synthetic well expansion can map many virtual OpenHCS filenames onto the
    same immutable source image. Candidate identity must stay virtual so source
    binding inheritance sees the correct well/site/channel metadata.
    """

    source_path: str
    adapter: CellProfilerRuntimeAdapter
    include_virtual_paths: bool = True

    def paths(self) -> tuple["SourceCandidatePathResolution", ...]:
        context = self.adapter.source_binding_context
        resolved_path = str(_resolved_source_path(self.source_path, self.adapter))
        explicit_virtual_paths = self.explicit_virtual_paths()
        if explicit_virtual_paths:
            return tuple(
                SourceCandidatePathResolution(
                    path=virtual_path,
                    resolved_path=str(context.step_input_source_paths[virtual_path]),
                    virtual_path=virtual_path,
                )
                for virtual_path in explicit_virtual_paths
            )

        virtual_paths = self.virtual_paths_for_resolved_path(resolved_path)
        if virtual_paths:
            return tuple(
                SourceCandidatePathResolution(
                    path=virtual_path,
                    resolved_path=resolved_path,
                    virtual_path=virtual_path,
                )
                for virtual_path in virtual_paths
            )
        return (
            SourceCandidatePathResolution(
                path=str(self.source_path),
                resolved_path=resolved_path,
                virtual_path=None,
            ),
        )

    def explicit_virtual_paths(self) -> tuple[str, ...]:
        if not self.include_virtual_paths:
            return ()
        context = self.adapter.source_binding_context
        return tuple(
            key
            for key in SourceRuntimePathLookup(
                self.source_path,
                context.step_input_dir,
            ).keys()
            if key in context.step_input_source_paths
        )

    def virtual_paths_for_resolved_path(self, resolved_path: str) -> tuple[str, ...]:
        if not self.include_virtual_paths:
            return ()
        context = self.adapter.source_binding_context
        return context.virtual_source_paths_by_identity.get(
            source_path_identity_key(resolved_path),
            (),
        )

@dataclass(frozen=True, slots=True)
class SourceCandidateMetadataRequest:
    """Runtime dependencies for source-candidate metadata resolution."""

    adapter: CellProfilerRuntimeAdapter
    parser: CellProfilerFilenameParser

@dataclass(frozen=True, slots=True)
class SourceCandidateMetadataResolver(SourceCandidateMetadataRequest):
    """Resolve metadata precedence for parsed source-binding candidates."""

    def metadata(
        self,
        path_resolution: SourceCandidatePathResolution,
    ) -> MutableParsedSourceMetadata:
        metadata: MutableParsedSourceMetadata = {}
        context = self.adapter.source_binding_context
        context_paths = path_resolution.metadata_paths(context)
        if ContextSourceMetadataAuthority.has_metadata_for_any(context_paths, context):
            return self.metadata_from_context(
                metadata,
                path_resolution=path_resolution,
                context=context,
                context_paths=context_paths,
            )
        return self.metadata_from_paths(
            metadata,
            path_resolution=path_resolution,
            context=context,
            context_paths=context_paths,
        )

    def metadata_from_context(
        self,
        metadata: MutableParsedSourceMetadata,
        *,
        path_resolution: SourceCandidatePathResolution,
        context: SourceBindingRuntimeContext,
        context_paths: tuple[str, ...],
    ) -> MutableParsedSourceMetadata:
        ContextSourceMetadataAuthority.merge_into(metadata, context_paths, context)
        self.merge_rule_metadata(metadata, path_resolution.resolved_path)
        if not source_paths_equal(path_resolution.path, path_resolution.resolved_path):
            self.merge_rule_metadata(metadata, path_resolution.path)
        if (
            path_resolution.virtual_path is not None
            and path_resolution.virtual_path not in {path_resolution.path, path_resolution.resolved_path}
        ):
            self.merge_rule_metadata(metadata, path_resolution.virtual_path)
        return metadata

    def metadata_from_paths(
        self,
        metadata: MutableParsedSourceMetadata,
        *,
        path_resolution: SourceCandidatePathResolution,
        context: SourceBindingRuntimeContext,
        context_paths: tuple[str, ...],
    ) -> MutableParsedSourceMetadata:
        self.merge_path_metadata(metadata, path_resolution.resolved_path, strict=True)
        if not source_paths_equal(path_resolution.path, path_resolution.resolved_path):
            self.merge_path_metadata(
                metadata,
                path_resolution.path,
                strict=SourceRuntimePathLookup(
                    path_resolution.path,
                    context.step_input_dir,
                ).first_value(context.step_input_source_paths) is None,
            )
        if (
            path_resolution.virtual_path is not None
            and path_resolution.virtual_path not in {path_resolution.path, path_resolution.resolved_path}
        ):
            self.merge_path_metadata(metadata, path_resolution.virtual_path, strict=False)
        ContextSourceMetadataAuthority.merge_into(metadata, context_paths, context)
        return metadata

    def merge_path_metadata(
        self,
        metadata: MutableParsedSourceMetadata,
        metadata_path: str,
        *,
        strict: bool,
    ) -> None:
        _merge_candidate_path_metadata(
            metadata,
            metadata_path,
                self.adapter,
                self.parser,
            strict=strict,
        )

    def merge_rule_metadata(
        self,
        metadata: MutableParsedSourceMetadata,
        metadata_path: str,
    ) -> None:
        _merge_missing_source_metadata(
            metadata,
            metadata_from_rules(
                metadata_path,
                self.adapter.source_binding_plan.metadata_rules,
            ),
        )

class ContextSourceMetadataAuthority:
    """Lookup authority for runtime source metadata attached to source paths."""

    @classmethod
    def apply_loading_semantics(
        cls,
        payload: ImagePayloadValue,
        *,
        source_path: str,
        storage_path: str,
        request: PipelineStartSourceLoadRequest,
    ) -> ImagePayloadValue:
        source_metadata = SourceRuntimePathLookup(
            source_path,
            request.adapter.source_binding_context.step_input_dir,
        ).first_value(
            request.adapter.source_binding_context.source_metadata_by_path,
            include_native_path_fallback=True,
        )
        return SourceImagePayloadSemantics.from_source_metadata(
            source_metadata,
            storage_path,
            request.backend,
            RequireProcessingContextBoundaryPolicy(request.adapter).context.filemanager,
        ).apply(payload)

    @classmethod
    def has_metadata_for_any(
        cls,
        paths: tuple[str, ...],
        context: SourceBindingRuntimeContext,
    ) -> bool:
        return any(
            SourceRuntimePathLookup(path, context.step_input_dir).first_value(
                context.source_metadata_by_path,
                include_native_path_fallback=True,
            )
            is not None
            for path in paths
        )

    @classmethod
    def merge_into(
        cls,
        metadata: MutableParsedSourceMetadata,
        paths: tuple[str, ...],
        context: SourceBindingRuntimeContext,
    ) -> None:
        for path in dict.fromkeys(paths):
            context_metadata = SourceRuntimePathLookup(
                path,
                context.step_input_dir,
            ).first_value(
                context.source_metadata_by_path,
                include_native_path_fallback=True,
            )
            if context_metadata is not None:
                merge_source_metadata(metadata, context_metadata, path=path)

def _merge_candidate_path_metadata(
    metadata: MutableParsedSourceMetadata,
    metadata_path: str,
    adapter: CellProfilerRuntimeAdapter,
    parser: CellProfilerFilenameParser,
    *,
    strict: bool,
) -> None:
    parsed_metadata, extracted_metadata = _candidate_path_metadata_components(
        metadata_path,
        adapter,
        parser,
    )
    if strict:
        merge_source_metadata(metadata, parsed_metadata, path=metadata_path)
        merge_source_metadata(metadata, extracted_metadata, path=metadata_path)
        return
    _merge_missing_source_metadata(metadata, parsed_metadata)
    _merge_missing_source_metadata(metadata, extracted_metadata)

def _candidate_path_metadata_components(
    metadata_path: str,
    adapter: CellProfilerRuntimeAdapter,
    parser: CellProfilerFilenameParser,
) -> tuple[ParsedSourceMetadata, ParsedSourceMetadata]:
    cache_key = _candidate_path_metadata_cache_key(metadata_path, adapter, parser)
    cache = SOURCE_CANDIDATE_METADATA_PROCESS_CACHE
    cached = cache.cached_value(cache_key)
    if cached is not None:
        return cached
    parsed_metadata = parser.parse_filename(Path(metadata_path).name)
    if parsed_metadata is None:
        parsed_metadata = {}
    extracted_metadata = metadata_from_rules(
        metadata_path,
        adapter.source_binding_plan.metadata_rules,
    )
    result = (
        MappingProxyType(dict(parsed_metadata)),
        MappingProxyType(dict(extracted_metadata)),
    )
    return cache.store_value(cache_key, result)

def _candidate_path_metadata_cache_key(
    metadata_path: str,
    adapter: CellProfilerRuntimeAdapter,
    parser: CellProfilerFilenameParser,
) -> tuple[Hashable, ...]:
    return (
        source_path_identity_key(metadata_path),
        adapter.source_binding_plan.metadata_rules,
        parser.semantic_identity(),
    )

def cellprofiler_source_order_identity(
    adapter: CellProfilerRuntimeAdapter,
) -> tuple[Hashable, ...]:
    """Return source-order identity fields shared by CP image-number caches."""
    return adapter.source_binding_context.source_order_identity

def _merge_missing_source_metadata(
    metadata: MutableParsedSourceMetadata,
    additions: ParsedSourceMetadata,
) -> None:
    for key, value in additions.items():
        if value is None:
            continue
        if key not in metadata:
            metadata[key] = str(value)

class SourceCandidateMatcher:
    """Nominal owner for source-binding candidate selection semantics."""

    @classmethod
    def axis_scoped_candidates(
        cls,
        candidates: tuple[ParsedSourceCandidate, ...],
        axis_scope: SourceAxisMetadataScope,
        *,
        constraining_candidates: tuple[ParsedSourceCandidate, ...] | None = None,
        metadata_for_candidate: Callable[
            [ParsedSourceCandidate],
            ParsedSourceMetadata,
        ] = attrgetter("metadata"),
    ) -> tuple[ParsedSourceCandidate, ...]:
        """Return candidates visible inside the current runtime source axis."""
        scope_basis = (
            candidates
            if constraining_candidates is None
            else constraining_candidates
        )
        if not cls.axis_scope_is_constraining(
            scope_basis,
            axis_scope,
            metadata_for_candidate=metadata_for_candidate,
        ):
            return candidates
        return tuple(
            candidate
            for candidate in candidates
            if axis_scope.matches_metadata(metadata_for_candidate(candidate))
        )

    @staticmethod
    def axis_scope_is_constraining(
        candidates: tuple[ParsedSourceCandidate, ...],
        axis_scope: SourceAxisMetadataScope,
        *,
        metadata_for_candidate: Callable[
            [ParsedSourceCandidate],
            ParsedSourceMetadata,
        ] = attrgetter("metadata"),
    ) -> bool:
        if axis_scope.has_component:
            return True
        return any(
            axis_scope.matches_metadata(metadata_for_candidate(candidate))
            for candidate in candidates
        )

    @classmethod
    def match_candidates(
        cls,
        *,
        candidates: tuple[ParsedSourceCandidate, ...],
        binding: NamedSourceBinding,
        inherit_components: Mapping[str, str],
    ) -> tuple[ParsedSourceCandidate, ...]:
        selector_resolution = SourceCandidateSelectorResolution.from_binding(
            binding,
            inherit_components,
        )
        cache_key = cls.match_candidates_cache_key(
            candidates=candidates,
            binding=binding,
            selector_resolution=selector_resolution,
        )
        cache = SOURCE_CANDIDATE_MATCH_PROCESS_CACHE
        cached = cache.cached_value(cache_key)
        if cached is not None:
            return cached
        cls.validate_metadata_selectors(candidates, binding)

        matched = tuple(
            candidate
            for candidate in candidates
            if cls.matches_explicit_components(
                candidate,
                selector_resolution.component_selectors,
            )
            and cls.matches_inherited_scope(
                candidate,
                selector_resolution.inherited_component_items,
            )
            and cls.matches_metadata(candidate, binding.selector.metadata)
            and cls.matches_source_filters(candidate, binding.selector.filters)
        )
        return cache.store_value(cache_key, matched)

    @classmethod
    def matches_source_filters(
        cls,
        candidate: ParsedSourceCandidate,
        filters: tuple[SourceFilterClause, ...],
    ) -> bool:
        return any(
            source_filters_match(path, filters)
            for path in cls.source_filter_paths(candidate)
        )

    @staticmethod
    def source_filter_paths(candidate: ParsedSourceCandidate) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *candidate.paths,
                    *SourceMetadataRoleView(candidate.metadata).source_filter_paths(),
                )
            )
        )

    @staticmethod
    def match_candidates_cache_key(
        *,
        candidates: tuple[ParsedSourceCandidate, ...],
        binding: NamedSourceBinding,
        selector_resolution: SourceCandidateSelectorResolution,
    ) -> tuple[Hashable, ...]:
        """Return the semantic identity for one selector/candidate match."""
        return (
            "source_candidate_match",
            tuple(candidate.candidate_identity for candidate in candidates),
            binding.selector,
            selector_resolution.inherited_component_items,
        )

    @staticmethod
    def validate_metadata_selectors(
        candidates: tuple[ParsedSourceCandidate, ...],
        binding: NamedSourceBinding,
    ) -> None:
        unsupported = SourceCandidateMatcher.unsupported_metadata_selector_fields(
            candidates,
            binding,
        )
        if unsupported:
            raise NotImplementedError(
                "Source-binding metadata selectors are only supported when the "
                "native OpenHCS filename parser exposes those fields. Missing "
                f"fields: {list(unsupported)}."
            )

    @staticmethod
    def supports_metadata_selectors(
        candidates: tuple[ParsedSourceCandidate, ...],
        binding: NamedSourceBinding,
    ) -> bool:
        return not SourceCandidateMatcher.unsupported_metadata_selector_fields(
            candidates,
            binding,
        )

    @staticmethod
    def unsupported_metadata_selector_fields(
        candidates: tuple[ParsedSourceCandidate, ...],
        binding: NamedSourceBinding,
    ) -> tuple[str, ...]:
        metadata_fields = {selector.field for selector in binding.selector.metadata}
        if not metadata_fields:
            return ()
        return tuple(
            field
            for field in sorted(metadata_fields)
            if not any(
                source_metadata_value(candidate.metadata, field) is not None
                for candidate in candidates
            )
        )

    @classmethod
    def matches_explicit_components(
        cls,
        candidate: ParsedSourceCandidate,
        expected_components: Mapping[str, str],
    ) -> bool:
        return all(
            cls.matches_explicit_component(candidate, component_name, value)
            for component_name, value in expected_components.items()
        )

    @staticmethod
    def matches_explicit_component(
        candidate: ParsedSourceCandidate,
        component_name: str,
        expected_value: str,
    ) -> bool:
        component = source_metadata_component(component_name)
        if component is None:
            metadata_value = source_metadata_value(candidate.metadata, component_name)
            return metadata_value is not None and source_metadata_values_equal(
                metadata_value,
                expected_value,
            )
        return any(
            source_metadata_values_equal(metadata_value, expected_value)
            for metadata_value in source_component_metadata_values(
                candidate.metadata,
                component,
            )
        )

    @staticmethod
    def matches_inherited_scope(
        candidate: ParsedSourceCandidate,
        inherited_scope: tuple[tuple[str, str], ...],
    ) -> bool:
        return all(
            (
                metadata_value := semantic_source_metadata_value(
                    candidate.metadata,
                    field_name,
                )
            )
            is None
            or source_metadata_values_equal(metadata_value, value)
            for field_name, value in inherited_scope
        )

    @staticmethod
    def matches_metadata(
        candidate: ParsedSourceCandidate,
        metadata_selectors: tuple[MetadataSelector, ...],
    ) -> bool:
        return all(
            (metadata_value := source_metadata_value(candidate.metadata, selector.field))
            is not None
            and source_metadata_values_equal(metadata_value, selector.value)
            for selector in metadata_selectors
        )

    @staticmethod
    def matches_image_set_metadata(
        candidate: ParsedSourceCandidate,
        image_set_metadata: Mapping[str, SourceMetadataMatchConstraint],
    ) -> bool:
        return all(
            constraint.matches(candidate)
            for constraint in image_set_metadata.values()
        )

    @staticmethod
    def ordered_source_candidates(
        candidates: tuple[ParsedSourceCandidate, ...],
    ) -> tuple[ParsedSourceCandidate, ...]:
        cache_key = (
            "source_candidates_ordered",
            tuple(candidate.candidate_identity for candidate in candidates),
        )
        cache = SOURCE_CANDIDATE_MATCH_PROCESS_CACHE
        cached = cache.cached_value(cache_key)
        if cached is not None:
            return cached
        return cache.store_value(
            cache_key,
            tuple(sorted(candidates, key=lambda candidate: candidate.resolved_path)),
        )

    @classmethod
    def inherited_scope_components(
        cls,
        candidates: tuple[ParsedSourceCandidate, ...],
    ) -> Mapping[str, str]:
        if not candidates:
            return {}
        shared: dict[str, str] = {}
        first_metadata = candidates[0].metadata
        for field_name, value in first_metadata.items():
            if value is None:
                continue
            normalized_value = str(value)
            if all(
                (
                    candidate_value := semantic_source_metadata_value(
                        candidate.metadata,
                        field_name,
                    )
                )
                is not None
                and source_metadata_values_equal(candidate_value, normalized_value)
                for candidate in candidates[1:]
            ):
                shared[field_name] = normalized_value
        return MappingProxyType(shared)

    @classmethod
    def pipeline_start_inherited_components(
        cls,
        source_binding_plan: CompiledSourceBindingPlan,
        step_input_candidates: tuple[ParsedSourceCandidate, ...],
        *,
        binding: NamedSourceBinding,
        target_candidates: tuple[ParsedSourceCandidate, ...],
    ) -> Mapping[str, str]:
        if source_binding_plan.match_plan is not None:
            return MappingProxyType({})
        current_scope = cls.inherited_scope_components(step_input_candidates)
        if not current_scope:
            return current_scope
        selector_candidates = cls.match_candidates(
            candidates=target_candidates,
            binding=binding,
            inherit_components={},
        )
        if not selector_candidates:
            return current_scope
        return MappingProxyType(
            dict(
                cls.current_scope_items_present_in_target(
                    current_scope,
                    selector_candidates,
                )
            )
        )

    @classmethod
    def match_plan_candidates(
        cls,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        matchers = {
            SourceBindingMatchMethod.METADATA: cls.match_metadata_image_set_candidates,
            SourceBindingMatchMethod.ORDER: cls.match_order_image_set_candidates,
        }
        plan = request.binding_plan.plan
        if plan is None:
            return request.universe.target_candidates
        return matchers[plan.method](request)

    @classmethod
    def match_metadata_image_set_candidates(
        cls,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        constraints: dict[str, SourceMetadataMatchConstraint] = {}
        plan = request.binding_plan.plan
        if plan is None:
            return request.universe.target_candidates
        for dimension in plan.dimensions:
            target_field = dimension.field_for_alias(request.alias)
            if target_field is None:
                continue
            match_values = cls.dimension_match_values(
                dimension=dimension,
                request=request,
            )
            if not match_values:
                continue
            constraint = SourceMetadataMatchConstraint(
                target_field,
                match_values,
            )
            existing = constraints.get(target_field)
            if existing is not None:
                constraint = existing.intersect(
                    constraint.values,
                )
            constraints[target_field] = constraint
        metadata_constraints = MappingProxyType(constraints)
        return tuple(
            candidate
            for candidate in request.universe.target_candidates
            if cls.matches_image_set_metadata(
                candidate,
                metadata_constraints,
            )
        )

    @classmethod
    def match_order_image_set_candidates(
        cls,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        current_indexes = cls.order_match_indexes(request)
        if not current_indexes:
            scoped_candidates = cls.target_candidates_in_current_scope(request)
            return scoped_candidates or request.universe.target_candidates
        ordered_target_candidates = cls.ordered_source_candidates(
            request.universe.target_candidates
        )
        return tuple(
            ordered_target_candidates[index]
            for index in current_indexes
            if index < len(ordered_target_candidates)
        )

    @classmethod
    def target_candidates_in_current_scope(
        cls,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        current_scope = cls.inherited_scope_components(
            request.universe.step_input_candidates
        )
        if not current_scope:
            return ()
        current_scope_items = cls.current_scope_items_present_in_target(
            current_scope,
            request.universe.target_candidates,
        )
        if not current_scope_items:
            return ()
        return tuple(
            candidate
            for candidate in request.universe.target_candidates
            if cls.matches_inherited_scope(candidate, current_scope_items)
        )

    @classmethod
    def current_scope_items_present_in_target(
        cls,
        current_scope: Mapping[str, str],
        target_candidates: tuple[ParsedSourceCandidate, ...],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (field_name, value)
            for field_name, value in current_scope.items()
            if cls.target_candidates_contain_scope_value(
                target_candidates,
                field_name,
                value,
            )
        )

    @staticmethod
    def target_candidates_contain_scope_value(
        target_candidates: tuple[ParsedSourceCandidate, ...],
        field_name: str,
        value: str,
    ) -> bool:
        return any(
            (
                metadata_value := semantic_source_metadata_value(
                    candidate.metadata,
                    field_name,
                )
            )
            is not None
            and source_metadata_values_equal(metadata_value, value)
            for candidate in target_candidates
        )

    @classmethod
    def order_match_indexes(
        cls,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[int, ...]:
        indexes = {
            index
            for candidate in request.universe.step_input_candidates
            for index in (
                cls.source_alias_order_index(
                    SourceAliasOrderIndexRequest(
                        alias=request.alias,
                        binding_plan=request.binding_plan,
                        universe=request.universe,
                        candidate=candidate,
                    )
                ),
            )
            if index is not None
        }
        return tuple(sorted(indexes))

    @classmethod
    def source_alias_order_index(
        cls,
        request: SourceAliasOrderIndexRequest,
    ) -> int | None:
        matched_indexes: set[int] = set()
        for binding in request.binding_plan.bindings:
            if binding.alias == request.alias:
                continue
            for index, ordered_candidate in enumerate(
                cls.ordered_binding_candidates(
                    binding=binding,
                    candidates=request.universe.pipeline_candidates,
                )
            ):
                if ordered_candidate.path_identity.intersects(
                    request.candidate.path_identity
                ):
                    matched_indexes.add(index)
                    break
        if not matched_indexes:
            return None
        if len(matched_indexes) != 1:
            raise RuntimeError(
                f"Order-based image-set matching could not uniquely assign source file "
                f"{request.candidate.resolved_path!r} to one alias order index."
            )
        return next(iter(matched_indexes))

    @classmethod
    def ordered_binding_candidates(
        cls,
        *,
        binding: NamedSourceBinding,
        candidates: tuple[ParsedSourceCandidate, ...],
    ) -> tuple[ParsedSourceCandidate, ...]:
        return cls.ordered_source_candidates(
            cls.match_candidates(
                candidates=candidates,
                binding=binding,
                inherit_components={},
            )
        )

    @classmethod
    def dimension_match_values(
        cls,
        *,
        dimension: SourceBindingMatchDimension,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[str, ...]:
        target_alias = request.alias
        candidate_values = tuple(
            value
            for field in dimension.fields
            if field.alias != target_alias
            for value in cls.source_match_field_values(field, request)
        )
        return tuple(dict.fromkeys(candidate_values))

    @classmethod
    def source_match_field_values(
        cls,
        field: SourceBindingMatchField,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[str, ...]:
        binding = request.binding_plan.binding_for_alias(field.alias)
        if binding is None:
            return cls.shared_candidate_values(
                field,
                request.universe.step_input_candidates,
            )
        alias_candidates = cls.alias_scoped_step_input_candidates(field.alias, request)
        if alias_candidates:
            return cls.shared_candidate_values(field, alias_candidates)
        return cls.shared_candidate_values(
            field,
            request.universe.step_input_candidates,
        )

    @classmethod
    def alias_scoped_step_input_candidates(
        cls,
        alias: str,
        request: SourceBindingMatchPlanRequest,
    ) -> tuple[ParsedSourceCandidate, ...]:
        binding = request.binding_plan.binding_for_alias(alias)
        if binding is None:
            return ()
        matched = cls.match_candidates(
            candidates=request.universe.step_input_candidates,
            binding=binding,
            inherit_components={},
        )
        if matched == request.universe.step_input_candidates:
            return ()
        return matched

    @staticmethod
    def shared_candidate_values(
        field: SourceBindingMatchField,
        step_input_candidates: tuple[ParsedSourceCandidate, ...],
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                metadata_value
                for candidate in step_input_candidates
                for metadata_value in (
                    semantic_source_metadata_value(candidate.metadata, field.metadata_field),
                )
                if metadata_value is not None
            )
        )

def source_candidate_summary(
    candidates: tuple[ParsedSourceCandidate, ...],
) -> tuple[Mapping[str, ParsedSourceMetadata | str], ...]:
    return tuple(
        {
            "path": candidate.path,
            "resolved_path": candidate.resolved_path,
            "source_filter_paths": SourceCandidateMatcher.source_filter_paths(candidate),
            "metadata": dict(candidate.metadata),
        }
        for candidate in candidates[:5]
    )

def _resolved_source_path(
    file_path: str,
    adapter: CellProfilerRuntimeAdapter,
) -> str:
    source_path = SourceRuntimePathLookup(
        file_path,
        adapter.source_binding_context.step_input_dir,
    ).first_value(adapter.source_binding_context.step_input_source_paths)
    if source_path is not None:
        return source_path
    path = Path(file_path)
    if path.is_absolute():
        return str(path)
    step_input_dir = adapter.source_binding_context.step_input_dir
    if step_input_dir is None:
        return str(path)
    return str(Path(step_input_dir) / path)
