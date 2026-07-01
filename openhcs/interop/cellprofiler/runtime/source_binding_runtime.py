"""Source-binding runtime resolution for the CellProfiler adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from operator import attrgetter
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any, ClassVar, cast

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.constants.constants import Backend, FileFormat
from openhcs.core.aligned_image_payload import payload_slices_for_alignment
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import detect_memory_type
from openhcs.core.pipeline_image_schema import SOURCE_IMAGE_TYPE_METADATA_FIELD
from openhcs.core.process_local_cache import (
    IdentityBoundProcessCache,
    RegisteredProcessLocalBoundedCache,
)
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceBindingRuntimeContext,
    SourceRuntimePathLookup,
)
from openhcs.core.source_image_semantics import SourceImagePayloadSemantics
from openhcs.core.source_matching import (
    SourceImageSetIdentity,
    SourceImageSetIdentityPolicy,
    SourceAxisMetadataScope,
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
from openhcs.core.source_schema_workspace import source_schema_auxiliary_payload
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ImagePayloadMetadataInput,
    ImagePayloadMetadataCompositionRequest,
    ImagePayloadSourceMetadataContext,
    RuntimeArrayPayload,
    RuntimeImagePayloadContext,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
    SourceImageIdentity,
)
from openhcs.core.runtime_semantics import bounded_runtime_plane_index
from openhcs.interop.cellprofiler.runtime.adapter_profile import (
    AdapterProfileLog,
    SourceCandidateProfileEvent,
)
from openhcs.interop.cellprofiler.runtime.adapter_protocols import (
    CellProfilerFileManager,
    CellProfilerFilenameParser,
    CellProfilerProcessingContext,
    RequireProcessingContextBoundaryPolicy,
)
from openhcs.interop.cellprofiler.runtime.current_image_context import (
    CellProfilerRequiredCurrentImageContext,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    ImagePayloadMaskValue,
    ImagePayloadValue,
)
from openhcs.interop.cellprofiler.runtime.source_candidates import (
    CellProfilerImageNumberResolver,
    ContextSourceMetadataAuthority,
    CurrentStepInputSourceBindingPlaneResolution,
    MatchedSourceCandidatesRequest,
    ParsedSourceCandidate,
    PipelineStartSourceLoadRequest,
    PipelineStartSourcePayloadRequest,
    RuntimeSourceProvenanceSourceBindingPlaneResolution,
    SourceBindingAxisAliasResolution,
    SourceBindingImageSetMatchScope,
    SourceBindingMatchCandidateUniverse,
    SourceBindingPlaneCandidateContext,
    SourceBindingPlaneMatch,
    SourceBindingRequestBase,
    SourceCandidateMatcher,
    source_candidate_summary,
)
from openhcs.interop.cellprofiler.runtime.source_identity import (
    CellProfilerCurrentImage,
    MutableParsedSourceMetadata,
    ParsedSourceCandidateABC,
    ParsedSourceCandidatePathIdentity,
    ParsedSourceMetadata,
    SourceBindingRuntimePathIdentities,
)
from openhcs.interop.cellprofiler.runtime.runtime_value_authorities import (
    MatlabPayloadEntryName,
)

PipelineStartPayloadCacheValue = tuple[ImagePayloadValue, ...]
StepInputPayloadCacheValue = tuple[ImagePayloadValue, ...]
SourcePayloadCandidateCacheIdentity = tuple[Hashable, ...]


@dataclass(frozen=True, slots=True)
class PipelineStartSourcePayloadCacheKey:
    """Source-candidate identity for pipeline-start payload loads."""

    backend: str
    filemanager: CellProfilerFileManager
    selected_sources: tuple[SourcePayloadCandidateCacheIdentity, ...]

    @classmethod
    def from_sources(
        cls,
        *,
        backend: str,
        filemanager: CellProfilerFileManager,
        context: SourceBindingRuntimeContext,
        selected_sources: tuple[ParsedSourceCandidate, ...],
    ) -> "PipelineStartSourcePayloadCacheKey":
        return cls(
            backend=backend,
            filemanager=filemanager,
            selected_sources=tuple(
                candidate.cache_identity(context) for candidate in selected_sources
            ),
        )


@dataclass(frozen=True, slots=True)
class StepInputSourcePayloadCacheKey:
    """Source-metadata-aware identity for external step-input payload loads."""

    storage_backend: str
    source_backend: str
    filemanager: CellProfilerFileManager
    selected_sources: tuple[SourcePayloadCandidateCacheIdentity, ...]

    @classmethod
    def from_sources(
        cls,
        *,
        storage_backend: str,
        source_backend: str,
        filemanager: CellProfilerFileManager,
        context: SourceBindingRuntimeContext | None = None,
        selected_sources: tuple[ParsedSourceCandidate, ...],
    ) -> "StepInputSourcePayloadCacheKey":
        resolved_context = context or SourceBindingRuntimeContext.empty()
        return cls(
            storage_backend=storage_backend,
            source_backend=source_backend,
            filemanager=filemanager,
            selected_sources=tuple(
                candidate.cache_identity(resolved_context)
                for candidate in selected_sources
            ),
        )


class CurrentStepPayloadSelectionMode(str, Enum):
    """Origin-specific current-step payload reuse semantics."""

    PRESERVE_CURRENT = "preserve_current"
    STEP_INPUT_NATURAL = "step_input_natural"

@dataclass(frozen=True, slots=True)
class SourceBindingAxisCardinality:
    """Closed cardinality authority for source-binding axis sizes."""

    axis_size: int | None

    @property
    def is_single_source_axis(self) -> bool:
        return self.axis_size in {1}

@dataclass(frozen=True, slots=True)
class SourceBindingMatchedPlaneResolution:
    """Resolve matched source planes without confusing runtime site stacks for source axes."""

    axis_id: str
    match: SourceBindingPlaneMatch
    source_binding_axis_size: int | None

    @classmethod
    def plane_index_for_adapter_alias(
        cls,
        adapter: CellProfilerRuntimeAdapter,
        alias: str,
    ) -> int | None:
        source_binding_axis_size = adapter.source_binding_axis_size((alias,))
        if SourceBindingAxisCardinality(
            source_binding_axis_size
        ).is_single_source_axis:
            return None
        candidate_context = SourceBindingPlaneCandidateContext.from_adapter(
            adapter,
            alias,
        )
        if candidate_context is not None:
            matched_context = candidate_context.match()
            if matched_context is not None:
                path_identities = adapter.source_binding_runtime_path_identities()
                current_step_index = CurrentStepInputSourceBindingPlaneResolution(
                    path_identities=path_identities,
                    match=matched_context,
                ).plane_index()
                if current_step_index is not None:
                    return current_step_index
                runtime_provenance_index = (
                    RuntimeSourceProvenanceSourceBindingPlaneResolution(
                        path_identities=path_identities,
                        match=matched_context,
                    ).plane_index()
                )
                if runtime_provenance_index is not None:
                    return runtime_provenance_index
                matched_index = cls(
                    axis_id=adapter.axis_scope.axis_id,
                    match=matched_context,
                    source_binding_axis_size=source_binding_axis_size,
                ).value
                if matched_index is not None:
                    return matched_index
            if candidate_context.universe.axis_scope.has_component:
                return None
        for index, binding in enumerate(adapter.source_binding_plan.bindings):
            if binding.alias == alias:
                return index
        return None

    @property
    def value(self) -> int | None:
        single_index = self.match.index_set.single_index
        if single_index is not None:
            return single_index
        if self.match.index_set.covers_candidate_set(
            self.match.universe.plane_candidates
        ):
            return None
        if SourceBindingAxisCardinality(
            self.source_binding_axis_size
        ).is_single_source_axis:
            return None
        raise RuntimeError(
            f"Source binding alias {self.match.alias!r} matched multiple source planes "
            f"for axis {self.axis_id!r}: {self.match.index_set.indexes!r}."
        )

@dataclass(frozen=True, slots=True)
class SourceBindingAxisPlaneResolution:
    """Single-plane resolution for source-binding axis projection requests."""

    source_aliases: tuple[str, ...]
    indexes: tuple[int, ...]

    @classmethod
    def for_adapter_aliases(
        cls,
        adapter: CellProfilerRuntimeAdapter,
        source_aliases: tuple[str, ...],
    ) -> "SourceBindingAxisPlaneResolution":
        if not source_aliases:
            return cls(source_aliases=(), indexes=())
        resolved_aliases = SourceBindingAxisAliasResolution(
            requested_aliases=source_aliases,
            bindings=adapter.source_binding_plan.bindings,
        ).aliases()
        return cls(
            source_aliases=resolved_aliases,
            indexes=tuple(
                index
                for alias in resolved_aliases
                for index in (
                    SourceBindingMatchedPlaneResolution.plane_index_for_adapter_alias(
                        adapter,
                        alias,
                    ),
                )
                if index is not None
            ),
        )

    def plane_index(self) -> int | None:
        unique_indexes = tuple(dict.fromkeys(self.indexes))
        if not unique_indexes:
            return None
        if len(unique_indexes) == 1:
            return unique_indexes[0]
        if self.is_composed_axis_request():
            return None
        raise RuntimeError(
            "Source-binding plane resolution produced conflicting indexes: "
            f"{unique_indexes} for aliases {self.source_aliases!r}."
        )

    def is_composed_axis_request(self) -> bool:
        """Return whether aliases name the full composed source-binding axis."""
        return len(tuple(dict.fromkeys(self.source_aliases))) > 1

@dataclass(slots=True)
class SourceBindingAxisResolutionMemo:
    """Per-adapter memo for source-binding axis resolution."""

    axis_sizes: dict[tuple[str, ...], int | None] = field(default_factory=dict)
    plane_resolutions: dict[
        tuple[str, ...],
        SourceBindingAxisPlaneResolution,
    ] = field(default_factory=dict)


class SourceBindingAxisResolutionProcessCache(IdentityBoundProcessCache):
    """Process-local memo owner for source-binding axis projections."""

    registry_key = "source_binding_axis_resolution"


class SourceBindingAxisResolutionAuthority:
    """Central source-binding axis size and plane-resolution authority."""

    @classmethod
    def plane_index(
        cls,
        adapter: CellProfilerRuntimeAdapter,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        return cls.plane_resolution(adapter, source_aliases).plane_index()

    @classmethod
    def plane_resolution(
        cls,
        adapter: CellProfilerRuntimeAdapter,
        source_aliases: tuple[str, ...],
    ) -> SourceBindingAxisPlaneResolution:
        key = tuple(source_aliases)
        memo = cls._memo(adapter)
        cached = memo.plane_resolutions.get(key)
        if cached is not None:
            return cached
        resolution = SourceBindingAxisPlaneResolution.for_adapter_aliases(
            adapter,
            key,
        )
        memo.plane_resolutions[key] = resolution
        return resolution

    @classmethod
    def active_axis_aliases(
        cls,
        adapter: CellProfilerRuntimeAdapter,
    ) -> tuple[str, ...]:
        """Return source aliases that declare the adapter's active binding axis."""
        return tuple(
            binding.alias
            for binding in adapter.source_binding_plan.bindings
        )

    @classmethod
    def active_axis_plane_resolution(
        cls,
        adapter: CellProfilerRuntimeAdapter,
    ) -> SourceBindingAxisPlaneResolution:
        """Resolve the adapter's declared source-binding axis."""
        return cls.plane_resolution(adapter, cls.active_axis_aliases(adapter))

    @classmethod
    def axis_size(
        cls,
        adapter: CellProfilerRuntimeAdapter,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        key = tuple(source_aliases)
        memo = cls._memo(adapter)
        if key in memo.axis_sizes:
            return memo.axis_sizes[key]

        bindings = adapter.source_binding_plan.bindings
        if not bindings:
            memo.axis_sizes[key] = None
            return None

        resolved_aliases = SourceBindingAxisAliasResolution(
            requested_aliases=key,
            bindings=bindings,
        ).aliases()
        candidate_counts = tuple(
            len(candidate_context.universe.plane_candidates.candidates)
            for alias in resolved_aliases
            for candidate_context in (
                SourceBindingPlaneCandidateContext.from_adapter(adapter, alias),
            )
            if candidate_context is not None
            and candidate_context.universe.plane_candidates.candidates
        )
        axis_size = max(candidate_counts) if candidate_counts else len(bindings)
        memo.axis_sizes[key] = axis_size
        return axis_size

    @classmethod
    def _memo(
        cls,
        adapter: CellProfilerRuntimeAdapter,
    ) -> SourceBindingAxisResolutionMemo:
        cache = cast(
            SourceBindingAxisResolutionProcessCache,
            SourceBindingAxisResolutionProcessCache.process_cache(),
        )
        memo = cache.get_bound(adapter)
        if memo is not None:
            return cast(SourceBindingAxisResolutionMemo, memo)
        return cast(
            SourceBindingAxisResolutionMemo,
            cache.put_bound(adapter, SourceBindingAxisResolutionMemo()),
        )


@dataclass(frozen=True, slots=True)
class SourceBindingPayloadPlaneResolution:
    """Resolve a payload plane through source-binding alias and provenance metadata."""

    adapter: CellProfilerRuntimeAdapter
    payload: ImagePayloadMetadataInput
    plane_count: int

    def plane_index(self) -> int | None:
        alias_plane_index = self.source_binding_alias_plane_index()
        if alias_plane_index is not None:
            return alias_plane_index
        return self.grouped_component_plane_index()

    def source_binding_alias_plane_index(self) -> int | None:
        alias_set = SourceBindingPayloadAliasSet.from_payload(self.payload)
        if alias_set.is_empty:
            return None
        return bounded_runtime_plane_index(
            self.plane_count,
            SourceBindingAxisResolutionAuthority.plane_index(
                self.adapter,
                alias_set.aliases,
            ),
        )

    def grouped_component_plane_index(self) -> int | None:
        axis_scope = self.adapter.axis_scope
        if not axis_scope.has_value:
            return None
        axis_component = axis_scope.require_component_name()
        axis_component_value = axis_scope.require_value_text()
        values = self.component_values(axis_component)
        if len(values) != self.plane_count:
            return None
        present_values = tuple(value for value in values if value is not None)
        if len(set(present_values)) <= 1:
            return None
        matching_indexes = tuple(
            index
            for index, value in enumerate(values)
            if value is not None
            and source_metadata_values_equal(value, axis_component_value)
        )
        if len(matching_indexes) != 1:
            return None
        return matching_indexes[0]

    def component_values(
        self,
        axis_component: str,
    ) -> tuple[str | None, ...]:
        metadata = image_payload_metadata(self.payload)
        component_metadata = metadata.source_image_provenance_planes.component_metadata
        if component_metadata:
            return SourceBindingPayloadComponentMetadata(
                component_metadata
            ).component_values(axis_component)
        return tuple(
            self.component_value_for_source_path(path, axis_component)
            for path in metadata.source_image_provenance_planes.paths
        )

    def component_value_for_source_path(
        self,
        source_path: str | None,
        axis_component: str,
    ) -> str | None:
        if source_path is None:
            return None
        return SourcePathMetadataLookup(
            adapter=self.adapter,
            source_path=source_path,
        ).component_value(axis_component)


@dataclass(frozen=True, slots=True)
class SourceBindingPayloadAliasSet:
    """Source-binding aliases declared by an image payload."""

    aliases: tuple[str, ...]

    @classmethod
    def from_payload(
        cls,
        payload: ImagePayloadMetadataInput,
    ) -> "SourceBindingPayloadAliasSet":
        return cls(
            tuple(
                alias
                for alias in image_payload_metadata(payload).source_image_names
                if alias
            )
        )

    @property
    def is_empty(self) -> bool:
        return not self.aliases


@dataclass(frozen=True, slots=True)
class SourceBindingPayloadComponentMetadata:
    """Per-plane component metadata carried by a source-bound image payload."""

    plane_component_metadata: tuple[SourceComponentMetadata | None, ...]

    def component_values(self, axis_component: str) -> tuple[str | None, ...]:
        return tuple(
            self.component_value(component_metadata, axis_component)
            for component_metadata in self.plane_component_metadata
        )

    @staticmethod
    def component_value(
        component_metadata: SourceComponentMetadata | None,
        axis_component: str,
    ) -> str | None:
        if component_metadata is None:
            return None
        return semantic_source_metadata_value(component_metadata, axis_component)


@dataclass(frozen=True, slots=True)
class SourcePathMetadataLookup:
    """Resolve source metadata for one path through the adapter source context."""

    adapter: CellProfilerRuntimeAdapter
    source_path: str

    def component_value(self, axis_component: str) -> str | None:
        metadata = self.metadata()
        if metadata is None:
            return None
        return semantic_source_metadata_value(metadata, axis_component)

    def metadata(self) -> SourceComponentMetadata | None:
        metadata_by_path = self.adapter.source_binding_context.source_metadata_by_path
        for candidate_path in self.candidate_paths():
            metadata = metadata_by_path.get(candidate_path)
            if metadata is not None:
                return metadata
        return None

    def candidate_paths(self) -> tuple[str, ...]:
        return (
            self.source_path,
            str(Path(self.source_path).resolve(strict=False)),
            self.adapter.cellprofiler_source_order_path(self.source_path),
        )


@dataclass(frozen=True, slots=True)
class SourceBindingResolutionRequest(
    SourceBindingRequestBase,
    CellProfilerRequiredCurrentImageContext,
):
    """Source-binding resolution inputs for one external image alias."""

    adapter: CellProfilerRuntimeAdapter

class SourceBindingResolver(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for resolving typed source bindings."""

    __registry_key__ = "origin_key"
    __skip_if_no_key__ = True
    origin: ClassVar[SourceBindingOrigin | None] = None
    origin_key: ClassVar[str | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def for_origin(cls, origin: SourceBindingOrigin) -> "SourceBindingResolver":
        return cls.__registry__[origin.value]()

    @abstractmethod
    def resolve_image(self, request: SourceBindingResolutionRequest) -> ImagePayloadValue:
        """Resolve one named source image binding."""

    def require_matched_candidates(
        self,
        request: MatchedSourceCandidatesRequest,
    ) -> tuple["ParsedSourceCandidate", ...]:
        """Return matched source candidates or raise with resolver context."""

        if request.matched:
            return request.matched
        candidate_summary = source_candidate_summary(request.candidates)
        raise RuntimeError(
            f"CellProfiler source alias '{request.alias}' with selector "
            f"{request.binding.selector!r} matched no files in the "
            f"{request.source_description} source universe. "
            f"Candidate sample: {candidate_summary!r}."
        )

class StepInputSourceBindingResolver(SourceBindingResolver):
    """Resolve named images directly from the current FunctionStep input."""

    origin = SourceBindingOrigin.STEP_INPUT
    origin_key = SourceBindingOrigin.STEP_INPUT.value

    def resolve_image(self, request: SourceBindingResolutionRequest) -> ImagePayloadValue:
        if not request.binding.requires_selector_resolution:
            return _natural_step_input_payload(request.current_image)
        step_input_files = request.adapter.source_binding_context.step_input_files
        if not step_input_files:
            raise NotImplementedError(
                f"CellProfiler source alias '{request.alias}' needs step-input "
                "selector resolution, but no step input file universe was "
                "provided to the runtime adapter."
            )
        parsed_candidates = request.adapter.source_candidates(step_input_files)
        current_candidates = request.adapter.source_candidates(
            request.adapter.source_binding_context.current_step_input_files
        )
        match_started_at = time.perf_counter()
        matched = SourceCandidateMatcher.match_candidates(
            candidates=parsed_candidates,
            binding=request.binding,
            inherit_components=SourceCandidateMatcher.inherited_scope_components(
                current_candidates
            ),
        )
        AdapterProfileLog.source_candidates(
            SourceCandidateProfileEvent(
                label="source_candidates_match",
                seconds=time.perf_counter() - match_started_at,
                alias=request.alias,
                source=SourceBindingOrigin.STEP_INPUT,
                count=len(matched),
            )
        )
        selected_files = self.require_matched_candidates(
            MatchedSourceCandidatesRequest.from_resolution(
                request,
                matched=matched,
                candidates=parsed_candidates,
                source_description="step input",
            )
        )
        selection = CurrentStepPayloadSelector.from_step_input_context(
            request.adapter.source_binding_context,
        ).resolve(
            selected_paths=tuple(source.path for source in selected_files),
            current_image=request.current_image,
        )
        if selection.is_matched:
            return selection.require_payload()
        return _load_step_input_stack(
            request=request,
            selected_sources=selected_files,
        )

class PipelineStartSourceBindingResolver(SourceBindingResolver):
    """Resolve named images from the original pipeline-start source universe."""

    origin = SourceBindingOrigin.PIPELINE_START
    origin_key = SourceBindingOrigin.PIPELINE_START.value

    def resolve_image(self, request: SourceBindingResolutionRequest) -> ImagePayloadValue:
        pipeline_input_files = request.adapter.source_binding_context.pipeline_input_files
        if not pipeline_input_files:
            raise NotImplementedError(
                f"CellProfiler source alias '{request.alias}' needs pipeline-start "
                "selector resolution, but no pipeline-start file universe was "
                "provided to the runtime adapter."
            )
        step_input_candidates = request.adapter.source_candidates(
            request.adapter.source_binding_context.current_step_input_files
        )
        inherit_components = SourceCandidateMatcher.pipeline_start_inherited_components(
            request.adapter.source_binding_plan,
            step_input_candidates,
        )
        match_started_at = time.perf_counter()
        selected_files = self._current_step_matched_candidates(
            request,
            step_input_candidates=step_input_candidates,
            inherit_components=inherit_components,
        )
        parsed_candidates = step_input_candidates
        if not selected_files:
            parsed_candidates = request.adapter.source_candidates(pipeline_input_files)
            initially_matched = SourceCandidateMatcher.match_candidates(
                candidates=parsed_candidates,
                binding=request.binding,
                inherit_components=inherit_components,
            )
            selected_files = SourceBindingMatchCandidateUniverse(
                step_input_candidates=step_input_candidates,
                target_candidates=initially_matched,
                pipeline_candidates=parsed_candidates,
            ).image_set_candidates(
                request.alias,
                self._match_scope(request),
            )
        AdapterProfileLog.source_candidates(
            SourceCandidateProfileEvent(
                label="source_candidates_match",
                seconds=time.perf_counter() - match_started_at,
                alias=request.alias,
                source=SourceBindingOrigin.PIPELINE_START,
                count=len(selected_files),
            )
        )
        selected_files = self.require_matched_candidates(
            MatchedSourceCandidatesRequest.from_resolution(
                request,
                matched=selected_files,
                candidates=parsed_candidates,
                source_description="pipeline start",
            )
        )
        selected_files = _prefer_current_source_candidates(
            request.adapter.source_binding_context,
            selected_files,
        )
        load_started_at = time.perf_counter()
        payload = _load_pipeline_start_stack(
            adapter=request.adapter,
            selected_sources=selected_files,
            current_image=request.current_image,
        )
        AdapterProfileLog.source_candidates(
            SourceCandidateProfileEvent(
                label="source_candidates_load",
                seconds=time.perf_counter() - load_started_at,
                alias=request.alias,
                count=len(selected_files),
            )
        )
        return payload

    def _current_step_matched_candidates(
        self,
        request: SourceBindingResolutionRequest,
        *,
        step_input_candidates: tuple[ParsedSourceCandidate, ...],
        inherit_components: Mapping[str, str],
    ) -> tuple[ParsedSourceCandidate, ...]:
        """Resolve pipeline-start aliases from the current input universe when possible."""
        if not step_input_candidates:
            return ()
        if not SourceCandidateMatcher.supports_metadata_selectors(
            step_input_candidates,
            request.binding,
        ):
            return ()
        current_matches = SourceCandidateMatcher.match_candidates(
            candidates=step_input_candidates,
            binding=request.binding,
            inherit_components=inherit_components,
        )
        if not current_matches:
            return ()
        selected_sources = SourceBindingMatchCandidateUniverse(
            step_input_candidates=step_input_candidates,
            target_candidates=current_matches,
            pipeline_candidates=step_input_candidates,
        ).image_set_candidates(
            request.alias,
            self._match_scope(request),
        )
        if not selected_sources:
            return ()
        selected_paths = tuple(source.path for source in selected_sources)
        current_payload_selection = CurrentStepPayloadSelector.from_pipeline_start(
            adapter=request.adapter,
            current_image=request.current_image,
        ).resolve(
            selected_paths=selected_paths,
            current_image=request.current_image,
        )
        if not current_payload_selection.is_matched:
            return ()
        return selected_sources

    @staticmethod
    def _match_scope(
        request: SourceBindingResolutionRequest,
    ) -> SourceBindingImageSetMatchScope:
        return SourceBindingImageSetMatchScope(
            plan=request.adapter.source_binding_plan.match_plan,
            bindings=request.adapter.source_binding_plan.bindings,
        )


def _prefer_current_source_candidates(
    context: SourceBindingRuntimeContext,
    selected_sources: tuple[ParsedSourceCandidate, ...],
) -> tuple[ParsedSourceCandidate, ...]:
    """Narrow selected pipeline sources to current step-input identities."""
    current_paths = tuple(
        context.step_input_source_paths.get(path, path)
        for path in context.current_step_input_files
    )
    current_identity = ParsedSourceCandidatePathIdentity.from_paths(current_paths)
    if current_identity.is_empty:
        return selected_sources
    current_sources = tuple(
        source
        for source in selected_sources
        if source.path_identity.intersects(current_identity)
    )
    return current_sources or selected_sources

@dataclass(frozen=True, slots=True)
class CurrentStepPayloadSelection:
    """Typed result for matching selected sources against the current payload."""

    payload: ImagePayloadValue | None = None

    @property
    def is_matched(self) -> bool:
        return self.payload is not None

    def require_payload(self) -> ImagePayloadValue:
        if self.payload is None:
            raise RuntimeError("Current-step payload selection did not match.")
        return self.payload


@dataclass(slots=True)
class CurrentStepPayloadSelectionMemo:
    """Per-current-image memo for source-binding current-stack selections."""

    slices: tuple[ImagePayloadValue, ...] | None = None
    selected_payloads: dict[
        tuple[
            CurrentStepPayloadSelectionMode,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ],
        ImagePayloadValue,
    ] = field(default_factory=dict)


class CurrentStepPayloadSelectionProcessCache(IdentityBoundProcessCache):
    """Process-local owner for current-step source-binding selections."""

    registry_key = "current_step_payload_selection"


@dataclass(frozen=True, slots=True)
class CurrentStepPayloadSelector:
    """Select source-bound paths from the current step payload."""

    current_files: tuple[str, ...]
    current_source_paths: tuple[str, ...]
    current_image_paths: tuple[str, ...]
    mode: CurrentStepPayloadSelectionMode

    @classmethod
    def from_pipeline_start(
        cls,
        *,
        adapter: CellProfilerRuntimeAdapter,
        current_image: CellProfilerCurrentImage,
    ) -> "CurrentStepPayloadSelector":
        current_files = adapter.source_binding_context.current_step_input_files
        metadata = image_payload_metadata(current_image)
        current_image_paths = tuple(
            str(path)
            for path in (*metadata.source_image_provenance_planes.paths, metadata.source_path)
            if path is not None and str(path)
        )
        current_source_path_values: list[str] = []
        step_input_source_paths = adapter.source_binding_context.step_input_source_paths
        for path in current_files:
            if path in step_input_source_paths:
                current_source_path_values.append(step_input_source_paths[path])
                continue
            current_source_path_values.append(path)
        current_source_paths = tuple(current_source_path_values)
        return cls(
            current_files=current_files,
            current_source_paths=current_source_paths,
            current_image_paths=current_image_paths,
            mode=CurrentStepPayloadSelectionMode.PRESERVE_CURRENT,
        )

    @classmethod
    def from_step_input_context(
        cls,
        context: SourceBindingRuntimeContext,
    ) -> "CurrentStepPayloadSelector":
        current_files = context.current_image_files
        current_source_paths = tuple(
            context.step_input_source_paths[path]
            if path in context.step_input_source_paths
            else path
            for path in current_files
        )
        return cls(
            current_files=current_files,
            current_source_paths=current_source_paths,
            current_image_paths=(),
            mode=CurrentStepPayloadSelectionMode.STEP_INPUT_NATURAL,
        )

    def resolve(
        self,
        *,
        selected_paths: tuple[str, ...],
        current_image: CellProfilerCurrentImage,
    ) -> CurrentStepPayloadSelection:
        if not self.current_files:
            return CurrentStepPayloadSelection()
        if (
            self.mode is CurrentStepPayloadSelectionMode.PRESERVE_CURRENT
            and not self.current_image_paths
        ):
            return CurrentStepPayloadSelection()
        selected_indexes = self._selected_current_indexes(selected_paths)
        if selected_indexes == tuple(range(len(self.current_files))):
            return CurrentStepPayloadSelection(
                payload=self._exact_payload(current_image),
            )
        if selected_indexes is None:
            return CurrentStepPayloadSelection()

        memo = self._memo(current_image)
        cache_key = (
            self.mode,
            self.current_files,
            self.current_source_paths,
            selected_paths,
        )
        selected_payload = memo.selected_payloads.get(cache_key)
        if selected_payload is None:
            selected_payload = RestackLikePayloadAuthority.restack(
                [
                    self._slices(memo, current_image)[index]
                    for index in selected_indexes
                ],
                current_image,
            )
            memo.selected_payloads[cache_key] = selected_payload
        return CurrentStepPayloadSelection(payload=selected_payload)

    def _selected_current_indexes(
        self,
        selected_paths: tuple[str, ...],
    ) -> tuple[int, ...] | None:
        path_indexes = self._current_path_indexes()
        selected_indexes: list[int] = []
        for selected_path in selected_paths:
            selected_index = None
            for identity in SourceBindingRuntimeContext.source_path_identities(
                str(selected_path)
            ):
                selected_index = path_indexes.get(identity)
                if selected_index is not None:
                    break
            if selected_index is None:
                return None
            selected_indexes.append(selected_index)
        return tuple(selected_indexes)

    def _current_path_indexes(self) -> Mapping[str, int]:
        indexes: dict[str, int] = {}
        for index, path in enumerate(self.current_files):
            self._add_path_index(indexes, path, index)
        for index, path in enumerate(self.current_source_paths):
            self._add_path_index(indexes, path, index)
        for index, path in enumerate(self.current_image_paths):
            self._add_path_index(indexes, path, index)
        return MappingProxyType(indexes)

    @staticmethod
    def _add_path_index(
        indexes: dict[str, int],
        path: str,
        index: int,
    ) -> None:
        for identity in SourceBindingRuntimeContext.source_path_identities(str(path)):
            indexes.setdefault(identity, index)

    def _exact_payload(
        self,
        current_image: CellProfilerCurrentImage,
    ) -> ImagePayloadValue:
        if (
            self.mode is CurrentStepPayloadSelectionMode.STEP_INPUT_NATURAL
            and len(self.current_files) == 1
        ):
            return _natural_step_input_payload(current_image)
        return current_image

    def _slices(
        self,
        memo: CurrentStepPayloadSelectionMemo,
        current_image: CellProfilerCurrentImage,
    ) -> tuple[ImagePayloadValue, ...]:
        if memo.slices is None:
            memo.slices = tuple(_unstack_payload(current_image))
        return memo.slices

    @classmethod
    def _memo(
        cls,
        current_image: CellProfilerCurrentImage,
    ) -> CurrentStepPayloadSelectionMemo:
        cache = cast(
            CurrentStepPayloadSelectionProcessCache,
            CurrentStepPayloadSelectionProcessCache.process_cache(),
        )
        memo = cache.get_bound(current_image)
        if memo is not None:
            return cast(CurrentStepPayloadSelectionMemo, memo)
        return cast(
            CurrentStepPayloadSelectionMemo,
            cache.put_bound(current_image, CurrentStepPayloadSelectionMemo()),
        )


@dataclass(slots=True)
class StepInputSourcePayloadProcessCache(
    RegisteredProcessLocalBoundedCache[
        StepInputSourcePayloadCacheKey,
        StepInputPayloadCacheValue,
    ]
):
    """Process-local cache for loaded step-input source-binding payloads."""

    max_entries: int = 64


@dataclass(slots=True)
class PipelineStartSourcePayloadProcessCache(
    RegisteredProcessLocalBoundedCache[
        PipelineStartSourcePayloadCacheKey,
        PipelineStartPayloadCacheValue,
    ]
):
    """Process-local cache for loaded pipeline-start source-binding payloads."""

    max_entries: int = 64


class PipelineStartSourceFileLoader(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for loading selected pipeline-start source files."""

    __registry_key__ = "loader_key"
    __skip_if_no_key__ = True
    loader_key: ClassVar[str | None] = None

    @classmethod
    @lru_cache(maxsize=None)
    def for_paths(
        cls,
        selected_paths: tuple[str, ...],
    ) -> "PipelineStartSourceFileLoader":
        matching_loaders = tuple(
            loader
            for loader in (
                loader_type() for loader_type in cls.__registry__.values()
            )
            if loader.accepts_all(selected_paths)
        )
        if len(matching_loaders) == 1:
            return matching_loaders[0]
        suffixes = sorted({Path(path).suffix.lower() for path in selected_paths})
        if not matching_loaders:
            raise RuntimeError(
                "Pipeline-start source resolution has no registered loader for "
                f"selected source suffixes {suffixes!r}."
            )
        raise RuntimeError(
            "Pipeline-start source resolution has ambiguous registered loaders for "
            f"selected source suffixes {suffixes!r}."
        )

    def accepts_all(self, selected_paths: tuple[str, ...]) -> bool:
        return bool(selected_paths) and all(
            self.accepts_path(path) for path in selected_paths
        )

    @abstractmethod
    def accepts_path(self, path: str) -> bool:
        """Return whether this loader owns one source file path."""

    @abstractmethod
    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[ImagePayloadValue]:
        """Load selected source files as stackable image-like payloads."""

    def source_payload_with_metadata(
        self,
        request: PipelineStartSourcePayloadRequest,
    ) -> ImagePayloadValue:
        """Attach loader-owned source metadata to an image-like payload."""

        metadata = image_payload_metadata(request.payload)
        if not metadata.has_values:
            metadata = ImagePayloadSourceMetadataContext(
                SourceImageIdentity(request.metadata_source_path),
                read_backend=request.backend,
                filemanager=request.context.filemanager,
            ).metadata_request(request.payload).metadata()
        metadata = _source_payload_metadata_with_candidate_context(metadata, request)
        return cast(
            ImagePayloadValue,
            RuntimeImagePayloadContext(
                cast(ImagePayloadValue, image_payload_data(request.payload)),
                cast(ImagePayloadMaskValue, image_payload_mask(request.payload)),
                metadata,
            ).payload(),
        )

class OpenHCSImageSourceFileLoader(PipelineStartSourceFileLoader):
    """Load normal image sources through the OpenHCS VFS filemanager."""

    loader_key = "openhcs_image"

    def accepts_path(self, path: str) -> bool:
        return is_image_path(path)

    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[ImagePayloadValue]:
        context = RequireProcessingContextBoundaryPolicy(request.adapter).context
        loaded_images = context.filemanager.load_batch(
            list(request.storage_paths),
            request.backend,
            **request.source_load_plan.filemanager_load_kwargs(request.backend),
        )
        return [
            self.source_payload_with_metadata(
                _source_payload_for_declared_image_type(
                    PipelineStartSourcePayloadRequest.from_candidate(
                        payload=payload,
                        source=source,
                        storage_path=storage_path,
                        load_request=request,
                    )
                )
            )
            for payload, source, storage_path in zip(
                loaded_images,
                request.selected_sources,
                request.storage_paths,
                strict=True,
            )
        ]

class MatlabMatrixSourceFileLoader(PipelineStartSourceFileLoader):
    """Load CellProfiler MATLAB matrix image sources such as illumination files."""

    loader_key = "matlab_matrix"

    def accepts_path(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".mat"

    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[ImagePayloadValue]:
        return [
            self.source_payload_with_metadata(
                PipelineStartSourcePayloadRequest.from_candidate(
                    payload=cast(ImagePayloadValue, self._load_matrix(source.resolved_path)),
                    source=source,
                    storage_path=storage_path,
                    load_request=request,
                )
            )
            for source, storage_path in zip(
                request.selected_sources,
                request.storage_paths,
                strict=True,
            )
        ]

    def _load_matrix(self, path: str) -> ImagePayloadValue:
        from scipy.io import loadmat

        payloads = _matlab_numeric_arrays(loadmat(path))
        if not payloads:
            raise RuntimeError(
                f"MATLAB source file {path!r} contains no numeric image arrays."
            )
        if len(payloads) == 1:
            return payloads[0][1]
        image_payloads = tuple(
            payload for name, payload in payloads if name.strip().lower() == "image"
        )
        if len(image_payloads) == 1:
            return image_payloads[0]
        names = tuple(name for name, _payload in payloads)
        raise RuntimeError(
            f"MATLAB source file {path!r} contains multiple numeric arrays "
            f"{names!r}; expected exactly one payload or one 'Image' payload."
        )

class NumpyArraySourceFileLoader(PipelineStartSourceFileLoader):
    """Load NumPy array image sources such as saved illumination functions."""

    loader_key = "numpy_array"

    def accepts_path(self, path: str) -> bool:
        return Path(path).suffix.lower() in FileFormat.NUMPY.value

    def load_slices(self, request: PipelineStartSourceLoadRequest) -> list[ImagePayloadValue]:
        return [
            _numpy_array_source_payload_with_metadata(
                PipelineStartSourcePayloadRequest.from_candidate(
                    payload=cast(
                        ImagePayloadValue,
                        self._load_array(
                            storage_path,
                            request,
                        ),
                    ),
                    source=source,
                    storage_path=storage_path,
                    load_request=request,
                )
            )
            for source, storage_path in zip(
                request.selected_sources,
                request.storage_paths,
                strict=True,
            )
        ]

    def _load_array(
        self,
        path: str,
        request: PipelineStartSourceLoadRequest,
    ) -> ImagePayloadValue:
        payload = source_schema_auxiliary_payload(path)
        if payload is None:
            if request.backend == Backend.DISK.value:
                payload = np.load(path)
            else:
                payload = RequireProcessingContextBoundaryPolicy(
                    request.adapter
                ).context.filemanager.load_batch(
                    [path],
                    request.backend,
                )[0]
        if not _is_numeric_array_payload(payload):
            raise RuntimeError(
                f"NumPy source file {path!r} does not contain a numeric image array."
            )
        return payload

def _numpy_array_source_payload_with_metadata(
    request: PipelineStartSourcePayloadRequest,
) -> ImagePayloadValue:
    """Attach array payload metadata without image-file probing."""
    metadata = image_payload_metadata(request.payload)
    if not metadata.has_values:
        metadata = type(metadata).for_array_payload(
            image_payload_data(request.payload),
            source_path=request.source_path,
        )
    metadata = _source_payload_metadata_with_candidate_context(metadata, request)
    return cast(
        ImagePayloadValue,
        RuntimeImagePayloadContext(
            cast(ImagePayloadValue, image_payload_data(request.payload)),
            cast(ImagePayloadMaskValue, image_payload_mask(request.payload)),
            metadata,
        ).payload(),
    )

def _source_payload_metadata_with_candidate_context(
    metadata: ImagePayloadMetadata,
    request: PipelineStartSourcePayloadRequest,
) -> ImagePayloadMetadata:
    """Fill source payload metadata from the selected source candidate."""
    return metadata.with_source_context_from(
        ImagePayloadMetadata(
            source_path=request.source_path,
            source_component_metadata=request.source_metadata,
        )
    )

def _source_payload_for_declared_image_type(
    request: PipelineStartSourcePayloadRequest,
) -> PipelineStartSourcePayloadRequest:
    """Apply setup-declared source image semantics before module execution."""

    payload = ContextSourceMetadataAuthority.apply_loading_semantics(
        request.payload,
        source_path=request.source_path,
        storage_path=request.metadata_source_path,
        request=request,
    )
    return request.with_payload(cast(ImagePayloadValue, payload))

def _load_step_input_stack(
    *,
    request: SourceBindingResolutionRequest,
    selected_sources: tuple[ParsedSourceCandidate, ...],
) -> ImagePayloadValue:
    context = request.adapter.source_binding_context
    if context.step_input_dir is None or context.step_input_storage_backend is None:
        raise RuntimeError(
            "Step-input selector resolution needs step_input_dir and "
            "step_input_storage_backend when selected files are outside the current stack."
        )
    processing_context = RequireProcessingContextBoundaryPolicy(request.adapter).context
    cache_key = _step_input_payload_cache_key(
        request,
        selected_sources,
        processing_context.filemanager,
    )
    cache = StepInputSourcePayloadProcessCache.process_cache()
    contextualized = cache.cached_value(cache_key)
    if contextualized is None:
        storage_paths = tuple(candidate.path for candidate in selected_sources)
        loaded = processing_context.filemanager.load_batch(
            list(storage_paths),
            context.step_input_storage_backend,
        )
        if not loaded:
            raise RuntimeError(
                "Step-input source resolution loaded no payloads from "
                f"{list(storage_paths)}."
            )
        if len(loaded) != len(selected_sources):
            raise RuntimeError(
                "Step-input source resolution loaded a different number of payloads "
                f"than selected candidates: {len(loaded)} loaded for "
                f"{len(selected_sources)} selected."
            )
        contextualized = tuple(
            SourceImagePayloadSemantics.from_source_metadata(
                candidate.metadata,
                candidate.resolved_path,
                context.step_input_source_backend,
                processing_context.filemanager,
            ).apply(payload)
            for candidate, payload in zip(selected_sources, loaded, strict=True)
        )
        cache.store_value(cache_key, contextualized)
    return RestackLikePayloadAuthority.restack(list(contextualized), request.current_image)


def _step_input_payload_cache_key(
    request: SourceBindingResolutionRequest,
    selected_sources: tuple[ParsedSourceCandidate, ...],
    filemanager: CellProfilerFileManager,
) -> StepInputSourcePayloadCacheKey:
    context = request.adapter.source_binding_context
    if context.step_input_storage_backend is None:
        raise RuntimeError("Step-input payload cache requires a storage backend.")
    if context.step_input_source_backend is None:
        raise RuntimeError("Step-input payload cache requires a source backend.")
    return StepInputSourcePayloadCacheKey.from_sources(
        storage_backend=context.step_input_storage_backend,
        source_backend=context.step_input_source_backend,
        filemanager=filemanager,
        context=context,
        selected_sources=selected_sources,
    )

def _load_pipeline_start_stack(
    *,
    adapter: CellProfilerRuntimeAdapter,
    selected_sources: tuple[ParsedSourceCandidate, ...],
    current_image: CellProfilerCurrentImage,
) -> ImagePayloadValue:
    selected_paths = tuple(source.path for source in selected_sources)
    if not selected_paths:
        raise RuntimeError("Pipeline-start source selection cannot load zero paths.")
    current_payload_resolution = (
        CurrentStepPayloadSelector.from_pipeline_start(
            adapter=adapter,
            current_image=current_image,
        ).resolve(
            selected_paths=selected_paths,
            current_image=current_image,
        )
    )
    if current_payload_resolution.is_matched:
        return current_payload_resolution.payload
    backend = adapter.source_binding_context.pipeline_input_backend
    if backend is None:
        raise RuntimeError(
            "Pipeline-start source resolution requires pipeline_input_backend."
        )
    load_request = PipelineStartSourceLoadRequest(
        adapter=adapter,
        selected_sources=selected_sources,
        backend=backend,
        source_load_plan=adapter.source_load_plan,
    )
    processing_context = RequireProcessingContextBoundaryPolicy(adapter).context
    cache_key = PipelineStartSourcePayloadCacheKey.from_sources(
        backend=backend,
        filemanager=processing_context.filemanager,
        context=adapter.source_binding_context,
        selected_sources=selected_sources,
    )
    cache = PipelineStartSourcePayloadProcessCache.process_cache()
    loaded_payloads = cache.cached_value(cache_key)
    if loaded_payloads is None:
        loaded_payloads = tuple(
            PipelineStartSourceFileLoader.for_paths(
                load_request.storage_paths
            ).load_slices(load_request)
        )
        cache.store_value(cache_key, loaded_payloads)
    if not loaded_payloads:
        raise RuntimeError(
            "Pipeline-start source resolution loaded no payloads from "
            f"{list(selected_paths)}."
        )
    return RestackLikePayloadAuthority.restack(list(loaded_payloads), current_image)

def _matlab_numeric_arrays(
    mat_payload: Mapping[str, ImagePayloadValue],
) -> tuple[tuple[str, ImagePayloadValue], ...]:
    return tuple(
        (name, payload)
        for name, payload in mat_payload.items()
        if not MatlabPayloadEntryName(name).is_private_metadata
        and _is_numeric_array_payload(payload)
    )

def _is_numeric_array_payload(payload: ImagePayloadValue) -> bool:
    if not isinstance(payload, (RuntimeArrayPayload, np.ndarray)):
        return False
    return payload.dtype.kind in {"b", "u", "i", "f", "c"} and payload.ndim >= 2

def _unstack_payload(payload: ImagePayloadValue) -> list[ImagePayloadValue]:
    return list(payload_slices_for_alignment(payload))


def _natural_step_input_payload(
    current_image: CellProfilerCurrentImage,
) -> ImagePayloadValue:
    if not isinstance(current_image, (RuntimeArrayPayload, np.ndarray)):
        return current_image
    if current_image.ndim == 2:
        return current_image
    return RestackLikePayloadAuthority.restack(
        _unstack_payload(current_image),
        current_image,
    )


class RestackLikePayloadAuthority:
    """Restack selected image payload slices while preserving payload context."""

    @classmethod
    def restack(
        cls,
        slices: list[ImagePayloadValue],
        reference_payload: ImagePayloadValue,
    ) -> ImagePayloadValue:
        if not slices:
            raise ValueError("Cannot restack an empty slice list.")
        if len(slices) == 1:
            return slices[0]
        slice_data = tuple(image_payload_data(slice_payload) for slice_payload in slices)
        memory_type = detect_memory_type(image_payload_data(reference_payload))
        stacked = ImageStackLayout.stack_slices_or_single_stack(
            slice_data,
            memory_type=memory_type,
            gpu_id=0,
        )
        return cast(
            ImagePayloadValue,
            RuntimeImagePayloadContext(
                cast(ImagePayloadValue, stacked),
                cast(ImagePayloadMaskValue, cls.stack_masks(slices, memory_type)),
                ImagePayloadMetadataCompositionRequest(slices).metadata(),
            ).payload(),
        )

    @staticmethod
    def stack_masks(
        slices: list[ImagePayloadValue],
        memory_type: str,
    ) -> ImagePayloadMaskValue:
        masks = tuple(image_payload_mask(slice_payload) for slice_payload in slices)
        if not any(mask is not None for mask in masks):
            return None
        slice_data = tuple(image_payload_data(slice_payload) for slice_payload in slices)
        resolved_masks = [
            np.ones(np.asarray(data).shape[:2], dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
            for data, mask in zip(slice_data, masks)
        ]
        return ImageStackLayout.stack_slices_or_single_stack(
            resolved_masks,
            memory_type=memory_type,
            gpu_id=0,
        )
