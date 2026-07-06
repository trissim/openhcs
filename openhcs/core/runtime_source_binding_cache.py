"""Processing-context-local source-binding runtime caches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from openhcs.core.source_bindings import (
    SourceBindingRuntimeContext,
    SourceBindingRuntimeMetadataNormalizer,
)
from openhcs.core.source_metadata import SourceMetadataMapping

if TYPE_CHECKING:
    from openhcs.core.source_binding_selection import SourceUniverseRuntimeState


@dataclass(slots=True)
class RuntimeSourceBindingContextCache:
    """Cache source-binding data that is invariant across runtime contexts."""

    source_metadata_by_mapping_identity: dict[
        int,
        Mapping[str, SourceMetadataMapping],
    ] = field(default_factory=dict)
    runtime_context_by_request_identity: dict[
        tuple[int, tuple[str, ...], object, int | None],
        SourceBindingRuntimeContext,
    ] = field(default_factory=dict)
    runtime_universe_state_by_request_identity: dict[
        tuple[int, tuple[str, ...], object, int | None],
        "SourceUniverseRuntimeState",
    ] = field(default_factory=dict)

    def normalized_source_metadata(
        self,
        source_metadata_by_path: Mapping[str, SourceMetadataMapping],
    ) -> Mapping[str, SourceMetadataMapping]:
        """Return normalized source metadata for a projection-owned mapping."""
        cache_key = id(source_metadata_by_path)
        cached = self.source_metadata_by_mapping_identity.get(cache_key)
        if cached is None:
            cached = SourceBindingRuntimeMetadataNormalizer(
                source_metadata_by_path
            ).normalized()
            self.source_metadata_by_mapping_identity[cache_key] = cached
        return cached

    def runtime_context(
        self,
        *,
        plan: Any,
        matching_files: tuple[str, ...],
        source_backend: object,
        source_projection: object | None,
    ) -> SourceBindingRuntimeContext | None:
        """Return a cached runtime context for one immutable request identity."""
        return self.runtime_context_by_request_identity.get(
            self.runtime_context_key(
                plan=plan,
                matching_files=matching_files,
                source_backend=source_backend,
                source_projection=source_projection,
            )
        )

    def runtime_universe_state(
        self,
        *,
        plan: Any,
        matching_files: tuple[str, ...],
        source_backend: object,
        source_projection: object | None,
    ) -> "SourceUniverseRuntimeState | None":
        """Return cached source-universe state for one request identity."""
        return self.runtime_universe_state_by_request_identity.get(
            self.runtime_context_key(
                plan=plan,
                matching_files=matching_files,
                source_backend=source_backend,
                source_projection=source_projection,
            )
        )

    def store_runtime_universe_state(
        self,
        runtime_state: "SourceUniverseRuntimeState",
        *,
        plan: Any,
        matching_files: tuple[str, ...],
        source_backend: object,
        source_projection: object | None,
    ) -> "SourceUniverseRuntimeState":
        """Cache source-universe state for one request identity."""
        self.runtime_universe_state_by_request_identity[
            self.runtime_context_key(
                plan=plan,
                matching_files=matching_files,
                source_backend=source_backend,
                source_projection=source_projection,
            )
        ] = runtime_state
        return runtime_state

    def store_runtime_context(
        self,
        runtime_context: SourceBindingRuntimeContext,
        *,
        plan: Any,
        matching_files: tuple[str, ...],
        source_backend: object,
        source_projection: object | None,
    ) -> SourceBindingRuntimeContext:
        """Cache a runtime context for one immutable request identity."""
        self.runtime_context_by_request_identity[
            self.runtime_context_key(
                plan=plan,
                matching_files=matching_files,
                source_backend=source_backend,
                source_projection=source_projection,
            )
        ] = runtime_context
        return runtime_context

    @staticmethod
    def runtime_context_key(
        *,
        plan: Any,
        matching_files: tuple[str, ...],
        source_backend: object,
        source_projection: object | None,
    ) -> tuple[int, tuple[str, ...], object, int | None]:
        """Return the process-local identity for one source runtime context."""
        return (
            id(plan),
            tuple(matching_files),
            source_backend,
            None if source_projection is None else id(source_projection),
        )
