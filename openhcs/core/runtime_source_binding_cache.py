"""Processing-context-local source-binding runtime caches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from openhcs.core.source_bindings import SourceBindingRuntimeMetadataNormalizer
from openhcs.core.source_metadata import SourceMetadataMapping


@dataclass(slots=True)
class RuntimeSourceBindingContextCache:
    """Cache source-binding data that is invariant across runtime contexts."""

    source_metadata_by_mapping_identity: dict[
        int,
        Mapping[str, SourceMetadataMapping],
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
