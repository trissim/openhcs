"""Runtime artifact cache invalidation policies for the CellProfiler adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod

from openhcs.core.artifacts import (
    ArtifactType,
    ArtifactTypeStrategyMatchMixin,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
)
from openhcs.core.registry_strategies import MostDerivedContextStrategyMixin
from openhcs.interop.cellprofiler.runtime.object_label_measurements import (
    object_label_measurement_values_cache,
)
from openhcs.interop.cellprofiler.runtime.runtime_artifact_records import (
    RuntimeArtifactTypePolicyMixin,
)


class RuntimeArtifactCacheInvalidationPolicy(
    RuntimeArtifactTypePolicyMixin,
    ArtifactTypeStrategyMatchMixin,
    MostDerivedContextStrategyMixin[type[ArtifactType]],
    ABC,
):
    """Nominal cache invalidation policy for one runtime artifact domain."""

    @abstractmethod
    def invalidate(self, adapter: "CellProfilerRuntimeAdapter") -> None:
        """Invalidate adapter caches affected by this artifact type."""


class FullRuntimeArtifactCacheInvalidationPolicy(RuntimeArtifactCacheInvalidationPolicy):
    """Conservative invalidation for artifact types without narrower semantics."""

    def invalidate(self, adapter: "CellProfilerRuntimeAdapter") -> None:
        adapter.clear_runtime_query_caches()


RuntimeArtifactCacheInvalidationPolicy.default_policy_type = (
    FullRuntimeArtifactCacheInvalidationPolicy
)


class ImageRuntimeArtifactCacheInvalidationPolicy(FullRuntimeArtifactCacheInvalidationPolicy):
    """Image writes may affect image reads and image-derived measurement alignment."""

    artifact_type = ImageArtifactType

    def invalidate(self, adapter: "CellProfilerRuntimeAdapter") -> None:
        adapter._image_cache.clear()
        adapter._current_image_cache.clear()
        adapter._source_paths_by_image_name_cache.clear()
        adapter._artifact_availability_cache.clear()


class ObjectLabelRuntimeArtifactCacheInvalidationPolicy(
    FullRuntimeArtifactCacheInvalidationPolicy
):
    """Object writes may affect label reads, label domains, and measurement alignment."""

    artifact_type = ObjectLabelsArtifactType

    def invalidate(self, adapter: "CellProfilerRuntimeAdapter") -> None:
        adapter._object_cache.clear()
        object_label_measurement_values_cache(adapter.runtime_value_store).clear()
        adapter._measurement_cache.clear()
        adapter._artifact_availability_cache.clear()


class MeasurementRuntimeArtifactCacheInvalidationPolicy(
    RuntimeArtifactCacheInvalidationPolicy
):
    """Measurement writes invalidate measurement queries without discarding labels/images."""

    artifact_type = MeasurementsArtifactType

    def invalidate(self, adapter: "CellProfilerRuntimeAdapter") -> None:
        adapter.clear_measurement_query_cache()


class RelationshipRuntimeArtifactCacheInvalidationPolicy(
    RuntimeArtifactCacheInvalidationPolicy
):
    """Relationship writes are independent of image/object/measurement read caches."""

    artifact_type = RelationshipsArtifactType

    def invalidate(self, adapter: "CellProfilerRuntimeAdapter") -> None:
        return None
