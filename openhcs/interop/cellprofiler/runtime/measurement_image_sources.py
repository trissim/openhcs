"""CellProfiler measurement-image source ownership strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactSpec
from openhcs.interop.cellprofiler.runtime.mapping_lookup import MappingValueLookup
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerRuntimeValue
from openhcs.interop.cellprofiler.runtime.policy_registry import NoSourceImageNameMixin


class CellProfilerImageMeasurementSource(ABC, metaclass=AutoRegisterMeta):
    """Nominal source for image-owned measurement row identity."""

    __registry_key__ = "__name__"

    @abstractmethod
    def source_image_name(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> str | None:
        """Return the image name used to qualify recorded measurement rows."""

    @abstractmethod
    def source_image_payload(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerRuntimeValue | None:
        """Return the payload that anchors image-owned measurement rows."""

    def require_produced_artifact(self) -> "ProducedArtifactImageMeasurementSource":
        """Return this source as a produced artifact source, or fail loudly."""
        raise ValueError("Measurement ownership requires an image output.")


class ProducedArtifactImageMeasurementSourceBase(CellProfilerImageMeasurementSource):
    """Measurement source owned by a produced image artifact."""

    artifact_spec: ArtifactSpec

    def source_image_name(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> str | None:
        del request
        return self.artifact_spec.name

    def source_image_payload(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerRuntimeValue | None:
        return MappingValueLookup(
            request.output_values,
            self.artifact_spec.name,
        ).value_or(request.source.payload)

    def require_produced_artifact(self) -> "ProducedArtifactImageMeasurementSource":
        return self


@dataclass(frozen=True, slots=True)
class ProducedArtifactImageMeasurementSource(ProducedArtifactImageMeasurementSourceBase):
    """Measurement source owned by a produced image artifact."""

    artifact_spec: ArtifactSpec


class UnqualifiedRuntimeImageMeasurementSource(
    NoSourceImageNameMixin,
    CellProfilerImageMeasurementSource,
):
    """Measurement source backed by the runtime input image without row naming."""

    def source_image_payload(
        self,
        request: CellProfilerOutputRecordRequest,
    ) -> CellProfilerRuntimeValue | None:
        return request.source.payload
