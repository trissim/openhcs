"""Profiling event projection for the CellProfiler runtime adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_values import ObjectLabelValue
from openhcs.core.source_image_provenance import SourceComponentMetadata
from openhcs.core.source_bindings import SourceBindingOrigin
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    object_label_artifact_profile_fields,
)

logger = logging.getLogger(__name__)

AdapterProfileFieldValue = (
    str | int | bool | tuple[int, ...] | SourceComponentMetadata | None
)

AdapterProfileFields = dict[str, AdapterProfileFieldValue]

@dataclass(frozen=True, slots=True)
class SourceCandidateProfileEvent:
    """Nominal profiler event for source-candidate matching and loading."""

    label: str
    seconds: float
    count: int
    alias: str | None = None
    source: SourceBindingOrigin | None = None

    def fields(self) -> AdapterProfileFields:
        fields: AdapterProfileFields = {"count": self.count}
        if self.alias is not None:
            fields["alias"] = self.alias
        if self.source is not None:
            fields["source"] = self.source.value
        return fields

class AdapterProfileLog:
    """Authoritative projection for runtime adapter profiling event fields."""

    @staticmethod
    def log(label: str, seconds: float, **fields: AdapterProfileFieldValue) -> None:
        RuntimeProfileLogger.log(logger, label, seconds, **fields)

    @staticmethod
    def label_batch(
        label: str,
        seconds: float,
        *,
        feature_name: str,
        count: int | None = None,
    ) -> None:
        fields: AdapterProfileFields = {"feature": feature_name}
        if count is not None:
            fields["count"] = count
        AdapterProfileLog.log(label, seconds, **fields)

    @staticmethod
    def artifact(
        label: str,
        seconds: float,
        *,
        artifact_name: str,
        kind: ArtifactKind | str | None = None,
        payload_type: str | None = None,
        group_key: str | None = None,
        extra_fields: Mapping[str, AdapterProfileFieldValue] | None = None,
    ) -> None:
        fields: AdapterProfileFields = {"artifact": artifact_name}
        if kind is not None:
            fields["kind"] = kind.value if isinstance(kind, ArtifactKind) else kind
        if payload_type is not None:
            fields["payload_type"] = payload_type
        if group_key is not None:
            fields["group_key"] = group_key
        if extra_fields:
            fields.update(extra_fields)
        AdapterProfileLog.log(label, seconds, **fields)

    @staticmethod
    def measurement_artifact(
        label: str,
        seconds: float,
        *,
        artifact_name: str,
        object_name: str | None = None,
        fields_declared: bool | None = None,
    ) -> None:
        fields: AdapterProfileFields = {"artifact": artifact_name}
        if object_name is not None:
            fields["object"] = object_name
        if fields_declared is not None:
            fields["fields"] = fields_declared
        AdapterProfileLog.log(label, seconds, **fields)

    @staticmethod
    def object_label_artifact(
        label: str,
        seconds: float,
        *,
        artifact_name: str,
        payload_type: str,
        labels: ObjectLabelValue,
    ) -> None:
        AdapterProfileLog.artifact(
            label,
            seconds,
            artifact_name=artifact_name,
            kind=ArtifactKind.OBJECT_LABELS,
            payload_type=payload_type,
            extra_fields=object_label_artifact_profile_fields(labels),
        )

    @staticmethod
    def measurement_cache(
        label: str,
        seconds: float,
        *,
        object_count: int,
        feature_count: int,
    ) -> None:
        AdapterProfileLog.log(
            label,
            seconds,
            objects=object_count,
            features=feature_count,
        )

    @staticmethod
    def measurement_cache_policy(
        seconds: float,
        *,
        object_count: int,
        feature_count: int,
        policy_name: str,
    ) -> None:
        AdapterProfileLog.log(
            "adapter_measurement_cache_policy",
            seconds,
            policy=policy_name,
            objects=object_count,
            features=feature_count,
        )

    @staticmethod
    def measurement_slice_mismatch(
        seconds: float,
        *,
        object_name: str,
        measurement_slices: int,
        label_slices: int,
    ) -> None:
        AdapterProfileLog.log(
            "adapter_multiplane_measurement_slice_mismatch",
            seconds,
            object=object_name,
            measurement_slices=measurement_slices,
            label_slices=label_slices,
        )

    @staticmethod
    def source_candidates(event: SourceCandidateProfileEvent) -> None:
        AdapterProfileLog.log(event.label, event.seconds, **event.fields())

@dataclass(frozen=True, slots=True)
class NativeRecordProfileContext:
    """Profiling context for one native artifact materialization."""

    artifact_name: str
    kind: ArtifactKind

    def event(
        self,
        label: str,
        seconds: float,
        *,
        payload_type: str | None = None,
        group_key: str | None = None,
    ) -> None:
        AdapterProfileLog.artifact(
            label,
            seconds,
            artifact_name=self.artifact_name,
            kind=self.kind,
            payload_type=payload_type,
            group_key=group_key,
        )

    def group_event(
        self,
        label: str,
        seconds: float,
        group_key: str | None,
    ) -> None:
        self.event(label, seconds, group_key=group_key)

    def normalized_value(
        self,
        seconds: float,
        *,
        payload_type: str,
        group_key: str | None,
    ) -> None:
        self.event(
            "adapter_normalize_artifact_value",
            seconds,
            payload_type=payload_type,
            group_key=group_key,
        )
