"""Runtime profiling helpers for CellProfiler integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging

from openhcs.core.artifacts import ArtifactType, ObjectLabelsArtifactType
from openhcs.core.runtime_object_labels import ObjectLabelValue
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.interop.cellprofiler.runtime.profile_fields import (
    object_label_artifact_profile_fields,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument, RuntimeProfileFieldValue


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CellProfilerRuntimeProfileEvent:
    """Deferred profile event with explicit structured fields."""

    name: str
    elapsed: float
    fields: tuple[tuple[str, RuntimeProfileFieldValue], ...] = ()


class CellProfilerRuntimeProfileLogger:
    """Runtime profile sink with one owner for environment-gated logging."""

    @staticmethod
    def enabled() -> bool:
        """Return whether CellProfiler runtime profile events will be emitted."""
        return RuntimeProfileLogger.enabled()

    @staticmethod
    def log_module_profile(
        label: str,
        seconds: float,
        **fields: RuntimeCallableArgument,
    ) -> None:
        RuntimeProfileLogger.log(logger, label, seconds, **fields)

    @classmethod
    def log_module_profile_deferred(
        cls,
        label: str,
        seconds: float,
        fields: Callable[[], Mapping[str, RuntimeCallableArgument]],
    ) -> None:
        """Emit a profile event whose fields are only built when profiling is on."""
        if not cls.enabled():
            return
        cls.log_module_profile(label, seconds, **dict(fields()))

    @classmethod
    def label_batch(
        cls,
        label: str,
        seconds: float,
        *,
        feature_name: str,
        count: int | None = None,
    ) -> None:
        fields: dict[str, RuntimeCallableArgument] = {"feature": feature_name}
        if count is not None:
            fields["count"] = count
        cls.log_module_profile(label, seconds, **fields)

    @classmethod
    def artifact(
        cls,
        label: str,
        seconds: float,
        *,
        artifact_name: str,
        kind: ArtifactType | str | None = None,
        payload_type: str | None = None,
        group_key: str | None = None,
        extra_fields: Mapping[str, RuntimeCallableArgument] | None = None,
    ) -> None:
        fields: dict[str, RuntimeCallableArgument] = {"artifact": artifact_name}
        if kind is not None:
            fields["kind"] = ArtifactType.coerce(kind).value
        if payload_type is not None:
            fields["payload_type"] = payload_type
        if group_key is not None:
            fields["group_key"] = group_key
        if extra_fields:
            fields.update(extra_fields)
        cls.log_module_profile(label, seconds, **fields)

    @classmethod
    def measurement_artifact(
        cls,
        label: str,
        seconds: float,
        *,
        artifact_name: str,
        object_name: str | None = None,
        fields_declared: bool | None = None,
    ) -> None:
        fields: dict[str, RuntimeCallableArgument] = {"artifact": artifact_name}
        if object_name is not None:
            fields["object"] = object_name
        if fields_declared is not None:
            fields["fields"] = fields_declared
        cls.log_module_profile(label, seconds, **fields)

    @classmethod
    def object_label_artifact(
        cls,
        label: str,
        seconds: float,
        *,
        artifact_name: str,
        payload_type: str,
        labels: ObjectLabelValue,
    ) -> None:
        cls.artifact(
            label,
            seconds,
            artifact_name=artifact_name,
            kind=ObjectLabelsArtifactType,
            payload_type=payload_type,
            extra_fields=object_label_artifact_profile_fields(labels),
        )
