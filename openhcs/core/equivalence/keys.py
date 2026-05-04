"""Canonical runtime equivalence identity keys."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.equivalence.policy import (
    normalize_runtime_identifier,
    normalize_runtime_source_name,
)
from openhcs.core.runtime_semantics import MeasurementScope, MeasurementSubject


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementSubjectKey:
    """Canonical measured subject for semantic measurement comparison."""

    scope: MeasurementScope
    name: str | None = None

    def __post_init__(self) -> None:
        scope = (
            self.scope
            if isinstance(self.scope, MeasurementScope)
            else MeasurementScope(self.scope)
        )
        name = (
            normalize_runtime_identifier(self.name)
            if self.name is not None
            else None
        )
        if name == "":
            raise ValueError("RuntimeMeasurementSubjectKey.name cannot be empty.")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "name", name)

    @classmethod
    def from_subject(cls, subject: MeasurementSubject) -> "RuntimeMeasurementSubjectKey":
        """Build a comparison subject key from typed runtime semantics."""
        return cls(scope=subject.scope, name=subject.name)

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.scope.value, self.name or "")

    def to_cache_payload(self) -> tuple[str, str | None]:
        """Return a pickle/JSON-stable semantic cache payload."""
        return (self.scope.value, self.name)

    @classmethod
    def from_cache_payload(cls, payload: object) -> "RuntimeMeasurementSubjectKey":
        """Rebuild a measurement subject from a semantic cache payload."""
        scope, name = payload  # type: ignore[misc]
        return cls(MeasurementScope(str(scope)), None if name is None else str(name))


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureKey:
    """Canonical measured feature on one semantic subject."""

    subject: RuntimeMeasurementSubjectKey
    feature_name: str
    statistic: str = "value"
    source_name: str | None = None

    def __post_init__(self) -> None:
        feature_name = self.feature_name.strip()
        if not feature_name:
            raise ValueError("RuntimeMeasurementFeatureKey.feature_name cannot be empty.")
        statistic = normalize_runtime_identifier(self.statistic)
        if not statistic:
            raise ValueError("RuntimeMeasurementFeatureKey.statistic cannot be empty.")
        source_name = (
            normalize_runtime_source_name(self.source_name)
            if self.source_name is not None
            else None
        )
        if source_name == "":
            raise ValueError("RuntimeMeasurementFeatureKey.source_name cannot be empty.")
        object.__setattr__(self, "feature_name", feature_name)
        object.__setattr__(self, "statistic", statistic)
        object.__setattr__(self, "source_name", source_name)

    @property
    def sort_key(self) -> tuple[tuple[str, str], str, str, str]:
        return (
            self.subject.sort_key,
            self.statistic,
            self.feature_name,
            self.source_name or "",
        )

    def to_cache_payload(
        self,
    ) -> tuple[tuple[str, str | None], str, str, str | None]:
        """Return a pickle/JSON-stable semantic cache payload."""
        return (
            self.subject.to_cache_payload(),
            self.feature_name,
            self.statistic,
            self.source_name,
        )

    @classmethod
    def from_cache_payload(cls, payload: object) -> "RuntimeMeasurementFeatureKey":
        """Rebuild a measurement feature key from a semantic cache payload."""
        subject, feature_name, statistic, source_name = payload  # type: ignore[misc]
        return cls(
            RuntimeMeasurementSubjectKey.from_cache_payload(subject),
            str(feature_name),
            str(statistic),
            None if source_name is None else str(source_name),
        )
