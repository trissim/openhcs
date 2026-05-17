"""Typed CellProfiler measurement-name compatibility over core runtime queries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.public_api import declared_public_names
from openhcs.core.runtime_artifact_queries import (
    measurement_scalar_value_for_feature,
    measurement_values_for_feature,
    measurement_values_for_label_slices,
)


class CellProfilerMeasurementFeatureKind(Enum):
    """CellProfiler measurement feature families with structured semantics."""

    OBJECT_COUNT = "object_count"
    CHILD_COUNT = "child_count"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementFeature:
    """Structured view of one CellProfiler measurement feature name."""

    name: str
    kind: CellProfilerMeasurementFeatureKind
    object_name: str | None = None

    @classmethod
    def parse(cls, feature_name: str | None) -> "CellProfilerMeasurementFeature | None":
        """Parse a CellProfiler feature name into nominal feature semantics."""
        candidate = CellProfilerMeasurementFeatureParseCandidate.from_feature_name(
            feature_name
        )
        if candidate.is_absent:
            return None
        parsed = candidate.parse_registered()
        if parsed is not None:
            return parsed
        return candidate.other_feature()

    @classmethod
    def object_count(cls, object_name: str) -> "CellProfilerMeasurementFeature":
        return CellProfilerMeasurementFeatureParser.for_kind(
            CellProfilerMeasurementFeatureKind.OBJECT_COUNT
        ).feature_from_object_name(object_name)

    @classmethod
    def child_count(cls, child_object_name: str) -> "CellProfilerMeasurementFeature":
        return CellProfilerMeasurementFeatureParser.for_kind(
            CellProfilerMeasurementFeatureKind.CHILD_COUNT
        ).feature_from_object_name(child_object_name)

    @classmethod
    def child_count_object_names(
        cls,
        feature_names: tuple[object, ...],
    ) -> tuple[str, ...]:
        """Return ordered unique child object names referenced by count features."""
        child_names = tuple(
            parsed.object_name
            for feature_name in feature_names
            for parsed in (cls.parse(str(feature_name)),)
            if (
                parsed is not None
                and parsed.kind is CellProfilerMeasurementFeatureKind.CHILD_COUNT
                and parsed.object_name is not None
            )
        )
        return tuple(dict.fromkeys(child_names))


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementFeatureParseCandidate:
    """Normalized candidate for fail-soft CellProfiler feature parsing."""

    normalized: str | None

    @classmethod
    def from_feature_name(
        cls,
        feature_name: str | None,
    ) -> "CellProfilerMeasurementFeatureParseCandidate":
        if feature_name is None:
            return cls(None)
        normalized = feature_name.strip()
        return cls(normalized or None)

    @property
    def is_absent(self) -> bool:
        return self.normalized is None

    def parse_registered(self) -> CellProfilerMeasurementFeature | None:
        if self.normalized is None:
            return None
        for parser_type in CellProfilerMeasurementFeatureParser.__registry__.values():
            parsed = parser_type().parse_feature(self.normalized)
            if parsed is not None:
                return parsed
        return None

    def other_feature(self) -> CellProfilerMeasurementFeature:
        if self.normalized is None:
            raise ValueError("Cannot materialize an absent measurement feature.")
        return CellProfilerMeasurementFeature(
            self.normalized,
            CellProfilerMeasurementFeatureKind.OTHER,
        )


class CellProfilerMeasurementFeatureParser(ABC, metaclass=AutoRegisterMeta):
    """Registered parser/renderer for one CellProfiler measurement feature family."""

    __registry_key__ = "kind_key"
    __skip_if_no_key__ = True
    kind: ClassVar[CellProfilerMeasurementFeatureKind | None] = None
    kind_key: ClassVar[str | None] = None

    @classmethod
    def for_kind(
        cls,
        kind: CellProfilerMeasurementFeatureKind,
    ) -> "CellProfilerMeasurementFeatureParser":
        parser_type = cls.__registry__.get(kind.value)
        if parser_type is None:
            raise KeyError(f"No CellProfiler measurement parser registered for {kind}.")
        return parser_type()

    @abstractmethod
    def parse_feature(
        self,
        feature_name: str,
    ) -> CellProfilerMeasurementFeature | None:
        """Return parsed feature semantics when this parser owns the name."""

    @abstractmethod
    def feature_from_object_name(
        self,
        object_name: str,
    ) -> CellProfilerMeasurementFeature:
        """Render a feature name for an object-targeted feature family."""


class CellProfilerObjectCountFeatureParser(CellProfilerMeasurementFeatureParser):
    """Parser for CellProfiler ``Count_<object>`` image-level object counts."""

    kind = CellProfilerMeasurementFeatureKind.OBJECT_COUNT
    kind_key = kind.value
    prefix = "Count_"

    def parse_feature(
        self,
        feature_name: str,
    ) -> CellProfilerMeasurementFeature | None:
        if not feature_name.startswith(self.prefix):
            return None
        object_name = feature_name[len(self.prefix) :].strip()
        if not object_name:
            return None
        return CellProfilerMeasurementFeature(
            name=feature_name,
            kind=CellProfilerMeasurementFeatureKind.OBJECT_COUNT,
            object_name=object_name,
        )

    def feature_from_object_name(
        self,
        object_name: str,
    ) -> CellProfilerMeasurementFeature:
        normalized = object_name.strip()
        if not normalized:
            raise ValueError("Object-count feature requires a non-empty object name.")
        return CellProfilerMeasurementFeature(
            name=f"{self.prefix}{normalized}",
            kind=CellProfilerMeasurementFeatureKind.OBJECT_COUNT,
            object_name=normalized,
        )


class CellProfilerChildCountFeatureParser(CellProfilerMeasurementFeatureParser):
    """Parser for CellProfiler ``Children_<object>_Count`` relationships."""

    kind = CellProfilerMeasurementFeatureKind.CHILD_COUNT
    kind_key = kind.value
    prefix = "Children_"
    suffix = "_Count"

    def parse_feature(
        self,
        feature_name: str,
    ) -> CellProfilerMeasurementFeature | None:
        if not feature_name.startswith(self.prefix):
            return None
        if not feature_name.endswith(self.suffix):
            return None
        object_name = feature_name[len(self.prefix) : -len(self.suffix)].strip()
        if not object_name:
            return None
        return CellProfilerMeasurementFeature(
            name=feature_name,
            kind=CellProfilerMeasurementFeatureKind.CHILD_COUNT,
            object_name=object_name,
        )

    def feature_from_object_name(
        self,
        object_name: str,
    ) -> CellProfilerMeasurementFeature:
        normalized = object_name.strip()
        if not normalized:
            raise ValueError("Child-count feature requires a non-empty child object name.")
        return CellProfilerMeasurementFeature(
            name=f"{self.prefix}{normalized}{self.suffix}",
            kind=CellProfilerMeasurementFeatureKind.CHILD_COUNT,
            object_name=normalized,
        )


def count_feature_object_name(feature_name: str | None) -> str | None:
    """Return the object-set name encoded by a CellProfiler Count_* feature."""
    parsed = CellProfilerMeasurementFeature.parse(feature_name)
    if parsed is None or parsed.kind is not CellProfilerMeasurementFeatureKind.OBJECT_COUNT:
        return None
    return parsed.object_name


def child_count_feature_child_name(feature_name: str | None) -> str | None:
    """Return the child object name encoded by Children_<object>_Count."""
    parsed = CellProfilerMeasurementFeature.parse(feature_name)
    if parsed is None or parsed.kind is not CellProfilerMeasurementFeatureKind.CHILD_COUNT:
        return None
    return parsed.object_name


__all__ = declared_public_names(
    globals(),
    extra_names=(
    "measurement_scalar_value_for_feature",
    "measurement_values_for_feature",
    "measurement_values_for_label_slices",
    ),
)
