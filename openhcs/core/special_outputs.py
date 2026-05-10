"""Runtime classification for callable-declared special outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactKind
from openhcs.processing.materialization import (
    CsvOptions,
    MaterializationSpec,
    ROIOptions,
    TiffStackOptions,
)


class SpecialOutputKindClassifier(ABC, metaclass=AutoRegisterMeta):
    """Nominal classifier for function-declared special output specs."""

    __registry_key__ = "classifier_name"
    __skip_if_no_key__ = True
    classifier_name: ClassVar[str | None] = None
    priority: ClassVar[int] = 100

    @classmethod
    def kind_for(cls, spec: object) -> ArtifactKind:
        for classifier_type in sorted(
            cls.__registry__.values(),
            key=lambda candidate: candidate.priority,
        ):
            kind = classifier_type().classify(spec)
            if kind is not None:
                return kind
        raise ValueError(f"Cannot infer artifact kind for special output {spec!r}.")

    @abstractmethod
    def classify(self, spec: object) -> ArtifactKind | None:
        """Return an artifact kind when this classifier owns the output spec."""


class SpatialGridSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Classify special outputs that carry grid geometry, not measurement rows."""

    classifier_name = "spatial_grid"
    priority = 5

    def classify(self, spec: object) -> ArtifactKind | None:
        normalized = normalize_special_output_name(special_output_name(spec))
        if normalized in {"grid_info", "grid_definition", "spatial_grid"}:
            return ArtifactKind.SPATIAL_GRID
        return None


class RoiSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Classify ROI materialization specs as object labels."""

    classifier_name = "roi"
    priority = 10

    def classify(self, spec: object) -> ArtifactKind | None:
        return kind_for_materialization_option(spec, ROIOptions, ArtifactKind.OBJECT_LABELS)


class CsvSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Classify CSV materialization specs as measurement tables."""

    classifier_name = "csv"
    priority = 20

    def classify(self, spec: object) -> ArtifactKind | None:
        return kind_for_materialization_option(spec, CsvOptions, ArtifactKind.MEASUREMENTS)


class TiffSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Classify TIFF materialization specs as images."""

    classifier_name = "tiff"
    priority = 30

    def classify(self, spec: object) -> ArtifactKind | None:
        return kind_for_materialization_option(spec, TiffStackOptions, ArtifactKind.IMAGE)


class NameSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Name-based classifier for legacy special_outputs without materialization."""

    classifier_name = "name"
    priority = 40

    def classify(self, spec: object) -> ArtifactKind | None:
        normalized = normalize_special_output_name(special_output_name(spec))
        if "label" in normalized or "labels" in normalized:
            return ArtifactKind.OBJECT_LABELS
        if "relationship" in normalized:
            return ArtifactKind.RELATIONSHIPS
        if "image" in normalized:
            return ArtifactKind.IMAGE
        return ArtifactKind.MEASUREMENTS


def kind_for_materialization_option(
    spec: object,
    option_type: type[Any],
    output_kind: ArtifactKind,
) -> ArtifactKind | None:
    """Return ``output_kind`` when ``spec`` declares a matching materializer."""
    materialization = special_output_materialization(spec)
    if materialization is None:
        return None
    if any(isinstance(option, option_type) for option in materialization.outputs):
        return output_kind
    return None


def special_output_name(spec: object) -> str:
    """Return the declared output name from a callable special-output spec."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        return spec[0]
    raise ValueError(f"Invalid special output declaration: {spec!r}.")


def special_output_materialization(spec: object) -> MaterializationSpec | None:
    """Return the declared materialization from a callable special-output spec."""
    if isinstance(spec, tuple) and len(spec) == 2:
        materialization = spec[1]
        if materialization is None:
            return None
        if not isinstance(materialization, MaterializationSpec):
            raise TypeError(
                "special_outputs materialization must be MaterializationSpec "
                f"or None, got {type(materialization).__name__}."
            )
        return materialization
    return None


def normalize_special_output_name(name: str) -> str:
    """Normalize an output declaration name for semantic classification."""
    import re

    without_parentheses = re.sub(r"\([^)]*\)", "", name)
    without_questions = without_parentheses.replace("?", "")
    words = re.sub(r"[^\w\s]", " ", without_questions).lower().split()
    return "_".join(words)
