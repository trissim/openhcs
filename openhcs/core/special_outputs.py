"""Runtime classification for callable-declared special outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.special_output_declarations import SpecialOutputDeclaration
from openhcs.processing.materialization import (
    CsvOptions,
    MaterializationSpec,
    ROIOptions,
    TiffStackOptions,
)

SpecialOutputMaterializationOptionType: TypeAlias = (
    type[CsvOptions] | type[ROIOptions] | type[TiffStackOptions]
)

SPATIAL_GRID_SPECIAL_OUTPUT_NAMES = frozenset(
    {
        "grid_info",
        "grid_definition",
        "spatial_grid",
    }
)


class SpecialOutputKindClassifier(ABC, metaclass=AutoRegisterMeta):
    """Nominal classifier for function-declared special output specs."""

    __registry_key__ = "classifier_name"
    __skip_if_no_key__ = True
    classifier_name: ClassVar[str | None] = None

    @classmethod
    def kind_for(cls, spec: SpecialOutputDeclaration) -> ArtifactKind:
        matches: list[tuple[type[SpecialOutputKindClassifier], ArtifactKind]] = []
        for classifier_type in cls.classifier_types_by_mro():
            kind = classifier_type().classify(spec)
            if kind is not None:
                matches.append((classifier_type, kind))
        if not matches:
            raise ValueError(f"Cannot infer artifact kind for special output {spec!r}.")
        kind = matches[0][1]
        conflicting = tuple(
            classifier_type.__name__
            for classifier_type, candidate_kind in matches
            if candidate_kind is not kind
        )
        if conflicting:
            raise ValueError(
                f"Ambiguous artifact kind for special output {spec!r}: "
                f"{', '.join(conflicting)}."
            )
        return kind

    @classmethod
    def classifier_types_by_mro(
        cls,
    ) -> tuple[type["SpecialOutputKindClassifier"], ...]:
        registered = set(cls.__registry__.values())
        ordered: list[type[SpecialOutputKindClassifier]] = []
        seen: set[type[SpecialOutputKindClassifier]] = set()

        def visit(owner: type[SpecialOutputKindClassifier]) -> None:
            for child in owner.__subclasses__():
                visit(child)
            if owner in registered and owner not in seen:
                ordered.append(owner)
                seen.add(owner)

        visit(cls)
        return tuple(ordered)

    @abstractmethod
    def classify(self, spec: SpecialOutputDeclaration) -> ArtifactKind | None:
        """Return an artifact kind when this classifier owns the output spec."""


class SpatialGridSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Classify special outputs that carry grid geometry, not measurement rows."""

    classifier_name = "spatial_grid"

    def classify(self, spec: SpecialOutputDeclaration) -> ArtifactKind | None:
        normalized = normalize_special_output_name(special_output_name(spec))
        if normalized in SPATIAL_GRID_SPECIAL_OUTPUT_NAMES:
            return ArtifactKind.SPATIAL_GRID
        return None


class RoiSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Classify ROI materialization specs as object labels."""

    classifier_name = "roi"

    def classify(self, spec: SpecialOutputDeclaration) -> ArtifactKind | None:
        return kind_for_materialization_option(
            spec,
            ROIOptions,
            ArtifactKind.OBJECT_LABELS,
        )


class CsvSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Classify CSV materialization specs as measurement tables."""

    classifier_name = "csv"

    def classify(self, spec: SpecialOutputDeclaration) -> ArtifactKind | None:
        normalized = normalize_special_output_name(special_output_name(spec))
        if normalized in SPATIAL_GRID_SPECIAL_OUTPUT_NAMES:
            return None
        return kind_for_materialization_option(
            spec,
            CsvOptions,
            ArtifactKind.MEASUREMENTS,
        )


class TiffSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Classify TIFF materialization specs as images."""

    classifier_name = "tiff"

    def classify(self, spec: SpecialOutputDeclaration) -> ArtifactKind | None:
        return kind_for_materialization_option(spec, TiffStackOptions, ArtifactKind.IMAGE)


class NameSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Name-based classifier for legacy special_outputs without materialization."""

    classifier_name = "name"

    def classify(self, spec: SpecialOutputDeclaration) -> ArtifactKind | None:
        if special_output_materialization(spec) is not None:
            return None
        normalized = normalize_special_output_name(special_output_name(spec))
        if normalized in SPATIAL_GRID_SPECIAL_OUTPUT_NAMES:
            return None
        if "label" in normalized or "labels" in normalized:
            return ArtifactKind.OBJECT_LABELS
        if "relationship" in normalized:
            return ArtifactKind.RELATIONSHIPS
        if "image" in normalized:
            return ArtifactKind.IMAGE
        return ArtifactKind.MEASUREMENTS


def kind_for_materialization_option(
    spec: SpecialOutputDeclaration,
    option_type: SpecialOutputMaterializationOptionType,
    output_kind: ArtifactKind,
) -> ArtifactKind | None:
    """Return ``output_kind`` when ``spec`` declares a matching materializer."""
    materialization = special_output_materialization(spec)
    if materialization is None:
        return None
    if any(isinstance(option, option_type) for option in materialization.outputs):
        return output_kind
    return None


def special_output_name(spec: SpecialOutputDeclaration) -> str:
    """Return the declared output name from a callable special-output spec."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        return spec[0]
    raise ValueError(f"Invalid special output declaration: {spec!r}.")


def special_output_materialization(
    spec: SpecialOutputDeclaration,
) -> MaterializationSpec | None:
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
