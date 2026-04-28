"""Typed pipeline-level image schema for setup-derived source semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.source_bindings import (
    ComponentSelector,
    MetadataExtractionRule,
    NamedSourceBinding,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceSelector,
)


@dataclass(frozen=True, slots=True)
class ImagesRule:
    """One setup-module source universe rule."""

    filtering_mode: str
    criteria: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceAssignmentBase(ABC):
    """Shared source-assignment identity and selector contract."""

    alias: str
    selector: SourceSelector
    origin: SourceBindingOrigin

    def __post_init__(self) -> None:
        normalized_alias = self.alias.strip()
        if not normalized_alias:
            raise ValueError(f"{type(self).__name__}.alias cannot be empty.")
        object.__setattr__(self, "alias", normalized_alias)
        if not isinstance(self.selector, SourceSelector):
            raise TypeError(
                f"{type(self).__name__}.selector must be SourceSelector, "
                f"got {type(self.selector).__name__}."
            )
        if not isinstance(self.origin, SourceBindingOrigin):
            raise TypeError(
                f"{type(self).__name__}.origin must be SourceBindingOrigin, "
                f"got {type(self.origin).__name__}."
            )

    @property
    @abstractmethod
    def artifact_kind(self) -> ArtifactKind:
        """Artifact kind bound by this source assignment."""

    def to_binding(self) -> NamedSourceBinding:
        return NamedSourceBinding(
            alias=self.alias,
            artifact_kind=self.artifact_kind,
            selector=self.selector,
            origin=self.origin,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageAssignment(SourceAssignmentBase):
    """One pipeline-level semantic image alias assignment."""

    image_type: str

    def __post_init__(self) -> None:
        SourceAssignmentBase.__post_init__(self)
        object.__setattr__(self, "image_type", self.image_type.strip())

    @property
    def artifact_kind(self) -> ArtifactKind:
        return ArtifactKind.IMAGE


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceArtifactAssignment(SourceAssignmentBase):
    """One pipeline-start or step-input source artifact declaration."""

    kind: ArtifactKind
    payload_type: str = ""

    def __post_init__(self) -> None:
        SourceAssignmentBase.__post_init__(self)
        if not isinstance(self.kind, ArtifactKind):
            raise TypeError(
                "SourceArtifactAssignment.kind must be ArtifactKind, "
                f"got {type(self.kind).__name__}."
            )
        object.__setattr__(self, "payload_type", self.payload_type.strip())

    @property
    def artifact_kind(self) -> ArtifactKind:
        return self.kind

    @classmethod
    def from_image_assignment(
        cls,
        assignment: ImageAssignment,
    ) -> "SourceArtifactAssignment":
        return cls(
            alias=assignment.alias,
            kind=ArtifactKind.IMAGE,
            selector=assignment.selector,
            origin=assignment.origin,
            payload_type=assignment.image_type,
        )


class ImageTypeSourceRole(ABC, metaclass=AutoRegisterMeta):
    """Nominal role for CellProfiler image-type source semantics."""

    __registry_key__ = "image_type_key"
    __skip_if_no_key__ = True
    image_type_key: ClassVar[str | None] = None
    PARTICIPATES_IN_IMAGE_STACK: ClassVar[bool]

    @classmethod
    def for_image_type(cls, image_type: str) -> "ImageTypeSourceRole":
        key = image_type_source_role_key(image_type)
        role_type = cls.__registry__.get(key)
        if role_type is None:
            raise ValueError(
                f"Unsupported CellProfiler source image type {image_type!r}."
            )
        return role_type()

    @property
    def participates_in_image_stack(self) -> bool:
        """Whether this image type should become an OpenHCS channel."""

        return type(self).PARTICIPATES_IN_IMAGE_STACK


class ImageStackSourceRole(ImageTypeSourceRole):
    """CellProfiler image type that projects into the OpenHCS channel stack."""

    PARTICIPATES_IN_IMAGE_STACK = True


class SourceArtifactImageTypeSourceRole(ImageTypeSourceRole):
    """CellProfiler image type that remains an external source artifact."""

    PARTICIPATES_IN_IMAGE_STACK = False


@dataclass(frozen=True, slots=True)
class ImageTypeSourceRoleSpec:
    """Typed declaration for one CellProfiler image-type role class."""

    class_name: str
    image_type_key: str
    base_type: type[ImageTypeSourceRole]

    def declare(self) -> type[ImageTypeSourceRole]:
        return type(
            self.class_name,
            (self.base_type,),
            {
                "__module__": __name__,
                "image_type_key": self.image_type_key,
            },
        )


for _image_type_role_spec in (
    ImageTypeSourceRoleSpec(
        "GrayscaleImageTypeSourceRole",
        "grayscale image",
        ImageStackSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "ColorImageTypeSourceRole",
        "color image",
        ImageStackSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "BinaryImageTypeSourceRole",
        "binary image",
        ImageStackSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "BinaryMaskImageTypeSourceRole",
        "binary mask",
        ImageStackSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "MaskImageTypeSourceRole",
        "mask",
        ImageStackSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "IlluminationFunctionImageTypeSourceRole",
        "illumination function",
        SourceArtifactImageTypeSourceRole,
    ),
):
    globals()[_image_type_role_spec.class_name] = _image_type_role_spec.declare()


def image_type_participates_in_image_stack(image_type: str) -> bool:
    """Return whether a CellProfiler source image type is a native stack channel."""

    return ImageTypeSourceRole.for_image_type(image_type).participates_in_image_stack


def image_type_source_role_key(image_type: str) -> str:
    """Normalize CellProfiler image-type labels for role lookup."""

    return image_type.strip().lower()


@dataclass(frozen=True, slots=True)
class GroupingPlan:
    """Typed metadata grouping declaration for one pipeline image schema."""

    metadata_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata_fields",
            tuple(field.strip() for field in self.metadata_fields if field.strip()),
        )


@dataclass(frozen=True, slots=True)
class ImportedMetadataJoin:
    """One join key between image metadata and an imported metadata table."""

    image_metadata_field: str
    imported_metadata_field: str

    def __post_init__(self) -> None:
        if not self.image_metadata_field.strip():
            raise ValueError(
                "ImportedMetadataJoin.image_metadata_field cannot be empty."
            )
        if not self.imported_metadata_field.strip():
            raise ValueError(
                "ImportedMetadataJoin.imported_metadata_field cannot be empty."
            )
        object.__setattr__(
            self,
            "image_metadata_field",
            self.image_metadata_field.strip(),
        )
        object.__setattr__(
            self,
            "imported_metadata_field",
            self.imported_metadata_field.strip(),
        )


@dataclass(frozen=True, slots=True)
class ImportedMetadataTable:
    """Pipeline-level metadata imported from an external CellProfiler table."""

    location: str | None = None
    joins: tuple[ImportedMetadataJoin, ...] = ()

    def __post_init__(self) -> None:
        normalized_location = (
            None if self.location is None else self.location.strip() or None
        )
        object.__setattr__(self, "location", normalized_location)
        object.__setattr__(self, "joins", tuple(self.joins))
        for join in self.joins:
            if not isinstance(join, ImportedMetadataJoin):
                raise TypeError(
                    "ImportedMetadataTable.joins must contain "
                    "ImportedMetadataJoin values, got "
                    f"{type(join).__name__}."
                )


@dataclass(frozen=True, slots=True)
class CellProfilerImageSchema:
    """Pipeline-level image schema lowered from setup modules."""

    images_rule: ImagesRule | None = None
    metadata_rules: tuple[MetadataExtractionRule, ...] = ()
    imported_metadata_tables: tuple[ImportedMetadataTable, ...] = ()
    assignments_by_alias: Mapping[str, ImageAssignment] = MappingProxyType({})
    source_artifacts_by_alias: Mapping[str, SourceArtifactAssignment] = (
        MappingProxyType({})
    )
    match_plan: SourceBindingMatchPlan | None = None
    grouping: GroupingPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata_rules", tuple(self.metadata_rules))
        object.__setattr__(
            self,
            "imported_metadata_tables",
            tuple(self.imported_metadata_tables),
        )
        object.__setattr__(
            self,
            "assignments_by_alias",
            MappingProxyType(dict(self.assignments_by_alias)),
        )
        object.__setattr__(
            self,
            "source_artifacts_by_alias",
            MappingProxyType(dict(self.source_artifacts_by_alias)),
        )
        for table in self.imported_metadata_tables:
            if not isinstance(table, ImportedMetadataTable):
                raise TypeError(
                    "CellProfilerImageSchema.imported_metadata_tables must "
                    "contain ImportedMetadataTable values, got "
                    f"{type(table).__name__}."
                )
        for alias, assignment in self.assignments_by_alias.items():
            if alias != assignment.alias:
                raise ValueError(
                    f"CellProfilerImageSchema alias key {alias!r} does not match "
                    f"assignment alias {assignment.alias!r}."
                )
        for alias, assignment in self.source_artifacts_by_alias.items():
            if not isinstance(assignment, SourceArtifactAssignment):
                raise TypeError(
                    "CellProfilerImageSchema.source_artifacts_by_alias values "
                    "must be SourceArtifactAssignment, got "
                    f"{type(assignment).__name__}."
                )
            if alias != assignment.alias:
                raise ValueError(
                    f"CellProfilerImageSchema source-artifact key {alias!r} "
                    f"does not match assignment alias {assignment.alias!r}."
                )

    @classmethod
    def empty(cls) -> "CellProfilerImageSchema":
        return cls()

    @property
    def is_empty(self) -> bool:
        return (
            self.images_rule is None
            and not self.metadata_rules
            and not self.imported_metadata_tables
            and not self.assignments_by_alias
            and not self.source_artifacts_by_alias
            and self.match_plan is None
            and self.grouping is None
        )

    def assignment_for_alias(self, alias: str) -> ImageAssignment | None:
        return self.assignments_by_alias.get(alias)

    def resolved_assignment_for_alias(self, alias: str) -> ImageAssignment | None:
        assignment = self.assignment_for_alias(alias)
        if assignment is not None:
            return assignment
        return LegacyImageAssignmentStrategy.resolve(alias)

    def source_artifact_for_alias(
        self,
        alias: str,
    ) -> SourceArtifactAssignment | None:
        artifact_assignment = self.source_artifacts_by_alias.get(alias)
        if artifact_assignment is not None:
            return artifact_assignment
        image_assignment = self.resolved_assignment_for_alias(alias)
        if image_assignment is not None:
            return SourceArtifactAssignment.from_image_assignment(image_assignment)
        return None

    def resolved_source_artifact_for_alias(
        self,
        alias: str,
        kind: ArtifactKind,
    ) -> SourceArtifactAssignment | None:
        artifact_assignment = self.source_artifact_for_alias(alias)
        if artifact_assignment is None:
            return None
        if artifact_assignment.kind is not kind:
            raise ValueError(
                f"CellProfiler source artifact {alias!r} is declared as "
                f"{artifact_assignment.kind.value}, not {kind.value}."
            )
        return artifact_assignment


class LegacyImageAssignmentStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal fallback family for legacy semantic image aliases."""

    __registry_key__ = "strategy_name"
    __skip_if_no_key__ = True
    strategy_name: ClassVar[str | None] = None

    @classmethod
    def resolve(cls, alias: str) -> ImageAssignment | None:
        for strategy_type in cls.__registry__.values():
            strategy = strategy_type()
            if strategy.matches(alias):
                return strategy.assignment(alias)
        return None

    @abstractmethod
    def matches(self, alias: str) -> bool:
        """Whether this strategy applies to the alias."""

    @abstractmethod
    def assignment(self, alias: str) -> ImageAssignment:
        """Return the typed fallback assignment for the alias."""


class OrigColorLegacyImageAssignmentStrategy(LegacyImageAssignmentStrategy):
    """Map legacy Orig<Color> aliases onto native channel selectors."""

    strategy_name = "orig_color"
    _CHANNELS_BY_COLOR = MappingProxyType(
        {
            "blue": "1",
            "green": "2",
            "red": "3",
        }
    )

    def matches(self, alias: str) -> bool:
        normalized = alias.strip().lower()
        return normalized.startswith("orig") and normalized[4:] in self._CHANNELS_BY_COLOR

    def assignment(self, alias: str) -> ImageAssignment:
        normalized = alias.strip().lower()
        color = normalized[4:]
        return ImageAssignment(
            alias=alias,
            image_type="Grayscale image",
            selector=SourceSelector(
                components=(
                    ComponentSelector(
                        AllComponents.CHANNEL,
                        self._CHANNELS_BY_COLOR[color],
                    ),
                ),
            ),
            origin=SourceBindingOrigin.STEP_INPUT,
        )


def _is_public_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name for name, value in globals().items() if _is_public_export(name, value)
)
