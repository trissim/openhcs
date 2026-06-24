"""Typed pipeline-level image schema for setup-derived source semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents
from openhcs.core.component_set import ComponentSet
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.source_bindings import (
    ComponentSelector,
    MetadataExtractionRule,
    NamedSourceBinding,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceAssignmentBase,
    SourceFilterClause,
    SourceSelector,
    normalize_source_binding_values,
)


SOURCE_IMAGE_TYPE_METADATA_FIELD = "OpenHCSImageType"
SOURCE_SCHEMA_ORDERED_IMAGE_SET_COMPONENTS = (AllComponents.SITE,)


@dataclass(frozen=True, slots=True)
class ImagePlaneSource:
    """One explicit CellProfiler image-plane source URI embedded in a pipeline."""

    uri: str
    series: str | None = None
    index: str | None = None
    channel: str | None = None

    def __post_init__(self) -> None:
        normalized_uri = self.uri.strip()
        if not normalized_uri:
            raise ValueError("ImagePlaneSource.uri cannot be empty.")
        object.__setattr__(self, "uri", normalized_uri)
        object.__setattr__(self, "series", _normalized_optional_text(self.series))
        object.__setattr__(self, "index", _normalized_optional_text(self.index))
        object.__setattr__(self, "channel", _normalized_optional_text(self.channel))


@dataclass(frozen=True, slots=True)
class ImagesRule:
    """One setup-module source universe rule."""

    filters: tuple[SourceFilterClause, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "filters",
            normalize_source_binding_values(
                "ImagesRule.filters",
                self.filters,
                SourceFilterClause,
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageAssignment(SourceAssignmentBase):
    """One pipeline-level semantic image alias assignment."""

    assignment_kind = "image"
    image_type: str

    def __post_init__(self) -> None:
        SourceAssignmentBase.__post_init__(self)
        object.__setattr__(self, "image_type", self.image_type.strip())

    @property
    def artifact_kind(self) -> ArtifactKind:
        return ArtifactKind.IMAGE

    @property
    def participates_in_image_stack(self) -> bool:
        return ImageTypeSourceRole.for_image_type(
            self.image_type
        ).participates_in_image_stack


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceArtifactAssignment(SourceAssignmentBase):
    """One pipeline-start or step-input source artifact declaration."""

    assignment_kind = "source_artifact"
    artifact_kind: ArtifactKind
    payload_type: str = ""

    def __post_init__(self) -> None:
        SourceAssignmentBase.__post_init__(self)
        if not isinstance(self.artifact_kind, ArtifactKind):
            raise TypeError(
                "SourceArtifactAssignment.artifact_kind must be ArtifactKind, "
                f"got {type(self.artifact_kind).__name__}."
            )
        object.__setattr__(self, "payload_type", self.payload_type.strip())

    @classmethod
    def from_image_assignment(
        cls,
        assignment: ImageAssignment,
    ) -> "SourceArtifactAssignment":
        return cls(
            alias=assignment.alias,
            artifact_kind=ArtifactKind.IMAGE,
            selector=assignment.selector,
            origin=assignment.origin,
            payload_type=assignment.image_type,
        )

    @property
    def participates_in_image_stack(self) -> bool:
        return False


class ImageTypeSourceRole(ABC, metaclass=AutoRegisterMeta):
    """Nominal role for pipeline image-type source semantics."""

    __registry_key__ = "image_type_key"
    __skip_if_no_key__ = True
    image_type_key: ClassVar[str | None] = None
    PARTICIPATES_IN_IMAGE_STACK: ClassVar[bool]
    ARTIFACT_KIND: ClassVar[ArtifactKind] = ArtifactKind.IMAGE
    LOAD_AS_MONOCHROME: ClassVar[bool] = False
    MATERIALIZE_SOURCE_MASK: ClassVar[bool] = False

    @classmethod
    def for_image_type(cls, image_type: str) -> "ImageTypeSourceRole":
        key = image_type_source_role_key(image_type)
        role_type = cls.__registry__.get(key)
        if role_type is None:
            raise ValueError(
                f"Unsupported pipeline source image type {image_type!r}."
            )
        return role_type()

    @property
    def participates_in_image_stack(self) -> bool:
        """Whether this image type should become an OpenHCS channel."""

        return type(self).PARTICIPATES_IN_IMAGE_STACK

    @property
    def artifact_kind(self) -> ArtifactKind:
        """Artifact kind represented by this source image type."""

        return type(self).ARTIFACT_KIND

    @property
    def load_as_monochrome(self) -> bool:
        """Whether source pixels must be collapsed to CellProfiler monochrome."""

        return type(self).LOAD_AS_MONOCHROME

    @property
    def materialize_source_mask(self) -> bool:
        """Whether source pixels require an explicit per-pixel validity mask."""

        return type(self).MATERIALIZE_SOURCE_MASK


@dataclass(frozen=True, slots=True)
class ImageTypeSourceRoleClassSpec:
    """Typed declaration for one source-role class in the nominal hierarchy."""

    class_name: str
    base_type: type[ImageTypeSourceRole]
    participates_in_image_stack: bool
    artifact_kind: ArtifactKind = ArtifactKind.IMAGE
    load_as_monochrome: bool = False
    materialize_source_mask: bool = False

    def declare(self) -> type[ImageTypeSourceRole]:
        return type(
            self.class_name,
            (self.base_type,),
            {
                "__module__": __name__,
                "PARTICIPATES_IN_IMAGE_STACK": self.participates_in_image_stack,
                "ARTIFACT_KIND": self.artifact_kind,
                "LOAD_AS_MONOCHROME": self.load_as_monochrome,
                "MATERIALIZE_SOURCE_MASK": self.materialize_source_mask,
            },
        )


ImageStackSourceRole = ImageTypeSourceRoleClassSpec(
    "ImageStackSourceRole",
    ImageTypeSourceRole,
    participates_in_image_stack=True,
    materialize_source_mask=True,
).declare()
MonochromeImageStackSourceRole = ImageTypeSourceRoleClassSpec(
    "MonochromeImageStackSourceRole",
    ImageStackSourceRole,
    participates_in_image_stack=True,
    load_as_monochrome=True,
    materialize_source_mask=True,
).declare()
SourceArtifactImageTypeSourceRole = ImageTypeSourceRoleClassSpec(
    "SourceArtifactImageTypeSourceRole",
    ImageTypeSourceRole,
    participates_in_image_stack=False,
).declare()
ObjectLabelsImageTypeSourceRole = ImageTypeSourceRoleClassSpec(
    "ObjectLabelsImageTypeSourceRole",
    SourceArtifactImageTypeSourceRole,
    participates_in_image_stack=False,
    artifact_kind=ArtifactKind.OBJECT_LABELS,
).declare()


@dataclass(frozen=True, slots=True)
class ImageTypeSourceRoleSpec:
    """Typed declaration for one concrete pipeline image-type role class."""

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
        MonochromeImageStackSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "ColorImageTypeSourceRole",
        "color image",
        ImageStackSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "BinaryImageTypeSourceRole",
        "binary image",
        MonochromeImageStackSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "BinaryMaskImageTypeSourceRole",
        "binary mask",
        MonochromeImageStackSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "MaskImageTypeSourceRole",
        "mask",
        MonochromeImageStackSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "IlluminationFunctionImageTypeSourceRole",
        "illumination function",
        SourceArtifactImageTypeSourceRole,
    ),
    ImageTypeSourceRoleSpec(
        "ObjectsImageTypeSourceRole",
        "objects",
        ObjectLabelsImageTypeSourceRole,
    ),
):
    globals()[_image_type_role_spec.class_name] = _image_type_role_spec.declare()


def image_type_participates_in_image_stack(image_type: str) -> bool:
    """Return whether a source image type is a native stack channel."""

    return ImageTypeSourceRole.for_image_type(image_type).participates_in_image_stack


def image_type_artifact_kind(image_type: str) -> ArtifactKind:
    """Return the artifact kind represented by a source image type."""

    return ImageTypeSourceRole.for_image_type(image_type).artifact_kind


def image_type_loads_as_monochrome(image_type: str) -> bool:
    """Return whether source loading should mirror CellProfiler MonochromeImage."""

    return ImageTypeSourceRole.for_image_type(image_type).load_as_monochrome


def image_type_materializes_source_mask(image_type: str) -> bool:
    """Return whether source loading should create an explicit validity mask."""

    return ImageTypeSourceRole.for_image_type(image_type).materialize_source_mask


def image_type_source_role_key(image_type: str) -> str:
    """Normalize image-type labels for role lookup."""

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
class SourceImageStackPlan:
    """Pipeline-level source components that form one logical source image."""

    components: tuple[AllComponents, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "components",
            ComponentSet.collect(self.components).as_tuple(),
        )

    @property
    def is_empty(self) -> bool:
        return not self.components

    def merge(self, other: "SourceImageStackPlan") -> "SourceImageStackPlan":
        return SourceImageStackPlan((*self.components, *other.components))


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
    """Pipeline-level metadata imported from an external table."""

    location: str | None = None
    joins: tuple[ImportedMetadataJoin, ...] = ()

    def __post_init__(self) -> None:
        normalized_location = (
            None if self.location is None else self.location.strip() or None
        )
        object.__setattr__(self, "location", normalized_location)
        object.__setattr__(
            self,
            "joins",
            normalize_source_binding_values(
                "ImportedMetadataTable.joins",
                self.joins,
                ImportedMetadataJoin,
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelineImageSchema:
    """Pipeline-level image schema lowered from setup modules."""

    images_rule: ImagesRule | None = None
    image_plane_sources: tuple[ImagePlaneSource, ...] = ()
    source_image_stack: SourceImageStackPlan = field(default_factory=SourceImageStackPlan)
    metadata_rules: tuple[MetadataExtractionRule, ...] = ()
    imported_metadata_tables: tuple[ImportedMetadataTable, ...] = ()
    assignments_by_alias: Mapping[str, ImageAssignment] = field(
        default_factory=lambda: MappingProxyType({})
    )
    source_artifacts_by_alias: Mapping[str, SourceArtifactAssignment] = field(
        default_factory=lambda: MappingProxyType({})
    )
    match_plan: SourceBindingMatchPlan | None = None
    grouping: GroupingPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "image_plane_sources",
            normalize_source_binding_values(
                "PipelineImageSchema.image_plane_sources",
                self.image_plane_sources,
                ImagePlaneSource,
            ),
        )
        if not isinstance(self.source_image_stack, SourceImageStackPlan):
            raise TypeError(
                "PipelineImageSchema.source_image_stack must be "
                "SourceImageStackPlan, got "
                f"{type(self.source_image_stack).__name__}."
            )
        object.__setattr__(
            self,
            "metadata_rules",
            normalize_source_binding_values(
                "PipelineImageSchema.metadata_rules",
                self.metadata_rules,
                MetadataExtractionRule,
            ),
        )
        object.__setattr__(
            self,
            "imported_metadata_tables",
            normalize_source_binding_values(
                "PipelineImageSchema.imported_metadata_tables",
                self.imported_metadata_tables,
                ImportedMetadataTable,
            ),
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
        for alias, assignment in self.assignments_by_alias.items():
            if alias != assignment.alias:
                raise ValueError(
                    f"PipelineImageSchema alias key {alias!r} does not match "
                    f"assignment alias {assignment.alias!r}."
                )
        for alias, assignment in self.source_artifacts_by_alias.items():
            if not isinstance(assignment, SourceArtifactAssignment):
                raise TypeError(
                    "PipelineImageSchema.source_artifacts_by_alias values "
                    "must be SourceArtifactAssignment, got "
                    f"{type(assignment).__name__}."
                )
            if alias != assignment.alias:
                raise ValueError(
                    f"PipelineImageSchema source-artifact key {alias!r} "
                    f"does not match assignment alias {assignment.alias!r}."
                )

    @classmethod
    def empty(cls) -> "PipelineImageSchema":
        return cls()

    @property
    def is_empty(self) -> bool:
        return (
            self.images_rule is None
            and not self.image_plane_sources
            and self.source_image_stack.is_empty
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
        if (
            image_assignment is not None
            and not image_assignment.participates_in_image_stack
        ):
            return SourceArtifactAssignment.from_image_assignment(image_assignment)
        return None

    def source_assignment_for_alias(
        self,
        alias: str,
        kind: ArtifactKind,
    ) -> SourceAssignmentBase | None:
        artifact_assignment = self.source_artifact_for_alias(alias)
        if artifact_assignment is not None:
            if artifact_assignment.artifact_kind is not kind:
                raise ValueError(
                    f"Pipeline source artifact {alias!r} is declared as "
                    f"{artifact_assignment.artifact_kind.value}, not {kind.value}."
                )
            return artifact_assignment
        if kind is ArtifactKind.IMAGE:
            return self.resolved_assignment_for_alias(alias)
        return None

    @property
    def source_stack_components(self) -> tuple[AllComponents, ...]:
        """Return source components that must be kept inside one source image."""
        return self.source_image_stack.components

    @property
    def loaded_image_aliases(self) -> tuple[str, ...]:
        """Return aliases for source images loaded into the image set."""
        return tuple(
            assignment.alias
            for assignment in self.assignments_by_alias.values()
            if assignment.participates_in_image_stack
        )

    @property
    def measurement_source_names(self) -> tuple[str, ...]:
        """Return source names that may appear in measurement feature names."""
        names: set[str] = set()
        for assignment in self.assignments_by_alias.values():
            names.update(assignment.measurement_source_names)
        for assignment in self.source_artifacts_by_alias.values():
            names.update(assignment.measurement_source_names)
        return tuple(sorted(names, key=lambda value: value.lower()))

    def resolved_source_artifact_for_alias(
        self,
        alias: str,
        kind: ArtifactKind,
    ) -> SourceArtifactAssignment | None:
        artifact_assignment = self.source_artifact_for_alias(alias)
        if artifact_assignment is None:
            return None
        if artifact_assignment.artifact_kind is not kind:
            raise ValueError(
                f"Pipeline source artifact {alias!r} is declared as "
                f"{artifact_assignment.artifact_kind.value}, not {kind.value}."
            )
        return artifact_assignment


class PipelineImageSchemaBuilder:
    """Mutable accumulator for pipeline-level source schema declarations."""

    def __init__(self) -> None:
        self.images_rule: ImagesRule | None = None
        self.image_plane_sources: list[ImagePlaneSource] = []
        self.source_image_stack = SourceImageStackPlan()
        self.metadata_rules: list[MetadataExtractionRule] = []
        self.imported_metadata_tables: list[ImportedMetadataTable] = []
        self.assignments_by_alias: dict[str, ImageAssignment] = {}
        self.source_artifacts_by_alias: dict[str, SourceArtifactAssignment] = {}
        self.match_plan: SourceBindingMatchPlan | None = None
        self.grouping: GroupingPlan | None = None

    def build(self) -> PipelineImageSchema:
        return PipelineImageSchema(
            images_rule=self.images_rule,
            image_plane_sources=tuple(self.image_plane_sources),
            source_image_stack=self.source_image_stack,
            metadata_rules=tuple(self.metadata_rules),
            imported_metadata_tables=tuple(self.imported_metadata_tables),
            assignments_by_alias=MappingProxyType(dict(self.assignments_by_alias)),
            source_artifacts_by_alias=MappingProxyType(
                dict(self.source_artifacts_by_alias)
            ),
            match_plan=self.match_plan,
            grouping=self.grouping,
        )

    def add_metadata_rule(self, rule: MetadataExtractionRule) -> None:
        if rule not in self.metadata_rules:
            self.metadata_rules.append(rule)

    def add_image_plane_source(self, source: ImagePlaneSource) -> None:
        if source not in self.image_plane_sources:
            self.image_plane_sources.append(source)

    def declare_source_image_stack(
        self,
        stack_plan: SourceImageStackPlan,
    ) -> None:
        self.source_image_stack = self.source_image_stack.merge(stack_plan)

    def add_imported_metadata_table(self, table: ImportedMetadataTable) -> None:
        self.imported_metadata_tables.append(table)

    def declare_assignment(self, assignment: ImageAssignment) -> None:
        existing = self.assignments_by_alias.get(assignment.alias)
        if existing is not None and existing != assignment:
            raise ValueError(
                f"Pipeline image alias {assignment.alias!r} is already declared "
                "with different setup semantics."
            )
        if assignment.alias in self.source_artifacts_by_alias:
            raise ValueError(
                f"Pipeline alias {assignment.alias!r} is already declared as "
                "a non-image source artifact."
            )
        self.assignments_by_alias[assignment.alias] = assignment

    def declare_source_artifact(
        self,
        assignment: SourceArtifactAssignment,
    ) -> None:
        existing = self.source_artifacts_by_alias.get(assignment.alias)
        if existing is not None and existing != assignment:
            raise ValueError(
                f"Pipeline source artifact {assignment.alias!r} is already "
                "declared with different setup semantics."
            )
        if assignment.alias in self.assignments_by_alias:
            raise ValueError(
                f"Pipeline alias {assignment.alias!r} is already declared as "
                "an image assignment."
            )
        self.source_artifacts_by_alias[assignment.alias] = assignment

    def declare_match_plan(self, match_plan: SourceBindingMatchPlan) -> None:
        if self.match_plan is not None and self.match_plan != match_plan:
            raise ValueError(
                "Pipeline image schema already declared a different image-set "
                "match plan."
            )
        self.match_plan = match_plan

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


def _normalized_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _is_public_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name for name, value in globals().items() if _is_public_export(name, value)
)
