"""Typed pipeline-level image schema for setup-derived source semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents
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


@dataclass(frozen=True, slots=True)
class ImageAssignment:
    """One pipeline-level semantic image alias assignment."""

    alias: str
    image_type: str
    selector: SourceSelector
    origin: SourceBindingOrigin

    def __post_init__(self) -> None:
        normalized_alias = self.alias.strip()
        if not normalized_alias:
            raise ValueError("ImageAssignment.alias cannot be empty.")
        object.__setattr__(self, "alias", normalized_alias)
        object.__setattr__(self, "image_type", self.image_type.strip())
        if not isinstance(self.selector, SourceSelector):
            raise TypeError(
                "ImageAssignment.selector must be SourceSelector, "
                f"got {type(self.selector).__name__}."
            )
        if not isinstance(self.origin, SourceBindingOrigin):
            raise TypeError(
                "ImageAssignment.origin must be SourceBindingOrigin, "
                f"got {type(self.origin).__name__}."
            )

    def to_binding(self) -> NamedSourceBinding:
        return NamedSourceBinding(
            alias=self.alias,
            selector=self.selector,
            origin=self.origin,
        )


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
class CellProfilerImageSchema:
    """Pipeline-level image schema lowered from setup modules."""

    images_rule: ImagesRule | None = None
    metadata_rules: tuple[MetadataExtractionRule, ...] = ()
    assignments_by_alias: Mapping[str, ImageAssignment] = MappingProxyType({})
    match_plan: SourceBindingMatchPlan | None = None
    grouping: GroupingPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata_rules", tuple(self.metadata_rules))
        object.__setattr__(
            self,
            "assignments_by_alias",
            MappingProxyType(dict(self.assignments_by_alias)),
        )
        for alias, assignment in self.assignments_by_alias.items():
            if alias != assignment.alias:
                raise ValueError(
                    f"CellProfilerImageSchema alias key {alias!r} does not match "
                    f"assignment alias {assignment.alias!r}."
                )

    @classmethod
    def empty(cls) -> "CellProfilerImageSchema":
        return cls()

    @property
    def is_empty(self) -> bool:
        return (
            self.images_rule is None
            and not self.metadata_rules
            and not self.assignments_by_alias
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


__all__ = [
    "CellProfilerImageSchema",
    "GroupingPlan",
    "ImageAssignment",
    "ImagesRule",
    "LegacyImageAssignmentStrategy",
    "OrigColorLegacyImageAssignmentStrategy",
]
