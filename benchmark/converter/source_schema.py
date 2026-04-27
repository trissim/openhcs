"""Typed CellProfiler setup-module schema lowered onto OpenHCS source bindings."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents
from openhcs.core.source_bindings import (
    ComponentSelector,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingOrigin,
    SourceSelector,
)

from .parser import ModuleBlock, ModuleSetting

_METADATA_MATCH_PATTERN = re.compile(
    r"\(metadata does (?P<field>[A-Za-z0-9_]+) \"(?P<value>[^\"]+)\"\)"
)


class MetadataSource(Enum):
    """Where CellProfiler extracts metadata from."""

    FILE_NAME = "file_name"
    FOLDER_NAME = "folder_name"


@dataclass(frozen=True, slots=True)
class MetadataExtractionRule:
    """One Metadata-module extraction rule."""

    source: MetadataSource
    file_name_regex: str
    folder_name_regex: str
    filter_criteria: str


@dataclass(frozen=True, slots=True)
class ImagesRule:
    """One Images-module source universe rule."""

    filtering_mode: str
    criteria: str


@dataclass(frozen=True, slots=True)
class ImageAssignment:
    """One NamesAndTypes image alias assignment."""

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
    """Typed Groups-module metadata grouping declaration."""

    metadata_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata_fields",
            tuple(field.strip() for field in self.metadata_fields if field.strip()),
        )


@dataclass(frozen=True, slots=True)
class CellProfilerImageSchema:
    """Pipeline-level source/image schema compiled from CellProfiler setup modules."""

    images_rule: ImagesRule | None = None
    metadata_rules: tuple[MetadataExtractionRule, ...] = ()
    assignments_by_alias: Mapping[str, ImageAssignment] = MappingProxyType({})
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
            and self.grouping is None
        )

    def assignment_for_alias(self, alias: str) -> ImageAssignment | None:
        return self.assignments_by_alias.get(alias)


@dataclass(frozen=True, slots=True)
class SetupModuleCompilation:
    """Mutable-free compilation state for setup modules."""

    images_rule: ImagesRule | None = None
    metadata_rules: tuple[MetadataExtractionRule, ...] = ()
    assignments_by_alias: Mapping[str, ImageAssignment] = MappingProxyType({})
    grouping: GroupingPlan | None = None

    def to_schema(self) -> CellProfilerImageSchema:
        return CellProfilerImageSchema(
            images_rule=self.images_rule,
            metadata_rules=self.metadata_rules,
            assignments_by_alias=self.assignments_by_alias,
            grouping=self.grouping,
        )


class SetupModuleCompiler(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for compiler-owned setup-module lowering."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "SetupModuleCompiler | None":
        compiler_type = cls.__registry__.get(module_name)
        if compiler_type is None:
            return None
        return compiler_type()

    @abstractmethod
    def compile(
        self,
        module: ModuleBlock,
        state: _SchemaBuilder,
    ) -> None:
        """Lower one setup module into schema state."""


class _SchemaBuilder:
    def __init__(self) -> None:
        self.images_rule: ImagesRule | None = None
        self.metadata_rules: list[MetadataExtractionRule] = []
        self.assignments_by_alias: dict[str, ImageAssignment] = {}
        self.grouping: GroupingPlan | None = None

    def build(self) -> CellProfilerImageSchema:
        return SetupModuleCompilation(
            images_rule=self.images_rule,
            metadata_rules=tuple(self.metadata_rules),
            assignments_by_alias=MappingProxyType(dict(self.assignments_by_alias)),
            grouping=self.grouping,
        ).to_schema()

    def add_metadata_rule(self, rule: MetadataExtractionRule) -> None:
        self.metadata_rules.append(rule)

    def declare_assignment(self, assignment: ImageAssignment) -> None:
        existing = self.assignments_by_alias.get(assignment.alias)
        if existing is not None and existing != assignment:
            raise ValueError(
                f"CellProfiler image alias {assignment.alias!r} is already declared "
                "with different setup semantics."
            )
        self.assignments_by_alias[assignment.alias] = assignment


def compile_image_schema(modules: Iterable[ModuleBlock]) -> CellProfilerImageSchema:
    """Compile setup modules into a typed pipeline-level CellProfiler image schema."""
    builder = _SchemaBuilder()
    for module in modules:
        if not module.enabled:
            continue
        compiler = SetupModuleCompiler.for_module(module.name)
        if compiler is not None:
            compiler.compile(module, builder)
    return builder.build()


class ImagesModuleCompiler(SetupModuleCompiler):
    module_name = "Images"

    def compile(
        self,
        module: ModuleBlock,
        state: _SchemaBuilder,
    ) -> None:
        filtering_mode = module.get_setting("Filter images?", "")
        criteria = module.get_setting("Select the rule criteria", "")
        if filtering_mode or criteria:
            state.images_rule = ImagesRule(
                filtering_mode=filtering_mode,
                criteria=criteria,
            )


class MetadataModuleCompiler(SetupModuleCompiler):
    module_name = "Metadata"

    def compile(
        self,
        module: ModuleBlock,
        state: _SchemaBuilder,
    ) -> None:
        for block in _group_repeating_blocks(
            module.iter_settings(),
            start_name="Metadata extraction method",
        ):
            source = _metadata_source(
                _block_value(block, "Metadata source", default="File name")
            )
            state.add_metadata_rule(
                MetadataExtractionRule(
                    source=source,
                    file_name_regex=_block_value(
                        block,
                        "Regular expression to extract from file name",
                    ),
                    folder_name_regex=_block_value(
                        block,
                        "Regular expression to extract from folder name",
                    ),
                    filter_criteria=_block_value(
                        block,
                        "Select the filtering criteria",
                    ),
                )
            )


class NamesAndTypesModuleCompiler(SetupModuleCompiler):
    module_name = "NamesAndTypes"

    def compile(
        self,
        module: ModuleBlock,
        state: _SchemaBuilder,
    ) -> None:
        for block in _group_repeating_blocks(
            module.iter_settings(),
            start_name="Select the rule criteria",
        ):
            alias = _block_value(block, "Name to assign these images", default="")
            if not alias:
                continue
            selector = _selector_from_rule_criteria(
                _block_value(block, "Select the rule criteria")
            )
            state.declare_assignment(
                ImageAssignment(
                    alias=alias,
                    image_type=_block_value(
                        block,
                        "Select the image type",
                        default="Grayscale image",
                    ),
                    selector=selector,
                    origin=_origin_for_selector(selector),
                )
            )


class GroupsModuleCompiler(SetupModuleCompiler):
    module_name = "Groups"

    def compile(
        self,
        module: ModuleBlock,
        state: _SchemaBuilder,
    ) -> None:
        if module.get_setting("Do you want to group your images?", "No") != "Yes":
            return
        metadata_fields = tuple(
            setting.value
            for setting in module.iter_settings("Metadata category")
        )
        state.grouping = GroupingPlan(metadata_fields=metadata_fields)


def _group_repeating_blocks(
    settings: Sequence[ModuleSetting],
    *,
    start_name: str,
) -> tuple[tuple[ModuleSetting, ...], ...]:
    blocks: list[list[ModuleSetting]] = []
    current_block: list[ModuleSetting] = []
    started = False
    for setting in settings:
        if setting.name == start_name:
            if started and current_block:
                blocks.append(current_block)
                current_block = []
            started = True
        if started:
            current_block.append(setting)
    if current_block:
        blocks.append(current_block)
    return tuple(tuple(block) for block in blocks)


def _block_value(
    block: Sequence[ModuleSetting],
    name: str,
    *,
    default: str = "",
) -> str:
    for setting in block:
        if setting.name == name:
            return setting.value
    return default


def _metadata_source(value: str) -> MetadataSource:
    normalized = value.strip().lower()
    if normalized == "folder name":
        return MetadataSource.FOLDER_NAME
    return MetadataSource.FILE_NAME


def _selector_from_rule_criteria(rule_criteria: str) -> SourceSelector:
    component_selectors: list[ComponentSelector] = []
    metadata_selectors: list[MetadataSelector] = []
    for match in _METADATA_MATCH_PATTERN.finditer(rule_criteria):
        field = match.group("field")
        value = match.group("value")
        component = _component_for_metadata_field(field)
        if component is not None:
            component_selectors.append(ComponentSelector(component, value))
        else:
            metadata_selectors.append(MetadataSelector(field, value))
    return SourceSelector(
        components=tuple(component_selectors),
        metadata=tuple(metadata_selectors),
    )


def _component_for_metadata_field(field: str) -> AllComponents | None:
    normalized = field.strip().lower()
    for component in AllComponents:
        if component.value == normalized:
            return component
    if normalized == "channelnumber":
        return AllComponents.CHANNEL
    return None


def _origin_for_selector(selector: SourceSelector) -> SourceBindingOrigin:
    if selector.metadata:
        return SourceBindingOrigin.PIPELINE_START
    return SourceBindingOrigin.STEP_INPUT
