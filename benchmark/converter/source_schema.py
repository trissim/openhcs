"""CellProfiler setup-module lowering onto the core pipeline image schema."""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents
from openhcs.core.pipeline_image_schema import (
    CellProfilerImageSchema,
    GroupingPlan,
    ImageAssignment,
    ImagesRule,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    MetadataExtractionRule,
    MetadataSource,
    MetadataSelector,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceBindingOrigin,
    SourceSelector,
)

from .parser import ModuleBlock, ModuleSetting

_METADATA_MATCH_PATTERN = re.compile(
    r"\(metadata does (?P<field>[A-Za-z0-9_]+) \"(?P<value>[^\"]+)\"\)"
)
_FILTER_CLAUSE_PATTERN = re.compile(
    r"\((?P<subject>file|directory|extension) "
    r"does(?P<negation>not)? "
    r"(?P<operator>containregexp|contain|isimage)"
    r"(?: \"(?P<value>[^\"]*)\")?\)"
)
_FILTER_SUBJECTS_BY_LITERAL = MappingProxyType(
    {
        "file": SourceFilterSubject.FILE,
        "directory": SourceFilterSubject.DIRECTORY,
        "extension": SourceFilterSubject.EXTENSION,
    }
)
_FILTER_MATCH_TYPES_BY_LITERAL = MappingProxyType(
    {
        ("contain", False): SourceFilterMatchType.CONTAINS,
        ("contain", True): SourceFilterMatchType.DOES_NOT_CONTAIN,
        ("containregexp", False): SourceFilterMatchType.CONTAINS_REGEX,
        ("containregexp", True): SourceFilterMatchType.DOES_NOT_CONTAIN_REGEX,
        ("isimage", False): SourceFilterMatchType.IS_IMAGE,
    }
)
_SOURCE_BINDING_MATCH_METHODS_BY_LITERAL = MappingProxyType(
    {
        "metadata": SourceBindingMatchMethod.METADATA,
        "order": SourceBindingMatchMethod.ORDER,
    }
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

@dataclass(frozen=True, slots=True)
class _SetupModuleCompilation:
    """Mutable-free setup-module lowering state."""

    images_rule: ImagesRule | None = None
    metadata_rules: tuple[MetadataExtractionRule, ...] = ()
    assignments_by_alias: Mapping[str, ImageAssignment] = MappingProxyType({})
    match_plan: SourceBindingMatchPlan | None = None
    grouping: GroupingPlan | None = None

    def to_schema(self) -> CellProfilerImageSchema:
        return CellProfilerImageSchema(
            images_rule=self.images_rule,
            metadata_rules=self.metadata_rules,
            assignments_by_alias=self.assignments_by_alias,
            match_plan=self.match_plan,
            grouping=self.grouping,
        )


class _SchemaBuilder:
    def __init__(self) -> None:
        self.images_rule: ImagesRule | None = None
        self.metadata_rules: list[MetadataExtractionRule] = []
        self.assignments_by_alias: dict[str, ImageAssignment] = {}
        self.match_plan: SourceBindingMatchPlan | None = None
        self.grouping: GroupingPlan | None = None

    def build(self) -> CellProfilerImageSchema:
        return _SetupModuleCompilation(
            images_rule=self.images_rule,
            metadata_rules=tuple(self.metadata_rules),
            assignments_by_alias=MappingProxyType(dict(self.assignments_by_alias)),
            match_plan=self.match_plan,
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

    def declare_match_plan(self, match_plan: SourceBindingMatchPlan) -> None:
        if self.match_plan is not None and self.match_plan != match_plan:
            raise ValueError(
                "CellProfiler image schema already declared a different image-set "
                "match plan."
            )
        self.match_plan = match_plan


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
                    pattern=_metadata_pattern_for_block(block, source),
                    filters=_filter_clauses_from_criteria(
                        _block_value(
                            block,
                            "Select the filtering criteria",
                        )
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
        assignment_blocks = _names_and_types_blocks(module.iter_settings())
        match_plan = _match_plan_from_names_and_types(module, assignment_blocks)
        if match_plan is not None:
            state.declare_match_plan(match_plan)
        for block in assignment_blocks:
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


def _names_and_types_blocks(
    settings: Sequence[ModuleSetting],
) -> tuple[tuple[ModuleSetting, ...], ...]:
    if any(setting.name == "Assign a name to" for setting in settings):
        return _group_repeating_blocks(settings, start_name="Assign a name to")
    return _group_repeating_blocks(settings, start_name="Select the rule criteria")


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


def _metadata_pattern_for_block(
    block: Sequence[ModuleSetting],
    source: MetadataSource,
) -> str:
    if source is MetadataSource.FOLDER_NAME:
        return _block_value(
            block,
            "Regular expression to extract from folder name",
        )
    return _block_value(
        block,
        "Regular expression to extract from file name",
    )


def _filter_clauses_from_criteria(
    criteria: str,
) -> tuple[SourceFilterClause, ...]:
    stripped = criteria.strip()
    if not stripped:
        return ()
    matches = tuple(_FILTER_CLAUSE_PATTERN.finditer(criteria))
    if not matches:
        raise ValueError(
            "Unsupported CellProfiler source filter criteria: "
            f"{criteria!r}."
        )
    return tuple(
        SourceFilterClause(
            subject=_filter_subject(match.group("subject")),
            match_type=_filter_match_type(
                operator=match.group("operator"),
                negated=bool(match.group("negation")),
            ),
            value=match.group("value"),
        )
        for match in matches
    )


def _filter_subject(value: str) -> SourceFilterSubject:
    normalized = value.strip().lower()
    try:
        return _FILTER_SUBJECTS_BY_LITERAL[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported source filter subject: {value!r}.") from exc


def _filter_match_type(
    *,
    operator: str,
    negated: bool,
) -> SourceFilterMatchType:
    normalized_operator = operator.strip().lower()
    try:
        return _FILTER_MATCH_TYPES_BY_LITERAL[(normalized_operator, negated)]
    except KeyError as exc:
        raise ValueError(
            "Unsupported source filter operator/negation pair: "
            f"{operator!r}, negated={negated}."
        ) from exc


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


def _match_plan_from_names_and_types(
    module: ModuleBlock,
    blocks: Sequence[Sequence[ModuleSetting]],
) -> SourceBindingMatchPlan | None:
    method_values = tuple(
        value.strip()
        for value in module.get_setting_values("Image set matching method")
        if value.strip()
    )
    if not method_values:
        return None
    method = _source_binding_match_method(method_values[0])
    if any(
        _source_binding_match_method(value) is not method
        for value in method_values[1:]
    ):
        raise ValueError(
            "NamesAndTypes declared conflicting image set matching methods."
        )
    if method is SourceBindingMatchMethod.ORDER:
        return SourceBindingMatchPlan(method=method)
    raw_match_metadata_values = tuple(
        value.strip()
        for value in module.get_setting_values("Match metadata")
        if value.strip()
    )
    if not raw_match_metadata_values:
        return SourceBindingMatchPlan(method=method)
    if len(raw_match_metadata_values) == 1:
        return SourceBindingMatchPlan(
            method=method,
            dimensions=_match_dimensions(raw_match_metadata_values[0]),
        )
    return SourceBindingMatchPlan(
        method=method,
        dimensions=_merge_match_dimensions_from_blocks(blocks),
    )


def _source_binding_match_method(value: str) -> SourceBindingMatchMethod:
    normalized = value.strip().lower()
    try:
        return _SOURCE_BINDING_MATCH_METHODS_BY_LITERAL[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported NamesAndTypes image set matching method: {value!r}."
        ) from exc


def _match_dimensions(
    raw_match_metadata: str,
) -> tuple[SourceBindingMatchDimension, ...]:
    try:
        records = ast.literal_eval(bytes(raw_match_metadata, "utf-8").decode("unicode_escape"))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Invalid NamesAndTypes 'Match metadata' value: {raw_match_metadata!r}."
        ) from exc
    if not isinstance(records, list):
        raise TypeError(
            "NamesAndTypes 'Match metadata' must parse to a list of alias-field maps."
        )
    dimensions: list[SourceBindingMatchDimension] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError(
                "NamesAndTypes 'Match metadata' entries must be dictionaries."
            )
        fields = tuple(
            SourceBindingMatchField(alias=str(alias), metadata_field=str(field))
            for alias, field in record.items()
            if field is not None
        )
        if fields:
            dimensions.append(SourceBindingMatchDimension(fields=fields))
    return tuple(dimensions)


def _merge_match_dimensions_from_blocks(
    blocks: Sequence[Sequence[ModuleSetting]],
) -> tuple[SourceBindingMatchDimension, ...]:
    merged_dimensions: list[list[SourceBindingMatchField]] = []
    for block in blocks:
        raw_match_metadata = _block_value(block, "Match metadata").strip()
        if not raw_match_metadata:
            continue
        block_dimensions = _match_dimensions(raw_match_metadata)
        if not merged_dimensions:
            merged_dimensions = [[] for _ in block_dimensions]
        if len(merged_dimensions) != len(block_dimensions):
            raise ValueError(
                "NamesAndTypes declared incompatible image-set match dimensions "
                "across repeated image assignments."
            )
        for index, dimension in enumerate(block_dimensions):
            merged_dimensions[index].extend(dimension.fields)
    return tuple(
        SourceBindingMatchDimension(fields=tuple(fields))
        for fields in merged_dimensions
        if fields
    )
