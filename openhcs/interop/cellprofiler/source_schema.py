"""CellProfiler setup-module lowering onto the core pipeline image schema."""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.pipeline_image_schema import (
    GroupingPlan,
    GrayscaleImageTypeSourceRole,
    ImageAssignment,
    ImagePlaneSource,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    ImagesRule,
    PipelineImageSchema,
    PipelineImageSchemaBuilder,
    SourceImageStackPlan,
    SourceArtifactAssignment,
    image_type_artifact_kind,
    image_type_participates_in_image_stack,
)
from openhcs.core.component_set import ComponentSet
from openhcs.core.source_bindings import (
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
from openhcs.core.source_matching import source_metadata_component

from .parser import ModuleBlock, ModuleSetting
from .setting_names import (
    block_setting_value,
    block_setting_value_by_prefix,
    decode_cellprofiler_setting_literal,
    repeating_setting_blocks,
)

ModuleSettingBlock = tuple[ModuleSetting, ...]
ModuleSettingBlocks = tuple[ModuleSettingBlock, ...]
CELLPROFILER_SOURCE_SCHEMA_POLICY_KEY = "cellprofiler"

_METADATA_MATCH_PATTERN = re.compile(
    r"\(metadata does (?P<field>[A-Za-z0-9_]+) \"(?P<value>[^\"]+)\"\)"
)
_FILTER_CLAUSE_PATTERN = re.compile(
    r"\((?P<subject>file|directory|extension) "
    r"does\s*(?P<negation>not)?\s*"
    r"(?P<operator>containregexp|contain|startwith|endwith|is[a-z0-9]+|eq)"
    r"(?: \"(?P<value>[^\"]*)\")?\)"
)
_SOURCE_FILTER_SUBJECT_PATTERN = re.compile(
    r"\((file|directory|extension) does",
    re.IGNORECASE,
)
_LOAD_IMAGES_MATCH_TEXT_SETTING = (
    "Type the text that these images have in common (case-sensitive)"
)


class SourceSchemaLiteralResolver(ABC, metaclass=AutoRegisterMeta):
    """Nominal parser for one CellProfiler setup-schema literal family."""

    __registry_key__ = "literal"
    __skip_if_no_key__ = True
    literal: ClassVar[str | None] = None

    @classmethod
    def for_literal(cls, value: str) -> "SourceSchemaLiteralResolver":
        normalized = cls.normalize(value)
        resolver_type = cls.__registry__.get(normalized)
        if resolver_type is None:
            raise ValueError(cls.unsupported_message(value))
        return resolver_type()

    @classmethod
    def normalize(cls, value: str) -> str:
        return value.strip().lower()

    @classmethod
    @abstractmethod
    def unsupported_message(cls, value: str) -> str:
        """Return the error emitted for an unsupported literal."""


class SourceFilterSubjectLiteral(SourceSchemaLiteralResolver):
    """Registered CellProfiler source-filter subject literal."""

    __registry__: ClassVar[dict[str, type["SourceFilterSubjectLiteral"]]] = {}
    subject: ClassVar[SourceFilterSubject]

    @classmethod
    def unsupported_message(cls, value: str) -> str:
        return f"Unsupported source filter subject: {value!r}."


class FileFilterSubjectLiteral(SourceFilterSubjectLiteral):
    literal = "file"
    subject = SourceFilterSubject.FILE


class DirectoryFilterSubjectLiteral(SourceFilterSubjectLiteral):
    literal = "directory"
    subject = SourceFilterSubject.DIRECTORY


class ExtensionFilterSubjectLiteral(SourceFilterSubjectLiteral):
    literal = "extension"
    subject = SourceFilterSubject.EXTENSION


class SourceFilterOperatorLiteral(SourceSchemaLiteralResolver):
    """Registered CellProfiler source-filter operator literal."""

    __registry__: ClassVar[dict[str, type["SourceFilterOperatorLiteral"]]] = {}
    match_type: ClassVar[SourceFilterMatchType]
    negated_match_type: ClassVar[SourceFilterMatchType | None] = None

    @classmethod
    def unsupported_message(cls, value: str) -> str:
        return f"Unsupported source filter operator: {value!r}."

    def match_type_for_negation(self, negated: bool) -> SourceFilterMatchType:
        if negated:
            if self.negated_match_type is None:
                raise ValueError(
                    "Unsupported source filter operator/negation pair: "
                    f"{self.literal!r}, negated=True."
                )
            return self.negated_match_type
        return self.match_type


class ContainsFilterOperatorLiteral(SourceFilterOperatorLiteral):
    literal = "contain"
    match_type = SourceFilterMatchType.CONTAINS
    negated_match_type = SourceFilterMatchType.DOES_NOT_CONTAIN


class ContainsRegexFilterOperatorLiteral(SourceFilterOperatorLiteral):
    literal = "containregexp"
    match_type = SourceFilterMatchType.CONTAINS_REGEX
    negated_match_type = SourceFilterMatchType.DOES_NOT_CONTAIN_REGEX


class EqualsFilterOperatorLiteral(SourceFilterOperatorLiteral):
    literal = "eq"
    match_type = SourceFilterMatchType.EQUALS
    negated_match_type = SourceFilterMatchType.DOES_NOT_EQUAL


class StartsWithFilterOperatorLiteral(SourceFilterOperatorLiteral):
    literal = "startwith"
    match_type = SourceFilterMatchType.STARTS_WITH
    negated_match_type = SourceFilterMatchType.DOES_NOT_START_WITH


class EndsWithFilterOperatorLiteral(SourceFilterOperatorLiteral):
    literal = "endwith"
    match_type = SourceFilterMatchType.ENDS_WITH
    negated_match_type = SourceFilterMatchType.DOES_NOT_END_WITH


class IsImageFilterOperatorLiteral(SourceFilterOperatorLiteral):
    literal = "isimage"
    match_type = SourceFilterMatchType.IS_IMAGE


class IsTifFilterOperatorLiteral(SourceFilterOperatorLiteral):
    literal = "istif"
    match_type = SourceFilterMatchType.IS_TIF


class SourceBindingMatchMethodLiteral(SourceSchemaLiteralResolver):
    """Registered NamesAndTypes image-set matching method literal."""

    __registry__: ClassVar[dict[str, type["SourceBindingMatchMethodLiteral"]]] = {}
    method: ClassVar[SourceBindingMatchMethod]

    @classmethod
    def unsupported_message(cls, value: str) -> str:
        return f"Unsupported NamesAndTypes image set matching method: {value!r}."


class MetadataMatchMethodLiteral(SourceBindingMatchMethodLiteral):
    literal = "metadata"
    method = SourceBindingMatchMethod.METADATA


class OrderMatchMethodLiteral(SourceBindingMatchMethodLiteral):
    literal = "order"
    method = SourceBindingMatchMethod.ORDER


class SourceFilterCriteriaParser(ABC, metaclass=AutoRegisterMeta):
    """Nominal parser for CellProfiler source-filter criteria strings."""

    __registry_key__ = "parser_key"
    __skip_if_no_key__ = True
    parser_key: ClassVar[str | None] = None

    @classmethod
    def for_key(cls, parser_key: str) -> "SourceFilterCriteriaParser":
        return cls.__registry__[parser_key]()

    @abstractmethod
    def filter_clauses_from_criteria(
        self,
        criteria: str,
    ) -> tuple[SourceFilterClause, ...]:
        """Parse CellProfiler source-filter clauses from one criteria string."""


class CellProfilerSourceFilterCriteriaParser(SourceFilterCriteriaParser):
    """Parse CellProfiler's source-filter criteria expression subset."""

    parser_key = CELLPROFILER_SOURCE_SCHEMA_POLICY_KEY

    def filter_clauses_from_criteria(
        self,
        criteria: str,
    ) -> tuple[SourceFilterClause, ...]:
        decoded_criteria = decode_cellprofiler_setting_literal(criteria)
        stripped = decoded_criteria.strip()
        if not stripped:
            return ()
        matches = tuple(_FILTER_CLAUSE_PATTERN.finditer(decoded_criteria))
        if not matches:
            if not _SOURCE_FILTER_SUBJECT_PATTERN.search(decoded_criteria):
                return ()
            raise ValueError(
                "Unsupported CellProfiler source filter criteria: "
                f"{criteria!r}."
            )
        return tuple(
            clause
            for match in matches
            if (clause := self.filter_clause_from_match(match)) is not None
        )

    def filter_clause_from_match(
        self,
        match: re.Match[str],
    ) -> SourceFilterClause | None:
        """Return one source-filter clause for a regex match."""
        subject = self.filter_subject(match.group("subject"))
        operator = match.group("operator")
        negated = bool(match.group("negation"))
        value = match.group("value")
        if value == "":
            return None
        if (
            subject is SourceFilterSubject.EXTENSION
            and not negated
            and operator.startswith("is")
            and operator not in {"isimage", "istif"}
        ):
            return SourceFilterClause(
                subject=subject,
                match_type=SourceFilterMatchType.EQUALS,
                value=f".{operator.removeprefix('is')}",
            )
        return SourceFilterClause(
            subject=subject,
            match_type=self.filter_match_type(
                operator=operator,
                negated=negated,
            ),
            value=value,
        )

    @staticmethod
    def filter_subject(value: str) -> SourceFilterSubject:
        """Resolve a CellProfiler source-filter subject literal."""
        resolver = SourceFilterSubjectLiteral.for_literal(value)
        if not isinstance(resolver, SourceFilterSubjectLiteral):
            raise TypeError(
                "Expected source-filter subject resolver, got "
                f"{type(resolver).__name__}."
            )
        return resolver.subject

    @staticmethod
    def filter_match_type(
        *,
        operator: str,
        negated: bool,
    ) -> SourceFilterMatchType:
        """Resolve a CellProfiler source-filter operator literal."""
        resolver = SourceFilterOperatorLiteral.for_literal(operator)
        if not isinstance(resolver, SourceFilterOperatorLiteral):
            raise TypeError(
                "Expected source-filter operator resolver, got "
                f"{type(resolver).__name__}."
            )
        return resolver.match_type_for_negation(negated)


def cellprofiler_source_filter_criteria_parser() -> SourceFilterCriteriaParser:
    """Return the registered parser for CellProfiler source-filter criteria."""
    return SourceFilterCriteriaParser.for_key(CELLPROFILER_SOURCE_SCHEMA_POLICY_KEY)


class SourceBindingMatchMetadataParser(ABC, metaclass=AutoRegisterMeta):
    """Nominal parser for image-set match metadata declarations."""

    __registry_key__ = "parser_key"
    __skip_if_no_key__ = True
    parser_key: ClassVar[str | None] = None

    @classmethod
    def for_key(cls, parser_key: str) -> "SourceBindingMatchMetadataParser":
        return cls.__registry__[parser_key]()

    @abstractmethod
    def match_dimensions(
        self,
        raw_match_metadata: str,
    ) -> tuple[SourceBindingMatchDimension, ...]:
        """Parse one CellProfiler match-metadata declaration."""

    @abstractmethod
    def merge_match_dimensions_from_blocks(
        self,
        blocks: Sequence[Sequence[ModuleSetting]],
    ) -> tuple[SourceBindingMatchDimension, ...]:
        """Merge repeated assignment-block match-metadata declarations."""


class CellProfilerSourceBindingMatchMetadataParser(SourceBindingMatchMetadataParser):
    """Parse CellProfiler NamesAndTypes match-metadata declarations."""

    parser_key = CELLPROFILER_SOURCE_SCHEMA_POLICY_KEY

    def match_dimensions(
        self,
        raw_match_metadata: str,
    ) -> tuple[SourceBindingMatchDimension, ...]:
        try:
            records = ast.literal_eval(
                decode_cellprofiler_setting_literal(raw_match_metadata)
            )
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "Invalid NamesAndTypes 'Match metadata' value: "
                f"{raw_match_metadata!r}."
            ) from exc
        if not isinstance(records, list):
            raise TypeError(
                "NamesAndTypes 'Match metadata' must parse to a list of "
                "alias-field maps."
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

    def merge_match_dimensions_from_blocks(
        self,
        blocks: Sequence[Sequence[ModuleSetting]],
    ) -> tuple[SourceBindingMatchDimension, ...]:
        merged_dimensions: list[list[SourceBindingMatchField]] = []
        for block in blocks:
            raw_match_metadata = block_setting_value(block, "Match metadata").strip()
            if not raw_match_metadata:
                continue
            block_dimensions = self.match_dimensions(raw_match_metadata)
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


def cellprofiler_source_binding_match_metadata_parser() -> (
    SourceBindingMatchMetadataParser
):
    """Return the registered parser for CellProfiler match-metadata settings."""
    return SourceBindingMatchMetadataParser.for_key(
        CELLPROFILER_SOURCE_SCHEMA_POLICY_KEY
    )


class SourceBindingOriginPolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal policy for assigning source-binding origin domains."""

    __registry_key__ = "policy_key"
    __skip_if_no_key__ = True
    policy_key: ClassVar[str | None] = None

    @classmethod
    def for_key(cls, policy_key: str) -> "SourceBindingOriginPolicy":
        return cls.__registry__[policy_key]()

    @abstractmethod
    def origin_for_selector(self, selector: SourceSelector) -> SourceBindingOrigin:
        """Return the source origin implied by a selector."""

    @abstractmethod
    def origin_for_source_artifact(
        self,
        artifact_kind: ArtifactKind,
        image_type: str,
        selector: SourceSelector,
    ) -> SourceBindingOrigin:
        """Return the source origin implied by an artifact assignment."""


class CellProfilerSourceBindingOriginPolicy(SourceBindingOriginPolicy):
    """CellProfiler source-origin policy for setup-module lowering."""

    policy_key = CELLPROFILER_SOURCE_SCHEMA_POLICY_KEY

    def origin_for_selector(self, selector: SourceSelector) -> SourceBindingOrigin:
        if selector.filters:
            return SourceBindingOrigin.PIPELINE_START
        if selector.metadata and not all(
            source_metadata_component(metadata.field) is not None
            for metadata in selector.metadata
        ):
            return SourceBindingOrigin.PIPELINE_START
        return SourceBindingOrigin.STEP_INPUT

    def origin_for_source_artifact(
        self,
        artifact_kind: ArtifactKind,
        image_type: str,
        selector: SourceSelector,
    ) -> SourceBindingOrigin:
        if (
            artifact_kind is ArtifactKind.IMAGE
            and not image_type_participates_in_image_stack(image_type)
        ):
            return SourceBindingOrigin.PIPELINE_START
        return self.origin_for_selector(selector)


def cellprofiler_source_binding_origin_policy() -> SourceBindingOriginPolicy:
    """Return the registered origin policy for CellProfiler source bindings."""
    return SourceBindingOriginPolicy.for_key(CELLPROFILER_SOURCE_SCHEMA_POLICY_KEY)


class SourceImageStackPlanDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Nominal declaration of setup-module source-stack components."""

    __registry_key__ = "declaration_key"
    __skip_if_no_key__ = True
    declaration_key: ClassVar[str | None] = None

    @classmethod
    def plans_for_module(cls, module: ModuleBlock) -> tuple[SourceImageStackPlan, ...]:
        return tuple(
            declaration.stack_plan(module)
            for declaration_type in cls.__registry__.values()
            for declaration in (declaration_type(),)
            if declaration.matches(module)
        )

    @abstractmethod
    def matches(self, module: ModuleBlock) -> bool:
        """Return whether this setup module declares a source-stack plan."""

    @abstractmethod
    def stack_plan(self, module: ModuleBlock) -> SourceImageStackPlan:
        """Return the source-stack plan declared by the setup module."""


class NamesAndTypesProcessAs3DStackPlanDeclaration(SourceImageStackPlanDeclaration):
    """CellProfiler NamesAndTypes 3D mode stacks source images over Z."""

    declaration_key = "names_and_types_process_as_3d"

    def matches(self, module: ModuleBlock) -> bool:
        return (
            module.name == "NamesAndTypes"
            and module.get_setting("Process as 3D?", "No").strip().casefold() == "yes"
        )

    def stack_plan(self, module: ModuleBlock) -> SourceImageStackPlan:
        del module
        return SourceImageStackPlan((AllComponents.Z_INDEX,))


_LOAD_IMAGES_ALIAS_SETTING = (
    "What do you want to call this image in CellProfiler?"
)
_LOAD_IMAGES_METADATA_MODE_SETTING = (
    "Do you want to extract metadata from the file name, "
    "the subfolder path or both?"
)
_LOAD_IMAGES_FILE_PATTERN_SETTING_PREFIX = (
    "Type the regular expression that finds metadata in the file name"
)
_LOAD_IMAGES_FOLDER_PATTERN_SETTING_PREFIX = (
    "Type the regular expression that finds metadata in the subfolder path"
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
        state: PipelineImageSchemaBuilder,
    ) -> None:
        """Lower one setup module into schema state."""


def compile_image_schema(modules: Iterable[ModuleBlock]) -> PipelineImageSchema:
    """Compile setup modules into a typed pipeline-level image schema."""
    module_tuple = tuple(modules)
    builder = PipelineImageSchemaBuilder(
        source_image_types_by_alias=_source_image_types_by_alias(module_tuple)
    )
    for module in module_tuple:
        if not module.enabled:
            continue
        compiler = SetupModuleCompiler.for_module(module.name)
        if compiler is not None:
            compiler.compile(module, builder)
    _compile_embedded_image_plane_sources(module_tuple, builder)
    return builder.build()


def _source_image_types_by_alias(
    modules: Sequence[ModuleBlock],
) -> Mapping[str, str]:
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
    )

    image_types: dict[str, str] = {}
    for module in modules:
        if not module.enabled:
            continue
        module_type = CellProfilerModule.for_module(module.name)
        if module_type is None:
            continue
        for alias, image_type in module_type.source_image_types_by_alias(module).items():
            existing = image_types.get(alias)
            if existing is not None and existing != image_type:
                raise ValueError(
                    f"Source image alias {alias!r} has conflicting image-type "
                    f"declarations: {existing!r} != {image_type!r}."
                )
            image_types[alias] = image_type
    return image_types


def _compile_embedded_image_plane_sources(
    modules: Sequence[ModuleBlock],
    state: PipelineImageSchemaBuilder,
) -> None:
    for module in modules:
        source_rows = module.metadata.get("image_plane_sources")
        if not source_rows:
            continue
        for row in source_rows:
            if not isinstance(row, Mapping):
                continue
            uri = row.get("uri")
            if not uri:
                continue
            state.add_image_plane_source(
                ImagePlaneSource(
                    uri=str(uri),
                    series=_optional_image_plane_value(row.get("series")),
                    index=_optional_image_plane_value(row.get("index")),
                    channel=_optional_image_plane_value(row.get("channel")),
                )
            )
        return


def _optional_image_plane_value(value: object) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


class ImagesModuleCompiler(SetupModuleCompiler):
    module_name = "Images"

    def compile(
        self,
        module: ModuleBlock,
        state: PipelineImageSchemaBuilder,
    ) -> None:
        filters = _images_rule_filters(
            filtering_mode=module.get_setting("Filter images?", ""),
            criteria=module.get_setting("Select the rule criteria", ""),
        )
        if filters:
            state.images_rule = ImagesRule(filters=filters)


class LoadImagesModuleCompiler(SetupModuleCompiler):
    module_name = "LoadImages"

    def compile(
        self,
        module: ModuleBlock,
        state: PipelineImageSchemaBuilder,
    ) -> None:
        _require_legacy_load_images_source_type(module)
        _declare_load_images_grouping(module, state)
        for block in _load_images_blocks(module.iter_settings()):
            alias = block_setting_value(block, _LOAD_IMAGES_ALIAS_SETTING)
            if not alias:
                continue
            filters = _load_images_source_filters(module, block)
            selector = SourceSelector(filters=filters)
            image_type = state.source_image_type_for_alias(
                alias,
                GrayscaleImageTypeSourceRole.image_type(),
            )
            if image_type_participates_in_image_stack(image_type):
                state.declare_assignment(
                    ImageAssignment(
                        alias=alias,
                        image_type=image_type,
                        selector=selector,
                        origin=(
                            cellprofiler_source_binding_origin_policy()
                            .origin_for_selector(selector)
                        ),
                    )
                )
            else:
                state.declare_source_artifact(
                    SourceArtifactAssignment(
                        alias=alias,
                        artifact_kind=image_type_artifact_kind(image_type),
                        selector=selector,
                        origin=(
                            cellprofiler_source_binding_origin_policy()
                            .origin_for_source_artifact(
                                image_type_artifact_kind(image_type),
                                image_type,
                                selector,
                            )
                        ),
                        payload_type=image_type,
                    )
                )
            _compile_load_images_metadata_rules(block, filters, state)


class MetadataModuleCompiler(SetupModuleCompiler):
    module_name = "Metadata"

    def compile(
        self,
        module: ModuleBlock,
        state: PipelineImageSchemaBuilder,
    ) -> None:
        for block in repeating_setting_blocks(
            module.iter_settings(),
            start_name="Metadata extraction method",
        ):
            _compile_metadata_block(
                block,
                state,
                require_enabled=_metadata_extraction_enabled(module),
            )


class NamesAndTypesModuleCompiler(SetupModuleCompiler):
    module_name = "NamesAndTypes"

    def compile(
        self,
        module: ModuleBlock,
        state: PipelineImageSchemaBuilder,
    ) -> None:
        for stack_plan in SourceImageStackPlanDeclaration.plans_for_module(module):
            state.declare_source_image_stack(stack_plan)
        assignment_blocks = NamesAndTypesAssignmentBlockStrategy.blocks_for(
            module.iter_settings()
        )
        match_plan = _match_plan_from_names_and_types(module, assignment_blocks)
        if match_plan is not None:
            state.declare_match_plan(match_plan)
        for block in assignment_blocks:
            image_type = block_setting_value(
                block,
                "Select the image type",
                default="Grayscale image",
            )
            artifact_kind = image_type_artifact_kind(image_type)
            alias = _assignment_alias(block, artifact_kind)
            if not alias:
                continue
            selector = _selector_from_rule_criteria(
                block_setting_value(block, "Select the rule criteria")
            )
            if (
                artifact_kind is ArtifactKind.OBJECT_LABELS
                or not image_type_participates_in_image_stack(image_type)
            ):
                state.declare_source_artifact(
                    SourceArtifactAssignment(
                        alias=alias,
                        artifact_kind=artifact_kind,
                        selector=selector,
                        origin=(
                            cellprofiler_source_binding_origin_policy()
                            .origin_for_source_artifact(
                                artifact_kind,
                                image_type,
                                selector,
                            )
                        ),
                        payload_type=image_type,
                    )
                )
                continue
            state.declare_assignment(
                ImageAssignment(
                    alias=alias,
                    image_type=image_type,
                    selector=selector,
                    origin=(
                        cellprofiler_source_binding_origin_policy()
                        .origin_for_selector(selector)
                    ),
                )
            )


class GroupsModuleCompiler(SetupModuleCompiler):
    module_name = "Groups"

    def compile(
        self,
        module: ModuleBlock,
        state: PipelineImageSchemaBuilder,
    ) -> None:
        if module.get_setting("Do you want to group your images?", "No") != "Yes":
            return
        metadata_fields = tuple(
            setting.value
            for setting in module.iter_settings("Metadata category")
        )
        state.grouping = GroupingPlan(metadata_fields=metadata_fields)


@dataclass(frozen=True, slots=True)
class NamesAndTypesAssignmentLayout:
    """Declarative CellProfiler NamesAndTypes assignment layout variant."""

    strategy_name: str
    match_setting: str
    block_start_name: str
    exact_count: int | None = None
    minimum_count: int | None = None
    require_block_source_alias: bool = False

    def __post_init__(self) -> None:
        if (self.exact_count is None) == (self.minimum_count is None):
            raise ValueError(
                "NamesAndTypesAssignmentLayout must define exactly one of "
                "exact_count or minimum_count."
            )

    def matches(self, settings: Sequence[ModuleSetting]) -> bool:
        """Whether this layout applies to the ordered NamesAndTypes settings."""
        count = self.setting_count(
            settings,
            self.match_setting,
        )
        if self.exact_count is not None:
            return self.matches_blocks(settings, count == self.exact_count)
        return self.matches_blocks(settings, count >= self.minimum_count)

    def matches_blocks(
        self,
        settings: Sequence[ModuleSetting],
        count_matches: bool,
    ) -> bool:
        if not count_matches:
            return False
        if not self.require_block_source_alias:
            return True
        return all(
            self.block_declares_source_alias(block)
            for block in self.blocks(settings)
        )

    def blocks(
        self,
        settings: Sequence[ModuleSetting],
    ) -> ModuleSettingBlocks:
        """Return ordered assignment blocks for this layout."""
        return repeating_setting_blocks(
            settings,
            start_name=self.block_start_name,
        )

    @staticmethod
    def setting_count(
        settings: Sequence[ModuleSetting],
        name: str,
    ) -> int:
        """Return how often a setting appears in the candidate layout."""
        return sum(1 for setting in settings if setting.name == name)

    @staticmethod
    def block_declares_source_alias(block: Sequence[ModuleSetting]) -> bool:
        """Return whether a block declares either image or object source alias."""
        return bool(
            block_setting_value(block, "Name to assign these images", default="")
            or block_setting_value(block, "Name to assign these objects", default="")
        )


NAMES_AND_TYPES_ASSIGNMENT_LAYOUTS: tuple[NamesAndTypesAssignmentLayout, ...] = (
    NamesAndTypesAssignmentLayout(
        strategy_name="repeated_rule_criteria",
        match_setting="Select the rule criteria",
        block_start_name="Select the rule criteria",
        minimum_count=2,
        require_block_source_alias=True,
    ),
    NamesAndTypesAssignmentLayout(
        strategy_name="repeated_assignment",
        match_setting="Assign a name to",
        block_start_name="Assign a name to",
        minimum_count=2,
    ),
    NamesAndTypesAssignmentLayout(
        strategy_name="single_rule_criteria",
        match_setting="Select the rule criteria",
        block_start_name="Select the rule criteria",
        exact_count=1,
        require_block_source_alias=True,
    ),
    NamesAndTypesAssignmentLayout(
        strategy_name="single_assignment",
        match_setting="Assign a name to",
        block_start_name="Assign a name to",
        exact_count=1,
    ),
)


class NamesAndTypesAssignmentBlockStrategy:
    """Resolve CellProfiler NamesAndTypes assignment blocks from layout declarations."""

    layouts: ClassVar[tuple[NamesAndTypesAssignmentLayout, ...]] = (
        NAMES_AND_TYPES_ASSIGNMENT_LAYOUTS
    )

    @classmethod
    def blocks_for(
        cls,
        settings: Sequence[ModuleSetting],
    ) -> ModuleSettingBlocks:
        for layout in cls.layouts:
            if layout.matches(settings):
                return layout.blocks(settings)
        return ()


def _load_images_blocks(
    settings: Sequence[ModuleSetting],
) -> ModuleSettingBlocks:
    return repeating_setting_blocks(
        settings,
        start_name=_LOAD_IMAGES_MATCH_TEXT_SETTING,
    )


def _require_legacy_load_images_source_type(module: ModuleBlock) -> None:
    file_type = module.get_setting("What type of files are you loading?", "")
    if file_type and "individual" not in file_type.strip().lower():
        raise ValueError(
            "LoadImages setup lowering only supports individual-image source "
            f"declarations, got {file_type!r}."
        )


def _images_rule_filters(
    *,
    filtering_mode: str,
    criteria: str,
) -> tuple[SourceFilterClause, ...]:
    filters = list(
        cellprofiler_source_filter_criteria_parser().filter_clauses_from_criteria(
            criteria
        )
    )
    if _criteria_is_multi_clause_disjunction(criteria, filters):
        # PipelineImageSchema.ImagesRule is intentionally conjunctive. A
        # multi-clause CP Images disjunction is a source-universe prefilter, so
        # lowering it as AND is incorrect and can exclude valid per-alias
        # matches. Preserve correctness by not applying a global prefilter; the
        # NamesAndTypes selectors still bind the concrete aliases.
        return ()
    normalized_mode = filtering_mode.strip().lower()
    if "images" in normalized_mode and not any(
        clause.match_type is SourceFilterMatchType.IS_IMAGE
        for clause in filters
    ):
        filters.insert(
            0,
            SourceFilterClause(
                subject=SourceFilterSubject.FILE,
                match_type=SourceFilterMatchType.IS_IMAGE,
            ),
    )
    return tuple(dict.fromkeys(filters))


def _criteria_is_multi_clause_disjunction(
    criteria: str,
    filters: Sequence[SourceFilterClause],
) -> bool:
    stripped = criteria.strip().lower()
    return (
        (stripped.startswith("or ") and stripped.count("(") > 1)
        or "(or " in stripped
        or stripped.count("(") > len(filters)
    )


def _declare_load_images_grouping(
    module: ModuleBlock,
    state: PipelineImageSchemaBuilder,
) -> None:
    if module.get_setting("Do you want to group image sets by metadata?", "") != "Yes":
        return
    fields = tuple(
        field.strip()
        for field in re.split(
            r"[,;]",
            module.get_setting("What metadata fields do you want to group by?", ""),
        )
        if field.strip()
    )
    if fields:
        state.grouping = GroupingPlan(metadata_fields=fields)


def _load_images_source_filters(
    module: ModuleBlock,
    block: Sequence[ModuleSetting],
) -> tuple[SourceFilterClause, ...]:
    filters: list[SourceFilterClause] = []
    match_text = block_setting_value(block, _LOAD_IMAGES_MATCH_TEXT_SETTING)
    if match_text:
        filters.append(
            SourceFilterClause(
                subject=SourceFilterSubject.FILE,
                match_type=_load_images_match_type(module),
                value=decode_cellprofiler_setting_literal(match_text),
            )
        )
    if module.get_setting("Do you want to exclude certain files?", "") == "Yes":
        exclusion_text = module.get_setting(
            "Type the text that the excluded images have in common",
            "",
        )
        if exclusion_text:
            filters.append(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.DOES_NOT_CONTAIN,
                    value=decode_cellprofiler_setting_literal(exclusion_text),
                )
            )
    return tuple(filters)


def _load_images_match_type(module: ModuleBlock) -> SourceFilterMatchType:
    mode = module.get_setting("How do you want to load these files?", "")
    normalized = mode.strip().lower()
    if not normalized or "exact" in normalized:
        return SourceFilterMatchType.CONTAINS
    if "regular" in normalized or "regex" in normalized:
        return SourceFilterMatchType.CONTAINS_REGEX
    raise ValueError(f"Unsupported LoadImages matching mode: {mode!r}.")


def _compile_load_images_metadata_rules(
    block: Sequence[ModuleSetting],
    filters: tuple[SourceFilterClause, ...],
    state: PipelineImageSchemaBuilder,
) -> None:
    mode = block_setting_value(block, _LOAD_IMAGES_METADATA_MODE_SETTING)
    for source in _load_images_metadata_sources(mode):
        state.add_metadata_rule(
            MetadataExtractionRule(
                source=source,
                pattern=_required_load_images_metadata_pattern(block, source),
                filters=filters,
            )
        )


def _load_images_metadata_sources(mode: str) -> tuple[MetadataSource, ...]:
    normalized = mode.strip().lower()
    if not normalized or normalized == "none":
        return ()
    sources: list[MetadataSource] = []
    if "file" in normalized or "both" in normalized:
        sources.append(MetadataSource.FILE_NAME)
    if (
        "folder" in normalized
        or "subfolder" in normalized
        or "path" in normalized
        or "both" in normalized
    ):
        sources.append(MetadataSource.FOLDER_NAME)
    if sources:
        return tuple(dict.fromkeys(sources))
    raise ValueError(f"Unsupported LoadImages metadata extraction mode: {mode!r}.")


def _required_load_images_metadata_pattern(
    block: Sequence[ModuleSetting],
    source: MetadataSource,
) -> str:
    prefix = (
        _LOAD_IMAGES_FOLDER_PATTERN_SETTING_PREFIX
        if source is MetadataSource.FOLDER_NAME
        else _LOAD_IMAGES_FILE_PATTERN_SETTING_PREFIX
    )
    pattern = decode_cellprofiler_setting_literal(
        block_setting_value_by_prefix(block, prefix)
    )
    if not pattern or pattern.strip().lower() == "none":
        raise ValueError(
            "LoadImages metadata extraction requires a non-empty "
            f"{source.value} regular expression."
        )
    return pattern


def _assignment_alias(
    block: Sequence[ModuleSetting],
    artifact_kind: ArtifactKind,
) -> str:
    if artifact_kind is ArtifactKind.OBJECT_LABELS:
        return block_setting_value(block, "Name to assign these objects", default="")
    return block_setting_value(block, "Name to assign these images", default="")


def _metadata_source(value: str) -> MetadataSource:
    normalized = value.strip().lower()
    if normalized == "folder name":
        return MetadataSource.FOLDER_NAME
    return MetadataSource.FILE_NAME


def _metadata_extraction_enabled(module: ModuleBlock) -> bool:
    value = decode_cellprofiler_setting_literal(
        module.get_setting("Extract metadata?", "Yes")
    )
    normalized = value.replace("\x00", "").strip().lower()
    return normalized not in {"no", "false", "0"}


def _compile_metadata_block(
    block: Sequence[ModuleSetting],
    state: PipelineImageSchemaBuilder,
    *,
    require_enabled: bool,
) -> None:
    method = block_setting_value(block, "Metadata extraction method")
    if is_imported_metadata_method(method):
        if not require_enabled:
            return
        state.add_imported_metadata_table(_imported_metadata_table(block))
        return
    if not _is_path_metadata_extraction_method(method):
        raise ValueError(f"Unsupported CellProfiler metadata extraction method: {method!r}.")

    if not require_enabled and not DisabledPathMetadataRulePolicy.for_block(block).preserve:
        return

    source = _metadata_source(
        block_setting_value(block, "Metadata source", default="File name")
    )
    pattern_block = CellProfilerMetadataPatternBlock(block, source)
    pattern = pattern_block.pattern
    if not require_enabled and not pattern:
        return
    state.add_metadata_rule(
        MetadataExtractionRule(
            source=source,
            pattern=pattern_block.required_pattern if require_enabled else pattern,
            filters=(
                cellprofiler_source_filter_criteria_parser()
                .filter_clauses_from_criteria(
                    block_setting_value(block, "Select the filtering criteria")
                )
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class CellProfilerMetadataPatternBlock:
    """Typed access to CP metadata regex settings for one source domain."""

    block: Sequence[ModuleSetting]
    source: MetadataSource

    @property
    def pattern(self) -> str:
        if self.source is MetadataSource.FOLDER_NAME:
            folder_pattern = block_setting_value(
                self.block,
                "Regular expression to extract from folder name",
            )
            return decode_cellprofiler_setting_literal(
                folder_pattern or _legacy_regex_value(self.block, index=1)
            )
        file_pattern = block_setting_value(
            self.block,
            "Regular expression to extract from file name",
        )
        return decode_cellprofiler_setting_literal(
            file_pattern or _legacy_regex_value(self.block, index=0)
        )

    @property
    def required_pattern(self) -> str:
        pattern = self.pattern
        if not pattern:
            raise ValueError(
                "CellProfiler path metadata extraction requires a non-empty "
                f"{self.source.value} regular expression."
            )
        return pattern


@dataclass(frozen=True, slots=True)
class DisabledPathMetadataRulePolicy:
    """Preservation policy for disabled CP metadata rules needed by source binding."""

    pattern: str

    @classmethod
    def for_block(
        cls,
        block: Sequence[ModuleSetting],
    ) -> "DisabledPathMetadataRulePolicy":
        source = _metadata_source(
            block_setting_value(block, "Metadata source", default="File name")
        )
        return cls(CellProfilerMetadataPatternBlock(block, source).pattern)

    @property
    def preserve(self) -> bool:
        fields = self.capture_fields
        if not fields:
            return False
        components = tuple(source_metadata_component(field) for field in fields)
        return (
            any(component is None for component in components)
            and not any(
                DisabledMetadataAxisComponents().contains(component)
                for component in components
            )
        )

    @property
    def capture_fields(self) -> tuple[str, ...]:
        try:
            return tuple(re.compile(self.pattern).groupindex)
        except re.error:
            return ()


@dataclass(frozen=True, slots=True)
class DisabledMetadataAxisComponents:
    """Components whose disabled metadata rules should not be resurrected."""

    components: ComponentSet = field(
        default_factory=lambda: ComponentSet.from_enum_values((AllComponents.CHANNEL,))
    )

    def contains(self, component: AllComponents | None) -> bool:
        return component in self.components


def _is_path_metadata_extraction_method(value: str) -> bool:
    normalized = value.strip().lower()
    return "extract" in normalized and (
        "file/folder" in normalized
        or "file" in normalized
        or "folder" in normalized
    )


def is_imported_metadata_method(value: str) -> bool:
    """Return whether a CellProfiler Metadata method imports an external table."""
    normalized = value.strip().lower()
    return "import" in normalized and "file" in normalized


def _imported_metadata_table(block: Sequence[ModuleSetting]) -> ImportedMetadataTable:
    location = _imported_metadata_location(
        block_setting_value(block, "Metadata file location", default="")
    )
    file_name = decode_cellprofiler_setting_literal(
        block_setting_value(block, "Metadata file name", default="")
    ).strip()
    return ImportedMetadataTable(
        location=_imported_metadata_table_path(location, file_name),
        joins=_imported_metadata_joins(block),
    )


def _imported_metadata_table_path(
    location: str | None,
    file_name: str,
) -> str | None:
    if not file_name:
        return location
    if location is None:
        return file_name
    return str(Path(location) / file_name)


def _imported_metadata_location(value: str) -> str | None:
    decoded = decode_cellprofiler_setting_literal(value).strip()
    if not decoded:
        return None
    if "|" not in decoded:
        return decoded
    location_kind, location_path = decoded.split("|", 1)
    if location_kind.strip().lower() != "default input folder":
        raise ValueError(
            "Metadata imported-table lowering only supports Default Input Folder "
            f"locations, got {location_kind!r}."
        )
    normalized_path = location_path.strip()
    return normalized_path or None


def _imported_metadata_joins(
    block: Sequence[ModuleSetting],
) -> tuple[ImportedMetadataJoin, ...]:
    raw_match_metadata = block_setting_value(block, "Match file and image metadata")
    if not raw_match_metadata:
        return ()
    try:
        records = ast.literal_eval(
            decode_cellprofiler_setting_literal(raw_match_metadata)
        )
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "Invalid Metadata 'Match file and image metadata' value: "
            f"{raw_match_metadata!r}."
        ) from exc
    if not isinstance(records, list):
        raise TypeError(
            "Metadata 'Match file and image metadata' must parse to a list "
            "of join records."
        )
    joins: list[ImportedMetadataJoin] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError(
                "Metadata imported-table join records must be mappings."
            )
        image_field = record.get("Image Metadata")
        imported_field = record.get("CSV Metadata")
        if image_field is None or imported_field is None:
            continue
        joins.append(
            ImportedMetadataJoin(
                image_metadata_field=str(image_field),
                imported_metadata_field=str(imported_field),
            )
        )
    return tuple(joins)


def _legacy_regex_value(
    block: Sequence[ModuleSetting],
    *,
    index: int,
) -> str:
    values = tuple(
        setting.value
        for setting in block
        if setting.name == "Regular expression"
    )
    if index < len(values):
        return values[index]
    return ""


def _selector_from_rule_criteria(rule_criteria: str) -> SourceSelector:
    metadata_selectors: list[MetadataSelector] = []
    for match in _METADATA_MATCH_PATTERN.finditer(rule_criteria):
        field = match.group("field")
        value = match.group("value")
        metadata_selectors.append(MetadataSelector(field, value))
    return SourceSelector(
        metadata=tuple(metadata_selectors),
        filters=(
            cellprofiler_source_filter_criteria_parser()
            .filter_clauses_from_criteria(rule_criteria)
        ),
    )


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
            dimensions=(
                cellprofiler_source_binding_match_metadata_parser()
                .match_dimensions(raw_match_metadata_values[0])
            ),
        )
    return SourceBindingMatchPlan(
        method=method,
        dimensions=(
            cellprofiler_source_binding_match_metadata_parser()
            .merge_match_dimensions_from_blocks(blocks)
        ),
    )


def _source_binding_match_method(value: str) -> SourceBindingMatchMethod:
    resolver = SourceBindingMatchMethodLiteral.for_literal(value)
    if not isinstance(resolver, SourceBindingMatchMethodLiteral):
        raise TypeError(
            "Expected source-binding match-method resolver, got "
            f"{type(resolver).__name__}."
        )
    return resolver.method
