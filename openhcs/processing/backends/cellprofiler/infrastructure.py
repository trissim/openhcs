"""CellProfiler setup-module declarations and source lowering."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, ClassVar

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactType,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    ImagePlaneSource,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    MetadataExtractionRule,
    MetadataSelector,
    MetadataSource,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceProjectionRole,
    SourceSelector,
    SourceSetRole,
)
from openhcs.core.source_matching import source_metadata_component
from openhcs.core.source_metadata import SourceVoxelSpacing
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    decode_cellprofiler_setting_literal,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    SourceSetupCellProfilerModule,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    block_setting_value,
    block_setting_value_by_prefix,
    repeating_setting_blocks,
    setting_name_matches,
)
from openhcs.interop.cellprofiler.settings_binder import parse_cellprofiler_bool
from openhcs.interop.cellprofiler.source_metadata import (
    CellProfilerSourceMetadataField,
)

_METADATA_MATCH_PATTERN = re.compile(
    '\\(metadata does (?P<field>[A-Za-z0-9_]+) \\"(?P<value>[^\\"]+)\\"\\)'
)
_METADATA_PREDICATE_PATTERN = re.compile(
    r"\(metadata does(?:not)? [^)]*\)",
    re.IGNORECASE,
)
_FILTER_CLAUSE_PATTERN = re.compile(
    "\\((?P<subject>file|directory|extension) does\\s*"
    "(?P<negation>not)?\\s*"
    "(?P<operator>containregexp|contain|startwith|endwith|is[a-z0-9]+|eq)"
    '(?: \\"(?P<value>[^\\"]*)\\")?\\)'
)
_SOURCE_FILTER_SUBJECT_PATTERN = re.compile(
    "\\((file|directory|extension) does",
    re.IGNORECASE,
)
_SOURCE_FILTER_TOKEN_PATTERN = re.compile(r'\s*(\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+)')


class CellProfilerSourceFilterSubject(Enum):
    """CellProfiler filter subject translated to one generic subject."""

    def __new__(cls, literal: str, subject: SourceFilterSubject):
        member = object.__new__(cls)
        member._value_ = literal
        member.subject = subject
        return member

    FILE = ("file", SourceFilterSubject.FILE)
    DIRECTORY = ("directory", SourceFilterSubject.DIRECTORY)
    EXTENSION = ("extension", SourceFilterSubject.EXTENSION)

    @classmethod
    def from_literal(cls, value: str) -> "CellProfilerSourceFilterSubject":
        normalized = value.strip().casefold()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported source filter subject: {value!r}.")


class CellProfilerSourceFilterOperator(Enum):
    """CellProfiler filter operator translated to generic positive/negative forms."""

    def __new__(
        cls,
        literal: str,
        match_type: SourceFilterMatchType,
        negated_match_type: SourceFilterMatchType | None,
    ):
        member = object.__new__(cls)
        member._value_ = literal
        member.match_type = match_type
        member.negated_match_type = negated_match_type
        return member

    CONTAINS = (
        "contain",
        SourceFilterMatchType.CONTAINS,
        SourceFilterMatchType.DOES_NOT_CONTAIN,
    )
    CONTAINS_REGEX = (
        "containregexp",
        SourceFilterMatchType.CONTAINS_REGEX,
        SourceFilterMatchType.DOES_NOT_CONTAIN_REGEX,
    )
    EQUALS = (
        "eq",
        SourceFilterMatchType.EQUALS,
        SourceFilterMatchType.DOES_NOT_EQUAL,
    )
    STARTS_WITH = (
        "startwith",
        SourceFilterMatchType.STARTS_WITH,
        SourceFilterMatchType.DOES_NOT_START_WITH,
    )
    ENDS_WITH = (
        "endwith",
        SourceFilterMatchType.ENDS_WITH,
        SourceFilterMatchType.DOES_NOT_END_WITH,
    )
    IS_IMAGE = ("isimage", SourceFilterMatchType.IS_IMAGE, None)
    IS_TIF = ("istif", SourceFilterMatchType.IS_TIF, None)

    @classmethod
    def from_literal(cls, value: str) -> "CellProfilerSourceFilterOperator":
        normalized = value.strip().casefold()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported source filter operator: {value!r}.")

    def generic_match_type(self, *, negated: bool) -> SourceFilterMatchType:
        if not negated:
            return self.match_type
        if self.negated_match_type is None:
            raise ValueError(
                "Unsupported source filter operator/negation pair: "
                f"{self.value!r}, negated=True."
            )
        return self.negated_match_type


class CellProfilerSourceMatchMethod(Enum):
    """CellProfiler image-set matching method translated to generic matching."""

    def __new__(cls, literal: str, method: SourceBindingMatchMethod):
        member = object.__new__(cls)
        member._value_ = literal
        member.method = method
        return member

    METADATA = ("metadata", SourceBindingMatchMethod.METADATA)
    ORDER = ("order", SourceBindingMatchMethod.ORDER)

    @classmethod
    def from_literal(cls, value: str) -> "CellProfilerSourceMatchMethod":
        normalized = value.strip().casefold()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unsupported NamesAndTypes image set matching method: {value!r}."
        )


class CellProfilerSourceImageType(Enum):
    """CellProfiler source image type lowered to generic binding behavior."""

    def __new__(
        cls,
        literal: str,
        artifact_kind: type[ArtifactType],
        projection_role: SourceProjectionRole,
        load_as_monochrome: bool,
        load_as_mask: bool,
        source_channel_axis: int | None,
        source_channel_counts: frozenset[int] | None,
    ):
        member = object.__new__(cls)
        member._value_ = literal
        member.artifact_kind = artifact_kind
        member.projection_role = projection_role
        member.load_as_monochrome = load_as_monochrome
        member.load_as_mask = load_as_mask
        member.source_channel_axis = source_channel_axis
        member.source_channel_counts = source_channel_counts
        return member

    GRAYSCALE = (
        "grayscale image",
        ImageArtifactType,
        SourceProjectionRole.PRIMARY_PLANE,
        True,
        False,
        None,
        None,
    )
    COLOR = (
        "color image",
        ImageArtifactType,
        SourceProjectionRole.PRIMARY_PLANE,
        False,
        False,
        -1,
        None,
    )
    BINARY_IMAGE = (
        "binary image",
        ImageArtifactType,
        SourceProjectionRole.PRIMARY_PLANE,
        True,
        True,
        None,
        None,
    )
    BINARY_MASK = (
        "binary mask",
        ImageArtifactType,
        SourceProjectionRole.PRIMARY_PLANE,
        True,
        True,
        None,
        None,
    )
    MASK = (
        "mask",
        ImageArtifactType,
        SourceProjectionRole.PRIMARY_PLANE,
        True,
        True,
        None,
        None,
    )
    ILLUMINATION_FUNCTION = (
        "illumination function",
        ImageArtifactType,
        SourceProjectionRole.SOURCE_ARTIFACT,
        False,
        False,
        None,
        None,
    )
    OBJECTS = (
        "objects",
        ObjectLabelsArtifactType,
        SourceProjectionRole.SOURCE_ARTIFACT,
        False,
        False,
        None,
        None,
    )

    @classmethod
    def from_literal(cls, value: str) -> "CellProfilerSourceImageType":
        normalized = value.strip().casefold()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported CellProfiler source image type {value!r}.")

    def binding(
        self,
        alias: str,
        selector: SourceSelector,
        origin: SourceBindingOrigin,
        *,
        source_set_role: SourceSetRole,
        explicit_source: ImagePlaneSource | None = None,
    ) -> NamedSourceBinding:
        return NamedSourceBinding(
            alias=alias,
            selector=selector,
            origin=origin,
            artifact_kind=self.artifact_kind,
            source_set_role=source_set_role,
            projection_role=self.projection_role,
            explicit_source=explicit_source,
            load_as_monochrome=self.load_as_monochrome,
            load_as_mask=self.load_as_mask,
            source_channel_axis=self.source_channel_axis,
            source_channel_counts=self.source_channel_counts,
        )


def _source_filter_clauses(criteria: str) -> tuple[SourceFilterClause, ...]:
    decoded = decode_cellprofiler_setting_literal(criteria)
    stripped = decoded.strip()
    if not stripped:
        return ()
    if not _SOURCE_FILTER_SUBJECT_PATTERN.search(decoded):
        return ()
    if _METADATA_PREDICATE_PATTERN.search(decoded):
        normalized = stripped.casefold()
        if normalized.startswith("or ") or "(or " in normalized:
            raise ValueError(
                "CellProfiler selectors cannot disjoin source-path and metadata "
                "predicates in one OpenHCS source selector."
            )
        return tuple(
            _source_filter_clause_from_parts(
                subject_literal=match.group("subject"),
                operator_literal=match.group("operator"),
                negated=bool(match.group("negation")),
                value=match.group("value"),
            )
            for match in _FILTER_CLAUSE_PATTERN.finditer(decoded)
            if match.group("value") != ""
        )

    tokens = tuple(
        match.group(1) for match in _SOURCE_FILTER_TOKEN_PATTERN.finditer(decoded)
    )
    position = 0

    def parse_expression(
        *,
        closes_group: bool,
    ) -> tuple[tuple[SourceFilterClause, ...], ...]:
        nonlocal position
        operator = "and"
        if position < len(tokens) and tokens[position].casefold() in {"and", "or"}:
            operator = tokens[position].casefold()
            position += 1
        terms: list[tuple[tuple[SourceFilterClause, ...], ...]] = []
        while position < len(tokens) and tokens[position] != ")":
            if tokens[position] != "(":
                raise ValueError(
                    f"Unsupported CellProfiler source filter criteria: {criteria!r}."
                )
            position += 1
            if position < len(tokens) and tokens[position].casefold() in {"and", "or"}:
                terms.append(parse_expression(closes_group=True))
                continue
            leaf_tokens: list[str] = []
            while position < len(tokens) and tokens[position] != ")":
                if tokens[position] == "(":
                    raise ValueError(
                        f"Unsupported CellProfiler source filter criteria: {criteria!r}."
                    )
                leaf_tokens.append(tokens[position])
                position += 1
            if position >= len(tokens):
                raise ValueError(
                    f"Unclosed CellProfiler source filter group: {criteria!r}."
                )
            position += 1
            clause = _source_filter_clause_from_tokens(tuple(leaf_tokens))
            terms.append(() if clause is None else ((clause,),))
        if closes_group:
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError(
                    f"Unclosed CellProfiler source filter group: {criteria!r}."
                )
            position += 1
        if not terms:
            return ()
        if operator == "and":
            return tuple(group for term in terms for group in term)
        if any(not term for term in terms):
            return ()
        combined = terms[0]
        for term in terms[1:]:
            combined = tuple(
                tuple(dict.fromkeys((*left, *right)))
                for left in combined
                for right in term
            )
        return combined

    normal_form = parse_expression(closes_group=False)
    if position != len(tokens):
        raise ValueError(
            f"Unexpected CellProfiler source filter tokens: {tokens[position:]!r}."
        )
    clauses: list[SourceFilterClause] = []
    any_group = 0
    for alternatives in normal_form:
        unique_alternatives = tuple(dict.fromkeys(alternatives))
        if len(unique_alternatives) == 1:
            clauses.append(unique_alternatives[0])
            continue
        clauses.extend(
            replace(clause, any_group=any_group) for clause in unique_alternatives
        )
        any_group += 1
    return tuple(clauses)


def _source_filter_clause_from_tokens(
    tokens: tuple[str, ...],
) -> SourceFilterClause | None:
    if len(tokens) < 3 or tokens[1].casefold() not in {"does", "doesnot"}:
        raise ValueError(f"Unsupported CellProfiler source filter clause: {tokens!r}.")
    position = 2
    negated = tokens[1].casefold() == "doesnot"
    negated = negated or (
        position < len(tokens) and tokens[position].casefold() == "not"
    )
    if position < len(tokens) and tokens[position].casefold() == "not":
        position += 1
    if position >= len(tokens):
        raise ValueError(f"Missing CellProfiler source filter operator: {tokens!r}.")
    operator_literal = tokens[position]
    position += 1
    value_tokens = tokens[position:]
    if len(value_tokens) > 1:
        raise ValueError(f"Unsupported CellProfiler source filter value: {tokens!r}.")
    value = None
    if value_tokens:
        token = value_tokens[0]
        value = token[1:-1] if token.startswith('"') else token
        if value == "":
            return None
    return _source_filter_clause_from_parts(
        subject_literal=tokens[0],
        operator_literal=operator_literal,
        negated=negated,
        value=value,
    )


def _source_filter_clause_from_parts(
    *,
    subject_literal: str,
    operator_literal: str,
    negated: bool,
    value: str | None,
) -> SourceFilterClause:
    subject = CellProfilerSourceFilterSubject.from_literal(subject_literal).subject
    normalized_operator = operator_literal.casefold()
    if (
        subject is SourceFilterSubject.EXTENSION
        and not negated
        and normalized_operator.startswith("is")
        and normalized_operator not in {"isimage", "istif"}
    ):
        return SourceFilterClause(
            subject=subject,
            match_type=SourceFilterMatchType.EQUALS,
            value=f".{normalized_operator.removeprefix('is')}",
        )
    operator = CellProfilerSourceFilterOperator.from_literal(normalized_operator)
    return SourceFilterClause(
        subject=subject,
        match_type=operator.generic_match_type(negated=negated),
        value=value,
    )


def _selector_from_rule_criteria(rule_criteria: str) -> SourceSelector:
    return SourceSelector(
        metadata=tuple(
            MetadataSelector(match.group("field"), match.group("value"))
            for match in _METADATA_MATCH_PATTERN.finditer(rule_criteria)
        ),
        filters=_source_filter_clauses(rule_criteria),
    )


def _selector_component_identity(
    selector: SourceSelector,
    component: AllComponents,
) -> ComponentSelector | None:
    declared = tuple(
        candidate
        for candidate in selector.components
        if candidate.component is component
    )
    declared += tuple(
        ComponentSelector(component=component, value=metadata.value)
        for metadata in selector.metadata
        if source_metadata_component(metadata.field) is component
    )
    values = tuple(dict.fromkeys(candidate.value for candidate in declared))
    if len(values) > 1:
        raise ValueError(
            f"Source selector declares conflicting {component.value} values "
            f"{values!r}."
        )
    if not values:
        return None
    return ComponentSelector(component=component, value=values[0])


def _metadata_rules_declare_component(
    config: SourceBindingsConfig,
    component: AllComponents,
) -> bool:
    return any(
        source_metadata_component(field) is component
        for rule in config.metadata_rule_declarations
        for field in rule.capture_fields
    )


def _merge_bindings(
    config: SourceBindingsConfig,
    bindings: Sequence[NamedSourceBinding],
) -> SourceBindingsConfig:
    return replace(config, bindings=(*config.binding_declarations, *bindings))


def _images_filters(module: ModuleBlock) -> tuple[SourceFilterClause, ...]:
    filters = list(
        _source_filter_clauses(module.get_setting("Select the rule criteria", ""))
    )
    filtering_mode = module.get_setting("Filter images?", "").strip().casefold()
    if "images" in filtering_mode and not any(
        clause.match_type is SourceFilterMatchType.IS_IMAGE for clause in filters
    ):
        filters.insert(
            0,
            SourceFilterClause(
                subject=SourceFilterSubject.FILE,
                match_type=SourceFilterMatchType.IS_IMAGE,
            ),
        )
    return tuple(dict.fromkeys(filters))


class ImagesModule(SourceSetupCellProfilerModule):
    module_name = "Images"
    validated = True

    @classmethod
    def contribute_source_bindings(
        cls,
        module: ModuleBlock,
        config: SourceBindingsConfig,
    ) -> SourceBindingsConfig:
        del cls
        return replace(
            config,
            source_filters=(
                *config.source_filter_declarations,
                *_images_filters(module),
            ),
        )


_LOAD_IMAGES_ALIAS = SettingNameFamily(
    "What do you want to call this image in CellProfiler?"
)
_LOAD_IMAGES_MATCH_TEXT = SettingNameFamily(
    "Type the text that these images have in common (case-sensitive)"
)
_LOAD_IMAGES_METADATA_MODE = SettingNameFamily(
    "Do you want to extract metadata from the file name, the subfolder path or both?"
)
_LOAD_IMAGES_FILE_PATTERN = SettingNameFamily(
    "Type the regular expression that finds metadata in the file name"
)
_LOAD_IMAGES_FOLDER_PATTERN = SettingNameFamily(
    "Type the regular expression that finds metadata in the subfolder path"
)


def _load_images_match_type(module: ModuleBlock) -> SourceFilterMatchType:
    mode = module.get_setting("How do you want to load these files?", "")
    normalized = mode.strip().casefold()
    if not normalized or "exact" in normalized:
        return SourceFilterMatchType.CONTAINS
    if "regular" in normalized or "regex" in normalized:
        return SourceFilterMatchType.CONTAINS_REGEX
    raise ValueError(f"Unsupported LoadImages matching mode: {mode!r}.")


def _load_images_filters(
    module: ModuleBlock,
    block: Sequence[ModuleSetting],
) -> tuple[SourceFilterClause, ...]:
    filters: list[SourceFilterClause] = []
    match_text = block_setting_value(block, _LOAD_IMAGES_MATCH_TEXT)
    if match_text:
        filters.append(
            SourceFilterClause(
                subject=SourceFilterSubject.FILE,
                match_type=_load_images_match_type(module),
                value=decode_cellprofiler_setting_literal(match_text),
            )
        )
    if module.get_setting("Do you want to exclude certain files?", "") == "Yes":
        exclusion = module.get_setting(
            "Type the text that the excluded images have in common",
            "",
        )
        if exclusion:
            filters.append(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.DOES_NOT_CONTAIN,
                    value=decode_cellprofiler_setting_literal(exclusion),
                )
            )
    return tuple(filters)


def _load_images_metadata_sources(value: str) -> tuple[MetadataSource, ...]:
    normalized = value.strip().casefold()
    if not normalized or normalized == "none":
        return ()
    sources: list[MetadataSource] = []
    if "file" in normalized or "both" in normalized:
        sources.append(MetadataSource.FILE_NAME)
    if any(token in normalized for token in ("folder", "subfolder", "path", "both")):
        sources.append(MetadataSource.FOLDER_NAME)
    if not sources:
        raise ValueError(f"Unsupported LoadImages metadata extraction mode: {value!r}.")
    return tuple(dict.fromkeys(sources))


def _load_images_metadata_pattern(
    block: Sequence[ModuleSetting],
    source: MetadataSource,
) -> str:
    family = (
        _LOAD_IMAGES_FOLDER_PATTERN
        if source is MetadataSource.FOLDER_NAME
        else _LOAD_IMAGES_FILE_PATTERN
    )
    pattern = decode_cellprofiler_setting_literal(
        block_setting_value_by_prefix(block, family)
    )
    if not pattern or pattern.strip().casefold() == "none":
        raise ValueError(
            "LoadImages metadata extraction requires a non-empty "
            f"{source.value} regular expression."
        )
    return pattern


class LoadImagesModule(SourceSetupCellProfilerModule):
    module_name = "LoadImages"
    validated = True

    @classmethod
    def contribute_source_bindings(
        cls,
        module: ModuleBlock,
        config: SourceBindingsConfig,
    ) -> SourceBindingsConfig:
        del cls
        file_type = module.get_setting("What type of files are you loading?", "")
        if file_type and "individual" not in file_type.strip().casefold():
            raise ValueError(
                "LoadImages source declarations require individual images, got "
                f"{file_type!r}."
            )

        blocks = repeating_setting_blocks(
            module.iter_settings(),
            start_name=_LOAD_IMAGES_MATCH_TEXT,
        )
        bindings: list[NamedSourceBinding] = []
        metadata_rules = list(config.metadata_rule_declarations)
        for block in blocks:
            alias = block_setting_value(block, _LOAD_IMAGES_ALIAS)
            if not alias:
                continue
            filters = _load_images_filters(module, block)
            selector = SourceSelector(filters=filters)
            source_type = CellProfilerSourceImageType.GRAYSCALE
            bindings.append(
                source_type.binding(
                    alias,
                    selector,
                    SourceBindingOrigin.PIPELINE_START,
                    source_set_role=SourceSetRole.MATCHED,
                )
            )
            for source in _load_images_metadata_sources(
                block_setting_value(block, _LOAD_IMAGES_METADATA_MODE)
            ):
                metadata_rules.append(
                    MetadataExtractionRule(
                        source=source,
                        pattern=_load_images_metadata_pattern(block, source),
                        filters=filters,
                    )
                )

        grouping_fields = config.grouping_metadata_fields
        if (
            module.get_setting("Do you want to group image sets by metadata?", "")
            == "Yes"
        ):
            grouping_fields = tuple(
                field.strip()
                for field in re.split(
                    "[,;]",
                    module.get_setting(
                        "What metadata fields do you want to group by?",
                        "",
                    ),
                )
                if field.strip()
            )
        return replace(
            _merge_bindings(config, bindings),
            metadata_rules=tuple(metadata_rules),
            grouping_metadata_fields=grouping_fields,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerMetadataPatternBlock:
    """Typed access to one CellProfiler path-metadata regex block."""

    block: Sequence[ModuleSetting]
    source: MetadataSource

    @property
    def pattern(self) -> str:
        if self.source is MetadataSource.FOLDER_NAME:
            value = block_setting_value(
                self.block,
                "Regular expression to extract from folder name",
            ) or _legacy_regex_value(self.block, index=1)
        else:
            value = block_setting_value(
                self.block,
                "Regular expression to extract from file name",
            ) or _legacy_regex_value(self.block, index=0)
        return decode_cellprofiler_setting_literal(value)

    @property
    def required_pattern(self) -> str:
        if not self.pattern:
            raise ValueError(
                "CellProfiler path metadata extraction requires a non-empty "
                f"{self.source.value} regular expression."
            )
        return self.pattern


def _legacy_regex_value(
    block: Sequence[ModuleSetting],
    *,
    index: int,
) -> str:
    values = tuple(
        setting.value for setting in block if setting.name == "Regular expression"
    )
    return values[index] if index < len(values) else ""


def _metadata_extraction_enabled(module: ModuleBlock) -> bool:
    value = decode_cellprofiler_setting_literal(module.get_setting("Extract metadata?"))
    if not value:
        raise ValueError(
            "CellProfiler Metadata requires an 'Extract metadata?' setting."
        )
    return parse_cellprofiler_bool(value)


def _metadata_import_table(block: Sequence[ModuleSetting]) -> ImportedMetadataTable:
    location_value = decode_cellprofiler_setting_literal(
        block_setting_value(block, "Metadata file location")
    ).strip()
    if not location_value:
        location = None
    elif "|" not in location_value:
        location = location_value
    else:
        location_kind, location_path = location_value.split("|", 1)
        if location_kind.strip().casefold() != "default input folder":
            raise ValueError(
                "Metadata imported tables require Default Input Folder, got "
                f"{location_kind!r}."
            )
        location = location_path.strip() or None
    file_name = decode_cellprofiler_setting_literal(
        block_setting_value(block, "Metadata file name")
    ).strip()
    if file_name:
        location = file_name if location is None else str(Path(location) / file_name)

    raw_joins = block_setting_value(block, "Match file and image metadata")
    joins: list[ImportedMetadataJoin] = []
    if raw_joins:
        try:
            records = ast.literal_eval(decode_cellprofiler_setting_literal(raw_joins))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "Invalid Metadata 'Match file and image metadata' value: "
                f"{raw_joins!r}."
            ) from exc
        if not isinstance(records, list):
            raise TypeError("Metadata imported-table joins must parse to a list.")
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError(
                    "Metadata imported-table join records must be mappings."
                )
            image_field = record.get("Image Metadata")
            imported_field = record.get("CSV Metadata")
            if image_field is not None and imported_field is not None:
                joins.append(
                    ImportedMetadataJoin(
                        image_metadata_field=str(image_field),
                        imported_metadata_field=str(imported_field),
                    )
                )
    return ImportedMetadataTable(location=location, joins=tuple(joins))


class MetadataModule(SourceSetupCellProfilerModule):
    module_name = "Metadata"
    validated = True

    class ExtractionMethod(str, Enum):
        def __new__(
            cls,
            literal: str,
            supported_when_enabled: bool,
        ):
            member = str.__new__(cls, literal)
            member._value_ = literal
            member.supported_when_enabled = supported_when_enabled
            return member

        PATH = ("Extract from file/folder names", True)
        IMPORTED = ("Import from file", True)
        IMAGE_FILE_HEADERS = ("Extract from image file headers", False)

        def is_active(self, *, extraction_enabled: bool) -> bool:
            """Return whether this declared block contributes extraction semantics."""

            if not extraction_enabled:
                return False
            if not self.supported_when_enabled:
                raise ValueError(
                    "Unsupported active CellProfiler Metadata extraction method: "
                    f"{self.value!r}."
                )
            return True

    class ExtractionSource(Enum):
        def __new__(cls, literal: str, source: MetadataSource):
            member = object.__new__(cls)
            member._value_ = literal
            member.source = source
            return member

        FILE_NAME = ("File name", MetadataSource.FILE_NAME)
        FOLDER_NAME = ("Folder name", MetadataSource.FOLDER_NAME)

    class ExtractionScope(str, Enum):
        ALL_IMAGES = "All images"
        MATCHING_RULES = "Images matching a rule"

    class DataTypeMode(str, Enum):
        TEXT = "Text"
        CHOOSE = "Choose for each"

    class FieldDtype(Enum):
        def __new__(
            cls,
            literal: str,
            python_type: type[object] | None,
        ):
            member = object.__new__(cls)
            member._value_ = literal
            member.python_type = python_type
            return member

        TEXT = ("text", str)
        INTEGER = ("integer", int)
        FLOAT = ("float", float)
        NONE = ("none", None)

    @classmethod
    def metadata_fields(
        cls,
        module: ModuleBlock,
        *,
        rules: Sequence[MetadataExtractionRule],
        tables: Sequence[ImportedMetadataTable],
    ) -> tuple[FieldSpec, ...]:
        """Return the exact metadata schema declared by this setup module."""

        if not _metadata_extraction_enabled(module):
            return (CellProfilerSourceMetadataField.FILE_LOCATION.field_spec(),)

        mode = cls.DataTypeMode(
            decode_cellprofiler_setting_literal(
                module.get_setting("Metadata data type", "Text")
            ).strip()
        )

        declared_types: Mapping[object, object] = {}
        if mode is cls.DataTypeMode.CHOOSE:
            raw_types = decode_cellprofiler_setting_literal(
                module.get_setting("Metadata types", "{}")
            )
            try:
                parsed_types = json.loads(raw_types)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid CellProfiler Metadata types declaration: {raw_types!r}."
                ) from exc
            if not isinstance(parsed_types, Mapping):
                raise TypeError("CellProfiler Metadata types must decode to an object.")
            declared_types = parsed_types

        imported_join_fields = {
            join.imported_metadata_field for table in tables for join in table.joins
        }
        field_names = tuple(
            dict.fromkeys(
                (
                    *(field.field_name for field in CellProfilerSourceMetadataField),
                    *(name for rule in rules for name in rule.capture_fields),
                    *(
                        name
                        for name in declared_types
                        if str(name) not in imported_join_fields
                    ),
                )
            )
        )
        reserved_fields = {
            field.field_name: field for field in CellProfilerSourceMetadataField
        }
        fields: list[FieldSpec] = []
        for field_name in field_names:
            if mode is cls.DataTypeMode.TEXT:
                fields.append(FieldSpec(field_name, str, required=False))
                continue
            reserved = reserved_fields.get(field_name)
            if reserved is not None:
                fields.append(reserved.field_spec())
                continue
            declared = cls.FieldDtype(
                str(declared_types.get(field_name, "text")).strip().casefold()
            )
            if declared.python_type is not None:
                fields.append(
                    FieldSpec(field_name, declared.python_type, required=False)
                )
        return tuple(fields)

    @classmethod
    def contribute_source_bindings(
        cls,
        module: ModuleBlock,
        config: SourceBindingsConfig,
    ) -> SourceBindingsConfig:
        enabled = _metadata_extraction_enabled(module)
        rules = list(config.metadata_rule_declarations)
        tables = list(config.imported_metadata_tables)
        for block in repeating_setting_blocks(
            module.iter_settings(),
            start_name="Metadata extraction method",
        ):
            method = cls.ExtractionMethod(
                block_setting_value(block, "Metadata extraction method").strip()
            )
            if not method.is_active(extraction_enabled=enabled):
                continue
            if method is cls.ExtractionMethod.IMPORTED:
                tables.append(_metadata_import_table(block))
                continue
            source = cls.ExtractionSource(
                block_setting_value(block, "Metadata source").strip()
            ).source
            pattern_block = CellProfilerMetadataPatternBlock(block, source)
            extraction_scope = cls.ExtractionScope(
                decode_cellprofiler_setting_literal(
                    block_setting_value(block, "Extract metadata from")
                ).strip()
            )
            rules.append(
                MetadataExtractionRule(
                    source=source,
                    pattern=pattern_block.required_pattern,
                    filters=(
                        _source_filter_clauses(
                            block_setting_value(
                                block,
                                "Select the filtering criteria",
                            )
                        )
                        if extraction_scope is cls.ExtractionScope.MATCHING_RULES
                        else ()
                    ),
                )
            )
        metadata_fields = FieldSpec.merge_exact(
            (
                config.metadata_fields,
                cls.metadata_fields(module, rules=rules, tables=tables),
            ),
            context="CellProfiler Metadata source fields",
        )
        return replace(
            config,
            metadata_rules=tuple(rules),
            imported_metadata_tables=tuple(tables),
            metadata_fields=metadata_fields,
        )


def _positive_float_setting(
    module: ModuleBlock,
    name: str,
    default: float,
) -> float:
    value = module.get_setting(name, str(default)).strip()
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{module.name} setting {name!r} must be a float, got {value!r}."
        ) from exc
    if parsed <= 0:
        raise ValueError(
            f"{module.name} setting {name!r} must be positive, got {value!r}."
        )
    return parsed


def _names_and_types_count(
    module: ModuleBlock,
    setting_name: str,
    *,
    default: int | None = None,
) -> int:
    raw_count = module.get_setting(setting_name, "").strip()
    if not raw_count and default is not None:
        return default
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise ValueError(
            f"NamesAndTypes {setting_name!r} must be an integer, got {raw_count!r}."
        ) from exc
    if count < 0:
        raise ValueError(f"NamesAndTypes {setting_name!r} cannot be negative.")
    return count


def _settings_after(
    module: ModuleBlock,
    setting_name: str | SettingNameFamily,
) -> tuple[ModuleSetting, ...]:
    records = tuple(module.iter_settings())
    for index, setting in enumerate(records):
        if setting_name_matches(setting.name, setting_name):
            return records[index + 1 :]
    raise ValueError(
        f"{module.name}({module.module_num}) has no {setting_name!r} setting."
    )


def _required_block_setting(
    module: ModuleBlock,
    block: Sequence[ModuleSetting],
    setting_name: str | SettingNameFamily,
) -> str:
    values = tuple(
        setting.value.strip()
        for setting in block
        if setting_name_matches(setting.name, setting_name)
    )
    if len(values) != 1 or not values[0]:
        raise ValueError(
            f"{module.name}({module.module_num}) repeated block requires "
            f"exactly one non-empty setting {setting_name!r}, got {len(values)}."
        )
    return values[0]


def _declared_semantic_blocks(
    records: Sequence[ModuleSetting],
    count: int,
    *,
    semantic_settings: Sequence[str | SettingNameFamily],
    is_complete: Callable[[Sequence[ModuleSetting]], bool],
) -> tuple[tuple[tuple[ModuleSetting, ...], ...], tuple[ModuleSetting, ...]]:
    """Parse exactly declared repeated blocks independent of field order."""

    if count == 0:
        return (), tuple(records)
    blocks: list[tuple[ModuleSetting, ...]] = []
    current: list[ModuleSetting] = []
    for index, setting in enumerate(records):
        if not current and not any(
            setting_name_matches(setting.name, family)
            for family in semantic_settings
        ):
            continue
        current.append(setting)
        if not is_complete(current):
            continue
        blocks.append(tuple(current))
        current = []
        if len(blocks) == count:
            return tuple(blocks), tuple(records[index + 1 :])
    if current:
        blocks.append(tuple(current))
    return tuple(blocks), ()


def _setting_count(
    block: Sequence[ModuleSetting],
    setting_name: str | SettingNameFamily,
) -> int:
    return sum(
        setting_name_matches(setting.name, setting_name) for setting in block
    )


def _image_plane_source(value: str) -> ImagePlaneSource:
    fields = value.split(" ")
    if len(fields) > 4:
        raise ValueError(
            "NamesAndTypes 'Single image location' must contain URL, series, "
            "index, and channel fields."
        )
    fields.extend("" for _ in range(4 - len(fields)))
    uri, series, index, channel = fields
    if not uri:
        raise ValueError("NamesAndTypes 'Single image location' URL cannot be empty.")
    return ImagePlaneSource(
        uri=uri,
        series=series or None,
        index=index or None,
        channel=channel or None,
    )


def _match_metadata_dimensions(value: str) -> tuple[SourceBindingMatchDimension, ...]:
    try:
        records = ast.literal_eval(decode_cellprofiler_setting_literal(value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Invalid NamesAndTypes 'Match metadata' value: {value!r}."
        ) from exc
    if not isinstance(records, list):
        raise TypeError("NamesAndTypes 'Match metadata' must parse to a list.")
    dimensions: list[SourceBindingMatchDimension] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("NamesAndTypes match metadata entries must be mappings.")
        fields = tuple(
            SourceBindingMatchField(alias=str(alias), metadata_field=str(field))
            for alias, field in record.items()
            if field is not None
        )
        if fields:
            dimensions.append(SourceBindingMatchDimension(fields=fields))
    return tuple(dimensions)


def _names_and_types_match_plan(
    module: ModuleBlock,
) -> SourceBindingMatchPlan | None:
    method_values = tuple(
        value.strip()
        for value in module.get_setting_values("Image set matching method")
        if value.strip()
    )
    if not method_values:
        return None
    methods = tuple(
        CellProfilerSourceMatchMethod.from_literal(value).method
        for value in method_values
    )
    if any(method is not methods[0] for method in methods[1:]):
        raise ValueError("NamesAndTypes declared conflicting matching methods.")
    if methods[0] is SourceBindingMatchMethod.ORDER:
        return SourceBindingMatchPlan(method=methods[0])
    metadata_values = tuple(
        value.strip()
        for value in module.get_setting_values("Match metadata")
        if value.strip()
    )
    dimensions = ()
    if metadata_values:
        if len(metadata_values) != 1:
            raise ValueError(
                "NamesAndTypes Match metadata requires one shared declaration."
            )
        dimensions = _match_metadata_dimensions(metadata_values[0])
    return SourceBindingMatchPlan(method=methods[0], dimensions=dimensions)


class NamesAndTypesModule(SourceSetupCellProfilerModule):
    module_name = "NamesAndTypes"
    validated = True
    image_alias_settings: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Name to assign these images"
    )
    object_alias_settings: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Name to assign these objects"
    )
    single_image_alias_settings: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Name to assign this image"
    )
    single_image_location_settings: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Single image location"
    )
    image_type_settings: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the image type"
    )
    rule_criteria_settings: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the rule criteria",
    )
    match_metadata_settings: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Match metadata"
    )

    @classmethod
    def _assignment_block_is_complete(
        cls,
        block: Sequence[ModuleSetting],
    ) -> bool:
        return (
            _setting_count(block, cls.rule_criteria_settings) == 1
            and _setting_count(block, cls.image_type_settings) == 1
            and (
                _setting_count(block, cls.image_alias_settings) == 1
                or _setting_count(block, cls.object_alias_settings) == 1
            )
        )

    @classmethod
    def _single_block_is_complete(
        cls,
        block: Sequence[ModuleSetting],
    ) -> bool:
        return (
            _setting_count(block, cls.single_image_location_settings) == 1
            and _setting_count(block, cls.image_type_settings) == 1
            and (
                _setting_count(block, cls.single_image_alias_settings) == 1
                or _setting_count(block, cls.object_alias_settings) == 1
            )
        )

    @classmethod
    def contribute_source_bindings(
        cls,
        module: ModuleBlock,
        config: SourceBindingsConfig,
    ) -> SourceBindingsConfig:
        assignment_count = _names_and_types_count(module, "Assignments count")
        single_count = _names_and_types_count(
            module,
            "Single images count",
            default=0,
        )
        repeated_records = _settings_after(module, "Assignments count")
        assignment_blocks, single_records = _declared_semantic_blocks(
            repeated_records,
            assignment_count,
            semantic_settings=(
                cls.rule_criteria_settings,
                cls.image_type_settings,
                cls.image_alias_settings,
                cls.object_alias_settings,
            ),
            is_complete=cls._assignment_block_is_complete,
        )
        single_blocks, _trailing_records = _declared_semantic_blocks(
            single_records,
            single_count,
            semantic_settings=(
                cls.single_image_location_settings,
                cls.image_type_settings,
                cls.single_image_alias_settings,
                cls.object_alias_settings,
            ),
            is_complete=cls._single_block_is_complete,
        )
        if len(assignment_blocks) != assignment_count:
            raise ValueError(
                f"NamesAndTypes declared {assignment_count} assignments but "
                f"provided {len(assignment_blocks)} assignment blocks."
            )
        if len(single_blocks) != single_count:
            raise ValueError(
                f"NamesAndTypes declared {single_count} single images but "
                f"provided {len(single_blocks)} single-image blocks."
            )

        bindings: list[NamedSourceBinding] = []
        explicit_sources: list[ImagePlaneSource] = []
        metadata_declares_channel = _metadata_rules_declare_component(
            config,
            AllComponents.CHANNEL,
        )
        used_channel_values: set[str] = set()
        next_channel_index = 1
        for block in assignment_blocks:
            source_type = CellProfilerSourceImageType.from_literal(
                _required_block_setting(
                    module,
                    block,
                    cls.image_type_settings,
                )
            )
            alias_setting = (
                cls.object_alias_settings
                if source_type.artifact_kind is ObjectLabelsArtifactType
                else cls.image_alias_settings
            )
            alias = _required_block_setting(module, block, alias_setting)
            selector = _selector_from_rule_criteria(
                _required_block_setting(
                    module,
                    block,
                    cls.rule_criteria_settings,
                )
            )
            binding = source_type.binding(
                alias,
                selector,
                SourceBindingOrigin.PIPELINE_START,
                source_set_role=SourceSetRole.MATCHED,
            )
            if source_type.projection_role is SourceProjectionRole.PRIMARY_PLANE:
                channel_identity = _selector_component_identity(
                    selector,
                    AllComponents.CHANNEL,
                )
                if channel_identity is None and not metadata_declares_channel:
                    while str(next_channel_index) in used_channel_values:
                        next_channel_index += 1
                    channel_identity = ComponentSelector(
                        component=AllComponents.CHANNEL,
                        value=str(next_channel_index),
                    )
                if channel_identity is not None:
                    used_channel_values.add(channel_identity.value)
                    next_channel_index += 1
                    binding = replace(
                        binding,
                        component_identity=(channel_identity,),
                    )
            bindings.append(binding)

        for block in single_blocks:
            source_type = CellProfilerSourceImageType.from_literal(
                _required_block_setting(
                    module,
                    block,
                    cls.image_type_settings,
                )
            )
            alias_setting = (
                cls.object_alias_settings
                if source_type.artifact_kind is ObjectLabelsArtifactType
                else cls.single_image_alias_settings
            )
            alias = _required_block_setting(module, block, alias_setting)
            explicit_source = _image_plane_source(
                _required_block_setting(
                    module,
                    block,
                    cls.single_image_location_settings,
                )
            )
            binding = source_type.binding(
                alias,
                SourceSelector(),
                SourceBindingOrigin.PIPELINE_START,
                source_set_role=SourceSetRole.BROADCAST,
                explicit_source=explicit_source,
            )
            if source_type.projection_role is SourceProjectionRole.PRIMARY_PLANE:
                while str(next_channel_index) in used_channel_values:
                    next_channel_index += 1
                channel_identity = ComponentSelector(
                    component=AllComponents.CHANNEL,
                    value=str(next_channel_index),
                )
                used_channel_values.add(channel_identity.value)
                next_channel_index += 1
                binding = replace(
                    binding,
                    component_identity=(channel_identity,),
                )
            bindings.append(binding)
            explicit_sources.append(explicit_source)

        source_stack_components = config.source_stack_components
        if module.get_setting("Process as 3D?", "No").strip().casefold() == "yes":
            source_stack_components = tuple(
                dict.fromkeys((*source_stack_components, AllComponents.Z_INDEX))
            )
        voxel_spacing = SourceVoxelSpacing.from_cellprofiler_xyz(
            x=_positive_float_setting(
                module,
                "Relative pixel spacing in X",
                1.0,
            ),
            y=_positive_float_setting(
                module,
                "Relative pixel spacing in Y",
                1.0,
            ),
            z=_positive_float_setting(
                module,
                "Relative pixel spacing in Z",
                1.0,
            ),
        )
        return replace(
            _merge_bindings(config, bindings),
            match_plan=_names_and_types_match_plan(module),
            image_plane_sources=tuple(
                dict.fromkeys((*config.image_plane_sources, *explicit_sources))
            ),
            source_stack_components=source_stack_components,
            source_voxel_spacing=voxel_spacing,
        )


class GroupsModule(SourceSetupCellProfilerModule):
    module_name = "Groups"
    validated = True

    @classmethod
    def contribute_source_bindings(
        cls,
        module: ModuleBlock,
        config: SourceBindingsConfig,
    ) -> SourceBindingsConfig:
        del cls
        if module.get_setting("Do you want to group your images?", "No") != "Yes":
            return config
        return replace(
            config,
            grouping_metadata_fields=tuple(
                setting.value for setting in module.iter_settings("Metadata category")
            ),
        )
