"""Source path matching primitives shared by source bindings and materializers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import LOADABLE_IMAGE_EXTENSIONS
from openhcs.core.source_bindings import (
    MetadataExtractionRule,
    MetadataSource,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
)


@dataclass(frozen=True, slots=True)
class SourceFilterMatchRequest:
    """Typed request for one source-filter match evaluation."""

    file_path: str
    clause: SourceFilterClause
    target: str


def _string_contains(target: str, value: str) -> bool:
    return value in target


def _string_does_not_contain(target: str, value: str) -> bool:
    return value not in target


def _string_contains_regex(target: str, value: str) -> bool:
    return re.search(value, target) is not None


def _string_does_not_contain_regex(target: str, value: str) -> bool:
    return re.search(value, target) is None


def _string_equals(target: str, value: str) -> bool:
    return target == value


def _string_does_not_equal(target: str, value: str) -> bool:
    return target != value


def _string_starts_with(target: str, value: str) -> bool:
    return target.startswith(value)


def _string_does_not_start_with(target: str, value: str) -> bool:
    return not target.startswith(value)


def _string_ends_with(target: str, value: str) -> bool:
    return target.endswith(value)


def _string_does_not_end_with(target: str, value: str) -> bool:
    return not target.endswith(value)


def is_image_path(file_path: str) -> bool:
    """Return whether the path extension is a loadable image source."""

    suffix = Path(file_path).suffix.lower()
    return suffix in LOADABLE_IMAGE_EXTENSIONS


def is_tif_path(file_path: str) -> bool:
    """Return whether the path extension is a TIFF source."""

    return Path(file_path).suffix.lower() in {".tif", ".tiff"}


class SourceFilterMatcher(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for typed source-filter match behavior."""

    __registry_key__ = "match_type_key"
    __skip_if_no_key__ = True
    match_type: ClassVar[SourceFilterMatchType | None] = None
    match_type_key: ClassVar[str | None] = None

    @classmethod
    def for_match_type(
        cls,
        match_type: SourceFilterMatchType,
    ) -> "SourceFilterMatcher":
        return cls.__registry__[match_type.value]()

    @abstractmethod
    def matches(self, request: SourceFilterMatchRequest) -> bool:
        """Return whether one file path satisfies the filter clause."""


class ValuePredicateSourceFilterMatcher(SourceFilterMatcher):
    """Declarative matcher for source-filter clauses with scalar values."""

    value_predicate: ClassVar[Callable[[str, str], bool]]

    def matches(self, request: SourceFilterMatchRequest) -> bool:
        return type(self).value_predicate(
            request.target,
            _require_filter_value(request.clause),
        )


class PathPredicateSourceFilterMatcher(SourceFilterMatcher):
    """Declarative matcher for source-filter clauses that inspect the path."""

    path_predicate: ClassVar[Callable[[str], bool]]

    def matches(self, request: SourceFilterMatchRequest) -> bool:
        if request.clause.value is not None:
            raise ValueError(
                f"{request.clause.match_type.value} source filters do not accept "
                "a scalar clause value."
            )
        return type(self).path_predicate(request.file_path)


class ContainsSourceFilterMatcher(ValuePredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.CONTAINS
    match_type_key = SourceFilterMatchType.CONTAINS.value
    value_predicate = staticmethod(_string_contains)


class DoesNotContainSourceFilterMatcher(ValuePredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.DOES_NOT_CONTAIN
    match_type_key = SourceFilterMatchType.DOES_NOT_CONTAIN.value
    value_predicate = staticmethod(_string_does_not_contain)


class ContainsRegexSourceFilterMatcher(ValuePredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.CONTAINS_REGEX
    match_type_key = SourceFilterMatchType.CONTAINS_REGEX.value
    value_predicate = staticmethod(_string_contains_regex)


class DoesNotContainRegexSourceFilterMatcher(ValuePredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.DOES_NOT_CONTAIN_REGEX
    match_type_key = SourceFilterMatchType.DOES_NOT_CONTAIN_REGEX.value
    value_predicate = staticmethod(_string_does_not_contain_regex)


class EqualsSourceFilterMatcher(ValuePredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.EQUALS
    match_type_key = SourceFilterMatchType.EQUALS.value
    value_predicate = staticmethod(_string_equals)


class DoesNotEqualSourceFilterMatcher(ValuePredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.DOES_NOT_EQUAL
    match_type_key = SourceFilterMatchType.DOES_NOT_EQUAL.value
    value_predicate = staticmethod(_string_does_not_equal)


class StartsWithSourceFilterMatcher(ValuePredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.STARTS_WITH
    match_type_key = SourceFilterMatchType.STARTS_WITH.value
    value_predicate = staticmethod(_string_starts_with)


class DoesNotStartWithSourceFilterMatcher(ValuePredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.DOES_NOT_START_WITH
    match_type_key = SourceFilterMatchType.DOES_NOT_START_WITH.value
    value_predicate = staticmethod(_string_does_not_start_with)


class EndsWithSourceFilterMatcher(ValuePredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.ENDS_WITH
    match_type_key = SourceFilterMatchType.ENDS_WITH.value
    value_predicate = staticmethod(_string_ends_with)


class DoesNotEndWithSourceFilterMatcher(ValuePredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.DOES_NOT_END_WITH
    match_type_key = SourceFilterMatchType.DOES_NOT_END_WITH.value
    value_predicate = staticmethod(_string_does_not_end_with)


class IsImageSourceFilterMatcher(PathPredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.IS_IMAGE
    match_type_key = SourceFilterMatchType.IS_IMAGE.value
    path_predicate = staticmethod(is_image_path)


class IsTifSourceFilterMatcher(PathPredicateSourceFilterMatcher):
    match_type = SourceFilterMatchType.IS_TIF
    match_type_key = SourceFilterMatchType.IS_TIF.value
    path_predicate = staticmethod(is_tif_path)


class SourceFilterTargetResolver(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for source-filter target text resolution."""

    __registry_key__ = "subject_key"
    __skip_if_no_key__ = True
    subject: ClassVar[SourceFilterSubject | None] = None
    subject_key: ClassVar[str | None] = None

    @classmethod
    def for_subject(
        cls,
        subject: SourceFilterSubject,
    ) -> "SourceFilterTargetResolver":
        return cls.__registry__[subject.value]()

    @abstractmethod
    def resolve_text(self, file_path: str) -> str:
        """Return the subject-specific text inspected by one filter clause."""


class FileSourceFilterTargetResolver(SourceFilterTargetResolver):
    subject = SourceFilterSubject.FILE
    subject_key = SourceFilterSubject.FILE.value

    def resolve_text(self, file_path: str) -> str:
        return Path(file_path).name


class DirectorySourceFilterTargetResolver(SourceFilterTargetResolver):
    subject = SourceFilterSubject.DIRECTORY
    subject_key = SourceFilterSubject.DIRECTORY.value

    def resolve_text(self, file_path: str) -> str:
        return str(Path(file_path).parent)


class ExtensionSourceFilterTargetResolver(SourceFilterTargetResolver):
    subject = SourceFilterSubject.EXTENSION
    subject_key = SourceFilterSubject.EXTENSION.value

    def resolve_text(self, file_path: str) -> str:
        return Path(file_path).suffix.lower()


def metadata_from_rules(
    file_path: str,
    metadata_rules: tuple[MetadataExtractionRule, ...],
) -> dict[str, str]:
    """Extract metadata fields from one source path using typed rules."""

    extracted: dict[str, str] = {}
    for rule in metadata_rules:
        if not rule_filters_match(file_path, rule.filters):
            continue
        target = metadata_source_text(file_path, rule.source)
        match = re.search(rule.pattern, target)
        if match is None:
            continue
        merge_source_metadata(
            extracted,
            {
                key: str(value)
                for key, value in match.groupdict().items()
                if value is not None
            },
            path=file_path,
        )
    return extracted


def metadata_source_text(
    file_path: str,
    source: MetadataSource,
) -> str:
    """Return the path text inspected by one metadata extraction rule."""

    path = Path(file_path)
    if source is MetadataSource.FOLDER_NAME:
        return str(path.parent)
    return path.name


def rule_filters_match(
    file_path: str,
    filters: tuple[SourceFilterClause, ...],
) -> bool:
    """Return whether one source path satisfies metadata-rule filters."""

    return source_filters_match(file_path, filters)


def source_filters_match(
    file_path: str,
    filters: tuple[SourceFilterClause, ...],
) -> bool:
    """Return whether one source path satisfies all source-filter clauses."""

    return all(filter_clause_matches(file_path, clause) for clause in filters)


def filter_clause_matches(
    file_path: str,
    clause: SourceFilterClause,
) -> bool:
    """Return whether one source path satisfies one source-filter clause."""

    target = SourceFilterTargetResolver.for_subject(clause.subject).resolve_text(file_path)
    return SourceFilterMatcher.for_match_type(clause.match_type).matches(
        SourceFilterMatchRequest(
            file_path=file_path,
            clause=clause,
            target=target,
        )
    )


def merge_source_metadata(
    target: dict[str, Any],
    additions: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """Merge extracted metadata into a target map, failing on conflicts."""

    for key, value in additions.items():
        existing = target.get(key)
        normalized_value = str(value)
        if existing is not None and str(existing) != normalized_value:
            raise RuntimeError(
                f"Conflicting metadata field '{key}' while parsing source candidate "
                f"{path!r}: {existing!r} != {normalized_value!r}."
            )
        target[key] = normalized_value


def _require_filter_value(clause: SourceFilterClause) -> str:
    if clause.value is None:
        raise ValueError(
            "SourceFilterClause.value must be set unless match_type is IS_IMAGE."
        )
    return clause.value
