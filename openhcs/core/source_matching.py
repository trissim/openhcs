"""Source path matching primitives shared by source bindings and materializers."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Callable, ClassVar, Mapping, TypeAlias

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents, LOADABLE_IMAGE_EXTENSIONS
from openhcs.core.source_bindings import (
    MetadataExtractionRule,
    MetadataSource,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
)
from openhcs.core.source_metadata import (
    ORIGINAL_SOURCE_METADATA_FIELD,
    OriginalSourceMetadata,
    SourceMetadataMapping,
    SourceMetadataRoleView,
    SourceMetadataScalar,
    SourceMetadataValue,
    canonical_path_metadata_value,
    path_metadata_values_equivalent,
)
from openhcs.core.source_path_identity import (
    source_path_identity_key,
    source_paths_equal,
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

    suffix = os.path.splitext(file_path)[1].lower()
    return suffix in LOADABLE_IMAGE_EXTENSIONS


def is_tif_path(file_path: str) -> bool:
    """Return whether the path extension is a TIFF source."""

    return os.path.splitext(file_path)[1].lower() in {".tif", ".tiff"}


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
        return os.path.basename(file_path)


class DirectorySourceFilterTargetResolver(SourceFilterTargetResolver):
    subject = SourceFilterSubject.DIRECTORY
    subject_key = SourceFilterSubject.DIRECTORY.value

    def resolve_text(self, file_path: str) -> str:
        return os.path.dirname(file_path)


class ExtensionSourceFilterTargetResolver(SourceFilterTargetResolver):
    subject = SourceFilterSubject.EXTENSION
    subject_key = SourceFilterSubject.EXTENSION.value

    def resolve_text(self, file_path: str) -> str:
        return os.path.splitext(file_path)[1].lower()


def metadata_from_rules(
    file_path: str,
    metadata_rules: tuple[MetadataExtractionRule, ...],
    *,
    filter_path: str | None = None,
) -> dict[str, str]:
    """Extract metadata fields from one source path using typed rules."""

    extracted: dict[str, str] = {}
    filter_candidate = file_path if filter_path is None else filter_path
    for rule in metadata_rules:
        if not rule_filters_match(filter_candidate, rule.filters):
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

    return _source_filters_match_cached(str(file_path), tuple(filters))


@lru_cache(maxsize=65536)
def _source_filters_match_cached(
    file_path: str,
    filters: tuple[SourceFilterClause, ...],
) -> bool:
    return all(filter_clause_matches(file_path, clause) for clause in filters)


def filter_clause_matches(
    file_path: str,
    clause: SourceFilterClause,
) -> bool:
    """Return whether one source path satisfies one source-filter clause."""

    target = SourceFilterTargetResolver.for_subject(clause.subject).resolve_text(
        file_path
    )
    return SourceFilterMatcher.for_match_type(clause.match_type).matches(
        SourceFilterMatchRequest(
            file_path=file_path,
            clause=clause,
            target=target,
        )
    )


class SourceImageSetComponentRole(Enum):
    """Semantic role of a source metadata component within an image set."""

    IMAGE_SET_AXIS = "image_set_axis"
    IMAGE_PLANE_MEMBER = "image_plane_member"


@dataclass(frozen=True, slots=True)
class SourceImageSetIdentityPolicy:
    """Nominal policy for reducing source plane metadata to image-set identity."""

    plane_member_components: frozenset[AllComponents] = field(
        default_factory=lambda: frozenset((AllComponents.CHANNEL,))
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plane_member_components",
            frozenset(self.plane_member_components),
        )

    def role(self, component: AllComponents) -> SourceImageSetComponentRole:
        """Return whether a component identifies the image set or a plane in it."""
        if component in self.plane_member_components:
            return SourceImageSetComponentRole.IMAGE_PLANE_MEMBER
        return SourceImageSetComponentRole.IMAGE_SET_AXIS

    def identity_components(self) -> tuple[AllComponents, ...]:
        """Return generated source components that identify one image set."""
        return tuple(
            component
            for component in AllComponents
            if self.role(component) is SourceImageSetComponentRole.IMAGE_SET_AXIS
        )

    @classmethod
    def from_plane_member_fields(
        cls,
        fields: frozenset[str],
    ) -> "SourceImageSetIdentityPolicy":
        """Return an identity policy whose plane members include declared fields."""

        base_policy = cls()
        plane_member_components = set(base_policy.plane_member_components)
        for field in fields:
            component = source_metadata_component(field)
            if component is None:
                raise ValueError(
                    "Source image-set identity policy cannot resolve declared "
                    f"plane-member field {field!r} to a source component."
                )
            plane_member_components.add(component)
        return cls(frozenset(plane_member_components))

    def is_identity_component(self, component: AllComponents) -> bool:
        """Return whether a metadata component participates in image-set identity."""
        return self.role(component) is SourceImageSetComponentRole.IMAGE_SET_AXIS


@dataclass(frozen=True, slots=True)
class SourceImageSetIdentity:
    """Image-set identity from source metadata under a typed component policy."""

    DEFAULT_POLICY: ClassVar[SourceImageSetIdentityPolicy] = (
        SourceImageSetIdentityPolicy()
    )

    components: tuple[tuple[str, str], ...]

    @classmethod
    def components_from_metadata(
        cls,
        metadata: SourceMetadataMapping,
        *,
        policy: SourceImageSetIdentityPolicy = DEFAULT_POLICY,
    ) -> tuple[tuple[str, str], ...]:
        """Return ordered source image-set components from one metadata mapping."""
        component_values: dict[str, str] = {}
        for field_name, value in SourceMetadataRoleView(metadata).scalar_items():
            if value is None:
                continue
            component = source_metadata_component(str(field_name))
            if component is not None and policy.is_identity_component(component):
                component_values[component.value] = str(value)
        return tuple(
            (component.value, component_values[component.value])
            for component in policy.identity_components()
            if component.value in component_values
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: SourceMetadataMapping,
        *,
        fallback_source_path: str,
        policy: SourceImageSetIdentityPolicy = DEFAULT_POLICY,
    ) -> "SourceImageSetIdentity":
        """Return the source image-set identity represented by one image plane."""
        ordered_components = cls.components_from_metadata(metadata, policy=policy)
        if ordered_components:
            return cls(ordered_components)
        return cls((("source_path", source_path_identity_key(fallback_source_path)),))


@dataclass(frozen=True, slots=True)
class SourceImageSetIdentityPairPredicate(
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Base predicate over one source image-set identity pair."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: ClassVar[str | None] = None

    record_identity: SourceImageSetIdentity
    current_identity: SourceImageSetIdentity

    @classmethod
    def any_match(
        cls,
        record_identities: frozenset[SourceImageSetIdentity],
        current_identities: frozenset[SourceImageSetIdentity],
    ) -> bool:
        return any(
            cls(record_identity, current_identity).matches()
            for record_identity in record_identities
            for current_identity in current_identities
        )

    @abstractmethod
    def matches(self) -> bool:
        """Return whether this identity pair satisfies the predicate."""


@dataclass(frozen=True, slots=True)
class SourceImageSetIdentityCompatibility(SourceImageSetIdentityPairPredicate):
    """Compatibility predicate for full and partially scoped source identities."""

    registry_key = "compatible"

    def matches(self) -> bool:
        record_components = dict(self.record_identity.components)
        current_components = dict(self.current_identity.components)
        if "source_path" in record_components or "source_path" in current_components:
            return record_components == current_components
        shared_keys = set(record_components) & set(current_components)
        return bool(shared_keys) and all(
            record_components[key] == current_components[key]
            for key in shared_keys
        )


@dataclass(frozen=True, slots=True)
class SourceImageSetIdentityExactMatch(SourceImageSetIdentityPairPredicate):
    """Strict predicate for one fully identified source plane."""

    registry_key = "exact"

    def matches(self) -> bool:
        return dict(self.record_identity.components) == dict(
            self.current_identity.components
        )


def merge_source_metadata(
    target: dict[str, SourceMetadataValue],
    additions: SourceMetadataMapping,
    *,
    path: str,
) -> None:
    """Merge extracted metadata into a target map, failing on conflicts."""

    for key, value in additions.items():
        if key == ORIGINAL_SOURCE_METADATA_FIELD:
            OriginalSourceMetadata.from_reserved_value(
                value,
                path=path,
            ).merge_into(target, path=path)
            continue
        existing = target.get(key)
        normalized_value = str(value)
        if (
            existing is not None
            and str(existing) != normalized_value
            and not path_metadata_values_equivalent(str(existing), normalized_value)
        ):
            raise RuntimeError(
                f"Conflicting metadata field '{key}' while parsing source candidate "
                f"{path!r}: {existing!r} != {normalized_value!r}."
            )
        target[key] = canonical_path_metadata_value(normalized_value)


def with_original_source_metadata(
    metadata: SourceMetadataMapping,
    original_metadata: Mapping[str, SourceMetadataScalar],
    *,
    path: str,
) -> dict[str, SourceMetadataValue]:
    """Return metadata carrying source-literal fields in the reserved channel."""

    enriched = dict(metadata)
    OriginalSourceMetadata.from_mapping(original_metadata).merge_into(
        enriched,
        path=path,
    )
    return enriched


def source_metadata_value(
    metadata: SourceMetadataMapping,
    key: str,
) -> str | None:
    """Return a metadata value by semantic key, ignoring spelling separators."""

    normalized_key = normalize_source_metadata_key(key)
    role_view = SourceMetadataRoleView(metadata)
    for candidate_key, value in role_view.scalar_items():
        if str(candidate_key) == key and value is not None:
            return str(value)
    for candidate_key, value in role_view.original_items():
        if (
            normalize_source_metadata_key(str(candidate_key)) == normalized_key
            and value is not None
        ):
            return str(value)
    for candidate_key, value in role_view.scalar_items():
        if (
            normalize_source_metadata_key(str(candidate_key)) == normalized_key
            and value is not None
        ):
            return str(value)
    return None


def source_component_metadata_value(
    metadata: SourceMetadataMapping,
    component: AllComponents,
) -> str | None:
    """Return metadata for an OpenHCS component across canonical and alias fields."""

    normalized_component_key = normalize_source_metadata_key(component.value)
    for field, field_value in SourceMetadataRoleView(metadata).scalar_items():
        field_text = str(field)
        if (
            field_value is not None
            and (
                normalize_source_metadata_key(field_text) == normalized_component_key
                or source_metadata_component(field_text) is component
            )
        ):
            return str(field_value)
    return None


def source_component_metadata_raw_value(
    metadata: SourceMetadataMapping,
    component: AllComponents,
) -> SourceMetadataScalar:
    """Return raw metadata for an OpenHCS component across canonical and alias fields."""

    normalized_component_key = normalize_source_metadata_key(component.value)
    for field, field_value in SourceMetadataRoleView(metadata).scalar_items():
        field_text = str(field)
        if (
            normalize_source_metadata_key(field_text) == normalized_component_key
            or source_metadata_component(field_text) is component
        ) and field_value is not None:
            return field_value
    return None


def semantic_source_metadata_value(
    metadata: SourceMetadataMapping,
    field_name: str,
) -> str | None:
    """Return metadata by OpenHCS component semantics or literal field key."""

    component = source_metadata_component(field_name)
    if component is not None:
        return source_component_metadata_value(metadata, component)
    return source_metadata_value(metadata, field_name)


def source_component_metadata_values(
    metadata: SourceMetadataMapping,
    component: AllComponents,
) -> tuple[str, ...]:
    """Return all metadata values that semantically describe a component."""

    values: list[str] = []
    normalized_component_key = normalize_source_metadata_key(component.value)
    for field, field_value in SourceMetadataRoleView(metadata).scalar_items():
        field_text = str(field)
        if (
            normalize_source_metadata_key(field_text) == normalized_component_key
            or source_metadata_component(field_text) is component
        ) and field_value is not None:
            values.append(str(field_value))
    return tuple(dict.fromkeys(values))


def with_source_component_metadata(
    metadata: SourceMetadataMapping,
    component: AllComponents,
    value: SourceMetadataScalar,
) -> dict[str, SourceMetadataValue]:
    """Return metadata with one canonical component value.

    Source schemas can carry CellProfiler spellings such as ``Well`` and OpenHCS
    spellings such as ``well`` at the same time. Component updates must replace
    every semantic spelling so later normalized lookups cannot observe both the
    old and new component values.
    """
    return {
        **{
            key: field_value
            for key, field_value in metadata.items()
            if key == ORIGINAL_SOURCE_METADATA_FIELD
            or source_metadata_component(str(key)) is not component
        },
        component.value: str(value),
    }


def source_metadata_values_equal(left: str, right: str) -> bool:
    """Compare metadata selector values with source-schema numeric normalization."""

    if left == right:
        return True
    stripped_left = left.strip()
    stripped_right = right.strip()
    if stripped_left == stripped_right:
        return True
    if stripped_left.isdigit() and stripped_right.isdigit():
        return int(stripped_left) == int(stripped_right)
    return False


SourceAxisMetadataValue: TypeAlias = SourceMetadataScalar
SourceAxisMetadataRecord: TypeAlias = SourceMetadataMapping


@dataclass(frozen=True, slots=True)
class SourceAxisMetadataScope:
    """Runtime source-axis metadata constraints used for candidate alignment."""

    component_values: tuple[tuple[str | None, str], ...]

    @property
    def has_component(self) -> bool:
        return any(component is not None for component, _value in self.component_values)

    @classmethod
    def from_component_values(
        cls,
        component_values: tuple[
            tuple[str | None, SourceAxisMetadataValue],
            ...,
        ],
    ) -> "SourceAxisMetadataScope":
        """Return a source-axis scope with one value per semantic component."""
        normalized: dict[str | None, str] = {}
        for component, value in component_values:
            normalized_component = None if component is None else str(component)
            normalized_value = str(value)
            existing = normalized.get(normalized_component)
            if existing is not None:
                if not source_metadata_values_equal(existing, normalized_value):
                    raise ValueError(
                        "Conflicting source-axis metadata constraints for "
                        f"{normalized_component!r}: {existing!r} != "
                        f"{normalized_value!r}."
                    )
                continue
            normalized[normalized_component] = normalized_value
        return cls(tuple(normalized.items()))

    def matches_metadata(self, metadata: SourceAxisMetadataRecord) -> bool:
        return all(
            self.constraint_matches_metadata(metadata, component, value)
            for component, value in self.component_values
        )

    @staticmethod
    def constraint_matches_metadata(
        metadata: SourceAxisMetadataRecord,
        component: str | None,
        value: str,
    ) -> bool:
        if component is not None:
            metadata_value = semantic_source_metadata_value(metadata, component)
            return metadata_value is not None and source_metadata_values_equal(
                metadata_value,
                value,
            )
        return any(
            metadata_value is not None
            and source_metadata_values_equal(str(metadata_value), value)
            for metadata_value in SourceMetadataRoleView(metadata).scalar_values()
        )


@lru_cache(maxsize=4096)
def normalize_source_metadata_key(key: str) -> str:
    """Normalize metadata keys across parser, regex, and setup-module spellings."""

    return "".join(character for character in key.lower() if character.isalnum())


@lru_cache(maxsize=256)
def source_metadata_component(field: str) -> AllComponents | None:
    """Return the OpenHCS component identified by a metadata field name."""

    normalized = normalize_source_metadata_key(field)
    candidate_keys = (normalized,)
    if normalized.startswith("metadata"):
        candidate_keys = (*candidate_keys, normalized.removeprefix("metadata"))
    for component in AllComponents:
        if normalize_source_metadata_key(component.value) in candidate_keys:
            return component
    if candidate_keys[-1] in {"time", "frame", "framenumber"}:
        return AllComponents.TIMEPOINT
    if "channelnumber" in candidate_keys:
        return AllComponents.CHANNEL
    return None


def _require_filter_value(clause: SourceFilterClause) -> str:
    if clause.value is None:
        raise ValueError(
            "SourceFilterClause.value must be set unless match_type is IS_IMAGE."
        )
    return clause.value
