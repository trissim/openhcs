"""Typed source-binding semantics for named step input views."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import InitVar, dataclass, field, fields as dataclass_fields, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Self, TYPE_CHECKING, TypeVar
from urllib.parse import unquote, urlsplit

from metaclass_registry import AutoRegisterMeta
from python_introspect import Enableable
from python_introspect.enableable import EnableableMeta

from openhcs.constants.constants import AllComponents, GroupBy, Microscope
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    ArtifactType,
    ImageArtifactType,
)
from openhcs.core.component_set import ComponentSet
from openhcs.core.components.validation import convert_enum_by_value
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.source_metadata import (
    SourceMetadataIdentityItems,
    SourceMetadataIdentityProjection,
    SourceMetadataMapping,
    SourceMetadataRoleView,
    SourceMetadataScalar,
    SourceMetadataValue,
    SourceVoxelSpacing,
    source_metadata_dict,
    source_metadata_scalar,
)
from openhcs.core.source_path_identity import source_path_identity_key
from openhcs.core.xdg_paths import get_openhcs_cache_dir

if TYPE_CHECKING:
    from openhcs.core.component_group_scope import RuntimeExecutionAxisScope

SourceMetadataIdentity = tuple[tuple[str, SourceMetadataIdentityItems], ...]
SOURCE_ALIAS_PART_SEPARATOR = "__"
SOURCE_BINDING_ALIAS_METADATA_FIELD = "source_alias"
SourceBindingValue = TypeVar("SourceBindingValue")


@dataclass(frozen=True, slots=True)
class SourceBindingRuntimeContextProcessIdentity:
    """Hash-stable semantic identity for process-local source caches."""

    source_order_identity: tuple[Hashable, ...]
    source_metadata_identity: SourceMetadataIdentity
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_hash",
            hash((self.source_order_identity, self.source_metadata_identity)),
        )

    def __hash__(self) -> int:
        return self._hash


class SourceBindingPlanMeta(EnableableMeta, AutoRegisterMeta):
    """Auto-register source-binding plans while preserving Enableable semantics."""


class SourceBindingOrigin(Enum):
    """Where a named binding should be resolved from."""

    STEP_INPUT = "step_input"
    PIPELINE_START = "pipeline_start"


class SourceSetRole(Enum):
    """How one source binding participates in source-set assembly."""

    MATCHED = "matched"
    BROADCAST = "broadcast"


class SourceProjectionRole(Enum):
    """How one source binding is represented in a materialized workspace."""

    PRIMARY_PLANE = "primary_plane"
    SOURCE_ARTIFACT = "source_artifact"


class MetadataSource(Enum):
    """Where metadata extraction rules read source text from."""

    FILE_NAME = "file_name"
    FOLDER_NAME = "folder_name"


class SourceFilterSubject(Enum):
    """Which part of a source path one filter clause targets."""

    FILE = "file"
    DIRECTORY = "directory"
    EXTENSION = "extension"


class SourceFilterMatchType(Enum):
    """How one source filter clause matches its target text."""

    def __new__(cls, value: str, requires_value: bool = True):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.requires_value = requires_value
        return obj

    CONTAINS = ("contains", True)
    CONTAINS_REGEX = ("contains_regex", True)
    DOES_NOT_CONTAIN = ("does_not_contain", True)
    DOES_NOT_CONTAIN_REGEX = ("does_not_contain_regex", True)
    EQUALS = ("equals", True)
    DOES_NOT_EQUAL = ("does_not_equal", True)
    STARTS_WITH = ("starts_with", True)
    DOES_NOT_START_WITH = ("does_not_start_with", True)
    ENDS_WITH = ("ends_with", True)
    DOES_NOT_END_WITH = ("does_not_end_with", True)
    IS_IMAGE = ("is_image", False)
    IS_TIF = ("is_tif", False)


@dataclass(frozen=True, slots=True)
class SourceFilterClause:
    """Typed filter clause applied before metadata extraction."""

    subject: SourceFilterSubject
    match_type: SourceFilterMatchType
    value: str | None = None
    any_group: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subject",
            SourceFilterSubject(self.subject,
                ),
        )
        match_type = SourceFilterMatchType(self.match_type,
            )
        object.__setattr__(self, "match_type", match_type)
        if self.any_group is not None:
            if not isinstance(self.any_group, int) or self.any_group < 0:
                raise ValueError(
                    "SourceFilterClause.any_group must be a nonnegative integer."
                )
        normalized_value = None if self.value is None else str(self.value)
        if not match_type.requires_value:
            object.__setattr__(self, "value", None)
            return
        if normalized_value is None:
            raise ValueError(
                "SourceFilterClause.value is required unless match_type is IS_IMAGE."
            )
        object.__setattr__(self, "value", normalized_value)


def _normalized_optional_source_text(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


@dataclass(frozen=True, slots=True)
class ImagePlaneSource:
    """One explicit image-plane source URI declared by a source config."""

    uri: str
    series: str | None = None
    index: str | None = None
    channel: str | None = None

    def __post_init__(self) -> None:
        normalized_uri = self.uri.strip()
        if not normalized_uri:
            raise ValueError("ImagePlaneSource.uri cannot be empty.")
        object.__setattr__(self, "uri", normalized_uri)
        object.__setattr__(
            self,
            "series",
            _normalized_optional_source_text(self.series),
        )
        object.__setattr__(
            self,
            "index",
            _normalized_optional_source_text(self.index),
        )
        object.__setattr__(
            self,
            "channel",
            _normalized_optional_source_text(self.channel),
        )

    def resolved(self, source_root: Path) -> "ImagePlaneSource":
        """Return this source with its URI resolved to one verified local file."""

        return replace(self, uri=str(resolve_source_file(self.uri, source_root)))


@dataclass(frozen=True, slots=True)
class ImportedMetadataJoin:
    """One join key between image metadata and an imported metadata table."""

    image_metadata_field: str
    imported_metadata_field: str

    def __post_init__(self) -> None:
        image_field = self.image_metadata_field.strip()
        imported_field = self.imported_metadata_field.strip()
        if not image_field:
            raise ValueError(
                "ImportedMetadataJoin.image_metadata_field cannot be empty."
            )
        if not imported_field:
            raise ValueError(
                "ImportedMetadataJoin.imported_metadata_field cannot be empty."
            )
        object.__setattr__(self, "image_metadata_field", image_field)
        object.__setattr__(self, "imported_metadata_field", imported_field)


@dataclass(frozen=True, slots=True)
class ImportedMetadataTable:
    """Pipeline-level metadata imported from an external table."""

    location: str | None = None
    joins: tuple[ImportedMetadataJoin, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "location",
            _normalized_optional_source_text(self.location),
        )
        object.__setattr__(
            self,
            "joins",
            normalize_source_binding_values(
                "ImportedMetadataTable.joins",
                self.joins,
                ImportedMetadataJoin,
            ),
        )

    def resolved(
        self,
        source_root: Path,
        *,
        portable_roots: Sequence[Path] = (),
    ) -> "ImportedMetadataTable":
        """Return this table resolved against declared portable source roots."""

        if self.location is None:
            return self
        return replace(
            self,
            location=str(
                _resolved_imported_metadata_location(
                    self.location,
                    source_root,
                    portable_roots=portable_roots,
                )
            ),
        )


def _resolved_imported_metadata_location(
    location: str,
    source_root: Path,
    *,
    portable_roots: Sequence[Path] = (),
) -> Path:
    """Resolve imported metadata against explicit portable root anchors."""

    parsed = urlsplit(location)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file":
            if parsed.netloc not in ("", "localhost"):
                raise ValueError(
                    f"Unsupported non-local file URI authority in {location!r}."
                )
            path = Path(unquote(parsed.path))
        else:
            path = Path(location)
        portable_path = _portable_root_anchored_source_path(
            path,
            (Path(source_root), *(Path(root) for root in portable_roots)),
        )
        if portable_path is not None:
            return resolve_source_file(str(portable_path), source_root)
    return resolve_source_file(location, source_root)


def resolve_source_file(location: str, source_root: Path) -> Path:
    """Resolve a declared local or HTTP source into one verified local path."""

    parsed = urlsplit(location)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file":
            if parsed.netloc not in ("", "localhost"):
                raise ValueError(
                    f"Unsupported non-local file URI authority in {location!r}."
                )
            path = Path(unquote(parsed.path))
        else:
            path = Path(location)
        resolved = path if path.is_absolute() else Path(source_root) / path
        resolved = resolved.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Declared source file does not exist: {resolved}")
        return resolved
    if parsed.scheme in ("http", "https"):
        return _materialized_http_source(location, parsed.path)
    raise ValueError(
        f"Unsupported source URI scheme {parsed.scheme!r} in {location!r}."
    )


def _portable_root_anchored_source_path(
    declared_path: Path,
    portable_roots: Sequence[Path],
) -> Path | None:
    """Rebase a stale path only when it names an explicit portable root."""

    parts = declared_path.parts
    candidates: list[Path] = []
    for root in portable_roots:
        anchor_indexes = tuple(
            index for index, part in enumerate(parts) if part == root.name
        )
        if not anchor_indexes:
            continue
        tail = parts[anchor_indexes[-1] + 1 :]
        if not tail:
            continue
        candidates.append(root.joinpath(*tail))

    unique_candidates = tuple(dict.fromkeys(candidates))
    if len(unique_candidates) > 1:
        raise ValueError(
            f"Declared source path {declared_path!s} matches multiple portable roots: "
            f"{tuple(str(candidate) for candidate in unique_candidates)!r}."
        )
    return unique_candidates[0] if unique_candidates else None


def _materialized_http_source(uri: str, uri_path: str) -> Path:
    """Materialize one HTTP source atomically in its deterministic XDG cache."""

    cache_dir = (
        get_openhcs_cache_dir()
        / "source_imports"
        / hashlib.sha256(uri.encode("utf-8")).hexdigest()
    )
    suffix = Path(unquote(uri_path)).suffix
    if cache_dir.is_dir():
        cached = tuple(
            path
            for path in cache_dir.iterdir()
            if path.is_file() and path.suffix == suffix and len(path.stem) == 64
        )
        if len(cached) == 1:
            return cached[0]

    request = urllib.request.Request(uri, headers={"User-Agent": "OpenHCS"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    content_digest = hashlib.sha256(payload).hexdigest()
    target = cache_dir / f"{content_digest}{suffix}"
    if target.is_file():
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=cache_dir)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def normalize_source_binding_values(
    field_name: str,
    values: tuple[SourceBindingValue, ...],
    value_type: type[SourceBindingValue],
) -> tuple[SourceBindingValue, ...]:
    """Return a typed tuple for one source-binding field."""

    normalized_values = tuple(values)
    for value in normalized_values:
        if not isinstance(value, value_type):
            raise TypeError(
                f"{field_name} must contain {value_type.__name__} "
                f"values, got {type(value).__name__}."
            )
    return normalized_values


@dataclass(frozen=True, slots=True)
class MetadataExtractionRule:
    """Regex-backed metadata extraction rule for source binding resolution."""

    source: MetadataSource
    pattern: str
    filters: tuple[SourceFilterClause, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            MetadataSource(self.source,
                ),
        )
        if not self.pattern:
            raise ValueError("MetadataExtractionRule.pattern cannot be empty.")
        compiled_pattern = re.compile(str(self.pattern))
        if not compiled_pattern.groupindex:
            raise ValueError(
                "MetadataExtractionRule.pattern must define at least one named "
                "capture group."
            )
        object.__setattr__(self, "pattern", str(self.pattern))
        object.__setattr__(
            self,
            "filters",
            normalize_source_binding_values(
                "MetadataExtractionRule.filters",
                self.filters,
                SourceFilterClause,
            ),
        )

    @property
    def capture_fields(self) -> tuple[str, ...]:
        """Return named metadata fields in their regex declaration order."""

        return tuple(re.compile(self.pattern).groupindex)


class SourceBindingMatchMethod(Enum):
    """How selected source aliases are paired into one logical source set."""

    METADATA = "metadata"
    ORDER = "order"


@dataclass(frozen=True, slots=True)
class SourceBindingMatchField:
    """Metadata field from one alias used as a source-set pairing key."""

    alias: str
    metadata_field: str

    def __post_init__(self) -> None:
        _require_name(self.alias, "SourceBindingMatchField.alias")
        _require_name(
            self.metadata_field,
            "SourceBindingMatchField.metadata_field",
        )
        object.__setattr__(self, "alias", str(self.alias))
        object.__setattr__(self, "metadata_field", str(self.metadata_field))


@dataclass(frozen=True, slots=True)
class SourceBindingMatchFields:
    """Validated match fields with one field per source alias."""

    fields: tuple[SourceBindingMatchField, ...]

    def normalized(self) -> tuple[SourceBindingMatchField, ...]:
        fields = normalize_source_binding_values(
            "SourceBindingMatchDimension.fields",
            self.fields,
            SourceBindingMatchField,
        )
        seen_aliases: set[str] = set()
        for match_field in fields:
            if match_field.alias in seen_aliases:
                raise ValueError(
                    "SourceBindingMatchDimension contains duplicate alias "
                    f"{match_field.alias!r}."
                )
            seen_aliases.add(match_field.alias)
        return fields


@dataclass(frozen=True, slots=True)
class SourceBindingMatchDimension:
    """One shared source-set key, expressed as alias-to-metadata-field pairs."""

    fields: tuple[SourceBindingMatchField, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fields",
            SourceBindingMatchFields(self.fields).normalized(),
        )

    def field_for_alias(self, alias: str) -> str | None:
        for match_field in self.fields:
            if match_field.alias == alias:
                return match_field.metadata_field
        return None


@dataclass(frozen=True, slots=True)
class SourceBindingMatchPlan:
    """Cross-alias pairing plan for assembling selected sources into source sets."""

    method: SourceBindingMatchMethod
    dimensions: tuple[SourceBindingMatchDimension, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "method",
            SourceBindingMatchMethod(self.method,
                ),
        )
        object.__setattr__(
            self,
            "dimensions",
            normalize_source_binding_values(
                "SourceBindingMatchPlan.dimensions",
                self.dimensions,
                SourceBindingMatchDimension,
            ),
        )


@dataclass(frozen=True, slots=True)
class ComponentSelector:
    """Component-axis key/value pair used either to select sources or assign identity."""

    component: Any
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component",
            _coerce_component(self.component, "ComponentSelector.component"),
        )
        if self.value == "":
            raise ValueError("ComponentSelector.value cannot be empty.")
        object.__setattr__(self, "value", str(self.value))


@dataclass(frozen=True, slots=True)
class MetadataSelector:
    """Metadata field/value filter used to select source candidates for one alias."""

    field: str
    value: SourceMetadataScalar

    def __post_init__(self) -> None:
        _require_name(self.field, "MetadataSelector.field")
        if self.value is None or self.value == "":
            raise ValueError("MetadataSelector.value cannot be empty.")
        object.__setattr__(self, "field", str(self.field))
        object.__setattr__(self, "value", source_metadata_scalar(self.value))


@dataclass(frozen=True, slots=True)
class SourceSelector:
    """Choose store-emitted image planes for one named pipeline input.

    Component and metadata selectors use declared plane identity. Path filters use
    source provenance supplied by the owning image store; they never decode a
    storage format or infer its axes.
    """

    components: tuple[ComponentSelector, ...] = ()
    metadata: tuple[MetadataSelector, ...] = ()
    filters: tuple[SourceFilterClause, ...] = ()
    inherit_current_scope: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "components",
            normalize_source_binding_values(
                "SourceSelector.components",
                self.components,
                ComponentSelector,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            normalize_source_binding_values(
                "SourceSelector.metadata",
                self.metadata,
                MetadataSelector,
            ),
        )
        object.__setattr__(
            self,
            "filters",
            normalize_source_binding_values(
                "SourceSelector.filters",
                self.filters,
                SourceFilterClause,
            ),
        )


def source_alias_measurement_names(alias: str) -> tuple[str, ...]:
    """Return measurement source-name tokens represented by a source alias."""
    normalized_alias = alias.strip()
    if not normalized_alias:
        return ()
    parts = tuple(
        part for part in normalized_alias.split(SOURCE_ALIAS_PART_SEPARATOR) if part
    )
    return parts or (normalized_alias,)


@dataclass(frozen=True, slots=True)
class SourceAssignmentBase(metaclass=AutoRegisterMeta):
    """Shared contract for selecting an alias and assigning its semantic identity."""

    __registry_key__ = "assignment_kind"
    __skip_if_no_key__ = True
    assignment_kind: ClassVar[str | None] = None

    alias: str
    selector: SourceSelector = SourceSelector()
    origin: SourceBindingOrigin = SourceBindingOrigin.STEP_INPUT
    component_identity: tuple[ComponentSelector, ...] = ()
    """Semantic component axes assigned after selector resolution."""

    def __post_init__(self) -> None:
        normalized_alias = str(self.alias).strip()
        if not normalized_alias:
            raise ValueError(f"{type(self).__name__}.alias cannot be empty.")
        object.__setattr__(self, "alias", normalized_alias)
        if not isinstance(self.selector, SourceSelector):
            raise TypeError(
                f"{type(self).__name__}.selector must be SourceSelector, "
                f"got {type(self.selector).__name__}."
            )
        component_identity = normalize_source_binding_values(
            f"{type(self).__name__}.component_identity",
            self.component_identity,
            ComponentSelector,
        )
        seen_components: dict[AllComponents, str] = {}
        for selector in component_identity:
            existing = seen_components.get(selector.component)
            if existing is not None and existing != selector.value:
                raise ValueError(
                    f"{type(self).__name__}.component_identity contains "
                    f"conflicting {selector.component.value!r} values "
                    f"{existing!r} and {selector.value!r}."
                )
            seen_components[selector.component] = selector.value
        object.__setattr__(
            self,
            "component_identity",
            tuple(dict.fromkeys(component_identity)),
        )
        object.__setattr__(
            self,
            "origin",
            SourceBindingOrigin(self.origin,
                ),
        )

    @property
    def artifact_kind(self) -> ArtifactType:
        """Artifact kind bound by this source assignment."""
        raise NotImplementedError(f"{type(self).__name__} must provide artifact_kind.")

    @property
    def measurement_source_names(self) -> tuple[str, ...]:
        """Return measurement feature source qualifiers declared by this alias."""
        if not self.artifact_kind.participates_in_measurement_source_names:
            return ()
        return source_alias_measurement_names(self.alias)

    def component_identity_with(
        self,
        selector: ComponentSelector,
    ) -> tuple[ComponentSelector, ...]:
        """Return component identity extended by one non-conflicting selector."""
        if not isinstance(selector, ComponentSelector):
            raise TypeError(
                f"{type(self).__name__}.component_identity selector must be "
                f"ComponentSelector, got {type(selector).__name__}."
            )
        for existing in self.component_identity:
            if existing.component is not selector.component:
                continue
            if existing.value != selector.value:
                raise ValueError(
                    f"Source assignment {self.alias!r} declares "
                    f"{selector.component.value!r} identity {existing.value!r}, "
                    f"but {selector.value!r} was requested."
                )
            return self.component_identity
        return (*self.component_identity, selector)

    def is_compatible_with_component_group(
        self,
        component: AllComponents,
        group_key: str,
    ) -> bool:
        """Return whether this identity permits one typed execution group."""

        component_selectors = tuple(
            selector
            for selector in self.component_identity
            if selector.component is component
        )
        return not component_selectors or any(
            selector.value == group_key for selector in component_selectors
        )

    def with_component_identity(self, selector: ComponentSelector) -> Self:
        """Return this source assignment with one canonical component identity."""
        return replace(
            self,
            component_identity=self.component_identity_with(selector),
        )


@dataclass(frozen=True, slots=True)
class NamedSourceBinding(SourceAssignmentBase):
    """Name selected image planes and their optional component identity.

    ``alias`` is the source name presented to pipeline functions and user
    interfaces. The selected planes retain their exact sample/well, site, channel,
    Z, timepoint, and store-backed pixel identity.
    """

    assignment_kind = "named_source_binding"
    artifact_kind: type[ArtifactType] = ImageArtifactType
    required: bool = True
    source_set_role: SourceSetRole = SourceSetRole.MATCHED
    projection_role: SourceProjectionRole = SourceProjectionRole.PRIMARY_PLANE
    explicit_source: ImagePlaneSource | None = None
    load_as_monochrome: bool = False
    load_as_mask: bool = False
    source_channel_axis: int | None = None
    source_channel_counts: frozenset[int] | None = None

    def __post_init__(self) -> None:
        SourceAssignmentBase.__post_init__(self)
        object.__setattr__(
            self,
            "artifact_kind",
            ArtifactType.coerce(self.artifact_kind),
        )
        source_set_role = SourceSetRole(self.source_set_role,
            )
        projection_role = SourceProjectionRole(self.projection_role,
            )
        if projection_role is SourceProjectionRole.PRIMARY_PLANE and not issubclass(
            self.artifact_kind, ImageArtifactType
        ):
            raise ValueError(
                "NamedSourceBinding PRIMARY_PLANE projections require an image "
                f"artifact kind, got {self.artifact_kind.__name__}."
            )
        if self.explicit_source is not None and not isinstance(
            self.explicit_source,
            ImagePlaneSource,
        ):
            raise TypeError(
                "NamedSourceBinding.explicit_source must be ImagePlaneSource or None."
            )
        object.__setattr__(self, "source_set_role", source_set_role)
        object.__setattr__(self, "projection_role", projection_role)
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "load_as_monochrome", bool(self.load_as_monochrome))
        object.__setattr__(self, "load_as_mask", bool(self.load_as_mask))
        source_channel_axis = self.source_channel_axis
        if source_channel_axis is not None and (
            not isinstance(source_channel_axis, int)
            or isinstance(source_channel_axis, bool)
        ):
            raise TypeError(
                "NamedSourceBinding.source_channel_axis must be int or None."
            )
        source_channel_counts = self.source_channel_counts
        if source_channel_counts is not None:
            if source_channel_axis is None:
                raise ValueError(
                    "NamedSourceBinding source channel counts require a channel axis."
                )
            normalized_counts = frozenset(int(value) for value in source_channel_counts)
            if any(
                isinstance(value, bool) or int(value) <= 0
                for value in source_channel_counts
            ):
                raise ValueError(
                    "NamedSourceBinding.source_channel_counts must contain positive "
                    "integers."
                )
            source_channel_counts = normalized_counts
        object.__setattr__(self, "source_channel_axis", source_channel_axis)
        object.__setattr__(self, "source_channel_counts", source_channel_counts)

    def source_channel_axis_for_shape(
        self,
        shape: tuple[int, ...],
        *,
        source_axis_count: int = 0,
        observed_axis: int | None = None,
    ) -> int | None:
        """Resolve the binding and loaded-payload channel-axis declarations."""

        axis = self.source_channel_axis
        if len(shape) == source_axis_count + 2:
            return None
        normalized_axis = None
        if axis is not None:
            normalized_axis = axis if axis >= 0 else len(shape) + axis
            if normalized_axis < source_axis_count or normalized_axis >= len(shape):
                raise ValueError(
                    f"Source binding {self.alias!r} channel axis {axis} is "
                    f"incompatible with source shape {shape!r} and "
                    f"{source_axis_count} leading axes."
                )
            counts = self.source_channel_counts
            if counts is not None and shape[normalized_axis] not in counts:
                raise ValueError(
                    f"Source binding {self.alias!r} declares channel axis {axis} "
                    f"with cardinality in {tuple(sorted(counts))!r}, but source "
                    f"shape {shape!r} carries {shape[normalized_axis]} channels."
                )
        if observed_axis is None:
            return axis
        normalized_observed_axis = (
            observed_axis if observed_axis >= 0 else len(shape) + observed_axis
        )
        if (
            normalized_observed_axis < source_axis_count
            or normalized_observed_axis >= len(shape)
        ):
            raise ValueError(
                f"Loaded source channel axis {observed_axis} for binding "
                f"{self.alias!r} is incompatible with source shape {shape!r} and "
                f"{source_axis_count} leading axes."
            )
        if normalized_axis is not None and normalized_axis != normalized_observed_axis:
            raise ValueError(
                f"Source binding {self.alias!r} channel axis {axis} conflicts with "
                f"loaded payload channel axis {observed_axis}."
            )
        return observed_axis

    def input_spec(self) -> ArtifactSpec:
        """Project this source declaration into its canonical artifact input."""

        return ArtifactSpec.input(
            self.alias,
            self.artifact_kind,
            required=self.required,
        )

    def component_values(
        self,
        component: AllComponents,
        *,
        realized_source_metadata: Iterable[SourceMetadataMapping] | None = None,
    ) -> tuple[str, ...]:
        """Return declared or workspace-realized values for one component axis."""

        if realized_source_metadata is None:
            return tuple(
                selector.value
                for selector in self.component_identity
                if selector.component is component
            )

        from openhcs.core.source_matching import source_component_metadata_values

        values: list[str] = []
        for metadata in realized_source_metadata:
            if not self.matches_realized_source_metadata(metadata):
                continue
            values.extend(source_component_metadata_values(metadata, component))
        return tuple(dict.fromkeys(values))

    def matches_realized_source_metadata(
        self,
        metadata: SourceMetadataMapping,
    ) -> bool:
        """Return whether workspace metadata belongs to this source binding."""

        alias_value = metadata.get(SOURCE_BINDING_ALIAS_METADATA_FIELD)
        if isinstance(alias_value, Mapping):
            raise TypeError(
                f"{SOURCE_BINDING_ALIAS_METADATA_FIELD} must be scalar source "
                "metadata."
            )
        normalized_alias = source_metadata_scalar(alias_value)
        return normalized_alias is None or str(normalized_alias) == self.alias

    def input_plan(
        self,
        *,
        group_keys: tuple[str | None, ...],
        group_component: AllComponents | None,
    ) -> ArtifactInputPlan:
        """Project this source declaration into a compiler input plan."""

        return ArtifactInputPlan(
            name=self.alias,
            path=f"source-binding:{self.alias}",
            artifact_type=self.artifact_kind,
            group_keys=group_keys,
            group_component=group_component,
        )

    @property
    def requires_selector_resolution(self) -> bool:
        """Whether this binding needs file/metadata-aware source resolution."""

        return bool(
            self.selector.components
            or self.selector.metadata
            or self.selector.filters
            or not self.selector.inherit_current_scope
        )

    def requires_step_input_component_stack(
        self,
        components: tuple[AllComponents, ...],
    ) -> bool:
        """Whether resolving this binding needs component-varying step input."""
        if self.origin is not SourceBindingOrigin.STEP_INPUT:
            return False
        if not components:
            return False
        if self.selector.filters or self.selector.metadata:
            return True
        return any(
            selector.component in components for selector in self.selector.components
        )

    @property
    def requires_step_input_channel_stack(self) -> bool:
        """Whether resolving this binding needs channel-varying step input."""
        return self.requires_step_input_component_stack((AllComponents.CHANNEL,))


class SourceBindingDeclarationsMixin:
    """Shared named-binding view for editable and compiled source plans."""

    bindings: tuple[NamedSourceBinding, ...] | None
    source_stack_components: tuple[AllComponents, ...]

    @property
    def binding_declarations(self) -> tuple[NamedSourceBinding, ...]:
        """Named bindings explicitly declared on this plan."""

        return tuple(self.bindings or ())

    @property
    def matched_source_bindings(self) -> tuple[NamedSourceBinding, ...]:
        """Bindings that determine source-set membership and cardinality."""

        return tuple(
            binding
            for binding in self.binding_declarations
            if binding.source_set_role is SourceSetRole.MATCHED
        )

    @property
    def broadcast_source_bindings(self) -> tuple[NamedSourceBinding, ...]:
        """Bindings appended to every assembled source set."""

        return tuple(
            binding
            for binding in self.binding_declarations
            if binding.source_set_role is SourceSetRole.BROADCAST
        )

    @property
    def primary_plane_bindings(self) -> tuple[NamedSourceBinding, ...]:
        """Bindings materialized as primary image planes."""

        return tuple(
            binding
            for binding in self.binding_declarations
            if binding.projection_role is SourceProjectionRole.PRIMARY_PLANE
        )

    def binding_for_alias(
        self,
        alias: str,
    ) -> NamedSourceBinding | None:
        """Return the declaration for one source alias when present."""
        for binding in self.binding_declarations:
            if binding.alias == alias:
                return binding
        return None

    def binding_for_artifact_ref(
        self,
        ref: ArtifactSpecRef,
    ) -> NamedSourceBinding | None:
        """Return the source declaration for one exact artifact input ref."""
        if not isinstance(ref, ArtifactSpecRef):
            raise TypeError(
                "SourceBindingDeclarationsMixin.binding_for_artifact_ref requires "
                f"an ArtifactSpecRef, got {type(ref).__name__}."
            )
        return next(
            (
                binding
                for binding in self.binding_declarations
                if binding.input_spec().ref() == ref
            ),
            None,
        )

    def declares_artifact_ref(self, ref: ArtifactSpecRef) -> bool:
        """Return whether this plan declares one exact artifact input ref."""
        return self.binding_for_artifact_ref(ref) is not None

    @property
    def primary_plane_aliases(self) -> tuple[str, ...]:
        """Return aliases materialized into the primary source image stack."""
        return tuple(binding.alias for binding in self.primary_plane_bindings)

    @property
    def measurement_source_names(self) -> tuple[str, ...]:
        """Return source qualifiers available to measurement feature names."""
        names = {
            name
            for binding in self.binding_declarations
            for name in binding.measurement_source_names
        }
        return tuple(sorted(names, key=str.lower))

    def requires_step_input_component_stack(
        self,
        components: tuple[AllComponents, ...],
    ) -> bool:
        """Whether any declaration needs component-resolved step input."""
        return any(
            binding.requires_step_input_component_stack(components)
            for binding in self.binding_declarations
        )

    @property
    def requires_step_input_channel_stack(self) -> bool:
        """Whether any declaration needs channel-resolved step input."""
        return self.requires_step_input_component_stack((AllComponents.CHANNEL,))

    @property
    def requires_pipeline_start_resolution(self) -> bool:
        """Whether any declaration resolves from the pipeline-start universe."""
        return any(
            binding.origin is SourceBindingOrigin.PIPELINE_START
            for binding in self.binding_declarations
        )

    @property
    def requires_pipeline_start_source_set(self) -> bool:
        """Whether pipeline-start declarations form a multi-member source set."""
        return (
            sum(
                1
                for binding in self.binding_declarations
                if binding.origin is SourceBindingOrigin.PIPELINE_START
                and binding.source_set_role is SourceSetRole.MATCHED
            )
            > 1
        )

    @property
    def requires_step_input_selector_resolution(self) -> bool:
        """Whether step-input declarations need selector-aware matching."""
        return any(
            binding.origin is SourceBindingOrigin.STEP_INPUT
            and binding.requires_selector_resolution
            for binding in self.binding_declarations
        )

    def bindings_for_component_group(
        self,
        component: AllComponents | None,
        group_key: str | None,
    ) -> tuple[NamedSourceBinding, ...]:
        """Return bindings compatible with one typed execution group."""
        if component is None and group_key is None:
            return self.binding_declarations
        if component is None or group_key is None:
            raise ValueError(
                "Source binding component grouping requires both component and "
                "group_key, or neither."
            )
        normalized_group_key = str(group_key)
        normalized_component = _coerce_component(
            component,
            "SourceBindingDeclarationsMixin.bindings_for_component_group.component",
        )
        matching_bindings = tuple(
            binding
            for binding in self.binding_declarations
            if binding.is_compatible_with_component_group(
                normalized_component,
                normalized_group_key,
            )
        )
        if not matching_bindings:
            raise ValueError(
                f"No source binding declares {normalized_component.value} "
                f"group {normalized_group_key!r}."
            )
        return matching_bindings

    def for_component_group(
        self,
        component: AllComponents,
        group_key: str,
    ) -> Self:
        """Return these declarations scoped to one typed component group."""
        return replace(
            self,
            bindings=self.bindings_for_component_group(component, group_key),
        )

    def for_execution_axis_scope(
        self,
        axis_scope: "RuntimeExecutionAxisScope",
    ) -> Self:
        """Project declarations to one typed runtime component coordinate."""

        if not self.has_primary_content or not axis_scope.has_value:
            return self
        component = axis_scope.component
        if component is None:
            raise RuntimeError(
                "A valued runtime execution scope must declare its component."
            )
        return self.for_component_group(
            component,
            axis_scope.require_value_text(),
        )

    def component_group_keys_for_artifact_specs(
        self,
        component: AllComponents,
        specs: Iterable[ArtifactSpec],
        available_artifacts: ArtifactSpecCollection,
        *,
        realized_source_metadata: Iterable[SourceMetadataMapping] | None = None,
    ) -> tuple[str, ...]:
        """Trace artifact group lineage to this plan's component selectors."""

        resolved_component = _coerce_component(
            component,
            "SourceBindingDeclarationsMixin.component_group_keys_for_artifact_specs.component",
        )
        realized_metadata = (
            None
            if realized_source_metadata is None
            else tuple(realized_source_metadata)
        )
        return tuple(
            dict.fromkeys(
                value
                for binding in self.bindings_for_artifact_specs(
                    specs,
                    available_artifacts,
                )
                for value in binding.component_values(
                    resolved_component,
                    realized_source_metadata=realized_metadata,
                )
            )
        )

    def bindings_for_artifact_specs(
        self,
        specs: Iterable[ArtifactSpec],
        available_artifacts: ArtifactSpecCollection,
    ) -> tuple[NamedSourceBinding, ...]:
        """Trace exact artifact group lineage to its source declarations."""

        if not self.binding_declarations:
            return ()

        def source_bindings(
            spec: ArtifactSpec,
            visited: frozenset[ArtifactSpecRef],
        ) -> tuple[NamedSourceBinding, ...]:
            ref = spec.ref()
            if ref in visited:
                return ()
            next_visited = visited | {ref}
            binding = self.binding_for_artifact_ref(ref)
            direct_bindings = (binding,) if binding is not None else ()
            if spec.relations:
                declared_spec = spec
            elif direct_bindings:
                return direct_bindings
            else:
                declared_spec = available_artifacts.by_name_and_artifact_type(
                    spec.name,
                    spec.artifact_type,
                )
                if declared_spec is None:
                    raise ValueError(
                        f"Artifact group lineage references unavailable {ref!r}."
                    )
            relation_bindings: list[NamedSourceBinding] = []
            for source_ref in declared_spec.group_scope_sources():
                source_spec = available_artifacts.by_name_and_artifact_type(
                    source_ref.name,
                    source_ref.artifact_type,
                )
                if (
                    source_spec is None
                    or source_spec.for_plan_type(source_ref.plan_type).ref()
                    != source_ref
                ):
                    raise ValueError(
                        "Artifact group-lineage relation references unavailable "
                        f"source {source_ref!r}."
                    )
                relation_bindings.extend(source_bindings(source_spec, next_visited))
            return tuple(dict.fromkeys((*direct_bindings, *relation_bindings)))

        return self.bindings_for_artifact_refs(
            binding.input_spec().ref()
            for spec in specs
            for binding in source_bindings(spec, frozenset())
        )

    def for_artifact_specs(
        self,
        specs: Iterable[ArtifactSpec],
        available_artifacts: ArtifactSpecCollection,
    ) -> Self:
        """Project these declarations through exact artifact lineage."""

        return replace(
            self,
            bindings=self.bindings_for_artifact_specs(
                specs,
                available_artifacts,
            ),
        )

    def runtime_variable_components_for_artifact_specs(
        self,
        specs: Iterable[ArtifactSpec],
        available_artifacts: ArtifactSpecCollection,
        consumer_variable_components: ComponentSet,
    ) -> ComponentSet | None:
        """Return source axes visible to a consumer, or None without lineage."""

        if not isinstance(consumer_variable_components, ComponentSet):
            raise TypeError(
                "Source binding runtime variable components require a ComponentSet."
            )
        bindings = self.bindings_for_artifact_specs(specs, available_artifacts)
        if not bindings:
            return None

        fixed_components = ComponentSet(
            tuple(
                component
                for component in consumer_variable_components
                for binding_values in (
                    tuple(binding.component_values(component) for binding in bindings),
                )
                if all(len(values) == 1 for values in binding_values)
                and len({values[0] for values in binding_values}) == 1
            )
        )
        return consumer_variable_components.excluding(fixed_components)

    def bindings_for_artifact_refs(
        self,
        artifact_refs: Iterable[ArtifactSpecRef],
    ) -> tuple[NamedSourceBinding, ...]:
        """Return bindings whose exact input refs are selected."""
        refs = frozenset(artifact_refs)
        if any(not isinstance(ref, ArtifactSpecRef) for ref in refs):
            raise TypeError(
                "SourceBindingDeclarationsMixin.bindings_for_artifact_refs "
                "requires ArtifactSpecRef values."
            )
        return tuple(
            binding
            for binding in self.binding_declarations
            if binding.input_spec().ref() in refs
        )

    def for_artifact_refs(
        self,
        artifact_refs: Iterable[ArtifactSpecRef],
    ) -> Self:
        """Return this source-binding config scoped to exact artifact refs."""
        return replace(
            self,
            bindings=self.bindings_for_artifact_refs(artifact_refs),
        )


@dataclass(frozen=True, kw_only=True)
class _SourceBindingPlanBase(ABC, metaclass=SourceBindingPlanMeta):
    """Shared typed source-binding plan fields across editable and compiled views."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None
    metadata_rules: tuple[MetadataExtractionRule, ...] = ()
    """Regex/metadata extraction rules that add semantic fields for matching sources."""

    match_plan: SourceBindingMatchPlan | None = None
    """Optional strategy for pairing selected aliases into logical source sets."""

    metadata_fields: tuple[FieldSpec, ...] = ()
    """Exact scalar schema for source-literal metadata fields."""

    @classmethod
    def registered_plan_types(cls) -> tuple[type["_SourceBindingPlanBase"], ...]:
        """Return registered concrete source-binding plan views."""

        registered_types: list[type["_SourceBindingPlanBase"]] = []
        for plan_type in cls.__registry__.values():
            concrete_type = cls.concrete_registered_plan_type(plan_type)
            if concrete_type not in registered_types:
                registered_types.append(concrete_type)
        return tuple(registered_types)

    @classmethod
    def concrete_registered_plan_type(
        cls,
        plan_type: type["_SourceBindingPlanBase"],
    ) -> type["_SourceBindingPlanBase"]:
        """Return the concrete declaration type for a registered plan view."""

        for base_type in plan_type.__mro__[1:]:
            if base_type is _SourceBindingPlanBase:
                break
            if (
                issubclass(base_type, _SourceBindingPlanBase)
                and base_type.registry_key == plan_type.registry_key
            ):
                return base_type
        return plan_type

    def _normalize_common_fields(self) -> None:
        metadata_rules = object.__getattribute__(self, "metadata_rules")
        if metadata_rules is not None:
            metadata_rules = tuple(metadata_rules)
            for rule in metadata_rules:
                if not isinstance(rule, MetadataExtractionRule):
                    raise TypeError(
                        f"{type(self).__name__}.metadata_rules must contain "
                        "MetadataExtractionRule values, got "
                        f"{type(rule).__name__}."
                    )
        object.__setattr__(self, "metadata_rules", metadata_rules)

        match_plan = object.__getattribute__(self, "match_plan")
        if match_plan is not None and not isinstance(
            match_plan,
            SourceBindingMatchPlan,
        ):
            raise TypeError(
                f"{type(self).__name__}.match_plan must be SourceBindingMatchPlan "
                f"or None, got {type(match_plan).__name__}."
            )

        metadata_fields = object.__getattribute__(self, "metadata_fields")
        if metadata_fields is not None:
            metadata_fields = FieldSpec.merge_exact(
                (tuple(metadata_fields),),
                context=f"{type(self).__name__}.metadata_fields",
            )
        object.__setattr__(self, "metadata_fields", metadata_fields)

    @property
    def is_empty(self) -> bool:
        return (
            not self.has_primary_content
            and not self.metadata_rule_declarations
            and self.match_plan is None
            and not self.metadata_fields
        )

    @property
    def metadata_rule_declarations(self) -> tuple[MetadataExtractionRule, ...]:
        """Metadata rules explicitly declared on this plan."""

        return tuple(self.metadata_rules or ())

    @property
    @abstractmethod
    def has_primary_content(self) -> bool:
        """Whether the subclass-specific binding payload is empty."""


@dataclass(frozen=True)
class SourceBindingsConfig(SourceBindingDeclarationsMixin, _SourceBindingPlanBase):
    """Canonical pipeline declaration for selecting and naming plate sources.

    Normal plate initialization resolves these bindings over exact planes emitted
    by the image stores. Embedded sample/well and axis coordinates are preserved;
    an absent axis becomes ``"1"`` only when its store declares that axis absent
    or singleton. Conflicting dataset, plane, or component identities fail
    explicitly instead of being selected by a fallback.
    """

    registry_key: ClassVar[str] = "source"
    microscope_handler_name: ClassVar[str] = Microscope.SOURCE_BINDINGS.value
    source_filters: tuple[SourceFilterClause, ...] = ()
    """Filters limiting the source universe before named bindings are resolved."""

    bindings: tuple[NamedSourceBinding, ...] = ()
    """Named semantic source bindings available to pipelines and inherited by steps."""

    image_plane_sources: tuple[ImagePlaneSource, ...] = ()
    """Exact image-plane source records available for source-binding resolution."""

    imported_metadata_tables: tuple[ImportedMetadataTable, ...] = ()
    """External metadata tables joined to source records before matching."""

    source_stack_components: tuple[AllComponents, ...] = ()
    """Ordered plate components that form one logical source image stack."""

    grouping_metadata_fields: tuple[str, ...] = ()
    """Metadata field names used to partition matched sources into execution groups."""

    source_voxel_spacing: SourceVoxelSpacing = field(default_factory=SourceVoxelSpacing)
    """Physical source-pixel spacing in z, y, x order; omit values when unknown."""

    def __post_init__(self) -> None:
        self._normalize_common_fields()
        source_filters = object.__getattribute__(self, "source_filters")
        if source_filters is not None:
            source_filters = normalize_source_binding_values(
                "SourceBindingsConfig.source_filters",
                source_filters,
                SourceFilterClause,
            )
        bindings = object.__getattribute__(self, "bindings")
        if bindings is not None:
            bindings = normalize_source_binding_values(
                f"{type(self).__name__}.bindings",
                bindings,
                NamedSourceBinding,
            )
            bindings = tuple(
                replace(
                    binding,
                    selector=replace(
                        binding.selector,
                        metadata=tuple(
                            replace(
                                selector,
                                value=self.coerce_metadata_scalar(
                                    selector.field,
                                    selector.value,
                                ),
                            )
                            for selector in binding.selector.metadata
                        ),
                    ),
                )
                for binding in bindings
            )
        seen_aliases: set[str] = set()
        if bindings is not None:
            for binding in bindings:
                if binding.alias in seen_aliases:
                    raise ValueError(
                        f"{type(self).__name__}.bindings contains duplicate alias "
                        f"{binding.alias!r}."
                    )
                seen_aliases.add(binding.alias)
        object.__setattr__(self, "source_filters", source_filters)
        object.__setattr__(self, "bindings", bindings)
        image_plane_sources = object.__getattribute__(self, "image_plane_sources")
        if image_plane_sources is not None:
            image_plane_sources = normalize_source_binding_values(
                f"{type(self).__name__}.image_plane_sources",
                image_plane_sources,
                ImagePlaneSource,
            )
        object.__setattr__(self, "image_plane_sources", image_plane_sources)

        imported_metadata_tables = object.__getattribute__(
            self,
            "imported_metadata_tables",
        )
        if imported_metadata_tables is not None:
            imported_metadata_tables = normalize_source_binding_values(
                f"{type(self).__name__}.imported_metadata_tables",
                imported_metadata_tables,
                ImportedMetadataTable,
            )
        object.__setattr__(
            self,
            "imported_metadata_tables",
            imported_metadata_tables,
        )

        source_stack_components = object.__getattribute__(
            self,
            "source_stack_components",
        )
        if source_stack_components is not None:
            source_stack_components = ComponentSet.collect(
                source_stack_components
            ).as_tuple()
        object.__setattr__(
            self,
            "source_stack_components",
            source_stack_components,
        )

        grouping_metadata_fields = object.__getattribute__(
            self,
            "grouping_metadata_fields",
        )
        if grouping_metadata_fields is not None:
            grouping_metadata_fields = tuple(
                str(value).strip() for value in grouping_metadata_fields
            )
            if any(not value for value in grouping_metadata_fields):
                raise ValueError(
                    f"{type(self).__name__}.grouping_metadata_fields cannot contain "
                    "empty names."
                )
            grouping_metadata_fields = tuple(dict.fromkeys(grouping_metadata_fields))
        object.__setattr__(self, "grouping_metadata_fields", grouping_metadata_fields)

        source_voxel_spacing = object.__getattribute__(self, "source_voxel_spacing")
        if source_voxel_spacing is not None and not isinstance(
            source_voxel_spacing,
            SourceVoxelSpacing,
        ):
            raise TypeError(
                f"{type(self).__name__}.source_voxel_spacing must be "
                "SourceVoxelSpacing."
            )

    @property
    def has_primary_content(self) -> bool:
        return bool(self.binding_declarations or self.image_plane_sources)

    @property
    def source_filter_declarations(self) -> tuple[SourceFilterClause, ...]:
        """Source filters explicitly declared on this plan."""

        return tuple(self.source_filters or ())

    def resolved_source_locations(self, source_root: Path) -> Self:
        """Return this config with every path-bearing declaration resolved."""

        return replace(
            self,
            bindings=tuple(
                replace(
                    binding,
                    explicit_source=binding.explicit_source.resolved(source_root),
                )
                if binding.explicit_source is not None
                else binding
                for binding in self.binding_declarations
            ),
            image_plane_sources=tuple(
                source.resolved(source_root) for source in self.image_plane_sources
            ),
            imported_metadata_tables=tuple(
                table.resolved(source_root) for table in self.imported_metadata_tables
            ),
        )

    def resolved_imported_metadata_locations(
        self,
        source_root: Path,
        *,
        portable_roots: Sequence[Path] = (),
    ) -> Self:
        """Resolve imported tables through their nominal path authority."""

        return replace(
            self,
            imported_metadata_tables=tuple(
                table.resolved(source_root, portable_roots=portable_roots)
                for table in self.imported_metadata_tables
            ),
        )

    def coerce_metadata_scalar(
        self,
        field_name: str,
        value: SourceMetadataScalar,
    ) -> SourceMetadataScalar:
        """Apply the exact declared source-metadata field dtype."""

        field_spec = next(
            (field for field in self.metadata_fields if field.name == field_name),
            None,
        )
        if field_spec is None:
            return source_metadata_scalar(value)
        coerced = field_spec.coerce_scalar(value)
        if coerced is not None and not isinstance(coerced, (str, int, float, bool)):
            raise TypeError(
                f"Source metadata field {field_name!r} produced unsupported scalar "
                f"type {type(coerced).__name__}."
            )
        return source_metadata_scalar(coerced)

    def coerce_metadata(
        self,
        metadata: SourceMetadataMapping,
    ) -> dict[str, SourceMetadataValue]:
        """Apply declared field dtypes without changing metadata structure."""

        return {
            str(field_name): (
                {
                    str(nested_name): self.coerce_metadata_scalar(
                        str(nested_name),
                        nested_value,
                    )
                    for nested_name, nested_value in value.items()
                }
                if isinstance(value, Mapping)
                else self.coerce_metadata_scalar(str(field_name), value)
            )
            for field_name, value in metadata.items()
        }

    def metadata_fields_for_realized_source_metadata(
        self,
        realized_source_metadata: Iterable[SourceMetadataMapping] | None,
    ) -> tuple[FieldSpec, ...]:
        """Merge declarations with realized non-key source-literal fields."""

        declared_fields = tuple(self.metadata_fields or ())
        if realized_source_metadata is None:
            return declared_fields

        excluded_names = frozenset(
            join.imported_metadata_field
            for table in self.imported_metadata_tables
            for join in table.joins
        )
        declared_names = frozenset(field.name for field in declared_fields)
        values_by_name: dict[str, list[SourceMetadataScalar]] = {}
        for metadata in realized_source_metadata:
            for field_name, value in SourceMetadataRoleView(metadata).original_items():
                if field_name not in declared_names and field_name not in excluded_names:
                    values_by_name.setdefault(field_name, []).append(value)

        realized_fields = tuple(
            FieldSpec(
                field_name,
                next(iter(value_types)) if len(value_types) == 1 else None,
                required=False,
            )
            for field_name, values in values_by_name.items()
            for value_types in (
                frozenset(type(value) for value in values if value is not None),
            )
        )
        return FieldSpec.merge_exact(
            (declared_fields, realized_fields),
            context=f"{type(self).__name__} realized metadata fields",
        )

    @property
    def is_empty(self) -> bool:
        return (
            not self.has_primary_content
            and not self.source_filter_declarations
            and not self.metadata_rule_declarations
            and self.match_plan is None
            and not self.image_plane_sources
            and not self.imported_metadata_tables
            and not self.source_stack_components
            and not self.grouping_metadata_fields
            and not self.source_voxel_spacing.has_values
        )


@dataclass(frozen=True)
class StepSourceBindingsConfig(
    SourceBindingsConfig,
    Enableable,
):
    """Step-local source-binding config inheriting pipeline/plate defaults."""

    registry_key: ClassVar[str] = "editable"
    enabled: bool = False
    """Whether this step uses source-binding resolution instead of the prior step image stack."""

    def for_input_source(self, input_source: InputSource) -> Self:
        """Return the source declarations active for one resolved input source."""

        if self.enabled is None:
            raise ValueError(
                "StepSourceBindingsConfig.for_input_source requires "
                "ObjectState-resolved enabled state."
            )
        resolved_input_source = (
            input_source
            if isinstance(input_source, InputSource)
            else InputSource(input_source)
        )
        if self.enabled or resolved_input_source is InputSource.PIPELINE_START:
            return self
        return type(self)()


def source_binding_group_keys_for_group_by(
    source_bindings: StepSourceBindingsConfig,
    group_by: GroupBy,
    *,
    realized_source_metadata: Iterable[SourceMetadataMapping] | None = None,
) -> tuple[str, ...]:
    """Return ordered binding component values for the grouping component."""

    if not isinstance(source_bindings, StepSourceBindingsConfig):
        raise TypeError(
            "source_binding_group_keys_for_group_by requires "
            f"StepSourceBindingsConfig, got {type(source_bindings).__name__}."
        )
    resolved_group_by = group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
    if resolved_group_by.value is None:
        return ()
    component = AllComponents.from_value(resolved_group_by.value)
    realized_metadata = (
        None
        if realized_source_metadata is None
        else tuple(realized_source_metadata)
    )
    return tuple(
        dict.fromkeys(
            value
            for binding in source_bindings.primary_plane_bindings
            for value in binding.component_values(
                component,
                realized_source_metadata=realized_metadata,
            )
        )
    )


def source_bindings_defaults_to_base(
    defaults: SourceBindingsConfig,
) -> SourceBindingsConfig:
    """Return concrete source-binding defaults from eager or lazy config values."""
    from objectstate import get_base_type_for_lazy

    lazy_base_type = get_base_type_for_lazy(type(defaults))
    if lazy_base_type is SourceBindingsConfig:
        concrete_defaults = SourceBindingsConfig()
        return SourceBindingsConfig(
            **{
                item.name: (
                    resolved
                    if (
                        resolved := type(defaults).__getattribute__(
                            defaults,
                            item.name,
                        )
                    )
                    is not None
                    else object.__getattribute__(concrete_defaults, item.name)
                )
                for item in dataclass_fields(SourceBindingsConfig)
                if item.init
            }
        )
    if isinstance(defaults, SourceBindingsConfig):
        return defaults
    raise TypeError(
        "Source-binding defaults must resolve to SourceBindingsConfig, got "
        f"{type(defaults).__name__}."
    )


@dataclass(frozen=True, slots=True)
class CompiledSourceBindingPlan(SourceBindingDeclarationsMixin, _SourceBindingPlanBase):
    """Immutable compile-time source binding plan for one step."""

    registry_key: ClassVar[str] = "compiled"
    bindings: tuple[NamedSourceBinding, ...] = ()
    source_stack_components: tuple[AllComponents, ...] = ()
    enabled: bool = True

    @classmethod
    def empty(cls) -> "CompiledSourceBindingPlan":
        return cls(enabled=False)

    @classmethod
    def from_config(
        cls,
        config: StepSourceBindingsConfig,
        *,
        input_source: InputSource,
        realized_source_metadata: Iterable[SourceMetadataMapping] | None = None,
    ) -> "CompiledSourceBindingPlan":
        if not isinstance(config, StepSourceBindingsConfig):
            raise TypeError(
                "CompiledSourceBindingPlan.config must be "
                f"StepSourceBindingsConfig, got {type(config).__name__}."
            )
        enabled = bool(config.enabled) or input_source is InputSource.PIPELINE_START
        return cls(
            bindings=config.binding_declarations,
            metadata_rules=config.metadata_rule_declarations,
            match_plan=config.match_plan,
            metadata_fields=config.metadata_fields_for_realized_source_metadata(
                realized_source_metadata
            ),
            source_stack_components=config.source_stack_components,
            enabled=enabled,
        )

    def __post_init__(self) -> None:
        bindings = normalize_source_binding_values(
            "CompiledSourceBindingPlan.bindings",
            self.bindings,
            NamedSourceBinding,
        )
        seen_aliases: set[str] = set()
        for binding in bindings:
            if binding.alias in seen_aliases:
                raise ValueError(
                    "CompiledSourceBindingPlan.bindings contains duplicate alias "
                    f"{binding.alias!r}."
                )
            seen_aliases.add(binding.alias)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(
            self,
            "source_stack_components",
            ComponentSet.collect(self.source_stack_components).as_tuple(),
        )
        object.__setattr__(self, "enabled", bool(self.enabled))
        self._normalize_common_fields()

    @property
    def has_primary_content(self) -> bool:
        return self.enabled and bool(self.bindings)

    def __reduce__(
        self,
    ) -> tuple[
        object,
        tuple[
            tuple[NamedSourceBinding, ...],
            tuple[MetadataExtractionRule, ...],
            SourceBindingMatchPlan | None,
            tuple[FieldSpec, ...],
            tuple[AllComponents, ...],
            bool,
        ],
    ]:
        """Serialize source-binding plan state for multiprocessing."""
        return (
            self.__class__._from_pickled_state,
            (
                self.bindings,
                self.metadata_rules,
                self.match_plan,
                self.metadata_fields,
                self.source_stack_components,
                self.enabled,
            ),
        )

    @property
    def requires_step_input_selector_resolution(self) -> bool:
        """Whether any step-input binding needs selector-aware source matching."""

        return self.enabled and any(
            binding.origin is SourceBindingOrigin.STEP_INPUT
            and binding.requires_selector_resolution
            for binding in self.bindings
        )

    @classmethod
    def _from_pickled_state(
        cls,
        bindings: tuple[NamedSourceBinding, ...],
        metadata_rules: tuple[MetadataExtractionRule, ...],
        match_plan: SourceBindingMatchPlan | None,
        metadata_fields: tuple[FieldSpec, ...],
        source_stack_components: tuple[AllComponents, ...],
        enabled: bool,
    ) -> "CompiledSourceBindingPlan":
        return cls(
            bindings=bindings,
            metadata_rules=metadata_rules,
            match_plan=match_plan,
            metadata_fields=metadata_fields,
            source_stack_components=source_stack_components,
            enabled=enabled,
        )


@dataclass(frozen=True, slots=True)
class CompiledSourceUniversePlan:
    """Frozen source-file universe decisions for one compiled step."""

    requires_step_input_selector_resolution: bool = False
    requires_full_pipeline_source_universe: bool = False
    uses_pipeline_start_binding_origin: bool = False

    @classmethod
    def empty(cls) -> "CompiledSourceUniversePlan":
        return cls()

    @classmethod
    def from_source_binding_plan(
        cls,
        source_binding_plan: CompiledSourceBindingPlan,
    ) -> "CompiledSourceUniversePlan":
        uses_pipeline_start_binding_origin = (
            source_binding_plan.has_primary_content
            and any(
                binding.origin is SourceBindingOrigin.PIPELINE_START
                for binding in source_binding_plan.bindings
            )
        )
        return cls(
            requires_step_input_selector_resolution=(
                source_binding_plan.requires_step_input_selector_resolution
            ),
            requires_full_pipeline_source_universe=False,
            uses_pipeline_start_binding_origin=uses_pipeline_start_binding_origin,
        )


@dataclass(frozen=True, slots=True)
class SourceRuntimePathLookup:
    """Runtime path identities used by source-binding provenance maps."""

    file_path: str
    step_input_dir: str | None = None

    def keys(self) -> tuple[str, ...]:
        return _source_runtime_path_lookup_keys(self.file_path, self.step_input_dir)

    def first_value(
        self,
        mapping: Mapping[str, Any],
        *,
        include_native_path_fallback: bool = False,
    ) -> Any | None:
        for key in self.keys():
            value = mapping.get(key)
            if value is not None:
                return value
        if include_native_path_fallback:
            return mapping.get(_source_runtime_native_path(self.file_path))
        return None


@lru_cache(maxsize=65536)
def _source_runtime_path_lookup_keys(
    file_path: str,
    step_input_dir: str | None,
) -> tuple[str, ...]:
    """Return path lookup spellings for one runtime source path."""
    path = Path(file_path)
    keys = dict.fromkeys((str(file_path), path.as_posix()))
    if path.is_absolute() and step_input_dir is not None:
        try:
            relative_path = path.relative_to(step_input_dir)
        except ValueError:
            pass
        else:
            keys[relative_path.as_posix()] = None
    return tuple(keys)


@lru_cache(maxsize=65536)
def _source_runtime_native_path(file_path: str) -> str:
    """Return the native-path spelling used as the final runtime lookup fallback."""
    return str(Path(file_path))


@dataclass(frozen=True, slots=True)
class SourceBindingRuntimeMetadataNormalizer:
    """Normalize source metadata carried by a runtime source-binding context."""

    source_metadata_by_path: Mapping[str, SourceMetadataMapping]

    def normalized(self) -> Mapping[str, SourceMetadataMapping]:
        return MappingProxyType(
            {
                str(path): MappingProxyType(
                    {
                        str(key): self.normalized_value(value)
                        for key, value in source_metadata_dict(metadata).items()
                    }
                )
                for path, metadata in self.source_metadata_by_path.items()
            }
        )

    @classmethod
    def normalized_value(cls, value: SourceMetadataValue) -> SourceMetadataValue:
        if isinstance(value, Mapping):
            return MappingProxyType(
                {
                    str(key): cls.normalized_scalar(nested_value)
                    for key, nested_value in value.items()
                }
            )
        return cls.normalized_scalar(value)

    @staticmethod
    def normalized_scalar(value: SourceMetadataScalar) -> SourceMetadataScalar:
        return source_metadata_scalar(value)


@dataclass(frozen=True)
class SourceBindingRuntimeContext:
    """Execution-local file universe for selector-bearing source bindings."""

    step_input_files: tuple[str, ...] = ()
    current_step_input_files: tuple[str, ...] = ()
    current_image_files: tuple[str, ...] = ()
    step_input_dir: str | None = None
    step_input_source_backend: str | None = None
    step_input_storage_backend: str | None = None
    step_input_source_paths: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    source_metadata_by_path: Mapping[str, SourceMetadataMapping] = field(
        default_factory=lambda: MappingProxyType({})
    )
    pipeline_input_files: tuple[str, ...] = ()
    pipeline_source_candidate_files: tuple[str, ...] = ()
    pipeline_input_backend: str | None = None
    source_metadata_is_normalized: InitVar[bool] = False
    _source_metadata_identity: SourceMetadataIdentity | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _pipeline_input_files_identity: tuple[str, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _source_order_identity: tuple[Hashable, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _process_semantic_identity: SourceBindingRuntimeContextProcessIdentity | None = (
        field(
            default=None,
            init=False,
            repr=False,
            compare=False,
        )
    )
    _virtual_source_paths_by_identity: Mapping[str, tuple[str, ...]] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _source_metadata_by_runtime_lookup_key: (
        Mapping[
            str,
            SourceMetadataMapping,
        ]
        | None
    ) = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def empty(cls) -> "SourceBindingRuntimeContext":
        return cls()

    def __post_init__(
        self,
        source_metadata_is_normalized: bool,
    ) -> None:
        object.__setattr__(self, "step_input_files", tuple(self.step_input_files))
        object.__setattr__(
            self,
            "current_step_input_files",
            tuple(self.current_step_input_files or self.step_input_files),
        )
        object.__setattr__(
            self,
            "current_image_files",
            tuple(self.current_image_files or self.current_step_input_files),
        )
        if self.step_input_dir is not None:
            object.__setattr__(self, "step_input_dir", str(self.step_input_dir))
        if self.step_input_source_backend is not None:
            object.__setattr__(
                self,
                "step_input_source_backend",
                str(self.step_input_source_backend),
            )
        if self.step_input_storage_backend is not None:
            object.__setattr__(
                self,
                "step_input_storage_backend",
                str(self.step_input_storage_backend),
            )
        step_input_source_paths = self.step_input_source_paths
        if not isinstance(step_input_source_paths, MappingProxyType):
            step_input_source_paths = MappingProxyType(
                {
                    str(path): str(source)
                    for path, source in step_input_source_paths.items()
                }
            )
        object.__setattr__(self, "step_input_source_paths", step_input_source_paths)

        if source_metadata_is_normalized:
            if not isinstance(self.source_metadata_by_path, MappingProxyType):
                raise TypeError(
                    "Normalized SourceBindingRuntimeContext metadata must be "
                    "MappingProxyType."
                )
        else:
            object.__setattr__(
                self,
                "source_metadata_by_path",
                SourceBindingRuntimeMetadataNormalizer(
                    self.source_metadata_by_path
                ).normalized(),
            )
        object.__setattr__(
            self,
            "pipeline_input_files",
            tuple(self.pipeline_input_files),
        )
        object.__setattr__(
            self,
            "pipeline_source_candidate_files",
            tuple(
                self.pipeline_source_candidate_files
                or self.pipeline_input_files
                or self.step_input_files
            ),
        )
        if self.pipeline_input_backend is not None:
            object.__setattr__(
                self,
                "pipeline_input_backend",
                str(self.pipeline_input_backend),
            )

    @property
    def source_metadata_identity(
        self,
    ) -> SourceMetadataIdentity:
        """Stable identity for the complete source-metadata universe."""

        cached = self._source_metadata_identity
        if cached is None:
            cached = tuple(
                (path, SourceMetadataIdentityProjection(metadata).items())
                for path, metadata in sorted(self.source_metadata_by_path.items())
            )
            object.__setattr__(self, "_source_metadata_identity", cached)
        return cached

    @property
    def process_semantic_identity(self) -> SourceBindingRuntimeContextProcessIdentity:
        """Return the source context identity used by process-local caches."""
        cached = self._process_semantic_identity
        if cached is None:
            cached = SourceBindingRuntimeContextProcessIdentity(
                source_order_identity=self.source_order_identity,
                source_metadata_identity=self.source_metadata_identity,
            )
            object.__setattr__(self, "_process_semantic_identity", cached)
        return cached

    @property
    def source_metadata_by_runtime_lookup_key(
        self,
    ) -> Mapping[str, SourceMetadataMapping]:
        """Return source metadata indexed by every runtime path spelling."""
        cached = self._source_metadata_by_runtime_lookup_key
        if cached is None:
            indexed: dict[str, SourceMetadataMapping] = {}
            for path, metadata in self.source_metadata_by_path.items():
                for key in _source_runtime_path_lookup_keys(
                    str(path),
                    self.step_input_dir,
                ):
                    indexed.setdefault(key, metadata)
                indexed.setdefault(_source_runtime_native_path(str(path)), metadata)
            cached = MappingProxyType(indexed)
            object.__setattr__(
                self,
                "_source_metadata_by_runtime_lookup_key",
                cached,
            )
        return cached

    def source_metadata_for_runtime_path(
        self,
        path: str,
    ) -> SourceMetadataMapping | None:
        """Return source metadata for one runtime path spelling, if known."""
        lookup = self.source_metadata_by_runtime_lookup_key
        for key in _source_runtime_path_lookup_keys(str(path), self.step_input_dir):
            metadata = lookup.get(key)
            if metadata is not None:
                return metadata
        return lookup.get(_source_runtime_native_path(str(path)))

    @property
    def pipeline_input_files_identity(self) -> tuple[str, ...]:
        """Return sorted pipeline input files for source-order cache identities."""
        cached = self._pipeline_input_files_identity
        if cached is None:
            cached = tuple(sorted(self.pipeline_input_files))
            object.__setattr__(self, "_pipeline_input_files_identity", cached)
        return cached

    @property
    def source_order_identity(self) -> tuple[Hashable, ...]:
        """Return source-order mapping identity shared by runtime source caches."""
        cached = self._source_order_identity
        if cached is None:
            cached = (
                self.step_input_dir,
                tuple(sorted(self.pipeline_source_candidate_files)),
                tuple(sorted(self.step_input_source_paths.items())),
                tuple(sorted(self.virtual_source_paths_by_identity.items())),
            )
            object.__setattr__(self, "_source_order_identity", cached)
        return cached

    @property
    def virtual_source_paths_by_identity(self) -> Mapping[str, tuple[str, ...]]:
        """Return virtual source paths grouped by normalized physical identity."""

        cached = self._virtual_source_paths_by_identity
        if cached is None:
            grouped: dict[str, list[str]] = {}
            for virtual_path, source_path in self.step_input_source_paths.items():
                for identity in self.source_path_identities(source_path):
                    paths = grouped.get(identity)
                    if paths is None:
                        grouped[identity] = [virtual_path]
                        continue
                    paths.append(virtual_path)
            cached = MappingProxyType(
                {
                    identity: tuple(dict.fromkeys(paths))
                    for identity, paths in grouped.items()
                }
            )
            object.__setattr__(self, "_virtual_source_paths_by_identity", cached)
        return cached

    @staticmethod
    @lru_cache(maxsize=8192)
    def source_path_identities(source_path: str) -> tuple[str, ...]:
        """Return path identities for stored and resolved source-path spellings."""
        path = Path(source_path)
        return tuple(
            dict.fromkeys(
                (
                    source_path_identity_key(source_path),
                    source_path_identity_key(str(path.resolve(strict=False))),
                )
            )
        )

    def path_spellings(self, paths: Sequence[str]) -> tuple[str, ...]:
        """Return virtual and physical spellings represented by selected paths."""

        selected: list[str] = []
        for path in paths:
            selected.append(str(path))
            mapped = self.step_input_source_paths.get(str(path))
            if mapped is not None:
                selected.append(mapped)
            for identity in self.source_path_identities(str(path)):
                selected.extend(self.virtual_source_paths_by_identity.get(identity, ()))
        return tuple(dict.fromkeys(selected))

    def metadata_identity_for_paths(
        self,
        paths: tuple[str, ...],
    ) -> SourceMetadataIdentity:
        """Return the stable metadata identity for a selected source subset."""

        identity: list[tuple[str, SourceMetadataIdentityItems]] = []
        for path in paths:
            if path in self.source_metadata_by_path:
                metadata = SourceMetadataIdentityProjection(
                    self.source_metadata_by_path[path]
                ).items()
            else:
                metadata = ()
            identity.append((path, metadata))
        return tuple(identity)

    def source_candidate_file_universes(self) -> tuple[tuple[str, ...], ...]:
        """Return distinct non-empty file universes that may be source-parsed."""
        return tuple(
            dict.fromkeys(
                files
                for files in (
                    self.step_input_files,
                    self.current_step_input_files,
                    self.pipeline_source_candidate_files,
                )
                if files
            )
        )

    def __reduce__(
        self,
    ) -> tuple[
        object,
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            str | None,
            str | None,
            str | None,
            dict[str, str],
            dict[str, dict[str, SourceMetadataValue]],
            tuple[str, ...],
            tuple[str, ...],
            str | None,
        ],
    ]:
        """Serialize mappingproxy-backed provenance as a plain dict."""
        return (
            self.__class__,
            (
                self.step_input_files,
                self.current_step_input_files,
                self.current_image_files,
                self.step_input_dir,
                self.step_input_source_backend,
                self.step_input_storage_backend,
                dict(self.step_input_source_paths),
                {
                    path: source_metadata_dict(metadata)
                    for path, metadata in self.source_metadata_by_path.items()
                },
                self.pipeline_input_files,
                self.pipeline_source_candidate_files,
                self.pipeline_input_backend,
            ),
        )


EMPTY_SOURCE_BINDINGS = StepSourceBindingsConfig()


def _coerce_component(value: Any, field_name: str) -> Any:
    if isinstance(value, AllComponents):
        return value
    if isinstance(value, Enum) and (
        converted := convert_enum_by_value(value, AllComponents)
    ):
        return converted
    return AllComponents(value, )


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
