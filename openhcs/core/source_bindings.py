"""Typed source-binding semantics for named step input views."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.components.validation import convert_enum_by_value
from openhcs.core.runtime_semantics import coerce_enum


SourceBindingGroupMap = Mapping[str | None, tuple["NamedSourceBinding", ...]]
SourceBindingGroupDict = dict[str | None, tuple["NamedSourceBinding", ...]]
SourceMetadataIdentity = tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
SOURCE_ALIAS_PART_SEPARATOR = "__"


class SourceBindingOrigin(Enum):
    """Where a named binding should be resolved from."""

    STEP_INPUT = "step_input"
    PIPELINE_START = "pipeline_start"


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subject",
            coerce_enum(
                SourceFilterSubject,
                self.subject,
                "SourceFilterClause.subject",
            ),
        )
        match_type = coerce_enum(
            SourceFilterMatchType,
            self.match_type,
            "SourceFilterClause.match_type",
        )
        object.__setattr__(self, "match_type", match_type)
        normalized_value = None if self.value is None else str(self.value)
        if not match_type.requires_value:
            object.__setattr__(self, "value", None)
            return
        if normalized_value is None:
            raise ValueError(
                "SourceFilterClause.value is required unless match_type is IS_IMAGE."
            )
        object.__setattr__(self, "value", normalized_value)


@dataclass(frozen=True, slots=True)
class SourceBindingTypedValues:
    """Validated tuple of values for a typed source-binding field."""

    field_name: str
    values: tuple[object, ...]
    value_type: type[object]

    def normalized(self) -> tuple[object, ...]:
        normalized_values = tuple(self.values)
        for value in normalized_values:
            if not isinstance(value, self.value_type):
                raise TypeError(
                    f"{self.field_name} must contain {self.value_type.__name__} "
                    f"values, got {type(value).__name__}."
                )
        return normalized_values


@dataclass(frozen=True, slots=True)
class SourceBindingUniqueValues:
    """Validated source-binding values with unique semantic identities."""

    field_name: str
    values: tuple[object, ...]
    value_type: type[object]
    identity: Callable[[Any], object]
    duplicate_message: Callable[[object], str]

    def normalized(self) -> tuple[object, ...]:
        normalized_values = SourceBindingTypedValues(
            self.field_name,
            self.values,
            self.value_type,
        ).normalized()
        seen: set[object] = set()
        for value in normalized_values:
            identity = self.identity(value)
            if identity in seen:
                raise ValueError(self.duplicate_message(identity))
            seen.add(identity)
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
            coerce_enum(
                MetadataSource,
                self.source,
                "MetadataExtractionRule.source",
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
            SourceBindingTypedValues(
                "MetadataExtractionRule.filters",
                self.filters,
                SourceFilterClause,
            ).normalized(),
        )


class SourceBindingMatchMethod(Enum):
    """How a source binding plan matches related source aliases into one image set."""

    METADATA = "metadata"
    ORDER = "order"


@dataclass(frozen=True, slots=True)
class SourceBindingMatchField:
    """One alias-local metadata field participating in image-set matching."""

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
        return SourceBindingUniqueValues(
            "SourceBindingMatchDimension.fields",
            self.fields,
            SourceBindingMatchField,
            lambda field: field.alias,
            lambda alias: (
                "SourceBindingMatchDimension contains duplicate alias "
                f"{alias!r}."
            ),
        ).normalized()


@dataclass(frozen=True, slots=True)
class SourceBindingMatchDimension:
    """One logical image-set matching slot shared across aliases."""

    fields: tuple[SourceBindingMatchField, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fields",
            SourceBindingMatchFields(self.fields).normalized(),
        )

    def field_for_alias(self, alias: str) -> str | None:
        for field in self.fields:
            if field.alias == alias:
                return field.metadata_field
        return None


@dataclass(frozen=True, slots=True)
class SourceBindingMatchPlan:
    """Typed cross-alias matching plan for source image sets."""

    method: SourceBindingMatchMethod
    dimensions: tuple[SourceBindingMatchDimension, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "method",
            coerce_enum(
                SourceBindingMatchMethod,
                self.method,
                "SourceBindingMatchPlan.method",
            ),
        )
        object.__setattr__(
            self,
            "dimensions",
            SourceBindingTypedValues(
                "SourceBindingMatchPlan.dimensions",
                self.dimensions,
                SourceBindingMatchDimension,
            ).normalized(),
        )


@dataclass(frozen=True, slots=True)
class ComponentSelector:
    """Typed component-key selector in existing OpenHCS vocabulary."""

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
    """Typed metadata-field selector for source binding resolution."""

    field: str
    value: str

    def __post_init__(self) -> None:
        _require_name(self.field, "MetadataSelector.field")
        if self.value == "":
            raise ValueError("MetadataSelector.value cannot be empty.")
        object.__setattr__(self, "field", str(self.field))
        object.__setattr__(self, "value", str(self.value))


@dataclass(frozen=True, slots=True)
class SourceSelector:
    """Selector describing how a named source view maps to input space."""

    components: tuple[ComponentSelector, ...] = ()
    metadata: tuple[MetadataSelector, ...] = ()
    filters: tuple[SourceFilterClause, ...] = ()
    inherit_current_scope: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "components",
            SourceBindingTypedValues(
                "SourceSelector.components",
                self.components,
                ComponentSelector,
            ).normalized(),
        )
        object.__setattr__(
            self,
            "metadata",
            SourceBindingTypedValues(
                "SourceSelector.metadata",
                self.metadata,
                MetadataSelector,
            ).normalized(),
        )
        object.__setattr__(
            self,
            "filters",
            SourceBindingTypedValues(
                "SourceSelector.filters",
                self.filters,
                SourceFilterClause,
            ).normalized(),
        )


def source_alias_measurement_names(alias: str) -> tuple[str, ...]:
    """Return measurement source-name tokens represented by a source alias."""
    normalized_alias = alias.strip()
    if not normalized_alias:
        return ()
    parts = tuple(
        part
        for part in normalized_alias.split(SOURCE_ALIAS_PART_SEPARATOR)
        if part
    )
    return parts or (normalized_alias,)


@dataclass(frozen=True, slots=True)
class SourceAssignmentBase(metaclass=AutoRegisterMeta):
    """Shared source-assignment identity and selector contract."""

    __registry_key__ = "assignment_kind"
    __skip_if_no_key__ = True
    assignment_kind: ClassVar[str | None] = None

    alias: str
    selector: SourceSelector = SourceSelector()
    origin: SourceBindingOrigin = SourceBindingOrigin.STEP_INPUT

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
        object.__setattr__(
            self,
            "origin",
            coerce_enum(
                SourceBindingOrigin,
                self.origin,
                f"{type(self).__name__}.origin",
            ),
        )

    @property
    def artifact_kind(self) -> ArtifactKind:
        """Artifact kind bound by this source assignment."""
        raise NotImplementedError(
            f"{type(self).__name__} must provide artifact_kind."
        )

    @property
    def measurement_source_names(self) -> tuple[str, ...]:
        """Return measurement feature source qualifiers declared by this alias."""
        if not self.artifact_kind.participates_in_measurement_source_names:
            return ()
        return source_alias_measurement_names(self.alias)

    def to_binding(self) -> "NamedSourceBinding":
        """Project this source assignment into a step-local source binding."""
        return NamedSourceBinding(
            alias=self.alias,
            artifact_kind=self.artifact_kind,
            selector=self.selector,
            origin=self.origin,
        )


@dataclass(frozen=True, slots=True)
class NamedSourceBinding(SourceAssignmentBase):
    """Semantic alias mapped to a typed selector over step input space."""

    assignment_kind = "named_source_binding"
    artifact_kind: ArtifactKind = ArtifactKind.IMAGE
    required: bool = True

    def __post_init__(self) -> None:
        SourceAssignmentBase.__post_init__(self)
        object.__setattr__(
            self,
            "artifact_kind",
            coerce_enum(
                ArtifactKind,
                self.artifact_kind,
                "NamedSourceBinding.artifact_kind",
            ),
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

    @property
    def requires_step_input_channel_stack(self) -> bool:
        """Whether resolving this binding needs channel-varying step input."""

        if self.origin is not SourceBindingOrigin.STEP_INPUT:
            return False
        return bool(
            self.selector.filters
            or self.selector.metadata
            or any(
                selector.component is AllComponents.CHANNEL
                for selector in self.selector.components
            )
        )

    @property
    def participates_in_execution_anchoring(self) -> bool:
        """Whether this binding contributes source-file execution anchors."""

        return self.artifact_kind is ArtifactKind.IMAGE


@dataclass(frozen=True, slots=True)
class GroupedSourceBindings:
    """Bindings scoped to one function-pattern or execution group."""

    group_key: str | None = None
    bindings: tuple[NamedSourceBinding, ...] = ()

    def __post_init__(self) -> None:
        normalized_group_key = None if self.group_key is None else str(self.group_key)
        object.__setattr__(self, "group_key", normalized_group_key)
        object.__setattr__(self, "bindings", tuple(self.bindings))
        seen_aliases: set[str] = set()
        for binding in self.bindings:
            if not isinstance(binding, NamedSourceBinding):
                raise TypeError(
                    "GroupedSourceBindings.bindings must contain NamedSourceBinding values, "
                    f"got {type(binding).__name__}."
                )
            if binding.alias in seen_aliases:
                raise ValueError(
                    f"GroupedSourceBindings for group {normalized_group_key!r} contains "
                    f"duplicate alias {binding.alias!r}."
                )
            seen_aliases.add(binding.alias)


@dataclass(frozen=True, slots=True)
class SourceBindingGroups:
    """Validated grouped binding declarations keyed by source group."""

    groups: tuple[GroupedSourceBindings, ...]

    def normalized(self) -> tuple[GroupedSourceBindings, ...]:
        return SourceBindingUniqueValues(
            "StepSourceBindingsConfig.groups",
            self.groups,
            GroupedSourceBindings,
            lambda group: group.group_key,
            lambda group_key: (
                "StepSourceBindingsConfig contains duplicate group key "
                f"{group_key!r}."
            ),
        ).normalized()


@dataclass(frozen=True, slots=True, kw_only=True)
class _SourceBindingPlanBase(ABC, metaclass=AutoRegisterMeta):
    """Shared typed source-binding plan fields across editable and compiled views."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None
    metadata_rules: tuple[MetadataExtractionRule, ...] = ()
    match_plan: SourceBindingMatchPlan | None = None

    @classmethod
    def registered_plan_types(cls) -> tuple[type["_SourceBindingPlanBase"], ...]:
        """Return registered concrete source-binding plan views."""

        return tuple(cls.__registry__.values())

    def _normalize_common_fields(self) -> None:
        object.__setattr__(self, "metadata_rules", tuple(self.metadata_rules))
        for rule in self.metadata_rules:
            if not isinstance(rule, MetadataExtractionRule):
                raise TypeError(
                    f"{type(self).__name__}.metadata_rules must contain "
                    "MetadataExtractionRule values, got "
                    f"{type(rule).__name__}."
                )
        if self.match_plan is not None and not isinstance(
            self.match_plan,
            SourceBindingMatchPlan,
        ):
            raise TypeError(
                f"{type(self).__name__}.match_plan must be SourceBindingMatchPlan "
                f"or None, got {type(self.match_plan).__name__}."
            )

    @property
    def is_empty(self) -> bool:
        return (
            not self.has_primary_content
            and not self.metadata_rules
            and self.match_plan is None
        )

    @property
    @abstractmethod
    def has_primary_content(self) -> bool:
        """Whether the subclass-specific binding payload is empty."""


@dataclass(frozen=True, slots=True)
class StepSourceBindingsConfig(_SourceBindingPlanBase):
    """First-class FunctionStep field for named semantic input bindings."""

    registry_key: ClassVar[str] = "editable"
    groups: tuple[GroupedSourceBindings, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "groups",
            SourceBindingGroups(self.groups).normalized(),
        )
        self._normalize_common_fields()

    @property
    def has_primary_content(self) -> bool:
        return bool(self.groups)

    @property
    def requires_step_input_channel_stack(self) -> bool:
        """Whether any binding needs channel-resolved stack input."""

        return any(
            binding.requires_step_input_channel_stack
            for group in self.groups
            for binding in group.bindings
        )

    @property
    def requires_pipeline_start_resolution(self) -> bool:
        """Whether any binding resolves from the pipeline-start source universe."""

        return any(
            binding.origin is SourceBindingOrigin.PIPELINE_START
            for group in self.groups
            for binding in group.bindings
        )

    @property
    def requires_step_input_selector_resolution(self) -> bool:
        """Whether any step-input binding needs selector-aware source matching."""

        return any(
            binding.origin is SourceBindingOrigin.STEP_INPUT
            and binding.requires_selector_resolution
            for group in self.groups
            for binding in group.bindings
        )


@dataclass(frozen=True, slots=True)
class CompiledSourceBindingPlan(_SourceBindingPlanBase):
    """Immutable compile-time source binding plan for one step."""

    registry_key: ClassVar[str] = "compiled"
    bindings_by_group: SourceBindingGroupMap = field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    def empty(cls) -> "CompiledSourceBindingPlan":
        return cls()

    @classmethod
    def from_config(
        cls,
        config: StepSourceBindingsConfig,
    ) -> "CompiledSourceBindingPlan":
        if config.is_empty:
            return cls.empty()
        return cls(
            bindings_by_group=MappingProxyType(
                {group.group_key: group.bindings for group in config.groups}
            ),
            metadata_rules=config.metadata_rules,
            match_plan=config.match_plan,
        )

    def __post_init__(self) -> None:
        normalized: SourceBindingGroupDict = {}
        for group_key, bindings in self.bindings_by_group.items():
            normalized_group_key = None if group_key is None else str(group_key)
            normalized_bindings = tuple(bindings)
            for binding in normalized_bindings:
                if not isinstance(binding, NamedSourceBinding):
                    raise TypeError(
                        "CompiledSourceBindingPlan bindings must contain NamedSourceBinding values, "
                        f"got {type(binding).__name__}."
                    )
            if normalized_group_key in normalized:
                raise ValueError(
                    f"CompiledSourceBindingPlan contains duplicate group key "
                    f"{normalized_group_key!r}."
                )
            normalized[normalized_group_key] = normalized_bindings
        object.__setattr__(self, "bindings_by_group", MappingProxyType(normalized))
        self._normalize_common_fields()

    @property
    def has_primary_content(self) -> bool:
        return bool(self.bindings_by_group)

    def __reduce__(
        self,
    ) -> tuple[
        object,
        tuple[
            SourceBindingGroupDict,
            tuple[MetadataExtractionRule, ...],
            SourceBindingMatchPlan | None,
        ],
    ]:
        """Serialize mappingproxy-backed state as a plain dict for multiprocessing."""
        return (
            self.__class__._from_pickled_state,
            (dict(self.bindings_by_group), self.metadata_rules, self.match_plan),
        )

    def bindings_for_group(
        self,
        group_key: str | None,
    ) -> tuple[NamedSourceBinding, ...]:
        normalized_group_key = None if group_key is None else str(group_key)
        if normalized_group_key in self.bindings_by_group:
            return self.bindings_by_group[normalized_group_key]
        return self.bindings_by_group.get(None, ())

    def binding_for_alias(
        self,
        alias: str,
        group_key: str | None,
    ) -> NamedSourceBinding | None:
        for binding in self.bindings_for_group(group_key):
            if binding.alias == alias:
                return binding
        return None

    @property
    def requires_step_input_selector_resolution(self) -> bool:
        """Whether any step-input binding needs selector-aware source matching."""

        return any(
            binding.origin is SourceBindingOrigin.STEP_INPUT
            and binding.requires_selector_resolution
            for bindings in self.bindings_by_group.values()
            for binding in bindings
        )

    @classmethod
    def _from_pickled_state(
        cls,
        bindings_by_group: SourceBindingGroupDict,
        metadata_rules: tuple[MetadataExtractionRule, ...],
        match_plan: SourceBindingMatchPlan | None,
    ) -> "CompiledSourceBindingPlan":
        return cls(
            bindings_by_group=bindings_by_group,
            metadata_rules=metadata_rules,
            match_plan=match_plan,
        )


@dataclass(frozen=True, slots=True)
class SourceRuntimePathLookup:
    """Runtime path identities used by source-binding provenance maps."""

    file_path: str
    step_input_dir: str | None = None

    def keys(self) -> tuple[str, ...]:
        path = Path(self.file_path)
        keys = dict.fromkeys((str(self.file_path), path.as_posix()))
        if path.is_absolute() and self.step_input_dir is not None:
            try:
                relative_path = path.relative_to(self.step_input_dir)
            except ValueError:
                pass
            else:
                keys[relative_path.as_posix()] = None
        return tuple(keys)

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
            return mapping.get(str(Path(self.file_path)))
        return None


@dataclass(frozen=True, slots=True)
class SourceBindingRuntimeContext:
    """Execution-local file universe for selector-bearing source bindings."""

    step_input_files: tuple[str, ...] = ()
    current_step_input_files: tuple[str, ...] = ()
    step_input_dir: str | None = None
    step_input_backend: str | None = None
    step_input_source_paths: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    source_metadata_by_path: Mapping[str, Mapping[str, str]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    pipeline_input_files: tuple[str, ...] = ()
    pipeline_input_backend: str | None = None
    _source_metadata_identity: SourceMetadataIdentity | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def empty(cls) -> "SourceBindingRuntimeContext":
        return cls()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_input_files", tuple(self.step_input_files))
        object.__setattr__(
            self,
            "current_step_input_files",
            tuple(self.current_step_input_files or self.step_input_files),
        )
        if self.step_input_dir is not None:
            object.__setattr__(self, "step_input_dir", str(self.step_input_dir))
        if self.step_input_backend is not None:
            object.__setattr__(
                self,
                "step_input_backend",
                str(self.step_input_backend),
            )
        step_input_source_paths = self.step_input_source_paths
        if not isinstance(step_input_source_paths, MappingProxyType):
            step_input_source_paths = MappingProxyType(
                {str(path): str(source) for path, source in step_input_source_paths.items()}
            )
        object.__setattr__(self, "step_input_source_paths", step_input_source_paths)

        source_metadata_by_path = self.source_metadata_by_path
        if not isinstance(source_metadata_by_path, MappingProxyType):
            source_metadata_by_path = MappingProxyType(
                {
                    str(path): MappingProxyType(
                        {str(key): str(value) for key, value in metadata.items()}
                    )
                    for path, metadata in source_metadata_by_path.items()
                }
            )
        object.__setattr__(self, "source_metadata_by_path", source_metadata_by_path)
        object.__setattr__(
            self,
            "pipeline_input_files",
            tuple(self.pipeline_input_files),
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
                (path, tuple(sorted(metadata.items())))
                for path, metadata in sorted(self.source_metadata_by_path.items())
            )
            object.__setattr__(self, "_source_metadata_identity", cached)
        return cached

    def metadata_identity_for_paths(
        self,
        paths: tuple[str, ...],
    ) -> SourceMetadataIdentity:
        """Return the stable metadata identity for a selected source subset."""

        return tuple(
            (path, tuple(sorted(self.source_metadata_by_path.get(path, {}).items())))
            for path in paths
        )

    def __reduce__(
        self,
    ) -> tuple[
        object,
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            str | None,
            str | None,
            dict[str, str],
            dict[str, dict[str, str]],
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
                self.step_input_dir,
                self.step_input_backend,
                dict(self.step_input_source_paths),
                {
                    path: dict(metadata)
                    for path, metadata in self.source_metadata_by_path.items()
                },
                self.pipeline_input_files,
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
    return coerce_enum(AllComponents, value, field_name)


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
