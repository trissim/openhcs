"""Typed source-binding semantics for named step input views."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import InitVar, dataclass, field, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Self, TypeVar

from metaclass_registry import AutoRegisterMeta
from python_introspect import Enableable
from python_introspect.enableable import EnableableMeta

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactType, ImageArtifactType
from openhcs.core.components.validation import convert_enum_by_value
from openhcs.core.runtime_semantics import coerce_enum
from openhcs.core.source_metadata import (
    SourceMetadataIdentityItems,
    SourceMetadataIdentityProjection,
    SourceMetadataMapping,
    SourceMetadataScalar,
    SourceMetadataValue,
    source_metadata_scalar,
)
from openhcs.core.source_path_identity import source_path_identity_key


SourceMetadataIdentity = tuple[tuple[str, SourceMetadataIdentityItems], ...]
SOURCE_ALIAS_PART_SEPARATOR = "__"
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
            normalize_source_binding_values(
                "MetadataExtractionRule.filters",
                self.filters,
                SourceFilterClause,
            ),
        )


class SourceBindingMatchMethod(Enum):
    """How selected source aliases are paired into one logical image set."""

    METADATA = "metadata"
    ORDER = "order"


@dataclass(frozen=True, slots=True)
class SourceBindingMatchField:
    """Metadata field from one alias used as an image-set pairing key."""

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
        for field in fields:
            if field.alias in seen_aliases:
                raise ValueError(
                    "SourceBindingMatchDimension contains duplicate alias "
                    f"{field.alias!r}."
                )
            seen_aliases.add(field.alias)
        return fields


@dataclass(frozen=True, slots=True)
class SourceBindingMatchDimension:
    """One shared image-set key, expressed as alias-to-metadata-field pairs."""

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
    """Cross-alias pairing plan for assembling selected sources into image sets."""

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
    value: str

    def __post_init__(self) -> None:
        _require_name(self.field, "MetadataSelector.field")
        if self.value == "":
            raise ValueError("MetadataSelector.value cannot be empty.")
        object.__setattr__(self, "field", str(self.field))
        object.__setattr__(self, "value", str(self.value))


@dataclass(frozen=True, slots=True)
class SourceSelector:
    """Filters that choose candidate input sources for one named alias."""

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
        part
        for part in normalized_alias.split(SOURCE_ALIAS_PART_SEPARATOR)
        if part
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
            coerce_enum(
                SourceBindingOrigin,
                self.origin,
                f"{type(self).__name__}.origin",
            ),
        )

    @property
    def artifact_kind(self) -> ArtifactType:
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

    @property
    def participates_in_image_stack(self) -> bool:
        """Whether this source assignment contributes to the primary image stack."""
        return self.artifact_kind is ImageArtifactType

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

    def with_component_identity(self, selector: ComponentSelector) -> Self:
        """Return this source assignment with one canonical component identity."""
        return replace(
            self,
            component_identity=self.component_identity_with(selector),
        )

    def to_binding(self) -> "NamedSourceBinding":
        """Project this source assignment into a step-local source binding."""
        return NamedSourceBinding(
            alias=self.alias,
            artifact_kind=self.artifact_kind,
            selector=self.selector,
            origin=self.origin,
            component_identity=self.component_identity,
            participates_in_image_stack=self.participates_in_image_stack,
        )


@dataclass(frozen=True, slots=True)
class NamedSourceBinding(SourceAssignmentBase):
    """Function input alias mapped to selected sources and assigned identity."""

    assignment_kind = "named_source_binding"
    artifact_kind: ArtifactType = ImageArtifactType
    required: bool = True
    participates_in_image_stack: bool = True
    """Whether this image binding creates primary source-image execution anchors."""

    def __post_init__(self) -> None:
        SourceAssignmentBase.__post_init__(self)
        object.__setattr__(
            self,
            "artifact_kind",
            ArtifactType.coerce(self.artifact_kind),
        )
        object.__setattr__(
            self,
            "participates_in_image_stack",
            bool(self.participates_in_image_stack),
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
        return any(selector.component in components for selector in self.selector.components)

    @property
    def requires_step_input_channel_stack(self) -> bool:
        """Whether resolving this binding needs channel-varying step input."""
        return self.requires_step_input_component_stack((AllComponents.CHANNEL,))

    @property
    def participates_in_execution_anchoring(self) -> bool:
        """Whether this binding contributes source-file execution anchors."""

        return (
            self.artifact_kind is ImageArtifactType
            and self.participates_in_image_stack
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
    """Optional strategy for pairing selected aliases into logical image sets."""

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

    @property
    def is_empty(self) -> bool:
        return (
            not self.has_primary_content
            and not self.metadata_rule_declarations
            and self.match_plan is None
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
class SourceBindingsConfig(_SourceBindingPlanBase):
    """Pipeline/plate source-binding defaults and init-time discovery config."""

    registry_key: ClassVar[str] = "source"
    source_filters: tuple[SourceFilterClause, ...] = ()
    """Filters limiting the source universe before named bindings are resolved."""

    bindings: tuple[NamedSourceBinding, ...] = ()
    """Named semantic source bindings available to pipelines and inherited by steps."""

    def __post_init__(self) -> None:
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
        self._normalize_common_fields()

    @property
    def has_primary_content(self) -> bool:
        return bool(self.binding_declarations)

    @property
    def source_filter_declarations(self) -> tuple[SourceFilterClause, ...]:
        """Source filters explicitly declared on this plan."""

        return tuple(self.source_filters or ())

    @property
    def binding_declarations(self) -> tuple[NamedSourceBinding, ...]:
        """Named bindings explicitly declared on this plan."""

        return tuple(self.bindings or ())

    @property
    def image_stack_bindings(self) -> tuple[NamedSourceBinding, ...]:
        """Bindings that anchor execution to the primary source-image stack."""

        return tuple(
            binding
            for binding in self.binding_declarations
            if binding.participates_in_execution_anchoring
        )

    def bindings_for_group_key(
        self,
        group_key: str,
    ) -> tuple[NamedSourceBinding, ...]:
        """Return bindings whose declared component identity matches a pattern group."""
        normalized_group_key = str(group_key)
        if normalized_group_key == "default":
            return self.binding_declarations
        matching_bindings = tuple(
            binding
            for binding in self.binding_declarations
            if any(
                str(selector.value) == normalized_group_key
                for selector in binding.component_identity
            )
        )
        if matching_bindings:
            return matching_bindings
        if len(self.binding_declarations) <= 1:
            return self.binding_declarations
        binding_identities = {
            binding.alias: tuple(
                (selector.component.value, selector.value)
                for selector in binding.component_identity
            )
            for binding in self.binding_declarations
        }
        raise ValueError(
            f"Source binding group {normalized_group_key!r} does not match any "
            "declared component identity. Available binding identities: "
            f"{binding_identities!r}."
        )

    def for_group_key(
        self,
        group_key: str,
    ) -> Self:
        """Return this source-binding config scoped to one function-pattern group."""
        return replace(self, bindings=self.bindings_for_group_key(group_key))

    @property
    def is_empty(self) -> bool:
        return (
            not self.has_primary_content
            and not self.source_filter_declarations
            and not self.metadata_rule_declarations
            and self.match_plan is None
        )


@dataclass(frozen=True)
class StepSourceBindingsConfig(
    SourceBindingsConfig,
    Enableable,
):
    """Step-local source-binding config inheriting pipeline/plate defaults."""

    registry_key: ClassVar[str] = "editable"
    metadata_rules: tuple[MetadataExtractionRule, ...] | None = None
    """Step-local extraction rules; None inherits pipeline/plate metadata rules."""

    match_plan: SourceBindingMatchPlan | None = None
    """Step-local matching strategy; None inherits the pipeline/plate match plan."""

    source_filters: tuple[SourceFilterClause, ...] | None = None
    """Step-local source filters; None inherits pipeline/plate source filters."""

    bindings: tuple[NamedSourceBinding, ...] | None = None
    """Step-local named source bindings; None inherits pipeline/plate bindings."""

    enabled: bool = False
    """Whether this step uses source-binding resolution instead of the prior step image stack."""

    def requires_step_input_component_stack(
        self,
        components: tuple[AllComponents, ...],
    ) -> bool:
        """Whether any binding needs component-resolved stack input."""
        return any(
            binding.requires_step_input_component_stack(components)
            for binding in self.binding_declarations
        )

    @property
    def requires_step_input_channel_stack(self) -> bool:
        """Whether any binding needs channel-resolved step input."""
        return self.requires_step_input_component_stack((AllComponents.CHANNEL,))

    @property
    def requires_pipeline_start_resolution(self) -> bool:
        """Whether any binding resolves from the pipeline-start source universe."""

        return any(
            binding.origin is SourceBindingOrigin.PIPELINE_START
            for binding in self.binding_declarations
        )

    @property
    def requires_pipeline_start_image_set_stack(self) -> bool:
        """Whether pipeline-start bindings form multi-alias image sets."""

        return (
            sum(
                1
                for binding in self.binding_declarations
                if binding.origin is SourceBindingOrigin.PIPELINE_START
                and binding.participates_in_execution_anchoring
            )
            > 1
        )

    @property
    def requires_step_input_selector_resolution(self) -> bool:
        """Whether any step-input binding needs selector-aware source matching."""

        return any(
            binding.origin is SourceBindingOrigin.STEP_INPUT
            and binding.requires_selector_resolution
            for binding in self.binding_declarations
        )


@dataclass(frozen=True, slots=True)
class CompiledSourceBindingPlan(_SourceBindingPlanBase):
    """Immutable compile-time source binding plan for one step."""

    registry_key: ClassVar[str] = "compiled"
    bindings: tuple[NamedSourceBinding, ...] = ()

    @classmethod
    def empty(cls) -> "CompiledSourceBindingPlan":
        return cls()

    @classmethod
    def from_config(
        cls,
        config: StepSourceBindingsConfig,
    ) -> "CompiledSourceBindingPlan":
        if not isinstance(config, StepSourceBindingsConfig):
            raise TypeError(
                "CompiledSourceBindingPlan.config must be "
                f"StepSourceBindingsConfig, got {type(config).__name__}."
            )
        if config.enabled is None:
            raise ValueError(
                "CompiledSourceBindingPlan requires ObjectState-resolved "
                "StepSourceBindingsConfig.enabled; unresolved lazy enabled=None "
                "cannot be compiled."
            )
        if config.enabled:
            return cls.from_enabled_config(config)
        return cls.empty()

    @classmethod
    def from_enabled_config(
        cls,
        config: StepSourceBindingsConfig,
    ) -> "CompiledSourceBindingPlan":
        if config.is_empty:
            return cls.empty()
        return cls(
            bindings=config.binding_declarations,
            metadata_rules=config.metadata_rule_declarations,
            match_plan=config.match_plan,
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
        self._normalize_common_fields()

    @property
    def has_primary_content(self) -> bool:
        return bool(self.bindings)

    def __reduce__(
        self,
    ) -> tuple[
        object,
        tuple[
            tuple[NamedSourceBinding, ...],
            tuple[MetadataExtractionRule, ...],
            SourceBindingMatchPlan | None,
        ],
    ]:
        """Serialize source-binding plan state for multiprocessing."""
        return (
            self.__class__._from_pickled_state,
            (self.bindings, self.metadata_rules, self.match_plan),
        )

    def binding_for_alias(
        self,
        alias: str,
    ) -> NamedSourceBinding | None:
        for binding in self.bindings:
            if binding.alias == alias:
                return binding
        return None

    @property
    def requires_step_input_selector_resolution(self) -> bool:
        """Whether any step-input binding needs selector-aware source matching."""

        return any(
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
    ) -> "CompiledSourceBindingPlan":
        return cls(
            bindings=bindings,
            metadata_rules=metadata_rules,
            match_plan=match_plan,
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
        uses_pipeline_start_binding_origin = any(
            binding.origin is SourceBindingOrigin.PIPELINE_START
            for binding in source_binding_plan.bindings
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
                        for key, value in metadata.items()
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
    source_binding_context: InitVar["SourceBindingRuntimeContext | None"] = None
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
    _process_semantic_identity: SourceBindingRuntimeContextProcessIdentity | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _virtual_source_paths_by_identity: Mapping[str, tuple[str, ...]] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _source_metadata_by_runtime_lookup_key: Mapping[
        str,
        SourceMetadataMapping,
    ] | None = field(
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
        source_binding_context: "SourceBindingRuntimeContext | None",
        source_metadata_is_normalized: bool,
    ) -> None:
        if source_binding_context is not None:
            object.__setattr__(
                self,
                "step_input_files",
                source_binding_context.step_input_files,
            )
            object.__setattr__(
                self,
                "current_step_input_files",
                source_binding_context.current_step_input_files,
            )
            object.__setattr__(
                self,
                "current_image_files",
                source_binding_context.current_image_files,
            )
            object.__setattr__(
                self,
                "step_input_dir",
                source_binding_context.step_input_dir,
            )
            object.__setattr__(
                self,
                "step_input_source_backend",
                source_binding_context.step_input_source_backend,
            )
            object.__setattr__(
                self,
                "step_input_storage_backend",
                source_binding_context.step_input_storage_backend,
            )
            object.__setattr__(
                self,
                "step_input_source_paths",
                source_binding_context.step_input_source_paths,
            )
            object.__setattr__(
                self,
                "source_metadata_by_path",
                source_binding_context.source_metadata_by_path,
            )
            object.__setattr__(
                self,
                "pipeline_input_files",
                source_binding_context.pipeline_input_files,
            )
            object.__setattr__(
                self,
                "pipeline_source_candidate_files",
                source_binding_context.pipeline_source_candidate_files,
            )
            object.__setattr__(
                self,
                "pipeline_input_backend",
                source_binding_context.pipeline_input_backend,
            )
            object.__setattr__(
                self,
                "_source_metadata_identity",
                source_binding_context._source_metadata_identity,
            )
            object.__setattr__(
                self,
                "_pipeline_input_files_identity",
                source_binding_context._pipeline_input_files_identity,
            )
            object.__setattr__(
                self,
                "_source_order_identity",
                source_binding_context._source_order_identity,
            )
            object.__setattr__(
                self,
                "_process_semantic_identity",
                source_binding_context._process_semantic_identity,
            )
            object.__setattr__(
                self,
                "_virtual_source_paths_by_identity",
                source_binding_context._virtual_source_paths_by_identity,
            )
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
                {str(path): str(source) for path, source in step_input_source_paths.items()}
            )
        object.__setattr__(self, "step_input_source_paths", step_input_source_paths)

        if source_binding_context is None:
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
            dict[str, dict[str, str]],
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
                    path: dict(metadata)
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
    return coerce_enum(AllComponents, value, field_name)


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
