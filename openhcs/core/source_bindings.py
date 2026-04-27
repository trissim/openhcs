"""Typed source-binding semantics for named step input views."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from openhcs.constants.constants import AllComponents
from openhcs.core.components.validation import convert_enum_by_value
from openhcs.core.runtime_semantics import coerce_enum


class SourceBindingOrigin(Enum):
    """Where a named binding should be resolved from."""

    STEP_INPUT = "step_input"
    PIPELINE_START = "pipeline_start"


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
    inherit_current_scope: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "metadata", tuple(self.metadata))
        for selector in self.components:
            if not isinstance(selector, ComponentSelector):
                raise TypeError(
                    "SourceSelector.components must contain ComponentSelector values, "
                    f"got {type(selector).__name__}."
                )
        for selector in self.metadata:
            if not isinstance(selector, MetadataSelector):
                raise TypeError(
                    "SourceSelector.metadata must contain MetadataSelector values, "
                    f"got {type(selector).__name__}."
                )


@dataclass(frozen=True, slots=True)
class NamedSourceBinding:
    """Semantic alias mapped to a typed selector over step input space."""

    alias: str
    selector: SourceSelector = SourceSelector()
    origin: SourceBindingOrigin = SourceBindingOrigin.STEP_INPUT
    required: bool = True

    def __post_init__(self) -> None:
        _require_name(self.alias, "NamedSourceBinding.alias")
        object.__setattr__(self, "alias", str(self.alias))
        if not isinstance(self.selector, SourceSelector):
            raise TypeError(
                "NamedSourceBinding.selector must be SourceSelector, "
                f"got {type(self.selector).__name__}."
            )
        object.__setattr__(
            self,
            "origin",
            coerce_enum(
                SourceBindingOrigin,
                self.origin,
                "NamedSourceBinding.origin",
            ),
        )


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
class StepSourceBindingsConfig:
    """First-class FunctionStep field for named semantic input bindings."""

    groups: tuple[GroupedSourceBindings, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        seen_group_keys: set[str | None] = set()
        for group in self.groups:
            if not isinstance(group, GroupedSourceBindings):
                raise TypeError(
                    "StepSourceBindingsConfig.groups must contain GroupedSourceBindings values, "
                    f"got {type(group).__name__}."
                )
            if group.group_key in seen_group_keys:
                raise ValueError(
                    f"StepSourceBindingsConfig contains duplicate group key "
                    f"{group.group_key!r}."
                )
            seen_group_keys.add(group.group_key)

    @property
    def is_empty(self) -> bool:
        return not self.groups


@dataclass(frozen=True, slots=True)
class CompiledSourceBindingPlan:
    """Immutable compile-time source binding plan for one step."""

    bindings_by_group: Mapping[str | None, tuple[NamedSourceBinding, ...]]

    @classmethod
    def empty(cls) -> "CompiledSourceBindingPlan":
        return cls(bindings_by_group=MappingProxyType({}))

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
            )
        )

    def __post_init__(self) -> None:
        normalized: dict[str | None, tuple[NamedSourceBinding, ...]] = {}
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

    @property
    def is_empty(self) -> bool:
        return not self.bindings_by_group


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
