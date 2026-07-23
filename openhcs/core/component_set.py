"""Nominal component-set operations for dynamic OpenHCS component enums."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum

from openhcs.constants.constants import (
    AllComponents,
    VariableComponents,
    get_default_group_by,
    get_default_variable_components,
)
from openhcs.core.components.validation import convert_enum_by_value


@dataclass(slots=True)
class ComponentSet:
    """Ordered unique set of ``AllComponents`` with enum-aware operations."""

    components: tuple[AllComponents, ...] = ()

    def __post_init__(self) -> None:
        normalized: list[AllComponents] = []
        for component in self.components:
            if not isinstance(component, AllComponents):
                raise TypeError(
                    "ComponentSet components must be AllComponents values, "
                    f"got {type(component).__name__}."
                )
            if component not in normalized:
                normalized.append(component)
        self.components = tuple(normalized)

    @classmethod
    def collect(
        cls,
        *groups: Iterable[AllComponents | None],
    ) -> "ComponentSet":
        return cls(
            tuple(
                component
                for group in groups
                for component in group
                if component is not None
            )
        )

    @classmethod
    def from_enum_values(
        cls,
        values: Iterable[Enum],
    ) -> "ComponentSet":
        return cls(tuple(AllComponents(value.value) for value in values))

    @classmethod
    def coerce(
        cls,
        values: Iterable[AllComponents | Enum | str],
    ) -> "ComponentSet":
        return cls(tuple(cls.coerce_component(value) for value in values))

    @staticmethod
    def coerce_component(value: AllComponents | Enum | str) -> AllComponents:
        if isinstance(value, AllComponents):
            return value
        if isinstance(value, Enum):
            converted = convert_enum_by_value(value, AllComponents)
            if converted is None:
                raise ValueError(
                    f"Component enum {value!r} is not an OpenHCS component."
                )
            return converted
        return AllComponents(value)

    @classmethod
    def default_variable(cls) -> "ComponentSet":
        return cls.from_enum_values(get_default_variable_components())

    @classmethod
    def default_group_by(cls) -> "ComponentSet":
        group_by = get_default_group_by()
        if group_by is None or group_by.value is None:
            return cls()
        return cls.from_enum_values((group_by,))

    def __bool__(self) -> bool:
        return bool(self.components)

    def __contains__(self, component: AllComponents) -> bool:
        return component in self.components

    def __iter__(self) -> Iterator[AllComponents]:
        return iter(self.components)

    def as_tuple(self) -> tuple[AllComponents, ...]:
        return self.components

    def variable(self) -> "ComponentSet":
        return ComponentSet(
            tuple(
                component
                for component in self.components
                if component.name in VariableComponents.__members__
            )
        )

    def excluding(self, *others: "ComponentSet") -> "ComponentSet":
        excluded = frozenset(
            component
            for other in others
            for component in other.components
        )
        return ComponentSet(
            tuple(component for component in self.components if component not in excluded)
        )

    def intersection(self, other: "ComponentSet") -> "ComponentSet":
        return ComponentSet(
            tuple(component for component in self.components if component in other)
        )

    def required_last(self, error_message: str) -> AllComponents:
        if not self.components:
            raise ValueError(error_message)
        return self.components[-1]

    def single_or_none(self, multiple_error_message: str) -> AllComponents | None:
        if not self.components:
            return None
        if len(self.components) == 1:
            return self.components[0]
        raise ValueError(multiple_error_message)

    def last(self) -> AllComponents | None:
        if not self.components:
            return None
        return self.components[-1]
