"""Nominal value sets keyed by the canonical OpenHCS component declaration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Generic, TypeVar

from openhcs.constants.constants import AllComponents

ComponentValueT = TypeVar("ComponentValueT")


class OpenHCSComponentValues(
    Mapping[AllComponents, ComponentValueT],
    Generic[ComponentValueT],
):
    """Complete immutable values owned by the canonical OpenHCS axes."""

    __slots__ = ("_declared_values",)

    def __init__(
        self,
        component_values: Iterable[tuple[AllComponents, ComponentValueT]],
    ) -> None:
        supplied_values = tuple(component_values)
        if any(
            not isinstance(component, AllComponents) for component, _ in supplied_values
        ):
            raise TypeError("Component values require exact AllComponents members")

        values_by_component = dict(supplied_values)
        if len(values_by_component) != len(supplied_values):
            raise ValueError("Each OpenHCS component may be bound only once")

        missing = tuple(
            component
            for component in AllComponents
            if component not in values_by_component
        )
        if missing:
            raise ValueError(
                "Component values must bind the canonical declaration exactly: "
                + ", ".join(f"missing {component.value}" for component in missing)
            )

        self._declared_values = tuple(
            (component, values_by_component[component]) for component in AllComponents
        )

    @classmethod
    def from_member_projection(
        cls,
        source_values: Iterable[tuple[Enum, ComponentValueT]],
    ) -> "OpenHCSComponentValues[ComponentValueT]":
        """Project another nominal enum by matching its declared member names."""

        projected_by_name: dict[str, ComponentValueT] = {}
        for component, value in source_values:
            if not isinstance(component, Enum):
                raise TypeError("Projected components must be nominal enum members")
            if component.name in projected_by_name:
                raise ValueError(
                    f"Projected component {component.name!r} was bound more than once"
                )
            projected_by_name[component.name] = value

        missing = tuple(
            component.name
            for component in AllComponents
            if component.name not in projected_by_name
        )
        if missing:
            raise ValueError(
                "Component projection lacks declared members: " + ", ".join(missing)
            )
        return cls(
            (component, projected_by_name[component.name])
            for component in AllComponents
        )

    @classmethod
    def from_partial(
        cls,
        component_values: Iterable[tuple[AllComponents, ComponentValueT]],
        *,
        missing_value: ComponentValueT,
    ) -> "OpenHCSComponentValues[ComponentValueT]":
        """Complete a partial nominal binding with one explicit absent value."""

        supplied_values = tuple(component_values)
        if any(
            not isinstance(component, AllComponents) for component, _ in supplied_values
        ):
            raise TypeError("Component values require exact AllComponents members")
        values_by_component = dict(supplied_values)
        if len(values_by_component) != len(supplied_values):
            raise ValueError("Each OpenHCS component may be bound only once")
        return cls(
            (
                component,
                values_by_component.get(component, missing_value),
            )
            for component in AllComponents
        )

    def declared_values(
        self,
    ) -> tuple[tuple[AllComponents, ComponentValueT], ...]:
        """Return values in canonical declaration order."""

        return self._declared_values

    def __getitem__(self, component: AllComponents) -> ComponentValueT:
        return self.value_for(component)

    def __iter__(self) -> Iterator[AllComponents]:
        return (component for component, _ in self._declared_values)

    def __len__(self) -> int:
        return len(self._declared_values)

    def value_for(self, component: AllComponents) -> ComponentValueT:
        """Return the value for one exact canonical component."""

        if not isinstance(component, AllComponents):
            raise TypeError("Component lookup requires an exact AllComponents member")
        for declared_component, value in self._declared_values:
            if declared_component is component:
                return value
        raise KeyError(component)

    def with_value(
        self,
        component: AllComponents,
        value: ComponentValueT,
    ) -> "OpenHCSComponentValues[ComponentValueT]":
        """Return a value set with one canonical component replaced."""

        self.value_for(component)
        return type(self)(
            (
                declared_component,
                value if declared_component is component else current_value,
            )
            for declared_component, current_value in self._declared_values
        )

    def wire_mapping(self) -> Mapping[str, ComponentValueT]:
        """Serialize values at an explicit string-keyed boundary."""

        return MappingProxyType(
            {component.value: value for component, value in self._declared_values}
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpenHCSComponentValues):
            return NotImplemented
        return self._declared_values == other._declared_values
