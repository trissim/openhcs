"""Typed component-group scope shared by compilation and runtime execution."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from openhcs.constants.constants import AllComponents
from openhcs.core.component_set import ComponentSet

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.source_matching import SourceAxisMetadataScope

ComponentGroupKey = str | None

RuntimeComponentValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ComponentGroupScope:
    """One validated ungrouped, static, or runtime-discovered component scope."""

    keys: tuple[ComponentGroupKey, ...]
    component: AllComponents | None = None

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("ComponentGroupScope.keys cannot be empty.")
        if any(key is not None and not isinstance(key, str) for key in self.keys):
            raise TypeError(
                "ComponentGroupScope.keys must contain canonical strings or None."
            )
        if None in self.keys and self.keys != (None,):
            raise ValueError(
                "ComponentGroupScope may use None only as the sole dynamic or "
                "ungrouped key."
            )
        if self.component is not None and not isinstance(
            self.component,
            AllComponents,
        ):
            raise TypeError(
                "ComponentGroupScope.component must be an AllComponents value."
            )
        if self.component is None and self.keys != (None,):
            raise ValueError(
                "Concrete component-group keys require a component identity."
            )

    @classmethod
    def ungrouped(cls) -> "ComponentGroupScope":
        return cls((None,))

    @classmethod
    def dynamic(
        cls,
        component: AllComponents,
    ) -> "ComponentGroupScope":
        """Declare a typed scope whose concrete keys are discovered at runtime."""

        return cls((None,), component=component)

    @classmethod
    def from_raw(
        cls,
        group_keys: Iterable[Hashable | None],
        *,
        component: AllComponents | Enum | str | None = None,
    ) -> "ComponentGroupScope":
        return cls(
            tuple(cls.normalize_key(group_key) for group_key in group_keys),
            component=(
                None if component is None else ComponentSet.coerce_component(component)
            ),
        )

    @staticmethod
    def normalize_key(key: Hashable | None) -> ComponentGroupKey:
        if key is None:
            return None
        return str(key)

    @property
    def is_ungrouped(self) -> bool:
        return self.component is None

    @property
    def is_dynamic(self) -> bool:
        return self.component is not None and self.keys == (None,)

    def runtime_keys(
        self,
        discovered_keys: Iterable[Hashable | None],
    ) -> tuple[ComponentGroupKey, ...]:
        """Resolve concrete execution keys from this compiler-owned scope."""

        if self.is_ungrouped:
            return (None,)
        if not self.is_dynamic:
            return self.keys
        keys = tuple(
            dict.fromkeys(
                self.normalize_key(key) for key in discovered_keys if key is not None
            )
        )
        if not keys:
            raise ValueError(
                f"Dynamic {self.component.value} execution scope requires "
                "component-grouped runtime patterns."
            )
        return keys

    def contains_runtime_key(self, key: Hashable | None) -> bool:
        """Return whether one concrete invocation belongs to this scope."""

        normalized_key = self.normalize_key(key)
        if self.is_ungrouped:
            return normalized_key is None
        if self.is_dynamic:
            return normalized_key is not None
        return normalized_key in self.keys

    def contains_scope(self, required: "ComponentGroupScope") -> bool:
        """Return whether every invocation in ``required`` is available here."""

        if self.is_ungrouped or required.is_ungrouped:
            return self.is_ungrouped and required.is_ungrouped
        if self.component is not required.component:
            return False
        if self.is_dynamic:
            return True
        if required.is_dynamic:
            return False
        return set(required.keys).issubset(self.keys)

    def resolve_runtime_key(
        self,
        runtime_key: Hashable | None,
    ) -> ComponentGroupKey:
        """Resolve one concrete key against this compiler-owned scope."""

        if self.is_ungrouped:
            return None
        normalized_key = self.normalize_key(runtime_key)
        if self.is_dynamic:
            if normalized_key is None:
                raise ValueError(
                    f"Dynamic {self.component.value} component scope requires a "
                    "concrete runtime key."
                )
            return normalized_key
        if normalized_key in self.keys:
            return normalized_key
        raise ValueError(
            f"Component scope {self.component.value!r} groups {self.keys!r} do not "
            f"contain runtime key {normalized_key!r}."
        )

    def select_runtime_key(
        self,
        runtime_key: Hashable | None,
    ) -> ComponentGroupKey:
        """Select one coordinate from this compiled scope for an invocation."""

        if self.is_ungrouped:
            return None
        if not self.is_dynamic and len(self.keys) == 1:
            return self.require_single_static_key()
        return self.resolve_runtime_key(runtime_key)

    def require_single_static_key(self) -> str:
        """Return the sole coordinate of an exact compiler-owned scope."""

        if self.is_ungrouped or self.is_dynamic or len(self.keys) != 1:
            raise ValueError(
                "A single static component key is required, got "
                f"component={self.component!r}, keys={self.keys!r}."
            )
        key = self.keys[0]
        if key is None:
            raise RuntimeError("Static component scopes cannot contain None.")
        return key

    def output_lineage_scope(
        self,
        consumer_scope: "ComponentGroupScope",
        consumer_variable_components: ComponentSet,
    ) -> "ComponentGroupScope":
        """Propagate this source scope through one consumer invocation."""

        if self.is_ungrouped or self.component in consumer_variable_components:
            return consumer_scope
        if consumer_scope.is_ungrouped or self.component is consumer_scope.component:
            return self
        raise ValueError(
            f"Output lineage component {self.component.value!r} is neither the "
            f"consumer group component {consumer_scope.component.value!r} nor one "
            "of its variable components."
        )


@dataclass(frozen=True, slots=True)
class RuntimeExecutionAxisScope:
    """Typed runtime axis coordinate for source-axis projection."""

    axis_id: str
    component: AllComponents | None = None
    value: RuntimeComponentValue | None = None

    @classmethod
    def from_context(
        cls,
        context: "ProcessingContext",
        *,
        component: AllComponents | str | None = None,
        value: RuntimeComponentValue | None = None,
    ) -> "RuntimeExecutionAxisScope":
        axis_id = context.axis_id
        if not axis_id:
            raise RuntimeError(
                "ProcessingContext.axis_id is required for runtime execution."
            )
        return cls.from_raw(
            str(axis_id),
            component=component,
            value=value,
        )

    @classmethod
    def from_raw(
        cls,
        axis_id: str,
        *,
        component: AllComponents | Enum | str | None,
        value: RuntimeComponentValue | None,
    ) -> "RuntimeExecutionAxisScope":
        if not axis_id:
            raise ValueError("RuntimeExecutionAxisScope.axis_id cannot be empty.")
        if component is None and value is not None:
            raise ValueError(
                "RuntimeExecutionAxisScope component value requires a component."
            )
        if component is not None and value is None:
            raise ValueError(
                "RuntimeExecutionAxisScope component requires a component value."
            )
        return cls(
            axis_id=str(axis_id),
            component=ComponentSet.coerce_component(component),
            value=value,
        ) if component is not None else cls(axis_id=str(axis_id))

    def __post_init__(self) -> None:
        if not self.axis_id:
            raise ValueError("RuntimeExecutionAxisScope.axis_id cannot be empty.")
        if self.component is None and self.value is not None:
            raise ValueError(
                "RuntimeExecutionAxisScope component value requires a component."
            )
        if self.component is not None and self.value is None:
            raise ValueError(
                "RuntimeExecutionAxisScope component requires a component value."
            )

    @property
    def component_name(self) -> str | None:
        if self.component is None:
            return None
        return str(self.component.value)

    def require_component_name(self) -> str:
        component_name = self.component_name
        if component_name is None:
            raise ValueError("Runtime component-axis scope has no component.")
        return component_name

    @property
    def value_text(self) -> str | None:
        if self.value is None:
            return None
        return str(self.value)

    def require_value_text(self) -> str:
        value_text = self.value_text
        if value_text is None:
            raise ValueError("Runtime component-axis scope has no value.")
        return value_text

    def value_text_for_component(
        self,
        component: AllComponents | Enum | str | None,
    ) -> str | None:
        """Return this runtime scope's value for one component axis."""
        if component is None:
            return None
        resolved_component = ComponentSet.coerce_component(component)
        if resolved_component.is_multiprocessing_axis():
            return self.axis_id
        if self.component is not None and self.component == resolved_component:
            return self.value_text
        return None

    @property
    def has_value(self) -> bool:
        return self.component is not None and self.value is not None

    @property
    def cache_key(self) -> tuple[str | None, str | None]:
        return (self.component_name, self.value_text)

    def source_axis_metadata_scope(self) -> "SourceAxisMetadataScope":
        """Return metadata constraints for this runtime axis."""
        from openhcs.constants.constants import get_multiprocessing_axis
        from openhcs.core.source_matching import SourceAxisMetadataScope

        component_values: list[tuple[str | None, str]] = [
            (str(get_multiprocessing_axis().value), self.axis_id),
        ]
        component_name = self.component_name
        if component_name is not None:
            component_values.append(
                (
                    component_name,
                    self.require_value_text(),
                )
            )
        return SourceAxisMetadataScope.from_component_values(tuple(component_values))
