"""Typed component-group scope shared by compilation and runtime execution."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from openhcs.constants.constants import AllComponents
from openhcs.core.component_set import ComponentSet
from openhcs.core.source_metadata import (
    SourceMetadataMapping,
    SourceMetadataScalar,
    SourceMetadataValue,
)

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.source_matching import SourceAxisMetadataScope

ComponentGroupKey = str | None
RuntimeFixedComponentValues = tuple[tuple[AllComponents, str], ...]


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
    """Typed runtime axis coordinate for component projection."""

    axis_id: str
    component: AllComponents | None = None
    value: SourceMetadataScalar = None
    fixed_component_values: RuntimeFixedComponentValues = ()

    @classmethod
    def from_context(
        cls,
        context: "ProcessingContext",
        *,
        component: AllComponents | str | None = None,
        value: SourceMetadataScalar = None,
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
        value: SourceMetadataScalar,
        fixed_component_values: Iterable[
            tuple[AllComponents | Enum | str, SourceMetadataScalar]
        ] = (),
    ) -> "RuntimeExecutionAxisScope":
        if not axis_id:
            raise ValueError("RuntimeExecutionAxisScope.axis_id cannot be empty.")
        resolved_component = (
            None if component is None else ComponentSet.coerce_component(component)
        )
        canonical_value = None if value is None else str(value)
        canonical_fixed_values = cls._canonical_fixed_component_values(
            fixed_component_values,
            group_component=resolved_component,
        )
        return cls(
            axis_id=str(axis_id),
            component=resolved_component,
            value=canonical_value,
            fixed_component_values=canonical_fixed_values,
        )

    def __post_init__(self) -> None:
        if not self.axis_id:
            raise ValueError("RuntimeExecutionAxisScope.axis_id cannot be empty.")
        if self.component is not None and not isinstance(self.component, AllComponents):
            raise TypeError(
                "RuntimeExecutionAxisScope.component must be an AllComponents value. "
                "Use RuntimeExecutionAxisScope.from_raw() for coercion."
            )
        if (self.component is None) != (self.value is None):
            raise ValueError(
                "RuntimeExecutionAxisScope component and value must be declared "
                "together."
            )
        if self.value is not None and not isinstance(self.value, str):
            raise TypeError(
                "RuntimeExecutionAxisScope.value must be canonical text. Use "
                "RuntimeExecutionAxisScope.from_raw() for coercion."
            )
        canonical_fixed_values = self._canonical_fixed_component_values(
            self.fixed_component_values,
            group_component=self.component,
        )
        if self.fixed_component_values != canonical_fixed_values:
            raise ValueError(
                "RuntimeExecutionAxisScope.fixed_component_values must already be "
                "canonical. Use RuntimeExecutionAxisScope.from_raw() to construct "
                "from raw coordinates."
            )

    @staticmethod
    def _canonical_fixed_component_values(
        fixed_component_values: Iterable[
            tuple[AllComponents | Enum | str, SourceMetadataScalar]
        ],
        *,
        group_component: AllComponents | None,
    ) -> RuntimeFixedComponentValues:
        """Validate and order fixed coordinates before scope construction."""

        normalized_fixed_values: dict[AllComponents, str] = {}
        for component, value in fixed_component_values:
            resolved_component = ComponentSet.coerce_component(component)
            if resolved_component.is_multiprocessing_axis():
                raise ValueError(
                    "RuntimeExecutionAxisScope fixed components cannot repeat the "
                    "multiprocessing axis."
                )
            if value is None:
                raise ValueError(
                    "RuntimeExecutionAxisScope fixed component values cannot be None."
                )
            if resolved_component in normalized_fixed_values:
                raise ValueError(
                    "RuntimeExecutionAxisScope fixed components cannot contain "
                    f"duplicate {resolved_component.value!r} identity."
                )
            normalized_fixed_values[resolved_component] = str(value)
        if group_component in normalized_fixed_values:
            raise ValueError(
                "RuntimeExecutionAxisScope group component cannot also be declared "
                "as a fixed component."
            )
        return tuple(
            (component, normalized_fixed_values[component])
            for component in AllComponents
            if component in normalized_fixed_values
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
        if self.component == resolved_component:
            return self.value_text
        for fixed_component, fixed_value in self.fixed_component_values:
            if fixed_component is resolved_component:
                return fixed_value
        return None

    @property
    def has_fixed_components(self) -> bool:
        return bool(self.fixed_component_values)

    @property
    def source_component_values(self) -> RuntimeFixedComponentValues:
        """Return every typed source coordinate represented by this scope."""

        values: list[tuple[AllComponents, str]] = []
        for component in AllComponents:
            value = self.value_text_for_component(component)
            if value is not None:
                values.append((component, value))
        return tuple(values)

    @property
    def presentation_component_values(self) -> RuntimeFixedComponentValues:
        """Return coordinates in a user-facing axis-first order."""

        values = self.source_component_values
        return (
            *(item for item in values if item[0].is_multiprocessing_axis()),
            *(item for item in values if not item[0].is_multiprocessing_axis()),
        )

    @property
    def coordinate_label(self) -> str:
        """Return a stable user-facing label without enum representations."""

        return " / ".join(
            f"{component.value}={value}"
            for component, value in self.presentation_component_values
        )

    def for_group_coordinate(
        self,
        component: AllComponents | Enum | str | None,
        value: SourceMetadataScalar,
    ) -> "RuntimeExecutionAxisScope":
        """Project this execution identity onto one artifact group coordinate."""

        resolved_component = (
            None if component is None else ComponentSet.coerce_component(component)
        )
        if (resolved_component is None) != (value is None):
            raise ValueError(
                "Artifact group component and value must be declared together."
            )
        resolved_value = None if value is None else str(value)
        fixed_values = dict(self.fixed_component_values)
        if self.component is not None and self.component is not resolved_component:
            existing = fixed_values.get(self.component)
            if existing is not None and existing != self.require_value_text():
                raise ValueError(
                    "Runtime execution group conflicts with its fixed component "
                    f"identity for {self.component.value!r}."
                )
            fixed_values[self.component] = self.require_value_text()
        if resolved_component is not None:
            existing = fixed_values.pop(resolved_component, None)
            if existing is not None and existing != resolved_value:
                raise ValueError(
                    "Artifact group coordinate conflicts with fixed runtime identity "
                    f"for {resolved_component.value!r}: {existing!r} != "
                    f"{resolved_value!r}."
                )
            if self.component is resolved_component and self.value_text != resolved_value:
                raise ValueError(
                    "Artifact group coordinate conflicts with runtime execution group "
                    f"for {resolved_component.value!r}: {self.value_text!r} != "
                    f"{resolved_value!r}."
                )
        return type(self).from_raw(
            self.axis_id,
            component=resolved_component,
            value=resolved_value,
            fixed_component_values=tuple(fixed_values.items()),
        )

    def fixed_component_metadata(
        self,
        metadata: SourceMetadataMapping | None = None,
    ) -> dict[str, SourceMetadataValue]:
        """Merge fixed execution coordinates into source component metadata."""

        from openhcs.constants.constants import get_multiprocessing_axis
        from openhcs.core.source_matching import (
            source_component_metadata_value,
            with_source_component_metadata,
        )

        merged: dict[str, SourceMetadataValue] = dict(metadata or {})
        fixed_values = (
            (get_multiprocessing_axis(), self.axis_id),
            *self.fixed_component_values,
        )
        for component, value in fixed_values:
            existing = source_component_metadata_value(merged, component)
            if existing is not None and str(existing) != value:
                raise ValueError(
                    "Fixed execution scope conflicts with source component metadata "
                    f"for {component.value!r}: {existing!r} != {value!r}."
                )
            merged = with_source_component_metadata(merged, component, value)
        return merged

    @property
    def has_value(self) -> bool:
        return self.component is not None

    @property
    def group_scope(self) -> ComponentGroupScope:
        """Return this runtime coordinate as one concrete component-group scope."""
        if self.component is None:
            return ComponentGroupScope.ungrouped()
        return ComponentGroupScope(
            (self.require_value_text(),),
            component=self.component,
        )

    @property
    def cache_key(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (component.value, value)
            for component, value in self.source_component_values
            if not component.is_multiprocessing_axis()
        )

    def source_axis_metadata_scope(self) -> "SourceAxisMetadataScope":
        """Return metadata constraints for this runtime axis."""

        from openhcs.constants.constants import get_multiprocessing_axis
        from openhcs.core.source_matching import SourceAxisMetadataScope

        component_values = tuple(
            (component.value, value)
            for component, value in self.source_component_values
        )
        multiprocessing_component = get_multiprocessing_axis()
        if not any(
            component_name == multiprocessing_component.value
            for component_name, _value in component_values
        ):
            raise RuntimeError(
                "Runtime execution source scope is missing its multiprocessing axis."
            )
        return SourceAxisMetadataScope.from_component_values(component_values)

    def matching_component_plane_indices(
        self,
        component_metadata: Sequence[Mapping[str, object] | None],
    ) -> tuple[int, ...] | None:
        """Select payload planes owned by this scope's typed component value.

        ``None`` means the payload does not declare this component axis. A tuple
        is an exact selection, including the complete plane sequence when every
        plane belongs to the runtime scope.
        """

        if not self.has_value or not component_metadata:
            return None

        from openhcs.core.source_matching import (
            SourceAxisMetadataScope,
            semantic_source_metadata_value,
        )

        component_name = self.require_component_name()
        values = tuple(
            (
                None
                if metadata is None
                else semantic_source_metadata_value(metadata, component_name)
            )
            for metadata in component_metadata
        )
        present_values = tuple(value for value in values if value is not None)
        if not present_values:
            return None
        if len(present_values) != len(values):
            raise RuntimeError(
                "Runtime payload plane metadata only partially declares component "
                f"{component_name!r}: {values!r}."
            )

        component_scope = SourceAxisMetadataScope.from_component_values(
            ((component_name, self.require_value_text()),)
        )
        matching_indices = component_scope.matching_indices(component_metadata)
        if not matching_indices:
            raise RuntimeError(
                "Runtime payload plane metadata has no plane for execution scope "
                f"{component_name}={self.require_value_text()!r}: {values!r}."
            )
        return matching_indices
