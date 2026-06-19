"""Shared component semantics for streaming viewer backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Generic, TypeAlias, TypeVar

from metaclass_registry import AutoRegisterMeta
from zmqruntime.viewer_protocol import (
    ViewerBatchDisplayPayload,
    ViewerComponentMode,
    ViewerDisplayConfigWireField,
    viewer_component_mode_value,
)

from openhcs.runtime.viewer_protocol import ViewerComponentValueOrdering
from openhcs.utils.display_config_factory import ViewerDisplayConfigObject


ComponentValue: TypeAlias = str | int | float | bool | tuple | None
ComponentWireValue: TypeAlias = ComponentValue | Sequence[ComponentValue]
ComponentMap: TypeAlias = dict[str, ComponentValue]
ComponentValues: TypeAlias = dict[str, list[ComponentValue]]
ComponentDomainKey: TypeAlias = str | tuple[str, ...] | tuple[str, tuple[str, ...]]
ComponentModeMap: TypeAlias = dict[str, str]
DisplayModeValue: TypeAlias = str | Enum
DisplayComponentName: TypeAlias = str | Enum
DisplayConfigMappingValue: TypeAlias = Mapping[str, DisplayModeValue] | Sequence[DisplayComponentName]
ComponentMetadataItems: TypeAlias = Sequence[Mapping[str, ComponentValue] | None]
ComponentAxisValues: TypeAlias = Mapping[str, Sequence[ComponentValue]]
DisplayConfigT = TypeVar("DisplayConfigT")
WindowProjectionProviderT = TypeVar("WindowProjectionProviderT")


class ViewerComponentSemanticRole(Enum):
    """Semantic roles that viewer backends may request from component axes."""

    COLOR = "color"


@dataclass(frozen=True, slots=True)
class ViewerComponentRolePolicy:
    """Central policy for mapping conventional component names to viewer roles."""

    color_component_candidates: tuple[str, ...] = ("channel",)
    indexed_component_candidates: frozenset[str] = frozenset(
        {"site", "channel", "z_index", "timepoint"}
    )

    def role_component(
        self,
        *,
        role: ViewerComponentSemanticRole,
        layout: "ViewerComponentLayout",
    ) -> str | None:
        candidates = self._candidates(role)
        for component in layout.component_order:
            if component not in candidates:
                continue
            return component
        return None

    def role_component_for_mode(
        self,
        *,
        role: ViewerComponentSemanticRole,
        layout: "ViewerComponentLayout",
        mode: DisplayModeValue,
    ) -> str | None:
        candidates = self._candidates(role)
        mode_value = viewer_component_mode_value(mode)
        for component in layout.component_order:
            if component not in candidates:
                continue
            if layout.component_modes[component] != mode_value:
                continue
            return component
        return None

    def is_indexed(self, component: str) -> bool:
        return component in self.indexed_component_candidates

    def _candidates(self, role: ViewerComponentSemanticRole) -> tuple[str, ...]:
        if role is ViewerComponentSemanticRole.COLOR:
            return self.color_component_candidates
        raise ValueError(f"No component-role policy for {role!r}.")


@dataclass(frozen=True, slots=True)
class ViewerDisplayConfigInput(ABC, metaclass=AutoRegisterMeta):
    """Nominal input adapter for viewer display configs."""

    __registry_key__ = "DISPLAY_CONFIG_INPUT_KIND"
    __skip_if_no_key__ = True
    DISPLAY_CONFIG_INPUT_KIND: ClassVar[str | None] = None

    @abstractmethod
    def layout(self) -> "ViewerComponentLayout":
        """Return a normalized component layout."""


@dataclass(frozen=True, slots=True)
class ViewerMappingDisplayConfigInput(ViewerDisplayConfigInput):
    """Mapping-backed viewer display config input."""

    DISPLAY_CONFIG_INPUT_KIND = "mapping"
    mapping_display_config: Mapping[str, DisplayConfigMappingValue]

    def layout(self) -> "ViewerComponentLayout":
        return ViewerComponentLayoutMappingParser.layout(self.mapping_display_config)


@dataclass(frozen=True, slots=True)
class ViewerObjectDisplayConfigInput(ViewerDisplayConfigInput):
    """Object-backed viewer display config input."""

    DISPLAY_CONFIG_INPUT_KIND = "object"
    object_display_config: ViewerDisplayConfigObject

    def layout(self) -> "ViewerComponentLayout":
        return ViewerComponentLayout.from_parts(
            component_modes=self.object_display_config.component_modes(),
            component_order=self.object_display_config.COMPONENT_ORDER,
        )


class ViewerComponentAddress(ABC):
    """Minimal address contract for component-indexed stream items."""

    @property
    @abstractmethod
    def components(self) -> ComponentMap:
        """Return component metadata for the addressed payload."""


class ViewerComponentAddressedItem(ABC):
    """Minimal item contract for component-indexed stream payloads."""

    @property
    @abstractmethod
    def address(self) -> ViewerComponentAddress:
        """Return the component-addressed payload identity."""


class ViewerComponentLayoutMappingParser:
    """Validate mapping-style display configs before layout construction."""

    @classmethod
    def layout(
        cls,
        mapping_display_config: Mapping[str, DisplayConfigMappingValue],
    ) -> "ViewerComponentLayout":
        component_modes = cls._required_value(
            mapping_display_config,
            ViewerDisplayConfigWireField.COMPONENT_MODES,
            Mapping,
            "mapping",
        )
        component_order = cls._required_value(
            mapping_display_config,
            ViewerDisplayConfigWireField.COMPONENT_ORDER,
            Sequence,
            "sequence",
        )
        if isinstance(component_order, str):
            raise TypeError("display_config['component_order'] must be a sequence.")
        return ViewerComponentLayout.from_parts(
            component_modes=component_modes,
            component_order=component_order,
        )

    @staticmethod
    def _required_value(
        mapping_display_config: Mapping[str, DisplayConfigMappingValue],
        field_name: ViewerDisplayConfigWireField,
        expected_type: type | tuple[type, ...],
        expected_name: str,
    ) -> DisplayConfigMappingValue:
        value = mapping_display_config[field_name.value]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"display_config[{field_name.value!r}] must be a {expected_name}."
            )
        return value


@dataclass(frozen=True, slots=True)
class ViewerComponentLayout(ViewerBatchDisplayPayload):
    """Normalized component-mode layout shared by viewer receivers."""

    @classmethod
    def from_parts(
        cls,
        *,
        component_modes: Mapping[str, DisplayModeValue],
        component_order: Sequence[DisplayComponentName],
    ) -> "ViewerComponentLayout":
        order = tuple(str(component) for component in component_order)
        modes = {
            component: cls._mode_value(component_modes[component])
            for component in order
        }
        return cls(component_modes=modes, component_order=order)

    @staticmethod
    def _mode_value(mode: DisplayModeValue) -> str:
        if isinstance(mode, Enum):
            return str(mode.value)
        return str(mode)

    def group_window_sources(self, sources):
        from polystore.streaming.receivers.core import group_items_by_component_modes

        return group_items_by_component_modes(
            sources,
            display_layout=self,
        )

    def group_window_payloads(self, payloads: Sequence[Mapping[str, ComponentWireValue]]):
        from polystore.streaming.receivers.core import WindowProjectionSource

        return self.group_window_sources(
            WindowProjectionSource.from_wire_payloads(payloads)
        )

    def group_window_payload_providers(
        self,
        items: Sequence[WindowProjectionProviderT],
    ):
        from polystore.streaming.receivers.core import WindowProjectionSource

        return self.group_window_sources(
            WindowProjectionSource.from_payload_providers(items)
        )

    def component_value_counts(self, payloads: Sequence[Mapping]) -> tuple[tuple[str, int], ...]:
        counts = []
        for component_name in self.component_order:
            values = set()
            for payload in payloads:
                metadata = payload["metadata"]
                if component_name in metadata:
                    values.add(metadata[component_name])
            counts.append((component_name, len(values)))
        return tuple(counts)


@dataclass(slots=True)
class ViewerComponentMetadataNormalizer:
    """Normalize component metadata before viewer coordinate indexing."""

    role_policy: ViewerComponentRolePolicy = field(
        default_factory=ViewerComponentRolePolicy
    )

    def normalize(self, components: ComponentMap) -> ComponentMap:
        return {
            component: self.normalize_value(component, value)
            for component, value in components.items()
        }

    def normalize_value(self, component: str, value: ComponentValue) -> ComponentValue:
        if not self.role_policy.is_indexed(component):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped.lstrip("+-").isdigit():
                return int(stripped)
        return value


@dataclass(frozen=True, slots=True)
class ViewerComponentValueDomainEntry:
    """Declared observed values for one component in a viewer stream batch."""

    component: str
    values: tuple[ComponentValue, ...]

    @classmethod
    def from_values(
        cls,
        component: str,
        values: Sequence[ComponentValue],
    ) -> "ViewerComponentValueDomainEntry":
        unique_values = sorted(set(values), key=ViewerComponentValueOrdering.key)
        return cls(component=component, values=tuple(unique_values))


@dataclass(frozen=True, slots=True)
class ViewerComponentValueDomainPayload:
    """Batch-level component value domain shared by viewer receivers."""

    entries: tuple[ViewerComponentValueDomainEntry, ...]

    @classmethod
    def empty(cls) -> "ViewerComponentValueDomainPayload":
        return cls(entries=())

    @classmethod
    def from_component_metadata(
        cls,
        *,
        component_layout: ViewerComponentLayout,
        metadata_items: Sequence[Mapping[str, ComponentValue] | None],
        normalizer: ViewerComponentMetadataNormalizer | None = None,
    ) -> "ViewerComponentValueDomainPayload":
        if normalizer is None:
            normalizer = ViewerComponentMetadataNormalizer()
        values_by_component: dict[str, list[ComponentValue]] = {
            component: [] for component in component_layout.component_order
        }
        for metadata in metadata_items:
            if metadata is None:
                continue
            normalized_metadata = normalizer.normalize(dict(metadata))
            for component in component_layout.component_order:
                if component in normalized_metadata:
                    values_by_component[component].append(normalized_metadata[component])
        return cls(
            entries=tuple(
                ViewerComponentValueDomainEntry.from_values(component, values)
                for component, values in values_by_component.items()
                if values
            )
        )

    @classmethod
    def from_wire_mapping(
        cls,
        payload: Mapping[str, Sequence[ComponentWireValue]],
        *,
        context: str,
    ) -> "ViewerComponentValueDomainPayload":
        entries = []
        for component, raw_values in payload.items():
            if isinstance(raw_values, str) or not isinstance(raw_values, Sequence):
                raise TypeError(
                    f"{context} component domain for {component!r} must be a sequence."
                )
            entries.append(
                ViewerComponentValueDomainEntry.from_values(
                    str(component),
                    tuple(
                        ViewerComponentValueParser.parse(
                            value,
                            context=f"{context} component {component!r}",
                        )
                        for value in raw_values
                    ),
                )
            )
        return cls(tuple(entries))

    def component_values(self) -> ComponentValues:
        return {
            entry.component: list(entry.values)
            for entry in self.entries
        }

    def required_component_values(self, components: Sequence[str]) -> ComponentValues:
        values = self.component_values()
        missing = tuple(component for component in components if component not in values)
        if missing:
            raise ValueError(
                "Declared component value domain missing required component(s): "
                f"{missing!r}."
            )
        return {component: list(values[component]) for component in components}

    def observed_cardinality(self, component: str) -> int:
        values = self.component_values()
        if component not in values:
            return 0
        return len(values[component])

    def component_value_counts(
        self,
        component_order: Sequence[str],
    ) -> tuple[tuple[str, int], ...]:
        return tuple(
            (component, self.observed_cardinality(component))
            for component in component_order
        )

    def to_wire_mapping(self) -> dict[str, list[ComponentValue]]:
        return {
            entry.component: list(entry.values)
            for entry in self.entries
        }

    def __bool__(self) -> bool:
        return bool(self.entries)


class ViewerComponentValueParser:
    """Parse JSON/wire component values into OpenHCS component values."""

    @staticmethod
    def parse(value: ComponentWireValue, *, context: str) -> ComponentValue:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, tuple):
            return value
        if isinstance(value, Sequence):
            return tuple(value)
        raise TypeError(
            f"{context} component value must be scalar or tuple-like, "
            f"got {type(value).__name__}"
        )


@dataclass(frozen=True, slots=True)
class ViewerComponentValueNameEntry:
    """One display-name mapping for a component value."""

    value_key: str
    display_name: ComponentValue

    @classmethod
    def from_wire(
        cls,
        value_key: ComponentWireValue,
        display_name: ComponentWireValue,
        *,
        context: str,
    ) -> "ViewerComponentValueNameEntry":
        return cls(
            value_key=str(value_key),
            display_name=ViewerComponentValueParser.parse(
                display_name,
                context=context,
            ),
        )


@dataclass(slots=True)
class ViewerComponentValueNameStore:
    """Display names for values of one component."""

    names_by_value: dict[str, ComponentValue] = field(default_factory=dict)

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[ViewerComponentValueNameEntry],
    ) -> "ViewerComponentValueNameStore":
        store = cls()
        store.merge_entries(entries)
        return store

    @classmethod
    def from_mapping(
        cls,
        value_names: Mapping[ComponentWireValue, ComponentWireValue],
        *,
        context: str,
    ) -> "ViewerComponentValueNameStore":
        entries = []
        for value_key, display_name in value_names.items():
            entries.append(
                ViewerComponentValueNameEntry.from_wire(
                    value_key,
                    display_name,
                    context=context,
                )
            )
        return cls.from_entries(entries)

    def merge_entries(self, entries: Sequence[ViewerComponentValueNameEntry]) -> None:
        self.names_by_value.update(
            (entry.value_key, entry.display_name)
            for entry in entries
        )

    def merge_store(self, value_names: "ViewerComponentValueNameStore") -> None:
        self.merge_entries(value_names.entries())

    def entries(self) -> tuple[ViewerComponentValueNameEntry, ...]:
        return tuple(
            ViewerComponentValueNameEntry(value_key, display_name)
            for value_key, display_name in self.names_by_value.items()
        )

    def normalized_values(
        self,
        component: str,
        normalizer: ViewerComponentMetadataNormalizer,
    ) -> tuple[ComponentValue, ...]:
        return tuple(
            normalizer.normalize_value(component, value_key)
            for value_key in self.names_by_value
        )

    def display_name(self, value: ComponentValue) -> ComponentValue:
        return self.names_by_value.get(str(value))

    def to_wire_mapping(self) -> dict[str, ComponentValue]:
        return dict(self.names_by_value)


@dataclass(slots=True)
class ViewerComponentNameMetadataStore:
    """Mutable storage for component-value display names."""

    values: dict[str, ViewerComponentValueNameStore] = field(default_factory=dict)

    def merge_payload(
        self,
        incoming: "ViewerComponentNameMetadataPayload",
        *,
        context: str,
    ) -> None:
        for component, value_names in incoming.stores():
            component_key = str(component)
            if component_key not in self.values:
                self.values[component_key] = ViewerComponentValueNameStore()
            self.values[component_key].merge_store(value_names)

    def merge_store(self, incoming: "ViewerComponentNameMetadataStore") -> None:
        for component, value_names in incoming.values.items():
            if component not in self.values:
                self.values[component] = ViewerComponentValueNameStore()
            self.values[component].merge_store(value_names)

    def clear(self) -> None:
        self.values.clear()

    def normalized_values(
        self,
        component: str,
        normalizer: ViewerComponentMetadataNormalizer,
    ) -> tuple[ComponentValue, ...]:
        if component not in self.values:
            return ()
        return self.values[component].normalized_values(component, normalizer)

    def display_name(
        self,
        component: str,
        value: ComponentValue,
    ) -> ComponentValue:
        if component not in self.values:
            return None
        return self.values[component].display_name(value)

    def __bool__(self) -> bool:
        return bool(self.values)

    def __contains__(self, component: str) -> bool:
        return component in self.values


@dataclass(frozen=True, slots=True)
class ViewerComponentNameMetadataEntry:
    """One component's value-name metadata payload."""

    component: str
    value_names: ViewerComponentValueNameStore

    @classmethod
    def from_mapping(
        cls,
        component: str,
        value_names: Mapping[ComponentWireValue, ComponentWireValue],
        *,
        context: str,
    ) -> "ViewerComponentNameMetadataEntry":
        return cls(
            component=str(component),
            value_names=ViewerComponentValueNameStore.from_mapping(
                value_names,
                context=context,
            ),
        )


@dataclass(frozen=True, slots=True)
class ViewerComponentNameMetadataWirePayload:
    """Raw wire mapping for component-value display names."""

    entries: tuple[ViewerComponentNameMetadataEntry, ...]

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Mapping[ComponentWireValue, ComponentWireValue]],
        *,
        context: str,
    ) -> "ViewerComponentNameMetadataWirePayload":
        entries = []
        for component, value_names in payload.items():
            if not isinstance(value_names, Mapping):
                raise TypeError(
                    f"Component-name metadata entry for {component!r} must be a mapping."
                )
            entries.append(
                ViewerComponentNameMetadataEntry.from_mapping(
                    component,
                    value_names,
                    context=context,
                )
            )
        return cls(tuple(entries))


@dataclass(frozen=True, slots=True)
class ViewerComponentNameMetadataPayload:
    """Validated component-name metadata payload."""

    values: tuple[ViewerComponentNameMetadataEntry, ...]

    @classmethod
    def from_wire_payload(
        cls,
        payload: ViewerComponentNameMetadataWirePayload,
        *,
        context: str,
    ) -> "ViewerComponentNameMetadataPayload":
        return cls(payload.entries)

    def stores(self):
        return (
            (entry.component, entry.value_names)
            for entry in self.values
        )

    def to_wire_mapping(self):
        return {
            entry.component: entry.value_names.to_wire_mapping()
            for entry in self.values
        }


@dataclass(slots=True)
class ViewerComponentNameMetadata:
    """Component-value display names shared by viewer receivers."""

    store: ViewerComponentNameMetadataStore = field(
        default_factory=ViewerComponentNameMetadataStore
    )

    @classmethod
    def empty(cls) -> "ViewerComponentNameMetadata":
        return cls()

    @classmethod
    def from_wire_payload(
        cls,
        payload: ViewerComponentNameMetadataWirePayload,
        *,
        context: str,
    ) -> "ViewerComponentNameMetadata":
        return cls.from_payload(
            ViewerComponentNameMetadataPayload.from_wire_payload(
                payload,
                context=context,
            ),
            context=context,
        )

    @classmethod
    def from_payload(
        cls,
        payload: ViewerComponentNameMetadataPayload,
        *,
        context: str,
    ) -> "ViewerComponentNameMetadata":
        metadata = cls.empty()
        metadata.store.merge_payload(
            payload,
            context=context,
        )
        return metadata

    def merge(self, incoming: "ViewerComponentNameMetadata") -> None:
        self.store.merge_store(incoming.store)

    def clear(self) -> None:
        self.store.clear()

    def to_wire_mapping(self) -> dict[str, dict[str, ComponentValue]]:
        return {
            component: value_names.to_wire_mapping()
            for component, value_names in self.store.values.items()
        }

    def __bool__(self) -> bool:
        return bool(self.store)

    def __contains__(self, component: str) -> bool:
        return component in self.store

    def __iter__(self):
        return iter(self.store.values)


@dataclass(frozen=True, slots=True)
class ViewerComponentAxisSemantics:
    """Shared component-axis layout plus declared value-domain carrier."""

    layout: ViewerComponentLayout
    value_domain: ViewerComponentValueDomainPayload
    role_policy: ViewerComponentRolePolicy = field(
        default_factory=ViewerComponentRolePolicy
    )

    @property
    def component_order(self) -> tuple[str, ...]:
        return self.layout.component_order

    @property
    def component_modes(self) -> ComponentModeMap:
        return self.layout.component_modes

class ViewerComponentAxisSemanticsAuthority:
    """Build component-axis semantics from external config/domain inputs."""

    @staticmethod
    def from_display_config(
        display_config: ViewerDisplayConfigInput,
        value_domain: ViewerComponentValueDomainPayload,
    ) -> ViewerComponentAxisSemantics:
        return ViewerComponentAxisSemantics(
            layout=display_config.layout(),
            value_domain=value_domain,
        )

    @staticmethod
    def from_display_config_and_metadata(
        *,
        display_config: ViewerDisplayConfigInput,
        metadata_items: ComponentMetadataItems,
    ) -> ViewerComponentAxisSemantics:
        layout = display_config.layout()
        return ViewerComponentAxisSemantics(
            layout=layout,
            value_domain=ViewerComponentValueDomainPayload.from_component_metadata(
                component_layout=layout,
                metadata_items=metadata_items,
            ),
        )

    @staticmethod
    def empty() -> ViewerComponentAxisSemantics:
        return ViewerComponentAxisSemantics(
            layout=ViewerComponentLayout.from_parts(
                component_modes={},
                component_order=(),
            ),
            value_domain=ViewerComponentValueDomainPayload.empty(),
        )


@dataclass(frozen=True, slots=True)
class ViewerComponentAxisSemanticsCarrier:
    """Carrier for objects already bound to component-axis semantics."""

    component_axis_semantics: ViewerComponentAxisSemantics


@dataclass(frozen=True, slots=True)
class ViewerDisplayBatchContext(ViewerComponentAxisSemanticsCarrier, Generic[DisplayConfigT]):
    """Shared viewer batch context for display config and component domains."""

    viewer_display_config: DisplayConfigT
    component_names_metadata: ViewerComponentNameMetadata


@dataclass(frozen=True, slots=True)
class ViewerComponentLabelAuthority:
    """Build human-readable labels for component values."""

    component_names_metadata: ViewerComponentNameMetadata
    abbreviations: Mapping[str, str] = field(
        default_factory=lambda: {
            "channel": "Ch",
            "z_index": "Z",
            "timepoint": "T",
            "site": "Site",
            "well": "Well",
        }
    )
    metadata_formatters: Mapping[
        str,
        Callable[[ComponentValue, ComponentValue], str],
    ] = field(
        default_factory=lambda: {
            "channel": lambda value, name: f"Ch{value}: {name}",
            "well": lambda _value, name: str(name),
        }
    )

    @classmethod
    def empty(cls) -> "ViewerComponentLabelAuthority":
        return cls(ViewerComponentNameMetadata.empty())

    def display_name(self, component: str, value: ComponentValue) -> str | None:
        name = self.component_names_metadata.store.display_name(component, value)
        if name is None or str(name).lower() == "none":
            return None
        return str(name)

    def compact_label(self, component: str, value: ComponentValue) -> str:
        name = self.display_name(component, value)
        if name is not None:
            return name
        return f"{self.abbreviation(component)} {value}"

    def abbreviation(self, component: str) -> str:
        if component in self.abbreviations:
            return self.abbreviations[component]
        return component

    def axis_label(self, component: str, value: ComponentValue) -> str:
        name = self.display_name(component, value)
        if name is None:
            return self.compact_label(component, value)

        formatter = self.metadata_formatters.get(component)
        if formatter is not None:
            return formatter(value, name)
        return f"{component.title()} {value}: {name}"

    def axis_labels(
        self,
        component: str,
        values: Sequence[ComponentValue],
    ) -> list[str]:
        return [self.axis_label(component, value) for value in values]

    def compact_tuple_labels(
        self,
        components: Sequence[str],
        values: Sequence[ComponentValue],
        *,
        context: str,
    ) -> list[str]:
        labels = []
        for component_index, component in enumerate(components):
            try:
                value = values[component_index]
            except IndexError as error:
                raise ValueError(
                    f"{context} value {tuple(values)!r} does not include "
                    f"component {component!r}."
                ) from error
            labels.append(self.compact_label(component, value))
        return labels

    def compact_tuple_label(
        self,
        components: Sequence[str],
        values: Sequence[ComponentValue],
        *,
        default: str,
        context: str,
    ) -> str:
        labels = self.compact_tuple_labels(
            components,
            values,
            context=context,
        )
        if labels:
            return " | ".join(labels)
        return default


@dataclass(slots=True)
class ViewerComponentValueDomain:
    """Store observed component values for keyed streaming domains."""

    domain_values: dict[ComponentDomainKey, dict[str, set[ComponentValue]]] = field(
        default_factory=dict
    )

    def update(
        self,
        domain_key: ComponentDomainKey,
        axis_components: Sequence[str],
        layer_items: Sequence[ViewerComponentAddressedItem],
    ) -> None:
        if domain_key not in self.domain_values:
            self.domain_values[domain_key] = {
                component: set() for component in axis_components
            }

        observed_values = self.domain_values[domain_key]
        for item in layer_items:
            components = item.address.components
            for component in axis_components:
                if component in components:
                    observed_values[component].add(components[component])

    def values_for(
        self,
        domain_key: ComponentDomainKey,
        axis_components: Sequence[str],
    ) -> ComponentValues:
        if domain_key not in self.domain_values:
            return {component: [] for component in axis_components}

        return {
            component: sorted(values, key=ViewerComponentValueOrdering.key)
            for component, values in self.domain_values[domain_key].items()
        }


@dataclass(frozen=True, slots=True)
class ViewerComponentValueDomainView:
    """Named view over component values used by layer-axis projection."""

    values: ComponentValues
    domain_name: str

    def required_values(self, component: str) -> list[ComponentValue]:
        if component not in self.values:
            raise ValueError(
                f"{self.domain_name} component domain missing '{component}'."
            )
        component_values = self.values[component]
        if len(component_values) == 0:
            raise ValueError(
                f"{self.domain_name} component domain for '{component}' is empty."
            )
        return component_values

    def has_multiple_values(self, component: str) -> bool:
        if component not in self.values:
            return False
        return len(self.values[component]) > 1


@dataclass(slots=True)
class ViewerRouteComponentValueTracker:
    """Track observed component values for one routed viewer layer."""

    domain: ViewerComponentValueDomain = field(default_factory=ViewerComponentValueDomain)

    def update(
        self,
        route_key: str,
        axis_components: Sequence[str],
        layer_items: Sequence[ViewerComponentAddressedItem],
    ) -> None:
        self.domain.update(
            self.domain_key(route_key, axis_components),
            axis_components,
            layer_items,
        )

    @staticmethod
    def domain_key(
        route_key: str,
        axis_components: Sequence[str],
    ) -> tuple[str, tuple[str, ...]]:
        return (route_key, tuple(axis_components))


@dataclass(slots=True)
class ViewerDisplayAxisDomain:
    """Track shared viewer axis values for one stack-component layout."""

    domain: ViewerComponentValueDomain = field(default_factory=ViewerComponentValueDomain)

    def update(
        self,
        axis_components: Sequence[str],
        layer_items: Sequence[ViewerComponentAddressedItem],
    ) -> None:
        self.domain.update(tuple(axis_components), axis_components, layer_items)

    def values_for(self, axis_components: Sequence[str]) -> ComponentValues:
        return self.domain.values_for(tuple(axis_components), axis_components)


@dataclass(frozen=True, slots=True)
class ViewerLayerAxisProjection:
    """Route-local component axes projected into a shared viewer coordinate domain."""

    projected_axis_components: tuple[str, ...]
    component_values: ComponentValues
    axis_offsets: tuple[int, ...]

    def axis_offset(self, axis_index: int) -> int:
        """Return the viewer-domain offset for a projected axis."""
        if axis_index >= len(self.axis_offsets):
            return 0
        return self.axis_offsets[axis_index]

    def translate(self, payload_axis_labels: tuple[str, ...] = ()) -> tuple[float, ...]:
        return tuple(
            [
                *(float(offset) for offset in self.axis_offsets),
                *(0.0 for _ in payload_axis_labels),
                0.0,
                0.0,
            ]
        )


@dataclass(frozen=True, slots=True)
class ViewerLayerAxisProjectedComponent:
    """One component axis after route-local projection into viewer coordinates."""

    component: str
    values: list[ComponentValue]
    axis_offset: int


@dataclass(frozen=True, slots=True)
class ViewerLayerAxisProjectionStep:
    """Projection decision for one component axis."""

    component: str
    request: "ViewerLayerAxisProjectionRequest"

    def projected_axis(self) -> ViewerLayerAxisProjectedComponent | None:
        coordinate_values = self.coordinate_domain_values()
        if self.collapses_to_coordinate_singleton(coordinate_values):
            return None
        values, offset = self.project_component_values(coordinate_values)
        return ViewerLayerAxisProjectedComponent(
            component=self.component,
            values=values,
            axis_offset=offset,
        )

    def collapses_to_coordinate_singleton(
        self,
        coordinate_values: Sequence[ComponentValue],
    ) -> bool:
        return len(coordinate_values) == 1

    def coordinate_domain_values(self) -> list[ComponentValue]:
        declared_values = self.request.declared_domain.required_values(self.component)
        if len(declared_values) > 1:
            return declared_values

        viewer_values = self.request.viewer_domain.required_values(self.component)
        if self.viewer_domain_carries_route(viewer_values):
            return viewer_values

        return declared_values

    def viewer_domain_carries_route(
        self,
        viewer_values: Sequence[ComponentValue],
    ) -> bool:
        if len(viewer_values) <= 1:
            return False
        route_values = self.request.route_domain.required_values(self.component)
        return all(value in viewer_values for value in route_values)

    def project_component_values(
        self,
        coordinate_values: Sequence[ComponentValue],
    ) -> tuple[list[ComponentValue], int]:
        route_values = self.request.route_domain.required_values(self.component)
        start_index = self.viewer_index(route_values[0], coordinate_values)
        if self.is_contiguous_subset(start_index, route_values, coordinate_values):
            return route_values, start_index
        # Non-contiguous routes cannot be represented by one translated dense block.
        # Use the full coordinate domain so each value keeps its declared index.
        return list(coordinate_values), 0

    def viewer_index(
        self,
        value: ComponentValue,
        viewer_values: Sequence[ComponentValue],
    ) -> int:
        try:
            return list(viewer_values).index(value)
        except ValueError as error:
            raise ValueError(
                f"Route component value {value!r} for '{self.component}' is absent "
                f"from viewer domain {list(viewer_values)!r}."
            ) from error

    def is_contiguous_subset(
        self,
        start_index: int,
        route_values: Sequence[ComponentValue],
        viewer_values: Sequence[ComponentValue],
    ) -> bool:
        stop_index = start_index + len(route_values)
        return list(viewer_values)[start_index:stop_index] == list(route_values)


@dataclass(frozen=True, slots=True)
class ViewerLayerAxisProjectionRequest:
    """Typed domain bundle for projecting layer-local axes into viewer axes."""

    requested_components: tuple[str, ...]
    route_domain: ViewerComponentValueDomainView
    viewer_domain: ViewerComponentValueDomainView
    declared_domain: ViewerComponentValueDomainView

    @classmethod
    def from_component_values(
        cls,
        *,
        projected_axis_components: Sequence[str],
        route_component_values: ComponentValues,
        viewer_component_values: ComponentValues,
        declared_component_values: ComponentValues,
    ) -> "ViewerLayerAxisProjectionRequest":
        return cls(
            requested_components=tuple(projected_axis_components),
            route_domain=ViewerComponentValueDomainView(
                route_component_values,
                "route",
            ),
            viewer_domain=ViewerComponentValueDomainView(
                viewer_component_values,
                "viewer",
            ),
            declared_domain=ViewerComponentValueDomainView(
                declared_component_values,
                "declared",
            ),
        )

    def projection_steps(self) -> tuple[ViewerLayerAxisProjectionStep, ...]:
        return tuple(
            ViewerLayerAxisProjectionStep(
                component=component,
                request=self,
            )
            for component in self.requested_components
        )


class ViewerLayerAxisProjectionRequestAuthority:
    """Build viewer-axis projection requests from component-axis semantics."""

    @staticmethod
    def from_component_axis_semantics(
        *,
        route_key: str,
        component_axis_semantics: ViewerComponentAxisSemantics,
        layer_items: Sequence[ViewerComponentAddressedItem],
        route_value_tracker: ViewerRouteComponentValueTracker,
        display_axis_domain: ViewerDisplayAxisDomain,
    ) -> ViewerLayerAxisProjectionRequest:
        axis_components = component_axis_semantics.layout.components_for_mode(
            ViewerComponentMode.STACK
        )
        route_value_tracker.update(route_key, axis_components, layer_items)
        display_axis_domain.update(axis_components, layer_items)
        return ViewerLayerAxisProjectionRequest.from_component_values(
            projected_axis_components=axis_components,
            route_component_values=route_value_tracker.domain.values_for(
                route_value_tracker.domain_key(route_key, axis_components),
                axis_components,
            ),
            viewer_component_values=display_axis_domain.values_for(axis_components),
            declared_component_values=(
                component_axis_semantics.value_domain.required_component_values(
                    axis_components
                )
            ),
        )


class ViewerLayerAxisProjector:
    """Project route component values into the current viewer axis domain."""

    def project(
        self,
        request: ViewerLayerAxisProjectionRequest,
    ) -> ViewerLayerAxisProjection:
        projected_axes = tuple(
            projected_axis
            for step in request.projection_steps()
            if (projected_axis := step.projected_axis()) is not None
        )

        return ViewerLayerAxisProjection(
            projected_axis_components=tuple(axis.component for axis in projected_axes),
            component_values={
                axis.component: axis.values for axis in projected_axes
            },
            axis_offsets=tuple(axis.axis_offset for axis in projected_axes),
        )


class ViewerComponentCoordinateAuthority:
    """Fail-loud component coordinate lookup shared by viewer backends."""

    @staticmethod
    def required_value(
        components: Mapping[str, ComponentValue],
        component: str,
        *,
        context: str,
    ) -> ComponentValue:
        if component not in components:
            raise ValueError(f"{context} missing stack component {component!r}.")
        return components[component]

    @staticmethod
    def required_axis_values(
        component_values: ComponentAxisValues,
        component: str,
        *,
        context: str,
    ) -> Sequence[ComponentValue]:
        if component not in component_values:
            raise ValueError(f"{context} missing axis domain for {component!r}.")
        values = component_values[component]
        if not values:
            raise ValueError(f"{context} axis domain for {component!r} is empty.")
        return values

    @classmethod
    def index(
        cls,
        *,
        components: Mapping[str, ComponentValue],
        component_values: ComponentAxisValues,
        component: str,
        context: str,
    ) -> int:
        value = cls.required_value(
            components,
            component,
            context=context,
        )
        values = cls.required_axis_values(
            component_values,
            component,
            context=context,
        )
        try:
            return list(values).index(value)
        except ValueError as error:
            raise ValueError(
                f"{context} component {component!r} value {value!r} is outside "
                f"axis domain {list(values)!r}."
            ) from error

    @classmethod
    def value_tuple(
        cls,
        components: Mapping[str, ComponentValue],
        axis_components: Sequence[str],
        *,
        context: str,
    ) -> tuple[ComponentValue, ...]:
        return tuple(
            cls.required_value(
                components,
                component,
                context=context,
            )
            for component in axis_components
        )


class ViewerDimensionValueAuthority:
    """Collect and index tuple-valued dimensions for viewer coordinate systems."""

    @classmethod
    def collect_from_payloads(
        cls,
        items: Sequence[Mapping[str, ComponentValue | Mapping[str, ComponentValue]]],
        components: Sequence[str],
    ) -> list[tuple]:
        if not components:
            return [()]

        values = {
            cls.value_tuple(cls.metadata(item), components)
            for item in items
        }
        return sorted(values, key=ViewerComponentValueOrdering.tuple_key)

    @staticmethod
    def merge(stored_values: Sequence[tuple], new_values: Sequence[tuple]) -> list[tuple]:
        return sorted(
            set(stored_values) | set(new_values),
            key=ViewerComponentValueOrdering.tuple_key,
        )

    @staticmethod
    def metadata(
        item: Mapping[str, ComponentValue | Mapping[str, ComponentValue]]
    ) -> Mapping[str, ComponentValue]:
        metadata = item["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("Viewer payload metadata must be a mapping.")
        return metadata

    @staticmethod
    def value_tuple(
        metadata: Mapping[str, ComponentValue],
        components: Sequence[str],
    ) -> tuple:
        return ViewerComponentCoordinateAuthority.value_tuple(
            metadata,
            components,
            context="Viewer dimension metadata",
        )

    @classmethod
    def index(
        cls,
        metadata: Mapping[str, ComponentValue],
        components: Sequence[str],
        dimension_values: Sequence[tuple],
    ) -> int:
        key = cls.value_tuple(metadata, components)
        try:
            return list(dimension_values).index(key)
        except ValueError as error:
            raise ValueError(
                "Viewer dimension metadata tuple "
                f"{key!r} is outside axis domain {list(dimension_values)!r}."
            ) from error
