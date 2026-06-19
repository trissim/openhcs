"""Shared stream component semantics for image and artifact outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeAlias

from polystore.exceptions import MetadataNotFoundError
from polystore.streaming.identity import StreamProducerIdentity
from polystore.streaming.viewer_transport import (
    ViewerStreamRequest,
    ViewerStreamSource,
    ViewerStreamSourceMetadata,
)
from zmqruntime.viewer_protocol import (
    ViewerComponentMetadataPayload,
    ViewerWireValue,
)

from openhcs.constants.constants import get_multiprocessing_axis
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.runtime_values import SourceComponentMetadata
from openhcs.core.source_matching import source_metadata_value
from openhcs.core.streaming_config_factory import StreamingViewerSurface
from openhcs.runtime.viewer_component_system import (
    ComponentValue,
    ViewerComponentAxisSemantics,
    ViewerComponentAxisSemanticsAuthority,
    ViewerComponentLayout,
    ViewerComponentValueDomainPayload,
    ViewerObjectDisplayConfigInput,
)


StreamComponentMetadata = SourceComponentMetadata | None
ComponentDisplayName: TypeAlias = str | int | float | bool | None
StreamComponentDomainMetadataItems: TypeAlias = tuple[dict[str, ComponentValue], ...]


class StreamComponentNameMetadata(dict[str, dict[str, ComponentDisplayName]]):
    """Component value display names keyed by component then raw value."""


@dataclass(frozen=True)
class StreamComponentMessageExtraPayload(ViewerComponentMetadataPayload):
    """Viewer wire payload declaring stream component domains and display names."""

    @classmethod
    def from_axis_semantics(
        cls,
        *,
        component_axis_semantics: ViewerComponentAxisSemantics,
        component_names_metadata: StreamComponentNameMetadata,
    ) -> "StreamComponentMessageExtraPayload":
        return cls(
            component_names_metadata=component_names_metadata,
            component_value_domain=component_axis_semantics.value_domain.to_wire_mapping(),
        )


@dataclass(frozen=True, slots=True)
class StreamSourceComponentMetadataItems:
    """Source component metadata observed by one viewer streaming operation."""

    values: tuple[StreamComponentMetadata, ...]

    @classmethod
    def from_values(
        cls,
        values: Iterable[StreamComponentMetadata],
    ) -> "StreamSourceComponentMetadataItems":
        return cls(tuple(values))

    def domain_metadata_items(self) -> StreamComponentDomainMetadataItems:
        return tuple(
            dict(metadata)
            for metadata in self.values
            if metadata is not None
        )

    def viewer_source_metadata(self) -> ViewerStreamSourceMetadata:
        return ViewerStreamSourceMetadata(
            component_metadata_by_path=tuple(
                self._required_item_metadata(index, metadata)
                for index, metadata in enumerate(self.values)
            )
        )

    @staticmethod
    def _required_item_metadata(
        index: int,
        metadata: StreamComponentMetadata,
    ) -> dict[str, ComponentValue]:
        if metadata is None:
            raise ValueError(
                "Viewer streaming requires source component metadata for "
                f"item {index}."
            )
        return dict(metadata)

    def include_observed_values(
        self,
        name_map: "StreamComponentNameMap",
        component_order: tuple[str, ...],
    ) -> None:
        for metadata in self.values:
            if metadata is None:
                continue
            for component in component_order:
                value = source_metadata_value(metadata, component)
                if value is None:
                    continue
                name_map.include_observed_value(component, value)


@dataclass(slots=True)
class StreamComponentNameMap:
    """Mutable builder for stream component display-name metadata."""

    values: StreamComponentNameMetadata

    @classmethod
    def empty(cls) -> "StreamComponentNameMap":
        return cls(values=StreamComponentNameMetadata())

    def merge_component_values(
        self,
        component: str,
        values: dict[str, ComponentDisplayName],
    ) -> None:
        if not values:
            return
        if component not in self.values:
            self.values[component] = {}
        self.values[component].update(values)

    def include_observed_value(self, component: str, value: str) -> None:
        if component not in self.values:
            self.values[component] = {}
        if value not in self.values[component]:
            self.values[component][value] = None


@dataclass(frozen=True, slots=True)
class StreamMetadataRootAuthority:
    """Resolve unique metadata roots available to stream-output projection."""

    context: ProcessingContext

    def roots(self) -> tuple[Path, ...]:
        roots = []
        for value in (
            self.context.plate_path,
            self.context.input_dir,
        ):
            if value is None:
                continue
            root = Path(value)
            if root not in roots:
                roots.append(root)
        return tuple(
            root
            for root in roots
            if self._contains_metadata(root)
        )

    def _contains_metadata(self, root: Path) -> bool:
        try:
            self.context.microscope_handler.metadata_handler.find_metadata_file(root)
        except (FileNotFoundError, MetadataNotFoundError, NotADirectoryError):
            return False
        return True


class StreamComponentMessageExtraAuthority:
    """Build viewer wire metadata that declares component domains and display names."""

    @classmethod
    def payload(
        cls,
        source_scope: "OpenHCSViewerStreamSourceScope",
    ) -> StreamComponentMessageExtraPayload:
        display_input = ViewerObjectDisplayConfigInput(source_scope.display_config)
        component_axis_semantics = cls._axis_semantics(
            source_scope,
            display_input,
        )
        return StreamComponentMessageExtraPayload.from_axis_semantics(
            component_axis_semantics=component_axis_semantics,
            component_names_metadata=cls._component_names_metadata(
                source_scope,
                component_axis_semantics.component_order,
            ),
        )

    @classmethod
    def _axis_semantics(
        cls,
        source_scope: "OpenHCSViewerStreamSourceScope",
        display_input: ViewerObjectDisplayConfigInput,
    ) -> ViewerComponentAxisSemantics:
        layout = display_input.layout()
        value_domain = cls._value_domain(source_scope, layout)
        return ViewerComponentAxisSemanticsAuthority.from_display_config(
            display_config=display_input,
            value_domain=value_domain,
        )

    @classmethod
    def _value_domain(
        cls,
        source_scope: "OpenHCSViewerStreamSourceScope",
        layout: ViewerComponentLayout,
    ) -> ViewerComponentValueDomainPayload:
        return ViewerComponentValueDomainPayload.from_component_metadata(
            component_layout=layout,
            metadata_items=cls._metadata_items(source_scope, layout),
        )

    @classmethod
    def _metadata_items(
        cls,
        source_scope: "OpenHCSViewerStreamSourceScope",
        layout: ViewerComponentLayout,
    ) -> StreamComponentDomainMetadataItems:
        return (
            *cls._root_metadata_items(source_scope, layout),
            *source_scope.source_metadata_items.domain_metadata_items(),
        )

    @classmethod
    def _component_names_metadata(
        cls,
        source_scope: "OpenHCSViewerStreamSourceScope",
        component_order: tuple[str, ...],
    ) -> StreamComponentNameMetadata:
        metadata = StreamComponentNameMap.empty()
        cls._include_root_component_names(source_scope, metadata, component_order)
        source_scope.source_metadata_items.include_observed_values(
            metadata,
            component_order,
        )
        return metadata.values

    @classmethod
    def _include_root_component_names(
        cls,
        source_scope: "OpenHCSViewerStreamSourceScope",
        metadata: StreamComponentNameMap,
        component_order: tuple[str, ...],
    ) -> None:
        for root in StreamMetadataRootAuthority(source_scope.context).roots():
            for component in component_order:
                values = cls._component_values(source_scope, root, component)
                metadata.merge_component_values(component, values)

    @staticmethod
    def _component_values(
        source_scope: "OpenHCSViewerStreamSourceScope",
        root: Path,
        component: str,
    ) -> dict[str, ComponentDisplayName]:
        metadata_handler = source_scope.context.microscope_handler.metadata_handler
        values = metadata_handler.get_component_values(
            root,
            component,
        )
        if values is None:
            return {}
        return {
            str(value): name
            for value, name in dict(values).items()
        }

    @classmethod
    def _root_metadata_items(
        cls,
        source_scope: "OpenHCSViewerStreamSourceScope",
        layout: ViewerComponentLayout,
    ) -> StreamComponentDomainMetadataItems:
        items: list[dict[str, ComponentValue]] = []
        for root in StreamMetadataRootAuthority(source_scope.context).roots():
            for component in layout.component_order:
                for value in cls._declared_component_values(
                    source_scope,
                    root,
                    component,
                ):
                    items.append({component: value})
        return tuple(items)

    @classmethod
    def _declared_component_values(
        cls,
        source_scope: "OpenHCSViewerStreamSourceScope",
        root: Path,
        component: str,
    ) -> tuple[ComponentValue, ...]:
        if component == cls._multiprocessing_axis_component():
            return cls._execution_axis_values(source_scope)
        metadata_handler = source_scope.context.microscope_handler.metadata_handler
        values = metadata_handler.get_component_values(
            root,
            component,
        )
        if values is None:
            return ()
        return tuple(str(value) for value in values)

    @staticmethod
    def _multiprocessing_axis_component() -> str:
        return str(get_multiprocessing_axis().value)

    @staticmethod
    def _execution_axis_values(
        source_scope: "OpenHCSViewerStreamSourceScope",
    ) -> tuple[ComponentValue, ...]:
        if source_scope.context.owned_wells is None:
            raise RuntimeError(
                "Streaming component domain requires ProcessingContext.owned_wells "
                "for the multiprocessing axis."
            )
        values = tuple(str(value) for value in source_scope.context.owned_wells)
        if not values:
            raise RuntimeError(
                "Streaming component domain requires at least one owned axis value."
            )
        return values


@dataclass(frozen=True)
class OpenHCSViewerStreamSourceScope(StreamingViewerSurface):
    """OpenHCS source scope needed to construct one viewer stream request."""

    context: ProcessingContext
    producer_identity: StreamProducerIdentity
    source_metadata_items: StreamSourceComponentMetadataItems

    @classmethod
    def from_viewer_surface(
        cls,
        viewer_surface: StreamingViewerSurface,
        *,
        context: ProcessingContext,
        producer_identity: StreamProducerIdentity,
        source_metadata_items: StreamSourceComponentMetadataItems,
    ) -> "OpenHCSViewerStreamSourceScope":
        return cls(
            runtime_config=viewer_surface.runtime_config,
            display_config=viewer_surface.display_config,
            source=viewer_surface.source,
            context=context,
            producer_identity=producer_identity,
            source_metadata_items=source_metadata_items,
        )


class OpenHCSViewerStreamRequestAuthority:
    """Single OpenHCS authority for constructing PolyStore viewer stream requests."""

    @staticmethod
    def from_source_metadata(
        *,
        viewer_surface: StreamingViewerSurface,
        producer_identity: StreamProducerIdentity,
        source_metadata: ViewerStreamSourceMetadata,
        message_extra: dict[str, ViewerWireValue] | None = None,
        images_dir: str | None = None,
    ) -> ViewerStreamRequest:
        return ViewerStreamRequest(
            viewer_transport=viewer_surface.runtime_config.transport_endpoint,
            display_config=viewer_surface.display_config,
            source=ViewerStreamSource(
                identity=viewer_surface.source,
                metadata=source_metadata,
            ),
            producer_identity=producer_identity,
            message_extra=message_extra,
            images_dir=images_dir,
        )

    @staticmethod
    def from_source_scope(
        source_scope: OpenHCSViewerStreamSourceScope,
        *,
        images_dir: str | None = None,
    ) -> ViewerStreamRequest:
        return ViewerStreamRequest(
            viewer_transport=source_scope.runtime_config.transport_endpoint,
            display_config=source_scope.display_config,
            source=ViewerStreamSource(
                identity=source_scope.source,
                metadata=source_scope.source_metadata_items.viewer_source_metadata(),
            ),
            producer_identity=source_scope.producer_identity,
            message_extra=(
                StreamComponentMessageExtraAuthority.payload(
                    source_scope
                ).to_wire_mapping()
            ),
            images_dir=images_dir,
        )
