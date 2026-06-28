"""Shared stream component semantics for image and artifact outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable, TypeAlias

from metaclass_registry import AutoRegisterMeta
from polystore.exceptions import MetadataNotFoundError
from polystore.streaming.viewer_transport import (
    IndexedViewerStreamSourceMetadata,
    PathMappedViewerStreamSourceMetadata,
    ViewerStreamBackendKwargs,
    ViewerStreamMessageContext,
    ViewerStreamProducer,
    ViewerStreamSourceIdentity,
    ViewerStreamSourceMetadata,
)
from zmqruntime.viewer_protocol import (
    ViewerComponentMetadataPayload,
)

from openhcs.constants.constants import AllComponents, get_multiprocessing_axis
from openhcs.core.context.processing_context import ProcessingContext

from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
    SourceImageIdentity,
)
from openhcs.core.source_matching import (
    source_component_metadata_raw_value,
    source_metadata_value,
)
from openhcs.core.streaming_config_factory import (
    StreamingViewerSurface,
)
from openhcs.runtime.viewer_component_system import (
    ComponentValue,
    ViewerComponentMetadataNormalizer,
    ViewerComponentValueParser,
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
            component_value_domain=component_axis_semantics.to_wire_mapping(),
        )


@dataclass(frozen=True, slots=True)
class StreamViewerComponentMetadataProjector:
    """Project source metadata onto the viewer's declared component axes."""

    component_order: tuple[str, ...]

    def project_required(
        self,
        *,
        index: int,
        metadata: StreamComponentMetadata,
    ) -> dict[str, ComponentValue]:
        if metadata is None:
            raise ValueError(
                "Viewer streaming requires source component metadata for "
                f"item {index}."
            )
        projected = self.project(metadata)
        missing = tuple(
            component
            for component in self.component_order
            if component not in projected
        )
        if missing:
            raise ValueError(
                "Viewer streaming requires complete source component metadata "
                f"for item {index}; missing {missing!r}. "
                f"metadata={metadata!r}."
            )
        return projected

    def project(
        self,
        metadata: SourceComponentMetadata,
    ) -> dict[str, ComponentValue]:
        projected: dict[str, ComponentValue] = {}
        for component in self.component_order:
            value = self.component_value(metadata, component)
            if value is not None:
                projected[component] = value
        return ViewerComponentMetadataNormalizer().normalize(projected)

    def component_value(
        self,
        metadata: SourceComponentMetadata,
        component: str,
    ) -> ComponentValue:
        component_identity = AllComponents.from_value(component)
        if component_identity is None:
            value = metadata.get(component)
        else:
            value = source_component_metadata_raw_value(metadata, component_identity)
        if value is None:
            return None
        return ViewerComponentValueParser.parse(
            value,
            context=f"Viewer source metadata component {component!r}",
        )

    def indexed_source_metadata(
        self,
        values: tuple[StreamComponentMetadata, ...],
    ) -> ViewerStreamSourceMetadata:
        return IndexedViewerStreamSourceMetadata(
            metadata_by_index=tuple(
                self.project_required(index=index, metadata=metadata)
                for index, metadata in enumerate(values)
            )
        )

    def path_mapped_source_metadata(
        self,
        metadata_by_path: Mapping[str, StreamComponentMetadata],
    ) -> ViewerStreamSourceMetadata:
        return PathMappedViewerStreamSourceMetadata(
            metadata_by_path={
                path: self.project_required(index=index, metadata=metadata)
                for index, (path, metadata) in enumerate(metadata_by_path.items())
            }
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

    @classmethod
    def from_source_identities(
        cls,
        identities: Iterable[SourceImageIdentity],
        *,
        fallback_source_identity: SourceImageIdentity | None = None,
    ) -> "StreamSourceComponentMetadataItems":
        return cls.from_values(
            (
                identity.with_missing_from(fallback_source_identity)
                if fallback_source_identity is not None
                else identity
            ).component_metadata
            for identity in identities
        )

    def domain_metadata_items(self) -> StreamComponentDomainMetadataItems:
        return tuple(
            dict(metadata)
            for metadata in self.values
            if metadata is not None
        )

    def viewer_source_metadata(
        self,
        component_order: tuple[str, ...],
    ) -> ViewerStreamSourceMetadata:
        return StreamViewerComponentMetadataProjector(
            component_order
        ).indexed_source_metadata(self.values)

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

class StreamComponentNameMap(StreamComponentNameMetadata):
    """Mutable builder for stream component display-name metadata."""

    def merge_component_values(
        self,
        component: str,
        values: dict[str, ComponentDisplayName],
    ) -> None:
        if not values:
            return
        if component not in self:
            self[component] = {}
        self[component].update(values)

    def include_observed_value(self, component: str, value: str) -> None:
        if component not in self:
            self[component] = {}
        if value not in self[component]:
            self[component][value] = None

@dataclass(frozen=True, slots=True)
class StreamMetadataRoot:
    """Metadata root that has a metadata document discoverable by the handler."""

    source: ViewerStreamSourceIdentity
    root: Path

    @classmethod
    def from_candidate(
        cls,
        source: ViewerStreamSourceIdentity,
        root: Path,
    ) -> "StreamMetadataRoot":
        source.microscope_handler.metadata_handler.find_metadata_file(root)
        return cls(source=source, root=root)

    def component_display_names(
        self,
        component: str,
    ) -> dict[str, ComponentDisplayName]:
        values = self.source.microscope_handler.metadata_handler.get_component_values(
            self.root,
            component,
        )
        if values is None:
            return {}
        return {str(value): name for value, name in dict(values).items()}

    def component_values(self, component: str) -> tuple[ComponentValue, ...]:
        values = self.source.microscope_handler.metadata_handler.get_component_values(
            self.root,
            component,
        )
        if values is None:
            return ()
        return tuple(str(value) for value in values)

@dataclass(frozen=True, slots=True)
class StreamMetadataRootAuthority:
    """Resolve unique metadata roots available to stream-output projection."""

    source: ViewerStreamSourceIdentity
    candidate_roots: tuple[Path, ...]

    @classmethod
    def from_context(cls, context: ProcessingContext) -> "StreamMetadataRootAuthority":
        roots = []
        for value in (
            context.plate_path,
            context.input_dir,
        ):
            if value is None:
                continue
            root = Path(value)
            if root not in roots:
                roots.append(root)
        return cls(
            source=ViewerStreamSourceIdentity(
                microscope_handler=context.microscope_handler,
                plate_path=context.plate_path,
            ),
            candidate_roots=tuple(roots),
        )

    @classmethod
    def from_source_identity(
        cls,
        source: ViewerStreamSourceIdentity,
    ) -> "StreamMetadataRootAuthority":
        roots = ()
        if source.plate_path is not None:
            roots = (Path(source.plate_path),)
        return cls(
            source=source,
            candidate_roots=roots,
        )

    def roots(self) -> tuple[StreamMetadataRoot, ...]:
        metadata_roots = []
        for root in self.candidate_roots:
            try:
                metadata_roots.append(
                    StreamMetadataRoot.from_candidate(
                        self.source,
                        root,
                    )
                )
            except (FileNotFoundError, MetadataNotFoundError, NotADirectoryError):
                continue
        return tuple(metadata_roots)

@dataclass(frozen=True, slots=True)
class StreamComponentDomainProvider(ABC, metaclass=AutoRegisterMeta):
    """Nominal provider for declared stream component values and display names."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: ClassVar[str | None] = None

    component: str
    metadata_roots: tuple[StreamMetadataRoot, ...]

    @classmethod
    def for_component(
        cls,
        *,
        context: ProcessingContext,
        component: str,
        metadata_roots: tuple[StreamMetadataRoot, ...],
    ) -> "StreamComponentDomainProvider":
        for provider_type in cls.__registry__.values():
            if provider_type.supports(component):
                return provider_type.build_for_component(
                    context=context,
                    component=component,
                    metadata_roots=metadata_roots,
                )
        raise LookupError(
            f"No stream component domain provider registered for {component!r}."
        )

    @classmethod
    def build_for_component(
        cls,
        *,
        context: ProcessingContext,
        component: str,
        metadata_roots: tuple[StreamMetadataRoot, ...],
    ) -> "StreamComponentDomainProvider":
        del context
        return cls(
            component=component,
            metadata_roots=metadata_roots,
        )

    @classmethod
    @abstractmethod
    def supports(cls, component: str) -> bool:
        """Return whether this provider owns the component's declared domain."""

    @abstractmethod
    def domain_metadata_items(self) -> StreamComponentDomainMetadataItems:
        """Return metadata entries declaring this component's value domain."""

    def include_display_names(self, name_map: StreamComponentNameMap) -> None:
        for metadata_root in self.metadata_roots:
            name_map.merge_component_values(
                self.component,
                metadata_root.component_display_names(self.component),
            )

@dataclass(frozen=True, slots=True)
class StreamExecutionAxisDomainProvider(StreamComponentDomainProvider):
    """Declared domain provider for the execution multiprocessing axis."""

    registry_key: ClassVar[str] = "execution_axis"
    axis_component: ClassVar[str] = str(get_multiprocessing_axis().value)
    owned_axis_values: tuple[ComponentValue, ...] = ()

    @classmethod
    def build_for_component(
        cls,
        *,
        context: ProcessingContext,
        component: str,
        metadata_roots: tuple[StreamMetadataRoot, ...],
    ) -> "StreamExecutionAxisDomainProvider":
        if context.owned_wells is None:
            raise RuntimeError(
                "Streaming component domain requires ProcessingContext.owned_wells "
                "for the multiprocessing axis."
            )
        values = tuple(str(value) for value in context.owned_wells)
        if not values:
            raise RuntimeError(
                "Streaming component domain requires at least one owned axis value."
            )
        return cls(
            component=component,
            metadata_roots=metadata_roots,
            owned_axis_values=values,
        )

    @classmethod
    def supports(cls, component: str) -> bool:
        return component == cls.axis_component

    def domain_metadata_items(self) -> StreamComponentDomainMetadataItems:
        return tuple({self.component: value} for value in self.owned_axis_values)

@dataclass(frozen=True, slots=True)
class StreamMetadataBackedComponentDomainProvider(StreamComponentDomainProvider):
    """Declared domain provider for components described by plate metadata roots."""

    registry_key: ClassVar[str] = "metadata"

    @classmethod
    def supports(cls, component: str) -> bool:
        return True

    def domain_metadata_items(self) -> StreamComponentDomainMetadataItems:
        items: list[dict[str, ComponentValue]] = []
        for metadata_root in self.metadata_roots:
            for value in metadata_root.component_values(self.component):
                items.append({self.component: value})
        return tuple(items)

@dataclass(frozen=True, slots=True)
class StreamComponentDomainProviders:
    """Ordered domain-provider set for one viewer component layout."""

    values: tuple[StreamComponentDomainProvider, ...]

    @classmethod
    def from_layout(
        cls,
        *,
        context: ProcessingContext,
        layout: ViewerComponentLayout,
        metadata_roots: tuple[StreamMetadataRoot, ...],
    ) -> "StreamComponentDomainProviders":
        return cls(
            tuple(
                StreamComponentDomainProvider.for_component(
                    context=context,
                    component=component,
                    metadata_roots=metadata_roots,
                )
                for component in layout.component_order
            )
        )

    def domain_metadata_items(self) -> StreamComponentDomainMetadataItems:
        return tuple(
            item
            for provider in self.values
            for item in provider.domain_metadata_items()
        )

    def include_display_names(self, name_map: StreamComponentNameMap) -> None:
        for provider in self.values:
            provider.include_display_names(name_map)

@dataclass(frozen=True, slots=True)
class StreamComponentMessageExtraAuthority:
    """Build viewer wire metadata that declares component domains and display names."""

    viewer_surface: StreamingViewerSurface
    source_metadata_items: StreamSourceComponentMetadataItems
    metadata_roots: tuple[StreamMetadataRoot, ...]
    domain_providers: StreamComponentDomainProviders

    @classmethod
    def from_context(
        cls,
        viewer_surface: StreamingViewerSurface,
        *,
        context: ProcessingContext,
        source_metadata_items: StreamSourceComponentMetadataItems,
    ) -> "StreamComponentMessageExtraAuthority":
        display_input = ViewerObjectDisplayConfigInput(viewer_surface.display_config)
        metadata_roots = StreamMetadataRootAuthority.from_context(context).roots()
        return cls(
            viewer_surface=viewer_surface,
            source_metadata_items=source_metadata_items,
            metadata_roots=metadata_roots,
            domain_providers=StreamComponentDomainProviders.from_layout(
                context=context,
                layout=display_input.layout(),
                metadata_roots=metadata_roots,
            ),
        )

    @classmethod
    def from_viewer_surface(
        cls,
        viewer_surface: StreamingViewerSurface,
        *,
        source_metadata_items: StreamSourceComponentMetadataItems,
    ) -> "StreamComponentMessageExtraAuthority":
        return cls(
            viewer_surface=viewer_surface,
            source_metadata_items=source_metadata_items,
            metadata_roots=StreamMetadataRootAuthority.from_source_identity(
                viewer_surface.source,
            ).roots(),
            domain_providers=StreamComponentDomainProviders(()),
        )

    @property
    def display_input(self) -> ViewerObjectDisplayConfigInput:
        return ViewerObjectDisplayConfigInput(self.viewer_surface.display_config)

    @property
    def layout(self) -> ViewerComponentLayout:
        return self.display_input.layout()

    @property
    def metadata_items(self) -> StreamComponentDomainMetadataItems:
        return (
            *self.domain_providers.domain_metadata_items(),
            *self.source_metadata_items.domain_metadata_items(),
        )

    @property
    def component_axis_semantics(self) -> ViewerComponentAxisSemantics:
        return ViewerComponentAxisSemanticsAuthority.from_display_config(
            display_config=self.display_input,
            value_domain=ViewerComponentValueDomainPayload.from_component_metadata(
                component_layout=self.layout,
                metadata_items=self.metadata_items,
            ),
        )

    def component_names_metadata(
        self,
        component_order: tuple[str, ...],
    ) -> StreamComponentNameMetadata:
        metadata = StreamComponentNameMap()
        self.domain_providers.include_display_names(metadata)
        self.source_metadata_items.include_observed_values(
            metadata,
            component_order,
        )
        return metadata

    def payload(self) -> StreamComponentMessageExtraPayload:
        component_axis_semantics = self.component_axis_semantics
        return StreamComponentMessageExtraPayload.from_axis_semantics(
            component_axis_semantics=component_axis_semantics,
            component_names_metadata=self.component_names_metadata(
                component_axis_semantics.component_order,
            ),
        )

    def message_context(
        self,
        images_dir: str | None = None,
    ) -> ViewerStreamMessageContext:
        return ViewerStreamMessageContext(
            message_extra=self.payload().to_wire_mapping(),
            images_dir=images_dir,
        )

    def viewer_backend_kwargs(
        self,
        *,
        producer: ViewerStreamProducer,
        source_metadata: ViewerStreamSourceMetadata | None = None,
        images_dir: str | None = None,
    ) -> ViewerStreamBackendKwargs:
        if source_metadata is None:
            source_metadata = self.source_metadata_items.viewer_source_metadata(
                self.layout.component_order
            )
        return self.viewer_surface.viewer_backend_kwargs(
            producer=producer,
            source_metadata=source_metadata,
            message_context=self.message_context(images_dir=images_dir),
        )

    def path_mapped_source_metadata(
        self,
        metadata_by_path: Mapping[str, StreamComponentMetadata],
    ) -> ViewerStreamSourceMetadata:
        return StreamViewerComponentMetadataProjector(
            self.layout.component_order
        ).path_mapped_source_metadata(metadata_by_path)
