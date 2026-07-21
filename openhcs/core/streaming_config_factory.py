"""Shared behavior and utilities for concrete streaming configuration classes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from openhcs.core.streaming_config_declarations import (
    StreamingViewerConfigSpec,
    StreamingViewerPresentation,
)
from objectstate import DataclassFieldAccess, get_base_config_type
from polystore.filemanager import FileManager
from polystore.streaming.identity import StreamProducerIdentity
from polystore.streaming.viewer_transport import (
    ExplicitViewerTransportConfig,
    ViewerStreamBackendKwargs,
    ViewerStreamMessageContext,
    ViewerStreamProducer,
    ViewerStreamRequest,
    ViewerStreamSource,
    ViewerStreamSourceIdentity,
    ViewerStreamSourceMetadata,
)
from zmqruntime.config import ZMQConfig
from zmqruntime.viewer_protocol import ViewerTransportEndpoint, ViewerWireValue

from openhcs.constants.constants import Backend
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG

if TYPE_CHECKING:
    from openhcs.core.config import StreamingConfig
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.microscopes.microscope_base import MicroscopeHandler


@dataclass(frozen=True, slots=True)
class StreamingViewerSurface:
    """Viewer runtime, display, and source surface for one OpenHCS stream."""

    runtime_config: "StreamingViewerRuntimeConfig"
    display_config: "StreamingConfig"
    source: ViewerStreamSourceIdentity

    def viewer_stream_request(
        self,
        *,
        producer: ViewerStreamProducer,
        source_metadata: ViewerStreamSourceMetadata,
        message_context: ViewerStreamMessageContext,
    ) -> ViewerStreamRequest:
        return ViewerStreamRequest.from_message_context(
            message_context=message_context,
            viewer_transport=self.runtime_config.transport_endpoint,
            display_config=self.display_config,
            source=ViewerStreamSource(
                identity=self.source,
                metadata=source_metadata,
            ),
            producer=producer,
            transport_config=ExplicitViewerTransportConfig(
                self.runtime_config.transport_config
            ),
        )

    def viewer_backend_kwargs(
        self,
        *,
        producer: ViewerStreamProducer,
        source_metadata: ViewerStreamSourceMetadata,
        message_context: ViewerStreamMessageContext,
    ) -> ViewerStreamBackendKwargs:
        return ViewerStreamBackendKwargs(
            self.viewer_stream_request(
                producer=producer,
                source_metadata=source_metadata,
                message_context=message_context,
            )
        )


@dataclass(frozen=True, slots=True)
class StreamingViewerRuntimeConfig:
    """Concrete runtime objects shared by viewer lifecycle and backend dispatch."""

    transport_endpoint: ViewerTransportEndpoint
    persistent: bool
    presentation: StreamingViewerPresentation
    transport_config: ZMQConfig = OPENHCS_ZMQ_CONFIG
    display_enabled: bool = True


class StreamingConfigBehaviorMixin:
    """Shared implementation for concrete viewer streaming configs."""

    streaming_spec: ClassVar[StreamingViewerConfigSpec]

    @classmethod
    def port_from_config(cls, config) -> int | None:
        streaming_config = DataclassFieldAccess.raw_value(
            config,
            cls.streaming_spec.registry_key,
        )
        if streaming_config is None:
            return None
        return streaming_config.port

    @property
    def backend(self) -> Backend:
        return self.streaming_spec.backend

    @property
    def viewer_type(self) -> str:
        return self.streaming_spec.viewer_name

    @property
    def streaming_config_key(self) -> str:
        return self.streaming_spec.registry_key

    @property
    def display_name(self) -> str:
        return self.streaming_spec.display_name

    @property
    def step_plan_output_key(self) -> str:
        return self.streaming_spec.step_plan_output_key

    @property
    def viewer_title(self) -> str:
        return self.streaming_spec.presentation.title

    def viewer_runtime_config(
        self,
        transport_config: ZMQConfig = OPENHCS_ZMQ_CONFIG,
    ) -> StreamingViewerRuntimeConfig:
        return StreamingViewerRuntimeConfig(
            transport_endpoint=ViewerTransportEndpoint(
                port=self.port,
                host=self.host,
                transport_mode=self.transport_mode,
            ),
            transport_config=transport_config,
            persistent=self.persistent,
            display_enabled=self.enabled,
            presentation=self.streaming_spec.presentation,
        )

    def viewer_surface(
        self,
        source: ViewerStreamSourceIdentity,
        transport_config: ZMQConfig = OPENHCS_ZMQ_CONFIG,
    ) -> StreamingViewerSurface:
        return StreamingViewerSurface(
            runtime_config=self.viewer_runtime_config(transport_config),
            display_config=self,
            source=source,
        )

    def streaming_viewer_surface(
        self,
        context: "ProcessingContext | None",
    ) -> StreamingViewerSurface:
        if context is None:
            raise ValueError("Streaming viewer surface requires a ProcessingContext.")
        source = ViewerStreamSourceIdentity(
            microscope_handler=context.microscope_handler,
            plate_path=context.plate_path,
        )
        return self.viewer_surface(source, context.transport_config)

    def create_visualizer(
        self,
        filemanager: FileManager,
        visualizer_config=None,
        transport_config: ZMQConfig = OPENHCS_ZMQ_CONFIG,
    ):
        from importlib import import_module

        import_module(self.streaming_spec.visualizer_module)
        from openhcs.runtime.viewer_protocol import ManagedViewerLifecycleMixin

        visualizer_type = ManagedViewerTypeResolver.resolve(
            ManagedViewerLifecycleMixin,
            self.streaming_spec,
        )
        return visualizer_type(
            filemanager=filemanager,
            runtime_config=self.viewer_runtime_config(transport_config),
        )

class ManagedViewerTypeResolver:
    """Resolve managed viewer runtime classes from streaming config declarations."""

    @classmethod
    def resolve(
        cls,
        viewer_base_type: type,
        spec: StreamingViewerConfigSpec,
    ) -> type:
        for visualizer_type in cls.viewer_types(viewer_base_type):
            if visualizer_type.viewer_type == spec.viewer_name:
                return visualizer_type
        raise KeyError(
            f"Imported {spec.visualizer_module!r}, but no managed viewer type "
            f"declares viewer_type={spec.viewer_name!r}."
        )

    @classmethod
    def viewer_types(cls, viewer_base_type: type) -> Iterator[type]:
        for visualizer_type in viewer_base_type.__subclasses__():
            yield visualizer_type
            yield from cls.viewer_types(visualizer_type)


def get_all_streaming_ports(
    config=None,
    num_ports_per_type: int = 10,
) -> list[int]:
    """Get all configured streaming ports for all registered streaming config types."""

    from openhcs.config_framework.global_config import get_current_global_config
    from openhcs.core.config import StreamingConfig

    ports: list[int] = []
    base_config_type = get_base_config_type()

    if config is None:
        config = get_current_global_config(base_config_type)
        if config is None:
            return ports

    global_config = get_current_global_config(base_config_type)

    for streaming_config_type in tuple(StreamingConfig.__registry__.values()):
        port = streaming_config_type.port_from_config(config)
        if port is None and global_config is not None:
            port = streaming_config_type.port_from_config(global_config)
        if port is None:
            continue
        ports.extend(port + i for i in range(num_ports_per_type))

    return ports
