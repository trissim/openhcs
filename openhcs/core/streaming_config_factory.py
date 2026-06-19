"""Shared behavior and utilities for concrete streaming configuration classes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from objectstate import DataclassFieldAccess, get_base_config_type
from polystore.filemanager import FileManager
from polystore.streaming.viewer_transport import (
    ViewerStreamSourceIdentity,
)
from zmqruntime.viewer_protocol import ViewerTransportEndpoint

from openhcs.constants.constants import Backend, DEFAULT_EXECUTION_SERVER_PORT

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


@dataclass(frozen=True, slots=True)
class StreamingViewerConfigSpec:
    """Declarative identity for one OpenHCS viewer streaming config."""

    viewer_name: str
    registry_key: str
    display_name: str
    step_plan_output_key: str
    presentation: "StreamingViewerPresentation"
    backend: Backend
    visualizer_module: str


@dataclass(frozen=True, slots=True)
class StreamingViewerPresentation:
    """Viewer-facing presentation identity."""

    title: str


@dataclass(frozen=True, slots=True)
class StreamingViewerRuntimeConfig:
    """Concrete runtime objects shared by viewer lifecycle and backend dispatch."""

    transport_endpoint: ViewerTransportEndpoint
    persistent: bool
    presentation: StreamingViewerPresentation


class StreamingConfigBehaviorMixin:
    """Shared implementation for concrete viewer streaming configs."""

    streaming_spec: ClassVar[StreamingViewerConfigSpec]

    @classmethod
    def from_config(cls, config) -> "StreamingConfig | None":
        return cast(
            "StreamingConfig | None",
            DataclassFieldAccess.raw_value(config, cls.streaming_spec.registry_key),
        )

    @classmethod
    def port_from_config(cls, config) -> int | None:
        streaming_config = cls.from_config(config)
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

    def viewer_runtime_config(self) -> StreamingViewerRuntimeConfig:
        return StreamingViewerRuntimeConfig(
            transport_endpoint=ViewerTransportEndpoint(
                port=self.port,
                host=self.host,
                transport_mode=self.transport_mode,
            ),
            persistent=self.persistent,
            presentation=self.streaming_spec.presentation,
        )

    def viewer_surface(
        self,
        source: ViewerStreamSourceIdentity,
    ) -> StreamingViewerSurface:
        return StreamingViewerSurface(
            runtime_config=self.viewer_runtime_config(),
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
        return self.viewer_surface(source)

    def create_visualizer(
        self,
        filemanager: FileManager,
        visualizer_config=None,
    ):
        from importlib import import_module

        import_module(self.streaming_spec.visualizer_module)
        from openhcs.runtime.viewer_protocol import ManagedViewerLifecycleMixin

        visualizer_type = _resolve_visualizer_type(
            ManagedViewerLifecycleMixin,
            viewer_name=self.streaming_spec.viewer_name,
            visualizer_module_name=self.streaming_spec.visualizer_module,
        )
        return visualizer_type(
            filemanager=filemanager,
            runtime_config=self.viewer_runtime_config(),
        )

def build_component_order() -> list[str]:
    """Build canonical streaming component order from filename components."""

    from openhcs.constants import AllComponents

    component_order: list[str] = []
    seen: set[str] = set()
    for component in AllComponents:
        component_name = component.value
        if component_name in seen:
            continue
        component_order.append(component_name)
        seen.add(component_name)
    return component_order


def _resolve_visualizer_type(
    viewer_base_type: type,
    *,
    viewer_name: str,
    visualizer_module_name: str,
) -> type:
    for visualizer_type in _iter_viewer_types(viewer_base_type):
        if visualizer_type.viewer_type == viewer_name:
            return visualizer_type
    raise KeyError(
        f"Imported {visualizer_module_name!r}, but no managed viewer type "
        f"declares viewer_type={viewer_name!r}."
    )


def _iter_viewer_types(viewer_base_type: type) -> Iterator[type]:
    for visualizer_type in viewer_base_type.__subclasses__():
        yield visualizer_type
        yield from _iter_viewer_types(visualizer_type)


def get_all_streaming_ports(
    config=None,
    num_ports_per_type: int = 10,
) -> list[int]:
    """Get all configured streaming ports for all registered streaming config types."""

    from openhcs.config_framework.global_config import get_current_global_config
    from openhcs.core.config import StreamingConfig

    ports = [DEFAULT_EXECUTION_SERVER_PORT]
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
