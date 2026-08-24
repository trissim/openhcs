"""Nominal declarations for OpenHCS streaming viewer families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from openhcs.constants.constants import Backend

if TYPE_CHECKING:
    from polystore.filemanager import FileManager

    from openhcs.core.execution_visualizer import ExecutionVisualizerABC
    from openhcs.core.streaming_config_factory import StreamingViewerRuntimeConfig


class ViewerDeclarationABC(ABC):
    """Viewer-specific behavior owned by one :class:`ViewerType` member."""

    @property
    @abstractmethod
    def backend(self) -> Backend:
        """Return the storage backend used by this viewer family."""

    @abstractmethod
    def visualizer_type(self) -> type["ExecutionVisualizerABC"]:
        """Load and return this viewer's concrete lifecycle implementation."""

    def create_visualizer(
        self,
        *,
        filemanager: "FileManager",
        runtime_config: "StreamingViewerRuntimeConfig",
    ) -> "ExecutionVisualizerABC":
        """Construct this viewer's lifecycle implementation."""

        return self.visualizer_type()(
            filemanager=filemanager,
            runtime_config=runtime_config,
        )


class NapariViewerDeclaration(ViewerDeclarationABC):
    """Napari streaming and lifecycle integration."""

    @property
    def backend(self) -> Backend:
        return Backend.NAPARI_STREAM

    def visualizer_type(self) -> type["ExecutionVisualizerABC"]:
        from openhcs.runtime.napari_stream_visualizer import NapariStreamVisualizer

        return NapariStreamVisualizer


class FijiViewerDeclaration(ViewerDeclarationABC):
    """Fiji streaming and lifecycle integration."""

    @property
    def backend(self) -> Backend:
        return Backend.FIJI_STREAM

    def visualizer_type(self) -> type["ExecutionVisualizerABC"]:
        from openhcs.runtime.fiji_stream_visualizer import FijiStreamVisualizer

        return FijiStreamVisualizer


class ViewerType(Enum):
    """Streaming viewer identities carrying their nominal leaf declarations."""

    FIJI = ("fiji", FijiViewerDeclaration)
    NAPARI = ("napari", NapariViewerDeclaration)

    def __new__(
        cls,
        wire_value: str,
        declaration_type: type[ViewerDeclarationABC],
    ) -> "ViewerType":
        member = object.__new__(cls)
        member._value_ = wire_value
        member._declaration_type = declaration_type
        return member

    @property
    def declaration(self) -> ViewerDeclarationABC:
        """Return the nominal implementation owned by this viewer member."""

        return self._declaration_type()

    @property
    def wire_value(self) -> str:
        """Return the stable external identifier for transport boundaries."""

        return self.value

    @classmethod
    def from_wire_value(cls, value: str) -> "ViewerType":
        """Resolve one external viewer identifier at the transport boundary."""

        return cls(value)

    @property
    def config_key(self) -> str:
        """Return the ObjectState field projected from this viewer identity."""

        return f"{self.wire_value}_streaming_config"

    @classmethod
    def from_config_key(cls, config_key: str) -> "ViewerType":
        """Resolve an ObjectState field key at its configuration boundary."""

        matches = tuple(member for member in cls if member.config_key == config_key)
        if len(matches) != 1:
            raise ValueError(f"Unknown viewer streaming config key: {config_key!r}")
        return matches[0]

    @property
    def display_name(self) -> str:
        """Return the human-facing viewer name."""

        return self.name.title()

    @property
    def step_plan_output_key(self) -> str:
        """Return the compiled step-plan output key for this viewer."""

        return f"{self.wire_value}_streaming_paths"

    @property
    def title(self) -> str:
        """Return the managed viewer window title."""

        return f"OpenHCS {self.display_name} Visualization"

    @property
    def backend(self) -> Backend:
        """Return the backend declared by this viewer's nominal leaf."""

        return self.declaration.backend

    def create_visualizer(
        self,
        *,
        filemanager: "FileManager",
        runtime_config: "StreamingViewerRuntimeConfig",
    ) -> "ExecutionVisualizerABC":
        """Construct the lifecycle implementation owned by this member."""

        return self.declaration.create_visualizer(
            filemanager=filemanager,
            runtime_config=runtime_config,
        )
