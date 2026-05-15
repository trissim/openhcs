"""Unified streaming service for viewer communication (Napari/Fiji).

Eliminates duplication between Napari and Fiji streaming code by parametrizing
on viewer_type. All heavy operations run in background threads.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from objectstate import spawn_thread_with_context

if TYPE_CHECKING:
    from polystore.filemanager import FileManager

logger = logging.getLogger(__name__)

# Chunk size to prevent file descriptor exhaustion
# Each image creates a shared memory segment (file descriptor on Linux)
CHUNK_SIZE = 50

# ViewerType is now the registry key (e.g., 'napari_streaming_config', 'fiji_streaming_config')
# This provides type safety and consistency throughout the codebase
ViewerType = str  # Registry key from StreamingConfig.__registry__
NAPARI_VIEWER_TOKEN = "napari"


@dataclass(frozen=True, slots=True)
class StreamingMetadata:
    """Nominal metadata record passed to viewer streaming backends."""

    port: int
    host: str
    transport_mode: object
    display_config: object
    microscope_handler: object
    plate_path: Path
    source: str

    @classmethod
    def from_viewer_context(
        cls,
        *,
        viewer,
        config,
        microscope_handler,
        plate_path: Path,
        source: str,
    ) -> "StreamingMetadata":
        return cls(
            port=viewer.port,
            host=config.host,
            transport_mode=config.transport_mode,
            display_config=config,
            microscope_handler=microscope_handler,
            plate_path=plate_path,
            source=source,
        )

    def to_backend_kwargs(self) -> dict[str, object]:
        """Convert to kwargs at the third-party backend boundary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ViewerStreamingContext:
    """Shared viewer/request context for asynchronous streaming operations."""

    viewer: object
    plate_path: Path
    config: object
    viewer_type: ViewerType
    status_callback: Callable[[str], None]
    error_callback: Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ImageStreamingRequest:
    """Request to stream image files to one viewer."""

    context: ViewerStreamingContext
    filenames: tuple[str, ...]
    read_backend: str


@dataclass(frozen=True, slots=True)
class RoiStreamingRequest:
    """Request to stream ROI files to one viewer."""

    context: ViewerStreamingContext
    roi_filenames: tuple[str, ...]


class StreamingService:
    """Unified service for streaming images/ROIs to viewers.

    Handles all viewer communication in background threads.
    Uses callbacks for UI thread communication (status updates, errors).
    """

    def __init__(
        self,
        filemanager: FileManager,
        microscope_handler,
        plate_path: Path,
    ):
        self.filemanager = filemanager
        self.microscope_handler = microscope_handler
        self.plate_path = plate_path

    @staticmethod
    def display_name_for_viewer_type(viewer_type: str) -> str:
        """Get display name from registry key.

        Args:
            viewer_type: Registry key (e.g., 'napari_streaming_config')

        Returns:
            Display name (e.g., 'Napari')
        """
        # Extract viewer name from registry key (e.g., 'napari_streaming_config' -> 'napari')
        viewer_name = viewer_type.replace('_streaming_config', '')
        return viewer_name.title()

    _get_display_name = display_name_for_viewer_type

    @classmethod
    def supported_viewer_types(cls):
        """Return a list of supported viewer short names (e.g. 'napari', 'fiji').

        Centralized so UI can discover which viewer buttons to create instead
        of hardcoding Napari/Fiji in multiple places.
        """
        # Use the metaclass-driven StreamingConfig registry.
        # StreamingConfig uses AutoRegisterMeta with __registry_key__ = 'viewer_type'
        from openhcs.core.config import StreamingConfig

        # Return stable ordering from the registry keys
        return sorted(list(StreamingConfig.__registry__.keys()))

    def _get_backend_enum(self, viewer_type: str):
        """Get the backend enum for the viewer type.

        Args:
            viewer_type: Registry key (e.g., 'napari_streaming_config', 'fiji_streaming_config')
        """
        from openhcs.constants.constants import Backend as BackendEnum

        return (
            BackendEnum.NAPARI_STREAM
            if self.is_napari_viewer_type(viewer_type)
            else BackendEnum.FIJI_STREAM
        )

    @staticmethod
    def is_napari_viewer_type(viewer_type: str) -> bool:
        """Return whether a streaming registry key targets napari."""

        return NAPARI_VIEWER_TOKEN in viewer_type

    def _wait_for_viewer_ready(
        self,
        viewer,
        viewer_type: ViewerType,
        num_items: int,
    ) -> None:
        """Wait for viewer to be ready, registering as launching if needed."""
        # Use centralized ViewerStateManager for launching/queued state
        from zmqruntime.viewer_state import ViewerStateManager

        manager = ViewerStateManager.get_instance()

        is_already_running = viewer.wait_for_ready(timeout=0.1)

        # Update queued images for UI display via manager. The QueueTracker
        # will later update counts precisely as images are sent/acked.
        manager.update_queued_images(viewer_type, viewer.port, num_items)

        if not is_already_running:
            display_name = self._get_display_name(viewer_type)
            logger.info(
                f"Waiting for {display_name} viewer on port {viewer.port} to become ready"
            )

            if not viewer.wait_for_ready(timeout=15.0):
                # Clear queued count for UI if startup failed
                manager.update_queued_images(viewer_type, viewer.port, 0)
                raise RuntimeError(
                    f"{display_name} viewer on port {viewer.port} failed to become ready"
                )

            logger.info(
                f"{display_name} viewer on port {viewer.port} is ready"
            )

    def stream_images_async(
        self,
        request: ImageStreamingRequest,
    ) -> None:
        """Load and stream images to viewer in background thread.

        Uses chunked streaming to prevent file descriptor exhaustion.
        """
        context = request.context
        backend_enum = self._get_backend_enum(context.viewer_type)
        display_name = self.display_name_for_viewer_type(context.viewer_type)

        def _worker():
            try:
                self._wait_for_viewer_ready(
                    context.viewer,
                    context.viewer_type,
                    len(request.filenames),
                )

                total_images = len(request.filenames)
                num_chunks = (total_images + CHUNK_SIZE - 1) // CHUNK_SIZE
                logger.info(f"Streaming {total_images} images in {num_chunks} chunks")

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * CHUNK_SIZE
                    end_idx = min(start_idx + CHUNK_SIZE, total_images)
                    chunk_filenames = request.filenames[start_idx:end_idx]

                    context.status_callback(
                        f"Loading chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_filenames)} images)..."
                    )

                    # Load chunk
                    image_data_list = []
                    file_paths = []
                    for filename in chunk_filenames:
                        image_path = context.plate_path / filename
                        image_data = self.filemanager.load(
                            str(image_path), request.read_backend
                        )
                        image_data_list.append(image_data)
                        file_paths.append(filename)

                    logger.info(
                        f"Loaded chunk {chunk_idx + 1}/{num_chunks}: {len(image_data_list)} images"
                    )

                    source = (
                        Path(file_paths[0]).parent.name
                        if file_paths
                        else "unknown_source"
                    )
                    metadata = StreamingMetadata.from_viewer_context(
                        viewer=context.viewer,
                        config=context.config,
                        microscope_handler=self.microscope_handler,
                        plate_path=context.plate_path,
                        source=source,
                    ).to_backend_kwargs()

                    self.filemanager.save_batch(
                        image_data_list, file_paths, backend_enum.value, **metadata
                    )
                    logger.info(
                        f"Streamed chunk {chunk_idx + 1}/{num_chunks} to {display_name}"
                    )

                    if chunk_idx < num_chunks - 1:
                        time.sleep(0.1)

                logger.info(
                    f"Successfully streamed {total_images} images to {display_name}"
                )
                context.status_callback(
                    f"Streamed {total_images} images to {display_name}"
                )

            except Exception as e:
                logger.error(f"Failed to stream images to {display_name}: {e}")
                context.status_callback(f"Error: {e}")
                context.error_callback(str(e))

        spawn_thread_with_context(
            _worker,
            name=f"stream_images_{context.viewer_type}",
        )
        logger.info(
            f"Started streaming {len(request.filenames)} images to {display_name}"
        )

    def stream_rois_async(
        self,
        request: RoiStreamingRequest,
    ) -> None:
        """Load and stream ROI files to viewer in background thread."""
        context = request.context
        backend_enum = self._get_backend_enum(context.viewer_type)
        display_name = self.display_name_for_viewer_type(context.viewer_type)

        def _worker():
            try:
                from polystore.roi import load_rois_from_zip

                total = len(request.roi_filenames)
                if total == 0:
                    return

                context.status_callback(f"Loading {total} ROI file(s) from disk...")

                data_list: list = []
                paths: list[str] = []

                for i, filename in enumerate(request.roi_filenames, 1):
                    file_path = context.plate_path / filename
                    rois = load_rois_from_zip(file_path)
                    if not rois:
                        logger.warning(f"No ROIs found in {file_path.name}")
                        continue

                    data_list.append(rois)
                    paths.append(filename)

                    if i % 5 == 0 or i == total:
                        context.status_callback(f"Loading ROIs: {i}/{total} file(s)...")

                if not data_list:
                    msg = "No ROIs loaded from any selected files."
                    logger.warning(msg)
                    context.status_callback(msg)
                    return

                self._wait_for_viewer_ready(
                    context.viewer,
                    context.viewer_type,
                    len(paths),
                )

                source = Path(paths[0]).parent.name if paths else "unknown_source"
                metadata = StreamingMetadata.from_viewer_context(
                    viewer=context.viewer,
                    config=context.config,
                    microscope_handler=self.microscope_handler,
                    plate_path=context.plate_path,
                    source=source,
                ).to_backend_kwargs()

                context.status_callback(
                    f"Streaming {len(paths)} ROI file(s) to {display_name}..."
                )

                self.filemanager.save_batch(
                    data_list, paths, backend_enum.value, **metadata
                )

                msg = (
                    f"Streamed {len(paths)} ROI file(s) to {display_name} "
                    f"on port {context.viewer.port}"
                )
                logger.info(msg)
                context.status_callback(msg)

            except Exception as e:
                logger.error(f"Failed to stream ROIs to {display_name}: {e}")
                context.status_callback(f"Error: {e}")
                context.error_callback(str(e))

        spawn_thread_with_context(_worker, name=f"stream_rois_{context.viewer_type}")
