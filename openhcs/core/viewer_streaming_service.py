"""Public viewer streaming service for plate images and ROI artifacts.

Eliminates duplication between Napari and Fiji streaming code by parametrizing
on viewer_type. Callers provide UI or agent callbacks; all heavy operations run
in background threads.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from objectstate import spawn_thread_with_context
from polystore.streaming.identity import (
    FixedStreamProducerIdentityKind,
    StreamProducerIdentity,
)
from polystore.streaming.viewer_transport import (
    ViewerStreamProducer,
    ViewerStreamSourceIdentity,
)
from zmqruntime.viewer_protocol import ViewerWireMapping

from openhcs.core.streaming_config_factory import StreamingViewerSurface
from openhcs.core.steps.stream_component_semantics import (
    StreamComponentMessageExtraAuthority,
    StreamSourceComponentMetadataItems,
)

if TYPE_CHECKING:
    from openhcs.core.config import StreamingConfig
    from openhcs.microscopes.microscope_base import MicroscopeHandler
    from polystore.filemanager import FileManager
    from zmqruntime.streaming import VisualizerProcessManager

logger = logging.getLogger(__name__)

# Chunk size to prevent file descriptor exhaustion
# Each image creates a shared memory segment (file descriptor on Linux)
CHUNK_SIZE = 50
ROI_ARCHIVE_SUFFIX = ".roi.zip"
SOURCE_FILENAME_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

@dataclass(frozen=True, slots=True)
class ViewerStreamingContext:
    """Shared viewer/request context for asynchronous streaming operations."""

    viewer: VisualizerProcessManager
    config: StreamingConfig
    status_callback: Callable[[str], None]
    error_callback: Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ImageStreamingRequest(ViewerStreamingContext):
    """Request to stream image files to one viewer."""

    filenames: tuple[str, ...]
    read_backend: str


@dataclass(frozen=True, slots=True)
class RoiStreamingRequest(ViewerStreamingContext):
    """Request to stream ROI files to one viewer."""

    roi_filenames: tuple[str, ...]
    component_metadata_by_path: Mapping[str, ViewerWireMapping] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ViewerStreamingResult:
    """Synchronous result for one manual viewer streaming request."""

    viewer_type: str
    port: int
    display_name: str
    payload_kind: str
    requested_count: int
    streamed_count: int
    streamed_paths: tuple[str, ...]
    messages: tuple[str, ...] = ()


class StreamingViewerLifecycle:
    """Shared lifecycle entrypoint for manual and orchestrated viewer streaming."""

    @staticmethod
    def get_or_create_visualizer(
        *,
        filemanager: FileManager,
        config: StreamingConfig,
        visualizer_config=None,
        fresh: bool = True,
        ready_timeout: float = 30.0,
    ) -> VisualizerProcessManager:
        from zmqruntime import ViewerStateManager, get_or_create_viewer
        from zmqruntime.queue_tracker import GlobalQueueTrackerRegistry
        from openhcs.runtime.viewer_protocol import ManagedViewerLifecycleMixin

        registry = GlobalQueueTrackerRegistry()
        registry.get_or_create_tracker(config.port, config.viewer_type)
        manager = ViewerStateManager.get_instance()

        if fresh:
            manager.release_viewer(
                config.viewer_type,
                config.port,
                stop=True,
                force=True,
            )
        else:
            managed_viewer = manager.get_viewer(config.viewer_type, config.port)
            if managed_viewer is not None:
                return managed_viewer

            external_viewer = config.create_visualizer(filemanager, visualizer_config)
            if not isinstance(external_viewer, ManagedViewerLifecycleMixin):
                raise TypeError(
                    "Streaming viewer config produced an unsupported viewer "
                    f"lifecycle type: {type(external_viewer).__name__}"
                )
            if external_viewer.existing_viewer_is_ready():
                external_viewer.lifecycle_state.mark_connected_external()
                return external_viewer

        viewer, _created = get_or_create_viewer(
            viewer_type=config.viewer_type,
            port=config.port,
            factory=lambda: config.create_visualizer(filemanager, visualizer_config),
            wait_for_ready=True,
            ready_timeout=ready_timeout,
        )
        return viewer


class StreamingSourceFilenameAuthority:
    """Resolve source-image filename candidates for streamed viewer artifacts."""

    @staticmethod
    def roi_artifact_stem(filename: str) -> str:
        name = Path(filename).name
        if name.lower().endswith(ROI_ARCHIVE_SUFFIX):
            return name[: -len(ROI_ARCHIVE_SUFFIX)]
        return Path(name).stem

    @classmethod
    def source_filename_candidates_for_roi(cls, filename: str) -> tuple[str, ...]:
        """Return plausible source-image names for an analysis ROI artifact."""
        candidates: list[str] = []

        def add(value: str) -> None:
            if value and value not in candidates:
                candidates.append(value)

        name = Path(filename).name
        stem = cls.roi_artifact_stem(name)
        add(name)
        add(stem)

        bases = [stem]
        for pattern in (
            r"^(?P<base>.+)_step\d+_rois$",
            r"^(?P<base>.+)_step\d+$",
            r"^(?P<base>.+)_rois$",
        ):
            match = re.match(pattern, stem)
            if match:
                bases.append(match.group("base"))

        parts = stem.split("_")
        for end in range(len(parts) - 1, 0, -1):
            bases.append("_".join(parts[:end]))

        for base in bases:
            add(base)
            if not Path(base).suffix:
                for extension in SOURCE_FILENAME_EXTENSIONS:
                    add(f"{base}{extension}")

        return tuple(candidates)

    @staticmethod
    def source_filename_candidates_for_image(filename: str) -> tuple[str, ...]:
        name = Path(filename).name
        if filename == name:
            return (name,)
        return (filename, name)


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerStreamingSource(ViewerStreamSourceIdentity):
    """Source authority for viewer streaming from one initialized plate."""

    filemanager: FileManager

    def load_image(self, filename: str, read_backend: str):
        return self.filemanager.load(
            str(Path(self.plate_path) / filename),
            read_backend,
        )

    def component_metadata_by_path(
        self,
        paths: list[str],
        candidate_names_for_path: Callable[[str], tuple[str, ...]],
        artifact_label: str,
    ) -> dict[str, ViewerWireMapping]:
        parser = self.microscope_handler.parser
        metadata_by_path: dict[str, ViewerWireMapping] = {}

        for path in paths:
            parsed: ViewerWireMapping | None = None
            for candidate in candidate_names_for_path(path):
                parsed = parser.parse_filename(candidate)
                if parsed is not None:
                    break

            if parsed is None:
                raise ValueError(
                    "Could not resolve source-plane metadata for "
                    f"{artifact_label} {path!r}; streaming requires explicit "
                    "component metadata."
                )
            metadata_by_path[path] = dict(parsed)

        return metadata_by_path

    def image_component_metadata_by_path(
        self,
        paths: list[str],
    ) -> dict[str, ViewerWireMapping]:
        return self.component_metadata_by_path(
            paths,
            StreamingSourceFilenameAuthority.source_filename_candidates_for_image,
            "image",
        )

    def roi_component_metadata_by_path(
        self,
        paths: list[str],
    ) -> dict[str, ViewerWireMapping]:
        return self.component_metadata_by_path(
            paths,
            StreamingSourceFilenameAuthority.source_filename_candidates_for_roi,
            "ROI artifact",
        )


class StreamingService:
    """Unified service for streaming images/ROIs to viewers.

    Handles all viewer communication in background threads.
    Uses callbacks for caller-owned status updates and error handling.
    """

    def __init__(
        self,
        filemanager: FileManager,
        microscope_handler: MicroscopeHandler,
        plate_path: Path,
    ):
        self.source = ViewerStreamingSource(
            filemanager=filemanager,
            microscope_handler=microscope_handler,
            plate_path=plate_path,
        )

    def _wait_for_viewer_ready(
        self,
        viewer: VisualizerProcessManager,
        config: StreamingConfig,
        num_items: int,
    ) -> None:
        """Wait for viewer to be ready, registering as launching if needed."""
        # Use centralized ViewerStateManager for launching/queued state
        from zmqruntime.viewer_state import ViewerStateManager

        manager = ViewerStateManager.get_instance()

        is_already_running = viewer.runtime_endpoint.wait_ready(
            timeout=0.1,
            require_ready=True,
        )

        # Update queued images for UI display via manager. The QueueTracker
        # will later update counts precisely as images are sent/acked.
        manager.update_queued_images(config.viewer_type, viewer.port, num_items)

        if not is_already_running:
            logger.info(
                f"Waiting for {config.display_name} viewer on port {viewer.port} to become ready"
            )

            if not viewer.runtime_endpoint.wait_ready(
                timeout=15.0,
                require_ready=True,
            ):
                # Clear queued count for UI if startup failed
                manager.update_queued_images(config.viewer_type, viewer.port, 0)
                raise RuntimeError(
                    f"{config.display_name} viewer on port {viewer.port} failed to become ready"
                )

            logger.info(f"{config.display_name} viewer on port {viewer.port} is ready")

    def stream_images_async(
        self,
        request: ImageStreamingRequest,
    ) -> None:
        """Load and stream images to viewer in background thread.

        Uses chunked streaming to prevent file descriptor exhaustion.
        """
        display_name = request.config.display_name

        def _worker():
            try:
                self.stream_images(request)
            except Exception as e:
                logger.error(f"Failed to stream images to {display_name}: {e}")
                request.status_callback(f"Error: {e}")
                request.error_callback(str(e))

        spawn_thread_with_context(
            _worker,
            name=f"stream_images_{request.config.viewer_type}",
        )
        logger.info(
            f"Started streaming {len(request.filenames)} images to {display_name}"
        )

    def stream_images(
        self,
        request: ImageStreamingRequest,
    ) -> ViewerStreamingResult:
        """Load and stream images to a viewer before returning."""
        backend_enum = request.config.backend
        display_name = request.config.display_name
        messages: list[str] = []

        self._wait_for_viewer_ready(
            request.viewer,
            request.config,
            len(request.filenames),
        )

        total_images = len(request.filenames)
        num_chunks = (total_images + CHUNK_SIZE - 1) // CHUNK_SIZE
        logger.info(f"Streaming {total_images} images in {num_chunks} chunks")
        all_metadata_by_path = self.source.image_component_metadata_by_path(
            list(request.filenames)
        )
        viewer_surface = request.config.viewer_surface(self.source)
        source_metadata_items = StreamSourceComponentMetadataItems.from_values(
            all_metadata_by_path[path] for path in request.filenames
        )
        message_authority = StreamComponentMessageExtraAuthority.from_viewer_surface(
            viewer_surface,
            source_metadata_items=source_metadata_items,
        )
        stream_backend_kwargs = message_authority.viewer_backend_kwargs(
            producer=ViewerStreamProducer.from_identity(
                StreamProducerIdentity.fixed_output(
                    FixedStreamProducerIdentityKind.MANUAL,
                    "selected_images",
                )
            ),
            source_metadata=message_authority.path_mapped_source_metadata(
                all_metadata_by_path
            ),
        )

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE, total_images)
            chunk_filenames = request.filenames[start_idx:end_idx]

            message = (
                f"Loading chunk {chunk_idx + 1}/{num_chunks} "
                f"({len(chunk_filenames)} images)..."
            )
            messages.append(message)
            request.status_callback(message)

            image_data_list = []
            file_paths = []
            for filename in chunk_filenames:
                image_data = self.source.load_image(
                    filename,
                    request.read_backend,
                )
                image_data_list.append(image_data)
                file_paths.append(filename)

            logger.info(
                f"Loaded chunk {chunk_idx + 1}/{num_chunks}: {len(image_data_list)} images"
            )

            self.source.filemanager.save_batch(
                image_data_list,
                file_paths,
                backend_enum.value,
                **stream_backend_kwargs.to_kwargs(),
            )
            logger.info(
                f"Streamed chunk {chunk_idx + 1}/{num_chunks} to {display_name}"
            )

            if chunk_idx < num_chunks - 1:
                time.sleep(0.1)

        message = f"Streamed {total_images} images to {display_name}"
        logger.info("Successfully %s", message.lower())
        messages.append(message)
        request.status_callback(message)
        return ViewerStreamingResult(
            viewer_type=request.config.viewer_type,
            port=request.viewer.port,
            display_name=display_name,
            payload_kind="image",
            requested_count=total_images,
            streamed_count=total_images,
            streamed_paths=tuple(request.filenames),
            messages=tuple(messages),
        )

    def stream_rois_async(
        self,
        request: RoiStreamingRequest,
    ) -> None:
        """Load and stream ROI files to viewer in background thread."""
        display_name = request.config.display_name

        def _worker():
            try:
                self.stream_rois(request)
            except Exception as e:
                logger.error(f"Failed to stream ROIs to {display_name}: {e}")
                request.status_callback(f"Error: {e}")
                request.error_callback(str(e))

        spawn_thread_with_context(
            _worker,
            name=f"stream_rois_{request.config.viewer_type}",
        )

    def stream_rois(
        self,
        request: RoiStreamingRequest,
    ) -> ViewerStreamingResult:
        """Load and stream ROI files to a viewer before returning."""
        from polystore.roi import load_rois_from_zip

        backend_enum = request.config.backend
        display_name = request.config.display_name
        messages: list[str] = []

        total = len(request.roi_filenames)
        if total == 0:
            return ViewerStreamingResult(
                viewer_type=request.config.viewer_type,
                port=request.viewer.port,
                display_name=display_name,
                payload_kind="rois",
                requested_count=0,
                streamed_count=0,
                streamed_paths=(),
            )

        message = f"Loading {total} ROI file(s) from disk..."
        messages.append(message)
        request.status_callback(message)

        data_list: list = []
        paths: list[str] = []

        for i, filename in enumerate(request.roi_filenames, 1):
            file_path = Path(self.source.plate_path) / filename
            rois = load_rois_from_zip(file_path)
            if not rois:
                logger.warning(f"No ROIs found in {file_path.name}")
                continue

            data_list.append(rois)
            paths.append(filename)

            if i % 5 == 0 or i == total:
                message = f"Loading ROIs: {i}/{total} file(s)..."
                messages.append(message)
                request.status_callback(message)

        if not data_list:
            message = "No ROIs loaded from any selected files."
            logger.warning(message)
            messages.append(message)
            request.status_callback(message)
            return ViewerStreamingResult(
                viewer_type=request.config.viewer_type,
                port=request.viewer.port,
                display_name=display_name,
                payload_kind="rois",
                requested_count=total,
                streamed_count=0,
                streamed_paths=(),
                messages=tuple(messages),
            )

        self._wait_for_viewer_ready(
            request.viewer,
            request.config,
            len(paths),
        )

        viewer_surface = request.config.viewer_surface(self.source)
        if request.component_metadata_by_path:
            missing_metadata = tuple(
                path for path in paths if path not in request.component_metadata_by_path
            )
            if missing_metadata:
                raise ValueError(
                    "ROI streaming received explicit component metadata, but "
                    f"metadata was missing for {missing_metadata!r}."
                )
            metadata_by_path = {
                path: dict(request.component_metadata_by_path[path])
                for path in paths
            }
        else:
            metadata_by_path = self.source.roi_component_metadata_by_path(paths)
        source_metadata_items = StreamSourceComponentMetadataItems.from_values(
            metadata_by_path[path] for path in paths
        )
        message_authority = StreamComponentMessageExtraAuthority.from_viewer_surface(
            viewer_surface,
            source_metadata_items=source_metadata_items,
        )
        stream_backend_kwargs = message_authority.viewer_backend_kwargs(
            producer=ViewerStreamProducer.from_identity(
                StreamProducerIdentity.fixed_output(
                    FixedStreamProducerIdentityKind.MANUAL,
                    "selected_rois",
                )
            ),
            source_metadata=message_authority.path_mapped_source_metadata(
                metadata_by_path
            ),
        )

        message = f"Streaming {len(paths)} ROI file(s) to {display_name}..."
        messages.append(message)
        request.status_callback(message)

        self.source.filemanager.save_batch(
            data_list,
            paths,
            backend_enum.value,
            **stream_backend_kwargs.to_kwargs(),
        )

        message = (
            f"Streamed {len(paths)} ROI file(s) to {display_name} "
            f"on port {request.viewer.port}"
        )
        logger.info(message)
        messages.append(message)
        request.status_callback(message)
        return ViewerStreamingResult(
            viewer_type=request.config.viewer_type,
            port=request.viewer.port,
            display_name=display_name,
            payload_kind="rois",
            requested_count=total,
            streamed_count=len(paths),
            streamed_paths=tuple(paths),
            messages=tuple(messages),
        )
