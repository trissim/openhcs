"""Unified streaming service for viewer communication (Napari/Fiji).

Eliminates duplication between Napari and Fiji streaming code by parametrizing
on viewer_type. All heavy operations run in background threads.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

from objectstate import spawn_thread_with_context

if TYPE_CHECKING:
    from polystore.filemanager import FileManager

logger = logging.getLogger(__name__)

# Chunk size to prevent file descriptor exhaustion
# Each image creates a shared memory segment (file descriptor on Linux)
CHUNK_SIZE = 50
ROI_ARCHIVE_SUFFIX = ".roi.zip"
SINGLE_PLANE_COMPONENT_DEFAULTS: Mapping[str, object] = {
    "site": 1,
    "channel": 1,
    "z_index": 1,
    "timepoint": 1,
}
SOURCE_FILENAME_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

ViewerType = str
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
        """Convert to kwargs while preserving runtime object identities."""

        return {
            "port": self.port,
            "host": self.host,
            "transport_mode": self.transport_mode,
            "display_config": self.display_config,
            "microscope_handler": self.microscope_handler,
            "plate_path": self.plate_path,
            "source": self.source,
        }


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
        """Get display name from a viewer identity or streaming config key.

        Args:
            viewer_type: Viewer identity or registry key.

        Returns:
            Display name (e.g., 'Napari')
        """
        from openhcs.core.config import StreamingConfig

        config_cls = StreamingConfig.__registry__.get(viewer_type)
        if config_cls is not None:
            viewer_name = config_cls().viewer_type
        else:
            viewer_name = viewer_type
        return viewer_name.title()

    _get_display_name = display_name_for_viewer_type

    @staticmethod
    def _source_for_stream_paths(paths: list[str], fallback: str) -> str:
        if not paths:
            return fallback
        parent_name = Path(paths[0]).parent.name
        return parent_name or fallback

    @staticmethod
    def _roi_artifact_stem(filename: str) -> str:
        name = Path(filename).name
        if name.lower().endswith(ROI_ARCHIVE_SUFFIX):
            return name[: -len(ROI_ARCHIVE_SUFFIX)]
        return Path(name).stem

    @classmethod
    def _source_filename_candidates_for_roi(cls, filename: str) -> tuple[str, ...]:
        """Return plausible source-image names for an analysis ROI artifact."""
        candidates: list[str] = []

        def add(value: str) -> None:
            if value and value not in candidates:
                candidates.append(value)

        name = Path(filename).name
        stem = cls._roi_artifact_stem(name)
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
    def _complete_stream_component_metadata(
        metadata: Mapping[str, Any] | None,
        *,
        filename: str,
        config: object,
    ) -> dict[str, Any]:
        """Fill missing display axes so viewer review can place a standalone ROI."""
        complete = dict(metadata or {})
        component_order = set(getattr(config, "COMPONENT_ORDER", ()))
        expected_components = component_order or {
            "well",
            *SINGLE_PLANE_COMPONENT_DEFAULTS,
        }

        if "well" in expected_components and complete.get("well") is None:
            complete["well"] = StreamingService._roi_artifact_stem(filename)

        for component, default in SINGLE_PLANE_COMPONENT_DEFAULTS.items():
            if component in expected_components and complete.get(component) is None:
                complete[component] = default

        return complete

    def _roi_component_metadata_by_path(
        self,
        paths: list[str],
        config: object,
    ) -> dict[str, dict[str, Any]]:
        parser = self.microscope_handler.parser
        metadata_by_path: dict[str, dict[str, Any]] = {}

        for path in paths:
            parsed: Mapping[str, Any] | None = None
            for candidate in self._source_filename_candidates_for_roi(path):
                parsed = parser.parse_filename(candidate)
                if parsed is not None:
                    break

            metadata_by_path[path] = self._complete_stream_component_metadata(
                parsed,
                filename=path,
                config=config,
            )

        return metadata_by_path

    @classmethod
    def supported_viewer_types(cls):
        """Return supported streaming config field keys.

        Centralized so UI can discover which viewer buttons to create instead
        of hardcoding Napari/Fiji in multiple places.
        """
        from openhcs.core.config import StreamingConfig

        # Return stable ordering from the registry keys
        return sorted(list(StreamingConfig.__registry__.keys()))

    @staticmethod
    def is_napari_viewer_type(viewer_type: str) -> bool:
        """Return whether a viewer identity or registry key targets napari."""

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
        backend_enum = context.config.backend
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

                    source = self._source_for_stream_paths(
                        file_paths,
                        "selected_images",
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
        backend_enum = context.config.backend
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

                source = self._source_for_stream_paths(paths, "selected_rois")
                metadata = StreamingMetadata.from_viewer_context(
                    viewer=context.viewer,
                    config=context.config,
                    microscope_handler=self.microscope_handler,
                    plate_path=context.plate_path,
                    source=source,
                ).to_backend_kwargs()
                metadata["component_metadata_by_path"] = (
                    self._roi_component_metadata_by_path(paths, context.config)
                )

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
