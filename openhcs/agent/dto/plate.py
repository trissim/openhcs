"""Plate inspection DTOs for the headless OpenHCS agent API."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from zmqruntime.config import TransportMode

from openhcs.agent.dto.common import (
    AgentError,
    AgentResultEnvelope,
    AgentWarning,
    JsonObject,
    JsonValue,
)
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.constants.constants import AllComponents
from openhcs.core.plate_file_inventory import (
    PlateFileInventoryQuery,
    PlateFileKind,
    PlateFileKindSelection,
)
from openhcs.core.streaming_config_declarations import ViewerType
from openhcs.core.synthetic_plate_generation import (
    SYNTHETIC_PLATE_GENERATION_PROFILE,
    SyntheticPlateFormat,
)


class PlateInspectionStatus(str, Enum):
    """Machine-readable plate inspection outcome."""

    OK = "ok"
    PARTIAL = "partial"
    ERROR = "error"


class PlateInspectionConfidence(str, Enum):
    """Confidence that the detected handler describes the supplied plate."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class PlateInspectionValueSource(str, Enum):
    """Source of one component value summary."""

    METADATA = "metadata"
    PARSED_FILENAMES = "parsed_filenames"
    METADATA_AND_PARSED_FILENAMES = "metadata_and_parsed_filenames"
    UNAVAILABLE = "unavailable"


class PlateInspectionIssueCode(str, Enum):
    """Structured issue codes returned by plate inspection."""

    PATH_POLICY_REJECTED = "plate_path_policy_rejected"
    PATH_NOT_DIRECTORY = "plate_path_not_directory"
    HANDLER_DETECTION_FAILED = "plate_handler_detection_failed"
    METADATA_FILE_UNAVAILABLE = "plate_metadata_file_unavailable"
    GRID_DIMENSIONS_UNAVAILABLE = "plate_grid_dimensions_unavailable"
    PIXEL_SIZE_UNAVAILABLE = "plate_pixel_size_unavailable"
    AVAILABLE_BACKENDS_UNAVAILABLE = "plate_available_backends_unavailable"
    IMAGE_FILE_LISTING_FAILED = "plate_image_file_listing_failed"
    RESULT_FILE_LISTING_FAILED = "plate_result_file_listing_failed"
    RESULT_FILES_AVAILABLE = "plate_result_files_available"
    NO_IMAGE_FILES = "plate_no_image_files"
    PARSER_UNAVAILABLE = "plate_parser_unavailable"
    PARSE_LIMIT_REACHED = "plate_parse_limit_reached"
    PARSE_FAILURES = "plate_parse_failures"
    LOW_PARSE_COVERAGE = "plate_low_parse_coverage"
    SOURCE_DIAGNOSTICS_UNAVAILABLE = "plate_source_diagnostics_unavailable"
    PROBABLE_NATIVE_HANDLER = "plate_probable_native_handler"


class SelectedPlateFileQueryTarget(str, Enum):
    """Plate root used when querying files from a selected PlateManager row."""

    SELECTED = "selected"
    OUTPUT = "output"
    SOURCE = "source"


class PlateWorkspacePreparationOperation(str, Enum):
    """Workspace operation names surfaced by read-only inspection."""

    NONE = "none"
    INITIALIZE_WORKSPACE = "initialize_workspace"


class PlateInspectionWorkflowScope(str, Enum):
    """Operational scope of a plate-inspection response."""

    DIAGNOSTIC = "diagnostic"


class PlateInspectionIngestionRoute(str, Enum):
    """Owner selected, or still needed, for source ingestion."""

    DETECTED_HANDLER = "detected_handler"
    SOURCE_BINDINGS_HANDLER = "source_bindings_handler"
    UNRESOLVED = "unresolved"


class PlateInspectionSourceBindingRole(str, Enum):
    """Role source bindings play relative to the ingestion owner."""

    SEMANTIC_SELECTION = "semantic_selection"
    INGESTION_OWNER = "ingestion_owner"
    NOT_PROJECTED_BY_HANDLER = "not_projected_by_handler"
    UNRESOLVED = "unresolved"


class PlateInspectionDefaults:
    """Default and bounded plate-inspection limits."""

    MICROSCOPE_AUTO = "auto"
    DEFAULT_MAX_SAMPLE_FILES = 20
    DEFAULT_MAX_COMPONENT_VALUES = 25
    DEFAULT_MAX_PARSE_FAILURE_SAMPLES = 10
    DEFAULT_MAX_FILES_TO_PARSE = 50_000
    DEFAULT_MAX_AUTO_RESOLUTION_SIZE = 1024
    MIN_BOUND = 0
    MAX_SAMPLE_FILES = 200
    MAX_COMPONENT_VALUES = 500
    MAX_PARSE_FAILURE_SAMPLES = 200
    MAX_FILES_TO_PARSE = 250_000


@dataclass(frozen=True, slots=True)
class SelectedPlateTargetOptions:
    """Shared target controls for PlateManager-selected plate operations."""

    microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO
    pattern_format: str | None = None
    target: SelectedPlateFileQueryTarget = SelectedPlateFileQueryTarget.SELECTED

    @staticmethod
    def target_from_value(value: str) -> SelectedPlateFileQueryTarget:
        return SelectedPlateFileQueryTarget(value)


@dataclass(frozen=True, slots=True)
class SelectedPlateFileFilterOptions(SelectedPlateTargetOptions):
    """Shared file filtering controls for selected-plate operations."""

    kind: PlateFileKind | None = PlateFileKind.IMAGE
    path_contains: str | None = None
    well: str | None = None
    limit: int = 1


@dataclass(frozen=True, slots=True)
class PlateInspectionBounds:
    """Payload bounds for potentially large plate folder inspections."""

    max_sample_files: int = PlateInspectionDefaults.DEFAULT_MAX_SAMPLE_FILES
    max_component_values: int = PlateInspectionDefaults.DEFAULT_MAX_COMPONENT_VALUES
    max_parse_failure_samples: int = (
        PlateInspectionDefaults.DEFAULT_MAX_PARSE_FAILURE_SAMPLES
    )
    max_files_to_parse: int = PlateInspectionDefaults.DEFAULT_MAX_FILES_TO_PARSE

    def normalized(self) -> "PlateInspectionBounds":
        return PlateInspectionBounds(
            max_sample_files=self._bounded(
                self.max_sample_files,
                PlateInspectionDefaults.MAX_SAMPLE_FILES,
            ),
            max_component_values=self._bounded(
                self.max_component_values,
                PlateInspectionDefaults.MAX_COMPONENT_VALUES,
            ),
            max_parse_failure_samples=self._bounded(
                self.max_parse_failure_samples,
                PlateInspectionDefaults.MAX_PARSE_FAILURE_SAMPLES,
            ),
            max_files_to_parse=self._bounded(
                self.max_files_to_parse,
                PlateInspectionDefaults.MAX_FILES_TO_PARSE,
            ),
        )

    def sample_strings(self, values: tuple[str, ...], limit: int) -> tuple[str, ...]:
        bounded_limit = max(int(limit), PlateInspectionDefaults.MIN_BOUND)
        return tuple(values[:bounded_limit])

    @staticmethod
    def _bounded(value: int, maximum: int) -> int:
        return min(max(int(value), PlateInspectionDefaults.MIN_BOUND), maximum)


@dataclass(frozen=True, slots=True)
class PlatePathInspectionRequest:
    """Read-only inspection request for a local plate folder."""

    plate_path: str
    microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO
    pattern_format: str | None = None
    bounds: PlateInspectionBounds = field(default_factory=PlateInspectionBounds)

    @classmethod
    def from_fields(
        cls,
        *,
        plate_path: str,
        microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO,
        pattern_format: str | None = None,
        max_sample_files: int = PlateInspectionDefaults.DEFAULT_MAX_SAMPLE_FILES,
        max_component_values: int = PlateInspectionDefaults.DEFAULT_MAX_COMPONENT_VALUES,
        max_parse_failure_samples: int = (
            PlateInspectionDefaults.DEFAULT_MAX_PARSE_FAILURE_SAMPLES
        ),
        max_files_to_parse: int = PlateInspectionDefaults.DEFAULT_MAX_FILES_TO_PARSE,
    ) -> "PlatePathInspectionRequest":
        return cls(
            plate_path=plate_path,
            microscope_type=microscope_type,
            pattern_format=pattern_format,
            bounds=PlateInspectionBounds(
                max_sample_files=max_sample_files,
                max_component_values=max_component_values,
                max_parse_failure_samples=max_parse_failure_samples,
                max_files_to_parse=max_files_to_parse,
            ),
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        return {
            "plate_path": self.plate_path,
            "microscope_type": self.microscope_type,
            "pattern_format": self.pattern_format,
            "max_sample_files": self.bounds.max_sample_files,
            "max_component_values": self.bounds.max_component_values,
            "max_parse_failure_samples": self.bounds.max_parse_failure_samples,
            "max_files_to_parse": self.bounds.max_files_to_parse,
        }


@dataclass(frozen=True, slots=True)
class PlateImageSampleRequest:
    """Request a bounded pixel sample from an image exposed by a plate."""

    plate_path: str
    image_path: str
    microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO
    pattern_format: str | None = None
    y: int = 0
    x: int = 0
    height: int = 32
    width: int = 32
    resolution_index: int | None = None
    max_auto_resolution_size: int = (
        PlateInspectionDefaults.DEFAULT_MAX_AUTO_RESOLUTION_SIZE
    )
    include_array_values: bool = True
    max_array_elements: int = 4096

    @classmethod
    def from_fields(
        cls,
        *,
        plate_path: str,
        image_path: str,
        microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO,
        pattern_format: str | None = None,
        y: int = 0,
        x: int = 0,
        height: int = 32,
        width: int = 32,
        resolution_index: int | None = None,
        max_auto_resolution_size: int = (
            PlateInspectionDefaults.DEFAULT_MAX_AUTO_RESOLUTION_SIZE
        ),
        include_array_values: bool = True,
        max_array_elements: int = 4096,
    ) -> "PlateImageSampleRequest":
        return cls(
            plate_path=plate_path,
            image_path=image_path,
            microscope_type=microscope_type,
            pattern_format=pattern_format,
            y=y,
            x=x,
            height=height,
            width=width,
            resolution_index=resolution_index,
            max_auto_resolution_size=max_auto_resolution_size,
            include_array_values=include_array_values,
            max_array_elements=max_array_elements,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        return {
            "plate_path": self.plate_path,
            "image_path": self.image_path,
            "microscope_type": self.microscope_type,
            "pattern_format": self.pattern_format,
            "y": self.y,
            "x": self.x,
            "height": self.height,
            "width": self.width,
            "resolution_index": self.resolution_index,
            "max_auto_resolution_size": self.max_auto_resolution_size,
            "include_array_values": self.include_array_values,
            "max_array_elements": self.max_array_elements,
        }


@dataclass(frozen=True, slots=True)
class PlateFileQueryRequest:
    """Query image/result files exposed by a local plate inventory."""

    plate_path: str
    microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO
    pattern_format: str | None = None
    kind: PlateFileKind | None = PlateFileKind.IMAGE
    path_contains: str | None = None
    well: str | None = None
    offset: int = 0
    limit: int = 50
    include_previews: bool = True
    max_preview_lines: int = 8
    max_preview_bytes: int = 64 * 1024

    @classmethod
    def from_fields(
        cls,
        *,
        plate_path: str,
        microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO,
        pattern_format: str | None = None,
        kind: PlateFileKindSelection = PlateFileKind.IMAGE,
        path_contains: str | None = None,
        well: str | None = None,
        offset: int = 0,
        limit: int = 50,
        include_previews: bool = True,
        max_preview_lines: int = 8,
        max_preview_bytes: int = 64 * 1024,
    ) -> "PlateFileQueryRequest":
        return cls(
            plate_path=plate_path,
            microscope_type=microscope_type,
            pattern_format=pattern_format,
            kind=PlateFileInventoryQuery.kind_from_value(kind),
            path_contains=path_contains,
            well=well,
            offset=offset,
            limit=limit,
            include_previews=include_previews,
            max_preview_lines=max_preview_lines,
            max_preview_bytes=max_preview_bytes,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        return {
            "plate_path": self.plate_path,
            "microscope_type": self.microscope_type,
            "pattern_format": self.pattern_format,
            "kind": PlateFileInventoryQuery.kind_value(self.kind),
            "path_contains": self.path_contains,
            "well": self.well,
            "offset": self.offset,
            "limit": self.limit,
            "include_previews": self.include_previews,
            "max_preview_lines": self.max_preview_lines,
            "max_preview_bytes": self.max_preview_bytes,
        }


@dataclass(frozen=True, slots=True)
class PlateFileStreamRequest:
    """Stream image or ROI files exposed by a local plate inventory to a viewer."""

    plate_path: str
    context_plate_path: str | None = None
    file_paths: tuple[str, ...] = ()
    microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO
    pattern_format: str | None = None
    kind: PlateFileKind | None = PlateFileKind.IMAGE
    path_contains: str | None = None
    well: str | None = None
    limit: int = 1
    viewer_config_key: str = ViewerType.NAPARI.config_key
    connection: ExecutionConnectionSpec = field(default_factory=ExecutionConnectionSpec)
    fresh_viewer: bool = False

    @classmethod
    def from_fields(
        cls,
        *,
        plate_path: str,
        file_paths: list[str] | None = None,
        microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO,
        pattern_format: str | None = None,
        kind: PlateFileKindSelection = PlateFileKind.IMAGE,
        path_contains: str | None = None,
        well: str | None = None,
        limit: int = 1,
        viewer_config_key: str = ViewerType.NAPARI.config_key,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: TransportMode | None = None,
        persistent: bool = True,
        fresh_viewer: bool = False,
    ) -> "PlateFileStreamRequest":
        return cls(
            plate_path=plate_path,
            file_paths=tuple(file_paths or ()),
            microscope_type=microscope_type,
            pattern_format=pattern_format,
            kind=PlateFileInventoryQuery.kind_from_value(kind),
            path_contains=path_contains,
            well=well,
            limit=limit,
            viewer_config_key=viewer_config_key,
            connection=ExecutionConnectionSpec(
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
            ),
            fresh_viewer=fresh_viewer,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        return {
            "plate_path": self.plate_path,
            "file_paths": list(self.file_paths) if self.file_paths else None,
            "microscope_type": self.microscope_type,
            "pattern_format": self.pattern_format,
            "kind": PlateFileInventoryQuery.kind_value(self.kind),
            "path_contains": self.path_contains,
            "well": self.well,
            "limit": self.limit,
            "viewer_config_key": self.viewer_config_key,
            **self.connection.tool_arguments(),
            "fresh_viewer": self.fresh_viewer,
        }


@dataclass(frozen=True, slots=True)
class SelectedPlateImageInspectionRequest(SelectedPlateTargetOptions):
    """Inspect images for the plate currently selected in the UI."""

    max_sample_files: int = PlateInspectionDefaults.DEFAULT_MAX_SAMPLE_FILES
    max_component_values: int = PlateInspectionDefaults.DEFAULT_MAX_COMPONENT_VALUES
    max_parse_failure_samples: int = (
        PlateInspectionDefaults.DEFAULT_MAX_PARSE_FAILURE_SAMPLES
    )
    max_files_to_parse: int = PlateInspectionDefaults.DEFAULT_MAX_FILES_TO_PARSE

    @classmethod
    def from_fields(
        cls,
        *,
        microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO,
        pattern_format: str | None = None,
        target: str = SelectedPlateFileQueryTarget.SELECTED.value,
        max_sample_files: int = PlateInspectionDefaults.DEFAULT_MAX_SAMPLE_FILES,
        max_component_values: int = PlateInspectionDefaults.DEFAULT_MAX_COMPONENT_VALUES,
        max_parse_failure_samples: int = (
            PlateInspectionDefaults.DEFAULT_MAX_PARSE_FAILURE_SAMPLES
        ),
        max_files_to_parse: int = PlateInspectionDefaults.DEFAULT_MAX_FILES_TO_PARSE,
    ) -> "SelectedPlateImageInspectionRequest":
        return cls(
            microscope_type=microscope_type,
            pattern_format=pattern_format,
            target=cls.target_from_value(target),
            max_sample_files=max_sample_files,
            max_component_values=max_component_values,
            max_parse_failure_samples=max_parse_failure_samples,
            max_files_to_parse=max_files_to_parse,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        return {
            "microscope_type": self.microscope_type,
            "pattern_format": self.pattern_format,
            "target": self.target.value,
            "max_sample_files": self.max_sample_files,
            "max_component_values": self.max_component_values,
            "max_parse_failure_samples": self.max_parse_failure_samples,
            "max_files_to_parse": self.max_files_to_parse,
        }

    def to_plate_path_inspection_request(
        self,
        *,
        plate_path: str,
        microscope_type: str,
    ) -> PlatePathInspectionRequest:
        return PlatePathInspectionRequest.from_fields(
            plate_path=plate_path,
            microscope_type=microscope_type,
            pattern_format=self.pattern_format,
            max_sample_files=self.max_sample_files,
            max_component_values=self.max_component_values,
            max_parse_failure_samples=self.max_parse_failure_samples,
            max_files_to_parse=self.max_files_to_parse,
        )


@dataclass(frozen=True, slots=True)
class SelectedPlateFileQueryRequest(SelectedPlateFileFilterOptions):
    """Query files for the plate currently selected in the UI."""

    offset: int = 0
    limit: int = 50
    include_previews: bool = True
    max_preview_lines: int = 8
    max_preview_bytes: int = 64 * 1024

    @classmethod
    def from_fields(
        cls,
        *,
        microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO,
        pattern_format: str | None = None,
        kind: PlateFileKindSelection = PlateFileKind.IMAGE,
        target: str = SelectedPlateFileQueryTarget.SELECTED.value,
        path_contains: str | None = None,
        well: str | None = None,
        offset: int = 0,
        limit: int = 50,
        include_previews: bool = True,
        max_preview_lines: int = 8,
        max_preview_bytes: int = 64 * 1024,
    ) -> "SelectedPlateFileQueryRequest":
        return cls(
            microscope_type=microscope_type,
            pattern_format=pattern_format,
            kind=PlateFileInventoryQuery.kind_from_value(kind),
            target=cls.target_from_value(target),
            path_contains=path_contains,
            well=well,
            offset=offset,
            limit=limit,
            include_previews=include_previews,
            max_preview_lines=max_preview_lines,
            max_preview_bytes=max_preview_bytes,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        return {
            "microscope_type": self.microscope_type,
            "pattern_format": self.pattern_format,
            "kind": PlateFileInventoryQuery.kind_value(self.kind),
            "target": self.target.value,
            "path_contains": self.path_contains,
            "well": self.well,
            "offset": self.offset,
            "limit": self.limit,
            "include_previews": self.include_previews,
            "max_preview_lines": self.max_preview_lines,
            "max_preview_bytes": self.max_preview_bytes,
        }

    def to_plate_file_query_request(
        self,
        *,
        plate_path: str,
        microscope_type: str,
    ) -> PlateFileQueryRequest:
        return PlateFileQueryRequest(
            plate_path=plate_path,
            microscope_type=microscope_type,
            pattern_format=self.pattern_format,
            kind=self.kind,
            path_contains=self.path_contains,
            well=self.well,
            offset=self.offset,
            limit=self.limit,
            include_previews=self.include_previews,
            max_preview_lines=self.max_preview_lines,
            max_preview_bytes=self.max_preview_bytes,
        )


@dataclass(frozen=True, slots=True)
class SelectedPlateImageSampleRequest(SelectedPlateTargetOptions):
    """Sample one image from the plate currently selected in the UI."""

    image_path: str | None = None
    y: int = 0
    x: int = 0
    height: int = 32
    width: int = 32
    resolution_index: int | None = None
    max_auto_resolution_size: int = (
        PlateInspectionDefaults.DEFAULT_MAX_AUTO_RESOLUTION_SIZE
    )
    include_array_values: bool = True
    max_array_elements: int = 4096

    @classmethod
    def from_fields(
        cls,
        *,
        image_path: str | None = None,
        microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO,
        pattern_format: str | None = None,
        target: str = SelectedPlateFileQueryTarget.SELECTED.value,
        y: int = 0,
        x: int = 0,
        height: int = 32,
        width: int = 32,
        resolution_index: int | None = None,
        max_auto_resolution_size: int = (
            PlateInspectionDefaults.DEFAULT_MAX_AUTO_RESOLUTION_SIZE
        ),
        include_array_values: bool = True,
        max_array_elements: int = 4096,
    ) -> "SelectedPlateImageSampleRequest":
        return cls(
            image_path=image_path,
            microscope_type=microscope_type,
            pattern_format=pattern_format,
            target=cls.target_from_value(target),
            y=y,
            x=x,
            height=height,
            width=width,
            resolution_index=resolution_index,
            max_auto_resolution_size=max_auto_resolution_size,
            include_array_values=include_array_values,
            max_array_elements=max_array_elements,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        return {
            "image_path": self.image_path,
            "microscope_type": self.microscope_type,
            "pattern_format": self.pattern_format,
            "target": self.target.value,
            "y": self.y,
            "x": self.x,
            "height": self.height,
            "width": self.width,
            "resolution_index": self.resolution_index,
            "max_auto_resolution_size": self.max_auto_resolution_size,
            "include_array_values": self.include_array_values,
            "max_array_elements": self.max_array_elements,
        }

    def to_plate_image_sample_request(
        self,
        *,
        plate_path: str,
        image_path: str,
        microscope_type: str,
    ) -> PlateImageSampleRequest:
        return PlateImageSampleRequest(
            plate_path=plate_path,
            image_path=image_path,
            microscope_type=microscope_type,
            pattern_format=self.pattern_format,
            y=self.y,
            x=self.x,
            height=self.height,
            width=self.width,
            resolution_index=self.resolution_index,
            max_auto_resolution_size=self.max_auto_resolution_size,
            include_array_values=self.include_array_values,
            max_array_elements=self.max_array_elements,
        )


@dataclass(frozen=True, slots=True)
class SelectedPlateFileStreamRequest(SelectedPlateFileFilterOptions):
    """Stream files from the plate currently selected in the UI."""

    file_paths: tuple[str, ...] = ()
    viewer_config_key: str = ViewerType.NAPARI.config_key
    connection: ExecutionConnectionSpec = field(default_factory=ExecutionConnectionSpec)
    fresh_viewer: bool = False

    @classmethod
    def from_fields(
        cls,
        *,
        file_paths: list[str] | None = None,
        microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO,
        pattern_format: str | None = None,
        kind: PlateFileKindSelection = PlateFileKind.IMAGE,
        target: str = SelectedPlateFileQueryTarget.SELECTED.value,
        path_contains: str | None = None,
        well: str | None = None,
        limit: int = 1,
        viewer_config_key: str = ViewerType.NAPARI.config_key,
        host: str = "localhost",
        port: int | None = None,
        transport_mode: TransportMode | None = None,
        persistent: bool = True,
        fresh_viewer: bool = False,
    ) -> "SelectedPlateFileStreamRequest":
        return cls(
            file_paths=tuple(file_paths or ()),
            microscope_type=microscope_type,
            pattern_format=pattern_format,
            kind=PlateFileInventoryQuery.kind_from_value(kind),
            target=cls.target_from_value(target),
            path_contains=path_contains,
            well=well,
            limit=limit,
            viewer_config_key=viewer_config_key,
            connection=ExecutionConnectionSpec(
                host=host,
                port=port,
                transport_mode=transport_mode,
                persistent=persistent,
            ),
            fresh_viewer=fresh_viewer,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        return {
            "file_paths": list(self.file_paths) if self.file_paths else None,
            "microscope_type": self.microscope_type,
            "pattern_format": self.pattern_format,
            "kind": PlateFileInventoryQuery.kind_value(self.kind),
            "target": self.target.value,
            "path_contains": self.path_contains,
            "well": self.well,
            "limit": self.limit,
            "viewer_config_key": self.viewer_config_key,
            **self.connection.tool_arguments(),
            "fresh_viewer": self.fresh_viewer,
        }

    def to_plate_file_stream_request(
        self,
        *,
        plate_path: str,
        context_plate_path: str | None,
        microscope_type: str,
    ) -> PlateFileStreamRequest:
        return PlateFileStreamRequest(
            plate_path=plate_path,
            context_plate_path=context_plate_path,
            file_paths=self.file_paths,
            microscope_type=microscope_type,
            pattern_format=self.pattern_format,
            kind=self.kind,
            path_contains=self.path_contains,
            well=self.well,
            limit=self.limit,
            viewer_config_key=self.viewer_config_key,
            connection=self.connection,
            fresh_viewer=self.fresh_viewer,
        )


@dataclass(frozen=True, slots=True)
class SyntheticPlateGenerationRequest:
    """Request a bounded synthetic microscopy plate for MCP-driven workflows."""

    output_dir: str
    grid_rows: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.grid_rows
    grid_cols: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.grid_cols
    tile_width: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.tile_width
    tile_height: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.tile_height
    overlap_percent: int = (
        SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.overlap_percent
    )
    stage_error_px: int = (
        SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.stage_error_px
    )
    wavelengths: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.wavelengths
    z_stack_levels: int = (
        SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.z_stack_levels
    )
    num_cells: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.num_cells
    shared_cell_fraction: float = (
        SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.shared_cell_fraction
    )
    wells: tuple[str, ...] = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.wells
    format: SyntheticPlateFormat = (
        SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.format
    )
    openhcs_format: bool = (
        SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.openhcs_format
    )
    include_all_components: bool = (
        SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.include_all_components
    )
    random_seed: int | None = (
        SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.random_seed
    )
    sample_file_limit: int = (
        SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.sample_file_limit
    )

    @classmethod
    def from_fields(
        cls,
        *,
        output_dir: str,
        grid_rows: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.grid_rows,
        grid_cols: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.grid_cols,
        tile_width: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.tile_width,
        tile_height: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.tile_height,
        overlap_percent: int = (
            SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.overlap_percent
        ),
        stage_error_px: int = (
            SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.stage_error_px
        ),
        wavelengths: int = (
            SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.wavelengths
        ),
        z_stack_levels: int = (
            SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.z_stack_levels
        ),
        num_cells: int = SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.num_cells,
        shared_cell_fraction: float = (
            SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.shared_cell_fraction
        ),
        wells: list[str] | None = None,
        format: str = (SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.format.value),
        openhcs_format: bool = (
            SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.openhcs_format
        ),
        include_all_components: bool = (
            SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.include_all_components
        ),
        random_seed: int | None = (
            SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.random_seed
        ),
        sample_file_limit: int = (
            SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.sample_file_limit
        ),
    ) -> "SyntheticPlateGenerationRequest":
        return cls(
            output_dir=output_dir,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            tile_width=tile_width,
            tile_height=tile_height,
            overlap_percent=overlap_percent,
            stage_error_px=stage_error_px,
            wavelengths=wavelengths,
            z_stack_levels=z_stack_levels,
            num_cells=num_cells,
            shared_cell_fraction=shared_cell_fraction,
            wells=(
                tuple(wells)
                if wells is not None
                else SYNTHETIC_PLATE_GENERATION_PROFILE.default_request.wells
            ),
            format=SyntheticPlateFormat(format),
            openhcs_format=openhcs_format,
            include_all_components=include_all_components,
            random_seed=random_seed,
            sample_file_limit=sample_file_limit,
        )

    def as_tool_arguments(self) -> dict[str, JsonValue]:
        return {
            "output_dir": self.output_dir,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "overlap_percent": self.overlap_percent,
            "stage_error_px": self.stage_error_px,
            "wavelengths": self.wavelengths,
            "z_stack_levels": self.z_stack_levels,
            "num_cells": self.num_cells,
            "shared_cell_fraction": self.shared_cell_fraction,
            "wells": list(self.wells),
            "format": self.format.value,
            "openhcs_format": self.openhcs_format,
            "include_all_components": self.include_all_components,
            "random_seed": self.random_seed,
            "sample_file_limit": self.sample_file_limit,
        }


@dataclass(frozen=True, slots=True)
class PlateInspectionComponentValue:
    """One component value observed in metadata, filenames, or both."""

    key: str
    label: str | None = None
    declared_in_metadata: bool = False
    observed_in_filenames: bool = False


@dataclass(frozen=True, slots=True)
class PlateInspectionComponentSummary:
    """Bounded values for one HCS component dimension."""

    component: AllComponents
    source: PlateInspectionValueSource
    count: int
    values: tuple[PlateInspectionComponentValue, ...] = ()
    truncated_value_count: int = 0


@dataclass(frozen=True, slots=True)
class PlateInspectionImageRecordSummary:
    """One bounded image record exposed by plate inventory inspection."""

    virtual_path: str
    full_virtual_path: str
    source_path: str
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlateInspectionImageFileSummary:
    """Bounded image-file listing summary."""

    count: int = 0
    sampled_files: tuple[str, ...] = ()
    sampled_records: tuple[PlateInspectionImageRecordSummary, ...] = ()
    truncated_file_count: int = 0


@dataclass(frozen=True, slots=True)
class PlateInspectionResultFileRecordSummary:
    """One bounded analysis artifact record exposed by plate inspection."""

    relative_path: str
    full_path: str
    file_format: str
    metadata: JsonObject = field(default_factory=dict)
    preview: "PlateInspectionResultFilePreview | None" = None


@dataclass(frozen=True, slots=True)
class PlateInspectionResultFilePreview:
    """Bounded preview for one analysis artifact."""

    text_lines: tuple[str, ...] = ()
    csv_columns: tuple[str, ...] = ()
    csv_rows: tuple[JsonObject, ...] = ()
    roi_count: int | None = None
    roi_member_count: int | None = None
    roi_duplicate_member_count: int | None = None
    roi_area_min: float | None = None
    roi_area_max: float | None = None
    roi_area_mean: float | None = None
    roi_examples: tuple[JsonObject, ...] = ()
    truncated: bool = False
    omitted_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PlateFileQueryRecordSummary:
    """One unified file record returned by a plate file query."""

    kind: PlateFileKind
    key: str
    metadata: JsonObject = field(default_factory=dict)
    virtual_path: str | None = None
    full_virtual_path: str | None = None
    source_path: str | None = None
    relative_path: str | None = None
    full_path: str | None = None
    file_format: str | None = None
    preview: "PlateInspectionResultFilePreview | None" = None


@dataclass(frozen=True, kw_only=True, slots=True)
class PlateFileQueryResult(AgentResultEnvelope):
    """Bounded plate file query result."""

    plate_path: str
    requested_microscope_type: str
    detected_microscope_type: str | None = None
    handler_class: str | None = None
    parser_class: str | None = None
    total_count: int = 0
    returned_count: int = 0
    offset: int = 0
    limit: int = 0
    truncated_count: int = 0
    records: tuple[PlateFileQueryRecordSummary, ...] = ()
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, kw_only=True, slots=True)
class PlateFileStreamResult(AgentResultEnvelope):
    """Result from streaming plate inventory files to a live viewer."""

    plate_path: str
    requested_microscope_type: str
    detected_microscope_type: str | None = None
    handler_class: str | None = None
    parser_class: str | None = None
    viewer_config_key: str = ""
    viewer_type: ViewerType | None = None
    connection: ExecutionConnectionSpec = field(default_factory=ExecutionConnectionSpec)
    requested_paths: tuple[str, ...] = ()
    resolved_records: tuple[PlateFileQueryRecordSummary, ...] = ()
    streamed_image_paths: tuple[str, ...] = ()
    streamed_roi_paths: tuple[str, ...] = ()
    skipped_records: tuple[PlateFileQueryRecordSummary, ...] = ()
    status_messages: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True, slots=True)
class SyntheticPlateGenerationResult(AgentResultEnvelope):
    """Result from generating a bounded synthetic microscopy plate."""

    output_dir: str
    requested_format: SyntheticPlateFormat
    grid_size: tuple[int, int] = ()
    tile_size: tuple[int, int] = ()
    overlap_percent: int = 0
    stage_error_px: int = 0
    wells: tuple[str, ...] = ()
    wavelengths: int = 0
    z_stack_levels: int = 0
    num_cells: int = 0
    shared_cell_fraction: float = 0.0
    image_count: int = 0
    sampled_image_files: tuple[str, ...] = ()
    truncated_image_count: int = 0
    metadata_file_path: str | None = None
    detected_microscope_type: str | None = None
    handler_class: str | None = None
    include_all_components: bool = True


@dataclass(frozen=True, slots=True)
class PlateInspectionResultFileSummary:
    """Bounded result-artifact listing summary."""

    count: int = 0
    scanned_file_count: int = 0
    sampled_files: tuple[str, ...] = ()
    sampled_records: tuple[PlateInspectionResultFileRecordSummary, ...] = ()
    truncated_file_count: int = 0


@dataclass(frozen=True, slots=True)
class PlateInspectionParseFailure:
    """One filename that could not be parsed by the selected handler parser."""

    filename: str
    reason: str


@dataclass(frozen=True, slots=True)
class PlateInspectionParseSummary:
    """Filename parse coverage for the inspected image files."""

    attempted_file_count: int = 0
    skipped_file_count: int = 0
    parsed_file_count: int = 0
    failed_file_count: int = 0
    failure_samples: tuple[PlateInspectionParseFailure, ...] = ()
    truncated_failure_count: int = 0


@dataclass(frozen=True, slots=True)
class PlateInspectionWorkspacePreparation:
    """Whether execution still needs a mutating workspace-preparation step."""

    read_only_inspection: bool = True
    required_before_execution: bool = False
    operation: PlateWorkspacePreparationOperation = (
        PlateWorkspacePreparationOperation.NONE
    )
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PlateInspectionHandlerCandidate:
    """Format-specific handler evidence recovered from authoritative parsers."""

    microscope_type: str
    handler_class: str
    parser_class: str
    root_dir: str
    tested_file_count: int
    recognized_file_count: int
    recognizes_all_tested_files: bool
    files_under_expected_root: bool
    metadata_detected: bool
    metadata_file_path: str | None = None
    metadata_diagnostic: str | None = None


@dataclass(frozen=True, slots=True)
class PlateInspectionWorkflowAdvice:
    """Structured routing advice kept separate from decoder evidence."""

    workflow_scope: PlateInspectionWorkflowScope = (
        PlateInspectionWorkflowScope.DIAGNOSTIC
    )
    ingestion_route: PlateInspectionIngestionRoute = (
        PlateInspectionIngestionRoute.UNRESOLVED
    )
    ingestion_owner: str | None = None
    source_binding_role: PlateInspectionSourceBindingRole = (
        PlateInspectionSourceBindingRole.UNRESOLVED
    )
    ui_code_document_id: str | None = None
    ui_operation: str | None = None
    knowledge_query: str = "source model image sources"
    probable_native_ingestion_owners: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True, kw_only=True, slots=True)
class PlatePathInspectionResult(AgentResultEnvelope):
    """Read-only plate-folder inspection result."""

    plate_path: str
    requested_microscope_type: str
    status: PlateInspectionStatus = PlateInspectionStatus.ERROR
    confidence: PlateInspectionConfidence = PlateInspectionConfidence.NONE
    available_microscope_types: tuple[str, ...] = ()
    detected_microscope_type: str | None = None
    handler_class: str | None = None
    parser_class: str | None = None
    metadata_handler_class: str | None = None
    root_dir: str | None = None
    compatible_backends: tuple[str, ...] = ()
    available_backends: tuple[str, ...] = ()
    metadata_file_path: str | None = None
    grid_dimensions: tuple[int, int] | None = None
    pixel_size: float | None = None
    image_files: PlateInspectionImageFileSummary = field(
        default_factory=PlateInspectionImageFileSummary
    )
    result_files: PlateInspectionResultFileSummary = field(
        default_factory=PlateInspectionResultFileSummary
    )
    parse_summary: PlateInspectionParseSummary = field(
        default_factory=PlateInspectionParseSummary
    )
    components: tuple[PlateInspectionComponentSummary, ...] = ()
    source_diagnostics: tuple[JsonObject, ...] = ()
    format_specific_handler_candidates: tuple[PlateInspectionHandlerCandidate, ...] = ()
    workspace_preparation: PlateInspectionWorkspacePreparation = field(
        default_factory=PlateInspectionWorkspacePreparation
    )
    workflow_advice: PlateInspectionWorkflowAdvice = field(
        default_factory=PlateInspectionWorkflowAdvice
    )
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, kw_only=True, slots=True)
class PlateImageSampleResult(AgentResultEnvelope):
    """Bounded image sample resolved through plate virtual-workspace metadata."""

    plate_path: str
    requested_image_path: str
    virtual_path: str | None = None
    full_virtual_path: str | None = None
    source_path: str | None = None
    source_metadata: JsonObject = field(default_factory=dict)
    shape: tuple[int, ...] = ()
    resolution_shape: tuple[int, ...] = ()
    dtype: str | None = None
    minimum: JsonValue = None
    maximum: JsonValue = None
    mean: float | None = None
    requested_resolution_index: int | None = None
    selected_resolution_index: int | None = None
    resolution_count: int | None = None
    downsample_yx: tuple[float, float] | None = None
    statistics_scope: str | None = None
    sample_origin_yx: tuple[int, int] = (0, 0)
    sample_shape: tuple[int, ...] = ()
    sample_included: bool = False
    sample_values: JsonValue = ()
    sample_omitted_reason: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class SelectedPlateImageInspectionResult(AgentResultEnvelope):
    """Image inspection for the single plate currently selected in the UI."""

    selected_plate: JsonObject = field(default_factory=dict)
    target: SelectedPlateFileQueryTarget = SelectedPlateFileQueryTarget.SELECTED
    inspection: PlatePathInspectionResult | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class SelectedPlateFileQueryResult(AgentResultEnvelope):
    """File query for the single plate currently selected in the UI."""

    selected_plate: JsonObject = field(default_factory=dict)
    target: SelectedPlateFileQueryTarget = SelectedPlateFileQueryTarget.SELECTED
    query: PlateFileQueryResult | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class SelectedPlateImageSampleResult(AgentResultEnvelope):
    """Bounded image sample for the single plate currently selected in the UI."""

    selected_plate: JsonObject = field(default_factory=dict)
    target: SelectedPlateFileQueryTarget = SelectedPlateFileQueryTarget.SELECTED
    image_path: str | None = None
    auto_selected_image_path: bool = False
    sample: PlateImageSampleResult | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class SelectedPlateFileStreamResult(AgentResultEnvelope):
    """Streaming result for the single plate currently selected in the UI."""

    selected_plate: JsonObject = field(default_factory=dict)
    target: SelectedPlateFileQueryTarget = SelectedPlateFileQueryTarget.SELECTED
    stream: PlateFileStreamResult | None = None
