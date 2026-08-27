#!/usr/bin/env python3
"""Capture real application windows and derive validated web gallery media.

The tool never changes pixels to fabricate UI state. It can capture a real
window, crop a source capture, trim a recording, transcode it, and validate the
derived media against one typed JSON manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

from python_introspect import dataclass_from_mapping

from openhcs.serialization.json import to_jsonable
from scripts.gallery_catalog import (
    GalleryCatalogError,
    GalleryScenarioCatalogRecord,
    GallerySourceCaptureRequest,
    GallerySourceCaptureResult,
    OpenHCSGalleryScenarioCatalog,
    UiBridgeWindowCaptureTarget,
    gallery_scenarios,
)

CAPTURE_MANIFEST_SCHEMA_VERSION = 2
CAPTURE_TOOL_REPORT_SCHEMA_VERSION = 2
MAX_DERIVED_MOTION_SECONDS = 30.0
MAX_RAW_RECORDING_SECONDS = 180.0
OUTPUT_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*\.(?:gif|mp4|webm|webp)$")
WINDOW_ID_PATTERN = re.compile(r"^(?:0x[0-9a-fA-F]+|[1-9][0-9]*)$")
DISPLAY_PATTERN = re.compile(r"^(?:[A-Za-z0-9_.-]+)?:\d+(?:\.\d+)?$")


class MediaGalleryError(RuntimeError):
    """Raised when capture input or generated media violates the contract."""


class MediaCategoryPolicyABC(ABC):
    """Nominal owner of source/output rules for one media category leaf."""

    def validate_source(self, record: CaptureRecord) -> None:
        """Validate source-level rules, then every declared derivative."""

        self.validate_source_bounds(record)
        for output in record.outputs:
            self.validate_output(
                output.encoding.value.category.value,
                record,
                output,
            )

    @abstractmethod
    def validate_source_bounds(self, record: CaptureRecord) -> None:
        """Validate source fields owned by this category."""

    @abstractmethod
    def validate_output(
        self,
        output_policy: MediaCategoryPolicyABC,
        record: CaptureRecord,
        output: Derivative,
    ) -> None:
        """Select the source-specific output rule without caller dispatch."""

    @abstractmethod
    def validate_for_still_source(
        self,
        record: CaptureRecord,
        output: Derivative,
    ) -> None:
        """Validate this output category for a still source."""

    @abstractmethod
    def validate_for_motion_source(
        self,
        record: CaptureRecord,
        output: Derivative,
    ) -> None:
        """Validate this output category for a motion source."""

    @abstractmethod
    def transcode_duration_arguments(
        self,
        record: CaptureRecord,
    ) -> tuple[str, ...]:
        """Return output-category-specific FFmpeg duration arguments."""

    @abstractmethod
    def validate_derivative_duration(
        self,
        record: CaptureRecord,
        output: Derivative,
        probe: MediaProbe,
    ) -> None:
        """Validate output-category-specific duration evidence."""


class StillMediaCategoryPolicy(MediaCategoryPolicyABC):
    """Validation semantics for still sources and still derivatives."""

    def validate_source_bounds(self, record: CaptureRecord) -> None:
        if record.trim is not None:
            raise MediaGalleryError("Still captures cannot declare a trim interval.")

    def validate_output(
        self,
        output_policy: MediaCategoryPolicyABC,
        record: CaptureRecord,
        output: Derivative,
    ) -> None:
        output_policy.validate_for_still_source(record, output)

    def validate_for_still_source(
        self,
        record: CaptureRecord,
        output: Derivative,
    ) -> None:
        del record
        if output.fps is not None or output.frame_at_seconds is not None:
            raise MediaGalleryError(
                "Still derivatives cannot declare fps or frame_at_seconds."
            )

    def validate_for_motion_source(
        self,
        record: CaptureRecord,
        output: Derivative,
    ) -> None:
        if output.fps is not None or output.frame_at_seconds is None:
            raise MediaGalleryError(
                f"Poster output {output.filename} must declare "
                "frame_at_seconds and no fps."
            )
        assert record.trim is not None
        if output.frame_at_seconds > record.trim.duration_seconds:
            raise MediaGalleryError(
                f"Poster frame for {output.filename} falls outside the trim."
            )

    def transcode_duration_arguments(
        self,
        record: CaptureRecord,
    ) -> tuple[str, ...]:
        del record
        return ()

    def validate_derivative_duration(
        self,
        record: CaptureRecord,
        output: Derivative,
        probe: MediaProbe,
    ) -> None:
        del record, output, probe


class MotionMediaCategoryPolicy(MediaCategoryPolicyABC):
    """Validation semantics for motion sources and motion derivatives."""

    def validate_source_bounds(self, record: CaptureRecord) -> None:
        if record.trim is None:
            raise MediaGalleryError(
                "Motion captures must declare an explicit bounded trim interval."
            )

    def validate_output(
        self,
        output_policy: MediaCategoryPolicyABC,
        record: CaptureRecord,
        output: Derivative,
    ) -> None:
        output_policy.validate_for_motion_source(record, output)

    def validate_for_still_source(
        self,
        record: CaptureRecord,
        output: Derivative,
    ) -> None:
        del record, output
        raise MediaGalleryError(
            "Still captures can only produce still WebP derivatives."
        )

    def validate_for_motion_source(
        self,
        record: CaptureRecord,
        output: Derivative,
    ) -> None:
        del record
        if output.fps is None:
            raise MediaGalleryError(
                f"Motion output {output.filename} must declare fps."
            )
        if output.frame_at_seconds is not None:
            raise MediaGalleryError("Motion outputs cannot declare frame_at_seconds.")

    def transcode_duration_arguments(
        self,
        record: CaptureRecord,
    ) -> tuple[str, ...]:
        assert record.trim is not None
        return ("-t", _format_seconds(record.trim.duration_seconds))

    def validate_derivative_duration(
        self,
        record: CaptureRecord,
        output: Derivative,
        probe: MediaProbe,
    ) -> None:
        if probe.duration_seconds is None:
            raise MediaGalleryError(
                f"Motion derivative {output.filename} has no duration."
            )
        assert record.trim is not None
        tolerance = max(0.1, 2.0 / (output.fps or 1))
        if abs(probe.duration_seconds - record.trim.duration_seconds) > tolerance:
            raise MediaGalleryError(
                f"{output.filename} duration is {probe.duration_seconds:g}s; "
                f"expected {record.trim.duration_seconds:g}s within "
                f"{tolerance:g}s."
            )


class MediaCategory(Enum):
    """Whether media is still or time-varying, including its leaf policy."""

    STILL = StillMediaCategoryPolicy()
    MOTION = MotionMediaCategoryPolicy()

    def validate_source(self, record: CaptureRecord) -> None:
        """Run this member's source and derivative validation policy."""

        self.value.validate_source(record)

    def transcode_duration_arguments(
        self,
        record: CaptureRecord,
    ) -> tuple[str, ...]:
        """Return this member's FFmpeg duration arguments."""

        return self.value.transcode_duration_arguments(record)

    def validate_derivative_duration(
        self,
        record: CaptureRecord,
        output: Derivative,
        probe: MediaProbe,
    ) -> None:
        """Validate duration evidence through this member's policy."""

        self.value.validate_derivative_duration(record, output, probe)


@dataclass(frozen=True)
class HostToolDeclaration:
    """One external executable and its non-mutating version command."""

    executable: str
    version_arguments: tuple[str, ...]

    def inspect(self) -> HostToolStatus:
        """Inspect this declared executable without duplicating tool metadata."""

        executable_path = shutil.which(self.executable)
        version = None
        if executable_path is not None:
            process = run_checked((executable_path, *self.version_arguments))
            lines = process.stdout.splitlines() or process.stderr.splitlines()
            version = lines[0] if lines else "version command returned no text"
        return HostToolStatus(
            executable=self.executable,
            available=executable_path is not None,
            path=executable_path,
            version=version,
        )


@dataclass(frozen=True)
class HostToolStatus:
    """Observed availability and version of one declared capture tool."""

    executable: str
    available: bool
    path: str | None
    version: str | None


@dataclass(frozen=True)
class CaptureToolDoctorReport:
    """Typed host-capability report for the capture-tool boundary."""

    schema_version: int
    tools: tuple[HostToolStatus, ...]


class HostTool(Enum):
    """External tools used by capture and transcoding operations."""

    FFMPEG = HostToolDeclaration("ffmpeg", ("-version",))
    FFPROBE = HostToolDeclaration("ffprobe", ("-version",))
    MAGICK = HostToolDeclaration("magick", ("-version",))
    XDOTOOL = HostToolDeclaration("xdotool", ("version",))


@dataclass(frozen=True)
class SourceFormatDeclaration:
    """Source suffix and its media category."""

    suffix: str
    category: MediaCategory


class SourceFormat(Enum):
    """Lossless and common source-capture formats accepted by the tool."""

    PNG = SourceFormatDeclaration(".png", MediaCategory.STILL)
    TIFF = SourceFormatDeclaration(".tif", MediaCategory.STILL)
    TIFF_LONG = SourceFormatDeclaration(".tiff", MediaCategory.STILL)
    WEBP = SourceFormatDeclaration(".webp", MediaCategory.STILL)
    MATROSKA = SourceFormatDeclaration(".mkv", MediaCategory.MOTION)
    QUICKTIME = SourceFormatDeclaration(".mov", MediaCategory.MOTION)
    MP4 = SourceFormatDeclaration(".mp4", MediaCategory.MOTION)
    WEBM = SourceFormatDeclaration(".webm", MediaCategory.MOTION)

    @classmethod
    def for_path(cls, path: Path) -> SourceFormat:
        """Return the declaration selected by a capture's file suffix."""

        suffix = path.suffix.lower()
        for source_format in cls:
            if source_format.value.suffix == suffix:
                return source_format
        choices = ", ".join(item.value.suffix for item in cls)
        raise MediaGalleryError(
            f"Unsupported source format {suffix or '<none>'!r}; expected one of "
            f"{choices}."
        )


@dataclass(frozen=True)
class OutputEncodingDeclaration:
    """Authoritative web encoding declaration for one output suffix."""

    suffix: str
    category: MediaCategory
    codec_name: str
    ffmpeg_arguments: tuple[str, ...]
    map_arguments: tuple[str, ...] = ("-map", "0:v:0")
    filter_option: str = "-vf"
    filter_template: str = "{filters}"

    def render_filter(self, filters: str) -> tuple[str, str]:
        """Render the encoding's filter option without caller-side dispatch."""

        return self.filter_option, self.filter_template.format(filters=filters)


class OutputEncoding(Enum):
    """Supported web derivatives and their deterministic FFmpeg contracts."""

    WEBP = OutputEncodingDeclaration(
        suffix=".webp",
        category=MediaCategory.STILL,
        codec_name="webp",
        ffmpeg_arguments=(
            "-frames:v",
            "1",
            "-c:v",
            "libwebp",
            "-preset",
            "picture",
            "-quality",
            "84",
            "-compression_level",
            "6",
            "-threads",
            "1",
        ),
    )
    MP4 = OutputEncodingDeclaration(
        suffix=".mp4",
        category=MediaCategory.MOTION,
        codec_name="h264",
        ffmpeg_arguments=(
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-threads",
            "1",
        ),
    )
    WEBM = OutputEncodingDeclaration(
        suffix=".webm",
        category=MediaCategory.MOTION,
        codec_name="vp9",
        ffmpeg_arguments=(
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "30",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-row-mt",
            "0",
            "-threads",
            "1",
        ),
    )
    GIF = OutputEncodingDeclaration(
        suffix=".gif",
        category=MediaCategory.MOTION,
        codec_name="gif",
        ffmpeg_arguments=("-an", "-loop", "0", "-threads", "1"),
        map_arguments=("-map", "[gallery_output]"),
        filter_option="-filter_complex",
        filter_template=(
            "{filters},split[gallery_a][gallery_b];"
            "[gallery_a]palettegen=max_colors=192:"
            "reserve_transparent=0:stats_mode=diff[gallery_palette];"
            "[gallery_b][gallery_palette]paletteuse="
            "dither=sierra2_4a:diff_mode=rectangle[gallery_output]"
        ),
    )

    @classmethod
    def for_path(cls, path: Path) -> OutputEncoding:
        """Return the single encoding authority selected by output suffix."""

        suffix = path.suffix.lower()
        for encoding in cls:
            if encoding.value.suffix == suffix:
                return encoding
        choices = ", ".join(item.value.suffix for item in cls)
        raise MediaGalleryError(
            f"Unsupported output format {suffix or '<none>'!r}; expected one of "
            f"{choices}."
        )


@dataclass(frozen=True)
class Crop:
    """A rectangular crop in source pixels."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise MediaGalleryError("Crop x and y must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise MediaGalleryError("Crop width and height must be positive.")

    @property
    def ffmpeg_filter(self) -> str:
        """Return the exact FFmpeg crop expression."""

        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


@dataclass(frozen=True)
class Trim:
    """A bounded interval selected from a source recording."""

    start_seconds: float
    duration_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise MediaGalleryError(
                "Trim start_seconds must be a finite non-negative number."
            )
        if (
            not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
            or self.duration_seconds > MAX_DERIVED_MOTION_SECONDS
        ):
            raise MediaGalleryError(
                "Trim duration_seconds must be greater than zero and no more than "
                f"{MAX_DERIVED_MOTION_SECONDS:g}."
            )

    @property
    def end_seconds(self) -> float:
        """Return the exclusive end of the selected source interval."""

        return self.start_seconds + self.duration_seconds


@dataclass(frozen=True)
class Derivative:
    """One output file and all bounds required to validate it."""

    filename: str
    max_width: int
    max_height: int
    max_bytes: int
    fps: int | None = None
    frame_at_seconds: float | None = None

    def __post_init__(self) -> None:
        if not OUTPUT_NAME_PATTERN.fullmatch(self.filename):
            raise MediaGalleryError(
                "Output filename must be a lowercase caption-safe basename using "
                "letters, numbers, and single hyphens."
            )
        if self.max_width <= 0 or self.max_height <= 0:
            raise MediaGalleryError("Output dimensions must be positive.")
        if self.max_bytes <= 0:
            raise MediaGalleryError("Output max_bytes must be positive.")
        if self.fps is not None and not 1 <= self.fps <= 60:
            raise MediaGalleryError("Output fps must be between 1 and 60.")
        if self.frame_at_seconds is not None and (
            not math.isfinite(self.frame_at_seconds) or self.frame_at_seconds < 0
        ):
            raise MediaGalleryError(
                "Output frame_at_seconds must be a finite non-negative number."
            )

    @property
    def path(self) -> Path:
        """Return the validated output basename as a Path."""

        return Path(self.filename)

    @property
    def encoding(self) -> OutputEncoding:
        """Resolve web encoding from the authoritative output suffix."""

        return OutputEncoding.for_path(self.path)

    @property
    def scale_filter(self) -> str:
        """Fit within bounds without upscaling and keep codec-safe dimensions."""

        return (
            f"scale=w='min(iw,{self.max_width})':"
            f"h='min(ih,{self.max_height})':"
            "force_original_aspect_ratio=decrease:force_divisible_by=2"
        )


@dataclass(frozen=True)
class CaptureRecord:
    """One immutable source capture and all authorized derivatives."""

    source: Path
    scenario_id: str
    outputs: tuple[Derivative, ...]
    crop: Crop | None = None
    trim: Trim | None = None

    def __post_init__(self) -> None:
        _validate_relative_path(self.source, "Capture source")
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise MediaGalleryError("Capture scenario_id must be a non-empty string.")
        if not self.outputs:
            raise MediaGalleryError("Each capture must declare at least one output.")
        output_names = tuple(output.filename for output in self.outputs)
        if len(set(output_names)) != len(output_names):
            raise MediaGalleryError(
                f"Capture {self.source} declares duplicate output filenames."
            )
        self.source_format.value.category.validate_source(self)

    @property
    def source_format(self) -> SourceFormat:
        """Resolve source semantics from its suffix."""

        return SourceFormat.for_path(self.source)


@dataclass(frozen=True)
class CaptureManifest:
    """The sole authority for source transformations and output constraints."""

    captures: tuple[CaptureRecord, ...]
    schema_version: int = CAPTURE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAPTURE_MANIFEST_SCHEMA_VERSION:
            raise MediaGalleryError(
                f"Unsupported schema_version {self.schema_version}; expected "
                f"{CAPTURE_MANIFEST_SCHEMA_VERSION}."
            )
        if not self.captures:
            raise MediaGalleryError("The manifest must contain at least one capture.")
        source_names = tuple(str(capture.source) for capture in self.captures)
        if len(set(source_names)) != len(source_names):
            raise MediaGalleryError("Manifest capture source paths must be unique.")
        scenario_ids = tuple(capture.scenario_id for capture in self.captures)
        if len(set(scenario_ids)) != len(scenario_ids):
            raise MediaGalleryError("Manifest gallery scenario ids must be unique.")
        for capture in self.captures:
            try:
                scenario = OpenHCSGalleryScenarioCatalog.for_id(capture.scenario_id)
            except GalleryCatalogError as error:
                raise MediaGalleryError(str(error)) from error
            output_names = tuple(output.filename for output in capture.outputs)
            expected_output_names = scenario.published_paths()
            if output_names != expected_output_names:
                raise MediaGalleryError(
                    f"Capture {capture.scenario_id!r} outputs {output_names!r}; "
                    "expected the declaration-owned inventory "
                    f"{expected_output_names!r}."
                )
        output_names = tuple(
            output.filename for capture in self.captures for output in capture.outputs
        )
        if len(set(output_names)) != len(output_names):
            raise MediaGalleryError(
                "Manifest output filenames must be globally unique."
            )


@dataclass(frozen=True)
class MediaProbe:
    """Relevant FFprobe facts for one source or derivative."""

    width: int
    height: int
    codec_name: str
    duration_seconds: float | None


@dataclass(frozen=True)
class WindowGeometry:
    """Absolute X11 geometry for a selected visible window."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise MediaGalleryError("Selected window has invalid dimensions.")


@dataclass(frozen=True)
class PreparedCapture:
    """A source capture that has passed path, existence, probe, and crop checks."""

    record: CaptureRecord
    source_path: Path
    output_paths: tuple[Path, ...]
    probe: MediaProbe

    def source_result(self) -> GallerySourceCaptureResult:
        """Project the validated source through the gallery evidence authority."""

        return GallerySourceCaptureResult(
            path=str(self.record.source),
            sha256=sha256_file(self.source_path),
            width=self.probe.width,
            height=self.probe.height,
            duration_seconds=self.probe.duration_seconds,
            format=self.probe.codec_name,
        )


@dataclass(frozen=True)
class CapturePlan:
    """Exact write-free command projection for one declared capture."""

    scenario_id: str
    source: Path
    outputs: tuple[Path, ...]
    commands: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DerivativeValidationResult:
    """Validated identity and media facts for one published derivative."""

    path: str
    size_bytes: int
    sha256: str
    width: int
    height: int
    codec: str
    duration_seconds: float | None


@dataclass(frozen=True)
class CaptureValidationResult:
    """Validated source and derivatives for one gallery scenario."""

    scenario_id: str
    source: GallerySourceCaptureResult
    outputs: tuple[DerivativeValidationResult, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class WindowRecordingCaptureResult(GallerySourceCaptureResult):
    """Lossless window-recording evidence including pointer visibility."""

    mouse_visible: bool


def capture_ui_bridge_window_source(
    target: UiBridgeWindowCaptureTarget,
    request: GallerySourceCaptureRequest,
) -> GallerySourceCaptureResult:
    """Capture one declared UI-bridge window through the existing MCP capability."""

    from openhcs.agent.capabilities import (
        DesktopLocalCapabilitySurfaceProfile,
        UiSnapshotWindowCapability,
    )
    from openhcs.agent.dto.ui_bridge import (
        UiWindowSnapshotRequest,
        UiWindowSnapshotResult,
    )
    from openhcs.mcp.control_timeout import McpUiBridgeTimeoutPolicy
    from openhcs.mcp.dev_client import McpDevClient
    from openhcs.mcp.dev_client_core import UiConnectionArguments
    from openhcs.mcp.dev_client_rendering import McpDevPayloadProjection

    target_path = _capture_target(request.source_root, request.output, ".png")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    control_timeout_ms = McpUiBridgeTimeoutPolicy.resolve(request.timeout_ms)
    snapshot_request = UiWindowSnapshotRequest.from_fields(
        window_id=target.window_id,
        output_dir_path=str(target_path.parent),
        create_if_missing=False,
    )
    connection = UiConnectionArguments(
        host=None,
        port=None,
        transport_mode=None,
        auth_token=None,
        descriptor_file_path=(
            None
            if request.descriptor_file_path is None
            else str(request.descriptor_file_path)
        ),
        bridge_instance_id=None,
        timeout_ms=control_timeout_ms,
    )
    arguments = snapshot_request.as_tool_arguments()
    arguments["connection"] = connection.as_tool_arguments()
    timeout_seconds = max(30.0, control_timeout_ms / 1000.0 + 10.0)
    with McpDevClient(surface_profile=DesktopLocalCapabilitySurfaceProfile()) as client:
        execution = client.execute(
            (
                "call",
                UiSnapshotWindowCapability.name,
                "--arguments",
                json.dumps(arguments),
            ),
            timeout_seconds=timeout_seconds,
        )
    if execution.returncode != 0:
        raise MediaGalleryError(
            "MCP window snapshot command failed: "
            f"{json.dumps(execution.payload, sort_keys=True)}"
        )
    response_payload = McpDevPayloadProjection.tool_payload(
        execution.payload,
        UiSnapshotWindowCapability.name,
    )
    if response_payload is None:
        raise MediaGalleryError("MCP snapshot response contains no tool payload.")
    response = dataclass_from_mapping(UiWindowSnapshotResult, response_payload)
    if not response.captured or response.errors:
        raise MediaGalleryError(f"MCP window snapshot failed: {response!r}")
    if response.resource is None:
        raise MediaGalleryError("MCP snapshot response contains no resource.")
    if response.resource.path is None:
        raise MediaGalleryError("MCP snapshot resource contains no local path.")
    if response.resource.sha256 is None:
        raise MediaGalleryError("MCP snapshot resource contains no SHA-256.")
    if response.width is None or response.width <= 0:
        raise MediaGalleryError("MCP snapshot response contains no valid width.")
    if response.height is None or response.height <= 0:
        raise MediaGalleryError("MCP snapshot response contains no valid height.")
    snapshot_path = Path(response.resource.path).resolve()
    if snapshot_path.parent != target_path.parent.resolve():
        raise MediaGalleryError(
            "MCP snapshot resource escaped the requested capture directory: "
            f"{snapshot_path}"
        )
    if snapshot_path.suffix.lower() != ".png":
        raise MediaGalleryError(
            f"MCP snapshot resource is not a lossless PNG: {snapshot_path}"
        )
    if not snapshot_path.is_file():
        raise MediaGalleryError(
            f"MCP snapshot resource does not exist: {snapshot_path}"
        )
    actual_sha256 = sha256_file(snapshot_path)
    if actual_sha256 != response.resource.sha256:
        raise MediaGalleryError(
            "MCP snapshot resource changed before gallery ingestion."
        )
    os.replace(snapshot_path, target_path)
    return GallerySourceCaptureResult(
        path=str(request.output),
        sha256=actual_sha256,
        width=response.width,
        height=response.height,
        format="PNG",
    )


def _expect_object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise MediaGalleryError(f"{context} must be a JSON object.")
    if not all(isinstance(key, str) for key in value):
        raise MediaGalleryError(f"{context} keys must be strings.")
    return value


def _expect_array(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise MediaGalleryError(f"{context} must be a JSON array.")
    return value


def _validate_keys(
    payload: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    context: str,
) -> None:
    keys = frozenset(payload)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise MediaGalleryError(f"{context} is missing fields: {', '.join(missing)}.")
    if unknown:
        raise MediaGalleryError(
            f"{context} contains unknown fields: {', '.join(unknown)}."
        )


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediaGalleryError(f"{context} must be a number.")
    return float(value)


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MediaGalleryError(f"{context} must be an integer.")
    return value


def _optional_integer(
    payload: Mapping[str, Any],
    field: str,
    context: str,
) -> int | None:
    if field not in payload:
        return None
    return _integer(payload[field], f"{context}.{field}")


def _optional_number(
    payload: Mapping[str, Any],
    field: str,
    context: str,
) -> float | None:
    if field not in payload:
        return None
    return _number(payload[field], f"{context}.{field}")


def _parse_crop(value: Any, context: str) -> Crop:
    payload = _expect_object(value, context)
    fields = frozenset({"x", "y", "width", "height"})
    _validate_keys(payload, required=fields, optional=frozenset(), context=context)
    return Crop(
        x=_integer(payload["x"], f"{context}.x"),
        y=_integer(payload["y"], f"{context}.y"),
        width=_integer(payload["width"], f"{context}.width"),
        height=_integer(payload["height"], f"{context}.height"),
    )


def _parse_trim(value: Any, context: str) -> Trim:
    payload = _expect_object(value, context)
    fields = frozenset({"start_seconds", "duration_seconds"})
    _validate_keys(payload, required=fields, optional=frozenset(), context=context)
    return Trim(
        start_seconds=_number(payload["start_seconds"], f"{context}.start_seconds"),
        duration_seconds=_number(
            payload["duration_seconds"],
            f"{context}.duration_seconds",
        ),
    )


def _parse_derivative(value: Any, context: str) -> Derivative:
    payload = _expect_object(value, context)
    _validate_keys(
        payload,
        required=frozenset({"filename", "max_width", "max_height", "max_bytes"}),
        optional=frozenset({"fps", "frame_at_seconds"}),
        context=context,
    )
    filename = payload["filename"]
    if not isinstance(filename, str):
        raise MediaGalleryError(f"{context}.filename must be a string.")
    return Derivative(
        filename=filename,
        max_width=_integer(payload["max_width"], f"{context}.max_width"),
        max_height=_integer(payload["max_height"], f"{context}.max_height"),
        max_bytes=_integer(payload["max_bytes"], f"{context}.max_bytes"),
        fps=_optional_integer(payload, "fps", context),
        frame_at_seconds=_optional_number(payload, "frame_at_seconds", context),
    )


def _parse_capture(value: Any, context: str) -> CaptureRecord:
    payload = _expect_object(value, context)
    _validate_keys(
        payload,
        required=frozenset({"scenario_id", "source", "outputs"}),
        optional=frozenset({"crop", "trim"}),
        context=context,
    )
    source = payload["source"]
    if not isinstance(source, str):
        raise MediaGalleryError(f"{context}.source must be a string.")
    scenario_id = payload["scenario_id"]
    if not isinstance(scenario_id, str):
        raise MediaGalleryError(f"{context}.scenario_id must be a string.")
    outputs = tuple(
        _parse_derivative(output, f"{context}.outputs[{index}]")
        for index, output in enumerate(
            _expect_array(payload["outputs"], f"{context}.outputs")
        )
    )
    crop = (
        _parse_crop(payload["crop"], f"{context}.crop") if "crop" in payload else None
    )
    trim = (
        _parse_trim(payload["trim"], f"{context}.trim") if "trim" in payload else None
    )
    return CaptureRecord(
        source=Path(source),
        scenario_id=scenario_id,
        outputs=outputs,
        crop=crop,
        trim=trim,
    )


def load_manifest(path: Path) -> CaptureManifest:
    """Load and fully validate one capture manifest."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MediaGalleryError(f"Could not read manifest {path}: {error}") from error
    payload = _expect_object(document, "Manifest")
    _validate_keys(
        payload,
        required=frozenset({"schema_version", "captures"}),
        optional=frozenset(),
        context="Manifest",
    )
    return CaptureManifest(
        schema_version=_integer(payload["schema_version"], "Manifest.schema_version"),
        captures=tuple(
            _parse_capture(capture, f"Manifest.captures[{index}]")
            for index, capture in enumerate(
                _expect_array(payload["captures"], "Manifest.captures")
            )
        ),
    )


def _validate_relative_path(path: Path, context: str) -> None:
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise MediaGalleryError(
            f"{context} must be a contained relative path without '..': {path}"
        )


def resolve_contained_path(
    root: Path,
    relative_path: Path,
    *,
    context: str,
) -> Path:
    """Resolve a relative path and reject lexical or symlink root escapes."""

    _validate_relative_path(relative_path, context)
    root = root.resolve()
    candidate = root.joinpath(relative_path)
    if candidate.is_symlink():
        raise MediaGalleryError(f"{context} cannot be a symbolic link: {relative_path}")
    resolved_parent = candidate.parent.resolve()
    try:
        resolved_parent.relative_to(root)
    except ValueError as error:
        raise MediaGalleryError(
            f"{context} escapes its declared root: {relative_path}"
        ) from error
    return resolved_parent / candidate.name


def _require_tool(tool: HostTool) -> str:
    executable = shutil.which(tool.value.executable)
    if executable is None:
        raise MediaGalleryError(
            f"Required host tool {tool.value.executable!r} is unavailable. "
            "Install FFmpeg for "
            "ffmpeg/ffprobe, ImageMagick for magick, or xdotool for X11 window "
            "geometry."
        )
    return executable


def run_checked(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a command without a shell and return its captured output."""

    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "no output"
        raise MediaGalleryError(
            f"Command failed ({process.returncode}): {' '.join(command)}\n{detail}"
        )
    return process


def probe_media(path: Path) -> MediaProbe:
    """Read dimensions, codec, and duration with FFprobe."""

    ffprobe = _require_tool(HostTool.FFPROBE)
    process = run_checked(
        (
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,duration:format=duration",
            "-of",
            "json",
            str(path),
        )
    )
    try:
        payload = json.loads(process.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        codec_name = str(stream["codec_name"])
        if "duration" in stream:
            raw_duration = stream["duration"]
        elif "duration" in payload["format"]:
            raw_duration = payload["format"]["duration"]
        else:
            raw_duration = None
        duration = None if raw_duration in (None, "N/A") else float(raw_duration)
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise MediaGalleryError(
            f"FFprobe returned incomplete media metadata for {path}."
        ) from error
    return MediaProbe(
        width=width,
        height=height,
        codec_name=codec_name,
        duration_seconds=duration,
    )


def _format_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _filters(record: CaptureRecord, output: Derivative) -> str:
    filters = []
    if record.crop is not None:
        filters.append(record.crop.ffmpeg_filter)
    filters.append(output.scale_filter)
    if output.fps is not None:
        filters.append(f"fps={output.fps}")
    filters.append("setsar=1")
    return ",".join(filters)


def build_transcode_command(
    record: CaptureRecord,
    output: Derivative,
    source_path: Path,
    target_path: Path,
    *,
    ffmpeg_executable: str | None = None,
) -> tuple[str, ...]:
    """Project a typed record into one deterministic FFmpeg command."""

    ffmpeg = (
        ffmpeg_executable
        if ffmpeg_executable is not None
        else _require_tool(HostTool.FFMPEG)
    )
    command: list[str] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
    ]
    if record.trim is not None:
        seek_seconds = record.trim.start_seconds
        if output.frame_at_seconds is not None:
            seek_seconds += output.frame_at_seconds
        command.extend(("-ss", _format_seconds(seek_seconds)))
        command.extend(
            output.encoding.value.category.transcode_duration_arguments(record)
        )
    command.extend(output.encoding.value.render_filter(_filters(record, output)))
    command.extend(output.encoding.value.map_arguments)
    command.extend(output.encoding.value.ffmpeg_arguments)
    command.extend(
        (
            "-bitexact",
            "-map_metadata",
            "-1",
            "-metadata",
            "creation_time=1970-01-01T00:00:00Z",
        )
    )
    command.append(str(target_path))
    return tuple(command)


def _validate_source(
    record: CaptureRecord,
    source_path: Path,
    probe: MediaProbe,
) -> None:
    if record.crop is not None and (
        record.crop.x + record.crop.width > probe.width
        or record.crop.y + record.crop.height > probe.height
    ):
        raise MediaGalleryError(
            f"Crop for {record.source} exceeds its {probe.width}x{probe.height} "
            "source bounds."
        )
    if record.trim is not None:
        if probe.duration_seconds is None:
            raise MediaGalleryError(
                f"Motion source {source_path} has no probeable duration."
            )
        if record.trim.end_seconds > probe.duration_seconds + 0.02:
            raise MediaGalleryError(
                f"Trim ending at {record.trim.end_seconds:g}s exceeds source "
                f"duration {probe.duration_seconds:g}s."
            )


def validate_derivative(
    record: CaptureRecord,
    output: Derivative,
    target_path: Path,
    probe: MediaProbe | None = None,
) -> DerivativeValidationResult:
    """Validate one derivative and return reproducibility facts."""

    if not target_path.is_file():
        raise MediaGalleryError(f"Missing derived media: {target_path}")
    size_bytes = target_path.stat().st_size
    if size_bytes > output.max_bytes:
        raise MediaGalleryError(
            f"{output.filename} is {size_bytes} bytes, above its manifest bound "
            f"of {output.max_bytes}."
        )
    probe = probe if probe is not None else probe_media(target_path)
    if probe.width > output.max_width or probe.height > output.max_height:
        raise MediaGalleryError(
            f"{output.filename} is {probe.width}x{probe.height}, above its "
            f"{output.max_width}x{output.max_height} manifest bounds."
        )
    if probe.codec_name != output.encoding.value.codec_name:
        raise MediaGalleryError(
            f"{output.filename} codec is {probe.codec_name!r}, expected "
            f"{output.encoding.value.codec_name!r}."
        )
    output.encoding.value.category.validate_derivative_duration(record, output, probe)
    return DerivativeValidationResult(
        path=output.filename,
        size_bytes=size_bytes,
        sha256=sha256_file(target_path),
        width=probe.width,
        height=probe.height,
        codec=probe.codec_name,
        duration_seconds=probe.duration_seconds,
    )


def sha256_file(path: Path) -> str:
    """Hash one source or derivative without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _paths_for_capture(
    record: CaptureRecord,
    source_root: Path,
    output_root: Path,
) -> tuple[Path, tuple[Path, ...]]:
    source_path = resolve_contained_path(
        source_root,
        record.source,
        context="Capture source",
    )
    output_paths = tuple(
        resolve_contained_path(
            output_root,
            output.path,
            context="Derived output",
        )
        for output in record.outputs
    )
    if source_path in output_paths:
        raise MediaGalleryError(
            f"Refusing to overwrite source capture with a derivative: {source_path}"
        )
    return source_path, output_paths


def plan_manifest(
    manifest: CaptureManifest,
    source_root: Path,
    output_root: Path,
) -> tuple[CapturePlan, ...]:
    """Return exact commands without writing or probing media."""

    plans = []
    for record in manifest.captures:
        source_path, output_paths = _paths_for_capture(
            record,
            source_root,
            output_root,
        )
        ffmpeg_name = HostTool.FFMPEG.value.executable
        ffmpeg = shutil.which(ffmpeg_name) or ffmpeg_name
        commands = tuple(
            build_transcode_command(
                record,
                output,
                source_path,
                target_path,
                ffmpeg_executable=ffmpeg,
            )
            for output, target_path in zip(
                record.outputs,
                output_paths,
                strict=True,
            )
        )
        plans.append(
            CapturePlan(
                scenario_id=record.scenario_id,
                source=source_path,
                outputs=output_paths,
                commands=commands,
            )
        )
    return tuple(plans)


def _temporary_target(target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}-",
        suffix=target_path.suffix,
        dir=target_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    return temporary_path


def _prepare_captures(
    manifest: CaptureManifest,
    source_root: Path,
    output_root: Path,
) -> tuple[PreparedCapture, ...]:
    prepared = []
    for record in manifest.captures:
        source_path, output_paths = _paths_for_capture(
            record,
            source_root,
            output_root,
        )
        if not source_path.is_file():
            raise MediaGalleryError(f"Source capture does not exist: {source_path}")
        source_probe = probe_media(source_path)
        _validate_source(record, source_path, source_probe)
        prepared.append(
            PreparedCapture(
                record=record,
                source_path=source_path,
                output_paths=output_paths,
                probe=source_probe,
            )
        )
    return tuple(prepared)


def build_manifest(
    manifest: CaptureManifest,
    source_root: Path,
    output_root: Path,
    *,
    force: bool = False,
) -> tuple[CaptureValidationResult, ...]:
    """Build all derivatives atomically and validate every manifest bound."""

    prepared_captures = _prepare_captures(manifest, source_root, output_root)
    if not force:
        existing_outputs = tuple(
            target_path
            for prepared in prepared_captures
            for target_path in prepared.output_paths
            if target_path.exists()
        )
        if existing_outputs:
            formatted_paths = "\n".join(f"- {path}" for path in existing_outputs)
            raise MediaGalleryError(
                "Derived outputs already exist. Pass --force only after reviewing "
                f"the sources and manifest:\n{formatted_paths}"
            )

    report = []
    for prepared in prepared_captures:
        record = prepared.record
        source_path = prepared.source_path
        outputs_report: list[DerivativeValidationResult] = []
        for output, target_path in zip(
            record.outputs,
            prepared.output_paths,
            strict=True,
        ):
            temporary_path = _temporary_target(target_path)
            try:
                command = build_transcode_command(
                    record,
                    output,
                    source_path,
                    temporary_path,
                )
                run_checked(command)
                output_report = validate_derivative(
                    record,
                    output,
                    temporary_path,
                )
                os.replace(temporary_path, target_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            outputs_report.append(output_report)
        report.append(
            CaptureValidationResult(
                scenario_id=record.scenario_id,
                source=prepared.source_result(),
                outputs=tuple(outputs_report),
            )
        )
    return tuple(report)


def validate_manifest_outputs(
    manifest: CaptureManifest,
    source_root: Path,
    output_root: Path,
) -> tuple[CaptureValidationResult, ...]:
    """Validate existing sources and derivatives without modifying them."""

    report = []
    for prepared in _prepare_captures(manifest, source_root, output_root):
        record = prepared.record
        outputs_report = tuple(
            validate_derivative(record, output, target_path)
            for output, target_path in zip(
                record.outputs,
                prepared.output_paths,
                strict=True,
            )
        )
        report.append(
            CaptureValidationResult(
                scenario_id=record.scenario_id,
                source=prepared.source_result(),
                outputs=outputs_report,
            )
        )
    return tuple(report)


def _validate_window_id(window_id: str) -> str:
    if not WINDOW_ID_PATTERN.fullmatch(window_id):
        raise MediaGalleryError(
            "Window ID must be a positive decimal X11 ID or hexadecimal 0x ID."
        )
    return window_id


def read_window_geometry(window_id: str) -> WindowGeometry:
    """Read the selected X11 window's absolute geometry through xdotool."""

    xdotool = _require_tool(HostTool.XDOTOOL)
    process = run_checked(
        (xdotool, "getwindowgeometry", "--shell", _validate_window_id(window_id))
    )
    geometry_fields = frozenset(field.name for field in fields(WindowGeometry))
    values: dict[str, int] = {}
    for line in process.stdout.splitlines():
        key, separator, raw_value = line.partition("=")
        field_name = key.lower()
        if separator and field_name in geometry_fields:
            try:
                values[field_name] = int(raw_value)
            except ValueError as error:
                raise MediaGalleryError(
                    f"xdotool returned invalid {key} geometry."
                ) from error
    missing = sorted(geometry_fields - values.keys())
    if missing:
        raise MediaGalleryError(
            "xdotool did not report complete window geometry: "
            f"{', '.join(name.upper() for name in missing)}."
        )
    return dataclass_from_mapping(WindowGeometry, values)


def _capture_target(
    source_root: Path,
    output: Path,
    expected_suffix: str,
) -> Path:
    target_path = resolve_contained_path(
        source_root,
        output,
        context="Raw capture output",
    )
    if target_path.suffix.lower() != expected_suffix:
        raise MediaGalleryError(
            f"Raw capture output must use {expected_suffix}: {output}"
        )
    if target_path.exists():
        raise MediaGalleryError(f"Refusing to overwrite source capture: {target_path}")
    return target_path


def capture_window_still(
    source_root: Path,
    output: Path,
    window_id: str,
) -> GallerySourceCaptureResult:
    """Capture a real X11 window losslessly without overwriting any source."""

    magick = _require_tool(HostTool.MAGICK)
    target_path = _capture_target(source_root, output, ".png")
    temporary_path = _temporary_target(target_path)
    try:
        run_checked(
            (
                magick,
                "import",
                "-window",
                _validate_window_id(window_id),
                str(temporary_path),
            )
        )
        probe = probe_media(temporary_path)
        os.replace(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return GallerySourceCaptureResult(
        path=str(output),
        sha256=sha256_file(target_path),
        width=probe.width,
        height=probe.height,
        format="PNG",
    )


def record_window(
    source_root: Path,
    output: Path,
    window_id: str,
    *,
    duration_seconds: float,
    fps: int,
    display: str,
    draw_mouse: bool = True,
) -> WindowRecordingCaptureResult:
    """Record a fixed real X11 window rectangle into a lossless FFV1 source."""

    if (
        not math.isfinite(duration_seconds)
        or duration_seconds <= 0
        or duration_seconds > MAX_RAW_RECORDING_SECONDS
    ):
        raise MediaGalleryError(
            "Raw recording duration must be greater than zero and no more than "
            f"{MAX_RAW_RECORDING_SECONDS:g} seconds."
        )
    if not 1 <= fps <= 60:
        raise MediaGalleryError("Raw recording fps must be between 1 and 60.")
    if not DISPLAY_PATTERN.fullmatch(display):
        raise MediaGalleryError(f"Invalid X11 DISPLAY value: {display!r}")
    target_path = _capture_target(source_root, output, ".mkv")
    geometry = read_window_geometry(window_id)
    ffmpeg = _require_tool(HostTool.FFMPEG)
    temporary_path = _temporary_target(target_path)
    try:
        run_checked(
            (
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "x11grab",
                "-draw_mouse",
                str(int(draw_mouse)),
                "-framerate",
                str(fps),
                "-video_size",
                f"{geometry.width}x{geometry.height}",
                "-i",
                f"{display}+{geometry.x},{geometry.y}",
                "-t",
                _format_seconds(duration_seconds),
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-g",
                "1",
                "-threads",
                "1",
                str(temporary_path),
            )
        )
        probe = probe_media(temporary_path)
        os.replace(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return WindowRecordingCaptureResult(
        path=str(output),
        sha256=sha256_file(target_path),
        width=probe.width,
        height=probe.height,
        duration_seconds=probe.duration_seconds,
        format="FFV1",
        mouse_visible=draw_mouse,
    )


def capture_scenario_still(
    source_root: Path,
    output: Path,
    scenario_id: str,
    *,
    descriptor_file_path: Path | None = None,
    timeout_ms: int | None = None,
) -> GallerySourceCaptureResult:
    """Capture one still scenario through its nominal live-surface target."""

    try:
        scenario = OpenHCSGalleryScenarioCatalog.for_id(scenario_id)
        request = GallerySourceCaptureRequest(
            source_root=source_root,
            output=output,
            descriptor_file_path=descriptor_file_path,
            timeout_ms=timeout_ms,
        )
        return scenario.capture_source(request)
    except GalleryCatalogError as error:
        raise MediaGalleryError(str(error)) from error


def doctor() -> CaptureToolDoctorReport:
    """Report availability and versions from the typed host-tool authority."""

    return CaptureToolDoctorReport(
        schema_version=CAPTURE_TOOL_REPORT_SCHEMA_VERSION,
        tools=tuple(tool.value.inspect() for tool in HostTool),
    )


def _write_json(value: object) -> None:
    print(json.dumps(to_jsonable(value), indent=2, sort_keys=True))


def _manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def _doctor_operation(_arguments: argparse.Namespace) -> CaptureToolDoctorReport:
    return doctor()


def _catalog_operation(
    _arguments: argparse.Namespace,
) -> tuple[GalleryScenarioCatalogRecord, ...]:
    return tuple(scenario.catalog_record() for scenario in gallery_scenarios())


def _plan_operation(arguments: argparse.Namespace) -> tuple[CapturePlan, ...]:
    return plan_manifest(
        load_manifest(arguments.manifest),
        arguments.source_root,
        arguments.output_root,
    )


def _build_operation(
    arguments: argparse.Namespace,
) -> tuple[CaptureValidationResult, ...]:
    return build_manifest(
        load_manifest(arguments.manifest),
        arguments.source_root,
        arguments.output_root,
        force=arguments.force,
    )


def _validate_operation(
    arguments: argparse.Namespace,
) -> tuple[CaptureValidationResult, ...]:
    return validate_manifest_outputs(
        load_manifest(arguments.manifest),
        arguments.source_root,
        arguments.output_root,
    )


def _capture_still_operation(
    arguments: argparse.Namespace,
) -> GallerySourceCaptureResult:
    return capture_window_still(
        arguments.source_root,
        arguments.output,
        arguments.window_id,
    )


def _capture_scenario_still_operation(
    arguments: argparse.Namespace,
) -> GallerySourceCaptureResult:
    return capture_scenario_still(
        arguments.source_root,
        arguments.output,
        arguments.scenario_id,
        descriptor_file_path=arguments.descriptor_file_path,
        timeout_ms=arguments.timeout_ms,
    )


def _record_window_operation(
    arguments: argparse.Namespace,
) -> WindowRecordingCaptureResult:
    return record_window(
        arguments.source_root,
        arguments.output,
        arguments.window_id,
        duration_seconds=arguments.duration_seconds,
        fps=arguments.fps,
        display=arguments.display,
        draw_mouse=arguments.mouse,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Report capture and encoding tools.",
    )
    doctor_parser.set_defaults(operation=_doctor_operation)

    catalog_parser = subparsers.add_parser(
        "catalog",
        help="Project the declaration-owned scenarios and capture targets.",
    )
    catalog_parser.set_defaults(operation=_catalog_operation)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Validate the manifest and print exact commands without writing.",
    )
    _manifest_arguments(plan_parser)
    plan_parser.set_defaults(operation=_plan_operation)

    build_parser_instance = subparsers.add_parser(
        "build",
        help="Build and validate manifest derivatives atomically.",
    )
    _manifest_arguments(build_parser_instance)
    build_parser_instance.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace existing derived outputs, never source captures.",
    )
    build_parser_instance.set_defaults(operation=_build_operation)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Probe and validate existing source and derivative media.",
    )
    _manifest_arguments(validate_parser)
    validate_parser.set_defaults(operation=_validate_operation)

    still_parser = subparsers.add_parser(
        "capture-still",
        help="Capture one real X11 window into a lossless PNG source.",
    )
    still_parser.add_argument("--source-root", type=Path, required=True)
    still_parser.add_argument("--output", type=Path, required=True)
    still_parser.add_argument("--window-id", required=True)
    still_parser.set_defaults(operation=_capture_still_operation)

    scenario_still_parser = subparsers.add_parser(
        "capture-scenario-still",
        help="Capture one declared still scenario through its stable live target.",
    )
    scenario_still_parser.add_argument("scenario_id")
    scenario_still_parser.add_argument("--source-root", type=Path, required=True)
    scenario_still_parser.add_argument("--output", type=Path, required=True)
    scenario_still_parser.add_argument("--descriptor-file-path", type=Path)
    scenario_still_parser.add_argument("--timeout-ms", type=int)
    scenario_still_parser.set_defaults(operation=_capture_scenario_still_operation)

    record_parser = subparsers.add_parser(
        "record-window",
        help="Record one fixed real X11 window rectangle into lossless FFV1 MKV.",
    )
    record_parser.add_argument("--source-root", type=Path, required=True)
    record_parser.add_argument("--output", type=Path, required=True)
    record_parser.add_argument("--window-id", required=True)
    record_parser.add_argument("--duration-seconds", type=float, required=True)
    record_parser.add_argument("--fps", type=int, default=30)
    record_parser.add_argument(
        "--mouse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include the X11 pointer in the source recording; use --no-mouse "
            "for agent-only interaction."
        ),
    )
    record_parser.add_argument(
        "--display",
        default=os.environ.get("DISPLAY", ":0"),
        help="X11 display used by FFmpeg x11grab.",
    )
    record_parser.set_defaults(operation=_record_window_operation)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected capture/transcoding command."""

    arguments = build_parser().parse_args(argv)
    try:
        _write_json(arguments.operation(arguments))
        return 0
    except MediaGalleryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
