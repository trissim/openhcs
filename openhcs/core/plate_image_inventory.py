"""Public plate image inventory and bounded sampling helpers."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openhcs.constants.constants import FileFormat
from openhcs.core.pipeline.path_planner import PathPlannerPathAuthority
from openhcs.core.source_workspace_projection import (
    VirtualWorkspacePathLookup,
    VirtualWorkspaceSourceProjection,
)
from openhcs.core.virtual_workspace_metadata import (
    JsonScalar,
    JsonValue,
)

if TYPE_CHECKING:
    from openhcs.core.orchestrator.orchestrator import Orchestrator
    from openhcs.microscopes.microscope_interfaces import (
        AnalysisResultDirectory,
        FilenameParser,
        MetadataHandler,
    )


JsonLike = str | int | float | bool | None


class PlateFileKind(str, Enum):
    """Kind of file exposed by a plate inventory."""

    IMAGE = "image"
    RESULT = "result"


@dataclass(frozen=True, slots=True)
class PlateImageRecord:
    """One image exposed by a plate, including virtual and physical identity."""

    virtual_path: str
    full_virtual_path: str
    source_path: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    @property
    def source_path_obj(self) -> Path:
        return Path(self.source_path)

    @property
    def size_bytes(self) -> int | None:
        source_path = self.source_path_obj
        if not source_path.exists():
            return None
        return source_path.stat().st_size


@dataclass(frozen=True, slots=True)
class PlateImageInventory:
    """Inventory of plate images resolved through microscope metadata authority."""

    plate_path: Path
    records: tuple[PlateImageRecord, ...]

    @classmethod
    def from_orchestrator(
        cls,
        orchestrator: "Orchestrator",
        *,
        all_subdirs: bool = True,
    ) -> "PlateImageInventory":
        handler = orchestrator.microscope_handler
        if handler is None:
            orchestrator.initialize_microscope_handler()
            handler = orchestrator.microscope_handler
        if handler is None:
            return cls(plate_path=Path(orchestrator.plate_path), records=())
        return cls.from_handler(
            plate_path=Path(orchestrator.plate_path),
            metadata_handler=handler.metadata_handler,
            parser=handler.parser,
            all_subdirs=all_subdirs,
        )

    @classmethod
    def from_handler(
        cls,
        *,
        plate_path: Path,
        metadata_handler: "MetadataHandler",
        parser: "FilenameParser | None",
        all_subdirs: bool = True,
    ) -> "PlateImageInventory":
        image_files = tuple(
            str(image_file)
            for image_file in sorted(
                metadata_handler.get_image_files(
                    plate_path,
                    all_subdirs=all_subdirs,
                )
            )
        )
        projection = cls._projection(plate_path, metadata_handler)
        records = tuple(
            cls._record(
                plate_path=plate_path,
                image_file=image_file,
                parser=parser,
                projection=projection,
            )
            for image_file in image_files
        )
        return cls(plate_path=plate_path, records=records)

    @staticmethod
    def _projection(
        plate_path: Path,
        metadata_handler: "MetadataHandler",
    ) -> VirtualWorkspaceSourceProjection | None:
        metadata = metadata_handler.source_workspace_metadata_document(plate_path)
        if not isinstance(metadata, Mapping):
            return None
        return VirtualWorkspaceSourceProjection.from_openhcs_metadata_if_available(
            plate_path,
            metadata,
        )

    @staticmethod
    def _record(
        *,
        plate_path: Path,
        image_file: str,
        parser: "FilenameParser | None",
        projection: VirtualWorkspaceSourceProjection | None,
    ) -> PlateImageRecord:
        full_virtual_path = str(plate_path / image_file)
        lookup = VirtualWorkspacePathLookup.from_paths(
            image_file,
            full_virtual_path,
        )
        source_path = (
            str(plate_path / image_file)
            if projection is None
            else projection.source_path_for(lookup)
        )
        source_metadata = (
            None if projection is None else projection.source_metadata_for(lookup)
        )
        metadata: dict[str, JsonValue] = {
            "filename": image_file,
            "virtual_path": image_file,
            "full_virtual_path": full_virtual_path,
            "source_path": source_path,
            "type": "Image",
        }
        if source_metadata is not None:
            metadata.update(dict(source_metadata))
        if parser is not None:
            parsed = parser.parse_filename(image_file)
            if parsed:
                metadata.update(dict(parsed))
        source_file = Path(source_path)
        metadata["size"] = file_size_label(source_file)
        metadata["modified"] = file_modified_label(source_file)
        return PlateImageRecord(
            virtual_path=image_file,
            full_virtual_path=full_virtual_path,
            source_path=source_path,
            metadata=metadata,
        )

    def require_record(self, image_path: str) -> PlateImageRecord:
        """Return the uniquely matching image record for a virtual or source path."""
        exact_matches = tuple(
            record
            for record in self.records
            if image_path
            in {
                record.virtual_path,
                record.full_virtual_path,
                record.source_path,
            }
        )
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            raise ValueError(f"Image path {image_path!r} matched multiple records.")

        basename_matches = tuple(
            record
            for record in self.records
            if Path(record.virtual_path).name == image_path
            or Path(record.source_path).name == image_path
        )
        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            raise ValueError(f"Image basename {image_path!r} is ambiguous.")
        raise ValueError(f"Image path {image_path!r} was not found in the plate.")


@dataclass(frozen=True, slots=True)
class PlateResultFileRecord:
    """One analysis result artifact exposed by a plate metadata handler."""

    relative_path: str
    full_path: str
    file_format: FileFormat
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    @property
    def full_path_obj(self) -> Path:
        return Path(self.full_path)


@dataclass(frozen=True, slots=True)
class PlateFileRecord:
    """Unified file record shape shared by browser and agent projections."""

    kind: PlateFileKind
    key: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    virtual_path: str | None = None
    full_virtual_path: str | None = None
    source_path: str | None = None
    relative_path: str | None = None
    full_path: str | None = None
    file_format: FileFormat | None = None

    @classmethod
    def from_image(cls, record: PlateImageRecord) -> "PlateFileRecord":
        return cls(
            kind=PlateFileKind.IMAGE,
            key=record.virtual_path,
            metadata=record.metadata,
            virtual_path=record.virtual_path,
            full_virtual_path=record.full_virtual_path,
            source_path=record.source_path,
        )

    @classmethod
    def from_result(cls, record: PlateResultFileRecord) -> "PlateFileRecord":
        return cls(
            kind=PlateFileKind.RESULT,
            key=record.relative_path,
            metadata=record.metadata,
            relative_path=record.relative_path,
            full_path=record.full_path,
            file_format=record.file_format,
        )

    def matches(self, query: "PlateFileInventoryQuery") -> bool:
        if query.kinds and self.kind not in query.kinds:
            return False
        if query.path_contains is not None and query.path_contains:
            needle = query.path_contains.lower()
            haystack = " ".join(
                path.lower()
                for path in (
                    self.key,
                    self.virtual_path,
                    self.full_virtual_path,
                    self.source_path,
                    self.relative_path,
                    self.full_path,
                )
                if path is not None
            )
            if needle not in haystack:
                return False
        if query.well is not None:
            well = self.metadata.get("well")
            if well is None or str(well) != query.well:
                return False
        return True


@dataclass(frozen=True, slots=True)
class PlateFileInventoryQuery:
    """Bounded filter over unified plate file records."""

    kinds: tuple[PlateFileKind, ...] = ()
    path_contains: str | None = None
    well: str | None = None
    offset: int = 0
    limit: int = 50

    def normalized(self) -> "PlateFileInventoryQuery":
        return PlateFileInventoryQuery(
            kinds=self.kinds,
            path_contains=self.path_contains,
            well=self.well,
            offset=max(0, int(self.offset)),
            limit=max(0, int(self.limit)),
        )


@dataclass(frozen=True, slots=True)
class PlateFileInventoryQueryResult:
    """Bounded query result for unified plate file records."""

    records: tuple[PlateFileRecord, ...]
    total_count: int
    offset: int
    limit: int
    truncated_count: int


@dataclass(frozen=True, slots=True)
class PlateResultFilePreview:
    """Bounded preview for agent-facing result artifacts."""

    text_lines: tuple[str, ...] = ()
    csv_columns: tuple[str, ...] = ()
    csv_rows: tuple[Mapping[str, str], ...] = ()
    roi_count: int | None = None
    roi_member_count: int | None = None
    roi_duplicate_member_count: int | None = None
    roi_area_min: float | None = None
    roi_area_max: float | None = None
    roi_area_mean: float | None = None
    roi_examples: tuple[Mapping[str, JsonValue], ...] = ()
    truncated: bool = False
    omitted_reason: str | None = None


class PlateResultFilePreviewReader:
    """Read bounded previews for analysis result files."""

    DEFAULT_MAX_LINES = 8
    DEFAULT_MAX_BYTES = 64 * 1024

    @classmethod
    def preview(
        cls,
        record: PlateResultFileRecord,
        *,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> PlateResultFilePreview | None:
        if record.file_format is FileFormat.ROI:
            return cls._roi_preview(
                record,
                max_examples=max_lines,
                max_bytes=max_bytes,
            )
        if record.file_format not in {
            FileFormat.CSV,
            FileFormat.JSON,
            FileFormat.TEXT,
        }:
            return None
        path = record.full_path_obj
        if not path.exists():
            return PlateResultFilePreview(omitted_reason="file does not exist")
        if path.stat().st_size > max_bytes:
            return PlateResultFilePreview(
                omitted_reason=f"file exceeds max preview bytes ({max_bytes})"
            )
        lines, truncated = cls._text_lines(path, max_lines=max_lines)
        if record.file_format is not FileFormat.CSV:
            return PlateResultFilePreview(text_lines=lines, truncated=truncated)
        columns, rows = cls._csv_table_from_path(path, max_rows=max_lines)
        return PlateResultFilePreview(
            text_lines=lines,
            csv_columns=columns,
            csv_rows=rows,
            truncated=truncated,
        )

    @classmethod
    def _roi_preview(
        cls,
        record: PlateResultFileRecord,
        *,
        max_examples: int,
        max_bytes: int,
    ) -> PlateResultFilePreview:
        path = record.full_path_obj
        if not path.exists():
            return PlateResultFilePreview(omitted_reason="file does not exist")
        if path.stat().st_size > max_bytes:
            return PlateResultFilePreview(
                omitted_reason=f"file exceeds max preview bytes ({max_bytes})"
            )
        try:
            from polystore.roi import load_rois_from_zip

            rois = load_rois_from_zip(path)
        except Exception as exc:
            return PlateResultFilePreview(
                omitted_reason=f"ROI preview failed: {type(exc).__name__}: {exc}"
            )

        bounded_examples = max(0, max_examples)
        semantic_rois = cls._semantic_rois(rois)
        areas = tuple(
            float(area)
            for roi in semantic_rois
            if isinstance((area := roi.metadata.get("area")), int | float)
        )
        return PlateResultFilePreview(
            roi_count=len(semantic_rois),
            roi_member_count=len(rois),
            roi_duplicate_member_count=max(0, len(rois) - len(semantic_rois)),
            roi_area_min=min(areas) if areas else None,
            roi_area_max=max(areas) if areas else None,
            roi_area_mean=(sum(areas) / len(areas)) if areas else None,
            roi_examples=tuple(
                cls._roi_example(roi)
                for roi in semantic_rois[:bounded_examples]
            ),
            truncated=len(semantic_rois) > bounded_examples,
        )

    @classmethod
    def _semantic_rois(cls, rois: Sequence) -> tuple:
        """Collapse ImageJ archive members that represent one semantic ROI."""
        unique = {}
        for roi in rois:
            unique.setdefault(cls._semantic_roi_identity(roi), roi)
        return tuple(unique.values())

    @staticmethod
    def _semantic_roi_identity(roi):
        metadata = getattr(roi, "metadata", {})
        if metadata:
            return semantic_roi_identity_from_metadata(metadata)
        return tuple(
            _hashable_jsonable_metadata_value(getattr(shape, "__dict__", {}))
            for shape in (getattr(roi, "shapes", ()) or ())
        )

    @staticmethod
    def _roi_example(roi) -> Mapping[str, JsonValue]:
        metadata = getattr(roi, "metadata", {})
        example: dict[str, JsonValue] = {
            "shape_count": len(getattr(roi, "shapes", ()) or ()),
        }
        for key in ("label", "area", "bbox", "centroid", "plane_indices"):
            if key in metadata:
                example[key] = _jsonable_metadata_value(metadata[key])
        return example

    @staticmethod
    def _text_lines(path: Path, *, max_lines: int) -> tuple[tuple[str, ...], bool]:
        bounded_max_lines = max(0, max_lines)
        lines: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for index, line in enumerate(handle):
                if index >= bounded_max_lines:
                    return tuple(lines), True
                lines.append(line.rstrip("\r\n"))
        return tuple(lines), False

    @staticmethod
    def _csv_table(
        lines: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[Mapping[str, str], ...]]:
        records = tuple(
            PlateResultFilePreviewReader._csv_line_fields(line)
            for line in lines
        )
        return PlateResultFilePreviewReader._csv_table_from_records(
            records,
            max_rows=len(lines),
        )

    @staticmethod
    def _csv_table_from_path(
        path: Path,
        *,
        max_rows: int,
    ) -> tuple[tuple[str, ...], tuple[Mapping[str, str], ...]]:
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                records = tuple(
                    tuple(field.strip() for field in row)
                    for row in csv.reader(handle)
                )
        except csv.Error:
            return (), ()
        return PlateResultFilePreviewReader._csv_table_from_records(
            records,
            max_rows=max_rows,
        )

    @staticmethod
    def _csv_table_from_records(
        records: tuple[tuple[str, ...], ...],
        *,
        max_rows: int,
    ) -> tuple[tuple[str, ...], tuple[Mapping[str, str], ...]]:
        if not records:
            return (), ()
        fallback_columns = records[0]
        fallback_columns = (
            fallback_columns
            if PlateResultFilePreviewReader._valid_csv_header(fallback_columns)
            else ()
        )
        bounded_max_rows = max(0, max_rows)
        for index, columns in enumerate(records[:-1]):
            if not PlateResultFilePreviewReader._valid_csv_header(columns):
                continue

            rows: list[Mapping[str, str]] = []
            for fields in records[index + 1 :]:
                if len(fields) != len(columns) or not any(fields):
                    break
                rows.append(dict(zip(columns, fields, strict=True)))
                if len(rows) >= bounded_max_rows:
                    break
            if rows:
                return columns, tuple(rows)
        return fallback_columns, ()

    @staticmethod
    def _csv_line_fields(line: str) -> tuple[str, ...]:
        try:
            parsed = next(csv.reader((line,)))
        except csv.Error:
            return ()
        return tuple(field.strip() for field in parsed)

    @staticmethod
    def _valid_csv_header(columns: tuple[str, ...]) -> bool:
        return bool(columns) and all(columns) and len(set(columns)) == len(columns)


@dataclass(frozen=True, slots=True)
class PlateResultFileInventory:
    """Inventory of plate analysis artifacts resolved through metadata authority."""

    plate_path: Path
    records: tuple[PlateResultFileRecord, ...]
    scanned_file_count: int = 0

    @classmethod
    def from_orchestrator(
        cls,
        orchestrator: "Orchestrator",
    ) -> "PlateResultFileInventory":
        plate_path = Path(orchestrator.plate_path)
        handler = orchestrator.microscope_handler
        if handler is None:
            return cls.from_configured_output_root(
                plate_path=plate_path,
                path_config=orchestrator.get_effective_config().path_planning_config,
            )
        return cls.from_handler_and_configured_output_root(
            plate_path=plate_path,
            metadata_handler=handler.metadata_handler,
            parser=handler.parser,
            path_config=orchestrator.get_effective_config().path_planning_config,
        )

    @classmethod
    def from_handler_and_configured_output_root(
        cls,
        *,
        plate_path: Path,
        metadata_handler: "MetadataHandler",
        parser: "FilenameParser | None",
        path_config,
    ) -> "PlateResultFileInventory":
        """Read handler-declared artifacts plus the configured OpenHCS output root."""
        handler_inventory = cls.from_directories(
            plate_path=plate_path,
            result_directories=metadata_handler.analysis_result_directories(plate_path),
            parser=parser,
        )
        direct_output_inventory = cls.from_configured_output_root(
            plate_path=plate_path,
            path_config=path_config,
            parser=parser,
        )
        source_output_inventory = cls.from_configured_source_output_root(
            plate_path=plate_path,
            path_config=path_config,
            parser=parser,
        )
        return cls._merged(
            plate_path=plate_path,
            inventories=(
                handler_inventory,
                direct_output_inventory,
                source_output_inventory,
            ),
        )

    @classmethod
    def from_configured_source_output_root(
        cls,
        *,
        plate_path: Path,
        path_config,
        parser: "FilenameParser | None" = None,
    ) -> "PlateResultFileInventory":
        """Read artifacts from the output plate planned for a source plate."""
        output_plate_root = PathPlannerPathAuthority.build_output_plate_root(
            plate_path,
            path_config,
        )
        return cls.from_configured_output_root(
            plate_path=output_plate_root,
            path_config=path_config,
            parser=parser,
        )

    @classmethod
    def configured_output_result_directories(
        cls,
        *,
        plate_path: Path,
        path_config,
    ) -> tuple["AnalysisResultDirectory", ...]:
        """Return result directories for an already-built OpenHCS output root."""
        if not path_config.sub_dir:
            return ()
        result_path = PathPlannerPathAuthority.analysis_results_dir_for(
            plate_path / path_config.sub_dir,
        )
        if not result_path.is_dir():
            return ()
        from openhcs.microscopes.microscope_interfaces import AnalysisResultDirectory

        return (
            AnalysisResultDirectory(
                subdirectory_name=path_config.sub_dir,
                path=result_path,
            ),
        )

    @staticmethod
    def _merged(
        *,
        plate_path: Path,
        inventories: tuple["PlateResultFileInventory", ...],
    ) -> "PlateResultFileInventory":
        records_by_path: dict[str, PlateResultFileRecord] = {}
        scanned_file_count = 0
        for inventory in inventories:
            scanned_file_count += inventory.scanned_file_count
            records_by_path.update(
                (record.full_path, record) for record in inventory.records
            )
        return PlateResultFileInventory(
            plate_path=plate_path,
            records=tuple(
                sorted(records_by_path.values(), key=lambda record: record.relative_path)
            ),
            scanned_file_count=scanned_file_count,
        )

    @classmethod
    def from_directories(
        cls,
        *,
        plate_path: Path,
        result_directories: tuple["AnalysisResultDirectory", ...],
        parser: "FilenameParser | None",
    ) -> "PlateResultFileInventory":
        records: list[PlateResultFileRecord] = []
        scanned_file_count = 0
        for result_directory in result_directories:
            directory_records, directory_count = cls._records_from_directory(
                plate_path=plate_path,
                result_directory=result_directory,
                parser=parser,
            )
            records.extend(directory_records)
            scanned_file_count += directory_count
        return cls(
            plate_path=plate_path,
            records=tuple(sorted(records, key=lambda record: record.relative_path)),
            scanned_file_count=scanned_file_count,
        )

    @classmethod
    def from_handler(
        cls,
        *,
        plate_path: Path,
        metadata_handler: "MetadataHandler",
        parser: "FilenameParser | None",
    ) -> "PlateResultFileInventory":
        return cls.from_directories(
            plate_path=plate_path,
            result_directories=metadata_handler.analysis_result_directories(plate_path),
            parser=parser,
        )

    @classmethod
    def from_configured_output_root(
        cls,
        *,
        plate_path: Path,
        path_config,
        parser: "FilenameParser | None" = None,
    ) -> "PlateResultFileInventory":
        """Read artifacts from the output-root layout declared by path planning."""
        if not path_config.sub_dir:
            return cls(plate_path=plate_path, records=())
        return cls.from_directories(
            plate_path=plate_path,
            result_directories=cls.configured_output_result_directories(
                plate_path=plate_path,
                path_config=path_config,
            ),
            parser=parser,
        )

    @classmethod
    def _records_from_directory(
        cls,
        *,
        plate_path: Path,
        result_directory: "AnalysisResultDirectory",
        parser: "FilenameParser | None",
    ) -> tuple[tuple[PlateResultFileRecord, ...], int]:
        records: list[PlateResultFileRecord] = []
        scanned_file_count = 0
        for file_path in sorted(result_directory.path.rglob("*")):
            if not file_path.is_file():
                continue
            scanned_file_count += 1
            file_format = cls._result_file_format(file_path)
            if file_format is None:
                continue
            relative_path = file_path.relative_to(plate_path)
            metadata: dict[str, JsonValue] = {
                "filename": str(relative_path),
                "type": file_format.name,
                "size": file_size_label(file_path),
                "modified": file_modified_label(file_path),
                "result_subdirectory": result_directory.subdirectory_name,
                "full_path": str(file_path),
            }
            if parser is not None:
                parsed = parser.parse_filename(file_path.name)
                if parsed:
                    metadata.update(dict(parsed))
            records.append(
                PlateResultFileRecord(
                    relative_path=str(relative_path),
                    full_path=str(file_path),
                    file_format=file_format,
                    metadata=metadata,
                )
            )
        return tuple(records), scanned_file_count

    @staticmethod
    def _result_file_format(file_path: Path) -> FileFormat | None:
        if file_path.name.endswith(FileFormat.ROI.value[0]):
            return FileFormat.ROI
        suffix = file_path.suffix.lower()
        for file_format in (FileFormat.CSV, FileFormat.JSON, FileFormat.TEXT):
            if suffix in file_format.value:
                return file_format
        return None


@dataclass(frozen=True, slots=True)
class PlateFileInventory:
    """Unified plate file inventory shared by UI and agent services."""

    plate_path: Path
    image_records: tuple[PlateImageRecord, ...]
    result_records: tuple[PlateResultFileRecord, ...]
    scanned_result_file_count: int = 0

    @classmethod
    def from_orchestrator(
        cls,
        orchestrator: "Orchestrator",
        *,
        all_subdirs: bool = True,
    ) -> "PlateFileInventory":
        image_inventory = PlateImageInventory.from_orchestrator(
            orchestrator,
            all_subdirs=all_subdirs,
        )
        result_inventory = PlateResultFileInventory.from_orchestrator(orchestrator)
        return cls.from_inventories(image_inventory, result_inventory)

    @classmethod
    def from_handler(
        cls,
        *,
        plate_path: Path,
        metadata_handler: "MetadataHandler",
        parser: "FilenameParser | None",
        path_config=None,
        all_subdirs: bool = True,
    ) -> "PlateFileInventory":
        """Build the same file inventory shape when only a handler is available."""
        image_inventory = PlateImageInventory.from_handler(
            plate_path=plate_path,
            metadata_handler=metadata_handler,
            parser=parser,
            all_subdirs=all_subdirs,
        )
        if path_config is None:
            result_inventory = PlateResultFileInventory.from_handler(
                plate_path=plate_path,
                metadata_handler=metadata_handler,
                parser=parser,
            )
        else:
            result_inventory = (
                PlateResultFileInventory.from_handler_and_configured_output_root(
                    plate_path=plate_path,
                    metadata_handler=metadata_handler,
                    parser=parser,
                    path_config=path_config,
                )
            )
        return cls.from_inventories(image_inventory, result_inventory)

    @classmethod
    def from_inventories(
        cls,
        image_inventory: PlateImageInventory,
        result_inventory: PlateResultFileInventory,
    ) -> "PlateFileInventory":
        """Project image and result inventories into the unified file shape."""
        return cls(
            plate_path=image_inventory.plate_path,
            image_records=image_inventory.records,
            result_records=result_inventory.records,
            scanned_result_file_count=result_inventory.scanned_file_count,
        )

    @property
    def image_inventory(self) -> PlateImageInventory:
        return PlateImageInventory(
            plate_path=self.plate_path,
            records=self.image_records,
        )

    @property
    def result_inventory(self) -> PlateResultFileInventory:
        return PlateResultFileInventory(
            plate_path=self.plate_path,
            records=self.result_records,
            scanned_file_count=self.scanned_result_file_count,
        )

    def file_records(
        self,
        *,
        kinds: tuple[PlateFileKind, ...] = (),
    ) -> tuple[PlateFileRecord, ...]:
        """Return images and result artifacts in a common query shape."""
        records = (
            *(PlateFileRecord.from_image(record) for record in self.image_records),
            *(PlateFileRecord.from_result(record) for record in self.result_records),
        )
        if not kinds:
            return records
        return tuple(record for record in records if record.kind in kinds)

    def query_files(
        self,
        query: PlateFileInventoryQuery,
    ) -> PlateFileInventoryQueryResult:
        """Return a bounded file query over image and result records."""
        normalized = query.normalized()
        matched = tuple(
            record
            for record in self.file_records(kinds=normalized.kinds)
            if record.matches(normalized)
        )
        start = normalized.offset
        stop = start + normalized.limit
        records = matched[start:stop] if normalized.limit else ()
        return PlateFileInventoryQueryResult(
            records=records,
            total_count=len(matched),
            offset=start,
            limit=normalized.limit,
            truncated_count=max(0, len(matched) - start - len(records)),
        )

    def require_image_record(self, image_path: str) -> PlateImageRecord:
        """Return a uniquely matching image record from this file inventory."""
        return self.image_inventory.require_record(image_path)

    def require_file_record(
        self,
        file_path: str,
        *,
        kinds: tuple[PlateFileKind, ...] = (),
    ) -> PlateFileRecord:
        """Return a uniquely matching image or result record from this inventory."""
        records = self.file_records(kinds=kinds)
        exact_matches = tuple(
            record
            for record in records
            if file_path
            in {
                value
                for value in (
                    record.key,
                    record.virtual_path,
                    record.full_virtual_path,
                    record.source_path,
                    record.relative_path,
                    record.full_path,
                )
                if value is not None
            }
        )
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            raise ValueError(f"File path {file_path!r} matched multiple records.")

        basename_matches = tuple(
            record
            for record in records
            if any(
                Path(value).name == file_path
                for value in (
                    record.key,
                    record.virtual_path,
                    record.full_virtual_path,
                    record.source_path,
                    record.relative_path,
                    record.full_path,
                )
                if value is not None
            )
        )
        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            raise ValueError(f"File basename {file_path!r} is ambiguous.")
        raise ValueError(f"File path {file_path!r} was not found in the plate.")


@dataclass(frozen=True, slots=True)
class PlateImageSample:
    """Bounded pixel sample and summary statistics for one plate image."""

    record: PlateImageRecord
    shape: tuple[int, ...]
    dtype: str
    minimum: JsonLike
    maximum: JsonLike
    mean: float | None
    sample_origin_yx: tuple[int, int]
    sample_shape: tuple[int, ...]
    sample_included: bool
    sample_values: JsonValue = ()
    sample_omitted_reason: str | None = None


class PlateImageSampler:
    """Read and summarize plate images through a resolved image inventory record."""

    @staticmethod
    def sample(
        record: PlateImageRecord,
        *,
        y: int = 0,
        x: int = 0,
        height: int = 32,
        width: int = 32,
        include_array_values: bool = True,
        max_array_elements: int = 4096,
    ) -> PlateImageSample:
        if y < 0 or x < 0:
            raise ValueError("Sample origin y/x must be nonnegative.")
        if height <= 0 or width <= 0:
            raise ValueError("Sample height/width must be positive.")
        if max_array_elements < 0:
            raise ValueError("max_array_elements must be nonnegative.")

        image = _read_image(record.source_path_obj)
        array = np.asarray(image)
        if array.ndim < 2:
            raise ValueError(
                f"Plate image {record.source_path!r} must have at least 2 dimensions."
            )

        y_stop = min(array.shape[-2], y + height)
        x_stop = min(array.shape[-1], x + width)
        sample = array[..., y:y_stop, x:x_stop]
        sample_included = include_array_values and sample.size <= max_array_elements
        omitted_reason = None
        sample_values: JsonValue = ()
        if sample_included:
            sample_values = _jsonable_array_values(sample)
        elif not include_array_values:
            omitted_reason = "include_array_values is false"
        else:
            omitted_reason = (
                f"sample has {sample.size} elements, above max_array_elements="
                f"{max_array_elements}"
            )

        return PlateImageSample(
            record=record,
            shape=tuple(int(value) for value in array.shape),
            dtype=str(array.dtype),
            minimum=_jsonable_scalar(array.min()) if array.size else None,
            maximum=_jsonable_scalar(array.max()) if array.size else None,
            mean=float(array.mean()) if array.size else None,
            sample_origin_yx=(int(y), int(x)),
            sample_shape=tuple(int(value) for value in sample.shape),
            sample_included=sample_included,
            sample_values=sample_values,
            sample_omitted_reason=omitted_reason,
        )


def file_size_label(file_path: Path) -> str:
    if not file_path.exists():
        return "N/A"
    size_bytes = file_path.stat().st_size
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def file_modified_label(file_path: Path) -> str:
    if not file_path.exists():
        return "N/A"
    return datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(
        timespec="seconds"
    )


def _read_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        return np.asarray(tifffile.imread(path))
    import imageio.v3 as imageio

    return np.asarray(imageio.imread(path))


def _jsonable_array_values(array: np.ndarray) -> JsonValue:
    if array.ndim == 0:
        return _jsonable_scalar(array.item())
    return [_jsonable_array_values(np.asarray(value)) for value in array]


def _jsonable_metadata_value(value) -> JsonValue:
    if isinstance(value, np.generic):
        return _jsonable_scalar(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_metadata_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return tuple(_jsonable_metadata_value(item) for item in value)
    return str(value)


def semantic_roi_identity_from_metadata(metadata: Mapping):
    """Return a hashable semantic ROI identity from ROI metadata."""
    return _hashable_jsonable_metadata_value(metadata)


def _hashable_jsonable_metadata_value(value):
    jsonable = _jsonable_metadata_value(value)
    if isinstance(jsonable, Mapping):
        return tuple(
            sorted(
                (
                    str(key),
                    _hashable_jsonable_metadata_value(item),
                )
                for key, item in jsonable.items()
            )
        )
    if isinstance(jsonable, list | tuple):
        return tuple(_hashable_jsonable_metadata_value(item) for item in jsonable)
    return jsonable


def _jsonable_scalar(value: np.generic | JsonScalar) -> JsonLike:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
