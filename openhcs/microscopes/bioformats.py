"""Generic Bio-Formats microscope handler using OME-SPW projection."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from openhcs.constants.constants import Backend
from openhcs.microscopes.bioformats_adapter import (
    BioFormatsAdapterUnavailableError,
    BioFormatsCompositeAdapter,
    BioFormatsMetadataAdapter,
)
from openhcs.microscopes.bioformats_spw_projector import (
    BioFormatsDataset,
    BioFormatsImageEntry,
    BioFormatsLayoutProjector,
    BioFormatsProjectionError,
    BioFormatsSPWProjector,
)
from openhcs.microscopes.microscope_base import (
    MicroscopeHandler,
    register_metadata_handler,
)
from openhcs.microscopes.microscope_interfaces import MetadataHandler
from openhcs.microscopes.openhcs import (
    AtomicMetadataWriter,
    FIELDS,
    OpenHCSMetadata,
    get_metadata_path,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser
from polystore.filemanager import FileManager


class BioFormatsFilenameParser(SourceSchemaFilenameParser):
    """Parser for normalized Bio-Formats virtual workspace keys."""


class BioFormatsMetadataHandler(MetadataHandler):
    """Metadata handler backed by generic OME-SPW projection."""

    def __init__(self, filemanager: FileManager | None = None):
        super().__init__()
        self.filemanager = filemanager

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Path:
        try:
            BioFormatsDatasetAuthority().project(plate_path)
            return Path(plate_path)
        except BioFormatsAdapterUnavailableError as exc:
            raise FileNotFoundError(
                f"No Bio-Formats-readable OME-SPW metadata found for {plate_path}."
            ) from exc

    def projected_entries(
        self,
        plate_path: Union[str, Path],
    ) -> tuple[BioFormatsImageEntry, ...]:
        return BioFormatsDatasetAuthority().project(plate_path).entries

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> tuple[int, int]:
        return (1, 1)

    def get_pixel_size(self, plate_path: Union[str, Path]) -> float:
        values = {
            entry.pixel_size
            for entry in self.projected_entries(plate_path)
            if entry.pixel_size is not None
        }
        if not values:
            return self.FALLBACK_VALUES["pixel_size"]
        if len(values) != 1:
            raise ValueError(f"Multiple Bio-Formats pixel sizes found: {sorted(values)}")
        return next(iter(values))

    def get_channel_values(
        self,
        plate_path: Union[str, Path],
    ) -> Optional[Dict[str, Optional[str]]]:
        channel_values: dict[str, Optional[str]] = {}
        for entry in self.projected_entries(plate_path):
            key = str(entry.channel)
            value = entry.channel_name or f"Channel {entry.channel}"
            previous = channel_values.get(key)
            if previous is not None and previous != value:
                raise ValueError(
                    f"Conflicting Bio-Formats channel name for channel {key}: "
                    f"{previous!r} vs {value!r}"
                )
            channel_values[key] = value
        return channel_values or None

    def get_well_values(
        self,
        plate_path: Union[str, Path],
    ) -> Optional[Dict[str, Optional[str]]]:
        wells = {entry.well: entry.well for entry in self.projected_entries(plate_path)}
        return dict(sorted(wells.items())) or None

    def get_site_values(
        self,
        plate_path: Union[str, Path],
    ) -> Optional[Dict[str, Optional[str]]]:
        sites = {
            str(entry.site): f"Site {entry.site}"
            for entry in self.projected_entries(plate_path)
        }
        return dict(sorted(sites.items(), key=lambda item: int(item[0]))) or None

    def get_z_index_values(
        self,
        plate_path: Union[str, Path],
    ) -> Optional[Dict[str, Optional[str]]]:
        z_indexes = {
            str(entry.z_index): f"Z{entry.z_index}"
            for entry in self.projected_entries(plate_path)
        }
        return dict(sorted(z_indexes.items(), key=lambda item: int(item[0]))) or None

    def get_timepoint_values(
        self,
        plate_path: Union[str, Path],
    ) -> Optional[Dict[str, Optional[str]]]:
        timepoints = {
            str(entry.timepoint): f"T{entry.timepoint}"
            for entry in self.projected_entries(plate_path)
        }
        return dict(sorted(timepoints.items(), key=lambda item: int(item[0]))) or None


class BioFormatsDatasetAuthority:
    """Nominal authority for reading and projecting Bio-Formats metadata."""

    def __init__(
        self,
        adapter: BioFormatsMetadataAdapter | None = None,
        projector: BioFormatsSPWProjector | None = None,
        layout_projector: BioFormatsLayoutProjector | None = None,
        completeness_validator: "BioFormatsDatasetCompletenessValidator | None" = None,
    ):
        self.adapter = adapter or BioFormatsCompositeAdapter()
        self.projector = projector or BioFormatsSPWProjector()
        self.layout_projector = layout_projector or BioFormatsLayoutProjector()
        self.completeness_validator = (
            completeness_validator or BioFormatsDatasetCompletenessValidator()
        )

    def project(self, plate_path: Union[str, Path]) -> BioFormatsDataset:
        try:
            metadata = self.adapter.discover(plate_path)
        except BioFormatsAdapterUnavailableError:
            raise
        try:
            dataset = self._project_metadata(metadata)
            self.completeness_validator.validate(dataset)
            return dataset
        except BioFormatsProjectionError as exc:
            raise BioFormatsProjectionError(
                f"Bio-Formats dataset at {plate_path} is readable but cannot be "
                "projected into OpenHCS HCS axes. Provide an explicit source schema. "
                f"{exc}"
            ) from exc

    def _project_metadata(self, metadata) -> BioFormatsDataset:
        if metadata.plates:
            return self.projector.project(metadata)
        return self.layout_projector.project(metadata)


class BioFormatsDatasetCompletenessValidator:
    """Validate that projected Bio-Formats planes have source-file provenance."""

    def validate(self, dataset: BioFormatsDataset) -> None:
        missing_files = sorted(
            {
                path
                for entry in dataset.entries
                for path in _entry_source_files(entry)
                if not path.exists()
            }
        )
        unresolved_entries = tuple(
            entry
            for entry in dataset.entries
            if not _entry_has_pixel_source_provenance(entry)
        )
        if not missing_files and not unresolved_entries:
            return
        details = []
        if missing_files:
            details.append(
                "missing source files: "
                + ", ".join(str(path) for path in missing_files[:5])
                + ("" if len(missing_files) <= 5 else f", ... +{len(missing_files) - 5}")
            )
        if unresolved_entries:
            details.append(
                "metadata-only series without pixel source files: "
                + ", ".join(_entry_label(entry) for entry in unresolved_entries[:5])
                + (
                    ""
                    if len(unresolved_entries) <= 5
                    else f", ... +{len(unresolved_entries) - 5}"
                )
            )
        raise BioFormatsProjectionError(
            "Bio-Formats OME-SPW metadata describes planes that are not backed by "
            "concrete source image files; refusing to create a partial virtual "
            "workspace. "
            + "; ".join(details)
        )


class BioFormatsWorkspaceMetadataWriter:
    """Emit OpenHCS metadata for a projected Bio-Formats dataset."""

    def __init__(self, parser: BioFormatsFilenameParser | None = None):
        self.parser = parser or BioFormatsFilenameParser()

    def write(self, plate_root: Path, dataset: BioFormatsDataset) -> None:
        entries = tuple(sorted(dataset.entries, key=_entry_sort_key))
        workspace_mapping = {
            self.virtual_path(entry): self.ref_payload(plate_root, entry)
            for entry in entries
        }
        metadata = OpenHCSMetadata(
            microscope_handler_name=BioFormatsHandler._microscope_type,
            source_filename_parser_name="BioFormatsFilenameParser",
            grid_dimensions=[1, 1],
            pixel_size=_metadata_pixel_size(entries),
            image_files=list(workspace_mapping),
            channels=_component_values(
                (str(entry.channel), entry.channel_name or f"Channel {entry.channel}")
                for entry in entries
            ),
            wells=_component_values((entry.well, entry.well) for entry in entries),
            sites=_component_values(
                (str(entry.site), f"Site {entry.site}") for entry in entries
            ),
            z_indexes=_component_values(
                (str(entry.z_index), f"Z{entry.z_index}") for entry in entries
            ),
            timepoints=_component_values(
                (str(entry.timepoint), f"T{entry.timepoint}") for entry in entries
            ),
            available_backends={
                Backend.BIOFORMATS.value: True,
            },
            workspace_mapping=workspace_mapping,
            source_metadata={
                path: {
                    "source_path": str(payload["source_path"]),
                    "series_index": str(payload["series_index"]),
                    "plane_index": str(payload["plane_index"]),
                }
                for path, payload in workspace_mapping.items()
            },
            main=True,
        )
        AtomicMetadataWriter().merge_subdirectory_metadata(
            get_metadata_path(plate_root),
            {FIELDS.DEFAULT_SUBDIRECTORY: asdict(metadata)},
        )

    def virtual_path(self, entry: BioFormatsImageEntry) -> str:
        return self.parser.construct_filename(
            well=entry.well,
            site=entry.site,
            channel=entry.channel,
            z_index=entry.z_index,
            timepoint=entry.timepoint,
            extension=".tif",
        )

    def ref_payload(
        self,
        plate_root: Path,
        entry: BioFormatsImageEntry,
    ) -> dict[str, object]:
        try:
            source_path = entry.source_path.relative_to(plate_root).as_posix()
        except ValueError:
            source_path = str(entry.source_path)
        return {
            "backend": Backend.BIOFORMATS.value,
            "reader": entry.reader,
            "source_path": source_path,
            "series_index": entry.series_index,
            "plane_index": entry.plane_index,
            "c": entry.source_channel,
            "z": entry.source_z_index,
            "t": entry.source_timepoint,
        }


class BioFormatsHandler(MicroscopeHandler):
    """Brand-agnostic Bio-Formats handler for OME-SPW HCS datasets."""

    _microscope_type = "bioformats"
    _metadata_handler_class = BioFormatsMetadataHandler
    detection_priority = "fallback"

    def __init__(self, filemanager: FileManager, pattern_format: Optional[str] = None):
        self.parser = BioFormatsFilenameParser(filemanager, pattern_format)
        self.metadata_handler = BioFormatsMetadataHandler(filemanager)
        self.dataset_authority = BioFormatsDatasetAuthority()
        self.metadata_writer = BioFormatsWorkspaceMetadataWriter(self.parser)
        super().__init__(parser=self.parser, metadata_handler=self.metadata_handler)

    @property
    def root_dir(self) -> str:
        return "."

    @property
    def microscope_type(self) -> str:
        return self._microscope_type

    @property
    def metadata_handler_class(self) -> Type[MetadataHandler]:
        return self._metadata_handler_class

    @property
    def compatible_backends(self) -> List[Backend]:
        return [Backend.BIOFORMATS]

    def initialize_workspace(self, plate_path: Path, filemanager: FileManager) -> Path:
        plate_root = Path(plate_path)
        dataset = self.dataset_authority.project(plate_root)
        self.metadata_writer.write(plate_root, dataset)
        self._register_bioformats_backend(plate_root, filemanager)
        self.plate_folder = plate_root
        return plate_root

    def get_available_backends(self, plate_path: Union[str, Path]) -> List[Backend]:
        return [Backend.BIOFORMATS]

    def get_primary_backend(
        self,
        plate_path: Union[str, Path],
        filemanager: FileManager,
    ) -> str:
        if Backend.BIOFORMATS.value not in filemanager.registry:
            self._register_bioformats_backend(Path(plate_path), filemanager)
        return Backend.BIOFORMATS.value

    def post_workspace(
        self,
        plate_path: Union[str, Path],
        filemanager: FileManager,
        skip_preparation: bool = False,
    ) -> Path:
        plate_root = Path(plate_path)
        if not skip_preparation:
            self.metadata_writer.write(
                plate_root,
                self.dataset_authority.project(plate_root),
            )
        self._register_bioformats_backend(plate_root, filemanager)
        return plate_root

    def _register_bioformats_backend(
        self,
        plate_path: Path,
        filemanager: FileManager,
    ) -> None:
        from polystore.bioformats_storage import BioFormatsStorageBackend

        filemanager.registry[Backend.BIOFORMATS.value] = BioFormatsStorageBackend(
            plate_root=plate_path,
        )



def _metadata_pixel_size(entries: tuple[BioFormatsImageEntry, ...]) -> float:
    values = {entry.pixel_size for entry in entries if entry.pixel_size is not None}
    if len(values) > 1:
        raise ValueError(f"Multiple Bio-Formats pixel sizes found: {sorted(values)}")
    return next(iter(values), MetadataHandler.FALLBACK_VALUES["pixel_size"])


def _component_values(
    pairs,
) -> Optional[Dict[str, Optional[str]]]:
    values: dict[str, Optional[str]] = {}
    for key, value in pairs:
        values[str(key)] = value
    return dict(sorted(values.items())) or None


def _entry_sort_key(entry: BioFormatsImageEntry) -> tuple[str, int, int, int, int]:
    return (
        entry.well,
        entry.site,
        entry.channel,
        entry.z_index,
        entry.timepoint,
    )


def _entry_source_files(entry: BioFormatsImageEntry) -> tuple[Path, ...]:
    return entry.source_files or (entry.source_path,)


def _entry_has_pixel_source_provenance(entry: BioFormatsImageEntry) -> bool:
    source_files = _entry_source_files(entry)
    if any(not path.exists() for path in source_files):
        return False
    if not _metadata_only_source(entry.source_path):
        return True
    source_path = _normalized_path(entry.source_path)
    return any(
        _normalized_path(path) != source_path and _pixel_source_file(path)
        for path in source_files
    )


def _metadata_only_source(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith((".ome.tif", ".ome.tiff")):
        return False
    return path.suffix.lower() in {
        ".htd",
        ".ome",
        ".xml",
        ".wpi",
        ".xdce",
    }


def _pixel_source_file(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in {
        ".c01",
        ".dib",
        ".flex",
        ".jp2",
        ".png",
        ".tif",
        ".tiff",
    } or name.endswith((".tif.gz", ".tiff.gz", ".ome.tif", ".ome.tiff"))


def _normalized_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _entry_label(entry: BioFormatsImageEntry) -> str:
    return (
        f"{entry.well}/s{entry.site}/c{entry.channel}/z{entry.z_index}/"
        f"t{entry.timepoint} series={entry.series_index}"
    )


BioFormatsHandler._metadata_handler_class = BioFormatsMetadataHandler
register_metadata_handler(BioFormatsHandler, BioFormatsMetadataHandler)
