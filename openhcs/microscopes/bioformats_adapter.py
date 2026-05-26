"""Bio-Formats metadata adapter records and manifest-backed test adapter."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from metaclass_registry import AutoRegisterMeta
from polystore.bioformats_java import (
    BioFormatsJavaContext,
    BioFormatsJavaUnavailableError,
    java_float,
    java_int,
    java_str,
)

from openhcs.microscopes.bioformats_well_key import BIOFORMATS_WELL_KEYS


BIOFORMATS_MANIFEST_FILENAME = "bioformats_spw.json"


@dataclass(frozen=True, slots=True)
class BioFormatsPlane:
    """Stable reader plane index for one C/Z/T coordinate."""

    c: int
    z: int
    t: int
    index: int


@dataclass(frozen=True, slots=True)
class BioFormatsPixels:
    """Image pixel metadata needed for OpenHCS source-axis projection."""

    size_c: int
    size_z: int
    size_t: int
    planes: tuple[BioFormatsPlane, ...]


@dataclass(frozen=True, slots=True)
class BioFormatsLayoutCoordinates:
    """Shared HCS coordinates derived from non-OME-SPW layout metadata."""

    well: str
    site: int
    z_index: int
    timepoint: int
    channel_name: str | None


@dataclass(frozen=True, slots=True)
class BioFormatsLayoutAxes(BioFormatsLayoutCoordinates):
    """HCS axes derived from non-OME-SPW vendor layout metadata."""

    channel: int


@dataclass(frozen=True, slots=True)
class BioFormatsImage:
    """Bio-Formats image/series metadata linked from OME WellSample."""

    image_id: str
    source_path: Path
    series_index: int
    pixels: BioFormatsPixels
    source_files: tuple[Path, ...] = ()
    channel_names: tuple[str | None, ...] = ()
    pixel_size: float | None = None
    layout_axes: BioFormatsLayoutAxes | None = None
    reader: str = "bioformats"


@dataclass(frozen=True, slots=True)
class BioFormatsWellSample:
    """OME WellSample metadata projected to OpenHCS site identity."""

    image_id: str
    index: int | None = None


@dataclass(frozen=True, slots=True)
class BioFormatsWell:
    """OME Well metadata projected to OpenHCS well identity."""

    row: int | str
    column: int | str
    samples: tuple[BioFormatsWellSample, ...]


@dataclass(frozen=True, slots=True)
class BioFormatsPlate:
    """OME Plate metadata containing wells and linked samples."""

    wells: tuple[BioFormatsWell, ...]
    name: str | None = None


@dataclass(frozen=True, slots=True)
class BioFormatsMetadata:
    """Reader-neutral OME-SPW metadata discovered from a Bio-Formats root."""

    root: Path
    plates: tuple[BioFormatsPlate, ...]
    images: tuple[BioFormatsImage, ...]


class BioFormatsAdapterUnavailableError(RuntimeError):
    """Raised when no Bio-Formats adapter can inspect a dataset."""


class BioFormatsMetadataAdapter(ABC, metaclass=AutoRegisterMeta):
    """Adapter that emits reader-neutral OME-SPW metadata records."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None

    @abstractmethod
    def discover(self, root: str | Path) -> BioFormatsMetadata:
        """Discover Bio-Formats metadata under a root path."""

    def candidate_source_paths(self, root: Path) -> tuple[Path, ...]:
        """Return candidate Bio-Formats source files under a root."""
        if root.is_file():
            return (root,)
        if not root.is_dir():
            raise BioFormatsAdapterUnavailableError(
                f"Bio-Formats path does not exist: {root}"
            )
        return tuple(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != BIOFORMATS_MANIFEST_FILENAME
        )

    def images_from_java_reader(
        self,
        *,
        source_path: Path,
        reader: Any,
        metadata: Any,
    ) -> tuple[BioFormatsImage, ...]:
        """Project Java Bio-Formats reader state into typed image records."""
        images = []
        for image_index in range(int(metadata.getImageCount())):
            if image_index >= int(reader.getSeriesCount()):
                raise BioFormatsAdapterUnavailableError(
                    "OME image count exceeds Bio-Formats reader series count."
                )
            reader.setSeries(image_index)
            size_c = _java_size_or_reader_size(
                metadata.getPixelsSizeC(image_index),
                reader.getSizeC(),
            )
            size_z = _java_size_or_reader_size(
                metadata.getPixelsSizeZ(image_index),
                reader.getSizeZ(),
            )
            size_t = _java_size_or_reader_size(
                metadata.getPixelsSizeT(image_index),
                reader.getSizeT(),
            )
            images.append(
                BioFormatsImage(
                    image_id=_required_java_str(
                        metadata.getImageID(image_index),
                        "Image.ID",
                    ),
                    source_path=source_path,
                    series_index=image_index,
                    source_files=_series_used_files(reader, source_path),
                    pixels=BioFormatsPixels(
                        size_c=size_c,
                        size_z=size_z,
                        size_t=size_t,
                        planes=_java_planes(
                            metadata=metadata,
                            reader=reader,
                            image_index=image_index,
                            size_c=size_c,
                            size_z=size_z,
                            size_t=size_t,
                        ),
                    ),
                    channel_names=tuple(
                        java_str(metadata.getChannelName(image_index, channel_index))
                        for channel_index in range(size_c)
                    ),
                    pixel_size=java_float(metadata.getPixelsPhysicalSizeX(image_index)),
                )
            )
        return tuple(images)


class BioFormatsCompositeAdapter(BioFormatsMetadataAdapter):
    """Try explicit sidecar metadata first, then real Java Bio-Formats."""

    def __init__(
        self,
        adapters: tuple[BioFormatsMetadataAdapter, ...] | None = None,
    ):
        self.adapters = adapters or (
            self._registered_adapter("manifest"),
            self._registered_adapter("java"),
            self._registered_adapter("filename_layout"),
        )

    def discover(self, root: str | Path) -> BioFormatsMetadata:
        errors = []
        for adapter in self.adapters:
            try:
                return adapter.discover(root)
            except BioFormatsAdapterUnavailableError as exc:
                errors.append(str(exc))
        raise BioFormatsAdapterUnavailableError(
            "No Bio-Formats OME-SPW adapter could inspect this dataset: "
            + "; ".join(errors)
        )

    @classmethod
    def _registered_adapter(cls, registry_key: str) -> BioFormatsMetadataAdapter:
        return BioFormatsMetadataAdapter.__registry__[registry_key]()


class _BioFormatsManifestAdapter(BioFormatsMetadataAdapter):
    """Load Bio-Formats-shaped OME-SPW metadata from a local manifest.

    The manifest path is a deterministic test/integration seam. Production
    adapters should emit the same records after reading real Bio-Formats OME
    metadata.
    """

    registry_key = "manifest"

    def discover(self, root: str | Path) -> BioFormatsMetadata:
        root_path = Path(root)
        manifest_path = _manifest_path(root_path)
        if not manifest_path.exists():
            raise BioFormatsAdapterUnavailableError(
                f"No Bio-Formats manifest found at {manifest_path}. "
                "Install/configure a real Bio-Formats adapter or provide source schema metadata."
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return bioformats_metadata_from_mapping(root_path, payload)


class _BioFormatsJavaAdapter(BioFormatsMetadataAdapter):
    """Discover OME-SPW metadata through the Java Bio-Formats ImageReader."""

    registry_key = "java"

    def discover(self, root: str | Path) -> BioFormatsMetadata:
        root_path = Path(root)
        errors = []
        for source_path in self.candidate_source_paths(root_path):
            try:
                metadata = self._discover_source(root_path, source_path)
            except BioFormatsAdapterUnavailableError as exc:
                errors.append(str(exc))
                continue
            if metadata.plates:
                return metadata
        raise BioFormatsAdapterUnavailableError(
            f"No Bio-Formats OME-SPW plate metadata found under {root_path}."
        )

    def _discover_source(
        self,
        root: Path,
        source_path: Path,
    ) -> BioFormatsMetadata:
        try:
            context = BioFormatsJavaContext.instance()
            opened = context.open_reader(source_path)
        except BioFormatsJavaUnavailableError as exc:
            raise BioFormatsAdapterUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise BioFormatsAdapterUnavailableError(
                f"Bio-Formats could not open {source_path}: {exc}"
            ) from exc

        try:
            plates = _plates_from_java_metadata(opened.metadata)
            if not plates:
                raise BioFormatsAdapterUnavailableError(
                    f"Bio-Formats opened {source_path} but found no OME Plate metadata."
                )
            images = self.images_from_java_reader(
                source_path=source_path,
                reader=opened.reader,
                metadata=opened.metadata,
            )
            return BioFormatsMetadata(root=root, plates=plates, images=images)
        finally:
            opened.close()


@dataclass(frozen=True, slots=True)
class BioFormatsParsedLayout(BioFormatsLayoutCoordinates):
    """Filename-derived HCS axes before channel-number assignment."""

    channel_key: str


class BioFormatsFilenameLayoutParser(ABC, metaclass=AutoRegisterMeta):
    """Registered parser for Bio-Formats-readable HCS filename layouts."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None

    @abstractmethod
    def parse(self, path: Path) -> BioFormatsParsedLayout | None:
        """Return HCS axes for a source file, or None when the layout is unsupported."""


class InCellFilenameLayoutParser(BioFormatsFilenameLayoutParser):
    """Parse GE/Cytiva InCell TIFF names carrying well, field, wavelength, and Z."""

    registry_key = "incell"
    _pattern = re.compile(
        r"^(?P<row>[A-Za-z]+)\s*-\s*(?P<column>\d+)"
        r"\(fld\s*(?P<site>\d+)\s+wv\s*(?P<channel>.+?)"
        r"(?:\s+z\s*(?P<z_index>\d+))?\)\.(?:tif|tiff)$",
        re.IGNORECASE,
    )

    def parse(self, path: Path) -> BioFormatsParsedLayout | None:
        match = self._pattern.match(path.name)
        if match is None:
            return None
        channel_key = _normalize_channel_label(match.group("channel"))
        return BioFormatsParsedLayout(
            well=_well_key(match.group("row"), int(match.group("column"))),
            site=int(match.group("site")),
            channel_key=channel_key,
            channel_name=channel_key,
            z_index=int(match.group("z_index") or 1),
            timepoint=1,
        )


class InCell3000FilenameLayoutParser(BioFormatsFilenameLayoutParser):
    """Parse InCell 3000/BBBC013 BMP names carrying channel, site, row, and column."""

    registry_key = "incell3000"
    _pattern = re.compile(
        r"^Channel(?P<channel>\d+)-(?P<site>\d+)-(?P<row>[A-Za-z]+)-(?P<column>\d+)"
        r"\.(?:bmp|tif|tiff)$",
        re.IGNORECASE,
    )

    def parse(self, path: Path) -> BioFormatsParsedLayout | None:
        match = self._pattern.match(path.name)
        if match is None:
            return None
        channel = int(match.group("channel"))
        return BioFormatsParsedLayout(
            well=_well_key(match.group("row"), int(match.group("column"))),
            site=int(match.group("site")),
            channel_key=str(channel),
            channel_name=f"Channel {channel}",
            z_index=1,
            timepoint=1,
        )


class CV7000FilenameLayoutParser(BioFormatsFilenameLayoutParser):
    """Parse Yokogawa CV7000 TIFF names carrying well, field, tile, Z, T, and C."""

    registry_key = "cv7000"
    _pattern = re.compile(
        r"^.+_(?P<well>[A-Za-z]+\d+)_T(?P<timepoint>\d+)F(?P<field>\d+)"
        r"L(?P<line>\d+)A(?P<area>\d+)Z(?P<z_index>\d+)C(?P<channel>\d+)"
        r"\.(?:tif|tiff)$",
        re.IGNORECASE,
    )

    def parse(self, path: Path) -> BioFormatsParsedLayout | None:
        match = self._pattern.match(path.name)
        if match is None:
            return None
        channel = int(match.group("channel"))
        return BioFormatsParsedLayout(
            well=match.group("well").upper(),
            site=_compound_site_index(
                int(match.group("field")),
                int(match.group("area")),
            ),
            channel_key=str(channel),
            channel_name=f"C{channel:02d}",
            z_index=int(match.group("z_index")),
            timepoint=int(match.group("timepoint")),
        )


class ScanRFilenameLayoutParser(BioFormatsFilenameLayoutParser):
    """Parse Olympus ScanR TIFF names carrying well, position, Z, T, and channel."""

    registry_key = "scanr"
    _pattern = re.compile(
        r"^--W(?P<well>\d+)--P(?P<site>\d+)--Z(?P<z_index>\d+)"
        r"--T(?P<timepoint>\d+)--(?P<channel>.+?)\.(?:tif|tiff)$",
        re.IGNORECASE,
    )

    def parse(self, path: Path) -> BioFormatsParsedLayout | None:
        match = self._pattern.match(path.name)
        if match is None:
            return None
        channel_key = _normalize_channel_label(match.group("channel"))
        return BioFormatsParsedLayout(
            well=f"W{int(match.group('well')):05d}",
            site=int(match.group("site")),
            channel_key=channel_key,
            channel_name=channel_key,
            z_index=int(match.group("z_index")) + 1,
            timepoint=int(match.group("timepoint")) + 1,
        )


class MetaXpressFilenameLayoutParser(BioFormatsFilenameLayoutParser):
    """Parse Molecular Devices MetaXpress TIFF names carrying well, site, and channel."""

    registry_key = "metaxpress"
    _pattern = re.compile(
        r"^.*?_(?P<well>[A-Za-z]+\d+)(?:_s(?P<site>\d+))?_w(?P<channel>\d+)"
        r"\.(?:tif|tiff)$",
        re.IGNORECASE,
    )

    def parse(self, path: Path) -> BioFormatsParsedLayout | None:
        match = self._pattern.match(path.name)
        if match is None:
            return None
        channel = int(match.group("channel"))
        return BioFormatsParsedLayout(
            well=match.group("well").upper(),
            site=int(match.group("site") or 1),
            channel_key=str(channel),
            channel_name=f"W{channel}",
            z_index=1,
            timepoint=1,
        )


class OperettaFilenameLayoutParser(BioFormatsFilenameLayoutParser):
    """Parse PerkinElmer/Revvity Operetta TIFF names carrying well, field, plane, and channel."""

    registry_key = "operetta"
    _pattern = re.compile(
        r"^r(?P<row>\d+)c(?P<column>\d+)f(?P<field>\d+)p(?P<plane>\d+)"
        r"-ch(?P<channel>\d+)sk(?P<stack>\d+)fk(?P<timepoint>\d+)fl(?P<fl>\d+)"
        r"\.(?:tif|tiff)$",
        re.IGNORECASE,
    )

    def parse(self, path: Path) -> BioFormatsParsedLayout | None:
        match = self._pattern.match(path.name)
        if match is None:
            return None
        channel = int(match.group("channel"))
        return BioFormatsParsedLayout(
            well=_numeric_well_key(int(match.group("row")), int(match.group("column"))),
            site=int(match.group("field")),
            channel_key=str(channel),
            channel_name=f"Channel {channel}",
            z_index=int(match.group("plane")),
            timepoint=int(match.group("stack")),
        )


class _BioFormatsFilenameLayoutAdapter(BioFormatsMetadataAdapter):
    """Discover HCS axes from supported vendor filename layouts."""

    registry_key = "filename_layout"

    def __init__(
        self,
        parsers: tuple[BioFormatsFilenameLayoutParser, ...] | None = None,
    ):
        self.parsers = parsers or tuple(
            parser_type()
            for parser_type in BioFormatsFilenameLayoutParser.__registry__.values()
        )

    def discover(self, root: str | Path) -> BioFormatsMetadata:
        root_path = Path(root)
        candidates = []
        channel_numbers: dict[str, int] = {}
        for source_path in self.candidate_source_paths(root_path):
            parsed = self._parse(source_path)
            if parsed is None:
                continue
            channel_numbers.setdefault(parsed.channel_key, len(channel_numbers) + 1)
            candidates.append((source_path, parsed))
        if not candidates:
            raise BioFormatsAdapterUnavailableError(
                f"No supported Bio-Formats HCS filename layout found under {root_path}."
            )

        images: list[BioFormatsImage] = []
        errors = []
        for source_path, parsed in candidates:
            try:
                images.extend(
                    self._layout_images(
                        root_path,
                        source_path,
                        parsed,
                        channel=channel_numbers[parsed.channel_key],
                    )
                )
            except BioFormatsAdapterUnavailableError as exc:
                errors.append(str(exc))
        if not images:
            raise BioFormatsAdapterUnavailableError(
                "Supported HCS filename layout was found, but Bio-Formats could not "
                "open any matching source files: "
                + "; ".join(errors[:5])
            )
        return BioFormatsMetadata(root=root_path, plates=(), images=tuple(images))

    def _parse(self, source_path: Path) -> BioFormatsParsedLayout | None:
        for parser in self.parsers:
            parsed = parser.parse(source_path)
            if parsed is not None:
                return parsed
        return None

    def _layout_images(
        self,
        root: Path,
        source_path: Path,
        parsed: BioFormatsParsedLayout,
        *,
        channel: int,
    ) -> tuple[BioFormatsImage, ...]:
        try:
            context = BioFormatsJavaContext.instance()
            opened = context.open_reader(source_path)
        except BioFormatsJavaUnavailableError as exc:
            raise BioFormatsAdapterUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise BioFormatsAdapterUnavailableError(
                f"Bio-Formats could not open {source_path}: {exc}"
            ) from exc

        try:
            images = self.images_from_java_reader(
                source_path=source_path,
                reader=opened.reader,
                metadata=opened.metadata,
            )
            return tuple(
                replace(
                    image,
                    image_id=_layout_image_id(root, source_path, image.series_index),
                    layout_axes=BioFormatsLayoutAxes(
                        well=parsed.well,
                        site=parsed.site,
                        channel=channel,
                        z_index=parsed.z_index,
                        timepoint=parsed.timepoint,
                        channel_name=parsed.channel_name,
                    ),
                )
                for image in images
            )
        finally:
            opened.close()


def _manifest_path(root: str | Path) -> Path:
    return Path(root) / BIOFORMATS_MANIFEST_FILENAME


def bioformats_metadata_from_mapping(
    root: Path,
    payload: Mapping[str, Any],
) -> BioFormatsMetadata:
    """Build typed Bio-Formats metadata records from a JSON-like mapping."""

    images = tuple(
        _image_from_mapping(root, image_payload)
        for image_payload in payload.get("images", ())
    )
    plates = tuple(
        BioFormatsPlate(
            name=plate_payload.get("name"),
            wells=tuple(
                BioFormatsWell(
                    row=well_payload["row"],
                    column=well_payload["column"],
                    samples=tuple(
                        BioFormatsWellSample(
                            image_id=sample_payload["image_id"],
                            index=sample_payload.get("index"),
                        )
                        for sample_payload in well_payload.get("samples", ())
                    ),
                )
                for well_payload in plate_payload.get("wells", ())
            ),
        )
        for plate_payload in payload.get("plates", ())
    )
    return BioFormatsMetadata(root=root, plates=plates, images=images)


def _image_from_mapping(
    root: Path,
    image_payload: Mapping[str, Any],
) -> BioFormatsImage:
    pixels_payload = image_payload["pixels"]
    planes = tuple(
        BioFormatsPlane(
            c=int(plane_payload["c"]),
            z=int(plane_payload["z"]),
            t=int(plane_payload["t"]),
            index=int(plane_payload["index"]),
        )
        for plane_payload in pixels_payload.get("planes", ())
    )
    source_path = Path(image_payload["source_path"])
    if not source_path.is_absolute():
        source_path = root / source_path
    source_files = tuple(
        _absolute_source_path(root, path)
        for path in image_payload.get("source_files", (str(source_path),))
    )
    return BioFormatsImage(
        image_id=str(image_payload["image_id"]),
        source_path=source_path,
        series_index=int(image_payload.get("series_index", 0)),
        source_files=source_files,
        pixels=BioFormatsPixels(
            size_c=int(pixels_payload["size_c"]),
            size_z=int(pixels_payload["size_z"]),
            size_t=int(pixels_payload["size_t"]),
            planes=planes,
        ),
        channel_names=tuple(image_payload.get("channel_names", ())),
        pixel_size=(
            None
            if image_payload.get("pixel_size") is None
            else float(image_payload["pixel_size"])
        ),
        layout_axes=_layout_axes_from_mapping(image_payload),
        reader=str(image_payload.get("reader", "bioformats")),
    )


def _layout_axes_from_mapping(
    image_payload: Mapping[str, Any],
) -> BioFormatsLayoutAxes | None:
    axes_payload = image_payload.get("layout_axes")
    if axes_payload is None:
        return None
    return BioFormatsLayoutAxes(
        well=str(axes_payload["well"]),
        site=int(axes_payload["site"]),
        channel=int(axes_payload["channel"]),
        z_index=int(axes_payload.get("z_index", 1)),
        timepoint=int(axes_payload.get("timepoint", 1)),
        channel_name=axes_payload.get("channel_name"),
    )


def _plates_from_java_metadata(metadata: Any) -> tuple[BioFormatsPlate, ...]:
    plate_count = int(metadata.getPlateCount())
    plates = []
    for plate_index in range(plate_count):
        wells = []
        for well_index in range(int(metadata.getWellCount(plate_index))):
            samples = []
            for sample_index in range(
                int(metadata.getWellSampleCount(plate_index, well_index))
            ):
                image_ref = java_str(
                    metadata.getWellSampleImageRef(
                        plate_index,
                        well_index,
                        sample_index,
                    )
                )
                if image_ref is None:
                    raise BioFormatsAdapterUnavailableError(
                        "OME WellSample is missing ImageRef metadata."
                    )
                samples.append(
                    BioFormatsWellSample(
                        image_id=image_ref,
                        index=java_int(
                            metadata.getWellSampleIndex(
                                plate_index,
                                well_index,
                                sample_index,
                            )
                        ),
                    )
                )
            wells.append(
                BioFormatsWell(
                    row=_required_java_int(
                        metadata.getWellRow(plate_index, well_index),
                        "Well.Row",
                    ),
                    column=_required_java_int(
                        metadata.getWellColumn(plate_index, well_index),
                        "Well.Column",
                    ),
                    samples=tuple(samples),
                )
            )
        plates.append(
            BioFormatsPlate(
                name=java_str(metadata.getPlateName(plate_index)),
                wells=tuple(wells),
            )
        )
    return tuple(plates)


def _absolute_source_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _series_used_files(reader: Any, source_path: Path) -> tuple[Path, ...]:
    try:
        files = reader.getSeriesUsedFiles(False)
    except Exception:
        return (source_path,)
    if files is None:
        return (source_path,)
    paths = tuple(_source_file_path(source_path, path) for path in files if str(path))
    return paths or (source_path,)


def _source_file_path(source_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return source_path.parent / path


def _java_planes(
    *,
    metadata: Any,
    reader: Any,
    image_index: int,
    size_c: int,
    size_z: int,
    size_t: int,
) -> tuple[BioFormatsPlane, ...]:
    expected_count = size_c * size_z * size_t
    explicit_planes = []
    for plane_index in range(int(metadata.getPlaneCount(image_index))):
        c_index = java_int(metadata.getPlaneTheC(image_index, plane_index))
        z_index = java_int(metadata.getPlaneTheZ(image_index, plane_index))
        t_index = java_int(metadata.getPlaneTheT(image_index, plane_index))
        if c_index is None or z_index is None or t_index is None:
            continue
        explicit_planes.append(
            BioFormatsPlane(
                c=c_index + 1,
                z=z_index + 1,
                t=t_index + 1,
                index=plane_index,
            )
        )
    if len(explicit_planes) == expected_count:
        return tuple(explicit_planes)

    return tuple(
        BioFormatsPlane(
            c=channel + 1,
            z=z_index + 1,
            t=timepoint + 1,
            index=int(reader.getIndex(z_index, channel, timepoint)),
        )
        for timepoint in range(size_t)
        for z_index in range(size_z)
        for channel in range(size_c)
    )


def _java_size_or_reader_size(java_value: Any, reader_value: Any) -> int:
    value = java_int(java_value)
    return int(reader_value if value is None else value)


def _required_java_int(value: Any, field_name: str) -> int:
    converted = java_int(value)
    if converted is None:
        raise BioFormatsAdapterUnavailableError(f"OME metadata missing {field_name}.")
    return converted


def _required_java_str(value: Any, field_name: str) -> str:
    converted = java_str(value)
    if converted is None:
        raise BioFormatsAdapterUnavailableError(f"OME metadata missing {field_name}.")
    return converted


def _well_key(row: str, column: int) -> str:
    return f"{row.strip().upper()}{column:02d}"


def _numeric_well_key(row: int, column: int) -> str:
    return BIOFORMATS_WELL_KEYS.key_from_one_based(row, column)


def _compound_site_index(field: int, area: int) -> int:
    return (field - 1) * 1000 + area


def _normalize_channel_label(value: str) -> str:
    label = re.sub(r"\s+", " ", value).strip()
    if " - " in label:
        first, second = (part.strip() for part in label.split(" - ", maxsplit=1))
        if first.lower() == second.lower():
            return first
    return label


def _layout_image_id(root: Path, source_path: Path, series_index: int) -> str:
    try:
        relative = source_path.relative_to(root)
    except ValueError:
        relative = source_path
    return f"layout:{relative.as_posix()}:{series_index}"
