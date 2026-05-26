"""Project OME Screen/Plate/Well metadata onto OpenHCS source axes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openhcs.microscopes.bioformats_adapter import (
    BioFormatsImage,
    BioFormatsLayoutAxes,
    BioFormatsMetadata,
    BioFormatsPlane,
    BioFormatsWell,
    BioFormatsWellSample,
)
from openhcs.microscopes.bioformats_well_key import BIOFORMATS_WELL_KEYS


@dataclass(frozen=True, slots=True)
class BioFormatsImageEntry:
    """One normalized OpenHCS source plane backed by a Bio-Formats ref."""

    source_path: Path
    series_index: int
    plane_index: int
    source_files: tuple[Path, ...]
    well: str
    site: int
    channel: int
    z_index: int
    timepoint: int
    source_channel: int
    source_z_index: int
    source_timepoint: int
    reader: str = "bioformats"
    channel_name: str | None = None
    pixel_size: float | None = None


@dataclass(frozen=True, slots=True)
class BioFormatsDataset:
    """Projected Bio-Formats dataset ready for virtual workspace emission."""

    root: Path
    entries: tuple[BioFormatsImageEntry, ...]


class BioFormatsProjectionError(ValueError):
    """Raised when OME-SPW metadata cannot be projected safely."""


class BioFormatsSPWProjector:
    """Brand-agnostic OME-SPW to OpenHCS axis projector."""

    def project(self, metadata: BioFormatsMetadata) -> BioFormatsDataset:
        image_by_id = {image.image_id: image for image in metadata.images}
        if not metadata.plates:
            raise BioFormatsProjectionError(
                "Bio-Formats metadata contains no OME Plate records; "
                "source schema is required for HCS plate axes."
            )

        entries: list[BioFormatsImageEntry] = []
        for plate in metadata.plates:
            if not plate.wells:
                raise BioFormatsProjectionError(
                    "OME Plate contains no Well records; source schema is required."
                )
            for well in plate.wells:
                if not well.samples:
                    continue
                well_key = well_key_from_row_column(well.row, well.column)
                for site, sample in self._site_samples(well):
                    try:
                        image = image_by_id[sample.image_id]
                    except KeyError as exc:
                        raise BioFormatsProjectionError(
                            f"WellSample references unknown image {sample.image_id!r}."
                        ) from exc
                    entries.extend(self._image_entries(image, well_key, site))

        if not entries:
            raise BioFormatsProjectionError(
                "OME-SPW metadata produced no image-plane entries."
            )
        return BioFormatsDataset(root=metadata.root, entries=tuple(entries))

    def _site_samples(
        self,
        well: BioFormatsWell,
    ) -> tuple[tuple[int, BioFormatsWellSample], ...]:
        if not well.samples:
            raise BioFormatsProjectionError(
                f"Well {well_key_from_row_column(well.row, well.column)} contains no WellSample records."
            )

        used_sites: set[int] = set()
        projected: list[tuple[int, BioFormatsWellSample]] = []
        for ordinal, sample in enumerate(well.samples, start=1):
            site = ordinal
            if site in used_sites:
                raise BioFormatsProjectionError(
                    f"Duplicate WellSample site {site} in well "
                    f"{well_key_from_row_column(well.row, well.column)}."
                )
            used_sites.add(site)
            projected.append((site, sample))
        return tuple(projected)

    def _image_entries(
        self,
        image: BioFormatsImage,
        well: str,
        site: int,
    ) -> tuple[BioFormatsImageEntry, ...]:
        plane_by_coordinate = {
            (plane.c, plane.z, plane.t): plane
            for plane in image.pixels.planes
        }
        expected_plane_count = (
            image.pixels.size_c * image.pixels.size_z * image.pixels.size_t
        )
        if len(plane_by_coordinate) != expected_plane_count:
            raise BioFormatsProjectionError(
                f"Image {image.image_id!r} lacks a complete stable C/Z/T plane mapping."
            )

        entries: list[BioFormatsImageEntry] = []
        for timepoint in range(1, image.pixels.size_t + 1):
            for z_index in range(1, image.pixels.size_z + 1):
                for channel in range(1, image.pixels.size_c + 1):
                    plane = self._plane_for(
                        image,
                        plane_by_coordinate,
                        channel=channel,
                        z_index=z_index,
                        timepoint=timepoint,
                    )
                    entries.append(
                        BioFormatsImageEntry(
                            source_path=image.source_path,
                            series_index=image.series_index,
                            plane_index=plane.index,
                            source_files=image.source_files or (image.source_path,),
                            well=well,
                            site=site,
                            channel=channel,
                            z_index=z_index,
                            timepoint=timepoint,
                            source_channel=plane.c,
                            source_z_index=plane.z,
                            source_timepoint=plane.t,
                            reader=image.reader,
                            channel_name=_channel_name(image, channel),
                            pixel_size=image.pixel_size,
                        )
                    )
        return tuple(entries)

    def _plane_for(
        self,
        image: BioFormatsImage,
        plane_by_coordinate: dict[tuple[int, int, int], BioFormatsPlane],
        *,
        channel: int,
        z_index: int,
        timepoint: int,
    ) -> BioFormatsPlane:
        key = (channel, z_index, timepoint)
        try:
            return plane_by_coordinate[key]
        except KeyError as exc:
            raise BioFormatsProjectionError(
                f"Image {image.image_id!r} has no plane for "
                f"C={channel}, Z={z_index}, T={timepoint}."
            ) from exc


class BioFormatsLayoutProjector:
    """Project non-OME-SPW HCS filename-layout axes onto OpenHCS source axes."""

    def project(self, metadata: BioFormatsMetadata) -> BioFormatsDataset:
        entries = tuple(
            entry
            for image in metadata.images
            for entry in self._image_entries(image)
        )
        if not entries:
            raise BioFormatsProjectionError(
                "Bio-Formats metadata contains no supported HCS filename-layout axes."
            )
        return BioFormatsDataset(root=metadata.root, entries=entries)

    def _image_entries(self, image: BioFormatsImage) -> tuple[BioFormatsImageEntry, ...]:
        axes = image.layout_axes
        if axes is None:
            return ()
        return tuple(
            BioFormatsImageEntry(
                source_path=image.source_path,
                series_index=image.series_index,
                plane_index=plane.index,
                source_files=image.source_files or (image.source_path,),
                well=axes.well,
                site=axes.site,
                channel=_axis_coordinate(axes.channel, plane.c),
                z_index=_axis_coordinate(axes.z_index, plane.z),
                timepoint=_axis_coordinate(axes.timepoint, plane.t),
                source_channel=plane.c,
                source_z_index=plane.z,
                source_timepoint=plane.t,
                reader=image.reader,
                channel_name=_layout_channel_name(image, axes, plane),
                pixel_size=image.pixel_size,
            )
            for plane in image.pixels.planes
        )


def well_key_from_row_column(row: int | str, column: int | str) -> str:
    """Convert OME zero-based row/column coordinates to OpenHCS well keys."""

    try:
        return BIOFORMATS_WELL_KEYS.key_from_ome(row, column)
    except ValueError as exc:
        raise BioFormatsProjectionError(str(exc)) from exc


def _channel_name(image: BioFormatsImage, channel: int) -> str | None:
    if channel > len(image.channel_names):
        return None
    return image.channel_names[channel - 1]


def _axis_coordinate(base: int, local_coordinate: int) -> int:
    return base + local_coordinate - 1


def _layout_channel_name(
    image: BioFormatsImage,
    axes: BioFormatsLayoutAxes,
    plane: BioFormatsPlane,
) -> str | None:
    if axes.channel_name is not None:
        return axes.channel_name
    return _channel_name(image, plane.c)
