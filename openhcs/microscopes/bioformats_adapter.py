"""Bio-Formats store reader emitting generic addressable source planes."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
from metaclass_registry import AutoRegisterMeta
from ome_zarr.axes import Axes
from ome_zarr.format import format_from_version
import zarr
from polystore.bioformats_java import (
    BioFormatsJavaContext,
    BioFormatsJavaUnavailableError,
    java_float,
    java_int,
    java_str,
)
from polystore.bioformats_storage import BioFormatsPlaneRef
from polystore.ome_zarr_storage import OmeZarrArrayRef
from polystore.virtual_workspace import SourcePixelRef
from polystore.zarr_batch import ZarrStoredBatchSemantics

from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.image_file_serialization import ImageFileFormat
from openhcs.core.source_matching import with_source_component_metadata
from openhcs.core.source_projection import (
    OpenHCSPlaneAddress,
    SourceCandidate,
    SourceDatasetConflictError,
    SourceDatasetDiagnostic,
    SourceDatasetIdentity,
    SourcePlaneDataset,
    SourcePlaneStoreIdentity,
)
from openhcs.microscopes.bioformats_well_key import BIOFORMATS_WELL_KEYS


BIOFORMATS_MANIFEST_FILENAME = "bioformats_spw.json"


class BioFormatsAdapterUnavailableError(RuntimeError):
    """Raised when a Bio-Formats store cannot emit exact source planes."""


class BioFormatsDatasetAmbiguityError(BioFormatsAdapterUnavailableError):
    """Raised when Bio-Formats declarations cannot identify one exact dataset."""


@dataclass(frozen=True, slots=True)
class BioFormatsPackedRgbSeriesExclusion(SourceDatasetDiagnostic):
    """Exact packed-RGB reader series excluded from scalar-plane projection."""

    source_files: tuple[Path, ...]
    image_id: str
    image_name: str | None
    series_index: int
    rgb_channel_count: int

    def __post_init__(self) -> None:
        source_files = tuple(
            sorted(
                {Path(path).resolve(strict=False) for path in self.source_files},
                key=str,
            )
        )
        if not source_files:
            raise ValueError("Packed-RGB exclusion requires reader-declared files.")
        image_id = str(self.image_id).strip()
        if not image_id:
            raise ValueError("Packed-RGB exclusion requires an OME Image.ID.")
        if (
            not isinstance(self.series_index, int)
            or isinstance(self.series_index, bool)
            or self.series_index < 0
        ):
            raise TypeError(
                "Packed-RGB exclusion series_index must be a nonnegative integer."
            )
        if (
            not isinstance(self.rgb_channel_count, int)
            or isinstance(self.rgb_channel_count, bool)
            or self.rgb_channel_count <= 1
        ):
            raise TypeError(
                "Packed-RGB exclusion rgb_channel_count must exceed one."
            )
        object.__setattr__(self, "source_files", source_files)
        object.__setattr__(self, "image_id", image_id)

    @property
    def message(self) -> str:
        """Return actionable user-facing exclusion guidance."""

        label = (
            self.image_id
            if self.image_name is None
            else f"{self.image_id} ({self.image_name})"
        )
        return (
            f"Bio-Formats series {self.series_index} {label} declares "
            f"{self.rgb_channel_count} packed RGB channels and was excluded from "
            "OpenHCS scalar source planes. View or extract that series with an "
            "RGB-capable tool; do not treat its packed channels as microscopy "
            "channel planes."
        )

    def metadata_payload(self) -> Mapping[str, Any]:
        """Serialize exact exclusion identity without generic type dispatch."""

        return {
            "diagnostic_type": "bioformats_packed_rgb_series_exclusion",
            "image_id": self.image_id,
            "image_name": self.image_name,
            "message": self.message,
            "rgb_channel_count": self.rgb_channel_count,
            "series_index": self.series_index,
            "source_files": [str(path) for path in self.source_files],
        }


class BioFormatsNoScalarSourceError(BioFormatsAdapterUnavailableError):
    """Raised when a store exposes only typed non-scalar source exclusions."""

    def __init__(
        self,
        exclusions: tuple[BioFormatsPackedRgbSeriesExclusion, ...],
    ) -> None:
        exclusions = tuple(exclusions)
        if not exclusions:
            raise ValueError(
                "BioFormatsNoScalarSourceError requires typed exclusions."
            )
        self.exclusions = exclusions
        super().__init__(
            "Bio-Formats container exposes no scalar source planes. "
            + " ".join(exclusion.message for exclusion in exclusions)
        )


@dataclass(frozen=True, slots=True)
class BioFormatsPlane:
    """One exact OME C/Z/T coordinate and Bio-Formats plane index."""

    c: int
    z: int
    t: int
    index: int


@dataclass(frozen=True, slots=True)
class BioFormatsPixels:
    """Declared scalar axes for one OME Image/Pixels record."""

    size_c: int
    size_z: int
    size_t: int
    planes: tuple[BioFormatsPlane, ...]


@dataclass(frozen=True, slots=True)
class BioFormatsImage:
    """One reader series linked to an OME Image identity."""

    image_id: str
    image_name: str | None
    source_path: Path
    source_files: tuple[Path, ...]
    series_index: int
    pixels: BioFormatsPixels
    channel_names: tuple[str | None, ...]
    pixel_size: float
    reader: str = "bioformats"

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("OME Image.ID cannot be empty.")
        if not self.source_files:
            raise ValueError("Bio-Formats image requires reader-declared used files.")
        if self.pixel_size <= 0:
            raise ValueError("OME Pixels physical size must be positive.")
        expected = self.pixels.size_c * self.pixels.size_z * self.pixels.size_t
        if len(self.pixels.planes) != expected:
            raise ValueError(
                f"OME Image {self.image_id!r} lacks complete exact C/Z/T planes."
            )
        coordinates = {(plane.c, plane.z, plane.t) for plane in self.pixels.planes}
        reader_planes = {plane.index for plane in self.pixels.planes}
        if len(coordinates) != expected or len(reader_planes) != expected:
            raise ValueError(
                f"OME Image {self.image_id!r} has duplicate C/Z/T or reader planes."
            )
        if len(self.channel_names) != self.pixels.size_c:
            raise ValueError(
                f"OME Image {self.image_id!r} channel labels do not match SizeC."
            )


@dataclass(frozen=True, slots=True)
class BioFormatsWellSample:
    """Exact OME WellSample link to one OME Image."""

    sample_id: str
    image_id: str
    index: int


@dataclass(frozen=True, slots=True)
class BioFormatsWell:
    """Exact OME Well coordinates and samples."""

    well_id: str
    row: int
    column: int
    samples: tuple[BioFormatsWellSample, ...]


@dataclass(frozen=True, slots=True)
class BioFormatsPlate:
    """One explicitly identified OME Plate."""

    plate_id: str
    name: str | None
    wells: tuple[BioFormatsWell, ...]


@dataclass(frozen=True, slots=True)
class BioFormatsStoreMetadata:
    """Reader metadata for one physical Bio-Formats container."""

    root: Path
    images: tuple[BioFormatsImage, ...]
    plates: tuple[BioFormatsPlate, ...] = ()
    declared_dataset_id: str | None = None
    excluded_series: tuple[BioFormatsPackedRgbSeriesExclusion, ...] = ()

    def source_dataset(self) -> SourcePlaneDataset:
        """Emit exact generic source planes without filename interpretation."""

        if not self.images:
            if self.excluded_series:
                raise BioFormatsNoScalarSourceError(self.excluded_series)
            raise BioFormatsAdapterUnavailableError(
                "Bio-Formats metadata contains no OME Images."
            )
        if len(self.plates) > 1:
            plate_ids = tuple(plate.plate_id for plate in self.plates)
            raise BioFormatsDatasetAmbiguityError(
                "One Bio-Formats container declares multiple OME Plate.ID values "
                f"{plate_ids!r}. OpenHCS requires one embedded dataset identity per "
                "submitted plate root; extract or select one plate into each root "
                "before initialization. Source bindings select planes only after "
                "dataset identity is established."
            )
        image_by_id: dict[str, BioFormatsImage] = {}
        for image in self.images:
            if image.image_id in image_by_id:
                raise BioFormatsAdapterUnavailableError(
                    f"Duplicate OME Image.ID {image.image_id!r}."
                )
            image_by_id[image.image_id] = image

        if self.plates:
            plate = self.plates[0]
            dataset_identity = SourceDatasetIdentity(plate.plate_id)
            sample_groups = self._plate_sample_groups(plate, image_by_id)
        else:
            dataset_identity = (
                SourceDatasetIdentity(self.declared_dataset_id)
                if self.declared_dataset_id is not None
                else SourceDatasetIdentity.for_root(self.root)
            )
            sample_groups = tuple(
                (
                    image,
                    image.image_id,
                    OpenHCSPlaneAddress.component_token(
                        _relative_path(self.root, image.source_path)
                    ),
                    str(image.series_index + 1),
                    _relative_path(self.root, image.source_path),
                    image.image_name,
                )
                for image in self.images
            )

        candidates = tuple(
            candidate
            for image, sample_id, well, site, well_label, site_label in sample_groups
            for candidate in self._image_candidates(
                image,
                dataset_identity=dataset_identity,
                sample_id=sample_id,
                well=well,
                site=site,
                well_label=well_label,
                site_label=site_label,
            )
        )
        pixel_sizes = {image.pixel_size for image in self.images}
        if len(pixel_sizes) != 1:
            raise BioFormatsAdapterUnavailableError(
                f"Bio-Formats stores declare conflicting pixel sizes: {pixel_sizes!r}."
            )
        return SourcePlaneDataset(
            root=self.root,
            identity=dataset_identity,
            candidates=candidates,
            pixel_size=pixel_sizes.pop(),
            diagnostics=self.excluded_series,
        )

    def _plate_sample_groups(
        self,
        plate: BioFormatsPlate,
        image_by_id: Mapping[str, BioFormatsImage],
    ) -> tuple[tuple[BioFormatsImage, str, str, str, str, str | None], ...]:
        groups: list[
            tuple[BioFormatsImage, str, str, str, str, str | None]
        ] = []
        well_ids: set[str] = set()
        sample_ids: set[str] = set()
        referenced_images: set[str] = set()
        excluded_by_image_id = {
            exclusion.image_id: exclusion for exclusion in self.excluded_series
        }
        for well in plate.wells:
            if well.well_id in well_ids:
                raise BioFormatsAdapterUnavailableError(
                    f"Duplicate OME Well.ID {well.well_id!r}."
                )
            well_ids.add(well.well_id)
            well_key = BIOFORMATS_WELL_KEYS.key_from_one_based(
                well.row + 1,
                well.column + 1,
            )
            for sample in well.samples:
                if sample.sample_id in sample_ids:
                    raise BioFormatsAdapterUnavailableError(
                        f"Duplicate OME WellSample.ID {sample.sample_id!r}."
                    )
                if sample.image_id in referenced_images:
                    raise BioFormatsAdapterUnavailableError(
                        f"OME Image.ID {sample.image_id!r} has multiple WellSamples."
                    )
                try:
                    image = image_by_id[sample.image_id]
                except KeyError as exc:
                    exclusion = excluded_by_image_id.get(sample.image_id)
                    if exclusion is not None:
                        raise BioFormatsAdapterUnavailableError(
                            "OME WellSample "
                            f"{sample.sample_id!r} references a packed-RGB image that "
                            "cannot be excluded as ancillary. "
                            f"{exclusion.message}"
                        ) from exc
                    raise BioFormatsAdapterUnavailableError(
                        f"OME WellSample references missing Image.ID {sample.image_id!r}."
                    ) from exc
                sample_ids.add(sample.sample_id)
                referenced_images.add(sample.image_id)
                groups.append(
                    (
                        image,
                        sample.sample_id,
                        well_key,
                        str(sample.index + 1),
                        well_key,
                        None,
                    )
                )
        unreferenced = set(image_by_id).difference(referenced_images)
        if unreferenced:
            raise BioFormatsDatasetAmbiguityError(
                "OME Plate leaves scalar Images without WellSamples: "
                f"{sorted(unreferenced)!r}. Their well/site identity is undefined; "
                "repair the embedded OME links or submit those images as a separate "
                "non-plate dataset."
            )
        return tuple(groups)

    def _image_candidates(
        self,
        image: BioFormatsImage,
        *,
        dataset_identity: SourceDatasetIdentity,
        sample_id: str,
        well: str,
        site: str,
        well_label: str | None,
        site_label: str | None,
    ) -> tuple[SourceCandidate, ...]:
        container_paths = tuple(
            sorted(
                {path.resolve(strict=False) for path in image.source_files},
                key=str,
            )
        )
        canonical_source = container_paths[0]
        filter_paths = tuple(
            value
            for path in container_paths
            for value in _physical_path_identities(self.root, path)
        )
        candidates = []
        for plane in image.pixels.planes:
            address = OpenHCSPlaneAddress(
                well=well,
                site=site,
                channel=str(plane.c),
                z_index=str(plane.z),
                timepoint=str(plane.t),
            )
            metadata: dict[str, object] = {
                "ome_image_id": image.image_id,
                "ome_sample_id": sample_id,
            }
            if self.plates:
                metadata["ome_plate_id"] = self.plates[0].plate_id
            for component, value in address.component_values().items():
                metadata = with_source_component_metadata(metadata, component, value)
            if image.reader == "npy":
                source_ref = SourcePixelRef(
                    backend=Backend.DISK.value,
                    backend_address=_relative_path(self.root, canonical_source),
                    source_axis_indices=(plane.t - 1, plane.z - 1, plane.c - 1),
                )
                source_axis_shape = (
                    image.pixels.size_t,
                    image.pixels.size_z,
                    image.pixels.size_c,
                )
            else:
                source_ref = SourcePixelRef(
                    backend=Backend.BIOFORMATS.value,
                    backend_address=BioFormatsPlaneRef(
                        source_path=canonical_source,
                        series_index=image.series_index,
                        plane_index=plane.index,
                    ).to_backend_address(),
                )
                source_axis_shape = ()
            candidates.append(
                SourceCandidate(
                    source_ref=source_ref,
                    relative_path=_relative_path(self.root, canonical_source),
                    metadata=metadata,
                    source_axis_shape=source_axis_shape,
                    source_filter_paths=filter_paths,
                    component_labels={
                        AllComponents.WELL.value: well_label,
                        AllComponents.SITE.value: site_label,
                        AllComponents.CHANNEL.value: image.channel_names[plane.c - 1],
                        AllComponents.Z_INDEX.value: None,
                        AllComponents.TIMEPOINT.value: None,
                    },
                    declared_address=address,
                    dataset_identity=dataset_identity,
                    store_identity=SourcePlaneStoreIdentity(
                        container_paths=container_paths,
                        sample_group_id=sample_id,
                        image_id=image.image_id,
                        series_id=f"{image.image_id}:series:{image.series_index}",
                        plane_id=(
                            f"{image.image_id}:c{plane.c}:z{plane.z}:t{plane.t}"
                        ),
                    ),
                )
            )
        return tuple(candidates)


class SourcePlaneStoreAdapter(ABC, metaclass=AutoRegisterMeta):
    """Nominal store decoder emitting generic planes for one collection."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: ClassVar[str | None] = None

    @classmethod
    def claims_collection(cls, root: Path) -> bool:
        """Return whether this leaf exclusively owns the submitted collection."""
        del root
        return False

    @abstractmethod
    def discover_stores(self, root: Path) -> tuple[SourcePlaneDataset, ...]:
        """Decode every store owned by this leaf under one collection root."""

    @classmethod
    def discover_dataset(cls, root: str | Path) -> SourcePlaneDataset:
        root_path = Path(root).resolve(strict=False)
        if not root_path.exists():
            raise BioFormatsAdapterUnavailableError(
                f"Plane-store collection does not exist: {root_path}"
            )
        adapters = tuple(adapter_type() for adapter_type in cls.__registry__.values())
        collection_owners = tuple(
            adapter for adapter in adapters if adapter.claims_collection(root_path)
        )
        if len(collection_owners) > 1:
            raise BioFormatsAdapterUnavailableError(
                f"Multiple plane-store adapters claim {root_path}: "
                f"{tuple(type(adapter).__name__ for adapter in collection_owners)!r}."
            )
        datasets = tuple(
            dataset
            for adapter in (collection_owners or adapters)
            for dataset in adapter.discover_stores(root_path)
        )
        if not datasets:
            raise BioFormatsAdapterUnavailableError(
                f"No registered plane store declared sources under {root_path}."
            )
        try:
            return SourcePlaneDataset.aggregate(datasets)
        except SourceDatasetConflictError as exc:
            raise BioFormatsDatasetAmbiguityError(
                f"Cannot project {root_path} as one exact OpenHCS source dataset: "
                f"{exc} Keep distinct embedded plates in separate submitted roots, "
                "and repair colliding embedded well/site/channel/Z/time identities "
                "instead of namespacing them by filename."
            ) from exc
        except ValueError as exc:
            raise BioFormatsAdapterUnavailableError(str(exc)) from exc


class BioFormatsManifestAdapter(SourcePlaneStoreAdapter):
    """Deterministic explicit store declaration used by focused fixtures."""

    registry_key = "manifest"

    @classmethod
    def claims_collection(cls, root: Path) -> bool:
        return _manifest_path(root).is_file()

    def discover_stores(self, root: Path) -> tuple[SourcePlaneDataset, ...]:
        manifest_path = _manifest_path(root)
        if not manifest_path.is_file():
            return ()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (_metadata_from_mapping(root, payload).source_dataset(),)


class OmeZarrStoreAdapter(SourcePlaneStoreAdapter):
    """OME-NGFF store decoder using embedded plate and multiscale metadata."""

    registry_key = Backend.OME_ZARR.value

    @classmethod
    def declares_store(cls, path: Path) -> bool:
        """Return whether ``path`` is an explicit top-level NGFF image or plate."""
        if not path.is_dir() or not (path / ".zgroup").is_file():
            return False
        attrs = dict(zarr.open_group(str(path), mode="r").attrs)
        return "plate" in attrs or "multiscales" in attrs

    def discover_stores(self, root: Path) -> tuple[SourcePlaneDataset, ...]:
        return tuple(
            self._discover_store(root, store_root)
            for store_root in self._store_roots(root)
        )

    @classmethod
    def _store_roots(cls, root: Path) -> tuple[Path, ...]:
        if root.is_file():
            return ()
        group_roots = {root} | {
            marker.parent for marker in root.rglob(".zgroup")
        }
        declared = {
            path.resolve(strict=False)
            for path in group_roots
            if cls.declares_store(path)
        }
        return tuple(
            path
            for path in sorted(declared, key=str)
            if not any(parent in declared for parent in path.parents)
        )

    def _discover_store(
        self,
        collection_root: Path,
        store_root: Path,
    ) -> SourcePlaneDataset:
        group = zarr.open_group(str(store_root), mode="r")
        attrs = dict(group.attrs)
        if "plate" in attrs:
            return self._plate_dataset(
                collection_root,
                store_root,
                group,
                attrs["plate"],
            )
        return self._nonplate_dataset(
            collection_root,
            store_root,
            group,
            attrs["multiscales"],
        )

    def _plate_dataset(
        self,
        collection_root: Path,
        store_root: Path,
        group: Any,
        plate_payload: object,
    ) -> SourcePlaneDataset:
        plate = _required_mapping(plate_payload, "NGFF plate")
        plate_name = _required_text(plate, "name", "NGFF plate")
        identity = SourceDatasetIdentity(plate_name)
        rows = tuple(
            _required_text(
                _required_mapping(row, "NGFF plate row"),
                "name",
                "NGFF plate row",
            )
            for row in _required_sequence(plate, "rows", "NGFF plate")
        )
        columns = tuple(
            _required_text(
                _required_mapping(column, "NGFF plate column"),
                "name",
                "NGFF plate column",
            )
            for column in _required_sequence(plate, "columns", "NGFF plate")
        )
        candidates: list[SourceCandidate] = []
        pixel_sizes: set[float] = set()
        for well_payload in _required_sequence(plate, "wells", "NGFF plate"):
            well_record = _required_mapping(well_payload, "NGFF plate well")
            row_index = _required_index(well_record, "rowIndex", len(rows))
            column_index = _required_index(well_record, "columnIndex", len(columns))
            well_path = _required_text(well_record, "path", "NGFF plate well")
            expected_path = f"{rows[row_index]}/{columns[column_index]}"
            if well_path != expected_path:
                raise BioFormatsAdapterUnavailableError(
                    f"NGFF well path {well_path!r} conflicts with row/column identity "
                    f"{expected_path!r}."
                )
            well_group = group[well_path]
            well_attrs = dict(well_group.attrs)
            well = _required_mapping(well_attrs["well"], "NGFF well")
            images = _required_sequence(well, "images", "NGFF well")
            well_key = f"{rows[row_index]}{columns[column_index]}"
            for image_index, image_payload in enumerate(images):
                image = _required_mapping(image_payload, "NGFF well image")
                image_path = _required_text(image, "path", "NGFF well image")
                image_candidates, pixel_size = _ngff_image_candidates(
                    collection_root=collection_root,
                    store_root=store_root,
                    image_group=well_group[image_path],
                    image_prefix=f"{well_path}/{image_path}",
                    dataset_identity=identity,
                    well=well_key,
                    image_index=image_index,
                    image_count=len(images),
                )
                candidates.extend(image_candidates)
                pixel_sizes.add(pixel_size)
        if len(pixel_sizes) != 1:
            raise BioFormatsAdapterUnavailableError(
                f"NGFF images declare conflicting pixel sizes: {pixel_sizes!r}."
            )
        return SourcePlaneDataset(
            root=collection_root,
            identity=identity,
            candidates=tuple(candidates),
            pixel_size=pixel_sizes.pop(),
        )

    def _nonplate_dataset(
        self,
        collection_root: Path,
        store_root: Path,
        group: Any,
        multiscales_payload: object,
    ) -> SourcePlaneDataset:
        multiscales = _required_sequence_value(
            multiscales_payload,
            "NGFF multiscales",
        )
        if len(multiscales) != 1:
            raise BioFormatsAdapterUnavailableError(
                "Non-plate NGFF stores require one explicit multiscale image."
        )
        multiscale = _required_mapping(multiscales[0], "NGFF multiscale")
        _required_text(multiscale, "name", "NGFF multiscale")
        identity = SourceDatasetIdentity.for_root(collection_root)
        candidates, pixel_size = _ngff_image_candidates(
            collection_root=collection_root,
            store_root=store_root,
            image_group=group,
            image_prefix="",
            dataset_identity=identity,
            well=OpenHCSPlaneAddress.component_token(
                _relative_path(collection_root, store_root)
            ),
            image_index=0,
            image_count=1,
        )
        return SourcePlaneDataset(
            root=collection_root,
            identity=identity,
            candidates=candidates,
            pixel_size=pixel_size,
        )


class BioFormatsJavaAdapter(SourcePlaneStoreAdapter):
    """Java Bio-Formats decoder for positively identified rich containers."""

    registry_key = "java"

    def discover_stores(self, root: Path) -> tuple[SourcePlaneDataset, ...]:
        context = BioFormatsJavaContext.instance()
        datasets: list[SourcePlaneDataset] = []
        exclusions: list[BioFormatsPackedRgbSeriesExclusion] = []
        claimed_paths: set[Path] = set()
        for source_path in _candidate_source_paths(root):
            resolved_path = source_path.resolve(strict=False)
            if resolved_path in claimed_paths or not self._declares_path(
                context,
                source_path,
            ):
                continue
            try:
                dataset = self._discover_container(root, source_path)
            except BioFormatsNoScalarSourceError as exc:
                exclusions.extend(exc.exclusions)
                claimed_paths.update(
                    path
                    for exclusion in exc.exclusions
                    for path in exclusion.source_files
                )
                continue
            datasets.append(dataset)
            claimed_paths.update(
                path
                for candidate in dataset.candidates
                for path in candidate.store_identity.container_paths
            )
        if exclusions:
            if not datasets:
                raise BioFormatsNoScalarSourceError(tuple(exclusions))
            datasets[0] = replace(
                datasets[0],
                diagnostics=(*datasets[0].diagnostics, *exclusions),
            )
        return tuple(datasets)

    @staticmethod
    def _declares_path(context: BioFormatsJavaContext, path: Path) -> bool:
        if ImageFileFormat.is_image_path(path):
            image_format = ImageFileFormat.require_path(path)
            if not image_format.requires_plane_store_decoder(path):
                return False
        try:
            return context.declares_path(path)
        except BioFormatsJavaUnavailableError as exc:
            raise BioFormatsAdapterUnavailableError(str(exc)) from exc

    def _discover_container(
        self,
        root: Path,
        source_path: Path,
    ) -> SourcePlaneDataset:
        try:
            opened = BioFormatsJavaContext.instance().open_reader(source_path)
        except BioFormatsJavaUnavailableError as exc:
            raise BioFormatsAdapterUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise BioFormatsAdapterUnavailableError(
                f"Bio-Formats could not open {source_path}: {exc}"
            ) from exc
        try:
            images, excluded_series = _images_from_java(
                source_path,
                opened.reader,
                opened.metadata,
            )
            metadata = BioFormatsStoreMetadata(
                root=root,
                images=images,
                plates=_plates_from_java(opened.metadata),
                excluded_series=excluded_series,
            )
            return metadata.source_dataset()
        except SourceDatasetConflictError as exc:
            raise BioFormatsDatasetAmbiguityError(
                f"Bio-Formats container {source_path} declares conflicting source "
                f"identity: {exc} Repair the embedded metadata rather than assigning "
                "filename-derived coordinates."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise BioFormatsAdapterUnavailableError(str(exc)) from exc
        finally:
            opened.close()


class ImageFileStoreAdapter(SourcePlaneStoreAdapter):
    """Registered ordinary image files exposed as exact scalar planes."""

    registry_key = "image_file"

    def discover_stores(self, root: Path) -> tuple[SourcePlaneDataset, ...]:
        datasets = []
        identity = SourceDatasetIdentity.for_root(root)
        for source_path in _candidate_source_paths(root):
            if not ImageFileFormat.is_image_path(source_path):
                continue
            image_format = ImageFileFormat.require_path(source_path)
            if image_format.requires_plane_store_decoder(source_path):
                continue
            shape = tuple(int(size) for size in np.shape(image_format.read(source_path)))
            if len(shape) != 2:
                raise BioFormatsAdapterUnavailableError(
                    f"Ordinary image {source_path} must expose one scalar 2D plane; "
                    f"its declared shape is {shape!r}."
                )
            relative_path = _relative_path(root, source_path)
            sample_id = OpenHCSPlaneAddress.component_token(relative_path)
            address = OpenHCSPlaneAddress(sample_id, "1", "1", "1", "1")
            metadata: dict[str, object] = {}
            for component, value in address.component_values().items():
                metadata = with_source_component_metadata(metadata, component, value)
            candidate = SourceCandidate(
                source_ref=SourcePixelRef(
                    backend=Backend.DISK.value,
                    backend_address=relative_path,
                ),
                relative_path=relative_path,
                metadata=metadata,
                source_filter_paths=_physical_path_identities(root, source_path),
                component_labels={
                    AllComponents.WELL.value: relative_path,
                    AllComponents.SITE.value: None,
                    AllComponents.CHANNEL.value: None,
                    AllComponents.Z_INDEX.value: None,
                    AllComponents.TIMEPOINT.value: None,
                },
                declared_address=address,
                dataset_identity=identity,
                store_identity=SourcePlaneStoreIdentity(
                    container_paths=(source_path,),
                    sample_group_id=relative_path,
                    image_id=relative_path,
                    series_id=f"{relative_path}:series:0",
                    plane_id=f"{relative_path}:plane:0",
                ),
            )
            datasets.append(
                SourcePlaneDataset(
                    root=root,
                    identity=identity,
                    candidates=(candidate,),
                    pixel_size=1.0,
                )
            )
        return tuple(datasets)


def _images_from_java(
    source_path: Path,
    reader: Any,
    metadata: Any,
) -> tuple[
    tuple[BioFormatsImage, ...],
    tuple[BioFormatsPackedRgbSeriesExclusion, ...],
]:
    images = []
    excluded_series = []
    for image_index in range(int(metadata.getImageCount())):
        if image_index >= int(reader.getSeriesCount()):
            raise BioFormatsAdapterUnavailableError(
                "OME Image count exceeds Bio-Formats reader series count."
            )
        reader.setSeries(image_index)
        source_files = _series_used_files(reader, source_path)
        image_id = _required_str(metadata.getImageID(image_index), "Image.ID")
        image_name = java_str(metadata.getImageName(image_index))
        rgb_channel_count = int(reader.getRGBChannelCount())
        if rgb_channel_count != 1:
            excluded_series.append(
                BioFormatsPackedRgbSeriesExclusion(
                    source_files=source_files,
                    image_id=image_id,
                    image_name=image_name,
                    series_index=image_index,
                    rgb_channel_count=rgb_channel_count,
                )
            )
            continue
        size_c = _axis_size(metadata.getPixelsSizeC(image_index), reader.getSizeC())
        size_z = _axis_size(metadata.getPixelsSizeZ(image_index), reader.getSizeZ())
        size_t = _axis_size(metadata.getPixelsSizeT(image_index), reader.getSizeT())
        images.append(
            BioFormatsImage(
                image_id=image_id,
                image_name=image_name,
                source_path=source_files[0],
                source_files=source_files,
                series_index=image_index,
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
                pixel_size=_normalized_pixel_size(
                    metadata.getPixelsPhysicalSizeX(image_index)
                ),
            )
        )
    return tuple(images), tuple(excluded_series)


def _plates_from_java(metadata: Any) -> tuple[BioFormatsPlate, ...]:
    plates = []
    for plate_index in range(int(metadata.getPlateCount())):
        wells = []
        for well_index in range(int(metadata.getWellCount(plate_index))):
            samples = tuple(
                BioFormatsWellSample(
                    sample_id=_required_str(
                        metadata.getWellSampleID(
                            plate_index,
                            well_index,
                            sample_index,
                        ),
                        "WellSample.ID",
                    ),
                    image_id=_required_str(
                        metadata.getWellSampleImageRef(
                            plate_index,
                            well_index,
                            sample_index,
                        ),
                        "WellSample.ImageRef",
                    ),
                    index=_required_int(
                        metadata.getWellSampleIndex(
                            plate_index,
                            well_index,
                            sample_index,
                        ),
                        "WellSample.Index",
                    ),
                )
                for sample_index in range(
                    int(metadata.getWellSampleCount(plate_index, well_index))
                )
            )
            wells.append(
                BioFormatsWell(
                    well_id=_required_str(
                        metadata.getWellID(plate_index, well_index),
                        "Well.ID",
                    ),
                    row=_required_int(
                        metadata.getWellRow(plate_index, well_index),
                        "Well.Row",
                    ),
                    column=_required_int(
                        metadata.getWellColumn(plate_index, well_index),
                        "Well.Column",
                    ),
                    samples=samples,
                )
            )
        plates.append(
            BioFormatsPlate(
                plate_id=_required_str(metadata.getPlateID(plate_index), "Plate.ID"),
                name=java_str(metadata.getPlateName(plate_index)),
                wells=tuple(wells),
            )
        )
    return tuple(plates)


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
    plane_count = int(metadata.getPlaneCount(image_index))
    if plane_count:
        if plane_count != expected_count:
            raise BioFormatsAdapterUnavailableError(
                "OME Plane records do not cover the declared C/Z/T extent."
            )
        return tuple(
            BioFormatsPlane(
                c=_required_int(
                    metadata.getPlaneTheC(image_index, plane_index),
                    "Plane.TheC",
                )
                + 1,
                z=_required_int(
                    metadata.getPlaneTheZ(image_index, plane_index),
                    "Plane.TheZ",
                )
                + 1,
                t=_required_int(
                    metadata.getPlaneTheT(image_index, plane_index),
                    "Plane.TheT",
                )
                + 1,
                index=plane_index,
            )
            for plane_index in range(plane_count)
        )
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


def _metadata_from_mapping(
    root: Path,
    payload: Mapping[str, Any],
) -> BioFormatsStoreMetadata:
    plates = tuple(
        BioFormatsPlate(
            plate_id=str(plate["plate_id"]),
            name=(
                None
                if "name" not in plate or plate["name"] is None
                else str(plate["name"])
            ),
            wells=tuple(
                BioFormatsWell(
                    well_id=str(well["well_id"]),
                    row=int(well["row"]),
                    column=int(well["column"]),
                    samples=tuple(
                        BioFormatsWellSample(
                            sample_id=str(sample["sample_id"]),
                            image_id=str(sample["image_id"]),
                            index=int(sample["index"]),
                        )
                        for sample in well["samples"]
                    ),
                )
                for well in plate["wells"]
            ),
        )
        for plate in payload["plates"]
    )
    images = tuple(
        _image_from_mapping(root, image)
        for image in payload["images"]
    )
    declared_dataset_id = payload["dataset_id"] if "dataset_id" in payload else None
    return BioFormatsStoreMetadata(
        root=root,
        plates=plates,
        images=images,
        declared_dataset_id=(
            None if declared_dataset_id is None else str(declared_dataset_id)
        ),
    )


def _image_from_mapping(
    root: Path,
    payload: Mapping[str, Any],
) -> BioFormatsImage:
    source_path = _absolute_path(root, str(payload["source_path"]))
    source_file_values = (
        payload["source_files"]
        if "source_files" in payload
        else (payload["source_path"],)
    )
    source_files = tuple(
        _absolute_path(root, str(value))
        for value in source_file_values
    )
    pixels = payload["pixels"]
    return BioFormatsImage(
        image_id=str(payload["image_id"]),
        image_name=(
            None
            if "image_name" not in payload or payload["image_name"] is None
            else str(payload["image_name"])
        ),
        source_path=source_path,
        source_files=source_files,
        series_index=int(payload["series_index"]),
        reader=str(payload["reader"] if "reader" in payload else "bioformats"),
        channel_names=tuple(
            None if value is None else str(value)
            for value in payload["channel_names"]
        ),
        pixel_size=float(payload["pixel_size"]),
        pixels=BioFormatsPixels(
            size_c=int(pixels["size_c"]),
            size_z=int(pixels["size_z"]),
            size_t=int(pixels["size_t"]),
            planes=tuple(
                BioFormatsPlane(
                    c=int(plane["c"]),
                    z=int(plane["z"]),
                    t=int(plane["t"]),
                    index=int(plane["index"]),
                )
                for plane in pixels["planes"]
            ),
        ),
    )


def _ngff_image_candidates(
    *,
    collection_root: Path,
    store_root: Path,
    image_group: Any,
    image_prefix: str,
    dataset_identity: SourceDatasetIdentity,
    well: str,
    image_index: int,
    image_count: int,
) -> tuple[tuple[SourceCandidate, ...], float]:
    attrs = dict(image_group.attrs)
    multiscales = _required_sequence(attrs, "multiscales", "NGFF image")
    if len(multiscales) != 1:
        raise BioFormatsAdapterUnavailableError(
            "NGFF image requires exactly one multiscale declaration."
        )
    multiscale = _required_mapping(multiscales[0], "NGFF multiscale")
    image_id = _required_text(multiscale, "name", "NGFF multiscale")
    datasets = _required_sequence(multiscale, "datasets", "NGFF multiscale")
    if len(datasets) != 1:
        raise BioFormatsAdapterUnavailableError(
            "NGFF source projection requires exactly one declared resolution."
        )
    dataset = _required_mapping(datasets[0], "NGFF multiscale dataset")
    dataset_path = _required_text(dataset, "path", "NGFF multiscale dataset")
    array_path = "/".join(
        part for part in (image_prefix.strip("/"), dataset_path) if part
    )
    array = image_group[dataset_path]
    stored_batch_semantics = ZarrStoredBatchSemantics.from_attrs(dict(array.attrs))
    shape = tuple(int(size) for size in array.shape)
    axes_payload = _required_sequence(multiscale, "axes", "NGFF multiscale")
    axes = tuple(
        _required_text(
            _required_mapping(axis, "NGFF axis"),
            "name",
            "NGFF axis",
        )
        for axis in axes_payload
    )
    try:
        Axes(
            [dict(_required_mapping(axis, "NGFF axis")) for axis in axes_payload],
            format_from_version(
                _required_text(multiscale, "version", "NGFF multiscale")
            ),
        )
    except ValueError as exc:
        raise BioFormatsAdapterUnavailableError(
            f"Invalid NGFF axes for image {image_id!r}: {exc}"
        ) from exc
    if len(axes) != len(shape) or len(set(axes)) != len(axes):
        raise BioFormatsAdapterUnavailableError(
            "NGFF axes must be unique and match the array rank."
        )
    if axes[-2:] != ("y", "x"):
        raise BioFormatsAdapterUnavailableError(
            "NGFF source arrays must declare trailing y/x spatial axes."
        )
    leading_axes = axes[:-2]
    source_axis_shape = shape[:-2]
    channel_count = shape[axes.index("c")] if "c" in axes else 1
    channel_labels = _ngff_channel_labels(attrs, channel_count)
    pixel_size = _ngff_pixel_size(dataset, len(axes))
    container_paths = (store_root.resolve(strict=False),)
    filter_paths = _physical_path_identities(collection_root, store_root)
    candidates: list[SourceCandidate] = []
    for source_axis_indices in product(
        *(range(size) for size in source_axis_shape)
    ):
        coordinates = {
            axis: stored_batch_semantics.array_axis_value(
                axis,
                source_axis_indices[index],
                str(source_axis_indices[index] + 1),
            )
            for index, axis in enumerate(leading_axes)
        }
        site = stored_batch_semantics.image_axis_value(
            "field",
            str(coordinates.pop("field", image_index + 1 if image_count > 1 else 1)),
        )
        channel = coordinates.pop("c", 1)
        channel_index = (
            source_axis_indices[leading_axes.index("c")]
            if "c" in leading_axes
            else 0
        )
        z_index = coordinates.pop("z", 1)
        timepoint = coordinates.pop("t", 1)
        if coordinates:
            raise BioFormatsAdapterUnavailableError(
                "NGFF image declares unsupported nonspatial axes "
                f"{tuple(coordinates)!r}."
            )
        address = OpenHCSPlaneAddress(
            well=well,
            site=str(site),
            channel=str(channel),
            z_index=str(z_index),
            timepoint=str(timepoint),
        )
        metadata: dict[str, object] = {
            "ngff_dataset_id": dataset_identity.value,
            "ngff_image_id": image_id,
        }
        for component, value in address.component_values().items():
            metadata = with_source_component_metadata(metadata, component, value)
        field_image_id = f"{image_id}:field:{site}"
        candidates.append(
            SourceCandidate(
                source_ref=SourcePixelRef(
                    backend=Backend.OME_ZARR.value,
                    backend_address=OmeZarrArrayRef(
                        store_path=store_root,
                        array_path=array_path,
                    ).to_backend_address(),
                    source_axis_indices=source_axis_indices,
                ),
                relative_path=_relative_path(collection_root, store_root),
                metadata=metadata,
                source_axis_shape=source_axis_shape,
                source_filter_paths=filter_paths,
                component_labels={
                    AllComponents.WELL.value: well,
                    AllComponents.SITE.value: None,
                    AllComponents.CHANNEL.value: channel_labels[channel_index],
                    AllComponents.Z_INDEX.value: None,
                    AllComponents.TIMEPOINT.value: None,
                },
                declared_address=address,
                dataset_identity=dataset_identity,
                store_identity=SourcePlaneStoreIdentity(
                    container_paths=container_paths,
                    sample_group_id=field_image_id,
                    image_id=field_image_id,
                    series_id=f"{array_path}:field:{site}",
                    plane_id=(
                        f"{array_path}:"
                        + ":".join(str(index) for index in source_axis_indices)
                    ),
                ),
            )
        )
    return tuple(candidates), pixel_size


def _ngff_channel_labels(
    attrs: Mapping[str, object],
    channel_count: int,
) -> tuple[str | None, ...]:
    if "omero" not in attrs:
        return (None,) * channel_count
    omero = _required_mapping(attrs["omero"], "NGFF omero")
    channels = _required_sequence(omero, "channels", "NGFF omero")
    if len(channels) != channel_count:
        raise BioFormatsAdapterUnavailableError(
            "NGFF omero channel labels do not match the c-axis extent."
        )
    return tuple(
        _required_text(
            _required_mapping(channel, "NGFF omero channel"),
            "label",
            "NGFF omero channel",
        )
        for channel in channels
    )


def _ngff_pixel_size(dataset: Mapping[str, object], axis_count: int) -> float:
    transforms = _required_sequence(
        dataset,
        "coordinateTransformations",
        "NGFF multiscale dataset",
    )
    if len(transforms) != 1:
        raise BioFormatsAdapterUnavailableError(
            "NGFF source projection requires one scale transformation."
        )
    transform = _required_mapping(transforms[0], "NGFF coordinate transform")
    if _required_text(transform, "type", "NGFF coordinate transform") != "scale":
        raise BioFormatsAdapterUnavailableError(
            "NGFF source projection requires a scale transformation."
        )
    scale = _required_sequence(transform, "scale", "NGFF coordinate transform")
    if len(scale) != axis_count:
        raise BioFormatsAdapterUnavailableError(
            "NGFF scale vector must match the declared axes."
        )
    pixel_size = float(scale[-1])
    if float(scale[-2]) != pixel_size:
        raise BioFormatsAdapterUnavailableError(
            "NGFF source projection requires equal y/x pixel sizes."
        )
    if pixel_size <= 0:
        raise BioFormatsAdapterUnavailableError("NGFF pixel size must be positive.")
    return pixel_size


def _required_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BioFormatsAdapterUnavailableError(f"{context} must be a mapping.")
    return value


def _required_sequence(
    payload: Mapping[str, object],
    field: str,
    context: str,
) -> tuple[object, ...]:
    if field not in payload:
        raise BioFormatsAdapterUnavailableError(f"{context} missing {field}.")
    return _required_sequence_value(payload[field], f"{context}.{field}")


def _required_sequence_value(value: object, context: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise BioFormatsAdapterUnavailableError(f"{context} must be a sequence.")
    return tuple(value)


def _required_text(
    payload: Mapping[str, object],
    field: str,
    context: str,
) -> str:
    if field not in payload or not isinstance(payload[field], str):
        raise BioFormatsAdapterUnavailableError(f"{context} missing text {field}.")
    value = str(payload[field]).strip()
    if not value:
        raise BioFormatsAdapterUnavailableError(f"{context}.{field} cannot be empty.")
    return value


def _required_index(
    payload: Mapping[str, object],
    field: str,
    extent: int,
) -> int:
    if field not in payload or not isinstance(payload[field], int) or isinstance(
        payload[field],
        bool,
    ):
        raise BioFormatsAdapterUnavailableError(f"NGFF well missing integer {field}.")
    value = int(payload[field])
    if value < 0 or value >= extent:
        raise BioFormatsAdapterUnavailableError(
            f"NGFF well {field}={value} exceeds extent {extent}."
        )
    return value


def _candidate_source_paths(root: Path) -> tuple[Path, ...]:
    if root.is_file():
        return (root,)
    if not root.is_dir():
        raise BioFormatsAdapterUnavailableError(
            f"Bio-Formats path does not exist: {root}"
        )
    return tuple(path for path in sorted(root.rglob("*")) if path.is_file())


def _manifest_path(root: Path) -> Path:
    return (
        root.parent / BIOFORMATS_MANIFEST_FILENAME
        if root.is_file()
        else root / BIOFORMATS_MANIFEST_FILENAME
    )


def _series_used_files(reader: Any, source_path: Path) -> tuple[Path, ...]:
    try:
        files = reader.getSeriesUsedFiles(False)
    except Exception as exc:
        raise BioFormatsAdapterUnavailableError(
            f"Bio-Formats did not declare used files for {source_path}."
        ) from exc
    paths = tuple(
        _absolute_path(source_path.parent, str(value))
        for value in (() if files is None else files)
        if str(value)
    )
    if not paths:
        raise BioFormatsAdapterUnavailableError(
            f"Bio-Formats declared no used files for {source_path}."
        )
    return paths

def _physical_path_identities(root: Path, source_path: Path) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                _relative_path(root, source_path),
                str(source_path.resolve(strict=False)),
            )
        )
    )


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(
        root.resolve(strict=False)
    ).as_posix()


def _absolute_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _axis_size(metadata_value: Any, reader_value: Any) -> int:
    value = java_int(metadata_value)
    size = int(reader_value if value is None else value)
    if size <= 0:
        raise BioFormatsAdapterUnavailableError("OME axis size must be positive.")
    return size


def _required_int(value: Any, field_name: str) -> int:
    converted = java_int(value)
    if converted is None:
        raise BioFormatsAdapterUnavailableError(f"OME metadata missing {field_name}.")
    return converted


def _normalized_pixel_size(value: Any) -> float:
    """Normalize uncalibrated OME pixel coordinates to explicit unit spacing."""

    converted = java_float(value)
    if converted is None:
        return 1.0
    if converted <= 0:
        raise BioFormatsAdapterUnavailableError(
            "OME Pixels.PhysicalSizeX must be positive when declared."
        )
    return converted


def _required_str(value: Any, field_name: str) -> str:
    converted = java_str(value)
    if converted is None or not converted.strip():
        raise BioFormatsAdapterUnavailableError(f"OME metadata missing {field_name}.")
    return converted
