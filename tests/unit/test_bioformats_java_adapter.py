from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from openhcs.constants.constants import Backend
from openhcs.microscopes.bioformats_adapter import (
    BioFormatsContainerOpenError,
    BioFormatsDatasetAmbiguityError,
    BioFormatsNoScalarSourceError,
    BioFormatsPackedRgbSeriesExclusion,
    SourcePlaneStoreAdapter,
)
from polystore.bioformats_java import BioFormatsOpenedReader, BioFormatsJavaContext
from polystore.bioformats_java import sample_bioformats_plane
from polystore.base import ImageSamplingRequest


class JavaValue:
    def __init__(self, value):
        self._value = value

    def getValue(self):
        return self._value


class PhysicalSize:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


@dataclass
class FakeBioFormatsMetadata:
    plate_name: str = "plate"
    plate_id: str = "Plate:0"
    well_id: str = "Well:0:0"
    well_row: int = 0
    well_column: int = 0
    sample_id: str = "WellSample:0:0:0"
    sample_index: int = 0
    image_id: str = "Image:0"
    channel_names: tuple[str, str] = ("DAPI", "GFP")
    pixel_size: float | None = 0.65

    def getPlateCount(self):
        return 1

    def getPlateName(self, plate):
        return self.plate_name

    def getPlateID(self, plate):
        return self.plate_id

    def getWellCount(self, plate):
        return 1

    def getWellID(self, plate, well):
        return self.well_id

    def getWellRow(self, plate, well):
        return JavaValue(self.well_row)

    def getWellColumn(self, plate, well):
        return JavaValue(self.well_column)

    def getWellSampleCount(self, plate, well):
        return 1

    def getWellSampleImageRef(self, plate, well, sample):
        return self.image_id

    def getWellSampleID(self, plate, well, sample):
        return self.sample_id

    def getWellSampleIndex(self, plate, well, sample):
        return JavaValue(self.sample_index)

    def getImageCount(self):
        return 1

    def getImageID(self, image):
        return self.image_id

    def getImageName(self, image):
        return "A01 field"

    def getPixelsSizeC(self, image):
        return JavaValue(2)

    def getPixelsSizeZ(self, image):
        return JavaValue(1)

    def getPixelsSizeT(self, image):
        return JavaValue(1)

    def getPlaneCount(self, image):
        return 0

    def getChannelName(self, image, channel):
        return self.channel_names[channel]

    def getPixelsPhysicalSizeX(self, image):
        return None if self.pixel_size is None else PhysicalSize(self.pixel_size)


class FakeBioFormatsReader:
    def __init__(self, used_files=("plate.fake", "well-a01.tif")):
        self.series = None
        self.used_files = tuple(used_files)

    def getSeriesCount(self):
        return 1

    def setSeries(self, series):
        self.series = series

    def getSizeC(self):
        return 2

    def getSizeZ(self):
        return 1

    def getSizeT(self):
        return 1

    def getRGBChannelCount(self):
        return 1

    def getIndex(self, z_index, channel, timepoint):
        return channel

    def getSeriesUsedFiles(self, no_pixels):
        return self.used_files

    def close(self):
        pass


class FakeBioFormatsContext:
    def __init__(
        self,
        metadata_by_name=None,
        declared_suffixes=(".fake",),
    ):
        self.metadata_by_name = dict(metadata_by_name or {})
        self.declared_suffixes = tuple(declared_suffixes)

    def declares_path(self, source_path):
        return Path(source_path).name.endswith(self.declared_suffixes)

    def open_reader(self, source_path):
        source = Path(source_path)
        metadata = self.metadata_by_name.get(
            source.name,
            FakeBioFormatsMetadata(),
        )
        used_files = (
            (source.name,) if self.metadata_by_name else ("plate.fake", "well-a01.tif")
        )
        return BioFormatsOpenedReader(
            reader=FakeBioFormatsReader(used_files),
            metadata=metadata,
        )


def test_java_adapter_collapses_an_exact_nested_entrypoint_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root_entrypoint = tmp_path / "plate.HTD"
    nested_entrypoint = tmp_path / "TimePoint_1" / "plate.HTD"
    nested_entrypoint.parent.mkdir()
    root_entrypoint.write_bytes(b"identical container declaration")
    nested_entrypoint.write_bytes(root_entrypoint.read_bytes())
    opened: list[Path] = []
    context = FakeBioFormatsContext(declared_suffixes=(".HTD",))

    def open_reader(source_path: Path) -> BioFormatsOpenedReader:
        source_path = Path(source_path)
        opened.append(source_path)
        if source_path == nested_entrypoint:
            raise RuntimeError("nested copy cannot resolve its relative image paths")
        return BioFormatsOpenedReader(
            reader=FakeBioFormatsReader(
                (
                    str(root_entrypoint),
                    str(nested_entrypoint.parent / "well-a01.tif"),
                )
            ),
            metadata=FakeBioFormatsMetadata(),
        )

    monkeypatch.setattr(
        context,
        "open_reader",
        open_reader,
    )
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: context),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert len(dataset.candidates) == 2
    assert opened == [root_entrypoint, nested_entrypoint]


def test_java_adapter_retains_identical_nested_entrypoints_that_both_open(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root_entrypoint = tmp_path / "plate.HTD"
    nested_entrypoint = tmp_path / "TimePoint_1" / "plate.HTD"
    nested_entrypoint.parent.mkdir()
    root_entrypoint.write_bytes(b"identical container declaration")
    nested_entrypoint.write_bytes(root_entrypoint.read_bytes())
    opened: list[Path] = []
    context = FakeBioFormatsContext(declared_suffixes=(".HTD",))

    def open_reader(source_path: Path) -> BioFormatsOpenedReader:
        source_path = Path(source_path)
        opened.append(source_path)
        nested = source_path.parent == nested_entrypoint.parent
        metadata = FakeBioFormatsMetadata(
            well_id=f"Well:0:{1 if nested else 0}",
            well_column=1 if nested else 0,
            sample_id=f"WellSample:{'nested' if nested else 'root'}",
            image_id=f"Image:{'nested' if nested else 'root'}",
        )
        return BioFormatsOpenedReader(
            reader=FakeBioFormatsReader((str(source_path),)),
            metadata=metadata,
        )

    monkeypatch.setattr(context, "open_reader", open_reader)
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: context),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert len(dataset.candidates) == 4
    assert opened == [root_entrypoint, nested_entrypoint]


def test_java_adapter_does_not_hide_an_unclaimed_nested_open_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root_entrypoint = tmp_path / "plate.HTD"
    nested_entrypoint = tmp_path / "TimePoint_1" / "plate.HTD"
    nested_entrypoint.parent.mkdir()
    root_entrypoint.write_bytes(b"identical container declaration")
    nested_entrypoint.write_bytes(root_entrypoint.read_bytes())
    context = FakeBioFormatsContext(declared_suffixes=(".HTD",))

    def open_reader(source_path: Path) -> BioFormatsOpenedReader:
        source_path = Path(source_path)
        if source_path == nested_entrypoint:
            raise RuntimeError("nested source cannot open")
        return BioFormatsOpenedReader(
            reader=FakeBioFormatsReader((str(root_entrypoint),)),
            metadata=FakeBioFormatsMetadata(),
        )

    monkeypatch.setattr(context, "open_reader", open_reader)
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: context),
    )

    with pytest.raises(
        BioFormatsContainerOpenError,
        match="Bio-Formats could not open",
    ):
        SourcePlaneStoreAdapter.discover_dataset(tmp_path)


def test_java_context_preserves_native_multiresolution_series() -> None:
    events = []

    class _Reader:
        def setFlattenedResolutions(self, flattened):
            events.append(("flattened", flattened))

        def setMetadataStore(self, metadata):
            events.append(("metadata", metadata))

        def setId(self, source_path):
            events.append(("source", source_path))

        def close(self):
            events.append(("close", None))

    metadata = object()
    context = BioFormatsJavaContext(imagej_module=None, scyjava_module=None)
    context.ij = object()
    context.ImageReader = _Reader
    context.MetadataTools = type(
        "_MetadataTools",
        (),
        {"createOMEXMLMetadata": staticmethod(lambda: metadata)},
    )

    opened = context.open_reader("plate.czi")

    assert opened.metadata is metadata
    assert events == [
        ("flattened", False),
        ("metadata", metadata),
        ("source", "plate.czi"),
    ]


class _SamplingReader:
    def __init__(self, resolution_shapes):
        self.resolution_shapes = tuple(resolution_shapes)
        self.resolution_index = 0
        self.open_calls = []
        self.closed = False

    def setSeries(self, series_index):
        assert series_index == 2

    def getRGBChannelCount(self):
        return 1

    def getResolutionCount(self):
        return len(self.resolution_shapes)

    def setResolution(self, resolution_index):
        self.resolution_index = resolution_index

    def getSizeY(self):
        return self.resolution_shapes[self.resolution_index][0]

    def getSizeX(self):
        return self.resolution_shapes[self.resolution_index][1]

    def getPixelType(self):
        return 2

    def isLittleEndian(self):
        return True

    def openBytes(self, plane_index, x, y, width, height):
        self.open_calls.append((plane_index, x, y, width, height))
        return np.arange(width * height, dtype="<u2").tobytes()

    def close(self):
        self.closed = True


def _sampling_context(reader):
    context = SimpleNamespace(
        FormatTools=SimpleNamespace(
            INT8=0,
            UINT8=1,
            UINT16=2,
            INT16=3,
            INT32=4,
            UINT32=5,
            FLOAT=6,
            DOUBLE=7,
        ),
        open_reader=lambda _source_path: BioFormatsOpenedReader(
            reader=reader,
            metadata=object(),
        ),
    )
    return context


def test_java_sampling_auto_selects_bounded_native_pyramid_level(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reader = _SamplingReader(
        ((40_000, 32_000), (10_000, 8_000), (2_500, 2_000), (625, 500))
    )
    context = _sampling_context(reader)
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: context),
    )

    sampled = sample_bioformats_plane(
        source_path=tmp_path / "large.czi",
        series_index=2,
        plane_index=7,
        request=ImageSamplingRequest(origin_yx=(11, 13), shape_yx=(8, 9)),
    )

    assert sampled.source_shape == (40_000, 32_000)
    assert sampled.resolution_shape == (625, 500)
    assert sampled.selected_resolution_index == 3
    assert sampled.resolution_count == 4
    assert sampled.downsample_yx == (64.0, 64.0)
    assert sampled.data.shape == (8, 9)
    assert reader.open_calls == [(7, 13, 11, 9, 8)]
    assert reader.closed is True


def test_java_sampling_explicit_full_resolution_uses_only_bounded_open_bytes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reader = _SamplingReader(((40_000, 32_000), (625, 500)))
    context = _sampling_context(reader)
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: context),
    )

    sampled = sample_bioformats_plane(
        source_path=tmp_path / "large.czi",
        series_index=2,
        plane_index=7,
        request=ImageSamplingRequest(
            origin_yx=(101, 202),
            shape_yx=(4, 5),
            resolution_index=0,
        ),
    )

    assert sampled.selected_resolution_index == 0
    assert sampled.resolution_shape == (40_000, 32_000)
    assert sampled.data.shape == (4, 5)
    assert reader.open_calls == [(7, 202, 101, 5, 4)]


def test_java_sampling_nonpyramidal_source_is_bounded_and_invalid_level_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reader = _SamplingReader(((50_000, 40_000),))
    context = _sampling_context(reader)
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: context),
    )

    sampled = sample_bioformats_plane(
        source_path=tmp_path / "large.ome.tif",
        series_index=2,
        plane_index=7,
        request=ImageSamplingRequest(shape_yx=(6, 7)),
    )

    assert sampled.selected_resolution_index == 0
    assert sampled.data.shape == (6, 7)
    assert reader.open_calls == [(7, 0, 0, 7, 6)]

    with pytest.raises(ValueError, match="native resolution range 0..0"):
        sample_bioformats_plane(
            source_path=tmp_path / "large.ome.tif",
            series_index=2,
            plane_index=7,
            request=ImageSamplingRequest(resolution_index=1),
        )
    assert reader.open_calls == [(7, 0, 0, 7, 6)]


def test_java_adapter_retains_typed_packed_rgb_ancillary_exclusion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _ScalarAndRgbMetadata(FakeBioFormatsMetadata):
        def getPlateCount(self):
            return 0

        def getImageCount(self):
            return 2

        def getImageID(self, image):
            return ("Image:scalar", "Image:rgb")[image]

        def getImageName(self, image):
            return ("ScanRegion0", "label image")[image]

        def getPixelsSizeC(self, image):
            return JavaValue((2, 3)[image])

        def getChannelName(self, image, channel):
            return (("DAPI", "GFP"), ("Red", "Green", "Blue"))[image][channel]

        def getPixelsPhysicalSizeX(self, image):
            return PhysicalSize(0.65) if image == 0 else None

    class _ScalarAndRgbReader(FakeBioFormatsReader):
        def getSeriesCount(self):
            return 2

        def getSizeC(self):
            return (2, 3)[self.series]

        def getRGBChannelCount(self):
            return (1, 3)[self.series]

    source_path = tmp_path / "plate.czi"
    source_path.write_bytes(b"")
    context = FakeBioFormatsContext(declared_suffixes=(".czi",))
    monkeypatch.setattr(
        context,
        "open_reader",
        lambda source: BioFormatsOpenedReader(
            reader=_ScalarAndRgbReader((Path(source).name,)),
            metadata=_ScalarAndRgbMetadata(),
        ),
    )
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: context),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert dataset.pixel_size == 0.65
    assert len(dataset.candidates) == 2
    assert {candidate.store_identity.image_id for candidate in dataset.candidates} == {
        "Image:scalar"
    }
    assert {candidate.component_labels["well"] for candidate in dataset.candidates} == {
        "plate.czi"
    }
    assert {candidate.component_labels["site"] for candidate in dataset.candidates} == {
        "ScanRegion0"
    }
    assert len(dataset.diagnostics) == 1
    exclusion = dataset.diagnostics[0]
    assert isinstance(exclusion, BioFormatsPackedRgbSeriesExclusion)
    assert exclusion.image_id == "Image:rgb"
    assert exclusion.image_name == "label image"
    assert exclusion.series_index == 1
    assert exclusion.rgb_channel_count == 3
    assert {path.name for path in exclusion.source_files} == {source_path.name}
    assert "excluded from OpenHCS scalar source planes" in exclusion.message
    assert exclusion.metadata_payload()["diagnostic_type"] == (
        "bioformats_packed_rgb_series_exclusion"
    )


def test_java_adapter_retains_all_rgb_container_exclusion_with_scalar_container(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _NonPlateMetadata(FakeBioFormatsMetadata):
        def getPlateCount(self):
            return 0

    class _RgbMetadata(_NonPlateMetadata):
        def getPixelsSizeC(self, image):
            return JavaValue(3)

        def getChannelName(self, image, channel):
            return ("Red", "Green", "Blue")[channel]

    class _RgbReader(FakeBioFormatsReader):
        def getSizeC(self):
            return 3

        def getRGBChannelCount(self):
            return 3

    class _Context:
        def declares_path(self, source_path):
            return Path(source_path).suffix == ".czi"

        def open_reader(self, source_path):
            source = Path(source_path)
            if source.name == "rgb-only.czi":
                return BioFormatsOpenedReader(
                    reader=_RgbReader((source.name,)),
                    metadata=_RgbMetadata(image_id="Image:rgb"),
                )
            return BioFormatsOpenedReader(
                reader=FakeBioFormatsReader((source.name,)),
                metadata=_NonPlateMetadata(image_id="Image:scalar"),
            )

    for name in ("rgb-only.czi", "scalar.czi"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: _Context()),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert {candidate.store_identity.image_id for candidate in dataset.candidates} == {
        "Image:scalar"
    }
    assert len(dataset.diagnostics) == 1
    exclusion = dataset.diagnostics[0]
    assert isinstance(exclusion, BioFormatsPackedRgbSeriesExclusion)
    assert exclusion.image_id == "Image:rgb"
    assert {path.name for path in exclusion.source_files} == {"rgb-only.czi"}


def test_java_adapter_rejects_all_rgb_collection_with_typed_diagnostics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _RgbMetadata(FakeBioFormatsMetadata):
        def getPlateCount(self):
            return 0

        def getPixelsSizeC(self, image):
            return JavaValue(3)

    class _RgbReader(FakeBioFormatsReader):
        def getSizeC(self):
            return 3

        def getRGBChannelCount(self):
            return 3

    source_path = tmp_path / "rgb-only.czi"
    source_path.write_bytes(b"")
    context = FakeBioFormatsContext(declared_suffixes=(".czi",))
    monkeypatch.setattr(
        context,
        "open_reader",
        lambda source: BioFormatsOpenedReader(
            reader=_RgbReader((Path(source).name,)),
            metadata=_RgbMetadata(image_id="Image:rgb"),
        ),
    )
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: context),
    )

    with pytest.raises(BioFormatsNoScalarSourceError) as exc_info:
        SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert len(exc_info.value.exclusions) == 1
    assert exc_info.value.exclusions[0].image_id == "Image:rgb"
    assert "View or extract that series" in str(exc_info.value)


def test_java_adapter_projects_one_czi_ome_spw_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "plate.czi"
    source_path.write_bytes(b"")
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(
            lambda cls: FakeBioFormatsContext(
                {source_path.name: FakeBioFormatsMetadata()},
                declared_suffixes=(".czi",),
            )
        ),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert dataset.identity.value == "Plate:0"
    assert dataset.pixel_size == 0.65
    assert [candidate.declared_address.channel for candidate in dataset.candidates] == [
        "1",
        "2",
    ]
    assert {candidate.declared_address.well for candidate in dataset.candidates} == {
        "A01"
    }
    assert [
        candidate.component_labels["channel"] for candidate in dataset.candidates
    ] == ["DAPI", "GFP"]
    assert {
        path.name for path in dataset.candidates[0].store_identity.container_paths
    } == {source_path.name}


def test_java_adapter_normalizes_uncalibrated_pixels_to_unit_spacing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "plate.fake").write_bytes(b"")
    context = FakeBioFormatsContext()
    monkeypatch.setattr(
        context,
        "open_reader",
        lambda source_path: BioFormatsOpenedReader(
            reader=FakeBioFormatsReader(),
            metadata=FakeBioFormatsMetadata(pixel_size=None),
        ),
    )
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: context),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert dataset.pixel_size == 1.0


def test_java_adapter_aggregates_independent_czi_containers_with_one_plate_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metadata_by_name = {
        "sample-a.czi": FakeBioFormatsMetadata(
            well_id="Well:0:0",
            well_row=0,
            well_column=0,
            sample_id="WellSample:0:0:0",
            image_id="Image:a",
        ),
        "sample-b.czi": FakeBioFormatsMetadata(
            well_id="Well:0:1",
            well_row=0,
            well_column=1,
            sample_id="WellSample:0:1:0",
            image_id="Image:b",
        ),
    }
    for name in metadata_by_name:
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(
            lambda cls: FakeBioFormatsContext(
                metadata_by_name,
                declared_suffixes=(".czi",),
            )
        ),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert dataset.identity.value == "Plate:0"
    assert len(dataset.candidates) == 4
    assert {candidate.declared_address.well for candidate in dataset.candidates} == {
        "A01",
        "A02",
    }
    assert {
        path.name
        for candidate in dataset.candidates
        for path in candidate.store_identity.container_paths
    } == set(metadata_by_name)
    assert {
        source_path
        for candidate in dataset.candidates
        for source_path in candidate.source_filter_paths
        if not Path(source_path).is_absolute()
    } == set(metadata_by_name)


def test_java_adapter_aggregates_multiple_nonplate_czi_by_container_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _NonPlateMetadata(FakeBioFormatsMetadata):
        def getPlateCount(self):
            return 0

    metadata_by_name = {
        "sample-a.czi": _NonPlateMetadata(image_id="Image:0"),
        "sample-b.czi": _NonPlateMetadata(image_id="Image:0"),
    }
    for name in metadata_by_name:
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(
            lambda cls: FakeBioFormatsContext(
                metadata_by_name,
                declared_suffixes=(".czi",),
            )
        ),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert dataset.identity.value == tmp_path.resolve().as_uri()
    assert {candidate.declared_address.well for candidate in dataset.candidates} == {
        "sample-a.czi",
        "sample-b.czi",
    }
    assert {candidate.declared_address.site for candidate in dataset.candidates} == {
        "1"
    }
    assert {
        path.name
        for candidate in dataset.candidates
        for path in candidate.store_identity.container_paths
    } == set(metadata_by_name)


def test_java_adapter_rejects_conflicting_embedded_plate_ids_actionably(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metadata_by_name = {
        "plate-a.czi": FakeBioFormatsMetadata(
            plate_id="Plate:a",
            image_id="Image:a",
        ),
        "plate-b.czi": FakeBioFormatsMetadata(
            plate_id="Plate:b",
            image_id="Image:b",
        ),
    }
    for name in metadata_by_name:
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(
            lambda cls: FakeBioFormatsContext(
                metadata_by_name,
                declared_suffixes=(".czi",),
            )
        ),
    )

    with pytest.raises(
        BioFormatsDatasetAmbiguityError,
        match=r"Plate:a.*Plate:b.*separate submitted roots",
    ):
        SourcePlaneStoreAdapter.discover_dataset(tmp_path)


def test_java_adapter_rejects_cross_container_plane_address_collision(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metadata_by_name = {
        "sample-a.czi": FakeBioFormatsMetadata(
            sample_id="WellSample:a",
            image_id="Image:a",
        ),
        "sample-b.czi": FakeBioFormatsMetadata(
            sample_id="WellSample:b",
            image_id="Image:b",
        ),
    }
    for name in metadata_by_name:
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(
            lambda cls: FakeBioFormatsContext(
                metadata_by_name,
                declared_suffixes=(".czi",),
            )
        ),
    )

    with pytest.raises(
        BioFormatsDatasetAmbiguityError,
        match=r"Duplicate source plane address.*repair colliding embedded",
    ):
        SourcePlaneStoreAdapter.discover_dataset(tmp_path)


def test_ome_tiff_is_owned_by_java_store_not_ordinary_tiff_leaf(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "plate.ome.tiff"
    tifffile.imwrite(source_path, np.zeros((4, 5), dtype=np.uint16), ome=True)
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(
            lambda cls: FakeBioFormatsContext(
                {source_path.name: FakeBioFormatsMetadata()},
                declared_suffixes=(".ome.tiff",),
            )
        ),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert {candidate.source_ref.backend for candidate in dataset.candidates} == {
        Backend.BIOFORMATS.value
    }
    assert {
        path.name
        for candidate in dataset.candidates
        for path in candidate.store_identity.container_paths
    } == {source_path.name}


def test_ordinary_image_leaf_yields_to_a_rich_store_container(
    monkeypatch,
    tmp_path: Path,
) -> None:
    entrypoint = tmp_path / "plate.fake"
    plane = tmp_path / "plate_A01_w1.tif"
    entrypoint.write_bytes(b"rich container declaration")
    tifffile.imwrite(plane, np.zeros((4, 5), dtype=np.uint16))
    context = FakeBioFormatsContext(declared_suffixes=(".fake",))
    monkeypatch.setattr(
        context,
        "open_reader",
        lambda source_path: BioFormatsOpenedReader(
            reader=FakeBioFormatsReader((str(entrypoint), str(plane))),
            metadata=FakeBioFormatsMetadata(),
        ),
    )
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: context),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert len(dataset.candidates) == 2
    assert {candidate.source_ref.backend for candidate in dataset.candidates} == {
        Backend.BIOFORMATS.value
    }
    assert {
        path
        for candidate in dataset.candidates
        for path in candidate.store_identity.container_paths
    } == {entrypoint, plane}
