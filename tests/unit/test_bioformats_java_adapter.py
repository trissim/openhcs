from dataclasses import dataclass
from pathlib import Path

from openhcs.microscopes.bioformats_adapter import (
    BioFormatsCompositeAdapter,
    CV7000FilenameLayoutParser,
    MetaXpressFilenameLayoutParser,
    OperettaFilenameLayoutParser,
    ScanRFilenameLayoutParser,
)
from openhcs.microscopes.bioformats_well_key import BIOFORMATS_WELL_KEYS
from openhcs.microscopes.opera_phenix import OperaPhenixFilenameParser
from openhcs.microscopes.bioformats_spw_projector import BioFormatsSPWProjector
from polystore.bioformats_java import BioFormatsOpenedReader, BioFormatsJavaContext


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
    image_id: str = "Image:0"
    channel_names: tuple[str, str] = ("DAPI", "GFP")

    def getPlateCount(self):
        return 1

    def getPlateName(self, plate):
        return self.plate_name

    def getWellCount(self, plate):
        return 1

    def getWellRow(self, plate, well):
        return JavaValue(0)

    def getWellColumn(self, plate, well):
        return JavaValue(0)

    def getWellSampleCount(self, plate, well):
        return 1

    def getWellSampleImageRef(self, plate, well, sample):
        return self.image_id

    def getWellSampleIndex(self, plate, well, sample):
        return JavaValue(0)

    def getImageCount(self):
        return 1

    def getImageID(self, image):
        return self.image_id

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
        return PhysicalSize(0.65)


class FakeBioFormatsReader:
    def __init__(self):
        self.series = None

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

    def getIndex(self, z_index, channel, timepoint):
        return channel

    def getSeriesUsedFiles(self, no_pixels):
        return ("plate.fake", "well-a01.tif")

    def close(self):
        pass


class FakeBioFormatsContext:
    def open_reader(self, source_path):
        return BioFormatsOpenedReader(
            reader=FakeBioFormatsReader(),
            metadata=FakeBioFormatsMetadata(),
        )


def test_java_adapter_projects_ome_spw_metadata(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "plate.fake").write_bytes(b"")
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: FakeBioFormatsContext()),
    )

    metadata = BioFormatsCompositeAdapter().discover(tmp_path)
    dataset = BioFormatsSPWProjector().project(metadata)

    assert [entry.channel_name for entry in dataset.entries] == ["DAPI", "GFP"]
    assert [entry.plane_index for entry in dataset.entries] == [0, 1]
    assert [path.name for path in dataset.entries[0].source_files] == [
        "plate.fake",
        "well-a01.tif",
    ]
    assert {entry.well for entry in dataset.entries} == {"A01"}
    assert {entry.pixel_size for entry in dataset.entries} == {0.65}


def test_cv7000_filename_layout_parser_projects_hcs_axes() -> None:
    parsed = CV7000FilenameLayoutParser().parse(
        Path("Dest210531-152149_A01_T0001F001L01A02Z03C05.tif")
    )

    assert parsed is not None
    assert parsed.well == "A01"
    assert parsed.site == 2
    assert parsed.channel_key == "5"
    assert parsed.channel_name == "C05"
    assert parsed.z_index == 3
    assert parsed.timepoint == 1


def test_scanr_filename_layout_parser_projects_hcs_axes() -> None:
    parsed = ScanRFilenameLayoutParser().parse(
        Path("--W00002--P00003--Z00004--T00005--nucleus-dapi.tif")
    )

    assert parsed is not None
    assert parsed.well == "W00002"
    assert parsed.site == 3
    assert parsed.channel_key == "nucleus-dapi"
    assert parsed.channel_name == "nucleus-dapi"
    assert parsed.z_index == 5
    assert parsed.timepoint == 6


def test_metaxpress_filename_layout_parser_projects_hcs_axes() -> None:
    parsed = MetaXpressFilenameLayoutParser().parse(
        Path("Act1_Plate10-SP-A8_A01_s12_w3.TIF")
    )

    assert parsed is not None
    assert parsed.well == "A01"
    assert parsed.site == 12
    assert parsed.channel_key == "3"
    assert parsed.channel_name == "W3"
    assert parsed.z_index == 1
    assert parsed.timepoint == 1


def test_operetta_filename_layout_parser_projects_hcs_axes() -> None:
    parsed = OperettaFilenameLayoutParser().parse(
        Path("r03c07f02p04-ch5sk1fk6fl1.tiff")
    )

    assert parsed is not None
    assert parsed.well == "C07"
    assert parsed.site == 2
    assert parsed.channel_key == "5"
    assert parsed.channel_name == "Channel 5"
    assert parsed.z_index == 4
    assert parsed.timepoint == 1


def test_operetta_filename_layout_parser_matches_native_opera_phenix_axes() -> None:
    filename = "r03c07f002p04-ch5sk6fk2fl1.tiff"
    native = OperaPhenixFilenameParser().parse_filename(filename)
    bioformats = OperettaFilenameLayoutParser().parse(Path(filename))

    assert native is not None
    assert bioformats is not None
    assert bioformats.well == BIOFORMATS_WELL_KEYS.key_from_one_based(3, 7)
    assert native["well"] == "R03C07"
    assert bioformats.site == native["site"] == 2
    assert bioformats.channel_key == str(native["channel"]) == "5"
    assert bioformats.z_index == native["z_index"] == 4
    assert bioformats.timepoint == native["timepoint"] == 6
