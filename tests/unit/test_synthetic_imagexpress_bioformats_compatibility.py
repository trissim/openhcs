import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import tifffile

from openhcs.microscopes import create_microscope_handler
from openhcs.microscopes.imagexpress import ImageXpressFilenameParser
from openhcs.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager
from tests.unit.bioformats_imagexpress_fixture import IMAGE_XPRESS_PLATE_FACTORY


def test_imagexpress_parser_accepts_plate_prefixed_metaxpress_names() -> None:
    parser = ImageXpressFilenameParser()

    parsed = parser.parse_filename("plate_A01_s1_w2.tif")

    assert parsed == {
        "well": "A01",
        "site": 1,
        "channel": 2,
        "z_index": 1,
        "timepoint": 1,
        "extension": ".tif",
    }


def test_imagexpress_parser_defaults_missing_metaxpress_site_to_one() -> None:
    parser = ImageXpressFilenameParser()

    parsed = parser.parse_filename("plate_A01_w2.tif")

    assert parsed == {
        "well": "A01",
        "site": 1,
        "channel": 2,
        "z_index": 1,
        "timepoint": 1,
        "extension": ".tif",
    }


def test_synthetic_imagexpress_can_emit_bioformats_compatible_plate_prefix(
    tmp_path: Path,
) -> None:
    plate = tmp_path / "plate"

    IMAGE_XPRESS_PLATE_FACTORY.create(plate)

    assert sorted(path.relative_to(plate).as_posix() for path in plate.rglob("*.tif")) == [
        "TimePoint_1/ZStep_1/plate_A01_s1_w1.tif",
        "TimePoint_1/ZStep_1/plate_A01_s1_w2.tif",
        "TimePoint_1/ZStep_1/plate_A01_s2_w1.tif",
        "TimePoint_1/ZStep_1/plate_A01_s2_w2.tif",
        "TimePoint_1/ZStep_2/plate_A01_s1_w1.tif",
        "TimePoint_1/ZStep_2/plate_A01_s1_w2.tif",
        "TimePoint_1/ZStep_2/plate_A01_s2_w1.tif",
        "TimePoint_1/ZStep_2/plate_A01_s2_w2.tif",
    ]
    assert {
        tifffile.imread(path).shape
        for path in plate.rglob("*.tif")
    } == {(32, 32)}


def test_imagexpress_handler_normalizes_bioformats_compatible_plate_prefix(
    tmp_path: Path,
) -> None:
    ensure_storage_registry()
    plate = tmp_path / "plate"
    IMAGE_XPRESS_PLATE_FACTORY.create(plate)
    filemanager = FileManager(dict(storage_registry))
    handler = create_microscope_handler(
        "imagexpress",
        plate_folder=plate,
        filemanager=filemanager,
    )

    handler.initialize_workspace(plate, filemanager)

    metadata = json.loads((plate / "openhcs_metadata.json").read_text(encoding="utf-8"))
    mapping = metadata["subdirectories"]["."]["workspace_mapping"]
    assert mapping == {
        "A01_s001_w1_z001_t001.tif": "TimePoint_1/ZStep_1/plate_A01_s1_w1.tif",
        "A01_s001_w2_z001_t001.tif": "TimePoint_1/ZStep_1/plate_A01_s1_w2.tif",
        "A01_s002_w1_z001_t001.tif": "TimePoint_1/ZStep_1/plate_A01_s2_w1.tif",
        "A01_s002_w2_z001_t001.tif": "TimePoint_1/ZStep_1/plate_A01_s2_w2.tif",
        "A01_s001_w1_z002_t001.tif": "TimePoint_1/ZStep_2/plate_A01_s1_w1.tif",
        "A01_s001_w2_z002_t001.tif": "TimePoint_1/ZStep_2/plate_A01_s1_w2.tif",
        "A01_s002_w1_z002_t001.tif": "TimePoint_1/ZStep_2/plate_A01_s2_w1.tif",
        "A01_s002_w2_z002_t001.tif": "TimePoint_1/ZStep_2/plate_A01_s2_w2.tif",
    }


def test_imagexpress_handler_defaults_flat_bioformats_plate_to_z_one(
    tmp_path: Path,
) -> None:
    ensure_storage_registry()
    plate = tmp_path / "plate"
    with redirect_stdout(StringIO()):
        SyntheticMicroscopyGenerator(
            output_dir=str(plate),
            grid_size=(1, 1),
            tile_size=(32, 32),
            wavelengths=2,
            z_stack_levels=1,
            num_cells=4,
            wells=["A01"],
            format="ImageXpress",
            random_seed=7,
            imagexpress_bioformats_compatible=True,
        ).generate_dataset()

    filemanager = FileManager(dict(storage_registry))
    handler = create_microscope_handler(
        "imagexpress",
        plate_folder=plate,
        filemanager=filemanager,
    )

    handler.initialize_workspace(plate, filemanager)

    metadata = json.loads((plate / "openhcs_metadata.json").read_text(encoding="utf-8"))
    mapping = metadata["subdirectories"]["."]["workspace_mapping"]
    assert mapping == {
        "A01_s001_w1_z001_t001.tif": "TimePoint_1/plate_A01_w1.tif",
        "A01_s001_w2_z001_t001.tif": "TimePoint_1/plate_A01_w2.tif",
    }
