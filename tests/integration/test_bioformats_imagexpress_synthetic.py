import json
from pathlib import Path

import pytest

from openhcs.microscopes.bioformats import BioFormatsHandler
from openhcs.microscopes.bioformats_adapter import BioFormatsCompositeAdapter
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager
from tests.unit.bioformats_imagexpress_fixture import IMAGE_XPRESS_PLATE_FACTORY


@pytest.mark.integration
def test_bioformats_detects_synthetic_imagexpress_plate(tmp_path: Path) -> None:
    pytest.importorskip("imagej")
    plate = tmp_path / "plate"
    IMAGE_XPRESS_PLATE_FACTORY.create(plate)

    metadata = BioFormatsCompositeAdapter().discover(plate)
    sampled_wells = [
        (well.row, well.column, len(well.samples))
        for plate_record in metadata.plates
        for well in plate_record.wells
        if well.samples
    ]

    assert len(metadata.plates) == 1
    assert len(metadata.images) == 2
    assert sampled_wells == [(0, 0, 2)]
    assert {
        (
            image.pixels.size_c,
            image.pixels.size_z,
            image.pixels.size_t,
            len(image.pixels.planes),
        )
        for image in metadata.images
    } == {(2, 2, 1, 4)}

    ensure_storage_registry()
    filemanager = FileManager(dict(storage_registry))
    BioFormatsHandler(filemanager).initialize_workspace(plate, filemanager)
    openhcs_metadata = json.loads(
        (plate / "openhcs_metadata.json").read_text(encoding="utf-8")
    )
    mapping = openhcs_metadata["subdirectories"]["."]["workspace_mapping"]

    assert sorted(mapping) == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w1_z002_t001.tif",
        "A01_s001_w2_z001_t001.tif",
        "A01_s001_w2_z002_t001.tif",
        "A01_s002_w1_z001_t001.tif",
        "A01_s002_w1_z002_t001.tif",
        "A01_s002_w2_z001_t001.tif",
        "A01_s002_w2_z002_t001.tif",
    ]
    assert mapping["A01_s001_w1_z001_t001.tif"] == {
        "backend": "bioformats",
        "reader": "bioformats",
        "source_path": "plate.HTD",
        "series_index": 0,
        "plane_index": 0,
        "c": 1,
        "z": 1,
        "t": 1,
    }
