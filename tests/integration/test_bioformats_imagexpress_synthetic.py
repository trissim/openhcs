import json
from pathlib import Path

import pytest

from openhcs.constants.constants import AllComponents
from openhcs.microscopes.bioformats import BioFormatsHandler
from openhcs.microscopes.bioformats_adapter import (
    SourcePlaneStoreAdapter,
)
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager
from tests.unit.bioformats_imagexpress_fixture import IMAGE_XPRESS_PLATE_FACTORY


@pytest.mark.integration
def test_bioformats_detects_synthetic_imagexpress_plate(tmp_path: Path) -> None:
    pytest.importorskip("imagej")
    plate = tmp_path / "plate"
    IMAGE_XPRESS_PLATE_FACTORY.create(plate)

    dataset = SourcePlaneStoreAdapter.discover_dataset(plate)
    assert dataset.identity.value == "Plate:0"
    assert len(dataset.candidates) == 8
    assert {
        candidate.declared_address.value_for(AllComponents.WELL)
        for candidate in dataset.candidates
    } == {"A01"}

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
        "backend_address": (
            '{"plane_index":0,"series_index":0,"source_path":"plate.HTD"}'
        ),
        "source_axis_indices": [],
    }
