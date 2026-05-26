import json
from pathlib import Path

import numpy as np

from openhcs.constants.constants import Backend
from openhcs.microscopes.bioformats import BioFormatsHandler
from polystore.bioformats_storage import BioFormatsStorageBackend
from tests.unit.bioformats_fixture import (
    bioformats_filemanager,
    write_bioformats_manifest_fixture,
)


def test_bioformats_backend_lists_and_loads_structured_refs(tmp_path: Path) -> None:
    stack = write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()
    BioFormatsHandler(filemanager).initialize_workspace(tmp_path, filemanager)
    backend = filemanager.registry[Backend.BIOFORMATS.value]

    files = backend.list_files(tmp_path, extensions={".tif"})
    loaded = backend.load(tmp_path / "A01_s001_w2_z001_t001.tif")

    assert sorted(Path(path).name for path in files) == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    ]
    np.testing.assert_array_equal(loaded, stack[0, 0, 1])


def test_bioformats_backend_recreates_from_picklable_params(tmp_path: Path) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()
    BioFormatsHandler(filemanager).initialize_workspace(tmp_path, filemanager)
    backend = filemanager.registry[Backend.BIOFORMATS.value]

    recreated = BioFormatsStorageBackend()
    recreated.set_connection_params(backend.get_connection_params())

    assert recreated.exists(tmp_path / "A01_s001_w1_z001_t001.tif")
    assert recreated.is_dir(tmp_path)
    assert recreated.is_file(tmp_path / "A01_s001_w1_z001_t001.tif")
    assert recreated.list_dir(tmp_path) == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    ]


def test_bioformats_backend_dispatches_java_reader(monkeypatch, tmp_path: Path) -> None:
    metadata = {
        "subdirectories": {
            ".": {
                "available_backends": {Backend.BIOFORMATS.value: True},
                "workspace_mapping": {
                    "A01_s001_w1_z001_t001.tif": {
                        "reader": "bioformats",
                        "source_path": "plate.fake",
                        "series_index": 3,
                        "plane_index": 7,
                        "c": 1,
                        "z": 1,
                        "t": 1,
                    }
                },
            }
        }
    }
    (tmp_path / "openhcs_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    def fake_load_bioformats_plane(*, source_path, series_index, plane_index):
        assert source_path == tmp_path / "plate.fake"
        assert series_index == 3
        assert plane_index == 7
        return np.array([[42]], dtype=np.uint16)

    monkeypatch.setattr(
        "polystore.bioformats_java.load_bioformats_plane",
        fake_load_bioformats_plane,
    )

    backend = BioFormatsStorageBackend(plate_root=tmp_path)

    np.testing.assert_array_equal(
        backend.load("A01_s001_w1_z001_t001.tif"),
        np.array([[42]], dtype=np.uint16),
    )
