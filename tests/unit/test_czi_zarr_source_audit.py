from __future__ import annotations

import json
from pathlib import Path

from openhcs.constants.constants import Backend
from openhcs.microscopes.openhcs import OpenHCSMicroscopeHandler
from tests.unit.bioformats_fixture import bioformats_filemanager


def test_openhcs_zarr_reload_restores_registered_bioformats_parser(
    tmp_path: Path,
) -> None:
    zarr_dir = tmp_path / "zarr"
    zarr_dir.mkdir()
    (tmp_path / "openhcs_metadata.json").write_text(
        json.dumps(
            {
                "subdirectories": {
                    "zarr": {
                        "microscope_handler_name": "bioformats",
                        "source_filename_parser_name": "BioFormatsFilenameParser",
                        "grid_dimensions": [1, 1],
                        "pixel_size": 1.0,
                        "image_files": ["zarr/A01_s001_w1_z001_t001.tif"],
                        "channels": {"1": "CZI"},
                        "wells": {"A01": "A01"},
                        "sites": {"1": "Site 1"},
                        "z_indexes": {"1": "Z1"},
                        "timepoints": {"1": "T1"},
                        "available_backends": {Backend.ZARR.value: True},
                        "main": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    filemanager = bioformats_filemanager()
    handler = OpenHCSMicroscopeHandler(filemanager)

    input_dir = handler.initialize_workspace(tmp_path, filemanager)

    assert input_dir == zarr_dir
    assert type(handler.parser).__name__ == "BioFormatsFilenameParser"
    assert handler.get_primary_backend(input_dir, filemanager) == Backend.ZARR.value
