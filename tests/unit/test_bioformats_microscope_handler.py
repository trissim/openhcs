import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.constants.constants import Backend
from openhcs.core.steps.function_execution import SourceWorkspaceAnchorProjection
from openhcs.core.steps.function_runtime import PatternGroupRuntime
from openhcs.microscopes import create_microscope_handler
from openhcs.microscopes.bioformats import BioFormatsHandler, BioFormatsMetadataHandler
from openhcs.microscopes.bioformats_spw_projector import BioFormatsProjectionError
from openhcs.microscopes.openhcs import workspace_mapping_source_path
from tests.unit.bioformats_fixture import (
    bioformats_filemanager,
    write_bioformats_manifest_fixture,
)


def test_bioformats_handler_writes_normalized_workspace_metadata(tmp_path: Path) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()
    handler = BioFormatsHandler(filemanager)

    image_dir = handler.initialize_workspace(tmp_path, filemanager)

    metadata = json.loads((tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8"))
    subdirectory = metadata["subdirectories"]["."]
    assert image_dir == tmp_path
    assert subdirectory["microscope_handler_name"] == "bioformats"
    assert subdirectory["source_filename_parser_name"] == "BioFormatsFilenameParser"
    assert subdirectory["available_backends"] == {Backend.BIOFORMATS.value: True}
    assert subdirectory["image_files"] == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    ]
    assert subdirectory["channels"] == {"1": "DAPI", "2": "GFP"}
    first_ref = subdirectory["workspace_mapping"]["A01_s001_w1_z001_t001.tif"]
    assert first_ref["reader"] == "npy"
    assert first_ref["source_path"] == "stack.npy"
    assert first_ref["c"] == 1
    assert Backend.BIOFORMATS.value in filemanager.registry


def test_bioformats_structured_refs_project_to_source_paths(tmp_path: Path) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()
    BioFormatsHandler(filemanager).initialize_workspace(tmp_path, filemanager)
    metadata = json.loads((tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8"))

    projection = SourceWorkspaceAnchorProjection.from_openhcs_metadata(metadata)
    source_ref = metadata["subdirectories"]["."]["workspace_mapping"][
        "A01_s001_w1_z001_t001.tif"
    ]

    assert projection.paths_by_virtual_path["A01_s001_w1_z001_t001.tif"] == "stack.npy"
    assert workspace_mapping_source_path(tmp_path, source_ref) == tmp_path / "stack.npy"


def test_bioformats_structured_refs_project_inside_pattern_runtime(
    tmp_path: Path,
) -> None:
    class _RuntimeContext:
        __slots__ = ("plate_path", "__weakref__")

        def __init__(self, plate_path: Path) -> None:
            self.plate_path = plate_path

    write_bioformats_manifest_fixture(tmp_path)
    BioFormatsHandler(bioformats_filemanager()).initialize_workspace(
        tmp_path,
        bioformats_filemanager(),
    )
    metadata = json.loads((tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8"))
    runtime = PatternGroupRuntime(
        SimpleNamespace(
            context=_RuntimeContext(tmp_path),
            execution_plan=None,
            pattern_group_info="bioformats structured-ref projection",
        )
    )

    projection = runtime._virtual_workspace_source_projection_from_metadata(metadata)

    assert (
        projection.source_paths_by_virtual_path["A01_s001_w1_z001_t001.tif"]
        == str(tmp_path / "stack.npy")
    )


def test_bioformats_metadata_handler_reports_component_values(tmp_path: Path) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    handler = BioFormatsMetadataHandler(bioformats_filemanager())

    assert handler.find_metadata_file(tmp_path) == tmp_path
    assert handler.get_pixel_size(tmp_path) == 0.5
    assert handler.get_channel_values(tmp_path) == {"1": "DAPI", "2": "GFP"}
    assert handler.get_well_values(tmp_path) == {"A01": "A01"}
    assert handler.get_site_values(tmp_path) == {"1": "Site 1"}
    assert handler.get_z_index_values(tmp_path) == {"1": "Z1"}
    assert handler.get_timepoint_values(tmp_path) == {"1": "T1"}


def test_bioformats_auto_detection_is_late_fallback(tmp_path: Path) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()

    handler = create_microscope_handler(
        "auto",
        plate_folder=tmp_path,
        filemanager=filemanager,
    )

    assert isinstance(handler, BioFormatsHandler)


def test_create_microscope_handler_supports_explicit_bioformats(tmp_path: Path) -> None:
    handler = create_microscope_handler(
        "bioformats",
        plate_folder=tmp_path,
        filemanager=bioformats_filemanager(),
    )

    assert isinstance(handler, BioFormatsHandler)


def test_bioformats_metadata_handler_fails_without_spw_manifest(tmp_path: Path) -> None:
    handler = BioFormatsMetadataHandler(bioformats_filemanager())

    with pytest.raises(FileNotFoundError, match="Bio-Formats-readable"):
        handler.find_metadata_file(tmp_path)


def test_bioformats_handler_rejects_metadata_only_series_without_pixel_source(
    tmp_path: Path,
) -> None:
    (tmp_path / "plate.htd").write_text("metadata", encoding="utf-8")
    (tmp_path / "bioformats_spw.json").write_text(
        json.dumps(
            {
                "plates": [
                    {
                        "wells": [
                            {
                                "row": 0,
                                "column": 0,
                                "samples": [{"image_id": "image:0", "index": 0}],
                            }
                        ]
                    }
                ],
                "images": [
                    {
                        "image_id": "image:0",
                        "source_path": "plate.htd",
                        "source_files": ["plate.htd"],
                        "pixels": {
                            "size_c": 1,
                            "size_z": 1,
                            "size_t": 1,
                            "planes": [{"c": 1, "z": 1, "t": 1, "index": 0}],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    filemanager = bioformats_filemanager()
    with pytest.raises(BioFormatsProjectionError, match="metadata-only series"):
        BioFormatsHandler(filemanager).initialize_workspace(tmp_path, filemanager)


def test_bioformats_manifest_layout_axes_support_non_spw_hcs_layout(
    tmp_path: Path,
) -> None:
    stack = write_bioformats_manifest_fixture(tmp_path)
    payload = json.loads((tmp_path / "bioformats_spw.json").read_text(encoding="utf-8"))
    payload["plates"] = []
    payload["images"][0]["layout_axes"] = {
        "well": "C05",
        "site": 2,
        "channel": 4,
        "z_index": 3,
        "timepoint": 1,
        "channel_name": "DAPI",
    }
    (tmp_path / "bioformats_spw.json").write_text(json.dumps(payload), encoding="utf-8")
    filemanager = bioformats_filemanager()

    BioFormatsHandler(filemanager).initialize_workspace(tmp_path, filemanager)
    metadata = json.loads((tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8"))
    subdirectory = metadata["subdirectories"]["."]
    backend = filemanager.registry[Backend.BIOFORMATS.value]

    assert subdirectory["image_files"] == [
        "C05_s002_w4_z003_t001.tif",
        "C05_s002_w5_z003_t001.tif",
    ]
    assert subdirectory["channels"] == {"4": "DAPI", "5": "DAPI"}
    loaded = backend.load(tmp_path / "C05_s002_w4_z003_t001.tif")
    assert loaded.shape == stack[0, 0, 0].shape
