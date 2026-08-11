import json
from dataclasses import replace
from pathlib import Path

import pytest
from polystore.virtual_workspace import SourcePixelRef

from objectstate.lazy_factory import ensure_global_config_context
from openhcs.constants.constants import Backend
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.source_workspace_projection import (
    VirtualWorkspaceSourceProjectionAuthority,
    VirtualWorkspaceSourceProjectionCache,
)
from openhcs.microscopes import create_microscope_handler
from openhcs.microscopes.bioformats import BioFormatsHandler, BioFormatsMetadataHandler
from openhcs.microscopes.bioformats_adapter import (
    BioFormatsPackedRgbSeriesExclusion,
)
from openhcs.microscopes.openhcs import (
    OpenHCSMetadataHandler,
    OpenHCSMicroscopeHandler,
)
from tests.unit.bioformats_fixture import (
    bioformats_filemanager,
    write_bioformats_manifest_fixture,
)


def test_bioformats_handler_writes_normalized_workspace_metadata(
    tmp_path: Path,
) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()
    handler = BioFormatsHandler(filemanager)

    image_dir = handler.initialize_workspace(tmp_path, filemanager)

    metadata = json.loads(
        (tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8")
    )
    subdirectory = metadata["subdirectories"]["."]
    assert image_dir == tmp_path
    assert subdirectory["microscope_handler_name"] == "bioformats"
    assert subdirectory["source_filename_parser_name"] == "BioFormatsFilenameParser"
    assert subdirectory["available_backends"] == {Backend.DISK.value: True}
    assert subdirectory["image_files"] == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    ]
    assert subdirectory["channels"] == {"1": "DAPI", "2": "GFP"}
    first_ref = subdirectory["workspace_mapping"]["A01_s001_w1_z001_t001.tif"]
    assert first_ref == {
        "backend": Backend.DISK.value,
        "backend_address": "stack.npy",
        "source_axis_indices": [0, 0, 0],
    }
    assert Backend.BIOFORMATS.value in filemanager.registry
    assert Backend.VIRTUAL_WORKSPACE.value in filemanager.registry


def test_bioformats_handler_persists_dataset_diagnostics_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()
    handler = BioFormatsHandler(filemanager)
    dataset = handler.metadata_handler.source_dataset(tmp_path)
    exclusion = BioFormatsPackedRgbSeriesExclusion(
        source_files=(tmp_path / "plate.czi",),
        image_id="Image:label",
        image_name="Label",
        series_index=7,
        rgb_channel_count=3,
    )
    dataset = replace(dataset, diagnostics=(exclusion,))
    monkeypatch.setattr(
        handler.metadata_handler,
        "source_dataset",
        lambda plate_path: dataset,
    )

    handler.initialize_workspace(tmp_path, filemanager)

    metadata = json.loads(
        (tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8")
    )["subdirectories"]["."]
    assert metadata["source_diagnostics"] == [exclusion.metadata_payload()]
    assert all(
        "source_diagnostics" not in source_metadata
        for source_metadata in metadata["source_metadata"].values()
    )

    direct_diagnostics = handler.metadata_handler.source_diagnostics(tmp_path)
    assert direct_diagnostics == (exclusion.metadata_payload(),)
    direct_document = handler.metadata_handler.build_metadata_view_document(
        tmp_path,
        handler,
    )
    assert direct_document.entries[0].object_instance.source_diagnostics == [
        exclusion.metadata_payload()
    ]

    openhcs_metadata_handler = OpenHCSMetadataHandler(filemanager)
    assert openhcs_metadata_handler.source_diagnostics(tmp_path) == (
        exclusion.metadata_payload(),
    )
    persisted_document = openhcs_metadata_handler.build_metadata_view_document(
        tmp_path,
        handler,
    )
    assert persisted_document.entries[0].object_instance.source_diagnostics == [
        exclusion.metadata_payload()
    ]
    assert "source diagnostics: 1" in persisted_document.entries[0].summary


def test_bioformats_disk_fixture_refs_load_through_virtual_workspace(
    tmp_path: Path,
) -> None:
    stack = write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()
    BioFormatsHandler(filemanager).initialize_workspace(tmp_path, filemanager)
    metadata = json.loads(
        (tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8")
    )

    source_ref = SourcePixelRef.from_workspace_mapping(
        metadata["subdirectories"]["."]["workspace_mapping"][
            "A01_s001_w1_z001_t001.tif"
        ]
    )

    assert source_ref == SourcePixelRef(
        backend=Backend.DISK.value,
        backend_address="stack.npy",
        source_axis_indices=(0, 0, 0),
    )
    loaded = filemanager.load(
        tmp_path / "A01_s001_w1_z001_t001.tif",
        backend=Backend.VIRTUAL_WORKSPACE.value,
    )
    assert (loaded == stack[0, 0, 0]).all()


def test_bioformats_structured_refs_project_inside_pattern_runtime(
    tmp_path: Path,
) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()
    BioFormatsHandler(filemanager).initialize_workspace(
        tmp_path,
        filemanager,
    )
    authority = VirtualWorkspaceSourceProjectionAuthority(
        plate_path=tmp_path,
        metadata_handlers=(OpenHCSMetadataHandler(filemanager),),
        cache=VirtualWorkspaceSourceProjectionCache(),
    )

    projection = authority.projection_if_available()

    assert projection is not None
    assert projection.pipeline_start_files() == (
        str(tmp_path / "A01_s001_w1_z001_t001.tif"),
        str(tmp_path / "A01_s001_w2_z001_t001.tif"),
    )


def test_orchestrator_exposes_prepared_virtual_source_workspace(
    tmp_path: Path,
) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())

    orchestrator = PipelineOrchestrator(plate_path=tmp_path).initialize()

    projection = orchestrator.source_workspace_projection()

    assert projection.pipeline_start_files() == (
        str(tmp_path / "A01_s001_w1_z001_t001.tif"),
        str(tmp_path / "A01_s001_w2_z001_t001.tif"),
    )
    assert projection.source_refs_by_virtual_path[
        "A01_s001_w1_z001_t001.tif"
    ] == SourcePixelRef(
        backend=Backend.DISK.value,
        backend_address="stack.npy",
        source_axis_indices=(0, 0, 0),
    )


def test_openhcs_replay_registers_declared_source_handler_backends(
    tmp_path: Path,
) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    manifest_path = tmp_path / "bioformats_spw.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["images"][0]["reader"] = "bioformats"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    initial_filemanager = bioformats_filemanager()
    BioFormatsHandler(initial_filemanager).initialize_workspace(
        tmp_path,
        initial_filemanager,
    )
    reopened_filemanager = bioformats_filemanager()
    assert Backend.BIOFORMATS.value not in reopened_filemanager.registry

    handler = create_microscope_handler(
        "auto",
        plate_folder=tmp_path,
        filemanager=reopened_filemanager,
    )
    assert handler.plate_folder == tmp_path
    handler.initialize_workspace(tmp_path, reopened_filemanager)

    assert isinstance(handler, OpenHCSMicroscopeHandler)
    assert Backend.BIOFORMATS.value in reopened_filemanager.registry
    assert Backend.VIRTUAL_WORKSPACE.value in reopened_filemanager.registry
    assert (
        handler.get_primary_backend(tmp_path, reopened_filemanager)
        == Backend.VIRTUAL_WORKSPACE.value
    )


def test_bioformats_metadata_handler_reports_component_values(tmp_path: Path) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    handler = BioFormatsMetadataHandler(bioformats_filemanager())

    assert handler.find_metadata_file(tmp_path) == tmp_path
    assert handler.get_grid_dimensions(tmp_path) == (1, 1)
    assert handler.get_pixel_size(tmp_path) == 0.5
    assert handler.get_channel_values(tmp_path) == {"1": "DAPI", "2": "GFP"}
    assert handler.get_well_values(tmp_path) == {"A01": "A01"}
    assert handler.get_site_values(tmp_path) == {"1": None}
    assert handler.get_z_index_values(tmp_path) == {"1": None}
    assert handler.get_timepoint_values(tmp_path) == {"1": None}


def test_bioformats_auto_detection_is_late_fallback(tmp_path: Path) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()

    handler = create_microscope_handler(
        "auto",
        plate_folder=tmp_path,
        filemanager=filemanager,
    )

    assert isinstance(handler, BioFormatsHandler)


def test_openhcs_output_subdirectory_initializes_from_metadata_root(
    tmp_path: Path,
) -> None:
    plate = tmp_path / "plate_openhcs"
    images = plate / "images"
    results = plate / "images_results"
    images.mkdir(parents=True)
    results.mkdir()
    (images / "A01_s001_w1_z001_t001.tif").write_text(
        "placeholder",
        encoding="utf-8",
    )
    (plate / "openhcs_metadata.json").write_text(
        json.dumps(
            {
                "subdirectories": {
                    "images": {
                        "microscope_handler_name": "openhcsdata",
                        "source_filename_parser_name": "SourceSchemaFilenameParser",
                        "grid_dimensions": [1, 1],
                        "pixel_size": 1.0,
                        "image_files": ["images/A01_s001_w1_z001_t001.tif"],
                        "channels": {"1": "DNA"},
                        "wells": {"A01": "A01"},
                        "sites": {"1": "1"},
                        "z_indexes": {"1": "1"},
                        "timepoints": {"1": "1"},
                        "available_backends": {Backend.DISK.value: True},
                        "main": True,
                        "results_dir": "images_results",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    filemanager = bioformats_filemanager()

    handler = create_microscope_handler(
        "auto",
        plate_folder=results,
        filemanager=filemanager,
    )
    input_dir = handler.initialize_workspace(results, filemanager)

    assert isinstance(handler, OpenHCSMicroscopeHandler)
    assert handler.plate_folder == plate
    assert input_dir == images


def test_create_microscope_handler_supports_explicit_bioformats(tmp_path: Path) -> None:
    handler = create_microscope_handler(
        "bioformats",
        plate_folder=tmp_path,
        filemanager=bioformats_filemanager(),
    )

    assert isinstance(handler, BioFormatsHandler)


def test_bioformats_metadata_handler_fails_without_spw_manifest(tmp_path: Path) -> None:
    handler = BioFormatsMetadataHandler(bioformats_filemanager())

    with pytest.raises(FileNotFoundError, match="addressable Bio-Formats"):
        handler.find_metadata_file(tmp_path)


def test_bioformats_nonplate_manifest_maps_image_identity_to_well(
    tmp_path: Path,
) -> None:
    stack = write_bioformats_manifest_fixture(tmp_path)
    manifest_path = tmp_path / "bioformats_spw.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["plates"] = []
    payload["dataset_id"] = "Dataset:nonplate"
    payload["images"][0]["image_name"] = "Specimen A"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    filemanager = bioformats_filemanager()

    BioFormatsHandler(filemanager).initialize_workspace(tmp_path, filemanager)
    metadata = json.loads(
        (tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8")
    )
    subdirectory = metadata["subdirectories"]["."]
    backend = filemanager.registry[Backend.VIRTUAL_WORKSPACE.value]

    assert subdirectory["image_files"] == [
        "stack.npy_s001_w1_z001_t001.tif",
        "stack.npy_s001_w2_z001_t001.tif",
    ]
    assert subdirectory["wells"] == {"stack.npy": "stack.npy"}
    assert subdirectory["sites"] == {"1": "Specimen A"}
    loaded = backend.load(tmp_path / "stack.npy_s001_w1_z001_t001.tif")
    assert loaded.shape == stack[0, 0, 0].shape
