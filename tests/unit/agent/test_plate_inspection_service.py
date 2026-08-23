import os
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from polystore.virtual_workspace import SourcePixelRef

from openhcs.agent.dto.plate import (
    PlateFileQueryRequest,
    PlateImageSampleRequest,
    PlateInspectionConfidence,
    PlateInspectionIngestionRoute,
    PlateInspectionSourceBindingRole,
    PlateInspectionStatus,
    PlatePathInspectionRequest,
    SyntheticPlateGenerationRequest,
)
from openhcs.constants.constants import AllComponents
from openhcs.core.plate_image_inventory import PlateFileKind
from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.services.stdio import AgentStdoutRedirect
from openhcs.agent.services.plate_inspection_service import (
    PlateInspectionIssueCode,
    PlateInspectionService,
    PlateInspectionWorkflowAdvicePolicy,
)
from openhcs.agent.services.synthetic_plate_service import (
    SyntheticPlateGenerationService,
)
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline.path_planner import PathPlannerPathAuthority
from openhcs.microscopes.bioformats import BioFormatsHandler
from openhcs.microscopes.microscope_base import MicroscopeSourceSelectionRole
from openhcs.microscopes.source_bindings_handler import SourceBindingsHandler
from tests.unit.bioformats_fixture import (
    bioformats_filemanager,
    write_bioformats_manifest_fixture,
)


class ImageXpressPlateFixture:
    """Small ImageXpress-like fixture used by read-only plate inspection tests."""

    PLATE_FOLDER_NAME = "plate"
    TIMEPOINT_FOLDER_NAME = "TimePoint_1"
    ZSTEP_FOLDER_NAME = "ZStep_1"
    FIRST_IMAGE_NAME = "plate_A01_s1_w1.tif"
    SECOND_IMAGE_NAME = "plate_A01_s2_w2.tif"
    METADATA_FILE_NAME = "plate.HTD"
    METADATA_LINES = (
        '"XSites", 2',
        '"YSites", 1',
        '"PixelSizeUM", 0.5',
        '"WaveName1", "DAPI"',
        '"WaveName2", "GFP"',
    )

    @classmethod
    def write(cls, root: Path) -> Path:
        plate = root / cls.PLATE_FOLDER_NAME
        image_dir = plate / cls.TIMEPOINT_FOLDER_NAME / cls.ZSTEP_FOLDER_NAME
        image_dir.mkdir(parents=True)
        (image_dir / cls.FIRST_IMAGE_NAME).write_bytes(b"")
        (image_dir / cls.SECOND_IMAGE_NAME).write_bytes(b"")
        (plate / cls.METADATA_FILE_NAME).write_text(
            "\n".join(cls.METADATA_LINES),
            encoding="utf-8",
        )
        return plate


def _write_roi_archive(path: Path, *, multi_shape: bool = False) -> None:
    from polystore.base import ensure_storage_registry, storage_registry
    from polystore.filemanager import FileManager
    from polystore.roi import ROI, PolygonShape, materialize_rois

    ensure_storage_registry()
    shapes = [
        PolygonShape(
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 3.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                ]
            )
        )
    ]
    if multi_shape:
        shapes.append(
            PolygonShape(
                np.array(
                    [
                        [5.0, 5.0],
                        [5.0, 8.0],
                        [8.0, 8.0],
                        [5.0, 5.0],
                    ]
                )
            )
        )
    roi = ROI(
        shapes=shapes,
        metadata={
            "label": 1,
            "area": 4.5,
            "bbox": (0, 0, 3, 3),
            "centroid": (1.5, 1.5),
        },
    )
    materialize_rois([roi], str(path), FileManager(storage_registry), "disk")


def test_plate_request_dtos_own_mcp_tool_argument_projection():
    inspect_request = PlatePathInspectionRequest.from_fields(
        plate_path="/tmp/plate",
        max_sample_files=3,
        max_component_values=4,
    )
    query_request = PlateFileQueryRequest.from_fields(
        plate_path="/tmp/plate",
        kind="all",
        include_previews=False,
    )
    sample_request = PlateImageSampleRequest.from_fields(
        plate_path="/tmp/plate",
        image_path="A01.tif",
        resolution_index=0,
        max_auto_resolution_size=512,
        include_array_values=False,
    )
    synthetic_request = SyntheticPlateGenerationRequest.from_fields(
        output_dir="/tmp/synthetic",
        wells=["A01"],
        format="ImageXpress",
    )

    assert inspect_request.as_tool_arguments()["max_sample_files"] == 3
    assert query_request.as_tool_arguments()["kind"] == "all"
    assert query_request.as_tool_arguments()["include_previews"] is False
    assert sample_request.as_tool_arguments()["include_array_values"] is False
    assert sample_request.as_tool_arguments()["resolution_index"] == 0
    assert sample_request.as_tool_arguments()["max_auto_resolution_size"] == 512
    assert synthetic_request.as_tool_arguments()["wells"] == ["A01"]
    assert synthetic_request.as_tool_arguments()["format"] == "ImageXpress"


def test_plate_result_preview_collapses_multishape_roi_members(tmp_path: Path):
    plate = tmp_path / "plate_openhcs"
    config = GlobalPipelineConfig()
    result_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        plate / config.path_planning_config.sub_dir,
    )
    result_dir.mkdir(parents=True)
    roi_path = result_dir / "A01_w1_segmentation_masks_step0_rois.roi.zip"
    _write_roi_archive(roi_path, multi_shape=True)
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            kind=PlateFileKind.RESULT,
            limit=5,
        )
    )

    assert query.errors == ()
    assert query.returned_count == 1
    preview = query.records[0].preview
    assert preview is not None
    assert preview.roi_count == 1
    assert preview.roi_member_count == 2
    assert preview.roi_duplicate_member_count == 1
    assert preview.roi_area_mean == 4.5


def test_plate_inspection_auto_detects_imagexpress_without_mutating(tmp_path: Path):
    plate = ImageXpressPlateFixture.write(tmp_path)
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    result = service.inspect(
        PlatePathInspectionRequest.from_fields(
            plate_path=str(plate),
            max_sample_files=1,
            max_component_values=10,
        )
    )

    assert tuple(summary.component.value for summary in result.components) == (
        AllComponents.ordered_names()
    )
    well_summary = next(
        summary
        for summary in result.components
        if summary.component is AllComponents.WELL
    )
    site_summary = next(
        summary
        for summary in result.components
        if summary.component is AllComponents.SITE
    )
    channel_summary = next(
        summary
        for summary in result.components
        if summary.component is AllComponents.CHANNEL
    )
    assert result.schema_version == "openhcs.agent.v1"
    assert result.errors == ()
    assert result.status is not PlateInspectionStatus.ERROR
    assert result.detected_microscope_type == "imagexpress"
    assert result.handler_class == "ImageXpressHandler"
    assert result.parser_class == "ImageXpressFilenameParser"
    assert result.metadata_handler_class == "ImageXpressMetadataHandler"
    assert result.metadata_file_path == str(plate / "plate.HTD")
    assert result.grid_dimensions == (1, 2)
    assert result.pixel_size == 0.5
    assert result.image_files.count == 2
    assert result.image_files.truncated_file_count == 1
    assert result.parse_summary.parsed_file_count == 2
    assert well_summary.count == 1
    assert site_summary.count == 2
    assert channel_summary.count == 2
    assert result.workspace_preparation.read_only_inspection is True
    assert result.workspace_preparation.required_before_execution is True
    assert (
        result.workflow_advice.ingestion_route
        is PlateInspectionIngestionRoute.DETECTED_HANDLER
    )
    assert result.workflow_advice.ingestion_owner == "imagexpress"
    assert (
        result.workflow_advice.source_binding_role
        is PlateInspectionSourceBindingRole.NOT_PROJECTED_BY_HANDLER
    )
    assert result.workflow_advice.ui_code_document_id == (
        "plate_manager.orchestrator_config"
    )
    assert result.workflow_advice.ui_operation == "init"
    assert not (plate / "openhcs_metadata.json").exists()


def test_plate_inspection_replays_declared_bioformats_workspace_backends(
    tmp_path: Path,
) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    filemanager = bioformats_filemanager()
    BioFormatsHandler(filemanager).initialize_workspace(tmp_path, filemanager)
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        filemanager_factory=type(
            "BioFormatsFileManagerFactory",
            (),
            {"create": staticmethod(bioformats_filemanager)},
        )(),
    )

    result = service.inspect(
        PlatePathInspectionRequest.from_fields(
            plate_path=str(tmp_path),
            microscope_type="bioformats",
        )
    )

    assert result.errors == ()
    assert result.image_files.count == 2
    assert not any(
        warning.code == PlateInspectionIssueCode.IMAGE_FILE_LISTING_FAILED.value
        for warning in result.warnings
    )


def test_plate_image_sample_uses_bioformats_source_ref_before_workspace_export(
    tmp_path: Path,
) -> None:
    stack = write_bioformats_manifest_fixture(tmp_path)
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        filemanager_factory=type(
            "BioFormatsFileManagerFactory",
            (),
            {"create": staticmethod(bioformats_filemanager)},
        )(),
    )
    inspection = service.inspect(
        PlatePathInspectionRequest.from_fields(
            plate_path=str(tmp_path),
            microscope_type="bioformats",
        )
    )
    virtual_path = inspection.image_files.sampled_records[0].virtual_path

    result = service.sample_image(
        PlateImageSampleRequest(
            plate_path=str(tmp_path),
            image_path=virtual_path,
            microscope_type="bioformats",
            y=1,
            x=1,
            height=2,
            width=2,
        )
    )

    assert result.errors == ()
    assert result.virtual_path == virtual_path
    assert result.source_path == str((tmp_path / "stack.npy").resolve())
    assert result.shape == (3, 4)
    assert result.sample_values == stack[0, 0, 0, 1:3, 1:3].tolist()
    assert not (tmp_path / "openhcs_metadata.json").exists()


def test_plate_inspection_workflow_advice_keeps_projecting_store_as_owner():
    advice = PlateInspectionWorkflowAdvicePolicy.for_handler(
        BioFormatsHandler(bioformats_filemanager())
    )

    assert advice.ingestion_route is PlateInspectionIngestionRoute.DETECTED_HANDLER
    assert advice.ingestion_owner == "bioformats"
    assert (
        advice.source_binding_role
        is PlateInspectionSourceBindingRole.SEMANTIC_SELECTION
    )
    assert "does not open or replace the detected store" in advice.message


def test_plate_inspection_auto_surfaces_native_parser_for_incomplete_export(
    tmp_path: Path,
) -> None:
    images = tmp_path / "Images"
    images.mkdir()
    for channel in (1, 2, 4):
        tifffile.imwrite(
            images / f"r04c09f11p01-ch{channel}sk1fk1fl1.tiff",
            np.full((8, 8), channel, dtype=np.uint16),
        )
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        filemanager_factory=type(
            "BioFormatsFileManagerFactory",
            (),
            {"create": staticmethod(bioformats_filemanager)},
        )(),
    )

    result = service.inspect(
        PlatePathInspectionRequest.from_fields(plate_path=str(tmp_path))
    )

    assert result.detected_microscope_type == "bioformats"
    assert len(result.format_specific_handler_candidates) == 1
    candidate = result.format_specific_handler_candidates[0]
    assert candidate.microscope_type == "opera_phenix"
    assert candidate.parser_class == "OperaPhenixFilenameParser"
    assert candidate.root_dir == "Images"
    assert candidate.recognized_file_count == candidate.tested_file_count == 3
    assert candidate.recognizes_all_tested_files is True
    assert candidate.files_under_expected_root is True
    assert candidate.metadata_detected is False
    assert candidate.metadata_file_path is None
    assert "Index.xml" in (candidate.metadata_diagnostic or "")
    assert result.workflow_advice.probable_native_ingestion_owners == ()
    assert (
        result.workflow_advice.ingestion_route
        is PlateInspectionIngestionRoute.SOURCE_BINDINGS_HANDLER
    )
    assert result.workflow_advice.ingestion_owner == "source_bindings"
    assert (
        result.workflow_advice.source_binding_role
        is PlateInspectionSourceBindingRole.INGESTION_OWNER
    )
    assert "requires the complete native detection contract" in (
        result.workflow_advice.message
    )
    assert "SourceBindingsConfig" in result.workflow_advice.message
    assert any(
        warning.code == PlateInspectionIssueCode.PROBABLE_NATIVE_HANDLER.value
        for warning in result.warnings
    )
    probable_native_warning = next(
        warning
        for warning in result.warnings
        if warning.code == PlateInspectionIssueCode.PROBABLE_NATIVE_HANDLER.value
    )
    assert "requires its complete metadata detection contract" in (
        probable_native_warning.hint or ""
    )


def test_registered_handler_selection_roles_are_owned_polymorphically() -> None:
    from openhcs.microscopes.opera_phenix import OperaPhenixHandler

    assert (
        BioFormatsHandler.source_selection_role()
        is MicroscopeSourceSelectionRole.BROAD_STRUCTURED_STORE
    )
    assert (
        SourceBindingsHandler.source_selection_role()
        is MicroscopeSourceSelectionRole.DECLARED_FILE_FALLBACK
    )
    assert "structured or rich container" in (
        BioFormatsHandler.source_selection_guidance()
    )
    assert "arbitrary ordinary image files" in (
        SourceBindingsHandler.source_selection_guidance()
    )
    assert OperaPhenixHandler.supports_explicit_incomplete_export() is False
    assert "not a valid native dataset" in (
        OperaPhenixHandler.source_selection_guidance()
    )


def test_synthetic_plate_generation_service_writes_inspectable_plate(tmp_path: Path):
    policy = AgentPathPolicy.with_roots(
        readable_roots=(tmp_path,),
        writable_roots=(tmp_path,),
    )
    plate = tmp_path / "synthetic"
    generation = SyntheticPlateGenerationService(policy).generate(
        SyntheticPlateGenerationRequest(
            output_dir=str(plate),
            grid_rows=1,
            grid_cols=1,
            tile_width=32,
            tile_height=32,
            wavelengths=2,
            num_cells=4,
            wells=("A01",),
            random_seed=7,
            sample_file_limit=3,
        )
    )

    inspection = PlateInspectionService(policy).inspect(
        PlatePathInspectionRequest.from_fields(
            plate_path=str(plate),
            max_sample_files=5,
            max_component_values=10,
        )
    )
    channel_summary = next(
        summary
        for summary in inspection.components
        if summary.component is AllComponents.CHANNEL
    )

    assert generation.errors == ()
    assert generation.output_dir == str(plate.resolve(strict=False))
    assert generation.image_count == 2
    assert generation.sampled_image_files == (
        "TimePoint_1/A01_s001_w1_z001_t001.tif",
        "TimePoint_1/A01_s001_w2_z001_t001.tif",
    )
    assert generation.metadata_file_path == str(plate / "synthetic.HTD")
    assert inspection.errors == ()
    assert inspection.image_files.count == 2
    assert channel_summary.count == 2


def test_plate_file_query_reads_path_planned_results_for_source_plate(
    tmp_path: Path,
) -> None:
    plate = ImageXpressPlateFixture.write(tmp_path)
    config = GlobalPipelineConfig()
    output_root = PathPlannerPathAuthority.build_output_plate_root(
        plate,
        config.path_planning_config,
    )
    result_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        output_root / config.path_planning_config.sub_dir,
    )
    result_dir.mkdir(parents=True)
    csv_path = result_dir / "A01_w1_cell_counts_step0_details.csv"
    csv_path.write_text("slice_index,cell_count\n0,11\n", encoding="utf-8")
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    result = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            kind=PlateFileKind.RESULT,
            limit=5,
        )
    )

    assert result.errors == ()
    assert result.total_count == 1
    assert result.records[0].relative_path == (
        "images_results/A01_w1_cell_counts_step0_details.csv"
    )
    assert result.records[0].full_path == str(csv_path)
    assert result.records[0].preview is not None
    assert result.records[0].preview.csv_rows == (
        {"slice_index": "0", "cell_count": "11"},
    )


@pytest.mark.parametrize(
    ("workspace_handler_name", "source_parser_name"),
    (
        ("imagexpress", "ImageXpressFilenameParser"),
        ("source_bindings", "SourceSchemaFilenameParser"),
    ),
)
def test_plate_image_sample_resolves_openhcs_virtual_workspace(
    tmp_path: Path,
    workspace_handler_name: str,
    source_parser_name: str,
):
    plate = tmp_path / "plate"
    source_dir = plate / "TimePoint_1"
    source_dir.mkdir(parents=True)
    source_image = source_dir / "source_A01_w1.tif"
    tifffile.imwrite(
        source_image,
        np.arange(16, dtype=np.uint16).reshape(4, 4),
        compression=None,
    )
    results_dir = plate / "results"
    results_dir.mkdir()
    roi_path = results_dir / "A01_s001_w1_z001_t001.roi.zip"
    csv_path = results_dir / "B01_s001_w1_z001_t001.csv"
    text_result_path = results_dir / "notes.txt"
    _write_roi_archive(roi_path)
    csv_path.write_text("well,count\nA01,1\n", encoding="utf-8")
    text_result_path.write_text("cell count notes\nreviewed\n", encoding="utf-8")
    virtual_name = "A01_s001_w1_z001_t001.tif"
    (plate / "openhcs_metadata.json").write_text(
        json.dumps(
            {
                "subdirectories": {
                    ".": {
                        "workspace_mapping": {
                            virtual_name: SourcePixelRef(
                                backend="disk",
                                backend_address=str(source_image.relative_to(plate)),
                            ).to_workspace_mapping(),
                        },
                        "available_backends": {
                            "disk": True,
                            "virtual_workspace": True,
                        },
                        "microscope_handler_name": workspace_handler_name,
                        "source_filename_parser_name": source_parser_name,
                        "grid_dimensions": [1, 1],
                        "pixel_size": 0.65,
                        "image_files": [virtual_name],
                        "channels": {"1": "DAPI"},
                        "wells": {"A01": None},
                        "sites": {"1": None},
                        "z_indexes": {"1": None},
                        "timepoints": {"1": None},
                        "source_diagnostics": [
                            {
                                "diagnostic_type": (
                                    "bioformats_packed_rgb_series_exclusion"
                                ),
                                "message": "Packed RGB label series was excluded.",
                                "series_index": 7,
                            }
                        ],
                        "results_dir": "results",
                        "main": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    result = service.sample_image(
        PlateImageSampleRequest(
            plate_path=str(plate),
            image_path=virtual_name,
            microscope_type="openhcsdata",
            y=1,
            x=1,
            height=2,
            width=2,
        )
    )

    assert result.errors == ()
    assert result.virtual_path == virtual_name
    assert result.source_path == str(source_image.resolve(strict=False))
    assert result.source_metadata["virtual_path"] == virtual_name
    assert result.source_metadata["source_path"] == str(
        source_image.resolve(strict=False)
    )
    assert result.shape == (4, 4)
    assert result.resolution_shape == (4, 4)
    assert result.dtype == "uint16"
    assert result.minimum == 0
    assert result.maximum == 15
    assert result.requested_resolution_index is None
    assert result.selected_resolution_index == 0
    assert result.resolution_count == 1
    assert result.downsample_yx == (1.0, 1.0)
    assert result.statistics_scope == "source_resolution"
    assert result.sample_shape == (2, 2)
    assert result.sample_included is True
    assert result.sample_values == [[5, 6], [9, 10]]

    inspection = service.inspect(
        PlatePathInspectionRequest.from_fields(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            max_sample_files=3,
        )
    )

    assert inspection.image_files.sampled_files == (virtual_name,)
    assert inspection.source_diagnostics == (
        {
            "diagnostic_type": "bioformats_packed_rgb_series_exclusion",
            "message": "Packed RGB label series was excluded.",
            "series_index": 7,
        },
    )
    assert len(inspection.image_files.sampled_records) == 1
    inspection_record = inspection.image_files.sampled_records[0]
    assert inspection_record.virtual_path == virtual_name
    assert inspection_record.source_path == str(source_image.resolve(strict=False))
    assert inspection_record.metadata["virtual_path"] == virtual_name
    assert inspection_record.metadata["source_path"] == str(
        source_image.resolve(strict=False)
    )
    assert inspection_record.metadata["modified"] != "N/A"
    assert inspection.result_files.count == 3
    assert inspection.result_files.scanned_file_count == 3
    assert inspection.result_files.sampled_files == (
        str(roi_path.relative_to(plate)),
        str(csv_path.relative_to(plate)),
        str(text_result_path.relative_to(plate)),
    )
    assert inspection.result_files.truncated_file_count == 0
    roi_record, csv_record, text_record = inspection.result_files.sampled_records
    assert roi_record.relative_path == str(roi_path.relative_to(plate))
    assert roi_record.full_path == str(roi_path)
    assert roi_record.file_format == "ROI"
    assert roi_record.metadata["type"] == "ROI"
    assert roi_record.metadata["modified"] != "N/A"
    assert roi_record.preview is not None
    assert roi_record.preview.roi_count == 1
    assert roi_record.preview.roi_examples[0]["label"] == 1
    assert csv_record.file_format == "CSV"
    assert csv_record.preview is not None
    assert csv_record.preview.text_lines == ("well,count", "A01,1")
    assert csv_record.preview.csv_columns == ("well", "count")
    assert csv_record.preview.csv_rows == ({"well": "A01", "count": "1"},)
    assert text_record.file_format == "TEXT"
    assert text_record.preview is not None
    assert text_record.preview.text_lines == ("cell count notes", "reviewed")

    query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            kind=PlateFileKind.IMAGE,
            well="A01",
            limit=5,
        )
    )

    assert query.errors == ()
    assert query.total_count == 1
    assert query.returned_count == 1
    assert query.records[0].kind is PlateFileKind.IMAGE
    assert query.records[0].virtual_path == virtual_name
    assert query.records[0].source_path == str(source_image.resolve(strict=False))
    assert (
        query.records[0].metadata["source_path"]
        == result.source_metadata["source_path"]
    )

    auto_query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="auto",
            kind=PlateFileKind.IMAGE,
            well="A01",
            limit=5,
        )
    )

    assert auto_query.errors == ()
    assert auto_query.detected_microscope_type == "openhcsdata"
    assert tuple(record.virtual_path for record in auto_query.records) == (
        virtual_name,
    )

    result_query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            kind=PlateFileKind.RESULT,
            limit=3,
        )
    )

    assert result_query.errors == ()
    result_records = {record.relative_path: record for record in result_query.records}
    assert result_records[str(csv_path.relative_to(plate))].preview is not None
    assert result_records[str(csv_path.relative_to(plate))].preview.csv_rows == (
        {"well": "A01", "count": "1"},
    )
    assert result_records[str(text_result_path.relative_to(plate))].preview is not None
    assert result_records[
        str(text_result_path.relative_to(plate))
    ].preview.text_lines == (
        "cell count notes",
        "reviewed",
    )
    assert result_records[str(roi_path.relative_to(plate))].preview is not None
    assert result_records[str(roi_path.relative_to(plate))].preview.roi_count == 1
    assert result_records[str(roi_path.relative_to(plate))].preview.roi_area_mean == 4.5


def test_plate_inspection_reports_result_only_openhcs_output_root(tmp_path: Path):
    plate = tmp_path / "plate_openhcs"
    config = GlobalPipelineConfig()
    result_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        plate / config.path_planning_config.sub_dir,
    )
    result_dir.mkdir(parents=True)
    csv_path = result_dir / "A01_w1_cell_counts_step0_details.csv"
    roi_path = result_dir / "A01_w1_segmentation_masks_step0_rois.roi.zip"
    text_path = result_dir / "A01_w1_segmentation_masks_step0_summary.txt"
    csv_path.write_text("slice_index,cell_count\n0,11\n", encoding="utf-8")
    _write_roi_archive(roi_path)
    text_path.write_text("Segmentation ROIs: 11 cells\n", encoding="utf-8")
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    result = service.inspect(
        PlatePathInspectionRequest.from_fields(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            max_sample_files=3,
        )
    )

    assert result.errors == ()
    assert result.status is PlateInspectionStatus.PARTIAL
    assert result.confidence is PlateInspectionConfidence.LOW
    assert result.detected_microscope_type == "openhcsdata"
    assert result.handler_class == "OpenHCSMicroscopeHandler"
    assert result.image_files.count == 0
    assert result.result_files.count == 3
    assert result.result_files.scanned_file_count == 3
    assert result.result_files.sampled_files == (
        str(csv_path.relative_to(plate)),
        str(roi_path.relative_to(plate)),
        str(text_path.relative_to(plate)),
    )
    csv_record, roi_record, text_record = result.result_files.sampled_records
    assert csv_record.preview is not None
    assert csv_record.preview.csv_rows == ({"slice_index": "0", "cell_count": "11"},)
    assert roi_record.preview is not None
    assert roi_record.preview.roi_count == 1
    assert text_record.preview is not None
    assert text_record.preview.text_lines == ("Segmentation ROIs: 11 cells",)
    assert any(
        warning.code == PlateInspectionIssueCode.RESULT_FILE_LISTING_FAILED.value
        for warning in result.warnings
    )

    query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            kind=PlateFileKind.RESULT,
            limit=3,
        )
    )

    assert query.errors == ()
    assert query.total_count == 3
    assert [record.relative_path for record in query.records] == [
        str(csv_path.relative_to(plate)),
        str(roi_path.relative_to(plate)),
        str(text_path.relative_to(plate)),
    ]
    assert {warning.code for warning in query.warnings}.isdisjoint(
        {
            PlateInspectionIssueCode.PARSER_UNAVAILABLE.value,
            PlateInspectionIssueCode.IMAGE_FILE_LISTING_FAILED.value,
            PlateInspectionIssueCode.RESULT_FILE_LISTING_FAILED.value,
        }
    )

    auto_query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            kind=PlateFileKind.RESULT,
            limit=3,
        )
    )

    assert auto_query.errors == ()
    assert auto_query.detected_microscope_type is None
    assert auto_query.handler_class is None
    assert auto_query.total_count == 3
    assert [record.relative_path for record in auto_query.records] == [
        str(csv_path.relative_to(plate)),
        str(roi_path.relative_to(plate)),
        str(text_path.relative_to(plate)),
    ]
    assert auto_query.warnings == ()

    all_query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            kind=None,
            limit=3,
        )
    )

    assert all_query.errors == ()
    assert all_query.total_count == 3
    assert {warning.code for warning in all_query.warnings}.isdisjoint(
        {PlateInspectionIssueCode.RESULT_FILE_LISTING_FAILED.value}
    )

    image_query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            kind=PlateFileKind.IMAGE,
            limit=3,
        )
    )

    assert image_query.errors == ()
    assert image_query.total_count == 0
    assert any(
        warning.code == PlateInspectionIssueCode.RESULT_FILES_AVAILABLE.value
        and "3 analysis result artifact" in warning.message
        for warning in image_query.warnings
    )


def test_plate_file_query_auto_image_result_only_root_skips_handler_detection(
    tmp_path: Path,
) -> None:
    class ResultOnlyService(PlateInspectionService):
        handler_attempted = False

        @staticmethod
        def _create_handler(request, plate_path, filemanager):
            ResultOnlyService.handler_attempted = True
            raise RuntimeError("handler detection should not run")

    plate = tmp_path / "plate_openhcs"
    config = GlobalPipelineConfig()
    result_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        plate / config.path_planning_config.sub_dir,
    )
    result_dir.mkdir(parents=True)
    (result_dir / "A01_w1_cell_counts_step0_details.csv").write_text(
        "slice_index,cell_count\n0,11\n",
        encoding="utf-8",
    )
    service = ResultOnlyService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    image_query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            kind=PlateFileKind.IMAGE,
            limit=3,
        )
    )

    assert ResultOnlyService.handler_attempted is False
    assert image_query.errors == ()
    assert image_query.total_count == 0
    assert image_query.detected_microscope_type is None
    assert image_query.handler_class is None
    assert any(
        warning.code == PlateInspectionIssueCode.RESULT_FILES_AVAILABLE.value
        and "1 analysis result artifact" in warning.message
        for warning in image_query.warnings
    )


def test_plate_inspection_auto_result_only_root_skips_handler_detection(
    tmp_path: Path,
) -> None:
    class ResultOnlyService(PlateInspectionService):
        handler_attempted = False

        @staticmethod
        def _create_handler(request, plate_path, filemanager):
            ResultOnlyService.handler_attempted = True
            raise RuntimeError("handler detection should not run")

    plate = tmp_path / "plate_openhcs"
    config = GlobalPipelineConfig()
    result_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        plate / config.path_planning_config.sub_dir,
    )
    result_dir.mkdir(parents=True)
    (result_dir / "A01_w1_cell_counts_step0_details.csv").write_text(
        "slice_index,cell_count\n0,11\n",
        encoding="utf-8",
    )
    service = ResultOnlyService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    inspection = service.inspect(
        PlatePathInspectionRequest.from_fields(
            plate_path=str(plate),
            max_sample_files=3,
        )
    )

    assert ResultOnlyService.handler_attempted is False
    assert inspection.errors == ()
    assert inspection.status is PlateInspectionStatus.PARTIAL
    assert inspection.confidence is PlateInspectionConfidence.NONE
    assert inspection.result_files.count == 1
    assert any(
        warning.code == PlateInspectionIssueCode.RESULT_FILES_AVAILABLE.value
        for warning in inspection.warnings
    )


def test_plate_inspection_reads_no_main_openhcs_output_subdirectories(
    tmp_path: Path,
) -> None:
    plate = tmp_path / "plate_openhcs"
    step1_dir = plate / "checkpoints_step1"
    step2_dir = plate / "checkpoints_step2"
    result_dir = plate / "checkpoints_step2_results"
    step1_dir.mkdir(parents=True)
    step2_dir.mkdir()
    result_dir.mkdir()
    image_names = (
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    )
    for image_name in image_names:
        (step1_dir / image_name).write_bytes(b"")
        (step2_dir / image_name).write_bytes(b"")
    csv_path = result_dir / "A01_multi_channel_counts_step2_details.csv"
    json_path = result_dir / "A01_multi_channel_counts_step2.json"
    csv_path.write_text("slice_index,colocalized_count\n0,13\n", encoding="utf-8")
    json_path.write_text('{"colocalized_count": 13}\n', encoding="utf-8")
    shared_metadata = {
        "microscope_handler_name": "imagexpress",
        "source_filename_parser_name": "ImageXpressFilenameParser",
        "grid_dimensions": [2, 2],
        "pixel_size": 0.65,
        "channels": {"1": "W1", "2": "W2"},
        "wells": {"A01": None},
        "sites": {"1": None},
        "z_indexes": {"1": None},
        "timepoints": {"1": None},
        "available_backends": {"disk": True},
    }
    (plate / "openhcs_metadata.json").write_text(
        json.dumps(
            {
                "subdirectories": {
                    "checkpoints_step1": {
                        **shared_metadata,
                        "image_files": [
                            f"checkpoints_step1/{image_name}"
                            for image_name in image_names
                        ],
                        "results_dir": "checkpoints_step1_results",
                    },
                    "checkpoints_step2": {
                        **shared_metadata,
                        "image_files": [
                            f"checkpoints_step2/{image_name}"
                            for image_name in image_names
                        ],
                        "results_dir": "checkpoints_step2_results",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    inspection = service.inspect(
        PlatePathInspectionRequest.from_fields(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            max_sample_files=5,
            max_component_values=10,
        )
    )
    channel_summary = next(
        summary
        for summary in inspection.components
        if summary.component is AllComponents.CHANNEL
    )
    timepoint_summary = next(
        summary
        for summary in inspection.components
        if summary.component is AllComponents.TIMEPOINT
    )

    assert inspection.errors == ()
    assert inspection.status is PlateInspectionStatus.OK
    assert inspection.confidence is PlateInspectionConfidence.HIGH
    assert inspection.parser_class == "ImageXpressFilenameParser"
    assert inspection.grid_dimensions == (2, 2)
    assert inspection.pixel_size == 0.65
    assert inspection.image_files.count == 4
    assert inspection.result_files.count == 2
    assert channel_summary.count == 2
    assert timepoint_summary.count == 1
    assert {warning.code for warning in inspection.warnings}.isdisjoint(
        {
            PlateInspectionIssueCode.PARSER_UNAVAILABLE.value,
            PlateInspectionIssueCode.RESULT_FILE_LISTING_FAILED.value,
        }
    )

    result_query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            kind=PlateFileKind.RESULT,
            limit=5,
        )
    )

    assert result_query.errors == ()
    assert result_query.total_count == 2
    assert sorted(record.relative_path for record in result_query.records) == [
        str(path.relative_to(plate)) for path in sorted((csv_path, json_path))
    ]
    csv_record = next(
        record
        for record in result_query.records
        if record.relative_path == str(csv_path.relative_to(plate))
    )
    assert csv_record.preview is not None
    assert csv_record.preview.csv_rows == (
        {"slice_index": "0", "colocalized_count": "13"},
    )
    assert {warning.code for warning in result_query.warnings}.isdisjoint(
        {PlateInspectionIssueCode.RESULT_FILE_LISTING_FAILED.value}
    )


def test_plate_file_query_reads_path_planned_results_for_openhcs_output_root_with_metadata(
    tmp_path: Path,
) -> None:
    plate = tmp_path / "plate_openhcs"
    checkpoint_dir = plate / "checkpoints_step1"
    checkpoint_result_dir = plate / "checkpoints_step1_results"
    current_result_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        plate / GlobalPipelineConfig().path_planning_config.sub_dir,
    )
    checkpoint_dir.mkdir(parents=True)
    checkpoint_result_dir.mkdir()
    current_result_dir.mkdir()
    image_name = "A01_s001_w1_z001_t001.tif"
    (checkpoint_dir / image_name).write_bytes(b"")
    old_csv_path = checkpoint_result_dir / "A01_old_counts_step1.csv"
    current_csv_path = current_result_dir / "A01_w1_cell_counts_step0_details.csv"
    old_csv_path.write_text("slice_index,cell_count\n0,3\n", encoding="utf-8")
    current_csv_path.write_text("slice_index,cell_count\n0,11\n", encoding="utf-8")
    (plate / "openhcs_metadata.json").write_text(
        json.dumps(
            {
                "subdirectories": {
                    "checkpoints_step1": {
                        "microscope_handler_name": "openhcsdata",
                        "source_filename_parser_name": "ImageXpressFilenameParser",
                        "grid_dimensions": [1, 1],
                        "pixel_size": 0.65,
                        "image_files": [f"checkpoints_step1/{image_name}"],
                        "channels": {"1": "1"},
                        "wells": {"A01": None},
                        "sites": {"1": "1"},
                        "z_indexes": {"1": None},
                        "timepoints": {"1": "1"},
                        "available_backends": {"disk": True},
                        "results_dir": "checkpoints_step1_results",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    result_query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            kind=PlateFileKind.RESULT,
            path_contains="images_results",
            limit=5,
        )
    )

    assert result_query.errors == ()
    assert result_query.total_count == 1
    assert [record.relative_path for record in result_query.records] == [
        str(current_csv_path.relative_to(plate)),
    ]
    assert result_query.records[0].preview is not None
    assert result_query.records[0].preview.csv_rows == (
        {"slice_index": "0", "cell_count": "11"},
    )

    auto_checkpoint_query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            kind=PlateFileKind.RESULT,
            path_contains="checkpoints_step1_results",
            limit=5,
        )
    )

    assert auto_checkpoint_query.errors == ()
    assert auto_checkpoint_query.detected_microscope_type == "openhcsdata"
    assert auto_checkpoint_query.total_count == 1
    assert [record.relative_path for record in auto_checkpoint_query.records] == [
        str(old_csv_path.relative_to(plate)),
    ]


def test_plate_inspection_result_preview_detects_csv_table_after_preamble(
    tmp_path: Path,
):
    plate = tmp_path / "plate_openhcs"
    config = GlobalPipelineConfig()
    result_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        plate / config.path_planning_config.sub_dir,
    )
    result_dir.mkdir(parents=True)
    summary_path = result_dir / "metaxpress_style_summary.csv"
    summary_path.write_text(
        "\n".join(
            (
                "Barcode,OpenHCS-images_results,,,,,",
                "Plate Name,images_results,,,,,",
                "Well,Mean Cell Count (W1),Total Cell Count (W1),Mean Cell Count (W2),Total Cell Count (W2)",
                "A01,11.0,11,10.0,10",
            )
        ),
        encoding="utf-8",
    )
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            kind=PlateFileKind.RESULT,
            limit=1,
        )
    )

    assert query.errors == ()
    assert query.total_count == 1
    preview = query.records[0].preview
    assert preview is not None
    assert preview.text_lines == (
        "Barcode,OpenHCS-images_results,,,,,",
        "Plate Name,images_results,,,,,",
        "Well,Mean Cell Count (W1),Total Cell Count (W1),Mean Cell Count (W2),Total Cell Count (W2)",
        "A01,11.0,11,10.0,10",
    )
    assert preview.csv_columns == (
        "Well",
        "Mean Cell Count (W1)",
        "Total Cell Count (W1)",
        "Mean Cell Count (W2)",
        "Total Cell Count (W2)",
    )
    assert preview.csv_rows == (
        {
            "Well": "A01",
            "Mean Cell Count (W1)": "11.0",
            "Total Cell Count (W1)": "11",
            "Mean Cell Count (W2)": "10.0",
            "Total Cell Count (W2)": "10",
        },
    )


def test_plate_inspection_result_preview_parses_multiline_csv_record(
    tmp_path: Path,
):
    plate = tmp_path / "plate_openhcs"
    config = GlobalPipelineConfig()
    result_dir = PathPlannerPathAuthority.analysis_results_dir_for(
        plate / config.path_planning_config.sub_dir,
    )
    result_dir.mkdir(parents=True)
    summary_path = result_dir / "wide_details.csv"
    summary_path.write_text(
        (
            "slice_index,details,cell_count\n"
            '0,"first line\nsecond line\nthird line",11\n'
        ),
        encoding="utf-8",
    )
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    query = service.query_files(
        PlateFileQueryRequest(
            plate_path=str(plate),
            microscope_type="openhcsdata",
            kind=PlateFileKind.RESULT,
            limit=1,
        )
    )

    assert query.errors == ()
    assert query.total_count == 1
    preview = query.records[0].preview
    assert preview is not None
    assert preview.csv_columns == ("slice_index", "details", "cell_count")
    assert preview.csv_rows == (
        {
            "slice_index": "0",
            "details": "first line\nsecond line\nthird line",
            "cell_count": "11",
        },
    )


def test_plate_inspection_downgrades_low_filename_parse_coverage(tmp_path: Path):
    plate = ImageXpressPlateFixture.write(tmp_path)
    image_dir = (
        plate
        / ImageXpressPlateFixture.TIMEPOINT_FOLDER_NAME
        / ImageXpressPlateFixture.ZSTEP_FOLDER_NAME
    )
    for index in range(8):
        (image_dir / f"unrelated_{index}.tif").write_bytes(b"")
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        )
    )

    result = service.inspect(
        PlatePathInspectionRequest.from_fields(plate_path=str(plate))
    )

    assert result.status is PlateInspectionStatus.PARTIAL
    assert result.confidence is PlateInspectionConfidence.LOW
    assert result.parse_summary.attempted_file_count == 10
    assert result.parse_summary.parsed_file_count == 2
    assert any(
        warning.code == PlateInspectionIssueCode.LOW_PARSE_COVERAGE.value
        and warning.hint is not None
        and "wrong folder" in warning.hint
        for warning in result.warnings
    )


def test_plate_inspection_redirects_native_stdout_to_stderr(capfd):
    with AgentStdoutRedirect.to_stderr():
        print("python stdout noise")
        os.write(1, b"fd stdout noise\n")

    captured = capfd.readouterr()
    assert captured.out == ""
    assert "python stdout noise" in captured.err
    assert "fd stdout noise" in captured.err


def test_plate_inspection_reports_path_policy_errors(tmp_path: Path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    service = PlateInspectionService(
        AgentPathPolicy.with_roots(
            readable_roots=(allowed,),
            writable_roots=(allowed,),
        )
    )

    result = service.inspect(
        PlatePathInspectionRequest.from_fields(plate_path=str(outside))
    )

    assert result.status is PlateInspectionStatus.ERROR
    assert result.errors[0].code == PlateInspectionIssueCode.PATH_POLICY_REJECTED.value
    assert (
        result.workflow_advice.ingestion_route
        is PlateInspectionIngestionRoute.UNRESOLVED
    )
    assert "arbitrary TIFF, PNG" in result.workflow_advice.message
    assert "CZI, OME" in result.workflow_advice.message
