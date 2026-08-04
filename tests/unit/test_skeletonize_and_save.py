from inspect import unwrap
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from polystore.filemanager import FileManager
from polystore.memory import MemoryStorageBackend

from openhcs.core.artifacts import (
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.config import AnalysisConsolidationConfig, PlateMetadataConfig
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.processing.backends.analysis.consolidate_analysis_results import (
    MaterializedAnalysisTableFile,
    consolidate_analysis_file_groups,
    consolidate_materialized_analysis_table_file_groups,
    consolidate_analysis_results,
)
from openhcs.processing.backends.analysis.skeletonize_and_save import (
    SkeletonizationResult,
    skeletonize_and_save,
)
from openhcs.processing.materialization import CsvOptions, ROIOptions, materialize


def _skeletonize_and_save_impl():
    return unwrap(skeletonize_and_save)


def test_skeletonize_and_save_emits_measurements_and_labeled_masks():
    image = np.zeros((2, 12, 12), dtype=np.float32)
    image[0, 2:7, 2] = 1.0
    image[0, 3:8, 8] = 1.0
    image[1, 2:7, 2] = 1.0
    image[1, 9:11, 9] = 1.0

    output, results, masks = _skeletonize_and_save_impl()(
        image,
        threshold=0.5,
        min_component_size=3,
    )

    assert output is image
    assert results.row_mappings() == (
        {
            "slice_index": 0,
            "skeleton_count": 2,
            "skeleton_length_pixels": 10,
            "foreground_area_pixels": 10,
            "threshold": 0.5,
        },
        {
            "slice_index": 1,
            "skeleton_count": 1,
            "skeleton_length_pixels": 5,
            "foreground_area_pixels": 7,
            "threshold": 0.5,
        },
    )
    assert masks.dtype == np.int32
    assert [set(np.unique(mask)) for mask in masks] == [{0, 1, 2}, {0, 1}]


def test_skeletonize_and_save_declares_csv_and_roi_materialization():
    contract = CallableContract.from_callable(skeletonize_and_save)
    assert contract.input_memory_type == "numpy"
    assert contract.output_memory_type == "numpy"
    assert contract.artifact_outputs.names() == (
        "skeleton_measurements",
        "skeleton_rois",
    )

    measurement_spec, label_spec = contract.artifact_outputs
    csv_options = measurement_spec.materialization.outputs[0]
    roi_options = label_spec.materialization.outputs[0]

    assert isinstance(csv_options, CsvOptions)
    assert csv_options.filename_suffix == "_details.csv"
    assert csv_options.fields is None
    assert isinstance(roi_options, ROIOptions)
    assert roi_options.min_area == 1
    assert measurement_spec.artifact_type is MeasurementsArtifactType
    assert measurement_spec.relations[0].measurement_subject() is not None
    assert label_spec.artifact_type is ObjectLabelsArtifactType


def test_skeleton_measurements_materialize_as_csv():
    filemanager = FileManager({"memory": MemoryStorageBackend()})
    spec = (
        CallableContract.from_callable(skeletonize_and_save)
        .artifact_outputs[0]
        .materialization
    )
    measurement = SkeletonizationResult(
        slice_index=0,
        skeleton_count=2,
        skeleton_length_pixels=10,
        foreground_area_pixels=20,
        threshold=0.5,
    )
    measurements = DataclassMeasurementColumnarRows(
        (measurement,),
        row_type=SkeletonizationResult,
    )

    output_path = materialize(
        spec,
        data=measurements,
        path="/tmp/A01_skeleton_measurements_step1",
        filemanager=filemanager,
        backends=["memory"],
        backend_kwargs={},
    )

    assert output_path == "/tmp/A01_skeleton_measurements_step1_details.csv"
    csv_frame = pd.read_csv(StringIO(filemanager.load(output_path, "memory")))
    assert csv_frame.to_dict(orient="records") == [dict(measurements.row_mappings()[0])]


def test_skeleton_roi_materialization_retains_thin_components():
    image = np.zeros((1, 12, 12), dtype=np.float32)
    image[0, 2:7, 2] = 1.0
    _, _, masks = _skeletonize_and_save_impl()(
        image,
        threshold=0.5,
        min_component_size=1,
    )
    filemanager = FileManager({"memory": MemoryStorageBackend()})
    spec = (
        CallableContract.from_callable(skeletonize_and_save)
        .artifact_outputs[1]
        .materialization
    )

    output_path = materialize(
        spec,
        data=masks,
        path="/tmp/A01_skeleton_rois_step1",
        filemanager=filemanager,
        backends=["memory"],
        backend_kwargs={},
    )

    assert output_path == "/tmp/A01_skeleton_rois_step1_rois.roi.zip"
    assert len(filemanager.load(output_path, "memory")) == 1


def test_skeleton_measurements_generate_metaxpress_style_summary(tmp_path):
    details_path = tmp_path / "A01_skeleton_measurements_step1_details.csv"
    pd.DataFrame(
        [
            {
                "slice_index": 0,
                "skeleton_count": 2,
                "skeleton_length_pixels": 10,
                "foreground_area_pixels": 20,
                "threshold": 0.5,
            },
            {
                "slice_index": 1,
                "skeleton_count": 3,
                "skeleton_length_pixels": 15,
                "foreground_area_pixels": 30,
                "threshold": 0.5,
            },
        ]
    ).to_csv(details_path, index=False)

    output_path = tmp_path / "metaxpress_style_summary.csv"
    summary = consolidate_analysis_results(
        results_directory=str(tmp_path),
        well_ids=["A01"],
        analysis_consolidation_config=AnalysisConsolidationConfig(),
        plate_metadata_config=PlateMetadataConfig(),
        output_path=str(output_path),
    )

    assert summary.loc[0, "Well"] == "A01"
    count_column = next(
        column
        for column in summary.columns
        if column.startswith("Total Skeleton Count")
    )
    length_column = next(
        column
        for column in summary.columns
        if column.startswith("Total Skeleton Length Pixels")
    )
    assert summary.loc[0, count_column] == pytest.approx(5)
    assert summary.loc[0, length_column] == pytest.approx(25)

    rows = output_path.read_text().splitlines()
    assert rows[0].startswith("Barcode,")
    assert rows[1].startswith("Plate Name,")
    assert rows[6].startswith("Well,")


def test_execution_file_groups_skip_non_table_materializations(tmp_path):
    roi_path = tmp_path / "A01_skeleton_rois_step1.roi.zip"
    roi_path.write_bytes(b"not-a-table")
    config = AnalysisConsolidationConfig()

    successful_dirs, failed_dirs = consolidate_analysis_file_groups(
        analysis_files_by_directory={tmp_path: (roi_path,)},
        plate_path=tmp_path,
        analysis_consolidation_config=config,
        plate_metadata_config=PlateMetadataConfig(),
        filename_parser=object(),
    )

    assert successful_dirs == []
    assert failed_dirs == []
    assert not (tmp_path / config.output_filename).exists()


def test_execution_table_groups_use_runtime_identity_not_filename_parser(tmp_path):
    details_path = (
        tmp_path
        / "Image15_site-1_z_index-1_timepoint-1_cell_counts_step2_details.csv"
    )
    pd.DataFrame(({"cell_count": 7},)).to_csv(details_path, index=False)
    config = AnalysisConsolidationConfig()

    successful_dirs, failed_dirs = (
        consolidate_materialized_analysis_table_file_groups(
            analysis_files_by_directory={
                tmp_path: (
                    MaterializedAnalysisTableFile(
                        path=details_path,
                        well_id="Image15",
                        analysis_type=(
                            "site-1_z_index-1_timepoint-1_cell_counts_step2"
                        ),
                    ),
                )
            },
            plate_path=tmp_path,
            analysis_consolidation_config=config,
            plate_metadata_config=PlateMetadataConfig(),
        )
    )

    assert successful_dirs == [tmp_path.name]
    assert failed_dirs == []
    summary = pd.read_csv(tmp_path / config.output_filename, skiprows=6)
    assert summary.loc[0, "Well"] == "Image15"
    count_column = next(
        column for column in summary.columns if column.startswith("Total Cell Count")
    )
    assert summary.loc[0, count_column] == pytest.approx(7)


@pytest.mark.parametrize(
    ("image", "min_component_size", "message"),
    [
        (np.zeros((8, 8)), 1, "expects a 3D array"),
        (np.zeros((1, 8, 8)), 0, "must be at least 1"),
    ],
)
def test_skeletonize_and_save_validates_inputs(
    image,
    min_component_size,
    message,
):
    with pytest.raises(ValueError, match=message):
        _skeletonize_and_save_impl()(
            image,
            threshold=0.5,
            min_component_size=min_component_size,
        )
