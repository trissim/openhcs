from inspect import unwrap
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from polystore.filemanager import FileManager
from polystore.memory import MemoryStorageBackend

from openhcs.core.config import AnalysisConsolidationConfig, PlateMetadataConfig
from openhcs.processing.backends.analysis.consolidate_analysis_results import (
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
    assert results == [
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
    ]
    assert [mask.dtype for mask in masks] == [np.int32, np.int32]
    assert [set(np.unique(mask)) for mask in masks] == [{0, 1, 2}, {0, 1}]


def test_skeletonize_and_save_declares_csv_and_roi_materialization():
    assert skeletonize_and_save.input_memory_type == "numpy"
    assert skeletonize_and_save.output_memory_type == "numpy"
    assert skeletonize_and_save.__special_outputs__ == {
        "skeleton_measurements",
        "skeleton_rois",
    }

    specs = skeletonize_and_save.__materialization_specs__
    csv_options = specs["skeleton_measurements"].outputs[0]
    roi_options = specs["skeleton_rois"].outputs[0]

    assert isinstance(csv_options, CsvOptions)
    assert csv_options.filename_suffix == "_details.csv"
    assert csv_options.fields == SkeletonizationResult.csv_fields()
    assert isinstance(roi_options, ROIOptions)


def test_skeleton_measurements_materialize_as_csv():
    filemanager = FileManager({"memory": MemoryStorageBackend()})
    spec = skeletonize_and_save.__materialization_specs__["skeleton_measurements"]
    measurements = [
        {
            "slice_index": 0,
            "skeleton_count": 2,
            "skeleton_length_pixels": 10,
            "foreground_area_pixels": 20,
            "threshold": 0.5,
        }
    ]

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
    assert csv_frame.to_dict(orient="records") == measurements


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
        consolidation_config=AnalysisConsolidationConfig(),
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
