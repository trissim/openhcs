from inspect import unwrap

import numpy as np

from openhcs.core.artifacts import (
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.processing.backends.analysis.cell_counting_cpu import (
    DetectionMethod,
    _create_segmentation_visualization,
    count_cells_single_channel,
)


def test_cpu_cell_counter_declares_typed_measurements_and_object_labels() -> None:
    measurement_spec, label_spec = count_cells_single_channel.__artifact_outputs__

    assert measurement_spec.name == "cell_counts"
    assert measurement_spec.artifact_type is MeasurementsArtifactType
    assert measurement_spec.relations[0].measurement_subject() is not None
    assert label_spec.name == "segmentation_masks"
    assert label_spec.artifact_type is ObjectLabelsArtifactType


def test_cpu_cell_counter_always_returns_columnar_rows_and_aligned_labels() -> None:
    image = np.zeros((2, 16, 16), dtype=np.float32)
    image[0, 3:7, 3:7] = 1.0
    image[1, 9:13, 9:13] = 1.0

    output, rows, labels = unwrap(count_cells_single_channel)(
        image,
        detection_method=DetectionMethod.THRESHOLD,
        threshold=0.5,
        enable_preprocessing=False,
        min_cell_area=1,
        max_cell_area=100,
        remove_border_cells=False,
    )

    assert output.shape == image.shape
    assert rows.row_count() == 2
    assert labels.shape == image.shape
    assert labels.dtype == np.uint16
    assert [set(np.unique(plane)) for plane in labels] == [{0, 1}, {0, 1}]


def test_cpu_segmentation_visualization_pairs_each_position_with_its_area() -> None:
    labels = _create_segmentation_visualization(
        np.zeros((24, 24), dtype=np.float32),
        positions=[(5.0, 5.0), (17.0, 17.0)],
        max_sigma=20.0,
        cell_areas=[np.pi, 9.0 * np.pi],
    )

    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {0, 1, 2}
    assert np.count_nonzero(labels == 1) == 5
    assert np.count_nonzero(labels == 2) == 29
