from inspect import unwrap

import numpy as np

from openhcs.core.artifacts import (
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.processing.backends.analysis.skan_axon_analysis import (
    AnalysisDimension,
    ThresholdMethod,
    skan_axon_skeletonize_and_analyze,
)


def test_skan_axon_declares_split_typed_outputs_and_object_subject() -> None:
    summary, branches, visualization, labels = (
        skan_axon_skeletonize_and_analyze.__artifact_outputs__
    )

    assert summary.artifact_type is MeasurementsArtifactType
    assert branches.artifact_type is MeasurementsArtifactType
    assert branches.relations[0].measurement_subject().name == labels.name
    assert visualization.artifact_type is ImageArtifactType
    assert labels.artifact_type is ObjectLabelsArtifactType


def test_skan_axon_returns_columnar_rows_and_exact_label_plane_stack() -> None:
    image = np.zeros((2, 24, 24), dtype=np.float32)
    image[0, 12, 3:20] = 1.0

    output, summary, branches, visualization, labels = unwrap(
        skan_axon_skeletonize_and_analyze
    )(
        image,
        threshold_method=ThresholdMethod.MANUAL,
        threshold_value=0.5,
        min_object_size=1,
        analysis_dimension=AnalysisDimension.TWO_D,
    )

    assert output is image
    assert summary.row_count() == 1
    assert branches.row_count() >= 1
    assert visualization.shape == image.shape
    assert labels.shape == image.shape
    assert np.count_nonzero(labels[0]) > 0
    assert np.count_nonzero(labels[1]) == 0
    assert {row["object_label"] for row in branches.row_mappings()} == set(
        range(1, branches.row_count() + 1)
    )
