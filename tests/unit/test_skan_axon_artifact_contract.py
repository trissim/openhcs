from inspect import unwrap

import numpy as np
import pandas as pd
import pytest

from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.processing.backends.analysis.skan_axon_analysis import (
    AnalysisDimension,
    SkanAxonInvocationContractProviderFactory,
    ThresholdMethod,
    _compute_summary_metrics,
    skan_axon_skeletonize_and_analyze,
)
from openhcs.processing.materialization import TiffStackOptions


def test_skan_axon_declares_split_typed_outputs_and_object_subject() -> None:
    summary, branches, visualization, labels = (
        skan_axon_skeletonize_and_analyze.__artifact_outputs__
    )

    assert summary.artifact_type is MeasurementsArtifactType
    assert branches.artifact_type is MeasurementsArtifactType
    assert branches.relations[0].measurement_subject().name == labels.name
    assert visualization.artifact_type is ImageArtifactType
    assert labels.artifact_type is ObjectLabelsArtifactType

    visualization_options = visualization.materialization.outputs[0]
    assert isinstance(visualization_options, TiffStackOptions)
    assert visualization_options.normalize_uint8 is False


def test_skan_visualization_contract_preserves_exact_main_flow_stack() -> None:
    source = ArtifactSpec.input("preprocessed_axon", ImageArtifactType)
    context = ArtifactDeclarationStepContext(
        step_name="Full-field axon network",
        step_index=3,
        main_flow_artifacts=ArtifactSpecCollection((source,)),
    )
    invocation = next(
        normalize_function_pattern(skan_axon_skeletonize_and_analyze).iter_items()
    )
    provider = SkanAxonInvocationContractProviderFactory.provider_for_session(None)

    assert provider is not None
    plan = provider(invocation, context)

    assert plan is not None
    assert plan.contract.artifact_inputs == ArtifactSpecCollection((source,))
    visualization = next(
        output
        for output in plan.contract.artifact_outputs
        if output.name == "skeleton_visualizations"
    )
    assert visualization.source_context_sources() == (source.ref(),)
    assert visualization.source_stack_scope_sources() == (source.ref(),)


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


def test_skan_summary_tortuosity_uses_finite_positive_euclidean_branches() -> None:
    branches = pd.DataFrame(
        {
            "branch_distance": [5.0, 100.0, np.nan, np.inf],
            "euclidean_distance": [4.0, 0.0, 3.0, 2.0],
            "branch_type": [0, 3, 1, 2],
            "node_id_src": [1, 2, 3, 4],
        }
    )

    summary = _compute_summary_metrics(
        branches,
        skeleton_shape=(1, 16, 16),
        voxel_spacing=(1.0, 0.5, 0.5),
    )

    assert summary["num_branches"] == 4
    assert summary["mean_tortuosity"] == pytest.approx(1.25)
