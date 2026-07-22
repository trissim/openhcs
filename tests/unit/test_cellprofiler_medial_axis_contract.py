"""CellProfiler MedialAxis image-flow contract regressions."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
)
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.medial_axis import (
    MedialaxisModule,
    medialaxis,
)


def test_medial_axis_contract_publishes_declared_skeleton_image() -> None:
    threshold = ArtifactSpec.output("Threshold", ImageArtifactType)
    invocation_key = FunctionInvocationKey("medialaxis", "default", 0)
    module = ModuleBlock(
        name=MedialaxisModule.module_name,
        module_num=3,
        setting_records=[
            ModuleSetting("Select the input image", threshold.name),
            ModuleSetting("Name the output image", "Skeleton"),
        ],
    )
    contract = MedialaxisModule.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_name="MedialAxis",
            step_index=2,
            available_artifacts=ArtifactSpecCollection((threshold,)),
            main_flow_artifacts=ArtifactSpecCollection(
                (threshold.for_plan_type(ArtifactInputPlan),)
            ),
            available_artifact_producers=artifact_producers_for_outputs(
                (threshold,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey("threshold", "default", 0),
                ),
            ),
        ),
    )

    assert contract.artifact_inputs.names() == ("Threshold",)
    assert contract.artifact_outputs.names() == ("Skeleton",)
    assert contract.main_flow_outputs.names() == ("Skeleton",)


def test_medial_axis_callable_returns_thin_binary_skeleton() -> None:
    image = np.zeros((9, 9), dtype=np.float32)
    image[2:7, 2:7] = 1.0

    skeleton = medialaxis(image)

    assert skeleton.dtype == np.float32
    assert set(np.unique(skeleton)) <= {0.0, 1.0}
    assert 0 < np.count_nonzero(skeleton) < np.count_nonzero(image)


def test_medial_axis_callable_is_repeatable_for_tied_pixel_ordering() -> None:
    random_mask = np.random.default_rng(123).random((96, 96)) > 0.89
    image = ndi.binary_dilation(random_mask, iterations=4).astype(np.float32)

    skeletons = [medialaxis(image) for _ in range(5)]

    assert all(np.array_equal(skeletons[0], skeleton) for skeleton in skeletons[1:])
