"""Exact named-image projection through aligned main-flow carriers."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
    ImageOutputBundle,
    compose_aligned_image_payload,
)
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    GroupLineageSourceRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerModuleExecutor,
)
from openhcs.processing.backends.cellprofiler.color import color_to_gray


class _CompiledInputRequest:
    def __init__(self, specs: tuple[ArtifactSpec, ...]) -> None:
        self._specs = ArtifactSpecCollection(specs)
        self.artifact_inputs = {}

    def selected_artifact_input_specs(self) -> ArtifactSpecCollection:
        return self._specs


def _input_ref(
    name: str,
    artifact_type: type = ImageArtifactType,
) -> ArtifactSpecRef:
    return ArtifactSpec.input(name, artifact_type).ref()


def _named_carrier() -> tuple[AlignedImageStack, np.ndarray, np.ndarray]:
    mcherry = np.full((3, 4), 3.0, dtype=np.float32)
    gfp = np.full((3, 4), 2.0, dtype=np.float32)
    return (
        AlignedImageStack(
            (mcherry, gfp),
            (
                AlignedImageSliceContext.main_flow(
                    "mCherry",
                    artifact_kind=ImageArtifactType.require_value(),
                ),
                AlignedImageSliceContext.main_flow(
                    "GFP",
                    artifact_kind=ImageArtifactType.require_value(),
                ),
            ),
        ),
        mcherry,
        gfp,
    )


def _related_image_output(
    output_name: str,
    source_name: str,
) -> tuple[ArtifactOutputPlan, ArtifactSpec]:
    source = ArtifactSpec.input(source_name, ImageArtifactType)
    output = ArtifactSpec.output(
        output_name,
        ImageArtifactType,
        relations=(GroupLineageSourceRelation(source=source.ref()),),
    )
    return (
        ArtifactOutputPlan(
            name=output.name,
            path=f"/memory/{output.name}.pkl",
            artifact_type=output.artifact_type,
            relations=output.relations,
        ),
        output,
    )


def _single_output_bundle(name: str, payload: np.ndarray) -> ImageOutputBundle:
    return ImageOutputBundle(
        (payload,),
        (
            AlignedImageSliceContext.main_flow(
                name,
                artifact_kind=ImageArtifactType.require_value(),
            ),
        ),
    )


def _compiled_image_output(
    spec: ArtifactSpec,
    *,
    channel: str | None = None,
) -> ArtifactOutputPlan:
    return ArtifactOutputPlan(
        name=spec.name,
        path=f"/memory/{spec.name}.pkl",
        artifact_type=spec.artifact_type,
        relations=spec.relations,
        group_keys=(channel,),
        group_component=(None if channel is None else AllComponents.CHANNEL),
    )


def test_same_source_output_siblings_own_independent_projections() -> None:
    source = ArtifactSpec.input("Worms", ImageArtifactType)
    outputs = (
        ArtifactSpec.output(
            "OverlappedWormOutlines",
            ImageArtifactType,
            relations=(GroupLineageSourceRelation(source=source.ref()),),
        ),
        ArtifactSpec.output(
            "NonoverlappedWormOutlines",
            ImageArtifactType,
            relations=(GroupLineageSourceRelation(source=source.ref()),),
        ),
    )

    contexts = AlignedImageSliceContext.main_flow_for_output_plans(
        tuple(_compiled_image_output(output, channel="1") for output in outputs)
    )

    assert tuple(context.output_key for context in contexts) == (
        "OverlappedWormOutlines",
        "NonoverlappedWormOutlines",
    )
    assert tuple(context.projection_key for context in contexts) == (
        "OverlappedWormOutlines",
        "NonoverlappedWormOutlines",
    )


def test_distinct_source_outputs_require_compiled_disjoint_coordinate_proof() -> None:
    red = ArtifactSpec.input("Red", ImageArtifactType)
    green = ArtifactSpec.input("Green", ImageArtifactType)
    outputs = (
        ArtifactSpec.output(
            "StraightRed",
            ImageArtifactType,
            relations=(GroupLineageSourceRelation(source=red.ref()),),
        ),
        ArtifactSpec.output(
            "StraightGreen",
            ImageArtifactType,
            relations=(GroupLineageSourceRelation(source=green.ref()),),
        ),
    )

    contexts = AlignedImageSliceContext.main_flow_for_output_plans(
        tuple(_compiled_image_output(output, channel="1") for output in outputs)
    )

    assert tuple(context.projection_key for context in contexts) == (
        "StraightRed",
        "StraightGreen",
    )


def test_disjoint_compiled_output_coordinates_share_component_projection() -> None:
    outputs = (
        ArtifactSpec.output("StraightRed", ImageArtifactType),
        ArtifactSpec.output("StraightGreen", ImageArtifactType),
    )

    contexts = AlignedImageSliceContext.main_flow_for_output_plans(
        tuple(
            _compiled_image_output(output, channel=channel)
            for output, channel in zip(outputs, ("1", "2"), strict=True)
        )
    )

    assert tuple(context.projection_key for context in contexts) == ("main", "main")


def test_single_declared_output_uses_ordinary_main_projection() -> None:
    output = ArtifactSpec.output("CorrectedDNA", ImageArtifactType)

    (context,) = AlignedImageSliceContext.main_flow_for_output_plans(
        (_compiled_image_output(output),)
    )

    assert context.output_key == "CorrectedDNA"
    assert context.projection_key == "main"


def test_precompiled_multi_output_contexts_never_infer_projection_from_sources() -> (
    None
):
    source = ArtifactSpec.input("Composite", ImageArtifactType)
    outputs = tuple(
        ArtifactSpec.output(
            name,
            ImageArtifactType,
            relations=(GroupLineageSourceRelation(source=source.ref()),),
        )
        for name in ("Red", "Green")
    )

    contexts = AlignedImageSliceContext.main_flow_for_artifact_specs(outputs)

    assert tuple(context.projection_key for context in contexts) == ("Red", "Green")


def test_aligned_carrier_projects_exact_compiled_ref() -> None:
    carrier, mcherry, _gfp = _named_carrier()
    spec = ArtifactSpec.input("mCherry", ImageArtifactType)
    adapter = SimpleNamespace(request=_CompiledInputRequest((spec,)))
    request = RuntimeInputBindingRequest(
        adapter=adapter,
        kwargs={},
        current_image=carrier,
    )

    assert request.main_flow_value(spec) is mcherry
    assert carrier.output_payload(_input_ref("mCherry")) is mcherry
    assert carrier.output_payload(_input_ref("missing")) is None
    assert (
        carrier.output_payload(_input_ref("mCherry", ObjectLabelsArtifactType)) is None
    )


def test_untyped_context_does_not_match_typed_artifact_ref() -> None:
    payload = np.zeros((3, 4), dtype=np.float32)
    carrier = AlignedImageStack(
        (payload,),
        (AlignedImageSliceContext.main_flow("mCherry"),),
    )

    assert carrier.output_payload(_input_ref("mCherry")) is None


def test_non_main_context_does_not_match_main_flow_artifact_ref() -> None:
    payload = np.zeros((3, 4), dtype=np.float32)
    carrier = AlignedImageStack(
        (payload,),
        (
            AlignedImageSliceContext(
                output_kind="artifact",
                output_key="mCherry",
                projection_key="mCherry",
                artifact_kind=ImageArtifactType.require_value(),
            ),
        ),
    )

    assert carrier.output_payload(_input_ref("mCherry")) is None


def test_runtime_binding_rejects_missing_exact_carrier_ref() -> None:
    carrier, _mcherry, _gfp = _named_carrier()
    missing = ArtifactSpec.input("DAPI", ImageArtifactType)
    request = RuntimeInputBindingRequest(
        adapter=SimpleNamespace(request=_CompiledInputRequest((missing,))),
        kwargs={},
        current_image=carrier,
    )

    with pytest.raises(ValueError, match="does not carry callable input"):
        request.main_flow_value(missing)


def test_aligned_carrier_rejects_duplicate_exact_context() -> None:
    context = AlignedImageSliceContext.main_flow(
        "mCherry",
        artifact_kind=ImageArtifactType.require_value(),
    )
    carrier = AlignedImageStack(
        (
            np.zeros((3, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
        ),
        (context, context),
    )

    with pytest.raises(ValueError, match="duplicate main-flow output context"):
        carrier.output_payload(_input_ref("mCherry"))


def test_generic_aligned_composition_preserves_shared_exact_contexts() -> None:
    carrier, _mcherry, _gfp = _named_carrier()
    supplemental = AlignedImageStack(
        (
            np.zeros((3, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
        )
    )

    composition = compose_aligned_image_payload(
        "generic aligned composition",
        (carrier, supplemental),
    )

    assert isinstance(composition.payload, AlignedImageStack)
    assert composition.payload.slice_contexts == carrier.slice_contexts
    assert composition.payload.output_payload(_input_ref("mCherry")) is (
        composition.payload.slices[0]
    )


def test_generic_aligned_composition_rejects_conflicting_exact_contexts() -> None:
    first, _mcherry, _gfp = _named_carrier()
    second = AlignedImageStack(
        (
            np.zeros((3, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
        ),
        (
            AlignedImageSliceContext.main_flow(
                "DNA",
                artifact_kind=ImageArtifactType.require_value(),
            ),
            AlignedImageSliceContext.main_flow(
                "Protein",
                artifact_kind=ImageArtifactType.require_value(),
            ),
        ),
    )

    with pytest.raises(ValueError, match="conflicting exact slice contexts"):
        compose_aligned_image_payload(
            "generic aligned composition",
            (first, second),
        )


def test_named_output_transformation_preserves_base_carrier_for_later_exact_input() -> (
    None
):
    mcherry = np.full((3, 4), 3.0, dtype=np.float32)
    current = AlignedImageStack(
        (mcherry,),
        (
            AlignedImageSliceContext.main_flow(
                "mCherry",
                artifact_kind=ImageArtifactType.require_value(),
            ),
        ),
    )
    first_output = np.full((3, 4, 3), 1.0, dtype=np.float32)
    first_plan = _related_image_output(
        "GrayToColor_10_image_1",
        "Straightened_mCherry",
    )

    after_first = CellProfilerModuleExecutor._merge_named_image_outputs(
        current,
        _single_output_bundle(first_plan[1].name, first_output),
        (first_plan,),
    )

    assert isinstance(after_first, ImageOutputBundle)
    assert after_first.output_payload(_input_ref("mCherry")) is mcherry
    assert (
        after_first.output_payload(_input_ref("GrayToColor_10_image_1")) is first_output
    )

    second_output = np.full((3, 4, 3), 2.0, dtype=np.float32)
    second_plan = _related_image_output("OrigRG", "mCherry")
    after_second = CellProfilerModuleExecutor._merge_named_image_outputs(
        after_first,
        _single_output_bundle(second_plan[1].name, second_output),
        (second_plan,),
    )

    assert isinstance(after_second, ImageOutputBundle)
    assert after_second.output_payload(_input_ref("mCherry")) is None
    assert after_second.output_payload(_input_ref("OrigRG")) is second_output
    assert (
        after_second.output_payload(_input_ref("GrayToColor_10_image_1"))
        is first_output
    )


def test_active_publisher_uses_exact_canonical_artifact_kind() -> None:
    image = ArtifactSpec.output("Straightened_mCherry", ImageArtifactType)
    measurements = ArtifactSpec.output("Measurements", MeasurementsArtifactType)
    labels = ArtifactSpec.output("StraightenedWorms", ObjectLabelsArtifactType)
    contract = CallableContract.from_callable(color_to_gray)
    contract = replace(
        contract,
        metadata=replace(
            contract.metadata,
            artifact_outputs=(image, measurements, labels),
            runtime_adapter=CellProfilerRuntimeAdapter.runtime_adapter_spec(),
        ),
    )
    executor = CellProfilerModuleExecutor(color_to_gray, contract)
    image_plan = ArtifactOutputPlan(
        name=image.name,
        path=f"/memory/{image.name}.pkl",
        artifact_type=image.artifact_type,
    )
    measurement_plan = ArtifactOutputPlan(
        name=measurements.name,
        path=f"/memory/{measurements.name}.pkl",
        artifact_type=measurements.artifact_type,
    )
    labels_plan = ArtifactOutputPlan(
        name=labels.name,
        path=f"/memory/{labels.name}.pkl",
        artifact_type=labels.artifact_type,
    )
    image_value = np.ones((3, 4), dtype=np.float32)
    measurement_value = object()
    labels_value = object()
    values = {
        image_plan.ref(): image_value,
        measurement_plan.ref(): measurement_value,
        labels_plan.ref(): labels_value,
    }
    requested_refs: list[ArtifactSpecRef] = []

    def artifact_output_value(plan: ArtifactOutputPlan) -> object:
        requested_refs.append(plan.ref())
        return values[plan.ref()]

    assert contract.main_flow_outputs.specs == (image, labels)
    assert contract.canonical_return_output_specs.specs == (image,)
    result = executor._published_active_main_flow_output(
        matched_outputs=(
            (image_plan, image, image_value),
            (measurement_plan, measurements, measurement_value),
            (labels_plan, labels, labels_value),
        ),
        declared_only_outputs={},
        adapter=SimpleNamespace(artifact_output_value=artifact_output_value),
        current_image=np.zeros((3, 4), dtype=np.float32),
        invocation_image=np.zeros((3, 4), dtype=np.float32),
        plane_projection=None,
    )

    assert isinstance(result, ImageOutputBundle)
    assert result.output_payload(_input_ref(image.name)) is image_value
    assert (
        result.output_payload(_input_ref(measurements.name, measurements.artifact_type))
        is None
    )
    assert result.output_payload(_input_ref(labels.name, labels.artifact_type)) is None
    assert requested_refs == [image_plan.ref()]
