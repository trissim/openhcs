from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.core.aligned_image_payload import AlignedImageSliceContext
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactInputProjectionPlan,
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.function_patterns import (
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
    compile_function_pattern,
)
from openhcs.core.pipeline.function_contracts import artifact_inputs, artifact_outputs
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.steps.function_output_identity import FunctionOutputIdentity
from openhcs.core.steps.function_output_manifest import (
    FunctionStepOutputProducerIdentityRequest,
    ProducedOutputSemantics,
    StepOutputManifestStore,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


def test_artifact_output_kind_is_declared_by_producer_identity_request() -> None:
    request = FunctionStepOutputProducerIdentityRequest.from_artifact(
        SimpleNamespace(),
        SimpleNamespace(name="cells", artifact_type=ObjectLabelsArtifactType),
    )

    assert request.output_kind == (
        FunctionStepOutputProducerIdentityRequest.ARTIFACT_OUTPUT_KIND
    )


def _compiled_pattern_with_input_edges(
    specs_with_scopes: tuple[tuple[ArtifactSpec, str], ...],
):
    specs = tuple(spec for spec, _scope in specs_with_scopes)

    @artifact_inputs(*specs)
    def consume(image):
        return image

    compiled = compile_function_pattern(consume, {}, {})
    invocation = compiled.default_group.invocations[0]
    edges = []
    for input_index, (spec, source_scope_id) in enumerate(specs_with_scopes):
        storage_plan = ArtifactInputPlan(
            name=spec.name,
            path=spec.name,
            artifact_type=spec.artifact_type,
            source_step_scope_id=source_scope_id,
        )
        producer_scope = storage_plan.producer_group_scope()
        edges.append(
            InvocationArtifactInputEdgePlan(
                key=InvocationArtifactInputProjectionKey(
                    invocation_key=invocation.key,
                    input_index=input_index,
                ),
                spec=spec,
                storage_plan=storage_plan,
                projection=ArtifactInputProjectionPlan(
                    invocation_scope=producer_scope,
                    producer_selection_scope=producer_scope,
                ),
            )
        )
    invocation = invocation.with_artifact_input_edges(tuple(edges))
    group = replace(compiled.default_group, invocations=(invocation,))
    return replace(compiled, groups=(group,))


@pytest.mark.parametrize(
    (
        "dependency_scope",
        "producer_output_name",
        "producer_artifact_type",
        "producer_channel",
        "foreign_inputs",
    ),
    (
        pytest.param(
            "display_data",
            "DisplayImage",
            ImageArtifactType,
            2,
            (
                (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType), "identify"),
                (
                    ArtifactSpec.input("ObjectMeasurements", MeasurementsArtifactType),
                    "measure_intensity",
                ),
            ),
            id="ExamplePercentPositive-ClassifyObjects",
        ),
        pytest.param(
            "identify_tumor",
            "tumor",
            ObjectLabelsArtifactType,
            1,
            (
                (ArtifactSpec.input("GrayTumor", ImageArtifactType), "color_to_gray"),
                (ArtifactSpec.input("GrayLung", ImageArtifactType), "color_to_gray"),
            ),
            id="ExampleTumor-ImageMath",
        ),
    ),
)
def test_foreign_artifact_inputs_do_not_filter_lifecycle_producer(
    tmp_path: Path,
    dependency_scope: str,
    producer_output_name: str,
    producer_artifact_type,
    producer_channel: int,
    foreign_inputs: tuple[tuple[ArtifactSpec, str], ...],
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id=dependency_scope,
        step_name="LifecycleProducer",
        pipeline_position=1,
        axis_id="A01",
        output_dir=output_dir,
    )
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id=dependency_scope,
        ),
        compiled_function_pattern=_compiled_pattern_with_input_edges(foreign_inputs),
    )
    producer_path = output_dir / (
        f"A01_s001_w{producer_channel}_z001_t001_{producer_output_name}.tif"
    )
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        (
            ProducedOutputSemantics.from_output(
                producer,
                producer_path,
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": producer_channel,
                        "z_index": 1,
                        "timepoint": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key=producer_output_name,
                    artifact_kind=producer_artifact_type.value,
                ),
            ),
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        [
            "A01_s001_w9_z001_t001_unrelated.tif",
            producer_path.name,
        ],
        SourceSchemaFilenameParser(),
    ) == [producer_path.name]


def test_compiled_main_flow_edge_selects_exact_producer_identity(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="align",
        step_name="Align",
        pipeline_position=1,
        axis_id="A01",
        output_dir=output_dir,
    )
    input_spec = ArtifactSpec.input("Stain1", ImageArtifactType)

    @artifact_inputs(input_spec)
    def consume(image):
        return image

    compiled = compile_function_pattern(consume, {}, {})
    invocation = compiled.default_group.invocations[0]
    invocation = invocation.with_artifact_input_edges(
        (
            InvocationArtifactInputEdgePlan(
                key=InvocationArtifactInputProjectionKey(
                    invocation_key=invocation.key,
                    input_index=0,
                ),
                spec=input_spec,
                storage_plan=None,
                projection=None,
                consumes_main_flow=True,
            ),
        )
    )
    compiled = replace(
        compiled,
        groups=(
            replace(compiled.default_group, invocations=(invocation,)),
        ),
    )
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="align",
        ),
        compiled_function_pattern=compiled,
    )
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        tuple(
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / f"A01_s001_w{channel}_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": channel,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key=output_key,
                    artifact_kind=ImageArtifactType.value,
                ),
            )
            for channel, output_key in ((1, "Stain1"), (2, "Stain2"))
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        [
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        ],
        SourceSchemaFilenameParser(),
    ) == ["A01_s001_w1_z001_t001.tif"]


def test_storage_backed_primary_input_selects_exact_lifecycle_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="color_to_gray",
        step_name="ColorToGray",
        pipeline_position=1,
        axis_id="A01",
        output_dir=output_dir,
    )
    source = ArtifactSpec.input("OrigRed", ImageArtifactType)
    illumination = ArtifactSpec.output_inheriting_group_scope(
        "IllumRed",
        ImageArtifactType,
        source,
    )

    @artifact_inputs(source)
    @artifact_outputs(illumination)
    def calculate_illumination(image):
        return image

    compiled = compile_function_pattern(calculate_illumination, {}, {})
    invocation = compiled.default_group.invocations[0]
    storage_plan = ArtifactInputPlan(
        name=source.name,
        path=source.name,
        artifact_type=source.artifact_type,
        source_step_scope_id="color_to_gray",
    )
    producer_scope = storage_plan.producer_group_scope()
    invocation = invocation.with_artifact_input_edges(
        (
            InvocationArtifactInputEdgePlan(
                key=InvocationArtifactInputProjectionKey(
                    invocation_key=invocation.key,
                    input_index=0,
                ),
                spec=source,
                storage_plan=storage_plan,
                projection=ArtifactInputProjectionPlan(
                    invocation_scope=producer_scope,
                    producer_selection_scope=producer_scope,
                ),
            ),
        )
    )
    compiled = replace(
        compiled,
        groups=(replace(compiled.default_group, invocations=(invocation,)),),
    )
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="color_to_gray",
        ),
        compiled_function_pattern=compiled,
    )
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        tuple(
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / f"A01_s001_w1_z001_t001_{output_key}.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key=output_key,
                    artifact_kind=ImageArtifactType.value,
                ),
            )
            for output_key in ("OrigRed", "OrigGreen", "OrigBlue")
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        [
            "A01_s001_w1_z001_t001_OrigBlue.tif",
            "A01_s001_w1_z001_t001_OrigGreen.tif",
            "A01_s001_w1_z001_t001_OrigRed.tif",
        ],
        SourceSchemaFilenameParser(),
    ) == ["A01_s001_w1_z001_t001_OrigRed.tif"]


def test_storage_backed_input_does_not_reclassify_lifecycle_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="artifact_producer",
        step_name="ArtifactProducer",
        pipeline_position=1,
        axis_id="A01",
        output_dir=output_dir,
    )
    stored_input = ArtifactSpec.input("StoredLabels", ObjectLabelsArtifactType)
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="artifact_producer",
        ),
        compiled_function_pattern=_compiled_pattern_with_input_edges(
            ((stored_input, "artifact_producer"),)
        ),
    )
    lifecycle_path = output_dir / "A01_s001_w2_z001_t001.tif"
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        (
            ProducedOutputSemantics.from_output(
                producer,
                lifecycle_path,
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": 2,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key="LifecycleImage",
                    artifact_kind=ImageArtifactType.value,
                ),
            ),
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        [
            "A01_s001_w1_z001_t001.tif",
            lifecycle_path.name,
        ],
        SourceSchemaFilenameParser(),
    ) == [lifecycle_path.name]


@pytest.mark.parametrize(
    ("artifact_type", "parameter_name"),
    (
        pytest.param(
            ObjectLabelsArtifactType,
            "labels",
            id="object-label-runtime-argument",
        ),
        pytest.param(
            ImageArtifactType,
            "illumination_function",
            id="auxiliary-image-runtime-argument",
        ),
    ),
)
def test_same_scope_parameter_bound_input_does_not_select_lifecycle_output(
    tmp_path: Path,
    artifact_type,
    parameter_name: str,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="artifact_producer",
        step_name="ArtifactProducer",
        pipeline_position=1,
        axis_id="A01",
        output_dir=output_dir,
    )
    auxiliary_input = ArtifactSpec.input(
        "AuxiliaryArtifact",
        artifact_type,
        parameter_name=parameter_name,
    )
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="artifact_producer",
        ),
        compiled_function_pattern=_compiled_pattern_with_input_edges(
            ((auxiliary_input, "artifact_producer"),)
        ),
    )
    producer_path = output_dir / "A01_s001_w1_z001_t001.tif"
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        (
            ProducedOutputSemantics.from_output(
                producer,
                producer_path,
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key="PreservedInput",
                    artifact_kind=ImageArtifactType.value,
                ),
            ),
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        [producer_path.name],
        SourceSchemaFilenameParser(),
    ) == [producer_path.name]
