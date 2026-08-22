from dataclasses import replace

import numpy as np
import pytest
from arraybridge.decorators import DtypeConversion, DtypeConversionConfig

from openhcs.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactInputProjectionPlan,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactType,
    GroupLineageSourceRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
    MeasurementsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.callable_contract import CallableContract, FunctionStepExecutionScope
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.config import DtypeConfig
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
    RuntimeParameterBinding,
    RuntimeInvocationDomain,
    compile_function_pattern,
    inject_artifact_input_values,
    inject_kwargs_into_pattern,
    iter_enabled_function_invocations,
    normalize_function_pattern,
    strip_disabled_functions,
)
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    artifact_outputs,
    execution_scope,
    runtime_bound_parameters,
    special_inputs,
    validate_artifact_input_parameter_bindings,
)
from openhcs.core.pipeline.artifact_planning import (
    extract_artifact_declarations,
    normalize_pattern,
)
from openhcs.core.runtime_adapters import runtime_adapter
from openhcs.core.runtime_object_labels import ObjectLabelValue
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.steps.function_runtime import ComponentArtifactPlans
from openhcs.processing.materialization import csv_only


@artifact_outputs(ArtifactSpec.output("positions", SpecialArtifactType))
def first(image, *, sigma=None, dtype_config=None):
    return image


@artifact_outputs(ArtifactSpec.output("measurements", SpecialArtifactType))
def second(image, *, sigma=None, dtype_config=None):
    return image


def skipped(image):
    return image


@artifact_outputs(
    ArtifactSpec.output("positions", SpecialArtifactType),
    ArtifactSpec.output("measurements", SpecialArtifactType),
)
def third(image):
    return image


def exact_input_edge(
    invocation,
    *,
    input_index: int,
    spec: ArtifactSpec,
    storage_plan: ArtifactInputPlan,
    parameter_name: str | None,
) -> InvocationArtifactInputEdgePlan:
    producer_scope = storage_plan.producer_group_scope()
    return InvocationArtifactInputEdgePlan(
        key=InvocationArtifactInputProjectionKey(
            invocation_key=invocation.key,
            input_index=input_index,
        ),
        spec=replace(spec, parameter_name=parameter_name),
        storage_plan=storage_plan,
        projection=ArtifactInputProjectionPlan(
            invocation_scope=producer_scope,
            producer_selection_scope=producer_scope,
        ),
    )


def test_invocation_keys_preserve_callable_positions_for_list_patterns():
    pattern = [
        first,
        (skipped, {"enabled": False}),
        (second, {"sigma": 1}),
    ]

    keys = [invocation.key for invocation in iter_enabled_function_invocations(pattern)]

    assert keys == [
        FunctionInvocationKey("first", "default", 0),
        FunctionInvocationKey("second", "default", 1),
    ]


def test_invocation_positions_are_renumbered_per_dict_group():
    pattern = {
        "DAPI": [
            first,
            (skipped, {"enabled": False}),
            second,
        ],
        "GFP": [
            third,
        ],
    }

    keys = [invocation.key for invocation in iter_enabled_function_invocations(pattern)]

    assert keys == [
        FunctionInvocationKey("first", "DAPI", 0),
        FunctionInvocationKey("second", "DAPI", 1),
        FunctionInvocationKey("third", "GFP", 0),
    ]


def test_artifact_planning_normalize_pattern_returns_tuple_api():
    pattern = [first, (skipped, {"enabled": False}), second]

    normalized = [
        (func.__name__, group_key, position)
        for func, group_key, position in normalize_pattern(pattern)
    ]

    assert normalized == [
        ("first", "default", 0),
        ("second", "default", 1),
    ]


def test_artifact_graph_tracks_type_groups_and_invocation_ownership():
    nuclei_spec = ArtifactSpec.output("nuclei", ObjectLabelsArtifactType)

    @artifact_outputs(nuclei_spec)
    def identify(image):
        return image

    graph = extract_artifact_declarations({"DAPI": identify})

    assert graph.outputs[nuclei_spec.ref()] == nuclei_spec
    assert graph.output_groups[nuclei_spec.ref()] == {"DAPI"}
    assert graph.producers[0].invocation_keys == (
        FunctionInvocationKey("identify", "DAPI", 0),
    )


def test_artifact_graph_accepts_invocation_aware_declaration_provider():
    def configurable(image, output_name="objects"):
        return image

    def declarations_for(invocation, step_context):
        del step_context

        @artifact_outputs(
            ArtifactSpec.output(
                invocation.kwargs_dict["output_name"],
                ObjectLabelsArtifactType,
            )
        )
        def declared_artifact_owner(image):
            return image

        return CallableContract.from_callable(declared_artifact_owner)

    graph = extract_artifact_declarations(
        [
            (configurable, {"output_name": "nuclei"}),
            (configurable, {"output_name": "cells"}),
        ],
        declaration_provider=declarations_for,
    )

    assert tuple(graph.outputs) == tuple(
        spec.ref()
        for spec in (
            ArtifactSpec.output("nuclei", ObjectLabelsArtifactType),
            ArtifactSpec.output("cells", ObjectLabelsArtifactType),
        )
    )
    assert graph.producers[0].invocation_keys == (
        FunctionInvocationKey("configurable", "default", 0),
    )
    assert graph.producers[1].invocation_keys == (
        FunctionInvocationKey("configurable", "default", 1),
    )


def test_artifact_graph_distinguishes_same_name_producer_types():
    labels_spec = ArtifactSpec.output("objects", ObjectLabelsArtifactType)
    measurements_spec = ArtifactSpec.output("objects", MeasurementsArtifactType)

    @artifact_outputs(labels_spec)
    def identify(image):
        return image

    @artifact_outputs(measurements_spec)
    def measure(image):
        return image

    graph = extract_artifact_declarations([identify, measure])

    assert tuple(graph.outputs) == (labels_spec.ref(), measurements_spec.ref())


def test_artifact_graph_preserves_local_names_across_artifact_types():
    output_labels = ArtifactSpec.output("objects", ObjectLabelsArtifactType)
    input_measurements = ArtifactSpec.input("objects", MeasurementsArtifactType)

    @artifact_outputs(output_labels)
    def identify(image):
        return image

    @artifact_inputs(input_measurements)
    def measure(image, objects):
        return image

    graph = extract_artifact_declarations([identify, measure])

    assert tuple(producer.spec.ref() for producer in graph.producers) == (
        output_labels.ref(),
    )
    assert tuple(consumer.spec.ref() for consumer in graph.consumers) == (
        input_measurements.ref(),
    )


def test_normalize_function_pattern_is_grouped_source_of_truth():
    normalized = normalize_function_pattern(
        {
            "DAPI": [first, (skipped, {"enabled": False}), (second, {"sigma": 2})],
            "GFP": third,
        }
    )

    assert normalized.is_grouped
    assert [group.group_key for group in normalized.groups] == ["DAPI", "GFP"]
    assert [
        (item.key.function_name, item.key.group_key, item.key.position, item.kwargs)
        for item in normalized.iter_items()
    ] == [
        ("first", "DAPI", 0, ()),
        ("second", "DAPI", 1, (("sigma", 2),)),
        ("third", "GFP", 0, ()),
    ]


def test_normalize_function_pattern_rejects_normalized_invocation_key_collision():
    with pytest.raises(
        ValueError,
        match=(
            r"Original group keys 1 and '1'.*"
            r"FunctionInvocationKey\(function_name='skipped', "
            r"group_key='1', position=0\)"
        ),
    ):
        normalize_function_pattern({1: skipped, "1": skipped})


def test_callable_contract_is_nominal_source_for_callable_metadata():
    @artifact_inputs(ArtifactSpec.input("positions", SpecialArtifactType))
    @artifact_outputs(ArtifactSpec.output("positions", SpecialArtifactType))
    def metadata_owner(image):
        return image

    metadata_owner.input_memory_type = "numpy"
    metadata_owner.output_memory_type = "cupy"

    contract = CallableContract.from_callable(metadata_owner)
    positions_plan = ArtifactInputPlan("positions", "/tmp/positions.pkl")
    other_plan = ArtifactInputPlan("other", "/tmp/other.pkl")

    assert contract.function_name == "metadata_owner"
    assert contract.input_memory_type == "numpy"
    assert contract.output_memory_type == "cupy"
    assert contract.artifact_inputs.names() == ("positions",)
    assert contract.artifact_outputs.names() == ("positions",)
    assert contract.select_plans(
        ArtifactInputPlan,
        {
            positions_plan.ref(): positions_plan,
            other_plan.ref(): other_plan,
        },
    ) == (positions_plan,)


def test_artifact_plan_selection_follows_declaration_order_and_omits_optional():
    @artifact_inputs(
        ArtifactSpec.input("second", ImageArtifactType),
        ArtifactSpec.input(
            "optional",
            ImageArtifactType,
            required=False,
        ),
        ArtifactSpec.input("first", ImageArtifactType),
    )
    def declared_artifact_owner(image):
        return image

    first_plan = ArtifactInputPlan(
        "first",
        "/tmp/first.tif",
        artifact_type=ImageArtifactType,
    )
    second_plan = ArtifactInputPlan(
        "second",
        "/tmp/second.tif",
        artifact_type=ImageArtifactType,
    )
    selected = CallableContract.from_callable(declared_artifact_owner).select_plans(
        ArtifactInputPlan,
        {
            first_plan.ref(): first_plan,
            second_plan.ref(): second_plan,
        },
    )

    assert selected == (second_plan, first_plan)


def test_artifact_plan_selection_rejects_present_plan_with_wrong_exact_ref():
    @artifact_inputs(ArtifactSpec.input("required", ImageArtifactType))
    def declared_artifact_owner(image):
        return image

    with pytest.raises(TypeError, match="artifact plan maps require ArtifactSpecRef"):
        CallableContract.from_callable(declared_artifact_owner).select_plans(
            ArtifactInputPlan,
            {
                "required": ArtifactInputPlan(
                    "required",
                    "/tmp/required.pkl",
                    artifact_type=SpecialArtifactType,
                )
            },
        )


def test_artifact_plan_selection_omits_input_satisfied_without_runtime_plan():
    @artifact_inputs(ArtifactSpec.input("metadata_value", SpecialArtifactType))
    def declared_artifact_owner(image, metadata_value):
        return image

    assert (
        CallableContract.from_callable(declared_artifact_owner).select_plans(
            ArtifactInputPlan,
            {},
        )
        == ()
    )


def test_compiled_invocation_uses_exact_artifact_input_edge_parameter():
    positions_spec = ArtifactSpec.input("positions", SpecialArtifactType)

    @artifact_inputs(positions_spec)
    def assemble(image, positions):
        return image, positions

    positions_plan = ArtifactInputPlan("positions", "/tmp/positions.pkl")
    compiled = compile_function_pattern(
        assemble,
        {positions_plan.ref(): positions_plan},
        {},
    )

    invocation = compiled.default_group.invocations[0]
    edge = exact_input_edge(
        invocation,
        input_index=0,
        spec=positions_spec,
        storage_plan=positions_plan,
        parameter_name="positions",
    )
    invocation = invocation.with_artifact_input_edges((edge,))

    assert invocation.kwargs == ()
    assert invocation.artifact_input_edges == (edge,)
    assert edge.key == InvocationArtifactInputProjectionKey(
        invocation_key=invocation.key,
        input_index=0,
    )
    assert edge.spec == positions_spec
    assert edge.storage_plan is positions_plan
    assert edge.spec.parameter_name == "positions"


def test_compile_function_pattern_builds_invocation_source_of_truth():
    positions_spec = ArtifactSpec.input("positions", SpecialArtifactType)

    @artifact_inputs(positions_spec)
    @artifact_outputs(ArtifactSpec.output("positions", SpecialArtifactType))
    def consume_positions(image, *, positions, sigma=None, dtype_config=None):
        del positions
        return image

    @artifact_outputs(ArtifactSpec.output("measurements", SpecialArtifactType))
    def measure(image):
        return image

    consume_positions.input_memory_type = "numpy"
    consume_positions.output_memory_type = "numpy"
    measure.input_memory_type = "numpy"
    measure.output_memory_type = "numpy"
    positions_input = ArtifactInputPlan("positions", "/tmp/positions.pkl")
    positions_output = ArtifactOutputPlan("positions", "/tmp/positions.pkl")
    measurements_output = ArtifactOutputPlan(
        "measurements",
        "/tmp/measurements.pkl",
    )

    compiled = compile_function_pattern(
        {
            "DAPI": [
                (
                    consume_positions,
                    {
                        "sigma": 1,
                        "enabled": True,
                        "dtype_config": "inherited",
                        "__pyqt_reactive_scope_token__": "ui",
                    },
                ),
                measure,
            ]
        },
        {positions_input.ref(): positions_input},
        {
            positions_output.ref(): positions_output,
            measurements_output.ref(): measurements_output,
        },
    )

    group = compiled.group_for_component("DAPI")
    assert compiled.is_grouped
    assert group is not None
    assert [invocation.key for invocation in group.invocations] == [
        FunctionInvocationKey("consume_positions", "DAPI", 0),
        FunctionInvocationKey("measure", "DAPI", 1),
    ]
    assert group.invocations[0].contract.function_name == "consume_positions"
    assert group.invocations[0].kwargs == (
        ("sigma", 1),
        ("dtype_config", "inherited"),
    )
    assert group.invocations[0].contract.artifact_inputs == ArtifactSpecCollection(
        (positions_spec,)
    )
    assert group.invocations[0].artifact_output_plans == (positions_output,)
    assert group.invocations[1].artifact_output_plans == (measurements_output,)


def test_compile_function_pattern_uses_invocation_aware_declarations():
    def configurable(image, output_name="objects"):
        return image

    def declarations_for(invocation, step_context):
        del step_context
        output_name = invocation.kwargs_dict["output_name"]

        @artifact_outputs(ArtifactSpec.output(output_name, SpecialArtifactType))
        def declared_artifact_owner(image):
            return image

        return CallableContract.from_callable(declared_artifact_owner)

    nuclei_output = ArtifactOutputPlan("nuclei", "/tmp/nuclei.pkl")
    cells_output = ArtifactOutputPlan("cells", "/tmp/cells.pkl")
    compiled = compile_function_pattern(
        [
            (configurable, {"output_name": "nuclei"}),
            (configurable, {"output_name": "cells"}),
        ],
        {},
        {
            nuclei_output.ref(): nuclei_output,
            cells_output.ref(): cells_output,
        },
        declaration_provider=declarations_for,
    )

    assert [
        invocation.artifact_output_plans
        for invocation in compiled.default_group.invocations
    ] == [
        (nuclei_output,),
        (cells_output,),
    ]


def test_callable_artifact_metadata_drives_graph_and_compiled_output_plan():
    dna_spec = ArtifactSpec.input("DNA", ImageArtifactType)
    nuclei_spec = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)

    @artifact_inputs(dna_spec)
    @artifact_outputs(nuclei_spec)
    def identify_primary_objects(image, DNA):
        del DNA
        return image

    dna_plan = ArtifactInputPlan(
        name="DNA",
        path="/memory/DNA.pkl",
        artifact_type=ImageArtifactType,
    )
    nuclei_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    graph = extract_artifact_declarations(identify_primary_objects)
    compiled = compile_function_pattern(
        identify_primary_objects,
        {dna_plan.ref(): dna_plan},
        {nuclei_plan.ref(): nuclei_plan},
    )

    invocation = compiled.groups[0].invocations[0]
    assert list(graph.inputs) == [dna_spec.ref()]
    assert list(graph.outputs) == [nuclei_spec.ref()]
    assert invocation.contract.artifact_inputs == ArtifactSpecCollection((dna_spec,))
    assert invocation.contract.artifact_outputs == ArtifactSpecCollection(
        (nuclei_spec,)
    )
    assert invocation.artifact_output_plans == (nuclei_plan,)


def test_callable_contract_relation_sources_select_group_scope_inputs():
    source = ArtifactSpec.input("OrigWorms", ImageArtifactType)
    runtime_input = ArtifactSpec.input("ShrunkenWell", ObjectLabelsArtifactType)
    output = ArtifactSpec.output(
        "MaskedOrigWorms",
        ImageArtifactType,
        relations=(GroupLineageSourceRelation(source=source.ref()),),
    )

    @artifact_inputs(source, runtime_input)
    @artifact_outputs(output)
    def mask_image(image, OrigWorms, ShrunkenWell):
        del OrigWorms, ShrunkenWell
        return image

    callable_contract = CallableContract.from_callable(mask_image)
    assert callable_contract.artifact_inputs == ArtifactSpecCollection(
        (source, runtime_input)
    )
    assert callable_contract.artifact_outputs == ArtifactSpecCollection((output,))
    assert callable_contract.group_scope_inputs == ArtifactSpecCollection((source,))


def test_compiled_group_declares_managed_artifact_input_domain():
    illum_spec = ArtifactSpec.input("illum", ImageArtifactType)

    @artifact_inputs(illum_spec)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def apply_illumination(image, *, runtime):
        return image

    illum_plan = ArtifactInputPlan(
        name="illum",
        path="/memory/illum.pkl",
        artifact_type=ImageArtifactType,
    )
    compiled = compile_function_pattern(
        apply_illumination,
        {illum_plan.ref(): illum_plan},
        {},
    )

    invocation = compiled.default_group.invocations[0]
    edge = exact_input_edge(
        invocation,
        input_index=0,
        spec=illum_spec,
        storage_plan=illum_plan,
        parameter_name=None,
    )
    invocation = invocation.with_artifact_input_edges((edge,))
    group = replace(compiled.default_group, invocations=(invocation,))

    adapter = invocation.contract.runtime_adapter
    assert adapter is not None
    assert adapter.manages_artifact_inputs
    assert edge.spec.parameter_name is None
    assert invocation.runtime_domain is RuntimeInvocationDomain.ARTIFACT_MANAGED
    assert group.runtime_domain is RuntimeInvocationDomain.ARTIFACT_MANAGED


def test_source_bound_input_edge_keeps_source_anchored_runtime_domain():
    source_spec = ArtifactSpec.input("source", ImageArtifactType)

    @artifact_inputs(source_spec)
    def consume_source(image):
        return image

    compiled = compile_function_pattern(consume_source, {}, {})
    invocation = compiled.default_group.invocations[0]
    edge = InvocationArtifactInputEdgePlan(
        key=InvocationArtifactInputProjectionKey(
            invocation_key=invocation.key,
            input_index=0,
        ),
        spec=source_spec,
        storage_plan=None,
        projection=None,
        consumes_main_flow=True,
    )
    invocation = invocation.with_artifact_input_edges((edge,))
    group = replace(compiled.default_group, invocations=(invocation,))

    assert invocation.runtime_domain is RuntimeInvocationDomain.SOURCE_ANCHORED
    assert group.runtime_domain is RuntimeInvocationDomain.SOURCE_ANCHORED


def test_special_input_without_main_flow_edge_preserves_implicit_image_flow() -> None:
    @artifact_inputs("pixel_size")
    def measure(image, pixel_size=1.0):
        del pixel_size
        return image

    compiled = compile_function_pattern(measure, {}, {})

    assert compiled.default_group.main_flow_input_refs is None


def test_unstored_positional_artifact_does_not_override_compiled_main_flow() -> None:
    source = ArtifactSpec.input("Source", ImageArtifactType)

    @artifact_inputs(source)
    def consume_source(image):
        return image

    compiled = compile_function_pattern(consume_source, {}, {})
    invocation = compiled.default_group.invocations[0]
    edge = InvocationArtifactInputEdgePlan(
        key=InvocationArtifactInputProjectionKey(
            invocation_key=invocation.key,
            input_index=0,
        ),
        spec=source,
        storage_plan=None,
        projection=None,
        consumes_main_flow=False,
    )
    group = replace(
        compiled.default_group,
        invocations=(invocation.with_artifact_input_edges((edge,)),),
    )

    assert group.main_flow_input_refs is None


def test_component_projection_uses_compiled_per_group_source_lineage() -> None:
    blue = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    green = ArtifactSpec.input("OrigGreen", ImageArtifactType)
    measurements = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
        relations=tuple(
            GroupLineageSourceRelation(source=source.ref()) for source in (blue, green)
        ),
    )

    @artifact_inputs(blue, green)
    @artifact_outputs(measurements)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_outputs=True,
    )
    def measure_channels(image, *, runtime):
        del runtime
        return image

    output_plan = ArtifactOutputPlan(
        name=measurements.name,
        path="/memory/measurements.pkl",
        artifact_type=measurements.artifact_type,
        relations=measurements.relations,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        group_scope_sources_by_group={
            "1": (blue.ref(),),
            "2": (green.ref(),),
        },
    )
    compiled = compile_function_pattern(
        measure_channels,
        {},
        {output_plan.ref(): output_plan},
    )
    invocation = compiled.default_group.invocations[0]
    invocation = invocation.with_artifact_input_edges(
        tuple(
            InvocationArtifactInputEdgePlan(
                key=edge_key,
                spec=source,
                storage_plan=None,
                projection=None,
                consumes_main_flow=True,
            )
            for edge_key, source in zip(
                InvocationArtifactInputProjectionKey.for_input_count(
                    invocation.key,
                    2,
                ),
                (blue, green),
                strict=True,
            )
        )
    )
    group = replace(compiled.default_group, invocations=(invocation,))
    execution_scope = ComponentGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )

    assert group.main_flow_input_refs_for_component(execution_scope, "1") == (
        blue.ref(),
    )
    assert group.main_flow_input_refs_for_component(execution_scope, "2") == (
        green.ref(),
    )
    green_projection = invocation.for_component_execution(execution_scope, "2")
    assert green_projection is not None
    assert tuple(
        edge.key.input_index for edge in green_projection.artifact_input_edges
    ) == (1,)


def test_unscoped_active_output_retains_complete_compiled_invocation_inputs() -> None:
    source = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    aggregate = ArtifactSpec.output("Aggregate", MeasurementsArtifactType)
    scoped = ArtifactSpec.output(
        "BlueMeasurements",
        MeasurementsArtifactType,
        relations=(GroupLineageSourceRelation(source=source.ref()),),
    )

    @artifact_inputs(source)
    @artifact_outputs(aggregate, scoped)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_outputs=True,
    )
    def measure_with_aggregate(image, *, runtime):
        del runtime
        return image

    aggregate_plan = ArtifactOutputPlan(
        name=aggregate.name,
        path="/memory/aggregate.pkl",
        artifact_type=aggregate.artifact_type,
    )
    scoped_plan = ArtifactOutputPlan(
        name=scoped.name,
        path="/memory/blue.pkl",
        artifact_type=scoped.artifact_type,
        relations=scoped.relations,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        group_scope_sources_by_group={"1": (source.ref(),)},
    )
    invocation = compile_function_pattern(
        measure_with_aggregate,
        {},
        {
            aggregate_plan.ref(): aggregate_plan,
            scoped_plan.ref(): scoped_plan,
        },
    ).default_group.invocations[0]
    (edge_key,) = InvocationArtifactInputProjectionKey.for_input_count(
        invocation.key,
        1,
    )
    edge = InvocationArtifactInputEdgePlan(
        key=edge_key,
        spec=source,
        storage_plan=None,
        projection=None,
        consumes_main_flow=True,
    )
    invocation = invocation.with_artifact_input_edges((edge,))

    projection = invocation.for_component_execution(
        ComponentGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
        "2",
    )

    assert projection is not None
    assert tuple(plan.name for plan in projection.artifact_output_plans) == (
        "Aggregate",
    )
    assert projection.artifact_input_edges == (edge,)


def test_shared_output_plan_uses_each_invocation_declared_group_lineage() -> None:
    blue = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    green = ArtifactSpec.input("OrigGreen", ImageArtifactType)
    blue_measurements = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
        relations=(GroupLineageSourceRelation(source=blue.ref()),),
    )
    green_measurements = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
        relations=(GroupLineageSourceRelation(source=green.ref()),),
    )

    @artifact_inputs(blue)
    @artifact_outputs(blue_measurements)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_outputs=True,
    )
    def measure_blue(image, *, runtime):
        del runtime
        return image

    @artifact_inputs(green)
    @artifact_outputs(green_measurements)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_outputs=True,
    )
    def measure_green(image, *, runtime):
        del runtime
        return image

    output_plan = ArtifactOutputPlan(
        name=blue_measurements.name,
        path="/memory/measurements.pkl",
        artifact_type=blue_measurements.artifact_type,
        relations=(*blue_measurements.relations, *green_measurements.relations),
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        group_scope_sources_by_group={
            "1": (blue.ref(),),
            "2": (green.ref(),),
        },
    )
    blue_invocation, green_invocation = compile_function_pattern(
        [measure_blue, measure_green],
        {},
        {output_plan.ref(): output_plan},
    ).default_group.invocations
    execution_scope = ComponentGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )

    assert blue_invocation.for_component_execution(execution_scope, "1") is not None
    assert blue_invocation.for_component_execution(execution_scope, "2") is None
    assert green_invocation.for_component_execution(execution_scope, "1") is None
    assert green_invocation.for_component_execution(execution_scope, "2") is not None


def test_artifact_only_group_preserves_empty_explicit_main_flow_refs() -> None:
    payload = ArtifactSpec.input(
        "payload",
        SpecialArtifactType,
        parameter_name="payload",
    )

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @artifact_inputs(payload)
    def consume(*, payload):
        return payload

    compiled = compile_function_pattern(consume, {}, {})

    assert compiled.default_group.main_flow_input_refs == ()


def test_special_input_edges_use_nominal_artifact_payload_types() -> None:
    measurements = ArtifactSpec.input("Measurements", MeasurementsArtifactType)
    labels = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    mask = ArtifactSpec.input("Mask", ImageArtifactType, parameter_name="mask")

    @special_inputs("labels", "mask")
    def consume(
        image: np.ndarray,
        labels: ObjectLabelValue,
        mask: np.ndarray,
    ) -> np.ndarray:
        del labels, mask
        return image

    validate_artifact_input_parameter_bindings(
        consume,
        (measurements, labels, mask),
        adapter_manages_inputs=True,
    )
    assert tuple(spec.parameter_name for spec in (measurements, labels, mask)) == (
        None,
        "labels",
        "mask",
    )


def test_sequence_special_input_claims_all_compatible_artifacts() -> None:
    measurements = ArtifactSpec.input("Measurements", MeasurementsArtifactType)
    labels = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="topology_inputs",
    )
    mask = ArtifactSpec.input(
        "Mask",
        ImageArtifactType,
        parameter_name="topology_inputs",
    )

    @special_inputs("topology_inputs")
    def consume(
        image: np.ndarray,
        topology_inputs: tuple[np.ndarray | ObjectLabelValue, ...],
    ) -> np.ndarray:
        del topology_inputs
        return image

    validate_artifact_input_parameter_bindings(
        consume,
        (measurements, labels, mask),
        adapter_manages_inputs=True,
    )
    assert tuple(spec.parameter_name for spec in (measurements, labels, mask)) == (
        None,
        "topology_inputs",
        "topology_inputs",
    )


def test_adapter_managed_invocation_rejects_partial_component_inputs():
    first_spec = ArtifactSpec.input("derived_one", ImageArtifactType)
    second_spec = ArtifactSpec.input("derived_two", ImageArtifactType)

    @artifact_inputs(first_spec, second_spec)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def consume_mixed_inputs(image, *, runtime):
        return image

    first_plan = ArtifactInputPlan(
        name="derived_one",
        path="/memory/derived_one.pkl",
        artifact_type=ImageArtifactType,
    )
    second_plan = ArtifactInputPlan(
        name="derived_two",
        path="/memory/derived_two.pkl",
        artifact_type=ImageArtifactType,
    )
    declared_inputs = {
        first_plan.ref(): first_plan,
        second_plan.ref(): second_plan,
    }
    compiled = compile_function_pattern(
        consume_mixed_inputs,
        declared_inputs,
        {},
    )
    invocation = compiled.default_group.invocations[0]
    edges = (
        exact_input_edge(
            invocation,
            input_index=0,
            spec=first_spec,
            storage_plan=first_plan,
            parameter_name=None,
        ),
        exact_input_edge(
            invocation,
            input_index=1,
            spec=second_spec,
            storage_plan=second_plan,
            parameter_name=None,
        ),
    )
    invocation = invocation.with_artifact_input_edges(edges)

    assert invocation.runtime_domain is RuntimeInvocationDomain.ARTIFACT_MANAGED
    assert tuple(edge.key.input_index for edge in invocation.artifact_input_edges) == (
        0,
        1,
    )
    assert tuple(edge.spec for edge in invocation.artifact_input_edges) == (
        first_spec,
        second_spec,
    )
    with pytest.raises(ValueError, match="input plan.*unavailable"):
        ComponentArtifactPlans(
            inputs={first_plan.ref(): first_plan},
            outputs={},
        ).select_for_invocation(
            invocation,
            execution_scope=ComponentGroupScope.ungrouped(),
            component_key=None,
        )


def test_runtime_artifact_group_lineage_owns_mixed_input_invocation_domain():
    source = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    runtime_image = ArtifactSpec.input("RGBImage", ImageArtifactType)
    saved = ArtifactSpec.output(
        "SavedRGBImage",
        ImageArtifactType,
        relations=(GroupLineageSourceRelation(source=runtime_image.ref()),),
    )

    @artifact_inputs(source, runtime_image)
    @artifact_outputs(saved)
    def save_image(image, *, RGBImage):
        del RGBImage
        return image

    runtime_plan = ArtifactInputPlan(
        name=runtime_image.name,
        path="/memory/RGBImage.pkl",
        artifact_type=runtime_image.artifact_type,
    )
    saved_plan = ArtifactOutputPlan(
        name=saved.name,
        path="/memory/SavedRGBImage.pkl",
        artifact_type=saved.artifact_type,
        relations=saved.relations,
    )
    compiled = compile_function_pattern(
        save_image,
        {runtime_plan.ref(): runtime_plan},
        {saved_plan.ref(): saved_plan},
    )

    invocation = compiled.default_group.invocations[0]
    edge = exact_input_edge(
        invocation,
        input_index=0,
        spec=runtime_image,
        storage_plan=runtime_plan,
        parameter_name="RGBImage",
    )
    invocation = invocation.with_artifact_input_edges((edge,))

    assert invocation.contract.group_scope_inputs == ArtifactSpecCollection(
        (runtime_image,)
    )
    assert invocation.artifact_input_edges == (edge,)
    assert invocation.artifact_output_plans == (saved_plan,)
    assert invocation.runtime_domain is RuntimeInvocationDomain.ARTIFACT_MANAGED


def test_adapter_managed_invocation_rejects_cross_component_input_loss():
    @artifact_inputs(
        ArtifactSpec.input("cross_channel", ImageArtifactType),
        ArtifactSpec.input("current_channel", ImageArtifactType),
    )
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def consume_cross_channel(image, *, runtime):
        return image

    cross_channel_plan = ArtifactInputPlan(
        name="cross_channel",
        path="/memory/cross_channel.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": "/memory/cross_channel__3.pkl"},
    )
    current_channel_plan = ArtifactInputPlan(
        name="current_channel",
        path="/memory/current_channel.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/current_channel__1.pkl"},
    )
    compiled = compile_function_pattern(
        consume_cross_channel,
        {
            cross_channel_plan.ref(): cross_channel_plan,
            current_channel_plan.ref(): current_channel_plan,
        },
        {},
    )

    invocation = compiled.default_group.invocations[0]
    invocation = invocation.with_artifact_input_edges(
        (
            exact_input_edge(
                invocation,
                input_index=0,
                spec=invocation.contract.artifact_inputs[0],
                storage_plan=cross_channel_plan,
                parameter_name=None,
            ),
            exact_input_edge(
                invocation,
                input_index=1,
                spec=invocation.contract.artifact_inputs[1],
                storage_plan=current_channel_plan,
                parameter_name=None,
            ),
        )
    )
    scoped_artifacts = ComponentArtifactPlans(
        inputs={current_channel_plan.ref(): current_channel_plan.for_group("1")},
        outputs={},
    )

    with pytest.raises(ValueError, match="input plan.*unavailable"):
        scoped_artifacts.select_for_invocation(
            invocation,
            execution_scope=ComponentGroupScope.ungrouped(),
            component_key=None,
        )


def test_adapter_managed_outputs_use_exact_compiled_output_plans():
    cells_spec = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    measurements_spec = ArtifactSpec.output("Measurements", MeasurementsArtifactType)

    @artifact_outputs(cells_spec, measurements_spec)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_outputs=True,
    )
    def runtime_recorded_outputs(image, *, runtime):
        return image

    cells_output = ArtifactOutputPlan(
        name="Cells",
        path="/memory/Cells.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": "/memory/w3_Cells.pkl"},
    )
    measurement_output = ArtifactOutputPlan(
        name="Measurements",
        path="/memory/Measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/w1_Measurements.pkl"},
    )
    compiled = compile_function_pattern(
        runtime_recorded_outputs,
        {},
        {
            cells_output.ref(): cells_output,
            measurement_output.ref(): measurement_output,
        },
    )

    invocation = compiled.default_group.invocations[0]
    selected_outputs = invocation.select_outputs(
        {cells_output.ref(): cells_output, measurement_output.ref(): measurement_output}
    )

    adapter = invocation.contract.runtime_adapter
    assert adapter is not None
    assert adapter.manages_artifact_outputs
    assert invocation.artifact_output_plans == (cells_output, measurement_output)
    assert selected_outputs == {
        cells_output.ref(): cells_output,
        measurement_output.ref(): measurement_output,
    }

    with pytest.raises(ValueError, match="output plan.*unavailable"):
        invocation.select_outputs({measurement_output.ref(): measurement_output})


def test_metadata_satisfied_input_without_compiled_edge_is_source_anchored():
    @artifact_inputs(ArtifactSpec.input("illum", ImageArtifactType))
    def apply_illumination(image, illum=None):
        return image

    compiled = compile_function_pattern(apply_illumination, {}, {})

    group = compiled.default_group
    assert group.invocations[0].contract.artifact_inputs.names() == ("illum",)
    assert group.invocations[0].artifact_input_edges == ()
    assert (
        group.invocations[0].runtime_domain is RuntimeInvocationDomain.SOURCE_ANCHORED
    )
    assert group.runtime_domain is RuntimeInvocationDomain.SOURCE_ANCHORED


def test_function_pattern_tuple_leaf_requires_exactly_two_members():
    with pytest.raises(TypeError, match="exactly two"):
        compile_function_pattern((first, {"sigma": 1}, object()), {}, {})


def test_compile_function_pattern_preserves_runtime_context_and_adapter_declarations():
    @runtime_adapter("runtime", lambda _request: object())
    def configurable(
        image,
        *,
        sigma=1,
        context=None,
        runtime=None,
    ):
        return image

    compiled = compile_function_pattern(
        (configurable, {"sigma": 1}),
        {},
        {},
    )

    invocation = compiled.default_group.invocations[0]
    adapter = invocation.contract.runtime_adapter
    assert invocation.contract.runtime_context_parameter == (
        ProcessingContext.require_parameter_name()
    )
    assert adapter is not None
    assert adapter.require_parameter_name() == "runtime"


def test_adapter_free_inputs_bind_distinct_exact_artifact_parameters() -> None:
    first_spec = ArtifactSpec.input("first_image", ImageArtifactType)
    second_spec = ArtifactSpec.input("second_image", ImageArtifactType)

    @artifact_inputs(first_spec, second_spec)
    def save(image, *, first_image, second_image):
        return image, first_image, second_image

    first_plan = ArtifactInputPlan(
        "first_image",
        "/tmp/first-image.pkl",
        artifact_type=ImageArtifactType,
    )
    second_plan = ArtifactInputPlan(
        "second_image",
        "/tmp/second-image.pkl",
        artifact_type=ImageArtifactType,
    )

    compiled = compile_function_pattern(
        save,
        {
            first_plan.ref(): first_plan,
            second_plan.ref(): second_plan,
        },
        {},
    )

    invocation = compiled.default_group.invocations[0]
    invocation = invocation.with_artifact_input_edges(
        (
            exact_input_edge(
                invocation,
                input_index=0,
                spec=first_spec,
                storage_plan=first_plan,
                parameter_name="first_image",
            ),
            exact_input_edge(
                invocation,
                input_index=1,
                spec=second_spec,
                storage_plan=second_plan,
                parameter_name="second_image",
            ),
        )
    )

    assert tuple(edge.key.input_index for edge in invocation.artifact_input_edges) == (
        0,
        1,
    )
    assert tuple(edge.spec for edge in invocation.artifact_input_edges) == (
        first_spec,
        second_spec,
    )
    assert tuple(
        edge.spec.parameter_name for edge in invocation.artifact_input_edges
    ) == (
        "first_image",
        "second_image",
    )


def test_compile_function_pattern_excludes_nominally_owned_parameters():
    @artifact_inputs(ArtifactSpec.input("labels", ObjectLabelsArtifactType))
    @runtime_bound_parameters(DtypeConversionConfig)
    @runtime_adapter("runtime", lambda _request: object())
    def managed(
        image,
        labels,
        *,
        context,
        runtime,
        dtype_config,
        sigma=1,
    ):
        return image

    labels_plan = ArtifactInputPlan(
        "labels",
        "/tmp/labels.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    compiled = compile_function_pattern(
        managed,
        {labels_plan.ref(): labels_plan},
        {},
        runtime_parameter_bindings=(
            RuntimeParameterBinding(
                parameter_name=DtypeConversionConfig.require_parameter_name(),
                value=DtypeConfig(),
            ),
        ),
    )

    invocation = compiled.default_group.invocations[0]
    assert invocation.kwargs == ()
    assert tuple(
        binding.parameter_name for binding in invocation.runtime_parameter_bindings
    ) == ("dtype_config",)
    assert invocation.contract.runtime_context_parameter == "context"
    assert invocation.contract.runtime_adapter is not None
    assert invocation.contract.runtime_adapter.require_parameter_name() == "runtime"


def test_runtime_adapter_declaration_requires_signature_parameter():
    with pytest.raises(TypeError, match="runtime"):

        @runtime_adapter("runtime", lambda _request: object())
        def missing_runtime_parameter(image):
            return image


def test_callable_contract_reads_runtime_adapter_from_metadata_namespace():
    @runtime_adapter("runtime", lambda _request: object())
    def source(image, *, runtime):
        return image

    def copied(image, *, runtime):
        return image

    adapter_spec = vars(source)[FunctionContractAttribute.runtime_adapter]
    vars(copied)[FunctionContractAttribute.runtime_adapter] = adapter_spec

    contract = CallableContract.from_callable(copied)

    assert contract.runtime_adapter is adapter_spec
    assert contract.runtime_adapter is not None
    assert contract.runtime_adapter.require_parameter_name() == "runtime"


def test_inject_kwargs_rejects_three_member_tuple_leaf():
    with pytest.raises(TypeError, match="exactly two"):
        inject_kwargs_into_pattern(
            (first, {"sigma": 1}, object()),
            {"dtype_config": "inherited"},
        )


def test_compiled_function_pattern_filters_detected_groups_by_compiled_keys():
    compiled = compile_function_pattern({"1": skipped}, {}, {})

    grouped = compiled.prepare_grouped_patterns(
        {1: ["site1"], "2": ["site2"]},
        default_component="channel",
    )

    assert grouped == {1: ["site1"]}


def test_artifact_managed_runtime_domain_uses_one_lifecycle_anchor():
    anchors = ["first", "second"]

    assert RuntimeInvocationDomain.ARTIFACT_MANAGED.select_lifecycle_anchors(
        anchors
    ) == ["first"]
    assert (
        RuntimeInvocationDomain.SOURCE_ANCHORED.select_lifecycle_anchors(anchors)
        is anchors
    )


def test_ungrouped_compiled_function_pattern_preserves_grouped_source_patterns():
    compiled = compile_function_pattern(skipped, {}, {})
    grouped_patterns = {
        "1": ["A01_s001_w1_z001_t001.tif"],
        "2": ["A01_s002_w1_z001_t001.tif"],
    }

    grouped = compiled.prepare_grouped_patterns(
        grouped_patterns,
        default_component="site",
    )

    assert grouped == grouped_patterns


def test_contract_decorators_declare_artifact_specs():
    materialization = csv_only()

    @artifact_inputs("positions")
    @artifact_outputs("metadata", ("measurements", materialization))
    def analyze(image):
        return image

    contract = CallableContract.from_callable(analyze)
    assert contract.artifact_inputs.names() == ("positions",)
    assert contract.artifact_outputs.names() == (
        "metadata",
        "measurements",
    )
    assert contract.artifact_outputs[0] == ArtifactSpec.output(
        "metadata",
        SpecialArtifactType,
    )
    assert contract.artifact_outputs[1].materialization is materialization


def test_artifact_decorators_bind_role_neutral_typed_specs():
    labels = ArtifactSpec("nuclei", ObjectLabelsArtifactType)
    measurements = ArtifactSpec("measurements", MeasurementsArtifactType)

    @artifact_inputs(labels)
    @artifact_outputs(measurements)
    def measure(image, nuclei):
        return image

    contract = CallableContract.from_callable(measure)
    assert labels.plan_type is None
    assert measurements.plan_type is None
    assert contract.artifact_inputs == ArtifactSpecCollection(
        (labels.for_plan_type(ArtifactInputPlan),)
    )
    assert contract.artifact_outputs == ArtifactSpecCollection(
        (measurements.for_plan_type(ArtifactOutputPlan),)
    )


def test_role_neutral_artifact_spec_requires_declaration_before_ref():
    labels = ArtifactSpec("nuclei", ObjectLabelsArtifactType)

    with pytest.raises(ValueError, match="no plan role"):
        labels.ref()


def test_artifact_spec_rejects_unregistered_plan_type():
    class UnregisteredPlan(ArtifactPlan):
        pass

    with pytest.raises(ValueError, match="registered ArtifactPlan"):
        ArtifactSpec(
            name="positions",
            plan_type=UnregisteredPlan,
            artifact_type=SpecialArtifactType,
        )


def test_artifact_spec_rejects_unregistered_artifact_type():
    class UnregisteredArtifactType(ArtifactType):
        pass

    with pytest.raises(ValueError, match="registered ArtifactType"):
        ArtifactSpec(
            name="positions",
            plan_type=ArtifactInputPlan,
            artifact_type=UnregisteredArtifactType,
        )


def test_artifact_spec_rejects_relation_target_role_mismatch():
    source = ArtifactSpec.input("objects", ObjectLabelsArtifactType)

    with pytest.raises(ValueError, match="requires target plan role output"):
        ArtifactSpec.input(
            "filtered",
            ObjectLabelsArtifactType,
            relations=(GroupLineageSourceRelation(source=source.ref()),),
        )


def test_artifact_spec_relation_sources_are_full_refs():
    source_ref = ArtifactSpec.input("objects", ObjectLabelsArtifactType).ref()
    relation = GroupLineageSourceRelation(source=source_ref)

    assert relation.source == source_ref
    assert relation.source.plan_type is ArtifactInputPlan


def test_artifact_spec_collection_rejects_unknown_relation_sources():
    output = ArtifactSpec.output(
        "filtered",
        ObjectLabelsArtifactType,
        relations=(
            GroupLineageSourceRelation(
                source=ArtifactSpec.input("missing", ObjectLabelsArtifactType).ref(),
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown artifact specs"):
        ArtifactSpecCollection((output,)).validate_registered_relation_refs(
            owner_name="test",
        )


def test_artifact_spec_collection_accepts_endpoint_relations_on_relationship_output():
    parent = ArtifactSpec.input("parents", ObjectLabelsArtifactType)
    child = ArtifactSpec.input("children", ObjectLabelsArtifactType)
    relationships = ArtifactSpec.output(
        "relationships",
        RelationshipsArtifactType,
        relations=(
            ObjectRelationshipDeclaration(
                source=parent.ref(),
                target=child.ref(),
                producer_module_number=1,
                relationship_type="parent_child",
                source_role="parent",
                target_role="child",
                source_id_field="parent_id",
                target_id_field="child_id",
                source_runtime_slice_offset=0,
                target_runtime_slice_offset=0,
            ),
        ),
    )

    ArtifactSpecCollection(
        (parent, child, relationships)
    ).validate_registered_relation_refs(
        owner_name="test",
    )


def test_artifact_spec_collection_unique_keys_by_full_ref():
    source_input = ArtifactSpec.input("objects", ObjectLabelsArtifactType)
    source_output = ArtifactSpec.output("objects", ObjectLabelsArtifactType)
    materialized_output = ArtifactSpec.output(
        "objects",
        ObjectLabelsArtifactType,
        materialization=csv_only(),
    )

    assert ArtifactSpecCollection((source_input, source_output)).unique() == (
        source_input,
        source_output,
    )
    with pytest.raises(ValueError, match="Conflicting artifact spec"):
        ArtifactSpecCollection((source_output, materialized_output)).unique()


def test_strip_disabled_functions_removes_empty_pattern_branches():
    pattern = {
        "DAPI": [first, (skipped, {"enabled": False})],
        "GFP": [(skipped, {"enabled": False})],
    }

    assert strip_disabled_functions(pattern) == {"DAPI": [first]}


def test_inject_kwargs_into_pattern_preserves_user_kwargs_precedence():
    pattern = [first, (second, {"dtype_config": "explicit", "sigma": 2})]

    assert inject_kwargs_into_pattern(pattern, {"dtype_config": "inherited"}) == [
        (first, {"dtype_config": "inherited"}),
        (second, {"dtype_config": "explicit", "sigma": 2}),
    ]


def test_compile_function_pattern_moves_runtime_config_kwargs_to_bindings():
    @runtime_bound_parameters(DtypeConversionConfig)
    def accepts_dtype_config(image, *, dtype_config=None, sigma=None):
        return image

    inherited_config = DtypeConfig(
        default_dtype_conversion=DtypeConversion.NATIVE_OUTPUT,
    )
    explicit_config = DtypeConfig(
        default_dtype_conversion=DtypeConversion.UINT8,
    )

    compiled = compile_function_pattern(
        (accepts_dtype_config, {"dtype_config": explicit_config, "sigma": 2}),
        {},
        {},
        runtime_parameter_bindings=(
            RuntimeParameterBinding(
                parameter_name=DtypeConversionConfig.require_parameter_name(),
                value=inherited_config,
            ),
        ),
    )
    invocation = compiled.default_group.invocations[0]

    assert invocation.kwargs_dict == {"sigma": 2}
    assert len(invocation.runtime_parameter_bindings) == 1
    binding = invocation.runtime_parameter_bindings[0]
    assert binding.parameter_name == DtypeConversionConfig.require_parameter_name()
    assert binding.value is explicit_config


def test_compile_function_pattern_keeps_undeclared_runtime_config_kwargs_user_owned():
    def accepts_user_dtype_config(image, *, dtype_config=None, sigma=None):
        return image

    inherited_config = DtypeConfig(
        default_dtype_conversion=DtypeConversion.NATIVE_OUTPUT,
    )
    explicit_config = DtypeConfig(
        default_dtype_conversion=DtypeConversion.UINT8,
    )

    compiled = compile_function_pattern(
        (
            accepts_user_dtype_config,
            {"dtype_config": explicit_config, "sigma": 2},
        ),
        {},
        {},
        runtime_parameter_bindings=(
            RuntimeParameterBinding(
                parameter_name=DtypeConversionConfig.require_parameter_name(),
                value=inherited_config,
            ),
        ),
    )
    invocation = compiled.default_group.invocations[0]

    assert invocation.kwargs_dict == {
        "dtype_config": explicit_config,
        "sigma": 2,
    }
    assert invocation.runtime_parameter_bindings == ()


def test_inject_artifact_input_values_only_targets_declared_inputs():
    @artifact_inputs("grid_dimensions")
    def needs_grid(image, grid_dimensions):
        return image

    pattern = [first, (needs_grid, {"sigma": 2})]

    assert inject_artifact_input_values(
        pattern,
        {"grid_dimensions": (3, 4), "unused": "ignored"},
    ) == [
        first,
        (needs_grid, {"grid_dimensions": (3, 4), "sigma": 2}),
    ]


def test_inject_artifact_input_values_replaces_serialized_placeholders():
    @artifact_inputs("grid_dimensions")
    def needs_grid(image, grid_dimensions):
        return image

    pattern = (needs_grid, {"grid_dimensions": None, "sigma": 2})

    assert inject_artifact_input_values(
        pattern,
        {"grid_dimensions": (3, 4)},
    ) == (needs_grid, {"grid_dimensions": (3, 4), "sigma": 2})
