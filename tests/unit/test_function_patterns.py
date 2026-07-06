from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from arraybridge.decorators import DtypeConversionConfig

from openhcs.constants import DtypeConversion
from openhcs.core.config import DtypeConfig
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    ArtifactType,
    GroupLineageSourceRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.core.function_patterns import (
    COMPILE_TIME_FUNCTION_KWARGS_KEY,
    CompileTimeFunctionKwargs,
    CompileTimeFunctionKwarg,
    FunctionInvocationKey,
    RuntimeInvocationDomain,
    compile_function_pattern,
    inject_artifact_input_values,
    inject_kwargs_into_pattern,
    iter_enabled_function_invocations,
    normalize_function_pattern,
    strip_disabled_functions,
)
from openhcs.core.invocation_artifacts import InvocationArtifactDeclarations
from openhcs.core.module_artifact_contract import (
    ModuleArtifactContract,
    module_artifact_contract,
)
from openhcs.core.module_artifact_contract import (
    DeclaredArtifactOutputPartition,
    RecordedArtifactOutputPartition,
    RuntimeArtifactInputPartition,
    SourceArtifactInputPartition,
)
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    artifact_outputs,
    runtime_bound_parameters,
)
from openhcs.core.pipeline.artifact_planning import (
    extract_artifact_declarations,
    normalize_pattern,
)
from openhcs.core.runtime_invocation import (
    RuntimeInvocationOptions,
    RuntimeParameterBinding,
)
from openhcs.core.runtime_adapters import runtime_adapter
from openhcs.core.steps.function_runtime import ComponentArtifactPlans
from openhcs.processing.materialization import csv_only


def first(image):
    return image


def second(image):
    return image


def skipped(image):
    return image


def third(image):
    return image


@dataclass(frozen=True, slots=True, kw_only=True)
class ExampleInvocationOptions(RuntimeInvocationOptions):
    mode: str


first.__artifact_outputs__ = {
    "positions": ArtifactSpec.output("positions", SpecialArtifactType)
}
second.__artifact_outputs__ = {
    "measurements": ArtifactSpec.output("measurements", SpecialArtifactType)
}
third.__artifact_outputs__ = {
    "positions": ArtifactSpec.output("positions", SpecialArtifactType),
    "measurements": ArtifactSpec.output("measurements", SpecialArtifactType),
}


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
    @artifact_outputs(ArtifactSpec.output("nuclei", ObjectLabelsArtifactType))
    def identify(image):
        return image

    graph = extract_artifact_declarations({"DAPI": identify})

    assert graph.outputs["nuclei"].artifact_type is ObjectLabelsArtifactType
    assert graph.output_groups["nuclei"] == {"DAPI"}
    assert graph.producers[0].invocation_keys == (
        FunctionInvocationKey("identify", "DAPI", 0),
    )


def test_artifact_graph_accepts_invocation_aware_declaration_provider():
    def configurable(image, output_name="objects"):
        return image

    def declarations_for(invocation, step_context):
        del step_context
        return InvocationArtifactDeclarations(
            artifacts=(
                ArtifactSpec.output(
                    invocation.kwargs_dict["output_name"],
                    ObjectLabelsArtifactType,
                ),
            ),
        )

    graph = extract_artifact_declarations(
        [
            (configurable, {"output_name": "nuclei"}),
            (configurable, {"output_name": "cells"}),
        ],
        declaration_provider=declarations_for,
    )

    assert tuple(graph.outputs) == ("nuclei", "cells")
    assert graph.producers[0].invocation_keys == (
        FunctionInvocationKey("configurable", "default", 0),
    )
    assert graph.producers[1].invocation_keys == (
        FunctionInvocationKey("configurable", "default", 1),
    )


def test_artifact_graph_rejects_conflicting_producer_types():
    @artifact_outputs(ArtifactSpec.output("objects", ObjectLabelsArtifactType))
    def identify(image):
        return image

    @artifact_outputs(ArtifactSpec.output("objects", MeasurementsArtifactType))
    def measure(image):
        return image

    with pytest.raises(ValueError, match="Conflicting producer artifact type"):
        extract_artifact_declarations([identify, measure])


def test_artifact_graph_rejects_local_consumer_producer_kind_mismatch():
    @artifact_outputs(ArtifactSpec.output("objects", ObjectLabelsArtifactType))
    def identify(image):
        return image

    @artifact_inputs(ArtifactSpec.input("objects", MeasurementsArtifactType))
    def measure(image, objects):
        return image

    with pytest.raises(ValueError, match="produced as object_labels"):
        extract_artifact_declarations([identify, measure])


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


def test_callable_contract_is_nominal_source_for_callable_metadata():
    first.input_memory_type = "numpy"
    first.output_memory_type = "cupy"
    first.__artifact_inputs__ = {
        "positions": ArtifactSpec.input("positions", SpecialArtifactType)
    }

    contract = CallableContract.from_callable(first)

    assert contract.function_name == "first"
    assert contract.input_memory_type == "numpy"
    assert contract.output_memory_type == "cupy"
    assert contract.artifact_input_names == ("positions",)
    assert contract.artifact_output_names == ("positions",)
    assert contract.select_plan_keys(
        ArtifactInputPlan,
        {
            "positions": ArtifactInputPlan("positions", "/tmp/positions.pkl"),
            "other": ArtifactInputPlan("other", "/tmp/other.pkl"),
        },
    ) == ("positions",)


def test_compile_function_pattern_builds_invocation_source_of_truth():
    first.input_memory_type = "numpy"
    first.output_memory_type = "numpy"
    second.input_memory_type = "numpy"
    second.output_memory_type = "numpy"
    first.__artifact_inputs__ = {
        "positions": ArtifactSpec.input("positions", SpecialArtifactType)
    }

    compiled = compile_function_pattern(
        {
            "DAPI": [
                (
                    first,
                    {
                        "sigma": 1,
                        "enabled": True,
                        "dtype_config": "inherited",
                        "__pyqt_reactive_scope_token__": "ui",
                    },
                ),
                second,
            ]
        },
        {"positions": ArtifactInputPlan("positions", "/tmp/positions.pkl")},
        {
            "positions": ArtifactOutputPlan("positions", "/tmp/positions.pkl"),
            "measurements": ArtifactOutputPlan("measurements", "/tmp/measurements.pkl"),
        },
    )

    group = compiled.group_for_component("DAPI")
    assert compiled.is_grouped
    assert group is not None
    assert [invocation.key for invocation in group.invocations] == [
        FunctionInvocationKey("first", "DAPI", 0),
        FunctionInvocationKey("second", "DAPI", 1),
    ]
    assert group.invocations[0].contract.function_name == "first"
    assert group.invocations[0].kwargs == (
        ("sigma", 1),
        ("dtype_config", "inherited"),
    )
    assert group.invocations[0].artifact_input_keys == ("positions",)
    assert group.invocations[0].artifact_output_keys == ("positions",)
    assert group.invocations[1].artifact_output_keys == ("measurements",)


def test_compile_function_pattern_uses_invocation_aware_declarations():
    def configurable(image, output_name="objects"):
        return image

    def declarations_for(invocation, step_context):
        del step_context
        output_name = invocation.kwargs_dict["output_name"]
        return InvocationArtifactDeclarations(
            artifacts=(ArtifactSpec.output(output_name, SpecialArtifactType),)
        )

    compiled = compile_function_pattern(
        [
            (configurable, {"output_name": "nuclei"}),
            (configurable, {"output_name": "cells"}),
        ],
        {},
        {
            "nuclei": ArtifactOutputPlan("nuclei", "/tmp/nuclei.pkl"),
            "cells": ArtifactOutputPlan("cells", "/tmp/cells.pkl"),
        },
        declaration_provider=declarations_for,
    )

    assert [
        invocation.artifact_output_keys
        for invocation in compiled.default_group.invocations
    ] == [
        ("nuclei",),
        ("cells",),
    ]


def test_module_artifact_contract_drives_default_artifact_declarations():
    contract = ModuleArtifactContract(
        module_name="IdentifyPrimaryObjects",
        items=(
            *ModuleArtifactContract.items_for_partition(
                RuntimeArtifactInputPartition,
                (ArtifactSpec.input("DNA", ImageArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition,
                (ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition,
                (ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),),
            ),
        ),
    )

    @module_artifact_contract(contract)
    def identify_primary_objects(image):
        return image

    graph = extract_artifact_declarations(identify_primary_objects)
    compiled = compile_function_pattern(
        identify_primary_objects,
        {
            "DNA": ArtifactInputPlan(
                name="DNA",
                path="/memory/DNA.pkl",
                artifact_type=ImageArtifactType,
            ),
        },
        {
            "Nuclei": ArtifactOutputPlan(
                name="Nuclei",
                path="/memory/Nuclei.pkl",
                artifact_type=ObjectLabelsArtifactType,
            ),
        },
    )

    invocation = compiled.groups[0].invocations[0]
    assert list(graph.inputs) == ["DNA"]
    assert list(graph.outputs) == ["Nuclei"]
    assert invocation.artifact_input_keys == ("DNA",)
    assert invocation.artifact_output_keys == ("Nuclei",)


def test_module_artifact_contract_relation_sources_include_source_inputs():
    source = ArtifactSpec.input("OrigWorms", ImageArtifactType)
    runtime_input = ArtifactSpec.input("ShrunkenWell", ObjectLabelsArtifactType)
    output = ArtifactSpec.output(
        "MaskedOrigWorms",
        ImageArtifactType,
        relations=(GroupLineageSourceRelation(source=source.ref()),),
    )
    contract = ModuleArtifactContract(
        module_name="MaskImage",
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition,
                (source,),
            ),
            *ModuleArtifactContract.items_for_partition(
                RuntimeArtifactInputPartition,
                (runtime_input,),
            ),
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition,
                (output,),
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition,
                (output,),
            ),
        ),
    )

    declarations = InvocationArtifactDeclarations.from_module_contract(contract)

    assert [name for name, _spec in declarations.inputs] == [
        "OrigWorms",
        "ShrunkenWell",
    ]
    assert [name for name, _spec in declarations.outputs] == ["MaskedOrigWorms"]
    assert declarations.select_plan_keys(
        ArtifactInputPlan,
        {
            "OrigWorms": ArtifactInputPlan(
                name="OrigWorms",
                path="/memory/OrigWorms.pkl",
                artifact_type=ImageArtifactType,
            ),
            "ShrunkenWell": ArtifactInputPlan(
                name="ShrunkenWell",
                path="/memory/ShrunkenWell.pkl",
                artifact_type=ObjectLabelsArtifactType,
            ),
        },
    ) == ("ShrunkenWell",)


def test_compiled_group_declares_managed_artifact_input_domain():
    @artifact_inputs(ArtifactSpec.input("illum", ImageArtifactType))
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def apply_illumination(image, *, runtime):
        return image

    compiled = compile_function_pattern(
        apply_illumination,
        {
            "illum": ArtifactInputPlan(
                name="illum",
                path="/memory/illum.pkl",
                artifact_type=ImageArtifactType,
            ),
        },
        {},
    )

    group = compiled.default_group
    assert (
        group.invocations[0].runtime_domain is RuntimeInvocationDomain.ARTIFACT_MANAGED
    )
    assert group.runtime_domain is RuntimeInvocationDomain.ARTIFACT_MANAGED


def test_adapter_managed_invocation_keeps_full_declared_grouped_inputs():
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
        paths_by_group={"3": "/memory/cross_channel__3.pkl"},
    )
    current_channel_plan = ArtifactInputPlan(
        name="current_channel",
        path="/memory/current_channel.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        paths_by_group={"1": "/memory/current_channel__1.pkl"},
    )
    compiled = compile_function_pattern(
        consume_cross_channel,
        {
            "cross_channel": cross_channel_plan,
            "current_channel": current_channel_plan,
        },
        {},
    )

    invocation = compiled.default_group.invocations[0]
    scoped_artifacts = ComponentArtifactPlans(
        inputs={"current_channel": current_channel_plan.for_group("1")},
        outputs={},
    )

    selected = scoped_artifacts.select_for_invocation(
        invocation,
        declared_inputs={
            "cross_channel": cross_channel_plan,
            "current_channel": current_channel_plan,
        },
    )

    assert selected.inputs == {
        "cross_channel": cross_channel_plan,
        "current_channel": current_channel_plan,
    }


def test_adapter_recorded_outputs_use_declared_output_plans():
    contract = ModuleArtifactContract(
        module_name="RuntimeRecordsOutputs",
        items=(
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition,
                (
                    ArtifactSpec.output("Cells", ObjectLabelsArtifactType),
                    ArtifactSpec.output("Measurements", MeasurementsArtifactType),
                ),
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition,
                (
                    ArtifactSpec.output("Cells", ObjectLabelsArtifactType),
                    ArtifactSpec.output("Measurements", MeasurementsArtifactType),
                ),
            ),
        ),
    )

    @module_artifact_contract(contract)
    @runtime_adapter("runtime", lambda _request: object())
    def runtime_recorded_outputs(image, *, runtime):
        return image

    cells_output = ArtifactOutputPlan(
        name="Cells",
        path="/memory/Cells.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("3",),
        paths_by_group={"3": "/memory/w3_Cells.pkl"},
    )
    measurement_output = ArtifactOutputPlan(
        name="Measurements",
        path="/memory/Measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1",),
        paths_by_group={"1": "/memory/w1_Measurements.pkl"},
    )
    compiled = compile_function_pattern(
        runtime_recorded_outputs,
        {},
        {
            "Cells": cells_output,
            "Measurements": measurement_output,
        },
    )

    invocation = compiled.default_group.invocations[0]
    scoped_artifacts = ComponentArtifactPlans(
        inputs={},
        outputs={"Measurements": measurement_output.for_group("1")},
    )

    selected = scoped_artifacts.select_for_invocation(
        invocation,
        declared_outputs={
            "Cells": cells_output,
            "Measurements": measurement_output,
        },
    )

    assert invocation.adapter_records_artifact_outputs
    assert selected.outputs == {
        "Cells": cells_output,
        "Measurements": measurement_output,
    }


def test_compiled_group_does_not_treat_plain_artifact_inputs_as_managed_domain():
    @artifact_inputs(ArtifactSpec.input("illum", ImageArtifactType))
    def apply_illumination(image, illum):
        return image

    compiled = compile_function_pattern(
        apply_illumination,
        {
            "illum": ArtifactInputPlan(
                name="illum",
                path="/memory/illum.pkl",
                artifact_type=ImageArtifactType,
            ),
        },
        {},
    )

    group = compiled.default_group
    assert (
        group.invocations[0].runtime_domain is RuntimeInvocationDomain.SOURCE_ANCHORED
    )
    assert group.runtime_domain is RuntimeInvocationDomain.SOURCE_ANCHORED


def test_compile_function_pattern_preserves_typed_invocation_options():
    options = ExampleInvocationOptions(mode="once")

    compiled = compile_function_pattern(
        (first, {"sigma": 1}, options),
        {},
        {},
    )

    invocation = compiled.default_group.invocations[0]
    assert invocation.kwargs == (("sigma", 1),)
    assert invocation.invocation_options is options


def test_compile_function_pattern_builds_runtime_argument_plan():
    options = ExampleInvocationOptions(mode="once")

    @runtime_adapter("runtime", lambda _request: object())
    def configurable(
        image,
        *,
        context=None,
        runtime=None,
        runtime_invocation_options=None,
    ):
        return image

    compiled = compile_function_pattern(
        (configurable, {"sigma": 1}, options),
        {},
        {},
    )

    invocation = compiled.default_group.invocations[0]
    assert (
        invocation.runtime_argument_plan.context_parameter_name
        == ProcessingContext.require_parameter_name()
    )
    assert (
        invocation.runtime_argument_plan.invocation_options_parameter_name
        == RuntimeInvocationOptions.require_parameter_name()
    )
    assert invocation.runtime_argument_plan.adapter_parameter_name == "runtime"


def test_runtime_argument_plan_only_injects_options_when_invocation_declares_value():
    def accepts_options(image, *, runtime_invocation_options=None):
        return image

    compiled = compile_function_pattern(accepts_options, {}, {})

    invocation = compiled.default_group.invocations[0]
    assert invocation.contract.runtime_invocation_options_parameter == (
        RuntimeInvocationOptions.require_parameter_name()
    )
    assert invocation.runtime_argument_plan.invocation_options_parameter_name is None


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


def test_inject_kwargs_preserves_typed_invocation_options():
    options = ExampleInvocationOptions(mode="once")

    pattern = inject_kwargs_into_pattern(
        (first, {"sigma": 1}, options),
        {"dtype_config": "inherited"},
    )

    assert pattern == (
        first,
        {"dtype_config": "inherited", "sigma": 1},
        options,
    )


def test_compiled_function_pattern_filters_detected_groups_by_compiled_keys():
    compiled = compile_function_pattern({"1": first}, {}, {})

    grouped = compiled.prepare_grouped_patterns(
        {1: ["site1"], "2": ["site2"]},
        default_component="channel",
    )

    assert grouped == {1: ["site1"]}


def test_ungrouped_compiled_function_pattern_preserves_grouped_source_patterns():
    compiled = compile_function_pattern(first, {}, {})
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

    assert list(analyze.__artifact_inputs__) == ["positions"]
    assert list(analyze.__artifact_outputs__) == ["metadata", "measurements"]
    assert analyze.__artifact_outputs__["metadata"] == ArtifactSpec.output(
        "metadata",
        SpecialArtifactType,
    )
    assert (
        analyze.__artifact_outputs__["measurements"].materialization is materialization
    )


def test_artifact_decorators_accept_typed_artifact_specs():
    labels = ArtifactSpec.output("nuclei", ObjectLabelsArtifactType)
    measurements = ArtifactSpec.output("measurements", MeasurementsArtifactType)

    @artifact_inputs(labels)
    @artifact_outputs(measurements)
    def measure(image, nuclei):
        return image

    assert measure.__artifact_inputs__["nuclei"] == labels
    assert measure.__artifact_outputs__["measurements"] == measurements


def test_artifact_spec_rejects_unregistered_plan_type():
    class UnregisteredPlan(ArtifactPlan):
        pass

    with pytest.raises(ValueError, match="registered ArtifactPlan"):
        ArtifactSpec("positions", UnregisteredPlan, SpecialArtifactType)


def test_artifact_spec_rejects_unregistered_artifact_type():
    class UnregisteredArtifactType(ArtifactType):
        pass

    with pytest.raises(ValueError, match="registered ArtifactType"):
        ArtifactSpec("positions", ArtifactInputPlan, UnregisteredArtifactType)


def test_artifact_spec_rejects_relation_target_role_mismatch():
    source = ArtifactSpec.input("objects", ObjectLabelsArtifactType)

    with pytest.raises(ValueError, match="requires target plan role output"):
        ArtifactSpec.input(
            "filtered",
            ObjectLabelsArtifactType,
            relations=(GroupLineageSourceRelation(source=source.ref()),),
        )


def test_artifact_spec_relation_sources_are_full_refs():
    source_ref = ArtifactSpecRef.input("objects", ObjectLabelsArtifactType)
    relation = GroupLineageSourceRelation(source=source_ref)

    assert relation.source == source_ref
    assert relation.payload()["source"] == source_ref.payload()


def test_artifact_spec_collection_rejects_unknown_relation_sources():
    output = ArtifactSpec.output(
        "filtered",
        ObjectLabelsArtifactType,
        relations=(
            GroupLineageSourceRelation(
                source=ArtifactSpecRef.input("missing", ObjectLabelsArtifactType),
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown artifact specs"):
        ArtifactSpecCollection((output,)).validate_registered_relation_refs(
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
    def accepts_dtype_config(image, *, dtype_config=None):
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
                parameter_type=DtypeConversionConfig,
                value=inherited_config,
            ),
        ),
    )
    invocation = compiled.default_group.invocations[0]

    assert invocation.kwargs_dict == {"sigma": 2}
    assert len(invocation.runtime_parameter_bindings) == 1
    binding = invocation.runtime_parameter_bindings[0]
    assert binding.parameter_type is DtypeConversionConfig
    assert binding.value is explicit_config


def test_compile_function_pattern_strips_typed_compile_time_kwargs():
    @dataclass(frozen=True, slots=True)
    class LocalCompilePayload:
        value: str

    class LocalCompileTimeKwarg(CompileTimeFunctionKwarg[LocalCompilePayload]):
        payload_type = LocalCompilePayload

    payload = LocalCompilePayload("compile-only")

    compile_time_kwargs = CompileTimeFunctionKwargs.of(
        LocalCompileTimeKwarg,
        payload,
    )

    compiled = compile_function_pattern(
        (
            second,
            {
                COMPILE_TIME_FUNCTION_KWARGS_KEY: compile_time_kwargs,
                "sigma": 2,
            },
        ),
        {},
        {},
    )
    invocation = compiled.default_group.invocations[0]

    assert LocalCompileTimeKwarg.payload_from_kwargs(
        {COMPILE_TIME_FUNCTION_KWARGS_KEY: compile_time_kwargs}
    ) == payload
    assert invocation.kwargs_dict == {"sigma": 2}


def test_compile_function_pattern_keeps_undeclared_runtime_config_kwargs_user_owned():
    inherited_config = DtypeConfig(
        default_dtype_conversion=DtypeConversion.NATIVE_OUTPUT,
    )
    explicit_config = DtypeConfig(
        default_dtype_conversion=DtypeConversion.UINT8,
    )

    compiled = compile_function_pattern(
        (second, {"dtype_config": explicit_config, "sigma": 2}),
        {},
        {},
        runtime_parameter_bindings=(
            RuntimeParameterBinding(
                parameter_type=DtypeConversionConfig,
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
