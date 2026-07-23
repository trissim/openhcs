"""Declaration-owned CellProfiler artifact-flow and compiler tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.config_framework.object_state import ObjectState
from openhcs.config_framework.object_state_registry import ObjectStateRegistry
from openhcs.constants.constants import AllComponents, GroupBy
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    GroupLineageSourceRelation,
    ImageArtifactType,
    InputGroupLineageSourceRelation,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectMeasurementSubjectRelation,
    SourceStackLineageSourceRelation,
    SpatialGridArtifactType,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.callable_contract import CallableContract
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    PipelineInvocationContractProviderAuthority,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.artifact_planning import (
    ArtifactProducer,
    artifact_producers_for_outputs,
)
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    SourceProjectionRole,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.pipeline_import import (
    import_cellprofiler_pipeline,
)
from openhcs.processing.backends.cellprofiler.grid import (
    DefineGridManualModule,
    DefineGridVariant,
)
from openhcs.processing.backends.cellprofiler.classification import (
    ClassifyObjectsSingleMeasurementModule,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureObjectIntensityModule,
)
from openhcs.processing.backends.cellprofiler.outlines import OverlayObjectsModule
from openhcs.processing.backends.cellprofiler.morphology import MaskObjectsModule


def _module(
    module_num: int,
    name: str,
    settings: dict[str, str],
) -> ModuleBlock:
    return ModuleBlock(
        name=name,
        module_num=module_num,
        setting_records=[
            ModuleSetting(_setting_name, _setting_value)
            for (_setting_name, _setting_value) in settings.items()
        ],
    )


def _callable_contract(
    module: ModuleBlock,
    *,
    step_index: int,
    available_artifacts: ArtifactSpecCollection,
    main_flow_artifacts: ArtifactSpecCollection,
    available_artifact_producers: tuple[ArtifactProducer, ...] = (),
) -> CallableContract:
    module_type = CellProfilerModule.require_module(module.name)
    owned_refs = frozenset(
        producer.spec.ref() for producer in available_artifact_producers
    )
    fixture_outputs = tuple(
        spec
        for spec in available_artifacts
        if spec.plan_type is ArtifactOutputPlan and spec.ref() not in owned_refs
    )
    fixture_producers = artifact_producers_for_outputs(
        fixture_outputs,
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey("fixture_producer", DEFAULT_GROUP_KEY, 0),
        ),
    )
    contract = module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            function_name=str(module_type.function_name),
            group_key=DEFAULT_GROUP_KEY,
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=step_index,
            available_artifacts=available_artifacts,
            main_flow_artifacts=main_flow_artifacts,
            available_artifact_producers=(
                *available_artifact_producers,
                *fixture_producers,
            ),
        ),
    )
    assert isinstance(contract, CallableContract)
    return contract


def _compiler_contracts(
    steps: list[FunctionStep],
    pipeline_config: PipelineConfig,
) -> tuple[CallableContract, ...]:
    global_config = GlobalPipelineConfig()
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    ObjectStateRegistry.clear()
    pipeline_state = ObjectState(pipeline_config, scope_id="pipeline")

    try:
        ObjectStateRegistry.register(pipeline_state, _skip_snapshot=True)
        resolved_steps: list[FunctionStep] = []
        step_states: dict[int, ObjectState] = {}
        snapshots: list[StepSnapshot] = []
        for index, step in enumerate(steps):
            step_state = ObjectState(
                step,
                scope_id=f"pipeline::functionstep_{index}",
                parent_state=pipeline_state,
            )
            ObjectStateRegistry.register(step_state, _skip_snapshot=True)
            resolved_step = step_state.to_saved_resolved_object()
            assert isinstance(resolved_step, FunctionStep)
            resolved_steps.append(resolved_step)
            step_states[index] = step_state
            snapshots.append(
                StepSnapshot(
                    index=index,
                    scope_id=step_state.scope_id,
                    step=resolved_step,
                )
            )

        context = ProcessingContext(
            step_plans={
                index: CompiledStepPlan(
                    step_index=index,
                    step_name=step.name,
                    step_type=type(step).__name__,
                    axis_id="A01",
                )
                for index, step in enumerate(resolved_steps)
            },
            axis_id="A01",
        )
        session = CompilationSession.from_context(
            context=context,
            steps=resolved_steps,
            orchestrator=SimpleNamespace(
                pipeline_config=pipeline_state.to_object(),
            ),
            global_config=global_config,
            step_state_map=step_states,
            snapshots=tuple(snapshots),
        )
        provider = PipelineInvocationContractProviderAuthority.provider_for_session(
            session,
        )
        contracts: list[CallableContract] = []
        for snapshot in snapshots:
            invocations = tuple(
                normalize_function_pattern(snapshot.step.func).iter_items()
            )
            assert len(invocations) == 1
            plan = provider(
                invocations[0],
                ArtifactDeclarationStepContext(
                    step_name=snapshot.step.name,
                    step_index=snapshot.index,
                    source_bindings=snapshot.step.source_bindings,
                    group_by=snapshot.step.processing_config.group_by,
                    input_source=snapshot.step.processing_config.input_source,
                ),
            )
            assert plan is not None
            contracts.append(plan.contract)
        return tuple(contracts)
    finally:
        ObjectStateRegistry.clear()


def test_artifact_contract_preserves_declared_input_occurrences() -> None:
    objects_output = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)
    objects = objects_output.for_plan_type(ArtifactInputPlan)
    contract = _callable_contract(
        _module(
            1,
            "RelateObjects",
            {
                "Select the parent objects": "Objects",
                "Select the child objects": "Objects",
                "Calculate child-parent distances?": "None",
                "Calculate distances to other parents?": "No",
            },
        ),
        step_index=0,
        available_artifacts=ArtifactSpecCollection((objects_output,)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )

    assert tuple(spec.ref() for spec in contract.artifact_inputs) == (
        objects.ref(),
        objects.ref(),
    )


def test_callable_contract_combination_preserves_exact_occurrence_order() -> None:
    first_objects = ArtifactSpec.output("FirstObjects", ObjectLabelsArtifactType)
    second_objects = ArtifactSpec.output("SecondObjects", ObjectLabelsArtifactType)
    first = _callable_contract(
        _module(
            1,
            "RelateObjects",
            {
                "Select the parent objects": first_objects.name,
                "Select the child objects": first_objects.name,
                "Calculate child-parent distances?": "None",
                "Calculate distances to other parents?": "No",
            },
        ),
        step_index=0,
        available_artifacts=ArtifactSpecCollection((first_objects,)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )
    second = _callable_contract(
        _module(
            2,
            "RelateObjects",
            {
                "Select the parent objects": second_objects.name,
                "Select the child objects": second_objects.name,
                "Calculate child-parent distances?": "None",
                "Calculate distances to other parents?": "No",
            },
        ),
        step_index=1,
        available_artifacts=ArtifactSpecCollection((second_objects,)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )

    module_type = CellProfilerModule.require_module("RelateObjects")
    combined = module_type.combine_callable_contracts((first, second))

    assert combined.metadata.artifact_inputs == (
        first_objects.for_plan_type(ArtifactInputPlan),
        first_objects.for_plan_type(ArtifactInputPlan),
        second_objects.for_plan_type(ArtifactInputPlan),
        second_objects.for_plan_type(ArtifactInputPlan),
    )
    assert combined.metadata.artifact_outputs == (
        *first.metadata.artifact_outputs,
        *second.metadata.artifact_outputs,
    )


def test_callable_contract_combination_aligns_identical_occurrences() -> None:
    objects_output = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)
    contract = _callable_contract(
        _module(
            1,
            "RelateObjects",
            {
                "Select the parent objects": objects_output.name,
                "Select the child objects": objects_output.name,
                "Calculate child-parent distances?": "None",
                "Calculate distances to other parents?": "No",
            },
        ),
        step_index=0,
        available_artifacts=ArtifactSpecCollection((objects_output,)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )

    combined = CellProfilerModule.require_module(
        "RelateObjects"
    ).combine_callable_contracts((contract, contract))

    assert combined.metadata.artifact_inputs == contract.metadata.artifact_inputs
    assert combined.metadata.artifact_outputs == contract.metadata.artifact_outputs


def test_callable_contract_combination_rejects_relation_drift() -> None:
    objects_output = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)
    contract = _callable_contract(
        _module(
            1,
            "RelateObjects",
            {
                "Select the parent objects": objects_output.name,
                "Select the child objects": objects_output.name,
                "Calculate child-parent distances?": "None",
                "Calculate distances to other parents?": "No",
            },
        ),
        step_index=0,
        available_artifacts=ArtifactSpecCollection((objects_output,)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )
    first_output = contract.metadata.artifact_outputs[0]
    drifted = replace(
        contract,
        metadata=replace(
            contract.metadata,
            artifact_outputs=(
                replace(first_output, relations=()),
                *contract.metadata.artifact_outputs[1:],
            ),
        ),
    )

    with pytest.raises(ValueError, match="conflicting dynamic artifact"):
        CellProfilerModule.require_module("RelateObjects").combine_callable_contracts(
            (contract, drifted)
        )


def test_relate_objects_measurements_inherit_endpoint_group_scope() -> None:
    parent_output = ArtifactSpec.output("Objects1", ObjectLabelsArtifactType)
    child_output = ArtifactSpec.output("Objects2", ObjectLabelsArtifactType)
    parent = parent_output.for_plan_type(ArtifactInputPlan)
    child = child_output.for_plan_type(ArtifactInputPlan)
    contract = _callable_contract(
        _module(
            1,
            "RelateObjects",
            {
                "Select the parent objects": parent.name,
                "Select the child objects": child.name,
                "Calculate child-parent distances?": "None",
                "Calculate distances to other parents?": "No",
            },
        ),
        step_index=0,
        available_artifacts=ArtifactSpecCollection((parent_output, child_output)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )

    (measurement,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )
    relation_sources = {relation.source for relation in measurement.relations}
    assert {parent.ref(), child.ref()} < relation_sources
    ((relationship_output, _declaration),) = tuple(
        (spec, declaration)
        for spec, declaration in contract.artifact_outputs.relation_refs(
            ObjectRelationshipDeclaration
        )
        if declaration.projects_parent_child_measurements()
    )
    assert relationship_output.ref() in relation_sources
    assert measurement.group_scope_sources() == (child.ref(),)
    assert measurement.source_stack_scope_sources() == ()


def test_measurement_output_separates_provenance_from_invocation_group_scope() -> None:
    artifact_inputs = ArtifactSpecCollection(
        (
            ArtifactSpec.input("SourceImage", ImageArtifactType),
            ArtifactSpec.input(
                "SourceObjects",
                ObjectLabelsArtifactType,
                relations=(
                    InputGroupLineageSourceRelation(
                        ArtifactSpec.input(
                            "SourceObjects",
                            ObjectLabelsArtifactType,
                        ).ref()
                    ),
                ),
            ),
        )
    )
    invocation_key = FunctionInvocationKey(
        function_name="measure_object_intensity",
        group_key=DEFAULT_GROUP_KEY,
        position=0,
    )
    step_context = ArtifactDeclarationStepContext(
        step_name="Measurement",
        step_index=0,
        available_artifacts=artifact_inputs,
        main_flow_artifacts=ArtifactSpecCollection(()),
    )
    output = MeasureObjectIntensityModule.measurement_output_artifact(
        _module(1, "MeasureObjectIntensity", {}),
        invocation_key=invocation_key,
        step_context=step_context,
        artifact_inputs=artifact_inputs,
    )

    assert output.relations == (
        ArtifactSpecRelation(artifact_inputs.specs[0].ref()),
        ArtifactSpecRelation(artifact_inputs.specs[1].ref()),
        GroupLineageSourceRelation(artifact_inputs.specs[0].ref()),
        ObjectMeasurementSubjectRelation(artifact_inputs.specs[1].ref()),
    )
    assert output.group_scope_sources() == (artifact_inputs.specs[0].ref(),)
    assert output.source_stack_scope_sources() == ()


def test_prior_measurement_selects_its_declared_producer_group_scope() -> None:
    green = ArtifactSpec.input("OrigGreen", ImageArtifactType)
    blue = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    nuclei = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    measurements = ArtifactSpec.output(
        "MeasureObjectIntensity_4_measurements",
        MeasurementsArtifactType,
        relations=(
            ArtifactSpecRelation(
                ArtifactSpec.input(nuclei.name, ObjectLabelsArtifactType).ref()
            ),
            GroupLineageSourceRelation(green.ref()),
            GroupLineageSourceRelation(blue.ref()),
        ),
    )
    measurement_invocation = FunctionInvocationKey(
        function_name="measure_object_intensity",
        group_key=DEFAULT_GROUP_KEY,
        position=0,
    )
    contract = ClassifyObjectsSingleMeasurementModule.callable_contract(
        module=_module(
            7,
            "ClassifyObjects",
            {
                "Select the object to be classified": nuclei.name,
                "Select the measurement to classify by": (
                    "Intensity_MaxIntensity_OrigGreen"
                ),
            },
        ),
        invocation_key=FunctionInvocationKey(
            function_name="classify_objects_single_measurement",
            group_key=DEFAULT_GROUP_KEY,
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_name="ClassifyObjects",
            step_index=6,
            available_artifact_producers=(
                *artifact_producers_for_outputs(
                    (nuclei,),
                    groups=(None,),
                    invocation_keys=(
                        FunctionInvocationKey(
                            "identify_primary_objects",
                            DEFAULT_GROUP_KEY,
                            0,
                        ),
                    ),
                ),
                *artifact_producers_for_outputs(
                    (measurements,),
                    groups=("1", "2"),
                    invocation_keys=(measurement_invocation,),
                ),
            ),
            available_artifacts=ArtifactSpecCollection(
                (green, blue, nuclei, measurements)
            ),
            main_flow_artifacts=ArtifactSpecCollection(()),
        ),
    )

    measurement_input = contract.artifact_inputs.require_by_name_and_artifact_type(
        measurements.name,
        MeasurementsArtifactType,
    )
    assert measurement_input.relations == (
        InputGroupLineageSourceRelation(green.ref()),
    )


def test_declarations_carry_cross_step_object_and_measurement_flow() -> None:
    source_image = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    available = ArtifactSpecCollection((source_image,))
    main_flow = ArtifactSpecCollection((source_image,))
    identify_contract = _callable_contract(
        _module(
            1,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigBlue",
                "Name the primary objects to be identified": "Nuclei",
            },
        ),
        step_index=0,
        available_artifacts=available,
        main_flow_artifacts=main_flow,
    )
    available = ArtifactSpecCollection(
        (*available.specs, *identify_contract.artifact_outputs)
    )
    measurement_contract = _callable_contract(
        _module(
            2,
            "MeasureObjectSizeShape",
            {"Select objects to measure": "Nuclei"},
        ),
        step_index=1,
        available_artifacts=available,
        main_flow_artifacts=main_flow,
        available_artifact_producers=artifact_producers_for_outputs(
            identify_contract.artifact_outputs,
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    function_name="identify_primary_objects",
                    group_key=DEFAULT_GROUP_KEY,
                    position=0,
                ),
            ),
        ),
    )

    assert identify_contract.artifact_inputs.names() == ("OrigBlue",)
    assert identify_contract.artifact_outputs.names_of_artifact_type(
        ObjectLabelsArtifactType
    ) == ("Nuclei",)
    assert identify_contract.artifact_outputs.names() == (
        "IdentifyPrimaryObjects_1_measurements",
        "Nuclei",
    )
    identify_measurement = (
        identify_contract.artifact_outputs.require_by_name_and_artifact_type(
            "IdentifyPrimaryObjects_1_measurements",
            MeasurementsArtifactType,
        )
    )
    identify_objects = (
        identify_contract.artifact_outputs.require_by_name_and_artifact_type(
            "Nuclei",
            ObjectLabelsArtifactType,
        )
    )
    source_ref = ArtifactSpec.input("OrigBlue", ImageArtifactType).ref()
    assert identify_measurement.relations == (
        ArtifactSpecRelation(source_ref),
        GroupLineageSourceRelation(source_ref),
        ArtifactSpecRelation(identify_objects.ref()),
    )
    assert identify_objects.relations == (SourceStackLineageSourceRelation(source_ref),)
    assert identify_contract.metadata.artifact_inputs == (
        ArtifactSpec.input("OrigBlue", ImageArtifactType),
    )
    assert identify_contract.metadata.artifact_outputs == tuple(
        identify_contract.artifact_outputs
    )
    assert measurement_contract.artifact_inputs.names() == ("Nuclei",)
    assert measurement_contract.metadata.artifact_outputs == tuple(
        measurement_contract.artifact_outputs
    )
    assert measurement_contract.artifact_outputs.names_of_artifact_type(
        MeasurementsArtifactType
    ) == ("MeasureObjectSizeShape_2_measurements",)
    measurement = measurement_contract.artifact_outputs[0]
    assert measurement.relations == (
        ArtifactSpecRelation(
            ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType).ref()
        ),
        GroupLineageSourceRelation(
            ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType).ref()
        ),
        ObjectMeasurementSubjectRelation(
            ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType).ref()
        ),
    )
    assert measurement.group_scope_sources() == (
        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType).ref(),
    )
    assert measurement.source_stack_scope_sources() == ()


@pytest.mark.parametrize("in_main_flow", (True, False))
@pytest.mark.parametrize("projection_role", tuple(SourceProjectionRole))
def test_source_bound_input_is_declared_once_regardless_of_main_flow_membership(
    in_main_flow: bool,
    projection_role: SourceProjectionRole,
) -> None:
    source_image = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    module = _module(
        1,
        "IdentifyPrimaryObjects",
        {
            "Select the input image": source_image.name,
            "Name the primary objects to be identified": "Nuclei",
        },
    )
    module_type = CellProfilerModule.require_module(module.name)

    contract = module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            function_name=str(module_type.function_name),
            group_key=DEFAULT_GROUP_KEY,
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            source_bindings=StepSourceBindingsConfig(
                enabled=True,
                bindings=(
                    NamedSourceBinding(
                        alias=source_image.name,
                        artifact_kind=ImageArtifactType,
                        projection_role=projection_role,
                    ),
                ),
            ),
            available_artifacts=ArtifactSpecCollection((source_image,)),
            main_flow_artifacts=ArtifactSpecCollection(
                (source_image,) if in_main_flow else ()
            ),
        ),
    )

    assert contract.artifact_inputs.specs == (source_image,)


def test_primary_source_binding_resolves_the_exact_module_input() -> None:
    source_image = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    module = _module(
        1,
        "IdentifyPrimaryObjects",
        {
            "Select the input image": source_image.name,
            "Name the primary objects to be identified": "Nuclei",
        },
    )
    module_type = CellProfilerModule.require_module(module.name)
    source_binding = NamedSourceBinding(
        alias=source_image.name,
        artifact_kind=ImageArtifactType,
        component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
    )

    contract = module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            function_name=str(module_type.function_name),
            group_key="2",
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            source_bindings=StepSourceBindingsConfig(
                enabled=True,
                bindings=(source_binding,),
            ),
            group_by=GroupBy.CHANNEL,
            available_artifacts=ArtifactSpecCollection((source_image,)),
            main_flow_artifacts=ArtifactSpecCollection((source_image,)),
        ),
    )

    assert contract.artifact_inputs.specs == (source_image,)


def test_primary_source_binding_rejects_a_different_channel_group() -> None:
    source_image = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    module = _module(
        1,
        "IdentifyPrimaryObjects",
        {
            "Select the input image": source_image.name,
            "Name the primary objects to be identified": "Nuclei",
        },
    )
    module_type = CellProfilerModule.require_module(module.name)

    with pytest.raises(
        ValueError,
        match="No source binding declares channel group '1'",
    ):
        module_type.callable_contract(
            module=module,
            invocation_key=FunctionInvocationKey(
                function_name=str(module_type.function_name),
                group_key="1",
                position=0,
            ),
            step_context=ArtifactDeclarationStepContext(
                step_index=0,
                source_bindings=StepSourceBindingsConfig(
                    enabled=True,
                    bindings=(
                        NamedSourceBinding(
                            alias=source_image.name,
                            artifact_kind=ImageArtifactType,
                            component_identity=(
                                ComponentSelector(AllComponents.CHANNEL, "2"),
                            ),
                        ),
                    ),
                ),
                group_by=GroupBy.CHANNEL,
                available_artifacts=ArtifactSpecCollection((source_image,)),
                main_flow_artifacts=ArtifactSpecCollection((source_image,)),
            ),
        )


def test_group_artifact_context_reseeds_main_flow_from_exact_primary_bindings() -> None:
    dna = NamedSourceBinding(
        alias="DNA",
        component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
    )
    rna = NamedSourceBinding(
        alias="RNA",
        component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
    )
    illumination = NamedSourceBinding(
        alias="Illumination",
        artifact_kind=ImageArtifactType,
        projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
        component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
    )
    bindings = (dna, rna, illumination)
    context = ArtifactDeclarationStepContext(
        source_bindings=StepSourceBindingsConfig(enabled=True, bindings=bindings),
        group_by=GroupBy.CHANNEL,
        input_source=InputSource.PIPELINE_START,
    ).with_source_declarations(binding.input_spec() for binding in bindings)

    scoped = CellProfilerModule._artifact_context_for_group(context, group_key="2")

    assert tuple(
        binding.alias for binding in scoped.source_bindings.binding_declarations
    ) == ("RNA", "Illumination")
    assert scoped.main_flow_artifacts.names() == ("RNA",)


@pytest.mark.parametrize("module_name", ("ImageMath", "Tile"))
def test_ordinary_multi_image_inputs_use_only_scoped_main_flow(
    module_name: str,
) -> None:
    current_outputs = (
        ArtifactSpec.output("Current1", ImageArtifactType),
        ArtifactSpec.output("Current2", ImageArtifactType),
    )
    current_inputs = tuple(
        spec.for_plan_type(ArtifactInputPlan) for spec in current_outputs
    )
    context = ArtifactDeclarationStepContext(
        step_name=module_name,
        step_index=0,
        available_artifacts=ArtifactSpecCollection(current_outputs),
        available_artifact_producers=(
            *artifact_producers_for_outputs(
                (current_outputs[0],),
                groups=("1",),
                invocation_keys=(FunctionInvocationKey("upstream", "1", 0),),
            ),
            *artifact_producers_for_outputs(
                (current_outputs[1],),
                groups=("2",),
                invocation_keys=(FunctionInvocationKey("upstream", "2", 0),),
            ),
        ),
        main_flow_artifacts=ArtifactSpecCollection(current_inputs),
    )
    module_type = CellProfilerModule.require_module(module_name)
    invocation = next(
        normalize_function_pattern({"2": module_type.require_callable()}).iter_items()
    )
    blocks, consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = module_type.number_step_invocation_blocks(
        (blocks,),
        first_module_num=1,
    )
    contract, contract_consumed = module_type.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed,
        step_context=context,
    )

    assert contract_consumed == consumed
    assert contract.artifact_inputs.names_of_artifact_type(ImageArtifactType) == (
        "Current2",
    )


def test_measure_image_quality_all_loaded_uses_only_primary_plane_bindings() -> None:
    bindings = (
        NamedSourceBinding(alias="DNA"),
        NamedSourceBinding(alias="RNA"),
        NamedSourceBinding(
            alias="Illumination",
            artifact_kind=ImageArtifactType,
            projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
        ),
    )
    source_specs = tuple(binding.input_spec() for binding in bindings)
    module = _module(
        1,
        "MeasureImageQuality",
        {"Calculate metrics for which images?": "All loaded images"},
    )
    module_type = CellProfilerModule.require_module(module.name)

    contract = module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            str(module_type.function_name), DEFAULT_GROUP_KEY, 0
        ),
        step_context=ArtifactDeclarationStepContext(
            step_name=module.name,
            step_index=0,
            source_bindings=StepSourceBindingsConfig(
                enabled=True,
                bindings=bindings,
            ),
            input_source=InputSource.PIPELINE_START,
            available_artifacts=ArtifactSpecCollection(source_specs),
            main_flow_artifacts=ArtifactSpecCollection(source_specs),
        ),
    )

    assert contract.artifact_inputs.names() == ("DNA", "RNA")


def test_runtime_input_reconstruction_uses_exact_producer_invocation_group() -> None:
    module_type = CellProfilerModule.require_module("OverlayObjects")
    invocation = next(
        normalize_function_pattern({"2": module_type.require_callable()}).iter_items()
    )
    image = ArtifactSpec.input("DNA", ImageArtifactType)
    group_one_objects = ArtifactSpec.output("Objects1", ObjectLabelsArtifactType)
    group_two_objects = ArtifactSpec.output("Objects2", ObjectLabelsArtifactType)
    producers = (
        *artifact_producers_for_outputs(
            (group_one_objects,),
            groups=("1",),
            invocation_keys=(FunctionInvocationKey("identify", "1", 0),),
        ),
        *artifact_producers_for_outputs(
            (group_two_objects,),
            groups=("2",),
            invocation_keys=(FunctionInvocationKey("identify", "2", 0),),
        ),
    )

    (block,), consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=ArtifactDeclarationStepContext(
            step_name="OverlayObjects",
            step_index=1,
            available_artifacts=ArtifactSpecCollection(
                (image, group_one_objects, group_two_objects)
            ),
            available_artifact_producers=producers,
            main_flow_artifacts=ArtifactSpecCollection((image,)),
        ),
    )

    assert consumed == ()
    (objects_binding,) = module_type.declared_artifact_bindings(
        plan_type=ArtifactInputPlan,
        artifact_type=ObjectLabelsArtifactType,
    )
    assert module_type.artifact_names_for_binding(
        block,
        objects_binding,
    ) == ("Objects2",)


def test_multi_image_input_reconstruction_uses_exact_source_lineage() -> None:
    module_type = CellProfilerModule.require_module("CorrectIlluminationApply")
    invocation = next(
        normalize_function_pattern(module_type.require_callable()).iter_items()
    )
    sources = (
        ArtifactSpec.input("OrigStain1", ImageArtifactType),
        ArtifactSpec.input("OrigStain2", ImageArtifactType),
    )
    illumination = tuple(
        ArtifactSpec.output(
            f"IllumStain{position}",
            ImageArtifactType,
            relations=(SourceStackLineageSourceRelation(source=source.ref()),),
        )
        for position, source in enumerate(sources, start=1)
    )
    producers = tuple(
        producer
        for position, spec in enumerate(illumination, start=1)
        for producer in artifact_producers_for_outputs(
            (spec,),
            groups=(str(position),),
            invocation_keys=(
                FunctionInvocationKey(
                    "correct_illumination_calculate",
                    str(position),
                    0,
                ),
            ),
        )
    )

    blocks, consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=ArtifactDeclarationStepContext(
            step_name="CorrectIlluminationApply",
            step_index=1,
            available_artifacts=ArtifactSpecCollection((*sources, *illumination)),
            available_artifact_producers=producers,
            main_flow_artifacts=ArtifactSpecCollection(sources),
        ),
    )

    assert consumed == ()
    assert len(blocks) == 2
    assert tuple(
        module_type.artifact_names_for_binding(
            block,
            module_type.input_image_binding,
        )
        for block in blocks
    ) == (("OrigStain1",), ("OrigStain2",))
    assert tuple(
        module_type.artifact_names_for_binding(
            block,
            module_type.illumination_function_binding,
        )
        for block in blocks
    ) == (("IllumStain1",), ("IllumStain2",))


def test_missing_runtime_producer_requires_explicit_artifact_identity() -> None:
    module_type = CellProfilerModule.require_module("OverlayObjects")
    invocation = next(
        normalize_function_pattern({"2": module_type.require_callable()}).iter_items()
    )
    image = ArtifactSpec.input("DNA", ImageArtifactType)
    objects = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)
    context = ArtifactDeclarationStepContext(
        step_name="OverlayObjects",
        step_index=1,
        available_artifacts=ArtifactSpecCollection((image, objects)),
        available_artifact_producers=artifact_producers_for_outputs(
            (objects,),
            groups=("1",),
            invocation_keys=(FunctionInvocationKey("identify", "1", 0),),
        ),
        main_flow_artifacts=ArtifactSpecCollection((image,)),
    )

    blocks, consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )

    assert blocks == ()
    assert consumed == ()


def test_explicit_runtime_artifact_identity_requires_exact_producer() -> None:
    module_type = CellProfilerModule.require_module("OverlayObjects")
    identity_parameter = (
        OverlayObjectsModule.input_objects_binding.require_parameter_name()
    )
    invocation = next(
        normalize_function_pattern(
            {
                "2": (
                    module_type.require_callable(),
                    {identity_parameter: "Objects"},
                )
            }
        ).iter_items()
    )
    image = ArtifactSpec.input("DNA", ImageArtifactType)
    objects = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)
    context = ArtifactDeclarationStepContext(
        step_name="OverlayObjects",
        step_index=1,
        available_artifacts=ArtifactSpecCollection((image, objects)),
        main_flow_artifacts=ArtifactSpecCollection((image,)),
    )

    blocks, consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = module_type.number_step_invocation_blocks(
        (blocks,),
        first_module_num=1,
    )
    with pytest.raises(
        ValueError,
        match="no exact source binding, scoped main-flow declaration, or ArtifactProducer",
    ):
        module_type.invocation_callable_contract(
            invocation=invocation,
            numbered_module_blocks=numbered_blocks,
            consumed_kwarg_names=consumed,
            step_context=context,
        )

    assert consumed == (identity_parameter,)


def test_explicit_runtime_artifact_identity_uses_exact_cross_group_producer() -> None:
    module_type = CellProfilerModule.require_module("OverlayObjects")
    identity_parameter = (
        OverlayObjectsModule.input_objects_binding.require_parameter_name()
    )
    invocation = next(
        normalize_function_pattern(
            {
                "2": (
                    module_type.require_callable(),
                    {identity_parameter: "Objects"},
                )
            }
        ).iter_items()
    )
    image = ArtifactSpec.input("DNA", ImageArtifactType)
    objects = ArtifactSpec.output("Objects", ObjectLabelsArtifactType)
    context = ArtifactDeclarationStepContext(
        step_name="OverlayObjects",
        step_index=1,
        available_artifacts=ArtifactSpecCollection((image, objects)),
        available_artifact_producers=artifact_producers_for_outputs(
            (objects,),
            groups=("1",),
            invocation_keys=(FunctionInvocationKey("identify", "1", 0),),
        ),
        main_flow_artifacts=ArtifactSpecCollection((image,)),
    )

    blocks, consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = module_type.number_step_invocation_blocks(
        (blocks,),
        first_module_num=1,
    )
    contract, contract_consumed = module_type.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed,
        step_context=context,
    )

    assert consumed == (identity_parameter,)
    assert contract_consumed == consumed
    assert contract.artifact_inputs.names() == ("DNA", "Objects")


def test_public_union_artifact_identity_selects_exact_declared_type() -> None:
    module_type = CellProfilerModule.require_module("MaskObjects")
    invocation = next(
        normalize_function_pattern(
            (
                module_type.require_callable(),
                {
                    "select_the_input_objects": "Nuclei",
                    "select_the_masking_object": "SharedMask",
                },
            )
        ).iter_items()
    )
    image = ArtifactSpec.input("DNA", ImageArtifactType)
    nuclei = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    masking_objects = ArtifactSpec.output(
        "SharedMask",
        ObjectLabelsArtifactType,
    )

    (block,), consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=ArtifactDeclarationStepContext(
            step_name="MaskObjects",
            step_index=1,
            available_artifacts=ArtifactSpecCollection(
                (
                    image,
                    nuclei,
                    ArtifactSpec.input("SharedMask", ImageArtifactType),
                    masking_objects,
                )
            ),
            main_flow_artifacts=ArtifactSpecCollection((image,)),
        ),
    )

    assert consumed == (
        "select_the_input_objects",
        "select_the_masking_object",
    )
    assert module_type.artifact_names_for_binding(
        block,
        MaskObjectsModule.masking_objects_binding,
    ) == ("SharedMask",)
    assert (
        module_type.artifact_names_for_binding(
            block,
            MaskObjectsModule.masking_image_binding,
        )
        == ()
    )


def test_public_union_artifact_identity_selects_exact_image_type() -> None:
    module_type = CellProfilerModule.require_module("MaskObjects")
    invocation = next(
        normalize_function_pattern(
            (
                module_type.require_callable(),
                {
                    "select_the_input_objects": "Nuclei",
                    "select_the_masking_image": "SharedMask",
                },
            )
        ).iter_items()
    )
    image = ArtifactSpec.input("DNA", ImageArtifactType)
    nuclei = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)

    (block,), consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=ArtifactDeclarationStepContext(
            step_name="MaskObjects",
            step_index=1,
            available_artifacts=ArtifactSpecCollection(
                (
                    image,
                    nuclei,
                    ArtifactSpec.input("SharedMask", ImageArtifactType),
                    ArtifactSpec.output(
                        "SharedMask",
                        ObjectLabelsArtifactType,
                    ),
                )
            ),
            main_flow_artifacts=ArtifactSpecCollection((image,)),
        ),
    )

    assert consumed == (
        "select_the_input_objects",
        "select_the_masking_image",
    )
    assert module_type.artifact_names_for_binding(
        block,
        MaskObjectsModule.masking_image_binding,
    ) == ("SharedMask",)
    assert (
        module_type.artifact_names_for_binding(
            block,
            MaskObjectsModule.masking_objects_binding,
        )
        == ()
    )


def test_default_group_consumes_artifact_carried_by_main_flow() -> None:
    gray_tumor_output = ArtifactSpec.output("GrayTumor", ImageArtifactType)
    gray_tumor_input = gray_tumor_output.for_plan_type(ArtifactInputPlan)
    producer_invocation = FunctionInvocationKey(
        function_name="color_to_gray",
        group_key="1",
        position=0,
    )
    module = _module(
        2,
        "IdentifyPrimaryObjects",
        {
            "Select the input image": "GrayTumor",
            "Name the primary objects to be identified": "tumor",
        },
    )
    module_type = CellProfilerModule.require_module(module.name)

    (producer,) = artifact_producers_for_outputs(
        (gray_tumor_output,),
        groups=("1",),
        invocation_keys=(producer_invocation,),
    )
    contract = module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            function_name=str(module_type.function_name),
            group_key=DEFAULT_GROUP_KEY,
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_name=module.name,
            step_index=1,
            available_artifacts=ArtifactSpecCollection((gray_tumor_output,)),
            main_flow_artifacts=ArtifactSpecCollection((gray_tumor_input,)),
            available_artifact_producers=(producer,),
        ),
    )

    assert producer.groups == ("1",)
    assert producer.invocation_keys == (producer_invocation,)
    assert contract.artifact_inputs.names() == ("GrayTumor",)


def test_group_one_producer_satisfies_exact_group_two_consumer() -> None:
    gray_tumor_output = ArtifactSpec.output("GrayTumor", ImageArtifactType)
    gray_tumor_input = gray_tumor_output.for_plan_type(ArtifactInputPlan)
    producer_invocation = FunctionInvocationKey(
        function_name="color_to_gray",
        group_key="1",
        position=0,
    )
    (producer,) = artifact_producers_for_outputs(
        (gray_tumor_output,),
        groups=("1",),
        invocation_keys=(producer_invocation,),
    )
    module = _module(
        2,
        "IdentifyPrimaryObjects",
        {
            "Select the input image": "GrayTumor",
            "Name the primary objects to be identified": "tumor",
        },
    )
    module_type = CellProfilerModule.require_module(module.name)

    assert producer.groups == ("1",)
    assert producer.invocation_keys == (producer_invocation,)
    contract = module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            function_name=str(module_type.function_name),
            group_key="2",
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_name=module.name,
            step_index=1,
            available_artifacts=ArtifactSpecCollection((gray_tumor_output,)),
            main_flow_artifacts=ArtifactSpecCollection((gray_tumor_input,)),
            available_artifact_producers=(producer,),
        ),
    )

    assert contract.artifact_inputs.names() == ("GrayTumor",)


def test_object_output_lineage_uses_unique_object_input_among_other_artifacts() -> None:
    grid_output = ArtifactSpec.output("Grid", SpatialGridArtifactType)
    guides_output = ArtifactSpec.output("Guides", ObjectLabelsArtifactType)
    guides = guides_output.for_plan_type(ArtifactInputPlan)
    contract = _callable_contract(
        _module(
            1,
            "IdentifyObjectsInGrid",
            {
                "Select the defined grid": "Grid",
                "Select object shapes and locations": ("Natural Shape and Location"),
                "Select the guiding objects": "Guides",
                "Name the objects to be identified": "GridObjects",
            },
        ),
        step_index=0,
        available_artifacts=ArtifactSpecCollection((grid_output, guides_output)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )

    (output,) = contract.artifact_outputs.of_artifact_type(ObjectLabelsArtifactType)
    assert output.relations == (SourceStackLineageSourceRelation(source=guides.ref()),)
    (measurements,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )
    assert ArtifactSpecRelation(source=output.ref()) in measurements.relations


def test_expand_or_shrink_uses_standard_object_transform_measurement_contract() -> None:
    source_output = ArtifactSpec.output("Objects1", ObjectLabelsArtifactType)
    source = source_output.for_plan_type(ArtifactInputPlan)
    contract = _callable_contract(
        _module(
            1,
            "ExpandOrShrinkObjects",
            {
                "Select the input objects": source.name,
                "Name the output objects": "ExpandedObjects1",
                "Select the operation": "Expand objects until touching",
            },
        ),
        step_index=0,
        available_artifacts=ArtifactSpecCollection((source_output,)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )

    (output,) = contract.artifact_outputs.of_artifact_type(ObjectLabelsArtifactType)
    (measurements,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )

    assert output.name == "ExpandedObjects1"
    assert ArtifactSpecRelation(source=output.ref()) in measurements.relations


@pytest.mark.parametrize(
    ("module", "available_artifacts", "main_flow_artifacts"),
    (
        (
            _module(
                1,
                "IdentifySecondaryObjects",
                {
                    "Select the input image": "DNA",
                    "Select the input objects": "Nuclei",
                    "Name the objects to be identified": "Cells",
                },
            ),
            ArtifactSpecCollection(
                (
                    ArtifactSpec.input("DNA", ImageArtifactType),
                    ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
                )
            ),
            ArtifactSpecCollection((ArtifactSpec.input("DNA", ImageArtifactType),)),
        ),
        (
            _module(
                2,
                "IdentifyTertiaryObjects",
                {
                    "Select the larger identified objects": "Cells",
                    "Select the smaller identified objects": "Nuclei",
                    "Name the tertiary objects to be identified": "Cytoplasm",
                },
            ),
            ArtifactSpecCollection(
                (
                    ArtifactSpec.output("Cells", ObjectLabelsArtifactType),
                    ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
                )
            ),
            ArtifactSpecCollection(()),
        ),
    ),
)
def test_parent_child_measurements_depend_on_declared_relationship_outputs(
    module: ModuleBlock,
    available_artifacts: ArtifactSpecCollection,
    main_flow_artifacts: ArtifactSpecCollection,
) -> None:
    contract = _callable_contract(
        module,
        step_index=0,
        available_artifacts=available_artifacts,
        main_flow_artifacts=main_flow_artifacts,
    )
    (measurement,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )
    relationship_outputs = tuple(
        spec
        for spec, declaration in contract.artifact_outputs.relation_refs(
            ObjectRelationshipDeclaration
        )
        if declaration.projects_parent_child_measurements()
    )

    assert relationship_outputs
    assert all(
        ArtifactSpecRelation(source=relationship.ref()) in measurement.relations
        for relationship in relationship_outputs
    )


def test_relate_objects_measurements_depend_on_declared_relationship_output() -> None:
    parent_output = ArtifactSpec.output("Objects1", ObjectLabelsArtifactType)
    child_output = ArtifactSpec.output("Objects2", ObjectLabelsArtifactType)
    parent = parent_output.for_plan_type(ArtifactInputPlan)
    child = child_output.for_plan_type(ArtifactInputPlan)
    contract = _callable_contract(
        _module(
            1,
            "RelateObjects",
            {
                "Select the parent objects": parent.name,
                "Select the child objects": child.name,
                "Calculate child-parent distances?": "None",
                "Calculate distances to other parents?": "No",
            },
        ),
        step_index=0,
        available_artifacts=ArtifactSpecCollection((parent_output, child_output)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )
    (measurement,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )
    relationship_outputs = tuple(
        spec
        for spec, declaration in contract.artifact_outputs.relation_refs(
            ObjectRelationshipDeclaration
        )
        if declaration.projects_parent_child_measurements()
    )

    assert len(relationship_outputs) == 1
    assert (
        ArtifactSpecRelation(source=relationship_outputs[0].ref())
        in measurement.relations
    )


@pytest.mark.parametrize(
    ("definition_method", "variant", "expected_source"),
    (
        (
            DefineGridManualModule.DefinitionMethod.manual,
            DefineGridVariant.MANUAL,
            ("Brightfield", ImageArtifactType),
        ),
        (
            DefineGridManualModule.DefinitionMethod.automatic,
            DefineGridVariant.AUTOMATIC,
            ("Guides", ObjectLabelsArtifactType),
        ),
    ),
)
def test_define_grid_declares_its_exact_geometry_source(
    definition_method: DefineGridManualModule.DefinitionMethod,
    variant: DefineGridVariant,
    expected_source: tuple[str, type],
) -> None:
    current_image = ArtifactSpec.input("Brightfield", ImageArtifactType)
    guides = ArtifactSpec.output("Guides", ObjectLabelsArtifactType)
    module = _module(
        1,
        "DefineGrid",
        {
            "Name the grid": "Grid",
            "Select the method to define the grid": definition_method.value,
            "Select the image on which to display the grid": current_image.name,
            "Select the previously identified objects": guides.name,
        },
    )
    contract = DefineGridManualModule.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            function_name=variant.value,
            group_key=DEFAULT_GROUP_KEY,
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            available_artifact_producers=artifact_producers_for_outputs(
                (guides,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey("fixture_producer", DEFAULT_GROUP_KEY, 0),
                ),
            ),
            available_artifacts=ArtifactSpecCollection((current_image, guides)),
            main_flow_artifacts=ArtifactSpecCollection((current_image,)),
        ),
    )

    (grid,) = contract.artifact_outputs.of_artifact_type(SpatialGridArtifactType)
    expected_source_name, expected_source_type = expected_source
    assert grid.group_scope_sources() == (
        contract.artifact_inputs.require_by_name_and_artifact_type(
            expected_source_name,
            expected_source_type,
        ).ref(),
    )


def test_unguided_grid_output_uses_current_main_flow_image_lineage() -> None:
    current_image = ArtifactSpec.input("Brightfield", ImageArtifactType)
    darkfield = ArtifactSpec.input("Darkfield", ImageArtifactType)
    marker = ArtifactSpec.input("Marker", ImageArtifactType)
    grid = ArtifactSpec.output(
        "Grid",
        SpatialGridArtifactType,
        relations=(GroupLineageSourceRelation(source=current_image.ref()),),
    )
    contract = _callable_contract(
        _module(
            1,
            "IdentifyObjectsInGrid",
            {
                "Select the defined grid": "Grid",
                "Select object shapes and locations": "Rectangle Forced Location",
                "Name the objects to be identified": "GridObjects",
            },
        ),
        step_index=0,
        available_artifacts=ArtifactSpecCollection(
            (grid, current_image, darkfield, marker)
        ),
        main_flow_artifacts=ArtifactSpecCollection((current_image, darkfield, marker)),
    )

    assert tuple(spec.ref() for spec in contract.artifact_inputs) == (
        current_image.ref(),
        grid.ref().for_plan_type(ArtifactInputPlan),
    )
    (output,) = contract.artifact_outputs.of_artifact_type(ObjectLabelsArtifactType)
    assert output.relations == (
        SourceStackLineageSourceRelation(source=current_image.ref()),
    )


def test_unguided_grid_without_declared_source_fails_during_declaration() -> None:
    grid = ArtifactSpec.output("Grid", SpatialGridArtifactType)

    with pytest.raises(ValueError, match="declare exactly one source domain"):
        _callable_contract(
            _module(
                1,
                "IdentifyObjectsInGrid",
                {
                    "Select the defined grid": "Grid",
                    "Select object shapes and locations": "Rectangle Forced Location",
                    "Name the objects to be identified": "GridObjects",
                },
            ),
            step_index=0,
            available_artifacts=ArtifactSpecCollection((grid,)),
            main_flow_artifacts=ArtifactSpecCollection(()),
        )


def test_calculate_math_selects_measurements_by_nominal_feature_owner() -> None:
    nuclei = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    crop_blue = ArtifactSpec.output("CropBlue", ImageArtifactType)

    def measurement(
        name: str,
        function_name: str,
        *sources: ArtifactSpec,
    ) -> tuple[ArtifactSpec, ArtifactProducer]:
        spec = ArtifactSpec.output(
            name,
            MeasurementsArtifactType,
            relations=tuple(
                GroupLineageSourceRelation(
                    source=source.for_plan_type(ArtifactInputPlan).ref()
                )
                for source in sources
            ),
        )
        (producer,) = artifact_producers_for_outputs(
            (spec,),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    function_name=function_name,
                    group_key=DEFAULT_GROUP_KEY,
                    position=0,
                ),
            ),
        )
        return spec, producer

    shape, shape_producer = measurement(
        "shape_measurements",
        "measure_object_size_shape",
        nuclei,
    )
    image_intensity, image_intensity_producer = measurement(
        "image_intensity_measurements",
        "measure_image_intensity",
        crop_blue,
    )
    object_intensity, object_intensity_producer = measurement(
        "object_intensity_measurements",
        "measure_object_intensity",
        nuclei,
        crop_blue,
    )
    texture, texture_producer = measurement(
        "texture_measurements",
        "measure_texture_objects",
        nuclei,
        crop_blue,
    )
    colocalization, colocalization_producer = measurement(
        "colocalization_measurements",
        "measure_colocalization_objects",
        nuclei,
        crop_blue,
    )
    available = ArtifactSpecCollection(
        (
            nuclei,
            crop_blue,
            shape,
            image_intensity,
            object_intensity,
            texture,
            colocalization,
        )
    )
    module = _module(
        6,
        "CalculateMath",
        {
            "Name the output measurement": "Ratio",
            "Operation": "Divide",
            "Select the numerator objects": "Nuclei",
            "Select the numerator measurement": "Intensity_MeanIntensity_CropBlue",
            "Select the denominator objects": "Nuclei",
            "Select the denominator measurement": "AreaShape_Area",
        },
    )
    module_type = CellProfilerModule.require_module(module.name)
    source_producers = artifact_producers_for_outputs(
        (nuclei, crop_blue),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey("fixture_producer", DEFAULT_GROUP_KEY, 0),
        ),
    )
    contract = module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            function_name="calculate_math",
            group_key=DEFAULT_GROUP_KEY,
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_name=module.name,
            step_index=5,
            available_artifact_producers=(
                *source_producers,
                shape_producer,
                image_intensity_producer,
                object_intensity_producer,
                texture_producer,
                colocalization_producer,
            ),
            available_artifacts=available,
            main_flow_artifacts=ArtifactSpecCollection(
                (crop_blue.for_plan_type(ArtifactInputPlan),)
            ),
        ),
    )

    selected_measurements = contract.artifact_inputs.of_artifact_type(
        MeasurementsArtifactType
    )
    assert tuple(spec.name for spec in selected_measurements) == (
        "object_intensity_measurements",
        "shape_measurements",
    )
    assert tuple(
        spec.group_scope_sources() for spec in selected_measurements
    ) == (
        (crop_blue.for_plan_type(ArtifactInputPlan).ref(),),
        (nuclei.for_plan_type(ArtifactInputPlan).ref(),),
    )
    (calculation_output,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )
    assert {
        type(relation).relation_key for relation in calculation_output.relations
    } >= {
        "calculate_math_numerator_object",
        "calculate_math_denominator_object",
    }
    assert calculation_output.group_scope_sources() == tuple(
        spec.ref() for spec in selected_measurements
    )
    assert {
        relation.source
        for relation in calculation_output.relations
        if type(relation) is ArtifactSpecRelation
    } == {spec.ref() for spec in contract.artifact_inputs}
    assert calculation_output.source_stack_scope_sources() == ()


def test_cellprofiler_workspace_images_do_not_become_implicit_exports() -> None:
    source = ArtifactSpec.input("SourceImage", ImageArtifactType)
    contract = _callable_contract(
        _module(
            1,
            "ColorToGray",
            {
                "Select the input image": source.name,
                "Name the output image": "WorkspaceImage",
                "Conversion method": "Combine",
                "Image type": "RGB",
            },
        ),
        step_index=0,
        available_artifacts=ArtifactSpecCollection((source,)),
        main_flow_artifacts=ArtifactSpecCollection((source,)),
    )

    (image,) = contract.artifact_outputs
    assert image.materialization is None


def test_declaration_rejects_unknown_and_conflicting_artifact_inputs() -> None:
    module = _module(
        1,
        "MeasureObjectSizeShape",
        {"Select objects to measure": "Nuclei"},
    )
    empty = ArtifactSpecCollection(())

    with pytest.raises(
        ValueError,
        match="unknown object_labels artifact 'Nuclei'",
    ):
        _callable_contract(
            module,
            step_index=0,
            available_artifacts=empty,
            main_flow_artifacts=empty,
        )

    image_with_same_name = ArtifactSpecCollection(
        (ArtifactSpec.output("Nuclei", ImageArtifactType),)
    )
    with pytest.raises(
        ValueError,
        match="references 'Nuclei' as object_labels.*declare types.*image",
    ):
        _callable_contract(
            module,
            step_index=0,
            available_artifacts=image_with_same_name,
            main_flow_artifacts=empty,
        )


def test_area_occupied_object_selector_binds_exact_callable_parameter() -> None:
    objects = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    contract = _callable_contract(
        _module(
            2,
            "MeasureImageAreaOccupiedBinary",
            {
                "Measure the area occupied in a binary image, or in objects?": "Objects",
                "Select objects to measure": objects.name,
            },
        ),
        step_index=1,
        available_artifacts=ArtifactSpecCollection((objects,)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )

    (object_input,) = contract.artifact_inputs.of_artifact_type(
        ObjectLabelsArtifactType
    )
    assert object_input.parameter_name == "object_labels"


def test_declaration_resolves_exact_source_binding_without_cursor_artifact() -> None:
    module = _module(
        1,
        "MeasureObjectSizeShape",
        {"Select objects to measure": "Nuclei"},
    )
    module_type = CellProfilerModule.require_module(module.name)
    empty = ArtifactSpecCollection(())

    contract = module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            function_name=str(module_type.function_name),
            group_key=DEFAULT_GROUP_KEY,
            position=0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            source_bindings=StepSourceBindingsConfig(
                enabled=True,
                bindings=(
                    NamedSourceBinding(
                        alias="Nuclei",
                        artifact_kind=ObjectLabelsArtifactType,
                        projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                    ),
                ),
            ),
            available_artifacts=empty,
            main_flow_artifacts=empty,
        ),
    )

    assert contract.artifact_inputs.names() == ("Nuclei",)


def test_pure_imported_steps_compile_the_same_artifact_chain(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "object-measurement.cppipe"
    cppipe_path.write_text(
        """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain \"DNA\")
IdentifyPrimaryObjects:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
MeasureObjectSizeShape:[module_num:3|enabled:True]
    Select objects to measure:Nuclei
""",
        encoding="utf-8",
    )

    steps, pipeline_config = import_cellprofiler_pipeline(cppipe_path)

    assert isinstance(pipeline_config, PipelineConfig)
    assert all(isinstance(step, FunctionStep) for step in steps)
    assert [step.name for step in steps] == [
        "IdentifyPrimaryObjects",
        "MeasureObjectSizeShape",
    ]
    contracts = _compiler_contracts(steps, pipeline_config)
    module_names: list[str] = []
    for contract in contracts:
        module_type = CellProfilerModule.for_function_name(contract.function_name)
        assert module_type is not None
        module_names.append(module_type.require_module_name())
    assert module_names == [
        "IdentifyPrimaryObjects",
        "MeasureObjectSizeShape",
    ]
    assert contracts[0].artifact_inputs.names() == ("DNA",)
    assert contracts[0].artifact_outputs.names_of_artifact_type(
        ObjectLabelsArtifactType
    ) == ("Nuclei",)
    assert contracts[1].artifact_inputs.names() == ("Nuclei",)
    assert contracts[1].artifact_outputs.names_of_artifact_type(
        MeasurementsArtifactType
    ) == ("MeasureObjectSizeShape_2_measurements",)
