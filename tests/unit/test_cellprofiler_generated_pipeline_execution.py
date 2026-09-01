"""Public CellProfiler pipeline compilation and module-contract coverage."""

from __future__ import annotations

import inspect
from pathlib import Path
from queue import SimpleQueue

import numpy as np
import pytest
import tifffile

from objectstate.lazy_factory import ensure_global_config_context
from openhcs.constants.constants import AllComponents, Microscope
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ArtifactType,
    GroupLineageSourceRelation,
    ImageArtifactType,
    InputStackBroadcastSourceRelation,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
    RelationshipsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.callable_contract import (
    CallableContract,
    FunctionStepExecutionScope,
)
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyProcessingConfig,
    LazySourceBindingsConfig,
    PipelineConfig,
)
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    get_core_callable,
    normalize_function_pattern,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline.artifact_planning import (
    ArtifactProducer,
    artifact_producers_for_outputs,
)
from openhcs.core.pipeline.function_contracts import (
    validate_artifact_input_parameter_bindings,
)
from openhcs.core.progress import set_progress_queue
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerModuleExecutor,
)
from openhcs.processing.backends.cellprofiler import (
    correct_illumination_calculate,
)
from openhcs.processing.backends.cellprofiler.object_filtering import (
    FilterMode,
    FilterObjectsModule,
    FilterObjectsRemovedObjectSourceRelation,
    filter_objects,
)
from openhcs.processing.backends.cellprofiler.relationships import RelateObjectsModule
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
)
from openhcs.processing.backends.lib_registry.registry_service import RegistryService

PUBLIC_IMPORT_CPIPE = """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:1
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain "DNA")
MedianFilter:[module_num:2|enabled:True]
    Select the input image:DNA
    Name the output image:FilteredDNA
    Window:5
SaveImages:[module_num:3|enabled:True]
    Select the image to save:FilteredDNA
"""


def _write_public_import_cppipe(tmp_path: Path) -> Path:
    cppipe_path = tmp_path / "public-import.cppipe"
    cppipe_path.write_text(PUBLIC_IMPORT_CPIPE, encoding="utf-8")
    return cppipe_path


def _transport_round_trip(steps: list[FunctionStep]) -> tuple[str, list[FunctionStep]]:
    source = FunctionStepTransportAuthority.source_from_pipeline(steps)
    namespace: dict[str, object] = {}
    exec(compile(source, "<cellprofiler-pipeline>", "exec"), namespace)
    reconstructed = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )
    return source, reconstructed


def test_pure_import_returns_only_public_steps_and_pipeline_config(
    tmp_path: Path,
) -> None:
    steps, pipeline_config = import_cellprofiler_pipeline(
        _write_public_import_cppipe(tmp_path)
    )

    assert type(steps) is list
    assert isinstance(pipeline_config, PipelineConfig)
    assert [step.name for step in steps] == ["MedianFilter", "SaveImages"]
    assert all(isinstance(step, FunctionStep) for step in steps)
    assert all(get_core_callable(step.func) is not None for step in steps)
    assert all(
        not isinstance(get_core_callable(step.func), CellProfilerModuleExecutor)
        for step in steps
    )
    assert tuple(
        binding.alias for binding in pipeline_config.source_bindings_config.bindings
    ) == ("DNA",)


def test_generic_function_step_transport_round_trips_imported_declarations(
    tmp_path: Path,
) -> None:
    steps, _pipeline_config = import_cellprofiler_pipeline(
        _write_public_import_cppipe(tmp_path)
    )

    source, reconstructed = _transport_round_trip(steps)

    assert FunctionStepTransportAuthority.source_from_pipeline(reconstructed) == source
    assert [step.name for step in reconstructed] == [step.name for step in steps]


def test_compiler_derives_runtime_executor_after_generic_transport(
    tmp_path: Path,
) -> None:
    tifffile.imwrite(
        tmp_path / "A01_s001_OrigStain1.tif",
        np.ones((4, 4), dtype=np.uint16),
    )
    source_binding = NamedSourceBinding(
        alias="OrigStain1",
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.CONTAINS,
                    value="OrigStain1",
                ),
            )
        ),
        component_identity=(
            ComponentSelector(
                component=AllComponents.CHANNEL,
                value="1",
            ),
        ),
    )
    declared_step = FunctionStep(
        func=(
            correct_illumination_calculate,
            {"name_the_output_image": "IllumStain1"},
        ),
        name="CorrectIlluminationCalculate",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
        source_bindings=StepSourceBindingsConfig(enabled=True),
    )
    _source, transported_steps = _transport_round_trip([declared_step])
    transported_callable = get_core_callable(transported_steps[0].func)
    assert transported_callable is RegistryService.registered_callable(
        correct_illumination_calculate
    )
    assert not isinstance(transported_callable, CellProfilerModuleExecutor)

    global_config = GlobalPipelineConfig()
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    orchestrator = PipelineOrchestrator(
        tmp_path,
        pipeline_config=PipelineConfig(
            microscope=Microscope.SOURCE_BINDINGS,
            source_bindings_config=LazySourceBindingsConfig(
                bindings=(source_binding,),
            ),
        ),
    )
    set_progress_queue(SimpleQueue())
    try:
        orchestrator.initialize()
        compilation = orchestrator.compile_pipelines(
            pipeline_definition=transported_steps,
            well_filter=["A01"],
            is_zmq_execution=True,
        )
        context = compilation.runtime_contexts["A01"]
        compiled_pattern = context.step_plans[0].compiled_function_pattern
    finally:
        set_progress_queue(None)

    assert compiled_pattern is not None
    invocation = next(compiled_pattern.iter_invocations())
    contract = invocation.contract
    runtime_callable = contract.resolve_runtime_callable()
    module_type = CellProfilerModule.for_callable_contract(contract)

    assert invocation.kwargs == ()
    assert module_type is not None
    assert module_type.require_module_name() == "CorrectIlluminationCalculate"
    assert contract.artifact_inputs.names() == ("OrigStain1",)
    assert contract.artifact_outputs.names() == ("IllumStain1",)
    assert isinstance(runtime_callable, CellProfilerModuleExecutor)
    assert runtime_callable.raw_func is correct_illumination_calculate
    assert runtime_callable.callable_contract is contract
    assert (
        module_type.require_callable(contract.function_name)
        is runtime_callable.raw_func
    )


def _target_module_block(
    name: str,
    records: tuple[tuple[str, str], ...],
) -> ModuleBlock:
    setting_records = [ModuleSetting(key, value) for key, value in records]
    return ModuleBlock(
        name=name,
        module_num=47,
        enabled=True,
        setting_records=setting_records,
    )


def _target_module_contract(
    module: ModuleBlock,
    *,
    available: tuple[ArtifactSpec, ...],
    main_flow: tuple[ArtifactSpec, ...] = (),
    available_artifact_producers: tuple[ArtifactProducer, ...] | None = None,
    step_index: int = 3,
) -> CallableContract:
    module_type = CellProfilerModule.require_module(module.name)
    source_aliases = tuple(
        spec.name
        for spec in available
        if spec.plan_type is ArtifactInputPlan
        and spec.artifact_type is ImageArtifactType
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=bool(source_aliases),
        bindings=tuple(
            NamedSourceBinding(alias=alias, artifact_kind=ImageArtifactType)
            for alias in source_aliases
        ),
    )
    invocation_key = FunctionInvocationKey(
        function_name=module_type.require_callable().__name__,
        group_key="default",
        position=0,
    )
    producers = available_artifact_producers
    if producers is None:
        producers = artifact_producers_for_outputs(
            tuple(spec for spec in available if spec.plan_type is ArtifactOutputPlan),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    function_name=f"fixture_step_{step_index}_producer",
                    group_key=invocation_key.group_key,
                    position=0,
                ),
            ),
        )
    return module_type.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_name=module.name,
            step_index=step_index,
            source_bindings=source_bindings,
            available_artifact_producers=producers,
            available_artifacts=ArtifactSpecCollection(available),
            main_flow_artifacts=ArtifactSpecCollection(
                spec.for_plan_type(ArtifactInputPlan) for spec in main_flow
            ),
        ),
    )


def _artifact_names(
    specs: ArtifactSpecCollection,
    artifact_type: type[ArtifactType] | None = None,
) -> tuple[str, ...]:
    if artifact_type is not None:
        specs = tuple(spec for spec in specs if spec.artifact_type is artifact_type)
    return tuple(spec.name for spec in specs)


def _relationship_declarations(
    contract: CallableContract,
    artifact_type: type[ObjectLineageArtifactType],
) -> tuple[ObjectRelationshipDeclaration, ...]:
    return tuple(
        declaration
        for spec, declaration in contract.artifact_outputs.relation_refs(
            ObjectRelationshipDeclaration
        )
        if spec.artifact_type is artifact_type
    )


def test_cellprofiler_module_contract_boundary_has_exact_existing_value_inputs() -> (
    None
):
    signature = inspect.signature(CellProfilerModule.callable_contract)

    assert tuple(signature.parameters) == (
        "module",
        "invocation_key",
        "step_context",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


def test_illumination_contracts_derive_canonical_image_flow_from_artifacts() -> None:
    original = ArtifactSpec.input("OrigStain1", ImageArtifactType)
    calculate = _target_module_contract(
        _target_module_block(
            "CorrectIlluminationCalculate",
            (
                ("Select the input image", "OrigStain1"),
                ("Name the output image", "IllumStain1"),
                (
                    "Calculate function for each image individually, or based on all images?",
                    "Each",
                ),
            ),
        ),
        available=(original,),
        main_flow=(original,),
    )
    illumination = next(
        spec
        for spec in calculate.artifact_outputs
        if spec.artifact_type is ImageArtifactType
    )
    apply = _target_module_contract(
        _target_module_block(
            "CorrectIlluminationApply",
            (
                ("Select the input image", "OrigStain1"),
                ("Select the illumination function", "IllumStain1"),
                ("Name the output image", "CorrectedStain1"),
            ),
        ),
        available=(original, illumination),
        main_flow=(original,),
    )

    assert _artifact_names(calculate.artifact_inputs, ImageArtifactType) == (
        "OrigStain1",
    )
    assert _artifact_names(calculate.artifact_outputs, ImageArtifactType) == (
        "IllumStain1",
    )
    assert _artifact_names(apply.artifact_inputs, ImageArtifactType) == (
        "OrigStain1",
        "IllumStain1",
    )
    assert apply.artifact_inputs[1].relations == (
        InputStackBroadcastSourceRelation(source=original.ref()),
    )
    assert _artifact_names(apply.artifact_outputs, ImageArtifactType) == (
        "CorrectedStain1",
    )


def test_align_contract_preserves_ordered_multi_image_inputs_and_outputs() -> None:
    inputs = (
        ArtifactSpec.input("OrigStain1", ImageArtifactType),
        ArtifactSpec.input("OrigStain2", ImageArtifactType),
    )
    contract = _target_module_contract(
        _target_module_block(
            "Align",
            (
                ("Select the first input image", "OrigStain1"),
                ("Name the first output image", "Stain1"),
                ("Select the second input image", "OrigStain2"),
                ("Name the second output image", "Stain2"),
            ),
        ),
        available=inputs,
        main_flow=inputs,
    )

    assert _artifact_names(contract.artifact_inputs, ImageArtifactType) == (
        "OrigStain1",
        "OrigStain2",
    )
    assert _artifact_names(contract.artifact_outputs, ImageArtifactType) == (
        "Stain1",
        "Stain2",
    )
    assert len(contract.artifact_outputs) == 3


def test_measure_colocalization_contract_consumes_pair_and_emits_measurements() -> None:
    images = (
        ArtifactSpec.output("Stain1", ImageArtifactType),
        ArtifactSpec.output("Stain2", ImageArtifactType),
    )
    contract = _target_module_contract(
        _target_module_block(
            "MeasureColocalization",
            (
                ("Select images to measure", "Stain1"),
                ("Select images to measure", "Stain2"),
                ("Select where to measure correlation", "Across entire image"),
            ),
        ),
        available=images,
        main_flow=images,
    )

    assert _artifact_names(contract.artifact_inputs, ImageArtifactType) == (
        "Stain1",
        "Stain2",
    )
    measurement_names = _artifact_names(
        contract.artifact_outputs, MeasurementsArtifactType
    )
    assert measurement_names == ("MeasureColocalization_47_measurements",)


def test_object_topology_contracts_derive_from_declared_object_artifacts() -> None:
    image = ArtifactSpec.output("Stain1", ImageArtifactType)
    identify = _target_module_contract(
        _target_module_block(
            "IdentifyPrimaryObjects",
            (
                ("Select the input image", "Stain1"),
                ("Name the primary objects to be identified", "Objects1"),
            ),
        ),
        available=(image,),
        main_flow=(image,),
    )
    objects1 = next(
        spec
        for spec in identify.artifact_outputs
        if spec.artifact_type is ObjectLabelsArtifactType
    )
    objects2 = ArtifactSpec.output("Objects2", ObjectLabelsArtifactType)
    relate = _target_module_contract(
        _target_module_block(
            "RelateObjects",
            (
                ("Select the parent objects", "Objects1"),
                ("Select the child objects", "Objects2"),
                ("Calculate child-parent distances?", "None"),
                ("Calculate distances to other parents?", "No"),
            ),
        ),
        available=(objects1, objects2),
    )

    assert _artifact_names(identify.artifact_outputs, ObjectLabelsArtifactType) == (
        "Objects1",
    )
    assert _artifact_names(relate.artifact_inputs, ObjectLabelsArtifactType) == (
        "Objects1",
        "Objects2",
    )
    assert tuple(
        (
            declaration.relationship_type,
            declaration.source.name,
            declaration.target.name,
        )
        for declaration in _relationship_declarations(
            relate,
            RelationshipsArtifactType,
        )
    ) == (
        ("Parent", "Objects1", "Objects2"),
        ("Child", "Objects2", "Objects1"),
    )
    assert len(_artifact_names(relate.artifact_outputs, MeasurementsArtifactType)) == 1


def test_filter_objects_contract_declares_lineage_without_topology_kwargs() -> None:
    objects = ArtifactSpec.output("Objects1", ObjectLabelsArtifactType)
    measurements = ArtifactSpec.output(
        "MeasureObjectSizeShape_measurements",
        MeasurementsArtifactType,
        measurement_feature_owner=MeasureObjectSizeShapeModule,
        relations=(
            GroupLineageSourceRelation(
                source=objects.for_plan_type(ArtifactInputPlan).ref()
            ),
        ),
    )
    object_producer, measurement_producer = artifact_producers_for_outputs(
        (objects, measurements),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey(
                function_name="measure_object_size_shape",
                group_key="default",
                position=0,
            ),
        ),
    )
    contract = _target_module_contract(
        _target_module_block(
            "FilterObjects",
            (
                ("Select the object to filter", "Objects1"),
                ("Name the output objects", "FilteredObjects"),
                ("Filter using classifier rules or measurements?", "Measurements"),
                ("Select the filtering method", "Limits"),
                ("Additional object count", "0"),
                ("Select the measurement to filter by", "AreaShape_Area"),
            ),
        ),
        available=(objects, measurements),
        available_artifact_producers=(object_producer, measurement_producer),
    )

    assert _artifact_names(contract.artifact_inputs, ObjectLabelsArtifactType) == (
        "Objects1",
    )
    assert _artifact_names(contract.artifact_inputs, MeasurementsArtifactType) == (
        "MeasureObjectSizeShape_measurements",
    )
    assert _artifact_names(contract.artifact_outputs, ObjectLabelsArtifactType) == (
        "FilteredObjects",
    )
    assert tuple(
        (declaration.source.name, declaration.target.name)
        for declaration in _relationship_declarations(
            contract,
            ObjectLineageArtifactType,
        )
    ) == (("Objects1", "FilteredObjects"),)


def test_filter_objects_child_count_consumes_exact_declared_relationship() -> None:
    parent_output = ArtifactSpec.output("Objects1", ObjectLabelsArtifactType)
    child_output = ArtifactSpec.output("Objects2", ObjectLabelsArtifactType)
    parent_input = parent_output.for_plan_type(ArtifactInputPlan)
    child_input = child_output.for_plan_type(ArtifactInputPlan)
    relationship_declaration = ObjectRelationshipDeclaration.parent_child(
        source=parent_input.ref(),
        target=child_input.ref(),
        producer_module_number=12,
    )
    relationship_output = ArtifactSpec.output(
        relationship_declaration.artifact_name(),
        RelationshipsArtifactType,
        relations=(relationship_declaration,),
    )
    measurement_output = ArtifactSpec.output(
        "RelateObjects_12_measurements",
        MeasurementsArtifactType,
        measurement_feature_owner=RelateObjectsModule,
        relations=(
            GroupLineageSourceRelation(source=parent_input.ref()),
            ArtifactSpecRelation(source=relationship_output.ref()),
        ),
    )
    object_producers = artifact_producers_for_outputs(
        (parent_output, child_output),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey(
                function_name="identify_primary_objects",
                group_key="default",
                position=0,
            ),
        ),
    )
    relationship_producers = artifact_producers_for_outputs(
        (relationship_output, measurement_output),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey(
                function_name="relate_objects",
                group_key="default",
                position=0,
            ),
        ),
    )

    contract = _target_module_contract(
        _target_module_block(
            "FilterObjects",
            (
                ("Select the object to filter", parent_output.name),
                ("Name the output objects", "FilteredObjects"),
                ("Filter using classifier rules or measurements?", "Measurements"),
                ("Select the filtering method", "Limits"),
                ("Additional object count", "0"),
                ("Select the measurement to filter by", "Children_Objects2_Count"),
                ("Filter using a minimum measurement value?", "Yes"),
                ("Minimum value", "1"),
                ("Filter using a maximum measurement value?", "No"),
                ("Maximum value", "1"),
            ),
        ),
        available=(
            parent_output,
            child_output,
            relationship_output,
            measurement_output,
        ),
        available_artifact_producers=(
            *object_producers,
            *relationship_producers,
        ),
    )

    assert _artifact_names(
        contract.artifact_inputs,
        RelationshipsArtifactType,
    ) == (relationship_output.name,)
    validate_artifact_input_parameter_bindings(
        filter_objects,
        contract.artifact_inputs,
        adapter_manages_inputs=True,
    )


def test_filter_objects_contract_declares_removed_object_topology() -> None:
    objects = ArtifactSpec.output("Objects1", ObjectLabelsArtifactType)
    contract = _target_module_contract(
        _target_module_block(
            "FilterObjects",
            (
                ("Select the object to filter", "Objects1"),
                ("Name the output objects", "FilteredObjects"),
                ("Filter using classifier rules or measurements?", "Border"),
                ("Select the filtering method", "Limits"),
                ("Additional object count", "0"),
                ("Keep removed objects as a separate set?", "Yes"),
                ("Name the objects removed by the filter", "RemovedObjects"),
            ),
        ),
        available=(objects,),
    )

    output_objects = ArtifactSpecCollection(contract.artifact_outputs).of_artifact_type(
        ObjectLabelsArtifactType
    )
    assert tuple(spec.name for spec in output_objects) == (
        "FilteredObjects",
        "RemovedObjects",
    )
    assert tuple(
        (spec.name, relation.source.name)
        for spec, relation in ArtifactSpecCollection(output_objects).relation_refs(
            FilterObjectsRemovedObjectSourceRelation
        )
    ) == (("RemovedObjects", "Objects1"),)
    assert tuple(
        (declaration.source.name, declaration.target.name)
        for declaration in _relationship_declarations(
            contract,
            ObjectLineageArtifactType,
        )
    ) == (
        ("Objects1", "FilteredObjects"),
        ("Objects1", "RemovedObjects"),
    )


@pytest.mark.parametrize(
    "primary_object_name",
    ("Nuclei", "Prespots", "Tile_of_grid"),
)
def test_filter_objects_function_step_reconstructs_exact_public_topology(
    primary_object_name: str,
) -> None:
    primary = ArtifactSpec.output(primary_object_name, ObjectLabelsArtifactType)
    additional = ArtifactSpec.output("Additional", ObjectLabelsArtifactType)
    step = FunctionStep(
        func=(
            filter_objects,
            {
                "mode": FilterMode.BORDER,
                "additional_object_count": 1,
                "emit_removed_objects": True,
            },
        ),
        name="FilterObjects",
    )
    (invocation,) = tuple(normalize_function_pattern(step.func).iter_items())
    step_context = ArtifactDeclarationStepContext(
        step_name=step.name,
        step_index=4,
        available_artifact_producers=artifact_producers_for_outputs(
            (primary, additional),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "fixture_producer",
                    invocation.key.group_key,
                    0,
                ),
            ),
        ),
        available_artifacts=ArtifactSpecCollection((primary, additional)),
        main_flow_artifacts=ArtifactSpecCollection(()),
    )

    blocks, consumed = FilterObjectsModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=step_context,
    )
    assert len(blocks) == 1
    (numbered_blocks,), _next_module_num = (
        FilterObjectsModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract = FilterObjectsModule.callable_contract(
        module=numbered_blocks[0],
        invocation_key=invocation.key,
        step_context=step_context,
    )
    validate_artifact_input_parameter_bindings(
        filter_objects,
        contract.artifact_inputs,
        adapter_manages_inputs=True,
    )

    assert consumed == ()
    assert _artifact_names(contract.artifact_inputs, ObjectLabelsArtifactType) == (
        primary_object_name,
        "Additional",
    )
    assert {
        spec.parameter_name
        for spec in contract.artifact_inputs.of_artifact_type(ObjectLabelsArtifactType)
    } == {"object_labels"}
    object_outputs = ArtifactSpecCollection(contract.artifact_outputs).of_artifact_type(
        ObjectLabelsArtifactType
    )
    assert len(object_outputs) == 3
    assert not ArtifactSpecCollection(contract.artifact_outputs).of_artifact_type(
        ImageArtifactType
    )
    assert tuple(
        relation.source.name
        for _spec, relation in ArtifactSpecCollection(object_outputs).relation_refs(
            FilterObjectsRemovedObjectSourceRelation
        )
    ) == (primary_object_name,)


def test_filter_objects_parsed_cppipe_reconstructs_additional_and_removed_outputs(
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "filter-objects-topology.cppipe"
    cppipe.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: https://cellprofiler.org",
                "FilterObjects:[module_num:7|enabled:True]",
                "    Select the object to filter:Primary",
                "    Name the output objects:FilteredPrimary",
                "    Filter using classifier rules or measurements?:Border",
                "    Select the filtering method:Limits",
                "    Additional object count:1",
                "    Select additional object to relabel:Additional",
                "    Name the relabeled objects:FilteredAdditional",
                "    Keep removed objects as a separate set?:Yes",
                "    Name the objects removed by the filter:RemovedPrimary",
            )
        ),
        encoding="utf-8",
    )
    (module,) = CPPipeParser(cppipe).parse()
    contract = _target_module_contract(
        module,
        available=(
            ArtifactSpec.output("Primary", ObjectLabelsArtifactType),
            ArtifactSpec.output("Additional", ObjectLabelsArtifactType),
        ),
    )

    assert _artifact_names(contract.artifact_inputs, ObjectLabelsArtifactType) == (
        "Primary",
        "Additional",
    )
    assert _artifact_names(contract.artifact_outputs, ObjectLabelsArtifactType) == (
        "FilteredPrimary",
        "FilteredAdditional",
        "RemovedPrimary",
    )
    assert not ArtifactSpecCollection(contract.artifact_outputs).of_artifact_type(
        ImageArtifactType
    )
    assert tuple(
        (spec.name, relation.source.name)
        for spec, relation in ArtifactSpecCollection(
            contract.artifact_outputs
        ).relation_refs(FilterObjectsRemovedObjectSourceRelation)
    ) == (("RemovedPrimary", "Primary"),)


def test_area_occupied_and_calculate_math_chain_measurement_artifacts() -> None:
    objects = (
        ArtifactSpec.output("Objects1", ObjectLabelsArtifactType),
        ArtifactSpec.output("Objects2", ObjectLabelsArtifactType),
    )
    area = _target_module_contract(
        _target_module_block(
            "MeasureImageAreaOccupied",
            (
                (
                    "Measure the area occupied in a binary image, or in objects?",
                    "Objects",
                ),
                ("Select objects to measure", "Objects1"),
                (
                    "Measure the area occupied in a binary image, or in objects?",
                    "Objects",
                ),
                ("Select objects to measure", "Objects2"),
            ),
        ),
        available=objects,
    )
    area_measurement = next(
        spec
        for spec in area.artifact_outputs
        if spec.artifact_type is MeasurementsArtifactType
    )
    (area_producer,) = artifact_producers_for_outputs(
        (area_measurement,),
        groups=(None,),
        invocation_keys=(
            FunctionInvocationKey(
                function_name="measure_image_area_occupied",
                group_key="default",
                position=0,
            ),
        ),
    )
    math = _target_module_contract(
        _target_module_block(
            "CalculateMath",
            (
                ("Name the output measurement", "ObjectAreaRatio"),
                ("Operation", "Divide"),
                (
                    "Select the numerator measurement",
                    "AreaOccupied_AreaOccupied_Objects1",
                ),
                (
                    "Select the denominator measurement",
                    "AreaOccupied_AreaOccupied_Objects2",
                ),
                ("Select the numerator objects", "None"),
                ("Select the denominator objects", "None"),
            ),
        ),
        available=(*objects, area_measurement),
        available_artifact_producers=(area_producer,),
    )

    assert _artifact_names(area.artifact_inputs, ObjectLabelsArtifactType) == (
        "Objects1",
        "Objects2",
    )
    assert len(_artifact_names(area.artifact_outputs, MeasurementsArtifactType)) == 1
    assert _artifact_names(math.artifact_inputs, MeasurementsArtifactType) == (
        area_measurement.name,
    )
    assert len(_artifact_names(math.artifact_outputs, MeasurementsArtifactType)) == 1


def test_track_objects_contract_derives_object_identity_and_retained_image() -> None:
    objects = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    contract = _target_module_contract(
        _target_module_block(
            "TrackObjects",
            (
                ("Choose a tracking method", "Overlap"),
                ("Select the objects to track", "Nuclei"),
                ("Save color-coded image?", "Yes"),
                ("Name the output image", "TrackedCells"),
            ),
        ),
        available=(objects,),
    )

    assert _artifact_names(contract.artifact_inputs, ObjectLabelsArtifactType) == (
        "Nuclei",
    )
    assert _artifact_names(contract.artifact_outputs, ImageArtifactType) == (
        "TrackedCells",
    )
    assert (
        len(_artifact_names(contract.artifact_outputs, MeasurementsArtifactType)) == 1
    )


def test_area_occupied_object_rows_declare_measurements_only() -> None:
    objects = ArtifactSpec.output("Objects1", ObjectLabelsArtifactType)
    contract = _target_module_contract(
        _target_module_block(
            "MeasureImageAreaOccupied",
            (
                ("Measure the area occupied by", "Objects"),
                ("Select object sets to measure", "Objects1"),
            ),
        ),
        available=(objects,),
    )

    assert _artifact_names(contract.artifact_inputs, ObjectLabelsArtifactType) == (
        "Objects1",
    )
    assert _artifact_names(contract.artifact_outputs, ImageArtifactType) == ()


def test_tile_contract_uses_declared_images_in_order_without_group_wrapper() -> None:
    images = (
        ArtifactSpec.input("OrigColor", ImageArtifactType),
        ArtifactSpec.output("OutlineImage", ImageArtifactType),
        ArtifactSpec.output("TrackedCells", ImageArtifactType),
    )
    contract = _target_module_contract(
        _target_module_block(
            "Tile",
            (
                ("Select an input image", "OrigColor"),
                ("Select an additional image to tile", "OutlineImage"),
                ("Select an additional image to tile", "TrackedCells"),
                ("Name the output image", "TiledImage"),
            ),
        ),
        available=images,
        main_flow=(images[0],),
    )

    assert _artifact_names(contract.artifact_inputs, ImageArtifactType) == (
        "OrigColor",
        "OutlineImage",
        "TrackedCells",
    )
    assert _artifact_names(contract.artifact_outputs, ImageArtifactType) == (
        "TiledImage",
    )


def test_save_images_is_axis_step_with_declared_only_materialized_output() -> None:
    image = ArtifactSpec.output("TiledImage", ImageArtifactType)
    module = _target_module_block(
        "SaveImages",
        (
            ("Select the image to save", "TiledImage"),
            ("Saved file format", "tiff"),
            ("Image bit depth", "8-bit integer"),
        ),
    )
    module_type = CellProfilerModule.require_module(module.name)
    contract = _target_module_contract(module, available=(image,), main_flow=(image,))
    callable_contract = CallableContract.from_callable(module_type.require_callable())

    assert callable_contract.execution_scope is FunctionStepExecutionScope.AXIS
    assert _artifact_names(contract.artifact_inputs, ImageArtifactType) == (
        "TiledImage",
    )
    (materialized_output,) = contract.artifact_outputs
    assert materialized_output.artifact_type is ImageArtifactType
    assert materialized_output.materialization is not None


@pytest.mark.parametrize("module_name", ("ExportToSpreadsheet", "ExportToDatabase"))
def test_aggregate_exporters_are_plate_steps_over_exact_runtime_artifacts(
    module_name: str,
) -> None:
    available = (
        ArtifactSpec.output("Images_measurements", MeasurementsArtifactType),
        ArtifactSpec.output("Nuclei_measurements", MeasurementsArtifactType),
        ArtifactSpec.output("Nuclei_Cells_relationships", RelationshipsArtifactType),
        ArtifactSpec.output("DNA", ImageArtifactType),
    )
    records = (
        (
            ("Database type", "SQLite"),
            ("Name the SQLite database file", "analysis.db"),
            ("Experiment name", "Example"),
            ("Add a prefix to table names?", "No"),
            ("Create a CellProfiler Analyst properties file?", "Yes"),
            ("Export measurements for all objects to the database?", "All"),
            ("Export object relationships?", "Yes"),
            (
                "Create one table per object, a single object table or a single object view?",
                "Single object table",
            ),
        )
        if module_name == "ExportToDatabase"
        else ()
    )
    module = _target_module_block(module_name, records)
    module_type = CellProfilerModule.require_module(module_name)
    contract = _target_module_contract(module, available=available)
    callable_contract = CallableContract.from_callable(module_type.require_callable())

    assert callable_contract.execution_scope is FunctionStepExecutionScope.PLATE
    assert set(_artifact_names(contract.artifact_inputs)).issuperset(
        {
            "Images_measurements",
            "Nuclei_measurements",
            "Nuclei_Cells_relationships",
        }
    )
    (bundle_output,) = contract.artifact_outputs
    assert bundle_output.artifact_type is SpecialArtifactType
    assert bundle_output.materialization is not None
    assert tuple(relation.source for relation in bundle_output.relations) == tuple(
        spec.ref() for spec in contract.artifact_inputs
    )
