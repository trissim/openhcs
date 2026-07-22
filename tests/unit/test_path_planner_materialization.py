from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    GroupLineageSourceRelation,
    ImageArtifactType,
    InputGroupLineageSourceRelation,
    InputStackBroadcastSourceRelation,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.compiled_step_plan import (
    CompiledStepPlan,
    MaterializedOutputPlan,
)
from openhcs.core.callable_contract import CallableContract, FunctionStepExecutionScope
from openhcs.core.component_set import ComponentSet
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.config import ProcessingConfig, StepMaterializationConfig
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    CompositeInvocationContractProvider,
    InvocationContractPlan,
    InvocationContractProvider,
    callable_contract_artifact_declarations,
    unnamed_main_flow_artifact_name,
)
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
)
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    CompiledFunctionPattern,
    compile_function_pattern,
)
from openhcs.core.pipeline.artifact_planning import (
    ArtifactConsumer,
    ArtifactGraph,
    ArtifactProducer,
    extract_artifact_declarations,
)
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    artifact_outputs,
    execution_scope,
    runtime_bound_parameters,
    special_inputs,
)
from openhcs.core.pipeline.funcstep_contract_validator import FuncStepContractValidator
from openhcs.core.pipeline.path_planner import (
    ArtifactPlanMaps,
    MissingArtifactInputError,
    PathPlanner,
    PathPlannerComponentScopes,
    PathPlannerArtifactStage,
    PathPlannerExecutionGroups,
    PathPlannerGroupScope,
    PathPlannerMaterializationStage,
    PathPlannerPathAuthority,
    PathPlannerStepAssemblyStage,
    PathPlannerValidationStage,
)
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.runtime_adapters import runtime_adapter
from openhcs.core.runtime_object_labels import ObjectLabelValue
from openhcs.core.runtime_stores import RuntimeArtifactBatch
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    CompiledSourceUniversePlan,
    ComponentSelector,
    EMPTY_SOURCE_BINDINGS,
    NamedSourceBinding,
    SourceBindingsConfig,
    SourceProjectionRole,
    StepSourceBindingsConfig,
)
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.step_dependencies import StepInputDependencyKind
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.function_runtime import ComponentArtifactPlans


@dataclass(frozen=True)
class PathConfigStub:
    sub_dir: str
    output_dir_suffix: str = "_processed"
    global_output_folder: str | None = None


def _artifact_planner_stub() -> PathPlanner:
    planner = PathPlanner.__new__(PathPlanner)
    planner.plate_path = Path("/data/plate1")
    planner.cfg = PathConfigStub(sub_dir="images")
    planner.session = SimpleNamespace(
        global_config=SimpleNamespace(materialization_results_path="analysis"),
        realized_source_metadata=None,
        path_resolver=None,
    )
    planner.ctx = SimpleNamespace(
        axis_id="A01",
        microscope_handler=SimpleNamespace(
            can_resolve_metadata_artifact=lambda _artifact_name: False,
        ),
    )
    planner.orchestrator = SimpleNamespace(
        get_component_keys=lambda _component: (),
    )
    planner.plans = {
        2: CompiledStepPlan(
            step_index=2,
            step_scope_id="plate::functionstep_2",
            step_name="identify",
            step_type="FunctionStep",
            axis_id="A01",
        ),
        3: CompiledStepPlan(
            step_index=3,
            step_scope_id="plate::functionstep_3",
            step_name="filter",
            step_type="FunctionStep",
            axis_id="A01",
        ),
    }
    planner.declared = {}
    planner.future_artifact_inputs = [set() for _ in range(5)]
    planner.source_bindings_defaults = SourceBindingsConfig()
    planner.step_source_bindings_defaults = StepSourceBindingsConfig()
    planner.declaration_provider = callable_contract_artifact_declarations
    planner.invocation_contract_provider = CompositeInvocationContractProvider(())
    planner.artifact_context = ArtifactDeclarationStepContext.empty()
    planner.main_flow_component_scopes = {}
    planner.execution_groups = PathPlannerExecutionGroups(planner)
    planner.paths = PathPlannerPathAuthority(planner)
    planner.artifacts = PathPlannerArtifactStage(planner)
    planner.materialization = PathPlannerMaterializationStage(planner)
    planner.validation = PathPlannerValidationStage(planner)
    planner.steps = PathPlannerStepAssemblyStage(planner)
    return planner


def _record_declared_output(
    planner: PathPlanner,
    plan: ArtifactOutputPlan,
) -> ArtifactOutputPlan:
    """Record one test producer under its exact graph identity."""

    planner.declared[plan.ref()] = plan
    return plan


class _NonFunctionStep(AbstractStep):
    def process(self, context, step_index: int) -> None:
        del context, step_index


def _snapshot(
    *,
    index: int = 3,
    scope_id: str | None = None,
    name: str = "step",
    is_function_step: bool = True,
    func=None,
    source_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS,
    group_by: GroupBy = GroupBy.CHANNEL,
    variable_components: tuple[VariableComponents, ...] = (VariableComponents.SITE,),
    input_source: InputSource = InputSource.PREVIOUS_STEP,
    processing_config: ProcessingConfig | None = None,
    step_materialization_config=None,
) -> StepSnapshot:
    if func is None:

        def passthrough(image):
            return image

        func = passthrough
    processing_config = processing_config or ProcessingConfig(
        group_by=group_by,
        variable_components=list(variable_components),
        input_source=input_source,
    )
    step_kwargs = dict(
        name=name,
        processing_config=processing_config,
        source_bindings=source_bindings,
        step_materialization_config=(
            step_materialization_config or StepMaterializationConfig()
        ),
    )
    step = (
        FunctionStep(func=func, **step_kwargs)
        if is_function_step
        else _NonFunctionStep(**step_kwargs)
    )
    return StepSnapshot(
        index=index,
        scope_id=scope_id or f"plate::functionstep_{index}",
        step=step,
    )


def test_metadata_satisfied_artifact_input_compiles_without_runtime_plan():
    @artifact_inputs("grid_dimensions")
    def metadata_consumer(image, grid_dimensions):
        return image, grid_dimensions

    planner = _artifact_planner_stub()
    planner.ctx.microscope_handler = SimpleNamespace(
        can_resolve_metadata_artifact=lambda name: name == "grid_dimensions",
        resolve_metadata_artifact=lambda name, _plate_path: (
            (2, 3) if name == "grid_dimensions" else None
        ),
    )
    planner.ctx.plate_path = planner.plate_path
    pattern = (metadata_consumer, {"grid_dimensions": None})
    snapshot = _snapshot(
        func=pattern,
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(NamedSourceBinding(alias="DNA"),),
        ),
    )
    declarations, pattern, _, contracts = (
        planner.artifacts.prepare_step_declarations(snapshot)
    )
    execution_bindings = planner.artifacts.source_bindings_for_contracts(
        snapshot,
        contracts,
        StepInputDependency.step_output(
            source_step_index=2,
            source_step_scope_id="plate::functionstep_2",
        ),
    )
    execution_group_scope = planner.execution_groups.get_execution_groups(
        snapshot,
        PathPlannerComponentScopes.empty(),
        source_bindings=execution_bindings,
        contracts=contracts,
    )
    runtime_input_plans = planner.artifacts.process_artifact_inputs(
        declarations,
        snapshot.index,
        PathPlannerGroupScope.ungrouped(),
        execution_bindings,
        ComponentSet(),
        snapshot.step.name,
        execution_scope=FunctionStepExecutionScope.AXIS,
    )

    compiled = planner.artifacts.build_step_compiled_function_pattern(
        snapshot,
        True,
        planner.artifacts.inject_metadata(pattern, declarations.inputs),
        runtime_input_plans,
        {},
        {},
        PathPlannerGroupScope.ungrouped(),
    )

    assert execution_bindings == EMPTY_SOURCE_BINDINGS
    assert execution_group_scope == PathPlannerGroupScope.dynamic(
        AllComponents.CHANNEL
    )
    assert runtime_input_plans == {}
    assert compiled is not None
    (invocation,) = compiled.default_group.invocations
    assert invocation.kwargs_dict["grid_dimensions"] == (2, 3)
    (edge,) = invocation.artifact_input_edges
    assert edge.spec == invocation.contract.artifact_inputs[0]
    assert edge.spec.parameter_name == "grid_dimensions"
    assert edge.storage_plan is None
    assert edge.projection is None


def test_plate_artifact_consumer_omits_inherited_source_plans():
    source_image = ArtifactSpec.input("DNA", ImageArtifactType)
    measurements = ArtifactSpec.input("Measurements", MeasurementsArtifactType)
    export_bundle = ArtifactSpec.output("ExportBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_inputs(source_image, measurements)
    @artifact_outputs(export_bundle)
    def export_measurements(*, artifact_batch: RuntimeArtifactBatch):
        del artifact_batch
        return {"Measurements.csv": b""}

    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name=measurements.name,
        path="/memory/Measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        producer_step_index=2,
        producer_step_scope_id="plate::functionstep_2",
        producer_step_name="Measure",
    ))
    planner.artifact_context = replace(
        planner.artifact_context,
        available_artifacts=ArtifactSpecCollection((measurements,)),
    )
    snapshot = _snapshot(
        func=export_measurements,
        name="PlateExporter",
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(NamedSourceBinding(alias="DNA"),),
        ),
        group_by=GroupBy.NONE,
        variable_components=(),
        input_source=InputSource.PIPELINE_START,
    )
    (
        declarations,
        pattern,
        callable_scope,
        contracts,
    ) = planner.artifacts.prepare_step_declarations(snapshot)
    execution_bindings = planner.artifacts.source_bindings_for_contracts(
        snapshot,
        contracts,
        StepInputDependency.no_main_flow(),
    )
    maps = planner.artifacts.compile_plan_maps(
        snapshot,
        snapshot.index,
        declarations,
        PathPlannerGroupScope.ungrouped(),
        callable_scope,
        execution_bindings,
    )
    compiled = planner.artifacts.build_step_compiled_function_pattern(
        snapshot,
        True,
        pattern,
        maps.inputs,
        maps.outputs,
        maps.relation_source_scopes,
        maps.group_scope,
    )
    plan = planner.plans[snapshot.index]
    plan.step_name = snapshot.step.name
    plan.func = pattern
    plan.main_input_dependency = StepInputDependency.no_main_flow()
    plan.artifact_inputs = maps.inputs
    plan.artifact_outputs = maps.outputs
    plan.compiled_function_pattern = compiled
    plan.source_binding_plan = maps.source_binding_plan
    plan.source_universe_plan = maps.source_universe_plan

    FuncStepContractValidator.validate_compiled_step_plan(plan)

    assert callable_scope is FunctionStepExecutionScope.PLATE
    assert tuple(contracts[0].artifact_inputs) == (source_image, measurements)
    assert tuple(contracts[0].artifact_outputs) == (export_bundle,)
    assert execution_bindings.declares_artifact_ref(source_image.ref())
    assert maps.source_binding_plan.declares_artifact_ref(source_image.ref())
    assert (
        maps.source_universe_plan
        == CompiledSourceUniversePlan.from_source_binding_plan(maps.source_binding_plan)
    )
    assert tuple(maps.inputs) == (measurements.ref(),)
    assert maps.inputs[measurements.ref()].source_step_id == 2
    invocation = next(compiled.iter_invocations())
    assert tuple(edge.spec for edge in invocation.artifact_input_edges) == (
        source_image,
        measurements,
    )
    assert tuple(
        edge.spec for edge in invocation.artifact_input_edges
        if edge.storage_plan is not None
    ) == (measurements,)
    assert (
        invocation.artifact_input_edges[1].storage_plan
        is maps.inputs[measurements.ref()]
    )


def test_plate_artifact_output_plan_is_independent_of_axis_context():
    export_bundle = ArtifactSpec.output("ExportBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(export_bundle)
    def export_measurements(*, artifact_batch: RuntimeArtifactBatch):
        del artifact_batch
        return {"Measurements.csv": b""}

    declarations = extract_artifact_declarations(export_measurements)
    compiled_by_axis = {}
    for axis_id in ("A01", "B01"):
        planner = _artifact_planner_stub()
        planner.ctx.axis_id = axis_id
        output_plans = planner.artifacts.process_artifact_outputs(
            declarations,
            3,
            execution_scope=FunctionStepExecutionScope.PLATE,
            artifact_inputs={},
            source_bindings=EMPTY_SOURCE_BINDINGS,
            variable_components=ComponentSet(),
            step_name="PlateExporter",
        )
        compiled_by_axis[axis_id] = compile_function_pattern(
            export_measurements,
            {},
            output_plans,
        ).default_group.invocations[0]

    assert compiled_by_axis["A01"] == compiled_by_axis["B01"]
    (output_plan,) = compiled_by_axis["A01"].artifact_output_plans
    assert Path(output_plan.path).name == "ExportBundle_step3.pkl"

    axis_planner = _artifact_planner_stub()
    (axis_output_plan,) = axis_planner.artifacts.process_artifact_outputs(
        declarations,
        3,
        execution_scope=FunctionStepExecutionScope.AXIS,
        artifact_inputs={},
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet(),
        step_name="AxisExporter",
    ).values()
    assert Path(axis_output_plan.path).name == "A01_ExportBundle_step3.pkl"


def test_compiled_pattern_rejects_accumulator_owned_output_conflict():
    def identify():
        return None

    crop_mask_measurements = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
        sidecar_role=ArtifactSidecarRole.CROP_MASK,
    )
    image_copy_measurements = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
        sidecar_role=ArtifactSidecarRole.MATERIALIZED_IMAGE_COPY,
    )
    base_contract = CallableContract.from_callable(identify)
    contracts = tuple(
        replace(
            base_contract,
            metadata=replace(
                base_contract.metadata,
                artifact_outputs=(output,),
            ),
        )
        for output in (crop_mask_measurements, image_copy_measurements)
    )

    class RepeatedOutputContractProvider(InvocationContractProvider):
        def __call__(self, invocation, step_context):
            del step_context
            return InvocationContractPlan(contracts[invocation.key.position])

    planner = _artifact_planner_stub()
    planner.invocation_contract_provider = RepeatedOutputContractProvider()
    output_plan = ArtifactOutputPlan(
            name=crop_mask_measurements.name,
            path=f"/memory/{crop_mask_measurements.name}.pkl",
            artifact_type=crop_mask_measurements.artifact_type,
    )
    output_plans = {output_plan.ref(): output_plan}
    pattern = [identify, identify]

    with pytest.raises(
        ValueError,
        match="Conflicting compiled invocation output artifact sidecar role",
    ):
        planner.artifacts.build_step_compiled_function_pattern(
            _snapshot(func=pattern),
            True,
            pattern,
            {},
            output_plans,
            {},
            PathPlannerGroupScope.ungrouped(),
        )


def test_materialization_collision_updates_results_dir_and_config():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plate_path = Path("/data/plate1")
    planner.plans = {
        3: CompiledStepPlan(
            step_index=3,
            step_name="materialize",
            step_type="FunctionStep",
            axis_id="A01",
            materialized_output=MaterializedOutputPlan(
                output_dir=Path("/data/plate1_processed/images"),
                backend="disk",
                plate_root="/data/plate1_processed",
                sub_dir="images",
                analysis_results_dir="/data/plate1_processed/images_results",
            ),
            materialization_config=PathConfigStub(sub_dir="images"),
        )
    }
    snapshot = _snapshot(
        index=3,
        name="materialize",
        step_materialization_config=PathConfigStub(sub_dir="images"),
    )

    planner.paths = PathPlannerPathAuthority(planner)
    planner.validation = PathPlannerValidationStage(planner)

    planner.validation.resolve_and_update_paths(
        snapshot,
        3,
        Path("/data/plate1_processed/images"),
        "main flow",
    )

    assert snapshot.step.step_materialization_config.sub_dir == "images"
    materialized_output = planner.plans[3].materialized_output
    assert materialized_output.output_dir == Path("/data/plate1_processed/images_step3")
    assert materialized_output.sub_dir == "images_step3"
    assert materialized_output.analysis_results_dir == (
        "/data/plate1_processed/images_step3_results"
    )
    assert planner.plans[3].materialization_config.sub_dir == "images_step3"


def test_artifact_output_plans_preserve_declared_kind():
    planner = _artifact_planner_stub()
    output = ArtifactSpec.output("nuclei", ObjectLabelsArtifactType)

    outputs = planner.artifacts.process_artifact_outputs(
        ArtifactGraph(
            producers=(
                ArtifactProducer(
                    spec=output,
                    groups=(None,),
                    invocation_keys=(
                        FunctionInvocationKey("identify", DEFAULT_GROUP_KEY, 0),
                    ),
                ),
            )
        ),
        sid=2,
        output_groups={output.ref(): PathPlannerGroupScope.ungrouped()},
        artifact_inputs={},
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet(),
        step_name="identify",
    execution_scope = FunctionStepExecutionScope.AXIS)

    assert outputs[output.ref()].artifact_type is ObjectLabelsArtifactType
    assert planner.declared[outputs[output.ref()].ref()].artifact_type is (
        ObjectLabelsArtifactType
    )


def test_same_name_typed_artifacts_compile_through_producer_and_consumer_plans():
    planner = _artifact_planner_stub()
    image_output = ArtifactSpec.output("shared", ImageArtifactType)
    labels_output = ArtifactSpec.output("shared", ObjectLabelsArtifactType)

    @artifact_outputs(image_output, labels_output)
    def produce_shared():
        return None

    producer_snapshot = _snapshot(
        index=2,
        name="produce_shared",
        func=produce_shared,
        group_by=GroupBy.NONE,
        variable_components=(),
        input_source=InputSource.PIPELINE_START,
    )
    (
        producer_declarations,
        producer_pattern,
        producer_scope,
        _producer_contracts,
    ) = planner.artifacts.prepare_step_declarations(producer_snapshot)
    producer_maps = planner.artifacts.compile_plan_maps(
        producer_snapshot,
        producer_snapshot.index,
        producer_declarations,
        PathPlannerGroupScope.ungrouped(),
        producer_scope,
        EMPTY_SOURCE_BINDINGS,
    )
    producer_compiled = planner.artifacts.build_step_compiled_function_pattern(
        producer_snapshot,
        True,
        producer_pattern,
        producer_maps.inputs,
        producer_maps.outputs,
        producer_maps.relation_source_scopes,
        producer_maps.group_scope,
    )

    assert tuple(producer_maps.outputs) == (
        image_output.ref(),
        labels_output.ref(),
    )
    producer_paths = {
        ref: plan.path for ref, plan in producer_maps.outputs.items()
    }
    assert Path(producer_paths[image_output.ref()]).name == (
        "A01_shared__image_step2.pkl"
    )
    assert Path(producer_paths[labels_output.ref()]).name == (
        "A01_shared__object_labels_step2.pkl"
    )
    assert len(set(producer_paths.values())) == 2
    assert tuple(
        plan.ref()
        for plan in next(producer_compiled.iter_invocations()).artifact_output_plans
    ) == (image_output.ref(), labels_output.ref())

    image_input = ArtifactSpec.input(
        "shared",
        ImageArtifactType,
        parameter_name="shared_image",
    )
    labels_input = ArtifactSpec.input(
        "shared",
        ObjectLabelsArtifactType,
        parameter_name="shared_labels",
    )

    @artifact_inputs(image_input, labels_input)
    def consume_shared(image, shared_image, shared_labels):
        del shared_image, shared_labels
        return image

    consumer_snapshot = _snapshot(
        index=3,
        name="consume_shared",
        func=consume_shared,
        group_by=GroupBy.NONE,
        variable_components=(),
    )
    (
        consumer_declarations,
        consumer_pattern,
        consumer_scope,
        _consumer_contracts,
    ) = planner.artifacts.prepare_step_declarations(consumer_snapshot)
    consumer_maps = planner.artifacts.compile_plan_maps(
        consumer_snapshot,
        consumer_snapshot.index,
        consumer_declarations,
        PathPlannerGroupScope.ungrouped(),
        consumer_scope,
        EMPTY_SOURCE_BINDINGS,
    )
    consumer_compiled = planner.artifacts.build_step_compiled_function_pattern(
        consumer_snapshot,
        True,
        consumer_pattern,
        consumer_maps.inputs,
        consumer_maps.outputs,
        consumer_maps.relation_source_scopes,
        consumer_maps.group_scope,
    )

    assert tuple(consumer_maps.inputs) == (image_input.ref(), labels_input.ref())
    assert {
        input_ref.for_plan_type(ArtifactOutputPlan): input_plan.path
        for input_ref, input_plan in consumer_maps.inputs.items()
    } == producer_paths
    consumer_edges = next(consumer_compiled.iter_invocations()).artifact_input_edges
    assert tuple(edge.spec.ref() for edge in consumer_edges) == (
        image_input.ref(),
        labels_input.ref(),
    )
    assert tuple(edge.storage_plan.path for edge in consumer_edges) == tuple(
        producer_paths[output_ref]
        for output_ref in (image_output.ref(), labels_output.ref())
    )


def test_derived_artifact_storage_keys_reject_declared_name_collision():
    outputs = (
        ArtifactSpec.output("shared", ImageArtifactType),
        ArtifactSpec.output("shared", ObjectLabelsArtifactType),
        ArtifactSpec.output("shared__image", SpecialArtifactType),
    )
    declarations = ArtifactGraph(
        producers=tuple(
            ArtifactProducer(
                spec=spec,
                groups=(None,),
                invocation_keys=(),
            )
            for spec in outputs
        )
    )

    with pytest.raises(ValueError, match="conflicting storage keys"):
        declarations.output_storage_keys()


def test_artifact_output_plan_only_preserves_explicit_source_stack_scope():
    planner = _artifact_planner_stub()
    source = ArtifactSpec.input("source", ImageArtifactType)
    output_specs = (
        ArtifactSpec.output_inheriting_group_scope(
            "group_lineage",
            ImageArtifactType,
            source,
        ),
        ArtifactSpec.output_preserving_source_stack_scope(
            "stack_lineage",
            ImageArtifactType,
            source,
        ),
    )

    outputs = planner.artifacts.process_artifact_outputs(
        ArtifactGraph(
            producers=tuple(
                ArtifactProducer(
                    spec=spec,
                    groups=(None,),
                    invocation_keys=(
                        FunctionInvocationKey("transform", DEFAULT_GROUP_KEY, 0),
                    ),
                )
                for spec in output_specs
            ),
            consumers=(
                ArtifactConsumer(
                    spec=source,
                    invocation_keys=(
                        FunctionInvocationKey("transform", DEFAULT_GROUP_KEY, 0),
                    ),
                ),
            ),
        ),
        sid=2,
        artifact_inputs={
            source.ref(): ArtifactInputPlan(
                name=source.name,
                path="/memory/source.pkl",
                artifact_type=source.artifact_type,
                variable_components=(AllComponents.CHANNEL,),
            )
        },
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet((AllComponents.CHANNEL,)),
        step_name="transform",
    execution_scope = FunctionStepExecutionScope.AXIS)

    assert outputs[output_specs[0].ref()].variable_components == ()
    assert outputs[output_specs[1].ref()].variable_components == (
        AllComponents.CHANNEL,
    )


def test_artifact_output_source_lookup_combines_repeated_main_flow_inputs():
    planner = _artifact_planner_stub()
    source = ArtifactSpec.input("MembFinal", ImageArtifactType)
    output = ArtifactSpec.output_preserving_source_stack_scope(
        "Cells",
        ObjectLabelsArtifactType,
        source,
    )
    invocation_key = FunctionInvocationKey("watershed", DEFAULT_GROUP_KEY, 0)
    repeated_source = ArtifactConsumer(
        spec=source,
        invocation_keys=(invocation_key,),
    )
    planner.artifact_context = ArtifactDeclarationStepContext(
        available_artifacts=ArtifactSpecCollection((source,)),
        main_flow_artifacts=ArtifactSpecCollection((source,)),
    )

    outputs = planner.artifacts.process_artifact_outputs(
        ArtifactGraph(
            producers=(
                ArtifactProducer(
                    spec=output,
                    groups=(None,),
                    invocation_keys=(invocation_key,),
                ),
            ),
            non_plan_consumers=(repeated_source, repeated_source),
        ),
        sid=2,
        artifact_inputs={},
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet((AllComponents.Z_INDEX,)),
        step_name="watershed",
    execution_scope = FunctionStepExecutionScope.AXIS)

    assert outputs[output.ref()].variable_components == (AllComponents.Z_INDEX,)


def test_compiled_source_edges_only_consume_relation_owned_main_flow():
    source_specs = tuple(
        ArtifactSpec.input(name, ImageArtifactType)
        for name in ("DNA", "Membrane", "Mitochondria")
    )
    output_spec = ArtifactSpec.output_preserving_source_stack_scope(
        "Combined",
        ImageArtifactType,
        source_specs[0],
    )
    output_plan = ArtifactOutputPlan(
        name=output_spec.name,
        path="/memory/Combined.pkl",
        artifact_type=output_spec.artifact_type,
        relations=output_spec.relations,
    )

    @artifact_inputs(*source_specs)
    @artifact_outputs(output_spec)
    def combine_sources(image):
        return image

    compiled = compile_function_pattern(
        combine_sources,
        {},
        {plan.ref(): plan for plan in (output_plan,)},
    )
    compiled = _artifact_planner_stub().artifacts.compile_invocation_input_edges(
        compiled,
        artifact_inputs={},
        relation_source_scopes={},
        execution_group_scope=PathPlannerGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet((AllComponents.Z_INDEX,)),
        main_flow_artifacts=ArtifactSpecCollection(source_specs),
    )

    edges = next(compiled.iter_invocations()).artifact_input_edges
    assert tuple(edge.spec for edge in edges) == source_specs
    assert tuple(edge.consumes_main_flow for edge in edges) == (True, False, False)


def test_implicit_native_main_flow_provenance_drives_artifact_owned_scope():
    from openhcs.core.config import (
        GlobalPipelineConfig,
        LazyProcessingConfig,
        PipelineConfig,
    )
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.core.function_patterns import normalize_function_pattern
    from openhcs.core.pipeline.compilation_session import CompilationSession
    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProviderFactory,
    )
    from openhcs.processing.backends.cellprofiler.thresholding import threshold
    from openhcs.processing.backends.processors.numpy_processor import (
        percentile_normalize,
    )

    processing_config = LazyProcessingConfig(
        variable_components=[VariableComponents.SITE],
        group_by=GroupBy.CHANNEL,
        input_source=InputSource.PREVIOUS_STEP,
    )
    steps = (
        FunctionStep(
            func=percentile_normalize,
            name="percentile_normalize",
            processing_config=processing_config,
        ),
        FunctionStep(
            func=(threshold, {"name_the_output_image": "Thresholded"}),
            name="Threshold",
            processing_config=processing_config,
        ),
    )
    snapshots = tuple(
        StepSnapshot(
            index=index,
            scope_id=f"implicit-main-flow::functionstep_{index}",
            step=step,
        )
        for index, step in enumerate(steps)
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                index: CompiledStepPlan(
                    step_index=index,
                    step_name=step.name,
                    step_type=step.__class__.__name__,
                    axis_id="A01",
                )
                for index, step in enumerate(steps)
            },
            axis_id="A01",
        ),
        steps=steps,
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={index: object() for index in range(len(steps))},
        snapshots=snapshots,
    )
    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )
    assert provider is not None

    planner = _artifact_planner_stub()
    planner.artifact_context = ArtifactDeclarationStepContext(
        step_name="percentile_normalize",
        step_index=0,
    )
    channel_scope = PathPlannerGroupScope.from_raw(
        ("1", "2", "4"),
        component=AllComponents.CHANNEL,
    )
    native_pattern = compile_function_pattern(percentile_normalize, {}, {})
    native_invocation = next(native_pattern.iter_invocations())
    planner.artifact_context = (
        planner.artifacts.advance_artifact_context_after_compiled_pattern(
            ArtifactGraph.empty(),
            native_pattern,
            channel_scope,
        )
    )
    cursor_name = unnamed_main_flow_artifact_name(0, native_invocation.key)
    cursor = ArtifactSpec.input(cursor_name, ImageArtifactType)
    consumer_context = replace(
        planner.artifact_context,
        step_name=steps[1].name,
        step_index=1,
        group_by=GroupBy.CHANNEL,
        input_source=InputSource.PREVIOUS_STEP,
    )
    planner.artifact_context = consumer_context
    consumer_invocation = next(
        normalize_function_pattern(steps[1].func).iter_items()
    )
    consumer_plan = provider(consumer_invocation, consumer_context)
    assert consumer_plan is not None
    consumer_contract = consumer_plan.contract
    assert consumer_contract.artifact_inputs.names() == (cursor_name,)
    assert consumer_contract.group_scope_inputs.names() == (cursor_name,)
    consumer_graph = extract_artifact_declarations(
        steps[1].func,
        invocation_contract_provider=provider,
        step_context=consumer_context,
    )

    assert planner.artifact_context.main_flow_artifacts == ArtifactSpecCollection(
        (cursor,)
    )
    assert planner.artifact_context.available_artifact_producer_for(cursor) == (
        ArtifactProducer(
            spec=cursor.for_plan_type(ArtifactOutputPlan),
            groups=("1", "2", "4"),
            invocation_keys=(native_invocation.key,),
            producer_step_index=0,
        )
    )
    assert planner.declared == {}

    execution_scope = planner.execution_groups.get_execution_groups(
        snapshots[1],
        PathPlannerComponentScopes.empty(),
        contracts=(consumer_contract,),
    )
    assert execution_scope == channel_scope

    compiled_inputs = planner.artifacts.process_artifact_inputs(
        consumer_graph,
        sid=1,
        consumer_scope=execution_scope,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet((AllComponents.SITE,)),
        step_name="Threshold",
        execution_scope=FunctionStepExecutionScope.AXIS,
    )
    assert compiled_inputs == {}

    compiled_consumer = compile_function_pattern(
        steps[1].func,
        {},
        {},
        invocation_contract_provider=provider,
        step_context=consumer_context,
    )
    compiled_consumer = planner.artifacts.compile_invocation_input_edges(
        compiled_consumer,
        artifact_inputs=compiled_inputs,
        relation_source_scopes={},
        execution_group_scope=execution_scope,
        consumer_variable_components=ComponentSet((AllComponents.SITE,)),
        main_flow_artifacts=planner.artifact_context.main_flow_artifacts,
    )
    edge = next(compiled_consumer.iter_invocations()).artifact_input_edges[0]
    assert edge.spec == cursor
    assert edge.storage_plan is None
    assert edge.consumes_main_flow


def test_artifact_output_source_uses_compiled_plan_across_parameter_occurrences():
    planner = _artifact_planner_stub()
    measured = ArtifactSpec.input(
        "Cells",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    neighbors = replace(measured, parameter_name="neighbor_labels")
    output = ArtifactSpec.output_preserving_source_stack_scope(
        "Neighbors",
        RelationshipsArtifactType,
        measured,
    )
    invocation_key = FunctionInvocationKey(
        "measure_object_neighbors",
        DEFAULT_GROUP_KEY,
        0,
    )

    outputs = planner.artifacts.process_artifact_outputs(
        ArtifactGraph(
            producers=(
                ArtifactProducer(
                    spec=output,
                    groups=(None,),
                    invocation_keys=(invocation_key,),
                ),
            ),
            consumers=(
                ArtifactConsumer(measured, (invocation_key,)),
                ArtifactConsumer(neighbors, (invocation_key,)),
            ),
        ),
        sid=2,
        artifact_inputs={
            measured.ref(): ArtifactInputPlan(
                name=measured.name,
                path="/memory/Cells.pkl",
                artifact_type=measured.artifact_type,
                variable_components=(AllComponents.Z_INDEX,),
            )
        },
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet((AllComponents.Z_INDEX,)),
        step_name="MeasureObjectNeighbors",
        execution_scope=FunctionStepExecutionScope.AXIS,
    )

    assert outputs[output.ref()].variable_components == (AllComponents.Z_INDEX,)
    assert outputs[output.ref()].relations == output.relations


def test_artifact_output_source_lookup_ignores_shared_input_broadcast_projections():
    planner = _artifact_planner_stub()
    green = ArtifactSpec.input("OrigGreen", ImageArtifactType)
    red = ArtifactSpec.input("OrigRed", ImageArtifactType)
    mask_name = ArtifactSidecarRole.CROP_MASK.name_for("CropBlue")
    green_mask = ArtifactSpec.input(
        mask_name,
        ImageArtifactType,
        sidecar_role=ArtifactSidecarRole.CROP_MASK,
        relations=(InputStackBroadcastSourceRelation(source=green.ref()),),
    )
    red_mask = replace(
        green_mask,
        relations=(InputStackBroadcastSourceRelation(source=red.ref()),),
    )
    green_output = ArtifactSpec.output_preserving_source_stack_scope(
        "CropGreen",
        ImageArtifactType,
        green,
    )
    red_output = ArtifactSpec.output_preserving_source_stack_scope(
        "CropRed",
        ImageArtifactType,
        red,
    )
    green_key = FunctionInvocationKey("crop", "2", 0)
    red_key = FunctionInvocationKey("crop", "3", 0)
    consumers = (
        ArtifactConsumer(green, (green_key,)),
        ArtifactConsumer(green_mask, (green_key,)),
        ArtifactConsumer(red, (red_key,)),
        ArtifactConsumer(red_mask, (red_key,)),
    )
    artifact_inputs = {
        green.ref(): ArtifactInputPlan(
            name=green.name,
            path="/memory/OrigGreen.pkl",
            artifact_type=ImageArtifactType,
            group_keys=("2",),
            group_component=AllComponents.CHANNEL,
            variable_components=(AllComponents.SITE,),
        ),
        red.ref(): ArtifactInputPlan(
            name=red.name,
            path="/memory/OrigRed.pkl",
            artifact_type=ImageArtifactType,
            group_keys=("3",),
            group_component=AllComponents.CHANNEL,
            variable_components=(AllComponents.SITE,),
        ),
        green_mask.ref(): ArtifactInputPlan(
            name=mask_name,
            path="/memory/CropBlue__crop_mask.pkl",
            artifact_type=ImageArtifactType,
            sidecar_role=ArtifactSidecarRole.CROP_MASK,
            group_keys=("1",),
            group_component=AllComponents.CHANNEL,
            variable_components=(AllComponents.SITE,),
        ),
    }

    outputs = planner.artifacts.process_artifact_outputs(
        ArtifactGraph(
            producers=(
                ArtifactProducer(
                    spec=green_output,
                    groups=("2",),
                    invocation_keys=(green_key,),
                ),
                ArtifactProducer(
                    spec=red_output,
                    groups=("3",),
                    invocation_keys=(red_key,),
                ),
            ),
            consumers=consumers,
        ),
        sid=2,
        artifact_inputs=artifact_inputs,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet((AllComponents.SITE,)),
        step_name="Crop",
    execution_scope = FunctionStepExecutionScope.AXIS)

    for output_spec, source in (
        (green_output, green),
        (red_output, red),
    ):
        output_plan = outputs[output_spec.ref()]
        assert output_plan.source_context_source() == source.ref()
        assert output_plan.group_scope_sources() == (source.ref(),)
        assert output_plan.materialization_source() is None

    ambiguous_output = ArtifactSpec.output_preserving_source_stack_scope(
        "ProjectedCropMask",
        ImageArtifactType,
        green_mask,
    )
    projected = planner.artifacts.process_artifact_outputs(
        ArtifactGraph(
            producers=(
                ArtifactProducer(
                    spec=ambiguous_output,
                    groups=("2",),
                    invocation_keys=(green_key,),
                ),
            ),
            consumers=consumers,
        ),
        sid=2,
        artifact_inputs=artifact_inputs,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet((AllComponents.SITE,)),
        step_name="Crop",
        execution_scope=FunctionStepExecutionScope.AXIS,
    )

    assert projected[ambiguous_output.ref()].variable_components == (
        AllComponents.SITE,
    )


def test_enabled_source_bindings_preserve_previous_step_main_flow():
    input_image = ArtifactSpec.input("Input", ImageArtifactType)
    measurements = ArtifactSpec.input("Measurements", MeasurementsArtifactType)
    context = ArtifactDeclarationStepContext(
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(NamedSourceBinding(alias="Input"),),
        ),
        input_source=InputSource.PREVIOUS_STEP,
        main_flow_artifacts=ArtifactSpecCollection(
            (ArtifactSpec.input("Previous", ImageArtifactType),)
        ),
    ).with_source_declarations((input_image, measurements))

    assert context.available_artifacts.names() == ("Input", "Measurements")
    assert context.main_flow_artifacts.names() == ("Previous",)


def test_artifact_lineage_projects_exact_source_binding_component():
    source_binding_spec = ArtifactSpec.input("OrigStain1", ImageArtifactType)
    aligned_output = ArtifactSpec.output_preserving_source_stack_scope(
        "Stain1",
        ImageArtifactType,
        source_binding_spec,
    )
    aligned_input = aligned_output.for_plan_type(ArtifactInputPlan)
    object_output = ArtifactSpec.output_preserving_source_stack_scope(
        "Objects1",
        ObjectLabelsArtifactType,
        aligned_input,
    )

    @artifact_inputs(aligned_input)
    @artifact_outputs(object_output)
    def identify(main_image, Stain1):
        del Stain1
        return main_image

    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias=source_binding_spec.name,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
        ),
    )
    planner = _artifact_planner_stub()
    planner.artifact_context = (
        ArtifactDeclarationStepContext(source_bindings=source_bindings)
        .with_source_declarations((source_binding_spec,))
        .advance_artifact_graph(
            ArtifactGraph(
                producers=(
                    ArtifactProducer(
                        spec=aligned_output,
                        groups=("1", "2"),
                        invocation_keys=(
                            FunctionInvocationKey("align", DEFAULT_GROUP_KEY, 0),
                        ),
                    ),
                ),
            ),
            main_flow_artifacts=ArtifactSpecCollection((aligned_input,)),
        )
    )
    _record_declared_output(planner, ArtifactOutputPlan(
        name=aligned_input.name,
        path="/memory/Stain1.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        variable_components=(AllComponents.CHANNEL,),
        paths_by_group={
            "1": "/memory/Stain1_site_1.pkl",
            "2": "/memory/Stain1_site_2.pkl",
        },
        relations=aligned_output.relations,
        producer_step_index=2,
        producer_step_name="Align",
    ))
    snapshot = _snapshot(
        name="IdentifyPrimaryObjects",
        func=identify,
        source_bindings=source_bindings,
    )
    declarations = extract_artifact_declarations(identify)
    execution_scope = planner.execution_groups.get_execution_groups(
        snapshot,
        PathPlannerComponentScopes.empty(),
        source_bindings=source_bindings,
    )
    declarations = planner.artifacts.namespace_grouped_outputs_for_runtime_consumers(
        identify,
        declarations,
        execution_scope,
    )

    maps = planner.artifacts.compile_plan_maps(
        snapshot,
        3,
        declarations,
        execution_scope,
        source_bindings=source_bindings,
    )
    compiled = planner.artifacts.build_step_compiled_function_pattern(
        snapshot,
        True,
        identify,
        maps.inputs,
        maps.outputs,
        maps.relation_source_scopes,
        maps.group_scope,
    )

    expected_channel_scope = ComponentGroupScope.from_raw(
        ("1",),
        component=AllComponents.CHANNEL,
    )
    assert maps.group_scope.keys == expected_channel_scope.keys
    assert maps.group_scope.component is expected_channel_scope.component
    assert maps.outputs[object_output.ref()].group_keys == ("1",)
    assert maps.outputs[object_output.ref()].variable_components == (AllComponents.SITE,)
    edge = next(compiled.iter_invocations()).artifact_input_edges[0]
    assert edge.projection.invocation_scope == expected_channel_scope
    assert edge.projection.component_scope(AllComponents.CHANNEL) == (
        expected_channel_scope
    )
    assert edge.projection.producer_selection_scope == ComponentGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.SITE,
    )


@pytest.mark.parametrize(
    ("artifact_type", "output_name"),
    (
        (ImageArtifactType, "CorrProtein"),
        (ObjectLabelsArtifactType, "AdvancedObjects"),
    ),
    ids=("produced-image", "produced-object-labels"),
)
def test_fixed_source_component_domain_survives_produced_artifact_lineage(
    artifact_type,
    output_name,
):
    source = ArtifactSpec.input("OrigProtein", ImageArtifactType)
    output_spec = ArtifactSpec.output_preserving_source_stack_scope(
        output_name,
        artifact_type,
        source,
    )
    invocation_key = FunctionInvocationKey("produce", DEFAULT_GROUP_KEY, 0)
    producer_declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=output_spec,
                groups=(None,),
                invocation_keys=(invocation_key,),
            ),
        ),
        non_plan_consumers=(ArtifactConsumer(source, (invocation_key,)),),
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias=source.name,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
        ),
    )
    planner = _artifact_planner_stub()
    planner.artifact_context = ArtifactDeclarationStepContext(
        source_bindings=source_bindings,
    ).with_source_declarations((source,))
    producer_scope = PathPlannerGroupScope.from_raw(
        ("1",),
        component=AllComponents.SITE,
    )
    relation_scopes = planner.artifacts.relation_source_scopes_by_ref(
        producer_declarations,
        {},
        group_scope=producer_scope,
        source_bindings=source_bindings,
        group_by=GroupBy.SITE,
    )
    output_groups = planner.artifacts.output_groups_from_declared_relations(
        producer_declarations,
        group_scope=producer_scope,
        relation_source_scopes=relation_scopes,
        consumer_variable_components=ComponentSet((AllComponents.CHANNEL,)),
        step_index=2,
        step_name="produce",
    )
    output_plan = planner.artifacts.process_artifact_outputs(
        producer_declarations,
        2,
        output_groups,
        artifact_inputs={},
        relation_source_scopes=relation_scopes,
        source_bindings=source_bindings,
        variable_components=ComponentSet((AllComponents.CHANNEL,)),
        step_name="produce",
    execution_scope = FunctionStepExecutionScope.AXIS)[output_spec.ref()]

    fixed_channel = ComponentGroupScope.from_raw(
        ("1",),
        component=AllComponents.CHANNEL,
    )
    assert output_plan.component_domain(AllComponents.CHANNEL) == fixed_channel

    input_spec = output_spec.for_plan_type(ArtifactInputPlan)

    @artifact_inputs(input_spec)
    def consume(image):
        return image

    consumer_declarations = ArtifactGraph(
        consumers=(
            ArtifactConsumer(
                input_spec,
                (FunctionInvocationKey("consume", "2", 0),),
            ),
        ),
    )
    consumer_scope = PathPlannerGroupScope.from_raw(
        ("2",),
        component=AllComponents.CHANNEL,
    )
    input_plan = planner.artifacts.process_artifact_inputs(
        consumer_declarations,
        3,
        consumer_scope,
        EMPTY_SOURCE_BINDINGS,
        ComponentSet((AllComponents.SITE,)),
        step_name="consume",
    execution_scope = FunctionStepExecutionScope.AXIS)[input_spec.ref()]
    consumer_relation_scopes = planner.artifacts.relation_source_scopes_by_ref(
        consumer_declarations,
        {input_plan.ref(): input_plan},
        group_scope=consumer_scope,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        group_by=GroupBy.CHANNEL,
    )
    compiled = planner.artifacts.compile_invocation_input_edges(
        compile_function_pattern(
            {"2": consume},
            {plan.ref(): plan for plan in (input_plan,)},
            {},
        ),
        artifact_inputs={input_plan.ref(): input_plan},
        relation_source_scopes=consumer_relation_scopes,
        execution_group_scope=consumer_scope,
        consumer_variable_components=ComponentSet((AllComponents.SITE,)),
    )
    edge = next(compiled.iter_invocations()).artifact_input_edges[0]

    assert input_plan.component_domain(AllComponents.CHANNEL) == fixed_channel
    assert consumer_relation_scopes[input_spec.ref()] == (
        PathPlannerGroupScope.from_raw(
            fixed_channel.keys,
            component=fixed_channel.component,
        )
    )
    assert edge.projection.component_scope(AllComponents.CHANNEL) == fixed_channel


@pytest.mark.parametrize(
    ("step_name", "consumer_axes", "source_axes"),
    (
        (
            "Crop",
            (AllComponents.SITE,),
            ((), (AllComponents.SITE,)),
        ),
        (
            "Align",
            (AllComponents.CHANNEL,),
            ((AllComponents.CHANNEL,), (), ()),
        ),
    ),
    ids=("crop-site-and-scalar-sources", "align-channel-and-scalar-sources"),
)
def test_measurement_provenance_does_not_preserve_source_stack_axes(
    step_name,
    consumer_axes,
    source_axes,
):
    planner = _artifact_planner_stub()
    sources = tuple(
        ArtifactSpec.input(f"source_{index}", ImageArtifactType)
        for index in range(len(source_axes))
    )
    measurement = ArtifactSpec.output(
        f"{step_name}_measurements",
        MeasurementsArtifactType,
        relations=tuple(
            ArtifactSpecRelation(source=source.ref()) for source in sources
        ),
    )
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=measurement,
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        step_name.lower(),
                        DEFAULT_GROUP_KEY,
                        0,
                    ),
                ),
            ),
        ),
    )
    artifact_inputs = {
        source.ref(): ArtifactInputPlan(
            name=source.name,
            path=f"/memory/{source.name}.pkl",
            artifact_type=source.artifact_type,
            variable_components=axes,
        )
        for source, axes in zip(sources, source_axes, strict=True)
    }

    outputs = planner.artifacts.process_artifact_outputs(
        declarations,
        sid=2,
        artifact_inputs=artifact_inputs,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet(consumer_axes),
        step_name=step_name,
    execution_scope = FunctionStepExecutionScope.AXIS)

    assert outputs[measurement.ref()].variable_components == ()
    assert all(
        relation.source_stack_scope_source() is None
        for relation in outputs[measurement.ref()].relations
    )
    assert {relation.source for relation in outputs[measurement.ref()].relations} == {
        source.ref() for source in sources
    }


def test_group_by_namespaces_compiler_owned_outputs():
    @artifact_outputs(ArtifactSpec.output("nuclei", ObjectLabelsArtifactType))
    def identify(image):
        return image

    planner = _artifact_planner_stub()
    declarations = extract_artifact_declarations(identify)

    namespaced = planner.artifacts.namespace_grouped_outputs_for_runtime_consumers(
        identify,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
    )

    assert namespaced.output_groups[
        ArtifactSpec.output("nuclei", ObjectLabelsArtifactType).ref()
    ] == {"1", "2"}


def test_artifact_graph_preserves_same_name_outputs_of_different_types():
    image_output = ArtifactSpec.output("shared", ImageArtifactType)
    object_output = ArtifactSpec.output("shared", ObjectLabelsArtifactType)

    @artifact_outputs(image_output, object_output)
    def produce(image):
        return image, image

    declarations = extract_artifact_declarations(produce)

    assert tuple(declarations.outputs) == (
        image_output.ref(),
        object_output.ref(),
    )


def test_group_by_namespaces_runtime_adapter_artifact_outputs():
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def correct_illumination(image, *, runtime):
        return image

    def declarations_for_invocation(invocation, step_context):
        del invocation, step_context

        @artifact_outputs(ArtifactSpec.output("Hoechst", ImageArtifactType))
        def declared_artifact_owner(image):
            return image

        return CallableContract.from_callable(declared_artifact_owner)

    planner = _artifact_planner_stub()
    declarations = extract_artifact_declarations(
        correct_illumination,
        declaration_provider=declarations_for_invocation,
    )

    namespaced = planner.artifacts.namespace_grouped_outputs_for_runtime_consumers(
        correct_illumination,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
    )

    output_ref = ArtifactSpec.output("Hoechst", ImageArtifactType).ref()
    assert declarations.output_groups[output_ref] == {None}
    assert namespaced.output_groups[output_ref] == {"1", "2"}


def test_declared_group_lineage_scopes_outputs_without_rewriting_execution():
    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="Tile_of_grid",
        path="/memory/Tile_of_grid.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/Tile_of_grid_1.pkl"},
    ))
    source = ArtifactSpec.input("Tile_of_grid", ObjectLabelsArtifactType).ref()
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "Filtered_tiles",
                    ObjectLabelsArtifactType,
                    source,
                ),
                groups=("2",),
                invocation_keys=(),
            ),
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "FilterObjects_8_measurements",
                    MeasurementsArtifactType,
                    source,
                ),
                groups=("2",),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input("Tile_of_grid", ObjectLabelsArtifactType),
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="FilterObjects"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(("2",), component=AllComponents.CHANNEL),
    )

    filtered_tiles_ref = ArtifactSpec.output(
        "Filtered_tiles",
        ObjectLabelsArtifactType,
    ).ref()
    measurements_ref = ArtifactSpec.output(
        "FilterObjects_8_measurements",
        MeasurementsArtifactType,
    ).ref()
    assert maps.outputs[filtered_tiles_ref].group_keys == ("1",)
    assert maps.outputs[measurements_ref].group_keys == ("1",)
    assert maps.outputs[filtered_tiles_ref].group_component is AllComponents.CHANNEL
    assert (
        maps.outputs[measurements_ref].group_component
        is AllComponents.CHANNEL
    )
    assert maps.group_scope == PathPlannerGroupScope.from_raw(
        ("2",),
        component=AllComponents.CHANNEL,
    )


def test_declared_group_lineage_uses_main_flow_scope_without_artifact_plan():
    planner = _artifact_planner_stub()
    source = ArtifactSpec.input("Stain1", ImageArtifactType)
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "MeasureColocalization_4_measurements",
                    MeasurementsArtifactType,
                    source.ref(),
                ),
                groups=(None,),
                invocation_keys=(),
            ),
        ),
        non_plan_consumers=(
            ArtifactConsumer(
                spec=source,
                invocation_keys=(),
            ),
        ),
    )
    group_scope = PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.SITE,
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="MeasureColocalization"),
        3,
        declarations,
        group_scope,
    )

    assert maps.inputs == {}
    measurements_ref = ArtifactSpec.output(
        "MeasureColocalization_4_measurements",
        MeasurementsArtifactType,
    ).ref()
    assert maps.outputs[measurements_ref].group_keys == (
        "1",
        "2",
    )
    assert (
        maps.outputs[measurements_ref].group_component
        is AllComponents.SITE
    )


def test_prior_main_flow_artifact_scopes_output_without_rewriting_execution():
    planner = _artifact_planner_stub()
    producer = ArtifactOutputPlan(
        name="CropRed",
        path="/memory/CropRed.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": "/memory/CropRed_3.pkl"},
    )
    _record_declared_output(planner, producer)
    source = ArtifactSpec.input("CropRed", ImageArtifactType)
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "Nuclei",
                    ObjectLabelsArtifactType,
                    source.ref(),
                ),
                groups=("2", "3"),
                invocation_keys=(),
            ),
        ),
        non_plan_consumers=(
            ArtifactConsumer(
                spec=source,
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="IdentifyPrimaryObjects"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("2", "3"),
            component=AllComponents.CHANNEL,
        ),
    )

    assert maps.inputs == {}
    assert maps.group_scope == PathPlannerGroupScope.from_raw(
        ("2", "3"),
        component=AllComponents.CHANNEL,
    )
    nuclei_ref = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType).ref()
    assert maps.outputs[nuclei_ref].group_keys == ("3",)


def test_dict_invocation_lineage_uses_its_non_plan_input_group_scope():
    planner = _artifact_planner_stub()
    source = ArtifactSpec.input("Stain1", ImageArtifactType)
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "Objects1",
                    ObjectLabelsArtifactType,
                    source.ref(),
                ),
                groups=("1",),
                invocation_keys=(FunctionInvocationKey("identify", "1", 0),),
            ),
        ),
        non_plan_consumers=(
            ArtifactConsumer(
                spec=source,
                invocation_keys=(FunctionInvocationKey("identify", "1", 0),),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="IdentifyPrimaryObjects"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
    )

    objects_ref = ArtifactSpec.output("Objects1", ObjectLabelsArtifactType).ref()
    assert maps.outputs[objects_ref].group_keys == ("1",)


def test_measurement_output_scope_compiles_exact_cross_group_consumer_edge():
    planner = _artifact_planner_stub()
    planner.ctx.microscope_handler = SimpleNamespace(
        can_resolve_metadata_artifact=lambda artifact_name: artifact_name == "DF_image",
    )
    _record_declared_output(planner, ArtifactOutputPlan(
        name="Tile_of_grid",
        path="/memory/Tile_of_grid.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/Tile_of_grid_1.pkl"},
    ))
    image_ref = ArtifactSpec.input("DF_image", ImageArtifactType).ref()
    object_ref = ArtifactSpec.input(
        "Tile_of_grid",
        ObjectLabelsArtifactType,
    ).ref()
    invocation_key = FunctionInvocationKey("measure_object_intensity", "2", 0)
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output(
                    "MeasureObjectIntensity_3_measurements",
                    MeasurementsArtifactType,
                    relations=(
                        ArtifactSpecRelation(image_ref),
                        ArtifactSpecRelation(object_ref),
                        GroupLineageSourceRelation(image_ref),
                    ),
                ),
                groups=("2",),
                invocation_keys=(invocation_key,),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input(
                    "Tile_of_grid",
                    ObjectLabelsArtifactType,
                    relations=(InputGroupLineageSourceRelation(source=image_ref),),
                ),
                invocation_keys=(invocation_key,),
            ),
        ),
        non_plan_consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input("DF_image", ImageArtifactType),
                invocation_keys=(invocation_key,),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="MeasureObjectIntensity"),
        2,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("2",),
            component=AllComponents.CHANNEL,
        ),
    )

    assert maps.group_scope == PathPlannerGroupScope.from_raw(
        ("2",),
        component=AllComponents.CHANNEL,
    )
    measurement_name = "MeasureObjectIntensity_3_measurements"
    measurement_ref = ArtifactSpec.output(
        measurement_name,
        MeasurementsArtifactType,
    ).ref()
    assert maps.outputs[measurement_ref].group_keys == ("2",)
    _record_declared_output(planner, maps.outputs[measurement_ref])
    measurement_input = ArtifactSpec.input(
        measurement_name,
        MeasurementsArtifactType,
    )

    @artifact_inputs(measurement_input)
    def filter_objects(main_image, MeasureObjectIntensity_3_measurements):
        del MeasureObjectIntensity_3_measurements
        return main_image

    consumer_invocation = FunctionInvocationKey(
        "filter_objects",
        DEFAULT_GROUP_KEY,
        0,
    )
    consumer_snapshot = _snapshot(
        name="FilterObjects",
        func=filter_objects,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
    )
    consumer_maps = planner.artifacts.compile_plan_maps(
        consumer_snapshot,
        3,
        ArtifactGraph(
            consumers=(
                ArtifactConsumer(
                    spec=measurement_input,
                    invocation_keys=(consumer_invocation,),
                ),
            ),
        ),
        PathPlannerGroupScope.from_raw(
            ("1",),
            component=AllComponents.CHANNEL,
        ),
    )
    compiled = planner.artifacts.build_step_compiled_function_pattern(
        consumer_snapshot,
        True,
        filter_objects,
        consumer_maps.inputs,
        consumer_maps.outputs,
        consumer_maps.relation_source_scopes,
        consumer_maps.group_scope,
    )

    edge = next(compiled.iter_invocations()).artifact_input_edges[0]
    assert edge.projection.producer_selection_scope == ComponentGroupScope(
        ("2",),
        component=AllComponents.CHANNEL,
    )


def test_artifact_managed_lineage_keeps_exact_named_inputs_in_one_invocation():
    planner = _artifact_planner_stub()
    producer_declarations = (
        ("CropBlue", ImageArtifactType, "1"),
        ("Nuclei", ObjectLabelsArtifactType, "1"),
        ("Cells", ObjectLabelsArtifactType, "2"),
        ("Cytoplasm", ObjectLabelsArtifactType, "2"),
    )
    for name, artifact_type, channel in producer_declarations:
        _record_declared_output(planner, ArtifactOutputPlan(
            name=name,
            path=f"/memory/{name}.pkl",
            artifact_type=artifact_type,
            group_keys=(channel,),
            group_component=AllComponents.CHANNEL,
            variable_components=(AllComponents.SITE,),
            paths_by_group={channel: f"/memory/{name}_{channel}.pkl"},
        ))

    declared_inputs = tuple(
        ArtifactSpec.input(name, artifact_type)
        for name, artifact_type, _channel in producer_declarations
    )
    inputs = tuple(
        replace(
            spec,
            relations=(InputGroupLineageSourceRelation(source=spec.ref()),),
        )
        for spec in declared_inputs
    )
    input_by_name = {spec.name: spec for spec in inputs}
    invocation_key = FunctionInvocationKey("measure", DEFAULT_GROUP_KEY, 0)
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output(
                    "Measurements",
                    MeasurementsArtifactType,
                    relations=(
                        GroupLineageSourceRelation(input_by_name["Nuclei"].ref()),
                        GroupLineageSourceRelation(input_by_name["Cells"].ref()),
                        GroupLineageSourceRelation(input_by_name["Cytoplasm"].ref()),
                        ArtifactSpecRelation(input_by_name["CropBlue"].ref()),
                    ),
                ),
                groups=(None,),
                invocation_keys=(invocation_key,),
            ),
        ),
        consumers=tuple(
            ArtifactConsumer(
                spec=spec,
                invocation_keys=(invocation_key,),
            )
            for spec in inputs
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="MeasureObjectSizeShape"),
        3,
        declarations,
        PathPlannerGroupScope.ungrouped(),
    )

    assert maps.group_scope == PathPlannerGroupScope.ungrouped()
    assert set(maps.inputs) == {spec.ref() for spec in inputs}


def test_artifact_managed_single_source_output_retains_source_group_scope():
    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="CropBlue",
        path="/memory/CropBlue.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/CropBlue_1.pkl"},
    ))
    declared_source = ArtifactSpec.input("CropBlue", ImageArtifactType)
    source = replace(
        declared_source,
        relations=(InputGroupLineageSourceRelation(source=declared_source.ref()),),
    )
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output(
                    "Nuclei",
                    ObjectLabelsArtifactType,
                    relations=(GroupLineageSourceRelation(source.ref()),),
                ),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey("identify", DEFAULT_GROUP_KEY, 0),
                ),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=source,
                invocation_keys=(
                    FunctionInvocationKey("identify", DEFAULT_GROUP_KEY, 0),
                ),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="IdentifyPrimaryObjects"),
        3,
        declarations,
        PathPlannerGroupScope.ungrouped(),
    )

    assert maps.group_scope == PathPlannerGroupScope.ungrouped()
    nuclei_ref = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType).ref()
    assert maps.outputs[nuclei_ref].group_keys == ("1",)
    assert maps.outputs[nuclei_ref].group_component is AllComponents.CHANNEL


def test_declared_group_lineage_unions_compatible_source_groups():
    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="OrigStain1",
        path="/memory/OrigStain1.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/OrigStain1_1.pkl"},
    ))
    _record_declared_output(planner, ArtifactOutputPlan(
        name="OrigStain2",
        path="/memory/OrigStain2.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/OrigStain2_2.pkl"},
    ))
    first_ref = ArtifactSpec.input("OrigStain1", ImageArtifactType).ref()
    second_ref = ArtifactSpec.input("OrigStain2", ImageArtifactType).ref()
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output(
                    "Measurements",
                    MeasurementsArtifactType,
                    relations=(
                        GroupLineageSourceRelation(first_ref),
                        GroupLineageSourceRelation(second_ref),
                    ),
                ),
                groups=("1", "2"),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input("OrigStain1", ImageArtifactType),
                invocation_keys=(FunctionInvocationKey("measure", "1", 0),),
            ),
            ArtifactConsumer(
                spec=ArtifactSpec.input("OrigStain2", ImageArtifactType),
                invocation_keys=(FunctionInvocationKey("measure", "2", 0),),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="Measure"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
    )

    measurements_ref = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
    ).ref()
    assert maps.outputs[measurements_ref].group_keys == ("1", "2")
    assert maps.relation_source_scopes[first_ref] == (
        PathPlannerGroupScope.from_raw(("1",), component=AllComponents.CHANNEL)
    )
    assert maps.relation_source_scopes[second_ref] == (
        PathPlannerGroupScope.from_raw(("2",), component=AllComponents.CHANNEL)
    )


def test_dynamic_group_scope_union_remains_dynamic():
    dynamic_scope = PathPlannerGroupScope.dynamic(AllComponents.SITE)
    concrete_scope = PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.SITE,
    )

    assert (
        PathPlannerGroupScope.union_compatible((dynamic_scope, concrete_scope))
        == dynamic_scope
    )


def test_collected_lineage_outputs_use_relation_owned_group_scope():
    planner = _artifact_planner_stub()
    for name, channel, artifact_type in (
        ("CropBlue", "1", ImageArtifactType),
        ("CropGreen", "2", ImageArtifactType),
        ("Nuclei", "1", ObjectLabelsArtifactType),
    ):
        _record_declared_output(planner, ArtifactOutputPlan(
            name=name,
            path=f"/memory/{name}.pkl",
            artifact_type=artifact_type,
            group_keys=(channel,),
            group_component=AllComponents.CHANNEL,
            variable_components=(AllComponents.SITE,),
            paths_by_group={channel: f"/memory/{name}_{channel}.pkl"},
        ))

    input_specs = (
        ArtifactSpec.input("CropBlue", ImageArtifactType),
        ArtifactSpec.input("CropGreen", ImageArtifactType),
        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
    )
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output(
                    "Measurements",
                    MeasurementsArtifactType,
                    relations=tuple(
                        GroupLineageSourceRelation(spec.ref()) for spec in input_specs
                    ),
                ),
                groups=(None,),
                invocation_keys=(),
            ),
        ),
        consumers=tuple(
            ArtifactConsumer(
                spec=spec,
                invocation_keys=(),
            )
            for spec in input_specs
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(
            name="MeasureColocalization",
            group_by=GroupBy.SITE,
            variable_components=(VariableComponents.CHANNEL,),
        ),
        3,
        declarations,
        PathPlannerGroupScope.dynamic(AllComponents.SITE),
    )

    assert maps.group_scope == PathPlannerGroupScope.dynamic(AllComponents.SITE)
    measurements_ref = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
    ).ref()
    assert maps.outputs[measurements_ref].group_keys == (None,)
    assert maps.outputs[measurements_ref].group_component is AllComponents.SITE


def test_declared_group_lineage_cannot_rewrite_scalar_step_execution_scope():
    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="MembInvertRemoveHoles",
        path="/memory/MembInvertRemoveHoles.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": "/memory/MembInvertRemoveHoles_3.pkl"},
    ))
    _record_declared_output(planner, ArtifactOutputPlan(
        name="MonolayerMask",
        path="/memory/MonolayerMask.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/MonolayerMask_1.pkl"},
    ))
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "MembMasked",
                    ImageArtifactType,
                    ArtifactSpec.input(
                        "MembInvertRemoveHoles",
                        ImageArtifactType,
                    ).ref(),
                ),
                groups=("1", "3"),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input("MembInvertRemoveHoles", ImageArtifactType),
                invocation_keys=(),
            ),
            ArtifactConsumer(
                spec=ArtifactSpec.input(
                    "MonolayerMask",
                    ImageArtifactType,
                    relations=(
                        InputGroupLineageSourceRelation(
                            source=ArtifactSpec.input(
                                "MembInvertRemoveHoles",
                                ImageArtifactType,
                            ).ref()
                        ),
                    ),
                ),
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="MaskImage"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(("1", "3"), component=AllComponents.CHANNEL),
    )

    assert maps.group_scope == PathPlannerGroupScope.from_raw(
        ("1", "3"),
        component=AllComponents.CHANNEL,
    )
    masked_ref = ArtifactSpec.output("MembMasked", ImageArtifactType).ref()
    assert maps.outputs[masked_ref].group_keys == ("3",)
    assert maps.outputs[masked_ref].relations == (
        GroupLineageSourceRelation(
            ArtifactSpec.input(
                "MembInvertRemoveHoles",
                ImageArtifactType,
            ).ref()
        ),
    )
    monolayer_ref = ArtifactSpec.input("MonolayerMask", ImageArtifactType).ref()
    assert tuple(maps.inputs) == (
        ArtifactSpec.input("MembInvertRemoveHoles", ImageArtifactType).ref(),
        monolayer_ref,
    )
    assert maps.inputs[monolayer_ref].path == ("/memory/MonolayerMask_1.pkl")
    assert maps.inputs[monolayer_ref].group_keys == ("1",)


def test_artifact_output_storage_scope_is_independent_of_execution_scope():
    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="PriorMeasurements",
        path="/memory/PriorMeasurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "1": "/memory/PriorMeasurements_1.pkl",
            "2": "/memory/PriorMeasurements_2.pkl",
        },
        producer_step_index=1,
        producer_step_name="MeasureObjectIntensity",
    ))
    _record_declared_output(planner, ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/Nuclei_1.pkl"},
        producer_step_index=2,
        producer_step_name="IdentifyPrimaryObjects",
    ))
    nuclei_input = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output(
                    "Ratio",
                    MeasurementsArtifactType,
                    relations=(GroupLineageSourceRelation(nuclei_input.ref()),),
                ),
                groups=("3",),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input(
                    "PriorMeasurements",
                    MeasurementsArtifactType,
                ),
                invocation_keys=(),
            ),
            ArtifactConsumer(
                spec=nuclei_input,
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="CalculateMath"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(("3",), component=AllComponents.CHANNEL),
    )

    assert maps.group_scope == PathPlannerGroupScope.from_raw(
        ("3",),
        component=AllComponents.CHANNEL,
    )
    ratio_ref = ArtifactSpec.output("Ratio", MeasurementsArtifactType).ref()
    assert maps.outputs[ratio_ref].group_keys == ("1",)
    assert tuple(maps.inputs) == (
        ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType).ref(),
        ArtifactSpec.input("PriorMeasurements", MeasurementsArtifactType).ref(),
    )


def test_each_output_storage_scope_is_independent_of_execution_scope():
    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="source",
        path="/memory/source.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/source_2.pkl"},
    ))
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "scoped",
                    ImageArtifactType,
                    ArtifactSpec.input("source", ImageArtifactType).ref(),
                ),
                groups=("1", "2"),
                invocation_keys=(),
            ),
            ArtifactProducer(
                spec=ArtifactSpec.output("ambiguous", MeasurementsArtifactType),
                groups=("1", "2"),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input(
                    "source",
                    ImageArtifactType,
                    relations=(
                        InputGroupLineageSourceRelation(
                            source=ArtifactSpec.input(
                                "source",
                                ImageArtifactType,
                            ).ref()
                        ),
                    ),
                ),
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="invocation-scoped-output"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
    )

    execution_scope = PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )
    relation_scope = PathPlannerGroupScope.from_raw(
        ("2",),
        component=AllComponents.CHANNEL,
    )
    assert maps.group_scope == execution_scope
    scoped_ref = ArtifactSpec.output("scoped", ImageArtifactType).ref()
    ambiguous_ref = ArtifactSpec.output(
        "ambiguous",
        MeasurementsArtifactType,
    ).ref()
    assert PathPlannerGroupScope.from_output_plan(maps.outputs[scoped_ref]) == (
        relation_scope
    )
    assert PathPlannerGroupScope.from_output_plan(maps.outputs[ambiguous_ref]) == (
        execution_scope
    )


def test_dict_pattern_output_groups_do_not_drive_scalar_scope_narrowing():
    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="source",
        path="/memory/source.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": "/memory/source_3.pkl"},
    ))
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "output",
                    ImageArtifactType,
                    ArtifactSpec.input("source", ImageArtifactType).ref(),
                ),
                groups=("1",),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input(
                    "source",
                    ImageArtifactType,
                    relations=(
                        InputGroupLineageSourceRelation(
                            source=ArtifactSpec.input(
                                "source",
                                ImageArtifactType,
                            ).ref()
                        ),
                    ),
                ),
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="dict_pattern"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(("1", "3"), component=AllComponents.CHANNEL),
    )

    assert maps.group_scope == PathPlannerGroupScope.from_raw(
        ("1", "3"),
        component=AllComponents.CHANNEL,
    )


def test_source_binding_component_identity_narrows_declared_output_lineage():
    planner = _artifact_planner_stub()
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "Cells",
                    ObjectLabelsArtifactType,
                    ArtifactSpec.input("origMemb", ImageArtifactType).ref(),
                ),
                groups=("1", "2", "3"),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input("origMemb", ImageArtifactType),
                invocation_keys=(),
            ),
        ),
    )
    source_bindings = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="origMemb",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "3"),),
            ),
        ),
        enabled=True,
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="Watershed", source_bindings=source_bindings),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("1", "2", "3"),
            component=AllComponents.CHANNEL,
        ),
        source_bindings=source_bindings,
    )

    cells_ref = ArtifactSpec.output("Cells", ObjectLabelsArtifactType).ref()
    assert maps.outputs[cells_ref].group_keys == ("3",)
    assert maps.outputs[cells_ref].group_component is AllComponents.CHANNEL


def test_source_binding_identity_scopes_outputs_without_execution_fanout():
    planner = _artifact_planner_stub()
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "Cells",
                    ObjectLabelsArtifactType,
                    ArtifactSpec.input("origMemb", ImageArtifactType).ref(),
                ),
                groups=(None,),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input("origMemb", ImageArtifactType),
                invocation_keys=(),
            ),
        ),
    )
    source_bindings = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="origMemb",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "3"),),
            ),
        ),
        enabled=True,
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="Watershed", source_bindings=source_bindings),
        3,
        declarations,
        PathPlannerGroupScope.ungrouped(),
        source_bindings=source_bindings,
    )

    assert maps.group_scope == PathPlannerGroupScope.ungrouped()
    cells_ref = ArtifactSpec.output("Cells", ObjectLabelsArtifactType).ref()
    assert maps.outputs[cells_ref].group_keys == ("3",)
    assert maps.outputs[cells_ref].group_component is AllComponents.CHANNEL


def test_image_object_outputs_keep_declared_image_execution_group_scope():
    planner = _artifact_planner_stub()
    planner.ctx.microscope_handler = SimpleNamespace(
        can_resolve_metadata_artifact=lambda artifact_name: artifact_name == "DF_image",
    )
    _record_declared_output(planner, ArtifactOutputPlan(
        name="Tile_of_grid",
        path="/memory/Tile_of_grid.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/Tile_of_grid_1.pkl"},
    ))
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output(
                    "MeasureObjectIntensity_7_measurements",
                    MeasurementsArtifactType,
                ),
                groups=("2",),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input(
                    "Tile_of_grid",
                    ObjectLabelsArtifactType,
                    relations=(
                        InputGroupLineageSourceRelation(
                            source=ArtifactSpec.input(
                                "DF_image",
                                ImageArtifactType,
                            ).ref()
                        ),
                    ),
                ),
                invocation_keys=(),
            ),
        ),
        non_plan_consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input("DF_image", ImageArtifactType),
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="MeasureObjectIntensity"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("2",),
            component=AllComponents.CHANNEL,
        ),
    )

    measurements_ref = ArtifactSpec.output(
        "MeasureObjectIntensity_7_measurements",
        MeasurementsArtifactType,
    ).ref()
    assert maps.outputs[measurements_ref].group_keys == ("2",)


def test_group_lineage_source_resolves_prior_main_flow_output_without_store_input():
    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="Tile_of_grid",
        path="/memory/Tile_of_grid.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/Tile_of_grid_1.pkl"},
    ))
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "Filtered_tiles",
                    ObjectLabelsArtifactType,
                    ArtifactSpec.input("Tile_of_grid", ObjectLabelsArtifactType).ref(),
                ),
                groups=("2",),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input(
                    "Tile_of_grid",
                    ObjectLabelsArtifactType,
                ),
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _snapshot(name="FilterObjects"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("2",),
            component=AllComponents.CHANNEL,
        ),
    )

    assert tuple(maps.inputs) == (
        ArtifactSpec.input("Tile_of_grid", ObjectLabelsArtifactType).ref(),
    )
    filtered_tiles_ref = ArtifactSpec.output(
        "Filtered_tiles",
        ObjectLabelsArtifactType,
    ).ref()
    assert maps.outputs[filtered_tiles_ref].group_keys == ("1",)


def test_planner_uses_invocation_aware_artifact_declaration_provider():
    def identify(image, artifact_name: str):
        return image

    def declarations_for_invocation(invocation, step_context):
        assert step_context.step_name == "identify_cells"
        artifact_name = dict(invocation.kwargs)["artifact_name"]

        @artifact_outputs(ArtifactSpec.output(artifact_name, ObjectLabelsArtifactType))
        def declared_artifact_owner(image):
            return image

        return CallableContract.from_callable(declared_artifact_owner)

    planner = _artifact_planner_stub()
    planner.declaration_provider = declarations_for_invocation
    snapshot = _snapshot(
        is_function_step=True,
        func=(identify, {"artifact_name": "cells"}),
        group_by=GroupBy.NONE,
        variable_components=(VariableComponents.SITE,),
        name="identify_cells",
        index=2,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )

    (
        declarations,
        func_pattern,
        execution_scope,
        _contracts,
    ) = planner.artifacts.prepare_step_declarations(
        snapshot,
    )
    assert execution_scope is FunctionStepExecutionScope.AXIS
    output_plan = ArtifactOutputPlan(
        name="cells",
        path="/memory/cells.pkl",
        artifact_type=ObjectLabelsArtifactType,
    )
    compiled = planner.artifacts.build_step_compiled_function_pattern(
        snapshot,
        True,
        func_pattern,
        {},
        {output_plan.ref(): output_plan},
        {},
        PathPlannerGroupScope.ungrouped(),
    )

    assert list(declarations.outputs) == [
        ArtifactSpec.output("cells", ObjectLabelsArtifactType).ref()
    ]
    assert tuple(
        plan.ref() for plan in compiled.groups[0].invocations[0].artifact_output_plans
    ) == (ArtifactSpec.output("cells", ObjectLabelsArtifactType).ref(),)


def test_artifact_managed_regular_pattern_preserves_group_by_scope():
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    @artifact_inputs(ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType))
    def filter_objects(image, *, runtime):
        del runtime
        return image

    planner = _artifact_planner_stub()
    snapshot = _snapshot(
        is_function_step=True,
        func=filter_objects,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="FilterObjects",
        index=2,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )
    input_component_scopes = PathPlannerComponentScopes(
        {
            VariableComponents.CHANNEL: PathPlannerGroupScope.from_raw(
                ("1", "2"),
                component=AllComponents.CHANNEL,
            )
        }
    )

    _declarations, _pattern, execution_scope, _contracts = (
        planner.artifacts.prepare_step_declarations(
            snapshot,
        )
    )
    execution_groups = planner.execution_groups.get_execution_groups(
        snapshot,
        input_component_scopes,
    )

    assert execution_groups == PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )
    assert execution_scope is FunctionStepExecutionScope.AXIS

    snapshot.step.func = {
        "1": [filter_objects],
        "2": [filter_objects],
    }
    _declarations, _pattern, execution_scope, _contracts = (
        planner.artifacts.prepare_step_declarations(
            snapshot,
        )
    )
    execution_groups = planner.execution_groups.get_execution_groups(
        snapshot,
        input_component_scopes,
    )

    assert execution_groups == PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )
    assert execution_scope is FunctionStepExecutionScope.AXIS


def test_adapter_managed_edges_bind_only_compatible_special_parameters():
    labels = ArtifactSpec.input(
        "Nuclei",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    measurements = ArtifactSpec.input("Measurements", MeasurementsArtifactType)

    @artifact_inputs(labels, measurements)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    @special_inputs("labels")
    def consume(image, labels: ObjectLabelValue, *, runtime):
        del labels, runtime
        return image

    input_plans = {
        spec.ref(): ArtifactInputPlan(
            name=spec.name,
            path=f"/memory/{spec.name}.pkl",
            artifact_type=spec.artifact_type,
        )
        for spec in (labels, measurements)
    }
    compiled = compile_function_pattern(consume, input_plans, {})
    compiled = _artifact_planner_stub().artifacts.compile_invocation_input_edges(
        compiled,
        artifact_inputs=input_plans,
        relation_source_scopes={
            spec.ref(): input_plans[spec.ref()].producer_group_scope()
            for spec in (labels, measurements)
        },
        execution_group_scope=PathPlannerGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet(),
    )

    invocation = next(compiled.iter_invocations())
    adapter = invocation.contract.runtime_adapter
    assert adapter is not None
    assert adapter.manages_artifact_inputs
    assert tuple(
        (edge.spec, edge.spec.parameter_name)
        for edge in invocation.artifact_input_edges
    ) == ((labels, "labels"), (measurements, None))


def test_artifact_managed_regular_pattern_uses_declared_owner_scope():
    nuclei = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)

    @artifact_inputs(nuclei)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def identify_primary_objects(image, *, runtime):
        del runtime
        return image

    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name=nuclei.name,
        path="/memory/Nuclei.pkl",
        artifact_type=nuclei.artifact_type,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
        component_domains=(
            ComponentGroupScope.from_raw(
                ("1",),
                component=AllComponents.CHANNEL,
            ),
            ComponentGroupScope.from_raw(
                ("1", "2"),
                component=AllComponents.SITE,
            ),
        ),
    ))
    snapshot = _snapshot(
        is_function_step=True,
        func=identify_primary_objects,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="IdentifyPrimaryObjects",
        index=2,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )
    main_flow_scopes = PathPlannerComponentScopes(
        {
            VariableComponents.CHANNEL: PathPlannerGroupScope.from_raw(
                ("2", "3"),
                component=AllComponents.CHANNEL,
            )
        }
    )

    scope = planner.execution_groups.get_execution_groups(
        snapshot,
        main_flow_scopes,
        contracts=(CallableContract.from_callable(identify_primary_objects),),
    )

    assert scope == PathPlannerGroupScope.from_raw(
        ("1",),
        component=AllComponents.CHANNEL,
    )


def test_artifact_managed_regular_pattern_unions_compatible_owner_scopes():
    nuclei_1 = ArtifactSpec.input("Nuclei1", ObjectLabelsArtifactType)
    nuclei_2 = ArtifactSpec.input("Nuclei2", ObjectLabelsArtifactType)

    @artifact_inputs(nuclei_1, nuclei_2)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def relate_objects(image, *, runtime):
        del runtime
        return image

    planner = _artifact_planner_stub()
    for spec, channel in ((nuclei_1, "1"), (nuclei_2, "2")):
        _record_declared_output(planner, ArtifactOutputPlan(
            name=spec.name,
            path=f"/memory/{spec.name}.pkl",
            artifact_type=spec.artifact_type,
            group_keys=(channel,),
            group_component=AllComponents.CHANNEL,
        ))
    snapshot = _snapshot(
        is_function_step=True,
        func=relate_objects,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="RelateObjects",
        index=2,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )
    main_flow_scopes = PathPlannerComponentScopes(
        {
            VariableComponents.CHANNEL: PathPlannerGroupScope.from_raw(
                ("1", "2"),
                component=AllComponents.CHANNEL,
            )
        }
    )

    scope = planner.execution_groups.get_execution_groups(
        snapshot,
        main_flow_scopes,
        contracts=(CallableContract.from_callable(relate_objects),),
    )

    assert scope == PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )


def test_artifact_owner_variable_axis_projects_to_consumer_scope():
    comet_outline = ArtifactSpec.input("CometOutline", ObjectLabelsArtifactType)

    @artifact_inputs(comet_outline)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def measure_object_size_shape(image, *, runtime):
        del runtime
        return image

    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name=comet_outline.name,
        path="/memory/CometOutline.pkl",
        artifact_type=comet_outline.artifact_type,
        group_keys=("1", "2"),
        group_component=AllComponents.SITE,
    ))
    snapshot = _snapshot(
        is_function_step=True,
        func=measure_object_size_shape,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="MeasureObjectSizeShape",
        index=2,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )
    main_flow_scopes = PathPlannerComponentScopes(
        {
            VariableComponents.CHANNEL: PathPlannerGroupScope.from_raw(
                ("1",),
                component=AllComponents.CHANNEL,
            )
        }
    )

    scope = planner.execution_groups.get_execution_groups(
        snapshot,
        main_flow_scopes,
        contracts=(CallableContract.from_callable(measure_object_size_shape),),
    )

    assert scope == PathPlannerGroupScope.from_raw(
        ("1",),
        component=AllComponents.CHANNEL,
    )


def test_execution_groups_resolve_non_grouped_variable_component_conflicts():
    planner = _artifact_planner_stub()
    snapshot = _snapshot(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE, VariableComponents.CHANNEL),
        name="source_bound_cellprofiler_step",
        source_bindings=EMPTY_SOURCE_BINDINGS,
    )

    assert (
        planner.execution_groups.get_execution_groups(snapshot)
        == PathPlannerGroupScope.ungrouped()
    )


def test_non_dict_group_by_declares_dynamic_scope_without_plate_key_lookup():
    planner = _artifact_planner_stub()
    planner.orchestrator = SimpleNamespace(
        get_component_keys=lambda group_by: pytest.fail(
            "non-dict group_by must not request plate component keys"
        )
    )
    source_snapshot = _snapshot(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="enhance",
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )

    source_scope = planner.execution_groups.get_execution_groups(
        source_snapshot,
        PathPlannerComponentScopes.empty(),
    )
    assert source_scope == PathPlannerGroupScope.from_raw(
        (None,),
        component=AllComponents.CHANNEL,
    )


def test_non_dict_group_by_uses_dynamic_source_scope_for_pipeline_start():
    planner = _artifact_planner_stub()
    source_snapshot = _snapshot(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="source_loaded_channel_callable",
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PIPELINE_START,
    )

    source_scope = planner.execution_groups.get_execution_groups(
        source_snapshot,
        PathPlannerComponentScopes.empty(),
    )
    assert source_scope == PathPlannerGroupScope.from_raw(
        (None,),
        component=AllComponents.CHANNEL,
    )


def test_dict_pattern_group_by_declares_execution_group_component():
    planner = _artifact_planner_stub()
    snapshot = _snapshot(
        is_function_step=True,
        func={
            "1": lambda image: image,
            "2": lambda image: image,
        },
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="channel_dispatch",
        source_bindings=EMPTY_SOURCE_BINDINGS,
    )

    scope = planner.execution_groups.get_execution_groups(
        snapshot,
        PathPlannerComponentScopes.empty(),
    )

    assert scope == PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )


def test_dict_pattern_rejects_group_by_none_execution_component():
    planner = _artifact_planner_stub()
    snapshot = _snapshot(
        is_function_step=True,
        func={
            "1": lambda image: image,
            "2": lambda image: image,
        },
        group_by=GroupBy.NONE,
        variable_components=(VariableComponents.CHANNEL,),
        name="channel_dispatch",
        source_bindings=EMPTY_SOURCE_BINDINGS,
    )

    with pytest.raises(
        ValueError,
        match="dict function pattern without a concrete group_by component",
    ):
        planner.execution_groups.get_execution_groups(
            snapshot,
            PathPlannerComponentScopes.empty(),
        )


def test_execution_groups_reject_grouped_group_by_axis_conflict():
    planner = _artifact_planner_stub()

    composite_snapshot = _snapshot(
        is_function_step=True,
        func={"1": lambda image: image},
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.CHANNEL,),
        name="channel_dispatch",
        source_bindings=EMPTY_SOURCE_BINDINGS,
    )
    with pytest.raises(
        ValueError,
        match="channel_dispatch.*group_by=CHANNEL cannot also appear",
    ):
        planner.execution_groups.get_execution_groups(
            composite_snapshot,
            PathPlannerComponentScopes.empty(),
        )


def test_non_dict_group_by_declares_dynamic_scope_when_input_axis_is_collapsed():
    planner = _artifact_planner_stub()
    input_scopes = PathPlannerComponentScopes(
        {
            VariableComponents.CHANNEL: PathPlannerGroupScope.ungrouped(),
            VariableComponents.SITE: PathPlannerGroupScope.from_raw(
                ("1", "2"),
                component=AllComponents.SITE,
            ),
        }
    )
    snapshot = _snapshot(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="measure_channel_named_artifacts_over_site_stack",
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )

    scope = planner.execution_groups.get_execution_groups(snapshot, input_scopes)

    assert scope == PathPlannerGroupScope.dynamic(AllComponents.CHANNEL)


def test_module_special_outputs_preserve_existing_main_flow_component_scopes():
    measurement_spec = ArtifactSpec.output("Measurements", MeasurementsArtifactType)

    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_outputs=True,
    )
    @artifact_outputs(measurement_spec)
    def measurement_only(image, *, runtime):
        del runtime
        return image, object()

    compiled_contract = CallableContract.from_callable(measurement_only)
    pattern = CompiledFunctionPattern(
        groups=(
            CompiledFunctionGroup(
                group_key="default",
                invocations=(
                    CompiledFunctionInvocation(
                        key=FunctionInvocationKey.from_contract(
                            compiled_contract,
                            "default",
                            0,
                        ),
                        contract=compiled_contract,
                    ),
                ),
            ),
        ),
        is_grouped=False,
    )
    input_scopes = PathPlannerComponentScopes(
        {
            VariableComponents.CHANNEL: PathPlannerGroupScope.from_raw(
                ("1",),
                component=AllComponents.CHANNEL,
            ),
            VariableComponents.SITE: PathPlannerGroupScope.ungrouped(),
        }
    )
    snapshot = _snapshot(
        is_function_step=True,
        variable_components=(VariableComponents.CHANNEL,),
        group_by=GroupBy.SITE,
        name="measurement_only",
    )

    output_scopes = input_scopes.output_after(
        snapshot,
        PathPlannerGroupScope.dynamic(AllComponents.SITE),
        pattern,
    )

    assert output_scopes == input_scopes


def test_module_canonical_output_applies_functionstep_component_transformation():
    image_spec = ArtifactSpec.output("Enhanced", ImageArtifactType)

    @artifact_outputs(image_spec)
    def image_output(image):
        return image

    compiled_contract = CallableContract.from_callable(image_output)
    pattern = CompiledFunctionPattern(
        groups=(
            CompiledFunctionGroup(
                group_key="default",
                invocations=(
                    CompiledFunctionInvocation(
                        key=FunctionInvocationKey.from_contract(
                            compiled_contract,
                            "default",
                            0,
                        ),
                        contract=compiled_contract,
                    ),
                ),
            ),
        ),
        is_grouped=False,
    )
    input_scopes = PathPlannerComponentScopes(
        {
            VariableComponents.CHANNEL: PathPlannerGroupScope.from_raw(
                ("1",),
                component=AllComponents.CHANNEL,
            ),
            VariableComponents.SITE: PathPlannerGroupScope.ungrouped(),
        }
    )
    snapshot = _snapshot(
        is_function_step=True,
        variable_components=(VariableComponents.CHANNEL,),
        group_by=GroupBy.SITE,
        name="image_output",
    )

    output_scopes = input_scopes.output_after(
        snapshot,
        PathPlannerGroupScope.dynamic(AllComponents.SITE),
        pattern,
    )

    assert output_scopes == PathPlannerComponentScopes(
        {
            VariableComponents.CHANNEL: PathPlannerGroupScope.ungrouped(),
            VariableComponents.SITE: PathPlannerGroupScope.dynamic(AllComponents.SITE),
        }
    )


def test_non_dict_group_by_namespaces_artifact_outputs_with_dynamic_component():
    planner = _artifact_planner_stub()
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=ArtifactSpec.output(
                    "segmentation_masks",
                    ObjectLabelsArtifactType,
                ),
                groups=(None,),
                invocation_keys=(),
            ),
        )
    )
    snapshot = _snapshot(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="single_callable_channel_artifacts",
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )

    maps = planner.artifacts.compile_plan_maps(
        snapshot,
        2,
        declarations,
        PathPlannerGroupScope.from_raw(
            (None,),
            component=AllComponents.CHANNEL,
        ),
    )

    output_plan = maps.outputs[
        ArtifactSpec.output("segmentation_masks", ObjectLabelsArtifactType).ref()
    ]
    assert output_plan.group_keys == (None,)
    assert output_plan.group_component is AllComponents.CHANNEL


def test_non_dict_group_by_uses_source_binding_identity_for_pipeline_start_scope():
    planner = _artifact_planner_stub()
    snapshot = _snapshot(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="source_bound_channel_groups",
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="OrigStain1",
                    component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
                NamedSourceBinding(
                    alias="OrigStain2",
                    component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
                ),
            ),
        ),
        input_source=InputSource.PIPELINE_START,
    )

    scope = planner.execution_groups.get_execution_groups(
        snapshot,
        PathPlannerComponentScopes.empty(),
    )

    assert scope == PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )


def test_main_flow_source_anchor_restricts_execution_to_its_exact_channel():
    planner = _artifact_planner_stub()
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="BF_image",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="DF_image",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
                projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
            ),
        ),
    )
    snapshot = _snapshot(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="bf_source_consumer",
        source_bindings=source_bindings,
        input_source=InputSource.PIPELINE_START,
    )
    source_contract = CallableContract.from_callable(lambda image: image)
    source_contract = replace(
        source_contract,
        metadata=replace(
            source_contract.metadata,
            artifact_inputs=(
                ArtifactSpec.input("BF_image", ImageArtifactType),
                ArtifactSpec.input("DF_image", ImageArtifactType),
            ),
        ),
    )

    contract_bindings = planner.artifacts.source_bindings_for_contracts(
        snapshot,
        (source_contract,),
        StepInputDependency.pipeline_start(),
    )
    source_anchor_specs = tuple(
        binding.input_spec() for binding in contract_bindings.primary_plane_bindings
    )
    execution_bindings = contract_bindings.for_artifact_specs(
        source_anchor_specs,
        planner.artifact_context.available_artifacts,
    )
    scope = planner.execution_groups.get_execution_groups(
        snapshot,
        PathPlannerComponentScopes.empty(),
        source_bindings=execution_bindings,
    )

    assert tuple(binding.alias for binding in contract_bindings.bindings) == (
        "BF_image",
        "DF_image",
    )
    assert tuple(binding.alias for binding in execution_bindings.bindings) == (
        "BF_image",
    )
    assert scope == PathPlannerGroupScope.from_raw(
        ("1",),
        component=AllComponents.CHANNEL,
    )


def test_site_execution_preserves_channel_grouped_producer_and_output_lineage():
    source = ArtifactSpec.input("CropBlue", ImageArtifactType)
    measurements = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
        relations=(GroupLineageSourceRelation(source.ref()),),
    )

    @artifact_inputs(source)
    @artifact_outputs(measurements)
    def measure(image, CropBlue):
        del CropBlue
        return image

    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name=source.name,
        path="/memory/CropBlue.pkl",
        artifact_type=source.artifact_type,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        variable_components=(AllComponents.SITE,),
        paths_by_group={
            "1": "/memory/CropBlue_channel_1.pkl",
            "2": "/memory/CropBlue_channel_2.pkl",
        },
        producer_step_index=2,
        producer_step_name="Crop",
    ))
    snapshot = _snapshot(
        name="Measure",
        func=measure,
        group_by=GroupBy.SITE,
        variable_components=(VariableComponents.CHANNEL,),
    )
    execution_scope = PathPlannerGroupScope.from_raw(
        ("1", "2", "3"),
        component=AllComponents.SITE,
    )
    declarations = extract_artifact_declarations(measure)

    maps = planner.artifacts.compile_plan_maps(
        snapshot,
        3,
        declarations,
        execution_scope,
    )
    compiled = planner.artifacts.build_step_compiled_function_pattern(
        snapshot,
        True,
        measure,
        maps.inputs,
        maps.outputs,
        maps.relation_source_scopes,
        maps.group_scope,
    )

    assert maps.group_scope == execution_scope
    assert maps.inputs[source.ref()].producer_group_scope() == (
        ComponentGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        )
    )
    assert PathPlannerGroupScope.from_output_plan(maps.outputs[measurements.ref()]) == (
        execution_scope
    )
    edge = next(compiled.iter_invocations()).artifact_input_edges[0]
    assert edge.projection.invocation_scope == ComponentGroupScope.dynamic(
        AllComponents.SITE
    )
    assert edge.projection.producer_selection_scope == (
        maps.inputs[source.ref()].producer_group_scope()
    )


def test_execution_anchor_ignores_source_artifact_lineage():
    planner = _artifact_planner_stub()
    original_blue = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    original_red = ArtifactSpec.input("OrigRed", ImageArtifactType)
    rgb_image = ArtifactSpec.input(
        "RGBImage",
        ImageArtifactType,
        relations=(InputGroupLineageSourceRelation(original_red.ref()),),
    )
    planner.artifact_context = replace(
        planner.artifact_context,
        available_artifacts=ArtifactSpecCollection(
            (original_blue, original_red, rgb_image)
        ),
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="OrigBlue",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="OrigRed",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "3"),),
                projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
            ),
        ),
    )
    snapshot = _snapshot(
        source_bindings=source_bindings,
        input_source=InputSource.PIPELINE_START,
    )
    source_contract = CallableContract.from_callable(lambda image: image)
    source_contract = replace(
        source_contract,
        metadata=replace(
            source_contract.metadata,
            artifact_inputs=(original_blue, rgb_image),
        ),
    )

    contract_bindings = planner.artifacts.source_bindings_for_contracts(
        snapshot,
        (source_contract,),
        StepInputDependency.pipeline_start(),
    )
    source_anchor_specs = tuple(
        binding.input_spec() for binding in contract_bindings.primary_plane_bindings
    )
    execution_bindings = contract_bindings.for_artifact_specs(
        source_anchor_specs,
        planner.artifact_context.available_artifacts,
    )

    assert tuple(binding.alias for binding in contract_bindings.bindings) == (
        "OrigBlue",
        "OrigRed",
    )
    assert tuple(binding.alias for binding in execution_bindings.bindings) == (
        "OrigBlue",
    )


def test_runtime_artifact_input_plan_owns_relation_source_scope():
    planner = _artifact_planner_stub()
    original_red = ArtifactSpec.input("OrigRed", ImageArtifactType)
    rgb_image = ArtifactSpec.input(
        "RGBImage",
        ImageArtifactType,
        relations=(InputGroupLineageSourceRelation(original_red.ref()),),
    )
    planner.artifact_context = replace(
        planner.artifact_context,
        available_artifacts=ArtifactSpecCollection((original_red, rgb_image)),
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="OrigRed",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "3"),),
            ),
        ),
    )
    artifact_input = ArtifactInputPlan(
        name=rgb_image.name,
        path="/memory/RGBImage.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2", "3"),
        group_component=AllComponents.SITE,
        paths_by_group={
            "1": "/memory/RGBImage_site_1.pkl",
            "2": "/memory/RGBImage_site_2.pkl",
            "3": "/memory/RGBImage_site_3.pkl",
        },
    )
    declarations = ArtifactGraph(
        consumers=(
            ArtifactConsumer(
                spec=rgb_image,
                invocation_keys=(),
            ),
        ),
    )

    relation_scopes = planner.artifacts.relation_source_scopes_by_ref(
        declarations,
        {artifact_input.ref(): artifact_input},
        group_scope=PathPlannerGroupScope.from_raw(
            ("3",),
            component=AllComponents.CHANNEL,
        ),
        source_bindings=source_bindings,
        group_by=GroupBy.CHANNEL,
    )

    assert relation_scopes[rgb_image.ref()] == PathPlannerGroupScope.from_raw(
        ("1", "2", "3"),
        component=AllComponents.SITE,
    )


def test_relation_source_scopes_rejects_malformed_exact_input_plan_maps():
    planner = _artifact_planner_stub()
    input_spec = ArtifactSpec.input("RGBImage", ImageArtifactType)
    input_plan = ArtifactInputPlan(
        name=input_spec.name,
        path="/memory/RGBImage.pkl",
        artifact_type=input_spec.artifact_type,
    )
    declarations = ArtifactGraph(
        consumers=(ArtifactConsumer(input_spec, invocation_keys=()),),
    )
    invalid_maps = (
        (
            {input_spec.name: input_plan},
            TypeError,
            "require ArtifactSpecRef keys",
        ),
        (
            {
                input_spec.ref(): ArtifactOutputPlan(
                    name=input_spec.name,
                    path=input_plan.path,
                    artifact_type=input_spec.artifact_type,
                )
            },
            TypeError,
            "require ArtifactInputPlan values",
        ),
        (
            {
                ArtifactSpec.input("OtherImage", ImageArtifactType).ref(): (
                    input_plan
                )
            },
            ValueError,
            "conflicts with plan ref",
        ),
    )

    for artifact_inputs_by_ref, error_type, message in invalid_maps:
        with pytest.raises(error_type, match=message):
            planner.artifacts.relation_source_scopes_by_ref(
                declarations,
                artifact_inputs_by_ref,
                group_scope=PathPlannerGroupScope.ungrouped(),
                source_bindings=EMPTY_SOURCE_BINDINGS,
                group_by=None,
            )


def test_process_artifact_outputs_rejects_malformed_exact_maps():
    planner = _artifact_planner_stub()
    output_spec = ArtifactSpec.output("OutputImage", ImageArtifactType)
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=output_spec,
                groups=(None,),
                invocation_keys=(),
            ),
        ),
    )
    input_plan = ArtifactInputPlan(
        name="InputImage",
        path="/memory/InputImage.pkl",
        artifact_type=ImageArtifactType,
    )
    call_kwargs = {
        "execution_scope": FunctionStepExecutionScope.AXIS,
        "source_bindings": EMPTY_SOURCE_BINDINGS,
        "variable_components": ComponentSet(),
    }

    with pytest.raises(TypeError, match="artifact input maps require ArtifactSpecRef"):
        planner.artifacts.process_artifact_outputs(
            declarations,
            3,
            artifact_inputs={input_plan.name: input_plan},
            **call_kwargs,
        )

    invalid_output_groups = (
        (
            {output_spec.name: PathPlannerGroupScope.ungrouped()},
            TypeError,
            "output-group maps require ArtifactSpecRef keys",
        ),
        (
            {output_spec.ref(): (None,)},
            TypeError,
            "output-group maps require PathPlannerGroupScope values",
        ),
        (
            {
                ArtifactSpec.output("OtherOutput", ImageArtifactType).ref(): (
                    PathPlannerGroupScope.ungrouped()
                )
            },
            ValueError,
            "is not an exact declared output",
        ),
    )
    for output_groups, error_type, message in invalid_output_groups:
        with pytest.raises(error_type, match=message):
            planner.artifacts.process_artifact_outputs(
                declarations,
                3,
                output_groups,
                artifact_inputs={},
                **call_kwargs,
            )


def test_artifact_graph_output_groups_validates_exact_partial_map() -> None:
    first = ArtifactSpec.output("First", ImageArtifactType)
    second = ArtifactSpec.output("Second", ObjectLabelsArtifactType)
    graph = ArtifactGraph(
        producers=(
            ArtifactProducer(first, ("first-old",), ()),
            ArtifactProducer(second, ("second-old",), ()),
        ),
    )

    updated = graph.with_output_groups({first.ref(): ("first-new", "first-new")})

    assert updated.producers[0].groups == ("first-new",)
    assert updated.producers[1].groups == ("second-old",)

    invalid_maps = (
        (
            {first.name: ("first-new",)},
            TypeError,
            "require ArtifactSpecRef keys",
        ),
        (
            {
                ArtifactSpec.output("Unknown", ImageArtifactType).ref(): (
                    "unknown",
                )
            },
            ValueError,
            "is not an exact declared output",
        ),
        (
            {first.ref(): "first-new"},
            TypeError,
            "not a string",
        ),
        (
            {first.ref(): 1},
            TypeError,
            "must be iterable",
        ),
        (
            {first.ref(): ("first-new", 1)},
            TypeError,
            "require string or None keys",
        ),
    )
    for output_groups, error_type, message in invalid_maps:
        with pytest.raises(error_type, match=message):
            graph.with_output_groups(output_groups)


def test_non_dict_group_by_ignores_source_binding_identity_for_other_components():
    planner = _artifact_planner_stub()
    snapshot = _snapshot(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.SITE,
        variable_components=(VariableComponents.CHANNEL,),
        name="source_bound_site_groups",
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="OrigStain1",
                    component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
                NamedSourceBinding(
                    alias="OrigStain2",
                    component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
                ),
            ),
        ),
        input_source=InputSource.PIPELINE_START,
    )

    scope = planner.execution_groups.get_execution_groups(
        snapshot,
        PathPlannerComponentScopes.empty(),
    )

    assert scope == PathPlannerGroupScope.from_raw(
        (None,),
        component=AllComponents.SITE,
    )


def test_compiled_group_by_preserves_dynamic_execution_scope():
    planner = _artifact_planner_stub()
    planner.cfg = PathConfigStub(sub_dir="images", output_dir_suffix="_generated")
    planner.plans[3].group_by = GroupBy.CHANNEL
    planner.plans[3].variable_components = (VariableComponents.SITE,)
    planner.plans[3].func = lambda image: image
    snapshot = _snapshot(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="measure_after_channel_collapse",
        scope_id="plate::functionstep_3",
        input_source=InputSource.PREVIOUS_STEP,
    )
    artifact_maps = ArtifactPlanMaps(
        declarations=ArtifactGraph.empty(),
        group_scope=PathPlannerGroupScope.dynamic(AllComponents.CHANNEL),
        inputs={},
        outputs={},
        relation_source_scopes={},
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        source_universe_plan=CompiledSourceUniversePlan.empty(),
    )

    planner.steps.update_core_step_plan(
        snapshot,
        3,
        StepInputDependency.step_output(
            source_step_index=2,
            source_step_scope_id="plate::functionstep_2",
        ),
        Path("/input"),
        Path("/output"),
        artifact_maps,
        None,
    )

    assert planner.plans[3].group_by is GroupBy.CHANNEL
    assert planner.plans[3].execution_group_scope == PathPlannerGroupScope.dynamic(
        AllComponents.CHANNEL
    )


def test_artifact_input_plan_requires_an_exact_producer_kind():
    planner = _artifact_planner_stub()
    _record_declared_output(
        planner,
        ArtifactOutputPlan(
            name="nuclei",
            path="/memory/nuclei.pkl",
            artifact_type=ObjectLabelsArtifactType,
            producer_step_index=1,
            producer_step_name="identify",
        ),
    )

    with pytest.raises(MissingArtifactInputError, match="needs artifact input"):
        planner.artifacts.process_artifact_inputs(
            ArtifactGraph(
                consumers=(
                    ArtifactConsumer(
                        spec=ArtifactSpec.input("nuclei", MeasurementsArtifactType),
                        invocation_keys=(),
                    ),
                )
            ),
            consumer_scope=PathPlannerGroupScope.ungrouped(),
            sid=2,
            step_name="measure",
            source_bindings=EMPTY_SOURCE_BINDINGS,
            variable_components=ComponentSet(),
        execution_scope = FunctionStepExecutionScope.AXIS)


def test_artifact_input_plan_rejects_corrupt_exact_producer_kind():
    planner = _artifact_planner_stub()
    input_spec = ArtifactSpec.input("nuclei", MeasurementsArtifactType)
    producer_ref = input_spec.ref().for_plan_type(ArtifactOutputPlan)
    planner.declared[producer_ref] = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        producer_step_index=1,
        producer_step_name="identify",
    )

    with pytest.raises(ValueError, match="expects measurements"):
        planner.artifacts.process_artifact_inputs(
            ArtifactGraph(
                consumers=(
                    ArtifactConsumer(
                        spec=input_spec,
                        invocation_keys=(),
                    ),
                )
            ),
            consumer_scope=PathPlannerGroupScope.ungrouped(),
            sid=2,
            step_name="measure",
            source_bindings=EMPTY_SOURCE_BINDINGS,
            variable_components=ComponentSet(),
            execution_scope=FunctionStepExecutionScope.AXIS,
        )


def test_same_name_source_and_output_use_exact_typed_graph_identity():
    planner = _artifact_planner_stub()
    invocation_key = FunctionInvocationKey("identify_primary_objects", "2", 0)
    image_input = ArtifactSpec.input("PH3", ImageArtifactType)
    object_output = ArtifactSpec.output("PH3", ObjectLabelsArtifactType)
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                spec=object_output,
                groups=("2",),
                invocation_keys=(invocation_key,),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                spec=image_input,
                invocation_keys=(invocation_key,),
            ),
        ),
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(NamedSourceBinding(alias="PH3"),),
    )

    inputs = planner.artifacts.process_artifact_inputs(
        declarations,
        sid=2,
        consumer_scope=PathPlannerGroupScope.ungrouped(),
        source_bindings=source_bindings,
        variable_components=ComponentSet(),
        step_name="IdentifyPrimaryObjects",
        execution_scope=FunctionStepExecutionScope.AXIS,
    )
    outputs = planner.artifacts.process_artifact_outputs(
        declarations,
        sid=2,
        artifact_inputs=inputs,
        source_bindings=source_bindings,
        variable_components=ComponentSet(),
        step_name="IdentifyPrimaryObjects",
        execution_scope=FunctionStepExecutionScope.AXIS,
    )

    assert inputs == {}
    assert tuple(declarations.inputs) == (image_input.ref(),)
    assert tuple(declarations.outputs) == (object_output.ref(),)
    assert tuple(planner.declared) == (object_output.ref(),)

    matching_inputs = planner.artifacts.process_artifact_inputs(
        ArtifactGraph(
            consumers=(
                ArtifactConsumer(
                    spec=object_output.for_plan_type(ArtifactInputPlan),
                    invocation_keys=(invocation_key,),
                ),
            )
        ),
        sid=3,
        consumer_scope=PathPlannerGroupScope.ungrouped(),
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet(),
        step_name="ConsumePH3Objects",
        execution_scope=FunctionStepExecutionScope.AXIS,
    )

    object_input_ref = object_output.for_plan_type(ArtifactInputPlan).ref()
    assert matching_inputs[object_input_ref].ref() == object_output.for_plan_type(
        ArtifactInputPlan
    ).ref()
    assert outputs[object_output.ref()].ref() == object_output.ref()


def test_artifact_input_plan_preserves_single_grouped_producer_scope():
    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="illumination",
        path="/memory/illumination.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/illumination_channel_1.pkl"},
        producer_step_index=1,
        producer_step_name="calculate_illumination",
    ))

    inputs = planner.artifacts.process_artifact_inputs(
        ArtifactGraph(
            consumers=(
                ArtifactConsumer(
                    spec=ArtifactSpec.input("illumination", ImageArtifactType),
                    invocation_keys=(),
                ),
            )
        ),
        consumer_scope=PathPlannerGroupScope.from_raw(
            ("2", "3"),
            component=AllComponents.SITE,
        ),
        sid=2,
        step_name="apply_illumination",
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet(),
    execution_scope = FunctionStepExecutionScope.AXIS)

    illumination_ref = ArtifactSpec.input("illumination", ImageArtifactType).ref()
    plan = inputs[illumination_ref]
    assert plan.group_keys == ("1",)
    assert plan.group_component is AllComponents.CHANNEL
    assert plan.path == "/memory/illumination_channel_1.pkl"
    assert plan.paths_by_group == {"1": "/memory/illumination_channel_1.pkl"}


@pytest.mark.parametrize(
    ("available", "required", "contains"),
    (
        (PathPlannerGroupScope.ungrouped(), PathPlannerGroupScope.ungrouped(), True),
        (
            PathPlannerGroupScope.dynamic(AllComponents.CHANNEL),
            PathPlannerGroupScope.dynamic(AllComponents.CHANNEL),
            True,
        ),
        (
            PathPlannerGroupScope.dynamic(AllComponents.CHANNEL),
            PathPlannerGroupScope.from_raw(("1",), component=AllComponents.CHANNEL),
            True,
        ),
        (
            PathPlannerGroupScope.from_raw(("1", "2"), component=AllComponents.CHANNEL),
            PathPlannerGroupScope.from_raw(("2",), component=AllComponents.CHANNEL),
            True,
        ),
        (
            PathPlannerGroupScope.from_raw(("1",), component=AllComponents.CHANNEL),
            PathPlannerGroupScope.dynamic(AllComponents.CHANNEL),
            False,
        ),
        (
            PathPlannerGroupScope.dynamic(AllComponents.CHANNEL),
            PathPlannerGroupScope.dynamic(AllComponents.SITE),
            False,
        ),
    ),
)
def test_component_group_scope_contains_exact_required_scope(
    available: PathPlannerGroupScope,
    required: PathPlannerGroupScope,
    contains: bool,
) -> None:
    assert available.contains_scope(required) is contains


def test_component_group_scope_selects_runtime_key_from_static_domain():
    scope = ComponentGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )

    assert scope.select_runtime_key("2") == "2"
    with pytest.raises(ValueError, match="do not contain runtime key"):
        scope.select_runtime_key("3")


def test_compilation_rejects_ambiguous_cross_component_artifact_selection():
    input_spec = ArtifactSpec.input("image", ImageArtifactType)

    @artifact_inputs(input_spec)
    def consume(main_image, image):
        del image
        return main_image

    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "1": "/memory/image_channel_1.pkl",
            "2": "/memory/image_channel_2.pkl",
        },
        producer_step_index=1,
        producer_step_name="producer",
    ))
    declarations = ArtifactGraph(
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input("image", ImageArtifactType),
                invocation_keys=(
                    FunctionInvocationKey("consumer", DEFAULT_GROUP_KEY, 0),
                ),
            ),
        )
    )

    snapshot = _snapshot(
        name="consumer",
        func=consume,
        group_by=GroupBy.NONE,
        variable_components=(VariableComponents.SITE,),
    )
    maps = planner.artifacts.compile_plan_maps(
        snapshot,
        3,
        declarations,
        PathPlannerGroupScope.ungrouped(),
    )

    with pytest.raises(ValueError, match="no exact relation-owned selection"):
        planner.artifacts.build_step_compiled_function_pattern(
            snapshot,
            True,
            consume,
            maps.inputs,
            maps.outputs,
            maps.relation_source_scopes,
            maps.group_scope,
        )


def test_compilation_selects_exact_singleton_cross_component_artifact():
    input_spec = ArtifactSpec.input("image", ImageArtifactType)

    @artifact_inputs(input_spec)
    def consume(main_image, image):
        del image
        return main_image

    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="image",
        path="/memory/image_channel_2.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/image_channel_2.pkl"},
        producer_step_index=1,
        producer_step_name="producer",
    ))
    declarations = ArtifactGraph(
        consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input("image", ImageArtifactType),
                invocation_keys=(
                    FunctionInvocationKey("consumer", DEFAULT_GROUP_KEY, 0),
                ),
            ),
        )
    )

    snapshot = _snapshot(
        name="consumer",
        func=consume,
        group_by=GroupBy.NONE,
        variable_components=(VariableComponents.SITE,),
    )
    maps = planner.artifacts.compile_plan_maps(
        snapshot,
        3,
        declarations,
        PathPlannerGroupScope.ungrouped(),
    )

    producer_scope = maps.inputs[input_spec.ref()].producer_group_scope()
    assert producer_scope.keys == ("2",)
    assert producer_scope.component is AllComponents.CHANNEL
    compiled = planner.artifacts.build_step_compiled_function_pattern(
        snapshot,
        True,
        consume,
        maps.inputs,
        maps.outputs,
        maps.relation_source_scopes,
        maps.group_scope,
    )

    edge = next(compiled.iter_invocations()).artifact_input_edges[0]
    assert edge.projection.producer_selection_scope == producer_scope


def test_compilation_accepts_exact_single_artifact_from_another_group():
    owner_spec = ArtifactSpec.input("relationship_owner", ImageArtifactType)
    input_spec = ArtifactSpec.input(
        "relationships",
        RelationshipsArtifactType,
        relations=(InputGroupLineageSourceRelation(owner_spec.ref()),),
    )

    @artifact_inputs(input_spec)
    def consume(main_image, relationships):
        del relationships
        return main_image

    input_plan = ArtifactInputPlan(
        name="relationships",
        path="/memory/relationships_channel_2.pkl",
        artifact_type=RelationshipsArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/relationships_channel_2.pkl"},
    )
    compiled = compile_function_pattern(
        consume,
        {plan.ref(): plan for plan in (input_plan,)},
        {},
    )
    compiled = _artifact_planner_stub().artifacts.compile_invocation_input_edges(
        compiled,
        artifact_inputs={input_plan.ref(): input_plan},
        relation_source_scopes={
            input_spec.ref(): input_plan.producer_group_scope(),
            owner_spec.ref(): PathPlannerGroupScope.from_raw(
                ("2",),
                component=AllComponents.CHANNEL,
            ),
        },
        execution_group_scope=PathPlannerGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet((AllComponents.SITE,)),
    )
    edge = next(compiled.iter_invocations()).artifact_input_edges[0]
    assert edge.projection.invocation_scope.is_ungrouped
    assert edge.projection.producer_selection_scope == (
        input_plan.producer_group_scope()
    )


def test_realized_source_scopes_compile_cross_group_artifact_consumption():
    planner = _artifact_planner_stub()
    planner.plans[4] = CompiledStepPlan(
        step_index=4,
        step_scope_id="plate::functionstep_4",
        step_name="consume",
        step_type="FunctionStep",
        axis_id="A01",
    )
    planner.session.realized_source_metadata = (
        {"source_alias": "Blue", "channel": "1"},
        {"source_alias": "Green", "channel": "2"},
    )
    broad_channel_scope = PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )

    def compile_source_producer(
        *,
        step_index: int,
        alias: str,
        output_name: str,
        expected_group: str,
    ) -> ArtifactSpec:
        source = ArtifactSpec.input(alias, ImageArtifactType)
        output = ArtifactSpec.output_inheriting_group_scope(
            output_name,
            ObjectLabelsArtifactType,
            source,
        )

        @artifact_inputs(source)
        @artifact_outputs(output)
        def produce(image, source_value):
            del source_value
            return image

        source_bindings = StepSourceBindingsConfig(
            enabled=True,
            bindings=(NamedSourceBinding(alias=alias),),
        )
        snapshot = _snapshot(
            index=step_index,
            name=f"produce_{output_name}",
            func=produce,
            source_bindings=source_bindings,
            input_source=InputSource.PIPELINE_START,
        )
        execution_scope = planner.execution_groups.get_execution_groups(
            snapshot,
            PathPlannerComponentScopes.empty(),
            source_bindings=source_bindings,
        )
        declarations = (
            planner.artifacts.namespace_grouped_outputs_for_runtime_consumers(
                produce,
                extract_artifact_declarations(produce),
                execution_scope,
            )
        )
        maps = planner.artifacts.compile_plan_maps(
            snapshot,
            step_index,
            declarations,
            execution_scope,
            source_bindings=source_bindings,
        )

        assert maps.group_scope == PathPlannerGroupScope.from_raw(
            (expected_group,),
            component=AllComponents.CHANNEL,
        )
        return output

    nuclei_output = compile_source_producer(
        step_index=2,
        alias="Blue",
        output_name="Nuclei",
        expected_group="1",
    )
    ph3_output = compile_source_producer(
        step_index=3,
        alias="Green",
        output_name="PH3",
        expected_group="2",
    )
    nuclei_input = nuclei_output.for_plan_type(ArtifactInputPlan)
    ph3_input = ph3_output.for_plan_type(ArtifactInputPlan)
    relationships = ArtifactSpec.output(
        "Nuclei_PH3_relationships",
        RelationshipsArtifactType,
        relations=(
            GroupLineageSourceRelation(nuclei_input.ref()),
            GroupLineageSourceRelation(ph3_input.ref()),
        ),
    )

    @artifact_inputs(nuclei_input, ph3_input)
    @artifact_outputs(relationships)
    def consume(image, Nuclei, PH3):
        del Nuclei, PH3
        return image

    snapshot = _snapshot(index=4, name="consume", func=consume)
    declarations = planner.artifacts.namespace_grouped_outputs_for_runtime_consumers(
        consume,
        extract_artifact_declarations(consume),
        broad_channel_scope,
    )
    maps = planner.artifacts.compile_plan_maps(
        snapshot,
        4,
        declarations,
        broad_channel_scope,
    )
    compiled = compile_function_pattern(consume, maps.inputs, maps.outputs)
    compiled = planner.artifacts.compile_invocation_input_edges(
        compiled,
        artifact_inputs=maps.inputs,
        relation_source_scopes=maps.relation_source_scopes,
        execution_group_scope=maps.group_scope,
        consumer_variable_components=ComponentSet((AllComponents.SITE,)),
    )

    assert maps.group_scope == PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )
    edges = {
        edge.spec.name: edge
        for edge in next(compiled.iter_invocations()).artifact_input_edges
    }
    assert edges["Nuclei"].projection.producer_selection_scope.keys == ("1",)
    assert edges["PH3"].projection.producer_selection_scope.keys == ("2",)


def test_compilation_rejects_declared_lineage_from_multi_group_producer():
    source_spec = ArtifactSpec.input("site_source", ImageArtifactType)
    image_spec = ArtifactSpec.input(
        "image",
        ImageArtifactType,
        relations=(InputGroupLineageSourceRelation(source_spec.ref()),),
    )

    @artifact_inputs(image_spec, source_spec)
    def consume(main_image, image, site_source):
        del image, site_source
        return main_image

    image_plan = ArtifactInputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "1": "/memory/image_channel_1.pkl",
            "2": "/memory/image_channel_2.pkl",
        },
    )
    source_plan = ArtifactInputPlan(
        name="site_source",
        path="/memory/site_source.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.SITE,
        paths_by_group={"1": "/memory/site_source_1.pkl"},
    )
    input_plans = {
        image_plan.ref(): image_plan,
        source_plan.ref(): source_plan,
    }
    compiled = compile_function_pattern(consume, input_plans, {})
    planner = _artifact_planner_stub()

    with pytest.raises(ValueError, match="no exact relation-owned selection"):
        planner.artifacts.compile_invocation_input_edges(
            compiled,
            artifact_inputs=input_plans,
            relation_source_scopes={
                image_spec.ref(): image_plan.producer_group_scope(),
                source_spec.ref(): source_plan.producer_group_scope(),
            },
            execution_group_scope=PathPlannerGroupScope.ungrouped(),
            consumer_variable_components=ComponentSet((AllComponents.SITE,)),
        )


def test_artifact_plan_rejects_group_component_as_variable_axis():
    with pytest.raises(ValueError, match="cannot group by.*site.*variable component"):
        ArtifactInputPlan(
            name="objects",
            path="/memory/objects.pkl",
            artifact_type=ObjectLabelsArtifactType,
            group_keys=(None,),
            group_component=AllComponents.SITE,
            variable_components=(AllComponents.SITE,),
        )


def test_artifact_input_plan_preserves_multi_grouped_producer_across_components():
    planner = _artifact_planner_stub()
    _record_declared_output(planner, ArtifactOutputPlan(
        name="illumination",
        path="/memory/illumination_channel_1.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "1": "/memory/illumination_channel_1.pkl",
            "2": "/memory/illumination_channel_2.pkl",
        },
        producer_step_index=1,
        producer_step_name="calculate_illumination",
    ))

    inputs = planner.artifacts.process_artifact_inputs(
        ArtifactGraph(
            consumers=(
                ArtifactConsumer(
                    spec=ArtifactSpec.input("illumination", ImageArtifactType),
                    invocation_keys=(),
                ),
            )
        ),
        consumer_scope=PathPlannerGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.SITE,
        ),
        sid=2,
        step_name="apply_illumination",
        source_bindings=EMPTY_SOURCE_BINDINGS,
        variable_components=ComponentSet(),
    execution_scope = FunctionStepExecutionScope.AXIS)

    illumination_ref = ArtifactSpec.input("illumination", ImageArtifactType).ref()
    plan = inputs[illumination_ref]
    assert plan.group_keys == ("1", "2")
    assert plan.group_component is AllComponents.CHANNEL
    assert plan.paths_by_group == {
        "1": "/memory/illumination_channel_1.pkl",
        "2": "/memory/illumination_channel_2.pkl",
    }


def test_realized_component_domain_does_not_replace_dynamic_projection_coordinate():
    source_spec = ArtifactSpec.input("source", ImageArtifactType)
    illumination_spec = ArtifactSpec.input(
        "illumination",
        ImageArtifactType,
        relations=(InputStackBroadcastSourceRelation(source_spec.ref()),),
    )

    @artifact_inputs(illumination_spec, source_spec)
    def apply(image, illumination):
        del illumination
        return image

    input_plan = ArtifactInputPlan(
        name="illumination",
        path="/memory/illumination_channel_1.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        variable_components=(AllComponents.SITE,),
        paths_by_group={
            "1": "/memory/illumination_channel_1.pkl",
            "2": "/memory/illumination_channel_2.pkl",
        },
    )
    compiled = compile_function_pattern(
        apply,
        {plan.ref(): plan for plan in (input_plan,)},
        {},
    )
    planner = _artifact_planner_stub()
    planner.session.realized_source_metadata = tuple(
        {"source_alias": "source", "site": str(site)} for site in range(1, 4)
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(NamedSourceBinding(alias="source"),),
    )

    compiled = planner.artifacts.compile_invocation_input_edges(
        compiled,
        artifact_inputs={input_plan.ref(): input_plan},
        relation_source_scopes={
            illumination_spec.ref(): input_plan.producer_group_scope(),
            source_spec.ref(): PathPlannerGroupScope.dynamic(AllComponents.SITE),
        },
        execution_group_scope=PathPlannerGroupScope.dynamic(AllComponents.SITE),
        consumer_variable_components=ComponentSet((AllComponents.CHANNEL,)),
        source_bindings=source_bindings,
        available_artifacts=ArtifactSpecCollection((source_spec, illumination_spec)),
    )
    edge = next(compiled.iter_invocations()).artifact_input_edges[0]

    assert edge.storage_plan.producer_group_scope() == input_plan.producer_group_scope()
    assert edge.projection.producer_selection_scope == (
        edge.storage_plan.producer_group_scope()
    )
    assert edge.projection.invocation_scope == ComponentGroupScope.dynamic(
        AllComponents.SITE
    )
    assert edge.projection.component_scope(AllComponents.SITE) == (
        ComponentGroupScope.dynamic(AllComponents.SITE)
    )


def test_runtime_selects_inputs_from_exact_grouped_invocation_edges():
    planner = _artifact_planner_stub()
    first_spec = ArtifactSpec.input("IllumStain1", ImageArtifactType)
    second_spec = ArtifactSpec.input("IllumStain2", ImageArtifactType)

    @artifact_inputs(first_spec)
    def apply_first(image):
        return image

    @artifact_inputs(second_spec)
    def apply_second(image):
        return image

    first = ArtifactInputPlan(
        name="IllumStain1",
        path="/memory/IllumStain1.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
    )
    second = ArtifactInputPlan(
        name="IllumStain2",
        path="/memory/IllumStain2.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
    )
    execution_scope = PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )
    storage = {first.ref(): first, second.ref(): second}
    compiled = compile_function_pattern(
        {"1": apply_first, "2": apply_second},
        storage,
        {},
    )
    compiled = planner.artifacts.compile_invocation_input_edges(
        compiled,
        artifact_inputs=storage,
        relation_source_scopes={
            first_spec.ref(): first.producer_group_scope(),
            second_spec.ref(): second.producer_group_scope(),
        },
        execution_group_scope=execution_scope,
        consumer_variable_components=ComponentSet((AllComponents.SITE,)),
    )
    execution_plan = CompiledStepPlan(
        step_index=2,
        step_scope_id="plate::functionstep_2",
        step_name="CorrectIlluminationApply",
        step_type="FunctionStep",
        axis_id="A01",
        artifact_inputs=storage,
        execution_group_scope=execution_scope,
        compiled_function_pattern=compiled,
    )
    component_plans = ComponentArtifactPlans.from_step_component(execution_plan, "1")
    invocations = tuple(compiled.iter_invocations())
    first_plans = component_plans.select_for_invocation(
        invocations[0],
        execution_scope=execution_scope,
        component_key="1",
    )
    second_plans = component_plans.select_for_invocation(
        invocations[1],
        execution_scope=execution_scope,
        component_key="1",
    )

    assert tuple(
        edge.storage_plan.name for edge in first_plans.inputs.values()
        if edge.storage_plan is not None
    ) == (first.name,)
    assert tuple(
        edge.storage_plan.name for edge in second_plans.inputs.values()
        if edge.storage_plan is not None
    ) == (second.name,)
    first_edge = next(iter(first_plans.inputs.values()))
    second_edge = next(iter(second_plans.inputs.values()))
    assert first_edge.projection is not None
    assert second_edge.projection is not None
    assert first_edge.projection.invocation_scope == (
        ComponentGroupScope.from_raw(
            ("1",),
            component=AllComponents.CHANNEL,
        )
    )
    assert second_edge.projection.invocation_scope == (
        ComponentGroupScope.from_raw(
            ("2",),
            component=AllComponents.CHANNEL,
        )
    )


def test_grouped_invocation_is_independent_of_source_artifact_domain():
    input_spec = ArtifactSpec.input("Objects1", ObjectLabelsArtifactType)

    @artifact_inputs(input_spec)
    def consume(image, Objects1):
        del Objects1
        return image

    input_plan = ArtifactInputPlan(
        name=input_spec.name,
        path="/memory/Objects1.pkl",
        artifact_type=input_spec.artifact_type,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
    )
    invocation_scope = PathPlannerGroupScope.from_raw(
        ("1",),
        component=AllComponents.CHANNEL,
    )
    producer_domain = PathPlannerGroupScope.from_raw(
        ("3",),
        component=AllComponents.CHANNEL,
    )
    invocation = next(
        compile_function_pattern(
            {"1": consume},
            {plan.ref(): plan for plan in (input_plan,)},
            {},
        ).iter_invocations()
    )

    component_scopes = PathPlannerArtifactStage.exact_component_scopes(
        (producer_domain,),
        (),
        component_domains=(producer_domain,),
        invocation_scope=invocation_scope,
        invocation=invocation,
        artifact_ref=input_spec.ref(),
    )

    assert component_scopes == (producer_domain,)


def test_fixed_producer_coordinate_precedes_consumer_group_lineage():
    input_spec = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)

    @artifact_inputs(input_spec)
    def consume(image, Nuclei):
        del Nuclei
        return image

    input_plan = ArtifactInputPlan(
        name=input_spec.name,
        path="/memory/Nuclei.pkl",
        artifact_type=input_spec.artifact_type,
    )
    invocation = next(
        compile_function_pattern(
            {"2": consume},
            {plan.ref(): plan for plan in (input_plan,)},
            {},
        ).iter_invocations()
    )
    producer_channel = PathPlannerGroupScope.from_raw(
        ("1",),
        component=AllComponents.CHANNEL,
    )
    consumer_channel = PathPlannerGroupScope.from_raw(
        ("2",),
        component=AllComponents.CHANNEL,
    )

    component_scopes = PathPlannerArtifactStage.exact_component_scopes(
        (producer_channel,),
        (consumer_channel,),
        component_domains=(producer_channel,),
        invocation_scope=consumer_channel,
        invocation=invocation,
        artifact_ref=input_spec.ref(),
    )

    assert component_scopes == (producer_channel,)


def test_relation_selects_one_coordinate_from_multi_coordinate_producer_domain():
    input_spec = ArtifactSpec.input("Objects", ObjectLabelsArtifactType)

    @artifact_inputs(input_spec)
    def consume(image, Objects):
        del Objects
        return image

    input_plan = ArtifactInputPlan(
        name=input_spec.name,
        path="/memory/Objects.pkl",
        artifact_type=input_spec.artifact_type,
    )
    invocation = next(
        compile_function_pattern(
            {"2": consume},
            {plan.ref(): plan for plan in (input_plan,)},
            {},
        ).iter_invocations()
    )
    producer_domain = PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )
    selected_channel = PathPlannerGroupScope.from_raw(
        ("2",),
        component=AllComponents.CHANNEL,
    )

    component_scopes = PathPlannerArtifactStage.exact_component_scopes(
        (producer_domain,),
        (selected_channel,),
        component_domains=(producer_domain,),
        invocation_scope=selected_channel,
        invocation=invocation,
        artifact_ref=input_spec.ref(),
    )

    assert component_scopes == (selected_channel,)


def test_produced_artifact_projection_does_not_revalidate_source_binding_domain():
    source = ArtifactSpec.input("OrigMito", ImageArtifactType)
    consumed = ArtifactSpec.input(
        "Cells",
        ObjectLabelsArtifactType,
        relations=(InputGroupLineageSourceRelation(source.ref()),),
    )

    @artifact_inputs(consumed)
    def relate_objects(image, Cells):
        del Cells
        return image

    input_plan = ArtifactInputPlan(
        name=consumed.name,
        path="/memory/Cells.pkl",
        artifact_type=consumed.artifact_type,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        source_step_id=4,
    )
    compiled = compile_function_pattern(
        {"5": relate_objects},
        {plan.ref(): plan for plan in (input_plan,)},
        {},
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias=source.name,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "5"),),
            ),
        ),
    )

    compiled = _artifact_planner_stub().artifacts.compile_invocation_input_edges(
        compiled,
        artifact_inputs={input_plan.ref(): input_plan},
        relation_source_scopes={source.ref(): input_plan.producer_group_scope()},
        execution_group_scope=PathPlannerGroupScope.from_raw(
            ("5",),
            component=AllComponents.CHANNEL,
        ),
        consumer_variable_components=ComponentSet((AllComponents.SITE,)),
        source_bindings=source_bindings,
        available_artifacts=ArtifactSpecCollection((source, consumed)),
    )

    edge = next(compiled.iter_invocations()).artifact_input_edges[0]
    assert edge.projection.component_scope(AllComponents.CHANNEL) == (
        ComponentGroupScope.from_raw(("3",), component=AllComponents.CHANNEL)
    )


def test_non_dict_artifact_input_uses_function_step_execution_scope():
    input_spec = ArtifactSpec.input("positions", SpecialArtifactType)

    @artifact_inputs(input_spec)
    def assemble(image, positions):
        del positions
        return image

    input_plan = ArtifactInputPlan(
        name="positions",
        path="/memory/positions.pkl",
        artifact_type=SpecialArtifactType,
        group_keys=(None,),
        group_component=AllComponents.CHANNEL,
    )
    execution_scope = PathPlannerGroupScope.dynamic(AllComponents.CHANNEL)
    compiled = compile_function_pattern(
        assemble,
        {plan.ref(): plan for plan in (input_plan,)},
        {},
    )

    compiled = _artifact_planner_stub().artifacts.compile_invocation_input_edges(
        compiled,
        artifact_inputs={input_plan.ref(): input_plan},
        relation_source_scopes={
            input_spec.ref(): input_plan.producer_group_scope(),
        },
        execution_group_scope=execution_scope,
        consumer_variable_components=ComponentSet((AllComponents.SITE,)),
    )

    edge = next(compiled.iter_invocations()).artifact_input_edges[0]
    expected_scope = ComponentGroupScope.dynamic(AllComponents.CHANNEL)
    assert edge.projection.invocation_scope == expected_scope
    assert edge.projection.producer_selection_scope == expected_scope


def test_grouped_invocations_keep_distinct_edges_for_same_artifact_ref():
    planner = _artifact_planner_stub()
    mask_name = "CropBlue__crop_mask"
    mask_spec = ArtifactSpec.input(mask_name, ImageArtifactType)

    @artifact_inputs(mask_spec)
    def crop_green(image):
        return image

    @artifact_inputs(mask_spec)
    def crop_red(image):
        return image

    input_plan = ArtifactInputPlan(
        name=mask_name,
        path="/memory/crop_mask.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("2", "3"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "2": "/memory/crop_mask_2.pkl",
            "3": "/memory/crop_mask_3.pkl",
        },
    )
    compiled = compile_function_pattern(
        {"2": crop_green, "3": crop_red},
        {plan.ref(): plan for plan in (input_plan,)},
        {},
    )
    compiled = planner.artifacts.compile_invocation_input_edges(
        compiled,
        artifact_inputs={input_plan.ref(): input_plan},
        relation_source_scopes={
            mask_spec.ref(): input_plan.producer_group_scope(),
        },
        execution_group_scope=PathPlannerGroupScope.from_raw(
            ("2", "3"),
            component=AllComponents.CHANNEL,
        ),
        consumer_variable_components=ComponentSet((AllComponents.SITE,)),
    )
    edges = compiled.artifact_input_edges_by_key()

    assert len(edges) == 2
    assert tuple(
        edge.projection.producer_selection_scope.keys for edge in edges.values()
    ) == (("2",), ("3",))
    assert len({key.invocation_key for key in edges}) == 2
    assert {edge.spec.ref() for edge in edges.values()} == {mask_spec.ref()}


def test_main_input_dependency_uses_scope_identity_for_step_output_edges():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plans = {
        0: CompiledStepPlan(
            step_index=0,
            step_scope_id="plate::functionstep_0",
            step_name="load",
            step_type="FunctionStep",
            axis_id="A01",
            output_dir=Path("/data/plate1_processed/images"),
        ),
        1: CompiledStepPlan(
            step_index=1,
            step_scope_id="plate::functionstep_1",
            step_name="measure",
            step_type="FunctionStep",
            axis_id="A01",
        ),
    }
    snapshots_by_index = {
        0: _snapshot(scope_id="plate::functionstep_0"),
        1: _snapshot(scope_id="plate::functionstep_1"),
    }
    planner.session = SimpleNamespace(
        snapshot=lambda index: snapshots_by_index[index],
    )
    planner.steps = PathPlannerStepAssemblyStage(planner)

    dependency = planner.steps.main_input_dependency(
        _snapshot(input_source=None, is_function_step=False),
        1,
    )

    assert dependency.kind is StepInputDependencyKind.STEP_OUTPUT
    assert dependency.source_step_index == 0
    assert dependency.source_step_scope_id == "plate::functionstep_0"

    input_dir, output_dir = planner.steps.step_io_dirs(dependency, 1)
    assert input_dir == Path("/data/plate1_processed/images")
    assert output_dir == Path("/data/plate1_processed/images")


def test_main_input_dependency_uses_declared_artifact_producer_not_previous_step():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plans = {
        index: CompiledStepPlan(
            step_index=index,
            step_scope_id=f"plate::functionstep_{index}",
            step_name=name,
            step_type="FunctionStep",
            axis_id="A01",
        )
        for index, name in enumerate(("CropBlue", "CropRed", "Identify"))
    }
    planner.artifact_context = ArtifactDeclarationStepContext.empty()
    crop_blue = ArtifactOutputPlan(
            name="CropBlue",
            path="/memory/CropBlue.pkl",
            artifact_type=ImageArtifactType,
            producer_step_index=0,
            producer_step_scope_id="plate::functionstep_0",
        )
    crop_red = ArtifactOutputPlan(
            name="CropRed",
            path="/memory/CropRed.pkl",
            artifact_type=ImageArtifactType,
            producer_step_index=1,
            producer_step_scope_id="plate::functionstep_1",
        )
    planner.declared = {plan.ref(): plan for plan in (crop_blue, crop_red)}
    planner.steps = PathPlannerStepAssemblyStage(planner)
    declarations = ArtifactGraph(
        non_plan_consumers=(
            ArtifactConsumer(
                spec=ArtifactSpec.input("CropBlue", ImageArtifactType),
                invocation_keys=(),
            ),
        )
    )

    dependency = planner.steps.main_input_dependency(
        _snapshot(
            input_source=InputSource.PREVIOUS_STEP,
            is_function_step=True,
            name="Identify",
        ),
        2,
        declarations=declarations,
    )

    assert dependency == StepInputDependency.step_output(
        source_step_index=0,
        source_step_scope_id="plate::functionstep_0",
    )


def test_main_input_dependency_skips_main_flow_preserving_steps():
    measurement_spec = ArtifactSpec.output(
        "Measurements",
        MeasurementsArtifactType,
    )

    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_outputs=True,
    )
    @artifact_outputs(measurement_spec)
    def measure(image, *, runtime):
        del runtime
        return image, object()

    compiled_contract = CallableContract.from_callable(measure)
    preserving_pattern = CompiledFunctionPattern(
        groups=(
            CompiledFunctionGroup(
                group_key="default",
                invocations=(
                    CompiledFunctionInvocation(
                        key=FunctionInvocationKey.from_contract(
                            compiled_contract,
                            "default",
                            0,
                        ),
                        contract=compiled_contract,
                    ),
                ),
            ),
        ),
        is_grouped=False,
    )
    source_dependency = StepInputDependency.step_output(
        source_step_index=0,
        source_step_scope_id="plate::functionstep_0",
    )
    planner = PathPlanner.__new__(PathPlanner)
    planner.plans = {
        0: CompiledStepPlan(
            step_index=0,
            step_scope_id="plate::functionstep_0",
            step_name="load",
            step_type="FunctionStep",
            axis_id="A01",
        ),
        1: CompiledStepPlan(
            step_index=1,
            step_scope_id="plate::functionstep_1",
            step_name="measure",
            step_type="FunctionStep",
            axis_id="A01",
            main_input_dependency=source_dependency,
            compiled_function_pattern=preserving_pattern,
        ),
        2: CompiledStepPlan(
            step_index=2,
            step_scope_id="plate::functionstep_2",
            step_name="consume",
            step_type="FunctionStep",
            axis_id="A01",
        ),
    }
    planner.session = SimpleNamespace(
        snapshot=lambda index: _snapshot(scope_id=f"plate::functionstep_{index}"),
    )
    planner.steps = PathPlannerStepAssemblyStage(planner)

    dependency = planner.steps.main_input_dependency(
        _snapshot(input_source=None, is_function_step=False),
        2,
    )

    assert dependency == source_dependency


def test_main_input_dependency_preserves_pipeline_start_edges():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plans = {
        1: CompiledStepPlan(
            step_index=1,
            step_scope_id="plate::functionstep_1",
            step_name="qc",
            step_type="FunctionStep",
            axis_id="A01",
        )
    }
    planner.initial_input = Path("/data/plate1/images")
    planner.session = SimpleNamespace(
        snapshot=lambda index: {1: _snapshot(scope_id="plate::functionstep_1")}[index],
    )
    planner.paths = SimpleNamespace(
        build_output_path=lambda *_args, **_kwargs: Path(
            "/data/plate1_processed/images"
        )
    )
    planner.steps = PathPlannerStepAssemblyStage(planner)

    dependency = planner.steps.main_input_dependency(
        _snapshot(
            input_source=InputSource.PIPELINE_START,
            is_function_step=False,
        ),
        1,
    )

    assert dependency.kind is StepInputDependencyKind.PIPELINE_START
    input_dir, output_dir = planner.steps.step_io_dirs(dependency, 1)
    assert input_dir == Path("/data/plate1/images")
    assert output_dir == Path("/data/plate1_processed/images")
