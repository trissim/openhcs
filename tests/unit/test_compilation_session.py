from inspect import signature
from types import SimpleNamespace

import pytest

from objectstate.lazy_factory import ensure_global_config_context
from objectstate.object_state import ObjectState
from objectstate.object_state_registry import ObjectStateRegistry
from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import ArtifactSpec, ImageArtifactType
from openhcs.core.callable_contract import CallableContract
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyNapariStreamingConfig,
    LazySourceBindingsConfig,
    LazyStepSourceBindingsConfig,
    PipelineConfig,
    ProcessingConfig,
    StepMaterializationConfig,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import (
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.compiler import AxisCompilationRequest, PipelineCompiler
from openhcs.core.pipeline.function_contracts import artifact_inputs
from openhcs.core.pipeline.path_planner import (
    PathPlanner,
    PathPlannerArtifactStage,
    PathPlannerExecutionGroups,
)
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.source_bindings import (
    EMPTY_SOURCE_BINDINGS,
    ComponentSelector,
    MetadataExtractionRule,
    MetadataSource,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    neurite_outgrowth_metaxpress,
)


def _identity(image):
    return image


def test_axis_session_initialization_requires_pipeline_resolved_state() -> None:
    parameters = signature(
        PipelineCompiler.initialize_step_plans_for_context
    ).parameters

    assert "step_state_map" in parameters
    assert "step_snapshots" in parameters
    assert "steps_already_resolved" not in parameters
    assert "_resolve_steps_for_context" not in vars(PipelineCompiler)


@artifact_inputs(
    ArtifactSpec.input(
        "DNA",
        ImageArtifactType,
        parameter_name="image",
    )
)
def _external_source_consumer(image):
    return image


@artifact_inputs(
    ArtifactSpec.input(
        "DNA",
        ImageArtifactType,
        parameter_name="dna",
    )
)
def _previous_step_and_external_source(image, dna):
    return image, dna


def _snapshot(
    step: FunctionStep,
    index: int,
    variable_components=(VariableComponents.SITE,),
    source_bindings=EMPTY_SOURCE_BINDINGS,
    input_source: InputSource = InputSource.PREVIOUS_STEP,
) -> StepSnapshot:
    step.source_bindings = source_bindings
    step.processing_config = ProcessingConfig(
        variable_components=list(variable_components),
        group_by=GroupBy.NONE,
        input_source=input_source,
    )
    step.step_materialization_config = StepMaterializationConfig(enabled=False)
    return StepSnapshot(
        index=index,
        scope_id=f"plate::functionstep_{index}",
        step=step,
    )


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        axis_id="A01",
        plate_path=None,
        current_sequential_combination=None,
        step_plans={
            0: CompiledStepPlan(
                step_index=0,
                step_name="step",
                step_type="FunctionStep",
                axis_id="A01",
            )
        },
    )


def _orchestrator(pipeline_config: PipelineConfig | None = None) -> SimpleNamespace:
    return SimpleNamespace(pipeline_config=pipeline_config or PipelineConfig())


def _compile_source_plans_for_contract(
    session: CompilationSession,
    snapshot: StepSnapshot,
    func,
    main_input_dependency: StepInputDependency = StepInputDependency.pipeline_start(),
):
    planner = SimpleNamespace(
        session=session,
        artifact_context=ArtifactDeclarationStepContext.empty(),
        source_bindings_for_snapshot=(lambda value: value.step.source_bindings),
    )
    stage = PathPlannerArtifactStage(planner)
    execution_bindings = stage.source_bindings_for_contracts(
        snapshot,
        (CallableContract.from_callable(func),),
        main_input_dependency,
    )
    return execution_bindings, stage.compile_source_plans(
        snapshot,
        execution_bindings,
    )


class _EffectiveConfigContextOrchestrator:
    def create_context(self, axis_id: str) -> ProcessingContext:
        return ProcessingContext(
            axis_id=axis_id,
            auto_add_output_plate_to_plate_manager=True,
        )


def test_axis_compilation_request_preserves_effective_auto_add_flag():
    request = AxisCompilationRequest(
        orchestrator=_EffectiveConfigContextOrchestrator(),
        global_config=GlobalPipelineConfig(auto_add_output_plate_to_plate_manager=True),
        pipeline_config=PipelineConfig(),
        pipeline=SimpleNamespace(),
        path_resolver=SimpleNamespace(),
        global_step_axis_filters={},
        enable_visualizer_override=False,
        is_zmq_execution=True,
    )

    context = request.context_for("A01")

    assert context.auto_add_output_plate_to_plate_manager is True
    assert context.source_image_set_identity_policy.plane_member_components == (
        frozenset((AllComponents.CHANNEL,))
    )


def test_compilation_session_owns_step_snapshot_plan_invariants():
    step = FunctionStep(func=_identity, name="step")
    step_state = object()
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: step_state},
        snapshots=(_snapshot(step, 0),),
    )

    assert session.axis_id == "A01"
    assert session.step(0) is step
    assert session.step_state(0) is step_state
    assert session.snapshot(0).step.name == "step"
    assert session.plan(0).step_name == "step"


def test_compilation_session_rejects_missing_snapshot():
    step = FunctionStep(func=_identity, name="step")

    with pytest.raises(ValueError, match="one StepSnapshot per step"):
        CompilationSession.from_context(
            context=_context(),
            steps=[step],
            orchestrator=_orchestrator(),
            global_config=GlobalPipelineConfig(),
            step_state_map={0: object()},
            snapshots=(),
        )


def test_compilation_session_rejects_non_contiguous_snapshot_index():
    step = FunctionStep(func=_identity, name="step")

    with pytest.raises(ValueError, match="index mismatch"):
        CompilationSession.from_context(
            context=_context(),
            steps=[step],
            orchestrator=_orchestrator(),
            global_config=GlobalPipelineConfig(),
            step_state_map={0: object()},
            snapshots=(_snapshot(step, 1),),
        )


def test_compiler_keeps_variable_components_as_stack_source():
    step = FunctionStep(func=_identity, name="step")
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(
            _snapshot(
                step,
                0,
                variable_components=(VariableComponents.CHANNEL,),
            ),
        ),
    )

    PipelineCompiler._supplement_step_plans(session)

    assert session.plan(0).variable_components == [VariableComponents.CHANNEL]


def test_path_planner_source_binding_plan_comes_from_objectstate_snapshot():
    ObjectStateRegistry.clear()
    metadata_rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=r"(?P<well>[A-H]\d{2})\.tif",
    )
    match_plan = SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER)
    binding = NamedSourceBinding(
        alias="DNA",
        selector=SourceSelector(
            components=(ComponentSelector(AllComponents.CHANNEL, "1"),)
        ),
    )
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    pipeline_state = ObjectState(
        PipelineConfig(
            source_bindings_config=LazySourceBindingsConfig(
                metadata_rules=(metadata_rule,),
                match_plan=match_plan,
            ),
        ),
        scope_id="plate",
    )

    try:
        ObjectStateRegistry.register(pipeline_state, _skip_snapshot=True)
        step = FunctionStep(
            func=_external_source_consumer,
            name="source-bound",
            source_bindings=LazyStepSourceBindingsConfig(
                bindings=(binding,),
                enabled=True,
            ),
        )
        step_state = ObjectState(
            step,
            scope_id="plate::functionstep_0",
            parent_state=pipeline_state,
            exclude_params=["func"],
        )
        ObjectStateRegistry.register(step_state, _skip_snapshot=True)
        resolved_step = step_state.to_saved_resolved_object()
        snapshot = StepSnapshot(
            index=0,
            scope_id=step_state.scope_id,
            step=resolved_step,
        )
        session = CompilationSession.from_context(
            context=_context(),
            steps=[resolved_step],
            orchestrator=_orchestrator(pipeline_state.to_object()),
            global_config=GlobalPipelineConfig(),
            step_state_map={0: step_state},
            snapshots=(snapshot,),
        )

        execution_bindings, (source_binding_plan, _source_universe_plan) = (
            _compile_source_plans_for_contract(
                session,
                snapshot,
                _external_source_consumer,
            )
        )
    finally:
        ObjectStateRegistry.clear()

    assert snapshot.step.source_bindings.bindings == (binding,)
    assert snapshot.step.source_bindings.metadata_rules == (metadata_rule,)
    assert snapshot.step.source_bindings.match_plan == match_plan
    assert execution_bindings.bindings == (binding,)
    assert source_binding_plan.bindings == (binding,)
    assert source_binding_plan.metadata_rules == (metadata_rule,)
    assert source_binding_plan.match_plan == match_plan


def test_compiler_streaming_config_snapshot_preserves_inherited_port():
    ObjectStateRegistry.clear()
    global_config = GlobalPipelineConfig(
        napari_streaming_config=LazyNapariStreamingConfig(
            enabled=True,
            persistent=False,
        ),
    )
    ensure_global_config_context(GlobalPipelineConfig, global_config)
    global_state = ObjectState(global_config, scope_id="")
    pipeline_state = ObjectState(
        PipelineConfig(),
        scope_id="plate",
        parent_state=global_state,
    )

    try:
        ObjectStateRegistry.register(global_state, _skip_snapshot=True)
        ObjectStateRegistry.register(pipeline_state, _skip_snapshot=True)
        step = FunctionStep(func=_identity, name="streamed")
        step_state = ObjectState(
            step,
            scope_id="plate::functionstep_0",
            parent_state=pipeline_state,
            exclude_params=["func"],
        )
        ObjectStateRegistry.register(step_state, _skip_snapshot=True)
        resolved_step = step_state.to_saved_resolved_object()
        snapshot = StepSnapshot(
            index=0,
            scope_id=step_state.scope_id,
            step=resolved_step,
        )
        context = _context()
        context.required_visualizers = []
        session = CompilationSession.from_context(
            context=context,
            steps=[resolved_step],
            orchestrator=_orchestrator(pipeline_state.to_object()),
            global_config=global_config,
            step_state_map={0: step_state},
            snapshots=(snapshot,),
        )

        PipelineCompiler._collect_streaming_configs(session)
    finally:
        ObjectStateRegistry.clear()

    assert context.required_visualizers
    required = context.required_visualizers[0]
    assert required.config.enabled is True
    assert required.config.persistent is False
    assert required.config.port == 5555
    assert session.plan(0).streaming_configs["napari_streaming_config"].port == 5555


def test_compiler_disabled_source_bindings_stay_inert_without_contract_requirement():
    binding = NamedSourceBinding(alias="DNA")
    step = FunctionStep(func=_identity, name="source-bound")
    snapshot = _snapshot(
        step,
        0,
        source_bindings=StepSourceBindingsConfig(bindings=(binding,)),
    )
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )

    PipelineCompiler._supplement_step_plans(session)

    assert session.plan(0).source_binding_plan.is_empty


def test_path_planner_activates_declared_source_binding_for_pipeline_start():
    binding = NamedSourceBinding(alias="DNA")
    step = FunctionStep(func=_external_source_consumer, name="source-bound")
    snapshot = _snapshot(
        step,
        0,
        source_bindings=StepSourceBindingsConfig(bindings=(binding,)),
        input_source=InputSource.PIPELINE_START,
    )
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )

    _execution_bindings, (source_binding_plan, _source_universe_plan) = (
        _compile_source_plans_for_contract(
            session,
            snapshot,
            _external_source_consumer,
        )
    )

    assert source_binding_plan.bindings == (binding,)


def test_plate_export_contract_construction_projects_inputs_to_runtime_batch():
    from openhcs.interop.cellprofiler.compile_time_contracts import (
        CellProfilerInvocationContractProviderFactory,
    )
    from openhcs.processing.backends.cellprofiler import export_to_database

    binding = NamedSourceBinding(alias="DNA")
    step = FunctionStep(func=export_to_database, name="ExportToDatabase")
    snapshot = _snapshot(
        step,
        0,
        variable_components=(),
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(binding,),
        ),
        input_source=InputSource.PIPELINE_START,
    )
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )
    dependency_before_provider = session.plan(0).main_input_dependency

    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )
    invocation = next(normalize_function_pattern(step.func).iter_items())
    assert provider is not None
    plan = provider(
        invocation,
        ArtifactDeclarationStepContext(
            step_name=step.name,
            step_index=0,
            source_bindings=step.source_bindings,
            group_by=step.processing_config.group_by,
            input_source=step.processing_config.input_source,
        ),
    )

    assert plan is not None
    assert dependency_before_provider == StepInputDependency.unresolved()
    assert session.plan(0).main_input_dependency == dependency_before_provider
    assert plan.contract.artifact_inputs.names() == ("DNA",)


def test_path_planner_preserves_pipeline_start_bindings_for_implicit_main_flow():
    binding = NamedSourceBinding(alias="DNA")
    step = FunctionStep(func=_identity, name="source-bound measurement")
    snapshot = _snapshot(
        step,
        0,
        source_bindings=StepSourceBindingsConfig(bindings=(binding,)),
        input_source=InputSource.PIPELINE_START,
    )
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )
    execution_bindings, (source_binding_plan, source_universe_plan) = (
        _compile_source_plans_for_contract(session, snapshot, _identity)
    )

    assert execution_bindings.bindings == (binding,)
    assert source_binding_plan.bindings == (binding,)
    assert source_universe_plan == source_universe_plan.empty()


def test_path_planner_step_output_projects_only_exact_source_artifacts() -> None:
    dna = NamedSourceBinding(alias="DNA")
    unrelated = NamedSourceBinding(alias="OrigRed")
    step = FunctionStep(
        func=_previous_step_and_external_source,
        name="previous-step plus exact source",
    )
    snapshot = _snapshot(
        step,
        0,
        source_bindings=StepSourceBindingsConfig(bindings=(dna, unrelated)),
    )
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )

    execution_bindings, (source_binding_plan, _source_universe_plan) = (
        _compile_source_plans_for_contract(
            session,
            snapshot,
            _previous_step_and_external_source,
            StepInputDependency.step_output(
                source_step_index=0,
                source_step_scope_id="plate::functionstep_0",
            ),
        )
    )

    assert execution_bindings.bindings == (dna,)
    assert source_binding_plan.bindings == (dna,)


def test_path_planner_preserves_metaxpress_primary_source_order() -> None:
    pipeline_bindings = (
        NamedSourceBinding(alias="Hoechst"),
        NamedSourceBinding(alias="MAP2"),
        NamedSourceBinding(alias="SMI312"),
    )
    selected_bindings = (
        NamedSourceBinding(alias="SMI312"),
        NamedSourceBinding(alias="Hoechst"),
    )
    step = FunctionStep(func=neurite_outgrowth_metaxpress, name="neurite")
    snapshot = _snapshot(
        step,
        0,
        variable_components=(VariableComponents.CHANNEL,),
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=selected_bindings,
        ),
        input_source=InputSource.PIPELINE_START,
    )
    pipeline_config = PipelineConfig(
        source_bindings_config=LazySourceBindingsConfig(bindings=pipeline_bindings)
    )
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(pipeline_config),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )
    contract = CallableContract.from_callable(neurite_outgrowth_metaxpress)

    execution_bindings, (source_binding_plan, _source_universe_plan) = (
        _compile_source_plans_for_contract(
            session,
            snapshot,
            neurite_outgrowth_metaxpress,
        )
    )

    assert contract.accepts_implicit_main_flow_input is True
    assert contract.artifact_inputs.names() == ("pixel_size",)
    assert tuple(binding.alias for binding in pipeline_bindings) == (
        "Hoechst",
        "MAP2",
        "SMI312",
    )
    assert tuple(binding.alias for binding in execution_bindings.bindings) == (
        "SMI312",
        "Hoechst",
    )
    assert source_binding_plan.bindings == execution_bindings.bindings


def test_path_planner_execution_groups_use_resolved_source_bindings():
    step = FunctionStep(func=_identity, name="source-bound")
    bindings = (
        NamedSourceBinding(
            alias="OrigStain1",
            component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
        ),
        NamedSourceBinding(
            alias="OrigStain2",
            component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
        ),
    )
    snapshot = _snapshot(
        step,
        0,
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=bindings,
        ),
    )
    planner = PathPlanner.__new__(PathPlanner)
    planner.session = SimpleNamespace(realized_source_metadata=None)

    scope = PathPlannerExecutionGroups(planner).source_binding_scope_for_group_by(
        snapshot,
        GroupBy.CHANNEL,
        source_bindings=snapshot.step.source_bindings,
    )

    assert scope.keys == ("1", "2")
    assert scope.component is AllComponents.CHANNEL


def test_path_planner_freezes_only_contract_selected_source_bindings():
    binding = NamedSourceBinding(alias="DNA")
    unused_binding = NamedSourceBinding(alias="Unused")
    step = FunctionStep(func=_external_source_consumer, name="source-bound")
    snapshot = _snapshot(
        step,
        0,
        source_bindings=StepSourceBindingsConfig(
            bindings=(binding, unused_binding),
            enabled=True,
        ),
    )
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )
    execution_bindings, (source_binding_plan, _source_universe_plan) = (
        _compile_source_plans_for_contract(
            session,
            snapshot,
            _external_source_consumer,
        )
    )

    assert execution_bindings.bindings == (binding,)
    assert source_binding_plan.bindings == (binding,)


def test_compiler_pipeline_scope_prevents_cross_pipeline_source_binding_inheritance(
    tmp_path,
):
    ObjectStateRegistry.clear()
    binding = NamedSourceBinding(alias="DNA")
    plate_path = tmp_path / "plate"
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())

    first_orchestrator = SimpleNamespace(
        plate_path=plate_path,
        pipeline_config=PipelineConfig(
            source_bindings_config=LazySourceBindingsConfig(bindings=(binding,)),
        ),
    )
    second_orchestrator = SimpleNamespace(
        plate_path=plate_path,
        pipeline_config=PipelineConfig(),
    )
    first_pipeline = [FunctionStep(func=_identity, name="first")]
    second_pipeline = [FunctionStep(func=_identity, name="second")]
    registered_scopes: list[tuple[SimpleNamespace, str]] = []

    try:
        first_scope, _first_config_state, first_resolved = (
            PipelineCompiler._register_and_resolve_pipeline_once(
                first_orchestrator,
                first_pipeline,
                is_zmq_execution=False,
            )
        )
        registered_scopes.append((first_orchestrator, first_scope))

        second_scope, _second_config_state, second_resolved = (
            PipelineCompiler._register_and_resolve_pipeline_once(
                second_orchestrator,
                second_pipeline,
                is_zmq_execution=False,
            )
        )
        registered_scopes.append((second_orchestrator, second_scope))
    finally:
        for orchestrator, scope_id in registered_scopes:
            PipelineCompiler._cleanup_compilation_object_states(
                orchestrator,
                scope_id,
            )

    assert first_scope != second_scope
    assert first_resolved.snapshots[0].step.source_bindings.bindings == (binding,)
    assert second_resolved.snapshots[0].step.source_bindings.is_empty
