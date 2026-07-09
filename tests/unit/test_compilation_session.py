from types import SimpleNamespace

import pytest

from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.config_framework.object_state import ObjectState
from openhcs.config_framework.object_state_registry import ObjectStateRegistry
from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import ArtifactSpec, ImageArtifactType
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyNapariStreamingConfig,
    LazyStepSourceBindingsConfig,
    PipelineConfig,
    ProcessingConfig,
    StepMaterializationConfig,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import compile_function_pattern
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
from openhcs.core.function_step_invocation_contracts import (
    EMPTY_FUNCTION_STEP_INVOCATION_CONTRACTS,
)
from openhcs.core.pipeline import Pipeline
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.compiler import AxisCompilationRequest, PipelineCompiler
from openhcs.core.pipeline.path_planner import PathPlannerExecutionGroups
from openhcs.core.pipeline.step_config_universe import (
    StepConfigRoot,
    StepConfigUniverse,
    step_config_declarations,
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
    SourceBindingsConfig,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep


def _identity(image):
    return image


@module_artifact_contract(
    ModuleArtifactContract(
        module_name="ExternalSourceConsumer",
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition,
                (ArtifactSpec.input("DNA", ImageArtifactType),),
            ),
        ),
    )
)
def _external_source_consumer(image):
    return image


def _config_universe(*configs) -> StepConfigUniverse:
    roots = []
    declarations = step_config_declarations()
    for config in configs:
        declaration = next(
            declaration
            for declaration in declarations
            if type(config) is declaration.config_type
        )
        roots.append(StepConfigRoot(declaration=declaration, value=config))
    return StepConfigUniverse(tuple(roots))


def _snapshot(
    index: int,
    name: str = "step",
    variable_components=(VariableComponents.SITE,),
    source_bindings=EMPTY_SOURCE_BINDINGS,
    input_source: InputSource = InputSource.PREVIOUS_STEP,
) -> StepSnapshot:
    return StepSnapshot(
        index=index,
        scope_id=f"plate::functionstep_{index}",
        name=name,
        step_type="FunctionStep",
        enabled=True,
        is_function_step=True,
        func=_identity,
        invocation_contracts=EMPTY_FUNCTION_STEP_INVOCATION_CONTRACTS,
        configs=_config_universe(
            source_bindings,
            ProcessingConfig(
                variable_components=list(variable_components),
                group_by=None,
                input_source=input_source,
            ),
            StepMaterializationConfig(enabled=False),
        ),
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
        global_step_axis_filters={},
        enable_visualizer_override=False,
        is_zmq_execution=True,
    )

    context = request.context_for("A01")

    assert context.auto_add_output_plate_to_plate_manager is True


def test_compilation_session_owns_step_snapshot_plan_invariants():
    step = FunctionStep(func=_identity, name="step")
    step_state = object()
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: step_state},
        snapshots=(_snapshot(0),),
    )

    assert session.axis_id == "A01"
    assert session.step(0) is step
    assert session.step_state(0) is step_state
    assert session.snapshot(0).name == "step"
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
            snapshots=(_snapshot(1),),
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
                0,
                variable_components=(VariableComponents.CHANNEL,),
            ),
        ),
    )

    PipelineCompiler._supplement_step_plans(session)

    assert session.plan(0).variable_components == [VariableComponents.CHANNEL]


def test_compiler_enabled_source_binding_plan_comes_from_objectstate_snapshot():
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
            source_bindings_config=SourceBindingsConfig(
                metadata_rules=(metadata_rule,),
                match_plan=match_plan,
            ),
        ),
        scope_id="plate",
    )

    try:
        ObjectStateRegistry.register(pipeline_state, _skip_snapshot=True)
        step = FunctionStep(
            func=_identity,
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
        resolved_step = step_state.to_object()
        snapshot = StepSnapshot.from_resolved_step(
            index=0,
            step=resolved_step,
            step_state=step_state,
        )
        session = CompilationSession.from_context(
            context=_context(),
            steps=[resolved_step],
            orchestrator=_orchestrator(pipeline_state.to_object()),
            global_config=GlobalPipelineConfig(),
            step_state_map={0: step_state},
            snapshots=(snapshot,),
        )

        PipelineCompiler._supplement_step_plans(session)
    finally:
        ObjectStateRegistry.clear()

    assert snapshot.source_bindings.bindings == (binding,)
    assert snapshot.source_bindings.metadata_rules == (metadata_rule,)
    assert snapshot.source_bindings.match_plan == match_plan
    assert session.plan(0).source_binding_plan.bindings == (binding,)
    assert session.plan(0).source_binding_plan.metadata_rules == (metadata_rule,)
    assert session.plan(0).source_binding_plan.match_plan == match_plan


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
        resolved_step = step_state.to_object()
        snapshot = StepSnapshot.from_resolved_step(
            index=0,
            step=resolved_step,
            step_state=step_state,
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


def test_compiler_activates_pipeline_source_binding_defaults_for_pipeline_start():
    binding = NamedSourceBinding(alias="DNA")
    step = FunctionStep(func=_identity, name="source-bound")
    snapshot = _snapshot(
        0,
        input_source=InputSource.PIPELINE_START,
    )
    session = CompilationSession.from_context(
        context=_context(),
        steps=[step],
        orchestrator=_orchestrator(
            PipelineConfig(
                source_bindings_config=SourceBindingsConfig(bindings=(binding,)),
                step_source_bindings_config=LazyStepSourceBindingsConfig(
                    enabled=None,
                    metadata_rules=None,
                    match_plan=None,
                    source_filters=None,
                    bindings=None,
                ),
            ),
        ),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )

    PipelineCompiler._supplement_step_plans(session)

    assert session.plan(0).source_binding_plan.bindings == (binding,)


def test_path_planner_execution_groups_use_effective_source_binding_defaults():
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
    snapshot = SimpleNamespace(
        source_bindings=LazyStepSourceBindingsConfig(enabled=None),
    )
    planner = SimpleNamespace(
        source_bindings_for_snapshot=lambda _snapshot: StepSourceBindingsConfig(
            enabled=True,
            bindings=bindings,
        )
    )

    scope = PathPlannerExecutionGroups(planner).source_binding_scope_for_group_by(
        snapshot,
        GroupBy.CHANNEL,
    )

    assert scope.keys == ("1", "2")
    assert scope.component is AllComponents.CHANNEL


def test_compiler_freezes_enabled_source_binding_set_without_contract_filtering():
    binding = NamedSourceBinding(alias="DNA")
    unused_binding = NamedSourceBinding(alias="Unused")
    step = FunctionStep(func=_external_source_consumer, name="source-bound")
    snapshot = _snapshot(
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
    session.plan(0).compiled_function_pattern = compile_function_pattern(
        _external_source_consumer,
        {},
        {},
    )

    PipelineCompiler._supplement_step_plans(session)

    assert session.plan(0).source_binding_plan.bindings == (binding, unused_binding)


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
            source_bindings_config=SourceBindingsConfig(bindings=(binding,)),
        ),
    )
    second_orchestrator = SimpleNamespace(
        plate_path=plate_path,
        pipeline_config=PipelineConfig(),
    )
    first_pipeline = Pipeline(
        [FunctionStep(func=_identity, name="first")],
        name="first_pipeline",
    )
    second_pipeline = Pipeline(
        [FunctionStep(func=_identity, name="second")],
        name="second_pipeline",
    )
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
    assert first_resolved.snapshots[0].source_bindings.bindings == (binding,)
    assert second_resolved.snapshots[0].source_bindings.is_empty
