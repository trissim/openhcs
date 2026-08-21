import ast
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.constants.constants import AllComponents, MemoryType, VariableComponents
from openhcs.core.compiled_step_plan import (
    CompiledStepPlan,
    FrameworkDeviceAssignment,
    InputConversionPlan,
    MaterializedOutputPlan,
)
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.artifacts import (
    ArtifactInputProjectionPlan,
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ImageArtifactType,
    InputStackBroadcastSourceRelation,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.function_patterns import (
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
    compile_function_pattern,
)
from openhcs.core.pipeline.function_contracts import artifact_inputs, artifact_outputs
from openhcs.core.runtime_adapters import RuntimeAdapterRequest, runtime_adapter
from openhcs.core.runtime_output_matching import RuntimeReturnedOutputMatcher
from openhcs.core.step_dependencies import (
    StepInputDependency,
    StepInputDependencyKind,
)
from openhcs.core.steps.function_artifact_materialization import (
    AnalysisOutputDescriptorAuthority,
)
from openhcs.core.steps.function_runtime import ComponentArtifactPlans


def noop(image):
    return image


class ContextStub:
    def __init__(self, compiled_plan):
        self.step_plans = {2: compiled_plan}
        self.filemanager = object()
        self.microscope_handler = SimpleNamespace(
            parser=SimpleNamespace(parse_filename=lambda _filename: None),
            microscope_type="test",
        )


def _compiled_plan(**overrides):
    plan = CompiledStepPlan(
        step_index=2,
        step_scope_id="plate::functionstep_2",
        step_name="measure",
        step_type="FunctionStep",
        axis_id="A01",
        input_dir=Path("/tmp/input"),
        output_dir=Path("/tmp/output"),
        variable_components=(VariableComponents.SITE,),
        group_by=None,
        func=noop,
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="plate::functionstep_1",
        ),
        artifact_inputs={},
        artifact_outputs={},
        read_backend="memory",
        write_backend="memory",
        input_memory_type="numpy",
        output_memory_type="numpy",
        zarr_config=None,
        pipeline_position=9,
        output_plate_root="/tmp/plate_processed",
        sub_dir="images",
        analysis_results_dir="/tmp/output_results",
        input_conversion=InputConversionPlan(
            output_dir=Path("/tmp/converted"),
            backend="zarr",
            uses_virtual_workspace=False,
            original_subdir="input",
        ),
        materialized_output=MaterializedOutputPlan(
            output_dir=Path("/tmp/materialized"),
            backend="disk",
            plate_root="/tmp/plate_materialized",
            sub_dir="images",
            analysis_results_dir="/tmp/materialized_results",
        ),
        compiled_function_pattern=compile_function_pattern(noop, {}, {}),
    )
    for key, value in overrides.items():
        setattr(plan, key, value)
    return plan


def test_compiled_step_plan_is_the_runtime_plan_owner():
    from openhcs.core.steps.function_execution import FunctionStepExecutor

    compiled_plan = _compiled_plan(
        execution_group_scope=ComponentGroupScope.from_raw(
            ("2",),
            component=VariableComponents.SITE,
        )
    )
    plan = compiled_plan.require_function_execution_ready()
    executor = FunctionStepExecutor(ContextStub(compiled_plan), 2)

    assert plan is compiled_plan
    assert executor.plan is compiled_plan
    assert "get_paths_for_axis" not in {
        field.name for field in fields(CompiledStepPlan)
    }
    assert plan.step_scope_id == "plate::functionstep_2"
    assert plan.execution_group_scope.keys == ("2",)
    assert plan.variable_components == (VariableComponents.SITE,)
    assert plan.main_input_dependency.kind is StepInputDependencyKind.STEP_OUTPUT
    assert plan.main_input_dependency.source_step_scope_id == "plate::functionstep_1"
    assert plan.source_binding_plan.is_empty
    assert plan.device_id_for(plan.input_memory_type) is None
    assert not plan.requires_gpu
    assert plan.gpu_memory_types == frozenset()
    assert plan.input_conversion is not None
    assert plan.input_conversion.output_dir == Path("/tmp/converted")
    assert plan.input_conversion.original_subdir == "input"
    assert plan.materialized_output is not None
    assert plan.materialized_output.output_dir == Path("/tmp/materialized")
    assert plan.artifact_analysis_output_dir == Path("/tmp/materialized_results")
    assert plan.streaming_configs == {}


def test_compiled_step_plan_owns_gpu_memory_classification() -> None:
    compiled_plan = _compiled_plan(
        input_memory_type="numpy",
        output_memory_type="cupy",
        device_assignment=FrameworkDeviceAssignment.from_mapping({MemoryType.CUPY: 2}),
    )

    assert compiled_plan.requires_gpu
    assert compiled_plan.gpu_memory_types == frozenset({MemoryType.CUPY})
    assert compiled_plan.device_id_for("cupy") == 2


def test_compiled_step_plan_includes_invocation_execution_memory() -> None:
    def numpy_boundary_with_torch_execution(image):
        return image

    numpy_boundary_with_torch_execution.input_memory_type = "numpy"
    numpy_boundary_with_torch_execution.output_memory_type = "numpy"
    numpy_boundary_with_torch_execution.execution_memory_type = "torch"
    compiled_plan = _compiled_plan(
        compiled_function_pattern=compile_function_pattern(
            numpy_boundary_with_torch_execution,
            {},
            {},
        ),
        device_assignment=FrameworkDeviceAssignment.from_mapping({MemoryType.TORCH: 4}),
    )

    assert compiled_plan.gpu_memory_types == frozenset({MemoryType.TORCH})
    assert compiled_plan.device_id_for("torch") == 4


def test_compiled_step_plan_includes_intermediate_invocation_memory() -> None:
    def to_cupy(image):
        return image

    def from_cupy(image):
        return image

    to_cupy.input_memory_type = "numpy"
    to_cupy.output_memory_type = "cupy"
    to_cupy.execution_memory_type = "numpy"
    from_cupy.input_memory_type = "cupy"
    from_cupy.output_memory_type = "numpy"
    from_cupy.execution_memory_type = "numpy"
    compiled_plan = _compiled_plan(
        input_memory_type="numpy",
        output_memory_type="numpy",
        compiled_function_pattern=compile_function_pattern(
            [to_cupy, from_cupy],
            {},
            {},
        ),
    )

    assert compiled_plan.gpu_memory_types == frozenset({MemoryType.CUPY})


def test_compiled_step_plan_rejects_missing_variable_components():
    compiled_plan = _compiled_plan(variable_components=None)

    with pytest.raises(ValueError, match="missing compiled variable_components"):
        compiled_plan.require_function_execution_ready()


def test_compiled_step_plan_rejects_missing_pipeline_position_without_fallback():
    compiled_plan = _compiled_plan(pipeline_position=None)

    with pytest.raises(ValueError, match="has no pipeline_position"):
        compiled_plan.require_function_execution_ready()


def test_compiled_step_plan_rejects_missing_compiled_function_pattern():
    compiled_plan = _compiled_plan(compiled_function_pattern=None)

    with pytest.raises(ValueError, match="has no compiled_function_pattern"):
        compiled_plan.require_function_execution_ready()


def test_function_step_executor_rejects_non_nominal_plan():
    from openhcs.core.steps.function_execution import FunctionStepExecutor

    context = ContextStub(SimpleNamespace(step_name="structural-copy"))

    with pytest.raises(TypeError, match="requires CompiledStepPlan"):
        FunctionStepExecutor(context, 2)


def test_function_step_execution_plan_mirror_is_deleted():
    repository_root = Path(__file__).resolve().parents[2]
    mirror_path = repository_root / "openhcs/core/steps/function_plan.py"
    assert not mirror_path.exists()

    violations = []
    for source_path in (repository_root / "openhcs").rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "FunctionStepExecutionPlan":
                violations.append(f"{source_path}:{node.lineno}:name")
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "FunctionStepExecutionPlan"
            ):
                violations.append(f"{source_path}:{node.lineno}:attribute")
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "FunctionStepExecutionPlan"
            ):
                violations.append(f"{source_path}:{node.lineno}:class")
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "openhcs.core.steps.function_plan"
            ):
                violations.append(f"{source_path}:{node.lineno}:import")
    assert violations == []


def test_function_step_execution_does_not_prepare_callables_in_hot_path(monkeypatch):
    from openhcs.core.steps import function_execution

    events = []

    class ManifestStub:
        def producer_records_for(self, plan):
            del plan
            return None

        def begin_step(self, plan, input_records=()):
            assert input_records == ()
            events.append(("begin", plan.step_name))

    class ExecutionStub(function_execution.FunctionStepExecutor):
        def __init__(self):
            self.context = SimpleNamespace()
            self.plan = SimpleNamespace(
                step_index=3,
                step_name="prepared-at-compile",
                axis_id="A01",
            )

        def _log_execution_start(self):
            events.append(("start", self.plan.step_name))

        def _detect_patterns(self):
            return {"A01": ["image.tif"]}

        def _log_discovered_patterns(self, patterns_by_axis):
            events.append(("patterns", tuple(patterns_by_axis)))

        def _convert_input_if_needed(self):
            events.append(("convert", self.plan.step_name))

        def _require_patterns(self, patterns_by_axis):
            events.append(("require", tuple(patterns_by_axis)))

        def _apply_sequential_filter(self, patterns_by_axis):
            events.append(("filter", tuple(patterns_by_axis)))

        def _prepare_groups(self, patterns_by_axis):
            events.append(("groups", tuple(patterns_by_axis)))
            return function_execution.PatternGroups.from_prepared(
                {"default": ["image.tif"]}
            )

        def _preload_inputs_if_needed(self, grouped_patterns):
            events.append(("preload", tuple(grouped_patterns.groups)))

        def _prepare_callables(self, grouped_patterns):
            raise AssertionError("callable warmup belongs to compilation")

        def _execute_pattern_groups(self, grouped_patterns, total_groups):
            events.append(("execute", total_groups))

    monkeypatch.setattr(
        function_execution,
        "step_output_manifest",
        lambda _context: ManifestStub(),
    )
    monkeypatch.setattr(
        function_execution,
        "finalize_function_step_outputs",
        lambda _context, plan: events.append(("finalize", plan.step_name)),
    )

    ExecutionStub().run()

    assert ("execute", 1) in events
    assert ("finalize", "prepared-at-compile") in events


def test_grouped_pattern_requires_concrete_execution_group_component():
    from openhcs.core.steps.function_execution import FunctionStepExecutor

    def first(image):
        return image

    def second(image):
        return image

    executor = object.__new__(FunctionStepExecutor)
    executor.plan = SimpleNamespace(
        axis_id="A01",
        step_name="dict-none",
        execution_group_value=None,
        execution_group_scope=ComponentGroupScope.ungrouped(),
        compiled_function_pattern=compile_function_pattern(
            {"1": first, "2": second},
            {},
            {},
        ),
    )

    with pytest.raises(
        ValueError,
        match="dict function pattern without a concrete execution group component",
    ):
        executor._prepare_groups(
            {
                "A01": (
                    "A01_s{iii}_w1_z001_t001.tif",
                    "A01_s{iii}_w2_z001_t001.tif",
                )
            }
        )


def test_build_analysis_filename_falls_back_to_axis_and_pipeline_position_without_record():
    context = ContextStub(_compiled_plan(pipeline_position=7))
    plan = context.step_plans[2].require_function_execution_ready()

    assert (
        AnalysisOutputDescriptorAuthority.build(
            "measurements",
            plan,
            context=context,
        ).filename
        == "A01_measurements_step7.roi.zip"
    )


def test_component_artifact_plan_selection_merges_global_and_group_outputs():
    global_output = ArtifactOutputPlan(
        name="objects",
        path="/tmp/objects",
        artifact_type=ObjectLabelsArtifactType,
    )
    grouped_output = ArtifactOutputPlan(
        name="measurements",
        path="/tmp/measurements/A01",
        artifact_type=MeasurementsArtifactType,
        group_keys=("A01",),
        group_component=AllComponents.WELL,
        paths_by_group={"A01": "/tmp/measurements/A01"},
    )

    selected = ComponentArtifactPlans._select_output_plans_for_component(
        {
            global_output.ref(): global_output,
            grouped_output.ref(): grouped_output,
        },
        ComponentGroupScope(("A01",), component=AllComponents.WELL),
        "A01",
    )

    assert selected == {
        global_output.ref(): global_output,
        grouped_output.ref(): grouped_output,
    }


def test_component_artifact_plan_selection_omits_unscoped_outputs_for_missing_group():
    output = ArtifactOutputPlan(
        name="objects",
        path="/tmp/objects",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": "/tmp/objects_w3"},
    )

    selected = ComponentArtifactPlans._select_output_plans_for_component(
        {output.ref(): output},
        ComponentGroupScope(("1", "3"), component=AllComponents.CHANNEL),
        "1",
    )

    assert selected == {}


def test_default_invocation_keeps_compiled_grouped_output_plan():
    grouped_output = ArtifactOutputPlan(
        name="measurements",
        path="/tmp/measurements",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1", "3", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "1": "/tmp/w1_measurements",
            "3": "/tmp/w3_measurements",
            "2": "/tmp/w2_measurements",
        },
    )

    selected = ComponentArtifactPlans._select_output_plans_for_component(
        {grouped_output.ref(): grouped_output},
        ComponentGroupScope.ungrouped(),
        None,
    )

    assert selected == {grouped_output.ref(): grouped_output}


def _cross_channel_output_invocation():
    channel_one_spec = ArtifactSpec.output("channel_one", ImageArtifactType)
    channel_two_spec = ArtifactSpec.output("channel_two", ImageArtifactType)

    @artifact_outputs(channel_one_spec, channel_two_spec)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_outputs=True,
    )
    def publish_channels(image, *, runtime):
        del runtime
        return image

    channel_one = ArtifactOutputPlan(
        name=channel_one_spec.name,
        path="/memory/channel_one.pkl",
        artifact_type=channel_one_spec.artifact_type,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/channel_one__1.pkl"},
    )
    channel_two = ArtifactOutputPlan(
        name=channel_two_spec.name,
        path="/memory/channel_two.pkl",
        artifact_type=channel_two_spec.artifact_type,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/channel_two__2.pkl"},
    )
    invocation = compile_function_pattern(
        publish_channels,
        {},
        {
            channel_one.ref(): channel_one,
            channel_two.ref(): channel_two,
        },
    ).default_group.invocations[0]
    return invocation, channel_one, channel_two


def test_invocation_output_selection_omits_inactive_component_outputs():
    invocation, channel_one, _channel_two = _cross_channel_output_invocation()
    channel_one = channel_one.for_group("1")

    selected = ComponentArtifactPlans(
        inputs={},
        outputs={channel_one.ref(): channel_one},
    ).select_for_invocation(
        invocation,
        execution_scope=ComponentGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
        component_key="1",
    )

    assert selected.outputs == {channel_one.ref(): channel_one}


def test_compiler_handoff_preserves_exact_same_name_output_types():
    image_spec = ArtifactSpec.output("shared", ImageArtifactType)
    labels_spec = ArtifactSpec.output("shared", ObjectLabelsArtifactType)

    @artifact_outputs(image_spec, labels_spec)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_outputs=True,
    )
    def publish_shared_outputs(image, *, runtime):
        del runtime
        return image

    image_plan = ArtifactOutputPlan(
        name=image_spec.name,
        path="/memory/shared_image.pkl",
        artifact_type=image_spec.artifact_type,
    )
    labels_plan = ArtifactOutputPlan(
        name=labels_spec.name,
        path="/memory/shared_labels.pkl",
        artifact_type=labels_spec.artifact_type,
    )
    available = {image_plan.ref(): image_plan, labels_plan.ref(): labels_plan}
    invocation = compile_function_pattern(
        publish_shared_outputs,
        {},
        available,
    ).default_group.invocations[0]

    selected = invocation.select_outputs(available)
    request = RuntimeAdapterRequest(
        context=object(),
        callable_contract=invocation.contract,
        artifact_outputs=selected,
        axis_scope=RuntimeExecutionAxisScope("A01"),
    )
    image_value = object()
    labels_value = object()
    _returned, matched = RuntimeReturnedOutputMatcher(
        callable_contract=invocation.contract,
        returned_output=(image_value, labels_value),
    ).resolve_plan_values(tuple(selected.values()))

    assert selected == {
        image_spec.ref(): image_plan,
        labels_spec.ref(): labels_plan,
    }
    assert request.require_artifact_output_plan(image_spec.ref()) == image_plan
    assert request.require_artifact_output_plan(labels_spec.ref()) == labels_plan
    with pytest.raises(ValueError, match="output key.*conflicts with plan ref"):
        RuntimeAdapterRequest(
            context=object(),
            callable_contract=invocation.contract,
            artifact_outputs={image_spec.ref(): labels_plan},
            axis_scope=RuntimeExecutionAxisScope("A01"),
        )
    assert matched == (
        (image_plan, image_spec, image_value),
        (labels_plan, labels_spec, labels_value),
    )


def test_invocation_component_selection_projects_relation_owned_inputs():
    channel_one_input = ArtifactSpec.input("source_one", ImageArtifactType)
    channel_two_input = ArtifactSpec.input("source_two", ImageArtifactType)
    channel_one_illumination = ArtifactSpec.input(
        "illumination_one",
        ImageArtifactType,
        parameter_name="illumination_function",
        relations=(InputStackBroadcastSourceRelation(source=channel_one_input.ref()),),
    )
    channel_two_illumination = ArtifactSpec.input(
        "illumination_two",
        ImageArtifactType,
        parameter_name="illumination_function",
        relations=(InputStackBroadcastSourceRelation(source=channel_two_input.ref()),),
    )
    shared_object_input = ArtifactSpec.input("Objects", ObjectLabelsArtifactType)
    channel_one_output = ArtifactSpec.output(
        "derived_one",
        ImageArtifactType,
        relations=(SourceStackLineageSourceRelation(source=channel_one_input.ref()),),
    )
    channel_two_output = ArtifactSpec.output(
        "derived_two",
        ImageArtifactType,
        relations=(SourceStackLineageSourceRelation(source=channel_two_input.ref()),),
    )

    @artifact_inputs(
        channel_one_input,
        channel_one_illumination,
        channel_two_input,
        channel_two_illumination,
        shared_object_input,
    )
    @artifact_outputs(channel_one_output, channel_two_output)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
        manages_artifact_outputs=True,
    )
    def publish_derived_channels(image, *, runtime):
        del runtime
        return image

    input_plans = {
        spec.ref(): ArtifactInputPlan(
            name=spec.name,
            path=f"/memory/{spec.name}.pkl",
            artifact_type=spec.artifact_type,
        )
        for spec in (
            channel_one_input,
            channel_one_illumination,
            channel_two_input,
            channel_two_illumination,
            shared_object_input,
        )
    }
    output_plans = {
        spec.ref(): ArtifactOutputPlan(
            name=spec.name,
            path=f"/memory/{spec.name}.pkl",
            artifact_type=spec.artifact_type,
            group_keys=(channel,),
            group_component=AllComponents.CHANNEL,
            paths_by_group={channel: f"/memory/{spec.name}__{channel}.pkl"},
            relations=spec.relations,
        )
        for spec, channel in (
            (channel_one_output, "1"),
            (channel_two_output, "2"),
        )
    }
    invocation = compile_function_pattern(
        publish_derived_channels,
        input_plans,
        output_plans,
    ).default_group.invocations[0]
    execution_scope = ComponentGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )
    invocation = invocation.with_artifact_input_edges(
        tuple(
            InvocationArtifactInputEdgePlan(
                key=InvocationArtifactInputProjectionKey(
                    invocation_key=invocation.key,
                    input_index=index,
                ),
                spec=spec,
                storage_plan=input_plans[spec.ref()],
                projection=ArtifactInputProjectionPlan(
                    invocation_scope=execution_scope,
                    producer_selection_scope=ComponentGroupScope.ungrouped(),
                    component_scopes=(execution_scope,),
                    consumer_variable_components=(),
                ),
            )
            for index, spec in enumerate(
                (
                    channel_one_input,
                    channel_one_illumination,
                    channel_two_input,
                    channel_two_illumination,
                    shared_object_input,
                )
            )
        )
    )
    active_output = output_plans[channel_one_output.ref()].for_group("1")

    selected = ComponentArtifactPlans(
        inputs=input_plans,
        outputs={active_output.ref(): active_output},
    ).select_for_invocation(
        invocation,
        execution_scope=execution_scope,
        component_key="1",
    )

    assert tuple(edge.spec for edge in selected.inputs.values()) == (
        channel_one_input,
        channel_one_illumination,
        shared_object_input,
    )
    assert selected.outputs == {active_output.ref(): active_output}

    second_active_output = output_plans[channel_two_output.ref()].for_group("2")
    second_selected = ComponentArtifactPlans(
        inputs=input_plans,
        outputs={second_active_output.ref(): second_active_output},
    ).select_for_invocation(
        invocation,
        execution_scope=execution_scope,
        component_key="2",
    )

    assert tuple(edge.spec for edge in second_selected.inputs.values()) == (
        channel_two_input,
        channel_two_illumination,
        shared_object_input,
    )
    assert second_selected.outputs == {second_active_output.ref(): second_active_output}


def test_invocation_output_selection_rejects_missing_active_component_output():
    invocation, _channel_one, _channel_two = _cross_channel_output_invocation()

    with pytest.raises(ValueError, match="channel_one.*unavailable"):
        ComponentArtifactPlans(
            inputs={},
            outputs={},
        ).select_for_invocation(
            invocation,
            execution_scope=ComponentGroupScope.from_raw(
                ("1", "2"),
                component=AllComponents.CHANNEL,
            ),
            component_key="1",
        )


def test_invocation_output_selection_rejects_active_projection_drift():
    invocation, channel_one, _channel_two = _cross_channel_output_invocation()
    drifted_channel_one = ArtifactOutputPlan(
        name=channel_one.name,
        path="/memory/drifted_channel_one.pkl",
        artifact_type=channel_one.artifact_type,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/drifted_channel_one__1.pkl"},
    )

    with pytest.raises(ValueError, match="runtime projection.*compiled owner"):
        ComponentArtifactPlans(
            inputs={},
            outputs={drifted_channel_one.ref(): drifted_channel_one},
        ).select_for_invocation(
            invocation,
            execution_scope=ComponentGroupScope.from_raw(
                ("1", "2"),
                component=AllComponents.CHANNEL,
            ),
            component_key="1",
        )


def test_adapter_invocation_preserves_component_selected_artifact_inputs():
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    @artifact_inputs(
        ArtifactSpec.input("IllumStain1", ImageArtifactType),
    )
    def apply_illumination(image, *, runtime):
        del runtime
        return image

    first_input = ArtifactInputPlan(
        name="IllumStain1",
        path="/tmp/IllumStain1",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
    )
    invocation = compile_function_pattern(
        apply_illumination,
        {first_input.ref(): first_input},
        {},
    ).default_group.invocations[0]
    scope = ComponentGroupScope.from_raw(
        ("1",),
        component=AllComponents.CHANNEL,
    )
    edge = InvocationArtifactInputEdgePlan(
        key=InvocationArtifactInputProjectionKey(
            invocation_key=invocation.key,
            input_index=0,
        ),
        spec=ArtifactSpec.input(
            first_input.name,
            first_input.artifact_type,
            sidecar_role=first_input.sidecar_role,
        ),
        storage_plan=first_input,
        projection=ArtifactInputProjectionPlan(
            invocation_scope=scope,
            producer_selection_scope=scope,
            component_scopes=(scope,),
            consumer_variable_components=(AllComponents.SITE,),
        ),
    )
    invocation = invocation.with_artifact_input_edges((edge,))

    selected = ComponentArtifactPlans(
        inputs={first_input.ref(): first_input},
        outputs={},
    ).select_for_invocation(
        invocation,
        execution_scope=scope,
        component_key="1",
    )

    assert selected.inputs == {edge.key: edge}
